"""Opt-in retained-texture picture presenter (VP1 of the viewer presentation work).

The default viewer (`Viewer`, a QGraphicsView fed by `Window._show_image`) is untouched and stays the
fallback. This module adds a second, explicitly experimental surface: a `QRenderWidget` with
`present_method="screen"` that keeps the newest CPU-produced picture in one wgpu texture and pans and
zooms it on the GPU, so a view change never runs `to_qimage` or rebuilds a pixmap.

Contracts (the tests in tests/test_directviewer.py pin each one):

* Ownership. `FrameLedger` is plain Python and owns the rules; `RetainedPresenter` owns the wgpu
  resources and is the only code that creates or destroys them. A picture is uploaded once, inside
  the draw callback, and only the newest ready picture is ever uploaded (an older pending picture is
  dropped, never converted). A same-size picture is written into the existing texture (queue writes
  are ordered before later submits); a different size makes a new texture and the old one is
  destroyed after the draw that replaced it is submitted (WebGPU frees it once the queue has finished
  the submissions that used it); closing waits for the queue to go idle.
* Revision and epoch. Every offer carries the source generation and the epoch it was made in. An
  offer from an older epoch (a device that has since been replaced) or an older generation is
  rejected; submission order inside a generation is last-writer-wins.
* Surface textures. `context.get_current_texture()` is called only inside one draw and the view made
  from it is dropped before the draw returns. Nothing from the surface survives a present or a resize.
* Pacing. Input handlers only change view state and call `request_draw()`, which rendercanvas
  schedules from Qt's own paint/timer callbacks. Nothing spins or waits for a surface.
* Device. The presenter owns one device, requested from the adapter that the surface reports it can
  present to (`pick_display_adapter`); the texture, pipelines and surface configuration all come from
  that one device. Sharing the renderer's device or a renderer texture is the later L4 step.
* Device loss. `LossWatch` hears wgpu's "device lost" log line and a failed draw counts as a loss;
  `RetainedPresenter.recover()` starts a new epoch, releases the old resources and device, builds a new
  device from the same adapter and queues the retained CPU picture again. Too many recoveries in a row end
  in the dock hiding itself, like any other failure.

Known gaps are listed in docs/DIRECT_VIEWER.md and shown in the dock's title.
"""
from __future__ import annotations

import collections
import logging
import os
import sys
import threading
import time

import numpy as np

ENV = "NODEBASED_DIRECT_VIEWER"
ADAPTER_ENV = "NODEBASED_DIRECT_ADAPTER"
BACKEND_ENV = "NODEBASED_DIRECT_BACKEND"
UNIFORM_BYTES = 160           # the 80-byte view block, then carddisplay's display vec4s and its three compare vec4s
ZOOM_RANGE = (0.05, 20.0)     # logical zoom, the same bounds PanZoomView.wheelEvent applies
SURROUND = (0.0, 0.0, 0.0)    # VIEWER_SURROUND is pure black in every theme
RECOVERY_LIMIT = (3, 30.0)    # at most this many device recoveries in this many seconds, then the dock gives up
STAGING_SLOTS = 3             # reused upload buffers: one being filled, one waiting for a draw, one the card copies from
ROW_ALIGN = 256               # WebGPU's bytes_per_row alignment for a buffer-to-texture copy
HALF_ABOVE = 4_000_000        # float pictures with more pixels than this go to the card as half floats
FILL_BANDS = 8                # a large picture is copied into the staging buffer in this many bands at once
# What an overlay provider returns instead of an image: nothing is painted / the same image as last time.
EMPTY_OVERLAY = object()
SAME_OVERLAY = object()
BADGE_CACHE = 12              # tier badge labels kept on the card
PAINT_BAND_GAP = 48           # changed overlay rows closer together than this go up as one band


class TooLarge(ValueError):
    """A picture this card cannot hold as one texture; the Viewer keeps showing it."""


def too_large(limits, width, height, texel, device=""):
    """Why a width x height picture (texel bytes per pixel) does not fit this card, or "" when it does.

    Safe to call from any thread: `limits` is the presenter's plain dict of the device's limits."""
    if not limits:
        return ""
    card = device or "this card"
    side = limits.get("max-texture-dimension-2d", 8192)
    if width > side or height > side:
        return (f"Direct picture: the picture is {width} x {height} pixels and {card} takes at most {side} "
                f"pixels on a side; the Viewer shows it")
    size = padded_row(width, texel) * height
    if size > limits.get("max-buffer-size", 1 << 28):
        return (f"Direct picture: the picture is {width} x {height} pixels ({size / 1048576:.0f} MB to upload) and "
                f"{card} takes at most {limits['max-buffer-size'] / 1048576:.0f} MB at once; the Viewer shows it")
    return ""


def pair_too_large(limits, width, height, texel, texel_b, device=""):
    """Why the A and B pictures (the same width x height, texel and texel_b bytes per pixel) do not fit this card
    together, or "" when they do. Both go up through one staging buffer, so the buffer limit is what A and B
    share; each is also a texture of its own and must fit the texture side limit."""
    if not limits:
        return ""
    card = device or "this card"
    side = limits.get("max-texture-dimension-2d", 8192)
    if width > side or height > side:
        return (f"Direct picture: the A and B pictures are {width} x {height} pixels and {card} takes at most "
                f"{side} pixels on a side; the Viewer shows them")
    size = padded_row(width, texel) * height + padded_row(width, texel_b) * height + ROW_ALIGN
    if size > limits.get("max-buffer-size", 1 << 28):
        return (f"Direct picture: the A and B pictures are {width} x {height} pixels each ({size / 1048576:.0f} MB to "
                f"upload together) and {card} takes at most {limits['max-buffer-size'] / 1048576:.0f} MB at once; "
                f"the Viewer shows them")
    return ""


class LossWatch(logging.Handler):
    """Notices wgpu's own "device lost" report (it only logs it; the promise API is not implemented here)."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.lost = threading.Event()
        self.message = ""

    def emit(self, record):
        text = record.getMessage()
        if "device was lost" in text.lower() or "device lost" in text.lower():
            self.message = text.splitlines()[0] if text else "device lost"
            self.lost.set()


def enabled():
    return os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


class FrameLedger:
    """Which picture is wanted, which are stale, and which GPU resources may be released."""

    def __init__(self):
        self.epoch = 0
        self.generation = -1
        self.sequence = 0
        self.pending = None
        self.stats = collections.Counter()
        self._retired = []          # (present_index_when_retired, resource)
        self.presents = 0

    def offer(self, frame, generation, epoch):
        """Queue `frame` as the newest ready picture. False (and nothing queued) when it is stale."""
        if epoch != self.epoch:
            self.stats["rejected_epoch"] += 1
            return False
        if generation is not None:
            if generation < self.generation:
                self.stats["rejected_generation"] += 1
                return False
            self.generation = generation
        if self.pending is not None:
            self.stats["dropped_pending"] += 1
        self.sequence += 1
        self.pending = (self.sequence, frame)
        self.stats["accepted"] += 1
        return True

    def take(self):
        """The newest ready picture as (sequence, frame), or None; clears the slot."""
        taken, self.pending = self.pending, None
        return taken

    def retire(self, resource):
        self._retired.append((self.presents, resource))

    def note_present(self):
        """Called once per draw, after its submit; returns every resource retired before that submit.

        A replaced texture is referenced only by command buffers already submitted, so it may be
        destroyed now: WebGPU's `destroy()` contract (implemented by wgpu-core's lifetime tracker)
        frees the memory only once the queue has completed every submission that used it. This
        replaces VP1's "hold for three presents" heuristic, which counted draws, not GPU completion."""
        self.presents += 1
        ready = [r for _, r in self._retired]
        self._retired = []
        return ready

    def new_epoch(self):
        """The device was replaced: forget the pending picture and hand back everything to destroy."""
        self.epoch += 1
        self.pending = None
        self.generation = -1
        everything = [r for _, r in self._retired]
        self._retired = []
        return everything


def fit_view(widget_size, rect, insets=(0, 0), margin=0.0):
    """(centre_x, centre_y, zoom) that shows scene rectangle `rect` = (x, y, w, h) whole in `widget_size`."""
    w, h = widget_size
    usable_h = max(h - insets[0] - insets[1], 1)
    zoom = min(w / max(rect[2], 1e-6), usable_h / max(rect[3], 1e-6)) * (1.0 - margin)
    cx = rect[0] + rect[2] / 2.0
    cy = rect[1] + rect[3] / 2.0 + (insets[1] - insets[0]) / 2.0 / zoom
    return cx, cy, zoom


def zoom_about(view, cursor, size, factor):
    """View after zooming by `factor` keeping the scene point under `cursor` (physical px) fixed."""
    cx, cy, zoom = view
    new_zoom = zoom * factor
    sx = cx + (cursor[0] - size[0] / 2.0) / zoom
    sy = cy + (cursor[1] - size[1] / 2.0) / zoom
    return (sx - (cursor[0] - size[0] / 2.0) / new_zoom, sy - (cursor[1] - size[1] / 2.0) / new_zoom, new_zoom)


PAINT_SHADER = """
// The painted overlay (wipe, region of interest, masks, Roto/Tracker/Transform handles, the pixel readout):
// a premultiplied RGBA8 picture the size of the surface, drawn one texel per surface pixel over the picture.
@group(0) @binding(0) var paint: texture_2d<f32>;
@group(0) @binding(1) var<uniform> srgb: vec4f;
struct Out { @builtin(position) p: vec4f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var p = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
    var o: Out; o.p = vec4f(p[i], 0., 1.); return o;
}
fn encode(c: vec3f) -> vec3f {
    if (srgb.x < 0.5) { return c; }
    let lo = c / 12.92; let hi = pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
    return select(hi, lo, c <= vec3f(0.04045));
}
@fragment fn fs(o: Out) -> @location(0) vec4f {
    let t = textureLoad(paint, vec2i(o.p.xy), 0);
    if (t.a <= 0.0) { discard; }
    return vec4f(encode(t.rgb / t.a) * t.a, t.a);
}
"""

BADGE_SHADER = """
// The tier badge ("proxy 1/2", "ROI"): a small premultiplied RGBA8 texture of the label, made once per text and
// kept, drawn as one quad at a physical pixel rectangle. Uniform: x, y, w, h of the quad, then the surface
// width and height, the sRGB flag, unused.
@group(0) @binding(0) var badge: texture_2d<f32>;
@group(0) @binding(1) var<uniform> q: array<vec4f, 2>;
struct Out { @builtin(position) p: vec4f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var c = array<vec2f, 6>(vec2f(0., 0.), vec2f(1., 0.), vec2f(0., 1.), vec2f(0., 1.), vec2f(1., 0.), vec2f(1., 1.));
    let px = q[0].xy + c[i] * q[0].zw;
    var o: Out; o.p = vec4f(px.x / q[1].x * 2. - 1., 1. - px.y / q[1].y * 2., 0., 1.); return o;
}
fn encode(c: vec3f) -> vec3f {
    if (q[1].z < 0.5) { return c; }
    let lo = c / 12.92; let hi = pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
    return select(hi, lo, c <= vec3f(0.04045));
}
@fragment fn fs(o: Out) -> @location(0) vec4f {
    let t = textureLoad(badge, vec2i(o.p.xy - q[0].xy), 0);
    if (t.a <= 0.0) { discard; }
    return vec4f(encode(t.rgb / t.a) * t.a, t.a);
}
"""

SHADER = """
struct View {
    viewport: vec2f, centre: vec2f,
    image: vec4f,            // scene rectangle the picture covers: x, y, w, h
    format: vec4f,           // scene rectangle of the format (outline overlay)
    tex: vec2f, zoom: f32, srgb: f32,
    surround: vec4f,
};
@group(0) @binding(0) var img: texture_2d<f32>;
@group(0) @binding(1) var<uniform> v: View;
struct Out { @builtin(position) p: vec4f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    var p = array<vec2f, 3>(vec2f(-1., -1.), vec2f(3., -1.), vec2f(-1., 3.));
    var o: Out; o.p = vec4f(p[i], 0., 1.); return o;
}
fn encode(c: vec3f) -> vec3f {
    // The picture is already display-encoded sRGB bytes. An sRGB surface re-encodes on write, so undo it once.
    if (v.srgb < 0.5) { return c; }
    let lo = c / 12.92; let hi = pow((c + vec3f(0.055)) / 1.055, vec3f(2.4));
    return select(hi, lo, c <= vec3f(0.04045));
}
fn scene_at(frag: vec2f) -> vec2f { return v.centre + (frag - v.viewport * 0.5) / v.zoom; }
// A one-pixel outline of the format rectangle plus an orange corner mark that says "experimental" without
// text (Qt widgets cannot be drawn over a screen-presented surface). It is folded into the picture pass:
// `surround.w` is 1 when it is drawn, and the blend happens on the encoded colour, as a blend pass would.
fn outline_over(enc: vec3f, p: vec2f, s: vec2f) -> vec3f {
    if (v.surround.w < 0.5) { return enc; }
    let px = 1.0 / v.zoom;
    let lo = v.format.xy; let hi = v.format.xy + v.format.zw;
    let inside_outer = s.x >= lo.x - px && s.y >= lo.y - px && s.x <= hi.x + px && s.y <= hi.y + px;
    let inside_inner = s.x >= lo.x + px && s.y >= lo.y + px && s.x <= hi.x - px && s.y <= hi.y - px;
    var a = 0.0; var c = vec3f(0.9, 0.9, 0.4);
    if (inside_outer && !inside_inner) { a = 0.85; }
    if (p.x + p.y < 28.0) { a = 1.0; c = vec3f(1.0, 0.55, 0.1); }
    if (a == 0.0) { return enc; }
    return mix(enc, encode(c), a);
}
@fragment fn fs(o: Out) -> @location(0) vec4f {
    let s = scene_at(o.p.xy);
    let rel = (s - v.image.xy) / v.image.zw;
    if (rel.x < 0. || rel.y < 0. || rel.x >= 1. || rel.y >= 1.) {
        return vec4f(outline_over(encode(v.surround.rgb), o.p.xy, s), 1.);
    }
    let t = vec2i(min(rel * v.tex, v.tex - vec2f(1.)));
    return vec4f(outline_over(encode(textureLoad(img, t, 0).rgb), o.p.xy, s), 1.);
}
"""


def backend_order(choice=None, platform=None):
    """The wgpu backends the display request tries, in order; None means "let wgpu choose".

    `choice` (or NODEBASED_DIRECT_BACKEND): d3d12, vulkan or auto. On Windows the default is D3D12
    first (W1, docs/WINDOWS_HYBRID_PRESENTATION.md: one presentation API with DXGI's documented
    hybrid flip path), with Vulkan the measured alternative; wgpu's own default there is Vulkan.
    Elsewhere the default is auto. A named backend that offers nothing falls back to auto."""
    choice = (choice if choice is not None else os.environ.get(BACKEND_ENV, "")).strip().lower()
    if not choice:
        choice = "d3d12" if (platform or sys.platform) == "win32" else "auto"
    named = {"d3d12": "D3D12", "dx12": "D3D12", "vulkan": "Vulkan"}.get(choice)
    return [named, None] if named else [None]


def _request(context, backend, kwargs):
    """One surface-filtered request, with wgpu-py's WGPU_BACKEND_TYPE set for the call only."""
    import wgpu
    saved = os.environ.get("WGPU_BACKEND_TYPE")
    try:
        if backend is not None:
            os.environ["WGPU_BACKEND_TYPE"] = backend
        return wgpu.gpu.request_adapter_sync(canvas=context, **kwargs)
    finally:
        if backend is not None:
            if saved is None:
                os.environ.pop("WGPU_BACKEND_TYPE", None)
            else:
                os.environ["WGPU_BACKEND_TYPE"] = saved


def choose_display_adapter(context, preference=None, backend=None):
    """(adapter, selection): an adapter wgpu reports as able to present to this window's surface.

    `get_preferred_format` alone proves nothing: on a display server without DRI3 it answers for an
    adapter whose `configure` then aborts the process in native code. The only safe test is to let wgpu
    filter by the surface (`request_adapter_sync(canvas=context)`, which passes `compatibleSurface`).
    The wish (`preference` or NODEBASED_DIRECT_ADAPTER: discrete, integrated, cpu) picks the power
    preference of the request; `backend_order` picks the backends. `selection` records what was
    requested, what was selected and every request that failed, with no claim about adapters that
    were never asked for (a handful of preference requests does not prove the others incompatible)."""
    from . import gpu3d
    wanted = (preference if preference is not None else os.environ.get(ADAPTER_ENV, "")).strip().lower()
    wanted_kind = {"discrete": "discretegpu", "integrated": "integratedgpu", "cpu": "cpu", "software": "cpu"}.get(wanted, "")
    requests = {"integrated": [dict(power_preference="low-power")], "cpu": [dict(force_fallback_adapter=True)]}.get(
        wanted, [dict(power_preference="high-performance")])
    requests += [dict(power_preference="high-performance"), dict(power_preference="low-power"), dict(force_fallback_adapter=True)]
    order = backend_order(backend)
    attempts, found = [], None
    for wanted_backend in order:
        for kwargs in requests:
            label = ", ".join(f"{k}={v}" for k, v in kwargs.items()) + f", backend={wanted_backend or 'auto'}"
            try:
                adapter = _request(context, wanted_backend, kwargs)
            except Exception as exc:
                attempts.append({"request": label, "error": str(exc)})
                continue
            attempts.append({"request": label, "device": gpu3d._name_of(adapter.info),
                             "backend": adapter.info.get("backend_type", "")})
            if found is None:
                found = adapter
            if not wanted_kind or gpu3d._kind_of(adapter.info) == wanted_kind:
                found = adapter
                break
        if found is not None:
            break
    if found is None:
        raise RuntimeError("no adapter can present to this window: " + "; ".join(
            f"{a['request']}: {a['error']}" for a in attempts))
    name, got_backend = gpu3d._name_of(found.info), found.info.get("backend_type", "")
    notes = []
    if wanted_kind and gpu3d._kind_of(found.info) != wanted_kind:
        notes.append(f"{wanted} was asked for; the surface-filtered requests selected {name}")
    if order[0] is not None and got_backend != order[0]:
        notes.append(f"{order[0]} was asked for; selected {got_backend or 'an unnamed backend'}")
    selection = {"requested_kind": wanted or "auto", "requested_backends": [b or "auto" for b in order],
                 "selected_device": name, "selected_backend": got_backend,
                 "attempts": attempts, "note": "; ".join(notes)}
    return found, selection


def pick_display_adapter(context, preference=None, backend=None):
    """(adapter, format, note): `choose_display_adapter` for callers that want only the note."""
    adapter, selection = choose_display_adapter(context, preference, backend)
    return adapter, context.get_preferred_format(adapter), selection["note"]


def padded_row(width, texel):
    """Bytes per row in a staging buffer: the picture's row rounded up to WebGPU's 256-byte copy alignment."""
    return (width * texel + ROW_ALIGN - 1) // ROW_ALIGN * ROW_ALIGN


def picture_layout(kind, source):
    """(width, height, texture format, bytes per texel) of a ledger payload's picture.

    Float frames above HALF_ABOVE pixels (4K and up) go up as half floats: a 4K float32 frame is 133 MB
    and took 48 ms to reach the RTX 3080 Ti over its USB4 link, more than a frame at 24 fps; as half
    floats it is 25 ms. The display cache already keeps settled frames as half floats."""
    if kind == "linear":
        width, height = int(source.shape[1]), int(source.shape[0])
        half = source.dtype == np.float16 or width * height > HALF_ABOVE
        return width, height, ("rgba16float" if half else "rgba32float"), (8 if half else 16)
    return int(source.width()), int(source.height()), "rgba8unorm", 4


def payload_layouts(payload):
    """(layout of A, layout of B or None) for a ledger payload; only a linear payload can carry a B frame."""
    layout = picture_layout(payload[0], payload[1])
    if payload[0] == "linear" and len(payload) > 5 and payload[5] is not None:
        return layout, picture_layout("linear", payload[5])
    return layout, None


_band_pool = None


def _bands(height):
    """Row ranges that split a copy across FILL_BANDS threads (numpy releases the GIL while it copies)."""
    step = max(1, -(-height // FILL_BANDS))
    return [(top, min(height, top + step)) for top in range(0, height, step)]


def _copy_in_bands(dst, src):
    global _band_pool
    if dst.shape[0] * dst.shape[1] < 1_000_000:
        np.copyto(dst, src, casting="unsafe")
        return
    if _band_pool is None:
        import concurrent.futures
        _band_pool = concurrent.futures.ThreadPoolExecutor(FILL_BANDS, thread_name_prefix="direct-band")
    jobs = [_band_pool.submit(np.copyto, dst[top:bottom], src[top:bottom], casting="unsafe")
            for top, bottom in _bands(dst.shape[0])]
    for job in jobs:
        job.result()


def fill_rows(view, kind, source, width, height, row, fmt=None):
    """Copy one picture into a mapped staging buffer (`view`, writable bytes), row by padded row.

    Runs on the filler thread. A float frame is copied straight into the buffer (a cropped,
    non-contiguous cache view included: numpy walks the strides, no intermediate array), cast to half
    floats on the way when `fmt` says so, in bands on several threads for a large picture; a QImage is
    converted to RGBA8888 first unless it already is. Returns True when a conversion was needed."""
    converted = False
    if kind == "linear":
        fmt = fmt or picture_layout(kind, source)[2]
        dtype = np.float16 if fmt == "rgba16float" else np.float32
        itemsize = np.dtype(dtype).itemsize
        dst = np.ndarray((height, row // itemsize), dtype, buffer=view)[:, :width * 4].reshape(height, width, 4)
        _copy_in_bands(dst, source[..., :4])
        converted = dtype != source.dtype
    else:
        from PySide6.QtGui import QImage
        image = source
        if image.format() != QImage.Format.Format_RGBA8888:
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)
            converted = True
        line = image.bytesPerLine()
        raw = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(height, line)
        dst = np.ndarray((height, row), np.uint8, buffer=view)
        dst[:, :width * 4] = raw[:, :width * 4]
    return converted


class StagingRing:
    """A few reused upload buffers, filled off the Qt thread, copied into the texture inside a draw.

    A slot moves free -> filling -> ready -> free. `stage` (Qt thread) maps a free slot (it is the one the
    card finished with longest ago, so the map does not wait in practice) and hands the newest pending
    picture to the filler thread, which copies it into the mapped memory. `take_newest` (Qt thread, inside
    a draw) returns the newest filled slot and puts any older filled slot back as free: only the newest
    ready picture is ever copied to the card. `release` unmaps a slot whose copy has been recorded.
    The buffers are made once and grow only when a larger picture arrives."""

    def __init__(self, device, wgpu, count=STAGING_SLOTS, threaded=True, notify=None, counters=None):
        import concurrent.futures
        self.device, self.wgpu = device, wgpu
        self.slots = [{"buffer": None, "capacity": 0, "state": "free", "sequence": 0, "used": 0, "meta": None}
                      for _ in range(count)]
        self.lock = threading.Lock()
        self.notify = notify
        self.pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="direct-fill") if threaded else None
        self.jobs = []
        self.counters = counters if counters is not None else collections.Counter()
        self.clock = 0

    def idle_slot(self):
        with self.lock:
            free = [slot for slot in self.slots if slot["state"] == "free"]
            if not free:
                return None
            slot = min(free, key=lambda s: s["used"])
            slot["state"] = "filling"
            return slot

    def stage(self, sequence, payload, layout, layout_b=None):
        """Start filling a slot with `payload` (a ledger payload). False when every slot is busy.

        With `layout_b` the B frame goes into the same buffer after A (its offset a multiple of 256), so A and B
        always arrive together and the pair counts once against the buffer limit."""
        width, height, fmt, texel = layout
        row = padded_row(width, texel)
        size = row * height
        b = None
        if layout_b is not None:
            b_width, b_height, b_fmt, b_texel = layout_b
            b_row = padded_row(b_width, b_texel)
            offset = (size + ROW_ALIGN - 1) // ROW_ALIGN * ROW_ALIGN
            b = {"offset": offset, "width": b_width, "height": b_height, "fmt": b_fmt, "texel": b_texel, "row": b_row}
            size = offset + b_row * b_height
        slot = self.idle_slot()
        if slot is None:
            return False
        wgpu = self.wgpu
        started = time.perf_counter()
        buffer = slot["buffer"]
        if buffer is None or slot["capacity"] < size:
            if buffer is not None:
                if buffer.map_state == "mapped":
                    buffer.unmap()
                buffer.destroy()
            buffer = self.device.create_buffer(size=size, usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
                                               mapped_at_creation=True)
            slot["buffer"], slot["capacity"] = buffer, size
            self.counters["buffer_creates"] += 1
        elif buffer.map_state != "mapped":
            buffer.map_sync("WRITE")
        self.counters["map_ms_total"] += (time.perf_counter() - started) * 1000.0
        view = buffer._experimental_get_mapped_range(0, size)
        slot["meta"] = (payload, width, height, fmt, texel, row, b)
        slot["sequence"] = sequence
        if self.pool is None:
            self._fill(slot, view)
        else:
            self.jobs = [job for job in self.jobs if not job.done()]
            self.jobs.append(self.pool.submit(self._fill, slot, view))
        return True

    def _fill(self, slot, view):
        payload, width, height, fmt, texel, row, b = slot["meta"]
        started = time.perf_counter()
        try:
            converted = fill_rows(view, payload[0], payload[1], width, height, row, fmt)
            if b is not None:
                end = b["offset"] + b["row"] * b["height"]
                converted = fill_rows(view[b["offset"]:end], "linear", payload[5], b["width"], b["height"], b["row"],
                                      b["fmt"]) or converted
            state = "ready"
        except Exception:
            converted, state = False, "failed"
        with self.lock:
            slot["state"] = state
            self.counters["fills"] += 1
            self.counters["conversions"] += int(converted)
            self.counters["fill_ms_total"] += (time.perf_counter() - started) * 1000.0
        if self.notify is not None:
            try:
                self.notify()
            except Exception:
                pass

    def take_newest(self):
        """The newest filled slot, unmapped and ready to copy from, or None. Older filled slots go back to free
        (still mapped, so the next `stage` writes into them without a map)."""
        with self.lock:
            ready = [slot for slot in self.slots if slot["state"] in ("ready", "failed")]
            if not ready:
                return None
            newest = max(ready, key=lambda s: s["sequence"])
            for slot in ready:
                if slot is not newest:
                    slot["state"] = "free"
                    self.counters["dropped_ready"] += 1
            if newest["state"] == "failed":
                newest["state"] = "free"
                return None
            newest["state"] = "copying"
        newest["buffer"].unmap()          # a copy source must be unmapped before the submit
        return newest

    def release(self, slot):
        """After the copy from `slot` was submitted: the slot is free again (mapped afresh when next used)."""
        self.clock += 1
        with self.lock:
            slot["used"] = self.clock
            slot["state"] = "free"
            slot["meta"] = None

    def newest_unshown(self, after):
        """(sequence, payload) of the newest staged picture newer than sequence `after`, waiting for fills first."""
        self.wait()
        with self.lock:
            staged = [slot for slot in self.slots if slot["meta"] is not None and slot["sequence"] > after
                      and slot["state"] in ("ready", "filling", "copying")]
            if not staged:
                return None
            newest = max(staged, key=lambda s: s["sequence"])
            return newest["sequence"], newest["meta"][0]

    def waiting(self):
        """True while a picture is being filled or is filled and waiting for its draw. One picture at a time
        sits between the offer and the copy, so a newer offer replaces the pending one and an older picture
        is never converted only to be dropped."""
        with self.lock:
            return any(slot["state"] in ("filling", "ready") for slot in self.slots)

    def wait(self):
        """Until every fill in flight has finished (tests, close, device loss)."""
        for job in list(self.jobs):
            try:
                job.result()
            except Exception:
                pass
        self.jobs = []

    def bytes(self):
        return sum(slot["capacity"] for slot in self.slots)

    def close(self):
        self.wait()
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        for slot in self.slots:
            buffer = slot["buffer"]
            if buffer is not None:
                try:
                    if buffer.map_state == "mapped":
                        buffer.unmap()
                    buffer.destroy()
                except Exception:
                    pass
            slot.update(buffer=None, capacity=0, state="free", meta=None)


class RetainedPresenter:
    """Owns the device, the retained picture texture and the two pipelines; draws into the surface."""

    def __init__(self, context, adapter=None, preference=None, overlay=True, fill_thread=True, notify=None):
        import wgpu
        self.overlay = overlay
        self.fill_thread = fill_thread
        self.notify = notify         # called from the filler thread when a picture is ready to copy
        self.ring = None
        self.wgpu = wgpu
        self.context = context
        self.selection_note = ""
        self.selection = {}
        if adapter is None:
            adapter, self.selection = choose_display_adapter(context, preference)
            self.selection_note = self.selection["note"]
        fmt = context.get_preferred_format(adapter)
        self.adapter = adapter
        self.adapter_info = dict(adapter.info)
        self.format = fmt
        self.ledger = FrameLedger()
        self.counters = collections.Counter()
        self.device = None
        self.texture = None
        self.texture_size = None     # size of the picture in the texture (the texture may be larger)
        self.texture_capacity = (0, 0)
        self.bind_group = None
        self.picture = None          # the scene geometry of the retained picture
        self.linear = None           # display settings when the retained picture is a linear float frame
        self.linear_texture = None
        self.linear_size = None
        self.linear_capacity = (0, 0)
        self.linear_bind_group = None
        self.linear_by_format = {}   # texture format -> (texture, capacity, bind group)
        self.linear_b_by_format = {}  # texture format -> (texture, capacity): the B buffer of an A/B compare
        self.linear_b_texture = None
        self.compare = "A only"      # the compare mode of the retained picture ("A only" when it has no B)
        self.group_key = None        # (A texture, B texture) the current linear bind group was made for
        self.uniform_data = None     # the view block last written to the uniform buffer
        self.limits = {}
        self.refusal = ""            # why the last picture was refused (too large for this card), or ""
        self.linear_pipelines = {}   # view -> pipeline running carddisplay's fragment shader
        self.paint_texture = None    # the painted overlay, one texel per surface pixel (RGBA8, premultiplied)
        self.paint_size = None
        self.paint_group = None
        self.paint_key = None        # what the texture was painted for; the widget compares before painting again
        self.paint_image = None      # the QImage the paint texture holds (kept to skip an identical upload)
        self.paint_copies = []       # (buffer, bytes per row, first row, rows, width, texture, first column): overlay bands the next draw copies in
        self.badges = collections.OrderedDict()
        self.badge_shown = None
        self.linear_error = ""
        self.uploaded_sequence = 0
        self.last_frame = None       # the ledger payload ("qimage"|"linear", ...) the texture was made from
        self.recoveries = []         # perf_counter stamps of device recoveries
        self.recovery_reason = ""
        self.loss_watch = LossWatch()
        logging.getLogger("wgpu").addHandler(self.loss_watch)
        self._start_device()

    # -- device ---------------------------------------------------------------------------------
    def _start_device(self):
        wgpu = self.wgpu
        # A 4K float picture is 133 MB in one staging buffer; ask for what the adapter can do, not the 256 MB default.
        offered = dict(self.adapter.limits)
        wanted = {key: offered[key] for key in ("max-buffer-size", "max-texture-dimension-2d") if key in offered}
        try:
            self.device = self.adapter.request_device_sync(required_limits=wanted)
        except Exception:
            self.device = self.adapter.request_device_sync()
        self.limits = {key: int(self.device.limits[key]) for key in ("max-buffer-size", "max-texture-dimension-2d")}
        self.ring = StagingRing(self.device, wgpu, threaded=self.fill_thread, notify=self.notify, counters=self.counters)
        self.context.configure(device=self.device, format=self.format, alpha_mode="opaque")
        module = self.device.create_shader_module(code=SHADER)
        target = {"format": self.format}
        self.pipeline = self.device.create_render_pipeline(
            layout="auto", vertex={"module": module, "entry_point": "vs"},
            fragment={"module": module, "entry_point": "fs", "targets": [target]})
        self.uniform = self.device.create_buffer(size=UNIFORM_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.uniform_data = None
        paint_module = self.device.create_shader_module(code=PAINT_SHADER)
        self.paint_pipeline = self.device.create_render_pipeline(
            layout="auto", vertex={"module": paint_module, "entry_point": "vs"},
            fragment={"module": paint_module, "entry_point": "fs", "targets": [{"format": self.format, "blend": {
                "color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}}]})
        self.paint_flags = self.device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(self.paint_flags, 0, np.array(
            [1.0 if self.format.endswith("srgb") else 0.0, 0, 0, 0], np.float32))
        self.paint_texture = self.paint_group = self.paint_size = self.paint_key = self.paint_image = None
        self.paint_copies = []
        badge_module = self.device.create_shader_module(code=BADGE_SHADER)
        self.badge_pipeline = self.device.create_render_pipeline(
            layout="auto", vertex={"module": badge_module, "entry_point": "vs"},
            fragment={"module": badge_module, "entry_point": "fs", "targets": [{"format": self.format, "blend": {
                "color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}}]})
        self.badges = collections.OrderedDict()   # key -> (texture, uniform buffer, bind group): labels already on the card
        self.badge_shown = None                    # the badges key drawn at each draw, or None for no badge
        self.vertex_module = module
        self.linear_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "non-filtering"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}}])
        self.tables_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "read-only-storage"}}])
        self.linear_sampler = self.device.create_sampler()
        self.linear_pipelines = {}

    def linear_pipeline(self, view):
        """(pipeline, tables bind group) for `view`, built once; raises carddisplay.Unsupported."""
        from . import carddisplay
        wgpu = self.wgpu
        if view not in self.linear_pipelines:
            source = carddisplay.fragment_source(view)
            tables = carddisplay.view_tables(view)
            try:
                module = self.device.create_shader_module(code=source, label=f"display_{view}_frag")
                pipeline = self.device.create_render_pipeline(
                    layout=self.device.create_pipeline_layout(bind_group_layouts=[self.linear_layout, self.tables_layout]),
                    vertex={"module": self.vertex_module, "entry_point": "vs"},
                    fragment={"module": module, "entry_point": "main", "targets": [{"format": self.format}]})
                buffer = self.device.create_buffer_with_data(data=tables.tobytes(), usage=wgpu.BufferUsage.STORAGE)
                group = self.device.create_bind_group(layout=self.tables_layout, entries=[
                    {"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": tables.nbytes}}])
                self.linear_pipelines[view] = (pipeline, group, buffer)
            except Exception as exc:
                self.linear_pipelines[view] = None
                self.counters["linear_shader_failures"] += 1
                self.linear_error = str(exc)
            if self.linear_pipelines[view] is None:
                raise carddisplay.Unsupported(f"the card could not build the {view} shader")
        if self.linear_pipelines[view] is None:
            raise carddisplay.Unsupported(f"the card could not build the {view} shader")
        return self.linear_pipelines[view]

    def lose_device(self, wait=True):
        """Drop every resource and bump the epoch; the caller makes a new presenter or reports failure.

        Closing while work is queued waits for the queue to finish first, so the device and the
        surface go away with nothing in flight. A device wgpu reported lost, or one after a failed draw
        (that caller passes wait=False), is not waited on."""
        if self.device is not None and wait and not self.device_lost():
            try:
                self.wait_idle()
                self.counters["idle_waits"] += 1
            except Exception:
                self.counters["idle_wait_failures"] += 1
        if self.ring is not None:
            self.ring.close()
            self.ring = None
        for resource in self.ledger.new_epoch():
            self._destroy(resource)
        tables = [entry[2] for entry in self.linear_pipelines.values() if entry is not None]
        floats = [entry[0] for entry in self.linear_by_format.values()]
        floats_b = [entry[0] for entry in self.linear_b_by_format.values()]
        for resource in (self.texture, *floats, *floats_b, self.uniform, self.paint_texture, self.paint_flags, *tables,
                         *self._badge_resources()):
            self._destroy(resource)
        self.badges, self.badge_shown = collections.OrderedDict(), None
        self.paint_texture = self.paint_group = self.paint_size = self.paint_key = self.paint_image = None
        self.paint_copies = []
        self.linear_by_format = {}
        self.linear_b_by_format = {}
        self.linear_b_texture = self.group_key = None
        self.compare = "A only"
        self.texture = self.bind_group = None
        self.texture_size = None
        self.texture_capacity = self.linear_capacity = (0, 0)
        self.linear_texture = self.linear_bind_group = self.linear = None
        self.linear_size = None
        self.linear_pipelines = {}
        self.last_frame = None       # the retained CPU picture goes with the device (dock closed)
        self.device = None
        logging.getLogger("wgpu").removeHandler(self.loss_watch)

    def device_lost(self):
        """True once wgpu has reported the device lost (checked at the top of every draw)."""
        return self.loss_watch.lost.is_set()

    def recover(self, reason=""):
        """Replace a lost or failing device: new epoch, every old resource released, a new device from the
        same adapter, the surface configured again, and the retained CPU picture queued for upload.

        Raises RuntimeError when recoveries come too fast (the dock then falls back to the Viewer)."""
        now = time.perf_counter()
        limit, window = RECOVERY_LIMIT
        self.recoveries = [t for t in self.recoveries if now - t < window]
        if len(self.recoveries) >= limit:
            raise RuntimeError(f"the device was lost {limit} times in {window:.0f} s ({reason})")
        self.recoveries.append(now)
        pending = self.ledger.pending            # newer than anything uploaded; survives the new epoch
        old_device = self.device
        if self.ring is not None:
            staged = self.ring.newest_unshown(self.uploaded_sequence)
            if staged is not None and (pending is None or staged[0] > pending[0]):
                pending = staged                 # staged but never copied: newer than the retained picture
            self.ring.close()
            self.ring = None
        for resource in self.ledger.new_epoch():
            self._destroy(resource)
        for resource in (self.texture, *[entry[0] for entry in self.linear_by_format.values()],
                         *[entry[0] for entry in self.linear_b_by_format.values()], self.uniform,
                         self.paint_texture, self.paint_flags, *self._badge_resources()):
            self._destroy(resource)
        self.badges, self.badge_shown = collections.OrderedDict(), None
        self.paint_texture = self.paint_group = self.paint_size = self.paint_key = self.paint_image = None
        self.paint_copies = []
        self.linear_by_format = {}
        self.linear_b_by_format = {}
        self.linear_b_texture = self.group_key = None
        self.compare = "A only"
        self.texture = self.bind_group = None
        self.texture_size = None
        self.texture_capacity = self.linear_capacity = (0, 0)
        self.linear_texture = self.linear_bind_group = self.linear = None
        self.linear_size = None
        self.picture = None
        try:
            if old_device is not None:
                old_device.destroy()
        except Exception:
            pass
        self.loss_watch.lost.clear()
        self._start_device()
        frame = pending[1] if pending is not None else self.last_frame
        if frame is not None:
            self.ledger.offer(frame, None, self.ledger.epoch)
            self.stage_pending()
        self.counters["recoveries"] += 1
        self.recovery_reason = reason

    def wait_idle(self):
        """Block until the queue has completed every submission so far.

        wgpu-py 0.32's `on_submitted_work_done` cannot be called against wgpu-native 29 (its callback
        type no longer matches), so this maps a 4-byte copy instead: queue order means the map cannot
        complete before every earlier submission has."""
        wgpu = self.wgpu
        probe = self.device.create_buffer(size=4, usage=wgpu.BufferUsage.COPY_SRC)
        try:
            self.device.queue.read_buffer(probe)
        finally:
            probe.destroy()

    @staticmethod
    def _destroy(resource):
        try:
            if resource is not None:
                resource.destroy()
        except Exception:
            pass

    # -- picture --------------------------------------------------------------------------------
    def check_size(self, width, height, texel):
        """Raise TooLarge when this card cannot hold a width x height picture as one texture."""
        reason = too_large(self.limits, width, height, texel, self.adapter_info.get("device", ""))
        if reason:
            raise TooLarge(reason)

    def check_pair(self, width, height, texel, layout_b):
        """Raise TooLarge when A and B together do not fit this card, or B is not the size of A."""
        b_width, b_height, _, b_texel = layout_b
        if (b_width, b_height) != (width, height):
            raise TooLarge(f"Direct picture: the B picture is {b_width} x {b_height} pixels and A is "
                           f"{width} x {height}; the Viewer shows them")
        reason = pair_too_large(self.limits, width, height, texel, b_texel, self.adapter_info.get("device", ""))
        if reason:
            raise TooLarge(reason)

    def _queue(self, payload, generation):
        (width, height, _, texel), layout_b = payload_layouts(payload)
        try:
            self.check_size(width, height, texel)
            if layout_b is not None:
                self.check_pair(width, height, texel, layout_b)
        except TooLarge as exc:
            self.refusal = str(exc)
            self.counters["refused_too_large"] += 1
            raise
        self.refusal = ""
        fresh = self.ledger.offer(payload, generation, self.ledger.epoch)
        if fresh:
            self.stage_pending()
        return fresh

    def offer(self, qimage, rect, format_rect, generation=None):
        """Queue the newest picture. `rect` and `format_rect` are scene rectangles. The copy into a staging
        buffer starts at once on the filler thread when a slot is free; the draw only records the GPU copy.
        Raises TooLarge for a picture the card cannot hold."""
        return self._queue(("qimage", qimage, tuple(rect), tuple(format_rect)), generation)

    def offer_linear(self, frame, settings, rect, format_rect, generation=None, frame_b=None, compare="A only"):
        """Queue a premultiplied scene-linear float frame (H x W x 4) and its display settings
        (`carddisplay.settings_from`). The card does the display transform; nothing converts it here.

        `frame_b` (the same shape, aligned to A) and `compare` (one of `carddisplay.COMPARE_IDS`) put an A/B
        compare on the card: B goes up beside A and the shader does the wipe, B alone or the composite."""
        if frame_b is None:
            compare = "A only"
        return self._queue(("linear", frame, settings, tuple(rect), tuple(format_rect), frame_b, compare), generation)

    def stage_pending(self):
        """Hand the newest pending picture to a free staging slot (Qt thread). While another picture is still
        being filled or waits for its draw, it stays pending: the next draw tries again, and a newer offer
        replaces it meanwhile."""
        if self.ring is None or self.ledger.pending is None or self.ring.waiting():
            return False
        sequence, payload = self.ledger.pending
        layout, layout_b = payload_layouts(payload)
        if not self.ring.stage(sequence, payload, layout, layout_b):
            return False
        self.ledger.take()
        return True

    def retained_bytes(self):
        """CPU memory the presenter holds for recovery: the newest picture only (the ledger payload)."""
        frame = self.last_frame
        if frame is None:
            return 0
        source = frame[1]
        if frame[0] != "linear":
            return int(source.sizeInBytes())
        extra = frame[5] if len(frame) > 5 else None
        return int(source.nbytes) + (int(extra.nbytes) if extra is not None else 0)

    def b_texture_bytes(self):
        """Card memory of the B textures (one per texture format, grown but never shrunk)."""
        total = 0
        for texture, capacity in self.linear_b_by_format.values():
            total += capacity[0] * capacity[1] * (16 if texture.format == "rgba32float" else 8)
        return total

    def staging_bytes(self):
        return self.ring.bytes() if self.ring is not None else 0

    def _texture_for(self, kind, width, height, fmt):
        """The retained texture for `kind`, grown (never shrunk) so a width x height picture fits.

        Float pictures keep one texture per format: during playback a 4K frame (half floats) and the
        Viewer's visible crop (float32) alternate, and neither should replace the other's texture."""
        wgpu = self.wgpu
        linear = kind == "linear"
        if linear:
            texture, capacity, group = self.linear_by_format.get(fmt, (None, (0, 0), None))
            if texture is not None and capacity[0] >= width and capacity[1] >= height:
                self.linear_texture, self.linear_capacity = texture, capacity
                return texture
        else:
            texture, capacity = self.texture, self.texture_capacity
            if texture is not None and capacity[0] >= width and capacity[1] >= height:
                return texture
        if texture is not None:
            self.ledger.retire(texture)
        capacity = (max(capacity[0], width), max(capacity[1], height))
        texture = self.device.create_texture(size=(*capacity, 1), format=fmt,
                                             usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        if linear:
            self.linear_texture, self.linear_capacity = texture, capacity
            self.linear_bind_group = None      # `pair_group` makes it for the draw, from A and B together
            self.linear_by_format[fmt] = (texture, capacity, None)
        else:
            self.texture, self.texture_capacity = texture, capacity
            self.bind_group = self.device.create_bind_group(
                layout=self.pipeline.get_bind_group_layout(0),
                entries=[{"binding": 0, "resource": texture.create_view()},
                         {"binding": 1, "resource": {"buffer": self.uniform, "offset": 0, "size": 80}}])
        self.counters["texture_creates"] += 1
        return texture

    def _b_texture_for(self, width, height, fmt):
        """The texture for the B buffer of an A/B compare: one per format, grown (never shrunk) like A's."""
        texture, capacity = self.linear_b_by_format.get(fmt, (None, (0, 0)))
        if texture is not None and capacity[0] >= width and capacity[1] >= height:
            return texture
        if texture is not None:
            self.ledger.retire(texture)
        capacity = (max(capacity[0], width), max(capacity[1], height))
        texture = self.device.create_texture(size=(*capacity, 1), format=fmt,
                                             usage=self.wgpu.TextureUsage.TEXTURE_BINDING | self.wgpu.TextureUsage.COPY_DST)
        self.linear_b_by_format[fmt] = (texture, capacity)
        self.counters["b_texture_creates"] += 1
        return texture

    def pair_group(self, with_b):
        """The linear bind group for the retained A texture and (when the picture has one) its B texture. A
        picture without B binds A in B's place; the shader never reads it. Made again only when a texture was
        replaced (the group holds the texture objects it was made for)."""
        a = self.linear_texture
        b = self.linear_b_texture if with_b and self.linear_b_texture is not None else a
        key = self.group_key
        if self.linear_bind_group is None or key is None or key[0] is not a or key[1] is not b:
            self.linear_bind_group = self.device.create_bind_group(layout=self.linear_layout, entries=[
                {"binding": 0, "resource": a.create_view()},
                {"binding": 1, "resource": {"buffer": self.uniform, "offset": 0, "size": UNIFORM_BYTES}},
                {"binding": 2, "resource": self.linear_sampler},
                {"binding": 3, "resource": b.create_view()}])
            self.group_key = (a, b)
            self.counters["pair_groups"] += 1
        return self.linear_bind_group

    def _copy_slot(self, encoder, slot):
        """Record the copy from a filled staging slot into the retained texture; returns the payload."""
        payload, width, height, fmt, texel, row, b = slot["meta"]
        kind = payload[0]
        texture = self._texture_for(kind, width, height, fmt)
        encoder.copy_buffer_to_texture(
            {"buffer": slot["buffer"], "offset": 0, "bytes_per_row": row, "rows_per_image": height},
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)}, (width, height, 1))
        if b is not None:
            texture_b = self._b_texture_for(b["width"], b["height"], b["fmt"])
            encoder.copy_buffer_to_texture(
                {"buffer": slot["buffer"], "offset": b["offset"], "bytes_per_row": b["row"], "rows_per_image": b["height"]},
                {"texture": texture_b, "mip_level": 0, "origin": (0, 0, 0)}, (b["width"], b["height"], 1))
            self.linear_b_texture = texture_b
            self.counters["b_uploads"] += 1
            self.counters["upload_bytes"] += b["row"] * b["height"]
        if kind == "linear":
            self.linear_size = (width, height)
            self.counters["linear_uploads"] += 1
        else:
            self.texture_size = (width, height)
            self.counters["uploads"] += 1
        self.counters["upload_bytes"] += row * height
        return payload

    def set_paint_empty(self, key):
        """The overlay has nothing painted in it (the Viewer found out without making the image): the same state
        `set_paint` reaches for an all-clear image, minus the image. The texture keeps its last content."""
        self.paint_key = key
        self.paint_size = None
        self.counters["paint_empty"] += 1

    def set_paint(self, image, key, nonempty=False):
        """Show `image` (a premultiplied RGBA8888 QImage the size of the surface) over the picture, or nothing
        for None. `key` records what it was painted for. One upload when it changes, none otherwise.

        An image with nothing painted in it (a paused picture with no handle, wipe or readout) is not uploaded
        and its pass is skipped; an image identical to the one on the card is not uploaded again. Over the eGPU
        a 4 MB `write_texture` costs the Qt thread about 4 ms (10/22), and the Viewer repaints several times
        per edit, so most of these images were empty or repeats."""
        self.paint_key = key
        if image is None:
            self.paint_size = None
            self.paint_image = None
            return
        width, height = image.width(), image.height()
        pixels = np.frombuffer(image.constBits(), np.uint32, count=width * height)
        if not nonempty and not pixels.any():      # `nonempty`: the painter recorded a drawing command, so the scan is skipped
            self.paint_size = None          # nothing to draw: the pass is skipped; the texture keeps its last content
            self.counters["paint_empty"] += 1
            return
        previous = self.paint_image
        bands = [(0, height, 0, width)]
        if (previous is not None and (previous.width(), previous.height()) == (width, height)
                and self.paint_texture is not None):
            before = np.frombuffer(previous.constBits(), np.uint32, count=width * height)
            changed = np.flatnonzero((pixels != before).reshape(height, width).any(axis=1))
            if changed.size == 0:
                self.paint_size = (width, height)     # shown again; the card already holds exactly this
                self.counters["paint_unchanged"] += 1
                return
            # Only the rows that differ go up (a tier badge, a dragged handle), in bands: changes at the top and
            # the bottom of the surface are two small uploads, not one that spans it.
            breaks = np.flatnonzero(np.diff(changed) > PAINT_BAND_GAP)
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [changed.size - 1]))
            now, then = pixels.reshape(height, width), before.reshape(height, width)
            bands = []
            for a, b in zip(starts, ends):
                first, last = int(changed[a]), int(changed[b])
                # ... and within a band only the columns that differ: a tier badge is a hundred texels wide, and over
                # the eGPU the staging buffer costs the Qt thread about 4 ms a megabyte (10/22).
                columns = np.flatnonzero((now[first:last + 1] != then[first:last + 1]).any(axis=0))
                bands.append((first, last - first + 1, int(columns[0]), int(columns[-1]) - int(columns[0]) + 1))
        elif self.paint_texture is None or self.paint_texture.size[:2] != (width, height):
            self.ledger.retire(self.paint_texture)
            self.paint_texture = self.device.create_texture(
                size=(width, height, 1), format="rgba8unorm",
                usage=self.wgpu.TextureUsage.TEXTURE_BINDING | self.wgpu.TextureUsage.COPY_DST)
            self.paint_group = self.device.create_bind_group(
                layout=self.paint_pipeline.get_bind_group_layout(0),
                entries=[{"binding": 0, "resource": self.paint_texture.create_view()},
                         {"binding": 1, "resource": {"buffer": self.paint_flags, "offset": 0, "size": 16}}])
        line = image.bytesPerLine()
        whole = memoryview(image.constBits())
        for first_row, rows, first_column, columns in bands:
            # Recorded into the next draw's own command buffer (a copy from a small buffer), not sent with
            # `queue.write_texture`: over the eGPU a queue write issued right after the last draw's picture copy
            # held the Qt thread for the length of that copy (about 8 ms of CPU a call, 10/22).
            padded = padded_row(columns, 4)
            staged = np.zeros((rows, padded), np.uint8)
            staged[:, :columns * 4] = np.frombuffer(whole[first_row * line:(first_row + rows) * line], np.uint8,
                                                    count=rows * line).reshape(rows, line)[
                :, first_column * 4:(first_column + columns) * 4]
            buffer = self.device.create_buffer_with_data(data=staged.tobytes(), usage=self.wgpu.BufferUsage.COPY_SRC)
            self.paint_copies.append((buffer, padded, first_row, rows, columns, self.paint_texture, first_column))
            self.counters["paint_upload_rows"] += rows
            self.counters["paint_upload_texels"] += rows * columns
        self.paint_size = (width, height)
        self.paint_image = image
        self.counters["paint_uploads"] += 1

    def _badge_resources(self):
        return [item for entry in self.badges.values() for item in entry[:2]]

    def set_badge(self, key, make):
        """Show the tier badge for `key` (None for no badge) over the picture.

        `make()` is called only when this card has not seen `key` yet and returns (image, x, y): a premultiplied
        RGBA8888 QImage of the label and its top left corner in physical surface pixels. The texture, its
        uniform and its bind group are kept (the last BADGE_CACHE labels), so a tier change that goes back to a
        label already drawn costs no upload, and the label never goes through the overlay image."""
        self.badge_shown = None
        if key is None:
            return
        entry = self.badges.get(key)
        if entry is None:
            image, x, y = make()
            width, height = image.width(), image.height()
            padded = padded_row(width, 4)
            staged = np.zeros((height, padded), np.uint8)
            staged[:, :width * 4] = np.frombuffer(image.constBits(), np.uint8, count=height * image.bytesPerLine()).reshape(
                height, image.bytesPerLine())[:, :width * 4]
            texture = self.device.create_texture(
                size=(width, height, 1), format="rgba8unorm",
                usage=self.wgpu.TextureUsage.TEXTURE_BINDING | self.wgpu.TextureUsage.COPY_DST)
            self.device.queue.write_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)}, staged.tobytes(),
                {"offset": 0, "bytes_per_row": padded, "rows_per_image": height}, (width, height, 1))
            sized = key[-1]
            data = np.array([x, y, width, height, sized[0], sized[1], 1.0 if self.format.endswith("srgb") else 0.0, 0.0], np.float32)
            buffer = self.device.create_buffer_with_data(data=data.tobytes(), usage=self.wgpu.BufferUsage.UNIFORM)
            group = self.device.create_bind_group(layout=self.badge_pipeline.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": {"buffer": buffer, "offset": 0, "size": 32}}])
            entry = self.badges[key] = (texture, buffer, group)
            self.counters["badge_uploads"] += 1
            while len(self.badges) > BADGE_CACHE:
                _, old = self.badges.popitem(last=False)
                for resource in old[:2]:
                    self.ledger.retire(resource)
        else:
            self.badges.move_to_end(key)
            self.counters["badge_cached"] += 1
        self.badge_shown = key

    def has_picture(self):
        return self.texture is not None or self.linear_texture is not None

    # -- drawing --------------------------------------------------------------------------------
    def draw(self, view, physical_size, wipe=None):
        """One draw: upload the newest ready picture if any, then render picture + overlay into the surface.

        `view` is (centre_x, centre_y, zoom) in scene units and physical px per scene px. `wipe` is the wipe line
        as (centre_x, centre_y, normal_x, normal_y) in scene units, read by the caller at draw time so a drag
        needs no new upload; without one the line runs through the middle of the format. The surface texture and
        its view are locals here and are gone when this returns."""
        if self.device_lost():
            self.recover(self.loss_watch.message)
        self.stage_pending()
        encoder = self.device.create_command_encoder()
        copies, self.paint_copies = self.paint_copies, []
        for buffer, line, first_row, rows, width, texture, first_column in copies:
            encoder.copy_buffer_to_texture(
                {"buffer": buffer, "offset": 0, "bytes_per_row": line, "rows_per_image": rows},
                {"texture": texture, "mip_level": 0, "origin": (first_column, first_row, 0)}, (width, rows, 1))
            self.ledger.retire(buffer)
        slot = self.ring.take_newest()
        if slot is not None and slot["sequence"] > self.uploaded_sequence:
            started = time.perf_counter()
            payload = slot["meta"][0]
            if payload[0] == "linear":
                self.linear_pipeline(payload[2][3])
            self._copy_slot(encoder, slot)
            self.uploaded_sequence = slot["sequence"]
            self.last_frame = payload
            if payload[0] == "linear":
                _, frame, settings, rect, format_rect = payload[:5]
                self.linear = settings
                self.compare = payload[6] if len(payload) > 6 and payload[5] is not None else "A only"
            else:
                _, qimage, rect, format_rect = payload
                self.linear = None
                self.compare = "A only"
            self.picture = (rect, format_rect)
            self.counters["copy_record_ms_total"] += (time.perf_counter() - started) * 1000.0
        elif slot is not None:
            self.counters["dropped_ready"] += 1
        started = time.perf_counter()
        surface = self.context.get_current_texture()
        target = surface.create_view()
        if self.picture is None:
            encoder.begin_render_pass(color_attachments=[self._attachment(target)]).end()
            self.device.queue.submit([encoder.finish()])
        else:
            rect, format_rect = self.picture
            cx, cy, zoom = view
            if self.linear is not None:
                from . import carddisplay
                pipeline, tables, _ = self.linear_pipeline(self.linear[3])
                bind_group, size = self.pair_group(self.compare != "A only"), self.linear_size
                if wipe is None:
                    wipe = (format_rect[0] + format_rect[2] / 2.0, format_rect[1] + format_rect[3] / 2.0, 1.0, 0.0)
                display = carddisplay.uniform_values(self.linear) + carddisplay.compare_values(self.compare, wipe)
            else:
                pipeline, bind_group, size, tables = self.pipeline, self.bind_group, self.texture_size, None
                display = [0.0] * 20
            data = np.array([*physical_size, cx, cy, *rect, *format_rect, *size, zoom,
                             1.0 if self.format.endswith("srgb") else 0.0, *SURROUND, 1.0 if self.overlay else 0.0, *display], np.float32)
            if self.uniform_data is None or not np.array_equal(self.uniform_data, data):
                self.device.queue.write_buffer(self.uniform, 0, data)
                self.uniform_data = data
            rp = encoder.begin_render_pass(color_attachments=[self._attachment(target)])
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, bind_group)
            if tables is not None:
                rp.set_bind_group(1, tables)
            rp.draw(3)
            if self.paint_size is not None and tuple(physical_size) == self.paint_size:
                rp.set_pipeline(self.paint_pipeline)
                rp.set_bind_group(0, self.paint_group)
                rp.draw(3)
            if self.badge_shown is not None and self.badge_shown[-1] == tuple(physical_size) and self.badge_shown in self.badges:
                rp.set_pipeline(self.badge_pipeline)
                rp.set_bind_group(0, self.badges[self.badge_shown][2])
                rp.draw(6)
            rp.end()
            self.device.queue.submit([encoder.finish()])
        if slot is not None:
            self.ring.release(slot)
            self.stage_pending()          # a pending picture that found every slot busy goes now
        self.counters["draws"] += 1
        self.counters["draw_ms_total"] += (time.perf_counter() - started) * 1000.0
        for resource in self.ledger.note_present():
            self._destroy(resource)

    @staticmethod
    def _attachment(view):
        return {"view": view, "resolve_target": None, "clear_value": (*SURROUND, 1.0),
                "load_op": "clear", "store_op": "store"}


def unsupported_platform(name=None):
    """Why this Qt platform cannot show the direct picture, or "" when it can.

    The installed rendercanvas hands wgpu an X11 surface (its Wayland branch is switched off), so under Qt's
    native Wayland platform the window id is not an X window and the process dies with an X error. Run the
    application with QT_QPA_PLATFORM=xcb (Xwayland) for the direct picture."""
    if name is None:
        from PySide6.QtGui import QGuiApplication
        name = QGuiApplication.platformName()
    if str(name).lower().startswith("wayland"):
        return ("native Wayland is not supported by the screen surface (rendercanvas offers only X11 window ids); "
                "start NodeBased with QT_QPA_PLATFORM=xcb")
    return ""


# Qt platforms whose windows are not windows of the operating system: their window ids are Qt's own counters.
NO_NATIVE_WINDOW = ("offscreen", "minimal", "minimalegl", "vnc", "linuxfb")


def surface_problem(platform, window_id, physical_size, window_alive=None):
    """(why the card must not be given this window's surface, permanent) or ("", False) when it may.

    wgpu-native does not raise on an invalid surface: configure panics in its own thread and the process
    aborts (10/21, v0.41.0 Windows package job: offscreen Qt handed D3D12 a made-up window id, "Invalid
    surface"). So every reason is found here, before the surface is made. A permanent reason (no native
    windows on this platform) fails the dock; the others (no id yet, a minimised or hidden window, a
    window that went away) wait for the next picture, show or resize."""
    name = str(platform).lower()
    if name in NO_NATIVE_WINDOW:
        return f"the {name} Qt platform has no native window for the card to draw into", True
    if not window_id:
        return "the window has no native id yet", False
    if window_alive is not None and not window_alive(window_id):
        return "the native window is not a live window", False
    if physical_size[0] < 1 or physical_size[1] < 1:
        return "the window has no area (minimised or hidden)", False
    return "", False


def native_window_alive(window_id):
    """Windows: is the handle a live window with a client area. Elsewhere Qt's id is trusted."""
    if os.name != "nt":
        return True
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    hwnd = wintypes.HWND(window_id)
    if not user32.IsWindow(hwnd):
        return False
    rect = wintypes.RECT()
    return bool(user32.GetClientRect(hwnd, ctypes.byref(rect)))


def make_widget(parent=None):
    """The Qt surface. Imported lazily so the module (and its tests of the pure rules) load without rendercanvas."""
    reason = unsupported_platform()
    if reason:
        raise RuntimeError(reason)
    from PySide6.QtCore import Qt, Signal
    from rendercanvas.qt import QRenderWidget

    class DirectViewerWidget(QRenderWidget):
        failed = Signal(str)
        filled = Signal()              # emitted on the filler thread; Qt queues it to this widget's thread

        def __init__(self, parent=None):
            super().__init__(parent=parent, present_method="screen", update_mode="ondemand", max_fps=120)
            self.presenter = None
            self.error = None
            self.view = (0.0, 0.0, 1.0)
            self.auto_fit = True
            self.rect = None              # scene rectangle of the picture
            self.format_rect = None
            self._pan = None
            self.draw_requests = 0
            self.recoveries = 0
            self.refusal = ""             # why the last picture was refused (too large for the card), or ""
            self.overlay_provider = None  # (view, physical_size, ratio) -> premultiplied RGBA QImage or None
            self.badge_text = None        # () -> the tier badge's label ("" for none), when the owner draws it from the card
            self.badge_maker = None       # (text, physical_size, ratio) -> (QImage, x, y): the label's pixels, asked once per label
            self.overlay_probed = False   # the provider records first and answers EMPTY/SAME_OVERLAY; an image it returns has content
            self.compare_provider = None  # () -> the wipe line (centre x, y, normal x, y) in scene units, or None
            self.wipe_handle_provider = None   # (logical zoom) -> the turn handle's scene point (x, y) when the card draws the line and handles, or None
            self.hover = None             # (scene point or None) -> None; the owner updates its readout
            self.edit = None              # (kind, scene point, button, modifiers, zoom, pressure) -> handled?; None: view only
            self._editing = False         # a left press was taken by a handle and its release is owed
            self.overlay_serial = 0       # bumped by the owner when the overlays changed
            self.overlay_draws = 0
            self.waiting = ""             # why the surface is not made yet (no area, no native window), or ""
            self._retry = None            # the hand-over to repeat once the window can take a surface
            self._window_id = None        # the native window the surface was made for
            self.setMouseTracking(True)
            self.filled.connect(self._request)
            self.request_draw(self._draw)

        # -- feeding ----------------------------------------------------------------------------
        def submit(self, qimage, rect, format_rect=None, generation=None):
            """Hand over the newest picture. No GPU work and no waiting: the draw callback does the upload."""
            if self.error is not None:
                return False
            if self.presenter is None and not self._start(lambda: self.submit(qimage, rect, format_rect, generation)):
                return False
            try:
                fresh = self.presenter.offer(qimage, tuple(rect), tuple(format_rect or rect), generation)
            except TooLarge as exc:
                self.refusal = str(exc)
                return False
            self.refusal = ""
            self.rect = tuple(rect)
            self.format_rect = tuple(format_rect or rect)
            if fresh:
                self._request()
            return fresh

        def submit_linear(self, frame, settings, rect, format_rect=None, generation=None, frame_b=None,
                          compare="A only"):
            """Hand over a scene-linear float frame; the card runs the display transform. `frame_b` and
            `compare` add an A/B compare (a second float texture, the wipe, B alone or a composite).
            False when the direct viewer is down or the card cannot build this view's shader."""
            from . import carddisplay
            if self.error is not None:
                return False
            if self.presenter is None and not self._start(
                    lambda: self.submit_linear(frame, settings, rect, format_rect, generation, frame_b, compare)):
                return False
            try:
                self.presenter.linear_pipeline(settings[3])
            except carddisplay.Unsupported:
                return False
            try:
                fresh = self.presenter.offer_linear(frame, settings, tuple(rect), tuple(format_rect or rect), generation,
                                                    frame_b, compare)
            except TooLarge as exc:
                self.refusal = str(exc)
                return False
            self.refusal = ""
            self.rect = tuple(rect)
            self.format_rect = tuple(format_rect or rect)
            if fresh:
                self._request()
            return fresh

        def _surface_problem(self):
            from PySide6.QtGui import QGuiApplication
            return surface_problem(QGuiApplication.platformName(), int(self.winId()), self._physical(),
                                   native_window_alive)

        def _start(self, retry=None):
            reason, permanent = self._surface_problem()
            if reason:
                if permanent:
                    self._fail(f"direct viewer could not start: {reason}")
                else:
                    self.waiting = reason
                    self._retry = retry
                return False
            self.waiting = ""
            self._retry = None
            self._window_id = int(self.winId())
            try:
                self.presenter = RetainedPresenter(self.get_wgpu_context(), notify=self.filled.emit)
            except Exception as exc:
                self._fail(f"direct viewer could not start: {exc}")
                return False
            return True

        def _request(self):
            self.draw_requests += 1
            self.request_draw()

        def overlay_dirty(self):
            """The overlays (or the readout) changed: paint them again at the next draw."""
            self.overlay_serial += 1
            self._request()

        def scene_point(self, x, y):
            """Scene point under physical surface pixel (x, y)."""
            cx, cy, zoom = self.view
            width, height = self._physical()
            return cx + (x - width / 2.0) / zoom, cy + (y - height / 2.0) / zoom

        def _paint_overlay(self, size):
            provider = self.overlay_provider
            presenter = self.presenter
            if provider is None:
                if presenter.paint_size is not None:
                    presenter.set_paint(None, None)
                return
            ratio = self.get_pixel_ratio()
            key = (self.view, tuple(size), ratio, self.overlay_serial)
            if presenter.paint_key == key:
                return
            image = provider(self.view, tuple(size), ratio)
            if image is SAME_OVERLAY and presenter.paint_size is None:
                image = provider(self.view, tuple(size), ratio, force=True)       # the card holds nothing to keep (cleared, recovered)
            if image is EMPTY_OVERLAY:
                presenter.set_paint_empty(key)
            elif image is SAME_OVERLAY:
                presenter.paint_key = key
                presenter.counters["paint_unchanged"] += 1
            else:
                presenter.set_paint(image, key, nonempty=self.overlay_probed)
            self.overlay_draws += 1

        def _paint_badge(self, size):
            """The tier badge comes from a texture the card keeps per label, not from the overlay image."""
            presenter = self.presenter
            text = self.badge_text() if self.badge_text is not None and self.badge_maker is not None else ""
            if not text:
                presenter.set_badge(None, None)
                return
            ratio = float(self.get_pixel_ratio())
            size = tuple(size)
            presenter.set_badge((text, ratio, size), lambda: self.badge_maker(text, size, ratio))

        def _fail(self, message):
            self.error = message
            if self.presenter is not None:
                self.presenter.lose_device(wait=False)
            self.presenter = None
            self.failed.emit(message)

        def _physical(self):
            return self.get_physical_size()

        def _draw(self):
            if self.presenter is None or self.error is not None:
                return
            size = self._physical()
            if size[0] < 1 or size[1] < 1:
                return
            if self._window_id is not None and not native_window_alive(self._window_id):
                # A configure on the surface of a destroyed window panics inside wgpu: give the card up first.
                self._fail("direct viewer stopped: its native window went away")
                return
            if self.auto_fit and self.rect is not None:
                self.view = fit_view(size, self.rect)
            try:
                wipe = self.compare_provider() if self.compare_provider is not None else None
                ratio = float(self.get_pixel_ratio())
                handle = self.wipe_handle_provider(self.view[2] / ratio) if self.wipe_handle_provider is not None else None
                if wipe is not None and handle is not None:
                    wipe = (*wipe[:4], handle[0], handle[1], ratio)
                try:
                    self._paint_overlay(size)
                    self._paint_badge(size)
                    self.presenter.draw(self.view, size, wipe)
                except Exception as exc:
                    # A lost or failing device is replaced once; the same draw is then repeated on the new one.
                    self.presenter.recover(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
                    self.recoveries += 1
                    self.presenter.draw(self.view, size, wipe)
            except Exception as exc:
                self._fail(f"direct viewer draw failed: {exc}")

        # -- input: change view state, ask for a draw, return ------------------------------------
        def _cursor(self, event):
            ratio = self.get_pixel_ratio()
            point = event.position()
            return point.x() * ratio, point.y() * ratio

        def fit(self):
            self.auto_fit = True
            self._request()

        def wheelEvent(self, event):
            delta = event.angleDelta()
            steps = (delta.y() or delta.x()) / 120
            if steps and self.rect is not None:
                ratio = self.get_pixel_ratio()
                factor = 1.15 ** steps
                new = zoom_about(self.view, self._cursor(event), self._physical(), factor)
                if ZOOM_RANGE[0] < new[2] / ratio < ZOOM_RANGE[1]:
                    self.view = new
                    self.auto_fit = False
                    self._request()
            event.accept()

        def _edit_point(self, event):
            return self.scene_point(*self._cursor(event))

        def _edit_zoom(self):
            """Logical pixels per scene unit: the Viewer's tolerances are in these."""
            return self.view[2] / self.get_pixel_ratio()

        @staticmethod
        def _pressure(event):
            return float(event.pressure()) if callable(getattr(event, "pressure", None)) else 1.0

        def mirror_cursor(self, cursor):
            """The Viewer set (or cleared) its cursor for a handle drag or a pick; show the same here."""
            if cursor is None:
                self.unsetCursor()
            else:
                self.setCursor(cursor)

        def edit_release(self):
            """Forget a press in flight (editing switched off, or the dock closing)."""
            self._editing = False

        def mousePressEvent(self, event):
            mods = event.modifiers()
            button = event.button()
            if self._pan is None and (button == Qt.MouseButton.MiddleButton or (
                    button == Qt.MouseButton.LeftButton and mods == Qt.KeyboardModifier.AltModifier)):
                self._pan = (self._cursor(event), button)
            elif button == Qt.MouseButton.LeftButton and self.edit is not None and self.rect is not None:
                self._editing = bool(self.edit("press", self._edit_point(event), button, mods,
                                               self._edit_zoom(), self._pressure(event)))
            event.accept()

        def mouseMoveEvent(self, event):
            if self._pan is None and self.hover is not None and self.rect is not None:
                self.hover(self.scene_point(*self._cursor(event)))
            if self._pan is not None:
                (px, py), button = self._pan
                x, y = self._cursor(event)
                cx, cy, zoom = self.view
                self.view = (cx - (x - px) / zoom, cy - (y - py) / zoom, zoom)
                self._pan = ((x, y), button)
                self.auto_fit = False
                self._request()
            elif self.edit is not None and self.rect is not None:
                # Nothing is held except while a press was taken, or a Roto/warp shape is being drawn.
                self.edit("move", self._edit_point(event), None, event.modifiers(), self._edit_zoom(),
                          self._pressure(event))
            event.accept()

        def mouseReleaseEvent(self, event):
            if self._pan is not None and event.button() == self._pan[1]:
                self._pan = None
            elif self._editing and event.button() == Qt.MouseButton.LeftButton and self.edit is not None:
                self._editing = False
                self.edit("release", self._edit_point(event), event.button(), event.modifiers(),
                          self._edit_zoom(), self._pressure(event))
            event.accept()

        def leaveEvent(self, event):
            if self.hover is not None:
                self.hover(None)
            super().leaveEvent(event)

        def mouseDoubleClickEvent(self, event):
            self.fit()
            event.accept()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._retry_start()
            self._request()

        def showEvent(self, event):
            super().showEvent(event)
            self._retry_start()

        def _retry_start(self):
            """A hand-over that waited for the window (no area, no native id yet) is repeated once it may work."""
            if self._retry is not None and self.presenter is None and self.error is None:
                from PySide6.QtCore import QTimer
                retry, self._retry = self._retry, None
                QTimer.singleShot(0, retry)

        def release(self):
            """Drop the device, the textures, the staging buffers and the retained CPU picture (dock closed).
            The next picture handed over starts a fresh presenter."""
            self._retry = None
            if self.presenter is not None:
                self.presenter.lose_device()
                self.presenter = None

        def retained_bytes(self):
            return 0 if self.presenter is None else self.presenter.retained_bytes()

        def closeEvent(self, event):
            self.release()
            super().closeEvent(event)

    return DirectViewerWidget(parent)
