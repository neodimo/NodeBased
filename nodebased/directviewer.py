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
  are ordered before later submits); a different size makes a new texture and the old one is held
  for `RETIRE_PRESENTS` presents before `destroy()`.
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
import threading
import time

import numpy as np

ENV = "NODEBASED_DIRECT_VIEWER"
ADAPTER_ENV = "NODEBASED_DIRECT_ADAPTER"
RETIRE_PRESENTS = 3
UNIFORM_BYTES = 112           # the 80-byte view block, then carddisplay's two display vec4s
ZOOM_RANGE = (0.05, 20.0)     # logical zoom, the same bounds PanZoomView.wheelEvent applies
SURROUND = (0.0, 0.0, 0.0)    # VIEWER_SURROUND is pure black in every theme
RECOVERY_LIMIT = (3, 30.0)    # at most this many device recoveries in this many seconds, then the dock gives up


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
        """Called once per completed draw; returns the retired resources that are now safe to destroy."""
        self.presents += 1
        ready = [r for at, r in self._retired if self.presents - at >= RETIRE_PRESENTS]
        self._retired = [(at, r) for at, r in self._retired if self.presents - at < RETIRE_PRESENTS]
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
@fragment fn fs(o: Out) -> @location(0) vec4f {
    let s = scene_at(o.p.xy);
    let rel = (s - v.image.xy) / v.image.zw;
    if (rel.x < 0. || rel.y < 0. || rel.x >= 1. || rel.y >= 1.) { return vec4f(encode(v.surround.rgb), 1.); }
    let t = vec2i(min(rel * v.tex, v.tex - vec2f(1.)));
    return vec4f(encode(textureLoad(img, t, 0).rgb), 1.);
}
// Overlay pass: a one-pixel outline of the format rectangle plus an orange corner mark that says
// "experimental" without text (Qt widgets cannot be drawn over a screen-presented surface).
@fragment fn fs_overlay(o: Out) -> @location(0) vec4f {
    let s = scene_at(o.p.xy);
    let px = 1.0 / v.zoom;
    let lo = v.format.xy; let hi = v.format.xy + v.format.zw;
    let inside_outer = s.x >= lo.x - px && s.y >= lo.y - px && s.x <= hi.x + px && s.y <= hi.y + px;
    let inside_inner = s.x >= lo.x + px && s.y >= lo.y + px && s.x <= hi.x - px && s.y <= hi.y - px;
    var a = 0.0; var c = vec3f(0.9, 0.9, 0.4);
    if (inside_outer && !inside_inner) { a = 0.85; }
    if (o.p.x + o.p.y < 28.0) { a = 1.0; c = vec3f(1.0, 0.55, 0.1); }
    if (a == 0.0) { discard; }
    return vec4f(encode(c), a);
}
"""


def pick_display_adapter(context, preference=None):
    """(adapter, format, note): an adapter wgpu reports as able to present to this window's surface.

    `get_preferred_format` alone proves nothing: on a display server without DRI3 it answers for an
    adapter whose `configure` then aborts the process in native code. The only safe test is to let wgpu
    filter by the surface (`request_adapter_sync(canvas=context)`, which passes `compatibleSurface`).
    The wish (`preference` or NODEBASED_DIRECT_ADAPTER: discrete, integrated, cpu) picks the power
    preference of the request; when the surface-compatible adapter is a different kind, the note says so."""
    import wgpu
    from . import gpu3d
    wanted = (preference if preference is not None else os.environ.get(ADAPTER_ENV, "")).strip().lower()
    wanted_kind = {"discrete": "discretegpu", "integrated": "integratedgpu", "cpu": "cpu", "software": "cpu"}.get(wanted, "")
    requests = {"integrated": [dict(power_preference="low-power")], "cpu": [dict(force_fallback_adapter=True)]}.get(
        wanted, [dict(power_preference="high-performance")])
    requests += [dict(power_preference="high-performance"), dict(power_preference="low-power"), dict(force_fallback_adapter=True)]
    errors, found = [], None
    for kwargs in requests:
        try:
            adapter = wgpu.gpu.request_adapter_sync(canvas=context, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
            continue
        if found is None:
            found = adapter
        if not wanted_kind or gpu3d._kind_of(adapter.info) == wanted_kind:
            found = adapter
            break
    if found is None:
        raise RuntimeError("no adapter can present to this window: " + "; ".join(errors))
    name = gpu3d._name_of(found.info)
    note = "" if not wanted_kind or gpu3d._kind_of(found.info) == wanted_kind else (
        f"{wanted} was asked for but this window can only be presented to by {name}")
    return found, context.get_preferred_format(found), note


class RetainedPresenter:
    """Owns the device, the retained picture texture and the two pipelines; draws into the surface."""

    def __init__(self, context, adapter=None, preference=None, overlay=True):
        import wgpu
        self.overlay = overlay
        self.wgpu = wgpu
        self.context = context
        self.selection_note = ""
        if adapter is None:
            adapter, fmt, self.selection_note = pick_display_adapter(context, preference)
        else:
            fmt = context.get_preferred_format(adapter)
        self.adapter = adapter
        self.adapter_info = dict(adapter.info)
        self.format = fmt
        self.ledger = FrameLedger()
        self.counters = collections.Counter()
        self.device = None
        self.texture = None
        self.texture_size = None
        self.bind_group = None
        self.picture = None          # the scene geometry of the retained picture
        self.linear = None           # display settings when the retained picture is a linear float frame
        self.linear_texture = None
        self.linear_size = None
        self.linear_bind_group = None
        self.linear_pipelines = {}   # view -> pipeline running carddisplay's fragment shader
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
        self.device = self.adapter.request_device_sync()
        self.context.configure(device=self.device, format=self.format, alpha_mode="opaque")
        module = self.device.create_shader_module(code=SHADER)
        target = {"format": self.format}
        overlay_target = {"format": self.format, "blend": {
            "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
            "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}}
        self.pipeline = self.device.create_render_pipeline(
            layout="auto", vertex={"module": module, "entry_point": "vs"},
            fragment={"module": module, "entry_point": "fs", "targets": [target]})
        self.overlay_pipeline = self.device.create_render_pipeline(
            layout="auto", vertex={"module": module, "entry_point": "vs"},
            fragment={"module": module, "entry_point": "fs_overlay", "targets": [overlay_target]})
        self.uniform = self.device.create_buffer(size=UNIFORM_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.overlay_bind_group = None
        self.vertex_module = module
        self.linear_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "non-filtering"}}])
        self.linear_sampler = self.device.create_sampler()
        self.linear_pipelines = {}

    def linear_pipeline(self, view):
        """The display-transform pipeline for `view`, built once; raises carddisplay.Unsupported."""
        from . import carddisplay
        if view not in self.linear_pipelines:
            source = carddisplay.fragment_source(view)
            try:
                module = self.device.create_shader_module(code=source, label=f"display_{view}_frag")
                self.linear_pipelines[view] = self.device.create_render_pipeline(
                    layout=self.device.create_pipeline_layout(bind_group_layouts=[self.linear_layout]),
                    vertex={"module": self.vertex_module, "entry_point": "vs"},
                    fragment={"module": module, "entry_point": "main", "targets": [{"format": self.format}]})
            except Exception as exc:
                self.linear_pipelines[view] = None
                self.counters["linear_shader_failures"] += 1
                self.linear_error = str(exc)
            if self.linear_pipelines[view] is None:
                raise carddisplay.Unsupported(f"the card could not build the {view} shader")
        if self.linear_pipelines[view] is None:
            raise carddisplay.Unsupported(f"the card could not build the {view} shader")
        return self.linear_pipelines[view]

    def lose_device(self):
        """Drop every resource and bump the epoch; the caller makes a new presenter or reports failure."""
        for resource in self.ledger.new_epoch():
            self._destroy(resource)
        for resource in (self.texture, self.linear_texture, self.uniform):
            self._destroy(resource)
        self.texture = self.bind_group = self.overlay_bind_group = None
        self.texture_size = None
        self.linear_texture = self.linear_bind_group = self.linear = None
        self.linear_size = None
        self.linear_pipelines = {}
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
        for resource in self.ledger.new_epoch():
            self._destroy(resource)
        for resource in (self.texture, self.linear_texture, self.uniform):
            self._destroy(resource)
        self.texture = self.bind_group = self.overlay_bind_group = None
        self.texture_size = None
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
        self.counters["recoveries"] += 1
        self.recovery_reason = reason

    @staticmethod
    def _destroy(resource):
        try:
            if resource is not None:
                resource.destroy()
        except Exception:
            pass

    # -- picture --------------------------------------------------------------------------------
    def offer(self, qimage, rect, format_rect, generation=None):
        """Queue the newest picture (no GPU work here). `rect` and `format_rect` are scene rectangles."""
        return self.ledger.offer(("qimage", qimage, tuple(rect), tuple(format_rect)), generation, self.ledger.epoch)

    def offer_linear(self, frame, settings, rect, format_rect, generation=None):
        """Queue a premultiplied scene-linear float frame (H x W x 4) and its display settings
        (`carddisplay.settings_from`). The card does the display transform; nothing converts it here."""
        return self.ledger.offer(("linear", frame, settings, tuple(rect), tuple(format_rect)), generation, self.ledger.epoch)

    def _upload_linear(self, frame):
        """Write the float frame into the retained rgba32float texture (one copy, no conversion)."""
        wgpu = self.wgpu
        frame = np.ascontiguousarray(frame, np.float32)
        height, width = frame.shape[:2]
        if self.linear_texture is None or self.linear_size != (width, height):
            if self.linear_texture is not None:
                self.ledger.retire(self.linear_texture)
            self.linear_texture = self.device.create_texture(
                size=(width, height, 1), format="rgba32float",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self.linear_size = (width, height)
            self.linear_bind_group = self.device.create_bind_group(layout=self.linear_layout, entries=[
                {"binding": 0, "resource": self.linear_texture.create_view()},
                {"binding": 1, "resource": {"buffer": self.uniform, "offset": 0, "size": UNIFORM_BYTES}},
                {"binding": 2, "resource": self.linear_sampler}])
            if self.overlay_bind_group is None:
                self.overlay_bind_group = self.device.create_bind_group(
                    layout=self.overlay_pipeline.get_bind_group_layout(0),
                    entries=[{"binding": 1, "resource": {"buffer": self.uniform, "offset": 0, "size": 80}}])
            self.counters["texture_creates"] += 1
        self.device.queue.write_texture({"texture": self.linear_texture}, frame,
                                        {"bytes_per_row": width * 16, "rows_per_image": height}, (width, height, 1))
        self.counters["linear_uploads"] += 1

    def _upload(self, qimage):
        """Convert once to RGBA8 and write it to the retained texture; returns True when a size change made a new texture."""
        from PySide6.QtGui import QImage
        wgpu = self.wgpu
        rgba = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        self.counters["conversions"] += 1
        width, height = rgba.width(), rgba.height()
        raw = np.frombuffer(rgba.constBits(), np.uint8, count=rgba.sizeInBytes())
        new_texture = self.texture is None or self.texture_size != (width, height)
        if new_texture:
            if self.texture is not None:
                self.ledger.retire(self.texture)
            self.texture = self.device.create_texture(
                size=(width, height, 1), format="rgba8unorm",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self.texture_size = (width, height)
            self.bind_group = self.device.create_bind_group(
                layout=self.pipeline.get_bind_group_layout(0),
                entries=[{"binding": 0, "resource": self.texture.create_view()},
                         {"binding": 1, "resource": {"buffer": self.uniform, "offset": 0, "size": 80}}])
            if self.overlay_bind_group is None:
                self.overlay_bind_group = self.device.create_bind_group(
                layout=self.overlay_pipeline.get_bind_group_layout(0),
                entries=[{"binding": 1, "resource": {"buffer": self.uniform, "offset": 0, "size": 80}}])
            self.counters["texture_creates"] += 1
        self.device.queue.write_texture({"texture": self.texture}, raw,
                                        {"bytes_per_row": rgba.bytesPerLine(), "rows_per_image": height}, (width, height, 1))
        self.counters["uploads"] += 1
        return new_texture

    def has_picture(self):
        return self.texture is not None or self.linear_texture is not None

    # -- drawing --------------------------------------------------------------------------------
    def draw(self, view, physical_size):
        """One draw: upload the newest ready picture if any, then render picture + overlay into the surface.

        `view` is (centre_x, centre_y, zoom) in scene units and physical px per scene px. The surface
        texture and its view are locals here and are gone when this returns."""
        if self.device_lost():
            self.recover(self.loss_watch.message)
        taken = self.ledger.take()
        if taken is not None:
            self.last_frame = taken[1]
            kind, payload = taken[1][0], taken[1][1:]
            if kind == "linear":
                frame, settings, rect, format_rect = payload
                self.linear_pipeline(settings[3])
                self._upload_linear(frame)
                self.linear = settings
            else:
                qimage, rect, format_rect = payload
                self._upload(qimage)
                self.linear = None
            self.picture = (rect, format_rect)
            self.uploaded_sequence = taken[0]
        started = time.perf_counter()
        surface = self.context.get_current_texture()
        target = surface.create_view()
        if self.picture is None:
            encoder = self.device.create_command_encoder()
            encoder.begin_render_pass(color_attachments=[self._attachment(target)]).end()
            self.device.queue.submit([encoder.finish()])
        else:
            rect, format_rect = self.picture
            cx, cy, zoom = view
            if self.linear is not None:
                from . import carddisplay
                pipeline, bind_group, size = self.linear_pipeline(self.linear[3]), self.linear_bind_group, self.linear_size
                display = carddisplay.uniform_values(self.linear)
            else:
                pipeline, bind_group, size, display = self.pipeline, self.bind_group, self.texture_size, [0.0] * 8
            data = np.array([*physical_size, cx, cy, *rect, *format_rect, *size, zoom,
                             1.0 if self.format.endswith("srgb") else 0.0, *SURROUND, 1.0, *display], np.float32)
            self.device.queue.write_buffer(self.uniform, 0, data)
            encoder = self.device.create_command_encoder()
            rp = encoder.begin_render_pass(color_attachments=[self._attachment(target)])
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, bind_group)
            rp.draw(3)
            if self.overlay:
                rp.set_pipeline(self.overlay_pipeline)
                rp.set_bind_group(0, self.overlay_bind_group)
                rp.draw(3)
            rp.end()
            self.device.queue.submit([encoder.finish()])
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


def make_widget(parent=None):
    """The Qt surface. Imported lazily so the module (and its tests of the pure rules) load without rendercanvas."""
    reason = unsupported_platform()
    if reason:
        raise RuntimeError(reason)
    from PySide6.QtCore import Qt, Signal
    from rendercanvas.qt import QRenderWidget

    class DirectViewerWidget(QRenderWidget):
        failed = Signal(str)

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
            self.request_draw(self._draw)

        # -- feeding ----------------------------------------------------------------------------
        def submit(self, qimage, rect, format_rect=None, generation=None):
            """Hand over the newest picture. No GPU work and no waiting: the draw callback does the upload."""
            if self.error is not None:
                return False
            self.rect = tuple(rect)
            self.format_rect = tuple(format_rect or rect)
            if self.presenter is None and not self._start():
                return False
            fresh = self.presenter.offer(qimage, self.rect, self.format_rect, generation)
            if fresh:
                self._request()
            return fresh

        def _start(self):
            try:
                self.presenter = RetainedPresenter(self.get_wgpu_context())
            except Exception as exc:
                self._fail(f"direct viewer could not start: {exc}")
                return False
            return True

        def _request(self):
            self.draw_requests += 1
            self.request_draw()

        def _fail(self, message):
            self.error = message
            if self.presenter is not None:
                self.presenter.lose_device()
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
            if self.auto_fit and self.rect is not None:
                self.view = fit_view(size, self.rect)
            try:
                try:
                    self.presenter.draw(self.view, size)
                except Exception as exc:
                    # A lost or failing device is replaced once; the same draw is then repeated on the new one.
                    self.presenter.recover(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
                    self.recoveries += 1
                    self.presenter.draw(self.view, size)
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

        def mousePressEvent(self, event):
            mods = event.modifiers()
            if self._pan is None and (event.button() == Qt.MouseButton.MiddleButton or (
                    event.button() == Qt.MouseButton.LeftButton and mods == Qt.KeyboardModifier.AltModifier)):
                self._pan = (self._cursor(event), event.button())
            event.accept()

        def mouseMoveEvent(self, event):
            if self._pan is not None:
                (px, py), button = self._pan
                x, y = self._cursor(event)
                cx, cy, zoom = self.view
                self.view = (cx - (x - px) / zoom, cy - (y - py) / zoom, zoom)
                self._pan = ((x, y), button)
                self.auto_fit = False
                self._request()
            event.accept()

        def mouseReleaseEvent(self, event):
            if self._pan is not None and event.button() == self._pan[1]:
                self._pan = None
            event.accept()

        def mouseDoubleClickEvent(self, event):
            self.fit()
            event.accept()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._request()

        def closeEvent(self, event):
            if self.presenter is not None:
                self.presenter.lose_device()
                self.presenter = None
            super().closeEvent(event)

    return DirectViewerWidget(parent)
