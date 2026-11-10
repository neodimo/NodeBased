"""Roto shape mattes drawn on the graphics card (an opt-in path; the processor drawer stays the reference).

`nodebased.roto.rasterise` draws a Roto node's shapes into one premultiplied matte on the processor. With the
"Roto mattes on the GPU" preference on (or ``NODEBASED_ROTO_GPU=1``), a full-frame request whose shapes the
kernel covers is drawn here instead, on the same device the 3D and 2D paths use, and anything the kernel does
not cover (or any graphics error) returns to the processor drawer, so a turned-on switch can only make a
matte faster, never wrong or missing. With the switch off, nothing here runs.

What runs on the card, per shape, in this order (the processor drawer's own order and arithmetic):

1. **Fill** (one compute thread per pixel row of the shape's patch). The processor flattens the closed Bezier or
   B-Spline outline (`roto.flatten_points`) and works out, in double precision, which of the eight
   sub-scanlines of each pixel row every edge crosses and where it crosses the first of them
   (`rotofast._first_at_or_above`, the drawer's own rule); the card then does what the drawer does for each
   sub-scanline: the crossings of that line, sorted by x, a non-zero winding sum, exact horizontal coverage of
   each inside span, and the eight lines averaged. A compute pass was chosen over a render pass because the
   fill needs a per-row sort of the crossings and a sum into a row the thread owns; a rasteriser has no such
   step, and its multisample patterns could not give the drawer's exact horizontal coverage.
2. **Feather**: the box blur of the Blur node (`Evaluator._box_blur_axis`), horizontal then vertical, edge
   pixels repeated, over the same patch the drawer blurs (so the patch edge behaves the same).
3. **Finish and composite**: clamp, the feather falloff curve (smooth or gaussian), the falloff power, the shape's
   own invert, opacity, then the shape's blend into the running canvas (over, plus, multiply, the shape's
   subtract mode and the stated blend formulas).

Out of scope here, so each of these makes the whole call run on the processor: open strokes, RotoPaint brushes,
the signed-distance ("edge") feather of a Bezier shape, motion blur samples, picture sources (clone,
reveal), the Merge-style blend modes, any request for a region of the frame, and a patch that outgrows the card's
buffers. Tiles stay on the processor because a tile asks for part of a frame that the next tile also asks for.

The numbers are not bit-identical to the processor's: the card works in 32-bit floats where the drawer fills in
64-bit, so the two differ by about 1e-4 at most (docs/ROTO_TRACKING.md, "GPU mattes (past Nuke)", states the
measured differences). That is why the switch is off by default.
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np

ENV = "NODEBASED_ROTO_GPU"
SUBSAMPLES = 8
MAX_CROSSINGS = 128            # crossings one sub-scanline can hold on the card; a shape with more runs on the processor
MAX_BLUR_RADIUS = 256          # the box blur sums its window directly; a wider feather runs on the processor
FAIL_LIMIT = 3                 # graphics errors in a row after which the path turns itself off for the session

BLEND_CODES = {"over": 0, "plus": 2, "multiply": 3, "minus": 4, "from": 5, "difference": 6, "exclusion": 7,
               "max": 8, "min": 9, "screen": 10}
SUBTRACT = 1
FALLOFFS = {"smooth": 1, "gaussian": 2}

_lock = threading.RLock()
_wanted = False                # what the window last asked for (the saved preference)
_wanted_pid = None
_failures = 0
_choice = "default"
_buffers = {}                  # per device: reusable buffers
counters = {"gpu_mattes": 0, "cpu_fallbacks": 0, "unsupported": 0, "errors": 0, "last_error": "", "last_reason": "",
            "gpu_ms": 0.0}


class Unsupported(Exception):
    """The kernel does not cover this request; the processor drawer draws it."""


def set_enabled(on, choice="default"):
    """Switch the path on or off for this process (the window calls this with the saved preference)."""
    global _wanted, _wanted_pid, _failures, _choice
    _wanted = bool(on)
    _wanted_pid = os.getpid() if on else None
    _choice = choice or "default"
    if on:
        _failures = 0
    return enabled()


def enabled(environ=None):
    """True when Roto mattes should be tried on the card: the environment variable if it says 1 or 0, else the preference.

    Never true in a child process (the read-ahead matte process): it does not own the card."""
    try:
        import multiprocessing
        if multiprocessing.parent_process() is not None:
            return False
    except Exception:
        pass
    if _failures >= FAIL_LIMIT:
        return False
    forced = (os.environ if environ is None else environ).get(ENV, "").strip().lower()
    if forced in ("1", "true", "on", "yes"):
        return True
    if forced in ("0", "false", "off", "no"):
        return False
    return _wanted and _wanted_pid == os.getpid()


def forced(environ=None):
    """True when the environment fixes the switch, so the Settings choice has no effect."""
    return (os.environ if environ is None else environ).get(ENV, "").strip().lower() in (
        "1", "true", "on", "yes", "0", "false", "off", "no")


def reset_counters():
    global _failures
    _failures = 0
    counters.update(gpu_mattes=0, cpu_fallbacks=0, unsupported=0, errors=0, last_error="", last_reason="", gpu_ms=0.0)


# ----------------------------------------------------------------------------------------------
# Planning on the processor: which shapes, which patches, which edges.

def _blur_radius(feather):
    if feather < 0.5:
        return 0
    return max(0, int(round(feather)))


def plan(shapes, width, height, *, flattened_cache=None):
    """The kernel's inputs for `shapes`, or `Unsupported` naming what it cannot draw.

    Returns ``(items, edges)``: one dict per shape that paints something (patch, blur radius, curve codes ...)
    and a float32 ``(edge count, 2, 4)`` array."""
    from . import roto, rotofast
    w, h = int(width), int(height)
    items, edge_blocks, edge_total = [], [], 0
    for shape in shapes:
        if not shape.get("visible", True):
            continue
        if not shape.get("closed", True):
            raise Unsupported("an open stroke")
        curve = shape.get("curve_type", "bezier")
        if curve not in ("bezier", "bspline"):
            raise Unsupported(f"the {curve} curve type")
        if shape.get("feather_model", "box") == "edge" and curve == "bezier":
            raise Unsupported("the edge feather")
        if shape.get("_motion_samples") or shape.get("_motion_only"):
            raise Unsupported("motion blur")
        if shape.get("source", "color") != "color":
            raise Unsupported("a picture source")
        subtract = shape.get("mode") == "subtract"
        blend = shape.get("blend_mode", "over")
        if not subtract and blend not in BLEND_CODES:
            raise Unsupported(f"the {blend} blend mode")
        feather = float(shape.get("feather", 0.0))
        power = float(shape.get("falloff_power", 1.0))
        opacity = np.float32(shape["opacity"])
        color = np.asarray(shape.get("color", [1, 1, 1, 1]), dtype=np.float32)
        if not (math.isfinite(feather) and math.isfinite(power) and np.isfinite(opacity) and np.isfinite(color).all()):
            raise Unsupported("a non-finite number")
        radius = _blur_radius(feather)
        if radius > MAX_BLUR_RADIUS:
            raise Unsupported("a feather wider than the card's blur")
        inverted = bool(shape.get("invert", False))
        if inverted:
            sx0, sy0, sx1, sy1 = 0, 0, w, h
        else:
            bx0, by0, bx1, by1 = roto._shape_bounds(shape)
            sx0, sy0 = max(0, int(math.floor(bx0))), max(0, int(math.floor(by0)))
            sx1, sy1 = min(w, int(math.ceil(bx1))), min(h, int(math.ceil(by1)))
            if sx1 <= sx0 or sy1 <= sy0:
                continue
        sw, sh = sx1 - sx0, sy1 - sy0
        points = shape["points"]
        polyline = roto._flatten_cached(points, flattened_cache, curve_type=curve)
        if not np.isfinite(polyline).all():
            raise Unsupported("a non-finite point")
        edges = _edges(polyline, sx0, sy0, sw, sh, rotofast)
        items.append({
            "rect": (sx0, sy0, sw, sh), "edge_offset": edge_total, "edge_count": len(edges), "radius": radius,
            "falloff": FALLOFFS.get(shape.get("feather_falloff", "linear"), 0), "power": power,
            "has_power": power != 1.0, "invert": inverted,
            "full_alpha": bool(opacity * color[3] == 1.0), "opacity": float(opacity), "alpha": float(color[3]),
            "color": tuple(float(c) for c in color[:3]), "blend": SUBTRACT if subtract else BLEND_CODES[blend]})
        edge_blocks.append(edges)
        edge_total += len(edges)
    packed = np.concatenate(edge_blocks) if edge_blocks else np.zeros((0, 2, 4), dtype=np.float32)
    return items, packed


def _edges(polyline, ox, oy, width, height, rotofast):
    """The drawer's edge set-up (`rotofast.coverage`) for one patch, as float32 rows (x at the edge's first
    sub-scanline, x step per sub-scanline, first and one-past-last sub-scanline, winding direction)."""
    if polyline.shape[0] < 3:
        return np.zeros((0, 2, 4), dtype=np.float32)
    ax, ay = polyline[:, 0], polyline[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    rising = ay != by
    ax, ay, bx, by = ax[rising], ay[rising], bx[rising], by[rising]
    if ax.size == 0:
        return np.zeros((0, 2, 4), dtype=np.float32)
    direction = np.where(by > ay, 1.0, -1.0)
    top, bottom = np.minimum(ay, by), np.maximum(ay, by)
    slope = (bx - ax) / (by - ay)
    first = rotofast._first_at_or_above(top, float(oy), 0)
    last = np.minimum(rotofast._first_at_or_above(bottom, float(oy), 0), height * SUBSAMPLES)
    live = last > first
    if not live.any():
        return np.zeros((0, 2, 4), dtype=np.float32)
    ax, ay, slope, direction, first, last = ax[live], ay[live], slope[live], direction[live], first[live], last[live]
    y_first = oy + (first + 0.5) / SUBSAMPLES
    x_first = ax + (y_first - ay) * slope - ox
    out = np.zeros((len(first), 2, 4), dtype=np.float32)
    out[:, 0, 0] = x_first
    out[:, 0, 1] = slope / SUBSAMPLES
    out[:, 0, 2] = first
    out[:, 0, 3] = last
    out[:, 1, 0] = direction
    return out


# ----------------------------------------------------------------------------------------------
# Shaders

_STRUCT = """
struct P {
    rect: vec4<i32>,     // patch x, y, width, height (canvas pixels)
    canv: vec4<i32>,     // canvas width, canvas height, first edge, edge count
    mode: vec4<i32>,     // blur radius, falloff curve, flags (1 invert, 2 power, 4 full alpha), blend
    f0: vec4<f32>,       // falloff power, opacity, colour alpha, 1 / blur window
    col: vec4<f32>,      // colour
};
"""

_COVER = _STRUCT + """
@group(0) @binding(0) var<storage, read> edges: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cover: array<f32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
const MAXC: u32 = %(maxc)du;
var<private> xs: array<f32, %(maxc)d>;
var<private> ds: array<i32, %(maxc)d>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let sw = u32(p.rect.z);
    let sh = u32(p.rect.w);
    if (row >= sh) { return; }
    let base = row * sw;
    for (var x = 0u; x < sw; x++) { cover[base + x] = 0.0; }
    let eoff = u32(p.canv.z);
    let ecount = u32(p.canv.w);
    let swf = f32(sw);
    for (var sub = 0u; sub < 8u; sub++) {
        let s = f32(row * 8u + sub);
        var n = 0u;
        for (var i = 0u; i < ecount; i++) {
            let e0 = edges[2u * (eoff + i)];
            if (s >= e0.z && s < e0.w) {
                if (n < MAXC) {
                    xs[n] = clamp(e0.x + (s - e0.z) * e0.y, 0.0, swf);
                    ds[n] = i32(edges[2u * (eoff + i) + 1u].x);
                    n++;
                } else {
                    flags[0] = 1u;
                }
            }
        }
        for (var a = 1u; a < n; a++) {
            let kx = xs[a];
            let kd = ds[a];
            var b = a;
            while (b > 0u && xs[b - 1u] > kx) { xs[b] = xs[b - 1u]; ds[b] = ds[b - 1u]; b--; }
            xs[b] = kx;
            ds[b] = kd;
        }
        var wind = 0;
        for (var k = 0u; k + 1u < n; k++) {
            wind += ds[k];
            if (wind != 0) {
                let x0 = xs[k];
                let x1 = xs[k + 1u];
                if (x1 > x0 && x0 < swf) {
                    let f0 = u32(floor(x0));
                    let f1 = u32(floor(x1));
                    if (f0 == f1) {
                        cover[base + f0] += x1 - x0;
                    } else {
                        cover[base + f0] += f32(f0 + 1u) - x0;
                        for (var j = f0 + 1u; j < f1; j++) { cover[base + j] += 1.0; }
                        if (f1 < sw) { cover[base + f1] += x1 - f32(f1); }
                    }
                }
            }
        }
    }
    for (var x = 0u; x < sw; x++) { cover[base + x] = cover[base + x] * 0.125; }
}
""" % {"maxc": MAX_CROSSINGS}


def _blur(axis):
    # The window is summed in order, the first term then the next 2r, the same order as the Blur node's small-radius
    # kernel, with the edge pixel repeated beyond the patch; the sum is scaled by the reciprocal of the window.
    along, limit = ("x", "sw") if axis == 0 else ("y", "sh")
    return _STRUCT + """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sw = p.rect.z;
    let sh = p.rect.w;
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= sw || y >= sh) { return; }
    let r = p.mode.x;
    var acc = 0.0;
    for (var k = 0; k <= 2 * r; k++) {
        let q = clamp(%(along)s - r + k, 0, %(limit)s - 1);
        %(fetch)s
        if (k == 0) { acc = v; } else { acc = acc + v; }
    }
    dst[u32(y * sw + x)] = acc * p.f0.w;
}
""" % {"along": along, "limit": limit,
       "fetch": "let v = src[u32(y * sw + q)];" if axis == 0 else "let v = src[u32(q * sw + x)];"}


_FINISH = _STRUCT + """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> canvas: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: P;

fn blend_op(code: i32, a: f32, b: f32) -> f32 {
    if (code == 4) { return a - b; }
    if (code == 5) { return b - a; }
    if (code == 6) { return abs(a - b); }
    if (code == 7) { return a + b - 2.0 * a * b; }
    if (code == 8) { return max(a, b); }
    if (code == 9) { return min(a, b); }
    return a + b - a * b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sw = p.rect.z;
    let sh = p.rect.w;
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= sw || y >= sh) { return; }
    var c = clamp(src[u32(y * sw + x)], 0.0, 1.0);
    if (p.mode.y == 1) {
        c = c * c * (3.0 - 2.0 * c);
    } else if (p.mode.y == 2) {
        let d = 1.0 - c;
        c = exp(-4.0 * d * d);
    }
    if ((p.mode.z & 2) != 0) {
        c = select(0.0, pow(max(c, 0.0), p.f0.x), c > 0.0);
    }
    if ((p.mode.z & 1) != 0) { c = 1.0 - c; }
    var alpha = c;
    if ((p.mode.z & 4) == 0) { alpha = c * p.f0.y * p.f0.z; }
    let index = u32((p.rect.y + y) * p.canv.x + p.rect.x + x);
    let below = canvas[index];
    var rgb = below.xyz;
    var matte = below.w;
    let blend = p.mode.w;
    let colour = p.col.xyz;
    if (blend == 1) {
        rgb = rgb * (1.0 - alpha);
        matte = matte * (1.0 - alpha);
    } else if (blend == 0) {
        let keep = 1.0 - alpha;
        rgb = rgb * keep + alpha * colour;
        matte = matte * keep + alpha;
    } else if (blend == 2) {
        rgb = clamp(rgb + colour * alpha, vec3<f32>(0.0), vec3<f32>(1.0));
        matte = clamp(matte + alpha, 0.0, 1.0);
    } else if (blend == 3) {
        rgb = rgb * (1.0 - alpha + colour * alpha);
        matte = alpha + matte * (1.0 - alpha);
    } else {
        let mixed = vec3<f32>(blend_op(blend, colour.x, rgb.x), blend_op(blend, colour.y, rgb.y), blend_op(blend, colour.z, rgb.z));
        rgb = clamp(rgb * (1.0 - alpha) + alpha * mixed, vec3<f32>(0.0), vec3<f32>(1.0));
        matte = clamp(matte * (1.0 - alpha) + alpha * blend_op(blend, 1.0, matte), 0.0, 1.0);
    }
    canvas[index] = vec4<f32>(rgb, matte);
}
"""

_SHADERS = {"cover": _COVER, "blur_h": _blur(0), "blur_v": _blur(1), "finish": _FINISH}
PARAM_SLOT = 256


# ----------------------------------------------------------------------------------------------
# Running

def _pipeline(state, name):
    key = f"rotogpu:{name}"
    pipeline = state["pipelines"].get(key)
    if pipeline is None:
        module = state["device"].create_shader_module(code=_SHADERS[name])
        pipeline = state["device"].create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
        state["pipelines"][key] = pipeline
    return pipeline


def _buffer(state, name, nbytes, usage_names):
    """A buffer of at least ``nbytes`` kept for the next call (sizes round up so it is rarely re-made)."""
    cache = _buffers.setdefault(id(state["device"]), {})
    held = cache.get(name)
    if held is not None and held[1] >= nbytes:
        return held[0]
    if held is not None:
        held[0].destroy()
    size = max(256, ((nbytes + (1 << 20) - 1) >> 20) << 20) if nbytes > (1 << 20) else max(256, (nbytes + 255) & ~255)
    usage = state["wgpu"].BufferUsage
    flags = 0
    for usage_name in usage_names:
        flags |= getattr(usage, usage_name)
    buffer = state["device"].create_buffer(size=size, usage=flags)
    cache[name] = (buffer, size)
    return buffer


def _params(item, width, height):
    sx0, sy0, sw, sh = item["rect"]
    flags = (1 if item["invert"] else 0) | (2 if item["has_power"] else 0) | (4 if item["full_alpha"] else 0)
    window = 2 * item["radius"] + 1
    values = np.zeros(PARAM_SLOT // 4, dtype=np.int32)
    values[0:4] = (sx0, sy0, sw, sh)
    values[4:8] = (width, height, item["edge_offset"], item["edge_count"])
    values[8:12] = (item["radius"], item["falloff"], flags, item["blend"])
    floats = values.view(np.float32)
    floats[12:16] = (item["power"], item["opacity"], item["alpha"], np.float32(1.0 / window))
    floats[16:19] = item["color"]
    return values.tobytes()


def rasterise(shapes, width, height, invert=False, *, flattened_cache=None):
    """The premultiplied RGBA float32 matte of ``shapes`` drawn on the card. Raises `Unsupported` for a request the
    kernel does not cover, and any graphics error as itself; `try_rasterise` turns both into a processor fallback."""
    from . import gpu3d
    w, h = int(width), int(height)
    items, edges = plan(shapes, w, h, flattened_cache=flattened_cache)
    out = np.zeros((h, w, 4), dtype=np.float32)
    if not items:
        if invert:
            out[..., 3] = 1.0
        return out
    state = gpu3d._state(_choice)
    device, wgpu = state["device"], state["wgpu"]
    limit = int(getattr(device, "limits", {}).get("max-storage-buffer-binding-size", 128 << 20))
    if w * h * 16 > limit:
        raise Unsupported("a frame larger than the card's buffer limit")
    patch_cells = max(item["rect"][2] * item["rect"][3] for item in items)
    if patch_cells * 4 > limit or len(edges) * 32 > limit:
        raise Unsupported("a patch larger than the card's buffer limit")
    row0 = min(item["rect"][1] for item in items)
    row1 = max(item["rect"][1] + item["rect"][3] for item in items)
    started = time.perf_counter()
    with _lock:
        edge_bytes = np.ascontiguousarray(edges).tobytes() or bytes(32)
        edge_buf = _buffer(state, "edges", len(edge_bytes), ("STORAGE", "COPY_DST"))
        cover_a = _buffer(state, "cover_a", patch_cells * 4, ("STORAGE", "COPY_DST"))
        cover_b = _buffer(state, "cover_b", patch_cells * 4, ("STORAGE", "COPY_DST"))
        canvas = _buffer(state, "canvas", w * h * 16, ("STORAGE", "COPY_SRC", "COPY_DST"))
        flags = _buffer(state, "flags", 16, ("STORAGE", "COPY_SRC", "COPY_DST"))
        ring = _buffer(state, "ring", PARAM_SLOT * len(items), ("UNIFORM", "COPY_DST"))
        flag_stage = _buffer(state, "flag_stage", 16, ("COPY_DST", "MAP_READ"))
        read_bytes = (row1 - row0) * w * 16
        stage = _buffer(state, "stage", read_bytes, ("COPY_DST", "MAP_READ"))
        queue = device.queue
        queue.write_buffer(edge_buf, 0, edge_bytes)
        queue.write_buffer(flags, 0, np.zeros(4, np.uint32))
        queue.write_buffer(ring, 0, b"".join(_params(item, w, h) for item in items))
        encoder = device.create_command_encoder()
        encoder.clear_buffer(canvas, row0 * w * 16, read_bytes)

        def run(name, bindings, slot, groups):
            pipeline = _pipeline(state, name)
            entries = [{"binding": b, "resource": {"buffer": buf, "offset": 0, "size": buf.size}} for b, buf in bindings]
            entries.append({"binding": 3, "resource": {"buffer": ring, "offset": slot, "size": PARAM_SLOT}})
            group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, group)
            cp.dispatch_workgroups(*groups)
            cp.end()

        for index, item in enumerate(items):
            _x, _y, sw, sh = item["rect"]
            slot = index * PARAM_SLOT
            tiles = (-(-sw // 8), -(-sh // 8), 1)
            run("cover", [(0, edge_buf), (1, cover_a), (2, flags)], slot, (-(-sh // 64), 1, 1))
            current = cover_a
            if item["radius"] > 0:
                run("blur_h", [(0, cover_a), (1, cover_b)], slot, tiles)
                run("blur_v", [(0, cover_b), (1, cover_a)], slot, tiles)
            run("finish", [(0, current), (1, canvas)], slot, tiles)
        encoder.copy_buffer_to_buffer(flags, 0, flag_stage, 0, 16)
        encoder.copy_buffer_to_buffer(canvas, row0 * w * 16, stage, 0, read_bytes)
        queue.submit([encoder.finish()])
        flag_stage.map_sync(wgpu.MapMode.READ)
        try:
            overflow = int(np.frombuffer(flag_stage.read_mapped(0, 16, copy=False), np.uint32)[0])
        finally:
            flag_stage.unmap()
        stage.map_sync(wgpu.MapMode.READ)
        try:
            pixels = np.frombuffer(stage.read_mapped(0, read_bytes, copy=False), np.float32).reshape(row1 - row0, w, 4)
            out[row0:row1] = pixels
        finally:
            stage.unmap()
    counters["gpu_ms"] += (time.perf_counter() - started) * 1000
    if overflow:
        raise Unsupported(f"more than {MAX_CROSSINGS} crossings on one scanline")
    if not np.isfinite(out).all():
        raise Unsupported("a non-finite result")
    if invert:
        np.subtract(np.float32(1.0), out[..., 3], out=out[..., 3])
    return out


def try_rasterise(shapes, width, height, invert=False, *, flattened_cache=None):
    """`rasterise`, or None when the processor drawer must draw the matte: the path is off, the request is outside what the
    kernel covers, or the card failed (which is counted; three failures in a row turn the path off)."""
    global _failures
    if not enabled():
        return None
    try:
        result = rasterise(shapes, width, height, invert, flattened_cache=flattened_cache)
    except Unsupported as exc:
        counters["unsupported"] += 1
        counters["cpu_fallbacks"] += 1
        counters["last_reason"] = str(exc)
        return None
    except Exception as exc:            # any graphics error: the processor draws it
        _failures += 1
        counters["errors"] += 1
        counters["cpu_fallbacks"] += 1
        counters["last_error"] = f"{type(exc).__name__}: {exc}"
        return None
    _failures = 0
    counters["gpu_mattes"] += 1
    return result
