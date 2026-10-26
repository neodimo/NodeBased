"""Opt-in GPU-resident 2D island: Grade, Transform, Blur and Merge on one device.

The CPU kernels in :mod:`nodebased.imaging` stay the truth (``_grade``, ``_transform`` /
``_resample``, ``_blur`` / ``_box_blur_axis``, ``_merge_gated``). This module runs the same
formulas as WGSL compute passes on float32 storage buffers, keeps every intermediate image on
the device, and reads back once, at the island's exit, for the CPU nodes that follow.

How the pieces fit (research spike R6-1, ``briefs/proposed/R6-1-resident-four-node-spike.md``):

* The evaluator walks the graph as before. With ``Evaluator.gpu2d`` set, a Grade, Transform,
  Blur or Merge whose readers are all island nodes becomes a :class:`DeferredImage` (no pixels,
  no work); the island's exit node runs the whole chain on the device and returns a normal
  ``Raster``. Anything unsupported is reported as a CPU boundary and evaluated by the existing
  CPU kernels, so the default behaviour (``gpu2d = None``) is unchanged.
* One small ``map_*`` function per node turns the node's parameters into the shader inputs.
  New Nuke knobs on Grade, Blur or Merge later mean a change to that mapping (plus a fallback
  for the new mode), not a rewrite of the kernels.
* Windows follow the CPU rules exactly (``_filter_window``, union for Merge, ``fit`` zero-fill):
  every shader fetches an input through its data window and returns zero outside it.
* Pipelines and the parameter ring are cached per device; storage buffers are pooled and
  counted against a byte budget; the counters in :class:`Counters` are the evidence that no
  intermediate image is read back.
"""
from __future__ import annotations

import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .raster import Raster
from .tiers import Region

KINDS = ("Grade", "Transform", "Blur", "Merge")

# Blur radii whose GPU result is validated against the CPU kernel. The CPU uses a float32 shifted
# sum up to radius 4 (the algorithm the shader repeats operation for operation) and float64
# cumulative sums above it; a float32 port of those cannot match them inside the pixel gate, so
# wider radii stay an explicit CPU boundary (step B of the spike).
BLUR_VALIDATED_MAX = 4

MERGE_OPS = ("over", "under", "plus", "minus", "multiply", "screen", "max", "min", "difference",
             "divide", "mask", "stencil", "in", "out", "atop", "xor", "average", "from", "hypot")

DEFAULT_BUDGET_BYTES = 1 << 30
PARAM_SLOT = 256            # uniform bindings start at multiples of the largest alignment in use
PARAM_SLOTS = 32
WORKGROUP = 8


class Unsupported(Exception):
    """The island cannot run this node on the device; the caller evaluates it on the CPU.

    ``reason`` is a short stable phrase that is counted and reported as a CPU boundary."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class DeviceLost(Unsupported):
    """The device failed while the island ran; every lease of that epoch is invalid."""


class StaleImage(Unsupported):
    """A resident image from an earlier device epoch was used."""


# ----------------------------------------------------------------------------------------------
# Counters


@dataclass
class Counters:
    """What the island did, cumulative since the backend opened (``snapshot``/``delta`` per frame)."""
    runs: int = 0
    uploads: int = 0
    upload_bytes: int = 0
    upload_cache_hits: int = 0
    downloads: int = 0
    download_bytes: int = 0
    intermediate_readbacks: int = 0
    submissions: int = 0
    dispatches: int = 0
    allocations: int = 0
    frees: int = 0
    pool_hits: int = 0
    nonfinite_fallbacks: int = 0
    pipeline_compiles: int = 0
    shader_compile_ms: float = 0.0
    upload_ms: float = 0.0
    encode_ms: float = 0.0
    gpu_ms: float = 0.0          # submit until the device reports the work done (bounded completion)
    readback_ms: float = 0.0
    readback_map_ms: float = 0.0     # of readback_ms: the copy submission and the wait for the map (device to host transfer)
    readback_host_ms: float = 0.0    # of readback_ms: the host copy out of the mapped buffer into the Raster
    cpu_boundaries: list = field(default_factory=list)
    nodes_on_gpu: dict = field(default_factory=dict)
    live_bytes: int = 0
    peak_bytes: int = 0

    _NUMERIC = ("runs", "uploads", "upload_bytes", "upload_cache_hits", "downloads", "download_bytes",
                "intermediate_readbacks", "submissions", "dispatches", "allocations", "frees", "pool_hits",
                "nonfinite_fallbacks", "pipeline_compiles", "shader_compile_ms", "upload_ms", "encode_ms",
                "gpu_ms", "readback_ms", "readback_map_ms", "readback_host_ms")

    def snapshot(self):
        row = {name: getattr(self, name) for name in self._NUMERIC}
        row["cpu_boundaries"] = list(self.cpu_boundaries)
        row["nodes_on_gpu"] = dict(self.nodes_on_gpu)
        row["live_bytes"], row["peak_bytes"] = self.live_bytes, self.peak_bytes
        return row

    @staticmethod
    def delta(after, before):
        row = {name: after[name] - before[name] for name in Counters._NUMERIC}
        row["cpu_boundaries"] = after["cpu_boundaries"][len(before["cpu_boundaries"]):]
        row["nodes_on_gpu"] = {k: v - before["nodes_on_gpu"].get(k, 0) for k, v in after["nodes_on_gpu"].items()}
        row["live_bytes"], row["peak_bytes"] = after["live_bytes"], after["peak_bytes"]
        return row


# ----------------------------------------------------------------------------------------------
# Parameter mapping: one function per node. Everything a shader needs from a node's parameters
# passes through here, so a new knob is a change to one of these (and its fallback), nothing else.


def map_grade(p):
    """Grade's parameters as the shader sees them (``Evaluator._grade``: exposure and multiply scale
    premultiplied RGB, offset adds ``alpha * offset``; alpha is untouched)."""
    exposure, multiply, offset = p["exposure"], p["multiply"], p["offset"]
    return {
        "identity": exposure == 0.0 and multiply == 1.0 and offset == 0.0,
        "gain": float(np.float32(2.0 ** exposure)),
        "multiply": float(np.float32(multiply)), "use_multiply": multiply != 1.0,
        "offset": float(np.float32(offset)), "use_offset": offset != 0.0,
        "mix": float(p.get("mix", 1.0)),
    }


def map_blur(p):
    """Blur's parameters (``Evaluator._blur``: separable edge-padded box blur of integer radius)."""
    radius = p["radius"]
    if radius < 0.5:
        return {"radius": 0, "scale": 1.0, "mix": float(p.get("mix", 1.0))}
    r = int(round(radius))
    if r > BLUR_VALIDATED_MAX:
        raise Unsupported(f"Blur radius {r} is outside the validated GPU range 1-{BLUR_VALIDATED_MAX}")
    return {"radius": r, "scale": float(np.float32(1.0 / (2 * r + 1))), "mix": float(p.get("mix", 1.0))}


def map_merge(p):
    """Merge's parameters (``Evaluator._merge_gated``)."""
    op = p.get("operation", "over")
    if op not in MERGE_OPS:
        raise Unsupported(f"Merge operation {op!r} has no GPU kernel yet")
    return {"op": MERGE_OPS.index(op), "name": op, "mix": float(p["mix"])}


_FILTERS = {"nearest": 0, "bilinear": 1, "cubic": 2}


def _split(values):
    """float64 coordinates as an exact integer part (a float32 array) and the fraction as two 24-bit fixed-point
    limbs (float32 arrays holding integers below 2**24): the fraction is ``f1 / 2**24 + f0 / 2**48``, exact to
    3.6e-15. The shader adds a column term and a row term in integers, then rounds the sum to float32 once, which is
    what the CPU's float64 coordinate does when it casts its fraction: summing two float32 fractions would not."""
    values = np.asarray(values, np.float64)
    whole = np.floor(values)
    fraction = (values - whole) * 16777216.0
    high = np.floor(fraction)
    low = np.floor((fraction - high) * 16777216.0)
    return whole.astype(np.float32), high.astype(np.float32), low.astype(np.float32)


def _separable_weight(distance):
    """``Evaluator._resample_cubic_separable``'s Catmull-Rom weight, expression for expression: the distance is a
    float64 difference there (float32 coordinate minus an int32 tap index) and only the result is cast to float32."""
    a = -0.5
    absolute = np.abs(distance)
    square, cube = absolute * absolute, absolute * absolute * absolute
    return np.where(absolute <= 1, (a + 2) * cube - (a + 3) * square + 1,
                    a * cube - 5 * a * square + 8 * a * absolute - 4 * a).astype(np.float32)


def _separable_weights(coordinates, length, clamp):
    """The four tap weights of every coordinate along one axis, shape ``(n, 4)`` float32 (taps -1, 0, +1, +2)."""
    if clamp:
        coordinates = np.clip(coordinates, 0.0, max(0.0, length - 1.0))
    base = np.floor(coordinates).astype(np.int32)
    return np.stack([_separable_weight(coordinates - (base + offset)) for offset in (-1, 0, 1, 2)], axis=-1)


@dataclass
class TransformPlan:
    mode: str                       # "copy" (whole-pixel move) or "sample"
    shift: tuple = (0, 0)           # copy: whole pixels the image moves by
    filter: int = 2
    clamp: bool = False
    black_outside: bool = False
    zero_rows: bool = False         # rows the source does not cover are zero even with clamp (see map_transform)
    tables: np.ndarray | None = None   # float32, 6*(w+h): integer part and two fraction limbs of each axis's column and row terms;
                                       # a separable cubic adds 4*(w+h) exact tap weights (see `weighted`)
    mix: float = 1.0
    weighted: bool = False          # the cubic weights come from the table: the CPU's separable path builds them in float64


def map_transform(p, src: Region, dst: Region):
    """Transform's parameters for a source window and the output window the evaluator chose.

    Mirrors ``Evaluator._transform`` branch for branch: the identity and whole-pixel shortcuts
    (a copy of the overlapping window), the translation-only cubic path (float32 coordinates
    exactly as the CPU builds them) and the general inverse-mapped path (float64 coordinates,
    split into an integer part and a fraction so the float32 shader loses nothing).
    """
    from .imaging import Evaluator
    if p.get("motionblur", 0):
        raise Unsupported("Transform motion blur")
    filt = p["filter"]
    if filt not in _FILTERS:
        raise Unsupported(f"Transform filter {filt!r}")
    translate_x, translate_y = p["translate_x"], p["translate_y"]
    rotate, scale = p["rotate"], p["scale"]
    center_x, center_y = p["center_x"], p["center_y"]
    mix = float(p.get("mix", 1.0))
    clamp, black = bool(p.get("clamp", 0)), bool(p.get("black_outside", 0))
    uniform = p.get("scale_mode", "uniform") == "uniform"
    plain = (uniform and p.get("skew_x", 0.0) == 0.0 and p.get("skew_y", 0.0) == 0.0 and not p.get("invert", 0))
    # Identity: the CPU returns the source itself (only when the output window is the source's).
    if (translate_x == 0 and translate_y == 0 and (rotate % 360.0) == 0 and scale == 1
            and center_x == 0 and center_y == 0 and plain and not clamp and not black and dst == src):
        return TransformPlan("copy", (0, 0), _FILTERS[filt], mix=mix)
    if (rotate % 360.0 == 0 and scale == 1.0 and plain and not clamp and not black
            and float(translate_x).is_integer() and float(translate_y).is_integer()):
        return TransformPlan("copy", (int(translate_x), int(translate_y)), _FILTERS[filt], mix=mix)
    merged = dict(p, translate_x=translate_x, translate_y=translate_y, rotate=rotate, scale=scale,
                  center_x=center_x, center_y=center_y)
    inverse = np.linalg.inv(Evaluator._transform_forward_matrix(merged))
    separable_cubic = filt == "cubic" and rotate % 360.0 == 0 and scale == 1.0 and plain
    w, h = dst.width, dst.height
    zeros_w, zeros_h = np.zeros(w, np.float32), np.zeros(h, np.float32)
    zero_rows = weighted = False
    if separable_cubic:
        # The CPU builds these two vectors in float32; reproduce them with the same expressions.
        x = np.arange(w, dtype=np.float32) + dst.x + 0.5
        y = np.arange(h, dtype=np.float32) + dst.y + 0.5
        sx = x + np.float32(inverse[0, 2]) - 0.5 - src.x
        sy = y + np.float32(inverse[1, 2]) - 0.5 - src.y
        sx_ci, sx_c1, sx_c0 = _split(sx)
        sy_ri, sy_r1, sy_r0 = _split(sy)
        tables = np.concatenate([sx_ci, sx_c1, sx_c0, zeros_h, zeros_h, zeros_h,
                                 zeros_w, zeros_w, zeros_w, sy_ri, sy_r1, sy_r0,
                                 _separable_weights(sx, src.width, clamp).reshape(-1),
                                 _separable_weights(sy, src.height, clamp).reshape(-1)])
        weighted = True
        # `_resample_cubic_separable` handles a whole-pixel vertical map by cropping and zero-padding rows,
        # even with `clamp` on (the horizontal pass still clamps). The device reproduces that exactly.
        zero_rows = bool(sy.size and float(sy[0]).is_integer()
                         and np.array_equal(sy, np.float32(sy[0]) + np.arange(sy.size, dtype=np.float32)))
    else:
        gx = (np.arange(w, dtype=np.float32) + dst.x + 0.5).astype(np.float64)
        gy = (np.arange(h, dtype=np.float64) + dst.y + 0.5)
        columns_x, rows_x = _split(inverse[0, 0] * gx), _split(inverse[0, 1] * gy + inverse[0, 2] - 0.5 - src.x)
        columns_y, rows_y = _split(inverse[1, 0] * gx), _split(inverse[1, 1] * gy + inverse[1, 2] - 0.5 - src.y)
        tables = np.concatenate([*columns_x, *rows_x, *columns_y, *rows_y])
    return TransformPlan("sample", (0, 0), _FILTERS[filt], clamp, black, zero_rows,
                         np.ascontiguousarray(tables, np.float32), mix, weighted)


# ----------------------------------------------------------------------------------------------
# Windows: the same rectangles the evaluator computes for the CPU nodes.


def _gated(mask, mix):
    return mask is not None or mix != 1.0


def plan_window(kind, p, inputs):
    """``(data window, display window)`` of the node's result, or Unsupported for a case the CPU
    must handle (including every case where the CPU raises, so the error stays the CPU's)."""
    from .imaging import Evaluator
    if kind == "Merge":
        a, b = inputs[0], inputs[1]
        mask = inputs[2] if len(inputs) > 2 else None
        if a is None or b is None:
            raise Unsupported("Merge with an unwired input")
        if a.display != b.display:
            raise Unsupported("Merge inputs have different display windows")
        if mask is not None and mask.display != b.display:
            raise Unsupported("Merge mask display window differs")
        out = a.data.union(b.data)
        if out.is_empty:
            raise Unsupported("empty window")
        return out, b.display
    source = inputs[0]
    mask = inputs[1] if len(inputs) > 1 else None
    if source is None:
        raise Unsupported(f"{kind} without an input")
    if mask is not None and mask.display != source.display:
        raise Unsupported(f"{kind} mask display window differs")
    if source.data.is_empty:
        raise Unsupported("empty window")
    mix = float(p.get("mix", 1.0))
    box = Evaluator._filter_window(kind, p, source)
    out = box.union(source.data) if _gated(mask, mix) else box
    if out.is_empty:
        raise Unsupported("empty window")
    return out, source.display


# ----------------------------------------------------------------------------------------------
# Resident images and the deferred island graph


class DeviceImage:
    """A float32 RGBA image resident on the device, with the windows a ``Raster`` carries.

    It owns a counted buffer lease and never converts to numpy implicitly: ``GPU2D.download``
    is the only way back to the host."""
    __slots__ = ("buffer", "data", "display", "epoch", "nbytes", "key", "meta", "verified", "plane")

    def __init__(self, buffer, data, display, epoch, nbytes, key=None):
        self.buffer, self.data, self.display = buffer, data, display
        self.epoch, self.nbytes, self.key, self.meta = epoch, nbytes, key, None
        self.verified = False       # scanned for NaN/inf and used by a finished island
        self.plane = False          # only the alpha plane is resident (a mask input)


class DeferredImage:
    """An island node staged by the evaluator: its inputs, parameters and result windows. It has no
    pixels and has done no work; ``GPU2D.run`` executes the chain that ends at an exit node."""
    __slots__ = ("kind", "params", "inputs", "keys", "data", "display", "meta", "node_id", "plan")

    def __init__(self, kind, params, inputs, keys, data, display, node_id=None):
        self.kind, self.params, self.inputs, self.keys = kind, params, list(inputs), list(keys)
        self.data, self.display, self.meta, self.node_id, self.plan = data, display, None, node_id, None

    @property
    def shape(self):
        return (self.data.height, self.data.width, 4)


def inherited_meta(kind, inputs):
    """What ``imaging._inherited_metadata`` gives the node's result (Merge: B first)."""
    from .core import MERGE_LIKE_KINDS
    order = list(inputs[1::-1] if kind in MERGE_LIKE_KINDS else inputs)
    return next((m for m in (getattr(image, "meta", None) for image in order) if m is not None), None)


# ----------------------------------------------------------------------------------------------
# WGSL


def _fetch_alpha(name, region):
    """Like _fetch for a mask uploaded as its alpha plane only (one float per pixel): every gate in this
    module reads nothing but the mask's alpha, so the other three planes never cross the bus."""
    return f"""
fn fetch_{name}(ax: i32, ay: i32) -> f32 {{
    let lx = ax - p.{region}.x; let ly = ay - p.{region}.y;
    if (lx < 0 || ly < 0 || lx >= p.{region}.z || ly >= p.{region}.w) {{ return 0.0; }}
    return {name}[u32(ly) * u32(p.{region}.z) + u32(lx)];
}}
"""


def _fetch(name, region):
    return f"""
fn fetch_{name}(ax: i32, ay: i32) -> vec4<f32> {{
    let lx = ax - p.{region}.x; let ly = ay - p.{region}.y;
    if (lx < 0 || ly < 0 || lx >= p.{region}.z || ly >= p.{region}.w) {{ return vec4<f32>(0.0); }}
    return {name}[u32(ly) * u32(p.{region}.z) + u32(lx)];
}}
"""


_PRELUDE = """
struct Params {
    out_r: vec4<i32>, r0: vec4<i32>, r1: vec4<i32>, rm: vec4<i32>,
    i0: vec4<i32>, f0: vec4<f32>, f1: vec4<f32>,
};
"""

# Mask and mix after a filter, exactly Evaluator._apply_mask_mix: gate = mask.a * mix (a scalar mix
# when no mask is wired); no gate at all when mix is 1 and there is no mask. i1.x of the block
# below is "has mask", i1.y "needs blending": both live in i0.x / i0.y for every filter shader.
_GATE = """
fn gated(source: vec4<f32>, filtered: vec4<f32>, ax: i32, ay: i32) -> vec4<f32> {
    if (p.i0.x != 0) {
        let g = fetch_mask(ax, ay) * p.f1.w;
        return filtered * g + source * (1.0 - g);
    }
    if (p.i0.y != 0) {
        let g = p.f1.w;
        return filtered * g + source * (1.0 - g);
    }
    return filtered;
}
"""

_BINDINGS_3 = """
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
"""

_MAIN_HEAD = """
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (i32(gid.x) >= p.out_r.z || i32(gid.y) >= p.out_r.w) { return; }
    let ax = p.out_r.x + i32(gid.x); let ay = p.out_r.y + i32(gid.y);
    let oi = gid.y * u32(p.out_r.z) + gid.x;
"""

# i0 = (has_mask, blend, identity/aux, aux2); f0 = (gain, multiply, offset, _); f1 = (_, _, _, mix)
_GRADE = _PRELUDE + _BINDINGS_3 + _fetch("src", "r0") + _fetch_alpha("mask", "rm") + _GATE + _MAIN_HEAD + """
    let s = fetch_src(ax, ay);
    var f = s;
    if (p.i0.z == 0) {
        var rgb = s.xyz * p.f0.x;
        if ((p.i0.w & 1) != 0) { rgb = rgb * p.f0.y; }
        if ((p.i0.w & 2) != 0) { rgb = rgb + s.w * p.f0.z; }
        f = vec4<f32>(rgb, s.w);
    }
    out[oi] = gated(s, f, ax, ay);
}
"""

# Blur: horizontal pass src -> out (no gating), vertical pass tmp(src slot) -> out with gating
# against the original image (bound as `orig`). i0.z = radius, f0.x = 1/(2r+1).
_BLUR_H = _PRELUDE + """
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = p.r0.z; let h = p.r0.w;
    if (i32(gid.x) >= w || i32(gid.y) >= h) { return; }
    let x = i32(gid.x); let y = i32(gid.y);
    let r = p.i0.z;
    let row = u32(y) * u32(w);
    var acc = src[row + u32(clamp(x - r, 0, w - 1))];
    for (var k = 1; k <= 2 * r; k = k + 1) {
        acc = acc + src[row + u32(clamp(x - r + k, 0, w - 1))];
    }
    out[row + u32(x)] = acc * p.f0.x;
}
"""

_BLUR_V = _PRELUDE + """
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read> orig: array<vec4<f32>>;
""" + _fetch("orig", "r1") + _fetch_alpha("mask", "rm") + _GATE + """
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = p.r0.z; let h = p.r0.w;
    if (i32(gid.x) >= w || i32(gid.y) >= h) { return; }
    let x = i32(gid.x); let y = i32(gid.y);
    let r = p.i0.z;
    var acc = src[u32(clamp(y - r, 0, h - 1)) * u32(w) + u32(x)];
    for (var k = 1; k <= 2 * r; k = k + 1) {
        acc = acc + src[u32(clamp(y - r + k, 0, h - 1)) * u32(w) + u32(x)];
    }
    let blurred = acc * p.f0.x;
    let ax = p.out_r.x + x; let ay = p.out_r.y + y;
    out[u32(y) * u32(w) + u32(x)] = gated(fetch_orig(ax, ay), blurred, ax, ay);
}
"""

# Transform. i0 = (has_mask, blend, mode, flags) with flags bit 1 clamp, bit 2 black_outside;
# i1 lives in rm? No: filter and shift ride in r1 = (filter, shift_x, shift_y, _).
_TRANSFORM = _PRELUDE + _BINDINGS_3 + """
@group(0) @binding(4) var<storage, read> tbl: array<f32>;
""" + _fetch("src", "r0") + _fetch_alpha("mask", "rm") + _GATE + """
fn tap(ix: i32, iy: i32, clamped: bool) -> vec4<f32> {
    let sw = p.r0.z; let sh = p.r0.w;
    if (clamped) {
        return src[u32(clamp(iy, 0, sh - 1)) * u32(sw) + u32(clamp(ix, 0, sw - 1))];
    }
    if (ix < 0 || iy < 0 || ix >= sw || iy >= sh) { return vec4<f32>(0.0); }
    return src[u32(iy) * u32(sw) + u32(ix)];
}
// The CPU rounds every product on its own before adding it. A GPU compiler may fuse a multiply with the add that
// follows it (one rounding instead of two), which shows next to an extreme HDR value. Passing each product through
// an AND with an all-ones mask taken from the parameters (r1.w = -1, opaque to the compiler) keeps the two roundings.
fn rounded(x: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(x) & vec4<u32>(bitcast<u32>(p.r1.w)));
}
fn rounded1(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) & bitcast<u32>(p.r1.w));
}
fn cubic_w(t: f32) -> f32 {
    // Every intermediate goes through `rounded1`: the weight is a difference of terms several times larger than
    // itself, so a compiler that reorders or fuses the polynomial (the Radeon and software drivers do) moves it by
    // up to 1e-6, which an extreme HDR neighbour turns into a visible error. The order below is the CPU's.
    let a = -0.5;
    let at = abs(t);
    let at2 = rounded1(at * at);
    let at3 = rounded1(at2 * at);
    if (at <= 1.0) {
        let u = rounded1(rounded1((a + 2.0) * at3) - rounded1((a + 3.0) * at2));
        return rounded1(u + 1.0);
    }
    let u = rounded1(rounded1(a * at3) - rounded1(5.0 * a * at2));
    let v = rounded1(u + rounded1(8.0 * a * at));
    return rounded1(v - 4.0 * a);
}
fn cubic_weights(f: f32) -> vec4<f32> {
    return vec4<f32>(cubic_w(f + 1.0), cubic_w(f), cubic_w(f - 1.0), cubic_w(f - 2.0));
}
// One source coordinate = column term + row term, each an exact integer part and a 48-bit fixed-point fraction.
// Returns the integer part and the fraction rounded once to float32.
struct Coord { x: i32, y: f32 };
fn coord_int(base: u32, w: u32, h: u32, ix: u32, iy: u32) -> Coord {
    let c_i = base + ix; let c_1 = base + w + ix; let c_0 = base + 2u * w + ix;
    let r_i = base + 3u * w + iy; let r_1 = base + 3u * w + h + iy; let r_0 = base + 3u * w + 2u * h + iy;
    var low = u32(tbl[c_0]) + u32(tbl[r_0]);
    let carry_low = low >> 24u;
    low = low & 0xFFFFFFu;
    var high = u32(tbl[c_1]) + u32(tbl[r_1]) + carry_low;
    let carry_high = high >> 24u;
    high = high & 0xFFFFFFu;
    var result: Coord;
    result.x = i32(tbl[c_i] + tbl[r_i] + f32(carry_high));
    result.y = f32(high) * 5.9604644775390625e-8 + f32(low) * 3.552713678800501e-15;
    return result;
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (i32(gid.x) >= p.out_r.z || i32(gid.y) >= p.out_r.w) { return; }
    let ax = p.out_r.x + i32(gid.x); let ay = p.out_r.y + i32(gid.y);
    let oi = gid.y * u32(p.out_r.z) + gid.x;
    let s = fetch_src(ax, ay);
    var f = vec4<f32>(0.0);
    if (p.i0.z == 0) {
        f = fetch_src(ax - p.r1.y, ay - p.r1.z);
    } else {
        let w = u32(p.out_r.z); let h = u32(p.out_r.w);
        let ix = gid.x; let iy = gid.y;
        // tbl: per axis (x, then y) the column terms (w entries each: integer part, high limb, low limb), then the
        // row terms (h entries each). A coordinate is column + row; its fraction is summed as integers.
        let ax_len = 3u * (w + h);
        let sx_i = coord_int(0u, w, h, ix, iy);
        let sy_i = coord_int(ax_len, w, h, ix, iy);
        var xi = sx_i.x; var xf = sx_i.y;
        var yi = sy_i.x; var yf = sy_i.y;
        let sw = p.r0.z; let sh = p.r0.w;
        let clamped = (p.i0.w & 1) != 0;
        let black = (p.i0.w & 2) != 0;
        let zero_row = (p.i0.w & 4) != 0 && (yi < 0 || yi >= sh);
        var outside = false;
        if (black) {
            outside = xi < -1 || (xi == -1 && xf < 0.5) || xi >= sw || (xi == sw - 1 && xf > 0.5)
                   || yi < -1 || (yi == -1 && yf < 0.5) || yi >= sh || (yi == sh - 1 && yf > 0.5);
        }
        if (clamped) {
            if (xi < 0) { xi = 0; xf = 0.0; } else if (xi >= sw - 1) { xi = max(sw - 1, 0); xf = 0.0; }
            if (yi < 0) { yi = 0; yf = 0.0; } else if (yi >= sh - 1) { yi = max(sh - 1, 0); yf = 0.0; }
        }
        let filt = p.r1.x;
        if (zero_row) {
            f = vec4<f32>(0.0);
        } else if (filt == 0) {
            let nx = xi + select(0, 1, xf >= 0.5);
            let ny = yi + select(0, 1, yf >= 0.5);
            f = tap(nx, ny, clamped);
        } else if (filt == 1) {
            let wx0 = 1.0 - xf; let wy0 = 1.0 - yf;
            f = rounded(tap(xi, yi, clamped) * wx0 * wy0) + rounded(tap(xi + 1, yi, clamped) * xf * wy0)
              + rounded(tap(xi, yi + 1, clamped) * wx0 * yf) + rounded(tap(xi + 1, yi + 1, clamped) * xf * yf);
        } else {
            var wx = cubic_weights(xf); var wy = cubic_weights(yf);
            if ((p.i0.w & 8) != 0) {
                let wb = 6u * (w + h);
                wx = vec4<f32>(tbl[wb + 4u * ix], tbl[wb + 4u * ix + 1u], tbl[wb + 4u * ix + 2u], tbl[wb + 4u * ix + 3u]);
                let wr = wb + 4u * w + 4u * iy;
                wy = vec4<f32>(tbl[wr], tbl[wr + 1u], tbl[wr + 2u], tbl[wr + 3u]);
            }
            var total = vec4<f32>(0.0);
            for (var dy = 0; dy < 4; dy = dy + 1) {
                let wyv = wy[dy];
                if (wyv != 0.0) {
                    var row = vec4<f32>(0.0);
                    for (var dx = 0; dx < 4; dx = dx + 1) {
                        let wxv = wx[dx];
                        if (wxv != 0.0) { row = row + rounded(tap(xi - 1 + dx, yi - 1 + dy, clamped) * wxv); }
                    }
                    total = total + rounded(row * wyv);
                }
            }
            f = total;
        }
        if (outside) { f = vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    }
    out[oi] = gated(s, f, ax, ay);
}
"""


def _merge_op_wgsl():
    return """
fn merge_op(op: i32, a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let aa = a.w; let ba = b.w;
    switch (op) {
        case 0: { return a + b * (1.0 - aa); }
        case 1: { return b + a * (1.0 - ba); }
        case 2: { return a + b; }
        case 3: { return a - b; }
        case 4: { return a * b; }
        case 5: { return a + b - a * b; }
        case 6: { return max(a, b); }
        case 7: { return min(a, b); }
        case 8: { return abs(a - b); }
        case 9: {
            let usable = abs(b) > vec4<f32>(1e-6);
            let safe = select(vec4<f32>(1.0), b, usable);
            return select(vec4<f32>(0.0), a / safe, usable);
        }
        case 10: { return b * aa; }
        case 11: { return b * (1.0 - aa); }
        case 12: { return a * ba; }
        case 13: { return a * (1.0 - ba); }
        case 14: { return a * ba + b * (1.0 - aa); }
        case 15: { return a * (1.0 - ba) + b * (1.0 - aa); }
        case 16: { return (a + b) * 0.5; }
        case 17: { return b - a; }
        default: { return sqrt(a * a + b * b); }
    }
}
"""


# Merge: A=src, B=b, mask. r0=A window, r1=B window, rm=mask window. i0 = (has_mask, _, op, _), f1.w = mix.
_MERGE = _PRELUDE + """
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read> b_img: array<vec4<f32>>;
""" + _fetch("src", "r0") + _fetch_alpha("mask", "rm") + """
fn fetch_b(ax: i32, ay: i32) -> vec4<f32> {
    let lx = ax - p.r1.x; let ly = ay - p.r1.y;
    if (lx < 0 || ly < 0 || lx >= p.r1.z || ly >= p.r1.w) { return vec4<f32>(0.0); }
    return b_img[u32(ly) * u32(p.r1.z) + u32(lx)];
}
""" + _merge_op_wgsl() + _MAIN_HEAD + """
    let a = fetch_src(ax, ay);
    let b = fetch_b(ax, ay);
    let op = p.i0.z;
    let mix = p.f1.w;
    let has_mask = p.i0.x != 0;
    if (op == 0 && !has_mask && mix == 1.0) {
        out[oi] = b * (1.0 - a.w) + a;
        return;
    }
    var gate = mix;
    if (has_mask) {
        let m = fetch_mask(ax, ay);
        gate = select(mix * m, m, mix == 1.0);
    }
    if (op == 0) {
        let foreground = a * gate;
        let remaining = 1.0 - a.w * gate;
        out[oi] = foreground + b * remaining;
        return;
    }
    let full = merge_op(op, a, b);
    out[oi] = full * gate + b * (1.0 - gate);
}
"""

# Non-finite scan: sets bit 0 of flags[0] when any component of `src` is NaN or infinite.
_SCAN = _PRELUDE + """
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(p.i0.x);
    if (gid.x >= n) { return; }
    let bits = bitcast<vec4<u32>>(src[gid.x]);
    let bad = (bits & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u);
    if (any(bad)) { atomicOr(&flags[0], 1u); }
}
"""

_SCAN_PLANE = _PRELUDE + """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u32(p.i0.x)) { return; }
    if ((bitcast<u32>(src[gid.x]) & 0x7f800000u) == 0x7f800000u) { atomicOr(&flags[0], 1u); }
}
"""

_SHADERS = {"scan_plane": _SCAN_PLANE, "grade": _GRADE, "blur_h": _BLUR_H, "blur_v": _BLUR_V, "transform": _TRANSFORM,
            "merge": _MERGE, "scan": _SCAN}


def _params(out_r=(0, 0, 0, 0), r0=(0, 0, 0, 0), r1=(0, 0, 0, 0), rm=(0, 0, 0, 0), i0=(0, 0, 0, 0),
            f0=(0.0, 0.0, 0.0, 0.0), f1=(0.0, 0.0, 0.0, 0.0)):
    """One 112-byte parameter block: five int4 rows (window, three input windows, integers), two float4 rows."""
    return struct.pack("<20i", *out_r, *r0, *r1, *rm, *i0) + struct.pack("<8f", *f0, *f1)


def _rect(region):
    return (region.x, region.y, region.width, region.height)


# ----------------------------------------------------------------------------------------------
# The backend


class GPU2D:
    """Executes staged island chains on one wgpu device. One owner thread at a time (the
    evaluator's worker); ``run`` is serialised by a lock."""

    open_error = None

    def __init__(self, state, budget_bytes=DEFAULT_BUDGET_BYTES, label="default"):
        self.state = state
        self.wgpu = state["wgpu"]
        self.device = state["device"]
        self.label = label
        self.budget = int(budget_bytes)
        self.counters = Counters()
        self.epoch = 0
        self.lost = False
        self.adapter = dict(state.get("info", {}))
        self._pool = {}                       # nbytes -> [buffer]
        self._inputs = OrderedDict()          # content key -> DeviceImage (uploaded sources)
        self._input_bytes = 0
        self._ring = None
        self._flags = None
        self._zero = None
        self._staging = {}
        self._lock = threading.Lock()
        limits = getattr(self.device, "limits", None) or {}
        self.max_binding = int(limits.get("max-storage-buffer-binding-size", 128 << 20))
        self._pipelines = state["pipelines"]
        self._live = 0

    # -- opening -----------------------------------------------------------------------------

    @classmethod
    def open(cls, choice="default", budget_bytes=DEFAULT_BUDGET_BYTES):
        """A backend on the named adapter, or None with ``open_error`` set when there is none."""
        from . import gpu3d
        try:
            state = gpu3d._state(choice)
        except Exception as exc:        # no device is a normal answer; the island simply stays off
            cls.open_error = f"{type(exc).__name__}: {exc}"
            return None
        cls.open_error = None
        return cls(state, budget_bytes, choice)

    def describe(self):
        info = self.adapter
        return {"device": info.get("device"), "backend": info.get("backend_type"),
                "adapter_type": info.get("adapter_type"), "driver": info.get("driver"),
                "driver_info": info.get("driver_info"), "budget_bytes": self.budget,
                "max_binding_bytes": self.max_binding, "label": self.label}

    # -- memory ------------------------------------------------------------------------------

    def _count_live(self, delta):
        self._live += delta
        self.counters.live_bytes = self._live
        self.counters.peak_bytes = max(self.counters.peak_bytes, self._live)

    def _reserve(self, nbytes):
        if nbytes > self.max_binding:
            raise Unsupported(f"image of {nbytes >> 20} MiB exceeds the device's storage binding limit")
        if self._live + nbytes > self.budget:
            self._trim_pool(self._live + nbytes - self.budget)
        if self._live + nbytes > self.budget:
            raise Unsupported(f"over the {self.budget >> 20} MiB 2D device budget")

    def _trim_pool(self, need=None):
        """Destroy pooled buffers (largest first) until ``need`` bytes are freed, or all of them."""
        for size in sorted(self._pool, reverse=True):
            bucket = self._pool[size]
            while bucket and (need is None or need > 0):
                bucket.pop().destroy()
                self.counters.frees += 1
                self._count_live(-size)
                if need is not None:
                    need -= size
        self._pool = {size: bucket for size, bucket in self._pool.items() if bucket}

    @staticmethod
    def _bucket(nbytes):
        """Buffer sizes round up (to 1 MiB above 1 MiB) so frames whose windows differ by a few columns
        reuse the same pooled buffers instead of growing the pool by one size per frame."""
        if nbytes <= (1 << 20):
            return (nbytes + 15) & ~15
        return ((nbytes + (1 << 20) - 1) >> 20) << 20

    def _trim_pool_to(self, limit):
        pooled = sum(size * len(bucket) for size, bucket in self._pool.items())
        if pooled > limit:
            self._trim_pool(pooled - limit)

    def _acquire(self, nbytes):
        nbytes = self._bucket(nbytes)
        bucket = self._pool.get(nbytes)
        if bucket:
            self.counters.pool_hits += 1
            return bucket.pop(), nbytes
        self._reserve(nbytes)
        usage = self.wgpu.BufferUsage
        buffer = self.device.create_buffer(size=nbytes, usage=usage.STORAGE | usage.COPY_SRC | usage.COPY_DST)
        self.counters.allocations += 1
        self._count_live(nbytes)
        return buffer, nbytes

    def _release(self, buffer, nbytes):
        self._pool.setdefault(nbytes, []).append(buffer)

    def _fixed_buffers(self):
        if self._ring is None:
            usage = self.wgpu.BufferUsage
            self._reserve(PARAM_SLOT * PARAM_SLOTS + 32)
            self._ring = self.device.create_buffer(size=PARAM_SLOT * PARAM_SLOTS, usage=usage.UNIFORM | usage.COPY_DST)
            self._flags = self.device.create_buffer(size=16, usage=usage.STORAGE | usage.COPY_SRC | usage.COPY_DST)
            self._zero = self.device.create_buffer(size=16, usage=usage.STORAGE | usage.COPY_DST)
            self.device.queue.write_buffer(self._zero, 0, np.zeros(4, np.float32))
            self.counters.allocations += 3
            self._count_live(PARAM_SLOT * PARAM_SLOTS + 32)

    def release_all(self):
        """Drop every cached input and pooled buffer (cancel, device change, shutdown)."""
        for image in list(self._inputs.values()):
            self._release(image.buffer, image.nbytes)
        self._inputs.clear()
        self._input_bytes = 0
        self._trim_pool()
        for name, size in (("_ring", PARAM_SLOT * PARAM_SLOTS), ("_flags", 16), ("_zero", 16)):
            buffer = getattr(self, name)
            if buffer is not None:
                buffer.destroy()
                self._count_live(-size)
                setattr(self, name, None)
        for buffer in self._staging.values():
            buffer.destroy()
        self._staging.clear()

    def invalidate(self, reason):
        """Device loss: every resident image is stale; the whole affected frame reruns on the CPU."""
        self.lost = True
        self.epoch += 1
        self._pool.clear()
        self._inputs.clear()
        self._input_bytes = 0
        self._ring = self._flags = self._zero = None
        self._staging.clear()
        self._live = 0
        self.counters.live_bytes = 0
        self.counters.cpu_boundaries.append(f"device lost: {reason}")

    # -- pipelines ---------------------------------------------------------------------------

    def _pipeline(self, name):
        key = f"gpu2d:{name}"
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            started = time.perf_counter()
            module = self.device.create_shader_module(code=_SHADERS[name])
            pipeline = self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
            self._pipelines[key] = pipeline
            self.counters.pipeline_compiles += 1
            self.counters.shader_compile_ms += (time.perf_counter() - started) * 1000
        return pipeline

    def warm(self):
        """Compile every pipeline now (the cold shader setup the spike reports separately), in ms."""
        started = time.perf_counter()
        for name in _SHADERS:
            self._pipeline(name)
        return (time.perf_counter() - started) * 1000

    # -- uploads -----------------------------------------------------------------------------

    def upload(self, raster, key=None, plane=False):
        """A resident copy of ``raster`` (kept for reuse under ``key`` while the cache budget allows).

        ``plane`` uploads only the alpha channel, one float per pixel: that is all a mask input is ever read for."""
        if key is not None:
            key = (key, plane)
        if key is not None and key in self._inputs:
            image = self._inputs.pop(key)
            if image.epoch == self.epoch:
                self._inputs[key] = image
                self.counters.upload_cache_hits += 1
                return image
            self._input_bytes -= image.nbytes
        started = time.perf_counter()
        pixels = np.ascontiguousarray(raster.pixels[..., 3] if plane else raster.pixels, dtype=np.float32)
        buffer, nbytes = self._acquire(pixels.nbytes)
        self.device.queue.write_buffer(buffer, 0, pixels)
        self.counters.uploads += 1
        self.counters.upload_bytes += pixels.nbytes
        self.counters.upload_ms += (time.perf_counter() - started) * 1000
        image = DeviceImage(buffer, raster.data, raster.display, self.epoch, nbytes, key)
        image.meta = getattr(raster, "meta", None)
        image.plane = plane
        if key is not None:
            self._inputs[key] = image
            self._input_bytes += nbytes
        return image

    # -- staging -----------------------------------------------------------------------------

    def stage(self, kind, params, images, keys=(), node_id=None):
        """Stage one island node. Returns a DeferredImage, or raises Unsupported for the CPU path.

        ``images`` are the node's inputs in slot order: Rasters, DeferredImages or None."""
        if self.lost:
            raise Unsupported("device lost; GPU 2D stays off until reopened")
        if kind == "Grade":
            map_grade(params)
        elif kind == "Blur":
            map_blur(params)
        elif kind == "Merge":
            map_merge(params)
        elif kind != "Transform":
            raise Unsupported(f"{kind} is not an island node")
        data, display = plan_window(kind, params, images)
        if data.width * data.height * 16 > min(self.max_binding, self.budget // 4):
            raise Unsupported("window larger than the device budget allows")
        keys = list(keys) + [None] * (len(images) - len(keys))
        node = DeferredImage(kind, params, images, keys, data, display, node_id)
        node.meta = inherited_meta(kind, images)
        if kind == "Transform":      # build the plan now so an unsupported mode is reported at staging
            node.plan = map_transform(params, images[0].data, data)
        return node

    # -- execution ---------------------------------------------------------------------------

    def run(self, root, cancel=None):
        """Execute the chain ending at ``root`` on the device and return one ``Raster`` (the only
        readback). Raises Unsupported or DeviceLost for the caller's CPU fallback."""
        with self._lock:
            self.counters.runs += 1
            try:
                return self._execute(root, cancel)
            except Unsupported:
                raise
            except Exception as exc:
                if type(exc).__name__ == "Cancelled":
                    raise
                self.invalidate(f"{type(exc).__name__}: {exc}")
                raise DeviceLost(str(exc)) from exc

    @staticmethod
    def _is_mask(kind, slot):
        """Mask inputs (read only for their alpha): Merge's third input, the second of the filters."""
        return slot == (2 if kind == "Merge" else 1)

    @staticmethod
    def _check(cancel):
        if cancel is not None and cancel.is_set():
            from .cancellation import Cancelled
            raise Cancelled()

    def _execute(self, root, cancel):
        counters, device, wgpu = self.counters, self.device, self.wgpu
        self._check(cancel)
        self._fixed_buffers()
        encode_started = time.perf_counter()
        upload_before = counters.upload_ms
        slots, slot_data, passes = [], [], []
        held = []                          # upload scratch and outputs returned to the pool at the end

        def add_params(blob):
            if len(slots) >= PARAM_SLOTS:
                raise Unsupported("island has too many passes")
            slots.append(len(slots) * PARAM_SLOT)
            slot_data.append(blob)
            return slots[-1]

        order, seen = [], set()

        def visit(node):
            if isinstance(node, DeferredImage) and id(node) not in seen:
                seen.add(id(node))
                for item in node.inputs:
                    visit(item)
                order.append(node)
        visit(root)
        if not order:
            raise Unsupported("nothing staged")
        keymap, last_use = {}, {}
        for index, node in enumerate(order):
            for slot, (item, key) in enumerate(zip(node.inputs, node.keys)):
                if item is not None:
                    last_use[id(item)] = index
                    if isinstance(item, Raster) and key is not None:
                        keymap[(id(item), self._is_mask(node.kind, slot))] = key
        resident, fresh_inputs, tables = {}, [], {}
        outputs = []
        try:
            self._evict_inputs()
            # Every host write (source uploads, transform tables) happens first, before any pass
            # output is allocated: a write is applied before the whole command buffer runs, so it
            # must never land in a buffer an earlier pass of this island has already released.
            for node in order:
                for slot, item in enumerate(node.inputs):
                    plane = self._is_mask(node.kind, slot)
                    if item is not None and not isinstance(item, DeferredImage) and (id(item), plane) not in resident:
                        image = self.upload(item, keymap.get((id(item), plane)), plane)
                        if image.epoch != self.epoch:
                            raise StaleImage("resident image from an earlier device epoch")
                        resident[(id(item), plane)] = image
                        if image.key is None:
                            held.append((image.buffer, image.nbytes))
                        if not image.verified:
                            self._scan(image, add_params, passes)
                            fresh_inputs.append(image)
                if node.plan is not None and node.plan.tables is not None:
                    tbl, tbl_bytes = self._acquire(node.plan.tables.nbytes)
                    device.queue.write_buffer(tbl, 0, node.plan.tables)
                    counters.uploads += 1
                    counters.upload_bytes += node.plan.tables.nbytes
                    held.append((tbl, tbl_bytes))
                    tables[id(node)] = (tbl, tbl_bytes)

            for index, node in enumerate(order):
                self._check(cancel)
                ins = [None if item is None else resident[id(item) if isinstance(item, DeferredImage) else (id(item), self._is_mask(node.kind, slot))]
                       for slot, item in enumerate(node.inputs)]
                out_buffer, out_bytes = self._acquire(node.data.width * node.data.height * 16)
                out = DeviceImage(out_buffer, node.data, node.display, self.epoch, out_bytes)
                resident[id(node)] = out
                outputs.append(out)
                self._encode(node, ins, out, add_params, passes, tables.get(id(node)))
                counters.nodes_on_gpu[node.kind] = counters.nodes_on_gpu.get(node.kind, 0) + 1
                for item in node.inputs:       # a deferred input nobody reads again goes back to the pool
                    if isinstance(item, DeferredImage) and last_use.get(id(item)) == index:
                        self._free(resident[id(item)], outputs)
            final = resident[id(root)]
            self._scan(final, add_params, passes)
            # encode_ms is the host time spent building the passes; the uploads inside it are counted in upload_ms.
            counters.encode_ms += (time.perf_counter() - encode_started) * 1000 - (counters.upload_ms - upload_before)

            blob = bytearray(PARAM_SLOT * len(slots))
            for slot, data in zip(slots, slot_data):
                blob[slot:slot + len(data)] = data
            device.queue.write_buffer(self._ring, 0, bytes(blob))
            device.queue.write_buffer(self._flags, 0, np.zeros(4, np.uint32))
            encoder = device.create_command_encoder()
            for pipeline, bindings, slot, groups, uniform_binding in passes:
                entries = [{"binding": b, "resource": {"buffer": buf, "offset": 0, "size": size}} for b, buf, size in bindings]
                entries.append({"binding": uniform_binding, "resource": {"buffer": self._ring, "offset": slot, "size": PARAM_SLOT}})
                group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
                cp = encoder.begin_compute_pass()
                cp.set_pipeline(pipeline)
                cp.set_bind_group(0, group)
                cp.dispatch_workgroups(*groups)
                cp.end()
                counters.dispatches += 1
            out_bytes = final.data.width * final.data.height * 16
            flag_staging = self._staging_buffer(16)
            encoder.copy_buffer_to_buffer(self._flags, 0, flag_staging, 0, 16)
            self._check(cancel)
            submitted = time.perf_counter()
            device.queue.submit([encoder.finish()])
            counters.submissions += 1
            # Mapping the small flags buffer returns once every pass has finished: that is the
            # completion time (this wgpu build's queue work-done callback is not usable).
            flag_staging.map_sync(wgpu.MapMode.READ)
            try:
                flag_word = struct.unpack_from("<I", flag_staging.read_mapped(0, 16, copy=False), 0)[0]
            finally:
                flag_staging.unmap()
            gpu_done = time.perf_counter()
            pixels = None
            mapped_at = gpu_done
            if not flag_word & 1:
                staging = self._staging_buffer(out_bytes)
                copy = device.create_command_encoder()
                copy.copy_buffer_to_buffer(final.buffer, 0, staging, 0, out_bytes)
                device.queue.submit([copy.finish()])
                counters.submissions += 1
                staging.map_sync(wgpu.MapMode.READ)
                mapped_at = time.perf_counter()
                try:
                    mapped = staging.read_mapped(0, out_bytes, copy=False)
                    pixels = np.frombuffer(mapped, np.float32, out_bytes // 4).reshape(final.data.height, final.data.width, 4).copy()
                finally:
                    staging.unmap()
            done = time.perf_counter()
        finally:
            for buffer, nbytes in held:
                self._release(buffer, nbytes)
            for image in list(outputs):
                self._free(image, outputs)
        self._trim_pool_to(self.budget // 2)
        counters.gpu_ms += (gpu_done - submitted) * 1000
        counters.readback_ms += (done - gpu_done) * 1000
        counters.readback_map_ms += (mapped_at - gpu_done) * 1000
        counters.readback_host_ms += (done - mapped_at) * 1000
        counters.downloads += 1
        counters.download_bytes += (out_bytes if pixels is not None else 0) + 16
        if pixels is None:
            counters.nonfinite_fallbacks += 1
            counters.cpu_boundaries.append("non-finite pixels: whole frame recomputed on the CPU")
            for image in list(self._inputs.values()):     # a poisoned source must not stay cached
                self._release(image.buffer, image.nbytes)
            self._inputs.clear()
            self._input_bytes = 0
            raise Unsupported("non-finite pixels")
        for image in fresh_inputs:
            image.verified = True
        raster = Raster(pixels, final.data, final.display)
        raster.meta = root.meta
        return raster

    def _free(self, image, outputs):
        """Return one pass output to the pool exactly once."""
        if image in outputs:
            outputs.remove(image)
            self._release(image.buffer, image.nbytes)

    def _evict_inputs(self):
        """Keep the resident source cache within a quarter of the budget (only between islands)."""
        while self._input_bytes > self.budget // 4 and len(self._inputs) > 1:
            _key, old = self._inputs.popitem(last=False)
            self._input_bytes -= old.nbytes
            self._release(old.buffer, old.nbytes)

    def _staging_buffer(self, nbytes):
        """A mappable readback buffer of at least ``nbytes`` (sizes round up to 4 MiB so a window
        that changes a few pixels between frames keeps reusing one)."""
        size = max(16, -(-nbytes // (4 << 20)) * (4 << 20)) if nbytes > 16 else 16
        buffer = self._staging.get(size)
        if buffer is None:
            while len(self._staging) >= 3:
                oldest = next(iter(self._staging))
                self._staging.pop(oldest).destroy()
            usage = self.wgpu.BufferUsage
            buffer = self.device.create_buffer(size=size, usage=usage.COPY_DST | usage.MAP_READ)
            self._staging[size] = buffer
            self.counters.allocations += 1
        return buffer

    def _scan(self, image, add_params, passes):
        n = image.data.width * image.data.height
        slot = add_params(_params(i0=(n, 0, 0, 0)))
        passes.append((self._pipeline("scan_plane" if image.plane else "scan"), [(0, image.buffer, image.nbytes), (1, self._flags, 16)],
                       slot, (-(-n // 64), 1, 1), 2))

    def _encode(self, node, ins, out, add_params, passes, table):
        kind, p = node.kind, node.params
        dst = node.data
        groups = (-(-dst.width // WORKGROUP), -(-dst.height // WORKGROUP), 1)
        zero = self._zero

        def mask_info(image):
            if image is None:
                return zero, 16, (0, 0, 0, 0), 0
            return image.buffer, image.nbytes, _rect(image.data), 1

        if kind == "Grade":
            m = map_grade(p)
            mbuf, mbytes, mrect, has_mask = mask_info(ins[1] if len(ins) > 1 else None)
            blend = 1 if (has_mask or m["mix"] != 1.0) else 0
            aux = (1 if m["use_multiply"] else 0) | (2 if m["use_offset"] else 0)
            slot = add_params(_params(_rect(dst), _rect(ins[0].data), (0, 0, 0, 0), mrect,
                                      (has_mask, blend, 1 if m["identity"] else 0, aux),
                                      (m["gain"], m["multiply"], m["offset"], 0.0), (0.0, 0.0, 0.0, m["mix"])))
            passes.append((self._pipeline("grade"), [(0, ins[0].buffer, ins[0].nbytes), (1, mbuf, mbytes),
                                                     (2, out.buffer, out.nbytes)], slot, groups, 3))
        elif kind == "Blur":
            m = map_blur(p)
            mbuf, mbytes, mrect, has_mask = mask_info(ins[1] if len(ins) > 1 else None)
            blend = 1 if (has_mask or m["mix"] != 1.0) else 0
            src = ins[0]
            if m["radius"] == 0:
                tmp, tmp_buffer = src, None            # below half a pixel the CPU copies; so does a radius-0 pass
            else:
                tmp_buffer, tmp_bytes = self._acquire(src.nbytes)
                tmp = DeviceImage(tmp_buffer, src.data, src.display, self.epoch, tmp_bytes)
                slot_h = add_params(_params(_rect(dst), _rect(src.data), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, m["radius"], 0),
                                            (m["scale"], 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
                passes.append((self._pipeline("blur_h"), [(0, src.buffer, src.nbytes), (2, tmp.buffer, tmp.nbytes)],
                               slot_h, groups, 3))
            scale = m["scale"] if m["radius"] else 1.0
            slot_v = add_params(_params(_rect(dst), _rect(src.data), _rect(src.data), mrect, (has_mask, blend, m["radius"], 0),
                                        (scale, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, m["mix"])))
            passes.append((self._pipeline("blur_v"), [(0, tmp.buffer, tmp.nbytes), (1, mbuf, mbytes), (2, out.buffer, out.nbytes),
                                                      (4, src.buffer, src.nbytes)], slot_v, groups, 3))
            if tmp_buffer is not None:
                self._release(tmp_buffer, tmp.nbytes)     # only passes encoded before the next writer read it
        elif kind == "Transform":
            source = ins[0]
            plan = node.plan
            mbuf, mbytes, mrect, has_mask = mask_info(ins[1] if len(ins) > 1 else None)
            blend = 1 if (has_mask or plan.mix != 1.0) else 0
            flags_word = (1 if plan.clamp else 0) | (2 if plan.black_outside else 0) | (4 if plan.zero_rows else 0) | (8 if plan.weighted else 0)
            mode = 0 if plan.mode == "copy" else 1
            tbl, tbl_bytes = table if table is not None else (zero, 16)
            slot = add_params(_params(_rect(dst), _rect(source.data), (plan.filter, plan.shift[0], plan.shift[1], -1), mrect,
                                      (has_mask, blend, mode, flags_word), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, plan.mix)))
            passes.append((self._pipeline("transform"), [(0, source.buffer, source.nbytes), (1, mbuf, mbytes),
                                                        (2, out.buffer, out.nbytes), (4, tbl, tbl_bytes)], slot, groups, 3))
        elif kind == "Merge":
            m = map_merge(p)
            a, b = ins[0], ins[1]
            mbuf, mbytes, mrect, has_mask = mask_info(ins[2] if len(ins) > 2 else None)
            slot = add_params(_params(_rect(dst), _rect(a.data), _rect(b.data), mrect, (has_mask, 0, m["op"], 0),
                                      (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, m["mix"])))
            passes.append((self._pipeline("merge"), [(0, a.buffer, a.nbytes), (1, mbuf, mbytes), (2, out.buffer, out.nbytes),
                                                    (4, b.buffer, b.nbytes)], slot, groups, 3))
        else:
            raise Unsupported(f"{kind} is not an island node")

    # -- CPU boundary ------------------------------------------------------------------------

    def boundary(self, kind, reason):
        self.counters.cpu_boundaries.append(f"{kind}: {reason}")

    def realize(self, items, cancel=None):
        """Materialize deferred inputs for a CPU consumer (an intermediate readback, counted)."""
        result = []
        for item in items:
            if isinstance(item, DeferredImage):
                before = self.counters.downloads
                item = self.run(item, cancel)
                self.counters.intermediate_readbacks += self.counters.downloads - before
            result.append(item)
        return result


def island_interior(nodes, order, target):
    """Keys of island nodes whose every reader is an enabled island node (their results stay on the
    device). The target is always an exit. Computed once per evaluation."""
    readers = {}
    for key in order:
        for source in nodes[key]["inputs"].values():
            if source is not None:
                readers.setdefault(source, []).append(key)
    interior = set()
    for key in order:
        node = nodes[key]
        if key == target or node["disabled"] or node["type"] not in KINDS:
            continue
        users = readers.get(key, [])
        if users and all(nodes[u]["type"] in KINDS and not nodes[u]["disabled"] for u in users):
            interior.add(key)
    return interior
