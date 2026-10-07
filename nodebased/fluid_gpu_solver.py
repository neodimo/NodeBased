"""GPU-resident 3D smoke and fire solver (Lane 6, step D). See docs/FLUIDS_SPIKE.md, "Step D".

`GpuSmoke3D` is the `fluid3d.Smoke3D` substep written as wgpu compute shaders. The fields live in GPU buffers
between substeps; the host uploads the per-substep emission lists, force parameters and (for a moving collider)
the solid mask, and reads back one number per substep, the largest cell residual of the pressure solve. A
`State` is read back only when something asks for its arrays (a frame checkpoint, a test), and in sparse mode
only the active tiles are read.

Pressure: a geometric multigrid V-cycle (2 red-black Gauss-Seidel sweeps down and up, unsmoothed aggregation
restriction and prolongation with a scaled correction, Galerkin coarse operators, a fixed number of sweeps on
the coarsest grid), all on the GPU; `GpuMultigrid3D.solve` is also a `pressure_solver` hook for the CPU solver.
The cycle count is calibrated on the first solve that has a right-hand side (cycle until the largest cell
residual is at most the tolerance, one readback per cycle) and recorded in `State.meta["mg_cycles"]`; every later
solve runs exactly that many cycles and reads the residual once, adding cycles only if it is above the tolerance
(and recording the new count). The count is a function of the state, so a re-solve from a checkpoint is
bit-identical to the run that made it.

Sparse tiles (`sparse=True`): the fields stay in dense device buffers but every kernel is dispatched over the
list of active 8 x 8 x 8 tiles only (an indirect dispatch built on the GPU: activity per tile, growth by one
tile, a deterministic ascending compaction). A tile is active where density, temperature above ambient, fuel or
flame exceed `sparse_threshold`, or a face speed exceeds `sparse_velocity`, plus the tiles that hold a source
footprint, dilated by one tile. Tiles that leave the mask are zeroed (mass below the threshold is dropped);
inactive space is treated as a wall by the pressure solve. Level 0 of the multigrid follows the tile list; the
coarse levels are dense (an eighth of the fine grid, and smaller).

Determinism: the kernels have no atomics that change results and every reduction has a fixed order, so the same
inputs give bit-identical grids on the same adapter.
"""
from __future__ import annotations

import logging
import math
import time
import uuid

import numpy as np

from .cancellation import Cancelled
from . import fluid3d
from .fluid3d import Smoke3D, _sl
from .simcache import State

TILE = 8
GPU_MEMORY_BUDGET = 8 << 30          # bytes a solver may hold on the card (12 GB card, shared with the viewport)
MG_MAX_CYCLES = 60
MG_COARSE_SWEEPS = 24
MG_PRE = MG_POST = 2
REDUCE_GROUPS = 1024
RED_SLOTS = 16
DEFAULT_SPARSE_THRESHOLD = 1.0e-3
DEFAULT_SPARSE_VELOCITY = 5.0e-2
LOG = logging.getLogger(__name__)
STATS = {"dispatches": 0, "submits": 0, "readbacks": 0}


class Unsupported(Exception):
    """The caller should solve on the CPU: no compute adapter, or the grid does not fit the card."""


# --- WGSL -----------------------------------------------------------------------------------------

_PRELUDE = """
struct UB { a: vec4<u32>, b: vec4<u32>, c: vec4<f32>, d: vec4<f32> };
@group(0) @binding(0) var<uniform> P: UB;
fn ic(x: i32, y: i32, z: i32) -> u32 { return u32((x * i32(P.a.y) + y) * i32(P.a.z) + z); }
fn iv(x: i32, y: i32, z: i32) -> u32 { return u32((x * (i32(P.a.y) + 1) + y) * i32(P.a.z) + z); }
fn iw(x: i32, y: i32, z: i32) -> u32 { return u32((x * i32(P.a.y) + y) * (i32(P.a.z) + 1) + z); }
"""

_CELL_OF = """
fn wlin(wid: vec3<u32>) -> u32 { return wid.x + wid.y * 32768u; }
fn cell_of(wid: vec3<u32>, lid: vec3<u32>) -> vec3<i32> {
    var c: vec3<u32>;
    if (P.a.w == 1u) {
        let lin = wlin(wid);
        let t = tiles[lin >> 1u];
        if (t == 0xFFFFFFFFu) { return vec3<i32>(-1, -1, -1); }
        let ntz = (P.a.z + 7u) / 8u;
        let nty = (P.a.y + 7u) / 8u;
        let tz = t % ntz;
        let ty = (t / ntz) % nty;
        let tx = t / (ntz * nty);
        c = vec3<u32>(tx * 8u + (lin & 1u) * 4u + lid.z, ty * 8u + lid.y, tz * 8u + lid.x);
    } else {
        c = vec3<u32>(wid.z * 4u + lid.z, wid.y * 8u + lid.y, wid.x * 8u + lid.x);
    }
    if (c.x >= P.a.x || c.y >= P.a.y || c.z >= P.a.z) { return vec3<i32>(-1, -1, -1); }
    return vec3<i32>(c);
}
"""

_TYPES = {"f32": "f32", "u32": "u32", "v4": "vec4<f32>"}


def _tri(name, dx, dy, dz):
    """WGSL function tri_<name>(p) = clamped trilinear sample of `name` (array dims dx, dy, dz as WGSL i32 expressions)."""
    return """
fn tri_NAME(p: vec3<f32>) -> f32 {
    let nx = DX; let ny = DY; let nz = DZ;
    let x = clamp(p.x, 0.0, f32(nx - 1)); let y = clamp(p.y, 0.0, f32(ny - 1)); let z = clamp(p.z, 0.0, f32(nz - 1));
    let i0 = min(i32(x), nx - 2); let j0 = min(i32(y), ny - 2); let k0 = min(i32(z), nz - 2);
    let tx = x - f32(i0); let ty = y - f32(j0); let tz = z - f32(k0);
    let b = (i0 * ny + j0) * nz + k0;
    let sy = nz; let sx = ny * nz;
    let c000 = NAME[b]; let c001 = NAME[b + 1]; let c010 = NAME[b + sy]; let c011 = NAME[b + sy + 1];
    let c100 = NAME[b + sx]; let c101 = NAME[b + sx + 1]; let c110 = NAME[b + sx + sy]; let c111 = NAME[b + sx + sy + 1];
    let c00 = c000 + (c001 - c000) * tz; let c01 = c010 + (c011 - c010) * tz;
    let c10 = c100 + (c101 - c100) * tz; let c11 = c110 + (c111 - c110) * tz;
    let c0 = c00 + (c01 - c00) * ty; let c1 = c10 + (c11 - c10) * ty;
    return c0 + (c1 - c0) * tx;
}
fn bounds_NAME(p: vec3<f32>) -> vec2<f32> {
    let nx = DX; let ny = DY; let nz = DZ;
    let x = clamp(p.x, 0.0, f32(nx - 1)); let y = clamp(p.y, 0.0, f32(ny - 1)); let z = clamp(p.z, 0.0, f32(nz - 1));
    let i0 = min(i32(x), nx - 2); let j0 = min(i32(y), ny - 2); let k0 = min(i32(z), nz - 2);
    let b = (i0 * ny + j0) * nz + k0;
    let sy = nz; let sx = ny * nz;
    var lo = NAME[b]; var hi = lo;
    var o = array<i32, 7>(1, sy, sy + 1, sx, sx + 1, sx + sy, sx + sy + 1);
    for (var q = 0; q < 7; q++) { let c = NAME[b + o[q]]; lo = min(lo, c); hi = max(hi, c); }
    return vec2<f32>(lo, hi);
}
""".replace("NAME", name).replace("DX", dx).replace("DY", dy).replace("DZ", dz)


_TRI_UVW = (_tri("u", "i32(P.a.x) + 1", "i32(P.a.y)", "i32(P.a.z)")
            + _tri("v", "i32(P.a.x)", "i32(P.a.y) + 1", "i32(P.a.z)")
            + _tri("w", "i32(P.a.x)", "i32(P.a.y)", "i32(P.a.z) + 1"))
_TRI_CELL = lambda name: _tri(name, "i32(P.a.x)", "i32(P.a.y)", "i32(P.a.z)")

_BACKTRACE = """
fn vel(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(tri_u(p - vec3<f32>(0.0, 0.5, 0.5)), tri_v(p - vec3<f32>(0.5, 0.0, 0.5)), tri_w(p - vec3<f32>(0.5, 0.5, 0.0)));
}
fn depart(p: vec3<f32>) -> vec3<f32> {
    let dt = P.c.x;
    let v1 = vel(p);
    let pm = clamp(p - 0.5 * dt * v1, vec3<f32>(0.0), vec3<f32>(f32(P.a.x), f32(P.a.y), f32(P.a.z)));
    let v2 = vel(pm);
    return p - dt * v2;
}
"""

_DIAG = """
fn diag_at(i: u32, x: i32, y: i32, z: i32) -> f32 {
    let sx = P.a.y * P.a.z; let sy = P.a.z;
    let w = wt[i];
    var d = w.x + w.y + w.z + w.w;
    if (x > 0) { d += wt[i - sx].x; }
    if (y > 0) { d += wt[i - sy].y; }
    if (z > 0) { d += wt[i - 1u].z; }
    return d;
}
"""

_KERNELS = {}


def _kernel(name, bindings, body, lib="", cellwise=True, size="8, 8, 4", extra=""):
    decl = "\n".join(f"@group(0) @binding({i}) var<storage, {'read' if acc == 'r' else 'read_write'}> {n}: array<{_TYPES[t]}>;"
                     for i, (n, t, acc) in enumerate(bindings, start=1))
    if cellwise:
        head = (_PRELUDE + decl + _CELL_OF + lib + extra + "\n@compute @workgroup_size(SIZE)\n"
                "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n"
                "    let c = cell_of(wid, lid);\n    if (c.x < 0) { return; }\n"
                "    let x = c.x; let y = c.y; let z = c.z;\n    let nx = i32(P.a.x); let ny = i32(P.a.y); let nz = i32(P.a.z);\n"
                "    let i = ic(x, y, z);\n").replace("SIZE", size)
        src = head + body + "\n}\n"
    else:
        src = _PRELUDE + decl + lib + extra + body
    _KERNELS[name] = (tuple(b[0] for b in bindings), src, cellwise)


def _c(name, ty="f32", acc="r"):
    return (name, ty, acc)


TILES = _c("tiles", "u32")

# entries: sources ----------------------------------------------------------------------------------
_kernel("emit_scalar", [_c("d", acc="rw"), _c("t", acc="rw"), _c("f", acc="rw"), _c("eidx", "u32"), _c("eval")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= P.b.y) { return; }
    let e = P.b.x + g.x;
    let i = eidx[e];
    d[i] = d[i] + eval[e * 8u];
    t[i] = t[i] + eval[e * 8u + 1u];
    f[i] = f[i] + eval[e * 8u + 2u];
}
""", cellwise=False)

_kernel("emit_vel", [_c("u", acc="rw"), _c("v", acc="rw"), _c("w", acc="rw"), _c("eidx", "u32"), _c("eval")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= P.b.y) { return; }
    let e = P.b.x + g.x;
    let side = i32(P.b.z);
    let cidx = eidx[e];
    let nz = i32(P.a.z); let ny = i32(P.a.y);
    let z = i32(cidx) % nz;
    let y = (i32(cidx) / nz) % ny;
    let x = i32(cidx) / (nz * ny);
    let blend = eval[e * 8u + 3u];
    let iu_ = ic(x + side, y, z);
    u[iu_] = u[iu_] + blend * (eval[e * 8u + 4u] - u[iu_]);
    let iv_ = iv(x, y + side, z);
    v[iv_] = v[iv_] + blend * (eval[e * 8u + 5u] - v[iv_]);
    let iw_ = iw(x, y, z + side);
    w[iw_] = w[iw_] + blend * (eval[e * 8u + 6u] - w[iw_]);
}
""", cellwise=False)

# advection -----------------------------------------------------------------------------------------
_kernel("advect_vel", [_c("u"), _c("v"), _c("w"), _c("o", acc="rw"), TILES], """
    let axis = P.b.x;
    for (var s = 0; s < 2; s++) {
        var fx = x; var fy = y; var fz = z;
        if (s == 1) {
            if (axis == 0u && x == nx - 1) { fx = nx; }
            else if (axis == 1u && y == ny - 1) { fy = ny; }
            else if (axis == 2u && z == nz - 1) { fz = nz; }
            else { break; }
        }
        var off = vec3<f32>(0.0, 0.5, 0.5);
        if (axis == 1u) { off = vec3<f32>(0.5, 0.0, 0.5); }
        if (axis == 2u) { off = vec3<f32>(0.5, 0.5, 0.0); }
        let p = vec3<f32>(f32(fx), f32(fy), f32(fz)) + off;
        let b = depart(p) - off;
        if (axis == 0u) { o[ic(fx, fy, fz)] = tri_u(b); }
        else if (axis == 1u) { o[iv(fx, fy, fz)] = tri_v(b); }
        else { o[iw(fx, fy, fz)] = tri_w(b); }
    }
""", lib=_TRI_UVW + _BACKTRACE)

_kernel("advect_pred", [_c("u"), _c("v"), _c("w"), _c("src"), _c("pred", acc="rw"), TILES], """
    let p = vec3<f32>(f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5);
    let b = depart(p);
    pred[i] = tri_src(b - vec3<f32>(0.5));
""", lib=_TRI_UVW + _BACKTRACE + _TRI_CELL("src"))

_kernel("advect_final", [_c("u"), _c("v"), _c("w"), _c("src"), _c("pred"), _c("out", acc="rw"), TILES], """
    let p = vec3<f32>(f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5);
    let b = depart(p);
    var res = pred[i];
    if (P.b.x == 1u) {
        let travel = length(p - b);
        let trust = clamp(2.0 - travel, 0.0, 1.0);
        let back = tri_pred(2.0 * p - b - vec3<f32>(0.5));
        res = res + trust * 0.5 * (src[i] - back);
        let lh = bounds_src(b - vec3<f32>(0.5));
        res = min(max(res, lh.x), lh.y);
    }
    out[i] = res;
""", lib=_TRI_UVW + _BACKTRACE + _TRI_CELL("src") + _TRI_CELL("pred"))

_kernel("scale_apply", [_c("u"), _c("v"), _c("w"), _c("out", acc="rw"), _c("red"), TILES], """
    let base = P.c.y;
    var val = out[i];
    if (P.b.z == 1u) {
        let g = red[P.b.w + 1u];
        if (g > 1.0e-12) { val = base + (val - base) * clamp(red[P.b.w] / g, 0.5, 2.0); }
    }
    let p = vec3<f32>(f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5);
    let b = depart(p);
    let op = P.b.y;
    if (((op & 1u) != 0u && b.x < 0.0) || ((op & 2u) != 0u && b.x > f32(nx))
        || ((op & 4u) != 0u && b.y < 0.0) || ((op & 8u) != 0u && b.y > f32(ny))
        || ((op & 16u) != 0u && b.z < 0.0) || ((op & 32u) != 0u && b.z > f32(nz))) { val = base; }
    out[i] = val;
""", lib=_TRI_UVW + _BACKTRACE)

# local physics: combustion, decay, solids ----------------------------------------------------------
_kernel("local_phys", [_c("d", acc="rw"), _c("t", acc="rw"), _c("f", acc="rw"), _c("burn", acc="rw"), _c("solid", "u32"), TILES], """
    var dd = d[i]; var tt = t[i]; var ff = f[i]; var bb = 0.0;
    let amb = P.d.w;
    if (P.b.x == 1u) {
        let hot = tt >= P.c.x && ff > 0.0;
        var consumed = 0.0;
        if (hot) { consumed = ff * P.c.y; }
        ff = ff - consumed;
        tt = tt + consumed * P.c.z;
        dd = dd + consumed * P.c.w;
        bb = consumed / P.d.x;
    }
    if (P.b.y == 1u) { dd = dd * P.d.y; }
    if (P.b.z == 1u) { tt = tt - amb; tt = tt * P.d.z; tt = tt + amb; }
    if (P.b.w == 1u && solid[i] != 0u) { dd = 0.0; ff = 0.0; bb = 0.0; tt = amb; }
    d[i] = dd; t[i] = tt; f[i] = ff; burn[i] = bb;
""")

# forces --------------------------------------------------------------------------------------------
_kernel("buoy", [_c("v", acc="rw"), _c("d"), _c("t"), TILES], """
    if (y >= 1) {
        let f1 = -P.c.x * d[ic(x, y - 1, z)] + P.c.y * (t[ic(x, y - 1, z)] - P.c.z);
        let f2 = -P.c.x * d[i] + P.c.y * (t[i] - P.c.z);
        let j = iv(x, y, z);
        v[j] = v[j] + P.c.w * (f1 + f2);
    }
""")

_kernel("grav", [_c("u", acc="rw"), _c("v", acc="rw"), _c("w", acc="rw"), _c("d"), TILES], """
    let dc = d[i];
    if (x >= 1) { let j = ic(x, y, z); u[j] = u[j] + 0.5 * P.c.x * (d[ic(x - 1, y, z)] + dc); }
    if (y >= 1) { let j = iv(x, y, z); v[j] = v[j] + 0.5 * P.c.y * (d[ic(x, y - 1, z)] + dc); }
    if (z >= 1) { let j = iw(x, y, z); w[j] = w[j] + 0.5 * P.c.z * (d[ic(x, y, z - 1)] + dc); }
""")

_kernel("face_op", [_c("u", acc="rw"), _c("v", acc="rw"), _c("w", acc="rw"), TILES], """
    let scale = P.b.x == 1u;
    let k = P.c;
    { let j = ic(x, y, z); if (scale) { u[j] = u[j] * k.x; } else { u[j] = u[j] + k.x; } }
    { let j = iv(x, y, z); if (scale) { v[j] = v[j] * k.x; } else { v[j] = v[j] + k.y; } }
    { let j = iw(x, y, z); if (scale) { w[j] = w[j] * k.x; } else { w[j] = w[j] + k.z; } }
    if (x == nx - 1) { let j = ic(nx, y, z); if (scale) { u[j] = u[j] * k.x; } else { u[j] = u[j] + k.x; } }
    if (y == ny - 1) { let j = iv(x, ny, z); if (scale) { v[j] = v[j] * k.x; } else { v[j] = v[j] + k.y; } }
    if (z == nz - 1) { let j = iw(x, y, nz); if (scale) { w[j] = w[j] * k.x; } else { w[j] = w[j] + k.z; } }
""")

_kernel("apply_cforce", [_c("u", acc="rw"), _c("v", acc="rw"), _c("w", acc="rw"), _c("cx", acc="rw"), _c("cy", acc="rw"),
                         _c("cz", acc="rw"), _c("solid", "u32"), TILES], """
    let zero = P.b.x == 1u;
    let s = P.c.x;
    if (x >= 1) {
        let j = ic(x - 1, y, z);
        var a = cx[j] * s; if (zero && solid[j] != 0u) { a = 0.0; }
        var b = cx[i] * s; if (zero && solid[i] != 0u) { b = 0.0; }
        let q = ic(x, y, z); u[q] = u[q] + 0.5 * (a + b);
    }
    if (y >= 1) {
        let j = ic(x, y - 1, z);
        var a = cy[j] * s; if (zero && solid[j] != 0u) { a = 0.0; }
        var b = cy[i] * s; if (zero && solid[i] != 0u) { b = 0.0; }
        let q = iv(x, y, z); v[q] = v[q] + 0.5 * (a + b);
    }
    if (z >= 1) {
        let j = ic(x, y, z - 1);
        var a = cz[j] * s; if (zero && solid[j] != 0u) { a = 0.0; }
        var b = cz[i] * s; if (zero && solid[i] != 0u) { b = 0.0; }
        let q = iw(x, y, z); w[q] = w[q] + 0.5 * (a + b);
    }
""")

_CC = """
fn ucc(x: i32, y: i32, z: i32) -> f32 { return 0.5 * (u[ic(x, y, z)] + u[ic(x + 1, y, z)]); }
fn vcc(x: i32, y: i32, z: i32) -> f32 { return 0.5 * (v[iv(x, y, z)] + v[iv(x, y + 1, z)]); }
fn wcc(x: i32, y: i32, z: i32) -> f32 { return 0.5 * (w[iw(x, y, z)] + w[iw(x, y, z + 1)]); }
fn span(c: i32, n: i32) -> vec2<i32> { return vec2<i32>(max(c - 1, 0), min(c + 1, n - 1)); }
"""
_FIELD = """
fn ucc(x: i32, y: i32, z: i32) -> f32 { return 0.5 * (u[ic(x,y,z)] + u[ic(x+1,y,z)]); }
fn vcc(x: i32, y: i32, z: i32) -> f32 { return 0.5 * (v[iv(x,y,z)] + v[iv(x,y+1,z)]); }
fn wcc(x: i32, y: i32, z: i32) -> f32 { return 0.5 * (w[iw(x,y,z)] + w[iw(x,y,z+1)]); }
fn weight(i: u32, x: i32, y: i32, z: i32) -> f32 {
    if (P.b.x == 0u) { return 1.0; }
    var value = d[i];
    if (P.b.x == 2u) { value = t[i]; }
    if (P.b.x == 3u) {
        let a = ucc(x,y,z); let b = vcc(x,y,z); let c = wcc(x,y,z);
        value = sqrt(a*a+b*b+c*c);
    }
    if (P.b.x == 4u) { value = mag[i]; }
    let lo = P.c.x; let hi = max(P.c.y, lo + 1.0e-9);
    let width = max(P.c.z, 0.0) * (hi-lo);
    if (width <= 1.0e-12) { return select(0.0, 1.0, value >= lo && value <= hi); }
    return min(clamp((value-lo+width)/width, 0.0, 1.0), clamp((hi+width-value)/width, 0.0, 1.0));
}
"""
_kernel("field_weight", [_c("d"), _c("t"), _c("u"), _c("v"), _c("w"), _c("mag"),
                         _c("out", acc="rw"), TILES], """out[i] = weight(i,x,y,z);""", lib=_FIELD)
_kernel("decay_field", [_c("d", acc="rw"), _c("weight"), TILES],
        """d[i] = d[i] * (1.0 - weight[i] * (1.0 - P.c.x));""")
_kernel("cool_field", [_c("t", acc="rw"), TILES],
        """t[i] = (t[i] - P.c.x) * P.c.y + P.c.x;""")
_kernel("force_weight", [_c("cx", acc="rw"), _c("cy", acc="rw"), _c("cz", acc="rw"),
                          _c("weight"), TILES], """cx[i] *= weight[i]; cy[i] *= weight[i]; cz[i] *= weight[i];""")
_kernel("shred", [_c("u"), _c("v"), _c("w"), _c("cx", acc="rw"), _c("cy", acc="rw"),
                  _c("cz", acc="rw"), TILES], """
    let ax = max(x-1,0); let bx = min(x+1,nx-1);
    let ay = max(y-1,0); let by = min(y+1,ny-1);
    let az = max(z-1,0); let bz = min(z+1,nz-1);
    let dx = f32(bx-ax); let dy = f32(by-ay); let dz = f32(bz-az);
    let dux = (ucc(bx,y,z)-ucc(ax,y,z))/dx; let duy = (ucc(x,by,z)-ucc(x,ay,z))/dy;
    let duz = (ucc(x,y,bz)-ucc(x,y,az))/dz;
    let dvx = (vcc(bx,y,z)-vcc(ax,y,z))/dx; let dvy = (vcc(x,by,z)-vcc(x,ay,z))/dy;
    let dvz = (vcc(x,y,bz)-vcc(x,y,az))/dz;
    let dwx = (wcc(bx,y,z)-wcc(ax,y,z))/dx; let dwy = (wcc(x,by,z)-wcc(x,ay,z))/dy;
    let dwz = (wcc(x,y,bz)-wcc(x,y,az))/dz;
    let omega = vec3<f32>(dwy-dvz,duz-dwx,dvx-duy);
    let oh = omega / (length(omega)+1.0e-9);
    let delta = P.c.x * vec3<f32>(dux*oh.x+0.5*(duy+dvx)*oh.y+0.5*(duz+dwx)*oh.z,
        0.5*(dvx+duy)*oh.x+dvy*oh.y+0.5*(dvz+dwy)*oh.z,
        0.5*(dwx+duz)*oh.x+0.5*(dwy+dvz)*oh.y+dwz*oh.z);
    let speed = length(vec3<f32>(ucc(x,y,z),vcc(x,y,z),wcc(x,y,z)));
    let capped = delta * min(1.0,max(speed,0.05)/(length(delta)+1.0e-9));
    cx[i]=capped.x; cy[i]=capped.y; cz[i]=capped.z;
""", lib=_CC)
_kernel("curl", [_c("u"), _c("v"), _c("w"), _c("cx", acc="rw"), _c("cy", acc="rw"), _c("cz", acc="rw"), _c("mag", acc="rw"), TILES], """
    let sx = span(x, nx); let sy = span(y, ny); let sz = span(z, nz);
    let dx = f32(sx.y - sx.x); let dy = f32(sy.y - sy.x); let dz = f32(sz.y - sz.x);
    let dwdy = (wcc(x, sy.y, z) - wcc(x, sy.x, z)) / dy;
    let dvdz = (vcc(x, y, sz.y) - vcc(x, y, sz.x)) / dz;
    let dudz = (ucc(x, y, sz.y) - ucc(x, y, sz.x)) / dz;
    let dwdx = (wcc(sx.y, y, z) - wcc(sx.x, y, z)) / dx;
    let dvdx = (vcc(sx.y, y, z) - vcc(sx.x, y, z)) / dx;
    let dudy = (ucc(x, sy.y, z) - ucc(x, sy.x, z)) / dy;
    let ox = dwdy - dvdz; let oy = dudz - dwdx; let oz = dvdx - dudy;
    cx[i] = ox; cy[i] = oy; cz[i] = oz;
    mag[i] = sqrt(ox * ox + oy * oy + oz * oz);
""", lib=_CC)

_kernel("confine", [_c("cx", acc="rw"), _c("cy", acc="rw"), _c("cz", acc="rw"), _c("mag"), TILES], """
    let sx = span(x, nx); let sy = span(y, ny); let sz = span(z, nz);
    let gx = (mag[ic(sx.y, y, z)] - mag[ic(sx.x, y, z)]) / f32(sx.y - sx.x);
    let gy = (mag[ic(x, sy.y, z)] - mag[ic(x, sy.x, z)]) / f32(sy.y - sy.x);
    let gz = (mag[ic(x, y, sz.y)] - mag[ic(x, y, sz.x)]) / f32(sz.y - sz.x);
    let norm = sqrt(gx * gx + gy * gy + gz * gz) + 1.0e-9;
    let nxv = gx / norm; let nyv = gy / norm; let nzv = gz / norm;
    let ox = cx[i]; let oy = cy[i]; let oz = cz[i];
    let s = P.c.x;
    cx[i] = s * (nyv * oz - nzv * oy);
    cy[i] = s * (nzv * ox - nxv * oz);
    cz[i] = s * (nxv * oy - nyv * ox);
""", lib="""
fn span(c: i32, n: i32) -> vec2<i32> { return vec2<i32>(max(c - 1, 0), min(c + 1, n - 1)); }
""")

# projection ----------------------------------------------------------------------------------------
_kernel("constrain", [_c("u", acc="rw"), _c("v", acc="rw"), _c("w", acc="rw"), _c("blk", "u32"), _c("svel", "v4"), TILES], """
    let openbits = P.b.x;
    let has_s = P.b.w == 1u;
    let has_v = P.c.x > 0.5;
    for (var axis = 0; axis < 3; axis++) {
        for (var s = 0; s < 2; s++) {
            var fc = vec3<i32>(x, y, z);
            var pos = c[axis];
            var n = nx;
            if (axis == 1) { n = ny; }
            if (axis == 2) { n = nz; }
            if (s == 1) {
                if (pos != n - 1) { break; }
                pos = n;
                fc[axis] = n;
            }
            var lo = -1; var hi = -1;
            if (pos >= 1) { var cl = fc; cl[axis] = pos - 1; lo = i32(ic(cl.x, cl.y, cl.z)); }
            if (pos <= n - 1) { var ch = fc; ch[axis] = pos; hi = i32(ic(ch.x, ch.y, ch.z)); }
            var fi: u32;
            if (axis == 0) { fi = ic(fc.x, fc.y, fc.z); } else if (axis == 1) { fi = iv(fc.x, fc.y, fc.z); } else { fi = iw(fc.x, fc.y, fc.z); }
            var val: f32 = 0.0;
            var write = false;
            if (pos == 0 || pos == n) {
                let bit = 1u << u32(2 * axis + s);
                if ((openbits & bit) == 0u) { write = true; }
                else {
                    let cell = select(lo, hi, pos == 0);
                    if (blk[u32(cell)] == 1u) { write = true; }
                }
            } else if (blk[u32(lo)] == 1u || blk[u32(hi)] == 1u) {
                write = true;
                if (has_s && has_v) {
                    if (blk[u32(lo)] == 1u) { val = svel[u32(lo)][axis]; }
                    else { val = svel[u32(hi)][axis]; }
                }
            }
            if (write) {
                if (axis == 0) { u[fi] = val; } else if (axis == 1) { v[fi] = val; } else { w[fi] = val; }
            }
        }
    }
""")

_kernel("rhs_build", [_c("u"), _c("v"), _c("w"), _c("burn"), _c("blk", "u32"), _c("rhs", acc="rw"), _c("p", acc="rw"), TILES], """
    if (blk[i] != 0u) { rhs[i] = 0.0; p[i] = 0.0; return; }
    let sx = u32(ny * nz); let sy = u32(nz);
    var uh = u[ic(x + 1, y, z)]; var vh = v[iv(x, y + 1, z)]; var wh = w[iw(x, y, z + 1)];
    if (x < nx - 1 && blk[i + sx] == 2u) { uh = 0.0; }
    if (y < ny - 1 && blk[i + sy] == 2u) { vh = 0.0; }
    if (z < nz - 1 && blk[i + 1u] == 2u) { wh = 0.0; }
    let div = (uh - u[i]) + (vh - v[iv(x, y, z)]) + (wh - w[iw(x, y, z)]);
    var r = -div;
    r = r + P.c.x * burn[i];
    rhs[i] = r;
""")

_kernel("weights0", [_c("blk", "u32"), _c("wt", "v4", "rw"), TILES], """
    if (blk[i] != 0u) { wt[i] = vec4<f32>(0.0); return; }
    let sx = u32(ny * nz); let sy = u32(nz);
    var w = vec4<f32>(0.0);
    if (x < nx - 1 && blk[i + sx] == 0u) { w.x = 1.0; }
    if (y < ny - 1 && blk[i + sy] == 0u) { w.y = 1.0; }
    if (z < nz - 1 && blk[i + 1u] == 0u) { w.z = 1.0; }
    var dw = 0.0;
    if (x < nx - 1 && blk[i + sx] == 2u) { dw = dw + 1.0; }
    if (y < ny - 1 && blk[i + sy] == 2u) { dw = dw + 1.0; }
    if (z < nz - 1 && blk[i + 1u] == 2u) { dw = dw + 1.0; }
    if (x > 0 && blk[i - sx] == 2u) { dw = dw + 1.0; }
    if (y > 0 && blk[i - sy] == 2u) { dw = dw + 1.0; }
    if (z > 0 && blk[i - 1u] == 2u) { dw = dw + 1.0; }
    if ((P.b.x & 1u) != 0u && x == 0) { dw = dw + 1.0; }
    if ((P.b.x & 2u) != 0u && x == nx - 1) { dw = dw + 1.0; }
    if ((P.b.x & 4u) != 0u && y == 0) { dw = dw + 1.0; }
    if ((P.b.x & 8u) != 0u && y == ny - 1) { dw = dw + 1.0; }
    if ((P.b.x & 16u) != 0u && z == 0) { dw = dw + 1.0; }
    if ((P.b.x & 32u) != 0u && z == nz - 1) { dw = dw + 1.0; }
    w.w = dw;
    wt[i] = w;
""")

_kernel("smooth", [_c("q", acc="rw"), _c("b"), _c("wt", "v4"), TILES], """
    if (u32((x + y + z) & 1) != P.b.x) { return; }
    let sx = P.a.y * P.a.z; let sy = P.a.z;
    let d = diag_at(i, x, y, z);
    if (d <= 0.0) { return; }
    let w = wt[i];
    var s = 0.0;
    if (x < nx - 1) { s = s + w.x * q[i + sx]; }
    if (y < ny - 1) { s = s + w.y * q[i + sy]; }
    if (z < nz - 1) { s = s + w.z * q[i + 1u]; }
    if (x > 0) { s = s + wt[i - sx].x * q[i - sx]; }
    if (y > 0) { s = s + wt[i - sy].y * q[i - sy]; }
    if (z > 0) { s = s + wt[i - 1u].z * q[i - 1u]; }
    q[i] = (b[i] + s) / d;
""", lib=_DIAG)

_kernel("residual", [_c("q"), _c("b"), _c("wt", "v4"), _c("r", acc="rw"), TILES], """
    let sx = P.a.y * P.a.z; let sy = P.a.z;
    let d = diag_at(i, x, y, z);
    if (d <= 0.0) { r[i] = 0.0; return; }
    let w = wt[i];
    var s = 0.0;
    if (x < nx - 1) { s = s + w.x * q[i + sx]; }
    if (y < ny - 1) { s = s + w.y * q[i + sy]; }
    if (z < nz - 1) { s = s + w.z * q[i + 1u]; }
    if (x > 0) { s = s + wt[i - sx].x * q[i - sx]; }
    if (y > 0) { s = s + wt[i - sy].y * q[i - sy]; }
    if (z > 0) { s = s + wt[i - 1u].z * q[i - 1u]; }
    r[i] = b[i] - (d * q[i] - s);
""", lib=_DIAG)

# thread domain = coarse cells (P.a), fine dims in P.b.xyz
_FINE = """
fn fidx(x: i32, y: i32, z: i32) -> u32 { return u32((x * i32(P.b.y) + y) * i32(P.b.z) + z); }
fn infine(x: i32, y: i32, z: i32) -> bool { return x < i32(P.b.x) && y < i32(P.b.y) && z < i32(P.b.z); }
"""
_kernel("restrict", [_c("rf"), _c("bc", acc="rw"), _c("qc", acc="rw"), TILES], """
    var s = 0.0;
    for (var a = 0; a < 2; a++) { for (var b = 0; b < 2; b++) { for (var k = 0; k < 2; k++) {
        let fx = 2 * x + a; let fy = 2 * y + b; let fz = 2 * z + k;
        if (infine(fx, fy, fz)) { s = s + rf[fidx(fx, fy, fz)]; }
    } } }
    bc[i] = s;
    qc[i] = 0.0;
""", lib=_FINE)

_kernel("prolong", [_c("qf", acc="rw"), _c("qc"), TILES], """
    let cx = x / 2; let cy = y / 2; let cz = z / 2;
    let ci = u32((cx * i32(P.b.y) + cy) * i32(P.b.z) + cz);
    qf[i] = qf[i] + P.c.x * qc[ci];
""")

_kernel("coarsen", [_c("wf", "v4"), _c("wc", "v4", "rw"), TILES], """
    var acc = vec4<f32>(0.0);
    for (var a = 0; a < 2; a++) { for (var b = 0; b < 2; b++) {
        let fx = 2 * x + 1; let fy = 2 * y + a; let fz = 2 * z + b;
        if (infine(fx, fy, fz)) { acc.x = acc.x + wf[fidx(fx, fy, fz)].x; }
        let gx = 2 * x + a; let gy = 2 * y + 1; let gz = 2 * z + b;
        if (infine(gx, gy, gz)) { acc.y = acc.y + wf[fidx(gx, gy, gz)].y; }
        let hx = 2 * x + a; let hy = 2 * y + b; let hz = 2 * z + 1;
        if (infine(hx, hy, hz)) { acc.z = acc.z + wf[fidx(hx, hy, hz)].z; }
        for (var k = 0; k < 2; k++) {
            let ex = 2 * x + a; let ey = 2 * y + b; let ez = 2 * z + k;
            if (infine(ex, ey, ez)) { acc.w = acc.w + wf[fidx(ex, ey, ez)].w; }
        }
    } }
    wc[i] = acc;
""", lib=_FINE)

_kernel("project", [_c("u", acc="rw"), _c("v", acc="rw"), _c("w", acc="rw"), _c("q"), _c("wt", "v4"), _c("blk", "u32"), TILES], """
    let sx = u32(ny * nz); let sy = u32(nz);
    let qi = q[i];
    if (x >= 1) {
        let j = ic(x, y, z);
        if (blk[i - sx] == 2u) { u[j] = u[j] - qi; } else { u[j] = u[j] - (qi - q[i - sx]) * wt[i - sx].x; }
    } else if ((P.b.x & 1u) != 0u) { u[ic(0, y, z)] = u[ic(0, y, z)] - qi; }
    if (x == nx - 1) { if ((P.b.x & 2u) != 0u) { let j = ic(nx, y, z); u[j] = u[j] + qi; } }
    else if (blk[i + sx] == 2u) { u[ic(x + 1, y, z)] = qi; }
    if (y >= 1) {
        let j = iv(x, y, z);
        if (blk[i - sy] == 2u) { v[j] = v[j] - qi; } else { v[j] = v[j] - (qi - q[i - sy]) * wt[i - sy].y; }
    } else if ((P.b.x & 4u) != 0u) { v[iv(x, 0, z)] = v[iv(x, 0, z)] - qi; }
    if (y == ny - 1) { if ((P.b.x & 8u) != 0u) { let j = iv(x, ny, z); v[j] = v[j] + qi; } }
    else if (blk[i + sy] == 2u) { v[iv(x, y + 1, z)] = qi; }
    if (z >= 1) {
        let j = iw(x, y, z);
        if (blk[i - 1u] == 2u) { w[j] = w[j] - qi; } else { w[j] = w[j] - (qi - q[i - 1u]) * wt[i - 1u].z; }
    } else if ((P.b.x & 16u) != 0u) { w[iw(x, y, 0)] = w[iw(x, y, 0)] - qi; }
    if (z == nz - 1) { if ((P.b.x & 32u) != 0u) { let j = iw(x, y, nz); w[j] = w[j] + qi; } }
    else if (blk[i + 1u] == 2u) { w[iw(x, y, z + 1)] = qi; }
""")

# reductions ----------------------------------------------------------------------------------------
_kernel("reduce1", [_c("a"), _c("m", "u32"), _c("part", acc="rw")], """
var<workgroup> wg: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = P.a.x;
    let mode = P.b.x;
    let masked = P.b.y == 1u;
    let base = P.c.x;
    var acc = 0.0;
    var i = wid.x * 256u + lid.x;
    loop {
        if (i >= n) { break; }
        if (!(masked && m[i] != 0u)) {
            if (mode == 0u) { acc = acc + (a[i] - base); }
            else if (mode == 1u) { acc = max(acc, abs(a[i])); }
            else { acc = acc + 1.0; }
        }
        i = i + 1024u * 256u;
    }
    wg[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            if (mode == 1u) { wg[lid.x] = max(wg[lid.x], wg[lid.x + s]); } else { wg[lid.x] = wg[lid.x] + wg[lid.x + s]; }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) { part[wid.x] = wg[0]; }
}
""", cellwise=False)

_kernel("reduce2", [_c("part"), _c("red", acc="rw")], """
var<workgroup> wg: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let mode = P.b.x;
    var acc = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let v = part[lid.x + 256u * k];
        if (mode == 1u) { acc = max(acc, v); } else { acc = acc + v; }
    }
    wg[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            if (mode == 1u) { wg[lid.x] = max(wg[lid.x], wg[lid.x + s]); } else { wg[lid.x] = wg[lid.x] + wg[lid.x + s]; }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) { red[P.b.z] = wg[0]; }
}
""", cellwise=False)

_kernel("sub_mean", [_c("q", acc="rw"), _c("blk", "u32"), _c("red"), TILES], """
    var val = q[i];
    if (P.b.x == 1u || (P.b.x == 2u && red[P.b.z] < 0.5)) {
        let cnt = red[P.b.y + 1u];
        if (cnt > 0.0) { val = val - red[P.b.y] / cnt; }
    }
    if (blk[i] != 0u) { val = 0.0; }
    q[i] = val;
""")

_kernel("dw_copy", [_c("wt", "v4"), _c("r", acc="rw"), TILES], """
    r[i] = wt[i].w;
""")

_kernel("copy_f", [_c("src"), _c("dst", acc="rw")], """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= P.a.x) { return; }
    dst[g.x] = src[g.x];
}
""", cellwise=False)

# sparse tiles --------------------------------------------------------------------------------------
_kernel("tile_activity", [_c("d"), _c("t"), _c("f"), _c("burn"), _c("u"), _c("v"), _c("w"), _c("act", "u32", "rw")], """
var<workgroup> hit: atomic<u32>;
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ntx = (P.a.x + 7u) / 8u; let nty = (P.a.y + 7u) / 8u; let ntz = (P.a.z + 7u) / 8u;
    let tile = ((wid.z / 2u) * nty + wid.y) * ntz + wid.x;
    if (lid.x == 0u && lid.y == 0u && lid.z == 0u) { atomicStore(&hit, 0u); }
    workgroupBarrier();
    let cx = (wid.z / 2u) * 8u + (wid.z & 1u) * 4u + lid.z; let cy = wid.y * 8u + lid.y; let cz = wid.x * 8u + lid.x;
    if (cx < P.a.x && cy < P.a.y && cz < P.a.z) {
        let nx = i32(P.a.x); let ny = i32(P.a.y); let nz = i32(P.a.z);
        let x = i32(cx); let y = i32(cy); let z = i32(cz);
        let i = ic(x, y, z);
        var on = abs(d[i]) > P.c.x || abs(t[i] - P.c.z) > P.c.x || abs(f[i]) > P.c.x || abs(burn[i]) > P.c.x;
        let sp = max(abs(u[i]), max(abs(v[iv(x, y, z)]), abs(w[iw(x, y, z)])));
        if (sp > P.c.y) { on = true; }
        if (x == nx - 1 && abs(u[ic(nx, y, z)]) > P.c.y) { on = true; }
        if (y == ny - 1 && abs(v[iv(x, ny, z)]) > P.c.y) { on = true; }
        if (z == nz - 1 && abs(w[iw(x, y, nz)]) > P.c.y) { on = true; }
        if (on) { atomicStore(&hit, 1u); }
    }
    workgroupBarrier();
    if (lid.x == 0u && lid.y == 0u && lid.z == 0u) {
        if (atomicLoad(&hit) == 1u) { act[tile] = 1u; }
    }
}
""", cellwise=False)

_kernel("mark_tiles", [_c("eidx", "u32"), _c("act", "u32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= P.b.y) { return; }
    let cidx = eidx[P.b.x + g.x];
    let nz = P.a.z; let ny = P.a.y;
    let z = cidx % nz; let y = (cidx / nz) % ny; let x = cidx / (nz * ny);
    let ntz = (P.a.z + 7u) / 8u; let nty = (P.a.y + 7u) / 8u;
    act[((x / 8u) * nty + y / 8u) * ntz + z / 8u] = 1u;
}
""", cellwise=False)

_kernel("dilate", [_c("act", "u32"), _c("mask", "u32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let ntx = (P.a.x + 7u) / 8u; let nty = (P.a.y + 7u) / 8u; let ntz = (P.a.z + 7u) / 8u;
    let t = g.x;
    if (t >= ntx * nty * ntz) { return; }
    let tz = i32(t % ntz); let ty = i32((t / ntz) % nty); let tx = i32(t / (ntz * nty));
    var on = 0u;
    for (var a = -1; a <= 1; a++) { for (var b = -1; b <= 1; b++) { for (var c = -1; c <= 1; c++) {
        let x = tx + a; let y = ty + b; let z = tz + c;
        if (x >= 0 && y >= 0 && z >= 0 && x < i32(ntx) && y < i32(nty) && z < i32(ntz)) {
            on = on | act[(u32(x) * nty + u32(y)) * ntz + u32(z)];
        }
    } } }
    mask[t] = on;
}
""", cellwise=False)

_kernel("ring_or", [_c("a", "u32"), _c("b", "u32"), _c("m", "u32"), _c("out", "u32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = g.x;
    if (t >= P.a.x) { return; }
    out[t] = select(0u, 1u, (a[t] != 0u || b[t] != 0u) && m[t] == 0u);
}
""", cellwise=False)

# list of tiles selected by (a, b): mode 0 = new, mode 1 = a and not b. One workgroup, ascending order.
_kernel("compact", [_c("ma", "u32"), _c("mb", "u32"), _c("list", "u32", "rw"), _c("args", "u32", "rw")], """
var<workgroup> counts: array<u32, 256>;
var<workgroup> wtotal: u32;
fn keep(t: u32) -> bool {
    if (P.b.x == 0u) { return ma[t] != 0u; }
    return ma[t] != 0u && mb[t] == 0u;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = P.a.x;
    let per = (n + 255u) / 256u;
    let lo = lid.x * per;
    let hi = min(lo + per, n);
    var c = 0u;
    for (var t = lo; t < hi; t++) { if (keep(t)) { c = c + 1u; } }
    counts[lid.x] = c;
    workgroupBarrier();
    if (lid.x == 0u) {
        var run = 0u;
        for (var k = 0u; k < 256u; k++) { let v = counts[k]; counts[k] = run; run = run + v; }
        let w = run * 2u;
        args[0] = min(w, 32768u); args[1] = max(1u, (w + 32767u) / 32768u); args[2] = 1u; args[3] = run;
        wtotal = run;
    }
    workgroupBarrier();
    var pos = counts[lid.x];
    for (var t = lo; t < hi; t++) { if (keep(t)) { list[pos] = t; pos = pos + 1u; } }
    // the last row of workgroups may run past the list: mark the spare slots
    let tiles_ = wtotal;
    let rows = ((tiles_ + 16383u) / 16384u) * 16384u;
    for (var k = tiles_ + lid.x; k < rows; k = k + 256u) { list[k] = 0xFFFFFFFFu; }
}
""", cellwise=False)

# zero the tiles of a list in one field; P.b.x = face kind (0 cell, 1 u, 2 v, 3 w), P.c.x = value
_kernel("clear_tiles", [_c("fld", acc="rw"), TILES], """
    let kind = P.b.x;
    let val = P.c.x;
    if (kind == 0u) { fld[i] = val; }
    else if (kind == 1u) { fld[ic(x, y, z)] = val; if (x == nx - 1) { fld[ic(nx, y, z)] = val; } }
    else if (kind == 2u) { fld[iv(x, y, z)] = val; if (y == ny - 1) { fld[iv(x, ny, z)] = val; } }
    else if (kind == 3u) { fld[iw(x, y, z)] = val; if (z == nz - 1) { fld[iw(x, y, nz)] = val; } }
""")

_kernel("clear_v4", [_c("fld", "v4", "rw"), TILES], """
    fld[i] = vec4<f32>(0.0);
""")

_kernel("set_blk", [_c("blk", "u32", "rw"), TILES], """
    blk[i] = 2u;
""")

_kernel("mk_blk", [_c("solid", "u32"), _c("blk", "u32", "rw"), TILES], """
    blk[i] = solid[i];
""")

# read back the active tiles: 11 floats per cell (u v w, top faces u v w, d t f burn p) -> two passes
_kernel("pack_a", [_c("u"), _c("v"), _c("w"), _c("out", acc="rw"), TILES], """
    let base = ((wlin(wid) >> 1u) * 512u + ((u32(x) % 8u) * 8u + (u32(y) % 8u)) * 8u + (u32(z) % 8u)) * 11u;
    out[base] = u[i]; out[base + 1u] = v[iv(x, y, z)]; out[base + 2u] = w[iw(x, y, z)];
    out[base + 3u] = 0.0; out[base + 4u] = 0.0; out[base + 5u] = 0.0;
    if (x == nx - 1) { out[base + 3u] = u[ic(nx, y, z)]; }
    if (y == ny - 1) { out[base + 4u] = v[iv(x, ny, z)]; }
    if (z == nz - 1) { out[base + 5u] = w[iw(x, y, nz)]; }
""")

_kernel("pack_b", [_c("d"), _c("t"), _c("f"), _c("burn"), _c("p"), _c("out", acc="rw"), TILES], """
    let base = ((wlin(wid) >> 1u) * 512u + ((u32(x) % 8u) * 8u + (u32(y) % 8u)) * 8u + (u32(z) % 8u)) * 11u;
    out[base + 6u] = d[i]; out[base + 7u] = t[i]; out[base + 8u] = f[i]; out[base + 9u] = burn[i]; out[base + 10u] = p[i];
""")


# --- device, recording and buffers ----------------------------------------------------------------

_CTX = {}


_NOT_OURS = ("_Ctx", "GpuSmoke3D", "GpuLiquid3D", "GPUDevice", "GPUQueue")


def destroy_buffers(obj, _seen=None):
    """Free every device buffer reachable from `obj` (a grid, a set of bins, a list of them) now, and forget the bind
    groups that cached them. The solvers allocate a new grid whenever the adaptive box changes shape; left to the garbage
    collector the old grids stayed resident (4.5 GB of the card after 24 frames of a 128-cell liquid)."""
    seen = set() if _seen is None else _seen
    if obj is None or id(obj) in seen:
        return 0
    if _seen is None:
        _ctx().flush()          # recorded work that still names these buffers goes out before they do
    seen.add(id(obj))
    kind = type(obj).__name__
    if kind == "GPUBuffer":
        _ctx().defer_destroy(obj)
        return 1
    if kind in _NOT_OURS:
        return 0
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    else:
        return 0
    count = sum(destroy_buffers(child, seen) for child in children)
    if _seen is None:
        _ctx()._groups.clear()
    return count


def _ctx():
    if "ctx" not in _CTX:
        _CTX["ctx"] = _Ctx()
    return _CTX["ctx"]


def available():
    """True when a wgpu compute adapter can be opened here (cached)."""
    try:
        _ctx()
        return True
    except Unsupported:
        return False


def adapter_name():
    return _ctx().name


def estimate_bytes(shape, moving_collider=False):
    n = int(shape[0]) * int(shape[1]) * int(shape[2])
    return 4 * n * 31 + _MG.estimate_bytes(shape) + (4 * n * 4 if moving_collider else 0)


def fits(shape, moving_collider=False, memory_budget=GPU_MEMORY_BUDGET):
    """(True, "") when a GpuSmoke3D of this grid can be built here, else (False, the reason): no adapter, a buffer
    over the adapter's limit, or more than `memory_budget` bytes on the card."""
    try:
        ctx = _ctx()
    except Unsupported as error:
        return False, str(error)
    n = int(shape[0]) * int(shape[1]) * int(shape[2])
    if 16 * n > min(ctx.max_binding, ctx.max_buffer):
        return False, f"a {n:,}-cell grid needs a {16 * n:,}-byte buffer; the adapter allows {min(ctx.max_binding, ctx.max_buffer):,}"
    need = estimate_bytes(shape, moving_collider)
    if need > memory_budget:
        return False, f"about {need / 2 ** 30:.1f} GiB on the card against a budget of {memory_budget / 2 ** 30:.1f} GiB"
    return True, ""


def create_solver(params=None, sparse=False, **kwargs):
    """A GpuSmoke3D, or on `Unsupported` (no compute adapter, grid over the budget) the CPU `Smoke3D` with the reason in
    `.fallback_reason`, so a caller never has to care which one it got."""
    try:
        return GpuSmoke3D(params, sparse=sparse, **kwargs)
    except Unsupported as error:
        LOG.warning("GPU fluid solver unavailable, solving on the CPU: %s", error)
        solver = Smoke3D(params, **{k: v for k, v in kwargs.items() if k in
                                    ("sources", "forces", "colliders", "replace_buoyancy", "cancel")})
        solver.fallback_reason = str(error)
        return solver


class _Ctx:
    """The shared device, compiled pipelines, cached bind groups and the recorded command list."""

    def __init__(self):
        from . import gpu3d
        try:
            state = gpu3d._state()
        except Exception as error:                       # no wgpu package, no adapter, no driver
            raise Unsupported(f"no wgpu compute adapter: {error}") from error
        self.wgpu = state["wgpu"]
        self.device = state["device"]
        info = state["info"]
        self.name = str(info.get("device") or info.get("description") or "wgpu adapter")
        self.kind = str(info.get("adapter_type", "unknown"))
        self.backend = str(info.get("backend_type", "unknown"))
        limits = self.device.limits
        self.max_binding = int(limits["max-storage-buffer-binding-size"])
        self.max_buffer = int(limits.get("max-buffer-size", self.max_binding))
        if int(limits.get("max-storage-buffers-per-shader-stage", 0)) < 8:
            raise Unsupported("the adapter allows fewer than 8 storage buffers per shader stage")
        if int(limits.get("max-compute-invocations-per-workgroup", 0)) < 256:
            raise Unsupported("the adapter allows fewer than 256 invocations per workgroup")
        self.usage = self.wgpu.BufferUsage
        self._pipes = {}
        self._groups = {}
        self.ops = []
        self._graveyard = []
        self.live_bytes = 0

    def pipe(self, name):
        got = self._pipes.get(name)
        if got is None:
            names, src, _ = _KERNELS[name]
            module = self.device.create_shader_module(code=src)
            pipeline = self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
            got = self._pipes[name] = (pipeline, names)
        return got

    def buffer(self, nbytes, indirect=False):
        u = self.usage
        flags = u.STORAGE | u.COPY_DST | u.COPY_SRC | (u.INDIRECT if indirect else 0)
        self.live_bytes += int(nbytes)
        return self.device.create_buffer(size=max(16, (int(nbytes) + 3) // 4 * 4), usage=flags)

    def write(self, buf, array, offset=0):
        self.device.queue.write_buffer(buf, offset, np.ascontiguousarray(array))

    def dispatch(self, kernel, bufs, params, grid):
        """Record one dispatch. `params` = (a0..a3 u32, b0..b3 u32, c0..c3 f32, d0..d3 f32); `grid` = ("wg", (x, y, z))
        or ("indirect", buffer)."""
        import struct
        pipeline, names = self.pipe(kernel)
        blob = struct.pack("<4I4I4f4f", *params)
        key = (kernel, tuple(id(bufs[n]) for n in names), blob)
        got = self._groups.get(key)
        if got is None:
            if len(self._groups) > 6000:
                self._groups.clear()
            uniform = self.device.create_buffer_with_data(data=blob, usage=self.usage.UNIFORM)
            entries = [{"binding": 0, "resource": {"buffer": uniform}}]
            entries += [{"binding": i, "resource": {"buffer": bufs[n]}} for i, n in enumerate(names, start=1)]
            group = self.device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
            got = self._groups[key] = (group, uniform, tuple(bufs[n] for n in names))
        self.ops.append((pipeline, got[0], grid))

    def clear(self, buf):
        self.ops.append((None, buf, None))

    def flush(self):
        if not self.ops:
            return
        ops, self.ops = self.ops, []
        encoder = self.device.create_command_encoder()
        cp = None
        for pipeline, thing, grid in ops:
            if pipeline is None:
                if cp is not None:
                    cp.end()
                    cp = None
                encoder.clear_buffer(thing)
                continue
            if cp is None:
                cp = encoder.begin_compute_pass()
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, thing)
            if grid[0] == "indirect":
                cp.dispatch_workgroups_indirect(grid[1], 0)
            else:
                cp.dispatch_workgroups(*grid[1])
            STATS["dispatches"] += 1
        if cp is not None:
            cp.end()
        self.device.queue.submit([encoder.finish()])
        STATS["submits"] += 1

    def read(self, buf, size, offset=0):
        self.flush()
        STATS["readbacks"] += 1
        dead, self._graveyard = self._graveyard, []
        data = np.frombuffer(self.device.queue.read_buffer(buf, offset, size), np.uint8)
        for old in dead:                # the read waited for every submission before it: nothing reads these any more
            old.destroy()
        return data

    def defer_destroy(self, buf):
        """Free `buf` at the next readback, which waits for every submission before it, so no queued copy or kernel can
        still be reading the buffer when its memory goes back to the card."""
        self._graveyard.append(buf)


def _cdiv(a, b):
    return (a + b - 1) // b


def _u(a=(0, 0, 0, 0), b=(0, 0, 0, 0), c=(0, 0, 0, 0), d=(0, 0, 0, 0)):
    a, b, c, d = (tuple(x) + (0,) * (4 - len(x)) for x in (a, b, c, d))
    return tuple(int(v) for v in a) + tuple(int(v) for v in b) + tuple(float(v) for v in c) + tuple(float(v) for v in d)


def _lin(n, threads=64):
    return ("wg", (max(1, _cdiv(int(n), threads)), 1, 1))


class _MG:
    """The multigrid hierarchy of one grid size: per level q, b, r and the packed weights (wx, wy, wz, dirichlet)."""

    def __init__(self, ctx, dims, q=None, b=None, r=None, wt=None):
        self.ctx = ctx
        self.levels = []
        dims = tuple(int(n) for n in dims)
        first = True
        while True:
            n = dims[0] * dims[1] * dims[2]
            lvl = {"dims": dims,
                   "q": (q if first and q is not None else ctx.buffer(4 * n)),
                   "b": (b if first and b is not None else ctx.buffer(4 * n)),
                   "r": (r if first and r is not None else ctx.buffer(4 * n)),
                   "wt": (wt if first and wt is not None else ctx.buffer(16 * n))}
            self.levels.append(lvl)
            first = False
            if max(dims) <= 4:
                break
            dims = tuple(_cdiv(v, 2) for v in dims)

    @staticmethod
    def estimate_bytes(dims):
        total, dims = 0, tuple(dims)
        while True:
            total += 28 * dims[0] * dims[1] * dims[2]
            if max(dims) <= 4:
                return total
            dims = tuple(_cdiv(v, 2) for v in dims)

    def _grid(self, dims, sparse, args):
        if sparse:
            return ("indirect", args)
        return ("wg", (_cdiv(dims[2], 8), _cdiv(dims[1], 8), _cdiv(dims[0], 4)))

    def coarsen(self, dummy, sparse_tiles=None):
        """Build the weights of every coarse level from level 0's (Galerkin for piecewise-constant aggregation)."""
        for l in range(1, len(self.levels)):
            f, c = self.levels[l - 1], self.levels[l]
            self.ctx.dispatch("coarsen", {"wf": f["wt"], "wc": c["wt"], "tiles": dummy},
                              _u(a=c["dims"] + (0,), b=f["dims"]), self._grid(c["dims"], False, None))

    def smooth(self, l, colour, dummy, tiles=None, args=None):
        lvl = self.levels[l]
        sparse = tiles is not None and l == 0
        self.ctx.dispatch("smooth", {"q": lvl["q"], "b": lvl["b"], "wt": lvl["wt"], "tiles": tiles if sparse else dummy},
                          _u(a=lvl["dims"] + (1 if sparse else 0,), b=(colour,)), self._grid(lvl["dims"], sparse, args))

    def residual(self, l, dummy, tiles=None, args=None):
        lvl = self.levels[l]
        sparse = tiles is not None and l == 0
        self.ctx.dispatch("residual", {"q": lvl["q"], "b": lvl["b"], "wt": lvl["wt"], "r": lvl["r"],
                                       "tiles": tiles if sparse else dummy},
                          _u(a=lvl["dims"] + (1 if sparse else 0,)), self._grid(lvl["dims"], sparse, args))

    def cycle(self, scale, dummy, tiles=None, args=None, l=0):
        ctx = self.ctx
        lvl = self.levels[l]
        if l == len(self.levels) - 1:
            for _ in range(MG_COARSE_SWEEPS):
                self.smooth(l, 0, dummy, tiles, args)
                self.smooth(l, 1, dummy, tiles, args)
            return
        for _ in range(MG_PRE):
            self.smooth(l, 0, dummy, tiles, args)
            self.smooth(l, 1, dummy, tiles, args)
        self.residual(l, dummy, tiles, args)
        nxt = self.levels[l + 1]
        ctx.dispatch("restrict", {"rf": lvl["r"], "bc": nxt["b"], "qc": nxt["q"], "tiles": dummy},
                     _u(a=nxt["dims"] + (0,), b=lvl["dims"]), self._grid(nxt["dims"], False, None))
        self.cycle(scale, dummy, tiles, args, l + 1)
        sparse = tiles is not None and l == 0
        ctx.dispatch("prolong", {"qf": lvl["q"], "qc": nxt["q"], "tiles": tiles if sparse else dummy},
                     _u(a=lvl["dims"] + (1 if sparse else 0,), b=nxt["dims"], c=(scale,)), self._grid(lvl["dims"], sparse, args))
        for _ in range(MG_POST):
            self.smooth(l, 0, dummy, tiles, args)
            self.smooth(l, 1, dummy, tiles, args)


def _correction_scale(open_axes):
    return 1.8 if not any(open_axes) else 1.5


class _Reducer:
    """Deterministic device reductions into `red` slots (a fixed decomposition, so the same inputs sum identically)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.part = ctx.buffer(4 * REDUCE_GROUPS)
        self.red = ctx.buffer(4 * RED_SLOTS)

    def run(self, a, mask, n, mode, slot, base=0.0, masked=False):
        self.ctx.dispatch("reduce1", {"a": a, "m": mask, "part": self.part}, _u(a=(n,), b=(mode, 1 if masked else 0), c=(base,)),
                          ("wg", (REDUCE_GROUPS, 1, 1)))
        self.ctx.dispatch("reduce2", {"part": self.part, "red": self.red}, _u(b=(mode, 0, slot)), ("wg", (1, 1, 1)))

    def value(self, slot):
        return float(self.ctx.read(self.red, 4, 4 * slot).view(np.float32)[0])


class GpuMultigrid3D:
    """`pressure_solver` hook for `fluid3d.Smoke3D`: the same call as `conjugate_gradient`, solved by multigrid on the GPU.

    The first call with a non-trivial residual calibrates the cycle count (`cycles`); later calls run exactly that
    many cycles, plus more only when the residual is still above the tolerance (the count then grows and stays)."""

    def __init__(self):
        self.ctx = _ctx()
        self.adapter_name = self.ctx.name
        self.cycles = 0
        self._shape = None
        self._system = None
        self.last_residual = 0.0

    def _allocate(self, shape):
        ctx = self.ctx
        n = int(np.prod(shape))
        self.mg = _MG(ctx, shape)
        self.dummy = ctx.buffer(4)
        self.blk = ctx.buffer(4 * n)
        self.reducer = _Reducer(ctx)
        self._shape = shape
        self._system = None

    def solve(self, rhs, x0, tolerance, max_iterations, cancel=None, system=None):
        shape = tuple(rhs.shape)
        if self._shape != shape:
            self._allocate(shape)
        ctx, mg = self.ctx, self.mg
        n = int(np.prod(shape))
        if self._system is not system:
            wt = np.zeros(shape + (4,), np.float32)
            for axis, coef in enumerate((system.cx, system.cy, system.cz)):
                key = [slice(None)] * 3
                key[axis] = slice(None, -1)
                wt[tuple(key) + (axis,)] = 1.0 if coef is None else coef
            for axis in range(3):
                if system.ends[axis] is not None:
                    wt[_sl(axis, 0) + (3,)] += system.ends[axis][0]
                    wt[_sl(axis, -1) + (3,)] += system.ends[axis][1]
            diag = wt.sum(axis=-1)
            diag[1:] += wt[:-1, :, :, 0]
            diag[:, 1:] += wt[:, :-1, :, 1]
            diag[:, :, 1:] += wt[:, :, :-1, 2]
            ctx.write(mg.levels[0]["wt"], wt)
            ctx.write(self.blk, (diag <= 0).astype(np.uint32))
            mg.coarsen(self.dummy)
            self._system = system
        ctx.write(mg.levels[0]["q"], np.asarray(x0, np.float32))
        ctx.write(mg.levels[0]["b"], np.asarray(rhs, np.float32))
        scale = _correction_scale(system.open_axes)
        cap = max(1, min(int(max_iterations), MG_MAX_CYCLES))

        def residual():
            mg.residual(0, self.dummy)
            self.reducer.run(mg.levels[0]["r"], self.blk, n, 1, 0, masked=True)
            return self.reducer.value(0)

        res = residual()
        done = 0
        if res > tolerance:
            for _ in range(min(self.cycles, cap)):          # the recorded count, in one submission
                mg.cycle(scale, self.dummy)
                done += 1
            if done:
                res = residual()
            while res > tolerance and done < cap:           # calibration, or a solve that needs more than the record
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                mg.cycle(scale, self.dummy)
                done += 1
                res = residual()
            self.cycles = max(self.cycles, done)
        self.last_residual = res
        q = ctx.read(mg.levels[0]["q"], 4 * n).view(np.float32).reshape(shape).astype(np.float64)
        return q, done, res


# --- the solver -----------------------------------------------------------------------------------

class _OpenSystem:
    """Stands in for Poisson3D: the GPU builds its own operator; `_solid_for` only needs these two facts."""

    def __init__(self, open_faces):
        values = tuple(open_faces)
        self.open_faces = values if len(values) == 6 else tuple(v for a in values for v in (a, a))
        self.open_axes = tuple(self.open_faces[2*a] or self.open_faces[2*a+1] for a in range(3))
        self.singular = not any(self.open_faces)


class GpuState(State):
    """A solved substep that still lives on the card. `arrays` is read back on first use (the active tiles only in
    sparse mode) and cached. Read it before the next `step`: once the solver moves on, an unread state is gone."""

    def __init__(self, solver, token, meta):
        self.meta = dict(meta)
        self._solver = solver
        self.token = token
        self._arrays = None

    @property
    def arrays(self):
        if self._arrays is None:
            self._arrays = self._solver._readback(self.token)
            self._solver = None        # read: the state no longer pins the solver, and with it the card's buffers
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self._arrays = value
        self._solver = None


class _Gpu:
    """The buffers of one GpuSmoke3D on the card."""

    def __init__(self, ctx, solver):
        self.ctx = ctx
        nx, ny, nz = solver.shape
        self.dims = (nx, ny, nz)
        n = nx * ny * nz
        self.n = n
        b = ctx.buffer
        self.sizes = {"u": (nx + 1) * ny * nz, "v": nx * (ny + 1) * nz, "w": nx * ny * (nz + 1)}
        self.f = {"d": b(4 * n), "t": b(4 * n), "f": b(4 * n), "burn": b(4 * n), "p": None,
                  "u": b(4 * self.sizes["u"]), "v": b(4 * self.sizes["v"]), "w": b(4 * self.sizes["w"])}
        self.a = {"d": b(4 * n), "t": b(4 * n), "f": b(4 * n),
                  "u": b(4 * self.sizes["u"]), "v": b(4 * self.sizes["v"]), "w": b(4 * self.sizes["w"])}
        self.pred = b(4 * n)
        self.cx, self.cy, self.cz, self.mag = b(4 * n), b(4 * n), b(4 * n), b(4 * n)
        self.tx = self.ty = self.tz = None
        self.shape_weight = b(4 * n)
        self.shape_force = None
        self.mg = _MG(ctx, self.dims, q=b(4 * n), b=b(4 * n), r=b(4 * n), wt=b(16 * n))
        self.f["p"] = self.mg.levels[0]["q"]
        self.rhs = self.mg.levels[0]["b"]
        self.r0 = self.mg.levels[0]["r"]
        self.wt0 = self.mg.levels[0]["wt"]
        self.solid = b(4 * n)
        self.svel = b(16)
        self.svel_full = False
        self.dummy = b(16)
        self.reducer = _Reducer(ctx)
        self.eidx = b(4 * 64)
        self.eval = b(4 * 8 * 64)
        self.ecap = 64
        self.solid_key = None
        self.sparse = solver.sparse
        if self.sparse:
            self.nt = tuple(_cdiv(v, TILE) for v in self.dims)
            nt = self.nt[0] * self.nt[1] * self.nt[2]
            self.ntiles = nt
            self.act, self.mask, self.oldmask, self.mask2 = b(4 * nt), b(4 * nt), b(4 * nt), b(4 * nt)
            self.ring, self.ring2 = b(4 * nt), b(4 * nt)
            self.ctiles, self.cargs = b(4 * (nt + 16384)), b(16, indirect=True)
            self.tiles, self.rtiles = b(4 * (nt + 16384)), b(4 * (nt + 16384))
            self.args, self.rargs = b(16, indirect=True), b(16, indirect=True)
            self.blk = b(4 * n)
            self.pack = None
            self.pack_cap = 0
        else:
            self.blk = self.solid


class GpuSmoke3D(Smoke3D):
    """`Smoke3D` with the whole substep on the GPU. Same constructor (plus the sparse-tile options), same simcache API.

    `sparse=True` dispatches over active 8-cubed tiles only. `sparse_threshold` (density, temperature above ambient,
    fuel, flame) and `sparse_velocity` (face speed, cells per frame) decide what is active. `profile=True` puts a
    synchronisation point between phases so `phase_seconds` says where the time goes (slower overall)."""

    def __init__(self, params=None, sources=None, forces=(), colliders=(), replace_buoyancy=False, cancel=None,
                 sparse=False, sparse_threshold=DEFAULT_SPARSE_THRESHOLD, sparse_velocity=DEFAULT_SPARSE_VELOCITY,
                 memory_budget=GPU_MEMORY_BUDGET, profile=False):
        super().__init__(params, pressure_solver=None, cancel=cancel, dtype=np.float32, sources=sources, forces=forces,
                         colliders=colliders, replace_buoyancy=replace_buoyancy)
        self.ctx = _ctx()                       # raises Unsupported without an adapter
        self.adapter_name = self.ctx.name
        self.sparse = bool(sparse)
        self.sparse_threshold = float(sparse_threshold)
        self.sparse_velocity = float(sparse_velocity)
        self.profile = bool(profile)
        self.phase_seconds = {}
        n = self.nx * self.ny * self.nz
        need = 4 * n * 31 + _MG.estimate_bytes(self.shape) + (4 * n * 4 if self.colliders else 0)
        if 16 * n > self.ctx.max_binding or 16 * n > self.ctx.max_buffer:
            raise Unsupported(f"a {self.nx} x {self.ny} x {self.nz} grid needs a {16 * n:,}-byte buffer; the adapter allows "
                              f"{min(self.ctx.max_binding, self.ctx.max_buffer):,}")
        if need > memory_budget:
            raise Unsupported(f"a {self.nx} x {self.ny} x {self.nz} grid needs about {need / 2 ** 30:.1f} GiB on the card; "
                              f"the budget is {memory_budget / 2 ** 30:.1f} GiB")
        self.estimated_bytes = need
        LOG.info("GPU fluid solver on %s (%s, %s): %d x %d x %d, %s, about %.2f GiB on the card", self.adapter_name,
                 self.ctx.kind, self.ctx.backend, self.nx, self.ny, self.nz, "sparse tiles" if self.sparse else "dense",
                 need / 2 ** 30)
        self._gpu = None
        self._uid = uuid.uuid4().hex
        self._serial = 0
        self._token = None
        self._pending = False
        self.cycles_total = 0
        self.substeps_done = 0
        self.stored_tiles = None

    # -- the collider system is built on the GPU; only the boundary facts are kept on the host ----------
    def _system(self, solid, key):
        return _OpenSystem(self.open_faces)

    # -- profiling ----------------------------------------------------------------------------------
    def _mark(self, name, since):
        if not self.profile:
            return since
        self._sync()
        now = time.perf_counter()
        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + now - since
        return now

    def _sync(self):
        g = self._gpu
        self.ctx.read(g.reducer.red, 4)

    @property
    def active_tiles(self):
        """Tiles in the mask of the latest substep (None for a dense solver or before the first step)."""
        if not self.sparse or self._gpu is None or self._token is None:
            return None
        return int(self.ctx.read(self._gpu.args, 16).view(np.uint32)[3])

    def release(self):
        """Give the card's buffers back now (a solver that nothing refers to does this from `__del__`)."""
        destroy_buffers(self._gpu)
        self._gpu = None
        self._token = None

    def __del__(self):
        try:
            self.release()
        except Exception:               # interpreter shutdown, device already gone
            pass

    # -- state in and out ---------------------------------------------------------------------------
    def _alloc(self):
        if self._gpu is None:
            self._gpu = _Gpu(self.ctx, self)
        return self._gpu

    def _upload(self, state):
        g = self._alloc()
        ctx = self.ctx
        arrays = state.arrays
        names = {"u": "u", "v": "v", "w": "w", "density": "d", "temperature": "t", "fuel": "f", "burn": "burn",
                 "pressure": "p"}
        for src, dst in names.items():
            data = np.ascontiguousarray(arrays[src], np.float32)
            ctx.write(g.f[dst], data)
            if dst in g.a:
                ctx.write(g.a[dst], data)
        if g.sparse:
            nt = g.ntiles
            mask = arrays.get("tile_mask")
            # the mask the state was solved under (so a resume retires exactly what the run would have), else "all"
            old = (np.ones(nt, np.uint32) if mask is None
                   else np.ascontiguousarray(np.asarray(mask).reshape(-1), np.uint32))
            ctx.write(g.oldmask, old)
            ctx.write(g.mask2, old)
            ctx.write(g.blk, np.full(g.n, 2, np.uint32))
        g.solid_key = None

    def _readback(self, token):
        if token != self._token:
            raise RuntimeError("this GpuState is no longer on the card: read its arrays before the next step")
        g = self._gpu
        ctx = self.ctx
        self._flush_pending()
        ctx.flush()
        nx, ny, nz = self.shape
        shapes = {"u": (nx + 1, ny, nz), "v": (nx, ny + 1, nz), "w": (nx, ny, nz + 1)}
        cells = {"density": "d", "temperature": "t", "fuel": "f", "burn": "burn", "pressure": "p"}
        if not g.sparse:
            out = {}
            for name in ("u", "v", "w"):
                out[name] = ctx.read(g.f[name], 4 * g.sizes[name]).view(np.float32).reshape(shapes[name]).copy()
            for name, key in cells.items():
                out[name] = ctx.read(g.f[key], 4 * g.n).view(np.float32).reshape(self.shape).copy()
            return out
        # Store the active tiles and their neighbours: the faces on the border between an active and an inactive tile
        # belong to the inactive one and carry the outflow into open air.
        ntx, nty, ntz = g.nt
        ctx.dispatch("dilate", {"act": g.oldmask, "mask": g.act}, _u(a=self.shape), _lin(g.ntiles))
        ctx.dispatch("compact", {"ma": g.act, "mb": g.oldmask, "list": g.rtiles, "args": g.rargs},
                     _u(a=(g.ntiles,), b=(0,)), ("wg", (1, 1, 1)))
        count = int(ctx.read(g.rargs, 16).view(np.uint32)[3])
        self.stored_tiles = count
        out = {name: np.zeros(shapes[name], np.float32) for name in ("u", "v", "w")}
        for name in cells:
            out[name] = np.zeros(self.shape, np.float32)
        out["temperature"][...] = np.float32(self.params["ambient_temperature"])
        out["tile_mask"] = ctx.read(g.oldmask, 4 * g.ntiles).view(np.uint32).astype(np.uint8).reshape(g.nt)
        if count == 0:
            return out
        if g.pack_cap < count:
            g.pack = ctx.buffer(4 * 11 * 512 * count)
            g.pack_cap = count
        tiles = ctx.read(g.rtiles, 4 * count).view(np.uint32).copy()
        common = _u(a=self.shape + (1,))
        ctx.dispatch("pack_a", {"u": g.f["u"], "v": g.f["v"], "w": g.f["w"], "out": g.pack, "tiles": g.rtiles}, common,
                     ("indirect", g.rargs))
        ctx.dispatch("pack_b", {"d": g.f["d"], "t": g.f["t"], "f": g.f["f"], "burn": g.f["burn"], "p": g.f["p"],
                                "out": g.pack, "tiles": g.rtiles}, common, ("indirect", g.rargs))
        data = ctx.read(g.pack, 4 * 11 * 512 * count).view(np.float32).reshape(count, 8, 8, 8, 11)
        nty, ntz = g.nt[1], g.nt[2]
        t = tiles.astype(np.int64)
        tz, ty, tx = t % ntz, (t // ntz) % nty, t // (ntz * nty)
        ntx = g.nt[0]
        big = np.empty((ntx, nty, ntz, TILE, TILE, TILE), np.float32)

        def channel(c, rest=0.0):
            big.fill(rest)
            big[tx, ty, tz] = data[..., c]
            dense = big.transpose(0, 3, 1, 4, 2, 5).reshape(ntx * TILE, nty * TILE, ntz * TILE)
            return dense[:nx, :ny, :nz]

        out["u"][:nx] = channel(0)
        out["v"][:, :ny] = channel(1)
        out["w"][:, :, :nz] = channel(2)
        out["u"][nx] = channel(3)[nx - 1]
        out["v"][:, ny] = channel(4)[:, ny - 1]
        out["w"][:, :, nz] = channel(5)[:, :, nz - 1]
        for c, name in enumerate(("density", "temperature", "fuel", "burn", "pressure")):
            out[name][...] = channel(6 + c, float(self.params["ambient_temperature"]) if name == "temperature" else 0.0)
        return out

    def restore(self, state):
        restored = super().restore(state)
        if "tile_mask" in state.arrays:
            restored.arrays["tile_mask"] = np.array(state.arrays["tile_mask"], np.uint8)
        return restored

    def sparse_tiles(self, state):
        """(coords (T, 3) of the active tiles, fields dict name -> (T, 8, 8, 8)) of a solved state, or None when the solver is dense."""
        from .sparsevol import SparseGrid
        return SparseGrid.from_dense({k: state.arrays[k] for k in ("density", "temperature", "fuel", "burn")}, TILE,
                                     rest={"temperature": float(self.params["ambient_temperature"])})

    # -- host-side preparation of one substep -------------------------------------------------------
    def _records(self, frame, dt):
        """Emission lists of every source live at `frame`: (flat cell indices u32, values (M, 8) f32, has_velocity)."""
        out = []
        for source in self.sources:
            if not (source.start_frame <= frame <= source.end_frame):
                continue
            knobs = source._knobs(frame)
            flat, weight, motion = source.footprint(self, frame)
            if not len(flat):
                continue
            if knobs["noise_amount"]:
                from .particles import _value_noise
                ijk = np.stack(np.unravel_index(flat, self.shape), axis=1) + 0.5
                drift = np.array((0.11 * frame, 0.0, 0.0))
                noise = _value_noise((ijk * self.voxel) / max(source.noise_scale, 1e-9) + drift, source.seed, 41)
                weight = weight * np.clip(1.0 + knobs["noise_amount"] * noise, 0.0, None)
            wf = weight.astype(np.float32)
            vals = np.zeros((len(flat), 8), np.float32)
            for c, amount in enumerate((knobs["density"], knobs["temperature"], knobs["fuel"])):
                if amount:
                    vals[:, c] = wf * np.float32(amount * dt)
            target = np.asarray(knobs["velocity"], np.float64) / self.voxel
            inherit = float(knobs["inherit_velocity"])
            has_velocity = bool(target.any() or (inherit and motion is not None))
            if has_velocity:
                vals[:, 3] = np.minimum(weight, 1.0).astype(np.float32)
                for axis in range(3):
                    goal = np.full(len(flat), target[axis])
                    if inherit and motion is not None:
                        goal = goal + inherit * motion[:, axis]
                    vals[:, 4 + axis] = goal.astype(np.float32)
            out.append((np.asarray(flat, np.uint32), vals, has_velocity))
        return out

    def _force_ops(self, frame, substep, dt):
        p, ambient = self.params, float(self.params["ambient_temperature"])
        ops = []
        if not self.replace_buoyancy:
            ops.append(("buoy", float(p["buoyancy_density"]), float(p["buoyancy_temperature"]), ambient))
        turb = None
        for force in self.forces:
            if not force.active(frame):
                continue
            q = force._p(frame)
            kind = force.kind
            if kind == "drag":
                ops.append(("scale", math.exp(-float(q["drag"]) * dt)))
            elif kind in ("gravity", "wind"):
                d = np.array((q["dir_x"], q["dir_y"], q["dir_z"]), np.float64)
                length = np.linalg.norm(d)
                if length < 1e-12:
                    continue
                accel = d / length * float(q["strength"]) / self.voxel * dt
                ops.append(("grav" if kind == "gravity" else "add", tuple(float(a) for a in accel)))
            elif kind == "buoyancy":
                ops.append(("buoy", float(q["buoyancy_settle"]) / self.voxel, float(q["buoyancy_lift"]) / self.voxel,
                            float(q["ambient_temperature"])))
            elif kind == "turbulence":
                t = frame + substep * dt
                field = force._turbulence(self.shape, float(q["turbulence_scale"]) / self.voxel,
                                          t * float(q["turbulence_speed"]))
                field = field * np.float32(float(q["strength"]) / self.voxel * dt)
                if turb is None:
                    turb = field.copy()
                    ops.append(("turb",))
                else:
                    turb += field
        return ops, turb

    def _shape_force(self, kind, frame, substep, dt):
        """Produce the CPU reference's seeded forcing values without reading resident state."""
        p = self.params
        if kind == "disturbance":
            size = max(1.0, float(p["disturbance_size"]))
            blocks = tuple(max(1, int(math.ceil(n / size))) for n in self.shape)
            rng = np.random.default_rng((int(p.get("seed", 0)), int(frame), int(substep), 0x4453))
            kicks = rng.random(blocks + (3,)) * 2.0 - 1.0
            idx = [np.minimum(np.arange(n) // int(size), blocks[axis] - 1)
                   for axis, n in enumerate(self.shape)]
            field = kicks[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
        else:
            from .particles import turbulence_field
            swirl = max(float(p["swirl_size"]), 1e-3)
            grain = max(1, int(p["grain"]))
            pulse = max(float(p["pulse_length"]), 1e-3)
            t = (float(frame) + substep * dt) / pulse
            s0 = math.floor(t)
            tw = fluid3d._smooth(t - s0)
            xs, ys, zs = np.meshgrid(*(np.arange(n) + 0.5 for n in self.shape), indexing="ij")
            positions = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
            seed = int(p.get("seed", 0))
            a = turbulence_field(positions, "curl", swirl, grain, seed + 1013 * s0)
            b = turbulence_field(positions, "curl", swirl, grain, seed + 1013 * (s0 + 1))
            field = (a + (b - a) * tw).reshape(*self.shape, 3)
        return np.moveaxis(field, -1, 0).astype(np.float32) * np.float32(float(p[kind]) * dt)

    def _put_solid(self, g, solid, velocity):
        key = (solid, velocity)             # the arrays themselves: their ids are reused once one is freed
        if isinstance(g.solid_key, tuple) and g.solid_key[0] is solid and g.solid_key[1] is velocity:
            return
        ctx = self.ctx
        ctx.write(g.solid, np.zeros(g.n, np.uint32) if solid is None else np.ascontiguousarray(solid.reshape(-1), np.uint32))
        if velocity is not None:
            if not g.svel_full:
                g.svel = ctx.buffer(16 * g.n)
                g.svel_full = True
            packed = np.zeros((g.n, 4), np.float32)
            packed[:, :3] = velocity.reshape(-1, 3)
            ctx.write(g.svel, packed)
        g.solid_key = key
        g.has_solid = solid is not None
        g.has_svel = velocity is not None

    # -- one substep --------------------------------------------------------------------------------
    def step(self, state, frame=0, substep=0, seed=0):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()
        adaptive = bool(int(self.params.get("auto_resize", 0)))
        if adaptive:
            self._sync_domain(state)
            old_shape = tuple(state.meta.get("domain_shape", self.shape))
            old_origin = tuple(state.meta.get("domain_origin", self.origin))
            prepared = Smoke3D._resize_active_domain(self, state, frame=frame, include_sources=True)
            changed = (tuple(prepared.arrays["density"].shape) != old_shape or
                       tuple(prepared.meta.get("domain_origin", old_origin)) != old_origin)
            self._sync_domain(prepared)
            if changed:
                # The readback above completes the prior dispatch. New dimensions get fresh
                # GPU buffers and tile masks; every extent remains a multiple of the sparse tile.
                destroy_buffers(self._gpu)
                self._gpu = None
                self._token = None
                self._pending = False
                state = prepared
        fluid3d.SOLVER_STATS["steps"] += 1
        t0 = time.perf_counter()
        g = self._alloc()
        ctx = self.ctx
        resident = getattr(state, "token", None) is not None and state.token == self._token
        if not resident:
            self._upload(state)
        meta = dict(state.meta)
        solid, svel, _ = self._solid_for(frame, substep)
        self._put_solid(g, solid, svel)
        dt, p = self.dt, self.params
        ambient = float(p["ambient_temperature"])
        records = self._records(frame, dt)
        ops, turb = self._force_ops(frame, substep, dt)
        total = sum(len(r[0]) for r in records)
        if total > g.ecap:
            g.ecap = max(total, 2 * g.ecap)
            g.eidx, g.eval = ctx.buffer(4 * g.ecap), ctx.buffer(4 * 8 * g.ecap)
        offsets = []
        if records:
            ctx.write(g.eidx, np.concatenate([r[0] for r in records]))
            ctx.write(g.eval, np.concatenate([r[1] for r in records]).reshape(-1))
            at = 0
            for r in records:
                offsets.append((at, len(r[0])))
                at += len(r[0])
        if turb is not None:
            if g.tx is None:
                g.tx, g.ty, g.tz = ctx.buffer(4 * g.n), ctx.buffer(4 * g.n), ctx.buffer(4 * g.n)
            for buf, comp in zip((g.tx, g.ty, g.tz), turb):
                ctx.write(buf, comp)
        t0 = self._mark("host", t0)
        dims = self.shape
        sp = 1 if g.sparse else 0
        grid = ("indirect", g.args) if sp else ("wg", (_cdiv(dims[2], 8), _cdiv(dims[1], 8), _cdiv(dims[0], 4)))
        tiles = g.tiles if sp else g.dummy
        openbits = sum(1 << a for a in range(3) if self.open_axes[a])
        open_face_bits = sum(1 << i for i, opened in enumerate(self.open_faces) if opened)
        maccormack = p["advection"] == "maccormack"
        has_solid = 1 if getattr(g, "has_solid", False) else 0
        has_svel = 1 if getattr(g, "has_svel", False) else 0
        fire = int(p["fire"])

        def cell(kernel, bufs, b=(), c=(), d=()):
            bufs = dict(bufs)
            bufs["tiles"] = tiles
            ctx.dispatch(kernel, bufs, _u(a=dims + (sp,), b=b, c=c, d=d), grid)

        F, A = g.f, g.a
        # 0. tiles
        if sp:
            self._encode_tiles(g, offsets, ambient, dims)
            t0 = self._mark("tiles", t0)
        # 1. emit
        for off, cnt in offsets:
            ctx.dispatch("emit_scalar", {"d": F["d"], "t": F["t"], "f": F["f"], "eidx": g.eidx, "eval": g.eval},
                         _u(a=dims, b=(off, cnt)), _lin(cnt))
        for (off, cnt), rec in zip(offsets, records):
            if rec[2]:
                for side in (0, 1):
                    ctx.dispatch("emit_vel", {"u": F["u"], "v": F["v"], "w": F["w"], "eidx": g.eidx, "eval": g.eval},
                                 _u(a=dims, b=(off, cnt, side)), _lin(cnt))
        t0 = self._mark("emit", t0)
        # 2. advect
        for axis, name in enumerate(("u", "v", "w")):
            cell("advect_vel", {"u": F["u"], "v": F["v"], "w": F["w"], "o": A[name]}, b=(axis,), c=(dt,))
        fresh = {"d": 0.0, "t": ambient, "f": 0.0}
        for k, name in enumerate(("d", "t", "f")):
            cell("advect_pred", {"u": F["u"], "v": F["v"], "w": F["w"], "src": F[name], "pred": g.pred}, c=(dt,))
            cell("advect_final", {"u": F["u"], "v": F["v"], "w": F["w"], "src": F[name], "pred": g.pred, "out": A[name]},
                 b=(1 if maccormack else 0,), c=(dt,))
            if maccormack:
                mask = g.blk if sp else g.dummy
                g.reducer.run(g.pred, mask, g.n, 0, 2 * k, fresh[name], masked=bool(sp))
                g.reducer.run(A[name], mask, g.n, 0, 2 * k + 1, fresh[name], masked=bool(sp))
            if maccormack or openbits:
                cell("scale_apply", {"u": F["u"], "v": F["v"], "w": F["w"], "out": A[name], "red": g.reducer.red},
                     b=(0, open_face_bits, 1 if maccormack else 0, 2 * k), c=(dt, fresh[name]))
        for name in ("d", "t", "f", "u", "v", "w"):
            F[name], A[name] = A[name], F[name]
        F["p"] = g.mg.levels[0]["q"]
        t0 = self._mark("advect", t0)
        # 3-5. combustion, decay, solids
        kd = math.exp(-float(p["dissipation"]) * dt) if p["dissipation"] else 1.0
        kc = math.exp(-float(p["cooling_rate"]) * dt) if p["cooling_rate"] else 1.0
        ineff = min(1.0, max(0.0, float(p.get("fuel_inefficiency", 0.0))))
        lifespan = max(float(p.get("flame_lifespan", 1.0)), 1e-6)
        fraction = (1.0 - math.exp(-float(p["burn_rate"]) * dt / lifespan)) * (1.0 - ineff)
        heat = float(p.get("temperature_output", 2.0))
        smoke = float(p.get("smoke_output", 0.3))
        if heat == 2.0 and float(p["burn_heat"]) != 2.0: heat = float(p["burn_heat"])
        if smoke == 0.3 and float(p["burn_smoke"]) != 0.3: smoke = float(p["burn_smoke"])
        field_names = {"none": 0, "density": 1, "temperature": 2, "speed": 3, "vorticity": 4}

        def shape_weight(prefix):
            field = str(p[prefix + "_field"])
            if field == "none":
                return False
            if field == "vorticity":
                cell("curl", {"u": F["u"], "v": F["v"], "w": F["w"],
                              "cx": g.cx, "cy": g.cy, "cz": g.cz, "mag": g.mag})
            cell("field_weight", {"d": F["d"], "t": F["t"], "u": F["u"], "v": F["v"],
                                  "w": F["w"], "mag": g.mag, "out": g.shape_weight},
                 b=(field_names[field],), c=(float(p[prefix + "_range_lo"]),
                                              float(p[prefix + "_range_hi"]), float(p[prefix + "_ramp"])))
            return True

        limited_decay = bool(p["dissipation"] and p["dissipation_field"] != "none")
        cell("local_phys", {"d": F["d"], "t": F["t"], "f": F["f"], "burn": F["burn"], "solid": g.solid},
             b=(fire, 1 if p["dissipation"] and not limited_decay else 0,
                1 if p["cooling_rate"] and not limited_decay else 0, has_solid),
             c=(float(p["ignition_temperature"]), fraction, heat, smoke), d=(dt, kd, kc, ambient))
        if limited_decay:
            shape_weight("dissipation")
            cell("decay_field", {"d": F["d"], "weight": g.shape_weight}, c=(kd,))
            if p["cooling_rate"]:
                cell("cool_field", {"t": F["t"]}, c=(ambient, kc))
        # The CPU reference applies disturbance before the force list, then shredding and
        # curl noise, then confinement. Each resident force uses the same face averaging.
        def shape_force(kind):
            if not float(p[kind]):
                return
            if g.shape_force is None:
                g.shape_force = tuple(ctx.buffer(4 * g.n) for _ in range(3))
            sx, sy, sz = g.shape_force
            if kind == "shredding":
                cell("shred", {"u": F["u"], "v": F["v"], "w": F["w"],
                               "cx": sx, "cy": sy, "cz": sz}, c=(float(p[kind]) * dt,))
            else:
                ctx.flush()  # queued users of this upload buffer must finish before replacement
                for buf, component in zip((sx, sy, sz), self._shape_force(kind, frame, substep, dt)):
                    ctx.write(buf, component)
                if shape_weight(kind):
                    cell("force_weight", {"cx": sx, "cy": sy, "cz": sz,
                                          "weight": g.shape_weight})
            cell("apply_cforce", {"u": F["u"], "v": F["v"], "w": F["w"], "cx": sx,
                                  "cy": sy, "cz": sz, "solid": g.solid},
                 b=(1 if has_solid else 0,), c=(1.0,))

        # 6. forces
        for op in ops:
            kind = op[0]
            if kind == "buoy":
                cell("buoy", {"v": F["v"], "d": F["d"], "t": F["t"]}, c=(op[1], op[2], op[3], 0.5 * dt))
            elif kind == "grav":
                cell("grav", {"u": F["u"], "v": F["v"], "w": F["w"], "d": F["d"]}, c=op[1])
            elif kind == "add":
                cell("face_op", {"u": F["u"], "v": F["v"], "w": F["w"]}, b=(0,), c=op[1])
            elif kind == "scale":
                cell("face_op", {"u": F["u"], "v": F["v"], "w": F["w"]}, b=(1,), c=(op[1],))
            elif kind == "turb":
                cell("apply_cforce", {"u": F["u"], "v": F["v"], "w": F["w"], "cx": g.tx, "cy": g.ty, "cz": g.tz,
                                      "solid": g.solid}, b=(0,), c=(1.0,))
        t0 = self._mark("forces", t0)
        shape_force("disturbance")
        shape_force("shredding")
        shape_force("turbulence")
        # 7. vorticity confinement
        if float(p["vorticity"]) != 0.0:
            cell("curl", {"u": F["u"], "v": F["v"], "w": F["w"], "cx": g.cx, "cy": g.cy, "cz": g.cz, "mag": g.mag})
            cell("confine", {"cx": g.cx, "cy": g.cy, "cz": g.cz, "mag": g.mag}, c=(float(p["vorticity"]) * dt,))
            cell("apply_cforce", {"u": F["u"], "v": F["v"], "w": F["w"], "cx": g.cx, "cy": g.cy, "cz": g.cz,
                                  "solid": g.solid}, b=(1 if has_solid else 0,), c=(1.0,))
        t0 = self._mark("confine", t0)
        # 8. project
        cell("constrain", {"u": F["u"], "v": F["v"], "w": F["w"], "blk": g.blk, "svel": g.svel},
             b=(open_face_bits, 0, 0, has_solid), c=(float(has_svel),))
        expansion = float(p.get("gas_release", 0.0))
        if expansion == 0.0: expansion = float(p["burn_expansion"])
        cell("rhs_build", {"u": F["u"], "v": F["v"], "w": F["w"], "burn": F["burn"], "blk": g.blk, "rhs": g.rhs, "p": F["p"]},
             c=(expansion if fire else 0.0,))
        singular = not any(self.open_faces)
        mask = g.blk
        cell("weights0", {"blk": g.blk, "wt": g.wt0}, b=(open_face_bits,))
        g.mg.coarsen(g.dummy)
        # A closed box with no open-air face is singular: remove the mean of the right-hand side (and of the pressure
        # below). In sparse mode the inactive space is open air, so the system is singular only while every tile is
        # active and the box is closed; the GPU decides from the total open-air weight.
        mean_mode = 1 if (singular and not sp) else 0
        if singular and sp:
            cell("dw_copy", {"wt": g.wt0, "r": g.r0})
            g.reducer.run(g.r0, mask, g.n, 0, 12)
            mean_mode = 2
        if mean_mode:
            g.reducer.run(g.rhs, mask, g.n, 0, 8, masked=True)
            g.reducer.run(g.rhs, mask, g.n, 2, 9, masked=True)
            cell("sub_mean", {"q": g.rhs, "blk": g.blk, "red": g.reducer.red}, b=(mean_mode, 8, 12))
        t0 = self._mark("setup", t0)
        scale = 1.5 if sp else _correction_scale(self.open_axes)
        args = g.args if sp else None
        cap = MG_MAX_CYCLES if int(p["max_iterations"]) <= 0 else min(MG_MAX_CYCLES, int(p["max_iterations"]))
        tol = float(p["tolerance"])
        cycles_meta = int(meta.get("mg_cycles", 0))

        def residual():
            g.mg.residual(0, g.dummy, g.tiles if sp else None, args)
            g.reducer.run(g.r0, mask, g.n, 1, 0, masked=True)
            return g.reducer.value(0)

        res = residual()
        done = 0
        if res > tol:
            fixed = min(cycles_meta, cap)
            for _ in range(fixed):
                g.mg.cycle(scale, g.dummy, g.tiles if sp else None, args)
                done += 1
            if done:
                res = residual()
            while res > tol and done < cap:
                if self.cancel is not None and self.cancel.is_set():
                    raise Cancelled()
                g.mg.cycle(scale, g.dummy, g.tiles if sp else None, args)
                done += 1
                res = residual()
        cycles_meta = max(cycles_meta, done)
        self.cycles_total += done
        t0 = self._mark("pressure", t0)
        self.pressure_seconds += 0.0
        # finish: zero blocked cells (and remove the mean of a closed box), subtract the gradient. Queued, not flushed:
        # it goes out with the next substep's first submission, or when the arrays are read.
        if mean_mode:
            g.reducer.run(F["p"], mask, g.n, 0, 10, masked=True)
            g.reducer.run(F["p"], mask, g.n, 2, 11, masked=True)
        cell("sub_mean", {"q": F["p"], "blk": g.blk, "red": g.reducer.red}, b=(mean_mode, 10, 12))
        cell("project", {"u": F["u"], "v": F["v"], "w": F["w"], "q": F["p"], "wt": g.wt0, "blk": g.blk},
             b=(open_face_bits,))
        self._pending = True
        t0 = self._mark("finish", t0)
        self._serial += 1
        self._token = (self._uid, self._serial)
        self.substeps_done += 1
        meta.update(substep_count=int(meta.get("substep_count", 0)) + 1, cg_iterations=int(done), cg_residual=float(res),
                    mg_cycles=int(cycles_meta))
        result = GpuState(self, self._token, meta)
        if adaptive:
            # Checkpoints carry the exact dynamic box. If it changes, the following substep
            # uploads this carried state into a fresh dense/sparse GPU allocation.
            resized = Smoke3D._resize_active_domain(self, result, frame=frame, include_sources=False)
            changed = (tuple(resized.arrays["density"].shape) != self.shape or
                       tuple(resized.meta.get("domain_origin", self.origin)) != tuple(self.origin))
            if changed:
                self._sync_domain(resized)
                destroy_buffers(self._gpu)
                self._gpu = None
                self._token = None
                self._pending = False
                return resized
        return result

    def _flush_pending(self):
        self._pending = False

    def _encode_tiles(self, g, offsets, ambient, dims):
        ctx = self.ctx
        ntx, nty, ntz = g.nt
        nt = g.ntiles
        F, A = g.f, g.a
        ctx.clear(g.act)
        ctx.dispatch("tile_activity", {"d": F["d"], "t": F["t"], "f": F["f"], "burn": F["burn"], "u": F["u"], "v": F["v"],
                                       "w": F["w"], "act": g.act},
                     _u(a=dims, c=(self.sparse_threshold, self.sparse_velocity, ambient)), ("wg", (ntz, nty, ntx * 2)))
        for off, cnt in offsets:
            ctx.dispatch("mark_tiles", {"eidx": g.eidx, "act": g.act}, _u(a=dims, b=(off, cnt)), _lin(cnt))
        ctx.dispatch("dilate", {"act": g.act, "mask": g.mask}, _u(a=dims), _lin(nt))
        ctx.dispatch("compact", {"ma": g.mask, "mb": g.oldmask, "list": g.tiles, "args": g.args}, _u(a=(nt,), b=(0,)),
                     ("wg", (1, 1, 1)))
        ctx.dispatch("compact", {"ma": g.oldmask, "mb": g.mask, "list": g.rtiles, "args": g.rargs}, _u(a=(nt,), b=(1,)),
                     ("wg", (1, 1, 1)))
        rest = {"t": ambient}
        jobs = [(F["d"], 0, 0.0), (A["d"], 0, 0.0), (F["t"], 0, ambient), (A["t"], 0, ambient), (F["f"], 0, 0.0),
                (A["f"], 0, 0.0), (F["burn"], 0, 0.0), (F["p"], 0, 0.0), (F["u"], 1, 0.0), (A["u"], 1, 0.0),
                (F["v"], 2, 0.0), (A["v"], 2, 0.0), (F["w"], 3, 0.0), (A["w"], 3, 0.0), (g.pred, 0, 0.0),
                (g.cx, 0, 0.0), (g.cy, 0, 0.0), (g.cz, 0, 0.0), (g.mag, 0, 0.0), (g.r0, 0, 0.0)]
        for buf, kind, value in jobs:
            ctx.dispatch("clear_tiles", {"fld": buf, "tiles": g.rtiles}, _u(a=dims + (1,), b=(kind,), c=(value,)),
                         ("indirect", g.rargs))
        ctx.dispatch("clear_v4", {"fld": g.wt0, "tiles": g.rtiles}, _u(a=dims + (1,)), ("indirect", g.rargs))
        ctx.dispatch("set_blk", {"blk": g.blk, "tiles": g.rtiles}, _u(a=dims + (1,)), ("indirect", g.rargs))
        # Outflow faces that a projection left in inactive tiles next to the mask are only a trigger for activation:
        # the tiles that stay inactive have them cleared, in both velocity sets, so no stale flow survives a swap and
        # a resume from a stored state (which starts without any) matches the run that made it.
        ctx.dispatch("dilate", {"act": g.oldmask, "mask": g.ring}, _u(a=dims), _lin(nt))
        ctx.dispatch("dilate", {"act": g.mask2, "mask": g.ring2}, _u(a=dims), _lin(nt))
        ctx.dispatch("ring_or", {"a": g.ring, "b": g.ring2, "m": g.mask, "out": g.act}, _u(a=(nt,)), _lin(nt))
        ctx.dispatch("compact", {"ma": g.act, "mb": g.mask, "list": g.ctiles, "args": g.cargs}, _u(a=(nt,), b=(0,)),
                     ("wg", (1, 1, 1)))
        for buf, kind in ((F["u"], 1), (A["u"], 1), (F["v"], 2), (A["v"], 2), (F["w"], 3), (A["w"], 3)):
            ctx.dispatch("clear_tiles", {"fld": buf, "tiles": g.ctiles}, _u(a=dims + (1,), b=(kind,), c=(0.0,)),
                         ("indirect", g.cargs))
        g.mask2, g.oldmask, g.mask = g.oldmask, g.mask, g.mask2
        # blocked flags of the tiles now in the list: 0 fluid, 1 solid (retired tiles were set to 2, open air, above)
        ctx.dispatch("mk_blk", {"solid": g.solid, "blk": g.blk, "tiles": g.tiles}, _u(a=dims + (1,)), ("indirect", g.args))
