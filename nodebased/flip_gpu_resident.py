"""GPU-resident FLIP liquid solver (Lane 6, Fluids 7 step N1). See docs/FLUIDS_SPIKE.md, "N1".

`GpuLiquid3D` is `flip3d.Liquid3D` with the whole substep on the card: the particles, the MAC grid, the
validity masks and the extrapolation stay in device buffers between substeps. The host uploads the emission
list of a source (the substep it fires on), the collider mask when it changes and the small force parameters, and
reads back a handful of control numbers per substep (the particle total after maintenance, the largest pressure
residual, the escaped-particle counter for open faces). A `State` is read back only when something asks for its
arrays: a frame checkpoint, the viewport frame, a test. `Liquid3D` stays the reference.

One substep (every stage a compute pass; stage numbers follow the docstring of flip3d.py)
    Bin       particle cell keys by atomic counts, an exclusive scan to cell offsets, a scatter, then a per-cell
              insertion sort by particle id: the order inside a cell is canonical, so every sum below has a fixed
              order and the same inputs give bit-identical particles on the same adapter.
    Maintain  (2) cells over the maximum drop their highest ids; interior cells with fewer than 3 particles are
              topped up with the mean velocity of their neighbours, ids handed out by a scan over the gap cells.
              The jitter of a top-up particle comes from a counter-based hash of (cell, index, frame, substep,
              seed), not from numpy's generator, so a top-up differs from the CPU reference's random draw while
              following the same rule. Source emission (1) is the one stage that stays on the host, using the
              reference's own seeded draws, so a dam break is seeded identically.
              The result is written, sorted by cell, into the work particle set.
    P2G       (3) a gather per face over the occupied 8-cubed tiles (dilated by one tile).
    Forces    (4) gravity, wind, drag, gravity-weighted-by-liquid and turbulence, through the smoke solver's kernels.
    Project   (5, 6) cells are liquid, solid or air; the 7-point system over the liquid cells with p = 0 in air is
              the smoke multigrid's own operator with air as the "open" kind of blocked cell. The cycle count is
              calibrated once and kept in `State.meta["mg_cycles"]`, so a restart from a checkpoint is bit-identical.
    Extrapolate (7) four layers into the air on both velocity sets, ping-ponged between buffers.
    G2P, advect (8, 9) FLIP/PIC blend, midpoint advection with the largest speed reduced on the card.

Transactions: the step writes only the work particle set; the committed set and the previous token stay valid until
the substep has finished, so a cancel inside a substep leaves nothing behind (the next `step` of the same state
repeats the substep from the committed particles).

Not supported on the card (the stream falls back to the `gpu` backend with the reason): viscosity, surface
tension, a narrow band, auto-resize.
"""
from __future__ import annotations

import logging
import time
import uuid

import numpy as np

from . import fluid_gpu_solver as fgs
from . import flip3d
from . import flip_gpu_extrapolate  # noqa: F401  (registers the flip_extrapolate kernel)
from .cancellation import Cancelled
from .flip3d import Liquid3D, LIFE, LIQUID_COLOR, MIN_PER_CELL
from .fluid_gpu_solver import Unsupported, _c, _u, _cdiv
from .simcache import State

LOG = logging.getLogger(__name__)

PART = np.dtype([("px", "<f4"), ("py", "<f4"), ("pz", "<f4"), ("age", "<u4"),
                 ("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4"), ("id", "<u4"), ("temp", "<f4")])
DEAD = 0xFFFFFFFF
ROW = 4194240                          # threads in one full row of a 2-D dispatch of 64-wide workgroups
SLOT_VMAX = 4                          # reducer slot holding the largest particle speed
GPU_MEMORY_BUDGET = 8 << 30
CORRECTION = 1.3                       # coarse-grid over-correction; 1.5 and up diverge on the free-surface operator

fgs._TYPES["au32"] = "atomic<u32>"
fgs._TYPES["part"] = "Particle"

_LIB = """
struct Particle { px: f32, py: f32, pz: f32, age: u32, vx: f32, vy: f32, vz: f32, id: u32, temp: f32 };
fn tocell(w: vec3<f32>) -> vec3<f32> { return (w - vec3<f32>(P.d.x, P.d.y, P.d.z)) * P.d.w; }
fn cellof(w: vec3<f32>) -> u32 {
    let p = tocell(w);
    let x = clamp(i32(floor(p.x)), 0, i32(P.a.x) - 1);
    let y = clamp(i32(floor(p.y)), 0, i32(P.a.y) - 1);
    let z = clamp(i32(floor(p.z)), 0, i32(P.a.z) - 1);
    return u32((x * i32(P.a.y) + y) * i32(P.a.z) + z);
}
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
"""


def _kernel(name, bindings, body, lib="", extra=""):
    fgs._kernel(name, bindings, body, lib=_LIB + lib, cellwise=False, extra=extra)


def _lin(n):
    """A 1-D thread domain of `n` as a (possibly 2-D) dispatch; the shader index is `g.x + g.y * ROW`."""
    groups = max(1, _cdiv(int(n), 64))
    return ("wg", (min(groups, 65535), _cdiv(groups, 65535), 1))


# --- binning ------------------------------------------------------------------------------------------

_kernel("lq_key", [_c("part", "part"), _c("key", "u32", "rw"), _c("counts", "au32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = P.b.x + g.x + g.y * 4194240u;
    if (k >= P.b.y) { return; }
    let q = part[k];
    if (q.id == 0xFFFFFFFFu) { key[k] = 0xFFFFFFFFu; return; }
    let c = cellof(vec3<f32>(q.px, q.py, q.pz));
    key[k] = c;
    atomicAdd(&counts[c], 1u);
}
""")

_kernel("lq_scan_block", [_c("src", "u32"), _c("dst", "u32", "rw"), _c("bsum", "u32", "rw")], """
var<workgroup> sh: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = P.a.x;
    let base = wid.x * 1024u + lid.x * 4u;
    var v0 = 0u; var v1 = 0u; var v2 = 0u; var v3 = 0u;
    if (base < n) { v0 = src[base]; }
    if (base + 1u < n) { v1 = src[base + 1u]; }
    if (base + 2u < n) { v2 = src[base + 2u]; }
    if (base + 3u < n) { v3 = src[base + 3u]; }
    let s = v0 + v1 + v2 + v3;
    sh[lid.x] = s;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
        var add = 0u;
        if (lid.x >= off) { add = sh[lid.x - off]; }
        workgroupBarrier();
        sh[lid.x] = sh[lid.x] + add;
        workgroupBarrier();
    }
    let ex = sh[lid.x] - s;
    if (base < n) { dst[base] = ex; }
    if (base + 1u < n) { dst[base + 1u] = ex + v0; }
    if (base + 2u < n) { dst[base + 2u] = ex + v0 + v1; }
    if (base + 3u < n) { dst[base + 3u] = ex + v0 + v1 + v2; }
    if (lid.x == 255u) { bsum[wid.x] = sh[255]; }
}
""")

_kernel("lq_scan_sums", [_c("bsum", "u32", "rw")], """
var<workgroup> sh: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let nb = P.a.x;
    let per = (nb + 255u) / 256u;
    let lo = lid.x * per;
    let hi = min(lo + per, nb);
    var s = 0u;
    for (var i = lo; i < hi; i++) { s = s + bsum[i]; }
    sh[lid.x] = s;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
        var add = 0u;
        if (lid.x >= off) { add = sh[lid.x - off]; }
        workgroupBarrier();
        sh[lid.x] = sh[lid.x] + add;
        workgroupBarrier();
    }
    var run = sh[lid.x] - s;
    for (var i = lo; i < hi; i++) { let v = bsum[i]; bsum[i] = run; run = run + v; }
    if (lid.x == 255u) { bsum[nb] = sh[255]; }
}
""")

_kernel("lq_scan_add", [_c("dst", "u32", "rw"), _c("bsum", "u32")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 4194240u;
    let n = P.a.x;
    if (i < n) { dst[i] = dst[i] + bsum[i / 1024u]; }
    else if (i == n) { dst[n] = bsum[P.b.x]; }
}
""")

_kernel("lq_scatter", [_c("key", "u32"), _c("offsets", "u32"), _c("cursor", "au32", "rw"), _c("order", "u32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let k = g.x + g.y * 4194240u;
    if (k >= P.b.y) { return; }
    let c = key[k];
    if (c == 0xFFFFFFFFu) { return; }
    let slot = atomicAdd(&cursor[c], 1u);
    order[offsets[c] + slot] = k;
}
""")

_kernel("lq_cellsort", [_c("offsets", "u32"), _c("order", "u32", "rw"), _c("part", "part")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x + g.y * 4194240u;
    if (c >= P.a.w) { return; }
    let b = offsets[c];
    let e = offsets[c + 1u];
    if (e - b < 2u) { return; }
    for (var i = b + 1u; i < e; i++) {
        let kk = order[i];
        let id = part[kk].id;
        var j = i;
        loop {
            if (j <= b) { break; }
            let prev = order[j - 1u];
            if (part[prev].id <= id) { break; }
            order[j] = prev;
            j = j - 1u;
        }
        order[j] = kk;
    }
}
""")

_kernel("lq_cellstats", [_c("offsets", "u32"), _c("order", "u32"), _c("part", "part"), _c("cnt2", "u32", "rw"),
                         _c("vsum", "v4", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x + g.y * 4194240u;
    if (c >= P.a.w) { return; }
    let b = offsets[c];
    let cnt = min(offsets[c + 1u] - b, P.b.x);
    var s = vec4<f32>(0.0);
    for (var i = 0u; i < cnt; i++) {
        let q = part[order[b + i]];
        s = s + vec4<f32>(q.vx, q.vy, q.vz, q.temp);
    }
    cnt2[c] = cnt;
    vsum[c] = s;
}
""")

_NEIGHBOURS = """
fn nbr(c: u32, a: u32, s: u32) -> i32 {
    // neighbour cell of c along axis a (side 0 low, 1 high), or -1 outside the domain
    let nx = P.a.x; let ny = P.a.y; let nz = P.a.z;
    let z = c % nz; let y = (c / nz) % ny; let x = c / (nz * ny);
    if (a == 0u) { if (s == 0u) { if (x == 0u) { return -1; } return i32(c - ny * nz); } if (x + 1u >= nx) { return -1; } return i32(c + ny * nz); }
    if (a == 1u) { if (s == 0u) { if (y == 0u) { return -1; } return i32(c - nz); } if (y + 1u >= ny) { return -1; } return i32(c + nz); }
    if (s == 0u) { if (z == 0u) { return -1; } return i32(c - 1u); }
    if (z + 1u >= nz) { return -1; }
    return i32(c + 1u);
}
"""

_kernel("lq_gapneed", [_c("cnt2", "u32"), _c("solid", "u32"), _c("need", "u32", "rw"), _c("newcnt", "u32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x + g.y * 4194240u;
    if (c >= P.a.w) { return; }
    let n0 = cnt2[c];
    var gap = n0 < P.b.y && n0 > 0u && solid[c] == 0u;
    if (gap) {
        for (var a = 0u; a < 3u; a++) { for (var s = 0u; s < 2u; s++) {
            let j = nbr(c, a, s);
            if (j >= 0) { if (cnt2[u32(j)] == 0u && solid[u32(j)] == 0u) { gap = false; } }
        } }
    }
    var nd = 0u;
    if (gap) { nd = P.b.y - n0; }
    need[c] = nd;
    newcnt[c] = n0 + nd;
}
""", lib=_NEIGHBOURS)

_kernel("lq_totals", [_c("offsets", "u32"), _c("needoff", "u32"), _c("newoff", "u32"), _c("totals", "u32", "rw")], """
@compute @workgroup_size(1)
fn main() {
    let n = P.a.w;
    totals[0] = needoff[n];
    totals[1] = newoff[n];
    totals[2] = offsets[n];
}
""")

_kernel("lq_compact", [_c("key", "u32"), _c("order", "u32"), _c("offsets", "u32"), _c("newoff", "u32"),
                       _c("src", "part"), _c("dst", "part", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let s = g.x + g.y * 4194240u;
    if (s >= P.b.y || s >= offsets[P.a.w]) { return; }
    let k = order[s];
    let c = key[k];
    let rank = s - offsets[c];
    if (rank >= P.b.x) { return; }
    dst[newoff[c] + rank] = src[k];
}
""")

_kernel("lq_gapgen", [_c("cnt2", "u32"), _c("vsum", "v4"), _c("need", "u32"), _c("needoff", "u32"), _c("newoff", "u32"),
                      _c("dst", "part", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x + g.y * 4194240u;
    if (c >= P.a.w) { return; }
    let nd = need[c];
    if (nd == 0u) { return; }
    var total = vec4<f32>(0.0);
    var count = 0.0;
    for (var a = 0u; a < 3u; a++) { for (var s = 0u; s < 2u; s++) {
        let j = nbr(c, a, s);
        if (j >= 0) { total = total + vsum[u32(j)]; count = count + f32(cnt2[u32(j)]); }
    } }
    let mean = total / max(count, 1.0);
    let z = c % P.a.z; let y = (c / P.a.z) % P.a.y; let x = c / (P.a.z * P.a.y);
    for (var j = 0u; j < nd; j++) {
        let h = pcg(pcg(pcg(c ^ pcg(P.b.z)) + j) ^ P.b.w);
        let u = vec3<f32>(unit(pcg(h)), unit(pcg(h + 1u)), unit(pcg(h + 2u)));
        let w = (vec3<f32>(f32(x), f32(y), f32(z)) + u) * P.c.w + vec3<f32>(P.d.x, P.d.y, P.d.z);
        var q: Particle;
        q.px = w.x; q.py = w.y; q.pz = w.z; q.age = 0u;
        q.vx = mean.x; q.vy = mean.y; q.vz = mean.z; q.temp = mean.w;
        q.id = P.b.x + needoff[c] + j;
        dst[newoff[c] + cnt2[c] + j] = q;
    }
}
""", lib=_NEIGHBOURS)

_kernel("lq_gather", [_c("src", "u32"), _c("idx", "u32"), _c("out", "u32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 4194240u;
    if (i >= P.b.x) { return; }
    out[i] = src[idx[i]];
}
""")

# --- grid stages --------------------------------------------------------------------------------------

_kernel("lq_classify", [_c("newcnt", "u32"), _c("solid", "u32"), _c("dens", "f32", "rw"), _c("blk", "u32", "rw"),
                        _c("tmark", "au32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x + g.y * 4194240u;
    if (c >= P.a.w) { return; }
    let liquid = newcnt[c] > 0u && solid[c] == 0u;
    dens[c] = select(0.0, 1.0, liquid);
    blk[c] = select(select(2u, 0u, liquid), 1u, solid[c] != 0u);
    if (newcnt[c] > 0u) {
        let z = c % P.a.z; let y = (c / P.a.z) % P.a.y; let x = c / (P.a.z * P.a.y);
        let nty = (P.a.y + 7u) / 8u; let ntz = (P.a.z + 7u) / 8u;
        atomicStore(&tmark[((x / 8u) * nty + y / 8u) * ntz + z / 8u], 1u);
    }
}
""")


def _p2g(axis):
    name = "xyz"[axis]
    point = [("f32(c.x)", "f32(c.y) + 0.5", "f32(c.z) + 0.5"),
             ("f32(c.x) + 0.5", "f32(c.y)", "f32(c.z) + 0.5"),
             ("f32(c.x) + 0.5", "f32(c.y) + 0.5", "f32(c.z)")][axis]
    hi = list(point)
    hi[axis] = f"f32(c.{name} + 1)"
    sy = "(i32(P.a.y) + 1)" if axis == 1 else "i32(P.a.y)"
    sz = "(i32(P.a.z) + 1)" if axis == 2 else "i32(P.a.z)"
    lo_index = f"((c.x * {sy} + c.y) * {sz} + c.z)"
    hi_index = lo_index.replace(f"c.{name}", f"(c.{name} + 1)")
    return f"""
fn flat(x: i32, y: i32, z: i32) -> u32 {{ return u32((x * i32(P.a.y) + y) * i32(P.a.z) + z); }}
fn weighted(f: vec3<f32>) -> vec2<f32> {{
    let lo = max(vec3<i32>(0), vec3<i32>(floor(f - vec3<f32>(1.0))));
    let hi = min(vec3<i32>(i32(P.a.x) - 1, i32(P.a.y) - 1, i32(P.a.z) - 1),
                 vec3<i32>(ceil(f + vec3<f32>(1.0))) - vec3<i32>(1));
    var num = 0.0; var den = 0.0;
    for (var x = lo.x; x <= hi.x; x++) {{ for (var y = lo.y; y <= hi.y; y++) {{ for (var z = lo.z; z <= hi.z; z++) {{
        let cc = flat(x, y, z);
        let end = offs[cc + 1u];
        for (var k = offs[cc]; k < end; k++) {{
            let q = part[k];
            let p = tocell(vec3<f32>(q.px, q.py, q.pz));
            let w = max(0.0, 1.0 - abs(p.x - f.x)) * max(0.0, 1.0 - abs(p.y - f.y)) * max(0.0, 1.0 - abs(p.z - f.z));
            num = num + w * (q.v{name} * P.d.w);
            den = den + w;
        }}
    }} }} }}
    return vec2<f32>(select(0.0, num / max(den, 1e-20), den > 1e-20), den);
}}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let k = g.x + g.y * 4194240u;
    let ti = k / 512u;
    let ntz = (P.a.z + 7u) / 8u; let nty = (P.a.y + 7u) / 8u; let ntx = (P.a.x + 7u) / 8u;
    if (ti >= ntx * nty * ntz) {{ return; }}
    if (tmask[ti] == 0u) {{ return; }}
    let lane = k % 512u;
    let tx = ti / (nty * ntz); let ty = (ti / ntz) % nty; let tz = ti % ntz;
    let c = vec3<i32>(i32(tx * 8u + lane / 64u), i32(ty * 8u + (lane / 8u) % 8u), i32(tz * 8u + lane % 8u));
    if (c.x >= i32(P.a.x) || c.y >= i32(P.a.y) || c.z >= i32(P.a.z)) {{ return; }}
    let f = weighted(vec3<f32>({', '.join(point)}));
    let li = u32({lo_index});
    grid[li] = f.x; old[li] = f.x; valid[li] = select(0u, 1u, f.y > 1e-20);
    if (c.{name} == i32(P.a.{name}) - 1) {{
        let h = weighted(vec3<f32>({', '.join(hi)}));
        let hi_i = u32({hi_index});
        grid[hi_i] = h.x; old[hi_i] = h.x; valid[hi_i] = select(0u, 1u, h.y > 1e-20);
    }}
}}
"""


for _axis, _name in enumerate("uvw"):
    _kernel(f"lq_p2g_{_name}", [_c("part", "part"), _c("offs", "u32"), _c("tmask", "u32"), _c("grid", "f32", "rw"),
                                _c("old", "f32", "rw"), _c("valid", "u32", "rw")], _p2g(_axis))

fgs._kernel("lq_rhs", [_c("u"), _c("v"), _c("w"), _c("blk", "u32"), _c("rhs", acc="rw"), _c("q", acc="rw"), fgs.TILES], """
    q[i] = 0.0;
    if (blk[i] != 0u) { rhs[i] = 0.0; return; }
    let div = (u[ic(x + 1, y, z)] - u[ic(x, y, z)]) + (v[iv(x, y + 1, z)] - v[iv(x, y, z)])
            + (w[iw(x, y, z + 1)] - w[iw(x, y, z)]);
    rhs[i] = -div;
""")

_kernel("lq_project", [_c("f", acc="rw"), _c("q"), _c("blk", "u32")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 4194240u;
    if (i >= P.b.w) { return; }
    let axis = P.b.x;
    let nx = P.a.x; let ny = P.a.y; let nz = P.a.z;
    var fy = ny; var fz = nz;
    if (axis == 1u) { fy = ny + 1u; }
    if (axis == 2u) { fz = nz + 1u; }
    let z = i % fz; let y = (i / fz) % fy; let x = i / (fz * fy);
    var pos = x; var n = nx; var stride = ny * nz;
    if (axis == 1u) { pos = y; n = ny; stride = nz; }
    if (axis == 2u) { pos = z; n = nz; stride = 1u; }
    // the two cells either side of this face: hi is the cell with the same index as the face
    let hc = (x * ny + y) * nz + z;
    if (pos == 0u) {
        if ((P.b.y & (1u << (2u * axis))) != 0u) { f[i] = f[i] + q[hc]; }
        return;
    }
    if (pos == n) {
        let lc = hc - stride;
        if ((P.b.y & (2u << (2u * axis))) != 0u) { f[i] = f[i] - q[lc]; }
        return;
    }
    let lc = hc - stride;
    let bl = blk[lc]; let bh = blk[hc];
    if (bl != 1u && bh != 1u && (bl == 0u || bh == 0u)) { f[i] = f[i] - (q[hc] - q[lc]); }
}
""")

_kernel("lq_touched", [_c("blk", "u32"), _c("f", acc="rw"), _c("o", acc="rw"), _c("vo", "u32", "rw"),
                       _c("vt", "u32", "rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 4194240u;
    if (i >= P.b.w) { return; }
    let axis = P.b.x;
    let nx = P.a.x; let ny = P.a.y; let nz = P.a.z;
    var fy = ny; var fz = nz;
    if (axis == 1u) { fy = ny + 1u; }
    if (axis == 2u) { fz = nz + 1u; }
    let z = i % fz; let y = (i / fz) % fy; let x = i / (fz * fy);
    var pos = x; var n = nx; var stride = ny * nz;
    if (axis == 1u) { pos = y; n = ny; stride = nz; }
    if (axis == 2u) { pos = z; n = nz; stride = 1u; }
    var touched = true;
    if (pos != 0u && pos != n) {
        let hc = (x * ny + y) * nz + z;
        touched = blk[hc] != 2u || blk[hc - stride] != 2u;
    }
    let old_ok = touched || vo[i] != 0u;
    vt[i] = select(0u, 1u, touched);
    vo[i] = select(0u, 1u, old_ok);
    f[i] = select(0.0, f[i], touched);
    o[i] = select(0.0, o[i], old_ok);
}
""")

_TRI = (fgs._tri("u", "i32(P.a.x) + 1", "i32(P.a.y)", "i32(P.a.z)")
        + fgs._tri("v", "i32(P.a.x)", "i32(P.a.y) + 1", "i32(P.a.z)")
        + fgs._tri("w", "i32(P.a.x)", "i32(P.a.y)", "i32(P.a.z) + 1"))
_TRI_OLD = (fgs._tri("ou", "i32(P.a.x) + 1", "i32(P.a.y)", "i32(P.a.z)")
            + fgs._tri("ov", "i32(P.a.x)", "i32(P.a.y) + 1", "i32(P.a.z)")
            + fgs._tri("ow", "i32(P.a.x)", "i32(P.a.y)", "i32(P.a.z) + 1"))
_SAMPLE = """
fn sample(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(tri_u(p - vec3<f32>(0.0, 0.5, 0.5)), tri_v(p - vec3<f32>(0.5, 0.0, 0.5)), tri_w(p - vec3<f32>(0.5, 0.5, 0.0)));
}
"""
_SAMPLE_OLD = """
fn sample_old(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(tri_ou(p - vec3<f32>(0.0, 0.5, 0.5)), tri_ov(p - vec3<f32>(0.5, 0.0, 0.5)), tri_ow(p - vec3<f32>(0.5, 0.5, 0.0)));
}
"""

_kernel("lq_g2p", [_c("part", "part", "rw"), _c("u"), _c("v"), _c("w"), _c("ou"), _c("ov"), _c("ow")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 4194240u;
    if (i >= P.b.x) { return; }
    var q = part[i];
    let p = tocell(vec3<f32>(q.px, q.py, q.pz));
    let oldv = vec3<f32>(q.vx, q.vy, q.vz) * P.d.w;
    let pic = sample(p);
    let flip = oldv + pic - sample_old(p);
    let vnew = P.c.y * flip + (1.0 - P.c.y) * pic;
    q.vx = vnew.x * P.c.w; q.vy = vnew.y * P.c.w; q.vz = vnew.z * P.c.w;
    part[i] = q;
}
""", lib=_TRI + _TRI_OLD + _SAMPLE + _SAMPLE_OLD)

_kernel("lq_vmax", [_c("part", "part"), _c("out", acc="rw")], """
var<workgroup> wg: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = P.b.x;
    var acc = 0.0;
    var i = wid.x * 256u + lid.x;
    loop {
        if (i >= n) { break; }
        let q = part[i];
        acc = max(acc, max(abs(q.vx), max(abs(q.vy), abs(q.vz))));
        i = i + 1024u * 256u;
    }
    wg[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { wg[lid.x] = max(wg[lid.x], wg[lid.x + s]); }
        workgroupBarrier();
    }
    if (lid.x == 0u) { out[wid.x] = wg[0]; }
}
""")

_kernel("lq_advect", [_c("part", "part", "rw"), _c("u"), _c("v"), _c("w"), _c("red"), _c("solid", "u32"),
                      _c("esc", "au32", "rw")], """
fn limit(raw: vec3<f32>) -> vec3<f32> {
    let hi = vec3<f32>(f32(P.a.x) - 0.001, f32(P.a.y) - 0.001, f32(P.a.z) - 0.001);
    var q = clamp(raw, vec3<f32>(0.001), hi);
    if ((P.b.y & 1u) != 0u && raw.x < 0.0) { q.x = raw.x; }
    if ((P.b.y & 2u) != 0u && raw.x >= f32(P.a.x)) { q.x = raw.x; }
    if ((P.b.y & 4u) != 0u && raw.y < 0.0) { q.y = raw.y; }
    if ((P.b.y & 8u) != 0u && raw.y >= f32(P.a.y)) { q.y = raw.y; }
    if ((P.b.y & 16u) != 0u && raw.z < 0.0) { q.z = raw.z; }
    if ((P.b.y & 32u) != 0u && raw.z >= f32(P.a.z)) { q.z = raw.z; }
    return q;
}
fn solid_at(p: vec3<f32>) -> bool {
    let x = clamp(i32(floor(p.x)), 0, i32(P.a.x) - 1);
    let y = clamp(i32(floor(p.y)), 0, i32(P.a.y) - 1);
    let z = clamp(i32(floor(p.z)), 0, i32(P.a.z) - 1);
    return solid[u32((x * i32(P.a.y) + y) * i32(P.a.z) + z)] != 0u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 4194240u;
    if (i >= P.b.x) { return; }
    var q = part[i];
    var pos = tocell(vec3<f32>(q.px, q.py, q.pz));
    let dt = P.c.x;
    let vmax = red[P.b.z] * P.d.w;
    let steps = i32(clamp(ceil(vmax * dt / 0.9), 1.0, 6.0));
    let h = dt / f32(steps);
    let hi = vec3<f32>(f32(P.a.x) - 0.001, f32(P.a.y) - 0.001, f32(P.a.z) - 0.001);
    for (var s = 0; s < steps; s++) {
        let v1 = sample(pos);
        let mid = clamp(pos + 0.5 * h * v1, vec3<f32>(0.001), hi);
        let v2 = sample(mid);
        let raw = pos + h * v2;
        var nw = limit(raw);
        if (solid_at(nw)) { nw = pos; }
        pos = nw;
    }
    var gone = false;
    if ((P.b.y & 1u) != 0u && pos.x < 0.0) { gone = true; }
    if ((P.b.y & 2u) != 0u && pos.x >= f32(P.a.x)) { gone = true; }
    if ((P.b.y & 4u) != 0u && pos.y < 0.0) { gone = true; }
    if ((P.b.y & 8u) != 0u && pos.y >= f32(P.a.y)) { gone = true; }
    if ((P.b.y & 16u) != 0u && pos.z < 0.0) { gone = true; }
    if ((P.b.y & 32u) != 0u && pos.z >= f32(P.a.z)) { gone = true; }
    if (gone) { q.id = 0xFFFFFFFFu; atomicAdd(&esc[0], 1u); }
    else { q.age = q.age + 1u; q.temp = q.temp * P.c.z; }
    let w = pos * P.c.w + vec3<f32>(P.d.x, P.d.y, P.d.z);
    q.px = w.x; q.py = w.y; q.pz = w.z;
    part[i] = q;
}
""", lib=_TRI + _SAMPLE)


def _scan(ctx, src, dst, bsum, n):
    blocks = _cdiv(n, 1024)
    ctx.dispatch("lq_scan_block", {"src": src, "dst": dst, "bsum": bsum}, _u(a=(n,)), ("wg", (blocks, 1, 1)))
    ctx.dispatch("lq_scan_sums", {"bsum": bsum}, _u(a=(blocks,)), ("wg", (1, 1, 1)))
    ctx.dispatch("lq_scan_add", {"dst": dst, "bsum": bsum}, _u(a=(n,), b=(blocks,)), _lin(n + 1))


def _grow(ctx, buf, old_bytes, new_bytes):
    """A larger buffer holding the first `old_bytes` of `buf`."""
    new = ctx.buffer(new_bytes)
    if old_bytes:
        ctx.flush()
        encoder = ctx.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(buf, 0, new, 0, old_bytes)
        ctx.device.queue.submit([encoder.finish()])
    ctx.live_bytes -= old_bytes
    return new


_STAGING = {}


def read_staged(ctx, buf, size):
    """`size` bytes of `buf` as a uint8 array through a persistent mappable staging buffer (the one-shot
    `queue.read_buffer` allocates and maps a fresh staging buffer per call and was 5 times slower on 70 MB)."""
    ctx.flush()
    wgpu = ctx.wgpu
    staging = _STAGING.get(id(ctx.device))
    if staging is None or staging.size < size:
        staging = ctx.device.create_buffer(size=max(int(size), 1 << 20) + 4095 & ~4095,
                                           usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
        _STAGING[id(ctx.device)] = staging
    encoder = ctx.device.create_command_encoder()
    encoder.copy_buffer_to_buffer(buf, 0, staging, 0, (size + 3) // 4 * 4)
    ctx.device.queue.submit([encoder.finish()])
    staging.map_sync(wgpu.MapMode.READ)
    try:
        data = np.frombuffer(staging.read_mapped(0, (size + 3) // 4 * 4, copy=True), np.uint8)[:size]
    finally:
        staging.unmap()
    return data


class Bins:
    """Cell-sorted particle lists of a device particle buffer on one grid: `counts`, the exclusive scan `offsets`
    (cells + 1 entries) and `order` (slot numbers, ascending particle id inside a cell)."""

    def __init__(self, ctx, shape):
        self.ctx = ctx
        self.shape = tuple(int(n) for n in shape)
        cells = self.shape[0] * self.shape[1] * self.shape[2]
        self.cells = cells
        self.counts = ctx.buffer(4 * cells)
        self.cursor = ctx.buffer(4 * cells)
        self.offsets = ctx.buffer(4 * (cells + 1))
        self.bsum = ctx.buffer(4 * (_cdiv(cells + 1, 1024) + 2))
        self.cap = 0
        self.key = None
        self.order = None

    def ensure(self, n):
        if n <= self.cap:
            return
        cap = max(n, int(self.cap * 1.5), 1024)
        self.key = _grow(self.ctx, self.key, 4 * self.cap, 4 * cap)
        self.order = _grow(self.ctx, self.order, 4 * self.cap, 4 * cap)
        self.cap = cap

    def begin(self):
        self.ctx.clear(self.counts)
        self.ctx.clear(self.cursor)

    def count(self, part, start, end, common):
        """Key and count the particles of slots [start, end)."""
        if end > start:
            self.ctx.dispatch("lq_key", {"part": part, "key": self.key, "counts": self.counts},
                              _u(a=self.shape, b=(start, end), c=common[0], d=common[1]), _lin(end - start))

    def sort(self, part, n, common):
        ctx = self.ctx
        _scan(ctx, self.counts, self.offsets, self.bsum, self.cells)
        ctx.dispatch("lq_scatter", {"key": self.key, "offsets": self.offsets, "cursor": self.cursor, "order": self.order},
                     _u(a=self.shape, b=(0, n)), _lin(n))
        ctx.dispatch("lq_cellsort", {"offsets": self.offsets, "order": self.order, "part": part},
                     _u(a=self.shape + (self.cells,)), _lin(self.cells))


class _Grid:
    """The device buffers of one grid size."""

    def __init__(self, ctx, shape):
        nx, ny, nz = self.shape = shape
        n = self.n = nx * ny * nz
        b = ctx.buffer
        self.face_shapes = ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))
        self.face_n = [int(np.prod(s)) for s in self.face_shapes]
        self.f = [b(4 * m) for m in self.face_n]
        self.old = [b(4 * m) for m in self.face_n]
        self.valid_new = [b(4 * m) for m in self.face_n]
        self.valid_old = [b(4 * m) for m in self.face_n]
        self.scratch = [b(4 * m) for m in self.face_n]
        self.scratch_valid = [b(4 * m) for m in self.face_n]
        self.cnt2, self.need, self.newcnt = b(4 * n), b(4 * n), b(4 * n)
        self.vsum = b(16 * n)
        self.needoff, self.newoff = b(4 * (n + 1)), b(4 * (n + 1))
        self.bsum = b(4 * (_cdiv(n + 1, 1024) + 2))
        self.totals = b(16)
        self.dens, self.blk, self.solid = b(4 * n), b(4 * n), b(4 * n)
        self.nt = tuple(_cdiv(v, 8) for v in shape)
        tiles = self.nt[0] * self.nt[1] * self.nt[2]
        self.tiles = tiles
        self.tmark, self.tmask = b(4 * tiles), b(4 * tiles)
        self.mg = fgs._MG(ctx, shape, q=b(4 * n), b=b(4 * n), r=b(4 * n), wt=b(16 * n))
        self.q, self.rhs, self.r0, self.wt0 = (self.mg.levels[0][k] for k in ("q", "b", "r", "wt"))
        self.dummy = b(16)
        self.svel = b(16)
        self.svel_full = False
        self.solid_key = "unset"
        self.has_solid = False
        self.has_svel = False
        self.reducer = fgs._Reducer(ctx)
        self.esc = b(16)
        self.gather_idx = b(4 * 1024)
        self.gather_out = b(4 * 1024)
        self.gather_cap = 1024
        self.turb = None


def estimate_bytes(shape, particles=0):
    n = int(shape[0]) * int(shape[1]) * int(shape[2])
    return 4 * n * 56 + fgs._MG.estimate_bytes(shape) + 2 * 36 * int(particles) + 8 * int(particles)


def unsupported_reason(params, forces=()):
    """None when the resident solver can run these parameters and forces, else a sentence saying why not."""
    p = {**flip3d.DEFAULTS, **params}
    if float(p["viscosity"]) > 0.0:
        return "viscosity is not on the card yet"
    if float(p.get("surface_tension", 0.0)) != 0.0:
        return "surface tension is not on the card yet"
    if float(p.get("narrow_band", 0.0)) > 0.0:
        return "the narrow band is not on the card yet"
    if int(p.get("auto_resize", 0)):
        return "auto-resize is not on the card yet"
    for force in forces:
        if force.kind not in ("gravity", "wind", "drag", "buoyancy", "turbulence"):
            return f"the {force.kind} force is not on the card"
    return None


class GpuLiquidState(State):
    """A solved substep that still lives on the card. `arrays` is read back on first use and cached; read it before
    the next `step` of the solver (a cache's `put` does)."""

    def __init__(self, solver, token, meta):
        self.meta = dict(meta)
        self._solver = solver
        self.token = token
        self._arrays = None

    @property
    def arrays(self):
        if self._arrays is None:
            self._arrays = self._solver._readback(self.token)
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self._arrays = value


class GpuLiquid3D(Liquid3D):
    """`Liquid3D` with the whole substep on the GPU. Same constructor and simcache API; opt-in."""

    def __init__(self, params=None, pressure_solver=None, cancel=None, sources=None, forces=(), colliders=(),
                 memory_budget=GPU_MEMORY_BUDGET, profile=False):
        super().__init__(params, None, cancel, sources, forces, colliders)
        reason = unsupported_reason(self.params, self.forces)
        if reason:
            raise Unsupported(reason)
        self.ctx = fgs._ctx()
        self.adapter_name = self.ctx.name
        n = self.nx * self.ny * self.nz
        need = estimate_bytes(self.shape)
        if 16 * n > min(self.ctx.max_binding, self.ctx.max_buffer):
            raise Unsupported(f"a {self.nx} x {self.ny} x {self.nz} grid needs a {16 * n:,}-byte buffer; the adapter "
                              f"allows {min(self.ctx.max_binding, self.ctx.max_buffer):,}")
        if need > memory_budget:
            raise Unsupported(f"a {self.nx} x {self.ny} x {self.nz} grid needs about {need / 2 ** 30:.1f} GiB on the "
                              f"card; the budget is {memory_budget / 2 ** 30:.1f} GiB")
        self.backend = "resident"
        self.profile = bool(profile)
        self.phase_seconds = {}
        self.copy_seconds = {"host_to_device": 0.0, "device_to_host": 0.0}
        self._grid = None
        self._bins = None
        self._sets = [None, None]
        self._caps = [0, 0]
        self._cur = 0
        self._n_slots = 0
        self._n_live = 0
        self._uid = uuid.uuid4().hex
        self._serial = 0
        self._token = None
        self._mark_at = None
        self.cycles_total = 0

    # -- device bookkeeping ---------------------------------------------------------------------------
    def _alloc(self):
        if self._grid is None:
            self._grid = _Grid(self.ctx, self.shape)
            self._bins = Bins(self.ctx, self.shape)
        return self._grid

    def _ensure_set(self, which, n):
        if n <= self._caps[which]:
            return
        cap = max(n, int(self._caps[which] * 1.5), 1024)
        limit = min(self.ctx.max_binding, self.ctx.max_buffer)
        if 36 * n > limit:
            raise ValueError(f"the resident liquid holds {n:,} particles, {36 * n:,} bytes; this adapter's largest "
                             f"buffer is {limit:,} bytes (use pressure = gpu or cpu for this grid)")
        cap = min(cap, limit // 36)
        self._sets[which] = _grow(self.ctx, self._sets[which], 36 * self._caps[which], 36 * cap)
        self._caps[which] = cap

    def _common(self):
        """The uniform tail every particle kernel reads: voxel size, origin and the inverse voxel size."""
        return ((0.0, 0.0, 0.0, self.voxel), (float(self.origin[0]), float(self.origin[1]), float(self.origin[2]),
                                              1.0 / self.voxel))

    def _mark(self, name):
        if not self.profile:
            return
        self.ctx.read(self._grid.reducer.red, 4)
        now = time.perf_counter()
        if self._mark_at is not None:
            self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + now - self._mark_at
        self._mark_at = now

    def _cancelled(self):
        return self.cancel is not None and self.cancel.is_set()

    def _put_solid(self, g, solid, velocity):
        key = (id(solid), id(velocity))
        if g.solid_key == key:
            return
        ctx = self.ctx
        ctx.write(g.solid, np.zeros(g.n, np.uint32) if solid is None
                  else np.ascontiguousarray(solid.reshape(-1), np.uint32))
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

    # -- state in and out -----------------------------------------------------------------------------
    def _upload(self, state):
        started = time.perf_counter()
        arrays = state.arrays
        n = len(arrays["position"])
        self._alloc()
        cur = self._cur
        self._ensure_set(cur, max(n, 1))
        rec = np.zeros(n, PART)
        if n:
            pos, vel = np.asarray(arrays["position"], np.float32), np.asarray(arrays["velocity"], np.float32)
            rec["px"], rec["py"], rec["pz"] = pos[:, 0], pos[:, 1], pos[:, 2]
            rec["vx"], rec["vy"], rec["vz"] = vel[:, 0], vel[:, 1], vel[:, 2]
            rec["age"] = np.asarray(arrays["age"]).astype(np.uint32)
            rec["id"] = np.asarray(arrays["id"]).astype(np.uint32)
            rec["temp"] = np.asarray(arrays.get("temperature", np.ones(n, np.float32)), np.float32)
            self.ctx.write(self._sets[cur], rec)
        self._n_slots = self._n_live = n
        self.copy_seconds["host_to_device"] += time.perf_counter() - started

    def _readback(self, token):
        if token != self._token:
            raise RuntimeError("this GpuLiquidState is no longer on the card: read its arrays before the next step")
        started = time.perf_counter()
        n = self._n_slots
        if n:
            rec = read_staged(self.ctx, self._sets[self._cur], 36 * n).view(PART)
            rec = rec[rec["id"] != DEAD]
        else:
            rec = np.zeros(0, PART)
        count = len(rec)
        out = {"position": np.empty((count, 3), np.float32), "velocity": np.empty((count, 3), np.float32),
               "age": rec["age"].astype(np.int32), "life": np.full(count, LIFE, np.int32),
               "id": rec["id"].astype(np.int64), "temperature": np.ascontiguousarray(rec["temp"])}
        for axis, (p, v) in enumerate((("px", "vx"), ("py", "vy"), ("pz", "vz"))):
            out["position"][:, axis] = rec[p]
            out["velocity"][:, axis] = rec[v]
        spacing = 1.0 / self.ppc ** (1.0 / 3.0)
        out["size"] = np.full(count, spacing * self.voxel, np.float32)
        out["color"] = np.empty((count, 4), np.float32)
        out["color"][:] = np.asarray(LIQUID_COLOR, np.float32)
        self.copy_seconds["device_to_host"] += time.perf_counter() - started
        return out

    # -- host-side emission ---------------------------------------------------------------------------
    def _emit_new(self, frame, substep, solid, rng, count_of):
        """The reference's `_emit`, returning only the new particles as world-unit float32 (position, velocity,
        temperature), or None. Pouring sources ask `count_of(flat)` for the particles already in the cells."""
        added = []
        for source in self.sources:
            knobs = source._knobs(frame)
            velocity = np.asarray(knobs["velocity"], np.float64) / self.voxel
            pouring = bool(np.any(velocity))
            first = max(int(source.start_frame), self.start_frame)
            if frame < first or frame > int(source.end_frame):
                continue
            if not pouring and not (frame == first and substep == 0):
                continue
            flat, weight, _motion = source.footprint(self, frame)
            if not len(flat):
                continue
            flat = flat[weight > 0]
            if solid is not None:
                flat = flat[~solid.reshape(-1)[flat]]
            if pouring:
                flat = flat[count_of(flat) < max(1, self.ppc // 2)]
            if not len(flat):
                continue
            fresh = self._seed_cells(flat, self.ppc, rng)
            added.append((fresh, np.tile(velocity, (len(fresh), 1)),
                          np.full(len(fresh), float(knobs.get("temperature", 1.0)))))
        if not added:
            return None
        pos = np.concatenate([a for a, _, _ in added])
        vel = np.concatenate([b for _, b, _ in added])
        temperature = np.concatenate([c for _, _, c in added])
        return ((pos * self.voxel + self.origin).astype(np.float32), (vel * self.voxel).astype(np.float32),
                temperature.astype(np.float32))

    def _counts_at(self, flat):
        g, ctx = self._grid, self.ctx
        m = len(flat)
        if m > g.gather_cap:
            g.gather_cap = max(m, 2 * g.gather_cap)
            g.gather_idx, g.gather_out = ctx.buffer(4 * g.gather_cap), ctx.buffer(4 * g.gather_cap)
        ctx.write(g.gather_idx, np.ascontiguousarray(flat, np.uint32))
        ctx.dispatch("lq_gather", {"src": self._bins.counts, "idx": g.gather_idx, "out": g.gather_out},
                     _u(b=(m,)), _lin(m))
        return ctx.read(g.gather_out, 4 * m).view(np.uint32).astype(np.int64)

    # -- forces ---------------------------------------------------------------------------------------
    def _force_ops(self, frame, substep, dt):
        import math
        ops, turb = [], None
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

    # -- one substep ----------------------------------------------------------------------------------
    def step(self, state, frame=0, substep=0, seed=0):
        """One substep of dt = 1 / substeps frames. The input state stays valid until this call returns."""
        if self._cancelled():
            raise Cancelled()
        self._sync_domain(state)
        flip3d.SOLVER_STATS["steps"] += 1
        ctx = self.ctx
        g = self._alloc()
        bins = self._bins
        p, dt = self.params, self.dt
        ctx.flush()                 # the committed substep's queued work finishes before this one can be abandoned
        self._mark_at = time.perf_counter()
        resident = getattr(state, "token", None) is not None and state.token == self._token
        if not resident:
            self._upload(state)
        n_a = self._n_slots
        cur = self._cur
        meta = dict(state.meta)
        next_id = int(meta.get("next_id", 0))
        solid, svel, _unused = self._solid_for(frame, substep)
        self._put_solid(g, solid, svel)
        rng = np.random.default_rng((int(seed) & 0x7FFFFFFF, int(frame) & 0x7FFFFFFF, int(substep), 1))
        c_tail, d_tail = self._common()
        ncells = g.n
        grid_a = self.shape + (ncells,)
        open_bits = sum(1 << i for i, opened in enumerate(self.open_faces) if opened)

        # ---- bin the resident particles, emit, bin the new ones
        bins.ensure(n_a)
        bins.begin()
        bins.count(self._sets[cur], 0, n_a, (c_tail, d_tail))
        started = time.perf_counter()
        fresh = self._emit_new(frame, substep, solid, rng, self._counts_at)
        m = 0 if fresh is None else len(fresh[0])
        if m:
            self._ensure_set(cur, n_a + m)
            bins.ensure(n_a + m)
            rec = np.zeros(m, PART)
            rec["px"], rec["py"], rec["pz"] = fresh[0][:, 0], fresh[0][:, 1], fresh[0][:, 2]
            rec["vx"], rec["vy"], rec["vz"] = fresh[1][:, 0], fresh[1][:, 1], fresh[1][:, 2]
            rec["id"] = np.arange(next_id, next_id + m, dtype=np.uint32)
            rec["temp"] = fresh[2]
            ctx.write(self._sets[cur], rec, 36 * n_a)
            self.copy_seconds["host_to_device"] += time.perf_counter() - started
            bins.count(self._sets[cur], n_a, n_a + m, (c_tail, d_tail))
        n = n_a + m
        next_id += m
        if n == 0:
            return self._commit_empty(state, meta, next_id, frame)
        A = self._sets[cur]
        bins.sort(A, n, (c_tail, d_tail))
        maxper = int(self.max_per_cell)
        ctx.dispatch("lq_cellstats", {"offsets": bins.offsets, "order": bins.order, "part": A, "cnt2": g.cnt2,
                                      "vsum": g.vsum}, _u(a=grid_a[:3] + (ncells,), b=(maxper,)), _lin(ncells))
        ctx.dispatch("lq_gapneed", {"cnt2": g.cnt2, "solid": g.solid, "need": g.need, "newcnt": g.newcnt},
                     _u(a=self.shape + (ncells,), b=(0, MIN_PER_CELL)), _lin(ncells))
        _scan(ctx, g.need, g.needoff, g.bsum, ncells)
        _scan(ctx, g.newcnt, g.newoff, g.bsum, ncells)
        ctx.dispatch("lq_totals", {"offsets": bins.offsets, "needoff": g.needoff, "newoff": g.newoff,
                                   "totals": g.totals}, _u(a=self.shape + (ncells,)), ("wg", (1, 1, 1)))
        need_total, n_final, live = (int(v) for v in ctx.read(g.totals, 12).view(np.uint32))
        self._mark("binning and maintenance totals")
        if self._cancelled():
            raise Cancelled()
        if n_final == 0:
            return self._commit_empty(state, meta, next_id, frame)

        # ---- write the work particle set: sorted by cell, kept particles first, top-ups after them
        work = 1 - cur
        self._ensure_set(work, n_final)
        W = self._sets[work]
        ctx.dispatch("lq_compact", {"key": bins.key, "order": bins.order, "offsets": bins.offsets, "newoff": g.newoff,
                                    "src": A, "dst": W}, _u(a=self.shape + (ncells,), b=(maxper, n)), _lin(n))
        ctx.dispatch("lq_gapgen", {"cnt2": g.cnt2, "vsum": g.vsum, "need": g.need, "needoff": g.needoff,
                                   "newoff": g.newoff, "dst": W},
                     _u(a=self.shape + (ncells,), b=(next_id, 0, (int(seed) & 0x7FFFFFFF) * 31 + int(frame),
                                                     int(substep) * 977 + 13), c=c_tail, d=d_tail), _lin(ncells))
        next_id += need_total
        self._mark("maintenance")

        # ---- grid: classify, tiles, particle to grid
        for buf in g.f + g.old + g.valid_new + g.valid_old:
            ctx.clear(buf)
        ctx.clear(g.tmark)
        ctx.clear(g.esc)
        ctx.dispatch("lq_classify", {"newcnt": g.newcnt, "solid": g.solid, "dens": g.dens, "blk": g.blk,
                                     "tmark": g.tmark}, _u(a=self.shape + (ncells,)), _lin(ncells))
        ctx.dispatch("dilate", {"act": g.tmark, "mask": g.tmask}, _u(a=self.shape), fgs._lin(g.tiles))
        threads = g.tiles * 512
        for axis, name in enumerate("uvw"):
            ctx.dispatch(f"lq_p2g_{name}", {"part": W, "offs": g.newoff, "tmask": g.tmask, "grid": g.f[axis],
                                            "old": g.old[axis], "valid": g.valid_old[axis]},
                         _u(a=self.shape, c=c_tail, d=d_tail), _lin(threads))
        self._mark("particle to grid")

        # ---- forces and constraints
        dims = self.shape
        cgrid = ("wg", (_cdiv(dims[2], 8), _cdiv(dims[1], 8), _cdiv(dims[0], 4)))
        u, v, w = g.f

        def cell(kernel, bufs, b=(), c=(), d=()):
            bufs = dict(bufs)
            bufs["tiles"] = g.dummy
            ctx.dispatch(kernel, bufs, _u(a=dims + (0,), b=b, c=c, d=d), cgrid)

        cell("face_op", {"u": u, "v": v, "w": w}, b=(0,), c=(0.0, float(-p["gravity"] * dt), 0.0))
        ops, turb = self._force_ops(frame, substep, dt)
        if turb is not None:
            if g.turb is None:
                g.turb = [ctx.buffer(4 * g.n) for _ in range(3)]
            started = time.perf_counter()
            for buf, comp in zip(g.turb, turb):
                ctx.write(buf, comp)
            self.copy_seconds["host_to_device"] += time.perf_counter() - started
        for op in ops:
            kind = op[0]
            if kind == "grav":
                cell("grav", {"u": u, "v": v, "w": w, "d": g.dens}, c=op[1])
            elif kind == "add":
                cell("face_op", {"u": u, "v": v, "w": w}, b=(0,), c=op[1])
            elif kind == "scale":
                cell("face_op", {"u": u, "v": v, "w": w}, b=(1,), c=(op[1],))
            elif kind == "turb":
                cell("apply_cforce", {"u": u, "v": v, "w": w, "cx": g.turb[0], "cy": g.turb[1], "cz": g.turb[2],
                                      "solid": g.solid}, b=(0,), c=(1.0,))
        cell("constrain", {"u": u, "v": v, "w": w, "blk": g.blk, "svel": g.svel},
             b=(open_bits, 0, 0, 1 if g.has_solid else 0), c=(float(g.has_svel),))
        self._mark("forces and constraints")

        # ---- pressure
        mg, red = g.mg, g.reducer
        cell("lq_rhs", {"u": u, "v": v, "w": w, "blk": g.blk, "rhs": g.rhs, "q": g.q})
        cell("weights0", {"blk": g.blk, "wt": g.wt0}, b=(open_bits,))
        mg.coarsen(g.dummy)
        cell("dw_copy", {"wt": g.wt0, "r": g.r0})
        red.run(g.r0, g.blk, ncells, 0, 12)
        red.run(g.rhs, g.blk, ncells, 0, 8, masked=True)
        red.run(g.rhs, g.blk, ncells, 2, 9, masked=True)
        cell("sub_mean", {"q": g.rhs, "blk": g.blk, "red": red.red}, b=(2, 8, 12))
        cap = fgs.MG_MAX_CYCLES if int(p["max_iterations"]) <= 0 else min(fgs.MG_MAX_CYCLES, int(p["max_iterations"]))
        tol = float(p["tolerance"])
        recorded = int(meta.get("mg_cycles", 0))

        def residual():
            mg.residual(0, g.dummy)
            red.run(g.r0, g.blk, ncells, 1, 0, masked=True)
            return red.value(0)

        start = residual()
        res, done, scale = start, 0, CORRECTION
        if res > tol:
            while True:
                for _ in range(min(recorded, cap) if scale == CORRECTION else 0):
                    mg.cycle(scale, g.dummy)
                    done += 1
                if done:
                    res = residual()
                while res > tol and done < cap and (res <= 4.0 * start or done < 6):
                    if self._cancelled():
                        raise Cancelled()
                    mg.cycle(scale, g.dummy)
                    done += 1
                    res = residual()
                if res <= tol or done >= cap or scale != CORRECTION:
                    break
                # The over-corrected cycle diverged on this operator: start again from zero with the plain one.
                cell("lq_rhs", {"u": u, "v": v, "w": w, "blk": g.blk, "rhs": g.rhs, "q": g.q})
                cell("sub_mean", {"q": g.rhs, "blk": g.blk, "red": red.red}, b=(2, 8, 12))
                scale, done = 1.0, 0
                res = start
        recorded = max(recorded, done)
        self.cycles_total += done
        self._mark("pressure")
        red.run(g.q, g.blk, ncells, 0, 10, masked=True)
        red.run(g.q, g.blk, ncells, 2, 11, masked=True)
        cell("sub_mean", {"q": g.q, "blk": g.blk, "red": red.red}, b=(2, 10, 12))
        for axis in range(3):
            ctx.dispatch("lq_project", {"f": g.f[axis], "q": g.q, "blk": g.blk},
                         _u(a=self.shape, b=(axis, open_bits, 0, g.face_n[axis])), _lin(g.face_n[axis]))
        self._mark("project")

        # ---- extrapolate both velocity sets four layers into the air
        for axis in range(3):
            ctx.dispatch("lq_touched", {"blk": g.blk, "f": g.f[axis], "o": g.old[axis], "vo": g.valid_old[axis],
                                        "vt": g.valid_new[axis]},
                         _u(a=self.shape, b=(axis, 0, 0, g.face_n[axis])), _lin(g.face_n[axis]))
        for axis in range(3):
            shape, count = g.face_shapes[axis], g.face_n[axis]
            for field, valid in ((g.f[axis], g.valid_new[axis]), (g.old[axis], g.valid_old[axis])):
                a, b, va, vb = field, g.scratch[axis], valid, g.scratch_valid[axis]
                for _ in range(flip3d.EXTRAPOLATE_LAYERS):
                    ctx.dispatch("flip_extrapolate", {"field": a, "valid": va, "out": b, "out_valid": vb},
                                 _u(a=shape, b=(count,)), _lin(count))
                    a, b, va, vb = b, a, vb, va
        self._mark("extrapolation")

        # ---- grid to particle, advection, ageing; escaped particles are marked dead
        ctx.dispatch("lq_g2p", {"part": W, "u": g.f[0], "v": g.f[1], "w": g.f[2], "ou": g.old[0], "ov": g.old[1],
                                "ow": g.old[2]},
                     _u(a=self.shape, b=(n_final,), c=(0.0, float(self.flip_ratio), 0.0, self.voxel), d=d_tail),
                     _lin(n_final))
        ctx.dispatch("lq_vmax", {"part": W, "out": red.part}, _u(b=(n_final,)), ("wg", (fgs.REDUCE_GROUPS, 1, 1)))
        ctx.dispatch("reduce2", {"part": red.part, "red": red.red}, _u(b=(1, 0, SLOT_VMAX)), ("wg", (1, 1, 1)))
        ctx.dispatch("lq_advect", {"part": W, "u": g.f[0], "v": g.f[1], "w": g.f[2], "red": red.red,
                                   "solid": g.solid, "esc": g.esc},
                     _u(a=self.shape, b=(n_final, open_bits, SLOT_VMAX),
                        c=(float(dt), 0.0, float(0.995 ** dt), self.voxel), d=d_tail), _lin(n_final))
        escaped = 0
        if open_bits:
            escaped = int(ctx.read(g.esc, 4).view(np.uint32)[0])
        self._mark("grid to particle and advection")
        if self._cancelled():
            raise Cancelled()

        # ---- commit
        self._cur = work
        self._n_slots = n_final
        self._n_live = n_final - escaped
        self._serial += 1
        self._token = (self._uid, self._serial)
        meta["escaped_mass"] = float(meta.get("escaped_mass", 0.0)) + escaped / self.ppc * self.voxel ** 3
        meta.update(next_id=int(next_id), substep_count=int(meta.get("substep_count", 0)) + 1,
                    cg_iterations=int(done), cg_residual=float(res), mg_cycles=int(recorded))
        self.stats.update(cg_iterations=int(done), particles=int(self._n_live))
        return GpuLiquidState(self, self._token, meta)

    def level_set_device(self, state, radius, support, smoothing=0, resolution=1):
        """The blended signed-distance field (`liquid_surface.level_set`) of a state, left on the card: (device buffer,
        shape). A resident state is read where it lies; any other state is uploaded to a scratch buffer."""
        from . import flip_gpu_levelset
        res = max(1, int(resolution))
        if getattr(state, "token", None) is not None and state.token == self._token:
            part, n = self._sets[self._cur], self._n_slots
        else:
            arrays = state.arrays
            n = len(arrays["position"])
            rec = np.zeros(n, PART)
            pos = np.asarray(arrays["position"], np.float32).reshape(-1, 3)
            rec["px"], rec["py"], rec["pz"] = pos[:, 0], pos[:, 1], pos[:, 2]
            rec["id"] = np.arange(n, dtype=np.uint32)
            part = self.ctx.buffer(36 * max(n, 1))
            self.ctx.write(part, rec)
        shape = tuple(v * res for v in self.shape)
        if n == 0:
            phi = self.ctx.buffer(4 * int(np.prod(shape)))
            self.ctx.write(phi, np.full(shape, min(flip_gpu_levelset.FAR, 2.0 * support), np.float32))
            return phi, shape
        return flip_gpu_levelset.level_set_device(part, n, tuple(float(v) for v in self.origin), self.voxel / res, shape,
                                                  radius, support, smoothing)

    def _commit_empty(self, state, meta, next_id, frame):
        """A substep with no particles anywhere: nothing runs on the card."""
        self._n_slots = self._n_live = 0
        self._serial += 1
        self._token = (self._uid, self._serial)
        meta.update(next_id=int(next_id), substep_count=int(meta.get("substep_count", 0)) + 1,
                    cg_iterations=0, cg_residual=0.0)
        self.stats.update(cg_iterations=0, particles=0)
        return GpuLiquidState(self, self._token, meta)


def create_solver(params=None, **kwargs):
    """A GpuLiquid3D, or on `Unsupported` the reference `Liquid3D` on the `gpu` backend (CPU when no adapter) with the
    reason in `.fallback_reason`."""
    try:
        return GpuLiquid3D(params, **kwargs)
    except Unsupported as error:
        LOG.warning("resident liquid solver unavailable, using the transfer path: %s", error)
        merged = {**(params or {})}
        merged["backend"] = "gpu" if fgs.available() else "cpu"
        hook = None
        if merged["backend"] == "gpu":
            from .fluid3d import _gpu_solver
            hook = _gpu_solver().solve
        keep = {k: v for k, v in kwargs.items() if k in ("cancel", "sources", "forces", "colliders")}
        solver = Liquid3D(merged, pressure_solver=hook, **keep)
        solver.fallback_reason = str(error)
        return solver
