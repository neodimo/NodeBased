"""The liquid surface's blended signed-distance field (`liquid_surface.level_set`) as a gather on the GPU.

The CPU version scatters every particle into the 4 x 4 x 4 grid points around it with numpy bincounts; this one
gathers per grid point from the cell-sorted particle lists the resident solver already keeps (`Bins`), so the
particles never leave the card when the solver's own state is the input, and only the finished field is read back
for the viewport frame. The result agrees with the CPU field to float32 rounding (tests/test_flip_gpu_resident.py).
"""
from __future__ import annotations

import numpy as np

from . import fluid_gpu_solver as fgs
from .flip_gpu_resident import Bins, PART, _LIB, _lin, _kernel
from .fluid_gpu_solver import _c, _cdiv, _u

FAR = 1.0e3
_STATE = {}


_kernel("lq_phi", [_c("part", "part"), _c("offsets", "u32"), _c("order", "u32"), _c("phi", acc="rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    // P.a = fine grid, P.b = coarse (binned) grid; P.c = radius, support, fine voxel; P.d = origin, 1 / coarse voxel
    let i = g.x + g.y * 4194240u;
    let fx = P.a.x; let fy = P.a.y; let fz = P.a.z;
    if (i >= fx * fy * fz) { return; }
    let z = i % fz; let y = (i / fz) % fy; let x = i / (fz * fy);
    let origin = vec3<f32>(P.d.x, P.d.y, P.d.z);
    let xg = origin + (vec3<f32>(f32(x), f32(y), f32(z)) + 0.5) * P.c.z;
    let pc = (xg - origin) * P.d.w;
    let sc = P.c.y * P.d.w;
    let cn = vec3<i32>(i32(P.b.x), i32(P.b.y), i32(P.b.z));
    let lo = max(vec3<i32>(0), vec3<i32>(floor(pc - vec3<f32>(sc))));
    let hi = min(cn - vec3<i32>(1), vec3<i32>(floor(pc + vec3<f32>(sc))));
    let s2 = P.c.y * P.c.y;
    var ksum = 0.0;
    var mean = vec3<f32>(0.0);
    for (var cx = lo.x; cx <= hi.x; cx++) { for (var cy = lo.y; cy <= hi.y; cy++) { for (var cz = lo.z; cz <= hi.z; cz++) {
        let c = u32((cx * cn.y + cy) * cn.z + cz);
        let end = offsets[c + 1u];
        for (var k = offsets[c]; k < end; k++) {
            let q = part[order[k]];
            let r = vec3<f32>(q.px, q.py, q.pz) - xg;
            let t = max(0.0, 1.0 - dot(r, r) / s2);
            let w = t * t * t;
            ksum = ksum + w;
            mean = mean + w * r;
        }
    } } }
    var value = 1000.0;
    if (ksum > 1e-12) { value = length(mean / ksum) - P.c.x; }
    phi[i] = min(value, 2.0 * P.c.y);
}
""")

_kernel("lq_box", [_c("src"), _c("dst", acc="rw")], """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x + g.y * 4194240u;
    let nx = i32(P.a.x); let ny = i32(P.a.y); let nz = i32(P.a.z);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let z = i32(i % P.a.z); let y = i32((i / P.a.z) % P.a.y); let x = i32(i / (P.a.z * P.a.y));
    var total = 0.0;
    for (var a = -1; a <= 1; a++) { for (var b = -1; b <= 1; b++) { for (var c = -1; c <= 1; c++) {
        let xx = clamp(x + a, 0, nx - 1); let yy = clamp(y + b, 0, ny - 1); let zz = clamp(z + c, 0, nz - 1);
        total = total + src[u32((xx * ny + yy) * nz + zz)];
    } } }
    dst[i] = total / 27.0;
}
""")


def fits(shape):
    """True when a field of this (fine) shape fits one storage buffer of this adapter."""
    ctx = fgs._ctx()
    return 4 * int(np.prod(shape)) <= min(ctx.max_binding, ctx.max_buffer)


def _bins(ctx, shape):
    key = tuple(shape)
    got = _STATE.get(key)
    if got is None:
        if len(_STATE) > 3:
            _STATE.clear()
        got = _STATE[key] = Bins(ctx, shape)
    return got


def level_set_device(part, n, origin, voxel, shape, radius, support, smoothing=0):
    """The field of `n` particles held in the device AoS buffer `part` on the grid of `shape` cells of `voxel` world
    units at `origin` (the same arguments as `liquid_surface.level_set`): a device buffer plus the shape. The
    particles are binned on cells of at least half the support, so a grid point gathers from at most 4 per axis."""
    ctx = fgs._ctx()
    shape = tuple(int(v) for v in shape)
    bin_voxel = max(float(voxel), float(support) / 2.0)
    bshape = tuple(max(1, int(np.ceil(shape[a] * float(voxel) / bin_voxel - 1e-9))) for a in range(3))
    bins = _bins(ctx, bshape)
    common = ((0.0, 0.0, 0.0, bin_voxel),
              (float(origin[0]), float(origin[1]), float(origin[2]), 1.0 / bin_voxel))
    bins.ensure(n)
    bins.begin()
    bins.count(part, 0, n, common)
    bins.sort(part, n, common)
    count = shape[0] * shape[1] * shape[2]
    phi = ctx.buffer(4 * count)
    ctx.dispatch("lq_phi", {"part": part, "offsets": bins.offsets, "order": bins.order, "phi": phi},
                 _u(a=shape, b=bshape, c=(float(radius), float(support), float(voxel), 0.0), d=common[1]),
                 _lin(count))
    if smoothing:
        other = ctx.buffer(4 * count)
        for _ in range(int(smoothing)):
            ctx.dispatch("lq_box", {"src": phi, "dst": other}, _u(a=shape), _lin(count))
            phi, other = other, phi
    return phi, shape


def read_field(phi, shape):
    ctx = fgs._ctx()
    return ctx.read(phi, 4 * int(np.prod(shape))).view(np.float32).reshape(shape).copy()


def level_set(positions, origin, voxel, shape, radius, support, smoothing=0):
    """`liquid_surface.level_set` for host positions: upload, gather, read the finished field back."""
    ctx = fgs._ctx()
    positions = np.asarray(positions, np.float32).reshape(-1, 3)
    n = len(positions)
    shape = tuple(int(v) for v in shape)
    if not n:
        return np.full(shape, min(FAR, 2.0 * support), np.float32)
    rec = np.zeros(n, PART)
    rec["px"], rec["py"], rec["pz"] = positions[:, 0], positions[:, 1], positions[:, 2]
    rec["id"] = np.arange(n, dtype=np.uint32)
    part = ctx.buffer(36 * n)
    ctx.write(part, rec)
    phi, shape = level_set_device(part, n, origin, voxel, shape, radius, support, smoothing)
    return read_field(phi, shape)
