"""Dense GPU extrapolation for FLIP's new and carried MAC face fields."""
from __future__ import annotations

import numpy as np
import time

from . import fluid_gpu_solver as fgs

PHASE_SECONDS = 0.0
_BUFFERS = {}


fgs._kernel("flip_extrapolate", [fgs._c("field"), fgs._c("valid", "u32"),
                                  fgs._c("out", acc="rw"), fgs._c("out_valid", "u32", "rw")], r"""
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = P.b.x;
    if (i >= n) { return; }
    let ny = P.a.y;
    let nz = P.a.z;
    let yz = ny * nz;
    let x = i / yz;
    let y = (i / nz) % ny;
    let z = i % nz;
    var total = 0.0;
    var count = 0.0;
    if (x > 0u) { let j = i - yz; if (valid[j] != 0u) { total += field[j]; count += 1.0; } }
    if (x + 1u < P.a.x) { let j = i + yz; if (valid[j] != 0u) { total += field[j]; count += 1.0; } }
    if (y > 0u) { let j = i - nz; if (valid[j] != 0u) { total += field[j]; count += 1.0; } }
    if (y + 1u < ny) { let j = i + nz; if (valid[j] != 0u) { total += field[j]; count += 1.0; } }
    if (z > 0u) { let j = i - 1u; if (valid[j] != 0u) { total += field[j]; count += 1.0; } }
    if (z + 1u < nz) { let j = i + 1u; if (valid[j] != 0u) { total += field[j]; count += 1.0; } }
    let fill = valid[i] == 0u && count > 0.0;
    out[i] = select(field[i], total / max(count, 1.0), fill);
    out_valid[i] = select(valid[i], 1u, fill);
}
""", cellwise=False)


def extrapolate(field, valid, layers=4):
    """Fill invalid entries from their valid 6-neighbours, keeping each layer on the GPU."""
    global PHASE_SECONDS
    started = time.perf_counter()
    mask = np.ascontiguousarray(np.asarray(valid, np.uint32))
    source = np.ascontiguousarray(np.where(mask != 0, np.asarray(field, np.float32), 0.0))
    shape = tuple(int(n) for n in source.shape)
    ctx = fgs._ctx()
    count = source.size
    buffers = _BUFFERS.get(shape)
    if buffers is None:
        buffers = (ctx.buffer(source.nbytes), ctx.buffer(source.nbytes),
                   ctx.buffer(mask.nbytes), ctx.buffer(mask.nbytes))
        _BUFFERS[shape] = buffers
    a, b, va, vb = buffers
    ctx.write(a, source)
    ctx.write(va, mask)
    params = fgs._u(a=shape, b=(count,))
    for _ in range(max(0, int(layers))):
        ctx.dispatch("flip_extrapolate", {"field": a, "valid": va, "out": b, "out_valid": vb},
                     params, ("wg", ((count + 63) // 64, 1, 1)))
        a, b = b, a
        va, vb = vb, va
    result = ctx.read(a, source.nbytes).view(np.float32).reshape(shape).astype(np.float64)
    result_valid = ctx.read(va, mask.nbytes).view(np.uint32).reshape(shape).astype(bool)
    PHASE_SECONDS += time.perf_counter() - started
    return result, result_valid
