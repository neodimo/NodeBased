"""GPU trilinear reconstruction and guided transport for FluidUpres3D."""
from __future__ import annotations

import numpy as np

from .fluid_gpu_solver import _Ctx, _c, _kernel, _u, TILES, Unsupported


_kernel("upres_sample", [_c("src"), _c("out", acc="rw"), TILES], """
    let coarse = vec3<f32>(f32(P.b.x), f32(P.b.y), f32(P.b.z));
    let fine = vec3<f32>(f32(P.a.x), f32(P.a.y), f32(P.a.z));
    let p = vec3<f32>(f32(x), f32(y), f32(z)) * (coarse - vec3<f32>(1.0)) / (fine - vec3<f32>(1.0));
    let lo = vec3<i32>(floor(p)); let hi = min(lo + vec3<i32>(1), vec3<i32>(P.b.xyz) - vec3<i32>(1));
    let f = p - vec3<f32>(lo);
    let sy = i32(P.b.z); let sx = i32(P.b.y) * sy;
    let base = lo.x * sx + lo.y * sy + lo.z;
    let dx = (hi.x-lo.x)*sx; let dy = (hi.y-lo.y)*sy; let dz = hi.z-lo.z;
    let a = mix(mix(src[base],src[base+dz],f.z),mix(src[base+dy],src[base+dy+dz],f.z),f.y);
    let b = mix(mix(src[base+dx],src[base+dx+dz],f.z),mix(src[base+dx+dy],src[base+dx+dy+dz],f.z),f.y);
    out[i] = mix(a,b,f.x);
""", size="8, 8, 4")

_kernel("upres_advect", [_c("src"), _c("vx"), _c("vy"), _c("vz"), _c("out", acc="rw"), TILES], """
    let p = clamp(vec3<f32>(f32(x),f32(y),f32(z)) -
        P.c.x * vec3<f32>(vx[i],vy[i],vz[i]), vec3<f32>(0.0),
        vec3<f32>(f32(nx-1),f32(ny-1),f32(nz-1)));
    let lo = vec3<i32>(floor(p)); let hi = min(lo + vec3<i32>(1),vec3<i32>(nx-1,ny-1,nz-1));
    let f = p - vec3<f32>(lo);
    let sy = nz; let sx = ny*nz;
    let base = lo.x*sx + lo.y*sy + lo.z;
    let dx = (hi.x-lo.x)*sx; let dy=(hi.y-lo.y)*sy; let dz=hi.z-lo.z;
    let a=mix(mix(src[base],src[base+dz],f.z),mix(src[base+dy],src[base+dy+dz],f.z),f.y);
    let b=mix(mix(src[base+dx],src[base+dx+dz],f.z),mix(src[base+dx+dy],src[base+dx+dy+dz],f.z),f.y);
    out[i]=mix(a,b,f.x);
""", size="8, 8, 4")


def reconstruct(source, factor, guide_velocity=None, previous=None):
    """Return fine density, temperature, flame, fuel and velocity arrays; all sampling executes on GPU."""
    ctx = _Ctx()
    coarse = tuple(int(n) for n in source.density.shape)
    fine = tuple(n * factor for n in coarse)
    coarse_n = int(np.prod(coarse))
    fine_n = int(np.prod(fine))
    tiles = ctx.buffer(16)
    if 4 * fine_n > ctx.max_binding:
        raise Unsupported("up-res output exceeds the adapter's storage buffer limit")
    grid = ("wg", ((fine[2] + 7) // 8, (fine[1] + 7) // 8, (fine[0] + 3) // 4))

    def put(array):
        buf = ctx.buffer(4 * array.size)
        ctx.write(buf, np.asarray(array, np.float32).reshape(-1))
        return buf

    def sample(array):
        src = put(array)
        out = ctx.buffer(4 * fine_n)
        ctx.dispatch("upres_sample", {"src": src, "out": out, "tiles": tiles}, _u(a=fine, b=coarse), grid)
        return out

    def read(buf):
        return ctx.read(buf, 4 * fine_n).view(np.float32).reshape(fine).copy()

    guide = source.velocity if guide_velocity is None else guide_velocity
    velocity_buf = None if guide is None else tuple(sample(guide[..., axis]) for axis in range(3))
    step = 1.0 / (max(float(getattr(getattr(source, "stream", None), "fps", 24.0)), 1e-6)
                  * float(source.voxel_size) / factor)
    result = []
    for name in ("density", "temperature", "flame", "fuel"):
        value = getattr(source, name)
        if value is None:
            result.append(None)
            continue
        if previous is not None and getattr(previous, name) is not None:
            buf = put(getattr(previous, name))
        else:
            buf = sample(value)
        if velocity_buf is not None:
            out = ctx.buffer(4 * fine_n)
            ctx.dispatch("upres_advect", {"src": buf, "vx": velocity_buf[0], "vy": velocity_buf[1],
                                          "vz": velocity_buf[2], "out": out, "tiles": tiles},
                         _u(a=fine, c=(step,)), grid)
            buf = out
        result.append(buf)
    arrays = [None if buf is None else read(buf) for buf in result]
    velocity = None if velocity_buf is None else np.stack([read(buf) for buf in velocity_buf], axis=-1)
    return (*arrays, velocity)
