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


_kernel("upres_sparse_sample", [_c("src"), _c("coords", "u32"), _c("out", acc="rw")], """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x + P.b.w;
    if (index >= P.a.w * 512u) { return; }
    let tile = index / 512u;
    let local = index % 512u;
    let x = i32(coords[tile * 3u]) * 8 + i32(local / 64u);
    let y = i32(coords[tile * 3u + 1u]) * 8 + i32((local / 8u) % 8u);
    let z = i32(coords[tile * 3u + 2u]) * 8 + i32(local % 8u);
    if (x >= i32(P.a.x) || y >= i32(P.a.y) || z >= i32(P.a.z)) {
        out[gid.x] = 0.0; return;
    }
    let coarse = vec3<f32>(f32(P.b.x), f32(P.b.y), f32(P.b.z));
    let fine = vec3<f32>(f32(P.a.x), f32(P.a.y), f32(P.a.z));
    let p = vec3<f32>(f32(x), f32(y), f32(z)) * (coarse - vec3<f32>(1.0)) / (fine - vec3<f32>(1.0));
    let lo = vec3<i32>(floor(p));
    let hi = min(lo + vec3<i32>(1), vec3<i32>(P.b.xyz) - vec3<i32>(1));
    let f = p - vec3<f32>(lo);
    let sy = i32(P.b.z); let sx = i32(P.b.y) * sy;
    let base = lo.x * sx + lo.y * sy + lo.z;
    let dx = (hi.x - lo.x) * sx; let dy = (hi.y - lo.y) * sy; let dz = hi.z - lo.z;
    let a = mix(mix(src[base], src[base+dz], f.z), mix(src[base+dy], src[base+dy+dz], f.z), f.y);
    let b = mix(mix(src[base+dx], src[base+dx+dz], f.z), mix(src[base+dx+dy], src[base+dx+dy+dz], f.z), f.y);
    out[gid.x] = mix(a, b, f.x);
}
""", cellwise=False)


_kernel("upres_sparse_guided", [_c("src"), _c("coords", "u32"), _c("tilemap", "u32"),
                                  _c("vx"), _c("vy"), _c("vz"), _c("out", acc="rw")], """
fn coarse_get(axis: u32, dims: vec3<i32>, c: vec3<i32>) -> f32 {
    if (any(c < vec3<i32>(0)) || any(c >= dims)) { return 0.0; }
    let i = u32((c.x * dims.y + c.y) * dims.z + c.z);
    if (axis == 0u) { return vx[i]; }
    if (axis == 1u) { return vy[i]; }
    return vz[i];
}
fn coarse_sample(axis: u32, dims: vec3<i32>, p: vec3<f32>) -> f32 {
    let base = vec3<i32>(floor(p)); let f = p - vec3<f32>(base);
    var value = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let c = base + vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
        let w = select(1.0-f.x, f.x, (k & 1u) != 0u) *
                select(1.0-f.y, f.y, (k & 2u) != 0u) *
                select(1.0-f.z, f.z, (k & 4u) != 0u);
        value += w * coarse_get(axis, dims, c);
    }
    return value;
}
fn scalar_get(dims: vec3<i32>, c: vec3<i32>, packed: bool) -> f32 {
    if (any(c < vec3<i32>(0)) || any(c >= dims)) { return 0.0; }
    if (!packed) { return src[u32((c.x * dims.y + c.y) * dims.z + c.z)]; }
    let tile_dims = (dims + vec3<i32>(7)) / 8;
    let tile = tilemap[u32(((c.x / 8) * tile_dims.y + c.y / 8) * tile_dims.z + c.z / 8)];
    if (tile == 0xFFFFFFFFu) { return 0.0; }
    let local = ((c.x % 8) * 8 + c.y % 8) * 8 + c.z % 8;
    return src[tile * 512u + u32(local)];
}
fn scalar_sample(dims: vec3<i32>, p: vec3<f32>, packed: bool) -> f32 {
    let base = vec3<i32>(floor(p)); let f = p - vec3<f32>(base);
    var value = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let c = base + vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
        let w = select(1.0-f.x, f.x, (k & 1u) != 0u) *
                select(1.0-f.y, f.y, (k & 2u) != 0u) *
                select(1.0-f.z, f.z, (k & 4u) != 0u);
        value += w * scalar_get(dims, c, packed);
    }
    return value;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x + P.b.w;
    if (index >= P.a.w * 512u) { return; }
    let tile = index / 512u; let local = index % 512u;
    let p = vec3<i32>(i32(coords[tile*3u])*8+i32(local/64u),
                      i32(coords[tile*3u+1u])*8+i32((local/8u)%8u),
                      i32(coords[tile*3u+2u])*8+i32(local%8u));
    let fine = vec3<i32>(P.a.xyz); let coarse = vec3<i32>(P.b.xyz);
    if (any(p >= fine)) { out[gid.x] = 0.0; return; }
    let scale = (vec3<f32>(coarse) - vec3<f32>(1.0)) / (vec3<f32>(fine) - vec3<f32>(1.0));
    let q = vec3<f32>(p) * scale;
    let velocity = vec3<f32>(coarse_sample(0u, coarse, q),
                             coarse_sample(1u, coarse, q), coarse_sample(2u, coarse, q));
    let back = clamp(vec3<f32>(p) - velocity * P.c.x, vec3<f32>(0.0), vec3<f32>(fine-1));
    let prior = P.c.y > 0.5;
    out[gid.x] = scalar_sample(select(coarse, fine, prior), select(back * scale, back, prior), prior);
}
""", cellwise=False)


def reconstruct_sparse(fields, coords, fine, guide=None, previous=None, step=0.0):
    """GPU reconstruction into packed active-tile buffers, without a dense fine allocation."""
    ctx = _Ctx()
    coarse = next(iter(fields.values())).shape
    count = len(coords) * 512
    if not count:
        return {name: np.zeros((0, 8, 8, 8), np.float32) for name in fields}
    coord_buf = ctx.buffer(coords.nbytes)
    ctx.write(coord_buf, np.asarray(coords, np.uint32).reshape(-1))
    result = {}
    chunk = min((ctx.max_binding // 4 // 512) * 512, 65535 * 256)
    if chunk < 512:
        raise Unsupported("adapter cannot hold one up-res tile")
    guide_buf = None
    tilemap_buf = None
    prior_grid = None
    if guide is not None:
        guide_buf = []
        for axis in range(3):
            buf = ctx.buffer(guide[..., axis].size * 4)
            ctx.write(buf, np.asarray(guide[..., axis], np.float32).reshape(-1))
            guide_buf.append(buf)
        if previous is not None:
            prior_grid = previous.sparse if previous.sparse is not None else previous.to_sparse()
            tile_dims = tuple((n + 7) // 8 for n in fine)
            tilemap = np.full(tile_dims, np.uint32(0xFFFFFFFF), np.uint32)
            for index, (x, y, z) in enumerate(prior_grid.coords):
                tilemap[x, y, z] = index
            tilemap_buf = ctx.buffer(tilemap.nbytes)
            ctx.write(tilemap_buf, tilemap.reshape(-1))
        else:
            tilemap_buf = ctx.buffer(16)
    for name, field in fields.items():
        prior = prior_grid.data.get(name) if prior_grid is not None else None
        input_array = prior if prior is not None else field
        src = ctx.buffer(input_array.size * 4)
        ctx.write(src, np.asarray(input_array, np.float32).reshape(-1))
        packed = np.empty(count, np.float32)
        for start in range(0, count, chunk):
            length = min(chunk, count - start)
            out = ctx.buffer(length * 4)
            workgroups = (length + 255) // 256
            if guide_buf is None:
                ctx.dispatch("upres_sparse_sample", {"src": src, "coords": coord_buf, "out": out},
                             _u(a=(*fine, len(coords)), b=(*coarse, start)),
                             ("wg", (workgroups, 1, 1)))
            else:
                ctx.dispatch("upres_sparse_guided",
                             {"src": src, "coords": coord_buf, "tilemap": tilemap_buf,
                              "vx": guide_buf[0], "vy": guide_buf[1], "vz": guide_buf[2], "out": out},
                             _u(a=(*fine, len(coords)), b=(*coarse, start),
                                c=(step, 1.0 if prior is not None else 0.0)),
                             ("wg", (workgroups, 1, 1)))
            packed[start:start + length] = ctx.read(out, length * 4).view(np.float32)
        result[name] = packed.reshape(-1, 8, 8, 8)
    if guide_buf is not None:
        components = []
        for axis in range(3):
            packed = np.empty(count, np.float32)
            for start in range(0, count, chunk):
                length = min(chunk, count - start)
                out = ctx.buffer(length * 4)
                ctx.dispatch("upres_sparse_sample",
                             {"src": guide_buf[axis], "coords": coord_buf, "out": out},
                             _u(a=(*fine, len(coords)), b=(*coarse, start)),
                             ("wg", ((length + 255) // 256, 1, 1)))
                packed[start:start + length] = ctx.read(out, length * 4).view(np.float32)
            components.append(packed.reshape(-1, 8, 8, 8))
        result["velocity"] = np.stack(components, axis=-1)
    return result


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
