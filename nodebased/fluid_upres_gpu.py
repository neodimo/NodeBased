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
                                  _c("vx"), _c("vy"), _c("vz"), _c("sv"), _c("out", acc="rw")], """
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
    var carry = velocity;
    if (P.c.z > 0.5) { carry = velocity + vec3<f32>(sv[index*3u], sv[index*3u+1u], sv[index*3u+2u]); }
    let back = clamp(vec3<f32>(p) - carry * P.c.x, vec3<f32>(0.0), vec3<f32>(fine-1));
    let prior = P.c.y > 0.5;
    out[gid.x] = scalar_sample(select(coarse, fine, prior), select(back * scale, back, prior), prior);
}
""", cellwise=False)


# --- swirl: dense velocity on the box around the active tiles (see fluid_upres.swirl_dense_cpu for the maths) -----------------
# `waves` holds a 16-float header (strength, confinement, wave count, voxel size, fine x y z, low x y z, speed limit) and then 8 floats
# per wave (K xyz, C xyz, phase, pad). P.a = box size, P.b = coarse size.
_SWIRL_LIB = """
fn glin(gid: vec3<u32>) -> u32 { return gid.x + gid.y * 16776960u; }
fn cell3(i: u32) -> vec3<i32> {
    let nz = P.a.z; let ny = P.a.y;
    return vec3<i32>(i32(i / (ny * nz)), i32((i / nz) % ny), i32(i % nz));
}
fn cidx(c: vec3<i32>) -> u32 {
    let d = vec3<i32>(P.a.xyz);
    let k = clamp(c, vec3<i32>(0), d - vec3<i32>(1));
    return u32((k.x * d.y + k.y) * d.z + k.z);
}
"""

_kernel("swirl_u", [_c("cvx"), _c("cvy"), _c("cvz"), _c("waves"), _c("ux", acc="rw"), _c("uy", acc="rw"), _c("uz", acc="rw")], """
fn cget(a: u32, c: vec3<i32>, d: vec3<i32>) -> f32 {
    let i = u32((c.x * d.y + c.y) * d.z + c.z);
    if (a == 0u) { return cvx[i]; }
    if (a == 1u) { return cvy[i]; }
    return cvz[i];
}
fn csample(a: u32, q: vec3<f32>, d: vec3<i32>) -> f32 {
    let lo = vec3<i32>(floor(q)); let f = q - vec3<f32>(lo);
    let hi = min(lo + vec3<i32>(1), d - vec3<i32>(1));
    let x0 = mix(mix(cget(a, vec3<i32>(lo.x,lo.y,lo.z), d), cget(a, vec3<i32>(lo.x,lo.y,hi.z), d), f.z),
                 mix(cget(a, vec3<i32>(lo.x,hi.y,lo.z), d), cget(a, vec3<i32>(lo.x,hi.y,hi.z), d), f.z), f.y);
    let x1 = mix(mix(cget(a, vec3<i32>(hi.x,lo.y,lo.z), d), cget(a, vec3<i32>(hi.x,lo.y,hi.z), d), f.z),
                 mix(cget(a, vec3<i32>(hi.x,hi.y,lo.z), d), cget(a, vec3<i32>(hi.x,hi.y,hi.z), d), f.z), f.y);
    return mix(x0, x1, f.x);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let fine = vec3<f32>(waves[4], waves[5], waves[6]);
    let low = vec3<f32>(waves[7], waves[8], waves[9]);
    let d = vec3<i32>(P.b.xyz);
    let p = vec3<f32>(c) + low;
    let q = clamp(p * (vec3<f32>(d) - vec3<f32>(1.0)) / (fine - vec3<f32>(1.0)), vec3<f32>(0.0), vec3<f32>(d) - vec3<f32>(1.0));
    let vc = vec3<f32>(csample(0u, q, d), csample(1u, q, d), csample(2u, q, d));
    var noise = vec3<f32>(0.0);
    let n = u32(waves[2]);
    for (var w = 0u; w < n; w++) {
        let o = 16u + w * 8u;
        let k = vec3<f32>(waves[o], waves[o+1u], waves[o+2u]);
        let amp = vec3<f32>(waves[o+3u], waves[o+4u], waves[o+5u]);
        noise += amp * cos(dot(k, p) + waves[o+6u]);
    }
    let u = vc + waves[0] * length(vc) * noise;
    ux[i] = u.x; uy[i] = u.y; uz[i] = u.z;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("swirl_omega", [_c("ux"), _c("uy"), _c("uz"), _c("wx", acc="rw"), _c("wy", acc="rw"), _c("wz", acc="rw")], """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let s = 0.5 / P.c.x;
    let xp = cidx(c + vec3<i32>(1,0,0)); let xm = cidx(c - vec3<i32>(1,0,0));
    let yp = cidx(c + vec3<i32>(0,1,0)); let ym = cidx(c - vec3<i32>(0,1,0));
    let zp = cidx(c + vec3<i32>(0,0,1)); let zm = cidx(c - vec3<i32>(0,0,1));
    wx[i] = ((uz[yp] - uz[ym]) - (uy[zp] - uy[zm])) * s;
    wy[i] = ((ux[zp] - ux[zm]) - (uz[xp] - uz[xm])) * s;
    wz[i] = ((uy[xp] - uy[xm]) - (ux[yp] - ux[ym])) * s;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("swirl_conf", [_c("wx"), _c("wy"), _c("wz"), _c("ux", acc="rw"), _c("uy", acc="rw"), _c("uz", acc="rw")], """
fn wmag(i: u32) -> f32 { return length(vec3<f32>(wx[i], wy[i], wz[i])); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let s = 0.5 / P.c.x;
    let eta = vec3<f32>(wmag(cidx(c + vec3<i32>(1,0,0))) - wmag(cidx(c - vec3<i32>(1,0,0))),
                        wmag(cidx(c + vec3<i32>(0,1,0))) - wmag(cidx(c - vec3<i32>(0,1,0))),
                        wmag(cidx(c + vec3<i32>(0,0,1))) - wmag(cidx(c - vec3<i32>(0,0,1)))) * s;
    let unit = eta / (length(eta) + 1e-20);
    let f = P.c.y * P.c.x * cross(unit, vec3<f32>(wx[i], wy[i], wz[i]));
    ux[i] = ux[i] + f.x; uy[i] = uy[i] + f.y; uz[i] = uz[i] + f.z;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("swirl_final", [_c("cvx"), _c("cvy"), _c("cvz"), _c("waves"), _c("ux", acc="rw"), _c("uy", acc="rw"), _c("uz", acc="rw")], """
fn cget(a: u32, c: vec3<i32>, d: vec3<i32>) -> f32 {
    let i = u32((c.x * d.y + c.y) * d.z + c.z);
    if (a == 0u) { return cvx[i]; }
    if (a == 1u) { return cvy[i]; }
    return cvz[i];
}
fn csample(a: u32, q: vec3<f32>, d: vec3<i32>) -> f32 {
    let lo = vec3<i32>(floor(q)); let f = q - vec3<f32>(lo);
    let hi = min(lo + vec3<i32>(1), d - vec3<i32>(1));
    let x0 = mix(mix(cget(a, vec3<i32>(lo.x,lo.y,lo.z), d), cget(a, vec3<i32>(lo.x,lo.y,hi.z), d), f.z),
                 mix(cget(a, vec3<i32>(lo.x,hi.y,lo.z), d), cget(a, vec3<i32>(lo.x,hi.y,hi.z), d), f.z), f.y);
    let x1 = mix(mix(cget(a, vec3<i32>(hi.x,lo.y,lo.z), d), cget(a, vec3<i32>(hi.x,lo.y,hi.z), d), f.z),
                 mix(cget(a, vec3<i32>(hi.x,hi.y,lo.z), d), cget(a, vec3<i32>(hi.x,hi.y,hi.z), d), f.z), f.y);
    return mix(x0, x1, f.x);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let fine = vec3<f32>(waves[4], waves[5], waves[6]);
    let low = vec3<f32>(waves[7], waves[8], waves[9]);
    let d = vec3<i32>(P.b.xyz);
    let p = vec3<f32>(c) + low;
    let q = clamp(p * (vec3<f32>(d) - vec3<f32>(1.0)) / (fine - vec3<f32>(1.0)), vec3<f32>(0.0), vec3<f32>(d) - vec3<f32>(1.0));
    let vc = vec3<f32>(csample(0u, q, d), csample(1u, q, d), csample(2u, q, d));
    let added = vec3<f32>(ux[i], uy[i], uz[i]) - vc;
    let cap = waves[10] * length(vc);
    let size = length(added);
    let k = min(1.0, cap / max(size, 1e-20));
    ux[i] = added.x * k; uy[i] = added.y * k; uz[i] = added.z * k;
}
""", lib=_SWIRL_LIB, cellwise=False)


def swirl_dense(guide, factor, low, dims, waves, strength, confine, limit, voxel):
    """GPU twin of `fluid_upres.swirl_dense_cpu`: the swirl velocity on the dense box, shape (*dims, 3)."""
    ctx = _Ctx()
    coarse = tuple(int(n) for n in guide.shape[:3])
    count = int(np.prod(dims))
    if 4 * count > ctx.max_binding:
        raise Unsupported("the swirl box exceeds the adapter's storage buffer limit")
    K, C, phase = waves
    head = np.zeros(16, np.float32)
    head[:11] = (strength, confine, len(phase), voxel, *(n * factor for n in coarse), *low, limit)
    table = np.concatenate([head, np.concatenate([K, C, phase[:, None], np.zeros((len(phase), 1), np.float32)],
                                                 axis=1).astype(np.float32).reshape(-1)])
    wave_buf = ctx.buffer(table.nbytes)
    ctx.write(wave_buf, table)
    cv = []
    for axis in range(3):
        buf = ctx.buffer(guide[..., axis].size * 4)
        ctx.write(buf, np.asarray(guide[..., axis], np.float32).reshape(-1))
        cv.append(buf)
    u = [ctx.buffer(count * 4) for _ in range(3)]
    w = [ctx.buffer(count * 4) for _ in range(3)]
    groups = (count + 255) // 256
    grid = ("wg", (min(groups, 65535), (groups + 65534) // 65535, 1))
    params = _u(a=dims, b=coarse, c=(voxel, confine))
    ctx.dispatch("swirl_u", {"cvx": cv[0], "cvy": cv[1], "cvz": cv[2], "waves": wave_buf,
                             "ux": u[0], "uy": u[1], "uz": u[2]}, params, grid)
    ctx.dispatch("swirl_omega", {"ux": u[0], "uy": u[1], "uz": u[2], "wx": w[0], "wy": w[1], "wz": w[2]}, params, grid)
    ctx.dispatch("swirl_conf", {"wx": w[0], "wy": w[1], "wz": w[2], "ux": u[0], "uy": u[1], "uz": u[2]}, params, grid)
    ctx.dispatch("swirl_final", {"cvx": cv[0], "cvy": cv[1], "cvz": cv[2], "waves": wave_buf,
                                 "ux": u[0], "uy": u[1], "uz": u[2]}, params, grid)
    out = np.stack([ctx.read(buf, count * 4).view(np.float32).reshape(tuple(int(n) for n in dims)) for buf in u], axis=-1)
    return out.copy()


# --- persistent swirl (Fluids 17 C): a fine velocity field carried from frame to frame on the dense box --------------------------
# Velocity buffers hold three floats per cell. P.a = box size, P.b = coarse size, P.c = (voxel, spin per step, voxels per unit velocity,
# injection per step), P.d = (decay per step, speed limit as a multiple of the coarse speed). `waves` is as for `swirl_u`.
_kernel("pswirl_vc", [_c("cg"), _c("waves"), _c("vc", acc="rw")], """
fn cget(c: vec3<i32>, d: vec3<i32>) -> vec3<f32> {
    let i = u32((c.x * d.y + c.y) * d.z + c.z) * 3u;
    return vec3<f32>(cg[i], cg[i + 1u], cg[i + 2u]);
}
fn csample(q: vec3<f32>, d: vec3<i32>) -> vec3<f32> {
    let lo = vec3<i32>(floor(q)); let f = q - vec3<f32>(lo);
    let hi = min(lo + vec3<i32>(1), d - vec3<i32>(1));
    let x0 = mix(mix(cget(vec3<i32>(lo.x,lo.y,lo.z), d), cget(vec3<i32>(lo.x,lo.y,hi.z), d), f.z),
                 mix(cget(vec3<i32>(lo.x,hi.y,lo.z), d), cget(vec3<i32>(lo.x,hi.y,hi.z), d), f.z), f.y);
    let x1 = mix(mix(cget(vec3<i32>(hi.x,lo.y,lo.z), d), cget(vec3<i32>(hi.x,lo.y,hi.z), d), f.z),
                 mix(cget(vec3<i32>(hi.x,hi.y,lo.z), d), cget(vec3<i32>(hi.x,hi.y,hi.z), d), f.z), f.y);
    return mix(x0, x1, f.x);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let fine = vec3<f32>(waves[4], waves[5], waves[6]);
    let low = vec3<f32>(waves[7], waves[8], waves[9]);
    let d = vec3<i32>(P.b.xyz);
    let p = vec3<f32>(c) + low;
    let q = clamp(p * (vec3<f32>(d) - vec3<f32>(1.0)) / (fine - vec3<f32>(1.0)), vec3<f32>(0.0), vec3<f32>(d) - vec3<f32>(1.0));
    let v = csample(q, d);
    vc[i * 3u] = v.x; vc[i * 3u + 1u] = v.y; vc[i * 3u + 2u] = v.z;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_adv", [_c("vc"), _c("u0"), _c("u1", acc="rw")], """
fn uget(c: vec3<i32>) -> vec3<f32> {
    let i = cidx(c) * 3u;
    return vec3<f32>(u0[i], u0[i + 1u], u0[i + 2u]);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let d = vec3<i32>(P.a.xyz);
    let total = vec3<f32>(vc[i*3u] + u0[i*3u], vc[i*3u+1u] + u0[i*3u+1u], vc[i*3u+2u] + u0[i*3u+2u]);
    let p = clamp(vec3<f32>(c) - total * P.c.z, vec3<f32>(0.0), vec3<f32>(d) - vec3<f32>(1.0));
    let lo = vec3<i32>(floor(p)); let f = p - vec3<f32>(lo);
    let hi = min(lo + vec3<i32>(1), d - vec3<i32>(1));
    let x0 = mix(mix(uget(vec3<i32>(lo.x,lo.y,lo.z)), uget(vec3<i32>(lo.x,lo.y,hi.z)), f.z),
                 mix(uget(vec3<i32>(lo.x,hi.y,lo.z)), uget(vec3<i32>(lo.x,hi.y,hi.z)), f.z), f.y);
    let x1 = mix(mix(uget(vec3<i32>(hi.x,lo.y,lo.z)), uget(vec3<i32>(hi.x,lo.y,hi.z)), f.z),
                 mix(uget(vec3<i32>(hi.x,hi.y,lo.z)), uget(vec3<i32>(hi.x,hi.y,hi.z)), f.z), f.y);
    let r = mix(x0, x1, f.x);
    u1[i*3u] = r.x; u1[i*3u+1u] = r.y; u1[i*3u+2u] = r.z;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_src", [_c("waves"), _c("vc"), _c("u", acc="rw")], """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let low = vec3<f32>(waves[7], waves[8], waves[9]);
    let p = vec3<f32>(c) + low;
    var noise = vec3<f32>(0.0);
    let n = u32(waves[2]);
    for (var w = 0u; w < n; w++) {
        let o = 16u + w * 8u;
        let k = vec3<f32>(waves[o], waves[o+1u], waves[o+2u]);
        let amp = vec3<f32>(waves[o+3u], waves[o+4u], waves[o+5u]);
        noise += amp * cos(dot(k, p) + waves[o+6u]);
    }
    let speed = length(vec3<f32>(vc[i*3u], vc[i*3u+1u], vc[i*3u+2u]));
    let s = P.c.w * speed;
    u[i*3u] = u[i*3u] + s * noise.x; u[i*3u+1u] = u[i*3u+1u] + s * noise.y; u[i*3u+2u] = u[i*3u+2u] + s * noise.z;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_omega", [_c("vc"), _c("u"), _c("w", acc="rw")], """
fn tot(c: vec3<i32>) -> vec3<f32> {
    let i = cidx(c) * 3u;
    return vec3<f32>(vc[i] + u[i], vc[i+1u] + u[i+1u], vc[i+2u] + u[i+2u]);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let s = 0.5 / P.c.x;
    let dx = tot(c + vec3<i32>(1,0,0)) - tot(c - vec3<i32>(1,0,0));
    let dy = tot(c + vec3<i32>(0,1,0)) - tot(c - vec3<i32>(0,1,0));
    let dz = tot(c + vec3<i32>(0,0,1)) - tot(c - vec3<i32>(0,0,1));
    w[i*3u] = (dy.z - dz.y) * s; w[i*3u+1u] = (dz.x - dx.z) * s; w[i*3u+2u] = (dx.y - dy.x) * s;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_conf", [_c("w"), _c("u", acc="rw")], """
fn wmag(c: vec3<i32>) -> f32 { let i = cidx(c) * 3u; return length(vec3<f32>(w[i], w[i+1u], w[i+2u])); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let s = 0.5 / P.c.x;
    let eta = vec3<f32>(wmag(c + vec3<i32>(1,0,0)) - wmag(c - vec3<i32>(1,0,0)),
                        wmag(c + vec3<i32>(0,1,0)) - wmag(c - vec3<i32>(0,1,0)),
                        wmag(c + vec3<i32>(0,0,1)) - wmag(c - vec3<i32>(0,0,1))) * s;
    let unit = eta / (length(eta) + 1e-20);
    let f = P.c.y * P.c.x * cross(unit, vec3<f32>(w[i*3u], w[i*3u+1u], w[i*3u+2u]));
    u[i*3u] = (u[i*3u] + f.x) * P.d.x; u[i*3u+1u] = (u[i*3u+1u] + f.y) * P.d.x; u[i*3u+2u] = (u[i*3u+2u] + f.z) * P.d.x;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_div", [_c("u"), _c("dv", acc="rw")], """
fn ug(c: vec3<i32>) -> vec3<f32> { let i = cidx(c) * 3u; return vec3<f32>(u[i], u[i+1u], u[i+2u]); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    dv[i] = ((ug(c + vec3<i32>(1,0,0)).x - ug(c - vec3<i32>(1,0,0)).x) +
             (ug(c + vec3<i32>(0,1,0)).y - ug(c - vec3<i32>(0,1,0)).y) +
             (ug(c + vec3<i32>(0,0,1)).z - ug(c - vec3<i32>(0,0,1)).z)) * (0.5 / P.c.x);
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_jac", [_c("dv"), _c("pa"), _c("pb", acc="rw")], """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let s = pa[cidx(c + vec3<i32>(1,0,0))] + pa[cidx(c - vec3<i32>(1,0,0))] +
            pa[cidx(c + vec3<i32>(0,1,0))] + pa[cidx(c - vec3<i32>(0,1,0))] +
            pa[cidx(c + vec3<i32>(0,0,1))] + pa[cidx(c - vec3<i32>(0,0,1))];
    pb[i] = (s - P.c.x * P.c.x * dv[i]) / 6.0;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_grad", [_c("p"), _c("u", acc="rw")], """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let c = cell3(i);
    let s = 0.5 / P.c.x;
    u[i*3u] = u[i*3u] - (p[cidx(c + vec3<i32>(1,0,0))] - p[cidx(c - vec3<i32>(1,0,0))]) * s;
    u[i*3u+1u] = u[i*3u+1u] - (p[cidx(c + vec3<i32>(0,1,0))] - p[cidx(c - vec3<i32>(0,1,0))]) * s;
    u[i*3u+2u] = u[i*3u+2u] - (p[cidx(c + vec3<i32>(0,0,1))] - p[cidx(c - vec3<i32>(0,0,1))]) * s;
}
""", lib=_SWIRL_LIB, cellwise=False)

_kernel("pswirl_clamp", [_c("vc"), _c("u", acc="rw")], """
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = glin(gid);
    if (i >= P.a.x * P.a.y * P.a.z) { return; }
    let v = vec3<f32>(u[i*3u], u[i*3u+1u], u[i*3u+2u]);
    let cap = P.d.y * length(vec3<f32>(vc[i*3u], vc[i*3u+1u], vc[i*3u+2u]));
    let k = min(1.0, cap / max(length(v), 1e-20));
    u[i*3u] = v.x * k; u[i*3u+1u] = v.y * k; u[i*3u+2u] = v.z * k;
}
""", lib=_SWIRL_LIB, cellwise=False)


def swirl_persist(guide, factor, low, dims, waves, uf0, strength, voxel, dt, life, vmax, iters, spin, inject):
    """GPU twin of `fluid_upres.swirl_persist_cpu`: the persistent fine velocity on the dense box, shape (*dims, 3)."""
    ctx = _Ctx()
    coarse = tuple(int(n) for n in guide.shape[:3])
    count = int(np.prod(dims))
    if 12 * count > ctx.max_binding:
        raise Unsupported("the swirl box exceeds the adapter's storage buffer limit")
    K, C, phase = waves
    head = np.zeros(16, np.float32)
    head[:11] = (strength, 0.0, len(phase), voxel, *(n * factor for n in coarse), *low, 0.0)
    table = np.concatenate([head, np.concatenate([K, C, phase[:, None], np.zeros((len(phase), 1), np.float32)],
                                                 axis=1).astype(np.float32).reshape(-1)])
    wave_buf = ctx.buffer(table.nbytes)
    ctx.write(wave_buf, table)
    cg = ctx.buffer(guide.size * 4)
    ctx.write(cg, np.ascontiguousarray(guide, np.float32).reshape(-1))
    vc, u0, u1, w = (ctx.buffer(count * 12) for _ in range(4))
    ctx.write(u0, np.ascontiguousarray(uf0, np.float32).reshape(-1))
    dv, pa, pb = (ctx.buffer(count * 4) for _ in range(3))
    groups = (count + 255) // 256
    grid = ("wg", (min(groups, 65535), (groups + 65534) // 65535, 1))
    step = dt / voxel
    params = _u(a=dims, b=coarse, c=(voxel, spin * strength * dt, step, inject * strength * dt), d=(float(np.exp(-dt / life)), vmax))
    ctx.dispatch("pswirl_vc", {"cg": cg, "waves": wave_buf, "vc": vc}, params, grid)
    ctx.dispatch("pswirl_adv", {"vc": vc, "u0": u0, "u1": u1}, params, grid)
    ctx.dispatch("pswirl_src", {"waves": wave_buf, "vc": vc, "u": u1}, params, grid)
    ctx.dispatch("pswirl_omega", {"vc": vc, "u": u1, "w": w}, params, grid)
    ctx.dispatch("pswirl_conf", {"w": w, "u": u1}, params, grid)
    if iters > 0:
        ctx.dispatch("pswirl_div", {"u": u1, "dv": dv}, params, grid)
        src, dst = pa, pb
        for _ in range(int(iters)):
            ctx.dispatch("pswirl_jac", {"dv": dv, "pa": src, "pb": dst}, params, grid)
            src, dst = dst, src
        ctx.dispatch("pswirl_grad", {"p": src, "u": u1}, params, grid)
    ctx.dispatch("pswirl_clamp", {"vc": vc, "u": u1}, params, grid)
    out = ctx.read(u1, count * 12).view(np.float32).reshape((*(int(n) for n in dims), 3))
    return out.copy()


def reconstruct_sparse(fields, coords, fine, guide=None, previous=None, step=0.0, swirl=None):
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
    swirl_buf = ctx.buffer(16)
    if swirl is not None:
        if count * 12 > ctx.max_binding:
            raise Unsupported("the swirl velocity exceeds the adapter's storage buffer limit")
        swirl_buf = ctx.buffer(count * 12)
        ctx.write(swirl_buf, np.ascontiguousarray(swirl, np.float32).reshape(-1))
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
                              "vx": guide_buf[0], "vy": guide_buf[1], "vz": guide_buf[2], "sv": swirl_buf, "out": out},
                             _u(a=(*fine, len(coords)), b=(*coarse, start),
                                c=(step, 1.0 if prior is not None else 0.0,
                                   1.0 if swirl is not None and name == "density" else 0.0)),
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
