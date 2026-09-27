"""GPU raymarch of `scene3d.Volume` members (docs/FLUIDS_SPIKE.md, docs/3D_FOUNDATION.md).

The shader is a line-by-line port of the CPU reference in nodebased/volumerender.py, the parity oracle:
one primary ray per pixel centre, clipped to the volume's box and to the opaque mesh depth, marched front
to back in segments of at most `step_size` world units with the exact per-segment integral, single
scattering from every scene light with a shadow ray through the same grid (`shadow_steps` equal segments,
Spot cone and falloff from the shared `attenuation` function), composited over what is already on the
target with the premultiplied over operator. Volumes are drawn far to near; that composites to the same
image as the reference's near-to-far accumulation.

Density lives in one `r32float` 3D texture per volume, sampled with a manual zero-padded trilinear filter
(eight `textureLoad`s) so it needs no filtering feature and matches the reference exactly at the border.
Textures are cached per adapter by a digest of the density (and temperature) bytes, so scrubbing a cached
simulation, orbiting the camera or editing a light re-uploads nothing; `upload_count(state)` counts the
uploads for the tests and the status line.

Used by `gpu3d.render` (Render3D) and by the editor viewport (`viewportgpu`), which draws with a coarser
step and a multisampled depth buffer. Beauty only: the control passes and the `depth` output stay CPU.
"""
from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import replace
from collections import OrderedDict

import numpy as np

from . import scene3d

# Density lookups (one march sample or one shadow sample, each eight texel loads) one submission may cost,
# counted the way `work_estimate` counts them (box-clipped spans, so an upper bound: rays that stop early on
# opaque smoke cost less). Calibrated 2026-09-26 (docs/3D_FOUNDATION.md, "GPU volume raymarch"): RTX 3080 Ti
# and Radeon 8060S both ran 1.6e9 estimated lookups in about 0.06 s (1e10 to 2.7e10 per second), llvmpipe
# about 1.3e8 per second. The budgets are about half a second on each adapter type, because a submitted job
# cannot be cancelled and display drivers time out near two seconds; 'other' is a guess.
VOLUME_WORK_BUDGETS = {'discrete': 1e10, 'integrated': 8e9, 'cpu': 4e7, 'other': 2e9}
# Texture memory the cached density (and temperature) grids may occupy per adapter.
VOLUME_MEMORY_BUDGETS = {'discrete': 3 << 30, 'integrated': 1 << 30, 'cpu': 512 << 20, 'other': 512 << 20}
MAX_SEGMENTS = 100_000   # per ray; a step size that asks for more is refused as a mistake
_ESTIMATE_RAYS = 20_000

_SHADER = '''
struct Params {
    screen: vec4<f32>,     // width, height, aspect, focal
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    planes: vec4<f32>,     // near, far, ambient, light count
    march: vec4<f32>,      // step size, density scale, shadow density, shadow steps
    medium: vec4<f32>,     // scattering, absorption, lit (1 when the scene has lights), 0
    colour: vec4<f32>,
    extra: vec4<f32>,      // motion shutter in seconds, motion samples, pass flags (see march_sums), depth threshold
    scene: vec4<f32>,      // shadow ray epsilon (the scene's bias), mesh triangle count, 0, 0
};
struct Light { position: vec4<f32>, direction: vec4<f32>, colour: vec4<f32>, cone: vec4<f32>, shadow: vec4<f32> };
struct Vol {
    row0: vec4<f32>,       // inverse world-to-object rows: xyz, translation
    row1: vec4<f32>,
    row2: vec4<f32>,
    box_min: vec4<f32>,    // xyz, voxel size
    box_max: vec4<f32>,
    dims: vec4<f32>,       // grid size; w: 1 when the volume has a velocity field, plus 2 for a temperature field
    fwd0: vec4<f32>,       // forward object-to-world rows: xyz, translation
    fwd1: vec4<f32>,
    fwd2: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var depth_tex: %(DEPTH_TYPE)s;
struct Triangle { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32> };
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;
@group(1) @binding(0) var<uniform> vol: Vol;
@group(1) @binding(1) var density: texture_3d<f32>;
@group(1) @binding(2) var velocity_tex: texture_3d<f32>;
@group(1) @binding(3) var temperature_tex: texture_3d<f32>;
@group(1) @binding(4) var vorticity_tex: texture_3d<f32>;

fn to_object(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(vol.row0.xyz, p) + vol.row0.w, dot(vol.row1.xyz, p) + vol.row1.w,
                     dot(vol.row2.xyz, p) + vol.row2.w);
}
fn dir_to_object(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(vol.row0.xyz, d), dot(vol.row1.xyz, d), dot(vol.row2.xyz, d));
}
fn attenuation(lp: vec4<f32>, ld: vec4<f32>, cone: vec4<f32>, power: f32, point: vec3<f32>) -> f32 {
    // scene3d.light_attenuation: distance falloff, times the spot cone; cone = (is spot, inner, outer, exponent) in degrees.
    if (lp.w <= 0.0) { return 1.0; }
    let offset = point - lp.xyz;
    let dist = length(offset);
    var result = 1.0;
    if (power > 0.0) { result = min(1.0, pow(max(dist, 1e-8), -power)); }
    if (cone.x > 0.0) {
        let angle = degrees(acos(clamp(dot(offset, ld.xyz) / max(dist, 1e-12), -1.0, 1.0)));
        var k = select(0.0, 1.0, angle <= cone.y);
        if (cone.z > cone.y) {
            let t = clamp((cone.z - angle) / (cone.z - cone.y), 0.0, 1.0);
            k = select(0.0, pow(t * t * (3.0 - 2.0 * t), cone.w), t > 0.0);
        }
        result = result * k;
    }
    return result;
}
// Zero-padded trilinear sample of a cell-centred grid at object-space point p (volumerender._trilinear).
%(SAMPLERS)s
// The density at p averaged over the shutter (volumerender._shutter_points): read at p - v t for the mid-points
// t of the shutter; sample_density_sharp when no shutter is open or the volume has no velocity.
fn shutter_shift(p: vec3<f32>) -> vec3<f32> {
    return sample_velocity(p) * params.extra.x;
}
fn sample_density(p: vec3<f32>) -> f32 {
    if (params.extra.x <= 0.0 || (i32(vol.dims.w) & 1) == 0) { return sample_density_sharp(p); }
    let shift = shutter_shift(p);
    let n = i32(params.extra.y);
    var total = 0.0;
    for (var s = 0; s < n; s += 1) { total += sample_density_sharp(p - shift * ((f32(s) + 0.5) / f32(n))); }
    return total / f32(n);
}
fn sample_temperature(p: vec3<f32>) -> f32 {
    if (params.extra.x <= 0.0 || (i32(vol.dims.w) & 1) == 0) { return sample_temperature_sharp(p); }
    let shift = shutter_shift(p);
    let n = i32(params.extra.y);
    var total = 0.0;
    for (var s = 0; s < n; s += 1) { total += sample_temperature_sharp(p - shift * ((f32(s) + 0.5) / f32(n))); }
    return total / f32(n);
}
// (enter, exit) of a ray against one axis slab; a parallel ray is inside for every t or outside for none.
fn slab(o: f32, d: f32, lo: f32, hi: f32) -> vec2<f32> {
    if (abs(d) < 1e-12) {
        if (o >= lo && o <= hi) { return vec2<f32>(-3.0e38, 3.0e38); }
        return vec2<f32>(3.0e38, -3.0e38);
    }
    let a = (lo - o) / d;
    let b = (hi - o) / d;
    return vec2<f32>(min(a, b), max(a, b));
}
// volumerender._shadow_transmittance: exp(-shadow_density * sigma_t_unit * scale * integral(density)) toward the light.
fn shadow(p: vec3<f32>, light: Light, sigma_t_unit: f32) -> f32 {
    var to_light = -light.direction.xyz;
    var limit = 3.0e38;
    if (light.position.w > 0.0) {
        let offset = light.position.xyz - p;
        let dist = max(length(offset), 1e-12);
        to_light = offset / dist;
        limit = dist;
    }
    let o = to_object(p);
    let d = dir_to_object(to_light);
    var far = 3.0e38;
    if (abs(d.x) >= 1e-12) { far = min(far, slab(o.x, d.x, vol.box_min.x, vol.box_max.x).y); }
    if (abs(d.y) >= 1e-12) { far = min(far, slab(o.y, d.y, vol.box_min.y, vol.box_max.y).y); }
    if (abs(d.z) >= 1e-12) { far = min(far, slab(o.z, d.z, vol.box_min.z, vol.box_max.z).y); }
    let length_t = max(min(far, limit), 0.0);
    let steps = i32(params.march.w);
    var total = 0.0;
    for (var j = 0; j < steps; j += 1) {
        total += sample_density_sharp(o + d * ((f32(j) + 0.5) / f32(steps) * length_t));
    }
    return exp(-params.march.z * sigma_t_unit * params.march.y * total * (length_t / f32(steps)));
}
// Meshes between a smoke sample and a light (scene3d._volume_occluders): hard shadows, two-sided
// Moller-Trumbore over every mesh triangle, material alpha only; `direction.w` is the light's Shadows switch.
fn scene_shadow(p: vec3<f32>, light: Light) -> f32 {
    let count = u32(params.scene.y);
    if (light.direction.w <= 0.0 || count == 0u) { return 1.0; }
    let near_bias = params.scene.x * light.shadow.x * 0.01;
    var ray = -light.direction.xyz;
    var limit = 0.0;
    if (light.position.w > 0.0) {
        let delta = light.position.xyz - p;
        limit = length(delta);
        ray = delta / max(limit, 1e-8);
    }
    var transmission = 1.0;
    for (var j = 0u; j < count; j += 1u) {
        let tri = triangles[j];
        let h = cross(ray, tri.e2.xyz);
        let det = dot(h, tri.e1.xyz);
        if (abs(det) > 1e-10) {
            let inverse = 1.0 / det;
            let delta = p - tri.v0.xyz;
            let u = dot(delta, h) * inverse;
            let q = cross(delta, tri.e1.xyz);
            let v = dot(ray, q) * inverse;
            let t = dot(tri.e2.xyz, q) * inverse;
            if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > near_bias && (light.position.w == 0.0 || t < limit)) {
                transmission *= 1.0 - tri.v0.w;
            }
        }
    }
    return transmission;
}
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // The beauty march never reads the temperature or vorticity grids; naming them keeps them in the shared
    // bind group layout that `layout='auto'` would otherwise shrink.
    if (params.extra.z < -1.0 && textureDimensions(temperature_tex).x + textureDimensions(vorticity_tex).x == 0u) {
        return vec4<f32>(0.0);
    }
    let width = params.screen.x;
    let height = params.screen.y;
    let xs = frag.x / width * 2.0 - 1.0;
    let ys = 1.0 - frag.y / height * 2.0;
    let world = params.right.xyz * (xs * params.screen.z / params.screen.w) + params.up.xyz * (ys / params.screen.w)
        + params.forward.xyz;
    let ray_length = length(world);
    let dir = world / ray_length;
    let eye = params.eye.xyz;
    // Nearest opaque mesh: the raster depth buffer holds far*(z-near)/((far-near)*z), inverted to view depth z.
    var t_mesh = 3.0e38;
    let pixel = vec2<i32>(frag.xy);
    let near = params.planes.x;
    let far = params.planes.y;
    %(DEPTH_LOAD)s
    let o = to_object(eye);
    let d = dir_to_object(dir);
    let sx = slab(o.x, d.x, vol.box_min.x, vol.box_max.x);
    let sy = slab(o.y, d.y, vol.box_min.y, vol.box_max.y);
    let sz = slab(o.z, d.z, vol.box_min.z, vol.box_max.z);
    let t0 = max(max(max(sx.x, sy.x), sz.x), 0.0);
    let t1 = min(min(min(sx.y, sy.y), sz.y), t_mesh);
    if (!(t1 > t0)) { return vec4<f32>(0.0); }
    let seg_len = params.march.x;
    let count = i32(ceil((t1 - t0) / seg_len - 1e-9));
    let scattering = params.medium.x;
    let absorption = params.medium.y;
    let sigma_t_unit = scattering + absorption;
    let lit = params.medium.z > 0.5;
    let light_count = i32(params.planes.w);
    var trans = 1.0;
    var rgb = vec3<f32>(0.0);
    for (var k = 0; k < count; k += 1) {
        let ds = clamp(t1 - (t0 + f32(k) * seg_len), 0.0, seg_len);
        if (ds <= 0.0 || trans <= 1e-6) { break; }
        let tm = t0 + f32(k) * seg_len + ds * 0.5;
        let p_world = eye + dir * tm;
        let sigma = params.march.y * sample_density(to_object(p_world));
        if (sigma <= 0.0) { continue; }
        let st = sigma_t_unit * sigma;
        let a = st * ds;
        let t_seg = exp(-a);
        let one_minus = select(1.0 - t_seg, a * (1.0 - a * (0.5 - a / 6.0)), a < 1e-3);
        let frac = select(0.0, scattering * sigma / st, st > 0.0);
        var source = params.colour.rgb;
        if (lit) {
            var incident = vec3<f32>(params.planes.z);
            for (var i = 0; i < light_count; i += 1) {
                let light = lights[i];
                let attn = attenuation(light.position, light.direction, light.cone, light.colour.w, p_world);
                if (attn > 0.0) {
                    incident += light.colour.rgb * (attn * shadow(p_world, light, sigma_t_unit) * scene_shadow(p_world, light));
                }
            }
            source = params.colour.rgb * incident;
        }
        rgb += trans * (frac * one_minus) * source;
        trans *= t_seg;
    }
    return vec4<f32>(rgb, 1.0 - trans);
}

// The control passes (volumerender.integrate, wanted terms): one volume's sums along the pixel ray, cut at the mesh
// depth. `flags`: bit 0 temperature, bit 1 vorticity, bit 2 motion (position and velocity sums), bit 3 depth, bit 4 the density is blurred by the shutter (density, temperature and depth).
struct PassSums { density: f32, temperature: f32, vorticity: f32, first_t: f32, position: vec3<f32>, velocity: vec3<f32> };
fn march_sums(frag: vec4<f32>) -> PassSums {
    var out: PassSums;
    out.first_t = 3.0e38;
    let width = params.screen.x;
    let height = params.screen.y;
    let xs = frag.x / width * 2.0 - 1.0;
    let ys = 1.0 - frag.y / height * 2.0;
    let world = params.right.xyz * (xs * params.screen.z / params.screen.w) + params.up.xyz * (ys / params.screen.w)
        + params.forward.xyz;
    let ray_length = length(world);
    let dir = world / ray_length;
    let eye = params.eye.xyz;
    var t_mesh = 3.0e38;
    let pixel = vec2<i32>(frag.xy);
    let near = params.planes.x;
    let far = params.planes.y;
    %(DEPTH_LOAD)s
    let o = to_object(eye);
    let d = dir_to_object(dir);
    let sx = slab(o.x, d.x, vol.box_min.x, vol.box_max.x);
    let sy = slab(o.y, d.y, vol.box_min.y, vol.box_max.y);
    let sz = slab(o.z, d.z, vol.box_min.z, vol.box_max.z);
    let t0 = max(max(max(sx.x, sy.x), sz.x), 0.0);
    let t1 = min(min(min(sx.y, sy.y), sz.y), t_mesh);
    if (!(t1 > t0)) { return out; }
    let seg_len = params.march.x;
    let count = i32(ceil((t1 - t0) / seg_len - 1e-9));
    let flags = u32(params.extra.z);
    var velocity_obj = vec3<f32>(0.0);
    for (var k = 0; k < count; k += 1) {
        let ds = clamp(t1 - (t0 + f32(k) * seg_len), 0.0, seg_len);
        if (ds <= 0.0) { break; }
        let tm = t0 + f32(k) * seg_len + ds * 0.5;
        let p_world = eye + dir * tm;
        let p = to_object(p_world);
        var sigma = params.march.y * sample_density_sharp(p);
        if ((flags & 16u) != 0u) { sigma = params.march.y * sample_density(p); }   // the shutter, for the passes it blurs
        let w = sigma * ds;
        out.density += w;
        if ((flags & 1u) != 0u && (i32(vol.dims.w) & 2) != 0) { out.temperature += sample_temperature(p) * w; }
        if ((flags & 2u) != 0u && (i32(vol.dims.w) & 1) != 0) { out.vorticity += sample_vorticity_sharp(p) * ds; }
        if ((flags & 4u) != 0u) {
            out.position += p_world * w;
            if ((i32(vol.dims.w) & 1) != 0) { velocity_obj += sample_velocity(p) * w; }
        }
        if ((flags & 8u) != 0u && sigma >= params.extra.w && out.first_t > 1.0e38) { out.first_t = tm; }
    }
    out.velocity = vec3<f32>(dot(vol.fwd0.xyz, velocity_obj), dot(vol.fwd1.xyz, velocity_obj), dot(vol.fwd2.xyz, velocity_obj));
    return out;
}
@fragment fn fs_sums(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let s = march_sums(frag);
    return vec4<f32>(s.density, s.temperature, s.vorticity, s.first_t);
}
struct MotionOut { @location(0) position: vec4<f32>, @location(1) velocity: vec4<f32> };
@fragment fn fs_motion(@builtin(position) frag: vec4<f32>) -> MotionOut {
    let s = march_sums(frag);
    return MotionOut(vec4<f32>(s.position, s.density), vec4<f32>(s.velocity, 0.0));
}
'''

# The control passes read a view-depth image (scene3d.render's depth buffer, inf where empty) uploaded as r32float.
_DEPTH_VIEW = ('texture_2d<f32>',
               'let raw = textureLoad(depth_tex, pixel, 0).r;\n'
               '    if (raw < 3.0e37) { t_mesh = raw * ray_length; }')
_DEPTH_SINGLE = ('texture_depth_2d',
                 'let raw = textureLoad(depth_tex, pixel, 0);\n'
                 '    if (raw < 1.0) { t_mesh = far * near / (far - raw * (far - near)) * ray_length; }')
# The viewport's depth buffer is multisampled: the nearest of the four samples hides the smoke, so a
# silhouette edge is never smoked over.
_DEPTH_MULTI = ('texture_depth_multisampled_2d',
                'var raw = 1.0;\n'
                '    for (var s = 0; s < 4; s += 1) { raw = min(raw, textureLoad(depth_tex, pixel, s)); }\n'
                '    if (raw < 1.0) { t_mesh = far * near / (far - raw * (far - near)) * ray_length; }')


def _samplers():
    pieces = []
    for name, suffix, texture, ret, zero, swizzle in (
            ('density', '_sharp', 'density', 'f32', '0.0', '.r'),
            ('temperature', '_sharp', 'temperature_tex', 'f32', '0.0', '.r'),
            ('vorticity', '_sharp', 'vorticity_tex', 'f32', '0.0', '.r'),
            ('velocity', '', 'velocity_tex', 'vec3<f32>', 'vec3<f32>(0.0)', '.xyz')):
        pieces.append(f'''fn sample_{name}{suffix}(p: vec3<f32>) -> {ret} {{
    let g = (p - vol.box_min.xyz) / vol.box_min.w - vec3<f32>(0.5);
    let base = floor(g);
    let f = g - base;
    let i0 = vec3<i32>(base);
    let dims = vec3<i32>(vol.dims.xyz);
    var total = {zero};
    for (var c = 0; c < 8; c += 1) {{
        let dx = c & 1;
        let dy = (c >> 1) & 1;
        let dz = (c >> 2) & 1;
        let idx = i0 + vec3<i32>(dx, dy, dz);
        if (idx.x < 0 || idx.y < 0 || idx.z < 0 || idx.x >= dims.x || idx.y >= dims.y || idx.z >= dims.z) {{ continue; }}
        let w = select(1.0 - f.x, f.x, dx == 1) * select(1.0 - f.y, f.y, dy == 1) * select(1.0 - f.z, f.z, dz == 1);
        total += w * textureLoad({texture}, idx, 0){swizzle};
    }}
    return total;
}}
''')
    return ''.join(pieces)


def _volumes(scene):
    return tuple(getattr(scene, 'volumes', ()) or ())


def adapter_kind(state):
    normalized = str(state['info'].get('adapter_type', 'unknown')).lower().replace('_', '').replace(' ', '')
    return {'discretegpu': 'discrete', 'integratedgpu': 'integrated', 'cpu': 'cpu'}.get(normalized, 'other')


def _bytes(volume, settings, temperature=False, vorticity=False):
    """Texture bytes the shader reads for `volume`: density, plus the velocity (motion blur, motion and
    vorticity passes), temperature and vorticity grids when wanted."""
    total = volume.density.nbytes
    if volume.velocity is not None and (settings.blurred(volume) or vorticity):
        total += volume.density.nbytes * 4 + (volume.density.nbytes if vorticity else 0)
    if temperature and volume.temperature is not None:
        total += volume.density.nbytes
    return total


def check(state, scene, settings, temperature=False, vorticity=False):
    """Raise `gpu3d.Unsupported` when a grid exceeds the adapter's 3D texture or memory limits."""
    from .gpu3d import Unsupported
    settings.validated()
    limits = state['device'].limits
    top = int(limits.get('max-texture-dimension-3d', 256))
    total = 0
    for volume in _volumes(scene):
        if max(volume.shape) > top:
            raise Unsupported(f'volume grid {volume.shape} exceeds the adapter 3D texture limit of {top} voxels per side')
        total += _bytes(volume, settings, temperature, vorticity)
    budget = VOLUME_MEMORY_BUDGETS[adapter_kind(state)]
    if total > budget:
        raise Unsupported(f'volume grids need {total:,} bytes of texture memory, over the {budget:,} byte '
                          f'budget of this {adapter_kind(state)} adapter')


def upload_count(state):
    return state.get('volume_uploads', 0)


def cache_bytes(state):
    return sum(entry[2] for entry in state.get('volume_textures', {}).values())


_digests = weakref.WeakKeyDictionary()


def _digest(volume, name='density'):
    """Content key of an array the shader reads; memoised on the (immutable) Volume object."""
    keys = _digests.setdefault(volume, {})
    key = keys.get(name)
    if key is None:
        array = getattr(volume, name)
        h = hashlib.blake2b(digest_size=16)
        h.update(name.encode())
        h.update(str(array.shape).encode())
        h.update(memoryview(np.ascontiguousarray(array)).cast('B'))
        key = keys[name] = h.hexdigest()
    return key


def _field(state, key, make, used):
    """A cached 3D texture (texture, view, bytes) for `key`; `make()` builds `(array x-fastest, format)` on a miss."""
    wgpu, device = state['wgpu'], state['device']
    cache = state.setdefault('volume_textures', OrderedDict())
    used.add(key)
    entry = cache.get(key)
    if entry is None:
        data, fmt = make()
        nz, ny, nx = data.shape[:3]
        channels = 1 if fmt == 'r32float' else 4
        gpu_texture = device.create_texture(
            size=(nx, ny, nz), dimension='3d', format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        device.queue.write_texture({'texture': gpu_texture}, data,
                                   {'bytes_per_row': nx * 4 * channels, 'rows_per_image': ny}, (nx, ny, nz))
        entry = cache[key] = (gpu_texture, gpu_texture.create_view(dimension='3d'), data.nbytes)
        state['volume_uploads'] = upload_count(state) + 1
    cache.move_to_end(key)
    budget = VOLUME_MEMORY_BUDGETS[adapter_kind(state)]
    for old in [k for k in cache if k not in used]:
        if sum(e[2] for e in cache.values()) <= budget:
            break
        cache.pop(old)[0].destroy()
    return entry[1]


_ONE = {'r32float': np.zeros((1, 1, 1), np.float32), 'rgba32float': np.zeros((1, 1, 1, 4), np.float32)}


def texture(state, volume, used):
    """The cached `r32float` 3D density texture view of `volume`; uploads only on a content miss."""
    # [ix, iy, iz] C order has iz fastest; the texture wants x fastest.
    return _field(state, _digest(volume), lambda: (
        np.ascontiguousarray(volume.density.transpose(2, 1, 0), np.float32), 'r32float'), used)


def extra_textures(state, volume, used, velocity, temperature, vorticity):
    """Views of the velocity (rgba32float), temperature and vorticity (r32float) textures; a field that is not
    needed, or that the volume lacks, binds a one-texel dummy so every draw has the same layout."""
    def dummy(fmt):
        return _field(state, ('dummy', fmt), lambda: (_ONE[fmt], fmt), used)
    if velocity and volume.velocity is not None:
        def make_velocity():
            data = np.zeros(volume.shape[::-1] + (4,), np.float32)
            data[..., :3] = volume.velocity.transpose(2, 1, 0, 3)
            return data, 'rgba32float'
        vel = _field(state, _digest(volume, 'velocity'), make_velocity, used)
    else:
        vel = dummy('rgba32float')
    if temperature and volume.temperature is not None:
        temp = _field(state, _digest(volume, 'temperature'), lambda: (
            np.ascontiguousarray(volume.temperature.transpose(2, 1, 0), np.float32), 'r32float'), used)
    else:
        temp = dummy('r32float')
    if vorticity and volume.velocity is not None:
        from . import volumerender
        vort = _field(state, ('vorticity', _digest(volume, 'velocity'), volume.voxel_size), lambda: (
            np.ascontiguousarray(volumerender.vorticity_magnitude(volume).transpose(2, 1, 0), np.float32),
            'r32float'), used)
    else:
        vort = dummy('r32float')
    return vel, temp, vort


def pipeline(state, target, samples=1, multisampled_depth=False):
    key = ('volume', target, samples, multisampled_depth)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    depth_type, depth_load = _DEPTH_MULTI if multisampled_depth else _DEPTH_SINGLE
    module = device.create_shader_module(code=_SHADER % {'DEPTH_TYPE': depth_type, 'DEPTH_LOAD': depth_load,
                                                         'SAMPLERS': _samplers()})
    blend = {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}
    state['pipelines'][key] = device.create_render_pipeline(
        layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
        primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
        multisample={'count': samples},
        fragment={'module': module, 'entry_point': 'fs',
                  'targets': [{'format': target, 'blend': {'color': blend, 'alpha': blend}}]})
    return state['pipelines'][key]


def work_estimate(scene, camera, width, height, settings, light_count, triangle_lights=0):
    """Upper bound of density lookups for the frame: box-clipped ray spans on a coarse ray grid, scaled up.
    `triangle_lights` is mesh triangles times shadowed lights: each lit sample tests them all for the meshes'
    shadows, counted as one lookup each."""
    from . import volumerender
    volumes = _volumes(scene)
    if not volumes:
        return 0.0
    scale = max(1, int(math.sqrt(width * height / _ESTIMATE_RAYS)))
    w, h = max(1, width // scale), max(1, height // scale)
    eye, dirs, _length = volumerender._pixel_rays(camera, w, h)
    total = 0.0
    for volume in volumes:
        blur = 1 + int(settings.motion_samples) if settings.blurred(volume) else 1
        per_sample = blur + light_count * int(settings.shadow_steps) + triangle_lights
        prep = volumerender._Volume(volume, False)
        t0, t1 = prep.clip(eye, dirs)
        span = np.clip(t1 - t0, 0.0, None)
        segments = np.ceil(span / settings.step_size)
        if segments.size and float(segments.max()) > MAX_SEGMENTS:
            raise ValueError(f'volume_step_size {settings.step_size} needs more than {MAX_SEGMENTS:,} march '
                             'segments per ray; raise volume_step_size')
        total += float(segments.sum()) * per_sample * (width * height) / (w * h)
    return total


def band_plan(state, work, height):
    """Row bands whose volume work each fit one submission; raises ValueError when even the cap cannot."""
    kind = adapter_kind(state)
    budget = VOLUME_WORK_BUDGETS[kind]
    from .gpu3d import GPU_MAX_BANDS
    bands = max(1, math.ceil(work / budget))
    if bands > min(height, GPU_MAX_BANDS):
        raise ValueError(f'Volume raymarch exceeds the GPU budget: adapter {state["info"].get("adapter_type", "unknown")} '
                         f'({kind}), {work:,.0f} density lookups > {budget * GPU_MAX_BANDS:,.0f} even split into '
                         f'{GPU_MAX_BANDS} bands; raise volume_step_size, cut volume_shadow_steps or the resolution')
    return bands


class Prepared:
    """Everything a frame's volume draw needs; `record(render_pass)` issues one draw per volume."""

    def __init__(self, pipeline_, group0, groups1):
        self.pipeline, self.group0, self.groups1 = pipeline_, group0, groups1

    def record(self, render_pass):
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.group0)
        for group in self.groups1:
            render_pass.set_bind_group(1, group)
            render_pass.draw(3)


def prepare(state, scene, camera, width, height, ambient, settings, light_buffer, light_count, lit,
            depth_view, keep, *, target, samples=1, multisampled_depth=False, used=None, pipe=None, pass_flags=0,
            shadow_buffer=None, shadow_count=0, shadow_bias=0.0):
    """Upload (cached) the density textures and build the bind groups for a frame's volumes, far to near.

    `light_buffer` is the gpu3d light table (20 floats per light, intensity folded into the colour);
    `keep` registers a per-frame resource for destruction; `depth_view` is the depth attachment's texture
    view created with TEXTURE_BINDING usage."""
    wgpu, device = state['wgpu'], state['device']
    settings = settings.validated()
    volumes = _volumes(scene)
    used = set() if used is None else used
    eye, view = scene3d._view_basis(camera)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    params = np.zeros((11, 4), 'f4')
    params[0] = width, height, width / max(height, 1), focal
    params[1, :3] = eye
    params[2, :3], params[3, :3], params[4, :3] = view[0], view[1], -view[2]
    params[5] = camera.near, camera.far, ambient, light_count
    params[6] = settings.step_size, settings.density_scale, settings.shadow_density, settings.shadow_steps
    params[7] = settings.scattering, settings.absorption, 1.0 if lit else 0.0, 0.0
    params[8, :3] = settings.color
    params[9] = (settings.motion_blur / settings.fps if settings.motion_blur > 0 else 0.0, settings.motion_samples,
                 pass_flags, settings.depth_threshold)
    params[10, :2] = shadow_bias, shadow_count
    pipe = pipe if pipe is not None else pipeline(state, target, samples, multisampled_depth)
    params_buffer = keep(device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM))
    group0 = device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {'binding': 0, 'resource': {'buffer': params_buffer}},
        {'binding': 2, 'resource': depth_view}] + (
        [] if light_buffer is None else [
            {'binding': 1, 'resource': {'buffer': light_buffer}},
            # The mesh triangle table the raster shadows use (gpu3d._shadow_data); a zero triangle when there is none.
            {'binding': 3, 'resource': {'buffer': shadow_buffer if shadow_buffer is not None else keep(
                device.create_buffer_with_data(data=np.zeros((1, 12), 'f4'), usage=wgpu.BufferUsage.STORAGE))}}]))
    eye64 = eye.astype(np.float64)
    order = sorted(volumes, key=lambda v: -float(np.linalg.norm(
        (np.asarray(v.matrix, np.float64) @ np.append((np.array(v.origin) + np.array(v.shape) * v.voxel_size / 2), 1.0))[:3]
        - eye64)))
    groups1 = []
    for volume in order:
        inverse = np.linalg.inv(np.asarray(volume.matrix, np.float64))
        block = np.zeros((9, 4), 'f4')
        block[0:3] = inverse[:3, :]
        block[6:9] = np.asarray(volume.matrix, np.float64)[:3, :]
        box_min = np.array(volume.origin, np.float64)
        box_max = box_min + np.array(volume.shape, np.float64) * volume.voxel_size
        block[3, :3], block[3, 3] = box_min, volume.voxel_size
        block[4, :3] = box_max
        block[5, :3] = volume.shape
        block[5, 3] = (volume.velocity is not None) + 2 * (volume.temperature is not None)
        uniform = keep(device.create_buffer_with_data(data=block, usage=wgpu.BufferUsage.UNIFORM))
        groups1.append(device.create_bind_group(layout=pipe.get_bind_group_layout(1), entries=[
            {'binding': 0, 'resource': {'buffer': uniform}},
            {'binding': 1, 'resource': texture(state, volume, used)}] + [
                {'binding': 2 + i, 'resource': view} for i, view in enumerate(extra_textures(
                    state, volume, used, settings.blurred(volume) or bool(pass_flags & 6),
                    bool(pass_flags & 1), bool(pass_flags & 2)))]))
    return Prepared(pipe, group0, groups1)


def _pass_pipeline(state, entry, targets):
    """The control-pass pipeline: `entry` writes `targets` rgba32float attachments, no blending (each volume gets
    its own image and the host adds them, so nothing needs the float32-blendable feature)."""
    key = ('volume-pass', entry, targets)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    depth_type, depth_load = _DEPTH_VIEW
    module = device.create_shader_module(code=_SHADER % {'DEPTH_TYPE': depth_type, 'DEPTH_LOAD': depth_load,
                                                         'SAMPLERS': _samplers()})
    state['pipelines'][key] = device.create_render_pipeline(
        layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
        primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
        multisample={'count': 1},
        fragment={'module': module, 'entry_point': entry,
                  'targets': [{'format': 'rgba32float'} for _ in range(targets)]})
    return state['pipelines'][key]


def render_passes(state, scene, camera, width, height, mesh_depth, settings, name, cancel=None):
    """The accumulators `volumerender.integrate` returns for the control pass `name` (a `VOLUME_PASSES` entry, or
    'depth' for the merged depth output), computed on the GPU, as a dict of the same arrays (float32).

    `mesh_depth` is the opaque meshes' view-space depth (H, W), inf where empty, or None. Every volume is drawn
    into its own rgba32float image in row bands sized by the work budget, and the host adds the images in the
    reference's order. The motion pass ignores the shutter (it carries the unblurred vectors)."""
    from .gpu3d import GPU_MAX_BANDS, _cancel
    from . import volumerender
    wgpu, device = state['wgpu'], state['device']
    settings = settings.validated()
    key = 'depth' if name == 'depth' else name.split('_', 1)[1]
    if key == 'motion':
        settings = replace(settings, motion_blur=0.0)
    flags = {'density': 16, 'temperature': 17, 'vorticity': 2, 'motion': 4, 'depth': 24, 'id': 0}[key]
    volumes = _volumes(scene)
    check(state, scene, settings, temperature=key == 'temperature', vorticity=key == 'vorticity')
    work = work_estimate(scene, camera, width, height, settings, 0)
    bands = band_plan(state, work, height)
    used = set()
    resources = []

    def keep(resource):
        resources.append(resource)
        return resource
    try:
        depth = np.full((height, width), 3.0e38, np.float32)
        if mesh_depth is not None:
            depth[:] = np.where(np.isfinite(mesh_depth), mesh_depth, 3.0e38)
        depth_texture = keep(device.create_texture(
            size=(width, height, 1), format='r32float',
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
        device.queue.write_texture({'texture': depth_texture}, depth,
                                   {'bytes_per_row': width * 4, 'rows_per_image': height}, (width, height, 1))
        targets = 2 if key == 'motion' else 1
        pipe = _pass_pipeline(state, 'fs_motion' if key == 'motion' else 'fs_sums', targets)
        prepared = prepare(state, scene, camera, width, height, 0.0, settings, None, 0, False,
                           depth_texture.create_view(), keep, target='rgba32float', used=used, pipe=pipe,
                           pass_flags=flags)
        # `prepare` draws far to near; the host wants the reference's near-to-far order for the depth and id.
        order = list(range(len(prepared.groups1)))
        images = [[np.empty((height, width, 4), 'f4') for _ in range(len(order))] for _ in range(targets)]
        stride = ((width * 16 + 255) // 256) * 256
        rows_max = max(1, -(-height // bands))
        staging = keep(device.create_buffer(size=stride * rows_max,
                                            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
        splits = [(i * height // bands, (i + 1) * height // bands) for i in range(bands)]
        for slot, group in enumerate(prepared.groups1):
            attachments = [keep(device.create_texture(
                size=(width, height, 1), format='rgba32float',
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)) for _ in range(targets)]
            views = [a.create_view() for a in attachments]
            for y0, y1 in splits:
                _cancel(cancel)
                rows = y1 - y0
                encoder = device.create_command_encoder()
                rp = encoder.begin_render_pass(color_attachments=[
                    {'view': v, 'resolve_target': None, 'clear_value': (0, 0, 0, 0) if y0 == 0 else (0, 0, 0, 0),
                     'load_op': 'clear' if y0 == 0 else 'load', 'store_op': 'store'} for v in views])
                rp.set_scissor_rect(0, y0, width, rows)
                rp.set_pipeline(prepared.pipeline)
                rp.set_bind_group(0, prepared.group0)
                rp.set_bind_group(1, group)
                rp.draw(3)
                rp.end()
                device.queue.submit([encoder.finish()])
            for index, attachment in enumerate(attachments):
                for y0, y1 in splits:
                    rows = y1 - y0
                    encoder = device.create_command_encoder()
                    encoder.copy_texture_to_buffer({'texture': attachment, 'origin': (0, y0, 0)},
                                                   {'buffer': staging, 'bytes_per_row': stride, 'rows_per_image': rows},
                                                   (width, rows, 1))
                    device.queue.submit([encoder.finish()])
                    staging.map_sync(wgpu.MapMode.READ)
                    try:
                        raw = np.frombuffer(staging.read_mapped(), 'f4', count=rows * stride // 4)
                        images[index][slot][y0:y1] = raw.reshape(rows, stride // 4)[:, :width * 4].reshape(rows, width, 4)
                    finally:
                        staging.unmap()
        return volumerender.gpu_accumulators(scene, camera, width, height, mesh_depth, key, images, prepared_order(scene, camera))
    finally:
        for resource in reversed(resources):
            resource.destroy()


def prepared_order(scene, camera):
    """Original scene.volumes indices in the order `prepare` draws them (far to near by box centre)."""
    eye = scene3d._view_basis(camera)[0].astype(np.float64)
    volumes = _volumes(scene)

    def distance(i):
        v = volumes[i]
        centre = (np.asarray(v.matrix, np.float64)
                  @ np.append(np.array(v.origin) + np.array(v.shape) * v.voxel_size / 2, 1.0))[:3]
        return float(np.linalg.norm(centre - eye))
    return sorted(range(len(volumes)), key=lambda i: -distance(i))


MESH_SHADOW_VOLUMES = 4   # volumes the raster shader can read for the smoke's shadow on meshes

# Spliced into the raster mesh shader (gpu3d._pipeline) when the scene's smoke shadows meshes: the optical depth
# of up to four volumes from a shaded point toward a light (volumerender.ShadowCasters). Group 1: the volume table
# and one r32float density texture per slot (a one-texel dummy in unused slots).
MESH_SHADOW_WGSL = '''
struct VolShadow { row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, box_min: vec4<f32>, box_max: vec4<f32> };
struct VolShadowSet { count: vec4<f32>, march: vec4<f32>, vols: array<VolShadow, 4> };   // march: k, steps
@group(1) @binding(0) var<uniform> vs_set: VolShadowSet;
@group(1) @binding(1) var vs_tex0: texture_3d<f32>;
@group(1) @binding(2) var vs_tex1: texture_3d<f32>;
@group(1) @binding(3) var vs_tex2: texture_3d<f32>;
@group(1) @binding(4) var vs_tex3: texture_3d<f32>;
fn vs_load(i: u32, idx: vec3<i32>) -> f32 {
    switch i {
        case 0u: { return textureLoad(vs_tex0, idx, 0).r; }
        case 1u: { return textureLoad(vs_tex1, idx, 0).r; }
        case 2u: { return textureLoad(vs_tex2, idx, 0).r; }
        default: { return textureLoad(vs_tex3, idx, 0).r; }
    }
}
// volumerender._trilinear: zero-padded trilinear sample of volume i at its object-space point p.
fn vs_density(i: u32, p: vec3<f32>) -> f32 {
    let vol = vs_set.vols[i];
    let g = (p - vol.box_min.xyz) / vol.box_min.w - vec3<f32>(0.5);
    let base = floor(g);
    let f = g - base;
    let i0 = vec3<i32>(base);
    let dims = vec3<i32>(round((vol.box_max.xyz - vol.box_min.xyz) / vol.box_min.w));
    var total = 0.0;
    for (var c = 0; c < 8; c += 1) {
        let dx = c & 1;
        let dy = (c >> 1) & 1;
        let dz = (c >> 2) & 1;
        let idx = i0 + vec3<i32>(dx, dy, dz);
        if (idx.x < 0 || idx.y < 0 || idx.z < 0 || idx.x >= dims.x || idx.y >= dims.y || idx.z >= dims.z) { continue; }
        let w = select(1.0 - f.x, f.x, dx == 1) * select(1.0 - f.y, f.y, dy == 1) * select(1.0 - f.z, f.z, dz == 1);
        total += w * vs_load(i, idx);
    }
    return total;
}
// Transmittance of the scene's smoke from world point `world` toward `light`: exp(-tau), tau summed over volumes.
fn volume_transmission(world: vec3<f32>, light: Light) -> f32 {
    var ray = -light.direction.xyz;
    var limit = 3.0e38;
    if (light.position.w > 0.0) {
        let delta = light.position.xyz - world;
        limit = length(delta);
        ray = delta / max(limit, 1e-12);
    }
    let steps = i32(vs_set.march.y);
    var tau = 0.0;
    for (var i = 0u; i < u32(vs_set.count.x); i += 1u) {
        let vol = vs_set.vols[i];
        let o = vec3<f32>(dot(vol.row0.xyz, world) + vol.row0.w, dot(vol.row1.xyz, world) + vol.row1.w,
                          dot(vol.row2.xyz, world) + vol.row2.w);
        let d = vec3<f32>(dot(vol.row0.xyz, ray), dot(vol.row1.xyz, ray), dot(vol.row2.xyz, ray));
        var near = -3.0e38;
        var far = 3.0e38;
        for (var axis = 0u; axis < 3u; axis += 1u) {
            let lo = vol.box_min[axis];
            let hi = vol.box_max[axis];
            if (abs(d[axis]) < 1e-12) {
                if (o[axis] < lo || o[axis] > hi) { far = -3.0e38; }
            } else {
                let a = (lo - o[axis]) / d[axis];
                let b = (hi - o[axis]) / d[axis];
                near = max(near, min(a, b));
                far = min(far, max(a, b));
            }
        }
        near = max(near, 0.0);
        let length_t = max(min(far, limit) - near, 0.0);
        if (length_t > 0.0) {
            var total = 0.0;
            for (var j = 0; j < steps; j += 1) {
                total += vs_density(i, o + d * (near + (f32(j) + 0.5) / f32(steps) * length_t));
            }
            tau += vs_set.march.x * total * (length_t / f32(steps));
        }
    }
    return exp(-tau);
}
'''


def mesh_shadow_group(state, pipeline, scene, settings, used, keep):
    """The bind group 1 of the raster mesh shader's volume shadows (MESH_SHADOW_WGSL) for `scene`'s volumes."""
    wgpu, device = state['wgpu'], state['device']
    volumes = _volumes(scene)
    table = np.zeros((2 + 5 * MESH_SHADOW_VOLUMES, 4), 'f4')
    table[0, 0] = len(volumes)
    table[1] = (settings.shadow_density * (settings.scattering + settings.absorption) * settings.density_scale,
                settings.shadow_steps, 0, 0)
    views = []
    for slot in range(MESH_SHADOW_VOLUMES):
        if slot < len(volumes):
            volume = volumes[slot]
            inverse = np.linalg.inv(np.asarray(volume.matrix, np.float64))
            box_min = np.array(volume.origin, np.float64)
            box_max = box_min + np.array(volume.shape, np.float64) * volume.voxel_size
            row = 2 + 5 * slot
            table[row:row + 3] = inverse[:3, :]
            table[row + 3, :3], table[row + 3, 3] = box_min, volume.voxel_size
            table[row + 4, :3] = box_max
            views.append(texture(state, volume, used))
        else:
            views.append(_field(state, ('dummy', 'r32float'), lambda: (_ONE['r32float'], 'r32float'), used))
    uniform = keep(device.create_buffer_with_data(data=table, usage=wgpu.BufferUsage.UNIFORM))
    return device.create_bind_group(layout=pipeline.get_bind_group_layout(1), entries=[
        {'binding': 0, 'resource': {'buffer': uniform}}] + [{'binding': 1 + i, 'resource': v} for i, v in enumerate(views)])
