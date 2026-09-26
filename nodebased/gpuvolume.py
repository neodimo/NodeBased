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
};
struct Light { position: vec4<f32>, direction: vec4<f32>, colour: vec4<f32>, cone: vec4<f32>, shadow: vec4<f32> };
struct Vol {
    row0: vec4<f32>,       // inverse world-to-object rows: xyz, translation
    row1: vec4<f32>,
    row2: vec4<f32>,
    box_min: vec4<f32>,    // xyz, voxel size
    box_max: vec4<f32>,
    dims: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var depth_tex: %(DEPTH_TYPE)s;
@group(1) @binding(0) var<uniform> vol: Vol;
@group(1) @binding(1) var density: texture_3d<f32>;

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
// Zero-padded trilinear sample of the cell-centred grid at object-space point p (volumerender._trilinear).
fn sample_density(p: vec3<f32>) -> f32 {
    let g = (p - vol.box_min.xyz) / vol.box_min.w - vec3<f32>(0.5);
    let base = floor(g);
    let f = g - base;
    let i0 = vec3<i32>(base);
    let dims = vec3<i32>(vol.dims.xyz);
    var total = 0.0;
    for (var c = 0; c < 8; c += 1) {
        let dx = c & 1;
        let dy = (c >> 1) & 1;
        let dz = (c >> 2) & 1;
        let idx = i0 + vec3<i32>(dx, dy, dz);
        if (idx.x < 0 || idx.y < 0 || idx.z < 0 || idx.x >= dims.x || idx.y >= dims.y || idx.z >= dims.z) { continue; }
        let w = select(1.0 - f.x, f.x, dx == 1) * select(1.0 - f.y, f.y, dy == 1) * select(1.0 - f.z, f.z, dz == 1);
        total += w * textureLoad(density, idx, 0).r;
    }
    return total;
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
        total += sample_density(o + d * ((f32(j) + 0.5) / f32(steps) * length_t));
    }
    return exp(-params.march.z * sigma_t_unit * params.march.y * total * (length_t / f32(steps)));
}
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
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
                    incident += light.colour.rgb * (attn * shadow(p_world, light, sigma_t_unit));
                }
            }
            source = params.colour.rgb * incident;
        }
        rgb += trans * (frac * one_minus) * source;
        trans *= t_seg;
    }
    return vec4<f32>(rgb, 1.0 - trans);
}
'''

_DEPTH_SINGLE = ('texture_depth_2d',
                 'let raw = textureLoad(depth_tex, pixel, 0);\n'
                 '    if (raw < 1.0) { t_mesh = far * near / (far - raw * (far - near)) * ray_length; }')
# The viewport's depth buffer is multisampled: the nearest of the four samples hides the smoke, so a
# silhouette edge is never smoked over.
_DEPTH_MULTI = ('texture_depth_multisampled_2d',
                'var raw = 1.0;\n'
                '    for (var s = 0; s < 4; s += 1) { raw = min(raw, textureLoad(depth_tex, pixel, s)); }\n'
                '    if (raw < 1.0) { t_mesh = far * near / (far - raw * (far - near)) * ray_length; }')


def _volumes(scene):
    return tuple(getattr(scene, 'volumes', ()) or ())


def adapter_kind(state):
    normalized = str(state['info'].get('adapter_type', 'unknown')).lower().replace('_', '').replace(' ', '')
    return {'discretegpu': 'discrete', 'integratedgpu': 'integrated', 'cpu': 'cpu'}.get(normalized, 'other')


def check(state, scene, settings):
    """Raise `gpu3d.Unsupported` when a grid exceeds the adapter's 3D texture or memory limits."""
    from .gpu3d import Unsupported
    settings.validated()
    limits = state['device'].limits
    top = int(limits.get('max-texture-dimension-3d', 256))
    total = 0
    for volume in _volumes(scene):
        if max(volume.shape) > top:
            raise Unsupported(f'volume grid {volume.shape} exceeds the adapter 3D texture limit of {top} voxels per side')
        total += volume.density.nbytes
    budget = VOLUME_MEMORY_BUDGETS[adapter_kind(state)]
    if total > budget:
        raise Unsupported(f'volume grids need {total:,} bytes of texture memory, over the {budget:,} byte '
                          f'budget of this {adapter_kind(state)} adapter')


def upload_count(state):
    return state.get('volume_uploads', 0)


def cache_bytes(state):
    return sum(entry[2] for entry in state.get('volume_textures', {}).values())


_digests = weakref.WeakKeyDictionary()


def _digest(volume):
    """Content key of the arrays the shader reads; memoised on the (immutable) Volume object."""
    key = _digests.get(volume)
    if key is None:
        h = hashlib.blake2b(digest_size=16)
        h.update(str(volume.density.shape).encode())
        h.update(memoryview(volume.density).cast('B'))
        key = _digests[volume] = h.hexdigest()
    return key


def texture(state, volume, used):
    """The cached `r32float` 3D density texture view of `volume`; uploads only on a content miss."""
    wgpu, device = state['wgpu'], state['device']
    cache = state.setdefault('volume_textures', OrderedDict())
    key = _digest(volume)
    used.add(key)
    entry = cache.get(key)
    if entry is None:
        nx, ny, nz = volume.shape
        gpu_texture = device.create_texture(
            size=(nx, ny, nz), dimension='3d', format='r32float',
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        # [ix, iy, iz] C order has iz fastest; the texture wants x fastest.
        data = np.ascontiguousarray(volume.density.transpose(2, 1, 0), np.float32)
        device.queue.write_texture({'texture': gpu_texture}, data,
                                   {'bytes_per_row': nx * 4, 'rows_per_image': ny}, (nx, ny, nz))
        entry = cache[key] = (gpu_texture, gpu_texture.create_view(dimension='3d'), volume.density.nbytes)
        state['volume_uploads'] = upload_count(state) + 1
    cache.move_to_end(key)
    budget = VOLUME_MEMORY_BUDGETS[adapter_kind(state)]
    for old in [k for k in cache if k not in used]:
        if sum(e[2] for e in cache.values()) <= budget:
            break
        cache.pop(old)[0].destroy()
    return entry[1]


def pipeline(state, target, samples=1, multisampled_depth=False):
    key = ('volume', target, samples, multisampled_depth)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    depth_type, depth_load = _DEPTH_MULTI if multisampled_depth else _DEPTH_SINGLE
    module = device.create_shader_module(code=_SHADER % {'DEPTH_TYPE': depth_type, 'DEPTH_LOAD': depth_load})
    blend = {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}
    state['pipelines'][key] = device.create_render_pipeline(
        layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
        primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
        multisample={'count': samples},
        fragment={'module': module, 'entry_point': 'fs',
                  'targets': [{'format': target, 'blend': {'color': blend, 'alpha': blend}}]})
    return state['pipelines'][key]


def work_estimate(scene, camera, width, height, settings, light_count):
    """Upper bound of density lookups for the frame: box-clipped ray spans on a coarse ray grid, scaled up."""
    from . import volumerender
    volumes = _volumes(scene)
    if not volumes:
        return 0.0
    scale = max(1, int(math.sqrt(width * height / _ESTIMATE_RAYS)))
    w, h = max(1, width // scale), max(1, height // scale)
    eye, dirs, _length = volumerender._pixel_rays(camera, w, h)
    per_sample = 1 + light_count * int(settings.shadow_steps)
    total = 0.0
    for volume in volumes:
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
            depth_view, keep, *, target, samples=1, multisampled_depth=False, used=None):
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
    params = np.zeros((9, 4), 'f4')
    params[0] = width, height, width / max(height, 1), focal
    params[1, :3] = eye
    params[2, :3], params[3, :3], params[4, :3] = view[0], view[1], -view[2]
    params[5] = camera.near, camera.far, ambient, light_count
    params[6] = settings.step_size, settings.density_scale, settings.shadow_density, settings.shadow_steps
    params[7] = settings.scattering, settings.absorption, 1.0 if lit else 0.0, 0.0
    params[8, :3] = settings.color
    pipe = pipeline(state, target, samples, multisampled_depth)
    params_buffer = keep(device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM))
    group0 = device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {'binding': 0, 'resource': {'buffer': params_buffer}},
        {'binding': 1, 'resource': {'buffer': light_buffer}},
        {'binding': 2, 'resource': depth_view}])
    eye64 = eye.astype(np.float64)
    order = sorted(volumes, key=lambda v: -float(np.linalg.norm(
        (np.asarray(v.matrix, np.float64) @ np.append((np.array(v.origin) + np.array(v.shape) * v.voxel_size / 2), 1.0))[:3]
        - eye64)))
    groups1 = []
    for volume in order:
        inverse = np.linalg.inv(np.asarray(volume.matrix, np.float64))
        block = np.zeros((6, 4), 'f4')
        block[0:3] = inverse[:3, :]
        box_min = np.array(volume.origin, np.float64)
        box_max = box_min + np.array(volume.shape, np.float64) * volume.voxel_size
        block[3, :3], block[3, 3] = box_min, volume.voxel_size
        block[4, :3] = box_max
        block[5, :3] = volume.shape
        uniform = keep(device.create_buffer_with_data(data=block, usage=wgpu.BufferUsage.UNIFORM))
        groups1.append(device.create_bind_group(layout=pipe.get_bind_group_layout(1), entries=[
            {'binding': 0, 'resource': {'buffer': uniform}},
            {'binding': 1, 'resource': texture(state, volume, used)}]))
    return Prepared(pipe, group0, groups1)
