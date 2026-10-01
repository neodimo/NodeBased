"""Optional wgpu rasterizer, adapted from tools/spike_3d_backends.py.

RGBA uses float32 when float32-blendable is available, otherwise half precision.
Data outputs always use float32. Textures are half precision with CPU-generated
mips and a trilinear sampler; explicit per-triangle LOD matches the reference's
rounded area heuristic rather than hardware derivative-based LOD.
"""
from __future__ import annotations

import math
import threading
from dataclasses import replace

import numpy as np

from . import scene3d, raytrace


class Unsupported(Exception):
    """The caller should render this scene with the CPU backend."""


_lock = threading.RLock()
_states = {}
_errors = {}

# Calibration (2026-09-19). The first budgets were derived from whole-render times, which include large
# host-side costs (per-triangle Python preparation: ~270 ms at 10k triangles, ~1.1 s at 40k), so they
# understated GPU shadow throughput. Re-measured by subtracting a no-shadow render of the same scene
# (tools: see TASKLOG), 960x540, brute-force loop, one directional light:
#   RTX 3080 Ti   ~40-300e9 pair tests/s (noisy, 40k tris: 209e9)   Radeon 8060S  ~19-60e9/s (40k tris: 19e9)
#   llvmpipe      ~0.4-0.5e9/s
# BVH traversal in the fragment shader (units: rays x 16 x log2(triangles+2), as on the CPU):
#   RTX 3080 Ti   ~0.4-1.4e9 units/s   Radeon 8060S ~0.45-1.6e9/s   llvmpipe ~0.01-0.2e9/s
# The BVH is NOT faster than the brute loop on the discrete GPU up to 40k triangles (40k: 286 ms vs 99 ms
# of shadow cost); it wins on the integrated GPU from ~10k triangles and on software adapters from ~1k.
# Budgets bound one submission to about a second on each adapter type (display-driver timeouts; a
# submitted job cannot be cancelled); 'other' is a guess. Windows and other adapters are unmeasured.
SHADOW_WORK_BUDGETS = {'discrete': 4e10, 'integrated': 1e10, 'cpu': 3e8, 'other': 2e9}
SHADOW_BVH_WORK_BUDGETS = {'discrete': 4e8, 'integrated': 4e8, 'cpu': 1e7, 'other': 2e8}
# Triangle count above which the BVH path is chosen (when the adapter is capable). None = per-adapter table.
SHADOW_BVH_THRESHOLDS = {'discrete': 20000, 'integrated': 5000, 'cpu': 500, 'other': 5000}
SHADOW_BVH_THRESHOLD = None
SHADOW_BVH_STACK_SIZE = 64
GPU_MAX_BANDS = 64
GPU_FORCE_BANDS = None
last_shadow_path = 'brute'


def _bvh_capable(limits, node_bytes, order_bytes):
    # Fragment storage: lights, triangles, nodes, primitive order.
    return (limits.get('max-storage-buffers-per-shader-stage', 0) >= 4
            and max(node_bytes, order_bytes) <= limits.get('max-storage-buffer-binding-size', 0))


def _pack_bvh(bvh, cancel=None):
    pending = [(0, 1)] if len(bvh.left) else []
    while pending:
        _cancel(cancel)
        node, depth = pending.pop()
        if depth > SHADOW_BVH_STACK_SIZE:
            raise ValueError(f'Shadow BVH depth {depth} exceeds traversal stack {SHADOW_BVH_STACK_SIZE}')
        if bvh.prim_count[node] == 0:
            pending.extend(((int(bvh.left[node]), depth+1), (int(bvh.right[node]), depth+1)))
    dtype = np.dtype([('lo', '<f4', 3), ('left', '<i4'), ('hi', '<f4', 3),
                      ('right', '<i4'), ('offset', '<u4'), ('count', '<u4'), ('pad', '<u4', 2)])
    assert dtype.itemsize == 48
    nodes = np.zeros(len(bvh.left), dtype=dtype)
    nodes['lo'] = np.nextafter(bvh.node_lo.astype('f4'), np.float32(-np.inf))
    nodes['hi'] = np.nextafter(bvh.node_hi.astype('f4'), np.float32(np.inf))
    for name, value in [('left', bvh.left), ('right', bvh.right),
                        ('offset', bvh.prim_offset), ('count', bvh.prim_count)]:
        nodes[name] = value
    return nodes, bvh.prim_order.astype('u4')


def _adapter_kind(state):
    normalized = str(state['info'].get('adapter_type', 'unknown')).lower().replace('_', '').replace(' ', '')
    return {'discretegpu': 'discrete', 'integratedgpu': 'integrated', 'cpu': 'cpu'}.get(normalized, 'other')


def _shadow_budget(state, work, path='brute'):
    reported = str(state['info'].get('adapter_type', 'unknown'))
    kind = _adapter_kind(state)
    budget = (SHADOW_BVH_WORK_BUDGETS if path == 'bvh' else SHADOW_WORK_BUDGETS)[kind]
    if work > budget:
        raise ValueError(f'Shadow rays exceed the GPU budget: adapter {reported} ({kind}), '
                         f'{work:,.0f} > {budget:,.0f} {path} work units; '
                         'reduce resolution/samples/triangles or switch shadows off')
    return budget


def _band_plan(state, work, path, height):
    """Return contiguous (start, stop) row bands within the submission budget."""
    if height < 1:
        raise ValueError('Render dimensions must be positive')
    budget = _shadow_budget(state, 0, path)
    required = max(1, math.ceil(work / budget)) if budget > 0 else (1 if work == 0 else math.inf)
    cap = min(height, GPU_MAX_BANDS)
    if required > cap:
        reported = state['info'].get('adapter_type', 'unknown')
        raise ValueError(f'Shadow rays exceed the GPU budget: adapter {reported} ({_adapter_kind(state)}), '
                         f'{work:,.0f} total {path} work units, {budget:,.0f} per-submission budget; '
                         f'even split into {GPU_MAX_BANDS} bands (band cap, at most {height} rows); '
                         'reduce resolution/samples/triangles or switch shadows off')
    bands = required
    if GPU_FORCE_BANDS is not None:
        if not isinstance(GPU_FORCE_BANDS, int) or not 1 <= GPU_FORCE_BANDS <= GPU_MAX_BANDS:
            raise ValueError(f'GPU_FORCE_BANDS must be an integer from 1 to {GPU_MAX_BANDS}')
        bands = min(height, GPU_FORCE_BANDS)
    _shadow_budget(state, work / bands, path)
    return [(i * height // bands, (i + 1) * height // bands) for i in range(bands)]


def _state(choice=None):
    choice = choice or 'default'
    if choice not in ('default', 'discrete', 'integrated', 'cpu'):
        raise ValueError(f'Unknown wgpu adapter {choice!r}')
    with _lock:
        if choice in _states:
            return _states[choice]
        if choice in _errors:
            raise RuntimeError(_errors[choice])
        try:
            import wgpu
            if choice == 'default':
                adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
            else:
                target = {'discrete': 'discretegpu', 'integrated': 'integratedgpu', 'cpu': 'cpu'}[choice]
                adapter = next((a for a in wgpu.gpu.enumerate_adapters_sync()
                                if str(a.info.get('adapter_type', '')).lower().replace('_', '').replace(' ', '') == target), None)
            if adapter is None:
                raise RuntimeError(f'no {choice} adapter')
            storage_buffers = adapter.limits.get('max-storage-buffers-per-shader-stage', 2)
            if storage_buffers < 2:
                raise RuntimeError('adapter allows fewer than 2 storage buffers per shader stage')
            features = ['float32-blendable'] if 'float32-blendable' in adapter.features else []
            device = adapter.request_device_sync(required_features=features, required_limits={
                'max-storage-buffer-binding-size': adapter.limits['max-storage-buffer-binding-size'],
                'max-storage-buffers-per-shader-stage': min(8, storage_buffers)})
            # Data outputs render to float32; downlevel adapters (GL/GLES class) reject that attachment.
            try:
                device.create_texture(size=(1, 1, 1), format='rgba32float',
                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC).destroy()
            except Exception as exc:
                raise RuntimeError(f'adapter cannot render to rgba32float (downlevel): {exc}') from exc
            state = dict(wgpu=wgpu, device=device, info=dict(adapter.info), pipelines={},
                         format='rgba32float' if features else 'rgba16float',
                         features=sorted(adapter.features))
            _states[choice] = state
            return state
        except Exception as exc:
            reason = f'wgpu unavailable ({choice}): {type(exc).__name__}: {exc}'
            _errors[choice] = reason
            raise RuntimeError(reason) from exc


def available():
    """Cached adapter/device probe; safe when the optional dependency is absent."""
    try:
        _state()
        return True
    except Exception:
        return False


def describe():
    try:
        state = _state()
        info = state['info']
        return f"{info.get('device', info.get('description', 'wgpu adapter'))} / {info.get('backend_type', 'unknown')} ({state['format']})"
    except Exception as exc:
        return str(exc)


def adapter_report(choice=None):
    """Multi-line adapter, backend and precision summary for test and CI logs."""
    try:
        state = _state(choice)
    except Exception as exc:
        return f'wgpu adapter: none ({exc})'
    info, limits = state['info'], state['device'].limits
    try:
        import wgpu
        version = wgpu.__version__
    except Exception:
        version = 'unknown'
    blendable = 'float32-blendable' in state.get('features', ())
    return '\n'.join((
        f"wgpu adapter: {info.get('device') or info.get('description') or 'unnamed'}",
        f"  type {info.get('adapter_type', 'unknown')}, backend {info.get('backend_type', 'unknown')}, "
        f"vendor {info.get('vendor') or info.get('vendor_id', 'unknown')}, driver {info.get('description') or 'unknown'}, wgpu {version}",
        f"  rgba target {state['format']} (float32-blendable {'present' if blendable else 'MISSING: beauty and splat layers blend in half precision'})",
        f"  storage buffers per stage {limits.get('max-storage-buffers-per-shader-stage')}, "
        f"binding size {limits.get('max-storage-buffer-binding-size')}, max texture {limits.get('max-texture-dimension-2d')}"))


_SHADER = '''
struct Params { projection: vec4<f32>, eye: vec4<f32>, settings: vec4<f32>, shadow: vec4<f32> };
struct Light { position: vec4<f32>, direction: vec4<f32>, colour: vec4<f32>, cone: vec4<f32>, shadow: vec4<f32> };  // colour.w = falloff power; shadow = (bias scale, tan blur, samples, 0)
var<private> near_bias: f32 = 0.0;
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var filtering: sampler;
struct Triangle { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32> };
@group(0) @binding(4) var<storage, read> triangles: array<Triangle>;
// Y3 of 3, part 1: the four remaining PBR texture maps, matching `scene3d._shade_fragments`. Each is
// a single (top) level, sampled with the same `filtering` sampler as the mip-chained base colour
// texture; a geometry missing a map still binds a harmless 1x1 dummy so every pipeline variant
// compiles and draws alike, but its `maps` flag (packed in `_prepare`) keeps the fragment shader from
// ever sampling it.
@group(0) @binding(7) var mr_tex: texture_2d<f32>;
@group(0) @binding(8) var normal_tex: texture_2d<f32>;
@group(0) @binding(9) var occlusion_tex: texture_2d<f32>;
@group(0) @binding(10) var emissive_tex: texture_2d<f32>;
// Z1 of 2 finish: Rect/Disc/Sphere area lights, matching `scene3d._area_light_contribution`'s fixed
// low-discrepancy sample set exactly -- `_area_light_resources` computes the same sample points and
// normals on the host (`scene3d._area_light_samples`) and uploads them once, reused by every
// fragment, the way the CPU reference reuses them across every shading point. `center.w` is the
// two-sided flag; `radiance_area.w` is the light's surface area; `info` is (sample offset, sample
// count, shadow bias scale matching `light.shadow.x` above, shadows-on flag). A scene with no area
// lights still binds a harmless one-row dummy so every pipeline variant compiles alike.
struct AreaLight { center: vec4<f32>, radiance_area: vec4<f32>, info: vec4<f32> };
struct AreaSample { point: vec4<f32>, normal: vec4<f32> };
@group(0) @binding(11) var<storage, read> area_lights: array<AreaLight>;
@group(0) @binding(12) var<storage, read> area_samples: array<AreaSample>;
// A single environment light on meshes (docs/3D_FOUNDATION.md "Left out" of Y1 of 2 finish (1)):
// image-based diffuse (9 SH coefficients, matching envlight._sh_irradiance) and specular (a six-tile
// vertical atlas, one GGX-roughness level per tile from envlight.LEVEL_ROUGHNESS, sampled with a
// manual two-tile lerp, matching envlight.Prefiltered.lookup). `rot0`/`rot1`/`rot2` are the columns of
// envlight.Environment._local's combined rotation (the environment's own turn and its parent's,
// unlike viewportgpu.py's dome which assumes a root-level environment) so a parented or rotated
// Environment still matches the CPU reference. `enabled.x` is 0 with no environment bound (a scene
// with more than one Environment, or the ray-traced render mode, still refuses on the CPU: group(2) is
// always bound so the shader compiles either way, but its atlas is a harmless 1-texel-wide default then).
struct EnvGlobals { sh: array<vec4<f32>, 9>, rot0: vec4<f32>, rot1: vec4<f32>, rot2: vec4<f32>, enabled: vec4<f32> };
@group(2) @binding(0) var<uniform> envg: EnvGlobals;
@group(2) @binding(1) var env_tex: texture_2d<f32>;
@group(2) @binding(2) var env_sampler: sampler;
fn env_local(d: vec3<f32>) -> vec3<f32> {
    return mat3x3<f32>(envg.rot0.xyz, envg.rot1.xyz, envg.rot2.xyz) * d;
}
fn env_uv(d: vec3<f32>) -> vec2<f32> {
    let u = 0.5 + atan2(d.x, -d.z) / (2.0 * 3.14159265);
    let v = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}
// envlight._sh_irradiance: cosine-convolved radiance, folded with the environment's gain already.
fn env_diffuse(n: vec3<f32>) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let d = env_local(n);
    let x = d.x; let y = d.y; let z = d.z;
    var total = envg.sh[0].rgb * 0.282095;
    total += envg.sh[1].rgb * (0.488603 * y) * (2.0 / 3.0);
    total += envg.sh[2].rgb * (0.488603 * z) * (2.0 / 3.0);
    total += envg.sh[3].rgb * (0.488603 * x) * (2.0 / 3.0);
    total += envg.sh[4].rgb * (1.092548 * x * y) * 0.25;
    total += envg.sh[5].rgb * (1.092548 * y * z) * 0.25;
    total += envg.sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) * 0.25;
    total += envg.sh[7].rgb * (1.092548 * x * z) * 0.25;
    total += envg.sh[8].rgb * (0.546274 * (x * x - y * y)) * 0.25;
    return total;
}
// envlight.Prefiltered.lookup: a two-tile lerp over the atlas's six GGX-roughness levels. Each
// `textureSampleLevel` call is inset by half a texel from its tile's own top/bottom row so hardware
// bilinear filtering cannot blend in the next tile's edge (a different roughness level) the way
// `envlight.sample_map`'s per-level clamp never would -- otherwise a reflection direction near the
// map's pole (v near 0 or 1) picks up a sliver of the wrong level.
fn env_specular(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let uv = env_uv(env_local(direction));
    let pos = clamp(roughness, 0.0, 1.0) * 5.0;
    let lo = floor(pos);
    let hi = min(lo + 1.0, 5.0);
    let half_texel = 0.5 / f32(textureDimensions(env_tex).y);
    let row_lo = clamp((lo + uv.y) / 6.0, lo / 6.0 + half_texel, (lo + 1.0) / 6.0 - half_texel);
    let row_hi = clamp((hi + uv.y) / 6.0, hi / 6.0 + half_texel, (hi + 1.0) / 6.0 - half_texel);
    let a = textureSampleLevel(env_tex, env_sampler, vec2<f32>(uv.x, row_lo), 0.0).rgb;
    let b = textureSampleLevel(env_tex, env_sampler, vec2<f32>(uv.x, row_hi), 0.0).rgb;
    return mix(a, b, pos - lo);
}
fn triangle_transmission(index: u32, origin: vec3<f32>, ray: vec3<f32>, limit: f32, point: f32) -> f32 {
    let tri = triangles[index];
    let h = cross(ray, tri.e2.xyz);
    let det = dot(h, tri.e1.xyz);
    if (abs(det) > 1e-10) {
        let inverse = 1.0 / det;
        let delta = origin - tri.v0.xyz;
        let u = dot(delta, h) * inverse;
        let q = cross(delta, tri.e1.xyz);
        let v = dot(ray, q) * inverse;
        let t = dot(tri.e2.xyz, q) * inverse;
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 &&
            t > near_bias && (point == 0.0 || t < limit)) {
            return 1.0 - tri.v0.w;
        }
    }
    return 1.0;
}
// Uniform-controlled loops keep compilation independent of scene complexity.
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
fn seed_hash(p: vec3<f32>) -> u32 {
    // scene3d._point_seeds: the same hash of the float32 bits, so CPU and GPU share the sample pattern.
    var h = (bitcast<u32>(p.x) * 73856093u) ^ (bitcast<u32>(p.y) * 19349663u) ^ (bitcast<u32>(p.z) * 83492791u);
    h ^= h >> 16u; h *= 0x85ebca6bu; h ^= h >> 13u; h *= 0xc2b2ae35u; h ^= h >> 16u;
    return h;
}
fn trace_visibility(origin: vec3<f32>, ray: vec3<f32>, limit: f32, light: Light) -> f32 {
    var transmission = 1.0;
    // BVH_TRAVERSAL
    for (var j = 0u; j < u32(params.shadow.y); j += 1u) {
        transmission *= triangle_transmission(j, origin, ray, limit, light.position.w);
    }
    return transmission;
}
fn visibility(position: vec3<f32>, normal: vec3<f32>, light: Light) -> f32 {
    let bias = params.shadow.x * light.shadow.x;
    near_bias = bias * 0.01;
    let origin = position + normal * bias;
    var ray = -light.direction.xyz;
    var limit = 0.0;
    if (light.position.w > 0.0) {
        ray = light.position.xyz - origin;
        limit = length(ray);
        ray = ray / max(limit, 1e-8);
    }
    if (light.shadow.y <= 0.0) { return trace_visibility(origin, ray, limit, light); }
    // Soft shadow: scene3d._shadow_trace, a Vogel spiral of jittered rays turned by a per-point hash angle.
    let count = max(u32(light.shadow.z), 1u);
    let phi = f32(seed_hash(origin)) * (6.2831853 / 4294967296.0);
    let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(ray.y) < 0.9);
    let a = normalize(cross(ray, axis));
    let b = cross(ray, a);
    var total = 0.0;
    for (var k = 0u; k < count; k += 1u) {
        let r = sqrt((f32(k) + 0.5) / f32(count));
        let theta = f32(k) * 2.3999632 + phi;
        let spread = a * (r * cos(theta)) + b * (r * sin(theta));
        var jray = normalize(ray + spread * light.shadow.y);
        var jlimit = 0.0;
        if (light.position.w > 0.0) {
            let delta = light.position.xyz + spread * (limit * light.shadow.y) - origin;
            jlimit = length(delta);
            jray = delta / max(jlimit, 1e-8);
        }
        total += trace_visibility(origin, jray, jlimit, light);
    }
    return total / f32(count);
}
// Z1 of 2 finish: scene3d._area_light_contribution/_area_light_shading for a Rect/Disc/Sphere light --
// a Monte Carlo diffuse irradiance over the light's fixed sample set (`area_samples`, uploaded once by
// `_area_light_resources` and reused by every fragment, like the CPU reference reuses them across every
// shading point), and a centre-point, inverse-square specular colour (no test exercises area-light
// specular directly on the CPU reference either; see `_area_light_shading`'s own docstring).
struct AreaResult { irradiance: vec3<f32>, to_light: vec3<f32>, spec_colour: vec3<f32> };
fn area_light_shade(light_index: u32, position: vec3<f32>, normal: vec3<f32>) -> AreaResult {
    let al = area_lights[light_index];
    let two_sided = al.center.w > 0.5;
    let offset = u32(al.info.x);
    let count = u32(al.info.y);
    let bias = params.shadow.x * al.info.z;
    near_bias = bias * 0.01;
    let origin = position + normal * bias;
    let shadows_on = al.info.w > 0.5 && params.shadow.y > 0.0;
    let point_light = Light(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var total = vec3<f32>(0.0);
    var vis_sum = 0.0;
    for (var s = 0u; s < count; s += 1u) {
        let smp = area_samples[offset + s];
        let to_recv = position - smp.point.xyz;
        let dist2 = max(dot(to_recv, to_recv), 1e-10);
        let dist = sqrt(dist2);
        let wi = to_recv / dist;
        var cos_light = dot(wi, smp.normal.xyz);
        cos_light = select(max(cos_light, 0.0), abs(cos_light), two_sided);
        let cos_surface = max(dot(-wi, normal), 0.0);
        let weight = cos_light * cos_surface / dist2;
        var vis = 1.0;
        if (shadows_on && weight > 0.0) {
            let ray = smp.point.xyz - origin;
            let limit = length(ray);
            vis = trace_visibility(origin, ray / max(limit, 1e-8), limit, point_light);
        }
        total += (weight * vis) * al.radiance_area.xyz;
        vis_sum += vis;
    }
    let fcount = f32(max(count, 1u));
    var result: AreaResult;
    result.irradiance = total * (al.radiance_area.w / fcount);
    let to_centre = al.center.xyz - position;
    let dist2c = max(dot(to_centre, to_centre), 1e-6);
    result.to_light = to_centre / sqrt(dist2c);
    result.spec_colour = (vis_sum / fcount / dist2c) * (al.radiance_area.xyz * al.radiance_area.w);
    return result;
}
// scene3d._shade_pbr_mesh / splatshade._cook_torrance: Cook-Torrance GGX specular response times n.l,
// with a per-channel Fresnel `f0` (Y3 of 3, part 1: `pbr` mesh materials on the GPU raster path, factors
// only -- the texture maps and an Environment together with `pbr` are still CPU-only, see `render`'s
// `pbr_geometries` checks in gpu3d.py).
// scene3d._orthonormal_tangent: Duff et al. branchless ONB, the fallback frame when a triangle's
// UV-gradient tangent degenerates (parallel to the normal).
fn orthonormal_tangent(n: vec3<f32>) -> vec3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
}
// scene3d._shade_fragments' tangent-space normal map decode: Gram-Schmidt the per-triangle tangent
// against the (possibly bump-mapped-so-far) normal, build the bitangent, and reproject the texel.
fn apply_normal_map(n: vec3<f32>, tangent: vec3<f32>, uv: vec2<f32>, scale: f32) -> vec3<f32> {
    let texel = textureSampleLevel(normal_tex, filtering, uv, 0.0).rgb;
    let local_n = (texel * 2.0 - 1.0) * vec3<f32>(scale, scale, 1.0);
    let t_ortho = tangent - n * dot(tangent, n);
    let t_norm = length(t_ortho);
    var t = orthonormal_tangent(n);
    if (t_norm >= 1e-12) { t = t_ortho / t_norm; }
    let b = cross(n, t);
    let result = t * local_n.x + b * local_n.y + n * local_n.z;
    return result / max(length(result), 1e-8);
}
fn ggx_response(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
    let h = normalize(l + v);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let alpha = max(roughness, 0.05) * max(roughness, 0.05);
    let alpha2 = alpha * alpha;
    let denom = nh * nh * (alpha2 - 1.0) + 1.0;
    let d = alpha2 / (3.14159265 * denom * denom);
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let vis = 1.0 / (max(nl * (1.0 - k) + k, 1e-4) * max(nv * (1.0 - k) + k, 1e-4) * 4.0);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vh, 5.0);
    return vec3<f32>(3.14159265 * d * vis * nl) * fresnel;
}
// envlight.dfg (Z1 of 2 finish): Karis's analytic fit of the split-sum BRDF table, `(A, B)` with
// `F0 * A + B` the specular albedo -- the same closed form as the CPU reference, so no lookup texture.
fn dfg(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    let nv = clamp(n_dot_v, 1e-4, 1.0);
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let t = roughness * c0 + c1;
    let a004 = min(t.x * t.x, pow(2.0, -9.28 * nv)) * t.x + t.y;
    return vec2<f32>(-1.04 * a004 + t.z, 1.04 * a004 + t.w);
}
// scene3d._mesh_pbr_environment: split-sum image-based light for a `pbr` mesh. `base_rgb` is the F0
// used for the metal term's own split-sum weight (the metallic-workflow convention: a full metal's
// specular colour is its base colour), independent of the `f0` already mixed for direct lighting.
fn pbr_env_diffuse(n: vec3<f32>, nv: f32, roughness: f32, metallic: f32, f0_dielectric: f32) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let fit = dfg(nv, roughness);
    let dielectric_w = (f0_dielectric * fit.x + fit.y) * (1.0 + f0_dielectric * (1.0 / max(fit.x + fit.y, 1e-4) - 1.0));
    let kd = (1.0 - metallic) * (1.0 - dielectric_w);
    return env_diffuse(n) * kd;
}
fn pbr_env_specular(n: vec3<f32>, to_eye: vec3<f32>, roughness: f32, metallic: f32,
                    f0_dielectric: f32, base_rgb: vec3<f32>) -> vec3<f32> {
    if (envg.enabled.x < 0.5) { return vec3<f32>(0.0); }
    let nv = max(dot(n, to_eye), 0.0);
    let fit = dfg(nv, roughness);
    let compensation = 1.0 / max(fit.x + fit.y, 1e-4);
    let dielectric_w = vec3<f32>((f0_dielectric * fit.x + fit.y) * (1.0 + f0_dielectric * (compensation - 1.0)));
    let metal_w = (base_rgb * fit.x + fit.y) * (vec3<f32>(1.0) + base_rgb * (compensation - 1.0));
    let total_w = (1.0 - metallic) * dielectric_w + metallic * metal_w;
    let reflected = 2.0 * nv * n - to_eye;
    return total_w * env_specular(reflected, roughness);
}
override PASS: u32 = 0u;
struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) depth: f32,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) colour: vec4<f32>,
    @location(5) @interpolate(flat) lod: f32,
    @location(6) @interpolate(flat) material: vec3<f32>,
    @location(7) @interpolate(flat) object_id: f32,
    // (metallic, roughness, dielectric F0, is-pbr flag); scene3d.Geometry's metallic/pbr_roughness/
    // pbr_specular, packed only when `material` is "pbr" (`_prepare` below).
    @location(8) @interpolate(flat) pbr: vec4<f32>,
    // Y3 of 3, part 1: a per-triangle UV-gradient tangent (world space, flat -- see `_prepare`), which
    // of the four texture maps are bound (metallic-roughness, normal, occlusion, emissive), the flat
    // `emissive_color` tint, and (normal_scale, occlusion_strength) -- every value is zero for a
    // non-`pbr` or untextured geometry, so every earlier render is untouched.
    @location(9) @interpolate(flat) tangent: vec3<f32>,
    @location(10) @interpolate(flat) maps: vec4<f32>,
    @location(11) @interpolate(flat) emissive_color: vec3<f32>,
    @location(12) @interpolate(flat) scales: vec4<f32>,
};
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) world: vec3<f32>,
             @location(2) normal: vec3<f32>, @location(3) uv: vec2<f32>,
             @location(4) colour: vec4<f32>, @location(5) lod: f32, @location(6) material: vec3<f32>,
             @location(7) object_id: f32, @location(8) pbr: vec4<f32>,
             @location(9) tangent: vec3<f32>, @location(10) maps: vec4<f32>,
             @location(11) emissive_color: vec3<f32>, @location(12) scales: vec4<f32>) -> Vertex {
    let q = params.projection;
    var v: Vertex;
    // WebGPU depth is 0..w: crossing triangles are clipped, not rejected.
    v.position = vec4<f32>(p.x*q.x, p.y*q.y,
        -q.w/(q.w-q.z)*p.z - q.w*q.z/(q.w-q.z), -p.z);
    v.depth = -p.z; v.world = world; v.normal = normal;
    v.uv = uv; v.colour = colour; v.lod = lod; v.material = material; v.object_id = object_id;
    v.pbr = pbr; v.tangent = tangent; v.maps = maps; v.emissive_color = emissive_color; v.scales = scales;
    return v;
}
@fragment fn fs(v: Vertex) -> @location(0) vec4<f32> {
    let flipped_uv = vec2<f32>(v.uv.x, 1.0-v.uv.y);
    var source = textureSampleLevel(tex, filtering, flipped_uv, v.lod) * v.colour;
    if (source.a <= 0.0) { discard; }
    if (PASS == 0u && source.a < 0.999) { discard; }
    if (PASS == 1u && source.a >= 0.999) { discard; }
    var normal = v.normal / max(length(v.normal), 1e-8);
    if (v.maps.y > 0.5) {
        // Y3 of 3, part 1: tangent-space normal map (`scene3d._shade_fragments`'s normal-map block
        // runs before the eye-facing flip below, same order here).
        normal = apply_normal_map(normal, v.tangent, flipped_uv, v.scales.x);
    }
    if (dot(normal, params.eye.xyz-v.world) < 0.0) { normal = -normal; }
    if (params.settings.z == 1.0) { return vec4<f32>(vec3<f32>(v.depth), 1.0); }
    if (params.settings.z == 2.0) { return vec4<f32>(normal, 1.0); }
    if (params.settings.z == 7.0) { return vec4<f32>(v.world, 1.0); }
    if (params.settings.z == 8.0) { return vec4<f32>(v.uv, 0.0, 1.0); }
    if (params.settings.z == 9.0) { return vec4<f32>(v.object_id, 0.0, 0.0, 1.0); }
    if (params.settings.z == 3.0) { return source; }
    var emission = source.rgb * v.material.z;
    if (v.pbr.w > 0.5) {
        // scene3d._shade_fragments' emissive block: a flat `emissive_color`, tinted by an emissive
        // texture when one is bound, added independently of the base colour.
        var emissive_tint = v.emissive_color;
        if (v.maps.w > 0.5) {
            emissive_tint = emissive_tint * textureSampleLevel(emissive_tex, filtering, flipped_uv, 0.0).rgb;
        }
        emission += emissive_tint;
    }
    if (params.settings.z == 6.0) { return vec4<f32>(emission, source.a); }
    if (params.settings.y > 0.0 || params.settings.w > 0.5 || envg.enabled.x > 0.5) {
        var specular = vec3<f32>(0.0);
        let eye_delta = params.eye.xyz-v.world;
        let to_eye = eye_delta / max(length(eye_delta), 1e-8);
        var radiance: vec3<f32>;
        if (v.pbr.w > 0.5) {
            // scene3d._shade_pbr_mesh (Y3 of 3): metallic/roughness/dielectric-F0 factors, optionally
            // overridden per-texel by a metallic-roughness map (G roughness, B metallic, glTF packing,
            // matching `scene3d._shade_fragments`), lit by Directional/Point/Spot/area lights and
            // (Z1 of 2 finish) a single Environment's split-sum diffuse/specular cross term
            // (`pbr_env_diffuse`/`pbr_env_specular`, matching `scene3d._mesh_pbr_environment`).
            var metallic = v.pbr.x;
            var roughness = v.pbr.y;
            if (v.maps.x > 0.5) {
                let mr = textureSampleLevel(mr_tex, filtering, flipped_uv, 0.0);
                roughness = mr.g; metallic = mr.b;
            }
            let base_rgb = source.rgb / max(source.a, 1e-6);
            let f0 = mix(vec3<f32>(v.pbr.z), base_rgb, metallic);
            radiance = vec3<f32>(params.settings.x) * (1.0 - metallic);
            for (var i = 0u; i < u32(params.settings.y); i += 1u) {
                var toward = -lights[i].direction.xyz;
                if (lights[i].position.w > 0.0) {
                    toward = lights[i].position.xyz-v.world;
                    toward = toward/max(length(toward), 1e-8);
                }
                var transmission = 1.0;
                if (params.shadow.y > 0.0 && lights[i].direction.w > 0.0) {
                    transmission = visibility(v.world, normal, lights[i]);
                    // VOLUME_SHADOW
                }
                let factor = attenuation(lights[i].position, lights[i].direction, lights[i].cone, lights[i].colour.w, v.world);
                let scale = transmission * factor;
                let nl = max(dot(normal, toward), 0.0);
                let half_delta = toward + to_eye;
                let half_vector = half_delta / max(length(half_delta), 1e-8);
                let vh = max(dot(to_eye, half_vector), 0.0);
                let kd = (1.0 - metallic) * (1.0 - (0.04 + 0.96 * pow(1.0 - vh, 5.0)));
                radiance += nl * kd * scale * lights[i].colour.xyz;
                specular += ggx_response(normal, to_eye, toward, roughness, f0) * scale * lights[i].colour.xyz;
            }
            for (var i = 0u; i < u32(params.settings.w); i += 1u) {
                // scene3d._area_light_shading, through `_shade_pbr_mesh`'s area branch: the same
                // Cook-Torrance response as a direct light, fed the Monte Carlo diffuse irradiance and
                // the centre-point specular colour instead of a light's own (position, colour).
                let area = area_light_shade(i, v.world, normal);
                let half_delta = area.to_light + to_eye;
                let half_vector = half_delta / max(length(half_delta), 1e-8);
                let vh = max(dot(to_eye, half_vector), 0.0);
                let kd = (1.0 - metallic) * (1.0 - (0.04 + 0.96 * pow(1.0 - vh, 5.0)));
                radiance += kd * area.irradiance;
                specular += ggx_response(normal, to_eye, area.to_light, roughness, f0) * area.spec_colour;
            }
            let env_nv = max(dot(normal, to_eye), 0.0);
            radiance += pbr_env_diffuse(normal, env_nv, roughness, metallic, v.pbr.z);
            specular += pbr_env_specular(normal, to_eye, roughness, metallic, v.pbr.z, base_rgb);
            if (v.maps.z > 0.5) {
                // scene3d._shade_fragments: occlusion attenuates the diffuse response only (ambient,
                // direct lights and the environment's diffuse term alike), never the specular term --
                // `_mesh_pbr_environment`'s diffuse is folded into `diffuse_radiance` before
                // `_shade_fragments` applies `occlusion_factor`, so it is here too.
                let occ = textureSampleLevel(occlusion_tex, filtering, flipped_uv, 0.0).r;
                let strength = clamp(v.scales.y, 0.0, 1.0);
                radiance *= 1.0 - strength * (1.0 - occ);
            }
        } else {
            radiance = vec3<f32>(params.settings.x) + env_diffuse(normal);
            for (var i = 0u; i < u32(params.settings.y); i += 1u) {
                var toward = -lights[i].direction.xyz;
                if (lights[i].position.w > 0.0) {
                    toward = lights[i].position.xyz-v.world;
                    toward = toward/max(length(toward), 1e-8);
                }
                var transmission = 1.0;
                if (params.shadow.y > 0.0 && lights[i].direction.w > 0.0) {
                    transmission = visibility(v.world, normal, lights[i]);
                    // VOLUME_SHADOW
                }
                let factor = attenuation(lights[i].position, lights[i].direction, lights[i].cone, lights[i].colour.w, v.world);
                radiance += max(dot(normal, toward), 0.0)*transmission*factor*lights[i].colour.xyz;
                if (v.material.x > 0.0 && dot(normal, toward) > 0.0) {
                    let half_delta = toward + to_eye;
                    let half_vector = half_delta / max(length(half_delta), 1e-8);
                    specular += v.material.x * pow(max(dot(normal, half_vector), 0.0), v.material.y)
                        * transmission * factor * lights[i].colour.xyz;
                }
            }
            for (var i = 0u; i < u32(params.settings.w); i += 1u) {
                // scene3d._shade_fragments' non-`pbr` area branch: no `front`/visibility gate beyond
                // what `area_light_shade`'s own Monte Carlo visibility already folded in.
                let area = area_light_shade(i, v.world, normal);
                radiance += area.irradiance;
                if (v.material.x > 0.0) {
                    let half_delta = area.to_light + to_eye;
                    let half_vector = half_delta / max(length(half_delta), 1e-8);
                    let lobe = pow(max(dot(normal, half_vector), 0.0), v.material.y);
                    specular += v.material.x * lobe * area.spec_colour;
                }
            }
            if (v.material.x > 0.0) {
                // scene3d._mesh_environment_specular: Blinn-Phong shininess mapped to a GGX roughness.
                let refl = 2.0 * dot(normal, to_eye) * normal - to_eye;
                specular += env_specular(refl, sqrt(2.0 / (v.material.y + 2.0))) * v.material.x;
            }
        }
        if (params.settings.z == 4.0) { return vec4<f32>(source.rgb*radiance, source.a); }
        if (params.settings.z == 5.0) { return vec4<f32>(specular*source.a, source.a); }
        source = vec4<f32>(source.rgb*radiance + specular*source.a, source.a);
    }
    if (params.settings.z == 4.0) { return source; }
    if (params.settings.z == 5.0) { return vec4<f32>(vec3<f32>(0.0), source.a); }
    return vec4<f32>(source.rgb + emission, source.a);
}
'''


_BVH_DECL = """
struct BvhNode { lo: vec3<f32>, left: i32, hi: vec3<f32>, right: i32,
                 offset: u32, count: u32, pad0: u32, pad1: u32 };
@group(0) @binding(5) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(6) var<storage, read> prim_order: array<u32>;
fn box_hit(node: BvhNode, origin: vec3<f32>, ray: vec3<f32>, limit: f32) -> bool {
    var near = near_bias;
    var far = limit;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if (ray[axis] == 0.0) {
            if (origin[axis] < node.lo[axis] || origin[axis] > node.hi[axis]) { return false; }
        } else {
            let a = (node.lo[axis] - origin[axis]) / ray[axis];
            let b = (node.hi[axis] - origin[axis]) / ray[axis];
            near = max(near, min(a, b));
            far = min(far, max(a, b));
            if (near > far) { return false; }
        }
    }
    return true;
}
"""
_BVH_TRAVERSAL = """
    if (params.shadow.z > 0.0) {
        var stack: array<u32, 64>;
        stack[0] = 0u;
        var size = 1u;
        var box_limit = 3.402823e38;
        if (light.position.w > 0.0) { box_limit = limit; }
        loop {
            if (size == 0u) { break; }
            size -= 1u;
            let node = nodes[stack[size]];
            if (!box_hit(node, origin, ray, box_limit)) { continue; }
            if (node.count > 0u) {
                for (var k = 0u; k < node.count; k += 1u) {
                    transmission *= triangle_transmission(prim_order[node.offset+k], origin, ray, limit, light.position.w);
                }
            } else {
                stack[size] = u32(node.left);
                stack[size+1u] = u32(node.right);
                size += 2u;
            }
        }
        return transmission;
    }
"""


_PARTICLE_SHADER = '''
struct PParams { screen: vec4<f32>, planes: vec4<f32>, light: vec4<f32>,
                  mode: vec4<f32>, view0: vec4<f32>, view1: vec4<f32>, view2: vec4<f32> };
// (width, height, 0, 0), (near, far, 0, 0), view-space light, (output code, id_base, 0, 0), the view
// rotation matrix's rows (scene3d._view_basis), read as columns to undo it: world = VT * view_space.
struct PInst { centre: vec2<f32>, radius: f32, z: f32, colour: vec4<f32>,
               world_radius: f32, shape: u32, tex_offset: u32, tex_w: u32, tex_h: u32, pad0: u32, pad1: u32, pad2: u32,
               world_centre: vec3<f32>, particle_id: u32 };
@group(0) @binding(0) var<uniform> pp: PParams;
@group(0) @binding(1) var<storage, read> instances: array<PInst>;
@group(0) @binding(2) var<storage, read> texels: array<vec4<f32>>;
struct POut { @builtin(position) position: vec4<f32>, @location(0) @interpolate(flat) index: u32 };
@vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32) -> POut {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
                                      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let inst = instances[index];
    let pixel = inst.centre + corners[vertex] * inst.radius;
    var out: POut;
    out.position = vec4<f32>(pixel.x / pp.screen.x * 2.0 - 1.0, 1.0 - pixel.y / pp.screen.y * 2.0, 0.0, 1.0);
    out.index = index;
    return out;
}
struct PFrag { @location(0) colour: vec4<f32>, @builtin(frag_depth) depth: f32 };
@fragment fn fs(v: POut) -> PFrag {
    // scene3d._composite_particle_chunk: u, v run -1..1 across the sprite, v down; pixel centres are at +0.5.
    let inst = instances[v.index];
    let offset = (v.position.xy - inst.centre) / inst.radius;
    let square = inst.shape == 2u;
    if (square) {
        if (abs(offset.x) > 1.0 || abs(offset.y) > 1.0) { discard; }
    } else {
        if (dot(offset, offset) > 1.0) { discard; }
    }
    let sphere = inst.shape == 1u || inst.shape == 3u;
    let facing = sqrt(max(0.0, 1.0 - dot(offset, offset)));
    var colour = inst.colour;
    var z = inst.z;
    if (sphere) {
        z = z - facing * inst.world_radius;
    }
    var out: PFrag;
    // The mesh pass writes far*(z-near)/((far-near)*z); the particle depth test compares against it.
    out.depth = clamp(pp.planes.y * (z - pp.planes.x) / ((pp.planes.y - pp.planes.x) * max(z, 1e-8)), 0.0, 1.0);
    if (pp.mode.x == 1.0) {
        // depth: scene3d._composite_particle_data_chunk repeats the same view-space z across rgb.
        out.colour = vec4<f32>(z, z, z, 1.0);
        return out;
    }
    if (pp.mode.x == 2.0) {
        // position: scene3d._composite_particle_data_chunk's world_center + world_normal * offset.
        let normal_view = vec3<f32>(select(0.0, offset.x, sphere), select(0.0, -offset.y, sphere),
                                    select(1.0, facing, sphere));
        let vt = mat3x3<f32>(pp.view0.xyz, pp.view1.xyz, pp.view2.xyz);
        let world_normal = normalize(vt * normal_view);
        let world_pos = inst.world_centre + world_normal * select(0.0, inst.world_radius, sphere);
        out.colour = vec4<f32>(world_pos, 1.0);
        return out;
    }
    if (pp.mode.x == 3.0) {
        out.colour = vec4<f32>(pp.mode.y + f32(inst.particle_id), 0.0, 0.0, 1.0);
        return out;
    }
    if (sphere) {
        let lit = max(0.0, offset.x * pp.light.x - offset.y * pp.light.y + facing * pp.light.z);
        colour = vec4<f32>(colour.rgb * (0.25 + 0.75 * lit), colour.a);
    }
    if (inst.shape == 3u) {
        // foam: a soft rim, premultiplied colour and alpha fade together (scene3d._composite_particle_chunk)
        let rim = 1.0 - min(1.0, dot(offset, offset));
        colour = colour * (rim * rim);
    }
    if (square && inst.tex_w > 0u) {
        let column = clamp(i32(floor((offset.x + 1.0) * 0.5 * f32(inst.tex_w))), 0, i32(inst.tex_w) - 1);
        let line = clamp(i32(floor((offset.y + 1.0) * 0.5 * f32(inst.tex_h))), 0, i32(inst.tex_h) - 1);
        colour = colour * texels[inst.tex_offset + u32(line) * inst.tex_w + u32(column)];
    }
    out.colour = colour;
    return out;
}
'''


def particle_pipeline(state, target=None, depth='depth32float', samples=1, data=False):
    """The instanced particle pipeline: blended and depth-tested-but-not-written for `rgba` (painter's
    order, sorted far to near by `particle_data`), or opaque and depth-written for a data output
    (`depth`/`position`/`object_id`, R7 of 7 finish (2)) so hardware depth resolves nearest-wins the same
    way it already does for the mesh passes, in any draw order. The editor viewport asks for its own
    target formats."""
    target = target or state['format']
    key = ('particles', target, depth, samples, data)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    module = device.create_shader_module(code=_PARTICLE_SHADER)
    fragment_target = {'format': target}
    if not data:
        blend = {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}
        fragment_target['blend'] = {'color': blend, 'alpha': blend}
    pipeline = device.create_render_pipeline(layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
        primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
        depth_stencil={'format': depth, 'depth_write_enabled': data, 'depth_compare': 'less'},
        multisample={'count': samples},
        fragment={'module': module, 'entry_point': 'fs', 'targets': [fragment_target]})
    state['pipelines'][key] = pipeline
    return pipeline


_PARTICLE_OUTPUT_MODES = {'rgba': 0.0, 'depth': 1.0, 'position': 2.0, 'object_id': 3.0}


def particle_data(scene, camera, width, height, limits, cancel, output='rgba'):
    """Instance and sprite-texel arrays for the particle draw, sorted far to near, or None. `output`
    "rgba" draws the old fixed look (a "pbr" particle_material is refused earlier, in `render`); `depth`,
    `position` and `object_id` (R7 of 7 finish (2)) instead write that data output exactly like
    `scene3d._composite_particle_data_chunk`, matching a mesh's own first-hit, unantialiased data pass."""
    _cancel(cancel)
    eye, view = scene3d._view_basis(camera)
    focal = 1 / math.tan(math.radians(camera.fov) / 2)
    sprites = scene3d.particle_sprites(scene, camera, width, height, eye, view, focal, width / height)
    if sprites is None:
        return None
    # The "pbr" material/ramp fields (R7 of 7) are CPU-only for now (docs/3D_FOUNDATION.md "Particles"
    # known limits): the GPU draw keeps its old fixed look and simply ignores them here.
    (z, centre, radius, color, shape, world_radius, texture_id, textures,
     world_center, _pbr, _metallic, _roughness, _specular, _emission, particle_id) = sprites
    count = len(z)
    offsets, dims, chunks, total = [], [], [], 0
    for image in textures:
        offsets.append(total)
        dims.append(image.shape[:2])
        chunks.append(np.ascontiguousarray(image, 'f4').reshape(-1, 4))
        total += len(chunks[-1])
    if count * 80 > limits['max-storage-buffer-binding-size'] or total * 16 > limits['max-storage-buffer-binding-size']:
        raise Unsupported('particle data exceeds the adapter storage buffer limit')
    packed = np.zeros((count, 20), 'f4')
    packed[:, 0:2], packed[:, 2], packed[:, 3], packed[:, 4:8] = centre, radius, z, color
    packed[:, 8] = world_radius
    packed[:, 16:19] = world_center
    words = packed.view('u4')
    words[:, 9] = shape
    words[:, 19] = particle_id
    textured = texture_id >= 0
    if textured.any():
        table = np.array(offsets, 'u4'), np.array([d[1] for d in dims], 'u4'), np.array([d[0] for d in dims], 'u4')
        words[textured, 10] = table[0][texture_id[textured]]
        words[textured, 11] = table[1][texture_id[textured]]
        words[textured, 12] = table[2][texture_id[textured]]
    texel_data = np.concatenate(chunks) if chunks else np.zeros((1, 4), 'f4')
    params = np.array([width, height, 0, 0, camera.near, camera.far, 0, 0,
                       *scene3d._VIEW_LIGHT, 0, _PARTICLE_OUTPUT_MODES[output], scene3d.particle_id_base(scene), 0, 0,
                       *view[0], 0, *view[1], 0, *view[2], 0], 'f4')
    return packed, texel_data, params


def _pipeline(state, data, phase, bvh=False, smoke=False):
    key = (data, phase, bvh, smoke)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    code = _SHADER
    if bvh:
        code = _BVH_DECL + code.replace('// BVH_TRAVERSAL', _BVH_TRAVERSAL)
    if smoke:
        from . import gpuvolume
        code = gpuvolume.MESH_SHADOW_WGSL + code.replace(
            '// VOLUME_SHADOW', 'transmission *= volume_transmission(v.world, lights[i]);')
    module = device.create_shader_module(code=code)
    target = {'format': 'rgba32float' if data else state['format']}
    if not data:
        blend = {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}
        target['blend'] = {'color': blend, 'alpha': blend}
    attributes = [dict(format=f, offset=o, shader_location=i) for i, (f, o) in enumerate(
        [('float32x3', 0), ('float32x3', 12), ('float32x3', 24), ('float32x2', 36), ('float32x4', 44), ('float32', 60), ('float32x3', 64), ('float32', 76), ('float32x4', 80),
         ('float32x3', 96), ('float32x4', 108), ('float32x3', 124), ('float32x4', 136)])]
    pipeline = device.create_render_pipeline(layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': [
            {'array_stride': 152, 'step_mode': 'vertex', 'attributes': attributes}]},
        primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
        depth_stencil={'format': 'depth32float', 'depth_write_enabled': phase != 1, 'depth_compare': 'less'},
        fragment={'module': module, 'entry_point': 'fs', 'constants': {'PASS': phase}, 'targets': [target]})
    state['pipelines'][key] = pipeline
    return pipeline


def _cancel(event):
    if event is not None and event.is_set():
        from .cancellation import Cancelled
        raise Cancelled()


def _prepare(scene, camera, width, height, cancel):
    eye, view = scene3d._view_basis(camera)
    focal = 1 / math.tan(math.radians(camera.fov) / 2)
    vertices, queue, materials = [], [], []
    for object_id, geometry in enumerate(scene.geometries, 1):
        _cancel(cancel)
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        local = (view @ (world-eye).T).T
        normals = None
        if geometry.normals is not None:
            normals = (np.linalg.inv(matrix[:3, :3]).T @ geometry.normals.T).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        uvs = geometry.uvs if geometry.uvs is not None else np.zeros((len(world), 2), 'f4')
        texture = geometry.texture if geometry.uvs is not None else None
        mips = scene3d._mip_chain(texture) if texture is not None else [np.ones((1, 1, 4), 'f4')]
        # Y3 of 3, part 1: the four remaining PBR texture maps (base colour already shades through
        # `mips` above, pbr or not) plus flat `emissive_color`, matching `scene3d._shade_fragments`'
        # glTF-style metallic-roughness packing (G roughness, B metallic) and tangent-space normal
        # decode. Only read for a `pbr` geometry with UVs, like the CPU reference's own checks.
        is_pbr = geometry.material == 'pbr' and geometry.uvs is not None
        mr_texture = geometry.metallic_roughness_texture if is_pbr else None
        normal_texture = geometry.normal_texture if is_pbr else None
        occlusion_texture = geometry.occlusion_texture if is_pbr else None
        emissive_texture = geometry.emissive_texture if is_pbr else None
        material = len(materials)
        materials.append((mips, mr_texture, normal_texture, occlusion_texture, emissive_texture))
        tint = np.asarray(geometry.color, 'f4').copy()
        tint[:3] *= tint[3]
        # Y3 of 3, part 1: (metallic, roughness, dielectric F0, is-pbr flag), matching
        # scene3d._shade_fragments' `0.08 * clip(pbr_specular, 0, 1)` dielectric F0 derivation; zero
        # (the is-pbr flag off) reproduces every earlier (Blinn-Phong) render exactly.
        if geometry.material == 'pbr':
            pbr = (float(np.clip(geometry.metallic, 0, 1)), float(np.clip(geometry.pbr_roughness, 0, 1)),
                  0.08 * float(np.clip(geometry.pbr_specular, 0, 1)), 1.0)
        else:
            pbr = (0.0, 0.0, 0.0, 0.0)
        maps = (1.0 if mr_texture is not None else 0.0, 1.0 if normal_texture is not None else 0.0,
                1.0 if occlusion_texture is not None else 0.0, 1.0 if emissive_texture is not None else 0.0)
        emissive_color = np.asarray(geometry.emissive_color, 'f4') if geometry.material == 'pbr' else np.zeros(3, 'f4')
        scales = (float(geometry.normal_scale), float(geometry.occlusion_strength), 0.0, 0.0) if is_pbr else (0.0, 0.0, 0.0, 0.0)
        for index, tri in enumerate(geometry.triangles):
            if index % 256 == 0:
                _cancel(cancel)
            z = -local[tri, 2]
            if (z <= camera.near).all() or (z >= camera.far).all():
                continue
            if normals is None:
                face = np.cross(world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]])
                ns = np.broadcast_to(face / max(float(np.linalg.norm(face)), 1e-8), (3, 3))
            else:
                ns = normals[tri]
            # scene3d.render's raster reference `_flat_tangents`-style UV-gradient tangent: one per
            # triangle (world space), broadcast to its three vertices, so normal mapping needs no
            # separate per-vertex tangent pass.
            e1, e2 = world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]]
            duv1, duv2 = uvs[tri[1]]-uvs[tri[0]], uvs[tri[2]]-uvs[tri[0]]
            det = duv1[0]*duv2[1]-duv2[0]*duv1[1]
            tri_tangent = (e1*duv2[1]-e2*duv1[1])/det if abs(det) > 1e-12 else e1
            ts = np.broadcast_to(tri_tangent, (3, 3))
            attrs = np.concatenate((local[tri], world[tri], ns, uvs[tri], ts), axis=1).astype('f4')
            # Reference clipping also defines sorting and the area-based mip footprint.
            # Hardware still performs homogeneous near/far and viewport clipping.
            for clipped in scene3d._clip_near(attrs, z, camera.near):
                zs = -clipped[:, 2]
                a, b, c = scene3d._to_pixels(clipped[:, :3], zs, focal, width/height, width, height)
                den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
                if abs(den) < 1e-8:
                    continue
                e, f = clipped[1, 9:11]-clipped[0, 9:11], clipped[2, 9:11]-clipped[0, 9:11]
                area = abs(float(e[0]*f[1]-e[1]*f[0]))*mips[0].shape[0]*mips[0].shape[1]
                lod = np.clip(round(.5*math.log2(max(area/max(abs(den), 1e-8), 1))), 0, len(mips)-1)
                packed = np.empty((3, 38), 'f4')
                packed[:, :11] = clipped[:, :11]
                # All three vertices agree, regardless of the provoking vertex.
                packed[:, 11:15] = tint
                packed[:, 15] = lod
                packed[:, 16:19] = (geometry.specular, geometry.shininess, geometry.emission)
                packed[:, 19] = object_id
                packed[:, 20:24] = pbr
                packed[:, 24:27] = clipped[:, 11:14]
                packed[:, 27:31] = maps
                packed[:, 31:34] = emissive_color
                packed[:, 34:38] = scales
                queue.append((float(zs.mean()), len(vertices)*3, material))
                vertices.append(packed)
    return eye, focal, vertices, sorted(queue, key=lambda q: q[0], reverse=True), materials


def _shadow_data(scene, count, limit, cancel):
    """Pack all world triangles, including invisible/off-camera geometry."""
    size = max(1, count) * 48
    if size > limit:
        raise ValueError(f'Shadow triangle buffer exceeds max_storage_buffer_binding_size: {size} > {limit} bytes')
    packed = np.zeros((max(1, count), 3, 4), 'f4')
    low, high = np.full(3, np.inf, 'f4'), np.full(3, -np.inf, 'f4')
    offset = 0
    if count:
        for geometry in scene.geometries:
            _cancel(cancel)
            n = len(geometry.triangles)
            if not n:
                continue
            matrix = geometry.world_matrix()
            world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
            triangles = world[geometry.triangles]
            low = np.minimum(low, triangles.min(axis=(0, 1)))
            high = np.maximum(high, triangles.max(axis=(0, 1)))
            block = packed[offset:offset+n]
            block[:, 0, :3] = triangles[:, 0]
            block[:, 1, :3] = triangles[:, 1] - triangles[:, 0]
            block[:, 2, :3] = triangles[:, 2] - triangles[:, 0]
            block[:, 0, 3] = np.clip(geometry.color[3], 0, 1)
            offset += n
    return packed, 1e-3 * max(1.0, float((high-low).max()) if count else 1.0)


def _splat_extras(scene, ambient=0.0, provider=None, cancel=None):
    """What splat shading reads beyond the scene's lights, or None: the environments and, for instances
    with Indirect samples, the CPU `splatindirect.IndirectLight` (its hemisphere rays run on the CPU
    casters; the drawer then gets the same per-splat colours the reference renderer computes). `provider`
    is the scene's `_SplatShadows` when one exists, so hit splats reuse its visibility cache."""
    from .envlight import SplatLighting
    environments = getattr(scene, 'environments', ())
    indirect = None
    if scene3d._indirect_instances(scene) > 0:
        from .splatindirect import IndirectLight
        scene3d._indirect_budget(scene, 0)
        if provider is None:
            provider = scene3d._SplatShadows(scene.splats, scene.lights, None, None, .001, cancel)
            provider.relit_shadows = False
        indirect = IndirectLight(provider, ambient, environments, cancel=cancel)
    return SplatLighting(environments, None, indirect) if (environments or indirect) else None


def render(scene, camera, width, height, background=(0, 0, 0, 0), ambient=0.0,
           samples=1, output='rgba', cancel=None, adapter=None, *, mode='raster', volume=None):
    """Render a read-only premultiplied float32 image; raise on unavailable GPUs.

    Projection and viewport shade rendering are unsupported. Callers can catch
    Unsupported/RuntimeError and use scene3d.render as their fallback.
    """
    if getattr(scene, 'instances', ()):
        # gpuinstance traces InstanceSet items on a two-level GPU BVH without flattening them
        # (docs/3D_ROADMAP.md "Instancing"); everything else here reads only scene.geometries,
        # so a scene with instances alongside other content would otherwise silently drop them.
        pure = not (scene.geometries or scene.splats or getattr(scene, 'particles', ())
                    or getattr(scene, 'volumes', ()))
        if mode == 'raytrace' and pure:
            if output == 'splats':
                raise Unsupported('splats output is CPU-only')
            if output == 'relight':
                raise Unsupported('the relight bundle output is CPU-only for now')
            if output not in scene3d.RENDER_OUTPUTS:
                raise ValueError(f'Unknown 3D render output {output!r}')
            _cancel(cancel)
            from . import gpuinstance
            with _lock:
                state = _state(adapter)
                return gpuinstance.render(state, scene, camera, width, height, background,
                                          ambient, output, samples, cancel=cancel)
        raise Unsupported('GPU rendering of Instance3D instances needs raytrace mode and a scene made '
                          'entirely of instances (no ordinary geometry, splats, particles or volumes '
                          'alongside them); the CPU renderer draws every combination')
    pbr_geometries = [g for g in scene.geometries if g.material == 'pbr']
    if pbr_geometries:
        # Y3 of 3 (docs/3D_FOUNDATION.md "Materials"): the wgpu rasterizer shades a `pbr` geometry's
        # metallic/roughness/specular factors with the same Cook-Torrance GGX as the CPU reference
        # (`ggx_response` in `_SHADER`, matching `scene3d._shade_pbr_mesh`), lit by Directional/Point/
        # Spot lights, its own texture bindings (part 1) and (Z1 of 2 finish) a single Environment's
        # split-sum image-based diffuse and specular cross term (`pbr_env_diffuse`/`pbr_env_specular`
        # in `_SHADER`, matching `scene3d._mesh_pbr_environment`) and Rect/Disc/Sphere area lights
        # (the same Monte Carlo estimator as `scene3d._area_light_contribution`, see the area-light
        # checks below). The GPU ray tracer still has no `pbr` material table at all, and a scene with
        # more than one Environment stays CPU-only either way (`_environment_resources` only ever
        # binds `environments[0]`).
        if mode != 'raster':
            raise Unsupported('physically based (metal/roughness) mesh materials are CPU-only for now on the ray-traced mode')
    if any(light.kind in scene3d._AREA for light in scene.lights) and mode != 'raster':
        # Z1 of 2 finish: Rect/Disc/Sphere area lights shade on the raster path now (see
        # `_area_light_resources`, matching `scene3d._area_light_contribution`'s fixed low-discrepancy
        # sample set); the ray tracer (`gpurt_render.py`) still has no material table for them at all.
        raise Unsupported('Rect/Disc/Sphere area lights are CPU-only for now on the ray-traced mode')
    has_scene_volumes = bool(getattr(scene, 'volumes', ()))
    if output in scene3d.VOLUME_OUTPUTS or (output == 'depth' and has_scene_volumes):
        return _render_volume_passes(scene, camera, width, height, background, output, cancel, adapter, mode, volume)
    volumes = output == 'rgba' and has_scene_volumes
    if (volumes and scene.geometries and len(scene.volumes) > 4 and volume is not None and volume.shadow_density > 0
            and any(light.shadows and light.intensity > 0 for light in scene.lights)):
        raise Unsupported('smoke shadows on meshes from more than four volumes are CPU-only')
    if volumes and (mode == 'raytrace' or scene.splats):
        raise Unsupported('volumes drawn with the ray tracer or together with splats are CPU-only')
    environments = getattr(scene, 'environments', ())
    if environments and scene.geometries and (mode != 'raster' or len(environments) > 1):
        # Y1 of 2 finish (2): a mesh lit by a single Environment now shades in the GPU raster path
        # (`env_diffuse`/`env_specular` in `_SHADER`, matching `scene3d._shade_fragments` and
        # `_mesh_environment_specular`). The ray tracer (`gpurt_render.py`) has no environment
        # sampling yet, and more than one Environment (like `gpupathtrace`'s own "one environment"
        # scope) is still the CPU reference either way.
        raise Unsupported('environment light on meshes is CPU-only' if mode != 'raster' else
                          'more than one environment light on meshes is CPU-only')
    if scene.geometries and any(getattr(i, 'relight', 0) > 0 and getattr(i, 'reflection_samples', 0) > 0
                                for i in scene.splats):
        raise Unsupported('splat reflections of meshes are CPU-only')
    if scene.geometries and scene3d._indirect_instances(scene) > 0:
        raise Unsupported('splat indirect light and occlusion with meshes in the scene is CPU-only')
    particles = bool(getattr(scene, 'particles', ()))
    if particles and (mode == 'raytrace' or scene.splats):
        raise Unsupported('particles drawn with the ray tracer or together with splats are CPU-only')
    if particles and output == 'rgba' and any(i.material == 'pbr' for i in scene.particles):
        # R7 of 7 finish (2), like the mesh "pbr" check above: the instanced sprite pipeline's fragment
        # shader only has the old fixed "headlight" look (`_PARTICLE_SHADER`), so a "pbr" particle used
        # to draw with that fixed look silently instead of its lit/shadowed material; raising here sends
        # `auto` scenes to the CPU reference (`scene3d._draw_particles`, "pbr" branch) for a correct picture.
        raise Unsupported('particle_material "pbr" lighting and shadows are CPU-only for now')
    if mode == 'raytrace':
        if output == 'splats':
            raise Unsupported('splats output is CPU-only')
        if output == 'relight':
            raise Unsupported('the relight bundle output is CPU-only for now')
        if output not in scene3d.RENDER_OUTPUTS:
            raise ValueError(f'Unknown 3D render output {output!r}')
        if scene.splats:
            if output != 'rgba':
                raise Unsupported('splat data passes are CPU-only')
            if not scene3d._opaque_meshes(scene):
                raise Unsupported('transparent meshes mixed with splats are CPU-only')
        if any(g.projection is not None for g in scene.geometries):
            raise Unsupported('Camera-projected geometry is not implemented by wgpu')
        _cancel(cancel)
        from . import gpurt_render
        with _lock:
            state = _state(adapter)
            reason = gpurt_render.check_capability(state)
            if reason is not None:
                raise Unsupported(reason)
            if output == 'rgba':
                return gpurt_render.render_beauty(
                    state, scene, camera, width, height, background, ambient, samples, cancel=cancel)
            return gpurt_render.render(
                state, scene, camera, width, height, background, ambient, output, samples, cancel=cancel)
    if output == 'rgba' and any(g.material == 'liquid' for g in scene.geometries):
        raise Unsupported('the raster approximation of liquid surfaces is CPU-only (the ray tracer does them on the GPU)')
    if scene.splats:
        if output != 'rgba':
            raise Unsupported('splat data passes and the `splats` output are CPU-only')
        if not scene3d._opaque_meshes(scene):
            raise Unsupported('transparent meshes mixed with splats are CPU-only')
        # The raster shader has no splat casters, so shadow work that involves splats (meshes shadowing
        # relit or caught splats, splats shadowing meshes and each other) goes to the ray tracer, whose
        # opaque-mesh result equals the raster one.
        shadowed = any(light.shadows and light.intensity > 0 for light in scene.lights)
        if shadowed and (scene.geometries or any(getattr(i, 'relight', 0) > 0 for i in scene.splats)):
            return render(scene, camera, width, height, background, ambient, samples, output,
                          cancel, adapter, mode='raytrace')
    # The splat contribution layer is CPU-only, including empty scenes.
    if output == 'splats':
        raise Unsupported('splats output is CPU-only')
    # The relight bundle is raster-only on the CPU for now (docs/3D_ROADMAP.md "Design: relight
    # passes"); it has no GPU raster shader branch, so reject it explicitly rather than let an
    # unrecognised output index reach the WGSL output-mode switch.
    if output == 'relight':
        raise Unsupported('the relight bundle output is CPU-only for now')
    if mode != 'raster':
        raise ValueError(f'Unknown 3D render mode {mode!r}')
    if output == 'normals_blend':
        output = 'normals'   # without splats (they returned above) the blend is the plain first-hit pass
    if output == 'shade':
        raise Unsupported('Viewport shade mode is not implemented by wgpu')
    if output not in scene3d.RENDER_OUTPUTS:
        raise ValueError(f'Unknown 3D render output {output!r}')
    if any(g.projection is not None for g in scene.geometries):
        raise Unsupported('Camera-projected geometry is not implemented by wgpu')
    if sum(len(g.triangles) for g in scene.geometries) > scene3d.MAX_TRIANGLES:
        raise Unsupported(f'Scene exceeds {scene3d.MAX_TRIANGLES} triangles')
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    samples = max(1, min(int(samples), 4)) if output in scene3d.LIGHT_OUTPUTS else 1
    _cancel(cancel)
    # Z1 of 2 finish: an area light's Monte Carlo loop traces one shadow ray per sample per fragment
    # (`area_light_shade`), so it costs `light_samples` shadow-ray units, not one, in the same work
    # estimate a Directional/Point/Spot light uses (`_band_plan` below then still picks the right BVH
    # threshold and submission band count for the real ray count).
    shadow_count = sum(
        (int(np.clip(light.light_samples, 1, scene3d.SHADOW_SAMPLES_MAX * 4)) if light.kind in scene3d._AREA else 1)
        for light in scene.lights if light.shadows and light.intensity > 0
    ) if output in ('rgba', 'diffuse', 'specular') else 0
    triangles = sum(len(g.triangles) for g in scene.geometries) if shadow_count else 0
    work = width * height * samples ** 2 * shadow_count * triangles
    with _lock:
        state = _state(adapter)
        volume_bands = 0
        if volumes:
            from . import gpuvolume, volumerender
            volume = volume if volume is not None else volumerender.VolumeSettings()
            gpuvolume.check(state, scene, volume)
            lit_lights = sum(light.intensity > 0 for light in scene.lights)
            volume_bands = gpuvolume.band_plan(state, gpuvolume.work_estimate(
                scene, camera, width*samples, height*samples, volume, lit_lights, triangles), height*samples)
        if scene.splats:
            from . import gpusplat
            reason = gpusplat.check_capability(state)
            if reason is not None:
                raise Unsupported(reason)
        global last_shadow_path
        last_shadow_path = 'brute'
        shadow_prepared = None
        bvh_data = None
        limits = state['device'].limits if 'device' in state else {}
        bvh_threshold = (SHADOW_BVH_THRESHOLD if SHADOW_BVH_THRESHOLD is not None
                         else SHADOW_BVH_THRESHOLDS[_adapter_kind(state)])
        if triangles > bvh_threshold and _bvh_capable(limits, 48, triangles*4):
            shadow_prepared = _shadow_data(scene, triangles, limits['max-storage-buffer-binding-size'], cancel)
            packed = shadow_prepared[0]
            tri_set = raytrace.TriangleSet(packed[:, 0, :3], packed[:, 1, :3],
                                          packed[:, 2, :3], packed[:, 0, 3])
            bvh = raytrace.Bvh.build(*tri_set.aabbs(), cancel=cancel)
            if _bvh_capable(limits, len(bvh.left)*48, triangles*4):
                bvh_data = _pack_bvh(bvh, cancel)
                last_shadow_path = 'bvh'
                levels = math.log2(triangles+2)
                work = width * height * samples**2 * shadow_count * 16 * levels + 16 * triangles * levels
        bands = _band_plan(state, work, last_shadow_path, height*samples)
        if volume_bands > len(bands):
            rows = height*samples
            bands = [(i * rows // volume_bands, (i + 1) * rows // volume_bands) for i in range(volume_bands)]
        _cancel(cancel)
        result = _render(state, scene, camera, width*samples, height*samples,
                         background, ambient, output, cancel, triangles, shadow_prepared, bvh_data, bands=bands,
                         volume=volume if volumes else None)
        if scene.splats:
            mesh_depth = None
            if scene.geometries:
                depth = _render(state, scene, camera, width*samples, height*samples,
                                (0, 0, 0, 0), ambient, 'depth', cancel)
                # The depth shader interpolates -view.z, already positive
                # camera-forward distance (not radial ray distance).
                mesh_depth = np.where(depth[..., 3] > 0, depth[..., 0], np.inf)
            lighting = ((scene.lights, ambient, None, _splat_extras(scene, ambient, None, cancel)) if any(
                getattr(i, 'relight', 0) > 0 for i in scene.splats) else None)
            splat_rgb, splat_alpha = gpusplat.render_layer(
                state, scene.splats, camera, width*samples, height*samples,
                mesh_depth, lighting=lighting, cancel=cancel)
            result[..., :3] = splat_rgb + (1-splat_alpha[..., None])*result[..., :3]
            result[..., 3] = splat_alpha + (1-splat_alpha)*result[..., 3]
        if output == 'rgba' and any(getattr(e, 'visible_to_camera', False) for e in environments):
            # Y1 of 2 finish (2), the "background image" part: a `visible_to_camera` Environment
            # replaces the flat background on camera rays that hit nothing, exactly like
            # `scene3d._visible_background` -- reused directly (not reimplemented) on this GPU
            # readback, since it is plain NumPy and area lights never reach here (still refused
            # above), so its own area-light branch is always a no-op in this caller.
            depth_image = _render(state, scene, camera, width*samples, height*samples,
                                  (0, 0, 0, 0), ambient, 'depth', cancel)
            depth = np.where(depth_image[..., 3] > 0, depth_image[..., 0], np.inf)
            view_eye, view_matrix = scene3d._view_basis(camera)
            focal = 1 / math.tan(math.radians(camera.fov) / 2)
            scene3d._visible_background(scene, width*samples, height*samples, result, depth, view_eye,
                                        view_matrix, focal, (width*samples)/(height*samples),
                                        float(np.clip(background[3], 0, 1)))
    if samples > 1:
        result = result.reshape(height, samples, width, samples, 4).mean(axis=(1, 3))
    result.flags.writeable = False
    return result


def _render_volume_passes(scene, camera, width, height, background, output, cancel, adapter, mode, volume):
    """A volume control pass, or the depth output merged with the volumes' first hit, raymarched on the GPU
    (gpuvolume.render_passes) over the opaque meshes' depth, which the GPU raster path renders first."""
    from . import gpuvolume, volumerender
    if mode == 'raytrace' or scene.splats:
        raise Unsupported('volume passes drawn with the ray tracer or together with splats are CPU-only')
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    settings = volume if volume is not None else volumerender.VolumeSettings()
    if not scene.volumes:
        return np.zeros((height, width, 4), np.float32)
    meshes = replace(scene, volumes=(), particles=(), lights=(), splats=())
    if meshes.geometries:
        image = render(meshes, camera, width, height, output='depth', cancel=cancel, adapter=adapter, mode=mode)
        mesh_depth = np.where(image[..., 3] > 0, image[..., 0], np.inf).astype(np.float32)
    else:
        image = np.zeros((height, width, 4), np.float32)
        mesh_depth = np.full((height, width), np.inf, np.float32)
    with _lock:
        state = _state(adapter)
        got = gpuvolume.render_passes(state, scene, camera, width, height, mesh_depth, settings, output, cancel)
    if output == 'depth':
        first = (got['first_t'] / got['length']).astype(np.float32)
        out = np.array(image, np.float32)
        hit = first < mesh_depth
        out[hit, :3] = first[hit, None]
        out[hit, 3] = 1
    else:
        out = volumerender.finish_pass(got, output.split('_', 1)[1], camera, settings)
    out.flags.writeable = False
    return out


def light_table(lights):
    """The 20-float-per-light storage table the raster and volume shaders read; `lights` are (light, position, direction)."""
    light_data = np.zeros((max(1, len(lights)), 20), 'f4')
    for i, (light, position, direction) in enumerate(lights):
        light_data[i, :3] = position
        light_data[i, 3] = light.kind in scene3d._POSITIONAL
        light_data[i, 4:7] = direction
        light_data[i, 7] = light.shadows
        light_data[i, 8:11] = np.asarray(light.color)*light.intensity
        light_data[i, 11], light_data[i, 12:16] = scene3d._falloff_power(light), scene3d._cone_terms(light)
        light_data[i, 16:19] = scene3d._shadow_terms(light)
    return light_data


def _area_light_resources(lights):
    """(lights_table (L, 12) f32, samples_table (S, 8) f32) for the `AreaLight`/`AreaSample` storage
    buffers `_SHADER` reads (Z1 of 2 finish): the same fixed low-discrepancy sample points and normals
    `scene3d._area_light_contribution` computes on the CPU (`scene3d._area_light_samples`), uploaded
    once and reused by every fragment. One row per light even with none, like `light_table` above.

    `lights_table` columns: center.xyz, two_sided, radiance.xyz, area, sample offset, sample count,
    shadow bias scale (matching `scene3d._shadow_terms`'s own bias-scale column), shadows-on flag."""
    lights_data = np.zeros((max(1, len(lights)), 12), 'f4')
    sample_rows = []
    offset = 0
    for i, light in enumerate(lights):
        center, _ = light.world()
        radiance, area = scene3d._area_light_radiance(light)
        count = int(np.clip(light.light_samples, 1, scene3d.SHADOW_SAMPLES_MAX * 4))
        points, normals = scene3d._area_light_samples(light, count)
        lights_data[i, 0:3] = center
        lights_data[i, 3] = float(bool(light.two_sided))
        lights_data[i, 4:7] = radiance
        lights_data[i, 7] = area
        lights_data[i, 8] = offset
        lights_data[i, 9] = count
        lights_data[i, 10] = light.shadow_bias / scene3d.SHADOW_BIAS_DEFAULT
        lights_data[i, 11] = float(bool(light.shadows))
        rows = np.zeros((count, 8), 'f4')
        rows[:, 0:3] = points
        rows[:, 4:7] = normals
        sample_rows.append(rows)
        offset += count
    samples_data = np.concatenate(sample_rows) if sample_rows else np.zeros((1, 8), 'f4')
    return lights_data, samples_data


def _environment_resources(state, keep, environments):
    """The `EnvGlobals` uniform block and the atlas texture `_SHADER`'s group(2) reads: `environments[0]`
    (raster mode allows at most one, checked by the caller), or a disabled/zeroed block when there is
    none -- the shader always declares group(2), so a bind group is always needed. `rot0`/`rot1`/`rot2`
    are the columns of `envlight.Environment._local`'s combined rotation (the environment's own turn
    and its parent's), read back into a `mat3x3<f32>` by `env_local`, so a parented or rotated
    Environment matches the CPU reference exactly, unlike viewportgpu.py's root-level-only dome.

    The overwhelmingly common case has no Environment at all, so that disabled block/atlas/sampler is
    built once per adapter `state` and cached there (`env_disabled`), not rebuilt (and not `keep`-ed
    for per-call destruction) on every render -- the ordinary per-material textures are already rebuilt
    fresh each call, but this one never varies, and a fresh 64x32x6 texture upload on every mesh render
    would be pure waste."""
    from . import envlight
    wgpu, device = state['wgpu'], state['device']
    if not environments:
        cached = state.get('env_disabled')
        if cached is None:
            block = np.zeros(9 * 4 + 4 * 4, np.float32)
            atlas = np.ones((6, 1, 4), np.float32)
            buffer = device.create_buffer_with_data(data=block, usage=wgpu.BufferUsage.UNIFORM)
            texture = device.create_texture(size=(1, 6, 1), format='rgba16float',
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            device.queue.write_texture({'texture': texture, 'mip_level': 0, 'origin': (0, 0, 0)},
                np.ascontiguousarray(atlas, 'f2'), {'bytes_per_row': 8, 'rows_per_image': 6}, (1, 6, 1))
            sampler = device.create_sampler(mag_filter='linear', min_filter='linear', address_mode_u='repeat')
            cached = state['env_disabled'] = (buffer, texture.create_view(), sampler)
        return cached
    tile_w, tile_h = envlight.PREFILTER_SIZE
    block = np.zeros(9 * 4 + 4 * 4, np.float32)
    env = environments[0]
    pre = env._pre()
    gain = float(env.intensity) * np.asarray(env.tint, np.float64)
    dirs, _ = envlight.direction_grid(tile_w, tile_h)
    level0 = envlight.sample_map(pre.levels[0], dirs.reshape(-1, 3)).reshape(tile_h, tile_w, 3)
    levels = [level0] + [np.asarray(level, np.float64) for level in pre.levels[1:]]
    rgb = (np.concatenate(levels, axis=0) * gain[None, None, :]).astype(np.float32)
    atlas = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], axis=-1)
    # `array<vec4<f32>, 9>` pads every SH coefficient to 16 bytes; only the first 3 floats of
    # each 4 are used (see `env_diffuse`'s `envg.sh[i].rgb`).
    block[:36].reshape(9, 4)[:, :3] = (pre.sh * gain[None, :]).astype(np.float32)
    angle = math.radians(float(env.rotation))
    turn = np.array(((math.cos(angle), 0, -math.sin(angle)), (0, 1, 0), (math.sin(angle), 0, math.cos(angle))))
    parent = np.asarray(env.parent, np.float64)[:3, :3]
    norms = np.linalg.norm(parent, axis=0)
    rot = parent / np.where(norms > 1e-12, norms, 1.0)
    matrix = (rot.T @ turn).astype(np.float32)   # envlight.Environment._local: d @ matrix.T
    block[36:39], block[40:43], block[44:47] = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    block[48] = 1.0
    env_buffer = keep(device.create_buffer_with_data(data=block, usage=wgpu.BufferUsage.UNIFORM))
    atlas_texture = keep(device.create_texture(size=(atlas.shape[1], atlas.shape[0], 1), format='rgba16float',
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
    device.queue.write_texture({'texture': atlas_texture, 'mip_level': 0, 'origin': (0, 0, 0)},
        np.ascontiguousarray(atlas, 'f2'), {'bytes_per_row': atlas.shape[1] * 8, 'rows_per_image': atlas.shape[0]},
        (atlas.shape[1], atlas.shape[0], 1))
    # `envlight.sample_map` wraps in U (longitude) but clamps in V; `address_mode_u='repeat'` matches
    # the wrap (the default is clamp-to-edge, which would seam at the u=0/1 meridian). V clamps to the
    # whole atlas, which is harmless on its own -- `env_specular` insets its own V reads by half a
    # texel so hardware bilinear never bleeds across two tiles' shared row into the wrong roughness level.
    sampler = device.create_sampler(mag_filter='linear', min_filter='linear', address_mode_u='repeat')
    return env_buffer, atlas_texture.create_view(), sampler


def _render(state, scene, camera, width, height, background, ambient, output, cancel, shadow_triangles=0, shadow_prepared=None, bvh_data=None, *, bands=None, volume=None):
    wgpu, device = state['wgpu'], state['device']
    data = output in scene3d.DATA_OUTPUTS
    if bands is None:
        bands = _band_plan(state, 0, 'brute', height)
    shadow_data, bias = shadow_prepared if shadow_prepared is not None else _shadow_data(scene, shadow_triangles,
        device.limits['max-storage-buffer-binding-size'], cancel)
    _cancel(cancel)
    bg = np.asarray(background, 'f4').copy()
    bg[3] = np.clip(bg[3], 0, 1)
    bg[:3] *= bg[3]
    if output != 'rgba':
        bg[:] = 0
    eye, focal, vertices, queue, materials = _prepare(scene, camera, width, height, cancel)
    _cancel(cancel)
    sprites = None
    if output in _PARTICLE_OUTPUT_MODES and getattr(scene, 'particles', ()):
        sprites = particle_data(scene, camera, width, height, device.limits, cancel, output=output)
    has_volumes = volume is not None and output == 'rgba' and bool(getattr(scene, 'volumes', ()))
    if not vertices and sprites is None and not has_volumes:
        return np.broadcast_to(bg, (height, width, 4)).copy()
    resources = []
    def keep(resource):
        resources.append(resource)
        return resource
    try:
        all_lights = [light for light in scene.lights if light.intensity > 0]
        # Z1 of 2 finish: Rect/Disc/Sphere lights go through the separate `AreaLight`/`AreaSample`
        # buffers below (`_area_light_resources`), not the Directional/Point/Spot `lights` table --
        # `light_table` has no "kind" column, so a mis-filed area light would shade as a directional one.
        area = [light for light in all_lights if light.kind in scene3d._AREA]
        lights = [(light, *light.world()) for light in all_lights if light.kind not in scene3d._AREA]
        light_data = light_table(lights)
        area_lights_data, area_samples_data = _area_light_resources(area)
        params = np.array([focal/(width/height), focal, camera.near, camera.far,
                           *eye, 0, ambient, len(lights), scene3d.RENDER_OUTPUTS.index(output), len(area),
                           bias, shadow_triangles, bvh_data is not None, 0], 'f4')
        uniform = keep(device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM))
        light_buffer = keep(device.create_buffer_with_data(data=light_data, usage=wgpu.BufferUsage.STORAGE))
        area_light_buffer = keep(device.create_buffer_with_data(data=area_lights_data, usage=wgpu.BufferUsage.STORAGE))
        area_sample_buffer = keep(device.create_buffer_with_data(data=area_samples_data, usage=wgpu.BufferUsage.STORAGE))
        shadow_buffer = keep(device.create_buffer_with_data(data=shadow_data, usage=wgpu.BufferUsage.STORAGE))
        bvh_entries = []
        if bvh_data is not None:
            for binding, array in zip((5, 6), bvh_data):
                buffer = keep(device.create_buffer_with_data(data=array, usage=wgpu.BufferUsage.STORAGE))
                bvh_entries.append({'binding': binding, 'resource': {'buffer': buffer}})
        vertex_buffer = keep(device.create_buffer_with_data(data=np.concatenate(vertices), usage=wgpu.BufferUsage.VERTEX)) if vertices else None
        sampler = device.create_sampler(mag_filter='linear', min_filter='linear', mipmap_filter='linear')
        # `_SHADER` always declares group(2) (the environment), so every mesh pipeline needs a bind
        # group there even with no Environment in the scene (`_environment_resources` then returns
        # the disabled/zeroed block and a harmless 1x1 atlas).
        env_buffer, env_atlas, env_sampler = (
            _environment_resources(state, keep, getattr(scene, 'environments', ())) if vertices else (None, None, None))
        # Y3 of 3, part 1: a single harmless 1x1 texture, shared by every material missing a given PBR
        # map, so every pipeline variant binds something valid at group(0) bindings 7..10 even though
        # the fragment shader's `maps` flags (`_prepare`) keep it from ever sampling the dummy.
        dummy = keep(device.create_texture(size=(1, 1, 1), format='rgba16float',
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
        device.queue.write_texture({'texture': dummy, 'mip_level': 0}, np.zeros((1, 1, 4), 'f2'),
            {'bytes_per_row': 8, 'rows_per_image': 1}, (1, 1, 1))
        dummy_view = dummy.create_view()
        def single_level(array):
            if array is None:
                return dummy_view
            h, w = array.shape[:2]
            t = keep(device.create_texture(size=(w, h, 1), format='rgba16float',
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
            device.queue.write_texture({'texture': t, 'mip_level': 0}, np.ascontiguousarray(array, 'f2'),
                {'bytes_per_row': w*8, 'rows_per_image': h}, (w, h, 1))
            return t.create_view()
        textures = []
        for mips, mr_array, normal_array, occlusion_array, emissive_array in materials:
            _cancel(cancel)
            h, w = mips[0].shape[:2]
            # Complete the rectangular tail too; reference LOD is capped at its last mip.
            mips = list(mips)
            while max(mips[-1].shape[:2]) > 1:
                t = mips[-1]
                if t.shape[0] == 1:
                    t = t[:, :t.shape[1]//2*2].reshape(1, -1, 2, 4).mean(axis=2)
                else:
                    t = t[:t.shape[0]//2*2].reshape(-1, 2, 1, 4).mean(axis=1)
                mips.append(t)
            texture = keep(device.create_texture(size=(w, h, 1), format='rgba16float', mip_level_count=len(mips),
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
            for level, mip in enumerate(mips):
                mh, mw = mip.shape[:2]
                device.queue.write_texture({'texture': texture, 'mip_level': level}, np.ascontiguousarray(mip, 'f2'),
                    {'bytes_per_row': mw*8, 'rows_per_image': mh}, (mw, mh, 1))
            textures.append((texture.create_view(), single_level(mr_array), single_level(normal_array),
                             single_level(occlusion_array), single_level(emissive_array)))
        fmt = 'rgba32float' if data else state['format']
        target = keep(device.create_texture(size=(width, height, 1), format=fmt,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC))
        depth = keep(device.create_texture(size=(width, height, 1), format='depth32float',
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | (wgpu.TextureUsage.TEXTURE_BINDING if has_volumes else 0)))
        passes = []
        used = set()
        # The smoke shadows meshes: the shadowed lights' rays are marched through the volumes' density (gpuvolume).
        smoke = bool(has_volumes and shadow_triangles and volume.shadow_density > 0
                     and any(light.shadows and light.intensity > 0 for light in scene.lights))
        smoke_groups = []      # per mesh pass: an auto layout belongs to its own pipeline
        for phase in (() if not vertices else (2,) if data else (0, 1)):
            _cancel(cancel)
            pipeline = _pipeline(state, data, phase, bvh_data is not None, smoke)
            if smoke:
                from . import gpuvolume
                smoke_groups.append(gpuvolume.mesh_shadow_group(state, pipeline, scene, volume, used, keep))
            groups = [device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {'binding': 0, 'resource': {'buffer': uniform}}, {'binding': 1, 'resource': {'buffer': light_buffer}},
                {'binding': 2, 'resource': texture}, {'binding': 3, 'resource': sampler},
                {'binding': 4, 'resource': {'buffer': shadow_buffer}},
                {'binding': 7, 'resource': mr_tex}, {'binding': 8, 'resource': normal_tex},
                {'binding': 9, 'resource': occlusion_tex}, {'binding': 10, 'resource': emissive_tex},
                {'binding': 11, 'resource': {'buffer': area_light_buffer}},
                {'binding': 12, 'resource': {'buffer': area_sample_buffer}}]
                + bvh_entries) for texture, mr_tex, normal_tex, occlusion_tex, emissive_tex in textures]
            env_group = device.create_bind_group(layout=pipeline.get_bind_group_layout(2), entries=[
                {'binding': 0, 'resource': {'buffer': env_buffer}}, {'binding': 1, 'resource': env_atlas},
                {'binding': 2, 'resource': env_sampler}])
            passes.append((pipeline, groups, env_group))
        particle_pass = None
        if sprites is not None:
            instance_data, texel_data, particle_params = sprites
            pipeline = particle_pipeline(state, target=fmt, data=data)
            group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {'binding': i, 'resource': {'buffer': keep(device.create_buffer_with_data(data=array, usage=usage))}}
                for i, (array, usage) in enumerate(((particle_params, wgpu.BufferUsage.UNIFORM),
                    (instance_data, wgpu.BufferUsage.STORAGE), (texel_data, wgpu.BufferUsage.STORAGE)))])
            particle_pass = (pipeline, group, None)
        volume_pass = None
        if has_volumes:
            from . import gpuvolume
            volume_pass = gpuvolume.prepare(state, scene, camera, width, height, ambient, volume, light_buffer,
                                            len(lights), bool(scene.lights), depth.create_view(), keep, target=fmt, used=used,
                                            shadow_buffer=shadow_buffer, shadow_count=shadow_triangles, shadow_bias=bias)
        target_view, depth_view = target.create_view(), depth.create_view()
        dtype = np.dtype('f4' if fmt == 'rgba32float' else 'f2')
        stride = ((width*4*dtype.itemsize+255)//256)*256
        staging = keep(device.create_buffer(size=stride*max(y1-y0 for y0, y1 in bands),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
        result = np.empty((height, width, 4), 'f4')
        # Replaying the draw list multiplies host recording cost, acceptable only
        # for heavy-shadow scenes. Ordinary renders retain one submission.
        for y0, y1 in bands:
            _cancel(cancel)
            rows = y1-y0
            encoder = device.create_command_encoder()
            if not passes and particle_pass is None:
                # Volumes only: the first pass still has to clear the colour and the depth the volumes read.
                encoder.begin_render_pass(color_attachments=[{'view': target_view, 'resolve_target': None,
                    'clear_value': (0, 0, 0, 0), 'load_op': 'clear', 'store_op': 'store'}],
                    depth_stencil_attachment={'view': depth_view, 'depth_clear_value': 1.0,
                        'depth_load_op': 'clear', 'depth_store_op': 'store'}).end()
            for pass_number, (pipeline, groups, env_group) in enumerate(passes + ([particle_pass] if particle_pass else [])):
                _cancel(cancel)
                rp = encoder.begin_render_pass(color_attachments=[{'view': target_view,
                    'resolve_target': None, 'clear_value': (0, 0, 0, 0),
                    'load_op': 'clear' if pass_number == 0 else 'load', 'store_op': 'store'}],
                    depth_stencil_attachment={'view': depth_view, 'depth_clear_value': 1.0,
                        'depth_load_op': 'clear' if pass_number == 0 else 'load', 'depth_store_op': 'store'})
                rp.set_scissor_rect(0, y0, width, rows)
                rp.set_pipeline(pipeline)
                if smoke_groups and pass_number < len(passes):
                    rp.set_bind_group(1, smoke_groups[pass_number])
                if env_group is not None:
                    rp.set_bind_group(2, env_group)
                if pass_number == len(passes) and particle_pass:
                    # Drawn last, over the meshes: one instanced quad per sprite, far to near.
                    rp.set_bind_group(0, groups)
                    rp.draw(6, len(instance_data), 0, 0)
                else:
                    rp.set_vertex_buffer(0, vertex_buffer)
                    for _, start, material in queue:
                        rp.set_bind_group(0, groups[material])
                        rp.draw(3, 1, start, 0)
                rp.end()
            if volume_pass is not None:
                _cancel(cancel)
                rp = encoder.begin_render_pass(color_attachments=[{'view': target_view, 'resolve_target': None,
                    'clear_value': (0, 0, 0, 0), 'load_op': 'load', 'store_op': 'store'}])
                rp.set_scissor_rect(0, y0, width, rows)
                volume_pass.record(rp)
                rp.end()
            encoder.copy_texture_to_buffer({'texture': target, 'origin': (0, y0, 0)},
                {'buffer': staging, 'bytes_per_row': stride, 'rows_per_image': rows}, (width, rows, 1))
            # A submitted band cannot be interrupted; cancellation bounds the next submission.
            _cancel(cancel)
            device.queue.submit([encoder.finish()])
            staging.map_sync(wgpu.MapMode.READ)
            try:
                raw = np.frombuffer(staging.read_mapped(), dtype, count=rows*stride//dtype.itemsize)
                result[y0:y1] = raw.reshape(rows, stride//dtype.itemsize)[:, :width*4].reshape(rows, width, 4)
            finally:
                staging.unmap()
            _cancel(cancel)
        if not data:
            result += bg*(1-result[..., 3:4])
        return result
    finally:
        for resource in reversed(resources):
            resource.destroy()
