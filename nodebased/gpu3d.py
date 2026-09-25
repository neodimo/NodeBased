"""Optional wgpu rasterizer, adapted from tools/spike_3d_backends.py.

RGBA uses float32 when float32-blendable is available, otherwise half precision.
Data outputs always use float32. Textures are half precision with CPU-generated
mips and a trilinear sampler; explicit per-triangle LOD matches the reference's
rounded area heuristic rather than hardware derivative-based LOD.
"""
from __future__ import annotations

import math
import threading

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
struct Light { position: vec4<f32>, direction: vec4<f32>, colour: vec4<f32> };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var filtering: sampler;
struct Triangle { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32> };
@group(0) @binding(4) var<storage, read> triangles: array<Triangle>;
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
            t > params.shadow.x * 0.01 && (point == 0.0 || t < limit)) {
            return 1.0 - tri.v0.w;
        }
    }
    return 1.0;
}
// Uniform-controlled loops keep compilation independent of scene complexity.
fn visibility(position: vec3<f32>, normal: vec3<f32>, light: Light) -> f32 {
    let origin = position + normal * params.shadow.x;
    var ray = -light.direction.xyz;
    var limit = 0.0;
    if (light.position.w > 0.0) {
        ray = light.position.xyz - origin;
        limit = length(ray);
        ray = ray / max(limit, 1e-8);
    }
    var transmission = 1.0;
    // BVH_TRAVERSAL
    for (var j = 0u; j < u32(params.shadow.y); j += 1u) {
        transmission *= triangle_transmission(j, origin, ray, limit, light.position.w);
    }
    return transmission;
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
};
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) world: vec3<f32>,
             @location(2) normal: vec3<f32>, @location(3) uv: vec2<f32>,
             @location(4) colour: vec4<f32>, @location(5) lod: f32, @location(6) material: vec3<f32>,
             @location(7) object_id: f32) -> Vertex {
    let q = params.projection;
    var v: Vertex;
    // WebGPU depth is 0..w: crossing triangles are clipped, not rejected.
    v.position = vec4<f32>(p.x*q.x, p.y*q.y,
        -q.w/(q.w-q.z)*p.z - q.w*q.z/(q.w-q.z), -p.z);
    v.depth = -p.z; v.world = world; v.normal = normal;
    v.uv = uv; v.colour = colour; v.lod = lod; v.material = material; v.object_id = object_id;
    return v;
}
@fragment fn fs(v: Vertex) -> @location(0) vec4<f32> {
    var source = textureSampleLevel(tex, filtering, vec2<f32>(v.uv.x, 1.0-v.uv.y), v.lod) * v.colour;
    if (source.a <= 0.0) { discard; }
    if (PASS == 0u && source.a < 0.999) { discard; }
    if (PASS == 1u && source.a >= 0.999) { discard; }
    var normal = v.normal / max(length(v.normal), 1e-8);
    if (dot(normal, params.eye.xyz-v.world) < 0.0) { normal = -normal; }
    if (params.settings.z == 1.0) { return vec4<f32>(vec3<f32>(v.depth), 1.0); }
    if (params.settings.z == 2.0) { return vec4<f32>(normal, 1.0); }
    if (params.settings.z == 7.0) { return vec4<f32>(v.world, 1.0); }
    if (params.settings.z == 8.0) { return vec4<f32>(v.uv, 0.0, 1.0); }
    if (params.settings.z == 9.0) { return vec4<f32>(v.object_id, 0.0, 0.0, 1.0); }
    if (params.settings.z == 3.0) { return source; }
    let emission = source.rgb * v.material.z;
    if (params.settings.z == 6.0) { return vec4<f32>(emission, source.a); }
    if (params.settings.y > 0.0) {
        var specular = vec3<f32>(0.0);
        let eye_delta = params.eye.xyz-v.world;
        let to_eye = eye_delta / max(length(eye_delta), 1e-8);
        var radiance = vec3<f32>(params.settings.x);
        for (var i = 0u; i < u32(params.settings.y); i += 1u) {
            var toward = -lights[i].direction.xyz;
            if (lights[i].position.w > 0.0) {
                toward = lights[i].position.xyz-v.world;
                toward = toward/max(length(toward), 1e-8);
            }
            var transmission = 1.0;
            if (params.shadow.y > 0.0 && lights[i].direction.w > 0.0) {
                transmission = visibility(v.world, normal, lights[i]);
            }
            radiance += max(dot(normal, toward), 0.0)*transmission*lights[i].colour.xyz;
            if (v.material.x > 0.0 && dot(normal, toward) > 0.0) {
                let half_delta = toward + to_eye;
                let half_vector = half_delta / max(length(half_delta), 1e-8);
                specular += v.material.x * pow(max(dot(normal, half_vector), 0.0), v.material.y)
                    * transmission * lights[i].colour.xyz;
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
    var near = params.shadow.x * 0.01;
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


def _pipeline(state, data, phase, bvh=False):
    key = (data, phase, bvh)
    if key in state['pipelines']:
        return state['pipelines'][key]
    device = state['device']
    code = _SHADER
    if bvh:
        code = _BVH_DECL + code.replace('// BVH_TRAVERSAL', _BVH_TRAVERSAL)
    module = device.create_shader_module(code=code)
    target = {'format': 'rgba32float' if data else state['format']}
    if not data:
        blend = {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}
        target['blend'] = {'color': blend, 'alpha': blend}
    attributes = [dict(format=f, offset=o, shader_location=i) for i, (f, o) in enumerate(
        [('float32x3', 0), ('float32x3', 12), ('float32x3', 24), ('float32x2', 36), ('float32x4', 44), ('float32', 60), ('float32x3', 64), ('float32', 76)])]
    pipeline = device.create_render_pipeline(layout='auto',
        vertex={'module': module, 'entry_point': 'vs', 'buffers': [
            {'array_stride': 80, 'step_mode': 'vertex', 'attributes': attributes}]},
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
        material = len(materials)
        materials.append(mips)
        tint = np.asarray(geometry.color, 'f4').copy()
        tint[:3] *= tint[3]
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
            attrs = np.concatenate((local[tri], world[tri], ns, uvs[tri]), axis=1).astype('f4')
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
                packed = np.empty((3, 20), 'f4')
                packed[:, :11] = clipped
                # All three vertices agree, regardless of the provoking vertex.
                packed[:, 11:15] = tint
                packed[:, 15] = lod
                packed[:, 16:19] = (geometry.specular, geometry.shininess, geometry.emission)
                packed[:, 19] = object_id
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


def render(scene, camera, width, height, background=(0, 0, 0, 0), ambient=0.0,
           samples=1, output='rgba', cancel=None, adapter=None, *, mode='raster'):
    """Render a read-only premultiplied float32 image; raise on unavailable GPUs.

    Projection and viewport shade rendering are unsupported. Callers can catch
    Unsupported/RuntimeError and use scene3d.render as their fallback.
    """
    if getattr(scene, 'particles', ()):
        raise Unsupported('particle sets are drawn by the CPU renderer only for now')
    if mode == 'raytrace':
        if output == 'splats':
            raise Unsupported('splats output is CPU-only')
        if output == 'relight':
            raise Unsupported('the relight bundle output is CPU-only for now')
        if output not in scene3d.RENDER_OUTPUTS:
            raise ValueError(f'Unknown 3D render output {output!r}')
        if scene.splats:
            if any(i.shadow_catch > 0 for i in scene.splats) and scene.geometries and any(
                    light.shadows and light.intensity > 0 for light in scene.lights):
                raise Unsupported('caught splat shadows are CPU-only')
            if output != 'rgba':
                raise Unsupported('splat data passes are CPU-only')
            if any(i.relight > 0 for i in scene.splats) and any(light.shadows for light in scene.lights):
                raise Unsupported('splat shadows are CPU-only')
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
    if scene.splats:
        if output != 'rgba':
            raise Unsupported('splat data passes and the `splats` output are CPU-only')
        if not scene3d._opaque_meshes(scene):
            raise Unsupported('transparent meshes mixed with splats are CPU-only')
        # Stated on its own so it survives changes to the clause below.
        if any(getattr(i, 'shadow_catch', 0) > 0 for i in scene.splats) and scene.geometries and any(
                light.shadows and light.intensity > 0 for light in scene.lights):
            raise Unsupported('caught splat shadows are CPU-only')
        relit = any(getattr(i, 'relight', 0) > 0 for i in scene.splats)
        # CPU splats also cast shadows onto meshes, even with baked colour.
        if (relit and any(light.shadows for light in scene.lights)) or (
                scene.geometries and any(light.shadows and light.intensity > 0
                                         for light in scene.lights)):
            raise Unsupported('splat shadows are CPU-only')
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
    shadow_count = sum(light.shadows and light.intensity > 0 for light in scene.lights) if output in ('rgba', 'diffuse', 'specular') else 0
    triangles = sum(len(g.triangles) for g in scene.geometries) if shadow_count else 0
    work = width * height * samples ** 2 * shadow_count * triangles
    with _lock:
        state = _state(adapter)
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
        _cancel(cancel)
        result = _render(state, scene, camera, width*samples, height*samples,
                         background, ambient, output, cancel, triangles, shadow_prepared, bvh_data, bands=bands)
        if scene.splats:
            mesh_depth = None
            if scene.geometries:
                depth = _render(state, scene, camera, width*samples, height*samples,
                                (0, 0, 0, 0), ambient, 'depth', cancel)
                # The depth shader interpolates -view.z, already positive
                # camera-forward distance (not radial ray distance).
                mesh_depth = np.where(depth[..., 3] > 0, depth[..., 0], np.inf)
            lighting = ((scene.lights, ambient, None) if any(
                getattr(i, 'relight', 0) > 0 for i in scene.splats) else None)
            splat_rgb, splat_alpha = gpusplat.render_layer(
                state, scene.splats, camera, width*samples, height*samples,
                mesh_depth, lighting=lighting, cancel=cancel)
            result[..., :3] = splat_rgb + (1-splat_alpha[..., None])*result[..., :3]
            result[..., 3] = splat_alpha + (1-splat_alpha)*result[..., 3]
    if samples > 1:
        result = result.reshape(height, samples, width, samples, 4).mean(axis=(1, 3))
    result.flags.writeable = False
    return result


def _render(state, scene, camera, width, height, background, ambient, output, cancel, shadow_triangles=0, shadow_prepared=None, bvh_data=None, *, bands=None):
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
    if not vertices:
        return np.broadcast_to(bg, (height, width, 4)).copy()
    resources = []
    def keep(resource):
        resources.append(resource)
        return resource
    try:
        lights = [(light, *light.world()) for light in scene.lights if light.intensity > 0]
        light_data = np.zeros((max(1, len(lights)), 12), 'f4')
        for i, (light, position, direction) in enumerate(lights):
            light_data[i, :3] = position
            light_data[i, 3] = light.kind == 'Point'
            light_data[i, 4:7] = direction
            light_data[i, 7] = light.shadows
            light_data[i, 8:11] = np.asarray(light.color)*light.intensity
        params = np.array([focal/(width/height), focal, camera.near, camera.far,
                           *eye, 0, ambient, len(lights), scene3d.RENDER_OUTPUTS.index(output), 0,
                           bias, shadow_triangles, bvh_data is not None, 0], 'f4')
        uniform = keep(device.create_buffer_with_data(data=params, usage=wgpu.BufferUsage.UNIFORM))
        light_buffer = keep(device.create_buffer_with_data(data=light_data, usage=wgpu.BufferUsage.STORAGE))
        shadow_buffer = keep(device.create_buffer_with_data(data=shadow_data, usage=wgpu.BufferUsage.STORAGE))
        bvh_entries = []
        if bvh_data is not None:
            for binding, array in zip((5, 6), bvh_data):
                buffer = keep(device.create_buffer_with_data(data=array, usage=wgpu.BufferUsage.STORAGE))
                bvh_entries.append({'binding': binding, 'resource': {'buffer': buffer}})
        vertex_buffer = keep(device.create_buffer_with_data(data=np.concatenate(vertices), usage=wgpu.BufferUsage.VERTEX))
        sampler = device.create_sampler(mag_filter='linear', min_filter='linear', mipmap_filter='linear')
        textures = []
        for mips in materials:
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
            textures.append(texture.create_view())
        fmt = 'rgba32float' if data else state['format']
        target = keep(device.create_texture(size=(width, height, 1), format=fmt,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC))
        depth = keep(device.create_texture(size=(width, height, 1), format='depth32float', usage=wgpu.TextureUsage.RENDER_ATTACHMENT))
        passes = []
        for phase in ((2,) if data else (0, 1)):
            _cancel(cancel)
            pipeline = _pipeline(state, data, phase, bvh_data is not None)
            groups = [device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                {'binding': 0, 'resource': {'buffer': uniform}}, {'binding': 1, 'resource': {'buffer': light_buffer}},
                {'binding': 2, 'resource': texture}, {'binding': 3, 'resource': sampler},
                {'binding': 4, 'resource': {'buffer': shadow_buffer}}] + bvh_entries) for texture in textures]
            passes.append((pipeline, groups))
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
            for pass_number, (pipeline, groups) in enumerate(passes):
                _cancel(cancel)
                rp = encoder.begin_render_pass(color_attachments=[{'view': target_view,
                    'resolve_target': None, 'clear_value': (0, 0, 0, 0),
                    'load_op': 'clear' if pass_number == 0 else 'load', 'store_op': 'store'}],
                    depth_stencil_attachment={'view': depth_view, 'depth_clear_value': 1.0,
                        'depth_load_op': 'clear' if pass_number == 0 else 'load', 'depth_store_op': 'store'})
                rp.set_scissor_rect(0, y0, width, rows)
                rp.set_pipeline(pipeline)
                rp.set_vertex_buffer(0, vertex_buffer)
                for _, start, material in queue:
                    rp.set_bind_group(0, groups[material])
                    rp.draw(3, 1, start, 0)
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
