"""Standalone EWA GPU layer. CPU stable depth sort, GPU projection and over.

Unlike the reference, blending does not stop at transmittance 1e-4. For bounded
unit colours that adds at most 1e-4; float projection/blending (especially f16)
needs additional tolerance. The 2e-3 max / 2e-4 mean parity target is verified
for rgba32float; rgba16float accumulation can exceed it (about 5e-3 max
on the 2,000-splat regression scene). Cache ownership is per device state, serialized by
GPU3D's lock, and limited to the instances in the most recent nonempty call.
"""
from dataclasses import replace
from time import perf_counter
import numpy as np
from . import gpu3d, splatshade, splatraster
from .scene3d import SplatInstance, _view_basis
from .cancellation import Cancelled

GPU_SPLAT_MEMORY_CAP = 2 * 1024**3
last_timings = {}


def estimate_bytes(n_splats):
    """Conservative splat GPU storage: cached + combined geometry, output/order/RGB."""
    return int(n_splats) * (64 + 64 + 80 + 4 + 16)


def check_capability(state):
    limits = state.get('limits', getattr(state.get('device'), 'limits', {}))
    for name, minimum in [('max-storage-buffers-per-shader-stage', 3),
                          ('max-storage-buffer-binding-size', 80),
                          ('max-buffer-size', 80),
                          ('max-compute-invocations-per-workgroup', 64),
                          ('max-compute-workgroup-size-x', 64),
                          ('max-compute-workgroups-per-dimension', 1)]:
        if limits.get(name, 0) < minimum:
            return f'GPU splats unavailable: {name} too small (needs {minimum})'
    # Pipeline creation also probes vertex-stage storage on downlevel backends.
    if 'device' in state and 'wgpu' in state:
        try:
            _pipelines(state)
        except Exception as exc:
            return f'GPU splats require vertex-stage storage buffers and blending: {exc}'
    return None


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _create_static_buffer(state, data):
    return state['device'].create_buffer_with_data(data=data, usage=
        state['wgpu'].BufferUsage.STORAGE | state['wgpu'].BufferUsage.COPY_SRC)


_SHADER = r'''
struct Params { right: vec4<f32>, up: vec4<f32>, forward: vec4<f32>,
 eye: vec4<f32>, frame: vec4<f32>, settings: vec4<f32> };
struct Geometry { pos: vec4<f32>, cov0: vec4<f32>, cov1: vec4<f32>, normal: vec4<f32> };
struct Projected { centre: vec4<f32>, conic: vec4<f32>, bounds: vec4<f32>,
 normal: vec4<f32>, extra: vec4<f32> };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> geom: array<Geometry>;
@group(0) @binding(2) var<storage, read> colours: array<f32>;
@group(0) @binding(3) var<storage, read_write> projected: array<Projected>;
fn view(v: vec3<f32>) -> vec3<f32> {
 return vec3<f32>(dot(p.right.xyz,v), dot(p.up.xyz,v), dot(p.forward.xyz,v));
}
fn colour(k: u32, position: vec3<f32>) -> vec3<f32> {
 let degree = u32(p.right.w);
 if (degree == 0u) {
   return vec3<f32>(colours[3u*k], colours[3u*k+1u], colours[3u*k+2u]);
 }
 let delta = position-p.eye.xyz;
 let d = delta/max(length(delta),1e-30);
 let x=d.x; let y=d.y; let z=d.z;
 var b: array<f32,16>;
 b[0]=0.28209479177387814;
 b[1]=-0.4886025119029199*y; b[2]=0.4886025119029199*z; b[3]=-0.4886025119029199*x;
 b[4]=1.0925484305920792*x*y; b[5]=-1.0925484305920792*y*z;
 b[6]=0.31539156525252005*(2.0*z*z-x*x-y*y);
 b[7]=-1.0925484305920792*x*z; b[8]=0.5462742152960396*(x*x-y*y);
 b[9]=-0.5900435899266435*y*(3.0*x*x-y*y);
 b[10]=2.890611442640554*x*y*z;
 b[11]=-0.4570457994644658*y*(4.0*z*z-x*x-y*y);
 b[12]=0.3731763325901154*z*(2.0*z*z-3.0*x*x-3.0*y*y);
 b[13]=-0.4570457994644658*x*(4.0*z*z-x*x-y*y);
 b[14]=1.445305721320277*z*(x*x-y*y);
 b[15]=-0.5900435899266435*x*(x*x-3.0*y*y);
 let count=(degree+1u)*(degree+1u);
 var rgb=vec3<f32>(0.0);
 for (var j=0u; j<count; j+=1u) {
   let base=3u*(k*count+j);
   rgb+=b[j]*vec3<f32>(colours[base],colours[base+1u],colours[base+2u]);
 }
 rgb=max(rgb+vec3<f32>(0.5),vec3<f32>(0.0));
 if (p.up.w != 0.0) {
   rgb=select(pow((rgb+vec3<f32>(0.055))/1.055,vec3<f32>(2.4)),
              rgb/12.92,rgb<=vec3<f32>(0.04045));
 }
 return rgb;
}
@compute @workgroup_size(64) fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
 let k = gid.x + gid.y * u32(p.settings.w) * 64u;
 if (k >= u32(p.settings.z)) { return; }
 let i = k + u32(p.eye.w); let g = geom[i]; let v = view(g.pos.xyz-p.eye.xyz);
 let z = v.z; let f = p.frame.zw;
 let centre = p.frame.xy*0.5 + vec2<f32>(f.x*v.x/z, -f.y*v.y/z);
 let tx = clamp(v.x/z, -1.3*p.settings.x, 1.3*p.settings.x);
 let ty = clamp(v.y/z, -1.3*p.settings.y, 1.3*p.settings.y);
 // J*V, avoiding a temporary view-space covariance.
 let jx = (f.x/z)*(p.right.xyz-tx*p.forward.xyz);
 let jy = (-f.y/z)*(p.up.xyz-ty*p.forward.xyz);
 let cov = mat3x3<f32>(vec3<f32>(g.cov0.x,g.cov0.y,g.cov0.z),
   vec3<f32>(g.cov0.y,g.cov0.w,g.cov1.x), vec3<f32>(g.cov0.z,g.cov1.x,g.cov1.y));
 let a = dot(jx,cov*jx)+0.3; let b = dot(jx,cov*jy); let c = dot(jy,cov*jy)+0.3;
 let radius = 3.0*sqrt(0.5*(a+c+sqrt((a-c)*(a-c)+4.0*b*b)));
 let lo = floor(clamp(centre-vec2<f32>(radius),vec2<f32>(0.0),p.frame.xy));
 let hi = ceil(clamp(centre+vec2<f32>(radius),vec2<f32>(0.0),p.frame.xy));
 let n = view(g.normal.xyz);
 projected[i] = Projected(vec4<f32>(centre,z,g.pos.w),
   vec4<f32>(vec3<f32>(c,-b,a)/(a*c-b*b),g.cov1.w),
   vec4<f32>(lo,hi),vec4<f32>(n,dot(n,v)),vec4<f32>(g.cov1.z,colour(k,g.pos.xyz)));
}
'''
_RENDER = r'''
struct Params { right: vec4<f32>, up: vec4<f32>, forward: vec4<f32>,
 eye: vec4<f32>, frame: vec4<f32>, settings: vec4<f32> };
struct Projected { centre: vec4<f32>, conic: vec4<f32>, bounds: vec4<f32>,
 normal: vec4<f32>, extra: vec4<f32> };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var mesh: texture_2d<f32>;
struct Vertex { @builtin(position) position: vec4<f32>,
 @location(0) @interpolate(flat) index: u32 };
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) k: u32) -> Vertex {
 let i = order[k]; let s = projected[i];
 let corners = array<vec2<f32>,6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),
   vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1));
 var xy = mix(s.bounds.xy,s.bounds.zw,corners[v]);
 if (any(s.bounds.zw <= s.bounds.xy)) { xy = vec2<f32>(0.0); }
 return Vertex(vec4<f32>(xy.x/p.frame.x*2.0-1.0,1.0-xy.y/p.frame.y*2.0,0.0,1.0),i);
}
@fragment fn fs(v: Vertex) -> @location(0) vec4<f32> {
 let s = projected[v.index]; let d = v.position.xy-s.centre.xy;
 let q = s.conic.x*d.x*d.x+2.0*s.conic.y*d.x*d.y+s.conic.z*d.y*d.y;
 let alpha = min(0.99,s.centre.w*exp(-0.5*q));
 if (alpha < 1.0/255.0) { discard; }
 let ray = vec3<f32>((v.position.x-p.frame.x*0.5)/p.frame.z,
   (p.frame.y*0.5-v.position.y)/p.frame.w,1.0);
 let den = dot(ray,s.normal.xyz); var zp = s.centre.z;
 if (den != 0.0) { zp = s.normal.w/den; }
 if (s.conic.w != 0.0 || abs(den)/length(ray) < 0.05 || abs(zp-s.centre.z) > 3.0*s.extra.x) {
   zp = s.centre.z;
 }
 if (zp >= textureLoad(mesh,vec2<i32>(v.position.xy),0).x) { discard; }
 return vec4<f32>(s.extra.yzw*alpha,alpha);
}
'''


def _pipelines(state):
    key = '_gpusplat_pipelines'
    if key not in state:
        device = state['device']
        compute = device.create_compute_pipeline(layout='auto', compute={
            'module': device.create_shader_module(code=_SHADER), 'entry_point': 'project'})
        module = device.create_shader_module(code=_RENDER)
        blend = dict(src_factor='one', dst_factor='one-minus-src-alpha', operation='add')
        render = device.create_render_pipeline(layout='auto', vertex=dict(module=module, entry_point='vs', buffers=[]),
            primitive=dict(topology='triangle-list', cull_mode='none'),
            fragment=dict(module=module, entry_point='fs', targets=[dict(format=state['format'],
                blend=dict(color=blend, alpha=blend))]))
        state[key] = compute, render
    return state[key]


def _candidates(world, instance, eye, basis, camera, width, height):
    """Indices of the splats the CPU rasterizer would give shadow rays: in front of the camera,
    opaque enough to draw, and (conservatively, by a 3-sigma sphere) touching the frame."""
    local = (world.positions.astype('f8') - eye) @ basis.T
    z = local[:, 2]
    keep = (z > camera.near) & (z >= splatraster.SPLAT_MIN_VIEW_DEPTH) & (z < camera.far)
    keep &= np.minimum(.99, np.clip(world.opacity * instance.opacity_scale, 0, 1)) >= 1/255
    tan = np.tan(np.deg2rad(camera.fov)/2)
    focal = height/(2*tan)
    radius = 3*world.scales.max(axis=1)*abs(instance.scale_scale)*focal/np.maximum(z, 1e-6) + 2
    x = width/2 + focal*local[:, 0]/np.maximum(z, 1e-6)
    y = height/2 - focal*local[:, 1]/np.maximum(z, 1e-6)
    keep &= (x+radius > 0) & (x-radius < width) & (y+radius > 0) & (y-radius < height)
    return np.flatnonzero(keep)


def _geometry_and_order(state, instances, camera, cancel):
    """Static per-instance geometry buffers (cached) and the back-to-front order of the splats the CPU
    rasterizer would project: shared by the beauty layer and the data passes."""
    keys = [(id(i.cloud), np.asarray(i.matrix, dtype='f8').tobytes(), float(i.scale_scale),
             float(i.opacity_scale)) for i in instances]
    cache = state.setdefault('_gpusplat_geometry', {})
    for key in list(cache):
        if key not in keys:
            cache.pop(key)[2].destroy()
    entries = []
    static_upload_ms = 0.0
    for key, instance in zip(keys, instances):
        _check(cancel)
        if not len(instance.cloud):
            continue
        if key not in cache:
            geometry = splatshade.instance_geometry(instance)
            cloud = geometry['cloud']
            cov = cloud.covariance()*instance.scale_scale**2
            packed = np.zeros((len(cloud),16),'f4')
            packed[:,:3], packed[:,3] = cloud.positions, geometry['opacity']
            packed[:,4:8] = cov[:,(0,0,0,1),(0,1,2,1)]
            packed[:,8:10] = cov[:,(1,2),(2,2)]
            packed[:,10] = cloud.scales.max(axis=1)*instance.scale_scale
            scales = np.sort(cloud.scales,axis=1)
            packed[:,11] = scales[:,0] > .8*scales[:,1]
            packed[:,12:15] = cloud.normals()
            _check(cancel)
            start = perf_counter()
            geometry_buffer = _create_static_buffer(state,packed)
            static_upload_ms += (perf_counter()-start)*1000
            cache[key] = (instance.cloud, cloud.positions, geometry_buffer, cloud)
        entries.append(cache[key])
    t = perf_counter()
    eye, view = _view_basis(camera)
    basis = view.astype('f8').copy(); basis[2] *= -1
    depth_parts = [(entry[1].astype('f8')-eye) @ basis[2] for entry in entries]
    depths = depth_parts[0] if len(depth_parts) == 1 else np.concatenate(depth_parts)
    # Match CPU exactly: near is strict, the fixed 0.2 guard is inclusive.
    valid = np.flatnonzero((depths > camera.near) & (depths >= splatraster.SPLAT_MIN_VIEW_DEPTH) & (depths < camera.far))
    order = valid[np.argsort(depths[valid],kind='stable')[::-1]].astype('u4')
    last_timings['depth_sort_ms'] = (perf_counter()-t)*1000
    return keys, entries, static_upload_ms, eye, basis, order


def render_layer(state, instances, camera, width, height, mesh_depth=None, *,
                 lighting=None, cancel=None, budget_bytes=None):
    """Return float32 premultiplied RGB and alpha; mesh depth is positive view Z.

    Lighting visibility is a per-instance list. Static geometry is cached using
    immutable cloud identity, matrix bytes, scale and opacity multipliers.
    last_timings reports host wall milliseconds: depth_sort_ms, colour_ms,
    upload_ms (buffer/texture setup), gpu_ms (encoding, execution and readback).
    """
    with gpu3d._lock:
        return _render(state, instances, camera, int(width), int(height), mesh_depth,
                       lighting, cancel, budget_bytes)


def _render(state, instances, camera, width, height, mesh_depth, lighting, cancel, budget_bytes):
    _check(cancel)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    if mesh_depth is not None and np.shape(mesh_depth) != (height, width):
        raise ValueError('mesh_depth must have shape (height, width)')
    instances = [i if hasattr(i, 'cloud') else SplatInstance(*i) for i in instances]
    n = sum(len(i.cloud) for i in instances)
    device, wgpu = state['device'], state['wgpu']
    limits = device.limits
    cap = min(GPU_SPLAT_MEMORY_CAP, limits['max-buffer-size'], limits['max-storage-buffer-binding-size'])
    if budget_bytes is not None:
        cap = min(cap, budget_bytes)
    needed = estimate_bytes(n) + sum(len(i.cloud)*12*((i.cloud.sh_degree if i.sh_degree is None
        else max(0, min(int(i.sh_degree), i.cloud.sh_degree)))+1)**2
        for i in instances if i.relight <= 0)
    if needed > cap:
        raise ValueError(f'GPU splat render needs about {needed/1024**2:.1f} MiB for {n:,} splats, '
                         f'more than the adapter allows ({cap/1024**2:.1f} MiB): lower '
                         'the splat count, or use the CPU renderer')
    zeros = lambda: (np.zeros((height,width,3),'f4'), np.zeros((height,width),'f4'))
    if not n:
        return zeros()
    reason = check_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    if max(width,height) > limits['max-texture-dimension-2d']:
        raise ValueError('GPU splat render dimensions exceed adapter texture limits')
    keys, entries, static_upload_ms, eye, basis, order = _geometry_and_order(state, instances, camera, cancel)
    _check(cancel)
    if not len(order):
        return zeros()
    t = perf_counter()
    colour_cache = state.setdefault('_gpusplat_colours', {})
    colour_keys = [(*key, i.sh_degree, float(i.relight), i.cloud.colorspace)
                   for key, i in zip(keys, instances)]
    for key in list(colour_cache):
        if key not in colour_keys:
            colour_cache.pop(key)[1].destroy()
    appearances = []
    upload_ms = 0.0
    geometry_index = 0
    for index, (instance, key) in enumerate(zip(instances, colour_keys)):
        _check(cancel)
        size = len(instance.cloud)
        if not size:
            continue
        world = entries[geometry_index][3]
        geometry_index += 1
        degree = world.sh_degree if instance.sh_degree is None else max(0, min(int(instance.sh_degree), world.sh_degree))
        source = lighting[2] if lighting is not None and len(lighting) > 2 else None
        catching = (source is not None and hasattr(source, 'catch_for_indices') and instance.relight < 1
                    and float(getattr(instance, 'shadow_catch', 0.0)) > 0)
        dynamic = instance.relight > 0 or catching
        if dynamic or key not in colour_cache:
            if not dynamic and degree > 0:
                data = np.ascontiguousarray(world.sh[:, :(degree+1)**2], dtype='f4')
            else:
                lights, ambient, visibility, catch, extras = (), 0.0, None, None, None
                evaluated = instance
                if lighting is None or instance.relight <= 0:
                    evaluated = replace(instance, relight=0)
                if lighting is not None:
                    lights, ambient = lighting[:2]
                    extras = lighting[3] if len(lighting) > 3 else None
                if catching:
                    candidates = _candidates(world, instance, eye, basis, camera, width, height)
                    mesh_visibility = np.ones((size, len(lights)))
                    mesh_visibility[candidates] = source.catch_for_indices(index, candidates)
                    catch = splatshade.shadow_catch(lights, ambient, mesh_visibility, instance.shadow_catch)
                if lighting is not None and instance.relight > 0 and source is not None:
                    if hasattr(source, 'for_indices'):
                        candidates = _candidates(world, instance, eye, basis, camera, width, height)
                        visibility = np.ones((size, len(lights)))
                        visibility[candidates] = source.for_indices(index, candidates)
                    else:
                        visibility = source[index]
                data = splatshade.instance_colors(evaluated,eye,lights,ambient,visibility,catch,extras)
            start = perf_counter()
            uploaded = data if dynamic else _create_static_buffer(state, data)
            upload_ms += (perf_counter()-start)*1000
            if not dynamic:
                colour_cache[key] = (instance.cloud, uploaded)
        else:
            uploaded = colour_cache[key][1]
        appearances.append((uploaded, 0 if dynamic else degree, world.colorspace == 'srgb', size, dynamic))
    last_timings['colour_ms'] = (perf_counter()-t)*1000-upload_ms
    t = perf_counter(); resources = []
    def keep(resource):
        resources.append(resource); return resource
    def buffer(data, usage):
        return keep(device.create_buffer_with_data(data=data, usage=usage))
    try:
        storage = wgpu.BufferUsage.STORAGE
        combined = keep(device.create_buffer(size=n*64, usage=storage | wgpu.BufferUsage.COPY_DST))
        output = keep(device.create_buffer(size=n*80, usage=storage))
        indices = buffer(order,storage)
        groups = (len(order)+63)//64
        gx = min(groups,limits['max-compute-workgroups-per-dimension'])
        gy = (groups+gx-1)//gx
        if gy > limits['max-compute-workgroups-per-dimension']:
            raise gpu3d.Unsupported('GPU splat dispatch exceeds adapter limits')
        tan = np.tan(np.deg2rad(camera.fov)/2); focal = height/(2*tan)
        params = np.zeros((6,4),'f4'); params[:3,:3] = basis; params[3,:3] = eye
        params[4] = width,height,focal,focal
        params[5] = tan*width/height,tan,len(order),gx
        uniform = buffer(params,wgpu.BufferUsage.UNIFORM)
        mesh = keep(device.create_texture(size=(width,height,1),format='r32float',
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
        depth = np.full((height,width),np.inf,'f4') if mesh_depth is None else np.ascontiguousarray(mesh_depth,'f4')
        device.queue.write_texture({'texture':mesh},depth,{'bytes_per_row':width*4,'rows_per_image':height},(width,height,1))
        target = keep(device.create_texture(size=(width,height,1),format=state['format'],
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC))
        compute, render = _pipelines(state)
        def bind(pipeline, buffers, texture=None):
            entries = [dict(binding=i,resource={'buffer':b}) for i,b in enumerate(buffers)]
            if texture is not None:
                entries.append(dict(binding=len(buffers),resource=texture.create_view()))
            return device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=entries)
        compute_groups = []
        base = 0
        for appearance, degree, srgb, size, dynamic in appearances:
            if dynamic:
                appearance = buffer(appearance,storage)
            count = (size+63)//64
            sx = min(count,limits['max-compute-workgroups-per-dimension'])
            sy = (count+sx-1)//sx
            if sy > limits['max-compute-workgroups-per-dimension']:
                raise gpu3d.Unsupported('GPU splat dispatch exceeds adapter limits')
            local = params.copy()
            local[0,3], local[1,3], local[3,3] = degree, srgb, base
            local[5,2:] = size, sx
            ub = buffer(local,wgpu.BufferUsage.UNIFORM)
            compute_groups.append((bind(compute,[ub,combined,appearance,output]),sx,sy))
            base += size
        rg = bind(render,[uniform,output,indices],mesh)
        last_timings['upload_ms'] = static_upload_ms + upload_ms + (perf_counter()-t)*1000
        t = perf_counter()
        encoder = device.create_command_encoder(); offset = 0
        for entry in entries:
            size = len(entry[1])*64
            encoder.copy_buffer_to_buffer(entry[2],0,combined,offset,size); offset += size
        cp = encoder.begin_compute_pass(); cp.set_pipeline(compute)
        for cg, sx, sy in compute_groups:
            cp.set_bind_group(0,cg)
            cp.dispatch_workgroups(sx,sy,1)
        cp.end()
        rp = encoder.begin_render_pass(color_attachments=[dict(view=target.create_view(),resolve_target=None,
            clear_value=(0,0,0,0),load_op='clear',store_op='store')])
        rp.set_pipeline(render); rp.set_bind_group(0,rg); rp.draw(6,len(order),0,0); rp.end()
        dtype = np.dtype('f4' if state['format'] == 'rgba32float' else 'f2')
        stride = ((width*4*dtype.itemsize+255)//256)*256
        staging = keep(device.create_buffer(size=stride*height,usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
        encoder.copy_texture_to_buffer({'texture':target},{'buffer':staging,'bytes_per_row':stride,'rows_per_image':height},(width,height,1))
        _check(cancel); device.queue.submit([encoder.finish()]); staging.map_sync(wgpu.MapMode.READ)
        try:
            raw = np.frombuffer(staging.read_mapped(),dtype).reshape(height,stride//dtype.itemsize)
            result = raw[:,:width*4].reshape(height,width,4).astype('f4',copy=True)
        finally:
            staging.unmap()
        _check(cancel)
        last_timings['gpu_ms'] = (perf_counter()-t)*1000
        return result[...,:3].copy(), result[...,3].copy()
    finally:
        for resource in reversed(resources):
            resource.destroy()


# ---------------------------------------------------------------------------------------------------------
# Data passes (depth, position, object_id): the first splat, per pixel and in plane-depth order, where the
# accumulated opacity reaches SPLAT_AOV_OPACITY, behind-the-mesh splats dropped. This is
# `splatraster.accumulate_splats`' data branch run per pixel on the GPU: the projection pass above, a
# tile binning pass (16x16 tiles, count then fill, with a host prefix sum between them), and a resolve pass
# that keeps the nearest few fragments in registers and walks them in (plane depth, authored index) order.
# Integers are stored as float values, never bitcast.
# ---------------------------------------------------------------------------------------------------------
DATA_OUTPUTS = ('depth', 'position', 'object_id')
_NEAREST = 12     # fragments held per pass of the resolve; more than that are found by further passes

_BIN = r'''
struct Dims { frame: vec4<f32>, tiles: vec4<u32> };
struct Projected { centre: vec4<f32>, conic: vec4<f32>, bounds: vec4<f32>,
 normal: vec4<f32>, extra: vec4<f32> };
@group(0) @binding(0) var<uniform> q: Dims;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> lists: array<u32>;
fn tile_box(i: u32) -> vec4<i32> {
 let b = projected[i].bounds;
 if (any(b.zw <= b.xy)) { return vec4<i32>(0, 0, -1, -1); }
 let lo = vec2<i32>(floor(b.xy/16.0));
 let hi = vec2<i32>(floor((b.zw-vec2<f32>(1.0))/16.0));
 return vec4<i32>(lo, min(hi, vec2<i32>(i32(q.tiles.x)-1, i32(q.tiles.y)-1)));
}
@compute @workgroup_size(64) fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
 let k = gid.x + gid.y * q.tiles.w * 64u;
 if (k >= q.tiles.z) { return; }
 let t = tile_box(order[k]);
 for (var ty = t.y; ty <= t.w; ty += 1) {
   for (var tx = t.x; tx <= t.z; tx += 1) {
     atomicAdd(&cursor[u32(ty)*q.tiles.x + u32(tx)], 1u);
   }
 }
}
@compute @workgroup_size(64) fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
 let k = gid.x + gid.y * q.tiles.w * 64u;
 if (k >= q.tiles.z) { return; }
 let i = order[k];
 let t = tile_box(i);
 for (var ty = t.y; ty <= t.w; ty += 1) {
   for (var tx = t.x; tx <= t.z; tx += 1) {
     let tile = u32(ty)*q.tiles.x + u32(tx);
     let slot = atomicAdd(&cursor[tile], 1u);
     lists[offsets[tile] + slot] = i;
   }
 }
}
'''
_RESOLVE = r'''
struct Dims { frame: vec4<f32>, tiles: vec4<u32> };
struct Projected { centre: vec4<f32>, conic: vec4<f32>, bounds: vec4<f32>,
 normal: vec4<f32>, extra: vec4<f32> };
@group(0) @binding(0) var<uniform> q: Dims;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> lists: array<u32>;
@group(0) @binding(4) var<storage, read> ids: array<f32>;
@group(0) @binding(5) var mesh: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> result: array<vec4<f32>>;
const NEAREST = 12u;
const THRESHOLD = 0.5;
@compute @workgroup_size(8, 8) fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
 let width = u32(q.frame.x); let height = u32(q.frame.y);
 if (gid.x >= width || gid.y >= height) { return; }
 let pixel = vec2<f32>(f32(gid.x)+0.5, f32(gid.y)+0.5);
 let ray = vec3<f32>((pixel.x-q.frame.x*0.5)/q.frame.z, (q.frame.y*0.5-pixel.y)/q.frame.w, 1.0);
 let ray_length = length(ray);
 let tile = (gid.y/16u)*q.tiles.x + gid.x/16u;
 let first = offsets[tile]; let last = offsets[tile+1u];
 let mesh_depth = textureLoad(mesh, vec2<i32>(gid.xy), 0).x;
 var kz: array<f32, 12>; var ki: array<u32, 12>; var ka: array<f32, 12>;
 var resumed = false; var rz = 0.0; var ri = 0u;
 var through = 1.0; var hit = false; var hit_z = 0.0; var hit_i = 0u;
 for (var round = 0u; round < 4096u; round += 1u) {
   var n = 0u;
   for (var e = first; e < last; e += 1u) {
     let i = lists[e]; let s = projected[i];
     let d = pixel - s.centre.xy;
     let qq = s.conic.x*d.x*d.x + 2.0*s.conic.y*d.x*d.y + s.conic.z*d.y*d.y;
     let alpha = min(0.99, s.centre.w*exp(-0.5*qq));
     if (alpha < 1.0/255.0) { continue; }
     if (pixel.x < s.bounds.x || pixel.y < s.bounds.y || pixel.x >= s.bounds.z || pixel.y >= s.bounds.w) { continue; }
     let den = dot(ray, s.normal.xyz); var zp = s.centre.z;
     if (den != 0.0) { zp = s.normal.w/den; }
     if (s.conic.w != 0.0 || abs(den)/ray_length < 0.05 || abs(zp-s.centre.z) > 3.0*s.extra.x) {
       zp = s.centre.z;
     }
     if (zp >= mesh_depth) { continue; }
     if (resumed && !(zp > rz || (zp == rz && i > ri))) { continue; }
     if (n == NEAREST && !(zp < kz[NEAREST-1u] || (zp == kz[NEAREST-1u] && i < ki[NEAREST-1u]))) { continue; }
     var at = min(n, NEAREST-1u);
     while (at > 0u && (kz[at-1u] > zp || (kz[at-1u] == zp && ki[at-1u] > i))) {
       kz[at] = kz[at-1u]; ki[at] = ki[at-1u]; ka[at] = ka[at-1u]; at -= 1u;
     }
     kz[at] = zp; ki[at] = i; ka[at] = alpha;
     if (n < NEAREST) { n += 1u; }
   }
   for (var j = 0u; j < n; j += 1u) {
     through = through*(1.0-ka[j]);
     if (1.0-through >= THRESHOLD) { hit = true; hit_z = kz[j]; hit_i = ki[j]; break; }
   }
   if (hit || n < NEAREST) { break; }
   resumed = true; rz = kz[NEAREST-1u]; ri = ki[NEAREST-1u];
 }
 var out = vec4<f32>(0.0);
 if (hit) { out = vec4<f32>(1.0, hit_z, ids[hit_i], 0.0); }
 result[gid.y*width + gid.x] = out;
}
'''


def _data_pipelines(state):
    key = '_gpusplat_data_pipelines'
    if key not in state:
        device = state['device']
        binning = device.create_shader_module(code=_BIN)
        count = device.create_compute_pipeline(layout='auto', compute={'module': binning, 'entry_point': 'count'})
        fill = device.create_compute_pipeline(layout='auto', compute={'module': binning, 'entry_point': 'fill'})
        resolve = device.create_compute_pipeline(layout='auto', compute={
            'module': device.create_shader_module(code=_RESOLVE), 'entry_point': 'resolve'})
        state[key] = count, fill, resolve
    return state[key]


def check_data_capability(state):
    reason = check_capability(state)
    if reason is not None:
        return reason
    limits = state.get('limits', getattr(state.get('device'), 'limits', {}))
    if limits.get('max-storage-buffers-per-shader-stage', 0) < 6:
        return 'GPU splat data passes unavailable: max-storage-buffers-per-shader-stage too small (needs 6)'
    if 'device' in state and 'wgpu' in state:
        try:
            _data_pipelines(state)
        except Exception as exc:
            return f'GPU splat data passes unavailable: {exc}'
    return None


def render_data(state, instances, camera, width, height, mesh_depth=None, *, object_id_offset=0,
                cancel=None, budget_bytes=None):
    """First-hit data for splats: `(hit, depth, object_id)`, arrays of shape (height, width) (bool, float32,
    float32). `hit` is where accumulated splat opacity reaches one half in front of the opaque mesh depth
    (`mesh_depth`, positive view Z, inf where empty); `depth` is that splat's plane depth along the view
    axis, `object_id` its 1-based scene id (`object_id_offset` + 1 + the instance's position)."""
    with gpu3d._lock:
        return _render_data(state, instances, camera, int(width), int(height), mesh_depth,
                            int(object_id_offset), cancel, budget_bytes)


def _render_data(state, instances, camera, width, height, mesh_depth, object_id_offset, cancel, budget_bytes):
    _check(cancel)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    if mesh_depth is not None and np.shape(mesh_depth) != (height, width):
        raise ValueError('mesh_depth must have shape (height, width)')
    instances = [i if hasattr(i, 'cloud') else SplatInstance(*i) for i in instances]
    n = sum(len(i.cloud) for i in instances)
    device, wgpu = state['device'], state['wgpu']
    limits = device.limits
    cap = min(GPU_SPLAT_MEMORY_CAP, limits['max-buffer-size'], limits['max-storage-buffer-binding-size'])
    if budget_bytes is not None:
        cap = min(cap, budget_bytes)
    needed = estimate_bytes(n)
    if needed > cap:
        raise ValueError(f'GPU splat render needs about {needed/1024**2:.1f} MiB for {n:,} splats, '
                         f'more than the adapter allows ({cap/1024**2:.1f} MiB): lower '
                         'the splat count, or use the CPU renderer')
    empty = (np.zeros((height, width), bool), np.zeros((height, width), 'f4'), np.zeros((height, width), 'f4'))
    if not n:
        return empty
    reason = check_data_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    if max(width, height) > limits['max-texture-dimension-2d']:
        raise ValueError('GPU splat render dimensions exceed adapter texture limits')
    keys, entries, static_upload_ms, eye, basis, order = _geometry_and_order(state, instances, camera, cancel)
    _check(cancel)
    if not len(order):
        return empty
    object_ids = np.repeat(np.array([object_id_offset+1+k for k in range(len(instances))], 'f4'),
                           [len(i.cloud) for i in instances])
    nx, ny = (width+15)//16, (height+15)//16
    t = perf_counter(); resources = []
    def keep(resource):
        resources.append(resource); return resource
    def buffer(data, usage):
        return keep(device.create_buffer_with_data(data=data, usage=usage))
    def read(size):
        return keep(device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
    try:
        storage = wgpu.BufferUsage.STORAGE
        combined = keep(device.create_buffer(size=n*64, usage=storage | wgpu.BufferUsage.COPY_DST))
        projected = keep(device.create_buffer(size=n*80, usage=storage))
        indices = buffer(order, storage)
        ids = buffer(object_ids, storage)
        colours = buffer(np.zeros(4, 'f4'), storage)    # degree 0 reads are clamped; the data pass has no colour
        groups = (len(order)+63)//64
        gx = min(groups, limits['max-compute-workgroups-per-dimension'])
        gy = (groups+gx-1)//gx
        if gy > limits['max-compute-workgroups-per-dimension']:
            raise gpu3d.Unsupported('GPU splat dispatch exceeds adapter limits')
        tan = np.tan(np.deg2rad(camera.fov)/2); focal = height/(2*tan)
        params = np.zeros((6, 4), 'f4'); params[:3, :3] = basis; params[3, :3] = eye
        params[4] = width, height, focal, focal
        params[5] = tan*width/height, tan, len(order), gx
        mesh = keep(device.create_texture(size=(width, height, 1), format='r32float',
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
        depth = np.full((height, width), np.inf, 'f4') if mesh_depth is None else np.ascontiguousarray(mesh_depth, 'f4')
        device.queue.write_texture({'texture': mesh}, depth, {'bytes_per_row': width*4, 'rows_per_image': height},
                                   (width, height, 1))
        project = _pipelines(state)[0]
        count, fill, resolve = _data_pipelines(state)
        def bind(pipeline, resources_):
            return device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                dict(binding=b, resource={'buffer': r}) for b, r in resources_])
        projections = []
        base = 0
        for entry in entries:
            size = len(entry[1])
            sx = min((size+63)//64, limits['max-compute-workgroups-per-dimension'])
            sy = (((size+63)//64)+sx-1)//sx
            if sy > limits['max-compute-workgroups-per-dimension']:
                raise gpu3d.Unsupported('GPU splat dispatch exceeds adapter limits')
            local = params.copy()
            local[0, 3], local[1, 3], local[3, 3] = 0, 0, base
            local[5, 2:] = size, sx
            projections.append((bind(project, [(0, buffer(local, wgpu.BufferUsage.UNIFORM)), (1, combined),
                                               (2, colours), (3, projected)]), sx, sy))
            base += size
        frame = np.array([width, height, focal, focal], 'f4')
        def dims_buffer():
            data = bytearray(32)
            data[:16] = frame.tobytes()
            data[16:] = np.array([nx, ny, len(order), gx], 'u4').tobytes()
            return buffer(bytes(data), wgpu.BufferUsage.UNIFORM)
        uniform = dims_buffer()
        tile_total = nx*ny
        counts = keep(device.create_buffer(size=tile_total*4, usage=storage | wgpu.BufferUsage.COPY_SRC))
        count_group = bind(count, [(0, uniform), (1, projected), (2, indices), (3, counts)])
        counts_staging = read(tile_total*4)
        last_timings['data_upload_ms'] = static_upload_ms + (perf_counter()-t)*1000
        t = perf_counter()
        encoder = device.create_command_encoder(); offset = 0
        for entry in entries:
            size = len(entry[1])*64
            encoder.copy_buffer_to_buffer(entry[2], 0, combined, offset, size); offset += size
        cp = encoder.begin_compute_pass(); cp.set_pipeline(project)
        for group, sx, sy in projections:
            cp.set_bind_group(0, group); cp.dispatch_workgroups(sx, sy, 1)
        cp.set_pipeline(count); cp.set_bind_group(0, count_group); cp.dispatch_workgroups(gx, gy, 1)
        cp.end()
        encoder.copy_buffer_to_buffer(counts, 0, counts_staging, 0, tile_total*4)
        _check(cancel); device.queue.submit([encoder.finish()]); counts_staging.map_sync(wgpu.MapMode.READ)
        try:
            per_tile = np.frombuffer(counts_staging.read_mapped(), 'u4').astype('i8', copy=True)
        finally:
            counts_staging.unmap()
        _check(cancel)
        offsets = np.concatenate(([0], np.cumsum(per_tile)))
        entries_total = int(offsets[-1])
        last_timings['data_bin_ms'] = (perf_counter()-t)*1000
        last_timings['data_tile_entries'] = entries_total
        if entries_total*4 > min(limits['max-storage-buffer-binding-size'], limits['max-buffer-size'], cap):
            raise ValueError(f'GPU splat tile lists need {entries_total*4/1024**2:.1f} MiB, more than the adapter '
                             'allows: lower the resolution or the splat count, or use the CPU renderer')
        t = perf_counter()
        offsets_buffer = buffer(offsets.astype('u4'), storage)
        lists = keep(device.create_buffer(size=max(entries_total, 1)*4, usage=storage))
        cursor = keep(device.create_buffer(size=tile_total*4, usage=storage))
        fill_group = bind(fill, [(0, uniform), (1, projected), (2, indices), (3, cursor),
                                 (4, offsets_buffer), (5, lists)])
        output = keep(device.create_buffer(size=width*height*16, usage=storage | wgpu.BufferUsage.COPY_SRC))
        resolve_group = device.create_bind_group(layout=resolve.get_bind_group_layout(0), entries=[
            dict(binding=0, resource={'buffer': uniform}), dict(binding=1, resource={'buffer': projected}),
            dict(binding=2, resource={'buffer': offsets_buffer}), dict(binding=3, resource={'buffer': lists}),
            dict(binding=4, resource={'buffer': ids}), dict(binding=5, resource=mesh.create_view()),
            dict(binding=6, resource={'buffer': output})])
        staging = read(width*height*16)
        encoder = device.create_command_encoder()
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(fill); cp.set_bind_group(0, fill_group); cp.dispatch_workgroups(gx, gy, 1)
        cp.set_pipeline(resolve); cp.set_bind_group(0, resolve_group)
        cp.dispatch_workgroups((width+7)//8, (height+7)//8, 1)
        cp.end()
        encoder.copy_buffer_to_buffer(output, 0, staging, 0, width*height*16)
        _check(cancel); device.queue.submit([encoder.finish()]); staging.map_sync(wgpu.MapMode.READ)
        try:
            raw = np.frombuffer(staging.read_mapped(), 'f4').reshape(height, width, 4).copy()
        finally:
            staging.unmap()
        _check(cancel)
        last_timings['data_resolve_ms'] = (perf_counter()-t)*1000
        return raw[..., 0] > 0, raw[..., 1].copy(), raw[..., 2].copy()
    finally:
        for resource in reversed(resources):
            resource.destroy()
