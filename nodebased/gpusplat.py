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


def _appearances(state, instances, keys, entries, eye, basis, camera, width, height, lighting, cancel):
    """Per-instance colour buffers, as (buffer or array, SH degree, sRGB flag, count, dynamic) tuples, and the host
    milliseconds spent uploading static ones. Baked colours are cached by cloud identity; relit and shadow-caught
    instances are recomputed (and uploaded by the caller) every call."""
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
    return appearances, upload_ms


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
    appearances, upload_ms = _appearances(state, instances, keys, entries, eye, basis, camera, width, height,
                                          lighting, cancel)
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

# Limits that keep one submission under the driver's watchdog (a job over about 10 s killed the AMD iGPU's graphics
# ring on Linux; Windows resets a display adapter after 2 s). A resolve is cut into rectangles of whole tiles, one
# submission each, with at most DATA_BAND_EVALUATIONS fragment evaluations (a list entry seen by one pixel) per
# rectangle. Cutting does not shorten a single pixel's walk of a very long tile list, which is what a low-resolution
# render of a dense capture makes: the walk of the busiest tile takes about 1.6 microseconds per list entry on the
# RTX 3080 Ti (270,000 entries, the capture at 320 x 180: 0.46 s; 622,000 at 160 x 90: 0.98 s) and about 11 on the
# AMD Radeon 8060S (103,000 entries, 640 x 360: 1.14 s). A list longer than DATA_MAX_TILE_LIST, which keeps that
# walk under about 1.5 s, is refused (`auto` renders on the CPU).
DATA_BAND_EVALUATIONS = {'discrete': 3e9, 'integrated': 2e8, 'cpu': 1e9, 'other': 2e8}
DATA_MAX_TILE_LIST = {'discrete': 700_000, 'integrated': 130_000, 'cpu': 2_000_000, 'other': 100_000}


def resolve_bands(per_tile, nx, ny, width, height, budget, tile=16):
    """Rectangles (x, y, w, h) in pixels covering the frame, cut along whole tiles so that each holds at most
    `budget` fragment evaluations, except a single tile whose own list is longer than that. Whole tile rows are
    merged while they fit; a row that does not fit alone is split into runs of tiles."""
    work = np.asarray(per_tile, 'f8').reshape(ny, nx)
    cols = np.minimum(tile, width - tile*np.arange(nx))
    rows = np.minimum(tile, height - tile*np.arange(ny))
    cost = work * rows[:, None] * cols[None, :]
    bands, ty = [], 0
    while ty < ny:
        total, stop = 0.0, ty
        while stop < ny and (stop == ty or total + cost[stop].sum() <= budget) and cost[stop].sum() <= budget:
            total += cost[stop].sum(); stop += 1
        if stop > ty:
            bands.append((0, tile*ty, width, min(height, tile*stop)-tile*ty)); ty = stop; continue
        run, start = 0.0, 0                         # one row alone is over budget: split it by columns
        for tx in range(nx):
            if tx > start and run + cost[ty, tx] > budget:
                bands.append((tile*start, tile*ty, min(width, tile*tx)-tile*start, int(rows[ty]))); start, run = tx, 0.0
            run += cost[ty, tx]
        bands.append((tile*start, tile*ty, width-tile*start, int(rows[ty]))); ty += 1
    return bands


_BIN = r'''
struct Dims { frame: vec4<f32>, tiles: vec4<u32>, region: vec4<u32> };
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
struct Dims { frame: vec4<f32>, tiles: vec4<u32>, region: vec4<u32> };
struct Projected { centre: vec4<f32>, conic: vec4<f32>, bounds: vec4<f32>,
 normal: vec4<f32>, extra: vec4<f32> };
@group(0) @binding(0) var<uniform> q: Dims;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> lists: array<u32>;
@group(0) @binding(4) var<storage, read> ids: array<f32>;
@group(0) @binding(5) var mesh: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> result: array<vec4<f32>>;
// One workgroup per 16 x 16 pixel tile, one thread per pixel. The tile's fragment list is streamed through
// workgroup memory in batches (every thread reads the same entries; the global reads are dependent and random, so
// a thread fetching its own would spend nearly all its time waiting).
//
// The hit is the first fragment, in (plane depth, index) order, where the opacity accumulated so far reaches
// THRESHOLD. Phase 0 is one pass over the list that keeps the NEAREST nearest fragments, the product of
// (1 - alpha) over all of them and the range of their depth keys. Most pixels end there: no hit when the product
// stays above one half, the hit among the nearest when they already reach it. Otherwise phase 1 is a radix select
// on the depth key five bits per pass (a product of (1 - alpha) and a count per bin; the bin where the running
// product first reaches the threshold is kept and the bins below it are folded into `below`) until few enough
// fragments are left, and phase 2 walks those in order, NEAREST at a time. Depth keys order floats as unsigned
// integers, so ranges and bins are exact. The threads of a tile run the same number of passes (each pass is a
// pass over the shared list for every thread still working; finished threads wait).
const NEAREST = 12u;
const THRESHOLD = 0.5;
const BINS = 32u;
const BATCH = 128u;
var<workgroup> batch: array<Projected, 128>;
var<workgroup> batch_index: array<u32, 128>;
var<workgroup> pending: atomic<u32>;
var<workgroup> go: u32;
struct Fragment { alpha: f32, depth: f32 };
fn order_key(z: f32) -> u32 {
 let b = bitcast<u32>(z);
 return select(~b, b | 0x80000000u, (b & 0x80000000u) == 0u);
}
fn fragment(s: Projected, pixel: vec2<f32>, ray: vec3<f32>, ray_length: f32, mesh_depth: f32) -> Fragment {
 let d = pixel - s.centre.xy;
 let qq = s.conic.x*d.x*d.x + 2.0*s.conic.y*d.x*d.y + s.conic.z*d.y*d.y;
 let alpha = min(0.99, s.centre.w*exp(-0.5*qq));
 if (alpha < 1.0/255.0) { return Fragment(0.0, 0.0); }
 if (pixel.x < s.bounds.x || pixel.y < s.bounds.y || pixel.x >= s.bounds.z || pixel.y >= s.bounds.w) {
   return Fragment(0.0, 0.0);
 }
 let den = dot(ray, s.normal.xyz); var zp = s.centre.z;
 if (den != 0.0) { zp = s.normal.w/den; }
 if (s.conic.w != 0.0 || abs(den)/ray_length < 0.05 || abs(zp-s.centre.z) > 3.0*s.extra.x) {
   zp = s.centre.z;
 }
 if (zp >= mesh_depth) { return Fragment(0.0, 0.0); }
 return Fragment(alpha, zp);
}
fn in_range(key: u32, fixed_bits: u32, prefix: u32) -> bool {
 return fixed_bits == 0u || (key >> (32u - fixed_bits)) == prefix;
}
@compute @workgroup_size(16, 16) fn resolve(@builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
 let width = u32(q.frame.x); let height = u32(q.frame.y);
 let at_x = q.region.x + wid.x*16u + lid3.x; let at_y = q.region.y + wid.y*16u + lid3.y;
 let live = at_x < width && at_y < height;
 let pixel = vec2<f32>(f32(at_x)+0.5, f32(at_y)+0.5);
 let ray = vec3<f32>((pixel.x-q.frame.x*0.5)/q.frame.z, (q.frame.y*0.5-pixel.y)/q.frame.w, 1.0);
 let ray_length = length(ray);
 let tile = (q.region.y/16u + wid.y)*q.tiles.x + q.region.x/16u + wid.x;
 let first = offsets[tile]; let last = offsets[tile+1u];
 var mesh_depth = 0.0;
 if (live) { mesh_depth = textureLoad(mesh, vec2<i32>(i32(at_x), i32(at_y)), 0).x; }
 var kz: array<f32, 12>; var ki: array<u32, 12>; var ka: array<f32, 12>;
 var product: array<f32, 32>; var counts: array<u32, 32>;
 var phase = select(3u, 0u, live);     // 0 first pass, 1 radix select, 2 ordered walk, 3 finished
 var m = 0u; var total = 1.0; var count = 0u; var kmin = 0xffffffffu; var kmax = 0u;
 var fixed_bits = 0u; var prefix = 0u; var below = 1.0; var remaining_count = 0u; var level = 0u;
 var resumed = false; var rz = 0.0; var ri = 0u; var through = 1.0;
 var hit = false; var hit_z = 0.0; var hit_i = 0u;
 for (var iteration = 0u; iteration < 64u; iteration += 1u) {
   if (lid == 0u) { atomicStore(&pending, 0u); }
   workgroupBarrier();
   if (phase != 3u) { atomicStore(&pending, 1u); }
   workgroupBarrier();
   if (lid == 0u) { go = atomicLoad(&pending); }
   workgroupBarrier();
   if (workgroupUniformLoad(&go) == 0u) { break; }
   if (phase == 1u) {
     for (var b = 0u; b < BINS; b += 1u) { product[b] = 1.0; counts[b] = 0u; }
   }
   m = 0u;
   let bits = min(5u, 32u - fixed_bits);
   let shift = 32u - fixed_bits - bits;
   let mask = (1u << bits) - 1u;
   for (var base = first; base < last; base += BATCH) {
     workgroupBarrier();
     if (lid < BATCH && base + lid < last) {
       let i = lists[base + lid];
       batch[lid] = projected[i]; batch_index[lid] = i;
     }
     workgroupBarrier();
     // D3D12 (FXC) rule: a barrier may not follow a varying continue/break/return, so a finished thread skips the
     // work under a guard and still reaches the next iteration's barriers.
     if (phase != 3u) {
       let used = min(BATCH, last - base);
       for (var j = 0u; j < used; j += 1u) {
         let f = fragment(batch[j], pixel, ray, ray_length, mesh_depth);
         if (f.alpha <= 0.0) { continue; }
         let i = batch_index[j];
         let key = order_key(f.depth);
         if (phase == 0u) {
           kmin = min(kmin, key); kmax = max(kmax, key); total = total*(1.0-f.alpha); count += 1u;
         } else if (phase == 1u) {
           if (!in_range(key, fixed_bits, prefix)) { continue; }
           let b = (key >> shift) & mask;
           product[b] = product[b]*(1.0-f.alpha); counts[b] += 1u;
           continue;
         } else {
           if (!in_range(key, fixed_bits, prefix)) { continue; }
           if (resumed && !(f.depth > rz || (f.depth == rz && i > ri))) { continue; }
         }
         if (m == NEAREST && !(f.depth < kz[NEAREST-1u] || (f.depth == kz[NEAREST-1u] && i < ki[NEAREST-1u]))) { continue; }
         var at = min(m, NEAREST-1u);
         while (at > 0u && (kz[at-1u] > f.depth || (kz[at-1u] == f.depth && ki[at-1u] > i))) {
           kz[at] = kz[at-1u]; ki[at] = ki[at-1u]; ka[at] = ka[at-1u]; at -= 1u;
         }
         kz[at] = f.depth; ki[at] = i; ka[at] = f.alpha;
         if (m < NEAREST) { m += 1u; }
       }
     }
   }
   // What each thread does with the pass it just made (no barriers below).
   if (phase == 0u) {
     if (count == 0u || total > THRESHOLD) {
       phase = 3u;
     } else {
       through = 1.0;
       for (var j = 0u; j < m; j += 1u) {
         through = through*(1.0-ka[j]);
         if (1.0-through >= THRESHOLD) { hit = true; hit_z = kz[j]; hit_i = ki[j]; break; }
       }
       if (!hit && count <= NEAREST) { hit = true; hit_z = kz[m-1u]; hit_i = ki[m-1u]; }   // rounding at the threshold
       if (hit) {
         phase = 3u;
       } else {
         fixed_bits = countLeadingZeros(kmin ^ kmax);
         if (fixed_bits > 0u) { prefix = kmin >> (32u - fixed_bits); }
         remaining_count = count;
         phase = select(1u, 2u, remaining_count <= 2u*NEAREST || fixed_bits >= 32u);
         through = 1.0;
       }
     }
   } else if (phase == 1u) {
     var running = below; var chosen = BINS; var last_used = 0u;
     for (var b = 0u; b <= mask; b += 1u) {
       if (counts[b] == 0u) { continue; }
       last_used = b;
       if (running*product[b] <= THRESHOLD) { chosen = b; break; }
       running = running*product[b];
     }
     if (chosen == BINS) {
       // The passes disagree about the threshold by rounding: the hit is the last fragment of the range.
       chosen = last_used;
       running = below;
       for (var b = 0u; b < last_used; b += 1u) { running = running*product[b]; }
     }
     prefix = (prefix << bits) | chosen; fixed_bits += bits;
     below = running; remaining_count = counts[chosen];
     level += 1u;
     if (remaining_count <= 2u*NEAREST || fixed_bits >= 32u || level >= 8u) { phase = 2u; through = below; }
   } else if (phase == 2u) {
     for (var j = 0u; j < m; j += 1u) {
       through = through*(1.0-ka[j]);
       if (1.0-through >= THRESHOLD) { hit = true; hit_z = kz[j]; hit_i = ki[j]; break; }
     }
     if (!hit && m > 0u && m < NEAREST) { hit = true; hit_z = kz[m-1u]; hit_i = ki[m-1u]; }
     if (hit || m < NEAREST) {
       phase = 3u;
     } else {
       resumed = true; rz = kz[NEAREST-1u]; ri = ki[NEAREST-1u];
     }
   }
 }
 if (live) {
   var out = vec4<f32>(0.0);
   if (hit) { out = vec4<f32>(1.0, hit_z, ids[hit_i], 0.0); }
   result[at_y*width + at_x] = out;
 }
}
'''


def _data_pipelines(state, tile=16):
    """(count, fill, resolve) pipelines for tiles of `tile` x `tile` pixels; only the 16 x 16 ones have the data
    resolve (the layered beauty pass has its own, `_beauty_pipeline`)."""
    key = '_gpusplat_data_pipelines' if tile == 16 else f'_gpusplat_bin_pipelines_{tile}'
    if key not in state:
        device = state['device']
        binning = device.create_shader_module(code=_BIN.replace('/16.0', f'/{tile}.0'))
        count = device.create_compute_pipeline(layout='auto', compute={'module': binning, 'entry_point': 'count'})
        fill = device.create_compute_pipeline(layout='auto', compute={'module': binning, 'entry_point': 'fill'})
        resolve = None
        if tile == 16:
            resolve = device.create_compute_pipeline(layout='auto', compute={
                'module': device.create_shader_module(code=_RESOLVE), 'entry_point': 'resolve'})
        state[key] = count, fill, resolve
    return state[key]


def check_data_capability(state):
    reason = check_capability(state)
    if reason is not None:
        return reason
    limits = state.get('limits', getattr(state.get('device'), 'limits', {}))
    for name, minimum in (('max-storage-buffers-per-shader-stage', 6), ('max-compute-invocations-per-workgroup', 256),
                          ('max-compute-workgroup-size-x', 16), ('max-compute-workgroup-size-y', 16),
                          ('max-compute-workgroup-storage-size', 12*1024)):
        if limits.get(name, 0) < minimum:
            return f'GPU splat data passes unavailable: {name} too small (needs {minimum})'
    if 'device' in state and 'wgpu' in state:
        try:
            _data_pipelines(state)
        except Exception as exc:
            return f'GPU splat data passes unavailable: {exc}'
    return None


class _Bins:
    """The projected splats and their 16 x 16 tile lists, ready for a resolve pass (see `_project_and_bin`)."""
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _project_and_bin(state, entries, order, camera, width, height, basis, eye, appearances, cap, keep, buffer,
                     static_upload_ms, cancel=None, tile=16):
    """Project every splat (colours from `appearances`, as `_render` builds them, or none for a data pass), count
    how many splats touch each 16 x 16 tile, read the counts back for the host prefix sum and fill the tile lists
    (splats in `order`). `keep` registers a resource for the caller to destroy; `buffer(data, usage)` creates and
    keeps an uploaded buffer. Returns the buffers and per-tile counts a resolve pass needs, `dims(region)` making the
    uniform for a rectangle of whole tiles."""
    device, wgpu = state['device'], state['wgpu']
    limits = device.limits
    n = sum(len(entry[1]) for entry in entries)
    nx, ny = (width+tile-1)//tile, (height+tile-1)//tile
    t = perf_counter()
    storage = wgpu.BufferUsage.STORAGE
    combined = keep(device.create_buffer(size=n*64, usage=storage | wgpu.BufferUsage.COPY_DST))
    projected = keep(device.create_buffer(size=n*80, usage=storage))
    indices = buffer(order, storage)
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
    project = _pipelines(state)[0]
    count, fill, _ = _data_pipelines(state, tile)
    def bind(pipeline, resources_):
        return device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
            dict(binding=b, resource={'buffer': r}) for b, r in resources_])
    projections = []
    base = 0
    for index, entry in enumerate(entries):
        size = len(entry[1])
        sx = min((size+63)//64, limits['max-compute-workgroups-per-dimension'])
        sy = (((size+63)//64)+sx-1)//sx
        if sy > limits['max-compute-workgroups-per-dimension']:
            raise gpu3d.Unsupported('GPU splat dispatch exceeds adapter limits')
        local = params.copy()
        if appearances is None:
            local[0, 3], local[1, 3], local[3, 3] = 0, 0, base
            source = colours
        else:
            source, degree, srgb, _, dynamic = appearances[index]
            if dynamic:
                source = buffer(source, storage)
            local[0, 3], local[1, 3], local[3, 3] = degree, srgb, base
        local[5, 2:] = size, sx
        projections.append((bind(project, [(0, buffer(local, wgpu.BufferUsage.UNIFORM)), (1, combined),
                                           (2, source), (3, projected)]), sx, sy))
        base += size
    frame = np.array([width, height, focal, focal], 'f4')
    def dims(region):
        return buffer(frame.tobytes() + np.array([nx, ny, len(order), gx, *region], 'u4').tobytes(),
                      wgpu.BufferUsage.UNIFORM)
    uniform = dims((0, 0, width, height))
    tile_total = nx*ny
    counts = keep(device.create_buffer(size=tile_total*4, usage=storage | wgpu.BufferUsage.COPY_SRC))
    count_group = bind(count, [(0, uniform), (1, projected), (2, indices), (3, counts)])
    counts_staging = keep(device.create_buffer(size=tile_total*4,
                                               usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
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
    kind = gpu3d._adapter_kind(state)
    longest = int(per_tile.max())
    last_timings['data_longest_tile_list'] = longest
    if longest > DATA_MAX_TILE_LIST[kind]:
        raise gpu3d.Unsupported(f'a {width}x{height} splat pass puts {longest:,} splats in one {tile}x{tile} tile, '
                                f'more than the {DATA_MAX_TILE_LIST[kind]:,} the {kind} adapter takes in one '
                                'submission: render larger or on the CPU')
    t = perf_counter()
    offsets_buffer = buffer(offsets.astype('u4'), storage)
    lists = keep(device.create_buffer(size=max(entries_total, 1)*4, usage=storage))
    cursor = keep(device.create_buffer(size=tile_total*4, usage=storage))
    fill_group = bind(fill, [(0, uniform), (1, projected), (2, indices), (3, cursor),
                             (4, offsets_buffer), (5, lists)])
    encoder = device.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(fill); cp.set_bind_group(0, fill_group); cp.dispatch_workgroups(gx, gy, 1)
    cp.end()
    _check(cancel); device.queue.submit([encoder.finish()])
    last_timings['data_fill_ms'] = (perf_counter()-t)*1000
    return _Bins(projected=projected, offsets=offsets_buffer, lists=lists, per_tile=per_tile, nx=nx, ny=ny,
                 dims=dims, focal=focal, order_count=len(order))


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
    resources = []
    def keep(resource):
        resources.append(resource); return resource
    def buffer(data, usage):
        return keep(device.create_buffer_with_data(data=data, usage=usage))
    def read(size):
        return keep(device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ))
    try:
        storage = wgpu.BufferUsage.STORAGE
        ids = buffer(object_ids, storage)
        mesh = keep(device.create_texture(size=(width, height, 1), format='r32float',
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
        depth = np.full((height, width), np.inf, 'f4') if mesh_depth is None else np.ascontiguousarray(mesh_depth, 'f4')
        device.queue.write_texture({'texture': mesh}, depth, {'bytes_per_row': width*4, 'rows_per_image': height},
                                   (width, height, 1))
        resolve = _data_pipelines(state)[2]
        bins = _project_and_bin(state, entries, order, camera, width, height, basis, eye, None, cap, keep, buffer,
                                static_upload_ms, cancel)
        output = keep(device.create_buffer(size=width*height*16, usage=storage | wgpu.BufferUsage.COPY_SRC))
        staging = read(width*height*16)
        kind = gpu3d._adapter_kind(state)
        budget = DATA_BAND_EVALUATIONS[kind]
        t = perf_counter()
        bands = resolve_bands(bins.per_tile, nx, ny, width, height, budget)
        last_timings['data_bands'] = len(bands)
        for x0, y0, w, h in bands:
            _check(cancel)
            group = device.create_bind_group(layout=resolve.get_bind_group_layout(0), entries=[
                dict(binding=0, resource={'buffer': bins.dims((x0, y0, w, h))}),
                dict(binding=1, resource={'buffer': bins.projected}),
                dict(binding=2, resource={'buffer': bins.offsets}), dict(binding=3, resource={'buffer': bins.lists}),
                dict(binding=4, resource={'buffer': ids}), dict(binding=5, resource=mesh.create_view()),
                dict(binding=6, resource={'buffer': output})])
            encoder = device.create_command_encoder()
            cp = encoder.begin_compute_pass()
            cp.set_pipeline(resolve); cp.set_bind_group(0, group)
            cp.dispatch_workgroups((w+15)//16, (h+15)//16, 1)
            cp.end()
            device.queue.submit([encoder.finish()])
        encoder = device.create_command_encoder()
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


# ---------------------------------------------------------------------------------------------------------
# Beauty with transparent meshes (Rendering 7 step S2). The CPU reference (`splatraster.accumulate_splats` with
# `mesh_layers`) merges, per pixel, every mesh surface along the primary ray (up to MAX_MESH_LAYERS, recorded by the
# ray tracer with their shaded premultiplied colour and alpha) with the splat fragments in (depth, authored index)
# order, mesh surfaces first on equal depth, and composites them front to back; what is left of the light over the
# background. The ray tracer writes the mesh surfaces into a GPU buffer (`gpurt_render`, `layers` mode) and this
# resolve reads them, so they never leave the GPU.
#
# A pixel cannot hold every splat fragment it sees, so the sorted walk is made in passes of BEAUTY_NEAREST
# fragments: a pass reads the tile's whole list, keeps the nearest BEAUTY_NEAREST fragments past a cursor (the
# (depth, index) of the last fragment already composited), merges them with the mesh surfaces up to the last of
# them, composites, and moves the cursor. The state of every pixel (colour, transmittance, cursor, how many mesh
# surfaces are done) lives in buffers between passes, which are separate submissions (a pass costs one walk of the
# longest tile list, the unit the per-adapter list cap keeps under the driver watchdog); the host repeats passes
# until the pixels still working are counted as zero. A pixel stops when its transmittance is under 1e-4 (the
# reference's own stopping point in the opaque path; it adds at most that much to a colour of 1) or after a mesh
# surface of alpha 0.9999 and above, whose depth also removes every splat fragment behind it from the walk.
# ---------------------------------------------------------------------------------------------------------
BEAUTY_NEAREST = 16        # fragments a pass keeps per pixel (measured: 8, 32 and 64 are no faster)
BEAUTY_TILE = 8            # the resolve works on 8 x 8 pixel tiles: lists a half to a third as long as 16 x 16, 1.6-2.3x
                           # the entries (4 x 4 is faster still but needs 250 MiB of lists for the capture at 1080p)
BEAUTY_MAX_PASSES = 2048
MAX_MESH_LAYERS = 16
LAYER_RECORD_VEC4 = 2 * MAX_MESH_LAYERS
_BEAUTY = r'''
struct Dims { frame: vec4<f32>, tiles: vec4<u32>, region: vec4<u32>, band: vec4<u32> };
struct Projected { centre: vec4<f32>, conic: vec4<f32>, bounds: vec4<f32>,
 normal: vec4<f32>, extra: vec4<f32> };
@group(0) @binding(0) var<uniform> q: Dims;
@group(0) @binding(1) var<storage, read> projected: array<Projected>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> lists: array<u32>;
@group(0) @binding(4) var<storage, read> layers: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> state_f: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> state_u: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> counters: array<atomic<u32>>;
// band.x: the first row of the band the ray tracer filled, band.y: where its layer records start in `layers`
// (after the four vec4 of each ray's record). A layer record is two vec4: (depth, 0, 0, 0), (premultiplied rgb, alpha);
// the count of a ray's layers is in the x of its third header vec4, and -1 in the w says it had too many.
const NEAREST = @K@u;
const BATCH = 128u;
const LAYERS = 16u;
const OPAQUE = 0.9999;
const STOP = 1e-4;
var<workgroup> batch: array<Projected, 128>;
var<workgroup> batch_index: array<u32, 128>;
var<workgroup> pending: atomic<u32>;
var<workgroup> go: u32;
struct Fragment { alpha: f32, depth: f32 };
fn fragment(s: Projected, pixel: vec2<f32>, ray: vec3<f32>, ray_length: f32, behind: f32) -> Fragment {
 if (pixel.x < s.bounds.x || pixel.y < s.bounds.y || pixel.x >= s.bounds.z || pixel.y >= s.bounds.w) {
   return Fragment(0.0, 0.0);
 }
 let d = pixel - s.centre.xy;
 let qq = s.conic.x*d.x*d.x + 2.0*s.conic.y*d.x*d.y + s.conic.z*d.y*d.y;
 let alpha = min(0.99, s.centre.w*exp(-0.5*qq));
 if (alpha < 1.0/255.0) { return Fragment(0.0, 0.0); }
 let den = dot(ray, s.normal.xyz); var zp = s.centre.z;
 if (den != 0.0) { zp = s.normal.w/den; }
 if (s.conic.w != 0.0 || abs(den)/ray_length < 0.05 || abs(zp-s.centre.z) > 3.0*s.extra.x) {
   zp = s.centre.z;
 }
 if (zp >= behind) { return Fragment(0.0, 0.0); }
 return Fragment(alpha, zp);
}
@compute @workgroup_size(@T@, @T@) fn resolve(@builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
 let width = u32(q.frame.x); let height = u32(q.frame.y);
 let at_x = q.region.x + wid.x*@T@u + lid3.x; let at_y = q.region.y + wid.y*@T@u + lid3.y;
 let live = at_x < width && at_y < height;
 let pixel = vec2<f32>(f32(at_x)+0.5, f32(at_y)+0.5);
 let ray = vec3<f32>((pixel.x-q.frame.x*0.5)/q.frame.z, (q.frame.y*0.5-pixel.y)/q.frame.w, 1.0);
 let ray_length = length(ray);
 let tile = (q.region.y/@T@u + wid.y)*q.tiles.x + q.region.x/@T@u + wid.x;
 let first = offsets[tile]; let last = offsets[tile+1u];
 let local = select(0u, (at_y - q.band.x)*width + at_x, live);
 let mesh = q.band.y + local*(2u*LAYERS);
 var colour = vec3<f32>(0.0); var trans = 1.0; var cursor_z = 0.0; var cursor_i = 0u; var done_layers = 0u;
 var finished = !live; var count = 0u; var behind = 3.4e38;
 if (live) {
   let saved = state_u[local];
   if (saved.w != 0u) {
     let s0 = state_f[local*2u]; colour = s0.xyz; trans = s0.w; cursor_z = state_f[local*2u+1u].x;
     cursor_i = saved.x; done_layers = saved.y; finished = saved.z != 0u;
   }
   let header = layers[local*4u+2u];
   if (header.w < 0.0) {
     if (!finished) { atomicAdd(&counters[1], 1u); }
     finished = true;
   } else if (!finished) {
     count = u32(header.x);
     for (var l = 0u; l < count; l += 1u) {
       if (layers[mesh + 2u*l + 1u].w >= OPAQUE) { behind = layers[mesh + 2u*l].x; break; }
     }
   }
 }
 if (lid == 0u) { atomicStore(&pending, 0u); }
 workgroupBarrier();
 if (!finished) { atomicStore(&pending, 1u); }
 workgroupBarrier();
 if (lid == 0u) { go = atomicLoad(&pending); }
 workgroupBarrier();
 if (workgroupUniformLoad(&go) == 0u) { return; }
 var kz: array<f32, @K@>; var ki: array<u32, @K@>; var ka: array<f32, @K@>;
 var m = 0u;
 for (var base = first; base < last; base += BATCH) {
   workgroupBarrier();
   for (var k = lid; k < BATCH && base + k < last; k += @T@u*@T@u) {
     let i = lists[base + k];
     batch[k] = projected[i]; batch_index[k] = i;
   }
   workgroupBarrier();
   // D3D12 (FXC) rule: no barrier may follow a varying continue, so a finished thread skips the work under a guard.
   if (!finished) {
     let used = min(BATCH, last - base);
     for (var j = 0u; j < used; j += 1u) {
       let f = fragment(batch[j], pixel, ray, ray_length, behind);
       if (f.alpha <= 0.0) { continue; }
       let i = batch_index[j];
       if (cursor_i != 0u && !(f.depth > cursor_z || (f.depth == cursor_z && i + 1u > cursor_i))) { continue; }
       if (m == NEAREST && !(f.depth < kz[NEAREST-1u] || (f.depth == kz[NEAREST-1u] && i < ki[NEAREST-1u]))) { continue; }
       var at = min(m, NEAREST-1u);
       while (at > 0u && (kz[at-1u] > f.depth || (kz[at-1u] == f.depth && ki[at-1u] > i))) {
         kz[at] = kz[at-1u]; ki[at] = ki[at-1u]; ka[at] = ka[at-1u]; at -= 1u;
       }
       kz[at] = f.depth; ki[at] = i; ka[at] = f.alpha;
       if (m < NEAREST) { m += 1u; }
     }
   }
 }
 if (finished) { return; }
 // Composite the window with the mesh surfaces up to its last fragment (all of them when the window is not full,
 // which is the end of the walk). Mesh surfaces come before a splat fragment of equal depth.
 var stopped = false;
 for (var j = 0u; j < m && !stopped; j += 1u) {
   while (done_layers < count && layers[mesh + 2u*done_layers].x <= kz[j]) {
     let layer = layers[mesh + 2u*done_layers + 1u];
     colour += trans*layer.xyz; trans = trans*(1.0-layer.w); done_layers += 1u;
     if (trans < STOP) { stopped = true; break; }
   }
   if (stopped) { break; }
   colour += trans*ka[j]*projected[ki[j]].extra.yzw; trans = trans*(1.0-ka[j]);
   if (trans < STOP) { stopped = true; }
 }
 var more = false;
 if (!stopped) {
   if (m == NEAREST) {
     more = true; cursor_z = kz[NEAREST-1u]; cursor_i = ki[NEAREST-1u] + 1u;
   } else {
     while (done_layers < count) {
       let layer = layers[mesh + 2u*done_layers + 1u];
       colour += trans*layer.xyz; trans = trans*(1.0-layer.w); done_layers += 1u;
       if (trans < STOP) { break; }
     }
   }
 }
 state_f[local*2u] = vec4<f32>(colour, trans);
 state_f[local*2u+1u] = vec4<f32>(cursor_z, 0.0, 0.0, 0.0);
 state_u[local] = vec4<u32>(cursor_i, done_layers, select(1u, 0u, more), 1u);
 if (more) { atomicAdd(&counters[0], 1u); }
}
'''


def _beauty_pipeline(state, tile=None):
    tile = BEAUTY_TILE if tile is None else tile
    nearest = BEAUTY_NEAREST
    key = f'_gpusplat_beauty_pipeline_{tile}_{nearest}'
    if key not in state:
        state[key] = state['device'].create_compute_pipeline(layout='auto', compute={
            'module': state['device'].create_shader_module(
                code=_BEAUTY.replace('@T@', str(tile)).replace('@K@', str(nearest))),
            'entry_point': 'resolve'})
    return state[key]


def check_layered_capability(state):
    """Why splats cannot be resolved against recorded mesh layers on this adapter, or None."""
    reason = check_data_capability(state)
    if reason is not None:
        return reason
    limits = state.get('limits', getattr(state.get('device'), 'limits', {}))
    if limits.get('max-storage-buffers-per-shader-stage', 0) < 8:
        return 'GPU splats with transparent meshes unavailable: max-storage-buffers-per-shader-stage too small (needs 8)'
    if 'device' in state and 'wgpu' in state:
        try:
            _beauty_pipeline(state)
        except Exception as exc:
            return f'GPU splats with transparent meshes unavailable: {exc}'
    return None


class LayeredResolve:
    """Splats projected and binned once for a frame, resolved a band of rows at a time against the mesh layers a
    ray-traced band left in a GPU buffer. Use `open_layered`; call `close()` (or use it as a context manager)."""
    def __init__(self, state, width, height, bins, resources, tile):
        self.state, self.width, self.height, self.bins, self._resources = state, width, height, bins, resources
        self.tile = tile

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for resource in reversed(self._resources):
            resource.destroy()
        self._resources = []

    def resolve(self, y0, y1, layers, layer_start, cancel=None):
        """Premultiplied float32 colour (rows, width, 3) and alpha (rows, width) of rows y0..y1 of the frame (y0 a
        multiple of the tile size): the splats merged with the mesh layers in `layers`, a GPU buffer holding that band's ray
        records (four vec4 each, row-major from y0) and, from vec4 `layer_start`, 32 vec4 of layer records per ray.
        The background is not included."""
        state, width, bins = self.state, self.width, self.bins
        device, wgpu = state['device'], state['wgpu']
        storage = wgpu.BufferUsage.STORAGE
        rows, count = y1-y0, (y1-y0)*width
        nx = bins.nx
        tile = self.tile
        t0, t1 = y0//tile, (y1+tile-1)//tile
        per_tile = bins.per_tile.reshape(bins.ny, nx)[t0:t1]
        kind = gpu3d._adapter_kind(state)
        regions = [(x, y+y0, w, h) for x, y, w, h in
                   resolve_bands(per_tile, nx, t1-t0, width, rows, DATA_BAND_EVALUATIONS[kind], tile)]
        keep_list = []
        def keep(resource):
            keep_list.append(resource); return resource
        try:
            state_f = keep(device.create_buffer(size=count*32, usage=storage | wgpu.BufferUsage.COPY_SRC))
            state_u = keep(device.create_buffer(size=count*16, usage=storage))
            counters = keep(device.create_buffer(size=8, usage=storage | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST))
            dims_buffers = [keep(device.create_buffer_with_data(
                data=np.array([width, self.height, bins.focal, bins.focal], 'f4').tobytes()
                + np.array([nx, bins.ny, bins.order_count, 0, *region, y0, layer_start, count, 0], 'u4').tobytes(),
                usage=wgpu.BufferUsage.UNIFORM)) for region in regions]
            pipeline = _beauty_pipeline(state, tile)
            groups = [device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                dict(binding=0, resource={'buffer': dims}), dict(binding=1, resource={'buffer': bins.projected}),
                dict(binding=2, resource={'buffer': bins.offsets}), dict(binding=3, resource={'buffer': bins.lists}),
                dict(binding=4, resource={'buffer': layers}), dict(binding=5, resource={'buffer': state_f}),
                dict(binding=6, resource={'buffer': state_u}), dict(binding=7, resource={'buffer': counters})])
                for dims in dims_buffers]
            zero = np.zeros(2, 'u4')
            passes = 0
            while True:
                _check(cancel)
                if passes >= BEAUTY_MAX_PASSES:
                    raise gpu3d.Unsupported(f'a pixel needs more than {BEAUTY_MAX_PASSES} passes over its tile '
                                            'list to become opaque: render on the CPU')
                device.queue.write_buffer(counters, 0, zero)
                for group, (x, y, w, h) in zip(groups, regions):
                    encoder = device.create_command_encoder()
                    cp = encoder.begin_compute_pass()
                    cp.set_pipeline(pipeline); cp.set_bind_group(0, group)
                    cp.dispatch_workgroups((w+tile-1)//tile, (h+tile-1)//tile, 1)
                    cp.end()
                    device.queue.submit([encoder.finish()])
                passes += 1
                pending, errors = np.frombuffer(device.queue.read_buffer(counters), 'u4')[:2]
                if errors:
                    raise ValueError(f'Ray-traced render exceeds MAX_MESH_LAYERS ({MAX_MESH_LAYERS}): more than '
                                     f'{MAX_MESH_LAYERS} surfaces along a ray')
                if not pending:
                    break
            last_timings['beauty_passes'] = max(passes, last_timings.get('beauty_passes', 0))   # the most any band needed
            raw = np.frombuffer(device.queue.read_buffer(state_f), 'f4').reshape(count, 2, 4)
            rgb = raw[:, 0, :3].reshape(rows, width, 3).copy()
            alpha = (1-raw[:, 0, 3]).reshape(rows, width).astype('f4')
            return rgb, alpha
        finally:
            for resource in reversed(keep_list):
                resource.destroy()


def open_layered(state, instances, camera, width, height, lighting=None, cancel=None, budget_bytes=None):
    """Project and bin the splats of `instances` for a `width` x `height` frame and return a `LayeredResolve`, or
    None when no splat reaches the picture (nothing to merge with the meshes). Raises `gpu3d.Unsupported` where the
    adapter cannot do it and ValueError beyond the memory cap, as `render_layer` does. The caller holds the GPU lock."""
    _check(cancel)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    instances = [i if hasattr(i, 'cloud') else SplatInstance(*i) for i in instances]
    n = sum(len(i.cloud) for i in instances)
    if not n:
        return None
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
    reason = check_layered_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    if max(width, height) > limits['max-texture-dimension-2d']:
        raise ValueError('GPU splat render dimensions exceed adapter texture limits')
    keys, entries, static_upload_ms, eye, basis, order = _geometry_and_order(state, instances, camera, cancel)
    _check(cancel)
    if not len(order):
        return None
    appearances, upload_ms = _appearances(state, instances, keys, entries, eye, basis, camera, width, height,
                                          lighting, cancel)
    resources = []
    def keep(resource):
        resources.append(resource); return resource
    def buffer(data, usage):
        return keep(device.create_buffer_with_data(data=data, usage=usage))
    try:
        bins = _project_and_bin(state, entries, order, camera, width, height, basis, eye, appearances, cap, keep,
                                buffer, static_upload_ms + upload_ms, cancel, BEAUTY_TILE)
    except BaseException:
        for resource in reversed(resources):
            resource.destroy()
        raise
    last_timings['beauty_passes'] = 0
    return LayeredResolve(state, width, height, bins, resources, BEAUTY_TILE)
