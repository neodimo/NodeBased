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
