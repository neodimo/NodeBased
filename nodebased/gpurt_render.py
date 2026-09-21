"""GPU triangle AOV renderer; one invocation owns the complete ray peel.

Seven storage bindings. Textures retain the CPU float32 mip chain, including
second near-clipped triangle mip selection. Traversal has gpurt's f32 edge
band; shading and accumulation are f32 rather than the reference's f64.
"""
import math
import numpy as np
from . import gpurt, gpu3d, raytrace, scene3d as s

GPU_RT_RAYS_PER_SUBMISSION = 1 << 19


def check_capability(state):
    reason = gpurt.check_capability(state)
    if reason:
        return reason
    if gpurt._limits(state).get('max-storage-buffers-per-shader-stage', 0) < 7:
        return 'GPU ray tracing unavailable: max-storage-buffers-per-shader-stage too small (needs 7)'
    if 'device' in state and 'wgpu' in state:
        try:
            _pipeline(state)
        except Exception as exc:
            return f'GPU ray tracing beauty compute unavailable: {exc}'
    return None


# Reuse the tested outward-bound slab traversal verbatim.
_SLAB = '// ' + gpurt._SHADER.split('// Explicit parallel slabs')[1].split('@compute')[0]
_SHADER = r'''
struct Node { lo: vec3<f32>, left: i32, hi: vec3<f32>, right: i32,
 offset: u32, count: u32, pad: vec2<u32> };
struct Triangle { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32> };
// n0.w stores the 1-based geometry ID, independently of material offsets.
struct Attr { n0: vec4<f32>, n1: vec4<f32>, n2: vec4<f32>,
 uv: vec4<f32>, info: vec4<f32>, a: vec4<f32>, e: vec4<f32>, f: vec4<f32> };
// output uses scene3d.RENDER_OUTPUTS indices (splats is rejected on the host).
struct Params { count: u32, gx: u32, lights: u32, maxhits: u32,
 ambient: f32, bias: f32, empty: u32, light_offset: u32,
 output: u32, pad0: u32, pad1: u32, pad2: u32 };
struct Hit { t: f32, id: i32, u: f32, v: f32 };
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> order: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> attrs: array<Attr>;
// Material = tint, (specular, shininess, emission, unused), then mip metadata.
// Lights follow materials and mip metadata at the uniform-specified offset.
@group(0) @binding(4) var<storage, read> table: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> texels: array<vec4<f32>>;
// Ray record: origin/near, direction/far, result; result.w=-1 signals overflow.
@group(0) @binding(6) var<storage, read_write> rays: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> params: Params;
''' + _SLAB + r'''
fn intersect(id: u32, o: vec3<f32>, d: vec3<f32>, lo: f32, hi: f32, edge: f32) -> Hit {
 let tr=triangles[id]; let h=cross(d,tr.e2.xyz); let det=dot(h,tr.e1.xyz);
 if (abs(det)<=1e-10) { return Hit(hi,-1,0.,0.); }
 let delta=o-tr.v0.xyz; let inv=1./det; let u=dot(delta,h)*inv;
 let q=cross(delta,tr.e1.xyz); let v=dot(d,q)*inv; let t=dot(tr.e2.xyz,q)*inv;
 if (u>=-edge && v>=-edge && u+v<=1.+edge && t>lo && t<hi) { return Hit(t,i32(id),u,v); }
 return Hit(hi,-1,0.,0.);
}
fn nearest(o: vec3<f32>, d: vec3<f32>, lo: f32, hi: f32, ct: f32, cp: i32) -> Hit {
 var best=Hit(hi,-1,0.,0.);
 var stack: array<i32,64>; stack[0]=0; var size=1u;
 if (params.empty!=0u) { return best; }
 loop {
  if (size==0u) { break; } size--; let index=stack[size];
  if (entry(index,o,d,lo,best.t)==bitcast<f32>(0x7f800000u)) { continue; }
  let node=nodes[index];
  if (node.count==0u) { stack[size]=node.left; stack[size+1u]=node.right; size+=2u; }
  else { for (var p=0u;p<node.count;p++) {
   let id=order[node.offset+p]; let hit=intersect(id,o,d,lo,hi,1e-6);
   if (hit.id<0 || !(hit.t>ct || (hit.t==ct && hit.id>cp))) { continue; }
   if (hit.t<best.t || (hit.t==best.t && (best.id<0 || hit.id<best.id))) { best=hit; }
  } }
 }
 return best;
}
fn visibility(o: vec3<f32>, d: vec3<f32>, limit: f32) -> f32 {
 var transmission=1.; var stack: array<i32,64>; stack[0]=0; var size=1u;
 loop {
  if (size==0u) { break; } size--; let index=stack[size];
  if (entry(index,o,d,params.bias*.01,limit)==bitcast<f32>(0x7f800000u)) { continue; }
  let node=nodes[index];
  if (node.count==0u) { stack[size]=node.left; stack[size+1u]=node.right; size+=2u; }
  else { for (var p=0u;p<node.count;p++) {
   let id=order[node.offset+p]; let hit=intersect(id,o,d,params.bias*.01,limit,0.);
   if (hit.id>=0) { transmission*=1.-triangles[id].v0.w; }
  } }
 }
 return transmission;
}
fn texel(descriptor: vec4<f32>, xy: vec2<i32>) -> vec4<f32> {
 let p=clamp(xy,vec2<i32>(0),vec2<i32>(descriptor.yz)-vec2<i32>(1));
 return texels[u32(descriptor.x)+u32(p.y)*u32(descriptor.y)+u32(p.x)];
}
fn sample_texture(descriptor: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
 let p=vec2<f32>(clamp(uv.x,0.,1.),1.-clamp(uv.y,0.,1.))*descriptor.yz-.5;
 let ij=vec2<i32>(floor(p)); let f=fract(p);
 return mix(mix(texel(descriptor,ij),texel(descriptor,ij+vec2<i32>(1,0)),f.x),
            mix(texel(descriptor,ij+vec2<i32>(0,1)),texel(descriptor,ij+vec2<i32>(1,1)),f.x),f.y);
}
fn unit(v: vec3<f32>) -> vec3<f32> { return v/max(length(v),1e-8); }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 let r=(group.y*params.gx+group.x)*64u+lane;
 if (r>=params.count) { return; }
 let origin=rays[r*3u]; let direction=rays[r*3u+1u];
 var ct=-bitcast<f32>(0x7f800000u); var cp=-1; var previous_object=-1.; var previous_edge=false;
 var accum=vec4<f32>(0.); var transmission=1.; var surfaces=0u;
 loop {
  let hit=nearest(origin.xyz,direction.xyz,origin.w,direction.w,ct,cp);
  if (hit.id<0) { break; }
  let at=attrs[hit.id];
  let edge=min(min(abs(hit.u),abs(hit.v)),abs(1.-hit.u-hit.v))<1e-5;
  // float32 rounding differs between GPUs (FMA), so a shared-edge hit found through both triangles differs
  // by ~1e-6 relative in t and barycentrics: the CPU's 1e-10 thresholds are widened to 1e-5 here.
  let duplicate=at.info.x==previous_object && edge && previous_edge && abs(hit.t-ct)<=1e-5*max(1.,abs(hit.t));
  ct=hit.t; cp=hit.id; previous_object=at.info.x; previous_edge=edge;
  if (at.info.x<0. || duplicate) { continue; }
  surfaces++;
  if (surfaces>params.maxhits) { rays[r*3u+2u]=vec4<f32>(0.,0.,0.,-1.); return; }
  let w=vec3<f32>(1.-hit.u-hit.v,hit.u,hit.v);
  let tr=triangles[hit.id]; let position=tr.v0.xyz+hit.u*tr.e1.xyz+hit.v*tr.e2.xyz;
  var normal=unit(at.n0.xyz*w.x+at.n1.xyz*w.y+at.n2.xyz*w.z);
  let uv=at.uv.xy*w.x+at.uv.zw*w.y+at.info.zw*w.z;
  let material=u32(at.info.x); let properties=table[material+1u];
  var source=table[material]; var level=at.info.y;
  if (at.a.w>=0.) {
   let delta=position-at.a.xyz; let ee=dot(at.e.xyz,at.e.xyz); let ef=dot(at.e.xyz,at.f.xyz); let ff=dot(at.f.xyz,at.f.xyz);
   let den=ee*ff-ef*ef;
   if (abs(den)>1e-20) {
    let u=(ff*dot(delta,at.e.xyz)-ef*dot(delta,at.f.xyz))/den;
    let v=(ee*dot(delta,at.f.xyz)-ef*dot(delta,at.e.xyz))/den;
    if (u>=-1e-9 && v>=-1e-9 && u+v<=1.+1e-9) { level=at.a.w; }
   }
  }
  source*=sample_texture(table[material+2u+u32(level)],uv);
  // Data passes stop only at positive surface alpha, including texture alpha.
  // Count skipped transparent surfaces above, exactly as the CPU peel does.
  if (params.output==1u || params.output==2u || params.output>=7u) {
   if (source.w<=0.) { continue; }
   var value=vec3<f32>(hit.t);
   if (params.output==2u) {
    if (dot(normal,origin.xyz-position)<0.) { normal=-normal; }
    value=normal;
   } else if (params.output==7u) { value=position;
   } else if (params.output==8u) { value=vec3<f32>(uv,0.);
   } else if (params.output==9u) { value=vec3<f32>(at.n0.w,0.,0.); }
   accum=vec4<f32>(value,1.); break;
  }
  let emission=source.xyz*properties.z;
  if (params.output==6u) { source=vec4<f32>(emission,source.w); }
  else if (params.output!=3u && params.lights>0u) {
   let toward=unit(origin.xyz-position);
   if (dot(normal,origin.xyz-position)<0.) { normal=-normal; }
   var radiance=vec3<f32>(params.ambient); var specular=vec3<f32>(0.);
   for (var j=0u;j<params.lights;j++) {
    let start=params.light_offset+j*3u; let lp=table[start]; let ld=table[start+1u]; let lc=table[start+2u];
    var to_light=-ld.xyz;
    if (lp.w>0.) { to_light=unit(lp.xyz-position); }
    let lambert=dot(normal,to_light); var vis=1.;
    if (ld.w>0.) {
     let o=position+normal*params.bias; var d=-ld.xyz; var limit=bitcast<f32>(0x7f800000u);
     if (lp.w>0.) { let delta=lp.xyz-o; limit=length(delta); d=delta/max(limit,1e-8); }
     vis=visibility(o,d,limit);
    }
    radiance+=max(lambert*vis,0.)*lc.xyz;
    if (params.output==0u || params.output==5u) {
     let lobe=pow(max(dot(normal,unit(to_light+toward)),0.),properties.y);
     specular+=properties.x*lobe*select(0.,1.,lambert>0.)*vis*lc.xyz;
    }
   }
   if (params.output==5u) { source=vec4<f32>(specular*source.w,source.w); }
   else { source=vec4<f32>(source.xyz*radiance+specular*source.w,source.w); }
  } else if (params.output==5u) { source=vec4<f32>(0.,0.,0.,source.w); }
  if (params.output==0u) { source=vec4<f32>(source.xyz+emission,source.w); }
  accum+=transmission*source; transmission*=1.-source.w;
  if (source.w>=.999) { break; }
 }
 rays[r*3u+2u]=accum;
}
'''


def _pipeline(state):
    if '_gpurt_beauty_pipeline' not in state:
        device = state['device']
        state['_gpurt_beauty_pipeline'] = device.create_compute_pipeline(
            layout='auto', compute={'module': device.create_shader_module(code=_SHADER), 'entry_point': 'main'})
    return state['_gpurt_beauty_pipeline']


def _prepare(scene, camera, width, height, cancel=None):
    if scene.splats or any(g.projection is not None for g in scene.geometries):
        raise gpu3d.Unsupported('GPU ray tracing beauty supports triangle meshes without projections or splats')
    eye, view = s._view_basis(camera)
    focal = 1/math.tan(math.radians(camera.fov)*.5)
    count = sum(len(g.triangles) for g in scene.geometries)
    attributes = np.zeros((max(1, count), 8, 4), 'f4')
    attributes[:, 4, 0] = -1
    attributes[:, 5, 3] = -1
    vertices, alphas, table, textures = [], [], [], []
    offset, primitive = 0, 0
    for object_id, geometry in enumerate(scene.geometries, 1):
        raytrace._cancel(cancel)
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        local = (view @ (world-eye).T).T
        vertices.append(world[geometry.triangles].astype('f8'))
        alphas.extend([np.clip(geometry.color[3], 0, 1)]*len(geometry.triangles))
        normals = None
        if geometry.normals is not None:
            normals = (np.linalg.inv(matrix[:3, :3]).T @ geometry.normals.T).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        uv = geometry.uvs if geometry.uvs is not None else np.zeros((len(world), 2), 'f4')
        mips = s._mip_chain(geometry.texture) if geometry.texture is not None and geometry.uvs is not None else None
        material = len(table)
        tint = np.asarray(geometry.color, 'f4').copy(); tint[:3] *= tint[3]
        table.extend([tint, (geometry.specular, geometry.shininess, geometry.emission, 0)])
        for mip in mips if mips is not None else [np.ones((1, 1, 4), 'f4')]:
            h, w = mip.shape[:2]
            table.append((offset, w, h, 0)); textures.append(mip.reshape(-1, 4)); offset += w*h
        for tri in geometry.triangles:
            if primitive % 256 == 0:
                raytrace._cancel(cancel)
            at = attributes[primitive]; primitive += 1
            z = -local[tri, 2]
            if (z <= camera.near).all() or (z >= camera.far).all():
                continue
            if normals is None:
                face = np.cross(world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]])
                ns = np.broadcast_to(face/max(float(np.linalg.norm(face)), 1e-8), (3, 3))
            else:
                ns = normals[tri]
            attrs = np.concatenate((local[tri], world[tri], ns, uv[tri]), axis=1).astype('f4')
            at[:3, :3] = ns
            at[0, 3] = object_id
            at[3] = uv[tri[:2]].reshape(4)
            at[4] = (material, 0, *uv[tri[2]])
            for piece, clipped in enumerate(s._clip_near(attrs, z, camera.near)):
                a, b, c = s._to_pixels(clipped[:, :3], -clipped[:, 2], focal, width/height, width, height)
                den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
                level = s._triangle_mip(clipped, den, mips, None)
                if piece == 0:
                    at[4, 1] = level
                else:
                    a, b, c = clipped[:, 3:6]
                    at[5] = (*a, level); at[6, :3] = b-a; at[7, :3] = c-a
    light_offset = len(table)
    lights = [light for light in scene.lights if light.intensity > 0]
    for light in lights:
        position, direction = light.world()
        table.extend([(*position, light.kind == 'Point'), (*direction, light.shadows),
                      (* (np.asarray(light.color)*light.intensity), 0)])
    triangles = np.concatenate(vertices) if vertices else np.empty((0, 3, 3), 'f8')
    primitives = raytrace.TriangleSet(triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0], np.asarray(alphas))
    bvh = raytrace.Bvh.build(*primitives.aabbs(), cancel=cancel)
    bias = 1e-3*max(1., float(np.ptp(triangles.reshape(-1, 3), axis=0).max())) if count else .001
    return (primitives, bvh, attributes, np.asarray(table or [(0, 0, 0, 0)], 'f4'),
            np.concatenate(textures) if textures else np.zeros((1, 4), 'f4'), len(lights), light_offset, bias)


def render_beauty(state, scene, camera, width, height, background, ambient, samples=1, cancel=None):
    """Compatibility entry point for premultiplied rgba beauty."""
    return render(state, scene, camera, width, height, background, ambient,
                  'rgba', samples, cancel)


def render(state, scene, camera, width, height, background, ambient,
           output='rgba', samples=1, cancel=None):
    """Return a read-only float32 AOV using one GPU peel per primary ray.

    Data passes select the first positive-alpha hit, use binary coverage and
    ignore samples/background. Light components composite without background.
    """
    if output == 'splats':
        raise gpu3d.Unsupported('splats output is CPU-only')
    if output not in s.RENDER_OUTPUTS:
        raise ValueError(f'Unknown 3D render output {output!r}')
    raytrace._cancel(cancel)
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError('Render dimensions must be positive')
    samples = 1 if output in s.DATA_OUTPUTS else max(1, min(int(samples), 4))
    iw, ih = width*samples, height*samples
    prepared = _prepare(scene, camera, iw, ih, cancel)
    primitives, bvh, attrs, table, texels, lights, light_offset, bias = prepared
    reason = check_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    gpurt._memory_check(state, max(attrs.nbytes, table.nbytes, texels.nbytes, iw*48))
    dimension = min(65535, gpurt._limits(state)['max-compute-workgroups-per-dimension'])
    budget = min(int(GPU_RT_RAYS_PER_SUBMISSION), gpurt._cap(state)//48, dimension*dimension*64)
    if budget < iw:
        raise ValueError(f'GPU ray tracing needs about {iw*48/2**20:.3f} MiB for one row; submission ray limit is {budget}')
    rows = max(1, budget//iw)
    result = np.empty((ih, iw, 4), 'f4')
    device, wgpu = state['device'], state['wgpu']
    resources = []
    def upload(data, usage):
        resource = device.create_buffer_with_data(data=data, usage=usage)
        resources.append(resource)
        return resource
    try:
        with gpurt.GpuTriangleScene(state, primitives, bvh) as triangles:
            persistent = [upload(data, wgpu.BufferUsage.STORAGE) for data in (attrs, table, texels)]
            pipeline = _pipeline(state)
            for y0 in range(0, ih, rows):
                raytrace._cancel(cancel)
                y1 = min(ih, y0+rows)
                o, d, lo, hi = gpurt.primary_rays(camera, iw, ih, rows=(y0, y1))
                n = len(o); raw = np.zeros((n, 3, 4), 'f4')
                raw[:, 0, :3], raw[:, 0, 3] = o, lo
                raw[:, 1, :3], raw[:, 1, 3] = d, hi
                groups = (n+63)//64; gx = min(dimension, groups); gy = (groups+gx-1)//gx
                params = np.array([n, gx, lights, s.MAX_HITS_PER_RAY, 0, 0, triangles.empty, light_offset,
                                   s.RENDER_OUTPUTS.index(output), 0, 0, 0], 'u4')
                params.view('f4')[4:6] = ambient, bias
                mark = len(resources)
                try:
                    io = upload(raw, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
                    uniform = upload(params, wgpu.BufferUsage.UNIFORM)
                    bindings = [*triangles.buffers, *persistent, io, uniform]
                    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                        {'binding': i, 'resource': {'buffer': b}} for i, b in enumerate(bindings)])
                    encoder = device.create_command_encoder(); compute = encoder.begin_compute_pass()
                    compute.set_pipeline(pipeline); compute.set_bind_group(0, group)
                    compute.dispatch_workgroups(gx, gy, 1); compute.end()
                    raytrace._cancel(cancel)
                    device.queue.submit([encoder.finish()])
                    read = np.frombuffer(device.queue.read_buffer(io), 'f4').reshape(n, 3, 4)[:, 2]
                    raytrace._cancel(cancel)
                    if np.any(read[:, 3] < 0):
                        raise ValueError(f'Ray-traced render exceeds MAX_HITS_PER_RAY ({s.MAX_HITS_PER_RAY}): more than {s.MAX_HITS_PER_RAY} surfaces composited along a ray')
                    result[y0:y1] = read.reshape(y1-y0, iw, 4)
                finally:
                    for resource in reversed(resources[mark:]):
                        resource.destroy()
                    del resources[mark:]
    finally:
        for resource in reversed(resources):
            resource.destroy()
    if output == 'rgba':
        bg = np.asarray(background, 'f4').copy(); bg[3] = np.clip(bg[3], 0, 1); bg[:3] *= bg[3]
        result += bg*(1-result[..., 3:4])
    if samples > 1:
        result = result.reshape(height, samples, width, samples, 4).mean(axis=(1, 3))
    result.flags.writeable = False
    return result
