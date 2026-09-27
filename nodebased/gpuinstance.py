"""GPU ray tracer for `scene.instances` (Instance3D) that never flattens them to geometries.

A top-level tree (`raytrace.Bvh.build` over each instance's world-space bounds) points into one
bottom-level tree per unique source mesh, built once in that mesh's own local space and shared by
every instance of it: N instances of one mesh upload that mesh's triangles once, not N times
(docs/3D_ROADMAP.md "Instancing", which measured the CPU scene representation's equivalent saving
and named this two-level GPU structure as the deferred alternative). `nodebased.scene3d.resolve_instances`
flattens instances into ordinary geometries for every other consumer (CPU render, WriteGeo3D,
exporters); this module is the one GPU/ray-traced path that keeps them as instances all the way
through traversal and shading.

Scope, named here rather than silently assumed done: only scenes made entirely of `scene.instances`
(no ordinary geometries, splats, particles or volumes alongside them -- gpu3d.render already has a
flattened path for those and combining the two is future work); instance sources may not carry a
texture, a projection or the liquid material; shading supports one surface per ray (no
order-independent transparency peel across overlapping instances -- fine for the opaque or
single-layer scenes instancing is used for today); the 'uv' output is not implemented. Any of these
raise gpu3d.Unsupported, the same signal every other GPU limit uses, so callers fall back to the CPU
reference (scene3d.render, which resolves instances the usual way).
"""
import numpy as np

from . import gpu3d, gpurt, raytrace, scene3d as s

GPU_RT_RAYS_PER_SUBMISSION = 1 << 19

SUPPORTED_OUTPUTS = ('rgba', 'depth', 'normals', 'albedo', 'diffuse', 'specular',
                     'emission', 'position', 'object_id')


def check_capability(state):
    for name, minimum in [('max-storage-buffers-per-shader-stage', 7),
                          ('max-storage-buffer-binding-size', 64), ('max-buffer-size', 64),
                          ('max-compute-invocations-per-workgroup', 64),
                          ('max-compute-workgroup-size-x', 64), ('max-compute-workgroups-per-dimension', 1)]:
        if gpurt._limits(state).get(name, 0) < minimum:
            return f'GPU instanced ray tracing unavailable: {name} too small (needs {minimum})'
    if 'device' in state and 'wgpu' in state:
        try:
            _pipeline(state)
        except Exception as exc:
            return f'GPU instanced ray tracing requires compute storage buffers: {exc}'
    return None


_SHADER = r'''
struct Node { lo: vec3<f32>, left: i32, hi: vec3<f32>, right: i32, offset: u32, count: u32, pad: vec2<u32> };
struct Triangle { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32> };
struct Attr { n0: vec4<f32>, n1: vec4<f32>, n2: vec4<f32> };
struct Instance { matrix0: vec4<f32>, matrix1: vec4<f32>, matrix2: vec4<f32>, matrix3: vec4<f32>,
                   inv0: vec4<f32>, inv1: vec4<f32>, inv2: vec4<f32>, inv3: vec4<f32>,
                   tint: vec4<f32>, info: vec4<f32> };   // info = (blas_root, material row, object id, pad)
struct Params { count: u32, gx: u32, lights: u32, light_offset: u32, ambient: f32, bias: f32, output: u32, pad0: u32 };
struct Hit { t: f32, tri: i32, u: f32, v: f32 };
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> order: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> attrs: array<Attr>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> table: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> rays: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> params: Params;
var<private> hit_instance: i32 = -1;
fn unit(v: vec3<f32>) -> vec3<f32> { return v/max(length(v),1e-8); }
fn entry(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32) -> f32 {
 var near=lower; var far=upper;
 for (var a=0u;a<3u;a++) {
  if (d[a]==0.) { if (o[a]<lo[a]||o[a]>hi[a]) { return bitcast<f32>(0x7f800000u); } }
  else { let x=(lo[a]-o[a])/d[a]; let y=(hi[a]-o[a])/d[a]; near=max(near,min(x,y)); far=min(far,max(x,y)); }
 }
 if (near>far) { return bitcast<f32>(0x7f800000u); }
 return near;
}
fn intersect(id: u32, o: vec3<f32>, d: vec3<f32>, lo: f32, hi: f32) -> Hit {
 let tr=triangles[id]; let h=cross(d,tr.e2.xyz); let det=dot(h,tr.e1.xyz);
 if (abs(det)<=1e-10) { return Hit(hi,-1,0.,0.); }
 let delta=o-tr.v0.xyz; let inv=1./det; let u=dot(delta,h)*inv;
 let q=cross(delta,tr.e1.xyz); let v=dot(d,q)*inv; let t=dot(tr.e2.xyz,q)*inv;
 if (u>=-1e-6 && v>=-1e-6 && u+v<=1.000001 && t>lo && t<hi) { return Hit(t,i32(id),u,v); }
 return Hit(hi,-1,0.,0.);
}
// Top-level tree of instance bounds; each leaf enters that instance's own bottom-level tree with the
// ray transformed into its local space (unnormalized, so `t` stays identical in both spaces).
fn trace(o: vec3<f32>, d: vec3<f32>, lo: f32, hi: f32) -> Hit {
 var best=Hit(hi,-1,0.,0.); hit_instance=-1;
 var stack: array<i32,64>; stack[0]=0; var size=1u;
 loop {
  if (size==0u) { break; }
  size--; let index=stack[size]; let node=nodes[index];
  if (entry(node.lo,node.hi,o,d,lo,best.t)==bitcast<f32>(0x7f800000u)) { continue; }
  if (node.count==0u) { stack[size]=node.left; stack[size+1u]=node.right; size+=2u; continue; }
  for (var ii=0u; ii<node.count; ii++) {
   let inst_id=order[node.offset+ii]; let inst=instances[inst_id];
   let Minv=mat4x4<f32>(inst.inv0,inst.inv1,inst.inv2,inst.inv3);
   let oo=(Minv*vec4<f32>(o,1.)).xyz; let dd=(Minv*vec4<f32>(d,0.)).xyz;
   var bstack: array<i32,64>; bstack[0]=i32(inst.info.x); var bsize=1u;
   loop {
    if (bsize==0u) { break; }
    bsize--; let bindex=bstack[bsize]; let bnode=nodes[bindex];
    if (entry(bnode.lo,bnode.hi,oo,dd,lo,best.t)==bitcast<f32>(0x7f800000u)) { continue; }
    if (bnode.count==0u) { bstack[bsize]=bnode.left; bstack[bsize+1u]=bnode.right; bsize+=2u; continue; }
    for (var p=0u;p<bnode.count;p++) {
     let tri_id=order[bnode.offset+p]; let h=intersect(tri_id,oo,dd,lo,best.t);
     if (h.tri>=0 && h.t<best.t) { best=h; hit_instance=i32(inst_id); }
    }
   }
  }
 }
 return best;
}
fn visibility(o: vec3<f32>, d: vec3<f32>, limit: f32, near_bias: f32) -> f32 {
 var transmission=1.;
 var stack: array<i32,64>; stack[0]=0; var size=1u;
 loop {
  if (size==0u) { break; }
  size--; let index=stack[size]; let node=nodes[index];
  if (entry(node.lo,node.hi,o,d,near_bias,limit)==bitcast<f32>(0x7f800000u)) { continue; }
  if (node.count==0u) { stack[size]=node.left; stack[size+1u]=node.right; size+=2u; continue; }
  for (var ii=0u; ii<node.count; ii++) {
   let inst_id=order[node.offset+ii]; let inst=instances[inst_id];
   let Minv=mat4x4<f32>(inst.inv0,inst.inv1,inst.inv2,inst.inv3);
   let oo=(Minv*vec4<f32>(o,1.)).xyz; let dd=(Minv*vec4<f32>(d,0.)).xyz;
   var bstack: array<i32,64>; bstack[0]=i32(inst.info.x); var bsize=1u;
   loop {
    if (bsize==0u) { break; }
    bsize--; let bindex=bstack[bsize]; let bnode=nodes[bindex];
    if (entry(bnode.lo,bnode.hi,oo,dd,near_bias,limit)==bitcast<f32>(0x7f800000u)) { continue; }
    if (bnode.count==0u) { bstack[bsize]=bnode.left; bstack[bsize+1u]=bnode.right; bsize+=2u; continue; }
    for (var p=0u;p<bnode.count;p++) {
     let tri_id=order[bnode.offset+p]; let h=intersect(tri_id,oo,dd,near_bias,limit);
     if (h.tri>=0) { transmission*=1.-triangles[tri_id].v0.w; if (transmission<.001) { return 0.; } }
    }
   }
  }
 }
 return transmission;
}
fn seed_hash(p: vec3<f32>) -> u32 {
 var h=(bitcast<u32>(p.x)*73856093u)^(bitcast<u32>(p.y)*19349663u)^(bitcast<u32>(p.z)*83492791u);
 h^=h>>16u; h*=0x85ebca6bu; h^=h>>13u; h*=0xc2b2ae35u; h^=h>>16u;
 return h;
}
fn soft_visibility(o: vec3<f32>, d: vec3<f32>, limit: f32, near_bias: f32, lp: vec4<f32>, ls: vec4<f32>) -> f32 {
 if (ls.y<=0.) { return visibility(o,d,limit,near_bias); }
 let count=max(u32(ls.z),1u);
 let phi=f32(seed_hash(o))*(6.2831853/4294967296.);
 let axis=select(vec3<f32>(1.,0.,0.),vec3<f32>(0.,1.,0.),abs(d.y)<.9);
 let a=normalize(cross(d,axis)); let b=cross(d,a);
 var total=0.;
 for (var k=0u;k<count;k++) {
  let r=sqrt((f32(k)+.5)/f32(count)); let theta=f32(k)*2.3999632+phi;
  let spread=a*(r*cos(theta))+b*(r*sin(theta));
  var jd=normalize(d+spread*ls.y); var jlimit=limit;
  if (lp.w>0.) { let delta=lp.xyz+spread*(limit*ls.y)-o; jlimit=length(delta); jd=delta/max(jlimit,1e-8); }
  total+=visibility(o,jd,jlimit,near_bias);
 }
 return total/f32(count);
}
fn attenuation(lp: vec4<f32>, ld: vec4<f32>, cone: vec4<f32>, power: f32, point: vec3<f32>) -> f32 {
 if (lp.w<=0.) { return 1.; }
 let offset=point-lp.xyz; let dist=length(offset); var result=1.;
 if (power>0.) { result=min(1.,pow(max(dist,1e-8),-power)); }
 if (cone.x>0.) {
  let angle=degrees(acos(clamp(dot(offset,ld.xyz)/max(dist,1e-12),-1.,1.)));
  var k=select(0.,1.,angle<=cone.y);
  if (cone.z>cone.y) { let t=clamp((cone.z-angle)/(cone.z-cone.y),0.,1.); k=select(0.,pow(t*t*(3.-2.*t),cone.w),t>0.); }
  result=result*k;
 }
 return result;
}
// scene3d._shade_solid's formula, one surface per ray (no OIT peel -- see module docstring).
fn shade(hit: Hit, inst_id: i32, origin: vec3<f32>) -> vec4<f32> {
 let inst=instances[u32(inst_id)];
 let at=attrs[hit.tri]; let tr=triangles[hit.tri];
 let w=vec3<f32>(1.-hit.u-hit.v,hit.u,hit.v);
 let local_pos=tr.v0.xyz+hit.u*tr.e1.xyz+hit.v*tr.e2.xyz;
 let local_normal=at.n0.xyz*w.x+at.n1.xyz*w.y+at.n2.xyz*w.z;
 let M=mat4x4<f32>(inst.matrix0,inst.matrix1,inst.matrix2,inst.matrix3);
 let position=(M*vec4<f32>(local_pos,1.)).xyz;
 let Linv=mat3x3<f32>(inst.inv0.xyz,inst.inv1.xyz,inst.inv2.xyz);
 var normal=unit(transpose(Linv)*local_normal);
 if (dot(normal,origin-position)<0.) { normal=-normal; }
 if (params.output==1u) { return vec4<f32>(vec3<f32>(hit.t),1.); }
 if (params.output==2u) { return vec4<f32>(normal,1.); }
 if (params.output==7u) { return vec4<f32>(position,1.); }
 if (params.output==9u) { return vec4<f32>(inst.info.z,0.,0.,1.); }
 let material=u32(inst.info.y);
 var source=table[material]*inst.tint;
 let properties=table[material+1u];
 if (params.output==3u) { return source; }
 let emission=source.xyz*properties.z;
 if (params.output==6u) { return vec4<f32>(emission,source.w); }
 var radiance=vec3<f32>(params.ambient); var specular=vec3<f32>(0.);
 let toward=unit(origin-position);
 for (var j=0u;j<params.lights;j++) {
  let start=params.light_offset+j*5u;
  let lp=table[start]; let ld=table[start+1u]; let lc=table[start+2u]; let lk=table[start+3u]; let ls=table[start+4u];
  let factor=attenuation(lp,ld,lk,lc.w,position);
  var to_light=-ld.xyz;
  if (lp.w>0.) { to_light=unit(lp.xyz-position); }
  let lambert=dot(normal,to_light); var vis=1.;
  if (ld.w>0.) {
   let bias=params.bias*ls.x; let near_bias=bias*.01;
   let o=position+normal*bias; var d=-ld.xyz; var limit=bitcast<f32>(0x7f800000u);
   if (lp.w>0.) { let delta=lp.xyz-o; limit=length(delta); d=delta/max(limit,1e-8); }
   vis=soft_visibility(o,d,limit,near_bias,lp,ls);
  }
  radiance+=max(lambert*vis,0.)*factor*lc.xyz;
  if (params.output==0u || params.output==5u) {
   let lobe=pow(max(dot(normal,unit(to_light+toward)),0.),properties.y);
   specular+=properties.x*lobe*select(0.,1.,lambert>0.)*vis*factor*lc.xyz;
  }
 }
 if (params.output==5u) { return vec4<f32>(specular*source.w,source.w); }
 var rgb=source.xyz*radiance+specular*source.w;
 if (params.output==0u) { rgb+=emission; }
 return vec4<f32>(rgb,source.w);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 let r=(group.y*params.gx+group.x)*64u+lane;
 if (r>=params.count) { return; }
 let origin=rays[r*4u]; let direction=rays[r*4u+1u];
 let hit=trace(origin.xyz,direction.xyz,origin.w,direction.w);
 var accum=vec4<f32>(0.);
 if (hit.tri>=0) { accum=shade(hit,hit_instance,origin.xyz); }
 rays[r*4u+2u]=accum;
}
'''


def _pipeline(state):
    if '_gpuinstance_pipeline' not in state:
        device = state['device']
        state['_gpuinstance_pipeline'] = device.create_compute_pipeline(
            layout='auto', compute={'module': device.create_shader_module(code=_SHADER), 'entry_point': 'main'})
    return state['_gpuinstance_pipeline']


def _check_scope(scene):
    if scene.geometries or scene.splats or scene.particles or getattr(scene, 'volumes', ()):
        raise gpu3d.Unsupported('GPU instanced ray tracing needs a scene made entirely of Instance3D '
                                'instances for now (no ordinary geometry, splats, particles or volumes '
                                'alongside them)')
    if getattr(scene, 'environments', ()):
        raise gpu3d.Unsupported('environment light on instances is CPU-only')
    for instance_set in scene.instances:
        for source in instance_set.sources:
            if source.texture is not None:
                raise gpu3d.Unsupported('textured instance sources are CPU-only for now')
            if source.material == 'liquid':
                raise gpu3d.Unsupported('liquid instance sources are CPU-only')
            if source.projection is not None:
                raise gpu3d.Unsupported('camera-projected instance sources are CPU-only')


def _prepare(scene, cancel):
    """Build the two-level acceleration structure and every buffer the shader reads.

    Returns None for an instance-free scene (every source empty or no instances). Otherwise
    (nodes, order, triangles, attrs, instances, table, light_count, light_offset, bias).
    """
    _check_scope(scene)
    source_order, sources_by_id = [], {}
    for instance_set in scene.instances:
        for source in instance_set.sources:
            key = id(source)
            if key not in sources_by_id:
                sources_by_id[key] = len(source_order)
                source_order.append(source)
    local_aabb = []
    for source in source_order:
        raytrace._cancel(cancel)
        tri = source.triangles
        v = source.vertices.astype(np.float64)[tri].reshape(-1, 3) if len(tri) else np.zeros((0, 3))
        local_aabb.append((v.min(0), v.max(0)) if len(v) else (np.zeros(3), np.zeros(3)))
    instance_records, inst_lo, inst_hi = [], [], []
    next_id = 1
    for instance_set in scene.instances:
        if not len(instance_set) or not instance_set.sources:
            continue
        parent = instance_set.parent.astype(np.float64)
        bases = [src.world_matrix().astype(np.float64) for src in instance_set.sources]
        for i in range(len(instance_set)):
            raytrace._cancel(cancel) if i % 256 == 0 else None
            variant = int(instance_set.variant[i])
            source = instance_set.sources[variant]
            source_idx = sources_by_id[id(source)]
            matrix = parent @ instance_set.matrices[i] @ bases[variant]
            tint = (instance_set.colors[i].astype(np.float64) if instance_set.colors is not None
                   else np.ones(4, np.float64))
            lo, hi = local_aabb[source_idx]
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            world_corners = (matrix[:3, :3] @ corners.T).T + matrix[:3, 3]
            inst_lo.append(world_corners.min(0)); inst_hi.append(world_corners.max(0))
            instance_records.append((matrix, source_idx, tint, next_id))
            next_id += 1
    if not instance_records:
        return None
    tlas = raytrace.Bvh.build(np.array(inst_lo), np.array(inst_hi), cancel=cancel)
    tlas_nodes, tlas_order = gpu3d._pack_bvh(tlas, cancel)
    combined_nodes, combined_order = [tlas_nodes], [tlas_order]
    node_base, order_base, tri_running = len(tlas_nodes), len(tlas_order), 0
    tri_parts, attr_parts, table = [], [], []
    blas_root_by_source = []
    for source in source_order:
        raytrace._cancel(cancel)
        tri = source.triangles
        v = source.vertices.astype(np.float64)
        v0l, v1l, v2l = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        e1l, e2l = v1l - v0l, v2l - v0l
        alpha = float(np.clip(source.color[3], 0, 1))
        triset = raytrace.TriangleSet(v0l, e1l, e2l, np.full(len(tri), alpha, 'f4'))
        bvh = raytrace.Bvh.build(*triset.aabbs(), cancel=cancel)
        nodes, order = gpu3d._pack_bvh(bvh, cancel)
        blas_root_by_source.append(node_base)
        internal = nodes['count'] == 0
        nodes['left'] = np.where(internal, nodes['left'] + node_base, -1)
        nodes['right'] = np.where(internal, nodes['right'] + node_base, -1)
        nodes['offset'] = np.where(~internal, nodes['offset'] + order_base, nodes['offset'])
        combined_nodes.append(nodes); combined_order.append(order + tri_running)
        node_base += len(nodes); order_base += len(order)
        packed = np.zeros((max(1, len(tri)), 3, 4), 'f4')
        if len(tri):
            packed[:, 0, :3], packed[:, 1, :3], packed[:, 2, :3] = v0l, e1l, e2l
            packed[:, 0, 3] = alpha
        tri_parts.append(packed)
        if source.normals is None:
            face = np.cross(e1l, e2l)
            norm = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-8)
            n0 = n1 = n2 = norm
        else:
            normals = source.normals.astype(np.float64)
            n0, n1, n2 = normals[tri[:, 0]], normals[tri[:, 1]], normals[tri[:, 2]]
        attrs = np.zeros((max(1, len(tri)), 3, 4), 'f4')
        if len(tri):
            attrs[:, 0, :3], attrs[:, 1, :3], attrs[:, 2, :3] = n0, n1, n2
        attr_parts.append(attrs)
        tint = np.asarray(source.color, 'f4').copy(); tint[:3] *= tint[3]
        table.append(tint); table.append((source.specular, source.shininess, source.emission, 0.0))
        tri_running += len(tri)
    light_offset = len(table)
    lights = [light for light in scene.lights if light.intensity > 0]
    for light in lights:
        position, direction = light.world()
        table.extend([(*position, light.kind in s._POSITIONAL), (*direction, light.shadows),
                      (*(np.asarray(light.color) * light.intensity), s._falloff_power(light)),
                      tuple(s._cone_terms(light)), (*s._shadow_terms(light), 0)])
    instances = np.zeros((len(instance_records), 10, 4), 'f4')
    for i, (matrix, source_idx, tint, object_id) in enumerate(instance_records):
        inv = np.linalg.inv(matrix)
        instances[i, 0:4] = matrix.T.astype('f4')
        instances[i, 4:8] = inv.T.astype('f4')
        instances[i, 8] = tint.astype('f4')
        instances[i, 9] = (blas_root_by_source[source_idx], 2 * source_idx, float(object_id), 0.0)
    all_lo, all_hi = np.array(inst_lo), np.array(inst_hi)
    extent = float((all_hi.max(0) - all_lo.min(0)).max())
    bias = 1e-3 * max(1.0, extent)
    return (np.concatenate(combined_nodes), np.concatenate(combined_order).astype('u4'),
            np.concatenate(tri_parts), np.concatenate(attr_parts), instances,
            np.asarray(table, 'f4'), len(lights), light_offset, bias)


def render_beauty(state, scene, camera, width, height, background, ambient, samples=1, cancel=None):
    return render(state, scene, camera, width, height, background, ambient, 'rgba', samples, cancel)


def render(state, scene, camera, width, height, background, ambient, output='rgba', samples=1, cancel=None):
    """Two-level GPU ray trace of a pure-instance scene; see the module docstring for scope."""
    if output not in s.RENDER_OUTPUTS:
        raise ValueError(f'Unknown 3D render output {output!r}')
    if output not in SUPPORTED_OUTPUTS:
        raise gpu3d.Unsupported(f'GPU instanced ray tracing does not implement the {output!r} output yet')
    raytrace._cancel(cancel)
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError('Render dimensions must be positive')
    samples = 1 if output in s.DATA_OUTPUTS else max(1, min(int(samples), 4))
    iw, ih = width * samples, height * samples
    prepared = _prepare(scene, cancel)
    bg = np.asarray(background, 'f4').copy(); bg[3] = np.clip(bg[3], 0, 1); bg[:3] *= bg[3]
    if prepared is None:
        result = np.broadcast_to(bg if output == 'rgba' else 0, (height, width, 4)).astype('f4').copy()
        result.flags.writeable = False
        return result
    nodes, order, triangles, attrs, instances, table, lights, light_offset, bias = prepared
    reason = check_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    gpurt._memory_check(state, max(nodes.nbytes, attrs.nbytes, instances.nbytes, table.nbytes, iw * 64))
    dimension = min(65535, gpurt._limits(state)['max-compute-workgroups-per-dimension'])
    budget = min(int(GPU_RT_RAYS_PER_SUBMISSION), gpurt._cap(state) // 64, dimension * dimension * 64)
    if budget < iw:
        raise ValueError(f'GPU ray tracing needs about {iw*64/2**20:.3f} MiB for one row; submission ray limit is {budget}')
    rows = max(1, budget // iw)
    result = np.empty((ih, iw, 4), 'f4')
    device, wgpu = state['device'], state['wgpu']
    resources = []

    def upload(data, usage):
        resource = device.create_buffer_with_data(data=data, usage=usage)
        resources.append(resource)
        return resource

    try:
        persistent = [upload(data, wgpu.BufferUsage.STORAGE) for data in (nodes, order, triangles, attrs, instances, table)]
        pipeline = _pipeline(state)
        for y0 in range(0, ih, rows):
            raytrace._cancel(cancel)
            y1 = min(ih, y0 + rows)
            o, d, lo, hi = gpurt.primary_rays(camera, iw, ih, rows=(y0, y1))
            n = len(o); raw = np.zeros((n, 4, 4), 'f4')
            raw[:, 0, :3], raw[:, 0, 3] = o, lo
            raw[:, 1, :3], raw[:, 1, 3] = d, hi
            groups = (n + 63) // 64; gx = min(dimension, groups); gy = (groups + gx - 1) // gx
            params = np.array([n, gx, lights, light_offset, 0, 0, s.RENDER_OUTPUTS.index(output), 0], 'u4')
            params.view('f4')[4:6] = ambient, bias
            mark = len(resources)
            try:
                io = upload(raw, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
                uniform = upload(params, wgpu.BufferUsage.UNIFORM)
                bindings = [*persistent, io, uniform]
                group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                    {'binding': i, 'resource': {'buffer': b}} for i, b in enumerate(bindings)])
                encoder = device.create_command_encoder(); compute = encoder.begin_compute_pass()
                compute.set_pipeline(pipeline); compute.set_bind_group(0, group)
                compute.dispatch_workgroups(gx, gy, 1); compute.end()
                raytrace._cancel(cancel)
                device.queue.submit([encoder.finish()])
                read = np.frombuffer(device.queue.read_buffer(io), 'f4').reshape(n, 4, 4)[:, 2]
                raytrace._cancel(cancel)
                result[y0:y1] = read.reshape(y1 - y0, iw, 4)
            finally:
                for resource in reversed(resources[mark:]):
                    resource.destroy()
                del resources[mark:]
    finally:
        for resource in reversed(resources):
            resource.destroy()
    if samples > 1:
        result = result.reshape(height, samples, width, samples, 4).mean(axis=(1, 3))
    if output == 'rgba':
        result = result + bg * (1 - result[..., 3:4])
    result = np.ascontiguousarray(result, 'f4')
    result.flags.writeable = False
    return result
