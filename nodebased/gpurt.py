"""Compute BVH queries. Distances/barycentrics are f32, exposed as f64.

Moller-Trumbore uses |det| > 1e-10 and an inclusive 1e-6 barycentric
edge band (dimensionless, relative to triangle edges). Bounds and cursors
are converted to f32 too; peeling reuses the exact returned f32 distances.
Four storage bindings work with gpu3d's existing device configuration.
"""
import math
import numpy as np
from . import gpu3d, raytrace

STACK_SIZE = 64
HIT_DTYPE = np.dtype([('t', 'f8'), ('primitive', 'i4'), ('u', 'f8'), ('v', 'f8')])


def _limits(state):
    return state.get('limits', getattr(state.get('device'), 'limits', {}))


def check_capability(state):
    """Return a fallback reason, including compute compilation failures."""
    for name, minimum in [('max-storage-buffers-per-shader-stage', 4),
                          ('max-storage-buffer-binding-size', 64),
                          ('max-buffer-size', 64),
                          ('max-compute-invocations-per-workgroup', 64),
                          ('max-compute-workgroup-size-x', 64),
                          ('max-compute-workgroups-per-dimension', 1)]:
        if _limits(state).get(name, 0) < minimum:
            return f'GPU ray tracing unavailable: {name} too small (needs {minimum})'
    if 'device' in state and 'wgpu' in state:
        try:
            _pipeline(state)
        except Exception as exc:
            return f'GPU ray tracing requires compute storage buffers: {exc}'
    return None


_SHADER = r'''
struct Node { lo: vec3<f32>, left: i32, hi: vec3<f32>, right: i32,
 offset: u32, count: u32, pad: vec2<u32> };
struct Triangle { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32> };
struct Params { n: u32, k: u32, gx: u32, cursor: u32 };
struct Hit { t: f32, primitive: i32, u: f32, v: f32 };
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> order: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read_write> data: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
fn f(i: u32) -> f32 { return bitcast<f32>(data[i]); }
// Explicit parallel slabs avoid 0 * infinity and preserve boundary rays.
fn entry(index: i32, o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32) -> f32 {
 let node = nodes[index]; var near = lower; var far = upper;
 for (var a = 0u; a < 3u; a++) {
  if (d[a] == 0.0) {
   if (o[a] < node.lo[a] || o[a] > node.hi[a]) { return bitcast<f32>(0x7f800000u); }
  } else {
   let x = (node.lo[a]-o[a])/d[a]; let y = (node.hi[a]-o[a])/d[a];
   near = max(near, min(x,y)); far = min(far, max(x,y));
  }
 }
 if (near > far) { return bitcast<f32>(0x7f800000u); }
 return near;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 let ray = (group.y*params.gx+group.x)*64u+lane;
 if (ray >= params.n) { return; }
 let base = ray*(12u+4u*params.k);
 let o = vec3<f32>(f(base),f(base+1u),f(base+2u)); let lower = f(base+3u);
 let d = vec3<f32>(f(base+4u),f(base+5u),f(base+6u)); let upper = f(base+7u);
 let ct = f(base+8u); let cp = bitcast<i32>(data[base+9u]);
 let inf = bitcast<f32>(0x7f800000u);
 var best: array<Hit,8>;
 for (var j=0u; j<8u; j++) { best[j] = Hit(inf,2147483647,0.0,0.0); }
 var stack: array<i32,64>; stack[0]=0; var size=1u;
 loop {
  if (size == 0u) { break; }
  size--; let index=stack[size]; let bound=min(upper,best[params.k-1u].t);
  if (entry(index,o,d,lower,bound) == inf) { continue; }
  let node=nodes[index];
  if (node.count == 0u) {
   let a=entry(node.left,o,d,lower,bound); let b=entry(node.right,o,d,lower,bound);
   var near=node.left; var far=node.right; var nt=a; var ft=b;
   if (b<a) { near=node.right; far=node.left; nt=b; ft=a; }
   if (ft != inf) { stack[size]=far; size++; }
   if (nt != inf) { stack[size]=near; size++; }
  } else {
   for (var p=0u; p<node.count; p++) {
    let id=order[node.offset+p]; let tri=triangles[id];
    let h=cross(d,tri.e2.xyz); let det=dot(h,tri.e1.xyz);
    if (abs(det)<=1e-10) { continue; }
    let inv=1.0/det; let delta=o-tri.v0.xyz;
    let u=dot(delta,h)*inv; let q=cross(delta,tri.e1.xyz);
    let v=dot(d,q)*inv; let t=dot(tri.e2.xyz,q)*inv;
    if (!(u>=-1e-6 && v>=-1e-6 && u+v<=1.000001 && t>lower && t<upper)) { continue; }
    if (params.cursor!=0u && !(t>ct || (t==ct && i32(id)>cp))) { continue; }
    var candidate=Hit(t,i32(id),u,v);
    for (var j=0u; j<params.k; j++) {
     if (candidate.t<best[j].t || (candidate.t==best[j].t && candidate.primitive<best[j].primitive)) {
      let old=best[j]; best[j]=candidate; candidate=old;
     }
    }
   }
  }
 }
 for (var j=0u; j<params.k; j++) {
  let dst=base+12u+j*4u; let hit=best[j];
  data[dst]=bitcast<u32>(hit.t); data[dst+1u]=bitcast<u32>(select(hit.primitive,-1,hit.t==inf));
  data[dst+2u]=bitcast<u32>(hit.u); data[dst+3u]=bitcast<u32>(hit.v);
 }
}
'''


def _pipeline(state):
    if '_gpurt_pipeline' not in state:
        device = state['device']
        state['_gpurt_pipeline'] = device.create_compute_pipeline(layout='auto', compute={
            'module': device.create_shader_module(code=_SHADER), 'entry_point': 'main'})
    return state['_gpurt_pipeline']


def _cap(state):
    limits = _limits(state)
    return min(limits.get('max-buffer-size', 0), limits.get('max-storage-buffer-binding-size', 0))


def _memory_check(state, needed):
    cap = _cap(state)
    if needed > cap:
        raise ValueError(f'GPU ray tracing needs about {needed/2**20:.3f} MiB per buffer; adapter allows {cap/2**20:.3f} MiB')


class GpuTriangleScene:
    """Own three immutable uploads; close is idempotent, including after errors."""
    def __init__(self, state, triangles, bvh):
        self.state, self.buffers, self.closed = state, [], False
        self.empty = not len(bvh.left)
        pending = [(0, 1)] if not self.empty else []
        while pending:
            node, depth = pending.pop()
            if depth > min(STACK_SIZE, 64):
                raise ValueError(f'GPU ray tracing BVH depth {depth} exceeds traversal stack {min(STACK_SIZE, 64)}')
            if not bvh.prim_count[node]:
                pending.extend(((int(bvh.left[node]), depth+1), (int(bvh.right[node]), depth+1)))
        _memory_check(state, max(48, len(bvh.left)*48, len(bvh.prim_order)*4, len(triangles.v0)*48))
        reason = check_capability(state)
        if reason:
            raise gpu3d.Unsupported(reason)
        nodes, order = gpu3d._pack_bvh(bvh)
        packed = np.zeros((max(1, len(triangles.v0)), 3, 4), 'f4')
        if len(triangles.v0):
            packed[:, 0, :3], packed[:, 1, :3], packed[:, 2, :3] = triangles.v0, triangles.e1, triangles.e2
            packed[:, 0, 3] = triangles.alpha
        try:
            for data in (nodes if len(nodes) else np.zeros(48, 'u1'),
                         order if len(order) else np.zeros(1, 'u4'), packed):
                self.buffers.append(state['device'].create_buffer_with_data(data=data, usage=state['wgpu'].BufferUsage.STORAGE))
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            for buffer in self.buffers:
                buffer.destroy()
            self.buffers.clear()

    def __enter__(self):
        if self.closed:
            raise ValueError('GPU ray tracing scene is closed')
        return self

    def __exit__(self, *args):
        self.close()


def nearest_hits(scene, origins, dirs, tmin, tmax, k, *, after_t=None,
                 after_primitive=None, cancel=None, chunk=1 << 20):
    """Fixed (N,K) structured result; missing slots have t=inf, primitive=-1."""
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= 8 or chunk < 1:
        raise ValueError('k must be in 1..8 and chunk must be positive')
    if scene.closed:
        raise ValueError('GPU ray tracing scene is closed')
    if (after_t is None) != (after_primitive is None):
        raise ValueError('after_t and after_primitive must be supplied together')
    origins, dirs = np.asarray(origins), np.asarray(dirs)
    if origins.ndim != 2 or origins.shape[1] != 3 or dirs.shape != origins.shape:
        raise ValueError('rays must have matching (N, 3) shapes')
    n = len(origins)
    lo, hi = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
    ct = None if after_t is None else np.broadcast_to(after_t, (n,))
    cp = None if ct is None else np.broadcast_to(after_primitive, (n,))
    result = np.zeros((n, k), HIT_DTYPE)
    result['t'], result['primitive'] = np.inf, -1
    raytrace._cancel(cancel)
    if not n or scene.empty:
        return result
    state = scene.state
    device, wgpu = state['device'], state['wgpu']
    stride = 12+4*k
    _memory_check(state, stride*4)
    dimension = min(65535, _limits(state)['max-compute-workgroups-per-dimension'])
    chunk = min(int(chunk), _cap(state)//(stride*4), dimension*dimension*64)
    pipeline = _pipeline(state)
    for start in range(0, n, chunk):
        raytrace._cancel(cancel)
        stop = min(n, start+chunk); count = stop-start
        raw = np.zeros((count, stride), 'u4'); floats = raw.view('f4')
        floats[:, :3], floats[:, 3] = origins[start:stop], lo[start:stop]
        floats[:, 4:7], floats[:, 7] = dirs[start:stop], hi[start:stop]
        if ct is not None:
            floats[:, 8], raw[:, 9] = ct[start:stop], cp[start:stop].astype('i4').view('u4')
        groups = (count+63)//64; gx = min(dimension, groups); gy = (groups+gx-1)//gx
        resources = []
        try:
            def buffer(**kwargs):
                b = device.create_buffer_with_data(**kwargs); resources.append(b); return b
            io = buffer(data=raw, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            uniform = buffer(data=np.array([count, k, gx, ct is not None], 'u4'), usage=wgpu.BufferUsage.UNIFORM)
            group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                dict(binding=i, resource={'buffer': b}) for i, b in enumerate([*scene.buffers, io, uniform])])
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass(); compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group); compute.dispatch_workgroups(gx, gy, 1); compute.end()
            raytrace._cancel(cancel)
            device.queue.submit([encoder.finish()])
            read = np.frombuffer(device.queue.read_buffer(io), 'u4').reshape(count, stride)[:, 12:].reshape(count, k, 4)
            raytrace._cancel(cancel)
            result['t'][start:stop] = read[:, :, 0].view('f4')
            result['primitive'][start:stop] = read[:, :, 1].view('i4')
            result['u'][start:stop] = read[:, :, 2].view('f4')
            result['v'][start:stop] = read[:, :, 3].view('f4')
        finally:
            for resource in reversed(resources):
                resource.destroy()
    return result


def all_hits(scene, origins, dirs, tmin=0., tmax=np.inf, *, max_hits=64, k=8, cancel=None):
    """Peel lexicographic cursors, including coincident primitive hits."""
    if max_hits < 1 or not 1 <= k <= 8:
        raise ValueError('max_hits must be positive and k must be in 1..8')
    origins, dirs = np.asarray(origins), np.asarray(dirs)
    n = len(origins); active = np.arange(n)
    lo, hi = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
    ct, cp = np.full(n, -np.inf), np.full(n, -1, 'i4')
    batches = [[] for _ in range(n)]; counts = np.zeros(n, int)
    raytrace._cancel(cancel)
    while len(active):
        hits = nearest_hits(scene, origins[active], dirs[active], lo[active], hi[active], k,
                            after_t=ct[active], after_primitive=cp[active], cancel=cancel)
        more = []
        for index, row in zip(active, hits):
            valid = row[row['primitive'] >= 0]; counts[index] += len(valid)
            if counts[index] > max_hits:
                raise ValueError(f'Ray-traced render exceeds MAX_HITS_PER_RAY ({max_hits})')
            batches[index].append(valid)
            if len(valid) == k:
                ct[index], cp[index] = valid[-1]['t'], valid[-1]['primitive']; more.append(index)
        active = np.asarray(more, int)
    return [np.concatenate(b) if b else np.empty(0, HIT_DTYPE) for b in batches]


def primary_rays(camera, width, height, rows=None):
    """Pixel-centre rays using the renderer's actual inverse f32 view basis."""
    from .scene3d import _view_basis
    if width < 1 or height < 1:
        raise ValueError('Render dimensions must be positive')
    y0, y1 = (0, height) if rows is None else rows
    if not 0 <= y0 <= y1 <= height:
        raise ValueError('invalid row interval')
    eye, view = _view_basis(camera)
    inverse_view = np.linalg.inv(view.astype(np.float64))
    focal = 1 / math.tan(math.radians(camera.fov)*.5)
    aspect = width/height
    pixels = np.arange(width*(y1-y0))
    x, y = pixels % width+.5, pixels//width+y0+.5
    local_dirs = np.column_stack(((2*x/width-1)*aspect/focal,
                                 (1-2*y/height)/focal, -np.ones(len(pixels))))
    dirs = local_dirs @ inverse_view.T
    return np.broadcast_to(eye, dirs.shape), dirs, np.full(len(dirs), camera.near), np.full(len(dirs), camera.far)
