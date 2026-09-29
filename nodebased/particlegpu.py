"""GPU uniform-grid self-collision for ParticleCollide3D.

Each particle owns its output. A hashed uniform grid is rebuilt on the adapter for every
relaxation pass; the CPU sends only particle state and receives the final state. Hash hits
are checked against actual cell coordinates, so hash collisions cannot become contacts.
"""
from __future__ import annotations

import math
import numpy as np

from . import gpu3d

_SHADER = r'''
struct Params { count: u32, buckets: u32, cell_size: f32, restitution: f32,
                friction: f32, sleep: f32, iteration: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> heads: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> links: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_pos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_vel: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> cfg: Params;
@group(0) @binding(7) var<storage, read> ids: array<u32>;
fn cell(p: vec3<f32>) -> vec3<i32> { return vec3<i32>(floor(p / cfg.cell_size)); }
fn bucket(c: vec3<i32>) -> u32 {
 let x=bitcast<u32>(c.x); let y=bitcast<u32>(c.y); let z=bitcast<u32>(c.z);
 return ((x*73856093u) ^ (y*19349663u) ^ (z*83492791u)) & (cfg.buckets-1u);
}
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
 if (gid.x < cfg.buckets) { atomicStore(&heads[gid.x], 0xffffffffu); }
}
@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i=gid.x; if (i>=cfg.count) { return; }
 let b=bucket(cell(pos[i].xyz));
 links[i]=atomicExchange(&heads[b], i);
}
fn coincident(a:u32,b:u32)->vec3<f32> {
 let lo=min(a,b); let hi=max(a,b);
 var h=(lo*1664525u)^(hi*1013904223u);
 h^=h>>16u; h*=2246822519u; h^=h>>13u;
 let angle=f32(h)*(6.28318530718/4294967296.0);
 let dir=vec3<f32>(cos(angle),sin(angle),0.0);
 return select(-dir,dir,a==lo);
}
@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i=gid.x; if (i>=cfg.count) { return; }
 let pi=pos[i]; let vi=vel[i]; let ci=cell(pi.xyz);
 // Quantized integer sums are associative, so the atomic linked-list traversal order
 // cannot change the result. The 2^20 scale keeps contact corrections below 1e-6.
 var dp_acc=vec3<i32>(0); var dv_acc=vec3<i32>(0); var contacts=0u;
 for (var x=-1; x<=1; x++) { for (var y=-1; y<=1; y++) { for (var z=-1; z<=1; z++) {
  let neighbor_cell=ci+vec3<i32>(x,y,z);
  var j=atomicLoad(&heads[bucket(neighbor_cell)]);
  loop {
   if (j==0xffffffffu) { break; }
   let pj=pos[j];
   if (j!=i && all(cell(pj.xyz)==neighbor_cell)) {
    let delta=pi.xyz-pj.xyz; let dist=length(delta);
    let overlap=pi.w+pj.w-dist;
    if (overlap>0.0) {
     let normal=select(coincident(ids[i],ids[j]),delta/max(dist,1e-20),dist>1e-9);
     dp_acc+=vec3<i32>(round(0.5*overlap*normal*1048576.0));
     let relative=vi.xyz-vel[j].xyz;
     let vn=dot(relative,normal);
     let impulse=max(0.0,-(1.0+cfg.restitution)*0.5*vn);
     let tangential=relative-vn*normal;
     let speed=length(tangential);
     let scale=max(0.0,1.0-cfg.friction*impulse/max(speed,1e-12));
     dv_acc+=vec3<i32>(round((impulse*normal+0.5*tangential*(scale-1.0))*1048576.0));
     contacts++;
    }
   }
   j=links[j];
  }
 } } }
 var p=pi.xyz; var v=vi.xyz;
 if (contacts>0u) {
  p+=vec3<f32>(dp_acc)/(1048576.0*f32(contacts));
  v+=vec3<f32>(dv_acc)/(1048576.0*f32(contacts));
  if (length(v)<cfg.sleep) { v=vec3<f32>(0.0); }
 }
 out_pos[i]=vec4<f32>(p,pi.w);
 out_vel[i]=vec4<f32>(v,vi.w);
}
'''


def available():
    try:
        gpu3d._state()
        return True
    except Exception:
        return False


def resolve(ids, position, velocity, radius, *, iterations, restitution, friction, sleep_threshold):
    """Return GPU-resolved positions and velocities. Raise if the adapter cannot run the kernel."""
    import wgpu
    state = gpu3d._state()
    device = state['device']
    count = len(position)
    if count < 2:
        return position.copy(), velocity.copy()
    buckets = 1 << max(10, (count * 2 - 1).bit_length())
    cell_size = max(float(np.max(radius)) * 2.0, 1e-6)
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    def buffer(data):
        return device.create_buffer_with_data(data=np.ascontiguousarray(data).tobytes(), usage=usage)
    packed_pos = np.column_stack((position, radius)).astype('f4')
    packed_vel = np.column_stack((velocity, np.zeros(count, 'f4'))).astype('f4')
    pbuf = [buffer(packed_pos), device.create_buffer(size=packed_pos.nbytes, usage=usage)]
    vbuf = [buffer(packed_vel), device.create_buffer(size=packed_vel.nbytes, usage=usage)]
    heads = device.create_buffer(size=buckets * 4, usage=usage)
    links = device.create_buffer(size=count * 4, usage=usage)
    idbuf = buffer(np.asarray(ids, 'u4'))
    pipelines = state.get('_particle_collision_pipelines')
    if pipelines is None:
        module = device.create_shader_module(code=_SHADER)
        pipelines = [device.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': name})
                     for name in ('clear', 'build', 'solve')]
        state['_particle_collision_pipelines'] = pipelines
    encoder = device.create_command_encoder()
    params_buffers = []
    for iteration in range(max(1, int(iterations))):
        import struct
        params = device.create_buffer_with_data(data=struct.pack('IIffffII', count, buckets, cell_size,
            float(restitution), float(friction), float(sleep_threshold), iteration, 0),
            usage=wgpu.BufferUsage.UNIFORM)
        params_buffers.append(params)
        bindings = []
        for binding, resource in enumerate((pbuf[iteration % 2], vbuf[iteration % 2], heads, links,
                                            pbuf[(iteration + 1) % 2], vbuf[(iteration + 1) % 2],
                                            params, idbuf)):
            bindings.append({'binding': binding, 'resource': {'buffer': resource}})
        # Automatic pipeline layouts are entry-point specific on some backends. Bind groups
        # are therefore created for each pipeline, using the same resources.
        groups = [device.create_bind_group(layout=p.get_bind_group_layout(0),
                  entries=[bindings[k] for k in used]) for p, used in zip(pipelines,
                  ((2, 6), (0, 2, 3, 6), tuple(range(8))))]
        for pipeline, bg, work in zip(pipelines, groups, (buckets, count, count)):
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, bg)
            compute.dispatch_workgroups(math.ceil(work / 64))
            compute.end()
    device.queue.submit([encoder.finish()])
    idx = int(iterations) % 2
    out_p = np.frombuffer(device.queue.read_buffer(pbuf[idx]), 'f4').reshape(count, 4)[:, :3].copy()
    out_v = np.frombuffer(device.queue.read_buffer(vbuf[idx]), 'f4').reshape(count, 4)[:, :3].copy()
    return out_p, out_v
