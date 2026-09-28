"""Optional wgpu dense Lucas--Kanade flow solver.

The CPU pyramid, parameter validation and consistency mask stay authoritative. Each pyramid
level's dense local least-squares iterations run as WGSL compute passes; adapters without wgpu
can continue using :mod:`nodebased.opticalflow` directly.
"""
from __future__ import annotations

import struct
import numpy as np

from . import gpu3d
from .opticalflow import _down, _luma, _sample, flow_pair_oneway


_WGSL = r"""
struct Params { width: u32, height: u32, window_radius: u32, blend: f32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> flow_in: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> flow_out: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> p: Params;

fn index(x: i32, y: i32) -> u32 {
    return u32(clamp(y, 0, i32(p.height)-1) * i32(p.width) + clamp(x, 0, i32(p.width)-1));
}
fn sample_a(x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x)); let y0 = i32(floor(y));
    let fx = x - f32(x0); let fy = y - f32(y0);
    return mix(mix(a[index(x0,y0)], a[index(x0+1,y0)], fx),
               mix(a[index(x0,y0+1)], a[index(x0+1,y0+1)], fx), fy);
}
fn sample_b(x: f32, y: f32) -> f32 {
    let x0 = i32(floor(x)); let y0 = i32(floor(y));
    let fx = x - f32(x0); let fy = y - f32(y0);
    return mix(mix(b[index(x0,y0)], b[index(x0+1,y0)], fx),
               mix(b[index(x0,y0+1)], b[index(x0+1,y0+1)], fx), fy);
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }
    let x = i32(gid.x); let y = i32(gid.y); let i = gid.y*p.width + gid.x;
    let f = flow_in[i];
    var xx=0.0; var yy=0.0; var xy=0.0; var xt=0.0; var yt=0.0;
    let r = i32(p.window_radius);
    for (var dy=-r; dy<=r; dy=dy+1) {
        for (var dx=-r; dx<=r; dx=dx+1) {
            let qx=x+dx; let qy=y+dy;
            let u=f32(qx)+f.x; let v=f32(qy)+f.y;
            let ix=(sample_a(f32(qx+1),f32(qy))-sample_a(f32(qx-1),f32(qy))
                   +sample_b(u+1.0,v)-sample_b(u-1.0,v))*0.25;
            let iy=(sample_a(f32(qx),f32(qy+1))-sample_a(f32(qx),f32(qy-1))
                   +sample_b(u,v+1.0)-sample_b(u,v-1.0))*0.25;
            let it=sample_b(u,v)-sample_a(f32(qx),f32(qy));
            xx=xx+ix*ix; yy=yy+iy*iy; xy=xy+ix*iy; xt=xt+ix*it; yt=yt+iy*it;
        }
    }
    let det=xx*yy-xy*xy;
    var delta=vec2<f32>(0.0);
    if (det>1e-9) { delta=vec2<f32>((-yy*xt+xy*yt)/det, (xy*xt-xx*yt)/det); }
    var mean=vec2<f32>(0.0); var count=0.0;
    for(var sy=-2;sy<=2;sy=sy+1){ for(var sx=-2;sx<=2;sx=sx+1){
        mean=mean+flow_in[index(x+sx,y+sy)]; count=count+1.0;
    }}
    flow_out[i]=p.blend*(f+delta)+(1.0-p.blend)*(mean/count);
}
"""


def _adapter(choice="default"):
    state = gpu3d._state(choice)
    device = state["device"]
    pipeline = state["pipelines"].get("dense_flow_lk")
    if pipeline is None:
        module = device.create_shader_module(code=_WGSL)
        pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
        state["pipelines"]["dense_flow_lk"] = pipeline
    return state["wgpu"], device, pipeline


def _run_direction(a, b, *, vector_detail=4, smoothness=1.0, iterations=5, choice="default"):
    wgpu, device, pipeline = _adapter(choice)
    pa, pb = [np.asarray(a, np.float32)], [np.asarray(b, np.float32)]
    for _ in range(vector_detail-1):
        if min(pa[-1].shape[:2]) < 12: break
        pa.append(_down(pa[-1])); pb.append(_down(pb[-1]))
    flow = np.zeros((*pa[-1].shape, 2), np.float32)
    for level in range(len(pa)-1, -1, -1):
        aa, bb = np.ascontiguousarray(pa[level]), np.ascontiguousarray(pb[level])
        h,w=aa.shape
        if flow.shape[:2] != (h,w):
            flow=_sample(flow,np.broadcast_to(np.linspace(0,flow.shape[1]-1,w),(h,w)),
                np.broadcast_to(np.linspace(0,flow.shape[0]-1,h)[:,None],(h,w))).astype(np.float32)*2
        n=h*w
        a_buf=device.create_buffer_with_data(data=aa.reshape(-1),usage=wgpu.BufferUsage.STORAGE)
        b_buf=device.create_buffer_with_data(data=bb.reshape(-1),usage=wgpu.BufferUsage.STORAGE)
        f0=device.create_buffer_with_data(data=flow.reshape(-1,2),usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
        f1=device.create_buffer(size=n*8,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_SRC)
        uniform=device.create_buffer_with_data(data=struct.pack("<IIIf",w,h,max(1,int(round(smoothness))),0.15),
                                               usage=wgpu.BufferUsage.UNIFORM)
        groups=[]
        for source,target in ((f0,f1),(f1,f0)):
            groups.append(device.create_bind_group(layout=pipeline.get_bind_group_layout(0),entries=[
                {"binding":0,"resource":{"buffer":a_buf,"offset":0,"size":n*4}},
                {"binding":1,"resource":{"buffer":b_buf,"offset":0,"size":n*4}},
                {"binding":2,"resource":{"buffer":source,"offset":0,"size":n*8}},
                {"binding":3,"resource":{"buffer":target,"offset":0,"size":n*8}},
                {"binding":4,"resource":{"buffer":uniform,"offset":0,"size":16}}]))
        encoder=device.create_command_encoder()
        source_index=0
        for _ in range(iterations):
            cp=encoder.begin_compute_pass(); cp.set_pipeline(pipeline); cp.set_bind_group(0,groups[source_index])
            cp.dispatch_workgroups(-(-w//8),-(-h//8),1); cp.end(); source_index=1-source_index
        result=f0 if source_index==0 else f1
        staging=device.create_buffer(size=n*8,usage=wgpu.BufferUsage.COPY_DST|wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(result,0,staging,0,n*8)
        device.queue.submit([encoder.finish()]); staging.map_sync(wgpu.MapMode.READ)
        try: flow=np.frombuffer(staging.read_mapped(),np.float32).reshape(h,w,2).copy()
        finally: staging.unmap()
        for buffer in (a_buf,b_buf,f0,f1,uniform,staging): buffer.destroy()
    return flow


def flow_pair_gpu(first, second, *, vector_detail=4, smoothness=1.0, flow_on="luminance",
                  iterations=8, choice="default"):
    """Compute dense forward/backward flow on the selected wgpu adapter."""
    a,b=_luma(first,flow_on),_luma(second,flow_on)
    if a.shape!=b.shape: raise ValueError("Optical flow frames must have matching dimensions")
    if vector_detail not in (1,2,3,4,5,6): raise ValueError("vector_detail must be in 1..6")
    if smoothness<0: raise ValueError("smoothness must be non-negative")
    f=_run_direction(a,b,vector_detail=vector_detail,smoothness=smoothness,iterations=iterations,choice=choice)
    back=_run_direction(b,a,vector_detail=vector_detail,smoothness=smoothness,iterations=iterations,choice=choice)
    h,w=a.shape; yy,xx=np.mgrid[:h,:w].astype(np.float32)
    back_at=_sample(back,xx+f[...,0],yy+f[...,1])
    occ=np.linalg.norm(f+back_at,axis=2)>(.5+.01*np.linalg.norm(f,axis=2))
    return f,back,occ
