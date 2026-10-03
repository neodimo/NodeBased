"""GPU gather-form FLIP transfers over active 8-cubed tiles.

Particle lists are binned deterministically on the CPU; weighted particle/grid interpolation,
free-surface occupancy and grid-to-particle advection execute as wgpu compute kernels. Binning
is the broad phase only. The CPU Liquid3D implementation remains the reference.
"""
from __future__ import annotations

import numpy as np

from . import fluid_gpu_solver as fgs


_C = fgs._c
_U = fgs._u
_TILE_NEIGHBOURS = np.asarray([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)],
                              dtype=np.int64)


def _p2g_source(axis):
    axis_name = "xyz"[axis]
    point = [("f32(c.x)", "f32(c.y) + 0.5", "f32(c.z) + 0.5"),
             ("f32(c.x) + 0.5", "f32(c.y)", "f32(c.z) + 0.5"),
             ("f32(c.x) + 0.5", "f32(c.y) + 0.5", "f32(c.z)")][axis]
    point_hi = list(point)
    point_hi[axis] = f"f32(c.{axis_name} + 1)"
    point_hi = ", ".join(point_hi)
    sy = "(i32(P.a.y) + 1)" if axis == 1 else "i32(P.a.y)"
    sz = "(i32(P.a.z) + 1)" if axis == 2 else "i32(P.a.z)"
    index = f"((c.x * {sy} + c.y) * {sz} + c.z)"
    index_hi = index.replace(f"c.{axis_name}", f"(c.{axis_name} + 1)")
    src = r"""
fn flat(x: i32, y: i32, z: i32) -> u32 { return u32((x * i32(P.a.y) + y) * i32(P.a.z) + z); }
fn weighted(f: vec3<f32>) -> vec2<f32> {
    let nx = i32(P.a.x); let ny = i32(P.a.y); let nz = i32(P.a.z);
    let x0 = max(0, i32(floor(f.x)) - 1); let y0 = max(0, i32(floor(f.y)) - 1); let z0 = max(0, i32(floor(f.z)) - 1);
    let x1 = min(nx - 1, i32(floor(f.x)) + 1); let y1 = min(ny - 1, i32(floor(f.y)) + 1); let z1 = min(nz - 1, i32(floor(f.z)) + 1);
    var numerator = 0.0; var denominator = 0.0;
    for (var x = x0; x <= x1; x++) { for (var y = y0; y <= y1; y++) { for (var z = z0; z <= z1; z++) {
        let c = flat(x, y, z); let begin = offsets[c]; let end = offsets[c + 1u];
        for (var k = begin; k < end; k++) {
            let pid = particles[k]; let p = vec3<f32>(xyz[3u*pid], xyz[3u*pid+1u], xyz[3u*pid+2u]);
            let w = max(0.0, 1.0-abs(p.x-f.x)) * max(0.0, 1.0-abs(p.y-f.y)) * max(0.0, 1.0-abs(p.z-f.z));
            numerator += w * VEL; denominator += w;
        }
    } } }
    return vec2<f32>(select(0.0, numerator / max(denominator, 1e-20), denominator > 1e-20), denominator);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = gid.x / 512u; if (ti >= P.b.y) { return; }
    let lane = gid.x % 512u;
    let l = vec3<u32>(lane / 64u, (lane / 8u) % 8u, lane % 8u);
    let t = tiles[ti]; let ntz = (P.a.z + 7u) / 8u; let nty = (P.a.y + 7u) / 8u;
    let tx = t / (nty * ntz); let ty = (t / ntz) % nty; let tz = t % ntz;
    let c = vec3<i32>(i32(tx*8u+l.x), i32(ty*8u+l.y), i32(tz*8u+l.z));
    if (c.x >= i32(P.a.x) || c.y >= i32(P.a.y) || c.z >= i32(P.a.z)) { return; }
    let f = weighted(vec3<f32>(POINT));
    let out_i = u32(INDEX);
    grid[out_i] = f.x; old[out_i] = f.x; valid[out_i] = select(0u, 1u, f.y > 1e-20);
    if (c.AXIS == i32(P.a.AXISDIM) - 1) {
        let hi = weighted(vec3<f32>(POINT_HI)); let hi_i = u32(INDEX_HI);
        grid[hi_i] = hi.x; old[hi_i] = hi.x; valid[hi_i] = select(0u, 1u, hi.y > 1e-20);
    }
}
"""
    return (src.replace("POINT_HI", point_hi).replace("INDEX_HI", index_hi)
            .replace("POINT", ", ".join(point)).replace("INDEX", index)
            .replace("AXISDIM", axis_name).replace("AXIS", axis_name)
            .replace("VEL", f"velocity[3u * pid + {axis}u]"))


def _register():
    base = [_C("xyz"), _C("velocity"), _C("offsets", "u32"), _C("particles", "u32"),
            _C("tiles", "u32"), _C("grid", acc="rw"), _C("old", acc="rw"), _C("valid", "u32", "rw")]
    for axis, name in enumerate("uvw"):
        _C_dummy = axis
        body = _p2g_source(axis)
        fgs._kernel(f"flip_p2g_{name}", base, body, cellwise=False)

    fgs._kernel("flip_mask", [_C("offsets", "u32"), _C("mask", "u32", "rw"), _C("tiles", "u32")], r"""
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let ti=gid.x/512u; if(ti>=P.b.y){return;} let lane=gid.x%512u;
 let l=vec3<u32>(lane/64u,(lane/8u)%8u,lane%8u); let t=tiles[ti];
 let ntz=(P.a.z+7u)/8u; let nty=(P.a.y+7u)/8u; let tx=t/(nty*ntz); let ty=(t/ntz)%nty; let tz=t%ntz;
 let x=tx*8u+l.x; let y=ty*8u+l.y; let z=tz*8u+l.z;
 if(x>=P.a.x||y>=P.a.y||z>=P.a.z){return;}
 let c=(x*P.a.y+y)*P.a.z+z; mask[c]=select(0u,1u,offsets[c+1u]>offsets[c]);
}
""", cellwise=False)

    tri = (fgs._tri("u", "i32(P.a.x)+1", "i32(P.a.y)", "i32(P.a.z)")
           + fgs._tri("v", "i32(P.a.x)", "i32(P.a.y)+1", "i32(P.a.z)")
           + fgs._tri("w", "i32(P.a.x)", "i32(P.a.y)", "i32(P.a.z)+1")
           + fgs._tri("ou", "i32(P.a.x)+1", "i32(P.a.y)", "i32(P.a.z)")
           + fgs._tri("ov", "i32(P.a.x)", "i32(P.a.y)+1", "i32(P.a.z)")
           + fgs._tri("ow", "i32(P.a.x)", "i32(P.a.y)", "i32(P.a.z)+1"))
    sample = """
fn sample(p:vec3<f32>)->vec3<f32>{return vec3<f32>(tri_u(p-vec3<f32>(0.,.5,.5)),tri_v(p-vec3<f32>(.5,0.,.5)),tri_w(p-vec3<f32>(.5,.5,0.)));}
fn sample_old(p:vec3<f32>)->vec3<f32>{return vec3<f32>(tri_ou(p-vec3<f32>(0.,.5,.5)),tri_ov(p-vec3<f32>(.5,0.,.5)),tri_ow(p-vec3<f32>(.5,.5,0.)));}
"""
    fgs._kernel("flip_g2p", [_C("xyz", acc="rw"), _C("velocity", acc="rw"), _C("u"), _C("v"), _C("w"), _C("ou"), _C("ov"), _C("ow")], r"""
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=P.b.x){return;}
 let b=3u*i;let p=vec3<f32>(xyz[b],xyz[b+1u],xyz[b+2u]);let oldv=vec3<f32>(velocity[b],velocity[b+1u],velocity[b+2u]);
 let pic=sample(p);let flip=oldv+pic-sample_old(p);let vnew=P.c.y*flip+(1.0-P.c.y)*pic;
 velocity[b]=vnew.x;velocity[b+1u]=vnew.y;velocity[b+2u]=vnew.z;
}
""", lib=tri, extra=sample, cellwise=False)
    tri_advect = (fgs._tri("u", "i32(P.a.x)+1", "i32(P.a.y)", "i32(P.a.z)")
                  + fgs._tri("v", "i32(P.a.x)", "i32(P.a.y)+1", "i32(P.a.z)")
                  + fgs._tri("w", "i32(P.a.x)", "i32(P.a.y)", "i32(P.a.z)+1"))
    fgs._kernel("flip_advect", [_C("xyz", acc="rw"), _C("u"), _C("v"), _C("w")], r"""
fn sample(p:vec3<f32>)->vec3<f32>{return vec3<f32>(tri_u(p-vec3<f32>(0.,.5,.5)),tri_v(p-vec3<f32>(.5,0.,.5)),tri_w(p-vec3<f32>(.5,.5,0.)));}
fn bounded(raw:vec3<f32>)->vec3<f32>{
 var q=clamp(raw,vec3<f32>(0.001),vec3<f32>(f32(P.a.x)-0.001,f32(P.a.y)-0.001,f32(P.a.z)-0.001));
 if((P.b.z&1u)!=0u&&raw.x<0.0){q.x=raw.x;} if((P.b.z&2u)!=0u&&raw.x>=f32(P.a.x)){q.x=raw.x;}
 if((P.b.z&4u)!=0u&&raw.y<0.0){q.y=raw.y;} if((P.b.z&8u)!=0u&&raw.y>=f32(P.a.y)){q.y=raw.y;}
 if((P.b.z&16u)!=0u&&raw.z<0.0){q.z=raw.z;} if((P.b.z&32u)!=0u&&raw.z>=f32(P.a.z)){q.z=raw.z;}
 return q;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=P.b.x){return;}let b=3u*i;
 let p=vec3<f32>(xyz[b],xyz[b+1u],xyz[b+2u]);let steps=i32(P.b.y);let h=P.c.x/f32(steps);var q=p;
 for(var s=0;s<steps;s++){let v1=sample(q);let mid=bounded(q+0.5*h*v1);let v2=sample(mid);q=bounded(q+h*v2);}
 xyz[b]=q.x;xyz[b+1u]=q.y;xyz[b+2u]=q.z;
}
""",lib=tri_advect,extra="",cellwise=False)


_register()


class GpuFlipTransfers:
    """Per-substep GPU FLIP transfer kernels; sparse tile tasks are dilated around particles."""
    def __init__(self, shape):
        self.shape = tuple(int(x) for x in shape)
        self.ctx = fgs._ctx()
        nx, ny, nz = self.shape
        self.grid_shapes = ((nx+1, ny, nz), (nx, ny+1, nz), (nx, ny, nz+1))
        self.grids = [self.ctx.buffer(4*int(np.prod(s))) for s in self.grid_shapes]
        self.old = [self.ctx.buffer(4*int(np.prod(s))) for s in self.grid_shapes]
        self.valid = [self.ctx.buffer(4*int(np.prod(s))) for s in self.grid_shapes]
        self.mask = self.ctx.buffer(4*nx*ny*nz)

    def _buffers(self, pos, vel):
        nx, ny, nz = self.shape
        ncell = nx*ny*nz
        cells = np.floor(np.asarray(pos, np.float64)).astype(np.int64)
        cells = np.clip(cells, 0, np.asarray(self.shape)-1)
        flat = (cells[:,0]*ny+cells[:,1])*nz+cells[:,2]
        counts = np.bincount(flat, minlength=ncell).astype(np.uint32)
        offsets = np.empty(ncell+1, np.uint32); offsets[0]=0; np.cumsum(counts, out=offsets[1:])
        order = np.argsort(flat, kind="stable").astype(np.uint32)
        nt = tuple((d+7)//8 for d in self.shape)
        # Mark active tiles directly, then dilate in tile space. Sorting particle
        # coordinates with unique(axis=0) dominates large scenes; this keeps the
        # work proportional to the much smaller tile lattice instead.
        particle_tiles = cells // 8
        active = np.zeros(nt, dtype=bool)
        active[particle_tiles[:, 0], particle_tiles[:, 1], particle_tiles[:, 2]] = True
        expanded = np.zeros_like(active)
        for dx, dy, dz in _TILE_NEIGHBOURS:
            src = []
            dst = []
            for shift, size in zip((dx, dy, dz), nt):
                if shift < 0:
                    src.append(slice(1, size)); dst.append(slice(0, size-1))
                elif shift > 0:
                    src.append(slice(0, size-1)); dst.append(slice(1, size))
                else:
                    src.append(slice(None)); dst.append(slice(None))
            expanded[tuple(dst)] |= active[tuple(src)]
        tiles = np.flatnonzero(expanded.reshape(-1)).astype(np.uint32)
        if not len(tiles): tiles=np.zeros(1,np.uint32)
        def buffer(data):
            b=self.ctx.buffer(np.asarray(data).nbytes);self.ctx.write(b,data);return b
        xyz=buffer(np.asarray(pos,np.float32).reshape(-1)); velocity=buffer(np.asarray(vel,np.float32).reshape(-1))
        offs=buffer(offsets); ids=buffer(order); tilebuf=buffer(tiles)
        return xyz,velocity,offs,ids,tilebuf,len(tiles),len(pos)

    def to_grid(self, pos, vel):
        nx,ny,nz=self.shape
        if not len(pos):
            return tuple(np.zeros(s,np.float64) for s in self.grid_shapes), {k:np.zeros(s,bool) for k,s in zip("uvw",self.grid_shapes)}, np.zeros(self.shape,bool), tuple(np.zeros(s,np.float64) for s in self.grid_shapes)
        xyz,velocity,offsets,ids,tiles,ntiles,n=self._buffers(pos,vel)
        self.ctx.clear(self.mask)
        for buf in self.grids + self.old + self.valid:
            self.ctx.clear(buf)
        self.ctx.dispatch("flip_mask",{"offsets":offsets,"mask":self.mask,"tiles":tiles},
                          _U(a=self.shape,b=(0,ntiles)),("wg",(ntiles*8,1,1)))
        uout=[];valid={};old=[]
        for axis,name in enumerate("uvw"):
            self.ctx.dispatch(f"flip_p2g_{name}",{"xyz":xyz,"velocity":velocity,"offsets":offsets,"particles":ids,"tiles":tiles,
                              "grid":self.grids[axis],"old":self.old[axis],"valid":self.valid[axis]},
                              _U(a=self.shape,b=(0,ntiles)),("wg",(ntiles*8,1,1)))
        mask=self.ctx.read(self.mask,4*nx*ny*nz).view(np.uint32).reshape(self.shape).astype(bool)
        for axis,name in enumerate("uvw"):
            size=int(np.prod(self.grid_shapes[axis]));shape=self.grid_shapes[axis]
            uout.append(self.ctx.read(self.grids[axis],4*size).view(np.float32).reshape(shape).astype(np.float64))
            old.append(self.ctx.read(self.old[axis],4*size).view(np.float32).reshape(shape).astype(np.float64))
            valid[name]=self.ctx.read(self.valid[axis],4*size).view(np.uint32).reshape(shape).astype(bool)
        return tuple(uout),valid,mask,tuple(old)

    def from_grid(self, pos, vel, new_fields, old_fields, flip_ratio, dt, open_faces=(False,)*6):
        if not len(pos): return pos,vel
        n=len(pos)
        names=("u","v","w","ou","ov","ow")
        data=list(new_fields[name] for name in "uvw")+list(old_fields[name] for name in "uvw")
        bufs={"xyz":self.ctx.buffer(np.asarray(pos,np.float32).nbytes),"velocity":self.ctx.buffer(np.asarray(vel,np.float32).nbytes)}
        self.ctx.write(bufs["xyz"],np.asarray(pos,np.float32).reshape(-1));self.ctx.write(bufs["velocity"],np.asarray(vel,np.float32).reshape(-1))
        for name,field in zip(names,data):
            buf=self.ctx.buffer(np.asarray(field,np.float32).nbytes);self.ctx.write(buf,np.asarray(field,np.float32).reshape(-1));bufs[name]=buf
        self.ctx.dispatch("flip_g2p",bufs,_U(a=self.shape,b=(n,),c=(float(dt),float(flip_ratio))),
                          ("wg",((n+63)//64,1,1)))
        self.ctx.flush()
        outvel=self.ctx.read(bufs["velocity"],12*n).view(np.float32).reshape(n,3).copy()
        vmax=float(np.abs(outvel).max()) if outvel.size else 0.0
        steps=max(1,min(6,int(np.ceil(vmax*float(dt)/0.9))))
        advect_bufs={"xyz":bufs["xyz"],"u":bufs["u"],"v":bufs["v"],"w":bufs["w"]}
        open_mask=sum((1 << i) for i, opened in enumerate(open_faces) if opened)
        self.ctx.dispatch("flip_advect",advect_bufs,_U(a=self.shape,b=(n,steps,open_mask),c=(float(dt),)),
                          ("wg",((n+63)//64,1,1)))
        outpos=self.ctx.read(bufs["xyz"],12*n).view(np.float32).reshape(n,3).copy()
        return outpos.astype(np.float64),outvel.astype(np.float64)
