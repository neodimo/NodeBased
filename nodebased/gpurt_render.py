"""GPU triangle AOV renderer; one invocation owns the complete ray peel.

Eight storage bindings, the most any local adapter guarantees (RTX 3080 Ti, AMD Radeon 8060S and
llvmpipe all report max-storage-buffers-per-shader-stage 8): BVH nodes, primitive order, triangles,
per-triangle attributes, the `table`, the `texels` atlas, the ray records and the splat casters. The
`table` is one vec4 array that holds every scene record, so a feature adds records to it rather than a
binding: per-geometry materials (tint, Blinn-Phong factors, PBR factors, flat emissive colour, then the
descriptors of the metallic-roughness, normal, occlusion and emissive maps and of the base-colour mip
chain), the analytic lights, the liquid records, the Environment block (nine spherical-harmonic
coefficients, the rotation, gain and blur, six level descriptors) and the area-light records with their
fixed sample points. `texels` is the single texture atlas all of those descriptors point into: every
map at level 0 and every base-colour mip, and the Environment's six prefiltered levels. Indices are
stored as float values. Textures retain the CPU float32 mip chain, including second near-clipped
triangle mip selection. Traversal has gpurt's f32 edge band; shading and accumulation are f32 rather
than the reference's f64.
"""
import math
import numpy as np
from . import gpurt, gpu3d, raytrace, scene3d as s

GPU_RT_RAYS_PER_SUBMISSION = 1 << 19
MATERIAL_HEADER = 9     # vec4 records before a material's base-colour mip descriptors
PARAM_WORDS = 16        # u32 words in the Params uniform


def check_capability(state):
    reason = gpurt.check_capability(state)
    if reason:
        return reason
    if gpurt._limits(state).get('max-storage-buffers-per-shader-stage', 0) < 8:
        return 'GPU ray tracing unavailable: max-storage-buffers-per-shader-stage too small (needs 8)'
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
 output: u32, splat_offset: u32, pad1: u32, pad2: u32,
 areas: u32, area_offset: u32, env_offset: u32, pad3: u32 };
struct Hit { t: f32, id: i32, u: f32, v: f32 };
var<private> near_bias: f32 = 0.;
// Light linking: a material's (normal scale, occlusion strength, link mask, 0) record carries the bit mask of the lights its
// mesh excludes (light j of the table is bit j, then the area lights, then the environment), and every triangle's e1.w the
// same mask (a shadow ray reads triangles only). `shadow_bit` is the light a shadow ray goes to; a triangle whose mesh
// excludes it casts no shadow from it. 99 is any other ray.
var<private> shadow_bit: u32 = 99u;
fn excl_bit(mask: u32, bit: u32) -> bool { return bit < 24u && ((mask >> bit) & 1u) != 0u; }
// Splat-shadow rays of a splat centre start beyond its own footprint and skip its own caster; -1 = unused.
var<private> splat_near: f32 = -1.;
var<private> exclude_id: i32 = -1;
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> order: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> attrs: array<Attr>;
// Material = tint, (specular, shininess, emission, liquid), (metallic, roughness, dielectric F0, is-pbr),
// (normal scale, occlusion strength, 0, 0), flat emissive colour, then descriptors (offset, width, height,
// present) of the metallic-roughness, normal, occlusion and emissive maps, then the base-colour mip
// descriptors (MATERIAL_HEADER records precede them). Analytic lights follow all materials at the
// uniform-specified offset; liquid records, the background, the Environment block and the area-light
// records and samples follow at their own uniform offsets. `texels` is the one texture atlas.
@group(0) @binding(4) var<storage, read> table: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> texels: array<vec4<f32>>;
// Ray record: origin/near, direction/far, result, view depth.
// Primary directions have view z=-1, so hit.t is positive view depth.
// result.w=-1 signals overflow.
@group(0) @binding(6) var<storage, read_write> rays: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read> splat_data: array<vec4<f32>>;
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
fn splat_visibility(o: vec3<f32>, d: vec3<f32>, limit: f32) -> f32 {
 if (params.splat_offset==0u) { return 1.; }
 var transmission=1.; var stack: array<u32,64>; stack[0]=0u; var size=1u;
 loop {
  if (size==0u) { break; } size--; let index=stack[size]*3u;
  let low=splat_data[index]; let high=splat_data[index+1u];
  // The slab test must index bounds copies that leave out .w. A leaf packs its child
  // link as -1, whose bits are a NaN; indexing the vec4 dynamically made D3D12 carry
  // that NaN into the axis comparison, so every leaf was rejected and a light with an
  // exactly-zero direction component cast no splat shadow at all.
  let lo=low.xyz; let hi=high.xyz;
  var near=select(near_bias,splat_near,splat_near>=0.); var far=limit; var valid=true;
  for (var a=0u;a<3u;a++) {
   if (d[a]==0.) { if (o[a]<lo[a] || o[a]>hi[a]) { valid=false; } }
   else { let x=(lo[a]-o[a])/d[a]; let y=(hi[a]-o[a])/d[a];
    near=max(near,min(x,y)); far=min(far,max(x,y)); }
  }
  if (!valid || near>far) { continue; }
  let metadata=bitcast<vec4<u32>>(splat_data[index+2u]);
  if (metadata.y==0u) {
   if (size+2u>64u) { return 0.; }
   stack[size]=bitcast<u32>(low.w); stack[size+1u]=bitcast<u32>(high.w); size+=2u;
  } else {
   for (var p=0u;p<metadata.y;p++) {
    let start=params.splat_offset+(metadata.x+p)*4u;
    let center=splat_data[start]; let delta=o-center.xyz;
    let xrow=splat_data[start+1u];
    if (i32(xrow.w)==exclude_id) { continue; }
    let x=xrow.xyz; let y=splat_data[start+2u].xyz; let z=splat_data[start+3u].xyz;
    let wo=vec3<f32>(dot(x,delta),dot(y,delta),dot(z,delta));
    let wd=vec3<f32>(dot(x,d),dot(y,d),dot(z,d));
    let dd=dot(wd,wd); var closest=0.;
    if (dd>0.) { closest=-dot(wo,wd)/dd; }
    closest=clamp(closest,select(near_bias,splat_near,splat_near>=0.),limit);
    let q=wo+closest*wd; let d2=dot(q,q);
    if (d2<=9.) { transmission*=1.-min(.99,center.w*exp(-.5*d2)); }
    if (transmission<.001) { return 0.; }
   }
  }
 }
 return transmission;
}
fn visibility(o: vec3<f32>, d: vec3<f32>, limit: f32) -> f32 {
 var transmission=1.; var stack: array<i32,64>; stack[0]=0; var size=select(1u,0u,params.empty!=0u);
 loop {
  if (size==0u) { break; } size--; let index=stack[size];
  if (entry(index,o,d,near_bias,limit)==bitcast<f32>(0x7f800000u)) { continue; }
  let node=nodes[index];
  if (node.count==0u) { stack[size]=node.left; stack[size+1u]=node.right; size+=2u; }
  else { for (var p=0u;p<node.count;p++) {
   let id=order[node.offset+p]; let hit=intersect(id,o,d,near_bias,limit,0.);
   if (hit.id>=0 && !excl_bit(u32(triangles[id].e1.w),shadow_bit)) { transmission*=1.-triangles[id].v0.w; }
  } }
 }
 return transmission*splat_visibility(o,d,limit);
}
fn seed_hash(p: vec3<f32>) -> u32 {
 // scene3d._point_seeds: the same hash of the float32 bits, so CPU and GPU share the sample pattern.
 var h=(bitcast<u32>(p.x)*73856093u)^(bitcast<u32>(p.y)*19349663u)^(bitcast<u32>(p.z)*83492791u);
 h^=h>>16u; h*=0x85ebca6bu; h^=h>>13u; h*=0xc2b2ae35u; h^=h>>16u;
 return h;
}
// scene3d._shadow_trace: hard shadow when ls.y (tan of the blur half-angle) is 0, else ls.z jittered rays.
fn soft_visibility(o: vec3<f32>, d: vec3<f32>, limit: f32, lp: vec4<f32>, ls: vec4<f32>) -> f32 {
 if (ls.y<=0.) { return visibility(o,d,limit); }
 let count=max(u32(ls.z),1u);
 let phi=f32(seed_hash(o))*(6.2831853/4294967296.);
 let axis=select(vec3<f32>(1.,0.,0.),vec3<f32>(0.,1.,0.),abs(d.y)<.9);
 let a=normalize(cross(d,axis)); let b=cross(d,a);
 var total=0.;
 for (var k=0u;k<count;k++) {
  let r=sqrt((f32(k)+.5)/f32(count)); let theta=f32(k)*2.3999632+phi;
  let spread=a*(r*cos(theta))+b*(r*sin(theta));
  var jd=normalize(d+spread*ls.y); var jlimit=limit;
  if (lp.w>0.) {
   let delta=lp.xyz+spread*(limit*ls.y)-o; jlimit=length(delta); jd=delta/max(jlimit,1e-8);
  }
  total+=visibility(o,jd,jlimit);
 }
 return total/f32(count);
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
fn attenuation(lp: vec4<f32>, ld: vec4<f32>, cone: vec4<f32>, power: f32, point: vec3<f32>) -> f32 {
    // scene3d.light_attenuation: distance falloff, times the spot cone; cone = (is spot, inner, outer, exponent) in degrees.
    if (lp.w <= 0.0) { return 1.0; }
    let offset = point - lp.xyz;
    let dist = length(offset);
    var result = 1.0;
    if (power > 0.0) { result = min(1.0, pow(max(dist, 1e-8), -power)); }
    if (cone.x > 0.0) {
        let angle = degrees(acos(clamp(dot(offset, ld.xyz) / max(dist, 1e-12), -1.0, 1.0)));
        var k = select(0.0, 1.0, angle <= cone.y);
        if (cone.z > cone.y) {
            let t = clamp((cone.z - angle) / (cone.z - cone.y), 0.0, 1.0);
            k = select(0.0, pow(t * t * (3.0 - 2.0 * t), cone.w), t > 0.0);
        }
        result = result * k;
    }
    return result;
}
fn unit(v: vec3<f32>) -> vec3<f32> { return v/max(length(v),1e-8); }
// scene3d._orthonormal_tangent: Duff et al. branchless ONB, the frame when a UV-gradient tangent degenerates.
fn orthonormal_tangent(n: vec3<f32>) -> vec3<f32> {
 let sign=select(-1.,1.,n.z>=0.); let a=-1./(sign+n.z); let b=n.x*n.y*a;
 return vec3<f32>(1.+sign*n.x*n.x*a,sign*b,-sign*n.x);
}
// scene3d._shade_fragments' tangent-space normal map: Gram-Schmidt the per-triangle tangent against the normal.
fn apply_normal_map(n: vec3<f32>, tangent: vec3<f32>, uv: vec2<f32>, scale: f32, descriptor: vec4<f32>) -> vec3<f32> {
 let texel=sample_texture(descriptor,uv).xyz;
 let local_n=(texel*2.-1.)*vec3<f32>(scale,scale,1.);
 let t_ortho=tangent-n*dot(tangent,n); let t_norm=length(t_ortho);
 var t=orthonormal_tangent(n);
 if (t_norm>=1e-12) { t=t_ortho/t_norm; }
 let b=cross(n,t);
 return unit(t*local_n.x+b*local_n.y+n*local_n.z);
}
// splatshade._cook_torrance: GGX specular times n.l with a per-channel Fresnel f0.
fn ggx_response(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32, f0: vec3<f32>) -> vec3<f32> {
 let h=unit(l+v); let nl=max(dot(n,l),0.); let nv=max(dot(n,v),1e-4);
 let nh=max(dot(n,h),0.); let vh=max(dot(v,h),0.);
 let alpha=max(roughness,.05)*max(roughness,.05); let alpha2=alpha*alpha;
 let denom=nh*nh*(alpha2-1.)+1.; let d=alpha2/(3.14159265*denom*denom);
 let k=(roughness+1.)*(roughness+1.)/8.;
 let vis=1./(max(nl*(1.-k)+k,1e-4)*max(nv*(1.-k)+k,1e-4)*4.);
 let fresnel=f0+(vec3<f32>(1.)-f0)*pow(1.-vh,5.);
 return vec3<f32>(3.14159265*d*vis*nl)*fresnel;
}
// envlight.dfg: Karis's analytic fit of the split-sum BRDF table, (A, B) with F0*A+B the specular albedo.
fn dfg(n_dot_v: f32, roughness: f32) -> vec2<f32> {
 let nv=clamp(n_dot_v,1e-4,1.);
 let c0=vec4<f32>(-1.,-.0275,-.572,.022); let c1=vec4<f32>(1.,.0425,1.04,-.04);
 let t=roughness*c0+c1;
 let a004=min(t.x*t.x,pow(2.,-9.28*nv))*t.x+t.y;
 return vec2<f32>(-1.04*a004+t.z,1.04*a004+t.w);
}
// The Environment block at params.env_offset (0 = none): 9 SH coefficients, 3 rotation columns
// (envlight.Environment._local's combined turn), (gain rgb, blur), then 6 level descriptors.
fn env_local(d: vec3<f32>) -> vec3<f32> {
 let e=params.env_offset;
 return mat3x3<f32>(table[e+9u].xyz,table[e+10u].xyz,table[e+11u].xyz)*d;
}
fn env_diffuse(n: vec3<f32>) -> vec3<f32> {
 if (params.env_offset==0u) { return vec3<f32>(0.); }
 let e=params.env_offset; let d=env_local(n); let x=d.x; let y=d.y; let z=d.z;
 var total=table[e].xyz*.282095;
 total+=table[e+1u].xyz*(.488603*y)*(2./3.);
 total+=table[e+2u].xyz*(.488603*z)*(2./3.);
 total+=table[e+3u].xyz*(.488603*x)*(2./3.);
 total+=table[e+4u].xyz*(1.092548*x*y)*.25;
 total+=table[e+5u].xyz*(1.092548*y*z)*.25;
 total+=table[e+6u].xyz*(.315392*(3.*z*z-1.))*.25;
 total+=table[e+7u].xyz*(1.092548*x*z)*.25;
 total+=table[e+8u].xyz*(.546274*(x*x-y*y))*.25;
 return total*table[e+12u].xyz;
}
// envlight.sample_map: bilinear lookup of a latitude-longitude level, wrapping in longitude, clamped in latitude.
fn env_level(level: vec4<f32>, d: vec3<f32>) -> vec3<f32> {
 let w=i32(level.y); let h=i32(level.z);
 let u=.5+atan2(d.x,-d.z)/6.28318531; let v=acos(clamp(d.y,-1.,1.))/3.14159265;
 let x=u*f32(w)-.5; let y=clamp(v*f32(h)-.5,0.,f32(h-1));
 let x0=i32(floor(x)); let y0=i32(floor(y)); let fx=x-floor(x); let fy=y-floor(y);
 let xa=u32(((x0%w)+w)%w); let xb=u32((((x0+1)%w)+w)%w); let ya=u32(y0); let yb=u32(min(y0+1,h-1));
 let base=u32(level.x); let row=u32(w);
 let top=mix(texels[base+ya*row+xa].xyz,texels[base+ya*row+xb].xyz,fx);
 let bottom=mix(texels[base+yb*row+xa].xyz,texels[base+yb*row+xb].xyz,fx);
 return mix(top,bottom,fy);
}
// envlight.Environment.specular / Prefiltered.lookup: a two-level lerp over the GGX-roughness levels.
fn env_specular(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
 if (params.env_offset==0u) { return vec3<f32>(0.); }
 let e=params.env_offset;
 let blur=clamp(table[e+12u].w,0.,1.);
 let r=1.-(1.-clamp(roughness,0.,1.))*(1.-blur);
 let position=r*5.; let lower=min(floor(position),4.); let fraction=position-lower;
 let d=env_local(direction);
 let a=env_level(table[e+13u+u32(lower)],d); let b=env_level(table[e+14u+u32(lower)],d);
 return mix(a,b,fraction)*table[e+12u].xyz;
}
// scene3d._mesh_pbr_environment: split-sum image-based light for a pbr mesh; base_rgb is the metal's F0.
fn pbr_env_diffuse(n: vec3<f32>, nv: f32, roughness: f32, metallic: f32, f0d: f32) -> vec3<f32> {
 if (params.env_offset==0u) { return vec3<f32>(0.); }
 let fit=dfg(nv,roughness);
 let dielectric_w=(f0d*fit.x+fit.y)*(1.+f0d*(1./max(fit.x+fit.y,1e-4)-1.));
 return env_diffuse(n)*((1.-metallic)*(1.-dielectric_w));
}
fn pbr_env_specular(n: vec3<f32>, to_eye: vec3<f32>, roughness: f32, metallic: f32, f0d: f32, base_rgb: vec3<f32>) -> vec3<f32> {
 if (params.env_offset==0u) { return vec3<f32>(0.); }
 let nv=max(dot(n,to_eye),0.); let fit=dfg(nv,roughness);
 let compensation=1./max(fit.x+fit.y,1e-4);
 let dielectric_w=vec3<f32>((f0d*fit.x+fit.y)*(1.+f0d*(compensation-1.)));
 let metal_w=(base_rgb*fit.x+fit.y)*(vec3<f32>(1.)+base_rgb*(compensation-1.));
 let total_w=(1.-metallic)*dielectric_w+metallic*metal_w;
 return total_w*env_specular(2.*nv*n-to_eye,roughness);
}
// scene3d._area_light_shading for a Rect/Disc/Sphere light: a Monte Carlo diffuse irradiance over the light's fixed
// sample points (host-packed after the area records, two vec4 each: point, outward normal), each with a shadow ray
// through `visibility`, and a centre-point inverse-square specular colour. Record: (centre, two-sided),
// (radiance, area), (first sample, count, shadow bias scale, shadows on).
struct AreaResult { irradiance: vec3<f32>, to_light: vec3<f32>, spec_colour: vec3<f32> };
fn area_light_shade(j: u32, position: vec3<f32>, normal: vec3<f32>) -> AreaResult {
 shadow_bit=params.lights+j;
 let base=params.area_offset+j*3u;
 let a0=table[base]; let a1=table[base+1u]; let a2=table[base+2u];
 let two_sided=a0.w>.5; let first=u32(a2.x); let count=u32(a2.y);
 let bias=params.bias*a2.z; near_bias=bias*.01;
 let origin=position+normal*bias; let shadows_on=a2.w>.5;
 var total=vec3<f32>(0.); var vis_sum=0.;
 for (var k=0u;k<count;k++) {
  let point=table[first+k*2u].xyz; let sample_normal=table[first+k*2u+1u].xyz;
  let to_recv=position-point; let dist2=max(dot(to_recv,to_recv),1e-10); let wi=to_recv/sqrt(dist2);
  var cos_light=dot(wi,sample_normal);
  cos_light=select(max(cos_light,0.),abs(cos_light),two_sided);
  let weight=cos_light*max(dot(-wi,normal),0.)/dist2;
  var vis=1.;
  if (shadows_on && weight>0.) {
   let ray=point-origin; let limit=length(ray);
   vis=visibility(origin,ray/max(limit,1e-8),limit);
  }
  total+=(weight*vis)*a1.xyz; vis_sum+=vis;
 }
 let fcount=f32(max(count,1u));
 var result: AreaResult;
 result.irradiance=total*(a1.w/fcount);
 let to_centre=a0.xyz-position; let dist2c=max(dot(to_centre,to_centre),1e-6);
 result.to_light=to_centre/sqrt(dist2c);
 result.spec_colour=(vis_sum/fcount/dist2c)*(a1.xyz*a1.w);
 return result;
}
// Liquid material (liquid_render.py is the CPU reference). The tree of reflection/refraction rays is followed with an
// explicit stack: each stacked ray carries its throughput, so the sum of throughput x colour is the same linear
// combination the CPU evaluates recursively. table[pad1] holds 2 records per liquid geometry (ior, reflection,
// roughness, thin distance | sigma), table[pad2] the background colour and the ray epsilon.
struct LRay { o: vec3<f32>, medium: u32, d: vec3<f32>, depth: i32, w: vec3<f32>, pad: f32 };
var<private> lstack: array<LRay,24>;
var<private> ltop: i32 = 0;
var<private> ltotal: vec3<f32> = vec3<f32>(0.);
fn l_escape() -> vec3<f32> { return table[params.pad2].xyz; }
fn l_push(o: vec3<f32>, d: vec3<f32>, medium: u32, depth: i32, w: vec3<f32>) {
 if (ltop<24) { lstack[ltop]=LRay(o,medium,d,depth,w,0.); ltop++; }
}
fn l_highlights(pos: vec3<f32>, n: vec3<f32>, d: vec3<f32>, roughness: f32) -> vec3<f32> {
 var total=vec3<f32>(0.);
 let shininess=clamp(2./max(roughness*roughness,1e-4)-2.,8.,2000.);
 for (var j=0u;j<params.lights;j++) {
  let start=params.light_offset+j*5u; let lp=table[start]; let ld=table[start+1u]; let lc=table[start+2u]; let lk=table[start+3u];
  var to_light=-ld.xyz;
  if (lp.w>0.) { to_light=unit(lp.xyz-pos); }
  let lobe=pow(max(dot(n,unit(to_light-d)),0.),shininess)*select(0.,1.,dot(n,to_light)>0.);
  total+=lobe*attenuation(lp,ld,lk,lc.w,pos)*lc.xyz;
 }
 return total;
}
fn l_event(pos: vec3<f32>, nrm: vec3<f32>, d: vec3<f32>, obj: f32, k: u32, medium: u32, depth: i32, w: vec3<f32>) {
 let rec=table[params.pad1+(k-1u)*2u]; let sigma=table[params.pad1+(k-1u)*2u+1u].xyz;
 let eps=table[params.pad2].w; let inf=bitcast<f32>(0x7f800000u);
 var n=unit(nrm); let entering=dot(d,n)<0.;
 if (!entering) { n=-n; }
 let eta_i=select(rec.x,1.,entering); let eta_t=select(1.,rec.x,entering);
 let cos_i=clamp(-dot(d,n),0.,1.);
 let ratio=eta_i/eta_t; let sin2=ratio*ratio*(1.-cos_i*cos_i); let tir=sin2>1.;
 let cos_t=sqrt(max(0.,1.-sin2));
 let rs=(eta_i*cos_i-eta_t*cos_t)/max(eta_i*cos_i+eta_t*cos_t,1e-12);
 let rp=(eta_t*cos_i-eta_i*cos_t)/max(eta_t*cos_i+eta_i*cos_t,1e-12);
 let f=select(.5*(rs*rs+rp*rp),1.,tir);
 let wr=select(rec.y*f,1.,tir);
 let refl=d+2.*cos_i*n; let trans=unit(ratio*d+(ratio*cos_i-cos_t)*n);
 ltotal+=w*wr*l_highlights(pos,n,d,rec.z);
 var thin=false; var thin_t=0.;
 if (entering) {
  let h2=nearest(pos,d,eps,inf,-inf,-1);
  if (h2.id>=0 && attrs[h2.id].n0.w==obj && h2.t<rec.w) { thin=true; thin_t=h2.t; }
 }
 let last=depth+1>=4;
 if (wr>1e-4) {
  let child=w*wr;
  if (last) { ltotal+=child*l_escape(); } else { l_push(pos+n*eps,refl,medium,depth+1,child); }
 }
 let wt=select(1.-wr,0.,tir);
 if (wt>1e-4) {
  if (thin) {
   let child=w*wt*exp(-sigma*thin_t);
   if (last) { ltotal+=child*l_escape(); } else { l_push(pos+d*(thin_t+eps),d,medium,depth+1,child); }
  } else {
   let child=w*wt;
   if (last) { ltotal+=child*l_escape(); } else { l_push(pos-n*eps,trans,select(0u,k,entering),depth+1,child); }
  }
 }
}
// A non-liquid surface met by a secondary ray: lit by the lights without shadows, as the CPU's _shade_solid.
fn shade_solid(hit: Hit, origin: vec3<f32>) -> vec3<f32> {
 let at=attrs[hit.id]; let w=vec3<f32>(1.-hit.u-hit.v,hit.u,hit.v);
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
 source*=sample_texture(table[material+9u+u32(level)],uv);
 var rgb=source.xyz;
 if (params.lights>0u) {
  let toward=unit(origin-position);
  if (dot(normal,origin-position)<0.) { normal=-normal; }
  var radiance=vec3<f32>(params.ambient); var specular=vec3<f32>(0.);
  for (var j=0u;j<params.lights;j++) {
   let start=params.light_offset+j*5u; let lp=table[start]; let ld=table[start+1u]; let lc=table[start+2u]; let lk=table[start+3u];
   let factor=attenuation(lp,ld,lk,lc.w,position);
   var to_light=-ld.xyz;
   if (lp.w>0.) { to_light=unit(lp.xyz-position); }
   let lambert=dot(normal,to_light);
   radiance+=max(lambert,0.)*factor*lc.xyz;
   let lobe=pow(max(dot(normal,unit(to_light+toward)),0.),properties.y);
   specular+=properties.x*lobe*select(0.,1.,lambert>0.)*factor*lc.xyz;
  }
  rgb=source.xyz*radiance+specular*source.w;
 }
 rgb+=source.xyz*properties.z;
 return rgb+(1.-source.w)*l_escape();
}
// Radiance leaving a liquid point toward `d`'s origin; `nrm` is the outward interpolated normal.
fn liquid_shade(pos: vec3<f32>, nrm: vec3<f32>, d: vec3<f32>, obj: f32, k: u32) -> vec3<f32> {
 ltop=0; ltotal=vec3<f32>(0.);
 let eps=table[params.pad2].w; let inf=bitcast<f32>(0x7f800000u);
 l_event(pos,nrm,d,obj,k,0u,0,vec3<f32>(1.));
 loop {
  if (ltop==0) { break; }
  ltop--; let ray=lstack[ltop];
  let hit=nearest(ray.o,ray.d,eps,inf,-inf,-1);
  if (hit.id<0) { ltotal+=ray.w*l_escape(); continue; }
  var w=ray.w;
  if (ray.medium>0u) { w*=exp(-table[params.pad1+(ray.medium-1u)*2u+1u].xyz*hit.t); }
  let at=attrs[hit.id];
  let kk=u32(table[u32(at.info.x)+1u].w);
  if (kk>0u) {
   let ww=vec3<f32>(1.-hit.u-hit.v,hit.u,hit.v); let tr=triangles[hit.id];
   let p=tr.v0.xyz+hit.u*tr.e1.xyz+hit.v*tr.e2.xyz;
   l_event(p,at.n0.xyz*ww.x+at.n1.xyz*ww.y+at.n2.xyz*ww.z,ray.d,at.n0.w,kk,ray.medium,ray.depth,w);
  } else { ltotal+=w*shade_solid(hit,ray.o); }
 }
 return ltotal;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 let r=(group.y*params.gx+group.x)*64u+lane;
 if (r>=params.count) { return; }
 let origin=rays[r*4u]; let direction=rays[r*4u+1u];
 var ct=-bitcast<f32>(0x7f800000u); var cp=-1; var previous_object=-1.; var previous_edge=false;
 rays[r*4u+3u]=vec4<f32>(bitcast<f32>(0x7f800000u));
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
  if (surfaces>params.maxhits) { rays[r*4u+2u]=vec4<f32>(0.,0.,0.,-1.); return; }
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
  source*=sample_texture(table[material+9u+u32(level)],uv);
  if (surfaces==1u) { rays[r*4u+3u]=vec4<f32>(hit.t); }
  if (params.output==0u && properties.w>0.) {
   let wet=liquid_shade(position,normal,unit(direction.xyz),at.n0.w,u32(properties.w));
   accum+=transmission*vec4<f32>(wet,1.); break;
  }
  let normal_map=table[material+6u];
  if (normal_map.w>.5) {
   // The per-triangle UV-gradient tangent rides in the spare .w of n1, n2 and e.
   normal=apply_normal_map(normal,vec3<f32>(at.n1.w,at.n2.w,at.e.w),uv,table[material+3u].x,normal_map);
  }
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
  var emissive_tint=table[material+4u].xyz;
  let emissive_map=table[material+8u];
  if (emissive_map.w>.5) { emissive_tint*=sample_texture(emissive_map,uv).xyz; }
  let emission=source.xyz*properties.z+emissive_tint;
  let lit=params.lights>0u || params.areas>0u || params.env_offset!=0u;
  let link=u32(table[material+3u].z);
  let env_gate=select(1.,0.,excl_bit(link,params.lights+params.areas));
  if (params.output==6u) { source=vec4<f32>(emission,source.w); }
  else if (params.output!=3u && lit) {
   let toward=unit(origin.xyz-position);
   if (dot(normal,origin.xyz-position)<0.) { normal=-normal; }
   var radiance=vec3<f32>(params.ambient); var specular=vec3<f32>(0.);
   let want_specular=params.output==0u || params.output==5u;
   if (table[material+2u].w>.5) {
    // scene3d._shade_pbr_mesh: Cook-Torrance GGX for every light and the Environment's split-sum terms.
    let base_rgb=source.xyz/max(source.w,1e-6);
    var metallic=clamp(table[material+2u].x,0.,1.); var roughness=clamp(table[material+2u].y,0.,1.);
    let mr_map=table[material+5u];
    if (mr_map.w>.5) { let mr=sample_texture(mr_map,uv); roughness=clamp(mr.y,0.,1.); metallic=clamp(mr.z,0.,1.); }
    let f0d=table[material+2u].z;
    let f0=mix(vec3<f32>(f0d),base_rgb,metallic);
    radiance=vec3<f32>(params.ambient)*(1.-metallic);
    for (var j=0u;j<params.lights;j++) {
     if (excl_bit(link,j)) { continue; }
     shadow_bit=j;
     let start=params.light_offset+j*5u; let lp=table[start]; let ld=table[start+1u]; let lc=table[start+2u]; let lk=table[start+3u]; let ls=table[start+4u];
     let factor=attenuation(lp,ld,lk,lc.w,position);
     var to_light=-ld.xyz;
     if (lp.w>0.) { to_light=unit(lp.xyz-position); }
     var vis=1.;
     if (ld.w>0.) {
      let bias=params.bias*ls.x; near_bias=bias*.01;
      let o=position+normal*bias; var d=-ld.xyz; var limit=bitcast<f32>(0x7f800000u);
      if (lp.w>0.) { let delta=lp.xyz-o; limit=length(delta); d=delta/max(limit,1e-8); }
      vis=soft_visibility(o,d,limit,lp,ls);
     }
     let scale=vis*factor;
     let vh=max(dot(toward,unit(to_light+toward)),0.);
     let kd=(1.-metallic)*(1.-(.04+.96*pow(1.-vh,5.)));
     radiance+=max(dot(normal,to_light),0.)*kd*scale*lc.xyz;
     if (want_specular) { specular+=ggx_response(normal,toward,to_light,roughness,f0)*scale*lc.xyz; }
    }
    for (var j=0u;j<params.areas;j++) {
     if (excl_bit(link,params.lights+j)) { continue; }
     let area=area_light_shade(j,position,normal);
     let vh=max(dot(toward,unit(area.to_light+toward)),0.);
     let kd=(1.-metallic)*(1.-(.04+.96*pow(1.-vh,5.)));
     radiance+=kd*area.irradiance;
     if (want_specular) { specular+=ggx_response(normal,toward,area.to_light,roughness,f0)*area.spec_colour; }
    }
    radiance+=pbr_env_diffuse(normal,max(dot(normal,toward),0.),roughness,metallic,f0d)*env_gate;
    if (want_specular) { specular+=pbr_env_specular(normal,toward,roughness,metallic,f0d,base_rgb)*env_gate; }
   } else {
    radiance+=env_diffuse(normal)*env_gate;
    for (var j=0u;j<params.lights;j++) {
     if (excl_bit(link,j)) { continue; }
     shadow_bit=j;
     let start=params.light_offset+j*5u; let lp=table[start]; let ld=table[start+1u]; let lc=table[start+2u]; let lk=table[start+3u]; let ls=table[start+4u];
     let factor=attenuation(lp,ld,lk,lc.w,position);
     var to_light=-ld.xyz;
     if (lp.w>0.) { to_light=unit(lp.xyz-position); }
     let lambert=dot(normal,to_light); var vis=1.;
     if (ld.w>0.) {
      let bias=params.bias*ls.x; near_bias=bias*.01;
      let o=position+normal*bias; var d=-ld.xyz; var limit=bitcast<f32>(0x7f800000u);
      if (lp.w>0.) { let delta=lp.xyz-o; limit=length(delta); d=delta/max(limit,1e-8); }
      vis=soft_visibility(o,d,limit,lp,ls);
     }
     radiance+=max(lambert*vis,0.)*factor*lc.xyz;
     if (want_specular) {
      let lobe=pow(max(dot(normal,unit(to_light+toward)),0.),properties.y);
      specular+=properties.x*lobe*select(0.,1.,lambert>0.)*vis*factor*lc.xyz;
     }
    }
    for (var j=0u;j<params.areas;j++) {
     if (excl_bit(link,params.lights+j)) { continue; }
     let area=area_light_shade(j,position,normal);
     radiance+=area.irradiance;
     if (want_specular) {
      let lobe=pow(max(dot(normal,unit(area.to_light+toward)),0.),properties.y);
      specular+=properties.x*lobe*area.spec_colour;
     }
    }
    // scene3d._mesh_environment_specular: the Blinn-Phong shininess maps to a GGX roughness.
    if (want_specular && params.env_offset!=0u && properties.x!=0.) {
     specular+=env_specular(2.*dot(normal,toward)*normal-toward,sqrt(2./(properties.y+2.)))*properties.x*env_gate;
    }
   }
   let occlusion_map=table[material+7u];
   if (occlusion_map.w>.5) {
    // Occlusion attenuates the diffuse response (ambient, direct and Environment alike), never the specular.
    let occ=sample_texture(occlusion_map,uv).x;
    radiance*=1.-clamp(table[material+3u].y,0.,1.)*(1.-occ);
   }
   if (params.output==5u) { source=vec4<f32>(specular*source.w,source.w); }
   else { source=vec4<f32>(source.xyz*radiance+specular*source.w,source.w); }
  } else if (params.output==5u) { source=vec4<f32>(0.,0.,0.,source.w); }
  if (params.output==0u) { source=vec4<f32>(source.xyz+emission,source.w); }
  accum+=transmission*source; transmission*=1.-source.w;
  if (source.w>=.999) { break; }
 }
 rays[r*4u+2u]=accum;
}

// Shadow rays from splat centres (host packs one light in the table). Ray record: origin/mesh start,
// direction/limit, result, splat start/excluded caster (both as float values).
@compute @workgroup_size(64)
fn visibility_main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
 let r=(group.y*params.gx+group.x)*64u+lane;
 if (r>=params.count) { return; }
 let a=rays[r*4u]; let b=rays[r*4u+1u]; let c=rays[r*4u+3u];
 near_bias=a.w; splat_near=c.x; exclude_id=i32(c.y);
 let lp=table[params.light_offset]; let ls=table[params.light_offset+4u];
 rays[r*4u+2u]=vec4<f32>(soft_visibility(a.xyz,b.xyz,b.w,lp,ls),0.,0.,0.);
}
'''


def _pipeline(state):
    if '_gpurt_beauty_pipeline' not in state:
        device = state['device']
        state['_gpurt_beauty_pipeline'] = device.create_compute_pipeline(
            layout='auto', compute={'module': device.create_shader_module(code=_SHADER), 'entry_point': 'main'})
    return state['_gpurt_beauty_pipeline']


def _visibility_pipeline(state):
    if '_gpurt_visibility_pipeline' not in state:
        device = state['device']
        state['_gpurt_visibility_pipeline'] = device.create_compute_pipeline(
            layout='auto', compute={'module': device.create_shader_module(code=_SHADER), 'entry_point': 'visibility_main'})
    return state['_gpurt_visibility_pipeline']


def _prepare(scene, camera, width, height, cancel=None, background=(0., 0., 0., 0.)):
    if any(g.projection is not None for g in scene.geometries):
        raise gpu3d.Unsupported('GPU ray tracing beauty supports triangle meshes without projections')
    order = gpu3d.link_order(scene)
    eye, view = s._view_basis(camera)
    focal = 1/math.tan(math.radians(camera.fov)*.5)
    count = sum(len(g.triangles) for g in scene.geometries)
    attributes = np.zeros((max(1, count), 8, 4), 'f4')
    attributes[:, 4, 0] = -1
    attributes[:, 5, 3] = -1
    vertices, alphas, table, textures, link_masks = [], [], [], [], []
    liquids = []        # per liquid geometry: (ior, reflection, roughness, thin distance), (sigma xyz, 0)
    offset, primitive = 0, 0
    for object_id, geometry in enumerate(scene.geometries, 1):
        raytrace._cancel(cancel)
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        local = (view @ (world-eye).T).T
        vertices.append(world[geometry.triangles].astype('f8'))
        alphas.extend([np.clip(geometry.color[3], 0, 1)]*len(geometry.triangles))
        link_masks.append(np.full(len(geometry.triangles), gpu3d.link_mask(geometry.light_link, order)))
        normals = None
        if geometry.normals is not None:
            normals = (np.linalg.inv(matrix[:3, :3]).T @ geometry.normals.T).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        uv = geometry.uvs if geometry.uvs is not None else np.zeros((len(world), 2), 'f4')
        mips = s._mip_chain(geometry.texture) if geometry.texture is not None and geometry.uvs is not None else None
        material = len(table)
        tint = np.asarray(geometry.color, 'f4').copy(); tint[:3] *= tint[3]
        liquid = 0
        if geometry.material == 'liquid':
            from .liquid_render import THIN_SHEET_FRACTION, sigma_of
            extent = float(np.ptp(world[geometry.triangles].reshape(-1, 3), axis=0).max()) if len(geometry.triangles) else 1.
            liquids.extend([(max(float(geometry.ior), 1.), float(np.clip(geometry.reflection, 0, 1)),
                             float(np.clip(geometry.roughness, 0, 1)), THIN_SHEET_FRACTION*max(extent, 1e-6)),
                            (*sigma_of(geometry.absorption_color, geometry.absorption_distance), 0.)])
            liquid = len(liquids)//2
        table.extend([tint, (geometry.specular, geometry.shininess, geometry.emission, liquid),
                      (float(np.clip(geometry.metallic, 0, 1)), float(np.clip(geometry.pbr_roughness, 0, 1)),
                       0.08*float(np.clip(geometry.pbr_specular, 0, 1)), float(geometry.material == 'pbr')),
                      (float(geometry.normal_scale), float(geometry.occlusion_strength),
                       gpu3d.link_mask(geometry.light_link, order), 0.),
                      (*np.asarray(geometry.emissive_color, 'f4')[:3], 0.)])
        for texture in (geometry.metallic_roughness_texture, geometry.normal_texture,
                        geometry.occlusion_texture, geometry.emissive_texture):
            if texture is None:
                table.append((0, 1, 1, 0))
                continue
            level = s._mip_chain(texture)[0]     # the CPU reference reads these maps at level 0 only
            h, w = level.shape[:2]
            table.append((offset, w, h, 1)); textures.append(level.reshape(-1, 4)); offset += w*h
        tangent_map = geometry.normal_texture is not None
        for mip in mips if mips is not None else [np.ones((1, 1, 4), 'f4')]:
            h, w = mip.shape[:2]
            table.append((offset, w, h, 0)); textures.append(mip.reshape(-1, 4)); offset += w*h
        for tri in geometry.triangles:
            if primitive % 256 == 0:
                raytrace._cancel(cancel)
            at = attributes[primitive]; primitive += 1
            z = -local[tri, 2]
            culled = (z <= camera.near).all() or (z >= camera.far).all()
            if normals is None:
                face = np.cross(world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]])
                ns = np.broadcast_to(face/max(float(np.linalg.norm(face)), 1e-8), (3, 3))
            else:
                ns = normals[tri]
            attrs = np.concatenate((local[tri], world[tri], ns, uv[tri]), axis=1).astype('f4')
            at[:3, :3] = ns
            at[0, 3] = object_id
            if tangent_map:
                # scene3d._render_primary's flat UV-gradient tangent, kept in the spare .w of n1, n2 and e.
                e1, e2 = world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]]
                duv1, duv2 = uv[tri[1]]-uv[tri[0]], uv[tri[2]]-uv[tri[0]]
                det = duv1[0]*duv2[1]-duv2[0]*duv1[1]
                tangent = (e1*duv2[1]-e2*duv1[1])/det if abs(det) > 1e-12 else e1
                at[1, 3], at[2, 3], at[6, 3] = tangent
            at[3] = uv[tri[:2]].reshape(4)
            at[4] = (material, 0, *uv[tri[2]])
            if culled:
                continue     # kept for reflected and refracted rays (liquids); no primary ray reaches it
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
    lights = [light for light in scene.lights if light.intensity > 0 and light.kind not in s._AREA]
    for light in lights:
        position, direction = light.world()
        table.extend([(*position, light.kind in s._POSITIONAL), (*direction, light.shadows),
                      (*(np.asarray(light.color)*light.intensity), s._falloff_power(light)),
                      tuple(s._cone_terms(light)), (*s._shadow_terms(light), 0)])
    triangles = np.concatenate(vertices) if vertices else np.empty((0, 3, 3), 'f8')
    liquid_offset = len(table)
    table.extend(liquids)
    extent = max(float(np.ptp(triangles.reshape(-1, 3), axis=0).max()), 1e-6) if len(triangles) else 1.
    background_index = len(table)
    table.append((*np.asarray(background, 'f8')[:3], 1e-5*extent))
    env_offset, level_texels = _pack_environment(scene, table, offset)
    textures.extend(level_texels)
    area_offset, area_count = _pack_area_lights(scene, table)
    primitives = raytrace.TriangleSet(triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0], np.asarray(alphas))
    if link_masks:
        primitives.link_mask = np.concatenate(link_masks).astype('f4')
    bvh = raytrace.Bvh.build(*primitives.aabbs(), cancel=cancel)
    bias = 1e-3*max(1., float(np.ptp(triangles.reshape(-1, 3), axis=0).max())) if count else .001
    return (primitives, bvh, attributes, np.asarray(table or [(0, 0, 0, 0)], 'f4'),
            np.concatenate(textures) if textures else np.zeros((1, 4), 'f4'), len(lights), light_offset, bias,
            liquid_offset, background_index, area_count, area_offset, env_offset)


def _pack_environment(scene, table, texel_offset):
    """Append the Environment block to `table` and return (its table index or 0, the level texels to append to the atlas).

    Layout (what `env_diffuse`/`env_specular` in the shader read): 9 spherical-harmonic coefficients,
    the 3 columns of `envlight.Environment._local`'s combined rotation, (gain rgb, blur), then 6 level
    descriptors (atlas offset, width, height, 0). The levels are the CPU reference's own prefiltered
    maps, level 0 at full resolution, so a mirror reflection samples what `Environment.specular` samples."""
    environments = getattr(scene, 'environments', ())
    if not environments or not scene.geometries:
        return 0, []
    env = environments[0]
    pre = env._pre()
    gain = float(env.intensity)*np.asarray(env.tint, 'f8')
    angle = math.radians(float(env.rotation))
    turn = np.array(((math.cos(angle), 0, -math.sin(angle)), (0, 1, 0), (math.sin(angle), 0, math.cos(angle))))
    parent = np.asarray(env.parent, 'f8')[:3, :3]
    norms = np.linalg.norm(parent, axis=0)
    matrix = (parent/np.where(norms > 1e-12, norms, 1.0)).T @ turn      # d @ matrix.T, as `_local`
    index = len(table)
    table.extend((*row, 0.) for row in pre.sh)
    table.extend((*matrix[:, c], 0.) for c in range(3))
    table.append((*gain, float(np.clip(env.blur, 0, 1))))
    texels = []
    for level in pre.levels:
        h, w = level.shape[:2]
        table.append((texel_offset, w, h, 0))
        texels.append(np.concatenate((np.asarray(level, 'f4'), np.ones((h, w, 1), 'f4')), axis=2).reshape(-1, 4))
        texel_offset += w*h
    return index, texels


def _pack_area_lights(scene, table):
    """Append the Rect/Disc/Sphere light records and their fixed sample points to `table`.

    Per light 3 vec4: (centre, two-sided), (radiance, area), (index of its first sample, sample count, shadow
    bias scale, shadows on); then per sample 2 vec4: (point), (outward normal) -- the same points
    `scene3d._area_light_contribution` uses, so the two renderers integrate the same quadrature."""
    areas = [light for light in scene.lights if light.intensity > 0 and light.kind in s._AREA]
    if not areas:
        return 0, 0
    index = len(table)
    first = index+3*len(areas)
    records, samples = [], []
    for light in areas:
        centre, _ = light.world()
        radiance, area = s._area_light_radiance(light)
        count = int(np.clip(light.light_samples, 1, s.SHADOW_SAMPLES_MAX*4))
        points, normals = s._area_light_samples(light, count)
        records.extend([(*centre, float(bool(light.two_sided))), (*radiance, area),
                        (first+len(samples), count, light.shadow_bias/s.SHADOW_BIAS_DEFAULT, float(bool(light.shadows)))])
        for point, normal in zip(points, normals):
            samples.extend([(*point, 0.), (*normal, 0.)])
    table.extend(records)
    table.extend(samples)
    return index, len(areas)


def _pack_casters(state, scene, cancel=None):
    """One binding: outward-rounded BVH nodes then BVH-ordered ellipsoids."""
    if not (scene.splats and scene.geometries and any(
            light.shadows and light.intensity > 0 for light in scene.lights)
            and any(i.cast_shadows for i in scene.splats)):
        return np.zeros((1, 4), 'f4'), 0
    return _pack_caster_set(state, s._SplatCasters(scene.splats, cancel), cancel)


def _pack_caster_set(state, casters, cancel=None):
    """Pack a `scene3d._SplatCasters` (fresh or from the shared cache) for the shader.

    Each record is centre + opacity, then the three inverse-scaled rotation rows; the first row's
    spare `.w` carries the caster's index in the CPU caster set, so a splat can skip its own record.
    """
    from . import gpusplat
    if casters.empty:
        return np.zeros((1, 4), 'f4'), 0
    primitives, bvh = casters.primitives, casters.bvh
    n = len(primitives.opacity)
    needed = len(bvh.left)*48+n*64
    limits = gpurt._limits(state)
    binding = limits.get('max-storage-buffer-binding-size', 0)
    cap = min(gpusplat.GPU_SPLAT_MEMORY_CAP, limits.get('max-buffer-size', 0), binding)
    if needed > cap:
        raise ValueError(f'GPU ray tracing needs about {needed/2**20:.3f} MiB for {n:,} splat casters, '
                         f'more than the adapter allows ({cap/2**20:.3f} MiB, binding limit {binding/2**20:.3f} MiB)')
    if n >= 1 << 24:
        raise ValueError(f'GPU ray tracing addresses at most {1 << 24:,} splat casters, not {n:,}')
    nodes, order = gpu3d._pack_bvh(bvh, cancel)
    records = np.zeros((n, 4, 4), 'f4')
    records[:, 0, :3] = primitives.positions[order]
    records[:, 0, 3] = primitives.opacity[order]
    records[:, 1:, :3] = (primitives.rotations_matrix.transpose(0, 2, 1) /
                           np.maximum(primitives.scales[:, :, None], 1e-30))[order]
    records[:, 1, 3] = order
    return np.concatenate((nodes.view('f4').reshape(-1, 4), records.reshape(-1, 4))), len(nodes)*3


def mesh_occluders(scene, cancel=None):
    """(TriangleSet, Bvh, epsilon) over every world triangle, as `scene3d.render` builds them for splat shadows."""
    triangles = [(g.world_matrix(), g) for g in scene.geometries if len(g.triangles)]
    if not triangles:
        return None, None, .001
    world = []
    alphas = []
    for matrix, geometry in triangles:
        raytrace._cancel(cancel)
        points = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        world.append(points[geometry.triangles])
        alphas.append(np.full(len(geometry.triangles), np.clip(geometry.color[3], 0, 1), 'f4'))
    world = np.concatenate(world)
    primitives = raytrace.TriangleSet(world[:, 0], world[:, 1]-world[:, 0], world[:, 2]-world[:, 0],
                                      np.concatenate(alphas))
    bvh = raytrace.Bvh.build(*primitives.aabbs(), cancel=cancel)
    return primitives, bvh, 1e-3*max(1., float(np.ptp(world.reshape(-1, 3), axis=0).max()))


class GpuSplatShadows(s._SplatShadows):
    """`scene3d._SplatShadows` with the shadow rays traced on the GPU.

    The cache, the per-splat bookkeeping, the soft-shadow jitter and the answers are the CPU
    reference's; only the two `_trace_*` steps change. Mesh transmittance runs through the triangle
    BVH and splat transmittance through the packed caster BVH in one compute pass per band of rays.
    Use as a context manager (or call `close`) so the GPU buffers are released.
    """
    def __init__(self, state, instances, lights, mesh, mesh_bvh, bias, cancel=None):
        super().__init__(instances, lights, mesh, mesh_bvh, bias, cancel)
        # Own cache namespace: GPU float32 answers never stand in for the CPU reference's.
        self._mesh_key = ('gpu', self._mesh_key)
        self.state = state
        self._triangles = self._caster_buffer = None
        self._caster_offset = 0
        self._owned = []

    def close(self):
        for resource in reversed(self._owned):
            resource.destroy()
        self._owned.clear()
        if self._triangles is not None:
            self._triangles.close()
            self._triangles = None
        self._caster_buffer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _upload(self, data, usage=None):
        wgpu = self.state['wgpu']
        buffer = self.state['device'].create_buffer_with_data(data=data, usage=usage or wgpu.BufferUsage.STORAGE)
        self._owned.append(buffer)
        return buffer

    def _mesh_scene(self):
        if self._triangles is None:
            self._triangles = gpurt.GpuTriangleScene(self.state, self.mesh, self.mesh_bvh) if self.mesh is not None \
                else gpurt.GpuTriangleScene(self.state, raytrace.TriangleSet(*(np.zeros((0, 3)),)*3, np.zeros(0)),
                                            raytrace.Bvh(np.empty((0, 3)), np.empty((0, 3)), *(np.empty(0, np.int32),)*5))
        return self._triangles

    def _splat_buffer(self):
        if self._caster_buffer is None:
            data, self._caster_offset = _pack_caster_set(self.state, self._casters()[1], self.cancel)
            self._caster_buffer = self._upload(data)
        return self._caster_buffer, self._caster_offset

    def _trace_catch(self, light, cloud, matrix, ids):
        position, direction = light.world()
        origin = (matrix[:3, :3] @ np.asarray(cloud.positions[ids], dtype=np.float64).T).T + matrix[:3, 3]
        ray, limit = self._rays(light, position, direction, origin)
        return self._trace(light, origin, ray, limit, None)

    def _trace_relit(self, index, light, ids):
        position, direction = light.world()
        origin = self.positions[index][ids].astype(float)
        ray, limit = self._rays(light, position, direction, origin)
        splat = None
        if not self.empty:
            splat = (2.5*np.max(self.scales[index][ids], axis=1), self.ids[self.offsets[index]+ids])
        return self._trace(light, origin, ray, limit, splat)

    @staticmethod
    def _rays(light, position, direction, origin):
        if light.kind in s._POSITIONAL:
            ray = position-origin
            limit = np.linalg.norm(ray, axis=1)
            return ray/np.maximum(limit[:, None], 1e-30), limit
        return np.broadcast_to(-direction, origin.shape), np.full(len(origin), np.inf)

    def _trace(self, light, origin, ray, limit, splat):
        """Visibility (N,) for hard or soft shadow rays; `splat` is (start offsets, excluded caster ids) or None."""
        state, device, wgpu = self.state, self.state['device'], self.state['wgpu']
        n = len(origin)
        bias = s._light_bias(self.bias, light)
        samples = int(np.clip(light.shadow_samples, 1, s.SHADOW_SAMPLES_MAX)) if light.shadow_blur > 0 else 1
        mesh_levels = math.log2((len(self.mesh.v0) if self.mesh is not None else 0)+2)
        caster_levels = math.log2(len(self._casters()[1].primitives.opacity)+2) if splat is not None else 0
        work = n*samples*16*(mesh_levels+caster_levels)
        bands = gpu3d._band_plan(state, work, 'bvh', n)
        raytrace._cancel(self.cancel)
        mesh = self._mesh_scene()
        caster_buffer, caster_offset = self._splat_buffer() if splat is not None else (self._upload(np.zeros((1, 4), 'f4')), 0)
        table = np.zeros((5, 4), 'f4')
        table[0, :3], table[0, 3] = light.world()[0], light.kind in s._POSITIONAL
        table[1, :3], table[1, 3] = light.world()[1], light.shadows
        table[4, :3] = s._shadow_terms(light)
        table_buffer = self._upload(table)
        pipeline = _visibility_pipeline(state)
        dimension = min(65535, gpurt._limits(state)['max-compute-workgroups-per-dimension'])
        per_submission = min(int(GPU_RT_RAYS_PER_SUBMISSION), gpurt._cap(state)//64, dimension*dimension*64)
        result = np.empty(n)
        for band_start, band_stop in bands:
            for start in range(band_start, band_stop, per_submission):
                raytrace._cancel(self.cancel)
                stop = min(band_stop, start+per_submission)
                count = stop-start
                raw = np.zeros((count, 4, 4), 'f4')
                raw[:, 0, :3], raw[:, 0, 3] = origin[start:stop], bias*.01
                raw[:, 1, :3], raw[:, 1, 3] = ray[start:stop], limit[start:stop]
                raw[:, 3, 0], raw[:, 3, 1] = (splat[0][start:stop], splat[1][start:stop]) if splat is not None else (-1, -1)
                groups = (count+63)//64
                gx = min(dimension, groups)
                gy = (groups+gx-1)//gx
                params = np.zeros(PARAM_WORDS, 'u4')
                params[:4] = count, gx, 1, s.MAX_HITS_PER_RAY
                params[6], params[7], params[9] = mesh.empty, 0, caster_offset
                mark = len(self._owned)
                try:
                    io = self._upload(raw, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
                    uniform = self._upload(params, wgpu.BufferUsage.UNIFORM)
                    binding = {0: mesh.buffers[0], 1: mesh.buffers[1], 2: mesh.buffers[2], 4: table_buffer,
                               6: io, 7: uniform, 8: caster_buffer}
                    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                        {'binding': i, 'resource': {'buffer': b}} for i, b in binding.items()])
                    encoder = device.create_command_encoder()
                    compute = encoder.begin_compute_pass()
                    compute.set_pipeline(pipeline)
                    compute.set_bind_group(0, group)
                    compute.dispatch_workgroups(gx, gy, 1)
                    compute.end()
                    raytrace._cancel(self.cancel)
                    device.queue.submit([encoder.finish()])
                    read = np.frombuffer(device.queue.read_buffer(io), 'f4').reshape(count, 4, 4)
                    raytrace._cancel(self.cancel)
                    result[start:stop] = read[:, 2, 0]
                finally:
                    for resource in reversed(self._owned[mark:]):
                        resource.destroy()
                    del self._owned[mark:]
        return result


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
    if scene.splats:
        if output != 'rgba':
            raise gpu3d.Unsupported('splat data passes and the splats output are CPU-only')
        if not s._opaque_meshes(scene):
            raise gpu3d.Unsupported('transparent meshes mixed with splats are CPU-only')
    if output == 'splats':
        raise gpu3d.Unsupported('splats output is CPU-only')
    if output not in s.RENDER_OUTPUTS:
        raise ValueError(f'Unknown 3D render output {output!r}')
    if output == 'normals_blend':
        output = 'normals'   # splats returned above: without them the blend is the plain first-hit pass
    raytrace._cancel(cancel)
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError('Render dimensions must be positive')
    samples = 1 if output in s.DATA_OUTPUTS else max(1, min(int(samples), 4))
    iw, ih = width*samples, height*samples
    prepared = _prepare(scene, camera, iw, ih, cancel, background)
    (primitives, bvh, attrs, table, texels, lights, light_offset, bias, liquid_offset, background_index,
     area_count, area_offset, env_offset) = prepared
    reason = check_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    gpurt._memory_check(state, max(attrs.nbytes, table.nbytes, texels.nbytes, iw*64))
    dimension = min(65535, gpurt._limits(state)['max-compute-workgroups-per-dimension'])
    budget = min(int(GPU_RT_RAYS_PER_SUBMISSION), gpurt._cap(state)//64, dimension*dimension*64)
    if budget < iw:
        raise ValueError(f'GPU ray tracing needs about {iw*64/2**20:.3f} MiB for one row; submission ray limit is {budget}')
    rows = max(1, budget//iw)
    result = np.empty((ih, iw, 4), 'f4')
    mesh_depth = np.full((ih, iw), np.inf, 'f4') if scene.splats else None
    caster_data, caster_offset = _pack_casters(state, scene, cancel)
    device, wgpu = state['device'], state['wgpu']
    resources = []
    def upload(data, usage):
        resource = device.create_buffer_with_data(data=data, usage=usage)
        resources.append(resource)
        return resource
    try:
        with gpurt.GpuTriangleScene(state, primitives, bvh) as triangles:
            persistent = [upload(data, wgpu.BufferUsage.STORAGE) for data in (attrs, table, texels)]
            caster_buffer = upload(caster_data, wgpu.BufferUsage.STORAGE)
            pipeline = _pipeline(state)
            for y0 in range(0, ih, rows):
                raytrace._cancel(cancel)
                y1 = min(ih, y0+rows)
                o, d, lo, hi = gpurt.primary_rays(camera, iw, ih, rows=(y0, y1))
                n = len(o); raw = np.zeros((n, 4, 4), 'f4')
                raw[:, 0, :3], raw[:, 0, 3] = o, lo
                raw[:, 1, :3], raw[:, 1, 3] = d, hi
                groups = (n+63)//64; gx = min(dimension, groups); gy = (groups+gx-1)//gx
                params = np.array([n, gx, lights, s.MAX_HITS_PER_RAY, 0, 0, triangles.empty, light_offset,
                                   s.RENDER_OUTPUTS.index(output), caster_offset, liquid_offset, background_index,
                                   area_count, area_offset, env_offset, 0], 'u4')
                params.view('f4')[4:6] = ambient, bias
                mark = len(resources)
                try:
                    io = upload(raw, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
                    uniform = upload(params, wgpu.BufferUsage.UNIFORM)
                    bindings = [*triangles.buffers, *persistent, io, uniform, caster_buffer]
                    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                        {'binding': i, 'resource': {'buffer': b}} for i, b in enumerate(bindings)])
                    encoder = device.create_command_encoder(); compute = encoder.begin_compute_pass()
                    compute.set_pipeline(pipeline); compute.set_bind_group(0, group)
                    compute.dispatch_workgroups(gx, gy, 1); compute.end()
                    raytrace._cancel(cancel)
                    device.queue.submit([encoder.finish()])
                    raw_read = np.frombuffer(device.queue.read_buffer(io), 'f4').reshape(n, 4, 4)
                    read = raw_read[:, 2]
                    if mesh_depth is not None:
                        mesh_depth[y0:y1] = raw_read[:, 3, 0].reshape(y1-y0, iw)
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
    if scene.splats:
        from . import gpusplat
        relit = any(i.relight > 0 for i in scene.splats)
        shadowed = any(light.shadows and light.intensity > 0 for light in scene.lights)
        catching = shadowed and bool(scene.geometries) and any(i.shadow_catch > 0 and i.relight < 1 for i in scene.splats)
        occluders = mesh_occluders(scene, cancel) if (relit and shadowed) or catching else (None, None, .001)
        provider = (GpuSplatShadows(state, scene.splats, scene.lights, *occluders, cancel)
                    if (relit and shadowed) or catching else None)
        try:
            if provider is not None:
                provider.relit_shadows = relit and shadowed
            lighting = ((scene.lights, ambient, provider, gpu3d._splat_extras(scene, ambient, provider, cancel))
                        if relit or catching else None)
            rgb, alpha = gpusplat.render_layer(state, scene.splats, camera, iw, ih,
                mesh_depth, lighting=lighting, cancel=cancel)
        finally:
            if provider is not None:
                provider.close()
        result[..., :3] = rgb+(1-alpha[..., None])*result[..., :3]
        result[..., 3] = alpha+(1-alpha)*result[..., 3]
    if samples > 1:
        result = result.reshape(height, samples, width, samples, 4).mean(axis=(1, 3))
    if output == 'rgba':
        bg = np.asarray(background, 'f4').copy(); bg[3] = np.clip(bg[3], 0, 1); bg[:3] *= bg[3]
        result += bg*(1-result[..., 3:4])
    result.flags.writeable = False
    return result
