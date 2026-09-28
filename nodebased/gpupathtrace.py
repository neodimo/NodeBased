"""GPU path tracer: the WGSL twin of `nodebased/pathtrace.py`.

One compute invocation owns one pixel and runs `spp` complete paths for it per dispatch, so a pass is
one dispatch (in row bands, each within the adapter's submission budget). The maths, the random numbers
(`pcg`, `rand`) and the light and BSDF conventions are the CPU reference's; only the arithmetic differs
(f32 here, f64 there), and the two agree statistically, not bit for bit.

Scene layout, seven storage bindings and one uniform: `nodes`/`order` hold one top-level tree over the
shapes (instances and geometries alike) followed by one bottom-level tree per unique mesh, so N
instances of a mesh upload its triangles once and are never flattened; `triangles` carries positions,
normals and uvs; `shapes` each shape's inverse matrix, material and bottom-level root; `lights` the
analytic lights; `env` the environment map (RGB and luminance texels, then the marginal and conditional
sampling tables); `accum` two vec4 per pixel (channel sum and coverage; luminance moments).

Scope, named rather than assumed: textures, more than one environment and scenes with splats, particles
or volumes raise `gpu3d.Unsupported` (callers fall back to the CPU reference).
"""
import math
import time

import numpy as np

from . import gpu3d, gpurt, pathtrace as pt, raytrace, scene3d as s

GPU_PATHS_PER_SUBMISSION = 1 << 19
STACK = 64
TLAS_STACK = 32
_OUTPUT_CODES = {"rgba": 0, "diffuse": 1, "specular": 2, "emission": 3, "albedo": 4, "diffuse_indirect": 5,
                 "specular_indirect": 6, "depth": 7, "normals": 8, "position": 9, "uv": 10, "object_id": 11}
_KIND_CODES = {"Directional": 0, "Point": 1, "Spot": 2, "Rect": 3, "Disc": 4, "Sphere": 5}
LIGHT_VECS = 6
SHAPE_VECS = 8
TRI_VECS = 8


def check_capability(state):
    for name, minimum in [("max-storage-buffers-per-shader-stage", 7), ("max-storage-buffer-binding-size", 64),
                          ("max-buffer-size", 64), ("max-compute-invocations-per-workgroup", 64),
                          ("max-compute-workgroup-size-x", 8), ("max-compute-workgroups-per-dimension", 1)]:
        if gpurt._limits(state).get(name, 0) < minimum:
            return f"GPU path tracing unavailable: {name} too small (needs {minimum})"
    if "device" in state and "wgpu" in state:
        try:
            _pipeline(state)
        except Exception as exc:
            return f"GPU path tracing compute unavailable: {exc}"
    return None


_SHADER = r'''
struct Node { lo: vec3<f32>, left: i32, hi: vec3<f32>, right: i32, offset: u32, count: u32, pad: vec2<u32> };
struct Tri { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32>, n0: vec4<f32>, n1: vec4<f32>, n2: vec4<f32>,
             uv01: vec4<f32>, uv2: vec4<f32> };
struct Shape { i0: vec4<f32>, i1: vec4<f32>, i2: vec4<f32>, base: vec4<f32>, mat: vec4<f32>, mat2: vec4<f32>,
               sigma: vec4<f32>, ids: vec4<u32> };
struct Params {
  a: vec4<u32>,        // width, height, row0, row1
  b: vec4<u32>,        // sample_base, spp, seed, max_bounces
  c: vec4<u32>,        // diffuse cap, specular cap, transmission cap, light count
  d: vec4<u32>,        // env width, env height (0 = no environment), output code, tiles per row
  e: vec4<u32>,        // shape count (0 = empty scene), env cdf base (in vec4s), pad, pad
  right: vec4<f32>,    // camera right, aspect
  up: vec4<f32>,       // camera up, 1 / focal
  forward: vec4<f32>,  // camera forward, near
  eye: vec4<f32>,      // eye, far
  f: vec4<f32>,        // eps, shadow eps, ambient, env scale
  gain: vec4<f32>,     // environment gain rgb
  m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,   // world -> map rotation rows
  tiles: array<vec4<u32>, 512>,                   // active tile bits
};
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> order: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<Tri>;
@group(0) @binding(3) var<storage, read> shapes: array<Shape>;
@group(0) @binding(4) var<storage, read> lights: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> env: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> params: Params;

const PI: f32 = 3.14159265358979;
const INF: f32 = 3.0e38;
const DELTA_ROUGHNESS: f32 = 0.02;
const DIM_BASE: u32 = 8u;
const DIM_STRIDE: u32 = 64u;
const BSDF_DIM: u32 = 50u;

// ---------------------------------------------------------------------------------------------- random
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn path_key(pixel: u32, sample: u32, seed: u32) -> u32 { return pcg(pixel + pcg(sample + pcg(seed))); }
fn rnd(key: u32, dim: u32) -> f32 { return f32(pcg(key ^ (dim * 2654435769u)) >> 8u) * (1.0 / 16777216.0); }

// ------------------------------------------------------------------------------------------- traversal
var<private> tlas_stack: array<i32, 32>;
var<private> blas_stack: array<i32, 64>;

fn entry(index: i32, o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32) -> f32 {
  let node = nodes[index];
  var near = lower;
  var far = upper;
  for (var a = 0u; a < 3u; a++) {
    if (d[a] == 0.0) {
      if (o[a] < node.lo[a] || o[a] > node.hi[a]) { return INF; }
    } else {
      let x = (node.lo[a] - o[a]) / d[a];
      let y = (node.hi[a] - o[a]) / d[a];
      near = max(near, min(x, y));
      far = min(far, max(x, y));
    }
  }
  if (near > far) { return INF; }
  return near;
}

struct Hit { t: f32, shape: i32, tri: i32, u: f32, v: f32 };

fn tri_hit(id: u32, o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32) -> vec3<f32> {
  // (t, u, v); t = -1 for a miss
  let tri = triangles[id];
  let h = cross(d, tri.e2.xyz);
  let det = dot(h, tri.e1.xyz);
  if (abs(det) <= 1e-12) { return vec3<f32>(-1.0, 0.0, 0.0); }
  let inv = 1.0 / det;
  let delta = o - tri.v0.xyz;
  let u = dot(delta, h) * inv;
  let q = cross(delta, tri.e1.xyz);
  let v = dot(d, q) * inv;
  let t = dot(tri.e2.xyz, q) * inv;
  if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > lower && t < upper) { return vec3<f32>(t, u, v); }
  return vec3<f32>(-1.0, 0.0, 0.0);
}

fn to_local_o(sh: Shape, o: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(sh.i0.xyz, o) + sh.i0.w, dot(sh.i1.xyz, o) + sh.i1.w, dot(sh.i2.xyz, o) + sh.i2.w);
}
fn to_local_d(sh: Shape, d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(sh.i0.xyz, d), dot(sh.i1.xyz, d), dot(sh.i2.xyz, d));
}

// Nearest hit over the whole scene; `any_hit` returns at the first opaque hit found (shadow rays).
fn intersect(o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32, any_hit: bool) -> Hit {
  var best = Hit(upper, -1, -1, 0.0, 0.0);
  if (params.e.x == 0u) { return best; }
  var top = 1u;
  tlas_stack[0] = 0;
  loop {
    if (top == 0u) { break; }
    top--;
    let index = tlas_stack[top];
    if (entry(index, o, d, lower, best.t) == INF) { continue; }
    let node = nodes[index];
    if (node.count == 0u) {
      tlas_stack[top] = node.left;
      tlas_stack[top + 1u] = node.right;
      top += 2u;
      continue;
    }
    for (var p = 0u; p < node.count; p++) {
      let sid = order[node.offset + p];
      let sh = shapes[sid];
      if (any_hit && sh.base.w < 0.5) { continue; }
      let lo = to_local_o(sh, o);
      let ld = to_local_d(sh, d);
      var bsize = 1u;
      blas_stack[0] = i32(sh.ids.x);
      loop {
        if (bsize == 0u) { break; }
        bsize--;
        let bi = blas_stack[bsize];
        if (entry(bi, lo, ld, lower, best.t) == INF) { continue; }
        let bn = nodes[bi];
        if (bn.count == 0u) {
          blas_stack[bsize] = bn.left;
          blas_stack[bsize + 1u] = bn.right;
          bsize += 2u;
        } else {
          for (var q = 0u; q < bn.count; q++) {
            let tid = order[bn.offset + q];
            let r = tri_hit(tid, lo, ld, lower, best.t);
            if (r.x > 0.0) {
              best = Hit(r.x, i32(sid), i32(tid), r.y, r.z);
              if (any_hit) { return best; }
            }
          }
        }
      }
    }
  }
  return best;
}

// ---------------------------------------------------------------------------------------------- surface
struct Surf { pos: vec3<f32>, ns: vec3<f32>, ng: vec3<f32>, uv: vec2<f32>, base: vec3<f32>, alpha: f32 };

fn unit(v: vec3<f32>) -> vec3<f32> { return v / max(length(v), 1e-30); }

fn surface(h: Hit, o: vec3<f32>, d: vec3<f32>) -> Surf {
  let sh = shapes[h.shape];
  let tri = triangles[h.tri];
  let w0 = 1.0 - h.u - h.v;
  let ln = w0 * tri.n0.xyz + h.u * tri.n1.xyz + h.v * tri.n2.xyz;
  let lg = cross(tri.e1.xyz, tri.e2.xyz);
  var s: Surf;
  s.pos = o + d * h.t;
  var ns = unit(sh.i0.xyz * ln.x + sh.i1.xyz * ln.y + sh.i2.xyz * ln.z);
  var ng = unit(sh.i0.xyz * lg.x + sh.i1.xyz * lg.y + sh.i2.xyz * lg.z);
  if (dot(ng, ns) < 0.0) { ng = -ng; }
  s.ns = ns;
  s.ng = ng;
  s.uv = w0 * tri.uv01.xy + h.u * tri.uv01.zw + h.v * tri.uv2.xy;
  s.base = sh.base.xyz;
  s.alpha = sh.base.w;
  return s;
}

// ------------------------------------------------------------------------------------------------- BSDF
struct Lobe { diffuse: vec3<f32>, f0: vec3<f32>, k: vec3<f32>, spec_albedo: vec3<f32>, has_spec: bool,
              delta: bool, rough: f32, p_spec: f32 };

fn dfg(nv_in: f32, r: f32) -> vec2<f32> {
  let nv = clamp(nv_in, 1e-4, 1.0);
  let t = vec4<f32>(-1.0, -0.0275, -0.572, 0.022) * r + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let a004 = min(t.x * t.x, exp2(-9.28 * nv)) * t.x + t.y;
  return vec2<f32>(-1.04 * a004 + t.z, 1.04 * a004 + t.w);
}

fn make_lobe(sh: Shape, base: vec3<f32>, nv: f32) -> Lobe {
  var l: Lobe;
  let m = sh.mat.x;
  let r = sh.mat.y;
  let f0d = sh.mat.z;
  let ab = dfg(nv, r);
  let comp = 1.0 / max(ab.x + ab.y, 1e-4);
  l.f0 = vec3<f32>(f0d * (1.0 - m)) + base * m;
  l.k = vec3<f32>(1.0) + l.f0 * (comp - 1.0);
  l.spec_albedo = (l.f0 * ab.x + vec3<f32>(ab.y)) * l.k;
  let dielectric = (f0d * ab.x + ab.y) * (1.0 + f0d * (comp - 1.0));
  if (sh.mat2.x < 0.5) { l.diffuse = base; } else { l.diffuse = base * ((1.0 - m) * (1.0 - dielectric)); }
  l.has_spec = max(l.f0.x, max(l.f0.y, l.f0.z)) > 0.0;
  l.rough = r;
  l.delta = r <= DELTA_ROUGHNESS && l.has_spec;
  let ws = select(0.0, (l.spec_albedo.x + l.spec_albedo.y + l.spec_albedo.z) / 3.0, l.has_spec);
  let wd = (l.diffuse.x + l.diffuse.y + l.diffuse.z) / 3.0;
  let p = clamp(ws / max(ws + wd, 1e-12), 0.05, 0.95);
  if (wd <= 1e-9) { l.p_spec = select(0.0, 1.0, l.has_spec); }
  else if (ws <= 1e-9) { l.p_spec = 0.0; }
  else { l.p_spec = p; }
  return l;
}

struct Ev { fd: vec3<f32>, fs: vec3<f32>, pdf: f32 };

fn bsdf_eval(l: Lobe, n: vec3<f32>, v: vec3<f32>, wi: vec3<f32>) -> Ev {
  var e: Ev;
  let nv = max(dot(n, v), 1e-4);
  let nl = max(dot(n, wi), 0.0);
  e.fd = l.diffuse * (nl / PI);
  let h = unit(wi + v);
  let nh = max(dot(n, h), 0.0);
  let vh = max(dot(v, h), 1e-6);
  let alpha = max(l.rough, 0.05) * max(l.rough, 0.05);
  let a2 = alpha * alpha;
  let dd = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
  let kk = (l.rough + 1.0) * (l.rough + 1.0) / 8.0;
  let vis = 1.0 / (max(nl * (1.0 - kk) + kk, 1e-4) * max(nv * (1.0 - kk) + kk, 1e-4) * 4.0);
  let fres = l.f0 + (vec3<f32>(1.0) - l.f0) * pow(max(1.0 - vh, 0.0), 5.0);
  if (l.has_spec && !l.delta) { e.fs = fres * (dd * vis * nl) * l.k; } else { e.fs = vec3<f32>(0.0); }
  let g1 = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
  let pdf_spec = g1 * dd / (4.0 * nv);
  let p_eff = select(l.p_spec, 0.0, l.delta);
  e.pdf = (1.0 - l.p_spec) * nl / PI + p_eff * pdf_spec;
  return e;
}

fn frame_t(n: vec3<f32>) -> vec3<f32> {
  let sg = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (sg + n.z);
  let b = n.x * n.y * a;
  return vec3<f32>(1.0 + sg * n.x * n.x * a, sg * b, -sg * n.x);
}
fn frame_b(n: vec3<f32>) -> vec3<f32> {
  let sg = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (sg + n.z);
  let b = n.x * n.y * a;
  return vec3<f32>(b, sg + n.y * n.y * a, -n.y);
}

fn cosine_sample(u1: f32, u2: f32) -> vec3<f32> {
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
}

fn vndf_sample(vl: vec3<f32>, alpha: f32, u1: f32, u2: f32) -> vec3<f32> {
  let vh = unit(vec3<f32>(alpha * vl.x, alpha * vl.y, vl.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  var t1 = vec3<f32>(1.0, 0.0, 0.0);
  if (lensq > 1e-12) { t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq); }
  let t2 = cross(vh, t1);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let a = r * cos(phi);
  var b = r * sin(phi);
  let s = 0.5 * (1.0 + vh.z);
  b = (1.0 - s) * sqrt(max(0.0, 1.0 - a * a)) + s * b;
  let nh = a * t1 + b * t2 + sqrt(max(0.0, 1.0 - a * a - b * b)) * vh;
  return unit(vec3<f32>(alpha * nh.x, alpha * nh.y, max(nh.z, 0.0)));
}

fn mis(pa: f32, pb: f32) -> f32 { let a = pa * pa; let b = pb * pb; return a / max(a + b, 1e-30); }

// -------------------------------------------------------------------------------------------- environment
fn env_texel(x: i32, y: i32) -> vec4<f32> { return env[u32(y) * params.d.x + u32(x)]; }
fn env_local(dir: vec3<f32>) -> vec3<f32> { return vec3<f32>(dot(params.m0.xyz, dir), dot(params.m1.xyz, dir), dot(params.m2.xyz, dir)); }
fn env_world(l: vec3<f32>) -> vec3<f32> {
  return params.m0.xyz * l.x + params.m1.xyz * l.y + params.m2.xyz * l.z;
}
fn env_uv(l: vec3<f32>) -> vec2<f32> { return vec2<f32>(0.5 + atan2(l.x, -l.z) / (2.0 * PI), acos(clamp(l.y, -1.0, 1.0)) / PI); }

fn env_radiance(dir: vec3<f32>) -> vec3<f32> {
  let w = i32(params.d.x);
  let h = i32(params.d.y);
  let uv = env_uv(env_local(dir));
  let x = min(i32(uv.x * f32(w)), w - 1);
  let y = min(i32(uv.y * f32(h)), h - 1);
  return env_texel(x, y).xyz * params.gain.xyz;
}
fn env_pdf(dir: vec3<f32>) -> f32 {
  let w = i32(params.d.x);
  let h = i32(params.d.y);
  let uv = env_uv(env_local(dir));
  let x = min(i32(uv.x * f32(w)), w - 1);
  let y = min(i32(uv.y * f32(h)), h - 1);
  return env_texel(x, y).w * params.f.w;
}
fn cdf_at(i: u32) -> f32 {
  let v = env[params.e.y + (i >> 2u)];
  return v[i & 3u];
}
struct EnvSample { dir: vec3<f32>, pdf: f32, radiance: vec3<f32> };
fn env_sample(u1: f32, u2: f32) -> EnvSample {
  let w = params.d.x;
  let h = params.d.y;
  // row: first index whose marginal cdf exceeds u1
  var lo = 0u;
  var hi = h - 1u;
  loop {
    if (lo >= hi) { break; }
    let mid = (lo + hi) / 2u;
    if (cdf_at(mid) > u1) { hi = mid; } else { lo = mid + 1u; }
  }
  let row = lo;
  var prev = 0.0;
  if (row > 0u) { prev = cdf_at(row - 1u); }
  let fu = clamp((u1 - prev) / max(cdf_at(row) - prev, 1e-30), 0.0, 0.999999);
  let base = h + row * w;
  var clo = 0u;
  var chi = w - 1u;
  loop {
    if (clo >= chi) { break; }
    let mid = (clo + chi) / 2u;
    if (cdf_at(base + mid) > u2) { chi = mid; } else { clo = mid + 1u; }
  }
  let col = clo;
  var cprev = 0.0;
  if (col > 0u) { cprev = cdf_at(base + col - 1u); }
  let fv = clamp((u2 - cprev) / max(cdf_at(base + col) - cprev, 1e-30), 0.0, 0.999999);
  let theta = (f32(row) + fu) / f32(h) * PI;
  let phi = ((f32(col) + fv) / f32(w) - 0.5) * 2.0 * PI;
  let local = vec3<f32>(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
  var e: EnvSample;
  e.dir = env_world(local);
  e.pdf = env_texel(i32(col), i32(row)).w * params.f.w;
  e.radiance = env_texel(i32(col), i32(row)).xyz * params.gain.xyz;
  return e;
}

// ---------------------------------------------------------------------------------------- analytic lights
// light layout (6 vec4): l0 = (kind, two-sided, falloff power, spot flag); l1 = (position, radius or half width);
// l2 = (normal or direction, half height or cone exponent); l3 = (right, area); l4 = (up, cone inner);
// l5 = (colour: radiance for area lights, irradiance otherwise; cone outer)
fn light_field(li: u32, k: u32) -> vec4<f32> { return lights[li * 6u + k]; }

fn area_sample(li: u32, u1: f32, u2: f32, point: ptr<function, vec3<f32>>, normal: ptr<function, vec3<f32>>) {
  let l0 = light_field(li, 0u);
  let l1 = light_field(li, 1u);
  let l2 = light_field(li, 2u);
  let l3 = light_field(li, 3u);
  let l4 = light_field(li, 4u);
  let kind = u32(l0.x);
  if (kind == 5u) {
    let z = 1.0 - 2.0 * u1;
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = 2.0 * PI * u2;
    let local = vec3<f32>(r * cos(phi), r * sin(phi), z);
    *point = l1.xyz + local * l1.w;
    *normal = local;
    return;
  }
  var ox = 0.0;
  var oy = 0.0;
  if (kind == 3u) {
    ox = (u1 * 2.0 - 1.0) * l1.w;
    oy = (u2 * 2.0 - 1.0) * l2.w;
  } else {
    let r = sqrt(u1) * l1.w;
    let phi = 2.0 * PI * u2;
    ox = r * cos(phi);
    oy = r * sin(phi);
  }
  *point = l1.xyz + l3.xyz * ox + l4.xyz * oy;
  *normal = l2.xyz;
}

fn area_hit(li: u32, o: vec3<f32>, d: vec3<f32>, tmax: f32) -> f32 {
  let l0 = light_field(li, 0u);
  let l1 = light_field(li, 1u);
  let l2 = light_field(li, 2u);
  let l3 = light_field(li, 3u);
  let l4 = light_field(li, 4u);
  let kind = u32(l0.x);
  if (kind == 5u) {
    let oc = o - l1.xyz;
    let b = dot(oc, d);
    let c = dot(oc, oc) - l1.w * l1.w;
    let disc = b * b - c;
    if (disc <= 0.0) { return INF; }
    let t0 = -b - sqrt(disc);
    if (t0 > 1e-6 && t0 < tmax) { return t0; }
    return INF;
  }
  let denom = dot(d, l2.xyz);
  if (abs(denom) <= 1e-12) { return INF; }
  let t = dot(l1.xyz - o, l2.xyz) / denom;
  if (!(t > 1e-6 && t < tmax)) { return INF; }
  if (!(denom < 0.0 || l0.y > 0.5)) { return INF; }
  let p = o + d * t - l1.xyz;
  let x = dot(p, l3.xyz);
  let y = dot(p, l4.xyz);
  if (kind == 3u) {
    if (abs(x) <= l1.w && abs(y) <= l2.w) { return t; }
    return INF;
  }
  if (x * x + y * y <= l1.w * l1.w) { return t; }
  return INF;
}

fn light_atten(li: u32, pos: vec3<f32>) -> f32 {
  // scene3d.light_attenuation for Point and Spot lights
  let l0 = light_field(li, 0u);
  let l1 = light_field(li, 1u);
  let l2 = light_field(li, 2u);
  let l4 = light_field(li, 4u);
  let l5 = light_field(li, 5u);
  let off = pos - l1.xyz;
  let dist = length(off);
  var a = 1.0;
  if (l0.z > 0.5) { a = min(1.0, pow(max(dist, 1e-8), -l0.z)); }
  if (l0.w > 0.5) {
    let cosine = dot(off, l2.xyz) / max(dist, 1e-12);
    let angle = degrees(acos(clamp(cosine, -1.0, 1.0)));
    let inner = l4.w;
    let outer = l5.w;
    var cone = 0.0;
    if (outer > inner) {
      let t = clamp((outer - angle) / (outer - inner), 0.0, 1.0);
      let sm = t * t * (3.0 - 2.0 * t);
      if (t > 0.0) { cone = pow(sm, max(l2.w, 0.0)); }
    } else {
      cone = select(0.0, 1.0, angle <= inner);
    }
    a = a * cone;
  }
  return a;
}

// ----------------------------------------------------------------------------------------- path buckets
struct Acc { emission: vec3<f32>, diffuse: vec3<f32>, specular: vec3<f32>, diffuse_i: vec3<f32>,
             specular_i: vec3<f32>, alpha: f32, albedo: vec3<f32> };

fn add_class(acc: ptr<function, Acc>, cls: i32, depth: u32, v: vec3<f32>) {
  if (cls == 0) {
    if (depth <= 1u) { (*acc).diffuse += v; } else { (*acc).diffuse_i += v; }
  } else if (cls == 1) {
    if (depth <= 1u) { (*acc).specular += v; } else { (*acc).specular_i += v; }
  }
}

fn total_of(a: Acc) -> vec3<f32> { return a.emission + a.diffuse + a.specular + a.diffuse_i + a.specular_i; }

struct First { shape: i32, t: f32, ns: vec3<f32>, pos: vec3<f32>, uv: vec2<f32> };

fn fresnel_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> vec3<f32> {
  // (reflectance, tir flag, cos_t)
  let ratio = eta_i / eta_t;
  let sin2 = ratio * ratio * (1.0 - cos_i * cos_i);
  if (sin2 > 1.0) { return vec3<f32>(1.0, 1.0, 0.0); }
  let cos_t = sqrt(max(0.0, 1.0 - sin2));
  let rs = (eta_i * cos_i - eta_t * cos_t) / max(eta_i * cos_i + eta_t * cos_t, 1e-12);
  let rp = (eta_t * cos_i - eta_i * cos_t) / max(eta_t * cos_i + eta_i * cos_t, 1e-12);
  return vec3<f32>(0.5 * (rs * rs + rp * rp), 0.0, cos_t);
}

fn light_count() -> u32 { return params.c.w; }

// One camera ray's whole path. `data` is the data-pass code (0 for beauty and its components).
fn trace(key: u32, o_in: vec3<f32>, d_in: vec3<f32>, cam_c: f32, acc: ptr<function, Acc>, first: ptr<function, First>) {
  var o = o_in;
  var d = d_in;
  var tmin = params.forward.w * cam_c;
  var tmax = params.eye.w * cam_c;
  var thr = vec3<f32>(1.0);
  var bounces = 0u;
  var cd = 0u;
  var cs = 0u;
  var ct = 0u;
  var cls = -1;
  var prev_delta = true;
  var capped = false;
  var prev_pdf = 0.0;
  var medium = -1;
  var vdepth = 0u;
  let max_b = params.b.w;
  let eps = params.f.x;
  let seps = params.f.y;
  let lc = light_count();
  let has_env = params.d.y > 0u;
  for (var turn = 0u; turn < max_b + 40u; turn++) {
    let dim = DIM_BASE + turn * DIM_STRIDE;
    let h = intersect(o, d, tmin, tmax, false);
    tmin = 0.0;
    tmax = INF;
    var t_light = INF;
    var light_id = -1;
    for (var li = 0u; li < lc; li++) {
      if (light_field(li, 0u).x > 2.5) {
        let tl = area_hit(li, o, d, h.t);
        if (tl < t_light) { t_light = tl; light_id = i32(li); }
      }
    }
    if (light_id >= 0 && vdepth >= 1u) {
      let li = u32(light_id);
      let l0 = light_field(li, 0u);
      let l1 = light_field(li, 1u);
      let l2 = light_field(li, 2u);
      let l3 = light_field(li, 3u);
      let l5 = light_field(li, 5u);
      let point = o + d * t_light;
      var normal = l2.xyz;
      if (l0.x > 4.5) { normal = unit(point - l1.xyz); }
      let facing = dot(normal, d);
      var cos_l = max(-facing, 0.0);
      if (l0.y > 0.5) { cos_l = abs(facing); }
      let pdf_l = t_light * t_light / max(cos_l * l3.w, 1e-30);
      var weight = 1.0;
      if (!prev_delta) { weight = mis(prev_pdf, pdf_l); }
      add_class(acc, cls, vdepth, thr * l5.xyz * weight);
      break;
    }
    if (h.shape < 0) {
      if (vdepth >= 1u) {
        var sky = vec3<f32>(params.f.z);
        if (has_env) {
          var weight = 1.0;
          if (!prev_delta) { weight = mis(prev_pdf, env_pdf(d)); }
          sky += env_radiance(d) * weight;
        }
        add_class(acc, cls, vdepth, thr * sky);
      }
      break;
    }
    let sh = shapes[h.shape];
    let sf = surface(h, o, d);
    let wo = -d;
    var pos = sf.pos;
    var ns = sf.ns;
    var ng = sf.ng;
    if (medium >= 0) {
      let sg = shapes[medium].sigma.xyz;
      thr *= exp(-sg * h.t);
    }
    let liquid = sh.mat2.x > 1.5;
    if (!liquid && dot(ns, wo) < 0.0) { ns = -ns; ng = -ng; }
    var cover = true;
    if (sf.alpha < 1.0 && params.d.z < 7u) { cover = rnd(key, dim) < sf.alpha; }
    if (!cover) { o = pos + d * eps; continue; }
    if (vdepth == 0u) {
      (*acc).alpha = sf.alpha;
      (*acc).albedo = sf.base * sf.alpha;
      (*first).shape = h.shape; (*first).t = h.t; (*first).ns = ns; (*first).pos = pos; (*first).uv = sf.uv;
    }
    if (params.d.z >= 7u) { return; }
    if (sh.mat.w > 0.0) {
      let g = thr * sf.base * sh.mat.w;
      if (vdepth == 0u) { (*acc).emission += g; } else { add_class(acc, cls, vdepth, g); }
    }
    if (bounces >= max_b || capped) { break; }
    if (liquid) {
      let entering = dot(d, ns) < 0.0;
      var n = ns;
      if (!entering) { n = -ns; }
      let ior = sh.mat2.y;
      let eta_i = select(ior, 1.0, entering);
      let eta_t = select(1.0, ior, entering);
      let cos_i = clamp(-dot(d, n), 0.0, 1.0);
      let fr = fresnel_dielectric(cos_i, eta_i, eta_t);
      var weight_r = sh.mat2.z * fr.x;
      if (fr.y > 0.5) { weight_r = 1.0; }
      let reflect_ = rnd(key, dim + 1u) < weight_r;
      let ratio = eta_i / eta_t;
      var nd = d + 2.0 * cos_i * n;
      var side = n;
      if (!reflect_) {
        nd = unit(ratio * d + (ratio * cos_i - fr.z) * n);
        side = -n;
      }
      if (reflect_) {
        if (cs >= params.c.y) { break; }
        cs++;
        capped = cs >= params.c.y;
      } else {
        if (ct >= params.c.z) { break; }
        ct++;
        capped = ct >= params.c.z;
        if (entering) { medium = h.shape; } else { medium = -1; }
      }
      o = pos + side * eps;
      d = nd;
      if (cls < 0) { cls = 1; }
      prev_delta = true;
      bounces++;
      vdepth++;
      continue;
    }
    // ---- a surface with a BSDF ----
    let nv = max(dot(ns, wo), 1e-4);
    let lobe = make_lobe(sh, sf.base, nv);
    var direct_d = vec3<f32>(0.0);
    var direct_s = vec3<f32>(0.0);
    var slot = 0u;
    for (var li = 0u; li < lc; li++) {
      let l0 = light_field(li, 0u);
      let l5 = light_field(li, 5u);
      let kind = u32(l0.x);
      if (kind <= 2u) {
        var wi = vec3<f32>(0.0);
        var dist = INF;
        var irr = l5.xyz;
        if (kind == 0u) {
          wi = -light_field(li, 2u).xyz;
        } else {
          let to = light_field(li, 1u).xyz - pos;
          dist = length(to);
          wi = to / max(dist, 1e-12);
          irr = irr * light_atten(li, pos);
        }
        let nl = dot(ns, wi);
        if (nl > 0.0 && max(irr.x, max(irr.y, irr.z)) > 0.0) {
          if (intersect(pos + ns * seps, wi, 0.0, dist - seps, true).shape < 0) {
            let e = bsdf_eval(lobe, ns, wo, wi);
            direct_d += e.fd * irr;
            direct_s += e.fs * irr;
          }
        }
      } else {
        let u1 = rnd(key, dim + 2u + slot);
        let u2 = rnd(key, dim + 3u + slot);
        slot += 2u;
        var q = vec3<f32>(0.0);
        var nq = vec3<f32>(0.0);
        area_sample(li, u1, u2, &q, &nq);
        let to = q - pos;
        let dist = length(to);
        let wi = to / max(dist, 1e-12);
        let cosr = dot(-wi, nq);
        var cos_l = max(cosr, 0.0);
        if (l0.y > 0.5) { cos_l = abs(cosr); }
        let nl = dot(ns, wi);
        if (nl > 0.0 && cos_l > 1e-9 && dist > 1e-9) {
          if (intersect(pos + ns * seps, wi, 0.0, dist * (1.0 - 1e-4) - seps, true).shape < 0) {
            let pdf_l = dist * dist / max(cos_l * light_field(li, 3u).w, 1e-30);
            let e = bsdf_eval(lobe, ns, wo, wi);
            let wgt = mis(pdf_l, e.pdf) / pdf_l;
            direct_d += e.fd * l5.xyz * wgt;
            direct_s += e.fs * l5.xyz * wgt;
          }
        }
      }
    }
    if (has_env) {
      let u1 = rnd(key, dim + 2u + slot);
      let u2 = rnd(key, dim + 3u + slot);
      slot += 2u;
      let es = env_sample(u1, u2);
      let nl = dot(ns, es.dir);
      if (nl > 0.0 && es.pdf > 0.0) {
        if (intersect(pos + ns * seps, es.dir, 0.0, INF, true).shape < 0) {
          let e = bsdf_eval(lobe, ns, wo, es.dir);
          let wgt = mis(es.pdf, e.pdf) / es.pdf;
          direct_d += e.fd * es.radiance * wgt;
          direct_s += e.fs * es.radiance * wgt;
        }
      }
    }
    if (vdepth == 0u) {
      (*acc).diffuse += thr * direct_d;
      (*acc).specular += thr * direct_s;
    } else {
      add_class(acc, cls, vdepth + 1u, thr * (direct_d + direct_s));
    }
    // ---- BSDF sampling ----
    let u_lobe = rnd(key, dim + BSDF_DIM);
    let u1 = rnd(key, dim + BSDF_DIM + 1u);
    let u2 = rnd(key, dim + BSDF_DIM + 2u);
    let choose_spec = u_lobe < lobe.p_spec;
    let tt = frame_t(ns);
    let bb = frame_b(ns);
    let vl = vec3<f32>(dot(wo, tt), dot(wo, bb), dot(wo, ns));
    var wi = vec3<f32>(0.0);
    if (!choose_spec) {
      let c = cosine_sample(u1, u2);
      wi = tt * c.x + bb * c.y + ns * c.z;
    } else if (lobe.delta) {
      wi = 2.0 * nv * ns - wo;
    } else {
      let alpha = max(lobe.rough, 0.05) * max(lobe.rough, 0.05);
      let hl = vndf_sample(vl, alpha, u1, u2);
      let rl = 2.0 * dot(vl, hl) * hl - vl;
      wi = tt * rl.x + bb * rl.y + ns * rl.z;
    }
    let cosn = max(dot(ns, wi), 0.0);
    let e = bsdf_eval(lobe, ns, wo, wi);
    var weight = vec3<f32>(0.0);
    let is_delta = choose_spec && lobe.delta;
    if (e.pdf > 1e-12) { weight = (e.fd + e.fs) / e.pdf; }
    if (is_delta) {
      let schlick = lobe.f0 + (vec3<f32>(1.0) - lobe.f0) * pow(max(1.0 - nv, 0.0), 5.0);
      weight = schlick * lobe.k / max(lobe.p_spec, 1e-12);
    }
    if (!(cosn > 0.0 || is_delta)) { break; }
    if (choose_spec) { if (cs >= params.c.y) { break; } } else { if (cd >= params.c.x) { break; } }
    let new_thr = thr * weight;
    if (max(new_thr.x, max(new_thr.y, new_thr.z)) <= 0.0) { break; }
    thr = new_thr;
    var side = ng;
    if (dot(wi, ng) < 0.0) { side = -ng; }
    o = pos + side * eps;
    d = wi;
    if (choose_spec) { cs++; capped = cs >= params.c.y; } else { cd++; capped = cd >= params.c.x; }
    if (cls < 0) { cls = select(0, 1, choose_spec); }
    prev_delta = is_delta;
    prev_pdf = e.pdf;
    bounces++;
    vdepth++;
    if (bounces >= 3u) {
      let q = clamp(max(thr.x, max(thr.y, thr.z)), 0.05, 0.95);
      if (rnd(key, dim + BSDF_DIM + 3u) < q) { thr = thr / q; } else { break; }
    }
  }
}

fn channel_of(a: Acc, code: u32) -> vec3<f32> {
  switch code {
    case 0u: { return total_of(a); }
    case 1u: { return a.diffuse; }
    case 2u: { return a.specular; }
    case 3u: { return a.emission; }
    case 4u: { return a.albedo; }
    case 5u: { return a.diffuse_i; }
    default: { return a.specular_i; }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.a.x;
  let x = gid.x;
  let y = gid.y + params.a.z;
  if (x >= width || y >= params.a.w) { return; }
  let tile = (y / 16u) * params.d.w + (x / 16u);
  let word = params.tiles[tile >> 7u][(tile >> 5u) & 3u];
  if (((word >> (tile & 31u)) & 1u) == 0u) { return; }
  let pixel = y * width + x;
  let code = params.d.z;
  var rgb = vec3<f32>(0.0);
  var alpha = 0.0;
  var lum_sum = 0.0;
  var lum_sq = 0.0;
  let count = select(params.b.y, 1u, code >= 7u);
  for (var s = 0u; s < count; s++) {
    let key = path_key(pixel, params.b.x + s, params.b.z);
    var jx = 0.5;
    var jy = 0.5;
    if (code < 7u) { jx = rnd(key, 0u); jy = rnd(key, 1u); }
    let px = f32(x) + jx;
    let py = f32(y) + jy;
    let aspect = params.right.w;
    let lx = (2.0 * px / f32(width) - 1.0) * aspect / (1.0 / params.up.w);
    let ly = (1.0 - 2.0 * py / f32(params.a.y)) / (1.0 / params.up.w);
    let dv = params.right.xyz * lx + params.up.xyz * ly + params.forward.xyz;
    let c = length(dv);
    var acc: Acc;
    var first: First;
    first.shape = -1;
    trace(key, params.eye.xyz, dv / c, c, &acc, &first);
    if (code >= 7u) {
      var value = vec3<f32>(0.0);
      var cov = 0.0;
      if (first.shape >= 0) {
        cov = 1.0;
        switch code {
          case 7u: { value = vec3<f32>(first.t / c); }
          case 8u: { value = first.ns; }
          case 9u: { value = first.pos; }
          case 10u: { value = vec3<f32>(first.uv, 0.0); }
          default: { value = vec3<f32>(f32(first.shape + 1), 0.0, 0.0); }
        }
      }
      accum[pixel * 2u] = vec4<f32>(value, cov);
      return;
    }
    let total = total_of(acc);
    rgb += channel_of(acc, code);
    alpha += acc.alpha;
    let lum = dot(total, vec3<f32>(0.2126, 0.7152, 0.0722));
    lum_sum += lum;
    lum_sq += lum * lum;
  }
  let previous = accum[pixel * 2u];
  accum[pixel * 2u] = previous + vec4<f32>(rgb, alpha);
  let stats = accum[pixel * 2u + 1u];
  accum[pixel * 2u + 1u] = stats + vec4<f32>(lum_sum, lum_sq, f32(count), 0.0);
}
'''


def _pipeline(state):
    if "_gpupt_pipeline" not in state:
        device = state["device"]
        state["_gpupt_pipeline"] = device.create_compute_pipeline(layout="auto", compute={
            "module": device.create_shader_module(code=_SHADER), "entry_point": "main"})
    return state["_gpupt_pipeline"]


# --- packing --------------------------------------------------------------------------------------------

def _f4(*arrays):
    return np.ascontiguousarray(np.concatenate([np.asarray(a, "f4") for a in arrays], axis=-1))


class Packed:
    """The buffers of one scene and the numbers the uniform needs."""


def pack(ps, environment_size=None, cancel=None):
    if any(t is not None for t in ps.texture):
        raise gpu3d.Unsupported("textured surfaces are not path traced on the GPU yet")
    if len(ps.envs) > 1:
        raise gpu3d.Unsupported("the GPU path tracer takes one environment light")
    if len(ps.area_lights) + len(ps.point_lights) + len(ps.envs) > pt._MAX_LIGHT_SAMPLES:
        raise gpu3d.Unsupported("too many lights for the path tracer's random-number layout")
    packed = Packed()
    n_shapes = ps.shapes
    nodes_parts, order_parts, tri_parts = [], [], []
    node_base = 0
    root_of_blas = []
    order_base = 0
    tri_base = 0
    # top-level tree first: its leaves point at shape ids
    # a mesh without triangles has no tree to enter: its shapes stay out of the top-level tree
    solid = np.array([i for i in range(n_shapes) if len(ps.blases[ps.shape_blas[i]].v0)], np.int64)
    shapes_empty = len(solid) == 0
    if not shapes_empty:
        tlas = raytrace.Bvh.build(ps.world_lo[solid], ps.world_hi[solid], cancel=cancel)
        tlas_nodes, tlas_order = gpu3d._pack_bvh(tlas, cancel)
        tlas_order = solid[tlas_order]
    if shapes_empty:
        nodes_parts.append(np.zeros(1, gpu3d._pack_bvh(raytrace.Bvh.build(np.zeros((1, 3)), np.ones((1, 3))))[0].dtype))
        order_parts.append(np.zeros(1, "u4"))
        tri_parts.append(np.zeros((1, TRI_VECS * 4), "f4"))
        packed.empty = True
    else:
        packed.empty = False
        nodes_parts.append(tlas_nodes)
        order_parts.append(tlas_order.astype("u4"))
        node_base, order_base = len(tlas_nodes), len(tlas_order)
        for blas in ps.blases:
            if not len(blas.v0):
                root_of_blas.append(0)
                continue
            bvh = raytrace.Bvh.build(*blas.tris.aabbs(), cancel=cancel)
            nodes, order = gpu3d._pack_bvh(bvh, cancel)
            nodes = nodes.copy()
            interior = nodes["count"] == 0
            nodes["left"] = np.where(interior, nodes["left"] + node_base, nodes["left"])
            nodes["right"] = np.where(interior, nodes["right"] + node_base, nodes["right"])
            nodes["offset"] = nodes["offset"] + order_base
            root_of_blas.append(node_base)
            nodes_parts.append(nodes)
            order_parts.append(order.astype("u4") + tri_base)
            count = len(blas.v0)
            rows = np.zeros((count, TRI_VECS * 4), "f4")
            rows[:, 0:3], rows[:, 4:7], rows[:, 8:11] = blas.v0, blas.e1, blas.e2
            rows[:, 12:15], rows[:, 16:19], rows[:, 20:23] = (blas.normals[:, k] for k in range(3))
            if blas.uvs is not None:
                rows[:, 24:26], rows[:, 26:28], rows[:, 28:30] = blas.uvs[:, 0], blas.uvs[:, 1], blas.uvs[:, 2]
            tri_parts.append(rows)
            node_base += len(nodes)
            order_base += len(order)
            tri_base += count
    if len(root_of_blas) < len(ps.blases):
        root_of_blas = [0] * len(ps.blases)      # nothing to enter: every ray misses (`params.e.x` is 0)
    packed.nodes = np.concatenate(nodes_parts)
    packed.order = np.concatenate(order_parts).astype("u4")
    packed.triangles = np.ascontiguousarray(np.concatenate(tri_parts).astype("f4"))
    shapes = np.zeros((max(n_shapes, 1), SHAPE_VECS * 4), "f4")
    for i in range(n_shapes):
        inv = ps.inverse[i]
        shapes[i, 0:4], shapes[i, 4:8], shapes[i, 8:12] = inv[0], inv[1], inv[2]
        shapes[i, 12:15], shapes[i, 15] = ps.base[i], ps.alpha[i]
        shapes[i, 16:20] = (ps.metallic[i], ps.roughness[i], ps.f0[i], ps.emission[i])
        shapes[i, 20:24] = (ps.kind[i], ps.ior[i], ps.reflection[i], ps.object_id[i])
        shapes[i, 24:27] = ps.sigma[i]
        shapes[i, 28:29] = np.array([root_of_blas[ps.shape_blas[i]]], np.uint32).view("f4")
    packed.shapes = shapes
    lights = []
    for light in ps.point_lights:
        row = np.zeros(LIGHT_VECS * 4, "f4")
        power = s._falloff_power(light.light) if light.kind != "Directional" else 0.0
        spot, inner, outer, exponent = s._cone_terms(light.light)
        row[0:4] = (_KIND_CODES[light.kind], 0, power, spot)
        row[4:7] = light.position
        row[8:11], row[11] = light.direction, exponent
        row[16], row[17], row[18] = 0, 0, 0
        row[19] = inner
        row[20:23], row[23] = light.irradiance, outer
        lights.append(row)
    for light in ps.area_lights:
        row = np.zeros(LIGHT_VECS * 4, "f4")
        row[0:4] = (_KIND_CODES[light.kind], float(light.two_sided), 0, 0)
        row[4:7], row[7] = light.position, (light.half_w if light.kind == "Rect" else light.radius)
        row[8:11], row[11] = light.normal, light.half_h
        row[12:15], row[15] = light.right, light.area
        row[16:19] = light.up
        row[20:23] = light.radiance
        lights.append(row)
    lights.sort(key=lambda r: r[0] > 2.5)      # point-like lights first; the order does not matter to the maths
    packed.lights = np.array(lights, "f4").reshape(-1, LIGHT_VECS * 4) if lights else np.zeros((1, LIGHT_VECS * 4), "f4")
    packed.light_count = len(lights)
    if ps.envs:
        env = ps.envs[0]
        h, w = env.lum.shape
        texels = np.zeros((h * w + (h + h * w + 3) // 4 + 1, 4), "f4")
        texels[:h * w, :3] = env.rgb.reshape(-1, 3)
        texels[:h * w, 3] = env.lum.reshape(-1)
        cdf = np.concatenate((env.marginal, env.conditional.reshape(-1))).astype("f4")
        flat = np.zeros(((len(cdf) + 3) // 4) * 4, "f4")
        flat[:len(cdf)] = cdf
        texels[h * w:h * w + len(flat) // 4] = flat.reshape(-1, 4)
        packed.env, packed.env_size, packed.env_cdf_base = texels, (w, h), h * w
        forward = env.local(np.eye(3))       # rows: images of the world axes, i.e. the map rotation's transpose
        packed.env_rotation = np.asarray(forward.T, "f4")
        packed.env_gain, packed.env_scale = env.gain.astype("f4"), float(env.scale)
    else:
        packed.env = np.zeros((1, 4), "f4")
        packed.env_size, packed.env_cdf_base = (0, 0), 0
        packed.env_rotation, packed.env_gain, packed.env_scale = np.eye(3, dtype="f4"), np.zeros(3, "f4"), 0.0
    return packed


def _uniform(packed, ps, camera, width, height, row0, row1, sample_base, spp, settings, code, tile_bits, tiles_x):
    eye, view = s._view_basis(camera)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    a = np.array([width, height, row0, row1], "u4")
    b = np.array([sample_base, spp, settings.seed, settings.max_bounces], "u4")
    c = np.array([settings.diffuse_bounces, settings.specular_bounces, settings.transmission_bounces,
                  packed.light_count], "u4")
    d = np.array([packed.env_size[0], packed.env_size[1], code, tiles_x], "u4")
    e = np.array([0 if packed.empty else ps.shapes, packed.env_cdf_base, 0, 0], "u4")
    f32 = np.zeros((9, 4), "f4")
    right, up, forward = view[0], view[1], -view[2]
    f32[0, :3], f32[0, 3] = right, width / max(height, 1)
    f32[1, :3], f32[1, 3] = up, 1.0 / focal
    f32[2, :3], f32[2, 3] = forward, camera.near
    f32[3, :3], f32[3, 3] = eye, camera.far
    f32[4] = (ps.eps, ps.shadow_eps, ps.ambient, packed.env_scale)
    f32[5, :3] = packed.env_gain
    rotation = packed.env_rotation
    f32[6, :3], f32[7, :3], f32[8, :3] = rotation[0], rotation[1], rotation[2]
    tiles = np.zeros((512, 4), "u4")
    tiles.reshape(-1)[:len(tile_bits)] = tile_bits
    return b"".join((a.tobytes(), b.tobytes(), c.tobytes(), d.tobytes(), e.tobytes(), f32.tobytes(), tiles.tobytes()))


def _tile_bits(tile_done):
    active = ~tile_done
    words = np.zeros((len(active) + 31) // 32, "u4")
    idx = np.flatnonzero(active)
    np.bitwise_or.at(words, idx >> 5, (np.uint32(1) << (idx & 31).astype("u4")).astype("u4"))
    return words


# --- rendering ------------------------------------------------------------------------------------------

def render(scene, camera, width, height, background, ambient, output, settings, cancel=None, progress=None,
           stats=None):
    """Path trace on the GPU; raises `gpu3d.Unsupported` for what it cannot take (the caller falls back)."""
    if output not in _OUTPUT_CODES:
        raise gpu3d.Unsupported(f"the GPU path tracer does not produce the {output!r} output")
    state = gpu3d._state()
    reason = check_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    started = time.perf_counter()
    raytrace._cancel(cancel)
    ps = pt.build_scene(scene, ambient)
    try:
        packed = pack(ps, cancel=cancel)
    except ValueError as exc:
        if "traversal stack" in str(exc):
            raise gpu3d.Unsupported(str(exc)) from exc
        raise
    width, height = int(width), int(height)
    tiles_x, tiles_y = -(-width // pt.TILE), -(-height // pt.TILE)
    if tiles_x * tiles_y > 512 * 4 * 32:
        raise gpu3d.Unsupported("the image has more tiles than the GPU path tracer tracks")
    limits = gpurt._limits(state)
    cap = min(limits.get("max-buffer-size", 0), limits.get("max-storage-buffer-binding-size", 0))
    accum_bytes = width * height * 32
    for name, data in (("nodes", packed.nodes), ("triangles", packed.triangles), ("env", packed.env)):
        if data.nbytes > cap:
            raise ValueError(f"GPU path tracing needs a {data.nbytes / 2**20:.1f} MiB {name} buffer; the adapter allows {cap / 2**20:.1f} MiB")
    if accum_bytes > cap:
        raise ValueError(f"GPU path tracing needs {accum_bytes / 2**20:.1f} MiB for the image; the adapter allows {cap / 2**20:.1f} MiB")
    device, wgpu = state["device"], state["wgpu"]
    resources = []

    def upload(data, usage):
        buffer = device.create_buffer_with_data(data=data, usage=usage)
        resources.append(buffer)
        return buffer
    data_pass = output in pt.DATA_OUTPUTS
    code = _OUTPUT_CODES[output]
    storage = wgpu.BufferUsage.STORAGE
    try:
        buffers = [upload(packed.nodes, storage), upload(packed.order, storage), upload(packed.triangles, storage),
                   upload(packed.shapes, storage), upload(packed.lights, storage), upload(packed.env, storage)]
        accum = upload(np.zeros((width * height, 8), "f4"), storage | wgpu.BufferUsage.COPY_SRC)
        pipeline = _pipeline(state)
        tile_of = ((np.arange(width * height) // width) // pt.TILE) * tiles_x + (np.arange(width * height) % width) // pt.TILE
        tile_done = np.zeros(tiles_x * tiles_y, bool)
        tile_count = np.zeros(tiles_x * tiles_y, np.int64)
        per_pass = 1 if data_pass else max(1, settings.pass_samples or 1)
        total_samples = 1 if data_pass else settings.samples
        rows_per_band = max(1, GPU_PATHS_PER_SUBMISSION // max(width * per_pass, 1))
        sample_index, passes = 0, 0
        while sample_index < total_samples:
            raytrace._cancel(cancel)
            take = min(per_pass, total_samples - sample_index)
            bits = _tile_bits(tile_done)
            if not len(np.flatnonzero(~tile_done)):
                break
            for y0 in range(0, height, rows_per_band):
                raytrace._cancel(cancel)
                y1 = min(height, y0 + rows_per_band)
                uniform = upload(_uniform(packed, ps, camera, width, height, y0, y1, sample_index, take, settings,
                                          code, bits, tiles_x), wgpu.BufferUsage.UNIFORM)
                group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
                    {"binding": i, "resource": {"buffer": b}} for i, b in enumerate([*buffers, accum, uniform])])
                encoder = device.create_command_encoder()
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(pipeline)
                compute.set_bind_group(0, group)
                compute.dispatch_workgroups(-(-width // 8), -(-(y1 - y0) // 8), 1)
                compute.end()
                device.queue.submit([encoder.finish()])
                device.queue.read_buffer(accum, 0, 16)     # wait for this band: keeps submissions short and cancellation prompt
                resources.pop().destroy()
            tile_count[~tile_done] += take
            sample_index += take
            passes += 1
            if settings.noise_threshold > 0 and not data_pass:
                raw = np.frombuffer(device.queue.read_buffer(accum), "f4").reshape(-1, 2, 4)
                lum_sum, lum_sq = raw[:, 1, 0].astype(np.float64), raw[:, 1, 1].astype(np.float64)
                count = tile_count[tile_of]
                pt._retire_tiles(tile_done, tile_of, lum_sum, lum_sq, count, tiles_x * tiles_y, settings.noise_threshold)
            elapsed = time.perf_counter() - started
            if progress is not None:
                progress("pathtrace", sample_index / total_samples,
                         dict(samples=sample_index, passes=passes, seconds=elapsed, tiles_active=int((~tile_done).sum())))
            if settings.time_limit and elapsed >= settings.time_limit:
                break
        raytrace._cancel(cancel)
        raw = np.frombuffer(device.queue.read_buffer(accum), "f4").reshape(width * height, 2, 4).astype(np.float64)
    finally:
        for resource in reversed(resources):
            resource.destroy()
    count = np.maximum(tile_count[tile_of], 1).astype(np.float64)
    if data_pass:
        image = raw[:, 0, :]
        image = np.where(image[:, 3:4] > 0, image, 0.0)
    else:
        image = np.zeros((width * height, 4))
        image[:, :3] = raw[:, 0, :3] / count[:, None]
        image[:, 3] = raw[:, 0, 3] / count
        if output == "rgba":
            bg = np.asarray(background, np.float64).copy()
            bg[3] = np.clip(bg[3], 0, 1)
            bg[:3] *= bg[3]
            image = image + bg * (1 - image[:, 3:4])
    if stats is not None:
        stats.update(backend="gpu", samples=tile_count[tile_of].reshape(height, width).copy(), passes=passes,
                     seconds=time.perf_counter() - started, adapter=gpu3d.describe())
    result = image.reshape(height, width, 4).astype(np.float32)
    result.flags.writeable = False
    return result
