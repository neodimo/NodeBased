"""GPU path tracer: the WGSL twin of `nodebased/pathtrace.py`.

One compute invocation owns one pixel and runs `spp` complete paths for it per dispatch, so a pass is
one dispatch (in row bands, each within the adapter's submission budget). The maths, the random numbers
(`pcg`, `rand`) and the light and BSDF conventions are the CPU reference's; only the arithmetic differs
(f32 here, f64 there), and the two agree statistically, not bit for bit.

Scene layout, seven storage bindings and one uniform: `nodes`/`order` hold one top-level tree over the
shapes (instances and geometries alike) followed by one bottom-level tree per unique mesh, so N
instances of a mesh upload its triangles once and are never flattened, and then the splats' own tree;
`triangles` carries positions, normals and uvs; `shapes` each shape's inverse matrix, material and bottom-level
root; `lights` the analytic lights; `env` the environment map (RGB and luminance texels, then the marginal and
conditional sampling tables) and, after it, the splat records, their spherical harmonics, the volume headers
and grids and the fire table; `accum` two vec4 per pixel (channel sum and coverage; luminance moments, sample count and the adaptive done flag),
then, after all pixels, one done-flag float per pixel (adaptive sampling reads only that tail after each pass).

Splats (ellipsoidal Gaussians met with probability alpha) and smoke and fire (delta tracking) follow
`ptsplats.py` and `ptvolume.py`. The shader is compiled per scene kind (`shader_source`): the splat and
smoke code is only in the variants whose scene has them, because it costs registers whether it runs or not.

Between frames the device keeps what does not change (`_Frame`): the pipelines, the scene's storage buffers while the next frame
packs to the same bytes, the image accumulator (cleared on the card), a uniform buffer and bind group per dispatch slot, and the
readback buffers. A fixed-sampling render puts several passes in one submission (about `BAND_TARGET_SECONDS` of work), and a caller
that sets `stats["skip_noise_maps"]` gets the image finished on the card (`_RESOLVE_SHADER`), 16 bytes a pixel over the bus.
`reuse_counters` counts what was built and what was reused.

Scope, named rather than assumed: textures, more than one environment and particles raise
`gpu3d.Unsupported` (callers fall back to the CPU reference).
"""
import math
import struct
import threading
import time

import numpy as np

from . import gpu3d, gpurt, gpuvolume, lens, pathtrace as pt, raytrace, scene3d as s

GPU_PATHS_PER_SUBMISSION = 1 << 19
ADAPTIVE_PATHS_PER_DISPATCH = 1 << 20   # adaptive sampling: paths one indirect dispatch may cover (a heavy scene gets SOFT_SLOWDOWN times fewer)
ADAPTIVE_MAX_GROUPS = 32768              # and work-groups, well under the 65535 a dispatch dimension allows
ADAPTIVE_SPARSE_PIXELS = 1 << 17         # at most this many open pixels get their pass's samples run side by side (sample-parallel tail)
ADAPTIVE_SPARSE_THREADS = 1 << 21        # and at most this many (pixel, sample) threads: 64 MiB of per-sample sums after the image
ADAPTIVE_PACE = 4                        # passes the host lets the card run ahead before it waits (it waits every pass when asked for progress)
SOFT_SLOWDOWN = 8          # splats and smoke make a path this much heavier: bands get this much smaller
BAND_TARGET_SECONDS = 0.008   # a smoke or splat render grows its row bands until one dispatch takes about this long (0: fixed bands)
BAND_MAX_PATHS = 1 << 22      # and never past this many paths in one dispatch, however fast the first bands were
MAX_JOBS = 32                 # dispatches one submission may carry (one uniform slot and bind group each)
SURFACE_PASS_SAMPLES = 4      # samples a surface-only fixed render takes per pixel in one dispatch when the settings leave it open (8 made X1 on the Radeon 1.4 times slower than 4 and left Z1 on the RTX 0.7 ms faster; 2 to 6 are slower on X1 and Y1 on both)
SOFT_PASS_SAMPLES = 2         # and a smoke or splat render (more is slower on the RTX 3080 Ti and the Radeon, step R2)
CACHE_LIMIT_BYTES = 128 << 20  # a scene whose buffers, host copies and image need more than this is not kept between frames
# what the renderer built or reused since the process started (tests read the difference around a render)
COUNTERS = dict(pipelines=0, resolve_pipelines=0, scene_uploads=0, scene_reuses=0, accum_buffers=0, accum_reuses=0,
                bind_groups=0, uniform_buffers=0, staging_buffers=0, submissions=0)


def reuse_counters():
    """A copy of `COUNTERS`: pipelines compiled, scene uploads and reuses, image accumulators made and reused, bind groups and
    uniform buffers created, readback buffers made, command buffers submitted."""
    return dict(COUNTERS)
COUNT_COLLISIONS = False   # compile the volume counting variant: a render's rgb is the mean (tentative, real, region hops) per path
ENABLE_VOLUME_SKIP = True  # allows a same-shader baseline for the 64-sample image comparison
STACK = 64
TLAS_STACK = 32
WG_SIZE = 8
WG_SIZE_SPLATS_AND_VOLUMES = 4  # 8x8 miscompiles this variant (and splats with light links) on AMD (wrong albedo, 10x-dark beauty): fewer
                                # threads per workgroup gives the driver's register allocator room (0.31.0 tag, 9/29)
_OUTPUT_CODES = {"rgba": 0, "diffuse": 1, "specular": 2, "emission": 3, "albedo": 4, "diffuse_indirect": 5,
                 "specular_indirect": 6, "depth": 7, "normals": 8, "position": 9, "uv": 10, "object_id": 11}
_KIND_CODES = {"Directional": 0, "Point": 1, "Spot": 2, "Rect": 3, "Disc": 4, "Sphere": 5}
LIGHT_VECS = 6
SPLAT_VECS = 9
VOL_VECS = 10
SHAPE_VECS = 13
TRI_VECS = 8


def _wg_size(splats, volumes, links=False):
    # light linking on a scene with splats is the second variant AMD's driver miscompiled at 8x8 (wrong colour channels)
    return WG_SIZE_SPLATS_AND_VOLUMES if (splats and (volumes or links)) else WG_SIZE


def soft_supported(state):
    """Whether splats and smoke may be path traced on this adapter. The 0.31.0 tag showed the shader's splat and
    smoke variants giving wrong pictures on Microsoft's software driver (NaN, 27 tests) and on an AMD integrated
    GPU (wrong albedo and a beauty pass about ten times too dark when splats and smoke shared a scene, 6 tests),
    while an NVIDIA card and llvmpipe agreed with the CPU reference. The cause on AMD: the splats-and-volumes
    shader variant at an 8x8 workgroup miscompiled on AMD's driver (RADV/ACO); a 4x4 workgroup for that variant
    (`_wg_size`) fixed every case measured here (0.31.1, 9/29) and AMD now runs on the GPU too. Microsoft's
    software driver is still untested (nobody here can run D3D12) and stays off the GPU path until it is.
    NB_GPU_SOFT=1 lifts the limit (for debugging)."""
    import os
    if os.environ.get("NB_GPU_SOFT") == "1":
        return True
    info = state.get("info", {})
    text = " ".join(str(info.get(key, "")) for key in ("vendor", "device", "description")).lower()
    if "microsoft" in text or "basic render" in text:
        return False
    return ("nvidia" in text or "llvmpipe" in text or "amd" in text or "radv" in text
            or str(info.get("vendor_id", "")).lower() in ("4318", "0x10de", "4098", "0x1002"))


def check_capability(state):
    for name, minimum in [("max-storage-buffers-per-shader-stage", 8), ("max-storage-buffer-binding-size", 64),
                          ("max-buffer-size", 64), ("max-compute-invocations-per-workgroup", 64),
                          ("max-compute-workgroup-size-x", 8), ("max-compute-workgroups-per-dimension", 1)]:
        if gpurt._limits(state).get(name, 0) < minimum:
            return f"GPU path tracing unavailable: {name} too small (needs {minimum})"
    if "device" in state and "wgpu" in state:
        try:
            _pipeline(state)
            _compact_pipelines(state)
        except Exception as exc:
            return f"GPU path tracing compute unavailable: {exc}"
    return None


_SHADER = r'''
struct Node { lo: vec3<f32>, left: i32, hi: vec3<f32>, right: i32, offset: u32, count: u32, pad: vec2<u32> };
struct Tri { v0: vec4<f32>, e1: vec4<f32>, e2: vec4<f32>, n0: vec4<f32>, n1: vec4<f32>, n2: vec4<f32>,
             uv01: vec4<f32>, uv2: vec4<f32> };
struct Shape { i0: vec4<f32>, i1: vec4<f32>, i2: vec4<f32>, base: vec4<f32>, mat: vec4<f32>, mat2: vec4<f32>,
               sigma: vec4<f32>, ids: vec4<u32>,
               // PBR texture maps (materials 3, step X3): tex0 = (base, metallic-roughness, normal,
               // occlusion) offsets into `env` (vec4 index, 0xffffffff = no map); tex1 = (emissive offset,
               // base dims, mr dims, normal dims); tex2 = (occlusion dims, emissive dims, pad, pad); dims
               // pack as width<<16 | height. pbr0/pbr1 = (normal_scale, occlusion_strength, emissive_color).
               tex0: vec4<u32>, tex1: vec4<u32>, tex2: vec4<u32>, pbr0: vec4<f32>, pbr1: vec4<f32> };
struct Params {
  a: vec4<u32>,        // width, height, row0, row1 (an adaptive render: the word where `open_pixels` lists its first pixel, the word that holds the list's length)
  b: vec4<u32>,        // sample_base (an adaptive render: this dispatch's first slot in the list), spp, seed, max_bounces
  c: vec4<u32>,        // diffuse cap, specular cap, transmission cap, light count
  d: vec4<u32>,        // env width, env height (0 = no environment), output code, tiles per row
  e: vec4<u32>,        // shape count (0 = empty scene), env cdf base (in vec4s), env visible_to_camera, linked lights (bit mask)
  g: vec4<u32>,        // mesh shape count, splat count, splat record base (vec4s), splat tree root
  h: vec4<u32>,        // volume count, volume header base (vec4s), fire table base (vec4s), flags (1 splat SH, 2 fire)
  right: vec4<f32>,    // camera right, aspect
  up: vec4<f32>,       // camera up, 1 / focal
  forward: vec4<f32>,  // camera forward, near
  eye: vec4<f32>,      // eye, far
  f: vec4<f32>,        // eps, shadow eps, ambient, env scale
  gain: vec4<f32>,     // environment gain rgb
  vp0: vec4<f32>,      // smoke: extinction per unit density, scattering share, anisotropy, density scale
  vp1: vec4<f32>,      // smoke colour, shadow density
  vp2: vec4<f32>,      // temperature scale, fire threshold, fire intensity, shadow steps
  m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,   // world -> map rotation rows
  lens: vec4<f32>,     // aperture radius (0 = pinhole), focus distance, blades, blade rotation (radians)
  lens2: vec4<f32>,    // 1 / anamorphic squeeze, max samples (adaptive)
  ad: vec4<f32>,       // adaptive sampling: noise threshold, min samples, 2 when adaptive (0 = every pixel takes every sample), pass size
  vp3: vec4<f32>,      // smoke: volume_multi_scatter amount, volume_fire_light gain
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
// adaptive sampling: the pixels that are not done yet, listed on the device by `_COMPACT_SHADER` (a.z = first word of the
// list, a.w = the word holding its length)
@group(0) @binding(8) var<storage, read> open_pixels: array<u32>;

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
//#if LINKS
// Light linking: a shape's `sigma.w` holds the bit mask of the lights it excludes (light `li` is bit `li` in the packed
// order, the environment is bit `light_count`). A shadow ray (any_hit) toward light `shadow_bit` skips the shapes that
// exclude it; NO_LIGHT is any other any_hit.
const NO_LIGHT: u32 = 99u;
var<private> shadow_bit: u32 = 99u;
fn excl_bit(mask: u32, bit: u32) -> bool { return bit < 24u && ((mask >> bit) & 1u) != 0u; }
// the lights a hit surface excludes: a mesh shape's `sigma.w`, or a splat record's spare z (read where it is used, not held)
//#if SPLATS
fn recv_mask(shape: i32, splat: i32) -> u32 {
  if (splat >= 0) { return u32(sp(u32(splat), 7u).z); }
  return u32(shapes[shape].sigma.w);
}
//#else
fn recv_mask(shape: i32, splat: i32) -> u32 { return u32(shapes[shape].sigma.w); }
//#endif
//#endif

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
//#if LINKS
      if (any_hit && excl_bit(u32(sh.sigma.w), shadow_bit)) { continue; }
//#endif
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
struct Surf { pos: vec3<f32>, ns: vec3<f32>, ng: vec3<f32>, uv: vec2<f32>, base: vec3<f32>, alpha: f32,
               metallic_ovr: f32, roughness_ovr: f32, occlusion: f32, emissive: vec3<f32> };

fn unit(v: vec3<f32>) -> vec3<f32> { return v / max(length(v), 1e-30); }

const NO_TEX: u32 = 0xffffffffu;

// Bilinear, edge-clamp sample of the top (and only) mip packed into `env` at `offset` (vec4 index),
// `dims` = width<<16 | height; matches `scene3d._sample` (v=0 is the bottom row).
fn sample_tex(offset: u32, dims: u32, u_in: f32, v_in: f32) -> vec4<f32> {
  let w = i32(dims >> 16u);
  let h = i32(dims & 0xffffu);
  let x = clamp(u_in, 0.0, 1.0) * f32(w) - 0.5;
  let y = (1.0 - clamp(v_in, 0.0, 1.0)) * f32(h) - 0.5;
  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let fx = x - f32(x0);
  let fy = y - f32(y0);
  let x0c = clamp(x0, 0, w - 1);
  let y0c = clamp(y0, 0, h - 1);
  let x1c = clamp(x0 + 1, 0, w - 1);
  let y1c = clamp(y0 + 1, 0, h - 1);
  let c00 = env[offset + u32(y0c * w + x0c)];
  let c10 = env[offset + u32(y0c * w + x1c)];
  let c01 = env[offset + u32(y1c * w + x0c)];
  let c11 = env[offset + u32(y1c * w + x1c)];
  return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

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
  s.metallic_ovr = -1.0;
  s.roughness_ovr = -1.0;
  s.occlusion = 1.0;
  s.emissive = vec3<f32>(0.0);
  var base = sh.base.xyz;
  var alpha = sh.base.w;
  // PBR texture maps (materials 3, step X3): same overrides and blending as pathtrace.py's `_surface`.
  if (sh.tex0.x != NO_TEX) {
    let texel = sample_tex(sh.tex0.x, sh.tex1.y, s.uv.x, s.uv.y);
    let a = max(texel.w, 1e-6);
    base = base * (texel.xyz / a);
    alpha = alpha * texel.w;
  }
  if (sh.tex0.y != NO_TEX) {
    let texel = sample_tex(sh.tex0.y, sh.tex1.z, s.uv.x, s.uv.y);
    s.roughness_ovr = texel.y;
    s.metallic_ovr = texel.z;
  }
  if (sh.tex0.z != NO_TEX) {
    let texel = sample_tex(sh.tex0.z, sh.tex1.w, s.uv.x, s.uv.y);
    let local = (texel.xyz * 2.0 - 1.0) * vec3<f32>(sh.pbr0.x, sh.pbr0.x, 1.0);
    let tw = sh.i0.xyz * tri.v0.w + sh.i1.xyz * tri.e1.w + sh.i2.xyz * tri.e2.w;
    var t = frame_t(s.ns);
    if (dot(tw, tw) >= 1e-24) { t = unit(tw - s.ns * dot(tw, s.ns)); }
    let b = cross(s.ns, t);
    s.ns = unit(t * local.x + b * local.y + s.ns * local.z);
  }
  if (sh.tex0.w != NO_TEX) {
    let texel = sample_tex(sh.tex0.w, sh.tex2.x, s.uv.x, s.uv.y);
    s.occlusion = 1.0 - sh.pbr0.y * (1.0 - texel.x);
  }
  if (sh.tex1.x != NO_TEX) {
    let texel = sample_tex(sh.tex1.x, sh.tex2.y, s.uv.x, s.uv.y);
    s.emissive = vec3<f32>(sh.pbr0.z, sh.pbr0.w, sh.pbr1.x) * texel.xyz;
  } else if (sh.pbr0.z != 0.0 || sh.pbr0.w != 0.0 || sh.pbr1.x != 0.0) {
    s.emissive = vec3<f32>(sh.pbr0.z, sh.pbr0.w, sh.pbr1.x);
  }
  s.base = base;
  s.alpha = alpha;
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

fn make_lobe(sh: Shape, base: vec3<f32>, nv: f32, metallic_ovr: f32, roughness_ovr: f32, occlusion: f32) -> Lobe {
  var l: Lobe;
  var m = sh.mat.x;
  if (metallic_ovr >= 0.0) { m = metallic_ovr; }
  var r = sh.mat.y;
  if (roughness_ovr >= 0.0) { r = roughness_ovr; }
  let f0d = sh.mat.z;
  let ab = dfg(nv, r);
  let comp = 1.0 / max(ab.x + ab.y, 1e-4);
  l.f0 = vec3<f32>(f0d * (1.0 - m)) + base * m;
  l.k = vec3<f32>(1.0) + l.f0 * (comp - 1.0);
  l.spec_albedo = (l.f0 * ab.x + vec3<f32>(ab.y)) * l.k;
  let dielectric = (f0d * ab.x + ab.y) * (1.0 + f0d * (comp - 1.0));
  if (sh.mat2.x < 0.5) { l.diffuse = base; } else { l.diffuse = base * ((1.0 - m) * (1.0 - dielectric)); }
  l.diffuse = l.diffuse * occlusion;
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
//#if LINKS
// Light linking: `params.e.w` is the bit mask of the lights some shape excludes. Such a light is sampled by next-event
// estimation alone (its shadow rays skip the excluding shapes, a BSDF ray cannot): weight 1 for the light sample, 0 for
// the BSDF hit, except after a perfect mirror (see pathtrace.PathScene.mis).
fn lmis(bit: u32, pa: f32, pb: f32) -> f32 {
  if (excl_bit(params.e.w, bit)) { return 1.0; }
  return mis(pa, pb);
}
fn lhit(bit: u32, prev_delta: bool, prev_pdf: f32, pdf_light: f32) -> f32 {
  if (prev_delta) { return 1.0; }
  if (excl_bit(params.e.w, bit)) { return 0.0; }
  return mis(prev_pdf, pdf_light);
}
//#endif

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

//#if SOFT
fn fenv(i: u32) -> f32 { return env[i >> 2u][i & 3u]; }
//#endif
//#if SPLATS
// ------------------------------------------------------------------------------------------------ splats
// A splat is an ellipsoidal Gaussian met with probability alpha (see ptsplats.py). Records live in the `env`
// buffer, 9 vec4 each: 0 position + hit opacity, 1 scale + shadow opacity, 2..4 axis columns + albedo,
// 5 normal + confidence, 6 (roughness, metallic, pbr, relight), 7 (min scale, max scale), 8 u32 (instance,
// SH float offset, SH degree, linear flag).
const SPLAT_VECS: u32 = 9u;
const SPLAT_ALPHA_FLOOR: f32 = 0.00392156862745098;
const SPLAT_START_SCALE: f32 = 1.0;
const COPLANAR_COS: f32 = 0.94;
const SPLAT_DIM: u32 = 56u;
var<private> splat_stack: array<i32, 64>;

fn sp(i: u32, k: u32) -> vec4<f32> { return env[params.g.z + i * SPLAT_VECS + k]; }
// Record 8 holds integers as ordinary float values (instance, SH offset low 24 bits, degree + 16 * linear,
// SH offset high bits). They used to be raw u32 bit patterns read back with bitcast, which are denormals and
// NaNs as floats; AMD and Microsoft's software driver do not hand those back bit for bit (0.31.0 tag, 9/29).
fn splat_info(i: u32) -> vec4<u32> {
  let r = sp(i, 8u);
  let packed = u32(r.z);
  return vec4<u32>(u32(r.x), u32(r.y) + (u32(r.w) << 24u), packed & 15u, packed >> 4u);
}

// (t, alpha) of splat `i` on the ray bounded by [lower, upper]: the closest approach, truncated at three sigma
fn splat_terms(i: u32, o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32, shadow: bool) -> vec2<f32> {
  let a = sp(i, 0u);
  let b = sp(i, 1u);
  let c0 = sp(i, 2u).xyz;
  let c1 = sp(i, 3u).xyz;
  let c2 = sp(i, 4u).xyz;
  let inv = vec3<f32>(1.0) / max(b.xyz, vec3<f32>(1e-12));
  let rel = o - a.xyz;
  let lo_o = vec3<f32>(dot(c0, rel), dot(c1, rel), dot(c2, rel)) * inv;
  let lo_d = vec3<f32>(dot(c0, d), dot(c1, d), dot(c2, d)) * inv;
  let dd = dot(lo_d, lo_d);
  var t = 0.0;
  if (dd > 0.0) { t = -dot(lo_o, lo_d) / dd; }
  t = min(max(t, lower), upper);
  let q = lo_o + t * lo_d;
  let d2 = dot(q, q);
  var alpha = 0.0;
  if (d2 <= 9.0 && lower <= upper) {
    alpha = min(0.99, select(a.w, b.w, shadow) * exp(-0.5 * d2));
  }
  return vec2<f32>(t, alpha);
}

// a splat lying in the tangent plane of the surface the ray leaves, with a parallel normal, is that surface's own
fn splat_coplanar(i: u32, o: vec3<f32>, pn: vec3<f32>, pth: f32) -> bool {
  let b = sp(i, 1u).xyz;
  var axis = 0u;
  if (b.y < b.x) { axis = 1u; }
  if (b.z < min(b.x, b.y)) { axis = 2u; }
  let own = sp(i, 2u + axis).xyz;
  let height = abs(dot(sp(i, 0u).xyz - o, pn));
  return abs(dot(own, pn)) > COPLANAR_COS && height < 3.0 * (min(b.x, min(b.y, b.z)) + pth);
}

struct SplatHit { t: f32, id: i32 };

// nearest splat that is present on this ray: each candidate is there with probability alpha, decided by a hash
fn splat_nearest(o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32, key: u32, dim: u32, exclude: i32,
                 pn: vec3<f32>, pth: f32) -> SplatHit {
  var best = SplatHit(upper, -1);
  var top = 1u;
  splat_stack[0] = i32(params.g.w);
  loop {
    if (top == 0u) { break; }
    top--;
    let index = splat_stack[top];
    if (entry(index, o, d, lower, best.t) == INF) { continue; }
    let node = nodes[index];
    if (node.count == 0u) {
      splat_stack[top] = node.left;
      splat_stack[top + 1u] = node.right;
      top += 2u;
      continue;
    }
    for (var p = 0u; p < node.count; p++) {
      let i = order[node.offset + p];
      if (i32(i) == exclude) { continue; }
      let r = splat_terms(i, o, d, lower, upper, false);
      if (r.y < SPLAT_ALPHA_FLOOR || r.x >= best.t) { continue; }
      if (splat_coplanar(i, o, pn, pth)) { continue; }
      if (rnd(pcg(key + i), dim) < r.y) { best = SplatHit(r.x, i32(i)); }
    }
  }
  return best;
}

// the data passes' rule: the first splat, front to back, where the accumulated opacity reaches one half
fn splat_first_opaque(o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32) -> SplatHit {
  var t_prev = -1.0;
  var id_prev = -1;
  var running = 0.0;
  for (var peel = 0u; peel < 512u; peel++) {
    var best = SplatHit(upper, -1);
    var best_alpha = 0.0;
    var top = 1u;
    splat_stack[0] = i32(params.g.w);
    loop {
      if (top == 0u) { break; }
      top--;
      let index = splat_stack[top];
      if (entry(index, o, d, lower, best.t) == INF) { continue; }
      let node = nodes[index];
      if (node.count == 0u) {
        splat_stack[top] = node.left;
        splat_stack[top + 1u] = node.right;
        top += 2u;
        continue;
      }
      for (var p = 0u; p < node.count; p++) {
        let i = order[node.offset + p];
        let r = splat_terms(i, o, d, lower, upper, false);
        if (r.y < SPLAT_ALPHA_FLOOR) { continue; }
        if (r.x < t_prev || (r.x == t_prev && i32(i) <= id_prev)) { continue; }
        if (best.id < 0 || r.x < best.t || (r.x == best.t && i32(i) < best.id)) {
          best = SplatHit(r.x, i32(i));
          best_alpha = r.y;
        }
      }
    }
    if (best.id < 0) { return best; }
    running += log(1.0 - best_alpha);
    if (1.0 - exp(running) >= 0.5) { return best; }
    t_prev = best.t;
    id_prev = best.id;
  }
  return SplatHit(upper, -1);
}

// `prod(1 - alpha)` of the casters a shadow ray crosses
fn splat_transmittance(o: vec3<f32>, d: vec3<f32>, lower: f32, upper: f32, exclude: i32, pn: vec3<f32>,
                       pth: f32) -> f32 {
  var out = 1.0;
  var top = 1u;
  splat_stack[0] = i32(params.g.w);
  loop {
    if (top == 0u) { break; }
    top--;
    let index = splat_stack[top];
    if (entry(index, o, d, lower, upper) == INF) { continue; }
    let node = nodes[index];
    if (node.count == 0u) {
      splat_stack[top] = node.left;
      splat_stack[top + 1u] = node.right;
      top += 2u;
      continue;
    }
    for (var p = 0u; p < node.count; p++) {
      let i = order[node.offset + p];
      if (i32(i) == exclude) { continue; }
//#if LINKS
      if (excl_bit(u32(sp(i, 7u).z), shadow_bit)) { continue; }
//#endif
      let r = splat_terms(i, o, d, lower, upper, true);
      if (r.y > 0.0 && !splat_coplanar(i, o, pn, pth)) {
        out = out * (1.0 - r.y);
        if (out < 1e-5) { return 0.0; }
      }
    }
  }
  return out;
}

fn sh_coef(base: u32, k: u32) -> vec3<f32> {
  let f = base + k * 3u;
  return vec3<f32>(fenv(f), fenv(f + 1u), fenv(f + 2u));
}

// splats.eval_sh followed by the render-time sRGB curve when the capture is sRGB
fn splat_capture(sinfo: vec4<u32>, d: vec3<f32>) -> vec3<f32> {
  let base = sinfo.y;
  let degree = sinfo.z;
  let x = d.x;
  let y = d.y;
  let z = d.z;
  var rgb = 0.28209479177387814 * sh_coef(base, 0u);
  if (degree >= 1u) {
    rgb += -0.4886025119029199 * y * sh_coef(base, 1u) + 0.4886025119029199 * z * sh_coef(base, 2u)
         - 0.4886025119029199 * x * sh_coef(base, 3u);
  }
  if (degree >= 2u) {
    rgb += 1.0925484305920792 * x * y * sh_coef(base, 4u) - 1.0925484305920792 * y * z * sh_coef(base, 5u)
         + 0.31539156525252005 * (2.0 * z * z - x * x - y * y) * sh_coef(base, 6u)
         - 1.0925484305920792 * x * z * sh_coef(base, 7u) + 0.5462742152960396 * (x * x - y * y) * sh_coef(base, 8u);
  }
  if (degree >= 3u) {
    rgb += -0.5900435899266435 * y * (3.0 * x * x - y * y) * sh_coef(base, 9u)
         + 2.890611442640554 * x * y * z * sh_coef(base, 10u)
         - 0.4570457994644658 * y * (4.0 * z * z - x * x - y * y) * sh_coef(base, 11u)
         + 0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y) * sh_coef(base, 12u)
         - 0.4570457994644658 * x * (4.0 * z * z - x * x - y * y) * sh_coef(base, 13u)
         + 1.445305721320277 * z * (x * x - y * y) * sh_coef(base, 14u)
         - 0.5900435899266435 * x * (x * x - 3.0 * y * y) * sh_coef(base, 15u);
  }
  rgb = max(rgb + vec3<f32>(0.5), vec3<f32>(0.0));
  if (sinfo.w == 0u) {
    let lo = rgb / 12.92;
    let hi = pow((rgb + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    rgb = select(hi, lo, rgb <= vec3<f32>(0.04045));
  }
  return rgb;
}

struct SplatSurf { ns: vec3<f32>, base: vec3<f32>, emit: vec3<f32>, oid: f32 };

// ptsplats.surface: the normal turned to the viewer and blended toward it by its confidence, the de-lit albedo, and
// the capture's own colour under the instance's `1 - relight` share as emission
fn splat_surface(i: u32, wo: vec3<f32>, wd: vec3<f32>) -> SplatSurf {
  let nrm = sp(i, 5u);
  var facing = nrm.xyz;
  if (dot(facing, wo) < 0.0) { facing = -facing; }
  let eff = unit(nrm.w * facing + (1.0 - nrm.w) * wo);
  var s: SplatSurf;
  s.ns = eff;
  s.base = vec3<f32>(sp(i, 2u).w, sp(i, 3u).w, sp(i, 4u).w);
  let sinfo = splat_info(i);
  s.oid = f32(params.g.x + 1u + sinfo.x);
  let share = 1.0 - sp(i, 6u).w;
  if (share > 0.0 && (params.h.w & 1u) != 0u) { s.emit = splat_capture(sinfo, wd) * share; }
  return s;
}

fn splat_shape(i: u32) -> Shape {
  let m = sp(i, 6u);
  var s: Shape;
  s.base = vec4<f32>(sp(i, 2u).w, sp(i, 3u).w, sp(i, 4u).w, 1.0);
  s.mat = vec4<f32>(m.y, m.x, select(0.0, 0.04, m.z > 0.5), 0.0);
  s.mat2 = vec4<f32>(select(0.0, 1.0, m.z > 0.5), 1.0, 1.0, 0.0);
  return s;
}

//#endif
//#if VOLUMES
// ------------------------------------------------------------------------------------------------ smoke and fire
// Volumes are delta tracked (see ptvolume.py). Headers (10 vec4 each) and grids live in the `env` buffer:
// 0 box min + voxel size, 1 box max + majorant, 2..4 world-to-object rows, 5 grid size, 6 the grid offsets as
// ordinary float values (density low 24 bits, temperature low 24 bits or -1 for none, density high bits,
// temperature high bits), 7 coarse-majorant offset low/high and tile edge, 8 a sparse volume's tile index offset
// low/high, 1 when the grids are tile atlases, and the value of an empty tile, 9 the coarse majorant offset low/high and the
// number of fine cells along a coarse cell; see splat_info for why
// integer offsets are not raw bit patterns.
const VOL_VECS: u32 = 10u;
const MAX_COLLISIONS: u32 = 4096u;
const VOL_ABSORB_DIM: u32 = 54u;
const FIRE_KNOTS: f32 = 64.0;

fn vh(vi: u32, k: u32) -> vec4<f32> { return env[params.h.y + vi * VOL_VECS + k]; }
// The volume the last real collision of `vol_flight_all` happened in. Light linking: header vec4 7's `.w` is the bit mask of
// the lights that volume excludes (the packed light numbering): they do not scatter in it and it casts no shadow from them.
var<private> hit_volume: u32 = 0u;
//#if COUNT
// Debug variant (`COUNT_COLLISIONS`): the tentative collisions, real collisions and region hops of a path, summed into
// the pixel instead of its colour (tools/benchmark_volume_majorant.py).
var<private> vol_counts: vec3<f32> = vec3<f32>(0.0);
//#endif
//#if LINKS
fn vol_reaches(bit: u32) -> bool { return !excl_bit(u32(vh(hit_volume, 7u).w), bit); }
//#else
fn vol_reaches(bit: u32) -> bool { return true; }
//#endif
fn vol_bases(vi: u32) -> vec4<u32> {
  let r = vh(vi, 6u);
  var temperature = 0xFFFFFFFFu;
  if (r.y >= 0.0) { temperature = u32(r.y) + (u32(r.w) << 24u); }
  return vec4<u32>(u32(r.x) + (u32(r.z) << 24u), temperature, 0u, 0u);
}

// `sp` is the volume's sparse header (vh(vi, 8)): a sparse grid is stored as 8-cubed tiles one after another (the
// SparseGrid blocks as they are) behind a table of slot + 1 per tile, 0 for an empty tile, which reads `sp.w`.
fn grid_at(base: u32, dims: vec3<i32>, sp: vec4<f32>, x: i32, y: i32, z: i32) -> f32 {
  if (x < 0 || y < 0 || z < 0 || x >= dims.x || y >= dims.y || z >= dims.z) { return 0.0; }
//#if SPARSE
  if (sp.z > 0.5) {
    let nty = (dims.y + 7) >> 3u;
    let ntz = (dims.z + 7) >> 3u;
    let slot = fenv(u32(sp.x) + (u32(sp.y) << 24u) + u32(((x >> 3u) * nty + (y >> 3u)) * ntz + (z >> 3u)));
    if (slot <= 0.0) { return sp.w; }
    return fenv(base + (u32(slot) - 1u) * 512u + u32((((x & 7) * 8 + (y & 7)) * 8) + (z & 7)));
  }
//#endif
  return fenv(base + u32((x * dims.y + y) * dims.z + z));
}

// zero-padded trilinear sample of a cell-centred grid at index-space point `g`
fn trilinear(base: u32, dims: vec3<i32>, sp: vec4<f32>, g: vec3<f32>) -> f32 {
  let fl = floor(g);
  let i = vec3<i32>(fl);
  let f = g - fl;
  var out = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let dx = i32(k & 1u);
    let dy = i32((k >> 1u) & 1u);
    let dz = i32((k >> 2u) & 1u);
    let w = select(1.0 - f.x, f.x, dx == 1) * select(1.0 - f.y, f.y, dy == 1) * select(1.0 - f.z, f.z, dz == 1);
    out += w * grid_at(base, dims, sp, i.x + dx, i.y + dy, i.z + dz);
  }
  return out;
}

fn vol_object(vi: u32, p: vec3<f32>) -> vec3<f32> {
  let r0 = vh(vi, 2u);
  let r1 = vh(vi, 3u);
  let r2 = vh(vi, 4u);
  return vec3<f32>(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
}
fn vol_object_dir(vi: u32, d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(vh(vi, 2u).xyz, d), dot(vh(vi, 3u).xyz, d), dot(vh(vi, 4u).xyz, d));
}

// world-distance interval of a ray inside the volume's box: (enter >= 0, exit); enter = INF when it misses
fn vol_clip(vi: u32, o: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
  let bmin = vh(vi, 0u).xyz;
  let bmax = vh(vi, 1u).xyz;
  let oo = vol_object(vi, o);
  let dd = vol_object_dir(vi, d);
  var near = -INF;
  var far = INF;
  for (var a = 0u; a < 3u; a++) {
    if (abs(dd[a]) < 1e-12) {
      if (oo[a] < bmin[a] || oo[a] > bmax[a]) { return vec2<f32>(INF, -INF); }
    } else {
      let x = (bmin[a] - oo[a]) / dd[a];
      let y = (bmax[a] - oo[a]) / dd[a];
      near = max(near, min(x, y));
      far = min(far, max(x, y));
    }
  }
  return vec2<f32>(max(near, 0.0), far);
}

fn vol_uniform(key: u32, turn: u32, vi: u32, k: u32, slot: u32) -> f32 {
  return rnd(pcg(key + (turn * 7919u + vi * 104729u + 12345u)), k * 4u + slot);
}

// The grid-space world distance at which a ray leaves the cell `cell` of `size` voxels it is in at `position`
// (a face already behind the ray, or within 1e-6 of it, is ignored); `t1` is the cap.
fn vol_leave(cell: vec3<i32>, size: i32, bmin: vec3<f32>, voxel: f32, oo: vec3<f32>, dd: vec3<f32>, position: f32,
             t1: f32) -> f32 {
  var exit = t1;
  for (var axis = 0u; axis < 3u; axis++) {
    if (abs(dd[axis]) > 1e-12) {
      let side = select(cell[axis], cell[axis] + 1, dd[axis] > 0.0);
      let boundary = bmin[axis] + f32(side * size) * voxel;
      let next = (boundary - oo[axis]) / dd[axis];
      if (next > position + 1e-6) { exit = min(exit, next); }
    }
  }
  return exit;
}

// The extinction bound of the region the ray is in at `position` and the distance at which it leaves that region: two
// levels (header vec4 7 holds the fine grid and its cell edge, vec4 9 the coarse grid over blocks of fine cells); an
// empty coarse cell is crossed in one stride, otherwise the fine cell sets both.
fn vol_majorant(vi: u32, oo: vec3<f32>, dd: vec3<f32>, position: f32, t1: f32)
    -> vec2<f32> {
  let h = vh(vi, 7u);
  if (h.x < 0.0) { return vec2<f32>(vh(vi, 1u).w, t1); }
  let edge = i32(h.z);
  let dims = vec3<i32>(vh(vi, 5u).xyz);
  let fine_dims = (dims + vec3<i32>(edge - 1)) / edge;
  let bmin = vh(vi, 0u).xyz;
  let voxel = vh(vi, 0u).w;
  let q = (oo + dd * position - bmin) / voxel;
  let cell = clamp(vec3<i32>(floor(q / f32(edge))), vec3<i32>(0), fine_dims - vec3<i32>(1));
  let hs = vh(vi, 9u);
  let ratio = i32(hs.z);
  let sup_dims = (fine_dims + vec3<i32>(ratio - 1)) / ratio;
  let sup = cell / vec3<i32>(ratio);
  let sup_mu = fenv(u32(hs.x) + (u32(hs.y) << 24u) + u32((sup.x * sup_dims.y + sup.y) * sup_dims.z + sup.z));
  if (sup_mu <= 0.0) {
    return vec2<f32>(0.0, vol_leave(sup, edge * ratio, bmin, voxel, oo, dd, position, t1));
  }
  let mu = fenv(u32(h.x) + (u32(h.y) << 24u) + u32((cell.x * fine_dims.y + cell.y) * fine_dims.z + cell.z));
  return vec2<f32>(mu, vol_leave(cell, edge, bmin, voxel, oo, dd, position, t1));
}

fn fire_radiance(kelvin: f32) -> vec3<f32> {
  var u = log(max(kelvin, 1e-3) / 400.0) / log(8000.0 / 400.0) * (FIRE_KNOTS - 1.0);
  u = clamp(u, 0.0, FIRE_KNOTS - 1.0);
  let i = min(u32(floor(u)), 62u);
  let f = u - f32(i);
  let a = env[params.h.z + i].xyz;
  let b = env[params.h.z + i + 1u].xyz;
  return a * (1.0 - f) + b * f;
}

// free flight through volume `vi` up to `tcap`: the first real collision (INF for none); with `want_glow` also
// sums the fire's emission up to and including that collision into `glow`
fn vol_flight(vi: u32, o: vec3<f32>, d: vec3<f32>, tcap: f32, key: u32, turn: u32, want_glow: bool,
              glow: ptr<function, vec3<f32>>) -> f32 {
  let head = vh(vi, 1u);
  if (head.w <= 1e-8) { return INF; }      // a volume with no density (an empty tile set) holds no collision
  let clip = vol_clip(vi, o, d);
  let t1 = min(clip.y, tcap);
  if (!(t1 > clip.x)) { return INF; }
  var position = clip.x;
  let bmin = vh(vi, 0u).xyz;
  let voxel = vh(vi, 0u).w;
  let dims = vec3<i32>(vh(vi, 5u).xyz);
  let bases = vol_bases(vi);
  let sp = vh(vi, 8u);
  let has_temp = bases.y != 0xFFFFFFFFu;
  let oo = vol_object(vi, o);
  let dd = vol_object_dir(vi, d);
  for (var k = 0u; k < MAX_COLLISIONS; k++) {
    let local = vol_majorant(vi, oo, dd, position, t1);
    let mu = local.x;
    if (mu <= 0.0) {
//#if COUNT
      vol_counts.z += 1.0;
//#endif
      position = local.y + 1e-5;
      if (position >= t1) { break; }
      continue;
    }
    let u_step = vol_uniform(key, turn, vi, k, 0u);
    let candidate = position - log(1.0 - min(u_step, 1.0 - 1e-7)) / mu;
    if (candidate >= local.y) {
//#if COUNT
      vol_counts.z += 1.0;
//#endif
      position = local.y + 1e-5;
      if (position >= t1) { break; }
      continue;
    }
    position = candidate;
    if (position >= t1) { break; }
//#if COUNT
    vol_counts.x += 1.0;
//#endif
    let g = (oo + dd * position - bmin) / voxel - vec3<f32>(0.5);
    let sigma = params.vp0.w * trilinear(bases.x, dims, sp, g);
    if (want_glow && has_temp) {
      let kelvin = params.vp2.x * trilinear(bases.y, dims, sp, g);
      if (kelvin > params.vp2.y && sigma > 0.0) {
        *glow += fire_radiance(kelvin) * (params.vp2.z * sigma / mu);
      }
    }
    if (vol_uniform(key, turn, vi, k, 1u) < params.vp0.x * sigma / mu) {
//#if COUNT
      vol_counts.y += 1.0;
//#endif
      return position;
    }
  }
  return INF;
}

// the first real collision of every volume (independent processes), and the fire seen on the way to it
fn vol_flight_all(o: vec3<f32>, d: vec3<f32>, t_end: f32, key: u32, turn: u32, glow: ptr<function, vec3<f32>>) -> f32 {
  let n = params.h.x;
  let fire = (params.h.w & 2u) != 0u;
  hit_volume = 0u;
  if (n == 1u) { return vol_flight(0u, o, d, t_end, key, turn, fire, glow); }
  var t_event = INF;
  for (var vi = 0u; vi < n; vi++) {
    let te = vol_flight(vi, o, d, min(t_end, t_event), key, turn, false, glow);
    if (te < t_event) { t_event = te; hit_volume = vi; }
  }
  if (fire) {
    var limit = t_end;
    if (t_event < INF) { limit = min(t_end, t_event * 1.000001 + 1e-6); }
    for (var vi = 0u; vi < n; vi++) { _ = vol_flight(vi, o, d, limit, key, turn, true, glow); }
  }
  return t_event;
}

// exp(-tau) through every volume along a shadow ray of unit direction `d` up to `dist`
fn vol_transmittance(o: vec3<f32>, d: vec3<f32>, dist: f32) -> f32 {
  if (params.vp1.w <= 0.0) { return 1.0; }
  let steps = u32(params.vp2.w);
  var out = 1.0;
  for (var vi = 0u; vi < params.h.x; vi++) {
//#if LINKS
    if (excl_bit(u32(vh(vi, 7u).w), shadow_bit)) { continue; }
//#endif
    if (vh(vi, 1u).w <= 1e-8) { continue; }
    let clip = vol_clip(vi, o, d);
    let len = max(min(clip.y, dist) - clip.x, 0.0);
    if (len <= 0.0) { continue; }
    let bmin = vh(vi, 0u).xyz;
    let voxel = vh(vi, 0u).w;
    let dims = vec3<i32>(vh(vi, 5u).xyz);
    let base = vol_bases(vi).x;
    let sp = vh(vi, 8u);
    let oo = vol_object(vi, o);
    let dd = vol_object_dir(vi, d);
    var total = 0.0;
    for (var j = 0u; j < steps; j++) {
      let t = clip.x + (f32(j) + 0.5) / f32(steps) * len;
      total += trilinear(base, dims, sp, (oo + dd * t - bmin) / voxel - vec3<f32>(0.5));
    }
    let tau = params.vp1.w * params.vp0.x * params.vp0.w * total * (len / f32(steps));
    out = out * exp(-tau);
  }
  return out;
}

fn hg_phase(g: f32, cosine: f32) -> f32 {
  let denom = 1.0 + g * g - 2.0 * g * cosine;
  return (1.0 - g * g) / (4.0 * PI * pow(max(denom, 1e-12), 1.5));
}

// a new direction for a path arriving along `d`: the cosine against `d` follows the phase function
struct PhaseSample { dir: vec3<f32>, pdf: f32 };
fn hg_sample(g: f32, d: vec3<f32>, u1: f32, u2: f32) -> PhaseSample {
  var cosine = 1.0 - 2.0 * u1;
  if (abs(g) >= 1e-3) {
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
    cosine = (1.0 + g * g - s * s) / (2.0 * g);
  }
  cosine = clamp(cosine, -1.0, 1.0);
  let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
  let phi = 2.0 * PI * u2;
  var helper = vec3<f32>(1.0, 0.0, 0.0);
  if (abs(d.x) > 0.9) { helper = vec3<f32>(0.0, 1.0, 0.0); }
  let t = unit(cross(helper, d));
  let b = cross(d, t);
  var r: PhaseSample;
  r.dir = unit(cosine * d + (sine * cos(phi)) * t + (sine * sin(phi)) * b);
  r.pdf = hg_phase(g, cosine);
  return r;
}

//#endif
//#if SOFT
// what a shadow ray meets that is soft: splat casters and smoke. `on_splat` (-1 for none) is the splat the ray leaves.
fn soft_visibility(o: vec3<f32>, wi: vec3<f32>, dist: f32, on_splat: i32, ns: vec3<f32>) -> f32 {
  var out = 1.0;
//#if SPLATS
  if (params.g.y > 0u) {
    var lower = 0.01 * params.f.x;
    var thick = -1e30;
    if (on_splat >= 0) {
      let m = sp(u32(on_splat), 7u);
      lower = SPLAT_START_SCALE * m.y;
      thick = m.x;
    }
    out = splat_transmittance(o, wi, lower, dist, on_splat, ns, thick);
  }
//#endif
//#if VOLUMES
  if (params.h.x > 0u && out > 0.0) { out = out * vol_transmittance(o, wi, dist); }
//#endif
  return out;
}
//#endif

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

struct First { shape: i32, t: f32, ns: vec3<f32>, pos: vec3<f32>, uv: vec2<f32>, oid: f32 };

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

// A point of the unit aperture (nodebased/lens.py aperture_points): round, or a regular polygon of `blades` corners.
fn aperture_point(u1: f32, u2: f32) -> vec2<f32> {
  let blades = u32(params.lens.z + 0.5);
  var p: vec2<f32>;
  if (blades < 3u) {
    let r = sqrt(u2);
    let a = 2.0 * PI * u1;
    p = vec2<f32>(r * cos(a), r * sin(a));
  } else {
    let n = min(blades, 16u);
    let scaled = u1 * f32(n);
    let i = min(u32(scaled), n - 1u);
    let r1 = sqrt(u2);
    let r2 = scaled - f32(i);
    let a0 = params.lens.w + 2.0 * PI * f32(i) / f32(n);
    let a1 = params.lens.w + 2.0 * PI * f32((i + 1u) % n) / f32(n);
    let v0 = vec2<f32>(cos(a0), sin(a0));
    let v1 = vec2<f32>(cos(a1), sin(a1));
    p = r1 * ((1.0 - r2) * v0 + r2 * v1);
  }
  return vec2<f32>(p.x * params.lens2.x, p.y);
}

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
//#if LINKS
  var prev_excl = 0u;                  // the exclusion mask of the surface the ray last scattered from (0 for liquid and smoke)
//#endif
  var medium = -1;
  var vdepth = 0u;
//#if SPLATS
  var skip_id = -1;                    // the splat the ray just left, with its plane (ptsplats' coplanar rule)
  var skip_t = 0.0;
  var skip_n = vec3<f32>(0.0);
  var skip_th = 0.0;
  let has_splats = params.g.y > 0u;
  let n_shapes = i32(params.g.x);
//#endif
//#if VOLUMES
  let has_vols = params.h.x > 0u && params.d.z < 7u;
  var smoke_run = 0u;         // smoke vertices in a row so far on this path (`volume_multi_scatter` acts at the second)
//#endif
  let max_b = params.b.w;
  let eps = params.f.x;
  let seps = params.f.y;
  let lc = light_count();
  let has_env = params.d.y > 0u;
  for (var turn = 0u; turn < max_b + 40u; turn++) {
    let dim = DIM_BASE + turn * DIM_STRIDE;
//#if SPLATS
    var h = intersect(o, d, tmin, tmax, false);
    var splat_id = -1;
    if (has_splats) {
      var sph: SplatHit;
      let near = max(tmin, skip_t);
      if (params.d.z >= 7u) { sph = splat_first_opaque(o, d, near, h.t); }
      else { sph = splat_nearest(o, d, near, h.t, key, dim + SPLAT_DIM, skip_id, skip_n, skip_th); }
      if (sph.id >= 0 && sph.t < h.t) {
        h = Hit(sph.t, n_shapes + sph.id, -1, 0.0, 0.0);
        splat_id = sph.id;
      }
    }
//#else
    let h = intersect(o, d, tmin, tmax, false);
//#if LINKS
    let splat_id = -1;
//#endif
//#endif
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
//#if VOLUMES
    var evented = false;
    var t_event = INF;
    if (has_vols) {
      var t_end = h.t;
      if (vdepth >= 1u) { t_end = min(t_end, t_light); }
      var glow = vec3<f32>(0.0);
      t_event = vol_flight_all(o, d, t_end, key, turn, &glow);
      evented = t_event < t_end;
      if (max(glow.x, max(glow.y, glow.z)) > 0.0) {
        // fire seen from a smoke vertex is the fire lighting the smoke: `volume_fire_light` is its gain
        let gathered = thr * glow * select(1.0, params.vp3.y, smoke_run > 0u);
        if (vdepth == 0u) { (*acc).emission += gathered; } else { add_class(acc, cls, vdepth, gathered); }
      }
    }
    if (!evented) { smoke_run = 0u; }
    if (evented) {
      // a real collision in the smoke: absorb, or scatter with light sampling and a phase-sampled continuation
      let pos = o + d * t_event;
      if (vdepth == 0u) {
        (*acc).alpha = 1.0;
        (*acc).albedo = params.vp1.xyz;
      }
      if (!(rnd(key, dim + VOL_ABSORB_DIM) < params.vp0.y) || bounces >= max_b || capped) { break; }
      let g = params.vp0.z;
      // the second smoke vertex in a row: everything the path gathers from here on gains `1 + volume_multi_scatter`, once
      if (smoke_run == 1u) { thr = thr * (1.0 + params.vp3.x); }
      var direct = vec3<f32>(0.0);
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
          if (max(irr.x, max(irr.y, irr.z)) > 0.0 && vol_reaches(li)) {
//#if LINKS
            shadow_bit = li;
//#endif
            if (intersect(pos, wi, 0.0, dist, true).shape < 0) {
              direct += hg_phase(g, dot(wi, d)) * irr * soft_visibility(pos, wi, dist, -1, vec3<f32>(0.0));
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
          if (cos_l > 1e-9 && dist > 1e-9 && vol_reaches(li)) {
            let reach = dist * (1.0 - 1e-4);
//#if LINKS
            shadow_bit = li;
//#endif
            if (intersect(pos, wi, 0.0, reach, true).shape < 0) {
              let pdf_l = dist * dist / max(cos_l * light_field(li, 3u).w, 1e-30);
              let f = hg_phase(g, dot(wi, d));
//#if LINKS
              let wgt = lmis(li, pdf_l, f) / pdf_l * soft_visibility(pos, wi, reach, -1, vec3<f32>(0.0));
//#else
              let wgt = mis(pdf_l, f) / pdf_l * soft_visibility(pos, wi, reach, -1, vec3<f32>(0.0));
//#endif
              direct += f * wgt * l5.xyz;
            }
          }
        }
      }
      if (has_env) {
        let u1 = rnd(key, dim + 2u + slot);
        let u2 = rnd(key, dim + 3u + slot);
        slot += 2u;
        let es = env_sample(u1, u2);
        if (es.pdf > 0.0 && vol_reaches(lc)) {
//#if LINKS
          shadow_bit = lc;
//#endif
          if (intersect(pos, es.dir, 0.0, INF, true).shape < 0) {
            let f = hg_phase(g, dot(es.dir, d));
//#if LINKS
            let wgt = lmis(lc, es.pdf, f) / es.pdf * soft_visibility(pos, es.dir, INF, -1, vec3<f32>(0.0));
//#else
            let wgt = mis(es.pdf, f) / es.pdf * soft_visibility(pos, es.dir, INF, -1, vec3<f32>(0.0));
//#endif
            direct += f * wgt * es.radiance;
          }
        }
      }
      let gathered = thr * params.vp1.xyz * direct;
      if (vdepth == 0u) { (*acc).diffuse += gathered; } else { add_class(acc, cls, vdepth + 1u, gathered); }
      let ps = hg_sample(g, d, rnd(key, dim + BSDF_DIM + 1u), rnd(key, dim + BSDF_DIM + 2u));
      if (cd >= params.c.x) { break; }
      thr = thr * params.vp1.xyz;
      o = pos;
      d = ps.dir;
      cd++;
      capped = cd >= params.c.x;
      if (cls < 0) { cls = 0; }
      prev_delta = false;
      prev_pdf = ps.pdf;
//#if LINKS
      prev_excl = u32(vh(hit_volume, 7u).w);      // the smoke vertex's own link: the lights its volume excludes
//#endif
      bounces++;
      vdepth++;
      smoke_run++;
//#if SPLATS
      skip_id = -1; skip_t = 0.0; skip_th = -1e30;
//#endif
      if (bounces >= 3u) {
        let q = clamp(max(thr.x, max(thr.y, thr.z)), 0.05, 0.95);
        if (rnd(key, dim + BSDF_DIM + 3u) < q) { thr = thr / q; } else { break; }
      }
      continue;
    }
//#endif
    if (light_id >= 0) {
      let li = u32(light_id);
      let l0 = light_field(li, 0u);
      // a camera ray (vdepth 0) only registers a light hit when that light's visible_to_camera is on
      if (vdepth >= 1u || l0.z > 0.5) {
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
//#if LINKS
        weight = lhit(li, prev_delta, prev_pdf, pdf_l);
        if (vdepth >= 1u && excl_bit(prev_excl, li)) { weight = 0.0; }
//#else
        if (!prev_delta) { weight = mis(prev_pdf, pdf_l); }
//#endif
        let gathered = thr * l5.xyz * weight;
        if (vdepth == 0u) {
          (*acc).emission += gathered;
          (*acc).alpha = 1.0;
          (*acc).albedo = l5.xyz;
        } else {
          add_class(acc, cls, vdepth, gathered);
        }
        break;
      }
    }
    if (h.shape < 0) {
      if (vdepth >= 1u) {
        var sky = vec3<f32>(params.f.z);
        if (has_env) {
          var weight = 1.0;
//#if LINKS
          weight = lhit(lc, prev_delta, prev_pdf, env_pdf(d));
          if (excl_bit(prev_excl, lc)) { weight = 0.0; }
//#else
          if (!prev_delta) { weight = mis(prev_pdf, env_pdf(d)); }
//#endif
          sky += env_radiance(d) * weight;
        }
        add_class(acc, cls, vdepth, thr * sky);
      } else if (has_env && params.e.z > 0u) {
        // the environment replaces the flat background on a camera ray that leaves the scene
        let radiance = env_radiance(d);
        (*acc).emission += thr * radiance;
        (*acc).alpha = 1.0;
        (*acc).albedo = radiance;
      }
      break;
    }
    let wo = -d;
//#if SPLATS
    var sh: Shape;
    var sf: Surf;
    var emit3 = vec3<f32>(0.0);
    var oid = 0.0;
    if (splat_id >= 0) {
      sh = splat_shape(u32(splat_id));
      let ss = splat_surface(u32(splat_id), wo, d);
      sf.pos = o + d * h.t;
      sf.ns = ss.ns;
      sf.ng = ss.ns;
      sf.uv = vec2<f32>(0.0);
      sf.base = ss.base;
      sf.alpha = 1.0;
      sf.metallic_ovr = -1.0;
      sf.roughness_ovr = -1.0;
      sf.occlusion = 1.0;
      sf.emissive = vec3<f32>(0.0);
      emit3 = ss.emit;
      oid = ss.oid;
    } else {
      sh = shapes[h.shape];
      sf = surface(h, o, d);
      emit3 = sf.base * sh.mat.w + sf.emissive;
      oid = f32(h.shape + 1);
    }
//#else
    let sh = shapes[h.shape];
    let sf = surface(h, o, d);
    let emit3 = sf.base * sh.mat.w + sf.emissive;
    let oid = f32(h.shape + 1);
//#endif
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
    if (!cover) {
      o = pos + d * eps;
//#if SPLATS
      skip_id = -1; skip_t = 0.0; skip_th = -1e30;
//#endif
      continue;
    }
    if (vdepth == 0u) {
      (*acc).alpha = sf.alpha;
      (*acc).albedo = sf.base * sf.alpha;
      (*first).shape = h.shape; (*first).t = h.t; (*first).ns = ns; (*first).pos = pos; (*first).uv = sf.uv; (*first).oid = oid;
    }
    if (params.d.z >= 7u) { return; }
    if (max(emit3.x, max(emit3.y, emit3.z)) > 0.0) {
      let g = thr * emit3;
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
//#if LINKS
      prev_excl = 0u;
//#endif
      bounces++;
      vdepth++;
//#if SPLATS
      skip_id = -1; skip_t = 0.0; skip_th = -1e30;
//#endif
      continue;
    }
    // ---- a surface with a BSDF ----
    let nv = max(dot(ns, wo), 1e-4);
    let lobe = make_lobe(sh, sf.base, nv, sf.metallic_ovr, sf.roughness_ovr, sf.occlusion);
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
//#if LINKS
        irr = irr * select(1.0, 0.0, excl_bit(recv_mask(h.shape, splat_id), li));
//#endif
        let nl = dot(ns, wi);
        if (nl > 0.0 && max(irr.x, max(irr.y, irr.z)) > 0.0) {
//#if LINKS
          shadow_bit = li;
//#endif
          if (intersect(pos + ns * seps, wi, 0.0, dist - seps, true).shape < 0) {
            let e = bsdf_eval(lobe, ns, wo, wi);
            var seen = 1.0;
//#if SOFT
            seen = soft_visibility(pos + ns * seps, wi, dist, SPLAT_ID, ns);
//#endif
            direct_d += e.fd * irr * seen;
            direct_s += e.fs * irr * seen;
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
//#if LINKS
        if (excl_bit(recv_mask(h.shape, splat_id), li)) { cos_l = 0.0; }
//#endif
        let nl = dot(ns, wi);
        if (nl > 0.0 && cos_l > 1e-9 && dist > 1e-9) {
//#if LINKS
          shadow_bit = li;
//#endif
          if (intersect(pos + ns * seps, wi, 0.0, dist * (1.0 - 1e-4) - seps, true).shape < 0) {
            let pdf_l = dist * dist / max(cos_l * light_field(li, 3u).w, 1e-30);
            let e = bsdf_eval(lobe, ns, wo, wi);
//#if LINKS
            var wgt = lmis(li, pdf_l, e.pdf) / pdf_l;
//#else
            var wgt = mis(pdf_l, e.pdf) / pdf_l;
//#endif
//#if SOFT
            wgt = wgt * soft_visibility(pos + ns * seps, wi, dist * (1.0 - 1e-4), SPLAT_ID, ns);
//#endif
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
//#if LINKS
      if (nl > 0.0 && es.pdf > 0.0 && !excl_bit(recv_mask(h.shape, splat_id), lc)) {
        shadow_bit = lc;
//#else
      if (nl > 0.0 && es.pdf > 0.0) {
//#endif
        if (intersect(pos + ns * seps, es.dir, 0.0, INF, true).shape < 0) {
          let e = bsdf_eval(lobe, ns, wo, es.dir);
//#if LINKS
          var wgt = lmis(lc, es.pdf, e.pdf) / es.pdf;
//#else
          var wgt = mis(es.pdf, e.pdf) / es.pdf;
//#endif
//#if SOFT
          wgt = wgt * soft_visibility(pos + ns * seps, es.dir, INF, SPLAT_ID, ns);
//#endif
          direct_d += e.fd * es.radiance * wgt;
          direct_s += e.fs * es.radiance * wgt;
        }
      }
    }
//#if SPLATS
    var relight_mix = 1.0;
    if (splat_id >= 0) { relight_mix = sp(u32(splat_id), 6u).w; }
    direct_d = direct_d * relight_mix;
    direct_s = direct_s * relight_mix;
//#else
    let relight_mix = 1.0;
//#endif
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
    let new_thr = thr * weight * relight_mix;
    if (max(new_thr.x, max(new_thr.y, new_thr.z)) <= 0.0) { break; }
    thr = new_thr;
    var side = ng;
    if (dot(wi, ng) < 0.0) { side = -ng; }
    o = pos + side * eps;
    d = wi;
//#if SPLATS
    if (splat_id >= 0) {
      let m7 = sp(u32(splat_id), 7u);
      skip_id = splat_id; skip_t = SPLAT_START_SCALE * m7.y; skip_th = m7.x;
    } else {
      skip_id = -1; skip_t = 0.0; skip_th = -1e30;
    }
    skip_n = ns;
//#endif
    if (choose_spec) { cs++; capped = cs >= params.c.y; } else { cd++; capped = cd >= params.c.x; }
    if (cls < 0) { cls = select(0, 1, choose_spec); }
    prev_delta = is_delta;
    prev_pdf = e.pdf;
//#if LINKS
    prev_excl = recv_mask(h.shape, splat_id);
//#endif
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

@compute @workgroup_size(WG_SIZE, WG_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let width = params.a.x;
  let adaptive = params.ad.z > 1.5;
  let sparse = params.ad.z > 2.5;
  var x = gid.x;
  var y = gid.y + params.a.z;
  var lane = 0u;
  var thread = 0u;
  if (adaptive) {
    // an adaptive render dispatches one thread per pixel in `open_pixels`, through an indirect dispatch the device sized
    // itself: a pixel that is done costs nothing, no warp holds one, and nothing came back to the host to say so.
    // When few pixels are open (the sparse dispatch, ad.z = 3) one thread takes one sample of one pixel, so the samples of
    // a pass run side by side instead of one dispatch after another, and `reduce` adds them up afterwards.
    thread = params.b.x + wid.x * (WG_SIZE * WG_SIZE) + lid.y * WG_SIZE + lid.x;
    var slot = thread;
    if (sparse) {
      let lanes = open_pixels[params.a.w + 1u];
      slot = thread / lanes;
      lane = thread % lanes;
    }
    if (slot >= open_pixels[params.a.w]) { return; }
    let listed = open_pixels[params.a.z + slot];
    x = listed % width;
    y = listed / width;
  } else {
    if (x >= width || y >= params.a.w) { return; }
    let tile = (y / 16u) * params.d.w + (x / 16u);
    let word = params.tiles[tile >> 7u][(tile >> 5u) & 3u];
    if (((word >> (tile & 31u)) & 1u) == 0u) { return; }
  }
  let pixel = y * width + x;
  let code = params.d.z;
  // adaptive sampling: a pixel whose noise estimate went under the threshold set its done flag (stats.w) and takes no more
  // samples; its next sample is its own count so far, which is the sample index the CPU reference gives it too
  var sample_base = params.b.x;
  if (adaptive && code < 7u) {
    let before = accum[pixel * 2u + 1u];
    if (before.w > 0.5) { return; }
    sample_base = u32(before.z + 0.5) + lane;
  }
  var rgb = vec3<f32>(0.0);
  var alpha = 0.0;
  var lum_sum = 0.0;
  var lum_sq = 0.0;
  let count = select(params.b.y, 1u, code >= 7u);
  // `var x: T;` inside the loop is not re-zeroed by every driver (more than one sample per dispatch summed each
  // sample's radiance cumulatively, step R1): copy these zeroed ones in at the top of every sample instead.
  var no_acc: Acc;
  var no_first: First;
  for (var s = 0u; s < count; s++) {
    let key = path_key(pixel, sample_base + s, params.b.z);
    var jx = 0.5;
    var jy = 0.5;
    if (code < 7u) { jx = rnd(key, 0u); jy = rnd(key, 1u); }
    let px = f32(x) + jx;
    let py = f32(y) + jy;
    let aspect = params.right.w;
    let lx = (2.0 * px / f32(width) - 1.0) * aspect / (1.0 / params.up.w);
    let ly = (1.0 - 2.0 * py / f32(params.a.y)) / (1.0 / params.up.w);
    var dv = params.right.xyz * lx + params.up.xyz * ly + params.forward.xyz;
    var origin = params.eye.xyz;
    if (code < 7u && params.lens.x > 0.0) {
      // thin lens: start at a point of the aperture, aim at the pixel's point on the focal plane
      let ap = aperture_point(rnd(key, 2u), rnd(key, 3u)) * params.lens.x;
      let shift = params.right.xyz * ap.x + params.up.xyz * ap.y;
      origin = origin + shift;
      dv = dv - shift / params.lens.y;
    }
    let c = length(dv);
    var acc = no_acc;
    var first = no_first;
    first.shape = -1;
//#if COUNT
    vol_counts = vec3<f32>(0.0);
//#endif
    trace(key, origin, dv / c, c, &acc, &first);
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
          default: { value = vec3<f32>(first.oid, 0.0, 0.0); }
        }
      }
      accum[pixel * 2u] = vec4<f32>(value, cov);
      return;
    }
    let total = total_of(acc);
//#if COUNT
    rgb += vol_counts;
//#else
    rgb += channel_of(acc, code);
//#endif
    alpha += acc.alpha;
    let lum = dot(total, vec3<f32>(0.2126, 0.7152, 0.0722));
    lum_sum += lum;
    lum_sq += lum * lum;
  }
  if (sparse) {
    let scratch = params.a.x * params.a.y * 2u + thread * 2u;
    accum[scratch] = vec4<f32>(rgb, alpha);
    accum[scratch + 1u] = vec4<f32>(lum_sum, lum_sq, 0.0, 0.0);
    return;
  }
  let previous = accum[pixel * 2u];
  accum[pixel * 2u] = previous + vec4<f32>(rgb, alpha);
  let stats = accum[pixel * 2u + 1u];
  let n = stats.z + f32(count);
  let sum = stats.x + lum_sum;
  let sq = stats.y + lum_sq;
  var done = stats.w;
  if (adaptive && params.ad.x > 0.0 && n >= params.ad.y) {
    // a pixel decides at the end of a pass: when it has its minimum plus a whole number of passes, or its last sample
    let taken = u32(n + 0.5);
    let minimum = u32(params.ad.y + 0.5);
    let pass_size = max(u32(params.ad.w + 0.5), 1u);
    if ((taken - minimum) % pass_size == 0u || taken >= u32(params.lens2.y + 0.5)) {
      // the same estimate as pathtrace.pixel_noise: variance of the mean luminance over (mean + 0.02)^2
      let mean = sum / n;
      let var_of_mean = max(sq / n - mean * mean, 0.0) * n / max(n - 1.0, 1.0) / n;
      let m = mean + 0.02;
      if (var_of_mean / (m * m) < params.ad.x) { done = 1.0; }
    }
  }
  accum[pixel * 2u + 1u] = vec4<f32>(sum, sq, n, done);
}
'''


# Adaptive sampling keeps its convergence mask on the device. After each pass `scan` looks at the done flags the render
# shader wrote and lists every pixel that is still open; `finish` turns the list length into the arguments of the indirect
# dispatches the next pass runs (one thread per listed pixel), so converged pixels cost nothing and the host never reads the
# mask. When only a few pixels are open `finish` instead sizes the sparse dispatch, in which every thread takes one sample
# of one pixel (the samples of a pass run side by side, a card with few pixels left being otherwise idle), and `reduce` adds
# the pass's samples up. `ctl` holds, in words: the indirect arguments (x, 1, 1, 0) of each chunk of the dense dispatch, of
# the sparse dispatch and of `reduce` live in their own buffer (a buffer cannot be both written by a shader and the source of
# an indirect dispatch that shader's pass runs); `ctl` holds a header (running length, 0, the next pass's list length, its
# samples per pixel), then the list of pixel indices.
_COMPACT_SHADER = r'''
struct CParams {
  a: vec4<u32>,      // width, height, 0, 0
  b: vec4<u32>,      // chunks, groups per chunk, list base, threads per group
  c: vec4<u32>,      // sparse pixel limit, sparse thread limit, min samples, max samples
  d: vec4<f32>,      // noise threshold, pass size, 0, 0
};
@group(0) @binding(0) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> ctl: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> cp: CParams;
@group(0) @binding(3) var<storage, read_write> args: array<atomic<u32>>;    // the indirect dispatch arguments, written by `finish` only
var<workgroup> found: atomic<u32>;
var<workgroup> reserved: u32;

// One 8 by 8 block of pixels per work-group: its open pixels are numbered inside the group, the group reserves that many
// places in the list with one atomic add, and each writes its pixel there. Neighbouring pixels stay neighbours in the list.
@compute @workgroup_size(8, 8, 1)
fn scan(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) index: u32) {
  let header = 0u;
  if (index == 0u) { atomicStore(&found, 0u); }
  workgroupBarrier();
  let inside = gid.x < cp.a.x && gid.y < cp.a.y;
  let pixel = gid.y * cp.a.x + gid.x;
  var mine = 0u;
  var open = false;
  if (inside) { open = accum[pixel * 2u + 1u].w < 0.5; }
  if (open) { mine = atomicAdd(&found, 1u); }
  workgroupBarrier();
  let total = atomicLoad(&found);
  if (index == 0u && total > 0u) { reserved = atomicAdd(&ctl[header], total); }
  workgroupBarrier();
  if (open) { atomicStore(&ctl[cp.b.z + reserved + mine], pixel); }
}

// The next pass: every open pixel has the same count n (a pixel only ever stops at a pass boundary), so its length is
// the minimum for the first pass and `pass size` after, up to the maximum.
@compute @workgroup_size(1)
fn finish() {
  let header = 0u;
  let count = atomicLoad(&ctl[header]);
  var lanes = 0u;
  var sparse = false;
  if (count > 0u) {
    let first = atomicLoad(&ctl[cp.b.z]);
    let taken = u32(accum[first * 2u + 1u].z + 0.5);
    let pass_size = max(u32(cp.d.y + 0.5), 1u);
    var end = cp.c.z;
    if (taken >= cp.c.z) { end = cp.c.z + ((taken - cp.c.z) / pass_size + 1u) * pass_size; }
    end = min(end, cp.c.w);
    if (end > taken) { lanes = end - taken; }
    sparse = lanes > 1u && count <= cp.c.x && count * lanes <= cp.c.y && count * lanes <= cp.b.y * cp.b.w;
  }
  let dense = select(count, 0u, sparse);
  let groups = (dense + cp.b.w - 1u) / cp.b.w;
  for (var i = 0u; i < cp.b.x; i++) {
    let before = min(groups, i * cp.b.y);
    atomicStore(&args[i * 4u], min(groups - before, cp.b.y));
    atomicStore(&args[i * 4u + 1u], 1u);
    atomicStore(&args[i * 4u + 2u], 1u);
  }
  let sparse_at = cp.b.x * 4u;
  atomicStore(&args[sparse_at], select(0u, (count * lanes + cp.b.w - 1u) / cp.b.w, sparse));
  atomicStore(&args[sparse_at + 1u], 1u);
  atomicStore(&args[sparse_at + 2u], 1u);
  atomicStore(&args[sparse_at + 4u], select(0u, (count + 63u) / 64u, sparse));
  atomicStore(&args[sparse_at + 5u], 1u);
  atomicStore(&args[sparse_at + 6u], 1u);
  atomicStore(&ctl[header + 2u], count);      // the list's length for the render shader and for the host's progress counter
  atomicStore(&ctl[header + 3u], lanes);
  atomicStore(&ctl[header], 0u);
  atomicStore(&ctl[header + 1u], 0u);
}

// After a sparse pass: add each open pixel's samples (written by the render shader at the end of the accumulator) to its
// totals and let it decide, the same estimate as `pathtrace.pixel_noise` and the render shader's own.
@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
  let header = 0u;
  let count = atomicLoad(&ctl[header + 2u]);
  let lanes = atomicLoad(&ctl[header + 3u]);
  let slot = gid.x;
  if (slot >= count) { return; }
  let pixel = atomicLoad(&ctl[cp.b.z + slot]);
  let scratch = cp.a.x * cp.a.y * 2u + slot * lanes * 2u;
  var rgb = vec4<f32>(0.0);
  var lum_sum = 0.0;
  var lum_sq = 0.0;
  for (var j = 0u; j < lanes; j++) {
    rgb += accum[scratch + j * 2u];
    let moments = accum[scratch + j * 2u + 1u];
    lum_sum += moments.x;
    lum_sq += moments.y;
  }
  accum[pixel * 2u] = accum[pixel * 2u] + rgb;
  let stats = accum[pixel * 2u + 1u];
  let n = stats.z + f32(lanes);
  let sum = stats.x + lum_sum;
  let sq = stats.y + lum_sq;
  var done = stats.w;
  if (cp.d.x > 0.0 && n >= f32(cp.c.z)) {
    let taken = u32(n + 0.5);
    let pass_size = max(u32(cp.d.y + 0.5), 1u);
    if ((taken - cp.c.z) % pass_size == 0u || taken >= cp.c.w) {
      let mean = sum / n;
      let var_of_mean = max(sq / n - mean * mean, 0.0) * n / max(n - 1.0, 1.0) / n;
      let m = mean + 0.02;
      if (var_of_mean / (m * m) < cp.d.x) { done = 1.0; }
    }
  }
  accum[pixel * 2u + 1u] = vec4<f32>(sum, sq, n, done);
}
'''


def _compact_pipelines(state):
    cache = state.setdefault("_gpupt_compact", {})
    if "pipelines" not in cache:
        device = state["device"]
        module = device.create_shader_module(code=_COMPACT_SHADER)
        cache["pipelines"] = tuple(device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": name})
                                   for name in ("scan", "finish", "reduce"))
    return cache["pipelines"]


def _preprocess(code, flags):
    """Keep the lines between `//#if NAME` and `//#endif` (with an optional `//#else`) when `flags[NAME]` holds."""
    out, stack = [], []          # stack of (enabled_by_this_level, parent_enabled)
    live = True
    for line in code.split("\n"):
        word = line.strip()
        if word.startswith("//#if "):
            stack.append((live, flags[word[6:].strip()]))
            live = live and flags[word[6:].strip()]
        elif word == "//#else":
            parent, cond = stack[-1]
            live = parent and not cond
        elif word == "//#endif":
            live = stack.pop()[0]
        elif live:
            out.append(line)
    if stack:
        raise ValueError("unbalanced //#if in the path tracer shader")
    return "\n".join(out)


def shader_source(splats=False, volumes=False, links=False, sparse=False, count=False):
    """The WGSL for a scene kind: the splat and smoke code costs registers whether it runs or not, so a mesh-only scene
    is compiled without any of it. The splats-and-volumes variant also compiles with a smaller workgroup
    (`_wg_size`): the same code at 8x8 miscompiles on AMD."""
    code = _preprocess(_SHADER, {"SPLATS": bool(splats), "VOLUMES": bool(volumes), "SOFT": bool(splats or volumes),
                                 "LINKS": bool(links), "SPARSE": bool(sparse and volumes),
                                 "COUNT": bool(count and volumes)})
    code = code.replace("SPLAT_ID", "splat_id" if splats else "-1")
    return code.replace("WG_SIZE", str(_wg_size(splats, volumes, links)))


def _pipeline(state, splats=False, volumes=False, links=False, sparse=False):
    cache = state.setdefault("_gpupt_pipelines", {})
    key = (bool(splats), bool(volumes), bool(links), bool(sparse and volumes), bool(COUNT_COLLISIONS and volumes))
    if key not in cache:
        device = state["device"]
        COUNTERS["pipelines"] += 1
        cache[key] = device.create_compute_pipeline(layout="auto", compute={
            "module": device.create_shader_module(code=shader_source(*key)), "entry_point": "main"})
    return cache[key]


# --- packing --------------------------------------------------------------------------------------------

def _f4(*arrays):
    return np.ascontiguousarray(np.concatenate([np.asarray(a, "f4") for a in arrays], axis=-1))


class Packed:
    """The buffers of one scene and the numbers the uniform needs."""


def pack(ps, environment_size=None, cancel=None):
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
            # Flat per-triangle tangent (normal-mapping, materials 3 step X3), riding the spare .w of
            # v0/e1/e2 (untouched otherwise): read back in the shader as `tri.v0.w` etc.
            if blas.tangent is not None:
                rows[:, 3], rows[:, 7], rows[:, 11] = blas.tangent[:, 0], blas.tangent[:, 1], blas.tangent[:, 2]
            tri_parts.append(rows)
            node_base += len(nodes)
            order_base += len(order)
            tri_base += count
    if len(root_of_blas) < len(ps.blases):
        root_of_blas = [0] * len(ps.blases)      # nothing to enter: every ray misses (`params.e.x` is 0)
    packed.splat_root = 0
    if ps.splats is not None:
        # the splats' own tree: leaves hold splat ids, so its order entries take no triangle offset
        node_base, order_base = sum(len(a) for a in nodes_parts), sum(len(a) for a in order_parts)
        nodes, order = gpu3d._pack_bvh(ps.splats.bvh, cancel)
        nodes = nodes.copy()
        interior = nodes["count"] == 0
        nodes["left"] = np.where(interior, nodes["left"] + node_base, nodes["left"])
        nodes["right"] = np.where(interior, nodes["right"] + node_base, nodes["right"])
        nodes["offset"] = nodes["offset"] + order_base
        packed.splat_root = node_base
        nodes_parts.append(nodes)
        order_parts.append(order.astype("u4"))
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
    light_bits = []        # the bit each row has in `ps.excl` (points, areas, environments), before the sort below
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
        light_bits.append(len(lights) - 1)
    for light in ps.area_lights:
        row = np.zeros(LIGHT_VECS * 4, "f4")
        row[0:4] = (_KIND_CODES[light.kind], float(light.two_sided), float(light.visible_to_camera), 0)
        row[4:7], row[7] = light.position, (light.half_w if light.kind == "Rect" else light.radius)
        row[8:11], row[11] = light.normal, light.half_h
        row[12:15], row[15] = light.right, light.area
        row[16:19] = light.up
        row[20:23] = light.radiance
        lights.append(row)
        light_bits.append(len(lights) - 1)
    ranked = sorted(range(len(lights)), key=lambda k: lights[k][0] > 2.5)   # stable: point-like lights first
    packed_of = {light_bits[k]: new for new, k in enumerate(ranked)}        # `ps.excl` bit -> packed light index
    packed_of[len(lights)] = len(lights)                                    # the environment follows the lights
    packed.bit_map = packed_of
    lights = [lights[k] for k in ranked]      # the order does not matter to the maths
    packed.linked = 0       # the lights some shape, splat instance or volume excludes, in packed numbering (`params.e.w`)
    if ps.linked_bits:
        if len(lights) + (1 if ps.envs else 0) > 24:
            raise gpu3d.Unsupported("light linking on the GPU takes at most 24 lights and environments together")
    if ps.excl is not None:
        for i in range(n_shapes):
            mask = int(ps.excl[i])
            packed.shapes[i, 27] = float(sum(1 << packed_of[b] for b in range(len(lights) + 1) if (mask >> b) & 1))
            packed.linked |= int(packed.shapes[i, 27])
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
    _pack_aux(packed, ps)
    _pack_textures(packed, ps)
    return packed


def _u4_as_f4(*values):
    return np.array(values, "u4").view("f4")


def _sparse_volume_grid(volume):
    """The tile grid a sparse volume's density and temperature are read from; Unsupported when the shader cannot index it."""
    grid = volume.sparse
    if grid.tile != 8:
        raise gpu3d.Unsupported(f"the GPU path tracer reads 8-voxel tiles, not {grid.tile}")
    temperature = volume.temperature
    if temperature is not None and getattr(temperature, "grid", None) is not grid:
        raise gpu3d.Unsupported("a sparse volume whose temperature is not stored in the same tiles stays on the CPU")
    if temperature is not None and grid.rest["temperature"] != 0.0:
        raise gpu3d.Unsupported("the GPU path tracer needs a zero rest temperature in a sparse volume")
    return grid


def _pack_aux(packed, ps):
    """Splat records, their spherical harmonics, volume headers, grids and the fire table go after the environment in
    the same buffer (the shader reads them as vec4s at the offsets the uniform carries)."""
    pieces = []
    size = [len(packed.env)]

    def add(rows):
        rows = np.ascontiguousarray(rows, "f4").reshape(-1, 4)
        start = size[0]
        pieces.append(rows)
        size[0] += len(rows)
        return start

    def add_floats(flat):
        flat = np.ascontiguousarray(flat, "f4").reshape(-1)
        if len(flat) % 4:           # only a ragged end is copied to pad it; the rest is copied once, into the buffer
            padded = np.zeros(-(-len(flat) // 4) * 4, "f4")
            padded[:len(flat)] = flat
            flat = padded
        return add(flat) * 4

    packed.splat_count, packed.splat_base = 0, 0
    packed.volume_count, packed.volume_base, packed.fire_base, packed.flags = 0, 0, 0, 0
    packed.sparse_volumes = False
    packed.smoke = np.zeros((3, 4), "f4")
    layer = ps.splats
    if layer is not None:
        m = len(layer)
        has_sh = np.zeros(m, bool)
        local = np.zeros(m, np.int64)
        degree = np.zeros(m, "u4")
        linear = np.zeros(m, "u4")
        chunks, offset = [], 0
        for first, count, cloud, deg, kept in layer.groups:
            if not count or not (layer.relight[first:first + count] < 1.0).any():
                continue
            k = (deg + 1) ** 2
            rows = np.asarray(cloud.sh, "f4")[kept, :k, :].reshape(count, k * 3)
            has_sh[first:first + count] = True
            local[first:first + count] = offset + np.arange(count) * (k * 3)
            degree[first:first + count] = deg
            linear[first:first + count] = 1 if cloud.colorspace == "linear" else 0
            chunks.append(rows.reshape(-1))
            offset += count * k * 3
        sh_start = add_floats(np.concatenate(chunks)) if chunks else 0
        packed.flags |= 1 if chunks else 0
        records = np.zeros((m, SPLAT_VECS, 4), "f4")
        records[:, 0, :3], records[:, 0, 3] = layer.hit.positions, layer.hit.opacity
        records[:, 1, :3], records[:, 1, 3] = layer.hit.scales, layer.shadow.opacity
        for j in range(3):
            records[:, 2 + j, :3] = layer.hit.rotations_matrix[:, :, j]
            records[:, 2 + j, 3] = layer.albedo[:, j]
        records[:, 5, :3], records[:, 5, 3] = layer.normal, layer.confidence
        records[:, 6] = np.stack((layer.roughness, layer.metallic, layer.pbr.astype("f4"), layer.relight), 1)
        records[:, 7, 0], records[:, 7, 1] = layer.scale_min, layer.scale_max
        if layer.excl is not None:
            # light linking: the instance's excluded lights, in the packed light numbering (a float, exact below 2^24)
            remapped = np.zeros(m, np.int64)
            for bit, target in packed.bit_map.items():
                remapped |= ((layer.excl >> bit) & 1) << target
            records[:, 7, 2] = remapped
            packed.linked |= int(np.bitwise_or.reduce(remapped))
        # Integers travel as float values, never as bit patterns (see splat_info in the shader).
        sh_offset = np.where(has_sh, sh_start + local, 0).astype(np.int64)
        records[:, 8, 0] = layer.instance
        records[:, 8, 1] = sh_offset & 0xFFFFFF
        records[:, 8, 2] = degree + 16 * linear
        records[:, 8, 3] = sh_offset >> 24
        packed.splat_count, packed.splat_base = m, add(records.reshape(-1, 4))
    smoke = ps.volumes
    if smoke is not None:
        settings = smoke.settings
        headers = np.zeros((len(smoke), VOL_VECS, 4), "f4")
        for vi, prep in enumerate(smoke.preps):
            volume = prep.volume
            headers[vi, 0, :3], headers[vi, 0, 3] = prep.box_min, volume.voxel_size
            headers[vi, 1, :3], headers[vi, 1, 3] = prep.box_max, smoke.majorant[vi]
            for r in range(3):
                headers[vi, 2 + r, :3], headers[vi, 2 + r, 3] = prep.inv[r, :3], prep.inv[r, 3]
            headers[vi, 5, :3] = volume.density.shape[:3]
            sparse = gpuvolume.is_sparse(volume)
            if sparse:
                grid = _sparse_volume_grid(volume)
                packed.sparse_volumes = True
                # A float value holds a tile slot exactly; the table is indexed (x * ny + y) * nz + z over tiles.
                table = np.zeros(tuple(-(-n // grid.tile) for n in grid.shape), "f4")
                if len(grid.coords):
                    table[grid.coords[:, 0], grid.coords[:, 1], grid.coords[:, 2]] = np.arange(1, len(grid.coords) + 1)
                tiles = add_floats(table)
                headers[vi, 8] = (tiles & 0xFFFFFF, tiles >> 24, 1.0, grid.rest["density"])
                density = add_floats(np.ascontiguousarray(grid.data["density"], "f4"))
            else:
                density = add_floats(np.ascontiguousarray(volume.density, "f4"))
            levels = smoke.levels(vi) if ENABLE_VOLUME_SKIP else None
            if levels is not None:
                fine_base, coarse_base = add_floats(levels.fine), add_floats(levels.coarse)
                headers[vi, 7] = (fine_base & 0xFFFFFF, fine_base >> 24, levels.tile, 0)
                headers[vi, 9] = (coarse_base & 0xFFFFFF, coarse_base >> 24, levels.ratio, 0)
            else:
                headers[vi, 7, 0] = -1.0
            if smoke.excl is not None:
                # light linking: the volume's excluded lights, in the packed light numbering (a float, exact below 2^24)
                mask = sum(((int(smoke.excl[vi]) >> bit) & 1) << target for bit, target in packed.bit_map.items())
                headers[vi, 7, 3] = mask
                packed.linked |= mask
            glow = smoke.fire_table is not None and volume.temperature is not None
            temperature = add_floats(np.ascontiguousarray(grid.data["temperature"] if sparse else volume.temperature,
                                                          "f4")) if glow else None
            headers[vi, 6] = (density & 0xFFFFFF, -1.0 if temperature is None else temperature & 0xFFFFFF,
                              density >> 24, 0 if temperature is None else temperature >> 24)
        packed.volume_count, packed.volume_base = len(smoke), add(headers)
        if smoke.fire_table is not None:
            table = np.zeros((len(smoke.fire_table), 4), "f4")
            table[:, :3] = smoke.fire_table
            packed.fire_base = add(table)
            packed.flags |= 2
        packed.smoke[0] = (smoke.sigma_t_unit, smoke.scatter_fraction, settings.anisotropy, settings.density_scale)
        packed.smoke[1, :3], packed.smoke[1, 3] = smoke.color, settings.shadow_density
        packed.smoke[2] = (settings.temperature_scale, settings.fire_threshold, settings.fire_intensity,
                           settings.shadow_steps)
    if pieces:
        packed.env = np.concatenate([packed.env, *pieces])


_NO_TEX = np.uint32(0xFFFFFFFF)


def _pack_textures(packed, ps):
    """PBR texture maps (materials 3, step X3): each present map's texels (top mip only, like the CPU
    reference) go after everything else in `env`, one texel per row (RGBA); a shape's `Shape.tex0/tex1/tex2`
    then carry an offset (vec4 index into `env`) and `width<<16 | height` dims per map, `_NO_TEX` where the
    shape has none. Runs after `_pack_aux` so the offsets land past the splat/volume data already there,
    and patches `packed.shapes` in place (built earlier in `pack`, before any offset was known)."""
    pieces = []
    base = [len(packed.env)]

    def add(tex):
        if tex is None:
            return _NO_TEX, np.uint32(0)
        arr = np.ascontiguousarray(tex, "f4")
        h, w = arr.shape[0], arr.shape[1]
        off = np.uint32(base[0])
        pieces.append(arr.reshape(-1, 4))
        base[0] += h * w
        return off, np.uint32((int(w) << 16) | int(h))

    n = ps.shapes
    off = {name: np.full(n, _NO_TEX, np.uint32) for name in
           ("base", "mr", "normal", "occlusion", "emissive")}
    dims = {name: np.zeros(n, np.uint32) for name in off}
    sources = {"base": ps.texture, "mr": ps.mr_texture, "normal": ps.normal_texture,
               "occlusion": ps.occlusion_texture, "emissive": ps.emissive_texture}
    for name, textures in sources.items():
        for i in range(n):
            off[name][i], dims[name][i] = add(textures[i])
    if pieces:
        packed.env = np.concatenate([packed.env, *pieces])
    shapes = packed.shapes

    def put(col, values):
        shapes[:n, col] = np.asarray(values, np.uint32).view("f4")

    put(32, off["base"]); put(33, off["mr"]); put(34, off["normal"]); put(35, off["occlusion"])
    put(36, off["emissive"]); put(37, dims["base"]); put(38, dims["mr"]); put(39, dims["normal"])
    put(40, dims["occlusion"]); put(41, dims["emissive"])
    if n:
        # normal_scale/occlusion_strength/emissive_color also carry entries for splat instances
        # (materials 3 step X1): only the first `n` (mesh shapes) apply here.
        shapes[:n, 44] = np.asarray(ps.normal_scale[:n], "f4")
        shapes[:n, 45] = np.asarray(ps.occlusion_strength[:n], "f4")
        shapes[:n, 46:49] = np.asarray(ps.emissive_color[:n], "f4")


def _uniform(packed, ps, camera, width, height, row0, row1, sample_base, spp, settings, code, tile_bits, tiles_x,
             mode=2.0):
    eye, view = s._view_basis(camera)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    a = np.array([width, height, row0, row1], "u4")
    b = np.array([sample_base, spp, settings.seed, settings.max_bounces], "u4")
    c = np.array([settings.diffuse_bounces, settings.specular_bounces, settings.transmission_bounces,
                  packed.light_count], "u4")
    d = np.array([packed.env_size[0], packed.env_size[1], code, tiles_x], "u4")
    env_visible = 1 if ps.envs and ps.envs[0].visible_to_camera else 0
    e = np.array([0 if packed.empty else ps.shapes, packed.env_cdf_base, env_visible, packed.linked], "u4")
    g = np.array([ps.shapes, packed.splat_count, packed.splat_base, packed.splat_root], "u4")
    h = np.array([packed.volume_count, packed.volume_base, packed.fire_base, packed.flags], "u4")
    f32 = np.zeros((16, 4), "f4")
    right, up, forward = view[0], view[1], -view[2]
    f32[0, :3], f32[0, 3] = right, width / max(height, 1)
    f32[1, :3], f32[1, 3] = up, 1.0 / focal
    f32[2, :3], f32[2, 3] = forward, camera.near
    f32[3, :3], f32[3, 3] = eye, camera.far
    f32[4] = (ps.eps, ps.shadow_eps, ps.ambient, packed.env_scale)
    f32[5, :3] = packed.env_gain
    f32[6:9] = packed.smoke
    rotation = packed.env_rotation
    f32[9, :3], f32[10, :3], f32[11, :3] = rotation[0], rotation[1], rotation[2]
    if lens.active(camera):
        radius, focus, blades, blade_rotation, inverse_squeeze = lens.lens_uniform(camera)
        f32[12] = (radius, focus, blades, blade_rotation)
        f32[13, 0] = inverse_squeeze
    if settings.adaptive and code < 7:
        f32[14] = (settings.noise_threshold, settings.min_samples, mode, settings.adaptive_pass_size)
        f32[13, 1] = settings.max_samples
    if ps.volumes is not None:
        f32[15, :2] = (ps.volumes.settings.multi_scatter, ps.volumes.settings.fire_light)
    tiles = np.zeros((512, 4), "u4")
    tiles.reshape(-1)[:len(tile_bits)] = tile_bits
    return b"".join((a.tobytes(), b.tobytes(), c.tobytes(), d.tobytes(), e.tobytes(), g.tobytes(), h.tobytes(),
                     f32.tobytes(), tiles.tobytes()))


def _tiles_done(pixel_done, width, height, tiles_x, tiles_y):
    """Per tile: True when every pixel of it is done (the tiles at the right and bottom edges count only their pixels)."""
    padded = np.ones((tiles_y * pt.TILE, tiles_x * pt.TILE), bool)
    padded[:height, :width] = pixel_done.reshape(height, width)
    return padded.reshape(tiles_y, pt.TILE, tiles_x, pt.TILE).all(axis=(1, 3)).reshape(-1)


def _tile_bits(tile_done):
    active = ~tile_done
    words = np.zeros((len(active) + 31) // 32, "u4")
    idx = np.flatnonzero(active)
    np.bitwise_or.at(words, idx >> 5, (np.uint32(1) << (idx & 31).astype("u4")).astype("u4"))
    return words


# --- what stays on the device between frames ---------------------------------------------------------------

_RESOLVE_SHADER = """
struct Params {
    npix: u32, width: u32, tiles_x: u32, tile: u32,
    data_pass: u32, stride: u32, adaptive: u32, pad1: u32,
    bg: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> acc: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;

fn put(i: u32, c: vec4<f32>) {
    out[4u * i] = bitcast<u32>(c.x);
    out[4u * i + 1u] = bitcast<u32>(c.y);
    out[4u * i + 2u] = bitcast<u32>(c.z);
    out[4u * i + 3u] = bitcast<u32>(c.w);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * p.stride;
    if (i >= p.npix) { return; }
    let a = acc[2u * i];
    let b = acc[2u * i + 1u];
    if (p.adaptive != 0u) {
        // the pixel's own sample count and done flag, after the image: the count in the low bits, the flag in the top one
        out[4u * p.npix + i] = u32(b.z) | select(0u, 0x80000000u, b.w > 0.5);
    }
    if (p.data_pass != 0u) {
        put(i, select(vec4<f32>(0.0), a, a.w > 0.0));
        return;
    }
    var n = b.z;
    if (p.adaptive == 0u) {
        let x = i % p.width;
        let y = i / p.width;
        n = f32(counts[(y / p.tile) * p.tiles_x + x / p.tile]);
    }
    let c = a / max(n, 1.0);
    put(i, c + p.bg * (1.0 - c.w));
}
"""


def _same_bytes(x, y):
    """True when two arrays hold the same bytes (a NaN that pads a record equals itself here)."""
    return (x.shape == y.shape and x.dtype == y.dtype
            and np.array_equal(np.ascontiguousarray(x).reshape(-1).view(np.uint8), np.ascontiguousarray(y).reshape(-1).view(np.uint8)))


def _same_arrays(a, b):
    return b is not None and len(a) == len(b) and all(_same_bytes(x, y) for x, y in zip(a, b))


class _Frame:
    """The device objects a fixed-sampling render of one scene keeps for the next frame of it: the scene's storage buffers (the
    host arrays they were made from decide whether a new frame may use them), the image accumulator (cleared, not made, each
    frame), a uniform buffer and bind group per dispatch slot, and the readback buffers. One per wgpu device; renders take
    `lock`, so two threads never share the accumulator or the uniform slots."""

    def __init__(self):
        self.lock = threading.RLock()
        self.release()

    def release(self):
        for buffer in [*getattr(self, "scene_buffers", ()), getattr(self, "blocks", None), getattr(self, "accum", None),
                       getattr(self, "image", None), getattr(self, "counts", None), getattr(self, "resolve_params", None),
                       getattr(self, "uniform_block", None), *getattr(self, "staging", {}).values()]:
            if buffer is not None:
                buffer.destroy()
        self.arrays, self.scene_buffers, self.blocks = None, [], None
        self.accum, self.accum_size, self.image, self.counts, self.resolve_params = None, 0, None, None, None
        self.uniform_block, self.uniform_stride, self.uniform_bytes, self.uniform_scratch, self.staging = None, 0, 0, None, {}
        self.groups, self.group_key = {}, None
        self.resolve_group, self.resolve_key = None, None

    def nbytes(self):
        host = sum(a.nbytes for a in self.arrays or ())
        device = sum(b.size for b in [*self.scene_buffers, self.accum, self.image, self.uniform_block, *self.staging.values()]
                     if b is not None)
        return host + device

    def scene(self, device, wgpu, packed):
        """The six scene buffers for `packed`: the ones from the last frame when its arrays are identical, else new ones."""
        arrays = [packed.nodes, packed.order, packed.triangles, packed.shapes, packed.lights, packed.env]
        if _same_arrays(arrays, self.arrays):
            COUNTERS["scene_reuses"] += 1
            return self.scene_buffers
        for buffer in self.scene_buffers:
            buffer.destroy()
        self.scene_buffers = [device.create_buffer_with_data(data=a, usage=wgpu.BufferUsage.STORAGE) for a in arrays]
        self.arrays = arrays
        if self.blocks is None:
            self.blocks = device.create_buffer_with_data(data=np.zeros(4, "u4"), usage=wgpu.BufferUsage.STORAGE)
        self.groups, self.group_key = {}, None
        COUNTERS["scene_uploads"] += 1
        return self.scene_buffers

    def accumulator(self, device, wgpu, size):
        """The image accumulator, zeroed: the same buffer as last frame when it has this size (cleared on the card), else new."""
        if self.accum is not None and self.accum_size == size:
            COUNTERS["accum_reuses"] += 1
            encoder = device.create_command_encoder()
            encoder.clear_buffer(self.accum)
            device.queue.submit([encoder.finish()])
            COUNTERS["submissions"] += 1
        else:
            if self.accum is not None:
                self.accum.destroy()
            if self.image is not None:           # sized for the old image
                self.image.destroy()
                self.image = None
            self.accum = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                                              | wgpu.BufferUsage.COPY_DST)
            self.accum_size = size
            self.groups, self.group_key = {}, None
            self.resolve_group = None
            COUNTERS["accum_buffers"] += 1
        return self.accum

    def uniforms(self, device, wgpu, uniform_bytes):
        """The stride of the dispatch slots in the one uniform buffer all of a submission's dispatches read (each binds its own
        `uniform_bytes` at an offset of the adapter-guaranteed 256-byte alignment), and a scratch array for writing them."""
        if self.uniform_block is None or self.uniform_bytes != uniform_bytes:
            if self.uniform_block is not None:
                self.uniform_block.destroy()
            self.uniform_bytes = uniform_bytes
            self.uniform_stride = -(-uniform_bytes // 256) * 256
            self.uniform_block = device.create_buffer(size=MAX_JOBS * self.uniform_stride,
                                                      usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            self.uniform_scratch = bytearray(MAX_JOBS * self.uniform_stride)
            self.groups = {}
            COUNTERS["uniform_buffers"] += 1
        return self.uniform_stride

    def group(self, device, wgpu, pipeline, slot):
        """The bind group of dispatch `slot`, built once per scene, accumulator and pipeline."""
        key = (pipeline, id(self.accum))
        if self.group_key != key:
            self.groups, self.group_key = {}, key
        if slot not in self.groups:
            entries = [{"binding": i, "resource": {"buffer": b}} for i, b in enumerate([*self.scene_buffers, self.accum])]
            entries.append({"binding": len(entries), "resource": {"buffer": self.uniform_block, "offset": slot * self.uniform_stride,
                                                                  "size": self.uniform_bytes}})
            entries.append({"binding": len(entries), "resource": {"buffer": self.blocks}})
            self.groups[slot] = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
            COUNTERS["bind_groups"] += 1
        return self.groups[slot]

    def stage(self, device, wgpu, size):
        """A mappable buffer of `size` bytes to read the card's answers through (made once per size)."""
        if size not in self.staging:
            self.staging[size] = device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            COUNTERS["staging_buffers"] += 1
        return self.staging[size]


def _frame(state):
    if "_gpupt_frame" not in state:
        state["_gpupt_frame"] = _Frame()
    return state["_gpupt_frame"]


def release_frame_cache(state=None):
    """Free what the renderer keeps on the device between frames (a scene's buffers, the accumulator, the readback buffers)."""
    state = state or gpu3d._state()
    frame = state.get("_gpupt_frame")
    if frame is not None:
        with frame.lock:
            frame.release()


def _resolve_pipeline(state):
    cache = state.setdefault("_gpupt_resolve", {})
    if "pipeline" not in cache:
        device = state["device"]
        COUNTERS["resolve_pipelines"] += 1
        cache["pipeline"] = device.create_compute_pipeline(layout="auto", compute={
            "module": device.create_shader_module(code=_RESOLVE_SHADER), "entry_point": "main"})
    return cache["pipeline"]


def _wait_for(device, frame, wgpu, encoder, source):
    """Finish `encoder` with a copy of the first 16 bytes of `source` into a mappable buffer, submit it and read those bytes
    back: the read returns when the card has run everything in the submission."""
    stage = frame.stage(device, wgpu, 16)
    encoder.copy_buffer_to_buffer(source, 0, stage, 0, 16)
    device.queue.submit([encoder.finish()])
    COUNTERS["submissions"] += 1
    stage.map_async("READ").sync_wait()
    stage.unmap()


_TILE_INDEX = {}


def _tile_index(width, height, tiles_x):
    """The tile (of `pt.TILE` pixels square) each pixel belongs to, as a flat index (kept for the last image size)."""
    key = (width, height, tiles_x)
    if _TILE_INDEX.get("key") != key:
        index = np.arange(width * height)
        _TILE_INDEX.update(key=key, value=((index // width) // pt.TILE) * tiles_x + (index % width) // pt.TILE)
    return _TILE_INDEX["value"]


def _resolved_image(state, device, wgpu, frame, accum, width, npix, tiles_x, tile_count, data_pass, background, adaptive=False):
    """The finished image as float32 (npix, 4), made on the card: the channel sums divided by their pixel's sample count (the
    count of its tile; its own, for adaptive sampling) with the premultiplied `background` laid under the coverage, or a data
    pass's values where covered. Only these 16 bytes a pixel cross to the host, and for adaptive sampling 4 more: the pixel's
    sample count with its done flag in the top bit (returned as the second value, else None). `_RESOLVE_SHADER` does the
    arithmetic of the host conversion."""
    pipeline = _resolve_pipeline(state)
    size = npix * (20 if adaptive else 16)
    if frame.image is None or frame.image.size != size:
        if frame.image is not None:
            frame.image.destroy()
        frame.image = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        frame.resolve_group = None
    need = max(16, 4 * len(tile_count))
    if frame.counts is None or frame.counts.size < need:
        if frame.counts is not None:
            frame.counts.destroy()
        frame.counts = device.create_buffer(size=need, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        frame.resolve_group = None
    if frame.resolve_params is None:
        frame.resolve_params = device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        frame.resolve_group = None
    groups = -(-npix // 64)
    groups_x = min(groups, 4096)
    groups_y = -(-groups // groups_x)
    device.queue.write_buffer(frame.counts, 0, np.ascontiguousarray(tile_count, "u4"))
    device.queue.write_buffer(frame.resolve_params, 0, struct.pack(
        "<8I4f", npix, width, tiles_x, pt.TILE, int(data_pass), groups_x * 64, int(adaptive), 0, *(float(v) for v in background)))
    if frame.resolve_group is None:
        frame.resolve_group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": b}}
            for i, b in enumerate((accum, frame.counts, frame.image, frame.resolve_params))])
        COUNTERS["bind_groups"] += 1
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, frame.resolve_group)
    compute.dispatch_workgroups(groups_x, groups_y, 1)
    compute.end()
    stage = frame.stage(device, wgpu, size)
    encoder.copy_buffer_to_buffer(frame.image, 0, stage, 0, size)
    device.queue.submit([encoder.finish()])
    COUNTERS["submissions"] += 1
    stage.map_async("READ").sync_wait()
    data = stage.read_mapped(copy=True)
    stage.unmap()
    image = np.frombuffer(data, "f4", npix * 4).reshape(npix, 4)
    return image, (np.frombuffer(data, "u4", npix, npix * 16) if adaptive else None)


# --- rendering ------------------------------------------------------------------------------------------

def _adaptive_passes(state, device, wgpu, pipeline, buffers, accum, upload, resources, packed, ps, camera, width, height, wg,
                     heavy, settings, code, tiles_x, total_samples, cancel, progress, started, readbacks, lap, sparse_threads):
    """Adaptive sampling on the device. Every sample is one indirect dispatch over the pixels the last `scan` listed as open;
    after each pass `scan` and `finish` (`_COMPACT_SHADER`) rebuild that list from the shader's per-pixel done flags. The
    running moments and the done flags stay in `accum`, so nothing is read back while the render runs: the host only waits for
    the card now and then (every pass when it must report progress, honour a time limit or a cancel, `ADAPTIVE_PACE` passes
    otherwise) and, when `progress` is given, reads the list's length (one word) after each pass. The finished accumulator
    holds the result: the caller reads it once."""
    scan, finish, reduce = _compact_pipelines(state)
    npix = width * height
    threads = wg * wg
    budget = ADAPTIVE_PATHS_PER_DISPATCH // (SOFT_SLOWDOWN if heavy else 1)
    chunk_groups = int(max(1, min(ADAPTIVE_MAX_GROUPS, budget // threads)))
    chunks = -(-npix // (chunk_groups * threads))
    sparse_args, reduce_args = chunks * 16, chunks * 16 + 16      # byte offsets of the other two indirect arguments
    header = 0
    list_base = header + 4
    ctl = device.create_buffer(size=4 * (list_base + npix), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    args = device.create_buffer(size=16 * (chunks + 2), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.INDIRECT)
    resources.extend((ctl, args))
    sparse_pixels = min(ADAPTIVE_SPARSE_PIXELS, sparse_threads)
    compact = np.zeros(16, "u4")
    compact[:12] = (width, height, 0, 0, chunks, chunk_groups, list_base, threads, sparse_pixels, sparse_threads,
                    settings.min_samples, settings.max_samples)
    compact.view("f4")[12:14] = (settings.noise_threshold, settings.adaptive_pass_size)
    compact = upload(compact, wgpu.BufferUsage.UNIFORM)

    def group(pipe, entries):
        return device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": b}} for i, b in enumerate(entries)])
    scan_group, reduce_group = (group(pipe, [accum, ctl, compact]) for pipe in (scan, reduce))
    finish_group = group(finish, [accum, ctl, compact, args])
    none = np.zeros(0, "u4")
    groups = []
    for c in range(chunks):
        uniform = upload(_uniform(packed, ps, camera, width, height, list_base, header + 2, c * chunk_groups * threads, 1,
                                  settings, code, none, tiles_x), wgpu.BufferUsage.UNIFORM)
        groups.append(group(pipeline, [*buffers, accum, uniform, ctl]))
    uniform = upload(_uniform(packed, ps, camera, width, height, list_base, header + 2, 0, 1, settings, code, none, tiles_x,
                              mode=3.0), wgpu.BufferUsage.UNIFORM)
    sparse_group = group(pipeline, [*buffers, accum, uniform, ctl])

    def sparse(compute):
        # the pass's samples side by side for the few pixels still open (zero groups when the pass was dense), then their sum
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, sparse_group)
        compute.dispatch_workgroups_indirect(args, sparse_args)
        compute.set_pipeline(reduce)
        compute.set_bind_group(0, reduce_group)
        compute.dispatch_workgroups_indirect(args, reduce_args)

    def compaction(compute):
        compute.set_pipeline(scan)
        compute.set_bind_group(0, scan_group)
        compute.dispatch_workgroups(-(-width // 8), -(-height // 8), 1)
        compute.set_pipeline(finish)
        compute.set_bind_group(0, finish_group)
        compute.dispatch_workgroups(1, 1, 1)

    def wait():
        # the read of the list's length (one word) is the wait; `on_submitted_work_done_sync` fails to build its callback in
        # wgpu-py 0.32. It is the number of pixels the next pass would take, 0 once every pixel is done.
        lap("encode")
        open_now = int(np.frombuffer(device.queue.read_buffer(ctl, (header + 2) * 4, 4), "u4")[0])
        readbacks["waits"] += 1
        lap("dispatch")
        return open_now
    lap("adaptive setup")
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compaction(compute)                    # every pixel is open before the first pass
    compute.end()
    device.queue.submit([encoder.finish()])
    sample_index, passes = 0, 0
    asked = progress is not None or bool(settings.time_limit) or cancel is not None
    while sample_index < total_samples:
        raytrace._cancel(cancel)
        take = min(settings.min_samples if sample_index == 0 else settings.adaptive_pass_size, total_samples - sample_index)
        if heavy:
            # a heavy sample takes long enough that each chunk is its own short submission
            for _ in range(take):
                for c in range(chunks):
                    raytrace._cancel(cancel)
                    encoder = device.create_command_encoder()
                    compute = encoder.begin_compute_pass()
                    compute.set_pipeline(pipeline)
                    compute.set_bind_group(0, groups[c])
                    compute.dispatch_workgroups_indirect(args, c * 16)
                    compute.end()
                    device.queue.submit([encoder.finish()])
                    wait()
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            sparse(compute)
            compaction(compute)
        else:
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            for c in range(chunks):
                # a sample's dispatches are interchangeable (each pixel counts its own samples), so a chunk's samples run
                # back to back under one bind group instead of re-binding it for every sample
                compute.set_bind_group(0, groups[c])
                for _ in range(take):
                    compute.dispatch_workgroups_indirect(args, c * 16)
            sparse(compute)
            compaction(compute)
        compute.end()
        device.queue.submit([encoder.finish()])
        sample_index += take
        passes += 1
        pixels_open = None
        if asked or heavy or passes % ADAPTIVE_PACE == 0:
            pixels_open = wait()
        else:
            lap("encode")
        if progress is not None:
            if pixels_open is None:
                pixels_open = int(np.frombuffer(device.queue.read_buffer(ctl, (header + 2) * 4, 4), "u4")[0])
                readbacks["counter"] += 1
                lap("flag readback")
            elapsed = time.perf_counter() - started
            converged = 1.0 - pixels_open / (width * height)
            progress("pathtrace", max(sample_index / total_samples, converged),
                     dict(samples=sample_index, passes=passes, seconds=elapsed, tiles_active=-(-pixels_open // threads),
                          converged=converged, pixels_active=pixels_open))
        if pixels_open == 0:
            break              # every pixel is done: the passes left would dispatch nothing
        if settings.time_limit and time.perf_counter() - started >= settings.time_limit:
            break
    wait()                     # the card may still owe the last passes: the final readback then times the transfer alone


def render(scene, camera, width, height, background, ambient, output, settings, cancel=None, progress=None,
           stats=None, volume=None, merge_volumes=True):
    """Path trace on the GPU; raises `gpu3d.Unsupported` for what it cannot take (the caller falls back).
    `merge_volumes` False leaves the smoke out of a data pass (its surface depth)."""
    if output not in _OUTPUT_CODES:
        raise gpu3d.Unsupported(f"the GPU path tracer does not produce the {output!r} output")
    state = gpu3d._state()
    reason = check_capability(state)
    if reason:
        raise gpu3d.Unsupported(reason)
    started = time.perf_counter()
    phases = {}           # seconds per phase of this render, reported as stats["phases"] (tools/benchmark_adaptive.py --breakdown)
    mark = started

    def lap(name):
        nonlocal mark
        now = time.perf_counter()
        phases[name] = phases.get(name, 0.0) + now - mark
        mark = now
    raytrace._cancel(cancel)
    ps = pt.build_scene(scene, ambient, eye=s._view_basis(camera)[0], volume=volume)
    lap("scene build")
    if (ps.splats is not None or ps.volumes is not None) and not soft_supported(state):
        raise gpu3d.Unsupported("splats and smoke are path traced on the GPU on NVIDIA adapters only for now")
    try:
        packed = pack(ps, cancel=cancel)
    except ValueError as exc:
        if "traversal stack" in str(exc):
            raise gpu3d.Unsupported(str(exc)) from exc
        raise
    lap("pack")
    width, height = int(width), int(height)
    tiles_x, tiles_y = -(-width // pt.TILE), -(-height // pt.TILE)
    if tiles_x * tiles_y > 512 * 4 * 32:
        raise gpu3d.Unsupported("the image has more tiles than the GPU path tracer tracks")
    limits = gpurt._limits(state)
    cap = min(limits.get("max-buffer-size", 0), limits.get("max-storage-buffer-binding-size", 0))
    npix = width * height
    accum_bytes = npix * 32
    for name, data in (("nodes", packed.nodes), ("triangles", packed.triangles), ("env", packed.env)):
        if data.nbytes > cap:
            raise ValueError(f"GPU path tracing needs a {data.nbytes / 2**20:.1f} MiB {name} buffer; the adapter allows {cap / 2**20:.1f} MiB")
    if accum_bytes > cap:
        raise ValueError(f"GPU path tracing needs {accum_bytes / 2**20:.1f} MiB for the image; the adapter allows {cap / 2**20:.1f} MiB")
    device, wgpu = state["device"], state["wgpu"]
    frame = _frame(state)
    resources = []          # buffers of this render only; the scene, accumulator and readback buffers belong to `frame`

    def upload(data, usage):
        buffer = device.create_buffer_with_data(data=data, usage=usage)
        resources.append(buffer)
        return buffer
    data_pass = output in pt.DATA_OUTPUTS
    code = _OUTPUT_CODES[output]
    storage = wgpu.BufferUsage.STORAGE
    adaptive = settings.adaptive and not data_pass
    readbacks = {"waits": 0, "mask": 0, "counter": 0, "image": 0}     # stats["readbacks"]: what came back from the card, and why
    heavy = ps.splats is not None or ps.volumes is not None
    # A caller that sets `stats["skip_noise_maps"]` does not read stats["variance"] or stats["noise"], so the render finishes
    # the image on the card (the divide, the background) and reads back 16 bytes a pixel (20 with adaptive sampling's counts
    # and done flags) instead of the 32 of sums and moments, converting nothing on the host.
    lean = stats is None or bool(stats.get("skip_noise_maps"))
    raw = lean_image = lean_meta = None
    tile_of = None
    done = False
    frame.lock.acquire()
    try:
        buffers = frame.scene(device, wgpu, packed)
        # the sparse passes' per-sample sums sit after the image: as many (pixel, sample) slots as a pass can use, the constant
        # allows and the adapter has room for (none: every pass is dense)
        sparse_threads = 0
        if adaptive:
            lanes = min(max(settings.min_samples, settings.adaptive_pass_size), 1024)
            sparse_threads = int(max(0, min(ADAPTIVE_SPARSE_THREADS, npix * lanes, (cap - accum_bytes) // 32)))
        accum = frame.accumulator(device, wgpu, npix * 32 + sparse_threads * 32)       # zeroed
        links = bool(ps.linked_bits)
        pipeline = _pipeline(state, ps.splats is not None, ps.volumes is not None, links, packed.sparse_volumes)
        wg = _wg_size(ps.splats is not None, ps.volumes is not None, links)
        lap("upload")
        if data_pass or adaptive:
            per_pass = 1
        elif settings.pass_samples:
            per_pass = max(1, settings.pass_samples)
        elif settings.noise_threshold > 0:       # the legacy noise test retires tiles after every pass
            per_pass = 1
        elif heavy:
            per_pass = SOFT_PASS_SAMPLES
        else:
            per_pass = SURFACE_PASS_SAMPLES       # a dispatch of one sample leaves the card mostly launching (Z1 on the RTX: 2.3 times slower)
        total_samples = 1 if data_pass else (settings.max_samples if adaptive else settings.samples)
        sample_index, passes = 0, 0
        tile_done = np.zeros(tiles_x * tiles_y, bool)
        tile_count = np.zeros(tiles_x * tiles_y, np.int64)
        if adaptive:
            _adaptive_passes(state, device, wgpu, pipeline, buffers, accum, upload, resources, packed, ps, camera, width, height,
                             wg, heavy, settings, code, tiles_x, total_samples, cancel, progress, started, readbacks, lap,
                             sparse_threads)
        else:
            legacy_mask = settings.noise_threshold > 0 and not data_pass
            # A submission carries dispatches (one pass over some rows each) until their paths reach `sub_paths`. A smoke or
            # splat submission starts small (SOFT_SLOWDOWN times fewer paths than a surface one: a path can be that much
            # slower) and it grows with the measured time per path to about BAND_TARGET_SECONDS; with the whole image in one
            # dispatch, the growth adds more passes to the submission. A band of 65 000 paths left the RTX 3080 Ti mostly
            # idle (the render took 2.6 times longer than one dispatch over the whole image at the same sample count), and a
            # wait after every pass of a surface scene costs more host time than the pass itself.
            sub_floor = sub_paths = GPU_PATHS_PER_SUBMISSION // (SOFT_SLOWDOWN if heavy else 1)
            jobs, job_paths = [], 0          # (row0, row1, first sample, samples) waiting for the next submission
            bits, base = None, None
            appended = flushed = reported = 0       # samples of the passes in `jobs` or submitted / finished / reported to `progress`
            last_flush = time.perf_counter()

            def flush(full=False):
                nonlocal job_paths, sub_paths, last_flush, flushed
                if not jobs:
                    return
                through = appended
                lap("job list")
                stride = frame.uniforms(device, wgpu, len(base))
                scratch = frame.uniform_scratch
                for slot, (y0, y1, first, take_) in enumerate(jobs):       # every dispatch's uniform, written in one go
                    scratch[slot * stride:slot * stride + len(base)] = base
                    struct.pack_into("<IIII", scratch, slot * stride + 8, y0, y1, first, take_)    # row range, first sample, samples
                device.queue.write_buffer(frame.uniform_block, 0, memoryview(scratch)[:len(jobs) * stride])
                lap("uniform write")
                encoder = device.create_command_encoder()
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(pipeline)
                for slot, (y0, y1, first, take_) in enumerate(jobs):
                    compute.set_bind_group(0, frame.group(device, wgpu, pipeline, slot))
                    compute.dispatch_workgroups(-(-width // wg), -(-(y1 - y0) // wg), 1)
                compute.end()
                lap("encode")
                began = time.perf_counter()
                _wait_for(device, frame, wgpu, encoder, accum)     # keeps submissions short and cancellation prompt
                took = time.perf_counter() - began
                readbacks["waits"] += 1
                lap("dispatch")
                if BAND_TARGET_SECONDS > 0 and full:       # a short last submission says nothing about the card
                    fit = job_paths * BAND_TARGET_SECONDS / max(took, 1e-4)
                    sub_paths = int(min(BAND_MAX_PATHS, max(sub_floor, min(fit, sub_paths * 4))))
                jobs.clear()
                job_paths = 0
                flushed = through
                last_flush = time.perf_counter()
            while sample_index < total_samples:
                raytrace._cancel(cancel)
                take = min(per_pass, total_samples - sample_index)
                if bits is None:                           # the tile mask changes only when the legacy noise test retires tiles
                    bits = _tile_bits(tile_done)
                    if tile_done.all():
                        break
                    base = bytearray(_uniform(packed, ps, camera, width, height, 0, height, 0, 1, settings, code, bits, tiles_x))
                    lap("mask update")
                y0 = 0
                while y0 < height:
                    raytrace._cancel(cancel)
                    rows = max(1, sub_paths // max(width * take, 1))
                    y1 = min(height, y0 + rows)
                    jobs.append((y0, y1, sample_index, take))
                    job_paths += (y1 - y0) * width * take
                    y0 = y1
                    if y0 >= height:
                        appended = sample_index + take         # this pass is now whole in the submission being filled
                    if job_paths >= sub_paths or len(jobs) >= MAX_JOBS:
                        flush(full=True)
                if legacy_mask:
                    tile_count[~tile_done] += take
                else:
                    tile_count += take             # no tile retires without the legacy noise test
                sample_index += take
                passes += 1
                if legacy_mask:
                    flush()
                    if tile_of is None:
                        tile_of = _tile_index(width, height, tiles_x)
                    raw = np.frombuffer(device.queue.read_buffer(accum, 0, npix * 32), "f4").reshape(-1, 2, 4)
                    readbacks["mask"] += 1
                    lum_sum, lum_sq = raw[:, 1, 0].astype(np.float64), raw[:, 1, 1].astype(np.float64)
                    count = tile_count[tile_of]
                    pt._retire_tiles(tile_done, tile_of, lum_sum, lum_sq, count, tiles_x * tiles_y, settings.noise_threshold)
                    raw = None
                    bits = None
                    lap("mask update")
                elif jobs and (passes == 1 or ((progress is not None or settings.time_limit)
                                               and time.perf_counter() - last_flush >= BAND_TARGET_SECONDS)):
                    # The first pass goes alone (its time sizes the next submissions, and `progress` and a time limit hear of it
                    # at once); after that they hear of a pass boundary about every BAND_TARGET_SECONDS, not after a long wait.
                    flush()
                if flushed == sample_index:
                    elapsed = time.perf_counter() - started
                    if progress is not None:
                        reported = sample_index
                        progress("pathtrace", sample_index / total_samples,
                                 dict(samples=sample_index, passes=passes, seconds=elapsed, tiles_active=int((~tile_done).sum()),
                                      converged=float(tile_done[tile_of].mean()) if tile_done.any() else 0.0))
                    if settings.time_limit and elapsed >= settings.time_limit:
                        break
            flush()
            if progress is not None and reported < sample_index == flushed:      # the last passes were submitted together
                progress("pathtrace", sample_index / total_samples,
                         dict(samples=sample_index, passes=passes, seconds=time.perf_counter() - started,
                              tiles_active=int((~tile_done).sum()),
                              converged=float(tile_done[tile_of].mean()) if tile_done.any() else 0.0))
        raytrace._cancel(cancel)
        if lean:
            background_premult = np.zeros(4, np.float64)
            if output == "rgba":
                background_premult = np.asarray(background, np.float64).copy()
                background_premult[3] = np.clip(background_premult[3], 0, 1)
                background_premult[:3] *= background_premult[3]
            lean_image, lean_meta = _resolved_image(state, device, wgpu, frame, accum, width, npix, tiles_x, tile_count,
                                                    data_pass, background_premult, adaptive)
        else:
            raw = np.frombuffer(device.queue.read_buffer(accum, 0, npix * 32), "f4").reshape(npix, 2, 4)
        readbacks["image"] += 1
        lap("final readback")
        done = True
    finally:
        for resource in reversed(resources):
            resource.destroy()
        if not done or frame.nbytes() > CACHE_LIMIT_BYTES:
            frame.release()
        frame.lock.release()
    if lean:
        pixel_done = None
        count = None
        if adaptive:
            pixel_samples = (lean_meta & 0x7FFFFFFF).astype(np.int64)
            pixel_done = (lean_meta >> 31).astype(bool)
        elif tile_count.min() == tile_count.max():
            pixel_samples = np.full(npix, tile_count[0], np.int64)
        else:
            tile_of = tile_of if tile_of is not None else _tile_index(width, height, tiles_x)
            pixel_samples = tile_count[tile_of]
        image = lean_image
    else:
        pixel_done = raw[:, 1, 3] > 0.5 if adaptive else None       # adaptive: the shader's own per-pixel done flags
        if not adaptive:
            tile_of = tile_of if tile_of is not None else _tile_index(width, height, tiles_x)
        pixel_samples = raw[:, 1, 2].astype(np.int64) if adaptive else tile_count[tile_of]
    if adaptive:
        # passes that did any work: the first takes the minimum, each later one `adaptive_pass_size` more for the pixels left
        # (passes after every pixel is done are encoded but dispatch nothing, and the host never read how many there were)
        passes = 1 + -(-max(int(pixel_samples.max()) - settings.min_samples, 0) // settings.adaptive_pass_size)
    if not lean:
        count = np.maximum(pixel_samples, 1).astype(np.float32)
        if data_pass:
            image = raw[:, 0, :]
            image = np.where(image[:, 3:4] > 0, image, 0.0)
        else:
            image = raw[:, 0, :] / count[:, None]                 # float32 throughout: the accumulator is float32 already
            if output == "rgba":
                bg = np.asarray(background, np.float64).copy()
                bg[3] = np.clip(bg[3], 0, 1)
                bg[:3] *= bg[3]
                image = image + bg.astype(np.float32) * (np.float32(1) - image[:, 3:4])
    lap("convert")
    if stats is not None:
        if lean and not adaptive:
            converged = (np.zeros(npix, bool) if not tile_done.any() else tile_done[tile_of if tile_of is not None
                                                                                   else _tile_index(width, height, tiles_x)])
        else:
            converged = pixel_done if adaptive else tile_done[tile_of]
        stats.update(backend="gpu", sampling=settings.sampling, samples=pixel_samples.reshape(height, width),
                     passes=passes, seconds=time.perf_counter() - started, adapter=gpu3d.describe(), phases=phases,
                     readbacks=readbacks, converged=converged.reshape(height, width))
        if not data_pass and not lean:
            # the variance of each pixel's mean luminance from the moments the shader summed, as the CPU reference has it
            n = count.astype(np.float64)
            lum_sum, lum_sq = raw[:, 1, 0].astype(np.float64), raw[:, 1, 1].astype(np.float64)
            mean_lum = lum_sum / n
            variance = np.maximum(lum_sq / n - mean_lum * mean_lum, 0.0) * n / np.maximum(n - 1, 1) / n
            stats["variance"] = variance.reshape(height, width)
            stats["noise"] = pt.pixel_noise(lum_sum, lum_sq, n).reshape(height, width)
        lap("stats")
    result = np.ascontiguousarray(image, np.float32).reshape(height, width, 4)
    if data_pass and ps.volumes is not None and merge_volumes:
        # the smoke's first sample is found by the CPU raymarch for both tracers (it is one ray per pixel), cut at the
        # card's own surface depth
        surface = result if output == "depth" else render(scene, camera, width, height, background, ambient, "depth",
                                                           settings, cancel=cancel, volume=volume, merge_volumes=False)
        result = pt.merge_volume_data(scene, camera, width, height, result, output, ps.volumes.settings, surface, cancel)
    result.flags.writeable = False
    lap("postprocess")
    return result
