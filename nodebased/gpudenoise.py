"""The path tracer's edge-aware denoiser (`ptdenoise.denoise`) on the graphics card (lane 4, plan "Render mode responsiveness", step C).

`ptdenoise.denoise` is four a-trous passes of a 5 x 5 kernel steered by albedo, normal, depth and a local variance estimate, written
as numpy array arithmetic: 1.4 s for a 1920 x 1080 picture on the 32-thread workstation, run again for every picture Render mode
shows. This is the same filter as compute shaders (the same weights, the same clamped taps, float32), about a hundredth of the
time on the RTX 3080 Ti. It handles what the viewport asks for (`variance=None`, no history, no strength); anything else, a card
that cannot take it or an image larger than one storage binding stays on the numpy filter (`available`, `denoise` raising
`gpu3d.Unsupported`).

The guides (albedo, normals, depth) of a view do not change from one pass to the next, so they are packed and uploaded once and
kept with the arrays they came from. Each call uploads the beauty, runs one prepare pass (demodulate by the albedo, the unit
albedo, the effective depth), the variance estimate (a 7 x 7 window, then a 3 x 3 mean), four filter iterations and the final
modulation, and reads the picture back. `cancel` (an event) is looked at between the iterations; each is waited for, so the card
never holds work a cancelled pass queued.
"""
from __future__ import annotations

import threading

import numpy as np

from . import gpu3d, gpurt, ptdenoise
from .cancellation import Cancelled

_SHADER = r'''
struct P {
  size: vec4<u32>,     // width, height, step, 0
  f: vec4<f32>,        // phi_color, phi_normal, phi_depth, sigma_albedo
};
@group(0) @binding(0) var<storage, read_write> src: array<vec4<f32>>;     // premultiplied beauty; also the final picture
@group(0) @binding(1) var<storage, read_write> alb: array<vec4<f32>>;     // albedo (rgb), view depth (w)
@group(0) @binding(2) var<storage, read_write> nrm: array<vec4<f32>>;     // world normal (xyz)
@group(0) @binding(3) var<storage, read_write> ea: array<vec4<f32>>;      // demodulated radiance (rgb), variance (w)
@group(0) @binding(4) var<storage, read_write> eb: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> au: array<vec4<f32>>;      // unit albedo (rgb), effective depth (w)
@group(0) @binding(6) var<storage, read_write> nz: array<vec4<f32>>;      // normal (xyz), covered (w)
@group(0) @binding(7) var<uniform> p: P;

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);
const ALBEDO_FLOOR = 1e-3;

fn at(x: i32, y: i32) -> u32 {
  let w = i32(p.size.x);
  let h = i32(p.size.y);
  return u32(clamp(y, 0, h - 1)) * p.size.x + u32(clamp(x, 0, w - 1));
}

fn inside(gid: vec3<u32>) -> bool { return gid.x < p.size.x && gid.y < p.size.y; }

@compute @workgroup_size(8, 8, 1)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!inside(gid)) { return; }
  let i = gid.y * p.size.x + gid.x;
  let c = src[i];
  let a = alb[i];
  let covered = c.w > 1e-4;
  let demod = max(a.x, max(a.y, a.z)) > ALBEDO_FLOOR;
  let denominator = select(vec3<f32>(1.0), max(a.xyz, vec3<f32>(ALBEDO_FLOOR)), demod);
  ea[i] = vec4<f32>(c.xyz / denominator, 0.0);
  au[i] = vec4<f32>(select(vec3<f32>(0.0), a.xyz / max(c.w, 1e-6), demod), select(0.0, a.w, covered));
  nz[i] = vec4<f32>(nrm[i].xyz, select(0.0, 1.0, covered));
}

// the spread of the luminance in a 7 x 7 window, a stand-in for the variance of the mean when the renderer gives none
@compute @workgroup_size(8, 8, 1)
fn spread(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!inside(gid)) { return; }
  let i = gid.y * p.size.x + gid.x;
  var total = 0.0;
  var squares = 0.0;
  for (var dy = -3; dy <= 3; dy++) {
    for (var dx = -3; dx <= 3; dx++) {
      let l = dot(ea[at(i32(gid.x) + dx, i32(gid.y) + dy)].xyz, LUMA);
      total += l;
      squares += l * l;
    }
  }
  let mean = total / 49.0;
  eb[i] = vec4<f32>(ea[i].xyz, max(squares / 49.0 - mean * mean, 0.0) * 0.25);
}

fn box3(buf_x: i32, buf_y: i32, from_b: bool) -> f32 {
  var s = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let j = at(buf_x + dx, buf_y + dy);
      s += select(ea[j].w, eb[j].w, from_b);
    }
  }
  return s / 9.0;
}

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!inside(gid)) { return; }
  let i = gid.y * p.size.x + gid.x;
  ea[i] = vec4<f32>(eb[i].xyz, box3(i32(gid.x), i32(gid.y), true));
}

fn z_at(x: i32, y: i32) -> f32 { return au[at(x, y)].w; }

// one a-trous iteration: reads `ea` (or `eb` when p.size.w is 1), writes the other
@compute @workgroup_size(8, 8, 1)
fn iterate(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!inside(gid)) { return; }
  let w = i32(p.size.x);
  let h = i32(p.size.y);
  let x = i32(gid.x);
  let y = i32(gid.y);
  let i = gid.y * p.size.x + gid.x;
  let flip = p.size.w == 1u;
  let step = i32(p.size.z);
  var e_here: vec4<f32>;
  if (flip) { e_here = eb[i]; } else { e_here = ea[i]; }
  // the 3 x 3 mean of the variance, then sigma of the radiance weight
  var vs = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let j = at(x + dx, y + dy);
      if (flip) { vs += eb[j].w; } else { vs += ea[j].w; }
    }
  }
  let sigma_l = p.f.x * sqrt(max(vs / 9.0, 0.0)) + 1e-4;
  // np.gradient of the effective depth: central differences inside, one-sided at the edges
  var gzx = 0.0;
  var gzy = 0.0;
  if (w > 1) {
    if (x == 0) { gzx = z_at(1, y) - z_at(0, y); }
    else if (x == w - 1) { gzx = z_at(w - 1, y) - z_at(w - 2, y); }
    else { gzx = 0.5 * (z_at(x + 1, y) - z_at(x - 1, y)); }
  }
  if (h > 1) {
    if (y == 0) { gzy = z_at(x, 1) - z_at(x, 0); }
    else if (y == h - 1) { gzy = z_at(x, h - 1) - z_at(x, h - 2); }
    else { gzy = 0.5 * (z_at(x, y + 1) - z_at(x, y - 1)); }
  }
  let n_p = nz[i].xyz;
  let cov_p = nz[i].w;
  let z_p = au[i].w;
  let a_p = au[i].xyz;
  let l_p = dot(e_here.xyz, LUMA);
  var e_sum = vec3<f32>(0.0);
  var v_sum = 0.0;
  var w_sum = 0.0;
  let k = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
  for (var ky = -2; ky <= 2; ky++) {
    for (var kx = -2; kx <= 2; kx++) {
      let dy = ky * step;
      let dx = kx * step;
      let h_k = k[ky + 2] * k[kx + 2];
      var weight = h_k;
      var e_q = e_here.xyz;
      var v_q = e_here.w;
      if (dx != 0 || dy != 0) {
        let j = at(x + dx, y + dy);
        var q: vec4<f32>;
        if (flip) { q = eb[j]; } else { q = ea[j]; }
        e_q = q.xyz;
        v_q = q.w;
        let nq = nz[j];
        let dn = dot(n_p, nq.xyz);
        var w_n = 0.0;
        if (dn > 0.0) { w_n = pow(dn, p.f.y); }
        let aq = au[j];
        let plane = abs(gzx * f32(dx)) + abs(gzy * f32(dy));
        let w_z = exp(-abs(z_p - aq.w) / (p.f.z * plane + 1e-3 * max(z_p, 1e-3)));
        let w_l = exp(-abs(l_p - dot(e_q, LUMA)) / sigma_l);
        let da = a_p - aq.xyz;
        let w_a = exp(-dot(da, da) / (2.0 * p.f.w * p.f.w));
        weight = h_k * w_n * w_z * w_l * w_a * select(0.0, 1.0, nq.w == cov_p);
      }
      e_sum += weight * e_q;
      v_sum += weight * weight * v_q;
      w_sum += weight;
    }
  }
  let result = vec4<f32>(e_sum / w_sum, v_sum / (w_sum * w_sum));
  if (flip) { ea[i] = result; } else { eb[i] = result; }
}

// the filtered radiance back to the beauty's scale; coverage is the tracer's own
@compute @workgroup_size(8, 8, 1)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (!inside(gid)) { return; }
  let i = gid.y * p.size.x + gid.x;
  let a = alb[i];
  let demod = max(a.x, max(a.y, a.z)) > ALBEDO_FLOOR;
  let denominator = select(vec3<f32>(1.0), max(a.xyz, vec3<f32>(ALBEDO_FLOOR)), demod);
  var e: vec4<f32>;
  if (p.size.w == 1u) { e = eb[i]; } else { e = ea[i]; }
  src[i] = vec4<f32>(e.xyz * denominator, src[i].w);
}
'''

_ENTRIES = ("prepare", "spread", "blur", "iterate", "finish")
_LOCK = threading.Lock()
_CACHE = {}          # (state id, width, height) -> {"buffers": {...}, "guides": the arrays whose packed form is uploaded, "groups": ...}


def available():
    try:
        return gpu3d.available()
    except Exception:
        return False


def _pipelines(state):
    cache = state.setdefault("_gpudenoise", {})
    if "pipelines" not in cache:
        device = state["device"]
        module = device.create_shader_module(code=_SHADER)
        cache["pipelines"] = {name: device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": name})
                              for name in _ENTRIES}
    return cache["pipelines"]


def release(state=None):
    """Free the buffers kept between calls (a picture's worth of storage for each of the seven arrays, for the last two sizes)."""
    with _LOCK:
        for entry in list(_CACHE.values()):
            for buffer in entry["buffers"].values():
                buffer.destroy()
        _CACHE.clear()


KEPT_SIZES = 2          # the first look's small picture and the full-size one alternate from one view to the next


def _buffers(state, device, wgpu, width, height):
    key = (id(state), width, height)
    entry = _CACHE.get(key)
    if entry is not None:
        _CACHE[key] = _CACHE.pop(key)             # most recently used last
        return entry
    nbytes = width * height * 16
    limits = gpurt._limits(state)
    cap = min(limits.get("max-buffer-size", nbytes), limits.get("max-storage-buffer-binding-size", nbytes))
    if nbytes > cap:
        raise gpu3d.Unsupported(f"a {width} x {height} picture needs a {nbytes / 2**20:.0f} MiB storage binding; the adapter allows {cap / 2**20:.0f} MiB")
    while len(_CACHE) >= KEPT_SIZES:
        for buffer in _CACHE.pop(next(iter(_CACHE)))["buffers"].values():
            buffer.destroy()
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    names = ("src", "alb", "nrm", "ea", "eb", "au", "nz")
    entry = {"buffers": {name: device.create_buffer(size=nbytes, usage=usage) for name in names}, "guides": None,
             "uniform": device.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST), "groups": {}}
    entry["buffers"]["uniform"] = entry["uniform"]
    _CACHE[key] = entry
    return entry


def _pack_guides(albedo, normal, depth):
    h, w = depth.shape[:2]
    alb = np.empty((h, w, 4), np.float32)
    alb[..., :3] = albedo[..., :3]
    alb[..., 3] = depth if depth.ndim == 2 else depth[..., 0]
    nrm = np.zeros((h, w, 4), np.float32)
    nrm[..., :3] = normal[..., :3]
    return alb, nrm


def denoise(color, albedo, normal, depth, *, cancel=None, phi_color=ptdenoise.PHI_COLOR, phi_normal=ptdenoise.PHI_NORMAL,
            phi_depth=ptdenoise.PHI_DEPTH, sigma_albedo=ptdenoise.SIGMA_ALBEDO, iterations=ptdenoise.ITERATIONS):
    """`ptdenoise.denoise(color, albedo, normal, depth)` (variance estimated from the picture, no history, full strength) on the
    graphics card, float32 (H, W, 4). Raises `gpu3d.Unsupported` when the card cannot take it (the caller uses the numpy filter)."""
    if iterations < 1 or iterations > 8:
        raise gpu3d.Unsupported("the card filter takes one to eight iterations")
    state = gpu3d._state()
    device, wgpu = state["device"], state["wgpu"]
    color = np.ascontiguousarray(color, np.float32)
    h, w = color.shape[:2]
    if h < 2 or w < 2:
        raise gpu3d.Unsupported("a picture one pixel wide has no gradient to filter with")
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    pipelines = _pipelines(state)
    with _LOCK:
        entry = _buffers(state, device, wgpu, w, h)
        buffers = entry["buffers"]
        guides = entry["guides"]
        if guides is None or guides[0] is not albedo or guides[1] is not normal or guides[2] is not depth:
            alb, nrm = _pack_guides(np.asarray(albedo), np.asarray(normal), np.asarray(depth))
            device.queue.write_buffer(buffers["alb"], 0, alb)
            device.queue.write_buffer(buffers["nrm"], 0, nrm)
            entry["guides"] = (albedo, normal, depth)        # the arrays themselves are kept, so an identical one is never a stale id
        device.queue.write_buffer(buffers["src"], 0, color)
        device.queue.write_buffer(entry["uniform"], 0, np.array([w, h, 1, 0], "u4").tobytes() + np.array(
            [phi_color, phi_normal, phi_depth, sigma_albedo], "f4").tobytes())
        groups = entry["groups"]
        for name in _ENTRIES:
            if name not in groups:
                pipe = pipelines[name]
                groups[name] = device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
                    {"binding": index, "resource": {"buffer": buffers[buf]}}
                    for index, buf in _BINDINGS[name]])
        xs, ys = -(-w // 8), -(-h // 8)

        def run(names, step=1, flip=0):
            device.queue.write_buffer(entry["uniform"], 0, np.array([w, h, step, flip], "u4").tobytes() + np.array(
                [phi_color, phi_normal, phi_depth, sigma_albedo], "f4").tobytes())
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            for name in names:
                compute.set_pipeline(pipelines[name])
                compute.set_bind_group(0, groups[name])
                compute.dispatch_workgroups(xs, ys, 1)
            compute.end()
            device.queue.submit([encoder.finish()])

        def wait():
            device.queue.read_buffer(buffers["nz"], 0, 4)      # returns when the card has run everything submitted

        run(("prepare", "spread", "blur"))
        flip = 0
        for i in range(int(iterations)):
            if cancel is not None:
                wait()
                if cancel.is_set():
                    raise Cancelled()
            run(("iterate",), step=2 ** i, flip=flip)
            flip ^= 1
        # `iterate` with flip 0 reads `ea` and writes `eb`; after an odd count of them the result is in `eb`, else in `ea`
        run(("finish",), flip=flip)
        data = device.queue.read_buffer(buffers["src"], 0, w * h * 16)
    return np.frombuffer(data, "f4").reshape(h, w, 4).copy()


# the buffers each entry point binds, as (binding index, buffer name); `layout="auto"` keeps only the bindings a shader uses
_BINDINGS = {
    "prepare": ((0, "src"), (1, "alb"), (2, "nrm"), (3, "ea"), (5, "au"), (6, "nz"), (7, "uniform")),
    "spread": ((3, "ea"), (4, "eb"), (7, "uniform")),
    "blur": ((3, "ea"), (4, "eb"), (7, "uniform")),
    "iterate": ((3, "ea"), (4, "eb"), (5, "au"), (6, "nz"), (7, "uniform")),
    "finish": ((0, "src"), (1, "alb"), (3, "ea"), (4, "eb"), (7, "uniform")),
}
