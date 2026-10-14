"""wgpu compute twin of the Undistort / Redistort warps (nodebased/lensmodel.py, `Evaluator._lens_warp`).

One WGSL kernel per output pixel: it maps the pixel centre through the Brown-Conrady model (Undistort: the
forward model; Redistort: the inverse, solved with the same 24 Newton steps and residual check as the CPU) and
reads the source with the node's filter (nearest, bilinear or Catmull-Rom cubic, black outside the source).
The CPU path stays the reference and the fallback; `lens_backend` on the node chooses (auto, cpu, gpu).

Agreement with the CPU (tests/test_gpulens.py): the CPU works in float64 and rounds the sample position to
float32; the shader works in float32 throughout. On smooth content they agree to 1e-4 per channel inside the
frame. Within `EDGE_BAND` pixels of a source edge (and wherever the inverse solve is near singular) the two
may disagree by a whole filter tap, because a position that rounds differently lands on the other side of the
black edge; the tests state that tolerance explicitly.
"""
from __future__ import annotations

import struct

import numpy as np

from . import gpu3d

# Pixels from the edge of the source inside which the GPU and CPU may pick different taps.
EDGE_BAND = 4

# Output pixels below which `auto` keeps the CPU: a dispatch plus two copies costs more than the warp.
AUTO_MIN_PIXELS = 256 * 256

UNIFORM_BYTES = 96       # six rows of four 32-bit values: the lens, the windows, the band

FILTERS = {"nearest": 0, "bilinear": 1, "cubic": 2}

_WGSL = r"""
struct Params {
    a: vec4<f32>,      // k1 k2 k3 p1
    b: vec4<f32>,      // p2 cx cy squeeze
    c: vec4<i32>,      // display width, display height, out.x - display.x, out.y - display.y
    d: vec4<i32>,      // out width, out height, direction (0 undistort, 1 redistort), filter
    e: vec4<i32>,      // source width, source height, display.x - source.x, display.y - source.y
    f: vec4<i32>,      // first output row of this band, rows in this band
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: Params;

fn radial_of(r2: f32) -> f32 {
    return 1.0 + r2 * (p.a.x + r2 * (p.a.y + r2 * p.a.z));
}

// Ideal (film-back) to distorted, with the squeeze and principal point of lensmodel.distort.
fn distort(x: f32, y: f32) -> vec2<f32> {
    let s = p.b.w;
    let dx = x / s - p.b.y;
    let dy = y - p.b.z;
    let r2 = dx * dx + dy * dy;
    let radial = radial_of(r2);
    let xd = dx * radial + 2.0 * p.a.w * dx * dy + p.b.x * (r2 + 2.0 * dx * dx);
    let yd = dy * radial + p.a.w * (r2 + 2.0 * dy * dy) + 2.0 * p.b.x * dx * dy;
    return vec2<f32>((xd + p.b.y) * s, yd + p.b.z);
}

const BAD: f32 = -1.0e6;

// Distorted to ideal by Newton's method; ok is false where the CPU would return NaN.
fn undistort(xd: f32, yd: f32) -> vec3<f32> {
    let s = p.b.w;
    let tx = xd / s - p.b.y;
    let ty = yd - p.b.z;
    var dx = tx;
    var dy = ty;
    var ok = 1.0;
    for (var i = 0; i < 24; i = i + 1) {
        let r2 = dx * dx + dy * dy;
        let radial = radial_of(r2);
        let fx = dx * radial + 2.0 * p.a.w * dx * dy + p.b.x * (r2 + 2.0 * dx * dx) - tx;
        let fy = dy * radial + p.a.w * (r2 + 2.0 * dy * dy) + 2.0 * p.b.x * dx * dy - ty;
        let dradial = 2.0 * (p.a.x + r2 * (2.0 * p.a.y + r2 * 3.0 * p.a.z));
        let ja = radial + dx * dx * dradial + 2.0 * p.a.w * dy + 6.0 * p.b.x * dx;
        let jb = dx * dy * dradial + 2.0 * p.a.w * dx + 2.0 * p.b.x * dy;
        let jd = radial + dy * dy * dradial + 6.0 * p.a.w * dy + 2.0 * p.b.x * dx;
        let det = ja * jd - jb * jb;
        if (abs(det) < 1.0e-12) { ok = 0.0; break; }
        dx = clamp(dx - (jd * fx - jb * fy) / det, -100.0, 100.0);
        dy = clamp(dy - (ja * fy - jb * fx) / det, -100.0, 100.0);
    }
    let x = (dx + p.b.y) * s;
    let y = dy + p.b.z;
    let back = distort(x, y);
    // lensmodel.undistort's residual test is 1e-6 in float64; float32 cannot see below about 3e-7.
    if (!(length(back - vec2<f32>(xd, yd)) < 1.0e-5)) { ok = 0.0; }
    return vec3<f32>(x, y, ok);
}

fn fetch(x: i32, y: i32) -> vec4<f32> {
    if (x < 0 || y < 0 || x >= p.e.x || y >= p.e.y) { return vec4<f32>(0.0); }
    return textureLoad(src, vec2<i32>(x, y), 0);
}

fn cubic_weight(t: f32) -> f32 {
    let a = -0.5;
    let at = abs(t);
    let at2 = at * at;
    let at3 = at * at * at;
    if (at <= 1.0) { return (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0; }
    return a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a;
}

fn sample_src(sx: f32, sy: f32) -> vec4<f32> {
    let mode = p.d.w;
    if (mode == 0) {
        return fetch(i32(floor(sx + 0.5)), i32(floor(sy + 0.5)));
    }
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let fx = sx - f32(x0);
    let fy = sy - f32(y0);
    if (mode == 1) {
        let p00 = fetch(x0, y0);
        let p10 = fetch(x0 + 1, y0);
        let p01 = fetch(x0, y0 + 1);
        let p11 = fetch(x0 + 1, y0 + 1);
        return p00 * (1.0 - fx) * (1.0 - fy) + p10 * fx * (1.0 - fy)
             + p01 * (1.0 - fx) * fy + p11 * fx * fy;
    }
    var total = vec4<f32>(0.0);
    for (var dy = 0; dy < 4; dy = dy + 1) {
        var row = vec4<f32>(0.0);
        for (var dx = 0; dx < 4; dx = dx + 1) {
            row = row + fetch(x0 - 1 + dx, y0 - 1 + dy) * cubic_weight(fx + 1.0 - f32(dx));
        }
        total = total + row * cubic_weight(fy + 1.0 - f32(dy));
    }
    return total;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow = u32(p.d.x);
    let oh = u32(p.d.y);
    let row = i32(gid.y) + p.f.x;
    if (gid.x >= ow || row >= i32(oh) || gid.y >= u32(p.f.y)) { return; }
    let dw = f32(p.c.x);
    let dh = f32(p.c.y);
    let hh = max(dh, 1.0) / 2.0;
    let gx = f32(i32(gid.x) + p.c.z) + 0.5;
    let gy = f32(row + p.c.w) + 0.5;
    let x = (gx - dw / 2.0) / hh;
    let y = (dh / 2.0 - gy) / hh;
    var sx = BAD;
    var sy = BAD;
    if (p.d.z == 0) {
        let q = distort(x, y);
        sx = q.x * hh + dw / 2.0;
        sy = dh / 2.0 - q.y * hh;
    } else {
        let q = undistort(x, y);
        if (q.z > 0.5) {
            sx = q.x * hh + dw / 2.0;
            sy = dh / 2.0 - q.y * hh;
        }
    }
    var colour = vec4<f32>(0.0);
    if (sx > -1.0e5 && sy > -1.0e5 && sx < 1.0e5 && sy < 1.0e5) {
        colour = sample_src(sx + f32(p.e.z) - 0.5, sy + f32(p.e.w) - 0.5);
    }
    dst[gid.y * ow + gid.x] = colour;
}
"""


class Unsupported(Exception):
    """The warp cannot run on this adapter (buffer too large, no wgpu); the caller falls back to the CPU."""


def available(choice=None):
    try:
        gpu3d._state(choice)
        return True
    except Exception:
        return False


def _pipeline(state):
    pipeline = state["pipelines"].get("lens_warp")
    if pipeline is None:
        module = state["device"].create_shader_module(code=_WGSL)
        pipeline = state["device"].create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": "main"})
        state["pipelines"]["lens_warp"] = pipeline
    return pipeline


def warp(lens, pixels, out, display, source_data, direction, filter_name, choice=None):
    """The resampled `out` region (height x width x 4, float32) for an Undistort (`direction` "undistort")
    or Redistort ("redistort"): the GPU twin of `lensmodel.sample_coordinates` + `Evaluator._resample`.

    The source is a float texture (no 128 MiB binding limit, so a 4K frame with overscan fits); the output is
    written in bands of rows that each fit one buffer."""
    if filter_name not in FILTERS:
        raise Unsupported(f"filter {filter_name!r}")
    try:
        state = gpu3d._state(choice)
    except Exception as exc:
        raise Unsupported(str(exc)) from exc
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise Unsupported("expects an HxWx4 image")
    sh, sw = pixels.shape[:2]
    if out.width * out.height == 0 or sw == 0 or sh == 0:
        return np.zeros((max(out.height, 0), max(out.width, 0), 4), np.float32)
    wgpu, device = state["wgpu"], state["device"]
    texture_limit = device.limits.get("max-texture-dimension-2d", 8192)
    if max(sw, sh) > texture_limit:
        raise Unsupported(f"a {sw} x {sh} source exceeds this adapter's texture size {texture_limit}")
    buffer_limit = min(device.limits.get("max-storage-buffer-binding-size", 1 << 27),
                       device.limits.get("max-buffer-size", 1 << 28))
    band = buffer_limit // (out.width * 16)
    if band < 1:
        raise Unsupported(f"a row of {out.width} pixels does not fit one buffer on this adapter")
    band = min(band, out.height, 8192)
    result = np.zeros((out.height, out.width, 4), np.float32)
    pipeline = _pipeline(state)
    texture = device.create_texture(size=(sw, sh, 1), format="rgba32float", dimension="2d",
                                    usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    dst_buf = device.create_buffer(size=out.width * band * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    staging = device.create_buffer(size=out.width * band * 16, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    uniform_buf = device.create_buffer(size=UNIFORM_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    try:
        device.queue.write_texture({"texture": texture}, pixels, {"bytes_per_row": sw * 16, "rows_per_image": sh}, (sw, sh, 1))
        group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": texture.create_view()},
            {"binding": 1, "resource": {"buffer": dst_buf, "offset": 0, "size": dst_buf.size}},
            {"binding": 2, "resource": {"buffer": uniform_buf, "offset": 0, "size": UNIFORM_BYTES}}])
        for row0 in range(0, out.height, band):
            rows = min(band, out.height - row0)
            uniform = struct.pack("<8f" "4i" "4i" "4i" "4i", lens.k1, lens.k2, lens.k3, lens.p1, lens.p2, lens.cx,
                                  lens.cy, lens.squeeze, display.width, display.height, out.x - display.x,
                                  out.y - display.y, out.width, out.height, 0 if direction == "undistort" else 1,
                                  FILTERS[filter_name], sw, sh, display.x - source_data.x, display.y - source_data.y,
                                  row0, rows, 0, 0)
            device.queue.write_buffer(uniform_buf, 0, uniform)
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(-(-out.width // 8), -(-rows // 8), 1)
            compute.end()
            encoder.copy_buffer_to_buffer(dst_buf, 0, staging, 0, out.width * rows * 16)
            device.queue.submit([encoder.finish()])
            staging.map_sync(wgpu.MapMode.READ)
            try:
                result[row0:row0 + rows] = np.frombuffer(staging.read_mapped(0, out.width * rows * 16),
                                                         np.float32).reshape(rows, out.width, 4)
            finally:
                staging.unmap()
    finally:
        for resource in (texture, dst_buf, staging, uniform_buf):
            resource.destroy()
    return result
