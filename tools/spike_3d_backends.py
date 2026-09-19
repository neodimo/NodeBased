#!/usr/bin/env python3
"""Standalone, deliberately brute-force 3D backend spike; never imported by NodeBased.

Times include both float RGBA and view-depth NumPy readbacks. CPU does two renders
because its correctness baseline explicitly uses output='depth'; GPU uses MRT or
paired storage buffers. Scene preparation is excluded; backend import, resource
upload and pipeline creation are included in init. No warmup precedes first frame.
Default compute workloads are intentionally expensive (pixels * triangles).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodebased import scene3d

W, H = 960, 540
COLOR = (0.3, 0.65, 0.9, 1.0)


def make_scene(name):
    camera = scene3d.Camera()
    if name == 'sphere':
        geo = scene3d._sphere(1.0, 32, COLOR, scene3d.Transform3D())
    else:
        nx, ny = (100, 100) if name == '20k' else (250, 200)
        rng = np.random.default_rng(20260919)
        x, y = np.meshgrid(np.linspace(-2.8, 2.8, nx + 1),
                           np.linspace(-1.5, 1.5, ny + 1))
        x[1:-1, 1:-1] += rng.uniform(-0.2, 0.2, (ny-1, nx-1)) * 5.6/nx
        y[1:-1, 1:-1] += rng.uniform(-0.2, 0.2, (ny-1, nx-1)) * 3/ny
        z = 0.2 * np.sin(x * 3) * np.cos(y * 4)
        vertices = np.stack((x, y, z), -1).reshape(-1, 3).astype('f4')
        a = (np.arange(ny)[:, None] * (nx+1) + np.arange(nx)).ravel()
        triangles = np.stack((np.stack((a, a+1, a+nx+2), -1),
                              np.stack((a, a+nx+2, a+nx+1), -1)), 1).reshape(-1, 3)
        geo = scene3d.Geometry(vertices, triangles.astype('i4'), COLOR)
    scene = scene3d.Scene((geo,))
    eye, view = scene3d._view_basis(camera)
    matrix = geo.world_matrix()
    world = (matrix[:3, :3] @ geo.vertices.T + matrix[:3, 3:4]).T
    local = (view @ (world - eye).T).T
    triangles = np.ascontiguousarray(local[geo.triangles], dtype='f4')
    return scene, camera, triangles


def measure(start, frame, frames, info):
    result = dict(status='ok', init_ms=(time.perf_counter()-start)*1000, info=info)
    t = time.perf_counter()
    rgba, depth = frame()
    result['first_ms'] = (time.perf_counter()-t)*1000
    timings = []
    for _ in range(frames):
        t = time.perf_counter()
        rgba, depth = frame()
        timings.append((time.perf_counter()-t)*1000)
    result.update(median_ms=float(np.median(timings)), min_ms=min(timings), frames_ms=timings)
    if rgba.shape != (H, W, 4) or depth.shape != (H, W):
        raise ValueError('unexpected readback shape')
    if not np.isfinite(rgba).all() or not np.isfinite(depth[rgba[..., 3] > 0]).all():
        raise ValueError('non-finite covered output')
    return result, rgba, depth


def cpu_backend(scene, camera, triangles, frames, adapter):
    start = time.perf_counter()
    def frame():
        rgba = scene3d.render(scene, camera, W, H)
        depth = scene3d.render(scene, camera, W, H, output='depth')[..., 0].copy()
        return rgba, depth
    return measure(start, frame, frames, {'renderer': 'nodebased.scene3d.render'})


GL_VERTEX = '''#version 330
in vec3 position;
uniform vec4 projection; // focal/aspect, focal, near, far
out float view_depth;
void main() {
    float n = projection.z, f = projection.w;
    gl_Position = vec4(position.x*projection.x, position.y*projection.y,
        -(f+n)/(f-n)*position.z - 2.0*f*n/(f-n), -position.z);
    view_depth = -position.z;
}
'''
GL_FRAGMENT = '''#version 330
in float view_depth;
layout(location=0) out vec4 colour;
layout(location=1) out float depth;
void main() { colour=vec4(0.3,0.65,0.9,1.0); depth=view_depth; }
'''


def moderngl_backend(scene, camera, triangles, frames, adapter):
    start = time.perf_counter()
    import moderngl
    errors = []
    ctx = None
    for options in ([{'backend': 'egl'}, {}] if sys.platform.startswith('linux') else [{}]):
        try:
            ctx = moderngl.create_standalone_context(require=330, **options)
            break
        except Exception as exc:
            errors.append(f'{options or "default"}: {exc}')
    if ctx is None:
        raise RuntimeError('; '.join(errors))
    try:
        program = ctx.program(vertex_shader=GL_VERTEX, fragment_shader=GL_FRAGMENT)
        focal = 1/math.tan(math.radians(camera.fov)/2)
        program['projection'].value = (focal/(W/H), focal, camera.near, camera.far)
        buffer = ctx.buffer(triangles.tobytes())
        vao = ctx.simple_vertex_array(program, buffer, 'position')
        colour = ctx.texture((W, H), 4, dtype='f4')
        linear = ctx.texture((W, H), 1, dtype='f4')
        z = ctx.depth_renderbuffer((W, H))
        fb = ctx.framebuffer([colour, linear], z)
        fb.use()
        ctx.viewport = (0, 0, W, H)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = '<'
        def frame():
            fb.clear(0, 0, 0, 0, depth=1)
            vao.render(mode=moderngl.TRIANGLES)
            rgba = np.frombuffer(colour.read(alignment=1), 'f4').reshape(H, W, 4)[::-1].copy()
            depth = np.frombuffer(linear.read(alignment=1), 'f4').reshape(H, W)[::-1].copy()
            return rgba, depth
        return measure(start, frame, frames, {'renderer': ctx.info['GL_RENDERER'],
                                             'context_attempt_errors': errors})
    finally:
        ctx.release()


WGSL_RASTER = '''
struct Params { projection: vec4<f32>, size: vec4<u32> };
@group(0) @binding(0) var<uniform> params: Params;
struct Vertex { @builtin(position) position: vec4<f32>, @location(0) depth: f32 };
@vertex fn vs(@location(0) p: vec3<f32>) -> Vertex {
    let q = params.projection;
    var out: Vertex;
    out.position = vec4<f32>(p.x*q.x, p.y*q.y,
        -q.w/(q.w-q.z)*p.z - q.w*q.z/(q.w-q.z), -p.z);
    out.depth = -p.z;
    return out;
}
struct Fragment { @location(0) colour: vec4<f32>, @location(1) depth: f32 };
@fragment fn fs(v: Vertex) -> Fragment {
    var out: Fragment;
    out.colour = vec4<f32>(0.3, 0.65, 0.9, 1.0);
    out.depth = v.depth;
    return out;
}
'''
WGSL_COMPUTE = '''
struct Params { projection: vec4<f32>, size: vec4<u32> };
struct Triangle { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read_write> colours: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> depths: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let pixel = gid.y*params.size.x + gid.x;
    let uv = (vec2<f32>(gid.xy)+vec2<f32>(0.5))/vec2<f32>(params.size.xy);
    // Deliberately unnormalized: t is -view.z, not Euclidean ray distance.
    let ray = vec3<f32>((2.0*uv.x-1.0)/params.projection.x,
                       (1.0-2.0*uv.y)/params.projection.y, -1.0);
    var nearest = params.projection.w;
    var hit = false;
    for (var i = 0u; i < params.size.z; i += 1u) {
        let tri = triangles[i];
        let e1 = tri.b.xyz-tri.a.xyz;
        let e2 = tri.c.xyz-tri.a.xyz;
        let p = cross(ray, e2);
        let det = dot(e1, p);
        if (abs(det) < 1e-10) { continue; }
        let inverse = 1.0/det;
        let s = -tri.a.xyz;
        let u = dot(s, p)*inverse;
        if (u < 0.0 || u > 1.0) { continue; }
        let q = cross(s, e1);
        let v = dot(ray, q)*inverse;
        if (v < 0.0 || u+v > 1.0) { continue; }
        let t = dot(e2, q)*inverse;
        if (t > params.projection.z && t < nearest) { nearest=t; hit=true; }
    }
    colours[pixel] = vec4<f32>(0.0);
    depths[pixel] = 0.0;
    if (hit) {
        colours[pixel] = vec4<f32>(0.3,0.65,0.9,1.0);
        depths[pixel] = nearest;
    }
}
'''


def wgpu_device(choice):
    import wgpu
    if choice == 'default':
        adapter = wgpu.gpu.request_adapter_sync()
    else:
        adapters = wgpu.gpu.enumerate_adapters_sync()
        target = {'discrete': 'discretegpu', 'integrated': 'integratedgpu', 'cpu': 'cpu'}[choice]
        adapter = next((a for a in adapters if
                        str(a.info.get('adapter_type', '')).lower().replace('_', '').replace(' ', '') == target), None)
        if adapter is None:
            raise RuntimeError(f'no {choice} adapter; available: {[dict(a.info) for a in adapters]}')
    if adapter is None:
        raise RuntimeError('no wgpu adapter')
    return wgpu, adapter.request_device_sync(), dict(adapter.info)


def uniform(device, wgpu, camera, triangles):
    focal = 1/math.tan(math.radians(camera.fov)/2)
    data = np.array([focal/(W/H), focal, camera.near, camera.far], 'f4').tobytes()
    data += np.array([W, H, len(triangles), 0], 'u4').tobytes()
    return device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM)


def wgpu_raster_backend(scene, camera, triangles, frames, adapter):
    start = time.perf_counter()
    wgpu, device, info = wgpu_device(adapter)
    try:
        # rgba16float is universally renderable; depth remains full precision r32float.
        fmt = 'rgba16float'
        module = device.create_shader_module(code=WGSL_RASTER)
        pipeline = device.create_render_pipeline(layout='auto',
            vertex={'module': module, 'entry_point': 'vs', 'buffers': [
                {'array_stride': 12, 'step_mode': 'vertex', 'attributes': [
                    {'format': 'float32x3', 'offset': 0, 'shader_location': 0}]}]},
            primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
            depth_stencil={'format': 'depth32float', 'depth_write_enabled': True, 'depth_compare': 'less'},
            fragment={'module': module, 'entry_point': 'fs', 'targets': [{'format': fmt}, {'format': 'r32float'}]})
        params = uniform(device, wgpu, camera, triangles)
        group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
            {'binding': 0, 'resource': {'buffer': params}}])
        vertices = device.create_buffer_with_data(data=triangles, usage=wgpu.BufferUsage.VERTEX)
        textures = [device.create_texture(size=(W, H, 1), format=f,
                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
                    for f in (fmt, 'r32float')]
        z = device.create_texture(size=(W, H, 1), format='depth32float', usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        views = [t.create_view() for t in textures]
        zview = z.create_view()
        strides = [((W*b+255)//256)*256 for b in (8, 4)]
        staging = [device.create_buffer(size=s*H, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ) for s in strides]
        def frame():
            encoder = device.create_command_encoder()
            render = encoder.begin_render_pass(color_attachments=[
                {'view': v, 'resolve_target': None, 'clear_value': (0,0,0,0), 'load_op': 'clear', 'store_op': 'store'} for v in views],
                depth_stencil_attachment={'view': zview, 'depth_clear_value': 1.0,
                                          'depth_load_op': 'clear', 'depth_store_op': 'store'})
            render.set_pipeline(pipeline)
            render.set_bind_group(0, group)
            render.set_vertex_buffer(0, vertices)
            render.draw(len(triangles)*3)
            render.end()
            for texture, buffer, stride in zip(textures, staging, strides):
                encoder.copy_texture_to_buffer({'texture': texture},
                    {'buffer': buffer, 'bytes_per_row': stride, 'rows_per_image': H}, (W, H, 1))
            device.queue.submit([encoder.finish()])
            arrays = []
            for buffer, stride, dtype, channels in zip(staging, strides, ('f2', 'f4'), (4, 1)):
                buffer.map_sync(wgpu.MapMode.READ)
                try:
                    raw = np.frombuffer(buffer.read_mapped(), dtype).reshape(H, stride//np.dtype(dtype).itemsize)
                    arrays.append(raw[:, :W*channels].reshape(H, W, channels).astype('f4', copy=True))
                finally:
                    buffer.unmap()
            return arrays[0], arrays[1][..., 0]
        return measure(start, frame, frames, {**info, 'colour_format': fmt, 'depth_output_format': 'r32float'})
    finally:
        device.destroy()


def wgpu_compute_backend(scene, camera, triangles, frames, adapter):
    start = time.perf_counter()
    wgpu, device, info = wgpu_device(adapter)
    try:
        module = device.create_shader_module(code=WGSL_COMPUTE)
        pipeline = device.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'main'})
        packed = np.zeros((len(triangles), 3, 4), 'f4')
        packed[..., :3] = triangles
        params = uniform(device, wgpu, camera, triangles)
        geometry = device.create_buffer_with_data(data=packed, usage=wgpu.BufferUsage.STORAGE)
        outputs = [device.create_buffer(size=W*H*b, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC) for b in (16, 4)]
        staging = [device.create_buffer(size=W*H*b, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ) for b in (16, 4)]
        group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
            {'binding': i, 'resource': {'buffer': buffer}} for i, buffer in enumerate([params, geometry, *outputs])])
        def frame():
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups((W+7)//8, (H+7)//8)
            compute.end()
            for source, target, size in zip(outputs, staging, (W*H*16, W*H*4)):
                encoder.copy_buffer_to_buffer(source, 0, target, 0, size)
            device.queue.submit([encoder.finish()])
            arrays = []
            for buffer, channels in zip(staging, (4, 1)):
                buffer.map_sync(wgpu.MapMode.READ)
                try:
                    arrays.append(np.frombuffer(buffer.read_mapped(), 'f4').reshape(H, W, channels).copy())
                finally:
                    buffer.unmap()
            return arrays[0], arrays[1][..., 0]
        return measure(start, frame, frames, info)
    finally:
        device.destroy()


def attempt(backend, *args):
    try:
        return backend(*args)
    except Exception as exc:
        return {'status': f'unavailable: {type(exc).__name__}: {exc}'}, None, None


def cpu_worker(connection, args):
    try:
        connection.send(attempt(cpu_backend, *args))
    finally:
        connection.close()


def budgeted_cpu(args):
    context = mp.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=cpu_worker, args=(writer, args))
    process.start()
    writer.close()
    try:
        if reader.poll(60):
            try:
                return reader.recv()
            except EOFError:
                return {'status': 'unavailable: CPU worker exited without a result'}, None, None
        return {'status': 'skipped: 100k CPU exceeded 60 s total budget (including depth baseline and timed frames)'}, None, None
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        reader.close()


def correctness(rgba, depth, reference):
    if reference[0] is None:
        return {'status': 'not checked: CPU baseline unavailable or budget exceeded'}
    expected, expected_depth = reference
    a, b = rgba[..., 3] > 0, expected[..., 3] > 0
    union, common = a | b, a & b
    delta = np.abs(depth[common]-expected_depth[common])
    return {'status': 'measured', 'coverage_iou': float((a & b).sum()/union.sum()) if union.any() else 1.0,
            'covered_pixels': int(a.sum()), 'common_pixels': int(common.sum()),
            'depth_max_abs': float(delta.max()) if delta.size else None,
            'depth_median_abs': float(np.median(delta)) if delta.size else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['all', 'sphere', '20k', '100k'], default='all')
    parser.add_argument('--frames', type=int, default=15, help='timed frames after first frame (default: 15)')
    parser.add_argument('--adapter', choices=['default', 'discrete', 'integrated', 'cpu'], default='default')
    parser.add_argument('--json', type=Path, help='write full measurements and adapter information')
    parser.add_argument('--backend', choices=['all', 'cpu', 'moderngl', 'wgpu-raster', 'wgpu-compute'], default='all',
                        help='CPU baseline always runs; useful for a lightweight CPU-only smoke run')
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    report = {'width': W, 'height': H, 'frames': args.frames, 'adapter_requested': args.adapter,
              'timing': 'wall time including colour and view-depth NumPy readback; CPU uses two renders',
              'sphere_note': 'scene3d sphere(segments=32) currently has 1024 triangles, including pole degenerates',
              'results': []}
    print('960x540; unlit opaque RGBA + view-space depth; first frame separate from timed frames.')
    print('CPU uses separate rgba/depth renders; GPU returns both in one pass. Init includes uploads.')
    print('Depth errors use intersection of coverage; raster edge rules and float precision may differ.')
    print('\n| Scene | Tris | Backend | Init ms | First ms | Median ms | Min ms | IoU | Depth max / median | Status |')
    print('|---|---:|---|---:|---:|---:|---:|---:|---:|---|', flush=True)
    backends = [('cpu', cpu_backend), ('moderngl', moderngl_backend),
                ('wgpu-raster', wgpu_raster_backend), ('wgpu-compute', wgpu_compute_backend)]
    for name in (['sphere', '20k', '100k'] if args.scene == 'all' else [args.scene]):
        scene, camera, triangles = make_scene(name)
        inputs = (scene, camera, triangles, args.frames, args.adapter)
        reference = (None, None)
        for label, backend in backends:
            if label != 'cpu' and args.backend not in ('all', label):
                continue
            print(f'Running {name}/{label} ...', file=sys.stderr, flush=True)
            result, rgba, depth = (budgeted_cpu(inputs) if name == '100k' and label == 'cpu'
                                   else attempt(backend, *inputs))
            if label == 'cpu':
                reference = rgba, depth
            check = correctness(rgba, depth, reference) if rgba is not None else {'status': 'not checked'}
            result.update(scene=name, triangles=len(triangles), backend=label, correctness=check)
            report['results'].append(result)
            def number(value):
                return '—' if value is None else f'{value:.6g}'
            fields = [name, str(len(triangles)), label,
                      *(number(result.get(k)) for k in ('init_ms', 'first_ms', 'median_ms', 'min_ms')),
                      number(check.get('coverage_iou')),
                      f"{number(check.get('depth_max_abs'))} / {number(check.get('depth_median_abs'))}", result['status']]
            print('| ' + ' | '.join(str(f).replace('|', '/').replace('\n', ' ') for f in fields) + ' |', flush=True)
    print('\nRenderer / adapter details (name, backend and driver where exposed):')
    for result in report['results']:
        if 'info' in result:
            print(f"- {result['scene']}/{result['backend']}: {json.dumps(result['info'], default=str)}")
        if result['status'] == 'ok' and result['correctness']['status'] != 'measured':
            print(f"- {result['scene']}/{result['backend']}: {result['correctness']['status']}")
    print('\nSphere uses the actual 32-segment scene3d primitive: 1,024 triangles (including pole degenerates).')
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
