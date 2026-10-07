"""GPU pair-potential evaluation for Ihmsen whitewater emission.

Spatial bins and their deterministic CSR neighbor list are assembled on the host;
all three potential sums run in one compute dispatch. Neighbor order matches the
CPU reference so seeded emission and cap ordering remain stable.
"""
from __future__ import annotations

import numpy as np

WGSL = """
struct Params { n: u32, base: u32, h: f32, mass: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> normal: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read> neighbors: array<u32>;
@group(0) @binding(6) var<storage, read_write> result: array<vec4<f32>>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let local = id.x;
    if (local >= p.n) { return; }
    let i = p.base + local;
    let vi = vel[i].xyz;
    let speed = length(vi);
    let ni = normalize(normal[i].xyz);
    let normal_speed = dot(vi / max(speed, 1e-12), ni);
    var trapped = 0.0;
    var crest = 0.0;
    for (var index = offsets[local]; index < offsets[local + 1u]; index++) {
        let j = neighbors[index];
        if (i == j) { continue; }
        let displacement = pos[i].xyz - pos[j].xyz;
        let distance = length(displacement);
        if (distance <= 1e-12 || distance > p.h) { continue; }
        let direction = displacement / distance;
        let relative = vi - vel[j].xyz;
        let relative_speed = length(relative);
        let approaching = 1.0 - dot(relative, direction) / max(relative_speed, 1e-12);
        let weight = 1.0 - distance / p.h;
        trapped += relative_speed * approaching * weight;
        if (normal_speed >= 0.6 && dot(-direction, ni) < 0.0) {
            crest += (1.0 - clamp(dot(normalize(normal[j].xyz), ni), -1.0, 1.0)) * weight;
        }
    }
    result[local] = vec4<f32>(trapped, crest * normal_speed, 0.5 * p.mass * speed * speed, 0.0);
}
"""

MOTION_WGSL = """
struct Motion { n: u32, pad0: u32, dt: f32, gravity: f32,
                spray_drag: f32, bubble_drag: f32, buoyancy: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> p: Motion;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> kind: array<u32>;
@group(0) @binding(4) var<storage, read> nearest: array<u32>;
@group(0) @binding(5) var<storage, read> liquid_vel: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> out_pos: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_vel: array<vec4<f32>>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.n) { return; }
    var v = vel[i].xyz;
    if (kind[i] == 1u) {
        v.y -= p.gravity * p.dt;
        v *= max(0.0, 1.0 - p.spray_drag * p.dt);
    } else if (kind[i] == 2u) {
        v += (liquid_vel[nearest[i]].xyz - v) * min(1.0, p.bubble_drag * p.dt);
        v.y += p.buoyancy * p.dt;
    } else {
        v = liquid_vel[nearest[i]].xyz;
    }
    out_pos[i] = vec4<f32>(pos[i].xyz + v * p.dt, 0.0);
    out_vel[i] = vec4<f32>(v, 0.0);
}
"""


def neighbor_lists(positions, h, chunk=20000):
    """The CSR neighbour list of every particle: all particles in the 27 bins of side `h` around its own, as (starts (n + 1,)
    int64, members uint32). The order is fixed: bins in (dx, dy, dz) order, ascending particle index inside a bin, which
    is the order the shader sums them in, so the potentials are the same bits however the list was built. Built with array
    operations (the loop over particles and bins that this replaced took seconds a frame at 100,000 particles)."""
    positions = np.asarray(positions, np.float64)
    n = len(positions)
    bins = np.floor(positions / h).astype(np.int64)
    low = bins.min(axis=0) - 1
    extent = bins.max(axis=0) - low + 2
    shifted = bins - low
    key = (shifted[:, 0] * extent[1] + shifted[:, 1]) * extent[2] + shifted[:, 2]
    order = np.argsort(key, kind="stable")                      # ascending index inside a bin
    sorted_key = key[order]
    unique, first, count = np.unique(sorted_key, return_index=True, return_counts=True)
    offsets = np.array([(dx * extent[1] + dy) * extent[2] + dz
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], np.int64)
    counts = np.zeros((n, 27), np.int64)
    firsts = np.zeros((n, 27), np.int64)
    for o, offset in enumerate(offsets):
        wanted = key + offset
        where = np.minimum(np.searchsorted(unique, wanted), len(unique) - 1)
        hit = unique[where] == wanted
        counts[:, o] = np.where(hit, count[where], 0)
        firsts[:, o] = np.where(hit, first[where], 0)
    per_particle = counts.sum(axis=1)
    starts = np.zeros(n + 1, np.int64)
    starts[1:] = np.cumsum(per_particle)
    members = np.zeros(max(int(per_particle.sum()), 1), np.uint32)
    written = 0
    for lo in range(0, n, chunk):
        c = counts[lo:lo + chunk].reshape(-1)
        f = firsts[lo:lo + chunk].reshape(-1)
        total = int(c.sum())
        if not total:
            continue
        run_start = np.cumsum(c) - c
        position = np.repeat(f - run_start, c) + np.arange(total)
        members[written:written + total] = order[position]
        written += total
    return starts, members


class GpuWhitewater3D:
    def __init__(self):
        from . import gpu3d
        state = gpu3d._state()
        self.device = state['device']
        self.wgpu = state['wgpu']
        self.adapter_name = str(state['info'].get('device', '?'))
        module = self.device.create_shader_module(code=WGSL)
        self.pipeline = self.device.create_compute_pipeline(layout='auto', compute={'module': module, 'entry_point': 'main'})
        motion_module = self.device.create_shader_module(code=MOTION_WGSL)
        self.motion_pipeline = self.device.create_compute_pipeline(layout='auto', compute={'module': motion_module, 'entry_point': 'main'})

    def move(self, positions, velocities, kinds, nearest, liquid_velocities, dt, params):
        n = len(positions)
        if not n:
            return np.asarray(positions, np.float64).copy(), np.asarray(velocities, np.float64).copy()
        def vec4(array):
            out = np.zeros((len(array), 4), np.float32)
            out[:, :3] = array
            return out
        usage = self.wgpu.BufferUsage
        def input_buffer(array):
            return self.device.create_buffer_with_data(data=np.ascontiguousarray(array).tobytes(), usage=usage.STORAGE)
        uniforms = np.array([n, 0], 'u4').tobytes() + np.array([
            dt, params['gravity'], params['spray_drag'], params['bubble_drag'], params['bubble_buoyancy'], 0], 'f4').tobytes()
        buffers = [self.device.create_buffer_with_data(data=uniforms, usage=usage.UNIFORM),
                   input_buffer(vec4(positions)), input_buffer(vec4(velocities)),
                   input_buffer(np.asarray(kinds, np.uint32)), input_buffer(np.asarray(nearest, np.uint32)),
                   input_buffer(vec4(liquid_velocities))]
        buffers.extend(self.device.create_buffer(size=n * 16, usage=usage.STORAGE | usage.COPY_SRC) for _ in range(2))
        layout = self.motion_pipeline.get_bind_group_layout(0)
        group = self.device.create_bind_group(layout=layout, entries=[
            {'binding': i, 'resource': {'buffer': b}} for i, b in enumerate(buffers)])
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.motion_pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups((n + 127) // 128)
        compute.end()
        self.device.queue.submit([encoder.finish()])
        return tuple(np.frombuffer(self.device.queue.read_buffer(b), 'f4').reshape(n, 4)[:, :3].astype(np.float64)
                     for b in buffers[-2:])

    def potentials(self, positions, velocities, normals, support, mass=1.0):
        positions = np.asarray(positions, np.float64)
        n = len(positions)
        if n == 0:
            empty = np.zeros(0, np.float64)
            return empty, empty, empty
        h = max(float(support), 1e-8)
        starts, neighbors = neighbor_lists(positions, h)
        limit = int(getattr(self, "binding_limit", 0) or self.device.limits['max-storage-buffer-binding-size'])
        capacity = max(limit // 4, 1)                              # list entries one storage binding holds
        widest = int(np.diff(starts).max())
        if widest > capacity:
            raise ValueError(f'whitewater neighbor list for one particle needs {widest * 4} bytes; adapter supports {limit}')
        def vec4(array):
            out = np.zeros((n, 4), np.float32)
            out[:, :3] = array
            return out
        normal = np.asarray(normals, np.float64)
        normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        usage = self.wgpu.BufferUsage
        def buffer(array, output=False):
            if output:
                return self.device.create_buffer(size=array.nbytes, usage=usage.STORAGE | usage.COPY_SRC)
            return self.device.create_buffer_with_data(data=np.ascontiguousarray(array).tobytes(), usage=usage.STORAGE)
        particle_buffers = [buffer(vec4(positions)), buffer(vec4(velocities)), buffer(vec4(normal))]
        layout = self.pipeline.get_bind_group_layout(0)
        values = np.zeros((n, 4), np.float32)
        low = 0
        while low < n:
            # the largest slice of particles [low, high) whose list fits one binding: the 2.2 GB list of a 256-cell pour
            # went over the adapter's 2 GiB and ended the bake
            high = int(np.searchsorted(starts, starts[low] + capacity, side="right")) - 1
            high = min(max(high, low + 1), n)
            local_starts = (starts[low:high + 1] - starts[low]).astype(np.uint32)
            members = neighbors[int(starts[low]):int(starts[high])]
            payload = np.array([high - low, low], 'u4').tobytes() + np.array([h, mass], 'f4').tobytes()
            params = self.device.create_buffer_with_data(data=payload, usage=usage.UNIFORM)
            out = buffer(np.empty((high - low, 4), np.float32), output=True)
            group = self.device.create_bind_group(layout=layout, entries=[
                {'binding': k, 'resource': {'buffer': b}} for k, b in enumerate(
                    [params, *particle_buffers, buffer(local_starts), buffer(members if len(members) else np.zeros(1, np.uint32)),
                     out])])
            encoder = self.device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(self.pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups((high - low + 127) // 128)
            compute.end()
            self.device.queue.submit([encoder.finish()])
            values[low:high] = np.frombuffer(self.device.queue.read_buffer(out), 'f4').reshape(high - low, 4)
            low = high
        return tuple(values[:, i].astype(np.float64) for i in range(3))
