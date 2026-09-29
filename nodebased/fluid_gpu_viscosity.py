"""Weighted implicit MAC-face viscosity on a wgpu compute adapter.

The CPU reference in flip3d.Liquid3D._viscosity supplies the same face coefficient.
Red/black SOR computes a correction in float32; CPU residual refinement preserves the
reference stopping target without relying on a device-dependent reduction order.
"""
from __future__ import annotations

import math
import numpy as np

from .cancellation import Cancelled

WGSL = """
struct Params { nx: u32, ny: u32, nz: u32, color: u32,
                k: f32, omega: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> rhs: array<f32>;
@group(0) @binding(3) var<storage, read> mu: array<f32>;
@compute @workgroup_size(8, 8, 4)
fn sweep(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y; let z = id.z;
    if (x >= p.nx || y >= p.ny || z >= p.nz || ((x + y + z) & 1u) != p.color) { return; }
    let i = (x * p.ny + y) * p.nz + z;
    var degree = 0.0;
    var neighbors = 0.0;
    if (x > 0u) { let j = i - p.ny * p.nz; let w = 0.5 * (mu[i] + mu[j]); degree += w; neighbors += w * q[j]; }
    if (x + 1u < p.nx) { let j = i + p.ny * p.nz; let w = 0.5 * (mu[i] + mu[j]); degree += w; neighbors += w * q[j]; }
    if (y > 0u) { let j = i - p.nz; let w = 0.5 * (mu[i] + mu[j]); degree += w; neighbors += w * q[j]; }
    if (y + 1u < p.ny) { let j = i + p.nz; let w = 0.5 * (mu[i] + mu[j]); degree += w; neighbors += w * q[j]; }
    if (z > 0u) { let j = i - 1u; let w = 0.5 * (mu[i] + mu[j]); degree += w; neighbors += w * q[j]; }
    if (z + 1u < p.nz) { let j = i + 1u; let w = 0.5 * (mu[i] + mu[j]); degree += w; neighbors += w * q[j]; }
    let updated = (rhs[i] + p.k * neighbors) / (1.0 + p.k * degree);
    q[i] += p.omega * (updated - q[i]);
}
"""


class GpuViscosity3D:
    def __init__(self, batch=24, max_sweeps=1200):
        import wgpu
        self.wgpu = wgpu
        from . import gpu3d
        state = gpu3d._state()
        self.adapter_name = str(state["info"].get("device", "?"))
        self.device = state["device"]
        shader = self.device.create_shader_module(code=WGSL)
        self.pipeline = self.device.create_compute_pipeline(layout="auto", compute={"module": shader, "entry_point": "sweep"})
        self.batch = batch
        self.max_sweeps = max_sweeps
        self._shape = None

    def _allocate(self, shape, k):
        usage = self.wgpu.BufferUsage
        size = int(np.prod(shape)) * 4
        self.q = self.device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST | usage.COPY_SRC)
        self.rhs = self.device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST)
        self.mu = self.device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST)
        nx, ny, nz = shape
        omega = min(1.8, 2.0 / (1.0 + math.sin(math.pi / max(shape))))
        self.uniforms = [self.device.create_buffer_with_data(
            data=np.array([nx, ny, nz, color], 'u4').tobytes() + np.array([k, omega, 0, 0], 'f4').tobytes(),
            usage=usage.UNIFORM | usage.COPY_DST) for color in (0, 1)]
        layout = self.pipeline.get_bind_group_layout(0)
        self.groups = [self.device.create_bind_group(layout=layout, entries=[
            {"binding": 0, "resource": {"buffer": uniform}},
            {"binding": 1, "resource": {"buffer": self.q}},
            {"binding": 2, "resource": {"buffer": self.rhs}},
            {"binding": 3, "resource": {"buffer": self.mu}}]) for uniform in self.uniforms]
        self._shape = shape

    @staticmethod
    def _apply(x, mu, k):
        out = x.copy()
        for axis in range(3):
            lo = [slice(None)] * 3; hi = lo.copy()
            lo[axis] = slice(None, -1); hi[axis] = slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            flux = k * 0.5 * (mu[lo] + mu[hi]) * (x[lo] - x[hi])
            out[lo] += flux
            out[hi] -= flux
        return out

    def solve(self, rhs, k, coefficient=None, cancel=None):
        rhs = np.asarray(rhs, np.float64)
        if k <= 0.0:
            return rhs.copy()
        mu = np.ones(rhs.shape, np.float64) if coefficient is None else np.asarray(coefficient, np.float64)
        if self._shape != rhs.shape:
            self._allocate(rhs.shape, k)
        for color, uniform in enumerate(self.uniforms):
            data = np.array([*rhs.shape, color], 'u4').tobytes() + np.array([k, min(1.8, 2.0 / (1.0 + math.sin(math.pi / max(rhs.shape)))), 0, 0], 'f4').tobytes()
            self.device.queue.write_buffer(uniform, 0, data)
        self.device.queue.write_buffer(self.mu, 0, np.ascontiguousarray(mu, 'f4'))
        x = rhs.copy()
        residual = rhs - self._apply(x, mu, k)
        target = max(1e-12, float(np.linalg.norm(rhs)) * 1e-8)
        groups = tuple((n + block - 1) // block for n, block in zip(rhs.shape, (8, 8, 4)))
        swept = 0
        zeros = np.zeros(rhs.shape, 'f4')
        while float(np.linalg.norm(residual)) > target and swept < self.max_sweeps:
            initial = float(np.linalg.norm(residual))
            goal = max(target, initial / 50.0)
            self.device.queue.write_buffer(self.q, 0, zeros)
            self.device.queue.write_buffer(self.rhs, 0, np.ascontiguousarray(residual, 'f4'))
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                encoder = self.device.create_command_encoder()
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(self.pipeline)
                for _ in range(self.batch):
                    for group in self.groups:
                        compute.set_bind_group(0, group)
                        compute.dispatch_workgroups(*groups)
                compute.end()
                self.device.queue.submit([encoder.finish()])
                swept += self.batch
                correction = np.frombuffer(self.device.queue.read_buffer(self.q), 'f4').reshape(rhs.shape).astype(np.float64)
                new_residual = residual - self._apply(correction, mu, k)
                if float(np.linalg.norm(new_residual)) <= goal or swept >= self.max_sweeps:
                    break
            x += correction
            residual = new_residual
        return x
