"""wgpu compute variant of the 3D pressure solve (Lane 6 step C). The NumPy conjugate gradient in
nodebased/fluid3d.py is the reference; this plugs into `Smoke3D(pressure_solver=GpuPressure3D().solve)`.

Red-black Gauss-Seidel with over-relaxation (SOR, omega = 2 / (1 + sin(pi / max(nx, ny, nz)))) on the same masked
7-point system as `fluid3d.Poisson3D`: solid cells and closed walls drop out through a per-cell neighbour bit
mask, open faces raise the per-cell diagonal. float32 on the GPU, in the iterative-refinement form of
tools/fluid_gpu.py: the GPU solves A e = r for the float64 residual r held on the CPU until it has dropped
50-fold, then q += e, so float32 only ever holds a correction and the stopping rule (largest cell residual at
most `tolerance`, measured on the CPU with the reference operator) is the same as the NumPy solve's.
"""
from __future__ import annotations

import numpy as np

from .cancellation import Cancelled

WGSL = """
struct P { nx: u32, ny: u32, nz: u32, color: u32, omega: f32, p1: f32, p2: f32, p3: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> rhs: array<f32>;
@group(0) @binding(3) var<storage, read> diag: array<f32>;
@group(0) @binding(4) var<storage, read> bits: array<u32>;

@compute @workgroup_size(8, 8, 4)
fn sweep(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y; let z = id.z;
    if (x >= p.nx || y >= p.ny || z >= p.nz) { return; }
    if (((x + y + z) & 1u) != p.color) { return; }
    let i = (x * p.ny + y) * p.nz + z;
    let d = diag[i];
    if (d <= 0.0) { return; }
    let b = bits[i];
    var sum = 0.0;
    if ((b & 1u) != 0u)  { sum += q[i - p.ny * p.nz]; }
    if ((b & 2u) != 0u)  { sum += q[i + p.ny * p.nz]; }
    if ((b & 4u) != 0u)  { sum += q[i - p.nz]; }
    if ((b & 8u) != 0u)  { sum += q[i + p.nz]; }
    if ((b & 16u) != 0u) { sum += q[i - 1u]; }
    if ((b & 32u) != 0u) { sum += q[i + 1u]; }
    q[i] = q[i] + p.omega * ((rhs[i] + sum) / d - q[i]);
}
"""

_ADAPTER = {}


def available():
    """True when a wgpu adapter can be opened here (the answer is cached for the process)."""
    if "ok" not in _ADAPTER:
        try:
            GpuPressure3D()
            _ADAPTER["ok"] = True
        except Exception:                      # no wgpu package, no adapter, no driver
            _ADAPTER["ok"] = False
    return _ADAPTER["ok"]


class GpuPressure3D:
    def __init__(self, batch=32, max_sweeps=None):
        import wgpu
        self.wgpu = wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("no wgpu adapter")
        self.adapter_name = str(adapter.info.get("device", "?"))
        self.device = adapter.request_device_sync()
        module = self.device.create_shader_module(code=WGSL)
        self.pipeline = self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "sweep"})
        self.batch = int(batch)
        self.max_sweeps = max_sweeps
        self._shape = None
        self._system = None

    def _allocate(self, shape):
        wgpu, device = self.wgpu, self.device
        size = int(np.prod(shape)) * 4
        usage = wgpu.BufferUsage
        self.q = device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST | usage.COPY_SRC)
        self.rhs = device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST)
        self.diag = device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST)
        self.bits = device.create_buffer(size=size, usage=usage.STORAGE | usage.COPY_DST)
        nx, ny, nz = shape
        omega = 2.0 / (1.0 + np.sin(np.pi / max(shape)))
        self.uniforms = [device.create_buffer_with_data(
            data=np.array([nx, ny, nz, c], "u4").tobytes() + np.array([omega, 0, 0, 0], "f4").tobytes(),
            usage=usage.UNIFORM) for c in (0, 1)]
        layout = self.pipeline.get_bind_group_layout(0)
        self.groups = [device.create_bind_group(layout=layout, entries=[
            {"binding": 0, "resource": {"buffer": u}}, {"binding": 1, "resource": {"buffer": self.q}},
            {"binding": 2, "resource": {"buffer": self.rhs}}, {"binding": 3, "resource": {"buffer": self.diag}},
            {"binding": 4, "resource": {"buffer": self.bits}}]) for u in self.uniforms]
        self._shape = shape
        self._system = None

    def solve(self, rhs, x0, tolerance, max_iterations, cancel=None, system=None):
        shape = rhs.shape
        if self._shape != shape:
            self._allocate(shape)
        if self._system is not system:
            self.device.queue.write_buffer(self.diag, 0, np.ascontiguousarray(system.diagonal(), "f4"))
            self.device.queue.write_buffer(self.bits, 0, np.ascontiguousarray(system.neighbour_bits(), "u4"))
            self._system = system
        device = self.device
        nx, ny, nz = shape
        groups = ((nx + 7) // 8, (ny + 7) // 8, (nz + 3) // 4)
        cap = self.max_sweeps or max_iterations
        scratch = np.empty(shape, np.float64)
        q = np.array(x0, np.float64)
        r = rhs - system.apply(q, scratch)
        residual = float(np.abs(r).max())
        sweeps = 0
        zeros = np.zeros(shape, "f4")
        while residual > tolerance and sweeps < cap:
            goal = max(tolerance, residual / 50.0)
            device.queue.write_buffer(self.q, 0, zeros)
            device.queue.write_buffer(self.rhs, 0, np.ascontiguousarray(r, "f4"))
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                encoder = device.create_command_encoder()
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(self.pipeline)
                for _ in range(self.batch):
                    for group in self.groups:
                        compute.set_bind_group(0, group)
                        compute.dispatch_workgroups(*groups)
                compute.end()
                device.queue.submit([encoder.finish()])
                sweeps += self.batch
                e = np.frombuffer(device.queue.read_buffer(self.q), "f4").reshape(shape).astype(np.float64)
                r_e = r - system.apply(e, scratch)
                inner = float(np.abs(r_e).max())
                if inner <= goal or sweeps >= cap:
                    break
            q += e
            r = r_e.copy()
            residual = float(np.abs(r).max())
        return q, sweeps, residual
