"""wgpu compute variant of the pressure solve only (Lane 6 step 2). The NumPy conjugate gradient in
nodebased/fluid2d.py is the reference; this plugs into `Smoke2D(pressure_solver=GpuPressure().solve)`.

Red-black Gauss-Seidel with over-relaxation (SOR, omega = 2 / (1 + sin(pi / max(nx, ny))), the standard
choice that makes the sweep count grow with the grid edge instead of its square) on the same Neumann 5-point system A q = rhs, float32, in place. Sweeps run in
batches of `batch` (one submit each); after a batch the pressure is read back and the residual is checked
on the CPU with the reference operator, so the stopping rule (max |rhs - A q| <= tolerance) is identical
to the NumPy solve. Warm start: the previous pressure is the starting point of the refinement.
Run under the GPU lock: `flock /tmp/nb-gpu.lock python tools/benchmark_fluid.py --gpu`.
"""
from __future__ import annotations

import numpy as np

from nodebased.cancellation import Cancelled
from nodebased.fluid2d import laplacian_apply

WGSL = """
struct P { nx: u32, ny: u32, color: u32, pad: u32, omega: f32, p1: f32, p2: f32, p3: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> rhs: array<f32>;

@compute @workgroup_size(16, 16)
fn sweep(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    if (x >= p.nx || y >= p.ny) { return; }
    if (((x + y) & 1u) != p.color) { return; }
    let i = y * p.nx + x;
    var sum = 0.0; var n = 0.0;
    if (x > 0u)        { sum += q[i - 1u];     n += 1.0; }
    if (x + 1u < p.nx) { sum += q[i + 1u];     n += 1.0; }
    if (y > 0u)        { sum += q[i - p.nx];   n += 1.0; }
    if (y + 1u < p.ny) { sum += q[i + p.nx];   n += 1.0; }
    q[i] = q[i] + p.omega * ((rhs[i] + sum) / n - q[i]);
}
"""


class GpuPressure:
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

    def _allocate(self, shape):
        wgpu, device = self.wgpu, self.device
        ny, nx = shape
        size = ny * nx * 4
        self.q = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
        self.rhs = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        omega = 2.0 / (1.0 + np.sin(np.pi / max(nx, ny)))
        self.uniforms = [device.create_buffer_with_data(
            data=np.array([nx, ny, c, 0], "u4").tobytes() + np.array([omega, 0, 0, 0], "f4").tobytes(),
            usage=wgpu.BufferUsage.UNIFORM) for c in (0, 1)]
        layout = self.pipeline.get_bind_group_layout(0)
        self.groups = [device.create_bind_group(layout=layout, entries=[
            {"binding": 0, "resource": {"buffer": u}}, {"binding": 1, "resource": {"buffer": self.q}},
            {"binding": 2, "resource": {"buffer": self.rhs}}]) for u in self.uniforms]
        self._shape = shape

    def solve(self, rhs, x0, tolerance, max_iterations, cancel=None):
        """Iterative refinement: the GPU solves A e = r for the current float64 residual r starting from
        e = 0 until the correction problem's residual has dropped 50-fold (or meets the tolerance), then
        q += e and r is recomputed on the CPU. float32 only ever holds the correction, so the precision
        floor that stalls a plain float32 solve at 512 x 512 (max divergence about 1.6e-3) does not apply."""
        shape = rhs.shape
        if self._shape != shape:
            self._allocate(shape)
        ny, nx = shape
        device = self.device
        groups = ((nx + 15) // 16, (ny + 15) // 16, 1)
        cap = self.max_sweeps or max_iterations
        scratch = np.empty(shape, np.float64)
        q = np.array(x0, np.float64)
        r = rhs - laplacian_apply(q, scratch)
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
                r_e = r - laplacian_apply(e, scratch)
                inner = float(np.abs(r_e).max())
                if inner <= goal or sweeps >= cap:
                    break
            q += e
            r = r_e.copy()
            residual = float(np.abs(r).max())
        return q, sweeps, residual
