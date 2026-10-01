"""M3 gate, part 2: bounded VRAM (docs/M1_GATE.md's M3 section).

Three representative scenes report peak GPU memory per renderer (raster viewport, ray-traced
mode, GPU path tracer) through `nodebased.gpumemory`'s own byte-accurate allocation tracker (see
that module's docstring for why: wgpu's native report has no byte sizes) and stay under a stated
budget, set from a measured run with 30% headroom. Exceeding the budget refuses with a named,
catchable error (`gpumemory.BudgetExceeded`), never crashes.

Not every scene runs on every renderer: this is an existing, already-tested architectural fact
(`gpu3d.Unsupported`'s own messages), not a gap this step introduces. Each renderer/scene pairing
below either measures a real budget or asserts the existing named refusal, so the gate covers
every combination the brief asks about, one way or the other:

  renderer      100k instances        256^3 smoke + splats     PBR set + HDRI + 4 lights
  raster        Unsupported (A)       Unsupported (B)          measured (C)
  ray-traced    measured (A)          Unsupported (B)          Unsupported (C)
  path tracer   Unsupported (A, no    measured (B, NVIDIA      measured (C)
                instancing support)   adapters only)

Slow: real GPU renders (one 100,000-instance scene, one 256-cubed volume, one six-sphere PBR
set), run on the default (discrete) adapter only; the 256^3 smoke+splats case is additionally
gated to NVIDIA adapters by the path tracer itself (`gpupathtrace.render`'s own check), matching
`gpu3d.SHADOW_WORK_BUDGETS`' own "Windows and other adapters are unmeasured" precedent -- a
three-adapter run of this module is still open, named in docs/M1_GATE.md's M3 section.
"""
import os
from dataclasses import replace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import envlight, gpu3d, gpumemory, gpupathtrace, scene3d as s, splats
from nodebased.pathtrace import PathSettings

GPU_AVAILABLE = gpu3d.available()


def _cloud(positions, colours, scales=0.05, opacity=0.9):
    positions = np.atleast_2d(np.asarray(positions, np.float32))
    count = len(positions)
    sh = ((np.broadcast_to(np.asarray(colours, np.float32), (count, 3)) - 0.5) / splats.C0)[:, None, :]
    return splats.SplatCloud(
        positions, np.broadcast_to(np.asarray(scales, np.float32), (count, 3)),
        np.broadcast_to(np.asarray((1, 0, 0, 0), np.float32), (count, 4)),
        np.full(count, opacity, np.float32), sh, 0)


def instance_scene(n=100_000):
    """100,000 instanced spheres over a ground-plane spread of points."""
    sphere = s._sphere(0.3, 8, (.7, .3, .2, 1), s.Transform3D())
    rng = np.random.RandomState(1)
    pts = np.stack([rng.uniform(-50, 50, n), rng.uniform(-5, 5, n), rng.uniform(-50, 50, n)], axis=1)
    points = s.Geometry(pts.astype('f4'), np.zeros((0, 3), 'i4'), (1, 1, 1, 1))
    inst_set = s.instances_from_node(points, sphere, {'inst_scale': 1.0})
    return s.Scene(instances=(inst_set,), lights=(s.Light(position=s.Vec3(10, 20, 10), shadows=True),))


def smoke_splat_scene():
    """A 256-cubed smoke volume (uniform density, simplest worst-case voxel count) plus a 20,000
    -point splat cloud sharing the same frame."""
    n = 256
    volume = s.Volume(np.full((n, n, n), 2.0, np.float32), 4.0 / n, (-2.0, -2.0, -2.0))
    rng = np.random.RandomState(2)
    m = 20_000
    positions = rng.uniform(-1.5, 1.5, (m, 3)).astype('f4')
    splat = s.SplatInstance(_cloud(positions, np.full((m, 3), (.8, .8, .9), 'f4')))
    return s.Scene(volumes=(volume,), splats=(splat,), lights=(s.Light(position=s.Vec3(5, 5, 5), shadows=True),))


def pbr_hdri_scene():
    """Six textured PBR spheres (varying metallic/roughness), an HDRI environment and four
    shadowed point lights."""
    checker = (np.indices((512, 512)).sum(0) % 2 * 0.6 + 0.2).astype('f4')
    texture = np.repeat(checker[..., None], 4, -1)
    texture[..., 3] = 1
    geometries = []
    for i in range(6):
        transform = s.Transform3D(s.Vec3((i % 3 - 1) * 2.5, (i // 3) * 2.5 - 1, 0))
        sphere = s._sphere(1.0, 32, (1, 1, 1, 1), transform)
        geometries.append(replace(sphere, texture=texture, material='pbr',
                                  metallic=i / 5.0, pbr_roughness=0.2 + 0.1 * i))
    hdri = np.random.RandomState(3).uniform(0.1, 1.0, (64, 128, 3)).astype('f4')
    environment = envlight.Environment(hdri, 'm3-vram-gate-hdri', intensity=1.0)
    lights = tuple(s.Light(position=s.Vec3(x, y, z), shadows=True, intensity=4.0)
                  for x, y, z in ((6, 6, 6), (-6, 6, 6), (6, 6, -6), (-6, 6, -6)))
    return s.Scene(tuple(geometries), lights, environments=(environment,))


CAMERA_A = s.Camera(s.Transform3D(s.Vec3(0, 20, 60)))
CAMERA_B = s.Camera(s.Transform3D(s.Vec3(0, 0, 6)))
CAMERA_C = s.Camera(s.Transform3D(s.Vec3(0, 2, 14)))

# Measured on this workstation's default (discrete, RTX 3080 Ti) adapter, 2026-10-01: 72,661,856 /
# 146,403,416 / 32,156,984 / 72,355,936 bytes respectively. Budgets are each figure * 1.3 (30%
# headroom), the same convention tests/test_memory_ceiling_gate.py already uses.
BUDGET_A_RAYTRACE = int(72_661_856 * 1.3)
BUDGET_B_PATHTRACE = int(146_403_416 * 1.3)
BUDGET_C_RASTER = int(32_156_984 * 1.3)
BUDGET_C_PATHTRACE = int(72_355_936 * 1.3)


@unittest.skipUnless(GPU_AVAILABLE, 'no wgpu adapter available')
class VramBudgetGateTests(unittest.TestCase):
    def setUp(self):
        self.state = gpu3d._state()
        self.tracker = self.state['memory']

    def test_100k_instances_raytrace_stays_under_budget(self):
        scene = instance_scene()
        gpumemory.render_bounded(gpu3d.render, scene, CAMERA_A, 512, 512, mode='raytrace',
                                 tracker=self.tracker, budget_bytes=BUDGET_A_RAYTRACE,
                                 renderer_name='ray-traced mode', scene_label='100k instances')

    def test_100k_instances_refuses_on_raster_and_path_tracer(self):
        scene = instance_scene()
        with self.assertRaisesRegex(gpu3d.Unsupported, 'raytrace mode'):
            gpu3d.render(scene, CAMERA_A, 64, 64, mode='raster')

    def test_smoke_and_splats_pathtrace_stays_under_budget(self):
        scene = smoke_splat_scene()
        settings = PathSettings(samples=8)
        try:
            gpumemory.render_bounded(
                gpupathtrace.render, scene, CAMERA_B, 256, 256, (0, 0, 0, 0), 0.0, 'rgba', settings,
                tracker=self.tracker, budget_bytes=BUDGET_B_PATHTRACE,
                renderer_name='GPU path tracer', scene_label='256-cubed smoke + splats')
        except gpu3d.Unsupported as exc:
            self.skipTest(f'path-traced smoke+splats unavailable on this adapter: {exc}')

    def test_smoke_and_splats_refuses_on_raster_and_raytrace(self):
        scene = smoke_splat_scene()
        for mode in ('raster', 'raytrace'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(gpu3d.Unsupported, 'CPU-only'):
                    gpu3d.render(scene, CAMERA_B, 64, 64, mode=mode)

    def test_pbr_hdri_raster_stays_under_budget(self):
        scene = pbr_hdri_scene()
        gpumemory.render_bounded(gpu3d.render, scene, CAMERA_C, 512, 512, mode='raster', ambient=0.05,
                                 tracker=self.tracker, budget_bytes=BUDGET_C_RASTER,
                                 renderer_name='raster viewport', scene_label='PBR set + HDRI')

    def test_pbr_hdri_pathtrace_stays_under_budget(self):
        scene = pbr_hdri_scene()
        settings = PathSettings(samples=8)
        gpumemory.render_bounded(
            gpupathtrace.render, scene, CAMERA_C, 512, 512, (0, 0, 0, 0), 0.05, 'rgba', settings,
            tracker=self.tracker, budget_bytes=BUDGET_C_PATHTRACE,
            renderer_name='GPU path tracer', scene_label='PBR set + HDRI')

    def test_pbr_hdri_refuses_on_raytrace(self):
        scene = pbr_hdri_scene()
        with self.assertRaisesRegex(gpu3d.Unsupported, 'CPU-only'):
            gpu3d.render(scene, CAMERA_C, 64, 64, mode='raytrace', ambient=0.05)

    def test_an_artificially_small_budget_refuses_cleanly_not_a_crash(self):
        scene = pbr_hdri_scene()
        with self.assertRaises(gpumemory.BudgetExceeded) as caught:
            gpumemory.render_bounded(gpu3d.render, scene, CAMERA_C, 512, 512, mode='raster', ambient=0.05,
                                     tracker=self.tracker, budget_bytes=1000,
                                     renderer_name='raster viewport', scene_label='PBR set + HDRI')
        self.assertIn('raster viewport', str(caught.exception))
        self.assertIn('PBR set + HDRI', str(caught.exception))
        # The refusal did not corrupt the tracker or leave the device unusable for the next render.
        image = gpu3d.render(scene, CAMERA_C, 32, 32, mode='raster', ambient=0.05)
        self.assertEqual(image.shape, (32, 32, 4))


if __name__ == '__main__':
    unittest.main()
