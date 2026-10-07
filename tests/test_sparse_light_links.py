"""Lane 6 step N2 (finish 1): sparse smoke and light linking work together.

Sparse volumes are sampled from a tile atlas (Lane 6); each volume's light-link mask decides which lights scatter in it and
which lights it shadows (Lane 4, step T1). A sparse plume excluded from one light and lit by another is rendered by the GPU
raster preview and the GPU path tracer and compared with the CPU reference of the same tiles, with the dense twin, and with
the scene that never had the excluded light. Run the module on the other adapters with
`scratch/nb-lanes/run/force-adapter.py integrated|cpu tests.test_sparse_light_links`.
"""
import os
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace as pt, scene3d as s
from tests import test_sparse_gpu_pathtrace as spt, test_sparse_gpu_volumes as sgv
from tests.test_3d_light_linking_media import KEY_SUN, FILL_LAMP, OVERHEAD, LINKED, NOBODY
from tests.test_3d_pathtrace_gpu_soft import READY

# The existing volume tolerance of the GPU raster against the CPU reference (test_3d_light_linking_media.SMOKE_TOLERANCE).
TOLERANCE = max(2e-3, sgv.TOLERANCE * 10)


def linked(volume, link):
    return replace(volume, light_link=link)


def raster(volume, *, gpu=True, lights=(KEY_SUN, FILL_LAMP), geometries=(), settings=sgv.SETTINGS):
    scene = s.Scene(geometries=geometries, volumes=(volume,), lights=lights)
    draw = gpu3d.render if gpu else s.render
    return draw(scene, sgv.CAMERA, 120, 90, volume=settings, ambient=0.1)


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class SparsePlumeOnTheGpuRaster(unittest.TestCase):
    def setUp(self):
        self.dense, self.sparse = sgv.twin()
        self.assertLess(self.sparse.sparse.tile_count, 0.4 * (40 // 8) ** 3, "the plume must leave tiles empty")

    def test_a_sparse_plume_excluded_from_one_light_matches_the_cpu_reference(self):
        volume = linked(self.sparse, LINKED)
        with sgv.never_dense():
            gpu = raster(volume)
        cpu = raster(volume, gpu=False)
        self.assertGreater(int((gpu[..., 3] > 0.02).sum()), 150, "the scene must show smoke")
        self.assertLess(float(np.abs(gpu - cpu).max()), TOLERANCE)
        self.assertLess(float(np.abs(gpu - cpu).mean()), TOLERANCE / 10)

    def test_the_excluded_lamp_leaves_no_in_scattering_and_the_other_light_still_lights_the_plume(self):
        with sgv.never_dense():
            plain = raster(linked(self.sparse, s.LIGHT_LINK_ALL))
            excluded = raster(linked(self.sparse, LINKED))
            key_only = raster(self.sparse, lights=(KEY_SUN,))
            nobody = raster(linked(self.sparse, NOBODY))
        self.assertGreater(float(np.abs(plain - excluded).max()), 0.05)                 # the lamp lit it
        np.testing.assert_allclose(excluded, key_only, atol=TOLERANCE, rtol=0)           # now only the key does
        self.assertGreater(float(excluded[..., :3].sum()), 2.0 * float(nobody[..., :3].sum()))   # and the key does light it

    def test_the_sparse_and_dense_plume_agree_for_every_link(self):
        for link in (s.LIGHT_LINK_ALL, LINKED, ("include", ("fill",)), NOBODY):
            with self.subTest(link=link):
                expected = raster(linked(self.dense, link))
                with sgv.never_dense():
                    actual = raster(linked(self.sparse, link))
                self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE)

    def test_a_sparse_plume_excluded_from_the_overhead_light_throws_no_shadow_on_the_floor(self):
        floor = (sgv.floor(),)
        clear_settings = replace(sgv.SETTINGS, shadow_density=0.0)
        with sgv.never_dense():
            shadowed = raster(linked(self.sparse, s.LIGHT_LINK_ALL), lights=(OVERHEAD,), geometries=floor)
            excluded = raster(linked(self.sparse, ("exclude", ("overhead",))), lights=(OVERHEAD,), geometries=floor)
            clear = raster(linked(self.sparse, ("exclude", ("overhead",))), lights=(OVERHEAD,), geometries=floor,
                           settings=clear_settings)
        bare = raster(replace(self.dense, density=np.zeros_like(np.asarray(self.dense.density))), lights=(OVERHEAD,),
                      geometries=floor)
        on_floor = (bare[..., 3] > 0.99) & (raster(self.sparse, lights=(), geometries=())[..., 3] < 0.002)
        self.assertGreater(int(on_floor.sum()), 200)
        self.assertGreater(float(np.abs(shadowed - clear)[on_floor].max()), 0.05)         # unlinked smoke shadows the floor
        np.testing.assert_allclose(excluded[on_floor], clear[on_floor], atol=TOLERANCE, rtol=0)

    def test_the_viewport_draw_follows_the_link_too(self):
        from nodebased import viewportgpu
        gpu = viewportgpu.renderer()
        self.assertIsNotNone(gpu, viewportgpu.failure())

        def draw(volume, lights=(KEY_SUN, FILL_LAMP)):
            scene = s.Scene((), lights, volumes=(volume,))
            return gpu.render(scene, sgv.CAMERA, 160, 120, sgv.BACKGROUND, headlight=False, ambient=0.1).astype(int)

        with sgv.never_dense():
            plain, excluded = draw(linked(self.sparse, s.LIGHT_LINK_ALL)), draw(linked(self.sparse, LINKED))
            key_only = draw(self.sparse, lights=(KEY_SUN,))
        self.assertGreater(int(np.abs(plain - excluded).max()), 20)
        self.assertLessEqual(int(np.abs(excluded - key_only).max()), 1)


PT_KEY = s.Light("Directional", (1, 0.95, 0.9), 1.5, s.Vec3(0, 0, 0), s.Vec3(0.4, -1, -0.3), name="key")
PT_FILL = s.Light("Point", (1.0, 0.5, 0.3), 5.0, s.Vec3(-2.0, 1.5, 3.0), name="fill")
PT_OVERHEAD = s.Light("Directional", (1, 1, 1), 1.5, s.Vec3(0, 0, 0), s.Vec3(0, -1, 0.01), name="overhead")


@unittest.skipUnless(READY, "no wgpu adapter cleared for splats and smoke in the path tracer")
class SparsePlumeOnTheGpuPathTracer(unittest.TestCase):
    def setUp(self):
        self.dense, self.sparse = spt.blob(fire=False)

    def scene(self, volume, lights=(PT_KEY, PT_FILL), geometries=()):
        return s.Scene(geometries, lights, volumes=(volume,))

    def test_the_gpu_trace_of_a_sparse_plume_excluded_from_one_light_agrees_with_the_cpu_reference(self):
        scene = self.scene(linked(self.sparse, LINKED))
        cpu = spt.trace(scene, "cpu", (16, 16), samples=96, seed=5)
        with spt.never_dense():
            gpu = spt.trace(scene, "gpu", (16, 16), samples=1536, seed=9)
        self.assertGreater(float(cpu[..., 3].max()), 0.3)
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(gpu[..., 3].mean()), float(cpu[..., 3].mean()), delta=0.01)
        np.testing.assert_allclose(gpu[..., 3], cpu[..., 3], atol=0.2)

    def test_the_sparse_and_dense_plume_trace_alike_for_every_link(self):
        for link in (s.LIGHT_LINK_ALL, LINKED, ("include", ("fill",)), NOBODY):
            with self.subTest(link=link):
                expected = spt.trace(self.scene(linked(self.dense, link)))
                with spt.never_dense():
                    actual = spt.trace(self.scene(linked(self.sparse, link)))
                np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_the_excluded_lamp_leaves_no_light_in_the_plume_and_the_key_still_lights_it(self):
        with spt.never_dense():
            plain = spt.trace(self.scene(linked(self.sparse, s.LIGHT_LINK_ALL)))
            excluded = spt.trace(self.scene(linked(self.sparse, LINKED)))
            key_only = spt.trace(self.scene(self.sparse, lights=(PT_KEY,)))
            nobody = spt.trace(self.scene(linked(self.sparse, NOBODY)))
        self.assertGreater(float(np.abs(plain - excluded).max()), 0.02)                  # the lamp lit it
        np.testing.assert_allclose(excluded, key_only, atol=2e-3, rtol=0)                # now only the key does
        self.assertLess(float(nobody[..., :3].max()), 1e-6)                              # excluded from both: dark
        self.assertGreater(float(excluded[..., :3].sum()), 0.0)

    def test_a_sparse_plume_excluded_from_the_overhead_light_casts_no_shadow_on_the_floor(self):
        floor = spt.FLOOR
        lights = (PT_OVERHEAD,)
        settings = dict(max_bounces=1, diffuse_bounces=1)
        clear_volume = replace(spt.SMOKE, shadow_density=0.0)

        def draw(volume, volume_settings=spt.SMOKE):
            return pt.render(self.scene(volume, lights, (floor,)), spt.CAMERA, 24, 24, ambient=0.0, volume=volume_settings,
                             backend="gpu", settings=pt.PathSettings(samples=96, seed=5, **settings))

        with spt.never_dense():
            shadowed = draw(linked(self.sparse, s.LIGHT_LINK_ALL))
            excluded = draw(linked(self.sparse, ("exclude", ("overhead",))))
            clear = draw(linked(self.sparse, ("exclude", ("overhead",))), clear_volume)
        bare = pt.render(s.Scene((floor,), lights), spt.CAMERA, 24, 24, ambient=0.0, volume=spt.SMOKE, backend="gpu",
                         settings=pt.PathSettings(samples=96, seed=5, **settings))
        smoke_free = pt.render(s.Scene(volumes=(self.sparse,)), spt.CAMERA, 24, 24, ambient=0.0, volume=spt.SMOKE,
                               backend="gpu", settings=pt.PathSettings(samples=8, seed=5))
        on_floor = (bare[..., 3] > 0.99) & (smoke_free[..., 3] < 0.01)
        self.assertGreater(int(on_floor.sum()), 20)
        self.assertGreater(float(np.abs(shadowed - clear)[on_floor].max()), 0.02)        # unlinked smoke shadows the floor
        np.testing.assert_allclose(excluded[on_floor], clear[on_floor], atol=2e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
