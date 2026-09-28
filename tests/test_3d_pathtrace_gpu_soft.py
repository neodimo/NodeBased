"""Splats and smoke in the GPU path tracer (lane L4, step R4 finish 1).

The WGSL twin has to pass what the CPU reference passes: every test of `test_3d_pathtrace_splats` and
`test_3d_pathtrace_volumes` that renders through `pathtrace.render` runs again here with the GPU backend forced
(`OnTheGpu` patches the default), so the same closed forms, furnaces, mirrors, shadows and fire cases hold on the
card. The classes below then compare the two backends directly, pixel for pixel where the light is deterministic
and by mean where it is sampled. GPU tests are guarded by `gpu_ready()` like every other GPU test."""
import math
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace as pt, scene3d as s, volumerender as vr
from tests import test_3d_pathtrace_splats as splat_tests, test_3d_pathtrace_volumes as volume_tests
from tests.test_3d_pathtrace import CORNELL_CAMERA, card, cornell, gpu_ready, sphere, sun_env
from tests.test_3d_pathtrace_splats import instance, plane_cloud
from tests.test_volume_scene import box

READY = gpu_ready()


class OnTheGpu:
    """Mixin: every `pathtrace.render` the inherited tests make uses the GPU unless they pick a backend."""

    def setUp(self):
        super().setUp()
        original = pt.render

        def render(*args, **kwargs):
            kwargs.setdefault("backend", "gpu")
            return original(*args, **kwargs)
        patcher = mock.patch.object(pt, "render", render)
        patcher.start()
        self.addCleanup(patcher.stop)


def _twin(module, name):
    cls = type(f"Gpu{name}", (OnTheGpu, getattr(module, name)), {"__module__": __name__})
    return unittest.skipUnless(READY, "wgpu adapter unavailable for the path tracer")(cls)


GpuRelightParityTests = _twin(splat_tests, "RelightParityTests")
GpuHitTests = _twin(splat_tests, "HitTests")
GpuMirrorAndGlassTests = _twin(splat_tests, "MirrorAndGlassTests")
GpuShadowAndBounceTests = _twin(splat_tests, "ShadowAndBounceTests")
GpuSplatInstanceTests = _twin(splat_tests, "InstanceTests")
GpuTransmittanceTests = _twin(volume_tests, "TransmittanceTests")
GpuPlacementTests = _twin(volume_tests, "PlacementTests")
GpuFurnaceTests = _twin(volume_tests, "FurnaceTests")
GpuDepthTests = _twin(volume_tests, "DepthTests")
GpuShadowTests = _twin(volume_tests, "ShadowTests")
GpuFireTests = _twin(volume_tests, "FireTests")
GpuThroughGlassTests = _twin(volume_tests, "ThroughGlassTests")
GpuNodeTests = unittest.skipUnless(READY, "wgpu adapter unavailable for the path tracer")(
    type("GpuNodeTests", (volume_tests.NodeTests,), {
        "__module__": __name__,
        "graph": lambda self, **render: volume_tests.NodeTests.graph(self, **{"render_backend": "gpu", **render}),
    }))


# --- the two backends against each other ------------------------------------------------------------------------

CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0.5, 0.8, 4.5)), s.Vec3(0, 0, 0), 35.0)
SMOKE = vr.VolumeSettings(absorption=0.6, scattering=0.8, step_size=0.05, shadow_steps=16, fire_intensity=1.0,
                          temperature_scale=2500.0)
SUN = s.Light("Directional", (1, 0.95, 0.9), 1.5, s.Vec3(0, 0, 0), s.Vec3(0.4, -1, -0.3))
RECT = s.Light("Rect", (1, 0.9, 0.8), 6.0, s.Vec3(0, 3, 1), s.Vec3(0, 0, 0), area_width=1.5, area_height=1.5)
POINT = s.Light("Point", (1, 1, 1), 5.0, s.Vec3(-2, 2, 2), s.Vec3(0, 0, 0))
FLOOR = card(8, 8, (0.7, 0.7, 0.7, 1), (0, -1, 0), (-90, 0, 0))


def blob(nx=14, ny=10, nz=18, temperature=True):
    """An off-centre Gaussian of smoke on a grid whose three sizes differ, turned about y: an axis mix-up shows."""
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    density = 2.5 * np.exp(-(((x - 4) / 3.0) ** 2 + ((y - 6) / 2.5) ** 2 + ((z - 11) / 4.0) ** 2))
    kelvin = np.where(density > 0.3, 1.0, 0.0) * (0.4 + y / ny) * 1.5
    matrix = np.eye(4)
    c, sn = math.cos(0.6), math.sin(0.6)
    matrix[:3, :3] = ((c, 0, sn), (0, 1, 0), (-sn, 0, c))
    matrix[:3, 3] = (0.2, 0.1, -0.1)
    return s.Volume(density.astype(np.float32), 0.15, (-1.0, -0.7, -1.4), matrix=matrix,
                    temperature=kelvin.astype(np.float32) if temperature else None)


def sheet_on_floor(**fields):
    matrix = np.eye(4)
    matrix[:3, :3] = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    matrix[1, 3] = -0.3
    return instance(plane_cloud(n=20, size=2.5, albedo=(0.6, 0.5, 0.4), delit=True), matrix=matrix, **fields)


def both(scene, size=(16, 16), cpu_samples=96, gpu_samples=1536, ambient=0.0, volume=SMOKE, **settings):
    settings.setdefault("max_bounces", 6)
    settings.setdefault("diffuse_bounces", 8)
    cpu = pt.render(scene, CAMERA, *size, ambient=ambient, volume=volume,
                    settings=pt.PathSettings(samples=cpu_samples, seed=5, **settings))
    gpu = pt.render(scene, CAMERA, *size, ambient=ambient, volume=volume, backend="gpu",
                    settings=pt.PathSettings(samples=gpu_samples, seed=9, **settings))
    return cpu, gpu


@unittest.skipUnless(READY, "wgpu adapter unavailable for the path tracer")
class BackendAgreementTests(unittest.TestCase):
    """The card and the reference draw one picture: pixel for pixel where the light is deterministic, by mean
    where it is sampled."""

    def test_a_sun_lit_splat_sheet_agrees_pixel_for_pixel(self):
        scene = s.Scene(lights=(splat_tests.SUN,), splats=(instance(plane_cloud(delit=False)),))
        settings = pt.PathSettings(samples=8, max_bounces=1)
        cpu = pt.render(scene, splat_tests.SPLAT_CAMERA, 16, 16, ambient=0.1, settings=settings)
        gpu = pt.render(scene, splat_tests.SPLAT_CAMERA, 16, 16, ambient=0.1, settings=settings, backend="gpu")
        self.assertLess(float(np.abs(cpu - gpu).max()), 1e-4)

    def test_meshes_splats_and_smoke_lit_by_lights_and_a_sky_agree(self):
        ball = sphere(0.5, (1, 1, 1, 1), (1.6, -0.5, 0), material="pbr", metallic=1.0, pbr_roughness=0.0)
        glass = sphere(0.5, (1, 1, 1, 1), (-1.6, -0.5, 0.5), material="liquid", ior=1.5,
                       absorption_color=(0.9, 0.95, 1.0), absorption_distance=3.0, reflection=1.0)
        scene = s.Scene((FLOOR, ball, glass), lights=(POINT, RECT, SUN), environments=(sun_env(0.3),),
                        splats=(sheet_on_floor(relight=0.6),), volumes=(blob(),))
        cpu, gpu = both(scene, ambient=0.05, cpu_samples=64)
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.04)
        self.assertAlmostEqual(float(gpu[..., 3].mean()), float(cpu[..., 3].mean()), delta=0.01)
        self.assertGreater(float(cpu[..., 3].mean()), 0.3)                       # the picture holds all three kinds

    def test_a_lit_oblong_turned_cloud_of_smoke_reads_its_grid_like_the_reference(self):
        scene = s.Scene(lights=(SUN,), environments=(sun_env(0.3),), volumes=(blob(temperature=False),))
        cpu, gpu = both(scene, cpu_samples=512, gpu_samples=4096)
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.03)
        np.testing.assert_allclose(gpu[..., 3], cpu[..., 3], atol=0.08)
        # where the smoke stands is compared pixel by pixel: a swapped axis moves it
        covered = (cpu[..., 3] > 0.05) | (gpu[..., 3] > 0.05)
        self.assertGreater(float(np.corrcoef(cpu[..., 3][covered], gpu[..., 3][covered])[0, 1]), 0.95)

    def test_fire_in_two_overlapping_volumes_glows_like_the_reference(self):
        second = replace(blob(), origin=(-0.6, -0.9, -1.0))
        scene = s.Scene((FLOOR,), volumes=(blob(), second))
        cpu, gpu = both(scene, cpu_samples=384, gpu_samples=3072)
        self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.04)
        self.assertAlmostEqual(float(gpu[..., 3].mean()), float(cpu[..., 3].mean()), delta=0.01)

    def test_the_data_passes_see_overlapping_splats_meshes_and_smoke_alike(self):
        a = plane_cloud(n=20, size=2.5, albedo=(0.6, 0.5, 0.4), delit=True, opacity=0.35)
        b = plane_cloud(n=20, size=2.5, albedo=(0.2, 0.5, 0.9), delit=False, opacity=0.35, z=-0.4)
        turn = np.eye(4)
        turn[:3, :3] = ((0.9, 0.1, 0), (0, 0.9, 0.2), (0, -0.2, 0.9))
        scene = s.Scene((FLOOR, sphere(0.6, (1, 0, 0, 1), (1, 0, 0))), lights=(splat_tests.SUN,),
                        splats=(instance(a, matrix=turn), instance(b, relight=0.0)), volumes=(box(16, 1.0),))
        for output in ("depth", "normals", "position", "uv", "object_id", "albedo"):
            with self.subTest(output=output):
                settings = pt.PathSettings(samples=32)
                cpu = pt.render(scene, CAMERA, 32, 24, output=output, volume=SMOKE, settings=settings)
                gpu = pt.render(scene, CAMERA, 32, 24, output=output, volume=SMOKE, settings=settings, backend="gpu")
                self.assertGreater(int((cpu[..., 3] > 0).sum()), 500)
                self.assertLess(int(((cpu[..., 3] > 0) != (gpu[..., 3] > 0)).sum()), 4)
                same = (cpu[..., 3] > 0) & (gpu[..., 3] > 0)
                self.assertLess(float(np.abs(cpu - gpu)[same].max()), 5e-3)

    def test_auto_runs_splats_and_smoke_on_the_gpu(self):
        for scene in (s.Scene(lights=(splat_tests.SUN,), splats=(instance(plane_cloud(n=6)),)),
                      s.Scene(volumes=(box(8, 1.0),))):
            stats = {}
            image = pt.render(scene, CAMERA, 8, 8, backend="auto", stats=stats, volume=SMOKE,
                              settings=pt.PathSettings(samples=4))
            self.assertEqual(stats["backend"], "gpu")
            self.assertNotIn("fallback", stats)
            self.assertEqual(image.shape, (8, 8, 4))

    def test_splats_and_volumes_are_packed_beside_the_meshes(self):
        scene = s.Scene((FLOOR,), lights=(SUN,), splats=(sheet_on_floor(),), volumes=(blob(), blob()))
        ps = pt.build_scene(scene, 0.0, volume=SMOKE)
        packed = gpupathtrace.pack(ps)
        self.assertEqual(packed.splat_count, len(ps.splats))
        self.assertEqual(packed.volume_count, 2)
        self.assertGreater(packed.splat_root, 0)                               # its tree follows the mesh trees
        self.assertTrue(packed.flags & 2)                                      # the fire table is there
        floats = sum(v.density.size + v.temperature.size for v in scene.volumes)
        self.assertGreater(len(packed.env) * 4, floats)                        # grids are in the environment buffer

    def test_the_render_can_be_cancelled_and_the_band_size_shrinks_for_soft_scenes(self):
        import threading
        from nodebased.cancellation import Cancelled
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            pt.render(s.Scene(volumes=(box(8, 1.0),)), CAMERA, 8, 8, backend="gpu", volume=SMOKE, cancel=event,
                      settings=pt.PathSettings(samples=4))
        self.assertGreater(gpupathtrace.SOFT_SLOWDOWN, 1)


class ShaderVariantTests(unittest.TestCase):
    """The splat and smoke code is compiled in only for scenes that have them: it costs registers whether it runs or
    not (a mesh-only scene rendered 2.4 times slower with it always on, measured)."""

    def test_a_mesh_only_scene_gets_a_shader_without_splat_or_smoke_code(self):
        plain = gpupathtrace.shader_source()
        for word in ("splat_nearest", "splat_transmittance", "vol_flight", "soft_visibility", "hg_sample", "skip_id"):
            self.assertNotIn(word, plain)
        self.assertNotIn("//#", plain)

    def test_each_kind_of_scene_keeps_only_what_it_uses(self):
        splats, volumes = gpupathtrace.shader_source(splats=True), gpupathtrace.shader_source(volumes=True)
        both_ = gpupathtrace.shader_source(splats=True, volumes=True)
        self.assertIn("splat_nearest", splats)
        self.assertNotIn("vol_flight", splats)
        self.assertIn("vol_flight", volumes)
        self.assertNotIn("splat_nearest", volumes)
        self.assertIn("splat_nearest", both_)
        self.assertIn("vol_flight", both_)
        for code in (splats, volumes, both_):
            self.assertNotIn("//#", code)
            self.assertNotIn("SPLAT_ID", code)

    def test_an_unbalanced_directive_is_refused(self):
        with self.assertRaises(ValueError):
            gpupathtrace._preprocess("//#if SPLATS\nx", {"SPLATS": True})

    @unittest.skipUnless(READY, "wgpu adapter unavailable for the path tracer")
    def test_every_variant_compiles(self):
        state = gpu3d._state()
        for splats in (False, True):
            for volumes in (False, True):
                self.assertIsNotNone(gpupathtrace._pipeline(state, splats, volumes))


@unittest.skipUnless(READY, "wgpu adapter unavailable for the path tracer")
class DenoiseOnTheGpuTests(unittest.TestCase):
    """The filter on a picture with splats and smoke, fed by the GPU tracer's own variance and guides."""

    SIZE = 32

    @classmethod
    def setUpClass(cls):
        self = cls
        matrix = np.eye(4)
        matrix[:3, :3] = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
        matrix[1, 3] = -0.6
        panel = instance(plane_cloud(n=24, size=1.0, albedo=(0.7, 0.55, 0.3), delit=True), matrix=matrix, relight=1.0)
        puff = replace(box(12, 1.5, size=0.7), matrix=np.array(((1, 0, 0, 0.55), (0, 1, 0, -0.5), (0, 0, 1, 0.2),
                                                                 (0, 0, 0, 1.0))))
        self.volume = vr.VolumeSettings(absorption=0.2, scattering=0.9, step_size=0.05, shadow_steps=12)
        self.scene = replace(cornell(), splats=(panel,), volumes=(puff,))
        self.reference = pt.render(self.scene, CORNELL_CAMERA, self.SIZE, self.SIZE, volume=self.volume, backend="gpu",
                                   settings=pt.PathSettings(samples=4096, max_bounces=6, seed=99))

    def test_the_filtered_beauty_is_closer_to_the_reference_at_16_samples_and_keeps_its_mean(self):
        settings = pt.PathSettings(samples=16, max_bounces=6, seed=5)
        stats = {}
        raw = pt.render(self.scene, CORNELL_CAMERA, self.SIZE, self.SIZE, volume=self.volume, backend="gpu",
                        settings=settings, stats=stats)
        filtered = pt.render(self.scene, CORNELL_CAMERA, self.SIZE, self.SIZE, output="denoise", volume=self.volume,
                             backend="gpu", settings=settings)
        mse = lambda a: float(np.mean((a[..., :3].astype(np.float64) - self.reference[..., :3]) ** 2))
        factor = mse(raw) / mse(filtered)
        print(f"\ndenoise on the GPU with splats and smoke: factor {factor:.2f}")
        self.assertGreater(factor, 1.8)
        self.assertLess(abs(float(filtered[..., :3].mean()) / float(raw[..., :3].mean()) - 1.0), 0.015)
        self.assertEqual(stats["variance"].shape, (self.SIZE, self.SIZE))      # the GPU hands the filter its variance
        self.assertGreater(float(stats["variance"].max()), 0.0)

    def test_the_guide_passes_agree_with_the_reference_tracer(self):
        settings = pt.PathSettings(samples=4, max_bounces=1)
        cpu = pt.guide_aovs(self.scene, CORNELL_CAMERA, 16, 16, settings, volume=self.volume)
        gpu = pt.guide_aovs(self.scene, CORNELL_CAMERA, 16, 16, settings, backend="gpu", volume=self.volume)
        self.assertEqual(sorted(gpu), ["albedo", "depth", "normals"])
        for name in ("normals", "depth"):
            differ = np.abs(gpu[name] - cpu[name]).max(axis=-1) > 1e-3
            self.assertLess(float(differ.mean()), 0.03, name)
        np.testing.assert_allclose(gpu["albedo"][..., :3].mean(axis=(0, 1)), cpu["albedo"][..., :3].mean(axis=(0, 1)),
                                   rtol=0.04)

    def test_the_multichannel_output_holds_the_raw_beauty_next_to_the_guides_and_the_filtered_image(self):
        beauty, layers = s.render_multichannel(self.scene, CORNELL_CAMERA, 16, 16, passes="beauty,albedo,normals,depth,denoise",
                                               mode="pathtrace", backend="gpu", volume=self.volume,
                                               path=pt.PathSettings(samples=8, max_bounces=3))
        self.assertEqual(sorted(layers), ["albedo", "denoise", "depth", "normals"])
        self.assertGreater(float(beauty[..., 3].max()), 0.9)
        for name, image in layers.items():
            self.assertEqual(image.shape, (16, 16, 4), name)
            self.assertGreater(float(np.abs(image[..., :3]).max()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
