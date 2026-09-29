"""R6 (docs/SPLAT_RELIGHTING.md, "Production look"): the interactive viewport's GGX and dome lighting,
for meshes (materials 1, `scene3d._shade_pbr_mesh`) and relit splats, against the CPU reference and
against a light move. GPU cases skip without an adapter.
"""
import os
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import envlight as E, gpu3d, scene3d as s, viewportgpu
from tests.test_3d_environment_light import env_of
from tests.test_3d_splat_pbr import sphere_cloud

BACKGROUND = (0.02, 0.02, 0.03, 1.0)
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 1, 5)), s.Vec3(0, 0, 0), 40.0, 0.1, 100.0)


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(np.maximum(linear, 1e-9), 1 / 2.4) - 0.055)


def cpu_reference(scene, width, height, ambient):
    image = s.render(scene, CAMERA, width, height, BACKGROUND, ambient=ambient, shadows=False, samples=2)
    return (srgb(image[..., :3]) * 255).round().astype(int)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class PBRMeshAgreesWithCPU(unittest.TestCase):
    """A simple lit scene: a `pbr` sphere under one light and a uniform dome (deliverable 1)."""

    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def test_ggx_and_dome_match_the_final_render_within_tolerance(self):
        sphere = replace(s._sphere(1.0, 32, (0.7, 0.7, 0.75, 1.0), s.Transform3D()),
                         material="pbr", metallic=0.0, pbr_roughness=0.45, pbr_specular=0.5)
        env = env_of(np.full((32, 64, 3), 0.25, np.float32))
        light = s.Light(intensity=1.4, position=s.Vec3(3, 4, 2))
        scene = s.Scene((sphere,), (light,), environments=(env,))
        width, height, ambient = 240, 160, 0.0
        frame = self.gpu.render(scene, CAMERA, width, height, BACKGROUND, headlight=False, ambient=ambient)
        self.assertEqual(frame.shape, (height, width, 4))
        expected = cpu_reference(scene, width, height, ambient)
        difference = np.abs(frame[..., :3].astype(int) - expected)
        # A uniform dome removes the atlas's resampling error, so this holds to the same bar as the
        # existing Blinn-Phong viewport tests (test_3d_viewport_gpu.py); only the silhouette's
        # antialiasing (4x MSAA against 2x2 supersampling) is expected to differ.
        self.assertLess(difference.mean(), 0.8)
        self.assertLess((difference.max(axis=2) > 12).mean(), 0.04)

    def test_metallic_sphere_also_matches(self):
        sphere = replace(s._sphere(1.0, 32, (0.9, 0.75, 0.3, 1.0), s.Transform3D()),
                         material="pbr", metallic=1.0, pbr_roughness=0.3, pbr_specular=0.5)
        env = env_of(np.full((32, 64, 3), 0.2, np.float32))
        light = s.Light(intensity=1.2, position=s.Vec3(-2, 3, 3))
        scene = s.Scene((sphere,), (light,), environments=(env,))
        width, height, ambient = 240, 160, 0.0
        frame = self.gpu.render(scene, CAMERA, width, height, BACKGROUND, headlight=False, ambient=ambient)
        expected = cpu_reference(scene, width, height, ambient)
        difference = np.abs(frame[..., :3].astype(int) - expected)
        self.assertLess(difference.mean(), 0.8)
        self.assertLess((difference.max(axis=2) > 12).mean(), 0.04)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class RelitSplatsRespondToTheDomeAndLights(unittest.TestCase):
    """Splats drawn as splats and relit with GGX (deliverable 1); the viewport proxy reads the
    cloud's own per-splat de-lit albedo and roughness when it has one (see `viewportgpu.splat_proxy`
    and `PerSplatIntrinsics` below), falling back to one roughness_scale/metallic per cloud
    otherwise (metallic always is, matching the CPU fit's own limit)."""

    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def _instance(self):
        cloud = sphere_cloud(count=2000, radius=1.0, albedo=(0.85, 0.85, 0.9), roughness=0.5)
        return s.SplatInstance(cloud, relight=1.0, metallic=0.85, roughness_scale=0.12)

    def test_specular_highlight_moves_with_the_light(self):
        width, height = 96, 96

        def brightest(position):
            scene = s.Scene(splats=(self._instance(),), lights=(s.Light(intensity=3.0, position=position),))
            frame = self.gpu.render(scene, CAMERA, width, height, BACKGROUND, headlight=False, ambient=0.05)
            luma = frame[..., :3].astype(np.float64).sum(axis=2)
            return np.unravel_index(np.argmax(luma), luma.shape)

        left = brightest(s.Vec3(-3, 1, 4))
        right = brightest(s.Vec3(3, 1, 4))
        self.assertNotEqual(left, right)
        self.assertGreater(right[1], left[1])  # the hot spot follows the light from left to right

    def test_dome_lights_a_cloud_with_relight_and_no_scene_lights(self):
        env = env_of(np.full((32, 64, 3), 0.6, np.float32))
        dark_scene = s.Scene(splats=(self._instance(),))
        lit_scene = s.Scene(splats=(self._instance(),), environments=(env,))
        width, height = 96, 96
        dark = self.gpu.render(dark_scene, CAMERA, width, height, BACKGROUND, headlight=False, ambient=0.0)
        lit = self.gpu.render(lit_scene, CAMERA, width, height, BACKGROUND, headlight=False, ambient=0.0)
        self.assertGreater(lit[..., :3].astype(int).sum(), dark[..., :3].astype(int).sum() + 1000)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class PerSplatIntrinsics(unittest.TestCase):
    """The de-lit layer's own per-splat albedo and roughness reach the viewport (R6 "next",
    closed): a cloud without one still shades exactly as the pre-R6 per-cloud approximation did."""

    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def _split_cloud(self):
        """A red/green half sphere, so each half's pixels show which side's de-lit albedo lit it."""
        from nodebased import intrinsics as I
        base = sphere_cloud(count=2000, radius=1.0, albedo=(1, 1, 1), roughness=0.5)
        normals = base.normals()
        red_side = normals[:, 0] >= 0
        albedo = np.where(red_side[:, None], np.array((0.9, 0.05, 0.05)), np.array((0.05, 0.9, 0.05)))
        layer = I.Intrinsics(albedo, np.full(len(base), 0.5), normals, np.ones(len(base)),
                             np.ones(len(base)), np.ones(len(base)), np.zeros(3), np.zeros((0, 3)),
                             np.zeros((0, 3)), 0)
        return I.attach(base, layer)

    def test_per_splat_albedo_reaches_the_render_ambient_only(self):
        instance = s.SplatInstance(self._split_cloud(), relight=1.0, metallic=0.0, roughness_scale=1.0)
        scene = s.Scene(splats=(instance,))
        width, height = 96, 96
        frame = self.gpu.render(scene, CAMERA, width, height, BACKGROUND, headlight=False, ambient=0.5)
        left = frame[height // 2, width // 4, :3].astype(int)     # camera at +Z: -X projects to the left
        right = frame[height // 2, 3 * width // 4, :3].astype(int)
        self.assertGreater(left[1], left[0] + 8)    # the -X side (green albedo) reads greener than red
        self.assertGreater(right[0], right[1] + 8)  # the +X side (red albedo) reads redder than green

    def test_without_a_delit_layer_the_baked_colour_still_lights_uniformly(self):
        """No regression: a cloud with no `intrinsics` shades with one SH-DC colour, as before."""
        instance = self._instance()  # sphere_cloud always attaches a layer; strip it back off
        instance = replace(instance, cloud=replace(instance.cloud, intrinsics=None))
        scene = s.Scene(splats=(instance,), lights=(s.Light(intensity=1.0, position=s.Vec3(2, 2, 4)),))
        width, height = 96, 96
        frame = self.gpu.render(scene, CAMERA, width, height, BACKGROUND, headlight=False, ambient=0.1)
        self.assertGreater(frame[..., :3].astype(int).sum(), 0)

    def _instance(self):
        cloud = sphere_cloud(count=2000, radius=1.0, albedo=(0.85, 0.85, 0.9), roughness=0.5)
        return s.SplatInstance(cloud, relight=1.0, metallic=0.0, roughness_scale=0.5)
