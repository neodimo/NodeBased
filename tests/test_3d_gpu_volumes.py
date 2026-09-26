"""Lane L4 plan 3, step A part 1: gpu3d.render raymarches scene.volumes and matches the CPU reference."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, gpuvolume, scene3d as s, volumerender
from tests import gpu_precision

W, H = 128, 96
# Half-float targets (Windows CI's basic render driver) quantise the accumulated colour.
TOLERANCE = 5e-3 if gpu_precision.half_float_target() else 2e-4
CAMERA = s.Camera(s.Transform3D(s.Vec3(0.3, 0.6, 3.2)), s.Vec3(0, 0.5, 0))
SETTINGS = volumerender.VolumeSettings(step_size=0.03, density_scale=8.0, shadow_steps=12)
SUN = s.Light('Directional', position=s.Vec3(4, 5, 3), target=s.Vec3())
LAMP = s.Light('Point', color=(1.0, 0.6, 0.3), intensity=3.0, position=s.Vec3(-1.5, 1.2, 1.0), target=s.Vec3(0, .5, 0),
               falloff_type='Quadratic')
SPOT = s.Light('Spot', color=(0.4, 0.6, 1.0), intensity=2.0, position=s.Vec3(1.5, 2.5, 1.5), target=s.Vec3(0, .5, 0),
               cone_angle=40.0, cone_penumbra_angle=15.0, falloff_type='Linear')


def plume(n=32, seed=0):
    return s.analytic_plume(n, seed)


def card(z, width=1.2, color=(.8, .3, .2, 1.0)):
    return s._card(width, width, color, s.Transform3D(s.Vec3(0, .5, z)))


def compare(scene, settings=SETTINGS, width=W, height=H, camera=CAMERA, **kwargs):
    expected = s.render(scene, camera, width, height, volume=settings, **kwargs)
    actual = gpu3d.render(scene, camera, width, height, volume=settings, **kwargs)
    return expected, actual


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUVolumeParity(unittest.TestCase):
    def check(self, scene, **kwargs):
        expected, actual = compare(scene, **kwargs)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertGreater(int((expected[..., 3] > 0.02).sum()), 400, 'the scene must actually show smoke')
        self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 10)
        self.assertLess(float(np.abs(actual - expected).mean()), TOLERANCE)
        return expected, actual

    def test_a_one_light_plume_matches_the_cpu_raymarch(self):
        expected, actual = self.check(s.Scene(volumes=(plume(),), lights=(SUN,)), ambient=0.2)
        self.assertGreater(float(actual[..., :3].max()), 0.1)

    def test_a_three_light_plume_matches_the_cpu_raymarch(self):
        scene = s.Scene(volumes=(plume(),), lights=(SUN, LAMP, SPOT))
        expected, actual = self.check(scene, ambient=0.1)
        one = gpu3d.render(replace(scene, lights=(SUN,)), CAMERA, W, H, ambient=0.1, volume=SETTINGS)
        self.assertGreater(float(np.abs(actual - one).max()), 0.05, 'the extra lights must change the image')

    def test_an_unlit_plume_and_a_denser_absorbing_one_match(self):
        self.check(s.Scene(volumes=(plume(),)))
        settings = replace(SETTINGS, absorption=0.8, scattering=0.6, shadow_density=2.5, color=(1.0, .8, .6))
        self.check(s.Scene(volumes=(plume(),), lights=(SUN, SPOT)), settings=settings, ambient=.05)

    def test_supersampling_and_a_background_match(self):
        scene = s.Scene(volumes=(plume(),), lights=(SUN,))
        expected, actual = compare(scene, samples=2, background=(0.1, 0.2, 0.3, 1.0), ambient=.1)
        self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 10)

    def test_a_transformed_volume_and_two_volumes_match(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.array([[0.8, 0, 0.6], [0, 1.4, 0], [-0.6, 0, 0.8]], np.float32)   # rotated and stretched
        matrix[:3, 3] = (0.25, 0.0, -0.2)
        moved = replace(plume(24, 3), matrix=matrix)
        self.check(s.Scene(volumes=(plume(), moved), lights=(SUN, LAMP)), ambient=.1)

    def test_a_card_inside_the_plume_is_partially_covered(self):
        volume = plume()
        lit = dict(lights=(SUN,))
        only_card = gpu3d.render(s.Scene(geometries=(card(0.0),), **lit), CAMERA, W, H, ambient=0.2)
        both, actual = compare(s.Scene(geometries=(card(0.0),), volumes=(volume,), **lit), ambient=0.2)
        self.assertLess(float(np.abs(actual - both).max()), TOLERANCE * 10)
        smoke = gpu3d.render(s.Scene(volumes=(volume,), **lit), CAMERA, W, H, ambient=0.2, volume=SETTINGS)
        covered = only_card[..., 3] > .99
        self.assertGreater(int(covered.sum()), 500)
        # Where the card is, the plume in front of it changes the pixel, and the plume behind it does not show through.
        front_only = np.ptp(actual[covered][:, :3] - only_card[covered][:, :3])
        self.assertGreater(front_only, 0.02)
        card_pixels = actual[covered]
        self.assertTrue(((card_pixels[:, 3] > 0.99)).all())
        # A ray that crosses the plume both before and after the card sees only the front half.
        behind_card = covered & (smoke[..., 3] > .3)
        self.assertGreater(int(behind_card.sum()), 50)
        self.assertTrue((actual[behind_card][:, :3].sum(axis=1) < smoke[behind_card][:, :3].sum(axis=1)
                         + only_card[behind_card][:, :3].sum(axis=1) + 1e-3).all())

    def test_a_card_in_front_of_and_a_card_behind_the_plume(self):
        volume = plume()
        for z in (1.2, -1.2):
            scene = s.Scene(geometries=(card(z, 3.0),), volumes=(volume,), lights=(SUN,))
            expected, actual = compare(scene, ambient=0.2)
            self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 10)
        front = gpu3d.render(s.Scene(geometries=(card(1.2, 3.0),), volumes=(volume,)), CAMERA, W, H, volume=SETTINGS)
        bare = gpu3d.render(s.Scene(geometries=(card(1.2, 3.0),)), CAMERA, W, H)
        np.testing.assert_allclose(front, bare, atol=1e-6)     # a card in front hides the smoke completely

    def test_bands_split_by_the_work_budget_render_the_same_image(self):
        scene = s.Scene(volumes=(plume(),), lights=(SUN,))
        whole = gpu3d.render(scene, CAMERA, W, H, volume=SETTINGS)
        work = gpuvolume.work_estimate(scene, CAMERA, W, H, SETTINGS, 1)
        kind = gpuvolume.adapter_kind(gpu3d._state())
        with patch.dict(gpuvolume.VOLUME_WORK_BUDGETS, {kind: work / 5}):
            self.assertEqual(gpuvolume.band_plan(gpu3d._state(), work, H), 5)
            banded = gpu3d.render(scene, CAMERA, W, H, volume=SETTINGS)
        np.testing.assert_array_equal(whole, banded)
        with patch.dict(gpuvolume.VOLUME_WORK_BUDGETS, {kind: 1.0}):
            with self.assertRaisesRegex(ValueError, 'exceeds the GPU budget'):
                gpu3d.render(scene, CAMERA, W, H, volume=SETTINGS)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUVolumeResources(unittest.TestCase):
    def test_the_density_texture_is_uploaded_once_per_content(self):
        state = gpu3d._state()
        volume = s.analytic_plume(20, 11)
        scene = s.Scene(volumes=(volume,), lights=(SUN,))
        gpu3d.render(scene, CAMERA, 32, 24, volume=SETTINGS)
        before = gpuvolume.upload_count(state)
        gpu3d.render(scene, CAMERA, 32, 24, volume=SETTINGS)                                  # a re-render
        moved = replace(CAMERA, transform=s.Transform3D(s.Vec3(1.0, 0.6, 3.0)))
        gpu3d.render(scene, moved, 32, 24, volume=SETTINGS)                                   # another camera
        gpu3d.render(scene, CAMERA, 32, 24, volume=replace(SETTINGS, step_size=0.05))         # other knobs
        scrubbed = s.Scene(volumes=(replace(volume, matrix=np.eye(4, dtype=np.float32) * 1.0),), lights=(SUN,))
        gpu3d.render(scrubbed, CAMERA, 32, 24, volume=SETTINGS)                               # same grid, new object
        self.assertEqual(gpuvolume.upload_count(state), before, 'the cached grid must not be uploaded again')
        edited = replace(volume, density=volume.density * 1.5)
        gpu3d.render(s.Scene(volumes=(edited,)), CAMERA, 32, 24, volume=SETTINGS)
        self.assertEqual(gpuvolume.upload_count(state), before + 1)

    def test_an_oversized_grid_is_unsupported(self):
        wide = s.Volume(np.ones((1 << 20, 1, 1), np.float32), 1e-4)
        with self.assertRaisesRegex(gpu3d.Unsupported, '3D texture limit'):
            gpu3d.render(s.Scene(volumes=(wide,)), CAMERA, 16, 16)
        with patch.dict(gpuvolume.VOLUME_MEMORY_BUDGETS, {gpuvolume.adapter_kind(gpu3d._state()): 1000}):
            with self.assertRaisesRegex(gpu3d.Unsupported, 'texture memory'):
                gpu3d.render(s.Scene(volumes=(plume(16),)), CAMERA, 16, 16)

    def test_control_passes_and_splat_scenes_stay_on_the_cpu(self):
        scene = s.Scene(volumes=(plume(8),))
        for output in ('volume_density', 'volume_motion', 'volume_temperature', 'volume_vorticity', 'depth'):
            with self.assertRaisesRegex(gpu3d.Unsupported, 'volumes are CPU-only'):
                gpu3d.render(scene, CAMERA, 16, 16, output=output)
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, CAMERA, 16, 16, mode='raytrace')

    def test_an_empty_frame_and_a_step_that_is_too_fine_are_refused_cleanly(self):
        image = gpu3d.render(s.Scene(volumes=(plume(8),)), s.Camera(s.Transform3D(s.Vec3(0, 0, -5)), s.Vec3(0, 0, -9)),
                             16, 16, volume=SETTINGS)
        self.assertEqual(float(np.abs(image).max()), 0.0)
        with self.assertRaisesRegex(ValueError, 'volume_step_size'):
            gpu3d.render(s.Scene(volumes=(plume(8),)), CAMERA, 16, 16, volume=replace(SETTINGS, step_size=1e-7))


if __name__ == '__main__':
    unittest.main()
