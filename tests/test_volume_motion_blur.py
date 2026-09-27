"""Lane L4 plan 3, step C part 1: volume motion blur from the velocity field, on the CPU reference and the GPU."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, scene3d as s, volumerender
from nodebased.core import SPECS, LIMITS
from nodebased.imaging import _volume_settings

N = 32
FPS = 24.0
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 3)), s.Vec3(0, 0, 0))
SIZE = 96
SUN = s.Light('Directional', position=s.Vec3(4, 5, 3), target=s.Vec3())


def blob(speed=1.5, temperature=False):
    """A cube of density 1 in the middle of the box, every cell moving `speed` object units per second along +x."""
    density = np.zeros((N,) * 3, np.float32)
    density[12:20, 12:20, 12:20] = 1.0
    velocity = np.zeros((N,) * 3 + (3,), np.float32)
    velocity[..., 0] = speed
    return s.Volume(density, 1.0 / N, (-.5, -.5, -.5), velocity=velocity,
                    temperature=np.full((N,) * 3, 2.0, np.float32) if temperature else None)


def settings(**kwargs):
    return volumerender.VolumeSettings(step_size=0.02, density_scale=4.0, shadow_steps=8, fps=FPS, **kwargs)


def density_row(volume, blur, samples=16, size=SIZE):
    image = s.render(s.Scene(volumes=(volume,)), CAMERA, size, size, output='volume_density',
                     volume=settings(motion_blur=blur, motion_samples=samples))
    return image


def pixels_per_unit():
    a, _ = s.project(CAMERA, SIZE, SIZE, np.array([[0, 0, 0], [1, 0, 0]], np.float32))
    return float(a[1][0] - a[0][0])


class CPUMotionBlur(unittest.TestCase):
    def test_a_moving_blob_streaks_along_its_motion_by_speed_times_shutter(self):
        volume = blob(speed=3.0)
        sharp = density_row(volume, 0.0)
        for frames in (1.0, 2.0):
            blurred = density_row(volume, frames)
            xs = np.arange(SIZE)

            def stats(image):
                column = image[..., 0].sum(axis=0)
                covered = np.nonzero(column > 0.02 * column.max())[0]
                return covered.max() - covered.min() + 1, float((xs * column).sum() / column.sum())
            width0, centre0 = stats(sharp)
            width1, centre1 = stats(blurred)
            shift = 3.0 * frames / FPS * pixels_per_unit()          # object units are world units here
            self.assertGreater(shift, 4.0, 'the streak must be several pixels long')
            self.assertAlmostEqual(width1 - width0, shift, delta=2.0)
            self.assertAlmostEqual(centre1 - centre0, shift / 2, delta=1.0)   # the shutter opens forward
            rows0 = np.nonzero(sharp[..., 0].sum(axis=1) > 0.02)[0]
            rows1 = np.nonzero(blurred[..., 0].sum(axis=1) > 0.02)[0]
            self.assertLessEqual(abs(len(rows1) - len(rows0)), 1)           # no blur across the motion
            # Blur redistributes the density; it does not add any (mass is conserved within the box).
            self.assertAlmostEqual(float(blurred[..., 0].sum()), float(sharp[..., 0].sum()), delta=0.03 * float(sharp[..., 0].sum()))

    def test_no_shutter_no_velocity_and_zero_speed_leave_the_image_alone(self):
        volume = blob()
        base = density_row(volume, 0.0)
        np.testing.assert_array_equal(density_row(replace(volume, velocity=None), 3.0), base)
        np.testing.assert_allclose(density_row(blob(speed=0.0), 3.0), base, atol=1e-6)

    def test_the_motion_pass_carries_unblurred_vectors(self):
        volume = blob(speed=3.0)
        scene = s.Scene(volumes=(volume,))
        sharp = s.render(scene, CAMERA, SIZE, SIZE, output='volume_motion', volume=settings())
        blurred = s.render(scene, CAMERA, SIZE, SIZE, output='volume_motion', volume=settings(motion_blur=4.0))
        np.testing.assert_array_equal(sharp, blurred)
        self.assertGreater(float(np.abs(sharp[..., 0]).max()), 1.0)

    def test_beauty_is_deterministic_and_smears(self):
        scene = s.Scene(volumes=(blob(speed=3.0),), lights=(SUN,))
        kwargs = dict(ambient=0.2, volume=settings(motion_blur=2.0, motion_samples=6))
        one = s.render(scene, CAMERA, SIZE, SIZE, **kwargs)
        np.testing.assert_array_equal(one, s.render(scene, CAMERA, SIZE, SIZE, **kwargs))
        sharp = s.render(scene, CAMERA, SIZE, SIZE, ambient=0.2, volume=settings())
        self.assertGreater(float(np.abs(one - sharp).max()), 0.05)

    def test_settings_validate_and_the_knobs_exist(self):
        with self.assertRaises(ValueError):
            settings(motion_blur=-1.0).validated()
        with self.assertRaises(ValueError):
            settings(motion_samples=0).validated()
        params = SPECS['Render3D']['params']
        self.assertEqual((params['volume_motion_blur'], params['volume_motion_samples']), (0.0, 8))
        self.assertIn('volume_motion_blur', LIMITS)
        made = _volume_settings(dict(params, volume_motion_blur=1.5, volume_motion_samples=5))
        self.assertEqual((made.motion_blur, made.motion_samples), (1.5, 5))


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUMotionBlur(unittest.TestCase):
    def test_the_gpu_beauty_and_density_match_the_cpu_with_a_shutter(self):
        plume = s.analytic_plume(24, 0)
        rng = np.random.default_rng(1)
        velocity = np.zeros((24,) * 3 + (3,), np.float32)
        velocity[..., 0], velocity[..., 1] = 1.5, 2.0
        velocity[..., 2] = rng.normal(0, .2, (24,) * 3)
        volume = replace(plume, velocity=velocity)
        camera = s.Camera(s.Transform3D(s.Vec3(0.3, 0.6, 3.2)), s.Vec3(0, 0.5, 0))
        st = volumerender.VolumeSettings(step_size=0.03, density_scale=8.0, shadow_steps=12, motion_blur=2.0,
                                         motion_samples=6)
        scene = s.Scene(volumes=(volume,), lights=(SUN,))
        cpu = s.render(scene, camera, 96, 72, ambient=0.2, volume=st)
        gpu = gpu3d.render(scene, camera, 96, 72, ambient=0.2, volume=st)
        self.assertLess(float(np.abs(cpu - gpu).max()), 2e-3)
        sharp = gpu3d.render(scene, camera, 96, 72, ambient=0.2, volume=replace(st, motion_blur=0.0))
        self.assertGreater(float(np.abs(gpu - sharp).max()), 0.02, 'the shutter must change the GPU image')
        np.testing.assert_array_equal(gpu, gpu3d.render(scene, camera, 96, 72, ambient=0.2, volume=st))
        cpu_density = s.render(scene, camera, 96, 72, output='volume_density', volume=st)
        gpu_density = gpu3d.render(scene, camera, 96, 72, output='volume_density', volume=st)
        self.assertLess(float(np.abs(cpu_density - gpu_density).max()), 1e-3)

    def test_the_gpu_streak_length_follows_the_velocity_and_the_shutter(self):
        volume = blob(speed=3.0)
        scene = s.Scene(volumes=(volume,))
        widths = []
        for frames in (0.0, 2.0):
            image = gpu3d.render(scene, CAMERA, SIZE, SIZE, output='volume_density',
                                 volume=settings(motion_blur=frames, motion_samples=16))
            column = image[..., 0].sum(axis=0)
            covered = np.nonzero(column > 0.02 * column.max())[0]
            widths.append(covered.max() - covered.min() + 1)
        self.assertAlmostEqual(widths[1] - widths[0], 3.0 * 2.0 / FPS * pixels_per_unit(), delta=2.0)


if __name__ == '__main__':
    unittest.main()
