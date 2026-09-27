"""Lane L4 plan 3, step C part 3: the volume control passes on the GPU, same names and conventions as the CPU."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, gpuvolume, scene3d as s, volumerender
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_volume_scene import make, wire

W, H = 96, 72
CAMERA = s.Camera(s.Transform3D(s.Vec3(0.3, 0.6, 3.2)), s.Vec3(0, 0.5, 0))
SETTINGS = volumerender.VolumeSettings(step_size=0.03, density_scale=6.0, motion_blur=1.5, motion_samples=5)
ALL = s.VOLUME_OUTPUTS + ('depth',)


def moving_plume(n=24, seed=0, shift=(0.0, 0.0, 0.0)):
    plume = s.analytic_plume(n, seed)
    rng = np.random.default_rng(seed + 1)
    velocity = np.zeros((n,) * 3 + (3,), np.float32)
    velocity[..., 0], velocity[..., 1] = 1.5, 2.0
    velocity[..., 2] = rng.normal(0, .3, (n,) * 3)
    temperature = np.linspace(0, 3, n ** 3, dtype=np.float32).reshape((n,) * 3)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = shift
    return replace(plume, velocity=velocity, temperature=temperature, matrix=matrix)


def card(z, width=1.2):
    return s._card(width, width, (.8, .3, .2, 1.0), s.Transform3D(s.Vec3(0, .5, z)))


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUPassParity(unittest.TestCase):
    def check(self, scene, names=ALL, atol=2e-3):
        for name in names:
            cpu = s.render(scene, CAMERA, W, H, output=name, volume=SETTINGS)
            gpu = gpu3d.render(scene, CAMERA, W, H, output=name, volume=SETTINGS)
            self.assertEqual(gpu.shape, cpu.shape)
            self.assertEqual(gpu.dtype, np.float32)
            self.assertGreater(float(np.abs(cpu).max()), 0.1, f'{name} must show something')
            tolerance = atol * max(1.0, float(np.abs(cpu).max()))
            self.assertLess(float(np.abs(gpu - cpu).max()), tolerance, name)

    def test_every_pass_matches_the_cpu_reference(self):
        self.check(s.Scene(volumes=(moving_plume(),)))

    def test_a_card_cuts_the_passes_and_the_depth_like_the_cpu(self):
        for z, width in ((0.0, 0.6), (1.2, 0.5)):
            scene = s.Scene(geometries=(card(z, width),), volumes=(moving_plume(),))
            self.check(scene)
            bare = s.render(s.Scene(volumes=(moving_plume(),)), CAMERA, W, H, output='volume_density', volume=SETTINGS)
            cut = gpu3d.render(scene, CAMERA, W, H, output='volume_density', volume=SETTINGS)
            self.assertLess(float(cut.sum()), 0.95 * float(bare.sum()), 'the card must hide some of the smoke')

    def test_two_transformed_volumes_match_and_carry_their_own_velocities(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.array([[0.8, 0, 0.6], [0, 1.4, 0], [-0.6, 0, 0.8]], np.float32)   # rotated: velocity turns too
        second = replace(moving_plume(20, 3), matrix=matrix)
        self.check(s.Scene(volumes=(moving_plume(), second)))

    def test_the_motion_pass_ignores_the_shutter_and_the_others_use_it(self):
        scene = s.Scene(volumes=(moving_plume(),))
        sharp = replace(SETTINGS, motion_blur=0.0)
        np.testing.assert_array_equal(gpu3d.render(scene, CAMERA, W, H, output='volume_motion', volume=SETTINGS),
                                      gpu3d.render(scene, CAMERA, W, H, output='volume_motion', volume=sharp))
        blurred = gpu3d.render(scene, CAMERA, W, H, output='volume_density', volume=SETTINGS)
        self.assertGreater(float(np.abs(blurred - gpu3d.render(scene, CAMERA, W, H, output='volume_density', volume=sharp)).max()), 0.05)

    def test_passes_are_deterministic_and_band_splitting_changes_nothing(self):
        scene = s.Scene(volumes=(moving_plume(),))
        for name in ALL:
            one = gpu3d.render(scene, CAMERA, W, H, output=name, volume=SETTINGS)
            np.testing.assert_array_equal(one, gpu3d.render(scene, CAMERA, W, H, output=name, volume=SETTINGS))
        work = gpuvolume.work_estimate(scene, CAMERA, W, H, SETTINGS, 0)
        kind = gpuvolume.adapter_kind(gpu3d._state())
        whole = gpu3d.render(scene, CAMERA, W, H, output='volume_temperature', volume=SETTINGS)
        with patch.dict(gpuvolume.VOLUME_WORK_BUDGETS, {kind: work / 4}):
            banded = gpu3d.render(scene, CAMERA, W, H, output='volume_temperature', volume=SETTINGS)
        np.testing.assert_array_equal(whole, banded)
        with patch.dict(gpuvolume.VOLUME_WORK_BUDGETS, {kind: 1.0}):
            with self.assertRaisesRegex(ValueError, 'exceeds the GPU budget'):
                gpu3d.render(scene, CAMERA, W, H, output='volume_density', volume=SETTINGS)

    def test_volume_id_numbers_the_members_and_the_nearest_wins(self):
        left = moving_plume(16, 1, shift=(-0.9, 0, 0))
        right = moving_plume(16, 2, shift=(0.9, 0, 0))
        behind = moving_plume(16, 3, shift=(0.9, 0, -0.3))             # overlaps `right`, further from the camera
        scene = s.Scene(volumes=(left, right, behind))
        cpu = s.render(scene, CAMERA, W, H, output='volume_id', volume=SETTINGS)
        gpu = gpu3d.render(scene, CAMERA, W, H, output='volume_id', volume=SETTINGS)
        np.testing.assert_array_equal(gpu[..., :3], cpu[..., :3])
        np.testing.assert_array_equal(cpu[..., 0], cpu[..., 1])
        self.assertTrue({1.0, 2.0} <= set(np.unique(cpu[..., 0])))
        np.testing.assert_array_equal(cpu[..., 3], (cpu[..., 0] > 0).astype(np.float32))
        alone = s.render(s.Scene(volumes=(right,)), CAMERA, W, H, output='volume_id', volume=SETTINGS)[..., 0] > 0
        self.assertTrue((cpu[alone][:, 0] == 2.0).all(), 'the nearer of two overlapping volumes takes the pixel')
        # Remove `right`: the volume behind it takes over its pixels.
        only = s.render(s.Scene(volumes=(left, behind)), CAMERA, W, H, output='volume_id', volume=SETTINGS)
        self.assertIn(2.0, np.unique(only[..., 0]))   # `behind` is now the second member
        # A card in front removes the pixels it hides.
        cut = gpu3d.render(s.Scene(geometries=(card(2.0, 6.0),), volumes=(left, right)), CAMERA, W, H,
                           output='volume_id', volume=SETTINGS)
        self.assertEqual(float(np.abs(cut).max()), 0.0)

    def test_ray_traced_and_splat_scenes_still_go_to_the_cpu(self):
        scene = s.Scene(volumes=(moving_plume(8),))
        for name in ALL:
            with self.assertRaises(gpu3d.Unsupported):
                gpu3d.render(scene, CAMERA, 16, 16, output=name, mode='raytrace')


class CPUId(unittest.TestCase):
    def test_volume_id_is_a_named_pass_and_numbers_from_one(self):
        self.assertIn('volume_id', s.VOLUME_OUTPUTS)
        self.assertIn('volume_id', s.MULTICHANNEL_PASSES)
        a, b = moving_plume(12, 1, (-0.9, 0, 0)), moving_plume(12, 2, (0.9, 0, 0))
        image = s.render(s.Scene(volumes=(a, b)), CAMERA, 48, 36, output='volume_id', volume=SETTINGS)
        self.assertEqual(set(np.unique(image[..., 0])), {0.0, 1.0, 2.0})
        self.assertEqual(float(image[0, 0, 3]), 0.0)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class MultichannelExr(unittest.TestCase):
    NAMES = ('volume_density', 'volume_motion', 'volume_temperature', 'volume_vorticity', 'volume_id')

    def test_the_gpu_passes_land_in_the_exr_within_tolerance_of_the_cpu(self):
        import OpenImageIO as oiio
        from nodebased import media
        scene = s.Scene(volumes=(moving_plume(),))
        passes = 'beauty,depth,' + ','.join(self.NAMES)
        beauty, gpu_layers = s.render_multichannel(scene, CAMERA, W, H, passes=passes, volume=SETTINGS, backend='gpu')
        _, cpu_layers = s.render_multichannel(scene, CAMERA, W, H, passes=passes, volume=SETTINGS, backend='cpu')
        for name in self.NAMES:
            self.assertLess(float(np.abs(gpu_layers[name] - cpu_layers[name]).max()),
                            2e-3 * max(1.0, float(np.abs(cpu_layers[name]).max())), name)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'smoke.exr'
            media.write_exr(path, beauty, bits='float', layers=gpu_layers)
            source = oiio.ImageInput.open(str(path))
            try:
                names = set(source.spec().channelnames)
                pixels = source.read_image('float')
                channels = list(source.spec().channelnames)
            finally:
                source.close()
        for name in ('volume_density.R', 'volume_temperature.B', 'volume_motion.X', 'volume_motion.Y',
                     'volume_vorticity.G', 'volume_id.R', 'depth.Z'):
            self.assertIn(name, names)
        self.assertNotIn('volume_id.G', names)
        written = pixels[..., channels.index('volume_density.R')]
        np.testing.assert_allclose(written, gpu_layers['volume_density'][..., 0], atol=1e-6)

    def test_render3d_multichannel_runs_the_volume_passes_on_the_gpu_and_auto_falls_back(self):
        def graph(backend):
            d = Dispatcher()
            make(d, p=('Plume3D', {'plume_resolution': 12}), s=('Scene3D', {}),
                 c=('Camera3D', {'ty': .5, 'tz': 3, 'target_y': .5}),
                 r=('Render3D', {'width': 32, 'height': 24, 'render_output': 'multichannel',
                                 'passes': 'beauty,volume_density,volume_id', 'volume_density_scale': 3.0,
                                 'render_backend': backend}))
            wire(d, 's', 'object0', 'p')
            wire(d, 'r', 'scene', 's')
            wire(d, 'r', 'camera', 'c')
            return Evaluator().evaluate_raster(dict(d.document, view='r'), frame=1)
        gpu, cpu = graph('gpu'), graph('cpu')
        for name in ('volume_density', 'volume_id'):
            np.testing.assert_allclose(gpu.layers[name].pixels, cpu.layers[name].pixels, atol=2e-3)
        self.assertGreater(float(gpu.layers['volume_density'].pixels[..., 0].max()), 0.1)
        auto = graph('auto')
        np.testing.assert_allclose(auto.layers['volume_density'].pixels, cpu.layers['volume_density'].pixels, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
