"""CPU-reference comparisons; optional wgpu is never required for this suite."""
from dataclasses import replace
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, scene3d as s


def card(color=(.2, .6, .9, 1), position=s.Vec3(), texture=None):
    return s._card(2.7, 2.1, color, s.Transform3D(position), texture)


def gradient(alpha=False, shape=(32, 64)):
    y, x = np.mgrid[:shape[0], :shape[1]]
    tex = np.stack((x/(shape[1]-1), y/(shape[0]-1), .2+.5*x/(shape[1]-1), np.ones_like(x)), -1).astype('f4')
    if alpha:
        tex[..., 3] = np.where(x < shape[1]//3, 0, np.where(x < shape[1]*2//3, .4, 1))
        tex[..., :3] *= tex[..., 3:4]
    return tex


class NoGPURequired(unittest.TestCase):
    def test_flat_attributes_agree_at_every_triangle_vertex(self):
        ground = s.Geometry(np.array([[-8, -1, 6], [8, -1, 6], [8, -1, -12], [-8, -1, -12]], 'f4'),
                            np.array([[0, 1, 2], [0, 2, 3]], 'i4'), (.3, .6, .2, 1))
        _, _, vertices, _, _ = gpu3d._prepare(s.Scene((ground,)), s.Camera(), 64, 48, None)
        self.assertTrue(vertices)
        for triangle in vertices:
            np.testing.assert_array_equal(triangle[:, 11:16], np.broadcast_to(triangle[0, 11:16], (3, 5)))

    def test_missing_dependency_is_cached_and_clear(self):
        with patch.dict(sys.modules, {'wgpu': None}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
            self.assertFalse(gpu3d.available())
            self.assertFalse(gpu3d.available())
            self.assertIn('wgpu unavailable', gpu3d.describe())
            with self.assertRaisesRegex(RuntimeError, 'wgpu unavailable'):
                gpu3d.render(s.Scene((card(),)), s.Camera(), 64, 48)

    def test_missing_adapter_and_device(self):
        for failure in (None, RuntimeError('device unavailable')):
            from unittest.mock import Mock
            module = Mock()
            if failure is None:
                module.gpu.request_adapter_sync.return_value = None
            else:
                module.gpu.request_adapter_sync.return_value.features = set()
                module.gpu.request_adapter_sync.return_value.request_device_sync.side_effect = failure
            with patch.dict(sys.modules, {'wgpu': module}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
                self.assertFalse(gpu3d.available())
                self.assertIn('unavailable', gpu3d.describe())
                self.assertFalse(gpu3d.available())
                self.assertEqual(module.gpu.request_adapter_sync.call_count, 1)

    def test_downlevel_adapter_without_float32_targets_is_unavailable(self):
        from unittest.mock import Mock
        module = Mock()
        adapter = module.gpu.request_adapter_sync.return_value
        adapter.features = set()
        adapter.limits = {'max-storage-buffer-binding-size': 1 << 27}
        adapter.request_device_sync.return_value.create_texture.side_effect = RuntimeError('downlevel restrictions')
        with patch.dict(sys.modules, {'wgpu': module}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
            self.assertFalse(gpu3d.available())
            self.assertIn('cannot render to rgba32float', gpu3d.describe())
            self.assertFalse(gpu3d.available())
            self.assertEqual(module.gpu.request_adapter_sync.call_count, 1)

    def test_adapter_with_too_few_storage_buffers_is_unavailable(self):
        from unittest.mock import Mock
        for limit in (0, 1):
            module = Mock()
            adapter = module.gpu.request_adapter_sync.return_value
            adapter.features = set()
            adapter.limits = {'max-storage-buffer-binding-size': 1 << 27,
                              'max-storage-buffers-per-shader-stage': limit}
            with patch.dict(sys.modules, {'wgpu': module}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
                self.assertFalse(gpu3d.available())
                reason = 'adapter allows fewer than 2 storage buffers per shader stage'
                self.assertIn(reason, gpu3d.describe())
                with self.assertRaisesRegex(RuntimeError, reason):
                    gpu3d._state()
                self.assertFalse(gpu3d.available())
                self.assertEqual(module.gpu.request_adapter_sync.call_count, 1)
                adapter.request_device_sync.assert_not_called()

    def test_unsupported_without_gpu(self):
        projected = replace(card(), projection=s.Projection(s.Camera(), gradient()))
        with self.assertRaisesRegex(gpu3d.Unsupported, 'projected'):
            gpu3d.render(s.Scene((projected,)), s.Camera(), 64, 48)
        with patch.object(s, 'MAX_TRIANGLES', 1):
            with self.assertRaises(gpu3d.Unsupported):
                gpu3d.render(s.Scene((card(),)), s.Camera(), 64, 48)
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene(), s.Camera(), 64, 48, output='shade')

    def test_cancel_before_device(self):
        from nodebased.imaging import Cancelled
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            gpu3d.render(s.Scene((card(),)), s.Camera(), 64, 48, cancel=event)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUComparison(unittest.TestCase):
    def compare(self, scene, camera=None, **kwargs):
        camera = camera or s.Camera()
        expected = s.render(scene, camera, 64, 48, **kwargs)
        actual = gpu3d.render(scene, camera, 64, 48, **kwargs)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (48, 64, 4))
        self.assertFalse(actual.flags.writeable)
        np.testing.assert_array_equal(actual, gpu3d.render(scene, camera, 64, 48, **kwargs))
        a, b = actual[..., 3] > 0, expected[..., 3] > 0
        union = (a | b).sum()
        self.assertGreater((a & b).sum()/max(union, 1), .99)
        mask = expected[..., 3] == 1
        # Erode by two pixels, with no scipy dependency or wraparound at borders.
        padded = np.pad(mask, 2)
        interior = np.logical_and.reduce([padded[y:y+48, x:x+64] for y in range(5) for x in range(5)])
        self.assertTrue(interior.any(), 'comparison must have opaque interior pixels')
        np.testing.assert_array_equal(actual[interior, 3], 1.0)
        delta = np.abs(actual[interior]-expected[interior])
        if kwargs.get('output') == 'depth':
            self.assertLess(float((delta[:, :3]/np.maximum(expected[interior, :3], 1e-8)).mean()), 1e-3)
        else:
            self.assertLess(float(delta.mean()), 5e-3)
        return actual, expected

    def test_half_precision_fallback(self):
        state = gpu3d._state()
        # Exercise the portable target even on float32-blendable adapters.
        with patch.dict(state, {'format': 'rgba16float', 'pipelines': {}}):
            self.compare(s.Scene((card(),)), background=(.1, .2, .3, 1))
            self.compare(s.Scene((card((.7, .2, .4, .3), s.Vec3(0, 0, 1)), card())))
            self.compare(s.Scene((card(),)), output='depth')

    def test_transformed_normals_and_camera(self):
        sphere = s._sphere(1.1, 32, (.3, .6, .7, 1),
                           s.Transform3D(s.Vec3(), s.Vec3(15, 30, 4), s.Vec3(1.2, .8, 1)))
        camera = s.Camera(s.Transform3D(s.Vec3(1, 1, 5)), roll=13)
        self.compare(s.Scene((sphere,), (s.Light(),)), camera, ambient=.2)
        self.compare(s.Scene((sphere,)), camera, output='normals')
        backface = replace(card(), triangles=np.array([[2, 1, 0], [3, 2, 0]], 'i4'))
        self.compare(s.Scene((backface,), (s.Light(),)), ambient=.2)

    def test_flat_card_cube(self):
        cube = s._cube(1.3, (.8, .3, .1, 1), s.Transform3D(s.Vec3(.65, .1, .6), s.Vec3(10, 23, 7)))
        self.compare(s.Scene((card(position=s.Vec3(-.4, 0, -.5)), cube)))

    def test_textures_and_mips(self):
        for shape in ((32, 64), (129, 257), (1, 64)):
            with self.subTest(shape=shape):
                tex = gradient(shape=shape) if shape[0] > 1 else np.ones((1, 64, 4), 'f4')
                self.compare(s.Scene((card(texture=tex),)))

    def test_lighting(self):
        sphere = s._sphere(1.3, 32, (.7, .4, .2, 1), s.Transform3D())
        lights = (s.Light(intensity=.7), s.Light('Point', (.3, .8, 1), 1.1, s.Vec3(-2, 1, 3)))
        self.compare(s.Scene((sphere,), lights), ambient=.17)
        # Ambient alone must not light an otherwise unlit scene.
        self.compare(s.Scene((sphere,), (s.Light(intensity=0),)), ambient=.9)

    def test_transparency(self):
        scene = s.Scene((card((.8, .1, .3, .35), s.Vec3(.4, .2, 1)),
                         card((.1, .8, .2, .55), s.Vec3(-.3, 0, .5)),
                         card((.2, .3, .8, 1), s.Vec3(0, 0, -.6))))
        a, b = self.compare(scene)
        # Include partially covered transparent regions, not just opaque interiors.
        common = (a[..., 3] > 0) & (b[..., 3] > 0)
        self.assertLess(float(np.abs(a[common]-b[common]).mean()), 5e-3)
        self.compare(s.Scene((card(texture=gradient(True), position=s.Vec3(0, 0, 1)), card())))

    def test_ground_crosses_near(self):
        ground = s.Geometry(np.array([[-8, -1, 6], [8, -1, 6], [8, -1, -12], [-8, -1, -12]], 'f4'),
                            np.array([[0, 1, 2], [0, 2, 3]], 'i4'), (.3, .6, .2, 1))
        a, _ = self.compare(s.Scene((ground,)))
        self.assertGreater(a[-1, :, 3].sum(), 60)

    def test_data_outputs(self):
        scene = s.Scene((s._sphere(1.3, 32, (.7, .4, .2, .2), s.Transform3D()),
                         card(position=s.Vec3(0, 0, -1))))
        for output in ('depth', 'normals'):
            with self.subTest(output=output):
                self.compare(scene, output=output, samples=4, background=(1, 0, 1, 1))
                self.compare(s.Scene((card(texture=gradient(True)),)), output=output)

    def test_supersampling_and_background(self):
        self.compare(s.Scene((card(),)), samples=2)
        self.compare(s.Scene((card((.3, .7, .2, .4)),)), samples=2, background=(.1, .3, .6, 1))
        for output in ('rgba', 'depth', 'normals'):
            a = gpu3d.render(s.Scene(), s.Camera(), 64, 48, background=(.8, .4, .2, .5), output=output)
            np.testing.assert_array_equal(a, s.render(s.Scene(), s.Camera(), 64, 48, background=(.8, .4, .2, .5), output=output))
            self.assertFalse(a.flags.writeable)


if __name__ == '__main__':
    unittest.main()
