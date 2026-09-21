"""Beauty parity: interior absolute error <=2e-3, silhouette IoU >=.99.

Each comparison prints its measured maximum for adapter-specific validation.
The compute device requests eight storage slots (raster's state requests four).
"""
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch, Mock
import numpy as np
from nodebased import gpurt_render as r, gpurt, gpu3d, scene3d as s
from nodebased.cancellation import Cancelled


def card(z=0, alpha=1, color=(.8, .3, .1), rotation=s.Vec3(), texture=None, size=3):
    return s._card(size, size, (*color, alpha), s.Transform3D(position=s.Vec3(0, 0, z), rotation=rotation), texture)


def gradient():
    y, x = np.mgrid[:64, :64]/63
    return np.stack((x, y, x*y, np.ones_like(x)), axis=-1).astype('f4')


class HostTests(unittest.TestCase):
    def test_capability(self):
        limits = {'max-storage-buffers-per-shader-stage': 8,
                  'max-storage-buffer-binding-size': 1024, 'max-buffer-size': 1024,
                  'max-compute-invocations-per-workgroup': 64,
                  'max-compute-workgroup-size-x': 64, 'max-compute-workgroups-per-dimension': 65535}
        self.assertIsNone(r.check_capability({'limits': limits}))
        for key in limits:
            with self.subTest(key=key):
                self.assertIn(key, r.check_capability({'limits': dict(limits, **{key: 0})}))
        with self.assertRaisesRegex(ValueError, 'needs about.*MiB'):
            gpurt._memory_check({'limits': limits}, 2048)

    def test_preparation_mips_and_clipping(self):
        scene = s.Scene((card(-20, texture=gradient(), size=.5),))
        prepared = r._prepare(scene, s.Camera(), 64, 48)
        self.assertTrue(np.all(prepared[2][:, 4, 1] > 0))
        np.testing.assert_array_equal(prepared[4], np.concatenate([a.reshape(-1, 4) for a in s._mip_chain(gradient())]))
        ground = card(4, rotation=s.Vec3(80, 0, 0), texture=gradient(), size=8)
        at = r._prepare(s.Scene((ground,)), s.Camera(), 64, 48)[2]
        self.assertTrue(np.any(at[:, 5, 3] >= 0))

    def test_unsupported_and_cancel(self):
        g = replace(card(), projection=s.Projection(s.Camera(), gradient()))
        with self.assertRaises(gpu3d.Unsupported):
            r.render_beauty({}, s.Scene((g,)), s.Camera(), 64, 48, (0, 0, 0, 0), 0)
        event = threading.Event(); event.set()
        with self.assertRaises(Cancelled):
            r.render_beauty({}, s.Scene(), s.Camera(), 64, 48, (0, 0, 0, 0), 0, cancel=event)


@unittest.skipUnless(gpu3d.available(), 'wgpu adapter unavailable')
class GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
        if adapter.limits['max-storage-buffers-per-shader-stage'] < 8:
            raise unittest.SkipTest('eight storage bindings unavailable')
        device = adapter.request_device_sync(required_limits={'max-storage-buffers-per-shader-stage': 8})
        cls.state = dict(wgpu=wgpu, device=device)
        reason = r.check_capability(cls.state)
        if reason:
            raise AssertionError(reason)

    def compare(self, scene, camera=None, size=(64, 48), samples=1, background=(0, 0, 0, 0), ambient=.13):
        camera = camera or s.Camera()
        a = r.render_beauty(self.state, scene, camera, *size, background, ambient, samples)
        b = s.render(scene, camera, *size, background, ambient=ambient, samples=samples, mode='raytrace')
        self.assertEqual(a.dtype, np.float32); self.assertFalse(a.flags.writeable)
        # Erode coverage by two pixels, excluding silhouettes, but not internal
        # material boundaries or shared triangle edges.
        coverage = b[..., 3] > 0
        actual = a[..., 3] > 0
        union = (coverage | actual).sum()
        self.assertGreaterEqual((coverage & actual).sum()/max(1, union), .99 if union else 0)
        interior = coverage.copy()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                interior &= np.roll(np.roll(coverage, dy, 0), dx, 1)
        error = np.abs(a-b)[interior].max(initial=0)
        print(f'{self._testMethodName} {size}: interior max={error:.8g}', flush=True)
        self.assertLessEqual(error, 2e-3)
        return a

    def test_unlit_textures_mips_and_empty(self):
        for scene in (s.Scene(), s.Scene((card(),)), s.Scene((card(texture=gradient()),)),
                      s.Scene((card(-20, texture=gradient(), size=2),))):
            self.compare(scene)
        self.compare(s.Scene((card(texture=gradient()),)), size=(67, 51))

    def test_transparency_and_opaque_termination(self):
        for geometries in ([card(z, a) for z, a in zip((0, 1, 2), (.3, .5, .3))],
                           [card(0, .5) for _ in range(12)],
                           [card(-i*.02) for i in range(70)],
                           [card(0, .5, rotation=s.Vec3(0, a, 0)) for a in (-35, 35)]):
            self.compare(s.Scene(tuple(geometries)))

    def test_near_clipped_and_nested(self):
        self.compare(s.Scene((card(4, rotation=s.Vec3(80, 0, 0), texture=gradient(), size=8),)))
        # This near-clipped triangle actually selects two different mip levels.
        ground = card(3, rotation=s.Vec3(35, 0, 0), texture=gradient(), size=8)
        ground = replace(ground, uvs=ground.uvs*16-8)
        at = r._prepare(s.Scene((ground,)), s.Camera(), 64, 48)[2]
        self.assertTrue(np.any((at[:, 5, 3] >= 0) & (at[:, 4, 1] != at[:, 5, 3])))
        self.compare(s.Scene((ground,)))
        parent = s.Transform3D(position=s.Vec3(.2, .1, -.5), rotation=s.Vec3(7, 18, 4), scale=s.Vec3(1.2, .8, 1)).matrix()
        self.compare(s.Scene((replace(card(texture=gradient()), parent=parent),)), camera=s.Camera(roll=13), size=(96, 72))

    def test_lighting_specular_emission_two_sided(self):
        for kind in ('Point', 'Directional'):
            for flip in (0, 180):
                g = replace(card(rotation=s.Vec3(0, flip, 0)), specular=.7, shininess=16, emission=.2)
                self.compare(s.Scene((g,), (s.Light(kind=kind),)))
        self.compare(s.Scene((replace(card(), emission=.3),)))

    def test_shadows(self):
        for alpha in (1, .5):
            ground = card(size=5)
            blocker = replace(card(1, alpha, size=.8), transform=s.Transform3D(position=s.Vec3(.8, 0, 1)))
            for kind in ('Point', 'Directional'):
                light = s.Light(kind=kind, position=s.Vec3(2, 1, 3), shadows=True)
                self.compare(s.Scene((ground, blocker), (light,)), size=(96, 72))
                self.compare(s.Scene((ground, blocker), (light, s.Light(position=s.Vec3(-2, 1, 3), intensity=.4))))
        # Blocker is beyond the point light, so must not attenuate the receiver.
        self.compare(s.Scene((card(size=5), card(4, size=.8)),
                             (s.Light(kind='Point', position=s.Vec3(1, 0, 2), shadows=True),)))

    def test_samples_background_bands_determinism(self):
        scene = s.Scene((card(.3, .5, texture=gradient()), card(-.5, .3)))
        a = self.compare(scene, samples=2, background=(.1, .3, .5, .5))
        with patch.object(r, 'GPU_RT_RAYS_PER_SUBMISSION', 128*3):
            b = self.compare(scene, samples=2, background=(.1, .3, .5, .5))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, self.compare(scene, samples=2, background=(.1, .3, .5, .5)))

    def test_overflow(self):
        scene = s.Scene(tuple(card(-i*.01, .1) for i in range(65)))
        with self.assertRaisesRegex(ValueError, 'Ray-traced render exceeds MAX_HITS_PER_RAY \\(64\\): more than 64 surfaces'):
            r.render_beauty(self.state, scene, s.Camera(), 64, 48, (0, 0, 0, 0), 0)

    def test_cancel_between_bands_destroy_once(self):
        # Wrap the real device: account for every owned allocation and cancel
        # after the first readback. The same device must work afterwards.
        device = self.state['device']; event = threading.Event(); buffers = []
        proxy = Mock(wraps=device); proxy.limits = device.limits
        queue = Mock(wraps=device.queue); proxy.queue = queue
        def upload(**kwargs):
            b = device.create_buffer_with_data(**kwargs)
            original = b.destroy
            b.destroy = Mock(wraps=original)
            buffers.append(b)
            return b
        def read(*args, **kwargs):
            result = device.queue.read_buffer(*args, **kwargs); event.set(); return result
        proxy.create_buffer_with_data.side_effect = upload
        queue.read_buffer.side_effect = read
        state = dict(self.state, device=proxy)
        with patch.object(r, 'GPU_RT_RAYS_PER_SUBMISSION', 64*2):
            with self.assertRaises(Cancelled):
                r.render_beauty(state, s.Scene((card(),)), s.Camera(), 64, 48, (0, 0, 0, 0), 0, cancel=event)
        self.assertEqual(queue.submit.call_count, 1)
        for buffer in buffers:
            buffer.destroy.assert_called_once()
        self.compare(s.Scene((card(),)))


if __name__ == '__main__':
    unittest.main()
