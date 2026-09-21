"""GPU ellipsoid shadows: interior tolerance 2e-3, measured per adapter."""
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch, Mock
import numpy as np
from nodebased import gpu3d, gpurt_render as r, scene3d as s, splats, gpusplat
from nodebased.cancellation import Cancelled
from tests.test_3d_gpu_rt_integration import GraphFixture


def instance(position=(0, 0, 1), scales=(.3, .3, .15), **kwargs):
    cloud = splats.SplatCloud(np.array([position]), np.array([scales]),
        np.array([[1., 0, 0, 0]]), np.array([.7]), np.zeros((1, 1, 3)), 0,
        colorspace='linear')
    return s.SplatInstance(cloud, **kwargs)


def card(z=0, size=6):
    return s._card(size, size, (.7, .5, .3, 1), s.Transform3D(s.Vec3(0, 0, z)))


def scene(inst=None, kind='Directional'):
    return s.Scene((card(),), (s.Light(kind=kind, position=s.Vec3(1, 0, 3),
        shadows=True),), (inst or instance(),))


class HostTests(GraphFixture, unittest.TestCase):
    def test_fallbacks(self):
        base = scene()
        cases = [(replace(base, splats=(instance(relight=1),)), 'rgba', 'splat shadows'),
                 (replace(base, splats=(instance(relight=1, shadow_catch=1),)), 'rgba', 'caught splat shadows'),
                 (replace(base, geometries=(replace(card(), color=(1, 1, 1, .5)),)), 'rgba', 'transparent meshes'),
                 (replace(base, geometries=(replace(card(), projection=s.Projection(s.Camera(), np.ones((2, 2, 4), 'f4'))),)), 'rgba', 'transparent meshes'),
                 (base, 'depth', 'splat data passes'), (base, 'splats', 'splats output')]
        for value, output, reason in cases:
            with self.subTest(reason=reason), patch.object(s, 'scene_from_node', return_value=value), patch.object(gpu3d, 'available', return_value=True):
                self.set('render', 'render_output', output)
                self.fallback('unsupported: .*'+reason)

    def test_no_adapter_and_capability(self):
        with patch.object(s, 'scene_from_node', return_value=scene()):
            with patch.object(gpu3d, 'available', return_value=False):
                self.fallback('unavailable')
            with patch.object(gpu3d, 'available', return_value=True), patch.object(gpu3d, '_state', return_value={}):
                self.fallback('unsupported: .*max-storage')

    def test_packing_matches_cpu_and_memory(self):
        value = replace(scene(), splats=(instance(scale_scale=-2, opacity_scale=.4), instance(cast_shadows=False)))
        state = {'limits': {'max-buffer-size': 2**30, 'max-storage-buffer-binding-size': 2**30}}
        packed, offset = r._pack_casters(state, value)
        cpu = s._SplatCasters(value.splats)
        records = packed[offset:].reshape(-1, 4, 4)
        np.testing.assert_array_equal(records[:, 0, :3], cpu.primitives.positions[cpu.bvh.prim_order].astype('f4'))
        np.testing.assert_array_equal(records[:, 0, 3], cpu.primitives.opacity[cpu.bvh.prim_order].astype('f4'))
        expected_rows = (cpu.primitives.rotations_matrix.transpose(0, 2, 1) /
                         np.maximum(cpu.primitives.scales[:, :, None], 1e-30))
        np.testing.assert_array_equal(records[:, 1:, :3], expected_rows[cpu.bvh.prim_order].astype('f4'))
        self.assertEqual(len(records), 1)
        with patch.object(gpusplat, 'GPU_SPLAT_MEMORY_CAP', 1):
            with self.assertRaisesRegex(ValueError, r'needs about .* MiB for 1 splat casters.*allows \(.* MiB, binding limit .* MiB\)'):
                r._pack_casters(state, value)
        for value in (replace(value, lights=(s.Light(shadows=False),)), replace(value, splats=(instance(cast_shadows=False),))):
            with patch.object(s, '_SplatCasters', side_effect=AssertionError('no caster construction')):
                self.assertEqual(r._pack_casters(state, value)[1], 0)


@unittest.skipUnless(gpu3d.available(), 'wgpu adapter unavailable')
class GPU(unittest.TestCase):
    def compare(self, value, size=(64, 48), **kwargs):
        a = gpu3d.render(value, s.Camera(), *size, mode='raytrace', **kwargs)
        b = s.render(value, s.Camera(), *size, mode='raytrace', **kwargs)
        # These cards fill the image, so every pixel is interior.
        error = np.abs(a-b).max(initial=0)
        print(f'{self._testMethodName} {size}: max={error:.9g}', flush=True)
        self.assertLessEqual(error, 2e-3)
        self.assertEqual(a.dtype, np.float32)
        self.assertFalse(a.flags.writeable)
        return a

    def test_lights_mesh_product_and_depth(self):
        for kind in ('Point', 'Directional'):
            value = scene(kind=kind)
            self.compare(value)
            self.compare(replace(value, geometries=(card(), card(2, .5))), (96, 72))
            self.compare(replace(value, splats=(), geometries=(card(), card(2, .5))))
        self.compare(scene(), (67, 51), samples=2, background=(.1, .2, .3, .5))
        self.compare(s.Scene(splats=(instance(),)), (67, 51))
        self.compare(replace(scene(), splats=(instance(relight=.5),), lights=(s.Light(),)))

    def test_analytic_shadow(self):
        value = replace(scene(), lights=(s.Light(position=s.Vec3(0, 0, 3), shadows=True),))
        # Hide only the visible layer to measure mesh illumination directly.
        zeros = (np.zeros((48, 64, 3), 'f4'), np.zeros((48, 64), 'f4'))
        with patch.object(gpusplat, 'render_layer', return_value=zeros):
            actual = gpu3d.render(value, s.Camera(), 64, 48, mode='raytrace')
        from nodebased import gpurt
        o, d, _, _ = gpurt.primary_rays(s.Camera(), 64, 48)
        positions = o+5*d
        d2 = np.sum((positions[:, :2]/.3)**2, axis=1)
        factor = np.where(d2 <= 9, 1-np.minimum(.99, .7*np.exp(-.5*d2)), 1)
        expected = factor.reshape(48, 64, 1)*np.array((.7, .5, .3))
        np.testing.assert_allclose(actual[..., :3], expected, atol=2e-3, rtol=0)
        self.assertLess(actual[24, 32, 0], .3)
        self.assertGreater(actual[24, 40, 0], actual[24, 32, 0])

    def test_grazing(self):
        # Thin large ellipsoid; near-parallel rays amplify f32 whitening error.
        # llvmpipe maximum 1.79e-7; retain 2e-3 for hardware FMA differences.
        value = scene(instance((.3, 0, .02), scales=(1.5, 1.5, .002)))
        value = replace(value, lights=(s.Light(position=s.Vec3(10, 0, .17), intensity=60, shadows=True),))
        actual = self.compare(value, (96, 72))
        off = replace(value, splats=(replace(value.splats[0], cast_shadows=False),))
        unshadowed = gpu3d.render(off, s.Camera(), 96, 72, mode='raytrace')
        self.assertGreater(np.abs(actual-unshadowed).max(), .01)

    def test_cast_switch_scaling_and_light_limit(self):
        value = scene(instance(cast_shadows=False))
        a = self.compare(value)
        with patch.object(r, '_pack_casters', return_value=(np.zeros((1, 4), 'f4'), 0)):
            np.testing.assert_array_equal(a, gpu3d.render(value, s.Camera(), 64, 48, mode='raytrace'))
        self.compare(replace(value, splats=(instance(scale_scale=1.7, opacity_scale=.4), instance((1, 0, 1), cast_shadows=False))))
        value = scene(instance((0, 0, 5)), kind='Point')
        a = self.compare(value)
        b = gpu3d.render(replace(value, splats=(replace(value.splats[0], cast_shadows=False),)), s.Camera(), 64, 48, mode='raytrace')
        np.testing.assert_array_equal(a, b)
        value = replace(scene(), lights=(s.Light(shadows=False),))
        with patch.object(s, '_SplatCasters', side_effect=AssertionError('no upload')):
            gpu3d.render(value, s.Camera(), 64, 48, mode='raytrace')

    def test_bands_and_cancel(self):
        value = scene()
        a = self.compare(value)
        with patch.object(r, 'GPU_RT_RAYS_PER_SUBMISSION', 64*2):
            np.testing.assert_array_equal(a, self.compare(value))
        event = threading.Event()
        state = gpu3d._state(None)
        device = state['device']; proxy = Mock(wraps=device); proxy.limits = device.limits
        queue = Mock(wraps=device.queue); proxy.queue = queue
        def read(*args, **kwargs):
            result = device.queue.read_buffer(*args, **kwargs); event.set(); return result
        queue.read_buffer.side_effect = read
        with patch.object(r, 'GPU_RT_RAYS_PER_SUBMISSION', 64*2), self.assertRaises(Cancelled):
            r.render_beauty(dict(state, device=proxy), value, s.Camera(), 64, 48, (0, 0, 0, 0), 0, cancel=event)
        self.assertEqual(queue.submit.call_count, 1)

    def test_memory_fallback(self):
        fixture = HostTests(); fixture.setUp()
        with patch.object(s, 'scene_from_node', return_value=scene()), patch.object(gpusplat, 'GPU_SPLAT_MEMORY_CAP', 1):
            fixture.fallback('GPU Render3D failed: GPU ray tracing needs about .*binding limit')
