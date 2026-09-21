"""Ray-traced Render3D backend routing, parity, and fallback contracts."""
from dataclasses import replace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nodebased import gpu3d, gpurt, gpurt_render, scene3d as s, splats
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator, Cancelled


def card(z=0, alpha=1):
    return s._card(3, 3, (.7, .3, .1, alpha),
                   s.Transform3D(position=s.Vec3(0, 0, z)))


class GraphFixture:
    def setUp(self):
        self.d = Dispatcher()
        for key, kind, params in (
                ('card', 'Card3D', dict(red=.7, green=.3, blue=.1)),
                ('scene', 'Scene3D', {}), ('camera', 'Camera3D', {}),
                ('render', 'Render3D', dict(width=48, height=36, samples=2,
                                           render_mode='raytrace'))):
            self.create(key, kind, **params)
        self.connect('scene', 'object0', 'card')
        self.connect('render', 'scene', 'scene')
        self.connect('render', 'camera', 'camera')

    def create(self, key, kind, **params):
        self.d.execute(dict(op='create', id=key, type=kind, params=params))

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def set(self, key, param, value):
        self.d.execute(dict(op='set', id=key, param=param, value=value))

    def render(self, backend, cancel=None):
        self.set('render', 'render_backend', backend)
        result = Evaluator().evaluate(self.d.document, 'render', cancel=cancel)
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(result.flags.writeable)
        return result

    def fallback(self, reason):
        expected = self.render('cpu')
        with patch.object(s, 'render', wraps=s.render) as cpu:
            self.assertTrue(np.array_equal(self.render('auto'), expected))
            self.assertTrue(cpu.called)
        with self.assertRaisesRegex(ValueError, reason):
            self.render('gpu')


class RoutingTests(GraphFixture, unittest.TestCase):
    def test_no_adapter(self):
        with patch.object(gpu3d, 'available', return_value=False), \
                patch.object(gpu3d, 'describe', return_value='no test adapter'):
            self.fallback('GPU Render3D unavailable: no test adapter')

    def test_unsupported_splats_projection_and_outputs(self):
        cloud = splats.SplatCloud(np.zeros((1, 3)), np.full((1, 3), .15),
            np.array([[1, 0, 0, 0]]), np.array([.8]), np.zeros((1, 1, 3)), 0)
        cases = (
            (s.Scene(splats=(s.SplatInstance(cloud),)), 'rgba',
             'splats in ray-traced mode are CPU-only for now'),
            (s.Scene((replace(card(), projection=s.Projection(
                s.Camera(), np.ones((2, 2, 4), 'f4'))),)), 'rgba',
             'Camera-projected geometry is not implemented by wgpu'),
            (s.Scene((card(),)), 'depth',
             'ray-traced mode on the GPU renders rgba only for now'))
        for scene, output, reason in cases:
            with self.subTest(output=output, reason=reason), \
                    patch.object(s, 'scene_from_node', return_value=scene), \
                    patch.object(gpu3d, 'available', return_value=True), \
                    patch.object(gpu3d, '_state', side_effect=AssertionError('must reject before device')):
                self.set('render', 'render_output', output)
                self.fallback('GPU Render3D unsupported: ' + reason)

    def test_capability_fallback(self):
        with patch.object(gpu3d, 'available', return_value=True), \
                patch.object(gpu3d, '_state', return_value={}), \
                patch.object(gpurt_render, 'check_capability', return_value='needs 7 bindings'):
            self.fallback('GPU Render3D unsupported: needs 7 bindings')

    def test_cancellation(self):
        event = threading.Event()
        event.set()
        for backend in ('gpu', 'auto'):
            with self.subTest(backend=backend), self.assertRaises(Cancelled):
                self.render(backend, event)
        with patch.object(gpu3d, '_state', side_effect=AssertionError('must cancel first')):
            with self.assertRaises(Cancelled):
                gpu3d.render(s.Scene((card(),)), s.Camera(), 16, 12,
                             mode='raytrace', cancel=event)
        with patch.object(gpu3d, 'available', return_value=True), \
                patch.object(gpu3d, '_state', return_value={}), \
                patch.object(gpurt_render, 'check_capability', return_value=None), \
                patch.object(gpurt_render, 'render_beauty', side_effect=Cancelled()), \
                patch.object(s, 'render', side_effect=AssertionError('must not fall back')):
            with self.assertRaises(Cancelled):
                self.render('auto')

    def test_state_requests_available_bindings_without_disabling_raster(self):
        import sys
        for slots in (2, 4, 6, 7, 8, 12):
            with self.subTest(slots=slots):
                adapter = Mock()
                adapter.limits = {'max-storage-buffers-per-shader-stage': slots,
                                  'max-storage-buffer-binding-size': 1024}
                adapter.features = []
                adapter.info = {}
                wgpu = Mock()
                wgpu.gpu.request_adapter_sync.return_value = adapter
                wgpu.TextureUsage.RENDER_ATTACHMENT = 1
                wgpu.TextureUsage.COPY_SRC = 2
                with patch.dict(sys.modules, {'wgpu': wgpu}), \
                        patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
                    gpu3d._state()
                requested = adapter.request_device_sync.call_args.kwargs['required_limits']
                self.assertEqual(requested['max-storage-buffers-per-shader-stage'], min(8, slots))
                self.assertEqual(gpu3d._bvh_capable(adapter.limits, 48, 4), slots >= 4)
                limits = dict(adapter.limits, **{'max-buffer-size': 1024,
                    'max-compute-invocations-per-workgroup': 64,
                    'max-compute-workgroup-size-x': 64,
                    'max-compute-workgroups-per-dimension': 65535})
                self.assertEqual(gpurt_render.check_capability({'limits': limits}) is None, slots >= 7)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUIntegrationTests(GraphFixture, unittest.TestCase):
    def parity(self):
        expected = self.render('cpu')
        coverage = expected[..., 3] > 0
        padded = np.pad(coverage, 2)
        interior = np.logical_and.reduce([
            padded[y:y+36, x:x+48] for y in range(5) for x in range(5)])
        self.assertTrue(interior.any())
        for backend in ('gpu', 'auto'):
            with self.subTest(backend=backend), \
                    patch.object(gpurt_render, 'render_beauty', wraps=gpurt_render.render_beauty) as render, \
                    patch.object(s, 'render', side_effect=AssertionError('unexpected CPU fallback')):
                actual = self.render(backend)
                render.assert_called_once()
                self.assertEqual(render.call_args.args[7], 2)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual[interior], expected[interior], atol=2e-3, rtol=0)

    def test_textured_transparent_lit_shadow_specular_emission(self):
        self.create('texture', 'Checker', width=32, height=32)
        self.connect('card', 'image', 'texture')
        self.parity()
        self.create('front', 'Card3D', tz=.8, alpha=.4, card_width=.8,
                    card_height=.8, red=.2, green=.7, blue=.4)
        self.connect('scene', 'object1', 'front')
        self.parity()
        self.create('light', 'Light3D', light_type='Point', tx=2, ty=1, tz=3, shadows='on')
        self.connect('scene', 'object2', 'light')
        self.set('card', 'spec_amount', .7)
        self.set('card', 'emission', .2)
        self.parity()
        self.set('front', 'alpha', 1)
        self.parity()

    def test_seventy_stacked_cards(self):
        with patch.object(s, 'scene_from_node', return_value=s.Scene(
                tuple(card(-i*.02) for i in range(70)))):
            self.parity()

    def test_max_hits_overflow_and_auto_cpu_error(self):
        with patch.object(s, 'scene_from_node', return_value=s.Scene(
                tuple(card(-i*.01, .1) for i in range(65)))):
            for backend in ('gpu', 'auto'):
                with self.subTest(backend=backend), patch.object(s, 'render', wraps=s.render) as cpu:
                    prefix = 'GPU Render3D failed: ' if backend == 'gpu' else '^'
                    with self.assertRaisesRegex(ValueError, prefix + 'Ray-traced render exceeds MAX_HITS_PER_RAY'):
                        self.render(backend)
                    self.assertEqual(cpu.called, backend == 'auto')

    def test_memory_refusal(self):
        with patch.object(gpurt, '_memory_check', side_effect=ValueError('GPU ray tracing needs about 1 MiB')):
            self.fallback('GPU Render3D failed: GPU ray tracing needs about 1 MiB')

    def test_bands_do_not_use_raster_shadow_budget(self):
        expected = self.render('gpu')
        with patch.object(gpurt_render, 'GPU_RT_RAYS_PER_SUBMISSION', 192), \
                patch.object(gpu3d, '_band_plan', side_effect=AssertionError('raster budget used')):
            self.assertTrue(np.array_equal(self.render('gpu'), expected))

    def test_raster_still_uses_raster_path(self):
        self.set('render', 'render_mode', 'raster')
        expected = self.render('cpu')
        with patch.object(gpurt_render, 'render_beauty', side_effect=AssertionError('raytrace used')), \
                patch.object(gpu3d, '_band_plan', wraps=gpu3d._band_plan) as bands:
            actual = self.render('gpu')
            bands.assert_called_once()
        interior = expected[..., 3] == 1
        np.testing.assert_allclose(actual[interior], expected[interior], atol=2e-3, rtol=0)


if __name__ == '__main__':
    unittest.main()
