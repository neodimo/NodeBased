"""Render3D integration parity: 3e-3 allows float blending plus the CPU's
1e-4 transmittance termination. Fallbacks must remain bitwise exact.
"""
from dataclasses import replace
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from nodebased import gpu3d, gpusplat, scene3d as s, splats
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.cancellation import Cancelled
from tests import gpu_precision


def cloud():
    x, y = np.meshgrid(np.linspace(-1.2, 1.2, 9), np.linspace(-.8, .8, 7))
    n = x.size
    sh = np.zeros((n, 1, 3)); sh[:, 0] = (np.array((.8, .25, .1))-.5)/splats.C0
    return splats.SplatCloud(np.column_stack((x.ravel(), y.ravel(), np.zeros(n))),
        np.tile((.18, .16, .01), (n, 1)), np.tile((1, 0, 0, 0), (n, 1)),
        np.full(n, .8), sh, 0, colorspace='linear')


def scene():
    return s.Scene(splats=(s.SplatInstance(cloud()),))


def card(z=0, angle=0, alpha=1):
    return s._card(2.2, 1.7, (.1, .3, .7, alpha),
                   s.Transform3D(s.Vec3(.13, .07, z), s.Vec3(0, angle, 0)))


class GraphFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        path = self.tmp.name + '/plane.ply'; splats.write_ply(cloud(), path)
        self.d = Dispatcher()
        for key, kind, params in (
            ('read', 'ReadSplat3D', dict(splat_path=path, splat_colorspace='linear')),
            ('scene', 'Scene3D', {}), ('camera', 'Camera3D', {}),
            ('render', 'Render3D', dict(width=47, height=35, samples=1))):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        self.connect('scene', 'object0', 'read')
        self.connect('render', 'scene', 'scene'); self.connect('render', 'camera', 'camera')

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def set(self, key, param, value):
        self.d.execute(dict(op='set', id=key, param=param, value=value))

    def render(self, backend):
        self.set('render', 'render_backend', backend)
        return Evaluator().evaluate(self.d.document, 'render')

    def fallback(self, reason):
        expected = self.render('cpu')
        self.assertTrue(np.array_equal(self.render('auto'), expected))
        with self.assertRaisesRegex(ValueError, reason):
            self.render('gpu')


class FallbackTests(GraphFixture, unittest.TestCase):
    def test_no_adapter(self):
        with patch.object(gpu3d, 'available', return_value=False):
            self.fallback('unavailable')

    def test_unsupported(self):
        cases = [
            (replace(scene(), geometries=(card(alpha=.5),)), 'transparent meshes'),
            (replace(scene(), geometries=(replace(card(), projection=s.Projection(
                s.Camera(), np.ones((2, 2, 4), 'f4'))),)), 'transparent meshes')]
        # Relit and caught splat shadows run on the GPU now (tests/test_3d_gpu_splat_shadows.py).
        for value, reason in cases:
            with self.subTest(reason=reason), patch.object(s, 'scene_from_node', return_value=value), \
                    patch.object(gpu3d, 'available', return_value=True):
                self.fallback('unsupported.*' + reason)
        for output in ('depth', 'splats', 'normals', 'diffuse'):
            with self.subTest(output=output), patch.object(gpu3d, 'available', return_value=True):
                self.set('render', 'render_output', output)
                self.fallback('unsupported.*data passes.*splats.*CPU-only')

    def test_capability(self):
        with patch.object(gpu3d, 'available', return_value=True), \
                patch.object(gpu3d, '_state', return_value={}), \
                patch.object(gpusplat, 'check_capability', return_value='test capability reason'):
            self.fallback('unsupported.*test capability reason')

    def test_cancellation_before_adapter(self):
        event = threading.Event(); event.set()
        with patch.object(gpu3d, '_state', side_effect=AssertionError('must cancel first')):
            with self.assertRaises(Cancelled):
                gpu3d.render(scene(), s.Camera(), 16, 12, cancel=event)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUParityTests(unittest.TestCase):
    def parity(self, value, width=96, height=72, **kwargs):
        expected = s.render(value, s.Camera(), width, height, **kwargs)
        actual = gpu3d.render(value, s.Camera(), width, height, **kwargs)
        gpu_precision.note(self.id())
        np.testing.assert_allclose(actual, expected, atol=gpu_precision.tolerance(3e-3), rtol=0)
        self.assertFalse(actual.flags.writeable)
        self.assertEqual(actual.dtype, np.float32)
        return actual, expected

    def test_baked_sizes_and_termination(self):
        for w, h in ((96, 72), (79, 53)):
            self.parity(scene(), w, h)
        c = cloud()
        c = replace(c, positions=np.zeros_like(c.positions), opacity=np.full(len(c), .9))
        self.parity(s.Scene(splats=(s.SplatInstance(c),)))

    def test_mesh_occlusion_and_grazing_view_depth(self):
        for z, angle in ((.8, 0), (-.8, 0), (.07, 43), (.07, 78)):
            with self.subTest(z=z, angle=angle):
                value = replace(scene(), geometries=(card(z, angle),))
                actual, expected = self.parity(value)
                # Blue card has no red splat contribution where it occludes.
                np.testing.assert_array_equal(actual[..., 0] > .101, expected[..., 0] > .101)
                depth = gpu3d.render(s.Scene(geometries=value.geometries), s.Camera(), 96, 72, output='depth')
                cpu = s.render(s.Scene(geometries=value.geometries), s.Camera(), 96, 72, output='depth')
                # Measured on an RTX 3080 Ti: up to 2.1e-4 at 43 and 78 degree cards (float32 interpolation order);
                # the llvmpipe run held 2e-5, so the bound is set by real hardware.
                np.testing.assert_allclose(depth, cpu, atol=5e-4, rtol=0)

    def test_samples_and_background(self):
        for alpha in (.5, 1):
            for geometries in ((), (card(-.3, 35),)):
                self.parity(replace(scene(), geometries=geometries), 47, 35,
                            samples=2, background=(.2, .4, .6, alpha))

    def test_relighting(self):
        for kind in ('Directional', 'Point'):
            value = replace(scene(), splats=(replace(scene().splats[0], relight=.7),),
                lights=(s.Light(kind=kind), s.Light(intensity=0)))
            self.parity(value, ambient=.17)

    def test_cancel_during_layer(self):
        event = threading.Event()
        original = gpusplat.render_layer
        def cancel_layer(*args, **kwargs):
            self.assertIs(kwargs['cancel'], event)
            event.set()
            return original(*args, **kwargs)
        with patch.object(gpusplat, 'render_layer', side_effect=cancel_layer):
            with self.assertRaises(Cancelled):
                gpu3d.render(scene(), s.Camera(), 31, 23, cancel=event)

    def test_baked_shadow_lights_and_mesh_path(self):
        value = replace(scene(), lights=(s.Light(shadows=True),))
        self.parity(value)
        self.assertEqual(gpu3d.last_shadow_path, 'brute')
        bare = replace(value, geometries=(card(-1),), splats=())
        with patch.object(gpusplat, 'check_capability', side_effect=AssertionError('no splats')):
            gpu3d.render(bare, s.Camera(), 31, 23)
        self.assertEqual(gpu3d.last_shadow_path, 'brute')


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUGraphTests(GraphFixture, unittest.TestCase):
    def test_gpu_and_auto_nested_instances(self):
        self.d.execute(dict(op='create', id='nested', type='Scene3D', params=dict(tx=.2, ry=23)))
        self.connect('nested', 'object0', 'read')
        self.connect('scene', 'object1', 'nested')
        self.set('scene', 'rz', 12)
        expected = self.render('cpu')
        np.testing.assert_allclose(self.render('gpu'), expected, atol=3e-3, rtol=0)
        np.testing.assert_allclose(self.render('auto'), expected, atol=3e-3, rtol=0)

    def test_memory_refusal(self):
        with patch.object(gpusplat, 'GPU_SPLAT_MEMORY_CAP', 1):
            self.fallback('GPU Render3D failed: GPU splat render needs about .*MiB')

    def test_auto_cancel_not_swallowed(self):
        with patch.object(gpusplat, 'render_layer', side_effect=Cancelled()):
            with self.assertRaises(Cancelled):
                self.render('auto')


if __name__ == '__main__':
    unittest.main()
