"""Shadow parity using the CPU shadow suite's scenes and world-space probes."""
from dataclasses import replace
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, scene3d as s
from nodebased.imaging import Cancelled
from tests import test_3d_shadows as reference


def dilate(mask, radius):
    h, w = mask.shape
    padded = np.pad(mask, radius)
    return np.logical_or.reduce([padded[y:y+h, x:x+w]
                                 for y in range(2*radius+1) for x in range(2*radius+1)])


def boundary(image):
    edge = np.zeros(image.shape[:2], bool)
    dx = np.max(np.abs(np.diff(image, axis=1)), axis=2) > .05
    dy = np.max(np.abs(np.diff(image, axis=0)), axis=2) > .05
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:] |= dy
    edge[:-1] |= dy
    return edge


class ShadowValidation(unittest.TestCase):
    def setUp(self):
        self.ref = reference.ShadowTests()
        self.scene = s.Scene((self.ref.ground, self.ref.blocker()), (self.ref.light(),))

    def test_adapter_budgets_without_gpu(self):
        for reported, expected in [('DiscreteGPU', 10e9), ('IntegratedGPU', 2e9),
                                   ('CPU', 3e8), ('unknown', 2e9), ('discrete_gpu', 10e9)]:
            state = {'info': {'adapter_type': reported}}
            self.assertEqual(gpu3d._shadow_budget(state, 0), expected)
        # 400 million tests passes discrete but is refused on software before preparation.
        for reported in ('CPU', 'DiscreteGPU'):
            with patch.object(gpu3d, '_state', return_value={'info': {'adapter_type': reported}}), patch.object(
                    gpu3d, '_render', return_value=np.zeros((1, 1, 4), 'f4')) as render:
                if reported == 'CPU':
                    with self.assertRaisesRegex(ValueError, 'CPU.*400,000,000 > 300,000,000'):
                        gpu3d.render(self.scene, self.ref.camera, 10000, 10000)
                    render.assert_not_called()
                else:
                    gpu3d.render(self.scene, self.ref.camera, 10000, 10000)
                    render.assert_called_once()

    def test_budget_sample_light_counts_and_inactive_modes(self):
        work = 64 * 48 * 4 * 4
        state = {'info': {'adapter_type': 'CPU'}}
        with patch.object(gpu3d, '_state', return_value=state), patch.object(
                gpu3d, '_render', return_value=np.zeros((96, 128, 4), 'f4')) as render:
            with patch.dict(gpu3d.SHADOW_WORK_BUDGETS, cpu=work-1):
                with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the GPU budget:'):
                    gpu3d.render(self.scene, self.ref.camera, 64, 48, samples=2)
                render.assert_not_called()
            with patch.dict(gpu3d.SHADOW_WORK_BUDGETS, cpu=work):
                gpu3d.render(self.scene, self.ref.camera, 64, 48, samples=2)
            with patch.dict(gpu3d.SHADOW_WORK_BUDGETS, cpu=0):
                for scene, output in [(self.scene, 'depth'), (self.scene, 'normals'),
                        (replace(self.scene, lights=(replace(self.ref.light(), intensity=0),)), 'rgba'),
                        (replace(self.scene, lights=(replace(self.ref.light(), shadows=False),)), 'rgba')]:
                    gpu3d.render(scene, self.ref.camera, 64, 48, output=output)

    def test_packing_extent_alpha_limits_and_cancel(self):
        scene = s.Scene((self.ref.ground, self.ref.blocker(.5, height=20)))
        packed, bias = gpu3d._shadow_data(scene, 4, 192, None)
        self.assertAlmostEqual(bias, .02)
        np.testing.assert_array_equal(packed[:, 0, 3], [1, 1, .5, .5])
        for i, geometry in enumerate(scene.geometries):
            matrix = geometry.world_matrix()
            world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
            tri = world[geometry.triangles]
            block = packed[i*2:i*2+2, :, :3]
            np.testing.assert_allclose(block[:, 0], tri[:, 0])
            np.testing.assert_allclose(block[:, 0]+block[:, 1], tri[:, 1], atol=1e-6)
            np.testing.assert_allclose(block[:, 0]+block[:, 2], tri[:, 2], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'max_storage_buffer_binding_size'):
            gpu3d._shadow_data(scene, 4, 191, None)
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            gpu3d.render(self.scene, self.ref.camera, 64, 48, cancel=event)
        with self.assertRaises(Cancelled):
            gpu3d._shadow_data(scene, 4, 192, event)

    def test_missing_wgpu_with_shadows(self):
        with patch.dict(sys.modules, {'wgpu': None}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
            with self.assertRaisesRegex(RuntimeError, 'wgpu unavailable'):
                gpu3d.render(self.scene, self.ref.camera, 64, 48)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUShadowComparison(unittest.TestCase):
    def setUp(self):
        self.ref = reference.ShadowTests()

    def compare(self, scene, camera=None, points=(), **kwargs):
        camera = camera or self.ref.camera
        expected = s.render(scene, camera, 64, 48, ambient=.1, **kwargs)
        actual = gpu3d.render(scene, camera, 64, 48, ambient=.1, **kwargs)
        unshadowed = s.render(scene, camera, 64, 48, ambient=.1, shadows=False, **kwargs)
        # Isolate shadow transitions from smooth Lambert/normal/depth gradients.
        edges = boundary(np.concatenate((expected-unshadowed, expected[..., 3:4]), axis=2))
        interior = (expected[..., 3] > 0) & ~dilate(edges, 2)
        self.assertGreater(interior.sum(), 10)
        delta = np.max(np.abs(actual-expected), axis=2)
        self.assertLess(float(np.abs(actual[interior]-expected[interior]).mean()), 5e-3)
        # Large errors must lie within a one-pixel dilation of reference edges.
        # At most one pixel per edge pixel may disagree: a one-pixel raster shift,
        # not a displaced shadow or a missing region (also checked by probes).
        mismatch = delta > .1
        self.assertFalse((mismatch & ~dilate(edges, 1)).any())
        self.assertLessEqual(int(mismatch.sum()), int(edges.sum()))
        for point in points:
            np.testing.assert_allclose(self.ref.pixel(actual, point), self.ref.pixel(expected, point), atol=5e-3, rtol=0)
        return actual, expected

    def scene(self, light=None, alpha=1, receiver=None):
        return s.Scene((receiver if receiver is not None else self.ref.ground, self.ref.blocker(alpha)),
                       (light or self.ref.light(),))

    def test_directional_alpha_and_supersampling(self):
        for alpha in (0, .5, 1):
            with self.subTest(alpha=alpha):
                self.compare(self.scene(alpha=alpha), points=((0, 0, 0), (2, 0, 0)))
        for samples in (1, 2):
            self.compare(self.scene(self.ref.light(s.Vec3(-4, 4, 0))), samples=samples,
                         points=((1, 0, 0), (0, 0, 0)))

    def test_point_distance_limit(self):
        self.compare(self.scene(self.ref.light(s.Vec3(-2, 4, 0), kind='Point')),
                     points=((2/3, 0, 0), (2, 0, 0)))
        self.compare(s.Scene((self.ref.ground, self.ref.blocker(height=5)),
                             (self.ref.light(kind='Point'),)), points=((0, 0, 0),))

    def test_two_lights(self):
        lights = (self.ref.light(color=(1, .5, .25), intensity=2),
                  replace(self.ref.light(s.Vec3(-4, 4, 0), color=(.2, .4, .8), intensity=3), shadows=False))
        self.compare(replace(self.scene(alpha=.5), lights=lights), points=((0, 0, 0),))

    def test_transparent_receiver_and_texture_alpha(self):
        self.compare(self.scene(receiver=replace(self.ref.ground, color=(1, 1, 1, .5))),
                     points=((0, 0, 0), (2, 0, 0)))
        hidden = replace(self.ref.blocker(), texture=np.zeros((2, 2, 4), 'f4'))
        self.compare(s.Scene((self.ref.ground, hidden), (self.ref.light(),)), points=((0, 0, 0),))
        upper = self.ref.blocker(.5, height=2)
        upper = replace(upper, triangles=upper.triangles[:, ::-1])
        self.compare(s.Scene((self.ref.ground, self.ref.blocker(.5), upper), (self.ref.light(),)),
                     points=((0, 0, 0),))

    def test_no_self_shadow(self):
        shapes = [s._card(2, 2, (1, 1, 1, 1), s.Transform3D()),
                  s._sphere(1, 32, (1, 1, 1, 1), s.Transform3D()),
                  s._cube(2, (1, 1, 1, 1), s.Transform3D())]
        for shape in shapes:
            scene = s.Scene((shape,), (self.ref.light(s.Vec3(4, 1, 2)),))
            actual, expected = self.compare(scene, s.Camera())
            off = s.render(scene, s.Camera(), 64, 48, ambient=.1, shadows=False)
            lit = (off[..., 0] > .4) & ~dilate(boundary(off[..., 3:4]), 2)
            self.assertTrue(lit.any())
            np.testing.assert_allclose(actual[lit], off[lit], atol=5e-3, rtol=0)

    def test_data_outputs_and_inactive_lights_skip_upload(self):
        with patch.dict(gpu3d.SHADOW_WORK_BUDGETS, dict.fromkeys(gpu3d.SHADOW_WORK_BUDGETS, 0)):
            for output in ('depth', 'normals'):
                self.compare(self.scene(), output=output)
            for light in (replace(self.ref.light(), intensity=0), replace(self.ref.light(), shadows=False)):
                self.compare(self.scene(light))

    def test_half_precision_shadows(self):
        with patch.dict(gpu3d._state(), {'format': 'rgba16float', 'pipelines': {}}):
            self.compare(self.scene(alpha=.5), points=((0, 0, 0),))

    def test_storage_limit_checked_by_render(self):
        from unittest.mock import Mock
        state = dict(gpu3d._state())
        state['device'] = Mock(limits={'max-storage-buffer-binding-size': 191})
        with patch.object(gpu3d, '_state', return_value=state):
            with self.assertRaisesRegex(ValueError, 'max_storage_buffer_binding_size'):
                gpu3d.render(self.scene(), self.ref.camera, 64, 48)
        state['device'].create_buffer_with_data.assert_not_called()

    def test_cancel_after_preparation_and_gpu_budget(self):
        event = threading.Event()
        original = gpu3d._prepare
        def prepare(*args):
            result = original(*args)
            event.set()
            return result
        with patch.object(gpu3d, '_prepare', side_effect=prepare):
            with self.assertRaises(Cancelled):
                gpu3d.render(self.scene(), self.ref.camera, 64, 48, cancel=event)
        with patch.dict(gpu3d.SHADOW_WORK_BUDGETS, dict.fromkeys(gpu3d.SHADOW_WORK_BUDGETS, 1)):
            with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the GPU budget:'):
                gpu3d.render(self.scene(), self.ref.camera, 64, 48)
