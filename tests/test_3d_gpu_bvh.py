"""GPU shadow BVH contracts; host validation also runs without an adapter."""
import math
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nodebased import gpu3d, raytrace, scene3d as s
from nodebased.imaging import Cancelled
from tests import test_3d_gpu_shadows as parity
from tests.test_3d_gpu_shadows import boundary, dilate
from tests import test_3d_shadows as reference


def scene(segments=32, alpha=.5, point=False, two=False):
    ref = reference.ShadowTests()
    sphere = s._sphere(.7, segments, (1, 1, 1, 1),
                       s.Transform3D(s.Vec3(2, .7, 0)))
    lights = (ref.light(kind='Point' if point else 'Directional'),)
    if two:
        lights += (ref.light(s.Vec3(-4, 4, 0), intensity=.4),)
    return s.Scene((ref.ground, ref.blocker(alpha), sphere), lights)


class BVHValidation(unittest.TestCase):
    def test_capability_fallback_without_gpu(self):
        sc = scene()
        count = sum(len(g.triangles) for g in sc.geometries)
        for limits in ({'max-storage-buffers-per-shader-stage': 3,
                        'max-storage-buffer-binding-size': 2**30},
                       {'max-storage-buffers-per-shader-stage': 4,
                        'max-storage-buffer-binding-size': count*4-1}):
            state = {'info': {'adapter_type': 'DiscreteGPU'}, 'device': Mock(limits=limits)}
            with patch.object(gpu3d, '_state', return_value=state), patch.object(
                    gpu3d, '_render', return_value=np.zeros((8, 8, 4), 'f4')) as render:
                gpu3d.render(sc, reference.ShadowTests.camera, 8, 8)
                self.assertEqual(gpu3d.last_shadow_path, 'brute')
                self.assertIsNone(render.call_args.args[-1])
        self.assertTrue(gpu3d._bvh_capable({'max-storage-buffers-per-shader-stage': 4,
                                         'max-storage-buffer-binding-size': 96}, 96, 4))
        self.assertFalse(gpu3d._bvh_capable({'max-storage-buffers-per-shader-stage': 4,
                                          'max-storage-buffer-binding-size': 95}, 96, 4))

    def test_packing_outward_and_stack_guard(self):
        sc = scene()
        count = sum(len(g.triangles) for g in sc.geometries)
        packed, _ = gpu3d._shadow_data(sc, count, 2**30, None)
        triangles = raytrace.TriangleSet(packed[:, 0, :3], packed[:, 1, :3],
                                        packed[:, 2, :3], packed[:, 0, 3])
        bvh = raytrace.Bvh.build(*triangles.aabbs())
        nodes, order = gpu3d._pack_bvh(bvh)
        self.assertEqual(nodes.dtype.itemsize, 48)
        self.assertEqual([nodes.dtype.fields[k][1] for k in ('lo', 'left', 'hi', 'right', 'offset', 'count', 'pad')],
                         [0, 12, 16, 28, 32, 36, 40])
        self.assertTrue((nodes['lo'].astype('f8') <= bvh.node_lo).all())
        self.assertTrue((nodes['hi'].astype('f8') >= bvh.node_hi).all())
        np.testing.assert_array_equal(order, bvh.prim_order)
        with patch.object(gpu3d, 'SHADOW_BVH_STACK_SIZE', 2):
            with self.assertRaisesRegex(ValueError, 'depth.*stack'):
                gpu3d._pack_bvh(bvh)
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            gpu3d._pack_bvh(bvh, event)
        with self.assertRaises(Cancelled):
            gpu3d.render(sc, reference.ShadowTests.camera, 8, 8, cancel=event)

    @patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0)  # force the BVH path; default is per adapter type
    def test_bvh_budget_and_build_once(self):
        for reported, kind in [('DiscreteGPU', 'discrete'), ('IntegratedGPU', 'integrated'),
                               ('CPU', 'cpu'), ('unknown', 'other')]:
            state = {'info': {'adapter_type': reported}}
            budget = gpu3d.SHADOW_BVH_WORK_BUDGETS[kind]
            self.assertEqual(gpu3d._shadow_budget(state, 0, 'bvh'), budget)
            with self.assertRaisesRegex(ValueError, 'bvh work units'):
                gpu3d._shadow_budget(state, budget+1, 'bvh')
        state = {'info': {'adapter_type': 'DiscreteGPU'}, 'device': Mock(limits={
            'max-storage-buffers-per-shader-stage': 4, 'max-storage-buffer-binding-size': 2**30})}
        sc = scene(two=True)
        count = sum(len(g.triangles) for g in sc.geometries)
        expected = (8*8*4*2*16 + 16*count)*math.log2(count+2)
        with patch.object(gpu3d, '_state', return_value=state), patch.object(
                gpu3d, '_render', return_value=np.zeros((16, 16, 4), 'f4')) as render, patch.object(
                raytrace.Bvh, 'build', wraps=raytrace.Bvh.build) as build:
            with patch.dict(gpu3d.SHADOW_BVH_WORK_BUDGETS, discrete=expected-1):
                with self.assertRaisesRegex(ValueError, 'bvh work units'):
                    gpu3d.render(sc, reference.ShadowTests.camera, 8, 8, samples=2)
                render.assert_not_called()
            build.reset_mock()
            with patch.dict(gpu3d.SHADOW_BVH_WORK_BUDGETS, discrete=expected):
                gpu3d.render(sc, reference.ShadowTests.camera, 8, 8, samples=2)
            build.assert_called_once()
            self.assertEqual(gpu3d.last_shadow_path, 'bvh')


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUBVHComparison(unittest.TestCase):
    def setUp(self):
        limits = gpu3d._state()['device'].limits
        if not gpu3d._bvh_capable(limits, 2**20, 2**20):
            self.skipTest('adapter lacks BVH storage limits')

    def compare(self, sc, samples=1):
        with patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0):
            actual = gpu3d.render(sc, reference.ShadowTests.camera, 64, 48, ambient=.1, samples=samples)
            self.assertEqual(gpu3d.last_shadow_path, 'bvh')
        with patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 10**9):
            brute = gpu3d.render(sc, reference.ShadowTests.camera, 64, 48, ambient=.1, samples=samples)
            self.assertEqual(gpu3d.last_shadow_path, 'brute')
        interior = (brute[..., 3] > 0) & ~dilate(boundary(brute), 1)
        self.assertGreater(interior.sum(), 10)
        np.testing.assert_allclose(actual[interior], brute[interior], atol=1e-5, rtol=0)
        # Reuse the established CPU parity tolerances and edge masking.
        ref = parity.GPUShadowComparison()
        ref.setUp()
        with patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0):
            ref.compare(sc, samples=samples)
        self.assertEqual(gpu3d.last_shadow_path, 'bvh')

    def test_directional_point_alpha_two_lights_samples(self):
        for point, alpha, two, samples in [(False, 1, False, 1), (False, .5, False, 1),
                                          (True, .5, False, 1), (True, .5, True, 2)]:
            with self.subTest(point=point, alpha=alpha, two=two, samples=samples):
                self.compare(scene(alpha=alpha, point=point, two=two), samples)

    @patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0)  # force the BVH path; default is per adapter type
    def test_ten_thousand_triangles(self):
        sc = scene(segments=104)
        self.assertGreater(sum(len(g.triangles) for g in sc.geometries), 10000)
        ref = parity.GPUShadowComparison()
        ref.setUp()
        ref.compare(sc)
        self.assertEqual(gpu3d.last_shadow_path, 'bvh')

    @patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0)  # force the BVH path; default is per adapter type
    def test_capability_fallback_matches_bvh(self):
        sc = scene()
        expected = gpu3d.render(sc, reference.ShadowTests.camera, 64, 48, ambient=.1)
        self.assertEqual(gpu3d.last_shadow_path, 'bvh')
        state = dict(gpu3d._state())
        limits = dict(state['device'].limits)
        limits['max-storage-buffers-per-shader-stage'] = 2
        state['device'] = Mock(wraps=state['device'], limits=limits)
        with patch.object(gpu3d, '_state', return_value=state):
            actual = gpu3d.render(sc, reference.ShadowTests.camera, 64, 48, ambient=.1)
        self.assertEqual(gpu3d.last_shadow_path, 'brute')
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)

    def test_half_precision(self):
        with patch.dict(gpu3d._state(), {'format': 'rgba16float', 'pipelines': {}}):
            self.compare(scene())

    def test_cancel_after_preparation(self):
        event = threading.Event()
        original = gpu3d._prepare
        def prepare(*args):
            result = original(*args)
            event.set()
            return result
        with patch.object(gpu3d, '_prepare', side_effect=prepare):
            with self.assertRaises(Cancelled):
                gpu3d.render(scene(), reference.ShadowTests.camera, 64, 48, cancel=event)


class PathSelection(unittest.TestCase):
    def test_default_threshold_is_per_adapter_type(self):
        sc = scene(segments=40)
        count = sum(len(g.triangles) for g in sc.geometries)
        limits = {'max-storage-buffers-per-shader-stage': 8, 'max-storage-buffer-binding-size': 2**30}
        for kind, adapter, table in [('discrete', 'DiscreteGPU', 20000), ('integrated', 'IntegratedGPU', 5000),
                                     ('cpu', 'CPU', 500), ('other', 'unknown', 5000)]:
            self.assertEqual(gpu3d.SHADOW_BVH_THRESHOLDS[kind], table)
            state = {'info': {'adapter_type': adapter}, 'device': Mock(limits=limits)}
            with patch.dict(gpu3d.SHADOW_BVH_THRESHOLDS, {kind: count - 1}), \
                    patch.object(gpu3d, '_state', return_value=state), \
                    patch.object(gpu3d, '_render', return_value=np.zeros((8, 8, 4), 'f4')):
                gpu3d.render(sc, reference.ShadowTests.camera, 8, 8)
                self.assertEqual(gpu3d.last_shadow_path, 'bvh')
            with patch.dict(gpu3d.SHADOW_BVH_THRESHOLDS, {kind: count}), \
                    patch.object(gpu3d, '_state', return_value=state), \
                    patch.object(gpu3d, '_render', return_value=np.zeros((8, 8, 4), 'f4')):
                gpu3d.render(sc, reference.ShadowTests.camera, 8, 8)
                self.assertEqual(gpu3d.last_shadow_path, 'brute')
