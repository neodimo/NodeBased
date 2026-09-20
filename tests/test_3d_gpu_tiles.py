"""Submission band planning, exact raster parity, and cancellation ownership."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import math
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nodebased import gpu3d, scene3d as s
from nodebased.cancellation import Cancelled
from tests.test_3d_gpu import card, gradient
from tests import test_3d_shadows as reference


class BandPlanning(unittest.TestCase):
    def test_boundary_tables_and_row_coverage(self):
        for reported in ('DiscreteGPU', 'IntegratedGPU', 'CPU', 'unknown'):
            state = {'info': {'adapter_type': reported}}
            kind = gpu3d._adapter_kind(state)
            for path, table in (('brute', gpu3d.SHADOW_WORK_BUDGETS),
                                ('bvh', gpu3d.SHADOW_BVH_WORK_BUDGETS)):
                with self.subTest(adapter=reported, path=path), patch.dict(table, {kind: 100}):
                    self.assertEqual(gpu3d._band_plan(state, 100, path, 19), [(0, 19)])
                    self.assertEqual(gpu3d._band_plan(state, 101, path, 19), [(0, 9), (9, 19)])
                    bands = gpu3d._band_plan(state, 500, path, 19)
                    self.assertEqual(len(bands), 5)
                    self.assertEqual([y for a, b in bands for y in range(a, b)], list(range(19)))
                    with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the GPU budget:.*even split into 64 bands'):
                        gpu3d._band_plan(state, 6401, path, 100)
                    with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the GPU budget:'):
                        gpu3d._band_plan(state, 2001, path, 19)
                    with patch.object(gpu3d, 'GPU_MAX_BANDS', 3):
                        with self.assertRaisesRegex(ValueError, 'even split into 3 bands'):
                            gpu3d._band_plan(state, 301, path, 19)
                    with patch.dict(table, {kind: 0}):
                        self.assertEqual(gpu3d._band_plan(state, 0, path, 19), [(0, 19)])
                        with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the GPU budget:'):
                            gpu3d._band_plan(state, 1, path, 19)

    def test_force_and_one_row(self):
        state = {'info': {'adapter_type': 'CPU'}}
        for count in (2, 3, 7, 19):
            with patch.object(gpu3d, 'GPU_FORCE_BANDS', count):
                self.assertEqual(len(gpu3d._band_plan(state, 0, 'brute', 19)), count)
                self.assertEqual(gpu3d._band_plan(state, 0, 'brute', 1), [(0, 1)])


@contextmanager
def observed_device(event=None, fail_read=False):
    """Delegate to the real device; track ownership without wrapping GPU handles."""
    state = dict(gpu3d._state())
    original = state['device']
    device = Mock(wraps=original, limits=original.limits)
    queue = Mock(wraps=original.queue)
    device.queue = queue
    resources = []
    with ExitStack() as stack:
        def create(name, *args, **kwargs):
            resource = getattr(original, name)(*args, **kwargs)
            destroy = stack.enter_context(patch.object(resource, 'destroy', wraps=resource.destroy))
            resources.append((resource, destroy))
            if fail_read and name == 'create_buffer':
                stack.enter_context(patch.object(resource, 'read_mapped', side_effect=RuntimeError('readback failed')))
            return resource
        for name in ('create_buffer', 'create_buffer_with_data', 'create_texture'):
            getattr(device, name).side_effect = lambda *a, _name=name, **kw: create(_name, *a, **kw)
        def submit(commands):
            original.queue.submit(commands)
            if event is not None:
                event.set()
        queue.submit.side_effect = submit
        state['device'] = device
        stack.enter_context(patch.object(gpu3d, '_state', return_value=state))
        yield device, resources


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUTiles(unittest.TestCase):
    def shadow_scene(self, point=False, alpha=1):
        ref = reference.ShadowTests()
        return s.Scene((ref.ground, ref.blocker(alpha)),
                       (ref.light(kind='Point' if point else 'Directional'),)), ref.camera

    def require_bvh(self):
        if not gpu3d._bvh_capable(gpu3d._state()['device'].limits, 2**20, 2**20):
            self.skipTest('adapter lacks BVH storage limits')

    def independence(self, path):
        if path == 'bvh':
            self.require_bvh()
        for point in (False, True):
            for transparent in (False, True):
                if path is None:
                    geometries = (card(texture=gradient()),)
                    if transparent:
                        geometries += (card((.8, .2, .4, .4), s.Vec3(.2, 0, 1), gradient(True)),)
                    scene, camera = s.Scene(geometries, (s.Light(kind='Point' if point else 'Directional'),)), s.Camera()
                else:
                    scene, camera = self.shadow_scene(point, .5 if transparent else 1)
                    scene = replace(scene, geometries=(replace(scene.geometries[0], texture=gradient()), scene.geometries[1]))
                with patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0 if path == 'bvh' else 10**9):
                    for output in ('rgba', 'depth', 'normals', 'position', 'uv', 'object_id',
                                   'albedo', 'diffuse', 'specular', 'emission'):
                        kwargs = dict(output=output, samples=2, ambient=.13, background=(.1, .2, .3, .7))
                        with patch.object(gpu3d, 'GPU_FORCE_BANDS', 1):
                            expected = gpu3d.render(scene, camera, 23, 19, **kwargs)
                        if path is not None and output in ('rgba', 'diffuse', 'specular'):
                            self.assertEqual(gpu3d.last_shadow_path, path)
                        for count in (2, 3, 7, 19):
                            with self.subTest(path=path, point=point, transparent=transparent, output=output, bands=count), \
                                    patch.object(gpu3d, 'GPU_FORCE_BANDS', count):
                                actual = gpu3d.render(scene, camera, 23, 19, **kwargs)
                                self.assertTrue(np.array_equal(actual, expected))

    def test_band_independence_unshadowed(self):
        self.independence(None)

    def test_band_independence_brute(self):
        self.independence('brute')

    def test_band_independence_bvh(self):
        self.independence('bvh')

    def budget_render(self, path, width=31, height=23, count=5):
        if path == 'bvh':
            self.require_bvh()
        scene, camera = self.shadow_scene()
        triangles = sum(len(g.triangles) for g in scene.geometries)
        work = width*height*triangles if path == 'brute' else (width*height*16+16*triangles)*math.log2(triangles+2)
        table = gpu3d.SHADOW_WORK_BUDGETS if path == 'brute' else gpu3d.SHADOW_BVH_WORK_BUDGETS
        with patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0 if path == 'bvh' else 10**9):
            with patch.dict(table, dict.fromkeys(table, work)):
                expected = gpu3d.render(scene, camera, width, height, ambient=.1)
            with observed_device() as (device, resources), patch.dict(table, dict.fromkeys(table, work/count)):
                actual = gpu3d.render(scene, camera, width, height, ambient=.1)
                self.assertGreaterEqual(device.queue.submit.call_count, count)
                self.assertTrue(np.array_equal(actual, expected))
                # Allocations and bind groups are frame-wide, independent of bands.
                self.assertEqual(device.create_buffer.call_count, 1)
                self.assertEqual(device.create_texture.call_count, len(scene.geometries)+2)
                self.assertEqual(device.create_bind_group.call_count, len(scene.geometries)*2)
                self.assertTrue(resources)
                for _, destroy in resources:
                    destroy.assert_called_once_with()
            with patch.dict(table, dict.fromkeys(table, work/(gpu3d.GPU_MAX_BANDS+1))):
                with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the GPU budget:.*even split into 64 bands'):
                    gpu3d.render(scene, camera, width, height)

    def test_budget_banding_brute(self):
        self.budget_render('brute')

    def test_budget_banding_bvh(self):
        self.budget_render('bvh')

    def test_hd_without_refusal(self):
        self.budget_render('brute', 1920, 1080, 10)

    def test_budget_boundary_submission_counts(self):
        scene, camera = self.shadow_scene()
        work = 31*23*sum(len(g.triangles) for g in scene.geometries)
        table = gpu3d.SHADOW_WORK_BUDGETS
        with patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 10**9):
            for budget, count in ((work, 1), (work-1, 2)):
                with observed_device() as (device, _), patch.dict(table, dict.fromkeys(table, budget)):
                    gpu3d.render(scene, camera, 31, 23)
                    self.assertEqual(device.queue.submit.call_count, count)
            with observed_device() as (device, _), patch.dict(table, dict.fromkeys(table, 0)):
                for output in ('depth', 'normals', 'position', 'uv', 'object_id', 'albedo'):
                    device.queue.submit.reset_mock()
                    gpu3d.render(scene, camera, 31, 23, output=output)
                    self.assertEqual(device.queue.submit.call_count, 1)
                for light in (replace(scene.lights[0], shadows=False), replace(scene.lights[0], intensity=0)):
                    device.queue.submit.reset_mock()
                    gpu3d.render(replace(scene, lights=(light,)), camera, 31, 23)
                    self.assertEqual(device.queue.submit.call_count, 1)

    def test_cancellation_cleanup_and_recovery(self):
        scene, camera = self.shadow_scene(alpha=.5)
        expected = gpu3d.render(scene, camera, 31, 23)
        event = threading.Event()
        with patch.object(gpu3d, 'GPU_FORCE_BANDS', 7):
            with observed_device(event) as (device, resources):
                with self.assertRaises(Cancelled):
                    gpu3d.render(scene, camera, 31, 23, cancel=event)
                self.assertEqual(device.queue.submit.call_count, 1)
                self.assertTrue(resources)
                for _, destroy in resources:
                    destroy.assert_called_once_with()
                device.queue.submit.reset_mock()
                with self.assertRaises(Cancelled):
                    gpu3d.render(scene, camera, 31, 23, cancel=event)
                device.queue.submit.assert_not_called()
            event.clear()
            self.assertTrue(np.array_equal(gpu3d.render(scene, camera, 31, 23, cancel=event), expected))

    def test_readback_exception_cleanup_and_recovery(self):
        scene, camera = self.shadow_scene()
        expected = gpu3d.render(scene, camera, 31, 23)
        with patch.object(gpu3d, 'GPU_FORCE_BANDS', 3), observed_device(fail_read=True) as (device, resources):
            with self.assertRaisesRegex(RuntimeError, 'readback failed'):
                gpu3d.render(scene, camera, 31, 23)
            self.assertEqual(device.queue.submit.call_count, 1)
            for _, destroy in resources:
                destroy.assert_called_once_with()
        self.assertTrue(np.array_equal(gpu3d.render(scene, camera, 31, 23), expected))
