"""GPU shadows on relit splats and GPU shadow catching, against the CPU reference.

The CPU renderer (`scene3d._SplatShadows`) is the reference; the GPU traces the same rays through the
triangle and caster BVHs of `gpurt_render`. Tolerance is the one the other GPU splat tests use (3e-3).
"""
from dataclasses import replace
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, gpurt_render, raytrace, scene3d as s, splats
from nodebased.cancellation import Cancelled
from nodebased.imaging import Evaluator
from tests.test_3d_gpu_splat_render import GraphFixture
from tests.test_3d_splat_shadow_catch import GRID, mesh_parts
from tests import gpu_precision

TOLERANCE = 3e-3
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 6)))


def field(n=9, spacing=.5, opacity=.9, scales=(.32, .32, .02), z=0.0):
    axis = (np.arange(n)-(n-1)/2)*spacing
    xs, ys = np.meshgrid(axis, axis)
    count = xs.size
    sh = np.zeros((count, 1, 3))
    sh[:, 0] = (np.array((.8, .5, .2))-.5)/splats.C0
    return splats.SplatCloud(np.column_stack((xs.ravel(), ys.ravel(), np.full(count, z))),
                             np.tile(scales, (count, 1)), np.tile((1., 0, 0, 0), (count, 1)),
                             np.full(count, opacity), sh, 0, colorspace='linear')


def occluder(x=1.0, z=1.2, alpha=1.0):
    """A card between the light and the right half of the field, hiding none of it from the camera."""
    return s._card(2.4, 6.0, (1, 1, 1, alpha), s.Transform3D(s.Vec3(x + 1.2 + .3, 0, z)))


def sun(**kwargs):
    return s.Light(**{'position': s.Vec3(1, 0, 1), 'shadows': True, **kwargs})


def relit_scene(light=None, geometries=None, relight=1.0, catch=0.0, cast=True, extra=()):
    return s.Scene((occluder(),) if geometries is None else geometries, (light or sun(),),
                   (s.SplatInstance(field(), relight=relight, shadow_catch=catch, cast_shadows=cast), *extra))


def cpu(value, mode='raster', size=(64, 48), **kwargs):
    s.clear_splat_shadow_cache()
    return s.render(value, CAMERA, *size, mode=mode, ambient=.15, **kwargs)


def gpu(value, mode='raster', size=(64, 48), **kwargs):
    return gpu3d.render(value, CAMERA, *size, mode=mode, ambient=.15, **kwargs)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class RelitShadowParity(unittest.TestCase):
    def setUp(self):
        s.clear_splat_shadow_cache()

    def parity(self, value, mode, label):
        expected, actual = cpu(value, mode), gpu(value, mode)
        error = float(np.abs(actual-expected).max())
        print(f'{label} [{mode}]: max |gpu - cpu| = {error:.6f}', flush=True)
        np.testing.assert_allclose(actual, expected, atol=gpu_precision.tolerance(TOLERANCE), rtol=0)
        self.assertFalse(actual.flags.writeable)
        self.assertEqual(actual.dtype, np.float32)
        return actual, expected

    def test_a_mesh_shadows_relit_splats_in_both_modes(self):
        value = relit_scene()
        for mode in ('raster', 'raytrace'):
            with self.subTest(mode=mode):
                actual, expected = self.parity(value, mode, 'mesh shadow on relit splats')
                unshadowed = cpu(relit_scene(light=sun(shadows=False)), mode)
                # The comparison is only meaningful if the shadow is a real, visible effect.
                self.assertGreater(np.abs(expected-unshadowed).max(), .05)
                self.assertGreater(np.abs(actual-unshadowed).max(), .05)

    def test_splats_shadow_splats_without_any_mesh(self):
        blocker = s.SplatInstance(field(n=3, spacing=1.6, opacity=.95, scales=(.9, .9, .05), z=1.0))
        value = relit_scene(geometries=(), extra=(blocker,))
        for mode in ('raster', 'raytrace'):
            with self.subTest(mode=mode):
                actual, expected = self.parity(value, mode, 'splat shadow on relit splats')
                uncast = replace(value, splats=(value.splats[0], replace(blocker, cast_shadows=False)))
                self.assertGreater(np.abs(cpu(uncast, mode)-expected).max(), .02)
                self.assertGreater(np.abs(gpu(uncast, mode)-actual).max(), .02)

    def test_point_and_soft_lights_and_bias(self):
        cases = {'point': s.Light(kind='Point', position=s.Vec3(3.5, 0, 3.0), shadows=True),
                 'soft directional': sun(shadow_blur=6.0, shadow_samples=6),
                 'soft point': s.Light(kind='Point', position=s.Vec3(3.5, 0, 3.0), shadows=True,
                                       shadow_blur=5.0, shadow_samples=4),
                 'bias': sun(shadow_bias=.05)}
        for name, light in cases.items():
            with self.subTest(light=name):
                self.parity(relit_scene(light=light), 'raytrace', name)

    def test_two_lights_one_unshadowed_and_relight_mix(self):
        lights = (sun(), s.Light(position=s.Vec3(0, 1, 1), shadows=False, intensity=.5),
                  s.Light(position=s.Vec3(-1, 0, 1), shadows=True, intensity=0))
        value = replace(relit_scene(relight=.6), lights=lights)
        self.parity(value, 'raster', 'mixed lights, relight .6')

    def test_samples_and_background(self):
        self.parity_kwargs(relit_scene(), samples=2, background=(.1, .2, .3, .5))

    def parity_kwargs(self, value, **kwargs):
        expected, actual = cpu(value, 'raytrace', (33, 25), **kwargs), gpu(value, 'raytrace', (33, 25), **kwargs)
        np.testing.assert_allclose(actual, expected, atol=gpu_precision.tolerance(TOLERANCE), rtol=0)

    def test_transformed_and_scaled_instances(self):
        matrix = np.eye(4)
        matrix[:3, :3] = 1.1*np.array(((.9, -.3, 0), (.3, .9, 0), (0, 0, 1)))
        matrix[:3, 3] = (.1, -.2, 0)
        instance = s.SplatInstance(field(), matrix=matrix, relight=1, scale_scale=1.2, opacity_scale=.8)
        self.parity(replace(relit_scene(), splats=(instance,)), 'raytrace', 'transformed instance')


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class CatchParity(unittest.TestCase):
    def setUp(self):
        s.clear_splat_shadow_cache()

    def test_caught_shadows_match_the_cpu_in_both_modes(self):
        for mode in ('raster', 'raytrace'):
            for strength in (1.0, .5):
                with self.subTest(mode=mode, strength=strength):
                    value = relit_scene(relight=0.0, catch=strength)
                    expected, actual = cpu(value, mode), gpu(value, mode)
                    print(f'caught shadow {strength} [{mode}]: max |gpu - cpu| = '
                          f'{np.abs(actual-expected).max():.6f}', flush=True)
                    np.testing.assert_allclose(actual, expected, atol=gpu_precision.tolerance(TOLERANCE), rtol=0)
                    plain = cpu(relit_scene(relight=0.0, catch=0.0), mode)
                    self.assertGreater(np.abs(expected-plain).max(), .05 * strength)
                    self.assertGreater(np.abs(actual-plain).max(), .05 * strength)

    def test_catch_blends_with_relight_and_takes_point_lights(self):
        light = s.Light(kind='Point', position=s.Vec3(3.5, 0, 3.0), shadows=True)
        for relight in (.5, 0.0):
            with self.subTest(relight=relight):
                value = relit_scene(light=light, relight=relight, catch=1.0)
                np.testing.assert_allclose(gpu(value, 'raytrace'), cpu(value, 'raytrace'), atol=gpu_precision.tolerance(TOLERANCE), rtol=0)

    def test_catching_traces_no_splat_bvh_and_is_cached(self):
        value = relit_scene(relight=0.0, catch=1.0)
        s.clear_splat_shadow_cache()
        before = dict(s.splat_shadow_stats)
        first = gpu(value, 'raytrace')
        self.assertEqual(s.splat_shadow_stats['caster_builds'], before['caster_builds'])
        traced = s.splat_shadow_stats['rays_traced']
        self.assertGreater(traced, before['rays_traced'])
        second = gpu(value, 'raytrace')
        self.assertEqual(s.splat_shadow_stats['rays_traced'], traced)
        np.testing.assert_array_equal(first, second)

    def test_gpu_answers_never_feed_the_cpu_reference(self):
        value = relit_scene()
        reference = cpu(value, 'raytrace')
        cpu_keys = {key: store.copy() for key, store in s._splat_visibility.items()}
        self.assertTrue(cpu_keys and not any(isinstance(key[1], tuple) and key[1][0] == 'gpu' for key in cpu_keys))
        gpu(value, 'raytrace')
        added = [key for key in s._splat_visibility if key not in cpu_keys]
        self.assertTrue(added and all(key[1][0] == 'gpu' for key in added))
        for key, store in cpu_keys.items():
            np.testing.assert_array_equal(s._splat_visibility[key], store)
        again = s.render(value, CAMERA, 64, 48, mode='raytrace', ambient=.15)
        np.testing.assert_array_equal(again, reference)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class VisibilityParity(unittest.TestCase):
    """Per-splat visibility, the quantity the images are built from."""
    def setUp(self):
        s.clear_splat_shadow_cache()

    def providers(self, value):
        mesh, bvh, bias = mesh_parts(value)
        reference = s._SplatShadows(value.splats, value.lights, mesh, bvh, bias)
        state = gpu3d._state(None)
        return reference, gpurt_render.GpuSplatShadows(state, value.splats, value.lights, mesh, bvh, bias)

    def test_relit_and_catch_visibility_agree_per_splat(self):
        rng = np.random.default_rng(3)
        cloud = field(n=12, spacing=.3, opacity=.6, scales=(.2, .2, .03))
        cloud = replace(cloud, positions=cloud.positions + rng.normal(0, .15, cloud.positions.shape))
        value = s.Scene((occluder(),), (sun(), s.Light(kind='Point', position=s.Vec3(2, 1, 2), shadows=True)),
                        (s.SplatInstance(cloud, relight=1, shadow_catch=1),))
        reference, candidate = self.providers(value)
        indices = np.arange(len(cloud))
        with candidate:
            for name in ('for_indices', 'catch_for_indices'):
                expected = getattr(reference, name)(0, indices)
                actual = getattr(candidate, name)(0, indices)
                np.testing.assert_allclose(actual, expected, atol=gpu_precision.tolerance(TOLERANCE), rtol=0)
                self.assertGreater(np.abs(expected-1).max(), .5, 'the test needs real shadows')
        self.assertIsNone(candidate._triangles)
        self.assertEqual(candidate._owned, [])

    def test_a_splat_never_shadows_itself(self):
        """A round opaque splat with the light along its axis: only its own tail lies beyond the start offset."""
        one = splats.SplatCloud(np.zeros((1, 3)), np.full((1, 3), .3), np.array([[1., 0, 0, 0]]),
                                np.array([.95]), np.zeros((1, 1, 3)), 0, colorspace='linear')
        value = s.Scene((), (s.Light(position=s.Vec3(1, 0, 0), shadows=True),), (s.SplatInstance(one, relight=1),))
        reference, candidate = self.providers(value)
        with candidate:
            self.assertEqual(float(reference.for_indices(0, [0])[0, 0]), 1.0)
            self.assertAlmostEqual(float(candidate.for_indices(0, [0])[0, 0]), 1.0, places=6)

    def test_a_scene_without_meshes_still_gets_splat_shadows(self):
        blocker = s.SplatInstance(field(n=3, spacing=1.6, opacity=.95, scales=(.9, .9, .05), z=1.0))
        value = s.Scene((), (sun(),), (s.SplatInstance(field(), relight=1), blocker))
        reference, candidate = self.providers(value)
        indices = np.arange(81)
        with candidate:
            np.testing.assert_allclose(candidate.for_indices(0, indices), reference.for_indices(0, indices),
                                       atol=gpu_precision.tolerance(TOLERANCE), rtol=0)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class BudgetsAndCancellation(unittest.TestCase):
    def setUp(self):
        s.clear_splat_shadow_cache()

    def test_an_oversized_scene_is_refused_like_the_other_gpu_paths(self):
        tiny = {kind: 1 for kind in gpu3d.SHADOW_BVH_WORK_BUDGETS}
        for value in (relit_scene(), relit_scene(relight=0.0, catch=1.0)):
            for mode in ('raster', 'raytrace'):
                with self.subTest(mode=mode, catch=value.splats[0].shadow_catch), \
                        patch.object(gpu3d, 'SHADOW_BVH_WORK_BUDGETS', tiny):
                    with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the GPU budget'):
                        gpu(value, mode)

    def test_caster_memory_is_refused(self):
        with patch('nodebased.gpusplat.GPU_SPLAT_MEMORY_CAP', 1000):
            with self.assertRaisesRegex(ValueError, 'splat casters'):
                gpu(relit_scene(), 'raytrace')

    def test_bands_split_the_rays_without_changing_the_answer(self):
        value = relit_scene()
        whole = gpu(value, 'raytrace')
        s.clear_splat_shadow_cache()
        with patch.object(gpu3d, 'GPU_FORCE_BANDS', 7):
            np.testing.assert_array_equal(gpu(value, 'raytrace'), whole)

    def test_cancel_before_any_ray_is_cast(self):
        event = threading.Event()
        original = gpu3d._band_plan

        def plan(*args, **kwargs):
            event.set()
            return original(*args, **kwargs)
        with patch.object(gpu3d, '_band_plan', side_effect=plan):
            with self.assertRaises(Cancelled):
                gpu(relit_scene(), 'raytrace', cancel=event)

    def test_cancel_between_bands_returns_promptly(self):
        event = threading.Event()
        submissions = []
        original = gpurt_render.GpuSplatShadows._upload

        def upload(self, data, usage=None):
            if usage is not None and usage & self.state['wgpu'].BufferUsage.COPY_SRC:
                submissions.append(len(submissions))
                if len(submissions) == 2:
                    event.set()
            return original(self, data, usage)
        with patch.object(gpu3d, 'GPU_FORCE_BANDS', 6), \
                patch.object(gpurt_render.GpuSplatShadows, '_upload', upload):
            start = time.perf_counter()
            with self.assertRaises(Cancelled):
                gpu(relit_scene(), 'raytrace', cancel=event)
        self.assertLess(time.perf_counter()-start, 2.0)
        self.assertEqual(len(submissions), 2, 'no band after the cancel may run')

    def test_buffers_are_released_after_a_cancel(self):
        seen = []
        original = gpurt_render.GpuSplatShadows.close

        def close(self):
            seen.append(len(self._owned) + (self._triangles is not None))
            original(self)
            seen.append(len(self._owned) + (self._triangles is not None))
        event = threading.Event()
        counted = []
        upload = gpurt_render.GpuSplatShadows._upload

        def cancelling(self, data, usage=None):
            counted.append(1)
            if len(counted) == 5:
                event.set()
            return upload(self, data, usage)
        with patch.object(gpurt_render.GpuSplatShadows, 'close', close), \
                patch.object(gpurt_render.GpuSplatShadows, '_upload', cancelling), \
                patch.object(gpu3d, 'GPU_FORCE_BANDS', 4):
            with self.assertRaises(Cancelled):
                gpu(relit_scene(), 'raytrace', cancel=event)
        self.assertEqual(len(seen), 2)
        self.assertGreater(seen[0], 0)
        self.assertEqual(seen[1], 0)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GraphRouting(GraphFixture, unittest.TestCase):
    def scene(self, **kwargs):
        return relit_scene(**kwargs)

    def check(self, value):
        with patch.object(s, 'scene_from_node', return_value=value):
            expected = self.render('cpu')
            with patch.object(s, 'render', side_effect=AssertionError('auto must not fall back')):
                automatic = self.render('auto')
                forced = self.render('gpu')
        np.testing.assert_allclose(automatic, expected, atol=gpu_precision.tolerance(TOLERANCE), rtol=0)
        np.testing.assert_array_equal(forced, automatic)
        return expected

    def test_auto_takes_the_gpu_for_a_caught_shadow_and_gpu_no_longer_raises(self):
        for mode in ('raster', 'raytrace'):
            with self.subTest(mode=mode):
                self.set('render', 'render_mode', mode)
                self.check(self.scene(relight=0.0, catch=1.0))

    def test_auto_takes_the_gpu_for_relit_shadows(self):
        for mode in ('raster', 'raytrace'):
            with self.subTest(mode=mode):
                self.set('render', 'render_mode', mode)
                self.check(self.scene())

    def test_the_removed_refusals_are_gone_and_the_others_stay(self):
        source = ''.join(Path(module.__file__).read_text() for module in (gpu3d, gpurt_render))
        self.assertNotIn('caught splat shadows are CPU-only', source)
        self.assertNotIn('splat shadows are CPU-only', source)
        with patch.object(s, 'scene_from_node', return_value=replace(
                self.scene(), geometries=(replace(occluder(), color=(1, 1, 1, .5)),))):
            self.set('render', 'render_backend', 'gpu')
            with self.assertRaisesRegex(ValueError, 'transparent meshes mixed with splats are CPU-only'):
                Evaluator().evaluate(self.d.document, 'render')

    def test_cancelled_graph_render_is_not_swallowed_by_auto(self):
        event = threading.Event()
        event.set()
        with patch.object(s, 'scene_from_node', return_value=self.scene()):
            with self.assertRaises(Cancelled):
                self.set('render', 'render_backend', 'auto')
                Evaluator().evaluate(self.d.document, 'render', cancel=event)


if __name__ == '__main__':
    unittest.main()
