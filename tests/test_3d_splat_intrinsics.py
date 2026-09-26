"""Intrinsic decomposition of captured splats: de-lighting, the ReadSplat3D knobs, the cache and the bundle."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from nodebased import intrinsics as I, scene3d as s, splats
from nodebased.cancellation import Cancelled
from nodebased.core import Dispatcher, SPECS, load_document
from nodebased.imaging import Evaluator
from nodebased.renderprogress import progress_text
from nodebased.simcache import SimCache
from tests.test_3d_splat_relight_benchmark import bench

KNOBS = dict(iterations=12, smoothness=0.5, light_order=1)


def card():
    asset = bench.ASSETS['bumpy_card']()
    scenes, fitted = bench.make_scene_set(asset)
    captured = splats.SplatCloud(fitted.positions, fitted.scales, fitted.rotations, fitted.opacity, fitted.sh,
                                 fitted.sh_degree, colorspace=fitted.colorspace)
    return asset, scenes, captured


ASSET, SCENES, CAPTURED = card()
LAYER = I.decompose(CAPTURED, **KNOBS)


class DecompositionTests(unittest.TestCase):
    def test_albedo_is_closer_to_the_truth_than_the_capture(self):
        fitted = np.linalg.norm(LAYER.albedo - ASSET.albedo, axis=1).mean()
        captured = np.linalg.norm(I.capture_colour(CAPTURED) - ASSET.albedo, axis=1).mean()
        self.assertLess(fitted, 0.08)
        self.assertLess(fitted, 0.5 * captured)

    def test_normal_error_drops_against_the_shipped_estimate(self):
        eye = np.asarray(bench.camera_for('bumpy_card').transform.position.array(), dtype=np.float64)
        shipped = bench.normal_error_degrees(ASSET, CAPTURED, eye, 0)['mean']
        fitted_cloud = I.attach(CAPTURED, LAYER)
        delit = bench.normal_error_degrees(ASSET, fitted_cloud, eye, 0, use_intrinsics=True)['mean']
        self.assertLess(delit, 0.4 * shipped)
        self.assertLess(delit, 5.0)
        np.testing.assert_allclose(np.linalg.norm(LAYER.normal, axis=1), 1, atol=1e-5)

    def test_fitted_light_reproduces_the_capture_within_the_reported_error(self):
        capture = I.capture_colour(CAPTURED)
        error = float(np.sqrt(np.mean((LAYER.reproduction() - capture) ** 2)))
        reported = LAYER.report['reproduction_rmse']
        self.assertAlmostEqual(error, reported, delta=2e-4)
        self.assertLess(reported, 0.03)
        # A flat light explains far less of a capture with a strong sun in it.
        self.assertLess(reported, 0.5 * LAYER.report['capture_rms_about_mean'])
        # The stored environment is a real light: a sun direction and its colour.
        self.assertEqual(LAYER.lobe_direction.shape, (1, 3))
        self.assertGreater(float(LAYER.lobe_color.max()), 0)

    def test_deterministic(self):
        again = I.decompose(CAPTURED, **KNOBS)
        for name in ('albedo', 'roughness', 'normal', 'normal_confidence', 'occlusion', 'visibility',
                     'ambient', 'lobe_direction', 'lobe_color'):
            self.assertEqual(getattr(again, name).tobytes(), getattr(LAYER, name).tobytes(), name)

    def test_layers_are_immutable_and_sized_per_splat(self):
        self.assertEqual(len(LAYER), len(CAPTURED))
        self.assertFalse(LAYER.albedo.flags.writeable)
        self.assertTrue(np.all((LAYER.albedo >= 0) & (LAYER.albedo <= 1)))
        self.assertTrue(np.all((LAYER.occlusion >= 0) & (LAYER.occlusion <= 1)))
        self.assertTrue(np.all((LAYER.roughness >= 0.05) & (LAYER.roughness <= 1)))
        with self.assertRaises(ValueError):
            splats.SplatCloud(CAPTURED.positions[:10], CAPTURED.scales[:10], CAPTURED.rotations[:10],
                              CAPTURED.opacity[:10], CAPTURED.sh[:10], 0, intrinsics=LAYER)

    def test_captured_colour_is_untouched_and_light_order_zero_is_a_flat_light(self):
        cloud = I.attach(CAPTURED, LAYER)
        self.assertEqual(cloud.sh.tobytes(), CAPTURED.sh.tobytes())
        self.assertIsNone(CAPTURED.intrinsics)
        flat = I.decompose(CAPTURED, iterations=4, smoothness=0.0, light_order=0)
        self.assertEqual(flat.lobe_direction.shape, (0, 3))
        # No sun to divide out: albedo is the capture up to one scale.
        ratio = flat.albedo / np.maximum(I.capture_colour(CAPTURED), 1e-6)
        self.assertLess(float(ratio.std() / ratio.mean()), 0.05)

    def test_smoothness_trades_reproduction_for_smoother_albedo(self):
        sharp = I.decompose(CAPTURED, iterations=12, smoothness=0.0, light_order=1)
        smooth = I.decompose(CAPTURED, iterations=12, smoothness=1.0, light_order=1)
        self.assertLess(sharp.report['reproduction_rmse'], 1e-4)
        self.assertGreater(smooth.report['reproduction_rmse'], sharp.report['reproduction_rmse'])

    def test_transform_turns_normals_and_lights_together(self):
        cloud = I.attach(CAPTURED, LAYER)
        turn = np.array(((0., 0, 1, 0), (0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 0, 1)))   # 90 degrees about Y
        moved = cloud.transformed(turn)
        np.testing.assert_allclose(moved.intrinsics.normal, LAYER.normal @ turn[:3, :3].T, atol=1e-5)
        np.testing.assert_allclose(moved.intrinsics.lobe_direction, LAYER.lobe_direction @ turn[:3, :3].T, atol=1e-5)
        np.testing.assert_allclose(moved.intrinsics.albedo, LAYER.albedo)
        again = cloud.transformed(np.eye(4))
        self.assertIs(again.intrinsics, LAYER)


class CacheAndCancelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        I._memory.clear()
        self.addCleanup(I._memory.clear)
        self.store = SimCache(root=self.tmp.name, memory_budget=1 << 30, disk_budget=1 << 30)

    def test_second_run_is_a_cache_hit_including_from_disk(self):
        identity = ['plane.ply', 1, 2, 'as_authored', 'linear']
        before = I.FIT_CALLS
        first = I.decompose_cached(CAPTURED, identity, store=self.store, **KNOBS)
        self.assertEqual(I.FIT_CALLS, before + 1)
        again = I.decompose_cached(CAPTURED, identity, store=self.store, **KNOBS)
        self.assertEqual(I.FIT_CALLS, before + 1)
        self.assertIs(again, first)
        I._memory.clear()                       # a new process: only the disk tier remains
        disk = I.decompose_cached(CAPTURED, identity, store=SimCache(root=self.tmp.name), **KNOBS)
        self.assertEqual(I.FIT_CALLS, before + 1)
        self.assertEqual(disk.albedo.tobytes(), first.albedo.tobytes())
        self.assertEqual(disk.report, first.report)
        # A different knob or a different file is a different fit.
        I.decompose_cached(CAPTURED, identity, store=self.store, **dict(KNOBS, smoothness=0.25))
        I.decompose_cached(CAPTURED, ['plane.ply', 1, 3, 'as_authored', 'linear'], store=self.store, **KNOBS)
        self.assertEqual(I.FIT_CALLS, before + 3)

    def test_cancel_raises_and_stores_nothing(self):
        identity = ['cancelled.ply', 1, 2]
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            I.decompose_cached(CAPTURED, identity, store=self.store, cancel=event, **KNOBS)
        self.assertEqual(self.store.stats()['disk_entries'], 0)
        # Cancelling part way, from a progress callback, stops the fit before it finishes.
        event = threading.Event()
        seen = []

        def progress(fraction):
            seen.append(fraction)
            if fraction > 0.3:
                event.set()
        with self.assertRaises(Cancelled):
            I.decompose_cached(CAPTURED, identity, store=self.store, cancel=event, progress=progress, **KNOBS)
        self.assertLess(max(seen), 1.0)
        self.assertEqual(self.store.stats()['disk_entries'], 0)

    def test_progress_is_monotonic_and_ends_at_one(self):
        seen = []
        I.decompose(CAPTURED, progress=seen.append, **KNOBS)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(progress_text('delight', 0.42, {}), 'De-lighting splats  ·  42%')


class ShadingTests(unittest.TestCase):
    def relit(self, cloud, **kw):
        scene = s.Scene(splats=(s.SplatInstance(cloud, relight=1.0, **kw),), lights=SCENES['shipped'].lights)
        return s.render(scene, bench.camera_for('bumpy_card'), 48, 32, ambient=bench.TARGET['ambient'])

    def test_toggle_off_restores_the_captured_colour_path_exactly(self):
        plain = self.relit(CAPTURED)
        with_layer = I.attach(CAPTURED, LAYER)
        off = self.relit(with_layer, use_intrinsics=False)
        on = self.relit(with_layer)
        self.assertEqual(off.tobytes(), plain.tobytes())
        self.assertNotEqual(on.tobytes(), plain.tobytes())

    def test_relit_from_intrinsics_is_closer_to_the_truth(self):
        truth = bench.to_display(s.render(SCENES['truth'], bench.camera_for('bumpy_card'), 48, 32,
                                          ambient=bench.TARGET['ambient']))
        shipped = bench.psnr(bench.to_display(self.relit(CAPTURED)), truth)
        delit = bench.psnr(bench.to_display(self.relit(I.attach(CAPTURED, LAYER))), truth)
        self.assertGreater(delit, shipped + 5)

    def test_captured_colour_render_is_unchanged_by_the_layer(self):
        camera = bench.camera_for('bumpy_card')
        bare = s.render(s.Scene(splats=(s.SplatInstance(CAPTURED),)), camera, 48, 32)
        layered = s.render(s.Scene(splats=(s.SplatInstance(I.attach(CAPTURED, LAYER)),)), camera, 48, 32)
        self.assertEqual(bare.tobytes(), layered.tobytes())


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'card.ply'
        splats.write_ply(CAPTURED, self.path)
        with splats._cloud_cache_lock:
            splats._cloud_cache.clear()
        I._memory.clear()
        self.addCleanup(I._memory.clear)
        self.evaluator = Evaluator(sim=SimCache(root=Path(self.tmp.name) / 'sim'))
        self.d = Dispatcher()
        for key, kind, params in (
                ('read', 'ReadSplat3D', dict(splat_path=str(self.path), splat_colorspace='linear', splat_relight=1.0,
                                             splat_sh_degree=0)),
                ('light', 'Light3D', dict(tx=7., ty=7.5, tz=3.5, intensity=1.2)),
                ('scene', 'Scene3D', {}), ('camera', 'Camera3D', dict(ty=2.6, tz=2.2)),
                ('render', 'Render3D', dict(width=48, height=32, samples=1, ambient=0.1, render_output='relight')),
                ('relight', 'Relight', dict(red=.1, green=.1, blue=.1))):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        for target, slot, source in (('scene', 'object0', 'read'), ('scene', 'object1', 'light'),
                                     ('render', 'scene', 'scene'), ('render', 'camera', 'camera'),
                                     ('relight', 'image', 'render'), ('relight', 'light0', 'light')):
            self.d.execute(dict(op='connect', id=target, input=slot, source=source))

    def set(self, key, param, value):
        self.d.execute(dict(op='set', id=key, param=param, value=value))

    def cloud(self):
        value = self.evaluator.evaluate_raster(self.d.document, 'read', typed=True)
        return value.splats[0].cloud

    def test_defaults_and_old_documents_are_unaffected_when_delight_is_off(self):
        params = SPECS['ReadSplat3D']['params']
        self.assertEqual((params['splat_delight'], params['splat_delight_iterations'],
                          params['splat_delight_smoothness'], params['splat_delight_light_order'],
                          params['splat_use_intrinsics']), ('off', 12, 0.5, 1, 'on'))
        before = I.FIT_CALLS
        self.assertIsNone(self.cloud().intrinsics)
        self.assertEqual(I.FIT_CALLS, before)
        # A document saved before the knobs existed loads with them off.
        doc = copy.deepcopy(self.d.document)
        for name in list(doc['nodes']['read']['params']):
            if name.startswith('splat_delight') or name == 'splat_use_intrinsics':
                del doc['nodes']['read']['params'][name]
        old_file = Path(self.tmp.name) / 'old.nbcomp'
        old_file.write_text(json.dumps(doc))
        loaded = load_document(old_file)
        self.assertEqual(loaded['nodes']['read']['params']['splat_delight'], 'off')
        self.assertEqual(loaded['nodes']['read']['params']['splat_use_intrinsics'], 'on')

    def test_delight_on_attaches_the_layer_and_a_second_evaluation_is_a_cache_hit(self):
        self.set('read', 'splat_delight', 'on')
        events = []
        self.evaluator.progress = lambda stage, fraction, info: events.append((stage, fraction))
        before = I.FIT_CALLS
        cloud = self.cloud()
        self.assertIsNotNone(cloud.intrinsics)
        self.assertEqual(I.FIT_CALLS, before + 1)
        self.assertEqual(cloud.sh.tobytes(), CAPTURED.sh.tobytes())      # the capture is kept beside it
        self.assertTrue(any(stage == 'delight' for stage, _ in events))
        self.assertEqual(events[-1], ('delight', 1.0))
        # A different, cheap knob elsewhere on the node does not refit; a fit knob does.
        self.set('read', 'splat_opacity', 0.9)
        self.cloud()
        self.assertEqual(I.FIT_CALLS, before + 1)
        self.set('read', 'splat_delight_smoothness', 0.25)
        self.cloud()
        self.assertEqual(I.FIT_CALLS, before + 2)
        # A fresh evaluator over the same disk cache does not refit either.
        I._memory.clear()
        again = Evaluator(sim=SimCache(root=Path(self.tmp.name) / 'sim'))
        again.evaluate_raster(self.d.document, 'read', typed=True)
        self.assertEqual(I.FIT_CALLS, before + 2)

    def test_cancelling_the_evaluation_cancels_the_fit(self):
        self.set('read', 'splat_delight', 'on')
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            self.evaluator.evaluate_raster(self.d.document, 'read', cancel=event, typed=True)

    def test_use_intrinsics_knob_switches_the_look(self):
        self.set('read', 'splat_delight', 'on')
        on = self.evaluator.evaluate_raster(self.d.document, 'relight').pixels.copy()
        self.set('read', 'splat_use_intrinsics', 'off')
        off = self.evaluator.evaluate_raster(self.d.document, 'relight').pixels
        self.set('read', 'splat_delight', 'off')
        self.set('read', 'splat_use_intrinsics', 'on')
        plain = self.evaluator.evaluate_raster(self.d.document, 'relight').pixels
        self.assertEqual(off.tobytes(), plain.tobytes())
        self.assertNotEqual(on.tobytes(), plain.tobytes())

    def test_relight_node_consumes_the_splat_bundle_and_occlusion_toggle(self):
        self.set('read', 'splat_delight', 'on')
        bundle = self.evaluator.evaluate_raster(self.d.document, 'render')
        for name in ('albedo', 'normals', 'roughness', 'occlusion', 'diffuse_L0', 'specular_L0', 'position'):
            self.assertIn(name, bundle.layers, name)
        with_occlusion = self.evaluator.evaluate_raster(self.d.document, 'relight').pixels.copy()
        self.set('relight', 'use_intrinsics', 'off')
        without = self.evaluator.evaluate_raster(self.d.document, 'relight').pixels
        self.assertNotEqual(with_occlusion.tobytes(), without.tobytes())
        self.assertTrue(np.all(np.isfinite(with_occlusion)))
        # With Use occlusion off the recombination is exactly the documented sum.
        layers = {name: raster.pixels for name, raster in bundle.layers.items()}
        colour = np.full(3, 1.2, np.float32)          # the Light3D node: white, intensity 1.2
        expected = layers['albedo'][..., :3] * (0.1 + layers['diffuse_L0'][..., :3] * colour) \
            + layers['specular_L0'][..., :3] * colour
        np.testing.assert_allclose(without[..., :3], expected, atol=1e-5)


class BundleTests(unittest.TestCase):
    def scene(self, cloud, **kw):
        return s.Scene(splats=(s.SplatInstance(cloud, relight=1.0, **kw),), lights=SCENES['shipped'].lights)

    def test_bundle_sum_equals_the_relit_beauty(self):
        camera = bench.camera_for('bumpy_card')
        for cloud in (CAPTURED, I.attach(CAPTURED, LAYER)):
            scene = self.scene(cloud)
            out, channels = s.render(scene, camera, 40, 28, ambient=0.1, output='relight')
            beauty = s.render(scene, camera, 40, 28, ambient=0.1)
            np.testing.assert_allclose(out[..., :3], beauty[..., :3], atol=1e-5)
            np.testing.assert_allclose(out[..., 3], beauty[..., 3], atol=1e-6)
            self.assertEqual(set(channels), {'albedo', 'roughness', 'occlusion', 'diffuse', 'specular', 'emission',
                                             'diffuse_L0', 'specular_L0', 'normals', 'position'})

    def test_albedo_pass_is_the_delit_layer_when_used_and_the_capture_otherwise(self):
        camera = bench.camera_for('bumpy_card')
        cloud = I.attach(CAPTURED, LAYER)
        _, on = s.render(self.scene(cloud), camera, 40, 28, ambient=0.1, output='relight')
        _, off = s.render(self.scene(cloud, use_intrinsics=False), camera, 40, 28, ambient=0.1, output='relight')
        _, plain = s.render(self.scene(CAPTURED), camera, 40, 28, ambient=0.1, output='relight')
        self.assertEqual(off['albedo'].tobytes(), plain['albedo'].tobytes())
        self.assertNotEqual(on['albedo'].tobytes(), plain['albedo'].tobytes())
        self.assertLess(float(on['occlusion'][..., :3].mean()), float(plain['occlusion'][..., :3].mean()))

    def test_multichannel_carries_the_splat_relight_layers_and_mixed_scenes_are_refused(self):
        camera = bench.camera_for('bumpy_card')
        _, layers = s.render_multichannel(self.scene(I.attach(CAPTURED, LAYER)), camera, 32, 20,
                                          passes='relight', ambient=0.1)
        self.assertEqual(sorted(layers), ['relight_light1_diffuse', 'relight_light1_specular'])
        mixed = s.Scene((s._card(3, 3, (.2, .3, .4, .5), s.Transform3D()),), splats=self.scene(CAPTURED).splats,
                        lights=SCENES['shipped'].lights)
        with self.assertRaisesRegex(ValueError, 'splats that also hold geometry'):
            s.render(mixed, camera, 32, 20, output='relight')


if __name__ == '__main__':
    unittest.main()
