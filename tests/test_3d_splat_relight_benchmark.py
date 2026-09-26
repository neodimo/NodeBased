"""The splat relighting benchmark (tools/benchmark_relight.py): deterministic, and today's numbers pinned.

If a change moves a number on purpose, run `python tools/benchmark_relight.py --write-refs` and
commit the refreshed tests/data/relight_benchmark files together with the note in docs/SPLAT_RELIGHTING.md.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from nodebased import scene3d as s
from nodebased.splatshade import shade_splats, normal_confidence, splat_albedo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'tests' / 'data' / 'relight_benchmark'
spec = importlib.util.spec_from_file_location('benchmark_relight', ROOT / 'tools' / 'benchmark_relight.py')
bench = importlib.util.module_from_spec(spec)
sys.modules['benchmark_relight'] = bench
spec.loader.exec_module(bench)

PSNR_TOL, SSIM_TOL, DEG_TOL = 0.25, 0.004, 0.6


class MetricTests(unittest.TestCase):
    def test_psnr_and_ssim_contracts(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(size=(40, 50, 3))
        self.assertEqual(bench.psnr(a, a), 99.0)
        self.assertAlmostEqual(bench.psnr(np.zeros((4, 4, 3)), np.full((4, 4, 3), .1)), 20.0, places=9)
        self.assertAlmostEqual(bench.ssim(a, a), 1.0, places=12)
        self.assertLess(bench.ssim(a, np.clip(a + rng.normal(0, .2, a.shape), 0, 1)), .9)
        flat = np.full((32, 32, 3), .5)
        self.assertLess(bench.ssim(flat, flat * .5), .95)  # a brightness change registers

    def test_display_transform_is_srgb(self):
        d = bench.to_display(np.array([[[0.2140411, 0.0, 1.0, 1.0]]]))
        np.testing.assert_allclose(d[0, 0], (0.5, 0.0, 1.0), atol=1e-6)

    def test_effective_normal_matches_the_shader(self):
        asset = bench.ASSETS['bumpy_card']()
        cloud = asset.cloud(asset.albedo, 7)
        eye = np.array((0.0, 2.6, 2.2))
        light = s.Light(kind='Directional', position=s.Vec3(1, 2, 1), target=s.Vec3(), intensity=1.0)
        toward = bench._unit(-light.world()[1])
        lit = shade_splats(np.zeros((len(cloud), 3)), np.ones((len(cloud), 3)), cloud.positions,
                           cloud.normals(), normal_confidence(cloud.scales), eye, (light,), 0.0, 1.0)
        expected = np.maximum(bench.effective_normals(cloud, eye, 0) @ toward, 0)
        np.testing.assert_allclose(lit[:, 0], expected, atol=1e-6)


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads((DATA / 'baseline.json').read_text())
        cls.runs = {name: bench.run_asset(name) for name in bench.ASSETS}

    def test_deterministic(self):
        for name in bench.ASSETS:
            again, images = bench.run_asset(name)
            self.assertEqual(again, self.runs[name][0])
            for cond, image in self.runs[name][1].items():
                self.assertEqual(images[cond].tobytes(), image.tobytes())

    def test_baseline_numbers_hold(self):
        for name in bench.ASSETS:
            expected, actual = self.baseline[name], self.runs[name][0]
            self.assertEqual(actual['splats'], expected['splats'])
            for cond in bench.CONDITIONS:
                with self.subTest(scene=name, condition=cond):
                    self.assertAlmostEqual(actual[cond]['psnr'], expected[cond]['psnr'], delta=PSNR_TOL)
                    self.assertAlmostEqual(actual[cond]['ssim'], expected[cond]['ssim'], delta=SSIM_TOL)
            for which in ('shipped', 'smoothed', 'delit'):
                for stat in ('mean', 'median', 'p90'):
                    self.assertAlmostEqual(actual['normal_error_deg'][which][stat],
                                           expected['normal_error_deg'][which][stat], delta=DEG_TOL,
                                           msg=f'{name} {which} {stat}')

    def test_delight_numbers_hold(self):
        for name in bench.ASSETS:
            expected, actual = self.baseline[name], self.runs[name][0]['delight']
            for key, value in expected['delight'].items():
                self.assertAlmostEqual(actual[key], value, delta=0.02, msg=f'{name} {key}')

    def test_delight_result_is_asserted_per_scene(self):
        # Recorded in docs/SPLAT_RELIGHTING.md, "Step B: measured". The card has one sun and no cast
        # shadow, the friendly case: de-lighting beats even the true-albedo oracle because its normals
        # are also better. The sphere on a ground has a cast shadow the fit cannot separate from the
        # ground's stripes: it must not lose ground against the shipped path, and it does not win.
        card, sphere = self.runs['bumpy_card'][0], self.runs['sphere_ground'][0]
        self.assertGreater(card['delit']['psnr'], card['shipped']['psnr'] + 8)
        self.assertGreater(card['delit']['ssim'], card['oracle']['ssim'])
        self.assertLess(card['delight']['albedo_error'], 0.5 * card['delight']['captured_albedo_error'])
        self.assertGreater(sphere['delit']['psnr'], sphere['shipped']['psnr'] - 0.3)
        self.assertGreater(sphere['delit']['ssim'], sphere['shipped']['ssim'] - 0.01)
        for result in (card, sphere):
            self.assertLess(result['normal_error_deg']['delit']['mean'], 0.25 * result['normal_error_deg']['shipped']['mean'])
            self.assertGreater(result['true_normals']['psnr'], result['shipped']['psnr'] - 0.05)

    def test_the_benchmark_ranks_the_conditions(self):
        # A benchmark that cannot tell a perfect de-lighting from the raw capture measures nothing.
        for name, (result, _) in self.runs.items():
            self.assertGreater(result['oracle']['ssim'], result['shipped']['ssim'], name)
            self.assertGreater(result['oracle']['psnr'], result['shipped']['psnr'] + 2, name)
            self.assertGreater(result['shipped']['psnr'], result['baked']['psnr'], name)
            self.assertLess(result['normal_error_deg']['shipped']['mean'], 30, name)

    def test_reference_pngs_match_the_renders(self):
        for name in bench.ASSETS:
            for cond in ('truth',) + bench.CONDITIONS:
                path = DATA / f'{name}_{cond}.png'
                self.assertTrue(path.exists(), path)
                actual = bench.to_display(self.runs[name][1][cond])
                self.assertGreater(bench.psnr(actual, bench.load_png(path)), 45, f'{name} {cond}')
                self.assertLess(path.stat().st_size, 40_000)

    def test_the_full_pipeline_moves_the_direct_only_scenes_by_little(self):
        # Their truth is Lambert with a flat ambient and no bounce (docs/SPLAT_RELIGHTING.md, step D), so the
        # traced terms can only cost points; the pipeline must not wreck either scene.
        for name, (result, _) in self.runs.items():
            self.assertGreater(result['full']['psnr'], result['delit']['psnr'] - 0.5, name)
            self.assertGreater(result['full']['ssim'], result['delit']['ssim'] - 0.01, name)

    @unittest.skipUnless(os.environ.get('NB_SCENE_PLY') and Path(os.environ.get('NB_SCENE_PLY', '')).exists(),
                         'set NB_SCENE_PLY to the shared capture to run the read-only scene')
    def test_shared_capture_runs(self):
        result, _ = bench.run_scene_ply(os.environ['NB_SCENE_PLY'], count=4000, size=(64, 36))
        self.assertEqual(result['splats'], 4000)
        self.assertTrue(0 <= result['confidence_mean'] <= 1)


class BleedTests(unittest.TestCase):
    """Indirect light and occlusion against their analytic truth (form factors of a floor and a red wall)."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads((DATA / 'baseline.json').read_text())['bleed']
        cls.result, cls.images = bench.run_bleed()

    def test_the_quadrature_agrees_with_the_closed_form_of_an_infinite_wall(self):
        # A floor point at distance d from the foot of an infinite wall of height H sees F = (1 - d / sqrt(d^2 + H^2)) / 2.
        point, normal = np.array(((0.5, 0.0, 0.0),)), np.array(((0.0, 1.0, 0.0),))
        wide = ((1.0, 0.0, -60.0), (0, 1, 0), (0, 0, 1), 2.0, 120.0, (-1, 0, 0))
        d, h = 0.5, 2.0
        self.assertAlmostEqual(float(bench.form_factor(point, normal, wide, step=0.05)[0]),
                               0.5 * (1 - d / np.hypot(d, h)), delta=0.004)

    def test_numbers_hold(self):
        for scene in ('wall_bleed', 'crease_ao'):
            for condition, expected in self.baseline[scene].items():
                actual = self.result[scene][condition]
                with self.subTest(scene=scene, condition=condition):
                    self.assertAlmostEqual(actual['psnr'], expected['psnr'], delta=PSNR_TOL)
                    self.assertAlmostEqual(actual['ssim'], expected['ssim'], delta=SSIM_TOL)
                    self.assertAlmostEqual(actual['floor_error'], expected['floor_error'], delta=0.02)
        self.assertEqual(self.result['form_factor']['splats'], self.baseline['form_factor']['splats'])

    def test_the_bounce_recovers_the_colour_bleed_and_more_samples_help(self):
        bleed = self.result['wall_bleed']
        self.assertGreater(bleed['off']['floor_error'], 0.7)           # without it the floor misses its light entirely
        self.assertLess(bleed['final']['floor_error'], 0.25)
        self.assertLess(bleed['final']['floor_error'], bleed['medium']['floor_error'])
        self.assertLess(bleed['medium']['floor_error'], bleed['preview']['floor_error'])
        self.assertGreater(bleed['final']['psnr'], bleed['off']['psnr'] + 10)
        self.assertGreater(bleed['final']['ssim'], 0.98)

    def test_the_occlusion_recovers_the_crease(self):
        ao = self.result['crease_ao']
        self.assertGreater(ao['off']['floor_error'], 0.2)
        self.assertLess(ao['final']['floor_error'], 0.12)
        self.assertGreater(ao['final']['psnr'], ao['off']['psnr'] + 3)

    def test_denoising_does_not_cost_accuracy(self):
        for scene in ('wall_bleed', 'crease_ao'):
            plain, smooth = self.result[scene]['final'], self.result[scene]['final_denoised']
            self.assertLessEqual(smooth['floor_error'], plain['floor_error'] + 0.005)
            self.assertGreater(smooth['psnr'], plain['psnr'] - 0.05)

    def test_reference_pngs_match_the_renders(self):
        for scene in ('wall_bleed', 'crease_ao'):
            for key in ('truth', 'off', 'final', 'final_denoised'):
                path = DATA / f'{scene}_{key}.png'
                self.assertTrue(path.exists(), path)
                actual = bench.to_display(self.images[f'{scene}_{key}'])
                self.assertGreater(bench.psnr(actual, bench.load_png(path)), 45, f'{scene} {key}')
                self.assertLess(path.stat().st_size, 40_000)


class TimingTests(unittest.TestCase):
    def test_timings_report_every_preset_or_skip_without_an_adapter(self):
        from nodebased import gpu3d
        result = bench.run_timings(size=(64, 36), repeats=1)
        if not gpu3d.available():
            self.assertTrue(str(result).startswith('skipped'))
            return
        for scene in ('bumpy_card', 'sphere_ground', 'wall_bleed'):
            rays = [result[scene][label]['rays'] for label in ('off', 'preview', 'medium', 'final')]
            self.assertEqual(rays[0], 0)
            self.assertEqual(rays[2], 4 * rays[1])              # the presets scale the sample counts 4 : 16 : 64
            self.assertEqual(rays[3], 4 * rays[2])
            self.assertTrue(all(result[scene][label]['seconds'] > 0 for label in ('off', 'preview', 'medium', 'final')))


if __name__ == '__main__':
    unittest.main()
