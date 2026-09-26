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
        for name, expected in self.baseline.items():
            actual = self.runs[name][0]
            self.assertEqual(actual['splats'], expected['splats'])
            for cond in bench.CONDITIONS:
                with self.subTest(scene=name, condition=cond):
                    self.assertAlmostEqual(actual[cond]['psnr'], expected[cond]['psnr'], delta=PSNR_TOL)
                    self.assertAlmostEqual(actual[cond]['ssim'], expected[cond]['ssim'], delta=SSIM_TOL)
            for which in ('shipped', 'smoothed'):
                for stat in ('mean', 'median', 'p90'):
                    self.assertAlmostEqual(actual['normal_error_deg'][which][stat],
                                           expected['normal_error_deg'][which][stat], delta=DEG_TOL,
                                           msg=f'{name} {which} {stat}')

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

    @unittest.skipUnless(os.environ.get('NB_SCENE_PLY') and Path(os.environ.get('NB_SCENE_PLY', '')).exists(),
                         'set NB_SCENE_PLY to the shared capture to run the read-only scene')
    def test_shared_capture_runs(self):
        result, _ = bench.run_scene_ply(os.environ['NB_SCENE_PLY'], count=4000, size=(64, 36))
        self.assertEqual(result['splats'], 4000)
        self.assertTrue(0 <= result['confidence_mean'] <= 1)


if __name__ == '__main__':
    unittest.main()
