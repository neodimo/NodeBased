"""The adaptive-overhead benchmark (lane 4 Rendering, step Q1): its rows pair a fixed render with adaptive renders of the same
64-sample cap, and its per-pass fit recovers a known cost per pass."""
import unittest

from tools import benchmark_adaptive_overhead as o


class Rows(unittest.TestCase):
    def test_never_stopping_rows_share_the_cap_and_differ_only_in_pass_count(self):
        rows = dict(o.rows_for(None))
        self.assertEqual(rows["fixed 64"].samples, 64)
        never = [s for label, s in rows.items() if label.startswith("never stops")]
        self.assertEqual(len(never), len(o.PASS_SIZES))
        for s in never:
            self.assertEqual((s.min_samples, s.max_samples, s.noise_threshold), (16, 64, o.NEVER))
            self.assertEqual(s.clamped().max_samples, 64)
        self.assertEqual(sorted({s.adaptive_pass_size for s in never}), sorted(o.PASS_SIZES))

    def test_the_two_adaptive_rows_differ_only_in_path_samples(self):
        rows = dict(o.rows_for(None))
        open_, capped = rows["adaptive 0.003, Path samples 65536"], rows["adaptive 0.003, Path samples 64"]
        self.assertEqual((open_.samples, capped.samples), (65536, 64))
        self.assertEqual(open_.clamped().max_samples, 256)
        self.assertEqual(capped.clamped().max_samples, 64)


class Fit(unittest.TestCase):
    def test_the_fit_recovers_milliseconds_per_pass_and_the_fixed_cost(self):
        rows = {f"never stops, pass size {n}": dict(passes=n, wall_ms=100.0 + 0.5 * n) for n in (2, 3, 7, 13, 25)}
        rows["fixed 64"] = dict(passes=64, wall_ms=500.0)         # not part of the fit
        fit = o.per_pass_fit(rows)
        self.assertAlmostEqual(fit["ms_per_pass"], 0.5, places=3)
        self.assertAlmostEqual(fit["intercept_ms"], 100.0, places=2)
        self.assertIsNone(o.per_pass_fit({"fixed 64": dict(passes=64, wall_ms=1.0)}))


if __name__ == "__main__":
    unittest.main()
