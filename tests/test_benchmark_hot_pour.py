import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nodebased.simcache import SimCache, State
from tools.benchmark_hot_pour import cache_report, percentile


class HotPourBenchmarkCacheReportTests(unittest.TestCase):
    def test_nearest_rank_p95_is_reported_conservatively(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95), 10.0)

    def test_report_counts_packed_sparse_frames_and_dense_equivalent(self):
        with tempfile.TemporaryDirectory() as root:
            run = "d" * 64
            cache = SimCache(Path(root) / "stores" / "256-2048")
            cache.put(run, 3, State({
                "coords": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
                "density": np.ones((2, 8, 8, 8), dtype=np.float32),
                "velocity": np.ones((2, 8, 8, 8, 3), dtype=np.float32),
            }, {"domain_shape": [16, 16, 8]}))
            packed = cache._path((run, 3))

            report = cache_report(root, SimpleNamespace())

            self.assertEqual(report["runs"][run[:10]]["kind"], "steam volume (sparse tiles)")
            self.assertEqual(report["runs"][run[:10]]["frames"], 1)
            self.assertEqual(report["steam_sparse_bytes"], packed.stat().st_size)
            self.assertEqual(report["steam_dense_equivalent_bytes"], 16 * 16 * 8 * 4 * 4)

    def test_report_identifies_cached_liquid_surface_meshes(self):
        with tempfile.TemporaryDirectory() as root:
            run = "s" * 64
            cache = SimCache(Path(root) / "stores" / "256-2048")
            cache.put(run, 60, State({
                "vertices": np.zeros((3, 3), dtype=np.float32),
                "triangles": np.array([[0, 1, 2]], dtype=np.int32),
                "normals": np.ones((3, 3), dtype=np.float32),
            }))

            report = cache_report(root, SimpleNamespace())

            self.assertEqual(report["runs"][run[:10]]["kind"], "liquid surface mesh")
            self.assertEqual(report["runs"][run[:10]]["frames"], 1)


if __name__ == "__main__":
    unittest.main()
