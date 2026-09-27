import tempfile
import unittest

import numpy as np

from nodebased import simcache, simstats


class SnapshotTextTests(unittest.TestCase):
    def test_text_reports_frame_substep_and_timing(self):
        snap = simstats.SimStatsSnapshot(frame=12, substep=1, substeps=4, ms_per_substep=3.5)
        text = snap.text()
        self.assertIn("Frame 12", text)
        self.assertIn("substep 2/4", text)
        self.assertIn("3.50 ms/substep", text)
        self.assertNotIn("GPU", text)

    def test_gpu_memory_is_included_when_given(self):
        snap = simstats.SimStatsSnapshot(frame=1, substep=0, substeps=1, ms_per_substep=1.0,
                                         gpu_memory_mb=512.0)
        self.assertIn("GPU mem 512 MB", snap.text())


class SolveWithStatsTests(unittest.TestCase):
    def test_overlay_updates_during_a_short_solve(self):
        def initial_state(seed):
            return simcache.State({"x": np.zeros(1, np.float32)}, {})

        def step(state, frame, substep, seed):
            return simcache.State({"x": state.arrays["x"] + 1}, {})

        snapshots = []
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            simstats.solve_with_stats(cache, "run", 3, 1, 2, 0, initial_state, step,
                                      snapshots.append)
        # 3 frames * 2 substeps = 6 callbacks, each with the frame/substep it actually computed.
        self.assertEqual(len(snapshots), 6)
        self.assertEqual([s.frame for s in snapshots], [1, 1, 2, 2, 3, 3])
        self.assertEqual([s.substep for s in snapshots], [0, 1, 0, 1, 0, 1])
        # The overlay text changes as the solve advances (frame/substep move on).
        texts = {s.text() for s in snapshots}
        self.assertGreater(len(texts), 1)

    def test_a_cache_hit_calls_back_zero_times(self):
        def initial_state(seed):
            return simcache.State({"x": np.zeros(1, np.float32)}, {})

        def step(state, frame, substep, seed):
            return simcache.State({"x": state.arrays["x"] + 1}, {})

        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            simstats.solve_with_stats(cache, "run", 2, 1, 1, 0, initial_state, step, lambda s: None)
            snapshots = []
            simstats.solve_with_stats(cache, "run", 2, 1, 1, 0, initial_state, step, snapshots.append)
        self.assertEqual(snapshots, [])


if __name__ == "__main__":
    unittest.main()
