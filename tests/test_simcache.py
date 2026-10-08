import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from nodebased import simcache
from nodebased.cancellation import Cancelled


class SimCacheTests(unittest.TestCase):
    def solver(self, calls=None, sized=False, cancel=None):
        def initial(seed):
            value = float(seed)
            array = np.zeros(32, dtype=np.float64) if sized else np.array([value])
            return simcache.State({"x": array + value}, {"steps": 0})

        def step(state, frame, substep, seed):
            if calls is not None:
                calls[0] += 1
                if cancel is not None and calls[0] == 5:
                    cancel.set()
            rng = np.random.default_rng((seed, frame, substep))
            value = state.arrays["x"] + 1.0 + rng.uniform(-0.01, 0.01)
            return simcache.State({"x": value}, {"steps": state.meta["steps"] + 1})
        return initial, step

    def test_reproducible_seeds(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            initial, step = self.solver()
            first = simcache.solve_to_frame(simcache.SimCache(a), "run", 8, 0, 2, 7, initial, step)
            second = simcache.solve_to_frame(simcache.SimCache(b), "run", 8, 0, 2, 7, initial, step)
            different = simcache.solve_to_frame(simcache.SimCache(), "run", 8, 0, 2, 8, initial, step)
            self.assertEqual(first, second)
            self.assertNotEqual(first, different)

    def test_frame_by_frame_reuse_and_determinism(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            calls_a = [0]
            initial, step = self.solver(calls_a)
            direct = simcache.solve_to_frame(simcache.SimCache(a), "run", 50, 1, 2, 3, initial, step)
            self.assertEqual(calls_a[0], 100)
            cache_b = simcache.SimCache(b)
            calls_b = [0]
            initial, step = self.solver(calls_b)
            simcache.solve_to_frame(cache_b, "run", 20, 1, 2, 3, initial, step)
            calls_b[0] = 0
            continued = simcache.solve_to_frame(cache_b, "run", 50, 1, 2, 3, initial, step)
            self.assertEqual(calls_b[0], 60)
            self.assertEqual(direct, continued)
            simcache.solve_to_frame(cache_b, "run", 50, 1, 2, 3, initial, step)
            self.assertEqual(calls_b[0], 60)

    def test_invalidation_on_upstream_or_parameter_change(self):
        run_a = simcache.run_key("upstream-a", {"rate": 1})
        run_b = simcache.run_key("upstream-b", {"rate": 1})
        run_c = simcache.run_key("upstream-a", {"rate": 2})
        self.assertNotEqual(run_a, run_b)
        self.assertNotEqual(run_a, run_c)
        cache = simcache.SimCache()
        initial, step = self.solver()
        simcache.solve_to_frame(cache, run_a, 10, 0, 1, 1, initial, step)
        simcache.solve_to_frame(cache, run_b, 10, 0, 1, 2, initial, step)
        self.assertIsNotNone(cache.get(run_a, 10))
        self.assertIsNotNone(cache.get(run_b, 10))
        self.assertIsNone(cache.latest_at_or_before("missing", 10))

    def test_restart(self):
        with tempfile.TemporaryDirectory() as root:
            calls = [0]
            initial, step = self.solver(calls)
            first = simcache.solve_to_frame(simcache.SimCache(root), "run", 30, 1, 2, 4, initial, step)
            calls[0] = 0
            second_cache = simcache.SimCache(root)
            second = simcache.solve_to_frame(second_cache, "run", 30, 1, 2, 4, initial, step)
            self.assertEqual(calls[0], 0)
            self.assertEqual(first, second)
            simcache.solve_to_frame(second_cache, "run", 40, 1, 2, 4, initial, step)
            self.assertEqual(calls[0], 20)

    def test_disk_frame_load_preserves_fields(self):
        with tempfile.TemporaryDirectory() as root:
            run = "b" * 64
            expected = simcache.State({
                "coords": np.arange(30, dtype=np.int32).reshape(10, 3),
                "density": np.linspace(0, 1, 64, dtype=np.float32).reshape(4, 4, 4),
            }, {"frame": 7, "shape": [4, 4, 4]})
            simcache.SimCache(root).put(run, 7, expected)

            loaded = simcache.SimCache(root).get(run, 7)

            self.assertEqual(expected, loaded)

    def test_legacy_npz_and_packed_cache_load_to_identical_fields(self):
        with tempfile.TemporaryDirectory() as root:
            run, frame = "c" * 64, 12
            expected = simcache.State({
                "coords": np.arange(12, dtype=np.int32).reshape(4, 3),
                "density": np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 3, 4),
            }, {"frame": frame, "origin": [0.25, -1.5, 3.0]})
            cache = simcache.SimCache(root)
            legacy_path = cache._legacy_path((run, frame))
            legacy_path.parent.mkdir(parents=True)
            with legacy_path.open("wb") as stream:
                np.savez(stream, **expected.arrays, __meta__=np.array(json.dumps(expected.meta)))

            from_legacy = simcache.SimCache(root).get(run, frame)
            from_packed = simcache.SimCache(root).get(run, frame)

            self.assertEqual(expected, from_legacy)
            self.assertEqual(expected, from_packed)
            self.assertEqual(from_legacy, from_packed)
            self.assertTrue(cache._path((run, frame)).is_file())
            self.assertFalse(legacy_path.exists())

    def test_scoped_cache_reads_and_migrates_direct_layout_entries(self):
        with tempfile.TemporaryDirectory() as root:
            run, frame = "e" * 64, 4
            old = simcache.SimCache(root)
            expected = simcache.State({"x": np.arange(6, dtype=np.float32)}, {"frame": frame})
            old.put(run, frame, expected)
            scoped = simcache.SimCache(Path(root) / "stores" / "256-2048",
                                       legacy_root=root)

            self.assertEqual(scoped.get(run, frame), expected)
            self.assertTrue(scoped._path((run, frame)).is_file())
            self.assertTrue(old._path((run, frame)).is_file())
            self.assertEqual(scoped.frames(run), [frame])

    def test_scoped_cache_uses_legacy_frames_as_solve_anchors(self):
        with tempfile.TemporaryDirectory() as root:
            run = "f" * 64
            old = simcache.SimCache(root)
            expected = simcache.State({"x": np.array([2.0])}, {"frame": 2})
            old.put(run, 2, expected)
            scoped = simcache.SimCache(Path(root) / "stores" / "256-2048",
                                       legacy_root=root)

            anchor = scoped.latest_at_or_before(run, 4)

            self.assertIsNotNone(anchor)
            self.assertEqual(anchor[0], 2)
            self.assertEqual(anchor[1], expected)

    def test_evaluator_budget_pairs_have_independent_disk_namespaces(self):
        from nodebased.imaging import Evaluator

        with tempfile.TemporaryDirectory() as root:
            evaluator = Evaluator(sim=simcache.SimCache(root))
            roomy = evaluator.sim_store(1, 4)
            tiny = evaluator.sim_store(1, 1)
            run = "9" * 64
            expected = simcache.State({"x": np.arange(4, dtype=np.float32)})
            roomy.put(run, 3, expected)
            # This entry exceeds the tiny scope's one-MiB budget and is evicted there.
            tiny.put(run, 4, simcache.State({"x": np.arange(300_000, dtype=np.float32)}))

            self.assertNotEqual(roomy.root, tiny.root)
            self.assertTrue(roomy._path((run, 3)).is_file())
            self.assertEqual(simcache.SimCache(roomy.root).get(run, 3), expected)

    def test_cancellation(self):
        cancel = threading.Event()
        calls = [0]
        initial, step = self.solver(calls, cancel=cancel)
        cache = simcache.SimCache()
        with self.assertRaises(Cancelled):
            simcache.solve_to_frame(cache, "run", 6, 0, 2, 1, initial, step, cancel)
        self.assertIsNotNone(cache.get("run", 1))
        calls[0] = 0
        initial, step = self.solver(calls)
        simcache.solve_to_frame(cache, "run", 6, 0, 2, 1, initial, step)
        self.assertEqual(calls[0], 10)

    def test_budget_eviction(self):
        with tempfile.TemporaryDirectory() as root:
            initial, step = self.solver(sized=True)
            cache = simcache.SimCache(root, memory_budget=200, disk_budget=3000)
            expected = simcache.solve_to_frame(cache, "run", 12, 1, 1, 2, initial, step)
            self.assertGreater(cache.stats()["evictions"], 0)
            self.assertIsNone(cache.get("run", 1))
            self.assertIsNotNone(cache.get("run", 12))
            actual = simcache.solve_to_frame(cache, "run", 12, 1, 1, 2, initial, step)
            self.assertEqual(expected, actual)

    def test_before_start_frame(self):
        calls = [0]
        initial, step = self.solver(calls)
        cache = simcache.SimCache()
        for target, start in [(-10, -5), (-6, -5), (0, 3)]:
            result = simcache.solve_to_frame(cache, "run", target, start, 2, 1, initial, step)
            self.assertEqual(result, initial(1))
        self.assertEqual(calls[0], 0)
        self.assertEqual(cache.stats()["memory_entries"], 0)

    def test_run_key_changes_with_either_input(self):
        same = simcache.run_key("u", {"a": 1})
        self.assertEqual(same, simcache.run_key("u", {"a": 1}))
        self.assertNotEqual(same, simcache.run_key("v", {"a": 1}))
        self.assertNotEqual(same, simcache.run_key("u", {"a": 2}))
        self.assertRegex(same, r"^[0-9a-f]{64}$")

    def test_corrupt_disk_entry_is_miss(self):
        with tempfile.TemporaryDirectory() as root:
            initial, step = self.solver()
            cache = simcache.SimCache(root)
            simcache.solve_to_frame(cache, "a" * 64, 3, 0, 1, 1, initial, step)
            path = Path(root) / "aa" / ("a" * 64) / "0000000003.nbc"
            path.write_bytes(b"not an npz")
            fresh = simcache.SimCache(root)
            self.assertIsNone(fresh.get("a" * 64, 3))
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
