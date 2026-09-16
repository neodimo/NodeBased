import threading
import time
import unittest

import numpy as np

from nodebased.decodepool import DecodeAheadPool


def _frame(value=1.0, shape=(4, 4, 4)):
    return np.full(shape, value, dtype=np.float32)


class DecodeAheadPoolTests(unittest.TestCase):
    def test_request_populates_the_cache_and_get_reports_hits_and_misses(self):
        pool = DecodeAheadPool(max_workers=2, budget_bytes=10_000_000)
        try:
            self.assertIsNone(pool.get("k1"))
            pool.request("k1", lambda: _frame(1.0))
            deadline = time.monotonic() + 5
            while pool.get("k1") is None and time.monotonic() < deadline:
                time.sleep(0.005)
            result = pool.get("k1")
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result, _frame(1.0))
            self.assertGreaterEqual(pool.hits, 1)
            self.assertGreaterEqual(pool.misses, 1)
        finally:
            pool.shutdown()

    def test_request_for_an_already_cached_or_pending_key_does_not_redecode(self):
        pool = DecodeAheadPool(max_workers=1, budget_bytes=10_000_000)
        try:
            calls = []
            started = threading.Event()
            release = threading.Event()

            def slow_decode():
                calls.append(1)
                started.set()
                release.wait(timeout=5)
                return _frame(2.0)

            pool.request("k", slow_decode)
            self.assertTrue(started.wait(timeout=5))
            # A second request for the same key while the first is still in flight must not
            # start a second decode -- read-ahead is supposed to de-duplicate work, not pile it up.
            pool.request("k", slow_decode)
            release.set()
            deadline = time.monotonic() + 5
            while pool.get("k") is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsNotNone(pool.get("k"))
            self.assertEqual(len(calls), 1)
        finally:
            pool.shutdown()

    def test_bounded_memory_evicts_oldest_entry_first(self):
        one_frame_bytes = _frame(0.0).nbytes
        pool = DecodeAheadPool(max_workers=2, budget_bytes=one_frame_bytes * 2)
        try:
            for key, value in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
                pool.request(key, lambda v=value: _frame(v))
                deadline = time.monotonic() + 5
                while pool.get(key) is None and time.monotonic() < deadline:
                    time.sleep(0.005)
            stats = pool.stats()
            self.assertLessEqual(stats["bytes"], pool.budget)
            # "a" was inserted first and the budget only holds two frames, so it must be the one
            # evicted -- an LRU that evicted "c" (the newest, still-wanted frame) instead would be
            # exactly backwards for read-ahead.
            self.assertIsNone(pool.get("a"))
            self.assertIsNotNone(pool.get("b"))
            self.assertIsNotNone(pool.get("c"))
        finally:
            pool.shutdown()

    def test_bump_epoch_drops_a_result_from_a_job_already_in_flight(self):
        """A seek must not let an already-superseded decode occupy a cache slot a still-wanted
        frame could use -- see decodepool.py's module docstring. The job itself cannot be
        interrupted mid-flight (matches PlaybackQueue's own documented limitation), so this
        checks the result is discarded, not that the job stops running."""
        pool = DecodeAheadPool(max_workers=1, budget_bytes=10_000_000)
        try:
            started = threading.Event()
            release = threading.Event()

            def slow_decode():
                started.set()
                release.wait(timeout=5)
                return _frame(9.0)

            pool.request("stale", slow_decode)
            self.assertTrue(started.wait(timeout=5))
            pool.bump_epoch()  # A seek happens while the decode above is already running.
            release.set()
            time.sleep(0.05)
            deadline = time.monotonic() + 5
            while "stale" in pool._pending and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsNone(pool.get("stale"))
        finally:
            pool.shutdown()

    def test_bump_epoch_before_a_pending_job_starts_skips_the_decode_entirely(self):
        pool = DecodeAheadPool(max_workers=1, budget_bytes=10_000_000)
        try:
            # Occupy the single worker so the next request stays queued, not yet started.
            gate = threading.Event()
            pool.request("occupy", lambda: (gate.wait(timeout=5), _frame(0.0))[1])
            calls = []
            pool.request("never", lambda: calls.append(1) or _frame(1.0))
            pool.bump_epoch()
            gate.set()
            deadline = time.monotonic() + 5
            while "never" in pool._pending and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(calls, [])
            self.assertIsNone(pool.get("never"))
        finally:
            pool.shutdown()

    def test_clear_empties_the_cache(self):
        pool = DecodeAheadPool(max_workers=1, budget_bytes=10_000_000)
        try:
            pool.request("k", lambda: _frame(1.0))
            deadline = time.monotonic() + 5
            while pool.get("k") is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsNotNone(pool.get("k"))
            pool.clear()
            self.assertIsNone(pool.get("k"))
        finally:
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()
