"""Cache sizing and the on-disk tier — contract clause C4 of docs/EVALUATION_TIERS.md."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nodebased import cachetier
from nodebased.cachetier import DiskCache
from nodebased.core import Dispatcher, demo_document
from nodebased.imaging import Evaluator

MIB = 1024 * 1024


def artifact(width=8, height=4, value=0.25):
    return np.full((height, width, 4), value, np.float32)


class BudgetSizingTests(unittest.TestCase):
    def test_budget_scales_with_installed_memory(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(cachetier, "physical_memory_bytes", return_value=16 * 1024 * MIB):
            self.assertEqual(cachetier.default_memory_bytes(), 4 * 1024 * MIB)

    def test_budget_is_clamped_at_both_ends(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(cachetier, "physical_memory_bytes", return_value=2 * 1024 * MIB):
                self.assertEqual(cachetier.default_memory_bytes(), cachetier.MEMORY_FLOOR)
            with mock.patch.object(cachetier, "physical_memory_bytes", return_value=512 * 1024 * MIB):
                self.assertEqual(cachetier.default_memory_bytes(), cachetier.MEMORY_CEILING)

    def test_unknown_memory_falls_back_rather_than_failing(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(cachetier, "physical_memory_bytes", return_value=None):
            self.assertEqual(cachetier.default_memory_bytes(), cachetier.MEMORY_FALLBACK)

    def test_environment_override_wins_and_survives_nonsense(self):
        with mock.patch.dict(os.environ, {"NODEBASED_CACHE_MB": "1500"}, clear=True):
            self.assertEqual(cachetier.default_memory_bytes(), 1500 * MIB)
        with mock.patch.dict(os.environ, {"NODEBASED_CACHE_MB": "banana"}, clear=True), \
                mock.patch.object(cachetier, "physical_memory_bytes", return_value=None):
            self.assertEqual(cachetier.default_memory_bytes(), cachetier.MEMORY_FALLBACK)

    def test_default_budget_holds_a_full_chain_at_4k(self):
        """The regression this work exists to prevent.

        The old fixed 256 MiB held two 4K results against a six-node chain, so every node evicted
        its predecessor and the cache returned zero hits.
        """
        self.assertEqual(cachetier.frame_bytes(3840, 2160), 3840 * 2160 * 4 * 4)
        self.assertLess(cachetier.resident_results(256 * MIB, 3840, 2160), 6)
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(cachetier, "physical_memory_bytes", return_value=16 * 1024 * MIB):
            budget = cachetier.default_memory_bytes()
        self.assertGreaterEqual(cachetier.resident_results(budget, 3840, 2160), 8)
        self.assertGreaterEqual(cachetier.resident_results(budget, 7680, 4320), 2)


class DiskTierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_round_trip_preserves_values_dtype_and_shape(self):
        store = DiskCache(root=self.root)
        pixels = artifact(6, 3, 0.5)
        self.assertTrue(store.put("a" * 64, pixels))
        loaded = store.get("a" * 64)
        np.testing.assert_array_equal(loaded, pixels)
        self.assertEqual(loaded.dtype, np.float32)
        self.assertFalse(loaded.flags.writeable)

    def test_miss_is_reported_without_raising(self):
        store = DiskCache(root=self.root)
        self.assertIsNone(store.get("b" * 64))
        self.assertEqual(store.misses, 1)

    def test_lru_eviction_respects_its_own_budget(self):
        one = artifact(8, 8)
        store = DiskCache(root=self.root, budget=int(one.nbytes * 2.5))
        for digest in ("1" * 64, "2" * 64, "3" * 64):
            store.put(digest, one)
        self.assertIsNone(store.get("1" * 64))
        self.assertIsNotNone(store.get("3" * 64))
        self.assertLessEqual(store.bytes, store.budget)
        self.assertGreaterEqual(store.evictions, 1)

    def test_result_larger_than_the_disk_budget_is_declined(self):
        store = DiskCache(root=self.root, budget=16)
        self.assertFalse(store.put("c" * 64, artifact()))

    def test_corrupt_entry_is_a_miss_not_an_error(self):
        store = DiskCache(root=self.root)
        store.put("d" * 64, artifact())
        path = store._path("d" * 64)
        path.write_bytes(b"\x93NUMPY truncated garbage")
        self.assertIsNone(store.get("d" * 64))
        self.assertFalse(path.exists())

    def test_mismatched_layout_is_rejected_never_reinterpreted(self):
        store = DiskCache(root=self.root)
        digest = "e" * 64
        path = store._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ones((4, 4, 3), np.float64))
        store._index[digest] = path.stat().st_size
        store._scanned = True
        self.assertIsNone(store.get(digest))
        self.assertEqual(store.rejections, 1)

    def test_index_rebuilds_from_disk_after_restart(self):
        first = DiskCache(root=self.root)
        first.put("f" * 64, artifact(5, 5, 0.75))
        second = DiskCache(root=self.root)
        np.testing.assert_array_equal(second.get("f" * 64), artifact(5, 5, 0.75))

    def test_disabled_store_touches_no_filesystem(self):
        store = DiskCache(root=self.root, enabled=False)
        self.assertFalse(store.put("0" * 64, artifact()))
        self.assertIsNone(store.get("0" * 64))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_shared_store_can_be_switched_off_by_environment(self):
        with mock.patch.dict(os.environ, {"NODEBASED_DISK_CACHE": "0"}, clear=True):
            self.assertFalse(DiskCache.shared().enabled)
        with mock.patch.dict(os.environ, {"NODEBASED_DISK_CACHE_MB": "64"}, clear=True):
            shared = DiskCache.shared()
        self.assertTrue(shared.enabled)
        self.assertEqual(shared.budget, 64 * MIB)

    def test_windows_disk_root_survives_a_sanitized_environment(self):
        import nodebased.cachetier as cachetier_module
        with mock.patch.object(cachetier_module.sys, "platform", "win32"), \
             mock.patch.dict(os.environ, {}, clear=True):
            root = cachetier_module.default_disk_root()
        self.assertIn("nodebased", str(root))


class EvaluatorTierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = DiskCache(root=Path(self.temp.name))

    def test_library_evaluator_does_not_enable_disk_by_default(self):
        """Importing the evaluator must never start writing to a user's cache directory."""
        self.assertFalse(Evaluator().disk.enabled)

    def test_evicted_results_spill_to_disk_and_come_back(self):
        document = Dispatcher(demo_document()).document
        evaluator = Evaluator(cache_bytes=1, disk=self.store)
        expected = evaluator.evaluate(document)
        self.assertGreater(self.store.writes, 0)

        # A fresh evaluator over the same store: every upstream node is a memory miss that the disk
        # tier answers, which is the "survives a restart" half of clause C4.
        restarted = Evaluator(cache_bytes=64 * MIB, disk=DiskCache(root=Path(self.temp.name)))
        np.testing.assert_array_equal(restarted.evaluate(document), expected)
        self.assertGreater(restarted.disk_hits, 0)

    def test_disk_hit_repopulates_memory(self):
        document = Dispatcher(demo_document()).document
        seed = Evaluator(cache_bytes=1, disk=self.store)
        seed.evaluate(document)
        evaluator = Evaluator(cache_bytes=64 * MIB, disk=DiskCache(root=Path(self.temp.name)))
        evaluator.evaluate(document)
        hits_before = evaluator.hits
        evaluator.evaluate(document)
        self.assertGreater(evaluator.hits, hits_before)

    def test_oversized_single_result_is_still_retained(self):
        """The 8K failure: a result larger than the whole budget used to be discarded outright,
        leaving the cache inert exactly where recompute is most expensive."""
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "type": "Constant", "id": "flat",
                            "params": {"width": 64, "height": 64}})
        evaluator = Evaluator(cache_bytes=1, disk=DiskCache(enabled=False))
        evaluator.evaluate(dispatcher.document, "flat")
        self.assertEqual(len(evaluator.cache), 1)
        self.assertGreater(evaluator.bytes, evaluator.budget)
        # The retained result answers the second request even though it alone exceeds the budget.
        # Under the old guard it was never stored, so this was a recompute every time.
        evaluator.evaluate(dispatcher.document, "flat")
        self.assertEqual(evaluator.hits, 1)

    def test_clear_keeps_the_disk_tier(self):
        document = Dispatcher(demo_document()).document
        evaluator = Evaluator(cache_bytes=1, disk=self.store)
        evaluator.evaluate(document)
        entries = self.store.writes
        evaluator.clear()
        self.assertEqual(evaluator.bytes, 0)
        self.assertEqual(self.store.writes, entries)


if __name__ == "__main__":
    unittest.main()
