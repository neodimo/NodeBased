import copy
import threading
import time
import unittest

import numpy as np

from nodebased.core import Dispatcher, demo_document
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.playback import MAX_PREFETCH, DisplayCache, PlaybackQueue


class PlaybackQueueTests(unittest.TestCase):
    def setUp(self):
        self.document = demo_document()

    def test_queue_is_one_display_plus_three_unique_prefetches(self):
        queue = PlaybackQueue()
        queue.replace(7, 10, self.document, [11, 12, 13, 14, 10])
        self.assertEqual(len(queue), 1 + MAX_PREFETCH)
        requests = [queue.take()[0] for _ in range(1 + MAX_PREFETCH)]
        self.assertEqual([(r.frame, r.display) for r in requests],
                         [(10, True), (11, False), (12, False), (13, False)])

    def test_replacement_cancels_active_before_installing_new_work(self):
        queue = PlaybackQueue()
        queue.replace(1, 1, self.document)
        _, active = queue.take()
        self.assertFalse(active.is_set())
        queue.replace(2, 9, self.document, [10])
        self.assertTrue(active.is_set())
        request, _ = queue.take()
        self.assertEqual((request.generation, request.frame), (2, 9))

    def test_cancel_drops_pending_and_marks_active(self):
        queue = PlaybackQueue()
        queue.replace(1, 1, self.document, [2, 3])
        _, active = queue.take()
        queue.cancel()
        self.assertTrue(active.is_set())
        self.assertEqual(len(queue), 0)

    def test_enqueue_path_stays_inside_ui_budget_while_work_is_active(self):
        queue = PlaybackQueue()
        queue.replace(1, 1, self.document)
        queue.take()  # Represents an evaluator blocked in the worker.
        started = time.perf_counter()
        queue.replace(2, 2, copy.deepcopy(self.document), [3, 4, 5])
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.016)

    def test_requests_carry_their_proxy_tier(self):
        """The tier travels with the request, so a result that lands after the artist changed tier
        can be recognised as stale instead of drawn at the wrong size."""
        queue = PlaybackQueue()
        queue.replace(1, 1, self.document, future_frames=(2, 3), tier=4)
        request, _ = queue.take()
        self.assertEqual(request.tier, 4)
        self.assertTrue(all(item.tier == 4 for item in queue._items))

    def test_undeclared_proxy_tier_is_rejected_explicitly(self):
        queue = PlaybackQueue()
        with self.assertRaisesRegex(ValueError, "Unsupported proxy tier"):
            queue.replace(1, 1, self.document, tier=3)


class DisplayCacheTests(unittest.TestCase):
    def setUp(self):
        self.document = demo_document()

    def _key(self, document=None, target="viewer", frame=1, tier=1):
        return DisplayCache.key(document or self.document, target, frame, tier)

    @staticmethod
    def _frame(value, height=1, width=1):
        return np.full((height, width, 4), value, dtype=np.float32)

    def test_identical_request_is_a_hit(self):
        cache = DisplayCache()
        key = self._key()
        frame = self._frame(0.25)
        cache.put(key, frame)
        np.testing.assert_array_equal(cache.get(self._key()), frame)

    def test_a_changed_document_misses_and_undoing_the_change_hits_again(self):
        """The whole point of keying on the document rather than just the frame number: a paused
        edit is a correct miss, and returning a parameter to a value already seen -- undo, or
        dragging a slider back -- is a correct hit again, exactly like a scrub back in playback."""
        cache = DisplayCache()
        original = self._key()
        before = self._frame(0.5)
        cache.put(original, before)
        edited_document = copy.deepcopy(self.document)
        edited_document["nodes"]["grade"]["params"]["exposure"] = 5.0
        self.assertIsNone(cache.get(self._key(document=edited_document)))
        # Undo restores the original document exactly -- same key, same cached bytes.
        np.testing.assert_array_equal(cache.get(self._key()), before)

    def test_key_no_longer_depends_on_display_parameters(self):
        cache = DisplayCache()
        first, second = self._frame(0.0), self._frame(1.0)
        cache.put(self._key(), first)
        cache.put(self._key(), second)
        np.testing.assert_array_equal(cache.get(self._key()), second)
        self.assertEqual(len(cache), 1)

    def test_budget_evicts_oldest_entry_first(self):
        cache = DisplayCache(budget_bytes=16)  # each 1x1x4 half frame is 8 bytes
        cache.put(self._key(frame=1), self._frame(0.0))
        cache.put(self._key(frame=2), self._frame(0.25))
        self.assertEqual(len(cache), 2)
        # A third entry pushes the total past budget; the least-recently-used (frame 1,
        # untouched since insertion) is evicted, not the most recent.
        cache.put(self._key(frame=3), self._frame(0.5))
        self.assertIsNone(cache.get(self._key(frame=1)))
        self.assertIsNotNone(cache.get(self._key(frame=2)))
        self.assertIsNotNone(cache.get(self._key(frame=3)))

    def test_an_entry_larger_than_the_whole_budget_is_never_stored(self):
        cache = DisplayCache(budget_bytes=4)
        cache.put(self._key(), self._frame(0.0))  # 8 stored half-float bytes > the 4-byte budget
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.bytes, 0)

    def test_a_crop_cached_for_one_region_is_a_miss_for_another(self):
        """A full-resolution preview is only the visible crop. Returning a crop cached before a
        pan would paint the old pixels at the new position, so the region is part of the match."""
        cache = DisplayCache()
        left = self._frame(0.25)
        cache.put(self._key(), left, region=(0, 0, 100, 50))
        self.assertIsNone(cache.get(self._key(), region=(40, 0, 100, 50)))
        self.assertIsNone(cache.get(self._key()))
        np.testing.assert_array_equal(cache.get(self._key(), region=(0, 0, 100, 50)), left)

    def test_one_frame_holds_a_whole_frame_and_a_crop_side_by_side(self):
        cache = DisplayCache()
        whole, crop = self._frame(0.0), self._frame(1.0)
        cache.put(self._key(), whole, region=(0, 0, 200, 100))
        cache.put(self._key(), crop, region=(50, 20, 40, 30))
        np.testing.assert_array_equal(cache.get(self._key(), region=(0, 0, 200, 100)), whole)
        np.testing.assert_array_equal(cache.get(self._key(), region=(50, 20, 40, 30)), crop)

    def test_resident_frames_track_eviction_across_regions(self):
        cache = DisplayCache(budget_bytes=16)  # two 8-byte half frames fit; a third forces eviction
        identity = self._key()[0]
        cache.put(self._key(frame=1), self._frame(0.0), region=(0, 0, 1, 1))
        cache.put(self._key(frame=1), self._frame(0.25), region=(0, 0, 2, 2))
        self.assertEqual(cache.resident_frames(identity, [1, 2]), {1})
        cache.put(self._key(frame=2), self._frame(0.5))
        # One of frame 1's two images was evicted; the other still makes it resident.
        self.assertEqual(cache.resident_frames(identity, [1, 2]), {1, 2})
        cache.put(self._key(frame=2), self._frame(0.75), region=(0, 0, 3, 3))
        self.assertEqual(cache.resident_frames(identity, [1, 2]), {2})
        cache.clear()
        self.assertEqual(cache.resident_frames(identity, [1, 2]), set())

    def test_a_frame_warmed_by_read_ahead_is_a_hit_once_playback_actually_reaches_it(self):
        """Read-ahead builds every prefetch FrameRequest from ONE document snapshot taken while
        the playhead is still on the current frame -- so a request warming frame 5 carries a
        document whose own time.current still says 1. Playback later reaching frame 5 takes a
        FRESH snapshot where time.current correctly says 5. Both must hash to the same key, or
        every read-ahead entry is warmed under a key real playback can never reproduce -- which
        is exactly what made display caching invisible during real playback despite passing in
        isolation (each test there re-requests the *same* already-current frame, never a
        prefetched one)."""
        cache = DisplayCache()
        stale_snapshot = copy.deepcopy(self.document)  # time.current == 1, as if read-ahead
        # built this while frame 1 was still playing and prefetched frame 5.
        warmed = self._frame(0.5)
        cache.put(self._key(document=stale_snapshot, frame=5), warmed)
        fresh_snapshot = copy.deepcopy(self.document)
        fresh_snapshot["time"]["current"] = 5  # playback has now actually reached frame 5.
        np.testing.assert_array_equal(cache.get(self._key(document=fresh_snapshot, frame=5)), warmed)

    def test_raw_data_is_not_rounded_but_colour_is_clamped_to_half_range(self):
        value = 70000.0  # above HALF_MAX (65504.0), so half storage must clamp it
        data_cache = DisplayCache()
        data_key = self._key(frame=2)
        data_cache.put(data_key, self._frame(value), is_data=True)
        self.assertEqual(float(data_cache.get(data_key)[0, 0, 0]), value)
        colour_key = self._key(frame=3)
        data_cache.put(colour_key, self._frame(value))
        self.assertEqual(float(data_cache.get(colour_key)[0, 0, 0]), 65504.0)

    def test_force_float32_preserves_colour_precision(self):
        cache = DisplayCache(force_float32=True)
        key = self._key()
        cache.put(key, self._frame(70000.0))
        self.assertEqual(float(cache.get(key)[0, 0, 0]), 70000.0)


class PlaybackTimeTests(unittest.TestCase):
    def test_transient_transport_frames_do_not_consume_undo_slots(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "time", "first": 1, "last": 10, "current": 2})
        slots = len(dispatcher.undo_stack)
        for frame in (3, 4, 5, 6):
            dispatcher.execute({"op": "time", "current": frame, "transient": True})
        self.assertEqual(len(dispatcher.undo_stack), slots)
        self.assertEqual(dispatcher.document["time"]["current"], 6)
        dispatcher.execute({"op": "undo"})
        self.assertEqual(dispatcher.document["time"]["current"], 1)

    def test_read_ahead_warms_the_real_sequence_cache(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            pattern = Path(folder) / "plate.####.exr"
            for frame in (1, 2, 3, 4):
                write_exr(Path(folder) / f"plate.{frame:04d}.exr",
                          np.full((2, 2, 4), frame / 10, np.float32))
            dispatcher = Dispatcher()
            dispatcher.execute({"op": "create", "id": "read", "type": "Read",
                                "params": {"path": str(pattern)}})
            dispatcher.execute({"op": "view", "id": "read"})
            queue = PlaybackQueue()
            queue.replace(1, 1, dispatcher.document, [2, 3, 4])
            evaluator = Evaluator()
            while len(queue):
                request, cancel = queue.take()
                evaluator.evaluate(request.document, cancel=cancel, frame=request.frame)
                queue.finish(cancel)
            hits = evaluator.hits
            evaluator.evaluate(dispatcher.document, frame=2)
            self.assertGreater(evaluator.hits, hits)


if __name__ == "__main__":
    unittest.main()
