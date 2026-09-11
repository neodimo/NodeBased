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

    def _key(self, document=None, target="viewer", frame=1, tier=1, view="ACES 2.0",
             exposure=0.0, channel="RGB", background="black"):
        return DisplayCache.key(document if document is not None else self.document,
                                target, frame, tier, view, exposure, channel, background)

    def test_identical_request_is_a_hit(self):
        cache = DisplayCache()
        key = self._key()
        cache.put(key, b"\x01\x02\x03", 1, 1, 3)
        self.assertEqual(cache.get(self._key()), (b"\x01\x02\x03", 1, 1, 3))

    def test_a_changed_document_misses_and_undoing_the_change_hits_again(self):
        """The whole point of keying on the document rather than just the frame number: a paused
        edit is a correct miss, and returning a parameter to a value already seen -- undo, or
        dragging a slider back -- is a correct hit again, exactly like a scrub back in playback."""
        cache = DisplayCache()
        original = self._key()
        cache.put(original, b"before", 1, 1, 3)
        edited_document = copy.deepcopy(self.document)
        edited_document["nodes"]["grade"]["params"]["exposure"] = 5.0
        self.assertIsNone(cache.get(self._key(document=edited_document)))
        # Undo restores the original document exactly -- same key, same cached bytes.
        self.assertEqual(cache.get(self._key()), (b"before", 1, 1, 3))

    def test_different_display_settings_on_the_same_frame_are_different_entries(self):
        cache = DisplayCache()
        cache.put(self._key(exposure=0.0), b"a", 1, 1, 3)
        cache.put(self._key(exposure=2.0), b"b", 1, 1, 3)
        self.assertEqual(cache.get(self._key(exposure=0.0)), (b"a", 1, 1, 3))
        self.assertEqual(cache.get(self._key(exposure=2.0)), (b"b", 1, 1, 3))
        self.assertEqual(len(cache), 2)

    def test_budget_evicts_oldest_entry_first(self):
        cache = DisplayCache(budget_bytes=10)
        cache.put(self._key(frame=1), b"12345", 1, 1, 5)
        cache.put(self._key(frame=2), b"67890", 1, 1, 5)
        self.assertEqual(len(cache), 2)
        # A third entry pushes the total past budget; the least-recently-used (frame 1,
        # untouched since insertion) is evicted, not the most recent.
        cache.put(self._key(frame=3), b"abcde", 1, 1, 5)
        self.assertIsNone(cache.get(self._key(frame=1)))
        self.assertIsNotNone(cache.get(self._key(frame=2)))
        self.assertIsNotNone(cache.get(self._key(frame=3)))

    def test_an_entry_larger_than_the_whole_budget_is_never_stored(self):
        cache = DisplayCache(budget_bytes=4)
        cache.put(self._key(), b"12345", 1, 1, 5)
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.bytes, 0)


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
