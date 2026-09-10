import copy
import threading
import time
import unittest

import numpy as np

from nodebased.core import Dispatcher, demo_document
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.playback import MAX_PREFETCH, PlaybackQueue


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

    def test_unimplemented_proxy_quality_is_rejected_explicitly(self):
        queue = PlaybackQueue()
        with self.assertRaisesRegex(ValueError, "Proxy tiers are not implemented"):
            queue.replace(1, 1, self.document, quality="half")


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
