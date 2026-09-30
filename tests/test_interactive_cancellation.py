"""M1 gate: interactive cancellation (docs/VISION.md).

On a 4K graph of at least eight nodes, a cancel issued mid-evaluation must stop work within
100 ms on both the full-frame path (`Evaluator.evaluate_raster`) and the tile path
(`TileExecutor.compose`), leave no partial result in the cache, and let the next, uncancelled
evaluation produce the correct pixels.

Both tests synchronise with the in-flight worker thread by *waiting on real progress*
(`Evaluator.misses` / `TileExecutor.cache.misses`), not a fixed sleep -- a sleep long enough to
be reliable on a loaded CI runner would also be long enough to occasionally let a fast graph
finish before the cancel ever lands, which would make the test pass for the wrong reason.
`Evaluator.evaluate_raster` only checks its cancel event once per node (imaging.py's per-node
walk), so a single node whose own kernel exceeds 100 ms cannot be interrupted inside itself --
this test's full-frame graph deliberately uses passthrough/format kinds (halo (0, 0), no
transcendental math) that measure a few milliseconds each at 4K, which is the graph shape this
100 ms contract actually promises. `TileExecutor.compose` checks its cancel event once per
256 px tile (`nodebased/tiles.py`'s `DEFAULT_TILE_EDGE`), so it can carry a heavier per-pixel
kernel (`Blur`) and still stop within a couple of tiles' worth of work.
"""
import threading
import time
import unittest

import numpy as np

from nodebased.cancellation import Cancelled
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.tileexec import CancelledTile, TileExecutor

WIDTH, HEIGHT = 3840, 2160
STOP_BUDGET_MS = 100


def _wait_until(predicate, timeout=5.0, interval=0.0002):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class FullFramePathCancellationTests(unittest.TestCase):
    def _graph(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Constant", "id": "src",
                   "params": {"width": WIDTH, "height": HEIGHT, "red": 0.4, "green": 0.3,
                              "blue": 0.2, "alpha": 1.0}})
        prev = "src"
        # Passthrough/format kinds: cheap enough per node (a few ms at 4K) that the once-per-node
        # cancel check (imaging.py line ~772) can land well inside the 100 ms budget.
        kinds = ["Dot", "Crop", "Mirror", "ChannelShuffle", "Dot", "Crop", "Mirror", "ChannelShuffle"]
        for index, kind in enumerate(kinds):
            node_id = f"n{index}"
            slot = "A" if kind == "ChannelShuffle" else ("input" if kind == "Dot" else "image")
            d.execute({"op": "create", "type": kind, "id": node_id, "params": {}})
            d.execute({"op": "connect", "id": node_id, "input": slot, "source": prev})
            prev = node_id
        self.assertGreaterEqual(len(kinds), 8)
        return d, prev

    def test_cancel_stops_within_budget_leaves_no_partial_result_and_next_eval_is_correct(self):
        d, target = self._graph()

        reference = Evaluator().evaluate_raster(d.document, target, tier=1, typed=True)
        target_digest = Evaluator().evaluate_raster(
            d.document, target, tier=1, typed=True, return_digest=True)[1]

        ev = Evaluator()
        cancel = threading.Event()
        outcome = {}

        def worker():
            try:
                ev.evaluate_raster(d.document, target, cancel=cancel, tier=1, typed=True)
                outcome["cancelled"] = False
            except Cancelled:
                outcome["cancelled"] = True

        thread = threading.Thread(target=worker)
        thread.start()
        # At least one node must finish (proving the graph is genuinely mid-flight) before more
        # than half of the eight nodes could possibly be done, so the cancel still pre-empts
        # real remaining work.
        self.assertTrue(_wait_until(lambda: ev.misses >= 1), "worker never started")
        t_cancel = time.perf_counter()
        cancel.set()
        thread.join(timeout=2.0)
        t_stop = time.perf_counter()

        self.assertFalse(thread.is_alive(), "evaluation did not stop")
        self.assertTrue(outcome.get("cancelled"), "evaluation completed instead of cancelling")
        stop_ms = (t_stop - t_cancel) * 1000.0
        self.assertLess(stop_ms, STOP_BUDGET_MS,
                        f"full-frame cancel took {stop_ms:.1f} ms, over the {STOP_BUDGET_MS} ms budget")
        self.assertNotIn(target_digest, ev.cache,
                         "a cancelled evaluation left a result cached under the target's digest")

        # The next, uncancelled look must still be correct.
        again = ev.evaluate_raster(d.document, target, tier=1, typed=True)
        np.testing.assert_array_equal(again.pixels, reference.pixels)


class TilePathCancellationTests(unittest.TestCase):
    def _graph(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Constant", "id": "src",
                   "params": {"width": WIDTH, "height": HEIGHT, "red": 0.4, "green": 0.3,
                              "blue": 0.2, "alpha": 1.0}})
        prev = "src"
        # Tile-native, non-trivial-cost kinds (Grade/Blur), so this path is proven under real
        # per-pixel work rather than a cheap passthrough -- what makes it fast is the 256 px
        # tile granularity, not a cheap kernel.
        kinds = ["Grade", "Blur", "Grade", "Blur", "Grade", "Blur", "Grade", "Blur"]
        for index, kind in enumerate(kinds):
            node_id = f"n{index}"
            params = {"radius": 6.0} if kind == "Blur" else {}
            d.execute({"op": "create", "type": kind, "id": node_id, "params": params})
            d.execute({"op": "connect", "id": node_id, "input": "image", "source": prev})
            prev = node_id
        self.assertGreaterEqual(len(kinds), 8)
        return d, prev

    def test_cancel_stops_within_budget_leaves_no_partial_result_and_next_eval_is_correct(self):
        d, target = self._graph()

        reference = TileExecutor().compose(d.document, target, frame=1, tier=1)

        exe = TileExecutor()
        cancel = threading.Event()
        outcome = {}

        def worker():
            try:
                exe.compose(d.document, target, frame=1, tier=1, cancel=cancel)
                outcome["cancelled"] = False
            except CancelledTile:
                outcome["cancelled"] = True

        thread = threading.Thread(target=worker)
        thread.start()
        # A handful of tiles in (of ~135 at 256 px over 3840x2160), well short of the full
        # render, so the cancel still pre-empts most of the work.
        self.assertTrue(_wait_until(lambda: exe.cache.misses >= 3), "worker never started")
        t_cancel = time.perf_counter()
        cancel.set()
        thread.join(timeout=2.0)
        t_stop = time.perf_counter()

        self.assertFalse(thread.is_alive(), "tile render did not stop")
        self.assertTrue(outcome.get("cancelled"), "tile render completed instead of cancelling")
        stop_ms = (t_stop - t_cancel) * 1000.0
        self.assertLess(stop_ms, STOP_BUDGET_MS,
                        f"tile cancel took {stop_ms:.1f} ms, over the {STOP_BUDGET_MS} ms budget")
        self.assertLess(exe.cache.misses, 135,
                        "a cancelled tile render should not have rendered every tile")

        # The next, uncancelled look must still be correct -- including any tiles the cancelled
        # pass had already primed the cache with.
        again = exe.compose(d.document, target, frame=1, tier=1)
        np.testing.assert_array_equal(again.pixels, reference.pixels)


if __name__ == "__main__":
    unittest.main()
