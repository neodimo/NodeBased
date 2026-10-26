"""Roto matte read-ahead for playback (nodebased/rotoahead.py, Evaluator.read_ahead_node, SourcePool.submit_ahead).

The comp is the playback fixture. A matte computed ahead must be the matte the walk would compute: same
pixels, same cache keys, the same error on the same node, nothing left running or stored after a cancel, and no
work handed to a pool that has shut down.
"""
import copy
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401
from nodebased import rotoahead, roto, sourcepool
from nodebased.cancellation import Cancelled
from nodebased.imaging import Evaluator
from nodebased.rotoahead import RotoAhead
from nodebased.sourcepool import SourcePool
from tests.test_sourcepool import FRAMES, comp, make_plates

APP = QApplication.instance() or QApplication([])


def small_comp(directory):
    """The playback comp with its Roto shape scaled onto the small canvas, so the mattes have pixels in them."""
    doc = comp(directory)
    factor = 0.05

    def scale(value):
        if isinstance(value, dict):
            if "value" in value:
                value["value"] *= factor
            for key in (value.get("curve") or {}).get("keys", []):
                key["value"] *= factor
            return value
        return value * factor
    for shape in doc["node_data"]["roto"]["shapes"]:
        shape["feather"] = shape.get("feather", 0) * factor
        for point in shape["points"]:
            for name in ("x", "y", "in_x", "in_y", "out_x", "out_y"):
                if name in point:
                    point[name] = scale(point[name])
    return doc


def wait_for(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        make_plates(cls._directory.name)
        cls.doc = comp(cls._directory.name)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_the_fixture_roto_animates_and_a_held_one_does_not(self):
        payload = self.doc["node_data"]["roto"]
        self.assertTrue(rotoahead.payload_animates(payload))
        held = copy.deepcopy(payload)

        def hold(value):
            if isinstance(value, dict):
                value.pop("curve", None)
                for item in value.values():
                    hold(item)
            elif isinstance(value, list):
                for item in value:
                    hold(item)
        hold(held)
        self.assertFalse(rotoahead.payload_animates(held))
        self.assertTrue(rotoahead.payload_animates({"shapes": [{"_motion_samples": [1, 2]}]}))
        self.assertTrue(rotoahead.payload_animates({"shapes": [{"track": {"node_id": "t"}}]}))
        self.assertFalse(rotoahead.payload_animates({"shapes": [{"track": None, "points": [{"x": 3, "y": 4}]}]}))

    def test_only_an_enabled_unwired_roto_is_read_ahead(self):
        flat, target, keys = rotoahead.read_ahead_sources(self.doc, "colorcorrect")
        self.assertEqual(keys, ["roto"])
        doc = copy.deepcopy(self.doc)
        doc["nodes"]["roto"]["disabled"] = True
        self.assertEqual(rotoahead.read_ahead_sources(doc, "colorcorrect")[2], [])
        doc = copy.deepcopy(self.doc)
        doc["nodes"]["roto"]["inputs"]["bg"] = "plate"
        self.assertEqual(rotoahead.read_ahead_sources(doc, "colorcorrect")[2], [])
        self.assertEqual(rotoahead.read_ahead_sources(self.doc, "blur")[2], [])    # no Roto upstream of the Blur

    def test_a_graph_that_remaps_time_is_not_read_ahead(self):
        doc = copy.deepcopy(self.doc)
        offset = copy.deepcopy(doc["nodes"]["blur"])
        offset.update(type="TimeOffset", inputs={"image": "colorcorrect"})
        doc["nodes"]["hold"] = offset
        self.assertEqual(rotoahead.read_ahead_sources(doc, "hold")[2], [])

    def test_the_environment_sets_the_lead(self):
        with mock.patch.dict(os.environ, {"NODEBASED_ROTO_AHEAD": "0"}):
            self.assertEqual(rotoahead.default_frames(), 0)
        with mock.patch.dict(os.environ, {"NODEBASED_ROTO_AHEAD": "3"}):
            self.assertEqual(rotoahead.default_frames(), 3)
        with mock.patch.dict(os.environ, {"NODEBASED_ROTO_AHEAD": "x"}):
            self.assertEqual(rotoahead.default_frames(), rotoahead.DEFAULT_FRAMES)


class PoolTests(unittest.TestCase):
    def test_four_cpus_get_workers_and_three_do_not(self):
        self.assertEqual(sourcepool.default_workers(3), 0)
        self.assertEqual(sourcepool.default_workers(2), 0)
        self.assertEqual(sourcepool.default_workers(4), sourcepool.FOUR_CPU_WORKERS)
        self.assertLessEqual(sourcepool.FOUR_CPU_WORKERS, 2)

    def test_ahead_work_waits_behind_the_frame_in_hand_and_runs_one_stream(self):
        pool = SourcePool(max_workers=3, budget_bytes=1 << 40)
        self.assertEqual(pool.ahead_limit, 1)
        release = threading.Event()
        running, order = [], []
        lock = threading.Lock()

        def slow(name):
            def run():
                with lock:
                    running.append(name)
                    order.append(name)
                release.wait(5)
                with lock:
                    running.remove(name)
            return run
        try:
            first = pool.submit_ahead(slow("a1"), size=1)
            second = pool.submit_ahead(slow("a2"), size=1)
            self.assertTrue(wait_for(lambda: "a1" in order))
            time.sleep(0.1)
            self.assertEqual(running, ["a1"])             # one read-ahead at a time
            source = pool.submit(lambda: order.append("source") or 1, size=1)
            self.assertEqual(source.result(), 1)
            release.set()
            self.assertTrue(wait_for(lambda: second._done.is_set()))
            self.assertTrue(first._done.is_set())
        finally:
            release.set()
            pool.shutdown(wait=True, timeout=2)

    def test_ahead_work_that_does_not_fit_the_budget_is_dropped(self):
        pool = SourcePool(max_workers=2, budget_bytes=100)
        release = threading.Event()
        started = threading.Event()
        try:
            held = pool.submit(lambda: started.set() or release.wait(5), size=80)
            self.assertTrue(started.wait(5))
            ran = []
            ahead = pool.submit_ahead(lambda: ran.append(1), size=80)
            self.assertTrue(wait_for(lambda: pool.stats()["ahead_dropped"] == 1))
            self.assertEqual(ran, [])
            self.assertTrue(ahead.claimed)
        finally:
            release.set()
            pool.shutdown(wait=True, timeout=2)

    def test_a_pool_without_workers_or_after_shutdown_takes_no_ahead_work(self):
        self.assertIsNone(SourcePool(max_workers=0).submit_ahead(lambda: 1))
        pool = SourcePool(max_workers=2)
        pool.shutdown(wait=True, timeout=2)
        self.assertIsNone(pool.submit_ahead(lambda: 1))

    def test_shutdown_drops_queued_ahead_work_and_never_runs_it(self):
        pool = SourcePool(max_workers=1, budget_bytes=1 << 40)
        release = threading.Event()
        started = threading.Event()
        ran = []
        held = pool.submit(lambda: started.set() or release.wait(5), size=1)
        self.assertTrue(started.wait(5))
        queued = pool.submit_ahead(lambda: ran.append(1), size=1)
        pool.shutdown()
        release.set()
        self.assertTrue(queued.claimed)
        time.sleep(0.05)
        self.assertEqual(ran, [])
        pool.shutdown(wait=True, timeout=2)


class ReadAheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        make_plates(cls._directory.name)
        cls.doc = small_comp(cls._directory.name)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def setUp(self):
        self.serial = Evaluator()
        self.ahead = Evaluator()
        self.pool = SourcePool(max_workers=2, budget_bytes=1 << 40)
        self.ahead.source_pool = self.pool
        self.scheduler = RotoAhead(self.ahead, self.pool, frames=2, byte_cap=1 << 40)
        self.addCleanup(self.pool.shutdown, True, 2)
        self.addCleanup(self.scheduler.cancel)

    def settle(self):
        self.assertTrue(wait_for(lambda: all(task._done.is_set() for task in list(self.scheduler._tasks.values()))))

    def test_a_matte_computed_ahead_is_the_matte_the_walk_computes(self):
        for tier in (1, 2):
            serial, ahead = Evaluator(), Evaluator()
            ahead.source_pool = self.pool
            scheduler = RotoAhead(ahead, self.pool, frames=3, byte_cap=1 << 40)
            self.assertEqual(scheduler.schedule(self.doc, "colorcorrect", [2, 3, 4], tier), 3)
            self.assertTrue(wait_for(lambda: all(t._done.is_set() for t in list(scheduler._tasks.values()))))
            for frame in (2, 3, 4):
                a, da = serial.evaluate_raster(self.doc, "roto", frame=frame, tier=tier, return_digest=True)
                hits = ahead.hits
                b, db = ahead.evaluate_raster(self.doc, "roto", frame=frame, tier=tier, return_digest=True)
                self.assertEqual(da, db, (tier, frame))
                self.assertGreater(ahead.hits, hits, "the walk should have found the matte ready")
                self.assertTrue(np.array_equal(a.pixels, b.pixels), (tier, frame))
                self.assertEqual(a.data, b.data)
                c = ahead.evaluate_raster(self.doc, "colorcorrect", frame=frame, tier=tier)
                d = serial.evaluate_raster(self.doc, "colorcorrect", frame=frame, tier=tier)
                self.assertTrue(np.array_equal(c.pixels, d.pixels), (tier, frame))
            self.assertTrue(set(serial.cache) <= set(ahead.cache))
            scheduler.cancel()

    def test_the_cache_holds_no_entry_the_walk_would_not_have_made(self):
        self.scheduler.schedule(self.doc, "colorcorrect", [2, 3], 1)
        self.settle()
        self.assertEqual(len(self.ahead.cache), 2)
        for frame in (2, 3):
            self.serial.evaluate_raster(self.doc, "roto", frame=frame)
        self.assertEqual(set(self.serial.cache), set(self.ahead.cache))

    def test_a_roto_that_raises_reports_the_same_error_on_the_same_node(self):
        def broken(shapes, width, height, invert, shape_cache=None):
            raise ValueError("Roto: this shape cannot be drawn")
        with mock.patch.object(roto, "rasterise", broken):
            self.scheduler.schedule(self.doc, "colorcorrect", [2, 3], 1)
            self.settle()
            self.assertEqual(len(self.ahead.cache), 0)
            with self.assertRaises(ValueError) as expected:
                self.serial.evaluate_raster(self.doc, "colorcorrect", frame=2)
            with self.assertRaises(ValueError) as got:
                self.ahead.evaluate_raster(self.doc, "colorcorrect", frame=2)
        self.assertEqual(str(got.exception), str(expected.exception))

    def test_a_walk_that_reaches_a_running_matte_waits_for_it_instead_of_drawing_it_again(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        real = roto.rasterise

        def slow(*args, **kwargs):
            calls.append(threading.get_ident())
            entered.set()
            release.wait(5)
            return real(*args, **kwargs)
        with mock.patch.object(roto, "rasterise", slow):
            self.scheduler.schedule(self.doc, "colorcorrect", [2], 1)
            self.assertTrue(entered.wait(5))
            result = []
            walker = threading.Thread(
                target=lambda: result.append(self.ahead.evaluate_raster(self.doc, "roto", frame=2)), daemon=True)
            walker.start()
            time.sleep(0.15)
            self.assertTrue(walker.is_alive())            # waiting on the worker's matte
            release.set()
            walker.join(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.ahead.ahead_joins, 1)
        expected = self.serial.evaluate_raster(self.doc, "roto", frame=2)
        self.assertTrue(np.array_equal(result[0].pixels, expected.pixels))

    def test_a_walk_waiting_for_a_matte_can_be_cancelled(self):
        entered, release = threading.Event(), threading.Event()
        real = roto.rasterise

        def slow(*args, **kwargs):
            entered.set()
            release.wait(5)
            return real(*args, **kwargs)
        cancel = threading.Event()
        with mock.patch.object(roto, "rasterise", slow):
            self.scheduler.schedule(self.doc, "colorcorrect", [2], 1)
            self.assertTrue(entered.wait(5))
            threading.Timer(0.05, cancel.set).start()
            with self.assertRaises(Cancelled):
                self.ahead.evaluate_raster(self.doc, "roto", frame=2, cancel=cancel)
            release.set()
        self.settle()

    def test_cancelling_drops_queued_mattes_and_nothing_is_stored_by_a_running_one(self):
        entered, release = threading.Event(), threading.Event()
        real = roto.rasterise
        calls = []

        def slow(*args, **kwargs):
            calls.append(1)
            entered.set()
            release.wait(5)
            return real(*args, **kwargs)
        with mock.patch.object(roto, "rasterise", slow):
            self.assertEqual(self.scheduler.schedule(self.doc, "colorcorrect", [2, 3], 1), 2)
            self.assertTrue(entered.wait(5))
            self.scheduler.cancel()                       # frame 2 is running, frame 3 is queued behind it
            release.set()
            self.assertTrue(wait_for(lambda: self.pool.stats()["in_flight"] == 0))
        time.sleep(0.05)
        self.assertEqual(len(calls), 1)                   # frame 3 never ran
        self.assertEqual(len(self.ahead.cache), 0)        # frame 2 stopped before it stored
        self.assertEqual(self.ahead._ahead_inflight, {})
        self.assertEqual(self.scheduler.stats()["pending"], 0)

    def test_a_new_epoch_after_a_cancel_schedules_again(self):
        self.scheduler.schedule(self.doc, "colorcorrect", [2], 1)
        self.settle()
        self.scheduler.cancel()
        self.ahead.clear()
        self.assertEqual(self.scheduler.schedule(self.doc, "colorcorrect", [2], 1), 1)
        self.settle()
        self.assertEqual(len(self.ahead.cache), 1)

    def test_an_edit_makes_a_new_matte_and_the_old_one_is_never_served(self):
        self.scheduler.schedule(self.doc, "colorcorrect", [2], 1)
        self.settle()
        edited = copy.deepcopy(self.doc)
        edited["nodes"]["roto"]["params"]["invert"] = 1
        self.scheduler.cancel()
        a = self.ahead.evaluate_raster(edited, "roto", frame=2)
        b = self.serial.evaluate_raster(edited, "roto", frame=2)
        self.assertTrue(np.array_equal(a.pixels, b.pixels))
        old = self.serial.evaluate_raster(self.doc, "roto", frame=2)
        self.assertFalse(np.array_equal(a.pixels, old.pixels))

    def test_a_frame_already_queued_or_running_is_not_queued_twice(self):
        release = threading.Event()
        real = roto.rasterise

        def slow(*args, **kwargs):
            release.wait(5)
            return real(*args, **kwargs)
        with mock.patch.object(roto, "rasterise", slow):
            self.assertEqual(self.scheduler.schedule(self.doc, "colorcorrect", [2, 3], 1), 2)
            self.assertEqual(self.scheduler.schedule(self.doc, "colorcorrect", [2, 3], 1), 0)
            self.assertEqual(self.scheduler.schedule(self.doc, "colorcorrect", [3, 4], 1), 1)
            release.set()
        self.settle()

    def test_a_held_roto_is_read_ahead_once_per_document_state(self):
        doc = copy.deepcopy(self.doc)

        def hold(value):
            if isinstance(value, dict):
                value.pop("curve", None)
                for item in value.values():
                    hold(item)
            elif isinstance(value, list):
                for item in value:
                    hold(item)
        hold(doc["node_data"]["roto"])
        self.assertEqual(self.scheduler.schedule(doc, "colorcorrect", [2, 3], 1), 1)
        self.settle()
        self.assertEqual(self.scheduler.schedule(doc, "colorcorrect", [3, 4], 1), 0)
        self.assertEqual(len(self.ahead.cache), 1)       # one matte for every frame
        a = self.ahead.evaluate_raster(doc, "roto", frame=5)
        self.assertEqual(self.ahead.hits, 1)
        b = self.serial.evaluate_raster(doc, "roto", frame=5)
        self.assertTrue(np.array_equal(a.pixels, b.pixels))

    def test_the_lead_follows_the_frame_count_and_the_byte_cap(self):
        self.ahead._source_sizes["roto"] = 100
        small = RotoAhead(self.ahead, self.pool, frames=3, byte_cap=150)
        self.assertEqual(small.schedule(self.doc, "colorcorrect", [2, 3, 4], 1), 1)     # 150 bytes hold one matte
        small.cancel()
        wide = RotoAhead(self.ahead, self.pool, frames=3, byte_cap=1000)
        self.assertEqual(wide.schedule(self.doc, "colorcorrect", [2, 3, 4, 5], 1), 3)
        wide.cancel()
        off = RotoAhead(self.ahead, self.pool, frames=0)
        self.assertFalse(off.active)
        self.assertEqual(off.schedule(self.doc, "colorcorrect", [2], 1), 0)

    def test_nothing_is_queued_without_workers_or_after_shutdown(self):
        none = RotoAhead(self.ahead, SourcePool(max_workers=0), frames=2)
        self.assertEqual(none.schedule(self.doc, "colorcorrect", [2], 1), 0)
        pool = SourcePool(max_workers=2)
        scheduler = RotoAhead(self.ahead, pool, frames=2)
        pool.shutdown(wait=True, timeout=2)
        self.assertEqual(scheduler.schedule(self.doc, "colorcorrect", [2], 1), 0)
        scheduler.cancel()                                  # idempotent after the pool is gone

    def test_closing_with_work_queued_hands_nothing_to_a_closed_pool(self):
        release = threading.Event()
        real = roto.rasterise

        def slow(*args, **kwargs):
            release.wait(5)
            return real(*args, **kwargs)
        errors = []
        with mock.patch.object(roto, "rasterise", slow):
            self.scheduler.schedule(self.doc, "colorcorrect", [2, 3], 1)
            self.scheduler.cancel()
            self.pool.shutdown()
            release.set()
            try:
                self.assertEqual(self.scheduler.schedule(self.doc, "colorcorrect", [4], 1), 0)
                a = self.ahead.evaluate_raster(self.doc, "colorcorrect", frame=4)   # a walk still works
            except RuntimeError as error:
                errors.append(str(error))
        self.assertEqual(errors, [])
        self.assertTrue(np.array_equal(a.pixels, self.serial.evaluate_raster(self.doc, "colorcorrect", frame=4).pixels))
        self.assertTrue(wait_for(lambda: self.pool.stats()["in_flight"] == 0))

    def test_a_playing_walk_and_the_read_ahead_do_not_deadlock_with_the_source_pool(self):
        """Every frame's walk launches its own sources on the same pool while the next frame's matte runs."""
        for frame in range(1, FRAMES):
            self.scheduler.schedule(self.doc, "colorcorrect", [frame + 1, frame + 2], 1)
            a = self.ahead.evaluate_raster(self.doc, "colorcorrect", frame=frame)
            b = self.serial.evaluate_raster(self.doc, "colorcorrect", frame=frame)
            self.assertTrue(np.array_equal(a.pixels, b.pixels), frame)
        self.settle()
        self.assertEqual(self.ahead._ahead_inflight, {})


class WindowTests(unittest.TestCase):
    def test_the_window_installs_the_read_ahead_and_closing_cancels_it(self):
        from nodebased.app import Window
        window = Window()
        try:
            self.assertIsInstance(window.roto_ahead, RotoAhead)
            self.assertIs(window.roto_ahead.pool, window.source_pool)
            self.assertIs(window.roto_ahead.evaluator, window.evaluator)
        finally:
            window.close()
        self.assertFalse(window.source_pool.active)
        self.assertFalse(window.roto_ahead.active)


if __name__ == "__main__":
    unittest.main()
