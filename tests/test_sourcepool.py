"""Independent sources of one frame compute at the same time (nodebased/sourcepool.py, Evaluator).

The comp is the playback fixture: a plate Read, a background Read and a Roto with no input between
them. The pooled evaluator must give the serial evaluator's pixels bit for bit, the same cache keys, the
same errors on the same nodes, and leave no work running after a call returns.
"""
import copy
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401
from nodebased import imaging, sourcepool
from nodebased.cancellation import Cancelled
from nodebased.core import load_document
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.sourcepool import SourcePool, SourceTask

APP = QApplication.instance() or QApplication([])

FIXTURE = Path(__file__).parent / "fixtures" / "playback_comp.nbcomp"
WIDTH, HEIGHT, FRAMES = 96, 54, 6


def make_plates(directory):
    xx = np.linspace(0, 1, WIDTH, dtype=np.float32)[None, :]
    yy = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    for role, phase in (("plate", 0.0), ("background", 0.4)):
        for frame in range(1, FRAMES + 1):
            pixels = np.empty((HEIGHT, WIDTH, 4), np.float32)
            pixels[..., 0] = 0.1 + 0.3 * xx + 0.05 * frame
            pixels[..., 1] = 0.1 + 0.3 * yy + phase
            pixels[..., 2] = 0.2 + 0.1 * (xx * yy)
            pixels[..., 3] = 1.0
            write_exr(Path(directory) / f"{role}.{frame:04d}.exr", pixels, bits="half")


def comp(directory):
    doc = load_document(FIXTURE)
    for role in ("plate", "background"):
        doc["nodes"][role]["params"]["path"] = str(Path(directory) / f"{role}.%04d.exr")
    doc["nodes"]["roto"]["params"].update(width=WIDTH, height=HEIGHT)
    doc["time"]["first"], doc["time"]["last"] = 1, FRAMES
    return doc


class TaskTests(unittest.TestCase):
    def test_a_task_runs_once_on_the_thread_that_joins_it_when_no_worker_took_it(self):
        pool = SourcePool(max_workers=0)
        seen = []
        task = pool.submit(lambda: seen.append(threading.get_ident()) or 7)
        self.assertEqual(task.result(), 7)
        self.assertEqual(task.result(), 7)
        self.assertEqual(seen, [threading.get_ident()])
        self.assertEqual(pool.stats()["threads"], 0)

    def test_a_task_raises_its_own_error_at_the_join(self):
        pool = SourcePool(max_workers=2)
        try:
            def boom():
                raise ValueError("bad source")
            task = pool.submit(boom)
            with self.assertRaisesRegex(ValueError, "bad source"):
                task.result()
        finally:
            pool.shutdown(wait=True, timeout=2)

    def test_workers_run_tasks_at_the_same_time_and_are_capped(self):
        pool = SourcePool(max_workers=2, budget_bytes=1 << 40)
        barrier = threading.Barrier(2, timeout=5)
        threads = set()

        def work():
            threads.add(threading.get_ident())
            barrier.wait()
            return 1
        try:
            tasks = [pool.submit(work, size=1) for _ in range(2)]
            self.assertEqual([task.result() for task in tasks], [1, 1])
            self.assertEqual(len(threads), 2)
            self.assertLessEqual(pool.stats()["threads"], 2)
        finally:
            pool.shutdown(wait=True, timeout=2)

    def test_the_worker_count_is_clamped_and_follows_the_cpu_count(self):
        self.assertEqual(sourcepool.default_workers(4), 0)
        self.assertEqual(sourcepool.default_workers(2), 0)
        self.assertEqual(sourcepool.default_workers(6), 3)
        self.assertEqual(sourcepool.default_workers(5), 2)
        self.assertEqual(sourcepool.default_workers(64), sourcepool.MAX_WORKERS)
        self.assertEqual(SourcePool(max_workers=99).max_workers, sourcepool.MAX_WORKERS)
        with mock.patch.dict(os.environ, {"NODEBASED_SOURCE_WORKERS": "0"}):
            self.assertEqual(sourcepool.default_workers(32), 0)

    def test_a_task_that_does_not_fit_the_byte_budget_waits_for_the_joining_thread(self):
        pool = SourcePool(max_workers=2, budget_bytes=100)
        release = threading.Event()
        started = threading.Event()

        def first():
            started.set()
            release.wait(5)
            return "a"
        try:
            running = pool.submit(first, size=80)
            self.assertTrue(started.wait(5))
            second_thread = []
            queued = pool.submit(lambda: second_thread.append(threading.get_ident()) or "b", size=80)
            time.sleep(0.05)
            self.assertFalse(queued.claimed)               # 80 + 80 would pass the 100 byte budget
            self.assertEqual(queued.result(), "b")           # so the joiner ran it itself
            self.assertEqual(second_thread, [threading.get_ident()])
            release.set()
            self.assertEqual(running.result(), "a")
            self.assertLessEqual(pool.stats()["peak_in_flight"], 100)
        finally:
            release.set()
            pool.shutdown(wait=True, timeout=2)

    def test_nothing_is_handed_to_a_pool_that_has_shut_down(self):
        pool = SourcePool(max_workers=2)
        pool.shutdown(wait=True, timeout=2)
        self.assertFalse(pool.active)
        ran = []
        task = pool.submit(lambda: ran.append(threading.get_ident()) or 3)
        self.assertEqual(task.result(), 3)
        self.assertEqual(ran, [threading.get_ident()])
        self.assertEqual(pool.stats()["threads"], 0 if not pool._threads else len(pool._threads))

    def test_finish_drops_an_unstarted_task_and_waits_for_a_running_one(self):
        gate = threading.Event()
        running = SourceTask(lambda: gate.wait(5) or 1, 1)
        threading.Thread(target=lambda: running.claim() and running.execute(), daemon=True).start()
        time.sleep(0.02)
        unstarted = SourceTask(lambda: 1 / 0, 1)
        unstarted.finish()
        self.assertTrue(unstarted.claimed)
        waiter = threading.Thread(target=running.finish, daemon=True)
        waiter.start()
        time.sleep(0.05)
        self.assertTrue(waiter.is_alive())                  # still waiting for the running task
        gate.set()
        waiter.join(2)
        self.assertFalse(waiter.is_alive())


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        make_plates(cls._directory.name)
        cls.doc = comp(cls._directory.name)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def evaluators(self, workers=2, budget=1 << 40):
        serial = Evaluator()
        pooled = Evaluator()
        pooled.source_pool = SourcePool(max_workers=workers, budget_bytes=budget)
        self.addCleanup(pooled.source_pool.shutdown, True, 2)
        return serial, pooled

    def test_pooled_pixels_are_bit_identical_to_serial_on_every_node_and_frame(self):
        for tier in (1, 2):
            serial, pooled = self.evaluators()
            for frame in range(1, FRAMES + 1):
                for target in ("plate", "roto", "background", "merge", "colorcorrect"):
                    a, da = serial.evaluate_raster(self.doc, target, frame=frame, tier=tier, return_digest=True)
                    b, db = pooled.evaluate_raster(self.doc, target, frame=frame, tier=tier, return_digest=True)
                    self.assertEqual(da, db, (tier, frame, target))
                    self.assertTrue(np.array_equal(a.pixels, b.pixels), (tier, frame, target))
                    self.assertEqual(a.data, b.data)
                    self.assertEqual(a.display, b.display)

    def test_the_cache_keys_and_counters_match_the_serial_walk(self):
        serial, pooled = self.evaluators()
        for frame in (1, 2, 2, 3):
            serial.evaluate_raster(self.doc, "colorcorrect", frame=frame)
            pooled.evaluate_raster(self.doc, "colorcorrect", frame=frame)
        self.assertEqual(set(serial.cache), set(pooled.cache))
        self.assertEqual((serial.hits, serial.misses), (pooled.hits, pooled.misses))
        self.assertEqual(serial.bytes, pooled.bytes)
        self.assertEqual((serial.roto_matte_hits, serial.roto_matte_misses),
                         (pooled.roto_matte_hits, pooled.roto_matte_misses))
        self.assertGreater(pooled.source_pool.stats()["submitted"], 0)

    def test_the_sources_really_compute_side_by_side(self):
        _, pooled = self.evaluators(workers=2)
        barrier = threading.Barrier(2, timeout=5)
        real = imaging.read_image_raster
        threads = set()

        def meeting(*args, **kwargs):
            threads.add(threading.get_ident())
            barrier.wait()               # both Reads must be inside the decode at once to pass
            return real(*args, **kwargs)
        doc = copy.deepcopy(self.doc)
        with mock.patch.object(imaging, "read_image_raster", meeting):
            pooled.evaluate_raster(doc, "merge", frame=2)
        self.assertEqual(len(threads), 2)

    def test_a_source_that_raises_reports_the_same_error_on_the_same_node(self):
        for broken in (("plate",), ("background",), ("plate", "background")):
            doc = copy.deepcopy(self.doc)
            for role in broken:
                doc["nodes"][role]["params"]["path"] = str(Path(self._directory.name) / f"{role}.missing.%04d.exr")
            serial, pooled = self.evaluators()
            with self.assertRaises(Exception) as expected:
                serial.evaluate_raster(doc, "colorcorrect", frame=2)
            with self.assertRaises(Exception) as got:
                pooled.evaluate_raster(doc, "colorcorrect", frame=2)
            self.assertIs(type(got.exception), type(expected.exception))
            self.assertEqual(str(got.exception), str(expected.exception), broken)
            self.assertEqual(pooled.source_pool.stats()["in_flight"], 0)

    def test_an_error_earlier_in_the_walk_wins_over_a_failing_source(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"]["background"]["params"]["path"] = str(Path(self._directory.name) / "nothing.%04d.exr")
        doc["nodes"]["grade"]["inputs"]["image"] = None      # the Grade is walked before the Merge
        serial, pooled = self.evaluators()
        with self.assertRaises(Exception) as expected:
            serial.evaluate_raster(doc, "colorcorrect", frame=2)
        with self.assertRaises(Exception) as got:
            pooled.evaluate_raster(doc, "colorcorrect", frame=2)
        self.assertEqual(str(got.exception), str(expected.exception))

    def test_cancelling_stops_the_walk_and_leaves_no_source_running(self):
        _, pooled = self.evaluators()
        cancel = threading.Event()
        real = imaging.read_image_raster
        entered = threading.Event()

        def slow(*args, **kwargs):
            entered.set()
            time.sleep(0.05)
            return real(*args, **kwargs)
        with mock.patch.object(imaging, "read_image_raster", slow):
            threading.Timer(0.02, cancel.set).start()
            with self.assertRaises(Cancelled):
                for frame in range(1, FRAMES + 1):         # fresh frames, so each walk decodes
                    pooled.evaluate_raster(self.doc, "colorcorrect", cancel=cancel, frame=frame)
        self.assertTrue(entered.is_set())
        stats = pooled.source_pool.stats()
        self.assertEqual(stats["in_flight"], 0)
        self.assertEqual(stats["queued"], 0)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(Cancelled):
            pooled.evaluate_raster(self.doc, "colorcorrect", cancel=cancelled, frame=1)

    def test_no_source_work_outlives_the_call(self):
        _, pooled = self.evaluators()
        active = []
        lock = threading.Lock()
        real = imaging.read_image_raster

        def counting(*args, **kwargs):
            with lock:
                active.append(1)
            try:
                time.sleep(0.02)
                return real(*args, **kwargs)
            finally:
                with lock:
                    active.pop()
        doc = copy.deepcopy(self.doc)
        doc["nodes"]["grade"]["inputs"]["image"] = None       # fails before the Merge joins the background
        with mock.patch.object(imaging, "read_image_raster", counting):
            with self.assertRaises(Exception):
                pooled.evaluate_raster(doc, "colorcorrect", frame=3)
            self.assertEqual(active, [])

    def test_a_shut_down_pool_still_evaluates_on_the_calling_thread(self):
        serial, pooled = self.evaluators()
        pooled.source_pool.shutdown(wait=True, timeout=2)
        a = serial.evaluate_raster(self.doc, "colorcorrect", frame=4)
        b = pooled.evaluate_raster(self.doc, "colorcorrect", frame=4)
        self.assertTrue(np.array_equal(a.pixels, b.pixels))

    def test_a_byte_budget_of_one_source_keeps_the_memory_in_flight_to_one_source(self):
        _, pooled = self.evaluators(workers=3, budget=1)
        for frame in range(1, FRAMES + 1):
            pooled.evaluate_raster(self.doc, "colorcorrect", frame=frame)
        self.assertLessEqual(pooled.source_pool.stats()["peak_in_flight"], sourcepool.DEFAULT_SOURCE_BYTES * 2)

    def test_a_disabled_or_connected_node_is_not_launched(self):
        _, pooled = self.evaluators()
        doc = copy.deepcopy(self.doc)
        doc["nodes"]["background"]["disabled"] = True
        scope = imaging._SourceScope()
        nodes = doc["nodes"]
        order = ["plate", "roto", "background", "grade"]
        pooled._launch_sources(scope, doc, order, nodes, None, 1, 1, False)
        self.assertEqual(sorted(scope.tasks), ["plate", "roto"])
        scope.close()
        scope = imaging._SourceScope()
        nodes["roto"]["inputs"]["bg"] = "plate"
        pooled._launch_sources(scope, doc, order, nodes, None, 1, 1, False)
        self.assertEqual(scope.tasks, {})
        scope.close()

    def test_a_single_source_is_left_to_the_walk(self):
        _, pooled = self.evaluators()
        scope = imaging._SourceScope()
        pooled._launch_sources(scope, self.doc, ["plate", "grade"], self.doc["nodes"], None, 1, 1, False)
        self.assertEqual(scope.tasks, {})

    def test_the_walk_goes_on_with_the_chain_while_a_slow_source_computes(self):
        """The Roto has no consumer until the Merge, so the Grade, Transform and Blur run beside it."""
        _, pooled = self.evaluators(workers=2)
        events = []
        lock = threading.Lock()
        real_matte = pooled._roto_matte
        real_inner = Evaluator._windowed_kernel_inner

        def slow_matte(params, data):
            with lock:
                events.append("roto begin")
            time.sleep(0.4)
            try:
                return real_matte(params, data)
            finally:
                with lock:
                    events.append("roto end")

        def inner(kind, *args, **kwargs):
            if kind == "Blur":
                with lock:
                    events.append("blur")
            return real_inner(kind, *args, **kwargs)
        pooled._roto_matte = slow_matte
        with mock.patch.object(Evaluator, "_windowed_kernel_inner", staticmethod(inner)):
            pooled.evaluate_raster(self.doc, "colorcorrect", frame=2)
        self.assertLess(events.index("blur"), events.index("roto end"))
        self.assertEqual(events[0], "roto begin")

    def test_the_earliest_failing_source_in_the_graph_is_the_one_reported(self):
        for broken in (("plate", "background"), ("background",), ("plate",)):
            doc = copy.deepcopy(self.doc)
            for role in broken:
                doc["nodes"][role]["params"]["path"] = str(Path(self._directory.name) / f"{role}.gone.%04d.exr")
            doc["nodes"]["merge"]["inputs"]["A"] = None      # the Merge cannot pass either: connect error
            serial, pooled = self.evaluators()
            for target in ("colorcorrect", "merge", "blur"):
                outcomes = []
                for evaluator in (serial, pooled):
                    try:
                        evaluator.evaluate_raster(doc, target, frame=3)
                        outcomes.append(None)
                    except Exception as error:
                        outcomes.append((type(error), str(error)))
                self.assertEqual(outcomes[0], outcomes[1], (broken, target))

    def test_without_a_pool_nothing_is_launched(self):
        serial = Evaluator()
        scope = imaging._SourceScope()
        serial._launch_sources(scope, self.doc, ["plate", "roto", "background"], self.doc["nodes"], None, 1, 1, False)
        self.assertEqual(scope.tasks, {})

    def test_the_window_installs_a_pool_and_closes_it(self):
        from nodebased.app import Window
        window = Window()
        try:
            self.assertIsInstance(window.source_pool, SourcePool)
            self.assertIs(window.evaluator.source_pool, window.source_pool)
        finally:
            window.close()
        self.assertFalse(window.source_pool.active)


if __name__ == "__main__":
    unittest.main()
