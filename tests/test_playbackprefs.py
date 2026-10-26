"""The playback settings (nodebased/playbackprefs.py): GPU nodes and the decode and source worker counts.

Defaults are today's behaviour and flip with one line; both settings round-trip through the saved preferences; a
changed setting takes effect in a running window without a restart; the pixels do not depend on either setting; and a
card that refuses the island, or loses its device, leaves the CPU nodes in use with a message.
"""
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401
from nodebased import playbackprefs, sourcepool
from nodebased.decodepool import DecodeAheadPool
from nodebased.imaging import Evaluator
from nodebased.sourcepool import SourcePool
from tests.test_rotoahead import small_comp
from tests.test_sourcepool import FRAMES, make_plates

APP = QApplication.instance() or QApplication([])

ENVIRONMENT = (playbackprefs.GPU_NODES_ENV, playbackprefs.DECODE_WORKERS_ENV, playbackprefs.SOURCE_WORKERS_ENV)


def clean_environment():
    """The environment the settings read, emptied for one test."""
    return mock.patch.dict(os.environ, {}, clear=False)


def wait_for(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    APP.processEvents()
    return condition()


def without_environment():
    patcher = clean_environment()
    patcher.start()
    for name in ENVIRONMENT:
        os.environ.pop(name, None)
    return patcher


class DefaultsTests(unittest.TestCase):
    def setUp(self):
        patcher = without_environment()
        self.addCleanup(patcher.stop)

    def test_the_defaults_are_todays_behaviour(self):
        self.assertIs(playbackprefs.DEFAULT_GPU_NODES, False)
        self.assertIsNone(playbackprefs.DEFAULT_DECODE_WORKERS)
        self.assertIsNone(playbackprefs.DEFAULT_SOURCE_WORKERS)
        self.assertFalse(playbackprefs.gpu_nodes_wanted(None))

    def test_automatic_counts_follow_the_cpu_rules_the_window_used_before_the_settings(self):
        for cpus, decode, source in ((2, 0, 0), (3, 0, 0), (4, 0, 1), (6, 1, 3), (16, 1, 3)):
            self.assertEqual(playbackprefs.decode_workers(None, cpus=cpus), decode, cpus)
            self.assertEqual(playbackprefs.source_workers(None, cpus=cpus), source, cpus)

    def test_a_saved_choice_replaces_the_automatic_count(self):
        self.assertEqual(playbackprefs.decode_workers(2, cpus=4), 2)
        self.assertEqual(playbackprefs.source_workers(2, cpus=4), 2)
        self.assertEqual(playbackprefs.decode_workers(0, cpus=16), 0)
        self.assertEqual(playbackprefs.source_workers("auto", cpus=16), 3)
        self.assertEqual(playbackprefs.decode_workers(99, cpus=16), playbackprefs.MAX_DECODE_WORKERS)
        self.assertEqual(playbackprefs.source_workers("garbage", cpus=16), 3)

    def test_the_environment_wins_over_the_saved_choice(self):
        os.environ[playbackprefs.DECODE_WORKERS_ENV] = "3"
        os.environ[playbackprefs.SOURCE_WORKERS_ENV] = "0"
        os.environ[playbackprefs.GPU_NODES_ENV] = "1"
        self.assertEqual(playbackprefs.decode_workers(1, cpus=16), 3)
        self.assertEqual(playbackprefs.source_workers(2, cpus=16), 0)
        self.assertTrue(playbackprefs.gpu_nodes_wanted(False))
        os.environ[playbackprefs.GPU_NODES_ENV] = "0"
        self.assertFalse(playbackprefs.gpu_nodes_wanted(True))
        self.assertTrue(playbackprefs.gpu_nodes_forced())
        self.assertTrue(playbackprefs.workers_forced(playbackprefs.DECODE_WORKERS_ENV))

    def test_a_default_is_flipped_by_one_line(self):
        with mock.patch.object(playbackprefs, "DEFAULT_GPU_NODES", True):
            self.assertTrue(playbackprefs.gpu_nodes_wanted(None))
            self.assertFalse(playbackprefs.gpu_nodes_wanted(False))     # an explicit choice is kept
        with mock.patch.object(playbackprefs, "DEFAULT_DECODE_WORKERS", 2), \
                mock.patch.object(playbackprefs, "DEFAULT_SOURCE_WORKERS", 2):
            self.assertEqual(playbackprefs.decode_workers(None, cpus=4), 2)
            self.assertEqual(playbackprefs.source_workers(None, cpus=16), 2)
            self.assertEqual(playbackprefs.decode_workers(1, cpus=4), 1)

    def test_the_flag_parser_reads_what_qsettings_hands_back(self):
        for raw, expected in (("true", True), ("false", False), ("1", True), ("0", False), (True, True), (False, False)):
            self.assertEqual(playbackprefs.parse_flag(raw, default=not expected), expected, raw)
        self.assertTrue(playbackprefs.parse_flag("nonsense", default=True))
        self.assertFalse(playbackprefs.parse_flag(None, default=False))

    def test_the_decode_pool_environment_can_turn_the_pool_off(self):
        os.environ[playbackprefs.DECODE_WORKERS_ENV] = "0"
        from nodebased import decodepool
        self.assertEqual(decodepool.default_workers(), 0)


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        patcher = without_environment()
        self.addCleanup(patcher.stop)
        from nodebased.app import Preferences
        self.preferences = Preferences()
        for key in (Preferences.PLAYBACK_GPU_NODES, Preferences.PLAYBACK_DECODE_WORKERS, Preferences.PLAYBACK_SOURCE_WORKERS):
            self.preferences._store.remove(key)
        self.addCleanup(self.clear)

    def clear(self):
        from nodebased.app import Preferences
        for key in (Preferences.PLAYBACK_GPU_NODES, Preferences.PLAYBACK_DECODE_WORKERS, Preferences.PLAYBACK_SOURCE_WORKERS):
            self.preferences._store.remove(key)

    def test_nothing_saved_reads_as_the_default(self):
        self.assertIs(self.preferences.gpu_nodes(), playbackprefs.DEFAULT_GPU_NODES)
        self.assertIsNone(self.preferences.decode_workers())
        self.assertIsNone(self.preferences.source_workers())

    def test_the_gpu_nodes_choice_round_trips(self):
        from nodebased.app import Preferences
        self.preferences.set_gpu_nodes(True)
        self.assertTrue(Preferences().gpu_nodes())
        self.preferences.set_gpu_nodes(False)
        self.assertFalse(Preferences().gpu_nodes())

    def test_the_worker_choices_round_trip_and_automatic_clears_them(self):
        from nodebased.app import Preferences
        self.preferences.set_decode_workers(2)
        self.preferences.set_source_workers(0)
        again = Preferences()
        self.assertEqual(again.decode_workers(), 2)
        self.assertEqual(again.source_workers(), 0)           # off is a choice, distinct from automatic
        self.preferences.set_decode_workers(None)
        self.preferences.set_source_workers(None)
        again = Preferences()
        self.assertIsNone(again.decode_workers())
        self.assertIsNone(again.source_workers())
        self.preferences.set_decode_workers(99)
        self.assertEqual(Preferences().decode_workers(), playbackprefs.MAX_DECODE_WORKERS)

    def test_the_settings_dialog_shows_and_returns_each_choice(self):
        from nodebased.app import ProjectSettingsDialog
        from nodebased.core import empty_document
        dialog = ProjectSettingsDialog(empty_document()["settings"], gpu_nodes=True, decode_workers=2, source_workers=None,
                                       automatic_workers=(1, 3))
        try:
            self.assertTrue(dialog.gpu_nodes.isChecked())
            self.assertEqual(dialog.decode_workers.currentData(), 2)
            self.assertIsNone(dialog.source_workers.currentData())
            self.assertIn("3", dialog.source_workers.currentText())
            dialog.gpu_nodes.setChecked(False)
            dialog.source_workers.setCurrentIndex(dialog.source_workers.findData(0))
            self.assertFalse(dialog.gpu_nodes.isChecked())
            self.assertEqual(dialog.source_workers.currentData(), 0)
        finally:
            dialog.close()


class PoolResizeTests(unittest.TestCase):
    def test_the_source_pool_grows_while_it_lives(self):
        pool = SourcePool(max_workers=1, budget_bytes=1 << 40)
        barrier = threading.Barrier(3, timeout=5)
        names = set()

        def task():
            names.add(threading.current_thread().name)
            barrier.wait()          # three tasks at once need three workers
            return 1
        try:
            pool.resize(3)
            tasks = [pool.submit(task, 1) for _ in range(3)]
            self.assertTrue(wait_for(lambda: len(names) == 3, 5), names)
            for item in tasks:
                self.assertEqual(item.result(), 1)
            self.assertEqual(pool.stats()["workers"], 3)
        finally:
            pool.shutdown(wait=True, timeout=2)

    def test_the_source_pool_shrinks_and_a_pool_of_none_runs_the_work_on_the_joiner(self):
        pool = SourcePool(max_workers=3, budget_bytes=1 << 40)
        try:
            for value in range(3):
                pool.submit(lambda: 0, 1).result()
            pool.submit(lambda: time.sleep(0.05), 1)
            pool.submit(lambda: time.sleep(0.05), 1)
            pool.submit(lambda: time.sleep(0.05), 1)
            self.assertTrue(wait_for(lambda: pool.stats()["threads"] >= 1, 2))
            pool.resize(1)
            self.assertTrue(wait_for(lambda: pool.stats()["threads"] <= 1, 5), pool.stats())
            self.assertEqual(pool.ahead_limit, 1)
            pool.resize(0)
            self.assertFalse(pool.active)
            self.assertTrue(wait_for(lambda: pool.stats()["threads"] == 0, 5), pool.stats())
            ran_on = []
            pool.submit(lambda: ran_on.append(threading.current_thread()), 1).result()
            self.assertEqual(ran_on, [threading.current_thread()])
            pool.resize(2)
            self.assertTrue(pool.active)
            seen = []
            pool.submit(lambda: seen.append(threading.current_thread().name), 1).result()
            self.assertEqual(len(seen), 1)
        finally:
            pool.shutdown(wait=True, timeout=2)

    def test_resizing_a_shut_down_pool_does_nothing(self):
        pool = SourcePool(max_workers=1)
        pool.shutdown(wait=True, timeout=2)
        pool.resize(3)
        self.assertFalse(pool.active)

    def test_the_decode_pool_grows_and_shrinks_while_it_lives(self):
        pool = DecodeAheadPool(max_workers=1, budget_bytes=1 << 30)
        try:
            self.assertEqual(len(pool._threads), 1)
            pool.resize(3)
            self.assertEqual(len(pool._threads), 3)
            barrier = threading.Barrier(3, timeout=5)
            for index in range(3):
                pool.request(("k", index), lambda: (barrier.wait(), np.zeros((2, 2, 4), np.float32))[1])
            self.assertTrue(wait_for(lambda: pool.stats()["entries"] == 3, 5), pool.stats())     # three decodes at once
            pool.resize(1)
            self.assertTrue(wait_for(lambda: len(pool._threads) == 1, 5))
            pool.request(("k", 9), lambda: np.zeros((2, 2, 4), np.float32))
            self.assertTrue(wait_for(lambda: pool.get(("k", 9)) is not None, 5))
            pool.resize(0)
            self.assertTrue(wait_for(lambda: len(pool._threads) == 0, 5))
            pool.request(("k", 10), lambda: np.zeros((2, 2, 4), np.float32))      # no worker: nothing is queued
            self.assertEqual(pool.stats()["pending"], 0)
            pool.resize(2)
            self.assertEqual(len(pool._threads), 2)
        finally:
            pool.shutdown(wait=True, timeout=2)


class WindowSettingTests(unittest.TestCase):
    """A changed setting reaches a running window: no restart, no stale pool."""

    def setUp(self):
        patcher = without_environment()
        self.addCleanup(patcher.stop)
        from nodebased.app import Window
        self.window = Window()
        self.addCleanup(self.window.close)
        for key in (self.window.preferences.PLAYBACK_GPU_NODES, self.window.preferences.PLAYBACK_DECODE_WORKERS,
                    self.window.preferences.PLAYBACK_SOURCE_WORKERS):
            self.window.preferences._store.remove(key)
        self.addCleanup(self.clear)

    def clear(self):
        for key in (self.window.preferences.PLAYBACK_GPU_NODES, self.window.preferences.PLAYBACK_DECODE_WORKERS,
                    self.window.preferences.PLAYBACK_SOURCE_WORKERS):
            self.window.preferences._store.remove(key)

    def test_a_new_window_starts_as_before_the_settings_existed(self):
        window = self.window
        cpus = sourcepool.available_cpus()
        self.assertIsNone(window.evaluator.gpu2d)
        self.assertEqual(window.decode_pool.max_workers, 0 if cpus <= 4 else 1)
        self.assertEqual(window.source_pool.max_workers, sourcepool.default_workers(cpus))
        self.assertEqual(window.gpu_nodes_state["error"], None)

    def test_worker_counts_change_without_a_restart(self):
        window = self.window
        window.preferences.set_decode_workers(2)
        window.preferences.set_source_workers(1)
        self.assertEqual(window.apply_playback_workers(), (2, 1))
        self.assertEqual(window.decode_pool.max_workers, 2)
        self.assertEqual(window.source_pool.max_workers, 1)
        window.preferences.set_decode_workers(0)
        window.preferences.set_source_workers(0)
        window.apply_playback_workers()
        self.assertFalse(window.source_pool.active)
        self.assertEqual(window.decode_pool.max_workers, 0)
        window.preferences.set_decode_workers(None)
        window.preferences.set_source_workers(None)
        window.apply_playback_workers()
        self.assertEqual(window.source_pool.max_workers, sourcepool.default_workers())

    def test_the_environment_fixes_a_count_and_the_dialog_says_so(self):
        window = self.window
        with mock.patch.dict(os.environ, {playbackprefs.SOURCE_WORKERS_ENV: "2"}):
            window.preferences.set_source_workers(0)
            self.assertEqual(window.apply_playback_workers()[1], 2)
            self.assertIn(playbackprefs.SOURCE_WORKERS_ENV, window.workers_note_text())
        with mock.patch.dict(os.environ, {playbackprefs.GPU_NODES_ENV: "1"}):
            self.assertIn(playbackprefs.GPU_NODES_ENV, window.gpu_nodes_note_text())

    def test_a_card_that_refuses_leaves_the_cpu_nodes_in_use_with_a_message(self):
        from nodebased import gpu2d
        window = self.window
        messages = []
        shown = window.statusBar().showMessage
        window.statusBar().showMessage = lambda text, *a, **k: (messages.append(text), shown(text, *a, **k))[1]
        window.preferences.set_gpu_nodes(True)
        with mock.patch.object(gpu2d.GPU2D, "open", classmethod(lambda cls, *a, **k: setattr(cls, "open_error", "no adapter") or None)):
            window.apply_gpu_nodes(announce=True)
            self.assertTrue(wait_for(lambda: window.gpu_nodes_state["error"] is not None))
        self.assertIsNone(window.evaluator.gpu2d)
        self.assertIn("no adapter", window.gpu_nodes_state["error"])
        self.assertTrue(any("CPU nodes stay in use" in m for m in messages), messages)
        self.assertIn("no adapter", window.gpu_nodes_note_text())
        window.preferences.set_gpu_nodes(False)         # turning it off again clears the refusal
        window.apply_gpu_nodes()
        self.assertIsNone(window.gpu_nodes_state["error"])


class IslandSettingTests(unittest.TestCase):
    """The island itself, on the software adapter so no physical card is needed; a physical card is exercised by the
    benchmarks (tools/benchmark_gpu2d.py playback)."""

    @classmethod
    def setUpClass(cls):
        from nodebased import gpu2d
        cls.probe = gpu2d.GPU2D.open("cpu", 256 << 20)
        if cls.probe is not None:
            cls.probe.release_all()
        cls._directory = tempfile.TemporaryDirectory()
        make_plates(cls._directory.name)
        cls.doc = small_comp(cls._directory.name)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def setUp(self):
        if self.probe is None:
            self.skipTest("no software wgpu adapter")
        patcher = without_environment()
        self.addCleanup(patcher.stop)
        from nodebased import gpu2d
        original = gpu2d.GPU2D.open.__func__
        # Every open in these tests lands on the software adapter, whatever the machine's default is.
        patched = mock.patch.object(gpu2d.GPU2D, "open", classmethod(lambda cls, choice="default", budget=256 << 20: original(cls, "cpu", budget)))
        patched.start()
        self.addCleanup(patched.stop)
        from nodebased.app import Window
        self.window = Window(self.doc)
        self.addCleanup(self.window.close)
        self.window.preferences._store.remove(self.window.preferences.PLAYBACK_GPU_NODES)
        self.addCleanup(lambda: self.window.preferences._store.remove(self.window.preferences.PLAYBACK_GPU_NODES))

    def record_messages(self):
        messages = []
        shown = self.window.statusBar().showMessage
        self.window.statusBar().showMessage = lambda text, *a, **k: (messages.append(text), shown(text, *a, **k))[1]
        return messages

    def frames(self, evaluator, numbers):
        return [evaluator.evaluate_raster(self.doc, "colorcorrect", frame=n) for n in numbers]

    def assert_gate(self, got, want, label):
        got, want = got.pixels, want.pixels
        self.assertEqual(got.shape, want.shape, label)
        finite = np.isfinite(want)
        np.testing.assert_array_equal(np.isfinite(got), finite, label)
        self.assertTrue(np.all(np.abs(got[finite] - want[finite]) <= 2e-5 + 2e-5 * np.abs(want[finite])), label)

    def test_turning_the_gpu_nodes_on_and_off_takes_effect_without_a_restart_and_keeps_the_pixels(self):
        window = self.window
        numbers = list(range(1, FRAMES))
        reference = self.frames(Evaluator(), numbers)               # the CPU nodes, a fresh evaluator
        before = self.frames(window.evaluator, numbers)
        self.assertIsNone(window.evaluator.gpu2d)
        messages = self.record_messages()
        window.preferences.set_gpu_nodes(True)
        window.apply_gpu_nodes(announce=True)
        self.assertTrue(wait_for(lambda: window.evaluator.gpu2d is not None and window.gpu_nodes_state["active"]), window.gpu_nodes_state)
        self.assertTrue(any("GPU nodes on" in m for m in messages), messages)
        runs = window.evaluator.gpu2d.counters.runs
        on = self.frames(window.evaluator, numbers)
        self.assertGreater(window.evaluator.gpu2d.counters.runs, runs, "the frames did not go through the island")
        for n, a, b, c in zip(numbers, reference, before, on):
            self.assertTrue(np.array_equal(a.pixels, b.pixels), n)  # the evaluator with the setting off is the CPU nodes
            self.assert_gate(c, a, f"frame {n} with the GPU nodes on")
        window.preferences.set_gpu_nodes(False)
        window.apply_gpu_nodes(announce=True)
        self.assertTrue(wait_for(lambda: window.evaluator.gpu2d is None and not window.gpu_nodes_state["active"]))
        off = self.frames(window.evaluator, numbers)
        for n, a, d in zip(numbers, reference, off):
            self.assertTrue(np.array_equal(a.pixels, d.pixels), f"frame {n} after turning them off again")

    def test_a_choice_made_while_the_card_opens_is_applied_afterwards(self):
        window = self.window
        window.preferences.set_gpu_nodes(True)
        window.apply_gpu_nodes()
        window.preferences.set_gpu_nodes(False)         # changed again before the card answered
        window.apply_gpu_nodes()
        self.assertTrue(wait_for(lambda: not window.gpu_nodes_state["pending"]))
        self.assertTrue(wait_for(lambda: window.evaluator.gpu2d is None and not window.gpu_nodes_state["active"], 10))

    def test_the_environment_turns_the_gpu_nodes_on_for_a_benchmark(self):
        with mock.patch.dict(os.environ, {playbackprefs.GPU_NODES_ENV: "1"}):
            self.assertTrue(playbackprefs.gpu_nodes_wanted(False))
            self.window.apply_gpu_nodes()
            self.assertTrue(wait_for(lambda: self.window.evaluator.gpu2d is not None))

    def test_a_card_that_loses_its_device_is_dropped_and_the_frames_stay_right(self):
        window = self.window
        messages = self.record_messages()
        window.preferences.set_gpu_nodes(True)
        window.apply_gpu_nodes()
        self.assertTrue(wait_for(lambda: window.evaluator.gpu2d is not None))
        backend = window.evaluator.gpu2d
        reference = Evaluator().evaluate_raster(self.doc, "colorcorrect", frame=2)
        backend.invalidate("simulated device loss")
        lost = window.evaluator.evaluate_raster(self.doc, "colorcorrect", frame=2)       # the frame reruns on the CPU nodes
        np.testing.assert_array_equal(lost.pixels, reference.pixels)
        window._check_gpu_nodes_lost()
        self.assertTrue(wait_for(lambda: window.evaluator.gpu2d is None and window.gpu_nodes_state["error"] is not None))
        self.assertIn("lost", str(window.gpu_nodes_state["error"]))
        self.assertTrue(any("CPU nodes stay in use" in m for m in messages), messages)

    def test_worker_counts_do_not_change_the_pixels(self):
        numbers = list(range(1, FRAMES))
        reference = self.frames(Evaluator(), numbers)
        for decode, source in ((0, 0), (1, 1), (2, 2), (3, 3)):
            window = self.window
            window.preferences.set_decode_workers(decode)
            window.preferences.set_source_workers(source)
            window.apply_playback_workers()
            window.evaluator.clear()
            got = self.frames(window.evaluator, numbers)
            for n, a, b in zip(numbers, reference, got):
                self.assertTrue(np.array_equal(a.pixels, b.pixels), (decode, source, n))
        window = self.window
        window.preferences.set_decode_workers(None)
        window.preferences.set_source_workers(None)


if __name__ == "__main__":
    unittest.main()
