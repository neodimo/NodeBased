"""Played frames are converted and shown while the next frame evaluates (docs/PERFORMANCE.md, Lane 2 playback pipeline)."""
import copy
import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.waiting import wait_until

from PySide6.QtWidgets import QApplication

import numpy as np

import nodebased.app as app_module
from nodebased import carddisplay
from nodebased import playback_trace
from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, demo_document
from nodebased.playback import ConversionStage

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


class ConversionStageTests(unittest.TestCase):
    def test_frames_convert_in_order_and_a_superseded_frame_is_counted_not_shown(self):
        done, gate = [], threading.Event()
        def convert(job):
            gate.wait(2.0)
            time.sleep(0.01)
            done.append(job)
        stage = ConversionStage(convert)
        self.addCleanup(stage.close)
        for job in range(1, 8):          # a slow converter: 1 is taken at once, 2..6 are replaced, 7 waits
            stage.submit(job)
            if job == 1:
                self.assertTrue(wait_until(lambda: not stage._pending, 2.0))
        gate.set()
        self.assertTrue(wait_until(lambda: stage.idle, 3.0))
        self.assertEqual(done, [1, 7])
        self.assertEqual(done, sorted(set(done)))                    # in order, none twice
        self.assertEqual(stage.dropped, 5)                           # 2..6
        self.assertEqual(stage.dropped_jobs, [2, 3, 4, 5, 6])
        self.assertEqual(len(done) + stage.dropped, 7)               # every frame is shown or counted

    def test_wait_for_room_stops_a_fast_evaluator_getting_ahead(self):
        done = []
        def convert(job):
            time.sleep(0.02)
            done.append(job)
        stage = ConversionStage(convert)
        self.addCleanup(stage.close)
        for job in range(10):
            stage.wait_for_room()
            stage.submit(job)
        self.assertTrue(wait_until(lambda: stage.idle, 3.0))
        self.assertEqual(done, list(range(10)))
        self.assertEqual(stage.dropped, 0)

    def test_wait_for_room_gives_up_when_asked_to(self):
        stage = ConversionStage(lambda job: time.sleep(0.3))
        self.addCleanup(stage.close)
        stage.submit(1)
        self.assertTrue(wait_until(lambda: stage._working, 2.0))
        stage.submit(2)                                              # now one waits
        self.assertFalse(stage.wait_for_room(abort=lambda: False, timeout=0.05))
        self.assertFalse(stage.wait_for_room(abort=lambda: True))

    def test_discard_drops_the_waiting_frame_and_counts_it(self):
        stage = ConversionStage(lambda job: time.sleep(0.2))
        self.addCleanup(stage.close)
        stage.submit(1)
        self.assertTrue(wait_until(lambda: stage._working, 2.0))
        stage.submit(2)
        stage.discard()
        self.assertEqual(stage.dropped, 1)
        stage.reset_count()
        self.assertEqual(stage.dropped, 0)


def animated_document():
    dispatcher = Dispatcher(demo_document())
    for frame, value in ((1, 0.0), (24, 1.0)):
        dispatcher.execute({"op": "set_key", "id": "grade", "param": "exposure", "frame": frame, "value": value})
    dispatcher.execute({"op": "time", "first": 1, "last": 24})
    return dispatcher.document


class PlaybackPipelineWindowTests(unittest.TestCase):
    EVALUATION_SECONDS = 0.09
    CONVERSION_SECONDS = 0.02

    def setUp(self):
        self.window = Window(animated_document())
        self.window.show()
        self.addCleanup(self.window.close)
        self.addCleanup(lambda: setattr(self.window, "saved_document", copy.deepcopy(self.window.dispatcher.document)))
        wait_until(lambda: not self.window.busy and not len(self.window.preview_queue), 15.0)
        APP.processEvents()
        self.lock = threading.Lock()
        self.evaluations, self.conversions, self.shown = [], [], []
        window = self.window
        window.playback_proxy.setChecked(False)     # the proxy tiers would make a frame cheap enough to keep up
        compose = window.tile_executor.compose_region
        evaluate = window.evaluator.evaluate
        def slow_compose(*a, **kw):
            began = time.monotonic()
            time.sleep(self.EVALUATION_SECONDS)
            out = compose(*a, **kw)
            with self.lock:
                self.evaluations.append((began, time.monotonic(), threading.current_thread().name))
            return out
        def slow_evaluate(*a, **kw):
            began = time.monotonic()
            time.sleep(self.EVALUATION_SECONDS)
            out = evaluate(*a, **kw)
            with self.lock:
                self.evaluations.append((began, time.monotonic(), threading.current_thread().name))
            return out
        window.tile_executor.compose_region = slow_compose
        window.evaluator.evaluate = slow_evaluate
        original = app_module.to_qimage
        def counting(*a, **kw):
            began = time.monotonic()
            out = original(*a, **kw)
            time.sleep(self.CONVERSION_SECONDS)
            with self.lock:
                if self.window.playing:
                    self.conversions.append((began, time.monotonic(), threading.current_thread().name))
            return out
        counting.__wrapped__ = original
        app_module.to_qimage = counting
        self.addCleanup(setattr, app_module, "to_qimage", original)
        window.signals.finished.connect(self._finished)
        window.display_cache.clear() if hasattr(window.display_cache, "clear") else None

    def _finished(self, payload, frame, image, status, region=None):
        request, _cancel = payload
        if request.display and frame is not None and self.window.frame_generation == request.generation:
            self.shown.append((request.generation, request.frame, self.window.statusBar().currentMessage()))

    def play(self, seconds):
        self.window.toggle_playback(True)
        wait_until(lambda: False, seconds)
        self.window.toggle_playback(False)
        wait_until(lambda: not self.window.busy and self.window.convert_stage.idle, 5.0)
        APP.processEvents()

    def test_the_next_frame_evaluates_while_the_last_one_converts(self):
        self.window.pipeline_playback = True
        self.play(2.0)
        with self.lock:
            evaluations, conversions = list(self.evaluations), list(self.conversions)
        self.assertGreaterEqual(len(self.shown), 4)
        self.assertGreaterEqual(len(conversions), 4)
        self.assertTrue(all(name.startswith("nodebased-convert") for _b, _e, name in conversions))
        self.assertTrue(all(name.startswith("nodebased-preview") for _b, _e, name in evaluations))
        overlaps = [1 for (cb, ce, _), (eb, ee, _) in ((c, e) for c in conversions for e in evaluations)
                    if eb < ce and cb < ee]
        self.assertGreater(len(overlaps), 0, "no evaluation ran while a conversion was in progress")

    def test_one_thread_order_when_the_pipeline_is_off(self):
        self.window.pipeline_playback = False
        self.play(1.5)
        with self.lock:
            conversions, evaluations = list(self.conversions), list(self.evaluations)
        self.assertTrue(conversions)
        self.assertTrue(all(name.startswith("nodebased-preview") for _b, _e, name in conversions))
        for (cb, ce, _), (eb, ee, _) in ((c, e) for c in conversions for e in evaluations):
            self.assertFalse(eb < ce and cb < ee, "evaluation overlapped a conversion on one worker")

    def test_frames_are_shown_in_order_and_once(self):
        self.window.pipeline_playback = True
        self.play(2.5)
        generations = [g for g, _f, _s in self.shown]
        self.assertGreaterEqual(len(generations), 4)
        self.assertEqual(generations, sorted(set(generations)))      # increasing: none twice, none late
        frames = [f for _g, f, _s in self.shown]
        for before, after in zip(frames, frames[1:]):
            self.assertTrue(after > before or after == 1, (before, after))   # forward, or the loop wrapped
        playing_rows = [s for _g, _f, s in self.shown if "dropped" in s]
        self.assertTrue(playing_rows, "the time row must report dropped frames while playing")
        self.assertRegex(playing_rows[-1], r"display (GPU|CPU)")

    def test_a_frame_the_conversion_could_not_take_is_counted_in_the_time_row(self):
        """Evaluation outruns a slow conversion and nothing holds it back: frames are replaced while they wait,
        and the time row counts them. What is shown is still increasing and never repeated."""
        window = self.window
        window.pipeline_playback = True
        self.EVALUATION_SECONDS = 0.0
        self.CONVERSION_SECONDS = 0.25
        window.convert_stage.wait_for_room = lambda *a, **k: True
        # every frame to be evaluated is a fresh one (no display cache hits): the animated grade is the only change
        window.display_cache = type(window.display_cache)(force_float32=False)
        self.play(3.0)
        dropped_by_stage = window.convert_stage.dropped
        self.assertGreater(dropped_by_stage, 0)
        generations = [g for g, _f, _s in self.shown]
        self.assertEqual(generations, sorted(set(generations)))
        rows = [s for _g, _f, s in self.shown if "dropped" in s]
        import re
        counts = [int(re.search(r"dropped (\d+)", row).group(1)) for row in rows]
        self.assertGreater(max(counts), 0)
        self.assertEqual(counts, sorted(counts))
        # every evaluated frame was shown or counted as dropped
        evaluated = len([1 for _b, _e, name in self.evaluations])
        self.assertGreaterEqual(len(self.shown) + dropped_by_stage + 2, evaluated)

    def test_stopping_leaves_no_picture_waiting(self):
        self.window.pipeline_playback = True
        self.play(1.0)
        shown = len(self.shown)
        wait_until(lambda: False, 0.4)
        self.assertEqual(len(self.shown), shown)
        self.assertTrue(self.window.convert_stage.idle)
        self.assertFalse(self.window.busy)

    def test_stopping_shows_the_frame_at_the_playhead(self):
        self.window.pipeline_playback = True
        self.play(1.5)
        self.assertTrue(wait_until(lambda: self.window.frame is not None and not self.window.busy, 5.0))
        current = self.window.dispatcher.document["time"]["current"]
        self.assertEqual(self.window._displayed_frame, current)

    def test_viewer_picture_is_unchanged_by_the_pipeline(self):
        """Direct picture off: the Viewer shows the same 8-bit picture `to_qimage` gives for the frame it holds."""
        window = self.window
        window.pipeline_playback = True
        self.play(1.5)
        self.assertTrue(wait_until(lambda: window.frame is not None and not window.busy, 5.0))
        document = window.dispatcher.document
        expected = app_module.to_qimage.__wrapped__(window.frame, window.exposure.value(), window.channels.currentText(),
                                                   background=document["settings"]["viewer"]["background"],
                                                   view=window.effective_view(document), look=window.look_args(document))
        shown = window.viewer.last_display_image
        self.assertEqual(shown.size(), expected.size())
        self.assertEqual(shown.format(), expected.format())
        self.assertEqual(bytes(shown.constBits()), bytes(expected.constBits()))

    def test_direct_picture_playback_builds_no_8bit_picture(self):
        window = self.window
        window.pipeline_playback = True
        window._direct_linear_on = True
        window._direct_limits = None
        pictures = []
        window._show_direct_linear = lambda image, scale, region, generation: (pictures.append(image), True)[1]
        self.play(1.5)
        self.assertGreaterEqual(len(pictures), 3)
        self.assertTrue(all(isinstance(p, carddisplay.LazyPicture) for p in pictures))
        with self.lock:
            self.assertEqual(self.conversions, [], "an 8-bit picture was built while the Direct picture played")

    def test_the_trace_accounts_for_every_displayed_frame(self):
        self.window.pipeline_playback = True
        playback_trace.enable()
        try:
            self.play(2.0)
        finally:
            events = playback_trace.disable()
        rows = playback_trace.frame_timeline(events)
        self.assertGreaterEqual(len(rows), 4)
        for row in rows:
            self.assertAlmostEqual(sum(row["stages_ms"].values()), row["latency_ms"], delta=0.05)
        summary = playback_trace.frame_summary(rows)
        self.assertGreater(summary["overlap_ms"], 10.0)
        self.assertAlmostEqual(summary["accounted_interval_ms"], summary["interval_ms"], delta=0.5)


class RecordedTimelineTests(unittest.TestCase):
    """The committed 10/22 timelines (docs/PERFORMANCE.md, Lane 2 playback pipeline)."""

    @staticmethod
    def load(name):
        import json
        import pathlib
        path = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "2d-playback" / f"pipeline-1022-{name}.timeline.json"
        return json.loads(path.read_text())

    NAMES = [f"{mode}-{target}" for mode in ("serial", "pipe")
             for target in ("rtx-1", "rtx-2", "radeon", "llvmpipe", "laptop", "direct-rtx")]

    def test_the_stages_add_up_to_the_measured_frame_interval_within_five_percent(self):
        for name in self.NAMES:
            accounting = self.load(name)["accounting"]
            self.assertLessEqual(abs(accounting["difference_percent"]), 5.0, name)
            self.assertGreaterEqual(accounting["fresh_frames_in_first_pass"], 30, name)

    def test_pipelined_rtx_frame_interval_is_below_evaluation_plus_conversion(self):
        for name in ("pipe-rtx-1", "pipe-rtx-2"):
            data = self.load(name)
            summary = data["fresh_frame_timeline"]["summary"]
            stages = summary["stages_ms"]
            self.assertLess(summary["accounted_interval_ms"],
                            stages["node evaluation"] + stages["display conversion"] - 40.0, name)
            self.assertGreater(summary["overlap_ms"], 40.0, name)
            self.assertLess(data["handoff_ms"]["p50"], 10.0, name)
        for name in ("serial-rtx-1", "serial-rtx-2"):
            self.assertEqual(self.load(name)["fresh_frame_timeline"]["summary"]["overlap_ms"], 0.0)

    def test_the_fresh_rate_gain_on_the_rtx_is_recorded(self):
        before = (self.load("serial-rtx-1")["first_pass_fps"] + self.load("serial-rtx-2")["first_pass_fps"]) / 2
        after = (self.load("pipe-rtx-1")["first_pass_fps"] + self.load("pipe-rtx-2")["first_pass_fps"]) / 2
        self.assertGreater(after / before, 1.15)

    def test_direct_picture_runs_built_no_8bit_picture(self):
        for name in ("serial-direct-rtx", "pipe-direct-rtx"):
            self.assertEqual(self.load(name)["display_conversions_8bit"], 0)

    def test_the_reduction_tool_reproduces_a_recorded_accounting(self):
        from tools.timeline_2d_playback import reduce
        raw = {"case": "x", "timeline": [
            {"stage": "ui_handoff", "t_ms": 1000.0 + 150.0 * i, "frame": i + 1, "presented": True} for i in range(5)],
            "fresh_frame_timeline": {"summary": {"accounted_interval_ms": 151.0, "stages_ms": {}}, "frames": []}}
        out = reduce(raw)
        self.assertEqual(out["accounting"]["measured_first_pass_interval_ms"], 150.0)
        self.assertEqual(out["accounting"]["difference_percent"], 0.67)


if __name__ == "__main__":
    unittest.main()
