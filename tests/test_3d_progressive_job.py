"""Lane 4 step H2: the Render mode traces off the interface thread (`progressivejob.ProgressiveJob`) and the viewport shows
what the worker published. Scripted steps stand in for the path tracer so the thread behaviour (supersession, cancellation
latency, settling, failure) and the viewport's labels are asserted without timing luck; the real tracer is covered by
`test_3d_viewport_render_mode`, `test_3d_first_look` and `test_3d_progressive_convergence`.
"""
import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import progressivejob, progressiverender as P
from nodebased.cancellation import Cancelled
from nodebased.progressivejob import ProgressiveJob, Request
from nodebased.progressiverender import ProgressiveState


def request(identity, width=64, height=48):
    return Request(identity, scene=object(), camera=object(), width=width, height=height, background=(0, 0, 0, 1), ambient=0.1)


def fake_state(key, samples, width, height, preview=False, stage=0, passes=1, **more):
    return ProgressiveState(key=key, samples=samples, width=width, height=height,
                            image=np.full((height, width, 4), 0.5, np.float32), passes=passes, preview=preview,
                            denoised=True, stage=stage, traced_width=width // 4, traced_height=height // 4, **more)


class Script:
    """A step that returns `plan[i]` for its i-th call of a view and waits at `gate[i]` first (cancel-aware, like the tracer)."""

    def __init__(self, plan, gated=True):
        self.plan, self.calls, self.started, self.seen_cancel = plan, {}, [], threading.Event()
        self.gates = [threading.Event() for _ in plan]
        if not gated:
            for gate in self.gates:
                gate.set()
        self.entered = threading.Event()

    def open(self, count=None):
        for gate in (self.gates if count is None else self.gates[:count]):
            gate.set()

    def __call__(self, state, scene, camera, width, height, background, ambient, key, backend, max_bounces, moments, cancel):
        index = self.calls.get(key, 0)
        self.calls[key] = index + 1
        self.started.append((key, index))
        self.entered.set()
        while not self.gates[min(index, len(self.gates) - 1)].wait(0.005):
            if cancel is not None and cancel.is_set():
                self.seen_cancel.set()
                raise Cancelled()
        if cancel is not None and cancel.is_set():
            self.seen_cancel.set()
            raise Cancelled()
        return self.plan[min(index, len(self.plan) - 1)](key, width, height)


def preview(key, w, h):
    return fake_state(key, 4, w, h, preview=True, stage=1, passes=1)


def refining(key, w, h):
    return fake_state(key, 32, w, h, passes=3)


def final(key, w, h):
    return fake_state(key, P.SAMPLE_CAP, w, h, passes=5)


class JobTests(unittest.TestCase):
    def make(self, script, settle=0.0, **more):
        job = ProgressiveJob(step=script, settle=settle, **more)
        self.addCleanup(job.close)
        return job

    def test_publishes_every_state_in_order_and_settles_at_the_cap(self):
        script = Script([preview, refining, final], gated=False)
        job = self.make(script)
        self.assertTrue(job.submit(request("a")))
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        state, phase = job.view("a")
        self.assertEqual((state.samples, phase), (P.SAMPLE_CAP, "settled"))
        self.assertEqual([frame.state.samples for frame in job.frames], [4, 32, P.SAMPLE_CAP])
        self.assertEqual([frame.serial for frame in job.frames], [1, 2, 3])
        self.assertTrue(all(frame.since_submit >= 0 for frame in job.frames))

    def test_a_view_nobody_asked_for_reads_idle_and_never_a_picture(self):
        job = self.make(Script([preview, final], gated=False))
        self.assertEqual(job.view("a"), (None, "idle"))
        job.submit(request("a"))
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        self.assertEqual(job.view("b"), (None, "idle"))

    def test_the_same_view_is_not_restarted(self):
        script = Script([preview, final], gated=False)
        job = self.make(script)
        job.submit(request("a"))
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        self.assertFalse(job.submit(request("a")))
        self.assertEqual(script.calls["a"], 2)

    def test_a_new_view_cancels_the_pass_in_flight_promptly_and_starts_over(self):
        script = Script([preview, refining, final])
        job = self.make(script)
        job.submit(request("old"))
        self.assertTrue(script.entered.wait(5))
        started = time.perf_counter()
        job.submit(request("new"))
        self.assertTrue(script.seen_cancel.wait(5))
        self.assertLess(time.perf_counter() - started, 1.0)           # the obsolete pass stopped at its next boundary
        script.open()
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        self.assertEqual(job.view("old"), (None, "idle"))
        self.assertEqual(job.view("new")[0].samples, P.SAMPLE_CAP)
        self.assertEqual(script.started[-3:], [("new", 0), ("new", 1), ("new", 2)])    # a fresh first look, not the old pass's next

    def test_older_pictures_are_dropped_from_the_record_unless_asked_for(self):
        for keep in (False, True):
            job = self.make(Script([preview, refining, final], gated=False), keep_images=keep)
            job.submit(request(("keep", keep)))
            self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
            self.assertEqual([frame.state.samples for frame in job.frames], [4, 32, P.SAMPLE_CAP])    # the numbers stay
            held = [frame.state.image is not None and frame.rgba is not None for frame in job.frames]
            self.assertEqual(held, [True, True, True] if keep else [False, False, True])

    def test_the_time_to_abandon_a_pass_is_recorded(self):
        script = Script([preview, final])
        job = self.make(script)
        job.submit(request("old"))
        self.assertTrue(script.entered.wait(5))
        job.submit(request("new"))
        script.open()
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        self.assertEqual(len(job.cancel_latencies), 1)
        self.assertLess(job.cancel_latencies[0], 1.0)

    def test_a_late_result_of_a_superseded_view_is_never_published(self):
        """The tracer finished the old pass just as the new request came in: its picture belongs to a view nobody shows."""
        release = threading.Event()

        def slow(state, scene, camera, width, height, background, ambient, key, backend, max_bounces, moments, cancel):
            if key == "old":
                release.wait(5)               # ignores cancel: it returns a picture after being superseded
            return fake_state(key, P.SAMPLE_CAP, width, height)
        job = self.make(slow)
        job.submit(request("old"))
        time.sleep(0.05)
        job.submit(request("new"))
        release.set()
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        self.assertEqual([frame.state.key for frame in job.frames], ["new"])
        self.assertEqual(job.view("old"), (None, "idle"))

    def test_a_burst_of_requests_traces_only_the_last_view(self):
        script = Script([final], gated=False)
        job = self.make(script, settle=0.15)
        for i in range(12):
            job.submit(request(("drag", i)))
            time.sleep(0.005)
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        self.assertEqual(sorted(script.calls), [("drag", 11)])       # the camera was still moving for the others

    def test_the_first_look_phase_is_waiting_until_a_picture_exists(self):
        script = Script([preview, final])
        job = self.make(script)
        job.submit(request("a"))
        self.assertTrue(script.entered.wait(5))
        self.assertEqual(job.view("a"), (None, "waiting"))
        script.open(1)
        self.assertTrue(job.wait(lambda state, phase: state is not None, 10))
        self.assertEqual(job.view("a")[1], "tracing")
        script.open()
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))

    def test_a_failing_step_is_reported_once_and_not_retried(self):
        calls = []

        def broken(*args):
            calls.append(1)
            raise ValueError("no particles")
        job = self.make(broken)
        job.submit(request("a"))
        self.assertTrue(job.wait(lambda state, phase: phase == "failed", 10))
        self.assertIsInstance(job.error, ValueError)
        self.assertFalse(job.submit(request("a")))
        time.sleep(0.1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(job.submit(request("b")))                     # another view works again
        self.assertTrue(job.wait(lambda state, phase: phase == "failed", 10))
        self.assertEqual(len(calls), 2)

    def test_cancel_stops_the_work_and_forgets_the_view(self):
        script = Script([preview, final])
        job = self.make(script)
        job.submit(request("a"))
        self.assertTrue(script.entered.wait(5))
        job.cancel()
        self.assertTrue(script.seen_cancel.wait(5))
        self.assertTrue(job.wait(timeout=5))
        self.assertEqual(job.view("a"), (None, "idle"))
        self.assertTrue(job.submit(request("a")))                      # asking again starts it again

    def test_close_ends_the_worker_thread(self):
        job = ProgressiveJob(step=Script([final], gated=False), settle=0.0)
        job.submit(request("a"))
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        thread = job._thread
        job.close()
        self.assertFalse(thread.is_alive())
        self.assertFalse(job.submit(request("b")))

    def test_a_cancelled_step_ends_quietly(self):
        def cancelled(*args):
            raise Cancelled()
        job = self.make(cancelled)
        job.submit(request("a"))
        self.assertTrue(job.wait(timeout=5))
        self.assertEqual(job.view("a")[1], "waiting")
        self.assertIsNone(job.error)

    def test_working_seconds_counts_the_pass_in_flight_and_is_zero_when_idle(self):
        script = Script([preview, final])
        job = self.make(script)
        self.assertEqual(job.working_seconds(), 0.0)
        job.submit(request("a"))
        self.assertTrue(script.entered.wait(5))
        time.sleep(0.12)
        self.assertGreaterEqual(job.working_seconds(), 0.1)
        script.open()
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 10))
        self.assertEqual(job.working_seconds(), 0.0)


class NextSamplesTests(unittest.TestCase):
    def test_the_pass_after_each_state(self):
        stages = ((480 * 270, 4), (960 * 540, 16))
        w, h = 1920, 1080
        self.assertEqual(P.next_samples(None, w, h, stages), 4)
        first = fake_state("a", 4, w, h, preview=True, stage=1)
        self.assertEqual(P.next_samples(first, w, h, stages), 16)
        second = fake_state("a", 16, w, h, preview=True, stage=2)
        self.assertEqual(P.next_samples(second, w, h, stages), 32)
        self.assertEqual(P.next_samples(fake_state("a", 32, w, h, stage=2), w, h, stages), 64)
        self.assertEqual(P.next_samples(fake_state("a", P.SAMPLE_CAP, w, h, stage=2), w, h, stages), P.SAMPLE_CAP)

    def test_a_slow_first_stage_skips_stage_two(self):
        stages = ((480 * 270, 4), (960 * 540, 16))
        slow = fake_state("a", 4, 1920, 1080, preview=True, stage=1, seconds=P.STAGE_TWO_MAX_FIRST_SECONDS + 0.1)
        self.assertEqual(P.next_samples(slow, 1920, 1080, stages), 8)

    def test_a_view_smaller_than_the_first_stage_starts_the_doubling(self):
        self.assertEqual(P.next_samples(None, 160, 90, ((480 * 270, 4),)), P.FIRST_PASS_SAMPLES)


try:
    from PySide6.QtWidgets import QApplication
    from nodebased.viewport3d import Viewport3D
    from tests.test_3d_viewport_render_mode import APP, graph
except Exception as error:      # pragma: no cover - no Qt in this environment
    Viewport3D = None


@unittest.skipIf(Viewport3D is None, "no Qt")
class ViewportLabelTests(unittest.TestCase):
    """What the viewport says and draws for each stage, with scripted passes."""

    def widget(self, script):
        widget = Viewport3D()
        self.addCleanup(widget.close)
        widget.resize(64, 48)
        widget.set_document(graph().document)
        widget._progressive_job = ProgressiveJob(on_update=widget.progressive_ready.emit, settle=0.0, step=script)
        widget.render_mode = True
        return widget

    def paint(self, widget, predicate):
        widget.grab()
        self.assertTrue(widget._progressive_job.wait(predicate, 10))
        widget.grab()

    def test_each_stage_is_labelled_with_its_samples_pass_and_progress(self):
        script = Script([preview, refining, final])
        widget = self.widget(script)
        widget.grab()
        self.assertIn("preparing", widget.render_note)
        self.assertEqual(widget.render_stage, "waiting")
        self.assertIsNone(widget._progressive_state)
        script.open(1)
        self.paint(widget, lambda state, phase: state is not None)
        self.assertEqual(widget.render_stage, "preview")
        self.assertRegex(widget.render_note, r"PREVIEW · denoised, enlarged from \d+×\d+ · 4 samples · pass 1 · tracing \d+ samples")
        self.assertAlmostEqual(widget.render_progress, 4 / P.SAMPLE_CAP)
        script.open(2)
        self.paint(widget, lambda state, phase: state is not None and state.samples == 32)
        self.assertEqual(widget.render_stage, "refining")
        self.assertIn("REFINING", widget.render_note)
        self.assertIn("32 samples", widget.render_note)
        self.assertIn("pass 3", widget.render_note)
        self.assertAlmostEqual(widget.render_progress, 0.5)
        script.open()
        self.paint(widget, lambda state, phase: phase == "settled")
        self.assertEqual(widget.render_stage, "settled")
        self.assertIn("SETTLED", widget.render_note)
        self.assertEqual(widget.render_progress, 1.0)

    def test_the_worker_converts_each_picture_for_display_and_the_widget_paints_that(self):
        widget = self.widget(Script([preview, final], gated=False))
        self.paint(widget, lambda state, phase: phase == "settled")
        frames = widget._progressive_job.frames
        self.assertEqual(frames[-1].rgba.dtype, np.uint8)
        np.testing.assert_array_equal(frames[-1].rgba, P.display_rgba8(frames[-1].state.image))
        self.assertIs(widget._progressive_image[3], frames[-1].rgba)        # no second conversion on the interface thread

    def test_an_edit_never_leaves_the_old_scenes_picture_on_screen(self):
        widget = self.widget(Script([final], gated=False))
        self.paint(widget, lambda state, phase: phase == "settled")
        self.assertIsNotNone(widget._progressive_state)
        widget.set_document(graph().document)         # any edit hands the viewport a document again
        widget.grab()
        self.assertIsNone(widget._progressive_state)
        self.assertIn("preparing", widget.render_note)

    def test_a_repaint_of_the_same_view_does_not_restart_it(self):
        script = Script([final], gated=False)
        widget = self.widget(script)
        self.paint(widget, lambda state, phase: phase == "settled")
        for _ in range(3):
            widget.grab()
        self.assertEqual(script.calls and sum(script.calls.values()), 1)

    def test_a_resize_is_a_new_view(self):
        script = Script([final], gated=False)
        widget = self.widget(script)
        self.paint(widget, lambda state, phase: phase == "settled")
        widget.resize(400, 300)
        self.paint(widget, lambda state, phase: phase == "settled")
        self.assertEqual(sum(script.calls.values()), 2)

    def test_leaving_the_mode_or_hiding_the_widget_cancels_the_pass(self):
        script = Script([preview, final])
        widget = self.widget(script)
        widget.show()
        self.addCleanup(widget.hide)
        widget.grab()
        self.assertTrue(script.entered.wait(5))
        widget.hide()
        self.assertTrue(script.seen_cancel.wait(5))
        self.assertEqual(widget.render_stage, "")
        script.seen_cancel.clear()
        widget.grab()                                    # painting again asks again
        self.assertTrue(widget._progressive_job.wait(lambda state, phase: state is not None or phase == "waiting", 10))

    def test_the_interface_thread_is_not_held_by_a_slow_pass(self):
        script = Script([preview, final])
        widget = self.widget(script)
        started = time.perf_counter()
        for _ in range(5):
            widget.grab()
        self.assertLess(time.perf_counter() - started, 2.0)     # five paints while the worker is still inside its first pass
        self.assertEqual(widget.render_stage, "waiting")
        script.open()

    def test_a_failure_falls_back_to_the_interactive_viewport_with_the_reason(self):
        def broken(*args):
            raise ValueError("no particles")
        widget = self.widget(broken)
        widget.grab()
        self.assertTrue(widget._progressive_job.wait(lambda state, phase: phase == "failed", 10))
        widget.grab()
        self.assertFalse(widget.render_mode)
        self.assertIn("no particles", widget.status)
        self.assertEqual(widget.render_note, "")


if __name__ == "__main__":
    unittest.main()
