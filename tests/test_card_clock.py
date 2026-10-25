"""The first frames after a discrete card has rested (docs/PERFORMANCE.md, "The first frames after the card idles").

After about 3.5 s without work a discrete card lowers its clocks and its link to the processor, and reading a 33 MB picture back takes
84 ms instead of 12 until the link has climbed (0.65 s of traffic that keeps it busy). The first played frames of a play pressed after
a rest arrived 60 to 150 ms apart. `nodebased/cardclock.py` is the answer: the window draws the first frames at half size, each followed
by a copy to the processor that fills the rest of the frame's time, and goes back to the chosen size once a frame's readback is fast. Each
test here fails without the part it names; the numbers are in the docs."""
import types
import unittest
import unittest.mock
import uuid

import tests.isolation  # noqa: F401
from nodebased import cardclock, gpu3d
from tests.test_first_second import splat_commands


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Buffer:
    def __init__(self, size, usage):
        self.size, self.usage = size, usage


class Encoder:
    def __init__(self):
        self.copies = []

    def copy_buffer_to_buffer(self, source, source_offset, target, target_offset, size):
        self.copies.append(size)

    def finish(self):
        return self


class Device:
    def __init__(self):
        self.buffers = []
        self.submitted = []
        self.queue = types.SimpleNamespace(submit=lambda commands: self.submitted.extend(c.copies[0] for c in commands))

    def create_buffer(self, size, usage):
        self.buffers.append(Buffer(size, usage))
        return self.buffers[-1]

    def create_command_encoder(self):
        return Encoder()


def fake_state(device=None):
    usage = types.SimpleNamespace(COPY_SRC=1, COPY_DST=2, MAP_READ=4)
    return {"wgpu": types.SimpleNamespace(BufferUsage=usage), "device": device or Device(), "info": {"adapter_type": "DiscreteGPU"}}


class RampTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.addCleanup(lambda: cardclock._active.__setitem__(0, None))

    def ramp(self):
        return cardclock.Ramp(1 / 24, clock=self.clock)

    def feed(self, ramp, seconds, ticks=1, step=0.0417):
        for _ in range(ticks):
            self.clock.now += step
            ramp.observe(seconds)

    def test_the_link_is_up_when_the_last_two_frames_took_a_third_of_the_first_three(self):
        ramp = self.ramp()
        self.feed(ramp, 0.022, 3)
        self.clock.now += cardclock.MIN_SECONDS
        self.feed(ramp, 0.0116, 2)         # the middle stage of the link: 8 MB in 11.6 ms is still more than a third
        self.assertFalse(ramp.up)
        self.feed(ramp, 0.0039, 2)
        self.assertTrue(ramp.up)

    def test_a_fast_frame_before_the_minimum_time_is_not_enough(self):
        ramp = self.ramp()
        self.feed(ramp, 0.022, 3, step=0.01)
        self.feed(ramp, 0.004, 3, step=0.01)    # 0.06 s in
        self.assertFalse(ramp.up)

    def test_a_card_whose_link_never_gets_faster_counts_as_up_after_the_longest_time(self):
        ramp = self.ramp()
        self.feed(ramp, 0.022, 20, step=0.1)
        self.assertTrue(ramp.up)

    def test_a_single_slow_frame_amid_fast_ones_does_not_hold_it_up_for_good(self):
        ramp = self.ramp()
        self.feed(ramp, 0.022, 3)
        self.clock.now += cardclock.MIN_SECONDS
        self.feed(ramp, 0.004, 3)
        self.feed(ramp, 0.03)
        self.assertFalse(ramp.up)
        self.feed(ramp, 0.004, 2)
        self.assertTrue(ramp.up)


class AfterFrameTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.device = Device()
        self.state = fake_state(self.device)
        self.sleeps = []
        patch = unittest.mock.patch.object(cardclock.time, "sleep", self.sleeps.append)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(lambda: cardclock._active.__setitem__(0, None))

    def ramp(self):
        ramp = cardclock.Ramp(1 / 24, clock=self.clock)
        return ramp

    def frame(self, seconds, megabytes=8, step=0.0417):
        self.clock.now += step
        cardclock.after_frame(self.state, int(megabytes * 2 ** 20), seconds)

    def reach_up(self, ramp):
        for _ in range(3):
            self.frame(0.022)
        self.clock.now += cardclock.MIN_SECONDS
        for _ in range(2):
            self.frame(0.0039)
        self.assertTrue(ramp.up)

    def test_nothing_is_copied_when_no_ramp_is_running(self):
        self.frame(0.022)
        self.assertEqual(self.device.submitted, [])
        self.assertEqual(self.device.buffers, [])

    def test_the_extra_copy_fills_the_rest_of_the_frames_time_at_the_speed_the_readback_showed(self):
        self.ramp()
        self.frame(0.022)        # 8 MB in 22 ms: the idle link, 0.36 MB per ms; 0.8 of 41.7 ms is 33 ms, 11 of them are left
        (size,) = self.device.submitted
        self.assertAlmostEqual(size / 2 ** 20, (1 / 24 * cardclock.BALLAST_FILL - 0.022) * 1000 * 8 / 22, delta=0.1)
        self.assertGreater(size, cardclock.BALLAST_MIN_BYTES)

    def test_a_frame_that_fills_its_time_gets_no_extra_copy(self):
        self.ramp()
        self.frame(0.035)
        self.assertEqual(self.device.submitted, [])

    def test_the_extra_copy_is_never_more_than_the_buffer_the_ramp_keeps(self):
        self.ramp()
        self.frame(0.0039)       # the link at full speed: 2 MB per ms would be 70 MB
        self.assertEqual(self.device.submitted, [cardclock.BALLAST_MAX_BYTES])

    def test_the_buffers_are_made_once_and_kept_for_the_next_ramp(self):
        self.ramp()
        self.frame(0.022)
        self.frame(0.022)
        self.assertEqual(len(self.device.buffers), 2)
        self.assertIn("clock_ballast", self.state)
        cardclock._active[0] = None
        self.ramp()
        self.frame(0.022)
        self.assertEqual(len(self.device.buffers), 2)      # destroying a buffer stalled the device for 87 to 135 ms, so none is destroyed

    def test_a_failing_card_ends_the_ramp_without_an_error(self):
        self.ramp()
        self.device.queue = types.SimpleNamespace(submit=unittest.mock.Mock(side_effect=RuntimeError("lost")))
        self.frame(0.022)
        self.assertIsNone(cardclock._active[0])

    def test_the_last_frames_at_the_lower_size_are_held_in_growing_steps_before_the_switch(self):
        ramp = self.ramp()
        self.reach_up(ramp)
        self.assertFalse(ramp.switch)
        # three times the 3.9 ms the lower size took, 12 ms, in steps of at most 10 ms: two frames
        self.assertEqual(ramp.hold_frames, 2)
        self.assertAlmostEqual(ramp.hold_step, 0.0059, places=3)
        self.sleeps.clear()
        self.frame(0.0039)
        self.assertFalse(ramp.switch)
        self.assertAlmostEqual(self.sleeps[-1], ramp.hold_step, places=4)
        self.frame(0.0039)
        self.assertTrue(ramp.switch)
        self.assertAlmostEqual(self.sleeps[-1], 2 * ramp.hold_step, places=4)

    def test_the_hold_counts_the_time_the_chosen_size_spends_packing_its_scene(self):
        ramp = self.ramp()
        cardclock.note_build(0.018)
        self.reach_up(ramp)
        self.assertEqual(ramp.hold_frames, 3)            # 11.7 + 18 ms in steps of at most 10 ms

    def test_a_hold_is_never_longer_than_the_cap(self):
        ramp = self.ramp()
        cardclock.note_build(0.5)
        self.reach_up(ramp)
        self.assertAlmostEqual(ramp.hold_step * ramp.hold_frames, cardclock.HOLD_MAX_SECONDS, places=6)

    def test_the_extra_copies_taper_off_over_two_seconds_after_the_switch_and_then_stop(self):
        ramp = self.ramp()
        self.reach_up(ramp)
        self.clock.now = ramp.up_at + 0.1
        before = len(self.device.submitted)
        self.frame(0.013, megabytes=33, step=0)
        full = self.device.submitted[before]
        self.clock.now = ramp.up_at + cardclock.TAPER_SECONDS / 2
        self.frame(0.013, megabytes=33, step=0)
        half = self.device.submitted[before + 1]
        self.assertLess(half, full)
        self.clock.now = ramp.up_at + cardclock.TAPER_SECONDS + 0.01
        count = len(self.device.submitted)
        self.frame(0.013, megabytes=33, step=0)
        self.assertEqual(len(self.device.submitted), count)
        self.assertIsNone(cardclock._active[0])

    def test_finish_stops_the_copies_of_a_play_that_was_stopped(self):
        ramp = self.ramp()
        ramp.finish()
        self.frame(0.022)
        self.assertEqual(self.device.submitted, [])


class StartRampTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(cardclock._last_busy.__setitem__, 0, cardclock._last_busy[0])
        self.addCleanup(lambda: cardclock._active.__setitem__(0, None))
        patch = unittest.mock.patch.dict(gpu3d._states, {"default": fake_state()}, clear=False)
        patch.start()
        self.addCleanup(patch.stop)

    def rested(self, seconds):
        cardclock._last_busy[0] = cardclock.time.monotonic() - seconds

    def test_a_discrete_card_that_has_rested_gets_a_ramp(self):
        self.rested(4.0)
        self.assertIsInstance(gpu3d.start_clock_ramp(1 / 24), cardclock.Ramp)

    def test_a_card_used_a_moment_ago_does_not(self):
        self.rested(0.5)
        self.assertIsNone(gpu3d.start_clock_ramp(1 / 24))

    def test_no_render_yet_means_no_ramp(self):
        cardclock._last_busy[0] = None
        self.assertIsNone(gpu3d.start_clock_ramp(1 / 24))

    def test_integrated_and_software_cards_do_not(self):
        self.rested(10.0)
        for kind in ("IntegratedGPU", "CPU", "VirtualGPU"):
            with self.subTest(kind=kind):
                gpu3d._states["default"]["info"] = {"adapter_type": kind}
                self.assertIsNone(gpu3d.start_clock_ramp(1 / 24))

    def test_a_render_that_waited_on_the_card_marks_it_used(self):
        self.rested(10.0)
        cardclock.note_busy()
        self.assertLess(cardclock.rest_seconds(), 1.0)


class WindowRampTests(unittest.TestCase):
    """`Window.toggle_playback` draws the first frames at half size when the card has rested, `_end_clock_ramp` puts the chosen tier back
    without a new request (a request cancels the frame in flight and costs the picture it was drawing: 135 ms at the switch), and
    a stop gives the tier back."""

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        global APP
        APP = QApplication.instance() or QApplication([])
        from nodebased.app import Window, CLOCK_RAMP_TIER
        self.tier = CLOCK_RAMP_TIER
        self.window = Window(agent_name="card-clock-" + uuid.uuid4().hex)
        self.window.agent_command({"op": "batch", "commands": splat_commands()})
        self.window.agent_command({"op": "view", "id": "render"})
        for name in ("start_preview",):
            held = unittest.mock.patch.object(self.window, name)
            held.start()
            self.addCleanup(held.stop)
        self.addCleanup(self.close)
        self.window.proxy.setCurrentIndex(self.window.proxy.findData(1))
        self.window.playback_proxy.setChecked(True)
        self.ramp = cardclock.Ramp(1 / 24)
        self.addCleanup(lambda: cardclock._active.__setitem__(0, None))

    def close(self):
        if self.window.playing:
            self.window.toggle_playback(False)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()

    def play(self, ramp):
        with unittest.mock.patch.object(gpu3d, "start_clock_ramp", return_value=ramp) as start:
            self.window.toggle_playback(True)
        return start

    def test_play_after_a_rest_draws_half_size_and_asks_the_card_for_a_ramp_at_the_comps_rate(self):
        start = self.play(self.ramp)
        start.assert_called_once_with(1 / 24, self.tier)
        self.assertEqual(self.window.proxy.currentData(), 2)
        self.assertIsNotNone(self.window.playback_auto_proxy_index)
        self.assertIs(self.window._clock_ramp, self.ramp)

    def test_play_on_a_warm_card_changes_nothing(self):
        self.play(None)
        self.assertEqual(self.window.proxy.currentData(), 1)
        self.assertIsNone(self.window.playback_auto_proxy_index)

    def test_proxy_while_playing_off_means_the_chosen_size_from_the_first_frame(self):
        self.window.playback_proxy.setChecked(False)
        start = self.play(self.ramp)
        start.assert_not_called()
        self.assertEqual(self.window.proxy.currentData(), 1)

    def test_a_tier_the_artist_chose_is_left_alone(self):
        self.window.proxy.setCurrentIndex(self.window.proxy.findData(4))
        start = self.play(self.ramp)
        start.assert_not_called()
        self.assertEqual(self.window.proxy.currentData(), 4)

    def test_a_comp_without_a_3d_render_has_no_ramp(self):
        self.window.agent_command({"op": "batch", "commands": [{"op": "create", "type": "Constant", "id": "flat", "params": {}}]})
        self.window.agent_command({"op": "view", "id": "flat"})
        self.window.dispatcher.document["nodes"] = {k: v for k, v in self.window.dispatcher.document["nodes"].items() if v["type"] != "Render3D"}
        start = self.play(self.ramp)
        start.assert_not_called()

    def test_the_switch_puts_the_chosen_tier_back_without_a_new_request(self):
        self.play(self.ramp)
        with unittest.mock.patch.object(self.window, "request_preview") as request:
            self.window._end_clock_ramp()
        request.assert_not_called()
        self.assertEqual(self.window.proxy.currentData(), 1)
        self.assertIsNone(self.window.playback_auto_proxy_index)
        self.assertIsNone(self.window._clock_ramp)

    def test_the_automatic_tier_leaves_the_lowered_frames_alone(self):
        self.play(self.ramp)
        request = types.SimpleNamespace(tier=2)
        self.assertFalse(self.window._adapt_playback_proxy(request, 500.0))
        self.assertEqual(self.window.proxy.currentData(), 2)

    def test_stopping_during_the_ramp_gives_the_tier_back_and_ends_the_copies(self):
        self.play(self.ramp)
        self.window.toggle_playback(False)
        self.assertEqual(self.window.proxy.currentData(), 1)
        self.assertIsNone(self.window._clock_ramp)
        self.assertIsNone(cardclock._active[0])

    def test_the_tick_makes_the_switch_when_the_ramp_says_so(self):
        self.play(self.ramp)
        self.ramp.switch = True
        self.window.playback_tick()
        self.assertEqual(self.window.proxy.currentData(), 1)


if __name__ == "__main__":
    unittest.main()
