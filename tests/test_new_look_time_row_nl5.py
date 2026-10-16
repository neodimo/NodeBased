"""New look, step 5 (Lane 2), part 3: the slim time row and the timeline track (transport buttons, the
frame field, the cached range in green, key diamonds, the playhead tab, the real-time light)."""
import time
import unittest

from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased.theme import TOKENS
from nodebased.timeline import TRACK_COLOR, TimelineBar
from tests.test_new_look_viewer_strip_nl5 import Base
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class TimeRowTests(Base):
    def test_the_transport_has_an_accent_play_button_and_steps(self):
        w = self.open_window()
        self.assertIsNotNone(w.step_back_button)
        w.set_time(first=1, last=48, current=10, fps=24.0)
        w.step_back_button.click()
        self.assertEqual(w.dispatcher.document["time"]["current"], 9)
        w.step_forward_button.click()
        w.step_forward_button.click()
        self.assertEqual(w.dispatcher.document["time"]["current"], 11)
        self.assertEqual(w.play_button.text(), "▶")
        w.play_button.click()
        self.assertTrue(w.playing)
        self.assertEqual(w.play_button.text(), "■")
        w.play_button.click()
        self.assertFalse(w.playing)
        self.assertEqual(w.play_button.text(), "▶")

    def test_the_frame_field_is_monospace_and_the_largest_field(self):
        w = self.open_window()
        self.assertEqual(w.frame_current.objectName(), "frame-current")
        self.assertGreater(w.frame_current.width(), w.frame_fps.width())
        self.assertGreater(w.frame_current.width(), w.frame_first.width())
        self.assertIn("Mono", w.frame_current.styleSheet() + QApplication.instance().styleSheet())

    def test_the_real_time_light_turns_warm_when_frames_are_dropped(self):
        w = self.open_window()
        self.assertEqual(w.realtime_label.text(), "real time")
        self.assertEqual(w.realtime_dot.color().name(), TOKENS["acc"])
        w.playing = True
        w.playback_origin_time = time.monotonic()   # playback just started: no rate to judge yet
        w.playback_dropped_frames = 3
        w.update_realtime_light()
        self.assertEqual(w.realtime_label.text(), "dropping frames")
        self.assertEqual(w.realtime_dot.color().name(), TOKENS["warn"])
        w.playing = False
        w.update_realtime_light()
        self.assertEqual(w.realtime_label.text(), "real time")
        self.assertEqual(w.realtime_dot.color().name(), TOKENS["acc"])

    def test_the_light_shows_the_achieved_rate_and_turns_warm_below_the_comp_rate(self):
        import time
        w = self.open_window()
        w.set_time(fps=24.0)
        w.playing = True
        w.playback_origin_time = time.monotonic() - 1.0
        now = time.monotonic()
        w.playback_shown_times.clear()
        w.playback_shown_times.extend(now - 0.95 + i * 0.0865 for i in range(11))   # 11 fps
        self.assertAlmostEqual(w.achieved_fps(), 11, delta=1.5)
        w.update_realtime_light()
        self.assertIn("of 24 fps", w.realtime_label.text())
        self.assertTrue(w.realtime_label.text().startswith("11"))
        self.assertEqual(w.realtime_dot.color().name(), TOKENS["warn"])
        w.playback_shown_times.clear()
        w.playback_shown_times.extend(now - 1.0 + i * (1.0 / 24) for i in range(1, 25))   # 24 fps
        w.update_realtime_light()
        self.assertEqual(w.realtime_label.text(), "real time · 24 fps")
        self.assertEqual(w.realtime_dot.color().name(), TOKENS["acc"])
        w.playing = False
        w.update_realtime_light()
        self.assertEqual(w.realtime_label.text(), "real time")

    def test_no_rate_is_claimed_before_half_a_second_of_playback(self):
        import time
        w = self.open_window()
        w.playing = True
        w.playback_origin_time = time.monotonic() - 0.2
        w.playback_shown_times.append(time.monotonic())
        self.assertIsNone(w.achieved_fps())
        w.playing = False

    def test_the_row_drops_status_text_before_it_clips_a_field(self):
        w = self.open_window((1280, 720), scale=1.25)
        row = w.time_row
        self.assertLessEqual(row.layout().minimumSize().width(), row.width())


class CachedRangeTests(unittest.TestCase):
    def band_color(self, bar, frame):
        image = bar.grab().toImage()
        rect = bar.cache_band_rect()
        x = int(bar.frame_x(frame) + bar.frame_width() / 2)
        return image.pixelColor(x, int(rect.center().y()))

    def test_the_green_range_follows_the_cache(self):
        bar = TimelineBar()
        bar.resize(800, 40)
        bar.setRange(1, 100)
        bar.show()
        APP.processEvents()
        self.assertEqual(self.band_color(bar, 20).name(), TRACK_COLOR.name(), "nothing cached: the quiet track")
        bar.set_marks(set(range(1, 51)), set())
        cached = self.band_color(bar, 20)
        self.assertEqual(cached.name(), TOKENS["acc2"], "a cached frame is the accent line")
        self.assertEqual(self.band_color(bar, 80).name(), TRACK_COLOR.name(), "frame 80 is not cached")
        bar.set_marks(set(range(60, 101)), set())
        self.assertEqual(self.band_color(bar, 20).name(), TRACK_COLOR.name(), "eviction clears the green")
        self.assertEqual(self.band_color(bar, 80).name(), TOKENS["acc2"])
        bar.close()

    def test_a_key_is_a_tick_in_the_key_colour_and_the_playhead_is_a_thin_accent_line(self):
        from nodebased.timeline import KEY_COLOR
        bar = TimelineBar()
        bar.resize(800, 40)
        bar.setRange(1, 100)
        bar.setValue(70)
        bar.set_marks(set(), {20})
        bar.show()
        APP.processEvents()
        image = bar.grab().toImage()
        centre = bar.key_mark_rect(20).center()
        self.assertEqual(image.pixelColor(int(centre.x()), int(centre.y())).name(), KEY_COLOR.name())
        line_x = round(bar.frame_x(70) + min(bar.frame_width(), 6.0) / 2)
        self.assertEqual(image.pixelColor(line_x, bar.height() // 2).name(), TOKENS["acc"], "the playhead line")
        self.assertNotEqual(image.pixelColor(line_x + 4, bar.height() // 2).name(), TOKENS["acc"], "and it is thin")
        bar.close()

    def test_the_window_pushes_the_cache_into_the_bar(self):
        class Holder(Base):
            pass
        holder = Holder()
        w = holder.open_window()
        try:
            w.set_time(first=1, last=24, current=1, fps=24.0)
            wait_until(lambda: w.frame_slider.cached_frames)
            self.assertTrue(w.frame_slider.cached_frames, "the first cooked frame is reported as cached")
        finally:
            holder.tearDown()


if __name__ == "__main__":
    unittest.main()
