"""The timeline strip's reporting logic, and the cache query that feeds it.

These are the parts that decide *what* the strip claims, tested without a window: the
label/tick density ladder, run collapsing, and `DisplayCache.resident_frames`. The wiring into
the properties panel and the transport is covered in test_desktop.py, where a real window
exists to click.
"""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from nodebased.core import empty_document
from nodebased.playback import DisplayCache
from nodebased.timeline import (MIN_LABEL_SPACING_PX, MIN_TICK_SPACING_PX, STEP_LADDER,
                                TimelineBar, choose_step, contiguous_runs)

APP = QApplication.instance() or QApplication([])


class StepChoiceTests(unittest.TestCase):
    def test_step_is_the_finest_ladder_entry_that_still_clears_the_spacing(self):
        # 20 px per frame: every frame clears 46 px? No -- 1*20 < 46, 2*20 < 46, 5*20 = 100 does.
        self.assertEqual(choose_step(20.0, MIN_LABEL_SPACING_PX), 5)
        self.assertEqual(choose_step(50.0, MIN_LABEL_SPACING_PX), 1)
        self.assertEqual(choose_step(20.0, MIN_TICK_SPACING_PX), 1)

    def test_labels_never_collide_at_any_range(self):
        # The property that matters: whatever the comp length, chosen labels are at least
        # MIN_LABEL_SPACING_PX apart on an 800 px strip. A fixed step cannot promise this.
        for span in (1, 24, 100, 250, 1001, 20000):
            per_frame = 800.0 / span
            step = choose_step(per_frame, MIN_LABEL_SPACING_PX)
            if step != STEP_LADDER[-1]:
                self.assertGreaterEqual(step * per_frame, MIN_LABEL_SPACING_PX, span)

    def test_pathological_range_still_returns_a_drawable_step(self):
        self.assertEqual(choose_step(0.0, MIN_LABEL_SPACING_PX), STEP_LADDER[-1])


class RunTests(unittest.TestCase):
    def test_runs_collapse_and_gaps_split(self):
        self.assertEqual(contiguous_runs([]), [])
        self.assertEqual(contiguous_runs({5}), [(5, 5)])
        self.assertEqual(contiguous_runs({1, 2, 3, 7, 8, 20}), [(1, 3), (7, 8), (20, 20)])

    def test_unordered_input_collapses_the_same(self):
        self.assertEqual(contiguous_runs({8, 1, 3, 2, 7}), [(1, 3), (7, 8)])


class TimelineBarTests(unittest.TestCase):
    def setUp(self):
        self.bar = TimelineBar()
        self.bar.resize(800, 34)
        self.bar.setRange(1, 100)

    def test_keeps_the_qslider_surface_the_window_was_built_against(self):
        self.assertEqual((self.bar.minimum(), self.bar.maximum()), (1, 100))
        seen = []
        self.bar.valueChanged.connect(seen.append)
        self.bar.setValue(42)
        self.assertEqual(self.bar.value(), 42)
        self.assertEqual(seen, [42])
        # No signal when nothing moved: the window syncs this widget from the document on every
        # frame change, and an echo there would fight the transport.
        self.bar.setValue(42)
        self.assertEqual(seen, [42])

    def test_value_is_clamped_into_the_range_like_a_slider(self):
        self.bar.setValue(9999)
        self.assertEqual(self.bar.value(), 100)
        self.bar.setRange(1001, 1003)
        self.assertEqual(self.bar.value(), 1001)

    def test_click_position_maps_to_the_frame_under_the_cursor(self):
        # Round trip through the geometry both ways rather than asserting one magic pixel.
        for frame in (1, 50, 100):
            self.assertEqual(self.bar.frame_at(self.bar.frame_x(frame) + 1), frame)

    def test_set_marks_repaints_only_on_an_actual_change(self):
        self.bar.set_marks({1, 2}, {5})
        self.assertEqual((self.bar.cached_frames, self.bar.key_frames), ({1, 2}, {5}))
        # Re-pushing the same sets is the common case during playback on a static graph.
        painted = []
        self.bar.update = lambda *a: painted.append(1)
        self.bar.set_marks({1, 2}, {5})
        self.assertEqual(painted, [])
        self.bar.set_marks({1, 2, 3}, {5})
        self.assertEqual(len(painted), 1)

    def test_paints_without_error_across_ranges_and_marks(self):
        # A paint that throws takes the whole window down, and the band/label code has enough
        # integer geometry in it to be worth actually exercising.
        from PySide6.QtGui import QPixmap
        for first, last in ((1, 1), (1, 24), (1, 100), (1001, 3000), (1, 20000)):
            self.bar.setRange(first, last)
            self.bar.set_marks(set(range(first, min(first + 40, last) + 1)), {first, last})
            pixmap = QPixmap(self.bar.size())
            self.bar.render(pixmap)


class ResidentFrameTests(unittest.TestCase):
    """`resident_frames` is what makes the orange band exact instead of bookkept."""

    def setUp(self):
        self.document = empty_document()
        self.cache = DisplayCache(budget_bytes=1 << 20)
        self.identity = DisplayCache.identity(self.document, 'viewer', 1, 'sRGB', 0.0, 'RGB',
                                              'checker')

    def store(self, frame):
        self.cache.put((self.identity, frame), b'\x00' * 64, 4, 4, 16)

    def test_reports_exactly_what_is_resident(self):
        self.store(3)
        self.store(4)
        self.assertEqual(self.cache.resident_frames(self.identity, range(1, 11)), {3, 4})

    def test_identity_ignores_the_playhead_so_prefetch_and_replay_share_a_key(self):
        # This is the read-ahead bug that made every prefetched frame a guaranteed miss: the
        # document snapshot a prefetch is built from still carries the *playing* frame in
        # time.current. Hashing that stale value gave the same frame two different keys.
        ahead = dict(self.document)
        ahead['time'] = {**self.document['time'], 'current': 5}
        later = dict(self.document)
        later['time'] = {**self.document['time'], 'current': 6}
        args = ('viewer', 1, 'sRGB', 0.0, 'RGB', 'checker')
        self.assertEqual(DisplayCache.identity(ahead, *args), DisplayCache.identity(later, *args))

    def test_a_graph_edit_changes_the_identity_so_the_band_clears_without_a_hook(self):
        self.store(3)
        edited = {**self.document, 'nodes': {'n1': {'type': 'Constant'}}}
        other = DisplayCache.identity(edited, 'viewer', 1, 'sRGB', 0.0, 'RGB', 'checker')
        self.assertNotEqual(other, self.identity)
        self.assertEqual(self.cache.resident_frames(other, range(1, 11)), set())

    def test_display_settings_are_part_of_the_identity(self):
        self.store(3)
        for changed in (('viewer', 2, 'sRGB', 0.0, 'RGB', 'checker'),
                        ('viewer', 1, 'ACES 2.0', 0.0, 'RGB', 'checker'),
                        ('viewer', 1, 'sRGB', 1.0, 'RGB', 'checker'),
                        ('viewer', 1, 'sRGB', 0.0, 'A', 'checker'),
                        ('viewer', 1, 'sRGB', 0.0, 'RGB', 'black')):
            identity = DisplayCache.identity(self.document, *changed)
            self.assertEqual(self.cache.resident_frames(identity, range(1, 11)), set(), changed)

    def test_eviction_stops_being_reported_immediately(self):
        small = DisplayCache(budget_bytes=200)
        for frame in range(1, 6):
            small.put((self.identity, frame), b'\x00' * 100, 5, 5, 20)
        resident = small.resident_frames(self.identity, range(1, 6))
        self.assertLess(len(resident), 5)
        self.assertIn(5, resident)


if __name__ == '__main__':
    unittest.main()
