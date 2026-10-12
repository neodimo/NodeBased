"""New look, punch list 2 (DiMo's review of 10/11): a black viewer surround, an empty Properties
panel with nothing selected, the full transport, a calmer accent and one quiet playback unit."""
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton, QWidget

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased import theme, viewport3d
from nodebased.timeline import PLAYHEAD_WIDTH, TimelineBar, playhead_at
from nodebased.theme import TOKENS
from tests.test_new_look_viewer_strip_nl5 import Base

APP = QApplication.instance() or QApplication([])


def settle(n=6):
    for _ in range(n):
        APP.processEvents()


class ViewerSurroundTests(Base):
    def test_surround_is_pure_black_in_every_theme(self):
        w = self.open_window((1440, 920))
        for name in theme.THEMES:
            w.setStyleSheet(theme.build_style(name))
            settle()
            self.assertEqual(w.viewer.backgroundBrush().color().name(), "#000000", name)
        self.assertEqual(theme.VIEWER_SURROUND, "#000000")
        self.assertEqual(tuple(viewport3d.BACKGROUND[:3]), (0.0, 0.0, 0.0))

    def test_surround_pixels_are_black(self):
        w = self.open_window((1440, 920))
        settle()
        image = w.viewer.viewport().grab().toImage()
        self.assertEqual(image.pixelColor(2, image.height() // 2).name(), "#000000")


class EmptyPropertiesTests(Base):
    def test_no_text_in_the_panel_with_nothing_selected(self):
        from PySide6.QtWidgets import QAbstractButton, QLabel, QLineEdit, QTextEdit
        w = self.open_window((1440, 920))
        w.graph.scene().clearSelection()
        w.inspect(None)
        settle()
        panel = w.properties.widget()
        self.assertIsNotNone(panel)
        for label in panel.findChildren(QLabel):
            self.assertEqual(label.text(), "", label.objectName())
        for kind in (QAbstractButton, QLineEdit, QTextEdit):
            self.assertEqual(panel.findChildren(kind), [])
        self.assertIsNone(panel.findChild(QLabel, "properties-family-badge"))

    def test_selecting_a_node_brings_the_badge_and_name_back(self):
        from PySide6.QtWidgets import QLabel
        w = self.open_window((1440, 920))
        key = next(k for k, n in w.graph_nodes().items() if n["type"] == "Blur" or n["type"] == "Grade")
        w.inspect(key)
        settle()
        self.assertIsNotNone(w.properties.widget().findChild(QLabel, "properties-family-badge"))


TRANSPORT_ORDER = ["transport-start", "transport-prev-key", "transport-back", "transport-play-back",
                   "transport-stop", "transport-play", "transport-forward", "transport-next-key",
                   "transport-end", "transport-loop"]


class TransportTests(Base):
    def keyed_window(self):
        w = self.open_window((1440, 920))
        w.set_time(first=1, last=48, current=20, fps=24.0)
        w.dispatcher.execute({"op": "create", "id": "kg", "type": "Grade", "pos": [0, 0]})
        for frame in (10, 30, 40):
            w.dispatcher.execute({"op": "set_key", "id": "kg", "param": "exposure", "frame": frame, "value": 0.5})
        w.refresh_timeline_marks()
        self.assertEqual(w.keyed_frames, [10, 30, 40])
        return w

    def key(self, w, key, modifier=Qt.KeyboardModifier.NoModifier, target=None):
        w.activateWindow()
        QTest.keyClick(target or w, key, modifier)
        settle()

    def current(self, w):
        return w.dispatcher.document["time"]["current"]

    def test_the_transport_has_every_control_in_order_and_icon_only(self):
        from PySide6.QtWidgets import QToolButton
        w = self.open_window((1440, 920))
        transport = w.time_row.findChild(QWidget, "transport")
        buttons = [b for b in transport.findChildren(QToolButton)]
        self.assertEqual([b.objectName() for b in buttons], TRANSPORT_ORDER)
        edges = [b.mapTo(w.time_row, b.rect().topLeft()).x() for b in buttons]
        self.assertEqual(edges, sorted(edges))
        for button in buttons:
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonIconOnly)
            self.assertFalse(button.icon().isNull(), button.objectName())
            self.assertTrue(button.toolTip(), button.objectName())
            self.assertEqual(button.height(), 28)
        self.assertEqual(w.fps_presets.count() > 1, True)
        self.assertTrue(w.frame_fps.isVisible())

    def test_each_button_does_its_job(self):
        w = self.keyed_window()
        find = lambda name: w.time_row.findChild(QToolButton, name)
        find("transport-start").click()
        self.assertEqual(self.current(w), 1)
        find("transport-end").click()
        self.assertEqual(self.current(w), 48)
        find("transport-prev-key").click()
        self.assertEqual(self.current(w), 40)
        find("transport-prev-key").click()
        self.assertEqual(self.current(w), 30)
        find("transport-next-key").click()
        self.assertEqual(self.current(w), 40)
        find("transport-next-key").click()
        self.assertEqual(self.current(w), 40, "no keyframe after the last one: stays put")
        find("transport-back").click()
        self.assertEqual(self.current(w), 39)
        find("transport-forward").click()
        self.assertEqual(self.current(w), 40)
        find("transport-play-back").click()
        self.assertTrue(w.playing)
        self.assertEqual(w.play_direction, -1)
        find("transport-stop").click()
        self.assertFalse(w.playing)
        find("transport-play").click()
        self.assertTrue(w.playing)
        self.assertEqual(w.play_direction, 1)
        find("transport-play-back").click()
        self.assertEqual(w.play_direction, -1, "play backwards turns round")
        find("transport-play-back").click()
        self.assertFalse(w.playing, "pressing it again stops")

    def test_loop_button_cycles_once_loop_pingpong_and_remembers(self):
        w = self.open_window((1440, 920))
        w.set_loop_mode("loop")
        seen = []
        for _ in range(3):
            w.loop_button.click()
            seen.append(w.loop_mode)
        self.assertEqual(seen, ["pingpong", "once", "loop"])
        w.set_loop_mode("once")
        self.assertEqual(w.preferences.loop_mode(), "once")
        self.assertIn("Play once", w.loop_button.toolTip())

    def test_every_keyboard_shortcut_triggers_its_control(self):
        w = self.keyed_window()
        Alt = Qt.KeyboardModifier.AltModifier
        self.key(w, Qt.Key.Key_Left)
        self.assertEqual(self.current(w), 19)
        self.key(w, Qt.Key.Key_Right)
        self.assertEqual(self.current(w), 20)
        self.key(w, Qt.Key.Key_Left, Alt)
        self.assertEqual(self.current(w), 10)
        self.key(w, Qt.Key.Key_Right, Alt)
        self.assertEqual(self.current(w), 30)
        self.key(w, Qt.Key.Key_Home)
        self.assertEqual(self.current(w), 1)
        self.key(w, Qt.Key.Key_End)
        self.assertEqual(self.current(w), 48)
        self.key(w, Qt.Key.Key_Space)
        self.assertTrue(w.playing)
        self.key(w, Qt.Key.Key_Space)
        self.assertFalse(w.playing)
        viewer = w.viewer
        self.key(w, Qt.Key.Key_L, target=viewer)
        self.assertTrue(w.playing)
        self.assertEqual(w.play_direction, 1)
        self.key(w, Qt.Key.Key_K, target=viewer)
        self.assertFalse(w.playing)
        self.key(w, Qt.Key.Key_J, target=viewer)
        self.assertTrue(w.playing)
        self.assertEqual(w.play_direction, -1)
        self.key(w, Qt.Key.Key_K, target=viewer)
        self.assertFalse(w.playing)

    def test_every_transport_shortcut_is_in_the_help(self):
        from nodebased.app import SHORTCUT_SECTIONS
        text = " ".join(f"{k} {d}" for _, rows in SHORTCUT_SECTIONS for k, d in rows)
        for needle in ("Alt+Left", "Alt+Right", "previous keyframe", "next keyframe", "play backwards",
                       "play forwards", "stop", "first frame", "last frame", "play/stop"):
            self.assertIn(needle, text)


class PlayheadTests(unittest.TestCase):
    def test_loop_wraps_both_ways(self):
        self.assertEqual(playhead_at(1, 10, 9, 3, 1, "loop"), (2, False))
        self.assertEqual(playhead_at(1, 10, 2, 3, -1, "loop"), (9, False))

    def test_once_stops_at_the_end(self):
        self.assertEqual(playhead_at(1, 10, 8, 1, 1, "once"), (9, False))
        self.assertEqual(playhead_at(1, 10, 8, 5, 1, "once"), (10, True))
        self.assertEqual(playhead_at(1, 10, 3, 9, -1, "once"), (1, True))

    def test_pingpong_turns_at_both_ends(self):
        frames = [playhead_at(1, 4, 3, n, 1, "pingpong")[0] for n in range(0, 9)]
        self.assertEqual(frames, [3, 4, 3, 2, 1, 2, 3, 4, 3])
        back = [playhead_at(1, 4, 2, n, -1, "pingpong")[0] for n in range(0, 6)]
        self.assertEqual(back, [2, 1, 2, 3, 4, 3])


def _hsl(value):
    import colorsys
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s, l


def _contrast(a, b):
    def luminance(value):
        channels = [int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class AccentTests(unittest.TestCase):
    def test_the_default_accent_is_calmer_than_the_old_mint(self):
        tokens = theme.TOKENS
        _, old_s, old_l = _hsl("#5ee0b5")
        _, s, l = _hsl(tokens["acc"])
        # DiMo 10/11 2:55 PM: a Nuke-like amber. Lightness stays in the band and under the old mint's; the hue
        # is the change, so saturation is not bounded.
        self.assertTrue(0.55 <= l <= 0.65, l)
        self.assertLessEqual(l, old_l)
        h, _, _ = _hsl(tokens["acc"])
        self.assertTrue(25 <= h <= 45, h)   # orange to amber, in degrees
        self.assertGreaterEqual(_contrast(tokens["acc"], tokens["acc_ink"]), 4.5)
        self.assertGreaterEqual(_contrast(tokens["acc2"], tokens["acc_ink"]), 3.0)
        # The washes follow the accent.
        rgb = ", ".join(str(int(tokens["acc"][i:i + 2], 16)) for i in (1, 3, 5))
        self.assertIn(rgb, tokens["accbg"])
        self.assertIn(rgb, tokens["acc_ring"])
        self.assertEqual(theme.theme_colors("NodeBased")["accent"], tokens["acc"])

    def test_the_old_mint_is_still_a_named_choice(self):
        self.assertEqual(theme.ACCENTS["Mint"], "#5ee0b5")
        self.assertEqual(theme.ACCENTS["Muted teal"], "#66b7ac")
        self.assertEqual(len(theme.ACCENTS), 12)
        self.assertEqual(len(set(theme.ACCENTS.values())), 12)
        for name, value in theme.ACCENTS.items():
            if value is not None:
                self.assertEqual(theme.valid_accent(value), value, name)
        self.assertIn("#5ee0b5", theme.build_style("NodeBased", theme.ACCENTS["Mint"]))

    def test_no_source_file_hard_codes_the_old_accent(self):
        import pathlib
        for path in pathlib.Path(theme.__file__).parent.glob("*.py"):
            text = path.read_text(encoding="utf-8")   # Windows CI decoded the sources as cp1252 and failed on a byte
            if path.name == "theme.py":
                continue
            self.assertNotIn("5ee0b5", text.lower(), path.name)


class PlaybackUnitTests(Base):
    def test_every_control_in_the_unit_is_28_px_and_on_the_8_px_grid(self):
        from PySide6.QtWidgets import QAbstractSpinBox, QComboBox
        w = self.open_window((1440, 920))
        settle()
        controls = [c for c in w.time_row.findChildren(QToolButton) + w.time_row.findChildren(QAbstractSpinBox)
                    if c.isVisible()]
        self.assertGreaterEqual(len(controls), 12)
        for control in controls:
            expected = 28 if isinstance(control, QToolButton) else 28
            self.assertLessEqual(abs(control.height() - expected), 2, control.objectName())
        strip = w.viewer_strip
        for control in strip.findChildren(QToolButton) + strip.findChildren(QAbstractSpinBox):
            if control.isVisible():
                self.assertEqual(control.height(), 28, control.objectName() or type(control).__name__)
        gaps = w.time_row.controls.spacing()
        self.assertEqual(gaps % 8, 0)
        self.assertEqual(w.time_row.controls.contentsMargins().left() % 8, 0)

    def test_the_unit_is_one_tone_with_no_separator_or_box(self):
        from PySide6.QtWidgets import QFrame
        w = self.open_window((1440, 920))
        settle()
        image = w.time_row.grab().toImage()
        ruler = w.frame_slider
        bar_y = ruler.mapTo(w.time_row, ruler.rect().topLeft()).y()
        tone = TOKENS["bg1"]
        # The ruler, the controls and the gaps between them all sit on the one tone (bg1).
        samples = [(w.time_row.width() - 4, bar_y + 18), (w.time_row.width() - 4, w.time_row.height() - 4),
                   (4, w.time_row.height() - 4)]
        for x, y in samples:
            self.assertEqual(image.pixelColor(x, y).name(), tone, (x, y))
        self.assertEqual([f for f in w.time_row.findChildren(QFrame) if f.isVisible() and f.frameShape() != QFrame.Shape.NoFrame
                          and not isinstance(f, TimelineBar)], [])
        strip_separators = w.viewer_strip.findChildren(QFrame, "strip-separator")
        self.assertTrue(strip_separators)
        self.assertEqual({f.width() for f in strip_separators}, {8})

    def test_the_ruler_has_thin_lines(self):
        bar = TimelineBar()
        bar.resize(800, 32)
        bar.setRange(1, 100)
        bar.setValue(50)
        bar.set_marks(set(range(1, 40)), {20, 70})
        bar.show()
        settle()
        self.assertLessEqual(bar.cache_band_rect().height(), 2)
        self.assertEqual(PLAYHEAD_WIDTH, 1)
        self.assertLessEqual(bar.key_mark_rect(20).width(), 2)
        image = bar.grab().toImage()
        mid = bar.height() // 2
        playhead_x = round(bar.frame_x(50) + min(bar.frame_width(), 6.0) / 2)
        self.assertEqual(image.pixelColor(playhead_x, mid).name(), TOKENS["acc"])
        self.assertNotEqual(image.pixelColor(playhead_x + 3, mid).name(), TOKENS["acc"])
        bar.close()


if __name__ == "__main__":
    unittest.main()
