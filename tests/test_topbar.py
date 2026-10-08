"""New look, step 1: the design tokens, the one stylesheet and the top bar (Lane 2, NL1)."""
import re
import unittest
import unittest.mock
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QAbstractButton, QMenu, QToolBar

from nodebased import theme, topbar
from nodebased.app import NodeSearch, Window
from tests.test_desktop import APP, release_window, wait_until

SIZES = ((1280, 720), (1440, 920))
NARROW = (800, 720)
PARTS = ("logo", "project", "tabs", "search", "gpu_pill", "menu_button", "update_button")


class ThemeTokenTests(unittest.TestCase):
    def test_the_surface_line_text_and_accent_tokens_match_the_mockup(self):
        t = theme.TOKENS
        self.assertEqual([t[k] for k in ("bg0", "bg1", "bg2", "bg3")],
                         ["#0b0d10", "#111418", "#161a1f", "#1c2127"])
        self.assertEqual((t["line"], t["line2"]), ("#232930", "#2c333b"))
        self.assertEqual([t[k] for k in ("tx0", "tx1", "tx2", "tx3")],
                         ["#e9edf1", "#b4bcc6", "#7d8793", "#56606b"])
        self.assertEqual(t["acc"], "#5ee0b5")

    def test_the_thirteen_family_colours_have_no_yellow(self):
        self.assertEqual(theme.FAMILY_COLORS, {
            "Image": "#a3acb7", "Draw": "#7cc8f2", "Time": "#ff8a65", "Channel": "#f2849e",
            "Color": "#34d399", "Filter": "#f39a4c", "Keyer": "#79d36b", "Merge": "#5b86f0",
            "Transform": "#a681f2", "3D": "#e8585f", "Particles": "#e676d6", "Fluids": "#3fd0d0",
            "Metadata": "#6c7682"})

    def test_the_type_scale_is_noto_sans_13_with_a_monospace_for_values(self):
        self.assertEqual(theme.TYPE["base"], 13)
        self.assertTrue(theme.TYPE["family"].startswith("'Noto Sans'"))
        self.assertIn("monospace", theme.TYPE["mono"])

    def test_radii_sit_between_nine_and_twelve_on_panels_and_controls(self):
        for name in ("control", "panel", "popup"):
            self.assertTrue(9 <= theme.RADIUS[name] <= 12, name)

    def test_the_default_theme_is_built_from_the_tokens(self):
        colors = theme.theme_colors()
        self.assertEqual(theme.DEFAULT_THEME, "NodeBased")
        self.assertEqual(colors["accent"], theme.TOKENS["acc"])
        self.assertEqual(colors["window"], theme.TOKENS["bg1"])
        style = theme.build_style()
        self.assertEqual(theme.STYLE, style)
        for token in ("bg2", "line", "acc", "acc_ink"):
            self.assertIn(theme.TOKENS[token], style)
        self.assertIn("Noto Sans", style)
        self.assertIn("1px solid", style)

    def test_every_theme_still_builds_a_stylesheet_with_the_top_bar_rules(self):
        for name in theme.THEMES:
            style = theme.build_style(name)
            self.assertIn("QPushButton#update", style, name)
            self.assertIn("QFrame#segmented", style, name)

    def test_the_chrome_widgets_carry_no_colour_literals_of_their_own(self):
        source = Path(topbar.__file__).read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{6}\b", source), [])


class TopBarTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        APP.setStyleSheet(theme.build_style(self.window.theme_name, self.window.accent_color))
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_the_stylesheet_is_applied_app_wide(self):
        w = self.window
        self.assertEqual(APP.styleSheet(), theme.build_style(w.theme_name, w.accent_color))
        self.assertEqual(w.update_button.palette().buttonText().color().name(), theme.TOKENS["acc_ink"])

    def test_the_bar_holds_logo_project_tabs_search_gpu_pill_menu_and_update_button(self):
        w, bar = self.window, self.window.topbar
        toolbar = w.workspace_toolbar
        self.assertIsInstance(toolbar, QToolBar)
        self.assertIs(toolbar.widgetForAction(toolbar.actions()[-1]), bar)
        for name in PARTS:
            self.assertTrue(bar.isAncestorOf(getattr(bar, name)), name)
        self.assertEqual(list(bar.tabs.buttons), ["Composite", "3D", "Simulate", "Render"])
        self.assertEqual(bar.tabs.current(), "Composite")
        self.assertIs(bar.update_button, w.update_button)
        self.assertEqual(w.update_button.text(), "Check for updates")
        self.assertEqual(w.update_button.objectName(), "update")

    def test_there_is_no_export_button_on_the_bar(self):
        bar = self.window.topbar
        texts = [b.text() for b in bar.findChildren(QAbstractButton)]
        self.assertFalse([t for t in texts if "export" in t.lower()], texts)

    def test_the_update_button_is_the_trailing_item_and_a_primary_button(self):
        bar = self.window.topbar
        xs = {name: getattr(bar, name).geometry().right() for name in PARTS}
        self.assertEqual(max(xs, key=xs.get), "update_button")
        self.assertGreater(bar.update_button.mapTo(bar, bar.update_button.rect().center()).x(), bar.width() // 2)
        self.assertIn("QPushButton#update {", theme.STYLE.replace("\n", " ").replace("{{", "{"))

    def test_the_search_box_opens_the_existing_search(self):
        w = self.window
        with unittest.mock.patch.object(NodeSearch, "choose", return_value="Grade") as choose, \
                unittest.mock.patch.object(w, "add_node") as add:
            w.topbar.search.click()
        choose.assert_called_once()
        add.assert_called_once()
        self.assertEqual(add.call_args.args[0], "Grade")

    def test_tab_in_the_graph_still_opens_the_search(self):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        w = self.window
        with unittest.mock.patch.object(type(w), "node_search") as search:
            w.graph.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier))
        search.assert_called_once()
        self.assertEqual(w.topbar.search.keycap.text(), "Tab")

    def test_the_gpu_pill_shows_the_adapter_and_the_last_frame_time(self):
        w = self.window
        self.assertIsNotNone(w.last_frame_ms, "a frame has been drawn")
        self.assertRegex(w.topbar.gpu_pill.text(), r"\d+ ms$")
        w.topbar.set_gpu("Radeon 8060S", 12)
        self.assertEqual(w.topbar.gpu_pill.text(), "Radeon 8060S · 12 ms")
        w.topbar.set_gpu("", None, gpu=False)
        self.assertEqual(w.topbar.gpu_pill.text(), "CPU")

    def test_the_gl_renderer_string_becomes_a_short_adapter_name(self):
        from nodebased.gpudisplay import _clean_renderer
        self.assertEqual(_clean_renderer("AMD Radeon Graphics (radeonsi, gfx1151, LLVM 20.1.8)"),
                         "AMD Radeon Graphics")
        self.assertEqual(_clean_renderer("NVIDIA GeForce RTX 3080 Ti/PCIe/SSE2"), "NVIDIA GeForce RTX 3080 Ti")
        self.assertEqual(_clean_renderer("llvmpipe (LLVM 20.1.8, 256 bits)"), "llvmpipe")
        self.assertEqual(_clean_renderer(""), "")

    def test_the_project_label_shows_project_and_shot_with_a_saved_dot(self):
        w, bar = self.window, self.window.topbar
        self.assertEqual(bar.project_name.text(), "Untitled")
        bar.set_project("hot_pour", "comp_v012", saved=True)
        self.assertEqual(bar.project_shot.text(), "/ comp_v012")
        self.assertEqual(bar.saved_dot.color().name(), theme.TOKENS["acc"])
        bar.set_project("hot_pour", "comp_v012", saved=False)
        self.assertNotEqual(bar.saved_dot.color().name(), theme.TOKENS["acc"])
        w.project_path = str(Path("shots") / "hot_pour" / "comp_v012.json")
        w.update_title()
        self.assertEqual((bar.project_name.text(), bar.project_shot.text()), ("hot_pour", "/ comp_v012"))

    def test_workspace_tabs_switch_the_viewer_and_follow_it(self):
        w, bar = self.window, self.window.topbar
        bar.tabs.buttons["3D"].click()
        self.assertEqual(w.viewer_mode(), "3d")
        bar.tabs.buttons["Composite"].click()
        self.assertEqual(w.viewer_mode(), "2d")
        w.set_viewer_mode("3d")
        self.assertEqual(bar.tabs.current(), "3D")
        w.set_viewer_mode("2d")
        self.assertEqual(bar.tabs.current(), "Composite")
        bar.tabs.buttons["Simulate"].click()
        self.assertEqual(w.viewer_mode(), "3d")
        self.assertTrue(w.slice_dock.isVisible() and w.cache_inspector_dock.isVisible())
        bar.tabs.buttons["Render"].click()
        self.assertEqual(w.viewer_mode(), "2d")

    def test_every_menu_stays_reachable_from_the_menu_button(self):
        w = self.window
        menu = w.topbar.menu_button.menu()
        self.assertIsInstance(menu, QMenu)
        titles = {a.text() for a in menu.actions() if a.menu() is not None}
        self.assertTrue({"File", "Edit", "Time", "Workspace", "Window", "Help"} <= titles, titles)
        quick = {a.text() for a in menu.actions()}
        self.assertTrue({"Open image", "Save project", "Export image", "3D viewport"} <= quick)

    def test_menu_shortcuts_still_work_with_the_menu_bar_hidden(self):
        w = self.window
        self.assertFalse(w.menuBar().isVisible())
        undo = next(a for m in w.menuBar().findChildren(QMenu) for a in m.actions() if a.text() == "Undo")
        self.assertIn(undo, w.actions())
        self.assertEqual(undo.shortcut().toString(), "Ctrl+Z")

    def _assert_bar_fits(self, width, height, scale=None):
        w = self.window
        base = w.font()
        if scale:
            font = QFont(base)
            font.setPixelSize(max(1, round(base.pixelSize() * scale)))
            w.setFont(font)
        try:
            w.resize(width, height)
            for _ in range(5):
                APP.processEvents()
            bar = w.topbar
            self.assertGreaterEqual(bar.width(), width - 2, "the bar spans the window")
            rects = []
            for name in PARTS:
                part = getattr(bar, name)
                if not part.isVisible():
                    self.assertIn(name, ("project",), f"{name} hidden at {width}x{height}")
                    continue
                rect = part.geometry()
                self.assertGreaterEqual(rect.left(), 0, f"{name} clipped left at {width}x{height}")
                self.assertLessEqual(rect.right(), bar.width(), f"{name} clipped right at {width}x{height}")
                self.assertGreaterEqual(rect.top(), 0, f"{name} clipped top at {width}x{height}")
                self.assertLessEqual(rect.bottom(), bar.height(), f"{name} clipped bottom at {width}x{height}")
                self.assertGreaterEqual(part.width(), part.minimumSizeHint().width(),
                                        f"{name} narrower than its text at {width}x{height}")
                rects.append((rect.left(), rect.right(), name))
            rects.sort()
            for (_, right, first), (left, _, second) in zip(rects, rects[1:]):
                self.assertLess(right, left, f"{first} overlaps {second} at {width}x{height}")
            for button in bar.tabs.buttons.values():
                self.assertGreaterEqual(button.width(), button.fontMetrics().horizontalAdvance(button.text()))
            self.assertGreaterEqual(w.update_button.width(),
                                    w.update_button.fontMetrics().horizontalAdvance(w.update_button.text()))
            self.assertLessEqual(bar.layout().minimumSize().width(), bar.width())
        finally:
            w.setFont(base)

    def test_nothing_clips_at_1280_by_720(self):
        self._assert_bar_fits(1280, 720)

    def test_nothing_clips_at_1440_by_920(self):
        self._assert_bar_fits(1440, 920)

    def test_nothing_clips_when_the_font_is_a_quarter_larger(self):
        for width, height in SIZES:
            self._assert_bar_fits(width, height, scale=1.25)

    def test_nothing_clips_at_the_narrowest_window(self):
        self._assert_bar_fits(*NARROW)

    def test_nothing_clips_with_a_wider_family_as_on_windows(self):
        # Windows fonts run wider than Linux's. The stylesheet fixes sizes per widget, so a window
        # font change does not widen the bar; a wider family does (Liberation Mono is ~19% wider).
        base = APP.styleSheet()
        wide = "QWidget#topbar *, QPushButton#update { font-family: 'Liberation Mono'; }"
        APP.setStyleSheet(base + wide)
        try:
            for width, height in SIZES:
                self._assert_bar_fits(width, height)
            self._assert_bar_fits(1280, 720, scale=1.25)
            # Windows' Segoe UI bar is about 1.4x the width of Noto Sans here, so the window's
            # narrowest size on Linux stands in for 1280 there.
            self._assert_bar_fits(*NARROW)
        finally:
            APP.setStyleSheet(base)

    def test_the_bar_sheds_search_then_shot_then_gpu_text_and_keeps_tabs_and_update_whole(self):
        w, bar = self.window, self.window.topbar
        bar.set_project("hot_pour_long_project_name", "comp_v012", saved=True)
        bar.set_gpu("AMD Radeon Graphics", 12)
        w.resize(1600, 800)
        for _ in range(5):
            APP.processEvents()
        self.assertTrue(bar.search.hint.isVisible() and bar.project_shot.isVisible())
        self.assertIn("AMD Radeon Graphics", bar.gpu_pill.text())
        order = []
        for width in range(1600, 500, -20):
            w.resize(width, 800)
            for _ in range(3):
                APP.processEvents()
            state = (not bar.search.hint.isVisible(), not bar.project_shot.isVisible(),
                     not bar.gpu_pill.label.isVisible())
            if not order or order[-1][0] != state:
                order.append((state, width))
            for button in bar.tabs.buttons.values():
                self.assertGreaterEqual(button.width(), button.fontMetrics().horizontalAdvance(button.text()), width)
            self.assertGreaterEqual(w.update_button.width(),
                                    w.update_button.fontMetrics().horizontalAdvance(w.update_button.text()), width)
        states = [state for state, _ in order]
        self.assertEqual(states[0], (False, False, False))
        self.assertEqual(states, sorted(states), "pieces return in the same order they leave")
        self.assertEqual(len(states), len(set(states)))


if __name__ == "__main__":
    unittest.main()
