"""New look, step 4: the left column of node families and the floating node panel (Lane 2, NL4)."""
import tests.isolation  # noqa: F401  (settings of the test run, not the developer's)

import unittest
import unittest.mock

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from nodebased import nodecatalog, nodeicons, nodeshelf, theme
from nodebased.app import NODE_KIND_MIME_TYPE, ProjectSettingsDialog, Window
from tests.test_desktop import APP, release_window, wait_until

KEYS = ("Favourites", "Recent", *nodecatalog.NODE_CATEGORIES, "Settings")


class RailTestCase(unittest.TestCase):
    size = (1280, 720)

    def setUp(self):
        self.window = Window()
        APP.setStyleSheet(theme.build_style(self.window.theme_name, self.window.accent_color))
        self.window.resize(*self.size)
        self.window.show()
        self.window.activateWindow()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.settle()
        self.rail, self.panel, self.shelf = self.window.node_rail, self.window.node_panel, self.window.node_shelf

    def tearDown(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def settle(self, rounds=6):
        for _ in range(rounds):
            APP.processEvents()

    def node_types(self):
        return sorted(node["type"] for node in self.window.dispatcher.document["nodes"].values())

    def count_of(self, kind):
        return self.node_types().count(kind)

    def click(self, widget):
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
        self.settle()


class ColumnTests(RailTestCase):
    def test_the_column_holds_favourites_recent_every_family_and_settings_in_that_order(self):
        self.assertEqual(list(self.rail.buttons), list(KEYS))
        ys = [self.rail.buttons[key].geometry().top() for key in KEYS]
        self.assertEqual(ys, sorted(ys))
        self.assertEqual(self.window.toolBarArea(self.window.node_rail_toolbar), Qt.ToolBarArea.LeftToolBarArea)
        self.assertEqual(self.window.node_rail_toolbar.parent(), self.window)

    def test_every_family_button_has_its_icon_and_its_colour_dot(self):
        for family in nodecatalog.NODE_CATEGORIES:
            button = self.rail.buttons[family]
            self.assertEqual(button.glyph, nodeicons.family_icon_name(family if family in theme.FAMILY_COLORS else "Other"))
            self.assertEqual(button.swatch, theme.FAMILY_COLORS.get(family))
            want = QColor(theme.FAMILY_COLORS[family]) if family in theme.FAMILY_COLORS else None
            if want is not None:
                image = button.grab().toImage()
                dots = [image.pixelColor(x, y) for x in range(image.width()) for y in range(image.height())
                        if abs(image.pixelColor(x, y).red() - want.red()) < 6
                        and abs(image.pixelColor(x, y).green() - want.green()) < 6
                        and abs(image.pixelColor(x, y).blue() - want.blue()) < 6]
                self.assertGreater(len(dots), 6, f"{family} shows no colour dot")
        for key in ("Favourites", "Recent", "Settings"):
            self.assertIsNone(self.rail.buttons[key].swatch)

    def test_the_column_fits_at_1280_by_720(self):
        self.assert_column_fits()

    def test_the_column_fits_at_the_smallest_window(self):
        self.window.resize(*nodebased_min_size())
        self.settle()
        self.assert_column_fits()

    def test_the_column_fits_when_the_text_is_wider(self):
        base = APP.styleSheet()
        APP.setStyleSheet(base + "QWidget { font-size: 17px; }")
        try:
            self.settle(10)
            self.assert_column_fits()
            self.assertGreater(self.rail.width(), 61)
        finally:
            APP.setStyleSheet(base)
            self.settle(10)

    def assert_column_fits(self):
        rail = self.rail
        self.assertEqual(rail.width(), rail.column_width())
        self.assertGreaterEqual(rail.height(), rail.minimumSizeHint().height())
        previous = -1
        for key in KEYS:
            rect = rail.buttons[key].geometry()
            self.assertGreaterEqual(rect.top(), 0, key)
            self.assertLessEqual(rect.bottom(), rail.height() - 1, key)
            self.assertGreaterEqual(rect.left(), 0, key)
            self.assertLessEqual(rect.right(), rail.width() - 1, key)
            self.assertGreaterEqual(rect.height(), rail.MIN_BUTTON, key)
            self.assertGreater(rect.top(), previous, f"{key} overlaps the button above")
            previous = rect.bottom()
        self.assertTrue(all(rail.buttons[k].isVisible() for k in KEYS))
        self.assertGreater(rail.buttons["Settings"].geometry().top(), rail.buttons["Other"].geometry().bottom())

    def test_the_active_family_shows_the_bar_on_the_edge_of_the_column(self):
        self.assertIsNone(self.rail.active())
        self.click(self.rail.buttons["Color"])
        self.assertEqual(self.rail.active(), "Color")
        self.assertTrue(self.rail.buttons["Color"].isChecked())
        self.assertFalse(self.rail.buttons["Draw"].isChecked())
        image = self.rail.grab().toImage()
        button = self.rail.buttons["Color"].geometry()
        want = QColor(theme.FAMILY_COLORS["Color"])
        got = image.pixelColor(1, button.center().y())
        self.assertEqual((got.red(), got.green(), got.blue()), (want.red(), want.green(), want.blue()))
        # and nothing beside the other buttons
        quiet = image.pixelColor(1, self.rail.buttons["Draw"].geometry().center().y())
        self.assertNotEqual((quiet.red(), quiet.green(), quiet.blue()), (want.red(), want.green(), want.blue()))
        self.shelf.close_panel()
        self.assertIsNone(self.rail.active())
        self.assertFalse(self.rail.buttons["Color"].isChecked())

    def test_settings_opens_the_settings_dialog(self):
        with unittest.mock.patch.object(ProjectSettingsDialog, "exec", return_value=QDialog.DialogCode.Rejected) as run:
            self.click(self.rail.buttons["Settings"])
        run.assert_called_once()


def nodebased_min_size():
    from nodebased.app import MIN_WINDOW_SIZE
    return MIN_WINDOW_SIZE


class PanelTests(RailTestCase):
    def open(self, family="Color"):
        self.click(self.rail.buttons[family])
        return self.panel

    def test_clicking_a_family_opens_the_panel_with_its_name_and_node_count(self):
        panel = self.open("Color")
        self.assertTrue(panel.isVisible())
        self.assertEqual(panel.title.text(), "Color")
        self.assertEqual(panel.count.text(), f"{len(nodecatalog.NODE_CATEGORIES['Color'])} nodes")
        self.assertEqual(panel.dot.color.name(), theme.FAMILY_COLORS["Color"])
        self.assertEqual(panel.filter.placeholderText(), "Filter Color…")
        self.assertIs(panel.parent(), self.window)
        self.assertGreater(panel.geometry().left(), self.rail.geometry().right())

    def test_the_filter_box_has_focus_on_open(self):
        self.open("Filter")
        self.assertIs(APP.focusWidget(), self.panel.filter)
        self.shelf.close_panel()
        self.open("Draw")
        self.assertIs(APP.focusWidget(), self.panel.filter)

    def test_the_panel_lists_nodes_two_to_a_row_with_the_common_ones_first(self):
        panel = self.open("Filter")
        kinds = panel.visible_kinds()
        self.assertCountEqual(kinds, list(nodecatalog.NODE_CATEGORIES["Filter"]))
        first, second = panel.list.visualItemRect(panel.list.item(0)), panel.list.visualItemRect(panel.list.item(1))
        third = panel.list.visualItemRect(panel.list.item(2))
        self.assertEqual(first.top(), second.top())
        self.assertGreater(second.left(), first.left())
        self.assertGreater(third.top(), first.top())
        self.assertEqual(third.left(), first.left())
        own = [nodeicons.has_own_icon(k) for k in kinds]
        self.assertEqual(own, sorted(own, reverse=True))

    def test_a_long_family_shows_twelve_and_counts_the_rest(self):
        panel = self.open("Color")
        total = len(nodecatalog.NODE_CATEGORIES["Color"])
        self.assertEqual(panel.more.text(), f"{total - 12} more")
        panel.more.clicked.emit()
        self.settle()
        self.assertGreater(panel.list.verticalScrollBar().value(), 0)
        short = self.open("Metadata")
        self.assertFalse(short.more.text().endswith("more"))

    def test_the_filter_narrows_the_list(self):
        panel = self.open("Filter")
        total = panel.list.count()
        QTest.keyClicks(panel.filter, "blur")
        shown = panel.visible_kinds()
        self.assertLess(len(shown), total)
        self.assertIn("Blur", shown)
        self.assertIn("DirBlur", shown)
        self.assertNotIn("Sharpen", shown)
        self.assertEqual(panel.count.text(), f"{len(shown)} of {total}")
        panel.filter.setText("")
        self.assertEqual(panel.list.count(), total)
        panel.filter.setText("zzzz")
        self.assertEqual(panel.list.count(), 0)
        self.assertEqual(panel.more.text(), "No nodes match")

    def test_the_filter_reads_descriptions_as_well_as_names(self):
        panel = self.open("Filter")
        description = nodecatalog.node_description("Median")
        word = max((w for w in description.lower().replace(".", "").split() if w.isalpha()), key=len)
        panel.filter.setText(word)
        self.assertIn("Median", panel.visible_kinds())

    def test_clicking_a_node_adds_it_and_closes_the_panel(self):
        panel = self.open("Color")
        item = next(panel.list.item(i) for i in range(panel.list.count()) if panel.list.item(i).text() == "Gamma")
        before = self.count_of("Gamma")
        QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton, pos=panel.list.visualItemRect(item).center())
        self.settle()
        self.assertEqual(self.count_of("Gamma"), before + 1)
        self.assertFalse(panel.isVisible())
        self.assertIsNone(self.rail.active())

    def test_enter_adds_the_highlighted_node(self):
        panel = self.open("Color")
        before = self.count_of("Grade")
        QTest.keyClick(panel.filter, Qt.Key.Key_Return)
        self.settle()
        self.assertEqual(self.count_of("Grade"), before + 1)       # the first node is highlighted on open
        self.assertFalse(panel.isVisible())

    def test_enter_adds_the_first_match_of_what_was_typed(self):
        panel = self.open("Filter")
        QTest.keyClicks(panel.filter, "defocus")
        before = self.count_of("Defocus")
        QTest.keyClick(panel.filter, Qt.Key.Key_Return)
        self.settle()
        self.assertEqual(self.count_of("Defocus"), before + 1)

    def test_the_arrow_keys_move_the_highlight_before_enter(self):
        panel = self.open("Color")
        self.assertEqual(panel.current_kind(), "Grade")
        QTest.keyClick(panel.filter, Qt.Key.Key_Down)
        self.assertEqual(panel.list.currentRow(), 2)
        QTest.keyClick(panel.filter, Qt.Key.Key_Right)
        self.assertEqual(panel.list.currentRow(), 3)
        QTest.keyClick(panel.filter, Qt.Key.Key_Up)
        self.assertEqual(panel.list.currentRow(), 1)
        kind = panel.current_kind()
        before = self.count_of(kind)
        QTest.keyClick(panel.filter, Qt.Key.Key_Return)
        self.assertEqual(self.count_of(kind), before + 1)

    def test_a_drag_into_the_graph_adds_the_node_where_it_is_dropped(self):
        panel = self.open("Filter")
        self.assertTrue(panel.list.dragEnabled())
        item = next(panel.list.item(i) for i in range(panel.list.count()) if panel.list.item(i).text() == "Blur")
        mime = panel.list.mimeData([item])
        self.assertEqual(bytes(mime.data(NODE_KIND_MIME_TYPE)).decode("utf-8"), "Blur")
        drop_point = QPointF(210, 140)
        expected = self.window.graph.mapToScene(drop_point.toPoint())
        before = set(self.window.dispatcher.document["nodes"])
        # a real drop event into the graph's viewport, carrying what the list's drag carries
        enter = QDragEnterEvent(drop_point.toPoint(), Qt.DropAction.CopyAction, mime,
                                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        APP.sendEvent(self.window.graph.viewport(), enter)
        self.assertTrue(enter.isAccepted())
        drop = QDropEvent(drop_point, Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        APP.sendEvent(self.window.graph.viewport(), drop)
        added = set(self.window.dispatcher.document["nodes"]) - before
        self.assertEqual(len(added), 1)
        node = self.window.dispatcher.document["nodes"][added.pop()]
        self.assertEqual(node["type"], "Blur")
        self.assertAlmostEqual(node["pos"][0], expected.x(), delta=1.0)
        self.assertAlmostEqual(node["pos"][1], expected.y(), delta=1.0)
        # the panel gives way when the drag ends
        panel.list.drag_finished.emit()
        self.assertFalse(panel.isVisible())

    def test_a_real_drag_carries_only_the_kind_name(self):
        panel = self.open("Color")
        mime = panel.list.mimeData([panel.list.item(0)])
        self.assertEqual(mime.formats(), [NODE_KIND_MIME_TYPE])
        self.assertEqual(panel.list.mimeData([]).formats(), [])

    def test_escape_closes_the_panel(self):
        panel = self.open("Color")
        QTest.keyClick(panel.filter, Qt.Key.Key_Escape)
        self.assertFalse(panel.isVisible())
        self.assertIsNone(self.rail.active())

    def test_a_click_outside_closes_the_panel_and_still_reaches_what_was_clicked(self):
        panel = self.open("Color")
        graph = self.window.graph.viewport()
        target = graph.rect().center() + graph.rect().bottomRight() / 4
        self.assertFalse(panel.geometry().contains(graph.mapTo(self.window, target)))
        QTest.mouseClick(graph, Qt.MouseButton.LeftButton, pos=target)
        self.settle()
        self.assertFalse(panel.isVisible())

    def test_a_click_inside_the_panel_keeps_it_open(self):
        panel = self.open("Color")
        QTest.mouseClick(panel.title, Qt.MouseButton.LeftButton)
        QTest.mouseClick(panel.filter, Qt.MouseButton.LeftButton)
        self.assertTrue(panel.isVisible())

    def test_the_same_family_button_closes_the_panel_and_another_switches_it(self):
        self.open("Color")
        self.click(self.rail.buttons["Color"])
        self.assertFalse(self.panel.isVisible())
        self.open("Color")
        self.click(self.rail.buttons["Draw"])
        self.assertTrue(self.panel.isVisible())
        self.assertEqual(self.panel.title.text(), "Draw")
        self.assertEqual(self.rail.active(), "Draw")
        self.assertFalse(self.rail.buttons["Color"].isChecked())

    def test_the_panel_stays_inside_the_window_beside_the_last_families(self):
        for family in ("Favourites", "Other", "Fluids"):
            panel = self.open(family)
            rect = panel.geometry()
            rail = self.rail.geometry().translated(self.rail.mapTo(self.window, self.rail.rect().topLeft()))
            self.assertGreaterEqual(rect.top(), rail.top(), family)
            self.assertLessEqual(rect.bottom(), rail.bottom(), family)
            self.assertLessEqual(rect.right(), self.window.width(), family)
            self.shelf.close_panel()

    def test_favourites_and_recent_list_the_nodes_the_artist_starred_and_added(self):
        self.window.preferences.set_favourite("Blur", True)
        panel = self.open("Favourites")
        self.assertEqual(panel.visible_kinds(), ["Blur"])
        self.assertEqual(panel.count.text(), "1 node")
        self.shelf.close_panel()
        self.window.add_node("Sharpen")
        panel = self.open("Recent")
        self.assertEqual(panel.visible_kinds()[0], "Sharpen")

    def test_resizing_the_window_closes_the_panel(self):
        panel = self.open("Color")
        self.window.resize(self.window.width() + 40, self.window.height())
        self.settle()
        self.assertFalse(panel.isVisible())

    def test_the_window_losing_focus_closes_the_panel(self):
        panel = self.open("Color")
        APP.sendEvent(self.window, QEvent(QEvent.Type.WindowDeactivate))
        self.assertFalse(panel.isVisible())

    def test_the_panel_follows_the_theme(self):
        self.open("Color")
        self.window.apply_theme_name("Charcoal", None)
        self.assertEqual(self.panel.delegate.colors["accent"], theme.theme_colors("Charcoal")["accent"])
        self.assertEqual(self.rail.colors["panel"], theme.theme_colors("Charcoal")["panel"])
        self.window.apply_theme_name("NodeBased", None)


class IconSetTests(unittest.TestCase):
    def test_the_column_and_panel_carry_no_colour_literals_of_their_own(self):
        import re
        from pathlib import Path
        source = Path(nodeshelf.__file__).read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{6}\b", source), [])

    def test_a_panel_row_wears_the_family_colour_of_its_own_node(self):
        for kind in ("Grade", "Blur", "Keyer", "Transform", "Dot", "Camera3D"):
            self.assertEqual(theme.family_color(kind), theme.FAMILY_COLORS[nodecatalog.node_family(kind)])

    def test_the_order_puts_nodes_with_their_own_glyph_first(self):
        ordered = nodeshelf.ordered_kinds(nodecatalog.NODE_CATEGORIES["Color"])
        self.assertCountEqual(ordered, list(nodecatalog.NODE_CATEGORIES["Color"]))
        flags = [nodeicons.has_own_icon(k) for k in ordered]
        self.assertEqual(flags, sorted(flags, reverse=True))
        self.assertEqual(ordered[0], "Grade")


if __name__ == "__main__":
    unittest.main()
