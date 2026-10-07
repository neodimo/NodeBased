"""Lane L2 (2D parity) plan 21, step Y2: new nodes land where the artist is looking (real-display QA
10/6, finding 2). A new node goes under the selected one, moves right past anything already there,
and the graph pans so it and its upstream neighbour are fully visible; F and Home frame."""
import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased import app as nodebased_app_module
from nodebased.app import NODE_GAP, Window
from nodebased.core import atomic_save, demo_document
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class PlacementTests(unittest.TestCase):
    def open_window(self, size=(1440, 920)):
        original = nodebased_app_module.DEFAULT_WINDOW_SIZE
        nodebased_app_module.DEFAULT_WINDOW_SIZE = size
        try:
            self.window = Window()
        finally:
            nodebased_app_module.DEFAULT_WINDOW_SIZE = original
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        APP.processEvents()
        return self.window

    def open_empty_window(self, size=(1440, 920)):
        w = self.open_window(size)
        self.assertIsNotNone(w.command({"op": "batch", "commands": [
            {"op": "delete", "id": key} for key in list(w.graph_nodes())]}))
        APP.processEvents()
        w.graph.centerOn(0, 0)
        APP.processEvents()
        return w

    def tearDown(self):
        window = self.__dict__.pop("window", None)
        if window is not None:
            window.saved_document = window.dispatcher.document
            window.close()
            window.deleteLater()
            APP.processEvents()

    # -- helpers ---------------------------------------------------------------------------------
    def put(self, w, key, kind, pos, source=None):
        commands = [{"op": "create", "id": key, "type": kind, "pos": list(pos), "params": {}}]
        if source:
            commands.append({"op": "connect", "id": key, "input": "image", "source": source})
        self.assertIsNotNone(w.command({"op": "batch", "commands": commands}))

    def select(self, w, key):
        w.graph.scene().clearSelection()
        w.graph.items_by_id[key].setSelected(True)

    def press(self, w, key):
        w.graph.setFocus()
        QTest.keyClick(w.graph, key)
        APP.processEvents()

    def new_ids(self, w, before):
        return [key for key in w.graph.items_by_id if key not in before]

    def rect(self, w, key):
        return w.graph.items_by_id[key].sceneBoundingRect()

    def inside_view(self, w, key):
        return w.graph.visible_scene_rect().contains(self.rect(w, key))

    def pan_to(self, w, point):
        w.graph.centerOn(point)
        APP.processEvents()

    # -- placement -------------------------------------------------------------------------------
    def test_g_goes_directly_below_the_selected_node_at_standard_spacing(self):
        w = self.open_empty_window()
        self.put(w, "r1", "Checker", (0, 0))
        self.select(w, "r1")
        before = set(w.graph.items_by_id)
        self.press(w, Qt.Key.Key_G)
        (new,) = self.new_ids(w, before)
        parent, child = self.rect(w, "r1"), self.rect(w, new)
        self.assertAlmostEqual(child.top() - parent.bottom(), NODE_GAP, delta=1)
        self.assertAlmostEqual(child.center().x(), parent.center().x(), delta=1)

    def test_second_g_on_the_same_parent_stands_beside_the_first_child(self):
        w = self.open_empty_window()
        self.put(w, "r1", "Checker", (0, 0))
        self.select(w, "r1")
        before = set(w.graph.items_by_id)
        self.press(w, Qt.Key.Key_G)
        (first,) = self.new_ids(w, before)
        self.select(w, "r1")
        before = set(w.graph.items_by_id)
        self.press(w, Qt.Key.Key_G)
        (second,) = self.new_ids(w, before)
        a, b = self.rect(w, first), self.rect(w, second)
        self.assertFalse(a.intersects(b))
        self.assertGreaterEqual(b.left() - a.right(), NODE_GAP - 1, "right of the first child, with space")
        self.assertAlmostEqual(b.top(), a.top(), delta=1)

    def test_nothing_selected_lands_at_the_centre_of_the_visible_graph(self):
        w = self.open_empty_window()
        self.pan_to(w, QPointF(1800, 1400))
        w.graph.scene().clearSelection()
        before = set(w.graph.items_by_id)
        self.press(w, Qt.Key.Key_G)
        (new,) = self.new_ids(w, before)
        centre = w.graph_center()
        self.assertLess((self.rect(w, new).center() - centre).manhattanLength(), 60)
        self.assertTrue(self.inside_view(w, new))

    def test_the_node_dock_and_add_node_use_the_visible_centre_too(self):
        w = self.open_empty_window()
        self.pan_to(w, QPointF(-2200, 900))
        w.graph.scene().clearSelection()
        before = set(w.graph.items_by_id)
        w.add_node("Blur")
        (new,) = self.new_ids(w, before)
        self.assertLess((self.rect(w, new).center() - w.graph_center()).manhattanLength(), 60)

    def test_tab_menu_choice_with_a_selection_goes_below_it(self):
        w = self.open_empty_window()
        self.put(w, "r1", "Checker", (0, 0))
        self.select(w, "r1")
        before = set(w.graph.items_by_id)
        w.add_node("Blur", position=QPointF(500, 500))  # the Tab search passes the pointer position
        (new,) = self.new_ids(w, before)
        self.assertAlmostEqual(self.rect(w, new).top() - self.rect(w, "r1").bottom(), NODE_GAP, delta=1)

    # -- visibility ------------------------------------------------------------------------------
    def stack_down_to_the_edge(self, w):
        """A chain of nodes ending a little above the bottom edge of the visible graph."""
        view = w.graph.visible_scene_rect()
        self.put(w, "edge", "Checker", (view.center().x() - 95, view.bottom() - 200))
        self.select(w, "edge")
        return "edge"

    def test_g_near_the_bottom_edge_ends_fully_inside_the_viewport(self):
        w = self.open_empty_window()
        key = self.stack_down_to_the_edge(w)
        self.assertTrue(self.inside_view(w, key), "the parent starts in view")
        before = set(w.graph.items_by_id)
        self.press(w, Qt.Key.Key_G)
        (new,) = self.new_ids(w, before)
        self.assertTrue(self.inside_view(w, new))
        self.assertTrue(self.inside_view(w, key), "its upstream neighbour stays in view")
        view = w.graph.visible_scene_rect()
        self.assertGreater(view.bottom() - self.rect(w, new).bottom(), 20, "with a margin")

    def test_w_near_the_bottom_edge_ends_fully_inside_the_viewport(self):
        w = self.open_empty_window()
        key = self.stack_down_to_the_edge(w)
        before = set(w.graph.items_by_id)
        self.press(w, Qt.Key.Key_W)
        (new,) = self.new_ids(w, before)
        self.assertTrue(self.inside_view(w, new))
        self.assertTrue(self.inside_view(w, key))

    def test_the_pan_keeps_the_zoom(self):
        w = self.open_empty_window()
        key = self.stack_down_to_the_edge(w)
        zoom = w.graph.transform().m11()
        self.press(w, Qt.Key.Key_G)
        self.assertAlmostEqual(w.graph.transform().m11(), zoom, places=6)

    def test_zoom_out_only_when_node_and_upstream_cannot_both_fit(self):
        w = self.open_empty_window()
        w.graph.scale(4, 4)
        APP.processEvents()
        view = w.graph.visible_scene_rect()
        self.put(w, "up", "Checker", (view.left() + 5, view.top() + 5))
        self.select(w, "up")
        zoom = w.graph.transform().m11()
        # A node dropped by paste far from its source: both cannot be shown at this zoom.
        self.put(w, "far", "Grade", (view.left() + 5, view.bottom() + 3000), source="up")
        w.graph.reveal(["far"])
        self.assertLess(w.graph.transform().m11(), zoom)
        self.assertTrue(self.inside_view(w, "far"))
        self.assertTrue(self.inside_view(w, "up"))

    def test_paste_reveals_the_pasted_node(self):
        w = self.open_empty_window()
        view = w.graph.visible_scene_rect()
        self.put(w, "src", "Checker", (view.center().x(), view.bottom() - 70))
        w.graph._paste_nodes([{"type": "Grade", "params": {}, "pos": [view.center().x(), view.bottom() + 20]}])
        new = [key for key in w.graph.items_by_id if key != "src" and w.graph.items_by_id[key].isSelected()]
        self.assertEqual(len(new), 1)
        self.assertTrue(self.inside_view(w, new[0]))

    # -- framing ---------------------------------------------------------------------------------
    def test_f_frames_the_selection(self):
        w = self.open_empty_window()
        self.put(w, "a", "Checker", (-3000, -3000))
        self.put(w, "b", "Grade", (3000, 3000))
        self.select(w, "a")
        self.press(w, Qt.Key.Key_F)
        self.assertTrue(self.inside_view(w, "a"))
        self.assertFalse(self.inside_view(w, "b"))
        self.assertLessEqual(w.graph.transform().m11(), 1.25 + 1e-6, "one node is not magnified to fill the panel")

    def test_f_with_nothing_selected_frames_every_node(self):
        w = self.open_empty_window()
        self.put(w, "a", "Checker", (-3000, -3000))
        self.put(w, "b", "Grade", (3000, 3000))
        w.graph.scene().clearSelection()
        self.press(w, Qt.Key.Key_F)
        for key in w.graph.items_by_id:
            self.assertTrue(self.inside_view(w, key), key)

    def test_home_frames_everything_in_the_graph_and_leaves_the_playhead_alone(self):
        w = self.open_empty_window()
        w.set_time(first=10, last=40, current=20)
        self.put(w, "a", "Checker", (-3000, -3000))
        self.put(w, "b", "Grade", (3000, 3000))
        self.select(w, "a")
        self.press(w, Qt.Key.Key_Home)
        for key in w.graph.items_by_id:
            self.assertTrue(self.inside_view(w, key), key)
        self.assertEqual(w.dispatcher.document["time"]["current"], 20)

    def test_home_on_the_viewer_still_goes_to_the_first_frame(self):
        w = self.open_window()
        w.set_time(first=10, last=40, current=20)
        w.viewer.setFocus()
        QTest.keyClick(w.viewer, Qt.Key.Key_Home)
        self.assertEqual(w.dispatcher.document["time"]["current"], 10)

    # -- old documents ---------------------------------------------------------------------------
    def test_old_documents_keep_their_saved_node_positions(self):
        w = self.open_window()
        document = demo_document()
        document["nodes"]["plate"]["pos"] = [-3333.0, 777.0]
        saved = {key: list(node["pos"]) for key, node in document["nodes"].items()}
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "old.nbcomp")
            atomic_save(path, document)
            self.assertIsNotNone(w.command({"op": "load", "path": path}))
        APP.processEvents()
        for key, pos in saved.items():
            self.assertEqual(list(w.graph_nodes()[key]["pos"]), pos, key)
            item = w.graph.items_by_id[key].pos()
            self.assertEqual([item.x(), item.y()], pos, key)


if __name__ == "__main__":
    unittest.main()
