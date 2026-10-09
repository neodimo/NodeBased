"""Lane L2 (2D parity) plan 21, step AA2: new nodes land in the stream the way Nuke places them
(real-display QA 10/7, findings 2 and 10). An inserted node goes directly below the selection, the
nodes downstream of it move down to make room in one undoable step, no node or port is covered, the
view pans without zooming, and the timeline's key ticks stand at least half the track tall."""
import unittest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased.app import NODE_GAP, NodeItem
from nodebased.timeline import PLAYHEAD_WIDTH, TimelineBar
from tests import test_2d_parity_placement_y2 as y2
from tests.test_2d_parity_placement_y2 import APP
from tests.waiting import wait_until

KEYS = (Qt.Key.Key_T, Qt.Key.Key_G, Qt.Key.Key_B, Qt.Key.Key_C,
        Qt.Key.Key_P, Qt.Key.Key_U, Qt.Key.Key_M, Qt.Key.Key_W)


class StreamPlacementTests(unittest.TestCase):
    # The window helpers of the Y2 placement tests, without re-running its tests.
    for _name in ("open_window", "open_empty_window", "tearDown", "put", "select", "press",
                  "new_ids", "rect", "inside_view"):
        locals()[_name] = y2.PlacementTests.__dict__[_name]
    del _name

    def stream(self, w):
        self.put(w, "src", "Checker", (0, 0))
        self.put(w, "grade", "Grade", (0, 120), source="src")
        APP.processEvents()

    def covered_ports(self, w):
        bad = []
        for key, item in w.graph.items_by_id.items():
            ports = list(item.inputs.values()) + ([item.output] if getattr(item, "output", None) else [])
            ports += list(getattr(item, "outputs", {}).values())
            for port in ports:
                point = port.scenePos()
                for other_key, other in w.graph.items_by_id.items():
                    if other_key != key and not other.is_backdrop and other.sceneBoundingRect().contains(point):
                        bad.append((key, other_key))
        return bad

    def positions(self, w):
        return {k: tuple(v["pos"]) for k, v in w.graph_nodes().items()}

    def insert_all(self, w):
        w.graph.scene().clearSelection()
        self.select(w, "src")
        made = []
        for key in KEYS:
            before = set(w.graph.items_by_id)
            self.press(w, key)
            made.extend(self.new_ids(w, before))
        return made

    def test_eight_inserts_stack_top_to_bottom_without_overlap(self):
        w = self.open_empty_window()
        self.stream(w)
        zoom = w.graph.transform().m11()
        made = self.insert_all(w)
        self.assertEqual(len(made), 8)
        self.assertAlmostEqual(w.graph.transform().m11(), zoom, places=6)
        rects = {k: self.rect(w, k) for k in w.graph.items_by_id}
        keys = list(rects)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                self.assertFalse(rects[a].intersects(rects[b]), (a, b))
        self.assertEqual(self.covered_ports(w), [])
        nodes = w.graph_nodes()
        for key in made:
            for src in nodes[key]["inputs"].values():
                if isinstance(src, str) and src in rects:
                    self.assertGreater(rects[key].top(), rects[src].top(), (key, src))
        # The Grade that was already there ended up below every inserted stream node.
        stream = [k for k in made if nodes[k]["type"] != "Write"]
        for key in stream:
            self.assertGreater(rects["grade"].top(), rects[key].top())
        self.assertTrue(self.inside_view(w, made[-1]))

    def test_undo_of_an_insert_puts_the_moved_nodes_back(self):
        w = self.open_empty_window()
        self.stream(w)
        before = self.positions(w)
        self.select(w, "src")
        self.press(w, Qt.Key.Key_T)
        self.assertNotEqual(self.positions(w)["grade"], before["grade"])
        self.assertIsNotNone(w.command({"op": "undo"}))
        APP.processEvents()
        self.assertEqual(self.positions(w), before)
        for key, pos in before.items():
            item = w.graph.items_by_id[key]
            self.assertAlmostEqual(item.pos().y(), pos[1], delta=0.01)

    def test_undo_after_eight_inserts_restores_every_node(self):
        w = self.open_empty_window()
        self.stream(w)
        before = self.positions(w)
        self.insert_all(w)
        for _ in range(8):
            self.assertIsNotNone(w.command({"op": "undo"}))
        APP.processEvents()
        self.assertEqual(self.positions(w), before)

    def test_a_write_goes_below_and_beside_and_leaves_the_stream_alone(self):
        w = self.open_empty_window()
        self.stream(w)
        before = self.positions(w)
        self.select(w, "src")
        ids = set(w.graph.items_by_id)
        self.press(w, Qt.Key.Key_W)
        (new,) = self.new_ids(w, ids)
        parent, child = self.rect(w, "src"), self.rect(w, new)
        self.assertGreaterEqual(child.top() - parent.bottom(), NODE_GAP - 1)
        self.assertGreater(child.left(), parent.right() - child.width() / 2)
        self.assertEqual({k: v for k, v in self.positions(w).items() if k in before}, before)

    def test_a_source_node_still_lands_beside_the_selection(self):
        w = self.open_empty_window()
        self.put(w, "src", "Checker", (0, 0))
        self.select(w, "src")
        ids = set(w.graph.items_by_id)
        w.add_node("Constant")
        (new,) = self.new_ids(w, ids)
        self.assertGreaterEqual(self.rect(w, new).left() - self.rect(w, "src").right(), NODE_GAP - 1)
        self.assertAlmostEqual(self.rect(w, new).top(), self.rect(w, "src").top(), delta=1)


class KeyMarkTests(unittest.TestCase):
    def test_key_marks_are_diamonds_wider_than_the_playhead_inside_the_track(self):
        # New look, step 5: a key is a diamond between the frame numbers and the cached range,
        # not a bottom-standing tick. It must still read at a glance next to the playhead.
        bar = TimelineBar()
        bar.resize(600, 34)
        bar.setRange(1, 100)
        bar.set_marks([], [10, 50])
        rect = bar.key_mark_rect(10)
        self.assertGreater(bar.key_diamond_size(), PLAYHEAD_WIDTH * 2)
        self.assertGreater(rect.width(), PLAYHEAD_WIDTH)
        self.assertGreaterEqual(rect.top(), 0)
        self.assertLessEqual(rect.bottom(), bar.cache_band_rect().top())
        centre = rect.center().x()
        self.assertAlmostEqual(centre, bar.frame_x(10) + bar.frame_width() / 2, delta=1)


if __name__ == "__main__":
    unittest.main()
