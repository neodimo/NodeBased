"""Lane L2 (2D parity) plan 21, step Z2: nodes made from a shortcut are wired to the selection the way
Nuke does it (QA 10/7 finding 2). Filters and Roto take the selection as their main (bg) input and are
spliced in before its downstream links, Merge takes it as B, generators stand beside it unwired; the new
node is selected, shown in Properties, viewed when the selection was, and in view."""
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased import app as nodebased_app_module
from nodebased.app import NODE_GAP, Window, creation_slot
from nodebased.imaging import Evaluator
from tests.test_roto import square
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])

# (key, kind, input the selection must land in, whether the new node is spliced into the branch)
HINT_ROW = [
    (Qt.Key.Key_R, "Read", None, False),
    (Qt.Key.Key_G, "Grade", "image", True),
    (Qt.Key.Key_M, "Merge", "B", True),
    (Qt.Key.Key_T, "Transform", "image", True),
    (Qt.Key.Key_B, "Blur", "image", True),
    (Qt.Key.Key_C, "ColorCorrect", "image", True),
    (Qt.Key.Key_S, "Shuffle", "image", True),
    (Qt.Key.Key_O, "Roto", "bg", True),
    (Qt.Key.Key_P, "Premult", "image", True),
    (Qt.Key.Key_U, "Unpremult", "image", True),
    (Qt.Key.Key_W, "Write", "image", False),
]


class ShortcutWiringTests(unittest.TestCase):
    def open_chain(self):
        self.window = w = Window()
        w.show()
        self.assertTrue(wait_until(lambda: w.frame is not None))
        self.assertIsNotNone(w.command({"op": "batch", "commands": [
            {"op": "delete", "id": key} for key in list(w.graph_nodes())]}))
        self.assertIsNotNone(w.command({"op": "batch", "commands": [
            {"op": "create", "id": "read", "type": "Checker", "pos": [0, 0], "params": {"width": 64, "height": 48}},
            {"op": "create", "id": "grade", "type": "Grade", "pos": [0, 120], "params": {}},
            {"op": "create", "id": "viewer", "type": "Viewer", "pos": [0, 240], "params": {}},
            {"op": "connect", "id": "grade", "input": "image", "source": "read"},
            {"op": "connect", "id": "viewer", "input": "image", "source": "grade"},
            {"op": "view", "id": "grade"}]}))
        APP.processEvents()
        w.graph.centerOn(0, 120)
        APP.processEvents()
        return w

    def tearDown(self):
        window = self.__dict__.pop("window", None)
        if window is not None:
            window.saved_document = window.dispatcher.document
            window.close()
            window.deleteLater()
            APP.processEvents()

    def select(self, key):
        graph = self.window.graph
        graph.scene().clearSelection()
        graph.items_by_id[key].setSelected(True)

    def make(self, how, kind):
        """Create `kind` with the Grade selected; `how` is a Qt key (shortcut) or "tab"."""
        w = self.window
        self.select("grade")
        before = set(w.graph.items_by_id)
        with mock.patch.object(w, "browse_read"):
            if how == "tab":
                with mock.patch.object(nodebased_app_module.NodeSearch, "choose", return_value=kind):
                    w.node_search()
            else:
                w.graph.setFocus()
                QTest.keyClick(w.graph, how)
        APP.processEvents()
        (new,) = [key for key in w.graph.items_by_id if key not in before]
        return new

    def check(self, how, kind, slot, spliced):
        w = self.window
        new = self.make(how, kind)
        nodes = w.dispatcher.document["nodes"]
        self.assertEqual(nodes[new]["type"], kind)
        wired = {name: src for name, src in nodes[new]["inputs"].items() if src}
        self.assertEqual(wired, {slot: "grade"} if slot else {}, f"{kind} input wiring")
        # The Grade's own input is untouched; the Viewer follows an inline node and stays put otherwise.
        self.assertEqual(nodes["grade"]["inputs"]["image"], "read")
        self.assertEqual(nodes["viewer"]["inputs"]["image"], new if spliced else "grade", f"{kind} downstream")
        self.assertEqual(w.dispatcher.document["view"], new if spliced else "grade", f"{kind} viewed")
        # Selected, shown in Properties, in view.
        self.assertEqual(w.graph.selected_id(), new)
        header = w.properties.widget().findChild(QLineEdit, "node-name-header")
        self.assertEqual(header.text(), nodes[new]["name"])
        self.assertTrue(w.graph.visible_scene_rect().contains(w.graph.items_by_id[new].sceneBoundingRect()))
        # Wired nodes go under the selection; a node with no input to take it stands beside it.
        parent, child = w.graph.items_by_id["grade"].sceneBoundingRect(), w.graph.items_by_id[new].sceneBoundingRect()
        if slot:
            self.assertGreaterEqual(child.top() - parent.bottom(), NODE_GAP - 1)
        else:
            self.assertGreaterEqual(child.left() - parent.right(), NODE_GAP - 1)
            self.assertFalse(child.intersects(parent))

    def test_every_create_shortcut_is_wired_like_nuke(self):
        for key, kind, slot, spliced in HINT_ROW:
            with self.subTest(shortcut=kind):
                self.open_chain()
                self.check(key, kind, slot, spliced)
                self.tearDown()

    def test_tab_search_creation_follows_the_same_wiring(self):
        for _key, kind, slot, spliced in HINT_ROW:
            with self.subTest(tab=kind):
                self.open_chain()
                self.check("tab", kind, slot, spliced)
                self.tearDown()

    def test_the_table_covers_every_shortcut_in_the_hint_row(self):
        w = Window()
        try:
            from PySide6.QtWidgets import QLabel
            hint = w.findChild(QLabel, "graph-shortcuts-hint").text()
        finally:
            w.close()
            w.deleteLater()
        listed = hint.split("R/G/M/T/B/C/S/O/P/U/W create")
        self.assertEqual(len(listed), 2, "the hint row still lists the create shortcuts")
        self.assertEqual({chr(int(key)) for key, *_ in HINT_ROW}, set("RGMTBCSOPUW"))

    def test_creation_slot_rules(self):
        self.assertEqual(creation_slot("Merge"), "B")
        self.assertEqual(creation_slot("Roto"), "bg")
        self.assertEqual(creation_slot("Grade"), "image")
        self.assertIsNone(creation_slot("Read"))
        self.assertIsNone(creation_slot("Checker"))

    # -- Roto over its plate ---------------------------------------------------------------------
    def test_o_on_a_viewed_plate_then_a_shape_shows_the_plate_with_coverage_in_alpha(self):
        w = self.open_chain()
        self.select("read")
        w.command({"op": "view", "id": "read"})
        before = set(w.graph.items_by_id)
        w.graph.setFocus()
        QTest.keyClick(w.graph, Qt.Key.Key_O)
        APP.processEvents()
        (roto,) = [key for key in w.graph.items_by_id if key not in before]
        document = w.dispatcher.document
        self.assertEqual(document["nodes"][roto]["inputs"]["bg"], "read")
        self.assertEqual(document["view"], roto)
        self.assertIsNotNone(w.command({"op": "set_shapes", "id": roto, "shapes": [square(8, 8, 40, 30)]}))
        evaluator = Evaluator()
        plate = evaluator.evaluate_raster(document, "read").pixels
        out = evaluator.evaluate_raster(w.dispatcher.document, roto).pixels
        self.assertEqual(out.shape, plate.shape)
        np.testing.assert_array_equal(out[..., :3], plate[..., :3])
        coverage = out[..., 3]
        self.assertEqual(coverage[20, 20], 1.0)
        self.assertEqual(coverage[2, 2], 0.0)
        self.assertEqual(coverage[40, 60], 0.0)
        self.assertGreater(float(coverage.sum()), 0.0)

    def test_unconnected_roto_renders_exactly_as_before(self):
        """An old document, saved when Roto had no inputs at all, still loads and gives the same matte."""
        import copy
        import json
        import tempfile
        from pathlib import Path

        from nodebased import roto, shapes
        w = self.open_chain()
        self.assertIsNotNone(w.command({"op": "batch", "commands": [
            {"op": "create", "id": "roto", "type": "Roto", "pos": [200, 0], "params": {"width": 64, "height": 48}},
            {"op": "set_shapes", "id": "roto", "shapes": [square(8, 8, 40, 30)]}]}))
        document = copy.deepcopy(w.dispatcher.document)
        document["nodes"]["roto"]["inputs"] = {}   # the old shape of the node
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.nbcomp"
            path.write_text(json.dumps(document))
            self.assertIsNotNone(w.command({"op": "load", "path": str(path)}))
        loaded = w.dispatcher.document
        out = Evaluator().evaluate_raster(loaded, "roto").pixels
        expected = roto.rasterise(shapes.resolve_shapes(loaded["node_data"]["roto"], 1), 64, 48)
        np.testing.assert_array_equal(out, expected)
        self.assertEqual(out[20, 20, 3], 1.0)

if __name__ == "__main__":
    unittest.main()
