"""The rest of the Node Graph against Nuke 17.0's User Guide (lane 10, step T6, 2026-11-17): dragging pipe ends to connect and
disconnect, Shift+drag to duplicate a pipe, Ctrl+click for a Dot, Ctrl+drag to take the upstream nodes along, Select Connected
Nodes, Shift and Ctrl/Cmd in the Tab menu for a new branch and a replace, dropping a node into a wire, and snap to grid. Each probe
in tools/nuke_panel_interactions.py drives the built window with mouse and key events and checks the document's graph, the undo
step and the selection."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import sys
import unittest
from pathlib import Path

import numpy as np

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402

T6 = ("graph.shift_new_branch", "graph.ctrl_replace", "graph.insert_between", "graph.drag_pipe_disconnect",
      "graph.drag_pipe_connect", "graph.shift_drag_duplicate_pipe", "graph.ctrl_click_dot", "graph.ctrl_drag_upstream",
      "graph.select_connected", "graph.snap_to_grid")


class NodeGraphT6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = interactions.reference()
        mine = [e for e in ref["behaviours"] if e["id"] in T6]
        original = interactions.reference
        interactions.reference = lambda: dict(ref, behaviours=mine)
        try:
            cls.results = interactions.probe_all()
        finally:
            interactions.reference = original

    def test_rest_of_the_node_graph_works_like_nuke(self):
        for behaviour in T6:
            with self.subTest(behaviour=behaviour):
                self.assertEqual(self.results[behaviour]["result"], "pass", self.results[behaviour]["detail"])


class DotPassesItsInputOnTests(unittest.TestCase):
    def test_a_dot_hands_on_its_input_unchanged(self):
        from nodebased.imaging import Evaluator
        src = np.random.default_rng(6).random((12, 16, 4)).astype(np.float32)
        out = Evaluator._kernel("Dot", {}, [src])
        np.testing.assert_array_equal(np.asarray(out), src)


class SnapPreferenceTests(unittest.TestCase):
    def test_snap_to_grid_is_off_by_default_as_in_nuke(self):
        from nodebased.app import Preferences
        from PySide6.QtCore import QSettings
        prefs = Preferences.__new__(Preferences)
        prefs._store = QSettings(str(Path(__file__).resolve().parent / ".l10-t6-snap.ini"), QSettings.Format.IniFormat)
        try:
            prefs._store.clear()
            self.assertFalse(prefs.graph_snap_to_grid())
            prefs.set_graph_snap_to_grid(True)
            self.assertTrue(prefs.graph_snap_to_grid())
        finally:
            prefs._store.clear()
            prefs._store.sync()
            Path(prefs._store.fileName()).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
