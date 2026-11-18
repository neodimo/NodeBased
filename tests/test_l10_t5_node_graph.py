"""The Node Graph against Nuke 17.0's User Guide (lane 10, step T5, 2026-11-17): the Tab menu, creating and wiring nodes, selecting,
disabling, renaming, copying and pasting, node colours and Backdrops, each driven with real key and mouse events on the built window.
The probes live in tools/nuke_panel_interactions.py (widget "node_graph"); the behaviours still open are listed in OPEN."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402

PASSING = ("graph.tab_create", "graph.tab_last_node", "graph.tab_weighting", "graph.add_connects_selected", "graph.disable_d",
           "graph.rename_n", "graph.select_all", "graph.shift_click_select", "graph.copy_paste", "graph.cut",
           "graph.ctrl_d_disconnect", "graph.autocolor", "graph.tile_color", "graph.backdrop_moves_contents")
OPEN = ("graph.clone",)


class NodeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = interactions.reference()
        mine = [e for e in ref["behaviours"] if e["widget"] == "node_graph"]
        original = interactions.reference
        interactions.reference = lambda: dict(ref, behaviours=mine)
        try:
            cls.results = interactions.probe_all()
        finally:
            interactions.reference = original

    def test_node_graph_behaviours_work_like_nuke(self):
        for behaviour in PASSING:
            with self.subTest(behaviour=behaviour):
                self.assertEqual(self.results[behaviour]["result"], "pass", self.results[behaviour]["detail"])

    def test_open_behaviours_are_reported(self):
        for behaviour in OPEN:
            self.assertNotEqual(self.results[behaviour]["result"], "pass", f"{behaviour} now passes: move it to PASSING")

    def test_every_entry_cites_a_source_or_is_undocumented(self):
        ref = interactions.reference()
        for entry in ref["behaviours"]:
            if entry["widget"] == "node_graph" and entry["status"] == "documented":
                self.assertIn(entry["source"], ref["sources"], entry["id"])
                self.assertTrue(entry["quote"], entry["id"])


if __name__ == "__main__":
    unittest.main()
