"""Nuke's Edit > Clone (lane 10, step T6 finish, 2026-11-17): a copy that keeps an active link to its parent, drawn with an
orange line in the Node Graph (17.0 User Guide, Working with Nodes > Cloning Nodes)."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import copy
import json
import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

from nodebased.core import Dispatcher, SCHEMA_VERSION, clone_group, validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402


def _doc():
    d = Dispatcher()
    d.execute({"op": "create", "id": "g", "type": "Grade", "params": {"gamma": 1.4}})
    d.execute({"op": "clone", "id": "g", "new_id": "c"})
    return d


class CloneDocumentTests(unittest.TestCase):
    def test_clone_copies_knobs_and_links_to_parent(self):
        d = _doc()
        self.assertEqual(d.document["settings"]["clones"], {"c": "g"})
        self.assertEqual(d.document["nodes"]["c"]["params"], d.document["nodes"]["g"]["params"])

    def test_an_edit_on_either_member_changes_both_in_one_undo_step(self):
        d = _doc()
        d.execute({"op": "set", "id": "c", "param": "gamma", "value": 2.0})
        self.assertEqual(d.document["nodes"]["g"]["params"]["gamma"], 2.0)
        d.execute({"op": "undo"})
        self.assertEqual((d.document["nodes"]["g"]["params"]["gamma"], d.document["nodes"]["c"]["params"]["gamma"]), (1.4, 1.4))
        d.execute({"op": "set_key", "id": "g", "param": "white", "frame": 1, "value": 0.8})
        self.assertIn("white", d.document["animation"]["curves"]["c"])
        d.execute({"op": "disable", "id": "g", "value": True})
        self.assertTrue(d.document["nodes"]["c"]["disabled"])

    def test_clone_of_a_clone_links_to_the_same_parent(self):
        d = _doc()
        d.execute({"op": "clone", "id": "c", "new_id": "c2"})
        self.assertEqual(d.document["settings"]["clones"], {"c": "g", "c2": "g"})
        self.assertEqual(clone_group(d.document, "c2"), ["g", "c", "c2"])

    def test_declone_and_delete_leave_independent_nodes(self):
        d = _doc()
        d.execute({"op": "clone", "id": "g", "new_id": "c2"})
        d.execute({"op": "delete", "id": "g"})
        self.assertEqual(d.document["settings"]["clones"], {"c2": "c"})
        d.execute({"op": "declone", "id": "c2"})
        self.assertNotIn("clones", d.document["settings"])
        d.execute({"op": "set", "id": "c", "param": "gamma", "value": 3.0})
        self.assertEqual(d.document["nodes"]["c2"]["params"]["gamma"], 1.4)

    def test_grouping_a_clone_drops_its_link(self):
        d = _doc()
        d.execute({"op": "group", "ids": ["c"]})
        self.assertNotIn("clones", d.document["settings"])

    def test_saved_clones_round_trip_and_older_documents_load(self):
        d = _doc()
        saved = json.loads(json.dumps(d.document))
        validate(saved)
        self.assertEqual(saved["settings"]["clones"], {"c": "g"})
        older = copy.deepcopy(saved)
        older["settings"].pop("clones")
        validate(older)
        self.assertEqual(older["version"], SCHEMA_VERSION)

    def test_bad_clone_sections_are_refused(self):
        d = _doc()
        d.execute({"op": "create", "id": "b", "type": "Blur"})
        for clones in ({"c": "b"}, {"c": "c"}, {"c": "missing"}, {}):
            bad = copy.deepcopy(d.document)
            bad["settings"]["clones"] = clones
            with self.subTest(clones=clones), self.assertRaises(ValueError):
                validate(bad)


class CloneInteractionTests(unittest.TestCase):
    def test_alt_k_clones_with_an_orange_link(self):
        ref = interactions.reference()
        mine = [e for e in ref["behaviours"] if e["id"] == "graph.clone"]
        original = interactions.reference
        interactions.reference = lambda: dict(ref, behaviours=mine)
        try:
            results = interactions.probe_all()
        finally:
            interactions.reference = original
        self.assertEqual(results["graph.clone"]["result"], "pass", results["graph.clone"]["detail"])


if __name__ == "__main__":
    unittest.main()
