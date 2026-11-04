"""2D behaviour lane 2, step P, part 3: new AddTimeCode, TimeEcho, TimeBlur and ContactSheet nodes start with the guide's defaults, and a saved
document keeps the defaults it had (schema 32 fills the param a saved node does not store with the older value)."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from nodebased import core, olderdefaults  # noqa: E402
from nodebased.core import SCHEMA_VERSION, SPECS, upgrade_document  # noqa: E402
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels  # noqa: E402

GUIDE = {
    "AddTimeCode": {"timecode": "01:00:00:00"},
    "TimeEcho": {"frames": 1, "method": "max"},
    "TimeBlur": {"shutter": 0.5, "shutter_offset": "start"},
    "ContactSheet": {"rows": 3, "gap": 0, "width": 3072, "height": 2048},
}


def saved_before(document, kind):
    """The same graph as an earlier release stored it: version 31, and the four nodes without the params whose defaults changed."""
    doc = copy.deepcopy(document)
    for node in doc["nodes"].values():
        if node["type"] == kind:
            for name in olderdefaults.OLD[kind]:
                node["params"].pop(name, None)
    doc["version"] = 31
    return doc


class NewNodeTests(unittest.TestCase):
    def test_a_new_node_starts_with_the_guides_defaults(self):
        for kind, wanted in GUIDE.items():
            for name, value in wanted.items():
                with self.subTest(kind=kind, param=name):
                    self.assertEqual(SPECS[kind]["params"][name], value)

    def test_the_older_values_are_the_ones_the_nodes_used_to_start_with(self):
        self.assertEqual(olderdefaults.OLD["TimeEcho"], {"frames": 3, "method": "average"})
        self.assertEqual(olderdefaults.OLD["ContactSheet"], {"rows": 4, "gap": 4, "width": 1920, "height": 1080})
        self.assertEqual(olderdefaults.OLD["TimeBlur"], {"shutter": 1.0, "shutter_offset": "centred"})
        self.assertEqual(olderdefaults.OLD["AddTimeCode"], {"timecode": "00:00:00:00"})

    def test_the_schema_moved_on_by_one(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 32)


class SavedDocumentTests(unittest.TestCase):
    def graph(self):
        g = Graph()
        g.add("a", "Constant", dict(width=64, height=48, red=0.9, green=0.4, blue=0.1, alpha=1.0))
        g.add("echo", "TimeEcho", {}, image="a")
        g.add("blur", "TimeBlur", {}, image="a")
        g.add("sheet", "ContactSheet", {}, clip0="a")
        g.add("tc", "AddTimeCode", {}, image="a")
        return g

    def test_a_saved_node_that_lacks_the_params_gets_the_older_values(self):
        for kind, older in olderdefaults.OLD.items():
            doc = saved_before(self.graph().doc, kind)
            upgraded = upgrade_document(doc)
            self.assertEqual(upgraded["version"], SCHEMA_VERSION)
            node = next(n for n in upgraded["nodes"].values() if n["type"] == kind)
            for name, value in older.items():
                with self.subTest(kind=kind, param=name):
                    self.assertEqual(node["params"][name], value)

    def test_a_saved_node_that_stores_its_params_keeps_them(self):
        g = Graph()
        g.add("a", "Constant", dict(width=8, height=8))
        g.add("echo", "TimeEcho", dict(frames=5, method="plus"), image="a")
        doc = copy.deepcopy(g.doc)
        doc["version"] = 31
        params = upgrade_document(doc)["nodes"]["echo"]["params"]
        self.assertEqual((params["frames"], params["method"]), (5, "plus"))

    def test_the_contact_sheet_of_a_saved_document_is_bit_for_bit_the_older_one(self):
        g = self.graph()
        doc = saved_before(g.doc, "ContactSheet")
        old = copy.deepcopy(g.doc)
        for name, value in olderdefaults.OLD["ContactSheet"].items():
            old["nodes"]["sheet"]["params"][name] = value
        got = evaluator_pixels(upgrade_document(doc), "sheet")
        want = evaluator_pixels(old, "sheet")
        self.assertEqual(got.shape, (1080, 1920, 4))
        np.testing.assert_array_equal(got, want)

    def test_the_new_contact_sheet_is_the_guides_size(self):
        got = evaluator_pixels(self.graph().doc, "sheet")
        self.assertEqual(got.shape, (2048, 3072, 4))

    def test_the_time_nodes_of_a_saved_document_keep_their_window(self):
        g = Graph()
        g.add("a", "Checker", dict(width=32, height=24, size=4))
        g.add("echo", "TimeEcho", {}, image="a")
        g.add("blur", "TimeBlur", {}, image="a")
        for key in ("echo", "blur"):
            doc = saved_before(g.doc, g.doc["nodes"][key]["type"])
            old = copy.deepcopy(g.doc)
            for name, value in olderdefaults.OLD[old["nodes"][key]["type"]].items():
                old["nodes"][key]["params"][name] = value
            np.testing.assert_array_equal(evaluator_pixels(upgrade_document(doc), key), evaluator_pixels(old, key))

    def test_the_timecode_a_saved_node_writes_is_unchanged(self):
        g = self.graph()
        doc = saved_before(g.doc, "AddTimeCode")
        self.assertEqual(upgrade_document(doc)["nodes"]["tc"]["params"]["timecode"], "00:00:00:00")
        self.assertEqual(g.doc["nodes"]["tc"]["params"]["timecode"], "01:00:00:00")


if __name__ == "__main__":
    unittest.main()
