"""Lane 2 step X5, part 2: ColorLookup's `use_precomputed` and `range`, as the Nuke 17.0 Reference Guide lists them (Color nodes, ColorLookup,
"ColorLookup Tab"): "use precomputed table" (default on) and "range" (default 1: "Sets the precomputed LUT search range between 0 and this value to
increase processing speed"). Both are speed settings of a table Nuke looks the curves up in; NodeBased evaluates the curves exactly, so on a new node
neither moves a pixel. A graph saved before keeps the scaling it had through `older_range_scaling`, where `range` does move them."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nodebased import knobs  # noqa: E402
from nodebased.core import SPECS  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.tileexec import TileExecutor  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph  # noqa: E402

CURVE = '{"interpolation":"linear","points":[[0,0],[0.5,0.9],[1,1]]}'


def kernel(**params):
    ramp = np.linspace(0.0, 1.0, 5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    base = dict(SPECS["ColorLookup"]["params"], curve_master=CURVE)
    return Evaluator._kernel("ColorLookup", dict(base, **params), [ramp])


class GuideKnobTests(unittest.TestCase):
    def guide(self):
        nodes = json.loads((ROOT / "docs" / "parity" / "nuke_nodes.json").read_text())
        rows = nodes["nodes"]["ColorLookup"]["knobs"] if "nodes" in nodes else nodes["ColorLookup"]["knobs"]
        return {row["name"]: row for row in rows}

    def test_the_guide_lists_both_in_the_main_tab_in_this_order(self):
        rows = self.guide()
        names = list(rows)
        self.assertEqual(rows["use_precomputed"]["label"], "use precomputed table")
        self.assertEqual(rows["range"]["label"], "range")
        self.assertEqual(rows["use_precomputed"]["tab"], "ColorLookup Tab")
        self.assertEqual(rows["range"]["tab"], "ColorLookup Tab")
        self.assertLess(names.index("use_precomputed"), names.index("range"))
        self.assertLess(names.index("range"), names.index("lut"))

    def test_the_panel_follows_it_and_the_defaults_are_the_guides(self):
        rows = self.guide()
        self.assertEqual(bool(SPECS["ColorLookup"]["params"]["use_precomputed"]), bool(rows["use_precomputed"]["default"]))
        self.assertEqual(float(SPECS["ColorLookup"]["params"]["range"]), float(rows["range"]["default"]))
        order = [name for group in knobs.KNOB_LAYOUT["ColorLookup"] for name in group.params]
        self.assertLess(order.index("use_precomputed"), order.index("range"))
        self.assertLess(order.index("range"), order.index("curve_master"))

    def test_the_audit_pairs_them_with_the_guide_knobs(self):
        mapping = json.loads((ROOT / "docs" / "parity" / "nodebased_to_nuke.json").read_text())
        mapping = mapping["pairs"]["ColorLookup"]["knobs"]
        pairs = {tuple(row["nodebased"]): row for row in mapping}
        for name in ("use_precomputed", "range"):
            row = pairs[(name,)]
            self.assertEqual(row["nuke"], [name])
            self.assertEqual(row["kind"], "same")


class PixelTests(unittest.TestCase):
    def test_the_precomputed_table_changes_no_pixel_on_a_new_node(self):
        np.testing.assert_array_equal(kernel(use_precomputed=1), kernel(use_precomputed=0))

    def test_range_changes_no_pixel_on_a_new_node(self):
        base = kernel(range=1.0)
        for value in (0.25, 2.0, 10.0):
            np.testing.assert_array_equal(kernel(range=value), base, err_msg=f"range {value}")

    def test_the_curve_is_applied_exactly(self):
        # The master curve maps 0.5 to 0.9; the lookup returns that, not a table's nearest entry.
        flat = np.full((4, 4, 4), 0.5, np.float32)
        base = dict(SPECS["ColorLookup"]["params"], curve_master=CURVE, range=0.1, use_precomputed=1)
        out = Evaluator._kernel("ColorLookup", base, [flat])
        np.testing.assert_allclose(out[..., :3], 0.9, atol=1e-5)

    def test_a_graph_saved_before_still_scales_by_range(self):
        # This is where `range` does move the pixels: the older reading divides the input by it.
        plain = kernel(older_range_scaling=1, range=1.0)
        scaled = kernel(older_range_scaling=1, range=2.0)
        self.assertGreater(float(np.abs(plain - scaled).max()), 1e-3)

    def test_the_tile_path_agrees_whatever_the_two_are_set_to(self):
        for use, rng in ((1, 1.0), (0, 1.0), (1, 4.0), (0, 0.25)):
            g = Graph()
            g.add("plate", "Constant", dict(width=16, height=12, red=0.3, green=0.5, blue=0.7, alpha=1.0))
            g.add("node", "ColorLookup", dict(curve_master=CURVE, use_precomputed=use, range=rng), image="plate")
            doc = dict(g.doc, view="node")
            full = Evaluator().evaluate(doc, "node", frame=1)
            for edge in (7, 16):
                with self.subTest(use=use, range=rng, edge=edge):
                    result = TileExecutor(evaluator=Evaluator(), tile_edge=edge).compose(doc, "node", frame=1, tier=1)
                    np.testing.assert_allclose(result.pixels, full, atol=1e-6)
            if (use, rng) != (1, 1.0):
                np.testing.assert_array_equal(full, self.reference())

    def reference(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=16, height=12, red=0.3, green=0.5, blue=0.7, alpha=1.0))
        g.add("node", "ColorLookup", dict(curve_master=CURVE), image="plate")
        return Evaluator().evaluate(dict(g.doc, view="node"), "node", frame=1)


if __name__ == "__main__":
    unittest.main()
