"""Lane L2 (2D parity) plan 23, step AA1 part 2: a Merge with nothing in A outputs B unchanged
(hands-on pass 10/7, finding 3). Nuke's Merge treats an empty B as transparent black."""
import unittest

import numpy as np


from nodebased.core import MERGE_OPERATIONS
from tests.test_2d_parity_group_4c import Graph, evaluator_pixels, tile_pixels

OPERATIONS = MERGE_OPERATIONS


class MergeEmptyInputTests(unittest.TestCase):
    def build(self, operation, wire_a, wire_b, down=True):
        g = Graph()
        g.add("plate", "Checker", {"width": 96, "height": 64})
        g.add("other", "Constant", {"width": 96, "height": 64, "red": 0.2, "green": 0.5, "blue": 0.8, "alpha": 0.6})
        kwargs = {}
        if wire_a:
            kwargs["A"] = "other"
        if wire_b:
            kwargs["B"] = "plate"
        g.add("m", "Merge", {"operation": operation}, **kwargs)
        g.add("after", "Grade", {}, image="m") if down else None
        return g

    def test_empty_a_renders_b_exactly_for_every_operation_on_both_paths(self):
        for op in OPERATIONS:
            with self.subTest(operation=op):
                g = self.build(op, wire_a=False, wire_b=True)
                reference = evaluator_pixels(g.doc, "plate")
                np.testing.assert_array_equal(evaluator_pixels(g.doc, "m"), reference)
                np.testing.assert_array_equal(tile_pixels(g.doc, "m"), reference)
                np.testing.assert_array_equal(tile_pixels(g.doc, "m", tile_edge=32), reference)

    def test_a_chain_below_an_empty_a_merge_renders_without_error(self):
        g = self.build("over", wire_a=False, wire_b=True)
        g.d.execute(dict(op="disable", id="after", value=True))
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "after"), evaluator_pixels(g.doc, "plate"))
        g.d.execute(dict(op="disable", id="after", value=False))
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "after"), evaluator_pixels(g.doc, "plate"))

    def test_empty_b_with_a_connected_is_a_over_transparent_black(self):
        g = self.build("over", wire_a=True, wire_b=False)
        expected = evaluator_pixels(g.doc, "other")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "m"), expected, atol=1e-6)
        np.testing.assert_allclose(tile_pixels(g.doc, "m"), expected, atol=1e-6)

    def test_both_empty_still_reports_the_missing_inputs(self):
        g = self.build("over", wire_a=False, wire_b=False)
        with self.assertRaisesRegex(ValueError, "connect required input"):
            evaluator_pixels(g.doc, "m")


if __name__ == "__main__":
    unittest.main()
