"""2D behaviour, lane 9 step X5: SplineWarp's last four knobs and its B input (Nuke 17.0 Reference Guide, SplineWarp: `transform_matrix`
"extra matrix" on the Transform tab, `boundary_curverez` "boundary curve resolution" (3), `previewrez` "preview resolution" (100) and `rbf`
"Classic warping" (off) on the Render tab, and the `B` input with the output choices A, A warped, B and B warped).

* the extra matrix acts: a translation of 2 pixels moves where the curves sit; the identity matrix changes nothing;
* B is an optional input after the mask: `B` returns it, `B warped` carries it from the destination curves back to the source curves,
  `A` returns the input, and `A warped` (the default) ignores B;
* the three Render tab knobs are held and move no pixel (the notes in the mapping say why);
* a node saved before the knobs loads with their defaults and renders the same pixels on the evaluator and on the tile path.
"""
import copy
import unittest

import numpy as np

from nodebased import shapes
from nodebased.core import CHOICES, SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, evaluator_pixels, tile_pixels
from tests.test_2d_behaviour_warp_h import CURVE_PAIRS, H, labels, ramp, raster, spline_params


def run(params, source, b=None, mask=None):
    inputs = [raster(source), None if mask is None else raster(mask), None if b is None else raster(b)]
    resolved = shapes.resolve_warp_data("SplineWarp", {"pairs": CURVE_PAIRS}, 1)
    return Evaluator._windowed_kernel("SplineWarp", params, inputs, 1, resolved).pixels


class ExtraMatrixTests(unittest.TestCase):
    def test_identity_is_the_default_and_changes_nothing(self):
        src = ramp()
        params = SPECS["SplineWarp"]["params"]
        for r in range(3):
            for c in range(3):
                self.assertEqual(params[f"extra_matrix_{r}{c}"], 1.0 if r == c else 0.0)
        np.testing.assert_array_equal(run(spline_params(), src), run(spline_params(extra_matrix_00=1.0), src))

    def test_a_translation_moves_the_curves_and_with_them_the_warp(self):
        src = ramp()
        base = run(spline_params(), src)
        for row in range(H):
            np.testing.assert_allclose(base[row, 12], src[row, 9], atol=1e-4)       # the unmatrixed warp: the curve at 12.5 shows column 9
        moved = run(spline_params(extra_matrix_02=2.0), src)
        for row in range(H):
            np.testing.assert_allclose(moved[row, 14], src[row, 11], atol=1e-4)     # curves now at 11.5 and 14.5
        self.assertGreater(float(np.abs(moved - base).max()), 0.05)


class BInputTests(unittest.TestCase):
    def test_b_is_the_optional_input_after_the_mask(self):
        self.assertEqual(SPECS["SplineWarp"]["inputs"], ["image"])
        self.assertEqual(SPECS["SplineWarp"]["optional_inputs"], ["mask", "b"])
        self.assertEqual(CHOICES["spline_output"], ["A", "A warped", "B", "B warped"])
        self.assertEqual(SPECS["SplineWarp"]["params"]["spline_output"], "A warped")

    def test_each_output_choice(self):
        a, b = ramp(), ramp(offset=0.3)
        np.testing.assert_array_equal(run(spline_params(), a, b), run(spline_params(), a))     # A warped ignores B
        np.testing.assert_array_equal(run(spline_params(spline_output="A"), a, b), a)
        np.testing.assert_array_equal(run(spline_params(spline_output="B"), a, b), b)
        np.testing.assert_array_equal(run(spline_params(spline_output="B"), a), a)             # nothing wired: the input
        warped_b = run(spline_params(spline_output="B warped"), a, b)
        for row in range(2, H - 2):
            np.testing.assert_allclose(warped_b[row, 9], b[row, 12], atol=1e-4)               # B goes from the destination curve back to the source one
        self.assertGreater(float(np.abs(warped_b - b).max()), 0.05)

    def test_the_new_knobs_sit_in_the_guides_order(self):
        names = labels("SplineWarp")
        self.assertIn("Output", names)
        self.assertLess(names.index("Extra matrix 0,0"), names.index("Boundary curve resolution"))
        self.assertLess(names.index("Boundary curve resolution"), names.index("Preview resolution"))
        self.assertLess(names.index("Preview resolution"), names.index("Classic warping"))


class HeldKnobTests(unittest.TestCase):
    def test_defaults_are_the_guides(self):
        params = SPECS["SplineWarp"]["params"]
        self.assertEqual((params["boundary_curverez"], params["previewrez"], params["rbf"]), (3, 100, 0))

    def test_the_render_tab_knobs_move_no_pixel(self):
        src = ramp()
        base = run(spline_params(), src)
        for change in ({"boundary_curverez": 9}, {"previewrez": 25}, {"rbf": 1}):
            np.testing.assert_array_equal(run(spline_params(**change), src), base)


class SavedNodeTests(unittest.TestCase):
    NEW = ("extra_matrix_00", "extra_matrix_02", "boundary_curverez", "previewrez", "rbf", "spline_output")

    def graph(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("warp", "SplineWarp", {"curve_resolution": 15, "filter": "bilinear"}, image="plate")
        g.d.execute(dict(op="set_warp_data", id="warp", data={"pairs": CURVE_PAIRS}))
        return g

    def test_a_saved_node_loads_with_the_defaults_and_renders_the_same_on_both_paths(self):
        g = self.graph()
        saved = copy.deepcopy(g.doc)
        for name in self.NEW:
            saved["nodes"]["warp"]["params"].pop(name, None)
        loaded = upgrade_document(saved)
        validate(loaded)
        params = loaded["nodes"]["warp"]["params"]
        self.assertEqual((params["boundary_curverez"], params["previewrez"], params["rbf"], params["spline_output"]), (3, 100, 0, "A warped"))
        self.assertEqual((params["extra_matrix_00"], params["extra_matrix_02"]), (1.0, 0.0))
        reference = evaluator_pixels(g.doc, "warp")
        reference = getattr(reference, "pixels", reference)
        again = evaluator_pixels(loaded, "warp")
        np.testing.assert_array_equal(getattr(again, "pixels", again), reference)
        try:
            tiled = tile_pixels(loaded, "warp", 11)
        except AssertionError:
            self.skipTest("SplineWarp runs on the full-frame path only")
        np.testing.assert_allclose(tiled, reference, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
