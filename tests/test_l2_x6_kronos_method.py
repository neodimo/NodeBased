"""Lane 2 step X6, part 2: Kronos picks its motion estimator like the Nuke 17.0 guide says.

The guide's Kronos tab: Method (`motionEstimation`), "Local - uses local block matching ... This option is for legacy compatibility.
Regularized - uses semi-global motion estimation to produce more consistent vectors between regions", Vector Detail (`vectorDetail`, the
density of the vectors) and Strength (`strength`, default 1.5, "only active if Method is set to Regularized"). OFlow's table in the guide
has none of these, so OFlow keeps the estimate it always made. A Kronos saved before this step reads Local, which is the estimate it always
made, so its pictures do not move (a missing Method is Local; no document version changes).

The fixture is Noise moving a whole 4 pixels a frame, so the picture half way between two frames is the plate moved by half the shift.
"""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nodebased import knobs, retimeknobs  # noqa: E402
from nodebased.core import CHOICES, LIMITS, SPECS, upgrade_document, validate  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.opticalflow import flow_pair  # noqa: E402
from nodebased.tileexec import TileExecutor  # noqa: E402
from tests.test_2d_behaviour_retime_vectors import moving_plate_picture  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph  # noqa: E402

MOTION = dict(interpolation="motion", speed=1.0, timing="frame", frame=1.5, vector_density=1.0)


def picture(params, kind="Kronos", frame=1):
    return moving_plate_picture(kind, dict(MOTION, **params), frame)


def moved_plate(shift):
    """The plate at frame 1 moved by `shift` pixels, built the way the fixture builds the later frames (nearest, whole pixels)."""
    g = Graph()
    g.add("plate", "Noise", dict(width=96, height=64, size=16, octaves=3))
    g.add("move", "Transform", dict(filter="nearest", translate_x=float(shift)), image="plate")
    return Evaluator().evaluate(dict(g.doc, view="move"), "move", frame=1)


def error_inside(result, expected, margin=12):
    """The mean error away from the edges, where the shifted plate has no data."""
    a, b = result[margin:-margin, margin:-margin, :3], expected[margin:-margin, margin:-margin, :3]
    return float(np.abs(a - b).mean())


def jitter(field):
    """Pixel-to-pixel change of the horizontal vectors: a smoother, more consistent field has less."""
    return float(np.abs(np.diff(field[..., 0], axis=1)).mean() + np.abs(np.diff(field[..., 0], axis=0)).mean())


ESTIMATE = dict(vector_density=1.0, vector_detail=4, smoothness=1.0, flicker_compensation=0, weight_red=0.3, weight_green=0.6, weight_blue=0.1)


class NewNodeTests(unittest.TestCase):
    def test_a_new_kronos_is_regularized_at_the_guides_strength(self):
        params = SPECS["Kronos"]["params"]
        self.assertEqual((params["motion_method"], params["motion_strength"]), ("Regularized", 1.5))
        self.assertEqual(CHOICES["motion_method"], ["Local", "Regularized"])
        self.assertIn("motion_strength", LIMITS)

    def test_oflow_has_neither_knob_because_the_guide_lists_neither(self):
        self.assertNotIn("motion_method", SPECS["OFlow"]["params"])
        self.assertNotIn("motion_strength", SPECS["OFlow"]["params"])
        names = {p for g in knobs.KNOB_LAYOUT["OFlow"] for p in g.params}
        self.assertFalse({"motion_method", "motion_strength"} & names)

    def test_the_panel_follows_the_guides_order_method_vector_detail_strength_smoothness(self):
        names = [g.params[0] for g in knobs.KNOB_LAYOUT["Kronos"]]
        order = [names.index(n) for n in ("motion_method", "vector_density", "motion_strength", "smoothness")]
        self.assertEqual(order, sorted(order))
        labels = {g.params[0]: g.label for g in knobs.KNOB_LAYOUT["Kronos"]}
        self.assertEqual((labels["motion_method"], labels["motion_strength"], labels["vector_density"]), ("Motion", "Strength", "Vector Detail"))


class MethodTests(unittest.TestCase):
    """Each Method on a moving fixture."""

    def test_both_methods_find_the_half_way_picture(self):
        expected = moved_plate(2.0)       # frame 1 moved by half of 4 pixels a frame
        for method in ("Local", "Regularized"):
            with self.subTest(method=method):
                self.assertLess(error_inside(picture(dict(motion_method=method)), expected), 0.12, method)

    def test_the_two_methods_give_different_pictures(self):
        self.assertFalse(np.array_equal(picture(dict(motion_method="Local")), picture(dict(motion_method="Regularized"))))

    def test_regularized_vectors_are_more_consistent_than_local(self):
        """The edge-aware pass averages each vector with its neighbours of similar colour: the field has less pixel-to-pixel jitter."""
        a, b = moved_plate(0.0), moved_plate(4.0)
        fields = {m: np.asarray(retimeknobs.estimate_flows(a, b, dict(ESTIMATE, motion_method=m), "cpu")[0], np.float32) for m in ("Local", "Regularized")}
        self.assertLess(jitter(fields["Regularized"]), jitter(fields["Local"]))
        self.assertFalse(np.array_equal(fields["Regularized"], fields["Local"]))


class StrengthTests(unittest.TestCase):
    def test_strength_acts_under_regularized(self):
        low = picture(dict(motion_method="Regularized", motion_strength=0.5))
        high = picture(dict(motion_method="Regularized", motion_strength=3.0))
        self.assertFalse(np.array_equal(low, high))

    def test_a_lower_strength_gives_a_smoother_field(self):
        a, b = moved_plate(0.0), moved_plate(4.0)
        fields = {s: np.asarray(retimeknobs.estimate_flows(a, b, dict(ESTIMATE, motion_method="Regularized", motion_strength=s), "cpu")[0], np.float32)
                  for s in (0.4, 4.0)}
        self.assertLess(jitter(fields[0.4]), jitter(fields[4.0]))

    def test_strength_is_inactive_under_local(self):
        low = picture(dict(motion_method="Local", motion_strength=0.5))
        high = picture(dict(motion_method="Local", motion_strength=3.0))
        np.testing.assert_array_equal(low, high)


class VectorDetailTests(unittest.TestCase):
    """Vector Detail is the guide's density: one vector per round(1 / density) pixels."""

    def test_the_density_sets_the_vector_spacing(self):
        for density, step in ((1.0, 1), (0.5, 2), (0.3, 3), (0.2, 5)):
            self.assertEqual(retimeknobs.density_step({"vector_density": density, retimeknobs.DENSITY_FLAG: 0}, 64, 96), step, density)

    def test_a_lower_density_changes_the_picture_under_each_method(self):
        for method in ("Local", "Regularized"):
            with self.subTest(method=method):
                full = picture(dict(motion_method=method, vector_density=1.0))
                sparse = picture(dict(motion_method=method, vector_density=0.25))
                self.assertFalse(np.array_equal(full, sparse))
                self.assertLess(error_inside(sparse, moved_plate(2.0)), 0.2)


def legacy_document(params):
    """A Kronos as the older documents hold it: none of the new params stored (`params` are then added on top)."""
    g = Graph()
    g.add("plate", "Noise", dict(width=96, height=64, size=16, octaves=3))
    g.add("move", "Transform", dict(filter="nearest"), image="plate")
    g.d.execute(dict(op="set_curve", id="move", param="translate_x",
                     curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(4 * (f - 1))) for f in range(-5, 30)])))
    g.add("node", "Kronos", dict(input_start=1.0, input_end=8.0, **MOTION), image="move")
    g.d.execute(dict(op="time", first=1, last=8, current=1))
    doc = copy.deepcopy(g.doc)
    for name in ("motion_method", "motion_strength"):
        doc["nodes"]["node"]["params"].pop(name)
    doc["nodes"]["node"]["params"].update(params)
    return dict(doc, view="node")


def render(doc):
    return Evaluator().evaluate_raster(doc, target="node", frame=1).pixels


class OldDocumentTests(unittest.TestCase):
    def test_a_saved_kronos_reads_local_and_renders_the_pictures_it_always_did(self):
        doc = upgrade_document(legacy_document({}))
        validate(doc)
        params = doc["nodes"]["node"]["params"]
        self.assertEqual((params["motion_method"], params["motion_strength"]), ("Local", 1.5))
        old = render(doc)
        np.testing.assert_array_equal(old, render(upgrade_document(legacy_document(dict(motion_method="Local")))))
        self.assertFalse(np.array_equal(old, render(upgrade_document(legacy_document(dict(motion_method="Regularized"))))))

    def test_local_is_the_estimate_the_node_made_before(self):
        """estimate_flows under Local is the window match at the node's Smoothness and Vector Detail, nothing added."""
        a, b = moved_plate(0.0), moved_plate(4.0)
        params = dict(ESTIMATE, vector_density=0.5, smoothness=0.7, motion_method="Local", motion_strength=0.2, **{retimeknobs.DENSITY_FLAG: 0})
        fw, bw, occ = retimeknobs.estimate_flows(a, b, params, "cpu")
        weights = retimeknobs.luminance_weights(params)
        ga, gb = retimeknobs.weighted_grey(a, weights), retimeknobs.weighted_grey(b, weights)
        rfw, rbw, rocc = flow_pair(ga, gb, vector_detail=4, smoothness=0.7, backend="cpu")
        rfw, rbw = (retimeknobs._block_field(np.asarray(f, np.float32), 2) for f in (rfw, rbw))
        np.testing.assert_array_equal(fw, rfw)
        np.testing.assert_array_equal(bw, rbw)
        np.testing.assert_array_equal(occ, rocc)

    def test_oflow_cannot_hold_the_new_knobs_and_keeps_its_estimate(self):
        with self.assertRaises(ValueError):
            picture(dict(motion_method="Regularized"), kind="OFlow")
        a, b = moved_plate(0.0), moved_plate(4.0)
        params = dict(ESTIMATE, vector_density=1.0, smoothness=1.0, **{retimeknobs.DENSITY_FLAG: 0})       # what an OFlow holds: no Method
        fw, _, _ = retimeknobs.estimate_flows(a, b, params, "cpu")
        weights = retimeknobs.luminance_weights(params)
        rfw = flow_pair(retimeknobs.weighted_grey(a, weights), retimeknobs.weighted_grey(b, weights), vector_detail=4, smoothness=1.0, backend="cpu")[0]
        np.testing.assert_array_equal(fw, np.asarray(rfw, np.float32))


# The pictures documents saved before this step rendered, measured on the code of commit f2472402 (the last one before the estimator
# change): sha256 of the float32 pixels. Kronos and OFlow share them because the fixture is the same plate and a motion retime.
SAVED_PICTURE_DIGESTS = {1.0: "a7eb221e9b9578122ba5946b6433a52aacf80598f3105f88c6dc1c16e6fd745e",
                         0.5: "547f54b696f20cfce9a9ba2a80cd19352af2afc3e62ad8e0ae663516f623d3f0"}


class SavedPictureDigestTests(unittest.TestCase):
    def test_old_documents_render_the_pixels_the_old_code_gave(self):
        import hashlib
        for kind in ("Kronos", "OFlow"):
            for density, digest in SAVED_PICTURE_DIGESTS.items():
                g = Graph()
                g.add("plate", "Noise", dict(width=96, height=64, size=16, octaves=3))
                g.add("move", "Transform", dict(filter="nearest"), image="plate")
                g.d.execute(dict(op="set_curve", id="move", param="translate_x",
                                 curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(4 * (f - 1))) for f in range(-5, 30)])))
                g.add("node", kind, dict(input_start=1.0, input_end=8.0, interpolation="motion", speed=1.0, timing="frame", frame=1.5,
                                         vector_density=density, smoothness=0.8), image="move")
                g.d.execute(dict(op="time", first=1, last=8, current=1))
                doc = copy.deepcopy(g.doc)
                for name in ("motion_method", "motion_strength"):
                    doc["nodes"]["node"]["params"].pop(name, None)           # an older document holds neither
                doc = upgrade_document(dict(doc, view="node"))
                with self.subTest(kind=kind, density=density):
                    self.assertEqual(hashlib.sha256(np.ascontiguousarray(render(doc)).tobytes()).hexdigest(), digest)


class TilePathTests(unittest.TestCase):
    def test_the_tile_path_runs_the_same_estimate_as_the_evaluator(self):
        from nodebased import tiers
        self.assertIs(tiers.REGION_RULES["Kronos"], tiers._identity)      # a whole-frame node: the tile path is the evaluator's code
        for method in ("Local", "Regularized"):
            doc = upgrade_document(legacy_document(dict(motion_method=method, motion_strength=1.0)))
            executor = TileExecutor(evaluator=Evaluator(), tile_edge=32)
            region = executor.canvas_region(doc, "node", frame=1, tier=1)
            tiled = executor.compose_region(doc, "node", region, frame=1, tier=1).pixels
            full = Evaluator().evaluate(doc, "node", frame=1)
            with self.subTest(method=method):
                self.assertEqual(tiled.shape, full.shape)
                np.testing.assert_allclose(tiled, full, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
