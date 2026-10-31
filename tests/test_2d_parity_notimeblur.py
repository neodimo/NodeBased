"""Lane L2 (2D parity), step N: NoTimeBlur, Nuke's node for cutting a time blur short.

The plate is a Constant whose red channel is keyed per frame (linear between keys), so the red value of a
result says which input frames were averaged. Keys 4 -> 0.0, 5 -> 1.0, 6 -> 0.2 make the average over
whole frames differ from the average over fractional frames, and floor from ceil.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import CHOICES, LIMITS, SPECS, bypass_slot
from nodebased.imaging import Evaluator, no_time_blur_frame
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from tests.test_2d_parity_group_c4 import Graph

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

KEYS = ((4, 0.0), (5, 1.0), (6, 0.2))
TIME_BLUR = dict(shutter=1.0, divisions=4, shutter_offset="centred", custom_offset=0.0)


def plate_value(frame):
    return float(np.interp(frame, [k for k, _ in KEYS], [v for _, v in KEYS]))


def red(document, target, frame):
    return float(Evaluator().evaluate(dict(document, view=target), target, frame=frame)[0, 0, 0])


def graph(rounding="rint", single=0, blur=True, size=(8, 8)):
    g = Graph()
    g.add("plate", "Constant", dict(width=size[0], height=size[1], red=0.0, green=0.4, blue=0.6, alpha=1.0))
    for frame, value in KEYS:
        g.key("plate", "red", frame, value)
    g.add("nt", "NoTimeBlur", dict(rounding=rounding, single=single), image="plate")
    if blur:
        g.add("tb", "TimeBlur", TIME_BLUR, image="nt")
    return g


class RegistrationTests(unittest.TestCase):
    def test_spec_matches_nukes_knobs(self):
        self.assertEqual(SPECS["NoTimeBlur"]["inputs"], ["image"])
        self.assertEqual(SPECS["NoTimeBlur"]["params"], {"rounding": "rint", "single": 1})
        self.assertEqual(CHOICES["rounding"], ["rint", "floor", "ceil", "none"])
        self.assertEqual(LIMITS["single"], (0, 1))

    def test_it_is_a_tiled_kind_and_filed_under_time(self):
        from nodebased.nodecatalog import NODE_CATEGORIES
        self.assertIn("NoTimeBlur", SUPPORTED_TILED_KINDS)
        self.assertIn("NoTimeBlur", NODE_CATEGORIES["Time"])

    def test_bypass_slot_is_the_image_input(self):
        g = graph(blur=False)
        self.assertEqual(bypass_slot(g.doc["nodes"]["nt"]), "image")


class FrameRoundingTests(unittest.TestCase):
    def test_modes_on_a_request(self):
        for rounding, expected in (("rint", 5), ("floor", 4), ("ceil", 5)):
            with self.subTest(rounding=rounding):
                self.assertEqual(no_time_blur_frame(4.625, 5, rounding, False), expected)
        self.assertEqual(no_time_blur_frame(4.5, 5, "rint", False), 4)      # halves go to even, like C's rint
        self.assertEqual(no_time_blur_frame(5.5, 5, "rint", False), 6)
        self.assertEqual(no_time_blur_frame(4.625, 5, "none", False), 4.625)
        self.assertEqual(no_time_blur_frame(4.625, 5, "none", True), 4.625)

    def test_single_rounds_the_output_frame_instead(self):
        for rounding in ("rint", "floor", "ceil"):
            self.assertEqual(no_time_blur_frame(4.625, 5, rounding, True), 5)
        self.assertEqual(no_time_blur_frame(4.625, 5.4, "ceil", True), 6)


class TimeBlurOverNoTimeBlurTests(unittest.TestCase):
    def test_without_it_the_shutter_blurs(self):
        g = graph()
        g.bypass("nt")
        expected = np.mean([plate_value(f) for f in (4.625, 4.875, 5.125, 5.375)])
        self.assertAlmostEqual(red(g.doc, "tb", 5), expected, places=5)
        self.assertNotAlmostEqual(red(g.doc, "tb", 5), plate_value(5), places=2)

    def test_each_rounding_mode_gives_whole_frame_samples(self):
        # Samples 4.625, 4.875, 5.125, 5.375 read whole frames: rint 5,5,5,5; floor 4,4,5,5; ceil 5,5,6,6.
        for rounding, frames in (("rint", (5, 5, 5, 5)), ("floor", (4, 4, 5, 5)), ("ceil", (5, 5, 6, 6))):
            with self.subTest(rounding=rounding):
                expected = np.mean([plate_value(f) for f in frames])
                self.assertAlmostEqual(red(graph(rounding, single=0).doc, "tb", 5), expected, places=5)
        self.assertNotAlmostEqual(red(graph("floor").doc, "tb", 5), red(graph("ceil").doc, "tb", 5), places=2)

    def test_none_passes_fractional_frames_through(self):
        g = graph("none", single=0)
        plain = graph()
        plain.bypass("nt")
        self.assertAlmostEqual(red(g.doc, "tb", 5), red(plain.doc, "tb", 5), places=6)
        self.assertAlmostEqual(red(g.doc, "nt", 4.625), plate_value(4.625), places=5)
        # Single does not change that: there is nothing to round.
        self.assertAlmostEqual(red(graph("none", single=1).doc, "tb", 5), red(plain.doc, "tb", 5), places=6)

    def test_single_makes_every_subframe_read_the_output_frame(self):
        for rounding in ("rint", "floor", "ceil"):
            with self.subTest(rounding=rounding):
                self.assertAlmostEqual(red(graph(rounding, single=1).doc, "tb", 5), plate_value(5), places=6)

    def test_single_with_a_fractional_output_frame_applies_the_rounding_to_it(self):
        for rounding, frame in (("rint", 5), ("floor", 5), ("ceil", 6)):
            with self.subTest(rounding=rounding):
                self.assertAlmostEqual(red(graph(rounding, single=1, blur=False).doc, "nt", 5.4),
                                       plate_value(frame), places=6)

    def test_a_time_offset_above_the_blur_moves_the_output_frame_with_it(self):
        # Output frame 6 through a TimeOffset of 1 asks the blur for frame 5, so the single frame is 5.
        g = graph("rint", single=1)
        g.add("off", "TimeOffset", dict(time_offset=1), image="tb")
        self.assertAlmostEqual(red(g.doc, "off", 6), plate_value(5), places=6)

    def test_edits_to_the_rounding_show_up_in_a_second_evaluation(self):
        ev = Evaluator()
        g = graph("floor")
        first = float(ev.evaluate(dict(g.doc, view="tb"), "tb", frame=5)[0, 0, 0])
        g.d.execute(dict(op="set", id="nt", param="rounding", value="ceil"))
        second = float(ev.evaluate(dict(g.doc, view="tb"), "tb", frame=5)[0, 0, 0])
        self.assertAlmostEqual(first, np.mean([plate_value(f) for f in (4, 4, 5, 5)]), places=5)
        self.assertAlmostEqual(second, np.mean([plate_value(f) for f in (5, 5, 6, 6)]), places=5)


class BypassTests(unittest.TestCase):
    def test_bypassed_node_passes_fractional_frames_through(self):
        g = graph("floor", single=1, blur=False)
        g.bypass("nt")
        self.assertAlmostEqual(red(g.doc, "nt", 4.625), plate_value(4.625), places=5)
        self.assertAlmostEqual(red(g.doc, "nt", 5), plate_value(5), places=6)


class TilePathTests(unittest.TestCase):
    """The node is solved whole and sliced per tile, like TimeBlur: tiled equals full frame."""

    def check(self, g, target, frame):
        doc = dict(g.doc, view=target)
        expected = Evaluator().evaluate(doc, target, frame=frame)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertTrue(executor.supports_tiled(doc, target))
        result = executor.compose(doc, target, frame=frame, tier=1)
        self.assertTrue(result.tiled)
        np.testing.assert_allclose(result.pixels, expected, atol=1e-6)
        return result.pixels

    def test_each_rounding_mode_tiled_equals_full_frame(self):
        for rounding in ("rint", "floor", "ceil", "none"):
            for single in (0, 1):
                with self.subTest(rounding=rounding, single=single):
                    self.check(graph(rounding, single, size=(48, 40)), "tb", 5)

    def test_tiled_values_are_the_whole_frame_samples(self):
        pixels = self.check(graph("floor", 0, size=(48, 40)), "tb", 5)
        self.assertAlmostEqual(float(pixels[0, 0, 0]), np.mean([plate_value(f) for f in (4, 4, 5, 5)]), places=5)

    def test_the_node_itself_as_the_target_tiled(self):
        g = graph("ceil", 1, blur=False, size=(48, 40))
        pixels = self.check(g, "nt", 5)
        self.assertAlmostEqual(float(pixels[0, 0, 0]), plate_value(5), places=6)

    def test_bypass_tiled_passes_through(self):
        g = graph("floor", 0, blur=False, size=(48, 40))
        g.bypass("nt")
        pixels = self.check(g, "nt", 5)
        self.assertAlmostEqual(float(pixels[0, 0, 0]), plate_value(5), places=6)


class RootPairingTests(unittest.TestCase):
    """Nuke's Root is the project settings. Each Root knob the mapping pairs with a document field has to name a field
    that exists and has to change behaviour; every other counted Root knob is listed once as unsupported."""

    @classmethod
    def setUpClass(cls):
        cls.pair = json.loads((ROOT / "docs/parity/nodebased_to_nuke.json").read_text())["pairs"]["Root"]
        cls.nuke = json.loads((ROOT / "docs/parity/nuke_nodes.json").read_text())["nodes"]["Root"]
        cls.paired = {e["nuke"][0]: e["nodebased"] for e in cls.pair["knobs"] if e["kind"] != "unsupported"}

    def test_every_counted_root_knob_is_listed_exactly_once(self):
        counted = [k["name"] for k in self.nuke["knobs"] if k["counted"]]
        listed = [n for e in self.pair["knobs"] for n in e["nuke"]]
        self.assertEqual(sorted(listed), sorted(counted))
        for e in self.pair["knobs"]:
            self.assertTrue(e["reason"], e)
            self.assertEqual(e["kind"] == "unsupported", e["nodebased"] == [], e)

    def test_the_paired_knobs_are_exactly_the_ones_that_act(self):
        self.assertEqual(set(self.paired), {"first_frame", "last_frame", "fps", "OCIO_config", "monitorLUT",
                                            "proxy", "proxy_scale"})
        from audit_nuke_fidelity import document_default
        for knob, fields in self.paired.items():
            for field in fields:
                with self.subTest(knob=knob, field=field):
                    document_default(field)             # raises KeyError for a field the document does not have

    def test_a_field_the_document_lacks_is_an_error_not_a_score(self):
        from audit_nuke_fidelity import document_default
        with self.assertRaises(KeyError):
            document_default("time.shutter")

    def test_the_score_counts_seven_knobs_and_the_defaults_that_agree(self):
        from audit_nuke_fidelity import score_pair
        res = score_pair("Root", self.pair, self.nuke)
        self.assertEqual(res["detail"]["knob_counts"], (7, 43))
        self.assertEqual(res["detail"]["default_counts"], (3, 5))   # frames and fps agree; the OCIO config and view do not
        self.assertEqual(set(res["components"]), {"knobs", "defaults"})

    def test_frame_range_bounds_the_playhead(self):
        from nodebased.core import Dispatcher
        d = Dispatcher()
        d.execute({"op": "time", "first": 10, "last": 20, "fps": 25})
        self.assertEqual(d.document["time"], {"first": 10, "last": 20, "current": 10, "fps": 25.0})
        d.execute({"op": "time", "first": 12})
        self.assertEqual(d.document["time"]["first"], 12)
        self.assertEqual(d.document["time"]["current"], 12)

    def test_fps_changes_what_a_time_based_file_reads(self):
        from nodebased.core import Dispatcher
        from tests.test_3d_alembic_nodes import FIXTURE
        d = Dispatcher()
        d.execute(dict(op="create", id="read", type="ReadAlembic3D", params={"abc_path": str(FIXTURE)}))
        ev = Evaluator()
        first = ev.evaluate_raster(d.document, "read", frame=1, typed=True).geometries[0].vertices.copy()
        d.document["time"]["fps"] = 12.0
        slower = ev.evaluate_raster(d.document, "read", frame=1, typed=True).geometries[0].vertices
        self.assertFalse(np.array_equal(first, slower))

    def test_the_ocio_config_reaches_a_colour_node_that_leaves_its_own_empty(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=8, height=8, red=0.5, green=0.5, blue=0.5, alpha=1.0))
        g.add("cs", "OCIOColorspace", dict(), image="plate")
        doc = dict(g.doc, view="cs")
        Evaluator().evaluate(doc, "cs", frame=1)
        broken = json.loads(json.dumps(doc))
        broken["settings"]["color"]["config"] = "/nonexistent/config.ocio"
        with self.assertRaises(Exception):
            Evaluator().evaluate(broken, "cs", frame=1)

    def test_the_view_changes_the_picture_the_viewer_shows(self):
        from nodebased.imaging import to_qimage
        frame = np.full((4, 4, 4), 0.2, np.float32)
        frame[..., 3] = 1.0
        grey = {view: to_qimage(frame, 0.0, "RGB", view=view).pixelColor(1, 1).red() for view in ("sRGB", "Linear")}
        self.assertNotEqual(grey["sRGB"], grey["Linear"])

    def test_the_proxy_tier_shrinks_what_the_evaluator_is_asked_for(self):
        from nodebased.core import Dispatcher, viewer_proxy
        d = Dispatcher()
        d.execute(dict(op="create", id="plate", type="Constant", params=dict(width=64, height=32)))
        d.execute(dict(op="view", id="plate"))
        self.assertEqual(viewer_proxy(d.document), 1)
        d.execute(dict(op="viewer_proxy", tier=2))
        tier = viewer_proxy(d.document)
        self.assertEqual(tier, 2)
        self.assertEqual(Evaluator().evaluate(d.document, "plate", frame=1, tier=1).shape[:2], (32, 64))
        self.assertEqual(Evaluator().evaluate(d.document, "plate", frame=1, tier=tier).shape[:2], (16, 32))


if __name__ == "__main__":
    unittest.main()
