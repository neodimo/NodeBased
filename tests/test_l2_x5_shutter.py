"""Lane 2 step X5, part 1: MotionBlur2D and MotionBlur3D open the shutter where the Nuke 17.0 Reference Guide says.

The guide (Filter nodes, MotionBlur2D and MotionBlur3D, "shutter offset"; both pages carry the same text):
  centred - "center the shutter around the current frame ... the shutter stays open from frame 29,5 to 30,5."
  start   - "open the shutter at the current frame ... the shutter stays open from frame 30 to 31."
  end     - "close the shutter at the current frame ... the shutter stays open from frame 29 to 30."
  custom  - "open the shutter at the time you specify ... a value of -0.5 would open the shutter half a frame before the current frame."
(each for a shutter of 1 at frame 30). A node saved before this step keeps its old window through `legacy_shutter_window` 1, which
the 37 -> 38 migration sets.

The plate's red channel is the frame number, so the red value a blur returns is the middle of the window it sampled.
"""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nodebased import core  # noqa: E402
from nodebased.core import LIMITS, SPECS, upgrade_document, validate  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.tileexec import TileExecutor  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph  # noqa: E402

KINDS = ("MotionBlur2D", "MotionBlur3D")


def stamp_plate(g, name="plate", first=-20, last=100):
    g.add(name, "Constant", dict(width=16, height=12, red=0.0, green=0.25, blue=0.5, alpha=1.0))
    g.d.execute(dict(op="set_curve", id=name, param="red",
                     curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(f)) for f in range(first, last + 1)])))
    return name


def blur_document(kind, params):
    g = Graph()
    stamp_plate(g)
    g.add("node", kind, params, image="plate")
    return dict(g.doc, view="node")


def middle(doc, frame=30, tiled=None):
    """The red value (the middle of the sampled window) of node `node` at `frame`, on the evaluator or, with `tiled` an edge, the tile path."""
    if tiled:
        result = TileExecutor(evaluator=Evaluator(), tile_edge=tiled).compose(doc, "node", frame=frame, tier=1)
        return float(result.pixels[0, 0, 0])
    return float(Evaluator().evaluate(doc, "node", frame=frame)[0, 0, 0])


class GuideWindowTests(unittest.TestCase):
    """A new node: the middle of the window is the guide's, whatever the offset."""

    WINDOWS = {("start", 0.0): (30.0, 31.0), ("centred", 0.0): (29.5, 30.5), ("end", 0.0): (29.0, 30.0),
               ("custom", -0.5): (29.5, 30.5), ("custom", 2.0): (32.0, 33.0), ("custom", -2.0): (28.0, 29.0)}

    def test_a_new_node_has_the_flag_off(self):
        for kind in KINDS:
            self.assertEqual(SPECS[kind]["params"]["legacy_shutter_window"], 0)
            self.assertIn("legacy_shutter_window", LIMITS)

    def test_each_offset_samples_the_window_the_guide_gives(self):
        for kind in KINDS:
            for (offset, custom), (low, high) in self.WINDOWS.items():
                with self.subTest(kind=kind, offset=offset, custom=custom):
                    doc = blur_document(kind, dict(shutter=1.0, samples=4, shutter_offset=offset, custom_offset=custom))
                    self.assertAlmostEqual(middle(doc), (low + high) / 2, places=4)

    def test_the_shutter_length_scales_the_window(self):
        for kind in KINDS:
            doc = blur_document(kind, dict(shutter=0.5, samples=4, shutter_offset="start"))
            self.assertAlmostEqual(middle(doc), 30.25, places=4, msg=kind)
            doc = blur_document(kind, dict(shutter=2.0, samples=4, shutter_offset="end"))
            self.assertAlmostEqual(middle(doc), 29.0, places=4, msg=kind)

    def test_the_first_and_last_samples_stay_inside_the_window(self):
        # a one-sample blur reads the middle of the window; a four-sample blur's samples are spread evenly inside it
        doc = blur_document("MotionBlur2D", dict(shutter=1.0, samples=1, shutter_offset="start"))
        self.assertAlmostEqual(middle(doc), 30.5, places=4)

    def test_the_tile_path_agrees_with_the_full_frame_at_every_offset(self):
        for kind in KINDS:
            for (offset, custom) in self.WINDOWS:
                doc = blur_document(kind, dict(shutter=1.0, samples=4, shutter_offset=offset, custom_offset=custom))
                full = Evaluator().evaluate(doc, "node", frame=30)
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, offset=offset, custom=custom, edge=edge):
                        result = TileExecutor(evaluator=Evaluator(), tile_edge=edge).compose(doc, "node", frame=30, tier=1)
                        np.testing.assert_allclose(result.pixels, full, atol=1e-6)
                        self.assertAlmostEqual(float(result.pixels[0, 0, 0]), middle(doc), places=4)


class OlderDocumentTests(unittest.TestCase):
    """A saved node keeps the window it had: start ended at the frame, end began there, custom was centred on the offset."""

    OLD = {("start", 0.0): (29.0, 30.0), ("centred", 0.0): (29.5, 30.5), ("end", 0.0): (30.0, 31.0),
           ("custom", 2.0): (31.5, 32.5), ("custom", -0.5): (29.25, 29.75)}

    def saved(self, kind, offset, custom):
        document = blur_document(kind, dict(shutter=1.0, samples=4, shutter_offset=offset, custom_offset=custom))
        document = copy.deepcopy(document)
        document["version"] = 37
        document["nodes"]["node"]["params"].pop("legacy_shutter_window")
        return document

    def test_the_migration_sets_the_flag_and_the_schema_is_38(self):
        self.assertGreaterEqual(core.SCHEMA_VERSION, 38)
        for kind in KINDS:
            loaded = upgrade_document(self.saved(kind, "start", 0.0))
            validate(loaded)
            self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_shutter_window"], 1)
            self.assertEqual(loaded["version"], core.SCHEMA_VERSION)

    def test_a_saved_node_renders_the_pixels_it_did_before(self):
        for kind in KINDS:
            for (offset, custom), (low, high) in self.OLD.items():
                with self.subTest(kind=kind, offset=offset, custom=custom):
                    loaded = upgrade_document(self.saved(kind, offset, custom))
                    loaded["view"] = "node"
                    self.assertAlmostEqual(middle(loaded), (low + high) / 2, places=4)
                    for edge in (7, 16):
                        self.assertAlmostEqual(middle(loaded, tiled=edge), (low + high) / 2, places=4)

    def test_a_new_node_and_a_saved_one_differ_where_the_guide_moved_the_window(self):
        for kind in KINDS:
            new = blur_document(kind, dict(shutter=1.0, samples=4, shutter_offset="start"))
            old = upgrade_document(self.saved(kind, "start", 0.0))
            old["view"] = "node"
            self.assertAlmostEqual(middle(new) - middle(old), 1.0, places=4)

    def test_a_node_inside_a_group_is_migrated_too(self):
        document = self.saved("MotionBlur2D", "start", 0.0)
        inner = copy.deepcopy(document["nodes"]["node"])
        document["nodes"]["grp"] = {"type": "Group", "params": {}, "inputs": {}, "graph": {"nodes": {"inner": inner}}, "disabled": False}
        loaded = upgrade_document(document)
        self.assertEqual(loaded["nodes"]["grp"]["graph"]["nodes"]["inner"]["params"]["legacy_shutter_window"], 1)

    def test_a_node_saved_with_the_flag_off_keeps_it_off(self):
        document = copy.deepcopy(blur_document("MotionBlur2D", dict(shutter=1.0, samples=4, shutter_offset="start")))
        loaded = upgrade_document(document)
        self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_shutter_window"], 0)


class PanelTests(unittest.TestCase):
    def test_the_flag_is_laid_out_once_with_its_label(self):
        from nodebased import knobs
        for kind in KINDS:
            names = [name for group in knobs.KNOB_LAYOUT[kind] for name in group.params]
            self.assertEqual(names.count("legacy_shutter_window"), 1, kind)
            label = next(g.label for g in knobs.KNOB_LAYOUT[kind] if "legacy_shutter_window" in g.params)
            self.assertEqual(label, "Legacy shutter")


if __name__ == "__main__":
    unittest.main()
