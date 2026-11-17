"""2D behaviour, lane 9 step W1: the Filter, Transform and Draw panels match the Nuke 17.0 Reference Guide's, from the review book's worklist.

Defaults follow the guide's tables; a document saved before keeps what it was saved with; Transform, CornerPin and Tracker take their motion blur
sample count from `motionblur` as Nuke does while an older node keeps its `samples` knob.
"""
import copy
import json
import os
import unittest

import numpy as np

from nodebased import core, knobs, transformbehaviour
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs, kern, render, texture
from tests.test_2d_parity_group_c4 import Graph as KeyGraph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def params(kind, name):
    return SPECS[kind]["params"][name]


class DefaultTests(unittest.TestCase):
    """Each default the worklist names, as the guide's table lists it."""

    GUIDE = (("Blur", "radius", 0.0), ("Crop", "intersect", 0), ("Sharpen", "sharpen_amount", 1.0), ("Glow", "glow_size", 15.0),
             ("Glow", "glow_threshold", 0.0), ("Soften", "soften_size", 1.0), ("Rectangle", "softness", 1.0), ("Radial", "softness", 1.0),
             ("Text", "font_size", 100.0), ("Text", "message", ""))

    def test_the_specs_carry_the_guides_defaults(self):
        for kind, name, value in self.GUIDE:
            with self.subTest(kind=kind, name=name):
                self.assertEqual(params(kind, name), value)

    def test_a_new_blur_does_nothing_until_it_is_sized(self):
        pixels = texture()
        np.testing.assert_array_equal(kern("Blur", pixels), pixels)
        self.assertTrue(differs(kern("Blur", pixels, radius=3.0), pixels))

    def test_a_blur_saved_at_size_eight_keeps_blurring_by_eight(self):
        g = Graph()
        g.add("src", "Checker", dict(width=45, height=37, size=5))
        g.add("n", "Blur", {"radius": 8.0}, image="src")
        saved = copy.deepcopy(g.doc)
        loaded = upgrade_document(saved)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["radius"], 8.0)
        blurred = render(loaded, "n")
        self.assertTrue(differs(blurred, render(loaded, "src")))
        np.testing.assert_array_equal(blurred, render(saved, "n"))

    def test_a_crop_saved_before_keeps_intersect_on(self):
        g = Graph()
        g.add("src", "Checker", dict(width=45, height=37, size=5))
        g.add("n", "Crop", {"x": -5, "y": -5, "width": 100, "height": 100}, image="src")
        old = copy.deepcopy(g.doc)
        old["nodes"]["n"]["params"].pop("intersect")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["intersect"], 1)

    def test_a_new_text_draws_nothing_until_it_has_a_message(self):
        g = Graph()
        g.add("n", "Text", {"width": 64, "height": 32})
        self.assertFalse(np.any(render(g.doc, "n")[..., :3]))

    def test_the_text_shadow_knobs_read_as_the_guides(self):
        labels = {group.params[0]: group.label for group in knobs.KNOB_LAYOUT["Text"] if group.label}
        self.assertEqual([labels[n] for n in ("shadow_angle", "shadow_distance", "shadow_opacity", "shadow_softness", "shadow_expand")],
                         ["Angle", "Distance", "Opacity", "Softness", "Shrink/expand"])
        self.assertEqual(labels["shadow_inherit_input_color"], "Inherit input color")


class AcceptedDefaultTests(unittest.TestCase):
    """A default that cannot follow the guide is recorded in the pairing with its reason, which shows on the node's page."""

    def test_the_accepted_defaults_carry_a_reason(self):
        with open(os.path.join(ROOT, "docs", "parity", "nodebased_to_nuke.json"), encoding="utf-8") as handle:
            pairs = json.load(handle)["pairs"]
        for node, knob in (("Sharpen", "channels"), ("Soften", "channels"), ("Text", "font"), ("Text", "message")):
            with self.subTest(node=node, knob=knob):
                self.assertIn(knob, pairs[node]["skip_default"])
                self.assertGreater(len(pairs[node]["default_notes"][knob]), 40)


class MotionBlurSampleTests(unittest.TestCase):
    """Nuke's `motionblur` is the number of samples; a node saved before took it as a switch and read `samples`."""

    def moving(self, kind="Transform", **extra):
        g = KeyGraph()
        g.add("plate", "Checker", dict(width=40, height=30, size=4))
        base = {"filter": "impulse", "shutter": 2.0, "shutter_offset": "start"}
        g.add("node", kind, {**base, **extra}, image="plate")
        name = "to1_x" if kind == "CornerPin" else "translate_x"
        g.key("node", name, 1, 0.0)
        g.key("node", name, 11, 10.0)
        return g

    def raster(self, g, frame=5):
        return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame)

    def test_the_count_is_motionblur_on_a_new_node(self):
        for kind in ("Transform", "CornerPin"):
            with self.subTest(kind=kind):
                self.assertEqual(params(kind, "legacy_blur_samples"), 0)
                new = self.raster(self.moving(kind, motionblur=2))
                legacy = self.raster(self.moving(kind, motionblur=1, samples=2, legacy_blur_samples=1))
                np.testing.assert_array_equal(new.pixels, legacy.pixels)
                self.assertEqual(new.data, legacy.data)

    def test_the_samples_knob_is_ignored_by_a_new_node(self):
        a = self.raster(self.moving(motionblur=3, samples=2))
        b = self.raster(self.moving(motionblur=3, samples=4))
        np.testing.assert_array_equal(a.pixels, b.pixels)

    def test_more_samples_change_the_blur(self):
        two = self.raster(self.moving(motionblur=2))
        eight = self.raster(self.moving(motionblur=8))
        union = two.data.union(eight.data)
        self.assertTrue(differs(two.fit(union), eight.fit(union)))

    def test_a_node_saved_before_reads_motionblur_as_a_switch(self):
        g = self.moving(motionblur=1, samples=4, legacy_blur_samples=1)
        old = copy.deepcopy(g.doc)
        old["nodes"]["node"]["params"].pop("legacy_blur_samples")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["node"]["params"]["legacy_blur_samples"], 1)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="node"), target="node", frame=5).pixels,
                                      self.raster(g).pixels)
        np.testing.assert_array_equal(self.raster(self.moving(motionblur=4)).pixels, self.raster(g).pixels)

    def test_the_count_helper(self):
        count = transformbehaviour.blur_sample_count
        self.assertEqual(count({"motionblur": 0, "samples": 4, "legacy_blur_samples": 0}), 0)
        self.assertEqual(count({"motionblur": 6, "samples": 4, "legacy_blur_samples": 0}), 6)
        self.assertEqual(count({"motionblur": 1, "samples": 4, "legacy_blur_samples": 1}), 4)
        self.assertEqual(count({"motionblur": 1, "samples": 4}), 4)                      # a missing flag is an older node
        self.assertEqual(count({"motionblur": 1000, "legacy_blur_samples": 0}), 256)

    def test_the_panel_has_a_number_for_motion_blur(self):
        for kind in ("Transform", "CornerPin", "Tracker"):
            with self.subTest(kind=kind):
                group = next(g for g in knobs.KNOB_LAYOUT[kind] if g.params == ("motionblur",))
                self.assertEqual(group.kind, "int")
        self.assertEqual(core.LIMITS["motionblur"], (0, 256))


if __name__ == "__main__":
    unittest.main()
