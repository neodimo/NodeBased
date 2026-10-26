"""Fidelity step F, part 3: TimeOffset and FrameHold take Nuke's path mask (enable_path_mask, mask_patterns, inject_mask).

The upstream is an Add whose picture is a Constant keyed to `frame / 100` in red and whose mask input, a second Constant keyed to
`frame / 100` in alpha, is injected as the layer `mask`: a pixel of the picture and a pixel of the layer each say which input frame
they were read at. The time nodes evaluate their input at other frames, which the tile executor has no per-tile notion of, so they
fall back to the evaluator (asserted)."""
import copy
import unittest

import numpy as np

from nodebased import pathmask, timeknobs
from nodebased.core import SPECS, bypass_slot, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_c4 import Graph


def clip(g):
    g.add("plate", "Constant", dict(width=8, height=6, red=0.0, green=0.4, blue=0.6, alpha=1.0))
    g.add("matte", "Constant", dict(width=8, height=6, red=0.0, green=0.0, blue=0.0, alpha=0.0))
    for frame in range(-10, 61):
        g.key("plate", "red", frame, frame / 100.0)
        if frame >= 0:
            g.key("matte", "alpha", frame, frame / 100.0)          # an alpha cannot be negative; before frame 0 the first key holds
    g.add("src", "Add", dict(offset=0.0, maskChannelInput="rgba.alpha", inject=1), image="plate", mask="matte")
    return "src"


def build(kind="TimeOffset", **params):
    g = Graph()
    clip(g)
    base = dict(time_offset=3) if kind == "TimeOffset" else dict(first_frame=5)
    g.add("node", kind, {**base, **params}, image="src")
    return g


def read(g, frame, target="node"):
    raster = Evaluator().evaluate_raster(dict(g.doc, view=target), target=target, frame=frame)
    picture = round(float(raster.pixels[0, 0, 0]) * 100.0, 4)
    layer = raster.layers.get("mask") if raster.layers else None
    return raster, picture, (None if layer is None else round(float(layer.pixels[0, 0, 3]) * 100.0, 4))


class SelectionTests(unittest.TestCase):
    def test_the_patterns_choose_layers_by_name(self):
        for patterns, name, want in (("rgba", "rgba", True), ("rgba", "mask", False), ("*", "depth", True), ("RGBA", "rgba", True),
                                     ("m*", "mask", True), ("m*", "rgba", False), ("rgba mask", "mask", True), ("rgba,mask", "mask", True),
                                     ("rgba\nmask", "mask", True), ("light.*", "light.diffuse", True), ("light.*", "lightmap", False),
                                     ("*", "mask", True), ("", "rgba", False), ("?ask", "mask", True), ("[dm]ask", "mask", True),
                                     ("-mask", "mask", False), ("-mask", "rgba", True), ("*  -depth*", "depth.z", False),
                                     ("*  -depth*", "mask", True), ("!rgba", "rgba", False), ("rgba -rgba", "rgba", False)):
            with self.subTest(patterns=patterns, name=name):
                self.assertEqual(pathmask.is_selected(name, pathmask.patterns_of(patterns)), want)

    def test_defaults_are_off_with_no_patterns(self):
        for kind in ("TimeOffset", "FrameHold"):
            params = SPECS[kind]["params"]
            self.assertEqual((params["enable_path_mask"], params["mask_patterns"], params["inject_mask"]), (0, "", 0))
        self.assertEqual(timeknobs.upgrade_params("TimeOffset")["enable_path_mask"], 0)
        self.assertEqual(pathmask.KINDS, ("TimeOffset", "FrameHold"))


class TimeOffsetPathMaskTests(unittest.TestCase):
    def test_off_the_node_changes_the_time_of_everything(self):
        _, picture, layer = read(build(), 10)
        self.assertEqual((picture, layer), (7.0, 7.0))

    def test_patterns_set_which_layers_take_the_time_change(self):
        for patterns, want in (("rgba", (7.0, 10.0)), ("mask", (10.0, 7.0)), ("rgba mask", (7.0, 7.0)), ("*", (7.0, 7.0)),
                               ("", (10.0, 10.0)), ("-mask", (7.0, 10.0)), ("-rgba", (10.0, 7.0)), ("depth", (10.0, 10.0))):
            with self.subTest(patterns=patterns):
                _, picture, layer = read(build(enable_path_mask=1, mask_patterns=patterns), 10)
                self.assertEqual((picture, layer), want)

    def test_the_pixels_not_just_one_corner_follow_the_selection(self):
        g = build(enable_path_mask=1, mask_patterns="rgba")
        raster, _, _ = read(g, 12)
        np.testing.assert_allclose(raster.pixels[..., 0], 0.09, atol=1e-6)
        np.testing.assert_allclose(raster.layers["mask"].pixels[..., 3], 0.12, atol=1e-6)

    def test_the_other_time_knobs_still_work_inside_the_selection(self):
        _, picture, layer = read(build(enable_path_mask=1, mask_patterns="rgba", time_offset=-4, rounding_mode="Fractional time"), 10)
        self.assertEqual((picture, layer), (14.0, 10.0))
        _, picture, layer = read(build(enable_path_mask=1, mask_patterns="mask", time_offset=0.5), 10)
        self.assertEqual((picture, layer), (10.0, 9.0))

    def test_inject_mask_writes_the_time_changed_alpha_as_the_pathmask_layer(self):
        raster, _, _ = read(build(enable_path_mask=1, mask_patterns="rgba", inject_mask=1), 10)
        injected = raster.layers["pathmask"].pixels
        np.testing.assert_array_equal(injected, np.ones_like(injected))              # the plate's alpha is 1 at the time-changed frame
        raster, _, _ = read(build(enable_path_mask=1, mask_patterns="mask", inject_mask=1), 10)
        np.testing.assert_array_equal(raster.layers["pathmask"].pixels, np.zeros_like(raster.layers["pathmask"].pixels))
        raster, _, _ = read(build(enable_path_mask=1, mask_patterns="rgba"), 10)
        self.assertNotIn("pathmask", raster.layers)
        raster, _, _ = read(build(inject_mask=1), 10)                                  # without the path mask the switch does nothing
        self.assertNotIn("pathmask", raster.layers or {})

    def test_a_frame_before_the_clip_starts_follows_the_same_rules(self):
        _, picture, layer = read(build(enable_path_mask=1, mask_patterns="mask", time_offset=5), -8)
        self.assertEqual((picture, layer), (-8.0, 0.0))                                # the layer's keyed range holds its first key


class FrameHoldPathMaskTests(unittest.TestCase):
    def test_off_the_hold_applies_to_everything(self):
        _, picture, layer = read(build("FrameHold"), 20)
        self.assertEqual((picture, layer), (5.0, 5.0))

    def test_patterns_set_which_layers_are_held(self):
        for patterns, want in (("rgba", (5.0, 20.0)), ("mask", (20.0, 5.0)), ("*", (5.0, 5.0)), ("", (20.0, 20.0)), ("-rgba", (20.0, 5.0))):
            with self.subTest(patterns=patterns):
                _, picture, layer = read(build("FrameHold", enable_path_mask=1, mask_patterns=patterns), 20)
                self.assertEqual((picture, layer), want)

    def test_the_increment_steps_the_held_layers_only(self):
        _, picture, layer = read(build("FrameHold", enable_path_mask=1, mask_patterns="rgba", increment=6), 20)
        self.assertEqual((picture, layer), (17.0, 20.0))                                # 5 + 2 * 6

    def test_inject_mask_on_a_hold(self):
        raster, _, _ = read(build("FrameHold", enable_path_mask=1, mask_patterns="rgba", inject_mask=1), 20)
        self.assertEqual(float(raster.layers["pathmask"].pixels.min()), 1.0)


class TileBypassAndOldDocumentTests(unittest.TestCase):
    def test_both_nodes_fall_back_to_the_evaluator_on_the_tile_path_with_the_same_pixels(self):
        for kind in ("TimeOffset", "FrameHold"):
            g = build(kind, enable_path_mask=1, mask_patterns="rgba mask", inject_mask=1)
            executor = TileExecutor(evaluator=Evaluator(), tile_edge=4)
            doc = dict(g.doc, view="node")
            self.assertFalse(executor.supports_tiled(doc, "node"))
            region = executor.canvas_region(doc, "node", frame=10, tier=1)
            got = executor.compose_region(doc, "node", region, frame=10, tier=1).pixels
            np.testing.assert_array_equal(got, read(g, 10)[0].pixels)

    def test_a_disabled_node_passes_its_input_through_with_the_path_mask_on(self):
        for kind in ("TimeOffset", "FrameHold"):
            g = build(kind, enable_path_mask=1, mask_patterns="rgba")
            g.bypass("node")
            self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")
            _, picture, layer = read(g, 10)
            self.assertEqual((picture, layer), (10.0, 10.0))

    def test_a_node_saved_before_the_path_mask_loads_and_draws_the_same(self):
        for kind in ("TimeOffset", "FrameHold"):
            g = build(kind)
            old = copy.deepcopy(g.doc)
            for name in ("enable_path_mask", "mask_patterns", "inject_mask"):
                old["nodes"]["node"]["params"].pop(name, None)
            loaded = upgrade_document(old)
            validate(loaded)
            self.assertEqual(loaded["nodes"]["node"]["params"]["enable_path_mask"], 0)
            np.testing.assert_array_equal(read(Holder(loaded), 10)[0].pixels, read(g, 10)[0].pixels)


class Holder:
    def __init__(self, doc):
        self.doc = doc


if __name__ == "__main__":
    unittest.main()
