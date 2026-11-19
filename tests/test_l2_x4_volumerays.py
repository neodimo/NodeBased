"""Lane 2 step X4: VolumeRays' secondary knobs, in the order, labels and defaults of Nuke 17.0's Reference Guide.

Reference: Nuke 17.0 Reference Guide, Filter nodes > VolumeRays (learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/volumerays.html).
Knobs, in the guide's order: Volumetric Creation, Luma Tolerance, Edge Size (0), Edge Blur (3), Volumetrics Center, Ray Length (20), Pre-Ray Blur (0),
Quality (Medium), Add on Top, Use Flickering (on), Flicker Speed (2), Flicker Size (40), Transform Noise with Volume Center, Desaturate before Coloring,
Volume Gamma, Volume Gain, Initial Volume Color (1), End Volume Color (0 0 0 0), Enable Radial Falloff, Radial Size, Radial Falloff, Enable Mask,
Mask Blur (0), Transform Mask with Volume Center.

What changes pixels here, and what the guide leaves open:
* Ray Length scales the rays' reach (the guide gives no unit: linearly over its default of 20, so 20 leaves Translate as set).
* Quality takes half, one, two or four times Steps (the guide gives no counts; Medium is the node's own).
* Enable Mask: the mask input limits the rays only when it is on; Mask Blur softens the mask's edge.
* Edge Size and Edge Blur (Alpha Edges and Shadow Making are not built), the four flicker knobs (no noise form stated), the two volume colours (neither
  described) and Transform Mask with Volume Center (no reference position) are on the node and change no pixel; the mapping's default_notes say so.

VolumeRays is outside the tile executor's whitelist (long radial reads), which routes it to the full-frame evaluator, so there is one kernel.
A saved node renders the same pixels.
"""
import copy
import json
import os
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import SPECS, Dispatcher, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_f import layout_order

W, H = 64, 48
GUIDE = {"edge_size": 0.0, "blur_size": 3.0, "raylength": 20.0, "ray_quality": "Medium", "chk_flicker": 1, "flicker_speed": 2.0, "flicker_size": 40.0,
         "xform_flicker": 0, "initcolor": 1.0, "volume_end_red": 0.0, "volume_end_green": 0.0, "volume_end_blue": 0.0, "volume_end_alpha": 0.0,
         "chk_use_mask": 1, "mask_blur": 0.0, "chk_xform_mask": 0}
INERT = {"edge_size": 25.0, "blur_size": 12.0, "chk_flicker": 0, "flicker_speed": 9.0, "flicker_size": 5.0, "xform_flicker": 1, "initcolor": 0.25,
         "volume_end_red": 1.0, "volume_end_alpha": 1.0, "chk_xform_mask": 1}


def rig(params=None, mask=None):
    d = Dispatcher()
    d.execute({"op": "create", "id": "img", "type": "Checker", "params": {"width": W, "height": H, "size": 8}})
    d.execute({"op": "create", "id": "rays", "type": "VolumeRays", "params": {"center_x": 32.0, "center_y": 24.0, "steps": 8, "decay": 0.9, **(params or {})}})
    d.execute({"op": "connect", "id": "rays", "input": "image", "source": "img"})
    d.execute({"op": "create", "id": "matte", "type": "Constant", "params": {"width": W, "height": H, "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0}})
    d.execute({"op": "connect", "id": "rays", "input": "matte", "source": "matte"})      # a solid matte lets every ray through
    if mask is not None:
        d.execute({"op": "create", "id": "mask", "type": mask[0], "params": {"width": W, "height": H, **mask[1]}})
        d.execute({"op": "connect", "id": "rays", "input": "mask", "source": "mask"})
    return d.document


def pixels(doc):
    return Evaluator().evaluate_raster(doc, target="rays", frame=1).pixels


def gap(a, b):
    return float(np.abs(a - b).max())


class VolumeRaysKnobs(unittest.TestCase):
    def test_defaults_are_the_guides(self):
        for name, value in GUIDE.items():
            self.assertEqual(SPECS["VolumeRays"]["params"][name], value, name)

    def test_the_main_tab_follows_the_guides_order_and_labels(self):
        kept, total = layout_order("VolumeRays")[0]
        self.assertEqual(kept, total)
        labels = [g.label for g in knobs.KNOB_LAYOUT["VolumeRays"]]
        wanted = ["Volumetric Creation", "Luma Tolerance", "Edge Size", "Edge Blur", "Volumetrics Center", "Ray Length", "Pre-Ray Blur", "Quality",
                  "Add on Top", "Use Flickering", "Flicker Speed", "Flicker Size", "Transform Noise with Volume Center", "Desaturate before Coloring",
                  "Volume Gamma", "Volume Gain", "Initial Volume Color", "End Volume Color", "Enable Radial Falloff", "Radial Size", "Radial Falloff",
                  "Enable Mask", "Mask Blur", "Transform Mask with Volume Center"]
        self.assertEqual(labels[:len(wanted)], wanted)

    def test_ray_length_scales_the_reach_and_twenty_leaves_translate_as_set(self):
        base = pixels(rig({"comp_me": 0}))
        longer, shorter = pixels(rig({"raylength": 40.0, "comp_me": 0})), pixels(rig({"raylength": 5.0, "comp_me": 0}))
        self.assertGreater(gap(base, longer), 0.01)
        self.assertGreater(gap(base, shorter), 0.01)
        # Ray Length 40 with Translate 0.5 reaches as far as Ray Length 20 with Translate 1
        np.testing.assert_allclose(pixels(rig({"raylength": 40.0, "translate": 0.5, "comp_me": 0})), base, atol=1e-6)

    def test_quality_takes_more_samples_the_higher_it_is(self):
        by = {q: pixels(rig({"ray_quality": q, "comp_me": 0})) for q in ("Low", "Medium", "High", "Very High")}
        np.testing.assert_array_equal(by["Medium"], pixels(rig({"comp_me": 0})))                # Medium is Steps as set
        np.testing.assert_allclose(by["High"], pixels(rig({"steps": 16, "comp_me": 0})), atol=1e-6)
        np.testing.assert_allclose(by["Very High"], pixels(rig({"steps": 32, "comp_me": 0})), atol=1e-6)
        np.testing.assert_allclose(by["Low"], pixels(rig({"steps": 4, "comp_me": 0})), atol=1e-6)
        for a, b in (("Low", "Medium"), ("Medium", "High"), ("High", "Very High")):
            self.assertGreater(gap(by[a], by[b]), 1e-4, (a, b))

    def test_enable_mask_decides_whether_the_mask_input_limits_the_rays(self):
        hide = ("Constant", {"red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 0.0})
        image = Evaluator().evaluate_raster(rig(), target="img", frame=1).pixels
        rays_only = pixels(rig({"comp_me": 0}))
        masked = pixels(rig({"comp_me": 0}, hide))
        np.testing.assert_allclose(masked, image, atol=1e-6)                                      # a mask with no alpha hides the effect
        ignored = pixels(rig({"comp_me": 0, "chk_use_mask": 0}, hide))
        np.testing.assert_allclose(ignored, rays_only, atol=1e-6)                                  # Enable Mask off: the mask is ignored
        self.assertGreater(gap(masked, ignored), 0.01)

    def test_mask_blur_softens_the_edge_of_the_mask(self):
        box = ("Rectangle", {"box_x": 16.0, "box_y": 12.0, "box_width": 32.0, "box_height": 24.0, "softness": 1.0})
        sharp = pixels(rig({"comp_me": 0}, box))
        soft = pixels(rig({"comp_me": 0, "mask_blur": 6.0}, box))
        self.assertGreater(gap(sharp, soft), 0.01)
        # Outside the rectangle the sharp mask shows the plain image; the blurred one lets some rays through there
        outside = (slice(2, 8), slice(2, 8))
        image = Evaluator().evaluate_raster(rig(), target="img", frame=1).pixels
        np.testing.assert_allclose(sharp[outside], image[outside], atol=1e-6)
        mask_edge = (slice(8, 12), slice(30, 34))                                                  # just above the rectangle's top edge
        self.assertGreater(gap(soft[mask_edge], image[mask_edge]), 1e-3)

    def test_the_knobs_without_a_stated_effect_change_no_pixel(self):
        base = pixels(rig({"comp_me": 0}))
        for name, value in INERT.items():
            np.testing.assert_array_equal(pixels(rig({name: value, "comp_me": 0})), base, name)

    def test_the_pixel_length_knobs_scale_with_the_proxy_tier(self):
        from nodebased import tiers
        scaled = tiers.scale_params("VolumeRays", {"edge_size": 8.0, "blur_size": 6.0, "flicker_size": 40.0, "mask_blur": 4.0, "raylength": 20.0}, 2)
        self.assertEqual((scaled["edge_size"], scaled["blur_size"], scaled["flicker_size"], scaled["mask_blur"]), (4.0, 3.0, 20.0, 2.0))
        self.assertEqual(scaled["raylength"], 20.0)                                                  # a length of rays in the guide's units, not pixels

    def test_the_mapping_pairs_the_thirteen_and_says_which_change_no_pixel(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity", "nodebased_to_nuke.json")
        entry = json.load(open(path))["pairs"]["VolumeRays"]
        pairs = {e["nuke"][0]: e for e in entry["knobs"] if e.get("nuke")}
        for name in ("edge_size", "blur_size", "raylength", "quality", "chk_flicker", "flicker_speed", "flicker_size", "xform_flicker", "initcolor",
                     "volume_end_color", "chk_use_mask", "mask_blur", "chk_xform_mask"):
            self.assertIn(pairs[name]["kind"], ("same", "renamed", "components"), name)
        self.assertEqual(pairs["quality"]["nodebased"], ["ray_quality"])
        for name in ("edge_size", "chk_flicker", "initcolor", "chk_use_mask", "chk_xform_mask", "raylength", "quality"):
            self.assertIn(name, entry["default_notes"])

    def test_a_saved_node_renders_the_same_pixels(self):
        doc = rig({"comp_me": 1, "steps": 12})
        before = pixels(doc)
        for name in GUIDE:
            doc["nodes"]["rays"]["params"].pop(name)
        migrated = upgrade_document(copy.deepcopy(doc))
        validate(migrated)
        for name, value in GUIDE.items():
            if name != "chk_flicker":
                self.assertEqual(migrated["nodes"]["rays"]["params"][name], value, name)
        self.assertEqual(migrated["nodes"]["rays"]["params"]["chk_flicker"], 0)                 # a saved node never flickered
        np.testing.assert_array_equal(pixels(migrated), before)
        # a saved node with its mask wired and no Enable Mask knob still uses the mask
        hide = ("Constant", {"alpha": 0.0})
        saved = rig({"comp_me": 0}, hide)
        for name in GUIDE:
            saved["nodes"]["rays"]["params"].pop(name)
        np.testing.assert_allclose(pixels(upgrade_document(saved)), Evaluator().evaluate_raster(rig(), target="img", frame=1).pixels, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
