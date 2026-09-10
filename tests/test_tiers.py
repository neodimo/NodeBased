"""Region-of-interest and proxy-tier rules (docs/EVALUATION_TIERS.md clauses C2 and C3).

These tests exercise the geometry and parameter arithmetic directly, without rendering. The
companion golden-image tests that prove an ROI render equals a crop of a full-frame render belong
with the evaluator once ROI execution lands; the rules have to be right first, because a wrong
inverse-map is invisible on an identity transform and catastrophic under rotation.
"""
import math
import unittest

import numpy as np

from nodebased.core import SPECS
from nodebased.tiers import (DATA_DEPENDENT_RULES, PIXEL_UNIT_PARAMS, PROXY_TIERS, REGION_RULES,
                             Region, SOLVED_TRANSFORM_FIELDS, UndeclaredRegionRule, input_regions,
                             scale_params, tier_of)
from nodebased.imaging import Evaluator


def arity(kind):
    spec = SPECS[kind]
    return len(spec["inputs"]) + len(spec.get("optional_inputs", []))


# Supplied to the kinds whose region rule depends on solved node_data rather than params.
IDENTITY_SOLVE = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                  "scale": 1.0, "center_x": 0.0, "center_y": 0.0}


class RegionGeometryTests(unittest.TestCase):
    def test_intersection_of_disjoint_regions_is_empty(self):
        a, b = Region(0, 0, 10, 10), Region(20, 20, 5, 5)
        self.assertTrue(a.intersect(b).is_empty)

    def test_intersection_is_commutative_and_clamped(self):
        a, b = Region(0, 0, 10, 10), Region(5, 5, 10, 10)
        self.assertEqual(a.intersect(b), Region(5, 5, 5, 5))
        self.assertEqual(b.intersect(a), Region(5, 5, 5, 5))

    def test_union_ignores_empty_operands(self):
        a, empty = Region(3, 4, 6, 6), Region(0, 0, 0, 0)
        self.assertEqual(a.union(empty), a)
        self.assertEqual(empty.union(a), a)

    def test_expand_grows_every_side(self):
        self.assertEqual(Region(10, 10, 5, 5).expand(2, 3), Region(8, 7, 9, 11))

    def test_clamp_confines_to_the_image(self):
        self.assertEqual(Region(-5, -5, 20, 20).clamp(10, 10), Region(0, 0, 10, 10))

    def test_scaled_rounds_outward_so_no_needed_pixel_is_dropped(self):
        # 1..7 at tier 2 must still cover the half-open source span, so the right edge rounds up.
        self.assertEqual(Region(1, 1, 6, 6).scaled(2), Region(0, 0, 4, 4))
        self.assertEqual(Region(3, 3, 3, 3).scaled(4), Region(0, 0, 2, 2))

    def test_scaled_at_tier_one_is_identity(self):
        region = Region(7, 9, 13, 15)
        self.assertIs(region.scaled(1), region)

    def test_as_slices_addresses_the_same_pixels(self):
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        region = Region(2, 3, 4, 5)
        self.assertEqual(image[region.as_slices()].shape, (5, 4))


class RegionRuleCoverageTests(unittest.TestCase):
    def test_every_node_kind_declares_a_rule(self):
        """Contract clause C2: a kernel with no declared mapping is a hard error."""
        missing = sorted(set(SPECS) - set(REGION_RULES))
        self.assertEqual(missing, [], f"node kinds without an ROI rule: {missing}")

    def test_rules_do_not_outlive_their_node_kinds(self):
        stale = sorted(set(REGION_RULES) - set(SPECS))
        self.assertEqual(stale, [], f"ROI rules for kinds that no longer exist: {stale}")

    def test_undeclared_kind_raises_rather_than_falling_back(self):
        with self.assertRaises(UndeclaredRegionRule):
            input_regions("Hypothetical", {}, Region(0, 0, 8, 8), 1)

    def test_every_rule_returns_one_entry_per_declared_slot(self):
        region = Region(0, 0, 16, 16)
        for kind in SPECS:
            params = dict(SPECS[kind]["params"])
            with self.subTest(kind=kind):
                self.assertEqual(len(input_regions(kind, params, region, arity(kind),
                                                   solved=IDENTITY_SOLVE)), arity(kind))

    def test_data_dependent_rules_refuse_to_guess(self):
        """A Tracker's map lives in node_data, not params. No solve means no region — never the
        whole frame, which would erase the optimisation where nobody would notice."""
        self.assertEqual(sorted(DATA_DEPENDENT_RULES), ["Tracker"])
        for kind in DATA_DEPENDENT_RULES:
            with self.subTest(kind=kind):
                with self.assertRaises(UndeclaredRegionRule):
                    input_regions(kind, dict(SPECS[kind]["params"]), Region(0, 0, 8, 8), arity(kind))
                with self.assertRaises(UndeclaredRegionRule):
                    input_regions(kind, dict(SPECS[kind]["params"]), Region(0, 0, 8, 8), arity(kind),
                                  solved={SOLVED_TRANSFORM_FIELDS[0]: 1.0})


class RegionRuleSemanticsTests(unittest.TestCase):
    def test_generators_read_nothing(self):
        for kind in ("Read", "Constant", "Checker"):
            with self.subTest(kind=kind):
                self.assertEqual(input_regions(kind, dict(SPECS[kind]["params"]),
                                               Region(0, 0, 8, 8), arity(kind)), [])

    def test_pointwise_filters_pass_the_region_through_unchanged(self):
        region = Region(4, 6, 10, 12)
        for kind in ("Grade", "ColorCorrect", "Shuffle", "Premult", "Unpremult", "Dot", "Viewer"):
            with self.subTest(kind=kind):
                for got in input_regions(kind, dict(SPECS[kind]["params"]), region, arity(kind)):
                    self.assertEqual(got, region)

    def test_blur_expands_by_its_radius(self):
        region = Region(20, 20, 10, 10)
        image, mask = input_regions("Blur", {"radius": 8.0}, region, arity("Blur"))
        self.assertEqual(image, Region(12, 12, 26, 26))
        # The mask gates the result, so it is sampled only at the output pixels.
        self.assertEqual(mask, region)

    def test_blur_below_the_kernel_cutoff_reads_only_what_it_is_asked_for(self):
        # Evaluator._blur copies the source below 0.5, so expanding would request dead pixels.
        region = Region(20, 20, 10, 10)
        image, _ = input_regions("Blur", {"radius": 0.25}, region, arity("Blur"))
        self.assertEqual(image, region)

    def test_blur_radius_expansion_covers_the_real_kernel_support(self):
        """The declared support must not undershoot Evaluator._box_blur_axis's actual reach."""
        radius = 6.0
        source = np.zeros((41, 41, 4), np.float32)
        source[20, 20] = 1.0
        blurred = Evaluator._blur(source, {"radius": radius})
        touched = np.argwhere(blurred[..., 3] > 0)
        reach = max(abs(touched[:, 0] - 20).max(), abs(touched[:, 1] - 20).max())
        declared = input_regions("Blur", {"radius": radius}, Region(20, 20, 1, 1),
                                 arity("Blur"))[0]
        self.assertLessEqual(reach, 20 - declared.x,
                             "declared blur ROI is narrower than the kernel's real support")

    def test_crop_intersects_with_its_rectangle(self):
        params = {"x": 10, "y": 10, "width": 20, "height": 20}
        image, _ = input_regions("Crop", params, Region(0, 0, 100, 100), arity("Crop"))
        self.assertEqual(image, Region(10, 10, 20, 20))

    def test_crop_outside_its_rectangle_reads_nothing(self):
        params = {"x": 0, "y": 0, "width": 10, "height": 10}
        image, _ = input_regions("Crop", params, Region(50, 50, 10, 10), arity("Crop"))
        self.assertTrue(image.is_empty)

    def test_merge_asks_both_layers_for_the_same_region(self):
        region = Region(2, 3, 9, 9)
        self.assertEqual(input_regions("Merge", dict(SPECS["Merge"]["params"]), region, 2),
                         [region, region])

    def test_switch_prunes_the_unselected_branch(self):
        region = Region(0, 0, 8, 8)
        self.assertEqual(input_regions("Switch", {"which": 0}, region, 2), [region, None])
        self.assertEqual(input_regions("Switch", {"which": 1}, region, 2), [None, region])


class TransformRegionTests(unittest.TestCase):
    """The inverse map must agree with Evaluator._transform, corner for corner."""

    def sample_coords(self, params, region):
        """Reproduce the kernel's own inverse map for the destination pixels in `region`."""
        theta = math.radians(params["rotate"])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        inv_scale = 1.0 / params["scale"]
        gx, gy = np.meshgrid(np.arange(region.x, region.right, dtype=np.float64) + 0.5,
                             np.arange(region.y, region.bottom, dtype=np.float64) + 0.5)
        ox = gx - params["center_x"] - params["translate_x"]
        oy = gy - params["center_y"] - params["translate_y"]
        sx = (ox * cos_t + oy * sin_t) * inv_scale + params["center_x"]
        sy = (-ox * sin_t + oy * cos_t) * inv_scale + params["center_y"]
        return sx - 0.5, sy - 0.5

    def assert_covers(self, params, region):
        declared = input_regions("Transform", params, region, arity("Transform"))[0]
        sx, sy = self.sample_coords(params, region)
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}[params["filter"]]
        self.assertLessEqual(declared.x, math.floor(sx.min()) - support + 1)
        self.assertLessEqual(declared.y, math.floor(sy.min()) - support + 1)
        self.assertGreaterEqual(declared.right, math.ceil(sx.max()) + support - 1)
        self.assertGreaterEqual(declared.bottom, math.ceil(sy.max()) + support - 1)

    def base(self, **overrides):
        params = dict(SPECS["Transform"]["params"])
        params.update(overrides)
        return params

    def test_identity_transform_reads_its_own_region(self):
        declared = input_regions("Transform", self.base(), Region(10, 10, 20, 20),
                                 arity("Transform"))[0]
        self.assertLessEqual(declared.x, 10)
        self.assertGreaterEqual(declared.right, 30)

    def test_translation_shifts_the_read_region_the_opposite_way(self):
        params = self.base(translate_x=25.0, translate_y=-10.0)
        declared = input_regions("Transform", params, Region(50, 50, 10, 10),
                                 arity("Transform"))[0]
        self.assertLess(declared.x, 50 - 20)
        self.assertGreater(declared.bottom, 60 + 5)
        self.assert_covers(params, Region(50, 50, 10, 10))

    def test_rotation_bounding_box_covers_every_sampled_pixel(self):
        params = self.base(rotate=37.0, center_x=50.0, center_y=50.0, filter="bilinear")
        self.assert_covers(params, Region(20, 20, 30, 30))

    def test_scale_down_reads_a_larger_source_area(self):
        params = self.base(scale=0.25)
        region = Region(0, 0, 40, 40)
        declared = input_regions("Transform", params, region, arity("Transform"))[0]
        self.assertGreater(declared.width, region.width)
        self.assert_covers(params, region)

    def test_cubic_filter_reserves_more_support_than_nearest(self):
        region = Region(30, 30, 10, 10)
        near = input_regions("Transform", self.base(filter="nearest"), region, arity("Transform"))[0]
        cubic = input_regions("Transform", self.base(filter="cubic"), region, arity("Transform"))[0]
        self.assertLess(cubic.x, near.x)
        self.assertGreater(cubic.right, near.right)

    def test_combined_rotate_scale_translate_covers_every_sampled_pixel(self):
        params = self.base(rotate=-115.0, scale=0.6, translate_x=-30.0, translate_y=45.0,
                           center_x=120.0, center_y=80.0, filter="cubic")
        self.assert_covers(params, Region(15, 25, 64, 48))


class ProxyTierTests(unittest.TestCase):
    def test_tier_one_returns_the_params_untouched(self):
        params = dict(SPECS["Blur"]["params"])
        self.assertIs(scale_params("Blur", params, 1), params)

    def test_unsupported_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            scale_params("Blur", dict(SPECS["Blur"]["params"]), 3)

    def test_blur_radius_scales_with_the_tier(self):
        self.assertEqual(scale_params("Blur", {"radius": 8.0, "mix": 1.0}, 2)["radius"], 4.0)
        self.assertEqual(scale_params("Blur", {"radius": 8.0, "mix": 1.0}, 4)["radius"], 2.0)

    def test_unitless_params_are_never_touched(self):
        params = {"exposure": 2.0, "multiply": 1.5, "offset": 0.25, "mix": 0.5}
        self.assertEqual(scale_params("Grade", params, 4), params)
        transform = dict(SPECS["Transform"]["params"], rotate=45.0, scale=2.0, mix=0.5)
        scaled = scale_params("Transform", transform, 2)
        self.assertEqual(scaled["rotate"], 45.0)
        self.assertEqual(scaled["scale"], 2.0)
        self.assertEqual(scaled["mix"], 0.5)

    def test_transform_offsets_keep_their_fractional_part(self):
        params = dict(SPECS["Transform"]["params"], translate_x=5.0, center_x=13.0)
        scaled = scale_params("Transform", params, 2)
        self.assertEqual(scaled["translate_x"], 2.5)
        self.assertEqual(scaled["center_x"], 6.5)

    def test_extents_never_collapse_to_nothing(self):
        scaled = scale_params("Crop", {"x": 0, "y": 0, "width": 2, "height": 1}, 4)
        self.assertEqual(scaled["width"], 1)
        self.assertEqual(scaled["height"], 1)

    def test_checker_scales_its_cell_size_with_its_canvas(self):
        scaled = scale_params("Checker", dict(SPECS["Checker"]["params"]), 2)
        self.assertEqual(scaled["width"], 480)
        self.assertEqual(scaled["height"], 270)
        self.assertEqual(scaled["size"], 32)

    def test_every_pixel_unit_param_exists_on_its_node(self):
        for kind, names in PIXEL_UNIT_PARAMS.items():
            for name in names:
                with self.subTest(kind=kind, param=name):
                    self.assertIn(name, SPECS[kind]["params"])

    def test_scaling_is_declared_for_every_kind_with_pixel_units(self):
        """A pixel-unit param outside the table would silently mean two things at two tiers."""
        pixel_names = {"width", "height", "size", "radius", "x", "y",
                       "translate_x", "translate_y", "center_x", "center_y"}
        for kind, spec in SPECS.items():
            present = pixel_names & set(spec["params"])
            with self.subTest(kind=kind):
                self.assertEqual(present, set(PIXEL_UNIT_PARAMS.get(kind, ())))

    def test_tier_of_tolerates_documents_written_before_tiers_existed(self):
        self.assertEqual(tier_of({}), 1)
        self.assertEqual(tier_of({"proxy": {"tier": 4}}), 4)
        self.assertEqual(tier_of(2), 2)

    def test_declared_tiers_are_the_ones_scale_params_accepts(self):
        for tier in PROXY_TIERS:
            scale_params("Blur", {"radius": 8.0}, tier)


if __name__ == "__main__":
    unittest.main()
