"""2D behaviour, lane 9 step X1: the Bokeh node (nodebased/bokehknobs.py), built on ZDefocus's kernels.

The cases in tools/nuke_fidelity_cases_filter.py (`_bokeh_cases`) pass; every knob of the page's main tabs acts (or is listed with the state it needs);
the Bokeh page's order is the panel's order; tiles of 7, 16 and 32 equal the full frame, with and without a mask; a bypassed node passes its image;
the node is registered where a kind must be (specs, limits, choices, the tile executor, the catalogue, the theme).
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import bokehknobs, core, knobs
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from tests.test_2d_behaviour_filter_a import Graph, evaluator_pixels, tile_pixels

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402

H, W = 12, 16
NEW = SPECS["Bokeh"]["params"]


def texture(h=H, w=W, seed=3):
    pixels = np.random.default_rng(seed).random((h, w, 4)).astype(np.float32)
    pixels[..., 3] = 1.0
    return pixels


def ramp(h=H, w=W, lo=0.5, hi=6.0):
    return np.tile(np.linspace(lo, hi, w, dtype=np.float32), (h, 1))


def impulse(h=H, w=W):
    picture = np.zeros((h, w, 4), np.float32)
    picture[h // 2, w // 2, :] = 1.0
    return picture


def bokeh(image, depth=None, kernel=None, **knobs_):
    params = {**NEW, **knobs_}
    planes = bokehknobs.planes_for(np.asarray(image, np.float32), params, lambda name: None)
    return np.asarray(bokehknobs.render(image, depth, params, kernel, planes), np.float64)


def differs(a, b):
    return bool(np.abs(np.asarray(a) - np.asarray(b)).max() > 1e-6)


def labels(kind):
    return [g.label for g in knobs.KNOB_LAYOUT[kind] if g.kind != "legacy" and not g.params[0].startswith("legacy_") and g.params != ("mix",)]


# ------------------------------------------------------------------------------------------------------------- the cases
class CaseTests(unittest.TestCase):
    def test_every_scored_bokeh_case_passes(self):
        scored = [c for c in ruler.CASES if c.nuke == "Bokeh" and not c.unscored]
        self.assertGreaterEqual(len(scored), 20)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))

    def test_every_unscored_bokeh_case_states_its_reason(self):
        unscored = [c for c in ruler.CASES if c.nuke == "Bokeh" and c.unscored]
        self.assertGreaterEqual(len(unscored), 9)
        for case in unscored:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40, case.id)
                self.assertEqual(audit.run_case(case)["status"], "not scored")


# --------------------------------------------------------------------------------------- what the cases leave to a test
class KernelTests(unittest.TestCase):
    BASE = dict(focalPlane=1.0, focusRegionSize=0.0, multiplier=0.6, max_kernelsize=16.0)
    DEPTH = np.full((H, W), 3.0, np.float32)

    def test_softness_dissolves_the_disc_into_a_gaussian(self):
        hard = bokeh(impulse(), self.DEPTH, **self.BASE, kSoftness=0.0)[..., 0]
        soft = bokeh(impulse(), self.DEPTH, **self.BASE, kSoftness=1.0)[..., 0]
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=5)
        self.assertGreater(float(soft[6, 8]), float(soft[6, 12]))              # the middle weighs more than the rim
        np.testing.assert_allclose(hard[hard > 0], 1.0 / np.count_nonzero(hard), atol=1e-6)

    def test_spherical_aberration_moves_the_light_to_the_edge_or_the_middle(self):
        plain = bokeh(impulse(), self.DEPTH, **self.BASE, kSoftness=0.0)[..., 0]
        centre = bokeh(impulse(), self.DEPTH, **self.BASE, kSoftness=0.0, kSphAbb=-0.8)[..., 0]
        edge = bokeh(impulse(), self.DEPTH, **self.BASE, kSoftness=0.0, kSphAbb=0.8)[..., 0]
        self.assertGreater(float(centre[6, 8]), float(plain[6, 8]))              # negative: brighter in the centre
        self.assertLess(float(edge[6, 8]), float(plain[6, 8]))                    # positive: brighter on the edges
        self.assertGreater(float(edge[6, 12]), float(plain[6, 12]))
        self.assertAlmostEqual(float(centre.sum()), 1.0, places=5)

    def test_the_bloom_curvature_acts_with_a_spherical_aberration(self):
        kwargs = dict(self.BASE, kSoftness=0.0, kSphAbb=0.8)
        self.assertTrue(differs(bokeh(impulse(), self.DEPTH, **kwargs, bloomCurvature=1.0), bokeh(impulse(), self.DEPTH, **kwargs, bloomCurvature=4.0)))

    def test_bloom_adds_the_part_above_the_threshold(self):
        point = impulse()
        point[6, 8, :3] = 1.5
        plain = bokeh(point, self.DEPTH, **self.BASE, kSoftness=0.0)
        bloomed = bokeh(point, self.DEPTH, **self.BASE, kSoftness=0.0, bokeh_bloom=2.0, bloomThreshold=1.0)
        # the excess over 1.0 is 0.5 in each colour; bloom 2 adds 2 * 0.5 spread over the same disc
        np.testing.assert_allclose((bloomed - plain)[..., :3].sum(axis=(0, 1)), 1.0, atol=1e-5)
        np.testing.assert_array_equal(bloomed[..., 3], plain[..., 3])
        only = bokeh(point, self.DEPTH, **self.BASE, kSoftness=0.0, bokeh_bloom=2.0, bloomThreshold=1.0, outputType="Lens Bloom")
        np.testing.assert_allclose(only[..., :3], (bloomed - plain)[..., :3], atol=1e-6)

    def test_chromatic_aberration_gives_the_bloom_a_radius_per_colour(self):
        point = impulse()
        point[6, 8, :3] = 2.0
        base = dict(self.BASE, kSoftness=0.0, bokeh_bloom=1.0, bloomThreshold=0.5, outputType="Lens Bloom")
        flat = bokeh(point, self.DEPTH, **base)
        fringed = bokeh(point, self.DEPTH, **base, kChrAbb=1.0)
        np.testing.assert_allclose(flat[..., 0], flat[..., 1], atol=1e-7)
        self.assertTrue(differs(fringed[..., 0], fringed[..., 1]))                # red (offset 0.6) has the smaller disc
        self.assertGreater(np.count_nonzero(fringed[..., 1] > 1e-9), np.count_nonzero(fringed[..., 0] > 1e-9))
        np.testing.assert_allclose(fringed[..., 1], flat[..., 1], atol=1e-7)      # green and blue have offset 1: unchanged

    def test_the_kernel_input_is_the_filter_and_the_missing_channel_stands_in_for_an_empty_alpha(self):
        kernel = np.zeros((H, W, 4), np.float32)
        kernel[5:8, 7:10, 0] = 1.0                                                   # a red 3 x 3 block; the alpha is empty
        base = dict(self.BASE, kernelType="Input", max_kernelsize=8.0)
        out = bokeh(impulse(), self.DEPTH, kernel, **base, missingKernelChannel="rgba.red")
        self.assertGreater(float(out[..., 0].sum()), 0.9)
        self.assertGreater(np.count_nonzero(out[..., 0] > 1e-9), 4)
        self.assertTrue(differs(out, bokeh(impulse(), self.DEPTH, kernel, **base)))


class FocusTests(unittest.TestCase):
    def test_the_falloff_shapes_the_blur_between_the_plane_and_the_regions(self):
        picture, depth = texture(), ramp()
        base = dict(focalPlane=2.0, focusRegionSize=0.5, multiplier=0.4, max_kernelsize=12.0)
        self.assertTrue(differs(bokeh(picture, depth, **base), bokeh(picture, depth, **base, focusRegionFalloff="Smooth")))

    def test_the_focus_region_output_keeps_the_other_region_sharp(self):
        picture, depth = texture(), ramp(lo=0.5, hi=6.0)
        base = dict(focalPlane=3.0, focusRegionSize=0.5, multiplier=0.4, max_kernelsize=12.0)
        both = bokeh(picture, depth, **base)
        front = bokeh(picture, depth, **base, focusOutput="Front")
        back = bokeh(picture, depth, **base, focusOutput="Back")
        in_front, behind = depth < 2.75, depth > 3.25
        np.testing.assert_allclose(front[in_front], both[in_front], atol=1e-6)
        np.testing.assert_allclose(front[behind], picture[behind], atol=1e-6)
        np.testing.assert_allclose(back[behind], both[behind], atol=1e-6)
        np.testing.assert_allclose(back[in_front], picture[in_front], atol=1e-6)

    def test_the_visualization_is_limited_to_zero_to_one_when_normalized(self):
        picture, depth = texture(), ramp(lo=0.3, hi=6.0)
        base = dict(focalPlane=5.0, focusRegionSize=0.0, outputType="Focal Distance Visualization")
        raw = bokeh(picture, depth, **base, normalizeVisualization=0)
        clipped = bokeh(picture, depth, **base, normalizeVisualization=1)
        self.assertGreater(float(raw[..., :3].max()), 1.0)                          # the front amount past the plane exceeds 1
        self.assertLessEqual(float(clipped[..., :3].max()), 1.0)
        np.testing.assert_allclose(clipped[..., :3], np.clip(raw[..., :3], 0, 1), atol=1e-7)

    def test_the_focus_mattes_are_the_front_and_the_back(self):
        picture, depth = texture(), ramp()
        out = bokeh(picture, depth, focalPlane=3.0, focusRegionSize=0.0, outputType="Focus Mattes")
        self.assertTrue(np.all(out[depth < 3.0][:, 1] == 0) and np.all(out[depth < 3.0][:, 0] > 0))
        self.assertTrue(np.all(out[depth > 3.0][:, 0] == 0) and np.all(out[depth > 3.0][:, 1] > 0))

    def test_1_over_z_reads_the_depth_as_a_distance_the_other_way_round(self):
        real = bokeh(impulse(), np.full((H, W), 4.0, np.float32), focalPlane=2.0, focusRegionSize=0.0, multiplier=0.4, max_kernelsize=16.0)
        inverse = bokeh(impulse(), np.full((H, W), 0.25, np.float32), focalPlane=2.0, focusRegionSize=0.0, multiplier=0.4, max_kernelsize=16.0,
                        depthStyle="1/z")
        np.testing.assert_allclose(inverse, real, atol=1e-6)

    def test_a_matte_channel_limits_the_defocus_to_its_non_black_areas(self):
        picture, depth = texture(), np.full((H, W), 3.0, np.float32)
        base = dict(focalPlane=1.0, focusRegionSize=0.0, multiplier=0.4, max_kernelsize=12.0)
        full = bokeh(picture, depth, **base)
        matted = bokeh(picture, depth, **base, defocusMatteChannel="rgba.red")
        weight = picture[..., 0:1].astype(np.float64)
        np.testing.assert_allclose(matted, picture + (full - picture) * weight, atol=1e-5)
        self.assertTrue(differs(matted, full))

    def test_bloom_matte_and_chromatic_matte_limit_their_own_effects(self):
        picture = texture()
        picture[..., :3] += 0.8
        depth = np.full((H, W), 3.0, np.float32)
        base = dict(focalPlane=1.0, focusRegionSize=0.0, multiplier=0.4, max_kernelsize=12.0, bokeh_bloom=1.0, bloomThreshold=0.9, outputType="Lens Bloom")
        plain = bokeh(picture, depth, **base)
        matted = bokeh(picture, depth, **base, bloomMatteChannel="rgba.green")
        self.assertTrue(differs(plain, matted))
        np.testing.assert_allclose(matted[..., :3], plain[..., :3] * np.clip(picture[..., 1:2], 0, 1), atol=1e-5)
        chroma = dict(base, kChrAbb=1.0)
        everywhere = bokeh(picture, depth, **chroma)
        limited = bokeh(picture, depth, **chroma, chromicAberrationMatteChannel="rgba.blue")
        self.assertTrue(differs(everywhere, limited))


class LensAndSliceTests(unittest.TestCase):
    def lens(self, **knobs_):
        # a 100 mm lens at f/1.4, the plane at 1 m and the point at 4 m: a circle of confusion of about 6 mm on the film, 3 pixels of radius on 16
        base = dict(focalPlane=1.0, focusRegionSize=0.0, realWorldLens=1, worldScale="m", focalLength=100.0, fStop=1.4, max_kernelsize=16.0)
        return bokeh(impulse(), np.full((H, W), 4.0, np.float32), **{**base, **knobs_})

    def spread(self, out):
        return int(np.count_nonzero(out[..., 0] > 1e-9))

    def test_a_wider_aperture_blurs_more_and_a_longer_lens_blurs_more(self):
        self.assertGreater(self.spread(self.lens()), self.spread(self.lens(fStop=11.0)))
        self.assertGreater(self.spread(self.lens()), self.spread(self.lens(focalLength=50.0)))

    def test_the_lens_controls_have_no_effect_when_the_simulation_is_off(self):
        a = bokeh(impulse(), np.full((H, W), 4.0, np.float32), focalPlane=1.0, focusRegionSize=0.0, fStop=2.0, focalLength=200.0, worldScale="m")
        b = bokeh(impulse(), np.full((H, W), 4.0, np.float32), focalPlane=1.0, focusRegionSize=0.0)
        np.testing.assert_array_equal(a, b)

    def test_the_world_scale_and_film_format_act(self):
        self.assertTrue(differs(self.lens(worldScale="cm"), self.lens()))
        self.assertTrue(differs(self.lens(worldScaleMultiplier=1.0), self.lens(worldScaleMultiplier=0.2)))
        self.assertTrue(differs(self.lens(), self.lens(filmFormat="Custom", apertureWidth=11.0)))
        self.assertTrue(differs(self.lens(), self.lens(filmFormat="Custom", apertureHeight=8.0)))
        np.testing.assert_allclose(self.lens(), self.lens(filmFormat="Custom"), atol=1e-7)   # Custom at the 35mm numbers (22 x 16) is the 35mm format

    def test_corrective_slices_order_the_overlap(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[..., 3] = 1.0
        picture[..., 0] = 0.2
        depth = np.full((H, W), 8.0, np.float32)
        picture[6, 6:10, 0], depth[6, 6:10] = 1.0, 2.0                       # a bright near bar in front of a far background
        base = dict(focalPlane=5.0, focusRegionSize=2.0, multiplier=0.4, max_kernelsize=10.0)
        one = bokeh(picture, depth, **base, correctiveSlices=1)
        two = bokeh(picture, depth, **base, correctiveSlices=2)
        ten = bokeh(picture, depth, **base, correctiveSlices=10)
        self.assertTrue(differs(one, two))                                    # one slice: pixels gather from any depth
        np.testing.assert_allclose(two, ten, atol=1e-6)                       # nothing lies between the two sides of the volume
        wide = dict(correctiveSlices=3, overrideNearAndFarForSlices=1)
        huge = bokeh(picture, depth, **base, **wide, slices_near=1.0, slices_far=100.0)      # both depths in one slice: no ordering, as with one slice
        self.assertTrue(differs(huge, two))
        np.testing.assert_allclose(huge, one, atol=1e-6)
        self.assertTrue(differs(huge, bokeh(picture, depth, **base, **wide, slices_near=1.0, slices_far=3.0)))
        np.testing.assert_array_equal(bokeh(picture, depth, **base, slices_near=1.0, slices_far=3.0), bokeh(picture, depth, **base))   # override off


# --------------------------------------------------------------------------------------------- every knob acts
class KnobsActTests(unittest.TestCase):
    """Each paired knob moved off its default changes the pixels, alone or with the state it needs (the second dict)."""
    SCENE = dict(focalPlane=3.0, focusRegionSize=1.0, multiplier=0.5, max_kernelsize=12.0)
    LENS = {"realWorldLens": 1, "focalPlane": 100.0}
    BLOOM = {"bokeh_bloom": 1.0, "bloomThreshold": 0.3}
    CHROMA = {"bokeh_bloom": 1.0, "bloomThreshold": 0.3, "kChrAbb": 1.0}
    SLICES = {"correctiveSlices": 3, "overrideNearAndFarForSlices": 1}
    NEEDS = {
        "outputType": ("Focal Distance Visualization", {}), "focusOutput": ("Back", {}), "multiplier": (1.0, {}), "frontmultiplier": (3.0, {}),
        "backmultiplier": (3.0, {}), "depthStyle": ("1/z", {}), "focalPlane": (2.0, {}), "focusRegionSize": (4.0, {}),
        "focusRegionFalloff": ("Smooth", {}), "defocusMatteChannel": ("rgba.red", {}),
        "normalizeVisualization": (0, {"outputType": "Focal Distance Visualization", "focalPlane": 6.0}),
        "realWorldLens": (1, {"focalPlane": 100.0}), "focalLength": (70.0, LENS), "fStop": (2.0, LENS), "worldScale": ("m", LENS),
        "worldScaleMultiplier": (0.3, LENS), "filmFormat": ("Custom", {**LENS, "apertureWidth": 11.0}),
        "apertureWidth": (11.0, {**LENS, "filmFormat": "Custom"}), "apertureHeight": (8.0, {**LENS, "filmFormat": "Custom"}),
        "bokeh_bloom": (1.0, {"bloomThreshold": 0.3}), "bloomCurvature": (4.0, {"kSphAbb": 0.8}), "bloomThreshold": (0.9, {"bokeh_bloom": 1.0}),
        "bloomMatteChannel": ("rgba.red", BLOOM), "kSphAbb": (0.8, {}), "kChrAbb": (1.0, BLOOM),
        "chromic_offset_red": (2.0, CHROMA), "chromic_offset_green": (2.0, CHROMA), "chromic_offset_blue": (2.0, CHROMA),
        "chromicAberrationMatteChannel": ("rgba.red", CHROMA),
        "kernelType": ("Aperture Blades", {}), "max_kernelsize": (4.0, {}), "kAspectRatio": (2.0, {}), "kSoftness": (0.9, {}),
        "kNumSides": (9, {"kernelType": "Aperture Blades"}), "kCurvature": (0.9, {"kernelType": "Aperture Blades"}),
        "kRotation": (30.0, {"kernelType": "Aperture Blades"}),
        "correctiveSlices": (1, {}), "overrideNearAndFarForSlices": (1, {"correctiveSlices": 3, "slices_near": 0.5, "slices_far": 7.0}),
        "slices_near": (3.5, {**SLICES, "slices_far": 7.0}), "slices_far": (4.0, {**SLICES, "slices_near": 0.5}),
    }
    # depthChannel names a layer of the input (tested through the node below); missingKernelChannel needs a wired kernel with an empty alpha (above).
    OTHER = {"depthChannel", "missingKernelChannel", "mix", "channels", "maskChannelInput", "invert_mask", "inject", "fringe"}

    def scene(self):
        picture = texture()
        picture[..., :3] += 0.4
        picture[3, 3, :3], picture[8, 12, :3] = 3.0, 3.0
        return picture, ramp(lo=0.5, hi=7.0)

    def test_every_bokeh_knob_is_listed_here_or_tested_elsewhere(self):
        for name in NEW:
            self.assertTrue(name in self.NEEDS or name in self.OTHER, name)

    def test_each_knob_changes_the_pixels(self):
        picture, depth = self.scene()
        for name, (value, needs) in self.NEEDS.items():
            with self.subTest(knob=name):
                base = {**self.SCENE, **needs}
                self.assertTrue(differs(bokeh(picture, depth, **base), bokeh(picture, depth, **{**base, name: value})), (name, value, needs))

    def test_the_depth_channel_picks_the_layer_and_a_missing_depth_blurs_uniformly(self):
        picture = texture()
        near = Raster(np.full((H, W, 4), 1.0, np.float32))
        far = Raster(np.full((H, W, 4), 6.0, np.float32))
        source = Raster(picture, layers={"depth": near, "other": far})
        params = {**NEW, "focalPlane": 1.0, "focusRegionSize": 0.0, "multiplier": 0.4, "max_kernelsize": 12.0}
        on_plane = Evaluator._windowed_kernel("Bokeh", {**params, "depthChannel": "depth.Z"}, [source]).pixels
        elsewhere = Evaluator._windowed_kernel("Bokeh", {**params, "depthChannel": "other.Z"}, [source]).pixels
        np.testing.assert_allclose(on_plane, picture, atol=1e-6)                      # depth 1 is the focal plane: sharp
        self.assertTrue(differs(elsewhere, picture))                                  # depth 6 is behind it: blurred
        nothing = Evaluator._windowed_kernel("Bokeh", dict(params), [Raster(picture)]).pixels
        self.assertTrue(differs(nothing, picture))                                    # no depth at all: uniform defocus by the multiplier

    def test_a_wired_depth_input_is_read(self):
        picture = texture()
        base = {**NEW, "focalPlane": 1.0, "focusRegionSize": 0.0, "multiplier": 0.4, "max_kernelsize": 12.0}
        flat = lambda value: Raster(np.full((H, W, 4), value, np.float32))
        sharp = Evaluator._windowed_kernel("Bokeh", base, [Raster(picture), flat(1.0)]).pixels
        soft = Evaluator._windowed_kernel("Bokeh", base, [Raster(picture), flat(6.0)]).pixels
        np.testing.assert_allclose(sharp, picture, atol=1e-6)
        self.assertTrue(differs(soft, picture))


# ------------------------------------------------------------------------------------------------------------- panel and registration
class LayoutAndRegistrationTests(unittest.TestCase):
    def test_the_panel_follows_the_guide_page_order(self):
        shown = labels("Bokeh")
        self.assertEqual(shown[-4:], ["Mask", "Inject", "Invert", "Fringe"])     # the shared mask strip follows the page's own knobs
        self.assertEqual(shown[:-4], [
            "Bokeh Channels", "Depth Channel", "Output Type", "Focus Region Output", "Multiplier", "Front Multiplier", "Back Multiplier", "Depth Style",
            "Focal Plane", "Focus Region Size", "Focus Region Falloff", "Defocus Matte Channel", "Normalize Visualization",
            "Real World Lens Simulation", "Focal Length", "Aperture", "World Scale", "World Scale Multiplier", "Film Format", "Aperture Width",
            "Aperture Height",
            "Bloom", "Curvature", "Bloom Threshold", "Bloom Matte Channel", "Spherical Aberration", "Chromic Aberration", "Chromic Aberration Offset",
            "Chromic Aberration Matte Channel",
            "Kernel Type", "Maximum Kernel Size", "Missing Kernel Channel", "Aspect Ratio", "Softness", "Blade Count", "Curvature", "Rotation",
            "Number of Corrective Slices", "Override Near/Far", "Near and Far"])

    def test_every_param_is_laid_out_once(self):
        shown = [name for group in knobs.KNOB_LAYOUT["Bokeh"] for name in group.params]
        self.assertEqual(len(shown), len(set(shown)))
        for name in NEW:
            if name != "channels":
                self.assertIn(name, shown, name)

    def test_the_node_is_registered(self):
        from nodebased import nodecatalog, theme, tiers
        self.assertEqual(SPECS["Bokeh"]["inputs"], ["image"])
        self.assertEqual(SPECS["Bokeh"]["optional_inputs"], ["depth", "kernel", "mask"])
        self.assertIn("Bokeh", core.MASK_MIX_KINDS)
        for name in ("outputType", "focusOutput", "depthStyle", "focusRegionFalloff", "worldScale", "filmFormat", "kernelType"):
            self.assertIn(name, core.CHOICES)
        self.assertIn("Bokeh", open(theme.__file__).read())
        self.assertIn("Bokeh", open(nodecatalog.__file__).read())
        self.assertIn("Bokeh", open(tiers.__file__).read())

    def test_a_bypassed_node_passes_its_image(self):
        self.assertEqual(core.bypass_slot({"type": "Bokeh", "inputs": {"image": "a", "depth": "b"}}), "image")

    def test_the_filter_preview_the_kernel_input_and_the_lens_run_on_the_full_frame(self):
        for extra in ({"outputType": "Filter Shape"}, {"kernelType": "Input"}, {"realWorldLens": 1}):
            self.assertTrue(bokehknobs.needs_full_frame({**NEW, **extra}), extra)
        self.assertFalse(bokehknobs.needs_full_frame(NEW))

    def test_the_tile_halo_is_half_the_maximum_kernel_size(self):
        self.assertEqual(bokehknobs.reach({"max_kernelsize": 7.0}), 4)
        self.assertEqual(bokehknobs.reach({"max_kernelsize": 256.0}), 128)


# ---------------------------------------------------------------------------------------------------------------- the tiles
class TileTests(unittest.TestCase):
    def doc(self, masked, **knobs_):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("depth", "Ramp", dict(width=45, height=37, p0_x=0, p0_y=0, p1_x=44, p1_y=0) if "p0_x" in SPECS["Ramp"]["params"] else dict(width=45, height=37))
        inputs = {"image": "plate", "depth": "depth"}
        if masked:
            g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
            inputs["mask"] = "matte"
        params = {"focalPlane": 0.5, "focusRegionSize": 0.1, "multiplier": 0.3, "max_kernelsize": 6.0, **knobs_}
        g.add("node", "Bokeh", params, **inputs)
        return g.doc

    def test_tiles_equal_the_full_frame(self):
        extras = ({}, {"outputType": "Focal Distance Visualization"}, {"outputType": "Focus Mattes"}, {"kernelType": "Aperture Blades", "kNumSides": 5},
                  {"kSphAbb": 0.5, "kAspectRatio": 1.5, "kSoftness": 0.6}, {"bokeh_bloom": 1.0, "bloomThreshold": 0.3, "kChrAbb": 1.0},
                  {"outputType": "Lens Bloom", "bokeh_bloom": 1.0, "bloomThreshold": 0.3}, {"correctiveSlices": 1, "focusOutput": "Back"},
                  {"defocusMatteChannel": "rgba.alpha", "frontmultiplier": 2.0, "backmultiplier": 0.5, "focusRegionFalloff": "Smooth", "depthStyle": "1/z"})
        for masked in (False, True):
            for extra in extras:
                full = evaluator_pixels(self.doc(masked, **extra), "node")
                for edge in (7, 16, 32):
                    with self.subTest(masked=masked, extra=extra, edge=edge):
                        np.testing.assert_allclose(tile_pixels(self.doc(masked, **extra), "node", edge), full, atol=1e-5)

    def test_the_filter_preview_runs_the_full_frame_path(self):
        from nodebased.tileexec import TileExecutor
        document = dict(self.doc(False, outputType="Filter Shape"), view="node")
        self.assertFalse(TileExecutor(evaluator=Evaluator(), tile_edge=16).supports_tiled(document, "node"))


if __name__ == "__main__":
    unittest.main()
