"""2D behaviour lane 2, step D: knobs that the one-knob pixel probe cannot see, each shown to change the pixels in the state where it acts.

tools/nuke_fidelity_requirements.py moves one paired knob at a time on the fixture, with no mask, no optional input, no colour config and
the default operation. The knobs below act only in another state: with a mask wired, with a replace mode chosen, with an optional input
connected, with a second layer present, through a panel button, or with another knob moved first. Each has a NOT_PIXEL entry there that names
the state; this module is the evidence behind the entry. It also holds the split-knob checks for Clamp and the layout checks of step D.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_requirements as requirements  # noqa: E402
from nodebased import colorlookupknobs, core, knobs  # noqa: E402
from nodebased.core import SPECS  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.raster import Raster  # noqa: E402
from tests.test_2d_behaviour_color_a import colour_image, fixture_image, run_node  # noqa: E402
from tests.test_2d_behaviour_keyer import green_plate, kernel, two_plate_graph  # noqa: E402
from tests.test_2d_fidelity_b_chrome import matte, render  # noqa: E402
from tests.test_2d_parity_group_3b import evaluator_pixels, evaluator_raster  # noqa: E402

MAPPING = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))
NUKE = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))["nodes"]


def differs(test, a, b, message):
    test.assertFalse(np.array_equal(np.asarray(a), np.asarray(b)), message)


class NotPixelEntriesTests(unittest.TestCase):
    """Every knob step D lists as NOT_PIXEL names its state, and the probe leaves exactly those knobs out of `acts`."""

    LISTED = {
        "Shuffle": ("fromInput1", "fromInput2", "out2"), "ColorLookup": ("source", "target"),
        "HSVTool": ("srccolor", "dstcolor", "huerolloffrange", "satrolloffrange", "brtrolloffrange"), "SoftClip": ("softclip_min",),
        "Toe": ("invert",), "ChromaKeyer": ("_replaceAlphaDiff", "_replaceColor", "_replaceAmount"),
        "Keylight": ("despillBias", "insideReplaceMethod", "insideReplaceColour", "SourceXMethod", "SourceYMethod", "insideComponent", "outsideComponent"),
        "IBKColor": ("Size", "mult"), "IBKGizmo": ("ublu",), "Merge": ("screen_alpha",), "KeyMix": ("invertMask",),
    }       # OCIOLookTransform and OCIOFileTransform `invert` left this list in step F: the probe now runs them against the fixture config with a look and a file named

    def test_each_listed_knob_has_a_reason(self):
        for node, names in self.LISTED.items():
            for name in names:
                with self.subTest(node=node, knob=name):
                    reason = requirements.not_pixel_reason(node, name)
                    self.assertTrue(reason and len(reason) > 30, f"{node}.{name}: no NOT_PIXEL reason")

    def test_each_listed_knob_is_paired_in_the_mapping(self):
        """A reason is for a knob that exists and is paired: an unpaired knob needs no entry."""
        for node, names in self.LISTED.items():
            paired = {n for entry in MAPPING["pairs"][node]["knobs"] if entry["kind"] in requirements.PAIRED for n in entry["nuke"]}
            for name in names:
                with self.subTest(node=node, knob=name):
                    self.assertIn(name, paired)


# ---------------------------------------------------------------------------------------------------------- Shuffle
class ShuffleTwoInputKnobsTests(unittest.TestCase):
    """Shuffle's input choices and second output route Rasters and layers, which the array kernel the probe calls never sees."""

    def shuffle(self, params):
        g = two_plate_graph("Shuffle", params, second="A", first="image")
        return evaluator_pixels(g.doc, "node"), g

    def test_from_input_1_reads_the_other_input(self):
        before, _ = self.shuffle({})
        after, _ = self.shuffle({"fromInput1": "A"})
        differs(self, before, after, "Shuffle: fromInput1 left the pixels alone")

    def test_from_input_2_reads_the_other_input(self):
        base = {"in2": "rgba", "red_from": "R2"}
        before, _ = self.shuffle(base)
        after, _ = self.shuffle(dict(base, fromInput2="B"))
        differs(self, before, after, "Shuffle: fromInput2 left the pixels alone")

    def test_out2_writes_a_second_layer_routed_by_its_own_four_choices(self):
        _, g = self.shuffle({"out2": "extra", "red_from2": "A", "green_from2": "A", "blue_from2": "0", "alpha_from2": "1"})
        raster = evaluator_raster(g.doc, "node")
        layer = raster.layers["extra"].pixels
        np.testing.assert_array_equal(layer[..., 0], raster.pixels[..., 3])
        np.testing.assert_array_equal(layer[..., 2], 0.0)
        np.testing.assert_array_equal(layer[..., 3], 1.0)
        _, off = self.shuffle({})
        self.assertNotIn("extra", evaluator_raster(off.doc, "node").layers or {})


# ------------------------------------------------------------------------------------------------------- ColorLookup
class ColorLookupSetButtonTests(unittest.TestCase):
    """`source` and `target` feed the Set buttons (the guide: "Sets a source color for adding points to the curve editor"); the buttons move the pixels."""

    def lookup(self, params):
        return run_node("ColorLookup", params).pixels

    def test_set_rgb_adds_a_point_that_maps_source_to_target(self):
        base = dict(SPECS["ColorLookup"]["params"], source=0.5, target=0.8)
        curves = colorlookupknobs.set_points(base, ("red", "green", "blue"))
        plain = self.lookup({})
        mapped = self.lookup(dict(curves))
        differs(self, plain, mapped, "ColorLookup: the Set RGB point changed nothing")
        flat = np.full((4, 4, 4), 0.5, np.float32)
        out = Evaluator._kernel("ColorLookup", dict(SPECS["ColorLookup"]["params"], **curves), [flat])
        np.testing.assert_allclose(out[..., :3], 0.8, atol=1e-5)

    def test_the_target_decides_where_the_point_goes(self):
        a = colorlookupknobs.set_points(dict(SPECS["ColorLookup"]["params"], source=0.5, target=0.8), ("red",))
        b = colorlookupknobs.set_points(dict(SPECS["ColorLookup"]["params"], source=0.5, target=0.2), ("red",))
        differs(self, self.lookup(a), self.lookup(b), "ColorLookup: target left the pixels alone")

    def test_the_source_decides_where_the_point_goes(self):
        a = colorlookupknobs.set_points(dict(SPECS["ColorLookup"]["params"], source=0.3, target=0.8), ("red",))
        b = colorlookupknobs.set_points(dict(SPECS["ColorLookup"]["params"], source=0.6, target=0.8), ("red",))
        differs(self, self.lookup(a), self.lookup(b), "ColorLookup: source left the pixels alone")


# ------------------------------------------------------------------------------------------------------------ HSVTool
class HSVToolStateKnobsTests(unittest.TestCase):
    """Colour replacement needs `color_replace`; a rolloff needs a range narrower than the whole and an adjustment to roll off."""

    def hsv(self, params):
        return Evaluator._kernel("HSVTool", dict(SPECS["HSVTool"]["params"], **params), [colour_image()])

    def test_srccolor_picks_the_colour_that_is_replaced(self):
        base = {"color_replace": 1, "dstcolor_r": 0.1, "dstcolor_g": 0.9, "dstcolor_b": 0.1}
        img = np.zeros((3, 3, 4), np.float32)
        img[..., :] = [0.8, 0.2, 0.2, 1.0]
        match = Evaluator._kernel("HSVTool", dict(SPECS["HSVTool"]["params"], **base, srccolor_r=0.8, srccolor_g=0.2, srccolor_b=0.2), [img])
        miss = Evaluator._kernel("HSVTool", dict(SPECS["HSVTool"]["params"], **base, srccolor_r=0.2, srccolor_g=0.2, srccolor_b=0.8), [img])
        differs(self, match, miss, "HSVTool: srccolor left the pixels alone")

    def test_dstcolor_is_the_colour_it_becomes(self):
        base = {"color_replace": 1, "srccolor_r": 0.8, "srccolor_g": 0.2, "srccolor_b": 0.2}
        img = np.zeros((3, 3, 4), np.float32)
        img[..., :] = [0.8, 0.2, 0.2, 1.0]
        a = Evaluator._kernel("HSVTool", dict(SPECS["HSVTool"]["params"], **base, dstcolor_r=0.1, dstcolor_g=0.9, dstcolor_b=0.1), [img])
        b = Evaluator._kernel("HSVTool", dict(SPECS["HSVTool"]["params"], **base, dstcolor_r=0.1, dstcolor_g=0.1, dstcolor_b=0.9), [img])
        differs(self, a, b, "HSVTool: dstcolor left the pixels alone")

    def test_a_rolloff_softens_the_edge_of_its_range(self):
        for name, narrow, roll in (("hue", {"hue_range_min": 100.0, "hue_range_max": 140.0}, {"hue_rolloff": 30.0}),
                                    ("saturation", {"saturation_range_min": 0.4, "saturation_range_max": 0.7}, {"saturation_rolloff": 0.2}),
                                    ("brightness", {"brightness_range_min": 0.4, "brightness_range_max": 0.7}, {"brightness_rolloff": 0.2})):
            with self.subTest(range=name):
                base = dict(narrow, hue_rotation=60.0, sat_adjust=0.3)
                differs(self, self.hsv(base), self.hsv(dict(base, **roll)), f"HSVTool: the {name} rolloff left the pixels alone")


# ---------------------------------------------------------------------------------------------- SoftClip, Toe, KeyMix
class SoftClipToeKeyMixTests(unittest.TestCase):
    def test_softclip_min_sets_where_compression_starts(self):
        base = {"conversion": "logarithmic compress", "softclip_max": 4.0}
        img = np.zeros((2, 4, 4), np.float32)
        img[..., :3] = np.array([0.3, 0.6, 1.5, 3.0], np.float32)[None, :, None]
        img[..., 3] = 1.0
        low = Evaluator._kernel("SoftClip", dict(SPECS["SoftClip"]["params"], **base, softclip_min=0.2), [img])
        high = Evaluator._kernel("SoftClip", dict(SPECS["SoftClip"]["params"], **base, softclip_min=0.8), [img])
        differs(self, low, high, "SoftClip: softclip_min left the pixels alone")
        self.assertTrue(np.array_equal(Evaluator._kernel("SoftClip", dict(SPECS["SoftClip"]["params"], conversion="none", softclip_min=0.2), [img]),
                                       Evaluator._kernel("SoftClip", dict(SPECS["SoftClip"]["params"], conversion="none", softclip_min=0.8), [img])),
                        "SoftClip: with no conversion the limits are not used")

    def test_toe_invert_limits_the_effect_to_the_non_white_mask(self):
        base = {"toe_lift": 0.2, "maskChannelInput": "rgba.alpha"}
        differs(self, run_node("Toe", base).pixels, run_node("Toe", dict(base, invert_mask=1)).pixels, "Toe: invert left the pixels alone")
        self.assertTrue(np.array_equal(render("Toe", {"toe_lift": 0.2}).pixels, render("Toe", {"toe_lift": 0.2, "invert_mask": 1}).pixels),
                        "Toe: invert acts on the mask, so with no mask wired it must leave the pixels alone")

    def test_keymix_invert_swaps_where_a_shows(self):
        g = two_plate_graph("Keymix", {}, masked=True)
        g2 = two_plate_graph("Keymix", {"invert_mask": 1}, masked=True)
        differs(self, evaluator_pixels(g.doc, "node"), evaluator_pixels(g2.doc, "node"), "KeyMix: invertMask left the pixels alone")
        a, b, mask = fixture_image(3), fixture_image(5), matte(2)
        plain = Evaluator._kernel("Keymix", {"mix": 1.0, "invert_mask": 0}, [a, b, mask])
        flipped = Evaluator._kernel("Keymix", {"mix": 1.0, "invert_mask": 1}, [a, b, mask])
        m = mask[..., 3:4]
        np.testing.assert_allclose(plain[..., :3], a[..., :3] * m + b[..., :3] * (1 - m), atol=1e-5)
        np.testing.assert_allclose(flipped[..., :3], a[..., :3] * (1 - m) + b[..., :3] * m, atol=1e-5)


# ------------------------------------------------------------------------------------------------------------ keyers
class ChromaKeyerReplaceKnobsTests(unittest.TestCase):
    """The replace amount, colour and alpha-difference flag act once a replace mode other than none is chosen."""

    BASE = {"replace_mode": "edge hard color", "chroma_gain": 0.5, "white_point": 0.8, "black_point": 0.1}

    def key(self, **params):
        return kernel("ChromaKeyer", dict(self.BASE, **params), green_plate())

    def test_replace_colour_amount_and_alpha_difference_act_under_a_replace_mode(self):
        for change in ({"replace_red": 0.2, "replace_green": 0.3, "replace_blue": 0.9}, {"replace_amount": 0.3}, {"replace_alpha_diff": 1}):
            with self.subTest(change=change):
                differs(self, self.key(), self.key(**change), f"ChromaKeyer: {sorted(change)} left the pixels alone")

    def test_with_the_replace_mode_off_none_of_the_three_acts(self):
        off = {"replace_mode": "ignore"}
        base = self.key(**off)
        for change in ({"replace_red": 0.2, "replace_blue": 0.9}, {"replace_amount": 0.3}, {"replace_alpha_diff": 1}):
            np.testing.assert_array_equal(base, self.key(**off, **change))


class KeylightStateKnobsTests(unittest.TestCase):
    """Keylight's inside and outside matte, edge replacement and despill bias act only with those inputs wired or the biases unganged."""

    BASE = {"source_crop_left": 0.2, "source_crop_right": 0.9, "shadow_gain": 1.6, "bias_colours": 1, "clip_black": 0.1, "clip_white": 0.9,
            "screen_replace": "soft color", "inside_replace": "soft color"}

    def run_sk(self, params, inside=True):
        return kernel("ScreenKeyer", dict(self.BASE, **params), colour_image(), matte(2) if inside else None, matte(4) if inside else None, None, None, None)

    def test_each_knob_acts_in_its_state(self):
        states = {
            "despillBias": {"use_alpha_bias_for_despill": 0, "despill_bias_red": 0.9, "despill_bias_green": 0.1, "despill_bias_blue": 0.3},
            "insideReplaceMethod": {"inside_replace": "none"},
            "insideReplaceColour": {"inside_replace_red": 0.9, "inside_replace_green": 0.1},
            "SourceXMethod": {"source_x_method": "repeat"},
            "SourceYMethod": {"source_crop_bottom": 0.2, "source_y_method": "wrap"},
            "insideComponent": {"inside_component": "luminance"},
            "outsideComponent": {"outside_component": "luminance"},
        }
        gang_off = {"use_alpha_bias_for_despill": 0}
        for knob, change in states.items():
            with self.subTest(knob=knob):
                base = dict(gang_off) if knob == "despillBias" else {}
                differs(self, self.run_sk(base), self.run_sk(dict(base, **change)), f"Keylight: {knob} left the pixels alone")

    def test_the_inside_and_outside_components_need_their_inputs(self):
        for change in ({"inside_component": "luminance"}, {"outside_component": "luminance"}):
            np.testing.assert_array_equal(self.run_sk({}, inside=False), self.run_sk(change, inside=False))


class IBKStateKnobsTests(unittest.TestCase):
    def test_ibkcolor_size_and_lights(self):
        base = {"screen_type": "green", "fill_size": 3, "screen_erode": 1.0, "darks": 0.9, "lights": 1000.0, "patch_black": 1}
        src = green_plate()
        a = kernel("IBKColor", base, src)
        differs(self, a, kernel("IBKColor", dict(base, fill_size=1), src), "IBKColor: Size left the pixels alone")
        differs(self, a, kernel("IBKColor", dict(base, lights=0.5), src), "IBKColor: lights left the pixels alone")

    def test_ibkgizmo_background_luminance_needs_a_background_and_luminance_matching(self):
        fg = colour_image()
        plate = np.tile(np.array([0.15, 0.7, 0.3, 1.0], np.float32), (fg.shape[0], fg.shape[1], 1))
        bg = matte(8)
        base = {"screen_type": "green", "luminance_match": 1}
        differs(self, kernel("IBKGizmo", base, fg, plate, bg), kernel("IBKGizmo", dict(base, use_bg_luminance=1), fg, plate, bg), "IBKGizmo: ublu left the pixels alone")
        np.testing.assert_array_equal(kernel("IBKGizmo", base, fg, plate, None), kernel("IBKGizmo", dict(base, use_bg_luminance=1), fg, plate, None))


class MergeAlphaMaskingTests(unittest.TestCase):
    def test_screen_alpha_acts_for_operations_that_do_not_already_use_alpha(self):
        a, b = fixture_image(3), fixture_image(5)
        a[..., 3] = np.linspace(0.0, 1.0, a.shape[0] * a.shape[1], dtype=np.float32).reshape(a.shape[:2])
        base = {"operation": "plus"}
        differs(self, Evaluator._kernel("Merge", dict(SPECS["Merge"]["params"], **base), [a, b]),
                Evaluator._kernel("Merge", dict(SPECS["Merge"]["params"], **base, screen_alpha=1), [a, b]), "Merge: screen_alpha left the pixels alone")


# -------------------------------------------------------------------------------------------------------------- OCIO
class OcioDirectionAndInterpolationTests(unittest.TestCase):
    """A direction needs something to run backwards: a look for the look transform, a file for the file transform."""

    def test_look_transform_invert_runs_the_look_backwards(self):
        img = fixture_image(3).copy()
        img[..., :3] = np.abs(img[..., :3])
        base = {"src": "ACEScg", "dst": "ACEScg", "look": "ACES 1.3 Reference Gamut Compression"}
        fwd = Evaluator._kernel("OCIOLookTransform", dict(SPECS["OCIOLookTransform"]["params"], **base), [img])
        back = Evaluator._kernel("OCIOLookTransform", dict(SPECS["OCIOLookTransform"]["params"], **base, transform_direction="inverse"), [img])
        differs(self, fwd, back, "OCIOLookTransform: invert left the pixels alone")
        both = Evaluator._kernel("OCIOLookTransform", dict(SPECS["OCIOLookTransform"]["params"], **base, transform_direction="inverse"), [fwd])
        np.testing.assert_allclose(both[..., :3], img[..., :3], atol=2e-4)

    def test_file_transform_invert_runs_the_file_backwards(self):
        cube = str(Path(__file__).parent / "fixtures" / "red-gain-2.cube")
        img = np.full((3, 3, 4), 0.4, np.float32)
        img[..., 3] = 1.0
        base = {"path": cube}
        fwd = Evaluator._kernel("OCIOFileTransform", dict(SPECS["OCIOFileTransform"]["params"], **base), [img])
        back = Evaluator._kernel("OCIOFileTransform", dict(SPECS["OCIOFileTransform"]["params"], **base, transform_direction="inverse"), [img])
        differs(self, fwd, back, "OCIOFileTransform: invert left the pixels alone")
        round_trip = Evaluator._kernel("OCIOFileTransform", dict(SPECS["OCIOFileTransform"]["params"], **base, transform_direction="inverse"), [fwd])
        np.testing.assert_allclose(round_trip[..., :3], img[..., :3], atol=2e-3)

    def test_file_transform_interpolation_acts_on_a_3d_lut_where_the_corners_are_coupled(self):
        """OpenColorIO 2.5.2 renders nearest as linear and interpolates a LUT whose axes are independent the same way every time, so the knob is
        paired (Nuke's `interpolation`) and acts through tetrahedral and best on a 3D LUT with coupled corners (tests/test_2d_behaviour_ocio_p.py)."""
        import tempfile
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "corners.cube"
            cube.write_text("LUT_3D_SIZE 2\n" + "\n".join(" ".join(f"{v:.3f}" for v in rng.random(3)) for _ in range(8)) + "\n", encoding="utf-8")
            img = np.zeros((3, 3, 4), np.float32)
            img[..., :] = [0.3, 0.6, 0.45, 1.0]
            out = {m: Evaluator._kernel("OCIOFileTransform", dict(SPECS["OCIOFileTransform"]["params"], path=str(cube), file_interpolation=m), [img])
                   for m in ("nearest", "linear", "tetrahedral")}
        np.testing.assert_array_equal(out["nearest"], out["linear"])
        differs(self, out["linear"], out["tetrahedral"], "OCIOFileTransform: tetrahedral interpolation left a coupled 3D LUT alone")
        entries = MAPPING["pairs"]["OCIOFileTransform"]["knobs"]
        self.assertTrue(any(e["kind"] == "renamed" and e["nuke"] == ["interpolation"] and e["nodebased"] == ["file_interpolation"] for e in entries))


# ------------------------------------------------------------------------------------------------------------ Clamp split
class ClampSplitTests(unittest.TestCase):
    """Clamp's minimum and maximum are r g b a knobs in Nuke (docs/parity/nuke_split_knobs.json): each channel clamps to its own entry."""

    def image(self):
        rng = np.random.default_rng(11)
        pixels = rng.random((10, 14, 4), dtype=np.float32) * 3.0 - 1.0
        pixels[..., 3] = rng.random((10, 14), dtype=np.float32) * 3.0 - 1.0
        return pixels

    def test_clamp_declares_both_knobs_as_rgba_and_lays_them_out_split(self):
        self.assertEqual(core.SPLITTABLE["Clamp"], {"minimum": "rgba", "maximum": "rgba"})
        self.assertEqual(knobs.split_layout_problems(), [])
        self.assertEqual({g.params[0]: g.kind for g in knobs.knob_layout("Clamp") if g.params[0] in ("minimum", "maximum")}, {"minimum": "split", "maximum": "split"})

    def test_the_dispatcher_takes_a_list_of_four_and_refuses_the_wrong_length(self):
        d = core.Dispatcher()
        d.execute({"op": "create", "id": "c", "type": "Clamp", "params": {}})
        d.execute({"op": "set", "id": "c", "param": "minimum", "value": [0.1, 0.2, 0.3, 0.4]})
        self.assertEqual(d.document["nodes"]["c"]["params"]["minimum"], [0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(Exception):
            d.execute({"op": "set", "id": "c", "param": "maximum", "value": [0.1, 0.2]})

    def test_each_channel_clamps_to_its_own_entries(self):
        src = self.image()
        lo, hi = [-0.5, 0.0, 0.25, -1.0], [0.5, 1.0, 2.0, 0.75]
        out = render("Clamp", {"minimum": lo, "maximum": hi, "channels": "rgba"}, source=src).pixels
        for c in range(4):
            np.testing.assert_array_equal(out[..., c], np.clip(src[..., c], lo[c], hi[c]), err_msg=f"channel {c}")

    def test_channels_choose_which_entries_land(self):
        src = self.image()
        lo, hi = [-0.5, 0.0, 0.25, -1.0], [0.5, 1.0, 2.0, 0.75]
        for channels, picked in (("rgb", (0, 1, 2)), ("alpha", (3,))):
            out = render("Clamp", {"minimum": lo, "maximum": hi, "channels": channels}, source=src).pixels
            for c in range(4):
                expected = np.clip(src[..., c], lo[c], hi[c]) if c in picked else src[..., c]
                np.testing.assert_array_equal(out[..., c], expected, err_msg=f"{channels} channel {c}")

    def test_clamp_to_values_use_the_split_limits(self):
        src = self.image()
        lo, hi = [0.0, 0.2, 0.4, 0.0], [0.6, 0.8, 1.0, 1.0]
        out = render("Clamp", {"minimum": lo, "maximum": hi, "MinClampTo_enable": 1, "MinClampTo": -9.0, "MaxClampTo_enable": 1, "MaxClampTo": 9.0},
                     source=src).pixels
        for c in range(3):
            expected = np.where(src[..., c] < lo[c], -9.0, np.where(src[..., c] > hi[c], 9.0, src[..., c]))
            np.testing.assert_array_equal(out[..., c], expected.astype(np.float32), err_msg=f"channel {c}")

    def test_equal_entries_equal_the_unsplit_node(self):
        src = self.image()
        np.testing.assert_array_equal(render("Clamp", {"minimum": [0.1] * 4, "maximum": [0.9] * 4, "channels": "rgba"}, source=src).pixels,
                                      render("Clamp", {"minimum": 0.1, "maximum": 0.9, "channels": "rgba"}, source=src).pixels)

    def test_an_unsplit_clamp_keeps_its_pixels_bit_for_bit(self):
        src = self.image()
        for params in ({}, {"minimum": -0.3, "maximum": 1.7, "channels": "rgba"}, {"clamp_min": 0, "maximum": 0.5}, {"MinClampTo_enable": 1, "MinClampTo": 0.25}):
            with self.subTest(params=params):
                p = dict(SPECS["Clamp"]["params"], **params)
                expected = src.copy()
                lo = p["minimum"] if p["clamp_min"] else -np.inf
                hi = p["maximum"] if p["clamp_max"] else np.inf
                for c in Evaluator._CHANNEL_SETS[p["channels"]]:
                    value = np.clip(src[..., c], lo, hi)
                    if p["MinClampTo_enable"]:
                        value = np.where(src[..., c] < lo, p["MinClampTo"], value)
                    if p["MaxClampTo_enable"]:
                        value = np.where(src[..., c] > hi, p["MaxClampTo"], value)
                    expected[..., c] = value
                np.testing.assert_array_equal(render("Clamp", params, source=src).pixels, expected)

    def test_tiles_equal_the_whole_frame(self):
        from tests.test_2d_behaviour_color_a import tiled
        from tests.test_2d_parity_group_3b import Graph
        for masked in (False, True):
            g = Graph()
            g.add("plate", "Ramp", dict(width=61, height=47, color0_red=-0.4, color1_red=1.6, color0_green=0.2, color1_green=1.3, color0_alpha=0.1, color1_alpha=0.9))
            g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
            inputs = {"image": "plate"}
            if masked:
                inputs["mask"] = "matte"
            g.add("node", "Clamp", {"minimum": [0.1, 0.3, 0.0, 0.0], "maximum": [0.9, 0.7, 1.0, 0.5], "channels": "rgba"}, **inputs)
            full = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)

    def test_the_nodes_split_table_has_no_clamp_entry_left_open(self):
        table = requirements.load_split_table()["knobs"]["Clamp"]
        self.assertEqual(table, {"minimum": "rgba", "maximum": "rgba"})
        frac, detail = requirements.splits_component("Clamp", "Clamp", MAPPING["pairs"]["Clamp"]["knobs"], requirements.load_split_table())
        self.assertEqual((frac, detail["split_missing"], detail["split_partial"]), (1.0, [], []))


# ----------------------------------------------------------------------------------------------------------- layout
def main_labels(kind):
    """The main tab's labels in panel order, for the groups that hold a param (the order the user sees)."""
    return [g.label for g in knobs.knob_layout(kind) if g.kind != "legacy"]


def assert_in_order(test, kind, wanted):
    """The labels in `wanted` appear in the main tab in that order (other knobs may sit between them)."""
    labels = main_labels(kind)
    at = []
    for label in wanted:
        test.assertIn(label, labels, f"{kind}: no knob reads {label!r}")
        at.append(labels.index(label))
    test.assertEqual(at, sorted(at), f"{kind}: {wanted} are not in this order in {labels}")


class LayoutTests(unittest.TestCase):
    def test_clamp(self):
        assert_in_order(self, "Clamp", ["Channels", "Minimum", "Maximum", "Minimum clamp to", "Maximum clamp to"])   # each enable check box shares its value's row

    def test_color_lookup(self):
        assert_in_order(self, "ColorLookup", ["Channels", "Use precomputed table", "Range", "Master", "Red", "Green", "Blue", "Alpha", "Source", "Target"])
        paired = {n for e in MAPPING["pairs"]["ColorLookup"]["knobs"] if e["kind"] in requirements.PAIRED for n in e["nuke"]}
        self.assertFalse(paired & {"setRGB", "setRGBA", "setA"}, "the Set buttons are panel buttons, not params")

    def test_exposure(self):
        assert_in_order(self, "Exposure", ["Channels", "Black point", "Adjust in", "Gang", "Red", "Green", "Blue"])

    def test_shuffle_copy(self):
        assert_in_order(self, "ShuffleCopy", ["1 in", "2 in", "Out1 R", "Out2 R"])

    def test_match_grade(self):
        assert_in_order(self, "MatchGrade", ["Gain R", "Offset R", "Gamma R"])

    def test_keyer(self):
        assert_in_order(self, "Keyer", ["Input", "Output", "Combine", "Invert", "Operation", "A", "B", "C", "D"])

    def test_keylight(self):
        assert_in_order(self, "ScreenKeyer", ["View", "Unpremultiply result", "Screen colour", "Screen gain", "Screen balance", "Alpha bias colour",
                                              "Despill bias colour", "Use alpha bias for despill", "Screen pre-blur", "Clip black", "Clip white",
                                              "Clip rollback", "Screen shrink/grow", "Screen softness", "Screen despot black", "Screen despot white",
                                              "Screen replace", "Screen replace colour", "Midtones at", "Shadow gain", "Midtone gain", "Highlight gain",
                                              "Source alpha", "Inside replace", "Inside replace colour", "Crop x method", "Crop y method",
                                              "Crop edge colour", "Crop left", "Crop right", "Crop bottom", "Crop top", "Inside component",
                                              "Outside component"])

    def test_ibk_gizmo_uses_the_guides_text_for_ublu(self):
        self.assertIn("Use bkg luminnance", main_labels("IBKGizmo"))
        assert_in_order(self, "IBKGizmo", ["Screen type", "Red weight", "Blue/green weight", "Luminance match", "Luminance level", "Screen subtraction",
                                           "Use bkg luminnance"])
        self.assertEqual(NUKE["IBKGizmo"]["knobs"][[k["name"] for k in NUKE["IBKGizmo"]["knobs"]].index("ublu")]["label"], "use bkg luminnance")

    def test_add_mix(self):
        assert_in_order(self, "AddMix", ["A", "B", "Output", "Premultiplied", "Nonlinear"])

    def test_the_ocio_family_reads_as_nukes_labels(self):
        assert_in_order(self, "OCIOColorspace", ["Channels", "Input", "Output"])
        assert_in_order(self, "OCIOLookTransform", ["Channels", "Input", "Look", "Invert direction", "Output"])
        assert_in_order(self, "OCIODisplay", ["Display", "View", "Invert direction"])
        assert_in_order(self, "OCIOFileTransform", ["Channels", "File", "Invert direction"])

    def test_vectorfield_file(self):
        self.assertIn("Vectorfield file", main_labels("Vectorfield"))

    def test_the_audit_finds_no_order_or_label_difference_left_in_these_nodes(self):
        fx_nodes = ("Clamp", "ColorLookup", "Exposure", "ShuffleCopy", "MatchGrade", "Keyer", "Keylight", "IBKGizmo", "AddMix", "OCIOColorspace",
                    "OCIOLookTransform", "OCIODisplay", "OCIOFileTransform", "Vectorfield")
        for key in fx_nodes:
            with self.subTest(node=key):
                pair = MAPPING["pairs"][key]
                frac, detail = requirements.layout_component(pair["nodebased"], pair["knobs"], NUKE[key]["knobs"])
                if frac is None:
                    continue
                kept, total = detail["layout_order"]
                self.assertEqual(kept, total, f"{key}: only {kept} of {total} paired knobs in Nuke's order")
                self.assertEqual(detail["label_mismatches"], [], f"{key}: labels differ")


if __name__ == "__main__":
    unittest.main()
