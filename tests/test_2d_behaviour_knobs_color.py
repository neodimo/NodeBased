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
        "OCIOLookTransform": ("invert",), "OCIOFileTransform": ("invert",),
    }

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

    def test_file_transform_interpolation_is_unpaired_because_opencolorio_ignores_it(self):
        """OpenColorIO 2.5.2 returns the same pixels for every interpolation on a .cube, so the mapping lists the knob as unsupported and the param as extra."""
        import tempfile
        size = 3
        rows = [f"{(r / 2) ** 2:.6f} {(g / 2) ** 3:.6f} {(b / 2):.6f}" for b in range(size) for g in range(size) for r in range(size)]
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "bend.cube"
            cube.write_text(f"LUT_3D_SIZE {size}\n" + "\n".join(rows) + "\n", encoding="utf-8")
            img = np.zeros((3, 3, 4), np.float32)
            img[..., :] = [0.3, 0.7, 0.45, 1.0]
            out = [Evaluator._kernel("OCIOFileTransform", dict(SPECS["OCIOFileTransform"]["params"], path=str(cube), file_interpolation=m), [img])
                   for m in ("nearest", "linear", "tetrahedral")]
        np.testing.assert_array_equal(out[0], out[1])
        np.testing.assert_array_equal(out[1], out[2])
        entries = MAPPING["pairs"]["OCIOFileTransform"]["knobs"]
        self.assertTrue(any(e["kind"] == "unsupported" and e["nuke"] == ["interpolation"] for e in entries))
        self.assertTrue(any(e["kind"] == "extra" and e["nodebased"] == ["file_interpolation"] for e in entries))


if __name__ == "__main__":
    unittest.main()
