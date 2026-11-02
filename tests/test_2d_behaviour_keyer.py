"""2D behaviour lane 2, step C: the keyers and the channel nodes compute what Nuke computes.

Keyer, HueKeyer, ChromaKeyer, Difference, ScreenKeyer, IBKColor, IBKGizmo, Copy, ChannelMerge and Shuffle. The same four checks as steps A and B for
each node: the behaviour cases of tools/nuke_fidelity_cases_color.py pass where they are scored; every knob the mapping pairs with a NodeBased
param changes the pixels; the tile path (tiles of 7, 16 and 32, with and without a mask) equals the whole frame for every kernel that changed; and an
older document renders bit for bit as it did before (tests/legacy_behaviour_c.py holds the older kernels). Where the guide gives no formula the
cases are not scored, and the checks here say what the guide does state in words (a higher screen gain keys more, a hue the curve raises is black in
the alpha) as properties of the pixels.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as fidelity  # noqa: E402
from nodebased.core import SPECS, upgrade_document  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from tests import test_2d_behaviour_color_a as step_a  # noqa: E402
from tests.legacy_behaviour_c import Legacy  # noqa: E402
from tests.test_2d_behaviour_color_a import CHROME, KnobsActTests, TilesMatchTests, colour_image, fixture_image, run_node  # noqa: E402
from tests.test_2d_fidelity_b_chrome import H, W, image, matte, render  # noqa: E402
from tests.test_2d_parity_group_3b import evaluator_pixels  # noqa: E402

NUKE = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))
MAPPING = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))

STEP_NODES = ("Keyer", "HueKeyer", "ChromaKeyer", "Difference", "Keylight", "IBKColor", "IBKGizmo", "Copy", "ChannelMerge", "Shuffle")      # the Nuke names: ScreenKeyer is Keylight
KIND_OF = {"Keylight": "ScreenKeyer"}
OLD_KNOBS = step_a.OLD_KNOBS          # step A's older-document helper reads this table: step C's kinds are added to it
FLAT = '{{"interpolation":"linear","points":[[0,{v}],[360,{v}]]}}'


def curve(*points):
    return '{"interpolation":"linear","points":' + json.dumps([list(p) for p in points], separators=(",", ":")) + "}"


class OlderDocumentTests(unittest.TestCase):
    """Loading a document that predates the knob switches the node to its older behaviour; the pixels are those of the older kernel in tests/legacy_behaviour_c.py."""

    LEGACY = {}      # kind -> (Legacy kernel name, [param sets])

    def _check(self, kind):
        name, param_sets = self.LEGACY[kind]
        legacy = getattr(Legacy, name)
        spec = SPECS[kind]
        for params in param_sets:
            with self.subTest(kind=kind, params=params):
                doc = upgrade_document(step_a.old_document(kind, params))
                for knob in OLD_KNOBS[kind]:
                    self.assertIn(knob, doc["nodes"]["node"]["params"], f"{kind}: an older document gets {knob}")
                src = fixture_image()
                full = dict(spec["params"], **doc["nodes"]["node"]["params"])
                got = Evaluator._kernel(kind, full, [src, matte()])
                np.testing.assert_array_equal(got, Evaluator._apply_mask_mix(src, legacy(src, full), mask=matte(), mix=full.get("mix", 1.0)))
                plate, gate = evaluator_pixels(doc, "under"), evaluator_pixels(doc, "matte")
                np.testing.assert_array_equal(evaluator_pixels(doc, "node"), Evaluator._apply_mask_mix(plate, legacy(plate, full), mask=gate, mix=full.get("mix", 1.0)))


# ------------------------------------------------------------------------------------------------------- shared checks
class CaseTests(unittest.TestCase):
    """The behaviour cases for these nodes pass wherever they are scored."""

    NODE_KEYS = set(STEP_NODES)

    def test_every_scored_case_passes(self):
        mine = [c for c in fidelity.CASES if c.nuke in self.NODE_KEYS]
        self.assertTrue(mine, "no behaviour cases for the step's nodes")
        for case in mine:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                if case.unscored:
                    self.assertEqual(result["status"], "not scored")
                elif case.inexpressible:
                    self.assertEqual(result["status"], "inexpressible")
                else:
                    self.assertEqual(result["status"], "pass", f"{case.id}: {result}")

    def test_every_scored_case_compares_pixels_the_node_changes(self):
        """A case whose valid mask is empty, or whose reference equals its input everywhere, would pass whatever the kernel does."""
        fx = fidelity.make_fixtures()
        identity_by_design = {"keyer_output_none"}
        for case in fidelity.CASES:
            if case.nuke not in self.NODE_KEYS or case.unscored or case.inexpressible:
                continue
            ref, valid = case.expected(fx)
            source = fx[case.nb_inputs[-1 if case.nb_kind in ("Copy", "ChannelMerge") else 0]].astype(np.float64)
            with self.subTest(case=case.id):
                self.assertGreater(int(valid.sum()), 40, "too few pixels compared")
                if case.id not in identity_by_design:
                    self.assertGreater(int((valid & (np.abs(ref - source) > 1e-9)).sum()), 15, "the reference changes too few of the compared values")

    def test_unscored_cases_say_why(self):
        for case in fidelity.CASES:
            if case.nuke in self.NODE_KEYS and (case.unscored or case.inexpressible):
                self.assertGreater(len(case.unscored or case.inexpressible), 30, case.id)

    def test_every_case_names_a_source(self):
        for case in fidelity.CASES:
            if case.nuke in self.NODE_KEYS:
                self.assertIn(case.source, ("guide", "convention"), case.id)


# ================================================================================================================ Keyer
_KEYER_BASE = {"keyer_operation": "red", "range_a": 0.1, "range_b": 0.3, "range_c": 0.6, "range_d": 0.9}
_KEYER_CHANGES = {
    "operation": {"keyer_operation": "green"},
    "range#2": {"range_a": 0.4}, "range#3": {"range_b": 0.5}, "range#4": {"range_c": 0.8}, "range#5": {"range_d": 0.7},
    "invert": {"invert": 1},
    "input": {"key_input": "rgba.alpha"},
    "output": {"key_output": "rgba.red"},
    "combine": {"key_combine": "union"},
}


class KeyerKnobsActTests(KnobsActTests):
    BASES = {"Keyer": _KEYER_BASE}
    CHANGES = {"Keyer": _KEYER_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("Keyer")


class KeyerBehaviourTests(unittest.TestCase):
    def key(self, pixels, **p):
        return Evaluator._kernel("Keyer", dict(SPECS["Keyer"]["params"], **p), [np.asarray(pixels, np.float32).reshape(1, -1, 4)])[0]

    def test_the_four_points_make_a_trapezoid(self):
        values = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
        pixels = [[v, 0, 0, 1] for v in values]
        alpha = self.key(pixels, keyer_operation="red", range_a=0.1, range_b=0.3, range_c=0.6, range_d=0.9)[:, 3]
        np.testing.assert_allclose(alpha, [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0], atol=1e-6)

    def test_the_default_range_keys_zero_to_one(self):
        self.assertEqual([SPECS["Keyer"]["params"][k] for k in ("range_a", "range_b", "range_c", "range_d")], [0.0, 0.0, 1.0, 1.0])

    def test_the_colour_passes_through_untouched(self):
        src = fixture_image()
        np.testing.assert_array_equal(Evaluator._kernel("Keyer", dict(SPECS["Keyer"]["params"], **_KEYER_BASE), [src])[..., :3], src[..., :3])

    def test_luminance_weights_add_up_to_one(self):
        """The guide does not give the weights, but a grey pixel must key like the same grey on the red operation, so they sum to one."""
        grey = [[v, v, v, 1] for v in (0.1, 0.25, 0.5, 0.8)]
        common = dict(range_a=0.0, range_b=1.0, range_c=1.0, range_d=2.0)
        np.testing.assert_allclose(self.key(grey, keyer_operation="luminance", **common)[:, 3], self.key(grey, keyer_operation="red", **common)[:, 3], atol=1e-6)

    def test_saturation_is_zero_for_grey_and_one_for_a_pure_primary(self):
        pixels = [[0.5, 0.5, 0.5, 1], [1, 0, 0, 1]]
        low = self.key(pixels, keyer_operation="saturation", range_a=0.9, range_b=0.95, range_c=1.0, range_d=1.5)[:, 3]
        np.testing.assert_allclose(low, [0.0, 1.0], atol=1e-6)

    def test_the_screen_operations_are_high_on_their_own_colour(self):
        pixels = [[1, 0.1, 0.1, 1], [0.1, 1, 0.1, 1], [0.1, 0.1, 1, 1]]
        for operation, index in (("redscreen", 0), ("greenscreen", 1), ("bluescreen", 2)):
            alpha = self.key(pixels, keyer_operation=operation, range_a=0.5, range_b=0.6, range_c=1.0, range_d=1.5)[:, 3]
            with self.subTest(operation=operation):
                np.testing.assert_allclose(alpha, [1.0 if i == index else 0.0 for i in range(3)], atol=1e-6)

    def test_combine_is_the_smaller_or_the_larger_of_the_two(self):
        src = np.array([[0.5, 0.0, 0.0, 0.3], [0.5, 0.0, 0.0, 0.9]], np.float32).reshape(1, 2, 4)
        base = dict(keyer_operation="red", range_a=0.0, range_b=1.0, range_c=1.0, range_d=2.0, key_output="rgba.alpha")
        replaced = Evaluator._kernel("Keyer", dict(SPECS["Keyer"]["params"], **base), [src])[0, :, 3]
        np.testing.assert_allclose(replaced, [0.5, 0.5], atol=1e-6)
        inter = Evaluator._kernel("Keyer", dict(SPECS["Keyer"]["params"], **base, key_combine="intersect"), [src])[0, :, 3]
        union = Evaluator._kernel("Keyer", dict(SPECS["Keyer"]["params"], **base, key_combine="union"), [src])[0, :, 3]
        np.testing.assert_allclose(inter, [0.3, 0.5], atol=1e-6)
        np.testing.assert_allclose(union, [0.5, 0.9], atol=1e-6)


class KeyerTilesTests(TilesMatchTests):
    TILE_PARAMS = {"Keyer": [dict(_KEYER_BASE), dict(_KEYER_BASE, keyer_operation="max", invert=1),
                             dict(_KEYER_BASE, key_input="rgba.alpha", key_output="rgba.green", key_combine="intersect")]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("Keyer")


# ================================================================================================================ HueKeyer
OLD_KNOBS["HueKeyer"] = ("older_hue_window", "curve_amount", "curve_sat_thrsh")
_HK_BUMP = curve((0, 0), (60, 1), (180, 1), (300, 0), (360, 0))
_HK_BASE = {"curve_amount": _HK_BUMP}
_HK_CHANGES = {
    "channels": CHROME["channels"],
    "lut": {"curve_sat_thrsh": curve((0, 0.5), (360, 0.5))},
    "invert": {"invert": 0},
    "output": {"output_enabled": 0},
    "output#2": {"output_channel": "rgba.red"},
}


class HueKeyerKnobsActTests(KnobsActTests):
    BASES = {"HueKeyer": _HK_BASE}
    CHANGES = {"HueKeyer": _HK_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("HueKeyer")


class HueKeyerBehaviourTests(unittest.TestCase):
    def key(self, params, src=None):
        src = colour_image() if src is None else src
        return Evaluator._kernel("HueKeyer", dict(SPECS["HueKeyer"]["params"], **params), [src]), src

    def test_a_new_node_keys_nothing_and_inverts(self):
        params = SPECS["HueKeyer"]["params"]
        self.assertEqual((params["invert"], params["older_hue_window"]), (1, 0))
        out, src = self.key({})
        np.testing.assert_array_equal(out[..., 3], 1.0)
        np.testing.assert_array_equal(out[..., :3], src[..., :3])

    def test_a_hue_raised_to_one_is_black_in_the_alpha(self):
        pixels = np.array([[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]], np.float32)
        out, _ = self.key({"curve_amount": curve((0, 0), (90, 0), (120, 1), (150, 0), (360, 0))}, pixels)
        np.testing.assert_allclose(out[0, :, 3], [1.0, 0.0, 1.0], atol=1e-6)

    def test_the_alpha_is_the_curve_when_invert_is_off(self):
        pixels = np.array([[[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]]], np.float32)     # hues 0, 60, 120, 180, 240
        out, _ = self.key({"curve_amount": curve((0, 0.1), (60, 0.5), (120, 1.0), (240, 0.2), (360, 0.1)), "invert": 0}, pixels)
        np.testing.assert_allclose(out[0, :, 3], [0.1, 0.5, 1.0, 1.0 - 0.8 * 60 / 120, 0.2], atol=1e-6)

    def test_a_saturation_threshold_rejects_the_hues_with_low_saturation(self):
        flat_one = curve((0, 1), (360, 1))
        plain, src = self.key({"curve_amount": flat_one})
        guarded, _ = self.key({"curve_amount": flat_one, "curve_sat_thrsh": curve((0, 0.3), (360, 0.3))})
        hsv_sat = (src[..., :3].max(-1) - src[..., :3].min(-1)) / src[..., :3].max(-1)
        np.testing.assert_allclose(plain[..., 3], 0.0, atol=1e-6)
        np.testing.assert_allclose(guarded[..., 3][hsv_sat <= 0.3], 1.0, atol=1e-6)         # below the threshold: not keyed
        np.testing.assert_allclose(guarded[..., 3][hsv_sat >= 0.6], 0.0, atol=1e-6)         # twice the threshold: keyed as without it
        between = (hsv_sat > 0.3) & (hsv_sat < 0.6)
        self.assertTrue(np.all((guarded[..., 3][between] >= 0.0) & (guarded[..., 3][between] <= 1.0)))

    def test_a_higher_threshold_keys_less(self):
        flat_one = curve((0, 1), (360, 1))
        keyed = [float((1.0 - self.key({"curve_amount": flat_one, "curve_sat_thrsh": curve((0, t), (360, t))})[0][..., 3]).sum()) for t in (0.0, 0.2, 0.4, 0.6)]
        self.assertTrue(all(a > b for a, b in zip(keyed, keyed[1:])), keyed)

    def test_the_curve_wraps_through_red(self):
        pixels = np.array([[[1, 0.02, 0, 1], [1, 0, 0.02, 1]]], np.float32)      # hue 1.2 and 358.8
        out, _ = self.key({"curve_amount": curve((0, 1), (30, 0), (330, 0), (360, 1)), "invert": 0}, pixels)
        np.testing.assert_allclose(out[0, :, 3], [1 - 1.2 / 30, 1 - 1.2 / 30], atol=2e-3)

    def test_the_older_hue_window_is_still_there(self):
        red = np.array([[[1, 0, 0, 1]]], np.float32)
        green = np.array([[[0, 1, 0, 1]]], np.float32)
        window = {"older_hue_window": 1, "invert": 0, "hue_center": 0.0, "hue_width": 30.0, "hue_softness": 15.0}
        self.assertEqual(float(self.key(window, red)[0][0, 0, 3]), 1.0)
        self.assertEqual(float(self.key(window, green)[0][0, 0, 3]), 0.0)


class HueKeyerTilesTests(TilesMatchTests):
    TILE_PARAMS = {"HueKeyer": [dict(_HK_BASE), dict(_HK_BASE, curve_sat_thrsh=curve((0, 0.4), (180, 0.1), (360, 0.4)), invert=0),
                                dict(_HK_BASE, output_channel="rgba.green"), {"older_hue_window": 1, "invert": 0, "hue_center": 40.0, "hue_width": 80.0}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("HueKeyer")


class HueKeyerOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"HueKeyer": ("_hue_keyer", [{"hue_center": 40.0, "hue_width": 80.0, "hue_softness": 20.0, "invert": 0},
                                          {"hue_center": 200.0, "hue_width": 50.0, "hue_softness": 10.0, "sat_min": 0.2, "sat_max": 0.9, "invert": 1, "mix": 0.6},
                                          {"hue_center": 120.0, "hue_width": 90.0, "invert": 0, "output_channel": "rgba.blue"}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("HueKeyer")

    def test_an_older_document_gets_the_window_switch_on(self):
        doc = upgrade_document(step_a.old_document("HueKeyer", {"hue_center": 40.0}))
        self.assertEqual(doc["nodes"]["node"]["params"]["older_hue_window"], 1)


# ================================================================================================================ plates for the screen keyers
def green_plate(blue=False):
    """A screen with a skin-toned patch in it: the screen is (0.1, 0.8, 0.2) (or the blue one), the patch (0.8, 0.6, 0.5), with a one-pixel soft edge."""
    plate = np.zeros((12, 16, 4), np.float32)
    plate[...] = [0.1, 0.8, 0.2, 1.0] if not blue else [0.2, 0.3, 0.9, 1.0]
    plate[3:9, 5:11] = [0.8, 0.6, 0.5, 1.0]
    plate[3:9, 4] = 0.5 * plate[3:9, 3] + 0.5 * plate[3:9, 5]
    return plate


def kernel(kind, params, *inputs):
    return Evaluator._kernel(kind, dict(SPECS[kind]["params"], **params), list(inputs))


# ================================================================================================================ ChromaKeyer
_CK_BASE = {"replace_mode": "edge hard color", "chroma_gain": 0.5, "white_point": 0.8, "black_point": 0.1, "override_despill_bias": 1, "despill_bias": 0.7}
_CK_CHANGES = {
    "_view": {"key_view": "intermediate result"},
    "_screenColor": {"key_red": 0.7, "key_green": 0.2, "key_blue": 0.3},
    "_screenGain": {"screen_gain": 2.5},
    "_screenBalance": {"screen_balance": 0.9},
    "_chromaGain": {"chroma_gain": 0.9},
    "_whitePoint": {"white_point": 0.5},
    "_blackPoint": {"black_point": 0.3},
    "_alphaBias": {"alpha_bias_red": 0.9, "alpha_bias_green": 0.2, "alpha_bias_blue": 0.2},
    "_insideMask": {"inside_mask": "source alpha"},
    "_overrideDespillBias": {"override_despill_bias": 0},
    "despillBias": {"despill_bias": 0.1},
    "_replace": {"replace_mode": "edge soft color"},
    "_replaceAlphaDiff": {"replace_alpha_diff": 1},
    "_replaceColor": {"replace_red": 0.2, "replace_green": 0.3, "replace_blue": 0.9},
    "_replaceAmount": {"replace_amount": 0.3},
    "_preMultiply": {"premultiply": 0},
}


class ChromaKeyerKnobsActTests(KnobsActTests):
    BASES = {"ChromaKeyer": _CK_BASE}
    CHANGES = {"ChromaKeyer": _CK_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("ChromaKeyer")


class ChromaKeyerBehaviourTests(unittest.TestCase):
    """What the page states in words: the screen colour is transparent, more screen gain keys more, more chroma gain gives a less transparent foreground, the white
    and black points cut the matte, premultiply multiplies the colour by the matte, despill limits the screen channel."""

    def key(self, params=None, src=None):
        return kernel("ChromaKeyer", dict(params or {}), colour_image() if src is None else src)

    def test_the_screen_colour_is_transparent_and_a_grey_stays_opaque(self):
        screen = np.array([[[0.1, 0.8, 0.2, 1.0], [0.5, 0.5, 0.5, 1.0], [0.8, 0.6, 0.5, 1.0]]], np.float32)
        out = self.key(src=screen)
        np.testing.assert_allclose(out[0, :, 3], [0.0, 1.0, 1.0], atol=1e-6)
        shaded = screen * np.array([0.4, 0.4, 0.4, 1.0], np.float32)             # the same screen in shadow keys the same
        self.assertEqual(float(self.key(src=shaded)[0, 0, 3]), 0.0)

    def test_the_default_is_premultiplied_as_the_page_says(self):
        self.assertEqual(SPECS["ChromaKeyer"]["params"]["premultiply"], 1)
        on, off = self.key({"premultiply": 1}), self.key({"premultiply": 0})
        np.testing.assert_allclose(on[..., :3], off[..., :3] * off[..., 3:4], atol=1e-6)
        np.testing.assert_array_equal(on[..., 3], off[..., 3])

    def test_a_higher_screen_gain_never_keys_less(self):
        alphas = [self.key({"screen_gain": gain})[..., 3] for gain in (0.5, 1.0, 2.0, 4.0)]
        for before, after in zip(alphas, alphas[1:]):
            self.assertTrue(np.all(after <= before + 1e-6))
            self.assertLess(float(after.mean()), float(before.mean()))

    def test_a_higher_chroma_gain_never_gives_a_more_transparent_matte(self):
        alphas = [self.key({"chroma_gain": gain})[..., 3] for gain in (0.0, 0.3, 0.6, 0.9)]
        for before, after in zip(alphas, alphas[1:]):
            self.assertTrue(np.all(after >= before - 1e-6))

    def test_the_balance_moves_the_matte(self):
        low, mid, high = (self.key({"screen_balance": b})[..., 3] for b in (0.05, 0.5, 0.95))
        self.assertFalse(np.array_equal(low, mid))
        self.assertFalse(np.array_equal(mid, high))

    def test_alphas_above_the_white_point_are_one_and_below_the_black_point_zero(self):
        base = self.key()[..., 3]
        high = self.key({"white_point": 0.6})[..., 3]
        low = self.key({"black_point": 0.4})[..., 3]
        np.testing.assert_allclose(high[base >= 0.6], 1.0, atol=1e-6)
        np.testing.assert_allclose(low[base <= 0.4], 0.0, atol=1e-6)
        self.assertTrue(np.any((base > 0.0) & (base < 0.4)))

    def test_the_despill_limits_the_screen_channel(self):
        plate = green_plate()
        out = self.key({"premultiply": 0, "override_despill_bias": 1, "despill_bias": 0.5}, plate)
        limit = 0.5 * plate[..., 0] + 0.5 * plate[..., 2]
        np.testing.assert_allclose(out[..., 1], np.minimum(plate[..., 1], limit), atol=1e-6)
        np.testing.assert_array_equal(out[..., 0], plate[..., 0])

    def test_the_intermediate_result_keeps_the_source_colour(self):
        plate = green_plate()
        out = self.key({"key_view": "intermediate result"}, plate)
        np.testing.assert_array_equal(out[..., :3], plate[..., :3])
        np.testing.assert_array_equal(out[..., 3], self.key({"premultiply": 0}, plate)[..., 3])

    def test_the_inside_mask_adds_up_the_earlier_keys(self):
        plate = green_plate()
        plate[..., 3] = 0.5
        ignore, added = self.key({"inside_mask": "ignore"}, plate)[..., 3], self.key({"inside_mask": "source alpha"}, plate)[..., 3]
        np.testing.assert_allclose(added, ignore + 0.5 * (1.0 - ignore), atol=1e-6)

    def test_the_replace_colour_is_added_where_the_alpha_was_raised(self):
        """Page: 'edge hard color - the despilled image has a corresponding amount of the replace color added for any increase in alpha'; ignore leaves it untouched;
        the add-in matte fix extends that to the areas the white and black points corrected."""
        raw = self.key({"premultiply": 0, "replace_mode": "ignore", "chroma_gain": 0.6})
        for mode in ("ignore", "edge hard color"):
            hard = self.key({"premultiply": 0, "replace_mode": mode, "chroma_gain": 0.6})
            raised = hard[..., 3] > self.key({"premultiply": 0, "chroma_gain": 0.0})[..., 3] + 1e-6
            self.assertTrue(raised.any())
            if mode == "ignore":
                np.testing.assert_array_equal(hard[..., :3], raw[..., :3])
            else:
                self.assertTrue(np.any(hard[..., :3][raised] > raw[..., :3][raised] + 1e-6))
                np.testing.assert_allclose(hard[..., :3][~raised], raw[..., :3][~raised], atol=1e-6)
        pushed = self.key({"premultiply": 0, "replace_mode": "edge hard color", "white_point": 0.7})
        fixed = self.key({"premultiply": 0, "replace_mode": "edge hard color", "white_point": 0.7, "replace_alpha_diff": 1})
        self.assertTrue(np.any(fixed[..., :3] > pushed[..., :3] + 1e-6))


class ChromaKeyerTilesTests(TilesMatchTests):
    TILE_PARAMS = {"ChromaKeyer": [dict(_CK_BASE), {"screen_balance": 0.9, "screen_gain": 2.0, "key_view": "intermediate result"},
                                   dict(_CK_BASE, replace_mode="edge soft color", replace_alpha_diff=1, premultiply=0)]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("ChromaKeyer")


# ================================================================================================================ Difference
def plates_ab():
    a = fixture_image(3)
    b = fixture_image(5)
    return a, b


class DifferenceKnobsActTests(KnobsActTests):
    BASES = {"Difference": {"offset": 0.2, "gain": 1.5}}
    CHANGES = {"Difference": {"offset": {"offset": 0.05}, "gain": {"gain": 3.0}, "output": {"output_enabled": 0}, "output#2": {"output_channel": "rgba.green"}}}

    def _differs(self, kind, nuke_knob):
        a, b = plates_ab()
        base = dict(self.BASES[kind])
        before = kernel(kind, base, a, b)
        after = kernel(kind, dict(base, **self.CHANGES[kind][nuke_knob]), a, b)
        self.assertFalse(np.array_equal(before, after), f"{kind}: changing {nuke_knob} left the pixels alone")

    def test_every_paired_knob_acts(self):
        self._check_kind("Difference")


class DifferenceBehaviourTests(unittest.TestCase):
    def test_identical_pictures_have_no_matte(self):
        a, _ = plates_ab()
        np.testing.assert_array_equal(kernel("Difference", {}, a, a)[..., 3], 0.0)

    def test_a_negative_offset_lifts_the_matte_by_that_much(self):
        a, _ = plates_ab()
        np.testing.assert_allclose(kernel("Difference", {"offset": -0.3}, a, a)[..., 3], 0.3, atol=1e-6)

    def test_the_matte_grows_with_the_difference_and_shrinks_with_the_offset(self):
        a, b = plates_ab()
        plain = kernel("Difference", {}, a, b)[..., 3]
        self.assertGreater(float(plain.mean()), 0.2)
        self.assertTrue(np.all(kernel("Difference", {"offset": 0.3}, a, b)[..., 3] <= plain + 1e-6))
        self.assertTrue(np.all(kernel("Difference", {"gain": 2.0}, a, b)[..., 3] >= plain - 1e-6))

    def test_the_subject_is_where_b_differs_from_the_plate(self):
        plate = np.full((6, 6, 4), 0.4, np.float32)
        subject = plate.copy()
        subject[2:4, 2:4, :3] = 0.9
        matte_ = kernel("Difference", {}, plate, subject)[..., 3]
        self.assertTrue(np.all(matte_[2:4, 2:4] > 0.4))
        self.assertEqual(float(matte_[0, 0]), 0.0)

    def test_the_output_channel_takes_the_matte(self):
        a, b = plates_ab()
        out = kernel("Difference", {"output_channel": "rgba.red"}, a, b)
        np.testing.assert_array_equal(out[..., 3], b[..., 3])
        np.testing.assert_array_equal(out[..., 0], kernel("Difference", {}, a, b)[..., 3])


class DifferenceTilesTests(TilesMatchTests):
    def test_tiles_equal_the_whole_frame(self):
        from tests.test_2d_behaviour_color_a import tiled
        from tests.test_2d_parity_group_3b import Graph
        for params in ({"offset": 0.1, "gain": 2.0}, {"output_channel": "rgba.blue"}):
            for masked in (False, True):
                g = Graph()
                g.add("a", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33, red=0.9, green=0.5, blue=0.2, alpha=0.8, softness=0.5))
                g.add("b", "Ramp", dict(width=61, height=47))
                g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
                inputs = {"A": "a", "B": "b"}
                if masked:
                    inputs["mask"] = "matte"
                g.add("node", "Difference", params, **inputs)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


# ================================================================================================================ ScreenKeyer (Keylight)
_SK_BASE = {"source_crop_left": 0.2, "source_crop_right": 0.9, "shadow_gain": 1.6, "bias_colours": 1, "clip_black": 0.1, "clip_white": 0.9,
            "screen_replace": "soft color", "inside_replace": "soft color"}
_SK_CHANGES = {
    "show": {"keyer_view": "status"},
    "screenColor": {"screen_red": 0.6, "screen_green": 0.3, "screen_blue": 0.2},
    "screenGain": {"screen_gain": 2.5}, "screenBalance": ({"bias_colours": 0}, {"screen_balance": 0.9}),
    "alphaBias": {"alpha_bias_red": 0.9, "alpha_bias_green": 0.2, "alpha_bias_blue": 0.3},
    "despillBias": ({"use_alpha_bias_for_despill": 0}, {"despill_bias_red": 0.9, "despill_bias_green": 0.1, "despill_bias_blue": 0.3}),
    "gangBiases": ({"use_alpha_bias_for_despill": 0, "despill_bias_red": 0.8, "alpha_bias_red": 0.3}, {"use_alpha_bias_for_despill": 1}),
    "preBlur": {"screen_preblur": 2.0},
    "screenClipMin": {"clip_black": 0.4}, "screenClipMax": {"clip_white": 0.6}, "screenClipRollback": {"clip_rollback": 0.6},
    "screenGrowShrink": {"screen_shrink": 2.0}, "screenSoftness": {"screen_softness": 2.0},
    "screenDespotBlack": {"screen_despot_black": 3.0}, "screenDespotWhite": {"screen_despot_white": 3.0},
    "screenReplaceMethod": {"screen_replace": "none"}, "screenReplaceColour": {"screen_replace_red": 0.9, "screen_replace_blue": 0.1},
    "midPoint": {"mid_point": 0.2}, "lowGain": {"shadow_gain": 0.3}, "midGain": {"midtone_gain": 1.8}, "highGain": {"highlight_gain": 2.0},
    "sourceAlphaHandling": {"source_alpha": "normal"}, "unPreMultiply": {"unpremultiply_result": 0},
    "insideReplaceMethod": {"inside_replace": "none"}, "insideReplaceColour": {"inside_replace_red": 0.9, "inside_replace_green": 0.1},
    "SourceXMethod": {"source_x_method": "repeat"}, "SourceYMethod": ({"source_crop_bottom": 0.2}, {"source_y_method": "wrap"}),
    "SourceEdgeColour": {"source_edge_red": 1.0, "source_edge_green": 0.5},
    "SourceCropL": {"source_crop_left": 0.4}, "SourceCropR": {"source_crop_right": 0.6}, "SourceCropB": {"source_crop_bottom": 0.3}, "SourceCropT": {"source_crop_top": 0.7},
    "insideComponent": {"inside_component": "luminance"}, "outsideComponent": {"outside_component": "luminance"},
}


class ScreenKeyerKnobsActTests(unittest.TestCase):
    def run_sk(self, params):
        src = colour_image()
        inside, outside = matte(2), matte(4)
        return kernel("ScreenKeyer", params, src, inside, outside, None, None, None)

    def test_every_paired_knob_acts(self):
        pair = MAPPING["pairs"]["Keylight"]
        seen = set()
        for entry in pair["knobs"]:
            if entry["kind"] not in ("same", "renamed", "components"):
                continue
            for nuke in entry["nuke"]:
                seen.add(nuke)
                with self.subTest(knob=nuke):
                    self.assertIn(nuke, _SK_CHANGES, f"ScreenKeyer: the paired knob {nuke} has no acting check")
                    change = _SK_CHANGES[nuke]
                    base_updates, change = change if isinstance(change, tuple) else ({}, change)
                    base = dict(_SK_BASE, **base_updates)
                    self.assertFalse(np.array_equal(self.run_sk(base), self.run_sk(dict(base, **change))), f"ScreenKeyer: changing {nuke} left the pixels alone")
        self.assertEqual(seen, set(_SK_CHANGES))


class ScreenKeyerBehaviourTests(unittest.TestCase):
    """The introduction page: clip black makes every pixel below it 100% transparent, clip white makes the grey in the white area opaque; the screen colour goes."""

    def test_clip_black_and_clip_white_cut_the_screen_matte(self):
        src = colour_image()
        base = kernel("ScreenKeyer", {}, src)[..., 3]
        low = kernel("ScreenKeyer", {"clip_black": 0.3}, src)[..., 3]
        high = kernel("ScreenKeyer", {"clip_white": 0.7}, src)[..., 3]
        np.testing.assert_allclose(low[base <= 0.3], 0.0, atol=1e-6)
        np.testing.assert_allclose(high[base >= 0.7], 1.0, atol=1e-6)
        self.assertTrue(np.any((base > 0.0) & (base < 0.3)) and np.any((base > 0.7) & (base < 1.0)))

    def test_the_screen_colour_is_transparent_and_the_foreground_opaque(self):
        out = kernel("ScreenKeyer", {"screen_red": 0.1, "screen_green": 0.8, "screen_blue": 0.2}, green_plate())
        self.assertEqual(float(out[0, 0, 3]), 0.0)
        self.assertEqual(float(out[5, 7, 3]), 1.0)

    def test_a_higher_screen_gain_keys_more(self):
        means = [float(kernel("ScreenKeyer", {"screen_gain": g}, colour_image())[..., 3].mean()) for g in (0.5, 1.0, 2.0, 4.0)]
        self.assertTrue(all(a >= b for a, b in zip(means, means[1:])), means)


class ScreenKeyerTilesTests(TilesMatchTests):
    TILE_PARAMS = {"ScreenKeyer": [{"screen_softness": 2.0, "clip_black": 0.1}, {"screen_shrink": 2.0, "screen_despot_black": 3.0}, {"screen_preblur": 2.0, "keyer_view": "status"}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("ScreenKeyer")


# ================================================================================================================ IBKColor and IBKGizmo
class IBKColorKnobsActTests(unittest.TestCase):
    CHANGES = {"Size": {"fill_size": 1}, "erode": {"screen_erode": 3.0}, "off": {"darks": 0.5}, "mult": {"lights": 0.5}, "multi": {"patch_black": 0},
               "screen_type": {"screen_type": "blue"}}

    def plate(self):
        """A bluish-green screen with a foreground patch and a few off-colour pixels."""
        return green_plate()

    def test_every_paired_knob_acts(self):
        base = {"screen_type": "green", "fill_size": 3, "screen_erode": 1.0, "darks": 0.9, "lights": 1000.0, "patch_black": 1}
        src = self.plate()
        for nuke, params in KnobsActTests.paired("IBKColor"):
            with self.subTest(knob=nuke):
                self.assertIn(nuke, self.CHANGES, f"IBKColor: the paired knob {nuke} has no acting check")
                self.assertFalse(np.array_equal(kernel("IBKColor", base, src), kernel("IBKColor", dict(base, **self.CHANGES[nuke]), src)))


class IBKColorBehaviourTests(unittest.TestCase):
    def test_a_green_screen_keeps_its_green_and_the_foreground_is_filled_from_the_screen_around_it(self):
        out = kernel("IBKColor", {"screen_type": "green", "fill_size": 10, "screen_erode": 0.0}, green_plate())
        np.testing.assert_allclose(out[0, 0, :3], [0.1, 0.8, 0.2], atol=1e-6)
        inside = out[5, 7, :3]
        self.assertTrue(inside[1] > inside[0] + 0.3 and inside[1] > inside[2] + 0.3, inside)       # screen coloured, though the foreground stood there
        self.assertEqual(float(out[5, 7, 3]), 1.0)

    def test_more_size_reaches_further_into_the_foreground(self):
        plate = green_plate()
        unresolved = [int(np.sum(np.all(kernel("IBKColor", {"screen_type": "green", "fill_size": size, "screen_erode": 0.0, "patch_black": 0}, plate)[..., :3] == 0.0, axis=-1)))
                      for size in (1, 2, 3, 10)]
        self.assertTrue(all(a >= b for a, b in zip(unresolved, unresolved[1:])), unresolved)
        self.assertEqual(unresolved[-1], 0)
        self.assertGreater(unresolved[0], 0)

    def test_the_blue_screen_type_looks_for_blue(self):
        out = kernel("IBKColor", {"screen_type": "blue", "fill_size": 10, "screen_erode": 0.0}, green_plate(blue=True))
        np.testing.assert_allclose(out[0, 0, :3], [0.2, 0.3, 0.9], atol=1e-6)
        inside = out[5, 7, :3]
        self.assertTrue(inside[2] > inside[0] + 0.3 and inside[2] > inside[1] + 0.3, inside)


class IBKColorTilesTests(unittest.TestCase):
    def test_ibkcolor_is_run_whole_by_the_tile_path(self):
        from nodebased.tileexec import SUPPORTED_TILED_KINDS
        from tests.test_2d_behaviour_color_a import tiled
        from tests.test_2d_parity_group_3b import Graph
        self.assertIn("IBKColor", SUPPORTED_TILED_KINDS)
        for params in ({"screen_type": "green", "fill_size": 5}, {"screen_type": "blue", "screen_erode": 2.0, "patch_black": 0}):
            for masked in (False, True):
                g = Graph()
                g.add("plate", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33, red=0.2, green=0.8, blue=0.3, alpha=1.0, softness=0.5))
                g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
                inputs = {"c": "plate"}
                if masked:
                    inputs["mask"] = "matte"
                g.add("node", "IBKColor", params, **inputs)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


class IBKGizmoKnobsActTests(unittest.TestCase):
    CHANGES = {"st": {"screen_type": "blue"}, "lm_enable": {"luminance_match": 1}, "luma": ({"luminance_match": 1, "luminance_level": 0.2}, {"luminance_level": 1.0}),
               "ublu": ({"luminance_match": 1}, {"use_bg_luminance": 1}), "ss": {"screen_subtraction": 0}, "red_weight": {"red_weight": 0.9}, "blue_green_weight": {"blue_green_weight": 0.1}}

    def test_every_paired_knob_acts(self):
        fg = colour_image()
        plate = np.tile(np.array([0.15, 0.7, 0.3, 1.0], np.float32), (fg.shape[0], fg.shape[1], 1))
        bg = matte(8)
        for nuke, params in KnobsActTests.paired("IBKGizmo"):
            with self.subTest(knob=nuke):
                self.assertIn(nuke, self.CHANGES, f"IBKGizmo: the paired knob {nuke} has no acting check")
                change = self.CHANGES[nuke]
                base_updates, change = change if isinstance(change, tuple) else ({}, change)
                base = dict({"screen_type": "green"}, **base_updates)
                self.assertFalse(np.array_equal(kernel("IBKGizmo", base, fg, plate, bg), kernel("IBKGizmo", dict(base, **change), fg, plate, bg)))


class IBKGizmoBehaviourTests(unittest.TestCase):
    """The tutorial: IBKGizmo makes 'essentially a difference matte' of the foreground against the smoothed screen plate."""

    def test_the_screen_is_transparent_and_the_foreground_opaque(self):
        fg = green_plate()
        plate = kernel("IBKColor", {"screen_type": "green", "fill_size": 10, "screen_erode": 0.0}, fg)
        out = kernel("IBKGizmo", {"screen_type": "green"}, fg, plate)
        self.assertAlmostEqual(float(out[0, 0, 3]), 0.0, places=4)
        self.assertAlmostEqual(float(out[5, 7, 3]), 1.0, places=4)

    def test_a_foreground_equal_to_the_plate_has_no_matte_and_one_that_is_not_the_screen_has_all_of_it(self):
        plate = np.tile(np.array([0.15, 0.7, 0.3, 1.0], np.float32), (4, 4, 1))
        np.testing.assert_allclose(kernel("IBKGizmo", {"screen_type": "green"}, plate, plate)[..., 3], 0.0, atol=1e-6)
        grey = np.tile(np.array([0.5, 0.5, 0.5, 1.0], np.float32), (4, 4, 1))
        np.testing.assert_allclose(kernel("IBKGizmo", {"screen_type": "green"}, grey, plate)[..., 3], 1.0, atol=1e-6)

    def test_screen_subtraction_takes_the_screen_out_of_the_colour(self):
        fg = green_plate()
        fg[0:3, 0:3] = 0.5 * fg[0:3, 0:3] + 0.5 * np.array([0.8, 0.6, 0.5, 1.0], np.float32)       # a half-transparent edge
        plate = kernel("IBKColor", {"screen_type": "green", "fill_size": 10, "screen_erode": 0.0}, green_plate())
        on = kernel("IBKGizmo", {"screen_type": "green", "screen_subtraction": 1}, fg, plate)
        off = kernel("IBKGizmo", {"screen_type": "green", "screen_subtraction": 0}, fg, plate)
        np.testing.assert_array_equal(on[..., 3], off[..., 3])
        edge = (on[..., 3] > 0.05) & (on[..., 3] < 0.95)
        self.assertTrue(edge.any())
        self.assertTrue(np.all(on[..., 1][edge] < off[..., 1][edge]))                  # less green is left where the screen shows through


class IBKGizmoTilesTests(unittest.TestCase):
    def test_tiles_equal_the_whole_frame(self):
        from tests.test_2d_behaviour_color_a import tiled
        from tests.test_2d_parity_group_3b import Graph
        for params in ({"screen_type": "green"}, {"screen_type": "blue", "luminance_match": 1, "screen_subtraction": 0}):
            for masked in (False, True):
                g = Graph()
                g.add("fg", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33, red=0.7, green=0.5, blue=0.3, alpha=1.0, softness=0.5))
                g.add("cp", "Rectangle", dict(width=61, height=47, box_x=0, box_y=0, box_width=61, box_height=47, red=0.1, green=0.7, blue=0.3, alpha=1.0, softness=0.1))
                g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
                inputs = {"fg": "fg", "c": "cp"}
                if masked:
                    inputs["mask"] = "matte"
                g.add("node", "IBKGizmo", params, **inputs)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
