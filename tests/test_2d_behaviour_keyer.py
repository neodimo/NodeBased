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

STEP_NODES = ("Keyer", "HueKeyer", "ChromaKeyer", "Difference", "ScreenKeyer", "IBKColor", "IBKGizmo", "Copy", "ChannelMerge", "Shuffle")
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


if __name__ == "__main__":
    unittest.main()
