"""2D behaviour lane 2, step B: the log, colourspace and lookup nodes compute what Nuke computes.

Log2Lin, PLogLin, Colorspace, ColorLookup, CrossTalk, Expression, HistEQ and MinColor. The same four checks as step A for each node: the
behaviour cases of tools/nuke_fidelity_cases_color.py pass where they are scored; every knob the mapping pairs with a NodeBased param changes
the pixels; the tile path (tiles of 7, 16 and 32, with and without a mask) equals the whole frame for every kernel that changed; and an older
document renders bit for bit as it did before (tests/legacy_behaviour_b.py holds the older kernels).
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
from tests.legacy_behaviour_b import Legacy  # noqa: E402
from tests.test_2d_behaviour_color_a import CHROME, KnobsActTests, TilesMatchTests, fixture_image, run_node  # noqa: E402
from tests.test_2d_fidelity_b_chrome import H, W, image, matte, render  # noqa: E402
from tests.test_2d_parity_group_3b import evaluator_pixels  # noqa: E402

NUKE = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))
MAPPING = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))

STEP_NODES = ("Log2Lin", "PLogLin", "Colorspace", "ColorLookup", "CrossTalk", "Expression", "HistEQ", "MinColor")
OLD_KNOBS = step_a.OLD_KNOBS          # step A's older-document helper reads this table: step B's kinds are added to it


def log_image():
    """Fixture L (Cineon code values over 1023) tiled to the chrome tests' size."""
    return np.tile(fidelity.make_fixtures()["L"], (H // 12, W // 16, 1)).astype(np.float32)


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
        identity_by_design = {"log2lin_alpha_untouched", "colorspace_bradford_equal_illuminants", "colorlookup_default_is_identity"}
        for case in fidelity.CASES:
            if case.nuke not in self.NODE_KEYS or case.unscored or case.inexpressible:
                continue
            ref, valid = case.expected(fx)
            source = fx[case.nb_inputs[0]].astype(np.float64)
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
                if case.source == "convention":
                    self.assertTrue(any(word in case.statement for word in ("Kodak", "McGaugh", "Foundry", "Reference Guide", "IEC", "ITU", "SMPTE", "SDK", "Lindbloom", "Wikipedia", "Natron", "Geremia")), case.id)


class OlderDocumentTests(unittest.TestCase):
    """Loading a document that predates the knob switches the node to its older behaviour; the pixels are those of the older kernel."""

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
                expected = Evaluator._apply_mask_mix(src, legacy(src, full), mask=matte(), mix=full.get("mix", 1.0))
                np.testing.assert_array_equal(got, expected)
                plate, gate = evaluator_pixels(doc, "under"), evaluator_pixels(doc, "matte")
                np.testing.assert_array_equal(evaluator_pixels(doc, "node"),
                                              Evaluator._apply_mask_mix(plate, legacy(plate, full), mask=gate, mix=full.get("mix", 1.0)))


# ================================================================================================================ Log2Lin
OLD_KNOBS["Log2Lin"] = ("older_log_maths",)
_L2L_BASE = {"log_direction": "log to lin"}
_L2L_CHANGES = {
    "channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
    "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
    "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "operation": {"log_direction": "lin to log"}, "black": {"black": 150.0}, "ignore_black": {"ignore_black": 1},
    "white": {"white": 800.0}, "gamma": {"gamma": 0.9},
}


class Log2LinKnobsActTests(KnobsActTests):
    BASES = {"Log2Lin": _L2L_BASE}
    CHANGES = {"Log2Lin": _L2L_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("Log2Lin")


class Log2LinTilesTests(TilesMatchTests):
    TILE_PARAMS = {"Log2Lin": [{}, {"log_direction": "lin to log"}, {"ignore_black": 1, "black": 120.0, "gamma": 0.8}, {"older_log_maths": 1, "gamma": 1.0}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("Log2Lin")


class Log2LinOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"Log2Lin": ("_log2lin", [{"gamma": 1.0}, {"gamma": 0.7, "log_direction": "lin to log"}, {"gamma": 1.0, "black": 64.0, "white": 940.0, "mix": 0.6}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("Log2Lin")

    def test_a_new_node_does_nukes_cineon_conversion(self):
        params = SPECS["Log2Lin"]["params"]
        self.assertEqual((params["black"], params["white"], params["gamma"], params["older_log_maths"], params["ignore_black"]), (95.0, 685.0, 0.6, 0, 0))
        codes = np.zeros((1, 3, 4), np.float32)
        codes[0, :, 0] = np.array([95, 685, 470], np.float32) / 1023.0
        codes[..., 3] = 1.0
        out = Evaluator._log2lin(codes, params)
        np.testing.assert_allclose(out[0, :2, 0], [0.0, 1.0], atol=2e-7)
        self.assertAlmostEqual(float(out[0, 2, 0]), 0.1832, places=3)      # code 470 is middle grey on Kodak's scale
        np.testing.assert_array_equal(out[..., 3], codes[..., 3])

    def test_lin_to_log_undoes_log_to_lin(self):
        codes = log_image()
        codes = codes[..., :]
        forward = Evaluator._log2lin(codes, SPECS["Log2Lin"]["params"])
        back = Evaluator._log2lin(forward, dict(SPECS["Log2Lin"]["params"], log_direction="lin to log"))
        keep = codes[..., :3] * 1023.0 > 96.0           # below black the offset hides the code value: the lin value is under the 1e-10 floor
        np.testing.assert_allclose(back[..., :3][keep], codes[..., :3][keep], atol=2e-5)

    def test_ignore_black_is_a_pure_log_function(self):
        codes = np.array([[[300 / 1023.0, 500 / 1023.0, 685 / 1023.0, 1.0]]], np.float32)
        out = Evaluator._log2lin(codes, dict(SPECS["Log2Lin"]["params"], ignore_black=1))
        np.testing.assert_allclose(out[0, 0, :3], 10 ** ((np.array([300, 500, 685]) - 685) * 0.002 / 0.6), rtol=1e-6)
        # the pure form never reaches 0, where the default conversion does at the black code
        black = np.array([[[95 / 1023.0] * 3 + [1.0]]], np.float32)
        self.assertAlmostEqual(float(Evaluator._log2lin(black, SPECS["Log2Lin"]["params"])[0, 0, 0]), 0.0, places=6)
        self.assertGreater(float(Evaluator._log2lin(black, dict(SPECS["Log2Lin"]["params"], ignore_black=1))[0, 0, 0]), 0.005)


# ================================================================================================================ PLogLin
OLD_KNOBS["PLogLin"] = ("older_log_maths",)
_PLL_BASE = {}
_PLL_CHANGES = {
    "channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
    "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
    "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "operation": {"log_direction": "lin to log"}, "linref": {"linear_reference": 0.3}, "logref": {"log_reference": 500.0},
    "ngamma": {"negative_gamma": 0.9}, "density": {"density_per_code_value": 0.003},
}


class PLogLinKnobsActTests(KnobsActTests):
    BASES = {"PLogLin": _PLL_BASE}
    CHANGES = {"PLogLin": _PLL_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("PLogLin")


class PLogLinTilesTests(TilesMatchTests):
    TILE_PARAMS = {"PLogLin": [{}, {"log_direction": "lin to log"}, {"linear_reference": 0.4, "log_reference": 520.0, "negative_gamma": 0.8},
                               {"older_log_maths": 1, "negative_gamma": 1.0}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("PLogLin")


class PLogLinOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"PLogLin": ("_ploglin", [{"negative_gamma": 1.0}, {"negative_gamma": 0.7, "linear_reference": 0.3, "log_reference": 300.0}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("PLogLin")

    def test_a_new_node_follows_josh_pines(self):
        params = SPECS["PLogLin"]["params"]
        self.assertEqual((params["negative_gamma"], params["older_log_maths"], params["log_direction"]), (0.6, 0, "log to lin"))
        codes = np.array([[[445 / 1023.0, 545 / 1023.0, 345 / 1023.0, 1.0]]], np.float32)
        out = Evaluator._ploglin(codes, params)
        np.testing.assert_allclose(out[0, 0, :3], 0.18 * 10 ** (np.array([0.0, 100.0, -100.0]) * 0.002 / 0.6), rtol=1e-6)

    def test_the_reverse_floors_the_linear_value_at_1e_10(self):
        lin = np.array([[[0.0, -3.0, 0.18, 1.0]]], np.float32)
        out = Evaluator._ploglin(lin, dict(SPECS["PLogLin"]["params"], log_direction="lin to log"))
        floor = (445.0 + np.log10(1e-10 / 0.18) * 0.6 / 0.002) / 1023.0
        np.testing.assert_allclose(out[0, 0, :2], [floor, floor], rtol=1e-6)
        self.assertAlmostEqual(float(out[0, 0, 2]), 445.0 / 1023.0, places=6)

    def test_alpha_is_converted_only_when_channels_include_it(self):
        src = log_image()
        everything = render("PLogLin", {"channels": "all"}, source=src).pixels
        colour = render("PLogLin", {"channels": "rgb"}, source=src).pixels
        np.testing.assert_array_equal(colour[..., 3], src[..., 3])
        self.assertFalse(np.array_equal(everything[..., 3], src[..., 3]))
        np.testing.assert_array_equal(everything[..., :3], colour[..., :3])


# ================================================================================================================ Colorspace
OLD_KNOBS["Colorspace"] = ("divide_by_alpha", "bradford_matrix")
_CS_BASE = {"transfer_in": "sRGB", "primaries_out": "Rec.2020", "whitepoint_out": "D60"}
_CS_CHANGES = {
    "channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
    "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
    "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "colorspace_in": {"transfer_in": "Gamma 2.2"}, "colorspace_out": {"transfer_out": "Gamma 2.4"},
    "illuninant_in": {"whitepoint_in": "DCI"}, "illuninant_out": {"whitepoint_out": "DCI"},
    "primary_in": {"primaries_in": "P3-D65"}, "primary_out": {"primaries_out": "P3-D65"},
    "bradford_matrix": {"bradford_matrix": 1},
}


class ColorspaceKnobsActTests(KnobsActTests):
    BASES = {"Colorspace": _CS_BASE}
    CHANGES = {"Colorspace": _CS_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("Colorspace")


class ColorspaceTilesTests(TilesMatchTests):
    TILE_PARAMS = {"Colorspace": [dict(_CS_BASE), {"transfer_out": "sRGB", "primaries_out": "P3-D65"},
                                  {"transfer_in": "Gamma 2.4", "whitepoint_out": "DCI", "bradford_matrix": 1, "divide_by_alpha": 1}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("Colorspace")


class ColorspaceOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"Colorspace": ("_colorspace", [dict(_CS_BASE), {"transfer_out": "sRGB", "primaries_out": "P3-D65", "mix": 0.5},
                                              {"transfer_in": "Gamma 2.4", "whitepoint_out": "DCI"}, {}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("Colorspace")

    def test_a_new_node_converts_the_stored_colour_and_keeps_xyz(self):
        params = SPECS["Colorspace"]["params"]
        self.assertEqual((params["divide_by_alpha"], params["bradford_matrix"]), (0, 0))
        src = fixture_image()
        full = dict(params, transfer_in="sRGB")
        converted = Evaluator._kernel("Colorspace", full, [src])
        # the stored colour is converted wherever the alpha is: a pixel's result does not depend on its alpha
        other = src.copy()
        other[..., 3] = 0.25
        np.testing.assert_array_equal(Evaluator._kernel("Colorspace", full, [other])[..., :3], converted[..., :3])

    def test_bradford_only_matters_between_different_illuminants(self):
        src = fixture_image()
        same = dict(SPECS["Colorspace"]["params"], transfer_in="sRGB", bradford_matrix=1)
        np.testing.assert_array_equal(Evaluator._kernel("Colorspace", same, [src]),
                                      Evaluator._kernel("Colorspace", dict(same, bradford_matrix=0), [src]))
        moved = dict(same, whitepoint_out="D60")
        self.assertFalse(np.array_equal(Evaluator._kernel("Colorspace", moved, [src]),
                                        Evaluator._kernel("Colorspace", dict(moved, bradford_matrix=0), [src])))

    def test_the_choices_nodebased_lacks_are_listed_in_the_cases(self):
        lacking = {c.id for c in fidelity.CASES if c.nuke == "Colorspace" and c.inexpressible}
        self.assertEqual(lacking, {"colorspace_hsv", "colorspace_ycbcr", "colorspace_cie", "colorspace_cineon", "colorspace_rec709_curve",
                                   "colorspace_other_illuminants"})
        from nodebased.core import CHOICES
        self.assertEqual(CHOICES["transfer_in"], ["Linear", "sRGB", "Gamma 2.2", "Gamma 2.4"])


# ================================================================================================================ ColorLookup
OLD_KNOBS["ColorLookup"] = ("older_range_scaling",)
_CL_CURVE_R = '{"interpolation":"linear","points":[[0,0],[0.5,0.8],[1,1]]}'
_CL_BASE = {"curve_red": _CL_CURVE_R, "curve_master": '{"interpolation":"linear","points":[[0,0],[0.6,0.5],[1,1]]}'}
_CL_CHANGES = {
    "channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
    "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
    "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "lut": {"curve_green": '{"interpolation":"linear","points":[[0,0.2],[1,0.6]]}'},
}
_CL_SET_KNOBS = {"source": ("red", "green", "blue"), "target": ("red", "green", "blue"), "setRGB": ("red", "green", "blue"),
                 "setRGBA": ("red", "green", "blue", "alpha"), "setA": ("alpha",)}


class ColorLookupKnobsActTests(KnobsActTests):
    BASES = {"ColorLookup": _CL_BASE}
    CHANGES = {"ColorLookup": dict(_CL_CHANGES, **{name: None for name in _CL_SET_KNOBS})}

    def _differs(self, kind, nuke_knob):
        if nuke_knob not in _CL_SET_KNOBS:
            return super()._differs(kind, nuke_knob)
        from nodebased import colorlookupknobs
        base = dict(SPECS["ColorLookup"]["params"], **_CL_BASE, source=0.3, target=0.6)
        before = run_node("ColorLookup", base).pixels
        if nuke_knob in ("setRGB", "setRGBA", "setA"):
            after = run_node("ColorLookup", dict(base, **colorlookupknobs.set_points(base, _CL_SET_KNOBS[nuke_knob]))).pixels
        else:
            # source and target act through the buttons: a different pair, pressed, gives different curves
            changed = dict(base, **{nuke_knob: 0.45 if nuke_knob == "source" else 0.25})
            first = colorlookupknobs.set_points(base, ("red", "green", "blue"))
            second = colorlookupknobs.set_points(changed, ("red", "green", "blue"))
            self.assertNotEqual(first, second, f"ColorLookup: changing {nuke_knob} left the Set RGB curves alone")
            before = run_node("ColorLookup", dict(base, **first)).pixels
            after = run_node("ColorLookup", dict(base, **second)).pixels
        self.assertFalse(np.array_equal(before, after), f"ColorLookup: {nuke_knob} left the pixels alone")

    def test_every_paired_knob_acts(self):
        self._check_kind("ColorLookup")

    def test_the_speed_knobs_are_not_paired(self):
        paired = {nuke for nuke, _ in self.paired("ColorLookup")}
        self.assertNotIn("range", paired)
        self.assertNotIn("use_precomputed", paired)
        base = run_node("ColorLookup", _CL_BASE).pixels
        np.testing.assert_array_equal(run_node("ColorLookup", dict(_CL_BASE, range=3.0)).pixels, base)
        np.testing.assert_array_equal(run_node("ColorLookup", dict(_CL_BASE, use_precomputed=0)).pixels, base)


class ColorLookupTilesTests(TilesMatchTests):
    TILE_PARAMS = {"ColorLookup": [dict(_CL_BASE), {"curve_blue": '{"interpolation":"smooth","points":[[0,0],[0.4,0.7],[1,1]]}', "range": 2.0},
                                   {"older_range_scaling": 1, "range": 2.0, "curve_green": _CL_CURVE_R}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("ColorLookup")


class ColorLookupOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"ColorLookup": ("_color_lookup", [dict(_CL_BASE), dict(_CL_BASE, range=2.0), dict(_CL_BASE, range=0.5, mix=0.6)])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("ColorLookup")

    def test_a_new_node_ignores_range(self):
        self.assertEqual(SPECS["ColorLookup"]["params"]["older_range_scaling"], 0)
        src = fixture_image()
        node = dict(SPECS["ColorLookup"]["params"], curve_red=_CL_CURVE_R)
        np.testing.assert_array_equal(Evaluator._kernel("ColorLookup", dict(node, range=2.0), [src]), Evaluator._kernel("ColorLookup", node, [src]))
        self.assertFalse(np.array_equal(Evaluator._kernel("ColorLookup", dict(node, range=2.0, older_range_scaling=1), [src]), Evaluator._kernel("ColorLookup", node, [src])))


class ColorLookupSetButtonsTests(unittest.TestCase):
    """Set RGB, Set RGBA and Set A add a point mapping source to target on the chosen curves, keeping the points they already hold."""

    def test_a_point_is_added_to_each_chosen_curve(self):
        from nodebased import colorcurves, colorlookupknobs
        params = dict(SPECS["ColorLookup"]["params"], source=0.25, target=0.5)
        curves = colorlookupknobs.set_points(params, ("red", "green", "blue"))
        self.assertEqual(sorted(curves), ["curve_blue", "curve_green", "curve_red"])
        for text in curves.values():
            curve = colorcurves.decode(text)
            self.assertAlmostEqual(float(colorcurves.evaluate(curve, 0.25)), 0.5, places=6)
            self.assertAlmostEqual(float(colorcurves.evaluate(curve, 0.0)), 0.0, places=6)
            self.assertAlmostEqual(float(colorcurves.evaluate(curve, 1.0)), 1.0, places=6)

    def test_earlier_points_stay_and_a_key_at_the_source_takes_the_target(self):
        from nodebased import colorcurves, colorlookupknobs
        params = dict(SPECS["ColorLookup"]["params"], source=0.25, target=0.5)
        params.update(colorlookupknobs.set_points(params, ("red",)))
        params.update(source=0.75, target=0.6)
        params.update(colorlookupknobs.set_points(params, ("red",)))
        curve = colorcurves.decode(params["curve_red"])
        self.assertEqual([round(x, 6) for x, _ in curve["points"]], [0.0, 0.25, 0.75, 1.0])
        params.update(source=0.25, target=0.4)
        params.update(colorlookupknobs.set_points(params, ("red",)))
        self.assertAlmostEqual(float(colorcurves.evaluate(colorcurves.decode(params["curve_red"]), 0.25)), 0.4, places=6)

    def test_set_a_touches_alpha_only_and_a_split_pair_maps_per_channel(self):
        from nodebased import colorcurves, colorlookupknobs
        params = dict(SPECS["ColorLookup"]["params"], source=[0.1, 0.2, 0.3, 0.4], target=[0.5, 0.6, 0.7, 0.8])
        self.assertEqual(sorted(colorlookupknobs.set_points(params, ("alpha",))), ["curve_alpha"])
        curves = colorlookupknobs.set_points(params, ("red", "green", "blue", "alpha"))
        for name, source, target in (("red", 0.1, 0.5), ("green", 0.2, 0.6), ("blue", 0.3, 0.7), ("alpha", 0.4, 0.8)):
            self.assertAlmostEqual(float(colorcurves.evaluate(colorcurves.decode(curves[f"curve_{name}"]), source)), target, places=6)

    def test_a_source_beyond_the_curve_extends_it(self):
        from nodebased import colorcurves, colorlookupknobs
        params = dict(SPECS["ColorLookup"]["params"], source=1.5, target=1.2)
        curve = colorcurves.decode(colorlookupknobs.set_points(params, ("red",))["curve_red"])
        self.assertEqual([round(x, 6) for x, _ in curve["points"]], [0.0, 1.0, 1.5])
        self.assertAlmostEqual(float(colorcurves.evaluate(curve, 1.5)), 1.2, places=6)


if __name__ == "__main__":
    unittest.main()
