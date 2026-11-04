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
        identity_by_design = {"log2lin_alpha_untouched", "colorspace_bradford_equal_illuminants", "colorlookup_default_is_identity", "crosstalk_default_is_identity"}
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
                slots = list(spec["inputs"]) + list(spec.get("optional_inputs", []))
                wired = [src] + [None] * (len(slots) - 1)
                wired[slots.index("mask")] = matte()
                got = Evaluator._kernel(kind, full, wired)
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
    "swap": ({"transfer_out": "sRGB"}, {"swap": 1}),
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
        self.assertEqual(lacking, {"colorspace_hsv", "colorspace_ycbcr", "colorspace_cie"})      # the guide gives no formula for them (cases unscored, step P)
        from nodebased.core import CHOICES
        self.assertEqual(CHOICES["transfer_in"][:4], ["Linear", "sRGB", "Gamma 2.2", "Gamma 2.4"])     # the older choices keep their place; step P adds Rec.709 and Cineon


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


# ================================================================================================================ Expression
OLD_KNOBS["Expression"] = ("older_expression_syntax",)
_EX_BASE = {"expr_r": "r * 1.5", "expr_g": "g + 0.1", "expr_b": "b * t0 + t1 * 0.1 + t2 * 0.01 + t3 * 0.001", "expr_a": "a * 0.5",
            "temp_name0": "t0", "temp_expr0": "1", "temp_name1": "t1", "temp_expr1": "2", "temp_name2": "t2", "temp_expr2": "3", "temp_name3": "t3", "temp_expr3": "4"}
_EX_CHANGES = {
    "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"], "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"],
    "mix": CHROME["mix"], "inject": {"inject": 1}, "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "expr0": {"expr_r": "r * 0.5"}, "expr1": {"expr_g": "g - 0.2"}, "expr2": {"expr_b": "b + t0"}, "expr3": {"expr_a": "a * 0.25"},
    "channel0": {"expr_channels_r": "alpha"}, "channel1": {"expr_channels_g": "alpha"}, "channel2": {"expr_channels_b": "alpha"}, "channel3": {"expr_channels_a": "rgba"},
}
for _slot in range(4):
    _other = (_slot + 1) % 4
    _EX_CHANGES[f"temp_name{_slot}"] = {f"temp_name{_slot}": f"t{_other}", f"temp_name{_other}": f"t{_slot}"}
    _EX_CHANGES[f"temp_expr{_slot}"] = {f"temp_expr{_slot}": "9"}


class ExpressionKnobsActTests(KnobsActTests):
    BASES = {"Expression": _EX_BASE}
    CHANGES = {"Expression": _EX_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("Expression")


class ExpressionTilesTests(TilesMatchTests):
    TILE_PARAMS = {"Expression": [dict(_EX_BASE), {"expr_r": "x / width", "expr_g": "y / height", "expr_b": "cx + cy", "expr_a": "r > 0.5 ? frame : 0"},
                                  {"expr_channels_r": "rgb", "expr_r": "hypot(cx, cy)", "temp_name0": "d", "temp_expr0": "x - y", "expr_b": "d"},
                                  {"older_expression_syntax": 1, "expr_r": "x / width", "expr_g": "y / height", "expr_b": "r if g > 0.5 else 0"}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("Expression")


class ExpressionOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"Expression": ("_expression", [{"expr_r": "r*2", "expr_g": "x/width", "expr_b": "b", "expr_a": "a"},
                                             {"expr_r": "r if g > 0.2 else 0", "expr_g": "y/height", "expr_b": "min(r, g) ** 2", "expr_a": "a", "mix": 0.6}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("Expression")

    def test_a_new_node_reads_nukes_syntax(self):
        params = SPECS["Expression"]["params"]
        self.assertEqual(params["older_expression_syntax"], 0)
        src = fixture_image()
        np.testing.assert_array_equal(Evaluator._kernel("Expression", dict(params), [src]), src)       # r, g, b, a rows: nothing changes
        with self.assertRaises(ValueError):
            Evaluator._kernel("Expression", dict(params, expr_r="r if g > 0.2 else 0"), [src])
        with self.assertRaises(ValueError):
            Evaluator._kernel("Expression", dict(params, older_expression_syntax=1, expr_r="r > 0.2 ? 1 : 0"), [src])


class ExpressionLanguageTests(unittest.TestCase):
    """What the cases cannot compare: coordinates, the refused functions, errors and the first reading of the page's other statements."""

    @staticmethod
    def run_rows(rows, shape=(5, 7), **extra):
        src = np.zeros(shape + (4,), np.float32)
        src[..., 0] = np.linspace(-1, 2, shape[0] * shape[1], dtype=np.float32).reshape(shape)
        src[..., 1], src[..., 2], src[..., 3] = 0.5, 0.25, 1.0
        params = dict(SPECS["Expression"]["params"], **{f"expr_{c}": rows.get(c, c) for c in "rgba"}, **extra)
        return src, Evaluator._kernel("Expression", params, [src])

    def test_coordinates_count_up_from_the_bottom_row_through_pixel_centres(self):
        _, out = self.run_rows({"r": "x", "g": "y", "b": "cx", "a": "cy"}, shape=(4, 8))
        np.testing.assert_allclose(out[0, :, 0], np.arange(8) + 0.5)
        np.testing.assert_allclose(out[:, 0, 1], [3.5, 2.5, 1.5, 0.5])                 # the top row is the highest y
        np.testing.assert_allclose(out[0, :, 2], (np.arange(8) + 0.5 - 4) / 4)          # -1 at the left edge, 1 at the right edge
        self.assertAlmostEqual(float(out[0, 0, 2]) - 1 / 8 * 0, -0.875)
        np.testing.assert_allclose(out[:, 0, 3], (np.array([3.5, 2.5, 1.5, 0.5]) - 2) / 4)       # cy is in the same units as cx

    def test_cx_and_cy_are_symmetric_about_the_centre(self):
        _, out = self.run_rows({"b": "cx", "a": "cy"}, shape=(6, 10))
        np.testing.assert_allclose(out[:, :, 2], -out[:, ::-1, 2], atol=1e-6)
        np.testing.assert_allclose(out[:, :, 3], -out[::-1, :, 3], atol=1e-6)

    def test_width_height_and_frame(self):
        _, out = self.run_rows({"r": "width", "g": "height", "b": "frame()"}, shape=(3, 4))
        self.assertEqual((float(out[0, 0, 0]), float(out[0, 0, 1])), (4.0, 3.0))
        _, out = self.run_rows({"r": "pi()", "g": "pi", "b": "x() - x"})
        self.assertAlmostEqual(float(out[0, 0, 0]), np.pi, places=6)
        self.assertEqual(float(out[0, 0, 2]), 0.0)

    def test_acos_and_asin_give_nan_outside_minus_one_to_one(self):
        src, out = self.run_rows({"r": "acos(r)", "g": "asin(r)"})
        outside = np.abs(src[..., 0]) > 1
        self.assertTrue(outside.any())
        self.assertTrue(np.isnan(out[..., 0][outside]).all() and np.isnan(out[..., 1][outside]).all())
        self.assertTrue(np.isfinite(out[..., 0][~outside]).all())

    def test_division_by_zero_is_an_infinity_like_any_32_bit_float_math(self):
        _, out = self.run_rows({"r": "1 / (r - r)", "g": "0 / (r - r)"})
        self.assertTrue(np.isinf(out[..., 0]).all() and np.isnan(out[..., 1]).all())

    def test_the_conditional_is_right_associative_and_logic_gives_one_or_zero(self):
        _, out = self.run_rows({"r": "r > 10 ? 1 : r > 0 ? 2 : 3", "g": "(r > 0) + (r > 1) + (r < 0) * 10", "b": "!(r > 0) + !0 + !5"})
        src = self.run_rows({})[0]
        np.testing.assert_array_equal(out[..., 0], np.where(src[..., 0] > 10, 1.0, np.where(src[..., 0] > 0, 2.0, 3.0)))
        np.testing.assert_array_equal(out[..., 1], (src[..., 0] > 0) * 1.0 + (src[..., 0] > 1) + (src[..., 0] < 0) * 10.0)
        np.testing.assert_array_equal(out[..., 2], (src[..., 0] <= 0) * 1.0 + 1.0 + 0.0)

    def test_function_calls_check_their_argument_count_and_names(self):
        for bad in ("sin()", "pow(r)", "lerp(r, g)", "clamp(r, g)", "nosuch(r)", "random()", "noise(r)", "r +", "r ? 1", "(r", "r $ g", "from_sRGB(r)", "Blur1.size", "r.red"):
            with self.subTest(expression=bad):
                with self.assertRaises(ValueError):
                    self.run_rows({"r": bad})

    def test_temporary_variables_can_use_earlier_ones_and_must_have_usable_names(self):
        _, out = self.run_rows({"r": "bb"}, temp_name0="aa", temp_expr0="r * 2", temp_name1="bb", temp_expr1="aa + 1")
        src = self.run_rows({})[0]
        np.testing.assert_allclose(out[..., 0], src[..., 0] * 2 + 1, atol=1e-6)
        for name in ("r", "sin", "1x", "a b", "width", "b2"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.run_rows({}, temp_name0=name, temp_expr0="1")
        self.run_rows({}, temp_name0="  ", temp_expr0="oops(")          # a row with no name is ignored, not parsed

    def test_the_row_channels_accept_words_and_layer_names(self):
        _, out = self.run_rows({"r": "0.5"}, expr_channels_r="rgb")
        np.testing.assert_array_equal(out[..., 0], 0.5)
        _, out = self.run_rows({"r": "0.5"}, expr_channels_r="rgba.green, blue")
        src = self.run_rows({})[0]
        np.testing.assert_array_equal(out[..., 0], src[..., 0])
        np.testing.assert_array_equal(out[..., 1], 0.5)
        with self.assertRaises(ValueError):
            self.run_rows({}, expr_channels_r="depth.Z")

    def test_a_nuke_expression_keeps_the_second_input_names(self):
        src = np.ones((2, 2, 4), np.float32)
        second = np.full((2, 2, 4), 0.25, np.float32)
        params = dict(SPECS["Expression"]["params"], expr_r="r2 + second_g")
        out = Evaluator._kernel("Expression", params, [src, second])
        np.testing.assert_allclose(out[..., 0], 0.5)


# ================================================================================================================ CrossTalk
_XT_LINE = '{"interpolation":"linear","points":[[0,0],[1,0.3]]}'
_XT_BASE = {"xt_curve_g_r": _XT_LINE, "xt_curve_b_b": '{"interpolation":"linear","points":[[0,0],[1,0.6]]}'}
_XT_CHANGES = {
    "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"], "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"],
    "mix": CHROME["mix"], "inject": {"inject": 1}, "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "crossover": {"xt_curve_r_g": '{"interpolation":"linear","points":[[0,0],[1,0.4]]}'},
}


class CrossTalkKnobsActTests(KnobsActTests):
    BASES = {"CrossTalk": _XT_BASE}
    CHANGES = {"CrossTalk": _XT_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("CrossTalk")

    def test_each_of_the_nine_curves_acts_on_its_own_output_channel(self):
        base = run_node("CrossTalk", _XT_BASE).pixels
        for o, out_index in zip("rgb", range(3)):
            for s_ in "rgb":
                with self.subTest(curve=f"{o}<-{s_}"):
                    changed = run_node("CrossTalk", dict(_XT_BASE, **{f"xt_curve_{o}_{s_}": '{"interpolation":"linear","points":[[0,0.1],[1,0.5]]}'})).pixels
                    differs = [c for c in range(4) if not np.array_equal(base[..., c], changed[..., c])]
                    self.assertEqual(differs, [out_index])

    def test_the_older_unpremult_and_fringe_knobs_act(self):
        base = dict(_XT_BASE, xt_curve_r_g='{"interpolation":"linear","points":[[0,0],[1,0.4]]}')
        src = fixture_image().copy()
        src[..., 3] = np.tile(np.array([0.0, 0.5, 1.0, 0.25], np.float32), src.shape[0] * src.shape[1] // 4 + 1)[:src.shape[0] * src.shape[1]].reshape(src.shape[:2])
        plain = render("CrossTalk", base, source=src).pixels
        self.assertFalse(np.array_equal(render("CrossTalk", dict(base, xt_unpremult="alpha"), source=src).pixels, plain))
        edge = render("CrossTalk", dict(base, xt_fringe=1), source=src).pixels
        self.assertFalse(np.array_equal(edge, plain))
        alpha = src[..., 3]
        solid = (alpha <= 1e-6) | (alpha >= 1 - 1e-6)
        np.testing.assert_array_equal(edge[solid], src[solid])               # only the partly transparent pixels take the effect
        self.assertFalse(np.array_equal(edge[~solid], src[~solid]))


class CrossTalkTilesTests(TilesMatchTests):
    TILE_PARAMS = {"CrossTalk": [dict(_XT_BASE), dict(_XT_BASE, xt_unpremult="alpha"), dict(_XT_BASE, xt_fringe=1, xt_curve_r_b=_XT_LINE)]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("CrossTalk")


class CrossTalkBehaviourTests(unittest.TestCase):
    def test_the_defaults_pass_the_colour_straight_through_and_alpha_is_untouched(self):
        src = fixture_image()
        np.testing.assert_allclose(Evaluator._crosstalk(src, SPECS["CrossTalk"]["params"]), src, atol=1e-6)
        out = Evaluator._crosstalk(src, dict(SPECS["CrossTalk"]["params"], **_XT_BASE))
        np.testing.assert_array_equal(out[..., 3], src[..., 3])

    def test_the_output_is_the_sum_of_three_lookups(self):
        src = np.array([[[0.2, 0.5, 0.8, 1.0]]], np.float32)
        params = dict(SPECS["CrossTalk"]["params"], xt_curve_g_r='{"interpolation":"linear","points":[[0,0],[1,0.3]]}',
                      xt_curve_g_b='{"interpolation":"linear","points":[[0,0.1],[1,0.2]]}')
        out = Evaluator._crosstalk(src, params)
        self.assertAlmostEqual(float(out[0, 0, 1]), 0.5 + 0.3 * 0.2 + (0.1 + 0.1 * 0.8), places=6)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.2, places=6)

    def test_unpremult_by_a_channel_divides_before_the_lookups_and_multiplies_back(self):
        src = np.array([[[0.2, 0.3, 0.1, 0.5]]], np.float32)
        params = dict(SPECS["CrossTalk"]["params"], xt_curve_g_r='{"interpolation":"linear","points":[[0,0],[1,1]]}', xt_unpremult="alpha")
        out = Evaluator._crosstalk(src, params)
        straight = src[0, 0, :3] / 0.5
        expected = np.array([straight[0], straight[1] + straight[0], straight[2]]) * 0.5
        np.testing.assert_allclose(out[0, 0, :3], expected, atol=1e-6)


# ================================================================================================================ HistEQ
OLD_KNOBS["HistEQ"] = ("histeqbins",)


class HistEQBehaviourTests(unittest.TestCase):
    """The page gives the aim and no algorithm, so the cases are not scored; these tests check the properties the aim implies."""

    @staticmethod
    def skewed(n=64):
        values = np.linspace(0.0, 1.0, n * n, dtype=np.float32).reshape(n, n) ** 3
        img = np.zeros((n, n, 4), np.float32)
        img[..., :3] = values[..., None]
        img[..., 1] = values ** 2
        img[..., 3] = 1.0
        return img

    @staticmethod
    def spread(values, bins=16):
        return float(np.histogram(values, bins=bins, range=(values.min(), values.max()))[0].std())

    def test_a_skewed_histogram_comes_out_flatter(self):
        src = self.skewed()
        for mode in ("channels", "luminance"):
            with self.subTest(mode=mode):
                out = Evaluator._hist_eq(src, {"hist_eq_mode": mode, "histeqbins": 4096})
                probe = src[..., 0] if mode == "channels" else src[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
                after = out[..., 0] if mode == "channels" else out[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
                self.assertLess(self.spread(after), self.spread(probe) / 3)

    def test_the_order_of_the_values_is_kept(self):
        src = self.skewed()
        out = Evaluator._hist_eq(src, {"hist_eq_mode": "channels", "histeqbins": 4096})
        order = np.argsort(src[..., 0].ravel(), kind="stable")
        self.assertTrue(np.all(np.diff(out[..., 0].ravel()[order]) >= -1e-7))

    def test_a_flat_image_and_alpha_are_left_alone(self):
        flat = np.full((6, 6, 4), 0.4, np.float32)
        np.testing.assert_array_equal(Evaluator._hist_eq(flat, {"hist_eq_mode": "channels", "histeqbins": 4096}), flat)
        src = self.skewed(16)
        np.testing.assert_array_equal(Evaluator._hist_eq(src, {"hist_eq_mode": "channels", "histeqbins": 64})[..., 3], src[..., 3])

    def test_histogram_entries_changes_the_result(self):
        src = self.skewed()
        coarse = Evaluator._hist_eq(src, {"hist_eq_mode": "channels", "histeqbins": 8})
        fine = Evaluator._hist_eq(src, {"hist_eq_mode": "channels", "histeqbins": 4096})
        self.assertFalse(np.array_equal(coarse, fine))
        for out in (coarse, fine):                       # either way the mapping keeps the order
            order = np.argsort(src[..., 0].ravel(), kind="stable")
            self.assertTrue(np.all(np.diff(out[..., 0].ravel()[order]) >= -1e-7))
        self.assertEqual(SPECS["HistEQ"]["params"]["histeqbins"], 4096)

    def test_an_older_document_keeps_its_256_entries(self):
        for params in ({"hist_eq_mode": "channels"}, {"hist_eq_mode": "luminance", "mix": 0.6}):
            with self.subTest(params=params):
                doc = upgrade_document(step_a.old_document("HistEQ", params))
                self.assertEqual(doc["nodes"]["node"]["params"]["histeqbins"], 256)
                plate, gate = evaluator_pixels(doc, "under"), evaluator_pixels(doc, "matte")
                full = dict(SPECS["HistEQ"]["params"], **doc["nodes"]["node"]["params"])
                np.testing.assert_array_equal(evaluator_pixels(doc, "node"),
                                              Evaluator._apply_mask_mix(plate, Legacy._hist_eq(plate, full), mask=gate, mix=full.get("mix", 1.0)))

    def test_every_paired_knob_acts(self):
        base = {"hist_eq_mode": "channels"}
        for nuke, params in KnobsActTests.paired("HistEQ"):
            with self.subTest(knob=nuke):
                self.assertEqual(nuke, "histeqbins")
                a = render("HistEQ", dict(base, histeqbins=4096), source=self.skewed(64)).pixels
                b = render("HistEQ", dict(base, histeqbins=4), source=self.skewed(64)).pixels
                self.assertFalse(np.array_equal(a, b))


# ================================================================================================================ MinColor
class MinColorBehaviourTests(unittest.TestCase):
    """The page: the darkest value is found over the search channels, and target minus it is added to the modify channels, as the direction allows."""

    def setUp(self):
        self.src = np.zeros((4, 5, 4), np.float32)
        self.src[..., 0] = np.linspace(0.3, 0.9, 20, dtype=np.float32).reshape(4, 5)
        self.src[..., 1] = 0.5
        self.src[..., 2] = np.linspace(0.2, 0.8, 20, dtype=np.float32).reshape(4, 5)[::-1]
        self.src[..., 3] = 0.75

    def delta(self, **p):
        from nodebased import mincolorknobs
        base = dict(SPECS["MinColor"]["params"], **p)
        return mincolorknobs.find_pixel_delta([mincolorknobs.search_minimum(self.src, (0, 0), (0, 0, 0, 0), base["channels"])], base["target"])

    def test_find_pixel_delta_is_the_target_minus_the_darkest_searched_value(self):
        self.assertAlmostEqual(self.delta(), 0.095 - 0.2, places=6)                       # rgb: the darkest is blue's 0.2
        self.assertAlmostEqual(self.delta(channels="red"), 0.095 - 0.3, places=6)
        self.assertAlmostEqual(self.delta(channels="rgba", target=0.0), -0.2, places=6)
        self.assertAlmostEqual(self.delta(channels="alpha", target=0.5), 0.5 - 0.75, places=6)
        self.assertAlmostEqual(self.delta(target=0.3), 0.3 - 0.2, places=6)

    def test_the_search_covers_all_frames_and_the_box(self):
        from nodebased import mincolorknobs
        other = self.src.copy()
        other[3, 4, :3] = 0.05
        whole = [mincolorknobs.search_minimum(self.src, (0, 0), (0, 0, 0, 0), "rgb"), mincolorknobs.search_minimum(other, (0, 0), (0, 0, 0, 0), "rgb")]
        self.assertAlmostEqual(mincolorknobs.find_pixel_delta(whole, 0.095), 0.095 - 0.05, places=6)
        boxed = mincolorknobs.search_minimum(other, (0, 0), (0, 0, 2, 2), "rgb")
        self.assertGreater(boxed, 0.05)
        self.assertIsNone(mincolorknobs.find_pixel_delta([None], 0.1))

    def test_adding_the_delta_puts_the_darkest_value_on_the_target(self):
        delta = self.delta()
        node = dict(SPECS["MinColor"]["params"], pixeldelta=delta, adjdir="Both")
        out = render("MinColor", node, source=self.src).pixels
        self.assertAlmostEqual(float(out[..., :3].min()), 0.095, places=6)
        np.testing.assert_allclose(out, self.src + np.float32(delta), atol=1e-7)               # modify defaults to all, alpha included

    def test_direction_decides_whether_the_delta_is_added(self):
        for direction, delta, added in (("Up", 0.2, True), ("Up", -0.2, False), ("Down", -0.2, True), ("Down", 0.2, False), ("Both", 0.2, True), ("Both", -0.2, True),
                                        ("Both", 0.0, False), ("Up", 0.0, False)):
            with self.subTest(direction=direction, delta=delta):
                out = render("MinColor", dict(pixeldelta=delta, adjdir=direction), source=self.src).pixels
                self.assertEqual(not np.array_equal(out, self.src), added)

    def test_modify_picks_the_channels(self):
        for modify, picked in (("all", [0, 1, 2, 3]), ("rgb", [0, 1, 2]), ("alpha", [3]), ("red", [0]), ("green blue", [1, 2])):
            with self.subTest(modify=modify):
                out = render("MinColor", dict(pixeldelta=0.25, modify=modify), source=self.src).pixels
                changed = [c for c in range(4) if not np.array_equal(out[..., c], self.src[..., c])]
                self.assertEqual(changed, picked)

    def test_a_new_node_and_an_older_document_add_nothing_until_the_delta_is_found(self):
        self.assertEqual(SPECS["MinColor"]["params"]["pixeldelta"], 0.0)
        self.assertEqual(SPECS["MinColor"]["params"]["target"], 0.095)
        np.testing.assert_array_equal(render("MinColor", {}, source=self.src).pixels, self.src)
        from tests.test_2d_behaviour_color_a import plate_graph
        g = plate_graph("MinColor", {"mincolor_mode": "maximum", "frame_start": 2}, masked=False)
        for name in ("channels", "modify", "target", "adjdir", "pixeldelta"):
            g.doc["nodes"]["node"]["params"].pop(name, None)
        doc = upgrade_document(g.doc)
        self.assertEqual(doc["nodes"]["node"]["params"].get("pixeldelta", 0.0), 0.0)
        np.testing.assert_array_equal(evaluator_pixels(doc, "node"), evaluator_pixels(doc, "under"))

    def test_every_paired_knob_acts(self):
        from nodebased import mincolorknobs
        base = dict(SPECS["MinColor"]["params"], pixeldelta=0.2)
        before = render("MinColor", base).pixels
        changes = {"modify": dict(base, modify="rgb"), "adjdir": dict(base, adjdir="Down"), "pixeldelta": dict(base, pixeldelta=0.4)}
        found = {}
        for knob in ("channels", "target"):
            # these two act through Find Pixel Delta: the delta it stores, then the pixels with it
            variant = dict(SPECS["MinColor"]["params"], **({"channels": "red"} if knob == "channels" else {"target": 0.4}))
            delta = mincolorknobs.find_pixel_delta([mincolorknobs.search_minimum(fixture_image(), (0, 0), (0, 0, 0, 0), variant["channels"])], variant["target"])
            default_delta = mincolorknobs.find_pixel_delta([mincolorknobs.search_minimum(fixture_image(), (0, 0), (0, 0, 0, 0), "rgb")], 0.095)
            self.assertNotEqual(delta, default_delta, knob)
            changes[knob] = dict(base, pixeldelta=delta)
        for nuke, params in KnobsActTests.paired("MinColor"):
            with self.subTest(knob=nuke):
                self.assertIn(nuke, changes, f"MinColor: the paired knob {nuke} has no acting check")
                self.assertFalse(np.array_equal(render("MinColor", changes[nuke]).pixels, before))


if __name__ == "__main__":
    unittest.main()
