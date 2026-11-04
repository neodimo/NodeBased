"""2D behaviour lane 2, step A: ColorCorrect, Saturation, HueCorrect, HSVTool, Toe, SoftClip and Posterize compute what Nuke computes.

Four things per node: the behaviour cases of tools/nuke_fidelity_cases_color.py pass where they are scored; every knob the mapping pairs
with a NodeBased param changes the pixels; the tile path (tiles of 7, 16 and 32, with and without a mask) equals the whole frame for every
kernel that changed; and an older document renders bit for bit as it did before (tests/legacy_behaviour_a.py holds the older kernels).
"""
import copy
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
from nodebased.raster import Raster  # noqa: E402
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor  # noqa: E402
from tests.legacy_behaviour_a import Legacy  # noqa: E402
from tests.test_2d_fidelity_b_chrome import H, W, image, matte, render  # noqa: E402
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels  # noqa: E402

NUKE = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))
MAPPING = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))

STEP_NODES = ("ColorCorrect", "Saturation", "HueCorrect", "HSVTool", "Toe", "SoftClip", "Posterize")
CHROME = {  # the shared chrome knobs: a baseline change that has an effect on a node with an active correction, per Nuke knob name
    "channels": {"channels": "red"},
    "maskChannelInput": {"maskChannelInput": "rgba.red"},
    "invert_mask": {"invert_mask": 1},
    "invert": {"invert_mask": 1},
    "fringe": {"fringe": 1},
    "unpremult": {"unpremult": "rgba.alpha"},
    "mix": {"mix": 0.5},
}
STRICT = ("channels", "maskChannelInput", "invert_mask", "invert", "fringe", "unpremult", "mix", "invert_unpremult", "inject")


def fixture_image(seed=3):
    """An HDR image with negatives and a varying alpha."""
    return image(seed)


def colour_image():
    """The HSV nodes' fixture C (colour inside 0 to 1 spread over hue, saturation and value), tiled to the chrome tests' size."""
    return np.tile(fidelity.make_fixtures()["C"], (H // 12, W // 16, 1)).astype(np.float32)


def run_node(kind, params, source=None):
    """The node's whole-image kernel on Rasters with a matte wired, the way the evaluator calls it (the shared chrome included)."""
    return render(kind, dict(params), source=source, mask=matte())


# ------------------------------------------------------------------------------------------------------- shared checks
class CaseTests(unittest.TestCase):
    """The behaviour cases for these nodes pass wherever they are scored."""

    NODE_KEYS = {"ColorCorrect", "Saturation", "HueCorrect", "HSVTool", "Toe", "SoftClip", "Posterize"}

    def test_every_scored_case_passes(self):
        mine = [c for c in fidelity.CASES if c.nuke in self.NODE_KEYS]
        self.assertTrue(mine, "no behaviour cases for the step's nodes")
        for case in mine:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                if case.unscored:
                    self.assertEqual(result["status"], "not scored")
                else:
                    self.assertEqual(result["status"], "pass", f"{case.id}: {result}")

    def test_every_scored_case_compares_pixels_the_node_changes(self):
        """A case whose valid mask is empty, or whose reference equals its input everywhere, would pass whatever the kernel does."""
        fx = fidelity.make_fixtures()
        identity_by_design = {"huecorrect_defaults", "hsv_defaults_leave_colour", "softclip_none", "toe_default_lift_is_none", "saturation_default_is_identity"}
        for case in fidelity.CASES:
            if case.nuke not in self.NODE_KEYS or case.unscored:
                continue
            ref, valid = case.expected(fx)
            source = fx[case.nb_inputs[0]].astype(np.float64)
            with self.subTest(case=case.id):
                self.assertGreater(int(valid.sum()), 50, "too few pixels compared")
                if case.id not in identity_by_design:
                    self.assertGreater(int((valid & (np.abs(ref - source) > 1e-9)).sum()), 20, "the reference changes too few of the compared values")

    def test_unscored_cases_say_why(self):
        for case in fidelity.CASES:
            if case.nuke in self.NODE_KEYS and case.unscored:
                self.assertGreater(len(case.unscored), 30, case.id)

    def test_every_case_names_a_source(self):
        for case in fidelity.CASES:
            if case.nuke in self.NODE_KEYS:
                self.assertIn(case.source, ("guide", "convention"), case.id)
                if case.source == "convention":
                    self.assertTrue(any(word in case.statement for word in ("andreageremia", "Foundry", "Natron", "SDK", "Geremia", "Reference Guide")), case.id)


class KnobsActTests(unittest.TestCase):
    """A knob the mapping pairs with a NodeBased param changes the output pixels."""

    BASES = {}          # kind -> baseline params (an active correction, so the chrome knobs have something to gate)
    CHANGES = {}        # kind -> {nuke knob: change}; a change is a dict of param updates, or (baseline updates, change)
    NO_EFFECT = {}      # kind -> knobs whose change cannot alter the pixels (a degree-one homogeneous correction commutes with (un)premult)

    @staticmethod
    def paired(kind):
        pair = MAPPING["pairs"][kind]
        names = []
        for entry in pair["knobs"]:
            if entry["kind"] in ("same", "renamed", "components"):
                names += [(nuke, entry["nodebased"]) for nuke in entry["nuke"]]
        return names

    def _differs(self, kind, nuke_knob):
        change = self.CHANGES[kind][nuke_knob]
        base_updates, change = change if isinstance(change, tuple) else ({}, change)
        base = dict(self.BASES[kind], **base_updates)
        a = run_node(kind, base)
        b = run_node(kind, dict(base, **change))
        if "inject" in change:
            self.assertIn("mask", b.layers or {}, f"{kind}.{nuke_knob}: inject wrote no mask layer")
            return
        if nuke_knob in self.NO_EFFECT.get(kind, ()):
            np.testing.assert_allclose(b.pixels, a.pixels, rtol=1e-5, atol=1e-6, err_msg=f"{kind}.{nuke_knob} is meant to leave the pixels alone")
            return
        self.assertFalse(np.array_equal(a.pixels, b.pixels), f"{kind}: changing {nuke_knob} left the pixels alone")

    def _check_kind(self, kind):
        for nuke_knob, params in self.paired(kind):
            if nuke_knob in ("N/A",):
                continue
            with self.subTest(kind=kind, knob=nuke_knob):
                self.assertIn(nuke_knob, self.CHANGES[kind], f"{kind}: the paired knob {nuke_knob} ({params}) has no acting check")
                self._differs(kind, nuke_knob)


# ------------------------------------------------------------------------------------------------------------ tiles
def tiled(document, edge):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
    document = dict(document, view="node")
    assert executor.supports_tiled(document, "node")
    region = executor.canvas_region(document, "node", frame=1, tier=1)
    return executor.compose_region(document, "node", region, frame=1, tier=1).pixels


def plate_graph(kind, params, masked):
    g = Graph()
    g.add("plate", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33,
                                    red=0.9, green=0.5, blue=0.2, alpha=0.8, softness=0.5))
    g.add("ramp", "Ramp", dict(width=61, height=47))
    g.add("under", "Merge", dict(operation="over"), A="plate", B="ramp")
    g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
    inputs = {"image": "under"}
    if masked:
        inputs["mask"] = "matte"
    g.add("node", kind, params, **inputs)
    return g


class TilesMatchTests(unittest.TestCase):
    TILE_PARAMS = {}

    def _check(self, kind):
        self.assertIn(kind, SUPPORTED_TILED_KINDS)
        for params in self.TILE_PARAMS[kind]:
            for masked in (False, True):
                g = plate_graph(kind, params, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


# ------------------------------------------------------------------------------------------------------ older documents
def old_document(kind, params):
    """A document saved before the behaviour knobs existed: the node carries only the params it had."""
    g = plate_graph(kind, params, masked=True)
    doc = copy.deepcopy(g.doc)
    for name in OLD_KNOBS.get(kind, ()):
        doc["nodes"]["node"]["params"].pop(name, None)
    return doc


OLD_KNOBS = {}


class OlderDocumentTests(unittest.TestCase):
    """Loading a document that predates the knob switches the node to its older behaviour; the pixels are those of the older kernel."""

    LEGACY = {}      # kind -> (Legacy kernel name, [param sets])

    def _check(self, kind):
        name, param_sets = self.LEGACY[kind]
        legacy = getattr(Legacy, name)
        spec = SPECS[kind]
        for params in param_sets:
            with self.subTest(kind=kind, params=params):
                doc = upgrade_document(old_document(kind, params))
                for knob in OLD_KNOBS[kind]:
                    self.assertIn(knob, doc["nodes"]["node"]["params"], f"{kind}: an older document gets {knob}")
                # whole-image kernel, mask and mix gating as the evaluator applies them
                src = fixture_image()
                full = dict(spec["params"], **doc["nodes"]["node"]["params"])
                got = Evaluator._kernel(kind, full, [src, matte()])
                np.testing.assert_array_equal(got, self.older(kind, legacy, src, full, matte()))
                # and the whole graph, through the evaluator
                plate, gate = evaluator_pixels(doc, "under"), evaluator_pixels(doc, "matte")
                np.testing.assert_array_equal(evaluator_pixels(doc, "node"), self.older(kind, legacy, plate, full, gate))

    @staticmethod
    def older(kind, legacy, src, full, mask):
        """The older kernel with the mask and mix gating the evaluator applies (HSVTool takes its wired mask as a range weight instead)."""
        if kind == "HSVTool":
            return Evaluator._apply_mask_mix(src, legacy(src, full, [mask, None, None, None]), mask=None, mix=full.get("mix", 1.0))
        return Evaluator._apply_mask_mix(src, legacy(src, full), mask=mask, mix=full.get("mix", 1.0))


# ================================================================================================================ ColorCorrect
OLD_KNOBS["ColorCorrect"] = ("divide_by_alpha",)
_CC_BASE = {"gain": 1.5, "offset": 0.1, "shadows_gain": 1.8, "highlights_gain": 0.7}
_CC_CHANGES = {
    "channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
    "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
    "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "lookup": {"shadows_curve": '{"interpolation":"smooth","points":[[0,1],[0.4,0]]}',
               "highlights_curve": '{"interpolation":"smooth","points":[[0.2,0],[0.6,1]]}'},
    "saturation": {"saturation": 0.4}, "contrast": {"contrast": 1.5}, "gamma": {"gamma": 1.5}, "gain": {"gain": 1.2}, "offset": {"offset": -0.1},
}
for _zone in ("shadows", "midtones", "highlights"):
    for _name, _value in (("saturation", 0.4), ("contrast", 1.5), ("gamma", 1.5), ("gain", 1.2), ("offset", -0.1)):
        _CC_CHANGES[f"{_zone}.{_name}"] = {f"{_zone}_{_name}": _value}


class ColorCorrectKnobsActTests(KnobsActTests):
    BASES = {"ColorCorrect": _CC_BASE}
    CHANGES = {"ColorCorrect": _CC_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("ColorCorrect")


class ColorCorrectTilesTests(TilesMatchTests):
    TILE_PARAMS = {"ColorCorrect": [dict(_CC_BASE), dict(saturation=0.5, contrast=1.4, gamma=1.3, midtones_offset=0.05, divide_by_alpha=1)]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("ColorCorrect")


class ColorCorrectOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"ColorCorrect": ("_color_correct", [dict(_CC_BASE), dict(saturation=0.5, contrast=1.4, gamma=1.3, midtones_offset=0.05, mix=0.6)])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("ColorCorrect")

    def test_a_new_node_corrects_the_stored_colour(self):
        self.assertEqual(SPECS["ColorCorrect"]["params"]["divide_by_alpha"], 0)


# ================================================================================================================ Saturation
OLD_KNOBS["Saturation"] = ("luminance_math",)
_SAT_BASE = {"saturation": 0.4}
_SAT_CHANGES = {
    "channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
    "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
    "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "saturation": {"saturation": 1.8},
}


class SaturationKnobsActTests(KnobsActTests):
    BASES = {"Saturation": _SAT_BASE}
    CHANGES = {"Saturation": _SAT_CHANGES}
    # Saturation is a degree-one homogeneous function of the colour (a lerp toward a grey that scales with it), so dividing by alpha and
    # multiplying back gives the same pixels, in Nuke as here: the two (un)premult knobs are wired and cannot show.
    NO_EFFECT = {"Saturation": ("unpremult", "invert_unpremult")}

    def test_every_paired_knob_acts(self):
        self._check_kind("Saturation")

    def test_each_luminance_math_changes_the_pixels(self):
        outputs = {mode: run_node("Saturation", dict(_SAT_BASE, luminance_math=mode)).pixels
                   for mode in ("Rec 709", "Ccir 601", "Average", "Maximum", "Rec 709 (old weights)")}
        modes = list(outputs)
        for i, first in enumerate(modes):
            for second in modes[i + 1:]:
                self.assertFalse(np.array_equal(outputs[first], outputs[second]), f"{first} and {second} give the same pixels")


class SaturationTilesTests(TilesMatchTests):
    TILE_PARAMS = {"Saturation": [{"saturation": 0.3}, {"saturation": 1.7, "luminance_math": "Maximum"}, {"saturation": -0.5, "luminance_math": "Ccir 601"},
                                  {"saturation": 0.6, "luminance_math": "Rec 709 (old weights)"}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("Saturation")


class SaturationOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"Saturation": ("_saturation", [{"saturation": 0.3}, {"saturation": 1.7, "mix": 0.6}, {"saturation": -0.5}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("Saturation")

    def test_a_new_node_uses_nukes_rec_709(self):
        self.assertEqual(SPECS["Saturation"]["params"]["luminance_math"], "Rec 709")
        pixel = np.array([[[0.3, 0.5, 0.2, 1.0]]], np.float32)
        out = Evaluator._saturation(pixel, {"saturation": 0.0})
        grey = 0.2125 * 0.3 + 0.7154 * 0.5 + 0.0721 * 0.2
        np.testing.assert_allclose(out[0, 0, :3], (grey,) * 3, rtol=1e-6)
        np.testing.assert_array_equal(out[0, 0, 3], 1.0)



# ================================================================================================================ HueCorrect
OLD_KNOBS["HueCorrect"] = ("older_curve_maths",)
_HC_CURVES = ("sat", "lum", "red", "green", "blue", "r_sup", "g_sup", "b_sup")
_HC_BASE = {"curve_sat": '{"interpolation":"linear","points":[[0,0.5],[360,0.5]]}', "curve_lum": '{"interpolation":"linear","points":[[0,1.5],[360,1.5]]}'}
_HC_CHANGES = {
    "channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
    "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
    "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1}),
    "hue": {"curve_sat": '{"interpolation":"linear","points":[[0,1.2],[360,1.2]]}'},
    "mix_luminance": {"mix_luminance": 1.0},
}


class HueCorrectKnobsActTests(KnobsActTests):
    BASES = {"HueCorrect": _HC_BASE}
    CHANGES = {"HueCorrect": _HC_CHANGES}
    # Every HueCorrect curve is read at the pixel's hue and saturation, both unchanged by scaling the colour, and applied as a gain or a lerp that scales
    # with it, so dividing by alpha and multiplying back gives the same pixels.
    NO_EFFECT = {"HueCorrect": ("unpremult", "invert_unpremult")}

    def test_every_paired_knob_acts(self):
        self._check_kind("HueCorrect")

    def test_each_curve_of_the_hue_knob_acts(self):
        flat = lambda v: f'{{"interpolation":"linear","points":[[0,{v}],[360,{v}]]}}'
        neutral = run_node("HueCorrect", {})
        for name in _HC_CURVES:
            value = 0.4 if name.endswith("_sup") or name in ("sat",) else 1.6
            changed = run_node("HueCorrect", {f"curve_{name}": flat(value)})
            with self.subTest(curve=name):
                self.assertFalse(np.array_equal(neutral.pixels, changed.pixels))

    def test_sat_thrsh_protects_low_saturation_pixels(self):
        flat = '{"interpolation":"linear","points":[[0,2],[360,2]]}'
        plain = run_node("HueCorrect", {"curve_red": flat}).pixels
        guarded = run_node("HueCorrect", {"curve_red": flat, "sat_thrsh": 0.5}).pixels
        self.assertFalse(np.array_equal(plain, guarded))


class HueCorrectTilesTests(TilesMatchTests):
    TILE_PARAMS = {"HueCorrect": [dict(_HC_BASE), {"curve_g_sup": '{"interpolation":"linear","points":[[0,1],[60,1],[120,0],[180,1],[360,1]]}'},
                                  {"curve_red": '{"interpolation":"smooth","points":[[0,2],[60,1],[300,1],[360,2]]}', "sat_thrsh": 0.3},
                                  dict(_HC_BASE, older_curve_maths=1)]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("HueCorrect")


class HueCorrectOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"HueCorrect": ("_hue_correct", [dict(_HC_BASE), {"curve_r_sup": '{"interpolation":"linear","points":[[0,0],[60,0.7],[360,0]]}', "mix": 0.6},
                                              {"curve_red": '{"interpolation":"smooth","points":[[0,2],[60,1],[300,1],[360,2]]}', "sat_thrsh": 0.3, "hue_shift": 40.0}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("HueCorrect")

    def test_a_new_node_has_neutral_suppression_curves(self):
        for name in ("r_sup", "g_sup", "b_sup"):
            self.assertEqual(SPECS["HueCorrect"]["params"][f"curve_{name}"], '{"interpolation":"linear","points":[[0,1],[360,1]]}')
        pixels = fixture_image()
        np.testing.assert_array_equal(Evaluator._hue_correct(pixels, dict(SPECS["HueCorrect"]["params"])), pixels)



# ================================================================================================================ HSVTool
OLD_KNOBS["HSVTool"] = ("adjust_by_scaling",)
_HSV_BASE = {"hue_rotation": 40.0, "sat_adjust": -0.2, "brt_adjust": -0.1}


def _hsv_change(**kw):
    return kw


_HSV_CHANGES = {
    "huesrcs": {"hue_range_min": 60.0, "hue_range_max": 200.0},
    "satsrcs": {"saturation_range_min": 0.3},
    "brtsrcs": {"brightness_range_min": 0.4},
    "srccolor": ({"color_replace": 1, "dstcolor_r": 0.2, "dstcolor_g": 0.6, "dstcolor_b": 0.9}, {"srccolor_r": 0.8, "srccolor_g": 0.3, "srccolor_b": 0.2}),
    "dstcolor": ({"color_replace": 1, "srccolor_r": 0.8, "srccolor_g": 0.3, "srccolor_b": 0.2}, {"dstcolor_r": 0.2, "dstcolor_g": 0.6, "dstcolor_b": 0.9}),
    "huerotation": {"hue_rotation": 90.0},
    "huerolloffrange": ({"hue_range_min": 100.0, "hue_range_max": 140.0}, {"hue_rolloff": 30.0}),
    "satrolloffrange": ({"saturation_range_min": 0.4, "saturation_range_max": 0.7}, {"saturation_rolloff": 0.2}),
    "brtrolloffrange": ({"brightness_range_min": 0.4, "brightness_range_max": 0.7}, {"brightness_rolloff": 0.2}),
    "saturation": {"sat_adjust": 0.1},
    "setsaturation": {"set_saturation": 1},
    "brightness": {"brt_adjust": 0.1},
    "setbrightness": {"set_brightness": 1},
    "output": {"output_alpha": "combined_weight"},
    # Lane 2 step P (11/4): the per-range masks pick a channel of the input, so each acts with an adjustment made; the inverts act with the mask chosen.
    "Conversion": {"output_alpha": "hue_weight"},
    "invert_alpha_mask": ({"output_alpha": "combined_weight"}, {"invert_alpha_mask": 1}),
    "HMask": ({"hue_rotation": 90.0}, {"HMask": "rgba.red"}),
    "invert_hue_mask": ({"hue_rotation": 90.0, "HMask": "rgba.red"}, {"invert_hue_mask": 1}),
    "SMask": ({"sat_adjust": -0.5}, {"SMask": "rgba.red"}),
    "invert_sat_mask": ({"sat_adjust": -0.5, "SMask": "rgba.red"}, {"invert_sat_mask": 1}),
    "BMask": ({"brt_adjust": 0.3}, {"BMask": "rgba.red"}),
    "invert_bright_mask": ({"brt_adjust": 0.3, "BMask": "rgba.red"}, {"invert_bright_mask": 1}),
}


class HSVToolKnobsActTests(KnobsActTests):
    BASES = {"HSVTool": _HSV_BASE}
    CHANGES = {"HSVTool": _HSV_CHANGES}

    def test_every_paired_knob_acts(self):
        self._check_kind("HSVTool")

    def test_the_adjustments_are_added(self):
        """Foundry: a saturation adjustment of -1 desaturates completely and a brightness adjustment of -1 reaches black, whatever the starting value."""
        colours = colour_image()
        gray = render("HSVTool", dict(SPECS["HSVTool"]["params"], sat_adjust=-1.0), source=colours).pixels
        spread = gray[..., :3].max(-1) - gray[..., :3].min(-1)
        np.testing.assert_allclose(spread, 0.0, atol=1e-6)
        black = render("HSVTool", dict(SPECS["HSVTool"]["params"], brt_adjust=-1.0), source=colours).pixels
        np.testing.assert_allclose(black[..., :3], 0.0, atol=1e-6)


def _run_node_with(kind, params, source):
    return run_node(kind, params, source=source)


class HSVToolTilesTests(TilesMatchTests):
    TILE_PARAMS = {"HSVTool": [dict(_HSV_BASE), {"hue_rotation": 60.0, "hue_range_min": 100.0, "hue_range_max": 140.0, "hue_rolloff": 25.0, "sat_adjust": 0.3},
                               {"set_saturation": 1, "sat_adjust": 0.5, "brt_adjust": -0.2, "adjust_by_scaling": 1},
                               {"output_alpha": "combined_weight", "hue_rotation": 20.0, "saturation_range_min": 0.3, "saturation_rolloff": 0.1}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("HSVTool")


class HSVToolOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"HSVTool": ("_hsv_tool", [dict(_HSV_BASE), {"hue_rotation": 60.0, "hue_range_min": 100.0, "hue_range_max": 140.0, "hue_rolloff": 25.0, "sat_adjust": 0.3, "mix": 0.6},
                                        {"set_saturation": 1, "sat_adjust": 0.5, "brt_adjust": -0.2}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("HSVTool")

    def test_a_new_node_adds_the_adjustment(self):
        self.assertEqual(SPECS["HSVTool"]["params"]["adjust_by_scaling"], 0)
        pixel = np.array([[[0.6, 0.3, 0.3, 1.0]]], np.float32)           # saturation 0.5, value 0.6
        out = Evaluator._hsv_tool(pixel, dict(SPECS["HSVTool"]["params"], sat_adjust=-0.25))
        self.assertAlmostEqual(float(out[0, 0, :3].min() / out[0, 0, :3].max()), 1.0 - 0.25, places=5)      # saturation 0.25 (a scaling would give 0.375)


# ======================================================================================================== Toe, SoftClip, Posterize
_TOE_BASE = {"toe_lift": 0.1}
_CHROME_CHANGES = {"channels": CHROME["channels"], "maskChannelInput": CHROME["maskChannelInput"], "invert_mask": CHROME["invert_mask"],
                   "fringe": CHROME["fringe"], "unpremult": CHROME["unpremult"], "mix": CHROME["mix"], "inject": {"inject": 1},
                   "invert_unpremult": ({"unpremult": "rgba.alpha"}, {"invert_unpremult": 1})}


class ToeKnobsActTests(KnobsActTests):
    BASES = {"Toe": _TOE_BASE}
    CHANGES = {"Toe": dict(_CHROME_CHANGES, lift={"toe_lift": 0.2}, invert={"invert_mask": 1})}

    def test_every_paired_knob_acts(self):
        self._check_kind("Toe")

    def test_a_new_toe_does_nothing_until_the_lift_is_raised(self):
        self.assertEqual(SPECS["Toe"]["params"]["toe_lift"], 0.0)
        pixels = fixture_image()
        np.testing.assert_array_equal(Evaluator._toe(pixels, dict(SPECS["Toe"]["params"])), pixels)


class ToeTilesTests(TilesMatchTests):
    TILE_PARAMS = {"Toe": [dict(_TOE_BASE), {"toe_lift": 0.3, "toe": 0.5}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("Toe")


_SC_BASE = {"conversion": "logarithmic compress", "softclip_min": 0.5, "softclip_max": 4.0}


class SoftClipKnobsActTests(KnobsActTests):
    BASES = {"SoftClip": _SC_BASE}
    CHANGES = {"SoftClip": {"conversion": {"conversion": "preserve hue and saturation"}, "softclip_min": {"softclip_min": 0.2}, "softclip_max": {"softclip_max": 8.0}}}

    def test_every_paired_knob_acts(self):
        self._check_kind("SoftClip")

    def test_the_logarithmic_curve_keeps_its_end_points_and_rises(self):
        values = np.linspace(-0.5, 4.0, 91, dtype=np.float32)
        pixels = np.zeros((1, values.size, 4), np.float32)
        pixels[..., :3] = values[None, :, None]
        pixels[..., 3] = 1.0
        out = Evaluator._softclip(pixels, dict(SPECS["SoftClip"]["params"], **_SC_BASE))[0, :, 0].astype(np.float64)
        np.testing.assert_allclose(out[values <= 0.5], values[values <= 0.5], atol=1e-6)
        self.assertAlmostEqual(out[-1], 1.0, places=5)
        self.assertTrue((np.diff(out) > 0).all())
        inside = (values > 0.5) & (values <= 4.0)
        self.assertTrue(((out[inside] > 0.5) & (out[inside] <= 1.0 + 1e-6)).all())

    def test_the_two_preserving_modes_end_below_softclip_max_and_keep_what_they_say(self):
        pixel = np.array([[[1.5, 0.6, 0.3, 1.0]]], np.float32)
        base = dict(SPECS["SoftClip"]["params"], softclip_max=1.0)
        both = Evaluator._softclip(pixel, dict(base, conversion="preserve hue and saturation"))[0, 0, :3].astype(np.float64)
        self.assertAlmostEqual(both.max(), 1.0, places=5)
        np.testing.assert_allclose(both / both.max(), np.array([1.5, 0.6, 0.3]) / 1.5, rtol=1e-5)               # hue and saturation: a pure scaling
        bright = Evaluator._softclip(pixel, dict(base, conversion="preserve hue and brightness"))[0, 0, :3].astype(np.float64)
        self.assertAlmostEqual(bright.max(), 1.0, places=5)
        grey = lambda c: 0.2125 * c[0] + 0.7154 * c[1] + 0.0721 * c[2]
        self.assertAlmostEqual(grey(bright), grey([1.5, 0.6, 0.3]), places=5)                                    # brightness unchanged


class SoftClipTilesTests(TilesMatchTests):
    TILE_PARAMS = {"SoftClip": [dict(_SC_BASE), {"conversion": "preserve hue and saturation", "softclip_max": 1.0},
                                {"conversion": "preserve hue and brightness", "softclip_max": 0.9}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("SoftClip")


OLD_KNOBS["SoftClip"] = ("older_grey",)


class SoftClipOlderDocumentTests(OlderDocumentTests):
    LEGACY = {"SoftClip": ("_softclip", [{"conversion": "preserve hue and brightness", "softclip_max": 0.9}, dict(_SC_BASE, mix=0.6),
                                         {"conversion": "preserve hue and saturation", "softclip_max": 1.0}])}

    def test_an_older_document_keeps_its_pixels(self):
        self._check("SoftClip")


class PosterizeKnobsActTests(KnobsActTests):
    BASES = {"Posterize": {"colors": 6}}
    CHANGES = {"Posterize": dict(_CHROME_CHANGES, Colors={"colors": 3})}

    def test_every_paired_knob_acts(self):
        self._check_kind("Posterize")


class PosterizeTilesTests(TilesMatchTests):
    TILE_PARAMS = {"Posterize": [{"colors": 6}, {"colors": 3, "channels": "red"}]}

    def test_tiles_equal_the_whole_frame(self):
        self._check("Posterize")


if __name__ == "__main__":
    unittest.main()
