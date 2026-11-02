"""Behaviour cases for the Colour group's grading nodes (2D behaviour lane 2, step A, November 2026).

ColorCorrect, Saturation, HueCorrect, HSVTool, Toe, SoftClip and Posterize. Each case is one stated behaviour of the Nuke node run on the
16 x 12 fixture through `Evaluator._kernel`, the way `tools/nuke_fidelity_cases.py` runs the Grade and Exposure cases.

Where the sources come from
* `guide`: the Nuke 17.0 Reference Guide (or the Foundry user-guide page it links) states the behaviour in words.
* `convention`: Foundry publishes no formula, and the form used is a long-published measurement or Foundry's own SDK example. The sources are
  - the ColorCorrect, HueCorrect and Saturation measurements of A. Geremia, "The Math Series" (www.andreageremia.it/tutorial_math_colorcorrect.html and
    .../tutorial_math_hue.html), taken from Nuke 17 by feeding known values and reading the results back; the Saturation page is a reading of the
    SDK example source that Foundry ships (its four luminance maths);
  - Natron's HSVTool documentation, which restates Nuke's adjustment as an addition;
  - Foundry's "Correcting HSV" page for the HSVTool end points.
* A case whose behaviour cannot be written down without guessing is `unscored` with the reason.

Nothing here has been compared with a real Nuke render yet; the reference pack (tools/make_nuke_reference_pack.py) writes the Nuke scripts.
Only names defined above `CASES = _cases()` in `nuke_fidelity_cases` are imported (this module is imported by that line's hook).
"""
from __future__ import annotations

import numpy as np

from nuke_fidelity_cases import Case, _all_valid

REC709 = (0.2125, 0.7154, 0.0721)
CCIR601 = (0.299, 0.587, 0.114)
_CC_SOURCE = "A. Geremia, 'The ColorCorrect Node in Nuke' (andreageremia.it/tutorial_math_colorcorrect.html), measured on Nuke 17"
_SAT_SOURCE = "A. Geremia, 'Saturation, Hue and HSV in Nuke' (andreageremia.it/tutorial_math_hue.html), a reading of Foundry's SDK Saturation example"


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _grey(rgb, weights):
    return (rgb * np.array(weights)).sum(axis=-1, keepdims=True)


def _finite(ref):
    return np.isfinite(ref)


# ------------------------------------------------------------------------------------------------------------ ColorCorrect
_ZONES = ("shadows", "midtones", "highlights")
_PIVOT = {"shadows": 0.04, "midtones": 0.18, "highlights": 0.65}
_KNOBS = ("saturation", "contrast", "gamma", "gain", "offset")
_NEUTRAL = {"saturation": 1.0, "contrast": 1.0, "gamma": 1.0, "gain": 1.0, "offset": 0.0}


def color_correct_reference(rgb, master=None, zones=None):
    """Nuke's ColorCorrect on the stored colour of `rgb` (no un-premultiply unless a knob asks for it).

    Three ranges from the luminance (Rec 709, 0.2125 / 0.7154 / 0.0721): shadows = 1 - smoothstep(0, 0.09, L), highlights = smoothstep(0.5, 1, L),
    midtones the rest. Each range corrects the pixel with the master woven in knob by knob, in the order saturation, contrast, gamma, gain, offset:
    saturation, gamma and gain multiply (range x master), offset adds (range + master), contrast is a power around the range's own pivot
    (0.04, 0.18, 0.65) and then around the master's 0.18. Values at or below zero skip contrast and gamma and are scaled by what 1 becomes.
    """
    m = dict(_NEUTRAL, **(master or {}))
    k = {z: dict(_NEUTRAL, **((zones or {}).get(z, {}))) for z in _ZONES}
    lum = _grey(rgb, REC709)
    shadows = 1.0 - _smoothstep(0.0, 0.09, lum)
    highlights = _smoothstep(0.5, 1.0, lum)
    weight = {"shadows": shadows, "midtones": 1.0 - shadows - highlights, "highlights": highlights}
    out = np.zeros_like(rgb)
    with np.errstate(all="ignore"):
        for z in _ZONES:
            z_knob = k[z]
            x = lum + z_knob["saturation"] * m["saturation"] * (rgb - lum)
            gamma = z_knob["gamma"] * m["gamma"]

            def tone(v):
                v = (v / _PIVOT[z]) ** z_knob["contrast"] * _PIVOT[z]
                v = (v / 0.18) ** m["contrast"] * 0.18
                return v ** (1.0 / gamma)
            positive = x > 0.0
            y = np.where(positive, tone(np.where(positive, x, 1.0)), x * tone(np.float64(1.0)))
            y = y * (z_knob["gain"] * m["gain"]) + (z_knob["offset"] + m["offset"])
            out = out + weight[z] * y
    return out


def _cc_expected(master=None, zones=None, mix=1.0, use_mask=False):
    def expected(fx):
        src = fx["A"].astype(np.float64)
        corrected = src.copy()
        corrected[..., :3] = color_correct_reference(src[..., :3], master, zones)
        gate = mix * (fx["mask"][..., 3:4].astype(np.float64) if use_mask else 1.0)
        ref = src + (corrected - src) * gate
        return ref, _finite(ref)
    return expected


def _cc_case(case_id, nb_params, nuke_knobs, statement, master=None, zones=None, mix=1.0, use_mask=False, source="convention",
             expected=None, extra_nb=()):
    note = statement + ("" if source == "guide" else f" ({_CC_SOURCE})")
    return Case(case_id, "ColorCorrect", "ColorCorrect", tuple(nuke_knobs), ("A", "mask") if use_mask else ("A",), "ColorCorrect", dict(nb_params),
                ("A", "mask") if use_mask else ("A",), expected or _cc_expected(master, zones, mix, use_mask), source, note)


def _color_correct_cases():
    out = []
    out.append(_cc_case("cc_master_chain", {"saturation": 1.3, "contrast": 1.2, "gamma": 1.4, "gain": 1.5, "offset": 0.05},
                        (("saturation", 1.3), ("contrast", 1.2), ("gamma", 1.4), ("gain", 1.5), ("offset", 0.05)),
                        "master knobs run in the order saturation, contrast, gamma, gain, offset",
                        master=dict(saturation=1.3, contrast=1.2, gamma=1.4, gain=1.5, offset=0.05)))
    out.append(_cc_case("cc_gamma_true_power", {"gamma": 2.0}, (("gamma", 2.0),),
                        "gamma is a true power x ** (1 / gamma) for every value above zero, above 1 included (Grade's is a straight line there)",
                        master=dict(gamma=2.0)))
    out.append(_cc_case("cc_saturation_zero", {"saturation": 0.0}, (("saturation", 0.0),),
                        "saturation 0 lerps to the pixel's Rec 709 luminance",
                        master=dict(saturation=0.0),
                        expected=lambda fx: (lambda src: (np.concatenate([_grey(src[..., :3], REC709).repeat(3, axis=-1), src[..., 3:4]], axis=-1),
                                                          _all_valid(src)))(fx["A"].astype(np.float64))))
    out.append(_cc_case("cc_saturation_two", {"saturation": 2.0}, (("saturation", 2.0),),
                        "saturation is L + s * (colour - L), unclamped: 2 can drive a channel negative", master=dict(saturation=2.0)))
    out.append(_cc_case("cc_master_contrast_pivot", {"contrast": 1.6}, (("contrast", 1.6),),
                        "master contrast is a power around the pivot 0.18 (middle grey); values at or below 0 are scaled by what 1 becomes",
                        master=dict(contrast=1.6)))
    for zone in _ZONES:
        out.append(_cc_case(f"cc_{zone}_contrast_pivot", {f"{zone}_contrast": 1.6}, ((f"{zone}.contrast", 1.6),),
                            f"{zone} contrast is a power around that range's own pivot ({_PIVOT[zone]})", zones={zone: dict(contrast=1.6)}))
    for zone in _ZONES:
        out.append(_cc_case(f"cc_{zone}_offset", {f"{zone}_offset": 0.1}, ((f"{zone}.offset", 0.1),),
                            f"a {zone} offset is added in proportion to how much of the pixel is {zone}", zones={zone: dict(offset=0.1)}))
    for zone in _ZONES:
        out.append(_cc_case(f"cc_{zone}_saturation_gamma_gain", {f"{zone}_saturation": 0.5, f"{zone}_gamma": 1.5, f"{zone}_gain": 1.3},
                            ((f"{zone}.saturation", 0.5), (f"{zone}.gamma", 1.5), (f"{zone}.gain", 1.3)),
                            f"the {zone} range's saturation, gamma and gain, blended by its weight",
                            zones={zone: dict(saturation=0.5, gamma=1.5, gain=1.3)}))

    def ranges_sum_to_one(fx):
        ref = fx["A"].astype(np.float64).copy()
        ref[..., :3] *= 1.5
        return ref, _all_valid(ref)
    out.append(_cc_case("cc_ranges_sum_to_one", {"shadows_gain": 1.5, "midtones_gain": 1.5, "highlights_gain": 1.5},
                        (("shadows.gain", 1.5), ("midtones.gain", 1.5), ("highlights.gain", 1.5)),
                        "the three range weights always add up to 1 (the midtones are 1 minus the other two), so the same gain on every range is that gain",
                        source="guide", expected=ranges_sum_to_one))
    out.append(_cc_case("cc_master_woven_into_ranges", {"gain": 2.0, "midtones_offset": 0.1}, (("gain", 2.0), ("midtones.offset", 0.1)),
                        "the master is woven into each range knob by knob: a master gain of 2 with a midtones offset of 0.1 gives 2x + 0.1 weighted by the midtones, not 2(x + 0.1)",
                        master=dict(gain=2.0), zones={"midtones": dict(offset=0.1)}))
    out.append(_cc_case("cc_everything", {"saturation": 1.2, "contrast": 1.3, "gamma": 1.2, "gain": 1.1, "offset": 0.02,
                                          "shadows_saturation": 0.8, "shadows_contrast": 1.4, "midtones_gamma": 1.3, "midtones_gain": 0.9,
                                          "highlights_contrast": 0.8, "highlights_offset": -0.05},
                        (("saturation", 1.2), ("contrast", 1.3), ("gamma", 1.2), ("gain", 1.1), ("offset", 0.02), ("shadows.saturation", 0.8),
                         ("shadows.contrast", 1.4), ("midtones.gamma", 1.3), ("midtones.gain", 0.9), ("highlights.contrast", 0.8),
                         ("highlights.offset", -0.05)),
                        "master and all three ranges together",
                        master=dict(saturation=1.2, contrast=1.3, gamma=1.2, gain=1.1, offset=0.02),
                        zones={"shadows": dict(saturation=0.8, contrast=1.4), "midtones": dict(gamma=1.3, gain=0.9),
                               "highlights": dict(contrast=0.8, offset=-0.05)}))
    out.append(_cc_case("cc_stored_colour_not_straight", {"offset": 0.2}, (("offset", 0.2),),
                        "with (un)premult by at none the correction is applied to the stored colour: an offset of 0.2 adds 0.2 to the rgb of a half-transparent pixel",
                        source="guide", master=dict(offset=0.2)))
    out.append(_cc_case("cc_mix_half", {"gain": 1.8, "mix": 0.5}, (("gain", 1.8), ("mix", 0.5)),
                        "mix dissolves between the original at 0 and the full correction at 1", source="guide", master=dict(gain=1.8), mix=0.5))
    out.append(_cc_case("cc_mask", {"gain": 1.8}, (("gain", 1.8),),
                        "the correction is limited to the non-black areas of the mask input", source="guide", master=dict(gain=1.8), use_mask=True))
    return out


# ------------------------------------------------------------------------------------------------------------- Saturation
def _saturation_expected(weights, saturation, mix=1.0, use_mask=False):
    def expected(fx):
        src = fx["A"].astype(np.float64)
        rgb = src[..., :3]
        y = weights(rgb)
        corrected = src.copy()
        corrected[..., :3] = y + saturation * (rgb - y)
        gate = mix * (fx["mask"][..., 3:4].astype(np.float64) if use_mask else 1.0)
        ref = src + (corrected - src) * gate
        return ref, _finite(ref)
    return expected


def _sat_case(case_id, math_name, math_param, weights, saturation, statement, mix=1.0, use_mask=False, source="convention"):
    knobs = (("saturation", saturation),) + ((("mix", mix),) if mix != 1.0 else ())
    if math_name != "Rec 709":
        knobs += (("luminance_math", math_name.replace(" ", "")),)
    params = {"saturation": saturation, "luminance_math": math_param, **({"mix": mix} if mix != 1.0 else {})}
    inputs = ("A", "mask") if use_mask else ("A",)
    note = statement + ("" if source == "guide" else f" ({_SAT_SOURCE})")
    return Case(case_id, "Saturation", "Saturation", knobs, inputs, "Saturation", params, inputs,
                _saturation_expected(weights, saturation, mix, use_mask), source, note)


def _saturation_cases():
    modes = {
        "Rec 709": ("Rec 709", lambda rgb: _grey(rgb, REC709), "y = 0.2125 r + 0.7154 g + 0.0721 b"),
        "Ccir 601": ("Ccir 601", lambda rgb: _grey(rgb, CCIR601), "y = 0.299 r + 0.587 g + 0.114 b"),
        "Average": ("Average", lambda rgb: rgb.mean(axis=-1, keepdims=True), "y = (r + g + b) / 3"),
        "Maximum": ("Maximum", lambda rgb: rgb.max(axis=-1, keepdims=True), "y = max(r, g, b)"),
    }
    out = []
    for name, (param, weights, text) in modes.items():
        slug = name.lower().replace(" ", "")
        out.append(_sat_case(f"saturation_{slug}_zero", name, param, weights, 0.0,
                             f"Saturation 0 with luminance math {name} gives every channel the grey ({text}): a black and white image"))
        out.append(_sat_case(f"saturation_{slug}_half", name, param, weights, 0.5,
                             f"Saturation 0.5 with luminance math {name} mixes half way to that grey, y + 0.5 * (colour - y)"))
    out.append(_sat_case("saturation_rec709_boost", "Rec 709", "Rec 709", modes["Rec 709"][1], 2.0,
                         "saturation above 1 pushes the colour away from the grey and the node does not clamp, so a channel can go negative"))
    out.append(_sat_case("saturation_rec709_negative", "Rec 709", "Rec 709", modes["Rec 709"][1], -1.0,
                         "a saturation below 0 carries on past the grey the same straight line: the colour inverts about its luminance"))
    out.append(_sat_case("saturation_default_is_identity", "Rec 709", "Rec 709", modes["Rec 709"][1], 1.0,
                         "saturation 1 equals no change", source="guide"))
    out.append(_sat_case("saturation_mix_half", "Rec 709", "Rec 709", modes["Rec 709"][1], 0.0,
                         "mix dissolves between the original at 0 and the full saturation adjustment at 1", mix=0.5, source="guide"))
    out.append(_sat_case("saturation_mask", "Rec 709", "Rec 709", modes["Rec 709"][1], 0.0,
                         "the adjustment is limited to the non-black areas of the mask input", use_mask=True, source="guide"))
    return out


def cases():
    return _color_correct_cases() + _saturation_cases()
