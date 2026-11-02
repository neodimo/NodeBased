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


# -------------------------------------------------------------------------------------------------------------- HueCorrect
_HUE_SOURCE = "A. Geremia, 'Saturation, Hue and HSV in Nuke' (andreageremia.it/tutorial_math_hue.html), each curve measured flat on Nuke 17"


def _hsv(rgb):
    """Hue in degrees (0 red, 120 green, 240 blue), saturation (max - min) / max and value max, per pixel; hue is 0 where there is no chroma."""
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    chroma = mx - mn
    safe = np.where(chroma > 0, chroma, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.where(mx == r, ((g - b) / safe) % 6.0, np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    hue = np.where(chroma > 0, hue, 0.0) % 360.0
    sat = np.where(mx > 0, chroma / np.where(mx > 0, mx, 1.0), 0.0)
    return hue, sat, mx


def _hsv_to_rgb(hue, sat, val):
    sector = (hue % 360.0) / 60.0
    i = np.floor(sector).astype(int) % 6
    f = sector - np.floor(sector)
    p, q, t = val * (1 - sat), val * (1 - sat * f), val * (1 - sat * (1 - f))
    r = np.choose(i, [val, q, p, p, t, val])
    g = np.choose(i, [t, val, val, q, p, p])
    b = np.choose(i, [p, p, t, val, val, q])
    return np.stack([r, g, b], axis=-1)


def _flat(value):
    return f'{{"interpolation":"linear","points":[[0,{value}],[360,{value}]]}}'


def _curve(points):
    return '{"interpolation":"linear","points":' + str([list(p) for p in points]).replace(" ", "") + "}"


def _curve_values(spec, hue):
    """A scalar, or the linear curve through `spec` points (degrees), per pixel hue."""
    if np.isscalar(spec):
        return np.full(hue.shape, float(spec))
    xs, ys = zip(*spec)
    return np.interp(hue, xs, ys)


def hue_correct_reference(rgb, sat=1.0, lum=1.0, red=1.0, green=1.0, blue=1.0, r_sup=1.0, g_sup=1.0, b_sup=1.0):
    """Nuke's HueCorrect curves, each applied the way its flat curve measures: sat is the Saturation node's lerp toward the Rec 709 grey, lum a plain gain
    on all three channels, red / green / blue the gain of one channel, and a suppression pulls a channel that exceeds the smaller of the other two back
    toward that smaller one (0 removes the excess, 1 changes nothing). Each argument is a scalar or a list of (hue degrees, value) points."""
    hue, _s, _v = _hsv(rgb)
    grey = _grey(rgb, REC709)
    out = grey + _curve_values(sat, hue)[..., None] * (rgb - grey)
    out = out * _curve_values(lum, hue)[..., None]
    out = out * np.stack([_curve_values(c, hue) for c in (red, green, blue)], axis=-1)
    pre = out.copy()
    for channel, sup in enumerate((r_sup, g_sup, b_sup)):
        lowest = np.minimum(pre[..., (channel + 1) % 3], pre[..., (channel + 2) % 3])
        pulled = lowest + _curve_values(sup, hue) * (pre[..., channel] - lowest)
        out[..., channel] = np.where(pre[..., channel] > lowest, pulled, pre[..., channel])
    return out


def _hue_expected(away_from=(), width=2.0, mix=1.0, use_mask=False, **curves):
    """`away_from`: hue degrees (curve break points) a pixel must be at least `width` degrees from to be compared."""
    def expected(fx):
        src = fx["A"].astype(np.float64)
        rgb = src[..., :3]
        corrected = src.copy()
        corrected[..., :3] = hue_correct_reference(rgb, **curves)
        gate = mix * (fx["mask"][..., 3:4].astype(np.float64) if use_mask else 1.0)
        ref = src + (corrected - src) * gate
        valid = np.isfinite(ref)
        if away_from:
            hue, _s, _v = _hsv(rgb)
            near = np.zeros(hue.shape, bool)
            for at in away_from:
                d = np.abs(hue - at) % 360.0
                near |= np.minimum(d, 360.0 - d) < width
            valid &= ~near[..., None]
        return ref, valid
    return expected


def _hue_case(case_id, nb_curves, nuke_curves, statement, source="convention", **kw):
    expected = _hue_expected(**kw)
    params = {f"curve_{name}": spec if isinstance(spec, str) else _flat(spec) for name, spec in nb_curves.items()}
    mix, use_mask = kw.get("mix", 1.0), kw.get("use_mask", False)
    if mix != 1.0:
        params["mix"] = mix
    inputs = ("A", "mask") if use_mask else ("A",)
    knobs = (("hue", nuke_curves),) + ((("mix", mix),) if mix != 1.0 else ()) if nuke_curves else ()
    note = statement + ("" if source == "guide" else f" ({_HUE_SOURCE})")
    return Case(case_id, "HueCorrect", "HueCorrect", knobs, inputs, "HueCorrect", params, inputs, expected, source, note)


def _nuke_hue(entries):
    """The `hue` knob text for flat or point curves: Nuke's x axis is the hue angle divided by 60."""
    parts = []
    for name, spec in entries.items():
        if np.isscalar(spec):
            parts.append(f"{{{name} {{curve L {spec}}}}}")
        else:
            parts.append(f"{{{name} {{curve L " + " ".join(f"x{x / 60:g} {y:g}" for x, y in spec) + "}}")
    return "{" + " ".join(parts) + "}"


def _hue_correct_cases():
    out = []
    out.append(_hue_case("huecorrect_defaults", {}, "", "every curve at its default leaves the pixels alone", source="guide"))
    for value in (0.0, 0.5, 2.0):
        out.append(_hue_case(f"huecorrect_sat_{value:g}", {"sat": value}, _nuke_hue({"sat": value}),
                             f"a flat sat curve of {value:g} is the Saturation node's lerp toward the Rec 709 grey (0 gives 0.2125 r + 0.7154 g + 0.0721 b)", sat=value))
    out.append(_hue_case("huecorrect_lum_2", {"lum": 2.0}, _nuke_hue({"lum": 2.0}),
                         "a flat lum curve is a plain gain on all three channels: 2 turns (0.3, 0.5, 0.2) into (0.6, 1.0, 0.4)", lum=2.0))
    for name, value in (("red", 2.0), ("green", 0.5), ("blue", 3.0)):
        out.append(_hue_case(f"huecorrect_{name}_gain", {name: value}, _nuke_hue({name: value}),
                             f"a flat {name} curve is the gain of the {name} channel alone", **{name: value}))
    for name, value in (("r_sup", 0.0), ("g_sup", 0.0), ("b_sup", 0.5), ("g_sup", 0.25)):
        out.append(_hue_case(f"huecorrect_{name}_{value:g}", {name: value}, _nuke_hue({name: value}),
                             f"a flat {name} curve of {value:g}: a channel above the smaller of the other two is pulled back toward it, g = min(r, b) + g_sup * (g - min(r, b)); "
                             "0 removes the excess and 1 changes nothing", **{name: value}))
    bump = [(0, 2.0), (60, 1.0), (300, 1.0), (360, 2.0)]
    out.append(_hue_case("huecorrect_red_gain_around_red", {"red": _curve(bump)}, _nuke_hue({"red": bump}),
                         "a red curve that rises toward 2 at hue 0 multiplies the red channel of reddish pixels most (the curves read the pixel's own hue; "
                         "pixels within 2 degrees of a curve point are left out)", red=bump, away_from=(60, 300)))
    dip = [(0, 1.0), (60, 1.0), (120, 0.0), (180, 1.0), (360, 1.0)]
    out.append(_hue_case("huecorrect_g_sup_around_green", {"g_sup": _curve(dip)}, _nuke_hue({"g_sup": dip}),
                         "a g_sup curve that dips to 0 at hue 120 removes green spill from greenish pixels only", g_sup=dip, away_from=(60, 120, 180)))
    out.append(_hue_case("huecorrect_mix_half_masked", {"lum": 2.0}, _nuke_hue({"lum": 2.0}),
                         "mix dissolves between the original and the correction, and the mask input limits it", source="guide", lum=2.0, mix=0.5, use_mask=True))
    out.append(Case("huecorrect_sat_thrsh", "HueCorrect", "HueCorrect", (), ("A",), "HueCorrect", None, ("A",),
                    lambda fx: (fx["A"].astype(np.float64), np.zeros(fx["A"].shape, bool)), "convention",
                    f"sat_thrsh fades the corrections out on low-saturation pixels ({_HUE_SOURCE})", inexpressible="",
                    unscored="the only published measurements are two points (threshold 0.5: a red gain of 2 became 1.4 at saturation 0.5 and 1.64 at saturation 0.83), which do not fix the fade's shape"))
    return out


# ---------------------------------------------------------------------------------------------------------------- HSVTool
_HSV_SOURCE = ("Foundry's 'Correcting HSV' page (rotation 0 to 360 on the colour wheel; saturation adjustment from -1, completely desaturated, to 1, completely "
               "saturated; brightness adjustment from -1, black, to 1, white) and Natron's HSVTool documentation, which adds the adjustment to the value")


def hsv_tool_reference(rgb, hue_range=(0.0, 360.0), rotation=0.0, sat_range=(0.0, 1.0), sat_adjust=0.0, force_sat=False,
                       brt_range=(0.0, 1.0), brt_adjust=0.0, force_brt=False, margin=1e-4):
    """Hue rotation, saturation and brightness adjustment inside hard ranges (no rolloff). A pixel takes the effect when its hue, saturation and value are
    all inside their ranges; the rotation is added to the hue in degrees, and the saturation and brightness adjustments are added to the value
    (or replace it, with force). Returns (rgb out, inside mask, clear mask); `clear` is False for pixels that sit on a range edge or whose result
    would need a clamp the sources do not state (saturation above 1, brightness outside 0 to 1).
    """
    hue, sat, val = _hsv(rgb)
    inside = ((hue >= hue_range[0]) & (hue <= hue_range[1]) & (sat >= sat_range[0]) & (sat <= sat_range[1]) & (val >= brt_range[0]) & (val <= brt_range[1]))
    clear = np.ones(hue.shape, bool)
    for value, (lo, hi) in ((hue, hue_range), (sat, sat_range), (val, brt_range)):
        clear &= (np.abs(value - lo) > margin * max(1.0, abs(hi - lo))) & (np.abs(value - hi) > margin * max(1.0, abs(hi - lo)))
    new_sat = np.where(inside, sat_adjust if force_sat else sat + sat_adjust, sat)
    new_val = np.where(inside, brt_adjust if force_brt else val + brt_adjust, val)
    clear &= (new_sat <= 1.0) & (new_sat >= 0.0) & (new_val >= 0.0) & (new_val <= 1.0)
    out = np.where(inside[..., None], _hsv_to_rgb(hue + np.where(inside, rotation, 0.0), new_sat, new_val), rgb)
    return out, inside, clear


def _hsv_expected(rolloff=None, **kw):
    def expected(fx):
        src = fx["C"].astype(np.float64)
        rgb = src[..., :3]
        out, inside, clear = hsv_tool_reference(rgb, **kw)
        ref = src.copy()
        ref[..., :3] = out
        valid = np.zeros(src.shape, bool)
        in_unit = ((rgb >= 0.0) & (rgb <= 1.0)).all(axis=-1)
        keep = in_unit & clear
        if rolloff is not None:
            # With a rolloff, pixels in the band beyond a range edge take a fraction of the effect whose shape the sources do not state: only the
            # pixels inside the range and the pixels farther outside than the rolloff are compared.
            hue, sat, val = _hsv(rgb)
            lo, hi = kw.get("hue_range", (0.0, 360.0))
            keep &= inside | ((hue < lo - rolloff) | (hue > hi + rolloff))
        valid[..., :3] = keep[..., None]
        return ref, valid
    return expected


def _hsv_case(case_id, nb_params, nuke_knobs, statement, rolloff=None, **kw):
    inputs = ("C",)
    return Case(case_id, "HSVTool", "HSVTool", tuple(nuke_knobs), inputs, "HSVTool", dict(nb_params), inputs, _hsv_expected(rolloff, **kw), "convention",
                f"{statement} ({_HSV_SOURCE}); compared on fixture C (colour inside 0 to 1), away from the range edges and where no clamp is in question")


def _hsv_tool_cases():
    out = []
    out.append(_hsv_case("hsv_defaults_leave_colour", {}, (), "every control at its default leaves the colour alone"))
    out.append(_hsv_case("hsv_hue_rotation_40", {"hue_rotation": 40.0}, (("huerotation", 40.0),), "Rotation adds degrees to every hue", rotation=40.0))
    out.append(_hsv_case("hsv_hue_rotation_in_range", {"hue_rotation": 60.0, "hue_range_min": 100.0, "hue_range_max": 140.0},
                         (("huerotation", 60.0), ("huesrcs", "{100 140}")), "only the hues inside the Range are rotated", rotation=60.0, hue_range=(100.0, 140.0)))
    out.append(_hsv_case("hsv_hue_rotation_negative", {"hue_rotation": -75.0, "hue_range_min": 0.0, "hue_range_max": 200.0},
                         (("huerotation", -75.0), ("huesrcs", "{0 200}")), "a negative rotation turns the hue the other way, wrapping past red",
                         rotation=-75.0, hue_range=(0.0, 200.0)))
    out.append(_hsv_case("hsv_saturation_adjust_down", {"sat_adjust": -0.3}, (("saturation", -0.3),),
                         "a saturation Adjustment is added to the saturation (-1 desaturates completely)", sat_adjust=-0.3))
    out.append(_hsv_case("hsv_saturation_adjust_up_in_range", {"sat_adjust": 0.25, "saturation_range_min": 0.1, "saturation_range_max": 0.6},
                         (("saturation", 0.25), ("satsrcs", "{0.1 0.6}")), "only pixels whose saturation is inside the Range take the adjustment",
                         sat_adjust=0.25, sat_range=(0.1, 0.6)))
    out.append(_hsv_case("hsv_saturation_force", {"sat_adjust": 0.5, "set_saturation": 1, "saturation_range_min": 0.2, "saturation_range_max": 1.0},
                         (("saturation", 0.5), ("setsaturation", True), ("satsrcs", "{0.2 1}")), "Force Value sets the saturation to the Adjustment instead of adding it",
                         sat_adjust=0.5, force_sat=True, sat_range=(0.2, 1.0)))
    out.append(_hsv_case("hsv_brightness_adjust_down", {"brt_adjust": -0.2}, (("brightness", -0.2),),
                         "a brightness Adjustment is added to the value (-1 reaches black)", brt_adjust=-0.2))
    out.append(_hsv_case("hsv_brightness_force_in_range", {"brt_adjust": 0.5, "set_brightness": 1, "brightness_range_min": 0.3, "brightness_range_max": 1.0},
                         (("brightness", 0.5), ("setbrightness", True), ("brtsrcs", "{0.3 1}")),
                         "Force Value sets the brightness of the pixels inside the Range", brt_adjust=0.5, force_brt=True, brt_range=(0.3, 1.0)))
    out.append(_hsv_case("hsv_all_three_ranges", {"hue_rotation": 30.0, "sat_adjust": 0.1, "brt_adjust": -0.1, "hue_range_min": 0.0, "hue_range_max": 180.0,
                                                  "saturation_range_min": 0.3, "saturation_range_max": 1.0, "brightness_range_min": 0.2, "brightness_range_max": 1.0},
                         (("huerotation", 30.0), ("saturation", 0.1), ("brightness", -0.1), ("huesrcs", "{0 180}"), ("satsrcs", "{0.3 1}"), ("brtsrcs", "{0.2 1}")),
                         "a pixel takes the corrections only when its hue, saturation and brightness are all inside their Ranges",
                         rotation=30.0, sat_adjust=0.1, brt_adjust=-0.1, hue_range=(0.0, 180.0), sat_range=(0.3, 1.0), brt_range=(0.2, 1.0)))
    out.append(_hsv_case("hsv_hue_rolloff_reaches_past_the_range",
                         {"hue_rotation": 60.0, "hue_range_min": 100.0, "hue_range_max": 140.0, "hue_rolloff": 20.0},
                         (("huerotation", 60.0), ("huesrcs", "{100 140}"), ("huerolloffrange", 20.0)),
                         "the Range Rolloff lets the correction fade out past the range; inside it the effect is whole and farther than the rolloff there is none",
                         rolloff=20.0, rotation=60.0, hue_range=(100.0, 140.0)))
    unscored = (
        ("hsv_rolloff_shape", (("huesrcs", "{100 140}"), ("huerolloffrange", 20.0)),
         "the shape of the fade inside a Range Rolloff (NodeBased falls off linearly)",
         "neither the guide nor the sources say whether the fade is linear, smooth or something else, so a pixel inside the rolloff band has no stated value"),
        ("hsv_default_output_matte", (),
         "what the output channel holds with the controls at their defaults",
         "the 17.0 page lists the output checkbox as enabled with rgba.alpha and 'is' hue, which reads as a default HSVTool writing the hue range matte into alpha "
         "(Natron's port does); NodeBased keeps the source alpha. Whether Nuke's default really overwrites alpha cannot be confirmed without a render"),
        ("hsv_color_replacement", (("srccolor", "{0.8 0.3 0.2}"), ("dstcolor", "{0.2 0.5 0.9}")),
         "srccolor and dstcolor replace one colour with another",
         "in Nuke the two pickers are a helper that sets the ranges and adjustments (Natron: 'Changing this parameter sets the hue, saturation and brightness "
         "ranges for this color'); the ranges and rolloffs it chooses are not published, so no per-pixel value can be written down"),
        ("hsv_masks", (("HMask", "rgba.alpha"),),
         "HMask, SMask and BMask limit each correction by a mask channel",
         "the masks are channel knobs of the node in Nuke and extra input sockets in NodeBased, and how a mask combines with the range weights is not published"),
        ("hsv_hdr_values", (("huerotation", 40.0),),
         "values above 1 and below 0",
         "the sources describe HSV for 0 to 1 colour; what Nuke does to hue, saturation and value outside it is not published"),
    )
    for case_id, knobs, statement, reason in unscored:
        out.append(Case(case_id, "HSVTool", "HSVTool", tuple(knobs), ("C",), "HSVTool", None, ("C",),
                        lambda fx: (fx["C"].astype(np.float64), np.zeros(fx["C"].shape, bool)), "convention",
                        f"{statement} ({_HSV_SOURCE})", inexpressible="", unscored=reason))
    return out


# ----------------------------------------------------------------------------------------- Toe, SoftClip and Posterize
_TOE_STATEMENT = "Toe lifts the black level with a rolloff so that whites are mostly not affected; lift 'changes the black values to this value, without affecting any original white values'"


def _toe_cases():
    lift = 0.1

    def black_and_white(fx):
        src = fx["A"].astype(np.float64)
        ref = src.copy()
        valid = np.zeros(src.shape, bool)
        black = src[..., :3] == 0.0
        white = src[..., :3] >= 1.0
        ref[..., :3] = np.where(black, lift, src[..., :3])
        valid[..., :3] = black | white
        return ref, valid
    out = [Case("toe_black_to_lift_white_untouched", "Toe", "Toe", (("lift", lift),), ("A",), "Toe", {"toe_lift": lift}, ("A",), black_and_white, "guide",
                _TOE_STATEMENT + "; compared on the fixture's exact blacks (they become the lift) and on values of 1 and above (they stay as they are)")]

    def default_is_identity(fx):
        ref = fx["A"].astype(np.float64)
        return ref, np.ones(ref.shape, bool)
    out.append(Case("toe_default_lift_is_none", "Toe", "Toe", (), ("A",), "Toe", {}, ("A",), default_is_identity, "guide",
                    "Toe's lift defaults to 0, so a Toe left alone changes nothing"))
    out.append(Case("toe_rolloff_shape", "Toe", "Toe", (("lift", lift),), ("A",), "Toe", None, ("A",), lambda fx: (fx["A"].astype(np.float64), np.zeros(fx["A"].shape, bool)),
                    "guide", _TOE_STATEMENT + ": the curve between black and white", unscored="the guide says only that the lift fades toward the whites; the shape of the "
                    "rolloff, where it ends, and what a negative value does are not published (NodeBased fades it by (1 - x / 0.2) squared up to 0.2, a width its own `toe` knob sets)"))
    return out


_SOFTCLIP_STATEMENT = "SoftClip (Nuke 17.0 Reference Guide)"


def _softclip_expected(mode, lo, hi):
    def expected(fx):
        src = fx["A"].astype(np.float64)
        rgb = src[..., :3]
        ref = src.copy()
        valid = np.zeros(src.shape, bool)
        if mode == "none":
            valid[..., :3] = True
        elif mode == "logarithmic compress":
            below = rgb <= lo
            top = np.isclose(rgb, hi)
            ref[..., :3] = np.where(top, 1.0, rgb)
            valid[..., :3] = below | top
        elif mode == "preserve hue and saturation":
            peak = rgb.max(axis=-1, keepdims=True)
            over = peak > hi
            ref[..., :3] = np.where(over, rgb * (hi / np.where(over, peak, 1.0)), rgb)
            valid[..., :3] = True
        else:
            peak = rgb.max(axis=-1, keepdims=True)
            over = peak > hi
            luma = _grey(rgb, REC709)
            room = np.where(over & (peak > luma), (hi - luma) / np.where(peak > luma, peak - luma, 1.0), 1.0)
            ref[..., :3] = np.where(over, luma + (rgb - luma) * np.clip(room, 0.0, 1.0), rgb)
            # A pixel whose luminance is itself at or above softclip max cannot be brought below it while keeping its brightness; the page does not say
            # what happens then, so those pixels are left out.
            valid[..., :3] = ~(over & (luma >= hi))
        return ref, valid
    return expected


def _softclip_cases():
    out = []
    out.append(Case("softclip_none", "SoftClip", "SoftClip", (("conversion", "none"),), ("A",), "SoftClip", {"conversion": "none"}, ("A",),
                    _softclip_expected("none", 0.8, 1.0), "guide", f"{_SOFTCLIP_STATEMENT}: conversion none applies no compression"))
    out.append(Case("softclip_log_endpoints", "SoftClip", "SoftClip",
                    (("conversion", "logarithmic compress"), ("softclip_min", 0.5), ("softclip_max", 4.0)), ("A",), "SoftClip",
                    {"conversion": "logarithmic compress", "softclip_min": 0.5, "softclip_max": 4.0}, ("A",),
                    _softclip_expected("logarithmic compress", 0.5, 4.0), "guide",
                    f"{_SOFTCLIP_STATEMENT}: values up to softclip min are left alone and softclip max is compressed to 1; compared on those two (the curve between them is "
                    "only said to be logarithmic, so the pixels in between and the values above softclip max are left out)"))
    out.append(Case("softclip_log_from_zero", "SoftClip", "SoftClip",
                    (("conversion", "logarithmic compress"), ("softclip_min", 0.0), ("softclip_max", 12.0)), ("A",), "SoftClip",
                    {"conversion": "logarithmic compress", "softclip_min": 0.0, "softclip_max": 12.0}, ("A",),
                    _softclip_expected("logarithmic compress", 0.0, 12.0), "guide",
                    f"{_SOFTCLIP_STATEMENT}: with softclip min 0 (the setting the page recommends) the largest value, softclip max, becomes 1 and values at or below 0 are left alone"))
    out.append(Case("softclip_preserve_hue_and_saturation", "SoftClip", "SoftClip",
                    (("conversion", "preserve hue and saturation"), ("softclip_max", 1.0)), ("A",), "SoftClip",
                    {"conversion": "preserve hue and saturation", "softclip_max": 1.0}, ("A",),
                    _softclip_expected("preserve hue and saturation", 0.8, 1.0), "convention",
                    f"{_SOFTCLIP_STATEMENT}: if r, g or b is greater than softclip max the brightness is reduced to get it below softclip max; the colour keeps its hue and saturation, "
                    "so the whole pixel is scaled until its largest channel is softclip max"))
    out.append(Case("softclip_preserve_hue_and_brightness", "SoftClip", "SoftClip",
                    (("conversion", "preserve hue and brightness"), ("softclip_max", 1.0)), ("A",), "SoftClip",
                    {"conversion": "preserve hue and brightness", "softclip_max": 1.0}, ("A",),
                    _softclip_expected("preserve hue and brightness", 0.8, 1.0), "convention",
                    f"{_SOFTCLIP_STATEMENT}: if r, g or b is greater than softclip max the saturation is reduced to get it below softclip max while preserving the brightness; "
                    "the colour moves toward its Rec 709 luminance (0.2125 / 0.7154 / 0.0721, Nuke's grey elsewhere) until its largest channel is softclip max"))
    out.append(Case("softclip_log_curve", "SoftClip", "SoftClip",
                    (("conversion", "logarithmic compress"), ("softclip_min", 0.8), ("softclip_max", 4.0)), ("A",), "SoftClip", None, ("A",),
                    lambda fx: (fx["A"].astype(np.float64), np.zeros(fx["A"].shape, bool)), "guide",
                    f"{_SOFTCLIP_STATEMENT}: the logarithmic curve between softclip min and softclip max",
                    unscored="the page says 'a logarithmic curve between softclip min and 1.0' and warns of artefacts when softclip min is not 0 or 1, but publishes no formula; "
                    "only the end points are written down (the cases above)"))
    return out


def _posterize_cases():
    return [Case("posterize_colors", "Posterize", "Posterize", (("Colors", 4), ("channels", "all")), ("A",), "Posterize", None, ("A",),
                 lambda fx: (fx["A"].astype(np.float64), np.zeros(fx["A"].shape, bool)), "guide",
                 "Posterize: Colors is 'the bit depth to use. For example, setting this control to 8 independently limits each affected channel to 256 colors'",
                 unscored="the page reads as a bit depth (8 gives 256 colors per channel, so the default 16 would give 65536 and show nothing), while the node is used with the "
                 "knob as a level count and its default of 16 as visible banding; no formula is published and the two readings disagree "
                 "(NodeBased uses round(v * (n - 1)) / (n - 1) clamped to 0 to 1, n levels per channel)")]


def cases():
    return (_color_correct_cases() + _saturation_cases() + _hue_correct_cases() + _hsv_tool_cases() + _toe_cases() + _softclip_cases()
            + _posterize_cases())
