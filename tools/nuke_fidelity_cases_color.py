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



# ----------------------------------------------------------------------------------------------------- Log2Lin and PLogLin (step B)
_CINEON_SOURCE = ("Kodak's Cineon conversion as Nuke's Log2Lin page describes it (black = the file value that becomes 0, white = the value that becomes 1, "
                  "gamma = the exposed-light multiplier of the film density, 0.002 density per 10-bit code value, 1023 code values = 1.0)")
_PINES_SOURCE = ("Josh Pines' log conversion, the formula Nuke's PLogLin page says it uses (K. McGaugh's summary on the OpenEXR list, 2 March 2005, "
                 "lists.gnu.org/archive/html/openexr-devel/2005-03/msg00006.html): lin = linref * 10 ** ((code - logref) * density / negative gamma) for a "
                 "10-bit code value, and the reverse log = logref + log10(max(lin, 1e-10) / linref) * negative gamma / density; code = 1023 * value")


def log2lin_reference(v, black=95.0, white=685.0, gamma=0.6, ignore_black=False, to_lin=True):
    """Cineon log to linear and back on float values (10-bit code / 1023). Returns (result, valid): the reverse is undefined at or below zero."""
    v = np.asarray(v, np.float64)
    step = 0.002 / gamma
    offset = 0.0 if ignore_black else 10.0 ** ((black - white) * step)
    gain = 1.0 / (1.0 - offset)
    if to_lin:
        return gain * (10.0 ** ((v * 1023.0 - white) * step) - offset), np.ones(v.shape, bool)
    arg = v / gain + offset
    with np.errstate(all="ignore"):
        out = (white + np.log10(np.where(arg > 1e-10, arg, 1.0)) / step) / 1023.0
    return out, arg > 1e-10


def pines_reference(v, linref=0.18, logref=445.0, ngamma=0.6, density=0.002, to_lin=True):
    v = np.asarray(v, np.float64)
    if to_lin:
        return linref * 10.0 ** ((v * 1023.0 - logref) * density / ngamma)
    return (logref + np.log10(np.maximum(v, 1e-10) / linref) * ngamma / density) / 1023.0


def _log_expected(fixture, fn, alpha_too=False):
    def expected(fx):
        src = fx[fixture].astype(np.float64)
        ref = src.copy()
        value, valid = fn(src[..., :4] if alpha_too else src[..., :3])
        count = 4 if alpha_too else 3
        ref[..., :count] = value
        mask = np.ones(src.shape, bool)
        mask[..., :count] = valid
        return ref, mask
    return expected


def _log2lin_case(case_id, nb_params, nuke_knobs, statement, source="convention", fixture="L", **kwargs):
    to_lin = nb_params.get("log_direction", "log to lin") == "log to lin"
    reference = dict(kwargs, to_lin=to_lin)
    note = statement + ("" if source == "guide" else f" ({_CINEON_SOURCE})")
    return Case(case_id, "Log2Lin", "Log2Lin", tuple(nuke_knobs), (fixture,), "Log2Lin", dict(nb_params), (fixture,),
                _log_expected(fixture, lambda v: log2lin_reference(v, **reference)), source, note)


def _log2lin_cases():
    out = []
    out.append(_log2lin_case("log2lin_default_to_linear", {}, (), "the defaults (black 95, white 685, gamma 0.6) convert Cineon log to linear: "
                             "lin = (10 ** ((code - white) * 0.002 / gamma) - offset) / (1 - offset) with offset = 10 ** ((black - white) * 0.002 / gamma)"))
    def anchors(fx):
        src = fx["L"].astype(np.float64)
        ref = src.copy()
        codes = np.rint(src[..., :3] * 1023.0)
        at_black, at_white = codes == 95, codes == 685
        ref[..., :3] = np.where(at_black, 0.0, np.where(at_white, 1.0, src[..., :3]))
        valid = np.ones(src.shape, bool)
        valid[..., :3] = at_black | at_white
        return ref, valid
    out.append(Case("log2lin_black_is_zero_white_is_one", "Log2Lin", "Log2Lin", (("black", 95.0), ("white", 685.0)), ("L",), "Log2Lin",
                   {"black": 95.0, "white": 685.0}, ("L",), anchors, "guide",
                   "Log2Lin: black ('the value in the Cineon file that is converted into 0') 95 becomes 0 and white ('converted into 1') 685 becomes 1; "
                   "compared on the pixels that hold those two code values"))
    out.append(_log2lin_case("log2lin_to_log", {"log_direction": "lin to log"}, (("operation", "lin2log"),),
                             "lin2log is the reverse: code = white + log10(lin * (1 - offset) + offset) * gamma / 0.002, divided by 1023 (undefined at or below the black offset, "
                             "so those pixels are not compared)", fixture="A"))
    out.append(_log2lin_case("log2lin_ignore_black", {"ignore_black": 1}, (("ignore_black", True),),
                             "ignore black: 'the black offset is not subtracted so that the conversion is a pure log function': lin = 10 ** ((code - white) * 0.002 / gamma)",
                             ignore_black=True))
    out.append(_log2lin_case("log2lin_ignore_black_to_log", {"ignore_black": 1, "log_direction": "lin to log"}, (("ignore_black", True), ("operation", "lin2log")),
                             "ignore black on lin2log: code = white + log10(lin) * gamma / 0.002 (a pure log function)", fixture="A", ignore_black=True))
    out.append(_log2lin_case("log2lin_custom_points", {"black": 64.0, "white": 940.0, "gamma": 0.5}, (("black", 64.0), ("white", 940.0), ("gamma", 0.5)),
                             "black, white and gamma move the curve (here the video range 64 and 940 and gamma 0.5)", black=64.0, white=940.0, gamma=0.5))
    out.append(_log2lin_case("log2lin_custom_points_to_log", {"black": 64.0, "white": 940.0, "gamma": 0.5, "log_direction": "lin to log"},
                             (("black", 64.0), ("white", 940.0), ("gamma", 0.5), ("operation", "lin2log")),
                             "lin2log with the video-range points and gamma 0.5", fixture="A", black=64.0, white=940.0, gamma=0.5))
    out.append(Case("log2lin_alpha_untouched", "Log2Lin", "Log2Lin", (), ("L",), "Log2Lin", {}, ("L",),
                    lambda fx: (fx["L"].astype(np.float64), np.concatenate([np.zeros(fx["L"].shape[:-1] + (3,), bool), np.ones(fx["L"].shape[:-1] + (1,), bool)], axis=-1)),
                    "guide", "Log2Lin: the default channels are rgb, so alpha is left as it is"))
    out.append(Case("log2lin_soft_clip", "Log2Lin", "Log2Lin", (), ("L",), "Log2Lin", None, ("L",),
                    lambda fx: (fx["L"].astype(np.float64), np.zeros(fx["L"].shape, bool)), "convention",
                    "Log2Lin: the soft clip region of Kodak's Cineon conversion",
                    unscored="Kodak's Cineon specification describes an optional soft clip below the black point, but the Log2Lin page has no soft clip knob and states no formula "
                    "for one, and NodeBased's node has none either, so the region below black is left out rather than guessed"))
    return out


def _pines_case(case_id, nb_params, nuke_knobs, statement, source="convention", fixture="L", alpha_too=True, **kwargs):
    to_lin = nb_params.get("log_direction", "log to lin") == "log to lin"
    note = statement + ("" if source == "guide" else f" ({_PINES_SOURCE})")
    return Case(case_id, "PLogLin", "PLogLin", tuple(nuke_knobs), (fixture,), "PLogLin", dict(nb_params), (fixture,),
                _log_expected(fixture, lambda v: (pines_reference(v, to_lin=to_lin, **kwargs), np.ones(v.shape, bool)), alpha_too), source, note)


def _ploglin_cases():
    out = []
    out.append(_pines_case("ploglin_defaults", {}, (),
                           "the defaults (linear reference 0.18, log reference 445, negative gamma 0.6, density 0.002) convert log to linear on every channel: "
                           "the page's default channels are all, so alpha is converted as well"))

    def anchor(fx):
        src = fx["L"].astype(np.float64)
        codes = np.rint(src * 1023.0)
        ref = src.copy()
        ref[..., :3] = np.where(codes[..., :3] == 445, 0.18, src[..., :3])
        valid = np.zeros(src.shape, bool)
        valid[..., :3] = codes[..., :3] == 445
        return ref, valid
    out.append(Case("ploglin_reference_point", "PLogLin", "PLogLin", (("linref", 0.18), ("logref", 445.0)), ("L",), "PLogLin",
                    {"linear_reference": 0.18, "log_reference": 445.0}, ("L",), anchor, "guide",
                    "PLogLin: the log reference value (445) is 'the logarithmic value that corresponds with the linear reference value' (0.18), so code 445 becomes 0.18; "
                    "compared on the pixels that hold that code value"))
    out.append(_pines_case("ploglin_to_log", {"log_direction": "lin to log"}, (("operation", "lin to log"),),
                           "lin to log is the reverse: log = logref + log10(max(lin, 1e-10) / linref) * negative gamma / density, divided by 1023", fixture="A"))
    out.append(_pines_case("ploglin_custom_reference", {"linear_reference": 0.5, "log_reference": 500.0}, (("linref", 0.5), ("logref", 500.0)),
                           "linear reference and log reference move the anchor point (0.5 at code 500)", linref=0.5, logref=500.0))
    out.append(_pines_case("ploglin_negative_gamma_and_density", {"negative_gamma": 0.8, "density_per_code_value": 0.003},
                           (("ngamma", 0.8), ("density", 0.003)),
                           "negative gamma divides and density per code value multiplies the code distance from the reference (the slope of the straight log line)",
                           ngamma=0.8, density=0.003))
    out.append(_pines_case("ploglin_custom_to_log", {"linear_reference": 0.5, "log_reference": 500.0, "negative_gamma": 0.8, "log_direction": "lin to log"},
                           (("linref", 0.5), ("logref", 500.0), ("ngamma", 0.8), ("operation", "lin to log")),
                           "lin to log with a moved reference and negative gamma 0.8", fixture="A", linref=0.5, logref=500.0, ngamma=0.8))
    return out



# ------------------------------------------------------------------------------------------------------------ Colorspace (step B)
_CS_GUIDE = "Colorspace (Nuke 17.0 Reference Guide)"
_SRGB_SOURCE = "IEC 61966-2-1, the sRGB transfer function (linear below 0.04045 / 12.92, otherwise ((v + 0.055) / 1.055) ** 2.4)"
_PRIMARIES_SOURCE = ("the published matrices between linear RGB spaces with a D65 white: ITU-R BT.2087-0 for Rec.709 to Rec.2020 and the P3-D65 matrix of "
                     "SMPTE RP 431-2 / EG 432-1")
_WHITES = {"D65": (0.3127, 0.3290), "D60": (0.32168, 0.33767), "DCI": (0.314, 0.351)}
_PRIMARIES = {"Rec.709": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), "Rec.2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
              "P3-D65": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060))}
# Rec.709 to Rec.2020 and to P3-D65 on linear light, as published (six digits).
_M709_TO_2020 = np.array([[0.627404, 0.329283, 0.043313], [0.069097, 0.919540, 0.011362], [0.016391, 0.088013, 0.895595]])
_M709_TO_P3D65 = np.array([[0.822462, 0.177538, 0.0], [0.033194, 0.966806, 0.0], [0.017083, 0.072397, 0.910520]])
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _srgb_decode(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode(v):
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.maximum(v, 0.0) ** (1.0 / 2.4) - 0.055)


def _xyz_of_xy(xy):
    return np.array([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]])


def rgb_to_xyz_matrix(primaries, white):
    """Lindbloom's derivation: the primaries' XYZ columns scaled so that RGB (1, 1, 1) is the white point."""
    columns = np.stack([_xyz_of_xy(xy) for xy in _PRIMARIES[primaries]], axis=1)
    return columns * np.linalg.solve(columns, _xyz_of_xy(_WHITES[white]))[None, :]


def bradford_adaptation(white_in, white_out):
    source, target = _BRADFORD @ _xyz_of_xy(_WHITES[white_in]), _BRADFORD @ _xyz_of_xy(_WHITES[white_out])
    return np.linalg.inv(_BRADFORD) @ np.diag(target / source) @ _BRADFORD


def _cs_expected(fn, fixture="C"):
    """`fn` takes the (h, w, 3) stored colour of the 0..1 fixture and returns the converted colour; alpha is left alone."""
    def expected(fx):
        src = fx[fixture].astype(np.float64)
        ref = src.copy()
        ref[..., :3] = fn(src[..., :3])
        return ref, np.ones(src.shape, bool)
    return expected


def _cs_case(case_id, nb_params, nuke_knobs, statement, fn, source="convention", note_source="", fixture="C"):
    note = f"{statement} ({note_source})" if note_source else statement
    return Case(case_id, "Colorspace", "Colorspace", tuple(nuke_knobs), (fixture,), "Colorspace", dict(nb_params), (fixture,), _cs_expected(fn, fixture), source, note)


def _colorspace_cases():
    out = []
    out.append(_cs_case("colorspace_srgb_to_linear", {"transfer_in": "sRGB"}, (("colorspace_in", "sRGB"),),
                        "input sRGB, output linear (primaries and illuminants equal): the sRGB curve is undone", _srgb_decode, note_source=_SRGB_SOURCE))
    out.append(_cs_case("colorspace_linear_to_srgb", {"transfer_out": "sRGB"}, (("colorspace_out", "sRGB"),),
                        "input linear, output sRGB: the sRGB curve is applied", _srgb_encode, note_source=_SRGB_SOURCE))
    out.append(_cs_case("colorspace_gamma22", {"transfer_in": "Gamma 2.2"}, (("colorspace_in", "Gamma2.2"),),
                        "input Gamma 2.2: a pure power, linear = v ** 2.2", lambda v: v ** 2.2,
                        note_source="a gamma encoding is the plain power by definition; the Foundry Reference Guide gives no formula for Nuke's Gamma2.2 choice"))
    out.append(_cs_case("colorspace_gamma24_out", {"transfer_out": "Gamma 2.4"}, (("colorspace_out", "Gamma2.4"),),
                        "output Gamma 2.4: v ** (1 / 2.4)", lambda v: v ** (1 / 2.4),
                        note_source="a gamma encoding is the plain power by definition; the Foundry Reference Guide gives no formula for Nuke's Gamma2.4 choice"))
    out.append(_cs_case("colorspace_srgb_to_gamma22", {"transfer_in": "sRGB", "transfer_out": "Gamma 2.2"},
                        (("colorspace_in", "sRGB"), ("colorspace_out", "Gamma2.2")),
                        "sRGB to Gamma 2.2 goes through linear light: (sRGB curve undone) ** (1 / 2.2)", lambda v: _srgb_decode(v) ** (1 / 2.2), note_source=_SRGB_SOURCE))
    out.append(_cs_case("colorspace_primaries_709_to_2020", {"primaries_out": "Rec.2020"}, (("primary_out", "Rec.2020"),),
                        "'It can adjust for different primaries': linear Rec.709 primaries to linear Rec.2020 primaries (white D65 both ways)",
                        lambda v: v @ _M709_TO_2020.T, note_source=_PRIMARIES_SOURCE))
    out.append(_cs_case("colorspace_primaries_2020_to_709", {"primaries_in": "Rec.2020"}, (("primary_in", "Rec.2020"),),
                        "linear Rec.2020 primaries to linear Rec.709 primaries", lambda v: v @ np.linalg.inv(_M709_TO_2020).T, note_source=_PRIMARIES_SOURCE))
    out.append(_cs_case("colorspace_primaries_709_to_p3d65", {"primaries_out": "P3-D65"}, (("primary_out", "P3-D65"),),
                        "linear Rec.709 primaries to linear P3-D65 primaries", lambda v: v @ _M709_TO_P3D65.T, note_source=_PRIMARIES_SOURCE))
    out.append(_cs_case("colorspace_primaries_with_curves", {"transfer_in": "sRGB", "primaries_out": "Rec.2020", "transfer_out": "Gamma 2.4"},
                        (("colorspace_in", "sRGB"), ("primary_out", "Rec.2020"), ("colorspace_out", "Gamma2.4")),
                        "decode the input curve, convert the primaries on linear light, encode the output curve",
                        lambda v: np.maximum(_srgb_decode(v) @ _M709_TO_2020.T, 0.0) ** (1 / 2.4), note_source=_SRGB_SOURCE + "; " + _PRIMARIES_SOURCE))

    def same_xyz(fn_white_in, fn_white_out):
        m_in, m_out = rgb_to_xyz_matrix("Rec.709", fn_white_in), rgb_to_xyz_matrix("Rec.709", fn_white_out)
        return lambda v: v @ (np.linalg.inv(m_out) @ m_in).T
    out.append(_cs_case("colorspace_illuminant_keeps_xyz", {"whitepoint_out": "D60"}, (("illuninant_out", "D60"),),
                        "with Bradford matrix off, 'the input and output colors have the same CIE colorspace XYZ values': the RGB values change so the XYZ stays "
                        "(Rec.709 primaries on a D65 white to the same primaries on a D60 white)", same_xyz("D65", "D60"), source="guide"))
    out.append(_cs_case("colorspace_illuminant_keeps_xyz_dci", {"whitepoint_in": "DCI", "whitepoint_out": "D65"},
                        (("illuninant_in", "DCI"), ("illuninant_out", "D65")), "the same rule with a DCI white as the input illuminant", same_xyz("DCI", "D65"), source="guide"))

    def bradford(white_in, white_out):
        m_in, m_out = rgb_to_xyz_matrix("Rec.709", white_in), rgb_to_xyz_matrix("Rec.709", white_out)
        return lambda v: v @ (np.linalg.inv(m_out) @ bradford_adaptation(white_in, white_out) @ m_in).T
    out.append(_cs_case("colorspace_bradford_adapts", {"whitepoint_out": "D60", "bradford_matrix": 1}, (("illuninant_out", "D60"), ("bradford_matrix", True)),
                        "Bradford matrix on, with different illuminants: 'an illuminant adaptation transform. This alters colors so they appear the same' "
                        "(the Bradford cone-response matrix, Lindbloom form)", bradford("D65", "D60"), note_source="the Bradford adaptation transform as published by B. Lindbloom and used in ICC profiles"))
    out.append(_cs_case("colorspace_bradford_equal_illuminants", {"bradford_matrix": 1}, (("bradford_matrix", True),),
                        "Bradford matrix on with equal illuminants changes nothing: 'if the illuminants are not the same, you can enable this'", lambda v: v, source="guide"))

    def premultiplied_colour(fx):
        src = fx["C"].astype(np.float64)
        ref = src.copy()
        ref[..., :3] = _srgb_decode(src[..., :3])
        return ref, np.ones(src.shape, bool)
    out.append(Case("colorspace_does_not_divide_by_alpha", "Colorspace", "Colorspace", (("colorspace_in", "sRGB"),), ("C",), "Colorspace",
                    {"transfer_in": "sRGB"}, ("C",), premultiplied_colour, "guide",
                    f"{_CS_GUIDE}: only '(un)premult by' divides the image by a channel 'before it is processed', so the stored colour is converted as it is, whatever the alpha"))
    # The in/out choices the page lists and NodeBased has no equivalent for.
    missing = (
        ("colorspace_hsv", "HSV, HSL and the other hue-based spaces", (("colorspace_out", "HSV"),),
         "the page lists HSV among the formats it converts to; NodeBased's Colorspace has no HSV, HSL or HSB choice (the hue tools Saturation, HSVTool and HueCorrect work in HSV internally)"),
        ("colorspace_ycbcr", "YCbCr / YPbPr", (("colorspace_out", "YCbCr"),),
         "the page lists YUV-type formats; NodeBased has no YCbCr, YPbPr or YUV choice"),
        ("colorspace_cie", "CIE XYZ, Yxy, Lab and LCh", (("colorspace_out", "CIE-XYZ"),),
         "the page lists CIE formats; NodeBased converts through XYZ internally but cannot output XYZ, Yxy, Lab or LCh"),
        ("colorspace_cineon", "Cineon and the other log encodings", (("colorspace_in", "Cineon"),),
         "Nuke's colorspace list holds Cineon, Panalog, REDLog, ViperLog, AlexaV3LogC and the vendor logs; NodeBased's Colorspace has linear, sRGB, Gamma 2.2 and Gamma 2.4 only (Log2Lin and PLogLin do Cineon separately)"),
        ("colorspace_rec709_curve", "the Rec.709 camera curve", (("colorspace_in", "rec709"),),
         "Nuke's rec709 input is ITU-R BT.709's camera transfer function, a different curve from sRGB's; NodeBased has no such choice"),
        ("colorspace_other_illuminants", "the other illuminants (A, B, C, D50 and so on)", (("illuninant_out", "A"),),
         "the page names illuminant A (a household tungsten bulb) and D65 as examples; NodeBased offers D65, D60 and DCI"),
    )
    for case_id, what, nuke_knobs, reason in missing:
        out.append(Case(case_id, "Colorspace", "Colorspace", nuke_knobs, ("C",), "Colorspace", None, ("C",),
                        lambda fx: (fx["C"].astype(np.float64), np.zeros(fx["C"].shape, bool)), "guide",
                        f"{_CS_GUIDE}: conversion to or from {what}", inexpressible=reason))
    return out



# ----------------------------------------------------------------------------------------------------------- ColorLookup (step B)
_CL_GUIDE = ("ColorLookup (Nuke 17.0 Reference Guide): 'The horizontal axis represents the channel's original, or input, values, and the vertical axis "
             "represents the channel's new, or output values'")
_CL_CHANNELS = ("red", "green", "blue", "alpha")


def _cl_curve_json(points):
    return '{"interpolation":"linear","points":' + str([list(map(float, p)) for p in points]).replace(" ", "") + "}"


def _cl_nuke_lut(entries):
    """The `lut` knob text for linear curves through points on the 0..1 axes: master, red, green, blue and alpha, as `{curve L x0 y0 x1 y1 ...}`."""
    return "{" + " ".join("{" + name + " {curve L " + " ".join(f"x{x:g} {y:g}" for x, y in pts) + "}}" for name, pts in entries.items()) + "}"


def color_lookup_reference(rgba, master=None, channels=None):
    """Each channel runs through the master curve and then its own (linear curves through the given points, held 0..1 by the fixture)."""
    out = rgba.astype(np.float64).copy()
    for index, name in enumerate(_CL_CHANNELS):
        value = out[..., index]
        if master:
            value = np.interp(value, *zip(*master))
        if channels and name in channels:
            value = np.interp(value, *zip(*channels[name]))
        out[..., index] = value
    return out


def _cl_case(case_id, curves, statement, source="guide", extra_nb=None, extra_nuke=(), expected_curves=None):
    """`curves`: {"master" | "red" | ...: points}. `expected_curves` lets a case compare against a different set of curves than it sets (the range cases)."""
    master = curves.get("master")
    channels = {k: v for k, v in curves.items() if k != "master"}
    reference = expected_curves if expected_curves is not None else (master, channels)
    params = {f"curve_{name}": _cl_curve_json(points) for name, points in curves.items()}
    params.update(extra_nb or {})

    def expected(fx):
        src = fx["C"].astype(np.float64)
        return color_lookup_reference(src, *reference), np.ones(src.shape, bool)
    knobs = (("lut", _cl_nuke_lut(curves)),) + tuple(extra_nuke) if curves else tuple(extra_nuke)
    return Case(case_id, "ColorLookup", "ColorLookup", knobs, ("C",), "ColorLookup", params, ("C",), expected, source,
                f"{statement} ({_CL_GUIDE})" if source == "guide" else statement)


def _color_lookup_cases():
    out = []
    bend = ((0.0, 0.0), (0.5, 0.8), (1.0, 1.0))
    out.append(_cl_case("colorlookup_default_is_identity", {}, "the default curves are straight lines from 0 to 1 on every channel, so nothing changes"))
    out.append(_cl_case("colorlookup_red_curve", {"red": bend}, "the red curve maps the red values (here 0.5 to 0.8) and leaves green, blue and alpha alone"))
    out.append(_cl_case("colorlookup_green_blue_curves", {"green": ((0.0, 0.1), (1.0, 0.7)), "blue": ((0.0, 0.0), (0.3, 0.6), (1.0, 0.9))},
                        "the green and blue curves each map their own channel"))
    out.append(_cl_case("colorlookup_alpha_curve", {"alpha": ((0.0, 0.0), (0.5, 0.2), (1.0, 1.0))},
                        "the alpha curve maps alpha (the default channels are rgba)"))
    out.append(_cl_case("colorlookup_master_curve", {"master": ((0.0, 0.0), (0.4, 0.7), (1.0, 0.9))},
                        "the master curve 'represents all channels': every channel is mapped through it"))
    out.append(_cl_case("colorlookup_range_does_not_scale_the_axis", {"red": bend, "green": bend},
                        "range only sets how far the precomputed table reaches ('to increase processing speed'): the curve maps the same values whatever the range",
                        extra_nb={"use_precomputed": 1, "range": 2.0}, extra_nuke=(("use_precomputed", True), ("range", 2.0))))
    out.append(_cl_case("colorlookup_without_precomputed_table", {"red": bend, "blue": ((0.0, 0.2), (1.0, 0.6))},
                        "with use precomputed table off the curve is evaluated directly and maps the same values",
                        extra_nb={"use_precomputed": 0}, extra_nuke=(("use_precomputed", False),)))
    unscored = (
        ("colorlookup_outside_the_curve", "values beyond the first and last key of a curve",
         "the page does not say whether a curve holds its end values or continues its slope past the keys (NodeBased extends the end segments linearly)"),
        ("colorlookup_smooth_keys", "interpolation between keys",
         "the page names the curve editor but not the spline Nuke fits through keys added there (smooth is its default); the cases above use linear segments, whose "
         "shape is the same in any editor, and NodeBased's smooth and Hermite segments cannot be compared without that spline"),
        ("colorlookup_master_with_channel", "master and channel curves together",
         "the page says the master curve represents all channels but not how it combines with a channel's own curve (NodeBased runs the master first, then the channel)"),
        ("colorlookup_precomputed_table_steps", "the precomputed table's resolution",
         "the page says a table 'between 0 and the value specified in the range field' is used but not how many entries it holds, so its rounding cannot be reproduced "
         "(NodeBased evaluates the curves exactly, as Nuke does with the table off)"),
    )
    for case_id, what, reason in unscored:
        out.append(Case(case_id, "ColorLookup", "ColorLookup", (), ("C",), "ColorLookup", None, ("C",),
                        lambda fx: (fx["C"].astype(np.float64), np.zeros(fx["C"].shape, bool)), "guide", f"ColorLookup: {what}", unscored=reason))
    return out



# ----------------------------------------------------------------------------------------------------------- Expression (step B)
_EXPR_GUIDE = "Expression (Nuke 17.0 Reference Guide)"
_F32 = np.float32


def _expr_case(case_id, rows, reference, statement, source="guide", channels=None, temps=None):
    """`rows`: {"r": expression, ...} for the rows the case sets (a row not named keeps the identity `r`, `g`, `b`, `a`); `reference(fx)` returns the
    expected (h, w, 4) float64 array. Non-finite reference values are left out of the comparison."""
    params = {f"expr_{c}": rows.get(c, c) for c in "rgba"}
    knobs = [(f"expr{i}", rows.get(c, c)) for i, c in enumerate("rgba")]
    for letter, text in (channels or {}).items():
        params[f"expr_channels_{letter}"] = text
        knobs.append((f"channel{'rgba'.index(letter)}", text))
    for slot, (name, expr) in enumerate(temps or ()):
        params[f"temp_name{slot}"], params[f"temp_expr{slot}"] = name, expr
        knobs += [(f"temp_name{slot}", name), (f"temp_expr{slot}", expr)]

    def expected(fx):
        ref = np.asarray(reference(fx), np.float64)
        return ref, np.isfinite(ref)
    note = f"{statement} ({_EXPR_GUIDE})" if source == "guide" else f"{statement} ({_EXPR_GUIDE} says 'C-like syntax expressions'; the operators themselves are C's)"
    return Case(case_id, "Expression", "Expression", tuple(knobs), ("A",), "Expression", params, ("A",), expected, source, note)


def _channels_of(fx):
    a = fx["A"].astype(np.float64)
    return a[..., 0], a[..., 1], a[..., 2], a[..., 3]


def _stack(r, g, b, a):
    return np.stack([np.broadcast_to(c, r.shape) for c in (r, g, b, a)], axis=-1)


def _expression_cases():
    out = []
    out.append(_expr_case("expression_blank_is_zero", {"r": "", "b": ""}, lambda fx: _stack(0.0 * _channels_of(fx)[0], _channels_of(fx)[1], 0.0 * _channels_of(fx)[2], _channels_of(fx)[3]),
                          "a blank expression is the same as zero: the red and blue rows are empty, so those channels become 0 and green and alpha pass through"))
    out.append(_expr_case("expression_arithmetic_precedence", {"r": "r + g * b", "g": "(r + g) * b", "b": "-r - g / 2 + b"},
                          lambda fx: _stack(*(lambda r, g, b, a: (r + g * b, (r + g) * b, -r - g / 2 + b, a))(*_channels_of(fx))),
                          "C-like syntax: * and / bind tighter than + and -, parentheses group, a leading minus negates", source="convention"))
    out.append(_expr_case("expression_conditional", {"r": "r > 0.5 ? r : 0", "g": "r > 0.2 && g < 1 ? 1 : 0.5", "b": "r < 0 || b >= 4 ? -1 : !a"},
                          lambda fx: _stack(*(lambda r, g, b, a: (np.where(r > 0.5, r, 0.0), np.where((r > 0.2) & (g < 1), 1.0, 0.5),
                                                                   np.where((r < 0) | (b >= 4), -1.0, (a == 0) * 1.0), a))(*_channels_of(fx))),
                          "C-like syntax: c ? a : b, the comparisons, && and || and !, which give 1 for true and 0 for false", source="convention"))
    out.append(_expr_case("expression_equality", {"r": "r == 0.5", "g": "g != 0.5", "b": "true() + false() + b == 1"},
                          lambda fx: _stack(*(lambda r, g, b, a: ((r == 0.5) * 1.0, (g != 0.5) * 1.0, (1.0 + 0.0 + b == 1) * 1.0, a))(*_channels_of(fx))),
                          "== and != compare, true() always returns 1 and false() always returns 0 (the function table)"))
    out.append(_expr_case("expression_layer_dot_channel", {"r": "rgba.green", "g": "rgba.blue", "b": "rgba.red"},
                          lambda fx: _stack(*(lambda r, g, b, a: (g, b, r, a))(*_channels_of(fx))),
                          "'use r, g, b, and a to reference the red, green, blue, and alpha channels in the rgba layer': layer.channel names the same channels"))
    out.append(_expr_case("expression_functions_rounding", {"r": "floor(r) + ceil(g)", "g": "int(r) - trunc(b)", "b": "abs(r) + fabs(g) + clamp(b, -0.25, 0.75)"},
                          lambda fx: _stack(*(lambda r, g, b, a: (np.floor(r) + np.ceil(g), np.trunc(r) - np.trunc(b), np.abs(r) + np.abs(g) + np.clip(b, -0.25, 0.75), a))(*_channels_of(fx))),
                          "floor, ceil, int and trunc (toward zero), abs and fabs, and clamp(x, min, max)"))
    out.append(_expr_case("expression_functions_clamp_min_max_lerp", {"r": "clamp(r)", "g": "max(r, g, b)", "b": "min(r, g, b, 0.3) + lerp(r, g, 0.25)"},
                          lambda fx: _stack(*(lambda r, g, b, a: (np.clip(r, 0.0, 1.0), np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(np.minimum(r, g), b), 0.3) + (r + (g - r) * 0.25), a))(*_channels_of(fx))),
                          "clamp(x) limits to 0..1, min and max take any number of values, lerp(a, b, x) is a at 0 and b at 1"))
    out.append(_expr_case("expression_functions_power", {"r": "pow(abs(r), 2.2) + pow2(g)", "g": "sqrt(abs(b)) + exp(r) - log10(abs(b) + 1)", "b": "log(abs(g) + 1) + hypot(r, g)"},
                          lambda fx: _stack(*(lambda r, g, b, a: (np.abs(r) ** 2.2 + g * g, np.sqrt(np.abs(b)) + np.exp(r) - np.log10(np.abs(b) + 1), np.log(np.abs(g) + 1) + np.hypot(r, g), a))(*_channels_of(fx))),
                          "pow, pow2, sqrt, exp, log, log10 and hypot"))
    out.append(_expr_case("expression_functions_trig", {"r": "sin(r) + cos(g)", "g": "tan(b / 10) + tanh(r)", "b": "degrees(r) - radians(g)"},
                          lambda fx: _stack(*(lambda r, g, b, a: (np.sin(r) + np.cos(g), np.tan(b / 10) + np.tanh(r), np.degrees(r) - np.radians(g), a))(*_channels_of(fx))),
                          "sin, cos, tan and tanh (radians), degrees and radians"))
    out.append(_expr_case("expression_functions_step", {"r": "smoothstep(0.2, 0.8, r)", "g": "step(0.5, g)", "b": "fmod(b, 0.7)"},
                          lambda fx: _stack(*(lambda r, g, b, a: (np.where(r < 0.2, 0.0, np.where(r >= 0.8, 1.0, (lambda t: t * t * (3 - 2 * t))((r - 0.2) / 0.6))),
                                                                   np.where(g < 0.5, 0.0, 1.0), np.fmod(b, 0.7), a))(*_channels_of(fx))),
                          "smoothstep(a, b, x) is 0 below a, 1 from b and a smooth cubic between, step(a, x) is 0 below a and 1 otherwise, fmod(x, y) is the remainder rounded toward zero"))
    out.append(_expr_case("expression_functions_rint", {"r": "rint(r * 3 + 0.1)"}, lambda fx: _stack(np.rint(_channels_of(fx)[0] * 3 + 0.1), *_channels_of(fx)[1:]),
                          "rint(x) rounds to the nearest integer (compared away from the exact halves, where the page does not say which way they go)"))
    out.append(_expr_case("expression_temp_variables", {"r": "lum", "g": "lum", "b": "lum * k"},
                          lambda fx: _stack(*(lambda r, g, b, a: (lambda lum: (lum, lum, lum * 2.0, a))(0.2125 * r + 0.7154 * g + 0.0721 * b))(*_channels_of(fx))),
                          "a temporary variable stands for a long expression in the '=' fields: lum is the Rec 709 grey and k is 2",
                          temps=(("lum", "0.2125 * r + 0.7154 * g + 0.0721 * b"), ("k", "2"))))
    out.append(_expr_case("expression_channel_row", {"r": "a", "g": "g", "b": "b", "a": "a"},
                          lambda fx: _stack(_channels_of(fx)[3], _channels_of(fx)[1], _channels_of(fx)[2], _channels_of(fx)[3]),
                          "the 'channels' of a row pick where its expression is written: the first row sends alpha to red and the later rows write green, blue and alpha as they are",
                          channels={"r": "red"}))
    out.append(_expr_case("expression_channel_set_rgb", {"r": "r * 0.5 + 0.25", "g": "g", "b": "b"},
                          lambda fx: _stack(*(lambda r, g, b, a: (r * 0.5 + 0.25, g, b, a))(*_channels_of(fx))),
                          "a row whose channels are 'rgb' writes its expression to red, green and blue; the later rows that name their own channel then overwrite green and blue with "
                          "themselves, so only red changes",
                          channels={"r": "rgb"}))
    out.append(_expr_case("expression_32_bit_float", {"r": "(r + 100000000) - 100000000"},
                          lambda fx: _stack(((_channels_of(fx)[0].astype(_F32) + _F32(1e8)) - _F32(1e8)).astype(np.float64), *_channels_of(fx)[1:]),
                          "'All math is done with 32-bit floating point numbers': adding and removing 1e8 leaves multiples of 8, the float32 step at that size"))
    unscored = (
        ("expression_pixel_coordinates", "x and y",
         "the page says x and y are the pixel coordinate but not whether a pixel's x is its left edge or its centre (a half pixel in), nor which row counts as zero "
         "(Nuke counts up from the bottom row; NodeBased's new node counts up from the bottom and uses pixel centres, its older node counted down from the top)"),
        ("expression_centred_coordinates", "cx and cy",
         "the page defines cx and cy by their geometry (0,0 at the centre, -1 and 1 at the left and right edges, 1,1 on the diagonal of the right edge) but, as with x and y, "
         "not whether the value belongs to a pixel's centre or its edge, so the half-pixel shift in every value is left unchecked"),
        ("expression_noise_functions", "random, noise, fBm and turbulence",
         "the page describes Perlin noise and pseudo-random values only in words; Nuke's own table and seed are not published, so no value can be reproduced (NodeBased refuses these)"),
        ("expression_colour_conversion_functions", "from_sRGB, to_sRGB, from_rec709, to_rec709, from_byte and to_byte",
         "the page words 'color component' as a value the function takes and returns without saying whether that is 0 to 1 or 0 to 255 (to_byte returns an 8-bit value, "
         "from_byte's text repeats from_sRGB's), so the scale cannot be pinned down (NodeBased refuses these)"),
        ("expression_atan2_and_mantissa", "atan2, exponent, logb and mantissa",
         "the page names atan2's arguments (x, y) but not which one is the numerator, and gives mantissa and exponent as the frexp pair without saying whether logb "
         "matches it or the C logb"),
        ("expression_knob_references", "references to other nodes' knobs (Blur1.size, Blur1.size(t))",
         "the page says any control on any node can be named in an expression; NodeBased's Expression node has no node-knob references (its parameter expressions are a separate system)"),
        ("expression_percent_operator", "the % operator",
         "the page lists no operators; NodeBased reads % as C's fmod, which the table's fmod describes"),
    )
    for case_id, what, reason in unscored:
        out.append(Case(case_id, "Expression", "Expression", (), ("A",), "Expression", None, ("A",),
                        lambda fx: (fx["A"].astype(np.float64), np.zeros(fx["A"].shape, bool)), "guide", f"{_EXPR_GUIDE}: {what}", unscored=reason))
    return out



# ------------------------------------------------------------------------------------------------------------ CrossTalk (step B)
_XT_GUIDE = ("Crosstalk (Nuke 17.0 Reference Guide): 'Each output color channel is the sum of three lookup tables, each indexed by one of the input color channels. "
             "The horizontal axis represents the channel's original, or input, values, and the vertical axis represents the channel's new, or output values'")
_XT_NO_SCRIPT = (" The guide gives no text form for the crossover curves, so the reference pack's script leaves them at their defaults for this case; "
                 "compare it with Nuke by hand.")
_XT_IDENTITY = {(o, s): ((0.0, 0.0), (1.0, 1.0)) if o == s else ((0.0, 0.0), (1.0, 0.0)) for o in "rgb" for s in "rgb"}


def crosstalk_reference(rgb, curves):
    out = np.zeros_like(rgb, dtype=np.float64)
    for oi, o in enumerate("rgb"):
        for si, s in enumerate("rgb"):
            out[..., oi] += np.interp(rgb[..., si], *zip(*curves[(o, s)]))
    return out


def _xt_case(case_id, changed, statement, source="guide"):
    curves = {**_XT_IDENTITY, **changed}
    params = {f"xt_curve_{o}_{s}": _cl_curve_json(pts) for (o, s), pts in curves.items()}

    def expected(fx):
        src = fx["C"].astype(np.float64)
        ref = src.copy()
        ref[..., :3] = crosstalk_reference(src[..., :3], curves)
        valid = np.ones(src.shape, bool)
        valid[..., 3] = False                       # the page says nothing about alpha
        return ref, valid
    note = f"{statement} ({_XT_GUIDE}){_XT_NO_SCRIPT}"
    return Case(case_id, "CrossTalk", "CrossTalk", (), ("C",), "CrossTalk", params, ("C",), expected, source, note)


def _crosstalk_cases():
    out = []
    out.append(_xt_case("crosstalk_default_is_identity", {}, "the default curves pass each channel straight through (red from red, green from green, blue from blue) and add nothing "
                        "from the others", source="convention"))
    out.append(_xt_case("crosstalk_bleed_red_into_green", {("g", "r"): ((0.0, 0.0), (1.0, 0.3))},
                        "the R->G curve adds a share of the red value to green: here green + 0.3 * red"))
    matrix = {("r", "r"): 0.9, ("r", "g"): 0.1, ("r", "b"): 0.0, ("g", "r"): 0.05, ("g", "g"): 0.8, ("g", "b"): 0.15, ("b", "r"): 0.0, ("b", "g"): 0.2, ("b", "b"): 0.7}
    out.append(_xt_case("crosstalk_nine_lines_make_a_matrix", {key: ((0.0, 0.0), (1.0, slope)) for key, slope in matrix.items()},
                        "nine straight curves are a 3 x 3 matrix on the colour: each output channel is the sum of the three input channels scaled by that row's slopes"))
    bent = {("r", "r"): ((0.0, 0.0), (0.5, 0.4), (1.0, 1.0)), ("r", "g"): ((0.0, 0.0), (0.5, 0.2), (1.0, 0.1)), ("r", "b"): ((0.0, 0.1), (1.0, 0.0)),
            ("g", "r"): ((0.0, 0.0), (0.3, 0.1), (1.0, 0.5)), ("g", "g"): ((0.0, 0.0), (0.7, 0.9), (1.0, 1.0)), ("g", "b"): ((0.0, 0.0), (1.0, 0.0)),
            ("b", "r"): ((0.0, 0.2), (0.6, 0.0), (1.0, 0.3)), ("b", "g"): ((0.0, 0.0), (1.0, 0.25)), ("b", "b"): ((0.0, 0.0), (0.4, 0.5), (1.0, 0.8))}
    out.append(_xt_case("crosstalk_sum_of_three_lookups", bent,
                        "every output channel is the sum of three lookups, each through its own bent curve (linear segments between the keys)"))
    out.append(_xt_case("crosstalk_remove_a_channel_from_itself", {("r", "r"): ((0.0, 0.0), (1.0, 0.0)), ("r", "g"): ((0.0, 0.0), (1.0, 1.0))},
                        "with the red curve flat at zero and R<-G passing green through, the red channel becomes the green one"))
    for case_id, what, reason in (
            ("crosstalk_outside_zero_to_one", "values below 0 and above 1",
             "the page does not say how a crossover curve continues past its end keys (NodeBased extends the end segments linearly), so the HDR and negative values are left out"),
            ("crosstalk_smooth_curves", "smooth curves between keys",
             "the page does not name the spline the curve editor fits through added keys, so only linear segments, whose shape is the same in any editor, are compared"),
            ("crosstalk_alpha", "alpha",
             "the page says the output colour channels are sums of lookups and says nothing about alpha (NodeBased leaves it alone)")):
        out.append(Case(case_id, "CrossTalk", "CrossTalk", (), ("C",), "CrossTalk", None, ("C",),
                        lambda fx: (fx["C"].astype(np.float64), np.zeros(fx["C"].shape, bool)), "guide", f"CrossTalk: {what}", unscored=reason))
    return out



# ------------------------------------------------------------------------------------ HistEQ and MinColor (step B), described only in words
def _unscored_case(case_id, nuke, nb_kind, statement, reason, fixture="A"):
    return Case(case_id, nuke, nuke, (), (fixture,), nb_kind, None, (fixture,), lambda fx: (fx[fixture].astype(np.float64), np.zeros(fx[fixture].shape, bool)),
                "guide", statement, unscored=reason)


def _histeq_cases():
    guide = "HistEQ (Nuke 17.0 Reference Guide): 'Alters the histogram of the input image to be flat. This process is called histogram equalization.'"
    return [
        _unscored_case("histeq_flat_histogram", "HistEQ", "HistEQ", guide,
                       "the page states the aim, a flat histogram, and no algorithm: not which channels are equalised together (each on its own or the luminance), the range the "
                       "histogram covers (the data's minimum to maximum, or 0 to 1, and what happens to values outside it), nor where inside a bin a value lands, so no pixel can "
                       "be written down (tests/test_2d_behaviour_color_b.py checks the property instead: the output's histogram is flatter and the order of the values is kept)"),
        _unscored_case("histeq_histogram_entries", "HistEQ", "HistEQ", "HistEQ: 'Histogram Entries' (default 4096) adjusts the number of entries in the histogram",
                       "the page names the knob and nothing else about it: how many entries are better than few, and how Nuke maps values into the entries, is not published"),
    ]


def _mincolor_cases():
    guide = ("MinColor (Nuke 17.0 Reference Guide): 'Calculates the difference between the darkest value in the input image and the target value, and then adds the result to "
             "the image. This sets the darkest pixel to the target value.'")
    return [
        _unscored_case("mincolor_adds_the_delta", "MinColor", "MinColor", guide,
                       "the delta that is added is calculated by the Find Pixel Delta button over a range of frames, not by rendering: the node adds the stored delta "
                       "(tests/test_2d_behaviour_color_b.py runs the button's calculation and the addition against this statement)"),
    ]


# ------------------------------------------------------------------------------------------------------------ Keyer (step C)
_KEYER_GUIDE = ("Keyer (Nuke 17.0 Reference Guide, keyer.html): 'any pixel with a luminance value between B and C becomes opaque. Any pixel value that had a luminance "
                "between C and D, or B and A, has a tapering value somewhere between opaque and transparent'")
_KEYER_NUKE_OPERATION = {"red": "red keyer", "green": "green keyer", "blue": "blue keyer", "max": "max keyer", "min": "min keyer", "luminance": "luminance key",
                         "saturation": "saturation key", "redscreen": "redscreen", "greenscreen": "greenscreen", "bluescreen": "bluescreen"}


def keyer_trapezoid(value, a, b, c, d):
    """The matte of the four range points: 0 up to A, a straight rise to 1 at B, 1 from B to C, a straight fall to 0 at D, 0 beyond (A < B <= C < D)."""
    return np.minimum(np.clip((value - a) / (b - a), 0.0, 1.0), np.clip((d - value) / (d - c), 0.0, 1.0))


def _keyer_value(operation, src):
    r, g, b = src[..., 0], src[..., 1], src[..., 2]
    return {"red": r, "green": g, "blue": b, "max": np.maximum(np.maximum(r, g), b), "min": np.minimum(np.minimum(r, g), b)}[operation]


def _keyer_expected(operation, rng, invert=False, channel=3, from_alpha=False):
    def expected(fx):
        src = fx["A"].astype(np.float64)
        value = src[..., 3] if from_alpha else _keyer_value(operation, src)
        matte = keyer_trapezoid(value, *rng)
        ref = src.copy()
        ref[..., channel] = 1.0 - matte if invert else matte
        return ref, _all_valid(ref)
    return expected


def _keyer_case(case_id, operation, rng, statement, source="convention", invert=False, nb_extra=None, nuke_extra=(), channel=3, from_alpha=False):
    nb = {"keyer_operation": operation, "range_a": rng[0], "range_b": rng[1], "range_c": rng[2], "range_d": rng[3], **(nb_extra or {})}
    if invert:
        nb["invert"] = 1
    nuke = (("operation", _KEYER_NUKE_OPERATION[operation]), ("range", "{%g %g %g %g}" % tuple(rng))) + (() if not invert else (("invert", True),)) + tuple(nuke_extra)
    note = statement if source == "guide" else (f"{statement} ({_KEYER_GUIDE}; the keyed quantity is named by the operation: the page says the operations 'target certain colors "
                                                  f"like green, or certain properties like saturation')")
    return Case(case_id, "Keyer", "Keyer", nuke, ("A",), "Keyer", nb, ("A",), _keyer_expected(operation, rng, invert, channel, from_alpha), source, note)


def _keyer_cases():
    out = []
    rng = (0.05, 0.4, 0.75, 1.5)
    for operation in ("red", "green", "blue", "max", "min"):
        out.append(_keyer_case(f"keyer_{operation}_trapezoid", operation, rng,
                               f"the {operation} operation keys the {'channel' if len(operation) > 3 and operation not in ('max', 'min') else 'value'} through the A, B, C, D trapezoid: "
                               "0 up to A, rising to 1 at B, 1 from B to C, falling to 0 at D, 0 above"))
    out.append(_keyer_case("keyer_triangle_range", "red", (0.0, 0.5, 0.5, 1.0), "with B equal to C the opaque stretch is one point: a triangle from A through B to D"))
    out.append(_keyer_case("keyer_hdr_range", "green", (-0.25, 0.5, 4.0, 12.5), "the range points are not limited to 0 to 1: negative and above-one values key by the same trapezoid"))
    out.append(_keyer_case("keyer_alpha_input", "red", (0.1, 0.3, 0.6, 0.95), "with the input set to the alpha channel the alpha is the keyed quantity",
                           nb_extra={"key_input": "rgba.alpha"}, nuke_extra=(("input", "rgba.alpha"),), from_alpha=True))
    out.append(_keyer_case("keyer_invert", "red", rng, "invert gives the matte's complement, 1 minus the trapezoid", invert=True,
                           source="convention"))
    out.append(_keyer_case("keyer_output_channel", "red", rng, "the output knob names the channel the matte is rendered into (here red): the others keep their values",
                           nb_extra={"key_output": "rgba.red"}, nuke_extra=(("output", "rgba.red"),), channel=0))
    out.append(Case("keyer_output_none", "Keyer", "Keyer", (("output", "none"),), ("A",), "Keyer", {"key_output": "none"}, ("A",),
                    lambda fx: (fx["A"].astype(np.float64), _all_valid(fx["A"])), "convention",
                    "with the output set to none the image passes unchanged (Foundry's Difference and HueKeyer pages: disabling the output 'is the same as setting the channel to none')"))
    for operation, what, reason in (
            ("luminance", "luminance key", "the page says the key is on 'luminance' and shows the graph running 0 to 1, but not the weights of the red, green and blue; Foundry's Rec 709 form "
                                           "(0.2125 / 0.7154 / 0.0721) and the exact Rec 709 form (0.2126 / 0.7152 / 0.0722) differ by up to 3e-4 of the value and the page names neither"),
            ("saturation", "saturation key", "the page names 'saturation' as a keyed property and gives no formula: HSV saturation (max - min) / max, HSL saturation and (max - min) all fit the words"),
            ("redscreen", "redscreen", "the page names no formula for the screen operations: 'red minus the larger of green and blue', 'red minus their mean' and 'red minus green' are all in use"),
            ("greenscreen", "greenscreen", "same as redscreen: no formula for the green screen difference"),
            ("bluescreen", "bluescreen", "same as redscreen: no formula for the blue screen difference")):
        out.append(_unscored_case(f"keyer_{operation}_operation", "Keyer", "Keyer", f"Keyer: the {what} operation ({_KEYER_GUIDE})", reason))
    out.append(_unscored_case("keyer_combine", "Keyer", "Keyer", "Keyer: combine (replace, intersect, union) joins the new matte to what the output channel already holds",
                              "the page gives no formula for intersect and union: the smaller and the larger of the two mattes, their product and A + B - AB all read as 'intersect' and 'union' "
                              "(tests/test_2d_behaviour_keyer.py checks that NodeBased's choice is the smaller and the larger)"))
    return out


# --------------------------------------------------------------------------------------------------------- HueKeyer (step C)
_HK_GUIDE = ("HueKeyer (Nuke 17.0 Reference Guide): 'It calculates hue from the incoming red, green, and blue channels, and uses the amount curve to determine the value to "
             "output in the alpha channel. In the graph, the x axis represents hue and the y axis the alpha value for that hue. [...] drag the point up to a value of 1 on the "
             "y axis to set that hue to black in the alpha channel.' The `invert` knob is on by default")
_HK_NO_SCRIPT = (" The guide gives no text form for the hue curves, so the reference pack's script leaves them at their defaults for this case; compare it with Nuke by hand "
                 "(the x axis here is the hue in degrees, 0 red, 120 green, 240 blue).")


def hue_of(rgb):
    """The standard HSV hue in degrees (0 red, 120 green, 240 blue) of rgb pixels that are not grey."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    top, low = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    spread = np.where(top > low, top - low, 1.0)
    sector = np.where(top == r, ((g - b) / spread) % 6.0, np.where(top == g, (b - r) / spread + 2.0, (r - g) / spread + 4.0))
    return (sector * 60.0) % 360.0


def _hk_expected(points, invert=True, channel=3):
    def expected(fx):
        src = fx["C"].astype(np.float64)
        amount = np.interp(hue_of(src[..., :3]), *zip(*points))
        ref = src.copy()
        ref[..., channel] = 1.0 - amount if invert else amount
        return ref, _all_valid(ref)
    return expected


def _hk_case(case_id, points, statement, invert=True, nb_extra=None, channel=3, source="guide"):
    nb = {"curve_amount": _cl_curve_json(points), **({} if invert else {"invert": 0}), **(nb_extra or {})}
    nuke = (() if invert else (("invert", False),))
    return Case(case_id, "HueKeyer", "HueKeyer", nuke, ("C",), "HueKeyer", nb, ("C",), _hk_expected(points, invert, channel), source, f"{statement} ({_HK_GUIDE}){_HK_NO_SCRIPT}")


def _hue_keyer_cases():
    out = []
    green = ((0.0, 0.0), (90.0, 0.0), (120.0, 1.0), (150.0, 0.0), (360.0, 0.0))
    out.append(_hk_case("huekeyer_default_keys_nothing", ((0.0, 0.0), (360.0, 0.0)), "with the amount curve at its default (0 everywhere) and invert on, the alpha is 1 for every hue"))
    out.append(_hk_case("huekeyer_amount_green_to_black", green, "raising the amount curve to 1 at green sets green hues to black in the alpha (1 minus the amount) and leaves the other hues white"))
    out.append(_hk_case("huekeyer_amount_curve_shape", ((0.0, 0.25), (60.0, 1.0), (180.0, 0.5), (240.0, 0.0), (300.0, 0.75), (360.0, 0.25)),
                        "the alpha follows the amount curve between its points hue by hue (straight lines between points)"))
    out.append(_hk_case("huekeyer_invert_off_is_the_amount", green, "with invert off the alpha is the amount curve itself: 1 at green, 0 elsewhere", invert=False))
    out.append(_hk_case("huekeyer_output_channel", green, "the matte is rendered into the chosen output channel (here red) and the other channels keep their values",
                        nb_extra={"output_channel": "rgba.red"}, channel=0, source="convention"))
    for case_id, what, reason in (
            ("huekeyer_sat_thrsh", "sat_thrsh: 'Increase this threshold to reject hues with low saturation'",
             "the page does not say how a pixel below the threshold is rejected (a hard cut, or a fade in over some range), nor whether saturation is the HSV or the HSL one "
             "(they differ widely for bright colours); tests/test_2d_behaviour_keyer.py checks that a higher threshold removes the keying from the lower-saturation hues"),
            ("huekeyer_curve_smoothing", "the amount curve between points",
             "the page does not name the spline the curve editor fits through added points, so only the straight segments, whose shape is the same in any editor, are compared"),
            ("huekeyer_channels", "channels: 'The hue is calculated from these channels'",
             "the page does not say what hue a pixel has when a channel is left out (as 0, or hue from the remaining two); NodeBased's channels knob limits the channels the node may "
             "write, like every node with a channels knob")):
        out.append(_unscored_case(case_id, "HueKeyer", "HueKeyer", f"HueKeyer: {what}", reason, fixture="C"))
    return out


def cases():
    # Imported here, not at the top: `nuke_fidelity_cases` imports this module from its own last lines, and a module-level import
    # would re-enter it half built when it was itself imported as `tools.nuke_fidelity_cases`.
    global Case, _all_valid
    from nuke_fidelity_cases import Case, _all_valid
    return (_color_correct_cases() + _saturation_cases() + _hue_correct_cases() + _hsv_tool_cases() + _toe_cases() + _softclip_cases()
            + _posterize_cases() + _log2lin_cases() + _ploglin_cases() + _colorspace_cases() + _color_lookup_cases() + _expression_cases() + _crosstalk_cases() + _histeq_cases() + _mincolor_cases()
            + _keyer_cases() + _hue_keyer_cases())
