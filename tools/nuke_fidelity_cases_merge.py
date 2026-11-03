"""Behaviour cases for the Merge group's selector and blend nodes, and the Image group's Constant (2D behaviour lane 2, step E, November 2026).

Switch, Dissolve, AddMix, Blend, CopyRectangle and Constant. TimeDissolve works across frames and is shown by tests/test_2d_behaviour_merge.py
over several frames; its one-frame kernel is Dissolve, which the Dissolve cases cover. Each case is one stated behaviour of the Nuke node run on
the 16 x 12 fixture through `Evaluator._kernel`, the way `tools/nuke_fidelity_cases.py` runs the Merge cases.

Where the sources come from
* `guide`: the Nuke 17.0 Reference Guide page (learn.foundry.com/nuke/17.0/content/reference_guide/merge_nodes/<node>.html, or image_nodes/constant.html)
  states the behaviour in words.
* `convention`: the guide names the knobs but states no formula for the part in question; the note says what the form rests on.
* A case that cannot be written down without guessing is `unscored` with the reason.

Only names defined above `CASES = _cases()` in `nuke_fidelity_cases` are imported (this module is imported by that line's hook).
"""
from __future__ import annotations

import numpy as np

_WEIGHTS = (0.3, 0.7)


def cases():
    from nuke_fidelity_cases import Case, _all_valid

    out = []

    def pick(which):
        def expected(fx):
            ref = fx["AB"[which]].astype(np.float64)
            return ref, _all_valid(ref)
        return expected

    # ------------------------------------------------------------------------------------------------------- Switch
    switch_inputs = ("A", "B")
    for which in (0, 1):
        out.append(Case(f"switch_which_{which}", "Switch", "Switch", (("which", float(which)),), switch_inputs, "Switch",
                        {"which": which}, switch_inputs, pick(which), "guide",
                        f"Switch: 'Each input is displayed at the value corresponding to the number of the input'; which = {which} shows input {which}"))
    out.append(Case("switch_which_past_last", "Switch", "Switch", (("which", 5.0),), switch_inputs, "Switch", {"which": 5}, switch_inputs, pick(1),
                    "convention", "Switch with which past the last input shows the last input", unscored=(
                        "the guide says only that which = N shows input N and does not say what a value past the last input shows; NodeBased shows "
                        "the last connected input (tests/test_2d_behaviour_merge.py), and no Foundry source to cite for Nuke's own answer was found")))
    out.append(Case("switch_which_fraction", "Switch", "Switch", (("which", 0.5),), switch_inputs, "Switch", {"which": 0}, switch_inputs, pick(0),
                    "convention", "Switch with a fractional which", unscored=(
                        "the guide gives only whole numbers for which; what Nuke does with 0.5 (round, floor, or a mix as Dissolve does) is not stated, "
                        "so NodeBased's whole-number reading is not scored")))

    # ----------------------------------------------------------------------------------------------------- Dissolve
    def dissolve(which):
        def expected(fx):
            ref = fx["A"].astype(np.float64) * (1.0 - which) + fx["B"].astype(np.float64) * which
            return ref, _all_valid(ref)
        return expected
    for which in (0.0, 0.25, 0.5, 1.0):
        out.append(Case(f"dissolve_which_{which}", "Dissolve", "Dissolve", (("which", which),), ("A", "B"), "Dissolve", {"which": which}, ("A", "B"),
                        dissolve(which), "guide",
                        f"Dissolve: 'set this to an integer to show that input, or a fractional value to dissolve between the two nearest inputs'; "
                        f"which = {which} mixes input 0 and input 1 by {which} in every channel, alpha included"))

    # ------------------------------------------------------------------------------------------------------ TimeDissolve
    # The reference pack renders frame 1. NodeBased turns the frame, `in`, `out` and the curve into `which` before the kernel runs, so each case
    # gives the `which` the range yields at frame 1 and tests/test_2d_behaviour_merge.py shows the turn itself over many frames.
    td_note = ("TimeDissolve: 'dissolves between two inputs, starting the dissolve at the in frame and ending at the out frame'; the curve is 0 before the "
               "dissolve (input A) and 1 after it (input B)")
    for case_id, knobs, which, what in (
            ("timedissolve_before_in", (("in", 1.0), ("out", 10.0)), 0.0, "frame 1 with in = 1: the dissolve has not begun, all A"),
            ("timedissolve_after_out", (("in", -9.0), ("out", 0.0)), 1.0, "frame 1 with out = 0: the dissolve is over, all B"),
            ("timedissolve_middle", (("in", 0.0), ("out", 2.0)), 0.5, "frame 1 halfway between in = 0 and out = 2: an even mix")):
        out.append(Case(case_id, "TimeDissolve", "TimeDissolve", knobs, ("A", "B"), "TimeDissolve", {"which": which}, ("A", "B"), dissolve(which), "guide",
                        f"{td_note}; {what}"))

    # ------------------------------------------------------------------------------------------------------- AddMix
    add_note = ("AddMix: 'the alpha of input A is used to index two color correction lookup curves; the first is used to multiply input A, "
                "the second to multiply input B, and the results are added together'")

    def addmix(lut_a, lut_b, premultiplied=False):
        def expected(fx):
            a, b = fx["A"].astype(np.float64), fx["B"].astype(np.float64)
            alpha = a[..., 3:4]
            xs_a, ys_a = zip(*lut_a)
            xs_b, ys_b = zip(*lut_b)
            weight_a, weight_b = np.interp(alpha, xs_a, ys_a), np.interp(alpha, xs_b, ys_b)
            colour = a[..., :3]
            if premultiplied:
                colour = np.where(alpha > 0, colour / np.where(alpha > 0, alpha, 1.0), 0.0)
            ref = np.zeros_like(a)
            ref[..., :3] = colour * weight_a + b[..., :3] * weight_b
            valid = np.ones(a.shape, bool)
            valid[..., 3] = False                 # the guide does not say how the alpha channel itself is multiplied
            return ref, valid
        return expected
    rising, falling = ((0.0, 0.0), (1.0, 1.0)), ((0.0, 1.0), (1.0, 0.0))
    out.append(Case("addmix_default_curves", "AddMix", "AddMix", (), ("B", "A"), "AddMix", {}, ("A", "B"), addmix(rising, falling), "convention",
                    add_note + ". Colour only: the guide gives no default curve shapes, but calls AddMix 'a similar operation to the Merge node's over "
                    "mode' that 'also premultiplies', which a rising A curve and a falling B curve give (A * a + B * (1 - a))"))
    moved_a, moved_b = ((0.0, 0.0), (0.5, 1.0), (1.0, 1.0)), ((0.0, 1.0), (0.5, 0.25), (1.0, 0.0))
    out.append(Case("addmix_moved_curves", "AddMix", "AddMix", (), ("B", "A"), "AddMix", {"lut_a": "0:0 0.5:1 1:1", "lut_b": "0:1 0.5:0.25 1:0"},
                    ("A", "B"), addmix(moved_a, moved_b), "guide", add_note + ". Colour only; A's curve reaches 1 at alpha 0.5, B's passes 0.25"))
    out.append(Case("addmix_premultiplied", "AddMix", "AddMix", (("premultiplied", True),), ("B", "A"), "AddMix", {"premultiplied": 1}, ("A", "B"),
                    addmix(rising, falling, premultiplied=True), "guide",
                    "AddMix premultiplied: 'colors in the A input are divided by their alpha before being multiplied by the A lookup curve'. Colour only"))
    out.append(Case("addmix_nonlinear", "AddMix", "AddMix", (("nonlinear", True),), ("B", "A"), "AddMix", {"nonlinear": 1}, ("A", "B"), addmix(rising, falling),
                    "convention", "AddMix nonlinear remaps the lookup index", unscored=(
                        "the guide says only that the zero-to-infinity range is remapped logarithmically into 0-1; it gives no formula, so the "
                        "log(1 + a) / (1 + log(1 + a)) remap NodeBased uses cannot be scored")))

    # -------------------------------------------------------------------------------------------------------- Blend
    def blend(weights, normalize=True):
        def expected(fx):
            a, b = fx["A"].astype(np.float64), fx["B"].astype(np.float64)
            total = sum(weights)
            ref = weights[0] * a + weights[1] * b
            if normalize:
                ref = ref / total if total else np.zeros_like(ref)
            return ref, _all_valid(ref)
        return expected
    blend_note = "Blend: 'creating the weighted average of all of the inputs'; normalize: 'the inputs are summed up and each of them divided by the total'"
    out.append(Case("blend_equal", "Blend", "Blend", (), ("A", "B"), "Blend", {}, ("A", "B"), blend((1.0, 1.0)), "guide", blend_note + "; equal weights give the mean"))
    out.append(Case("blend_unequal", "Blend", "Blend", (("weight0", _WEIGHTS[0]), ("weight1", _WEIGHTS[1])), ("A", "B"), "Blend",
                    {"weight0": _WEIGHTS[0], "weight1": _WEIGHTS[1]}, ("A", "B"), blend(_WEIGHTS), "guide",
                    blend_note + f"; weights {_WEIGHTS[0]} and {_WEIGHTS[1]} give (0.3 A + 0.7 B) / 1"))
    out.append(Case("blend_unequal_total", "Blend", "Blend", (("weight0", 2.0), ("weight1", 1.0)), ("A", "B"), "Blend", {"weight0": 2.0, "weight1": 1.0},
                    ("A", "B"), blend((2.0, 1.0)), "guide", blend_note + "; weights 2 and 1 are divided by their total of 3"))
    out.append(Case("blend_all_zero_weights", "Blend", "Blend", (("weight0", 0.0), ("weight1", 0.0)), ("A", "B"), "Blend", {"weight0": 0.0, "weight1": 0.0},
                    ("A", "B"), blend((0.0, 0.0)), "convention", "Blend with every weight zero and normalize on", unscored=(
                        "the weights sum to zero, so dividing by the total is undefined and the guide does not say what Nuke returns; "
                        "NodeBased returns black (tests/test_2d_behaviour_merge.py)")))

    # ---------------------------------------------------------------------------------------------- CopyRectangle
    area = {"area_x": 3.0, "area_y": 2.0, "area_r": 11.0, "area_t": 9.0}

    def copy_rect(channels=(0, 1, 2, 3), soft=0.0):
        def expected(fx):
            a, b = fx["A"].astype(np.float64), fx["B"].astype(np.float64)
            h, w = a.shape[:2]
            px, py = np.arange(w) + 0.5, np.arange(h) + 0.5
            depth = np.minimum(np.minimum(px[None, :] - area["area_x"], area["area_r"] - px[None, :]),
                               np.minimum(py[:, None] - area["area_y"], area["area_t"] - py[:, None]))
            band = soft * 0.5 * min(area["area_r"] - area["area_x"], area["area_t"] - area["area_y"])
            gate = np.clip(depth / band, 0.0, 1.0) if band > 0 else (depth > 0).astype(np.float64)
            ref = b.copy()
            for c in channels:
                ref[..., c] = b[..., c] + (a[..., c] - b[..., c]) * gate
            return ref, np.ones(a.shape, bool)
        return expected
    copy_note = ("CopyRectangle: 'copy a rectangular area from one input on top of another'; the rectangle is area (x, y, r, t), rows counted from the "
                 "top as everywhere in NodeBased, a pixel being inside when its centre is")
    out.append(Case("copyrectangle_default_channels", "CopyRectangle", "CopyRectangle", (("area", "3 2 11 9"),), ("B", "A"), "CopyRectangle", dict(area),
                    ("A", "B"), copy_rect(), "guide", copy_note + ". All four channels of A replace B inside the box, B stays outside"))
    out.append(Case("copyrectangle_channels", "CopyRectangle", "CopyRectangle", (("area", "3 2 11 9"), ("channels", "rgb")), ("B", "A"), "CopyRectangle",
                    dict(area, channels="rgb"), ("A", "B"), copy_rect((0, 1, 2)), "guide",
                    copy_note + ". 'Sets the channels to copy from input A within the specified rectangle': with rgb, alpha stays B's"))
    out.append(Case("copyrectangle_softness", "CopyRectangle", "CopyRectangle", (("area", "3 2 11 9"), ("softness", 0.5)), ("B", "A"), "CopyRectangle",
                    dict(area, softness=0.5), ("A", "B"), copy_rect(soft=0.5), "convention",
                    copy_note + ". Softness: 'vignette the edges of the copied rectangle; the larger the value, the more of the area around the edges is "
                    "faded to input B'", unscored=(
                        "the guide gives no fade width or curve for softness, so the linear fade is NodeBased's own and cannot be scored against Nuke")))

    # --------------------------------------------------------------------------------------------------- Constant
    def constant(width, height, colour):
        def expected(fx):
            ref = np.empty((height, width, 4), np.float64)
            ref[:] = colour
            return ref, np.ones(ref.shape, bool)
        return expected
    const_note = "Constant: 'generates an image where every pixel is the same color'"
    out.append(Case("constant_color", "Constant", "Constant", (("color", "0.25 0.5 0.75 1"),), (), "Constant",
                    {"red": 0.25, "green": 0.5, "blue": 0.75, "alpha": 1.0, "width": 16, "height": 12}, (), constant(16, 12, (0.25, 0.5, 0.75, 1.0)), "guide",
                    const_note + "; with an opaque colour the stored value is the colour itself (an alpha below 1 is listed separately)"))
    out.append(Case("constant_color_hdr", "Constant", "Constant", (("color", "4 -0.5 0 1"),), (), "Constant",
                    {"red": 4.0, "green": -0.5, "blue": 0.0, "alpha": 1.0, "width": 16, "height": 12}, (), constant(16, 12, (4.0, -0.5, 0.0, 1.0)), "guide",
                    const_note + "; values above 1 and below 0 are kept (the knob has no range)"))
    out.append(Case("constant_format", "Constant", "Constant", (("format", "10 7"), ("color", "0.25 0.5 0.75 1")), (), "Constant",
                    {"red": 0.25, "green": 0.5, "blue": 0.75, "alpha": 1.0, "width": 10, "height": 7}, (), constant(10, 7, (0.25, 0.5, 0.75, 1.0)), "guide",
                    "Constant format: 'sets the constant's format'; the picture is as wide and as tall as the format and every pixel is the colour"))
    out.append(Case("constant_color_alpha", "Constant", "Constant", (("color", "0.5 0.5 0.5 0.5"),), (), "Constant",
                    {"red": 0.5, "green": 0.5, "blue": 0.5, "alpha": 0.5, "width": 16, "height": 12}, (), constant(16, 12, (0.5, 0.5, 0.5, 0.5)), "convention",
                    const_note + "; a colour with alpha below 1", unscored=(
                        "NodeBased stores the colour multiplied by its alpha (0.25 for the case here); the guide does not say whether Nuke's colour knob is "
                        "written as typed or premultiplied, so the stored value is not scored")))
    out.append(Case("constant_channels_rgb", "Constant", "Constant", (("channels", "rgb"), ("color", "0.25 0.5 0.75 1")), (), "Constant",
                    {"red": 0.25, "green": 0.5, "blue": 0.75, "alpha": 1.0, "width": 16, "height": 12, "channels": "rgb"}, (),
                    constant(16, 12, (0.25, 0.5, 0.75, 0.0)), "guide",
                    "Constant channels: 'the constant only contains these channels'; with rgb the alpha is not generated and reads 0"))
    return out
