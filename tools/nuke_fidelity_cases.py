"""Fixtures and fixture cases shared by tools/audit_nuke_fidelity.py and tools/make_nuke_reference_pack.py.

A case is one node, one set of knob values, and the fixture images it reads. It carries two things the ruler needs:

* the NodeBased side: kind, params and input fixtures, run on the evaluator's own kernel (`Evaluator._kernel`);
* the Nuke side: the node class, knob values and input order, written out as a `.nk` script by the reference pack.

Where the Nuke 17.0 Reference Guide states what a node computes, `expected` implements that statement and the audit
compares NodeBased against it. `source` says where the statement comes from: "guide" (the guide's text), or
"convention" (the guide names the knobs but states no formula; the form is Nuke's long-standing one, and the lane
brief that asked for it). Real Nuke renders, when they exist, replace both.

Everything here is deterministic arithmetic: no random number generator, so every machine builds the same pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

H, W = 12, 16
_RGB = (-0.5, 0.0, 0.05, 0.18, 0.4, 0.5, 0.75, 1.0, 1.5, 4.0, 12.0)       # 11 values: negative, black, mid, one, HDR
_ALPHA = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)                                  # 6 values: empty, partial, opaque
_MASK = (0.0, 0.25, 0.5, 0.75, 1.0)


def _fixtures_ab():
    """A varies fastest, B slowest, so each channel's (A, B) value pairs cover the whole 11 x 11 table."""
    a = np.zeros((H, W, 4), np.float32)
    b = np.zeros((H, W, 4), np.float32)
    for y in range(H):
        for x in range(W):
            i = y * W + x
            for c in range(3):
                a[y, x, c] = _RGB[(i + 3 * c) % len(_RGB)]
                b[y, x, c] = _RGB[(i // len(_RGB) + 2 + 5 * c) % len(_RGB)]
            a[y, x, 3] = _ALPHA[i % len(_ALPHA)]
            b[y, x, 3] = _ALPHA[(i // len(_ALPHA) + 1) % len(_ALPHA)]
    return a, b


def make_fixtures():
    a, b = _fixtures_ab()
    mask = np.zeros((H, W, 4), np.float32)
    for y in range(H):
        for x in range(W):
            i = y * W + x
            for c in range(4):
                mask[y, x, c] = _MASK[(i // (c + 1) + c) % len(_MASK)]
    return {"A": a, "B": b, "mask": mask}


# ---------------------------------------------------------------------------------------------------- Merge formulas
MERGE_OPERATIONS_GUIDE = (
    "atop", "average", "color-burn", "color-dodge", "conjoint-over", "copy", "difference", "disjoint-over", "divide", "exclusion",
    "from", "geometric", "hard-light", "hypot", "in", "mask", "matte", "max", "min", "minus", "multiply", "out", "over", "overlay",
    "plus", "screen", "soft-light", "stencil", "under", "xor")
# Operations whose guide text is too garbled to implement without guessing: reported, not scored.
MERGE_UNSCORED = {"matte": "the guide's algorithm text for matte ('screen(A,a,B,b) if A==a, B if a<=0, ...') cannot be implemented without guessing"}


def merge_reference(op, a, b):
    """The guide's stated algorithm, per channel, with a and b the A and B alpha (the guide's lower-case symbols).

    Returns (reference, valid): `valid` is False where the stated formula divides by zero or is otherwise undefined.
    """
    A, B = a.astype(np.float64), b.astype(np.float64)
    al, be = A[..., 3:4], B[..., 3:4]           # a and b in the guide's notation
    valid = np.ones(A.shape, bool)
    with np.errstate(all="ignore"):
        if op == "atop":
            r = A * be + B * (1 - al)
        elif op == "average":
            r = (A + B) / 2
        elif op == "color-burn":
            r = 1 - (1 - B) / np.maximum(A, 1 - B)
        elif op == "color-dodge":
            r = B / np.maximum(1 - A, B)
        elif op == "conjoint-over":
            r = np.where(al > be, A, A + B * (1 - al / be))
        elif op == "copy":
            r = A.copy()
        elif op == "difference":
            r = np.abs(A - B)
        elif op == "disjoint-over":
            r = np.where(al + be < 1, A + B, A + B * (1 - al) / be)
        elif op == "divide":
            r = np.where(B > 0, A / B, A)
        elif op == "exclusion":
            r = A + B - 2 * A * B
        elif op == "from":
            r = B - A
        elif op == "geometric":
            r = np.where((A > 0) & (B > 0), 2 * A * B / (A + B), 0.0)
        elif op == "hard-light":
            r = np.where(2 * A < 1, 2 * A * B, np.where((A < 1) & (B < 1), 1 - 2 * (1 - A) * (1 - B), 0.0))
        elif op == "hypot":
            r = np.sqrt(A * A + B * B)
        elif op == "in":
            r = A * be
        elif op == "mask":
            r = B * al
        elif op == "max":
            r = np.maximum(A, B)
        elif op == "min":
            r = np.minimum(A, B)
        elif op == "minus":
            r = A - B
        elif op == "multiply":
            r = np.where((A < 0) & (B < 0), A, A * B)
        elif op == "out":
            r = A * (1 - be)
        elif op == "over":
            r = A + B * (1 - al)
        elif op == "overlay":
            r = np.where(2 * B < 1, 2 * A * B, 1 - 2 * (1 - A) * (1 - B))
        elif op == "plus":
            r = A + B
        elif op == "screen":
            r = np.where((A > 0) & (A < 1) & (B > 0) & (B < 1), A + B - A * B, np.maximum(A, B))
        elif op == "soft-light":
            ab = A * B
            r = np.where(ab < 1, 2 * ab + B * (1 - ab), 2 * ab)
        elif op == "stencil":
            r = B * (1 - al)
        elif op == "under":
            r = A * (1 - be) + B
        elif op == "xor":
            r = A * (1 - be) + B * (1 - al)
        else:
            raise ValueError(op)
    r = np.broadcast_to(r, A.shape).astype(np.float64)
    valid &= np.isfinite(r)
    return r, valid


# ----------------------------------------------------------------------------------------------------------- cases
@dataclass(frozen=True)
class Case:
    id: str
    nuke: str                                  # key in docs/parity/nuke_nodes.json
    nuke_class: str                            # node class in a .nk script
    nuke_knobs: tuple                          # ((knob, value), ...) in .nk spelling
    nuke_inputs: tuple                         # fixture names by Nuke input index (0 first)
    nb_kind: str
    nb_params: Optional[dict]                  # None: NodeBased has no knob that can express this setting
    nb_inputs: tuple                           # fixture names in NodeBased slot order (inputs, then optional inputs)
    expected: Callable                         # fixtures -> (reference array, valid mask)
    source: str                                # "guide" or "convention"
    statement: str                             # the stated behaviour, in our words
    inexpressible: str = ""                    # why nb_params is None
    unscored: str = ""                         # why the case is reported but not scored


def _all_valid(ref):
    return np.ones(ref.shape, bool)


def _rgb_only(fn):
    """Reference for a node that leaves alpha alone: apply `fn` to the rgb channels of input 0."""
    def expected(fx):
        src = fx["A"].astype(np.float64)
        out = src.copy()
        valid = np.ones(src.shape, bool)
        out[..., :3], valid[..., :3] = fn(src[..., :3])
        return out, valid
    return expected


def _cases():
    out = []
    for op in MERGE_OPERATIONS_GUIDE:
        out.append(Case(
            id=f"merge_{op}", nuke="Merge", nuke_class="Merge2", nuke_knobs=(("operation", op),), nuke_inputs=("B", "A"),
            nb_kind="Merge", nb_params={"operation": op}, nb_inputs=("A", "B"),
            expected=(lambda fx, op=op: merge_reference(op, fx["A"], fx["B"])), source="guide",
            statement=f"Merge {op}: the guide's stated algorithm", unscored=MERGE_UNSCORED.get(op, "")))

    def keymix(fx):
        m = fx["mask"][..., 3:4].astype(np.float64)
        ref = fx["A"].astype(np.float64) * m + fx["B"].astype(np.float64) * (1 - m)
        return ref, _all_valid(ref)
    out.append(Case("keymix_default", "KeyMix", "Keymix", (), ("B", "A", "mask"), "Keymix", {}, ("A", "B", "mask"), keymix, "guide",
                    "KeyMix follows the formula Aa + B(1-a), a being the mask"))

    def gamma(value):
        def fn(rgb):
            valid = rgb >= 0
            return np.where(valid, np.power(np.maximum(rgb, 0), 1.0 / value), 0.0), valid
        return _rgb_only(fn)
    for value in (2.2, 0.5):
        out.append(Case(f"gamma_{value}", "Gamma", "Gamma", (("channels", "rgb"), ("value", value)), ("A",), "Gamma",
                        {"gamma": value, "channels": "rgb"}, ("A",), gamma(value), "guide",
                        "Gamma: 'the actual function is 1/pow(value)', read as out = in ** (1 / value); tested for in >= 0, where it is unambiguous"))
    out.append(Case("invert_rgb", "Invert", "Invert", (("channels", "rgb"),), ("A",), "Invert", {"channels": "rgb"}, ("A",),
                    _rgb_only(lambda rgb: (1.0 - rgb, np.ones(rgb.shape, bool))), "guide", "Invert subtracts the channel's values from one"))
    out.append(Case("multiply_1.5", "Multiply", "Multiply", (("channels", "rgb"), ("value", 1.5)), ("A",), "Multiply",
                    {"multiply": 1.5, "channels": "rgb"}, ("A",), _rgb_only(lambda rgb: (rgb * 1.5, np.ones(rgb.shape, bool))), "guide",
                    "Multiply multiplies a channel's values by a factor"))
    out.append(Case("add_0.25", "Add", "Add", (("channels", "rgb"), ("value", 0.25)), ("A",), "Add", {"offset": 0.25, "channels": "rgb"}, ("A",),
                    _rgb_only(lambda rgb: (rgb + 0.25, np.ones(rgb.shape, bool))), "guide", "Add adds a fixed value to the channel"))

    def clamp(lo, hi, lo_to=None, hi_to=None):
        def fn(rgb):
            r = np.clip(rgb, lo, hi)
            if lo_to is not None:
                r = np.where(rgb < lo, lo_to, r)
            if hi_to is not None:
                r = np.where(rgb > hi, hi_to, r)
            return r, np.ones(rgb.shape, bool)
        return _rgb_only(fn)
    out.append(Case("clamp_default", "Clamp", "Clamp", (("channels", "rgb"),), ("A",), "Clamp", {"channels": "rgb"}, ("A",), clamp(0.0, 1.0), "guide",
                    "Clamp: by default clamps channels to values between 0 and 1"))
    out.append(Case("clamp_0.2_0.8", "Clamp", "Clamp", (("channels", "rgb"), ("minimum", 0.2), ("maximum", 0.8)), ("A",), "Clamp",
                    {"channels": "rgb", "minimum": 0.2, "maximum": 0.8}, ("A",), clamp(0.2, 0.8), "guide",
                    "Clamp: values below minimum become minimum, values above maximum become maximum"))
    out.append(Case("clamp_to_values", "Clamp", "Clamp",
                    (("channels", "rgb"), ("minimum", 0.2), ("maximum", 0.8), ("MinClampTo", 0.4), ("MinClampTo_enable", True), ("MaxClampTo", 0.6),
                     ("MaxClampTo_enable", True)), ("A",), "Clamp", None, ("A",), clamp(0.2, 0.8, 0.4, 0.6), "guide",
                    "Clamp: with MinClampTo/MaxClampTo enabled, out-of-range values are set to those values instead",
                    inexpressible="NodeBased's Clamp has no MinClampTo or MaxClampTo"))

    def premult(fx):
        s = fx["A"].astype(np.float64)
        ref = s.copy()
        ref[..., :3] = s[..., :3] * s[..., 3:4]
        return ref, _all_valid(ref)
    out.append(Case("premult_default", "Premult", "Premult", (), ("A",), "Premult", {}, ("A",), premult, "guide",
                    "Premult multiplies the input's rgb channels by its alpha"))

    def unpremult(fx):
        s = fx["A"].astype(np.float64)
        ref = s.copy()
        valid = np.ones(s.shape, bool)
        positive = s[..., 3:4] > 0
        ref[..., :3] = np.where(positive, s[..., :3] / np.where(positive, s[..., 3:4], 1.0), s[..., :3])
        valid[..., :3] = np.broadcast_to(positive, s[..., :3].shape)
        return ref, valid
    out.append(Case("unpremult_default", "Unpremult", "Unpremult", (), ("A",), "Unpremult", {}, ("A",), unpremult, "guide",
                    "Unpremult divides the channels by alpha; tested where alpha > 0 (the guide does not say what happens at alpha 0)"))

    rows = ((0.5, 1.0, 2.0), (0.0, 1.0, 0.0), (0.25, 0.0, 1.0))

    def colormatrix(fx):
        s = fx["A"].astype(np.float64)
        ref = s.copy()
        ref[..., :3] = np.einsum("ij,yxj->yxi", np.array(rows), s[..., :3])
        return ref, _all_valid(ref)
    out.append(Case("colormatrix_rows", "ColorMatrix", "ColorMatrix",
                    (("channels", "rgb"), ("matrix", "{{0.5 1 2} {0 1 0} {0.25 0 1}}")), ("A",), "ColorMatrix",
                    {f"matrix_{r}{c}": rows[r][c] for r in range(3) for c in range(3)}, ("A",), colormatrix, "guide",
                    "ColorMatrix: each output channel is the matrix row applied to the input r, g, b"))

    def grade(blackpoint=0.0, whitepoint=1.0, lift=0.0, gain=1.0, multiply=1.0, offset=0.0, gamma=1.0, black_clamp=True, white_clamp=False):
        def fn(rgb):
            scale = multiply * (gain - lift) / (whitepoint - blackpoint)
            shift = offset + lift - scale * blackpoint
            r = scale * rgb + shift
            if gamma != 1.0:
                r = np.where(r > 0, np.power(np.maximum(r, 0), 1.0 / gamma), r)
            if black_clamp:
                r = np.maximum(r, 0.0)
            if white_clamp:
                r = np.minimum(r, 1.0)
            return r, np.ones(rgb.shape, bool)
        return _rgb_only(fn)
    grade_note = ("Grade (convention, not stated in the 17.0 page): A = multiply * (gain - lift) / (whitepoint - blackpoint); B = offset + lift - A * blackpoint; "
                  "out = A * in + B, then ** (1 / gamma) for out > 0, then the clamps")
    out.append(Case("grade_affine", "Grade", "Grade", (("multiply", 1.5), ("add", 0.1), ("black_clamp", False)), ("A",), "Grade",
                    {"multiply": 1.5, "offset": 0.1}, ("A",), grade(multiply=1.5, offset=0.1, black_clamp=False), "convention", grade_note))
    out.append(Case("grade_default_clamps", "Grade", "Grade", (("multiply", 1.5), ("add", 0.1)), ("A",), "Grade",
                    {"multiply": 1.5, "offset": 0.1}, ("A",), grade(multiply=1.5, offset=0.1), "convention",
                    grade_note + "; the guide's defaults enable black clamp"))
    out.append(Case("grade_levels", "Grade", "Grade",
                    (("blackpoint", 0.05), ("whitepoint", 2.0), ("black", 0.1), ("white", 0.9), ("black_clamp", False)), ("A",), "Grade", None, ("A",),
                    grade(blackpoint=0.05, whitepoint=2.0, lift=0.1, gain=0.9, black_clamp=False), "convention", grade_note,
                    inexpressible="NodeBased's Grade has no blackpoint, whitepoint, lift or gain"))
    out.append(Case("grade_gamma", "Grade", "Grade", (("gamma", 1.8),), ("A",), "Grade", None, ("A",), grade(gamma=1.8), "convention", grade_note,
                    inexpressible="NodeBased's Grade has no gamma"))

    def exposure(stops, blackpoint=0.0):
        return _rgb_only(lambda rgb: ((rgb - blackpoint) * 2.0 ** stops, np.ones(rgb.shape, bool)))
    exp_note = "Exposure in Stops mode (convention, not stated in the 17.0 page): out = (in - blackpoint) * 2 ** stops"
    out.append(Case("exposure_stops", "Exposure", "Exposure", (("mode", "Stops"), ("red", 1.0), ("green", 1.0), ("blue", 1.0)), ("A",), "Exposure",
                    {"exposure_mode": "stops", "red": 1.0, "gang": 1}, ("A",), exposure(1.0), "convention", exp_note))
    out.append(Case("exposure_stops_blackpoint", "Exposure", "Exposure",
                    (("mode", "Stops"), ("red", -1.0), ("green", -1.0), ("blue", -1.0), ("blackpoint", 0.1)), ("A",), "Exposure",
                    {"exposure_mode": "stops", "red": -1.0, "gang": 1, "blackpoint": 0.1}, ("A",), exposure(-1.0, 0.1), "convention", exp_note))
    return out


CASES = _cases()


def run_nodebased(case, fx=None):
    """Run a case on NodeBased's evaluator kernel; returns the output array, or None when NodeBased cannot express the case."""
    if case.nb_params is None:
        return None
    from nodebased import core
    from nodebased.imaging import Evaluator
    fx = fx or make_fixtures()
    spec = core.SPECS[case.nb_kind]
    slots = [fx[name] for name in case.nb_inputs]
    slots += [None] * (len(spec["inputs"]) + len(spec.get("optional_inputs", [])) - len(slots))
    return np.asarray(Evaluator._kernel(case.nb_kind, dict(spec["params"], **case.nb_params), slots), dtype=np.float64)


def cases_for(nuke_key):
    return [c for c in CASES if c.nuke == nuke_key]
