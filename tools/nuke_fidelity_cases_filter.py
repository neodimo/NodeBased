"""Behaviour cases for the Filter group (lane 9, 2D behaviour step A): Blur, Erode, ErodeFilter, Median, Sharpen, Soften, Glow,
Convolve, Matrix, Laplacian, EdgeDetect and Emboss.

Each case states one behaviour of one node and compares NodeBased's kernel with a reference written here from that statement,
independently of NodeBased's own filter code (plain loops over shifted copies, in float64). `source` is "guide" when the Nuke 17.0
Reference Guide's text states the behaviour and "convention" when only the node's form is long-established; the note says which
page and which reading. Anything that cannot be written down without guessing is `unscored`, with the reason.

Edge handling is stated nowhere for these nodes, so every spatial case is compared on the interior only (the `valid` mask drops the
pixels whose neighbourhood leaves the 16 x 12 fixture); a case that reaches `r` pixels away drops `r` pixels at each edge.
"""
from __future__ import annotations

import math

import numpy as np

GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/"


# ------------------------------------------------------------------------------------------------ reference helpers
def _shifted(a, dy, dx, pad):
    h, w = a.shape[:2]
    p = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    return p[pad + dy:pad + dy + h, pad + dx:pad + dx + w]


def _interior(shape, reach):
    valid = np.zeros(shape, bool)
    if reach:
        valid[reach:-reach, reach:-reach, :] = True
    else:
        valid[...] = True
    return valid


def box_mean(a, r):
    """The plain average of the (2r + 1) x (2r + 1) pixels within r of each pixel."""
    a = a.astype(np.float64)
    total = np.zeros_like(a)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            total += _shifted(a, dy, dx, r)
    return total / float((2 * r + 1) ** 2)


def box_extreme(a, r, use_max):
    a = a.astype(np.float64)
    stack = np.stack([_shifted(a, dy, dx, r) for dy in range(-r, r + 1) for dx in range(-r, r + 1)])
    return stack.max(axis=0) if use_max else stack.min(axis=0)


def window_median(a, r, skip_top_row=False):
    a = a.astype(np.float64)
    rows = range(-r + (1 if skip_top_row else 0), r + 1)
    stack = np.stack([_shifted(a, dy, dx, r) for dy in rows for dx in range(-r, r + 1)])
    return np.median(stack, axis=0)


def correlate(a, weights):
    """Each output pixel is the sum of the weights times the pixels centred on it (the guide's description of Matrix and Convolve)."""
    a = a.astype(np.float64)
    n = weights.shape[0]
    r = n // 2
    out = np.zeros_like(a)
    for y in range(n):
        for x in range(n):
            out += weights[y, x] * _shifted(a, y - r, x - r, r)
    return out


def _channels(array, indices, fn):
    """`array` with `fn` applied to the channels in `indices` only."""
    out = array.copy()
    for c in indices:
        out[..., c] = fn(array)[..., c]
    return out


# ---------------------------------------------------------------------------------------------------------- the cases
def cases():
    from nuke_fidelity_cases import Case

    out = []

    def case(id, nuke, nuke_class, nuke_knobs, nb_kind, nb_params, expected, source, statement, unscored="", nuke_inputs=("A",),
             nb_inputs=("A",), inexpressible=""):
        out.append(Case(id=id, nuke=nuke, nuke_class=nuke_class, nuke_knobs=nuke_knobs, nuke_inputs=nuke_inputs, nb_kind=nb_kind,
                        nb_params=nb_params, nb_inputs=nb_inputs, expected=expected, source=source, statement=statement,
                        inexpressible=inexpressible, unscored=unscored))

    def interior(reach, fn, rgba=(0, 1, 2, 3)):
        """A reference that applies `fn` to the channels in `rgba`, leaves the others, and is valid `reach` pixels in from the edge."""
        def expected(fx):
            src = fx["A"].astype(np.float64)
            ref = src.copy()
            computed = fn(src)
            for c in rgba:
                ref[..., c] = computed[..., c]
            return ref, _interior(src.shape, reach)
        return expected

    blur_note = ("the guide calls size 'the radius within which pixels are compared' and lists box among the filters, taken as the plain "
                 "average of the (2 * size + 1) x (2 * size + 1) pixels within that radius; edge pixels are not compared (the guide does not "
                 "say what lies past the edge)")
    # ------------------------------------------------------------------------------------------------------------ Blur
    case("blur_size_zero_is_identity", "Blur", "Blur", (("size", 0.0), ("filter", "box")), "Blur",
         {"radius": 0.0, "kernel_filter": "box", "quality": 15.0, "crop": 1}, interior(0, lambda s: s), "guide",
         "Blur: size 0 (the guide's default) compares no neighbours, so the picture is unchanged")
    for r in (1, 2):
        case(f"blur_box_{r}", "Blur", "Blur", (("size", float(r)), ("filter", "box")), "Blur",
             {"radius": float(r), "kernel_filter": "box", "quality": 15.0, "crop": 1}, interior(r, lambda s, r=r: box_mean(s, r)),
             "convention", f"Blur box at size {r}: {blur_note}")
    case("blur_box_quality_not_limiting", "Blur", "Blur", (("size", 3.0), ("filter", "box"), ("quality", 3.0)), "Blur",
         {"radius": 3.0, "kernel_filter": "box", "quality": 3.0, "crop": 1}, interior(3, lambda s: box_mean(s, 3)), "guide",
         "Blur: 'the image is scaled down so the blur filter is no larger than' quality, so a filter that already fits (size 3, quality 3) "
         "is not scaled and equals the direct box blur; " + blur_note)
    gaussian_reason = ("the guide does not state the gaussian's sigma for a given size (only that gaussian is the default and the smoothest), "
                       "so its weights cannot be written down without guessing; the shape facts (separable, normalised, symmetric, "
                       "size 0 the identity) are tested in tests/test_2d_behaviour_filter_a.py")
    case("blur_gaussian_weights", "Blur", "Blur", (("size", 2.0), ("filter", "gaussian")), "Blur",
         {"radius": 2.0, "kernel_filter": "gaussian", "quality": 15.0, "crop": 1}, interior(2, lambda s: box_mean(s, 2)), "convention",
         "Blur gaussian: the weights for a given size", unscored=gaussian_reason)
    for shape in ("triangle", "quadratic"):
        case(f"blur_{shape}_weights", "Blur", "Blur", (("size", 2.0), ("filter", shape)), "Blur",
             {"radius": 2.0, "kernel_filter": shape, "quality": 15.0, "crop": 1}, interior(2, lambda s: box_mean(s, 2)), "convention",
             f"Blur {shape}: the weights for a given size",
             unscored=f"the guide names the {shape} filter but not its weights or how size maps to its width; shape facts are tested instead")
    case("blur_quality_scaled", "Blur", "Blur", (("size", 6.0), ("filter", "box"), ("quality", 2.0)), "Blur",
         {"radius": 6.0, "kernel_filter": "box", "quality": 2.0, "crop": 1}, interior(6, lambda s: box_mean(s, 6)), "guide",
         "Blur: a size above quality scales the image down, filters, and scales up with linear interpolation",
         unscored="the guide does not give the scale factor, the down-scaling filter or the block alignment, so the pixels cannot be "
                  "written down; tests check that the result is smooth, bounded and tile-exact")

    # ----------------------------------------------------------------------------------------------- Erode (fast)
    erode_note = ("Erode (fast) guide: 'Positive values increase the light pixels and vice versa' (positive size grows light pixels, "
                  "negative grows dark ones). The neighbourhood is the square of side 2 * |size| + 1, the long-established form of the "
                  "fast min/max filter; the guide does not state its shape")
    for size in (1, 2):
        case(f"erode_fast_positive_{size}_grows_light", "Erode (fast)", "Dilate", (("size", float(size)),), "Erode",
             {"erode_size": float(size), "channels": "rgba"}, interior(size, lambda s, r=size: box_extreme(s, r, use_max=True)),
             "convention", f"Erode (fast) size +{size}: each pixel takes the maximum of its neighbourhood. " + erode_note)
        case(f"erode_fast_negative_{size}_grows_dark", "Erode (fast)", "Dilate", (("size", float(-size)),), "Erode",
             {"erode_size": float(-size), "channels": "rgba"}, interior(size, lambda s, r=size: box_extreme(s, r, use_max=False)),
             "convention", f"Erode (fast) size -{size}: each pixel takes the minimum of its neighbourhood. " + erode_note)
    case("erode_fast_zero_is_identity", "Erode (fast)", "Dilate", (("size", 0.0),), "Erode", {"erode_size": 0.0, "channels": "rgba"},
         interior(0, lambda s: s), "guide", "Erode (fast): size 0 (the guide's default) changes nothing")

    # --------------------------------------------------------------------------------------------- Erode (filter)
    case("erode_filter_exact_response", "Erode (filter)", "FilterErode", (("size", 1.0), ("filter", "box")), "ErodeFilter",
         {"filter_size": 1.0, "filter_type": "box"}, interior(1, lambda s: box_extreme(s, 1, use_max=False)), "convention",
         "Erode (filter): the response at a given size and filter",
         unscored="the guide states the direction (positive size lets darker areas expand, negative lets brighter ones) and that the "
                  "filter trades speed for quality, but not the response curve or the footprint; the direction facts are tested in "
                  "tests/test_2d_behaviour_filter_a.py")

    # ------------------------------------------------------------------------------------------------------ Median
    median_note = ("Median guide: pixels 'within the filter area, defined by the size control, are sorted into order and the median pixel "
                   "is output'; size 3 (the default) is the 3 x 3 area; an even or fractional size is not covered")
    case("median_size_3", "Median", "Median", (("size", 3),), "Median", {"median_size": 3.0}, interior(1, lambda s: window_median(s, 1)),
         "convention", "Median size 3: the median of the 3 x 3 pixels around each pixel. " + median_note)
    case("median_size_5", "Median", "Median", (("size", 5),), "Median", {"median_size": 5.0}, interior(2, lambda s: window_median(s, 2)),
         "convention", "Median size 5: the median of the 5 x 5 pixels around each pixel. " + median_note)
    case("median_ignore_top_line", "Median", "Median", (("size", 3), ("ignore_top_line", True)), "Median",
         {"median_size": 3.0, "ignore_top_line": 1}, interior(1, lambda s: window_median(s, 1, skip_top_row=True)), "guide",
         "Median: ignore top line leaves the top row of the filter box out of the sort (the median of the other six pixels at size 3)")

    # ---------------------------------------------------------------------------------------- Sharpen and Soften
    laplacian_reason = ("the guide says only that Sharpen/Soften 'use Laplacian' (blur minus the original) 'to sharpen/soften specific "
                        "channels' and gives amount, minimum and maximum no description, so the scale of amount, the sign convention and "
                        "the range's role cannot be written down without guessing; tests/test_2d_behaviour_filter_a.py checks the "
                        "behaviour that follows from the names (flat is unchanged, sharpen raises contrast, soften lowers it)")
    for kind, nb_size, name in (("Sharpen", "sharpen_size", "sharpen"), ("Soften", "soften_size", "soften")):
        case(f"{name}_amount_response", kind, kind, (("size", 1.0), ("filter", "box"), ("amount", 1.0)), kind,
             {nb_size: 1.0, "kernel_filter": "box", "minimum": -1e6, "maximum": 1e6, "amount": 1.0} if kind == "Soften" else
             {nb_size: 1.0, "kernel_filter": "box", "minimum": -1e6, "maximum": 1e6, "sharpen_amount": 1.0},
             interior(1, lambda s: s + (s - box_mean(s, 1)) * (1 if kind == "Sharpen" else -1)), "convention",
             f"{kind}: the unsharp-mask reading of 'uses Laplacian'", unscored=laplacian_reason)

    # ------------------------------------------------------------------------------------------------------- Glow
    glow_unscored = ("the guide describes size only as 'values less than 15 increase the dark pixels' (garbled) and does not say how the "
                     "blurred bright part is scaled, tinted or added, so the glow's pixels cannot be written down")
    case("glow_exact_response", "Glow", "Glow2", (("tolerance", 0.2), ("size", 4.0)), "Glow",
         {"glow_threshold": 0.2, "glow_size": 4.0}, interior(0, lambda s: s), "convention", "Glow: the glow's pixels",
         unscored=glow_unscored)
    case("glow_tolerance_above_everything", "Glow", "Glow2", (("tolerance", 100.0),), "Glow",
         {"glow_threshold": 100.0, "glow_size": 4.0}, interior(0, lambda s: s), "guide",
         "Glow: tolerance is 'the low threshold below which no glow is applied', so with every pixel below it the picture is unchanged")
    case("glow_brightness_zero", "Glow", "Glow2", (("brightness", 0.0),), "Glow",
         {"brightness": 0.0, "glow_threshold": 0.0, "glow_size": 4.0}, interior(0, lambda s: s), "guide",
         "Glow: brightness 'sets the brightness of the glow effect', so 0 adds no glow")
    case("glow_tint_black", "Glow", "Glow2", (("tint", 0.0),), "Glow",
         {"red": 0.0, "green": 0.0, "blue": 0.0, "glow_threshold": 0.0, "glow_size": 4.0}, interior(0, lambda s: s), "guide",
         "Glow: tint 'sets the color of the glow effect', so a black tint adds no glow")
    case("glow_effect_only_without_glow", "Glow", "Glow2", (("tolerance", 100.0), ("effect_only", True)), "Glow",
         {"glow_threshold": 100.0, "glow_size": 4.0, "effect_only": 1},
         interior(0, lambda s: np.where(np.arange(4) < 3, 0.0, s), rgba=(0, 1, 2)), "guide",
         "Glow: effect only passes 'only the effect' downstream, so with no glow the colour channels are black (alpha is not glow)")

    # --------------------------------------------------------------------------------------------------- Convolve
    case("convolve_filter_weights", "Convolve", "Convolve2", (("normalize", True),), "Convolve", None, interior(0, lambda s: s), "guide",
         "Convolve: the new value is the sum of each neighbour times the filter pixel on it, the filter divided by its sum when normalize "
         "is on", unscored="the shared fixtures carry no small filter image and the evaluator's single-kernel path has no filter input, so "
                           "the case is run on a 3 x 3 filter in tests/test_2d_behaviour_filter_a.py", inexpressible="needs a filter input")

    # ---------------------------------------------------------------------------------------------------- Matrix
    edge_matrix = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float64)
    all_ones = np.ones((3, 3), np.float64)

    def matrix_params(rows, **extra):
        return {"matrix_size": "3", "mix": 1.0, **{f"weight{i}": float(rows.flat[i]) for i in range(rows.size)}, **extra}

    def matrix_nuke(rows, **extra):
        return (("matrix", "{" + " ".join("{" + " ".join(f"{v:g}" for v in row) + "}" for row in rows) + "}"),) + tuple(extra.items())

    clip_note = "the guide's note: 'Matrix results that fall outside the 0-1 range are clipped'"
    case("matrix_edge_detect_example", "Matrix", "Matrix", matrix_nuke(edge_matrix), "Matrix", matrix_params(edge_matrix),
         interior(1, lambda s: np.clip(correlate(s, edge_matrix), 0.0, 1.0)), "guide",
         "Matrix: the guide's worked example (-1 all round, 8 in the middle) multiplies the centre by 8 and its neighbours by -1 and adds "
         "the nine values; " + clip_note)
    case("matrix_normalize_box", "Matrix", "Matrix", matrix_nuke(all_ones, normalize=True), "Matrix", matrix_params(all_ones, normalize=1),
         interior(1, lambda s: np.clip(correlate(s, all_ones / all_ones.sum()), 0.0, 1.0)), "guide",
         "Matrix normalize: the matrix is divided by a constant so its sum is 1, which keeps the brightness (all ones becomes the 3 x 3 "
         "average); " + clip_note)
    case("matrix_asymmetric_orientation", "Matrix", "Matrix", matrix_nuke(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])), "Matrix",
         matrix_params(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])), interior(1, lambda s: s), "convention",
         "Matrix: whether an asymmetric matrix is applied as written or flipped",
         unscored="the guide's description ('multiplying each pixel by the matching matrix value') does not say whether the weights are "
                  "flipped, so only symmetric matrices are scored")

    # ------------------------------------------------------------------------------------------------------ Laplacian
    for r in (1, 2):
        case(f"laplacian_box_{r}", "Laplacian", "Laplacian", (("size", float(r)), ("filter", "box")), "Laplacian",
             {"laplacian_size": float(r), "kernel_filter": "box", "quality": 15.0, "crop": 1, "channels": "rgba"},
             interior(r, lambda s, r=r: box_mean(s, r) - s), "guide",
             f"Laplacian at size {r}: 'applies a blur, and then subtracts the original from the blurred input' (the blur here the box "
             "filter, whose weights are the plain average over the radius; see the Blur cases)")

    # ---------------------------------------------------------------------------------------------------- EdgeDetect
    for method in ("sobel", "prewitt", "simple"):
        case(f"edgedetect_{method}_response", "EdgeDetect", "EdgeDetectWrapper", (("edgedetector", method),), "EdgeDetect",
             {"edge_type": method.capitalize()}, interior(1, lambda s: s), "convention", f"EdgeDetect {method}: the response's scale",
             unscored="the guide lists the three detectors and says nothing of their kernels, normalisation, or how the channels combine; "
                      "tests check that a flat image gives no edge, that a step gives an edge on both sides of it, and that "
                      "the threshold, blur and erode knobs act")

    # ------------------------------------------------------------------------------------------------------- Emboss
    case("emboss_traditional_response", "Emboss", "Emboss", (("Angle", 0.0), ("Width", 1.0)), "Emboss", {"angle": 0.0, "width": 1.0},
         interior(1, lambda s: s), "convention", "Emboss traditional: the pixels at an angle and width",
         unscored="the guide says Emboss 'offsets the original input using the Angle and Width controls' and gives no formula, "
                  "shading origin or scale; tests check that the picture's relief is symmetric when the angle turns by 180 degrees "
                  "and that angle and width act")
    return out
