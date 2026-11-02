"""Behaviour cases for the Filter group (lane 9, 2D behaviour step A): Blur, Erode, ErodeFilter, Median, Sharpen, Soften, Glow,
Convolve, Matrix, Laplacian, EdgeDetect and Emboss.

Each case states one behaviour of one node and compares NodeBased's kernel with a reference written here from that statement,
independently of NodeBased's own filter code (plain loops over shifted copies, in float64). `source` is "guide" when the Nuke 17.0
Reference Guide's text states the behaviour and "convention" when only the node's form is long-established; the note says which
page and which reading. Anything that cannot be written down without guessing is `unscored`, with the reason.

Step B (the Transform group) adds the cases at the end of `cases()`; their sources are the guide's own tables (fetched from the node pages
under .../reference_guide/transform_nodes/) and, where the guide only draws a curve, the published kernel the curve belongs to.

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


# ------------------------------------------------------------------------------------------- Transform group helpers
TRANSFORM_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/transform_nodes/"


def _keys_kernel(a):
    def weight(x):
        x = abs(x)
        if x <= 1.0:
            return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
        if x < 2.0:
            return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
        return 0.0
    return weight


def _mitchell_kernel(b, c):
    def weight(x):
        x = abs(x)
        if x < 1.0:
            return ((12 - 9 * b - 6 * c) * x ** 3 + (-18 + 12 * b + 6 * c) * x ** 2 + (6 - 2 * b)) / 6.0
        if x < 2.0:
            return ((-b - 6 * c) * x ** 3 + (6 * b + 30 * c) * x ** 2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6.0
        return 0.0
    return weight


def _row_resample(a, positions, weight, radius=2):
    """`a` sampled along x at fractional pixel positions (pixel i is at i) with the 1-D kernel `weight`, in plain loops."""
    a = a.astype(np.float64)
    h, w = a.shape[:2]
    out = np.zeros((h, len(positions), a.shape[2]))
    inside = np.zeros(len(positions), bool)
    for n, pos in enumerate(positions):
        base = int(math.floor(pos))
        taps = [(j, weight(pos - j)) for j in range(base - radius + 1, base + radius + 1)]
        inside[n] = all(0 <= j < w for j, _ in taps)
        for j, wt in taps:
            if 0 <= j < w:
                out[:, n] += wt * a[:, j]
    return out, inside


def _shift_reference(kernel, shift=0.5, radius=2):
    """A reference for a picture moved `shift` pixels to the right: output pixel x reads the input at x - shift."""
    def expected(fx):
        src = fx["A"].astype(np.float64)
        h, w = src.shape[:2]
        rows, inside = _row_resample(src, [x - shift for x in range(w)], kernel, radius)
        # The same kernel along y at whole-pixel positions: zero off the centre for an interpolating kernel, not for Mitchell or Parzen.
        out = np.zeros_like(rows)
        ok = np.zeros(h, bool)
        for y in range(h):
            taps = [(j, kernel(float(y - j))) for j in range(y - radius + 1, y + radius + 1)]
            ok[y] = all(0 <= j < h for j, _ in taps)
            for j, wt in taps:
                if 0 <= j < h:
                    out[y] += wt * rows[j]
        return out, np.broadcast_to((ok[:, None] & inside[None, :])[..., None], src.shape).copy()
    return expected


def _moved(fx, dx, dy, edge):
    """The picture moved right by `dx` and up by `dy` whole pixels: with `edge` the outside takes the outermost pixels, else it is invalid."""
    src = fx["A"].astype(np.float64)
    h, w = src.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    sy, sx = ys + dy, xs - dx
    inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
    out = src[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)]
    return out, (np.ones(src.shape, bool) if edge else np.broadcast_to(inside[..., None], src.shape).copy()), inside


def _placed(fx, out_w, out_h, scale_x, scale_y, offset_x, offset_y, edge=False, flip=False, flop=False, turn=False):
    """The picture `fx["A"]` scaled by (`scale_x`, `scale_y`) with its upper left corner at (`offset_x`, `offset_y`) inside an
    `out_w` x `out_h` format, each output pixel taking the input pixel its centre lies in (impulse), then flipped, flopped or turned
    counter-clockwise. Outside the picture the colour is zero (alpha not compared) or, with `edge`, the nearest edge pixel."""
    src = fx["A"].astype(np.float64)
    h, w = src.shape[:2]
    if turn:                                                   # work in the frame the picture is placed in, then turn it
        out_w, out_h = out_h, out_w
    ref = np.zeros((out_h, out_w, 4))
    valid = np.ones((out_h, out_w, 4), bool)
    for y in range(out_h):
        for x in range(out_w):
            i = int(math.floor((x + 0.5 - offset_x) / scale_x))
            j = int(math.floor((y + 0.5 - offset_y) / scale_y))
            if 0 <= i < w and 0 <= j < h:
                ref[y, x] = src[j, i]
            elif edge:
                ref[y, x] = src[min(max(j, 0), h - 1), min(max(i, 0), w - 1)]
            else:
                valid[y, x, 3] = False
    if turn:
        ref, valid = np.rot90(ref, 1), np.rot90(valid, 1)       # numpy's rot90 is counter-clockwise on the picture as shown
    if flip:
        ref, valid = ref[::-1], valid[::-1]
    if flop:
        ref, valid = ref[:, ::-1], valid[:, ::-1]
    return np.ascontiguousarray(ref), np.ascontiguousarray(valid)


def _homography(src_pts, dst_pts):
    """The 3 x 3 projective map taking the four `src_pts` to the four `dst_pts`: the null vector of the 8 x 9 system, by SVD."""
    rows = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    h = np.linalg.svd(np.asarray(rows, np.float64))[2][-1].reshape(3, 3)
    return h / h[2, 2]


def _pinned(fx, from_pts, to_pts, edge=False, invert=False):
    """The picture `fx["A"]` pinned from the `from_pts` quad onto the `to_pts` quad, each output pixel taking the input pixel the
    inverse map puts under its centre (impulse); `invert` swaps the two quads. Outside the picture: zero (alpha not compared) or the edge."""
    src = fx["A"].astype(np.float64)
    h, w = src.shape[:2]
    forward = _homography(to_pts if invert else from_pts, from_pts if invert else to_pts)
    inverse = np.linalg.inv(forward)
    ref = np.zeros_like(src)
    valid = np.ones(src.shape, bool)
    for y in range(h):
        for x in range(w):
            u = inverse @ np.array([x + 0.5, y + 0.5, 1.0])
            i, j = int(math.floor(u[0] / u[2])), int(math.floor(u[1] / u[2]))
            if 0 <= i < w and 0 <= j < h:
                ref[y, x] = src[j, i]
            elif edge:
                ref[y, x] = src[min(max(j, 0), h - 1), min(max(i, 0), w - 1)]
            else:
                valid[y, x, 3] = False
    return ref, valid


FRAME_QUAD = ((0.0, 0.0), (16.0, 0.0), (0.0, 12.0), (16.0, 12.0))


def _pin_params(from_pts, to_pts, **more):
    out = {"filter": "nearest"}
    for i, ((fx_, fy_), (tx_, ty_)) in enumerate(zip(from_pts, to_pts), 1):
        out.update({f"from{i}_x": fx_, f"from{i}_y": fy_, f"to{i}_x": tx_, f"to{i}_y": ty_})
    return {**out, **more}


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
    # ================================================================================================ the Transform group (step B)
    axis_note = ("positive translate x moves the picture right and positive translate y up the screen (the guide's Position page: "
                 "'Positive values move the image up or to the right'; Transform and Position share Nuke's axes). Rows count from the top "
                 "in NodeBased, so up is fewer rows")

    # ------------------------------------------------------------------------------------------------------- Transform
    transform_guide = TRANSFORM_GUIDE + "transform.html"

    def transform(id, knobs, nb, expected, source, statement, **kw):
        case(id, "Transform", "Transform", tuple(knobs), "Transform", {"filter": "impulse", **nb}, expected, source,
             statement + f" ({transform_guide})", **kw)

    transform("transform_identity_exact", (("translate", "{0 0}"),), {}, interior(0, lambda s: s), "guide",
              "Transform: with every knob at its default the picture comes out unchanged (the guide's defaults: translate 0 0, rotate 0, scale 1)")
    transform("transform_translate_right_up", (("translate", "{3 2}"),), {"translate_x": 3.0, "translate_y": 2.0},
              lambda fx: _moved(fx, 3, 2, False)[:2], "convention", "Transform translate 3, 2 with impulse: " + axis_note +
              "; compared where the moved picture covers the frame")
    transform("transform_translate_negative", (("translate", "{-4 -1}"),), {"translate_x": -4.0, "translate_y": -1.0},
              lambda fx: _moved(fx, -4, -1, False)[:2], "convention", "Transform translate -4, -1 with impulse moves the picture left and down; " + axis_note)
    transform("transform_black_outside_off_takes_edge_pixels", (("translate", "{3 2}"), ("black_outside", False)),
              {"translate_x": 3.0, "translate_y": 2.0, "black_outside": 0}, lambda fx: _moved(fx, 3, 2, True)[:2], "guide",
              "Transform: 'If you uncheck this control, the outside area is filled with the outermost pixels of the image sequence'")

    def outside_black(fx):
        out, _valid, inside = _moved(fx, 3, 2, False)
        ref = np.where(inside[..., None], out, 0.0)
        valid = np.ones(ref.shape, bool)
        valid[..., 3] = inside                                  # the guide says black pixels, not what alpha the outside carries
        return ref, valid
    transform("transform_black_outside_on_is_black", (("translate", "{3 2}"), ("black_outside", True)),
              {"translate_x": 3.0, "translate_y": 2.0, "black_outside": 1}, outside_black, "guide",
              "Transform: black outside 'renders as black pixels outside the image boundary' (the outside alpha is not stated, so only the "
              "colour channels are compared there)")

    def scaled_up(fx):
        src = fx["A"].astype(np.float64)
        h, w = src.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        sx = np.floor((xs + 0.5 - w / 2) / 2 + w / 2).astype(int)         # the picture doubled about its centre
        sy = np.floor((ys + 0.5 - h / 2) / 2 + h / 2).astype(int)
        return src[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)], np.ones(src.shape, bool)
    transform("transform_scale_two_about_center", (("scale", 2.0), ("center", "{8 6}")),
              {"scale": 2.0, "center_x": 8.0, "center_y": 6.0}, scaled_up, "guide",
              "Transform scale 2: 'Scales the image width and height around the center x y coordinates'; with impulse each output pixel carries "
              "one input pixel's value (the guide's impulse), the one the doubled picture puts there")

    def turned(angle_steps):
        def expected(fx):
            src = fx["A"].astype(np.float64)
            h, w = src.shape[:2]
            out = np.zeros_like(src)
            valid = np.zeros(src.shape, bool)
            cx, cy = w / 2.0, h / 2.0
            for y in range(h):
                for x in range(w):
                    dx, dy_up = x + 0.5 - cx, -(y + 0.5 - cy)
                    # counter-clockwise by 90 degrees on the screen: the source is the point 90 degrees clockwise of the pixel
                    sx_up, sy_up = dy_up, -dx
                    sx, sy = int(math.floor(cx + sx_up)), int(math.floor(cy - sy_up))
                    if 0 <= sx < w and 0 <= sy < h:
                        out[y, x] = src[sy, sx]
                        valid[y, x] = True
            return out, valid
        return expected
    transform("transform_rotate_ninety_counter_clockwise", (("rotate", 90.0), ("center", "{8 6}")),
              {"rotate": 90.0, "center_x": 8.0, "center_y": 6.0}, turned(1), "convention",
              "Transform rotate 90 about the centre: positive angles turn the picture counter-clockwise on the screen (Nuke's rotate; the "
              "Reformat page says its turn 'rotates the image 90 degrees counter-clockwise' in the same axes)")
    transform("transform_invert_undoes_translate", (("translate", "{3 2}"), ("invert_matrix", True)),
              {"translate_x": 3.0, "translate_y": 2.0, "invert": 1}, lambda fx: _moved(fx, -3, -2, False)[:2], "guide",
              "Transform invert: 'any transform you applied using the translate, rotate, scale, skew or center controls is inverted', so "
              "translate 3, 2 inverted moves the picture by -3, -2")

    kernel_note = ("the guide draws each filter's curve and names it; the kernel is the published one the name belongs to (%s), applied "
                   "separably at a half-pixel move; compared where the whole kernel lies inside the frame")
    for name, kernel, note in (("keys", _keys_kernel(-0.5), "Keys 1981 cubic convolution, a = -0.5 (the Catmull-Rom spline)"),
                               ("rifman", _keys_kernel(-1.0), "Rifman 1973 cubic convolution, a = -1"),
                               ("mitchell", _mitchell_kernel(1 / 3.0, 1 / 3.0), "Mitchell-Netravali, B = C = 1/3"),
                               ("parzen", _mitchell_kernel(1.0, 0.0), "the cubic B-spline, which is the Parzen window")):
        case(f"transform_filter_{name}_half_pixel", "Transform", "Transform", (("translate", "{0.5 0}"), ("filter", name.capitalize())),
             "Transform", {"translate_x": 0.5, "filter": name}, _shift_reference(kernel), "convention",
             f"Transform filter {name} at a half-pixel move: " + kernel_note % note + f" ({transform_guide})")
    case("transform_filter_cubic_half_pixel", "Transform", "Transform", (("translate", "{0.5 0}"), ("filter", "Cubic")), "Transform",
         {"translate_x": 0.5, "filter": "cubic"}, _shift_reference(_keys_kernel(-0.5)), "convention",
         "Transform filter cubic at a half-pixel move: the guide says 'remapped pixels receive some smoothing' and does not give the curve, "
         "taken here as Keys' a = -0.5 cubic (NodeBased's own cubic is the Catmull-Rom spline)", unscored="the guide names Cubic without its coefficients; "
         "tests/test_2d_behaviour_transform.py checks that cubic smooths without overshoot at a step")
    case("transform_filter_simon_half_pixel", "Transform", "Transform", (("translate", "{0.5 0}"), ("filter", "Simon")), "Transform",
         {"translate_x": 0.5, "filter": "simon"}, _shift_reference(_keys_kernel(-0.75)), "convention",
         "Transform filter simon at a half-pixel move", unscored="the guide says Simon gives 'medium sharpening' between Keys (minor) and Rifman "
         "(significant) and publishes no coefficient; the order of the three is tested in tests/test_2d_behaviour_transform.py")
    for name, note in (("lanczos4", "Lanczos with two lobes"), ("lanczos6", "Lanczos with three lobes"), ("sinc4", "the sinc cut at two lobes")):
        case(f"transform_filter_{name}_half_pixel", "Transform", "Transform", (("translate", "{0.5 0}"), ("filter", name.capitalize())),
             "Transform", {"translate_x": 0.5, "filter": name}, _shift_reference(_keys_kernel(-0.5)), "convention",
             f"Transform filter {name} at a half-pixel move ({note})",
             unscored="the guide says only that Lanczos4, Lanczos6 and Sinc4 sharpen (Lanczos4 least, Sinc4 most); whether the taps are "
                      "renormalised to sum to 1 is not stated, so the weights are not scored; the order and constant-picture facts are "
                      "tested in tests/test_2d_behaviour_transform.py")
    case("transform_filter_notch", "Transform", "Transform", (("translate", "{0.5 0}"), ("filter", "Notch")), "Transform", None,
         interior(0, lambda s: s), "convention", "Transform filter notch: 'flat smoothing (which tends to hide moire patterns)'",
         inexpressible="NodeBased offers no notch filter",
         unscored="the guide says only that Notch is flat smoothing and gives neither its width nor its curve, so no kernel can be written "
                  "down without guessing; NodeBased does not offer it")
    transform("transform_skew_matrix", (("skewX", 0.2),), {"skew_x": 0.2}, interior(0, lambda s: s), "convention",
              "Transform skew X and skew Y: the shear's size and sign per knob value",
              unscored="the guide says only 'Skews the image on the x axis' and gives neither the unit of the knob nor the sign; "
                       "tests/test_2d_behaviour_transform.py checks that the two axes shear different ways and that skew order matters")
    transform("transform_matrix_order", (("rotate", 30.0), ("scale", "{2 1}")), {"rotate": 30.0, "scale_mode": "xy", "scale_x": 2.0, "scale_y": 1.0},
              interior(0, lambda s: s), "convention", "Transform: the order in which scale, skew, rotate and translate combine",
              unscored="the guide does not give the matrix order; NodeBased applies scale, then skew, then rotate, then translate about the centre "
                       "(tests/test_2d_behaviour_transform.py pins what it does)")
    transform("transform_motion_blur_shutter", (("motionblur", 1.0), ("shutter", 0.5)), {"motionblur": 1, "shutter": 0.5},
              interior(0, lambda s: s), "guide", "Transform motion blur: the blur over the shutter's frames",
              unscored="the case needs an animated transform over time and the harness runs one frame; tests/test_2d_behaviour_transform.py "
                       "checks the shutter windows the guide gives (centred, start, end, custom) and that a still transform stays sharp")
    # ------------------------------------------------------------------------------------------------------- Reformat
    reformat_page = TRANSFORM_GUIDE + "reformat.html"

    def reformat(id, nuke_knobs, nb, expected, source, statement, **kw):
        case(id, "Reformat", "Reformat", tuple(nuke_knobs), "Reformat",
             {"filter": "nearest", "reformat_type": "to_box", "box_fixed": 1, **nb}, expected, source,
             statement + f" ({reformat_page})", **kw)

    def box(w, h, **more):
        return {"width": w, "height": h, **more}

    def nk(w, h, **more):
        return (("type", "to box"), ("box_width", w), ("box_height", h), ("box_fixed", True), *more.items())

    reformat("reformat_distort_doubles", nk(32, 24, resize="distort"), box(32, 24, resize_type="distort"),
             lambda fx: _placed(fx, 32, 24, 2.0, 2.0, 0.0, 0.0), "guide",
             "Reformat distort: 'scale the original so that both sides fill the output dimensions' (a 16 x 12 picture into 32 x 24 doubles each pixel)")
    reformat("reformat_fit_letterboxes", nk(32, 48, resize="fit"), box(32, 48, resize_type="fit"),
             lambda fx: _placed(fx, 32, 48, 2.0, 2.0, 0.0, 12.0), "convention",
             "Reformat fit: the whole picture inside the output, proportions kept, centred (a 16 x 12 picture into 32 x 48 doubles and leaves "
             "12 black rows above and below). The guide's wording for fit and fill ('smallest side', 'longest side') is garbled; this is "
             "Nuke's established behaviour")
    reformat("reformat_fill_crops", nk(32, 48, resize="fill"), box(32, 48, resize_type="fill"),
             lambda fx: _placed(fx, 32, 48, 4.0, 4.0, -16.0, 0.0), "convention",
             "Reformat fill: the picture covers the whole output, proportions kept, centred (a 16 x 12 picture into 32 x 48 is scaled by 4 "
             "and loses 16 pixels each side); see the fit case for the guide's wording")
    reformat("reformat_width_scales_to_the_width", nk(32, 20, resize="width"), box(32, 20, resize_type="width"),
             lambda fx: _placed(fx, 32, 20, 2.0, 2.0, 0.0, -2.0), "guide",
             "Reformat width: 'scale the original until its width matches the output width. Height is then scaled to preserve the aspect "
             "ratio' (the 24 rows that result are centred in 20, two cut off above and below)")
    reformat("reformat_height_scales_to_the_height", nk(20, 24, resize="height"), box(20, 24, resize_type="height"),
             lambda fx: _placed(fx, 20, 24, 2.0, 2.0, -6.0, 0.0), "guide",
             "Reformat height: 'scale the original so that it fills the output height. Width is then scaled to preserve the aspect ratio'")
    reformat("reformat_none_does_not_resize", nk(24, 20, resize="none"), box(24, 20, resize_type="none"),
             lambda fx: _placed(fx, 24, 20, 1.0, 1.0, 4.0, 4.0), "guide",
             "Reformat none: 'do not resize the original' (centred in the output, nothing scaled)")
    reformat("reformat_center_off_lines_up_the_lower_left", nk(32, 48, resize="fit", center=False),
             box(32, 48, resize_type="fit", center=0, legacy_reformat_layout=0),
             lambda fx: _placed(fx, 32, 48, 2.0, 2.0, 0.0, 24.0), "guide",
             "Reformat: center off, 'Reformat translates the image so that the lower left corners line up' (rows run downwards here, so "
             "the picture sits against the bottom)")
    reformat("reformat_flip_is_upside_down", nk(16, 12, resize="distort", flip=True), box(16, 12, resize_type="distort", flip=1),
             lambda fx: _placed(fx, 16, 12, 1.0, 1.0, 0.0, 0.0, flip=True), "guide", "Reformat flip: 'flips the image upside down'")
    reformat("reformat_flop_is_left_right", nk(16, 12, resize="distort", flop=True), box(16, 12, resize_type="distort", flop=1),
             lambda fx: _placed(fx, 16, 12, 1.0, 1.0, 0.0, 0.0, flop=True), "guide", "Reformat flop: 'flops the image left and right'")
    reformat("reformat_turn_is_counter_clockwise", nk(12, 16, resize="distort", turn=True),
             box(12, 16, resize_type="distort", turn=1, legacy_reformat_layout=0),
             lambda fx: _placed(fx, 12, 16, 1.0, 1.0, 0.0, 0.0, turn=True), "guide",
             "Reformat turn: 'rotates the image 90 degrees counter-clockwise' (the box is the turned picture's size)")
    case("reformat_scale_type_halves", "Reformat", "Reformat", (("type", "scale"), ("scale", 0.5), ("resize", "distort")), "Reformat",
         {"filter": "nearest", "reformat_type": "scale", "scale": 0.5, "resize_type": "distort"},
         lambda fx: _placed(fx, 8, 6, 0.5, 0.5, 0.0, 0.0), "guide",
         "Reformat scale 0.5: 'set the output width and height to a multiple of the input size' (a 16 x 12 picture becomes 8 x 6, each output "
         f"pixel taking the input pixel its centre lies in) ({reformat_page})")
    case("reformat_scale_split_scales_each_direction", "Reformat", "Reformat", (("type", "scale"), ("scale", "{0.5 0.25}"), ("resize", "distort")),
         "Reformat", {"filter": "nearest", "reformat_type": "scale", "scale": [0.5, 0.25], "resize_type": "distort"},
         lambda fx: _placed(fx, 8, 3, 0.5, 0.25, 0.0, 0.0), "guide",
         "Reformat scale: 'To scale each direction separately using different scale factors, click the 2 button' (0.5 across and 0.25 "
         f"down make a 16 x 12 picture 8 x 3) ({reformat_page})")
    reformat("reformat_force_this_shape_off_keeps_the_shape", (("type", "to box"), ("box_width", 40), ("box_height", 40), ("box_fixed", False),
                                                                 ("resize", "width")),
             box(40, 40, resize_type="width", box_fixed=0), lambda fx: _placed(fx, 40, 30, 2.5, 2.5, 0.0, 0.0), "guide",
             "Reformat force this shape off: 'the output image is approximately the same shape as the original, rounded to the nearest integer "
             "number of pixels' (a 16 x 12 picture to a 40 wide box at resize width makes 40 x 30)")
    reformat("reformat_black_outside_off_takes_edge_pixels", nk(32, 48, resize="fit", black_outside=False),
             box(32, 48, resize_type="fit", black_outside=0), lambda fx: _placed(fx, 32, 48, 2.0, 2.0, 0.0, 12.0, edge=True), "guide",
             "Reformat: 'If you uncheck [black outside], the outside area is filled with the outermost pixels of the image sequence'")
    reformat("reformat_black_outside_on_is_black", nk(32, 48, resize="fit", black_outside=True),
             box(32, 48, resize_type="fit", black_outside=1), lambda fx: _placed(fx, 32, 48, 2.0, 2.0, 0.0, 12.0), "guide",
             "Reformat black outside on: 'This renders as black pixels outside the image boundary' (alpha outside is not compared)")
    reformat("reformat_pixel_aspect", nk(32, 24, resize="distort", box_pixel_aspect=2.0), box(32, 24, resize_type="distort", pixel_aspect=2.0),
             lambda fx: _placed(fx, 32, 24, 2.0, 2.0, 0.0, 0.0), "guide",
             "Reformat pixel aspect: 'sets the pixel aspect ratio for the output image'",
             unscored="the guide says only that it sets the output's pixel aspect ratio, not what happens to the pixels; NodeBased carries the "
                      "value and works in square pixels")
    # ------------------------------------------------------------------------------------------------------- CornerPin2D
    pin_page = TRANSFORM_GUIDE + "cornerpin2d.html"

    def pin(id, to_pts, expected, source, statement, from_pts=FRAME_QUAD, nuke_more=(), **more):
        knobs = tuple((f"to{i}", "{%g %g}" % pt) for i, pt in enumerate(to_pts, 1)) + tuple(nuke_more)
        case(id, "CornerPin2D", "CornerPin2D", knobs, "CornerPin", _pin_params(from_pts, to_pts, **more), expected, source,
             statement + f" ({pin_page})")

    shifted = tuple((x + 3.0, y + 2.0) for x, y in FRAME_QUAD)
    doubled = tuple((2 * x, 2 * y) for x, y in FRAME_QUAD)
    skewed = ((1.0, 1.0), (14.0, 2.0), (2.0, 11.0), (15.0, 9.7))
    pin("cornerpin_identity", FRAME_QUAD, lambda fx: _pinned(fx, FRAME_QUAD, FRAME_QUAD), "convention",
        "CornerPin2D with the to points on the from points leaves the picture as it is (the default)")
    pin("cornerpin_translation", shifted, lambda fx: _pinned(fx, FRAME_QUAD, shifted), "guide",
        "CornerPin2D: 'map the four corners of an image sequence to ... positions' (the corners move 3 right and 2 down, so the picture "
        "does; where it no longer reaches the frame is black)")
    pin("cornerpin_scale_two", doubled, lambda fx: _pinned(fx, FRAME_QUAD, doubled), "guide",
        "CornerPin2D: the four corners mapped onto a quad twice as large doubles the picture about the origin")
    pin("cornerpin_perspective", skewed, lambda fx: _pinned(fx, FRAME_QUAD, skewed), "guide",
        "CornerPin2D: the four corners of the picture land on the four to points, the picture in between following the projective map "
        "through them (a general quad, compared pixel by pixel against an independently solved homography)")
    pin("cornerpin_invert", shifted, lambda fx: _pinned(fx, FRAME_QUAD, shifted, invert=True), "guide",
        "CornerPin2D invert: 'the current to values are inverted' (the shift of 3 right and 2 down becomes 3 left and 2 up)",
        nuke_more=(("invert", True),), direction="inverse")
    pin("cornerpin_black_outside_off_takes_edge_pixels", shifted, lambda fx: _pinned(fx, FRAME_QUAD, shifted, edge=True), "guide",
        "CornerPin2D: with black outside unchecked 'the outside area is filled with the outermost pixels of the image sequence'",
        nuke_more=(("black_outside", False),), black_outside=0)
    pin("cornerpin_from_points_choose_the_part_that_is_pinned", FRAME_QUAD,
        lambda fx: _pinned(fx, ((4.0, 3.0), (12.0, 3.0), (4.0, 9.0), (12.0, 9.0)), FRAME_QUAD, edge=True), "guide",
        "CornerPin2D: the from points name the corners of the feature that is mapped onto the to points (the middle 8 x 6 of a 16 x 12 "
        "picture stretched over the frame), outside it the edge pixels",
        from_pts=((4.0, 3.0), (12.0, 3.0), (4.0, 9.0), (12.0, 9.0)), nuke_more=(("black_outside", False),), black_outside=0)
    # ================================================================================== the window nodes (step B, second half)
    def canvas_of(fx, window=None):
        """(source rows, the data window's x, y, w, h) of the picture `fx["A"]` as a case runs it: the whole format or `window`."""
        src = fx["A"].astype(np.float64)
        return src, (window or (0, 0, src.shape[1], src.shape[0]))

    def window_case(id, nuke, nuke_class, nuke_knobs, nb_kind, nb_params, expected, source, statement, page, **kw):
        case(id, nuke, nuke_class, tuple(nuke_knobs), nb_kind, nb_params, expected, source, statement + f" ({TRANSFORM_GUIDE}{page})", **kw)

    def painted(fx, rows, cols, source_fn):
        """A format-sized reference, zero where `rows` x `cols` (ranges) do not reach, `source_fn(y, x)` inside them."""
        src = fx["A"].astype(np.float64)
        ref = np.zeros_like(src)
        for y in rows:
            for x in cols:
                ref[y, x] = source_fn(src, y, x)
        return ref, np.ones(src.shape, bool)

    clip = lambda v, lo, hi: min(max(v, lo), hi)           # noqa: E731

    # ----------------------------------------------------------------------------------------------------------------- Crop
    box = (4, 3, 8, 6)                                                   # x, y (from the top), width, height
    crop_nb = {"_windowed": True, "x": 4, "y": 3, "width": 8, "height": 6}
    window_case("crop_keeps_the_box_and_blacks_the_rest", "Crop", "Crop", (("box", "{4 3 12 9}"),), "Crop", dict(crop_nb),
                lambda fx: painted(fx, range(3, 9), range(4, 12), lambda src, y, x: src[y, x]), "guide",
                "Crop: 'The area of the input image you want to keep. Anything outside this box is cropped' (the box counts rows from the top "
                "here, Nuke from the bottom)", "crop.html")
    window_case("crop_reformat_makes_the_box_the_format", "Crop", "Crop", (("box", "{4 3 12 9}"), ("reformat", True)), "Crop",
                {**crop_nb, "reformat": 1},
                lambda fx: (fx["A"].astype(np.float64)[3:9, 4:12], np.ones((6, 8, 4), bool)), "guide",
                "Crop reformat: 'the image output format is changed to match the cropped image' (the picture shown is the box)", "crop.html")
    window_case("crop_black_outside_off_takes_edge_pixels", "Crop", "Crop", (("box", "{4 3 12 9}"), ("crop", False)), "Crop",
                {**crop_nb, "crop": 0, "intersect": 0, "_data_window": (6, 4, 4, 4)},
                lambda fx: painted(fx, range(3, 9), range(4, 12), lambda src, y, x: src[clip(y, 4, 7), clip(x, 6, 9)]), "guide",
                "Crop: 'If you uncheck [black outside], the outside area is filled with the outermost pixels of the image sequence' (the part of "
                "the box the picture does not reach, here, is the picture's edge pixels)", "crop.html")
    window_case("crop_softness_vignette", "Crop", "Crop", (("box", "{4 3 12 9}"), ("softness", 2.0)), "Crop", {**crop_nb, "crop_softness": 2.0},
                lambda fx: painted(fx, range(3, 9), range(4, 12), lambda src, y, x: src[y, x]), "convention",
                "Crop softness: 'vignette the edges of the cropped portion ... The larger the value, the more of the area around the edges is "
                "faded to black'", "crop.html",
                unscored="the guide states that softness fades the edges toward black and that 0 is no vignette, but not the ramp's shape or "
                         "how softness converts to pixels; tests/test_2d_behaviour_transform.py checks that 0 changes nothing and that a "
                         "larger value darkens more")

    # ---------------------------------------------------------------------------------------------------------------- Mirror
    mirror_page = "mirror.html"
    window_case("mirror_horizontal_flops", "Mirror", "Mirror", (("Horizontal", True),), "Mirror", {"flip_x": 1},
                lambda fx: (fx["A"].astype(np.float64)[:, ::-1], np.ones(fx["A"].shape, bool)), "guide",
                "Mirror horizontal: 'flops the image' about the centre of the format (left and right change places)", mirror_page)
    window_case("mirror_vertical_flips", "Mirror", "Mirror", (("Vertical", True),), "Mirror", {"flip_y": 1},
                lambda fx: (fx["A"].astype(np.float64)[::-1], np.ones(fx["A"].shape, bool)), "guide",
                "Mirror vertical: 'flips the image' about the centre of the format (top and bottom change places)", mirror_page)
    window_case("mirror_both_turns_the_picture_half_way_round", "Mirror", "Mirror", (("Horizontal", True), ("Vertical", True)), "Mirror",
                {"flip_x": 1, "flip_y": 1}, lambda fx: (fx["A"].astype(np.float64)[::-1, ::-1], np.ones(fx["A"].shape, bool)), "guide",
                "Mirror horizontal and vertical together turn the picture 180 degrees about the centre of the format", mirror_page)
    window_case("mirror_reflects_a_partial_picture_about_the_format_centre", "Mirror", "Mirror", (("Horizontal", True),), "Mirror",
                {"flip_x": 1, "_data_window": (2, 3, 6, 6)},
                lambda fx: painted(fx, range(3, 9), range(8, 14), lambda src, y, x: src[y, 15 - x]), "guide",
                "Mirror: 'flips the input image around the center of the format area', so a picture covering only columns 2 to 7 of a 16 wide "
                "format ends up on columns 8 to 13", mirror_page)

    # -------------------------------------------------------------------------------------------------------------- Position
    position_page = "position.html"
    window_case("position_moves_up_and_right_and_repeats_the_edge", "Position", "Position", (("translate", "{3 2}"),), "Position",
                {"translate_x": 3, "translate_y": 2},
                lambda fx: painted(fx, range(12), range(16), lambda src, y, x: src[clip(y + 2, 0, 11), clip(x - 3, 0, 15)]), "guide",
                "Position: 'Positive values move the image up or to the right' and it 'fills the area between the original image and the moved "
                "image by repeating the edge pixels' (a 16 x 12 picture moved 3 right and 2 up, seen through its own format)", position_page)
    window_case("position_negative_moves_down_and_left", "Position", "Position", (("translate", "{-3 -2}"),), "Position",
                {"translate_x": -3, "translate_y": -2},
                lambda fx: painted(fx, range(12), range(16), lambda src, y, x: src[clip(y - 2, 0, 11), clip(x + 3, 0, 15)]), "guide",
                "Position: 'negative values move it down or to the left'", position_page)
    window_case("position_moves_a_partial_picture_and_fills_the_gap", "Position", "Position", (("translate", "{3 2}"),), "Position",
                {"translate_x": 3, "translate_y": 2, "_data_window": (4, 3, 8, 6)},
                lambda fx: painted(fx, range(1, 9), range(4, 15), lambda src, y, x: src[clip(y + 2, 3, 8), clip(x - 3, 4, 11)]), "guide",
                "Position: the area between the original picture (columns 4 to 11, rows 3 to 8) and the moved one (3 right, 2 up) is filled with "
                "the moved picture's edge pixels; nothing lies beyond the union of the two", position_page)

    # -------------------------------------------------------------------------------------------------------------- AdjustBBox
    adjust_page = "adjustbbox.html"
    window_case("adjustbbox_grows_and_repeats_the_edge", "AdjustBBox", "AdjustBBox", (("numpixels", 2.0),), "AdjustBBox",
                {"numpixels": 2, "_data_window": (4, 3, 8, 6)},
                lambda fx: painted(fx, range(1, 11), range(2, 14), lambda src, y, x: src[clip(y, 3, 8), clip(x, 4, 11)]), "guide",
                "AdjustBBox: adds 2 pixels to each side of the bounding box; the new area repeats the box's outermost pixels (the BlackOutside "
                "page: 'removes stripes ... caused by the edge pixels of the bounding box being replicated' after an AdjustBBox)", adjust_page)
    window_case("adjustbbox_negative_crops_the_box", "AdjustBBox", "AdjustBBox", (("numpixels", -1.0),), "AdjustBBox",
                {"numpixels": -1, "_data_window": (4, 3, 8, 6)},
                lambda fx: painted(fx, range(4, 8), range(5, 11), lambda src, y, x: src[y, x]), "guide",
                "AdjustBBox: 'adding or removing pixels': a negative number takes a pixel off each side of the box", adjust_page)
    window_case("adjustbbox_width_and_height_split", "AdjustBBox", "AdjustBBox", (("numpixels", "{3 1}"),), "AdjustBBox",
                {"numpixels": [3, 1], "_data_window": (4, 3, 8, 6)},
                lambda fx: painted(fx, range(2, 10), range(1, 15), lambda src, y, x: src[clip(y, 3, 8), clip(x, 4, 11)]), "guide",
                "AdjustBBox: 'Click the 2 button to display separate fields for width (w) and height (h)' (3 pixels each side across, 1 up and down)",
                adjust_page)

    # ------------------------------------------------------------------------------------------------------------ BlackOutside
    window_case("blackoutside_fills_outside_the_box_with_black", "BlackOutside", "BlackOutside", (), "BlackOutside",
                {"_data_window": (4, 3, 8, 6)},
                lambda fx: painted(fx, range(3, 9), range(4, 12), lambda src, y, x: src[y, x]), "guide",
                "BlackOutside: 'fills everything outside the bounding box area with black' (the box's own pixels are kept)", "blackoutside.html")

    # ---------------------------------------------------------------------------------------------------------------- Tile
    tile_page = "tile.html"
    window_case("tile_one_by_one_is_the_picture", "Tile", "Tile", (("rows", 1), ("columns", 1)), "Tile", {"rows": 1, "columns": 1, "filter": "cubic"},
                lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "convention",
                "Tile with one row and one column is the picture itself", tile_page)
    window_case("tile_rows_and_columns_scale_the_picture_into_tiles", "Tile", "Tile",
                (("rows", 2), ("columns", 2), ("filter", "Impulse")), "Tile", {"rows": 2, "columns": 2, "filter": "nearest"},
                lambda fx: painted(fx, range(12), range(16), lambda src, y, x: src[0, 0]), "convention",
                "Tile: the picture repeated rows x columns times, each copy scaled down to fit", tile_page,
                unscored="the page lists the filters and the knobs but states neither which source pixel an impulse-filtered tile takes (the "
                         "scaled-down pixel's centre lies on the boundary between two source pixels) nor how the tile edges round off when the "
                         "size does not divide; tests/test_2d_behaviour_transform.py checks the layout (every tile is the same scaled picture), "
                         "the mirrors and the filters")
    return out
