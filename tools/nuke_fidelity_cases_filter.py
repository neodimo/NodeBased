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
# ----------------------------------------------------------------------------------------------------- Draw group (step C)
DRAW_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/draw_nodes/"
# The Draw pages call the ramp's point 0 end "by default, the bottom end" and its point 1 end "the top end", so y counts upwards from the
# bottom of the format; the pixel's colour is a ramp's value at its centre, a half pixel in (Nuke's coordinates are pixel centres, the
# Expression node's x and y among them). A reference here works in Nuke's space: x is the column, y the row counted from the bottom.


def _nuke_xy(shape):
    height, width = np.shape(shape)[:2]
    rows, cols = np.mgrid[0:height, 0:width]
    return cols.astype(np.float64), (height - 1 - rows).astype(np.float64)


def _ramp_t(shape, p0, p1):
    """Where each pixel's centre lies from point 0 (0) to point 1 (1), clamped."""
    xs, ys = _nuke_xy(shape)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return np.clip(((xs + 0.5 - p0[0]) * dx + (ys + 0.5 - p0[1]) * dy) / (dx * dx + dy * dy), 0.0, 1.0)


def _eased(t, kind):
    """The five ramp types. The page gives the words (smooth0 eases into point 0, smooth1 into point 1, smooth into both) and not the curves."""
    return {"linear": t, "smooth0": t * t, "smooth1": 1.0 - (1.0 - t) ** 2, "smooth": t * t * (3.0 - 2.0 * t)}[kind]


def _blend(c0, c1, w):
    return np.asarray(c0, np.float64)[None, None, :] * (1.0 - w[..., None]) + np.asarray(c1, np.float64)[None, None, :] * w[..., None]


def _over_input(fx, shape):
    """The shape drawn over the input (premultiplied over), the Draw nodes' compositing."""
    return shape + fx["A"].astype(np.float64) * (1.0 - shape[..., 3:4])


def _in_box(shape, x, y, w, h):
    """Pixels whose centre lies inside the area x, y, w, h (y from the bottom)."""
    xs, ys = _nuke_xy(shape)
    return (xs + 0.5 >= x) & (xs + 0.5 <= x + w) & (ys + 0.5 >= y) & (ys + 0.5 <= y + h)



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
    skewed = ((1.0, 1.0), (14.0, 2.6), (2.0, 11.0), (15.0, 9.7))   # no output pixel centre sits on a source pixel boundary (least margin 0.006 px)
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
    _draw_cases(Case, out)
    _roto_cases(Case, out)
    _rotopaint_cases(Case, out)
    _tracker_cases(Case, out)
    _dropshadow_cases(Case, out)
    _dirblur_cases(Case, out)
    _lightwrap_cases(Case, out)
    _bilateral_cases(Case, out)
    _vectorcornerpin_cases(Case, out)
    _warp_cases(Case, out)
    return out


def _draw_cases(Case, out):
    """Rectangle, Radial and Ramp (the Draw group), then Noise and Grid, Grain and Dither."""
    fmt = {"width": 16, "height": 12}

    def case(id, nuke, nuke_class, nuke_knobs, nb_kind, nb_params, expected, source, statement, unscored="", nb_inputs=("A",)):
        out.append(Case(id=id, nuke=nuke, nuke_class=nuke_class, nuke_knobs=nuke_knobs, nuke_inputs=("A",), nb_kind=nb_kind,
                        nb_params={**fmt, **nb_params} if nb_params is not None else None, nb_inputs=nb_inputs, expected=expected,
                        source=source, statement=statement, unscored=unscored))

    def drawn(make, only=None):
        """A reference: the shape `make(shape_like)` (premultiplied float64 RGBA) over the input, valid where `only(shape_like)` says."""
        def expected(fx):
            like = fx["A"].astype(np.float64)
            ref = _over_input(fx, make(like))
            valid = np.ones(like.shape, bool) if only is None else np.broadcast_to(only(like)[..., None], like.shape).copy()
            return ref, valid
        return expected

    area = {"box_x": 3.0, "box_y": 2.0, "box_width": 6.0, "box_height": 5.0}
    colour = (0.25, 0.5, 0.75, 0.8)
    paint = {"red": colour[0], "green": colour[1], "blue": colour[2], "alpha": colour[3]}

    def solid(coverage):
        return lambda like: np.concatenate([np.asarray(colour[:3]) * colour[3] * coverage(like)[..., None], (colour[3] * coverage(like))[..., None]], axis=-1)

    def rect_in(like):
        return _in_box(like, 3, 2, 6, 5).astype(np.float64)

    page = DRAW_GUIDE + "rectangle.html"
    # ----------------------------------------------------------------------------------------------------------- Rectangle
    case("rectangle_fills_the_area_over_the_input", "Rectangle", "Rectangle", (("area", "3 2 9 7"), ("color", "0.25 0.5 0.75 0.8")), "Rectangle",
         {**area, **paint, "softness": 0.0}, drawn(solid(rect_in)), "convention",
         f"Rectangle: 'Generates a rectangle in a solid color on top of the input image' ({page}); the area counts y from the bottom, "
         "as Nuke's coordinates do (the Ramp page names the point 0 end the bottom)")
    case("rectangle_invert_draws_everything_else", "Rectangle", "Rectangle", (("area", "3 2 9 7"), ("invert", True)), "Rectangle",
         {**area, **paint, "softness": 0.0, "invert": 1}, drawn(solid(lambda like: 1.0 - rect_in(like))), "guide",
         "Rectangle: invert 'inverts the new rectangle and non-rectangle areas'")
    case("rectangle_opacity", "Rectangle", "Rectangle", (("area", "3 2 9 7"), ("opacity", 0.5)), "Rectangle",
         {**area, **paint, "softness": 0.0, "opacity": 0.5}, drawn(solid(lambda like: 0.5 * rect_in(like))), "guide",
         "Rectangle: opacity 'sets the rectangle opacity where 0 is equal to transparent, and 1 is equal to opaque'")
    case("rectangle_replace_clears_the_input", "Rectangle", "Rectangle", (("area", "3 2 9 7"), ("replace", True)), "Rectangle",
         {**area, **paint, "softness": 0.0, "replace": 1},
         lambda fx: (solid(rect_in)(fx["A"].astype(np.float64)), np.ones(fx["A"].shape, bool)), "guide",
         "Rectangle: replace 'existing channels are cleared to black before drawing into them'")
    low, high = (0.1, 0.2, 0.3, 1.0), (0.9, 0.7, 0.5, 1.0)
    ramp_knobs = {"p0_x": 0.0, "p0_y": 1.0, "p1_x": 0.0, "p1_y": 10.0, "color0_red": low[0], "color0_green": low[1], "color0_blue": low[2],
                  "color0_alpha": 1.0, "red": high[0], "green": high[1], "blue": high[2], "alpha": 1.0}
    whole = {"box_x": 0.0, "box_y": 0.0, "box_width": 16.0, "box_height": 12.0, "softness": 0.0}

    def ramped(kind, only=None):
        def make(like):
            weight = _eased(_ramp_t(like, (0.0, 1.0), (0.0, 10.0)), kind)
            straight = _blend(low, high, weight)
            return np.concatenate([straight[..., :3] * straight[..., 3:4], straight[..., 3:4]], axis=-1)
        return drawn(make, only)

    def outside_ramp(like):
        xs, ys = _nuke_xy(like)
        return (ys + 0.5 <= 1.0) | (ys + 0.5 >= 10.0)

    def mid_row(like):
        xs, ys = _nuke_xy(like)
        return ys + 0.5 == 5.5
    case("rectangle_ramp_linear", "Rectangle", "Rectangle", (("ramp", "linear"), ("p0", "0 1"), ("p1", "0 10"), ("color0", "0.1 0.2 0.3 1"), ("color", "0.9 0.7 0.5 1")),
         "Rectangle", {**whole, **ramp_knobs, "ramp": "linear"}, ramped("linear"), "convention",
         f"Rectangle ramp linear: 'the ramp changes linearly from one color into another', color 0 at point 0 (the bottom) and color at point 1 ({page}); "
         "each pixel takes the ramp's value at its centre")
    for kind in ("smooth0", "smooth1", "smooth"):
        case(f"rectangle_ramp_{kind}_ends_are_the_two_colours", "Rectangle", "Rectangle",
             (("ramp", kind), ("p0", "0 1"), ("p1", "0 10")), "Rectangle", {**whole, **ramp_knobs, "ramp": kind}, ramped(kind, outside_ramp), "guide",
             f"Rectangle ramp {kind}: before point 0 the ramp is color 0 and past point 1 it is color, whatever the easing (compared outside the ramp only)")
    case("rectangle_ramp_smooth_is_even_at_the_middle", "Rectangle", "Rectangle", (("ramp", "smooth"), ("p0", "0 1"), ("p1", "0 10")), "Rectangle",
         {**whole, **ramp_knobs, "ramp": "smooth"}, ramped("smooth", mid_row), "convention",
         "Rectangle ramp smooth: 'the ramp color gradually eases into both ends', so halfway between the points the two colours weigh the same "
         "(the row of pixels whose centres lie on the midpoint)")
    for kind in ("smooth0", "smooth1"):
        case(f"rectangle_ramp_{kind}_curve", "Rectangle", "Rectangle", (("ramp", kind),), "Rectangle", {**whole, **ramp_knobs, "ramp": kind},
             ramped(kind), "convention", f"Rectangle ramp {kind}: the easing curve",
             unscored=f"the page says only which end {kind} eases into ('colors in the point {kind[-1]} end are spread wider') and gives no curve; "
                      "tests/test_2d_behaviour_draw.py checks the end it favours and that it keeps both end colours")
    case("rectangle_softness_falloff", "Rectangle", "Rectangle", (("area", "3 2 9 7"), ("softness", 2.0)), "Rectangle",
         {**area, **paint, "softness": 0.5}, drawn(solid(rect_in)), "convention", "Rectangle: the soft edge's width and curve",
         unscored="the page says only that softness 'sets the softness of the rectangle edges where 0 is a hard edge' (default 1); it gives neither "
                  "the unit nor the curve, so the soft edge is not scored; tests/test_2d_behaviour_draw.py checks that 0 is a hard edge, that "
                  "more softness widens the fade and that the fade is symmetric")

    # ------------------------------------------------------------------------------------------------------------- Radial
    page = DRAW_GUIDE + "radial.html"
    bx, by, bw, bh = 2.0, 1.0, 12.0, 10.0
    disc = {"box_x": bx, "box_y": by, "box_width": bw, "box_height": bh}

    def radial_distance(like):
        xs, ys = _nuke_xy(like)
        return np.sqrt(((xs + 0.5 - (bx + bw / 2.0)) / (bw / 2.0)) ** 2 + ((ys + 0.5 - (by + bh / 2.0)) / (bh / 2.0)) ** 2)

    case("radial_softness_zero_is_a_hard_ellipse", "Radial", "Radial", (("area", "2 1 14 11"), ("softness", 0.0), ("color", "0.25 0.5 0.75 0.8")), "Radial",
         {**disc, **paint, "softness": 0.0}, drawn(solid(lambda like: (radial_distance(like) < 1.0).astype(np.float64))), "guide",
         f"Radial: softness 'sets the softness of the radial ramp edges, where 0 is a hard edge'; the area holds the ramp and its center is the "
         f"ramp's origin ({page}): a filled ellipse that touches the area's four sides, y counted from the bottom")
    case("radial_invert", "Radial", "Radial", (("area", "2 1 14 11"), ("softness", 0.0), ("invert", True)), "Radial",
         {**disc, **paint, "softness": 0.0, "invert": 1}, drawn(solid(lambda like: (radial_distance(like) >= 1.0).astype(np.float64))), "guide",
         "Radial: invert 'inverts the radial ramp and non-radial ramp areas'")
    case("radial_opacity", "Radial", "Radial", (("area", "2 1 14 11"), ("softness", 0.0), ("opacity", 0.5)), "Radial",
         {**disc, **paint, "softness": 0.0, "opacity": 0.5}, drawn(solid(lambda like: 0.5 * (radial_distance(like) < 1.0))), "guide",
         "Radial: opacity 'sets the radial ramp opacity. 0 is equal to transparent and 1 is equal to opaque'")

    def radial_ramped(like):
        weight = _eased(_ramp_t(like, (0.0, 1.0), (0.0, 10.0)), "linear")
        straight = _blend(low, high, weight)
        alpha = straight[..., 3:4] * (radial_distance(like) < 1.0)[..., None]
        return np.concatenate([straight[..., :3] * alpha, alpha], axis=-1)
    case("radial_ramp_linear", "Radial", "Radial", (("ramp", "linear"), ("p0", "0 1"), ("p1", "0 10"), ("softness", 0.0)), "Radial",
         {**disc, **ramp_knobs, "softness": 0.0, "ramp": "linear"}, drawn(radial_ramped), "convention",
         "Radial ramp linear: a gradient from color 0 at point 0 (the bottom) to color at point 1 across the ellipse, centred sampling")
    case("radial_soft_falloff", "Radial", "Radial", (("area", "2 1 14 11"), ("softness", 1.0)), "Radial", {**disc, **paint, "softness": 1.0},
         drawn(solid(lambda like: np.clip(1.0 - radial_distance(like), 0.0, 1.0))), "convention", "Radial: the soft ramp's curve",
         unscored="the page says only that the area's centre is the originating point and that 0 is a hard edge; it gives no curve for the soft ramp, "
                  "so it is not scored; tests/test_2d_behaviour_draw.py checks that it is full at the centre, falls to nothing at the edge and never rises")
    case("radial_perceptually_linear", "Radial", "Radial", (("area", "2 1 14 11"), ("softness", 1.0), ("plinear", True)), "Radial",
         {**disc, **paint, "softness": 1.0, "plinear": 1}, drawn(solid(lambda like: np.clip(1.0 - radial_distance(like), 0.0, 1.0) ** 2.2)),
         "convention", "Radial: perceptually linear",
         unscored="the page says only that the ramp 'appears more linear to the naked eye'; the transfer curve is not given, so the pixels are not "
                  "scored; tests/test_2d_behaviour_draw.py checks that the switch changes the ramp, keeps its centre and edge and darkens the middle")

    # -------------------------------------------------------------------------------------------------------------- Ramp
    page = DRAW_GUIDE + "ramp.html"
    c0, c1 = (0.1, 0.2, 0.3, 1.0), (0.9, 0.7, 0.5, 1.0)
    ramp_params = {"p0_x": 0.0, "p0_y": 1.0, "p1_x": 0.0, "p1_y": 10.0, "color0_red": c0[0], "color0_green": c0[1], "color0_blue": c0[2],
                   "color0_alpha": 1.0, "color1_red": c1[0], "color1_green": c1[1], "color1_blue": c1[2], "color1_alpha": 1.0}

    def ramp_ref(kind, only=None):
        def make(like):
            straight = _blend(c0, c1, _eased(_ramp_t(like, (0.0, 1.0), (0.0, 10.0)), kind))
            return np.concatenate([straight[..., :3] * straight[..., 3:4], straight[..., 3:4]], axis=-1)
        return drawn(make, only)
    case("ramp_linear", "Ramp", "Ramp", (("type", "linear"), ("p0", "0 1"), ("p1", "0 10")), "Ramp", {**ramp_params, "ramp_type": "linear"},
         ramp_ref("linear"), "convention",
         f"Ramp linear: 'the ramp has a linear gradient' from point 0 (the bottom end) to point 1 (the top end) ({page}); each pixel takes the "
         "value at its centre, y counted from the bottom")
    for kind in ("smooth0", "smooth1", "smooth", "plinear"):
        case(f"ramp_{kind}_ends_are_the_two_colours", "Ramp", "Ramp", (("type", kind), ("p0", "0 1"), ("p1", "0 10")), "Ramp",
             {**ramp_params, "ramp_type": kind}, ramp_ref("linear" if kind == "plinear" else kind, outside_ramp), "guide",
             f"Ramp {kind}: before point 0 the ramp is the point 0 colour and past point 1 the point 1 colour, whatever the type (compared outside the ramp only)")
    case("ramp_smooth_is_even_at_the_middle", "Ramp", "Ramp", (("type", "smooth"), ("p0", "0 1"), ("p1", "0 10")), "Ramp",
         {**ramp_params, "ramp_type": "smooth"}, ramp_ref("smooth", mid_row), "convention",
         "Ramp smooth: 'the ramp gradually eases into both ends', so the middle weighs the two colours alike (the row on the midpoint)")
    for kind in ("smooth0", "smooth1", "plinear"):
        case(f"ramp_{kind}_curve", "Ramp", "Ramp", (("type", kind),), "Ramp", {**ramp_params, "ramp_type": kind}, ramp_ref(kind if kind != "plinear" else "linear"),
             "convention", f"Ramp {kind}: the curve between the points",
             unscored="the page names the type and says in words which end it eases into (or, for plinear, that it looks more linear to the eye) "
                      "but gives no curve; tests/test_2d_behaviour_draw.py checks the ends, the direction of the easing and that plinear differs from linear")
    _noise_grid_cases(Case, out, case)
    _defocus_cases(Case, out)


def _noise_grid_cases(Case, out, case):
    """Noise and Grid. The Noise page names the Perlin noise function and the combination of octaves but not the function's values,
    so its pixels are not scored (the properties the page states are tested in tests/test_2d_behaviour_draw.py)."""
    page = DRAW_GUIDE + "noise.html"
    noise_reason = ("the page says the node is 'based on the Perlin noise function' and how octaves, lacunarity, gain and gamma combine it, but not "
                    "the function's values (its gradient table and permutation are Foundry's), so the pixels cannot be written down; "
                    "tests/test_2d_behaviour_draw.py checks what the page states: gain 0 leaves one octave, gamma is the output power, translate, "
                    "rotate and scale move the pattern as the transform tab says, the x and y sizes stretch it, zoffset moves through it smoothly "
                    "and the same settings give the same pixels")

    def noise_case(id, nuke_knobs, params, statement):
        case(id, "Noise", "Noise", nuke_knobs, "Noise", params, lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)),
             "convention", statement, unscored=noise_reason)
    noise_case("noise_fbm_values", (("type", "fBm"),), {"type": "fBm", "size": 6.0, "octaves": 3},
               f"Noise fBm: 'runs noise() octave times at different scales to get random numbers between -1 and +1 and adds them up' ({page})")
    noise_case("noise_turbulence_values", (("type", "turbulence"),), {"type": "turbulence", "size": 6.0, "octaves": 3},
               "Noise turbulence: 'similar to fBm except it uses the absolute value of the noise function so the values it sums up are all between 0 and 1'")

    page = DRAW_GUIDE + "grid.html"
    grid = {"number_x": 4, "number_y": 3, "line_width": 1.0, "grid_offset_x": 0.0, "grid_offset_y": 0.0, "red": 1.0, "green": 0.5, "blue": 0.25, "alpha": 1.0}

    def square_centres(fx):
        like = fx["A"].astype(np.float64)
        xs, ys = _nuke_xy(like)
        return like, ((xs % 4 == 2) & (ys % 4 == 2))
    out.append(Case(id="grid_number_makes_squares_with_empty_middles", nuke="Grid", nuke_class="Grid", nuke_knobs=(("number", "4 3"), ("size", 1.0)),
                    nuke_inputs=("A",), nb_kind="Grid", nb_params={"width": 16, "height": 12, **grid}, nb_inputs=("A",),
                    expected=lambda fx: (square_centres(fx)[0], np.broadcast_to(square_centres(fx)[1][..., None], fx["A"].shape).copy()),
                    source="guide", statement=f"Grid: number 'sets the number of grid squares on the x and y axes' ({page}): on a 16 x 12 frame a number of 4 across and 3 up "
                    "makes squares four pixels across with lines one pixel thick, so the middle of each square (compared there only) is the input untouched"))
    out.append(Case(id="grid_line_positions", nuke="Grid", nuke_class="Grid", nuke_knobs=(("number", "4 3"), ("size", 1.0)), nuke_inputs=("A",),
                    nb_kind="Grid", nb_params={"width": 16, "height": 12, **grid}, nb_inputs=("A",),
                    expected=lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), source="convention",
                    statement="Grid: where the lines of a grid with a given number, size and translate fall",
                    unscored="the page gives the number of squares, the line thickness in pixels and a translate but not where in a square a line of a given "
                             "thickness sits (starting at the grid position, centred on it or ending at it) or what the last line does at the frame's edge; "
                             "tests/test_2d_behaviour_draw.py checks the spacing (the frame divided by the number), the thickness, that translate moves "
                             "the whole grid and that lines are drawn in the colour and the ramp"))
    _grain_dither_cases(Case, out, case)


def _grain_dither_cases(Case, out, case):
    """Grain and Dither. Both add noise whose values the pages do not publish; what they do state is scored where it fixes pixels."""
    page = DRAW_GUIDE + "grain.html"
    quiet = {"red_intensity": 0.0, "green_intensity": 0.0, "blue_intensity": 0.0}
    noisy = {"minimum_red": -1000.0, "minimum_green": -1000.0, "minimum_blue": -1000.0}

    def minimum_ref(fx):
        src = fx["A"].astype(np.float64)
        ref = src.copy()
        ref[..., :3] = np.maximum(src[..., :3], np.array([0.1, 0.2, 0.3]))
        return ref, np.ones(src.shape, bool)
    case("grain_minimum_rgb_is_the_floor", "Grain", "Grain", (("red_m", 0.0), ("green_m", 0.0), ("blue_m", 0.0), ("minimum", "0.1 0.2 0.3")), "Grain",
         {**quiet, "minimum_red": 0.1, "minimum_green": 0.2, "minimum_blue": 0.3}, minimum_ref, "convention",
         f"Grain: minimum rgb is 'the minimum output rgb values; if the calculation results in pixel values lower than these, these values are used instead' "
         f"({page}); with every intensity at 0 the calculation adds no grain, so the output is the input raised to the minimum")

    def through_alpha(invert):
        def expected(fx):
            src = fx["A"].astype(np.float64)
            alpha = src[..., 3]
            kept = (alpha == 1.0) if invert else (alpha == 0.0)
            return src, np.broadcast_to(kept[..., None], src.shape).copy()
        return expected
    case("grain_apply_only_through_alpha_leaves_empty_alpha_alone", "Grain", "Grain", (("maskgrain", True),), "Grain",
         {**noisy, "maskgrain": 1}, through_alpha(False), "guide",
         "Grain: 'apply only through alpha': 'the effect is limited to the non-black areas of the image's alpha channel', so where alpha is 0 the "
         "picture is unchanged (compared there only)")
    case("grain_invert_alpha_leaves_full_alpha_alone", "Grain", "Grain", (("maskgrain", True), ("invert_mask", True)), "Grain",
         {**noisy, "maskgrain": 1, "invert_mask": 1}, through_alpha(True), "guide",
         "Grain: 'invert alpha' 'inverts the use of the alpha channel, so that ... the effect is limited to the non-white areas of the alpha', so "
         "where alpha is 1 the picture is unchanged (compared there only)")
    case("grain_pattern", "Grain", "Grain", (("seed", 134),), "Grain", {**noisy}, lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)),
         "convention", "Grain: the grain pattern for a given size, irregularity and intensity",
         unscored="the page describes the controls in words (size shrinks or enlarges, irregularity is 'the random quality', intensity is the contrast of "
                  "the effect against the image) and gives no grain function or amplitude; tests/test_2d_behaviour_draw.py checks what follows from the "
                  "words: zero intensity adds nothing, the grain is zero-mean, a larger size gives a coarser grain, the pattern changes with the frame "
                  "and holds when the seed is minus the frame, black sets the grain at black, and each channel is controlled by its own knobs")

    page = DRAW_GUIDE + "dither.html"
    case("dither_amount_zero_adds_nothing", "Dither", "Dither", (("amount", 0.0),), "Dither", {"dither_amount": 0.0, "bits": 0, "channels": "rgba"},
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "guide",
         f"Dither: amount 'up to 0.5 of this amount is added or subtracted from each pixel' ({page}), so an amount of 0 changes nothing")
    case("dither_pattern", "Dither", "Dither", (("amount", 0.1), ("seed", 1)), "Dither", {"dither_amount": 0.1, "bits": 0, "channels": "rgba"},
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "convention", "Dither: the noise added to each pixel",
         unscored="the page says the noise is random, up to 0.5 of the amount either way, and gives neither the generator nor its distribution; "
                  "tests/test_2d_behaviour_draw.py checks the bound, that the mean is the input's, that monodither gives every channel the same "
                  "offset, and that the seed, the frame and static seed change the pattern as the page says")


FILTER_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/"


def ellipse_mean(a, rx, ry):
    """The plain average of the pixels whose centres lie inside the ellipse of semi-axes (rx, ry) around each pixel."""
    a = a.astype(np.float64)
    px, py = int(math.floor(rx + 1e-9)), int(math.floor(ry + 1e-9))
    padded = np.pad(a, ((py, py), (px, px), (0, 0)), mode="edge")
    h, w = a.shape[:2]
    total, count = np.zeros_like(a), 0
    for dy in range(-py, py + 1):
        for dx in range(-px, px + 1):
            if (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0 + 1e-9:
                total += padded[py + dy:py + dy + h, px + dx:px + dx + w]
                count += 1
    return total / count


def _defocus_cases(Case, out):
    """Defocus, ZDefocus and EdgeExtend (lane 9 step C). Defocus's disc is the page's "disc filter"; the pages for ZDefocus and EdgeExtend
    state what their controls do but not the blur or the colour fill, so those pixels are reported unscored with the reason."""
    def case(id, nuke, nuke_knobs, nb_params, expected, source, statement, unscored="", nb_inputs=("A",)):
        out.append(Case(id=id, nuke=nuke, nuke_class=nuke, nuke_knobs=nuke_knobs, nuke_inputs=("A",), nb_kind=nuke, nb_params=nb_params,
                        nb_inputs=nb_inputs, expected=expected, source=source, statement=statement, unscored=unscored))

    def disc(rx, ry, channels=(0, 1, 2, 3)):
        def expected(fx):
            src = fx["A"].astype(np.float64)
            ref = src.copy()
            blurred = ellipse_mean(src, rx, ry)
            for c in channels:
                ref[..., c] = blurred[..., c]
            valid = np.zeros(src.shape, bool)
            ry_, rx_ = int(math.ceil(ry)), int(math.ceil(rx))
            valid[ry_:src.shape[0] - ry_, rx_:src.shape[1] - rx_, :] = True
            return ref, valid
        return expected
    page = FILTER_GUIDE + "defocus.html"
    note = ("the page calls the node a 'disc filter' that 'simulates the defocusing effect of a circular lens' and gives defocus as 'the size of "
            "the defocus disc'; taken as the plain average of the pixels whose centres lie inside a disc of that radius (a convention: the page "
            "does not give the weights at the disc's rim). Edge pixels are not compared")
    case("defocus_disc_average", "Defocus", (("defocus", 2.0),), {"defocus": 2.0, "aspect": 1.0, "channels": "rgba"}, disc(2.0, 2.0),
         "convention", f"Defocus at size 2: {note}")
    case("defocus_aspect_ratio_widens_the_disc", "Defocus", (("defocus", 2.0), ("ratio", 2.0)), {"defocus": 2.0, "aspect": 2.0, "channels": "rgba"},
         disc(4.0, 2.0), "guide", f"Defocus: aspect ratio 'adjusts the width of the defocus disc without altering its height' ({page}): "
         "at size 2 and ratio 2 the disc is 4 pixels in radius across and 2 up (compared where the whole disc is in the frame)")
    case("defocus_scaling_scales_both_axes", "Defocus", (("defocus", 1.0), ("scale", 3.0)), {"defocus": 1.0, "aspect": 1.0, "scale": 3.0, "channels": "rgba"},
         disc(3.0, 3.0), "guide", f"Defocus: scaling 'scales the defocus disc simultaneously on the x and y axes' ({page}): size 1 at scale 3 is a disc of radius 3")
    case("defocus_scaling_split_into_width_and_height", "Defocus", (("defocus", 1.0), ("scale", "3 2")),
         {"defocus": 1.0, "aspect": 1.0, "scale": [3.0, 2.0], "channels": "rgba"}, disc(3.0, 2.0), "guide",
         "Defocus: scaling can be 'split into separate width and height controls': size 1 at scale 3 wide and 2 high is a disc 3 across and 2 up")
    case("defocus_channels_limit_the_effect", "Defocus", (("defocus", 2.0), ("channels", "rgb")), {"defocus": 2.0, "aspect": 1.0, "channels": "rgb"},
         disc(2.0, 2.0, channels=(0, 1, 2)), "guide", f"Defocus: 'the defocus effect is only applied to these channels' ({page}): with rgb the alpha is the input's")
    case("defocus_quality_and_method", "Defocus", (("quality", 20), ("method", "fullprecision")), None,
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "guide",
         "Defocus: quality 'sets the interpolation used to smooth out the discs'; method accelerated or fullprecision",
         unscored="NodeBased has no quality or method knob: its disc is always summed exactly, which is what fullprecision means; the page gives no "
                  "interpolation to compare, so a quality below the default cannot be scored")

    page = FILTER_GUIDE + "zdefocus.html"
    depth = ("A", "mask")

    def same_as_input(fx):
        return fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)
    direct = {"zdefocus_math": "direct", "legacy_zdefocus": 0, "size": 25.0, "max_size": 0.0, "bokeh_shape": "disc"}
    case("zdefocus_direct_is_clamped_to_maximum", "ZDefocus", (("math", "direct"), ("size", 25.0), ("max_size", 0.0)), direct, same_as_input, "guide",
         f"ZDefocus: with math direct 'the blur size will be 0.5 times the value of the size control (unless this is bigger than maximum, in which "
         f"case it will be clamped to maximum)' ({page}): a maximum of 0 allows no blur whatever the depth", nb_inputs=depth)
    case("zdefocus_direct_size_zero_is_no_blur", "ZDefocus", (("math", "direct"), ("size", 0.0)), {**direct, "size": 0.0, "max_size": 50.0}, same_as_input,
         "guide", "ZDefocus: with math direct the blur size is the depth times the size control, so a size of 0 blurs nothing", nb_inputs=depth)
    case("zdefocus_blur_by_distance", "ZDefocus", (("math", "depth"), ("center", 1.0), ("dof", 0.0)),
         {"zdefocus_math": "depth", "legacy_zdefocus": 0, "focal_plane": 1.0, "depth_of_field": 1.0}, same_as_input, "convention",
         "ZDefocus: the blur at a given distance from the focus plane, depth of field and size", nb_inputs=depth,
         unscored="the page gives the depth-to-distance conversions and says the blur is 'the size of the blur at infinite depth' and grows nearer "
                  "the camera, but not the formula between a distance, the focus plane, the depth of field and the size, nor the layer spacing "
                  "(automatic layer spacing, depth layers, layer curve), the blend between layers, fill foreground or the filter shapes "
                  "(disc, bladed, image, with shape, roundness, inner size and the rest); the properties it does state are tested in "
                  "tests/test_2d_behaviour_draw.py: direct blur grows with depth to the maximum, -direct reverses the order, a larger maximum "
                  "allows more blur, the focus plane stays sharp, and the filter choices change the pixels")

    page = FILTER_GUIDE + "edgeextend.html"
    straight = {"legacy_edge_extend": 0, "source_premultiplied": 0, "premultiply": 0, "erode": 0.0, "extend_size": 2.0, "extend_threshold": 0.5}

    def solid(fx, scale_by_alpha):
        src = fx["A"].astype(np.float64)
        ref = src.copy()
        if scale_by_alpha:
            ref[..., :3] = src[..., :3] * src[..., 3:4]
        known = src[..., 3] >= 0.5
        return ref, np.broadcast_to(known[..., None], src.shape).copy()
    case("edgeextend_solid_matte_keeps_its_colour", "EdgeExtend", (("premultiply", False),), straight, lambda fx: solid(fx, False), "convention",
         f"EdgeExtend corrects colour 'at the edge of soft mattes' ({page}): where the source alpha is solid (at or above the threshold) the colour is the "
         "source's own (compared there only), with the source straight and premultiply off")
    case("edgeextend_premultiply_multiplies_by_alpha", "EdgeExtend", (("premultiply", True),), {**straight, "premultiply": 1}, lambda fx: solid(fx, True),
         "guide", f"EdgeExtend: premultiply 'when enabled, EdgeExtend premultiplies the output image' ({page}): the output colour is the corrected colour "
         "times the alpha (compared where the colour is the source's own)")
    case("edgeextend_edge_colour", "EdgeExtend", (("erode", 0.0), ("detailAmount", 1.0)), straight, lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)),
         "convention", "EdgeExtend: the colour that fills the soft edge of the matte, and what detail amount, the matte choices and the edge mask do",
         unscored="the page says the node 'corrects unpremultiplied foreground color at the edge of soft mattes by eroding or dilating the sample "
                  "region, pulling pixels from deeper inside or outside your matte' with an algorithm 'independent of the size of the fill regions', "
                  "but gives neither that algorithm nor what detail amount recovers; the edge fill here is a ring-by-ring average, an approximation. "
                  "tests/test_2d_behaviour_draw.py checks what the page states: positive erode takes colour from deeper inside the matte and "
                  "negative erode from outside it, source is premultiplied unpremultiplies the source first, and premultiply multiplies the output")


# ------------------------------------------------------------------------------------------------ Roto (lane 9, step E)
ROTO_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/draw_nodes/roto.html"
ROTO_PLATE = (0.2, 0.4, 0.6, 1.0)            # a flat background, straight RGBA with alpha 1, so its stored colour equals these numbers
_KAPPA = 0.5522847498                        # the Bezier handle length, as a fraction of the radius, that draws a quarter circle


def roto_point(x, y, ix=0.0, iy=0.0, ox=0.0, oy=0.0):
    return {"x": float(x), "y": float(y), "in_x": ix, "in_y": iy, "out_x": ox, "out_y": oy}


def roto_box(x0, y0, x1, y1, **extra):
    """A closed shape with four corner points, as the shape list stores it (edge feather, the model the viewer's new shapes use)."""
    shape = {"name": "box", "mode": "union", "opacity": 1.0, "feather": 0.0, "feather_model": "edge",
             "points": [roto_point(x0, y0), roto_point(x1, y0), roto_point(x1, y1), roto_point(x0, y1)]}
    shape.update(extra)
    return shape


def roto_circle(cx, cy, r, **extra):
    """Four Bezier points with quarter-circle handles (offsets from the point, as the shape list stores them)."""
    k = _KAPPA * r
    pts = [roto_point(cx + r, cy, 0, -k, 0, k), roto_point(cx, cy + r, k, 0, -k, 0),
           roto_point(cx - r, cy, 0, k, 0, -k), roto_point(cx, cy - r, -k, 0, k, 0)]
    shape = {"name": "ellipse", "mode": "union", "opacity": 1.0, "feather": 0.0, "feather_model": "edge", "points": pts}
    shape.update(extra)
    return shape


def _roto_cases(Case, out):
    """Roto: what the shapes in the node's payload draw (the guide's Roto page: output, premultiply, replace, opacity, feather, invert,
    the shape list and the Transform tab). Shapes are written in the case's `_shapes`, a flat plate in `_bg`; see
    nuke_fidelity_cases._run_payload_document. Pixel i spans [i, i + 1) in x and in y (a pixel's centre is at i + 0.5), the form the
    NDK's pixel coordinates use; a pixel the outline only partly covers is not scored unless a case says its rule."""
    def case(id, nuke_knobs, nb_params, expected, source, statement, unscored=""):
        out.append(Case(id=id, nuke="Roto", nuke_class="Roto", nuke_knobs=nuke_knobs, nuke_inputs=("bg",), nb_kind="Roto",
                        nb_params=nb_params, nb_inputs=(), expected=expected, source=source, statement=statement, unscored=unscored))

    yy, xx = np.mgrid[0:12, 0:16]
    cx, cy = xx + 0.5, yy + 0.5                      # pixel centres

    def alpha_of(cover, valid=None, bg=None):
        """The reference for a node that draws its matte into alpha only: alpha = `cover`, compared in the alpha channel; the other
        channels are compared with the plate (when there is one) or left out."""
        def expected(fx):
            ref = np.zeros((12, 16, 4))
            valid_all = np.zeros((12, 16, 4), bool)
            ref[..., 3] = cover
            valid_all[..., 3] = True if valid is None else valid
            if bg is not None:
                ref[..., :3] = np.asarray(bg[:3])
                valid_all[..., :3] = True
            return ref, valid_all
        return expected

    def inside_box(x0, y0, x1, y1):
        return ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).astype(np.float64)

    sq = (4, 3, 10, 8)                               # pixels 4..9 across and 3..7 down: every edge on a pixel boundary
    case("roto_bezier_square_fill", (("output", "alpha"),), {"_shapes": [roto_box(*sq)]}, alpha_of(inside_box(*sq)), "convention",
         f"Roto ({ROTO_GUIDE}): a closed Bezier square renders 1 into the output channel inside and 0 outside; the outline lies on pixel "
         "boundaries (pixel i spans [i, i + 1)), so no pixel is partly covered and none is left out")
    case("roto_bezier_ellipse_fill", (("output", "alpha"),), {"_shapes": [roto_circle(8, 6, 4)]},
         lambda fx: (lambda dist: alpha_of((dist < 2.8).astype(np.float64), valid=(dist < 2.8) | (dist > 5.2))(fx))(np.hypot(cx - 8, cy - 6)),
         "convention", "Roto: a closed Bezier circle of radius 4 fills to 1 inside and 0 outside; the pixels within 1.2 of the outline (partly covered, "
         "and the Bezier circle is within 0.03% of a true one) are not scored")
    case("roto_edge_pixel_coverage", (("output", "alpha"),), {"_shapes": [roto_box(6.5, 3, 10, 8)]},
         alpha_of(np.where((xx == 6), 0.5, inside_box(7, 3, 10, 8)), valid=(xx != 6)), "convention",
         "Roto: a pixel the outline cuts in half; every other pixel of the square is scored, this one is not",
         unscored="the guide does not say how Roto antialiases a pixel an outline cuts through (the covered fraction of the pixel is the "
                  "usual choice and what NodeBased draws: this pixel reads 0.5); interior and exterior pixels are scored by the cases above")

    fb = (5, 4, 9, 8)                                # the feather box; a feather of 2 reaches pixels 3..4 on the left and 9..10 on the right

    def feather_cover(width, curve):
        dx = np.maximum(np.maximum(fb[0] - cx, cx - fb[2]), 0.0)
        dy = np.maximum(np.maximum(fb[1] - cy, cy - fb[3]), 0.0)
        beyond = np.hypot(dx, dy)                    # distance outside the rectangle
        cover = np.where(beyond > 0, curve(np.clip(1.0 - beyond / width, 0.0, 1.0)), 1.0)
        straight = (np.abs(dx) < 1e-9) | (np.abs(dy) < 1e-9)       # the corners' feather is the outline's own business: not scored
        return cover, straight
    cov, straight = feather_cover(2.0, lambda t: t)
    case("roto_feather_widens_outward_linearly", (("feather", 2.0), ("feather_type", "linear")), {"_shapes": [roto_box(*fb, feather=2.0)]},
         alpha_of(cov, valid=straight), "guide",
         f"Roto feather ({ROTO_GUIDE}): 'with positive feather values the feather effect is outward'; feather type linear: 'the falloff changes "
         "linearly from the shape edge to the feather edge': the pixel whose centre is d outside the edge reads 1 - d / feather (0.75, 0.25, 0 for "
         "a feather of 2); the corners are not scored")
    inner = np.minimum(np.minimum(cx - 4, 12 - cx), np.minimum(cy - 3, 9 - cy))          # distance inside the edge (negative outside)
    case("roto_negative_feather_falls_inward", (("feather", -2.0),), {"_shapes": [roto_box(4, 3, 12, 9, feather=-2.0)]},
         alpha_of(np.clip(inner / 2.0, 0.0, 1.0)), "guide",
         "Roto feather: 'if your feather values are negative, the feather effect is inward': inside the edge by d the pixel reads d / |feather| "
         "up to 1 (0.25, 0.75 for a feather of -2); outside the edge 0")
    case("roto_feather_smooth_is_half_way_at_half_the_feather", (("feather", 4.0), ("feather_type", "smooth")),
         {"_shapes": [roto_box(6.5, 4, 10.5, 8, feather=4.0, feather_falloff="smooth")]},
         lambda fx: alpha_of(np.full((12, 16), 0.5), valid=((xx == 4) | (xx == 12)) & (yy >= 4) & (yy < 8))(fx), "guide",
         "Roto feather type smooth: 'the center point of the feather falloff lies halfway between the shape edge and the feather edge': half the "
         "feather out (2 of 4) the matte reads 0.5 (compared at those pixels only)")
    case("roto_feather_smooth_curve", (("feather_type", "smooth0"),), {"_shapes": [roto_box(*fb, feather=2.0)]}, alpha_of(np.zeros((12, 16)), valid=np.zeros((12, 16), bool)),
         "guide", "Roto feather types smooth0 and smooth1: where the falloff's centre lies, and the curve through it",
         unscored="the guide says where each curve's centre lies ('closer to the shape edge', 'closer to the feather edge', 'halfway') but gives "
                  "no curve; only the halfway property of smooth is scored, by the case above. NodeBased draws smooth as a smoothstep and has no "
                  "smooth0 or smooth1 (listed in the step E report)")

    case("roto_shape_opacity_scales_fill", (("opacity", 0.5),), {"_shapes": [roto_box(*sq, opacity=0.5)]}, alpha_of(0.5 * inside_box(*sq)), "guide",
         f"Roto opacity ({ROTO_GUIDE}): 'the opacity of a roto shape in the shape list; 1 is totally opaque and 0 is transparent': an opacity "
         "of 0.5 draws half the matte")
    case("roto_node_opacity_scales_fill", (("opacity", 0.25),), {"_shapes": [roto_box(*sq)], "opacity": 0.25}, alpha_of(0.25 * inside_box(*sq)), "guide",
         "Roto opacity: the node's own opacity knob scales the whole matte, 1 opaque and 0 transparent")
    case("roto_shape_invert_complements", (("inverted", True),), {"_shapes": [roto_box(*sq, invert=True)]}, alpha_of(1.0 - inside_box(*sq)), "guide",
         f"Roto shape list ({ROTO_GUIDE}): 'invert the shape or group': the matte inside the shape is 0 and outside it 1")
    other = (7, 5, 13, 10)
    both = np.clip(inside_box(*sq) * 0.5 + inside_box(*other) * 0.5 - inside_box(*sq) * inside_box(*other) * 0.25, 0, 1)
    case("roto_overlap_blends_over", (("blending_mode", "over"),),
         {"_shapes": [roto_box(*sq, opacity=0.5), roto_box(*other, opacity=0.5, name="second")]}, alpha_of(both), "convention",
         "Roto blending mode over (the default of the shape list): the second shape is composited over the first, alpha = a + b (1 - a), so "
         "where two half-opaque shapes overlap the matte reads 0.75; the formula is Merge over's (guide: A + B(1 - a))")
    plus = np.clip(inside_box(*sq) * 0.5 + inside_box(*other) * 0.5, 0, 1)
    case("roto_overlap_blends_plus", (("blending_mode", "plus"),),
         {"_shapes": [roto_box(*sq, opacity=0.5), roto_box(*other, opacity=0.5, name="second", blend_mode="plus")]}, alpha_of(plus), "convention",
         "Roto blending mode plus: the second shape's matte is added to what is below, clamped at 1 (Merge plus: A + B)")
    # The blending modes whose formulas the guide's list states (A the shape, B what is below). The second shape covers its pixels fully
    # (opacity 1), so there its matte is op(1, B); outside it the matte below is left alone. B is the first shape at half opacity.
    below = 0.5 * inside_box(*sq)
    on_top = inside_box(*other)
    for mode, statement, op in (
            ("minus", "minus (A-B): subtracts B from A", lambda b: 1.0 - b),
            ("from", "from (B-A): subtracts A from B", lambda b: np.maximum(b - 1.0, 0.0)),
            ("difference", "difference (abs(A-B))", lambda b: np.abs(1.0 - b)),
            ("exclusion", "exclusion (A+B-2AB)", lambda b: 1.0 - b),
            ("max", "max (max(A,B))", lambda b: np.ones_like(b)),
            ("min", "min (min(A,B))", lambda b: b),
            ("screen", "screen (A+B-AB)", lambda b: np.ones_like(b))):
        case(f"roto_overlap_blends_{mode}", (("blending_mode", mode),),
             {"_shapes": [roto_box(*sq, opacity=0.5), roto_box(*other, name="second", blend_mode=mode)]},
             alpha_of(below * (1.0 - on_top) + on_top * op(below)), "guide",
             f"Roto blending mode {statement} ({ROTO_GUIDE}), A the shape and B what is below: where the second shape covers a pixel fully its matte "
             f"is the formula with A = 1 and B = the half-opaque first shape (0.5 where they overlap, 0 elsewhere); where it covers nothing the matte below is unchanged")
    case("roto_overlap_blends_multiply", (("blending_mode", "multiply"),),
         {"_shapes": [roto_box(*sq, opacity=0.5), roto_box(*other, name="second", blend_mode="multiply")]},
         alpha_of(np.zeros((12, 16)), valid=np.zeros((12, 16), bool)), "guide", "Roto blending mode multiply (AB)",
         unscored="the guide's formula is AB; read with the shape's matte as A = 1 it leaves the matte below unchanged, while NodeBased composites a multiply "
         "shape over what is below (alpha = a + b (1 - a)) and keeps its colour channels multiplied; the colour channels are not scored by these matte cases")

    shifted = inside_box(sq[0] + 2, sq[1] + 1, sq[2] + 2, sq[3] + 1)
    case("roto_transform_translate_moves_the_shape", (("translate", [2, 1]),),
         {"_shapes": [roto_box(*sq, transform={"translate_x": 2.0, "translate_y": 1.0})]}, alpha_of(shifted), "guide",
         f"Roto Transform tab ({ROTO_GUIDE}): translate 'moves the selected shape on the x and y axes': 2 right and 1 down")
    case("roto_visible_off_draws_nothing", (("visible", False),), {"_shapes": [roto_box(*sq, visible=False)]}, alpha_of(np.zeros((12, 16))), "guide",
         "Roto shape list: 'whether the shape is visible and rendered or not': a shape that is not visible draws nothing")

    def convex_cover(polygon):
        """Exact-ish coverage of a convex polygon (clockwise or not) by 32 x 32 sub-pixel samples, and the mask of pixels at least one pixel away from every pixel an
        edge cuts through (those are not scored)."""
        n = 32
        sub_y, sub_x = np.mgrid[0:12 * n, 0:16 * n]
        px, py = (sub_x + 0.5) / n, (sub_y + 0.5) / n
        sign = None
        inside = np.ones(px.shape, bool)
        for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]):
            cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
            if sign is None:
                sign = 1.0 if ((x1 - x0) * (polygon[2][1] - y0) - (y1 - y0) * (polygon[2][0] - x0)) > 0 else -1.0
            inside &= cross * sign >= 0
        cover = inside.reshape(12, n, 16, n).mean(axis=(1, 3))
        partial = ((cover > 0.0) & (cover < 1.0)).astype(np.uint8)
        near = np.zeros_like(partial)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= np.roll(np.roll(partial, dy, axis=0), dx, axis=1)
        return cover, near == 0

    def moved(points, matrix, centre, translate=(0.0, 0.0)):
        cx0, cy0 = centre
        return [(cx0 + matrix[0][0] * (x - cx0) + matrix[0][1] * (y - cy0) + translate[0],
                 cy0 + matrix[1][0] * (x - cx0) + matrix[1][1] * (y - cy0) + translate[1]) for x, y in points]

    def tan_deg(deg):
        return float(np.tan(np.radians(deg)))
    wide = (4, 4, 12, 8)                           # 8 wide, 4 tall, centre (8, 6)
    case("roto_transform_rotate_quarter_turn_swaps_width_and_height", (("rotate", 90),),
         {"_shapes": [roto_box(*wide, transform={"rotate": 90.0, "center_x": 8.0, "center_y": 6.0})]}, alpha_of(inside_box(6, 2, 10, 10)), "guide",
         f"Roto Transform tab ({ROTO_GUIDE}): rotate 'spins the selected shape or group around a pivot point' (center x,y): a quarter turn about the "
         "box's centre turns the 8 wide, 4 tall box into a 4 wide, 8 tall one (the guide does not say which way a positive angle turns, which a "
         "quarter turn of a box about its own centre does not depend on)")
    case("roto_transform_rotate_half_turn_about_a_pivot", (("rotate", 180), ("center", [6, 4])),
         {"_shapes": [roto_box(5, 3, 9, 6, transform={"rotate": 180.0, "center_x": 6.0, "center_y": 4.0})]}, alpha_of(inside_box(3, 2, 7, 5)), "guide",
         "Roto Transform tab: rotate spins the shape about the pivot set by center x,y; a half turn maps each point p to 2c - p whichever way "
         "the angle turns, so the box (5,3)-(9,6) about (6,4) lands on (3,2)-(7,5)")
    case("roto_transform_scale_resizes_about_the_centre", (("scale", 2.0),),
         {"_shapes": [roto_box(6, 4, 10, 8, transform={"scale_x": 2.0, "scale_y": 2.0, "center_x": 8.0, "center_y": 6.0})]},
         alpha_of(inside_box(4, 2, 12, 10)), "guide",
         "Roto Transform tab: scale 'resizes the selected shape or group'; about its centre (8, 6) a scale of 2 doubles the 4 x 4 box to 8 x 8")
    case("roto_transform_center_alone_does_not_move_the_shape", (("center", [12, 9]),),
         {"_shapes": [roto_box(*sq, transform={"center_x": 12.0, "center_y": 9.0})]}, alpha_of(inside_box(*sq)), "guide",
         "Roto Transform tab: center x,y 'adjusts the center of rotation and skew ... this control does not translate roto shapes'")
    box_points = [(5.0, 3.0), (9.0, 3.0), (9.0, 8.0), (5.0, 8.0)]
    centre = (7.0, 5.5)
    kx = [[1.0, tan_deg(45)], [0.0, 1.0]]            # skew X shears x by the height above the centre
    ky = [[1.0, 0.0], [tan_deg(30), 1.0]]
    def mul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    skew_cover, skew_valid = convex_cover(moved(box_points, kx, centre))
    case("roto_transform_skew_x_shears_the_shape", (("skewX", 45),),
         {"_shapes": [roto_box(5, 3, 9, 8, transform={"skew_x": 45.0, "center_x": 7.0, "center_y": 5.5})]}, alpha_of(skew_cover, skew_valid), "convention",
         "Roto Transform tab: skew X (degrees) shears the shape about its centre; the x of each point moves by tan(skew X) times its distance from "
         "the centre's y, the usual 2D shear (the guide gives no formula); only the pixels wholly inside or outside the sheared quadrilateral are scored")
    for order, matrix in (("XY", mul(ky, kx)), ("YX", mul(kx, ky))):
        cover, valid = convex_cover(moved(box_points, matrix, centre))
        case(f"roto_transform_skew_order_{order.lower()}", (("skewX", 45), ("skewY", 30), ("skew_order", order)),
             {"_shapes": [roto_box(5, 3, 9, 8, transform={"skew_x": 45.0, "skew_y": 30.0, "skew_order": order, "center_x": 7.0, "center_y": 5.5})]},
             alpha_of(cover, valid), "guide",
             f"Roto Transform tab: skew order {order} ('{'Skew X is applied before skew Y' if order == 'XY' else 'Skew Y is applied before skew X'}'): "
             "the two shears are composed in that order about the centre; only the pixels wholly inside or outside the sheared quadrilateral are scored")

    # Open splines: a stroke of the stated overall width, rounded or square at each end.
    line = [roto_point(3, 6), roto_point(13, 6)]
    def stroke(**extra):
        return {"name": "line", "mode": "union", "opacity": 1.0, "feather": 0.0, "feather_model": "edge", "closed": False, "stroke_width": 4.0,
                "points": line, **extra}
    bar = inside_box(5, 4, 11, 8)                    # rows 4..7, columns 5..10: inside the stroke whatever the ends
    band_valid = np.zeros((12, 16), bool)
    band_valid[:, 5:11] = True                       # every row of these columns
    band_valid[(yy < 4) | (yy >= 8)] = True          # and, outside the band, every column (a rounded end reaches only to row 4 and row 8)
    case("roto_open_spline_width_is_the_stroke_width", (("openspline_width", 4),),
         {"_shapes": [stroke()]}, alpha_of(bar, band_valid), "guide",
         f"Roto Shape tab ({ROTO_GUIDE}): overall width 'sets the overall spline width': a horizontal open spline of width 4 covers four rows "
         "(rows 4 to 7 about the line y = 6) between its ends and none above or below them")
    odd = np.zeros((12, 16))
    odd[4, :] = 0.5
    odd[5:7, :] = 1.0
    odd[7, :] = 0.5
    case("roto_open_spline_odd_width_covers_the_stated_width", (("openspline_width", 3),),
         {"_shapes": [stroke(stroke_width=3.0)]}, alpha_of(odd, band_valid), "guide",
         "Roto Shape tab: overall width 'sets the overall spline width': a width of 3 about the line y = 6 spans y = 4.5 to 7.5, so the rows above "
         "and below the two full ones are half covered (three rows' worth in all, not four)")
    square_all = inside_box(1, 4, 15, 8)
    case("roto_open_spline_square_ends_extend_by_half_the_width", (("openspline_start_end_type", "square"), ("openspline_last_end_type", "square")),
         {"_shapes": [stroke(cap_start="square", cap_end="square")]}, alpha_of(square_all), "guide",
         "Roto Shape tab: start type and end type 'square - the starting (finishing) point of the spline has a square end': the stroke is a "
         "rectangle that reaches half the width (2 pixels) past each end point, columns 1 to 14")
    rounded_all = np.zeros((12, 16))
    rounded_valid = np.ones((12, 16), bool)
    rounded_valid[(xx < 2) | (xx >= 14)] = False     # the rounded ends' curved edges
    rounded_valid[:, :] &= ~(((xx >= 1) & (xx < 3) | (xx >= 13) & (xx < 15)) & (yy >= 4) & (yy < 8))
    rounded_all[:] = inside_box(3, 4, 13, 8)
    case("roto_open_spline_rounded_ends_stop_at_the_end_points", (("openspline_start_end_type", "rounded"), ("openspline_last_end_type", "rounded")),
         {"_shapes": [stroke(cap_start="rounded", cap_end="rounded")]}, alpha_of(rounded_all, rounded_valid), "guide",
         "Roto Shape tab: 'rounded - the starting (finishing) point of the spline has a rounded end': the stroke ends in a half disc of radius "
         "half the width centred on each end point, so columns 3 to 12 are full over the four rows and the ends' curved edges (columns 1, 2, 13 and 14) are not scored")

    # Output channels, premultiply and replace act on a wired background.
    plate_a = (0.2, 0.4, 0.6, 0.8)
    def colour_of_plate(fx):
        ref = np.zeros((12, 16, 4))
        valid = np.zeros((12, 16, 4), bool)
        ref[..., :3] = np.asarray(ROTO_PLATE[:3])
        valid[..., :3] = True
        return ref, valid
    case("roto_output_alpha_keeps_the_colour", (("output", "alpha"),), {"_shapes": [roto_box(*sq)], "_bg": ROTO_PLATE}, colour_of_plate, "guide",
         f"Roto output ({ROTO_GUIDE}): 'the roto shapes are rendered into these output channels' (alpha is the default): with the plate wired the "
         "colour channels stay the plate's, inside the shape and out (the alpha channel is not compared)")
    case("roto_output_rgb_keeps_the_alpha", (("output", "rgb"),), {"_shapes": [roto_box(*sq)], "_bg": plate_a, "output": "rgb"},
         alpha_of(np.full((12, 16), 0.8)), "guide",
         "Roto output rgb: the shapes are rendered into the colour channels only, so the plate's alpha (0.8) is left as it was")
    case("roto_existing_alpha_kept_outside_the_shapes", (("output", "alpha"), ("replace", False)), {"_shapes": [roto_box(*sq)], "_bg": plate_a},
         alpha_of(np.where(inside_box(*sq) > 0, 1.0, 0.8)), "guide",
         f"Roto replace ({ROTO_GUIDE}): 'when enabled, existing channels are cleared to black before drawing into them ... the incoming image "
         "already has an alpha channel which you want to throw away': with replace off the incoming alpha stays where the shapes do not reach "
         "(0.8) and a fully opaque shape over it reads 1")
    case("roto_replace_clears_the_existing_alpha", (("replace", True),), {"_shapes": [roto_box(*sq)], "_bg": plate_a, "replace": 1},
         alpha_of(inside_box(*sq)), "guide",
         "Roto replace on: the incoming alpha is cleared before drawing, so the matte is the shapes' alone (0 outside them)")
    cov_f, straight_f = feather_cover(2.0, lambda t: t)
    feathered = [roto_box(*fb, feather=2.0)]

    def premult_rgb(fx):
        ref = np.zeros((12, 16, 4))
        valid = np.zeros((12, 16, 4), bool)
        ref[..., :3] = np.asarray(ROTO_PLATE[:3]) * cov_f[..., None]
        valid[..., :3] = straight_f[..., None]
        return ref, valid
    case("roto_premultiply_rgb_multiplies_by_the_matte", (("premultiply", "rgb"),),
         {"_shapes": feathered, "_bg": ROTO_PLATE, "premultiply_mode": "rgb"}, premult_rgb, "guide",
         f"Roto premultiply ({ROTO_GUIDE}): 'premultiply multiplies the chosen input channels with a mask representing the roto shapes ... where "
         "there are no roto shapes the input channels are set to black; where the roto shapes are opaque the input channels keep their full value': "
         "rgb is the plate times the matte (the alpha channel is not compared)")

    def premult_rgba(fx):
        ref = np.zeros((12, 16, 4))
        valid = np.zeros((12, 16, 4), bool)
        ref[..., 3] = cov_f * cov_f
        valid[..., 3] = straight_f
        return ref, valid
    case("roto_premultiply_rgba_squares_the_alpha", (("premultiply", "rgba"),),
         {"_shapes": feathered, "_bg": (0.0, 0.0, 0.0, 0.0), "premultiply_mode": "rgba"}, premult_rgba, "guide",
         "Roto premultiply rgba: 'selecting rgba premultiplies the alpha against itself (a*a)': on an empty plate the alpha holds the matte, "
         "so it ends as the matte squared (0.5625 where the matte reads 0.75); the colour channels are not compared")


# ------------------------------------------------------------------------------------------------ RotoPaint (lane 9, step E)
ROTOPAINT_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/draw_nodes/rotopaint.html"


def paint_stroke(tool="paint", x=8.0, y=6.0, **extra):
    """One single-dab stroke as the node's payload stores it: size 6, hard (hardness 1), red, full opacity; pressure 1."""
    item = {"kind": "stroke", "name": "stroke1", "points": [{"x": float(x), "y": float(y), "pressure": 1.0}],
            "brush": {"size": 6.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2}, "tool": tool,
            "lifetime": {"mode": "all"}, "color": [1.0, 0.0, 0.0, 1.0], "source_offset": [0.0, 0.0], "source_frame": "relative",
            "opacity": 1.0, "blend": "over", "visible": True, "follow_track": None, "patch_blend": 0.0}
    item.update(extra)
    return item


def _plate_array(spec):
    """The picture of a plate spec ({"type", "params"} or a flat RGBA tuple) at the fixture's size, from the evaluator (the plate is not under test)."""
    from nodebased.core import Dispatcher, empty_document
    from nodebased.imaging import Evaluator
    d = Dispatcher(empty_document())
    if isinstance(spec, dict):
        d.execute({"op": "create", "id": "p", "type": spec["type"], "params": {"width": 16, "height": 12, **spec.get("params", {})}})
    else:
        d.execute({"op": "create", "id": "p", "type": "Constant", "params": {"width": 16, "height": 12, "red": spec[0], "green": spec[1],
                                                                              "blue": spec[2], "alpha": spec[3]}})
    d.execute({"op": "view", "id": "p"})
    return np.asarray(Evaluator().evaluate_raster(d.document, target="p").pixels, np.float64)


def _rotopaint_cases(Case, out):
    """RotoPaint: what a stroke in the node's payload draws. A single dab with hardness 1 is a disc of the brush size's diameter; a pixel the
    rim passes through is not scored (the rim's antialiasing is not stated), so the disc's interior and the picture outside the rim are."""
    def case(id, nuke_knobs, nb_params, expected, source, statement, unscored="", nuke_inputs=("bg",)):
        out.append(Case(id=id, nuke="RotoPaint", nuke_class="RotoPaint", nuke_knobs=nuke_knobs, nuke_inputs=nuke_inputs, nb_kind="RotoPaint",
                        nb_params=nb_params, nb_inputs=(), expected=expected, source=source, statement=statement, unscored=unscored))

    yy, xx = np.mgrid[0:12, 0:16]
    dist = np.hypot(xx + 0.5 - 8.0, yy + 0.5 - 6.0)
    flat = (0.2, 0.4, 0.6, 1.0)
    checker = {"type": "Checker", "params": {"size": 2}}

    def disc(radius, inside, plate=flat):
        """`inside` is the picture inside the disc (an RGBA tuple or an array), `plate` the picture outside; the rim (within 1 of the radius) is not scored."""
        def expected(fx):
            ground = _plate_array(plate)
            ref = np.where((dist < radius)[..., None], np.asarray(inside, np.float64) if not isinstance(inside, np.ndarray) else inside, ground)
            valid = ((dist < radius - 1.0) | (dist > radius + 1.0))[..., None] & np.ones((12, 16, 4), bool)
            return ref, valid
        return expected

    case("rotopaint_stroke_is_a_disc_of_the_brush_diameter", (("brush_size", 6),), {"_items": [paint_stroke()], "_bg": flat},
         disc(3.0, (1.0, 0.0, 0.0, 1.0)), "guide",
         f"RotoPaint brush size ({ROTOPAINT_GUIDE}): 'sets the brush stroke diameter in pixels'; with hardness 1 (a hard edge) one dab of size 6 paints "
         "the colour at full strength within 3 pixels of its centre and leaves the picture alone beyond; the rim is not scored")
    case("rotopaint_a_larger_brush_paints_a_larger_disc", (("brush_size", 10),),
         {"_items": [paint_stroke(brush={"size": 10.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2})], "_bg": flat},
         disc(5.0, (1.0, 0.0, 0.0, 1.0)), "guide", "RotoPaint brush size: a size of 10 paints a disc of diameter 10 (radius 5)")
    case("rotopaint_opacity_scales_the_stroke", (("opacity", 0.5),), {"_items": [paint_stroke(opacity=0.5)], "_bg": flat},
         disc(3.0, tuple(0.5 * np.array((1.0, 0.0, 0.0, 1.0)) + 0.5 * np.array(flat))), "guide",
         f"RotoPaint opacity ({ROTOPAINT_GUIDE}): 'the opacity of a shape or stroke in the list; 1 is totally opaque and 0 is transparent': at 0.5 the "
         "stroke's interior is the colour and the picture half and half")
    case("rotopaint_hardness_falloff_curve", (("brush_hardness", 0.2),),
         {"_items": [paint_stroke(brush={"size": 6.0, "hardness": 0.2, "opacity": 1.0, "spacing": 0.2, "strength": 0.2})], "_bg": flat},
         disc(3.0, (1.0, 0.0, 0.0, 1.0)), "guide", "RotoPaint brush hardness: the falloff from the brush centre to its edge",
         unscored="the guide says hardness 'defines the rate of falloff from the brush center to its edge' and that increasing it 'decreases blurring "
                  "at the stroke edges', but gives no curve; the monotone property (a harder brush paints more of a given pixel, a softer one less "
                  "at the same distance) is tested in tests/test_2d_behaviour_roto.py")

    def clone_expected(fx):
        ground = _plate_array(checker)
        shifted = np.roll(ground, 3, axis=1)          # a pixel takes the source three to its left: the clone offset moves the source 3 right
        shifted[:, :3] = ground[:, :1]                # beyond the source's left edge its edge pixel repeats
        ref = np.where((dist < 3.0)[..., None], shifted, ground)
        valid = ((dist < 2.0) | (dist > 4.0))[..., None] & np.ones((12, 16, 4), bool)
        return ref, valid
    case("rotopaint_clone_offset_moves_the_source", (("source_transform_translate", [3, 0]),),
         {"_items": [paint_stroke("clone", source_offset=[3.0, 0.0])], "_bg": checker}, clone_expected, "guide",
         f"RotoPaint clone ({ROTOPAINT_GUIDE}): 'Clone tool: Ctrl/Cmd+click and drag to set the clone offset': the pixels the clone dab covers show the "
         "plate moved by the offset (3 to the right: each covered pixel takes the plate pixel 3 to its left); the rim is not scored")
    second = (0.9, 0.1, 0.2, 1.0)
    case("rotopaint_reveal_shows_the_second_image", (("brush_type", "reveal"),),
         {"_items": [paint_stroke("reveal")], "_bg": flat, "_second": second}, disc(3.0, second), "convention",
         f"RotoPaint's reveal tool ({ROTOPAINT_GUIDE}, tool list: 'brush, clone/reveal, blur, and dodge/burn'): a reveal stroke shows the other image "
         "(Nuke's second bg input) where the brush paints, and leaves the first alone elsewhere",
         nuke_inputs=("bg", "bg2"))


# ------------------------------------------------------------------------------------------------------------- step F
TRACKER_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/transform_nodes/tracker.html"


def bilinear_shift(a, sx, sy):
    """`a` (rows, columns, channels) moved by (sx, sy) pixels, sampled bilinearly: out[y, x] = a[y - sy, x - sx], the edge pixel repeating past the picture."""
    a = np.asarray(a, np.float64)
    h, w = a.shape[:2]
    fx = np.arange(w)[None, :] - sx
    fy = np.arange(h)[:, None] - sy
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]

    def at(yy, xx):
        return a[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
    return ((at(y0, x0) * (1 - tx) + at(y0, x0 + 1) * tx) * (1 - ty) + (at(y0 + 1, x0) * (1 - tx) + at(y0 + 1, x0 + 1) * tx) * ty)


def _tracker_cases(Case, out):
    """Tracker: the transform its tracks give. One track moves by a sub-pixel (2.5, 1.5) between frames 1 and 2 (x across, y down, the
    track's own pixel coordinates); match-move applies that motion to the plate, stabilise removes it. The plate is a ramp, so a bilinear
    resample of it is exact; pixels whose source falls outside the plate are not scored. (Finding the track in footage is tested in
    tests/test_2d_behaviour_filter_f.py: a case here has one frame and no footage to follow.)"""
    plate = {"type": "Ramp", "params": {"p0_x": 0.0, "p0_y": 0.0, "p1_x": 15.0, "p1_y": 11.0}}
    track = {1: (8.0, 6.0), 2: (10.5, 7.5)}
    xx = np.arange(16)[None, :]
    yy = np.arange(12)[:, None]

    def moved_by(sx, sy, inside):
        def expected(fx):
            ref = bilinear_shift(_plate_array(plate), sx, sy)
            return ref, np.broadcast_to(inside[..., None], ref.shape).copy()
        return expected
    forward = (xx >= 3) & (yy >= 2)
    backward = (xx <= 12) & (yy <= 9)
    out.append(Case(
        id="tracker_match_move_follows_the_track", nuke="Tracker", nuke_class="Tracker4", nuke_knobs=(("transform", "match-move"),),
        nuke_inputs=("bg",), nb_kind="Tracker", nb_params={"_tracks": [track], "_frame": 2, "_bg": plate, "mode": "match_move", "filter": "bilinear"},
        nb_inputs=(), expected=moved_by(2.5, 1.5, forward), source="guide",
        statement=f"Tracker ({TRACKER_GUIDE}): the transform menu's match-move 'applies the tracked motion to the image': a track that moves 2.5 pixels "
                  "right and 1.5 down between the reference frame and frame 2 moves the plate by the same, resampled bilinearly"))
    out.append(Case(
        id="tracker_stabilise_removes_the_track", nuke="Tracker", nuke_class="Tracker4", nuke_knobs=(("transform", "stabilize"),),
        nuke_inputs=("bg",), nb_kind="Tracker", nb_params={"_tracks": [track], "_frame": 2, "_bg": plate, "mode": "stabilise", "filter": "bilinear"},
        nb_inputs=(), expected=moved_by(-2.5, -1.5, backward), source="guide",
        statement=f"Tracker ({TRACKER_GUIDE}): stabilize 'removes the tracked motion from the image': the same track moves the plate 2.5 left and 1.5 up"))
    out.append(Case(
        id="tracker_reference_frame_is_the_identity", nuke="Tracker", nuke_class="Tracker4", nuke_knobs=(("transform", "match-move"),),
        nuke_inputs=("bg",), nb_kind="Tracker", nb_params={"_tracks": [track], "_frame": 1, "_bg": plate, "mode": "match_move", "filter": "bilinear"},
        nb_inputs=(), expected=moved_by(0.0, 0.0, np.ones((12, 16), bool)), source="guide",
        statement="Tracker: at the reference frame the track has not moved, so match-move leaves the plate as it is"))


DROPSHADOW_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/dropshadow.html"


def _shift_zero(a, dx, dy):
    """`a` moved by (dx, dy) whole pixels, transparent (zero) where nothing arrives."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def _dropshadow_cases(Case, out):
    """DropShadow: the input's alpha, moved by the angle and distance, scaled by the opacity, tinted by the colour and put under the input.
    The guide: angle 'where 0/360 is equal to left and 180 is right', distance 'in pixels', opacity 'controls the opacity of the shadow effect',
    colour 'controls the color used for the drop shadow effect' (inherit input color off), 'enable drop shadow: when enabled, the drop shadow
    effect is applied'. It states no composite formula; the shadow goes under the input as the standard over (shadow * (1 - alpha) added to the
    premultiplied input). Pixels the shifted alpha reads from beyond the 16 x 12 picture are not scored (the guide does not say what is there)."""
    def case(id, nuke_knobs, nb_params, expected, source, statement, unscored=""):
        out.append(Case(id=id, nuke="DropShadow", nuke_class="DropShadow", nuke_knobs=nuke_knobs, nuke_inputs=("A",), nb_kind="DropShadow",
                        nb_params=nb_params, nb_inputs=("A",), expected=expected, source=source, statement=statement, unscored=unscored))

    def shadowed(dx, dy, opacity=1.0, tint=(0.0, 0.0, 0.0), inherit=False):
        def expected(fx):
            src = fx["A"].astype(np.float64)
            alpha = src[..., 3:4]
            planes = src if inherit else alpha
            moved = _shift_zero(planes, dx, dy) * opacity
            colour = np.ones(4) if inherit else np.array([*tint, 1.0])
            ref = src + moved * colour * (1.0 - alpha)
            valid = np.ones(src.shape, bool)
            if dx > 0:
                valid[:, :dx] = False
            if dx < 0:
                valid[:, dx:] = False
            if dy > 0:
                valid[:dy] = False
            if dy < 0:
                valid[dy:] = False
            return ref, valid
        return expected

    base = {"distance": 3.0, "shadow_size": 0.0, "opacity": 1.0, "red": 0.0, "green": 0.0, "blue": 0.0}
    knobs = (("dropshadow_distance", 3), ("softness", 0), ("opacity", 1.0))
    case("dropshadow_angle_180_casts_right", knobs + (("dropshadow_angle", 180),), {**base, "angle": 180.0}, shadowed(3, 0), "convention",
         f"DropShadow ({DROPSHADOW_GUIDE}): angle 'where 0/360 is equal to left and 180 is right' and distance 'in pixels': at 180 and 3 the shadow "
         "is the alpha moved 3 pixels right, under the input")
    case("dropshadow_angle_0_casts_left", knobs + (("dropshadow_angle", 0),), {**base, "angle": 0.0}, shadowed(-3, 0), "convention",
         "DropShadow: angle 0 casts the shadow to the left")
    case("dropshadow_distance_moves_the_shadow", (("dropshadow_distance", 5), ("softness", 0), ("opacity", 1.0), ("dropshadow_angle", 180)),
         {**base, "distance": 5.0, "angle": 180.0}, shadowed(5, 0), "convention", "DropShadow: distance 5 moves the shadow 5 pixels")
    case("dropshadow_opacity_scales_the_shadow", knobs[:2] + (("opacity", 0.4), ("dropshadow_angle", 180)),
         {**base, "opacity": 0.4, "angle": 180.0}, shadowed(3, 0, opacity=0.4), "convention",
         "DropShadow: opacity 'controls the opacity of the shadow effect': at 0.4 the shadow is 0.4 as strong")
    case("dropshadow_colour_tints_the_shadow", knobs + (("dropshadow_angle", 180), ("color", [1.0, 0.25, 0.0])),
         {**base, "angle": 180.0, "red": 1.0, "green": 0.25, "blue": 0.0}, shadowed(3, 0, tint=(1.0, 0.25, 0.0)), "convention",
         "DropShadow: colour 'controls the color used for the drop shadow effect': the shadow takes it")
    case("dropshadow_inherit_input_colour", knobs + (("dropshadow_angle", 180), ("inherit_input_color", True)),
         {**base, "angle": 180.0, "inherit_input_color": 1}, shadowed(3, 0, inherit=True), "convention",
         "DropShadow: inherit input color 'uses the input color as the shadow': the shadow carries the input's own colour and alpha")
    case("dropshadow_disabled_is_the_input", (("enable_dropshadow_effect", False),), {**base, "enable_dropshadow_effect": 0},
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "guide",
         "DropShadow: 'enable drop shadow: when enabled, the drop shadow effect is applied': off, the picture passes unchanged")
    case("dropshadow_angle_90_direction", knobs + (("dropshadow_angle", 90),), {**base, "angle": 90.0}, shadowed(0, -3), "convention",
         "DropShadow: whether 90 casts up or down", unscored="the guide gives 0 as left and 180 as right and does not say which way the angle turns, "
                                                             "so up or down at 90 and 270 is not stated")
    case("dropshadow_softness_blur", (("softness", 4), ("dropshadow_distance", 3), ("dropshadow_angle", 180)),
         {**base, "shadow_size": 4.0, "angle": 180.0}, shadowed(3, 0), "convention", "DropShadow: softness blurs the shadow",
         unscored="the guide says softness 'controls how much blur/feather is applied to the shadow effect' and names no filter or radius; that a larger "
                  "softness spreads the shadow and keeps its total is tested in tests/test_2d_behaviour_filter_f.py")


DIRBLUR_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/dirblur.html"


def _dirblur_cases(Case, out):
    """DirBlur: linear blur 'smears pixels in a given direction'; BlurLength is 'the length (in pixels) of the blur applied'; BlurAngle 'determines the
    direction of linear blur'. The smear is taken as a uniform box of that length along the direction (the form a motion blur has); on whole-pixel
    steps along the axes a length of 3 is the mean of three pixels. Pixels within one of the edge, where the box reaches past the picture, are not scored."""
    def case(id, nuke_knobs, nb_params, expected, source, statement, unscored=""):
        out.append(Case(id=id, nuke="DirBlur", nuke_class="DirBlur2", nuke_knobs=nuke_knobs, nuke_inputs=("A",), nb_kind="DirBlur",
                        nb_params=nb_params, nb_inputs=("A",), expected=expected, source=source, statement=statement, unscored=unscored))

    def box_along_rows(length):
        def expected(fx):
            src = fx["A"].astype(np.float64)
            half = length // 2
            ref = src.copy()
            acc = np.zeros_like(src)
            for t in range(-half, half + 1):
                acc += _shifted(src, 0, t, half)
            ref[...] = acc / length
            return ref, _interior(src.shape, half) & np.ones(src.shape, bool)
        return expected

    def box_along_columns(length):
        def expected(fx):
            src = fx["A"].astype(np.float64)
            half = length // 2
            acc = np.zeros_like(src)
            for t in range(-half, half + 1):
                acc += _shifted(src, t, 0, half)
            return acc / length, _interior(src.shape, half)
        return expected
    common = {"blur_type": "linear", "channels": "rgba"}
    case("dirblur_linear_length_3_along_x", (("BlurType", "linear"), ("BlurLength", 3), ("BlurAngle", 0)),
         {**common, "length": 3.0, "angle": 0.0}, box_along_rows(3), "convention",
         f"DirBlur ({DIRBLUR_GUIDE}): linear blur at angle 0 and length 3 averages each pixel with its two horizontal neighbours")
    case("dirblur_linear_length_5_along_x", (("BlurType", "linear"), ("BlurLength", 5), ("BlurAngle", 0)),
         {**common, "length": 5.0, "angle": 0.0}, box_along_rows(5), "convention", "DirBlur: length 5 is the mean of five pixels along the line")
    case("dirblur_linear_angle_90_runs_down_the_columns", (("BlurType", "linear"), ("BlurLength", 3), ("BlurAngle", 90)),
         {**common, "length": 3.0, "angle": 90.0}, box_along_columns(3), "convention",
         "DirBlur: BlurAngle 90 turns the linear smear to the vertical")
    case("dirblur_linear_length_1_is_the_input", (("BlurType", "linear"), ("BlurLength", 1), ("BlurAngle", 0)), {**common, "length": 1.0, "angle": 0.0},
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "convention",
         "DirBlur: a length of one pixel smears nothing")
    case("dirblur_zoom_and_radial_formulas", (("BlurType", "zoom"), ("BlurLength", 10)), {"blur_type": "zoom", "length": 10.0, "center_x": 8.5, "center_y": 6.5},
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "convention", "DirBlur: the zoom and radial smears",
         unscored="the guide says BlurLength is 'in pixels' for every type and gives no path for zoom or radial (NodeBased reads the zoom length as a percent of "
                  "the distance to the centre and the radial length as degrees); what is stated (the centre pixel stays, rays stay on their line, "
                  "flat areas stay) is tested in tests/test_2d_behaviour_filter_f.py")


LIGHTWRAP_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/draw_nodes/lightwrap.html"


def _lightwrap_cases(Case, out):
    """LightWrap: 'wrapping or spilling light from the background onto the foreground object ... a reflection of light around the edges of your foreground
    object by blending in whatever is in the background'. Intensity 'adjusts the brightness of the reflections' (Nuke's default is 0, no reflection) and
    'generate wrap only' makes 'the LightWrap effect only, without merging it with the foreground'. The guide gives no formula for the amount at an edge
    pixel, so the amount itself is not scored; that the wrap sits at the foreground's edge, widens with Diffuse and scales with Intensity and the
    background is tested in tests/test_2d_behaviour_filter_f.py."""
    def case(id, nuke_knobs, nb_params, expected, source, statement, unscored=""):
        out.append(Case(id=id, nuke="LightWrap", nuke_class="LightWrap", nuke_knobs=nuke_knobs, nuke_inputs=("A", "B"), nb_kind="LightWrap",
                        nb_params=nb_params, nb_inputs=("A", "B"), expected=expected, source=source, statement=statement, unscored=unscored))
    case("lightwrap_intensity_zero_adds_nothing", (("Intensity", 0),), {"intensity": 0.0},
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "convention",
         f"LightWrap ({LIGHTWRAP_GUIDE}): Intensity 'adjusts the brightness of the reflections on the foreground element': at 0 there are none, and the "
         "foreground comes through as it was")

    def only_wrap_is_black(fx):
        ref = np.zeros(fx["A"].shape, np.float64)
        valid = np.zeros(ref.shape, bool)
        valid[..., :3] = True
        return ref, valid
    case("lightwrap_generate_wrap_only_without_wrap_is_black", (("Intensity", 0), ("onlywrap", True)), {"intensity": 0.0, "onlywrap": 1},
         only_wrap_is_black, "guide",
         "LightWrap: 'generate wrap only: when enabled, generates the LightWrap effect only, without merging it with the foreground object': with no "
         "reflection (Intensity 0) there is nothing to show, so the colour is black and the foreground is not in it")
    case("lightwrap_wrap_amount_at_the_edge", (("Intensity", 1), ("size", 15)), {"intensity": 1.0},
         lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "convention", "LightWrap: how much of the background is added at an edge pixel",
         unscored="the guide describes the effect ('a reflection of light around the edges of your foreground object by blending in whatever is in the "
                  "background') and gives neither the falloff nor the blur's filter, so the amount cannot be written down without guessing")


BILATERAL_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/bilateral.html"


def _bilateral_cases(Case, out):
    """Bilateral: a smoothing filter that preserves edges. Size is 'the height and width of the area within which pixels are compared', Colour Sigma
    'controls the amount of blur that occurs over edges in the image, lower values preserve edges more reliably', Positional Sigma 'controls the
    amount that nearby pixels affect the overall blur', and the guide input 'computes color similarity while filtering'. The guide gives no weight
    formula, no pixel scale for the positional sigma and no median rule beyond 'weighted according to the middle value pixel', so the exact smoothed
    picture is not scored; the two limits the text does state are. The guide input needs a third slot the case runner cannot fill; the step's
    test module (tests/test_2d_behaviour_filter_g.py) shows a step edge staying where the guide puts it."""
    def case(id, nuke_knobs, nb_params, expected, source, statement, unscored=""):
        out.append(Case(id=id, nuke="Bilateral", nuke_class="Bilateral", nuke_knobs=nuke_knobs, nuke_inputs=("A",), nb_kind="Bilateral",
                        nb_params=nb_params, nb_inputs=("A",), expected=expected, source=source, statement=statement, unscored=unscored))

    same = lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool))
    case("bilateral_size_1_compares_only_the_pixel", (("size", 1.0),), {"spatial_size": 1.0}, same, "convention",
         f"Bilateral ({BILATERAL_GUIDE}): size 'sets the height and width of the area within which pixels are compared'; a 1 by 1 area holds only the "
         "pixel itself, so the picture is unchanged")
    case("bilateral_tiny_colour_sigma_keeps_every_edge", (("colourSigma", 1e-4),), {"colour_sigma": 1e-4}, same, "convention",
         "Bilateral: colour sigma 'controls the amount of blur that occurs over edges, lower values preserve edges more reliably'; in the limit no "
         "pixel is blended with one of a different colour, so the picture is unchanged")
    case("bilateral_gaussian_weights", (("size", 3.0), ("positionalSigma", 0.4), ("colourSigma", 0.4), ("filter", "gaussian")), {}, same, "convention",
         "Bilateral at its defaults: the smoothed pixels",
         unscored="the guide names no weight formula and no pixel scale for the positional sigma; NodeBased uses the classic Gaussian form with sigma "
                  "positionalSigma x size pixels (nodebased/bilateralknobs.py), and tests/test_2d_behaviour_filter_g.py shows a step edge staying sharp "
                  "while flat noise smooths")
    case("bilateral_median_filter", (("filter", "median"),), {"bilateral_filter": "median"}, same, "convention",
         "Bilateral: filter median, 'weighted according to the middle value pixel'",
         unscored="the guide does not say whether the middle value is a weighted median or a plain one; NodeBased takes the weighted median under the "
                  "gaussian weights")


VECTORCORNERPIN_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/transform_nodes/vectorcornerpin.html"


def _vectorcornerpin_cases(Case, out):
    """VectorCornerPin: 'takes paint or an image from a reference frame and propagates it through the rest of the sequence' like CornerPin2D. The
    user pins 'control the position of the pins at the current keyframe', the From quad is the corner pin's source (pin 1 the bottom left, 2 the
    bottom right, 3 the top right, 4 the top left). With no SmartVector wired the pins do not move, so the picture is the corner pin of the From
    quad onto the user pins. The plate is a ramp (a bilinear resample of a ramp is exact), the moves are whole pixels, and a pixel whose source
    falls outside the plate is not scored. Points are in Nuke's pixel space (y up from the bottom edge)."""
    plate = {"type": "Ramp", "params": {"p0_x": 0.0, "p0_y": 0.0, "p1_x": 15.0, "p1_y": 11.0}}
    w, h = 16, 12

    def pins(points, side="user"):
        return {f"{side}{i}_{axis}": float(v) for i, point in enumerate(points, 1) for axis, v in zip("xy", point)}

    def case(id, nuke_knobs, nb_params, expected, source, statement, unscored=""):
        out.append(Case(id=id, nuke="VectorCornerPin", nuke_class="VectorCornerPin", nuke_knobs=nuke_knobs, nuke_inputs=("A",), nb_kind="VectorCornerPin",
                        nb_params={"_plate": plate, **nb_params}, nb_inputs=("A",), expected=expected, source=source, statement=statement, unscored=unscored))

    def moved(dx, dy):
        def expected(fx):
            src = _plate_array(plate)
            ref = np.zeros_like(src)
            valid = np.ones(src.shape, bool)
            for row in range(h):
                for x in range(w):
                    i, j = x - dx, row + dy             # a pin raised by dy (y up) draws the picture dy rows higher
                    if 0 <= i < w and 0 <= j < h:
                        ref[row, x] = src[j, i]
                    else:
                        valid[row, x] = False
            return ref, valid
        return expected

    frame = ((0, 0), (w, 0), (w, h), (0, h))
    case("vectorcornerpin_pins_on_the_frame_are_the_identity", (), pins(frame), moved(0, 0), "convention",
         f"VectorCornerPin ({VECTORCORNERPIN_GUIDE}): the user pins on the format's four corners leave the picture where it is")
    quad = tuple((x + 3, y + 2) for x, y in frame)
    case("vectorcornerpin_pins_moved_carry_the_picture", (("user1", "3 2"), ("user2", "19 2"), ("user3", "19 14"), ("user4", "3 14")), pins(quad),
         moved(3, 2), "convention", "VectorCornerPin: user pins 3 pixels right and 2 up put the picture there (the corner pin of the From quad onto the pins)")
    flipped = (frame[1], frame[0], frame[3], frame[2])
    case("vectorcornerpin_swapped_pins_mirror_the_picture", (("user1", "16 0"), ("user2", "0 0"), ("user3", "0 12"), ("user4", "16 12")), pins(flipped),
         lambda fx: (_plate_array(plate)[:, ::-1], np.ones((h, w, 4), bool)), "convention",
         "VectorCornerPin: pins 1 and 2, 3 and 4 exchanged turn the picture over left to right")
    shifted_from = tuple((x + 3, y + 2) for x, y in frame)
    case("vectorcornerpin_from_quad_picks_the_source", (("to1#2", "3 2"), ("to2#2", "19 2"), ("to3#2", "19 14"), ("to4#2", "3 14")),
         {**pins(shifted_from, "from"), **pins(frame)}, moved(-3, -2), "convention",
         "VectorCornerPin: the From quad is the source of the pin: a From quad 3 right and 2 up of the format, pinned to the format, moves the picture "
         "3 left and 2 down")


WARP_GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/transform_nodes/gridwarp.html"


def warp_point(x, y, in_y=0.0, out_y=0.0):
    return {"x": float(x), "y": float(y), "in_x": 0.0, "in_y": float(in_y), "out_x": 0.0, "out_y": float(out_y)}


def _warp_cases(Case, out):
    """GridWarp and SplineWarp: 'the source grid (or curves) is carried onto the destination one'. A key (or curve) moved by a known offset moves the
    pixels under it by that offset, the controls left alone stay, and mix 0 is the input. The plate is a ramp (a bilinear resample of a ramp is exact),
    the moves are whole pixels, and only the pixels the statement is about are scored (the field around a control is the interpolant's, which the guide
    does not give). Payload coordinates are pixel centres, rows counted from the top."""
    plate = {"type": "Ramp", "params": {"p0_x": 0.0, "p0_y": 0.0, "p1_x": 15.0, "p1_y": 11.0}}
    w, h = 16, 12
    xs, ys = (0.5, 7.5, 15.5), (0.5, 5.5, 11.5)
    grid = [[{"x": x, "y": y} for x in xs] for y in ys]
    moved_grid = [[dict(point) for point in row] for row in grid]
    moved_grid[1][1] = {"x": 10.5, "y": 5.5}            # the one interior key, three pixels to the right
    grid_node = {"rows": 3, "columns": 3, "filter": "bilinear"}

    def case(id, kind, nb_params, expected, statement, unscored=""):
        out.append(Case(id=id, nuke=kind, nuke_class=kind, nuke_knobs=(), nuke_inputs=("A",), nb_kind=kind,
                        nb_params={"_plate": plate, **nb_params}, nb_inputs=("A",), expected=expected, source="convention",
                        statement=f"{kind} ({WARP_GUIDE}): {statement}", unscored=unscored))

    def key_moved(fx):
        src = _plate_array(plate)
        ref, valid = src.copy(), np.zeros(src.shape, bool)
        ref[5, 10] = src[5, 7]
        for row, col in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (5, 10)):
            valid[row, col] = True
        return ref, valid

    def whole(fx):
        src = _plate_array(plate)
        return src, np.ones(src.shape, bool)

    case("gridwarp_interior_key_moves_the_pixel_under_it", "GridWarp", {**grid_node, "_warp": {"source": grid, "destination": moved_grid}}, key_moved,
         "an interior key moved 3 pixels right carries the picture under it 3 pixels right and leaves the corner keys where they are")
    case("gridwarp_mix_zero_is_the_source", "GridWarp", {**grid_node, "mix": 0.0, "_warp": {"source": grid, "destination": moved_grid}}, whole,
         "mix 0 gives the source back")
    case("gridwarp_output_source_is_the_source", "GridWarp", {**grid_node, "grid_output": "source", "_warp": {"source": grid, "destination": moved_grid}},
         whole, "output 'source' shows the source picture, unwarped")
    case("gridwarp_background_over_the_source", "GridWarp",
         {**grid_node, "background": "on src", "background_mix": 1.0, "_warp": {"source": grid, "destination": moved_grid}}, key_moved,
         "the warped picture composited over the source (on src) with the background mix full, an opaque picture, is the warped picture",
         unscored="the guide's page for this node is a video transcript with no formula for the background and its mix; the reading is NodeBased's")

    def curve(x, extent=(0.5, 10.5)):
        # Handles a third of the length along the curve make the Bezier a straight, evenly sampled line (zero handles bunch the samples at the ends).
        third = (extent[1] - extent[0]) / 3.0
        return [warp_point(x, extent[0], out_y=third), warp_point(x, extent[1], in_y=-third)]

    pairs = {"pairs": [{"name": "edge", "source": curve(7.5), "destination": curve(10.5)}]}
    spline = {"curve_resolution": 5, "filter": "bilinear"}       # samples every 2 pixels: 0.5, 2.5 ... 10.5, each at a pixel centre

    def on_curve(fx):
        src = _plate_array(plate)
        ref, valid = src.copy(), np.zeros(src.shape, bool)
        for row in range(0, 11, 2):
            ref[row, 10] = src[row, 7]
            valid[row, 10] = True
        return ref, valid

    case("splinewarp_curve_moves_the_pixels_on_it", "SplineWarp", {**spline, "_warp": pairs}, on_curve,
         "a source curve paired with a destination curve 3 pixels right carries the pixels on the curve 3 pixels right (the pixels at the curve's samples are scored)")
    case("splinewarp_mix_zero_is_the_input", "SplineWarp", {**spline, "mix": 0.0, "_warp": pairs}, whole, "mix 0 gives the input back")
