"""Solving a 2D similarity from resolved tracks, and finding tracks in pixels.

Two separate things live here, and only the first one is wired into evaluation in this pass:

* :func:`solve` turns tracks resolved at a reference frame and at the requested frame into the
  ``(center, rotate, scale, translate)`` parameterisation ``Evaluator._transform`` already
  consumes, so a Tracker is a Transform whose numbers came from data instead of from a knob. It
  is pure arithmetic on plain numbers -- no document, no images -- which is what lets both the
  region rule and the kernel call it and get the same answer by construction.
* :func:`analyse` is the correlator that produces track positions *from pixels*. It is library
  surface only: no UI button, no agent op, no viewer markers. Wiring it to the interface is named
  work for the next owner rather than something this pass claims.

Sign convention, fixed once here: ``solve`` returns the transform that maps a point at the
**reference** frame onto where the tracks say it has moved to at the **requested** frame. That is
``match_move``. ``stabilise`` is its exact algebraic inverse, expressed in the same
parameterisation so both round-trip through one kernel rather than through two code paths that
can disagree. See docs/ROTO_TRACKING.md.
"""
from __future__ import annotations

import math

import numpy as np

# The field set `Evaluator._transform` and `tiers._transform_rule` both read. Returned in full
# every time, including the zeros, so a caller can splice it over a params dict without having to
# know which fields a particular solve happened to produce.
IDENTITY = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0, "scale": 1.0,
            "center_x": 0.0, "center_y": 0.0}

# Below this, the reference points are effectively coincident and no scale or rotation is
# observable from them. Reported as identity rather than divided by, because the alternative is a
# solve that explodes on a frame where two tracks momentarily overlap.
DEGENERATE_SPREAD = 1e-9


def participating(reference_tracks, current_tracks):
    """Tracks usable for a solve: present and enabled at *both* frames, paired by name.

    A track that drops out under an occlusion contributes nothing on those frames and comes back
    when it reappears. Erroring instead would make the node unusable mid-scrub, and holding the
    last good solve would make the result depend on scrub order, which breaks cache determinism.
    """
    current = {track["name"]: track for track in current_tracks if track["enabled"]}
    pairs = []
    for track in reference_tracks:
        if not track["enabled"]:
            continue
        match = current.get(track["name"])
        if match is not None:
            pairs.append(((track["x"], track["y"]), (match["x"], match["y"])))
    return pairs


def _fit(pairs, allow_rotate, allow_scale):
    """Least-squares similarity from reference points onto current points.

    `allow_rotate` / `allow_scale` **constrain the fit** rather than zeroing its result
    afterwards: the best translation-only fit against rotating tracks is not the translation
    component of the best rotation, and an artist who unticks "rotate" wants the former.
    """
    source = np.array([p for p, _ in pairs], dtype=np.float64)
    destination = np.array([q for _, q in pairs], dtype=np.float64)
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    centred_source = source - source_mean
    centred_destination = destination - destination_mean
    variance = float((centred_source ** 2).sum())

    theta, scale = 0.0, 1.0
    if variance > DEGENERATE_SPREAD and (allow_rotate or allow_scale):
        if allow_rotate:
            # Umeyama for the 2D similarity, written out rather than via SVD: in two dimensions
            # the rotation that maximises the correlation has a closed form, and excluding the
            # reflection is then simply not considering it.
            a = float((centred_source * centred_destination).sum())
            b = float((centred_source[:, 0] * centred_destination[:, 1]
                       - centred_source[:, 1] * centred_destination[:, 0]).sum())
            theta = math.atan2(b, a)
            if allow_scale:
                scale = math.hypot(a, b) / variance
        elif allow_scale:
            # Rotation pinned at zero: the best uniform scale is the projection of the destination
            # onto the source, which is the same formula with the rotation term dropped.
            scale = float((centred_source * centred_destination).sum()) / variance

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotated_x = scale * (cos_t * source_mean[0] - sin_t * source_mean[1])
    rotated_y = scale * (sin_t * source_mean[0] + cos_t * source_mean[1])
    return {"translate_x": float(destination_mean[0] - rotated_x),
            "translate_y": float(destination_mean[1] - rotated_y),
            "rotate": float(math.degrees(theta)), "scale": float(scale),
            "center_x": 0.0, "center_y": 0.0}


def invert(solved):
    """The exact algebraic inverse of a solve, in the same parameterisation.

    With centre at the origin the forward map is ``dst = R(t) * s * src + T``, so the inverse is
    ``src = R(-t) / s * dst - R(-t) / s * T``, which is again a rotate/scale/translate triple.
    Writing it this way is what lets `stabilise` run through the same kernel as `match_move`
    instead of growing a second sampling path that can drift from the first.
    """
    scale = solved["scale"]
    if abs(scale) < DEGENERATE_SPREAD:
        # A collapsed solve has no inverse. Identity is the honest answer and matches the
        # zero-track case; the alternative divides by ~0 and writes garbage into the viewer.
        return dict(IDENTITY)
    theta = math.radians(-solved["rotate"])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    inverse_scale = 1.0 / scale
    tx, ty = solved["translate_x"], solved["translate_y"]
    return {"translate_x": float(-inverse_scale * (cos_t * tx - sin_t * ty)),
            "translate_y": float(-inverse_scale * (sin_t * tx + cos_t * ty)),
            "rotate": float(-solved["rotate"]), "scale": float(inverse_scale),
            "center_x": 0.0, "center_y": 0.0}


def solve(reference_tracks, current_tracks, params):
    """The transform a Tracker applies at the requested frame.

    Returns every field in `IDENTITY`, always. Zero usable tracks is identity by declaration, not
    by fallback -- see `participating`.
    """
    pairs = participating(reference_tracks, current_tracks)
    apply_translate = bool(int(params.get("apply_translate", 1)))
    apply_rotate = bool(int(params.get("apply_rotate", 1)))
    apply_scale = bool(int(params.get("apply_scale", 1)))
    if not pairs:
        solved = dict(IDENTITY)
    elif len(pairs) == 1:
        # One point cannot observe rotation or scale. Reporting identity for those is a statement
        # about what the data supports; guessing would be a statement about nothing.
        (sx, sy), (dx, dy) = pairs[0]
        solved = {**IDENTITY, "translate_x": float(dx - sx), "translate_y": float(dy - sy)}
    else:
        solved = _fit(pairs, apply_rotate, apply_scale)
    if not apply_translate:
        # Unlike the other two switches this *is* a post-fit zero, because "match the rotation but
        # not the position" is exactly what unticking translate means.
        solved = {**solved, "translate_x": 0.0, "translate_y": 0.0}
    if params.get("mode", "match_move") == "stabilise":
        solved = invert(solved)
    return solved


# --- analysis ------------------------------------------------------------------------------
#
# Everything below finds tracks in pixels. Deterministic and directly testable, but not reachable
# from the UI or the agent bridge in this pass.

def _zero_mean_normalised(patch):
    centred = patch - patch.mean()
    norm = float(np.sqrt((centred ** 2).sum()))
    return centred, norm


def correlate(image, pattern, search):
    """Zero-mean normalised cross-correlation of `pattern` over `search` within `image`.

    `image` is HxWx4 float32; correlation runs on luminance-free channel mean so a matte or a
    colour plate behave the same. Returns the (score, x, y) of the integer peak, in image
    coordinates of the pattern's top-left corner.
    """
    grey = image[..., :3].mean(axis=2)
    px, py, pw, ph = pattern
    sx, sy, sw, sh = search
    template, template_norm = _zero_mean_normalised(grey[py:py + ph, px:px + pw])
    if template_norm == 0.0:
        # A flat pattern correlates equally everywhere; there is no peak to find and returning one
        # would be inventing a measurement.
        return (0.0, px, py)
    best = (-2.0, px, py)
    for y in range(sy, sy + sh - ph + 1):
        for x in range(sx, sx + sw - pw + 1):
            window, window_norm = _zero_mean_normalised(grey[y:y + ph, x:x + pw])
            if window_norm == 0.0:
                continue
            score = float((template * window).sum() / (template_norm * window_norm))
            if score > best[0]:
                best = (score, x, y)
    return best


def _parabolic(left, middle, right):
    """Sub-pixel offset of a peak sampled at -1, 0, +1. Zero when the three are not a peak."""
    denominator = left - 2.0 * middle + right
    if denominator == 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return offset if -1.0 < offset < 1.0 else 0.0


def analyse(images, start, size, search_radius=16):
    """Track a pattern through `images`, one entry per frame, in order.

    `start` is the pattern's top-left corner in the first image and `size` its extent. Each frame
    seeds its search window from the previous frame's result, which is what keeps the search
    bounded on a long move. Returns a list of (x, y) sub-pixel positions, one per image, the first
    being `start` itself.
    """
    px, py = int(start[0]), int(start[1])
    pw, ph = int(size[0]), int(size[1])
    positions = [(float(px), float(py))]
    pattern = (px, py, pw, ph)
    for image in images[1:]:
        height, width = image.shape[0], image.shape[1]
        sx = max(0, px - search_radius)
        sy = max(0, py - search_radius)
        sw = min(width - sx, pw + 2 * search_radius)
        sh = min(height - sy, ph + 2 * search_radius)
        grey = image[..., :3].mean(axis=2)
        score, best_x, best_y = correlate(image, pattern, (sx, sy, sw, sh))
        template, template_norm = _zero_mean_normalised(
            images[0][..., :3].mean(axis=2)[pattern[1]:pattern[1] + ph, pattern[0]:pattern[0] + pw])

        def score_at(x, y):
            if x < 0 or y < 0 or x + pw > width or y + ph > height or template_norm == 0.0:
                return -2.0
            window, window_norm = _zero_mean_normalised(grey[y:y + ph, x:x + pw])
            if window_norm == 0.0:
                return -2.0
            return float((template * window).sum() / (template_norm * window_norm))

        # Sub-pixel refinement on the correlation surface itself, independently per axis. It is a
        # refinement of a measured peak, never a substitute for one: an integer peak that failed
        # to find anything stays where it is.
        dx = _parabolic(score_at(best_x - 1, best_y), score, score_at(best_x + 1, best_y))
        dy = _parabolic(score_at(best_x, best_y - 1), score, score_at(best_x, best_y + 1))
        positions.append((float(best_x + dx), float(best_y + dy)))
        px, py = best_x, best_y
    return positions
