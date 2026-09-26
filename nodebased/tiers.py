"""Region-of-interest and proxy-tier rules for the evaluator.

This module is deliberately free of NumPy kernels and Qt: it is pure geometry and parameter
arithmetic, so the ROI contract in `docs/EVALUATION_TIERS.md` can be tested without rendering a
single pixel.

Two independent ideas live here:

* **Region of interest.** A consumer asks a node for a rectangle of its output. Each node type
  declares how that request maps onto rectangles it needs from its own inputs (contract clause C2).
  A node type with no declared rule is a hard error — never a silent fall back to full frame, which
  would quietly erase the optimisation and hide the omission.
* **Proxy tier.** A whole evaluation runs at 1/n linear scale. Parameters carrying pixel units must
  be scaled in the same pass, otherwise a blur radius or a crop rectangle means a different thing at
  tier 2 than it does at tier 1 (contract clause C3).

Both are expressed against node *kinds* rather than kernel functions, because the amended thesis in
`docs/VISION.md` expects generative operators to join this table later. An operator whose cost is
seconds and dollars still has to answer "what part of your input do you need" before it is worth
scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

# The solved-transform vocabulary lives with the solver so the region rule and the kernel cannot
# drift apart about what a solved Tracker contains. `tracker` pulls in `shapes` and `animation`
# only, so this stays free of the renderer.
from .tracker import SOLVED_FIELDS as SOLVED_TRANSFORM_FIELDS

# Proxy tiers are linear downscale divisors. 1 is full resolution; the viewer offers 1/8 as well.
PROXY_TIERS = (1, 2, 4, 8)

# The pixel count above which the ACES 2.0 display transform's CPU cost (measured: ~2.3s at 4K,
# ~277ns/pixel, dwarfing the raw composite) makes full-tier playback infeasible. 1920x1080 rather
# than a rounder number because it is the resolution real footage most often already is -- HD
# plays at full quality, 4K and above gets a proxy.
PLAYBACK_AUTO_TIER_PIXELS = 1920 * 1080


def auto_playback_tier(width: int, height: int) -> int:
    """The largest-quality (smallest downscale) tier that brings playback under budget.

    This is the standard proxy-resolution-playback technique every NLE and compositor uses for
    the same reason: decide once, from the source's own size, rather than adapting live off
    measured frame times, which would make playback quality depend on machine load history
    instead of the footage. Never used to override a tier an artist already chose manually --
    see `Window.toggle_playback`.
    """
    for tier in PROXY_TIERS:
        if (int(width) // tier) * (int(height) // tier) <= PLAYBACK_AUTO_TIER_PIXELS:
            return tier
    return PROXY_TIERS[-1]


@dataclass(frozen=True)
class Region:
    """An integer pixel rectangle in a node's own output space.

    `x`/`y` are the top-left corner in image coordinates (y down, matching the array layout used by
    the evaluator). A region with zero or negative extent is empty and means "this input is not
    needed", which is different from `None`, meaning "this input must not be evaluated at all".
    """

    x: int
    y: int
    width: int
    height: int

    @staticmethod
    def whole(width: int, height: int) -> "Region":
        return Region(0, 0, int(width), int(height))

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def intersect(self, other: "Region") -> "Region":
        x, y = max(self.x, other.x), max(self.y, other.y)
        right, bottom = min(self.right, other.right), min(self.bottom, other.bottom)
        return Region(x, y, max(0, right - x), max(0, bottom - y))

    def union(self, other: "Region") -> "Region":
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        x, y = min(self.x, other.x), min(self.y, other.y)
        right, bottom = max(self.right, other.right), max(self.bottom, other.bottom)
        return Region(x, y, right - x, bottom - y)

    def expand(self, dx: int, dy: int) -> "Region":
        """Grow on every side. Used for filters with spatial support."""
        dx, dy = int(dx), int(dy)
        return Region(self.x - dx, self.y - dy, self.width + 2 * dx, self.height + 2 * dy)

    def clamp(self, width: int, height: int) -> "Region":
        return self.intersect(Region.whole(width, height))

    def scaled(self, tier: int) -> "Region":
        """Map this region into tier space, rounding outward so no needed pixel is dropped."""
        tier = int(tier)
        if tier == 1:
            return self
        x, y = self.x // tier, self.y // tier
        right = -((-self.right) // tier)  # ceil division
        bottom = -((-self.bottom) // tier)
        return Region(x, y, right - x, bottom - y)

    def as_slices(self):
        return (slice(self.y, self.bottom), slice(self.x, self.right))


class UndeclaredRegionRule(ValueError):
    """Raised when a node kind reaches the scheduler without an ROI mapping.

    Adding a node type is therefore a deliberate act: the author has to state what the node reads
    before it can be evaluated at all.
    """


# --- Region-of-interest rules -------------------------------------------------------------------
#
# Each rule takes (params, requested_region) and returns a list of `Region | None`, positionally
# matching the node's declared input slots in SPECS order (required slots first, then optional).
# `None` means the input must not be evaluated; an empty Region means it is needed but contributes
# nothing.


def _identity(params, region, arity):
    return [region] * arity


def _generator(params, region, arity):
    return []


def _blur_rule(params, region, arity):
    # Two separable box passes: the horizontal pass reaches `radius` in x, the vertical in y.
    # Below the kernel's own 0.5 cut-off the node is a copy, so it reads only what it is asked for.
    radius = float(params.get("radius", 0.0))
    support = 0 if radius < 0.5 else int(math.ceil(radius))
    image = region.expand(support, support)
    # The mask gates the *result*, so it is only ever sampled at the output pixels themselves.
    return [image] + [region] * (arity - 1)


def _uv_lookup_rule(params, region, arity):
    """STMap and IDistort read the image wherever the map points, which is not known from params
    alone, so the image is requested whole; the uv map and the mask are read pointwise. Both are
    excluded from the tile path (tiles.SUPPORTED_TILED_KINDS), so this states the need for
    `input_regions` without being scheduled."""
    return [None] + [region] * (arity - 1)


def _vector_blur_rule(params, region, arity):
    """The streak reaches at most `max_length` pixels from the output pixel, in any direction
    (`vector_scale` times the vector is capped there). With no cap the reach is unbounded and the
    image is requested whole. The uv map and the mask are read pointwise."""
    reach = abs(float(params.get("max_length", 0.0)))
    image = None if reach <= 0 else region.expand(int(math.ceil(reach)) + 1, int(math.ceil(reach)) + 1)
    return [image] + [region] * (arity - 1)


def _support_rule(param_name):
    """A `_blur_rule`-shaped region rule for a box filter whose own pixel-radius param is
    `param_name`: the image input is requested with `ceil(|size|)` pixels of extra padding on
    every side so tiles have the neighbours their kernel reads, and mask/mix inputs are not."""
    def rule(params, region, arity):
        size = abs(float(params.get(param_name, 0.0)))
        support = 0 if size < 0.5 else int(math.ceil(size))
        image = region.expand(support, support)
        return [image] + [region] * (arity - 1)
    return rule


_erode_rule = _support_rule("erode_size")
_dilate_rule = _support_rule("dilate_size")
_median_rule = _support_rule("median_size")
_sharpen_rule = _support_rule("sharpen_size")
_glow_rule = _support_rule("glow_size")
_soften_rule = _support_rule("soften_size")


def _edge_blur_rule(params, region, arity):
    # The Gaussian reads ceil(size) pixels and the alpha band ceil(edge_mult * size); the wider
    # of the two decides the padding.
    size = abs(float(params.get("edgeblur_size", 0.0)))
    band = size * max(0.0, float(params.get("edge_mult", 1.0)))
    support = max(0 if size < 0.5 else int(math.ceil(size)), 0 if band < 0.5 else int(math.ceil(band)))
    return [region.expand(support, support)] + [region] * (arity - 1)


_edge_extend_rule = _support_rule("extend_size")


def _light_wrap_support(params):
    # fg alpha blurred by fgblur then its inverse by wrap_diffuse (reaches add); bg by bgblur.
    def reach(name, default):
        size = abs(float(params.get(name, default)))
        return 0 if size < 0.5 else int(math.ceil(size))
    return max(reach("fgblur", 1.0) + reach("wrap_diffuse", 10.0), reach("bgblur", 4.0))


def _light_wrap_rule(params, region, arity):
    # Both the foreground and the background are read with the same padding, so they reach the
    # kernel as equal-shaped arrays; the mask gates the result and only needs the output region.
    support = _light_wrap_support(params)
    return [region.expand(support, support)] * 2 + [region] * (arity - 2)


def _dirblur_rule(params, region, arity):
    # Linear reaches ceil(length / 2) + 1 pixels each way (whole-pixel taps plus bilinear). Zoom
    # and radial depend on the centre and can read anywhere in the frame, so they ask for
    # "everything"; the region is clamped to the canvas by the caller. They are excluded from the
    # tile path anyway (tileexec.supports_tiled), so this only has to be safe, not tight.
    if params.get("blur_type", "linear") == "linear":
        length = abs(float(params.get("length", 0.0)))
        support = 0 if length <= 1.0 else int(math.ceil(length / 2.0)) + 1
    else:
        support = 1 << 20
    return [region.expand(support, support)] + [region] * (arity - 1)


def _drop_shadow_rule(params, region, arity):
    # The shadow at an output pixel comes from the alpha `distance` pixels away (rounded up) and
    # then `shadow_size` more through the blur, so the image is padded by their sum.
    distance = abs(float(params.get("distance", 0.0)))
    size = abs(float(params.get("shadow_size", 0.0)))
    support = int(math.ceil(distance)) + (0 if size < 0.5 else int(math.ceil(size)))
    return [region.expand(support, support)] + [region] * (arity - 1)


def _position_rule(params, region, arity):
    # A pixel at output (x, y) comes from (x - dx, y - dy) of the input.
    return [Region(region.x - int(params.get("translate_x", 0)), region.y - int(params.get("translate_y", 0)),
                   region.width, region.height)] * arity


def _defocus_rule(params, region, arity):
    # The disc reaches `defocus` pixels along x and `defocus / aspect` along y; the padding is the
    # larger of the two, rounded up, and zero below the kernel's own half-pixel cut-off.
    radius = abs(float(params.get("defocus", 0.0)))
    aspect = float(params.get("aspect", 1.0))
    reach = max(radius, radius / aspect if aspect > 0 else radius)
    support = 0 if reach < 0.5 else int(math.ceil(reach))
    return [region.expand(support, support)] + [region] * (arity - 1)


def _crop_rule(params, region, arity):
    # Crop zeroes everything outside its rectangle, so pixels outside it are generated, not read.
    rect = Region(int(params.get("x", 0)), int(params.get("y", 0)),
                  int(params.get("width", 0)), int(params.get("height", 0)))
    return [region.intersect(rect)] + [region] * (arity - 1)


def _transform_rule(params, region, arity):
    """Inverse-map the output rectangle's corners and take the bounding box.

    This mirrors `Evaluator._transform` exactly: src = center + R(-theta) * (dst - center -
    translate) / scale, with destination pixel centres at i + 0.5. Getting this wrong is invisible
    at tier 1 on an identity transform and catastrophic under rotation, which is why the golden
    test in the gate compares against a full-frame crop for every kernel rather than spot-checking.
    """
    if region.is_empty:
        return [region] + [region] * (arity - 1)
    theta = math.radians(float(params.get("rotate", 0.0)))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    scale = float(params.get("scale", 1.0))
    inv_scale = 1.0 / scale if scale != 0 else 1.0
    cx, cy = float(params.get("center_x", 0.0)), float(params.get("center_y", 0.0))
    tx, ty = float(params.get("translate_x", 0.0)), float(params.get("translate_y", 0.0))

    xs, ys = [], []
    for gx, gy in ((region.x, region.y), (region.right, region.y),
                   (region.x, region.bottom), (region.right, region.bottom)):
        ox, oy = gx - cx - tx, gy - cy - ty
        xs.append((ox * cos_t + oy * sin_t) * inv_scale + cx)
        ys.append((-ox * sin_t + oy * cos_t) * inv_scale + cy)

    # Filter support in source pixels. Nearest samples one pixel; bilinear reaches its two
    # neighbours; the cubic kernel reaches two on each side.
    support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(params.get("filter", "nearest"), 2)
    left, top = math.floor(min(xs)) - support, math.floor(min(ys)) - support
    right, bottom = math.ceil(max(xs)) + support, math.ceil(max(ys)) + support
    image = Region(int(left), int(top), int(right - left), int(bottom - top))
    return [image] + [region] * (arity - 1)


def _cornerpin_rule(params, region, arity):
    """Inverse-map the output rectangle's corners through the solved homography and take the
    bounding box -- the projective counterpart of `_transform_rule`, sharing its reasoning
    (straight edges stay straight, so a rectangle's extrema are still at its four corners).

    Kept independent of `imaging.Evaluator._cornerpin_forward_matrix` (this module stays free of
    NumPy) but solving the same 4-point DLT and inverting the same way, exactly as
    `_transform_rule` and `imaging._transformed_window` are kept in sync by hand rather than by
    a shared import.
    """
    if region.is_empty:
        return [region] + [region] * (arity - 1)
    from_pts = [(params[f"from{i}_x"], params[f"from{i}_y"]) for i in (1, 2, 3, 4)]
    to_pts = [(params[f"to{i}_x"], params[f"to{i}_y"]) for i in (1, 2, 3, 4)]
    a = [[0.0] * 8 for _ in range(8)]
    b = [0.0] * 8
    for i, ((x, y), (u, v)) in enumerate(zip(from_pts, to_pts)):
        a[2 * i] = [x, y, 1, 0, 0, 0, -x * u, -y * u]
        b[2 * i] = u
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v]
        b[2 * i + 1] = v
    h = _solve_linear_8(a, b)
    forward = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
    if params.get("direction", "forward") != "forward":
        forward = _invert_3x3(forward)
    inverse = _invert_3x3(forward)

    xs, ys = [], []
    for gx, gy in ((region.x, region.y), (region.right, region.y),
                   (region.x, region.bottom), (region.right, region.bottom)):
        denom = inverse[2][0] * gx + inverse[2][1] * gy + inverse[2][2]
        denom = denom if abs(denom) > 1e-9 else 1e-9
        xs.append((inverse[0][0] * gx + inverse[0][1] * gy + inverse[0][2]) / denom)
        ys.append((inverse[1][0] * gx + inverse[1][1] * gy + inverse[1][2]) / denom)
    support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(params.get("filter", "nearest"), 2)
    left, top = math.floor(min(xs)) - support, math.floor(min(ys)) - support
    right, bottom = math.ceil(max(xs)) + support, math.ceil(max(ys)) + support
    image = Region(int(left), int(top), int(right - left), int(bottom - top))
    return [image] + [region] * (arity - 1)


def _solve_linear_8(a, b):
    """Gaussian elimination with partial pivoting for the fixed 8x8 system `_cornerpin_rule`
    builds. A plain, dependency-free solver -- this module carries no NumPy -- for the same
    4-point DLT `imaging.Evaluator._solve_homography` solves with `np.linalg.solve`."""
    a = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = 8
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = (a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))) / a[row][row]
    return x


def _invert_3x3(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    det = det if abs(det) > 1e-12 else 1e-12
    inv = [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
           [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
           [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]
    return inv


def _merge_rule(params, region, arity):
    # Both layers are combined pixel-for-pixel, so both see the same request.
    return [region] * arity


def _switch_rule(params, region, arity):
    # Only the selected branch is evaluated. Returning None for the other is what keeps an
    # expensive unselected branch — a generative operator, later — from being scheduled at all.
    which = int(params.get("which", 0))
    return [region if index == which else None for index in range(arity)]


def _relight_rule(params, region, arity):
    # `image` is the one raster slot and reads pointwise, like Grade. `camera` and the light0..7
    # slots carry typed 3D values, not pixels — like Render3D's own scene/camera inputs, they are
    # requested whole (None) rather than windowed to a region.
    return [region] + [None] * (arity - 1)


REGION_RULES = {
    "Read": _generator,
    "Constant": _generator,
    "Checker": _generator,
    "Roto": _generator,
    # Ramp/Radial/Rectangle/Noise/Text (group c2 Draw generators) state their own format like
    # Roto/Constant/Checker, but the optional "image"/"mask" inputs -- when wired -- are pointwise:
    # the shape is composited over the same pixels it is asked to produce, no halo, so identity is
    # the right rule for both slots (unlike Roto, which declares no input slots at all).
    "Ramp": _identity,
    "Radial": _identity,
    "Rectangle": _identity,
    "Noise": _identity,
    "Text": _identity,
    "Grid": _identity,
    "Grade": _identity,
    "ColorCorrect": _identity,
    "Invert": _identity,
    "Clamp": _identity,
    "Multiply": _identity,
    "Add": _identity,
    "Gamma": _identity,
    "Saturation": _identity,
    "Exposure": _identity,
    "HueCorrect": _identity,
    "ColorMatrix": _identity,
    "Keyer": _identity,
    "HueKeyer": _identity,
    "Blur": _blur_rule,
    "Erode": _erode_rule,
    "Dilate": _dilate_rule,
    "Median": _median_rule,
    "Sharpen": _sharpen_rule,
    "Glow": _glow_rule,
    "Soften": _soften_rule,
    "Defocus": _defocus_rule,
    "DirBlur": _dirblur_rule,
    "DropShadow": _drop_shadow_rule,
    "EdgeBlur": _edge_blur_rule,
    "EdgeExtend": _edge_extend_rule,
    "LightWrap": _light_wrap_rule,
    "Dither": _identity,
    "Grain": _identity,
    "Posterize": _identity,
    "SoftClip": _identity,
    "HSVTool": _identity,
    "AddMix": _merge_rule,
    "Blend": _merge_rule,
    "CopyRectangle": _merge_rule,
    # Position, BlackOutside and AdjustBBox move or resize the data window, which the tile
    # executor's fixed-canvas model has no notion of, so like Mirror/Transform/Crop they are
    # excluded from the tile path (tiles.SUPPORTED_TILED_KINDS). Their own read is still declared.
    "Position": _position_rule,
    "BlackOutside": _identity,
    "AdjustBBox": _identity,
    # Mirror is coordinate-dependent on the canvas origin (like Transform/Crop) and is excluded
    # from the tile path entirely (see tiles.SUPPORTED_TILED_KINDS); its own region need is still
    # the identity so `input_regions` has a declared rule per docs/EVALUATION_TIERS.md clause C2.
    "Mirror": _identity,
    "Transform": _transform_rule,
    # A Tracker is a Transform whose transform is solved from tracks rather than typed in, so its
    # region rule is Transform's rule applied to the solved values. Sharing the function rather
    # than copying it is the point: an error in the inverse map cannot drift between the two.
    "Tracker": _transform_rule,
    "Crop": _crop_rule,
    "CornerPin": _cornerpin_rule,
    "STMap": _uv_lookup_rule,
    "IDistort": _uv_lookup_rule,
    "VectorBlur": _vector_blur_rule,
    # Reformat's own resize/fit math needs its *input's* display size, which this table's rules
    # never receive (only their own params and the requested region) -- every other rule here is
    # invariant to the input's actual size, so this is the one kind that genuinely cannot compute
    # a precise inverse map from params alone, short of joining DATA_DEPENDENT_RULES for a shape
    # this file's one call site (tileexec.py) and its own generic coverage test do not expect.
    # Excluded from the tile path entirely like Mirror before it (tiles.SUPPORTED_TILED_KINDS), so
    # this is never reached for real ROI scheduling; identity keeps `input_regions` total per
    # docs/EVALUATION_TIERS.md clause C2 without guessing a source size it does not have.
    "Reformat": _identity,
    "Shuffle": _identity,
    "ChannelShuffle": _identity,
    "Merge": _merge_rule,
    "Dissolve": _merge_rule,
    "Keymix": _merge_rule,
    "Copy": _merge_rule,
    "ChannelMerge": _merge_rule,
    "Difference": _merge_rule,
    "Premult": _identity,
    "Unpremult": _identity,
    "Dot": _identity,
    "NoOp": _identity,
    # TimeOffset/FrameHold/Retime (group c4) never move pixels within the frame -- only *which*
    # frame is sourced changes, which `imaging.Evaluator` resolves through a nested evaluate call,
    # not through this table -- so their own spatial ROI need is the identity, same as Dot's.
    "TimeOffset": _identity,
    "FrameHold": _identity,
    "Retime": _identity,
    "TimeClip": _identity,
    "FrameRange": _identity,
    "AppendClip": _identity,   # arity is the eight optional clip slots; none of them is read per tile
    "Switch": _switch_rule,
    "Relight": _relight_rule,
    "Viewer": _identity,
    "Write": _identity,
    # 3D values are not rasters: a texture is needed whole whatever region the render is asked
    # for, so geometry and scenes request their complete inputs.
    **{kind: (lambda params, region, arity: [None] * arity)
       for kind in ("ReadSplat3D", "ReadAlembic3D", "ReadAlembicCamera3D", "ReadUSD3D", "ReadUSDCamera3D", "ReadGLTF3D", "Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "ReadGeo3D", "Light3D", "Camera3D", "Project3D", "Scene3D", "WriteGeo3D", "WriteSplat3D",
                    "Render3D", "Axis3D", "TransformGeo3D", "MergeGeo3D", "Normals3D", "DisplaceGeo3D",
                    "ParticleEmitter3D", "ParticleCache3D", "ParticleGravity3D", "ParticleDrag3D",
                    "ParticleWind3D", "ParticleTurbulence3D", "ParticleBounce3D", "ParticleRender3D")},
}


# Kinds whose region rule cannot be evaluated from `params` alone because the geometry lives in
# `document["node_data"]` and depends on the frame. The scheduler must solve first and pass the
# result in. There is no default: an unsolved Tracker returning the whole frame would be the
# silent fall back clause C2 exists to forbid, and it would be invisible until a rotated
# stabilise pass started reading pixels nobody scheduled.
DATA_DEPENDENT_RULES = frozenset({"Tracker"})


def input_regions(kind: str, params: dict, region: Region, arity: int, solved: dict | None = None):
    """Map a requested output region onto the regions needed from each input slot.

    `arity` is the total number of declared slots (required + optional) so the returned list lines
    up positionally with the evaluator's own slot ordering.

    `solved` carries the frame-resolved geometry for the kinds in `DATA_DEPENDENT_RULES` — for a
    Tracker, the `SOLVED_TRANSFORM_FIELDS` produced by `nodebased.tracker.solve`.
    """
    try:
        rule = REGION_RULES[kind]
    except KeyError:
        raise UndeclaredRegionRule(
            f"{kind} has no region-of-interest rule; declare one in nodebased/tiers.py "
            f"(see docs/EVALUATION_TIERS.md clause C2)") from None
    if kind in DATA_DEPENDENT_RULES:
        if solved is None:
            raise UndeclaredRegionRule(
                f"{kind}'s region rule depends on node_data solved at the requested frame; "
                f"pass solved={{{', '.join(SOLVED_TRANSFORM_FIELDS)}}}")
        missing = [name for name in SOLVED_TRANSFORM_FIELDS if name not in solved]
        if missing:
            raise UndeclaredRegionRule(f"{kind} solved transform is missing {missing}")
        params = {**params, **solved}
    return rule(params, region, arity)


# --- Proxy tier ---------------------------------------------------------------------------------
#
# Parameters carrying pixel units, per node kind. Anything absent from this table is unitless
# (exposure, gamma, mix, rotate, scale) and must NOT be touched by a tier change.

PIXEL_UNIT_PARAMS = {
    "Constant": ("width", "height"),
    "Checker": ("width", "height", "size"),
    "Roto": ("width", "height"),
    "Ramp": ("width", "height", "p0_x", "p0_y", "p1_x", "p1_y"),
    "Radial": ("width", "height", "box_x", "box_y", "box_width", "box_height"),
    "Rectangle": ("width", "height", "box_x", "box_y", "box_width", "box_height"),
    "Noise": ("width", "height", "size"),
    "Grid": ("width", "height", "spacing_x", "spacing_y", "grid_offset_x", "grid_offset_y", "line_width"),
    "Text": ("width", "height", "font_size", "box_x", "box_y", "box_width", "box_height"),
    "Blur": ("radius",),
    "Erode": ("erode_size",), "Dilate": ("dilate_size",), "Median": ("median_size",),
    "Sharpen": ("sharpen_size",), "Glow": ("glow_size",), "Soften": ("soften_size",), "Defocus": ("defocus",), "DirBlur": ("length", "center_x", "center_y"),
    "DropShadow": ("distance", "shadow_size"),
    "EdgeBlur": ("edgeblur_size",), "EdgeExtend": ("extend_size",),
    "LightWrap": ("wrap_diffuse", "fgblur", "bgblur"),
    "Grain": ("red_size", "green_size", "blue_size"),
    "CopyRectangle": ("area_x", "area_y", "area_r", "area_t"),
    "Position": ("translate_x", "translate_y"), "AdjustBBox": ("numpixels",),
    "Transform": ("translate_x", "translate_y", "center_x", "center_y"),
    "Crop": ("x", "y", "width", "height"),
    # Reformat's "scale" is a unitless ratio (like Transform's own "scale"), not a pixel count, so
    # it is deliberately absent here; width/height are the format's own pixel size and are the
    # only state a named-format preset resolves into (see core._resolve_reformat_format), which is
    # what keeps a proxy-tier playback of a named format correct.
    "Reformat": ("width", "height"),
    "CornerPin": ("from1_x", "from1_y", "from2_x", "from2_y", "from3_x", "from3_y",
                 "from4_x", "from4_y", "to1_x", "to1_y", "to2_x", "to2_y",
                 "to3_x", "to3_y", "to4_x", "to4_y"),
    "Render3D": ("width", "height"),
    # 3D positions and sizes are world units, never pixels: only Render3D scales with the tier.
}

# Multipliers and caps applied to per-pixel data measured in full-resolution pixels (step 5c):
# IDistort's uv scale, VectorBlur's vector scale and length cap. Divided by the tier, not
# rounded, and kept apart from PIXEL_UNIT_PARAMS because their names ("scale") are not pixel
# units on other nodes.
TIER_DIVIDED_PARAMS = {
    "IDistort": ("uv_scale_x", "uv_scale_y"),
    "VectorBlur": ("vector_scale", "max_length"),
}

# Extents may never round down to nothing: a 1-pixel-wide crop at tier 4 stays 1 pixel rather than
# collapsing to an empty image and silently changing the graph's meaning.
_MINIMUM_ONE = {"width", "height", "size"}


def scale_params(kind: str, params: dict, tier: int) -> dict:
    """Return `params` expressed at the given proxy tier.

    Offsets scale and round to nearest; extents scale and clamp to at least one pixel; continuous
    values keep their fractional part so sub-pixel transforms stay sub-pixel.
    """
    tier = int(tier)
    if tier == 1:
        return params
    if tier not in PROXY_TIERS:
        raise ValueError(f"Unsupported proxy tier {tier}; expected one of {PROXY_TIERS}")
    names = PIXEL_UNIT_PARAMS.get(kind)
    divided = TIER_DIVIDED_PARAMS.get(kind)
    if divided:
        # uv offsets and vector lengths are read from image data in full-resolution pixels; at a
        # proxy tier the image is smaller, so the factors that turn that data into pixels shrink.
        params = {**params, **{name: params[name] / tier for name in divided if name in params}}
    if not names:
        return params
    scaled = dict(params)
    for name in names:
        if name not in scaled:
            continue
        value = scaled[name] / tier
        if isinstance(params[name], int):
            # Extents round *outward*, offsets to nearest. Ceiling the extents is what keeps a
            # generated source the same size as a decimated Read at the same tier — both land on
            # ceil(n / tier) — so a Merge between them still sees matching formats.
            value = max(1, -(-params[name] // tier)) if name in _MINIMUM_ONE else int(round(value))
        scaled[name] = value
    return scaled


def scale_node_data(kind: str, payload, tier: int):
    """Return a `node_data` payload expressed at the given proxy tier.

    Clause C3 is usually discussed as a parameter problem, but from schema v8 onward pixel units
    also live in structured payloads: roto point positions, their tangent handles, feather radii
    and track positions are all in pixels. A shape left unscaled at tier 2 keys the wrong quarter
    of the frame, which is a worse outcome than a slow viewer, so it is scaled in the same pass.

    Only the base value and any curve key values move; frames, names, modes and interpolation are
    untouched. A curve on a pixel-unit scalar is therefore still a curve, just in tier space.
    """
    tier = int(tier)
    if tier == 1 or payload is None:
        return payload
    if tier not in PROXY_TIERS:
        raise ValueError(f"Unsupported proxy tier {tier}; expected one of {PROXY_TIERS}")
    from .shapes import NODE_DATA_SCHEMA, PIXEL_UNIT_SCALARS

    slot = NODE_DATA_SCHEMA.get(kind)
    if slot is None or slot not in payload:
        return payload

    def scaled_scalar(value):
        if isinstance(value, dict):
            curve = value.get("curve")
            out = {"value": value["value"] / tier}
            if "curve" in value:
                out["curve"] = curve if curve is None else {
                    "interpolation": curve["interpolation"],
                    "keys": [{"frame": key["frame"], "value": key["value"] / tier}
                             for key in curve["keys"]]}
            return out
        return value / tier

    items = []
    for item in payload[slot]:
        entry = {}
        for name, value in item.items():
            if name == "points":
                entry[name] = [{field: scaled_scalar(scalar) if field in PIXEL_UNIT_SCALARS else scalar
                                for field, scalar in point.items()} for point in value]
            elif name in PIXEL_UNIT_SCALARS:
                entry[name] = scaled_scalar(value)
            else:
                entry[name] = value
        items.append(entry)
    return {slot: items}


def tier_of(document_or_tier, default: int = 1) -> int:
    """Read the proxy tier off a document, tolerating documents written before tiers existed."""
    if isinstance(document_or_tier, int):
        return document_or_tier
    if isinstance(document_or_tier, dict):
        return int(document_or_tier.get("proxy", {}).get("tier", default))
    return default
