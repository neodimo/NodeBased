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

# Proxy tiers are linear downscale divisors. 1 is full resolution.
PROXY_TIERS = (1, 2, 4)

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


def _merge_rule(params, region, arity):
    # Both layers are combined pixel-for-pixel, so both see the same request.
    return [region] * arity


def _switch_rule(params, region, arity):
    # Only the selected branch is evaluated. Returning None for the other is what keeps an
    # expensive unselected branch — a generative operator, later — from being scheduled at all.
    which = int(params.get("which", 0))
    return [region if index == which else None for index in range(arity)]


REGION_RULES = {
    "Read": _generator,
    "Constant": _generator,
    "Checker": _generator,
    "Grade": _identity,
    "ColorCorrect": _identity,
    "Blur": _blur_rule,
    "Transform": _transform_rule,
    "Crop": _crop_rule,
    "Shuffle": _identity,
    "Merge": _merge_rule,
    "Premult": _identity,
    "Unpremult": _identity,
    "Dot": _identity,
    "Switch": _switch_rule,
    "Viewer": _identity,
}


def input_regions(kind: str, params: dict, region: Region, arity: int):
    """Map a requested output region onto the regions needed from each input slot.

    `arity` is the total number of declared slots (required + optional) so the returned list lines
    up positionally with the evaluator's own slot ordering.
    """
    try:
        rule = REGION_RULES[kind]
    except KeyError:
        raise UndeclaredRegionRule(
            f"{kind} has no region-of-interest rule; declare one in nodebased/tiers.py "
            f"(see docs/EVALUATION_TIERS.md clause C2)") from None
    return rule(params, region, arity)


# --- Proxy tier ---------------------------------------------------------------------------------
#
# Parameters carrying pixel units, per node kind. Anything absent from this table is unitless
# (exposure, gamma, mix, rotate, scale) and must NOT be touched by a tier change.

PIXEL_UNIT_PARAMS = {
    "Constant": ("width", "height"),
    "Checker": ("width", "height", "size"),
    "Blur": ("radius",),
    "Transform": ("translate_x", "translate_y", "center_x", "center_y"),
    "Crop": ("x", "y", "width", "height"),
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


def tier_of(document_or_tier, default: int = 1) -> int:
    """Read the proxy tier off a document, tolerating documents written before tiers existed."""
    if isinstance(document_or_tier, int):
        return document_or_tier
    if isinstance(document_or_tier, dict):
        return int(document_or_tier.get("proxy", {}).get("tier", default))
    return default
