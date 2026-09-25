"""Display-only A/B compositing for the 2D Viewer.

Everything here works on frames the graph has already evaluated (scene-linear float arrays,
height x width x 4) and on screen coordinates. Nothing calls the evaluator, so dragging the wipe or
switching the compare mode never re-evaluates the graph.
"""
import math

import numpy as np

from .core import COMPARE_MODES, viewer_state  # noqa: F401  (COMPARE_MODES re-exported for UI and tests)

# The wipe is stored as fractions of the format rectangle so it survives zoom and format changes.
# angle is in degrees; 0 is a vertical line with A on the left and B on the right.
WIPE_DEFAULT = {"x": 0.5, "y": 0.5, "angle": 0.0}
COMBINED_MODES = ("over", "under", "minus", "difference")


def active_b(doc):
    """(mode, B node id) for a document, with B None when the compare is off: "A only", no B
    input chosen, or the chosen input is empty."""
    state = viewer_state(doc)
    if state["compare"] == "A only" or state["b"] is None:
        return "A only", None
    node = state["inputs"][state["b"] - 1]
    return (state["compare"], node) if node is not None else ("A only", None)


def wipe_normal(angle):
    """Unit vector pointing from the A side into the B side."""
    radians = math.radians(angle)
    return math.cos(radians), math.sin(radians)


def wipe_direction(angle):
    """Unit vector along the line; at 0 degrees it points up the screen (y grows downward)."""
    radians = math.radians(angle)
    return math.sin(radians), -math.cos(radians)


def wipe_side(px, py, cx, cy, angle):
    """'B' when the point lies on B's side of the wipe line, otherwise 'A'."""
    nx, ny = wipe_normal(angle)
    return "B" if (px - cx) * nx + (py - cy) * ny > 0 else "A"


def wipe_angle_from_drag(cx, cy, px, py):
    """Angle that puts the rotation handle (on the line direction) under the pointer."""
    dx, dy = px - cx, py - cy
    if math.hypot(dx, dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, -dy))


def combine(mode, a, b):
    """The composite the Viewer shows for over, under, minus and difference. `a` and `b` are
    float arrays of the same shape ending in 4 channels (a single pixel works too)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if mode == "over":
        return a + b * (1.0 - a[..., 3:4])
    if mode == "under":
        return b + a * (1.0 - b[..., 3:4])
    if mode in ("minus", "difference"):
        rgb = a[..., :3] - b[..., :3]
        if mode == "difference":
            rgb = np.abs(rgb)
        alpha = np.maximum(a[..., 3:4], b[..., 3:4])
        return np.concatenate([rgb, alpha], axis=-1)
    raise ValueError(f"combine: {mode!r} is not a compositing mode")


def align(shape, region_origin, other, other_origin):
    """Place `other` (its top-left at `other_origin`, in the same pixel space as `region_origin`)
    on a transparent canvas of `shape`, so two buffers with different data windows still line up
    pixel for pixel. Returns `other` itself when it already fits exactly."""
    height, width = shape[:2]
    dx, dy = other_origin[0] - region_origin[0], other_origin[1] - region_origin[1]
    if other.shape[:2] == (height, width) and dx == 0 and dy == 0:
        return other
    canvas = np.zeros((height, width, 4), dtype=np.float32)
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(width, dx + other.shape[1]), min(height, dy + other.shape[0])
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = other[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return canvas


def pixel_at(mode, a, b, x, y, side="A"):
    """The values the pixel readout shows at array position (x, y): the buffer under the pointer.
    `side` says which half of the wipe the pointer is on ("A" or "B"); it only matters for wipe."""
    if b is None or mode == "A only":
        return a[y, x], "A"
    if mode == "B only" or (mode == "wipe" and side == "B"):
        return b[y, x], "B"
    if mode in COMBINED_MODES:
        return combine(mode, a[y, x], b[y, x]), mode
    return a[y, x], "A"
