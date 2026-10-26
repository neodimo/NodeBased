"""Nuke's CornerPin2D knobs (fidelity step G): per-corner enables, the extra matrix, clamp, black outside, motion blur and the
copy buttons.

Pure Python (no NumPy), so the evaluator, the region rule in `tiers` and the tests share one reading of them.

The readings of the guide's one-line knob descriptions (the guide states no formula for any of them, and NodeBased has compared
none with a Nuke render):

* `enable1` to `enable4`: a corner that is switched off is not pinned. Its `to` point is taken to be its `from` point, so the
  image is held there while the other corners move.
* `extra matrix` (`transform_matrix`): a 3 x 3 projective matrix, `extra_matrix_RC` for row R, column C, applied to the pinned
  picture after the pin itself, in canvas pixels. Nuke's knob is 4 x 4; a 2D warp reads its x, y and w rows and columns, which is
  what the nine values here are. The identity is no change.
* `black_outside`: on (the default, and what a CornerPin always did), the area the warp does not cover is transparent black. Off,
  the picture's outermost pixels are carried outward over that area.
* `clamp`: the filtered result is held inside the range of the input, per channel, so a cubic filter cannot overshoot the
  highlights and shadows of a hard edge.
* `motionblur`, `shutter`, `shutter_offset`, `custom_offset`: the Transform node's shutter blur, over `samples` frames.
* `Copy 'from'` (`copy_from_to`) copies the four `from` points into the `to` points; `Copy 'to'` (`copy_from_to#2`) does the
  reverse. `Set to input` (`set_to_input`) puts the `from` points on the corners of the input's format, in this node's own
  corner order: 1 top left, 2 top right, 3 bottom left, 4 bottom right.
"""
from __future__ import annotations

CORNERS = (1, 2, 3, 4)
IDENTITY = {f"extra_matrix_{r}{c}": (1.0 if r == c else 0.0) for r in range(3) for c in range(3)}
MATRIX_NAMES = tuple(IDENTITY)
BLUR = {"motionblur": 0, "samples": 4, "shutter": 0.5, "shutter_offset": "start", "custom_offset": 0.0}
# What a new CornerPin gets (Nuke's defaults where the guide states them) and what an older document's CornerPin gets, which is
# what it did before the knobs existed. Only the shutter's default differs between the two, and it has no effect while the
# blur is off.
NEW = {**{f"enable{i}": 1 for i in CORNERS}, **IDENTITY, "clamp": 0, "black_outside": 1, **BLUR}
OLD = dict(NEW, shutter=1.0, shutter_offset="centred")
KINDS = ("CornerPin",)
# Entries of the extra matrix that hold a position (pixels) or a perspective term (1 / pixel): they scale with the proxy tier.
POSITION_ENTRIES = ("extra_matrix_02", "extra_matrix_12")
PERSPECTIVE_ENTRIES = ("extra_matrix_20", "extra_matrix_21")


def new_params(kind):
    return dict(NEW) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(OLD) if kind in KINDS else {}


def points(p, prefix):
    return [(float(p[f"{prefix}{i}_x"]), float(p[f"{prefix}{i}_y"])) for i in CORNERS]


def effective_to_points(p):
    """The four `to` points the pin uses: a corner whose enable is off keeps its `from` point."""
    return [(float(p[f"from{i}_x"]), float(p[f"from{i}_y"])) if not int(p.get(f"enable{i}", 1))
            else (float(p[f"to{i}_x"]), float(p[f"to{i}_y"])) for i in CORNERS]


def extra_matrix(p):
    """The extra matrix as three rows of three numbers."""
    return [[float(p.get(f"extra_matrix_{r}{c}", 1.0 if r == c else 0.0)) for c in range(3)] for r in range(3)]


def has_extra(p):
    return any(float(p.get(name, value)) != value for name, value in IDENTITY.items())


def multiply(a, b):
    """The 3 x 3 product `a` times `b`, as rows."""
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def after_extra(forward, p):
    """`forward` (pre-warp point to post-warp point) followed by the extra matrix."""
    return multiply(extra_matrix(p), forward) if has_extra(p) else forward


# ------------------------------------------------------------------------------------------------------- the buttons
def copy_from_into_to(p):
    """The updates of `Copy 'from'`: every `to` point becomes its `from` point."""
    return {f"to{i}_{axis}": p[f"from{i}_{axis}"] for i in CORNERS for axis in "xy"}


def copy_to_into_from(p):
    """The updates of `Copy 'to'`: every `from` point becomes its `to` point."""
    return {f"from{i}_{axis}": p[f"to{i}_{axis}"] for i in CORNERS for axis in "xy"}


def from_input_corners(x, y, width, height):
    """The updates of `Set to input`, for an input whose format is the rectangle (x, y, width, height)."""
    x0, y0, x1, y1 = float(x), float(y), float(x + width), float(y + height)
    return {"from1_x": x0, "from1_y": y0, "from2_x": x1, "from2_y": y0,
            "from3_x": x0, "from3_y": y1, "from4_x": x1, "from4_y": y1}


def button_commands(node_id, updates):
    """Document commands that set `updates` on the node: one `set` each, as the panel's own edits do."""
    return [{"op": "set", "id": node_id, "param": name, "value": value} for name, value in updates.items()]
