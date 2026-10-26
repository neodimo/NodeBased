"""Nuke's Viewer knobs (fidelity step G): the settings that change how the artist sees and steps through the comp.

They live in `settings.viewer.knobs`, saved with the document like the viewer's look, region of interest, proxy and masks, and
stored only when one differs from `DEFAULT`, so an older document keeps its exact serialized form. Everything here is display
and transport only: no knob changes a pixel the graph produces or a Write renders.

What each knob does here, as read from the guide's one-line descriptions (it states no formulas, and NodeBased has compared
none with a Nuke render):

* `frame_range` ("first-last", empty for none) and `frame_range_lock`: playback, the jump-to-start and jump-to-end buttons and
  the increment steps stay inside this range, cut down to the project's range. With the lock off the range is kept but not in
  force, and the viewer follows the project's range.
* `frame_increment`: the `<<` and `>>` steps jump this many frames, to the edge of the viewer's range at most.
* `rgb_only` ("apply LUT to colour channels only"): on, the alpha channel is shown as it is. Off, gain, gamma and the display
  transform are applied to it as well. NodeBased has always shown alpha untouched, so a document without the knob has it on,
  where Nuke's own default is off.
* `ignore_pixel_aspect`: on, the picture is drawn with square pixels. Off, a format whose pixel aspect is not 1 (the pixel
  aspect of the nearest Reformat upstream of the viewed node) is drawn that much wider. NodeBased has always drawn square pixels,
  so a document without the knob has it on, where Nuke's own default is off. The Direct picture draws square pixels only: for
  a format with another pixel aspect it hands the picture back to the Viewer.
* `zoom_lock` ("prevent auto zoom"): a picture of another size no longer refits the view.
* `show_overscan` and `overscan`: the format's edge is outlined and, `overscan` pixels beyond it, so is the margin a node may
  have made pixels in (Undistort does); the first fit leaves room for the margin.
* `full_frame_processing`: the viewer evaluates the whole frame instead of only the part of it the zoomed view shows.
* `safe_zone`: title safe (80 %) and action safe (90 %) outlines, the long-standing television convention.
* `format_center`: a cross at the centre of the format.
* `center_fstop` and `shutter_angle`: the viewer gain is a number of stops; this reads it as an f-number about `center_fstop`
  (every stop of gain is one stop of aperture) and the shutter as `shutter_angle` degrees of a turn at the project's rate. They
  only label the gain control.
* `use_gpu_for_viewer`: off, the picture is converted on the processor and never goes to the Direct picture as floats.
"""
from __future__ import annotations

import math
import re

SAFE_ZONES = ("no guides", "title safe", "action safe", "title and action safe")
# Title safe and action safe, as a fraction of the format's width and height.
TITLE_SAFE = 0.80
ACTION_SAFE = 0.90
DEFAULT = {"frame_range": "", "frame_range_lock": True, "frame_increment": 10, "rgb_only": True, "ignore_pixel_aspect": True,
           "zoom_lock": False, "show_overscan": False, "overscan": 100, "full_frame_processing": False,
           "safe_zone": "no guides", "format_center": False, "center_fstop": 8.0, "shutter_angle": 180.0,
           "use_gpu_for_viewer": True}
BOOLEANS = ("frame_range_lock", "rgb_only", "ignore_pixel_aspect", "zoom_lock", "show_overscan", "full_frame_processing",
            "format_center", "use_gpu_for_viewer")
INTEGERS = {"frame_increment": (1, 100000), "overscan": (0, 100000)}
FLOATS = {"center_fstop": (0.5, 128.0), "shutter_angle": (1.0, 720.0)}
_RANGE = re.compile(r"^\s*(-?\d+)\s*(?:[-:,\s]\s*(-?\d+))?\s*$")


def parse_range(text):
    """(first, last) of a frame range typed as `10-20`, `10 20`, `10,20` or `12`, else None."""
    match = _RANGE.match(text or "")
    if match is None:
        return None
    first = int(match.group(1))
    last = int(match.group(2)) if match.group(2) is not None else first
    return (first, last) if first <= last else (last, first)


def validate(knobs):
    """Raise ValueError unless `knobs` defines exactly the Viewer knobs, each with a value in its range."""
    if not isinstance(knobs, dict) or set(knobs) != set(DEFAULT):
        raise ValueError(f"settings.viewer.knobs must define {sorted(DEFAULT)}")
    for name in BOOLEANS:
        if type(knobs[name]) is not bool:
            raise ValueError(f"settings.viewer.knobs.{name} must be boolean")
    for name, (low, high) in INTEGERS.items():
        if type(knobs[name]) is not int or not low <= knobs[name] <= high:
            raise ValueError(f"settings.viewer.knobs.{name} must be a whole number from {low} to {high}")
    for name, (low, high) in FLOATS.items():
        value = knobs[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"settings.viewer.knobs.{name} must be a number from {low} to {high}")
    text = knobs["frame_range"]
    if type(text) is not str or len(text) > 64 or (text.strip() and parse_range(text) is None):
        raise ValueError("settings.viewer.knobs.frame_range must be empty or a range such as 10-20")
    if knobs["safe_zone"] not in SAFE_ZONES:
        raise ValueError(f"settings.viewer.knobs.safe_zone must be one of {list(SAFE_ZONES)}")


def knobs_of(document):
    """The document's Viewer knobs, with the defaults where it stores none."""
    return {**DEFAULT, **document["settings"]["viewer"].get("knobs", {})}


def normalised(knobs):
    return {**knobs, "center_fstop": float(knobs["center_fstop"]), "shutter_angle": float(knobs["shutter_angle"])}


# ------------------------------------------------------------------------------------------------------------ time
def playback_range(document):
    """(first, last) the viewer plays and steps inside: its own range cut down to the project's, else the project's."""
    time = document["time"]
    first, last = int(time["first"]), int(time["last"])
    knobs = knobs_of(document)
    typed = parse_range(knobs["frame_range"]) if knobs["frame_range_lock"] else None
    if typed is None:
        return first, last
    low, high = max(typed[0], first), min(typed[1], last)
    return (low, high) if low <= high else (first, last)


def step_target(document, direction):
    """The frame a `<<` (-1) or `>>` (+1) step from the playhead lands on: `frame_increment` frames along, held at the range's edge."""
    first, last = playback_range(document)
    step = int(knobs_of(document)["frame_increment"]) * (-1 if direction < 0 else 1)
    return min(max(int(document["time"]["current"]) + step, first), last)


# ------------------------------------------------------------------------------------------------------- the picture
def pixel_aspect(document, target):
    """The pixel aspect of the format the viewed node carries: that of the nearest Reformat upstream of it (itself included), else 1."""
    nodes = document["nodes"]
    seen = set()
    while isinstance(target, str) and target in nodes and target not in seen:
        seen.add(target)
        node = nodes[target]
        if node["type"] == "Reformat" and not node.get("disabled"):
            return float(node["params"].get("pixel_aspect", 1.0))
        target = (node.get("inputs") or {}).get("image")
    return 1.0


def stretch(document, target=None):
    """How much wider than tall the viewer draws a pixel of the viewed picture."""
    if knobs_of(document)["ignore_pixel_aspect"]:
        return 1.0
    return pixel_aspect(document, document.get("view") if target is None else target)


def safe_rects(width, height, zone):
    """The outlines of a `safe_zone` choice inside a `width` x `height` format, as (x0, y0, x1, y1) in pixels, outermost first."""
    fractions = {"title safe": (TITLE_SAFE,), "action safe": (ACTION_SAFE,),
                 "title and action safe": (ACTION_SAFE, TITLE_SAFE)}.get(zone, ())
    return [tuple(round(value, 6) for value in ((1 - f) * width / 2, (1 - f) * height / 2, (1 + f) * width / 2, (1 + f) * height / 2))
            for f in fractions]


def overscan_rect(width, height, pixels):
    """The format's rectangle grown by `pixels` on every side."""
    return (-pixels, -pixels, width + pixels, height + pixels)


def exposure_label(gain, knobs, fps):
    """`f/8.0 · 180° · 1/48 s`: the viewer gain (stops) read as an f-number about the centre f-stop, and the shutter."""
    f_number = float(knobs["center_fstop"]) / (math.sqrt(2.0) ** float(gain))
    angle = float(knobs["shutter_angle"])
    seconds = angle / 360.0 / max(float(fps), 1e-6)
    return f"f/{f_number:.1f} · {angle:g}° · 1/{1.0 / seconds:.0f} s"
