"""Nuke's AddTimeCode, AppendClip and BurnIn knobs of fidelity step H.

Every default leaves an older node as it was; the guide's own defaults differ for several, and the mapping records that.

AddTimeCode
  * `metafps` (Nuke's "get FPS from metadata", off here where Nuke has it on): the frame rate comes from the input's metadata key
    `input/frame_rate` (or `exr/framesPerSecond`, a number or a fraction such as `24000/1001`) when it has one, else from `fps`.
  * `use_frame` (Nuke's `useFrame`, "use start frame?"; on here where Nuke has it off): on, `timecode` is the time code of
    `start_frame`, as it always was; off, it is the time code of the document's first frame.

AppendClip
  * `fade_in`, `fade_out` (Nuke's `fadeIn`, `fadeOut`): frames at the head and tail of the whole output that fade up from and down to
    black, `(k + 1) / (fade + 1)` at the k-th frame from that end (the retiming nodes' reading, nodebased/timeknobs.py).
  * `meta_from_first`: the metadata of the output is that of the first clip's first frame rather than of the clip being shown.
  * `last_frame(...)`: Nuke's `lastFrame`, the output's last frame, computed from the clips' lengths and the dissolve.

BurnIn
  * `opacity`: the whole overlay, text and bars, is blended at this strength.
  * `margin` is Nuke's `x-padding` (the distance of the text from the left and right edges); `y_padding` (Nuke's `y-padding`) is its
    distance from the top and bottom edges. A bar covers its text band plus that padding, from the edge.
  * `text_scale`: the font size is multiplied by it. `font_style`: `Regular`, `Bold`, `Italic`, `Bold Italic` or a face's own style
    name, as in the Text node.
  * `top_middle`, `bottom_middle`: two more text slots, centred, in the top and bottom bands.
  * `bar_red`, `bar_green`, `bar_blue`, `bar_alpha`: the bars' colour (black by default; `bar_alpha` multiplies `bar_opacity`).
  * `bar_y_border` grows each bar by that many pixels beyond its text band. `bar_fit` off keeps the full-width bars; on, a box is
    drawn around the text of each slot instead, as Nuke's background does, grown by `bar_x_border` and `bar_y_border` on every side.
"""
from __future__ import annotations

import re

import numpy as np

ADD_TIME_CODE = {"metafps": 0, "use_frame": 1}
APPEND_CLIP = {"fade_in": 0, "fade_out": 0, "meta_from_first": 0}
BURN_IN = {"opacity": 1.0, "y_padding": 0.0, "text_scale": 1.0, "font_style": "Regular", "top_middle": "", "bottom_middle": "",
           "bar_red": 0.0, "bar_green": 0.0, "bar_blue": 0.0, "bar_alpha": 1.0, "bar_x_border": 0.0, "bar_y_border": 0.0,
           "bar_fit": 0}
BY_KIND = {"AddTimeCode": ADD_TIME_CODE, "AppendClip": APPEND_CLIP, "BurnIn": BURN_IN}
FPS_KEYS = ("input/frame_rate", "exr/framesPerSecond")
FONT_STYLES = ("Regular", "Bold", "Italic", "Bold Italic")


def new_params(kind):
    return dict(BY_KIND.get(kind, {}))


def upgrade_params(kind):
    return dict(BY_KIND.get(kind, {}))


# ----------------------------------------------------------------------------------------------------- AddTimeCode
def parse_rate(text):
    """A frame rate from metadata text: a number or a fraction. None when the text is no usable rate."""
    try:
        match = re.fullmatch(r"\s*([0-9.]+)\s*(?:/\s*([0-9.]+))?\s*", str(text))
        if not match:
            return None
        rate = float(match.group(1)) / (float(match.group(2)) if match.group(2) else 1.0)
        return rate if 0.0 < rate <= 1000.0 else None
    except (ValueError, ZeroDivisionError):
        return None


def time_code_rate(params, meta):
    """The frame rate the AddTimeCode counts at: the input's metadata when `metafps` is on and it states one, else `fps`."""
    if int(params.get("metafps", 0)):
        for key in FPS_KEYS:
            rate = parse_rate((meta or {}).get(key))
            if rate is not None:
                return rate
    return float(params["fps"])


def time_code_start_frame(params):
    """The frame `timecode` belongs to: `start_frame`, or with `use_frame` off the document's first (`_first`, added by the
    evaluator)."""
    if int(params.get("use_frame", 1)):
        return int(params["start_frame"])
    return int(params.get("_first", params["start_frame"]))


# ------------------------------------------------------------------------------------------------------ AppendClip
def output_range(first_frame, lengths, overlaps):
    """(first, last) frame of an AppendClip's output: it starts at `first_frame`, each clip adds its length less the overlap with
    the next one."""
    last = int(first_frame)
    for length, overlap in zip(lengths, overlaps):
        last += length - overlap
    return int(first_frame), last - 1


def fade_gain(frame, first, last, fade_in, fade_out):
    """The multiplier on the output at `frame` (inside [first, last]); outside the range nothing fades."""
    if not (first <= frame <= last):
        return 1.0
    gain = 1.0
    position, count = frame - first, last - first + 1
    if fade_in > 0 and position < fade_in:
        gain *= (position + 1) / (fade_in + 1)
    if fade_out > 0 and count - position <= fade_out:
        gain *= (count - position) / (fade_out + 1)
    return gain


# --------------------------------------------------------------------------------------------------------- BurnIn
def band_rows(p, height):
    """(size, band, top rows, bottom rows): the text size after `text_scale`, the band a text line sits in, and where the top and
    bottom bars reach from their edge (the band plus the y padding plus the bar's y border)."""
    size = float(p["font_size"]) * float(p.get("text_scale", 1.0))
    band = max(1, int(round(size * 2.0)))
    pad = float(p.get("y_padding", 0.0))
    reach = int(round(pad + band + (0.0 if int(p.get("bar_fit", 0)) else float(p.get("bar_y_border", 0.0)))))
    return size, band, pad, max(1, reach)


def text_extent(coverage):
    """(x0, y0, x1, y1) of the non-empty part of a premultiplied (h, w, 4) text picture, or None when it is empty."""
    alpha = coverage[..., 3]
    columns = np.flatnonzero(alpha.max(axis=0) > 0.0)
    rows = np.flatnonzero(alpha.max(axis=1) > 0.0)
    if columns.size == 0 or rows.size == 0:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1
