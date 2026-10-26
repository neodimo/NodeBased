"""Nuke's Tracker knobs (fidelity step G): the matching options, the transform modes, the live-link transform, the per-track
toggles and the buttons that edit the track list.

What each knob does here, as read from the guide's one-line descriptions (it states no formulas, and NodeBased has compared none
with a Nuke render):

Matching (used by `tracker.match_pattern` and `tracker.analyse`)
* `pretrack_filter`: `median` (a 3 x 3 median on the analysis image, which takes salt and pepper noise out), `adjust contrast`
  (the image stretched to its own darkest and brightest value) or `none`. Both the reference and the frame being matched get it.
* `adjust_for_luminance_changes`: on matches the pattern after its own mean and spread are removed (normalised cross-correlation),
  so a change of exposure does not move the match; off compares the levels themselves (squared difference).
* `clamp_footage`: clamps the analysis image to 0..1, so a super-white or sub-zero value cannot dominate the match.
* `max_error`: tracking stops at the first frame whose error (one minus the match score) is above it; the frames before it are
  kept. At 1 or more it is off and the older rule applies: a score under 0.5 is an error and nothing is kept.
* `predict_track`: each search is centred on where the motion of the last two frames carries the point.
* `grab_behavior`: when the pattern is grabbed again from the frame just tracked: `on first frame` (never again), `every frame`,
  `every n frames` (`grab_interval`), `if error above` (`grab_error_above`) or `if error below` (`grab_error_below`).
* `snap_to_markers`: a picked point moves to the centre of the marker near it: the contrast-weighted centroid of the 9 x 9 pixels about it.

Solving (used by `tracker.solve`)
* `mode` is Nuke's `transform`: `none` (the transform knobs below are the transform), `match_move`, `stabilise`, the `1pt`
  forms (the first usable track alone gives translation), `remove_jitter` and `add_jitter` (the track against its own average
  over `jitter_period` frames: removing it moves the picture along the average path, adding it moves the picture along the shake).
* `livelink_transform`: on, the solved transform is the node's transform. Off, the knobs `translate`, `rotate`, `scale`,
  `skew_x`, `skew_y`, `skew_order` and `center` are, exactly as they would be on a Transform.
* Per track: `use_t`, `use_r` and `use_s` (Nuke's T, R and S) choose which tracks give translation, rotation and scale (a track
  with none of them set is all three, as every older track was); `pattern_radius` and `search_radius` give a track its own areas.

Buttons: `add track`, `delete track`, `select all`, `average tracks` edit the list of tracks; `Export CornerPin2D` is the node's
existing export.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PRETRACK_FILTERS = ("none", "adjust contrast", "median")
GRAB_BEHAVIOURS = ("on first frame", "every frame", "every n frames", "if error above", "if error below")
MODES = ("match_move", "stabilise", "none", "match_move_1pt", "stabilise_1pt", "remove_jitter", "add_jitter")
SKEW_ORDERS = ("XY", "YX")
SNAP_RADIUS = 4
MAX_TRACKS = 16
# The Transform knobs a Tracker takes (its own transform when the live link is off or the mode is none).
TRANSFORM = {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0, "scale": 1.0, "skew_x": 0.0, "skew_y": 0.0, "skew_order": "XY",
             "center_x": 0.0, "center_y": 0.0, "clamp": 0, "black_outside": 0, "motionblur": 0, "samples": 4, "shutter": 0.5,
             "shutter_offset": "start", "custom_offset": 0.0}
ANALYSIS = {"pretrack_filter": "median", "adjust_for_luminance_changes": 1, "max_error": 0.2, "clamp_footage": 1,
            "predict_track": 0, "grab_behavior": "if error above", "grab_interval": 0, "grab_error_above": 0.05,
            "grab_error_below": 0.0, "snap_to_markers": 0, "hide_progress_bar": 0, "show_error_on_track_links": 1,
            "create_key_on_move": 1, "autotracks_delete_keyframes": 1}
SOLVE = {"jitter_period": 10, "livelink_transform": 1}
# What an older document's Tracker did: no filter, the luminance-proof match, the score-under-0.5 rule, no clamp, the pattern
# kept (or `adaptive_update`), the picture following the tracks, the paths coloured by error.
OLD = {**TRANSFORM, **ANALYSIS, **SOLVE, "pretrack_filter": "none", "adjust_for_luminance_changes": 1, "max_error": 1.0,
       "clamp_footage": 0, "grab_behavior": "on first frame", "show_error_on_track_links": 1, "shutter": 1.0, "shutter_offset": "centred"}
# What a new Tracker gets: Nuke's defaults where the guide states them, except `adjust_for_luminance_changes` (on: NodeBased's matching has
# always been the luminance-proof one, and the plain squared difference is the weaker choice) and `show_error_on_track_links` (on: the
# paths have always been coloured by error). A new Tracker's picture still follows its tracks
# (`livelink_transform` on, `mode` match_move), which is what this node has always done and what its overlay draws.
NEW = {**TRANSFORM, **ANALYSIS, **SOLVE}
KINDS = ("Tracker",)


def new_params(kind):
    return dict(NEW) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(OLD) if kind in KINDS else {}


# ------------------------------------------------------------------------------------------------------ matching
def prefiltered(gray, mode):
    """The analysis image `gray` (a 2D array) after the pre-track filter `mode`."""
    if mode == "median":
        padded = np.pad(gray, 1, mode="edge")
        height, width = gray.shape
        stack = np.stack([padded[dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)])
        return np.ascontiguousarray(np.median(stack, axis=0))
    if mode == "adjust contrast":
        low, high = float(gray.min()), float(gray.max())
        return (gray - low) / (high - low) if high - low > 1e-12 else gray
    return gray


@dataclass(frozen=True)
class Grab:
    """When the pattern is grabbed again while tracking."""
    behavior: str = "on first frame"
    interval: int = 0
    above: float = 0.05
    below: float = 0.0

    def due(self, step, error):
        """True when, after tracking frame number `step` (1 for the first one tracked) with `error`, the pattern is grabbed again."""
        if self.behavior == "every frame":
            return True
        if self.behavior == "every n frames":
            return self.interval > 0 and step % self.interval == 0
        if self.behavior == "if error above":
            return error > self.above
        if self.behavior == "if error below":
            return error < self.below
        return False


def analysis_options(params):
    """The keyword arguments of `tracker.analyse` that a Tracker's knobs choose."""
    p = {**OLD, **params}
    max_error = float(p["max_error"])
    return {"metric": "ncc" if int(p["adjust_for_luminance_changes"]) else "ssd", "prefilter": p["pretrack_filter"],
            "clamp": bool(int(p["clamp_footage"])), "predict": bool(int(p["predict_track"])),
            # `adaptive_update`, the older switch, is "every frame"
            "grab": Grab("every frame" if int(p.get("adaptive_update", 0)) else p["grab_behavior"], int(p["grab_interval"]),
                         float(p["grab_error_above"]), float(p["grab_error_below"])),
            "max_error": None if max_error >= 1.0 else max_error}


def track_radii(params, track):
    """The pattern and search radius of one track: its own, else the node's. The search is never smaller than the pattern."""
    pattern = int((track or {}).get("pattern_radius", params.get("pattern_radius", 8)))
    search = int((track or {}).get("search_radius", params.get("search_radius", 24)))
    return pattern, max(search, pattern)


def snap_to_marker(gray, point, origin=(0, 0), radius=SNAP_RADIUS):
    """The centre of the marker near `point`: the contrast-weighted centroid of the window of `radius` pixels about it.

    The contrast of a pixel is how far it is from the window's median, so a small mark on a flat area gives its own centre and
    a flat area gives nothing: the point is returned as it was. `gray` is a 2D analysis image whose top left pixel is `origin`
    in canvas coordinates."""
    ix = int(math.floor(point[0] - origin[0]))
    iy = int(math.floor(point[1] - origin[1]))
    left, right = max(0, ix - radius), min(gray.shape[1], ix + radius + 1)
    top, bottom = max(0, iy - radius), min(gray.shape[0], iy + radius + 1)
    if right - left < 3 or bottom - top < 3:
        return float(point[0]), float(point[1])
    window = gray[top:bottom, left:right]
    weight = np.abs(window - np.median(window))
    weight = np.where(weight > 0.05 * max(float(weight.max()), 1e-12), weight, 0.0)
    total = float(weight.sum())
    if total <= 1e-9:
        return float(point[0]), float(point[1])
    ys, xs = np.mgrid[top:bottom, left:right]
    return (float(origin[0] + (weight * (xs + 0.5)).sum() / total), float(origin[1] + (weight * (ys + 0.5)).sum() / total))


# ------------------------------------------------------------------------------------------------ the track list
def track_flags(track):
    """(T, R, S) of a track: a track that sets none of them is all three, as every older track was."""
    t, r, s = (int(track.get(name, 0)) for name in ("use_t", "use_r", "use_s"))
    return (True, True, True) if not (t or r or s) else (bool(t), bool(r), bool(s))


def new_track(tracks, x, y):
    """A new track, named after the lowest free `trackN`, at (x, y)."""
    names = {track.get("name") for track in tracks}
    index = 1
    while f"track{index}" in names:
        index += 1
    return {"name": f"track{index}", "enabled": 1, "x": float(x), "y": float(y)}


def averaged_track(tracks, indices, frames, name="average"):
    """One new track that follows the mean position of the tracks at `indices`, with a key on every frame of `frames`.

    Only enabled tracks count at a frame; a frame where none is enabled holds the previous mean (the first mean before it)."""
    from . import shapes
    chosen = [tracks[i] for i in indices if 0 <= i < len(tracks)]
    if len(chosen) < 2:
        raise ValueError("Select at least two tracks to average")
    frames = list(frames)
    xs, ys, errors = [], [], []
    last = None
    for frame in frames:
        resolved = [shapes.resolve_tracks({"tracks": [track]}, frame)[0] for track in chosen]
        usable = [item for item in resolved if item["enabled"]]
        if usable:
            last = (sum(item["x"] for item in usable) / len(usable), sum(item["y"] for item in usable) / len(usable),
                    sum(item["error"] for item in usable) / len(usable))
        if last is None:
            first = [item for item in resolved]
            last = (sum(item["x"] for item in first) / len(first), sum(item["y"] for item in first) / len(first), 0.0)
        xs.append({"frame": int(frame), "value": float(last[0])})
        ys.append({"frame": int(frame), "value": float(last[1])})
        errors.append({"frame": int(frame), "value": float(last[2])})
    existing = {track.get("name") for track in tracks}
    unique, n = name, 1
    while unique in existing:
        n += 1
        unique = f"{name}{n}"
    scalar = lambda keys: {"value": keys[0]["value"], "curve": {"interpolation": "linear", "keys": keys}}
    return {"name": unique, "enabled": 1, "x": scalar(xs), "y": scalar(ys), "error": scalar(errors)}


def remove_tracks(tracks, indices):
    """`tracks` without the ones at `indices`."""
    drop = set(indices)
    return [track for i, track in enumerate(tracks) if i not in drop]


def moved_point(track, frame, x, y, create_key=True, resolve=None, set_at=None):
    """`track` with its point put at (x, y) on `frame`.

    With `create_key` (Nuke's `create new key when track is moved`) that frame gets a key. Without it the whole track moves by
    the same amount, as if the point had been dragged on every frame."""
    from . import shapes
    track = dict(track)
    if create_key:
        for field, value in (("x", x), ("y", y)):
            track[field] = set_at(track[field], frame, value)
        return track
    for field, value in (("x", x), ("y", y)):
        delta = value - shapes.resolve_scalar(track[field], frame, field)
        scalar = track[field]
        if isinstance(scalar, dict):
            curve = scalar.get("curve")
            track[field] = {"value": float(scalar["value"]) + delta,
                            "curve": None if not curve else {**curve, "keys": [{**key, "value": float(key["value"]) + delta}
                                                                                for key in curve["keys"]]}}
        else:
            track[field] = float(scalar) + delta
    return track


def merged_keys(existing, tracked, replace_range=True):
    """The keys of a track's curve after an analysis: `tracked` frames replace what was there.

    With `replace_range` (Nuke's `auto-tracks delete keyframes`) every old key inside the tracked frames goes and nothing else
    stays between them; without it the old keys outside the tracked frames stay."""
    if replace_range or not existing:
        return sorted(tracked, key=lambda key: key["frame"])
    frames = {key["frame"] for key in tracked}
    low, high = min(frames), max(frames)
    kept = [key for key in existing if not low <= key["frame"] <= high]
    return sorted(kept + list(tracked), key=lambda key: key["frame"])
