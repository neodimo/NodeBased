"""Automatic speck detection for RotoPaint's DustBust preset.

Real dust and dirt on a scanned plate flickers frame to frame (it sits on the film base, not the
scene), so a small region whose luminance jumps away from the temporal median of its neighbouring
frames and then falls back is dust; a real moving object in the scene is either larger than a
speck or keeps its difference across more than one frame. `detect_specks` is deterministic and
pure so it can run ahead of any UI decision; the caller (the RotoPaint panel's "Detect specks…"
button) shows the results as a review list the artist accepts or rejects one at a time before
`dustbust_items_for_specks` turns the accepted ones into ordinary DustBust clone strokes.
"""
from __future__ import annotations

import numpy as np


def _connected_components(mask, weight, max_size):
    """4-connected components of a boolean mask, each no larger than `max_size` pixels across.

    Returns a list of (top, left, height, width, magnitude) boxes. A component whose bounding box
    exceeds `max_size` in either dimension is dropped: that is real content, not a speck.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y0 in range(height):
        for x0 in range(width):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            visited[y0, x0] = True
            stack, pixels = [(y0, x0)], [(y0, x0)]
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                        pixels.append((ny, nx))
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
            h, w = bottom - top + 1, right - left + 1
            if h > max_size or w > max_size:
                continue
            magnitude = max(float(weight[y, x]) for y, x in pixels)
            boxes.append((top, left, h, w, magnitude))
    return boxes


def detect_specks(frames, sensitivity=0.5, max_size=12):
    """Find candidate dust specks across a sequence of already-rendered frames.

    `frames`: a mapping of frame number to an HxWx4 float32 raster; needs at least three distinct
    frame numbers (a speck needs a frame before and after it to compare against). `sensitivity` is
    0 (only the starkest anomalies) to 1 (nearly every small difference); it is clamped. Returns a
    list of {frame, x, y, width, height, magnitude} dicts, sorted by frame and then by descending
    magnitude, one entry per candidate speck, ready for an artist's accept/reject review.
    """
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    threshold = 0.5 * (1.0 - sensitivity) + 0.02
    numbers = sorted(frames)
    if len(numbers) < 3:
        return []
    results = []
    for i in range(1, len(numbers) - 1):
        frame_number = numbers[i]
        prev, cur, nxt = frames[numbers[i - 1]], frames[frame_number], frames[numbers[i + 1]]
        lum_prev = prev[..., :3].mean(axis=2)
        lum_cur = cur[..., :3].mean(axis=2)
        lum_next = nxt[..., :3].mean(axis=2)
        median = np.median(np.stack([lum_prev, lum_cur, lum_next]), axis=0)
        diff = np.abs(lum_cur - median)
        mask = diff > threshold
        for top, left, h, w, magnitude in _connected_components(mask, diff, max_size):
            results.append({"frame": frame_number, "x": float(left), "y": float(top),
                             "width": float(w), "height": float(h), "magnitude": magnitude})
    results.sort(key=lambda speck: (speck["frame"], -speck["magnitude"]))
    return results


def dustbust_items_for_specks(existing_items, specks, brush_padding=2.0, patch_blend=0.0):
    """Append one single-frame clone stroke per accepted speck to a RotoPaint item list.

    Each stroke is shaped exactly like the manual DustBust preset's click-to-dab stroke (a single
    point, `tool` clone, `source_frame` "relative" so it samples the previous frame): the only
    difference is the brush is sized to the detected speck's bounding box instead of the artist's
    current brush size, and the frame comes from the detector instead of the current frame.

    `patch_blend` (0 to 1, baked in at accept time from the node's `dustbust_patch_blend` knob) is
    stored on each stroke: 0 is the previous-frame clone alone; above 0, `paint.apply_stroke` mixes
    in a same-frame border fill of the stroke's own footprint (`flow_nodes.spatial_fill`), so a
    speck over a moving background does not clone in a stale previous-frame ghost.
    """
    items = list(existing_items)
    names = {item.get("name") for item in items}
    for speck in specks:
        index = 1
        while f"stroke{index}" in names:
            index += 1
        name = f"stroke{index}"
        names.add(name)
        center_x = speck["x"] + speck["width"] / 2.0
        center_y = speck["y"] + speck["height"] / 2.0
        size = max(speck["width"], speck["height"]) + brush_padding
        items.append({"kind": "stroke", "name": name,
                      "points": [{"x": center_x, "y": center_y, "pressure": 1.0}],
                      "brush": {"size": size, "hardness": 0.5, "opacity": 1.0, "spacing": 0.25, "strength": 0.2},
                      "tool": "clone", "lifetime": {"mode": "single", "first": int(speck["frame"])},
                      "color": [0.0, 0.0, 0.0, 1.0], "source_offset": [0.0, 0.0],
                      "source_frame": "relative", "opacity": 1.0, "blend": "over",
                      "visible": True, "follow_track": None,
                      "patch_blend": float(max(0.0, min(1.0, patch_blend)))})
    return items
