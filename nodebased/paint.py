"""Deterministic CPU RotoPaint stroke rasterisation and compositing."""
from __future__ import annotations

import math

import numpy as np


def active(stroke, frame):
    life = stroke["lifetime"]
    mode = life["mode"]
    if mode == "all":
        return True
    if mode == "single":
        return frame == life["first"]
    if mode == "range":
        return life["first"] <= frame <= life["last"]
    return frame >= life["first"]


def _distance_grid(width, height, points, size, spacing, hardness=1.0, origin=(0, 0)):
    """Stroke coverage in 0..1: the max over a chain of round dabs along the point path.

    A dab is solid out to `hardness` of its radius and falls off smoothly from there to the rim
    (hardness 1 is a hard edge with a one pixel antialiased rim, 0 a falloff across the whole
    radius), like Nuke's brush hardness. Pressure scales both the dab's radius and its strength.
    """
    ox, oy = map(int, origin)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    hardness = min(1.0, max(0.0, float(hardness)))
    coverage = np.zeros((height, width), np.float32)
    for a, b in zip(points, points[1:] or points):
        x0, y0, p0 = a["x"], a["y"], a["pressure"]
        x1, y1, p1 = b["x"], b["y"], b["pressure"]
        length = math.hypot(x1-x0, y1-y0)
        count = max(1, int(math.ceil(length / max(spacing * size, 0.25))))
        for i in range(count + 1):
            t = i / count
            x, y = x0 + t*(x1-x0), y0 + t*(y1-y0)
            pressure = p0 + t*(p1-p0)
            radius = max(0.25, size * pressure * 0.5)
            d = np.sqrt((xx + ox + 0.5 - x)**2 + (yy + oy + 0.5 - y)**2)
            ramp = np.clip((radius + 0.5 - d) / ((1.0 - hardness) * radius + 1.0), 0, 1)
            if hardness < 1.0:
                ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            coverage = np.maximum(coverage, ramp * pressure)
    return coverage


def _stroke_region(width, height, points, size, origin=(0, 0)):
    """Return the clipped pixel bounds that can receive nonzero coverage from a stroke."""
    if not points or width <= 0 or height <= 0:
        return None
    ox, oy = map(int, origin)
    pressure = max(0.0, *(float(point.get("pressure", 1.0)) for point in points))
    radius = max(0.25, float(size) * pressure * 0.5) + 0.5
    left = max(0, math.floor(min(float(p["x"]) for p in points) - radius - 1.0 - ox))
    top = max(0, math.floor(min(float(p["y"]) for p in points) - radius - 1.0 - oy))
    right = min(width, math.ceil(max(float(p["x"]) for p in points) + radius + 1.0 - ox))
    bottom = min(height, math.ceil(max(float(p["y"]) for p in points) + radius + 1.0 - oy))
    return (left, top, right, bottom) if right > left and bottom > top else None


def apply_stroke(image, stroke, frame, source=None, reveal=None, origin=(0, 0),
                 source_origin=(0, 0), source_bounds=None):
    if not active(stroke, frame) or not stroke["visible"]:
        return image
    tool, blend = stroke["tool"], stroke["blend"]
    if tool == "paint" and blend == "over":
        # Most brush marks touch a small part of a tile. Rasterise and composite only that
        # footprint; scanning the whole tile for every dab made a new stroke invalidate hundreds
        # of thousands of pixels it could never affect.
        brush = stroke["brush"]
        bounds = _stroke_region(image.shape[1], image.shape[0], stroke["points"],
                                brush["size"], origin)
        if bounds is None:
            return image
        x0, y0, x1, y1 = bounds
        local_origin = (int(origin[0]) + x0, int(origin[1]) + y0)
        coverage = _distance_grid(x1 - x0, y1 - y0, stroke["points"],
                                  brush["size"], brush["spacing"],
                                  brush["hardness"], local_origin)
        alpha = coverage[..., None] * np.float32(brush["opacity"] * stroke["opacity"])
        rgba = np.asarray(stroke["color"], np.float32)
        inverse = np.float32(1.0) - alpha
        source_patch = image[y0:y1, x0:x1]
        painted = source_patch * inverse
        painted[..., :3] += (rgba[:3] * rgba[3]) * alpha
        painted[..., 3:4] += rgba[3] * alpha
        out = image.copy()
        out[y0:y1, x0:x1] = painted
        return np.asarray(out, np.float32)
    # The source is read-only here. Every tool below either builds a fresh output or explicitly
    # copies before its in-place blend, so copying the complete frame up front is unnecessary.
    base = image
    ox, oy = map(int, origin)
    brush = stroke["brush"]
    coverage = _distance_grid(image.shape[1], image.shape[0], stroke["points"],
                              brush["size"], brush["spacing"], brush["hardness"], origin)
    alpha = coverage[..., None] * np.float32(brush["opacity"] * stroke["opacity"])
    tool = stroke["tool"]
    if tool == "paint":
        rgba = np.asarray(stroke["color"], np.float32)
        if blend == "over":
            # The common paint/over case has a constant source colour. Broadcast that colour
            # directly rather than materialising a full-frame `target` and two full-frame
            # multiplication temporaries. Keep the same float32 operation order as the general
            # blend path so the optimization is pixel-identical.
            inverse = np.float32(1.0) - alpha
            out = base * inverse
            out[..., :3] += (rgba[:3] * rgba[3]) * alpha
            out[..., 3:4] += rgba[3] * alpha
            return np.asarray(out, np.float32)
        target = np.empty_like(base); target[..., :3] = rgba[:3] * rgba[3]; target[..., 3] = rgba[3]
    elif tool == "eraser":
        target = np.zeros_like(base) if source is None else source
    elif tool == "clone":
        if source is None: return base
        dx, dy = stroke["source_offset"]
        h, w = base.shape[:2]
        sy, sx = np.mgrid[0:h, 0:w]
        sx = np.rint(sx + ox - dx).astype(int)
        sy = np.rint(sy + oy - dy).astype(int)
        if source_bounds is None:
            sx = np.clip(sx, 0, source.shape[1]-1)
            sy = np.clip(sy, 0, source.shape[0]-1)
        else:
            bx, by, bw, bh = source_bounds
            sx = np.clip(sx, bx, bx + bw - 1) - int(source_origin[0])
            sy = np.clip(sy, by, by + bh - 1) - int(source_origin[1])
            sx = np.clip(sx, 0, source.shape[1]-1)
            sy = np.clip(sy, 0, source.shape[0]-1)
        target = source[sy, sx]
        patch_blend = float(stroke.get("patch_blend", 0.0))
        if patch_blend > 0:
            # DustBust patch synthesis (2D parity plan 13, step E2): fill the stroke's own
            # footprint from the surrounding pixels of THIS frame (a border-in diffusion, same
            # shape as Inpaint's spatial fill) and blend it with the previous-frame clone, so a
            # speck over a moving background does not clone in a stale ghost of the old frame.
            from .flow_nodes import spatial_fill
            patched = spatial_fill(base, coverage, "diffusion")
            target = target * np.float32(1.0 - patch_blend) + patched * np.float32(patch_blend)
    elif tool == "reveal":
        target = base if reveal is None else reveal
    elif tool in ("blur", "sharpen"):
        # A bigger brush reaches a wider box-blur radius, like a real blur/sharpen dab rather than
        # a single fixed 3x3 average; reuses the Blur node's own separable filter (roto.py's
        # feather does the same) so this can never drift from it.
        from .imaging import Evaluator
        radius = max(1.0, brush["size"] / 6.0)
        blurred = Evaluator._box_blur_axis(Evaluator._box_blur_axis(base, radius, axis=1), radius, axis=0)
        target = blurred if tool == "blur" else np.clip(base + 1.5 * (base - blurred), 0, None)
    elif tool == "smear":
        # Drags colour from behind the stroke's own start-to-end direction, distance scaled by
        # brush size: a real directional smear, not a blur alias.
        points = stroke["points"]
        dx = dy = 0.0
        if len(points) >= 2:
            dx, dy = points[-1]["x"] - points[0]["x"], points[-1]["y"] - points[0]["y"]
        length = math.hypot(dx, dy)
        if length > 1e-6:
            reach = min(length, brush["size"] * 0.5)
            ux, uy = dx / length * reach, dy / length * reach
            h, w = base.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            sx = np.clip(np.round(xx - ux).astype(int), 0, w - 1)
            sy = np.clip(np.round(yy - uy).astype(int), 0, h - 1)
            target = base[sy, sx]
        else:
            target = base
    elif tool == "dodge":
        # brush["strength"] replaces a fixed 0.2 lift; 0.2 is still the default, so an old stroke
        # migrated without the field renders the same pixels.
        target = base + (1-base)*brush["strength"]
    else:
        target = base*(1.0-brush["strength"])
    if blend == "over" or tool == "eraser":
        out = target * alpha + base * (1-alpha)
    elif blend == "add":
        out = base.copy(); out[..., :3] += target[..., :3] * alpha
        out[..., 3:4] = target[..., 3:4] * alpha + base[..., 3:4] * (1-alpha)
    elif blend == "multiply":
        out = base.copy(); out[..., :3] *= (1-alpha + target[..., :3] * alpha)
        out[..., 3:4] = target[..., 3:4] * alpha + base[..., 3:4] * (1-alpha)
    else:
        out = base.copy(); out[..., :3] = 1-(1-base[..., :3])*(1-target[..., :3]*alpha)
        out[..., 3:4] = target[..., 3:4] * alpha + base[..., 3:4] * (1-alpha)
    return np.asarray(out, np.float32)


def rasterise(image, items, frame, source=None, reveal=None, origin=(0, 0),
              canvas_size=None):
    out = image.copy()
    for item in items:
        delta = item.get("_track_delta", (0.0, 0.0))
        if delta != (0.0, 0.0) and list(delta) != [0.0, 0.0]:
            item = dict(item, points=[dict(p, x=p["x"] + delta[0], y=p["y"] + delta[1])
                                      for p in item["points"]])
        if item["kind"] == "stroke":
            clone_map = item.get("_clone_sources", {})
            source_frame = item.get("_clone_frame", int(frame)-1 if item["source_frame"] == "relative"
                            else item["source_frame"])
            clone_raster = clone_map.get(source_frame)
            clone_source = source if clone_raster is None else clone_raster.fit(clone_raster.data)
            out = apply_stroke(out, item, frame,
                               source=clone_source[..., :4] if hasattr(clone_source, "shape") else source,
                               reveal=reveal, origin=origin,
                               source_origin=item.get("_clone_source_origins", {}).get(source_frame, origin),
                               source_bounds=item.get("_clone_source_bounds", {}).get(source_frame))
        else:
            from . import roto
            shape = {k: item[k] for k in ("name", "mode", "opacity", "feather", "points")}
            full_w, full_h = canvas_size or (image.shape[1], image.shape[0])
            matte = roto.rasterise([shape], full_w, full_h,
                                   region=(origin[0], origin[1], image.shape[1], image.shape[0]))[..., 3:4]
            alpha = matte * np.float32(item["opacity"] if item["visible"] else 0.0)
            base = out.copy()
            if item["blend"] == "over":
                out = np.concatenate((alpha + base[..., :3] * (1-alpha),
                                      alpha + base[..., 3:4] * (1-alpha)), axis=-1)
            elif item["blend"] == "add":
                out = base.copy(); out[..., :3] = np.clip(base[..., :3] + alpha, 0, 1)
                out[..., 3:4] = alpha + base[..., 3:4] * (1-alpha)
            elif item["blend"] == "multiply":
                out = base.copy(); out[..., :3] = base[..., :3]  # multiply by the white matte layer
                out[..., 3:4] = alpha + base[..., 3:4] * (1-alpha)
            elif item["blend"] == "screen":
                out = base.copy(); out[..., :3] = 1 - (1-base[..., :3]) * (1-alpha)
                out[..., 3:4] = alpha + base[..., 3:4] * (1-alpha)
    return out
