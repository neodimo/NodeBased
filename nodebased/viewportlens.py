"""Looking through a camera's lens in the live 3D viewport and in the GPU renderers (Rendering 13 step G2).

Every renderer here draws the ideal pinhole picture. A camera with lens distortion puts the same scene on the plate
somewhere else, so to see (or render) the plate the ideal picture is drawn a little larger, the overscan, and then
Redistort resamples it into the format: the same warp the Redistort node runs (nodebased/lensmodel.py,
nodebased/gpulens.py). This module is the part both callers share: how far the picture has to grow, the camera that
draws it, and the warp back down, on the card when there is one and on the CPU otherwise.

Pure functions, no Qt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from . import lensmodel
from .tiers import Region

# Largest overscan on each side, as a fraction of the format's width and height. The Undistort node grows up to
# `lensmodel.MAX_GROWTH` (two formats a side) for a lens that throws a corner far out; an interactive frame stops
# at this, and the corners beyond it stay black rather than costing a frame ten times the size.
MAX_MARGIN = 0.75

# Lowest pixel count at which the card repays the copies; below it the CPU resample is quicker.
GPU_MIN_PIXELS = 160 * 120


@dataclass(frozen=True)
class LensFrame:
    """One lens view: the format (`width` x `height`), the lens, and the overscan the ideal picture needs."""
    lens: lensmodel.Lens
    width: int
    height: int
    margin_x: int
    margin_y: int

    @property
    def render_width(self):
        return self.width + 2 * self.margin_x

    @property
    def render_height(self):
        return self.height + 2 * self.margin_y

    @property
    def source(self):
        """The overscan picture's data window, relative to the format's corner."""
        return Region(-self.margin_x, -self.margin_y, self.render_width, self.render_height)

    @property
    def display(self):
        return Region(0, 0, self.width, self.height)

    def to_plate(self, xy):
        """Pixels of the overscan picture (N, 2) to where the lens puts them on the format's plate."""
        xy = np.asarray(xy, np.float64)
        x, y = lensmodel.distort_pixels(self.lens, xy[:, 0] - self.margin_x, xy[:, 1] - self.margin_y,
                                        self.width, self.height)
        return np.column_stack((x, y)).astype(np.float32)

    def camera(self, camera):
        """`camera` widened to draw the overscan picture at the same scale per pixel as the format's picture."""
        if self.margin_y == 0 and self.margin_x == 0:
            return camera
        grow = self.render_height / max(self.height, 1)
        fov = 2.0 * math.degrees(math.atan(math.tan(math.radians(camera.fov) / 2.0) * grow))
        return replace(camera, fov=fov)


def active(camera):
    """True when `camera` has a lens that bends something."""
    return camera is not None and lensmodel.active(camera)


def margins(lens, width, height, limit=MAX_MARGIN):
    """(x, y) pixels the ideal picture needs on each side so that a Redistort never reads outside it."""
    if lens.is_identity or width < 1 or height < 1:
        return 0, 0
    display = Region(0, 0, width, height)
    x, y, w, h = lensmodel.undistorted_window(lens, display, display)
    mx, my = max(-x, x + w - width, 0), max(-y, y + h - height, 0)
    return min(int(mx), int(limit * width)), min(int(my), int(limit * height))


def frame_for(camera, width, height, limit=MAX_MARGIN):
    """The `LensFrame` of a width x height picture seen through `camera`, or None when it has no lens."""
    if not active(camera):
        return None
    lens = lensmodel.from_camera(camera)
    mx, my = margins(lens, width, height, limit)
    return LensFrame(lens, int(width), int(height), mx, my)


def fit(frame, picture):
    """`picture` stretched (nearest) to the overscan size when a low-resolution first pass drew it smaller."""
    picture = np.asarray(picture)
    if picture.shape[:2] == (frame.render_height, frame.render_width):
        return picture
    rows = np.minimum((np.arange(frame.render_height) * picture.shape[0]) // frame.render_height, picture.shape[0] - 1)
    columns = np.minimum((np.arange(frame.render_width) * picture.shape[1]) // frame.render_width, picture.shape[1] - 1)
    return np.ascontiguousarray(picture[rows][:, columns])


def redistort(frame, picture, filter_name="bilinear", choice=None, backend="auto"):
    """The overscan `picture` (render_height x render_width x 4, uint8 or float32) resampled to the format.

    `backend` "auto" uses the card when the output is big enough to repay the copies and falls back to the CPU when
    the card cannot take the frame; "cpu" and "gpu" force one (a forced "gpu" raises what stopped it)."""
    from . import gpulens
    picture = np.asarray(picture)
    if picture.shape[:2] != (frame.render_height, frame.render_width):
        raise ValueError(f"the picture is {picture.shape[1]} x {picture.shape[0]}, the lens frame needs "
                         f"{frame.render_width} x {frame.render_height}")
    if backend != "cpu" and (backend == "gpu" or frame.width * frame.height >= GPU_MIN_PIXELS):
        try:
            return gpulens.warp(frame.lens, picture, frame.display, frame.display, frame.source, "redistort",
                                filter_name, choice=choice)
        except gpulens.Unsupported:
            if backend == "gpu":
                raise
        except Exception:
            if backend == "gpu":
                raise
    return redistort_cpu(frame, picture, filter_name)


def redistort_cpu(frame, picture, filter_name="bilinear"):
    """The reference: the Redistort node's own sample coordinates and resampler."""
    from .imaging import Evaluator
    picture = np.asarray(picture)
    packed = picture.dtype == np.uint8
    sx, sy = lensmodel.sample_coordinates(frame.lens, frame.display, frame.display, frame.source, "redistort")
    result = Evaluator._resample(picture.astype(np.float32) / 255.0 if packed else picture, sx, sy, filter_name)
    if packed:
        return np.ascontiguousarray(np.clip(result * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return np.ascontiguousarray(result.astype(np.float32))
