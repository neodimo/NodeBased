"""Brown-Conrady lens distortion for Camera3D, and the Undistort / Redistort warps built on it.

Pure functions shared by the camera (projection, tracker rays, SceneState), the two 2D kernels and the tests.
Not to be confused with `lens.py`, the thin-lens depth of field.

Coordinates. The model works in film-back units about the centre of the format: x runs right and y runs up,
and 1.0 is half the format's height, so the numbers do not change with the image size, the proxy tier or an
overscan. A pixel (px, py) of a W x H format (image y points down, centres at half integers) is
    x = (px - W/2) / (H/2),   y = (H/2 - py) / (H/2).
`anamorphic_squeeze` s divides x before the model runs and multiplies it after: the lens distorts the
squeezed image on the sensor, and the plate has been stretched to s times that width.

The forward model maps an ideal (undistorted) point to where the lens put it on the plate, with
(dx, dy) = (x/s - cx, y - cy), r2 = dx^2 + dy^2:
    radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
    x' = dx radial + 2 p1 dx dy + p2 (r2 + 2 dx^2)
    y' = dy radial + p1 (r2 + 2 dy^2) + 2 p2 dx dy
and the result is ((x' + cx) s, y' + cy). All zeros is the identity. The inverse is solved with Newton's
method from the distorted point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Names of the Camera3D knobs, in the order `Lens` stores them.
KNOBS = ("lens_k1", "lens_k2", "lens_k3", "lens_p1", "lens_p2", "lens_cx", "lens_cy")
DEFAULTS = {name: 0.0 for name in KNOBS}

# Named starting points for the Undistort / Redistort nodes when no camera is wired. Values are typical
# magnitudes for the lens family (the units make a 16:9 frame corner sit at a radius of about 2), not measurements of a particular lens.
PRESETS = {
    "none": {},
    "mild barrel": {"lens_k1": -0.02},
    "strong barrel": {"lens_k1": -0.06, "lens_k2": 0.004},
    "pincushion": {"lens_k1": 0.03},
    "wide angle": {"lens_k1": -0.04, "lens_k2": 0.004},
    "fisheye": {"lens_k1": -0.08, "lens_k2": 0.01},
}

_NEWTON_STEPS = 24


@dataclass(frozen=True)
class Lens:
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    squeeze: float = 1.0

    @property
    def is_identity(self):
        """True when the lens bends nothing (a principal point or a squeeze alone moves no pixel)."""
        return not (self.k1 or self.k2 or self.k3 or self.p1 or self.p2)

    def as_dict(self):
        return {"k1": self.k1, "k2": self.k2, "k3": self.k3, "p1": self.p1, "p2": self.p2,
                "principal_point": [self.cx, self.cy], "anamorphic_squeeze": self.squeeze}


def from_camera(camera):
    """The lens carried by a `scene3d.Camera`."""
    return Lens(camera.lens_k1, camera.lens_k2, camera.lens_k3, camera.lens_p1, camera.lens_p2,
                camera.lens_cx, camera.lens_cy, camera.anamorphic_squeeze or 1.0)


def from_params(params, squeeze=1.0):
    return Lens(*(float(params.get(name, 0.0)) for name in KNOBS), float(squeeze) or 1.0)


def active(camera):
    """True when the camera's lens bends anything (a principal point alone moves nothing)."""
    return bool(camera.lens_k1 or camera.lens_k2 or camera.lens_k3 or camera.lens_p1 or camera.lens_p2)


def distort(lens, x, y):
    """Ideal film-back coordinates to the distorted ones (arrays of any shape)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    s = lens.squeeze
    dx, dy = x / s - lens.cx, y - lens.cy
    r2 = dx * dx + dy * dy
    radial = 1.0 + r2 * (lens.k1 + r2 * (lens.k2 + r2 * lens.k3))
    xd = dx * radial + 2.0 * lens.p1 * dx * dy + lens.p2 * (r2 + 2.0 * dx * dx)
    yd = dy * radial + lens.p1 * (r2 + 2.0 * dy * dy) + 2.0 * lens.p2 * dx * dy
    return (xd + lens.cx) * s, yd + lens.cy


def _jacobian(lens, dx, dy):
    r2 = dx * dx + dy * dy
    radial = 1.0 + r2 * (lens.k1 + r2 * (lens.k2 + r2 * lens.k3))
    dradial = 2.0 * (lens.k1 + r2 * (2.0 * lens.k2 + r2 * 3.0 * lens.k3))   # d radial / d r2, times 2
    a = radial + dx * dx * dradial + 2.0 * lens.p1 * dy + 6.0 * lens.p2 * dx
    b = dx * dy * dradial + 2.0 * lens.p1 * dx + 2.0 * lens.p2 * dy
    c = b
    d = radial + dy * dy * dradial + 6.0 * lens.p1 * dy + 2.0 * lens.p2 * dx
    return a, b, c, d


def undistort(lens, xd, yd):
    """Distorted film-back coordinates to the ideal ones. Points the model cannot invert (far outside any
    sensible lens) come back as NaN."""
    xd, yd = np.asarray(xd, np.float64), np.asarray(yd, np.float64)
    s = lens.squeeze
    tx, ty = xd / s - lens.cx, yd - lens.cy        # target in centred, desqueezed units
    dx, dy = tx.copy(), ty.copy()                  # start from "no distortion"
    for _ in range(_NEWTON_STEPS):
        r2 = dx * dx + dy * dy
        radial = 1.0 + r2 * (lens.k1 + r2 * (lens.k2 + r2 * lens.k3))
        fx = dx * radial + 2.0 * lens.p1 * dx * dy + lens.p2 * (r2 + 2.0 * dx * dx) - tx
        fy = dy * radial + lens.p1 * (r2 + 2.0 * dy * dy) + 2.0 * lens.p2 * dx * dy - ty
        a, b, c, d = _jacobian(lens, dx, dy)
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        step_x, step_y = (d * fx - b * fy) / det, (a * fy - c * fx) / det
        dx, dy = dx - step_x, dy - step_y
    x, y = (dx + lens.cx) * s, dy + lens.cy
    # A residual check turns a Newton run that wandered off into NaN instead of a plausible wrong point.
    ex, ey = distort(lens, x, y)
    bad = ~(np.hypot(ex - xd, ey - yd) < 1e-6)
    return np.where(bad, np.nan, x), np.where(bad, np.nan, y)


# --- pixels <-> film-back units --------------------------------------------------------------------------

def pixels_to_units(px, py, width, height):
    half = max(float(height), 1.0) / 2.0
    return (np.asarray(px, np.float64) - width / 2.0) / half, (height / 2.0 - np.asarray(py, np.float64)) / half


def units_to_pixels(x, y, width, height):
    half = max(float(height), 1.0) / 2.0
    return np.asarray(x, np.float64) * half + width / 2.0, height / 2.0 - np.asarray(y, np.float64) * half


def distort_pixels(lens, px, py, width, height):
    """Pixel position in the ideal image to the pixel position on the plate."""
    x, y = pixels_to_units(px, py, width, height)
    return units_to_pixels(*distort(lens, x, y), width, height)


def undistort_pixels(lens, px, py, width, height):
    """Pixel position on the plate to the pixel position in the ideal image."""
    x, y = pixels_to_units(px, py, width, height)
    return units_to_pixels(*undistort(lens, x, y), width, height)


# --- the two image warps ---------------------------------------------------------------------------------

# Largest data window an Undistort may grow to, as a multiple of the format in each direction. A lens that
# throws the corner to infinity would otherwise ask for an image of unbounded size.
MAX_GROWTH = 2.0


def undistorted_window(lens, data, display):
    """Rectangle (x, y, w, h) that holds `data` after undistortion, joined with the display window.

    The edge of the data window is sampled densely and mapped through the inverse model; rounding outward
    keeps every pixel the resampler can write inside the window."""
    dw, dh = display.width, display.height
    n = 64
    top = np.linspace(0.0, 1.0, n)
    xs = np.concatenate((data.x + top * data.width, data.x + top * data.width,
                         np.full(n, data.x), np.full(n, data.right)))
    ys = np.concatenate((np.full(n, data.y), np.full(n, data.bottom),
                         data.y + top * data.height, data.y + top * data.height))
    # The window is in pixel coordinates of the display; its origin is the display's own.
    ux, uy = undistort_pixels(lens, xs - display.x, ys - display.y, dw, dh)
    ok = np.isfinite(ux) & np.isfinite(uy)
    left, top_, right, bottom = display.x, display.y, display.right, display.bottom
    if ok.any():
        left, right = min(left, np.floor(ux[ok].min()) + display.x), max(right, np.ceil(ux[ok].max()) + display.x)
        top_, bottom = min(top_, np.floor(uy[ok].min()) + display.y), max(bottom, np.ceil(uy[ok].max()) + display.y)
    left = max(left, display.x - MAX_GROWTH * dw)
    right = min(right, display.right + MAX_GROWTH * dw)
    top_ = max(top_, display.y - MAX_GROWTH * dh)
    bottom = min(bottom, display.bottom + MAX_GROWTH * dh)
    return int(left), int(top_), int(right - left), int(bottom - top_)


def sample_coordinates(lens, out, display, source_data, direction):
    """Source pixel coordinates (relative to `source_data`'s corner) for every pixel centre of `out`.

    `direction` "undistort": the output is the ideal image, so each output pixel reads the plate where the
    lens put that point (the forward model). "redistort": the output is the plate, so each output pixel reads
    the ideal image where the inverse model says it came from. Coordinates are in display-window pixels
    with the display centre as the origin of the model."""
    dw, dh = display.width, display.height
    gx = np.arange(out.width, dtype=np.float64)[None, :] + out.x - display.x + 0.5
    gy = np.arange(out.height, dtype=np.float64)[:, None] + out.y - display.y + 0.5
    gx, gy = np.broadcast_to(gx, (out.height, out.width)), np.broadcast_to(gy, (out.height, out.width))
    if direction == "undistort":
        sx, sy = distort_pixels(lens, gx, gy, dw, dh)
    else:
        sx, sy = undistort_pixels(lens, gx, gy, dw, dh)
    sx = sx + display.x - source_data.x - 0.5
    sy = sy + display.y - source_data.y - 0.5
    # NaN (not invertible) reads as far outside the source, which the resampler turns into black.
    sx, sy = np.nan_to_num(sx, nan=-1e6), np.nan_to_num(sy, nan=-1e6)
    return sx.astype(np.float32), sy.astype(np.float32)
