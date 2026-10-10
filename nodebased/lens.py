"""Thin-lens depth of field for Camera3D: the aperture, the circle of confusion and the pick-focus query.

Pure functions shared by the CPU path tracer, its WGSL twin (gpupathtrace.py builds the same uniform from
`lens_uniform`), the ray-traced mode and the tests. One scene unit is one metre; the focal length is in
millimetres (Nuke's film back), so the aperture diameter is `focal / fstop` millimetres. `fstop` 0 is a pinhole:
no lens sampling happens and every renderer draws exactly what it drew before the lens existed.

The lens sits at the camera position. A pixel's rays start at a point of the aperture and all aim at the same
point on the focal plane (the plane `focus_distance` in front of the camera along the view axis), so anything on
that plane is sharp and everything else spreads into a copy of the aperture shape. The aperture radius is scaled
by S / (S - f) so the spread on the sensor equals the textbook thin-lens circle of confusion,
c = A f |S - s| / (s (S - f)), rather than its f << S approximation.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_FOCUS_DISTANCE = 5.0     # the default Camera3D looks at the origin from five units away
MIN_BLADES = 3                   # fewer blades than this is a round aperture
MAX_BLADES = 16
# Random dimensions of a path key the aperture reads (0 and 1 are the pixel jitter; see pathtrace._DIM_BASE).
DIM_U1, DIM_U2 = 2, 3


def active(camera):
    """True when `camera` has a lens to sample."""
    return getattr(camera, "fstop", 0.0) > 0.0


def focus_plane(camera):
    """The focus distance in scene units, kept beyond the focal length so the lens equation stays finite."""
    focal_m = camera.focal * 1e-3
    return max(float(camera.focus_distance), focal_m * 1.001)


def aperture_radius(camera):
    """Radius of the aperture in scene units (metres), after the S / (S - f) correction described above."""
    focal_m = camera.focal * 1e-3
    focus = focus_plane(camera)
    return 0.5 * (focal_m / camera.fstop) * focus / (focus - focal_m)


def circle_of_confusion_px(camera, depth, height):
    """Diameter in pixels of the blur circle of a point `depth` units in front of the camera (thin lens).

    `height` is the image height in pixels; the film back's vertical aperture maps onto it."""
    focal_m = camera.focal * 1e-3
    focus = focus_plane(camera)
    aperture = focal_m / camera.fstop
    depth = np.maximum(np.asarray(depth, np.float64), 1e-9)
    on_sensor = aperture * focal_m * np.abs(focus - depth) / (depth * (focus - focal_m))
    return on_sensor * 1000.0 / camera.vaperture * height


def blade_vertices(blades, rotation_degrees):
    """Corners of the aperture polygon on the unit circle, or None for a round aperture."""
    blades = int(blades)
    if blades < MIN_BLADES:
        return None
    blades = min(blades, MAX_BLADES)
    angles = math.radians(rotation_degrees) + 2.0 * math.pi * np.arange(blades) / blades
    return np.column_stack((np.cos(angles), np.sin(angles)))


def aperture_points(u1, u2, blades=0, rotation_degrees=0.0, squeeze=1.0):
    """Uniform points on the unit aperture from uniform numbers `u1`, `u2` in [0, 1): (x, y) arrays.

    Round for fewer than three blades; otherwise a regular polygon (a fan of triangles from the centre, one
    chosen by `u1`) turned by `rotation_degrees`. `squeeze` divides x, so an anamorphic squeeze of 2 makes the
    bokeh twice as tall as it is wide."""
    u1, u2 = np.asarray(u1, np.float64), np.asarray(u2, np.float64)
    corners = blade_vertices(blades, rotation_degrees)
    if corners is None:
        radius, angle = np.sqrt(u2), 2.0 * math.pi * u1
        x, y = radius * np.cos(angle), radius * np.sin(angle)
    else:
        n = len(corners)
        scaled = u1 * n
        index = np.minimum(scaled.astype(np.int64), n - 1)
        r1 = np.sqrt(u2)
        r2 = scaled - index
        a, b = corners[index], corners[(index + 1) % n]
        x = r1 * ((1.0 - r2) * a[..., 0] + r2 * b[..., 0])
        y = r1 * ((1.0 - r2) * a[..., 1] + r2 * b[..., 1])
    return x / max(float(squeeze), 1e-6), y


def lens_uniform(camera):
    """(radius, focus, blades, rotation radians, 1 / squeeze) as the GPU path tracer's uniform reads them."""
    return (aperture_radius(camera), focus_plane(camera), float(int(camera.aperture_blades)),
            math.radians(camera.blade_rotation), 1.0 / max(float(camera.anamorphic_squeeze), 1e-6))


def radical_inverse(k):
    """Van der Corput radical inverse base 2 of the integers `k` (the second Hammersley dimension)."""
    k = np.asarray(k, np.uint32)
    k = ((k << np.uint32(16)) | (k >> np.uint32(16))).astype(np.uint32)
    k = (((k & np.uint32(0x55555555)) << np.uint32(1)) | ((k & np.uint32(0xAAAAAAAA)) >> np.uint32(1))).astype(np.uint32)
    k = (((k & np.uint32(0x33333333)) << np.uint32(2)) | ((k & np.uint32(0xCCCCCCCC)) >> np.uint32(2))).astype(np.uint32)
    k = (((k & np.uint32(0x0F0F0F0F)) << np.uint32(4)) | ((k & np.uint32(0xF0F0F0F0)) >> np.uint32(4))).astype(np.uint32)
    k = (((k & np.uint32(0x00FF00FF)) << np.uint32(8)) | ((k & np.uint32(0xFF00FF00)) >> np.uint32(8))).astype(np.uint32)
    return k.astype(np.float64) * (1.0 / 4294967296.0)


def hammersley_offsets(camera, count):
    """`count` world-space lens offsets (ax, ay) along the view's right and up axes: a Hammersley set over the
    aperture, used by the ray-traced mode (one full render per offset)."""
    k = np.arange(count)
    x, y = aperture_points((k + 0.5) / count, radical_inverse(k), camera.aperture_blades,
                           camera.blade_rotation, camera.anamorphic_squeeze)
    radius = aperture_radius(camera)
    return x * radius, y * radius


def pick_focus_distance(camera, scene, width, height, x, y):
    """The view depth of the surface under pixel (x, y), for the viewer's pick-focus click; None over empty space.

    Traces one un-jittered pinhole ray with the path tracer's data pass, so instances, splats and volumes are
    picked like meshes."""
    from dataclasses import replace
    from . import pathtrace
    pin = replace(camera, fstop=0.0)
    ps = pathtrace.build_scene(scene, 0.0, eye=None)
    dirs, eye, cos, tmin, tmax = pathtrace.camera_rays(pin, width, height, np.array([x + 0.5]), np.array([y + 0.5]))
    t, shape, _, _, _ = pathtrace.closest(ps, np.broadcast_to(eye, dirs.shape), dirs, tmin, tmax,
                                        camera=np.ones(1, bool))
    if shape[0] < 0:
        return None
    return float(t[0] * cos[0])
