"""Nuke's camera film-back model: focal length and apertures in millimetres.

Pure functions, imported by both core (document upgrade, defaults) and scene3d (Camera). The
vertical field of view is the one number every renderer reads (`Camera.fov`); it is derived
here and never stored next to a focal length that could disagree with it.
"""
from __future__ import annotations

import math

# Nuke's own film back. The default focal length is the one that makes the derived vertical
# field of view exactly 45 degrees, the value Camera3D used before the film back existed, so a
# default camera and every document saved earlier render identically.
DEFAULT_HAPERTURE = 24.576
DEFAULT_VAPERTURE = 18.672
DEFAULT_FOV = 45.0
DEFAULT_FOCAL = DEFAULT_VAPERTURE / (2 * math.tan(math.radians(DEFAULT_FOV) / 2))

# Field of view is rounded to nine decimals (1e-9 degrees, far below a pixel) so the default
# film back yields exactly 45.0 and not 45.00000000000001.
_DECIMALS = 9


def fov_from_aperture(focal, aperture):
    """Field of view in degrees across one film-back dimension."""
    return round(math.degrees(2 * math.atan(aperture / (2 * focal))), _DECIMALS)


def focal_from_fov(fov, aperture):
    """Focal length in millimetres that gives `fov` degrees across `aperture`."""
    return aperture / (2 * math.tan(math.radians(fov) / 2))


def film_back_from_gltf(yfov, aspect=None):
    """(focal, haperture, vaperture) equivalent to a glTF camera: yfov in radians, aspect w/h."""
    vaperture = DEFAULT_VAPERTURE
    haperture = vaperture * aspect if aspect else DEFAULT_HAPERTURE
    return focal_from_fov(math.degrees(yfov), vaperture), haperture, vaperture
