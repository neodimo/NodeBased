"""A small material-ball preview for the 3D geometry nodes' properties panel (docs/SPLAT_RELIGHTING.md,
"Production look" step R6 "next", closed): a sphere carrying the node's own material knobs -- colour,
specular, the PBR metallic/roughness/specular set and the liquid fields (docs/3D_FOUNDATION.md,
"Rendering") -- under one light and a soft ambient, on a fixed ball shape rather than the node's own
geometry (Card3D and Cylinder3D get the same round preview a Sphere3D does, the look-dev convention).

Rendered once, with the CPU reference renderer at raster quality (`scene3d.render`, no shadows,
fast enough for a small fixed size): the app.py properties panel builds this once per panel build,
not live while a slider drags, a stated limit -- reselecting the node (or any other panel rebuild)
refreshes it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import envlight, scene3d as s
from .imaging import linear_to_srgb

SIZE = 96
_CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 3.2)), s.Vec3(0, 0, 0), 32.0, 0.1, 100.0)
_LIGHT = s.Light(intensity=1.5, position=s.Vec3(2.2, 2.6, 2.4))
_BACKGROUND = (0.09, 0.09, 0.11, 1.0)
_AMBIENT = 0.12
# A soft, uniform studio dome (docs/3D_FOUNDATION.md, "Environment light"): without one, a metallic
# ball (near-zero diffuse) reads as almost solid black outside the point light's own highlight, the
# opposite of what a look-dev preview is for.
_DOME_RGB = np.full((16, 32, 3), 0.35, np.float32)
_DOME = envlight.Environment(_DOME_RGB, envlight.fingerprint_of(_DOME_RGB))


def sphere_for(params):
    """A unit sphere carrying `params`' material knobs -- the same fields
    `scene3d.geometry_from_node` reads (spec_amount/spec_shininess/emission/metallic/pbr_roughness/
    pbr_specular, `material_fields`'s liquid set) -- on a fixed ball shape."""
    color = tuple(float(params.get(k, 0.8 if k != "alpha" else 1.0)) for k in ("red", "green", "blue", "alpha"))
    sphere = s._sphere(1.0, 24, color, s.Transform3D())
    return replace(sphere, specular=float(params.get("spec_amount", 0.0)),
                   shininess=float(params.get("spec_shininess", 32.0)),
                   emission=float(params.get("emission", 0.0)),
                   metallic=float(params.get("metallic", 0.0)),
                   pbr_roughness=float(params.get("pbr_roughness", 0.5)),
                   pbr_specular=float(params.get("pbr_specular", 0.5)),
                   **s.material_fields(params))


def render(params, size=SIZE):
    """(size, size, 4) uint8 straight-alpha sRGB image of the material ball for `params`."""
    scene = s.Scene((sphere_for(params),), (_LIGHT,), environments=(_DOME,))
    image = s.render(scene, _CAMERA, size, size, _BACKGROUND, ambient=_AMBIENT, shadows=False,
                     samples=2, mode="raster")
    rgb = np.clip(image[..., :3] / np.maximum(image[..., 3:4], 1e-6), 0, 1)
    srgb = np.clip(linear_to_srgb(rgb) * 255, 0, 255)
    alpha = np.full(srgb.shape[:2] + (1,), 255, np.float64)
    return np.concatenate((srgb, alpha), axis=2).astype(np.uint8)
