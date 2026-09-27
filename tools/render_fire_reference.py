"""Render docs/images/volume_fire_smoke.png: the analytic plume with fire, multiple scattering and a forward phase.

Left to right: the step A look (single scattering, no fire), then fire and smoke with the step B look. CPU reference
render, deterministic (analytic plume, fixed seed); run it from the repository root: `python tools/render_fire_reference.py`.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import scene3d as s, volumerender

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "volume_fire_smoke.png"
W, H = 320, 320
CAMERA = s.Camera(s.Transform3D(s.Vec3(0.0, 0.55, 3.4)), s.Vec3(0, 0.5, 0), fov=40.0)
KEY = s.Light("Directional", color=(0.6, 0.75, 1.0), intensity=1.2, position=s.Vec3(-3, 4, -2), target=s.Vec3(0, .5, 0))
BASE = volumerender.VolumeSettings(step_size=0.02, density_scale=7.0, absorption=0.35, scattering=0.8, shadow_steps=16,
                                   color=(0.9, 0.92, 1.0))
LOOK = replace(BASE, fire_intensity=0.9, temperature_scale=3200.0, fire_threshold=1000.0, fire_light=3.0, multi_scatter=0.7,
               multi_scatter_blur=0.6, anisotropy=0.35, quality="final")


def over_dark(image):
    """The premultiplied frame over a dark backdrop, opaque, in scene-linear ACEScg (write_png applies the sRGB view)."""
    backdrop = np.array([0.004, 0.005, 0.008], np.float32)
    rgb = image[..., :3] + (1 - image[..., 3:4]) * backdrop
    return np.concatenate((rgb, np.ones_like(image[..., 3:4])), axis=-1)


def main():
    from nodebased.imaging import write_png
    scene = s.Scene(volumes=(s.analytic_plume(48, 0),), lights=(KEY,))
    panels = [over_dark(s.render(scene, CAMERA, W, H, samples=1, volume=v, ambient=0.05)) for v in (BASE, LOOK)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_png(OUT, np.concatenate(panels, axis=1))
    print(OUT)


if __name__ == "__main__":
    main()
