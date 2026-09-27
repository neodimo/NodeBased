"""Render docs/images/liquid_dam_break.png: a FLIP dam break drawn with the liquid material, foam and spray.

Left: the raster mode's screen-space approximation. Right: the ray-traced mode (refraction, Fresnel reflection,
absorption, thin sheets). Both show the splash particles as foam. The dam is a block of liquid released at one end of a
box over a checkered floor; the frame is the splash after it reaches the far wall. CPU reference render, deterministic
(fixed solver seed); run it from the repository root: `python tools/render_liquid_reference.py [--frame 20]`.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "liquid_dam_break.png"
W, H = 288, 216
SKY = (0.45, 0.58, 0.75, 1.0)
CAMERA = s.Camera(s.Transform3D(s.Vec3(1.2, 1.1, 2.6)), s.Vec3(0.3, 0.4, 0.0), fov=42.0)
KEY = s.Light("Directional", color=(1.0, 0.96, 0.9), intensity=1.0, position=s.Vec3(-2, 5, 3), target=s.Vec3(0, 0.3, 0))
DOMAIN = {"division_size": 0.1, "bounds_min_x": -1.6, "bounds_min_y": 0.0, "bounds_min_z": -0.8,
          "bounds_max_x": 1.6, "bounds_max_y": 1.6, "bounds_max_z": 0.8}


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def dam_break_graph():
    d = Dispatcher()
    make(d, block=("Cube3D", {"cube_size": 1.0, "tx": -1.1, "ty": 0.6, "sx": 0.8, "sy": 1.2, "sz": 1.5}),
         src=("FluidSource3D", {"fluid_type": "liquid", "fluid_emit_from": "volume", "end_frame": 1}),
         sol=("FluidLiquidSolver3D", {**DOMAIN, "substeps": 2}),
         surface=("FluidSurface3D", {"smoothing": 2}),
         foam=("FluidFoam3D", {"foam_speed": 0.5, "foam_curvature": 1.0}),
         draw=("ParticleRender3D", {"representation": "foam", "foam_density": 0.8, "spray_size": 5.0}))
    wire(d, "src", "geo", "block")
    wire(d, "sol", "fluid", "src")
    wire(d, "surface", "particles", "sol")
    wire(d, "foam", "particles", "sol")
    wire(d, "draw", "particles", "foam")
    return d


def checker_floor():
    texels = np.zeros((128, 128, 4), np.float32)
    row, col = np.indices((128, 128))
    even = ((row // 16) + (col // 16)) % 2 == 0
    texels[even] = (0.85, 0.85, 0.82, 1.0)
    texels[~even] = (0.22, 0.26, 0.3, 1.0)
    return s._card(6.0, 4.0, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, 0, 0), s.Vec3(-90, 0, 0)), texels)


def over_backdrop(image):
    rgb = image[..., :3] + (1 - image[..., 3:4]) * np.asarray(SKY[:3], np.float32)
    return np.concatenate((rgb, np.ones_like(image[..., 3:4])), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=20)
    frame = parser.parse_args().frame
    graph, evaluator = dam_break_graph(), Evaluator()
    at = lambda key: evaluator.evaluate_raster(graph.document, key, frame=frame, typed=True)
    surface, foam = at("surface"), at("draw")
    print(f"frame {frame}: {len(surface.triangles)} triangles, {len(foam)} foam particles")
    scene = s.Scene((checker_floor(), replace(surface, ior=1.333, reflection=1.0,
                                              absorption_color=(0.5, 0.78, 0.9), absorption_distance=1.2, roughness=0.02)),
                    (KEY,), particles=(foam,))
    panels = [over_backdrop(s.render(scene, CAMERA, W, H, SKY, samples=2, mode=mode, ambient=0.25))
              for mode in ("raster", "raytrace")]
    from nodebased.imaging import write_png
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_png(OUT, np.concatenate(panels, axis=1))
    print(OUT)


if __name__ == "__main__":
    main()
