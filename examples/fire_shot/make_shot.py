"""Build the deterministic small fuel-fire scene used by render_evidence.py."""
from __future__ import annotations

import json
from pathlib import Path

from nodebased.core import Dispatcher

ROOT = Path(__file__).resolve().parent


def build_document(width=960, height=540):
    d = Dispatcher()
    nodes = {
        "fuel": ("FluidSource3D", dict(src_center_y=.20, src_radius=.22, src_fuel=2.0,
                                      src_temperature=1.5, src_density=.18, src_vel_y=.2)),
        "fire": ("FluidSolver3D", dict(division_size=.0625, bounds_min_x=-1, bounds_min_y=0,
                                      bounds_min_z=-1, bounds_max_x=1, bounds_max_y=2,
                                      bounds_max_z=1, auto_resize=0, fire=1, burn_rate=1.2,
                                      smoke_output=.25, temperature_output=3, vorticity=.25,
                                      max_iterations=200, pressure="auto")),
        "cache": ("FluidCache3D", {}),
        "floor": ("Cube3D", dict(cube_size=1, sx=3.5, sy=.08, sz=3.5, ty=-.07,
                                red=.19, green=.18, blue=.16)),
        "stone": ("Sphere3D", dict(sphere_radius=.29, tx=.58, ty=.25, tz=.06,
                                   red=.42, green=.43, blue=.45, rows=12, columns=24)),
        "fuel_block": ("Cube3D", dict(cube_size=.22, tx=0, ty=.10, tz=0,
                                      red=.09, green=.06, blue=.035)),
        "fill": ("Light3D", dict(light_type="Directional", intensity=.12, tx=-2, ty=3, tz=4,
                                 shadows="off", red=.35, green=.42, blue=.6)),
        "scene": ("Scene3D", {}),
        "camera": ("Camera3D", dict(tx=2.2, ty=1.3, tz=3.8, target_y=.45, focal=46)),
        "render": ("Render3D", dict(width=width, height=height, render_backend="gpu",
                                    render_mode="pathtrace", pt_samples=32, sampling="fixed",
                                    max_bounces=3, diffuse_bounces=2, specular_bounces=1,
                                    alpha=1.0, ambient=.13, volume_step_size=.045,
                                    volume_shadow_steps=4, volume_density_scale=1.0,
                                    volume_fire_intensity=.05, volume_temperature_scale=500,
                                    volume_flame_intensity=.5, volume_flame_temperature=1900,
                                    volume_fire_light_meshes=60.0)),
    }
    for name, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": name, "type": kind, "params": params})
    for target, slot, source in (
        ("fire", "fluid", "fuel"), ("cache", "volume", "fire"),
        ("scene", "object0", "floor"), ("scene", "object1", "stone"),
        ("scene", "object2", "cache"), ("scene", "object3", "fill"),
        ("scene", "object4", "fuel_block"),
        ("render", "scene", "scene"), ("render", "camera", "camera"),
    ):
        d.execute({"op": "connect", "id": target, "input": slot, "source": source})
    d.document["view"] = "render"
    d.document["time"] = {"first": 1, "last": 24, "current": 1, "fps": 24.0}
    return d.document


if __name__ == "__main__":
    ROOT.joinpath("fire_shot.nbcomp").write_text(json.dumps(build_document(), indent=2) + "\n")
