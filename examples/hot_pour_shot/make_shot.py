"""Build the small, reproducible Hot pour plate shot."""
from __future__ import annotations

import json
from pathlib import Path

from nodebased import hotpour
from nodebased.core import Dispatcher

ROOT = Path(__file__).resolve().parent


def build_document(render_size=(960, 540), cells=64, render_backend="gpu"):
    # A 64-cell liquid domain and 96-cell steam domain keep the shot practical while
    # meshing while keeping this artist-facing shot practical to bake;
    # all 120 frames remain in the timeline so frames 1, 60 and 120 are representative.
    ops = hotpour.hot_pour_ops(cells, 96, 120, spout_height=0.62, particles_per_cell=2,
                               substeps=2, render_size=render_size)
    commands = []
    for item in ops:
        op = dict(item)
        if op.get("op") == "connect" and op.get("id") == "$new:scene" and op.get("input") == "object6":
            continue  # the glass and liquid now share one live 3D render; a cup holdout would hide the pool
        if op.get("op") == "connect" and op.get("id") == "$new:scene" and op.get("input") == "object1":
            continue  # whitewater flecks obscure the tabletop and do not read at this camera distance
        if op.get("op") == "connect" and op.get("id") == "$new:steam_source" and op.get("input") == "geo":
            continue  # the shot emits vapour just above the rim, clear of the visible liquid pool
        if op.get("op") == "create" and op.get("id") == "$new:table_top":
            op["params"].update(shadow_catcher="off", sx=1.7, sz=2.2, red=0.28, green=0.17, blue=0.10)
        if op.get("op") == "create" and op.get("id") == "$new:key_light":
            op["params"].update(light_type="Rect", shadows="on", tx=0.9, ty=1.4, tz=1.2, intensity=3.0,
                                area_width=1.0, area_height=1.0, light_samples=30)
        if op.get("op") == "create" and op.get("id") == "$new:rim_light":
            op["params"].update(shadows="off")
        if op.get("op") == "create" and op.get("id") == "$new:spout":
            op["params"].update(src_radius=0.022)
        if op.get("op") == "create" and op.get("id") == "$new:surface":
            op["params"].update(smoothing=3, roughness=0.06)
        if op.get("op") == "create" and op.get("id") == "$new:steam_source":
            op["params"].update(fluid_emit_from="sphere", src_center_x=0.09, src_center_y=0.34,
                                src_center_z=-0.04, src_radius=0.075, src_vel_y=0.015,
                                src_density=0.35, src_noise_amount=0.0, src_noise_scale=0.5)
        if op.get("op") == "create" and op.get("id") == "$new:steam_lift":
            op["params"].update(buoyancy_lift=0.04, strength=0.0)
        if op.get("op") == "create" and op.get("id") == "$new:steam":
            op["params"].update(vorticity=0.04, dissipation=0.03)
        if op.get("op") == "create" and op.get("id") in ("$new:glass_view_wall", "$new:glass_view_base"):
            op["params"].update(alpha=0.05, spec_amount=0.10)
        if op.get("op") == "create" and op.get("id") == "$new:camera":
            op["params"].update(ty=1.15, tz=2.8, target_y=0.0, focal=45.0)
        if op.get("op") == "create" and op.get("id") == "$new:render":
            op["params"].update(render_backend=render_backend, render_mode="pathtrace", samples=2,
                                pt_samples=128, max_bounces=8, diffuse_bounces=1, specular_bounces=4,
                                transmission_bounces=8, volume_density_scale=1.5,
                                volume_shadow_steps=1, volume_shadow_density=0.0,
                                volumes="off",
                                ambient=1.5,
                                sampling="adaptive", min_samples=32, max_samples=128,
                                adaptive_pass_size=64, noise_threshold=0.01, denoise_iterations=8)
        for field in ("id", "source"):
            if isinstance(op.get(field), str):
                op[field] = op[field].replace("$new:", "")
        commands.append(op)
    commands.extend((
        {"op": "create", "id": "pour_bridge", "type": "Cylinder3D", "params": {
            "cyl_radius": 0.012, "cyl_height": 0.60, "cyl_caps": "closed", "tx": 0.0, "ty": 0.36,
            "tz": 0.03, "material": "liquid", "ior": 1.333, "absorption_red": 0.85,
            "absorption_green": 0.5, "absorption_blue": 0.15, "absorption_distance": 0.08,
            "roughness": 0.02}},
        {"op": "connect", "id": "scene", "input": "object7", "source": "pour_bridge"},
    ))
    for frame, x, z in ((1, -0.035, 0.0), (43, 0.03, 0.025), (84, 0.035, -0.025), (120, -0.02, 0.0)):
        commands.extend((
            {"op": "set_key", "id": "pour_bridge", "param": "tx", "frame": frame, "value": x},
            {"op": "set_key", "id": "pour_bridge", "param": "tz", "frame": frame, "value": z + 0.01},
        ))
    d = Dispatcher()
    d.execute({"op": "batch", "commands": commands})
    # Plate and shot share the same camera and set. The path tracer renders the liquid, glass and table;
    # a deterministic raster volume layer supplies the soft steam separately.
    d.execute({"op": "create", "id": "plate", "type": "Read", "params": {
        "path": "plate.png", "colorspace": "Auto", "alpha_mode": "Auto", "missing": "error"}})
    d.execute({"op": "create", "id": "plate_fit", "type": "Reformat", "params": {
        "reformat_type": "to_box", "width": int(render_size[0]), "height": int(render_size[1]),
        "resize_type": "fit", "filter": "bilinear"}})
    d.execute({"op": "connect", "id": "plate_fit", "input": "image", "source": "plate"})
    d.execute({"op": "create", "id": "base_comp", "type": "Merge", "params": {"operation": "over"}})
    d.execute({"op": "connect", "id": "base_comp", "input": "A", "source": "render"})
    d.execute({"op": "connect", "id": "base_comp", "input": "B", "source": "plate_fit"})
    d.execute({"op": "create", "id": "steam_scene", "type": "Scene3D"})
    d.execute({"op": "connect", "id": "steam_scene", "input": "object0", "source": "steam_cache"})
    d.execute({"op": "connect", "id": "steam_scene", "input": "object1", "source": "key_light"})
    d.execute({"op": "create", "id": "steam_render", "type": "Render3D", "params": {
        "width": int(render_size[0]), "height": int(render_size[1]), "render_backend": render_backend,
        "render_mode": "raster", "alpha": 0.0, "ambient": 1.5, "volumes": "on",
        "volume_step_size": 0.009, "volume_density_scale": 4.0, "volume_shadow_density": 0.0,
        "volume_shadow_steps": 1, "volume_scattering": 1.0, "volume_absorption": 0.05}})
    d.execute({"op": "connect", "id": "steam_render", "input": "scene", "source": "steam_scene"})
    d.execute({"op": "connect", "id": "steam_render", "input": "camera", "source": "camera"})
    d.execute({"op": "create", "id": "comp", "type": "Merge", "params": {"operation": "over"}})
    d.execute({"op": "connect", "id": "comp", "input": "A", "source": "steam_render"})
    d.execute({"op": "connect", "id": "comp", "input": "B", "source": "base_comp"})
    d.document["view"] = "comp"
    # Dispatcher creation normalizes Read paths against the process cwd; keep this portable
    # example relative to its own folder instead.
    d.document["nodes"]["plate"]["params"]["path"] = "plate.png"
    d.document["time"] = {"first": 1, "last": 120, "current": 1, "fps": 24.0}
    return d.document


def main():
    ROOT.joinpath("hot_pour_shot.nbcomp").write_text(json.dumps(build_document(), indent=2) + "\n")


if __name__ == "__main__":
    main()
