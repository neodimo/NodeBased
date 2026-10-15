"""Build the small, reproducible Hot pour plate shot."""
from __future__ import annotations

import json
from pathlib import Path

from nodebased import hotpour
from nodebased.core import Dispatcher

ROOT = Path(__file__).resolve().parent


def build_document(render_size=(960, 540), cells=64, render_backend="gpu"):
    # 64-cell liquid and steam domains give the falling stream enough width to survive
    # meshing while keeping this artist-facing shot practical to bake;
    # all 120 frames remain in the timeline so frames 1, 60 and 120 are representative.
    ops = hotpour.hot_pour_ops(cells, cells, 120, spout_height=0.42, particles_per_cell=2,
                               substeps=2, render_size=render_size)
    commands = []
    for item in ops:
        op = dict(item)
        if op.get("op") == "create" and op.get("id") == "$new:table_top":
            op["params"].update(shadow_catcher="on", sx=1.7, sz=2.2, red=0.28, green=0.17, blue=0.10)
        if op.get("op") == "create" and op.get("id") == "$new:key_light":
            op["params"].update(light_type="Point", shadows="on", tx=0.9, ty=1.4, tz=1.2, intensity=3.0)
        if op.get("op") == "create" and op.get("id") == "$new:camera":
            op["params"].update(ty=1.15, tz=2.8, target_y=0.0, focal=45.0)
        if op.get("op") == "create" and op.get("id") == "$new:render":
            op["params"].update(render_backend=render_backend, render_mode="pathtrace", samples=2,
                                pt_samples=2, max_bounces=2, diffuse_bounces=1, specular_bounces=1)
        if op.get("op") == "connect" and op.get("id") == "$new:scene" and op.get("input") == "object6":
            continue  # the secondary rim light yields this slot to the foreground holdout
        for field in ("id", "source"):
            if isinstance(op.get(field), str):
                op[field] = op[field].replace("$new:", "")
        commands.append(op)
    commands.extend((
        {"op": "create", "id": "front_cup", "type": "Cylinder3D", "params": {
            "cyl_radius": 0.12, "cyl_height": 0.30, "cyl_caps": "closed", "tx": 0.0, "ty": 0.15,
            "tz": 0.20, "red": 0.8, "green": 0.9, "blue": 0.95, "alpha": 1.0, "holdout": "on",
            "cast_shadows": "off"}},
        {"op": "create", "id": "backdrop", "type": "Card3D", "params": {
            "card_width": 4.0, "card_height": 2.5, "ty": 0.7, "tz": -0.55,
            "red": 0.16, "green": 0.19, "blue": 0.22, "spec_amount": 0.0}},
        {"op": "connect", "id": "scene", "input": "object6", "source": "front_cup"},
        {"op": "connect", "id": "scene", "input": "object7", "source": "backdrop"},
    ))
    d = Dispatcher()
    d.execute({"op": "batch", "commands": commands})
    # Plate and shot share the same camera and set. The plate is the solid-only render;
    # Merge overlays the transparent catcher/holdout/liquid render onto it.
    d.execute({"op": "create", "id": "plate", "type": "Read", "params": {
        "path": "plate.png", "colorspace": "Auto", "alpha_mode": "Auto", "missing": "error"}})
    d.execute({"op": "create", "id": "plate_fit", "type": "Reformat", "params": {
        "reformat_type": "to_box", "width": int(render_size[0]), "height": int(render_size[1]),
        "resize_type": "fit", "filter": "bilinear"}})
    d.execute({"op": "connect", "id": "plate_fit", "input": "image", "source": "plate"})
    d.execute({"op": "create", "id": "comp", "type": "Merge", "params": {"operation": "over"}})
    d.execute({"op": "connect", "id": "comp", "input": "A", "source": "render"})
    d.execute({"op": "connect", "id": "comp", "input": "B", "source": "plate_fit"})
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
