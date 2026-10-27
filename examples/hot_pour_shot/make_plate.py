"""Render the solid set of the shot once as its camera-matched plate."""
from __future__ import annotations

import copy
from pathlib import Path

from make_shot import ROOT, build_document
from nodebased.imaging import Evaluator, write_png


def main():
    doc = copy.deepcopy(build_document())
    scene = doc["nodes"]["scene"]
    for slot in ("object0", "object1", "object2"):
        scene["inputs"].pop(slot, None)  # liquid, whitewater, steam
    scene["inputs"].pop("object3", None)       # the live render supplies glass and liquid together
    scene["inputs"].pop("object6", None)
    scene["inputs"].pop("object7", None)       # the continuity stream belongs to the transparent render only
    doc["nodes"]["table_top"]["params"]["shadow_catcher"] = "off"
    render = doc["nodes"]["render"]["params"]
    render.update(render_backend="cpu", render_mode="raster", samples=1,
                  red=0.58, green=0.68, blue=0.73, alpha=1.0)
    pixels = Evaluator().evaluate_raster(doc, "render", frame=1).pixels
    write_png(ROOT / "plate.png", pixels)


if __name__ == "__main__":
    main()
