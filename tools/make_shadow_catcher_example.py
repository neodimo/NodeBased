"""Rebuild examples/shadow_catcher: a CG ball put onto the filmed plate of examples/holdout_plate.png.

The project follows the three switches of "Putting CG onto a plate" (docs/3D_FOUNDATION.md): a road card set to **Shadow
catcher** (the ball's shadow lands on the plate), a stand-in card for the real lamppost set to **Holdout** (the post passes in
front of the ball), and the CG's lights matched to the plate's sky (a warm key from the left, a cool fill from the right).
A Merge node puts the render over the plate. Run it from the repository root:

    python tools/make_shadow_catcher_example.py          # writes examples/shadow_catcher/shadow_catcher.nbcomp

The plate is the generated one that tools/render_holdout_reference.py writes (fixed seed, no licence issue).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodebased.core import Dispatcher, atomic_save  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "examples" / "shadow_catcher"
PLATE_PATH = str(Path(__file__).resolve().parents[1] / "examples" / "holdout_plate.png")   # saved relative to the project
VAPERTURE = 18.672
FOV = 36.0                                          # the plate's camera (tools/render_holdout_reference.py)
FOCAL = VAPERTURE / (2.0 * math.tan(math.radians(FOV) / 2.0))
POST = dict(card_width=0.25, card_height=3.8, tx=0.325, ty=0.7, tz=2.2)     # the stand-in for the lamppost, as in the R2 recipe


def build_document():
    d = Dispatcher()
    nodes = (
        ("plate", "Read", "Plate", (-480, -360), dict(path=PLATE_PATH)),
        ("ball", "Sphere3D", "CG ball", (-480, -260),
         dict(sphere_radius=1.1, ty=0.9, red=0.9, green=0.35, blue=0.15, material="pbr", pbr_roughness=0.28, pbr_specular=0.9,
              rows=32, columns=64)),
        ("road", "Card3D", "Road (shadow catcher)", (-480, -160),
         dict(card_width=60.0, card_height=60.0, ty=-0.2, tz=-20.0, rx=-90.0, red=0.05, green=0.05, blue=0.05,
              shadow_catcher="on")),
        ("post", "Card3D", "Lamppost stand-in (holdout)", (-480, -60),
         dict(**POST, red=0.6, green=0.2, blue=0.2, holdout="on", cast_shadows="off")),
        ("key", "Light3D", "Key light", (-480, 40),
         dict(light_type="Directional", tx=-5.0, ty=6.0, tz=4.0, target_y=0.9, red=1.0, green=0.85, blue=0.7, intensity=2.4,
              shadows="on")),
        ("fill", "Light3D", "Fill light", (-480, 140),
         dict(light_type="Directional", tx=5.0, ty=2.0, tz=3.0, target_y=0.9, red=0.35, green=0.5, blue=0.9, intensity=0.7,
              shadows="on")),
        ("camera", "Camera3D", "Camera", (-200, 40), dict(tx=0.0, ty=1.5, tz=7.0, target_y=0.9, focal=FOCAL, vaperture=VAPERTURE)),
        ("scene", "Scene3D", "Scene", (-200, -120), {}),
        ("render", "Render3D", "Render CG", (80, -120),
         dict(width=480, height=270, samples=2, ambient=0.12, render_mode="raytrace", render_backend="cpu", alpha=0.0)),
        ("merge", "Merge", "CG over plate", (360, -240), dict(operation="over")),
    )
    for key, kind, name, pos, params in nodes:
        d.execute({"op": "create", "id": key, "type": kind, "name": name, "pos": list(pos), "params": params})
    for slot, source in (("object0", "ball"), ("object1", "road"), ("object2", "post"), ("object3", "key"), ("object4", "fill")):
        d.execute({"op": "connect", "id": "scene", "input": slot, "source": source})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "camera"})
    d.execute({"op": "connect", "id": "merge", "input": "A", "source": "render"})
    d.execute({"op": "connect", "id": "merge", "input": "B", "source": "plate"})
    d.execute({"op": "view", "id": "merge"})
    return d.document


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "shadow_catcher.nbcomp"
    atomic_save(target, build_document())
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
