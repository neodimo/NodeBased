"""Shared pieces of the demo shot: the plate scene, its camera move and the document builders.

The plate is rendered by NodeBased itself (there is no filmed footage in the repository). Everything
here is plain data and deterministic; the scripts next to this file call it.
"""
from __future__ import annotations

import copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES = 24
PLATE_SIZE = (480, 270)
PASSES = "beauty,normals,depth,motion,object_id"

# Frame 1 and frame 24 of the camera move: a slow dolly in with a small pan to the right.
# The values are the ground truth the tracker is compared against.
CAMERA_START = dict(tx=0.0, ty=1.1, tz=8.0, target_x=-0.2, target_y=0.0, target_z=0.0)
CAMERA_END = dict(tx=0.45, ty=1.1, tz=6.6, target_x=0.25, target_y=0.0, target_z=0.0)
FOCAL = 22.539097818315216


def camera_at(frame, move=FRAMES):
    """The rendering camera on `frame`; the move always spans `move` frames, however many are rendered."""
    t = (frame - 1) / max(move - 1, 1)
    return {key: CAMERA_START[key] + (CAMERA_END[key] - CAMERA_START[key]) * t for key in CAMERA_START}




def _create(commands, key, kind, params=None, **extra):
    commands.append({"op": "create", "id": key, "type": kind, "params": params or {}, **extra})


def scene_commands(width, height):
    """Create the 3D scene of the plate: ground, set pieces, cube, sphere, backdrop, lights, camera.

    The meshes are named: an object's id in the export hashes its name, so two unnamed cubes (or two
    cards) would share one id and the verifier could not tell their measurements apart."""
    c = []
    _create(c, "checker", "Checker", {"width": 256, "height": 256, "size": 8})
    _create(c, "ground", "Card3D", dict(card_width=40.0, card_height=40.0, rx=-90.0, ty=-1.2, tz=0.0,
                                         red=1.0, green=1.0, blue=1.0), name="Ground")
    c.append({"op": "connect", "id": "ground", "input": "image", "source": "checker"})
    _create(c, "pillar", "Cylinder3D", dict(tx=3.2, ty=-0.2, tz=-1.6), name="Pillar")
    _create(c, "crate", "Cube3D", dict(cube_size=1.0, tx=-3.4, ty=-0.7, tz=-1.2, ry=-18.0,
                                        red=0.55, green=0.45, blue=0.3), name="Crate")
    _create(c, "backdrop", "Card3D", dict(card_width=50.0, card_height=10.0, tx=0.0, ty=2.8, tz=-4.5,
                                           red=0.18, green=0.2, blue=0.26), name="Backdrop")
    _create(c, "cube", "Cube3D", dict(cube_size=1.3, tx=-1.2, ty=-0.55, tz=0.5, ry=28.0,
                                       red=0.9, green=0.35, blue=0.25), name="Cube")
    _create(c, "ball", "Sphere3D", dict(sphere_radius=0.75, tx=1.3, ty=-0.45, tz=0.2,
                                         red=0.35, green=0.85, blue=0.45), name="Ball")
    _create(c, "key", "Light3D", dict(light_type="Directional", tx=2.5, ty=4.0, tz=4.0, intensity=1.1))
    _create(c, "sky", "Light3D", dict(light_type="Environment", intensity=0.35))
    _create(c, "camera", "Camera3D", dict(focal=FOCAL, **CAMERA_START))
    _create(c, "scene", "Scene3D")
    for slot, key in enumerate(("ground", "pillar", "crate", "backdrop", "cube", "ball", "key", "sky")):
        c.append({"op": "connect", "id": "scene", "input": f"object{slot}", "source": key})
    _create(c, "render", "Render3D", dict(width=width, height=height, ambient=0.25, samples=1,
                                           render_output="multichannel", render_backend="cpu",
                                           render_mode="raster", passes=PASSES))
    c.append({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    c.append({"op": "connect", "id": "render", "input": "camera", "source": "camera"})
    return c


def animate_camera(commands, camera="camera", frames=FRAMES, values=None):
    """Key the camera on frames 1..`frames`."""
    # `values(frame)` gives the parameter dict (default: the truth)
    for param in CAMERA_START:
        keys = [{"frame": f, "value": float((values or camera_at)(f)[param])} for f in range(1, frames + 1)]
        commands.append({"op": "set_curve", "id": camera, "param": param,
                         "curve": {"interpolation": "linear", "keys": keys}})


GROUND_Y = -1.2
GRID = 40.0 / 32          # one checker square on the ground card, scene units
CUBE = dict(size=1.3, centre=(-1.2, -0.55, 0.5), ry=28.0)
CRATE = dict(size=1.0, centre=(-3.4, -0.7, -1.2), ry=-18.0)


def _top_corners(box):
    import math
    half, (cx, cy, cz) = box["size"] / 2, box["centre"]
    c, s = math.cos(math.radians(box["ry"])), math.sin(math.radians(box["ry"]))
    return [(cx + c * x + s * z, cy + half, cz - s * x + c * z)
            for x, z in ((-half, -half), (half, -half), (half, half), (-half, half))]


def landmarks():
    """Survey points of the set in world coordinates: ground checker corners and box top corners."""
    points = [(i * GRID, GROUND_Y, k * GRID) for i in range(-4, 5) for k in range(-3, 4)]
    return points + _top_corners(CUBE) + _top_corners(CRATE)


BRIEF = "replace the backdrop with a rainy street, keep the cube and sphere, match the camera"
SEED = 7


def default_cache():
    import os
    return Path(os.environ.get("NODEBASED_CACHE", Path.home() / ".cache" / "nodebased")) / "demo_shot"


def plate_dir(base=None):
    return Path(base) / "plate" if base else default_cache() / "plate"


def plate_pattern(folder):
    return str(Path(folder) / "plate.%04d.exr")


def build_graph_commands(width, height, keys, frames=FRAMES):
    """The shot graph. `keys` maps frame to the solved camera's Camera3D parameter values."""
    c = []
    _create(c, "plate", "Read", dict(path="plate/plate.%04d.exr"), name="Plate (rendered by NodeBased)")
    # The layout is everything the plate shows (set pieces and backdrop included): the exported SceneState
    # must describe the plate, or no check could tell a faithful result from a different picture.
    for command in scene_commands(width, height):
        if command["id"] in ("camera", "scene", "render"):
            continue
        c.append(command)
    _create(c, "camera", "Camera3D", dict(focal=FOCAL, **CAMERA_START), name="Solved camera")
    for param in ("tx", "ty", "tz", "target_x", "target_y", "target_z", "roll", "focal"):
        c.append({"op": "set_curve", "id": "camera", "param": param, "curve": {"interpolation": "linear",
                  "keys": [{"frame": f, "value": float(keys[f][param])} for f in range(1, frames + 1)]}})
    _create(c, "scene", "Scene3D", name="Layout")
    for slot, key in enumerate(("ground", "pillar", "crate", "backdrop", "cube", "ball", "key", "sky")):
        c.append({"op": "connect", "id": "scene", "input": f"object{slot}", "source": key})
    # SceneState + ControlBundle export, then the render the export reads its size and passes from
    _create(c, "export", "WriteGeo3D", name="SceneState and ControlBundle export")
    c.append({"op": "connect", "id": "export", "input": "scene", "source": "scene"})
    c.append({"op": "connect", "id": "export", "input": "camera", "source": "camera"})
    _create(c, "render", "Render3D", dict(width=width, height=height, ambient=0.25, samples=1,
                                           render_output="multichannel", render_backend="cpu",
                                           render_mode="raster", passes=PASSES))
    c.append({"op": "connect", "id": "render", "input": "scene", "source": "export"})
    c.append({"op": "connect", "id": "render", "input": "camera", "source": "camera"})
    _create(c, "generate", "Generate", dict(provider="reproject", seed=SEED, text=BRIEF,
                                             output_path="generated.%04d.exr"))
    c.append({"op": "connect", "id": "generate", "input": "image", "source": "plate"})
    _create(c, "grade", "Grade", dict(exposure=-0.15, multiply=0.95), name="Grade the generated backdrop")
    _create(c, "merge", "Merge", name="Merge over the plate")
    c.append({"op": "connect", "id": "merge", "input": "A", "source": "grade"})
    c.append({"op": "connect", "id": "merge", "input": "B", "source": "plate"})
    _create(c, "write", "Write", dict(path="shot.%04d.exr", file_type="exr", bit_depth="half"))
    c.append({"op": "connect", "id": "write", "input": "image", "source": "merge"})
    c.append({"op": "time", "first": 1, "last": frames, "current": 1})
    return c


def build_graph(width, height, keys, frames=FRAMES):
    from nodebased.core import Dispatcher
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": build_graph_commands(width, height, keys, frames)})
    return dispatcher.document


def save_graph(path, width, height, keys, frames=FRAMES, plate_folder=None):
    """Write the shot graph; its Read node points at `plate_folder` (stored relative to `path`)."""
    from nodebased.core import atomic_save
    document = build_graph(width, height, keys, frames)
    document["nodes"]["plate"]["params"]["path"] = plate_pattern(plate_folder or Path(path).resolve().parent / "plate")
    atomic_save(path, document)
