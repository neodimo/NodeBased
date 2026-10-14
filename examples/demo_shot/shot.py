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
# The lens plate (make_plate.py --k1): the ideal render is made this much larger in each direction and bent into the
# plate's format with the Camera3D lens model, so the barrel-distorted corners show the set and not black.
LENS_K1 = -0.03
OVERSCAN = 1.35
# The distorted plate of the lens round trip (demo_shot_lens.nbcomp): 640 wide (the verifier's 2 pixel object check holds there and at 704, and
# not from 768 up, see the README) through
# a barrel lens with a small second term. `--plate distorted` on the run scripts selects it.
DISTORTED_K1, DISTORTED_K2 = -0.03, 0.004
DISTORTED_SIZE = (640, 360)


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


def distorted_plate_dir(base=None):
    return Path(base) / "plate_distorted" if base else default_cache() / "plate_distorted"


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


# ----------------------------------------------------------------------------- the lens round trip (demo_shot_lens.nbcomp)

LENS_GRAPH = HERE / "demo_shot_lens.nbcomp"
LENS_SOLVE = HERE / "solve_lens.json"
LENS_GRAPH_DPVO = HERE / "demo_shot_lens_dpvo.nbcomp"     # the same graph with the DPV-SLAM solve (not the default)
LENS_SOLVE_DPVO = HERE / "solve_lens_dpvo.json"
SOLVE_TYPED_FOCAL_SCALE = 1.2     # the artist types a focal length 20 percent off; the solve recovers it
LENS_KEYS = ("lens_k1", "lens_k2", "lens_k3", "lens_p1", "lens_p2", "lens_cx", "lens_cy")


def resolve_plate(choice):
    """`--plate`: "plain" (the default) or "distorted" pick the shot (graph and plate folder); anything else is a plate
    folder of the plain shot. Returns (graph file, plate folder, is the lens shot)."""
    if choice in (None, "plain"):
        return HERE / "demo_shot.nbcomp", plate_dir(), False
    if choice == "distorted":
        return LENS_GRAPH, distorted_plate_dir(), True
    return HERE / "demo_shot.nbcomp", Path(choice), False


def is_lens_graph(document):
    """True for a shot graph with the lens round trip (an Undistort node)."""
    return any(node["type"] == "Undistort" for node in document["nodes"].values())


def lens_of(k1, k2):
    return {"lens_k1": float(k1), "lens_k2": float(k2)}


def lens_for_format(lens, scale):
    """The same physical lens read on a format `scale` times as tall (the pixel size unchanged): the model's units are
    half the format height, so a radial term of order n scales with scale**n, tangential terms with scale and the
    principal point with 1/scale."""
    out = dict(lens)
    for name, power in (("lens_k1", 2), ("lens_k2", 4), ("lens_k3", 6), ("lens_p1", 1), ("lens_p2", 1)):
        if name in out:
            out[name] = float(out[name]) * scale ** power
    for name in ("lens_cx", "lens_cy"):
        if name in out:
            out[name] = float(out[name]) / scale
    return out


def overscan_margins(lens, size, extra=2, multiple=16):
    """Pixels to add on each side of the plate so that Undistort's grown frame fits: (mx, my), from the window the
    node itself would make (nodebased.lensmodel.undistorted_window) plus `extra` pixels, raised until the overscan frame
    is a multiple of `multiple` pixels each way (what a diffusion model's working size wants)."""
    from nodebased import lensmodel
    from nodebased.raster import Region
    width, height = size
    box = Region(0, 0, width, height)
    x, y, w, h = lensmodel.undistorted_window(lensmodel.from_params(lens), box, box)
    out = []
    for grow, length in ((max(-x, x + w - width), width), (max(-y, y + h - height), height)):
        margin = int(grow + extra)
        while (length + 2 * margin) % multiple:
            margin += 1
        out.append(margin)
    return tuple(out)


def overscan_camera_params(lens, size, margins):
    """The Camera3D film back and lens of the overscan camera: the same focal length on a taller film back keeps
    the pixel size, and the lens is the plate's, restated in the larger format's units."""
    from nodebased import filmback
    width, height = size
    scale = (height + 2 * margins[1]) / height
    return {"vaperture": filmback.DEFAULT_VAPERTURE * scale, **lens_for_format(lens, scale)}, scale


def pick_survey(size, lens, frames=FRAMES, plate_folder=None, radius=8, min_std=0.03):
    """Two picks far apart that stay in view on every frame, and the real distance between them: the survey the camera
    solve is given. Each pick is the true projection through the lens, rounded to a whole pixel (a click), on frame 1.
    With `plate_folder`, only points whose window on the plate has texture on every frame checked are offered (a click
    on a flat colour cannot be followed by the tracker)."""
    from dataclasses import replace
    import numpy as np
    from nodebased import scene3d
    import solve_camera
    width, height = size
    world = np.array(landmarks(), dtype=np.float64)
    shown = []
    for frame in (1, (frames + 1) // 2, frames):
        pixels, depth = scene3d.project_lens(replace(solve_camera.truth_camera(frame), **lens), width, height, world)
        shown.append((depth > 0.5) & (pixels[:, 0] > 40) & (pixels[:, 1] > 40)
                     & (pixels[:, 0] < width - 40) & (pixels[:, 1] < height - 40))
    if plate_folder is not None:
        import lens_check
        textured = []
        for frame, pixels_shown in zip((1, (frames + 1) // 2, frames), shown):
            gray = lens_check.read_display(Path(plate_folder) / (plate_pattern(".").split("/")[-1] % frame)).mean(axis=2)
            pixels, _ = scene3d.project_lens(replace(solve_camera.truth_camera(frame), **lens), width, height, world)
            ok = np.zeros(len(world), bool)
            for n in np.flatnonzero(pixels_shown):
                x, y = int(round(pixels[n, 0])), int(round(pixels[n, 1]))
                ok[n] = float(gray[y - radius:y + radius + 1, x - radius:x + radius + 1].std()) > min_std
            textured.append(ok)
        shown = [a & b for a, b in zip(shown, textured)]
    visible = np.flatnonzero(np.logical_and.reduce(shown))
    best = max(((a, b) for a in visible for b in visible if a < b),
               key=lambda pair: np.linalg.norm(world[pair[0]] - world[pair[1]]))
    first, _ = scene3d.project_lens(replace(solve_camera.truth_camera(1), **lens), width, height, world[list(best)])
    picks = [{"name": name, "frame": 1, "x": float(np.rint(pixel[0])), "y": float(np.rint(pixel[1])), "position": None}
             for name, pixel in zip(("Survey A", "Survey B"), first)]
    return picks, float(np.linalg.norm(world[best[0]] - world[best[1]]))


def survey_json(picks, distance):
    import json
    return json.dumps({"picks": picks, "distances": [{"a": picks[0]["name"], "b": picks[1]["name"],
                                                       "distance": float(distance)}]}, sort_keys=True)


def build_lens_graph_commands(size, keys, lens, margins, survey="", frames=FRAMES, solve_params=None):
    """The shot graph of the distorted plate: plate -> Undistort (overscan) -> Generate, the conditioning and the comp in the
    undistorted frame -> Redistort -> Write.

    `keys` maps frame to the solved camera's parameter values (as `build_graph_commands`), `lens` is the solved lens
    ({"lens_k1": .., "lens_k2": ..}, in units of half the plate's height). The export and the render use a second camera, the
    same pose and focal length on a film back `margins` taller and wider (so Render3D draws the overscan the Undistort node
    grows) with the same lens restated in that larger format's units; the SceneState therefore carries a lens a provider can
    apply to its overscan pictures to land on the plate. The generated read is placed in the plate's format by a Reformat
    that keeps its overscan, so Merge sees matching formats and the Redistort returns to the plate."""
    width, height = size
    big = (width + 2 * margins[0], height + 2 * margins[1])
    c = []
    _create(c, "plate", "Read", dict(path="plate_distorted/plate.%04d.exr"), name="Plate (distorted, rendered by NodeBased)")
    for command in scene_commands(*big):
        if command["id"] in ("camera", "scene", "render"):
            continue
        c.append(command)
    _create(c, "camera", "Camera3D", dict(focal=FOCAL, **CAMERA_START, **lens), name="Solved camera (plate lens)")
    for param in ("tx", "ty", "tz", "target_x", "target_y", "target_z", "roll", "focal"):
        c.append({"op": "set_curve", "id": "camera", "param": param, "curve": {"interpolation": "linear",
                  "keys": [{"frame": f, "value": float(keys[f][param])} for f in range(1, frames + 1)]}})
    film, _scale = overscan_camera_params(lens, size, margins)
    _create(c, "camera_os", "Camera3D", dict(focal=FOCAL, **CAMERA_START, **film),
            name="Overscan camera (the plate lens on a larger film back)")
    for param in ("tx", "ty", "tz", "target_x", "target_y", "target_z", "roll", "focal"):
        c.append({"op": "set_curve", "id": "camera_os", "param": param, "curve": {"interpolation": "linear",
                  "keys": [{"frame": f, "value": float(keys[f][param])} for f in range(1, frames + 1)]}})
    _create(c, "solve", "CameraSolve3D", dict(
        solve_focal="estimate", solve_distortion="k1 k2", survey=survey, focal=FOCAL * SOLVE_TYPED_FOCAL_SCALE,
        anchor_tx=CAMERA_START["tx"], anchor_ty=CAMERA_START["ty"], anchor_tz=CAMERA_START["tz"],
        anchor_target_x=CAMERA_START["target_x"], anchor_target_y=CAMERA_START["target_y"],
        anchor_target_z=CAMERA_START["target_z"], **(solve_params or {})), name="Camera solve (lens estimated)")
    c.append({"op": "connect", "id": "solve", "input": "image", "source": "plate"})
    _create(c, "undistort", "Undistort", name="Undistort (overscan)")
    c.append({"op": "connect", "id": "undistort", "input": "image", "source": "plate"})
    c.append({"op": "connect", "id": "undistort", "input": "camera", "source": "camera"})
    _create(c, "scene", "Scene3D", name="Layout")
    for slot, key in enumerate(("ground", "pillar", "crate", "backdrop", "cube", "ball", "key", "sky")):
        c.append({"op": "connect", "id": "scene", "input": f"object{slot}", "source": key})
    _create(c, "export", "WriteGeo3D", name="SceneState and ControlBundle export (overscan)")
    c.append({"op": "connect", "id": "export", "input": "scene", "source": "scene"})
    c.append({"op": "connect", "id": "export", "input": "camera", "source": "camera_os"})
    _create(c, "render", "Render3D", dict(width=big[0], height=big[1], ambient=0.25, samples=1,
                                           render_output="multichannel", render_backend="cpu",
                                           render_mode="raster", passes=PASSES))
    c.append({"op": "connect", "id": "render", "input": "scene", "source": "export"})
    c.append({"op": "connect", "id": "render", "input": "camera", "source": "camera_os"})
    _create(c, "generate", "Generate", dict(provider="reproject", seed=SEED, text=BRIEF,
                                             output_path="generated.%04d.exr"))
    c.append({"op": "connect", "id": "generate", "input": "image", "source": "undistort"})
    _create(c, "fit", "Reformat", dict(reformat_type="to_box", width=width, height=height, resize_type="none", center=1,
                                       filter="nearest", preserve_bbox=1), name="Place the overscan in the plate format")
    _create(c, "grade", "Grade", name="Grade the generated picture")
    c.append({"op": "connect", "id": "grade", "input": "image", "source": "fit"})
    _create(c, "merge", "Merge", name="Merge over the undistorted plate")
    c.append({"op": "connect", "id": "merge", "input": "A", "source": "grade"})
    c.append({"op": "connect", "id": "merge", "input": "B", "source": "undistort"})
    _create(c, "redistort", "Redistort", name="Redistort (back onto the plate)")
    c.append({"op": "connect", "id": "redistort", "input": "image", "source": "merge"})
    c.append({"op": "connect", "id": "redistort", "input": "camera", "source": "camera"})
    _create(c, "write", "Write", dict(path="shot.%04d.exr", file_type="exr", bit_depth="half"))
    c.append({"op": "connect", "id": "write", "input": "image", "source": "redistort"})
    c.append({"op": "time", "first": 1, "last": frames, "current": 1})
    return c


def build_lens_graph(size, keys, lens, margins, survey="", frames=FRAMES, solve_params=None):
    from nodebased.core import Dispatcher
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": build_lens_graph_commands(size, keys, lens, margins, survey, frames, solve_params)})
    return dispatcher.document


def save_lens_graph(path, size, keys, lens, margins, survey="", frames=FRAMES, plate_folder=None, solve_params=None):
    """Write the lens shot graph; its Read node points at `plate_folder` (stored relative to `path`)."""
    from nodebased.core import atomic_save
    document = build_lens_graph(size, keys, lens, margins, survey, frames, solve_params)
    document["nodes"]["plate"]["params"]["path"] = plate_pattern(plate_folder or Path(path).resolve().parent / "plate_distorted")
    atomic_save(path, document)
