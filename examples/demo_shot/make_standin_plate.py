"""A stand-in for a filmed plate: the demo set through a distorting lens and a slow handheld shake.

There is no filmed footage yet. The demo plate (make_plate.py) is a perfect pinhole on a smooth dolly, the best
case for any camera solve. This plate keeps the set and the dolly and adds the two things a real camera brings:

* **Lens distortion.** Brown-Conrady, in the convention of `nodebased/lensmodel.py` on main (the Camera3D
  lens_* knobs of Rendering 13 step L1): film-back units about the centre, 1.0 = half the format height, y up,
  `LENS` is the "wide angle" preset (k1 -0.04, k2 0.004, about 10% at the corners of a 16:9 frame). The set is
  rendered as an ideal pinhole with overscan (a wider view at the same focal length in pixels) and every plate
  pixel samples the ideal image where the lens maps it from (Newton inverse of the forward model, bilinear).
* **Handheld shake.** Sums of slow sines with seeded phases on the camera position (about 1 cm), the look-at
  target (about 0.4 degrees of pan and tilt) and the roll (about 0.3 degrees), 0.3 to 1.6 cycles a second at
  24 frames a second.

Writes plate.0001.exr .. (beauty only, ACEScg like the demo plate) and `truth.json`: the camera keys of every
frame, the focal length in pixels, the lens and the survey points' tracks on the distorted plate. Deterministic:
the same arguments give the same pixels.

    python examples/demo_shot/make_standin_plate.py --out DIR [--size 960x540] [--frames 24] [--no-lens] [--no-shake]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

sys.path.insert(0, str(shot.HERE.parents[1]))

LENS = {"k1": -0.04, "k2": 0.004, "k3": 0.0, "p1": 0.0, "p2": 0.0}   # lensmodel.PRESETS["wide angle"]
OVERSCAN = 1.3
SHAKE_SEED = 9
SHAKE = {"position": 0.01, "target_degrees": 0.4, "roll_degrees": 0.3}   # amplitudes
FPS = 24.0


# --- the lens, as nodebased/lensmodel.py (principal point at the centre, no squeeze) -------------------------

def distort(x, y, lens=LENS):
    r2 = x * x + y * y
    radial = 1.0 + r2 * (lens["k1"] + r2 * (lens["k2"] + r2 * lens["k3"]))
    xd = x * radial + 2.0 * lens["p1"] * x * y + lens["p2"] * (r2 + 2.0 * x * x)
    yd = y * radial + lens["p1"] * (r2 + 2.0 * y * y) + 2.0 * lens["p2"] * x * y
    return xd, yd


def undistort(xd, yd, lens=LENS, steps=24):
    """Distorted film-back coordinates to ideal ones: Newton's method from the distorted point with the
    model's analytic Jacobian, as lensmodel.undistort does."""
    x, y = np.array(xd, np.float64), np.array(yd, np.float64)
    for _ in range(steps):
        fx, fy = distort(x, y, lens)
        fx, fy = fx - xd, fy - yd
        r2 = x * x + y * y
        radial = 1.0 + r2 * (lens["k1"] + r2 * (lens["k2"] + r2 * lens["k3"]))
        dradial = 2.0 * (lens["k1"] + r2 * (2.0 * lens["k2"] + r2 * 3.0 * lens["k3"]))
        a = radial + x * x * dradial + 2.0 * lens["p1"] * y + 6.0 * lens["p2"] * x
        b = x * y * dradial + 2.0 * lens["p1"] * x + 2.0 * lens["p2"] * y
        d = radial + y * y * dradial + 6.0 * lens["p1"] * y + 2.0 * lens["p2"] * x
        det = a * d - b * b
        x, y = x - (d * fx - b * fy) / det, y - (a * fy - b * fx) / det
    return x, y


def pixels_to_units(px, py, width, height):
    half = height / 2.0
    return (np.asarray(px, np.float64) - width / 2.0) / half, (height / 2.0 - np.asarray(py, np.float64)) / half


def units_to_pixels(x, y, width, height):
    half = height / 2.0
    return np.asarray(x, np.float64) * half + width / 2.0, height / 2.0 - np.asarray(y, np.float64) * half


def distort_pixels(px, py, width, height, lens=LENS):
    """Where a pinhole pixel position lands on the distorted plate."""
    return units_to_pixels(*distort(*pixels_to_units(px, py, width, height), lens), width, height)


# --- the shake --------------------------------------------------------------------------------------------

def _wobble(rng, frames, amplitude):
    t = np.arange(frames) / FPS
    out = np.zeros(frames)
    for frequency in (0.3, 0.7, 1.6):
        out += np.sin(2 * math.pi * frequency * t + rng.uniform(0, 2 * math.pi)) / frequency   # slower sways are wider
    return amplitude * out / np.abs(out).max()


def camera_keys(frames=shot.FRAMES, shake=True):
    """The camera keys of every frame: the demo dolly plus the shake. {frame: {tx .. target_z, roll}}."""
    rng = np.random.default_rng(SHAKE_SEED)
    wobble = {name: _wobble(rng, frames, 1.0) for name in ("tx", "ty", "tz", "pan", "tilt", "roll")}
    keys = {}
    for i, frame in enumerate(range(1, frames + 1)):
        base = shot.camera_at(frame)
        row = dict(base, roll=0.0)
        if shake:
            for axis in ("tx", "ty", "tz"):
                row[axis] += SHAKE["position"] * wobble[axis][i]
            eye = np.array([row["tx"], row["ty"], row["tz"]])
            target = np.array([base["target_x"], base["target_y"], base["target_z"]])
            distance = np.linalg.norm(target - eye)
            forward = (target - eye) / distance
            right = np.cross(forward, (0.0, 1.0, 0.0))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            pan, tilt = (math.radians(SHAKE["target_degrees"]) * wobble[name][i] for name in ("pan", "tilt"))
            target = target + distance * (math.tan(pan) * right + math.tan(tilt) * up)
            row.update(target_x=float(target[0]), target_y=float(target[1]), target_z=float(target[2]),
                       roll=float(SHAKE["roll_degrees"] * wobble["roll"][i]))
        keys[frame] = {name: float(value) for name, value in row.items()}
    return keys


# --- the plate --------------------------------------------------------------------------------------------

def _bilinear(image, x, y):
    """Sample `image` (H, W, C) at continuous pixel-centre coordinates (centres at half integers)."""
    h, w = image.shape[:2]
    fx, fy = np.clip(x - 0.5, 0, w - 1.000001), np.clip(y - 0.5, 0, h - 1.000001)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ax, ay = (fx - x0)[..., None], (fy - y0)[..., None]
    top = image[y0, x0] * (1 - ax) + image[y0, x0 + 1] * ax
    bottom = image[y0 + 1, x0] * (1 - ax) + image[y0 + 1, x0 + 1] * ax
    return top * (1 - ay) + bottom * ay


def focal_pixels(height):
    from nodebased import filmback
    return height * shot.FOCAL / filmback.DEFAULT_VAPERTURE


def keyed_camera(row, focal=shot.FOCAL):
    from nodebased import filmback, scene3d
    return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(row["tx"], row["ty"], row["tz"])),
                          scene3d.Vec3(row["target_x"], row["target_y"], row["target_z"]),
                          filmback.fov_from_aperture(focal, filmback.DEFAULT_VAPERTURE), roll=row.get("roll", 0.0))


def visible(depth_map, camera, world):
    """Which world points the render shows (not hidden behind something): the depth pass at the point's pixel
    agrees with the point's own depth within 3%."""
    from nodebased import scene3d
    height, width = depth_map.shape[:2]
    pixels, depth = scene3d.project(camera, width, height, world)
    px, py = np.floor(pixels[:, 0]).astype(int), np.floor(pixels[:, 1]).astype(int)
    inside = (depth > 0.5) & (px >= 0) & (py >= 0) & (px < width) & (py < height)
    shown = np.zeros(len(world), bool)
    shown[inside] = np.abs(depth_map[py[inside], px[inside]] - depth[inside]) <= 0.03 * depth[inside]
    return shown


def survey_tracks(keys, width, height, lens=True, shown=None):
    """Every landmark's pixel on every frame where it is in view (`shown[frame]`, from the depth pass) and away
    from the edge, on the plate as filmed (through the lens), to a thousandth of a pixel."""
    from nodebased import scene3d
    world = np.array(shot.landmarks(), np.float64)
    points = [{"name": f"landmark {i}", "world": [float(v) for v in p], "track": []} for i, p in enumerate(world)]
    for frame, row in keys.items():
        pixels, depth = scene3d.project(keyed_camera(row), width, height, world)
        px, py = distort_pixels(pixels[:, 0], pixels[:, 1], width, height) if lens else (pixels[:, 0], pixels[:, 1])
        for i in range(len(world)):
            if shown is not None and not shown[frame][i]:
                continue
            if depth[i] > 0.5 and 4 <= px[i] <= width - 4 and 4 <= py[i] <= height - 4:
                points[i]["track"].append({"frame": frame, "x": round(float(px[i]), 3), "y": round(float(py[i]), 3)})
    return {"points": [p for p in points if p["track"]]}


def render(folder, size=(960, 540), frames=shot.FRAMES, lens=True, shake=True):
    """Write the stand-in plate and truth.json into `folder`; returns the truth dict."""
    from nodebased.core import Dispatcher
    from nodebased.imaging import Evaluator
    from nodebased.media import write_exr
    width, height = size
    scale = OVERSCAN if lens else 1.0
    big = (int(round(width * scale / 2)) * 2, int(round(height * scale / 2)) * 2)
    keys = camera_keys(frames, shake)
    commands = shot.scene_commands(*big)
    for command in commands:
        if command.get("id") == "camera" and command["op"] == "create":
            command["params"]["focal"] = shot.FOCAL * height / big[1]      # the same focal length in pixels
    for param in ("tx", "ty", "tz", "target_x", "target_y", "target_z", "roll"):
        commands.append({"op": "set_curve", "id": "camera", "param": param, "curve": {
            "interpolation": "linear", "keys": [{"frame": f, "value": keys[f][param]} for f in keys]}})
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": commands})
    evaluator, folder = Evaluator(), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    v, u = np.mgrid[0:height, 0:width].astype(np.float64) + 0.5
    if lens:
        ix, iy = units_to_pixels(*undistort(*pixels_to_units(u, v, width, height)), width, height)
        ix, iy = ix + (big[0] - width) / 2.0, iy + (big[1] - height) / 2.0
    from nodebased.media import raster_layer_arrays
    world, shown = np.array(shot.landmarks(), np.float64), {}
    for frame in range(1, frames + 1):
        raster = evaluator.evaluate_raster(dispatcher.document, "render", frame=frame)
        shown[frame] = visible(raster_layer_arrays(raster)["depth"][..., 0],
                               keyed_camera(keys[frame], shot.FOCAL * height / big[1]), world)
        ideal = np.asarray(raster.to_display(), np.float32)
        plate = _bilinear(ideal.astype(np.float64), ix, iy).astype(np.float32) if lens else ideal
        write_exr(folder / f"plate.{frame:04d}.exr", np.ascontiguousarray(plate), bits="float")
    truth = {"size": [width, height], "frames": frames, "focal_px": focal_pixels(height), "focal_mm": shot.FOCAL,
             "lens": LENS if lens else None, "shake": SHAKE if shake else None, "shake_seed": SHAKE_SEED,
             "overscan": scale, "camera_keys": {str(f): row for f, row in keys.items()},
             "survey": survey_tracks(keys, width, height, lens, shown)}
    (folder / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="960x540")
    parser.add_argument("--frames", type=int, default=shot.FRAMES)
    parser.add_argument("--no-lens", action="store_true")
    parser.add_argument("--no-shake", action="store_true")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    truth = render(args.out, size, args.frames, lens=not args.no_lens, shake=not args.no_shake)
    print(f"{args.frames} frames {size[0]}x{size[1]} in {args.out}; focal {truth['focal_px']:.2f} px; "
          f"{len(truth['survey']['points'])} survey points tracked")


if __name__ == "__main__":
    main()
