"""Render the holdout recipe picture (docs/images/holdout_recipe.png) and the plate it uses (examples/holdout_plate.png).

The plate is made here, by code, from a fixed seed: a dusk sky, a tree line, a road and a lamppost, drawn at the screen
position of a stand-in card the CG scene holds (so the stand-in lines up with the real post exactly). It is not a photograph,
needs no licence and may be used by anything in the repository (it is released under the repository's own licence);
step R3 of the rendering plan reuses it. Run it from the repository root:

    python tools/render_holdout_reference.py [--samples 64] [--size 480x270] [--plate-only]

The recipe picture has three panels, left to right: the plate; the CPU path tracer's render of a ball with the stand-in
card set to Holdout (the checkerboard is the picture's empty alpha); and the Merge over the plate, where the real post
passes in front of the ball. Deterministic: the sampling seed is the path tracer's default.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import pathtrace as pt, scene3d as s
from nodebased.imaging import write_png

ROOT = Path(__file__).resolve().parents[1]
PLATE = ROOT / "examples" / "holdout_plate.png"
RECIPE = ROOT / "docs" / "images" / "holdout_recipe.png"

CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0.0, 1.5, 7.0)), s.Vec3(0.0, 0.9, 0.0), 36.0)
KEY = s.Light("Directional", (1.0, 0.85, 0.7), 2.4, s.Vec3(-5, 6, 4), s.Vec3(0, 0.9, 0), shadows=True)
FILL = s.Light("Directional", (0.35, 0.5, 0.9), 0.7, s.Vec3(5, 2, 3), s.Vec3(0, 0.9, 0), shadows=True)
POST_Z = 2.2                                   # the stand-in card for the real post, in front of the ball
POST_X = (0.20, 0.45)
POST_Y = (-1.2, 2.6)


def stand_in(holdout=True):
    """The card that stands in for the lamppost: thin, tall, in front of the ball."""
    (x0, x1), (y0, y1) = POST_X, POST_Y
    return s.Geometry(np.array([[x0, y0, POST_Z], [x1, y0, POST_Z], [x1, y1, POST_Z], [x0, y1, POST_Z]], "f4"),
                      np.array([[0, 1, 2], [0, 2, 3]], "i4"), (0.6, 0.2, 0.2, 1.0), holdout=holdout, name="Post stand-in")


def ball():
    return replace(s._sphere(1.1, 48, (0.9, 0.35, 0.15, 1.0), s.Transform3D(position=s.Vec3(0.0, 0.9, 0.0))),
                   material="pbr", metallic=0.0, pbr_roughness=0.28, pbr_specular=0.9, name="Ball")


def smooth_noise(rng, height, width, cells):
    """Value noise: random values on a coarse grid, bilinearly resampled to the picture."""
    grid = rng.random((cells[0] + 2, cells[1] + 2))
    ys, xs = np.linspace(0, cells[0], height, endpoint=False), np.linspace(0, cells[1], width, endpoint=False)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x0 + 1] * fx
    bottom = grid[y0 + 1][:, x0] * (1 - fx) + grid[y0 + 1][:, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def make_plate(width, height):
    """The plate, linear RGB float32 with alpha 1: sky, tree line, road, lamppost with its lit head."""
    rng = np.random.default_rng(7)
    horizon = int(height * 0.60)
    v = np.linspace(0, 1, height)[:, None, None]
    sky_top, sky_low = np.array([0.03, 0.07, 0.20]), np.array([0.85, 0.42, 0.22])
    pic = np.zeros((height, width, 3))
    sky = sky_top * (1 - v ** 0.7) + sky_low * v ** 0.7
    clouds = smooth_noise(rng, height, width, (4, 9)) * 0.6 + smooth_noise(rng, height, width, (9, 20)) * 0.4
    pic += sky * (0.75 + 0.5 * clouds[..., None]) * (v < horizon / height + 0.001)
    # a tree line: a noisy dark skyline just above the horizon
    skyline = horizon - (smooth_noise(rng, 1, width, (1, 24))[0] * 0.07 + 0.03) * height
    rows = np.arange(height)[:, None]
    trees = rows >= skyline[None, :]
    pic[trees & (rows < horizon)] = np.array([0.010, 0.014, 0.012])
    # the road: dark, brighter toward the horizon haze, with a dashed centre line in perspective
    depth = np.clip((rows - horizon) / max(height - horizon, 1), 0, 1)
    road = (np.array([0.020, 0.020, 0.024]) * (1 - depth[..., None]) + np.array([0.045, 0.045, 0.05]) * depth[..., None]
            + smooth_noise(rng, height, width, (30, 60))[..., None] * 0.015)
    below = np.broadcast_to(rows >= horizon, (height, width))
    pic[below] = road[below]
    cx = width * 0.5
    for row in range(horizon + 1, height):
        t = (row - horizon) / max(height - horizon, 1)
        half = 1.0 + 5.0 * t
        phase = (1.0 / max(t, 1e-3)) % 6.0
        if phase < 3.0:
            pic[row, int(cx - half):int(cx + half) + 1] = (0.45, 0.42, 0.30)
    # haze at the horizon
    haze = np.exp(-(((rows - horizon) / (0.05 * height)) ** 2))
    pic += haze[..., None] * np.array([0.30, 0.15, 0.08])
    # the lamppost, at the stand-in card's position in this camera
    corners = np.array([[POST_X[0], POST_Y[0], POST_Z], [POST_X[1], POST_Y[0], POST_Z],
                        [POST_X[1], POST_Y[1], POST_Z], [POST_X[0], POST_Y[1], POST_Z]], np.float32)
    (xy, _) = s.project(CAMERA, width, height, corners)
    yy, xx = np.mgrid[0:height, 0:width]
    px, py = xx + 0.5, yy + 0.5
    # the post is the quadrilateral between its projected edges; cover = coverage of pixel centres, softened by one pixel
    left = xy[0, 0] + (xy[3, 0] - xy[0, 0]) * np.clip((xy[0, 1] - py) / max(xy[0, 1] - xy[3, 1], 1e-6), 0, 1)
    right = xy[1, 0] + (xy[2, 0] - xy[1, 0]) * np.clip((xy[1, 1] - py) / max(xy[1, 1] - xy[2, 1], 1e-6), 0, 1)
    cover = np.clip(np.minimum(px - left, right - px) + 0.5, 0, 1) * (py >= xy[2, 1]) * (py <= xy[0, 1])
    shade = 0.5 + 0.5 * np.clip((px - left) / np.maximum(right - left, 1e-6), 0, 1)       # a lit side to the right
    post = (np.array([0.012, 0.012, 0.016]) + shade[..., None] * np.array([0.05, 0.035, 0.025]))
    pic = pic * (1 - cover[..., None]) + post * cover[..., None]
    # the lamp: a small warm head at the top of the post, with a glow
    head = np.array([(xy[2, 0] + xy[3, 0]) / 2, xy[2, 1] + 5.0])
    d2 = (px - head[0]) ** 2 + (py - head[1]) ** 2
    pic += np.exp(-d2 / (2 * (0.02 * height) ** 2))[..., None] * np.array([2.2, 1.5, 0.7])
    pic += np.exp(-d2 / (2 * (0.10 * height) ** 2))[..., None] * np.array([0.30, 0.18, 0.07])
    pic += rng.normal(0.0, 0.004, pic.shape)            # film grain
    out = np.ones((height, width, 4), np.float32)
    out[..., :3] = np.clip(pic, 0, None)
    return out


def render_cg(width, height, samples):
    scene = s.Scene((ball(), stand_in(True)), (KEY, FILL))
    settings = pt.PathSettings(samples=samples, max_bounces=3, diffuse_bounces=2, specular_bounces=2)
    return pt.render(scene, CAMERA, width, height, (0.0, 0.0, 0.0, 0.0), 0.12, "rgba", settings, backend="cpu")


def over(foreground, background):
    """Premultiplied over: the Merge node's default."""
    return foreground + background * (1.0 - foreground[..., 3:4])


def checker(width, height):
    yy, xx = np.indices((height, width))
    even = ((xx // 12 + yy // 12) % 2 == 0)
    out = np.ones((height, width, 4), np.float32)
    out[..., :3] = np.where(even, 0.06, 0.12)[..., None]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--size", default="480x270")
    parser.add_argument("--plate-only", action="store_true")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))
    plate = make_plate(width, height)
    PLATE.parent.mkdir(parents=True, exist_ok=True)
    write_png(PLATE, plate)
    print("wrote", PLATE)
    if args.plate_only:
        return
    cg = np.asarray(render_cg(width, height, args.samples))
    merged = over(cg, plate)
    gap = np.zeros((height, 6, 4), np.float32)
    gap[..., 3] = 1
    on_checker = over(cg, checker(width, height))
    sheet = np.concatenate((plate, gap, on_checker, gap, merged), axis=1)
    sheet[..., 3] = 1.0
    RECIPE.parent.mkdir(parents=True, exist_ok=True)
    write_png(RECIPE, sheet.astype(np.float32))
    print("wrote", RECIPE)


if __name__ == "__main__":
    main()
