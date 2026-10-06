"""GPU beauty with transparent meshes mixed with splats: parity and wall times (run with PYTHONPATH=. from the repo
root).

Renders the `rgba` output of a scene that holds splats and transparent meshes on the GPU and on the CPU reference
and prints, per scene: the largest and mean absolute difference over all four channels, how many pixels differ by
more than 2e-3, and the wall times (best of ``--frames`` GPU frames after a warm-up; the CPU renders once). The same
scene with its meshes made opaque is timed too (the existing opaque path), so the cost of the layers is visible.
Scenes:

* ``small``: 20,000 random synthetic splats between and around two transparent cards. Synthetic, so it says nothing
  about a capture.
* ``capture``: the shared 3.4-million-splat capture read from ``$NB_SCENE_PLY`` or ``assets/splats/scene.ply``
  (read-only), with a translucent cube dropped into the street and a pane of glass across the left of the picture,
  from the camera of `tools/release_media_clips.py`. The CPU reference is slow on it (minutes at 160x90), so the
  parity size (``--parity-size``) is smaller than the timing sizes (``--sizes``).

    python tools/benchmark_splat_layers.py [--adapter default|integrated|cpu] [--scene small capture]
        [--sizes 640x360 1920x1080] [--parity-size 160x90] [--frames 3] [--cpu-cache DIR] [--capture-splats N]
        [--json out.json]

``--capture-splats N`` keeps every k-th splat of the capture (about N of them) for an adapter whose buffers cannot
hold all 3.4 million (llvmpipe allows 128 MiB); the scene is then reported as ``capture-N``.

``--cpu-cache`` keeps each CPU reference render as an .npy file so another adapter's run of the same scene and size
compares against the same reference instead of rendering it again.
"""
import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpu3d, gpusplat, scene3d as s, splats


def small_scene():
    rng = np.random.default_rng(5)
    count = 20_000
    sh = np.zeros((count, 1, 3))
    sh[:, 0] = (rng.uniform(.1, .9, (count, 3)) - .5) / splats.C0
    cloud = splats.SplatCloud(rng.uniform(-1, 1, (count, 3)) * (2.2, 1.4, 1.0),
                              rng.uniform(.01, .05, (count, 3)) * (1, 1, .15), rng.normal(size=(count, 4)),
                              rng.uniform(.2, .95, count), sh, 0, colorspace="linear")
    cards = (s._card(7.0, 5.0, (.4, .5, .9, .4), s.Transform3D(s.Vec3(0, 0, 1.2), s.Vec3(0, 0, 0))),
             s._card(7.0, 5.0, (.9, .5, .2, .5), s.Transform3D(s.Vec3(0, 0, -.3), s.Vec3(0, 35, 0))))
    return s.Scene(geometries=cards, splats=(s.SplatInstance(cloud),)), s.Camera()


def capture_scene(limit=0):
    path = Path(os.environ.get("NB_SCENE_PLY", "assets/splats/scene.ply"))
    cloud = splats.load_cloud_cached(str(path), "colmap", "srgb")
    if limit and limit < len(cloud):
        keep = np.arange(0, len(cloud), -(-len(cloud) // limit))
        cloud = splats.SplatCloud(cloud.positions[keep], cloud.scales[keep], cloud.rotations[keep], cloud.opacity[keep],
                                  cloud.sh[keep], cloud.sh_degree, colorspace=cloud.colorspace)
    camera = s.Camera(transform=s.Transform3D(s.Vec3(5.6456, 2.1610, 14.2390)),
                      target=s.Vec3(-.3802, -.3498, 6.2046), fov=50, near=3.0, far=5000.0)
    cube = s._cube(1.0, (.85, .12, .1, .5), s.Transform3D(s.Vec3(3.0, .12, 9.0), s.Vec3(0, 32, 0)))
    # A pane of glass across the left of the picture, square to the view direction, 4 units ahead of the camera.
    eye = np.array((5.6456, 2.1610, 14.2390)); aim = np.array((-.3802, -.3498, 6.2046))
    forward = (aim - eye) / np.linalg.norm(aim - eye)
    centre = eye + forward * 4.0 + np.cross(forward, (0, 1, 0)) * -.9
    yaw = float(np.degrees(np.arctan2(-forward[0], -forward[2])))
    pane = s._card(1.6, 1.8, (.5, .8, .9, .3), s.Transform3D(s.Vec3(*centre), s.Vec3(0, yaw, 0)))
    return s.Scene(geometries=(cube, pane), splats=(s.SplatInstance(cloud),)), camera


SCENES = {"small": lambda limit: small_scene(), "capture": capture_scene}


def opaque(scene):
    return replace(scene, geometries=tuple(replace(g, color=(*g.color[:3], 1.0)) for g in scene.geometries))


def best_of(function, frames):
    function()
    times = []
    for _ in range(frames):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def cpu_reference(name, scene, camera, size, cache):
    """The CPU render and the seconds it took (None when it came from the cache)."""
    width, height = size
    path = Path(cache) / f"{name}-layers-{width}x{height}.npy" if cache else None
    if path is not None and path.exists():
        return np.load(path), None
    start = time.perf_counter()
    cpu = np.asarray(s.render(scene, camera, width, height))
    elapsed = time.perf_counter() - start
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, cpu)
    return cpu, elapsed


def parity(name, scene, camera, size, cache):
    width, height = size
    try:
        gpu = np.asarray(gpu3d.render(scene, camera, width, height))
    except (ValueError, gpu3d.Unsupported) as exc:      # an adapter's list or memory cap: the CPU is the fallback
        return dict(size=f"{width}x{height}", refused=str(exc)[:200])
    cpu, cpu_s = cpu_reference(name, scene, camera, size, cache)
    difference = np.abs(gpu - cpu)
    worst = difference.max(axis=2)
    return dict(size=f"{width}x{height}", cpu_alpha_mean=round(float(cpu[..., 3].mean()), 4),
                max_abs_diff=float(difference.max()), mean_abs_diff=float(difference.mean()),
                pixels_over_2e3=int((worst > 2e-3).sum()), pixels=int(worst.size),
                cpu_s=None if cpu_s is None else round(cpu_s, 3))


def timings(scene, camera, size, frames):
    width, height = size
    row = dict(size=f"{width}x{height}")
    for label, value in (("layered", scene), ("opaque_meshes", opaque(scene))):
        try:
            gpusplat.last_timings.pop("beauty_passes", None)
            row[f"{label}_s"] = round(best_of(lambda: gpu3d.render(value, camera, width, height), frames), 4)
            if label == "layered":
                row["passes"] = gpusplat.last_timings.get("beauty_passes")
                row["tile_entries"] = gpusplat.last_timings.get("data_tile_entries")
                row["longest_tile_list"] = gpusplat.last_timings.get("data_longest_tile_list")
        except (ValueError, gpu3d.Unsupported) as exc:
            row[f"{label}_refused"] = str(exc)[:200]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="default", choices=("default", "integrated", "cpu"))
    parser.add_argument("--scene", nargs="+", default=["small"], choices=sorted(SCENES))
    parser.add_argument("--sizes", nargs="+", default=["640x360"])
    parser.add_argument("--parity-size", default="160x90")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--cpu-cache")
    parser.add_argument("--capture-splats", type=int, default=0)
    parser.add_argument("--json")
    args = parser.parse_args()
    original = gpu3d._state
    gpu3d._state = lambda choice=None: original(args.adapter if (choice or "default") == "default" else choice)
    size = lambda text: tuple(int(v) for v in text.split("x"))
    report = dict(adapter=gpu3d._state()["info"].get("device"), scenes={})
    for name in args.scene:
        scene, camera = SCENES[name](args.capture_splats)
        if name == "capture" and args.capture_splats:
            name = f"capture-{args.capture_splats}"
        entry = report["scenes"][name] = dict(splats=sum(len(i.cloud) for i in scene.splats),
                                              meshes=len(scene.geometries))
        entry["parity"] = parity(name, scene, camera, size(args.parity_size), args.cpu_cache)
        entry["timings"] = [timings(scene, camera, size(text), args.frames) for text in args.sizes]
        print(json.dumps(dict(adapter=report["adapter"], scene=name, **entry), indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
