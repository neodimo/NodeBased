"""GPU splat data-pass parity and wall times (run with PYTHONPATH=. from the repo root).

Renders the `depth`, `position` and `object_id` outputs of a scene that holds splats and an opaque mesh on the
GPU and on the CPU reference, and prints, per output: how many pixels each marks as covered, how many of those
differ in coverage, the largest numeric difference, and the wall times (best of ``--frames`` GPU frames after a
warm-up; the CPU renders once). Scenes:

* ``small``: 20,000 random synthetic splats and a card behind them. Synthetic, so it says nothing about a capture.
* ``capture``: the shared 3.4-million-splat capture read from ``$NB_SCENE_PLY`` or ``assets/splats/scene.ply``
  (read-only), with a red cube dropped into the street in front of some of it, from the camera of
  `tools/release_media_clips.py`. The CPU reference refuses it above its work budget, so the parity size
  (``--parity-size``) is smaller than the timing sizes (``--sizes``).

    python tools/benchmark_splat_data_passes.py [--adapter default|integrated|cpu] [--scene small capture]
        [--sizes 640x360 1920x1080] [--parity-size 320x180] [--frames 3] [--cpu-cache DIR] [--capture-splats N]
        [--json out.json]

``--capture-splats N`` keeps every k-th splat of the capture (about N of them) for an adapter whose buffers cannot
hold all 3.4 million (llvmpipe allows 128 MiB); the scene is then reported as ``capture-N``.

``--cpu-cache`` keeps each CPU reference render (minutes each on the capture) as an .npy file so another adapter's
run of the same scene and size compares against the same reference instead of rendering it again.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpu3d, gpusplat, scene3d as s, splats

OUTPUTS = ("depth", "position", "object_id")


def small_scene():
    rng = np.random.default_rng(5)
    count = 20_000
    sh = np.zeros((count, 1, 3))
    sh[:, 0] = (rng.uniform(.1, .9, (count, 3)) - .5) / splats.C0
    cloud = splats.SplatCloud(rng.uniform(-1, 1, (count, 3)) * (2.2, 1.4, 1.0),
                              rng.uniform(.01, .05, (count, 3)) * (1, 1, .15), rng.normal(size=(count, 4)),
                              rng.uniform(.2, .95, count), sh, 0, colorspace="linear")
    card = s._card(7.0, 5.0, (.4, .5, .6, 1), s.Transform3D(s.Vec3(0, 0, -1.2), s.Vec3(0, 0, 0)))
    return s.Scene(geometries=(card,), splats=(s.SplatInstance(cloud),)), s.Camera()


def capture_scene(limit=0):
    path = Path(os.environ.get("NB_SCENE_PLY", "assets/splats/scene.ply"))
    cloud = splats.load_cloud_cached(str(path), "colmap", "srgb")
    if limit and limit < len(cloud):
        # Every k-th splat of the real capture (a thinner cloud of the same scene, for adapters whose buffers
        # cannot hold all of it).
        keep = np.arange(0, len(cloud), -(-len(cloud) // limit))
        cloud = splats.SplatCloud(cloud.positions[keep], cloud.scales[keep], cloud.rotations[keep], cloud.opacity[keep],
                                  cloud.sh[keep], cloud.sh_degree, colorspace=cloud.colorspace)
    cube = s._cube(1.0, (.85, .12, .1, 1), s.Transform3D(s.Vec3(3.0, .12, 9.0), s.Vec3(0, 32, 0)))
    camera = s.Camera(transform=s.Transform3D(s.Vec3(5.6456, 2.1610, 14.2390)),
                      target=s.Vec3(-.3802, -.3498, 6.2046), fov=50, near=3.0, far=5000.0)
    return s.Scene(geometries=(cube,), splats=(s.SplatInstance(cloud),)), camera


SCENES = {"small": lambda limit: small_scene(), "capture": capture_scene}


def best_of(function, frames):
    function()
    times = []
    for _ in range(frames):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def cpu_reference(name, scene, camera, size, output, cache):
    """The CPU render and the seconds it took (None when it came from the cache)."""
    width, height = size
    path = Path(cache) / f"{name}-{width}x{height}-{output}.npy" if cache else None
    if path is not None and path.exists():
        return np.load(path), None
    start = time.perf_counter()
    cpu = np.asarray(s.render(scene, camera, width, height, output=output))
    elapsed = time.perf_counter() - start
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, cpu)
    return cpu, elapsed


def parity(name, scene, camera, size, cache):
    width, height = size
    rows = []
    for output in OUTPUTS:
        cpu, cpu_s = cpu_reference(name, scene, camera, size, output, cache)
        gpu = np.asarray(gpu3d.render(scene, camera, width, height, output=output))
        covered = cpu[..., 3] > 0
        difference = np.abs(gpu - cpu)[covered].max(axis=1) if covered.any() else np.zeros(0)
        rows.append(dict(output=output, size=f"{width}x{height}", cpu_covered=int(covered.sum()),
                         gpu_covered=int((gpu[..., 3] > 0).sum()),
                         coverage_differs=int(((gpu[..., 3] > 0) != covered).sum()),
                         pixels_differ_over_1e3=int((difference > 1e-3).sum()),
                         max_abs_diff=float(difference.max()) if len(difference) else 0.0,
                         ids_differ=int((gpu[..., 0] != cpu[..., 0]).sum()) if output == "object_id" else None,
                         cpu_s=None if cpu_s is None else round(cpu_s, 3)))
    return rows


def timings(scene, camera, size, frames):
    width, height = size
    rows = []
    for output in OUTPUTS:
        try:
            wall = best_of(lambda: gpu3d.render(scene, camera, width, height, output=output), frames)
            rows.append(dict(output=output, size=f"{width}x{height}", gpu_s=round(wall, 4),
                             tile_entries=gpusplat.last_timings.get("data_tile_entries"),
                             bin_ms=round(gpusplat.last_timings.get("data_bin_ms", 0), 1),
                             resolve_ms=round(gpusplat.last_timings.get("data_resolve_ms", 0), 1)))
        except (ValueError, gpu3d.Unsupported) as exc:
            rows.append(dict(output=output, size=f"{width}x{height}", refused=str(exc)[:200]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="default", choices=("default", "integrated", "cpu"))
    parser.add_argument("--scene", nargs="+", default=["small"], choices=sorted(SCENES))
    parser.add_argument("--sizes", nargs="+", default=["640x360"])
    parser.add_argument("--parity-size", default="320x180")
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
        entry["timings"] = [row for text in args.sizes for row in timings(scene, camera, size(text), args.frames)]
        print(json.dumps(dict(adapter=report["adapter"], scene=name, **entry), indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
