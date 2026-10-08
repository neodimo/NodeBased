"""Where the GPU path tracer's smoke time goes: phase times and dispatch-shape experiments, one JSON line per variant.

    python tools/profile_smoke_phases.py --adapter discrete --majorant grid [--variants base,band4,pass4] [--size 640x360]

Run it under the shared GPU lock (`flock /tmp/nb-gpu.lock ...`). Each variant is a short bounded render of the sparse smoke
plume of `tools/benchmark_volume_majorant.py` (fixed 32 samples, seed 4) at one dispatch shape: how many rows one submission
covers (`GPU_PATHS_PER_SUBMISSION`, `SOFT_SLOWDOWN`) and how many samples one pass takes (`pass_samples`). The renderer's own
phase laps (scene build, pack, upload, dispatch, final readback, postprocess) are reported with the number of submissions and
waits, so the time of the dispatch phase can be divided by the number of host round trips it contains. `fixed` is the
dispatch shape before step Q2 (bands of ~65 000 paths, a wait after every band); `grow` is the shipped one.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

# variant -> (SOFT_SLOWDOWN, GPU_PATHS_PER_SUBMISSION, pass_samples, BAND_TARGET_SECONDS)
VARIANTS = {
    "fixed": (8, 1 << 19, 0, 0.0),      # the shape before step Q2: 1 sample per pass, ~65 000 paths per band, a wait per band
    "grow": (8, 1 << 19, 0, 0.008),     # the shipped shape: the band grows to about 8 ms per dispatch
    "grow4": (8, 1 << 19, 0, 0.004),
    "grow16": (8, 1 << 19, 0, 0.016),
    "band2": (4, 1 << 19, 0, 0.0),      # fixed bands of twice the paths
    "band4": (2, 1 << 19, 0, 0.0),      # four times the paths: one band over the whole 640 by 360 image
    "pass4": (8, 1 << 19, 4, 0.0),      # four samples per pass, same bands
}


def render(size, mode, samples, pass_samples, aim="plume", shadow_steps=None, bounces=4, ambient=.7):
    from nodebased import gpupathtrace, pathtrace as pt, ptvolume
    from tools.benchmark_volume_majorant import tenth_plume
    ptvolume.ENABLE_SKIP = gpupathtrace.ENABLE_VOLUME_SKIP = mode != "box"
    ptvolume.MAJORANT_RATIO = 4
    scene, camera, volume, _ = tenth_plume(64)
    if aim == "away":        # the same pipeline and scene with every camera ray leaving the box: the shader's fixed cost per path
        from nodebased import scene3d
        camera = scene3d.Camera(camera.transform, scene3d.Vec3(0, 0, 16), 30)
    if shadow_steps is not None:       # an experiment: how much of the dispatch is the shadow rays' fixed-step march
        import dataclasses
        volume = dataclasses.replace(volume, shadow_steps=shadow_steps)
    settings = pt.PathSettings(sampling="fixed", samples=samples, max_bounces=bounces, seed=4, pass_samples=pass_samples)
    stats = {}
    started = time.perf_counter()
    image = pt.render(scene, camera, size[0], size[1], (0, 0, 0, 0), ambient, "rgba", settings, stats=stats, backend="gpu", volume=volume)
    return image, stats, time.perf_counter() - started


def gpupathtrace_module():
    from nodebased import gpupathtrace
    return gpupathtrace


def run_variant(name, size, mode, samples, timed, aim="plume", shadow_steps=None, bounces=4, ambient=.7):
    from nodebased import gpupathtrace
    slowdown, per_submission, pass_samples, target = VARIANTS[name]
    gpupathtrace.SOFT_SLOWDOWN, gpupathtrace.GPU_PATHS_PER_SUBMISSION = slowdown, per_submission
    gpupathtrace.BAND_TARGET_SECONDS = target
    render(size, mode, samples, pass_samples, aim, shadow_steps, bounces, ambient)      # warm-up (pipeline compile)
    times, last = [], {}
    for _ in range(timed):
        image, stats, seconds = render(size, mode, samples, pass_samples, aim, shadow_steps, bounces, ambient)
        times.append(seconds)
        last = stats
    per_pass = max(1, pass_samples or 1)
    rows = max(1, (per_submission // slowdown) // max(size[0] * per_pass, 1))
    submissions = last.get("readbacks", {}).get("waits", 0)
    return dict(variant=name, mode=mode, aim=aim, wg=getattr(gpupathtrace_module(), "WG_SIZE", None), shadow_steps=shadow_steps, bounces=bounces, ambient=ambient, size=list(size), samples=samples, soft_slowdown=slowdown, band_target_s=target,
                paths_per_submission=per_submission // slowdown, pass_samples=per_pass, rows_per_band=rows,
                submissions=submissions, waits=last.get("readbacks", {}).get("waits"),
                median_ms=round(statistics.median(times) * 1000, 2), times_ms=[round(t * 1000, 2) for t in times],
                phases_ms={k: round(v * 1000, 2) for k, v in last.get("phases", {}).items()},
                ms_per_wait=round(last.get("phases", {}).get("dispatch", 0) * 1000 / max(submissions, 1), 3),
                mean=float(np.asarray(image)[..., :3].mean())), image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="discrete", choices=("default", "discrete", "integrated", "cpu"))
    parser.add_argument("--majorant", default="grid", choices=("grid", "box"))
    parser.add_argument("--variants", default="base")
    parser.add_argument("--aim", default="plume", choices=("plume", "away"), help="away: every camera ray misses the box")
    parser.add_argument("--shadow-steps", type=int, default=None, help="override the volume settings' shadow steps (an experiment)")
    parser.add_argument("--wg", type=int, default=None, help="override the work-group edge (an experiment)")
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--ambient", type=float, default=.7)
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--timed", type=int, default=3)
    parser.add_argument("--save-image", help="write the last variant's image here as .npy")
    args = parser.parse_args()
    from tools.benchmark_portable_render import force_adapter
    force_adapter(args.adapter)
    if args.wg:
        from nodebased import gpupathtrace
        gpupathtrace._wg_size = lambda splats, volumes, links=False: args.wg
    size = tuple(int(v) for v in args.size.split("x"))
    image = None
    for name in args.variants.split(","):
        result, image = run_variant(name, size, args.majorant, args.samples, args.timed, args.aim, args.shadow_steps, args.bounces, args.ambient)
        print(json.dumps(result), flush=True)
    if args.save_image and image is not None:
        np.save(args.save_image, np.asarray(image, np.float32))


if __name__ == "__main__":
    sys.exit(main())
