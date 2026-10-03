"""Instanced-copy shadow benchmark for the interactive viewport (run with PYTHONPATH=. from the repo root).

Times `viewportgpu.ViewportRenderer.render` at 1920 x 1080 on 100,000 `Instance3D` copies of a 990-triangle
sphere under four shadow-casting Directional lights, in the cases `docs/BENCHMARKS-v0.34-instances.md` reports:

* ``outside``: the copies scatter over a +-200 box while the lights frame a small floor, so almost every copy
  falls outside every light's view (the case 0.33.0 already held at 30 fps).
* ``in_view``: a floor wide enough that all four lights frame the whole scatter, so every copy is inside every
  light's view at once (the case 0.33.0 listed as a known limit). The tool prints how many copies each light's
  frustum kept, so the case proves it is the one it claims to be.
* ``unshadowed``: the same ``in_view`` scene with the lights' shadows off, the floor under the shadow cost.
* ``in_view_wide`` and ``unshadowed_wide``: the same two scenes seen by a camera that frames the whole scatter, so
  the camera pass draws every copy too. That pass alone is over the 30 fps budget at 1080p with or without shadows
  (the viewport's camera pass draws every copy at full detail), so these two rows are reported for context and are
  not the budget case.

Wall time is the best of three batches of ten frames after a warm-up frame, including the read-back, the same
protocol as `tests/test_3d_viewport_shadows.py`.

    python tools/benchmark_instances.py [--adapter default|integrated|cpu] [--cull gpu|cpu] [--json out.json]
        [--width 1920] [--height 1080] [--count 100000] [--batches 3] [--frames 10]
        [--case outside in_view unshadowed in_view_wide unshadowed_wide]
"""
import argparse
import json
import math
import os
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpu3d, scene3d as s, viewportgpu

BACKGROUND = (0.02, 0.02, 0.03, 1.0)
FRAME_BUDGET = 1.0 / 30


def floor(size):
    return replace(s._card(size, size, (0.8, 0.8, 0.8, 1.0), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0))),
                   material="pbr", metallic=0.0, pbr_roughness=0.9, pbr_specular=0.5)


def four_lights(shadows=True):
    return tuple(s.Light(kind="Directional", intensity=1.0, shadows=shadows,
                         position=s.Vec3(6 * math.cos(a), 6, 6 * math.sin(a)), target=s.Vec3(0, 0, 0))
                 for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2))


def scatter(count, spread, seed=0):
    mesh = s._sphere(1, 55, (0.8, 0.2, 0.2, 1.0), s.Transform3D())
    rng = np.random.RandomState(seed)
    matrices = np.tile(np.eye(4), (count, 1, 1))
    matrices[:, :3, 3] = rng.uniform(-spread, spread, (count, 3))
    return s.InstanceSet((mesh,), matrices, np.zeros(count, np.int32), node_key="inst")


def scenes(count):
    """name -> (scene, camera): the three cases above."""
    small = s.Camera(s.Transform3D(s.Vec3(0, 4, 9)), s.Vec3(0, 0, 0), 45.0, 0.1, 100.0)
    wide = s.Camera(s.Transform3D(s.Vec3(0, 120, 420)), s.Vec3(0, 0, 0), 55.0, 1.0, 3000.0)
    lit = s.Scene((floor(440),), four_lights(), instances=(scatter(count, 150.0),))
    dark = s.Scene((floor(440),), four_lights(False), instances=(scatter(count, 150.0),))
    return {
        "outside": (s.Scene((floor(8),), four_lights(), instances=(scatter(count, 200.0),)), small),
        "in_view": (lit, small),
        "unshadowed": (dark, small),
        "in_view_wide": (lit, wide),
        "unshadowed_wide": (dark, wide),
    }


def time_frames(gpu, scene, camera, width, height, batches=3, frames=10):
    gpu.render(scene, camera, width, height, BACKGROUND)  # warm the mesh, instance and shadow caches
    best = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(frames):
            gpu.render(scene, camera, width, height, BACKGROUND)
        best.append((time.perf_counter() - start) / frames)
    return min(best)


def adapter_info():
    info = gpu3d._state()["info"]
    return {"adapter": str(info.get("device", "")), "type": str(info.get("adapter_type", ""))}


def force_adapter(kind):
    if kind == "default":
        return
    original = gpu3d._state
    gpu3d._state = lambda choice=None: original(kind if (choice or "default") == "default" else choice)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--adapter", default="default", choices=("default", "integrated", "cpu"))
    parser.add_argument("--json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cull", default="gpu", choices=("gpu", "cpu"),
                        help="cpu: the 0.33.0 path, copies culled with NumPy per light (kept as the fallback)")
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--batches", type=int, default=3, help="timed batches; the best one is reported")
    parser.add_argument("--frames", type=int, default=10, help="frames per batch")
    parser.add_argument("--case", nargs="+", default=["outside", "in_view", "unshadowed", "in_view_wide", "unshadowed_wide"])
    args = parser.parse_args()
    force_adapter(args.adapter)
    if not gpu3d.available():
        raise SystemExit("no wgpu adapter")
    gpu = viewportgpu.renderer()
    if gpu is None:
        raise SystemExit(viewportgpu.failure())
    gpu._cull_ready = gpu._cull_ready and args.cull == "gpu"
    result = {"adapter": adapter_info(), "cull": args.cull, "size": [args.width, args.height], "copies": args.count, "cases": {}}
    print(f"adapter: {result['adapter']['adapter']} ({result['adapter']['type']}), {args.width}x{args.height}, "
          f"{args.count} copies")
    cases = scenes(args.count)
    for name in args.case:
        scene, camera = cases[name]
        seconds = time_frames(gpu, scene, camera, args.width, args.height, args.batches, args.frames)
        entry = {"ms": round(seconds * 1000, 1), "fps": round(1 / seconds, 1), "within_budget": seconds < FRAME_BUDGET}
        report = getattr(gpu, "shadow_cull_report", None)
        if report is not None and not name.startswith("unshadowed"):
            entry["kept_per_light"] = report()["kept_per_light"]
        result["cases"][name] = entry
        print(f"{name:15s} {entry['ms']:7.1f} ms  {entry['fps']:6.1f} fps  "
              f"{'within' if entry['within_budget'] else 'over'} the 30 fps budget"
              + (f"  kept per light {entry['kept_per_light']}" if "kept_per_light" in entry else ""))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
