"""Where the time of the viewport's first look goes, stage by stage (lane 4, plan "Progress", step B of 3).

    flock /tmp/nb-gpu.lock python tools/profile_first_look.py [--scene mixed|furnished] [--warm N] [--adapter ...] [--json out.json]

Each warm trial moves the camera a little (a new progressive key, the same document), then runs the first look
(`progressiverender.step_interactive`) and the display conversion with timers wrapped around the stages that matter: the scene
build per object kind (mesh trees, splat layer, smoke), the pack for the card, the upload, the trace, the three guide passes,
the denoiser, the enlargement and the sRGB conversion. The first run in a fresh process is the cold row. Wrapped functions
nest, so the table lists each stage's own time (children subtracted) and the totals add up to the wall time of the call.
Run under the exclusive GPU lock.
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from benchmark_pathtrace import CAMERA, furnished_scene, soft_scene
from nodebased import gpu3d, gpupathtrace, pathtrace, ptdenoise, ptsplats, ptvolume, progressiverender as P, scene3d as s

WIDTH, HEIGHT = 1920, 1080
BACKGROUND = (0.02, 0.02, 0.03, 1.0)
AMBIENT = 0.05

_STACK = []
_OWN = defaultdict(float)       # stage -> own seconds in the current trial
_CALLS = defaultdict(int)


def timed(module, name, label):
    """Wrap `module.name` so its own time (children that are also wrapped subtracted) lands under `label`."""
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        frame = [0.0]
        _STACK.append(frame)
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            took = time.perf_counter() - started
            _STACK.pop()
            if _STACK:
                _STACK[-1][0] += took
            _OWN[label] += took - frame[0]
            _CALLS[label] += 1
    wrapper.__wrapped__ = original
    setattr(module, name, wrapper)


def install():
    timed(pathtrace, "build_scene", "scene build: instances, lights, bookkeeping")
    timed(pathtrace, "_blas_of", "scene build: mesh triangle tree")
    timed(ptsplats, "build", "scene build: splat layer (arrays and tree)")
    timed(ptvolume, "build", "scene build: smoke grid")
    timed(gpupathtrace, "pack", "pack for the card")
    timed(gpupathtrace, "render", "trace (dispatch, readback, upload)")
    timed(ptdenoise, "denoise_strips", "denoise")
    timed(P, "enlarge", "enlarge")
    timed(P, "display_rgba8", "display conversion")
    timed(pathtrace, "over_background", "background composite")


def moved(i):
    position = CAMERA.transform.position
    return replace(CAMERA, transform=replace(CAMERA.transform, position=s.Vec3(position.x + 0.01 * i, position.y, position.z)))


def one_look(scene, camera, key):
    _OWN.clear()
    _CALLS.clear()
    started = time.perf_counter()
    state = P.step_interactive(None, scene, camera, WIDTH, HEIGHT, BACKGROUND, AMBIENT, key)
    computed = time.perf_counter()
    P.display_rgba8(state.image)
    wall = time.perf_counter() - started
    stages = {k: round(v, 4) for k, v in _OWN.items()}
    accounted = sum(_OWN.values())
    stages["everything else (guide setup, python glue)"] = round(wall - accounted, 4)
    return {"wall_seconds": round(wall, 4), "step_seconds": round(computed - started, 4),
            "phases": {k: round(v, 4) for k, v in state.phases}, "traced": [state.traced_width, state.traced_height],
            "stages": stages, "calls": dict(_CALLS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="mixed", choices=("furnished", "mixed"))
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.adapter:
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    print(gpu3d.adapter_report())
    install()
    scene = furnished_scene() if args.scene == "furnished" else soft_scene()
    P._COST.clear()
    cold = one_look(scene, CAMERA, ("cold", args.scene))
    warm = [one_look(scene, moved(i), ("warm", args.scene, i)) for i in range(1, args.warm + 1)]
    names = sorted({k for r in warm for k in r["stages"]})
    median = {k: round(statistics.median(r["stages"].get(k, 0.0) for r in warm), 4) for k in names}
    out = {"scene": args.scene, "output": [WIDTH, HEIGHT], "adapter": gpu3d.adapter_report().splitlines()[0],
           "cold": cold, "warm_runs": warm,
           "warm_median_wall_seconds": round(statistics.median(r["wall_seconds"] for r in warm), 4),
           "warm_median_stage_seconds": dict(sorted(median.items(), key=lambda kv: -kv[1])),
           "warm_median_phases": {k: round(statistics.median(r["phases"].get(k, 0.0) for r in warm), 4)
                                  for k in ("render", "guides", "denoise", "enlarge")}}
    print("cold wall:", cold["wall_seconds"], "traced", cold["traced"])
    print("warm median wall:", out["warm_median_wall_seconds"], "traced", warm[-1]["traced"])
    for k, v in out["warm_median_stage_seconds"].items():
        print(f"  {v * 1000:8.1f} ms  {k}")
    print("  phases:", out["warm_median_phases"])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
