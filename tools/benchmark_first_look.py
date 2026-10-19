"""Time to the first displayed, denoised 1920 x 1080 image of the viewport's progressive Render mode (lane 4, step H1).

    flock /tmp/nb-gpu.lock python tools/benchmark_first_look.py [--adapter default|discrete|integrated|cpu]
        [--scene furnished|mixed|both] [--warm N] [--json out.json]

The clock starts when `progressiverender.step_interactive` is called for a fresh view and stops when the 8-bit image the
viewport would paint exists (`display_rgba8`), so it covers the scene build, the trace, the guides, the denoiser, the
enlargement and the sRGB conversion. Cold is the first call in a fresh process (pipeline compilation and the first scene
upload included); warm repeats it for a camera moved a little each time (a new key, the same device objects), and reports
the median and worst of N. "furnished" is `benchmark_pathtrace.furnished_scene`; "mixed" is that room plus a 40,000-splat
shell and a smoke puff, the difficult case (smoke and splats take the slower GPU variant). The pre-H1 baseline is the same
view through `pathtrace.render(output="denoise")` at 1920 x 1080 and 4 samples (`--baseline`).

Run it under the exclusive GPU lock: the timings are meaningless next to another job on the card.
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from benchmark_pathtrace import CAMERA, furnished_scene, soft_scene
from nodebased import gpu3d, pathtrace, progressiverender as P, scene3d as s

WIDTH, HEIGHT = 1920, 1080
BACKGROUND = (0.02, 0.02, 0.03, 1.0)
SEED = 1


def moved(i):
    """The benchmark camera nudged by `i` steps: a new progressive key each time, the same scene."""
    position = CAMERA.transform.position
    return replace(CAMERA, transform=replace(CAMERA.transform, position=s.Vec3(position.x + 0.01 * i, position.y, position.z)))


def first_look(scene, camera, key):
    started = time.perf_counter()
    state = P.step_interactive(None, scene, camera, WIDTH, HEIGHT, BACKGROUND, 0.05, key)
    computed = time.perf_counter()
    shown = P.display_rgba8(state.image)
    done = time.perf_counter()
    return state, computed - started, done - started, shown


def baseline(scene, camera, samples=4):
    started = time.perf_counter()
    image = pathtrace.render(scene, camera, WIDTH, HEIGHT, BACKGROUND, 0.05, "denoise",
                             pathtrace.PathSettings(samples=samples, max_bounces=P.PREVIEW_BOUNCES, seed=SEED), backend="auto")
    return time.perf_counter() - started, image


def measure(name, scene, warm, with_baseline):
    out = {"scene": name, "output": [WIDTH, HEIGHT]}
    state, computed, shown_at, shown = first_look(scene, CAMERA, ("cold", name))
    out["cold_seconds"] = shown_at
    out["cold_phases"] = {k: round(v, 4) for k, v in state.phases}
    out["traced"] = [state.traced_width, state.traced_height]
    out["samples"] = state.samples
    times, phases, sizes = [], [], []
    for i in range(1, warm + 1):
        state, computed, shown_at, shown = first_look(scene, moved(i), ("warm", name, i))
        times.append(shown_at)
        sizes.append([state.traced_width, state.traced_height])
        phases.append({k: v for k, v in state.phases})
    out["warm_seconds"] = {"median": statistics.median(times), "worst": max(times), "best": min(times), "runs": len(times)}
    out["warm_traced"] = sizes
    out["warm_phases_median"] = {k: round(statistics.median(p[k] for p in phases), 4) for k in phases[0]}
    out["target_seconds"] = P.FIRST_LOOK_SECONDS
    out["target_met_warm"] = out["warm_seconds"]["worst"] <= P.FIRST_LOOK_SECONDS
    out["target_met_cold"] = out["cold_seconds"] <= P.FIRST_LOOK_SECONDS
    out["nonempty"] = bool(shown[..., :3].std() > 0)
    state2 = P.step_interactive(state, scene, moved(warm), WIDTH, HEIGHT, BACKGROUND, 0.05, state.key)
    out["stage_two_ran"] = state2.preview
    out["stage_two_seconds"] = state2.seconds
    out["stage_two_traced"] = [state2.traced_width, state2.traced_height]
    if with_baseline:
        baseline(scene, CAMERA, 1)       # cold-ish: scene and pipelines exist already, so this is the warm figure below
        seconds, _ = baseline(scene, moved(99))
        out["baseline_denoise_1080p_4spp_seconds"] = seconds
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--scene", default="both", choices=("furnished", "mixed", "both"))
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.adapter:
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    print(gpu3d.adapter_report())
    results = []
    for name in (("furnished", "mixed") if args.scene == "both" else (args.scene,)):
        scene = furnished_scene() if name == "furnished" else soft_scene()
        result = measure(name, scene, args.warm, args.baseline)
        results.append(result)
        print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps({"adapter": gpu3d.adapter_report(), "results": results}, indent=1))


if __name__ == "__main__":
    main()
