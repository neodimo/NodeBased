#!/usr/bin/env python3
"""Time the Roto matte on the processor against the graphics card: `roto.rasterise` (switch off) against `rotogpu.rasterise`.

Each case is a set of shapes at 1080p or 4K. The two drawers alternate, three passes of REPS drawings each; the table is the
median draw time of each pass and the range over the passes, the card's time split into planning (flatten and edge set-up on the
processor), and the rest (upload, the passes, the read back). The two mattes are compared too, so a speed-up that moved a pixel
shows in `max_diff`. Run it on a quiet machine: `pgrep -af "integ-auto|unittest"` must find nothing.

    python tools/benchmark_roto_gpu.py --output matte-gpu.json
    python tools/benchmark_roto_gpu.py --sizes 1080p --cases shots30 --passes 3 --reps 9
    python tools/benchmark_roto_gpu.py --force-adapter integrated    # or cpu (llvmpipe), as the tests do
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nodebased import gpu3d, roto, rotogpu, shapes  # noqa: E402

SIZES = {"1080p": (1920, 1080), "4k": (3840, 2160)}


def _point(x, y, handle=0.0, angle=0.0):
    hx, hy = handle * math.cos(angle), handle * math.sin(angle)
    return {"x": x, "y": y, "in_x": -hx, "in_y": -hy, "out_x": hx, "out_y": hy}


def _blob(cx, cy, radius, count):
    points = []
    for i in range(count):
        a = 2 * math.pi * i / count
        r = radius * (1 + 0.15 * math.sin(5 * a) + 0.07 * math.cos(11 * a))
        points.append(_point(cx + r * math.cos(a), cy + r * math.sin(a) * 0.8, handle=2 * math.pi * r / count / 3.0, angle=a + math.pi / 2))
    return points


def _shape(points, feather, **extra):
    return {"name": "shape", "mode": "union", "opacity": 1.0, "feather": feather, "closed": True, "feather_model": "box", "points": points, **extra}


def scene(count, scale, points, feather, radius=95):
    """`count` shapes spread over the frame in rows, each a `points`-point curve with a box feather."""
    columns = max(1, int(math.ceil(math.sqrt(count * 16 / 9))))
    rows = int(math.ceil(count / columns))
    out = []
    for index in range(count):
        cx = (index % columns + 0.5) * 1920 * scale / columns
        cy = (index // columns + 0.5) * 1080 * scale / rows
        out.append(_shape(_blob(cx, cy, radius * scale, points), feather * scale, opacity=0.6 + 0.4 * (index % 2),
                          color=[1.0, 0.4 + 0.02 * (index % 20), 0.2, 1.0],
                          feather_falloff=("linear", "smooth", "gaussian")[index % 3]))
    return out


CASES = {
    "shots30": ("30 shapes, 12-point curves, feather 12 (box)", lambda s: scene(30, s, 12, 12.0)),
    "shots30_heavy": ("30 shapes, 50-point curves, feather 18 (box), larger", lambda s: scene(30, s, 50, 18.0, radius=130)),
    "one_blob": ("1 shape, 50-point curve, feather 18 (box)", lambda s: scene(1, s, 50, 18.0, radius=380)),
    "shots10": ("10 shapes, 12-point curves, feather 12 (box)", lambda s: scene(10, s, 12, 12.0)),
}


def write_heavy_comp(path, count=30):
    """The playback fixture with its one Roto shape replaced by `count` animated 12-point curves (each point moves 40 pixels
    over frames 1 to 48), feathered 12 (box): the Roto-heavy shot for `benchmark_2d_playback.py --comp PATH`."""
    document = json.loads((ROOT / "tests/fixtures/playback_comp.nbcomp").read_text())
    shape_list = scene(count, 1.0, 12, 12.0)
    for shape in shape_list:
        shape["points"] = [{**p, "x": {"value": p["x"], "curve": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": p["x"] - 20.0}, {"frame": 48, "value": p["x"] + 20.0}]}}} for p in shape["points"]]
    document["node_data"]["roto"] = {"shapes": shape_list}
    pathlib.Path(path).write_text(json.dumps(document))


def resolve(shape_list):
    return shapes.resolve_shapes({"shapes": shape_list}, 1, None)


def timed(call, reps):
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", choices=sorted(SIZES), default=["1080p"])
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=list(CASES))
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--force-adapter", choices=("default", "integrated", "cpu"), default="default")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--write-comp", type=pathlib.Path, help="write the 30-shape playback comp here and stop")
    args = parser.parse_args()
    if args.write_comp:
        write_heavy_comp(args.write_comp)
        return
    if args.force_adapter != "default":
        original = gpu3d._state
        gpu3d._state = lambda choice=None: original(args.force_adapter if (choice or "default") == "default" else choice)
    adapter = gpu3d._state()["info"].get("device")
    rotogpu.set_enabled(False)
    os.environ[rotogpu.ENV] = "0"
    rows = []
    for size in args.sizes:
        width, height = SIZES[size]
        scale = width / 1920.0
        for case in args.cases:
            title, build = CASES[case]
            resolved = resolve(build(scale))
            cpu_matte = roto.rasterise(resolved, width, height)
            gpu_matte = rotogpu.rasterise(resolved, width, height)        # also the warm-up: compiles the pipelines
            difference = np.abs(cpu_matte - gpu_matte)
            cpu_runs, gpu_runs, plan_runs = [], [], []
            for _ in range(args.passes):
                cpu_runs.append(timed(lambda: roto.rasterise(resolved, width, height), args.reps))
                gpu_runs.append(timed(lambda: rotogpu.rasterise(resolved, width, height), args.reps))
                plan_runs.append(timed(lambda: rotogpu.plan(resolved, width, height), args.reps))
            row = {"size": size, "case": case, "title": title, "shapes": len(resolved), "adapter": adapter,
                   "cpu_ms": [round(v, 2) for v in cpu_runs], "gpu_ms": [round(v, 2) for v in gpu_runs],
                   "gpu_plan_ms": [round(v, 2) for v in plan_runs],
                   "faster_by": round(statistics.median(cpu_runs) / statistics.median(gpu_runs), 2),
                   "max_diff": float(difference.max()), "mean_diff": float(difference.mean())}
            rows.append(row)
            print(f"{size} {title}: processor {min(cpu_runs):.1f} to {max(cpu_runs):.1f} ms, card {min(gpu_runs):.1f} to "
                  f"{max(gpu_runs):.1f} ms (planning {min(plan_runs):.1f} to {max(plan_runs):.1f}), {row['faster_by']}x, "
                  f"max diff {row['max_diff']:.2e}", flush=True)
    record = {"adapter": adapter, "reps": args.reps, "passes": args.passes, "rows": rows}
    if args.output:
        args.output.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
