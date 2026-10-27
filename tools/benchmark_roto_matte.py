#!/usr/bin/env python3
"""Time the Roto matte drawer alone: `nodebased.roto.rasterise` on resolved shapes, no evaluator, no window.

Each case is a set of shapes at 1080p or 4K, drawn by the fast drawer and by the legacy one (`NODEBASED_ROTO_DRAWER`),
rows interleaved, three passes of REPS drawings each; the table is the median draw time of each pass and the range
over the passes. The two outputs are compared too, so a speed-up that moved a pixel shows in `max_diff`.

    python tools/benchmark_roto_matte.py --output matte.json
    python tools/benchmark_roto_matte.py --sizes 1080p --cases simple feathered50 --passes 3 --reps 15
    python tools/benchmark_roto_matte.py --lock-check --output lock.json

`--lock-check` measures what a drawing costs the other threads of the process: a pure-Python counting loop runs for two seconds
while a second thread draws the 50-point feathered curve over and over, once per drawer, three interleaved rounds; the result is
the loop's speed as a percentage of its speed alone (100 means the drawing never held the interpreter lock) and how many matte
drawings fit in the same two seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nodebased import roto, shapes  # noqa: E402

SIZES = {"1080p": (1920, 1080), "4k": (3840, 2160)}


def _point(x, y, handle=0.0, angle=0.0, feather=0.0):
    hx, hy = handle * math.cos(angle), handle * math.sin(angle)
    return {"x": x, "y": y, "in_x": -hx, "in_y": -hy, "out_x": hx, "out_y": hy,
            "feather_x": feather, "feather_y": 0.0}


def _scalar(value, key=None):
    """A static number, or a number keyed linearly from `value` to `value + key` over frames 1 to 48."""
    if not key:
        return value
    return {"value": value, "curve": {"interpolation": "linear",
                                       "keys": [{"frame": 1, "value": value}, {"frame": 48, "value": value + key}]}}


def _shape(points, *, feather=0.0, closed=True, model="box", move=0.0, **extra):
    shape = {"name": "shape", "mode": "union", "opacity": 1.0, "feather": feather, "closed": closed,
             "feather_model": model, "points": [], **extra}
    for point in points:
        moved = dict(point)
        moved["x"] = _scalar(point["x"], move)
        shape["points"].append(moved)
    return shape


def polygon(sx):
    cx, cy = 700 * sx, 540 * sx
    corners = [(500, 300), (850, 300), (900, 540), (850, 780), (500, 780), (450, 540)]
    return [_point(x * sx, y * sx) for x, y in corners]


def blob(sx, count=50, feather=0.0):
    """A wobbly closed curve with `count` points and smooth handles."""
    cx, cy = 960 * sx, 540 * sx
    points = []
    for i in range(count):
        a = 2 * math.pi * i / count
        r = (380 + 60 * math.sin(5 * a) + 25 * math.cos(11 * a)) * sx
        points.append(_point(cx + r * math.cos(a), cy + r * math.sin(a) * 0.8,
                             handle=2 * math.pi * r / count / 3.0, angle=a + math.pi / 2, feather=feather))
    return points


def squiggle(sx, count=10):
    points = []
    for i in range(count):
        points.append(_point((300 + 150 * i) * sx, (540 + 200 * math.sin(i * 0.9)) * sx, handle=60 * sx,
                             angle=0.6 * math.cos(i * 0.9)))
    return points


def cases(sx):
    """name -> (payload shapes, resolve params, frame)."""
    blur = {"motion_blur": True, "motion_blur_samples": 4, "motion_blur_shutter": 0.5}
    return {
        "simple": ([_shape(polygon(sx))], {}),
        "simple_feathered": ([_shape(polygon(sx), feather=18 * sx)], {}),
        "feathered50": ([_shape(blob(sx), feather=18 * sx)], {}),
        "feathered50_motion": ([_shape(blob(sx), feather=18 * sx, move=40 * sx, **blur)], {}),
        "simple_motion": ([_shape(polygon(sx), feather=18 * sx, move=40 * sx, **blur)], {}),
        "edge50": ([_shape(blob(sx, feather=6 * sx), feather=18 * sx, model="edge")], {}),
        "stroke10": ([_shape(squiggle(sx), feather=8 * sx, closed=False, stroke_width=30 * sx)], {}),
        "two_shapes": ([_shape(polygon(sx), feather=10 * sx), _shape(blob(sx), feather=4 * sx, mode="subtract")], {}),
    }


def resolved(case, frame=10):
    payload, params = case
    return shapes.resolve_shapes({"shapes": payload}, frame, params)


def timed(call, reps):
    samples = []
    for _ in range(reps):
        begin = time.perf_counter()
        call()
        samples.append((time.perf_counter() - begin) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def run(args):
    out = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__, "cpus": os.cpu_count(),
                            "reps": args.reps, "passes": args.passes}, "rows": []}
    for size in args.sizes:
        w, h = SIZES[size]
        sx = w / 1920.0
        table = cases(sx)
        for name in args.cases or table:
            shape_list = resolved(table[name])
            row = {"size": size, "case": name}
            images = {}
            for drawer in roto.DRAWERS:
                roto.set_drawer(drawer)
                images[drawer] = roto.rasterise(shape_list, w, h)          # warm
            times = {drawer: [] for drawer in roto.DRAWERS}
            for _ in range(args.passes):
                for drawer in roto.DRAWERS:
                    roto.set_drawer(drawer)
                    times[drawer].append(round(timed(lambda: roto.rasterise(shape_list, w, h), args.reps), 2))
            row["ms"] = times
            row["max_diff"] = float(np.abs(images["fast"] - images["legacy"]).max())
            row["covered_fraction"] = round(float((images["fast"][..., 3] > 0).mean()), 4)
            out["rows"].append(row)
            print(f"{size:6} {name:20} legacy {min(times['legacy']):7.1f} to {max(times['legacy']):7.1f}   "
                  f"fast {min(times['fast']):7.1f} to {max(times['fast']):7.1f}   max diff {row['max_diff']:.2e}",
                  flush=True)
    roto.set_drawer("fast")
    return out


def lock_check(args):
    import threading

    shape_list = resolved(cases(1.0)["feathered50"])

    def spin(seconds):
        end, count = time.perf_counter() + seconds, 0
        while time.perf_counter() < end:
            for _ in range(1000):
                count += 1
        return count

    def beside(drawer, seconds=2.0):
        roto.set_drawer(drawer)
        roto.rasterise(shape_list, 1920, 1080)
        stop, drawings = threading.Event(), [0]

        def work():
            while not stop.is_set():
                roto.rasterise(shape_list, 1920, 1080)
                drawings[0] += 1
        thread = threading.Thread(target=work)
        thread.start()
        time.sleep(0.2)
        count = spin(seconds)
        done = drawings[0]
        stop.set()
        thread.join()
        return count, done
    solo = max(spin(2.0) for _ in range(2))
    rows = {drawer: [] for drawer in roto.DRAWERS}
    for _ in range(args.passes):
        for drawer in roto.DRAWERS:
            count, done = beside(drawer)
            rows[drawer].append({"loop_percent_of_solo": round(100.0 * count / solo, 1), "drawings_in_two_seconds": done})
    roto.set_drawer("fast")
    result = {"case": "50-point curve, feather 18 (box), 1080p", "solo_loop_iterations_in_two_seconds": solo, "rows": rows}
    for drawer, items in rows.items():
        print(f"{drawer:7} loop at {min(i['loop_percent_of_solo'] for i in items):.1f} to {max(i['loop_percent_of_solo'] for i in items):.1f} percent of its solo speed; "
              f"{min(i['drawings_in_two_seconds'] for i in items)} to {max(i['drawings_in_two_seconds'] for i in items)} drawings in two seconds")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", nargs="+", choices=sorted(SIZES), default=["1080p", "4k"])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--output")
    parser.add_argument("--lock-check", action="store_true", help="what a drawing costs a Python loop on another thread")
    args = parser.parse_args()
    result = lock_check(args) if args.lock_check else run(args)
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
