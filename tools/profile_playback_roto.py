#!/usr/bin/env python3
"""Profile the fast Roto stages on the animated playback-fixture shape.

The five frame rows make the cost split reproducible without a window or an
evaluator.  They call the drawer's coverage, box feather, curve-span and
Bezier helpers directly, then time the whole matte as a useful cross-check.

    PYTHONPATH=. .venv/bin/python tools/profile_playback_roto.py \
        --output benchmarks/2d-playback/roto-fixture-profile-1108.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodebased import roto, rotofast, shapes  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "playback_comp.nbcomp"


def median_ms(call, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    document = json.loads(FIXTURE.read_text())
    params = document["nodes"]["roto"]["params"]
    width, height = int(params["width"]), int(params["height"])
    payload = document["node_data"]["roto"]
    rows = []
    for frame in args.frames:
        shape = shapes.resolve_shapes(payload, frame, params)[0]
        points = shape["points"]
        bounds = roto._shape_bounds(shape)
        x0, y0 = int(bounds[0] // 1), int(bounds[1] // 1)
        x1, y1 = int(-(-bounds[2] // 1)), int(-(-bounds[3] // 1))
        patch_width, patch_height = x1 - x0, y1 - y0
        vertices = roto.flatten_points(points)
        curves, segments, _ = rotofast._spans(points, closed=True)
        sample_t = __import__("numpy").linspace(0.0, 1.0, max(1, int(segments.sum())))
        cover = lambda: rotofast.coverage(vertices, patch_width, patch_height, (x0, y0))
        timings = {
            "coverage": median_ms(cover, args.repeats),
            "box_feather": median_ms(lambda: roto.feather(cover(), shape["feather"]), args.repeats),
            "edge_feather": median_ms(lambda: rotofast.edge_feather(points, vertices, cover(),
                                         shape["feather"], "linear", (x0, y0)), args.repeats),
            "spans": median_ms(lambda: rotofast._spans(points, closed=True), args.repeats),
            "bezier": median_ms(lambda: rotofast._bezier(curves, sample_t), args.repeats),
            "motion_samples": median_ms(lambda: sum((cover() for _ in shape.get("_motion_samples", ())),
                                                      start=__import__("numpy").zeros((patch_height, patch_width),
                                                                                        dtype=__import__("numpy").float32)), args.repeats),
            "matte": median_ms(lambda: roto.rasterise([shape], width, height), args.repeats),
        }
        rows.append({"frame": frame, "patch": [patch_width, patch_height],
                     "motion_samples": len(shape.get("_motion_samples", ())), "ms": timings})
        print(f"frame {frame}: " + ", ".join(f"{key} {value:.3f} ms" for key, value in timings.items()))
    means = {key: statistics.mean(row["ms"][key] for row in rows) for key in rows[0]["ms"]}
    result = {"fixture": str(FIXTURE.relative_to(ROOT)), "frames": args.frames,
              "repeats": args.repeats, "rows": rows, "mean_ms": means}
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("mean: " + ", ".join(f"{key} {value:.3f} ms" for key, value in means.items()))


if __name__ == "__main__":
    main()
