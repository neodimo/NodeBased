#!/usr/bin/env python3
"""Summarize `tools/benchmark_2d_playback.py` result files into the spike's gate statistics.

For each run (the full result JSON, or its `--timeline-output` file) this reports, over the first
pass through the fresh frames only (later repeats can hit the display cache and are not playback
of fresh frames): the frames counted, total duration, mean interval and p50/p95/p99 of the
intervals between accepted pictures, the first-picture latency, and the node-evaluation stage
(mean, median, p95). Runs are grouped by their `case` name minus the trailing pass number.

    python tools/summarize_2d_playback.py runs/cpu-48-rtx3080ti-*.timeline.json --output summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import statistics
import sys

import numpy as np


def load(path):
    path = pathlib.Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def pct(values, q):
    return round(float(np.percentile(values, q)), 3) if len(values) else None


def stats(values):
    return {"n": len(values), "mean": round(float(np.mean(values)), 3) if len(values) else None,
            "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99),
            "max": round(float(max(values)), 3) if len(values) else None}


def first_pass(frames, expected):
    """The fresh frames of the first traversal, ordered by when their picture was accepted."""
    rows = sorted((f for f in frames if "shown" in f["points_ms"]), key=lambda f: f["points_ms"]["shown"])
    return rows[:expected] if expected else rows


def summarize_run(data):
    timeline = data["fresh_frame_timeline"]
    expected = data.get("first_pass_frames") or None
    rows = first_pass(timeline["frames"], expected)
    shown = [r["points_ms"]["shown"] for r in rows]
    intervals = [b - a for a, b in zip(shown, shown[1:])]
    evaluation = [r["stages_ms"].get("node evaluation") for r in rows if r["stages_ms"].get("node evaluation") is not None]
    conversion = [r["stages_ms"].get("display conversion") for r in rows if r["stages_ms"].get("display conversion") is not None]
    latency = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    duration = (shown[-1] - rows[0]["points_ms"]["request"]) if rows else None
    out = {"case": data.get("case"), "adapter": data.get("adapter"), "backend": data.get("backend", "cpu"),
           "frames_in_first_pass": data.get("first_pass_frames"), "fresh_frames_traced": len(rows),
           "first_pass_fps": data.get("first_pass_fps"), "total_duration_ms": round(duration, 3) if duration else None,
           "mean_interval_ms": round(float(np.mean(intervals)), 3) if intervals else None,
           "interval_ms": stats(intervals), "first_picture_latency_ms": round(latency[0], 3) if latency else None,
           "latency_ms": stats(latency), "node_evaluation_ms": stats(evaluation), "display_conversion_ms": stats(conversion),
           "frames_dropped_counted": data.get("frames_dropped_counted")}
    if "gpu2d" in data and data["gpu2d"]:
        islands = data["gpu2d"].get("islands", [])
        out["gpu2d"] = {"cold_shader_setup_ms": data["gpu2d"].get("cold_shader_setup_ms"),
                        "island_wall_ms": stats([i["wall_ms"] for i in islands]) if islands else None,
                        "island_gpu_ms": stats([i["gpu_ms"] for i in islands]) if islands else None,
                        "island_upload_ms": stats([i["upload_ms"] for i in islands]) if islands else None,
                        "island_readback_ms": stats([i["readback_ms"] for i in islands]) if islands else None,
                        "runs": len(islands), "downloads_per_run": sorted({i["downloads"] for i in islands}),
                        "intermediate_readbacks": sum(i["intermediate_readbacks"] for i in islands),
                        "cpu_boundaries": sorted({b for i in islands for b in i["cpu_boundaries"]})}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    runs = [summarize_run(load(path)) for path in args.runs]
    groups = {}
    for run in runs:
        groups.setdefault(re.sub(r"-\d+$", "", run["case"] or ""), []).append(run)
    summary = {"runs": runs, "groups": {}}
    for name, items in groups.items():
        fps = [r["first_pass_fps"] for r in items if r["first_pass_fps"]]
        summary["groups"][name] = {"passes": len(items), "first_pass_fps": fps,
                                   "first_pass_fps_median": round(statistics.median(fps), 3) if fps else None,
                                   "interval_p95_ms": [r["interval_ms"]["p95"] for r in items],
                                   "node_evaluation_median_ms": [r["node_evaluation_ms"]["p50"] for r in items]}
    text = json.dumps(summary, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(json.dumps(summary["groups"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
