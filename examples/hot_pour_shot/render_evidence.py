"""Bake the 32-cell shot, compare CPU/GPU path tracing, and write the three showcase frames."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nodebased import fluid3d, gpu3d, gpupathtrace, simcache
from nodebased.core import load_document
from nodebased.imaging import Evaluator, write_png

from make_shot import ROOT


RUN = Path("/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/big-L6/hot-pour-shot")
FRAMES = (1, 60, 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-width", type=int, default=96)
    parser.add_argument("--adapter", choices=("default", "integrated", "cpu"), default="default")
    parser.add_argument("--small-only", action="store_true", help="compare path tracers without writing 960 x 540 stills")
    args = parser.parse_args()
    original_state = gpu3d._state
    if args.adapter != "default":
        gpu3d._state = lambda choice=None: original_state(args.adapter if (choice or "default") == "default" else choice)
    if args.adapter == "cpu" and not gpupathtrace.soft_supported(gpu3d._state()):
        print("SKIP: llvmpipe refuses path tracing this liquid/smoke scene (soft_supported=False)", flush=True)
        return
    state = gpu3d._state()
    adapter_name = (getattr(state, "adapter", None) and state.adapter.info.get("device")) or gpu3d.describe()
    print(json.dumps({"adapter": args.adapter, "device": str(adapter_name)}), flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    doc = load_document(ROOT / "hot_pour_shot.nbcomp")
    doc["nodes"]["render"]["params"].update(render_backend="gpu", render_mode="pathtrace", pt_samples=2,
                                               max_bounces=2, diffuse_bounces=1, specular_bounces=1)
    doc["nodes"]["render"]["params"].update(width=args.small_width, height=round(args.small_width * 9 / 16))
    doc["nodes"]["plate_fit"]["params"].update(width=args.small_width, height=round(args.small_width * 9 / 16))
    fluid3d._SOLVERS.clear()
    evaluator = Evaluator(sim=simcache.SimCache(root=RUN, memory_budget=1 << 30, disk_budget=8 << 30))
    started = time.perf_counter()
    for frame in range(1, 121):
        evaluator.evaluate_raster(doc, "liquid_cache", frame=frame, typed=True)
        evaluator.evaluate_raster(doc, "steam_cache", frame=frame, typed=True)
        if frame in FRAMES:
            print(json.dumps({"baked_through": frame, "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    bake_seconds = time.perf_counter() - started

    gpu_small = {}
    try:
        for frame in FRAMES:
            gpu_small[frame] = evaluator.evaluate_raster(doc, "comp", frame=frame).pixels.copy()
            write_png(RUN / f"gpu-small-{frame:03d}.png", gpu_small[frame])
    except Exception as exc:
        if args.adapter == "cpu":
            print(f"SKIP: llvmpipe refused GPU path tracing the liquid/smoke scene: {type(exc).__name__}: {exc}", flush=True)
            return
        raise
    cpu_small = {}
    doc["nodes"]["render"]["params"]["render_backend"] = "cpu"
    for frame in FRAMES:
        cpu_small[frame] = evaluator.evaluate_raster(doc, "comp", frame=frame).pixels.copy()
        write_png(RUN / f"cpu-small-{frame:03d}.png", cpu_small[frame])
    comparisons = {}
    for frame in FRAMES:
        got, want = gpu_small[frame], cpu_small[frame]
        mean_error = float(np.abs(got - want).mean())
        reference = float(np.abs(want).mean())
        comparisons[str(frame)] = {"mean_absolute_error": mean_error, "limit": .02 * reference + .005,
                                   "passed": mean_error <= .02 * reference + .005}
    if args.small_only:
        result = {"adapter": args.adapter, "device": str(adapter_name), "bake_seconds": round(bake_seconds, 2),
                  "comparison": comparisons}
        (RUN / f"evidence-{args.adapter}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not all(row["passed"] for row in comparisons.values()):
            raise SystemExit("CPU/GPU path-tracer comparison failed")
        return
    doc["nodes"]["render"]["params"].update(render_backend="gpu", width=960, height=540, pt_samples=8,
                                               max_bounces=4, diffuse_bounces=2, specular_bounces=2)
    doc["nodes"]["plate_fit"]["params"].update(width=960, height=540)
    output_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "hot_pour_shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_seconds = {}
    for frame in FRAMES:
        before = time.perf_counter()
        shot = evaluator.evaluate_raster(doc, "comp", frame=frame).pixels
        write_png(output_dir / f"hot-pour-{frame:03d}.png", shot)
        render_seconds[str(frame)] = round(time.perf_counter() - before, 2)
        print(json.dumps({"rendered": frame, "elapsed_seconds": render_seconds[str(frame)]}), flush=True)
    result = {"adapter": args.adapter, "device": str(adapter_name), "cells": 32, "frames": 120,
              "bake_seconds": round(bake_seconds, 2),
              "render_seconds": render_seconds, "comparison": comparisons}
    (RUN / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    benchmark = Path(__file__).resolve().parents[2] / "benchmarks" / "hot_pour" / "hot-pour-shot-2026-10-14.json"
    benchmark.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if bake_seconds >= 1200 or not all(row["passed"] for row in comparisons.values()):
        raise SystemExit("bake exceeded 20 minutes or a CPU/GPU comparison failed")


if __name__ == "__main__":
    main()
