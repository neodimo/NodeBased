"""Fluids 18 B: the fire shot's curl read with a ruler that does not depend on the grid's cell size.

Bakes the shot at a given grid (`--cells` over the 2 m box, `--set key=value` for any other solver setting) and reads frames 8, 16 and 24:
the image-plane curl share of Fluids 18 A (30 per second over 2 cm image cells; `flame_metrics.projected_curl`), and the world-space
vorticity share on the grid's own cells and after a box blur of 6.25 cm (`fluid_diagnostics.curl_share`), plus peak speeds.

    python curl_ruler.py --cells 64 --pressure auto --out result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FRAMES = (8, 16, 24)


def measure(cells, pressure="auto", overrides=None, scratch=None, frames=FRAMES, upto=None):
    from nodebased import fluid_diagnostics as diag, simcache
    from nodebased.imaging import Evaluator
    import flame_metrics
    from make_shot import build_document
    document = build_document(240, 135, division_size=2.0 / cells, pt_samples=8, **({"pressure": pressure} | (overrides or {})))
    root = Path(scratch or tempfile.mkdtemp(prefix="curl-ruler-", dir="/var/home/omid/.openclaw/workspace/scratch/nb-lanes/run/big-L6"))
    root.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(sim=simcache.SimCache(root=root, memory_budget=1 << 30, disk_budget=16 << 30))
    trace = []
    from nodebased import fluid3d, fluid_gpu_solver
    for cls in (fluid3d.Smoke3D, fluid_gpu_solver.GpuSmoke3D):
        original = cls.step

        def traced(self, state, frame=0, substep=0, seed=0, _original=original):
            started = time.perf_counter()
            result = _original(self, state, frame, substep, seed)
            trace.append((int(frame), int(result.meta.get("cg_iterations", 0)), float(result.meta.get("cg_residual", 0.0)), time.perf_counter() - started))
            return result
        cls.step = traced
    started = time.perf_counter()
    evaluator.evaluate_raster(document, "cache", frame=upto or max(frames), typed=True)
    out = {"cells": cells, "pressure": pressure, "overrides": overrides or {}, "bake_seconds": round(time.perf_counter() - started, 2)}
    out["solver"] = type(evaluator).__name__
    out["pressure_trace"] = {"iterations_max": max((t[1] for t in trace), default=0), "iterations_mean": round(float(np.mean([t[1] for t in trace])), 1) if trace else 0,
                             "residual_max": max((t[2] for t in trace), default=0.0), "steps": len(trace),
                             "step_seconds_mean": round(float(np.mean([t[3] for t in trace])), 4) if trace else 0}
    per_frame = {}
    for entry in trace:
        per_frame[entry[0]] = per_frame.get(entry[0], 0) + 1
    out["substeps_by_frame"] = [per_frame[f] for f in sorted(per_frame)]       # Fluids 18 C: what the automatic rule chose, frame by frame
    camera = document["nodes"]["camera"]["params"]
    cam, target = (camera["tx"], camera["ty"], camera["tz"]), (0.0, camera["target_y"], 0.0)
    rows = {}
    for frame in frames:
        volume = evaluator.evaluate_raster(document, "cache", frame=frame, typed=True)
        density = np.asarray(volume.density if volume.density is not None else volume.to_sparse().to_dense()["density"], np.float64)
        velocity = np.asarray(volume.velocity, np.float64)
        voxel = float(volume.voxel_size)
        row = {"grid": list(density.shape), "voxel": voxel}
        row["projected_2cm"] = flame_metrics.projected_curl(density, velocity, np.asarray(volume.origin, np.float64), voxel, cam, target)
        row["world_native"] = diag.curl_share(density, velocity, voxel)
        row["world_blur_6cm"] = diag.curl_share(density, velocity, voxel, blur=0.0625)
        live = density >= 0.05 * density.max()
        speed = np.sqrt((velocity ** 2).sum(-1))
        row["speed_max"] = round(float(speed.max()), 3)
        row["speed_smoke_mean"] = round(float(speed[live].mean()), 3)
        omega = np.sqrt((diag.world_vorticity(velocity, voxel) ** 2).sum(-1))
        row["vorticity_max"] = round(float(omega.max()), 2)
        row["vorticity_smoke_mean"] = round(float(omega[live].mean()), 2)
        rows[str(frame)] = row
        if frame == 16:
            np.savez_compressed(root / "frame16.npz", density=density.astype(np.float32), velocity=velocity.astype(np.float32), voxel=voxel,
                                origin=np.asarray(volume.origin, np.float64))
    out["frames"] = rows
    return out, root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--pressure", default="auto")
    parser.add_argument("--set", nargs="*", default=[], help="key=value solver overrides (JSON values)")
    parser.add_argument("--keep", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    results = []
    for cells in args.cells:
        scratch = None if args.keep is None else Path(args.keep) / f"c{cells}"
        result, root = measure(cells, args.pressure, overrides, scratch)
        if args.keep is None:
            shutil.rmtree(root, ignore_errors=True)
        results.append(result)
        short = {f: {k: result["frames"][f][k] for k in ("projected_2cm", "world_native", "world_blur_6cm")} for f in ("16",)}
        print(json.dumps({"cells": cells, "bake": result["bake_seconds"], "f16": short}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
