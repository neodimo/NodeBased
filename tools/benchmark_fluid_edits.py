"""Measure Hot pour re-simulation after common artist edits (Lane 6 step A)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_hot_pour import build_document, force_adapter


EDITS = {
    "pour_rate": (("spout", "src_vel_y", 1.25),),
    "cup_position": (("glass_wall", "tx", 0.05), ("glass_base", "tx", 0.05)),
    "steam_vorticity": (("steam", "vorticity", 0.4),),
}
STAGES = ("liquid_cache", "surface", "whitewater", "steam_cache")


def measure(adapter, cells, frames, workdir):
    from nodebased import gpu3d, simcache
    from nodebased.imaging import Evaluator

    args = SimpleNamespace(liquid_cells=cells, smoke_cells=cells, frames=frames,
                           liquid_pressure="resident", smoke_pressure="resident_sparse", spout_height=0.62)
    results = {"adapter_request": adapter, "adapter": gpu3d._state()["info"].get("device"),
               "grid_cells": cells, "frames": frames, "edits": {}}
    for edit, changes in EDITS.items():
        case_dir = workdir / edit
        case_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NODEBASED_SIM_CACHE_MB"] = "400000"
        os.environ["XDG_CACHE_HOME"] = str(case_dir / "xdg")
        store = simcache.SimCache(root=case_dir / "simcache", memory_budget=512 << 20,
                                  disk_budget=1024 << 20)
        evaluator = Evaluator(sim=store)
        dispatcher = build_document(args)
        doc = dispatcher.document
        for node, param, delta in changes:
            doc["nodes"][node]["params"][param] += delta
        record = {"first_frame_s": None, "stages_s": {}}
        started = time.perf_counter()
        elapsed, _ = _timed(evaluator.evaluate_raster, doc, target="steam_cache", frame=1, typed=True)
        record["first_frame_s"] = round(elapsed, 4)
        record["first_output"] = "steam cache frame 1; display presentation is not instrumented"
        record["stages_s"]["liquid_cache"] = elapsed
        for stage in STAGES:
            stage_times = []
            for frame in range(1, frames + 1):
                dt, _ = _timed(evaluator.evaluate_raster, doc, target=stage, frame=frame, typed=True)
                stage_times.append(dt)
            record["stages_s"][stage] = round(sum(stage_times), 3)
            record.setdefault("stage_frame_ms", {})[stage] = {
                "mean": round(statistics.fmean(stage_times) * 1000, 2),
                "p95": round(sorted(stage_times)[max(0, int(0.95 * len(stage_times)) - 1)] * 1000, 2),
            }
        record["all_stages_s"] = round(time.perf_counter() - started, 2)
        record["cache_stores"] = [cache.stats() for cache in evaluator._sim_stores.values()]
        results["edits"][edit] = record
        # The per-edit cache is reproducible from the committed graph and seed; retain the
        # measurement, not hundreds of MB of regenerated checkpoints.
        import shutil
        shutil.rmtree(case_dir, ignore_errors=True)
    return results


def _timed(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return time.perf_counter() - started, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("default", "integrated", "cpu"), default="default")
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    force_adapter(args.adapter)
    result = measure(args.adapter, args.cells, args.frames, Path(args.workdir))
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    for edit, data in result["edits"].items():
        print(f"{edit}: first={data['first_frame_s']:.3f}s bake={data['all_stages_s']:.2f}s")


if __name__ == "__main__":
    main()
