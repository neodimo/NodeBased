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
BOUNDED_DISK_MB = {"ParticleCache3D": 150, "FluidWhitewater3D": 100, "FluidCache3D": 600}


def _profile_factories():
    """Instrument optional per-solver phase hooks without changing solver output."""
    from nodebased import flip_gpu_resident, fluid_gpu_solver, simcache

    solvers = []
    counters = {"cache_write_s": 0.0}
    smoke = fluid_gpu_solver.GpuSmoke3D
    liquid = flip_gpu_resident.create_solver
    write_disk = simcache.SimCache._write_disk_locked

    def profiled_smoke(*args, **kwargs):
        kwargs["profile"] = True
        solver = smoke(*args, **kwargs)
        solvers.append(("steam", solver))
        return solver

    def profiled_liquid(*args, **kwargs):
        kwargs["profile"] = True
        solver = liquid(*args, **kwargs)
        solvers.append(("liquid", solver))
        return solver

    def profiled_write(store, key, state):
        started = time.perf_counter()
        try:
            return write_disk(store, key, state)
        finally:
            counters["cache_write_s"] += time.perf_counter() - started

    fluid_gpu_solver.GpuSmoke3D = profiled_smoke
    flip_gpu_resident.create_solver = profiled_liquid
    simcache.SimCache._write_disk_locked = profiled_write
    return solvers, counters


def measure(adapter, cells, frames, workdir, profile_phases=False, only_edit=None, cache_mb=None,
            cache_memory_mb=None, whitewater_max_particles=None, smoke_iterations=None,
            cache_precision=None, cache_channels=None):
    from nodebased import gpu3d, simcache
    from nodebased.imaging import Evaluator

    profiled_solvers, profile_counters = _profile_factories() if profile_phases else ([], {})
    args = SimpleNamespace(liquid_cells=cells, smoke_cells=cells, frames=frames,
                           liquid_pressure="resident", smoke_pressure="resident_sparse", spout_height=0.62)
    results = {"adapter_request": adapter, "adapter": gpu3d._state()["info"].get("device"),
               "grid_cells": cells, "frames": frames,
               "cache_disk_limits_mb": BOUNDED_DISK_MB if cache_mb is None else "uniform",
               "cache_memory_mb": cache_memory_mb, "smoke_iterations": smoke_iterations,
               "cache_precision": cache_precision, "cache_channels": cache_channels,
               "whitewater_max_particles": whitewater_max_particles, "edits": {}}
    edits = {only_edit: EDITS[only_edit]} if only_edit else EDITS
    for edit, changes in edits.items():
        profiled_solvers.clear()
        if profile_phases:
            profile_counters["cache_write_s"] = 0.0
        case_dir = workdir / edit
        case_dir.mkdir(parents=True, exist_ok=True)
        # Bound this harness to <2 GB of on-disk cache: one shared simulation store,
        # its reduced-preview store, and a separate bounded surface-mesh store.
        total_disk_limit = sum(BOUNDED_DISK_MB.values()) + 300 + 3 * 256
        if cache_mb is not None:
            total_disk_limit = 3 * cache_mb + 300 + 3 * 256
        if total_disk_limit > 2000:
            raise ValueError("cache caps exceed the 2 GB scratch policy")
        os.environ["NODEBASED_SIM_CACHE_MB"] = str(cache_mb or 600)
        os.environ["XDG_CACHE_HOME"] = str(case_dir / "xdg")
        store = simcache.SimCache(root=case_dir / "simcache", memory_budget=512 << 20,
                                  disk_budget=300 << 20)
        evaluator = Evaluator(sim=store)
        dispatcher = build_document(args)
        doc = dispatcher.document
        for node in doc["nodes"].values():
            if "cache_disk_mb" in node.get("params", {}):
                if cache_memory_mb is not None:
                    node["params"]["cache_memory_mb"] = cache_memory_mb
                node["params"]["cache_disk_mb"] = (cache_mb if cache_mb is not None
                                                     else BOUNDED_DISK_MB.get(node.get("type"), 100))
            if (whitewater_max_particles is not None
                    and node.get("type") == "FluidWhitewater3D"):
                node["params"]["max_particles"] = int(whitewater_max_particles)
            if smoke_iterations is not None and node.get("type") == "FluidSolver3D":
                node["params"]["max_iterations"] = int(smoke_iterations)
            if cache_precision is not None and node.get("type") == "FluidCache3D":
                node["params"]["cache_precision"] = cache_precision
            if cache_channels is not None and node.get("type") == "FluidCache3D":
                node["params"]["cache_channels"] = cache_channels
        baseline_started = time.perf_counter()
        for stage in STAGES:
            for frame in range(1, frames + 1):
                evaluator.evaluate_raster(doc, target=stage, frame=frame, typed=True)
        baseline_elapsed = time.perf_counter() - baseline_started
        profiled_solvers.clear()
        if profile_phases:
            profile_counters["cache_write_s"] = 0.0
        for node, param, delta in changes:
            doc["nodes"][node]["params"][param] += delta
        record = {"base_cache_prep_s": round(baseline_elapsed, 2),
                  "first_frame_s": None, "stages_s": {}}
        started = time.perf_counter()
        elapsed, _ = _timed(evaluator.evaluate_raster, doc, target="steam_cache", frame=1, typed=True)
        record["first_frame_s"] = round(elapsed, 4)
        record["first_output"] = "steam cache frame 1; display presentation is not instrumented"
        record["stages_s"]["liquid_cache"] = elapsed
        for stage in STAGES:
            stage_times = []
            for frame in range(1, frames + 1):
                dt, value = _timed(evaluator.evaluate_raster, doc, target=stage, frame=frame, typed=True)
                stage_times.append(dt)
                if stage == "whitewater" and frame == frames and hasattr(value, "positions"):
                    record["whitewater_particles_last_frame"] = int(len(value.positions))
            record["stages_s"][stage] = round(sum(stage_times), 3)
            record.setdefault("stage_frame_ms", {})[stage] = {
                "mean": round(statistics.fmean(stage_times) * 1000, 2),
                "p95": round(sorted(stage_times)[max(0, int(0.95 * len(stage_times)) - 1)] * 1000, 2),
            }
        record["all_stages_s"] = round(time.perf_counter() - started, 2)
        record["cache_stores"] = [cache.stats() for cache in evaluator._sim_stores.values()]
        if profile_phases:
            record["solver_phases"] = [{"solver": name,
                                         "seconds": dict(getattr(solver, "phase_seconds", {})),
                                         "copies": dict(getattr(solver, "copy_seconds", {}))}
                                        for name, solver in profiled_solvers]
            record["cache_write_s"] = round(profile_counters["cache_write_s"], 3)
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
    parser.add_argument("--profile-phases", action="store_true",
                        help="enable solver phase timers; their synchronization makes wall time slower")
    parser.add_argument("--edit", choices=tuple(EDITS), help="measure only this edit")
    parser.add_argument("--cache-mb", type=int,
                        help="uniform per-store disk cap (default: per-node bounded caps below 2 GB total)")
    parser.add_argument("--cache-memory-mb", type=int,
                        help="override graph per-node memory budgets (default: use Hot pour values)")
    parser.add_argument("--whitewater-max-particles", type=int,
                        help="override the Hot pour foam/spray/bubble cap for a measured optimization")
    parser.add_argument("--smoke-iterations", type=int,
                        help="override the Hot pour steam solver iteration cap")
    parser.add_argument("--cache-precision", choices=("float32", "float16"),
                        help="override the served volume cache precision")
    parser.add_argument("--cache-channels", choices=("density", "density_temperature",
                                                        "density_temperature_velocity", "all"),
                        help="override the served volume cache channel set")
    args = parser.parse_args()
    force_adapter(args.adapter)
    if (args.cache_mb is not None and (args.cache_mb < 128 or args.cache_mb > 300)) or \
            (args.cache_memory_mb is not None and (args.cache_memory_mb < 1024 or args.cache_memory_mb > 8192)):
        parser.error("cache caps exceed the bounded harness policy")
    result = measure(args.adapter, args.cells, args.frames, Path(args.workdir), args.profile_phases,
                     args.edit, args.cache_mb, args.cache_memory_mb, args.whitewater_max_particles,
                     args.smoke_iterations, args.cache_precision, args.cache_channels)
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    for edit, data in result["edits"].items():
        print(f"{edit}: first={data['first_frame_s']:.3f}s bake={data['all_stages_s']:.2f}s")


if __name__ == "__main__":
    main()
