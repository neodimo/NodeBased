"""Measure worker-stop and restart latency when a second cup edit supersedes a fluid bake."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_fluid_edits import build_document
from benchmark_hot_pour import force_adapter


def trial(adapter, delay_ms):
    from nodebased import flip3d, flip_gpu_resident, fluid3d, gpu3d, simcache
    from nodebased.cancellation import Cancelled

    args = SimpleNamespace(liquid_cells=64, smoke_cells=64, frames=120,
                           liquid_pressure="resident", smoke_pressure="resident_sparse", spout_height=0.62)
    document = build_document(args).document
    liquid_key, node = next((key, value) for key, value in document["nodes"].items()
                            if value["type"] == "FluidLiquidSolver3D")

    def edited_stream(speed):
        params = dict(node["params"])
        source = fluid3d.Source(center=(0.0, 0.12, 0.0), radius=0.08,
                                velocity=(0.0, float(speed), 0.0), start_frame=1,
                                end_frame=120, fluid_type="liquid")
        # The resident solver registry is keyed by run identity. Give each trial a
        # fresh run so instrumentation always wraps the solver actually exercised.
        chain = fluid3d.FluidChain(sources=(source,), run=f"cancel-benchmark-{speed}-{time.monotonic_ns()}")
        return flip3d.build_stream(document, liquid_key, {"params": params}, chain)

    first_edit = edited_stream(0.04)
    second_edit = edited_stream(0.08)

    entered_step = threading.Event()
    cancel = threading.Event()
    original_create = flip_gpu_resident.create_solver

    def observed_create(*args, **kwargs):
        solver = original_create(*args, **kwargs)
        original_step = solver.step

        def observed_step(*step_args, **step_kwargs):
            entered_step.set()
            return original_step(*step_args, **step_kwargs)

        solver.step = observed_step
        return solver

    flip_gpu_resident.create_solver = observed_create
    outcome = {"error": None}

    def solve_old_edit():
        try:
            flip3d.solve_frame(first_edit, 2, simcache.SimCache(enabled=False), cancel=cancel)
        except Cancelled:
            outcome["error"] = "cancelled"
        except BaseException as error:  # preserve worker failures in the benchmark artifact
            outcome["error"] = "".join(traceback.format_exception(error))

    worker = threading.Thread(target=solve_old_edit, name="fluid-edit-cancel-benchmark")
    try:
        worker.start()
        deadline = time.perf_counter() + 30.0
        while not entered_step.is_set() and worker.is_alive() and time.perf_counter() < deadline:
            entered_step.wait(0.001)
        if not entered_step.is_set():
            worker.join(0.1)
            raise RuntimeError(f"liquid GPU worker did not enter a solve substep; worker result: {outcome['error']}")
        time.sleep(max(0.0, delay_ms / 1000.0))
        cancel_at = time.perf_counter()
        cancel.set()  # same event PlaybackQueue.replace sets when a newer edit arrives
        worker.join(10)
        stopped_at = time.perf_counter()
        if worker.is_alive():
            raise RuntimeError("superseded worker did not stop within 10 seconds")
        if outcome["error"] != "cancelled":
            raise RuntimeError(f"old bake completed or failed before cancellation: {outcome['error']}")
        restart_started = time.perf_counter()
        flip3d.solve_frame(second_edit, 1, simcache.SimCache(enabled=False))
        restart_seconds = time.perf_counter() - restart_started
        return {"adapter": gpu3d._state()["info"].get("device"),
                "cancel_after_step_entry_ms": delay_ms,
                "worker_stop_ms": round((stopped_at - cancel_at) * 1000, 3),
                "new_edit_first_frame_s": round(restart_seconds, 4),
                "cancelled": True, "new_edit_started": True, "target_ms": 50.0}
    finally:
        if worker.is_alive():
            cancel.set()
            worker.join(10)
        flip_gpu_resident.create_solver = original_create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("default", "integrated", "cpu"), required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--delay-ms", type=float, default=5.0)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    force_adapter(args.adapter)
    rows = [trial(args.adapter, args.delay_ms) for _ in range(args.trials)]
    result = {"adapter_request": args.adapter, "trials": rows,
              "worker_stop_ms": {"count": len(rows), "median": round(statistics.median(r["worker_stop_ms"] for r in rows), 3),
                                 "worst": round(max(r["worker_stop_ms"] for r in rows), 3)},
              "cancel_target_ms": 50.0,
              "all_trials_pass": all(r["worker_stop_ms"] < 50 for r in rows)}
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.adapter}: worker-stop worst={result['worker_stop_ms']['worst']:.3f} ms, "
          f"gate={'PASS' if result['all_trials_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
