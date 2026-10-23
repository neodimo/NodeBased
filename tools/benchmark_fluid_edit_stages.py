"""Measure production Hot pour proxy edits, full refinement and viewport playback."""
from __future__ import annotations

import argparse
import copy
import cProfile
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_fluid_edits import EDITS, STAGES
from benchmark_hot_pour import build_document, force_adapter

MAX_SCRATCH_BYTES = 23_000_000_000
EDIT_FRAME = 60


def directory_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def checkpoint(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def base_document(args):
    from nodebased.core import Dispatcher

    graph_args = SimpleNamespace(liquid_cells=args.liquid_cells, smoke_cells=args.smoke_cells,
                                 frames=args.frames, liquid_pressure="resident",
                                 smoke_pressure="resident_sparse", spout_height=0.62)
    doc = build_document(graph_args).document
    # The interactive app scopes cache invalidation through the selected viewer output.
    # Production bake builders leave it unset, so the benchmark must mirror the viewer.
    doc["view"] = "scene"
    # Each future edit adds a constant-interpolated key at frame 60 to these baseline curves.
    for edits in EDITS.values():
        for node_id, param, _delta in edits:
            value = doc["nodes"][node_id]["params"][param]
            doc["animation"]["curves"].setdefault(node_id, {})[param] = {
                "interpolation": "constant", "keys": [{"frame": 1, "value": value}]}
    # Make the geometry tracks explicit so a frame-60 cup move exercises collider lookahead.
    for node in doc["nodes"].values():
        if node.get("type") == "FluidCollide3D":
            node["params"]["animated"] = 1
    return Dispatcher(doc)


def apply_edit(base_doc, edit):
    from nodebased.app import Window
    from nodebased.core import Dispatcher

    dispatcher = Dispatcher(copy.deepcopy(base_doc))
    commands = []
    for node_id, param, delta in EDITS[edit]:
        value = dispatcher.document["nodes"][node_id]["params"][param] + delta
        command = {"op": "set_key", "id": node_id, "param": param,
                   "frame": EDIT_FRAME, "value": value, "interpolation": "constant"}
        dispatcher.execute(command)
        commands.append(command)
    checkpoints = [Window._fluid_checkpoint_frame(dispatcher.document, command)
                   for command in commands]
    checkpoint_frame = min((int(frame) for frame in checkpoints if frame is not None), default=None)
    return dispatcher.document, commands, checkpoint_frame


def prime_reuse(evaluator, doc, checkpoint_frame):
    from nodebased.app import Window
    if checkpoint_frame is None:
        return
    nodes, solver_ids = Window._fluid_simulation_scope(doc)
    for solver_id in solver_ids:
        if nodes[solver_id]["type"] not in ("FluidSolver3D", "FluidLiquidSolver3D"):
            continue
        kind = "smoke" if nodes[solver_id]["type"] == "FluidSolver3D" else "liquid"
        for tier in (1, 4):
            old_run = evaluator._fluid_run_history.get((kind, str(solver_id), tier))
            cached_through = []
            if old_run is not None:
                for cache in (*evaluator._sim_stores.values(), *evaluator._sim_preview_stores.values()):
                    frames = cache.frames(old_run)
                    present = [frame for frame in frames if frame < checkpoint_frame]
                    if present:
                        cached_through.append(max(present))
            print(f"reuse {kind}/{solver_id} tier {tier}: old stream {'present' if old_run else 'missing'}, "
                  f"checkpoint frames found {cached_through}", flush=True)
        evaluator.prime_fluid_reuse(kind, solver_id, checkpoint_frame, tier=1)
        evaluator.prime_fluid_reuse(kind, solver_id, checkpoint_frame, tier=4)
    _visible_nodes, visible = Window._fluid_view_scope(doc)
    for node_id in visible:
        if nodes[node_id].get("type") == "FluidWhitewater3D":
            for tier in (1, 4):
                evaluator.prime_fluid_reuse("whitewater", node_id, checkpoint_frame, tier=tier)


def capped_nodes(doc):
    for node in doc["nodes"].values():
        params = node.get("params", {})
        if node["type"] == "ParticleCache3D":
            params["cache_disk_mb"] = 6144
        elif node["type"] == "FluidWhitewater3D":
            params["cache_disk_mb"] = 1536
        elif node["type"] == "FluidCache3D":
            params["cache_disk_mb"] = 10240


def check_scratch(workdir):
    # Count the entire Lane 6 scratch root so separate adapter caches share one hard budget.
    size = directory_bytes(workdir.parent)
    if size > MAX_SCRATCH_BYTES:
        raise RuntimeError(f"benchmark scratch reached {size} bytes (23 GB cap)")
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("default", "integrated"), required=True)
    parser.add_argument("--liquid-cells", type=int, default=256)
    parser.add_argument("--smoke-cells", type=int, default=192)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--edit", choices=tuple(EDITS) + ("all",), default="all")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--profile-first", action="store_true",
                        help="profile the first edited proxy scene frame, then stop")
    parser.add_argument("--first-only", action="store_true",
                        help="measure and save the first displayed proxy frame, then stop")
    parser.add_argument("--play-width", type=int, default=1920)
    parser.add_argument("--play-height", type=int, default=1080)
    args = parser.parse_args()
    if args.proxy_only and args.full_only:
        parser.error("choose only one of --proxy-only and --full-only")
    if args.frames != 120 or args.liquid_cells != 256 or args.smoke_cells != 192:
        parser.error("gates require 120 frames, 256 liquid cells and 192 steam cells")
    args.workdir.mkdir(parents=True, exist_ok=True)
    force_adapter(args.adapter)

    from nodebased import gpu3d, simcache, viewportgpu
    from nodebased.imaging import Evaluator
    from nodebased.scene3d import proxy_scene

    os.environ["NODEBASED_SIM_CACHE_MB"] = "24000"
    os.environ["XDG_CACHE_HOME"] = str(args.workdir / "xdg")
    sim_root = args.workdir / "simcache"
    template = simcache.SimCache(root=sim_root, memory_budget=2 << 30,
                                 disk_budget=24_000 << 20)
    evaluator = Evaluator(sim=template)
    base = base_document(args)
    base_doc = base.document
    capped_nodes(base_doc)
    renderer = viewportgpu.renderer()
    camera = evaluator.evaluate_raster(base_doc, "camera", frame=1, typed=True)
    result = (json.loads(args.json.read_text()) if args.json.exists() else {
        "adapter_request": args.adapter,
        "adapter": str(gpu3d._state()["info"].get("device")),
        "liquid_cells": args.liquid_cells, "smoke_cells": args.smoke_cells,
        "frames": args.frames, "edit_frame": EDIT_FRAME,
        "proxy_resolution_cells": 64, "viewport": [args.play_width, args.play_height], "edits": {}})
    edit_names = list(EDITS) if args.edit == "all" else [args.edit]

    if args.proxy_only or not args.full_only:
        # Keep a solved, unedited prefix so the interactive edit begins at a realistic checkpoint.
        warm_started = time.perf_counter()
        for frame in range(1, EDIT_FRAME):
            scene = evaluator.evaluate_raster(base_doc, "scene", frame=frame, tier=4, typed=True)
            scene = proxy_scene(scene, 4)
            renderer.render(scene, camera, max(1, -(-args.play_width // 4)),
                            max(1, -(-args.play_height // 4)), (0.02, 0.02, 0.025, 1.0), ambient=0.2)
            if frame % 10 == 0:
                print(f"base proxy prefix {frame}/{EDIT_FRAME - 1}", flush=True)
            check_scratch(args.workdir)
        result["proxy_base_prefix_s"] = round(time.perf_counter() - warm_started, 2)
        checkpoint(args.json, result)

        for edit in edit_names:
            doc, commands, checkpoint_frame = apply_edit(base_doc, edit)
            capped_nodes(doc)
            prime_reuse(evaluator, doc, checkpoint_frame)
            first_frame = EDIT_FRAME if checkpoint_frame is not None else 1
            started = time.perf_counter()
            if args.profile_first:
                profiler = cProfile.Profile()
                scene = profiler.runcall(evaluator.evaluate_raster, doc, "scene", frame=first_frame,
                                         tier=4, typed=True)
                profiler.print_stats(sort="cumulative")
                print(f"profiled edited first frame {edit} at frame {first_frame}; stopping by request",
                      flush=True)
                return
            scene = evaluator.evaluate_raster(doc, "scene", frame=first_frame, tier=4, typed=True)
            scene = proxy_scene(scene, 4)
            renderer.render(scene, camera, max(1, -(-args.play_width // 4)),
                            max(1, -(-args.play_height // 4)), (0.02, 0.02, 0.025, 1.0), ambient=0.2)
            first_elapsed = time.perf_counter() - started
            row = result["edits"].setdefault(edit, {})
            row.update({"adapter": result["adapter"], "keyed_at_frame": EDIT_FRAME,
                        "first_changed_frame": checkpoint_frame or 1,
                        "first_proxy_frame": first_frame,
                        "proxy_first_frame_s": round(first_elapsed, 4),
                        "edit_commands": commands,
                        "proxy_frames": {}})
            checkpoint(args.json, result)
            if args.first_only:
                row["first_proxy_display_measured"] = True
                checkpoint(args.json, result)
                print(f"{edit}: first displayed proxy frame {first_elapsed:.3f}s", flush=True)
                return
            times = []
            for frame in range(1, args.frames + 1):
                started = time.perf_counter()
                scene = evaluator.evaluate_raster(doc, "scene", frame=frame, tier=4, typed=True)
                scene = proxy_scene(scene, 4)
                renderer.render(scene, camera, max(1, -(-args.play_width // 4)),
                                max(1, -(-args.play_height // 4)), (0.02, 0.02, 0.025, 1.0), ambient=0.2)
                elapsed = time.perf_counter() - started
                times.append(elapsed)
                row["proxy_frames"][str(frame)] = round(elapsed, 5)
                row["proxy_timeline_last_frame"] = frame
                if frame % 10 == 0:
                    checkpoint(args.json, result)
                    print(f"{edit}: proxy timeline frame {frame}/{args.frames}, {elapsed:.3f}s", flush=True)
                check_scratch(args.workdir)
            row.update({
                "proxy_120_frames_s": round(sum(times), 2),
                "proxy_frame_p95_ms": round(percentile(times, 0.95) * 1000, 2),
                "proxy_frame_max_ms": round(max(times) * 1000, 2),
                "proxy_mean_fps": round(len(times) / sum(times), 2),
                "scrub_12fps_holds": max(times) <= (1.0 / 12.0),
            })
            row["proxy_cache_bytes"] = check_scratch(args.workdir)
            checkpoint(args.json, result)
            print(f"{edit}: first proxy frame {row['proxy_first_frame_s']:.3f}s; "
                  f"120-frame proxy {row['proxy_120_frames_s']:.2f}s; "
                  f"scrub {row['proxy_mean_fps']:.2f} fps, p95 {row['proxy_frame_p95_ms']:.2f} ms",
                  flush=True)

    if args.full_only or not args.proxy_only:
        # Tier 1 gets the matching unchanged prefix once; subsequent keyed variants can reuse it.
        warm_started = time.perf_counter()
        for stage in STAGES:
            for frame in range(1, EDIT_FRAME):
                evaluator.evaluate_raster(base_doc, stage, frame=frame, tier=1, typed=True)
                if frame % 10 == 0:
                    print(f"base full {stage} frame {frame}/{EDIT_FRAME - 1}", flush=True)
                check_scratch(args.workdir)
        result["full_base_prefix_s"] = round(time.perf_counter() - warm_started, 2)
        checkpoint(args.json, result)

        for edit in edit_names:
            doc, commands, checkpoint_frame = apply_edit(base_doc, edit)
            capped_nodes(doc)
            prime_reuse(evaluator, doc, checkpoint_frame)
            row = result["edits"].setdefault(edit, {})
            row["full_detail_stage_s"] = {}
            row["full_detail_stage_frames"] = {}
            for stage in STAGES:
                frame_times = []
                for frame in range(1, args.frames + 1):
                    started = time.perf_counter()
                    evaluator.evaluate_raster(doc, stage, frame=frame, tier=1, typed=True)
                    elapsed = time.perf_counter() - started
                    frame_times.append(elapsed)
                    if frame % 10 == 0:
                        print(f"{edit}: full {stage} frame {frame}/{args.frames}, {elapsed:.2f}s", flush=True)
                    check_scratch(args.workdir)
                row["full_detail_stage_s"][stage] = round(sum(frame_times), 2)
                row["full_detail_stage_frames"][stage] = {
                    "mean_ms": round(statistics.fmean(frame_times) * 1000, 2),
                    "p95_ms": round(percentile(frame_times, 0.95) * 1000, 2),
                }
                checkpoint(args.json, result)
            row["full_detail_total_s"] = round(sum(row["full_detail_stage_s"].values()), 2)
            row["full_detail_cache_bytes"] = check_scratch(args.workdir)
            checkpoint(args.json, result)
            print(f"{edit}: full-detail {row['full_detail_total_s']:.1f}s; "
                  + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in row["full_detail_stage_s"].items()),
                  flush=True)

    checkpoint(args.json, result)


if __name__ == "__main__":
    main()
