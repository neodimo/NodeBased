"""Cold/warm first-picture stage timings for the 64-cell Hot pour shot.

    flock /tmp/nb-gpu.lock python tools/benchmark_fluid_firstlook.py \
        --cache-dir /path/to/hot-pour-sim-cache --json benchmarks/hot_pour/firstlook-timeline-YYYY-MM-DD.json

The cache directory must contain the baked frame-60 simulation and extracted surface mesh. The first
picture is measured through ProgressiveJob, including its debounce and display conversion.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased import flip3d, gpupathtrace, gpu3d, pathtrace, progressiverender as P, scene3d
from nodebased.core import load_document
from nodebased.imaging import Evaluator
from nodebased.progressivejob import ProgressiveJob, Request
from nodebased.simcache import SimCache

ROOT = Path(__file__).resolve().parents[1]
SHOT = ROOT / "examples/hot_pour_shot/hot_pour_shot.nbcomp"
WIDTH, HEIGHT, FRAME = 1920, 1080, 60
REQUIRED_STAGES = ("cache_read", "surface_mesh_load", "scene_evaluation", "volume_upload",
                   "acceleration_build", "shader_pipeline_build", "first_trace", "guide_passes", "first_present")


def timeline_stages(state, cache_read=0.0, surface_mesh_load=0.0, gpu_phases=None,
                    pipeline_build=0.0, present_convert=0.0, guide_seconds=None, scene_evaluation=0.0):
    """Return the stable stage names used by the JSON and the CPU timeline unit test."""
    phases = dict(state.phases)
    gpu_phases = gpu_phases or {}
    return {
        "cache_read": float(cache_read),
        "surface_mesh_load": float(surface_mesh_load),
        "scene_evaluation": float(scene_evaluation),
        "volume_upload": float(gpu_phases.get("upload", 0.0)),
        "acceleration_build": float(gpu_phases.get("scene build", 0.0) + gpu_phases.get("pack", 0.0)),
        "shader_pipeline_build": float(pipeline_build),
        "first_trace": float(phases.get("render", 0.0)),
        "guide_passes": float(phases.get("guides", 0.0) if guide_seconds is None else guide_seconds),
        "first_present": float(phases.get("denoise", 0.0) + phases.get("enlarge", 0.0) + present_convert),
    }


def _timed_first(job, scene, camera, identity):
    job.submit(Request(identity, scene, camera, WIDTH, HEIGHT, (0.02, 0.02, 0.025, 1.0), 0.15))
    if not job.wait(lambda state, _phase: state is not None, 90):
        raise RuntimeError("progressive first picture did not arrive")
    frame = job.frames[0]
    return frame.state, frame.since_submit, frame.rgba


def _camera_cancel_latency(scene, camera):
    """Supersede the active refinement with a camera move and time its cancellation boundary."""
    job = ProgressiveJob(settle=0.0, denoise_passes=False)
    moved = replace(camera, transform=replace(camera.transform,
                                               position=scene3d.Vec3(camera.transform.position.x + 0.01,
                                                                     camera.transform.position.y,
                                                                     camera.transform.position.z)))
    first_id, next_id = ("cancel-check", 0), ("cancel-check", 1)
    try:
        job.submit(Request(first_id, scene, camera, WIDTH, HEIGHT, (0.02, 0.02, 0.025, 1.0), 0.15))
        if not job.wait(lambda state, _phase: state is not None, 90):
            raise RuntimeError("first picture did not arrive before cancellation check")
        deadline = time.perf_counter() + 2.0
        while job.view(first_id)[1] != "tracing" and time.perf_counter() < deadline:
            time.sleep(0.001)
        time.sleep(0.005)
        job.submit(Request(next_id, scene, moved, WIDTH, HEIGHT, (0.02, 0.02, 0.025, 1.0), 0.15))
        if not job.wait(lambda state, _phase: state is not None, 90):
            raise RuntimeError("moved-camera first picture did not arrive after cancellation")
        latencies = list(job.cancel_latencies)
        return {"camera_move_cancel_latency_ms": 1000.0 * latencies[-1] if latencies else None,
                "new_view_first_picture_seconds": job.frames[0].since_submit if job.frames else None,
                "cancel_target_ms": 50.0}
    finally:
        job.close()


def measure(cache_dir, warm_runs=3, compare=False):
    cache_profile = {}
    evaluator = Evaluator(sim=SimCache(root=cache_dir, memory_budget=1 << 30,
                                       disk_budget=16 << 30, profile=cache_profile))
    document = load_document(SHOT)
    gpu_runs, pipeline_runs, surface_loads, surface_cache_loads, guide_runs, display_runs = [], [], [], [], [], []
    original_render = pathtrace.render
    original_pipeline = gpupathtrace._pipeline
    original_get = SimCache.get
    original_guides = pathtrace.guide_aovs
    original_surface = flip3d.surface_geometry
    original_display = P.display_rgba8

    def render_probe(*args, **kwargs):
        stats = kwargs.get("stats")
        value = original_render(*args, **kwargs)
        if stats is not None:
            gpu_runs.append({"backend": stats.get("backend"), "phases": dict(stats.get("phases", {}))})
        return value

    def pipeline_probe(*args, **kwargs):
        started = time.perf_counter()
        value = original_pipeline(*args, **kwargs)
        pipeline_runs.append(time.perf_counter() - started)
        return value

    def cache_probe(store, *args, **kwargs):
        started = time.perf_counter()
        value = original_get(store, *args, **kwargs)
        if store.root is not None and store.root.name == "surface-mesh-v1":
            surface_cache_loads.append(time.perf_counter() - started)
        return value

    def guide_probe(*args, **kwargs):
        started = time.perf_counter()
        value = original_guides(*args, **kwargs)
        guide_runs.append(time.perf_counter() - started)
        return value

    def surface_probe(*args, **kwargs):
        started = time.perf_counter()
        value = original_surface(*args, **kwargs)
        surface_loads.append(time.perf_counter() - started)
        return value

    def display_probe(*args, **kwargs):
        started = time.perf_counter()
        value = original_display(*args, **kwargs)
        display_runs.append(time.perf_counter() - started)
        return value

    pathtrace.render, pathtrace.guide_aovs, flip3d.surface_geometry = render_probe, guide_probe, surface_probe
    P.display_rgba8 = display_probe
    gpupathtrace._pipeline, SimCache.get = pipeline_probe, cache_probe
    rows = []
    job = ProgressiveJob(settle=0.04, denoise_passes=False, keep_images=True)
    try:
        for index, name in enumerate(("cold", *(f"warm-{i+1}" for i in range(warm_runs)))):
            cache_before = dict(cache_profile)
            start = time.perf_counter()
            scene = evaluator.evaluate_raster(document, "scene", frame=FRAME, typed=True)
            camera = evaluator.evaluate_raster(document, "camera", frame=FRAME, typed=True)
            evaluation = time.perf_counter() - start
            if index:
                position = camera.transform.position
                camera = replace(camera, transform=replace(camera.transform,
                                                           position=scene3d.Vec3(position.x + index * 0.01,
                                                                                 position.y, position.z)))
            gpu_start, pipe_start, surface_start, surface_cache_start, guide_start, display_start = (
                len(gpu_runs), len(pipeline_runs), len(surface_loads), len(surface_cache_loads),
                len(guide_runs), len(display_runs))
            counters_before = gpupathtrace.reuse_counters()
            state, shown_s, rgba = _timed_first(job, scene, camera, (name, index))
            counters_after = gpupathtrace.reuse_counters()
            present_start = time.perf_counter()
            if rgba is None:
                rgba = P.display_rgba8(state.image)
            display_s = sum(display_runs[display_start:]) + time.perf_counter() - present_start
            job.cancel()
            gpu_stats = [row for row in gpu_runs[gpu_start:] if row.get("backend") == "gpu"]
            gpu_phase = {}
            for row in gpu_stats:
                for key, value in row.get("phases", {}).items():
                    gpu_phase[key] = gpu_phase.get(key, 0.0) + float(value)
            cache_delta = {key: max(0.0, float(value) - float(cache_before.get(key, 0.0)))
                           for key, value in cache_profile.items()}
            cache_read = sum(cache_delta.get(key, 0.0) for key in
                             ("file_open_s", "file_read_s", "deserialize_s", "decompress_s", "sparse_rebuild_s"))
            surface_load = sum(surface_loads[surface_start:]) + sum(surface_cache_loads[surface_cache_start:])
            guide_seconds = sum(guide_runs[guide_start:])
            stages = timeline_stages(state, cache_read, surface_load, gpu_phase,
                                     sum(pipeline_runs[pipe_start:]), display_s, guide_seconds,
                                     max(0.0, evaluation - cache_read - surface_load))
            rows.append({"case": name, "tier": gpu3d.adapter_report().splitlines()[0],
                         "first_picture_seconds": float(shown_s), "evaluation_seconds": evaluation,
                         "press_to_first_picture_seconds": float(evaluation + shown_s),
                         "traced_size": [state.traced_width, state.traced_height], "stage_seconds": stages,
                         "gpu_counter_delta": {key: counters_after[key] - counters_before[key] for key in counters_before},
                         "presented_rgba": bool(rgba is not None), "simcache_profile": cache_delta,
                         "gpu_stats": gpu_stats[-1].get("phases", {}) if gpu_stats else {},
                         "gpu_backend": gpu_stats[-1].get("backend") if gpu_stats else "cpu"})
    finally:
        job.close()
        pathtrace.render, pathtrace.guide_aovs, flip3d.surface_geometry = original_render, original_guides, original_surface
        P.display_rgba8 = original_display
        gpupathtrace._pipeline, SimCache.get = original_pipeline, original_get
    warm = [row["press_to_first_picture_seconds"] for row in rows[1:]]
    result = {"adapter": gpu3d.adapter_report(), "resolution": [WIDTH, HEIGHT], "frame": FRAME,
            "before": {"first_picture_seconds": 26.69, "traced_size": [480, 270], "samples": 1,
                       "path": "ordinary progressive reset step, measured before this change"},
            "target_seconds": 1.0, "cold_target_met": rows[0]["press_to_first_picture_seconds"] < 3.0,
            "warm_target_met": max(warm, default=float("inf")) < 1.0,
            "warm_median_seconds": statistics.median(warm) if warm else None,
            "warm_worst_seconds": max(warm) if warm else None,
            "stage_names": list(REQUIRED_STAGES), "runs": rows}
    result["progressive_interaction"] = _camera_cancel_latency(scene, camera)
    if compare:
        scene = evaluator.evaluate_raster(document, "scene", frame=FRAME, typed=True)
        camera = evaluator.evaluate_raster(document, "camera", frame=FRAME, typed=True)
        comparison = {}
        for backend in ("gpu", "cpu"):
            state = P.step_interactive(None, scene, camera, 96, 54, (0.02, 0.02, 0.025, 1.0), 0.15,
                                       ("frame-60-compare", backend), backend=backend,
                                       stages=((64 * 36, 4),))
            comparison[backend] = state.image
        delta = abs(comparison["gpu"][..., :3] - comparison["cpu"][..., :3])
        alpha_delta = abs(comparison["gpu"][..., 3] - comparison["cpu"][..., 3])
        result["frame60_cpu_gpu_comparison"] = {"resolution": [96, 54], "samples": 4,
                                                 "mean_absolute_rgb_error": float(delta.mean()),
                                                 "p95_absolute_rgb_error": float(np.percentile(delta, 95)),
                                                 "mean_absolute_alpha_error": float(alpha_delta.mean()),
                                                 "gpu": gpu3d.adapter_report().splitlines()[0],
                                                 "cpu": "CPU reference"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--adapter", choices=("default", "integrated"), default="default")
    parser.add_argument("--compare", action="store_true", help="compare CPU and GPU frame-60 thumbnails")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    if args.adapter == "integrated":
        original = gpu3d._state
        gpu3d._state = lambda choice=None: original("integrated" if (choice or "default") == "default" else choice)
    result = measure(args.cache_dir, args.warm, args.compare)
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"adapter": result["adapter"], "cold_s": result["runs"][0]["press_to_first_picture_seconds"],
                      "warm_median_s": result["warm_median_seconds"], "warm_worst_s": result["warm_worst_seconds"],
                      "frame60_cpu_gpu_comparison": result.get("frame60_cpu_gpu_comparison"),
                      "stage_names": result["stage_names"], "json": str(target)}, indent=1))


if __name__ == "__main__":
    main()
