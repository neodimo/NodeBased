"""A reproducible 1920 x 1080 progression of the viewport's Render mode: time-stamped frames from the denoised first look to the
settled picture, plus the latencies an artist feels (lane 4, plan "Progress", step H2).

    flock /tmp/nb-gpu.lock python tools/benchmark_progression.py [--adapter default|discrete|integrated|cpu]
        [--scene furnished|mixed|both] [--warm N] [--frames DIR] [--json out.json]

Per scene it reports, on the adapter in use (the scene, the camera, the seed and the render settings are fixed in this file):

* ``cold`` and ``warm`` progressions: the frames `progressivejob.ProgressiveJob` published for one fresh view of the
  full-HD output (seconds since the request, samples, whether it is the enlarged preview, what was traced, the pass number),
  ``first_image_seconds`` (request to the first denoised picture converted for display, the debounce included) and
  ``settled_seconds`` (request to the converged picture). Cold is the first view in this process (pipelines compiled, scene
  uploaded); warm moves the camera a little for every repeat (a new view, the same device objects) and reports the median.
* ``interaction``: a real `Viewport3D` at 1920 x 1080 is orbited by camera nudges every 33 ms for a second while the worker is
  busy, painting after each one. It reports the paint handler's own time (median and worst: the interface thread), how long the
  worker took to abandon the pass it was in (``cancel_latency``), the number of first looks it actually started for the burst
  (the debounce collapses the rest) and the seconds from the last nudge to the first denoised picture on screen.
* ``responsive_during_refinement``: the worst paint time while a long full-size pass is in flight.

``--frames DIR`` writes the first look, each later stage and the settled picture as PNGs named with their time stamps. Run it
under the exclusive GPU lock: the timings are meaningless next to another job on the card.
"""
import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from benchmark_pathtrace import CAMERA, furnished_scene, soft_scene
from nodebased import gpu3d, progressiverender as P, scene3d as s
from nodebased.progressivejob import ProgressiveJob, Request

WIDTH, HEIGHT = 1920, 1080
BACKGROUND = (0.02, 0.02, 0.03, 1.0)
AMBIENT = 0.05
SETTLE_TIMEOUT = 600.0


def moved(i):
    position = CAMERA.transform.position
    return replace(CAMERA, transform=replace(CAMERA.transform, position=s.Vec3(position.x + 0.01 * i, position.y, position.z)))


def progression(job, scene, camera, identity):
    """Submit one view, wait for it to settle, return its record."""
    submitted = time.perf_counter()
    job.submit(Request(identity, scene, camera, WIDTH, HEIGHT, BACKGROUND, AMBIENT))
    if not job.wait(lambda state, phase: phase in ("settled", "failed"), SETTLE_TIMEOUT):
        raise RuntimeError("the view did not settle in time")
    if job.error is not None:
        raise job.error
    frames = []
    shown = None
    for frame in job.frames:
        state = frame.state
        rgba = frame.rgba          # converted on the worker before the frame was published, so `since_submit` includes it
        frames.append({"seconds": round(frame.since_submit, 4), "display_seconds": round(frame.since_submit, 4),
                       "samples": state.samples, "pass": state.passes, "preview": state.preview, "denoised": state.denoised,
                       "traced": [state.traced_width or state.width, state.traced_height or state.height],
                       "converged_fraction": state.converged_fraction, "phases": {k: round(v, 4) for k, v in state.phases}})
        if shown is None:
            shown = rgba
    return {"frames": frames, "first_image_seconds": frames[0]["display_seconds"],
            "settled_seconds": frames[-1]["seconds"], "frame_count": len(frames),
            "wall_seconds": round(time.perf_counter() - submitted, 4)}, [frame.state for frame in job.frames]


def save_frames(directory, name, states, record):
    from PySide6.QtGui import QImage
    Path(directory).mkdir(parents=True, exist_ok=True)
    for state, frame in zip(states, record["frames"]):
        rgba = np.ascontiguousarray(P.display_rgba8(state.image))
        image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888)
        label = "preview" if frame["preview"] else "pass"
        image.save(str(Path(directory) / f"{name}-{frame['seconds']:06.2f}s-{frame['samples']:02d}spp-{label}.png"))


def interaction(scene, camera):
    """Orbit a real widget at the output size while the worker is busy; see the module docstring for the numbers."""
    from PySide6.QtWidgets import QApplication
    import tests.isolation  # noqa: F401  (Qt settings of the run stay out of the user's real ones)
    from nodebased.viewport3d import Viewport3D
    app = QApplication.instance() or QApplication([])
    widget = Viewport3D()
    widget.resize(WIDTH, HEIGHT)
    widget._scene_cache = ((1, None), (scene, camera))
    widget.look_through = True
    widget.render_mode = True
    job = widget._job()
    paint_ms = []

    def paint():
        started = time.perf_counter()
        widget.grab()
        app.processEvents()
        paint_ms.append((time.perf_counter() - started) * 1000.0)

    # settle a first view so the device objects are warm, then orbit
    widget.grab()
    job.wait(lambda state, phase: phase == "settled", SETTLE_TIMEOUT)
    paint_ms.clear()
    nudges = 30
    last = None
    for i in range(nudges):
        widget._scene_cache = ((1, None), (scene, moved(100 + i)))
        last = time.perf_counter()
        paint()
        end = time.perf_counter() + 0.033
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.002)
    burst = list(paint_ms)
    started_first = len(job.frames)
    waited = time.perf_counter()
    job.wait(lambda state, phase: state is not None, SETTLE_TIMEOUT)
    to_first = job.frames[0].since_submit if job.frames else None
    paint_ms.clear()
    deadline = time.perf_counter() + 4.0          # paints while a full-size pass is running
    while time.perf_counter() < deadline and job.view(job._identity)[1] != "settled":
        paint()
        time.sleep(0.02)
    during = list(paint_ms)
    job.wait(lambda state, phase: phase == "settled", SETTLE_TIMEOUT)
    cancels = list(job.cancel_latencies)
    widget.close()
    return {"nudges": nudges, "interval_ms": 33,
            "paint_ms_during_orbit": {"median": round(statistics.median(burst), 2), "worst": round(max(burst), 2)},
            "cancel_latency_ms": {"count": len(cancels), "median": round(1000 * statistics.median(cancels), 2) if cancels else None,
                                  "worst": round(1000 * max(cancels), 2) if cancels else None},
            "last_nudge_to_first_look_seconds": round(to_first, 4) if to_first is not None else None,
            "responsive_during_refinement": {"paints": len(during), "median_ms": round(statistics.median(during), 2) if during else None,
                                             "worst_ms": round(max(during), 2) if during else None}}


def measure(name, scene, warm, frames_dir, with_interaction):
    out = {"scene": name, "output": [WIDTH, HEIGHT], "seed": "default", "ambient": AMBIENT}
    job = ProgressiveJob()
    try:
        cold, states = progression(job, scene, CAMERA, ("cold", name))
        out["cold"] = cold
        if frames_dir:
            save_frames(frames_dir, f"{name}-cold", states, cold)
        runs = []
        for i in range(1, warm + 1):
            record, states = progression(job, scene, moved(i), ("warm", name, i))
            runs.append(record)
        out["warm"] = {"runs": len(runs),
                       "first_image_seconds": {"median": round(statistics.median(r["first_image_seconds"] for r in runs), 4),
                                               "worst": round(max(r["first_image_seconds"] for r in runs), 4)},
                       "settled_seconds": {"median": round(statistics.median(r["settled_seconds"] for r in runs), 3),
                                           "worst": round(max(r["settled_seconds"] for r in runs), 3)},
                       "frames_median_seconds": [round(statistics.median(r["frames"][k]["seconds"] for r in runs), 3)
                                                 for k in range(min(len(r["frames"]) for r in runs))],
                       "last_run_frames": runs[-1]["frames"]}
        if frames_dir and runs:
            save_frames(frames_dir, f"{name}-warm", states, runs[-1])
    finally:
        job.close()
    out["target_seconds"] = P.FIRST_LOOK_SECONDS
    out["first_look_target_met_warm"] = out["warm"]["first_image_seconds"]["worst"] <= P.FIRST_LOOK_SECONDS
    out["first_look_target_met_cold"] = out["cold"]["first_image_seconds"] <= P.FIRST_LOOK_SECONDS
    if with_interaction:
        out["interaction"] = interaction(scene, CAMERA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--scene", default="both", choices=("furnished", "mixed", "both"))
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.adapter:
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    print(gpu3d.adapter_report())
    results = []
    for name in (("furnished", "mixed") if args.scene == "both" else (args.scene,)):
        scene = furnished_scene() if name == "furnished" else soft_scene()
        P._COST.clear()
        result = measure(name, scene, args.warm, args.frames, not args.no_interaction)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("cold", "warm")}, indent=1))
        print("cold:", [(f["seconds"], f["samples"], "preview" if f["preview"] else "pass") for f in result["cold"]["frames"]])
        print("warm median first image:", result["warm"]["first_image_seconds"], "settled:", result["warm"]["settled_seconds"])
    if args.json:
        Path(args.json).write_text(json.dumps({"adapter": gpu3d.adapter_report(), "results": results}, indent=1))


if __name__ == "__main__":
    main()
