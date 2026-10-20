"""A reproducible 1920 x 1080 progression of the viewport's Render mode: time-stamped frames from the denoised first look to the
settled picture, plus the latencies an artist feels (lane 4, plan "Progress", step H2).

    flock /tmp/nb-gpu.lock python tools/benchmark_progression.py [--adapter default|discrete|integrated|cpu]
        [--scene furnished|mixed|both] [--warm N] [--edits N] [--frames DIR] [--json out.json]
        [--field PATH [--quick]]

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

* ``edit_latency`` (lane 4, step A of 3; ``--edits N`` trials per kind, default 10): a camera nudge and a material knob change
  (the floor's roughness) land while a full-size pass (16 samples or more done, the next running) is in flight, at spread-out
  moments of the pass. Per trial: ``abandon_ms``, the edit to the moment the worker has stopped the obsolete pass, and
  ``edit_to_first_look_ms``, the edit to the new view's first denoised picture converted for display (the debounce included),
  beside ``idle_first_look_ms``, the same first look asked of an idle worker. Gates, applied on the RTX 3080 Ti only (no target
  on other adapters, whose rows are recorded): 95th-percentile abandon under 50 ms, and the 95th-percentile edit-to-first-look
  no worse than the idle first look plus 60 ms, on both scenes. ``--no-edits`` leaves the series out.
* ``--field PATH`` writes one small JSON (adapter name, backend, first look, abandon and settle times) and nothing else, for
  ``tools/field_report.py``; ``--quick`` with it runs the furnished room only with fewer trials and stops each view at its first
  full-size pass instead of waiting for it to settle (no settle time then).

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


SETTLE = True     # False (the field kit's --quick): a progression ends at the first full-size pass and reports no settle time


def progression(job, scene, camera, identity):
    """Submit one view, wait for it to settle, return its record."""
    submitted = time.perf_counter()
    job.submit(Request(identity, scene, camera, WIDTH, HEIGHT, BACKGROUND, AMBIENT))
    reached = (lambda state, phase: phase in ("settled", "failed")) if SETTLE else \
        (lambda state, phase: phase in ("settled", "failed") or (state is not None and state.samples >= 16 and phase == "tracing"))
    if not job.wait(reached, SETTLE_TIMEOUT):
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
            "settled_seconds": frames[-1]["seconds"] if SETTLE else None, "frame_count": len(frames),
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


EDIT_DELAYS = (0.0, 0.021, 0.047, 0.083, 0.131, 0.19, 0.27, 0.36, 0.05, 0.11)     # seconds into the pass the edit lands
ABANDON_GATE_MS = 50.0
EDIT_EXTRA_MS = 60.0


def percentile(values, q):
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(np.ceil(q / 100.0 * len(ordered))) - 1))]


def summary(values_ms):
    if not values_ms:
        return None
    return {"n": len(values_ms), "median": round(statistics.median(values_ms), 2), "p95": round(percentile(values_ms, 95), 2),
            "worst": round(max(values_ms), 2)}


def edited_scene(scene, i):
    """`scene` with the first geometry's (the floor's) roughness knob turned to a value of its own for trial `i`."""
    geometry = scene.geometries[0]
    rough = 0.05 + 0.9 * ((i * 0.37) % 1.0)
    return replace(scene, geometries=(replace(geometry, pbr_roughness=rough),) + tuple(scene.geometries[1:]))


def edit_request(kind, scene, i, tag):
    camera = moved(200 + i) if kind == "camera" else CAMERA
    return Request(("edit", kind, i, tag), edited_scene(scene, i) if kind == "material" else scene, camera, WIDTH, HEIGHT,
                   BACKGROUND, AMBIENT)


def edit_latency(job, scene, trials, label=None):
    """The edit-latency series (see the module docstring). `job` is warm. Returns the per-kind records and the pooled summary."""
    kinds = {}
    for kind in ("camera", "material"):
        rows = []
        for i in range(trials):
            # the idle first look of this very edit: nothing running when it arrives
            job.cancel()
            job.wait(timeout=60)
            job.submit(edit_request(kind, scene, 2 * i, "idle"))
            if not job.wait(lambda state, phase: state is not None, SETTLE_TIMEOUT):
                raise RuntimeError("no first look")
            idle = job.frames[0].since_submit * 1000.0
            # a long full-size pass in flight, then the edit
            if not job.wait(lambda state, phase: state is not None and state.samples >= 16 and phase == "tracing",
                            SETTLE_TIMEOUT):
                raise RuntimeError("the view never reached a full-size pass")
            time.sleep(EDIT_DELAYS[i % len(EDIT_DELAYS)])
            before = len(job.cancel_latencies)
            job.submit(edit_request(kind, scene, 2 * i + 1, "live"))
            deadline = time.perf_counter() + 30.0
            while len(job.cancel_latencies) == before and time.perf_counter() < deadline:
                time.sleep(0.0005)
            abandon = job.cancel_latencies[before] * 1000.0 if len(job.cancel_latencies) > before else None
            if not job.wait(lambda state, phase: state is not None, SETTLE_TIMEOUT):
                raise RuntimeError("no first look after the edit")
            first = job.frames[0].since_submit * 1000.0
            rows.append({"delay_s": EDIT_DELAYS[i % len(EDIT_DELAYS)], "idle_first_look_ms": round(idle, 2),
                         "abandon_ms": None if abandon is None else round(abandon, 2), "edit_to_first_look_ms": round(first, 2)})
        kinds[kind] = {"trials": rows,
                       "abandon_ms": summary([r["abandon_ms"] for r in rows if r["abandon_ms"] is not None]),
                       "edit_to_first_look_ms": summary([r["edit_to_first_look_ms"] for r in rows]),
                       "idle_first_look_ms": summary([r["idle_first_look_ms"] for r in rows])}
    every = [r for k in kinds.values() for r in k["trials"]]
    pooled = {"abandon_ms": summary([r["abandon_ms"] for r in every if r["abandon_ms"] is not None]),
              "edit_to_first_look_ms": summary([r["edit_to_first_look_ms"] for r in every]),
              "idle_first_look_ms": summary([r["idle_first_look_ms"] for r in every]),
              "abandoned_trials": sum(1 for r in every if r["abandon_ms"] is not None), "trials": len(every)}
    job.cancel()
    return {"by_kind": kinds, **pooled, "gate": edit_gate(pooled)}


def on_rtx_3080_ti():
    try:
        return "3080 Ti" in gpu3d.adapter_report().splitlines()[0]
    except Exception:
        return False


def edit_gate(pooled):
    if not on_rtx_3080_ti():
        return {"applies": False, "note": "no target on this adapter; the row is recorded"}
    abandon, first, idle = pooled["abandon_ms"], pooled["edit_to_first_look_ms"], pooled["idle_first_look_ms"]
    abandon_ok = abandon is not None and pooled["abandoned_trials"] == pooled["trials"] and abandon["p95"] < ABANDON_GATE_MS
    first_ok = first["p95"] <= idle["p95"] + EDIT_EXTRA_MS
    return {"applies": True, "abandon_p95_under_ms": ABANDON_GATE_MS, "abandon_ok": bool(abandon_ok),
            "edit_to_first_look_p95_within_ms_of_idle_p95": EDIT_EXTRA_MS, "edit_to_first_look_ok": bool(first_ok),
            "met": bool(abandon_ok and first_ok)}


def measure(name, scene, warm, frames_dir, with_interaction, edits=0):
    out = {"scene": name, "output": [WIDTH, HEIGHT], "seed": "default", "ambient": AMBIENT}
    job = ProgressiveJob(keep_images=bool(frames_dir))
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
                       "settled_seconds": ({"median": round(statistics.median(r["settled_seconds"] for r in runs), 3),
                                            "worst": round(max(r["settled_seconds"] for r in runs), 3)} if SETTLE
                                           else {"median": None, "worst": None}),
                       "frames_median_seconds": [round(statistics.median(r["frames"][k]["seconds"] for r in runs), 3)
                                                 for k in range(min(len(r["frames"]) for r in runs))],
                       "last_run_frames": runs[-1]["frames"]}
        if frames_dir and runs:
            save_frames(frames_dir, f"{name}-warm", states, runs[-1])
        if edits:
            out["edit_latency"] = edit_latency(job, scene, edits)      # the job is warm: pipelines compiled, scene uploaded
    finally:
        job.close()
    out["target_seconds"] = P.FIRST_LOOK_SECONDS
    out["first_look_target_met_warm"] = out["warm"]["first_image_seconds"]["worst"] <= P.FIRST_LOOK_SECONDS
    out["first_look_target_met_cold"] = out["cold"]["first_image_seconds"] <= P.FIRST_LOOK_SECONDS
    if with_interaction:
        out["interaction"] = interaction(scene, CAMERA)
    return out


def backend_name():
    """The backend a tiny path-traced render really ran on (`gpu`, also on llvmpipe, or `cpu`)."""
    from nodebased import pathtrace
    stats = {}
    pathtrace.render(furnished_scene(), CAMERA, 32, 18, BACKGROUND, AMBIENT, "rgba", pathtrace.PathSettings(samples=1),
                     backend="auto", stats=stats)
    return stats.get("backend", "unknown")


def field_record(results):
    """The small record the field kit keeps: what this machine measured, nothing from any other machine."""
    scenes = {}
    for result in results:
        edit = result.get("edit_latency") or {}
        scenes[result["scene"]] = {
            "first_look_seconds": {"cold": result["cold"]["first_image_seconds"],
                                   "warm_median": result["warm"]["first_image_seconds"]["median"],
                                   "warm_worst": result["warm"]["first_image_seconds"]["worst"]},
            "settled_seconds": {"cold": result["cold"]["settled_seconds"], "warm_median": result["warm"]["settled_seconds"]["median"]},
            "abandon_ms": edit.get("abandon_ms"), "edit_to_first_look_ms": edit.get("edit_to_first_look_ms"),
            "idle_first_look_ms": edit.get("idle_first_look_ms"), "edit_trials": edit.get("trials"),
            "gate": edit.get("gate")}
    report = gpu3d.adapter_report()
    return {"schema": 1, "output": [WIDTH, HEIGHT], "adapter": report.splitlines()[0], "adapter_report": report,
            "backend": backend_name(), "first_look_target_seconds": P.FIRST_LOOK_SECONDS, "scenes": scenes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--scene", default="both", choices=("furnished", "mixed", "both"))
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--edits", type=int, default=10, help="edit-latency trials per kind (camera, material)")
    parser.add_argument("--no-edits", action="store_true")
    parser.add_argument("--field", default=None, metavar="PATH", help="write the small field-kit JSON here and nothing else")
    parser.add_argument("--quick", action="store_true", help="with --field: the furnished room only, fewer trials")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.adapter:
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    print(gpu3d.adapter_report())
    if args.field:
        global SETTLE
        SETTLE = not args.quick
        args.warm, args.no_interaction, args.no_edits = 1, True, False
        args.edits = 3 if args.quick else 5
        args.scene = "furnished" if args.quick else "both"
    results = []
    for name in (("furnished", "mixed") if args.scene == "both" else (args.scene,)):
        scene = furnished_scene() if name == "furnished" else soft_scene()
        P._COST.clear()
        result = measure(name, scene, args.warm, args.frames, not args.no_interaction, 0 if args.no_edits else args.edits)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("cold", "warm", "edit_latency")}, indent=1))
        print("cold:", [(f["seconds"], f["samples"], "preview" if f["preview"] else "pass") for f in result["cold"]["frames"]])
        print("warm median first image:", result["warm"]["first_image_seconds"], "settled:", result["warm"]["settled_seconds"])
        edit = result.get("edit_latency")
        if edit:
            print("edit latency:", json.dumps({k: edit[k] for k in ("abandon_ms", "edit_to_first_look_ms", "idle_first_look_ms",
                                                                    "abandoned_trials", "trials", "gate")}))
    if args.field:
        Path(args.field).write_text(json.dumps(field_record(results), indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps({"adapter": gpu3d.adapter_report(), "results": results}, indent=1))


if __name__ == "__main__":
    main()
