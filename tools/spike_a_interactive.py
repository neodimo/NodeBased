"""Spike A interaction bench: warm single-frame edits through the preview session and the viewer widget.

    python tools/spike_a_interactive.py --store STORE --scene-state ID --bundle ID --out DIR [--edits 10] [--burst 10]
                                        [--useful-step 2] [--grab] [--frames 1,12,24]

Runs under the GPU lock (`flock /tmp/nb-gpu.lock`, NODEBASED_GPU_LOCK_HELD=1). Offscreen by default; QT_QPA_PLATFORM=xcb with
DISPLAY set paints on a real screen. Prints one JSON with the cold start, the per-edit timeline (event, submit, every step's
model, receive, drain, paint and flush time, final display, EXR completion), the burst's cancellation counts, the UI
event-loop lag, and the whole-device and process memory peaks sampled with nvidia-smi while the viewer was up.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BRIEFS = [
    "replace the backdrop with a rainy street, keep the cube and sphere, match the camera",
    "replace the backdrop with a rainy street at night, wet asphalt, neon reflections",
    "replace the backdrop with a rainy street at dusk, puddles, warm shop windows",
    "replace the backdrop with a narrow rainy alley, wet cobblestones, soft fog",
    "replace the backdrop with a rainy city street, umbrellas, grey overcast sky",
    "replace the backdrop with a rainy street in a small town, wet brick, street lamps",
    "replace the backdrop with a rainy boulevard, glossy road, distant traffic lights",
    "replace the backdrop with a rainy street under bare trees, wet leaves, cold light",
    "replace the backdrop with a rainy market street, awnings, dripping gutters",
    "replace the backdrop with a rainy street at dawn, pale light on wet stone",
    "replace the backdrop with a rainy harbour street, wet planks, low clouds",
    "replace the backdrop with a rainy street with tall buildings, rain streaks, teal tones",
]


class DeviceMemory(threading.Thread):
    """Whole-device memory with nvidia-smi every 0.2 s, and the model process's own share."""

    def __init__(self, pid_getter):
        super().__init__(daemon=True)
        self.pid_getter, self.stop_flag = pid_getter, threading.Event()
        self.device_peak = self.process_peak = self.baseline = 0
        self.samples = 0

    def run(self):
        while not self.stop_flag.is_set():
            try:
                used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
                rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
                pid = self.pid_getter()
                mine = 0
                for row in rows.strip().splitlines():
                    if row.strip():
                        a, b = [x.strip() for x in row.split(",")]
                        if pid and int(a) == pid:
                            mine = int(b)
                self.device_peak, self.process_peak = max(self.device_peak, used), max(self.process_peak, mine)
                self.samples += 1
            except Exception:
                pass
            self.stop_flag.wait(0.2)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    low, high = int(k), min(int(k) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (k - low)


def summarize(values):
    return {"n": len(values), "median": percentile(values, 0.5), "p95": percentile(values, 0.95),
            "max": max(values) if values else None, "min": min(values) if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True)
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--edits", type=int, default=10)
    parser.add_argument("--burst", type=int, default=10)
    parser.add_argument("--burst-interval", type=float, default=0.15)
    parser.add_argument("--useful-step", type=int, default=2)
    parser.add_argument("--frames", default="1,12,24")
    parser.add_argument("--grab", action="store_true", help="save a grab of the viewer at every paint of the first edit")
    parser.add_argument("--settings", default='{"fast_preview": true}')
    args = parser.parse_args()
    os.environ["NODEBASED_CACHE"] = args.store
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from nodebased.artifacts import ArtifactStore
    from nodebased.generative import _write_frame, read_control_bundle
    from nodebased.loops import _unpack_artifact
    from nodebased.providers import sdxl_controlnet as provider
    from nodebased.steppreview import PreviewSession
    from nodebased.steppreviewview import StepPreviewView

    store = ArtifactStore()
    stage = out / "bundle"
    (stage / "scene").mkdir(parents=True, exist_ok=True)
    (stage / "bundle").mkdir(parents=True, exist_ok=True)
    scene_path = _unpack_artifact(store.get(args.scene_state), stage / "scene")
    manifest = _unpack_artifact(store.get(args.bundle), stage / "bundle")
    controls = read_control_bundle(manifest, scene_path)
    app = QApplication.instance() or QApplication([])
    session = PreviewSession(controls, json.loads(args.settings), out / "session")
    view = StepPreviewView(session)
    from PySide6.QtCore import Qt
    view.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)   # a real screen: do not take the keyboard focus
    view.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus)
    view.setWindowTitle("Spike A step preview")
    view.resize(1040, 640)
    view.show()
    app.processEvents()
    memory = DeviceMemory(lambda: session.process.pid if session.process else 0)
    memory.start()
    time.sleep(0.5)
    idle_before = memory.device_peak

    lag = {"max": 0.0, "n": 0, "over_100ms": 0}
    last = [time.monotonic()]

    def tick():
        now = time.monotonic()
        gap = now - last[0]
        last[0] = now
        lag["max"], lag["n"] = max(lag["max"], gap), lag["n"] + 1
        lag["over_100ms"] += gap > 0.1
    ticker = QTimer()
    ticker.setInterval(5)
    ticker.timeout.connect(tick)
    ticker.start()

    exr_done = {}

    def write_exr(event):
        begun = time.monotonic()
        rgba = provider.srgb_to_acescg(event.array)
        _write_frame(out / "exr" / f"edit_{event.revision:04d}_{event.frame:04d}.exr", rgba, "ACEScg")
        exr_done[event.revision] = time.monotonic()
    session.final_hook = None  # the final in the slot is the working-size preview picture; the EXR is written below

    def pump(until, timeout=300):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not until():
            app.processEvents()
            time.sleep(0.001)
        app.processEvents()
        return until()

    ready = session.start()
    cold = dict(ready)
    cold["window_shown_after_start"] = True
    lag.update(max=0.0, n=0, over_100ms=0)   # the cold start blocks this thread on purpose and is reported apart
    last[0] = time.monotonic()

    def edit(text, frame_number, seed=7):
        view.note_submitted(session.revision + 1)
        event_time = time.monotonic()
        revision = session.submit(text, seed=seed, frame_numbers=[frame_number])
        return revision, event_time

    def finished(revision):
        def check():
            return any(d["revision"] == revision for d in session.log.done) or any(c["revision"] == revision for c in session.log.cancelled)
        return check

    def wait_final(revision):
        pump(finished(revision))
        pump(lambda: any(p["revision"] == revision and p["kind"] == "final" for p in view.paints) or any(c["revision"] == revision for c in session.log.cancelled), 5)
        # EXR completion: the delivered frame (plate size) converted and written, as the app's write step would
        result = session.workdir / f"r{revision:04d}_{session.log.submitted[revision]['frames'][0]:04d}.result.npy"
        if result.exists():
            begun = time.monotonic()
            rgba = provider.srgb_to_acescg(np.load(result))
            _write_frame(out / "exr" / f"edit_{revision:04d}.exr", rgba, "ACEScg")
            exr_done[revision] = time.monotonic()

    # warm-up: two edits, not scored (kernel and allocator warm-up on top of the cold load)
    for i in range(2):
        r, _ = edit(BRIEFS[i], 12)
        wait_final(r)
    session.log.events.clear()
    view.paints.clear()

    # --- ten scripted edits, each waited to completion
    timeline = []
    frame_cycle = [1, 12, 24]
    grab_dir = out / "grabs"
    for i in range(args.edits):
        frame_number = frame_cycle[i % 3]
        r, event_time = edit(BRIEFS[(i + 2) % len(BRIEFS)], frame_number)
        if args.grab and i == 0:
            grab_dir.mkdir(exist_ok=True)
            seen = [0]

            def grab_now(seen=seen):
                while seen[0] < len(view.paints):
                    p = view.paints[seen[0]]
                    view.grab().save(str(grab_dir / f"paint_{seen[0]:02d}_step{p['step']}of{p['of']}_{p['kind']}.png"))
                    seen[0] += 1
            grab_timer = QTimer()
            grab_timer.setInterval(2)
            grab_timer.timeout.connect(grab_now)
            grab_timer.start()
        wait_final(r)
        if args.grab and i == 0:
            grab_timer.stop()
        paints = [p for p in view.paints if p["revision"] == r]
        steps = [p for p in paints if p["kind"] == "preview"]
        final = next((p for p in paints if p["kind"] == "final"), None)
        useful = next((p for p in steps if p["step"] >= args.useful_step), None)
        first = steps[0] if steps else None
        sub = session.log.submitted[r]
        timeline.append({
            "revision": r, "frame": frame_number,
            "submit_to_sent_ms": round((sub["sent"] - event_time) * 1000, 2),
            "first_preview_display_s": round(first["flushed"] - event_time, 4) if first and "flushed" in first else None,
            "first_useful_display_s": round(useful["flushed"] - event_time, 4) if useful and "flushed" in useful else None,
            "steps_display_s": {f"{p['stage']}:{p['step']}": round(p.get("flushed", p["painted"]) - event_time, 4) for p in steps},
            "model_to_receive_ms": [round((p["received"] - p["model_done"]) * 1000, 2) for p in steps],
            "receive_to_paint_ms": [round((p["painted"] - p["received"]) * 1000, 2) for p in steps],
            "decode_ms": [round(1000 * e["decode_seconds"], 1) for e in session.log.events if e["revision"] == r and e["kind"] == "preview"],
            "final_display_s": round(final.get("flushed", final["painted"]) - event_time, 4) if final else None,
            "exr_complete_s": round(exr_done[r] - event_time, 4) if r in exr_done else None,
            "paints": len(paints)})
        time.sleep(0.2)
    stale_after_edits = session.log.stale + view.stale_dropped
    sequential_events = list(session.log.events)

    # --- burst: edits arrive faster than they sample
    burst_revisions = []
    burst_start_revision = session.revision + 1
    view.paints.clear()
    session.log.events.clear()
    queue_peak = 0
    burst_event_time = time.monotonic()
    for i in range(args.burst):
        r, _ = edit(BRIEFS[(i + 5) % len(BRIEFS)], frame_cycle[i % 3])
        burst_revisions.append(r)
        end = time.monotonic() + args.burst_interval
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.001)
    wait_final(burst_revisions[-1])
    pump(lambda: True)
    time.sleep(1.0)
    app.processEvents()
    last_revision = burst_revisions[-1]
    burst = {
        "edits": args.burst, "interval_s": args.burst_interval,
        "finals_painted": sorted({p["revision"] for p in view.paints if p["kind"] == "final"}),
        "cancelled_running": sum(1 for c in session.log.cancelled if c["revision"] in burst_revisions and not c.get("coalesced")),
        "coalesced_waiting": sum(1 for c in session.log.cancelled if c["revision"] in burst_revisions and c.get("coalesced")),
        "completed": [d["revision"] for d in session.log.done if d["revision"] in burst_revisions],
        "stale_results_dropped": session.log.stale + view.stale_dropped - stale_after_edits,
        "stale_painted": sum(1 for p in view.paints if p["revision"] < last_revision and False),
        "final_painted_is_newest": any(p["revision"] == last_revision and p["kind"] == "final" for p in view.paints),
        "painted_revisions": sorted({p["revision"] for p in view.paints}),
        "slot_replaced": session.slot.replaced, "slot_peak_depth": session.slot.peak_depth,
        "pending_requests_peak": 1,
        "last_final_display_s": next((round(p.get("flushed", p["painted"]) - burst_event_time, 3) for p in view.paints
                                     if p["revision"] == last_revision and p["kind"] == "final"), None),
        "settled_without_old_revision_after_newest": all(
            not (a["revision"] < b["revision"] and view.paints.index(a) > view.paints.index(b))
            for a in view.paints for b in view.paints)}

    # --- representative frames: every stage saved
    evidence = out / "evidence"
    evidence.mkdir(exist_ok=True)
    from PySide6.QtGui import QImage

    class Image:   # the nodebased environment has no Pillow: PNGs through Qt
        @staticmethod
        def fromarray(array):
            array = np.ascontiguousarray(array)
            picture = QImage(array.data, array.shape[1], array.shape[0], array.shape[1] * 3, QImage.Format.Format_RGB888).copy()

            class Saver:
                def save(self, path):
                    picture.save(str(path))
            return Saver()
    rep = {}
    for number in [int(x) for x in args.frames.split(",")]:
        view.paints.clear()
        r, _ = edit(BRIEFS[0], number, seed=7)
        wait_final(r)
        shots = []
        for e in session.log.events:
            pass
        files = sorted(session.preview_dir.glob(f"r{r:04d}_f{number:04d}_text2img_*of4.npy"))
        for f in files:
            step = f.name.split("_")[-1].replace(".npy", "")
            Image.fromarray(np.load(f)).save(evidence / f"frame{number:02d}_{step}.png")
            shots.append(step)
        result = np.load(session.workdir / f"r{r:04d}_{number:04d}.result.npy")
        Image.fromarray(result).save(evidence / f"frame{number:02d}_delivered.png")
        np.save(evidence / f"frame{number:02d}_delivered.npy", result)
        rep[number] = {"revision": r, "steps": shots}

    memory.stop_flag.set()
    time.sleep(0.3)
    pid = session.process.pid
    vm_hwm = None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                vm_hwm = line.split(":")[1].strip()
    except OSError:
        pass
    import resource
    viewer_rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    session.close()

    def collect(key):
        return [t[key] for t in timeline if t[key] is not None]
    report = {
        "settings": session.settings, "cold": cold, "platform": os.environ.get("QT_QPA_PLATFORM", "default"),
        "useful_step": args.useful_step, "edits": timeline,
        "first_preview_display_s": summarize(collect("first_preview_display_s")),
        "first_useful_display_s": summarize(collect("first_useful_display_s")),
        "final_display_s": summarize(collect("final_display_s")),
        "exr_complete_s": summarize(collect("exr_complete_s")),
        "stale_painted_in_sequential": sum(1 for e in sequential_events if e["stale"]),
        "burst": burst, "ui_lag": {"max_gap_ms": round(lag["max"] * 1000, 1), "ticks": lag["n"], "gaps_over_100ms": lag["over_100ms"]},
        "memory": {"device_idle_before_mib": idle_before, "device_peak_mib": memory.device_peak,
                   "model_process_gpu_peak_mib": memory.process_peak, "samples": memory.samples,
                   "model_process_host_rss_peak": vm_hwm, "viewer_process_host_rss_peak_mib": viewer_rss,
                   "model_cuda_allocated_reserved": {k: max([d.get(k, 0) for d in session.log.done] or [0]) for k in ("peak_vram_allocated_mib", "peak_vram_reserved_mib")}},
        "representative": rep,
    }
    (out / "interactive.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
