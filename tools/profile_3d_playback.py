"""Where does a played 3D frame spend its time? A stage timeline plus a sampling profile of both threads.

Plays one scene of tools/benchmark_3d_playback.py in the real Window with ``nodebased.playback_trace`` on, and
writes ``<out>.json`` (stage table, top costs, sampled stacks) and ``<out>.txt`` (the same, readable, plus cProfile
of the worker). The sampler reads ``sys._current_frames()`` every 4 ms for the Qt thread and the worker
(``py-spy record --native`` is used when it is installed and ``--py-spy`` is given; otherwise this one runs: it sees the
Python line a thread sits in, which for a blocking native call is the line that called it).

    DISPLAY=:94 QT_QPA_PLATFORM=xcb flock /tmp/nb-gpu.lock python tools/profile_3d_playback.py \\
        --scene cube --out benchmarks/3d-playback/profile-1022-cube
"""
import argparse
import cProfile
import collections
import io
import json
import os
import pstats
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# (stage name, from label, to label), in the order a frame meets them
STAGES = (
    ("scene evaluation", "worker_start", "render_begin"),
    ("scene prepare (shadow data, vertex packing)", "render_begin", "scene_ready"),
    ("scene upload (buffers, textures, pipelines, bind groups)", "scene_ready", "upload_done"),
    ("render encode", "upload_done", "submit"),
    ("GPU render and wait", "submit", "gpu_done"),
    ("readback copy", "gpu_done", "readback_done"),
    ("resolve (supersample, background)", "readback_done", "resolve_done"),
    ("after the render (node bookkeeping)", "resolve_done", "put_begin"),
    ("display cache put", "put_begin", "cache_put"),
    ("convert and emit", "cache_put", "emit"),
    ("hand-over to the display", "emit", "handover"),
)


def stage_table(events):
    """Split the marks into frames at each worker_start; per frame, the time between consecutive stage marks."""
    frames, current = [], None
    for stamp, thread, label in events:
        if label == "worker_start":
            current = {}
            frames.append(current)
        if current is None:
            continue
        if label in ("submit", "gpu_done", "readback_done"):
            current.setdefault(label + "_first", stamp)
            current[label] = stamp
        else:
            current.setdefault(label, stamp)
    rows = collections.defaultdict(list)
    for f in frames:
        if "handover" not in f or "render_begin" not in f:
            continue
        marks = [("worker_start", f["worker_start"]), ("render_begin", f["render_begin"]),
                 ("scene_ready", f.get("scene_ready")), ("upload_done", f.get("upload_done")),
                 ("submit", f.get("submit_first")), ("gpu_done", f.get("gpu_done")),
                 ("readback_done", f.get("readback_done")), ("resolve_done", f.get("resolve_done")),
                 ("put_begin", f.get("put_begin")), ("cache_put", f.get("cache_put")), ("emit", f.get("emit")), ("handover", f.get("handover"))]
        marks = [(n, t) for n, t in marks if t is not None]
        for (a, ta), (b, tb) in zip(marks, marks[1:]):
            rows[(a, b)].append((tb - ta) * 1000)
        rows[("total", "")].append((marks[-1][1] - marks[0][1]) * 1000)
    names = {(a, b): name for name, a, b in STAGES}
    table = []
    for (a, b), values in rows.items():
        name = names.get((a, b), f"{a} to {b}") if a != "total" else "worker start to hand-over (whole frame)"
        table.append({"stage": name, "from": a, "to": b, "n": len(values), "p50_ms": round(statistics.median(values), 2),
                      "p95_ms": round(sorted(values)[min(len(values) - 1, int(0.95 * len(values)))], 2),
                      "mean_ms": round(statistics.fmean(values), 2)})
    tick_gaps = [(b[0] - a[0]) * 1000 for a, b in zip([e for e in events if e[2] == "tick"], [e for e in events if e[2] == "tick"][1:])]
    return table, len(frames), tick_gaps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="cube")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    p.add_argument("--adapter", default="default")
    p.add_argument("--splats", type=int, default=20000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.adapter != "default":
        os.environ["NODEBASED_ADAPTER"] = args.adapter
    import tests.isolation  # noqa: F401
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from nodebased import playback_trace, gpu3d
    from nodebased.app import Window
    import benchmark_3d_playback as bench
    width, height = (int(v) for v in args.size.lower().split("x"))
    import tempfile
    tmp = tempfile.TemporaryDirectory(prefix="nb-prof-")
    splat_path = bench.make_splat_file(args.splats, tmp.name) if args.scene == "mixed" else None
    window = Window(agent_name="prof3d-" + uuid.uuid4().hex)
    window.resize(1800, 1100)
    window.move(0, 0)
    window.show()

    def spin(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.001)
    spin(0.5)
    ops, view = bench.build_scene(args.scene, width, height, args.backend, splat_path=splat_path)
    window.agent_command({"op": "batch", "commands": ops})
    window.set_time(first=1, last=240, current=1, fps=24.0)
    window.playback_proxy.setChecked(False)
    window.agent_command({"op": "view", "id": view})
    t0 = time.perf_counter()
    while (window.busy or len(window.preview_queue)) and time.perf_counter() - t0 < 180:
        spin(0.05)
    first_frame_s = round(time.perf_counter() - t0, 2)

    # cProfile of the worker: one Profile object, switched on and off around each job
    worker_profile = cProfile.Profile()
    submit = window.executor.submit

    def profiled_submit(fn, *a, **k):
        def run():
            worker_profile.enable()
            try:
                return fn(*a, **k)
            finally:
                worker_profile.disable()
        return submit(run)
    window.executor.submit = profiled_submit

    main_id = threading.get_ident()
    samples = {"qt": collections.Counter(), "worker": collections.Counter()}
    leaf = {"qt": collections.Counter(), "worker": collections.Counter()}
    counts = collections.Counter()
    stop = threading.Event()

    def frames_of(frame):
        out = []
        while frame is not None and len(out) < 12:
            code = frame.f_code
            out.append(f"{os.path.basename(code.co_filename)}:{code.co_name}:{frame.f_lineno}")
            frame = frame.f_back
        return out

    def sampler():
        while not stop.is_set():
            time.sleep(0.004)
            for ident, frame in list(sys._current_frames().items()):
                if ident == threading.get_ident():
                    continue
                stack = frames_of(frame)
                if ident == main_id:
                    kind = "qt"
                elif any("app.py:render" in s or "app.py:work" in s for s in stack):
                    kind = "worker"
                else:
                    continue
                counts[kind] += 1
                leaf[kind][stack[0]] += 1
                for entry in {s.rsplit(":", 1)[0] for s in stack}:      # inclusive, by function, once per sample
                    samples[kind][entry] += 1
    thread = threading.Thread(target=sampler, daemon=True)
    playback_trace.enable()
    window.playback_shown_times.clear()
    thread.start()
    window.toggle_playback(True)
    start = time.monotonic()
    while time.monotonic() - start < args.seconds:
        spin(0.1)
    elapsed = time.monotonic() - start
    window.toggle_playback(False)
    stop.set()
    events = playback_trace.disable()
    shown = [t for t in window.playback_shown_times if start <= t <= start + elapsed]
    table, frames, tick_gaps = stage_table(events)

    def top(counter, n=14):
        return [{"where": k, "samples": v} for k, v in counter.most_common(n)]
    try:
        adapter = gpu3d.chosen_adapter_name() if gpu3d.available() else "none (CPU rasteriser)"
    except Exception as exc:
        adapter = f"error: {exc}"
    start_marks = [i for i, e in enumerate(events) if e[2] == "worker_start"]
    one_frame = []
    if len(start_marks) > 6:           # the raw marks of one steady-state frame, relative to its worker_start
        first, last = start_marks[5], start_marks[6]
        one_frame = [[round((e[0] - events[first][0]) * 1000, 2), e[1], e[2]] for e in events[first:last]]
    result = {
        "one_frame_marks": one_frame,
        "scene": args.scene, "size": [width, height], "backend": args.backend, "adapter": adapter,
        "seconds": round(elapsed, 2), "first_frame_s": first_frame_s, "frames_shown": len(shown),
        "fps": round(len(shown) / elapsed, 2), "worker_jobs": frames,
        "stages": sorted(table, key=lambda r: -r["p50_ms"]),
        "tick_gap_ms_p50": round(statistics.median(tick_gaps), 1) if tick_gaps else None,
        "sampling": {"qt_samples": counts["qt"], "worker_samples": counts["worker"],
                     "qt_top_inclusive": top(samples["qt"]), "worker_top_inclusive": top(samples["worker"]),
                     "qt_top_lines": top(leaf["qt"], 8), "worker_top_lines": top(leaf["worker"], 12)},
        "status": window.statusBar().currentMessage(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    buffer = io.StringIO()
    buffer.write(f"{args.scene} at {width}x{height}, {adapter}, {result['fps']} fps shown, {frames} worker jobs in {elapsed:.1f} s\n")
    buffer.write(f"status: {result['status']}\n\nstage timeline (worker thread, per frame, milliseconds)\n")
    for row in result["stages"]:
        buffer.write(f"  {row['p50_ms']:9.2f} p50  {row['p95_ms']:9.2f} p95  {row['stage']}\n")
    for kind in ("qt", "worker"):
        buffer.write(f"\n{kind} thread, sampled every 4 ms ({counts[kind]} samples), most time in (inclusive, by function)\n")
        for row in top(samples[kind], 16):
            buffer.write(f"  {row['samples']:6d}  {row['where']}\n")
        buffer.write(f"{kind} thread, innermost line\n")
        for row in top(leaf[kind], 10):
            buffer.write(f"  {row['samples']:6d}  {row['where']}\n")
    buffer.write("\ncProfile of the worker, by cumulative time\n")
    pstats.Stats(worker_profile, stream=buffer).strip_dirs().sort_stats("cumulative").print_stats(30)
    buffer.write("\ncProfile of the worker, by time inside each function\n")
    pstats.Stats(worker_profile, stream=buffer).strip_dirs().sort_stats("tottime").print_stats(22)
    out.with_suffix(".txt").write_text(buffer.getvalue())
    print(buffer.getvalue()[:3500], flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
