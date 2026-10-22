"""VP1 bench: the default Viewer picture path against the opt-in retained-texture widget.

Run from the repo root on a real or private X display (xcb), for example
    DISPLAY=:61 QT_QPA_PLATFORM=xcb NODEBASED_DIRECT_ADAPTER=integrated \
        python tools/viewer_direct_bench.py --out benchmarks/viewer/direct-vp1-amd.json

Both paths show the same 1080p changing checker in a window of the same physical size. "Present return" is
the time the application call returns (the widget's wgpu present, or Qt's synchronous viewport paint); it
is not screen response and says nothing about scanout, compositor delay or tearing. Input response is the
time the wheel handler takes to return plus the time to the next present return after it.

`--paused` is the cost gate for the paused picture (docs/DIRECT_VIEWER.md, "Paused picture and compare on the
card"): the real Window on a 1080p frame, no playback, an edit every time (a Grade knob moves), and for each
path the Qt-thread time (`time.thread_time()` of the main thread from the edit to the settled picture) and
the process CPU (every thread) per edit:

    viewer    the Viewer alone, dock closed: `to_qimage` on the worker, pixmap built on the Qt thread
    dock-8bit the dock open with the pre-10/22 hand-off (8-bit picture built for the Viewer and offered to the card)
    direct    the dock open, the float frame to the card; the Viewer takes only the geometry

each with and without an A/B wipe. Gate (RTX 3080 Ti): direct spends at least 30 percent less median Qt-thread
time per edit than the Viewer, with and without B. `--adapter integrated|cpu` forces the app's adapter like
force-adapter.py does; run under `taskset -c 0-3` with `QT_SCALE_FACTOR=2` for the laptop stand-in.

    DISPLAY=:77 QT_QPA_PLATFORM=xcb NODEBASED_DIRECT_ADAPTER=discrete \
        python tools/viewer_direct_bench.py --paused --out benchmarks/viewer/paused-rtx.json
"""
import argparse
import json
import os
import statistics
import sys
import time
import uuid

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--frames", type=int, default=240)
p.add_argument("--inputs", type=int, default=240)
p.add_argument("--only", choices=["default", "direct", "both"], default="both")
p.add_argument("--size", default="1920x1080", help="WxH of the picture; the direct widget and the Viewer's viewport are made this size")
p.add_argument("--paused", action="store_true", help="the paused-edit cost gate (see the module docstring)")
p.add_argument("--edits", type=int, default=40, help="--paused: timed edits per path")
p.add_argument("--adapter", choices=("default", "discrete", "integrated", "cpu"), default="default",
               help="--paused: force the app's adapter (and the dock's) as force-adapter.py does")
p.add_argument("--profile", default="", help="--paused: write a cProfile of the Qt thread over the 'direct' rows to this text file")
p.add_argument("--min-saving", type=float, default=0.30, help="--paused: required fall in median Qt-thread time")
args = p.parse_args()
if args.paused and args.adapter != "default":
    os.environ["NODEBASED_DIRECT_ADAPTER"] = args.adapter

import tests.isolation  # noqa: F401  (Qt settings stay out of the real user file)
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, Qt

app = QtWidgets.QApplication([])
W, H = (int(v) for v in args.size.lower().split("x"))


def images(count=8):
    out = []
    y, x = np.indices((H, W))
    for k in range(count):
        a = np.empty((H, W, 3), np.uint8)
        a[:] = np.where(((x // 64 + y // 64 + k) % 2)[..., None], 180, 50)
        out.append(QtGui.QImage(a.data, W, H, W * 3, QtGui.QImage.Format.Format_RGB888).copy())
    return out


def dist(values):
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]
    return {"p50": pick(50), "p95": pick(95), "p99": pick(99), "max": v[-1], "mean": statistics.fmean(v)}


def wheel(widget, steps, pos):
    point = QPointF(pos[0], pos[1])
    event = QtGui.QWheelEvent(point, widget.mapToGlobal(point), QtCore.QPoint(0, 0), QtCore.QPoint(0, int(120 * steps)),
                              Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    return event


def measure_default(imgs):
    from nodebased.app import Window
    window = Window(agent_name="bench-" + uuid.uuid4().hex)
    window.show()
    window.resize(W + 400, H + 400)
    for _ in range(50):
        app.processEvents()
        time.sleep(0.01)
    view = window.viewer
    # Same output dimensions as the direct widget: the Viewer (still fed by Window._show_image) becomes a
    # top-level window whose viewport is exactly W x H physical pixels.
    view.setParent(None)
    view.show()
    for _ in range(6):
        for _ in range(30):
            app.processEvents()
            time.sleep(0.01)
        dw = view.viewport().width() - round(W / view.devicePixelRatioF())
        dh = view.viewport().height() - round(H / view.devicePixelRatioF())
        if dw == 0 and dh == 0:
            break
        view.resize(view.width() - dw, view.height() - dh)
    cpu0, t0 = time.process_time(), time.perf_counter()
    show_ms, paint_ms, stamps = [], [], []
    for i in range(args.frames):
        t = time.perf_counter()
        window._show_image(imgs[i % len(imgs)], 1, None)
        show_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        view.viewport().repaint()
        paint_ms.append((time.perf_counter() - t) * 1000)
        stamps.append(time.perf_counter())
        app.processEvents()
    frame_cpu = time.process_time() - cpu0
    frame_wall = time.perf_counter() - t0
    handler_ms, response_ms = [], []
    cpu0 = time.process_time()
    center = view.viewport().rect().center()
    for i in range(args.inputs):
        t = time.perf_counter()
        QtWidgets.QApplication.sendEvent(view.viewport(), wheel(view.viewport(), 1 if i % 2 == 0 else -1, (center.x(), center.y())))
        handler_ms.append((time.perf_counter() - t) * 1000)
        view.viewport().repaint()
        response_ms.append((time.perf_counter() - t) * 1000)
        app.processEvents()
    input_cpu = time.process_time() - cpu0
    result = {
        "path": "default Viewer (QGraphicsView + QPixmap.fromImage), Window._show_image",
        "new_frame": {"show_image_ms": dist(show_ms), "viewport_repaint_return_ms": dist(paint_ms),
                      "interval_ms": dist([(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]),
                      "frames_per_second": args.frames / frame_wall, "process_cpu_fraction": frame_cpu / frame_wall,
                      "to_qimage_runs_per_frame": 1, "pixmap_builds_per_frame": 1, "gpu_uploads_per_frame": 0},
        "pan_zoom": {"handler_ms": dist(handler_ms), "handler_plus_repaint_ms": dist(response_ms),
                     "process_cpu_ms_per_input": input_cpu / args.inputs * 1000},
        "viewport_physical": [round(view.viewport().width() * view.devicePixelRatioF()), round(view.viewport().height() * view.devicePixelRatioF())],
    }
    window.saved_document = window.dispatcher.document
    view.close()
    window.close()
    return result


def measure_direct(imgs):
    from nodebased import directviewer as dv
    widget = dv.make_widget()
    widget.show()
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    dpr = widget.get_pixel_ratio()      # only meaningful once the window is mapped
    widget.resize(round(W / dpr), round(H / dpr))
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    context = widget.get_wgpu_context()
    stamps, present_ms = [], []
    original = context._rc_present

    def present(**kwargs):
        t = time.perf_counter()
        result = original(**kwargs)
        present_ms.append((time.perf_counter() - t) * 1000)
        stamps.append(time.perf_counter())
        return result

    context._rc_present = present
    rect = (0, 0, W, H)
    widget.submit(imgs[0], rect, generation=1)
    deadline = time.time() + 5
    while not stamps and time.time() < deadline:
        app.processEvents()
    if widget.error:
        raise RuntimeError(widget.error)
    presenter = widget.presenter
    info = dict(presenter.adapter_info)
    from collections import Counter

    def wait_present(before, timeout=1.0):
        end = time.perf_counter() + timeout
        while len(stamps) <= before and time.perf_counter() < end:
            app.processEvents()

    for i in range(10):
        widget.submit(imgs[i % len(imgs)], rect, generation=1)
        wait_present(len(stamps))
    presenter.counters.clear()
    stamps.clear()
    present_ms.clear()
    cpu0, t0 = time.process_time(), time.perf_counter()
    submit_ms = []
    for i in range(args.frames):
        before = len(stamps)
        t = time.perf_counter()
        widget.submit(imgs[i % len(imgs)], rect, generation=2)
        submit_ms.append((time.perf_counter() - t) * 1000)
        wait_present(before)
    frame_cpu = time.process_time() - cpu0
    frame_wall = time.perf_counter() - t0
    frame_counts = dict(presenter.counters)
    frame_stamps = list(stamps)
    frame_present = list(present_ms)
    presenter.counters.clear()
    stamps.clear()
    present_ms.clear()
    handler_ms, response_ms = [], []
    cpu0 = time.process_time()
    centre = (widget.width() / 2, widget.height() / 2)
    for i in range(args.inputs):
        before = len(stamps)
        t = time.perf_counter()
        QtWidgets.QApplication.sendEvent(widget, wheel(widget, 1 if i % 2 == 0 else -1, centre))
        handler_ms.append((time.perf_counter() - t) * 1000)
        wait_present(before)
        if len(stamps) > before:
            response_ms.append((stamps[before] - t) * 1000)
    input_cpu = time.process_time() - cpu0
    result = {
        "path": "retained texture, QRenderWidget(present_method='screen')",
        "adapter": {k: str(v) for k, v in info.items()},
        "device_epoch": presenter.ledger.epoch, "selection_note": presenter.selection_note,
        "selection": getattr(presenter, "selection", {}),
        "physical_size": list(widget.get_physical_size()), "pixel_ratio": dpr,
        "new_frame": {"submit_return_ms": dist(submit_ms), "present_return_ms": dist(frame_present),
                      "interval_ms": dist([(b - a) * 1000 for a, b in zip(frame_stamps, frame_stamps[1:])]),
                      "frames_per_second": args.frames / frame_wall, "process_cpu_fraction": frame_cpu / frame_wall,
                      "counters": frame_counts, "to_qimage_runs_per_frame": 1, "pixmap_builds_per_frame": 0},
        "pan_zoom": {"handler_ms": dist(handler_ms), "event_to_present_return_ms": dist(response_ms),
                     "process_cpu_ms_per_input": input_cpu / args.inputs * 1000, "counters": dict(presenter.counters)},
    }
    widget.close()
    return result


def measure_paused():
    """Qt-thread time and process CPU per paused edit: Viewer, dock with the old hand-off, direct picture."""
    from nodebased.app import Window
    import nodebased.app as app_module
    from nodebased import gpu3d
    if args.adapter != "default":
        original = gpu3d._state
        gpu3d._state = lambda choice=None: original(args.adapter if (choice or "default") == "default" else choice)

    def spin(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.001)

    def wait(condition, limit=30.0, step_ms=4):
        """Block in the Qt event loop (no polling on this thread beyond a 4 ms timer) until `condition()`."""
        loop = QtCore.QEventLoop()
        timer = QtCore.QTimer()
        timer.setInterval(step_ms)
        deadline = time.perf_counter() + limit

        def check():
            if condition() or time.perf_counter() > deadline:
                loop.quit()
        timer.timeout.connect(check)
        timer.start()
        loop.exec()
        timer.stop()

    timings = {"preview_ready": 0.0, "present": 0.0, "to_qimage_calls": 0, "to_qimage_ms": 0.0}
    original_ready = Window.preview_ready
    original_qimage = app_module.to_qimage

    def timed_ready(self, *a, **k):
        t = time.perf_counter()
        try:
            return original_ready(self, *a, **k)
        finally:
            timings["preview_ready"] += (time.perf_counter() - t) * 1000

    def counted_qimage(*a, **k):
        t = time.perf_counter()
        try:
            return original_qimage(*a, **k)
        finally:
            timings["to_qimage_calls"] += 1
            timings["to_qimage_ms"] += (time.perf_counter() - t) * 1000
    Window.preview_ready = timed_ready
    app_module.to_qimage = counted_qimage

    def spy(owner, name, key):
        """Accumulate the Qt-thread CPU time of owner.name under timings[key] (and count its calls)."""
        real = getattr(owner, name)

        def wrapper(*a, **k):
            c = time.thread_time()
            try:
                return real(*a, **k)
            finally:
                timings[key] = timings.get(key, 0.0) + (time.thread_time() - c) * 1000
                timings[key + "_calls"] = timings.get(key + "_calls", 0) + 1
        setattr(owner, name, wrapper)
    spy(app_module.Window, "_show_image", "show_image")
    spy(app_module.Viewer, "paintEvent", "viewer_paint")
    spy(app_module.Viewer, "render_overlay", "overlay_render")
    spy(app_module.Window, "_show_direct_linear", "show_direct_linear")
    spy(app_module.Window, "_park_viewer", "park_viewer")
    spy(app_module.Window, "_frame_viewer_to", "frame_viewer_to")

    # Patched before the window exists: its signals connect to these.
    window = Window(agent_name="paused-bench-" + uuid.uuid4().hex)
    window.resize(1800, 1100)
    window.move(0, 0)
    window.show()
    spin(1.0)
    window.agent_command({"op": "batch", "commands": [
        {"op": "create", "type": "Constant", "id": "c", "pos": [0, 0],
         "params": {"width": W, "height": H, "red": 0.35, "green": 0.3, "blue": 0.25, "alpha": 1.0}},
        {"op": "create", "type": "Grade", "id": "g", "pos": [150, 0]},
        {"op": "connect", "id": "g", "input": "image", "source": "c"},
        {"op": "create", "type": "Constant", "id": "b", "pos": [300, 0],
         "params": {"width": W, "height": H, "red": 0.1, "green": 0.4, "blue": 0.7, "alpha": 1.0}},
        {"op": "view", "id": "g"}]})
    window.command({"op": "viewer_input", "slot": 2, "id": "b", "activate": False})

    def idle():
        return (not window.busy and not len(window.preview_queue) and window.frame_generation == window.generation
                and window._progressive_generation is None)

    wait(idle)
    spin(0.3)
    def edits(count, serial):
        """Run `count` timed edits; returns per-edit (thread ms, process ms) and the per-edit hand-off timings."""
        thread_ms, process_ms, ready_ms, draw_ms, qimage_calls, qimage_ms, present_ms = [], [], [], [], [], [], []
        breakdown = {}
        picture_ms = []
        for i in range(count):
            for key in list(timings):
                timings[key] = 0 if isinstance(timings[key], int) else 0.0
            serial[0] += 1
            value = 1.0 + 0.0013 * (serial[0] % 97)
            t0, c0, p0 = time.perf_counter(), time.thread_time(), time.process_time()
            window.command({"op": "set", "id": "g", "param": "multiply", "value": value})
            wait(lambda: window.generation > 0 and idle() and time.perf_counter() - t0 > 0.12)
            wait(lambda: time.perf_counter() - t0 > 0.25)        # let the paints and the card's draw land
            thread_ms.append((time.thread_time() - c0) * 1000)
            process_ms.append((time.process_time() - p0) * 1000)
            ready_ms.append(timings["preview_ready"])
            for key, value in timings.items():
                if key not in ("preview_ready", "present", "to_qimage_calls", "to_qimage_ms"):
                    breakdown.setdefault(key, []).append(value)
            draw_ms.append(timings.get("dock_draw", 0.0))
            picture_ms.append(sum(timings.get(k, 0.0) for k in ("show_image", "viewer_paint", "dock_draw", "dock_overlay")))
            present_ms.append(timings["present"])
            qimage_calls.append(timings["to_qimage_calls"])
            qimage_ms.append(timings["to_qimage_ms"])
        return thread_ms, process_ms, ready_ms, draw_ms, qimage_calls, qimage_ms, present_ms, breakdown, picture_ms

    def idle_baseline():
        """Qt-thread time of the same 0.25 s wait with no edit: what the harness itself costs per edit."""
        samples = []
        for _ in range(10):
            t0, c0 = time.perf_counter(), time.thread_time()
            wait(lambda: time.perf_counter() - t0 > 0.25)
            samples.append((time.thread_time() - c0) * 1000)
        return statistics.median(samples)

    state = {"serial": [0]}
    dock_info = {}

    def open_dock():
        window.direct_action.setChecked(True)
        spin(1.0)
        widget = window.direct_widget
        if widget is None or widget.error:
            raise RuntimeError(f"direct picture did not start: {getattr(widget, 'error', None)}")
        widget.parent().resize(1280, 760)
        widget.parent().move(1820, 0)
        spin(0.8)
        presenter = widget.presenter
        if presenter is not None:
            dock_info.update(adapter=dict(presenter.selection or {}, device=presenter.adapter_info.get("device", "")))
            spy(presenter, "draw", "dock_draw")
            spy(widget, "_paint_overlay", "dock_overlay")
            context = widget.get_wgpu_context()
            real_present = context._rc_present

            def timed_present(**kwargs):
                c = time.thread_time()
                try:
                    return real_present(**kwargs)
                finally:
                    timings["present"] += (time.thread_time() - c) * 1000
            context._rc_present = timed_present
        return widget

    def close_dock():
        window.direct_action.setChecked(False)
        spin(0.5)

    def row(label, with_b):
        window.command({"op": "viewer_compare", "b": 2, "mode": "wipe" if with_b else "A only"})
        spin(0.3)
        wait(idle)
        edits(6, state["serial"])                    # shader build, first uploads, cache warm
        (thread_ms, process_ms, ready_ms, draw_ms, calls, qimage_ms, present_ms, breakdown,
         picture_ms) = edits(args.edits, state["serial"])
        without_present = [t - p for t, p in zip(thread_ms, present_ms)]
        return {"path": label, "ab_wipe": with_b, "edits": args.edits,
                "qt_thread_ms_per_edit": dist(thread_ms), "process_cpu_ms_per_edit": dist(process_ms),
                "qt_thread_ms_per_edit_without_surface_present": dist(without_present),
                "picture_path_qt_thread_ms_per_edit": dist(picture_ms),
                "surface_present_qt_thread_ms_per_edit_mean": statistics.fmean(present_ms),
                "qt_thread_breakdown_per_edit_mean": {k: round(statistics.fmean(v), 2) for k, v in breakdown.items()},
                "preview_ready_ms_per_edit_mean": statistics.fmean(ready_ms),
                "dock_draw_ms_per_edit_mean": statistics.fmean(draw_ms),
                "to_qimage_runs_per_edit_mean": statistics.fmean(calls),
                "to_qimage_ms_per_edit_mean_worker_thread": statistics.fmean(qimage_ms)}

    rows = []
    for with_b in (False, True):
        rows.append(row("viewer", with_b))
    widget = open_dock()
    for with_b in (False, True):
        profiler = None
        if args.profile and not with_b:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        rows.append(row("direct", with_b))
        if profiler is not None:
            profiler.disable()
            import io
            import pstats
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).sort_stats("tottime").print_stats(35)
            with open(args.profile, "w") as fh:
                fh.write(text.getvalue())
        window._direct_linear_on = False                 # the hand-off before this step: 8 bits built, then offered
        rows.append(row("dock-8bit", with_b))
        window._direct_linear_on = True
    baseline = idle_baseline()
    gates = {}
    for with_b in (False, True):
        viewer = next(r for r in rows if r["path"] == "viewer" and r["ab_wipe"] == with_b)
        direct = next(r for r in rows if r["path"] == "direct" and r["ab_wipe"] == with_b)
        key = "qt_thread_ms_per_edit_without_surface_present"
        saving = 1.0 - direct[key]["p50"] / viewer[key]["p50"]
        saving_with = 1.0 - direct["qt_thread_ms_per_edit"]["p50"] / viewer["qt_thread_ms_per_edit"]["p50"]
        picture = 1.0 - (direct["picture_path_qt_thread_ms_per_edit"]["p50"]
                         / viewer["picture_path_qt_thread_ms_per_edit"]["p50"])
        gates["with_b" if with_b else "without_b"] = {
            "viewer_picture_path_p50_ms": viewer["picture_path_qt_thread_ms_per_edit"]["p50"],
            "direct_picture_path_p50_ms": direct["picture_path_qt_thread_ms_per_edit"]["p50"],
            "saving": round(picture, 3),
            "viewer_whole_edit_p50_ms": viewer["qt_thread_ms_per_edit"]["p50"],
            "direct_whole_edit_p50_ms": direct["qt_thread_ms_per_edit"]["p50"],
            "direct_whole_edit_p50_ms_without_surface_present": direct[key]["p50"],
            "saving_whole_edit_without_surface_present": round(saving, 3),
            "saving_whole_edit_with_surface_present": round(saving_with, 3),
            "required": args.min_saving, "passed": picture >= args.min_saving}
    result = {"mode": "paused edit, Qt-thread time and process CPU per edit", "frame": [W, H],
              "platform": app.platformName(), "qt": QtCore.qVersion(), "display": os.environ.get("DISPLAY", ""),
              "adapter_request": args.adapter, "adapter": dock_info.get("adapter", {}),
              "cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "qt_scale_factor": os.environ.get("QT_SCALE_FACTOR", ""), "device_pixel_ratio": window.devicePixelRatioF(),
              "idle_wait_qt_thread_ms_per_edit_window": round(baseline, 3), "rows": rows, "gate": gates,
              "gate_note": "the gate is on the Qt-thread time of putting the picture up (Window._show_image, the Viewer's "
                           "paint, the dock's draw and overlay paint), not the whole edit: the rest of an edit (command, "
                           "graph and panel refresh, status, timeline marks) is the same on every path and is reported as "
                           "the whole-edit figures. The wgpu surface present call is left out of every figure but the "
                           "with-present one: Xvfb has no DRI3, so presenting a Vulkan image to it copies the picture out "
                           "through the processor",
              "gate_passed": all(g["passed"] for g in gates.values())}
    window.saved_document = window.dispatcher.document
    return result


if args.paused:
    result = measure_paused()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"adapter": result["adapter"].get("device"), "gate": result["gate"]}, indent=1), flush=True)
    os._exit(0 if result["gate_passed"] else 1)

out = {"bounds": "present return is a CPU timestamp when the application call returns; it is not screen response",
       "platform": app.platformName(), "qt": QtCore.qVersion(), "frame": [W, H]}
imgs = images()
if args.only in ("default", "both"):
    out["default"] = measure_default(imgs)
if args.only in ("direct", "both"):
    out["direct"] = measure_direct(imgs)
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items()}, indent=1)[:3500])
