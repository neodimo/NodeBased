"""VP1 bench: the default Viewer picture path against the opt-in retained-texture widget.

Run from the repo root on a real or private X display (xcb), for example
    DISPLAY=:61 QT_QPA_PLATFORM=xcb NODEBASED_DIRECT_ADAPTER=integrated \
        python tools/viewer_direct_bench.py --out benchmarks/viewer/direct-vp1-amd.json

Both paths show the same 1080p changing checker in a window of the same physical size. "Present return" is
the time the application call returns (the widget's wgpu present, or Qt's synchronous viewport paint); it
is not screen response and says nothing about scanout, compositor delay or tearing. Input response is the
time the wheel handler takes to return plus the time to the next present return after it.
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
args = p.parse_args()

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
