#!/usr/bin/env python3
"""Roto interaction latency: what an artist feels when clicking and dragging in the Viewer.

Opens the real main window on the real display (DISPLAY=:0, QT_QPA_PLATFORM=xcb) with a shot the tool builds itself, then
posts real Qt mouse and key events to the viewer widget through the event loop (QCoreApplication.postEvent, never by calling
a handler) and times each one:

    handler    the viewer's mouse or key handler, from entry to return
    overlay    the event's scheduled time to the end of the Viewer paintEvent that follows it (points, tangents, outline,
               brush cursor are drawn in that paint). A handler that blocks delays the next post, and the delay counts here,
               because the scheduled time is when the hand moved.
    image      for each picture the viewer shows during a drag: the age of the edit it shows, from the edit's scheduled time
               to the end of the paint that shows the picture. After the release, the time from the release to the
               full-resolution picture.

Per action: median and p95 of each, the renders started and dropped, the events posted and handled (the difference was
coalesced by Qt), and with --profile the top functions by cumulative time on the GUI thread (cProfile, a separate pass).

    DISPLAY=:0 QT_QPA_PLATFORM=xcb python tools/benchmark_roto_interaction.py --output roto-interaction.json
    python tools/benchmark_roto_interaction.py --self-check        # offscreen, tiny shot, proves the tool runs (no numbers kept)
    python tools/benchmark_roto_interaction.py --profile --cases light --actions drag_point

Run it on a quiet machine: `pgrep -af "integ-auto|unittest"` must find nothing. The shot is `--case realistic` (1080p textured
plate through Read, a Roto of 20 closed Bezier shapes of 12 points keyed on 6 of 48 frames and feathered, a RotoPaint of 30
strokes, a Merge and a Grade) or `--case light` (one Roto on a constant). The Roto overlay is only editable while the Roto
node itself is the viewed node, so the Roto actions view it, and the RotoPaint actions view the RotoPaint.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import os
import pathlib
import platform
import pstats
import statistics
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _early_platform():
    """The Qt platform is chosen before PySide6 loads: xcb on the real display, offscreen only for --self-check."""
    if "--self-check" in sys.argv:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        os.environ.setdefault("DISPLAY", ":0")


_early_platform()

import numpy as np  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from nodebased import playback_trace  # noqa: E402

MOVE_SCALE = 1.0                   # --short shrinks every drag (12 moves, 24 for a stroke) for the slow profiled pass
FRAME_INTERVAL = 1 / 60.0          # display rate: a move event per 16.7 ms
KEY_FRAMES = (1, 10, 20, 30, 40, 48)
PLATE_PATTERN = "plate.####.png"


# --------------------------------------------------------------------------------------------- the shot

def _scalar(value, deltas=None):
    """A static number, or one keyed on KEY_FRAMES at value + delta."""
    if deltas is None:
        return value
    return {"value": value, "curve": {"interpolation": "smooth",
                                       "keys": [{"frame": f, "value": value + d} for f, d in zip(KEY_FRAMES, deltas)]}}


def make_shape(index, cx, cy, radius, count=12, keyed=True, feather=8.0):
    """A closed Bezier blob of `count` points around (cx, cy); every coordinate keyed on KEY_FRAMES when `keyed`."""
    rng = np.random.default_rng(1000 + index)
    points = []
    for i in range(count):
        a = 2 * math.pi * i / count
        r = radius * (1.0 + 0.18 * math.sin(3 * a + index))
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        handle = 2 * math.pi * r / count / 3.0
        hx, hy = -handle * math.sin(a), handle * math.cos(a)
        wobble = rng.uniform(-30, 30, size=(2, len(KEY_FRAMES))) if keyed else None
        points.append({"x": _scalar(x, list(wobble[0]) if keyed else None), "y": _scalar(y, list(wobble[1]) if keyed else None),
                       "in_x": -hx, "in_y": -hy, "out_x": hx, "out_y": hy,
                       "feather_x": 0.0, "feather_y": 0.0})
    return {"name": f"shape{index + 1}", "mode": "union", "opacity": 0.9, "feather": feather, "closed": True,
            "feather_model": "edge", "points": points, "locked": False}


def shape_grid(count, width, height, keyed=True):
    columns = 5 if count > 4 else count
    rows = math.ceil(count / columns)
    cell_w, cell_h = width / columns, height / rows
    radius = min(cell_w, cell_h) * 0.32
    return [make_shape(i, (i % columns + 0.5) * cell_w, (i // columns + 0.5) * cell_h, radius, keyed=keyed)
            for i in range(count)]


def make_strokes(count, width, height):
    rng = np.random.default_rng(77)
    strokes = []
    for i in range(count):
        x0, y0 = rng.uniform(0.1 * width, 0.9 * width), rng.uniform(0.1 * height, 0.9 * height)
        points = [{"x": float(x0 + 18 * k + rng.uniform(-6, 6)), "y": float(y0 + 9 * k + rng.uniform(-6, 6)),
                   "pressure": float(rng.uniform(0.4, 1.0))} for k in range(40)]
        strokes.append({"kind": "stroke", "name": f"stroke{i + 1}", "points": points,
                        "brush": {"size": 22.0, "hardness": 0.2, "opacity": 1.0, "spacing": 0.05, "strength": 0.2},
                        "tool": "paint", "lifetime": {"mode": "all"}, "color": [0.9, 0.3, 0.1, 1.0],
                        "source_offset": [0.0, 0.0], "source_frame": "relative", "opacity": 1.0,
                        "blend": "over", "visible": True, "follow_track": None, "patch_blend": 0.0})
    return strokes


def write_plate(directory, width, height, frames):
    """A textured plate sequence the Read node loads: low-frequency colour, fine noise, a few hard edges."""
    from nodebased.imaging import write_png
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([0.5 + 0.4 * np.sin(xx / 97.0), 0.5 + 0.4 * np.cos(yy / 71.0), 0.5 + 0.3 * np.sin((xx + yy) / 150.0)], -1)
    noise = rng.normal(0, 0.04, size=(height, width, 1)).astype(np.float32)
    frame = np.concatenate([np.clip(base + noise, 0, 1), np.ones((height, width, 1), np.float32)], -1).astype(np.float32)
    frame[(xx.astype(int) // 80 + yy.astype(int) // 80) % 2 == 0, :3] *= 0.85
    import shutil
    frames = list(frames)
    first = pathlib.Path(directory) / f"plate.{frames[0]:04d}.png"
    write_png(str(first), frame)            # one real write; the other frames are copies (the Read still decodes each file)
    for number in frames[1:]:
        shutil.copyfile(first, pathlib.Path(directory) / f"plate.{number:04d}.png")
    return str(pathlib.Path(directory) / PLATE_PATTERN)


def build_document(case, scratch, width, height, last_frame):
    from nodebased.core import Dispatcher, empty_document
    dispatcher = Dispatcher(empty_document())
    commands = []
    if case == "light":
        commands += [{"op": "create", "id": "plate", "type": "Constant", "params": {"width": width, "height": height}},
                     {"op": "create", "id": "roto", "type": "Roto", "params": {"width": width, "height": height}}]
        shapes, strokes = [], []
        commands += [{"op": "create", "id": "paint", "type": "RotoPaint"},
                     {"op": "connect", "id": "paint", "input": "image", "source": "plate"}]
    else:
        path = write_plate(scratch, width, height, range(1, last_frame + 1)) if last_frame > 1 else None
        commands += [{"op": "create", "id": "plate", "type": "Read", "params": {"path": path}},
                     {"op": "create", "id": "roto", "type": "Roto", "params": {"width": width, "height": height}},
                     {"op": "create", "id": "paint", "type": "RotoPaint"},
                     {"op": "connect", "id": "paint", "input": "image", "source": "plate"},
                     {"op": "create", "id": "merge", "type": "Merge"},
                     {"op": "connect", "id": "merge", "input": "A", "source": "roto"},
                     {"op": "connect", "id": "merge", "input": "B", "source": "paint"},
                     {"op": "create", "id": "grade", "type": "Grade", "params": {"multiply": 1.1}},
                     {"op": "connect", "id": "grade", "input": "image", "source": "merge"}]
    commands += [{"op": "view", "id": "roto"}]
    dispatcher.execute({"op": "batch", "commands": commands})
    if case == "light":
        dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": [make_shape(0, width / 2, height / 2, height * 0.3, keyed=False)]})
    else:
        dispatcher.execute({"op": "set_shapes", "id": "roto", "shapes": shape_grid(20, width, height)})
        dispatcher.execute({"op": "set_paint_items", "id": "paint", "items": make_strokes(30, width, height)})
    document = dispatcher.document
    document["time"].update(first=1, last=last_frame, current=1)
    return document


# ------------------------------------------------------------------------------------------ instrumentation

class Recorder:
    """Stamps from the GUI thread (handler entry/exit, paint end) and the render worker (trace marks), all time.monotonic()."""

    def __init__(self, viewer_class):
        """Installed before any Window exists: PySide6 caches a virtual override at the first call, so a hook added to a
        class after its first paint is never called."""
        self.window = None
        self.viewer = None
        self.viewer_class = viewer_class
        self.handlers = []          # (begin, end, kind)
        self.paints = []            # (begin, end)
        self.shown = []             # (time shown, generation)
        self.requests = []          # (time, generation)
        self.panels = []            # (begin, end) of every panel rebuild that ran
        self.spans = []             # (name, begin, end) of the functions --spans wraps
        self.heartbeat = None
        self.span_installed = False
        self.generation_of = {}     # generation -> index of the scheduled event whose handler asked for it
        self.current_event = None
        self._install()

    def attach(self, window):
        self.window = window
        self.viewer = window.viewer

    def _install(self):
        viewer_class = self.viewer_class
        self._originals = {}
        recorder = self

        def wrap(name):
            original = getattr(viewer_class, name)
            self._originals[name] = original

            def wrapped(widget, event, _original=original, _name=name):
                begin = time.monotonic()
                before = recorder.window.generation if recorder.window is not None else 0
                try:
                    return _original(widget, event)
                finally:
                    end = time.monotonic()
                    recorder.handlers.append((begin, end, _name))
                    if recorder.window is not None and recorder.window.generation != before and recorder.current_event is not None:
                        for generation in range(before + 1, recorder.window.generation + 1):
                            recorder.generation_of[generation] = recorder.current_event
            setattr(viewer_class, name, wrapped)
        for name in ("mousePressEvent", "mouseMoveEvent", "mouseReleaseEvent", "keyPressEvent"):
            wrap(name)
        original_paint = viewer_class.paintEvent
        self._originals["paintEvent"] = original_paint

        def paint(widget, event, _original=original_paint):
            begin = time.monotonic()
            try:
                return _original(widget, event)
            finally:
                recorder.paints.append((begin, time.monotonic()))
        viewer_class.paintEvent = paint
        playback_trace.enable()

    def remove(self):
        for name, original in self._originals.items():
            setattr(self.viewer_class, name, original)
        playback_trace.disable()

    def trace_events(self):
        events = playback_trace._events or []
        return list(events)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values):
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {"n": len(values), "median": round(statistics.median(values), 2), "p95": round(percentile(values, 0.95), 2),
            "max": round(max(values), 2)}


class GuiSampler:
    """Poor man's py-spy: a thread that samples the GUI thread's Python stack every millisecond (wall time, so a call that is
    blocked inside Qt counts). `top()` returns the innermost frames and the frames anywhere on the stack by share of samples."""

    def __init__(self, interval=0.001):
        import threading
        self.interval = interval
        self.main_id = threading.get_ident()
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        import threading
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            frame = sys._current_frames().get(self.main_id)
            if frame is not None:
                stack = []
                while frame is not None and len(stack) < 40:
                    code = frame.f_code
                    stack.append(f"{os.path.basename(code.co_filename)}:{code.co_firstlineno}({code.co_name})")
                    frame = frame.f_back
                self.samples.append((time.monotonic(), stack))
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def top(self, count=14, begin=0.0, end=float('inf')):
        idle = ("processEvents", "pump", "pump_until", "run", "finish", "run_action", "main", "<module>")
        window = [stack for stamp, stack in self.samples if begin <= stamp <= end]
        busy = [s for s in window if s and not any(s[0].endswith(f"({name})") for name in ("pump", "pump_until"))
                and "processEvents" not in s[0]]
        total = max(1, len(window))
        inner, cumulative = {}, {}
        for stack in busy:
            inner[stack[0]] = inner.get(stack[0], 0) + 1
            for name in set(stack):
                if name.split(":")[0] != "benchmark_roto_interaction.py":
                    cumulative[name] = cumulative.get(name, 0) + 1
        def rows(table):
            ordered = sorted(table.items(), key=lambda kv: -kv[1])[:count]
            return [{"where": name, "ms": round(n * self.interval * 1000), "share": round(n / total, 3)} for name, n in ordered]
        return {"samples": len(window), "busy_samples": len(busy), "innermost": rows(inner), "cumulative": rows(cumulative)}


class GilWatch:
    """Finds who holds the interpreter lock while the GUI thread waits. A thread sleeps 0.5 ms in a loop; when it wakes more than
    4 ms late it was waiting for the lock, and the stacks of every other thread at that instant name the code that just let go
    of it (the render worker's frame is the call that held it). Lateness above 4 ms is counted per innermost render-side frame."""

    def __init__(self):
        import threading
        self.main_id = threading.get_ident()
        self.late = []                        # (lateness ms, [innermost frames of the other threads])
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        import threading
        self.late = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import threading
        mine = threading.get_ident()
        while not self._stop.is_set():
            before = time.monotonic()
            time.sleep(0.0005)
            lateness = (time.monotonic() - before - 0.0005) * 1000
            if lateness > 4.0:
                frames = []
                for ident, frame in sys._current_frames().items():
                    if ident in (mine, self.main_id):
                        continue
                    stack = []
                    while frame is not None and len(stack) < 6:
                        code = frame.f_code
                        stack.append(f"{os.path.basename(code.co_filename)}:{frame.f_lineno}({code.co_name})")
                        frame = frame.f_back
                    frames.append(stack)
                self.late.append((lateness, frames))

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def top(self, count=12):
        counts, total = {}, {}
        for lateness, frames in self.late:
            for stack in frames:
                if not stack or stack[0].startswith("threading.py") or stack[0].startswith("thread.py"):
                    continue
                counts[stack[0]] = counts.get(stack[0], 0) + 1
                total[stack[0]] = total.get(stack[0], 0.0) + lateness
        rows = sorted(counts, key=lambda k: -total[k])[:count]
        return {"late_wakeups": len(self.late), "worst_ms": round(max((l for l, _ in self.late), default=0.0), 1),
                "by_innermost_frame": [{"where": k, "count": counts[k], "late_ms": round(total[k], 1)} for k in rows]}


class Heartbeat:
    """A GUI-thread timer every 4 ms: the longest gap between two beats inside a window is the longest the GUI thread was
    unreachable (a long handler, or a render thread holding the interpreter lock)."""
    PERIOD_MS = 4

    def __init__(self):
        from PySide6.QtCore import QTimer
        self.beats = []
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(self.PERIOD_MS)
        self.cpu = []
        self.timer.timeout.connect(self._beat)
        self.timer.start()

    def _beat(self):
        self.beats.append(time.monotonic())
        self.cpu.append(time.thread_time())

    def gap_cpu(self, begin, end, over=10.0):
        """For each gap over `over` ms: (gap ms, GUI thread CPU ms inside it). CPU far below the gap means the thread was
        waiting (the interpreter lock, a driver, the X server); CPU near the gap means it was working."""
        rows = []
        pairs = [(b, c) for b, c in zip(self.beats, self.cpu) if begin <= b <= end]
        for (b0, c0), (b1, c1) in zip(pairs, pairs[1:]):
            if (b1 - b0) * 1000 - self.PERIOD_MS > over:
                rows.append(((b1 - b0) * 1000, (c1 - c0) * 1000))
        return rows

    def longest_gap(self, begin, end):
        """Longest stall (ms beyond one period) between `begin` and `end`: beats before and after count, so a stall that
        straddles an edge of the window is clipped to the window."""
        inside = [begin] + [b for b in self.beats if begin <= b <= end] + [end]
        gaps = [(b - a) * 1000 - self.PERIOD_MS for a, b in zip(inside, inside[1:])]
        return max(0.0, max(gaps))

    def stalls(self, begin, end, over=50.0):
        inside = [begin] + [b for b in self.beats if begin <= b <= end] + [end]
        return sum(1 for a, b in zip(inside, inside[1:]) if (b - a) * 1000 - self.PERIOD_MS > over)


# ----------------------------------------------------------------------------------------------- the driver

class Driver:
    def __init__(self, app, window, recorder):
        self.app, self.window, self.recorder = app, window, recorder
        self.viewer = window.viewer
        self.events = []            # one dict per scheduled event

    def block(self, milliseconds):
        """Run the Qt event loop for `milliseconds`, blocking in it as the application does when idle. (A processEvents spin
        never blocks, so it holds the interpreter lock and slows the render thread by ten times: a number the artist never sees.)"""
        from PySide6.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        QTimer.singleShot(max(1, int(milliseconds)), loop.quit)
        loop.exec()

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self.block(min(remaining * 1000, 4))

    def pump_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.block(2)
            if predicate():
                return True
        return False

    def viewport_point(self, x, y):
        return self.viewer.mapFromScene(QPointF(x, y))

    def post_mouse(self, kind, scene_xy, buttons=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
        point = self.viewport_point(*scene_xy)
        local = QPointF(point)
        glob = QPointF(self.viewer.viewport().mapToGlobal(point))
        types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
                 "release": QEvent.Type.MouseButtonRelease}
        button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
        held = buttons if kind != "release" else Qt.MouseButton.NoButton
        event = QMouseEvent(types[kind], local, glob, button, held, modifiers)
        QCoreApplication.postEvent(self.viewer.viewport(), event)

    def post_key(self, key, text=""):
        for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            QCoreApplication.postEvent(self.viewer, QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier, text))

    def run(self, script, rate=FRAME_INTERVAL, settle=0.0):
        """`script` is a list of (kind, payload). Each step is scheduled `rate` after the previous one (discrete steps
        wait for their own paint first, as a person does before the next click). Returns the per-event records."""
        records = []
        start = time.monotonic() + 0.05
        scheduled = start
        for number, (kind, payload) in enumerate(script):
            while time.monotonic() < scheduled:
                self.block(min(max((scheduled - time.monotonic()) * 1000, 1), 2))
            record = {"kind": kind, "scheduled": scheduled, "posted": time.monotonic(), "number": number,
                      "handlers_before": len(self.recorder.handlers), "paints_before": len(self.recorder.paints)}
            self.recorder.current_event = len(self.events)
            self.events.append(record)
            records.append(record)
            if kind == "key":
                self.post_key(*payload)
            elif kind == "widget":
                payload()          # a callable that posts its own real events (a toolbar click)
            else:
                self.post_mouse(kind, payload)
            if kind in ("press", "release", "key", "widget"):
                # a click: let the handler and its paint happen before the next step
                self.pump_until(lambda: len(self.recorder.paints) > record["paints_before"]
                                and len(self.recorder.handlers) > record["handlers_before"], 1.0)
                scheduled = max(scheduled + rate, time.monotonic() + (settle or 0.02))
            else:
                scheduled += rate
        return records

    def finish(self, records, idle=0.05, wait_render=True):
        """After the last event: wait for the last picture. Returns when the render worker is idle and the viewer shows it."""
        self.pump(idle)
        if wait_render:
            self.pump_until(lambda: not self.window.busy and not len(self.window.preview_queue)
                        and not self.window.timer.isActive(), 15.0)
        self.pump(0.05)


def QEvent_all():
    from PySide6.QtCore import QEventLoop
    return QEventLoop.ProcessEventsFlag.AllEvents


def analyse(recorder, records, label):
    """Turn the stamps for `records` into the numbers of one action."""
    handlers = recorder.handlers
    paints = recorder.paints
    moves = [r for r in records if r["kind"] == "move"]
    discrete = [r for r in records if r["kind"] != "move"]
    overlay_moves, handler_moves, overlay_discrete, handler_discrete = [], [], [], []
    handled_moves = 0
    for record in records:
        index = record["handlers_before"]
        later = [h for h in handlers[index:] if h[0] >= record["posted"] - 1e-6]
        begin_after = record["posted"]
        record_handlers = [h for h in later if h[2] in {"move": ("mouseMoveEvent",), "press": ("mousePressEvent",),
                                                         "release": ("mouseReleaseEvent",),
                                                         "key": ("keyPressEvent",), "widget": ("mousePressEvent", "mouseReleaseEvent")}.get(record["kind"], ())]
        handler = record_handlers[0] if record_handlers else None
        record["handler_ms"] = (handler[1] - handler[0]) * 1000 if handler else None
        if record["kind"] == "move" and handler:
            handled_moves += 1
        done = None
        after = handler[1] if handler else record["posted"]
        for begin, end in paints:
            if begin >= after - 1e-4 and end >= record["posted"]:
                done = end
                break
        record["overlay_ms"] = (done - record["scheduled"]) * 1000 if done else None
        record["lateness_ms"] = (record["posted"] - record["scheduled"]) * 1000
        target = (moves, handler_moves, overlay_moves) if record["kind"] == "move" else (discrete, handler_discrete, overlay_discrete)
        if record["handler_ms"] is not None:
            target[1].append(record["handler_ms"])
        if record["overlay_ms"] is not None:
            target[2].append(record["overlay_ms"])
    trace = recorder.trace_events()
    started = sum(1 for _, _, text in trace if text.startswith("start#"))
    requests = [(t, int(text.split("#")[1])) for t, _, text in trace if text.startswith("request#")]
    shown = [(t, int(text.split("#")[1])) for t, _, text in trace if text.startswith("shown#")]
    ages, last_shown = [], None
    for stamp, generation in shown:
        event = recorder.generation_of.get(generation)
        if event is not None and event < len(recorder.driver.events):
            ages.append((stamp - recorder.driver.events[event]["scheduled"]) * 1000)
        last_shown = stamp
    final = None
    if records and shown:
        final = (shown[-1][0] - records[-1]["scheduled"]) * 1000
    marks = {}
    for stamp, _thread, text in trace:
        name, _, generation = text.partition("#")
        if generation:
            marks.setdefault(int(generation), {}).setdefault(name, stamp)
    def between(first, second):
        return [(m[second] - m[first]) * 1000 for m in marks.values() if first in m and second in m]
    stages = {"worker_wait_ms": summary(between("request", "start")), "evaluate_ms": summary(between("eval_begin", "eval_end")),
              "convert_ms": summary(between("convert_begin", "convert_end")), "emit_to_ready_ms": summary(between("emit", "ready_begin")),
              "ready_to_shown_ms": summary(between("ready_begin", "shown"))}
    return {"label": label, "render_stages": stages,
            "events_posted": len(records), "moves_posted": len(moves), "moves_handled": handled_moves,
            "moves_coalesced": len(moves) - handled_moves,
            "move_handler_ms": summary(handler_moves), "move_overlay_ms": summary(overlay_moves),
            "move_late_ms": summary([r["lateness_ms"] for r in moves]),
            "click_handler_ms": summary(handler_discrete), "click_overlay_ms": summary(overlay_discrete),
            "renders_requested": len(requests), "renders_started": started, "pictures_shown": len(shown),
            "renders_dropped": max(0, len(requests) - len(shown)),
            "picture_age_ms": summary(ages), "last_picture_after_last_event_ms": round(final, 1) if final is not None else None}


# ---------------------------------------------------------------------------------------------- the actions

class Shot:
    """Knows the shot's geometry so each action can aim at real handles."""

    def __init__(self, window, width, height):
        self.window, self.width, self.height = window, width, height

    @property
    def document(self):
        return self.window.dispatcher.document

    def shapes(self, key="roto"):
        from nodebased import shapes as shape_model
        frame = int(self.document["time"]["current"])
        payload = self.document["node_data"].get(key, {})
        return shape_model.resolve_shapes({"shapes": payload.get("shapes", [])}, frame)

    def point(self, shape, index):
        shapes = self.shapes()
        resolved = shapes[shape % len(shapes)]["points"][index % len(shapes[shape % len(shapes)]["points"])]
        return float(resolved["x"]), float(resolved["y"])

    def view(self, key):
        self.window.command({"op": "view", "id": key})
        self.window.graph.scene().clearSelection()
        self.window.graph.items_by_id[key].setSelected(True)
        QApplication.instance().processEvents()
        # An artist looks at the picture of a node before drawing on it: wait for its full-resolution picture, so the first
        # move is not measured against the render of a view the artist has not seen yet.
        deadline = time.monotonic() + 20
        while (self.window.busy or len(self.window.preview_queue) or self.window.timer.isActive()) and time.monotonic() < deadline:
            QApplication.instance().processEvents()
            time.sleep(0.005)
        time.sleep(0.2)
        QApplication.instance().processEvents()


def drag_script(start, end, count=60):
    count = max(3, int(count * MOVE_SCALE))
    script = [("press", start)]
    for i in range(1, count + 1):
        t = i / count
        script.append(("move", (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)))
    script.append(("release", end))
    return script


def circle_script(center, radius, count):
    return [("move", (center[0] + radius * math.cos(2 * math.pi * i / count),
                      center[1] + radius * math.sin(2 * math.pi * i / count))) for i in range(count)]


def build_actions(driver, shot, window):
    """name -> (prepare, script). `prepare` runs before measuring (not timed) and returns the script's steps."""
    viewer = window.viewer
    W, H = shot.width, shot.height
    toolbar = viewer.roto_toolbar

    def select_tool(group, tool=None):
        viewer.select_roto_tool(group, tool or {"Select": "Select Points", "Shapes": "Bezier"}.get(group, group))
        QApplication.instance().processEvents()

    actions = {}

    def tool_switch_keys():
        shot.view("roto")
        return [("key", (Qt.Key.Key_V, "v")), ("key", (Qt.Key.Key_V, "v")), ("key", (Qt.Key.Key_Q, "q")),
                ("key", (Qt.Key.Key_V, "v")), ("key", (Qt.Key.Key_Q, "q"))]

    def tool_switch_toolbar():
        shot.view("roto")
        steps = []
        for group in ("Shapes", "Select", "Shapes", "Select"):
            button = toolbar.buttons[group]

            def click(button=button):
                center = button.rect().center()
                glob = QPointF(button.mapToGlobal(center))
                for kind, event_type in (("press", QEvent.Type.MouseButtonPress), ("release", QEvent.Type.MouseButtonRelease)):
                    QCoreApplication.postEvent(button, QMouseEvent(
                        event_type, QPointF(center), glob, Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton if kind == "press" else Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
            steps.append(("widget", click))
        return steps

    def draw_bezier():
        shot.view("roto")
        select_tool("Shapes", "Bezier")
        cx, cy, r = W * 0.5, H * 0.5, H * 0.12     # inside the empty middle
        steps = []
        for i in range(12):
            a = 2 * math.pi * i / 12
            steps.append(("press", (cx + r * math.cos(a), cy + r * math.sin(a))))
            steps.append(("release", (cx + r * math.cos(a), cy + r * math.sin(a))))
        steps.append(("press", (cx + r, cy)))           # the first point again closes the shape
        steps.append(("release", (cx + r, cy)))
        return steps

    def click_select():
        shot.view("roto")
        select_tool("Select", "Select All")
        steps = []
        for si in (3, 8, 12, 17, 5, 14):
            p = shot.point(si, 0)
            steps += [("press", p), ("release", p)]
        return steps

    def marquee():
        shot.view("roto")
        select_tool("Select", "Select Points")
        return drag_script((W * 0.02, H * 0.02), (W * 0.62, H * 0.6), 60)

    def drag_point():
        shot.view("roto")
        select_tool("Select", "Select Points")
        start = shot.point(7, 2)
        return drag_script(start, (start[0] + 90, start[1] + 60), 60)

    def drag_feather():
        shot.view("roto")
        select_tool("Select", "Select Feather Points")
        from nodebased import shapes as shape_model
        resolved = shot.shapes()[6 % len(shot.shapes())]
        handles = shape_model.feather_handle_points(resolved["points"], resolved["feather"])
        start = (float(handles[3]["x"]), float(handles[3]["y"]))
        return drag_script(start, (start[0] + 40, start[1] + 25), 60)

    def drag_shape():
        shot.view("roto")
        select_tool("Select", "Select All")
        shape = 9 % len(shot.shapes())
        center_points = [shot.point(shape, i) for i in range(12)]
        cx = sum(p[0] for p in center_points) / 12
        cy = sum(p[1] for p in center_points) / 12
        viewer.roto_selected_shape_index = shape
        viewer.roto_selected_points = {(shape, i) for i in range(12)}
        viewer.viewport().update()
        QApplication.instance().processEvents()
        # A place inside the transform box where a press is a move: not on its pivot, a handle, a point or any outline.
        context = viewer._roto_context()
        resolved = viewer._roto_resolved(context)
        geometry = viewer._roto_box_geometry(resolved, context[3])
        bounds, pivot = geometry[1], geometry[3]
        start = None
        for fx in (0.3, 0.5, 0.7, 0.2, 0.8, 0.4, 0.6):
            for fy in (0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8):
                candidate = QPointF(bounds.left() + bounds.width() * fx, bounds.top() + bounds.height() * fy)
                action, _ = viewer._roto_box_action(candidate, bounds, abs(viewer.view_zoom()), pivot)
                if (action == "move" and viewer._roto_hit_spline(candidate, resolved, context[3]) is None
                        and viewer._roto_hit_point(candidate, resolved, context[3]) is None):
                    start = (candidate.x(), candidate.y())
                    break
            if start:
                break
        if start is None:
            return []
        return drag_script(start, (start[0] + 120, start[1] + 70), 60)

    def paint_stroke():
        shot.view("paint")
        select_tool("Brushes", "Brush")
        steps = [("press", (W * 0.2, H * 0.2))]
        moves = max(6, int(120 * MOVE_SCALE))
        for i in range(1, moves + 1):
            steps.append(("move", (W * 0.2 + 600.0 * i / moves, H * 0.2 + 120 * math.sin(i * 120 / moves / 14.0))))
        steps.append(("release", (W * 0.2 + 600.0, H * 0.2 + 20)))
        return steps

    def shape_list_row():
        shot.view("roto")
        steps = []
        window.rebuild_properties_dock()
        QApplication.instance().processEvents()
        from PySide6.QtWidgets import QTreeWidget
        tree = next((t for t in window.findChildren(QTreeWidget) if t.objectName() == "roto-shape-list"), None)
        if tree is None:
            return []
        count = tree.topLevelItemCount()
        rows = [i % count for i in (2, 11, 5, 17, 8, 14)] if count else []
        for row in rows:
            def click(row=row):
                tree = next((t for t in window.findChildren(QTreeWidget) if t.objectName() == "roto-shape-list"), None)
                item = tree.topLevelItem(row)
                tree.scrollToItem(item)
                rect = tree.visualItemRect(item)
                center = rect.center()
                center.setX(40)
                glob = QPointF(tree.viewport().mapToGlobal(center))
                for event_type, held in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                                         (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
                    QCoreApplication.postEvent(tree.viewport(), QMouseEvent(
                        event_type, QPointF(center), glob, Qt.MouseButton.LeftButton, held, Qt.KeyboardModifier.NoModifier))
            steps.append(("widget", click))
        return steps

    def knob_opacity():
        shot.view("roto")
        viewer.roto_selected_shape_index = min(4, len(shot.shapes()) - 1)
        window.rebuild_properties_dock()
        QApplication.instance().processEvents()
        from PySide6.QtWidgets import QDoubleSpinBox
        from nodebased.app import FloatSliderControl
        # The node's own Opacity knob (it commits when the slider is released, like every generic knob). The per-shape
        # slider in the Roto shapes panel commits only on Enter, so a drag there changes no picture.
        spin = next((s for s in window.findChildren(QDoubleSpinBox) if s.objectName() == "opacity-field"), None)
        if spin is None:
            return []
        control = spin.parentWidget()
        while control is not None and not isinstance(control, FloatSliderControl):
            control = control.parentWidget()
        widget = control.slider if control is not None else spin
        rect = widget.rect()
        y = rect.center().y()
        span = rect.width() - 16

        def post(kind, fraction):
            x = rect.left() + 8 + span * fraction
            glob = QPointF(widget.mapToGlobal(QPoint(int(x), y)))
            types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove, "release": QEvent.Type.MouseButtonRelease}
            QCoreApplication.postEvent(widget, QMouseEvent(types[kind], QPointF(x, y), glob,
                                       Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton,
                                       Qt.MouseButton.NoButton if kind == "release" else Qt.MouseButton.LeftButton,
                                       Qt.KeyboardModifier.NoModifier))
        steps = [("widget", lambda: post("press", 0.9))]
        for i in range(1, 41):
            steps.append(("widget_move", lambda f=0.9 - 0.7 * i / 40: post("move", f)))
        steps.append(("widget", lambda: post("release", 0.2)))
        return steps

    def step_frame_keyed():
        shot.view("roto")
        steps = []
        for _ in range(8):
            steps.append(("widget", lambda: window.step_frame(1) if hasattr(window, "step_frame") else window.command(
                {"op": "time", "current": int(window.dispatcher.document["time"]["current"]) + 1})))
        return steps

    def undo_redo():
        shot.view("roto")
        select_tool("Select", "Select Points")
        steps = []
        for i in range(3):
            start = shot.point(2 + i, 1)
            steps += drag_script(start, (start[0] + 30, start[1] + 20), 3)
        driver.run(steps, settle=0.3)
        driver.finish([])
        return [("key_undo", None)] * 3 + [("key_redo", None)] * 3

    actions.update({
        "tool_switch_keys": tool_switch_keys, "tool_switch_toolbar": tool_switch_toolbar, "draw_bezier": draw_bezier,
        "click_select": click_select, "marquee": marquee, "drag_point": drag_point, "drag_feather": drag_feather,
        "drag_shape": drag_shape, "paint_stroke": paint_stroke, "shape_list_row": shape_list_row,
        "knob_opacity": knob_opacity, "step_frame_keyed": step_frame_keyed, "undo_redo": undo_redo})
    return actions


def run_action(app, window, recorder, driver, shot, name, builder, profile, repeats, sample=False):
    """Runs one action `repeats` times with a reset between runs, returns the merged result (and the profile when asked)."""
    from PySide6.QtGui import QKeySequence
    results = []
    profile_text = None
    for attempt in range(repeats):
        reset(window, shot)
        script = builder()
        if not script:
            return {"label": name, "skipped": "the control was not found"}
        recorder.handlers.clear()
        recorder.paints.clear()
        recorder.panels.clear()
        recorder.spans.clear()
        playback_trace.disable(); playback_trace.enable()
        recorder.generation_of.clear()
        driver.events.clear()
        profiler = cProfile.Profile() if profile else None
        undo_before = len(window.dispatcher.undo_stack)
        scripted = []
        for kind, payload in script:
            if kind == "widget_move":
                scripted.append(("widget", payload))
            elif kind == "key_undo":
                scripted.append(("widget", lambda: window.command({"op": "undo"})))
            elif kind == "key_redo":
                scripted.append(("widget", lambda: window.command({"op": "redo"})))
            else:
                scripted.append((kind, payload))
        no_render = bool(os.environ.get("NODEBASED_BENCH_NO_RENDER"))
        if no_render and not profiler:
            window.timer.stop()
            window.timer.start = lambda *a, **k: None             # diagnostic: the overlay with no render thread competing
        if profiler:
            # Python 3.12's cProfile is process-wide, so the render worker would be in the profile. The profiled pass
            # therefore never starts a render: what is left is the GUI thread's own work (the part this tool is about).
            window.timer.stop()
            window.timer.start = lambda *a, **k: None               # request_preview arms this timer to start the worker
            window.thumbnail_timer.stop()
            window.thumbnail_timer.start = lambda *a, **k: None
            profiler.enable()
        sampler = GuiSampler() if sample else None
        gil = GilWatch() if os.environ.get("NODEBASED_BENCH_GIL") else None
        if sampler:
            sampler.start()
        if gil:
            gil.start()
        records = driver.run(scripted)
        release = records[-1]["posted"] if records and records[-1]["kind"] in ("release", "widget") else None
        driver.finish(records, wait_render=profiler is None)
        if release is not None and profiler is None:
            driver.pump_until(lambda: time.monotonic() > release + 1.1, 3.0)       # the second after the release is on the record
        if sampler:
            sampler.stop()
        if gil:
            gil.stop()
        if no_render and not profiler:
            del window.timer.start
            window.preview_queue.cancel()
        if profiler:
            profiler.disable()
            del window.timer.start
            del window.thumbnail_timer.start
            window.preview_queue.cancel()
        result = analyse(recorder, records, name)
        result["undo_steps_added"] = len(window.dispatcher.undo_stack) - undo_before
        if release is not None and profiler is None and recorder.heartbeat is not None:
            after = [(b, e) for b, e in recorder.panels if b >= release]
            result["panel_update_ms"] = round(max([(e - b) * 1000 for b, e in after], default=0.0), 1)
            result["panel_done_after_release_ms"] = round(max([(e - release) * 1000 for _, e in after], default=0.0), 1)
            result["post_release_stall_ms"] = round(recorder.heartbeat.longest_gap(release, release + 1.0), 1)
            result["post_release_stalls_over_50"] = recorder.heartbeat.stalls(release, release + 1.0)
            first_move = records[0]["posted"]
            result["drag_stall_ms"] = round(recorder.heartbeat.longest_gap(first_move, release), 1)
            gaps = recorder.heartbeat.gap_cpu(first_move, release)
            result["drag_gap_cpu"] = {"gaps": len(gaps), "gap_ms_total": round(sum(g for g, _ in gaps), 1),
                                      "gui_cpu_ms_in_gaps": round(sum(c for _, c in gaps), 1)}
            result["drag_gui_busy_gaps_over_10"] = recorder.heartbeat.stalls(first_move, release, over=10.0)
        if gil:
            result["gil_watch"] = gil.top()
        if recorder.span_installed and records:
            cut = release if release is not None else float("inf")
            result["spans"] = {"drag": span_summary(recorder, records[0]["posted"], cut),
                               "after_release": span_summary(recorder, cut, float("inf"))}
        if sampler:
            first = records[0]["posted"] if records else 0.0
            cut = release if release is not None else float("inf")
            result["sample_top"] = {"drag": sampler.top(30, first, cut), "after_release": sampler.top(30, cut)}
            if recorder.heartbeat is not None and release is not None:
                result["stall_stacks"] = stall_stacks(sampler, recorder.heartbeat, release, release + 1.0)
        results.append(result)
        if profiler:
            profile_text = top_functions(profiler)
            dump = os.environ.get("NODEBASED_PROFILE_DIR")         # the whole profile, for print_callers afterwards
            if dump:
                os.makedirs(dump, exist_ok=True)
                profiler.dump_stats(os.path.join(dump, f"{name}.prof"))
    merged = merge(results)
    if profile_text:
        merged["profile_top"] = profile_text
    return merged


def reset(window, shot):
    viewer = window.viewer
    viewer.roto_drag = None
    viewer.roto_drawing = False
    viewer.roto_draw_points = []
    viewer.roto_selected_points = set()
    viewer.roto_selected_shape_index = -1
    viewer.roto_active_group = None
    viewer.paint_drag = None
    window.command({"op": "time", "current": 1}) if int(window.dispatcher.document["time"]["current"]) != 1 else None
    QApplication.instance().processEvents()
    deadline = time.monotonic() + 15
    while (window.busy or len(window.preview_queue) or window.timer.isActive()) and time.monotonic() < deadline:
        QApplication.instance().processEvents()
        time.sleep(0.005)
    time.sleep(0.2)
    QApplication.instance().processEvents()


def merge(results):
    """Several repeats of one action: pool the first-level numbers by taking the median of the medians and the worst p95."""
    if len(results) == 1:
        return results[0]
    merged = dict(results[0])
    for key in ("move_handler_ms", "move_overlay_ms", "move_late_ms", "click_handler_ms", "click_overlay_ms", "picture_age_ms"):
        rows = [r[key] for r in results if r[key]["n"]]
        if rows:
            merged[key] = {"n": sum(r["n"] for r in rows), "median": round(statistics.median(r["median"] for r in rows), 2),
                           "p95": round(max(r["p95"] for r in rows), 2), "max": round(max(r["max"] for r in rows), 2)}
    values = [r["last_picture_after_last_event_ms"] for r in results if r.get("last_picture_after_last_event_ms") is not None]
    if values:
        merged["last_picture_after_last_event_ms"] = round(statistics.median(values), 1)     # the median repeat, not the first (cold) one
        merged["last_picture_after_last_event_worst_ms"] = round(max(values), 1)
    for key in ("moves_handled", "moves_coalesced", "renders_requested", "renders_started", "pictures_shown", "renders_dropped"):
        merged[key] = round(statistics.median(r[key] for r in results), 1)
    for key in ("panel_update_ms", "panel_done_after_release_ms", "post_release_stall_ms", "post_release_stalls_over_50",
                "drag_stall_ms", "drag_gui_busy_gaps_over_10"):
        values = [r[key] for r in results if key in r]
        if values:
            merged[key] = max(values)               # the worst repeat: three repeats are too few for a p95 of their own
    merged["repeats"] = len(results)
    return merged


def top_functions(profiler, count=int(os.environ.get("NODEBASED_BENCH_TOP", "14"))):
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    rows = []
    skip = ("processEvents", "driver", "benchmark_roto_interaction", "<built-in method builtins.exec>", "pump", "run", "finish")
    for (filename, line, function), (cc, nc, tt, ct, callers) in stats.stats.items():
        base = os.path.basename(filename)
        if base == "benchmark_roto_interaction.py" or function in ("processEvents", "pump_until", "<module>"):
            continue
        rows.append((ct, tt, nc, f"{base}:{line}({function})"))
    rows.sort(reverse=True)
    return [{"cum_ms": round(c * 1000, 1), "self_ms": round(t * 1000, 1), "calls": n, "where": w} for c, t, n, w in rows[:count]]


SPAN_TARGETS = ("Window.preview_ready", "Window._show_image", "Window.request_preview", "Window._flush_drag_preview",
                "Viewer._flush_drag_preview", "Viewer.paintEvent", "Viewer.drawForeground", "Viewer.paint_overlays",
                "Viewer._preview_roto_drag", "Viewer._preview_document_drag", "Window.refresh_timeline_marks",
                "Window.update_status_summary", "Window._offer_direct", "Window._frame_viewer_to", "Window._adapt_playback_proxy",
                "Window.update_corner_readout", "Window.refresh_output_preview", "Window.start_preview",
                "Window.after_command", "Window.sync_timeline", "Window.sync_viewer_look", "Window.sync_viewer_frame",
                "Window.sync_viewer_knobs", "Window.update_title", "Window.render_identity", "Window.refresh_panel_soon",
                "Window._commit_roto_shapes", "Window._commit_roto_drag_now", "Window._roto_panel_signature",
                "Dispatcher.execute", "NodeGraph.rebuild", "Window.viewport_set_document")


def stall_stacks(sampler, heartbeat, begin, end, over=50.0, count=24):
    """For each GUI stall over `over` ms between `begin` and `end`: when it began, how long it lasted, and the frames (outside this
    tool) most of its samples were in, so a stall is named by the code that held the thread."""
    inside = [begin] + [b for b in heartbeat.beats if begin <= b <= end] + [end]
    rows = []
    for a, b in zip(inside, inside[1:]):
        length = (b - a) * 1000 - heartbeat.PERIOD_MS
        if length <= over:
            continue
        window = [stack for stamp, stack in sampler.samples if a <= stamp <= b]
        table = {}
        for stack in window:
            for name in set(stack):
                if name.split(":")[0] != "benchmark_roto_interaction.py" and "processEvents" not in name and "exec" not in name:
                    table[name] = table.get(name, 0) + 1
        top = sorted(table.items(), key=lambda kv: -kv[1])[:count]
        rows.append({"begins_after_release_ms": round((a - begin) * 1000, 1), "length_ms": round(length, 1),
                     "samples": len(window), "frames": [{"where": n, "samples": c} for n, c in top]})
    return rows


def install_spans(recorder, window_class, viewer_class, names=SPAN_TARGETS):
    """Wrap each named method so every call is stamped (GUI thread and render thread alike; the name says which class)."""
    from nodebased.core import Dispatcher
    from nodebased.app import Graph as NodeGraph
    classes = {"Window": window_class, "Viewer": viewer_class, "Dispatcher": Dispatcher, "NodeGraph": NodeGraph}
    for name in names:
        class_name, _, attribute = name.partition(".")
        owner = classes[class_name]
        original = getattr(owner, attribute, None)
        if original is None:
            continue

        def wrapped(*args, _original=original, _name=name, **kwargs):
            begin = time.monotonic()
            try:
                return _original(*args, **kwargs)
            finally:
                recorder.spans.append((_name, begin, time.monotonic()))
        setattr(owner, attribute, wrapped)


def span_summary(recorder, begin, end):
    rows = {}
    for name, b, e in recorder.spans:
        if begin <= b <= end:
            rows.setdefault(name, []).append((e - b) * 1000)
    out = []
    for name, durations in rows.items():
        out.append({"where": name, "calls": len(durations), "total_ms": round(sum(durations), 1),
                    "median_ms": round(statistics.median(durations), 2), "max_ms": round(max(durations), 2)})
    return sorted(out, key=lambda r: -r["total_ms"])


# --------------------------------------------------------------------------------------------------- main

ACTION_ORDER = ["tool_switch_keys", "tool_switch_toolbar", "draw_bezier", "click_select", "marquee", "drag_point",
                "drag_feather", "drag_shape", "paint_stroke", "shape_list_row", "knob_opacity", "step_frame_keyed", "undo_redo"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--self-check", action="store_true", help="offscreen, tiny shot, two actions; proves the tool runs")
    parser.add_argument("--cases", nargs="+", default=["realistic", "light"], choices=["realistic", "light"])
    parser.add_argument("--actions", nargs="+", default=ACTION_ORDER)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="also run each action once under cProfile (separate pass)")
    parser.add_argument("--spans", action="store_true", help="time every call of the viewer/window methods in SPAN_TARGETS (a separate pass, real timings)")
    parser.add_argument("--sample", action="store_true", help="also run each action once with a 1 ms wall-clock sampler on the GUI thread (separate pass, real timings)")
    parser.add_argument("--table", nargs="+", type=pathlib.Path, help="print a markdown table of result files (label=path) and exit")
    parser.add_argument("--short", action="store_true", help="a fifth of the moves per drag (for the profiled pass of a slow build)")
    parser.add_argument("--profile-only", action="store_true", help="only the cProfile pass (its timings are inflated; the tops are the point)")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("roto-interaction.json"))
    parser.add_argument("--label", default="")
    parser.add_argument("--size", default="1080p", choices=["1080p", "tiny"])
    args = parser.parse_args()
    if args.table:
        runs = []
        for item in args.table:
            label, _, path = str(item).partition("=")
            runs.append((label, json.loads(pathlib.Path(path).read_text())))
        print(markdown_table(runs))
        return 0
    global MOVE_SCALE
    if os.environ.get("NODEBASED_SWITCH_INTERVAL"):        # experiment: how long the GUI thread waits for the render thread's GIL
        sys.setswitchinterval(float(os.environ["NODEBASED_SWITCH_INTERVAL"]))
    if args.short:
        MOVE_SCALE = 0.2
    if args.self_check:
        args.size, args.repeats = "tiny", 1
        args.actions = ["tool_switch_keys", "drag_point", "paint_stroke"]
        args.cases = ["light"]
    width, height = (1920, 1080) if args.size == "1080p" else (200, 120)
    last_frame = 48 if args.size == "1080p" else 3

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    from nodebased.app import STYLE, Window
    from PySide6.QtCore import QSettings
    app.setStyleSheet(STYLE)
    out = {"tool": "benchmark_roto_interaction", "platform": os.environ.get("QT_QPA_PLATFORM"), "display": os.environ.get("DISPLAY"),
           "python": platform.python_version(), "label": args.label, "size": args.size, "cases": {}}
    from nodebased.app import Viewer
    recorder = Recorder(Viewer)
    original_rebuild = Window._run_panel_rebuild

    def timed_rebuild(window_self, _original=original_rebuild):
        begin = time.monotonic()
        ran = window_self._panel_rebuild is not None and not window_self.viewer.gesture_active()
        try:
            return _original(window_self)
        finally:
            if ran:
                recorder.panels.append((begin, time.monotonic()))
    Window._run_panel_rebuild = timed_rebuild
    base = os.environ.get("NODEBASED_BENCH_SCRATCH")       # the plate frames are large: keep them off /tmp, which is RAM here
    with tempfile.TemporaryDirectory(prefix="roto-interaction-", dir=base) as scratch:
        for case in args.cases:
            QSettings("NodeBased", "NodeBased").clear() if args.self_check else None
            document = build_document(case, scratch, width, height, last_frame)
            window = Window(document)
            window.resize(1700, 1000) if not args.self_check else None
            window.show()
            deadline = time.monotonic() + 60
            while window.frame is None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            window.graph.items_by_id["roto"].setSelected(True)
            app.processEvents()
            window.viewer.fit()
            app.processEvents()
            window.viewer.setFocus()
            recorder.attach(window)
            if recorder.heartbeat is None:
                recorder.heartbeat = Heartbeat()
            driver = Driver(app, window, recorder)
            recorder.driver = driver
            recorder.handlers.clear(); recorder.paints.clear()
            shot = Shot(window, width, height)
            shot.view("roto")
            builders = build_actions(driver, shot, window)
            case_out = {"actions": {}, "proxy_tier": int(window.proxy.currentData())}
            # warm every cache once so the numbers are steady state
            for name in args.actions:
                if name not in builders:
                    continue
                print(f"[{case}] {name}", flush=True)
                try:
                    if args.profile_only:
                        case_out["actions"][name] = run_action(app, window, recorder, driver, shot, name, builders[name], True, 1)
                        continue
                    case_out["actions"][name] = run_action(app, window, recorder, driver, shot, name, builders[name], False, args.repeats)
                    if args.spans:
                        if not recorder.span_installed:
                            install_spans(recorder, Window, Viewer)
                            recorder.span_installed = True
                        spanned = run_action(app, window, recorder, driver, shot, name, builders[name], False, 1)
                        case_out["actions"][name]["spans"] = spanned.get("spans")
                    if args.sample:
                        sampled = run_action(app, window, recorder, driver, shot, name, builders[name], False, 1, sample=True)
                        case_out["actions"][name]["sample_top"] = sampled.get("sample_top")
                        case_out["actions"][name]["stall_stacks"] = sampled.get("stall_stacks")
                    if args.profile:
                        profiled = run_action(app, window, recorder, driver, shot, name, builders[name], True, 1)
                        case_out["actions"][name]["profile_top"] = profiled.get("profile_top", [])
                except Exception as error:      # one broken action must not lose the rest
                    import traceback
                    traceback.print_exc()
                    case_out["actions"][name] = {"label": name, "error": f"{type(error).__name__}: {error}"}
            out["cases"][case] = case_out
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()
    recorder.remove()
    args.output.write_text(json.dumps(out, indent=1))
    print_table(out)
    return 0


def markdown_table(runs, case="realistic"):
    """`runs` is [(label, result json)]: one row per action, the overlay, click and picture numbers of each run side by side."""
    names = []
    for _, data in runs:
        for name in data["cases"].get(case, {}).get("actions", {}):
            if name not in names:
                names.append(name)
    head = "| Action | " + " | ".join(f"{label}: move overlay, click overlay, picture age (ms median / p95)" for label, _ in runs) + " |"
    lines = [head, "|---|" + "---|" * len(runs)]
    for name in names:
        cells = []
        for _, data in runs:
            row = data["cases"].get(case, {}).get("actions", {}).get(name)
            if not row or "error" in row or "skipped" in row:
                cells.append("n/a")
                continue
            def pair(key):
                stat = row[key]
                return f"{stat['median']:g} / {stat['p95']:g}" if stat["n"] else "-"
            cells.append(f"{pair('move_overlay_ms')}; {pair('click_overlay_ms')}; {pair('picture_age_ms')}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_table(out):
    print(f"\n{'case':10} {'action':20} {'move ovl med/p95':>18} {'click ovl med/p95':>19} {'img age med/p95':>17} "
          f"{'req/drop/coal':>14} {'undo':>5}")
    for case, data in out["cases"].items():
        for name, row in data["actions"].items():
            if "error" in row or "skipped" in row:
                print(f"{case:10} {name:20} {row.get('error') or row.get('skipped')}")
                continue
            def pair(key):
                s = row[key]
                return f"{s['median']}/{s['p95']}" if s["n"] else "-"
            print(f"{case:10} {name:20} {pair('move_overlay_ms'):>18} {pair('click_overlay_ms'):>19} {pair('picture_age_ms'):>17} "
                  f"{row['renders_requested']}/{row['renders_dropped']}/{row['moves_coalesced']:>3} {row.get('undo_steps_added', ''):>5}"
                  + (f"   drag: worst GUI gap {row['drag_stall_ms']} ms, {row['drag_gui_busy_gaps_over_10']} gaps over 10 ms" if "drag_stall_ms" in row else "")
                  + (f"   after release: panel {row['panel_update_ms']} ms (done {row['panel_done_after_release_ms']} ms), "
                     f"worst GUI stall {row['post_release_stall_ms']} ms" if "post_release_stall_ms" in row else ""))


if __name__ == "__main__":
    sys.exit(main())
