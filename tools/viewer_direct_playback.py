"""VP2a playback check: does the main window keep responding while the direct picture plays?

Runs the real Window on a private X display (xcb), opens the Direct picture dock, plays a Checker whose cell
size is animated (every frame is a new picture), and while it plays sends real `xdotool` input:

* graph presses on the Checker node, timed from the moment xdotool is told to press to the end of the
  graph's next paint (so time the event spends queued behind a busy Qt thread counts);
* graph drags of that node, each move timed from the xdotool send to the next graph paint;
* timeline scrubs (press and drag on the frame bar), each step timed from the send to the next paint of
  the frame bar.

It also records every gap in a 4 ms timer on the Qt thread (the longest stall an input could have met),
the Qt-thread time spent in the direct dock's hand-over and draw, and how many frames were produced and
how many the dock uploaded.

    Xvfb :77 -screen 0 2560x1440x24 &
    DISPLAY=:77 QT_QPA_PLATFORM=xcb NODEBASED_DIRECT_ADAPTER=discrete \
        python tools/viewer_direct_playback.py OUT.json [--size 1920x1080] [--seconds 10] [--no-direct]

Never point it at the real display: it moves the pointer.
"""
import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--size", default="1920x1080")
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--no-direct", action="store_true", help="play in the Viewer only (the before case)")
p.add_argument("--proxy", action="store_true", help="leave 'Proxy while playing' on (default: full resolution)")
args = p.parse_args()

import tests.isolation  # noqa: F401
from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])
from nodebased import app as appmod, directviewer  # noqa: E402
from nodebased.app import Window  # noqa: E402

W, H = (int(v) for v in args.size.lower().split("x"))
MAIN = threading.get_ident()
T = collections.defaultdict(list)          # label -> [ms on the Qt thread]
SHAPES = collections.Counter()              # (shape, dtype) of every uploaded frame
EV = []                                     # (label, t_start, t_end)


def spin(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.001)


def dist(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    pick = lambda q: v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]
    return {"n": len(v), "p50": round(pick(50), 2), "p95": round(pick(95), 2), "max": round(v[-1], 2),
            "mean": round(statistics.fmean(v), 2)}


def timed(obj, name, label):
    f = getattr(obj, name)

    def g(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            if threading.get_ident() == MAIN:
                T[label].append((time.perf_counter() - t) * 1000)
    setattr(obj, name, g)


w = Window(agent_name="vp2a-" + uuid.uuid4().hex)
w.resize(1800, 1100)
w.move(0, 0)
w.show()
spin(1.0)
w.agent_command({"op": "batch", "commands": [
    {"op": "create", "type": "Checker", "id": "chk", "params": {"width": W, "height": H, "size": 64}, "pos": [0, 0]},
    {"op": "view", "id": "chk"},
    {"op": "set_key", "id": "chk", "param": "size", "frame": 1, "value": 16},
    {"op": "set_key", "id": "chk", "param": "size", "frame": 240, "value": 240}]})
w.set_time(first=1, last=240, current=1, fps=24.0)
if not args.proxy:
    w.playback_proxy.setChecked(False)
t0 = time.perf_counter()
while (w.busy or len(w.preview_queue)) and time.perf_counter() - t0 < 60:
    spin(0.05)

direct = not args.no_direct
selection = {}
if direct:
    w.direct_action.setChecked(True)
    spin(1.0)
    dock = w.direct_dock
    if dock is None or w.direct_widget is None or w.direct_widget.error:
        print("direct picture did not start:", getattr(w.direct_widget, "error", None), flush=True)
        sys.exit(2)
    dock.resize(960, 540)
    dock.move(1820, 0)          # beside the main window, never over the graph or the timeline
    spin(1.0)
    presenter = w.direct_widget.presenter
    selection = dict(presenter.selection or {}, adapter=presenter.adapter_info.get("device", ""))
    timed(w, "_show_direct_linear", "hand-over to the dock")
    timed(presenter, "draw", "dock draw (upload included)")
    timed(presenter, "_copy_slot", "copy recorded")
    _copy = presenter._copy_slot

    def noting_copy(encoder, slot):
        payload, width, height, fmt = slot["meta"][:4]
        SHAPES[((height, width), fmt)] += 1
        return _copy(encoder, slot)
    presenter._copy_slot = noting_copy
timed(w, "_show_image", "Window._show_image")
timed(w, "preview_ready", "Window.preview_ready")

G = w.graph
bar = w.frame_slider


class Watch(QObject):
    """Paint and mouse events on the graph viewport and the frame bar, stamped as Qt delivers them."""

    KINDS = {QEvent.Type.Paint: "paint", QEvent.Type.MouseButtonPress: "press", QEvent.Type.MouseMove: "move"}

    def __init__(self, name):
        super().__init__()
        self.name = name

    def eventFilter(self, obj, event):
        kind = self.KINDS.get(event.type())
        if kind is not None:
            now = time.perf_counter()
            EV.append((f"{kind}_{self.name}", now, now))
        return False


watches = [Watch("graph"), Watch("bar")]
G.viewport().installEventFilter(watches[0])
bar.installEventFilter(watches[1])

produced = [0]
_ready = w.preview_ready


def counting_ready(payload, frame, image, status, render_region=None):
    request = payload[0]
    if request.display and request.playing and frame is not None:
        produced[0] += 1
    return _ready(payload, frame, image, status, render_region)


w.signals.finished.disconnect()
w.signals.finished.connect(counting_ready)

gaps = []
last = [time.perf_counter()]


def beat():
    now = time.perf_counter()
    gaps.append((now - last[0]) * 1000)
    last[0] = now


STALLS = collections.Counter()              # where the Qt thread was while the 4 ms timer was over 30 ms late


def stall_sampler():
    import traceback
    while not stop_sampler.is_set():
        time.sleep(0.005)
        if heart.isActive() and (time.perf_counter() - last[0]) * 1000 > 30:
            frame = sys._current_frames().get(MAIN)
            if frame is not None:
                stack = traceback.extract_stack(frame)[-4:]
                STALLS[" < ".join(f"{os.path.basename(f.filename)}:{f.name}" for f in reversed(stack))] += 1


stop_sampler = threading.Event()
heart = QTimer()
heart.setInterval(4)
heart.timeout.connect(beat)

xd = subprocess.Popen(["xdotool", "-"], stdin=subprocess.PIPE, text=True, bufsize=1)


def send(line):
    t = time.perf_counter()
    xd.stdin.write(line + "\n")
    xd.stdin.flush()
    return t


def to_global(widget, point):
    g = widget.mapToGlobal(point)
    r = widget.devicePixelRatioF()
    return round(g.x() * r), round(g.y() * r)


def node_point():
    item = G.items_by_id["chk"]
    return to_global(G.viewport(), G.mapFromScene(item.sceneBoundingRect().center()))


def next_after(label, t):
    hit = next((e for e in EV if e[0] == label and e[2] >= t), None)
    return None if hit is None else (hit[2] - t) * 1000


w.toggle_playback(True)
spin(1.0)
heart.start()
last[0] = time.perf_counter()
threading.Thread(target=stall_sampler, daemon=True).start()
start = time.perf_counter()
produced[0] = 0
uploads_before = presenter.counters["linear_uploads"] if direct else 0
press, drag, scrub = [], [], []
while time.perf_counter() - start < args.seconds:
    # graph press on the node
    x, y = node_point()
    send(f"mousemove {x} {y}")
    spin(0.15)
    EV.clear()
    t = send("mousedown 1")
    spin(0.25)
    press.append(next_after("paint_graph", t))
    # drag it a little and back
    for i in range(12):
        t = send(f"mousemove_relative -- {3 if i < 6 else -3} 0")
        spin(0.03)
        drag.append(next_after("paint_graph", t))
    send("mouseup 1")
    spin(0.15)
    # scrub the frame bar
    geo = bar.rect()
    bx, by = to_global(bar, geo.center())
    send(f"mousemove {bx - 200} {by}")
    spin(0.1)
    EV.clear()
    send("mousedown 1")
    spin(0.05)
    for i in range(12):
        t = send(f"mousemove_relative -- 25 0")
        spin(0.03)
        scrub.append(next_after("paint_bar", t))
    send("mouseup 1")
    spin(0.1)
    if not w.playing:
        w.toggle_playback(True)
        spin(0.2)
elapsed = time.perf_counter() - start
stop_sampler.set()
heart.stop()
labels = collections.Counter(e[0] for e in EV)
uploads = (presenter.counters["linear_uploads"] - uploads_before) if direct else 0
w.toggle_playback(False)
spin(0.5)
xd.stdin.close()

result = {
    "display": os.environ.get("DISPLAY", ""), "platform": app.platformName(), "size": [W, H],
    "seconds": round(elapsed, 2), "direct": direct, "selection": selection,
    "proxy_while_playing": bool(args.proxy),
    "graph_press_to_paint_ms": dist(press), "graph_drag_move_to_paint_ms": dist(drag),
    "timeline_scrub_to_paint_ms": dist(scrub), "qt_thread_gap_ms": dist(gaps),
    "qt_thread_gaps_over_50ms": sum(1 for g in gaps if g > 50),
    "frames_produced": produced[0], "frames_uploaded_by_dock": uploads,
    "qt_thread_ms": {k: dist(v) for k, v in T.items()},
    "presenter_counters": dict(presenter.counters) if direct else {},
    "last_scrub_events": dict(labels),
    "qt_thread_stall_samples": dict(STALLS.most_common(12)),
    "uploaded_frames": {f"{shape[1]}x{shape[0]} {dtype}": n for (shape, dtype), n in SHAPES.items()},
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(result, fh, indent=1)
print(json.dumps({k: result[k] for k in ("graph_press_to_paint_ms", "graph_drag_move_to_paint_ms",
                                          "timeline_scrub_to_paint_ms", "qt_thread_gap_ms",
                                          "frames_produced", "frames_uploaded_by_dock")}), flush=True)
os._exit(0)
