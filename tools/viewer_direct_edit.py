"""Editing on the direct picture: how long after a drag step does the moved overlay reach a presented frame?

Runs the real Window on a private X display (xcb), opens the Direct picture dock with "Edit on the direct
picture" on, and drags two handles with real `xdotool` input on the dock itself:

* a Roto point (Roto node, one four-point shape);
* the Transform centre (the node's translate handle);
* the A/B wipe's centre handle (`--handles wipe`: B is on the card, the line is painted over it, and the card's
  shader moves B's half with every step; a view-state drag, so it leaves no undo entries).

Each drag is a press, a series of relative moves 30 ms apart and a release. Every move (and every press) is timed
from the moment xdotool is told to send it to the end of the first `draw` of the direct picture whose overlay image
was painted after that move reached the shared hit-test (the overlay redraw, not the picture's re-render,
which follows a few frames later). The 95th percentile over all moves is the gate: under 50 ms at 1080p; the
press is gated the same way. The time to the end of the surface's present call is recorded beside it (on a private
Xvfb display the Vulkan present copies the image out through the processor, which a real screen does not).

    Xvfb :77 -screen 0 2560x1440x24 &
    DISPLAY=:77 QT_QPA_PLATFORM=xcb NODEBASED_DIRECT_ADAPTER=discrete \
        python tools/viewer_direct_edit.py OUT.json [--size 1920x1080] [--drags 25] [--limit-ms 50]

`NODEBASED_DIRECT_ADAPTER=cpu` runs llvmpipe. Present return is when the application call returns, not
screen response. Never point it at the real display: it moves the pointer. Exit status 1 when a gate fails.

On Windows (`--driver windows`, the default there) the pointer is driven with `SetCursorPos` and `mouse_event`
instead of xdotool, so `tools/laptop_present_check.py --edit` can run the same drags on the laptop and the A6000
machine. That driver has not been run: no Windows number exists until DiMo runs it.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--size", default="1920x1080")
p.add_argument("--drags", type=int, default=25, help="drags per handle")
p.add_argument("--handles", default="roto,transform", help="comma list of roto, transform, wipe")
p.add_argument("--steps", type=int, default=8, help="moves per drag")
p.add_argument("--limit-ms", type=float, default=50.0)
p.add_argument("--driver", choices=("auto", "xdotool", "windows"), default="auto")
args = p.parse_args()

try:
    import tests.isolation  # noqa: F401  (keeps QSettings out of the real user file when run from a checkout)
except ImportError:
    pass
from PySide6.QtCore import QPoint, QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from nodebased import handles2d  # noqa: E402
from nodebased.app import Window  # noqa: E402

W, H = (int(v) for v in args.size.lower().split("x"))


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


w = Window(agent_name="edit-gate-" + uuid.uuid4().hex)
w.resize(1800, 1100)
w.move(0, 0)
w.show()
spin(1.0)
shape = {"name": "blob", "mode": "union", "opacity": 1.0, "feather": 0.0,
         "points": [{"x": W * 0.30, "y": H * 0.30, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                    {"x": W * 0.70, "y": H * 0.28, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                    {"x": W * 0.72, "y": H * 0.72, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                    {"x": W * 0.28, "y": H * 0.70, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}]}
w.agent_command({"op": "batch", "commands": [
    {"op": "create", "type": "Constant", "id": "c", "params": {"width": W, "height": H, "red": .4, "green": .4,
                                                               "blue": .4, "alpha": 1.0}, "pos": [0, 0]},
    {"op": "create", "type": "Roto", "id": "r", "pos": [150, 0]},
    {"op": "set", "id": "r", "param": "width", "value": W},
    {"op": "set", "id": "r", "param": "height", "value": H},
    {"op": "set_shapes", "id": "r", "shapes": [shape]},
    {"op": "create", "type": "Transform", "id": "x", "pos": [300, 0],
     "params": {"center_x": W / 2.0, "center_y": H / 2.0}},
    {"op": "connect", "id": "x", "input": "image", "source": "c"},
    {"op": "create", "type": "Constant", "id": "b", "pos": [450, 0],
     "params": {"width": W, "height": H, "red": .1, "green": .4, "blue": .7, "alpha": 1.0}},
    {"op": "view", "id": "x"}]})
t0 = time.perf_counter()
while (w.busy or len(w.preview_queue)) and time.perf_counter() - t0 < 60:
    spin(0.05)

w.direct_action.setChecked(True)
spin(1.0)
widget, dock = w.direct_widget, w.direct_dock
if dock is None or widget is None or widget.error:
    print("direct picture did not start:", getattr(widget, "error", None), flush=True)
    sys.exit(2)
dock.resize(1280, 760)
dock.move(1820, 0)             # beside the main window
spin(1.0)
presenter = widget.presenter
adapter = dict(presenter.selection or {}, device=presenter.adapter_info.get("device", ""))
if widget.edit is None:
    print("editing on the direct picture is off", flush=True)
    sys.exit(2)

HANDLED = []     # (time, kind, overlay serial after the hit-test)
DRAWS = []       # (time at the end of a draw, overlay serial of the image it drew)
PRESENTS = []    # time at the end of each surface present call
_edit = widget.edit


def watched_edit(kind, point, button, modifiers, zoom, pressure=1.0):
    handled = _edit(kind, point, button, modifiers, zoom, pressure)
    if handled:
        HANDLED.append((time.perf_counter(), kind, widget.overlay_serial))
    return handled


widget.edit = watched_edit
_draw = presenter.draw


def watched_draw(view, size, *rest):
    result = _draw(view, size, *rest)
    key = presenter.paint_key
    DRAWS.append((time.perf_counter(), key[3] if key else -1))
    return result


presenter.draw = watched_draw
_context = widget.get_wgpu_context()
_present = _context._rc_present


def watched_present(**kwargs):
    try:
        return _present(**kwargs)
    finally:
        PRESENTS.append(time.perf_counter())


_context._rc_present = watched_present

class XdotoolPointer:
    """X11: one `xdotool -` process fed a line per action. Each call returns the time it was sent."""

    def __init__(self):
        self.process = subprocess.Popen(["xdotool", "-"], stdin=subprocess.PIPE, text=True, bufsize=1)

    def _send(self, line):
        t = time.perf_counter()
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        return t

    def move_to(self, x, y):
        return self._send(f"mousemove {x} {y}")

    def down(self):
        return self._send("mousedown 1")

    def up(self):
        return self._send("mouseup 1")

    def move_by(self, dx, dy):
        return self._send(f"mousemove_relative -- {dx} {dy}")

    def close(self):
        self.process.stdin.close()


class WindowsPointer:
    """Windows: SetCursorPos and mouse_event through ctypes (physical pixels; Qt is per-monitor DPI aware)."""
    LEFT_DOWN, LEFT_UP = 0x0002, 0x0004

    def __init__(self):
        import ctypes
        self.user32 = ctypes.windll.user32

    def move_to(self, x, y):
        t = time.perf_counter()
        self.user32.SetCursorPos(int(x), int(y))
        return t

    def down(self):
        t = time.perf_counter()
        self.user32.mouse_event(self.LEFT_DOWN, 0, 0, 0, 0)
        return t

    def up(self):
        t = time.perf_counter()
        self.user32.mouse_event(self.LEFT_UP, 0, 0, 0, 0)
        return t

    def move_by(self, dx, dy):
        import ctypes
        from ctypes import wintypes
        point = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        t = time.perf_counter()
        self.user32.SetCursorPos(point.x + int(dx), point.y + int(dy))
        return t

    def close(self):
        pass


driver = args.driver if args.driver != "auto" else ("windows" if sys.platform == "win32" else "xdotool")
pointer = WindowsPointer() if driver == "windows" else XdotoolPointer()


def surface_to_global(scene_point):
    """X root coordinates of a scene point on the direct picture."""
    cx, cy, zoom = widget.view
    ratio = widget.get_pixel_ratio()
    width, height = widget.get_physical_size()
    lx = ((scene_point.x() - cx) * zoom + width / 2.0) / ratio
    ly = ((scene_point.y() - cy) * zoom + height / 2.0) / ratio
    g = widget.mapToGlobal(QPoint(round(lx), round(ly)))
    r = widget.devicePixelRatioF()
    return round(g.x() * r), round(g.y() * r)


def select(key):
    graph = w.graph
    graph.scene().clearSelection()
    graph.items_by_id[key].setSelected(True)
    w.command({"op": "view", "id": key})
    t = time.perf_counter()
    while (w.busy or len(w.preview_queue)) and time.perf_counter() - t < 30:
        spin(0.02)
    spin(0.4)


def handle_roto():
    rect = w.viewer.format_rect
    point = w.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][0]
    return QPointF(rect.left() + point["x"], rect.top() + point["y"])


def handle_transform():
    rect = w.viewer.format_rect
    p = w.dispatcher.document["nodes"]["x"]["params"]
    pivot = handles2d.pivot_point(p["translate_x"], p["translate_y"], p["center_x"], p["center_y"])
    return QPointF(rect.left() + pivot[0], rect.top() + pivot[1])


def handle_wipe():
    return w.viewer._wipe_scene_geometry()[0]


def prepare_wipe():
    w.command({"op": "viewer_input", "slot": 2, "id": "b", "activate": False})
    w.command({"op": "viewer_compare", "b": 2, "mode": "wipe"})
    t = time.perf_counter()
    while (w.busy or len(w.preview_queue)) and time.perf_counter() - t < 30:
        spin(0.02)
    spin(0.6)
    w.viewer.reset_wipe()
    if not w.viewer.card_wipe:
        raise SystemExit("the wipe is not on the card (card_wipe is off): nothing to time")


def run(label, key, locate, prepare=None, expect_undo=True):
    select(key)
    if prepare is not None:
        prepare()
    if label == "roto point":
        w.viewer.roto_selected_shape_index = 0
    steps = []                               # time each move was sent
    presses = []                             # time each press was sent
    committed = 0
    handled_before = len(HANDLED)
    for n in range(args.drags):
        start = surface_to_global(locate())
        pointer.move_to(*start)
        spin(0.12)
        undo_before = len(w.dispatcher.undo_stack)
        presses.append(pointer.down())
        spin(0.08)
        sign = 1 if n % 2 == 0 else -1
        for _ in range(args.steps):
            steps.append(pointer.move_by(6 * sign, 4 * sign))
            spin(0.03)
        pointer.up()
        spin(0.2)
        committed += len(w.dispatcher.undo_stack) - undo_before
    def timed(sent, kind):
        """(ms to the end of the first draw showing it, ms to the end of the present after that), or None."""
        latencies, presented, missed = [], [], 0
        for t in sent:
            hit = next((h for h in HANDLED if h[0] >= t and h[1] == kind), None)
            draw = None if hit is None else next((d for d in DRAWS if d[0] >= hit[0] and d[1] >= hit[2]), None)
            if draw is None:
                missed += 1
                continue
            latencies.append((draw[0] - t) * 1000)
            after = next((q for q in PRESENTS if q >= draw[0] - 0.0005), None)
            if after is not None:
                presented.append((after - t) * 1000)
        return latencies, presented, missed
    latencies, presented, missed = timed(steps, "move")
    press_latencies, press_presented, press_missed = timed(presses, "press")
    taken = {kind: sum(1 for h in HANDLED[handled_before:] if h[1] == kind) for kind in ("press", "move", "release")}
    return {"handle": label, "drags": args.drags, "moves_sent": len(steps), "moves_without_a_frame": missed,
            "taken_by_the_hit_test": taken,
            "undo_entries_from_drags": committed, "expect_undo": expect_undo,
            "move_to_presented_overlay_ms": dist(latencies), "move_to_present_return_ms": dist(presented),
            "press_to_presented_overlay_ms": dist(press_latencies), "press_to_present_return_ms": dist(press_presented),
            "presses_without_a_frame": press_missed}


wanted = [name.strip() for name in args.handles.split(",") if name.strip()]
runs = {"roto": lambda: run("roto point", "r", handle_roto),
        "transform": lambda: run("transform centre", "x", handle_transform),
        "wipe": lambda: run("wipe centre", "x", handle_wipe, prepare=prepare_wipe, expect_undo=False)}
results = [runs[name]() for name in wanted]
pointer.close()
all_p95 = max(max(r["move_to_presented_overlay_ms"].get("p95", 1e9), r["press_to_presented_overlay_ms"].get("p95", 1e9))
              for r in results)
failed = [r["handle"] for r in results
          if r["move_to_presented_overlay_ms"].get("p95", 1e9) >= args.limit_ms
          or r["press_to_presented_overlay_ms"].get("p95", 1e9) >= args.limit_ms
          or r["undo_entries_from_drags"] != (args.drags if r["expect_undo"] else 0)]
result = {"display": os.environ.get("DISPLAY", ""), "platform": app.platformName(), "size": [W, H],
          "adapter": adapter, "driver": driver, "draws": len(DRAWS), "handles": results,
          "gate": {"limit_ms": args.limit_ms, "worst_p95_ms": all_p95, "failed": failed, "passed": not failed}}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(result, fh, indent=1)
print(json.dumps({"adapter": adapter.get("device"), **{r["handle"]: r["move_to_presented_overlay_ms"] for r in results},
                  **{r["handle"] + " press": r["press_to_presented_overlay_ms"] for r in results},
                  "gate": result["gate"]}), flush=True)
os._exit(0 if not failed else 1)
