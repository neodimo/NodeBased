"""VP1 real-screen check: the direct picture is on the glass, not only in a texture.

Shows a changing checker in a `QRenderWidget(present_method="screen")` on the real display, takes a screenshot
of the composited desktop with Spectacle (it captures what the compositor shows, including screen-presented
surfaces that an X capture misses), and measures the checker in the capture: where it is, how big its cells
are, which colour sits in the top-left cell, and how all of that changes after a new picture, a wheel zoom, a
middle-drag pan and a resize. The expected numbers come from the presenter's own view state and a scale factor
calibrated from the capture, so a desktop scale of 1.75 (Xwayland windows are stretched by the compositor)
does not matter.

    DISPLAY=:0 XAUTHORITY=... QT_QPA_PLATFORM=xcb NODEBASED_DIRECT_ADAPTER=integrated \
        python tools/viewer_direct_screen.py OUT.json
    QT_QPA_PLATFORM=wayland ...   # native Wayland: Qt gives no display id, so rendercanvas takes its Wayland path

No pointer is moved: input is delivered to the widget as Qt events. It takes about ten seconds.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, Qt

import tests.isolation  # noqa: F401

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
OUT = sys.argv[1]
PICTURE = (640, 360)
CELL = 32
BLUE, ORANGE = (20, 60, 180), (200, 120, 40)       # the colours the checker uses: parity 0 is blue
TOL = 14
log = {"platform": app.platformName(), "checks": {}, "captures": []}


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()


def checker(k):
    w, h = PICTURE
    y, x = np.indices((h, w))
    on = ((x // CELL + y // CELL + k) % 2).astype(bool)
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[on], rgb[~on] = ORANGE, BLUE
    return QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()


def wheel(widget, steps, pos):
    p = QPointF(*pos)
    return QtGui.QWheelEvent(p, widget.mapToGlobal(p), QtCore.QPoint(0, 0), QtCore.QPoint(0, int(120 * steps)),
                             Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)


def mouse(widget, kind, pos, button, buttons):
    p = QPointF(*pos)
    return QtGui.QMouseEvent(kind, p, widget.mapToGlobal(p), button, buttons, Qt.KeyboardModifier.NoModifier)


def capture(name):
    """Composited desktop as an (h, w, 3) array, via Spectacle in background mode."""
    path = os.path.join(tempfile.mkdtemp(prefix="nb-screen-"), name + ".png")
    for _ in range(3):
        pump(0.5)
        subprocess.run(["spectacle", "-b", "-n", "-f", "-o", path], timeout=40, capture_output=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            break
    image = QtGui.QImage(path)
    if image.isNull():
        raise RuntimeError("Spectacle produced no capture")
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    arr = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(image.height(), image.bytesPerLine())
    arr = arr[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()
    os.remove(path)
    return arr


def near(arr, colour):
    return (np.abs(arr.astype(np.int16) - np.array(colour, np.int16)).max(axis=2) <= TOL)


def measure(arr):
    """Where the checker is in the capture: bounding box of the two checker colours, cell width, top-left colour."""
    mask = near(arr, BLUE) | near(arr, ORANGE)
    rows, cols = np.where(mask.sum(axis=1) > 40)[0], np.where(mask.sum(axis=0) > 40)[0]
    if not len(rows) or not len(cols):
        return None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    box = mask[y0:y1 + 1, x0:x1 + 1]
    mid = box[box.shape[0] // 2]
    edges = np.flatnonzero(np.diff(near(arr[y0:y1 + 1, x0:x1 + 1], BLUE)[box.shape[0] // 2].astype(np.int8)))
    cell = float(np.median(np.diff(edges))) if len(edges) > 2 else None
    inset = int(max(6, (cell or 24) / 3))
    corner = arr[y0 + inset, x0 + inset]
    first = "blue" if near(corner[None, None], BLUE)[0, 0] else "orange" if near(corner[None, None], ORANGE)[0, 0] else "other"
    return {"bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "cell_px": cell, "top_left_colour": first,
            "coverage": float(box.mean()), "edges": len(edges),
            "top_left_rgb": [int(v) for v in corner],
            "first_edge_x": int(edges[0]) if len(edges) else None}


def shot(name, **extra):
    m = measure(capture(name))
    if m:
        m.update(extra)
    log["captures"].append({"name": name, **(m or {"missing": True})})
    return m


from nodebased import directviewer as dv

widget = dv.make_widget()
widget.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
widget.setWindowTitle("NodeBased direct picture check")
widget.show()
pump(0.5)
dpr = widget.get_pixel_ratio()
logical = (int(os.environ.get("CHECK_W", "960")), int(os.environ.get("CHECK_H", "540")))
widget.resize(*logical)
widget.move(60, 60)
pump(1.0)
widget.submit(checker(0), (0, 0, *PICTURE), generation=1)
pump(1.0)
if widget.error:
    log["error"] = widget.error
    json.dump(log, open(OUT, "w"), indent=2)
    print(json.dumps(log, indent=1))
    sys.exit(1)
pr = widget.presenter
log["adapter"] = pr.adapter_info.get("device")
log["selection_note"] = pr.selection_note
log["pixel_ratio"] = dpr
log["widget_physical_size"] = list(widget.get_physical_size())
log["format"] = pr.format

a0 = shot("fit-frame0")
widget.submit(checker(1), (0, 0, *PICTURE), generation=2)
pump(1.0)
a1 = shot("fit-frame1")
widget.submit(checker(2), (0, 0, *PICTURE), generation=3)
pump(1.0)
a2 = shot("fit-frame2")
ok = lambda m: m is not None
log["checks"]["picture_is_visible_on_screen"] = ok(a0) and a0["coverage"] > 0.9
log["checks"]["new_pictures_reach_the_screen"] = (ok(a0) and ok(a1) and ok(a2) and
    [a0["top_left_colour"], a1["top_left_colour"], a2["top_left_colour"]] == ["blue", "orange", "blue"])
if ok(a0):
    # capture px per physical widget px, calibrated from the fitted picture's width
    s = a0["bbox"][2] / (PICTURE[0] * widget.view[2])
    log["capture_px_per_physical_px"] = s
    log["checks"]["aspect_16_9_kept"] = abs(a0["bbox"][2] / a0["bbox"][3] - PICTURE[0] / PICTURE[1]) < 0.03
    log["checks"]["cell_size_matches_the_view"] = (a0["cell_px"] is not None and
                                                   abs(a0["cell_px"] - CELL * widget.view[2] * s) < 2.5)

# zoom: five wheel steps about the centre; cells grow by 1.15^5
c = (widget.width() / 2, widget.height() / 2)
before_zoom = widget.view
for _ in range(5):
    QtWidgets.QApplication.sendEvent(widget, wheel(widget, 1, c))
    pump(0.05)
pump(1.0)
z = shot("zoomed", view_zoom=widget.view[2])
if ok(z) and ok(a0) and a0["cell_px"] and z["cell_px"]:
    log["checks"]["zoom_scaled_the_picture_on_screen"] = abs(z["cell_px"] / a0["cell_px"] - 1.15 ** 5) < 0.12

# pan: middle-drag by (60, 30) logical px
view0 = widget.view
QtWidgets.QApplication.sendEvent(widget, mouse(widget, QtCore.QEvent.Type.MouseButtonPress, c, Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton))
for step in range(1, 11):
    q = (c[0] + step * 6, c[1] + step * 3)
    QtWidgets.QApplication.sendEvent(widget, mouse(widget, QtCore.QEvent.Type.MouseMove, q, Qt.MouseButton.NoButton, Qt.MouseButton.MiddleButton))
    pump(0.03)
QtWidgets.QApplication.sendEvent(widget, mouse(widget, QtCore.QEvent.Type.MouseButtonRelease, c, Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton))
pump(1.0)
p = shot("panned")
log["pan_view_delta_scene_px"] = [widget.view[0] - view0[0], widget.view[1] - view0[1]]
if ok(p) and ok(z) and z["cell_px"]:
    log["checks"]["pan_moved_the_view"] = widget.view[:2] != view0[:2] and not widget.auto_fit
    # The grid phase on screen moves by the drag: 60 logical px right = 60 * dpr * capture-px-per-physical-px.
    if z["first_edge_x"] is not None and p["first_edge_x"] is not None and z["cell_px"]:
        expected = 60 * dpr * log["capture_px_per_physical_px"]
        shift = (p["first_edge_x"] + p["bbox"][0] - z["first_edge_x"] - z["bbox"][0]) % z["cell_px"]
        log["pan_on_screen_px"] = {"expected_mod_one_cell": expected % z["cell_px"], "measured_mod_one_cell": shift}
        diff = abs(shift - expected % z["cell_px"])
        log["checks"]["pan_shifted_the_grid_by_the_drag"] = min(diff, z["cell_px"] - diff) < 4

# resize: the fitted picture follows the widget
widget.fit()
pump(0.5)
widget.resize(logical[0] // 2, logical[1] // 2)
pump(1.2)
r = shot("resized-half", widget_physical=list(widget.get_physical_size()))
widget.resize(*logical)
pump(1.2)
r2 = shot("resized-back", widget_physical=list(widget.get_physical_size()))
if ok(r) and ok(a0):
    log["checks"]["resize_shrank_the_picture_on_screen"] = abs(r["bbox"][2] / a0["bbox"][2] - 0.5) < 0.05
if ok(r2) and ok(a0):
    log["checks"]["resize_back_restored_it"] = abs(r2["bbox"][2] / a0["bbox"][2] - 1.0) < 0.05
log["checks"]["no_error"] = widget.error is None
log["presenter_counters"] = dict(widget.presenter.counters) if widget.presenter else None

# close and reopen
widget.close()
pump(0.5)
widget2 = dv.make_widget()
widget2.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
widget2.show()
pump(0.5)
widget2.resize(*logical)
widget2.move(60, 60)
pump(0.8)
widget2.submit(checker(1), (0, 0, *PICTURE), generation=1)
pump(1.0)
q = shot("reopened")
log["checks"]["reopened_widget_shows_the_picture"] = ok(q) and q["coverage"] > 0.9 and q["top_left_colour"] == "orange"
widget2.close()
json.dump(log, open(OUT, "w"), indent=2)
print(json.dumps({"checks": log["checks"], "adapter": log["adapter"], "pixel_ratio": dpr,
                  "capture_px_per_physical_px": log.get("capture_px_per_physical_px")}, indent=1))
