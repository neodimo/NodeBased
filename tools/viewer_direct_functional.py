"""VP1 functional walk on a private X display: open the direct picture in a real Window, change the picture,
pan/zoom, resize, dock/undock, close and reopen, and print what the presenter counted. It cannot judge
pixels on screen (a screen-presented surface does not show up in X captures); pixel correctness is pinned
by tests/test_directviewer.py against an offscreen target.

    DISPLAY=:61 QT_QPA_PLATFORM=xcb NODEBASED_DIRECT_ADAPTER=cpu python tools/viewer_direct_functional.py OUT.json
"""
import json
import sys
import time
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPointF

import tests.isolation  # noqa: F401
from nodebased.app import Window

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
log = {"platform": app.platformName(), "checks": {}}


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()


def image(k, w=640, h=360):
    y, x = np.indices((h, w))
    a = np.empty((h, w, 3), np.uint8)
    a[:] = np.where(((x // 32 + y // 32 + k) % 2)[..., None], 190, 40)
    return QtGui.QImage(a.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()


def wheel(widget, steps, pos):
    p = QPointF(*pos)
    return QtGui.QWheelEvent(p, widget.mapToGlobal(p), QtCore.QPoint(0, 0), QtCore.QPoint(0, int(120 * steps)),
                             Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)


w = Window(agent_name="func-" + uuid.uuid4().hex)
w.show()
pump(1.5)
log["checks"]["default_viewer_has_picture"] = w.viewer.last_display_image is not None
w.direct_action.setChecked(True)
pump(1.0)
widget = w.direct_widget
log["checks"]["opened"] = widget is not None and widget.error is None
if widget is None:
    json.dump(log, open(sys.argv[1], "w"), indent=2); print(json.dumps(log, indent=1)); sys.exit(0)
pr = widget.presenter
log["adapter"] = pr.adapter_info.get("device") if pr else None
log["selection_note"] = getattr(pr, "selection_note", "")
log["selection"] = getattr(pr, "selection", {})
log["pixel_ratio"] = widget.get_pixel_ratio()
before = dict(pr.counters)
for k in range(20):
    w._show_image(image(k), 1, None, generation=100)
    pump(0.05)
after = dict(pr.counters)
log["checks"]["changing_checker_uploads"] = {"uploads": after["uploads"] - before.get("uploads", 0),
                                              "conversions": after["conversions"] - before.get("conversions", 0)}
# pan/zoom must not convert or upload
mark = dict(pr.counters)
c = (widget.width() / 2, widget.height() / 2)
for i in range(20):
    QtWidgets.QApplication.sendEvent(widget, wheel(widget, 1 if i % 2 else -1, c))
    pump(0.03)
pr = widget.presenter
view_after_wheel = widget.view
log["checks"]["wheel_changed_view_without_upload"] = (pr.counters["uploads"] == mark["uploads"] and pr.counters["conversions"] == mark["conversions"]
                                                      and pr.counters["draws"] > mark["draws"])
press = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonPress, QPointF(*c), widget.mapToGlobal(QPointF(*c)), Qt.MouseButton.MiddleButton,
                          Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier)
QtWidgets.QApplication.sendEvent(widget, press)
for dx in range(1, 11):
    q = QPointF(c[0] + dx * 5, c[1] + dx * 3)
    QtWidgets.QApplication.sendEvent(widget, QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, q, widget.mapToGlobal(q), Qt.MouseButton.NoButton,
                                                              Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier))
    pump(0.02)
rel = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonRelease, QPointF(*c), widget.mapToGlobal(QPointF(*c)), Qt.MouseButton.MiddleButton,
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
QtWidgets.QApplication.sendEvent(widget, rel)
log["checks"]["pan_moved_view"] = widget.view[:2] != view_after_wheel[:2] and not widget.auto_fit
# resize and re-dock
sizes = []
dock = w.direct_dock
for size in ((800, 450), (500, 300), (1000, 600)):
    dock.resize(*size)
    pump(0.4)
    sizes.append(list(widget.get_physical_size()))
log["resize_physical_sizes"] = sizes
log["checks"]["resize_no_error"] = widget.error is None
w.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
dock.setFloating(False)
dock.show()
pump(0.8)
log["docked_state"] = dict(error=widget.error, visible=dock.isVisible(), floating=dock.isFloating(), action=w.direct_action.isChecked(), presenter=widget.presenter is not None)
log["checks"]["docked_no_error"] = widget.error is None and dock.isVisible() and not dock.isFloating()
dock.setFloating(True)
pump(0.5)
log["checks"]["refloated_no_error"] = widget.error is None and dock.isVisible()
# close and reopen
w.direct_action.setChecked(False)
pump(0.3)
w.direct_action.setChecked(True)
pump(0.8)
log["checks"]["reopen_same_widget_still_draws"] = widget.error is None and widget.presenter is not None
n = widget.presenter.counters["uploads"]
w._show_image(image(99), 1, None, generation=101)
pump(0.3)
log["checks"]["reopen_receives_new_picture"] = widget.presenter.counters["uploads"] == n + 1
# full close, then a fresh widget
widget.close()
pump(0.5)
w.direct_action.setChecked(False)
w.direct_dock = None
w.direct_widget = None
w.direct_action.setChecked(True)
pump(1.0)
log["checks"]["fresh_widget_after_close"] = w.direct_widget is not None and w.direct_widget.error is None and w.direct_widget.presenter is not None
log["checks"]["default_viewer_still_has_picture"] = w.viewer.last_display_image is not None
w.saved_document = w.dispatcher.document
w.close()
json.dump(log, open(sys.argv[1], "w"), indent=2)
print(json.dumps(log, indent=1))
