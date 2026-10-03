"""Small editable graph for serialised colour curves."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
                               QMenu, QDoubleSpinBox)

from . import colorcurves
from .colorcurves import decode, encode, evaluate


class CurveCanvas(QWidget):
    changed = Signal(str)
    readout = Signal(str)
    selection = Signal(int)

    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.curve = decode(value)
        self.selected = 0
        self.setMinimumSize(360, 220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.drag = None
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self.pan = None

    def _bounds(self):
        points = self.curve["points"]
        xmin, xmax = points[0][0], points[-1][0]
        ys = [p[1] for p in points]
        ymin, ymax = min(0.0, min(ys)), max(1.0, max(ys))
        if ymax == ymin: ymax = ymin + 1.0
        xc, yc = (xmin+xmax)/2 + self.offset.x(), (ymin+ymax)/2 + self.offset.y()
        xhalf, yhalf = (xmax-xmin)/(2*self.zoom), (ymax-ymin)/(2*self.zoom)
        return xc-xhalf, xc+xhalf, yc-yhalf, yc+yhalf

    def _xy(self, x, y):
        r = self.rect().adjusted(34, 12, -12, -28)
        xmin, xmax, ymin, ymax = self._bounds()
        return QPointF(r.left() + (x - xmin) / (xmax - xmin) * r.width(),
                       r.bottom() - (y - ymin) / (ymax - ymin) * r.height())

    def _value(self, pos):
        r = self.rect().adjusted(34, 12, -12, -28)
        xmin, xmax, ymin, ymax = self._bounds()
        x = xmin + (pos.x() - r.left()) / max(r.width(), 1) * (xmax - xmin)
        y = ymin + (r.bottom() - pos.y()) / max(r.height(), 1) * (ymax - ymin)
        return x, y

    def _emit(self):
        self.changed.emit(encode(self.curve["points"], self.curve["interpolation"],
                                  self.curve.get("slopes"), self.curve.get("modes"),
                                  self.curve.get("broken")))
        self.update()

    def _ensure_handles(self):
        colorcurves.ensure_keyed(self.curve)

    def _keyed_view(self):
        """The curve with explicit tangent data (a legacy curve is converted on a copy)."""
        if "slopes" in self.curve:
            return self.curve
        return colorcurves.ensure_keyed({**self.curve, "points": [list(p) for p in self.curve["points"]]})

    def handle_slope(self, index, side):
        """The slope the segment beside this key's handle is actually drawn with."""
        view = self._keyed_view()
        j = index - 1 if side == 0 else index
        if 0 <= j < len(view["points"]) - 1:
            return colorcurves.segment_slopes(view, j)[1 if side == 0 else 0]
        return view["slopes"][index][side]

    def _handle_pos(self, index, side):
        x, y = self.curve["points"][index]
        xmin, xmax, _, _ = self._bounds()
        dx = (xmax-xmin) / 10.0 * (-1 if side == 0 else 1)
        return self._xy(x + dx, y + self.handle_slope(index, side) * dx)

    def select(self, index):
        self.selected = max(0, min(index, len(self.curve["points"]) - 1))
        self.selection.emit(self.selected)
        self.update()

    def set_point(self, index, x, y):
        """Numeric edit of one key; the curve is re-emitted so it is one document edit."""
        colorcurves.move_point(self.curve, index, x, y)
        self._emit()

    def set_slope(self, index, side, slope):
        colorcurves.set_tangent(self.curve, index, side, slope, break_tangent=False)
        self._emit()

    def reset(self, value):
        self.curve = decode(value)
        self.selected = 0
        self.selection.emit(0)
        self._emit()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(34, 12, -12, -28)
        p.fillRect(self.rect(), QColor("#202027")); p.setPen(QPen(QColor("#555560"), 1)); p.drawRect(r)
        xmin, xmax, ymin, ymax = self._bounds()
        for i in range(1, 5):
            x = r.left() + r.width() * i / 5
            y = r.top() + r.height() * i / 5
            p.setPen(QPen(QColor("#35353e"), 1)); p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom())); p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        path = []
        for i in range(201):
            x = xmin + (xmax - xmin) * i / 200
            path.append(self._xy(x, evaluate(self.curve, x)))
        p.setPen(QPen(QColor("#73cdb5"), 2))
        for a, b in zip(path, path[1:]): p.drawLine(a, b)
        for i, (x, y) in enumerate(self.curve["points"]):
            q = self._xy(x, y)
            for side in (0, 1):
                if (i == 0 and side == 0) or (i == len(self.curve["points"]) - 1 and side == 1):
                    continue
                h = self._handle_pos(i, side)
                p.setPen(QPen(QColor("#8395bd"), 1)); p.drawLine(q, h)
                p.setBrush(QColor("#aab8df")); p.drawEllipse(h, 3, 3)
            p.setPen(QPen(QColor("#ffffff"), 1)); p.setBrush(QColor("#e1a75e" if i in (self.drag, self.selected) else "#82d7bf")); p.drawEllipse(q, 5, 5)
        p.setPen(QColor("#aaaab5")); p.drawText(3, 15, f"{xmin:g}"); p.drawText(3, self.height()-8, f"{ymin:g} .. {ymax:g}")

    def _has_handle(self, index, side):
        return not ((index == 0 and side == 0) or (index == len(self.curve["points"]) - 1 and side == 1))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.pan = e.position(); return
        distances = [(self._xy(x, y) - e.position()).manhattanLength() for x, y in self.curve["points"]]
        index = min(range(len(distances)), key=distances.__getitem__)
        if e.button() == Qt.MouseButton.RightButton:
            if distances[index] < 16:
                self.select(index)
                menu = QMenu(self)
                for mode in ("smooth", "linear", "constant", "broken"):
                    menu.addAction(mode.title(), lambda checked=False, m=mode: self._set_mode(index, m))
                menu.addSeparator()
                delete = menu.addAction("Delete key")
                if len(self.curve["points"]) <= 2: delete.setEnabled(False)
                delete.triggered.connect(lambda: self._delete_key(index))
                menu.exec(e.globalPosition().toPoint())
            return
        if e.button() == Qt.MouseButton.LeftButton:
            handles = [((self._handle_pos(i, s)-e.position()).manhattanLength(), i, s)
                       for i in range(len(self.curve["points"])) for s in (0, 1) if self._has_handle(i, s)]
            hit = min(handles, default=(999, 0, 0))
            if hit[0] < 13:
                self.drag = hit[1]; self.drag_handle = hit[2]
                self.select(hit[1])
                self.readout.emit(self._readout(hit[1]))
                return
        if e.button() == Qt.MouseButton.LeftButton and distances[index] < 16:
            self.drag = index
            self.select(index)
            self.readout.emit(self._readout(index))

    def _readout(self, index):
        x, y = self.curve["points"][index]
        return f"x {x:.6g}   y {y:.6g}"

    def mouseMoveEvent(self, e):
        if self.pan is not None:
            old = self.pan
            xmin, xmax, ymin, ymax = self._bounds()
            r = self.rect().adjusted(34, 12, -12, -28)
            self.offset += QPointF(-(e.position().x()-old.x()) / max(r.width(), 1) * (xmax-xmin),
                                    (e.position().y()-old.y()) / max(r.height(), 1) * (ymax-ymin))
            self.pan = e.position(); self.update(); return
        if self.drag is None: return
        if hasattr(self, "drag_handle"):
            x, y = self.curve["points"][self.drag]
            hx, hy = self._value(e.position())
            if abs(hx - x) > 1e-7:
                # Ctrl-drag breaks the tangent (Nuke), so only the grabbed side follows the pointer.
                colorcurves.set_tangent(self.curve, self.drag, self.drag_handle, (hy - y) / (hx - x),
                                        break_tangent=bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier))
            self.selection.emit(self.drag)
            self.update(); return
        x, y = self._value(e.position())
        x, y = colorcurves.move_point(self.curve, self.drag, x, y)
        self.readout.emit(f"x {x:.6g}   y {y:.6g}")
        self.selection.emit(self.drag)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.drag is not None:
            self.drag = None
            if hasattr(self, "drag_handle"):
                del self.drag_handle
            self._emit()
        self.pan = None

    def mouseDoubleClickEvent(self, e):
        x, y = self._value(e.position()); pts = self.curve["points"]
        if pts[0][0] < x < pts[-1][0]:
            # The new key sits on the curve with the curve's own slope: adding it never moves the curve.
            self.select(colorcurves.insert_key(self.curve, x))
            self._emit()

    def _set_mode(self, index, mode):
        self._ensure_handles()
        if mode == "broken" and index < len(self.curve["points"]) - 1:
            colorcurves.bake_segment(self.curve, index)
        self.curve["modes"][index] = mode
        self.curve["broken"][index] = mode == "broken"; self._emit()

    def _delete_key(self, index):
        colorcurves.delete_key(self.curve, index)
        self.select(min(self.selected, len(self.curve["points"]) - 1))
        self._emit()

    def wheelEvent(self, e):
        self.zoom *= 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        self.setToolTip(f"Zoom {self.zoom:.2f}× · middle-drag to pan"); self.update()


class CurveEditorDialog(QDialog):
    def __init__(self, title, value, on_change, parent=None, default=None):
        super().__init__(parent); self.setWindowTitle(title); self.resize(520, 380)
        self.default = default
        layout = QVBoxLayout(self)
        self.canvas = CurveCanvas(value); layout.addWidget(self.canvas)
        tools = QHBoxLayout(); tools.addWidget(QLabel("Interpolation"))
        self.mode = QComboBox(); self.mode.addItems(("linear", "smooth")); self.mode.setCurrentText(self.canvas.curve["interpolation"]); tools.addWidget(self.mode)
        tools.addWidget(QLabel("Double-click: add · right-click: mode/delete · drag: move · Ctrl-drag handle: break"))
        self.point_readout = QLabel("Point: —"); tools.addWidget(self.point_readout); layout.addLayout(tools)
        self.canvas.readout.connect(lambda text: self.point_readout.setText("Point: " + text))
        fields = QHBoxLayout()
        self.point_x, self.point_y = QDoubleSpinBox(), QDoubleSpinBox()
        self.slope_in, self.slope_out = QDoubleSpinBox(), QDoubleSpinBox()
        for label, box, name in (("Key X", self.point_x, "curve-key-x"), ("Y", self.point_y, "curve-key-y"),
                                 ("In slope", self.slope_in, "curve-slope-in"),
                                 ("Out slope", self.slope_out, "curve-slope-out")):
            box.setObjectName(name); box.setDecimals(5); box.setRange(-1e6, 1e6); box.setKeyboardTracking(False)
            fields.addWidget(QLabel(label)); fields.addWidget(box)
        self.reset_button = QPushButton("Reset curve"); self.reset_button.setObjectName("curve-reset")
        self.reset_button.clicked.connect(self.reset_curve)
        fields.addWidget(self.reset_button); layout.addLayout(fields)
        self.canvas.selection.connect(lambda _index: self._sync_fields())
        self.canvas.changed.connect(lambda _text: self._sync_fields())
        self.point_x.editingFinished.connect(self._point_edited)
        self.point_y.editingFinished.connect(self._point_edited)
        self.slope_in.editingFinished.connect(lambda: self._slope_edited(0, self.slope_in))
        self.slope_out.editingFinished.connect(lambda: self._slope_edited(1, self.slope_out))
        self.canvas.changed.connect(on_change)
        self.mode.currentTextChanged.connect(self._mode)
        done = QPushButton("Close"); done.clicked.connect(self.accept); layout.addWidget(done, alignment=Qt.AlignmentFlag.AlignRight)
        self._sync_fields()

    def _sync_fields(self):
        """Show the selected key in the numeric fields without echoing them back as edits."""
        canvas = self.canvas
        index = canvas.selected
        x, y = canvas.curve["points"][index]
        last = len(canvas.curve["points"]) - 1
        for box, value in ((self.point_x, x), (self.point_y, y),
                           (self.slope_in, canvas.handle_slope(index, 0)),
                           (self.slope_out, canvas.handle_slope(index, 1))):
            box.blockSignals(True); box.setValue(value); box.blockSignals(False)
        self.slope_in.setEnabled(index > 0)
        self.slope_out.setEnabled(index < last)

    def _point_edited(self):
        self.canvas.set_point(self.canvas.selected, self.point_x.value(), self.point_y.value())

    def _slope_edited(self, side, box):
        self.canvas.set_slope(self.canvas.selected, side, box.value())

    def reset_curve(self):
        value = self.default
        if value is None:
            points = self.canvas.curve["points"]
            value = encode(((points[0][0], 0.0), (points[-1][0], 1.0)))
        self.canvas.reset(value)
        self.mode.blockSignals(True); self.mode.setCurrentText(self.canvas.curve["interpolation"]); self.mode.blockSignals(False)

    def _mode(self, mode):
        curve = self.canvas.curve
        curve["interpolation"] = mode
        if "slopes" in curve:
            # A keyed curve takes its segment mode from its keys, so the combo sets every key.
            curve["modes"] = [mode] * len(curve["points"])
            curve["broken"] = [False] * len(curve["points"])
        self.canvas._emit()
