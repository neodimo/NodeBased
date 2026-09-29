"""Small editable graph for serialised colour curves."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QMenu

from .colorcurves import decode, encode, evaluate


class CurveCanvas(QWidget):
    changed = Signal(str)
    readout = Signal(str)

    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.curve = decode(value)
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
        if "slopes" in self.curve:
            return
        pts = self.curve["points"]
        slopes = []
        for i, (x, y) in enumerate(pts):
            a, b = pts[max(0, i-1)], pts[min(len(pts)-1, i+1)]
            slope = (b[1]-a[1]) / max(b[0]-a[0], 1e-12)
            slopes.append([slope, slope])
        self.curve["slopes"] = slopes
        self.curve["modes"] = ["smooth" if self.curve["interpolation"] == "smooth" else "linear"] * len(pts)
        self.curve["broken"] = [False] * len(pts)

    def _handle_pos(self, index, side):
        x, y = self.curve["points"][index]
        xmin, xmax, _, _ = self._bounds()
        dx = (xmax-xmin) / 10.0 * (-1 if side == 0 else 1)
        slope = self.curve.get("slopes", self._initial_slopes())[index][side]
        return self._xy(x + dx, y + slope * dx)

    def _initial_slopes(self):
        pts = self.curve["points"]
        out = []
        for i, (x, y) in enumerate(pts):
            a, b = pts[max(0, i-1)], pts[min(len(pts)-1, i+1)]
            slope = (b[1]-a[1]) / max(b[0]-a[0], 1e-12)
            out.append([slope, slope])
        return out

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
                h = self._handle_pos(i, side)
                p.setPen(QPen(QColor("#8395bd"), 1)); p.drawLine(q, h)
                p.setBrush(QColor("#aab8df")); p.drawEllipse(h, 3, 3)
            p.setPen(QPen(QColor("#ffffff"), 1)); p.setBrush(QColor("#e1a75e" if i == self.drag else "#82d7bf")); p.drawEllipse(q, 5, 5)
        p.setPen(QColor("#aaaab5")); p.drawText(3, 15, f"{xmin:g}"); p.drawText(3, self.height()-8, f"{ymin:g} .. {ymax:g}")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.pan = e.position(); return
        distances = [(self._xy(x, y) - e.position()).manhattanLength() for x, y in self.curve["points"]]
        index = min(range(len(distances)), key=distances.__getitem__)
        if e.button() == Qt.MouseButton.RightButton:
            if distances[index] < 16:
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
                       for i in range(len(self.curve["points"])) for s in (0, 1)]
            hit = min(handles, default=(999, 0, 0))
            if hit[0] < 13:
                self._ensure_handles(); self.drag = hit[1]; self.drag_handle = hit[2]
                self.drag_broken = bool(e.modifiers() & Qt.KeyboardModifier.AltModifier)
                return
        if e.button() == Qt.MouseButton.LeftButton and distances[index] < 16:
            self.drag = index
            x, y = self.curve["points"][index]
            self.readout.emit(f"x {x:.6g}   y {y:.6g}")

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
            self._ensure_handles()
            x, y = self.curve["points"][self.drag]
            hx, hy = self._value(e.position())
            slope = (hy-y) / (hx-x) if abs(hx-x) > 1e-7 else self.curve["slopes"][self.drag][self.drag_handle]
            self.curve["slopes"][self.drag][self.drag_handle] = slope
            if not self.curve["broken"][self.drag] and not self.drag_broken:
                self.curve["slopes"][self.drag][1-self.drag_handle] = slope
            elif self.drag_broken:
                self.curve["broken"][self.drag] = True
            self.curve["modes"][self.drag] = "broken"
            if self.drag > 0:
                self.curve["modes"][self.drag-1] = "broken"
            self.update(); return
        x, y = self._value(e.position()); pts = self.curve["points"]
        lo = pts[self.drag-1][0] if self.drag else self._bounds()[0]
        hi = pts[self.drag+1][0] if self.drag+1 < len(pts) else self._bounds()[1]
        pts[self.drag] = [min(max(x, lo + 1e-5), hi - 1e-5), y]
        self.readout.emit(f"x {pts[self.drag][0]:.6g}   y {y:.6g}")
        self.update()

    def mouseReleaseEvent(self, e):
        if self.drag is not None:
            self.drag = None
            if hasattr(self, "drag_handle"):
                del self.drag_handle
                if self.curve.get("slopes"):
                    self._emit()
            else:
                self._emit()
        self.pan = None

    def mouseDoubleClickEvent(self, e):
        x, y = self._value(e.position()); pts = self.curve["points"]
        if pts[0][0] < x < pts[-1][0]:
            self._ensure_handles(); pts.append([x, y]); pts.sort(key=lambda v: v[0])
            # New key receives a neutral tangent pair; keep metadata aligned with key order.
            self.curve["slopes"] = [[0.0, 0.0] for _ in pts]
            self.curve["modes"] = ["smooth"] * len(pts); self.curve["broken"] = [False] * len(pts)
            self._emit()

    def _set_mode(self, index, mode):
        self._ensure_handles(); self.curve["modes"][index] = mode
        self.curve["broken"][index] = mode == "broken"; self._emit()

    def _delete_key(self, index):
        if len(self.curve["points"]) <= 2: return
        self._ensure_handles()
        for name in ("points", "slopes", "modes", "broken"):
            self.curve[name].pop(index)
        self._emit()

    def wheelEvent(self, e):
        self.zoom *= 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        self.setToolTip(f"Zoom {self.zoom:.2f}× · middle-drag to pan"); self.update()


class CurveEditorDialog(QDialog):
    def __init__(self, title, value, on_change, parent=None):
        super().__init__(parent); self.setWindowTitle(title); self.resize(520, 340)
        layout = QVBoxLayout(self)
        self.canvas = CurveCanvas(value); layout.addWidget(self.canvas)
        tools = QHBoxLayout(); tools.addWidget(QLabel("Interpolation"))
        self.mode = QComboBox(); self.mode.addItems(("linear", "smooth")); self.mode.setCurrentText(self.canvas.curve["interpolation"]); tools.addWidget(self.mode)
        tools.addWidget(QLabel("Double-click: add · right-click: delete · drag: move"))
        self.point_readout = QLabel("Point: —"); tools.addWidget(self.point_readout); layout.addLayout(tools)
        self.canvas.readout.connect(lambda text: self.point_readout.setText("Point: " + text))
        self.canvas.changed.connect(on_change)
        self.mode.currentTextChanged.connect(self._mode)
        done = QPushButton("Close"); done.clicked.connect(self.accept); layout.addWidget(done, alignment=Qt.AlignmentFlag.AlignRight)

    def _mode(self, mode):
        self.canvas.curve["interpolation"] = mode
        self.canvas._emit()
