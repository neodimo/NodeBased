"""Small editable graph for serialised colour curves."""
from __future__ import annotations

import ast
import json
import math

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
                               QMenu, QDoubleSpinBox, QLineEdit, QListWidget, QAbstractItemView, QFormLayout,
                               QDialogButtonBox)

from . import colorcurves
from .colorcurves import decode, encode, evaluate


INTERPOLATION_MENU = ("Constant", "Linear", "Smooth", "Catmull-Rom", "Cubic", "Horizontal", "Break")


def cubic_slopes(points):
    """Slopes of the natural cubic spline through the keys: the second derivative is continuous."""
    n = len(points)
    if n < 3:
        chord = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
        return [chord] * n
    x = [p[0] for p in points]; y = [p[1] for p in points]
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    a, b, c, r = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    b[0], c[0], r[0] = 2.0, 1.0, 3.0 * d[0]
    for i in range(1, n - 1):
        a[i], b[i], c[i] = h[i], 2.0 * (h[i - 1] + h[i]), h[i - 1]
        r[i] = 3.0 * (h[i] * d[i - 1] + h[i - 1] * d[i])
    a[-1], b[-1], r[-1] = 1.0, 2.0, 3.0 * d[-1]
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]; r[i] -= m * r[i - 1]
    m_out = [0.0] * n
    m_out[-1] = r[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        m_out[i] = (r[i] - c[i] * m_out[i + 1]) / b[i]
    return m_out


# -- small expressions of x and y (the Move dialog and the expression field) --------------------------------------------

_FUNCTIONS = {"abs": abs, "min": min, "max": max, "pow": pow, "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
              "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil,
              "clamp": lambda v, lo=0.0, hi=1.0: min(max(v, lo), hi), "lerp": lambda a, b, t: a + (b - a) * t,
              "smoothstep": lambda a, b, v: (lambda t: t * t * (3 - 2 * t))(min(max((v - a) / ((b - a) or 1), 0.0), 1.0))}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_BINARY = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b}


def curve_expression(text):
    """A function f(x, y) for an expression such as `x+10`, `y*2` or `pow(x, 2.2)`: numbers, x, y, pi, e,
    + - * / ** %, and a few functions. Raises ValueError on anything else, before it is ever evaluated."""
    try:
        tree = ast.parse(str(text).strip().replace("^", "**"), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"cannot read {text!r}") from exc

    def check(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return
        if isinstance(node, ast.Name) and (node.id in ("x", "y") or node.id in _CONSTANTS):
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            check(node.left); check(node.right); return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            check(node.operand); return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
            for arg in node.args: check(arg)
            return
        raise ValueError(f"{text!r} may use numbers, x, y and {', '.join(sorted(_FUNCTIONS))}")

    check(tree)

    def run(node, x, y):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return {"x": x, "y": y, **_CONSTANTS}[node.id]
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](run(node.left, x, y), run(node.right, x, y))
        if isinstance(node, ast.UnaryOp):
            value = run(node.operand, x, y)
            return -value if isinstance(node.op, ast.USub) else value
        return float(_FUNCTIONS[node.func.id](*(run(a, x, y) for a in node.args)))

    def evaluate_at(x, y=0.0):
        value = float(run(tree, float(x), float(y)))
        if not math.isfinite(value):
            raise ValueError(f"{text!r} is not finite at x {x:g}")
        return value
    return evaluate_at


EXPRESSION_SAMPLES = 17


class CurveCanvas(QWidget):
    changed = Signal(str)
    readout = Signal(str)
    selection = Signal(int)

    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.curve = decode(value)
        self.selected = 0
        self.chosen = {0}          # every selected key; `selected` is the one the fields show
        self.marquee = None
        self.overlays = []         # [(curve, QColor)]: other knobs' curves shown with this one (Nuke's parameter tree)
        self.snap_frames = False   # a keyframe curve's x is a frame and snaps to whole frames; a lookup's x does not
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

    def select(self, index, add=False):
        """Select key `index`; with `add` (Shift+click) it toggles in the selection kept so far."""
        index = max(0, min(index, len(self.curve["points"]) - 1))
        if add:
            self.chosen ^= {index}
            if not self.chosen:
                self.chosen = {index}
        else:
            self.chosen = {index}
        self.selected = index if index in self.chosen else min(self.chosen)
        self.selection.emit(self.selected)
        self.update()

    def select_all(self):
        """Ctrl/Cmd+A: select every key."""
        self.chosen = set(range(len(self.curve["points"])))
        self.selection.emit(self.selected)
        self.update()

    def keys_in(self, rect):
        rect = rect.normalized()
        return {i for i, (x, y) in enumerate(self.curve["points"]) if rect.contains(self._xy(x, y))}

    def move_chosen(self, starts, dx, dy):
        """Move every selected key by (dx, dy) from its position in `starts`, keeping their x order; whole frames
        when the canvas shows a keyframe curve."""
        order = sorted(starts, reverse=dx > 0)
        points = self.curve["points"]
        for i in order:
            x, y = starts[i][0] + dx, starts[i][1] + dy
            if self.snap_frames:   # whole frames: a key stays at least one frame from its neighbours
                if i > 0: x = max(x, points[i - 1][0] + 1)
                if i + 1 < len(points): x = min(x, points[i + 1][0] - 1)
            colorcurves.move_point(self.curve, i, x, y)

    def move_keys(self, x_text, y_text):
        """Edit > Move: set each selected key's x and y from expressions of its own x and y (`x+10`, `y*2`).
        One curve change, so one undo step. Raises ValueError and changes nothing on a bad expression."""
        fx = curve_expression(x_text or "x")
        fy = curve_expression(y_text or "y")
        moved = {i: (fx(*self.curve["points"][i]), fy(*self.curve["points"][i])) for i in self.chosen}
        right = sum(nx - self.curve["points"][i][0] for i, (nx, _ny) in moved.items()) > 0
        for i in sorted(moved, reverse=right):
            nx, ny = moved[i]
            if self.snap_frames:
                nx = round(nx)
            colorcurves.move_point(self.curve, i, nx, ny)
        self._emit()

    def set_expression(self, text):
        """The expression field: the curve becomes `text` (an expression of x) over the curve's own x range, sampled
        into keys whose tangents follow the expression. Raises ValueError and changes nothing on a bad expression."""
        f = curve_expression(text)
        lo, hi = self.curve["points"][0][0], self.curve["points"][-1][0]
        n = EXPRESSION_SAMPLES
        xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        ys = [f(x) for x in xs]
        h = (hi - lo) / (n - 1) / 4 or 1e-4
        slopes = [(f(x + h) - f(x - h)) / (2 * h) for x in xs]
        self.curve = decode(encode(list(zip(xs, ys)), "smooth", [[d, d] for d in slopes], ["broken"] * n, [False] * n))
        self.chosen, self.selected = {0}, 0
        self.selection.emit(0)
        self._emit()

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
        for other, colour in self.overlays:     # the other curves picked in the list, drawn thin under this one
            p.setPen(QPen(colour, 1))
            trace = [self._xy(xmin + (xmax - xmin) * i / 120, evaluate(other, xmin + (xmax - xmin) * i / 120))
                     for i in range(121)]
            for a, b in zip(trace, trace[1:]): p.drawLine(a, b)
        path = []
        for i in range(201):
            x = xmin + (xmax - xmin) * i / 200
            path.append(self._xy(x, evaluate(self.curve, x)))
        p.setPen(QPen(QColor("#73cdb5"), 2))
        for a, b in zip(path, path[1:]): p.drawLine(a, b)
        for i, (x, y) in enumerate(self.curve["points"]):
            q = self._xy(x, y)
            for side in (0, 1):
                if not self._has_handle(i, side):
                    continue
                h = self._handle_pos(i, side)
                p.setPen(QPen(QColor("#8395bd"), 1)); p.drawLine(q, h)
                p.setBrush(QColor("#aab8df")); p.drawEllipse(h, 3, 3)
            p.setPen(QPen(QColor("#ffffff"), 1)); p.setBrush(QColor("#e1a75e" if i == self.drag or i in self.chosen else "#82d7bf")); p.drawEllipse(q, 5, 5)
        if self.marquee is not None:
            p.setPen(QPen(QColor("#e1a75e"), 1, Qt.PenStyle.DashLine)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(self.marquee[0], self.marquee[1]).normalized())
        p.setPen(QColor("#aaaab5")); p.drawText(3, 15, f"{xmin:g}"); p.drawText(3, self.height()-8, f"{ymin:g} .. {ymax:g}")

    def _has_handle(self, index, side):
        if self.snap_frames:     # a keyframe curve stores one interpolation per curve, no tangents
            return False
        return not ((index == 0 and side == 0) or (index == len(self.curve["points"]) - 1 and side == 1))

    def mousePressEvent(self, e):
        mods = e.modifiers()
        ctrl, alt = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.AltModifier
        if e.button() == Qt.MouseButton.MiddleButton or (
                e.button() == Qt.MouseButton.LeftButton and mods & alt and not mods & ctrl):
            self.pan = e.position(); return
        if e.button() == Qt.MouseButton.LeftButton and mods & ctrl and mods & alt:
            self.add_key_at(e.position()); return
        distances = [(self._xy(x, y) - e.position()).manhattanLength() for x, y in self.curve["points"]]
        index = min(range(len(distances)), key=distances.__getitem__)
        if e.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            edit = menu.addMenu("Edit")
            edit.addAction("Move…", self.open_move_dialog)
            edit.addAction("Select all", self.select_all)
            if distances[index] < 16:
                if index not in self.chosen:
                    self.select(index)
                # Nuke 17.0's Curve Editor Interpolation menu, in its order.
                for label in INTERPOLATION_MENU:
                    menu.addAction(label, lambda checked=False, m=label: self.set_interpolation(index, m))
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
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if e.button() == Qt.MouseButton.LeftButton and distances[index] < 16:
            if shift:
                self.select(index, add=True)      # Shift+click adds or removes a key; no drag starts
                return
            if index not in self.chosen:
                self.select(index)
            else:
                self.selected = index; self.selection.emit(index)
            self.drag = index
            self.drag_start = (e.position(), tuple(self.curve["points"][index]),
                               {i: tuple(self.curve["points"][i]) for i in self.chosen}, self._value(e.position()))
            self.readout.emit(self._readout(index))
            return
        if e.button() == Qt.MouseButton.LeftButton:   # a drag on empty space draws a marquee
            self.marquee = (e.position(), e.position(), shift)

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
        if self.marquee is not None:
            self.marquee = (self.marquee[0], e.position(), self.marquee[2]); self.update(); return
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
        start = getattr(self, "drag_start", None)
        if start is not None:
            px, py = start[3]
            dx, dy = x - px, y - py
            if not e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Nuke: a plain drag moves the key along x or y only (the larger motion wins); Ctrl+drag is free.
                delta = e.position() - start[0]
                if abs(delta.x()) >= abs(delta.y()):
                    dy = 0.0
                else:
                    dx = 0.0
            if self.snap_frames and not e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Nuke: a moved key's x is rounded to the nearest whole frame; Shift turns that off.
                dx = round(start[1][0] + dx) - start[1][0]
            self.move_chosen(start[2], dx, dy)
            x, y = self.curve["points"][self.drag]
        x, y = colorcurves.move_point(self.curve, self.drag, x, y)
        self.readout.emit(f"x {x:.6g}   y {y:.6g}")
        self.selection.emit(self.drag)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.marquee is not None:
            start, _end, shift = self.marquee
            self.marquee = None
            end = e.position()
            if abs(end.x() - start.x()) > 3 or abs(end.y() - start.y()) > 3:
                inside = self.keys_in(QRectF(start, end))
                chosen = (self.chosen ^ inside) if shift else inside
                if chosen:
                    self.chosen = chosen
                    self.selected = min(chosen)
                    self.selection.emit(self.selected)
            self.update()
            return
        if self.drag is not None:
            self.drag = None
            if hasattr(self, "drag_handle"):
                del self.drag_handle
            self.drag_start = None
            self._emit()
        self.pan = None

    def mouseDoubleClickEvent(self, e):
        self.add_key_at(e.position(), on_curve=True)

    def add_key_at(self, position, on_curve=False):
        """Add a key at the pointer (Nuke's Ctrl/Cmd+Alt+click, which also adds off the curve)."""
        x, y = self._value(position); pts = self.curve["points"]
        if not pts[0][0] < x < pts[-1][0] or any(abs(p[0] - x) < 1e-5 for p in pts):
            return
        # The new key sits on the curve with the curve's own slope: adding it never moves the curve.
        index = colorcurves.insert_key(self.curve, x)
        if not on_curve:
            colorcurves.move_point(self.curve, index, x, y)
        self.select(index)
        self._emit()

    def set_interpolation(self, index, label):
        """One entry of the right-click Interpolation menu on key `index` (Nuke 17.0 User Guide)."""
        self._ensure_handles()
        points, n = self.curve["points"], len(self.curve["points"])
        if label in ("Constant", "Linear"):
            self._set_mode(index, label.lower()); return
        if label == "Break":
            for j in (index - 1, index):
                if 0 <= j < n - 1:
                    colorcurves.bake_segment(self.curve, j)
            self.curve["broken"][index] = True
            self._emit(); return
        if label == "Horizontal":
            slope = 0.0
        elif label == "Catmull-Rom":
            slope = colorcurves._auto_slope(points, index)
        elif label == "Cubic":
            slope = cubic_slopes(points)[index]
        else:  # Smooth: the neighbours' slope, flat when the key is above or below both neighbours.
            lo, hi = points[max(index - 1, 0)][1], points[min(index + 1, n - 1)][1]
            y = points[index][1]
            between = min(lo, hi) <= y <= max(lo, hi) and 0 < index < n - 1
            slope = colorcurves._auto_slope(points, index) if between else 0.0
        for j in (index - 1, index):
            if 0 <= j < n - 1:
                colorcurves.bake_segment(self.curve, j)
        self.curve["broken"][index] = False
        self.curve["slopes"][index] = [float(slope), float(slope)]
        self._emit()

    def frame_all(self):
        """Nuke's View > Frame All (A): reset zoom and pan so the whole curve shows."""
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.update()

    def frame_selected(self):
        """Nuke's View > Frame Selected (F): centre the selected key."""
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        xmin, xmax, ymin, ymax = self._bounds()
        x, y = self.curve["points"][self.selected]
        self.offset = QPointF(x - (xmin + xmax) / 2, y - (ymin + ymax) / 2)
        self.update()

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.select_all(); return
        if key == Qt.Key.Key_A and not e.modifiers():
            self.frame_all(); return
        if key == Qt.Key.Key_F and not e.modifiers():
            self.frame_selected(); return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom *= 1.25; self.update(); return
        if key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.zoom /= 1.25; self.update(); return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_chosen(); return
        super().keyPressEvent(e)

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

    def delete_chosen(self):
        """Delete every selected key (a curve keeps at least two), one undo step."""
        for index in sorted(self.chosen, reverse=True):
            colorcurves.delete_key(self.curve, index)
        self.select(min(min(self.chosen), len(self.curve["points"]) - 1))
        self._emit()

    def open_move_dialog(self):
        """Edit > Move: Nuke's Move Animation Keys dialog, x and y as expressions of each key's own x and y."""
        dialog = MoveKeysDialog(self)
        self.move_dialog = dialog
        dialog.show()
        return dialog

    def wheelEvent(self, e):
        self.zoom *= 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        self.setToolTip(f"Zoom {self.zoom:.2f}× · middle-drag to pan"); self.update()


class MoveKeysDialog(QDialog):
    """Move Animation Keys: type `x+10` to shift the selected keys ten along x, `y*2` to double their values."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self.setWindowTitle("Move Animation Keys")
        self.canvas = canvas
        form = QFormLayout(self)
        self.x_field, self.y_field = QLineEdit("x"), QLineEdit("y")
        self.x_field.setObjectName("move-keys-x"); self.y_field.setObjectName("move-keys-y")
        form.addRow("x", self.x_field); form.addRow("y", self.y_field)
        self.error = QLabel(""); form.addRow(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.setObjectName("move-keys-buttons")
        buttons.accepted.connect(self.apply); buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def apply(self):
        try:
            self.canvas.move_keys(self.x_field.text(), self.y_field.text())
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            self.error.setText(str(exc)); return
        self.accept()


CURVE_COLOURS = ("#d8d8d8", "#ff6b6b", "#7bd88f", "#6fa8ff", "#c9c9c9", "#e1a75e")


class CurveEditorDialog(QDialog):
    def __init__(self, title, value, on_change, parent=None, default=None, curves=None, current=None):
        """`curves` lists the node's curve knobs as (param, label, read value, on_change, default): the list beside the
        canvas picks which show (Shift for a run, Ctrl/Cmd for one more, as Nuke's parameter tree); the current one edits."""
        super().__init__(parent); self.setWindowTitle(title); self.resize(520, 380)
        self.default = default
        self.on_change = on_change
        self.curves = list(curves or [])
        self._before_expression = None
        layout = QVBoxLayout(self)
        self.canvas = CurveCanvas(value)
        self.curve_list = QListWidget(); self.curve_list.setObjectName("curve-list")
        self.curve_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.curve_list.setMaximumWidth(130)
        for _param, label, *_rest in self.curves:
            self.curve_list.addItem(label)
        body = QHBoxLayout(); body.addWidget(self.curve_list); body.addWidget(self.canvas, 1)
        self.curve_list.setVisible(len(self.curves) > 1)
        layout.addLayout(body)
        tools = QHBoxLayout(); tools.addWidget(QLabel("Interpolation"))
        self.mode = QComboBox(); self.mode.addItems(("linear", "smooth")); self.mode.setCurrentText(self.canvas.curve["interpolation"]); tools.addWidget(self.mode)
        tools.addWidget(QLabel("Ctrl+Alt+click: add · right-click: interpolation/delete · drag: move on one axis (Ctrl: free) · A/F: frame"))
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
        # Nuke: an expression field at the bottom of the Curve Editor sets the curve; Revert puts the keys back.
        expression = QHBoxLayout(); expression.addWidget(QLabel("Expression"))
        self.expression = QLineEdit(); self.expression.setObjectName("curve-expression")
        self.expression.setPlaceholderText("an expression of x, e.g. pow(x, 1/2.2)")
        self.expression.returnPressed.connect(self.apply_expression)
        self.revert = QPushButton("Revert"); self.revert.setObjectName("curve-expression-revert")
        self.revert.clicked.connect(self.revert_expression); self.revert.setEnabled(False)
        expression.addWidget(self.expression, 1); expression.addWidget(self.revert); layout.addLayout(expression)
        self.canvas.selection.connect(lambda _index: self._sync_fields())
        self.canvas.changed.connect(lambda _text: self._sync_fields())
        self.point_x.editingFinished.connect(self._point_edited)
        self.point_y.editingFinished.connect(self._point_edited)
        self.slope_in.editingFinished.connect(lambda: self._slope_edited(0, self.slope_in))
        self.slope_out.editingFinished.connect(lambda: self._slope_edited(1, self.slope_out))
        self.canvas.changed.connect(lambda text: self.on_change(text))
        self.mode.currentTextChanged.connect(self._mode)
        if len(self.curves) > 1:
            names = [c[0] for c in self.curves]
            row = names.index(current) if current in names else 0
            self.curve_list.setCurrentRow(row)
            self.curve_list.itemSelectionChanged.connect(self._curves_picked)
            self.curve_list.currentRowChanged.connect(lambda _row: self._curves_picked())
            self._active = names[row]
        done = QPushButton("Close"); done.clicked.connect(self.accept); layout.addWidget(done, alignment=Qt.AlignmentFlag.AlignRight)
        self._sync_fields()

    def shown_curves(self):
        """The params of the curves the list shows, in list order."""
        rows = sorted(self.curve_list.row(item) for item in self.curve_list.selectedItems())
        return [self.curves[r][0] for r in rows]

    def _curves_picked(self):
        """The list's current row is the curve the canvas edits; every other selected row is drawn under it."""
        row = self.curve_list.currentRow()
        if not 0 <= row < len(self.curves):
            return
        param, _label, read, on_change, default = self.curves[row]
        if param != getattr(self, "_active", None):
            self._active = param
            self.on_change, self.default = on_change, default
            self.canvas.curve = decode(read())
            self.canvas.chosen, self.canvas.selected = {0}, 0
            self._before_expression = None; self.revert.setEnabled(False); self.expression.clear()
            self.mode.blockSignals(True); self.mode.setCurrentText(self.canvas.curve["interpolation"]); self.mode.blockSignals(False)
        self.canvas.overlays = [(decode(self.curves[r][2]()), QColor(CURVE_COLOURS[r % len(CURVE_COLOURS)]))
                                for r in sorted(self.curve_list.row(i) for i in self.curve_list.selectedItems()) if r != row]
        self._sync_fields()
        self.canvas.update()

    def apply_expression(self):
        before = encode(self.canvas.curve["points"], self.canvas.curve["interpolation"], self.canvas.curve.get("slopes"),
                        self.canvas.curve.get("modes"), self.canvas.curve.get("broken"))
        try:
            self.canvas.set_expression(self.expression.text())
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            self.expression.setToolTip(str(exc)); self.expression.setStyleSheet("color: #ff6b6b"); return
        self.expression.setToolTip(""); self.expression.setStyleSheet("")
        if self._before_expression is None:
            self._before_expression = before
        self.revert.setEnabled(True)

    def revert_expression(self):
        if self._before_expression is None:
            return
        self.canvas.reset(self._before_expression)
        self._before_expression = None
        self.expression.clear(); self.revert.setEnabled(False)

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
