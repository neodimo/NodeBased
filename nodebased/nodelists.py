"""The two lists the Node tab feeds: the bookmark list and the Dope Sheet.

Nuke's Node tab has a "bookmark" box ("show the node in the bookmark list, so you can quickly navigate
to it") and a "dope sheet" box ("force the node to always display in the Dope Sheet"). These panels are
those lists. Both read the document on screen and never change it, apart from moving the playhead.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from .core import bookmarked_nodes, dope_sheet_rows

KEY_ROLE = Qt.ItemDataRole.UserRole
ROW_HEIGHT = 24
NAME_WIDTH = 170
RULER_HEIGHT = 22


class BookmarkPanel(QWidget):
    """The nodes whose Node tab has Bookmark on, by name; choosing one selects it and frames it."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.hint = QLabel("Tick Bookmark on a node's Node tab to list it here.")
        self.hint.setObjectName("bookmark-hint")
        self.hint.setWordWrap(True)
        self.list = QListWidget()
        self.list.setObjectName("bookmark-list")
        self.list.itemClicked.connect(self.go)
        self.list.itemActivated.connect(self.go)
        layout.addWidget(self.hint)
        layout.addWidget(self.list, 1)
        self.entries = []

    def refresh(self):
        entries = bookmarked_nodes(self.window.graph_nodes())
        if entries == self.entries:
            return
        self.entries = entries
        self.list.clear()
        for key, name in entries:
            item = QListWidgetItem(name)
            item.setData(KEY_ROLE, key)
            self.list.addItem(item)
        self.hint.setVisible(not entries)

    def go(self, item):
        self.window.show_node(item.data(KEY_ROLE))


class DopeSheet(QWidget):
    """One row per node (see core.dope_sheet_rows), a diamond on every frame it has a key. Clicking the
    time area moves the playhead; clicking a name selects the node and frames it in the Node Graph."""

    frame_chosen = Signal(int)

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setObjectName("dope-sheet")
        self.setMinimumHeight(RULER_HEIGHT + ROW_HEIGHT)
        self.rows = []
        self.first, self.last, self.current = 1, 100, 1
        self.selected = set()   # {(node key, frame)}: Nuke selects keys, not whole rows
        self.drag = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def refresh(self):
        window = self.window
        document = window.graph_document()
        opened = set(getattr(window, "pinned_panels", []))
        selected = window.graph.selected_id() if getattr(window, "graph", None) is not None else None
        if selected:
            opened.add(selected)
        curves = (document.get("animation") or {}).get("curves") or {}
        rows = dope_sheet_rows(window.graph_nodes(), curves, opened)
        time_range = window.dispatcher.document["time"]
        state = (rows, time_range["first"], time_range["last"], time_range["current"])
        if state == (self.rows, self.first, self.last, self.current):
            return
        self.rows = rows
        self.first, self.last, self.current = state[1:]
        present = {(key, frame) for key, _name, frames in rows for frame in frames}
        self.selected &= present
        self.setMinimumHeight(RULER_HEIGHT + ROW_HEIGHT * max(1, len(rows)))
        self.update()

    def frame_x(self, frame):
        span = max(1, self.last - self.first)
        width = max(1, self.width() - NAME_WIDTH - 16)
        return NAME_WIDTH + 8 + (frame - self.first) / span * width

    def frame_at(self, x):
        span = max(1, self.last - self.first)
        width = max(1, self.width() - NAME_WIDTH - 16)
        return int(round(min(max(self.first + (x - NAME_WIDTH - 8) / width * span, self.first), self.last)))

    def key_points(self, row_index):
        """Scene positions of the diamonds on row `row_index`, for tests and hit testing."""
        y = RULER_HEIGHT + row_index * ROW_HEIGHT + ROW_HEIGHT / 2
        return [(frame, QPointF(self.frame_x(frame), y)) for frame in self.rows[row_index][2]]

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        text, faint = palette.text().color(), QColor(palette.text().color())
        faint.setAlpha(60)
        step = max(1, (self.last - self.first) // 10)
        painter.setPen(text)
        for frame in range(self.first, self.last + 1, step):
            x = self.frame_x(frame)
            painter.drawText(QRectF(x - 24, 0, 48, RULER_HEIGHT), Qt.AlignmentFlag.AlignCenter, str(frame))
            painter.setPen(QPen(faint, 1))
            painter.drawLine(QPointF(x, RULER_HEIGHT), QPointF(x, self.height()))
            painter.setPen(text)
        for index, (_key, name, frames) in enumerate(self.rows):
            top = RULER_HEIGHT + index * ROW_HEIGHT
            painter.setPen(text)
            painter.drawText(QRectF(8, top, NAME_WIDTH - 12, ROW_HEIGHT),
                             int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), name)
            painter.setPen(QPen(faint, 1))
            painter.drawLine(QPointF(0, top + ROW_HEIGHT), QPointF(self.width(), top + ROW_HEIGHT))
            painter.setPen(QPen(QColor("#1c1c1e"), 1))
            for frame, point in self.key_points(index):
                chosen = (self.rows[index][0], frame) in self.selected
                painter.setBrush(QColor("#ff9a2e" if chosen else "#f2c94c"))
                painter.drawPolygon(QPolygonF([point + QPointF(0, -6), point + QPointF(5, 0),
                                               point + QPointF(0, 6), point + QPointF(-5, 0)]))
                if chosen:   # Nuke shows the frame number next to a selected key
                    painter.setPen(text)
                    painter.drawText(QRectF(point.x() + 7, top, 48, ROW_HEIGHT),
                                     int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), str(frame))
                    painter.setPen(QPen(QColor("#1c1c1e"), 1))
        painter.setPen(QPen(QColor("#58d7ff"), 2))
        x = self.frame_x(self.current)
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))

    # -- editing keys (Nuke 17.0 User Guide, Using the Dope Sheet) -----------------------------------------------

    def _curves(self):
        return (self.window.graph_document().get("animation") or {}).get("curves") or {}

    def key_at(self, position):
        """(row index, frame) of the diamond under `position`, or None."""
        row = int((position.y() - RULER_HEIGHT) // ROW_HEIGHT)
        if position.y() < RULER_HEIGHT or not 0 <= row < len(self.rows):
            return None
        hits = [(abs(point.x() - position.x()), frame) for frame, point in self.key_points(row)
                if abs(point.x() - position.x()) <= 7]
        return (row, min(hits)[1]) if hits else None

    def delete_selected(self):
        """Delete/Backspace: erase every selected key (all of the node's knobs keyed on that frame), one undo step."""
        curves = self._curves()
        commands = [{"op": "delete_key", "id": key, "param": param, "frame": frame}
                    for key, frame in sorted(self.selected)
                    for param, curve in sorted((curves.get(key) or {}).items())
                    if any(k["frame"] == frame for k in curve["keys"])]
        self.selected = set()
        if commands:
            self.window.command({"op": "batch", "commands": commands})
        self.update()

    def add_key(self, row, frame):
        """Ctrl/Cmd+Alt+click on an empty spot: key every animated knob of the row's node at its current value."""
        from .animation import evaluate_curve
        key = self.rows[row][0]
        commands = [{"op": "set_key", "id": key, "param": param, "frame": int(frame),
                     "value": float(evaluate_curve(curve, frame))}
                    for param, curve in sorted((self._curves().get(key) or {}).items())]
        if commands:
            self.selected = {(key, int(frame))}
            self.window.command({"op": "batch", "commands": commands})

    def move_selected(self, delta):
        """Move the selected keys `delta` frames, one undo step. A key never moves past another key."""
        if not delta or not self.selected:
            return 0
        curves = self._curves()
        by_node = {}
        for key, frame in self.selected:
            by_node.setdefault(key, set()).add(frame)
        lo, hi = -10**9, 10**9
        for key, frames in by_node.items():
            every = sorted({k["frame"] for curve in (curves.get(key) or {}).values() for k in curve["keys"]})
            for frame in frames:
                i = every.index(frame)
                before = [f for f in every[:i] if f not in frames]
                after = [f for f in every[i + 1:] if f not in frames]
                if before: lo = max(lo, before[-1] + 1 - frame)
                if after: hi = min(hi, after[0] - 1 - frame)
        delta = int(min(max(delta, lo), hi))
        if not delta:
            return 0
        commands = []
        for key, frames in sorted(by_node.items()):
            for param, curve in sorted((curves.get(key) or {}).items()):
                if not any(k["frame"] in frames for k in curve["keys"]):
                    continue
                keys = sorted(({"frame": k["frame"] + delta if k["frame"] in frames else k["frame"], "value": k["value"]}
                               for k in curve["keys"]), key=lambda k: k["frame"])
                commands.append({"op": "set_curve", "id": key, "param": param,
                                 "curve": {"interpolation": curve["interpolation"], "keys": keys}})
        self.selected = {(key, frame + delta) for key, frame in self.selected}
        self.window.command({"op": "batch", "commands": commands})
        return delta

    def mousePressEvent(self, event):
        position = event.position()
        self.setFocus()
        row = int((position.y() - RULER_HEIGHT) // ROW_HEIGHT)
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if position.x() < NAME_WIDTH:
            if 0 <= row < len(self.rows):
                key, _name, frames = self.rows[row]
                # Nuke: clicking the node name selects all the keyframes under it.
                self.selected = {(key, frame) for frame in frames} | (self.selected if shift else set())
                self.window.show_node(key)
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self.key_at(position)
        if hit is not None:
            pick = (self.rows[hit[0]][0], hit[1])
            if shift:
                self.selected ^= {pick}
            elif pick not in self.selected:
                self.selected = {pick}
            self.drag = (position, self.frame_at(position.x()), 0)
            self.update()
            return
        if (mods & Qt.KeyboardModifier.ControlModifier and mods & Qt.KeyboardModifier.AltModifier
                and 0 <= row < len(self.rows)):
            self.add_key(row, self.frame_at(position.x()))
            return
        if not shift:
            self.selected = set()
        self.frame_chosen.emit(self.frame_at(position.x()))

    def mouseMoveEvent(self, event):
        if self.drag is None:
            return
        start, frame, _moved = self.drag
        self.drag = (start, frame, self.frame_at(event.position().x()) - frame)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.drag is None:
            return
        _start, frame, _moved = self.drag
        self.drag = None
        self.move_selected(self.frame_at(event.position().x()) - frame)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            return
        super().keyPressEvent(event)
