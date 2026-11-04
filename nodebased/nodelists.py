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
            painter.setBrush(QColor("#f2c94c"))
            for _frame, point in self.key_points(index):
                painter.drawPolygon(QPolygonF([point + QPointF(0, -6), point + QPointF(5, 0),
                                               point + QPointF(0, 6), point + QPointF(-5, 0)]))
        painter.setPen(QPen(QColor("#58d7ff"), 2))
        x = self.frame_x(self.current)
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))

    def mousePressEvent(self, event):
        position = event.position()
        row = int((position.y() - RULER_HEIGHT) // ROW_HEIGHT)
        if position.x() < NAME_WIDTH:
            if 0 <= row < len(self.rows):
                self.window.show_node(self.rows[row][0])
            return
        self.frame_chosen.emit(self.frame_at(position.x()))
