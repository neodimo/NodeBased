"""The "?" overlay of the node graph (new look, step 5, Lane 2).

The graph used to carry its shortcuts as one long line of text above it. The mockup has no such
line; the same text now opens from the "?" button in the graph's corner and from the ? key, as a
small frosted panel over the graph. It closes with Escape, with ? again, or with a click outside.

`GRAPH_SHORTCUTS` is the old line, item by item: each pair is (keys, what they do), and
`keys + " " + what` is the text the line showed. `graph_shortcut_text()` joins them the way the
line did, so a test can hold the overlay to every shortcut the line listed.
"""
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

SEPARATOR = "  ·  "
GRAPH_SHORTCUTS = (
    ("Tab", "search/add"),
    ("R/G/M/T/B/C/S/O/P/U/W", "create"),
    ("Period", "Dot"),
    ("1", "view"),
    ("D", "bypass"),
    ("F", "frame"),
    ("Ctrl+A", "select all"),
    ("Ctrl+C/X/V", "copy/cut/paste"),
    ("Alt+C", "duplicate"),
    ("Ctrl+G", "group / Ctrl+Shift+G ungroup"),
    ("MMB or Alt+drag", "pan"),
    ("drag output ↔ input", "to wire"),
    ("Ctrl-drag noodle midpoint", "inserts Dot"),
    ("click a wired input", "to rewire"),
)
COLUMNS = 2


def graph_shortcut_items():
    """Every shortcut of the old line, as the line wrote it."""
    return [f"{keys} {what}" for keys, what in GRAPH_SHORTCUTS]


def graph_shortcut_text():
    return SEPARATOR.join(graph_shortcut_items())


class ShortcutOverlay(QFrame):
    """A frosted panel with every graph shortcut, in two columns: the keys as a keycap, then what
    they do. A popup window, so Escape and a click elsewhere close it without any bookkeeping."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("shortcut-overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel("Node graph shortcuts")
        title.setObjectName("shortcut-title")
        close = QLabel("? or Esc to close")
        close.setObjectName("shortcut-hint")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(close)
        outer.addLayout(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        rows = -(-len(GRAPH_SHORTCUTS) // COLUMNS)
        self.rows = []
        for index, (keys, what) in enumerate(GRAPH_SHORTCUTS):
            column, row = divmod(index, rows)
            keycap = QLabel(keys)
            keycap.setObjectName("shortcut-keys")
            action = QLabel(what)
            action.setObjectName("shortcut-action")
            grid.addWidget(keycap, row, column * 3)
            grid.addWidget(action, row, column * 3 + 1)
            if column + 1 < COLUMNS:
                grid.setColumnMinimumWidth(column * 3 + 2, 18)
            self.rows.append((keycap, action))
        outer.addLayout(grid)

    def text(self):
        """Everything the overlay lists, joined like the line it replaced."""
        return SEPARATOR.join(f"{keys.text()} {what.text()}" for keys, what in self.rows)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Question) or event.text() == "?":
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def show_above(self, widget):
        """Open with its bottom-right corner just above `widget`'s top-right corner."""
        self.adjustSize()
        anchor = widget.mapToGlobal(QPoint(widget.width(), 0))
        self.move(anchor.x() - self.width(), anchor.y() - self.height() - 8)
        self.show()
        self.raise_()
        self.setFocus()

    def event(self, event):
        if event.type() == QEvent.Type.ShortcutOverride and event.text() == "?":
            event.accept()
        return super().event(event)
