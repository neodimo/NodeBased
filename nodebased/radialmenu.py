"""RadialMenu: the graph's context-sensitive marking menu (plan "Radial menu", DiMo 9/27).

Hold Q over the graph to open a ring of eight slices at the cursor. Flick past the dead zone
toward a slice and let go of Q to run it, marking-menu style: only the direction matters, not
where exactly the pointer lands. Release Q back inside the dead zone (a tap, no flick) and the
menu stays open, sustained, for reading the labels and clicking one; Esc, or a click that lands
back in the dead zone, cancels it instead. `Graph` in `app.py` owns the Q key and mouse routing;
this module is the geometry (`slot_for_offset`) and the paint (`RadialMenu`) only, in physical
viewport pixels so the ring is the same size at any zoom -- see `radialrules.py` for what fills
the eight slices for a given selection.

The ring itself also carries one always-there "+ Add command..." button (deliverable R2, DiMo
9/27): a real child widget, not a ring slice, so it never competes with a context's eight slots
for space. It only shows once the menu is sustained (pinned open for reading/clicking) -- during
a live flick it would just be one more thing under the pointer.
"""
import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QPushButton, QWidget

from .radialrules import SLOT_COUNT

DEAD_ZONE_RADIUS = 26.0   # physical pixels: a release/click this close to centre commits to nothing
INNER_RADIUS = 34.0
OUTER_RADIUS = 84.0
LABEL_RADIUS = 108.0
_STEP_DEGREES = 360.0 / SLOT_COUNT


def _slot_center_angle(index):
    """Slot 0 points straight up; slots run clockwise from there, like a clock face."""
    return math.radians(-90 + index * _STEP_DEGREES)


def slot_for_offset(dx, dy):
    """The slot a pointer offset from the menu's centre points toward, or `None` inside the dead
    zone (no direction has been committed to yet)."""
    if math.hypot(dx, dy) < DEAD_ZONE_RADIUS:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    turned = (angle + 90.0) % 360.0
    return int((turned + _STEP_DEGREES / 2.0) // _STEP_DEGREES) % SLOT_COUNT


class RadialMenu(QWidget):
    """A screen-space overlay on the graph's viewport (a child of it, not a scene item), so its
    ring is the same physical size regardless of graph zoom. `Graph` drives it entirely: it never
    reads the mouse or keyboard itself."""

    add_command_requested = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.center = QPointF(0, 0)
        self.commands = [None] * SLOT_COUNT
        self.highlight = None
        self.sustained = False
        # A real child widget, not painted geometry: it needs its own mouse events, which
        # WA_TransparentForMouseEvents above denies this widget itself but not its children.
        self.add_command_button = QPushButton("+ Add command…", self)
        self.add_command_button.setObjectName("radial-add-command")
        self.add_command_button.hide()
        self.add_command_button.clicked.connect(self.add_command_requested.emit)
        self.hide()

    def is_open(self):
        return self.isVisible()

    def open_at(self, center, commands):
        """Show the ring at `center` (parent/viewport coordinates) with these eight slots."""
        self.center = QPointF(center)
        self.commands = list(commands)
        self.highlight = None
        self.sustained = False
        self.add_command_button.hide()
        span = int(OUTER_RADIUS + LABEL_RADIUS)
        self.setGeometry(int(self.center.x() - span), int(self.center.y() - span), span * 2, span * 2)
        self.show()
        self.raise_()
        self.update()

    def update_pointer(self, pos):
        """While Q is held: recompute the flick direction from the live pointer position."""
        offset = QPointF(pos) - self.center
        self.highlight = slot_for_offset(offset.x(), offset.y())
        self.update()

    def flicked(self):
        """True once the live gesture has moved past the dead zone toward a slot."""
        return self.highlight is not None

    def command_at_highlight(self):
        return self.commands[self.highlight] if self.highlight is not None else None

    def sustain(self):
        """Q released inside the dead zone with no flick: stay open for reading and clicking."""
        self.sustained = True
        self.highlight = None
        local_center = self.center - QPointF(self.x(), self.y())
        button = self.add_command_button
        button.adjustSize()
        button.move(int(local_center.x() - button.width() / 2),
                    int(local_center.y() + OUTER_RADIUS + LABEL_RADIUS * 0.5))
        button.show()
        button.raise_()
        self.update()

    def command_at(self, pos):
        """The command a click at `pos` (parent/viewport coordinates) resolves to, or `None`
        when the click lands in the dead zone or on an empty slot."""
        offset = QPointF(pos) - self.center
        slot = slot_for_offset(offset.x(), offset.y())
        return self.commands[slot] if slot is not None else None

    def close_menu(self):
        self.hide()
        self.add_command_button.hide()
        self.commands = [None] * SLOT_COUNT
        self.highlight = None
        self.sustained = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        local_center = self.center - QPointF(self.x(), self.y())
        painter.setPen(QPen(QColor("#5f5f6b"), 1.5))
        painter.setBrush(QBrush(QColor(25, 25, 27, 210)))
        painter.drawEllipse(local_center, OUTER_RADIUS, OUTER_RADIUS)
        painter.setBrush(QBrush(QColor(40, 40, 46, 235)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(local_center, INNER_RADIUS, INNER_RADIUS)
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        for index, command in enumerate(self.commands):
            if command is None:
                continue
            angle = _slot_center_angle(index)
            direction = QPointF(math.cos(angle), math.sin(angle))
            label_point = local_center + direction * LABEL_RADIUS
            if index == self.highlight:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#4c7ddc")))
                mid_point = local_center + direction * ((OUTER_RADIUS + INNER_RADIUS) / 2.0)
                painter.drawEllipse(mid_point, 22, 22)
            painter.setPen(QPen(QColor("#e8e8eb")))
            rect = QRectF(label_point.x() - 62, label_point.y() - 10, 124, 20)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, command.label)
