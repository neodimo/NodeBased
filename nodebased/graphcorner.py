"""The controls in the corner of the node graph (new look, step 2): the Curved / Right angle wire
toggle, the zoom control and the minimap, as in the mockup. Sizes come from font metrics."""
from PySide6.QtCore import QByteArray, QPoint, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QToolButton, QWidget

from . import graphlook
from .theme import RADIUS, TOKENS

MARGIN = 14       # distance from the graph's edges
GAP = 8           # between the three controls
CURVE_ICON = '<path d="M5 4c0 9 14 7 14 16"/>'
RIGHT_ANGLE_ICON = '<path d="M5 4v8h14v8"/>'


def token_color(value):
    """A QColor from a theme token, which may be "#rrggbb" or the CSS form "rgba(r, g, b, a)"."""
    if value.startswith("rgba("):
        r, g, b, a = [part.strip() for part in value[5:-1].split(",")]
        return QColor(int(r), int(g), int(b), round(float(a) * 255))
    return QColor(value)


def svg_icon(shapes, off_color, on_color, side=14):
    """A QIcon of a 24 x 24 glyph, `off_color` normally and `on_color` when checked."""
    icon = QIcon()
    for color, state in ((off_color, QIcon.State.Off), (on_color, QIcon.State.On)):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
               f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{shapes}</svg>')
        pixmap = QPixmap(side * 2, side * 2)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        QSvgRenderer(QByteArray(svg.encode())).render(painter, QRectF(0, 0, side * 2, side * 2))
        painter.end()
        pixmap.setDevicePixelRatio(2)
        icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
    return icon


def panel_button(text, tip):
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tip)
    button.setAutoRaise(True)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return button


class WireToggle(QFrame):
    """Curved / Right angle: two exclusive buttons; the graph keeps the choice in the settings."""

    def __init__(self, graph, parent):
        super().__init__(parent)
        self.graph = graph
        self.setObjectName("graph-panel")
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons = {}
        for mode, text, shapes, tip in (
                (graphlook.CURVED, "Curved", CURVE_ICON, "Draw wires as smooth curves"),
                (graphlook.RIGHT_ANGLE, "Right angle", RIGHT_ANGLE_ICON,
                 "Draw wires as horizontal and vertical runs with 90 degree turns")):
            button = panel_button(text, tip)
            button.setObjectName(f"wire-mode-{mode.replace('_', '-')}")
            button.setCheckable(True)
            button.setIcon(svg_icon(shapes, TOKENS["tx2"], TOKENS["acc"]))
            button.setIconSize(QSize(14, 14))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.clicked.connect(lambda checked=False, m=mode: self.graph.set_wire_mode(m))
            self.group.addButton(button)
            self.buttons[mode] = button
            row.addWidget(button)
        self.sync()

    def sync(self):
        for mode, button in self.buttons.items():
            button.setChecked(mode == self.graph.wire_mode)


class ZoomControl(QFrame):
    """[-] 100% [+]: the middle button returns to 100%."""

    def __init__(self, graph, parent):
        super().__init__(parent)
        self.graph = graph
        self.setObjectName("graph-panel")
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self.out = panel_button("−", "Zoom out")
        self.reset = panel_button("100%", "Zoom to 100%")
        self.into = panel_button("+", "Zoom in")
        self.out.setObjectName("graph-zoom-out")
        self.reset.setObjectName("graph-zoom-reset")
        self.into.setObjectName("graph-zoom-in")
        self.out.clicked.connect(lambda: graph.zoom_by(1 / 1.25))
        self.into.clicked.connect(lambda: graph.zoom_by(1.25))
        self.reset.clicked.connect(lambda: graph.zoom_to(1.0))
        for button in (self.out, self.reset, self.into):
            row.addWidget(button)
        self._text = None
        self.sync()

    def sync(self):
        text = f"{round(self.graph.transform().m11() * 100)}%"
        if text != self._text:
            self._text = text
            self.reset.setText(text)
            # The label is the widest at 4 digits; keep the box from jittering as the zoom changes.
            width = QFontMetrics(self.reset.font()).horizontalAdvance("0000%") + 18
            self.reset.setMinimumWidth(width)


class MiniMap(QWidget):
    """The whole graph in miniature: one small rectangle per node in its family colour, and the
    outline of what the graph view shows. Click or drag to move the view."""

    def __init__(self, graph, parent):
        super().__init__(parent)
        self.graph = graph
        self.setObjectName("graph-minimap")
        unit = QFontMetrics(self.font()).height()
        self.setFixedSize(unit * 9, unit * 5)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Overview of the graph: click or drag to move the view")

    def node_rects(self):
        """(scene rect, family colour, selected) for every node that takes part in the picture."""
        return [(item.sceneBoundingRect(), item.accent, item.isSelected())
                for item in self.graph.items_by_id.values() if not item.is_backdrop]

    def geometry_map(self):
        """(scale, offset) taking scene coordinates to widget coordinates."""
        view = self.graph.visible_scene_rect()
        bounds = QRectF(view)
        for rect, _color, _selected in self.node_rects():
            bounds = bounds.united(rect)
        pad = 6
        inner = QRectF(self.rect()).adjusted(pad, pad, -pad, -pad)
        scale = min(inner.width() / max(bounds.width(), 1.0), inner.height() / max(bounds.height(), 1.0))
        offset = QPointF(inner.center().x() - bounds.center().x() * scale,
                         inner.center().y() - bounds.center().y() * scale)
        return scale, offset

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(TOKENS["line2"]), 1))
        painter.setBrush(token_color(TOKENS["glass"]))
        painter.drawRoundedRect(frame, RADIUS["control"], RADIUS["control"])
        scale, offset = self.geometry_map()

        def mapped(rect):
            return QRectF(rect.x() * scale + offset.x(), rect.y() * scale + offset.y(),
                          max(rect.width() * scale, 3.0), max(rect.height() * scale, 2.0))

        painter.setPen(Qt.PenStyle.NoPen)
        for rect, color, selected in self.node_rects():
            painter.setBrush(color)
            painter.setPen(QPen(QColor(TOKENS["tx0"]), 1) if selected else Qt.PenStyle.NoPen)
            painter.drawRoundedRect(mapped(rect), 1.5, 1.5)
        painter.setPen(QPen(QColor(TOKENS["acc"]), 1.5))
        painter.setBrush(token_color(TOKENS["accbg"]))
        painter.drawRoundedRect(mapped(self.graph.visible_scene_rect()).intersected(frame), 4, 4)

    def move_view_to(self, pos):
        scale, offset = self.geometry_map()
        if scale > 0:
            self.graph.centerOn(QPointF((pos.x() - offset.x()) / scale, (pos.y() - offset.y()) / scale))

    def mousePressEvent(self, event):
        self.move_view_to(event.position())
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move_view_to(event.position())
        event.accept()


class GraphCorner:
    """Places the toggle, the zoom control and the minimap in the graph's bottom-right corner,
    in that order from the left, and keeps them in step with the view."""

    def __init__(self, graph):
        self.graph = graph
        viewport = graph.viewport()
        self.toggle = WireToggle(graph, viewport)
        self.zoom = ZoomControl(graph, viewport)
        self.minimap = MiniMap(graph, viewport)
        self.widgets = (self.toggle, self.zoom, self.minimap)
        self.reposition()
        for widget in self.widgets:
            widget.show()
            widget.raise_()

    def reposition(self):
        viewport = self.graph.viewport()
        right = viewport.width() - MARGIN
        for widget in reversed(self.widgets):
            if widget.sizeHint().isValid() and widget.size() != widget.sizeHint():
                widget.adjustSize()
            right -= widget.width()
            where = QPoint(right, viewport.height() - MARGIN - widget.height())
            if widget.pos() != where:
                widget.move(where)
            right -= GAP
    def sync(self):
        """Called whenever the graph repaints: the zoom label and the minimap follow the view."""
        self.toggle.sync()
        self.zoom.sync()
        self.reposition()
        self.minimap.update()
