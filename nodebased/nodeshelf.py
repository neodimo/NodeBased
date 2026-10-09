"""The left column of node families and the floating node panel it opens (New look, step 4, Lane 2).

`NodeRail` is the tall icon column of the mockup: Favourites, Recent, a hairline, one icon per
family with its family colour dot, and Settings at the bottom; the family whose panel is open shows
the bar on the column's edge. `NodePanel` is the frosted, rounded panel a click on a family opens:
the family's name and node count, a filter box that has focus on open, a two column grid of nodes
with their icons and an "N more" footer. Enter adds the highlighted node, a click adds it, a drag
into the graph adds it where it is dropped; Escape or a click outside closes the panel.
`NodeShelf` ties the two together and owns the panel's life.

Every size comes from font metrics (the column shrinks its buttons to fit a short window instead
of clipping), every colour from `theme`, every glyph from `nodeicons`.
"""
import math

from PySide6.QtCore import QEvent, QMimeData, QObject, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QApplication, QFrame, QGraphicsDropShadowEffect,
                               QGraphicsOpacityEffect,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QStyledItemDelegate,
                               QSizePolicy, QVBoxLayout, QWidget)

from . import nodeicons
from .nodecatalog import NODE_CATEGORIES, node_description
from .theme import FAMILY_COLORS, RADIUS, TOKENS, TYPE, family_color, theme_colors

NODE_KIND_MIME_TYPE = "application/x-nodebased-kind"
FAVOURITES = "Favourites"
RECENT = "Recent"
PANEL_ROWS = 6              # grid rows showing at once; more nodes scroll
PANEL_COLUMNS = 2


def mix(color_a, color_b, amount):
    """`color_a` moved `amount` (0..1) of the way to `color_b`, as a QColor."""
    a, b = QColor(color_a), QColor(color_b)
    return QColor(round(a.red() + (b.red() - a.red()) * amount), round(a.green() + (b.green() - a.green()) * amount),
                  round(a.blue() + (b.blue() - a.blue()) * amount))


def with_alpha(color, alpha):
    result = QColor(color)
    result.setAlphaF(alpha)
    return result


def ordered_kinds(kinds):
    """The panel's order: nodes with a glyph of their own (the common ones) first, each group in
    the catalog's order, so the first page of a long family holds the nodes artists reach for."""
    names = list(kinds)
    return [k for k in names if nodeicons.has_own_icon(k)] + [k for k in names if not nodeicons.has_own_icon(k)]


# =============================================================================================
# The column
class RailButton(QAbstractButton):
    """One square of the column: a glyph, the family colour dot, a lit state while its panel is open."""

    def __init__(self, key, glyph, swatch, tooltip, parent=None):
        super().__init__(parent)
        self.key = key
        self.glyph = glyph
        self.swatch = swatch            # the family colour, or None for Favourites / Recent / Settings
        self.setObjectName("rail-" + key.lower().replace(" ", "-"))
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)       # the graph keeps the keyboard
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.colors = theme_colors()
        self._hover = False

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def sizeHint(self):
        side = round(QFontMetrics(self.font()).height() * 2.2)
        return QSize(side, side)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self.colors
        scale = min(self.width(), self.height()) / 40.0
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 11 * scale
        active = self.isChecked()
        if active:
            painter.setPen(QPen(QColor(c["button_border"]), 1))
            painter.setBrush(QColor(c["raised"]))
            painter.drawRoundedRect(rect, radius, radius)
        elif self._hover:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(c["title"]))
            painter.drawRoundedRect(rect, radius, radius)
        tone = c["text"] if active else (mix(c["muted"], c["text"], 0.55).name() if self._hover else c["muted"])
        side = max(14, round(min(self.width(), self.height()) * 0.45))
        glyph = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        nodeicons.paint(painter, self.glyph, tone, glyph)
        if self.swatch:
            dot = max(4.0, 5.0 * scale)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(self.swatch))
            painter.drawEllipse(QRectF(self.width() - 7 * scale - dot, self.height() - 7 * scale - dot, dot, dot))


class NodeRail(QWidget):
    """The column. Buttons keep the mockup's order; heights adapt so every one is on screen."""

    family_clicked = Signal(str, object)         # (key, the button)
    settings_clicked = Signal()

    MIN_BUTTON = 22                              # the smallest a button gets in a very short window

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("node-rail")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.colors = theme_colors()
        self.buttons = {}
        self._active = None
        self._line_y = 0
        self._place(FAVOURITES, "ui-favourites", None, "Favourites")
        self._place(RECENT, "ui-recent", None, "Recently added")
        for family in NODE_CATEGORIES:
            self._place(family, nodeicons.family_icon_name(family if family in FAMILY_COLORS else "Other"),
                        FAMILY_COLORS.get(family), f"{family} nodes")
        self._place("Settings", "ui-settings", None, "Settings")
        # A tool bar gives its widget the height it asks for unless it is willing to expand.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._fit_width()

    def _place(self, key, glyph, swatch, tooltip):
        button = RailButton(key, glyph, swatch, tooltip, self)
        if key == "Settings":
            button.setCheckable(False)
            button.clicked.connect(self.settings_clicked)
        else:
            button.clicked.connect(lambda checked=False, k=key, b=button: self.family_clicked.emit(k, b))
        self.buttons[key] = button

    # ---- sizes: all from the font -----------------------------------------------------------
    def _unit(self):
        return QFontMetrics(self.font()).height()

    def _metrics(self):
        unit = self._unit()
        button = max(32, round(unit * 2.2))
        margin = max(6, round(unit * 0.55))
        gap = max(2, round(unit * 0.22))
        return button, margin, gap

    def column_width(self):
        """The column's width: a button, a margin either side and the hairline on its edge."""
        button, margin, _ = self._metrics()
        return button + 2 * margin + 1

    def _fit_width(self):
        self.setFixedWidth(self.column_width())
        self.updateGeometry()

    def sizeHint(self):
        return QSize(self.column_width(), self.minimumSizeHint().height())

    def minimumSizeHint(self):
        count = len(self.buttons)
        _, margin, _ = self._metrics()
        return QSize(self.column_width(), count * self.MIN_BUTTON + (count - 1) + self._separator() + 2 * margin // 2)

    def _separator(self):
        return 1 + 2 * max(3, round(self._unit() * 0.33))

    def event(self, event):
        handled = super().event(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange, QEvent.Type.ShowToParent):
            self._fit_width()
            self._layout_buttons()
        return handled

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_buttons()

    def _layout_buttons(self):
        """Stack the buttons top down with Settings pinned to the bottom; when the column is short
        the buttons (and then the gaps) shrink so none is cut off."""
        full, margin, gap = self._metrics()
        count = len(self.buttons)
        separator = self._separator()
        height = self.height()
        pad = margin
        side = full
        for gaps in (gap, 2, 1):
            side = (height - 2 * pad - separator - (count - 1) * gaps) // count
            if side >= min(full, 28):
                break
        side = max(self.MIN_BUTTON, min(full, side))
        used = side * count + (count - 1) * gaps + separator + 2 * pad
        if used > height:                       # a very short window: the margins give way first
            pad = max(2, pad - (used - height) // 2)
        left = margin
        y = pad
        for key, button in self.buttons.items():
            if key == "Settings":
                y = max(y, height - pad - side)
            button.setGeometry(left, y, full, side)
            y += side + gaps
            if key == RECENT:
                y += separator - gaps
        self._line_y = self.buttons[RECENT].geometry().bottom() + 1 + separator // 2
        self.update()

    # ---- state ------------------------------------------------------------------------------
    def set_active(self, key):
        """Light `key`'s button (and show its bar); None clears every button."""
        self._active = key
        for name, button in self.buttons.items():
            if name != "Settings":
                button.setChecked(name == key)
        self.update()

    def active(self):
        return self._active

    def set_theme(self, colors):
        self.colors = colors
        for button in self.buttons.values():
            button.colors = colors
            button.update()
        self.update()

    def bar_color(self, key):
        """The edge bar's colour: the family colour, the accent for Favourites and Recent."""
        return FAMILY_COLORS.get(key) or self.colors["accent"]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self.colors
        painter.fillRect(self.rect(), QColor(c["panel"]))
        painter.fillRect(QRect(self.width() - 1, 0, 1, self.height()), QColor(c["border"]))
        inset = max(8, round(self.width() * 0.2))
        painter.fillRect(QRect(inset, self._line_y, self.width() - 1 - 2 * inset, 1), QColor(c["border"]))
        if self._active in self.buttons:
            button = self.buttons[self._active].geometry()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(self.bar_color(self._active)))
            inset_y = round(button.height() * 0.25)
            painter.drawRoundedRect(QRectF(0, button.top() + inset_y, 3, button.height() - 2 * inset_y), 1.5, 1.5)


# =============================================================================================
# The panel
class PanelList(QListWidget):
    """The panel's grid. Dragging embeds only the kind name, like the NODES dock's list."""

    hovered = Signal(int)
    drag_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("node-panel-list")
        self.setViewMode(QListWidget.ViewMode.IconMode)       # left to right, wrapping into a grid
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setUniformItemSizes(True)
        self.setMovement(QListWidget.Movement.Static)
        self.setDragEnabled(True)                              # after the view mode, which resets it
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)   # wheel, keys and "N more" scroll
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setSpacing(0)
        self.empty_text = ""
        self.empty_color = QColor(TOKENS["tx3"])

    def mimeData(self, items):
        data = QMimeData()
        if items:
            kind = items[0].data(Qt.ItemDataRole.UserRole)
            if kind:
                data.setData(NODE_KIND_MIME_TYPE, kind.encode("utf-8"))
        return data

    def startDrag(self, supported_actions):
        super().startDrag(supported_actions)
        self.drag_finished.emit()

    def mouseMoveEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item is not None and not event.buttons():
            self.setCurrentItem(item)
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.count() and self.empty_text:
            painter = QPainter(self.viewport())
            painter.setPen(self.empty_color)
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)


class PanelDelegate(QStyledItemDelegate):
    """Paints a node: a 22 px tile with its glyph in the family colour, then its name."""

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.colors = theme_colors()

    def sizeHint(self, option, index):
        return self.view.gridSize() if not self.view.gridSize().isEmpty() else super().sizeHint(option, index)

    def paint(self, painter, option, index):
        c = self.colors
        kind = index.data(Qt.ItemDataRole.UserRole)
        rect = QRectF(option.rect).adjusted(2.5, 2.5, -2.5, -2.5)
        current = index == self.view.currentIndex()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        tint = QColor(family_color(kind))
        if current:
            painter.setPen(QPen(with_alpha(c["accent"], 0.25), 1))
            painter.setBrush(with_alpha(c["accent"], 0.10))
            painter.drawRoundedRect(rect, 8, 8)
        tile = QRectF(rect.left() + 8, rect.center().y() - 11, 22, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(with_alpha(tint, 0.15) if current else QColor(c["raised"]))
        painter.drawRoundedRect(tile, RADIUS["chip"], RADIUS["chip"])
        nodeicons.paint(painter, nodeicons.node_icon_name(kind), tint.name(), tile.adjusted(4, 4, -4, -4))
        painter.setPen(QColor(c["text"] if current else mix(c["muted"], c["text"], 0.62).name()))
        painter.setFont(option.font)
        text = QRectF(tile.right() + 8, rect.top(), rect.right() - tile.right() - 12, rect.height())
        painter.drawText(text, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         QFontMetrics(option.font).elidedText(kind, Qt.TextElideMode.ElideRight, int(text.width())))
        painter.restore()


class ClickLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class SwatchDot(QWidget):
    """The family colour square before the panel's title."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = QColor(TOKENS["tx2"])
        side = round(QFontMetrics(self.font()).height() * 0.45)
        self.setFixedSize(max(8, side), max(8, side))

    def set_color(self, color):
        self.color = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.color)
        painter.drawRoundedRect(QRectF(self.rect()), 3, 3)


class NodePanel(QFrame):
    """The floating panel for one family (or Favourites / Recent)."""

    node_chosen = Signal(str)
    closed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("node-panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.colors = theme_colors()
        self.family = None
        self._kinds = {}
        self._backdrop = None
        self.total = 0
        self.hide()
        fm = QFontMetrics(self.font())
        row_font = QFont(self.font())
        row_font.setPixelSize(TYPE["small"])
        row_fm = QFontMetrics(row_font)
        self._row = max(22, row_fm.height()) + 14                   # 36 px: 7 px above and below a 22 px tile
        # tile inset 8, the 22 px tile, 8 before the name, the longest everyday name, 10 after it
        column = 8 + 22 + 8 + row_fm.horizontalAdvance("ColorCorrect") + 10
        self.grid_width = column * PANEL_COLUMNS + 4 * PANEL_COLUMNS
        pad = 12
        layout = QVBoxLayout(self)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(4, 2, 4, 10)
        head.setSpacing(8)
        self.dot = SwatchDot()
        self.title = QLabel("")
        self.title.setObjectName("node-panel-title")
        self.count = QLabel("")
        self.count.setObjectName("node-panel-count")
        head.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(self.title)
        head.addWidget(self.count)
        head.addStretch(1)
        layout.addLayout(head)

        self.filter = QLineEdit()
        self.filter.setObjectName("node-panel-filter")
        self.filter.setClearButtonEnabled(False)
        self.filter.setFixedHeight(round(fm.height() * 1.65))
        self._search_action = self.filter.addAction(nodeicons.icon("ui-search", self.colors["faint"], 15),
                                                    QLineEdit.ActionPosition.LeadingPosition)
        self.filter.textChanged.connect(self._refilter)
        self.filter.returnPressed.connect(self.add_current)
        self.filter.installEventFilter(self)
        layout.addWidget(self.filter)
        layout.addSpacing(10)

        self.list = PanelList()
        self.list.setFont(row_font)
        self.delegate = PanelDelegate(self.list)
        self.list.setItemDelegate(self.delegate)
        self.list.setGridSize(QSize(column + 4, self._row + 4))
        self.list.clicked.connect(self._clicked)
        self.list.verticalScrollBar().valueChanged.connect(lambda _=0: self._update_more())
        self.list.drag_finished.connect(self.close_panel)
        self.list.installEventFilter(self)
        layout.addWidget(self.list)

        foot = QHBoxLayout()
        foot.setContentsMargins(4, 10, 4, 2)
        self.more = ClickLabel("")
        self.more.setObjectName("node-panel-more")
        self.more.clicked.connect(self._page_down)
        self.more.setCursor(Qt.CursorShape.PointingHandCursor)
        self.keycap = QLabel("↵")
        self.keycap.setObjectName("node-panel-kbd")
        self.hint = QLabel("add  ·  drag into graph")
        self.hint.setObjectName("node-panel-hint")
        foot.addWidget(self.more)
        foot.addStretch(1)
        foot.addWidget(self.keycap)
        foot.addSpacing(6)
        foot.addWidget(self.hint)
        layout.addLayout(foot)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(60)
        shadow.setOffset(0, 24)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)
        self.setFixedWidth(self.grid_width + 2 * pad + 2 + 6)      # a little slack: a grid wraps only if its cells fit

    # ---- frosted glass ----------------------------------------------------------------------
    def set_backdrop(self, pixmap):
        """What the window shows behind the panel, blurred, so the panel reads as frosted glass:
        the tint over it is translucent and would otherwise let the viewer's text show through."""
        self._backdrop = None
        if pixmap is not None and not pixmap.isNull():
            small = pixmap.scaled(max(1, pixmap.width() // 9), max(1, pixmap.height() // 9),
                                  Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._backdrop = small.scaled(pixmap.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)

    def paintEvent(self, event):
        if self._backdrop is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            clip = QPainterPath()
            clip.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS["flyout"], RADIUS["flyout"])
            painter.setClipPath(clip)
            painter.drawPixmap(self.rect(), self._backdrop)
            painter.end()
        super().paintEvent(event)

    # ---- content ----------------------------------------------------------------------------
    def set_theme(self, colors):
        self.colors = colors
        self.delegate.colors = colors
        self.list.empty_color = QColor(colors["faint"])
        self._search_action.setIcon(nodeicons.icon("ui-search", colors["faint"], 15))
        self.list.viewport().update()

    def show_family(self, key, kinds, color, label=None):
        """Fill the panel for `key` with `kinds` (kind -> description) and clear the filter."""
        self.family = key
        self._kinds = dict(kinds)
        self.total = len(kinds)
        self.dot.set_color(color)
        self.title.setText(label or key)
        self.filter.blockSignals(True)
        self.filter.clear()
        self.filter.blockSignals(False)
        self.filter.setPlaceholderText(f"Filter {label or key}…")
        self.list.empty_text = {FAVOURITES: "Right-click a node in the NODES list to star it",
                                RECENT: "Nodes you add show up here"}.get(key, "")
        self._populate(ordered_kinds(self._kinds) if key not in (FAVOURITES, RECENT) else list(self._kinds))

    def _populate(self, kinds):
        self.list.clear()
        for kind in kinds:
            item = QListWidgetItem(kind)
            item.setData(Qt.ItemDataRole.UserRole, kind)
            description = self._kinds.get(kind) or node_description(kind)
            item.setToolTip(f"{kind}\n{description}" if description else kind)
            item.setSizeHint(self.list.gridSize() - QSize(4, 4))
            self.list.addItem(item)
        if kinds:
            self.list.setCurrentRow(0)
        shown = len(kinds)
        rows = max(1, min(PANEL_ROWS, math.ceil(shown / PANEL_COLUMNS)))
        self.list.setFixedHeight(rows * self.list.gridSize().height() + 4)
        text = self.filter.text().strip()
        if text:
            self.count.setText(f"{shown} of {self.total}")
        else:
            self.count.setText(f"{self.total} node" + ("" if self.total == 1 else "s"))
        self.adjustSize()
        self.list.verticalScrollBar().setValue(0)
        self._update_more()

    def _update_more(self):
        """"N more": the nodes below the ones showing; "Back to top" once the end is on screen."""
        shown = self.list.count()
        row_height = self.list.gridSize().height()
        first_row = round(self.list.verticalScrollBar().value() / row_height) if row_height else 0
        hidden = max(0, shown - (first_row + PANEL_ROWS) * PANEL_COLUMNS)
        if hidden:
            self.more.setText(f"{hidden} more")
        elif first_row:
            self.more.setText("Back to top")
        else:
            self.more.setText("No nodes match" if not shown and self.filter.text().strip() else "")

    def _refilter(self, text):
        needle = text.casefold().strip()
        base = ordered_kinds(self._kinds) if self.family not in (FAVOURITES, RECENT) else list(self._kinds)
        if needle:
            base = [k for k in base if needle in k.casefold()
                    or needle in (self._kinds.get(k) or node_description(k)).casefold()]
        self._populate(base)

    def visible_kinds(self):
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]

    def current_kind(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    # ---- choosing ---------------------------------------------------------------------------
    def add_current(self):
        kind = self.current_kind()
        if kind:
            self.node_chosen.emit(kind)

    def _clicked(self, index):
        kind = index.data(Qt.ItemDataRole.UserRole)
        if kind:
            self.node_chosen.emit(kind)

    def _page_down(self):
        bar = self.list.verticalScrollBar()
        if bar.value() >= bar.maximum():
            bar.setValue(0)
        else:
            bar.setValue(min(bar.maximum(), bar.value() + self.list.viewport().height()))

    def _step(self, delta):
        count = self.list.count()
        if count:
            row = self.list.currentRow()
            self.list.setCurrentRow(0 if row < 0 else max(0, min(count - 1, row + delta)))
            self.list.scrollTo(self.list.currentIndex())

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and watched in (self.filter, self.list):
            key = event.key()
            if key == Qt.Key.Key_Escape:
                self.close_panel()
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.add_current()
                return True
            move = {Qt.Key.Key_Down: PANEL_COLUMNS, Qt.Key.Key_Up: -PANEL_COLUMNS,
                    Qt.Key.Key_PageDown: PANEL_ROWS * PANEL_COLUMNS, Qt.Key.Key_PageUp: -PANEL_ROWS * PANEL_COLUMNS}.get(key)
            if move is None and not self.filter.text():
                move = {Qt.Key.Key_Right: 1, Qt.Key.Key_Left: -1}.get(key)
            if move is not None:
                self._step(move)
                return True
        return super().eventFilter(watched, event)

    def close_panel(self):
        if self.isVisible():
            self.hide()
            self.closed.emit()


# =============================================================================================
class NodeShelf(QObject):
    """The column and its panel for one window. `kinds_for(key)` gives the nodes of a family (or
    Favourites / Recent) as kind -> description, `add(kind)` adds a node to the graph."""

    def __init__(self, window, kinds_for, add, settings):
        super().__init__(window)
        self.window = window
        self._kinds_for = kinds_for
        self._add = add
        self.rail = NodeRail()
        self.panel = NodePanel(window)
        self.rail.family_clicked.connect(self._family_clicked)
        self.rail.settings_clicked.connect(settings)
        self.panel.node_chosen.connect(self._chosen)
        self.panel.closed.connect(self._closed)
        self._watching = False

    def set_theme(self, colors):
        self.rail.set_theme(colors)
        self.panel.set_theme(colors)

    # ---- opening and closing ----------------------------------------------------------------
    def is_open(self):
        return self.panel.isVisible()

    def _family_clicked(self, key, button):
        if self.is_open() and self.panel.family == key:
            self.close_panel()
            return
        self.open_family(key, button)

    def open_family(self, key, button=None):
        button = button or self.rail.buttons[key]
        color = FAMILY_COLORS.get(key) or self.rail.colors["accent"]
        label = {FAVOURITES: "Favourites", RECENT: "Recent"}.get(key, key)
        self.panel.show_family(key, self._kinds_for(key), color, label)
        self.rail.set_active(key)
        self._position(button)
        self.panel.hide()                       # so the grab below shows the window, not the old panel
        self.panel.set_backdrop(self.window.grab(self.panel.geometry()))
        final_pos = self.panel.pos()
        opacity = None
        if self.window.motion.enabled:
            self.panel.move(final_pos.x() - 12, final_pos.y())
            opacity = QGraphicsOpacityEffect(self.panel)
            self.panel.setGraphicsEffect(opacity)
            opacity.setOpacity(0.0)
        self.panel.show()
        self.panel.raise_()
        if opacity is not None:
            self.window.motion.animate(("family-panel-x", id(self)), self.panel,
                                       self.panel.x(), final_pos.x(),
                                       lambda x, p=self.panel, y=final_pos.y(): p.move(round(x), y), 120)
            self.window.motion.animate(("family-panel-opacity", id(self)), self.panel, 0.0, 1.0,
                                       opacity.setOpacity, 120)
        self.panel.filter.setFocus(Qt.FocusReason.PopupFocusReason)
        self._watch(True)

    def _position(self, button):
        window = self.window
        origin = self.rail.mapTo(window, QPoint(0, 0))
        x = origin.x() + self.rail.width() + max(6, round(QFontMetrics(self.rail.font()).height() * 0.45))
        top = origin.y() + 4
        bottom = origin.y() + self.rail.height() - 4
        y = button.mapTo(window, QPoint(0, 0)).y()
        self.panel.adjustSize()
        height = self.panel.height()
        y = max(top, min(y, bottom - height))
        self.panel.move(x, y)

    def close_panel(self):
        self.panel.close_panel()
        self._closed()

    def _closed(self):
        self.rail.set_active(None)
        self._watch(False)

    def _chosen(self, kind):
        self.close_panel()
        self._add(kind)

    # ---- click outside and Escape ------------------------------------------------------------
    def _watch(self, on):
        if on == self._watching:
            return
        self._watching = on
        app = QApplication.instance()
        if on:
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape and self.is_open():
            self.close_panel()
            return True
        if kind == QEvent.Type.MouseButtonPress and self.is_open():
            point = event.globalPosition().toPoint()
            under = QApplication.widgetAt(point)
            inside = under is not None and (under is self.panel or self.panel.isAncestorOf(under)
                                            or under is self.rail or self.rail.isAncestorOf(under))
            if not inside:
                self.close_panel()
        elif kind == QEvent.Type.WindowDeactivate and watched is self.window:
            self.close_panel()
        return False
