"""The top bar of the new look: logo mark, project / shot with a saved dot, workspace tabs as a
segmented control, the one search box, the GPU status pill, a compact menu button and the
Check for updates button (the primary button where the mockup shows Export).

Every colour, radius and size comes from `theme` (tokens) or from the application stylesheet
(`theme.build_style`, which styles these widgets by objectName). Heights and widths follow font
metrics, so the bar fits the fonts of whichever platform it runs on.
"""

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QConicalGradient, QFontMetrics, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QToolButton, QWidget)

from .theme import SPACING, TOKENS

WORKSPACE_TABS = ("Composite", "3D", "Simulate", "Render")
SEARCH_PLACEHOLDER = "Search nodes…"


class LogoMark(QWidget):
    """The rounded conic-gradient mark with a dark core."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topbar-logo")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def sizeHint(self):
        side = round(QFontMetrics(self.font()).height() * 1.6)
        return QSize(side, side)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        rect = rect.adjusted(1, 1, -1, -1)
        radius = rect.width() * 8 / 26
        halo = QColor(TOKENS["acc"])
        halo.setAlpha(46)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(halo))
        painter.drawRoundedRect(rect.adjusted(-1.5, -1.5, 1.5, 1.5), radius + 1.5, radius + 1.5)
        gradient = QConicalGradient(rect.center(), 240)
        for stop, name in ((0.0, "acc"), (0.33, "mark_b"), (0.66, "mark_c"), (1.0, "acc")):
            gradient.setColorAt(stop, QColor(TOKENS[name]))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(rect, radius, radius)
        core = rect.width() * 10 / 26
        painter.setBrush(QColor(TOKENS["bg0"]))
        painter.drawRoundedRect(QRectF(rect.center().x() - core / 2, rect.center().y() - core / 2, core, core),
                                core * 0.3, core * 0.3)


class StatusDot(QWidget):
    """A small filled circle (saved state, GPU light). `glow` paints the soft halo of a live light."""

    def __init__(self, color, glow=False, parent=None):
        super().__init__(parent)
        self._color, self._glow = QColor(color), glow
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_color(self, color):
        self._color = QColor(color)
        self.update()

    def color(self):
        return QColor(self._color)

    def _diameter(self):
        return max(6, round(QFontMetrics(self.font()).height() * 0.5))

    def sizeHint(self):
        side = self._diameter() + (6 if self._glow else 0)
        return QSize(side, side)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        diameter = self._diameter()
        painter.setPen(Qt.PenStyle.NoPen)
        if self._glow:
            glow = QRadialGradient(center, diameter)
            faint = QColor(self._color)
            faint.setAlpha(110)
            clear = QColor(self._color)
            clear.setAlpha(0)
            glow.setColorAt(0.4, faint)
            glow.setColorAt(1.0, clear)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, diameter, diameter)
        painter.setBrush(self._color)
        painter.drawEllipse(center, diameter / 2, diameter / 2)


class SegmentedTabs(QFrame):
    """Workspace tabs as one rounded segmented control."""

    selected = Signal(str)

    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.setObjectName("segmented")
        layout = QHBoxLayout(self)
        margin = SPACING["xs"]
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for name in names:
            button = QPushButton(name)
            button.setObjectName("workspace-tab")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setToolTip(f"{name} workspace")
            layout.addWidget(button)
            self.group.addButton(button)
            self.buttons[name] = button
            button.clicked.connect(lambda checked=False, tab=name: self.selected.emit(tab))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def current(self):
        return next((name for name, button in self.buttons.items() if button.isChecked()), None)

    def set_current(self, name):
        """Select a tab without emitting `selected` (used to follow the viewer mode)."""
        button = self.buttons.get(name)
        if button is not None and not button.isChecked():
            button.setChecked(True)


class Magnifier(QWidget):
    """The search glyph, drawn so it needs no asset and takes its colour from `source`'s text."""

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self._source = source
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        side = round(QFontMetrics(self.font()).height() * 0.9)
        return QSize(side, side)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self._source.palette().windowText().color(), 1.5,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        side = min(self.width(), self.height()) - 3
        painter.drawEllipse(QRectF(1.5, 1.5, side * 0.72, side * 0.72))
        painter.drawLine(QPointF(1.5 + side * 0.64, 1.5 + side * 0.64), QPointF(1.5 + side, 1.5 + side))


class SearchBox(QPushButton):
    """The one search box. It looks like a field and opens the existing search on click; the Tab
    key does the same from the graph, so the keycap on the right is a reminder of that shortcut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topbar-search")
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("Search nodes (Tab in the node graph)")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["md"], 0, SPACING["md"], 0)
        layout.setSpacing(SPACING["sm"] + 3)
        self.hint = QLabel(SEARCH_PLACEHOLDER)
        self.hint.setObjectName("topbar-search-text")
        self.keycap = QLabel("Tab")
        self.keycap.setObjectName("topbar-kbd")
        for widget in (self.hint, self.keycap):
            widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(Magnifier(self.hint))
        layout.addWidget(self.hint, 1)
        layout.addWidget(self.keycap)

    def sizeHint(self):
        # A comfortable resting width of about 28 characters; the bar stretches it from there.
        base = self.layout().sizeHint()
        return QSize(max(base.width(), self.fontMetrics().averageCharWidth() * 40), base.height() + 2 * SPACING["sm"])

    def minimumSizeHint(self):
        metrics = self.fontMetrics()
        return QSize(metrics.averageCharWidth() * 14 + self.keycap.sizeHint().width(),
                     self.layout().minimumSize().height() + 2 * SPACING["sm"])


class GpuPill(QFrame):
    """The GPU status pill: a green light, the active adapter and the last frame time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gpu-pill")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["md"] + 1, SPACING["xs"], SPACING["md"] + 1, SPACING["xs"])
        layout.setSpacing(SPACING["sm"] + 1)
        self.light = StatusDot(TOKENS["acc"], glow=True)
        self.label = QLabel()
        self.label.setObjectName("gpu-pill-text")
        layout.addWidget(self.light)
        layout.addWidget(self.label)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._compact = False
        self.set_state("", None, gpu=False)

    def set_state(self, adapter, frame_ms, gpu=True):
        """`adapter` is the display adapter name ('' when unknown), `frame_ms` the last frame's
        time in milliseconds or None before the first frame."""
        self._adapter = adapter or ("GPU" if gpu else "CPU")
        self._frame_ms = frame_ms
        self.light.set_color(TOKENS["acc"] if gpu else TOKENS["tx2"])
        self.setToolTip(f"{self._adapter}: display adapter and the last frame time" if gpu
                        else "No GPU display adapter in use; the CPU is drawing the viewer")
        self._refresh()

    def set_compact(self, compact):
        """In a narrow bar the pill keeps the light and the frame time and drops the adapter name."""
        self._compact = bool(compact)
        self._refresh()

    def _refresh(self):
        if self._frame_ms is None:
            text = self._adapter
        elif self._compact:
            text = f"{self._frame_ms:.0f} ms"
        else:
            text = f"{self._adapter} · {self._frame_ms:.0f} ms"
        self.label.setText(text)

    def text(self):
        return self.label.text()


class TopBar(QWidget):
    """Lays the pieces out in the mockup's order. The window supplies the update button (it
    already owns its behaviour) and connects the signals."""

    search_requested = Signal()
    workspace_selected = Signal(str)

    def __init__(self, update_button, parent=None):
        super().__init__(parent)
        self.setObjectName("topbar")
        self.update_button = update_button
        layout = QHBoxLayout(self)
        gutter = SPACING["lg"]
        layout.setContentsMargins(gutter, SPACING["sm"], gutter, SPACING["sm"])
        layout.setSpacing(gutter)

        self.logo = LogoMark()
        layout.addWidget(self.logo)

        project = QWidget()
        project.setObjectName("topbar-brand")
        project_layout = QHBoxLayout(project)
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.setSpacing(SPACING["sm"] + 2)
        self.project_name = QLabel("Untitled")
        self.project_name.setObjectName("topbar-project-name")
        self.project_shot = QLabel("")
        self.project_shot.setObjectName("topbar-project")
        self.saved_dot = StatusDot(TOKENS["acc"])
        project_layout.addWidget(self.project_name)
        project_layout.addWidget(self.project_shot)
        project_layout.addWidget(self.saved_dot)
        self.project = project
        layout.addWidget(project)

        self.tabs = SegmentedTabs(WORKSPACE_TABS)
        self.tabs.selected.connect(self.workspace_selected)
        self.tabs.set_current(WORKSPACE_TABS[0])
        layout.addWidget(self.tabs)

        self.search = SearchBox()
        self.search.clicked.connect(lambda checked=False: self.search_requested.emit())
        layout.addWidget(self.search, 1)

        self.gpu_pill = GpuPill()
        layout.addWidget(self.gpu_pill)

        self.menu_button = QToolButton()
        self.menu_button.setObjectName("topbar-menu")
        self.menu_button.setText("☰")
        self.menu_button.setToolTip("Menu: file, edit, time, workspace, window and help")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self.menu_button)

        update_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(update_button)
        self._collapsed = 0

    # ---- content -------------------------------------------------------------------------------
    def set_project(self, name, shot="", saved=True):
        self.project_name.setText(name)
        self.project_shot.setText(f"/ {shot}" if shot else "")
        self.project_shot.setVisible(bool(shot) and self._collapsed < 1)
        self.saved_dot.set_color(TOKENS["acc"] if saved else TOKENS["tx2"])
        self.saved_dot.setToolTip("Saved" if saved else "Unsaved changes")

    def set_gpu(self, adapter, frame_ms, gpu=True):
        self.gpu_pill.set_state(adapter, frame_ms, gpu)
        self._fit()      # a longer adapter name or time may no longer fit

    # ---- fitting -------------------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        """Give up the least useful pieces first when the bar is narrower than its contents: the
        shot name, then the GPU pill's adapter name, then the search box's keycap hint."""
        layout = self.layout()
        for step in (0, 1, 2):
            self._collapsed = step
            self.project_shot.setVisible(bool(self.project_shot.text()) and step < 1)
            self.gpu_pill.set_compact(step >= 2)
            layout.invalidate()
            if layout.minimumSize().width() <= self.width():
                break
