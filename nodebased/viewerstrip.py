"""The viewer's floating control strip and its quiet corner readouts (new look, step 5, Lane 2).

The mockup's viewer carries one rounded strip at the top: 2D/3D, the channel buttons, the display
transform, gain and gamma, ROI and Fit. Everything else the old two control rows held (proxy,
zebra, format mask, 1:1, the gain and gamma resets, the viewer inputs) is one click away in the
strip's "more" menu. The corners carry the readouts: resolution, colour space and proxy state at
the bottom left; cursor position, pixel value with its swatch, and zoom at the bottom right.

Every width comes from font metrics (the strip sizes to its content and sheds its text labels
before it would clip), every colour from `theme` through object names in the stylesheet.
The widgets the window already owned (`channels`, `display_view`, `viewer_display`, `proxy`,
`exposure`, ...) keep their names and their signals: the strip only gives them a new face, so
the document, the shortcuts and the agent bridge read the same state as before.
"""
from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (QAbstractSpinBox, QButtonGroup, QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy,
                               QToolButton, QWidget)

from .graphcorner import svg_icon
from .theme import TOKENS
from .topbar import ElidedLabel

STRIP_TOP = 14        # distance from the top of the viewer, as in the mockup
STRIP_SIDE = 12       # smallest gap between the strip and the viewer's side edges
CORNER_SIDE = 18      # corner readouts sit this far from the sides and ...
CORNER_BOTTOM = 10    # ... this far from the bottom
CHANNELS = ("RGB", "R", "G", "B", "A")

SUN_ICON = ('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.4 1.4M17.6 17.6 19 19'
            'M5 19l1.4-1.4M17.6 6.4 19 5"/>')
ROI_ICON = '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 9h16M9 4v16"/>'
FIT_ICON = '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>'
CHEVRON_ICON = '<path d="m6 9 6 6 6-6"/>'
MORE_ICON = ('<circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/>')


def strip_icon(shapes, side=14):
    """A glyph in the strip's quiet colour, brighter while the button is checked."""
    return svg_icon(shapes, TOKENS["tx1"], TOKENS["tx0"], side)


class ChannelButtons(QWidget):
    """RGB, R, G, B and A as five buttons in one group. It answers the way the QComboBox it
    replaces did (`currentText`, `setCurrentText`, `currentTextChanged`), so the shortcuts that
    flip channels and the exporter that reads the pick are untouched."""

    currentTextChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("strip-channels")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for name in CHANNELS:
            button = QToolButton()
            button.setObjectName("strip-channel")
            button.setProperty("channel", name)
            button.setText(name)
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setToolTip("Show all colour channels" if name == "RGB" else
                              {"R": "Show red", "G": "Show green", "B": "Show blue", "A": "Show alpha"}[name])
            button.clicked.connect(lambda _checked=False, n=name: self.setCurrentText(n))
            self.group.addButton(button)
            self.buttons[name] = button
            row.addWidget(button)
        self.buttons["RGB"].setChecked(True)
        self._current = "RGB"

    def currentText(self):
        return self._current

    def setCurrentText(self, text):
        if text not in self.buttons:
            return
        self.buttons[text].setChecked(True)
        if text != self._current:
            self._current = text
            self.currentTextChanged.emit(text)

    def currentIndex(self):
        return CHANNELS.index(self._current)

    def setCurrentIndex(self, index):
        if 0 <= index < len(CHANNELS):
            self.setCurrentText(CHANNELS[index])

    def findText(self, text):
        return CHANNELS.index(text) if text in CHANNELS else -1

    def count(self):
        return len(CHANNELS)

    def itemText(self, index):
        return CHANNELS[index]


def separator():
    line = QFrame()
    line.setObjectName("strip-separator")
    line.setFixedWidth(1)
    line.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    return line


def tool_button(text, tip, icon=None, checkable=False):
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tip)
    button.setCheckable(checkable)
    button.setAutoRaise(False)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    if icon is not None:
        button.setIcon(icon)
        button.setIconSize(QSize(14, 14))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    return button


def add_combo_actions(menu, combo, on_pick=None):
    """One checkable action per item of `combo`, in one exclusive group, checked at the combo's
    current item. Picking one does what picking it in the combo does: the item becomes current and
    `activated` fires (the signal the window saves the artist's own choice on)."""
    group = QActionGroup(menu)
    group.setExclusive(True)
    for index in range(combo.count()):
        action = QAction(combo.itemText(index), menu)
        action.setCheckable(True)
        action.setChecked(index == combo.currentIndex())
        group.addAction(action)

        def pick(_checked=False, i=index):
            combo.setCurrentIndex(i)
            combo.activated.emit(i)
            if on_pick is not None:
                on_pick()
        action.triggered.connect(pick)
        menu.addAction(action)
    return group


class DisplayMenuButton(QToolButton):
    """The display transform: the project view (sRGB, ACES 2.0, Linear) and the viewer's own
    display (the project's default, or any view of the ACES config). The label names what is on
    screen; the menu lists both lists, each in its own section."""

    def __init__(self, display_view, viewer_display, parent=None):
        super().__init__(parent)
        self.display_view = display_view
        self.viewer_display = viewer_display
        self.setObjectName("strip-display")
        self.setAutoRaise(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setIcon(strip_icon(CHEVRON_ICON, 12))
        self.setIconSize(QSize(12, 12))
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)   # chevron after the text
        self.setToolTip("Display transform: the project's view, or any view of the ACES config. "
                        "Display only; exports stay independent of the viewer.")
        self._menu = QMenu(self)
        self._menu.aboutToShow.connect(self._fill)
        self.setMenu(self._menu)
        for combo in (display_view, viewer_display):
            combo.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def label(self):
        own = self.viewer_display.currentText()
        return self.display_view.currentText() if own in ("", "Project view") else own

    def refresh(self, *_):
        self.setText(self.label())
        self.updateGeometry()

    def _fill(self):
        menu = self._menu
        menu.clear()
        menu.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        menu.addSection("Project view")
        add_combo_actions(menu, self.display_view)
        menu.addSection("This viewer")
        add_combo_actions(menu, self.viewer_display)


class ViewerStrip(QFrame):
    """The floating strip. It is a child of the viewer stack, centred at the top, and sheds its
    text labels (ROI, Fit) when the viewer is too narrow for them, so nothing clips."""

    def __init__(self, window, parent):
        super().__init__(parent)
        self.window = window
        self.setObjectName("viewer-strip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(6, 3, 6, 3)
        self.row.setSpacing(4)
        self.compact = False

        for mode in ("2d", "3d"):
            self.row.addWidget(window.viewer_mode_buttons[mode])
        self.row.addWidget(separator())
        self.row.addWidget(window.channels)
        self.row.addWidget(separator())
        self.display_button = DisplayMenuButton(window.display_view, window.viewer_display)
        self.row.addWidget(self.display_button)
        self.row.addWidget(separator())
        self.sun = QLabel()
        self.sun.setPixmap(strip_icon(SUN_ICON, 15).pixmap(15, 15))
        self.sun.setToolTip(window.exposure.toolTip())
        self.gamma_mark = QLabel("γ")
        self.gamma_mark.setObjectName("strip-caption")
        self.gamma_mark.setToolTip(window.gamma.toolTip())
        for caption, spin in ((self.sun, window.exposure), (self.gamma_mark, window.gamma)):
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.row.addWidget(caption)
            self.row.addWidget(spin)
        window.exposure.setPrefix("")
        window.exposure.setSpecialValueText("")
        self.row.addWidget(separator())
        self.roi_button = window.roi_button
        self.roi_button.setIcon(strip_icon(ROI_ICON))
        self.roi_button.setIconSize(QSize(14, 14))
        self.roi_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.row.addWidget(self.roi_button)
        self.fit_button = tool_button("Fit", "Fit the picture to the viewer", strip_icon(FIT_ICON))
        self.fit_button.clicked.connect(lambda: window.viewer.fit())
        self.row.addWidget(self.fit_button)
        self.more_button = tool_button("", "More viewer options: proxy, zebra, format mask, 1:1, resets, inputs",
                                       strip_icon(MORE_ICON, 16))
        self.more_button.setObjectName("strip-more")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_menu = QMenu(self.more_button)
        self.more_menu.aboutToShow.connect(self.fill_more_menu)
        self.more_button.setMenu(self.more_menu)
        self.row.addWidget(self.more_button)
        parent.installEventFilter(self)
        self.show()

    # ---- the "more" menu: every control the old rows held that the strip itself leaves out ----
    def fill_more_menu(self):
        window, menu = self.window, self.more_menu
        menu.clear()
        proxy = menu.addMenu("Proxy")
        add_combo_actions(proxy, window.proxy)
        playing = menu.addAction("Proxy while playing")
        playing.setCheckable(True)
        playing.setChecked(window.playback_proxy.isChecked())
        playing.setToolTip(window.playback_proxy.toolTip())
        playing.toggled.connect(window.playback_proxy.setChecked)
        zebra = menu.addAction(window.zebra.text())
        zebra.setCheckable(True)
        zebra.setChecked(window.zebra.isChecked())
        zebra.setToolTip(window.zebra.toolTip())
        zebra.toggled.connect(window.zebra.setChecked)
        menu.addSeparator()
        add_combo_actions(menu.addMenu("Format mask"), window.mask_choice)
        add_combo_actions(menu.addMenu("Mask mode"), window.mask_mode)
        menu.addAction("1:1 pixels").triggered.connect(lambda: window.viewer.resetTransform())
        menu.addSeparator()
        menu.addAction("Reset gain").triggered.connect(lambda: window.set_viewer_look(gain=0.0))
        menu.addAction("Reset gamma").triggered.connect(lambda: window.set_viewer_look(gamma=1.0))
        menu.addSeparator()
        self.fill_inputs_menu(menu.addMenu("Viewer inputs"))

    def fill_inputs_menu(self, menu):
        """Input A (1-9), buffer B and the compare mode: the nine-button strip on the viewer is
        shown only when the corners leave it room, so the same choices live here too."""
        from .core import COMPARE_MODES, VIEWER_INPUT_COUNT
        viewer = self.window.viewer
        state = viewer.input_state()
        a_menu = menu.addMenu("Show input")
        a_group = QActionGroup(a_menu)
        for number in range(1, VIEWER_INPUT_COUNT + 1):
            action = QAction(f"Input {number}", a_menu)
            action.setCheckable(True)
            action.setChecked(number == state["active"])
            a_group.addAction(action)
            action.triggered.connect(lambda _checked=False, n=number: viewer.show_input(n))
            a_menu.addAction(action)
        b_menu = menu.addMenu("B buffer")
        b_group = QActionGroup(b_menu)
        for number in (None, *range(1, VIEWER_INPUT_COUNT + 1)):
            action = QAction("None" if number is None else f"Input {number}", b_menu)
            action.setCheckable(True)
            action.setChecked(number == state["b"])
            b_group.addAction(action)
            action.triggered.connect(lambda _checked=False, n=number: viewer.set_b(n))
            b_menu.addAction(action)
        mode_menu = menu.addMenu("Compare")
        mode_group = QActionGroup(mode_menu)
        for mode in COMPARE_MODES:
            action = QAction(mode, mode_menu)
            action.setCheckable(True)
            action.setChecked(mode == state["compare"])
            mode_group.addAction(action)
            action.triggered.connect(lambda _checked=False, m=mode: viewer.set_compare_mode(m))
            mode_menu.addAction(action)

    # ---- placement ----
    def set_compact(self, compact):
        if compact == self.compact:
            return
        self.compact = compact
        style = Qt.ToolButtonStyle.ToolButtonIconOnly if compact else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        for button in (self.roi_button, self.fit_button):
            button.setToolButtonStyle(style)
        self.layout().invalidate()

    def place(self):
        """Centre the strip at the top of its parent; go icon-only before it would be clipped."""
        parent = self.parentWidget()
        if parent is None:
            return
        room = max(parent.width() - 2 * STRIP_SIDE, 0)
        self.set_compact(False)
        self.layout().activate()
        if self.sizeHint().width() > room:
            self.set_compact(True)
            self.layout().activate()
        width = min(self.sizeHint().width(), room) if room else self.sizeHint().width()
        height = self.sizeHint().height()
        self.setGeometry((parent.width() - width) // 2, STRIP_TOP, width, height)
        self.raise_()

    def band_height(self):
        """How far down the strip reaches, with a gap: the picture is fitted below this line."""
        return STRIP_TOP + self.sizeHint().height() + 8

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (QEvent.Type.Resize, QEvent.Type.Show,
                                                               QEvent.Type.FontChange):
            self.place()
        return False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self.layout().invalidate()
            self.place()


class PixelReadout(QWidget):
    """Cursor position, pixel value and its swatch, and (during an A/B compare) the buffer the
    numbers come from. It lives in the viewer's bottom-right corner and appears only while the
    pointer is over the picture. `label` keeps the whole machine-readable line (position and the
    raw scene-linear floats, alpha included), which the tests and the agent bridge read."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pixel-readout")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.position = QLabel()
        self.position.setObjectName("corner-text")
        self.swatch = QLabel()
        self.swatch.setObjectName("corner-swatch")
        self.values = QLabel()
        self.values.setObjectName("corner-strong")
        self.buffer_tag = QLabel()
        self.buffer_tag.setObjectName("corner-buffer")
        for part in (self.position, self.swatch, self.values, self.buffer_tag):
            row.addWidget(part)
        # The whole line, kept out of sight: set_value fills it and `.label.text()` reports it.
        self.label = QLabel(self)
        self.label.hide()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setToolTip("Pixel values are raw scene-linear floats; y=0 is the bottom row, Nuke-style.")
        self._sync_swatch()

    def _sync_swatch(self):
        side = max(10, round(self.fontMetrics().height() * 0.8))
        self.swatch.setFixedSize(side, side)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._sync_swatch()

    def set_value(self, text, color, buffer="", position=None, values=None):
        self.label.setText(text)
        if position is None or values is None:
            head, _, tail = text.partition("  ")
            position, values = head, "  ".join(tail.split()[:3])
        self.position.setText(position)
        self.values.setText(values)
        self.buffer_tag.setText(buffer)
        self.buffer_tag.setVisible(bool(buffer))
        self.swatch.setStyleSheet(
            f"QLabel#corner-swatch {{ background: rgb({color.red()}, {color.green()}, {color.blue()}); }}")
        self.setToolTip(text + "\nRaw scene-linear floats; y=0 is the bottom row, Nuke-style.")


class CornerReadouts(QObject):
    """The two corners of the viewer: resolution, colour space and proxy state on the left; the
    pixel readout and the zoom on the right. They are children of the viewer stack, ignore the
    mouse and never move the picture."""

    def __init__(self, parent, pixel_readout):
        super().__init__(parent)
        self.parent_widget = parent
        self.left = QWidget(parent)
        self.left.setObjectName("viewer-corner")
        self.right = QWidget(parent)
        self.right.setObjectName("viewer-corner")
        for corner in (self.left, self.right):
            corner.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        left_row = QHBoxLayout(self.left)
        left_row.setContentsMargins(0, 0, 0, 0)
        left_row.setSpacing(14)
        self.resolution = ElidedLabel("")
        self.resolution.setObjectName("corner-strong")
        self.space = ElidedLabel("")
        self.space.setObjectName("corner-text")
        self.proxy = ElidedLabel("")
        self.proxy.setObjectName("corner-text")
        for part in (self.resolution, self.space, self.proxy):
            left_row.addWidget(part)
        right_row = QHBoxLayout(self.right)
        right_row.setContentsMargins(0, 0, 0, 0)
        right_row.setSpacing(14)
        self.pixel_readout = pixel_readout
        pixel_readout.setParent(self.right)
        self.zoom = QLabel("")
        self.zoom.setObjectName("corner-strong")
        right_row.addWidget(pixel_readout)
        right_row.addWidget(self.zoom)
        pixel_readout.hide()
        parent.installEventFilter(self)
        pixel_readout.installEventFilter(self)
        self.left.show()
        self.right.show()

    def set_image(self, width, height, space, proxy):
        self.resolution.setText(f"{width} × {height}" if width and height else "")
        self.space.setText(space)
        self.proxy.setText(proxy)
        self.place()

    def set_zoom(self, scale):
        text = f"{round(scale * 100)}%" if scale else ""
        if text != self.zoom.text():
            self.zoom.setText(text)
            self.place()

    def height(self):
        return self.left.sizeHint().height()

    def band_height(self):
        return self.height() + CORNER_BOTTOM + 6

    def free_span(self):
        """The gap between the two corners, in the parent's coordinates: (left edge, right edge)."""
        return self.left.geometry().right() + 8, self.right.geometry().left() - 8

    def place(self):
        parent = self.parent_widget
        half = max((parent.width() - 2 * CORNER_SIDE) // 2 - 7, 0)
        for corner in (self.left, self.right):
            corner.layout().activate()
        left_width = min(self.left.sizeHint().width(), half)
        right_width = min(self.right.sizeHint().width(), half)
        height = max(self.left.sizeHint().height(), self.right.sizeHint().height())
        y = parent.height() - height - CORNER_BOTTOM
        self.left.setGeometry(CORNER_SIDE, y, left_width, height)
        self.right.setGeometry(parent.width() - CORNER_SIDE - right_width, y, right_width, height)
        self.left.raise_()
        self.right.raise_()

    def eventFilter(self, watched, event):
        if watched is self.parent_widget and event.type() in (QEvent.Type.Resize, QEvent.Type.Show,
                                                              QEvent.Type.FontChange):
            self.place()
        elif watched is self.pixel_readout and event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
            self.place()
        return False
