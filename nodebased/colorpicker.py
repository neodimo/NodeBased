"""A colour picker that works in linear float, as a compositor's does (DiMo 10/18).

QColorDialog is an 8-bit, display-referred picker: 0-255 channels, a value that stops at 1, a
screen-colour eyedropper that reads the monitor's display-encoded pixel. A colour knob here holds
scene-linear floats, so this picker keeps them as floats end to end:

* a hue/saturation wheel and a value strip; the strip reaches past 1 for HDR colours;
* RGB(A), HSV and TMI (temperature, magenta, intensity, Nuke's third row) sliders, each with a
  float field that accepts any value, above 1 and below 0 included;
* an eyedropper that samples the viewer's linear pixel, never the screen;
* a swatch drawn through the viewer's display transform, old colour beside the new one, with
  Ctrl drag on it scrubbing intensity.

Values the artist does not touch come back exactly as they went in: the picker keeps the RGB(A)
it was given and only recomputes it from HSV or TMI when one of those is edited.
"""
from __future__ import annotations

import colorsys
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QConicalGradient, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

FIELD_RANGE = (-1.0e6, 1.0e6)


# -- colour maths ---------------------------------------------------------------------------------

def rgb_to_hsv(rgb):
    """HSV of linear RGB floats. V is the largest channel, so it passes 1 for HDR colours; hue in
    degrees. Negative channels keep a defined hue (colorsys works on any floats)."""
    r, g, b = (float(v) for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s, v


def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb((float(h) / 360.0) % 1.0, float(s), float(v))
    return [r, g, b]


def rgb_to_tmi(rgb):
    """Temperature (red against blue), magenta (red and blue against green) and intensity (the
    mean), an exactly invertible linear map like Nuke's TMI sliders."""
    r, g, b = (float(v) for v in rgb)
    return r - b, (r + b) / 2.0 - g, (r + g + b) / 3.0


def tmi_to_rgb(t, m, i):
    t, m, i = float(t), float(m), float(i)
    return [i + m / 3.0 + t / 2.0, i - 2.0 * m / 3.0, i + m / 3.0 - t / 2.0]


def display_color(rgb, view):
    """The swatch colour: linear RGB through the viewer's display transform, as an 8-bit QColor
    for the screen only (the knob's value is never read back from it)."""
    pixel = np.array([[list(rgb)[:3]]], dtype=np.float32)
    try:
        from . import color
        if view in ("Linear", "Raw"):
            shown = pixel
        elif view in ("sRGB", "ACES 2.0"):
            shown = pixel.copy()
            color.display_processor(view).applyRGB(shown)
        else:
            shown = color.display_rgb(pixel, view)
    except Exception:  # an unknown view or a missing OCIO config: a plain sRGB encode
        clipped = np.clip(pixel, 0.0, None)
        shown = np.where(clipped <= 0.0031308, clipped * 12.92, 1.055 * clipped ** (1 / 2.4) - 0.055)
    r, g, b = (min(max(float(v), 0.0), 1.0) for v in np.asarray(shown).reshape(-1)[:3])
    return QColor.fromRgbF(r, g, b)


# -- widgets --------------------------------------------------------------------------------------

class HueSaturationWheel(QWidget):
    """Hue around the rim, saturation out from the centre. Edits hue and saturation only."""
    changed = Signal(float, float)        # hue in degrees, saturation 0-1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("picker-wheel")
        self.setFixedSize(168, 168)
        self.hue, self.saturation = 0.0, 0.0

    def set_hs(self, hue, saturation):
        self.hue, self.saturation = float(hue), float(saturation)
        self.update()

    def _radius(self):
        return min(self.width(), self.height()) / 2 - 4

    def marker(self):
        c = QPointF(self.width() / 2, self.height() / 2)
        angle = math.radians(self.hue)
        reach = max(0.0, min(1.0, self.saturation)) * self._radius()
        return QPointF(c.x() + math.cos(angle) * reach, c.y() - math.sin(angle) * reach)

    def _pick(self, point):
        c = QPointF(self.width() / 2, self.height() / 2)
        dx, dy = point.x() - c.x(), c.y() - point.y()
        hue = math.degrees(math.atan2(dy, dx)) % 360.0
        saturation = min(1.0, math.hypot(dx, dy) / self._radius())
        self.set_hs(hue, saturation)
        self.changed.emit(hue, saturation)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick(event.position())
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._pick(event.position())
            event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        r = self._radius()
        hue = QConicalGradient(c, 0)
        for step in range(13):
            hue.setColorAt(step / 12, QColor.fromHsvF((step / 12) % 1.0, 1.0, 1.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(hue)
        painter.drawEllipse(c, r, r)
        white = QRadialGradient(c, r)
        white.setColorAt(0.0, QColor(255, 255, 255, 255))
        white.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setBrush(white)
        painter.drawEllipse(c, r, r)
        painter.setPen(QPen(QColor("#2c333b"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(c, r, r)
        painter.setPen(QPen(QColor("#111418"), 3))
        painter.drawEllipse(self.marker(), 5, 5)
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(self.marker(), 5, 5)
        painter.end()


class ValueStrip(QWidget):
    """The HSV value, bottom to top. The strip spans 0 to `top`, which grows past 1 to hold an HDR
    value, so a value of 4 can be picked and dragged as easily as 0.4."""
    changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("picker-value-strip")
        self.setFixedSize(26, 168)
        self.value, self.top, self.hue, self.saturation = 1.0, 1.0, 0.0, 0.0

    def set_state(self, value, hue, saturation):
        self.value, self.hue, self.saturation = float(value), float(hue), float(saturation)
        self.top = max(1.0, 2.0 ** math.ceil(math.log2(max(self.value, 1e-6) * 1.25)))
        self.update()

    def _pick(self, y):
        span = max(1, self.height() - 8)
        value = max(0.0, min(1.0, (self.height() - 4 - y) / span)) * self.top
        self.value = value
        self.update()
        self.changed.emit(value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick(event.position().y())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._pick(event.position().y())

    def mouseReleaseEvent(self, event):
        self.set_state(self.value, self.hue, self.saturation)   # rescale the strip to the new value

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(4, 4, self.width() - 8, self.height() - 8)
        gradient = QLinearGradient(rect.bottomLeft(), rect.topLeft())
        gradient.setColorAt(0.0, QColor("#000000"))
        gradient.setColorAt(min(1.0, 1.0 / self.top), QColor.fromHsvF(self.hue / 360.0 % 1.0,
                                                                       min(1.0, self.saturation), 1.0))
        gradient.setColorAt(1.0, QColor("#ffffff"))
        painter.setPen(QPen(QColor("#2c333b"), 1))
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 4, 4)
        if self.top > 1.0:
            one = rect.bottom() - rect.height() / self.top
            painter.setPen(QPen(QColor("#7d8793"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rect.left(), one), QPointF(rect.right(), one))
        y = rect.bottom() - min(1.0, self.value / self.top) * rect.height()
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(QPointF(0, y), QPointF(self.width(), y))
        painter.end()


class SwatchPair(QWidget):
    """New colour on the left, the colour the picker opened with on the right, both through the
    display transform. Ctrl drag scrubs intensity (all channels scale together, a stop per 60 px)."""
    scrubStarted = Signal()
    scrubbed = Signal(float)     # a factor relative to the colour at the press

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("picker-swatch")
        self.setFixedHeight(40)
        self.setToolTip("Through the viewer's display transform. Ctrl drag to scrub intensity.")
        self.new_color, self.old_color = QColor("#000000"), QColor("#000000")
        self._press_x = None

    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._press_x = event.position().x()
            self.scrubStarted.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_x is not None:
            self.scrubbed.emit(2.0 ** ((event.position().x() - self._press_x) / 60.0))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._press_x is not None:
            self._press_x = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        half = self.width() * 0.62
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.new_color)
        painter.drawRoundedRect(QRectF(0, 0, half, self.height()), 6, 6)
        painter.setBrush(self.old_color)
        painter.drawRoundedRect(QRectF(half + 4, 0, self.width() - half - 4, self.height()), 6, 6)
        painter.end()


class ChannelRow:
    """One labelled channel: a slider over a working range and a float field with no 0-255 or 0-1
    ceiling. The field is the value; the slider follows it within its range."""

    STEPS = 10000

    def __init__(self, grid, row, label, soft, on_edit, decimals=5):
        self.soft = soft
        self.on_edit = on_edit
        self._syncing = False
        self.label = QLabel(label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("knob-slider")
        self.slider.setRange(0, self.STEPS)
        self.slider.setMinimumWidth(120)
        self.field = QDoubleSpinBox()
        self.field.setObjectName(f"picker-{label.lower()}")
        self.field.setRange(*FIELD_RANGE)
        self.field.setDecimals(decimals)
        self.field.setSingleStep(0.01)
        self.field.setKeyboardTracking(False)
        self.field.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.field.setFixedWidth(96)
        grid.addWidget(self.label, row, 0)
        grid.addWidget(self.slider, row, 1)
        grid.addWidget(self.field, row, 2)
        self.field.valueChanged.connect(self._field_changed)
        self.slider.valueChanged.connect(self._slider_changed)

    def set_value(self, value):
        self._syncing = True
        self.field.setValue(float(value))
        lo, hi = self.soft
        self.slider.setValue(round(max(0.0, min(1.0, (float(value) - lo) / (hi - lo))) * self.STEPS))
        self._syncing = False

    def _field_changed(self, value):
        if not self._syncing:
            self.set_value(value)
            self.on_edit(float(value))

    def _slider_changed(self, position):
        if not self._syncing:
            lo, hi = self.soft
            value = lo + position / self.STEPS * (hi - lo)
            self._syncing = True
            self.field.setValue(value)
            self._syncing = False
            self.on_edit(value)


class LinearColorPicker(QDialog):
    """Pick a scene-linear colour. `rgba` is three or four floats; `view` is the display transform
    the swatch is drawn through; `sampler(callback)` starts a viewer sample that calls back with
    the linear RGBA under the cursor (or None). `colorChanged` fires on every edit, `colorAccepted`
    once on OK with the final floats."""

    colorChanged = Signal(list)
    colorAccepted = Signal(list)

    def __init__(self, rgba, parent=None, title="Colour", view="sRGB", sampler=None):
        super().__init__(parent)
        self.setObjectName("linear-color-picker")
        self.setWindowTitle(title)
        self.view = view
        self.sampler = sampler
        values = [float(v) for v in rgba]
        self._has_alpha = len(values) > 3
        self._rgb = values[:3]
        self._alpha = values[3] if self._has_alpha else None
        self._original = list(values)
        self._hue = rgb_to_hsv(self._rgb)[0]
        self._scrub_base = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)
        self.swatch = SwatchPair()
        self.swatch.scrubStarted.connect(lambda: setattr(self, "_scrub_base", list(self._rgb)))
        self.swatch.scrubbed.connect(self._scrub)
        outer.addWidget(self.swatch)
        top = QHBoxLayout()
        self.wheel = HueSaturationWheel()
        self.wheel.changed.connect(self._wheel_changed)
        self.strip = ValueStrip()
        self.strip.changed.connect(self._strip_changed)
        top.addWidget(self.wheel)
        top.addWidget(self.strip)
        side = QVBoxLayout()
        self.eyedropper = QPushButton("Sample viewer")
        self.eyedropper.setObjectName("picker-eyedropper")
        self.eyedropper.setCheckable(True)
        self.eyedropper.setToolTip("Click a pixel in the viewer: its linear float value, not the screen's colour")
        self.eyedropper.setEnabled(sampler is not None)
        self.eyedropper.toggled.connect(self._toggle_sampling)
        side.addWidget(self.eyedropper)
        side.addStretch(1)
        top.addLayout(side)
        outer.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        row = 0
        self.rows = {}

        def heading(text):
            nonlocal row
            label = QLabel(text)
            label.setStyleSheet("color: #56606b; font-size: 11px; letter-spacing: 1px")
            grid.addWidget(label, row, 0, 1, 3)
            row += 1

        heading("RGB")
        for index, name in enumerate("RGB"):
            self.rows[name] = ChannelRow(grid, row, name, (0.0, 1.0),
                                         lambda value, i=index: self._set_channel(i, value))
            row += 1
        if self._has_alpha:
            self.rows["A"] = ChannelRow(grid, row, "A", (0.0, 1.0), self._set_alpha)
            row += 1
        heading("HSV")
        self.rows["H"] = ChannelRow(grid, row, "H", (0.0, 360.0), lambda v: self._set_hsv(0, v), decimals=3)
        row += 1
        self.rows["S"] = ChannelRow(grid, row, "S", (0.0, 1.0), lambda v: self._set_hsv(1, v))
        row += 1
        self.rows["V"] = ChannelRow(grid, row, "V", (0.0, 1.0), lambda v: self._set_hsv(2, v))
        row += 1
        heading("TMI")
        self.rows["T"] = ChannelRow(grid, row, "T", (-1.0, 1.0), lambda v: self._set_tmi(0, v))
        row += 1
        self.rows["M"] = ChannelRow(grid, row, "M", (-1.0, 1.0), lambda v: self._set_tmi(1, v))
        row += 1
        self.rows["I"] = ChannelRow(grid, row, "I", (0.0, 1.0), lambda v: self._set_tmi(2, v))
        row += 1
        outer.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.swatch.old_color = display_color(self._original[:3], self.view)
        self._refresh()

    # -- state ------------------------------------------------------------------------------

    def color(self):
        """The picked colour as floats, RGB or RGBA as given."""
        return [*self._rgb, self._alpha] if self._has_alpha else list(self._rgb)

    def set_color(self, rgba):
        values = [float(v) for v in rgba]
        self._rgb = values[:3]
        if self._has_alpha and len(values) > 3:
            self._alpha = values[3]
        self._changed()

    def _changed(self, keep_hue=False):
        if not keep_hue:
            h, s, _v = rgb_to_hsv(self._rgb)
            if s > 0:
                self._hue = h
        self._refresh()
        self.colorChanged.emit(self.color())

    def _refresh(self):
        h, s, v = rgb_to_hsv(self._rgb)
        h = self._hue if s == 0 else h
        t, m, i = rgb_to_tmi(self._rgb)
        for name, value in zip("RGB", self._rgb):
            self.rows[name].set_value(value)
        if self._has_alpha:
            self.rows["A"].set_value(self._alpha)
        for name, value in (("H", h), ("S", s), ("V", v), ("T", t), ("M", m), ("I", i)):
            self.rows[name].set_value(value)
        self.wheel.set_hs(h, s)
        self.strip.set_state(v, h, s)
        self.swatch.new_color = display_color(self._rgb, self.view)
        self.swatch.update()

    def _set_channel(self, index, value):
        self._rgb[index] = float(value)
        self._changed()

    def _set_alpha(self, value):
        self._alpha = float(value)
        self._changed()

    def _set_hsv(self, index, value):
        h, s, v = rgb_to_hsv(self._rgb)
        h = self._hue if s == 0 else h
        hsv = [h, s, v]
        hsv[index] = float(value)
        if index == 0:
            self._hue = float(value)
        self._rgb = hsv_to_rgb(*hsv)
        self._changed(keep_hue=True)

    def _set_tmi(self, index, value):
        tmi = list(rgb_to_tmi(self._rgb))
        tmi[index] = float(value)
        self._rgb = tmi_to_rgb(*tmi)
        self._changed()

    def _wheel_changed(self, hue, saturation):
        v = rgb_to_hsv(self._rgb)[2]
        self._hue = hue
        self._rgb = hsv_to_rgb(hue, saturation, v if v > 0 else 1.0)
        self._changed(keep_hue=True)

    def _strip_changed(self, value):
        h, s, _v = rgb_to_hsv(self._rgb)
        h = self._hue if s == 0 else h
        self._rgb = hsv_to_rgb(h, s, value)
        self._changed(keep_hue=True)

    def _scrub(self, factor):
        if self._scrub_base is None:
            self._scrub_base = list(self._rgb)
        self._rgb = [value * factor for value in self._scrub_base]
        self._changed(keep_hue=True)

    # -- eyedropper -------------------------------------------------------------------------

    def _toggle_sampling(self, on):
        if on and self.sampler is not None:
            self.sampler(self.receive_sample)

    def receive_sample(self, rgba):
        """A viewer sample: linear floats straight from the frame (None off the image)."""
        self.eyedropper.setChecked(False)
        if rgba is None:
            return
        values = [float(v) for v in rgba]
        self._rgb = values[:3]
        if self._has_alpha and len(values) > 3:
            self._alpha = values[3]
        self._changed()

    def accept(self):
        self.colorAccepted.emit(self.color())
        super().accept()
