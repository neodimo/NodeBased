"""Numeric knob widgets that never show a shortened number.

An artist reading the Properties panel must never see "20" for 200.000 (10/7 hands-on pass, finding
1). A spin box inside a narrow row was squeezed until its editor was narrower than its text, and a
key-frame diamond drawn inside the editor took more of what was left. Three pieces fix that:

* the spin boxes ask for exactly the width their current text needs (plus the diamond when one is
  showing), so the layout hands them that width before it hands any to a slider or a spacer;
* WrappingRow moves a multi-component knob (translate, scale, colour) onto a second row before a
  field is squeezed below that width;
* when a value still cannot fit (a very narrow docked panel), the field shows the number cut short
  with a visible ellipsis and carries the full number in its tooltip. The field restores the full
  number the moment it takes focus, so an edit always starts from the real value.
"""

import math
import re

from PySide6.QtCore import QEvent, QObject, QSize, QTimer
from PySide6.QtGui import QFontMetrics, QValidator
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QStyle, QStyleOptionComboBox, QStylePainter, QGridLayout, QSpinBox, QToolButton, QWidget)

ELLIPSIS = "…"
# The editor keeps 2 px of margin on each side of its text.
TEXT_MARGIN = 4


def _text_area(line_edit):
    """Pixels of the editor that are free for text: its width less margins and side icons."""
    icons = sum(button.width() for button in line_edit.findChildren(QToolButton) if button.isVisibleTo(line_edit))
    margins = line_edit.textMargins()
    return line_edit.width() - TEXT_MARGIN - icons - margins.left() - margins.right()


def elide_to_width(text, metrics, width):
    """``text`` cut from the right with an ellipsis so that it fits ``width``; never a bare number."""
    if metrics.horizontalAdvance(text) <= width:
        return text
    cut = text
    while cut and metrics.horizontalAdvance(cut + ELLIPSIS) > width:
        cut = cut[:-1]
    return cut + ELLIPSIS


class _FocusFilter(QObject):
    """Restore the full number when the editor takes focus; elide again when it leaves."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.FocusIn:
            self.owner._restore_full_text()
        elif kind == QEvent.Type.FocusOut:
            self.owner._schedule_fit()
        elif kind == QEvent.Type.Resize:
            self.owner._schedule_fit()
        return False


class FittedSpinMixin:
    """Shared behaviour of the two fitted spin boxes. Put it before the Qt spin box class."""

    def _setup_fitted(self):
        self._eliding = False
        self._fit_pending = False
        self._focus_filter = _FocusFilter(self)
        self.lineEdit().installEventFilter(self._focus_filter)
        self.lineEdit().textChanged.connect(self._text_changed)

    # -- typing --------------------------------------------------------------------------------

    _NUMBER_PREFIX = re.compile(r"^[+-]?\d*\.?\d*(?:[eE][+-]?\d*)?$")

    def _bare_text(self, text):
        text = text.strip()
        if self.prefix() and text.startswith(self.prefix()):
            text = text[len(self.prefix()):]
        if self.suffix() and text.endswith(self.suffix()):
            text = text[:-len(self.suffix())]
        return text.strip()

    def validate(self, text, pos):
        """Qt refuses a digit as soon as the number could no longer fit the range, so a range of
        +-8192 only ever let four integer digits in and 123456 became 1234 (10/7 hands-on pass,
        finding 7). Any number is accepted while typing; the value is clamped to the range when
        the edit finishes."""
        state, text_out, pos_out = super().validate(text, pos)
        if state == QValidator.State.Acceptable:
            return state, text_out, pos_out
        bare = self._bare_text(text)
        decimal = isinstance(self, QDoubleSpinBox)
        pattern = self._NUMBER_PREFIX if decimal else re.compile(r"^[+-]?\d*$")
        if not pattern.match(bare):
            return state, text_out, pos_out
        try:
            float(bare)
        except ValueError:
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Acceptable, text, pos

    def valueFromText(self, text):
        bare = self._bare_text(text)
        if isinstance(self, QDoubleSpinBox):
            try:
                number = float(bare)
            except ValueError:
                return super().valueFromText(text)
            return number if math.isfinite(number) else self.value()
        try:
            return int(bare)
        except ValueError:
            return super().valueFromText(text)

    # -- size ----------------------------------------------------------------------------------

    def _full_text(self):
        return self.prefix() + self.textFromValue(self.value()) + self.suffix()

    def _glyph_width(self):
        return sum(button.width() for button in self.lineEdit().findChildren(QToolButton)
                   if button.isVisibleTo(self.lineEdit()))

    def fitted_width(self):
        """The width this box needs to show its current value in full."""
        metrics = self.fontMetrics()
        base = super().sizeHint().width()
        reserved = max(metrics.horizontalAdvance(self.prefix() + self.textFromValue(self.minimum()) + self.suffix()),
                       metrics.horizontalAdvance(self.prefix() + self.textFromValue(self.maximum()) + self.suffix()))
        glyph = self._glyph_width() or (22 if self.property("keyedHere") else 0)
        return base - reserved + metrics.horizontalAdvance(self._full_text()) + glyph + 2

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(self.fitted_width(), hint.height())

    def minimumSizeHint(self):
        return QSize(min(self.fitted_width(), 64), super().minimumSizeHint().height())

    # -- elision -------------------------------------------------------------------------------

    def _compose_tip(self):
        if self._eliding:
            return f"{self._full_text()}\n{self._tooltip_base}" if self._tooltip_base else self._full_text()
        return self._tooltip_base

    def setToolTip(self, text):
        self._tooltip_base = text
        super().setToolTip(self._compose_tip())

    def _restore_full_text(self):
        was = self._eliding
        self._eliding = False
        if was:
            edit = self.lineEdit()
            edit.blockSignals(True)
            edit.setText(self._full_text())
            edit.blockSignals(False)
        super().setToolTip(self._compose_tip())

    def _schedule_fit(self):
        if not self._fit_pending:
            self._fit_pending = True
            QTimer.singleShot(0, self, self.refit)

    def refit(self):
        """Show the full number if it fits the editor, otherwise an ellipsis with the tooltip."""
        self._fit_pending = False
        edit = self.lineEdit()
        if edit.hasFocus() or not self.isVisible():
            return
        full = self._full_text()
        metrics = QFontMetrics(edit.font())
        self._yield_room_to_text(edit, metrics)
        width = _text_area(edit)
        if metrics.horizontalAdvance(full) <= width:
            self._restore_full_text()
            return
        shown = elide_to_width(full, metrics, width)
        self._eliding = True
        if edit.text() != shown:
            edit.blockSignals(True)
            edit.setText(shown)
            edit.blockSignals(False)
        super().setToolTip(self._compose_tip())

    def _yield_room_to_text(self, edit, metrics):
        """The key-frame diamond inside the editor gives way when the editor has no room for even an
        ellipsis beside it; the key button next to the field still shows the key state."""
        glyphs = [a for a in edit.actions() if a.objectName() == "key-glyph"]
        if not glyphs:
            return
        room = edit.width() - TEXT_MARGIN - 22 - edit.textMargins().left() - edit.textMargins().right()
        show = room >= metrics.horizontalAdvance(ELLIPSIS)
        if glyphs[0].isVisible() != show:
            glyphs[0].setVisible(show)
            edit.updateGeometry()

    def _text_changed(self, text):
        # Qt rewrote the editor text (a new value, a step, a range change): measure it again.
        self._eliding = False
        super().setToolTip(self._compose_tip())
        self._schedule_fit()
        self.updateGeometry()

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_fit()


class FittedDoubleSpinBox(FittedSpinMixin, QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tooltip_base = ""
        self._setup_fitted()


class FittedSpinBox(FittedSpinMixin, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tooltip_base = ""
        self._setup_fitted()


class WrappingRow(QWidget):
    """A row of knob units that moves onto more rows when they no longer fit side by side.

    Every unit keeps the width its own size hint asks for. The row tries all units on one line,
    then half of them per line, then one per line, and takes the first arrangement whose widest line
    fits the width it was given. The choice depends only on the units' hints and the width, so it
    does not flip back and forth as the height changes."""

    def __init__(self, units, spacing=4, parent=None):
        super().__init__(parent)
        self.units = list(units)
        self._spacing = spacing
        self._columns = len(self.units)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(spacing)
        self._grid.setVerticalSpacing(spacing)
        self._place(self._columns)

    def _place(self, columns):
        self._columns = columns
        for unit in self.units:
            self._grid.removeWidget(unit)
        for index, unit in enumerate(self.units):
            self._grid.addWidget(unit, index // columns, index % columns)
        for column in range(len(self.units)):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)
        self.updateGeometry()

    def _needed(self, columns):
        widths = [unit.sizeHint().width() for unit in self.units]
        lines = [widths[start:start + columns] for start in range(0, len(widths), columns)]
        column_widths = [max((line[c] for line in lines if c < len(line)), default=0) for c in range(columns)]
        return sum(column_widths) + self._spacing * (columns - 1)

    def best_columns(self, width):
        count = len(self.units)
        for columns in dict.fromkeys((count, math.ceil(count / 2), 1)):
            if self._needed(columns) <= width:
                return columns
        return 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = self.best_columns(self.width())
        if columns != self._columns:
            self._place(columns)

    WRAP_CAP = 190

    def minimumSizeHint(self):
        # One unit on a line, capped: a form row whose field cannot get this much wraps its label
        # above the field (QFormLayout.WrapLongRows) before the row is squeezed, and a very narrow
        # dock still shrinks the row, where the fields elide.
        need = max((u.sizeHint().width() for u in self.units), default=0)
        return QSize(min(need, self.WRAP_CAP), super().minimumSizeHint().height())


class ElidingComboBox(QComboBox):
    """A choice box that draws a choice too long for it with an ellipsis and carries the whole choice
    in its tooltip, so a long name is never silently cut at the edge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tooltip_base = ""
        self.currentTextChanged.connect(self._sync_tip)

    def setToolTip(self, text):
        self._tooltip_base = text
        super().setToolTip(self._compose_tip())

    def _sync_tip(self, *_):
        if self.toolTip() != self._compose_tip():
            super().setToolTip(self._compose_tip())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_tip()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_tip()

    def _label_width(self):
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        return self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                           QStyle.SubControl.SC_ComboBoxEditField, self).width() - 4

    def is_elided(self):
        if self.isEditable():
            return False
        return QFontMetrics(self.font()).horizontalAdvance(self.currentText()) > self._label_width()

    def displayed_text(self):
        return elide_to_width(self.currentText(), QFontMetrics(self.font()), self._label_width())

    def _compose_tip(self):
        if self.is_elided():
            return f"{self.currentText()}\n{self._tooltip_base}" if self._tooltip_base else self.currentText()
        return self._tooltip_base

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        if not self.isEditable():
            option.currentText = self.displayed_text()
            painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
        if self.toolTip() != self._compose_tip():
            super().setToolTip(self._compose_tip())
