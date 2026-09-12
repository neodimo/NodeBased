"""The playhead strip: a scrubber that also reports cache and animation state per frame.

This replaces a plain ``QSlider``. It keeps that widget's contract on purpose -- ``setRange``,
``setValue``, ``value`` and a ``valueChanged(int)`` signal -- so the window wires it, blocks its
signals during sync, and reads it back exactly as before. What it adds is the information a
compositor reads off a timeline without clicking anything:

    * **Tick marks and frame numbers**, at a density chosen from the widget's actual pixel width
      rather than a fixed step. A 24-frame comp labels every frame; a 2000-frame comp labels every
      250th and thins the unlabelled ticks to match. Labels never collide, at any range.
    * **An orange underline** under frames whose finished display image is in RAM, so "what will
      replay instantly" is visible instead of inferred from playback behaviour.
    * **A blue underline** under frames carrying an animation key.

The two underlines occupy separate bands so a frame that is both cached and keyed shows both.

Ownership note: this widget renders state, it does not track it. ``cached_frames`` and
``key_frames`` are pushed in by the window from the display cache and the document, which keeps
the one source of truth for "is frame N cached?" inside the cache itself.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


# A frame is cached when its post-view-transform image is resident; it is keyed when some curve
# in scope has a key on it. Orange and blue are DiMo's call and match the house viewer palette.
CACHED_COLOR = QColor("#e0873c")
KEY_COLOR = QColor("#4f8cff")
PLAYHEAD_COLOR = QColor("#f2f2f5")
TICK_COLOR = QColor("#6d6d78")
LABEL_COLOR = QColor("#9a9aa4")
TRACK_COLOR = QColor("#26262c")

# Band geometry, measured up from the bottom edge.
CACHE_BAND_HEIGHT = 3
KEY_BAND_HEIGHT = 3
BAND_TOTAL = CACHE_BAND_HEIGHT + KEY_BAND_HEIGHT

# A label needs this much horizontal room before the next one, or the step coarsens. Sized for
# four digits plus breathing space at the default UI font.
MIN_LABEL_SPACING_PX = 46
# Unlabelled ticks are allowed to get closer than labels, but not so close they read as a fill.
MIN_TICK_SPACING_PX = 5
# Nice-number ladder for both label and tick steps. Frame counts are read in these units by
# everyone who has ever looked at a timeline; 3s and 7s are not.
STEP_LADDER = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def choose_step(frames_per_pixel_span, minimum_spacing):
    """Smallest ladder step whose on-screen spacing clears ``minimum_spacing`` pixels.

    ``frames_per_pixel_span`` is pixels-per-frame. Falls back to the coarsest ladder entry rather
    than returning something unbounded, so a pathological range still draws.
    """
    for step in STEP_LADDER:
        if step * frames_per_pixel_span >= minimum_spacing:
            return step
    return STEP_LADDER[-1]


class TimelineBar(QWidget):
    """Scrub-and-report playhead strip. API-compatible with the QSlider it replaced."""

    valueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._first = 1
        self._last = 100
        self._value = 1
        self.cached_frames: set[int] = set()
        self.key_frames: set[int] = set()
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("Scrub the playhead. ← → step, Home/End jump to the range ends.\n"
                        "Orange underline: frame is cached and will replay instantly.\n"
                        "Blue underline: frame carries an animation key.")

    # -- QSlider-compatible surface -------------------------------------------------------

    def setRange(self, first, last):
        self._first = int(first)
        self._last = max(int(last), int(first))
        self._value = min(max(self._value, self._first), self._last)
        self.update()

    def setValue(self, value):
        value = min(max(int(value), self._first), self._last)
        if value == self._value:
            return
        self._value = value
        self.update()
        self.valueChanged.emit(value)

    def value(self):
        return self._value

    def minimum(self):
        return self._first

    def maximum(self):
        return self._last

    # -- State pushed in by the window ----------------------------------------------------

    def set_marks(self, cached_frames, key_frames):
        """Replace both underline sets. Repaints only when something actually moved, so the
        per-frame refresh during playback does not cost a repaint per tick on a static graph."""
        cached = set(cached_frames)
        keys = set(key_frames)
        if cached == self.cached_frames and keys == self.key_frames:
            return
        self.cached_frames = cached
        self.key_frames = keys
        self.update()

    # -- Geometry --------------------------------------------------------------------------

    def span(self):
        return self._last - self._first + 1

    def frame_width(self):
        """Pixels per frame. The track spans the full widget width with no slider-handle inset,
        because there is no handle -- the playhead is painted at the frame's own position."""
        return self.width() / float(self.span())

    def frame_x(self, frame):
        """Left edge, in pixels, of ``frame``'s cell."""
        return (frame - self._first) * self.frame_width()

    def frame_at(self, x):
        frame = self._first + int(x // self.frame_width()) if self.frame_width() > 0 else self._first
        return min(max(frame, self._first), self._last)

    # -- Interaction -----------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.frame_at(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.setValue(self.frame_at(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    # -- Painting --------------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        width, height = self.width(), self.height()
        per_frame = self.frame_width()
        painter.fillRect(0, 0, width, height, TRACK_COLOR)

        label_step = choose_step(per_frame, MIN_LABEL_SPACING_PX)
        tick_step = choose_step(per_frame, MIN_TICK_SPACING_PX)

        band_top = height - BAND_TOTAL
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        painter.setFont(font)

        # Ticks and numbers. Both step sets are anchored on multiples of the step rather than on
        # the range start, so the labels stay on round frame numbers (100, 125, 150) instead of
        # drifting with wherever the comp happens to begin.
        first_tick = self._first - (self._first % tick_step)
        for frame in range(first_tick, self._last + 1, tick_step):
            if frame < self._first:
                continue
            x = int(self.frame_x(frame))
            labelled = frame % label_step == 0
            painter.setPen(QPen(TICK_COLOR))
            painter.drawLine(x, 0, x, 5 if labelled else 3)
            if labelled:
                painter.setPen(QPen(LABEL_COLOR))
                painter.drawText(QRectF(x + 2, 4, MIN_LABEL_SPACING_PX, 12),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 str(frame))

        # Underlines. Drawn as merged runs rather than per-frame rectangles: a 1000-frame cached
        # range is a handful of fills instead of a thousand, and at sub-pixel frame widths the
        # run still renders as a solid band instead of dropping frames to rounding.
        self._draw_band(painter, self.cached_frames, band_top, CACHE_BAND_HEIGHT, CACHED_COLOR)
        self._draw_band(painter, self.key_frames, band_top + CACHE_BAND_HEIGHT, KEY_BAND_HEIGHT,
                        KEY_COLOR)

        # Playhead last, so it is never buried under a band.
        playhead_x = self.frame_x(self._value)
        painter.fillRect(QRectF(playhead_x, 0, max(1.0, per_frame), height),
                         QColor(255, 255, 255, 40))
        painter.setPen(QPen(PLAYHEAD_COLOR, 1))
        painter.drawLine(int(playhead_x), 0, int(playhead_x), height)
        painter.end()

    def _draw_band(self, painter, frames, top, band_height, color):
        if not frames:
            return
        per_frame = self.frame_width()
        for start, end in contiguous_runs(frames):
            if end < self._first or start > self._last:
                continue
            start = max(start, self._first)
            end = min(end, self._last)
            x = self.frame_x(start)
            painter.fillRect(QRectF(x, top, max(1.0, (end - start + 1) * per_frame), band_height),
                             color)


def contiguous_runs(frames):
    """Collapse a frame set into ``(start, end)`` inclusive runs, in ascending order."""
    ordered = sorted(frames)
    if not ordered:
        return []
    runs = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        runs.append((start, previous))
        start = previous = frame
    runs.append((start, previous))
    return runs
