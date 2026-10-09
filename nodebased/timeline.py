"""The playhead strip: a scrubber that also reports cache and animation state per frame.

This replaces a plain ``QSlider``. It keeps that widget's contract on purpose -- ``setRange``,
``setValue``, ``value`` and a ``valueChanged(int)`` signal -- so the window wires it, blocks its
signals during sync, and reads it back exactly as before. What it adds is the information a
compositor reads off a timeline without clicking anything (new look, step 5: as in the mockup):

    * **Tick marks and frame numbers**, at a density chosen from the widget's actual pixel width
      rather than a fixed step. A 24-frame comp labels every frame; a 2000-frame comp labels every
      250th and thins the unlabelled ticks to match. Labels never collide, at any range.
    * **A green range** along the bottom of the track over the frames whose finished display image
      is in RAM, so "what will replay instantly" is visible instead of inferred from playback
      behaviour; frames that are not cached show the quiet track colour.
    * **Diamonds** over frames carrying an animation key, in the key blue the Properties panel uses.
    * **The playhead** as an accent line with a glow and its frame number on a small tab.

``TimeRow`` is the slim row around it: transport buttons, the frame field, this track, the range,
the rate and the real-time light. It drops its optional parts, last first, instead of clipping.

Ownership note: this widget renders state, it does not track it. ``cached_frames`` and
``key_frames`` are pushed in by the window from the display cache and the document, which keeps
the one source of truth for "is frame N cached?" inside the cache itself.
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QToolTip, QWidget

from .theme import TOKENS


# A frame is cached when its post-view-transform image is resident; it is keyed when some curve
# in scope has a key on it. Green is the accent of the new look, the key blue matches Properties.
CACHED_COLOR = QColor(TOKENS["acc"])
CACHED_COLOR_DEEP = QColor(TOKENS["acc2"])
KEY_COLOR = QColor("#4f8cff")
PLAYHEAD_COLOR = QColor(TOKENS["acc"])
PLAYHEAD_INK = QColor(TOKENS["acc_ink"])
TICK_COLOR = QColor(TOKENS["line2"])
LABEL_COLOR = QColor(TOKENS["tx3"])
TRACK_COLOR = QColor(TOKENS["line2"])

# Track geometry, measured up from the bottom edge. The cache range is a 3 px rounded bar 4 px
# above the bottom, as in the mockup; key diamonds sit above it, ticks and numbers hang from the top.
CACHE_BAND_HEIGHT = 3
CACHE_BAND_LIFT = 4
# A key diamond is a square turned 45 degrees, KEY_DIAMOND_PER_TEXT of a text line across and at
# least KEY_DIAMOND_MIN pixels, so it is always wider than the 2 px playhead and reads at a glance.
KEY_DIAMOND_PER_TEXT = 0.62
KEY_DIAMOND_MIN = 7
PLAYHEAD_WIDTH = 2

# A label needs this much horizontal room before the next one, or the step coarsens. Sized for
# four digits plus breathing space at the default UI font.
MIN_LABEL_SPACING_PX = 46
# Unlabelled ticks are allowed to get closer than labels, but not so close they read as a fill.
MIN_TICK_SPACING_PX = 5
# Nice-number ladder for both label and tick steps. Frame counts are read in these units by
# everyone who has ever looked at a timeline; 3s and 7s are not.
STEP_LADDER = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


# Transport glyphs (24 x 24, the mockup's own paths).
STEP_BACK_ICON = '<path d="M6 5v14M19 5 9 12l10 7z"/>'
STEP_FORWARD_ICON = '<path d="M18 5v14M5 5l10 7-10 7z"/>'
PLAY_ICON = '<path d="M8 5.5v13l11-6.5z"/>'
STOP_ICON = '<rect x="7" y="7" width="10" height="10" rx="1.5"/>'


def transport_icon(shapes, filled=False, ink=False, side=18):
    """A transport glyph as a QIcon: stroked in the quiet text colour, or filled; `ink` draws it in
    the dark ink the accent play button carries."""
    color = TOKENS["acc_ink"] if ink else TOKENS["tx1"]
    paint = f'fill="{color}" stroke="none"' if filled else \
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {paint}>{shapes}</svg>'
    pixmap = QPixmap(side * 2, side * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode())).render(painter, QRectF(0, 0, side * 2, side * 2))
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


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
        self._visual_value = 1.0
        self.cached_frames: set[int] = set()
        self.key_frames: set[int] = set()
        # frame -> ["Grade · exposure", ...]: what the key marks are keys of, for the hover tooltip.
        self.key_labels: dict[int, list[str]] = {}
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("Scrub the playhead. ← → step, Home/End jump to the range ends.\n"
                        "Green bar: frames that are cached and will replay instantly.\n"
                        "Blue diamond: frame carries an animation key; hover it to see which knobs.")

    def sizeHint(self):
        # Tall enough for the numbers, a diamond and the cache bar on any font.
        return QSize(160, max(34, round(self._text_height() * 2.3)))

    def minimumSizeHint(self):
        return QSize(60, self.sizeHint().height())

    # -- QSlider-compatible surface -------------------------------------------------------

    def setRange(self, first, last):
        self._first = int(first)
        self._last = max(int(last), int(first))
        self._value = min(max(self._value, self._first), self._last)
        self._visual_value = min(max(self._visual_value, self._first), self._last)
        self.update()

    def setValue(self, value):
        value = min(max(int(value), self._first), self._last)
        if value == self._value:
            return
        self._value = value
        self._visual_value = float(value)
        self.update()
        self.valueChanged.emit(value)

    def set_visual_value(self, value):
        """Move the painted playhead independently of the exact transport frame."""
        self._visual_value = min(max(float(value), self._first), self._last)
        self.update()

    def value(self):
        return self._value

    def minimum(self):
        return self._first

    def maximum(self):
        return self._last

    # -- State pushed in by the window ----------------------------------------------------

    def set_marks(self, cached_frames, key_frames, key_labels=None):
        """Replace both underline sets. Repaints only when something actually moved, so the
        per-frame refresh during playback does not cost a repaint per tick on a static graph.
        `key_labels` maps a keyed frame to the names of the knobs keyed there."""
        cached = set(cached_frames)
        keys = set(key_frames)
        labels = {frame: list(names) for frame, names in (key_labels or {}).items()}
        if cached == self.cached_frames and keys == self.key_frames and labels == self.key_labels:
            return
        self.cached_frames = cached
        self.key_frames = keys
        self.key_labels = labels
        self.update()

    def key_tooltip(self, frame):
        """Hover text for a keyed frame, or None: the frame and the knobs keyed on it."""
        names = self.key_labels.get(frame)
        if frame not in self.key_frames:
            return None
        if not names:
            return f"Frame {frame}: animation key"
        shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")
        return f"Frame {frame}: key on {shown}"

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            text = self.key_tooltip(self.frame_at(event.pos().x()))
            if text is not None:
                QToolTip.showText(event.globalPos(), text, self)
                return True
        return super().event(event)

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

    def _text_height(self):
        return QFontMetrics(self.font()).height()

    def key_diamond_size(self):
        """Pixels across a key diamond (corner to corner)."""
        return max(KEY_DIAMOND_MIN, round(self._text_height() * KEY_DIAMOND_PER_TEXT) | 1)

    def cache_band_rect(self):
        return QRectF(0, self.height() - CACHE_BAND_LIFT - CACHE_BAND_HEIGHT, self.width(), CACHE_BAND_HEIGHT)

    def key_mark_rect(self, frame):
        """The square that, turned 45 degrees, is the key diamond of ``frame``: centred on the
        frame's cell, in the middle of the space between the numbers and the cache bar."""
        size = self.key_diamond_size()
        side = size / 1.4142
        centre_x = self.frame_x(frame) + self.frame_width() / 2
        top = self._text_height() * 0.9
        bottom = self.cache_band_rect().top() - 2
        centre_y = (top + bottom) / 2 + 1
        return QRectF(centre_x - side / 2, centre_y - side / 2, side, side)

    def paintEvent(self, event):
        painter = QPainter(self)
        width, height = self.width(), self.height()
        per_frame = self.frame_width()
        text_height = self._text_height()

        label_step = choose_step(per_frame, MIN_LABEL_SPACING_PX)
        tick_step = choose_step(per_frame, MIN_TICK_SPACING_PX)

        font = QFont(self.font())
        font.setPixelSize(max(8, round(text_height * 0.72)))
        painter.setFont(font)

        # Ticks and numbers. Both step sets are anchored on multiples of the step rather than on
        # the range start, so the labels stay on round frame numbers (100, 125, 150) instead of
        # drifting with wherever the comp happens to begin.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        first_tick = self._first - (self._first % tick_step)
        for frame in range(first_tick, self._last + 1, tick_step):
            if frame < self._first:
                continue
            x = int(self.frame_x(frame))
            labelled = frame % label_step == 0
            painter.setPen(QPen(TICK_COLOR))
            painter.drawLine(x, 0, x, 6 if labelled else 3)
            if labelled:
                painter.setPen(QPen(LABEL_COLOR))
                painter.drawText(QRectF(x + 3, 2, MIN_LABEL_SPACING_PX, text_height),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 str(frame))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # The track under the cache range, then the range in green. Drawn as merged runs rather
        # than per-frame rectangles: a 1000-frame cached range is a handful of fills instead of
        # a thousand, and at sub-pixel frame widths a run still renders as a solid bar.
        band = self.cache_band_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(TRACK_COLOR)
        painter.drawRoundedRect(band, 1.5, 1.5)
        self._draw_cache_runs(painter, band)
        self._draw_key_diamonds(painter)

        # Playhead last, so nothing buries it: a glow, the line, and the frame number on a tab.
        # The motion clock moves _visual_value smoothly between frames during playback; the tab keeps the exact frame.
        playhead_x = self.frame_x(self._visual_value) + min(per_frame, 6.0) / 2
        playhead_x = min(max(playhead_x, 1.0), width - 1.0)
        glow = QColor(PLAYHEAD_COLOR)
        glow.setAlpha(60)
        painter.setBrush(glow)
        painter.drawRoundedRect(QRectF(playhead_x - 3, 0, 6, height), 3, 3)
        painter.setBrush(PLAYHEAD_COLOR)
        painter.drawRect(QRectF(playhead_x - PLAYHEAD_WIDTH / 2, 0, PLAYHEAD_WIDTH, height))
        self._draw_playhead_tab(painter, playhead_x, text_height)
        painter.end()

    def _draw_playhead_tab(self, painter, playhead_x, text_height):
        font = QFont(self.font())
        font.setPixelSize(max(9, round(text_height * 0.74)))
        font.setBold(True)
        painter.setFont(font)
        label = str(self._value)
        metrics = QFontMetrics(font)
        tab_width = metrics.horizontalAdvance(label) + 10
        tab_height = metrics.height() + 1
        left = min(max(playhead_x - tab_width / 2, 0), max(self.width() - tab_width, 0))
        tab = QRectF(left, 0, tab_width, tab_height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PLAYHEAD_COLOR)
        painter.drawRoundedRect(tab, 4, 4)
        painter.setPen(QPen(PLAYHEAD_INK))
        painter.drawText(tab, int(Qt.AlignmentFlag.AlignCenter), label)

    def _draw_key_diamonds(self, painter):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(KEY_COLOR)
        for frame in self.key_frames:
            if self._first <= frame <= self._last:
                rect = self.key_mark_rect(frame)
                painter.save()
                painter.translate(rect.center())
                painter.rotate(45)
                painter.drawRoundedRect(QRectF(-rect.width() / 2, -rect.height() / 2, rect.width(), rect.height()),
                                        1.5, 1.5)
                painter.restore()

    def _draw_cache_runs(self, painter, band):
        if not self.cached_frames:
            return
        per_frame = self.frame_width()
        gradient = QLinearGradient(QPointF(band.left(), 0), QPointF(band.right(), 0))
        gradient.setColorAt(0, CACHED_COLOR_DEEP)
        gradient.setColorAt(1, CACHED_COLOR)
        painter.setBrush(gradient)
        for start, end in contiguous_runs(self.cached_frames):
            if end < self._first or start > self._last:
                continue
            start = max(start, self._first)
            end = min(end, self._last)
            x = self.frame_x(start)
            painter.drawRoundedRect(QRectF(x, band.top(), max(1.0, (end - start + 1) * per_frame), band.height()),
                                    1.5, 1.5)


class TimeRow(QWidget):
    """The slim row under the viewer. `add` places a widget; `optional` names the widgets it may
    hide, in the order it hides them, when the row would otherwise be wider than the viewer: a
    narrow dock or a wide font loses the status text and the light's words before anything that
    is being edited is cut off."""

    def __init__(self):
        super().__init__()
        self.setObjectName("time-row")
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(16, 0, 16, 0)
        self.row.setSpacing(8)
        self._optional = []
        self._fitting = False

    def add(self, widget, stretch=0):
        self.row.addWidget(widget, stretch)
        return widget

    def set_optional(self, widgets):
        """Widgets the row may hide, first to go first."""
        self._optional = list(widgets)

    def fit(self):
        if self._fitting or not self._optional:
            return
        self._fitting = True
        try:
            for widget in self._optional:
                widget.setVisible(True)
            self.row.activate()
            for widget in self._optional:
                if self.row.minimumSize().width() <= self.width():
                    break
                widget.setVisible(False)
                self.row.activate()
        finally:
            self._fitting = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self.fit()


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
