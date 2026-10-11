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
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QToolTip, QVBoxLayout, QWidget

from .theme import TOKENS


# The playback unit's rhythm, in pixels: every control is 28 tall and sits on an 8 px grid.
UNIT_CONTROL = 28
UNIT_GAP = 8
UNIT_SIDE = 16

# A frame is cached when its post-view-transform image is resident; it is keyed when some curve
# in scope has a key on it. Green is the accent of the new look, the key blue matches Properties.
CACHED_COLOR = QColor(TOKENS["acc2"])
KEY_COLOR = QColor("#4f8cff")
PLAYHEAD_COLOR = QColor(TOKENS["acc"])
PLAYHEAD_INK = QColor(TOKENS["acc_ink"])
TICK_COLOR = QColor(TOKENS["line2"])
LABEL_COLOR = QColor(TOKENS["tx3"])
TRACK_COLOR = QColor(TOKENS["line2"])

# Track geometry, measured up from the bottom edge. The cache range is a 3 px rounded bar 4 px
# above the bottom, as in the mockup; key diamonds sit above it, ticks and numbers hang from the top.
CACHE_BAND_HEIGHT = 2
CACHE_BAND_LIFT = 0
KEY_TICK_WIDTH = 2
PLAYHEAD_WIDTH = 1

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
# The rest of the transport, drawn in the left column's language: 24 x 24, 1.8 px round strokes.
START_ICON = '<path d="M5 5v14M19 6l-5 6 5 6M13 6l-5 6 5 6"/>'
END_ICON = '<path d="M19 5v14M5 6l5 6-5 6M11 6l5 6-5 6"/>'
PREV_KEY_ICON = '<path d="M11 7l-5 5 5 5M19 12l-3.5-3.5L12 12l3.5 3.5z"/>'
NEXT_KEY_ICON = '<path d="M13 7l5 5-5 5M5 12l3.5-3.5L12 12l-3.5 3.5z"/>'
STEP_BACK_ONE_ICON = '<path d="M14 6l-6 6 6 6"/>'
STEP_FORWARD_ONE_ICON = '<path d="M10 6l6 6-6 6"/>'
PLAY_BACK_ICON = '<path d="M16 5.5v13L5 12z"/>'
LOOP_ICONS = {
    "once": '<path d="M4 12h12M12 8l4 4-4 4M20 6v12"/>',
    "loop": '<path d="M17 7H9a4 4 0 0 0-4 4v1M7 17h8a4 4 0 0 0 4-4v-1M14 4l3 3-3 3M10 14l-3 3 3 3"/>',
    "pingpong": '<path d="M3 12h18M7 8l-4 4 4 4M17 8l4 4-4 4"/>',
}
# The loop toggle cycles through these; "loop" is what playback has always done.
LOOP_MODES = ("once", "loop", "pingpong")
LOOP_LABELS = {"once": "Play once", "loop": "Loop", "pingpong": "Ping-pong"}


def playhead_at(first, last, origin, offset, direction, mode):
    """Where playback that started on ``origin`` has reached after ``offset`` frames, going
    ``direction`` (+1 or -1) in ``mode``. Returns ``(frame, finished)``; ``finished`` is True only
    in "once" mode, once the end of the range has been reached."""
    span = last - first + 1
    if span <= 1:
        return first, mode == "once" and offset > 0
    if mode == "once":
        target = origin + direction * offset
        frame = min(max(target, first), last)
        return frame, offset > 0 and frame == (last if direction > 0 else first)
    if mode == "pingpong":
        period = 2 * (span - 1)
        at = origin - first
        start = at if direction > 0 else (period - at) % period
        index = (start + offset) % period
        return first + (index if index <= span - 1 else period - index), False
    return first + (origin - first + direction * offset) % span, False


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
        # Numbers, the key ticks and the two thin lines, on a 4 px grid: 32 px on the default font.
        return QSize(160, max(32, round(self._text_height() * 2.0) // 4 * 4))

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

    def cache_band_rect(self):
        """The cached-frames line: 2 px along the bottom edge."""
        return QRectF(0, self.height() - CACHE_BAND_LIFT - CACHE_BAND_HEIGHT, self.width(), CACHE_BAND_HEIGHT)

    def key_mark_rect(self, frame):
        """The keyframe tick of ``frame``: a thin line in the middle of the frame's cell, between
        the numbers and the cache line."""
        centre_x = self.frame_x(frame) + self.frame_width() / 2
        top = self._text_height() * 0.9 + 2
        bottom = self.cache_band_rect().top() - 3
        return QRectF(round(centre_x) - KEY_TICK_WIDTH / 2, top, KEY_TICK_WIDTH, max(4.0, bottom - top))

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

        # Ticks and numbers: 1 px lines hanging from the top edge. Both step sets are anchored on
        # multiples of the step rather than on the range start, so the labels stay on round frame
        # numbers (100, 125, 150) instead of drifting with wherever the comp happens to begin.
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
                painter.drawText(QRectF(x + 4, 1, MIN_LABEL_SPACING_PX, text_height),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 str(frame))

        # The cached frames: a thin accent line over a hairline track, drawn as merged runs so a
        # 1000-frame cached range is a handful of fills.
        band = self.cache_band_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(TRACK_COLOR)
        painter.drawRect(band)
        self._draw_cache_runs(painter, band)
        self._draw_key_ticks(painter)

        # The playhead last, so nothing buries it: a 1 px accent line the full height and a small
        # flag at the top. The motion clock moves _visual_value smoothly between frames during
        # playback; the frame number itself is in the frame field.
        playhead_x = self.frame_x(self._visual_value) + min(per_frame, 6.0) / 2
        playhead_x = min(max(playhead_x, 1.0), width - 1.0)
        painter.setBrush(PLAYHEAD_COLOR)
        painter.drawRect(QRectF(round(playhead_x) - PLAYHEAD_WIDTH / 2, 0, PLAYHEAD_WIDTH, height))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        flag = QPolygonF([QPointF(playhead_x - 4, 0), QPointF(playhead_x + 4, 0), QPointF(playhead_x, 6)])
        painter.drawPolygon(flag)
        painter.end()

    def _draw_key_ticks(self, painter):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(KEY_COLOR)
        for frame in self.key_frames:
            if self._first <= frame <= self._last:
                painter.drawRect(self.key_mark_rect(frame))

    def _draw_cache_runs(self, painter, band):
        if not self.cached_frames:
            return
        per_frame = self.frame_width()
        painter.setBrush(CACHED_COLOR)
        for start, end in contiguous_runs(self.cached_frames):
            if end < self._first or start > self._last:
                continue
            start = max(start, self._first)
            end = min(end, self._last)
            x = self.frame_x(start)
            painter.drawRect(QRectF(x, band.top(), max(1.0, (end - start + 1) * per_frame), band.height()))


class TimeRow(QWidget):
    """The playback unit under the viewer: a ruler across the full width, and under it one row of
    28 px controls on an 8 px rhythm (the frame field, the transport, the range and the rate).
    `add` places a control in that row; `set_optional` names the controls it may hide, first to go
    first, when the row would otherwise be wider than the viewer: a narrow dock or a wide font
    loses the status text and the rate presets before anything that is being edited is cut off."""

    def __init__(self):
        super().__init__()
        self.setObjectName("time-row")
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(0)
        self.controls = QHBoxLayout()
        self.controls.setContentsMargins(UNIT_SIDE, UNIT_GAP, UNIT_SIDE, UNIT_GAP)
        self.controls.setSpacing(UNIT_GAP)
        self.row = self.controls
        self.column.addLayout(self.controls)
        self._optional = []
        self._fitting = False

    def add_ruler(self, widget):
        """The ruler sits above the controls and spans the whole unit."""
        self.column.insertWidget(0, widget)
        return widget

    def add(self, widget, stretch=0):
        self.controls.addWidget(widget, stretch)
        return widget

    def set_optional(self, widgets):
        """Widgets the row may hide, first to go first."""
        self._optional = list(widgets)

    def _needed(self):
        margins = self.controls.contentsMargins()
        return self.controls.minimumSize().width() + margins.left() + margins.right()

    def fit(self):
        if self._fitting or not self._optional:
            return
        self._fitting = True
        try:
            for widget in self._optional:
                widget.setVisible(True)
            self.column.activate()
            for widget in self._optional:
                if self._needed() <= self.width():
                    break
                widget.setVisible(False)
                self.column.activate()
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
