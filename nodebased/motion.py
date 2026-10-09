"""One lightweight, owner-scoped animation clock for the desktop UI.

Tracks run on the Qt main thread. They deliberately avoid QPropertyAnimation: dock and
panel lifetimes are controlled by Qt and a single timer makes cancellation deterministic.
"""
from __future__ import annotations

import time
import weakref

from PySide6.QtCore import QObject, QTimer, Qt
import shiboken6


class MotionAnimator(QObject):
    """Interpolate scalar values for live Qt objects using one precise timer."""

    def __init__(self, parent=None, *, enabled=True, interval_ms=16):
        super().__init__(parent)
        self.enabled = bool(enabled)
        self._tracks = {}
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(max(1, int(interval_ms)))
        self._timer.timeout.connect(self._tick)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.finish_all()

    def animate(self, key, target, start, end, setter, duration_ms=150):
        """Animate one value; replacing a key cleanly supersedes its previous track."""
        self._tracks.pop(key, None)
        if not self.enabled or duration_ms <= 0 or start == end:
            if shiboken6.isValid(target):
                setter(end)
            self._stop_if_idle()
            return
        self._tracks[key] = (weakref.ref(target), float(start), float(end), setter,
                             time.monotonic(), max(1, int(duration_ms)) / 1000.0)
        if not self._timer.isActive():
            self._timer.start()

    def cancel(self, key, *, finish=False):
        track = self._tracks.pop(key, None)
        if finish and track:
            target = track[0]()
            if target is not None and shiboken6.isValid(target):
                track[3](track[2])
        self._stop_if_idle()

    def finish_all(self):
        tracks, self._tracks = self._tracks, {}
        self._timer.stop()
        for target_ref, _start, end, setter, _began, _duration in tracks.values():
            target = target_ref()
            if target is not None and shiboken6.isValid(target):
                setter(end)

    def _tick(self):
        now = time.monotonic()
        for key, (target_ref, start, end, setter, began, duration) in list(self._tracks.items()):
            target = target_ref()
            if target is None or not shiboken6.isValid(target):
                self._tracks.pop(key, None)
                continue
            t = min(1.0, max(0.0, (now - began) / duration))
            eased = 1.0 - (1.0 - t) ** 3
            setter(end if t >= 1.0 else start + (end - start) * eased)
            if t >= 1.0:
                self._tracks.pop(key, None)
        self._stop_if_idle()

    def _stop_if_idle(self):
        if not self._tracks:
            self._timer.stop()

