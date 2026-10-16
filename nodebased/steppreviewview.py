"""The viewer for step-by-step previews (spike A, docs/SPIKE_A_FEWSTEP.md): shows the newest picture of the newest edit.

Pictures arrive from `steppreview.PreviewSession`'s reader thread through its `LatestSlot`; the thread only pokes this
widget with a queued signal, and everything that touches a QImage or paints happens on the UI thread. While a new edit
waits for its first preview the last valid picture stays up, labelled as superseded. A picture of an older revision than
the newest submitted one is never painted. Each paint is logged (`paints`) with its monotonic time, next to the picture's
model-side time, which is what the interaction scoring subtracts.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class _Bridge(QObject):
    available = Signal()


class StepPreviewView(QWidget):
    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self.setObjectName("step-preview-view")
        self.setMinimumSize(320, 200)
        self.bridge = _Bridge()
        self.bridge.available.connect(self._drain, Qt.ConnectionType.QueuedConnection)
        self.session = None
        self.floor = 0               # the newest revision the artist asked for
        self.image = None
        self.label = "No picture yet"
        self.shown = None            # the PreviewEvent now on screen
        self.paints = []             # {"revision", "frame", "step", "kind", "model_done", "received", "drained", "painted", "flushed"}
        self.stale_dropped = 0
        self._awaiting = None        # the log row of the picture drained but not yet painted
        if session is not None:
            self.attach(session)

    def attach(self, session):
        self.session = session
        session.on_available = self.bridge.available.emit   # safe from any thread: a queued signal

    def note_submitted(self, revision):
        """The artist changed something: keep the old picture up, marked superseded, until the new one arrives."""
        self.floor = max(self.floor, revision)
        if self.image is not None:
            self.label = "Superseded - waiting for the new preview"
            self.update()

    def _drain(self):
        if self.session is None:
            return
        event = self.session.slot.take()
        if event is None:
            return
        if event.revision < self.floor:
            self.stale_dropped += 1
            return
        drained = time.monotonic()
        # The array is owned by the event; copy so the QImage never outlives it.
        self.image = QImage(event.array.data, event.width, event.height, event.width * 3,
                            QImage.Format.Format_RGB888).copy()
        self.shown = event
        self.label = (f"Final - frame {event.frame}" if event.kind == "final" else
                      f"Preview {event.step}/{event.of} - frame {event.frame} - provisional") + f" - {event.width}x{event.height}"
        self._awaiting = {"revision": event.revision, "frame": event.frame, "stage": event.stage, "step": event.step,
                          "of": event.of, "kind": event.kind, "model_done": event.done, "received": event.received,
                          "drained": drained, "width": event.width, "height": event.height}
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#101216"))
        area = QRectF(self.rect()).adjusted(8, 8, -8, -30)
        if self.image is not None:
            scale = min(area.width() / self.image.width(), area.height() / self.image.height())
            width, height = self.image.width() * scale, self.image.height() * scale
            painter.drawImage(QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height), self.image)
        provisional = self.shown is not None and self.shown.provisional or self.label.startswith("Superseded")
        painter.setPen(QPen(QColor("#e0a040" if provisional else "#9ad0a0")))
        painter.setFont(QFont(self.font().family(), 10))
        painter.drawText(QRectF(8, self.height() - 26, self.width() - 16, 22),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.label)
        painter.end()
        if self._awaiting is not None:
            row, self._awaiting = self._awaiting, None
            row["painted"] = time.monotonic()
            self.paints.append(row)
            # The backing store is flushed to the window after this paint returns; a queued zero timer runs after it.
            QTimer.singleShot(0, lambda row=row: row.__setitem__("flushed", time.monotonic()))
