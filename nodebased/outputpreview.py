"""The Render workspace's output preview: the picture the viewed Write node would write (the
viewer's current frame), with the node's file, format and frame range under it.

It only shows what the viewer already computed; it never evaluates anything itself."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .theme import RADIUS, TOKENS


class _Picture(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("output-preview-picture")
        self.setMinimumSize(0, 0)
        self.image = None

    def set_image(self, image):
        self.image = image if isinstance(image, QImage) and not image.isNull() else None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(TOKENS["line"]), 1))
        painter.setBrush(QColor(TOKENS["bg0"]))
        painter.drawRoundedRect(frame, RADIUS["field"], RADIUS["field"])
        if self.image is None:
            painter.setPen(QColor(TOKENS["tx3"]))
            painter.drawText(frame, Qt.AlignmentFlag.AlignCenter, "No picture yet")
            return
        inner = frame.adjusted(6, 6, -6, -6)
        scale = min(inner.width() / self.image.width(), inner.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        target = QRectF(inner.center().x() - width / 2, inner.center().y() - height / 2, width, height)
        painter.drawImage(target, self.image)


class OutputPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("output-preview")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        self.title = QLabel("No Write node")
        self.title.setObjectName("output-preview-title")
        self.details = QLabel("")
        self.details.setObjectName("output-preview-details")
        self.details.setWordWrap(True)
        self.picture = _Picture()
        layout.addWidget(self.title)
        layout.addWidget(self.details)
        layout.addWidget(self.picture, 1)

    def set_write(self, name, details):
        """`name` is the Write node's label (None when none is viewed); `details` one short line."""
        self.title.setText(name or "No Write node")
        self.details.setText(details or "Select a Write node in the graph or the viewer's input to preview it.")

    def set_image(self, image):
        self.picture.set_image(image)
