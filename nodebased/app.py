"""Native Qt desktop workbench; UI writes only through Dispatcher commands."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, CancelledError
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time

from PySide6.QtCore import Qt, QLineF, QPointF, QRect, QRectF, QTimer, Signal, QObject, QEvent, QSettings, QSize
from PySide6.QtGui import (QAction, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap,
                           QKeySequence, QPolygonF, QIcon, QOffscreenSurface, QFont, QFontMetrics)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsSimpleTextItem,
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsItem, QDockWidget, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QLineEdit, QPushButton, QFormLayout, QFileDialog, QMessageBox, QToolBar,
    QInputDialog, QSplitter, QScrollArea, QDialog, QListWidget, QListWidgetItem, QStyle, QSlider,
    QCheckBox, QMenu, QSizePolicy, QProgressDialog, QTabWidget, QPlainTextEdit, QFrame,
    QColorDialog)

from . import __version__
from .updater import Updater
from .core import (Dispatcher, SPECS, LIMITS, TIME_LIMITS, demo_document, load_document,
                   MASK_MIX_KINDS, artifact_type, node_label, node_thumbnail,
                   DEFAULT_THUMBNAIL_TYPES)
from .imaging import Evaluator, Cancelled, to_qimage, write_png
from .playback import PlaybackQueue, DisplayCache

from .theme import (COLORS, STYLE, THEMES, DEFAULT_THEME, ACCENTS, build_style, grid_color,
                    valid_accent)
from .color import VIEWS
from . import gpudisplay
from .core import CHOICES
from .media import (write_exr, group_directory, IMAGE_EXTENSIONS, is_sequence, sequence_path)
from .cachetier import DiskCache
from .decodepool import DecodeAheadPool
from .tileexec import TileExecutor
from .tiles import TileRegion
from .tiers import PROXY_TIERS, auto_playback_tier
from .timeline import TimelineBar, KEY_COLOR
from .animation import CURVE_INTERPOLATIONS, resolve_document
from . import shapes as shape_model
from . import tracker as tracker_model
from .knobs import knob_layout

# Delivery rates an artist actually asks for, offered next to the free-form rate box. 24 leads
# because it is the document default; the rest are the rates a comp gets handed in practice.
FPS_PRESETS = (("24", 24.0), ("23.976", 24000.0 / 1001.0), ("25", 25.0), ("29.97", 30000.0 / 1001.0),
               ("30", 30.0), ("48", 48.0), ("50", 50.0), ("59.94", 60000.0 / 1001.0), ("60", 60.0))


# A Viewer's noodle is deliberately the quietest line in the graph: see Graph.rebuild.
VIEW_EDGE_COLOR = "#5f5f6b"


class RulerSlider(QSlider):
    """Integer-backed slider with a compact float ruler beneath its groove."""

    def __init__(self, soft_min, soft_max, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.soft_min = float(soft_min)
        self.soft_max = float(soft_max)
        self.setRange(0, 1000)
        self.setFixedHeight(38)
        self.setToolTip(f"Soft range: {self.soft_min:g} to {self.soft_max:g}")

    def float_value(self):
        if self.soft_max == self.soft_min:
            return self.soft_min
        return self.soft_min + (self.value() / 1000.0) * (self.soft_max - self.soft_min)

    def set_float_value(self, value):
        if self.soft_max == self.soft_min:
            position = 0
        else:
            value = max(self.soft_min, min(self.soft_max, float(value)))
            position = round((value - self.soft_min) / (self.soft_max - self.soft_min) * 1000)
        self.setValue(position)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(self.palette().color(self.foregroundRole()))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        left, right = 8, max(8, self.width() - 8)
        for fraction in (0.0, 0.5, 1.0):
            x = left + fraction * (right - left)
            painter.drawLine(QPointF(x, 24), QPointF(x, 28))
            label = f"{self.soft_min + fraction * (self.soft_max - self.soft_min):g}"
            bounds = painter.fontMetrics().boundingRect(label)
            painter.drawText(QRectF(x - bounds.width() / 2, 27, bounds.width(), 11),
                             Qt.AlignmentFlag.AlignCenter, label)


class FloatSliderControl(QWidget):
    """A float spin box paired with a soft-range ruler slider."""

    def __init__(self, hard_range, soft_range, value, parent=None):
        super().__init__(parent)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(*hard_range)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.1)
        self.spin.setKeyboardTracking(False)
        self.slider = RulerSlider(*(soft_range or hard_range))
        self.slider.set_float_value(value)
        self.spin.setValue(value)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.spin)
        layout.addWidget(self.slider, 1)
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.customContextMenuRequested.connect(self.customContextMenuRequested)
        self.slider.customContextMenuRequested.connect(self.customContextMenuRequested)

    def _spin_changed(self, value):
        self.slider.set_float_value(value)

    def _slider_changed(self, value):
        self.spin.setValue(self.slider.float_value())

    def value(self):
        return self.spin.value()

    def setValue(self, value):
        self.spin.setValue(value)
        self.slider.set_float_value(value)

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.spin.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def setContextMenuPolicy(self, policy):
        super().setContextMenuPolicy(policy)
        self.spin.setContextMenuPolicy(policy)
        self.slider.setContextMenuPolicy(policy)


class ClickableColorSwatch(QFrame):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


def resource_path(relative: str) -> Path:
    """Locate a source asset both from a checkout and a PyInstaller bundle."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / relative


class DisplayedFrame:
    """What `Window.frame` holds when a preview came straight from the display cache.

    The cache keeps the finished picture, not the scene-linear pixels, so there is no array to
    hand back -- and composing one only to discard it is exactly the cost the cache exists to
    avoid. Nothing needs the array: export re-renders at full resolution regardless, and the
    remaining readers only ask whether a valid frame is on screen and what shape it is.
    """
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = tuple(shape)


class Preferences:
    """Per-machine interface preferences, stored outside the document.

    A theme belongs to the artist looking at the screen, not to the comp: a .nbcomp handed to
    someone else must not repaint their application. QSettings is used rather than a file in the
    project so the two can never be confused for one another.
    """
    THEME = "interface/theme"
    THUMBNAILS = "interface/node_thumbnails"
    ACCENT = "interface/accent"

    def __init__(self):
        self._store = QSettings("NodeBased", "NodeBased")

    def theme(self):
        name = self._store.value(self.THEME, DEFAULT_THEME)
        return name if name in THEMES else DEFAULT_THEME

    def set_theme(self, name):
        if name in THEMES:
            self._store.setValue(self.THEME, name)
            self._store.sync()

    def thumbnails(self):
        value = self._store.value(self.THUMBNAILS, True)
        # QSettings hands booleans back as strings from some backends (INI on Linux).
        return value not in (False, "false", "0", 0)

    def accent(self):
        return valid_accent(self._store.value(self.ACCENT, None))

    def set_accent(self, value):
        value = valid_accent(value)
        if value is None:
            self._store.remove(self.ACCENT)
        else:
            self._store.setValue(self.ACCENT, value)
        self._store.sync()

    def set_thumbnails(self, enabled):
        self._store.setValue(self.THUMBNAILS, bool(enabled))
        self._store.sync()


def apply_theme(name, accent=None):
    """Restyle the whole running application. Returns the theme actually applied."""
    name = name if name in THEMES else DEFAULT_THEME
    application = QApplication.instance()
    if application is not None:
        application.setStyleSheet(build_style(name, accent))
    return name


class ElidedLabel(QLabel):
    """A status label whose text can never widen the layout it sits in.

    A QLabel's size hint *is* its text, so a status line that grows mid-playback -- the viewer
    gains "ahead 4/8 · dropped 12" the moment the transport starts -- raises the window's minimum
    width and visibly resizes the panels around it. That is the "viewer enlarges randomly during
    playback" bug: nothing about the image changed, only the length of a string describing it.

    `text()` deliberately still returns the whole status. The agent bridge and the tests read it
    as the real status of the last render, so truncating the stored value would trade a layout bug
    for a lying one; only the painted copy is elided, with the full text on hover.
    """
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def sizeHint(self):
        return QSize(0, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        elided = QFontMetrics(self.font()).elidedText(
            self.text(), Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), int(self.alignment()) | int(Qt.AlignmentFlag.AlignVCenter),
                         elided)



class PanZoomView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        scene.setParent(self)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#19191b"))
        self.pan = None

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if 0.05 < self.transform().m11() * factor < 20:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan is not None:
            delta = event.position() - self.pan
            self.pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan = None
            self.unsetCursor()
        else:
            super().mouseReleaseEvent(event)

    def fit(self):
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect.adjusted(-24, -24, 24, 24), Qt.AspectRatioMode.KeepAspectRatio)


class Viewer(PanZoomView):
    """Image viewer shortcuts are active only while the pointer/focus is in the viewer."""
    def __init__(self, window):
        self.window = window
        self.format_rect = None
        # Roto editing is a viewer overlay, not a scene item.  Keeping it out of the scene is
        # important: itemsBoundingRect() is the rendered display window and must not grow when
        # a point or a control path is painted (especially for EXR overscan and tiled previews).
        self.roto_key = None
        self.roto_drawing = False
        self.roto_draw_points = []
        self.roto_draw_cursor = None
        self.roto_drag = None
        self.tracker_picking = False
        super().__init__(QGraphicsScene())

    def _roto_context(self):
        """Return the selected Roto payload and display scale when it is safe to edit it.

        Coordinates in a payload are in the Roto node's own output format.  An overlay is only
        editable while that exact node is being viewed; drawing over a downstream Transform or
        Tracker would make a screen gesture mean something different from the stored coordinates.
        """
        graph = getattr(self.window, "graph", None)
        if graph is None:
            return None
        key = graph.selected_id()
        document = self.window.dispatcher.document
        node = document["nodes"].get(key) if key else None
        if node is None or node["type"] != "Roto" or document.get("view") != key:
            return None
        payload = document.get("node_data", {}).get(key, {"shapes": []})
        tier = getattr(getattr(self.window, "proxy", None), "currentData", lambda: 1)()
        try:
            tier = max(1, int(tier))
        except (TypeError, ValueError):
            tier = 1
        return key, node, payload, tier

    def _event_scene_pos(self, event):
        """Map a viewport mouse event into scene coordinates (Qt sends QMouseEvent here)."""
        return self.mapToScene(event.position().toPoint())

    def _roto_scene_point(self, point, tier):
        # preview_ready upscales a proxy pixmap back to the full format rectangle.  The scene
        # therefore remains in full-resolution display coordinates and the payload's pixel
        # coordinates are not divided by tier here.
        rect = self.format_rect
        origin_x = rect.left() if rect is not None else 0.0
        origin_y = rect.top() if rect is not None else 0.0
        return QPointF(origin_x + point["x"], origin_y + point["y"])

    def _roto_data_point(self, scene_pos, node, tier):
        rect = self.format_rect
        origin_x = rect.left() if rect is not None else 0.0
        origin_y = rect.top() if rect is not None else 0.0
        width, height = node["params"]["width"], node["params"]["height"]
        # See _roto_scene_point: the upscaled proxy still occupies the full format scene rect.
        x = scene_pos.x() - origin_x
        y = scene_pos.y() - origin_y
        return [min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))]

    def _tracker_data_point(self, scene_pos):
        rect = self.format_rect
        return [scene_pos.x() - (rect.left() if rect is not None else 0.0),
                scene_pos.y() - (rect.top() if rect is not None else 0.0)]

    def _roto_hit_point(self, scene_pos, resolved, tier):
        # A constant physical hit target remains usable at any viewer zoom.
        radius = 12.0 / max(abs(self.transform().m11()), 0.05)
        best = None
        best_distance = radius
        for shape_index, shape in enumerate(resolved):
            for point_index, point in enumerate(shape["points"]):
                scene_point = self._roto_scene_point(point, tier)
                distance = math.hypot(scene_point.x() - scene_pos.x(), scene_point.y() - scene_pos.y())
                if distance <= best_distance:
                    best = (shape_index, point_index)
                    best_distance = distance
        return best

    def _roto_scalar_at_frame(self, value, frame, replacement):
        """Change a point coordinate while retaining its v8 animated scalar envelope."""
        if not isinstance(value, dict) or "curve" not in value or value.get("curve") is None:
            return float(replacement)
        updated = copy.deepcopy(value)
        keys = updated["curve"]["keys"]
        for key in keys:
            if key["frame"] == frame:
                key["value"] = float(replacement)
                return updated
        keys.append({"frame": int(frame), "value": float(replacement)})
        keys.sort(key=lambda key: key["frame"])
        return updated

    def _roto_resolved(self, context):
        key, _, payload, _ = context
        return shape_model.resolve_shapes(payload, self.window.dispatcher.document["time"]["current"])

    def begin_roto_draw(self, key=None):
        context = self._roto_context()
        if context is None or (key is not None and context[0] != key):
            return False
        self.roto_key = context[0]
        self.roto_drawing = True
        self.roto_draw_points = []
        self.roto_draw_cursor = None
        self.roto_drag = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().update()
        self.window.statusBar().showMessage("Roto: click points · Enter closes the shape · Esc cancels")
        return True

    def cancel_roto_edit(self):
        self.roto_drawing = False
        self.roto_draw_points = []
        self.roto_draw_cursor = None
        self.roto_drag = None
        self.roto_key = None
        self.unsetCursor()
        self.viewport().update()

    def finish_roto_draw(self):
        if not self.roto_drawing:
            return False
        context = self._roto_context()
        if context is None or len(self.roto_draw_points) < shape_model.MINIMUM_SHAPE_POINTS:
            self.window.statusBar().showMessage("Roto: at least 3 points are required", 4000)
            return False
        key, _, payload, _ = context
        frame = int(self.window.dispatcher.document["time"]["current"])
        points = [{"x": x, "y": y, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
                  for x, y in self.roto_draw_points]
        names = {shape.get("name") for shape in payload.get("shapes", [])}
        index = 1
        while f"shape{index}" in names:
            index += 1
        shapes = copy.deepcopy(payload.get("shapes", []))
        shapes.append({"name": f"shape{index}", "mode": "union", "opacity": 1.0,
                       "feather": 0.0, "points": points})
        self.cancel_roto_edit()
        # Whole-payload replacement is the Dispatcher validation/undo boundary.  The local frame
        # variable documents that this draw is static; future point drags key animated scalars.
        self.window.command({"op": "set_shapes", "id": key, "shapes": shapes})
        return True

    def _commit_roto_drag(self):
        drag = self.roto_drag
        if drag is None:
            return False
        context = self._roto_context()
        if context is None:
            self.roto_drag = None
            return False
        key, _, payload, tier = context
        shape_index, point_index = drag["point"]
        shapes = copy.deepcopy(payload.get("shapes", []))
        if shape_index >= len(shapes) or point_index >= len(shapes[shape_index]["points"]):
            self.roto_drag = None
            return False
        x, y = self._roto_data_point(drag["scene"], context[1], tier)
        frame = int(self.window.dispatcher.document["time"]["current"])
        point = shapes[shape_index]["points"][point_index]
        point["x"] = self._roto_scalar_at_frame(point["x"], frame, x)
        point["y"] = self._roto_scalar_at_frame(point["y"], frame, y)
        self.roto_drag = None
        self.window.command({"op": "set_shapes", "id": key, "shapes": shapes})
        return True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.tracker_picking:
            self.tracker_picking = False
            self.unsetCursor()
            self.window.statusBar().showMessage("Tracker point picking cancelled")
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.roto_drawing:
            self.finish_roto_draw()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and (self.roto_drawing or self.roto_drag is not None):
            self.cancel_roto_edit()
            event.accept()
            return
        channel_for_key = {Qt.Key.Key_R: "R", Qt.Key.Key_G: "G", Qt.Key.Key_B: "B", Qt.Key.Key_A: "A"}
        if event.key() in channel_for_key and not event.modifiers():
            channel = channel_for_key[event.key()]
            # Nuke-style solo behavior: pressing an already-soloed channel returns to RGB.
            self.window.channels.setCurrentText("RGB" if self.window.channels.currentText() == channel else channel)
            event.accept()
        elif event.key() in (Qt.Key.Key_F, Qt.Key.Key_H) and not event.modifiers():
            # F is the direct fit command. H is the familiar home/frame alias: with a
            # single 2D image there is no separate selected-object extent to frame.
            self.fit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def mouseReleaseEvent(self, event):
        scene_pos = self._event_scene_pos(event)
        if event.button() == Qt.MouseButton.LeftButton and self.roto_drawing:
            if self.roto_draw_cursor is not None:
                context = self._roto_context()
                if context is not None:
                    self.roto_draw_points.append(self._roto_data_point(scene_pos, context[1], context[3]))
                    self.roto_draw_cursor = None
                    self.viewport().update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.roto_drag is not None:
            self.roto_drag["scene"] = scene_pos
            self.roto_drag["moved"] = (scene_pos - self.roto_drag["start"]).manhattanLength() > 2
            if not self.roto_drag["moved"]:
                self.roto_drag = None
                self.unsetCursor()
                event.accept()
                return
            self._commit_roto_drag()
            self.unsetCursor()
            event.accept()
            return
        was_panning = self.pan is not None
        super().mouseReleaseEvent(event)
        if was_panning:
            # The visible scene window changed; request only the newly exposed bounding box.
            self.window.request_preview()

    def draw_format_overlay(self, scene_rect):
        """Record the display window so drawForeground can paint Nuke-style format guides
        around it. Deliberately not scene items: itemsBoundingRect() is used elsewhere
        (see the canvas-resize regression test) to mean "the rendered image", and scene
        items would perturb that with zoom-dependent, overlay-shaped geometry."""
        self.format_rect = scene_rect
        self.viewport().update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        context = self._roto_context()
        if context is not None:
            _, _, payload, tier = context
            resolved = self._roto_resolved(context)
            if self.roto_drag is not None:
                # Keep the wireframe responsive while the document remains unchanged.  The
                # payload is only replaced on release, so a cancelled drag cannot leave a partial
                # command behind.
                shape_index, point_index = self.roto_drag["point"]
                if shape_index < len(resolved) and point_index < len(resolved[shape_index]["points"]):
                    x, y = self._roto_data_point(self.roto_drag["scene"], context[1], tier)
                    resolved[shape_index]["points"][point_index]["x"] = x
                    resolved[shape_index]["points"][point_index]["y"] = y
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for shape_index, shape in enumerate(resolved):
                points = shape["points"]
                if len(points) < 3:
                    continue
                path = QPainterPath(self._roto_scene_point(points[0], tier))
                for index in range(len(points)):
                    start, end = points[index], points[(index + 1) % len(points)]
                    path.cubicTo(self._roto_scene_point({"x": start["x"] + start["out_x"],
                                                          "y": start["y"] + start["out_y"]}, tier),
                                 self._roto_scene_point({"x": end["x"] + end["in_x"],
                                                         "y": end["y"] + end["in_y"]}, tier),
                                 self._roto_scene_point(end, tier))
                pen = QPen(QColor("#ff986f" if shape["mode"] == "subtract" else "#58d7ff"), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                for point in points:
                    center = self._roto_scene_point(point, tier)
                    radius = 5.0 / max(abs(self.transform().m11()), 0.05)
                    painter.setBrush(QColor("#202127"))
                    painter.drawEllipse(center, radius, radius)
            if self.roto_drawing:
                origin_x = self.format_rect.left() if self.format_rect is not None else 0.0
                origin_y = self.format_rect.top() if self.format_rect is not None else 0.0
                draw_points = [QPointF(origin_x + x, origin_y + y)
                               for x, y in self.roto_draw_points]
                if self.roto_draw_cursor is not None:
                    draw_points.append(self.roto_draw_cursor)
                if draw_points:
                    path = QPainterPath(draw_points[0])
                    for point in draw_points[1:]:
                        path.lineTo(point)
                    if len(draw_points) >= 3:
                        path.lineTo(draw_points[0])
                    pen = QPen(QColor("#f4ce63"), 2)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    painter.drawPath(path)
                    painter.setBrush(QColor("#f4ce63"))
                    for point in draw_points:
                        radius = 4.0 / max(abs(self.transform().m11()), 0.05)
                        painter.drawEllipse(point, radius, radius)
            if self.roto_drag is not None:
                point = self.roto_drag["scene"]
                radius = 6.0 / max(abs(self.transform().m11()), 0.05)
                painter.setBrush(QColor("#f4ce63"))
                painter.drawEllipse(point, radius, radius)
            painter.restore()
        if self.format_rect is None:
            return
        painter.save()
        pen = QPen(QColor("#6d6d78"))
        pen.setStyle(Qt.PenStyle.DotLine)
        pen.setCosmetic(True)  # a constant 1px dashed line regardless of zoom
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.format_rect)
        painter.restore()
        painter.save()
        painter.resetTransform()  # switch to raw viewport pixels for constant-size text
        corner = self.mapFromScene(self.format_rect.bottomRight())
        painter.setPen(QColor("#9a9aa4"))
        painter.drawText(corner.x() + 4, corner.y() + 14,
                         f"{int(self.format_rect.width())} x {int(self.format_rect.height())}")
        painter.restore()

    def mousePressEvent(self, event):
        scene_pos = self._event_scene_pos(event)
        if event.button() == Qt.MouseButton.LeftButton and self.tracker_picking:
            context = self.window._tracker_context()
            if context is not None:
                point = self._tracker_data_point(scene_pos)
                self.tracker_picking = False
                self.unsetCursor()
                self.window.add_tracker_point(point)
                event.accept()
                return
        context = self._roto_context()
        if event.button() == Qt.MouseButton.LeftButton and context is not None:
            if self.roto_drawing:
                self.roto_draw_cursor = scene_pos
                event.accept()
                return
            resolved = self._roto_resolved(context)
            hit = self._roto_hit_point(scene_pos, resolved, context[3])
            if hit is not None:
                self.roto_key = context[0]
                self.roto_drag = {"point": hit, "scene": scene_pos, "start": scene_pos,
                                  "moved": False}
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self._event_scene_pos(event)
        if self.roto_drawing and not event.buttons() & Qt.MouseButton.MiddleButton:
            self.roto_draw_cursor = scene_pos
            self.viewport().update()
            event.accept()
            return
        if self.roto_drag is not None and not event.buttons() & Qt.MouseButton.MiddleButton:
            self.roto_drag["scene"] = scene_pos
            self.roto_drag["moved"] = (scene_pos - self.roto_drag["start"]).manhattanLength() > 2
            self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)


def dot_grab_radius(graph):
    """Scene radius around a Dot centre that grabs the Dot: its own size, or ~8 screen px."""
    zoom = graph.transform().m11() if graph is not None else 1.0
    return max(10.0, 8.0 / max(zoom, 1e-3))


class Port(QGraphicsEllipseItem):
    def __init__(self, node, slot, x, y):
        # The hit target is intentionally much larger than the visible socket.
        # A 12 px drawn port was too easy to miss while the label/noodle occupied
        # nearby pixels, making wiring feel randomly broken.
        radius = 8 if node.is_dot else 13
        super().__init__(-radius, -radius, radius * 2, radius * 2, node)
        self.node, self.slot = node, slot
        self.setPos(x, y)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(3)
        visible = QGraphicsEllipseItem(-6, -6, 12, 12, self)
        visible.setBrush(QColor("#1b1b1d"))
        visible.setPen(QPen(QColor("#a4a4ae"), 1.5))
        visible.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._press_scene = None
        self._dragging = False
        self._rewire_input = None
        self.setToolTip("Output: drag to an input, or click then click an input" if slot is None else
                         f"Input {slot}: drag to an output; drag an output here; click to pick up and rewire; right-click disconnects")
        if slot and not node.is_dot:
            label = QGraphicsSimpleTextItem(slot, node)
            label.setBrush(QColor("#b4b4bd"))
            label.setPos(x + 9, y - 18)

    def shape(self):
        path = super().shape()
        if self.node.is_dot:
            # A Dot's two sockets sit on its rim; zoomed out, their generous hit circles swallow
            # the whole Dot. The Dot body always wins: carve it out of both sockets.
            radius = dot_grab_radius(self.node.graph)
            body = QPainterPath()
            body.addEllipse(self.mapFromItem(self.node, QPointF(10, 10)), radius, radius)
            path = path.subtracted(body)
        return path

    def mousePressEvent(self, event):
        graph = self.node.graph
        self._press_scene = event.scenePos()
        self._dragging = False
        self._rewire_input = None
        if event.button() == Qt.MouseButton.RightButton and self.slot is not None:
            graph.cancel_wire()
            graph.window.defer_command({"op": "connect", "id": self.node.key, "input": self.slot, "source": None})
        elif self.slot is None:
            if graph.wire_input:
                graph.finish_input_wire(self.node.key)
            else:
                graph.start_wire(self.node.key)
                graph.window.statusBar().showMessage("Connect: drag to an input port · Esc cancels")
        elif graph.wire_source:
            graph.finish_wire_at(event.scenePos())
        else:
            current_source = graph.window.dispatcher.document["nodes"][self.node.key]["inputs"][self.slot]
            if current_source:
                # An input is an endpoint: dragging it always seeks a new output and
                # replaces *this* input only after a valid drop.  Keep the existing
                # connection visible until then so a missed gesture cannot damage a comp.
                graph.start_input_wire(self.node.key, self.slot)
                graph.window.statusBar().showMessage("Input picked up: drag to an output · Esc or empty space keeps the original")
            else:
                graph.start_input_wire(self.node.key, self.slot)
                graph.window.statusBar().showMessage("Connect: drag to an output port · Esc cancels")
        event.accept()

    def mouseMoveEvent(self, event):
        if self._press_scene is not None and (event.scenePos() - self._press_scene).manhattanLength() > 4:
            self._dragging = True
        if graph := self.node.graph:
            if graph.wire_source or graph.wire_input:
                graph.update_pending_edge(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        # Ports own the mouse during a drag, so the destination never receives its
        # own press event. Resolve it here for the direct Nuke-style drag gesture.
        if self._dragging:
            graph = self.node.graph
            if graph.wire_source:
                graph.finish_wire_at(event.scenePos())
            elif graph.wire_input:
                graph.finish_input_wire_at(event.scenePos())
        self._press_scene = None
        self._dragging = False
        self._rewire_input = None
        event.accept()


NODE_WIDTH, NODE_HEIGHT = 190, 52
# A thumbnail band sits under the title. Nuke shows a postage stamp on the node itself; the band
# keeps the title and sockets where they always were and simply makes the card taller.
THUMB_WIDTH, THUMB_HEIGHT = 174, 72
NO_THUMBNAIL_TYPES = ("Dot", "Viewer")


def wants_thumbnail(node, thumbnails=True):
    """A node shows a stamp when the machine allows it and the node's own Node-tab switch is on."""
    return bool(thumbnails) and node["type"] not in NO_THUMBNAIL_TYPES and node_thumbnail(node)


def node_height(node, thumbnails):
    return NODE_HEIGHT + (THUMB_HEIGHT + 6 if wants_thumbnail(node, thumbnails) else 0)


def thumbnail_key(document, target, frame, view):
    """What decides a node's thumbnail: its upstream graph, the frame and the view.

    Deliberately narrower than the display-cache key. Node positions, the viewed node and every
    unrelated branch are left out, so dragging a node or editing a sibling never re-renders a
    thumbnail whose picture cannot have changed.
    """
    nodes = document["nodes"]
    expressions = document.get("expressions") or {}
    # A formula may read any node's knob, so with expressions present every node can matter.
    upstream, stack = {}, (list(nodes) if expressions else [target])
    while stack:
        key = stack.pop()
        if key in upstream or key not in nodes:
            continue
        node = nodes[key]
        # Only what changes pixels: position, name, label and the stamp switch never do.
        upstream[key] = {name: value for name, value in node.items()
                         if name not in ("pos", "name", "label", "thumbnail")}
        stack.extend(source for source in node["inputs"].values() if source)
    curves = document.get("animation", {}).get("curves", {})
    node_data = document.get("node_data") or {}
    payload = {"nodes": upstream, "curves": {key: curves[key] for key in upstream if key in curves},
               "node_data": {key: node_data[key] for key in upstream if key in node_data},
               "expressions": expressions, "frame": int(frame), "view": view, "target": target}
    return json.dumps(payload, sort_keys=True, default=str)


def top_aligned(page):
    """Keep a form's rows packed at the top of a tab instead of spread over its height."""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(page)
    layout.addStretch(1)
    return holder


class LabelEdit(QPlainTextEdit):
    """Multi-line label knob. QPlainTextEdit has no editingFinished, so focus-out stands in for it,
    matching how the single-line knobs commit."""
    finished = Signal()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.finished.emit()


class NodeItem(QGraphicsRectItem):
    def __init__(self, graph, key, node):
        self.is_dot = node["type"] == "Dot"
        self.thumbnail = None
        height = node_height(node, graph.window.show_thumbnails)
        super().__init__(0, 0, 20 if self.is_dot else NODE_WIDTH, 20 if self.is_dot else height)
        self.graph, self.key = graph, key
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(*node["pos"])
        self.setBrush(QColor("#303033" if not self.is_dot else "#23242a"))
        self.setPen(QPen(QColor(COLORS[node["type"]]), 1.5))
        self.disabled = bool(node.get("disabled", False))
        if self.disabled:
            # Keep the graph readable while making bypassed processing unmistakable.  The
            # opacity applies to the card, title, and sockets; paint() adds the persistent X.
            self.setOpacity(0.52)
        if self.is_dot:
            # Dots are graph routing points, not miniature processing cards.
            # Keep them compact and put their sockets on the vertical noodle path.
            self.setRect(0, 0, 20, 20)
            # Noodles live below nodes.  A Dot is deliberately above them so its small circular
            # control point remains visible where a Ctrl-drag inserts it into a connection.
            self.setZValue(5)
            self.inputs = {"input": Port(self, "input", 10, 0)}
            self.output = Port(self, None, 10, 20)
            return
        title = QGraphicsSimpleTextItem(node["name"][:26], self)
        title.setBrush(QColor("#eeeef2"))
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setPos((190 - title.boundingRect().width()) / 2, 5)
        # A Node-tab label draws under the name, as in Nuke. Only the first line fits the card;
        # the full text stays in the tooltip.
        label = node_label(node)
        # No type line: a node's name already says what it is, and repeating it underneath was
        # noise. The line carries only a label and state.
        caption = label.splitlines()[0][:30] if label else ""
        parts = [part for part in ("BYPASSED" if node["disabled"] else "", caption,
                                   "viewing" if graph.window.dispatcher.document["view"] == key else "")
                 if part]
        subtitle = QGraphicsSimpleTextItem("  ·  ".join(parts), self)
        self.setToolTip(f"{node['name']} ({node['type']})" + (f"\n{label}" if label else ""))
        subtitle.setBrush(QColor("#a6a6b0"))
        subtitle.setPos((190 - subtitle.boundingRect().width()) / 2, 32)
        # Inputs default to the top edge, which is where B lives: in Nuke the B stream is the
        # trunk flowing straight down through a Merge. Semantic side ports follow compositor
        # convention: A joins from the left and an optional mask from the right. Keep every
        # declared socket present even for a disabled Merge, so bypassing never hides B.
        slots = list(node["inputs"])
        spacing = 34
        top_slots = [slot for slot in slots if slot not in ("A", "mask")]
        self.inputs = {}
        for i, slot in enumerate(slots):
            if slot == "A":
                x, y = 0, 26
            elif slot == "mask":
                x, y = 190, 26
            else:
                top_i = top_slots.index(slot)
                x, y = 95 + (top_i - (len(top_slots) - 1) / 2) * spacing, 0
            self.inputs[slot] = Port(self, slot, x, y)
        if height > NODE_HEIGHT:
            self.thumbnail = QGraphicsPixmapItem(self)
            self.thumbnail.setPos((NODE_WIDTH - THUMB_WIDTH) / 2, NODE_HEIGHT)
            cached = graph.window.thumbnails.get(key)
            if cached is not None:
                self.set_thumbnail(cached[1])
        self.output = Port(self, None, 95, height)

    def set_thumbnail(self, image):
        if self.thumbnail is None:
            return
        pixmap = QPixmap.fromImage(image)
        # Centre the picture in its band; the band is fixed so cards never resize as stamps land.
        self.thumbnail.setPixmap(pixmap)
        self.thumbnail.setPos((NODE_WIDTH - pixmap.width()) / 2,
                              NODE_HEIGHT + (THUMB_HEIGHT - pixmap.height()) / 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and hasattr(self, "output"):
            self.graph.update_edges()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        if not self.is_dot:
            super().paint(painter, option, widget)
            if self.disabled:
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                pen = QPen(QColor("#f08a8a"), 5.0, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                inset = 12
                rect = self.rect().adjusted(inset, inset, -inset, -inset)
                painter.drawLine(rect.topLeft(), rect.bottomRight())
                painter.drawLine(rect.topRight(), rect.bottomLeft())
                painter.restore()
            return
        painter.save()
        pen = QPen(self.pen())
        if option.state & QStyle.StateFlag.State_Selected:
            pen.setWidthF(3)
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawEllipse(self.rect())
        painter.restore()


class Edge(QGraphicsPathItem):
    """A readable noodle with a small arrow showing output -> input direction."""
    def __init__(self, color="#898995", dashed=False, arrow=True, width=3.25):
        super().__init__()
        self.color = QColor(color)
        self.setPen(QPen(self.color, width, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.arrow = arrow
        self.arrowhead = QPolygonF()

    def set_curve(self, start, end):
        distance = max(40, abs(end.y() - start.y()) * 0.5)
        control = end - QPointF(0, distance)
        path = QPainterPath(start)
        path.cubicTo(start + QPointF(0, distance), control, end)
        tangent = end - control
        length = math.hypot(tangent.x(), tangent.y()) or 1.0
        unit = QPointF(tangent.x() / length, tangent.y() / length)
        normal = QPointF(-unit.y(), unit.x())
        base = end - unit * 10
        self.arrowhead = QPolygonF([end, base + normal * 4, base - normal * 4])
        self.setPath(path)
        self.handle = path.pointAtPercent(0.5)

    def paint(self, painter, option, widget=None):
        # Keep the curve stroked. Combining a closed arrow polygon with the curve
        # in one path causes Qt to fill the implied region as a ribbon.
        painter.save()
        painter.setPen(self.pen())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())
        if self.arrow and not self.arrowhead.isEmpty():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.color)
            painter.drawPolygon(self.arrowhead)
        painter.restore()


class NodeSearch(QDialog):
    """Small keyboard-first node picker, intentionally close to Nuke's Tab menu."""
    def __init__(self, parent, choices, global_pos):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("nodeSearch")
        self.setWindowTitle("Create node")
        self.setMinimumWidth(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search nodes…")
        self.list = QListWidget()
        self.list.setMaximumHeight(280)
        layout.addWidget(self.query)
        layout.addWidget(self.list)
        self.choices = list(choices)
        self.update_matches("")
        self.query.textChanged.connect(self.update_matches)
        self.query.returnPressed.connect(self.choose_current)
        self.list.itemActivated.connect(lambda _: self.choose_current())
        self.move(global_pos)
        self.query.setFocus()

    def update_matches(self, query):
        needle = query.casefold().strip()
        matches = [name for name in self.choices if not needle or needle in name.casefold()]
        self.list.clear()
        self.list.addItems(matches)
        if matches:
            self.list.setCurrentRow(0)

    def choose_current(self):
        item = self.list.currentItem()
        if item:
            self.selected_kind = item.text()
            self.accept()

    @classmethod
    def choose(cls, parent, choices, global_pos):
        picker = cls(parent, choices, global_pos)
        return picker.selected_kind if picker.exec() == QDialog.DialogCode.Accepted else None


class SequenceBrowser(QDialog):
    """A Read browser that understands image sequences.

    A generic file dialog shows 100 numbered EXRs as 100 rows, which is the wrong unit of work:
    the artist is loading one plate, and picking one member of it gives Read a still. This lists a
    directory with numbered frames batched into a single entry carrying its own range, the way
    Nuke's browser does, and hands back a printf pattern Read can map timeline frames onto.

    Grouping is a checkbox, enabled by default, because the one case it gets wrong is a folder of
    unrelated numbered stills -- and that case has to stay reachable.
    """
    def __init__(self, parent, directory=None):
        super().__init__(parent)
        self.setWindowTitle("Read image or sequence")
        self.setMinimumSize(720, 480)
        self.chosen = None
        layout = QVBoxLayout(self)
        path_row = QHBoxLayout()
        self.directory = QLineEdit(str(Path(directory or Path.home()).expanduser()))
        self.directory.setToolTip("Type a folder and press Return, or use Browse…")
        self.directory.returnPressed.connect(self.reload)
        up = QPushButton("Up")
        up.clicked.connect(self.go_up)
        pick = QPushButton("Browse…")
        pick.clicked.connect(self.pick_directory)
        path_row.addWidget(QLabel("Folder"))
        path_row.addWidget(self.directory, 1)
        path_row.addWidget(up)
        path_row.addWidget(pick)
        layout.addLayout(path_row)
        self.group = QCheckBox("Group image sequences into one entry")
        self.group.setChecked(True)
        self.group.setToolTip("Off lists every file separately, for a folder of unrelated stills")
        self.group.toggled.connect(self.reload)
        layout.addWidget(self.group)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda _: self.accept_current())
        layout.addWidget(self.list, 1)
        self.detail = QLabel("")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        self.list.currentItemChanged.connect(lambda *_: self.describe())
        layout.addWidget(self.detail)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.open_button = QPushButton("Open")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self.accept_current)
        buttons.addWidget(cancel)
        buttons.addWidget(self.open_button)
        layout.addLayout(buttons)
        self.reload()

    def go_up(self):
        current = Path(self.directory.text()).expanduser()
        if current.parent != current:
            self.directory.setText(str(current.parent))
            self.reload()

    def pick_directory(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose folder", self.directory.text())
        if chosen:
            self.directory.setText(chosen)
            self.reload()

    def reload(self):
        self.list.clear()
        directory = Path(self.directory.text()).expanduser()
        if not directory.is_dir():
            self.detail.setText(f"Not a folder: {directory}")
            return
        # Subfolders first, so navigating a plate tree does not mean retyping paths.
        try:
            children = sorted((entry for entry in directory.iterdir() if entry.is_dir()),
                              key=lambda entry: entry.name.casefold())
            entries = group_directory(directory, IMAGE_EXTENSIONS, self.group.isChecked())
        except OSError as error:
            self.detail.setText(str(error))
            return
        for child in children:
            item = QListWidgetItem(f"[ {child.name} ]")
            item.setData(Qt.ItemDataRole.UserRole, {"directory": str(child)})
            self.list.addItem(item)
        for entry in entries:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.list.addItem(item)
        self.detail.setText(f"{len(entries)} image entr{'y' if len(entries) == 1 else 'ies'} · "
                            f"{len(children)} subfolder{'' if len(children) == 1 else 's'}")
        if self.list.count():
            self.list.setCurrentRow(0)

    def describe(self):
        entry = self.current_entry()
        if not entry or "directory" in entry:
            return
        if entry["sequence"]:
            gap = (f"  ·  missing {', '.join(str(f) for f in entry['missing'][:12])}"
                   f"{'…' if len(entry['missing']) > 12 else ''}" if entry["missing"] else "")
            self.detail.setText(f"{entry['path']}\n{entry['frames']} frames, "
                                f"{entry['first']}–{entry['last']}{gap}")
        else:
            self.detail.setText(f"{entry['path']}\nsingle image")

    def current_entry(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def accept_current(self):
        entry = self.current_entry()
        if not entry:
            return
        if "directory" in entry:
            self.directory.setText(entry["directory"])
            self.reload()
            return
        self.chosen = entry
        self.accept()

    @classmethod
    def choose(cls, parent, directory=None):
        dialog = cls(parent, directory)
        return dialog.chosen if dialog.exec() == QDialog.DialogCode.Accepted else None


class Graph(PanZoomView):
    def __init__(self, window):
        self.window = window
        super().__init__(QGraphicsScene())
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.viewport().setMouseTracking(True)
        self.items_by_id, self.edges = {}, []
        self.wire_source = None
        self.wire_input = None
        self.picked_input = None
        self.pending_edge = None
        self.inserting_edge = None
        self.dot_preview = None
        self.ctrl_handles_visible = False
        self.last_click_scene_pos = QPointF(0, 0)
        self.scene().selectionChanged.connect(self.selection_changed)

    def _new_pending_edge(self):
        edge = Edge("#e3b18d", dashed=True, arrow=False)
        edge.setZValue(10)
        self.scene().addItem(edge)
        return edge

    def start_wire(self, source_key, picked_input=None):
        self.cancel_wire()
        self.wire_source = source_key
        self.picked_input = picked_input
        self.pending_edge = self._new_pending_edge()

    def start_input_wire(self, node_key, slot):
        self.cancel_wire()
        self.wire_input = (node_key, slot)
        self.pending_edge = self._new_pending_edge()

    def cancel_wire(self):
        if self.pending_edge is not None:
            self.scene().removeItem(self.pending_edge)
        self.wire_source = None
        self.wire_input = None
        self.picked_input = None
        self.pending_edge = None

    def cancel_dot_insert(self):
        self.inserting_edge = None
        if self.dot_preview is not None:
            self.scene().removeItem(self.dot_preview)
            self.dot_preview = None
        self.unsetCursor()

    def start_dot_insert(self, edge_data, scene_pos):
        self.cancel_dot_insert()
        self.inserting_edge = edge_data
        # A live compact Dot follows the pointer.  The graph is untouched until
        # release, so cancelling this gesture can never erase an existing noodle.
        preview = QGraphicsEllipseItem(-10, -10, 20, 20)
        preview.setBrush(QColor("#23242a"))
        preview.setPen(QPen(QColor(COLORS["Dot"]), 2))
        preview.setZValue(20)
        self.scene().addItem(preview)
        self.dot_preview = preview
        self.update_dot_preview(scene_pos)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def update_dot_preview(self, scene_pos):
        if self.dot_preview is not None:
            self.dot_preview.setPos(scene_pos)

    def update_pending_edge(self, scene_pos):
        if self.pending_edge is None:
            return
        if self.wire_source in self.items_by_id:
            self.pending_edge.set_curve(self.items_by_id[self.wire_source].output.scenePos(), scene_pos)
        elif self.wire_input and self.wire_input[0] in self.items_by_id:
            key, slot = self.wire_input
            self.pending_edge.set_curve(scene_pos, self.items_by_id[key].inputs[slot].scenePos())

    def nearest_port(self, scene_pos, input_port):
        ports = [port for node in self.items_by_id.values()
                 for port in (node.inputs.values() if input_port else [node.output])]
        return min(ports, key=lambda port: math.hypot(port.scenePos().x() - scene_pos.x(),
                                                       port.scenePos().y() - scene_pos.y()), default=None)

    def _snap_port(self, scene_pos, input_port):
        port = self.nearest_port(scene_pos, input_port)
        radius = 24 / max(self.transform().m11(), 0.05)  # 24 physical pixels
        if port and math.hypot(port.scenePos().x() - scene_pos.x(), port.scenePos().y() - scene_pos.y()) <= radius:
            return port
        return None

    def input_at(self, scene_pos):
        return self._snap_port(scene_pos, input_port=True)

    def output_at(self, scene_pos):
        return self._snap_port(scene_pos, input_port=False)

    def finish_wire_at(self, scene_pos):
        destination = self.input_at(scene_pos)
        if destination is None:
            self.cancel_wire()
            self.window.statusBar().showMessage("Wire dropped", 3000)
            return
        source, picked = self.wire_source, self.picked_input
        self.cancel_wire()
        if picked == (destination.node.key, destination.slot):
            return
        commands = []
        if picked:
            commands.append({"op": "connect", "id": picked[0], "input": picked[1], "source": None})
        commands.append({"op": "connect", "id": destination.node.key,
                         "input": destination.slot, "source": source})
        self.window.command({"op": "batch", "commands": commands})

    def finish_input_wire(self, source_key):
        if self.wire_input:
            key, slot = self.wire_input
            self.cancel_wire()
            self.window.defer_command({"op": "connect", "id": key, "input": slot, "source": source_key})

    def finish_input_wire_at(self, scene_pos):
        output = self.output_at(scene_pos)
        if output:
            self.finish_input_wire(output.node.key)
        else:
            self.cancel_wire()
            self.window.statusBar().showMessage("Wire dropped", 3000)

    def rebuild(self):
        selected = self.selected_id()
        self.scene().blockSignals(True)
        self.edges = []
        self.items_by_id = {}
        self.scene().clear()
        self.pending_edge = None  # scene().clear() already deleted the previous item, if any.
        doc = self.window.dispatcher.document
        for key, node in doc["nodes"].items():
            item = NodeItem(self, key, node)
            self.items_by_id[key] = item
            self.scene().addItem(item)
            item.setSelected(key == selected)
        for key, node in doc["nodes"].items():
            for slot, source in node["inputs"].items():
                if source:
                    # A Viewer's connection is a place the artist is looking from, not a stage in
                    # the comp. Drawing it like any other noodle makes the Viewer look like a
                    # consumer whose pixels matter downstream; it has none. Faint, dashed and
                    # arrowless says "this is a tap" without hiding where the view is pointed.
                    edge = (Edge(color=VIEW_EDGE_COLOR, dashed=True, arrow=False, width=1.6)
                            if node["type"] == "Viewer" else Edge())
                    edge.setZValue(-2 if node["type"] == "Viewer" else -1)
                    self.scene().addItem(edge)
                    self.edges.append((edge, source, key, slot))
        self.update_edges()
        self.scene().blockSignals(False)
        if self.wire_source or self.wire_input:
            # A picked-up wire's disconnect command rebuilds the scene mid-drag; keep the preview alive.
            if self.wire_source in self.items_by_id or (self.wire_input and self.wire_input[0] in self.items_by_id):
                self.pending_edge = self._new_pending_edge()
                self.update_pending_edge(self.mapToScene(self.viewport().mapFromGlobal(QCursor.pos())))
            else:
                self.cancel_wire()

    def update_edges(self):
        for edge, source, key, slot in self.edges:
            start = self.items_by_id[source].output.scenePos()
            end = self.items_by_id[key].inputs[slot].scenePos()
            edge.set_curve(start, end)

    def selected_id(self):
        return next((i.key for i in self.scene().selectedItems() if isinstance(i, NodeItem)), None)

    def selection_changed(self):
        self.window.inspect(self.selected_id())
        # The keyed-frame band is scoped to the selection, so it has to follow it.
        self.window.refresh_timeline_marks()

    def mousePressEvent(self, event):
        self.ctrl_handles_visible = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.viewport().update()
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            scene_pos = self.mapToScene(event.position().toPoint())
            edge = self.edge_handle_at(scene_pos)
            # A Dot already sits on its noodle; Ctrl-clicking the Dot selects it, not the wire.
            if edge and self.dot_at(scene_pos) is None:
                self.start_dot_insert(edge, self.mapToScene(event.position().toPoint()))
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_click_scene_pos = self.mapToScene(event.position().toPoint())
        if (self.wire_source or self.wire_input) and event.button() == Qt.MouseButton.LeftButton and self.port_item_at(event.position().toPoint()) is None:
            self.cancel_wire()
            self.window.statusBar().showMessage("Wire dropped", 3000)
            event.accept()
            return
        super().mousePressEvent(event)

    def dot_at(self, scene_pos):
        radius = dot_grab_radius(self)
        for item in self.items_by_id.values():
            if item.is_dot and QLineF(item.sceneBoundingRect().center(), scene_pos).length() <= radius:
                return item
        return None

    def port_item_at(self, viewport_pos):
        """Resolve the visible socket child back to its large invisible Port target."""
        item = self.itemAt(viewport_pos)
        while item is not None:
            if isinstance(item, Port):
                return item
            item = item.parentItem()
        return None

    def mouseMoveEvent(self, event):
        visible = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if visible != self.ctrl_handles_visible:
            self.ctrl_handles_visible = visible
            self.viewport().update()
        if self.inserting_edge is not None:
            self.update_dot_preview(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)
        if self.wire_source or self.wire_input:
            self.update_pending_edge(self.mapToScene(event.position().toPoint()))

    def mouseReleaseEvent(self, event):
        if self.inserting_edge is not None and event.button() == Qt.MouseButton.LeftButton:
            edge, source, destination, slot = self.inserting_edge
            pos = self.mapToScene(event.position().toPoint()) - QPointF(10, 10)
            self.cancel_dot_insert()
            dot_id = __import__("uuid").uuid4().hex[:12]
            # One atomic edit: a failed validation leaves the original connection untouched.
            self.window.command({"op": "batch", "commands": [
                {"op": "create", "id": dot_id, "type": "Dot", "pos": [pos.x(), pos.y()]},
                {"op": "connect", "id": dot_id, "input": "input", "source": source},
                {"op": "connect", "id": destination, "input": slot, "source": dot_id},
            ]})
            event.accept()
            return
        super().mouseReleaseEvent(event)
        edits = []
        for key, item in self.items_by_id.items():
            pos = [round(item.pos().x(), 2), round(item.pos().y(), 2)]
            if pos != self.window.dispatcher.document["nodes"][key]["pos"]:
                edits.append({"op": "move", "id": key, "pos": pos})
        if edits:
            # Defer rebuild until QGraphicsScene has finished delivering this event.
            QTimer.singleShot(0, lambda: self.window.command({"op": "batch", "commands": edits}, render=False))

    def keyPressEvent(self, event):
        key = self.selected_id()
        if event.key() == Qt.Key.Key_Tab:
            self.window.node_search()
        elif event.key() == Qt.Key.Key_F:
            self.fit()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_wire()
            self.cancel_dot_insert()
        elif event.key() == Qt.Key.Key_1 and key:
            self.window.command({"op": "view", "id": key})
        elif event.key() == Qt.Key.Key_D and key:
            self.window.command({"op": "disable", "id": key, "value": not self.window.dispatcher.document["nodes"][key]["disabled"]})
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            edits = [{"op": "delete", "id": item.key} for item in self.scene().selectedItems() if isinstance(item, NodeItem)]
            self.window.command({"op": "batch", "commands": edits})
        elif event.key() in (Qt.Key.Key_R, Qt.Key.Key_G, Qt.Key.Key_M, Qt.Key.Key_T, Qt.Key.Key_B, Qt.Key.Key_C, Qt.Key.Key_S, Qt.Key.Key_O,
                              Qt.Key.Key_P, Qt.Key.Key_U, Qt.Key.Key_Y, Qt.Key.Key_W):
            self.window.add_node({Qt.Key.Key_R: "Read", Qt.Key.Key_G: "Grade", Qt.Key.Key_M: "Merge", Qt.Key.Key_T: "Transform",
                                   Qt.Key.Key_B: "Blur", Qt.Key.Key_C: "Crop", Qt.Key.Key_S: "Shuffle", Qt.Key.Key_O: "ColorCorrect",
                                   Qt.Key.Key_P: "Premult", Qt.Key.Key_U: "Unpremult",
                                   Qt.Key.Key_Y: "Dot", Qt.Key.Key_W: "Switch"}[event.key()])
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Control:
            self.ctrl_handles_visible = False
            self.viewport().update()
        super().keyReleaseEvent(event)

    def edge_handle_at(self, scene_pos):
        radius = 14 / max(self.transform().m11(), 0.05)
        for edge, source, key, slot in self.edges:
            handle = getattr(edge, "handle", None)
            if handle and math.hypot(handle.x() - scene_pos.x(), handle.y() - scene_pos.y()) <= radius:
                return edge, source, key, slot
        return None

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        if self.transform().m11() < 0.25:
            return
        painter.setPen(getattr(self, "grid_pen", None) or QPen(QColor(grid_color()), 1))
        left, top = math.floor(rect.left() / 32) * 32, math.floor(rect.top() / 32) * 32
        points = [QPointF(x, y) for x in range(left, int(rect.right()), 32) for y in range(top, int(rect.bottom()), 32)]
        painter.drawPoints(points)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.ctrl_handles_visible:
            return
        scale = max(self.transform().m11(), 0.05)
        radius = 8 / scale
        painter.setPen(QPen(QColor("#f0c39d"), max(1.5 / scale, 0.75)))
        painter.setBrush(QColor("#242428"))
        for edge, *_ in self.edges:
            if hasattr(edge, "handle"):
                painter.drawEllipse(edge.handle, radius, radius)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#f0c39d"))
                painter.drawEllipse(edge.handle, radius * 0.35, radius * 0.35)
                painter.setPen(QPen(QColor("#f0c39d"), max(1.5 / scale, 0.75)))
                painter.setBrush(QColor("#242428"))


class PreviewSignals(QObject):
    finished = Signal(object, object, object, str, object)
    # A stand-in picture shown while the real one renders: (payload, image, status, scale, region).
    interim = Signal(object, object, str, int, object)
    # A finished node thumbnail: (node id, thumbnail key, image).
    thumbnail = Signal(str, str, object)


class ProjectSettingsDialog(QDialog):
    """Small project-settings surface modelled after a compositor's project settings.

    The bundled ACES config, display, and ACEScg processing space are explicit rather than
    magical constants. They are read-only until external OCIO configs are supported; the artist
    can choose the saved default view and the viewer background today.
    """
    CUSTOM_ACCENT = "Custom…"

    def __init__(self, settings, parent=None, theme=DEFAULT_THEME, thumbnails=True, accent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        interface_heading = QLabel("INTERFACE")
        interface_heading.setObjectName("brand")
        layout.addWidget(interface_heading)
        interface = QFormLayout()
        self.theme = QComboBox()
        self.theme.addItems(list(THEMES))
        self.theme.setCurrentText(theme if theme in THEMES else DEFAULT_THEME)
        self.theme.setToolTip("Applies immediately to the whole application")
        interface.addRow("Theme colour", self.theme)
        self.accent = QComboBox()
        self.accent.setObjectName("accent")
        for name, value in ACCENTS.items():
            self.accent.addItem(name, value)
        accent = valid_accent(accent)
        if accent is not None and self.accent.findData(accent) < 0:
            self.accent.addItem(f"Custom {accent}", accent)
        self.accent.addItem(self.CUSTOM_ACCENT, "custom")
        self.accent.setCurrentIndex(max(0, self.accent.findData(accent)))
        self._accent_index = self.accent.currentIndex()
        self.accent.setToolTip("Highlight colour for focus rings, selections and headings")
        self.accent.activated.connect(self._accent_activated)
        interface.addRow("Accent colour", self.accent)
        self.thumbnails = QCheckBox("Show thumbnails on nodes")
        self.thumbnails.setChecked(bool(thumbnails))
        self.thumbnails.setToolTip("Each node shows a small picture of its output at the current frame")
        interface.addRow("Node graph", self.thumbnails)
        layout.addLayout(interface)
        theme_note = QLabel("The theme is stored per machine, not in the project: a comp handed to "
                            "another artist keeps their colours, not yours. Node colours stay fixed "
                            "so a node family always reads the same.")
        theme_note.setWordWrap(True)
        theme_note.setObjectName("muted")
        layout.addWidget(theme_note)
        heading = QLabel("COLOR MANAGEMENT")
        heading.setObjectName("brand")
        layout.addWidget(heading)
        form = QFormLayout()
        color = settings["color"]
        config_name = QLabel("ACES CG Config v4.0 · ACES 2.0")
        config_name.setToolTip(color["config"])
        form.addRow("OCIO config", config_name)
        form.addRow("Working space", QLabel(color["working_space"] + " · scene-linear float32"))
        form.addRow("Display", QLabel(color["display"]))
        self.view = QComboBox()
        self.view.addItems(VIEWS)
        self.view.setCurrentText(color["view"])
        form.addRow("Default view", self.view)
        self.background = QComboBox()
        self.background.addItem("Pure black", "black")
        self.background.addItem("Checkerboard", "checker")
        self.background.setCurrentIndex(max(0, self.background.findData(settings["viewer"]["background"])))
        form.addRow("Viewer background", self.background)
        layout.addLayout(form)
        note = QLabel("Input nodes still own source color-space overrides. Auto-detected inputs are "
                      "converted into ACEScg before entering the graph.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        apply = QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(apply)
        layout.addLayout(buttons)

    def changes(self):
        return {"color": {"view": self.view.currentText()},
                "viewer": {"background": self.background.currentData()}}

    def _accent_activated(self, index):
        if self.accent.itemData(index) != "custom":
            self._accent_index = index
            return
        from PySide6.QtWidgets import QColorDialog
        start = self.accent.itemData(self._accent_index) or "#83cbb7"
        color = QColorDialog.getColor(QColor(start), self, "Accent colour")
        if not color.isValid():
            self.accent.setCurrentIndex(self._accent_index)
            return
        value = color.name().lower()
        found = self.accent.findData(value)
        if found < 0:
            found = self.accent.count() - 1
            self.accent.insertItem(found, f"Custom {value}", value)
        self.accent.setCurrentIndex(found)
        self._accent_index = found

    def chosen_accent(self):
        value = self.accent.currentData()
        return None if value == "custom" else value

    def chosen_theme(self):
        """The interface theme, returned separately from `changes` because it is a machine
        preference rather than a document setting and must not travel inside the comp."""
        return self.theme.currentText()


class Window(QMainWindow):
    def __init__(self, document=None, agent_name=None):
        super().__init__()
        self.dispatcher = Dispatcher(document or demo_document())
        self.saved_document = copy.deepcopy(self.dispatcher.document)
        self.preferences = Preferences()
        self.theme_name = self.preferences.theme()
        self.accent_color = self.preferences.accent()
        self.show_thumbnails = self.preferences.thumbnails()
        self.properties_tab = 0
        self.rendered_identity = None
        # node id -> (thumbnail_key, QImage). Lives on the window so a graph rebuild keeps them.
        self.thumbnails = {}
        self.thumbnail_cancel = threading.Event()
        self.thumbnails_closed = False
        self.thumbnail_timer = QTimer(self)
        self.thumbnail_timer.setSingleShot(True)
        self.thumbnail_timer.setInterval(250)
        self.thumbnail_timer.timeout.connect(self.schedule_thumbnails)
        # Where the sequence browser opens next. Kept on the window rather than in the document:
        # it is navigation history, not part of the comp.
        self.last_browse_directory = None
        self.project_path = None
        self.update_exit = False
        self.frame = None
        self.frame_generation = -1
        self.generation = 0
        # QStatusBar.currentMessage() is transient: an asynchronous preview can replace it while
        # a deferred document command is reporting an error. Keep command failures on a stable
        # surface as well, so the actionable validation error remains visible after the event loop
        # advances.
        self.last_command_error = None
        self.busy = False
        # Bounded live history of evaluation errors, oldest evicted first. Exists so an attached
        # agent can see every render failure through the "errors" op (see agent_command) instead
        # of only the single most recent one visible in the status bar -- read-ahead can error on
        # a frame that never becomes "current" and reaches viewer_info at all.
        self.render_errors = deque(maxlen=200)
        self.preview_queue = PlaybackQueue()
        self.display_cache = DisplayCache()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nodebased-preview")
        # QOffscreenSurface must be created on the GUI thread; the QOpenGLContext bound to
        # it is built lazily on nodebased-preview (the single worker thread above) on its
        # first display request and used only from that thread afterward. See the design
        # note at the top of gpudisplay.py.
        gpu_surface = QOffscreenSurface()
        gpu_surface.create()
        gpudisplay.configure_surface(gpu_surface, owner_thread_prefix="nodebased-preview")
        self._gpu_surface = gpu_surface
        self._tracker_future = None
        self._tracker_cancel = None
        self._tracker_index = None
        self._tracker_seed = None
        self._tracker_key = None
        self._tracker_job = None
        # The desktop app is where the persistent disk tier is switched on: results evicted from
        # memory survive a restart, so reopening yesterday's comp does not recompute it.
        self.evaluator = Evaluator(disk=DiskCache.shared())
        # Bounded, separately-threaded read-ahead for Read-node source decode only -- see
        # decodepool.py. Decode shares no state with the single-owner Evaluator/TileCache above,
        # so it is safe to run several of these in parallel while the preview worker stays single.
        self.decode_pool = DecodeAheadPool()
        # Preview takes the tile path when every upstream node supports it. The executor reports
        # an explicit fallback for unsupported graphs; export remains the reference evaluator.
        self.tile_executor = TileExecutor(evaluator=self.evaluator, decode_pool=self.decode_pool)
        self.signals = PreviewSignals()
        self.signals.finished.connect(self.preview_ready)
        self.signals.interim.connect(self.preview_interim)
        self.signals.thumbnail.connect(self.thumbnail_ready)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self.start_preview)
        self.playing = False
        self.playback_origin_frame = 0
        self.playback_origin_time = 0.0
        self.playback_elapsed_frames = 0
        self.playback_dropped_frames = 0
        # Nuke-style playback fallback: count of primary (display) frames actually completed
        # since playback started. playback_tick never lets the playhead advance more than one
        # frame past this, so a render pipeline that can't sustain real time degrades to slower,
        # strictly in-order playback instead of racing ahead on the wall clock and jumping to
        # wherever it landed -- including backward -- once a slow frame finally finishes.
        self.playback_frames_rendered = 0
        # Set only while playback has auto-switched the proxy combo away from an artist's
        # explicit "Full" choice; holds the index to restore on stop. See toggle_playback.
        self.playback_auto_proxy_index = None
        # Frame the properties panel was last built for. Animated knobs read their value at the
        # playhead, so the panel has to be rebuilt when that moves; this is how the rebuild is
        # kept to actual frame changes. See refresh_animated_panel.
        self.panel_frame = None
        self.playback_timer = QTimer(self)
        self.playback_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.playback_timer.timeout.connect(self.playback_tick)
        self.setWindowTitle("NodeBased · Untitled")
        icon_path = resource_path("assets/nodebased-icon.png")
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1440, 920)
        toolbar = QToolBar("Workspace")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        brand = QLabel("◈  NODEBASED")
        brand.setObjectName("brand")
        toolbar.addWidget(brand)
        toolbar.addSeparator()
        for name, callback in [("Open image", self.read_file), ("Add node", self.add_node),
                               ("Save project", self.save_project), ("Export image", self.export)]:
            action = toolbar.addAction(name)
            action.triggered.connect(lambda checked=False, fn=callback: fn())
        toolbar.addSeparator()
        info = QLabel("  2D WORKSPACE")
        info.setObjectName("muted")
        toolbar.addWidget(info)
        # An expanding spacer is how a QToolBar right-justifies: everything added after it is
        # pushed to the trailing edge and stays there as the window is resized.
        spacer = QWidget()
        spacer.setObjectName("toolbarSpacer")
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.update_button = QPushButton("Check for updates")
        self.update_button.setObjectName("update")
        self.update_button.setToolTip(f"NodeBased {__version__}")
        toolbar.addWidget(self.update_button)
        self.updater = Updater(self)
        self.updater.changed.connect(self.update_status)
        self.update_button.clicked.connect(self.update_clicked)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(splitter)
        viewer_panel = QWidget()
        vl = QVBoxLayout(viewer_panel)
        vl.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("  VIEWER"))
        self.channels = QComboBox()
        self.channels.addItems(["RGB", "R", "G", "B", "A"])
        self.channels.currentTextChanged.connect(self.request_preview)
        controls.addWidget(self.channels)
        self.display_view = QComboBox()
        self.display_view.addItems(VIEWS)
        self.display_view.setCurrentText(self.dispatcher.document["settings"]["color"]["view"])
        self.display_view.setToolTip("Display transform only; exports stay independent of the viewer")
        self.display_view.currentTextChanged.connect(self.request_preview)
        controls.addWidget(self.display_view)
        # Proxy is viewer state, never document state: it is how one artist is looking at the comp
        # right now. Export and the agent's render op always evaluate at tier 1 (clause C3).
        self.proxy = QComboBox()
        for label, tier in (("Full", 1), ("1/2", 2), ("1/4", 4)):
            self.proxy.addItem(label, tier)
        self.proxy.setToolTip("Proxy resolution for the viewer only. Sources generate at this "
                              "scale and pixel-unit parameters scale with them; exports are "
                              "always full resolution.")
        self.proxy.currentIndexChanged.connect(self.request_preview)
        controls.addWidget(self.proxy)
        # Proxy-while-playing used to be unconditional. It is the right default -- native 4K
        # through ACES 2.0 cannot hit real time on the CPU -- but "the viewer silently changed
        # resolution when I pressed play" is a decision the artist gets to make, not one the app
        # makes for them. Unchecking it plays at whatever tier is selected, however slow that is.
        self.playback_proxy = QCheckBox("Proxy while playing")
        self.playback_proxy.setChecked(True)
        self.playback_proxy.setToolTip(
            "Drop to the smallest proxy tier that can keep up, for the duration of playback only, "
            "then restore the tier you had. Never overrides a proxy tier you chose yourself.\n"
            "Uncheck to always play at the selected tier.")
        controls.addWidget(self.playback_proxy)
        controls.addWidget(QLabel("Exposure"))
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(-10, 10)
        self.exposure.setSingleStep(0.25)
        self.exposure.valueChanged.connect(self.request_preview)
        controls.addWidget(self.exposure)
        fit = QPushButton("Fit")
        fit.clicked.connect(lambda: self.viewer.fit())
        controls.addWidget(fit)
        one = QPushButton("1:1")
        one.clicked.connect(lambda: self.viewer.resetTransform())
        controls.addWidget(one)
        controls.addStretch()
        # Elided, so the length of the playback status can never resize the layout around it.
        self.viewer_info = ElidedLabel("Waiting for image")
        self.viewer_info.setObjectName("muted")
        self.viewer_info.setMinimumWidth(180)
        controls.addWidget(self.viewer_info, 1)
        self.command_error_label = ElidedLabel("")
        self.command_error_label.setObjectName("command-error")
        self.command_error_label.setStyleSheet("color: #e3b18d")
        # Same rule for the status bar: a long validation message must report a problem, not
        # cause a second one by stretching the window it is reported in.
        self.statusBar().addPermanentWidget(self.command_error_label, 1)
        vl.addLayout(controls)
        self.viewer = Viewer(self)
        self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # The viewer is a window onto an image of any size; its own footprint must not follow the
        # comp's format. AdjustIgnored is Qt's default and is stated here because it is load-bearing
        # for that promise, not incidental.
        self.viewer.setSizeAdjustPolicy(QGraphicsView.SizeAdjustPolicy.AdjustIgnored)
        vl.addWidget(self.viewer)
        vl.addLayout(self._timeline())
        splitter.addWidget(viewer_panel)
        graph_panel = QWidget()
        gl = QVBoxLayout(graph_panel)
        gl.setContentsMargins(0, 0, 0, 0)
        # Elided: one long single-line hint must not set the floor for the whole window's width.
        help_label = ElidedLabel("  NODE GRAPH     Tab search/add  ·  R/G/M/T/B/C/S/O create  ·  1 view  ·  D bypass  ·  F frame  ·  MMB pan  ·  "
                                 "drag output ↔ input to wire  ·  Ctrl-drag noodle midpoint inserts Dot  ·  click a wired input to rewire")
        help_label.setObjectName("muted")
        gl.addWidget(help_label)
        self.graph = Graph(self)
        self.graph.setSizeAdjustPolicy(QGraphicsView.SizeAdjustPolicy.AdjustIgnored)
        gl.addWidget(self.graph)
        # Tab is graph-contextual: it follows the mouse over the graph even if a
        # dock/editor owns Qt keyboard focus. QGraphicsView otherwise uses Tab for
        # focus traversal before Graph.keyPressEvent can see it.
        QApplication.instance().installEventFilter(self)
        splitter.addWidget(graph_panel)
        splitter.setSizes([500, 350])
        dock = QDockWidget("PROPERTIES", self)
        dock.setMinimumWidth(310)
        self.properties = QScrollArea()
        self.properties.setWidgetResizable(True)
        dock.setWidget(self.properties)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._menus()
        # Restore the stored theme before the first paint, so the app never flashes the default.
        self.apply_theme_name(self.theme_name, self.accent_color)
        self.graph.rebuild()
        self.inspect(None)
        self.server = None
        if agent_name:
            from .agent import LocalBridge
            self.server = LocalBridge(agent_name, self.agent_command, self)
        QTimer.singleShot(0, self.graph.fit)
        self.request_preview()

    def _timeline(self):
        """Playhead strip under the viewer. Scrubbing is a document edit, so it undoes like one."""
        row = QHBoxLayout()
        row.addWidget(QLabel("  TIME"))
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(34)
        self.play_button.setToolTip("Play / stop (Space)")
        self.play_button.clicked.connect(lambda: self.toggle_playback())
        row.addWidget(self.play_button)
        self.frame_first = QSpinBox()
        self.frame_first.setRange(*TIME_LIMITS["first"])
        self.frame_first.setToolTip("First frame of the comp's range")
        row.addWidget(self.frame_first)
        # TimelineBar keeps the QSlider surface this row was built against (setRange/setValue/
        # value/valueChanged), and adds the tick marks, frame numbers, and the cached/keyed
        # underlines. Everything below wires to it unchanged.
        self.frame_slider = TimelineBar()
        row.addWidget(self.frame_slider, 1)
        self.frame_last = QSpinBox()
        self.frame_last.setRange(*TIME_LIMITS["last"])
        self.frame_last.setToolTip("Last frame of the comp's range")
        row.addWidget(self.frame_last)
        self.frame_current = QSpinBox()
        self.frame_current.setRange(*TIME_LIMITS["current"])
        self.frame_current.setToolTip("Current frame")
        row.addWidget(self.frame_current)
        # Playback rate is a property of the comp, so it is an undoable document edit through the
        # same boundary as the range — an agent setting fps and an artist typing it share one path.
        self.frame_fps = QDoubleSpinBox()
        self.frame_fps.setRange(*TIME_LIMITS["fps"])
        self.frame_fps.setDecimals(3)
        self.frame_fps.setSingleStep(1.0)
        # Commit on enter/focus-out rather than per keystroke, so typing "29.97" is one undoable
        # edit instead of four intermediate rates.
        self.frame_fps.setKeyboardTracking(False)
        self.frame_fps.setSuffix(" fps")
        self.frame_fps.setToolTip("Playback rate. New comps start at 24 fps; the transport and the "
                                  "dropped-frame counter both follow this value.")
        row.addWidget(self.frame_fps)
        self.fps_presets = QComboBox()
        self.fps_presets.setToolTip("Common delivery rates")
        self.fps_presets.addItem("rate", None)
        for label, value in FPS_PRESETS:
            self.fps_presets.addItem(label, value)
        self.fps_presets.currentIndexChanged.connect(self.apply_fps_preset)
        row.addWidget(self.fps_presets)
        self.frame_info = QLabel("")
        self.frame_info.setObjectName("muted")
        row.addWidget(self.frame_info)
        for widget, name in ((self.frame_first, "first"), (self.frame_last, "last"),
                             (self.frame_current, "current")):
            widget.valueChanged.connect(lambda value, key=name: self.set_time(**{key: value}))
        self.frame_fps.valueChanged.connect(lambda value: self.set_time(fps=float(value)))
        self.frame_slider.valueChanged.connect(lambda value: self.set_time(current=value))
        self.sync_timeline()
        return row

    def sync_timeline(self):
        """Push document time into the widgets without re-emitting edits back into the dispatcher."""
        time_range = self.dispatcher.document["time"]
        widgets = (self.frame_first, self.frame_last, self.frame_current, self.frame_slider,
                   self.frame_fps, self.fps_presets)
        for widget in widgets:
            widget.blockSignals(True)
        self.frame_slider.setRange(time_range["first"], time_range["last"])
        self.frame_current.setRange(time_range["first"], time_range["last"])
        self.frame_first.setValue(time_range["first"])
        self.frame_last.setValue(time_range["last"])
        self.frame_current.setValue(time_range["current"])
        self.frame_slider.setValue(time_range["current"])
        self.frame_fps.setValue(float(time_range["fps"]))
        # The preset box follows the rate rather than leading it, so a document opened at 23.976
        # shows that name instead of leaving a stale selection from the previous comp.
        match = next((index for index in range(1, self.fps_presets.count())
                      if abs(self.fps_presets.itemData(index) - time_range["fps"]) < 1e-6), 0)
        self.fps_presets.setCurrentIndex(match)
        for widget in widgets:
            widget.blockSignals(False)
        span = time_range["last"] - time_range["first"] + 1
        seconds = span / float(time_range["fps"])
        self.frame_info.setText(f"{span} frame{'' if span == 1 else 's'} · {seconds:.2f} s")
        self.refresh_timeline_marks()
        self.refresh_animated_panel()

    def refresh_animated_panel(self):
        """Re-read the properties panel when the playhead lands on a new frame.

        Only when the selected node is actually animated, and never during playback. An animated
        knob displays its value *at the current frame*, and its key button says whether a key
        exists here -- both go stale the instant the playhead moves. Rebuilding unconditionally
        would destroy the widget under an artist mid-edit and cost a panel rebuild on every
        transport tick, so the narrow condition is the point rather than an optimisation.
        """
        frame = self.dispatcher.document["time"]["current"]
        if frame == self.panel_frame:
            return
        self.panel_frame = frame
        if self.playing or getattr(self, "graph", None) is None:
            return
        key = self.graph.selected_id()
        if key and ((self.dispatcher.document.get("animation") or {}).get("curves", {}).get(key)
                    or (self.dispatcher.document.get("expressions") or {}).get(key)):
            self.inspect(key)

    def refresh_timeline_marks(self):
        """Push the cached-frame and keyed-frame sets onto the strip.

        Cached frames are read straight out of the display cache under the current viewer
        identity, so the answer is exact: an LRU eviction stops showing immediately, and a graph
        edit changes the identity so the whole band clears without any invalidation hook to keep
        in sync with the edit operations.

        Keyed frames follow the selection, the way Nuke's does: with a node selected you see that
        node's keys, which is what you are actually animating. With nothing selected the strip
        shows every key in the comp, so an unselected animated graph is not silently blank.
        """
        document = self.dispatcher.document
        time_range = document["time"]
        frames = range(time_range["first"], time_range["last"] + 1)
        try:
            # A frame counts as cached at any proxy tier. Playback with "Proxy while playing"
            # fills a proxy tier; judging the band by the tier selected after stopping made it
            # collapse to the one or two frames that happened to be shown at full resolution.
            cached = set()
            for tier in PROXY_TIERS:
                identity = DisplayCache.identity(
                    document, document.get("view"), tier,
                    self.display_view.currentText(), self.exposure.value(),
                    self.channels.currentText(), document["settings"]["viewer"]["background"])
                cached |= self.display_cache.resident_frames(identity, frames)
        except Exception:
            # A malformed in-flight document must never take the timeline down with it; an empty
            # cache band is a truthful "we don't know" rather than a crash.
            cached = set()
        curves = (document.get("animation") or {}).get("curves") or {}
        # _timeline() syncs itself while it is being built, which is before the graph view exists.
        selected = self.graph.selected_id() if getattr(self, "graph", None) is not None else None
        scope = {selected: curves[selected]} if selected in curves else ({} if selected else curves)
        keyed = {key["frame"] for node_curves in scope.values()
                 for curve in node_curves.values() for key in curve["keys"]}
        self.frame_slider.set_marks(cached, keyed)

    def apply_fps_preset(self, index):
        rate = self.fps_presets.itemData(index)
        if rate is not None:
            self.set_time(fps=float(rate))

    def set_time(self, transient=False, playhead_only=False, **changes):
        """Single funnel for every playhead/range edit — UI, keys and agents share this boundary."""
        current = self.dispatcher.document["time"]
        if all(current.get(key) == value for key, value in changes.items()):
            return
        # Changing the rate mid-play re-anchors the transport on the current frame. Without this the
        # wall-clock origin still belongs to the old rate and the playhead jumps to wherever the new
        # rate says the elapsed time landed.
        if "fps" in changes and self.playing:
            self.playback_origin_frame = current["current"]
            self.playback_origin_time = time.monotonic()
            self.playback_elapsed_frames = 0
            self.playback_timer.start(max(5, round(500 / float(changes["fps"]))))
        # A range edit that would strand the playhead clamps it rather than failing validation.
        if "first" in changes and changes["first"] > current["last"]:
            changes.setdefault("last", changes["first"])
        if "last" in changes and changes["last"] < current["first"]:
            changes.setdefault("first", changes["last"])
        if transient:
            try:
                self.dispatcher.execute({"op": "time", "transient": True, **changes})
                self.sync_timeline()
                self.update_title()
                self.request_preview(playhead_only=playhead_only)
            except (ValueError, KeyError, TypeError) as error:
                self._show_command_error(error)
        else:
            self.command({"op": "time", **changes})

    def step_frame(self, delta):
        if self.playing:
            self.toggle_playback(False)
        time_range = self.dispatcher.document["time"]
        target = min(max(time_range["current"] + delta, time_range["first"]), time_range["last"])
        self.set_time(current=target)

    def future_frames(self, frame, count=3):
        """Return bounded forward read-ahead order, looping inside the comp range."""
        time_range = self.dispatcher.document["time"]
        first, last = time_range["first"], time_range["last"]
        span = last - first + 1
        return [first + ((frame - first + offset) % span)
                for offset in range(1, min(count, max(0, span - 1)) + 1)]

    def toggle_playback(self, checked=None):
        start = (not self.playing) if checked is None else bool(checked)
        if start == self.playing:
            return
        self.playing = start
        self.play_button.setText("■" if start else "▶")
        if start:
            time_range = self.dispatcher.document["time"]
            self.playback_origin_frame = time_range["current"]
            self.playback_origin_time = time.monotonic()
            self.playback_elapsed_frames = 0
            self.playback_dropped_frames = 0
            self.playback_frames_rendered = 0
            # Standard proxy-resolution playback: a source above HD makes the ACES 2.0 CPU
            # transform too slow for real-time (measured ~2.3s at 4K), so drop to the smallest
            # downscale that brings it under budget for the duration of playback only. Never
            # touches a tier the artist already chose below Full themselves -- that is their
            # call, not an automatic override.
            if self.playback_proxy.isChecked() and self.proxy.currentData() == 1:
                try:
                    target = self.dispatcher.document.get("view")
                    if target and self.tile_executor.supports_tiled(self.dispatcher.document, target):
                        bounds = self.tile_executor.canvas_region(self.dispatcher.document, target,
                                                                   frame=time_range["current"], tier=1)
                        tier = auto_playback_tier(bounds.width, bounds.height)
                        if tier != 1:
                            for index in range(self.proxy.count()):
                                if self.proxy.itemData(index) == tier:
                                    self.playback_auto_proxy_index = self.proxy.currentIndex()
                                    self.proxy.setCurrentIndex(index)
                                    break
                except Exception:
                    pass
            self.playback_timer.start(max(5, round(500 / time_range["fps"])))
            self.request_preview()
        else:
            self.playback_timer.stop()
            self.preview_queue.cancel()
            if self.playback_auto_proxy_index is not None:
                self.proxy.setCurrentIndex(self.playback_auto_proxy_index)
                self.playback_auto_proxy_index = None

    def playback_tick(self):
        """Follow the wall clock when rendering can keep up. When it can't, Nuke-style fallback:
        the playhead may only sit at an offset from playback_origin_frame equal to how many
        primary frames have actually finished rendering since playback started -- offset 0 (stay
        put) until the origin frame's own render completes, then offset 1, and so on. A render
        pipeline too slow for real time (native 4K/ACES) degrades to slower, strictly in-order
        playback instead of racing ahead on the wall clock and later jumping to wherever that
        landed -- including backward -- once a slow frame finally finishes. When rendering keeps
        up, playback_frames_rendered climbs at least as fast as elapsed_frames, so this is
        identical to plain wall-clock-following."""
        time_range = self.dispatcher.document["time"]
        span = time_range["last"] - time_range["first"] + 1
        elapsed_frames = int((time.monotonic() - self.playback_origin_time) * time_range["fps"])
        advanced = elapsed_frames - self.playback_elapsed_frames
        if advanced > 1:
            self.playback_dropped_frames += advanced - 1
        self.playback_elapsed_frames = elapsed_frames
        bounded_frames = min(elapsed_frames, self.playback_frames_rendered)
        target = time_range["first"] + (
            (self.playback_origin_frame - time_range["first"] + bounded_frames) % span)
        if target != time_range["current"]:
            self.set_time(transient=True, playhead_only=True, current=target)

    def _menus(self):
        file = self.menuBar().addMenu("File")
        edit = self.menuBar().addMenu("Edit")
        # Transport keys match the compositing convention (Nuke/Resolve): bare arrows step, Home/End
        # jump to the range ends. Qt's ShortcutOverride lets a focused spin box or line edit keep
        # them for text navigation, so typing a frame number still behaves normally.
        time_menu = self.menuBar().addMenu("Time")
        for menu, name, shortcut, callback in [
            (file, "Read image…", "Ctrl+I", self.read_file),
            (file, "Open project…", "Ctrl+O", self.open_project),
            (file, "Save", "Ctrl+S", self.save_project),
            (file, "Save as…", "Ctrl+Shift+S", lambda: self.save_project(True)),
            (file, "Export image…", "Ctrl+E", self.export),
            (edit, "Undo", "Ctrl+Z", lambda: self.command({"op": "undo"})),
            (edit, "Redo", "Ctrl+Shift+Z", lambda: self.command({"op": "redo"})),
            (edit, "Settings…", "S", self.project_settings),
            (time_menu, "Previous frame", "Left", lambda: self.step_frame(-1)),
            (time_menu, "Next frame", "Right", lambda: self.step_frame(1)),
            (time_menu, "First frame", "Home",
             lambda: self.set_time(current=self.dispatcher.document["time"]["first"])),
            (time_menu, "Last frame", "End",
             lambda: self.set_time(current=self.dispatcher.document["time"]["last"])),
            (time_menu, "Play / Stop", "Space", self.toggle_playback)]:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda checked=False, fn=callback: fn())
            menu.addAction(action)

    def command(self, cmd, render=True):
        try:
            before = self.dispatcher.revision
            result = self.dispatcher.execute(cmd)
            self.last_command_error = None
            self.command_error_label.clear()
            if cmd.get("op") in ("time", "settings"):
                self.sync_timeline()
                self.sync_project_settings()
                self.update_title()
                if render:
                    self.request_preview()
            else:
                self.after_command(render, sync_settings=cmd.get("op") in ("load", "undo", "redo"),
                                   revision=before)
            return result
        except (ValueError, KeyError, TypeError, OSError) as error:
            self._show_command_error(error)
            return None

    def _show_command_error(self, error):
        """Expose a command validation failure beyond the transient status-bar message."""
        self.last_command_error = str(error)
        self.command_error_label.setText(self.last_command_error)
        self.statusBar().showMessage(self.last_command_error, 10000)

    def agent_command(self, cmd):
        # Return machine-readable errors through the bridge, not only the status bar.
        if cmd.get("op") == "errors":
            # Live render-error visibility for an attached agent: the Dispatcher only knows about
            # document edits, never about the async render pipeline, so this reaches into Window
            # state directly rather than through self.dispatcher.execute like every other op.
            since = cmd.get("since", 0)
            return {"errors": [e for e in self.render_errors if e["timestamp"] > since],
                    "current_frame": self.dispatcher.document["time"]["current"],
                    "displayed_frame": self.frame_generation, "generation": self.generation,
                    "playing": self.playing, "status": self.viewer_info.text()}
        if cmd.get("op") == "reference_context":
            return self.reference_context(cmd)
        if cmd.get("op") == "load" and self.dispatcher.document != self.saved_document:
            raise ValueError("Save current changes before agent load; human edits are unsaved")
        result = self.dispatcher.execute(cmd)
        if cmd.get("op") in ("save", "load"):
            self.project_path = str(Path(cmd["path"]).resolve())
            self.saved_document = copy.deepcopy(self.dispatcher.document)
        if cmd.get("op") not in ("describe", "inspect"):
            if cmd.get("op") == "time":
                self.sync_timeline()
                self.update_title()
                self.request_preview()
            else:
                self.after_command(sync_settings=cmd.get("op") in ("load", "undo", "redo"))
        return result

    def reference_context(self, cmd):
        """Capture the bounded display context an attached agent needs to inspect a graph.

        This is deliberately GUI-local: it reads the live viewer controls and uses the same
        evaluator/display conversion as the preview, while leaving the Dispatcher document and
        revision untouched. The resulting PNGs are display previews, never graph artifacts.
        """
        prompt = cmd.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32768:
            raise ValueError("reference_context requires a non-empty prompt of at most 32768 characters")
        directory = cmd.get("directory")
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("reference_context requires an output directory")
        output = Path(directory).expanduser().resolve()
        if not output.exists() or not output.is_dir():
            raise ValueError(f"reference_context output directory does not exist: {output}")
        include_view = cmd.get("include_view", True)
        if type(include_view) is not bool:
            raise ValueError("reference_context include_view must be boolean")
        document = copy.deepcopy(self.dispatcher.document)
        view_id = document.get("view")
        reference_ids = list(document.get("references", []))
        ordered = []
        roles = {}
        if include_view and view_id is not None:
            ordered.append(view_id)
            roles[view_id] = ["view"]
        for node_id in reference_ids:
            if node_id not in roles:
                ordered.append(node_id)
                roles[node_id] = []
            roles[node_id].append("reference")
        if not ordered:
            raise ValueError("reference_context has no viewed or referenced node to capture")
        if len(ordered) > 8:
            raise ValueError("reference_context is limited to 8 distinct captures")
        frame = int(document["time"]["current"])
        view = self.display_view.currentText()
        exposure = self.exposure.value()
        channel = self.channels.currentText()
        background = document["settings"]["viewer"]["background"]
        captures = []
        for index, node_id in enumerate(ordered, 1):
            if node_id not in document["nodes"]:
                raise ValueError(f"reference_context node {node_id!r} does not exist")
            pixels = self.evaluator.evaluate(document, node_id, frame=frame, tier=1)
            image = to_qimage(pixels, exposure, channel, background=background, view=view)
            captures.append((index, node_id, image))
        # A caller-owned directory may already contain earlier captures. Put every response in a
        # fresh unpredictable child directory so context collection never overwrites an unrelated
        # file or a prior response from the same document revision.
        capture_root = Path(tempfile.mkdtemp(
            prefix=f"nodebased-context-r{self.dispatcher.revision}-f{frame}-", dir=output))
        artifacts = []
        for index, node_id, image in captures:
            # Node IDs are document data and may contain path separators. Keep them out of the
            # filename entirely so an agent can never turn a context capture into path traversal.
            path = capture_root / f"capture-{index:02d}.png"
            if not image.save(str(path), "PNG"):
                raise ValueError(f"Cannot write reference_context PNG: {path}")
            artifacts.append({"id": node_id, "name": document["nodes"][node_id]["name"],
                              "roles": roles[node_id], "path": str(path.resolve()),
                              "width": image.width(), "height": image.height()})
        return {"revision": self.dispatcher.revision, "current_frame": frame,
                "prompt": prompt, "artifacts": artifacts}

    def after_command(self, render=True, sync_settings=False, revision=None):
        key = self.graph.selected_id()
        # An edit that changed nothing must not rebuild the graph and the properties panel.
        # `editingFinished` fires on focus-out, so right-clicking a knob to open its context menu
        # re-submits the value the document already holds -- and the unconditional rebuild that
        # followed deleted the widget the menu was parented to, closing the menu a frame after it
        # appeared. That is the "menu flashes and vanishes" bug; the Dispatcher already treats
        # such an edit as a no-op and leaves its revision alone, so that is the signal to trust.
        changed = revision is None or self.dispatcher.revision != revision
        if changed:
            self.graph.rebuild()
            self.inspect(key)
        # Undo, project load and agent "time" edits all land here, so the strip follows the
        # document rather than only the widget that happened to be dragged.
        self.sync_timeline()
        if sync_settings:
            self.sync_project_settings()
        self.update_title()
        if changed and self.show_thumbnails:
            self.thumbnail_timer.start()
        # Only an edit that can change the viewed picture re-renders. Moving, renaming or
        # labelling a node -- or editing a branch the viewer does not see -- leaves it alone.
        if render and self.render_identity() != self.rendered_identity:
            self.request_preview()

    def render_identity(self):
        document = self.dispatcher.document
        target = document.get("view")
        return (thumbnail_key(document, target, document["time"]["current"],
                              self.display_view.currentText()) if target else None,
                json.dumps(document["settings"], sort_keys=True))

    def sync_project_settings(self):
        view = self.dispatcher.document["settings"]["color"]["view"]
        if self.display_view.currentText() != view:
            self.display_view.blockSignals(True)
            self.display_view.setCurrentText(view)
            self.display_view.blockSignals(False)

    def project_settings(self):
        dialog = ProjectSettingsDialog(copy.deepcopy(self.dispatcher.document["settings"]), self,
                                       theme=self.theme_name, thumbnails=self.show_thumbnails,
                                       accent=self.accent_color)
        # Preview the theme live while the dialog is open: picking a colour scheme you cannot see
        # until you commit is a guess, not a choice. Cancel restores the one in force.
        original, original_accent = self.theme_name, self.accent_color
        preview = lambda *_: self.apply_theme_name(dialog.chosen_theme(), dialog.chosen_accent())
        dialog.theme.currentTextChanged.connect(preview)
        dialog.accent.currentIndexChanged.connect(preview)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_theme_name(dialog.chosen_theme(), dialog.chosen_accent())
            self.preferences.set_theme(self.theme_name)
            self.preferences.set_accent(self.accent_color)
            self.set_show_thumbnails(dialog.thumbnails.isChecked())
            self.command({"op": "settings", "settings": dialog.changes()})
        else:
            self.apply_theme_name(original, original_accent)

    def apply_theme_name(self, name, accent=None):
        """Restyle the application and repaint the graph background for one theme."""
        self.accent_color = valid_accent(accent)
        self.theme_name = apply_theme(name, self.accent_color)
        self.graph.grid_pen = QPen(QColor(grid_color(self.theme_name)), 1)
        self.graph.viewport().update()

    def update_title(self):
        dirty = self.dispatcher.document != self.saved_document
        self.setWindowTitle(f"NodeBased {__version__} · {Path(self.project_path).name if self.project_path else 'Untitled'}{' *' if dirty else ''}")

    def node_tab(self, key, node):
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(16, 16, 16, 16)
        label = LabelEdit(node_label(node))
        label.setObjectName("node-label")
        label.setPlaceholderText("Shown on the node under its name")
        label.setFixedHeight(72)

        def commit_label(widget=label, k=key):
            text = widget.toPlainText().strip()
            if text != node_label(self.dispatcher.document["nodes"][k]):
                self.defer_command({"op": "label", "id": k, "value": text})
        label.finished.connect(commit_label)
        form.addRow("Label", label)
        enabled = QCheckBox("Enabled")
        enabled.setObjectName("node-enabled")
        enabled.setChecked(not node["disabled"])
        if not SPECS[node["type"]]["inputs"]:
            enabled.setEnabled(False)
            enabled.setToolTip("Source nodes have nothing to pass through, so they cannot be disabled")
        else:
            enabled.setToolTip("Disabled nodes pass their main input through unchanged  [D]")
        enabled.toggled.connect(lambda value, k=key: self.defer_command(
            {"op": "disable", "id": k, "value": not value}))
        form.addRow(enabled)
        stamp = QCheckBox("Postage stamp (thumbnail)")
        stamp.setObjectName("node-thumbnail")
        stamp.setChecked(node_thumbnail(node))
        if node["type"] in NO_THUMBNAIL_TYPES:
            stamp.setEnabled(False)
        elif not self.show_thumbnails:
            stamp.setToolTip("Thumbnails are switched off in Settings → Interface")
        else:
            default = "on" if node["type"] in DEFAULT_THUMBNAIL_TYPES else "off"
            stamp.setToolTip(f"Default for {node['type']} is {default}")
        stamp.toggled.connect(lambda value, k=key: self.defer_command(
            {"op": "thumbnail", "id": k, "value": value}))
        form.addRow(stamp)
        return page

    def inspect(self, key):
        panel = QWidget()
        form = QFormLayout(panel)
        form.setContentsMargins(16, 16, 16, 16)
        if key not in self.dispatcher.document["nodes"]:
            label = QLabel("Select a node to edit its controls.\n\nLinear ACEScg · float RGBA\nPremultiplied alpha\nEXR / PNG / JPEG / TIFF input\n\n3D and AI generation are roadmap\nmilestones, not active tools yet.")
            label.setObjectName("muted")
            form.addRow(label)
        else:
            node = self.dispatcher.document["nodes"][key]
            curves = (self.dispatcher.document.get("animation") or {}).get("curves", {}).get(key, {})
            expressions = (self.dispatcher.document.get("expressions") or {}).get(key, {})
            # Curves and expressions are resolved through the same document boundary used by the
            # evaluator. This keeps the inspector honest: an expression-driven knob shows the
            # number the artist will actually render at the current frame, including a formula
            # that reads an animated parameter on another node.
            resolved_document = resolve_document(self.dispatcher.document,
                                                 self.dispatcher.document["time"]["current"])
            resolved = resolved_document["nodes"][key]["params"]
            heading = QLabel(node["type"].upper())
            heading.setStyleSheet(f"color: {COLORS[node['type']]}; font-weight: 700; font-size: 15px")
            form.addRow(heading)
            name = QLineEdit(node["name"])
            # editingFinished also fires on focus-out, including focus lost to a context menu, so
            # every text knob compares against the document before submitting anything. Without
            # this a right-click posts an edit that changes nothing but still costs an undo slot.
            name.editingFinished.connect(
                lambda k=key, w=name: w.text() != self.dispatcher.document["nodes"][k]["name"]
                and self.defer_command({"op": "rename", "id": k, "name": w.text()}))
            self.attach_text_menu(name)
            form.addRow("Name", name)
            if artifact_type(node["type"]) in ("image", "matte"):
                reference = QCheckBox("Reference for agent")
                reference.blockSignals(True)
                reference.setChecked(key in self.dispatcher.document["references"])
                reference.blockSignals(False)
                reference.toggled.connect(lambda value, k=key: self.defer_command(
                    {"op": "reference", "id": k, "value": value}))
                form.addRow(reference)
            def add_legacy_param(param, value, kind=None):
                """Render one member of an unimplemented multi-param knob unchanged."""
                if param in CHOICES:
                    control = QComboBox()
                    control.addItems(CHOICES[param])
                    control.setCurrentText(value)
                    control.currentTextChanged.connect(lambda v, k=key, p=param: self.defer_command({"op": "set", "id": k, "param": p, "value": v}))
                    form.addRow({"colorspace": "Input space", "alpha_mode": "Alpha", "red_from": "Red", "green_from": "Green",
                                 "blue_from": "Blue", "alpha_from": "Alpha"}.get(param, param), control)
                elif isinstance(value, str):
                    control = QLineEdit(value)
                    control.editingFinished.connect(
                        lambda k=key, p=param, w=control:
                        w.text() != self.dispatcher.document["nodes"][k]["params"][p]
                        and self.defer_command({"op": "set", "id": k, "param": p, "value": w.text()}))
                    self.attach_text_menu(control, default=SPECS[node["type"]]["params"][param],
                                          commit=lambda text, k=key, p=param: self.defer_command(
                                              {"op": "set", "id": k, "param": p, "value": text}))
                    form.addRow(param.title(), control)
                    if kind == "file_read" or (kind is None and param == "path" and node["type"] == "Read"):
                        browse = QPushButton("Browse image sequence…")
                        browse.setToolTip("Sequence-aware browser: numbered frames arrive as one entry")
                        browse.clicked.connect(lambda checked=False, k=key: self.browse_read(k))
                        form.addRow(browse)
                    elif kind == "file_write" or (kind is None and param == "path" and node["type"] == "Write"):
                        browse = QPushButton("Choose output…")
                        browse.setToolTip("Use a padded pattern (render.%04d.exr) to write a sequence")
                        browse.clicked.connect(lambda checked=False, k=key: self.browse_write(k))
                        form.addRow(browse)
                    elif param == "layer":
                        control.setPlaceholderText("RGBA, or e.g. beauty / diffuse / Z")
                        control.setToolTip("Leave empty for root RGB; choose a named EXR layer or scalar channel")
                else:
                    control = QSpinBox() if type(value) is int else QDoubleSpinBox()
                    control.setRange(*LIMITS[param])
                    if isinstance(control, QDoubleSpinBox):
                        control.setDecimals(3)
                        control.setSingleStep(0.1)
                    curve = curves.get(param)
                    control.setValue(resolved[param] if (curve or param in expressions) else value)
                    if param in expressions:
                        control.setEnabled(False)
                        control.setToolTip("Driven by an expression. Edit the formula below.")
                    control.setKeyboardTracking(False)
                    control.editingFinished.connect(
                        lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                    control.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                    control.customContextMenuRequested.connect(
                        lambda point, k=key, p=param, w=control: self.curve_menu(k, p, w, w, point))
                    form.addRow(param.title(), self.animatable_row(
                        key, param, control, expression=expressions.get(param)))
                    form.addRow("Expression", self.expression_row(key, param,
                                                                    expressions.get(param)))

            def numeric_field(param):
                control = QDoubleSpinBox()
                control.setObjectName(f"{param}-field")
                control.setRange(*LIMITS[param])
                control.setDecimals(3)
                control.setSingleStep(0.1)
                curve = curves.get(param)
                control.setValue(resolved[param] if (curve or param in expressions)
                                 else node["params"][param])
                if param in expressions:
                    control.setEnabled(False)
                    control.setToolTip("Driven by an expression. Edit the formula below.")
                control.setKeyboardTracking(False)
                control.editingFinished.connect(
                    lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                control.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                control.customContextMenuRequested.connect(
                    lambda point, k=key, p=param, w=control: self.curve_menu(k, p, w, w, point))
                return control

            def add_animation_button(row, param, control):
                button = QPushButton()
                button.setFixedWidth(26)
                button.setFlat(True)
                button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                curve = self.node_curve(key, param)
                frame = self.dispatcher.document["time"]["current"]
                keyed_here = curve is not None and any(k["frame"] == frame for k in curve["keys"])
                expression = expressions.get(param)
                if expression is not None:
                    button.setText("ƒ")
                    button.setEnabled(False)
                    button.setStyleSheet("color: #c58cff; border: none; font-size: 14px")
                    button.setToolTip("Expression-driven. Clear the expression before keying this knob.")
                elif keyed_here:
                    button.setText("◆")
                    button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
                    button.setToolTip(f"Key set at frame {frame}. Click to remove it.\nRight-click for "
                                      f"curve options.")
                elif curve is not None:
                    button.setText("◇")
                    button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
                    button.setToolTip(f"Animated ({len(curve['keys'])} keys, {curve['interpolation']}). "
                                      f"Click to key the current value at frame {frame}.\nRight-click for "
                                      f"curve options.")
                else:
                    button.setText("○")
                    button.setStyleSheet("color: #6d6d78; border: none; font-size: 14px")
                    button.setToolTip(f"Not animated. Click to set the first key at frame {frame}.")
                button.clicked.connect(
                    lambda checked=False, k=key, p=param, w=control, on=keyed_here:
                    self.toggle_key(k, p, w, on))
                button.customContextMenuRequested.connect(
                    lambda point, k=key, p=param, w=control, b=button: self.curve_menu(k, p, w, b, point))
                row.addWidget(button)

            for group in knob_layout(node["type"]):
                if group.kind == "xy":
                    fields = QWidget()
                    layout = QHBoxLayout(fields)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)
                    x_field, y_field = (numeric_field(param) for param in group.params)
                    layout.addWidget(QLabel("x"))
                    layout.addWidget(x_field, 1)
                    layout.addWidget(QLabel("y"))
                    layout.addWidget(y_field, 1)
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(4)
                    row_layout.addWidget(fields, 1)
                    add_animation_button(row_layout, group.params[0], x_field)
                    form.addRow(group.label, row)
                    continue
                if group.kind == "color":
                    fields = QWidget()
                    layout = QHBoxLayout(fields)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.setSpacing(4)
                    color_fields = [numeric_field(param) for param in group.params]
                    swatch = ClickableColorSwatch()
                    swatch.setObjectName("color-swatch")
                    swatch.setFixedSize(24, 24)

                    def set_swatch():
                        rgb = [max(0.0, min(1.0, field.value())) for field in color_fields[:3]]
                        swatch.setStyleSheet("background-color: rgb(%d, %d, %d); border: 1px solid #777;" %
                                             tuple(int(value * 255) for value in rgb))

                    def pick_color():
                        current = [max(0.0, min(1.0, field.value())) for field in color_fields[:3]]
                        color = QColorDialog.getColor(
                            QColor(*(int(value * 255) for value in current)), self, "Choose color")
                        if color.isValid():
                            for param, value in zip(group.params[:3],
                                                    (color.redF(), color.greenF(), color.blueF())):
                                self.defer_command({"op": "set", "id": key, "param": param,
                                                    "value": value})
                            for field, value in zip(color_fields[:3],
                                                    (color.redF(), color.greenF(), color.blueF())):
                                field.setValue(value)
                            set_swatch()

                    swatch.clicked.connect(pick_color)
                    set_swatch()
                    layout.addWidget(swatch)
                    for field in color_fields:
                        layout.addWidget(field, 1)
                    form.addRow(group.label, fields)
                    continue
                param = group.params[0]
                value = node["params"][param]
                if group.kind == "bool":
                    control = QCheckBox()
                    control.setChecked(bool(value))
                    control.toggled.connect(lambda checked, k=key, p=param: self.defer_command(
                        {"op": "set", "id": k, "param": p, "value": 1 if checked else 0}))
                    form.addRow(group.label, control)
                elif group.kind in ("enum",):
                    add_legacy_param(param, value, group.kind)
                elif group.kind in ("string", "file_read", "file_write"):
                    add_legacy_param(param, value, group.kind)
                elif group.kind == "float_slider":
                    curve = curves.get(param)
                    shown = resolved[param] if (curve or param in expressions) else value
                    control = FloatSliderControl(LIMITS[param], group.soft_range, shown)
                    if param in expressions:
                        control.setEnabled(False)
                        control.setToolTip("Driven by an expression. Edit the formula below.")
                    control.spin.editingFinished.connect(
                        lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                    control.slider.sliderReleased.connect(
                        lambda k=key, p=param, w=control: self.commit_param(k, p, w.value()))
                    control.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                    control.customContextMenuRequested.connect(
                        lambda point, k=key, p=param, w=control: self.curve_menu(k, p, w, w, point))
                    form.addRow(group.label, self.animatable_row(
                        key, param, control, expression=expressions.get(param)))
                    form.addRow("Expression", self.expression_row(key, param,
                                                                    expressions.get(param)))
                else:
                    add_legacy_param(param, value)
            if node["type"] == "Merge":
                form.addRow(QLabel("A over B · scene-linear, premultiplied\nInputs must have matching dimensions.\n"
                                   "Optional mask gates the merge: where mask.a is 0 the result is B."))
            if node["type"] == "Transform":
                form.addRow(QLabel("Integer translation · fixed image bounds"))
            if node["type"] == "Crop":
                form.addRow(QLabel("Masks to a rectangle · fixed image bounds\n(canvas is not resized)"))
            if node["type"] == "Blur":
                form.addRow(QLabel("Separable box blur · radius in pixels"))
            if node["type"] == "ColorCorrect":
                form.addRow(QLabel("Lift / gamma / gain / saturation\napplied to unpremultiplied color"))
            if node["type"] == "Shuffle":
                form.addRow(QLabel("Remaps output channels from any input\nchannel, or constant 0 / 1"))
            if node["type"] == "Premult":
                form.addRow(QLabel("Multiplies RGB by alpha\n(premultiplies a straight-alpha input)"))
            if node["type"] == "Unpremult":
                form.addRow(QLabel("Divides RGB by alpha\n(alpha == 0 leaves RGB untouched, no NaN/inf)"))
            if node["type"] == "Dot":
                form.addRow(QLabel("Graph reroute / passthrough · pixel data unchanged"))
            if node["type"] == "Switch":
                form.addRow(QLabel("Selects one of its inputs via 'which' (0 or 1).\nNo resampling — pixel format must match."))
            if node["type"] == "ChannelShuffle":
                form.addRow(QLabel("Routes each output channel from A, B or a constant.\n"
                                   "Naming a B channel with B unwired is an error, not black."))
            if node["type"] == "Roto":
                draw = QPushButton("Draw shape…")
                draw.setToolTip("Click points in the Roto viewer; press Enter to close, Esc to cancel")
                draw.clicked.connect(lambda checked=False, k=key: self.begin_roto_draw(k))
                form.addRow(draw)
                form.addRow(QLabel("Animatable bezier/polygon shapes → premultiplied matte.\n"
                                   "View this node to drag existing points. Drawing and point edits\n"
                                   "commit through one validated set_shapes command."))
            if node["type"] == "Tracker":
                pick = QPushButton("Add track point at reference…")
                pick.setToolTip("View this Tracker, then click the reference point in the viewer")
                pick.clicked.connect(lambda checked=False, k=key: self.begin_tracker_pick(k))
                form.addRow(pick)
                analyse_button = QPushButton("Analyze forward")
                analyse_button.setToolTip("Analyze from the reference frame through the timeline end")
                analyse_button.clicked.connect(lambda checked=False, k=key: self.analyse_tracker(k))
                form.addRow(analyse_button)
                cancel_button = QPushButton("Cancel analysis")
                cancel_button.setEnabled(self._tracker_future is not None)
                cancel_button.clicked.connect(self.cancel_tracker_analysis)
                form.addRow(cancel_button)
                form.addRow(QLabel("Pixel NCC tracking uses float scene-linear pixels.\n"
                                   "One validated set_tracks command is committed after completion;\n"
                                   "failure or cancel leaves the document unchanged."))
            if node["type"] == "Viewer":
                form.addRow(QLabel("Shows whatever is being viewed · its input follows the\n"
                                   "view target and is drawn as a faint tap, never as a\n"
                                   "processing connection. Pixels pass through unchanged."))
            if node["type"] == "Write":
                render_frame = QPushButton("Render current frame")
                render_frame.setToolTip("Full-resolution reference render of the frame at the playhead")
                render_frame.clicked.connect(lambda checked=False, k=key: self.render_write(k, single=True))
                form.addRow(render_frame)
                render_range = QPushButton("Render frame range")
                render_range.setToolTip("Renders the project frame range; needs a padded path "
                                        "pattern such as render.%04d.exr")
                render_range.clicked.connect(lambda checked=False, k=key: self.render_write(k, single=False))
                form.addRow(render_range)
                form.addRow(QLabel("Where image output lives. Always full resolution through the\n"
                                   "reference evaluator — viewer proxy, exposure and channel\n"
                                   "controls are display-only and never reach a written file.\n"
                                   "Pixels pass through unchanged, so a Write mid-branch is inert."))
            if node["type"] in MASK_MIX_KINDS:
                form.addRow(QLabel("Optional mask input + 'mix' blend with original\n"
                                    "result = mix * mask.a * filtered + (1 - mix * mask.a) * source"))
            view = QPushButton("View this node   [1]")
            view.clicked.connect(lambda: self.command({"op": "view", "id": key}))
            form.addRow(view)
            # Nuke keeps presentation and bypass on a second "Node" tab, away from the knobs
            # that change pixels.
            tabs = QTabWidget()
            tabs.setObjectName("node-tabs")
            tabs.addTab(top_aligned(panel), node["type"])
            tabs.addTab(top_aligned(self.node_tab(key, node)), "Node")
            tabs.setCurrentIndex(min(self.properties_tab, 1))
            tabs.currentChanged.connect(lambda index: setattr(self, "properties_tab", index))
            panel = tabs
        old = self.properties.takeWidget()
        if old:
            old.deleteLater()
        self.properties.setWidget(panel)

    def attach_text_menu(self, editor, default=None, commit=None):
        """Give a text knob a context menu that outlives a panel rebuild.

        QLineEdit's own menu is parented to the line edit, so it is destroyed the moment the
        inspector is rebuilt -- which is what made a right-click look like a menu that flashed and
        vanished. This builds the same standard actions under a window-owned menu, and adds the
        "Set to default" entry a compositing knob is expected to have.
        """
        editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def show(point):
            menu = QMenu(self)
            standard = editor.createStandardContextMenu()
            for action in standard.actions():
                menu.addAction(action)
            if commit is not None:
                menu.addSeparator()
                label = f"Set to default ({default!r})" if default else "Clear"
                menu.addAction(label, lambda: commit(default or ""))
            menu.exec(editor.mapToGlobal(point))
            standard.deleteLater()

        editor.customContextMenuRequested.connect(show)

    def defer_command(self, cmd):
        # Do not destroy an editor while it is emitting editingFinished.
        QTimer.singleShot(0, lambda: self.command(cmd))

    def begin_roto_draw(self, key):
        """Enter viewer drawing mode for a Roto node after making it the viewed format."""
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] != "Roto":
            self._show_command_error(ValueError("Roto drawing requires a Roto node"))
            return False
        if self.dispatcher.document.get("view") != key:
            self.command({"op": "view", "id": key})
        return self.viewer.begin_roto_draw(key)

    def _tracker_context(self, key=None):
        selected = key or self.graph.selected_id()
        node = self.dispatcher.document["nodes"].get(selected) if selected else None
        if node is None or node["type"] != "Tracker" or self.dispatcher.document.get("view") != selected:
            return None
        return selected, node, self.dispatcher.document.get("node_data", {}).get(selected, {"tracks": []}), 1

    def begin_tracker_pick(self, key):
        if self._tracker_future is not None:
            self._show_command_error(ValueError("Cancel the running Tracker analysis before picking another point"))
            return False
        node = self.dispatcher.document["nodes"].get(key)
        if node is None or node["type"] != "Tracker":
            self._show_command_error(ValueError("Tracker point picking requires a Tracker node"))
            return False
        if self.dispatcher.document.get("view") != key:
            self.command({"op": "view", "id": key})
        if self._tracker_context(key) is None:
            self._show_command_error(ValueError("View the Tracker's image input before picking a point"))
            return False
        self.viewer.tracker_picking = True
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("Tracker: click a point in the reference frame · Esc cancels")
        return True

    def add_tracker_point(self, point):
        if self._tracker_future is not None:
            return False
        key = self.graph.selected_id()
        context = self._tracker_context(key)
        if context is None:
            return False
        payload = copy.deepcopy(context[2])
        names = {track.get("name") for track in payload.get("tracks", [])}
        index = 1
        while f"track{index}" in names:
            index += 1
        self._tracker_seed = {"name": f"track{index}", "enabled": 1,
                              "x": float(point[0]), "y": float(point[1])}
        self._tracker_key = key
        self._tracker_index = len(payload.get("tracks", []))
        self.statusBar().showMessage(f"Picked {self._tracker_seed['name']} at reference point; ready to analyze")
        return True

    def cancel_tracker_analysis(self):
        if self._tracker_cancel is not None:
            self._tracker_cancel.set()
            self.statusBar().showMessage("Tracker analysis: cancelling…")

    def analyse_tracker(self, key):
        if self._tracker_future is not None:
            return False
        context = self._tracker_context(key)
        index = self._tracker_index
        if context is None or index is None or self._tracker_seed is None or self._tracker_key != key:
            self._show_command_error(ValueError("Pick a track point before analyzing"))
            return False
        source = context[1]["inputs"].get("image")
        if source is None:
            self._show_command_error(ValueError("Tracker analysis requires a connected image input"))
            return False
        snapshot = copy.deepcopy(self.dispatcher.document)
        frame = int(snapshot["time"]["current"])
        last = int(snapshot["time"]["last"])
        point = (float(self._tracker_seed["x"]), float(self._tracker_seed["y"]))
        cancel = threading.Event()
        job = {"key": key, "index": index, "seed": copy.deepcopy(self._tracker_seed),
               "base_tracks": copy.deepcopy(context[2].get("tracks", [])),
               "reference_frame": frame}
        self._tracker_job = job
        self._tracker_cancel = cancel
        self._tracker_future = self.executor.submit(
            lambda: tracker_model.analyse(
                lambda f: self.evaluator.evaluate_raster(snapshot, target=source, frame=f),
                frame, point, first_frame=frame, last_frame=last, cancel=cancel))
        self.statusBar().showMessage(f"Tracker analysis: 0/{max(0, last-frame)} frames")
        self.inspect(key)
        QTimer.singleShot(50, self._poll_tracker_analysis)
        return True

    def _poll_tracker_analysis(self):
        future = self._tracker_future
        if future is None:
            return
        if not future.done():
            self.statusBar().showMessage("Tracker analysis: running…")
            QTimer.singleShot(50, self._poll_tracker_analysis)
            return
        self._tracker_future = None
        cancel = self._tracker_cancel
        self._tracker_cancel = None
        job = self._tracker_job
        self._tracker_job = None
        try:
            if job is None:
                raise tracker_model.AnalysisError("Tracker analysis lost its pending job state")
            results = future.result()
            if cancel is not None and cancel.is_set():
                raise CancelledError()
            key = job["key"]
            node = self.dispatcher.document["nodes"].get(key)
            if node is None or node["type"] != "Tracker":
                raise tracker_model.AnalysisError("Tracker was deleted or replaced during analysis; results discarded")
            current_tracks = self.dispatcher.document.get("node_data", {}).get(key, {}).get("tracks", [])
            if current_tracks != job["base_tracks"]:
                raise tracker_model.AnalysisError("Tracker data changed during analysis; results discarded")
            payload = {"tracks": copy.deepcopy(job["base_tracks"])}
            index = job["index"]
            payload["tracks"].append(copy.deepcopy(job["seed"]))
            track = payload["tracks"][index]
            for field in ("x", "y"):
                keys = [{"frame": int(f), "value": float(position[0 if field == "x" else 1])}
                        for f, position in sorted(results.items())]
                track[field] = {"value": float(track[field]) if not isinstance(track[field], dict) else float(track[field]["value"]),
                                "curve": {"interpolation": "constant", "keys": keys}}
            self.command({"op": "set_tracks", "id": key, "tracks": payload["tracks"]})
            self._tracker_seed = None
            self._tracker_index = None
            self._tracker_key = None
            self.statusBar().showMessage(f"Tracker analysis complete: {len(results)} frames")
        except CancelledError:
            self.statusBar().showMessage("Tracker analysis cancelled; document unchanged")
        except (ValueError, KeyError, OSError) as error:
            self._show_command_error(error)
            self.statusBar().showMessage(f"Tracker analysis failed: {error}", 10000)
        finally:
            self.inspect(self.graph.selected_id())

    # -- Animation on numeric knobs -------------------------------------------------------
    #
    # The curve engine (nodebased/animation.py) and its three atomic undoable ops -- set_key,
    # delete_key, clear_curve -- already existed and were exercised only through the agent
    # bridge. Everything below is the missing half: reaching them from the properties panel.
    # No new document shape, no second code path; the button issues the same command an agent
    # would, through the same dispatcher, so it undoes and saves identically.

    def node_curve(self, key, param):
        """The curve for ``(node, param)``, or None when that parameter is not animated."""
        return (self.dispatcher.document.get("animation") or {}).get("curves", {}).get(key, {}).get(param)

    def commit_param(self, key, param, value):
        """Route a knob edit to the base parameter, or to a key when the knob is animated.

        Editing an animated knob and having it silently change the base -- immediately overridden
        by the curve, so the viewer does not move -- is the single most confusing thing a
        keyframing UI can do. When a curve exists, typing a value means "make it that value here",
        which is a key at the current frame. Nuke behaves the same way.
        """
        if self.node_curve(key, param) is None:
            self.defer_command({"op": "set", "id": key, "param": param, "value": value})
            return
        frame = self.dispatcher.document["time"]["current"]
        self.defer_command({"op": "set_key", "id": key, "param": param,
                            "frame": int(frame), "value": float(value)})

    def animatable_row(self, key, param, control, expression=None):
        """Pack a numeric control next to its keyframe button."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(control, 1)
        button = QPushButton()
        button.setFixedWidth(26)
        button.setFlat(True)
        button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        curve = self.node_curve(key, param)
        frame = self.dispatcher.document["time"]["current"]
        keyed_here = curve is not None and any(k["frame"] == frame for k in curve["keys"])
        if expression is not None:
            button.setText("ƒ")
            button.setEnabled(False)
            button.setStyleSheet("color: #c58cff; border: none; font-size: 14px")
            button.setToolTip("Expression-driven. Clear the expression before keying this knob.")
        elif keyed_here:
            button.setText("◆")
            button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
            button.setToolTip(f"Key set at frame {frame}. Click to remove it.\nRight-click for "
                              f"curve options.")
        elif curve is not None:
            button.setText("◇")
            button.setStyleSheet(f"color: {KEY_COLOR.name()}; border: none; font-size: 14px")
            button.setToolTip(f"Animated ({len(curve['keys'])} keys, {curve['interpolation']}). "
                              f"Click to key the current value at frame {frame}.\nRight-click for "
                              f"curve options.")
        else:
            button.setText("○")
            button.setStyleSheet("color: #6d6d78; border: none; font-size: 14px")
            button.setToolTip(f"Not animated. Click to set the first key at frame {frame}.")
        button.clicked.connect(
            lambda checked=False, k=key, p=param, w=control, on=keyed_here:
                self.toggle_key(k, p, w, on))
        button.customContextMenuRequested.connect(
            lambda point, k=key, p=param, w=control, b=button: self.curve_menu(k, p, w, b, point))
        layout.addWidget(button)
        return row

    def expression_row(self, key, param, expression=None):
        """Return the formula editor for one numeric knob.

        The editor intentionally commits only from an explicit Set button or Return. Typing into
        a field must not mutate the document on every keystroke: an incomplete formula is normal
        while editing, and the Dispatcher is the atomic validation/undo boundary. Clear uses the
        same boundary, so setting and removing a link undo exactly like an agent command.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        editor = QLineEdit(expression or "")
        editor.setObjectName("expression-editor")
        editor.setAccessibleName(f"{key}.{param} expression")
        editor.setPlaceholderText("e.g. frame * 2 + 1 or knob(\"node\", \"param\")")
        editor.setToolTip("Safe arithmetic formula. Use frame or knob(\"node id\", \"param\")")
        apply = QPushButton("Set")
        apply.setObjectName("set-expression")
        clear = QPushButton("Clear")
        clear.setObjectName("clear-expression")
        clear.setEnabled(bool(expression))
        layout.addWidget(editor, 1)
        layout.addWidget(apply)
        layout.addWidget(clear)

        def set_expression():
            text = editor.text().strip()
            if not text:
                self._show_command_error(ValueError("Enter an expression or use Clear"))
                return
            self.defer_command({"op": "set_expression", "id": key, "param": param,
                                "expression": text})

        apply.clicked.connect(set_expression)
        editor.returnPressed.connect(set_expression)
        clear.clicked.connect(lambda: self.defer_command(
            {"op": "clear_expression", "id": key, "param": param}))
        return row

    def toggle_key(self, key, param, control, keyed_here):
        frame = int(self.dispatcher.document["time"]["current"])
        if keyed_here:
            self.defer_command({"op": "delete_key", "id": key, "param": param, "frame": frame})
        else:
            self.defer_command({"op": "set_key", "id": key, "param": param, "frame": frame,
                                "value": float(control.value())})

    def curve_menu(self, key, param, control, button, point):
        """Nuke-style knob context menu, opened from the knob itself as well as its key button.

        In Nuke the animation menu belongs to the knob: right-clicking the number is how you key
        it, and the small diamond is a shortcut rather than the only door. The menu is parented to
        the *window*, not to `button` -- a menu owned by a panel widget dies with the panel if
        anything rebuilds the inspector while it is open.
        """
        frame = int(self.dispatcher.document["time"]["current"])
        curve = self.node_curve(key, param)
        menu = QMenu(self)
        menu.addAction(f"Set key at frame {frame}",
                       lambda: self.defer_command({"op": "set_key", "id": key, "param": param,
                                                   "frame": frame, "value": float(control.value())}))
        delete = menu.addAction(
            f"Delete key at frame {frame}",
            lambda: self.defer_command({"op": "delete_key", "id": key, "param": param,
                                        "frame": frame}))
        delete.setEnabled(curve is not None and any(k["frame"] == frame for k in curve["keys"]))
        clear = menu.addAction("Remove animation",
                               lambda: self.defer_command({"op": "clear_curve", "id": key,
                                                           "param": param}))
        clear.setEnabled(curve is not None)
        if curve is not None:
            menu.addSeparator()
            interpolation = menu.addMenu("Interpolation")
            for name in CURVE_INTERPOLATIONS:
                action = interpolation.addAction(
                    name,
                    # set_key re-keys the frame it is given and carries the interpolation for the
                    # whole curve, so re-setting any existing key is the atomic way to change it.
                    lambda n=name, c=curve: self.defer_command(
                        {"op": "set_key", "id": key, "param": param,
                         "frame": c["keys"][0]["frame"], "value": float(c["keys"][0]["value"]),
                         "interpolation": n}))
                action.setCheckable(True)
                action.setChecked(curve["interpolation"] == name)
        menu.addSeparator()
        node_type = self.dispatcher.document["nodes"][key]["type"]
        default = SPECS[node_type]["params"][param]
        reset = menu.addAction(f"Set to default ({default})",
                              lambda: self.defer_command({"op": "set", "id": key, "param": param,
                                                          "value": default}))
        reset.setEnabled(param not in (self.dispatcher.document.get("expressions") or {}).get(key, {}))
        menu.exec(button.mapToGlobal(point))

    def node_search(self):
        graph_pos = self.graph.last_click_scene_pos
        global_pos = self.graph.viewport().mapToGlobal(self.graph.mapFromScene(graph_pos))
        kind = NodeSearch.choose(self, SPECS, global_pos)
        if kind:
            self.add_node(kind, position=graph_pos)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and event.key() == Qt.Key.Key_Control:
            graph_point = self.graph.viewport().mapFromGlobal(QCursor.pos())
            if (self.graph.viewport().rect().contains(graph_point)
                    or watched in (self.graph, self.graph.viewport())
                    or self.graph.hasFocus()):
                self.graph.ctrl_handles_visible = event.type() == QEvent.Type.KeyPress
                self.graph.viewport().update()
        if (event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab
                and not QApplication.activePopupWidget()):
            graph_point = self.graph.viewport().mapFromGlobal(QCursor.pos())
            if self.graph.viewport().rect().contains(graph_point):
                self.graph.last_click_scene_pos = self.graph.mapToScene(graph_point)
                self.node_search()
                return True
        return super().eventFilter(watched, event)

    def node_position(self, desired, below=False, kind=None):
        """Find a nearby vacant location; never drop a new node on an existing one.

        `below` searches straight down the column first, which is where a node added to a
        selection belongs: the stream reads top to bottom, as in Nuke. The radial search is the
        fallback only once the column is exhausted.
        """
        desired = QPointF(round(desired.x()), round(desired.y()))
        candidates = [QPointF(0, 0)]
        if below:
            candidates.extend(QPointF(0, step) for step in range(40, 1201, 40))
        for radius in range(80, 801, 80):
            candidates.extend(QPointF(x, y) for x, y in
                              ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                               (radius, radius), (-radius, radius), (radius, -radius), (-radius, -radius)))
        occupied = [item.sceneBoundingRect().adjusted(-12, -12, 12, 12)
                    for item in self.graph.items_by_id.values()]
        for offset in candidates:
            pos = desired + offset
            rect = QRectF(pos.x(), pos.y(), NODE_WIDTH,
                          node_height({"type": kind or ""}, self.show_thumbnails))
            if not any(rect.intersects(other) for other in occupied):
                return pos
        return desired + QPointF(0, 880)

    def add_node(self, kind=None, params=None, position=None):
        if not kind:
            kind, ok = QInputDialog.getItem(self, "Create node", "Node (type to search)", list(SPECS), 0, True)
            if not ok:
                return
        # A selected node with a real input slot takes priority over click position: the new
        # node lands in its branch, wired from its output, rather than wherever Tab/hotkey
        # last recorded a click. Generators (Read/Constant/Checker) have no input slot, so
        # selecting one falls back to plain click placement instead of a no-op connect.
        required_inputs = list(SPECS[kind]["inputs"])
        source = self.graph.selected_id() if required_inputs else None
        if source:
            # Directly underneath the selection, centred on it -- a Dot is far narrower than a
            # node, so centre on its bounds rather than aligning left edges.
            selected = self.graph.items_by_id[source].sceneBoundingRect()
            anchor = QPointF(selected.center().x() - 95, selected.bottom() + 40)
        else:
            anchor = position if position is not None else self.graph.last_click_scene_pos
        pos = self.node_position(anchor, below=bool(source), kind=kind)
        key = __import__("uuid").uuid4().hex[:12]
        commands = [{"op": "create", "id": key, "type": kind, "pos": [pos.x(), pos.y()], "params": params or {}}]
        if source:
            commands.append({"op": "connect", "id": key, "input": required_inputs[0], "source": source})
            # Splice into the branch: anything currently reading from the selected node's
            # output is rewired to read from the new node instead, so it's inserted inline
            # rather than just forking a new dead-end off the selection.
            nodes = self.dispatcher.document["nodes"]
            downstream = [(dest, slot) for dest, node in nodes.items()
                          for slot, src in node["inputs"].items() if src == source]
            commands.extend({"op": "connect", "id": dest, "input": slot, "source": key} for dest, slot in downstream)
        if self.command({"op": "batch", "commands": commands}) is not None:
            self.graph.scene().clearSelection()
            self.graph.items_by_id[key].setSelected(True)
            if kind == "Read" and not params:
                self.browse_read(key)

    def browse_read(self, key):
        chosen = SequenceBrowser.choose(self, self.last_browse_directory)
        if chosen is None:
            return
        self.last_browse_directory = str(Path(chosen["path"]).parent)
        self.command({"op": "set", "id": key, "param": "path", "value": chosen["path"]})
        self.offer_sequence_range(chosen)

    def offer_sequence_range(self, chosen):
        """Ask before re-ranging the comp to a freshly loaded sequence.

        Nuke sets the project range from the first clip you load; doing it silently on every load
        would quietly discard a range an artist set deliberately, so this asks and only when the
        range actually differs.
        """
        if not chosen.get("sequence") or chosen.get("first") is None:
            return
        time_range = self.dispatcher.document["time"]
        if (time_range["first"], time_range["last"]) == (chosen["first"], chosen["last"]):
            return
        gap = f" with {len(chosen['missing'])} frames missing" if chosen["missing"] else ""
        answer = QMessageBox.question(
            self, "Sequence range",
            f"{Path(chosen['path']).name} covers frames {chosen['first']}–{chosen['last']}{gap}.\n"
            f"Set the project range to match? "
            f"(currently {time_range['first']}–{time_range['last']})",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.set_time(first=chosen["first"], last=chosen["last"], current=chosen["first"])

    def browse_write(self, key):
        path, selected = QFileDialog.getSaveFileName(
            self, "Write output", self.dispatcher.document["nodes"][key]["params"]["path"] or "render.%04d.exr",
            "OpenEXR (*.exr);;PNG (*.png)")
        if path:
            self.command({"op": "set", "id": key, "param": "path", "value": path})

    def read_file(self):
        chosen = SequenceBrowser.choose(self, self.last_browse_directory)
        if chosen is None:
            return
        self.last_browse_directory = str(Path(chosen["path"]).parent)
        self.add_node("Read", {"path": chosen["path"]})
        self.command({"op": "view", "id": self.graph.selected_id()})
        self.offer_sequence_range(chosen)

    def confirm_discard(self):
        if self.dispatcher.document == self.saved_document:
            return True
        result = QMessageBox.question(self, "Unsaved project", "Save changes before continuing?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if result == QMessageBox.StandardButton.Save:
            return self.save_project()
        return result == QMessageBox.StandardButton.Discard

    def open_project(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "NodeBased (*.nbcomp)")
        if path and self.command({"op": "load", "path": path}) is not None:
            self.project_path = path
            self.saved_document = copy.deepcopy(self.dispatcher.document)
            self.update_title()
            self.graph.fit()

    def save_project(self, save_as=False):
        path = self.project_path
        if not path or save_as:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", path or "Untitled.nbcomp", "NodeBased (*.nbcomp)")
        if not path:
            return False
        if not Path(path).suffix:
            path += ".nbcomp"
        if self.command({"op": "save", "path": path}, render=False) is None:
            return False
        self.project_path = path
        self.saved_document = copy.deepcopy(self.dispatcher.document)
        self.update_title()
        return True

    def request_preview(self, *_, playhead_only=False):
        """Queue a preview. ``playhead_only`` marks a transport advance rather than a content
        change: the queued work is replaced but an in-flight render is left to finish, because
        a frame that outlives its own tick is still the newest frame we have."""
        self.generation += 1
        self.rendered_identity = self.render_identity()
        snapshot = copy.deepcopy(self.dispatcher.document)
        frame = snapshot["time"]["current"]
        future = self.future_frames(frame) if self.playing else ()
        viewport = None
        # A first preview, or one whose source canvas just changed size, has no scene extent
        # worth clamping against: request the complete data window so the first frame establishes
        # honest bounds for `fit()` to compute from. Without this check, reconnecting the viewer
        # to a differently sized source inherited whatever fraction of the OLD canvas happened to
        # be visible in the viewport, silently clamping the very first request for the new source
        # to a small top-left crop that `fit()` then zoomed into — the image looked stuck in the
        # corner with no way to recenter, because every later request re-derived its viewport from
        # that same wrongly-zoomed view. The canvas-size check is a header-only read for a Read
        # source (sub-millisecond once the OS file cache is warm), so it costs nothing during
        # ordinary playback, where the canvas size does not change frame to frame.
        target = snapshot.get("view")
        current_extent = self.viewer.scene().itemsBoundingRect()
        same_canvas = False
        if target and not current_extent.isEmpty():
            try:
                if self.tile_executor.supports_tiled(snapshot, target):
                    tier = self.proxy.currentData()
                    bounds = self.tile_executor.canvas_region(snapshot, target, frame=frame, tier=tier)
                    claimed = self.viewer.sceneRect()
                    same_canvas = (bounds.width * tier, bounds.height * tier) == \
                                  (claimed.width(), claimed.height())
            except Exception:
                same_canvas = False
        if same_canvas:
            rect = self.viewer.mapToScene(self.viewer.viewport().rect()).boundingRect()
            viewport = (math.floor(rect.left()), math.floor(rect.top()),
                        math.ceil(rect.right()), math.ceil(rect.bottom()))
        cancel_active = not (playhead_only and self.playing)
        if cancel_active:
            # A real seek or content edit, not a plain playback tick (docs/PLAYBACK.md
            # criterion 3): drop pending decode-ahead work instead of caching frames a scrub
            # just made irrelevant. See decodepool.py -- a job already dispatched to OIIO still
            # runs to completion, but its result is discarded rather than stored.
            self.decode_pool.bump_epoch()
        self.preview_queue.replace(self.generation, frame, snapshot, future,
                                   tier=self.proxy.currentData(), viewport=viewport,
                                   playing=self.playing,
                                   cancel_active=cancel_active)
        # The decode-ahead pool only ever gets consulted by the tier != 1 Read path (tileexec.py's
        # `_generator_tile`); a tier-1 request takes the bounded-region-read fast path instead and
        # never looks at it. Prefetching at tier 1 would just spend the decode pool's own worker
        # threads (and the GIL) decoding full frames nobody will ever read back -- measured as a
        # net loss for full-resolution playback (docs/BENCHMARKS-v0.17-playback.md).
        if target and future and self.proxy.currentData() != 1:
            self.tile_executor.prefetch_reads(snapshot, target, future)
        self.timer.start(0 if self.playing else 35)

    def start_preview(self):
        if self.busy:
            return
        queued = self.preview_queue.take()
        if queued is None:
            return
        request, cancel = queued
        self.busy = True
        # The viewer always outranks thumbnails on the single preview worker.
        self.thumbnail_cancel.set()
        exposure, channel = self.exposure.value(), self.channels.currentText()
        view = self.display_view.currentText()
        background = request.document["settings"]["viewer"]["background"]
        if request.display:
            self.statusBar().showMessage(f"Evaluating frame {request.frame}…")
        def work():
            start = time.perf_counter()
            try:
                target = request.document.get("view")
                render_region = None
                tiled = bool(target) and self.tile_executor.supports_tiled(request.document, target)
                if tiled:
                    bounds = self.tile_executor.canvas_region(request.document, target,
                                                              frame=request.frame, tier=request.tier)
                    if request.display and request.viewport is not None:
                        x0, y0, x1, y1 = request.viewport
                        wanted = TileRegion(x0, y0, max(0, x1 - x0), max(0, y1 - y0),
                                            full_x=bounds.x, full_y=bounds.y,
                                            full_width=bounds.width, full_height=bounds.height)
                        ox0, oy0 = max(wanted.x, bounds.x), max(wanted.y, bounds.y)
                        ox1, oy1 = min(wanted.right, bounds.right), min(wanted.bottom, bounds.bottom)
                        render_region = TileRegion(ox0, oy0, max(0, ox1 - ox0), max(0, oy1 - oy0),
                                                   full_x=bounds.x, full_y=bounds.y,
                                                   full_width=bounds.width, full_height=bounds.height)
                    else:
                        render_region = bounds
                region_key = (None if render_region is None else
                              (render_region.x, render_region.y,
                               render_region.width, render_region.height))
                display_key = DisplayCache.key(request.document, target, request.frame,
                                               request.tier, view, exposure, channel, background)
                # The display cache is consulted before anything is composed. It holds the finished
                # picture, so a hit needs nothing else: checking it only after composing -- as this
                # used to -- meant a cached 4K frame still re-assembled every tile (~600 ms) just to
                # throw the result away, which is what made scrubbing back over frames playback had
                # already shown feel uncached at full resolution.
                cached = self.display_cache.get(display_key, region_key)
                crop = None
                if cached is None and tiled and render_region != bounds:
                    # Read-ahead caches whole frames; a zoomed-in view asks for its visible crop.
                    # A whole frame contains every crop of itself, so serve the crop from it.
                    cached = self.display_cache.get(
                        display_key, (bounds.x, bounds.y, bounds.width, bounds.height))
                    crop = QRect(render_region.x - bounds.x, render_region.y - bounds.y,
                                 render_region.width, render_region.height)
                if cached is not None:
                    if cancel.is_set():
                        raise Cancelled()
                    data, width, height, bytes_per_line = cached
                    image = None
                    if request.display:
                        image = QImage(data, width, height, bytes_per_line,
                                       QImage.Format.Format_RGB888)
                        # copy() detaches from the cache's bytes either way.
                        image = image.copy(crop) if crop is not None else image.copy()
                        width, height = image.width(), image.height()
                    elif crop is not None:
                        width, height = crop.width(), crop.height()
                    elapsed = (time.perf_counter() - start) * 1000
                    proxy = "" if request.tier == 1 else f"  ·  proxy 1/{request.tier}"
                    self.signals.finished.emit(
                        (request, cancel), DisplayedFrame((height, width, 4)), image,
                        f"{width * request.tier} × {height * request.tier}{proxy}  ·  "
                        f"{elapsed:.0f} ms  ·  display cache hit  ·  display {gpudisplay.status()}",
                        render_region)
                    return
                if tiled and request.display and request.tier == 1 and not request.playing:
                    # Nothing at full resolution yet, but playback with "Proxy while playing" on
                    # has usually already cached this frame at a proxy tier. Put that up at once so
                    # scrubbing over played frames is immediate, then refine to full resolution
                    # below. The stand-in is display-only: it never becomes `self.frame`.
                    self._offer_cached_proxy(request, cancel, target, view, exposure, channel,
                                             background)
                if tiled:
                    tile_result = self.tile_executor.compose_region(request.document, target, render_region,
                                                                     frame=request.frame,
                                                                     tier=request.tier, cancel=cancel)
                    frame = tile_result.pixels
                    tile_detail = (f"  ·  tiles {tile_result.tile_hits} hit/{tile_result.tile_misses} miss"
                                   if tile_result.tiled else "  ·  tile fallback")
                else:
                    frame = self.evaluator.evaluate(request.document, cancel=cancel,
                                                    frame=request.frame, tier=request.tier)
                    tile_detail = "  ·  full-frame fallback"
                if cancel.is_set():
                    raise Cancelled()
                # Read-ahead warms the display cache too, not just the raw composite: a prefetch
                # request pays the OCIO transform cost on the executor's idle time, so by the time
                # forward playback actually reaches that frame, it is a cache hit instead of the
                # first-time ~3s cost. Only a display request needs the QImage handed back to the
                # viewer; a prefetch still runs the transform purely for its cache side effect.
                built = to_qimage(frame, exposure, channel, background=background, view=view)
                self.display_cache.put(display_key, bytes(built.constBits()),
                                       built.width(), built.height(), built.bytesPerLine(),
                                       region_key)
                image = built if request.display else None
                elapsed = (time.perf_counter() - start) * 1000
                proxy = "" if request.tier == 1 else f"  ·  proxy 1/{request.tier}"
                self.signals.finished.emit((request, cancel), frame, image, f"{frame.shape[1] * request.tier} × {frame.shape[0] * request.tier}{proxy}  ·  {elapsed:.0f} ms{tile_detail}  ·  cache {self.evaluator.bytes / 1048576:.1f} / {self.evaluator.budget / 1048576:.0f} MiB  ·  display {gpudisplay.status()}", render_region)
            except Cancelled:
                self.signals.finished.emit((request, cancel), None, None, "Cancelled", None)
            except Exception as error:
                self.signals.finished.emit((request, cancel), None, None, str(error), None)
        self.executor.submit(work)

    def _offer_cached_proxy(self, request, cancel, target, view, exposure, channel, background):
        """Emit the best cached proxy picture of `request.frame`, if any. Runs on the worker."""
        for tier in PROXY_TIERS:
            if tier == 1 or cancel.is_set():
                continue
            try:
                bounds = self.tile_executor.canvas_region(request.document, target,
                                                          frame=request.frame, tier=tier)
            except Exception:
                return
            key = DisplayCache.key(request.document, target, request.frame, tier, view,
                                   exposure, channel, background)
            cached = self.display_cache.get(key, (bounds.x, bounds.y, bounds.width, bounds.height))
            if cached is None:
                continue
            data, width, height, bytes_per_line = cached
            image = QImage(data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self.signals.interim.emit(
                (request, cancel), image,
                f"{width * tier} × {height * tier}  ·  proxy 1/{tier} from cache  ·  "
                f"rendering full resolution…", tier, bounds)
            return

    def preview_interim(self, payload, image, status, scale, render_region):
        request, cancel = payload
        current = self.dispatcher.document["time"]["current"]
        # Only for the scrub it was made for, and never over the finished picture: the final
        # result for this generation always wins, whichever order the two arrive in.
        if (cancel.is_set() or self.playing or request.generation != self.generation
                or request.frame != current or self.frame_generation == request.generation):
            return
        self.statusBar().showMessage(status)
        self.viewer_info.setText(status)
        self._show_image(image, scale, render_region)

    def _show_image(self, image, scale, render_region):
        previous = self.viewer.sceneRect().size()
        self.viewer.scene().clear()
        if scale != 1:
            # Show the proxy at the comp's real size so framing, pans and zooms do not change
            # when an artist drops the tier. Fast transform on purpose: this is a display upscale
            # of an approximation, and a smooth filter would only make it look more finished
            # than it is.
            image = image.scaled(image.width() * scale, image.height() * scale,
                                 Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.FastTransformation)
        pixmap = self.viewer.scene().addPixmap(QPixmap.fromImage(image))
        if render_region is not None:
            # Tile result coordinates are data-window coordinates. Keep the pixmap at that
            # location instead of rebasing the crop to (0,0), otherwise a pan would make the
            # image visibly jump and EXR overscan would be lost at display.
            pixmap.setPos(render_region.x * scale, render_region.y * scale)
            scene_rect = QRectF(render_region.full_x * scale, render_region.full_y * scale,
                                render_region.full_width * scale, render_region.full_height * scale)
        else:
            scene_rect = QRectF(0, 0, image.width(), image.height())
        self.viewer.setSceneRect(scene_rect)
        self.viewer.draw_format_overlay(scene_rect)
        if previous.width() != scene_rect.width() or previous.height() != scene_rect.height():
            self.viewer.fit()

    def preview_ready(self, payload, frame, image, status, render_region=None):
        request, cancel = payload
        self.preview_queue.finish(cancel)
        self.busy = False
        if frame is None and not cancel.is_set():
            # Logged regardless of whether this result ends up on screen: a read-ahead request
            # can fail on a frame that never becomes "current" and so never reaches viewer_info,
            # but "frame 47 of this sequence is corrupt" is real information either way.
            self.render_errors.append({
                "frame": request.frame, "generation": request.generation,
                "target": request.document.get("view"), "message": status,
                "timestamp": time.time(),
            })
        if request.display and request.playing and not cancel.is_set():
            # The pacing signal playback_tick bounds the playhead against: a completed attempt
            # (success or evaluation error) is one frame's worth of render capacity spent, so the
            # transport may advance by one more. A cancelled request never finished its own
            # target and must not count, or the playhead could outrun render capacity that was
            # never actually delivered.
            self.playback_frames_rendered += 1
        # The preview timer is single-shot and a tick that arrives while busy drops its own
        # wakeup, so without this the next queued frame waits for the following tick even though
        # the worker is free. Kicking it here is what lets playback run at the renderer's
        # sustained rate instead of one frame per tick-that-happened-to-find-us-idle.
        if self.playing and len(self.preview_queue):
            self.timer.start(0)
        current = self.dispatcher.document["time"]["current"]
        # Outside playback the rule stays strict: a result is displayable only if it is still the
        # playhead and still the newest request (docs/PLAYBACK.md criterion 4). During playback
        # that rule is unsatisfiable for any frame costing more than one frame interval, because
        # the transport has already ticked past it by the time it finishes — which froze the
        # viewer entirely instead of dropping frames. While playing, accept any display result
        # newer than what is on screen and show it; the transport keeps following the wall clock,
        # so this drops timeline positions rather than stalling.
        fresh = request.generation == self.generation and request.frame == current
        catching_up = (self.playing and request.playing
                       and request.generation > self.frame_generation)
        if request.display and (fresh or catching_up):
            if self.playing:
                status += (f"  ·  ahead {len(self.preview_queue)}/{self.preview_queue.max_prefetch}"
                           f"  ·  dropped {self.playback_dropped_frames}")
            self.frame = frame
            self.frame_generation = request.generation
            self.statusBar().showMessage(status)
            self.viewer_info.setText(status if frame is not None else "Evaluation error")
            if frame is not None:
                self._show_image(image, request.tier, render_region)
            else:
                self.viewer.scene().clear()
                text = self.viewer.scene().addText(status)
                text.setDefaultTextColor(QColor("#e3b18d"))
                self.viewer.fit()
                self.viewer.draw_format_overlay(None)
        # Every completed render -- display or read-ahead -- may have added a display-cache entry,
        # which is exactly when the orange band grows. Refreshing here is what makes the strip
        # fill in live during playback instead of only after the next document edit.
        self.refresh_timeline_marks()
        if len(self.preview_queue):
            self.start_preview()
        elif not self.playing and self.show_thumbnails:
            self.thumbnail_timer.start()

    def schedule_thumbnails(self):
        """Render postage stamps for nodes whose upstream picture changed, when the viewer is idle."""
        if (not self.show_thumbnails or self.thumbnails_closed or self.playing or self.busy
                or len(self.preview_queue)):
            return
        document = copy.deepcopy(self.dispatcher.document)
        frame = document["time"]["current"]
        view = self.display_view.currentText()
        wanted = []
        for key, node in document["nodes"].items():
            if not wants_thumbnail(node):
                continue
            identity = thumbnail_key(document, key, frame, view)
            cached = self.thumbnails.get(key)
            if cached is None or cached[0] != identity:
                wanted.append((key, identity))
        for key in list(self.thumbnails):
            if key not in document["nodes"]:
                del self.thumbnails[key]
        if not wanted:
            return
        cancel = threading.Event()
        self.thumbnail_cancel = cancel
        tier = max(PROXY_TIERS)

        def work():
            for key, identity in wanted:
                if cancel.is_set():
                    return
                try:
                    pixels = self.evaluator.evaluate(document, key, cancel=cancel,
                                                     frame=frame, tier=tier)
                except Cancelled:
                    return
                except Exception:
                    # An unset Read path or a broken branch has no picture; the band stays empty.
                    continue
                height, width = pixels.shape[:2]
                if not width or not height:
                    continue
                # Decimate before the view transform so the costly part runs on ~13k pixels.
                step = max(1, int(math.ceil(max(width / THUMB_WIDTH, height / THUMB_HEIGHT))))
                image = to_qimage(pixels[::step, ::step], view=view)
                image = image.scaled(THUMB_WIDTH, THUMB_HEIGHT, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                self.signals.thumbnail.emit(key, identity, image)
        self.executor.submit(work)

    def thumbnail_ready(self, key, identity, image):
        if key not in self.dispatcher.document["nodes"]:
            return
        self.thumbnails[key] = (identity, image)
        item = self.graph.items_by_id.get(key)
        if item is not None:
            item.set_thumbnail(image)

    def set_show_thumbnails(self, enabled):
        enabled = bool(enabled)
        if enabled == self.show_thumbnails:
            return
        self.show_thumbnails = enabled
        self.preferences.set_thumbnails(enabled)
        self.graph.rebuild()
        if enabled:
            self.thumbnail_timer.start()
        else:
            self.thumbnail_cancel.set()

    def write_target(self, key):
        """Resolve a Write node's (path, format, bits), or raise with the reason it cannot render."""
        params = self.dispatcher.document["nodes"][key]["params"]
        path = params["path"].strip()
        if not path:
            raise ValueError("Set an output path on the Write node first")
        chosen = params["file_type"]
        suffix = Path(path).suffix.lower().lstrip(".")
        if chosen == "Auto":
            if suffix not in ("exr", "png"):
                raise ValueError(f"Cannot infer a format from {Path(path).name!r}; "
                                 f"choose exr or png, or use that extension")
            chosen = suffix
        elif suffix != chosen:
            # Honour the explicit choice and make the filename agree with it, rather than writing
            # PNG bytes into a file called .exr.
            path = str(Path(path).with_suffix("." + chosen))
        return path, chosen, params["bit_depth"]

    def render_write(self, key, single=True):
        """Render a Write node. Always full resolution through the reference evaluator."""
        try:
            path, file_type, bits = self.write_target(key)
        except ValueError as error:
            QMessageBox.warning(self, "Write", str(error))
            return
        time_range = self.dispatcher.document["time"]
        frames = ([time_range["current"]] if single
                  else list(range(time_range["first"], time_range["last"] + 1)))
        if len(frames) > 1 and not is_sequence(path):
            QMessageBox.warning(self, "Write",
                                f"{Path(path).name!r} is a single file, so a {len(frames)}-frame "
                                f"range would overwrite it every frame. Use a padded pattern "
                                f"such as render.%04d.exr.")
            return
        # A range render is long and must stay interruptible; the document is snapshotted once so
        # an edit landing mid-render cannot change what the remaining frames are rendered from.
        document = copy.deepcopy(self.dispatcher.document)
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        progress = QProgressDialog(f"Rendering {Path(path).name}…", "Cancel", 0, len(frames), self)
        progress.setWindowTitle("Write")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        written = 0
        try:
            for index, frame_number in enumerate(frames):
                if progress.wasCanceled():
                    break
                progress.setValue(index)
                progress.setLabelText(f"Rendering frame {frame_number} of "
                                      f"{frames[0]}–{frames[-1]}…")
                QApplication.processEvents()
                # Render the Write node's own upstream tree. Omitting the target would evaluate
                # the document's view instead, so a render would silently follow whatever the
                # Viewer was pointed at -- in Nuke a Write is independent of the Viewer, and an
                # export that changes with the current view is the worst kind of wrong output:
                # plausible-looking frames of the wrong tree.
                pixels = self.evaluator.evaluate(document, key, frame=frame_number, tier=1)
                target = sequence_path(path, frame_number) if is_sequence(path) else path
                if file_type == "exr":
                    write_exr(target, pixels, bits=bits)
                else:
                    write_png(target, pixels)
                written += 1
        except (ValueError, OSError) as error:
            progress.close()
            QMessageBox.warning(self, "Write failed",
                                f"{error}\n\n{written} of {len(frames)} frames written.")
            return
        finally:
            progress.close()
        detail = f"{file_type} · {bits + ' float' if file_type == 'exr' else '8-bit sRGB'}"
        cancelled = " · cancelled" if written < len(frames) else ""
        self.statusBar().showMessage(
            f"Wrote {written} frame{'' if written == 1 else 's'} to {path} · {detail}{cancelled}",
            12000)

    def export(self):
        if self.frame is None or self.frame_generation != self.generation:
            self.statusBar().showMessage("Wait for a valid current preview before exporting", 8000)
            return
        frame = self.frame  # Keep the chosen image stable across the modal dialog.
        path, selected_filter = QFileDialog.getSaveFileName(self, "Export image", "output.exr", "OpenEXR half RGBA ZIPS (*.exr);;OpenEXR 32-bit float RGBA ZIPS (*.exr);;PNG sRGB 8-bit (*.png)")
        if not path:
            return
        try:
            if not Path(path).suffix:
                path += ".png" if "PNG" in selected_filter else ".exr"
            # Preview may now be a bounded tile crop even at tier 1. Exports always render a
            # complete full-resolution reference frame; a viewport artifact can never escape.
            self.statusBar().showMessage("Rendering full resolution for export…")
            QApplication.processEvents()
            frame = self.evaluator.evaluate(copy.deepcopy(self.dispatcher.document),
                                            frame=self.dispatcher.document["time"]["current"],
                                            tier=1)
            if Path(path).suffix.lower() == ".exr":
                # Half is the default; the second EXR filter is the explicit opt-in for data
                # passes that must keep all 32 bits.
                bits = "float" if "32-bit" in selected_filter else "half"
                write_exr(path, frame, bits=bits)
                detail = f"{bits} float, ZIPS"
            else:
                write_png(path, frame)
                detail = "8-bit sRGB"
            self.statusBar().showMessage(f"Exported {path} · {detail} · viewer exposure/channel controls are display-only", 10000)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Export failed", str(error))

    def update_status(self, state, detail, percent):
        labels = {"idle": "Check for updates", "checking": "Checking…",
                  "available": f"Download v{detail}", "downloading": f"Downloading {percent}%",
                  "ready": "Restart to update", "current": "Up to date",
                  "error": "Update failed · Retry", "unsupported": "Updates unavailable"}
        self.update_button.setText(labels.get(state, state))
        self.update_button.setEnabled(state not in ("checking", "downloading"))
        self.update_button.setToolTip(detail or f"NodeBased {__version__}")
        if state in ("error", "unsupported"):
            self.statusBar().showMessage(detail, 15000)

    def update_clicked(self):
        if self.updater.state == "available":
            self.updater.fetch()
        elif self.updater.state == "ready":
            if not self.confirm_discard():
                return
            try:
                self.updater.install()
            except (ValueError, OSError) as error:
                self.updater.changed.emit("error", str(error), 0)
                return
            self.update_exit = True
            self.close()
        else:
            self.updater.check()

    def closeEvent(self, event):
        if not self.update_exit and not self.confirm_discard():
            event.ignore()
            return
        self.updater.cancel.set()
        if self._tracker_cancel is not None:
            self._tracker_cancel.set()
        self.timer.stop()
        self.playback_timer.stop()
        self.thumbnails_closed = True
        self.thumbnail_timer.stop()
        self.thumbnail_cancel.set()
        self.preview_queue.cancel()
        # GL resources are thread-affine to the worker thread that built them; release them
        # there, before that thread stops, or the context can never be made current again.
        try:
            self.executor.submit(gpudisplay.shutdown).result(timeout=5)
        except Exception:
            pass
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.decode_pool.shutdown()
        if self.server:
            self.server.close()
        QApplication.instance().removeEventFilter(self)
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="NodeBased native 2D compositing workbench")
    parser.add_argument("project", nargs="?")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--network-probe", metavar="OUTPUT_JSON", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test", metavar="OUTPUT_JSON", help=argparse.SUPPRESS)
    parser.add_argument("--agent", metavar="LOCAL_NAME", help="opt-in user-local agent socket; no network listener")
    args = parser.parse_args()
    if args.network_probe:
        from .updater import open_url
        with open_url("https://api.github.com/repos/neodimo/NodeBased", timeout=20) as response:
            repository = json.loads(response.read())
        Path(args.network_probe).write_text(json.dumps({"ok": repository.get("full_name") == "neodimo/NodeBased"}), encoding="utf-8")
        return 0
    app = QApplication(sys.argv[:1])
    app.setApplicationName("NodeBased")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    try:
        window = Window(load_document(args.project) if args.project else None, args.agent)
    except (ValueError, OSError) as error:
        print(f"NodeBased: {error}", file=sys.stderr)
        return 1
    if args.project:
        window.project_path = str(Path(args.project).resolve())
        window.update_title()
    window.show()
    window.update_title()
    if args.smoke_test:
        deadline = time.monotonic() + 30
        def smoke():
            if window.frame is None and time.monotonic() < deadline:
                QTimer.singleShot(100, smoke)
                return
            from .media import selftest
            media = selftest()
            result = {"version": __version__, "ok": window.frame is not None and all(media.values()),
                      "update_button": window.update_button.text(), "media": media,
                      "shape": list(window.frame.shape) if window.frame is not None else None}
            Path(args.smoke_test).write_text(json.dumps(result), encoding="utf-8")
            window.saved_document = window.dispatcher.document
            window.close()
            app.exit(0 if result["ok"] else 1)
        QTimer.singleShot(100, smoke)
    return app.exec()
