"""Native Qt desktop workbench; UI writes only through Dispatcher commands."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
from pathlib import Path
import sys
import threading
import time

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal, QObject, QEvent
from PySide6.QtGui import QAction, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QKeySequence, QPolygonF, QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsSimpleTextItem,
    QGraphicsPathItem, QGraphicsItem, QDockWidget, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QLineEdit, QPushButton, QFormLayout, QFileDialog, QMessageBox, QToolBar,
    QInputDialog, QSplitter, QScrollArea, QDialog, QListWidget, QListWidgetItem, QStyle, QSlider)

from . import __version__
from .updater import Updater
from .core import (Dispatcher, SPECS, LIMITS, TIME_LIMITS, demo_document, load_document,
                   IMAGE_FILTER_KINDS)
from .imaging import Evaluator, Cancelled, to_qimage, write_png
from .playback import PlaybackQueue, DisplayCache

from .theme import COLORS, STYLE
from .color import VIEWS
from .core import CHOICES
from .media import write_exr
from .cachetier import DiskCache
from .tileexec import TileExecutor
from .tiles import TileRegion
from .tiers import auto_playback_tier

# Delivery rates an artist actually asks for, offered next to the free-form rate box. 24 leads
# because it is the document default; the rest are the rates a comp gets handed in practice.
FPS_PRESETS = (("24", 24.0), ("23.976", 24000.0 / 1001.0), ("25", 25.0), ("29.97", 30000.0 / 1001.0),
               ("30", 30.0), ("48", 48.0), ("50", 50.0), ("59.94", 60000.0 / 1001.0), ("60", 60.0))


def resource_path(relative: str) -> Path:
    """Locate a source asset both from a checkout and a PyInstaller bundle."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / relative



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
        super().__init__(QGraphicsScene())

    def keyPressEvent(self, event):
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


class NodeItem(QGraphicsRectItem):
    def __init__(self, graph, key, node):
        self.is_dot = node["type"] == "Dot"
        super().__init__(0, 0, 20 if self.is_dot else 190, 20 if self.is_dot else 52)
        self.graph, self.key = graph, key
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(*node["pos"])
        self.setBrush(QColor("#303033" if not self.is_dot else "#23242a"))
        self.setPen(QPen(QColor(COLORS[node["type"]]), 1.5))
        if self.is_dot:
            # Dots are graph routing points, not miniature processing cards.
            # Keep them compact and put their sockets on the vertical noodle path.
            self.setRect(0, 0, 20, 20)
            self.inputs = {"input": Port(self, "input", 10, 0)}
            self.output = Port(self, None, 10, 20)
            return
        accent = QGraphicsRectItem(0, 0, 4, 52, self)
        accent.setBrush(QColor(COLORS[node["type"]]))
        accent.setPen(QPen(Qt.PenStyle.NoPen))
        title = QGraphicsSimpleTextItem(node["name"][:26], self)
        title.setBrush(QColor("#eeeef2"))
        title.setPos(12, 7)
        subtitle = QGraphicsSimpleTextItem(("BYPASSED · " if node["disabled"] else "") + node["type"] + ("  • viewing" if graph.window.dispatcher.document["view"] == key else ""), self)
        subtitle.setBrush(QColor("#a6a6b0"))
        subtitle.setPos(12, 29)
        # Inputs live on the top centre. Multiple inputs fan out symmetrically around
        # it, keeping a single-input node exactly at the familiar centred position.
        slots = list(node["inputs"])
        spacing = 34
        self.inputs = {slot: Port(self, slot, 95 + (i - (len(slots) - 1) / 2) * spacing, 0)
                       for i, slot in enumerate(slots)}
        self.output = Port(self, None, 95, 52)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and hasattr(self, "output"):
            self.graph.update_edges()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        if not self.is_dot:
            return super().paint(painter, option, widget)
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
    def __init__(self, color="#898995", dashed=False, arrow=True):
        super().__init__()
        self.color = QColor(color)
        self.setPen(QPen(self.color, 3.25, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine,
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
                    edge = Edge()
                    edge.setZValue(-1)
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

    def mousePressEvent(self, event):
        self.ctrl_handles_visible = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.viewport().update()
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            edge = self.edge_handle_at(self.mapToScene(event.position().toPoint()))
            if edge:
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
        painter.setPen(QPen(QColor("#313135"), 1))
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


class ProjectSettingsDialog(QDialog):
    """Small project-settings surface modelled after a compositor's project settings.

    The bundled ACES config, display, and ACEScg processing space are explicit rather than
    magical constants. They are read-only until external OCIO configs are supported; the artist
    can choose the saved default view and the viewer background today.
    """
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project Settings")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
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


class Window(QMainWindow):
    def __init__(self, document=None, agent_name=None):
        super().__init__()
        self.dispatcher = Dispatcher(document or demo_document())
        self.saved_document = copy.deepcopy(self.dispatcher.document)
        self.project_path = None
        self.update_exit = False
        self.frame = None
        self.frame_generation = -1
        self.generation = 0
        self.busy = False
        self.preview_queue = PlaybackQueue()
        self.display_cache = DisplayCache()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nodebased-preview")
        # The desktop app is where the persistent disk tier is switched on: results evicted from
        # memory survive a restart, so reopening yesterday's comp does not recompute it.
        self.evaluator = Evaluator(disk=DiskCache.shared())
        # Preview takes the tile path when every upstream node supports it. The executor reports
        # an explicit fallback for unsupported graphs; export remains the reference evaluator.
        self.tile_executor = TileExecutor(evaluator=self.evaluator)
        self.signals = PreviewSignals()
        self.signals.finished.connect(self.preview_ready)
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
        for name, callback in [("Open image", self.read_file), ("Add node", self.add_node), ("Save project", self.save_project), ("Export image", self.export)]:
            action = toolbar.addAction(name)
            action.triggered.connect(lambda checked=False, fn=callback: fn())
        toolbar.addSeparator()
        info = QLabel("  2D WORKSPACE")
        info.setObjectName("muted")
        toolbar.addWidget(info)
        toolbar.addSeparator()
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
        self.viewer_info = QLabel("Waiting for image")
        self.viewer_info.setObjectName("muted")
        controls.addWidget(self.viewer_info)
        vl.addLayout(controls)
        self.viewer = Viewer(self)
        self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        vl.addWidget(self.viewer)
        vl.addLayout(self._timeline())
        splitter.addWidget(viewer_panel)
        graph_panel = QWidget()
        gl = QVBoxLayout(graph_panel)
        gl.setContentsMargins(0, 0, 0, 0)
        help_label = QLabel("  NODE GRAPH     Tab search/add  ·  R/G/M/T/B/C/S/O create  ·  1 view  ·  D bypass  ·  F frame  ·  MMB pan  ·  "
                            "drag output ↔ input to wire  ·  Ctrl-drag noodle midpoint inserts Dot  ·  click a wired input to rewire")
        help_label.setObjectName("muted")
        gl.addWidget(help_label)
        self.graph = Graph(self)
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
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setToolTip("Scrub the playhead. ← → step, Home/End jump to the range ends.")
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
                self.statusBar().showMessage(str(error), 10000)
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
            if self.proxy.currentData() == 1:
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
            (edit, "Project settings…", "S", self.project_settings),
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
            result = self.dispatcher.execute(cmd)
            if cmd.get("op") in ("time", "settings"):
                self.sync_timeline()
                self.sync_project_settings()
                self.update_title()
                if render:
                    self.request_preview()
            else:
                self.after_command(render, sync_settings=cmd.get("op") in ("load", "undo", "redo"))
            return result
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.statusBar().showMessage(str(error), 10000)
            return None

    def agent_command(self, cmd):
        # Return machine-readable errors through the bridge, not only the status bar.
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

    def after_command(self, render=True, sync_settings=False):
        key = self.graph.selected_id()
        self.graph.rebuild()
        self.inspect(key)
        # Undo, project load and agent "time" edits all land here, so the strip follows the
        # document rather than only the widget that happened to be dragged.
        self.sync_timeline()
        if sync_settings:
            self.sync_project_settings()
        self.update_title()
        if render:
            self.request_preview()

    def sync_project_settings(self):
        view = self.dispatcher.document["settings"]["color"]["view"]
        if self.display_view.currentText() != view:
            self.display_view.blockSignals(True)
            self.display_view.setCurrentText(view)
            self.display_view.blockSignals(False)

    def project_settings(self):
        dialog = ProjectSettingsDialog(copy.deepcopy(self.dispatcher.document["settings"]), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.command({"op": "settings", "settings": dialog.changes()})

    def update_title(self):
        dirty = self.dispatcher.document != self.saved_document
        self.setWindowTitle(f"NodeBased {__version__} · {Path(self.project_path).name if self.project_path else 'Untitled'}{' *' if dirty else ''}")

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
            heading = QLabel(node["type"].upper())
            heading.setStyleSheet(f"color: {COLORS[node['type']]}; font-weight: 700; font-size: 15px")
            form.addRow(heading)
            name = QLineEdit(node["name"])
            name.editingFinished.connect(lambda: self.defer_command({"op": "rename", "id": key, "name": name.text()}))
            form.addRow("Name", name)
            for param, value in node["params"].items():
                if param in CHOICES:
                    control = QComboBox()
                    control.addItems(CHOICES[param])
                    control.setCurrentText(value)
                    control.currentTextChanged.connect(lambda v, k=key, p=param: self.defer_command({"op": "set", "id": k, "param": p, "value": v}))
                    form.addRow({"colorspace": "Input space", "alpha_mode": "Alpha", "red_from": "Red", "green_from": "Green",
                                 "blue_from": "Blue", "alpha_from": "Alpha"}.get(param, param), control)
                elif isinstance(value, str):
                    control = QLineEdit(value)
                    control.editingFinished.connect(lambda k=key, p=param, w=control: self.defer_command({"op": "set", "id": k, "param": p, "value": w.text()}))
                    form.addRow(param.title(), control)
                    if param == "path":
                        browse = QPushButton("Browse image…")
                        browse.clicked.connect(lambda checked=False, k=key: self.browse_read(k))
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
                    control.setValue(value)
                    control.setKeyboardTracking(False)
                    control.editingFinished.connect(lambda k=key, p=param, w=control: self.defer_command({"op": "set", "id": k, "param": p, "value": w.value()}))
                    form.addRow(param.title(), control)
            if node["type"] == "Merge":
                form.addRow(QLabel("A over B · scene-linear, premultiplied\nInputs must have matching dimensions."))
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
            if node["type"] in IMAGE_FILTER_KINDS:
                form.addRow(QLabel("Optional mask input + 'mix' blend with original\n"
                                    "result = mix * mask.a * filtered + (1 - mix * mask.a) * source"))
            view = QPushButton("View this node   [1]")
            view.clicked.connect(lambda: self.command({"op": "view", "id": key}))
            form.addRow(view)
        old = self.properties.takeWidget()
        if old:
            old.deleteLater()
        self.properties.setWidget(panel)

    def defer_command(self, cmd):
        # Do not destroy an editor while it is emitting editingFinished.
        QTimer.singleShot(0, lambda: self.command(cmd))

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

    def node_position(self, desired):
        """Find a nearby vacant location; never drop a new node on an existing one."""
        desired = QPointF(round(desired.x()), round(desired.y()))
        candidates = [QPointF(0, 0)]
        for radius in range(80, 801, 80):
            candidates.extend(QPointF(x, y) for x, y in
                              ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                               (radius, radius), (-radius, radius), (radius, -radius), (-radius, -radius)))
        occupied = [item.sceneBoundingRect().adjusted(-12, -12, 12, 12)
                    for item in self.graph.items_by_id.values()]
        for offset in candidates:
            pos = desired + offset
            rect = QRectF(pos.x(), pos.y(), 190, 52)
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
            anchor = self.graph.items_by_id[source].pos() + QPointF(220, 0)
        else:
            anchor = position if position is not None else self.graph.last_click_scene_pos
        pos = self.node_position(anchor)
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
        path, _ = QFileDialog.getOpenFileName(self, "Read image", "", "Images (*.exr *.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.command({"op": "set", "id": key, "param": "path", "value": path})

    def read_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Read image", "", "Images (*.exr *.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.add_node("Read", {"path": path})
            self.command({"op": "view", "id": self.graph.selected_id()})

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
        self.preview_queue.replace(self.generation, frame, snapshot, future,
                                   tier=self.proxy.currentData(), viewport=viewport,
                                   playing=self.playing,
                                   cancel_active=not (playhead_only and self.playing))
        self.timer.start(0 if self.playing else 35)

    def start_preview(self):
        if self.busy:
            return
        queued = self.preview_queue.take()
        if queued is None:
            return
        request, cancel = queued
        self.busy = True
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
                if target and self.tile_executor.supports_tiled(request.document, target):
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
                display_key = DisplayCache.key(request.document, target, request.frame,
                                               request.tier, view, exposure, channel, background)
                cached = self.display_cache.get(display_key)
                image = None
                display_hit = cached is not None
                if display_hit:
                    if request.display:
                        data, width, height, bytes_per_line = cached
                        image = QImage(data, width, height, bytes_per_line,
                                      QImage.Format.Format_RGB888).copy()
                else:
                    built = to_qimage(frame, exposure, channel, background=background, view=view)
                    self.display_cache.put(display_key, bytes(built.constBits()),
                                           built.width(), built.height(), built.bytesPerLine())
                    if request.display:
                        image = built
                elapsed = (time.perf_counter() - start) * 1000
                proxy = "" if request.tier == 1 else f"  ·  proxy 1/{request.tier}"
                display = "  ·  display cache hit" if display_hit else ""
                self.signals.finished.emit((request, cancel), frame, image, f"{frame.shape[1] * request.tier} × {frame.shape[0] * request.tier}{proxy}  ·  {elapsed:.0f} ms{tile_detail}  ·  cache {self.evaluator.bytes / 1048576:.1f} / {self.evaluator.budget / 1048576:.0f} MiB{display}", render_region)
            except Cancelled:
                self.signals.finished.emit((request, cancel), None, None, "Cancelled", None)
            except Exception as error:
                self.signals.finished.emit((request, cancel), None, None, str(error), None)
        self.executor.submit(work)

    def preview_ready(self, payload, frame, image, status, render_region=None):
        request, cancel = payload
        self.preview_queue.finish(cancel)
        self.busy = False
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
            previous = self.viewer.scene().itemsBoundingRect().size()
            self.viewer.scene().clear()
            if frame is not None:
                if request.tier != 1:
                    # Show the proxy at the comp's real size so framing, pans and zooms do not
                    # change when an artist drops the tier. Fast transform on purpose: this is a
                    # display upscale of an approximation, and a smooth filter would only make it
                    # look more finished than it is.
                    image = image.scaled(image.width() * request.tier, image.height() * request.tier,
                                         Qt.AspectRatioMode.IgnoreAspectRatio,
                                         Qt.TransformationMode.FastTransformation)
                pixmap = self.viewer.scene().addPixmap(QPixmap.fromImage(image))
                if render_region is not None:
                    # Tile result coordinates are data-window coordinates. Keep the pixmap at
                    # that location instead of rebasing the crop to (0,0), otherwise a pan would
                    # make the image visibly jump and EXR overscan would be lost at display.
                    pixmap.setPos(render_region.x * request.tier, render_region.y * request.tier)
                    scene_rect = QRectF(render_region.full_x * request.tier,
                                        render_region.full_y * request.tier,
                                        render_region.full_width * request.tier,
                                        render_region.full_height * request.tier)
                else:
                    scene_rect = QRectF(0, 0, image.width(), image.height())
                self.viewer.setSceneRect(scene_rect)
                self.viewer.draw_format_overlay(scene_rect)
                if previous.width() != scene_rect.width() or previous.height() != scene_rect.height():
                    self.viewer.fit()
            else:
                text = self.viewer.scene().addText(status)
                text.setDefaultTextColor(QColor("#e3b18d"))
                self.viewer.fit()
                self.viewer.draw_format_overlay(None)
        if len(self.preview_queue):
            self.start_preview()

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
        self.timer.stop()
        self.playback_timer.stop()
        self.preview_queue.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
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
