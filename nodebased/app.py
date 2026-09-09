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

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal, QObject
from PySide6.QtGui import QAction, QColor, QCursor, QPainter, QPainterPath, QPen, QPixmap, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsSimpleTextItem,
    QGraphicsPathItem, QGraphicsItem, QDockWidget, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QLineEdit, QPushButton, QFormLayout, QFileDialog, QMessageBox, QToolBar,
    QInputDialog, QSplitter, QScrollArea)

from . import __version__
from .updater import Updater
from .core import Dispatcher, SPECS, LIMITS, demo_document, load_document
from .imaging import Evaluator, Cancelled, to_qimage, write_png

from .theme import COLORS, STYLE
from .color import VIEWS
from .core import CHOICES
from .media import write_exr



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


class Port(QGraphicsEllipseItem):
    def __init__(self, node, slot, x, y):
        super().__init__(-6, -6, 12, 12, node)
        self.node, self.slot = node, slot
        self.setPos(x, y)
        self.setBrush(QColor("#1b1b1d"))
        self.setPen(QPen(QColor("#a4a4ae"), 1.5))
        self.setToolTip("Output: click then an input" if slot is None else
                         f"Input {slot}: click after an output to wire; click again to pick the wire up and rewire; right-click disconnects")
        if slot:
            label = QGraphicsSimpleTextItem(slot, node)
            label.setBrush(QColor("#b4b4bd"))
            label.setPos(x + 9, y - 18)

    def mousePressEvent(self, event):
        graph = self.node.graph
        if event.button() == Qt.MouseButton.RightButton and self.slot is not None:
            graph.cancel_wire()
            graph.window.defer_command({"op": "connect", "id": self.node.key, "input": self.slot, "source": None})
        elif self.slot is None:
            graph.start_wire(self.node.key)
            graph.window.statusBar().showMessage("Connect: click an input port · Esc cancels")
        elif graph.wire_source:
            source = graph.wire_source
            graph.cancel_wire()
            graph.window.defer_command({"op": "connect", "id": self.node.key, "input": self.slot, "source": source})
        else:
            current_source = graph.window.dispatcher.document["nodes"][self.node.key]["inputs"][self.slot]
            if current_source:
                graph.start_wire(current_source)
                graph.window.defer_command({"op": "connect", "id": self.node.key, "input": self.slot, "source": None})
                graph.window.statusBar().showMessage("Wire picked up: click a new input, or empty space to drop · Esc cancels")
        event.accept()


class NodeItem(QGraphicsRectItem):
    def __init__(self, graph, key, node):
        super().__init__(0, 0, 190, 52)
        self.graph, self.key = graph, key
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(*node["pos"])
        self.setBrush(QColor("#303033"))
        self.setPen(QPen(QColor(COLORS[node["type"]]), 1.5))
        accent = QGraphicsRectItem(0, 0, 4, 52, self)
        accent.setBrush(QColor(COLORS[node["type"]]))
        accent.setPen(QPen(Qt.PenStyle.NoPen))
        title = QGraphicsSimpleTextItem(node["name"][:26], self)
        title.setBrush(QColor("#eeeef2"))
        title.setPos(12, 7)
        subtitle = QGraphicsSimpleTextItem(("BYPASSED · " if node["disabled"] else "") + node["type"] + ("  • viewing" if graph.window.dispatcher.document["view"] == key else ""), self)
        subtitle.setBrush(QColor("#a6a6b0"))
        subtitle.setPos(12, 29)
        self.inputs = {slot: Port(self, slot, 30 + i * 120, 0) for i, slot in enumerate(node["inputs"])}
        self.output = Port(self, None, 95, 52)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and hasattr(self, "output"):
            self.graph.update_edges()
        return super().itemChange(change, value)


class Graph(PanZoomView):
    def __init__(self, window):
        self.window = window
        super().__init__(QGraphicsScene())
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.viewport().setMouseTracking(True)
        self.items_by_id, self.edges = {}, []
        self.wire_source = None
        self.pending_edge = None
        self.scene().selectionChanged.connect(self.selection_changed)

    def _new_pending_edge(self):
        edge = QGraphicsPathItem()
        edge.setPen(QPen(QColor("#e3b18d"), 2, Qt.PenStyle.DashLine))
        edge.setZValue(10)
        self.scene().addItem(edge)
        return edge

    def start_wire(self, source_key):
        self.wire_source = source_key
        self.pending_edge = self._new_pending_edge()

    def cancel_wire(self):
        if self.pending_edge is not None:
            self.scene().removeItem(self.pending_edge)
        self.wire_source = None
        self.pending_edge = None

    def update_pending_edge(self, scene_pos):
        if self.pending_edge is None or self.wire_source not in self.items_by_id:
            return
        start = self.items_by_id[self.wire_source].output.scenePos()
        path = QPainterPath(start)
        distance = max(40, abs(scene_pos.y() - start.y()) * 0.5)
        path.cubicTo(start + QPointF(0, distance), scene_pos - QPointF(0, distance), scene_pos)
        self.pending_edge.setPath(path)

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
                    edge = QGraphicsPathItem()
                    edge.setPen(QPen(QColor("#898995"), 2))
                    edge.setZValue(-1)
                    self.scene().addItem(edge)
                    self.edges.append((edge, source, key, slot))
        self.update_edges()
        self.scene().blockSignals(False)
        if self.wire_source:
            # A picked-up wire's disconnect command rebuilds the scene mid-drag; keep the preview alive.
            if self.wire_source in self.items_by_id:
                self.pending_edge = self._new_pending_edge()
                self.update_pending_edge(self.mapToScene(self.viewport().mapFromGlobal(QCursor.pos())))
            else:
                self.wire_source = None

    def update_edges(self):
        for edge, source, key, slot in self.edges:
            start = self.items_by_id[source].output.scenePos()
            end = self.items_by_id[key].inputs[slot].scenePos()
            path = QPainterPath(start)
            distance = max(40, abs(end.y() - start.y()) * 0.5)
            path.cubicTo(start + QPointF(0, distance), end - QPointF(0, distance), end)
            edge.setPath(path)

    def selected_id(self):
        return next((i.key for i in self.scene().selectedItems() if isinstance(i, NodeItem)), None)

    def selection_changed(self):
        self.window.inspect(self.selected_id())

    def mousePressEvent(self, event):
        if self.wire_source and event.button() == Qt.MouseButton.LeftButton and not isinstance(self.itemAt(event.position().toPoint()), Port):
            self.cancel_wire()
            self.window.statusBar().showMessage("Wire dropped", 3000)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.wire_source:
            self.update_pending_edge(self.mapToScene(event.position().toPoint()))

    def mouseReleaseEvent(self, event):
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
            self.window.add_node()
        elif event.key() == Qt.Key.Key_F:
            self.fit()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_wire()
        elif event.key() == Qt.Key.Key_1 and key:
            self.window.command({"op": "view", "id": key})
        elif event.key() == Qt.Key.Key_D and key:
            self.window.command({"op": "disable", "id": key, "value": not self.window.dispatcher.document["nodes"][key]["disabled"]})
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            edits = [{"op": "delete", "id": item.key} for item in self.scene().selectedItems() if isinstance(item, NodeItem)]
            self.window.command({"op": "batch", "commands": edits})
        elif event.key() in (Qt.Key.Key_R, Qt.Key.Key_G, Qt.Key.Key_M, Qt.Key.Key_T, Qt.Key.Key_B, Qt.Key.Key_C, Qt.Key.Key_S, Qt.Key.Key_O):
            self.window.add_node({Qt.Key.Key_R: "Read", Qt.Key.Key_G: "Grade", Qt.Key.Key_M: "Merge", Qt.Key.Key_T: "Transform",
                                   Qt.Key.Key_B: "Blur", Qt.Key.Key_C: "Crop", Qt.Key.Key_S: "Shuffle", Qt.Key.Key_O: "ColorCorrect"}[event.key()])
        else:
            super().keyPressEvent(event)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        if self.transform().m11() < 0.25:
            return
        painter.setPen(QPen(QColor("#313135"), 1))
        left, top = math.floor(rect.left() / 32) * 32, math.floor(rect.top() / 32) * 32
        points = [QPointF(x, y) for x in range(left, int(rect.right()), 32) for y in range(top, int(rect.bottom()), 32)]
        painter.drawPoints(points)


class PreviewSignals(QObject):
    finished = Signal(int, object, object, str)


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
        self.pending = False
        self.cancel = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nodebased-preview")
        self.evaluator = Evaluator()
        self.signals = PreviewSignals()
        self.signals.finished.connect(self.preview_ready)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self.start_preview)
        self.setWindowTitle("NodeBased · Untitled")
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
        self.display_view.setToolTip("Display transform only; exports stay independent of the viewer")
        self.display_view.currentTextChanged.connect(self.request_preview)
        controls.addWidget(self.display_view)
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
        self.viewer = PanZoomView(QGraphicsScene())
        self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        vl.addWidget(self.viewer)
        splitter.addWidget(viewer_panel)
        graph_panel = QWidget()
        gl = QVBoxLayout(graph_panel)
        gl.setContentsMargins(0, 0, 0, 0)
        help_label = QLabel("  NODE GRAPH     Tab add  ·  R/G/M/T/B/C/S/O create  ·  1 view  ·  D bypass  ·  F frame  ·  MMB pan  ·  "
                            "click output → input to wire  ·  click a wired input to pick it up and rewire  ·  drop on empty space to disconnect")
        help_label.setObjectName("muted")
        gl.addWidget(help_label)
        self.graph = Graph(self)
        gl.addWidget(self.graph)
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

    def _menus(self):
        file = self.menuBar().addMenu("File")
        edit = self.menuBar().addMenu("Edit")
        for menu, name, shortcut, callback in [
            (file, "Read image…", "Ctrl+I", self.read_file),
            (file, "Open project…", "Ctrl+O", self.open_project),
            (file, "Save", "Ctrl+S", self.save_project),
            (file, "Save as…", "Ctrl+Shift+S", lambda: self.save_project(True)),
            (file, "Export image…", "Ctrl+E", self.export),
            (edit, "Undo", "Ctrl+Z", lambda: self.command({"op": "undo"})),
            (edit, "Redo", "Ctrl+Shift+Z", lambda: self.command({"op": "redo"}))]:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda checked=False, fn=callback: fn())
            menu.addAction(action)

    def command(self, cmd, render=True):
        try:
            result = self.dispatcher.execute(cmd)
            self.after_command(render)
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
            self.after_command()
        return result

    def after_command(self, render=True):
        key = self.graph.selected_id()
        self.graph.rebuild()
        self.inspect(key)
        self.update_title()
        if render:
            self.request_preview()

    def update_title(self):
        dirty = self.dispatcher.document != self.saved_document
        self.setWindowTitle(f"NodeBased {__version__} · {Path(self.project_path).name if self.project_path else 'Untitled'}{' *' if dirty else ''}")

    def inspect(self, key):
        panel = QWidget()
        form = QFormLayout(panel)
        form.setContentsMargins(16, 16, 16, 16)
        if key not in self.dispatcher.document["nodes"]:
            label = QLabel("Select a node to edit its controls.\n\nLinear Rec.709 · float RGBA\nPremultiplied alpha\nEXR / PNG / JPEG / TIFF input\n\n3D and AI generation are roadmap\nmilestones, not active tools yet.")
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

    def add_node(self, kind=None, params=None):
        if not kind:
            kind, ok = QInputDialog.getItem(self, "Create node", "Node (type to search)", list(SPECS), 0, True)
            if not ok:
                return
        selected = self.graph.selected_id()
        pos = self.graph.mapToScene(self.graph.viewport().rect().center())
        key = __import__("uuid").uuid4().hex[:12]
        commands = [{"op": "create", "id": key, "type": kind, "pos": [pos.x(), pos.y()], "params": params or {}}]
        if selected and kind in SPECS and SPECS[kind]["inputs"]:
            commands.append({"op": "connect", "id": key, "input": SPECS[kind]["inputs"][0], "source": selected})
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

    def request_preview(self, *_):
        self.generation += 1
        self.cancel.set()
        self.pending = True
        self.timer.start()

    def start_preview(self):
        if self.busy or not self.pending:
            return
        self.pending = False
        self.busy = True
        self.cancel = threading.Event()
        cancel = self.cancel
        generation = self.generation
        snapshot = copy.deepcopy(self.dispatcher.document)
        exposure, channel = self.exposure.value(), self.channels.currentText()
        view = self.display_view.currentText()
        self.statusBar().showMessage("Evaluating…")
        def work():
            start = time.perf_counter()
            try:
                frame = self.evaluator.evaluate(snapshot, cancel=cancel)
                if cancel.is_set():
                    raise Cancelled()
                image = to_qimage(frame, exposure, channel, view=view)
                elapsed = (time.perf_counter() - start) * 1000
                self.signals.finished.emit(generation, frame, image, f"{frame.shape[1]} × {frame.shape[0]}  ·  {elapsed:.0f} ms  ·  cache {self.evaluator.bytes / 1048576:.1f} / {self.evaluator.budget / 1048576:.0f} MiB")
            except Cancelled:
                self.signals.finished.emit(generation, None, None, "Cancelled")
            except Exception as error:
                self.signals.finished.emit(generation, None, None, str(error))
        self.executor.submit(work)

    def preview_ready(self, generation, frame, image, status):
        self.busy = False
        if generation == self.generation:
            self.frame = frame
            self.frame_generation = generation
            self.statusBar().showMessage(status)
            self.viewer_info.setText(status if frame is not None else "Evaluation error")
            previous = self.viewer.scene().itemsBoundingRect().size()
            self.viewer.scene().clear()
            if frame is not None:
                self.viewer.scene().addPixmap(QPixmap.fromImage(image))
                self.viewer.setSceneRect(QRectF(0, 0, image.width(), image.height()))
                if previous.width() != image.width() or previous.height() != image.height():
                    self.viewer.fit()
            else:
                text = self.viewer.scene().addText(status)
                text.setDefaultTextColor(QColor("#e3b18d"))
                self.viewer.fit()
        if self.pending:
            self.start_preview()

    def export(self):
        if self.frame is None or self.frame_generation != self.generation:
            self.statusBar().showMessage("Wait for a valid current preview before exporting", 8000)
            return
        frame = self.frame  # Keep the chosen image stable across the modal dialog.
        path, selected_filter = QFileDialog.getSaveFileName(self, "Export image", "output.exr", "OpenEXR float RGBA (*.exr);;PNG sRGB 8-bit (*.png)")
        if not path:
            return
        try:
            if not Path(path).suffix:
                path += ".png" if "PNG" in selected_filter else ".exr"
            (write_exr if Path(path).suffix.lower() == ".exr" else write_png)(path, frame)
            self.statusBar().showMessage(f"Exported {path} · viewer exposure/channel controls are display-only", 10000)
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
        self.pending = False
        self.cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.server:
            self.server.close()
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
