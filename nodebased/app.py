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
from PySide6.QtGui import QAction, QColor, QPainter, QPainterPath, QPen, QPixmap, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsSimpleTextItem,
    QGraphicsPathItem, QGraphicsItem, QDockWidget, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QLineEdit, QPushButton, QFormLayout, QFileDialog, QMessageBox, QToolBar,
    QInputDialog, QSplitter, QScrollArea)

from .core import Dispatcher, SPECS, LIMITS, demo_document, load_document
from .imaging import Evaluator, Cancelled, to_qimage, write_png

COLORS = {"Read": "#ceae60", "Checker": "#ceae60", "Constant": "#ceae60", "Grade": "#5fc4ad", "Transform": "#789ee2", "Merge": "#b998da", "Viewer": "#72849b"}
STYLE = """
QMainWindow, QWidget { background: #181c23; color: #d7dde7; font: 12px 'Inter', 'Segoe UI', sans-serif; }
QMenuBar, QMenu, QToolBar { background: #202630; border: 0; }
QMenu::item:selected { background: #354452; }
QDockWidget { font-weight: 600; }
QDockWidget::title { background: #252c37; padding: 9px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #10151c; border: 1px solid #343e4c; border-radius: 4px; padding: 5px; }
QPushButton { background: #2c3643; border: 1px solid #3b4858; border-radius: 4px; padding: 6px 12px; }
QPushButton:hover { background: #3a4d5a; }
QToolButton { padding: 7px; }
QToolButton:hover { background: #344350; }
QSplitter::handle { background: #303845; height: 4px; width: 4px; }
QStatusBar { background: #11161e; color: #9aabbd; }
QLabel#muted { color: #8291a4; }
QLabel#brand { color: #7ed9c2; font-size: 16px; font-weight: 700; padding: 6px; }
QScrollBar:vertical { background: #151a21; width: 10px; }
QScrollBar::handle:vertical { background: #3c4859; min-height: 25px; }
"""


class PanZoomView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        scene.setParent(self)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#11161e"))
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
        self.setBrush(QColor("#10151b"))
        self.setPen(QPen(QColor("#9babbf"), 1.5))
        self.setToolTip("Output: click then an input" if slot is None else f"Input {slot}: click after output; right-click disconnects")
        if slot:
            label = QGraphicsSimpleTextItem(slot, node)
            label.setBrush(QColor("#aab7c7"))
            label.setPos(x + 9, y - 18)

    def mousePressEvent(self, event):
        graph = self.node.graph
        if event.button() == Qt.MouseButton.RightButton and self.slot is not None:
            graph.window.defer_command({"op": "connect", "id": self.node.key, "input": self.slot, "source": None})
        elif self.slot is None:
            graph.wire_source = self.node.key
            graph.window.statusBar().showMessage("Connect: click an input port · Esc cancels")
        elif graph.wire_source:
            source, graph.wire_source = graph.wire_source, None
            graph.window.defer_command({"op": "connect", "id": self.node.key, "input": self.slot, "source": source})
        event.accept()


class NodeItem(QGraphicsRectItem):
    def __init__(self, graph, key, node):
        super().__init__(0, 0, 190, 52)
        self.graph, self.key = graph, key
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setPos(*node["pos"])
        self.setBrush(QColor("#29313e"))
        self.setPen(QPen(QColor(COLORS[node["type"]]), 1.5))
        accent = QGraphicsRectItem(0, 0, 4, 52, self)
        accent.setBrush(QColor(COLORS[node["type"]]))
        accent.setPen(QPen(Qt.PenStyle.NoPen))
        title = QGraphicsSimpleTextItem(node["name"][:26], self)
        title.setBrush(QColor("#eef3fa"))
        title.setPos(12, 7)
        subtitle = QGraphicsSimpleTextItem(("BYPASSED · " if node["disabled"] else "") + node["type"] + ("  • viewing" if graph.window.dispatcher.document["view"] == key else ""), self)
        subtitle.setBrush(QColor("#92a6bc"))
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
        self.items_by_id, self.edges = {}, []
        self.wire_source = None
        self.scene().selectionChanged.connect(self.selection_changed)

    def rebuild(self):
        selected = self.selected_id()
        self.scene().blockSignals(True)
        self.edges = []
        self.items_by_id = {}
        self.scene().clear()
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
                    edge.setPen(QPen(QColor("#73889d"), 2))
                    edge.setZValue(-1)
                    self.scene().addItem(edge)
                    self.edges.append((edge, source, key, slot))
        self.update_edges()
        self.scene().blockSignals(False)

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
            self.wire_source = None
        elif event.key() == Qt.Key.Key_1 and key:
            self.window.command({"op": "view", "id": key})
        elif event.key() == Qt.Key.Key_D and key:
            self.window.command({"op": "disable", "id": key, "value": not self.window.dispatcher.document["nodes"][key]["disabled"]})
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            edits = [{"op": "delete", "id": item.key} for item in self.scene().selectedItems() if isinstance(item, NodeItem)]
            self.window.command({"op": "batch", "commands": edits})
        elif event.key() in (Qt.Key.Key_R, Qt.Key.Key_G, Qt.Key.Key_M, Qt.Key.Key_T):
            self.window.add_node({Qt.Key.Key_R: "Read", Qt.Key.Key_G: "Grade", Qt.Key.Key_M: "Merge", Qt.Key.Key_T: "Transform"}[event.key()])
        else:
            super().keyPressEvent(event)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        if self.transform().m11() < 0.25:
            return
        painter.setPen(QPen(QColor("#2a3340"), 1))
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
        for name, callback in [("Open image", self.read_file), ("Add node", self.add_node), ("Save project", self.save_project), ("Export PNG", self.export)]:
            action = toolbar.addAction(name)
            action.triggered.connect(lambda checked=False, fn=callback: fn())
        toolbar.addSeparator()
        info = QLabel("  2D WORKSPACE   /   M0 reference build")
        info.setObjectName("muted")
        toolbar.addWidget(info)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(splitter)
        viewer_panel = QWidget()
        vl = QVBoxLayout(viewer_panel)
        vl.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("  VIEWER  /  sRGB preview"))
        self.channels = QComboBox()
        self.channels.addItems(["RGB", "R", "G", "B", "A"])
        self.channels.currentTextChanged.connect(self.request_preview)
        controls.addWidget(self.channels)
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
        help_label = QLabel("  NODE GRAPH     Tab add  ·  1 view  ·  D bypass  ·  F frame  ·  MMB pan  ·  click output → input to wire")
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
            (file, "Export PNG…", "Ctrl+E", self.export),
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
        self.setWindowTitle(f"NodeBased · {Path(self.project_path).name if self.project_path else 'Untitled'}{' *' if dirty else ''}")

    def inspect(self, key):
        panel = QWidget()
        form = QFormLayout(panel)
        form.setContentsMargins(16, 16, 16, 16)
        if key not in self.dispatcher.document["nodes"]:
            label = QLabel("Select a node to edit its controls.\n\nLinear float RGBA\nPremultiplied alpha\nPNG / JPEG input\n\n3D and AI generation are roadmap\nmilestones, not active tools yet.")
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
                if isinstance(value, str):
                    control = QLineEdit(value)
                    control.editingFinished.connect(lambda k=key, p=param, w=control: self.defer_command({"op": "set", "id": k, "param": p, "value": w.text()}))
                    form.addRow(param.title(), control)
                    browse = QPushButton("Browse image…")
                    browse.clicked.connect(lambda checked=False, k=key: self.browse_read(k))
                    form.addRow(browse)
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
        path, _ = QFileDialog.getOpenFileName(self, "Read image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.command({"op": "set", "id": key, "param": "path", "value": path})

    def read_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Read image", "", "Images (*.png *.jpg *.jpeg)")
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
        self.statusBar().showMessage("Evaluating…")
        def work():
            start = time.perf_counter()
            try:
                frame = self.evaluator.evaluate(snapshot, cancel=cancel)
                if cancel.is_set():
                    raise Cancelled()
                image = to_qimage(frame, exposure, channel)
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
                text.setDefaultTextColor(QColor("#e1ae89"))
                self.viewer.fit()
        if self.pending:
            self.start_preview()

    def export(self):
        if self.frame is None or self.frame_generation != self.generation:
            self.statusBar().showMessage("Wait for a valid current preview before exporting", 8000)
            return
        frame = self.frame  # Keep the chosen image stable across the modal dialog.
        path, _ = QFileDialog.getSaveFileName(self, "Export (8-bit sRGB, straight alpha)", "output.png", "PNG (*.png)")
        if not path:
            return
        try:
            write_png(path, frame)
            self.statusBar().showMessage(f"Exported {path} · viewer exposure/channel controls are display-only", 10000)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Export failed", str(error))

    def closeEvent(self, event):
        if not self.confirm_discard():
            event.ignore()
            return
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
    parser.add_argument("--agent", metavar="LOCAL_NAME", help="opt-in user-local agent socket; no network listener")
    args = parser.parse_args()
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
    return app.exec()
