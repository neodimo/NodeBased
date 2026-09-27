"""Artist-facing tools for fluid and particle simulation: the slice viewer, the cache inspector
and the sim stats overlay (docs/SIMULATION.md, docs/FLUIDS_SPIKE.md).

Each widget is a thin Qt shell over a pure module (`slice3d`, `cacheinspector`, `simstats`) so the
actual rendering, stats and invalidation logic is tested without a widget in the loop; the widget
tests only check that the shell wires the numbers onto the screen and the buttons onto the calls.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QHeaderView, QLabel,
                                QMessageBox, QPushButton, QSizePolicy, QSlider, QSpinBox,
                                QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import cacheinspector, slice3d


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                 QImage.Format.Format_RGB888).copy()


class SliceView(QWidget):
    """An axis-aligned slice through a fluid `Volume`, with a colour ramp, an optional velocity
    arrow overlay and a numeric readout under the cursor. `set_volume(None)` clears it."""

    valueUnderCursor = Signal(object)     # float, or None off the plane

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume = None
        self.axis = "z"
        self.index = 0
        self.field = "density"
        self.show_arrows = False
        self._plane = None
        self._image = QImage()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        header = QHBoxLayout()
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        self.axis_combo.setCurrentText(self.axis)
        self.axis_combo.currentTextChanged.connect(self.set_axis)
        header.addWidget(QLabel("Axis"))
        header.addWidget(self.axis_combo)
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self._on_field_changed)
        header.addWidget(QLabel("Field"))
        header.addWidget(self.field_combo)
        self.index_slider = QSlider(Qt.Orientation.Horizontal)
        self.index_slider.valueChanged.connect(self.set_index)
        header.addWidget(self.index_slider, 1)
        self.arrows_button = QPushButton("Velocity arrows")
        self.arrows_button.setCheckable(True)
        self.arrows_button.toggled.connect(self.set_show_arrows)
        header.addWidget(self.arrows_button)
        layout.addLayout(header)
        self.image_label = QLabel()
        self.image_label.setObjectName("slice-image")
        self.image_label.setMouseTracking(True)
        self.image_label.setMinimumSize(64, 64)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.mouseMoveEvent = self._image_mouse_move    # small widget: no subclass needed
        layout.addWidget(self.image_label, 1)
        self.readout_label = QLabel(" ")
        self.readout_label.setObjectName("slice-readout")
        layout.addWidget(self.readout_label)
        self.setMouseTracking(True)

    # -- state -----------------------------------------------------------------------------------
    def set_volume(self, volume):
        self.volume = volume
        if volume is None:
            self.field_combo.clear()
            self._plane, self._image = None, QImage()
            self.image_label.clear()
            return
        fields = slice3d.available_fields(volume)
        if self.field not in fields:
            self.field = fields[0]
        block = self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(fields)
        self.field_combo.setCurrentText(self.field)
        self.field_combo.blockSignals(False)
        count = slice3d.slice_index_count(volume, self.axis)
        self.index_slider.blockSignals(True)
        self.index_slider.setRange(0, max(0, count - 1))
        self.index_slider.setValue(slice3d.clamp_index(volume, self.axis, self.index))
        self.index_slider.blockSignals(False)
        self.index = self.index_slider.value()
        self._rebuild()

    def set_axis(self, axis: str):
        self.axis = axis
        if self.volume is not None:
            count = slice3d.slice_index_count(self.volume, self.axis)
            self.index_slider.setRange(0, max(0, count - 1))
        self._rebuild()

    def set_index(self, index: int):
        self.index = int(index)
        self._rebuild()

    def _on_field_changed(self, name: str):
        if name:
            self.field = name
            self._rebuild()

    def set_show_arrows(self, on: bool):
        self.show_arrows = bool(on)
        self._rebuild()

    # -- rendering ---------------------------------------------------------------------------------
    def _rebuild(self):
        if self.volume is None or not self.field:
            return
        self._plane = slice3d.slice_plane(self.volume, self.axis, self.index, self.field)
        # Normalised against the whole field's range, not just this plane's: the ramp stays put
        # while scrubbing, so a colour change on screen means the values changed, not the range.
        whole = slice3d.scalar_field(self.volume, self.field)
        rgb = slice3d.colorize(self._plane, vmin=float(np.min(whole)), vmax=float(np.max(whole)))
        self._image = _rgb_to_qimage(rgb)
        if self.show_arrows:
            arrows = slice3d.in_plane_velocity(self.volume, self.axis, self.index)
            if arrows is not None:
                self._draw_arrows(self._image, arrows)
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation)
        self.image_label.setPixmap(pixmap)

    @staticmethod
    def _draw_arrows(image, arrows):
        painter = QPainter(image)
        painter.setPen(Qt.GlobalColor.white)
        step_y = image.height() / max(1, arrows.shape[0])
        step_x = image.width() / max(1, arrows.shape[1])
        for i in range(arrows.shape[0]):
            for j in range(arrows.shape[1]):
                x, y = (j + 0.5) * step_x, (i + 0.5) * step_y
                dx, dy = float(arrows[i, j, 0]), float(arrows[i, j, 1])
                painter.drawLine(int(x), int(y), int(x + dx * 4), int(y + dy * 4))
        painter.end()

    def _image_mouse_move(self, event):
        if self._plane is None or self.image_label.pixmap() is None or self.image_label.pixmap().isNull():
            self.valueUnderCursor.emit(None)
            return
        pixmap = self.image_label.pixmap()
        label_size = self.image_label.size()
        offset_x = max(0, (label_size.width() - pixmap.width()) // 2)
        offset_y = max(0, (label_size.height() - pixmap.height()) // 2)
        px = event.position().x() - offset_x
        py = event.position().y() - offset_y
        if not (0 <= px < pixmap.width() and 0 <= py < pixmap.height()):
            self.valueUnderCursor.emit(None)
            return
        rows, cols = self._plane.shape
        col = int(px / max(1, pixmap.width()) * cols)
        row = int(py / max(1, pixmap.height()) * rows)
        value = slice3d.sample_value(self.volume, self.axis, self.index, self.field, row, col)
        self.readout_label.setText(f"{self.field} = {value:.4f}  (row {row}, col {col})")
        self.valueUnderCursor.emit(value)

    def value_at(self, row: int, col: int) -> "float | None":
        """The current field's value at a plane (row, col), for tests that do not want to
        simulate a mouse move over the label."""
        if self._plane is None:
            return None
        return slice3d.sample_value(self.volume, self.axis, self.index, self.field, row, col)


class SimStatsOverlay(QLabel):
    """The "sim stats" text shown over the viewport while a simulation solves."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sim-stats-overlay")
        self.setStyleSheet("color: white; background: rgba(0, 0, 0, 120); padding: 4px;")
        self.hide()

    def update_snapshot(self, snapshot):
        self.setText(snapshot.text())
        self.show()

    def clear_snapshot(self):
        self.setText("")
        self.hide()


class CacheInspectorPanel(QWidget):
    """Frame list, per-frame stats and cache maintenance for one FluidCache3D/ParticleCache3D
    node: `set_cache(cache, run, start_frame, end_frame)` populates it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cache = None
        self.run = None
        self.start_frame = 1
        self.end_frame = 1
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("Invalidate from frame"))
        self.invalidate_spin = QSpinBox()
        self.invalidate_spin.setRange(-(2 ** 30), 2 ** 30)
        header.addWidget(self.invalidate_spin)
        self.invalidate_button = QPushButton("Invalidate")
        self.invalidate_button.clicked.connect(self._on_invalidate_clicked)
        header.addWidget(self.invalidate_button)
        self.resolve_button = QPushButton("Re-solve range")
        self.resolve_button.clicked.connect(self._on_resolve_clicked)
        header.addWidget(self.resolve_button)
        self.open_folder_button = QPushButton("Open cache folder")
        self.open_folder_button.clicked.connect(self._on_open_folder_clicked)
        header.addWidget(self.open_folder_button)
        layout.addLayout(header)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Frame", "Status", "Voxels/Particles", "Active tiles", "Solve ms", "Disk", "Summary"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        # `resolve_callback(start, end)`, set by whoever owns the actual solve loop (the app);
        # left `None` in tests that only check invalidation and listing.
        self.resolve_callback = None

    def set_cache(self, cache, run: str, start_frame: int, end_frame: int):
        self.cache, self.run = cache, run
        self.start_frame, self.end_frame = int(start_frame), int(end_frame)
        self.invalidate_spin.setRange(self.start_frame, max(self.start_frame, self.end_frame))
        self.invalidate_spin.setValue(self.start_frame)
        self.refresh()

    def refresh(self):
        self.table.setRowCount(0)
        if self.cache is None or self.run is None:
            return
        entries = cacheinspector.list_frames(self.cache, self.run, self.start_frame, self.end_frame)
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            self.table.setItem(row, 0, QTableWidgetItem(str(entry.frame)))
            status = "cached" if entry.present else "missing"
            self.table.setItem(row, 1, QTableWidgetItem(status))
            if entry.stats is None:
                for col in range(2, 6):
                    self.table.setItem(row, col, QTableWidgetItem("-"))
                self.table.setItem(row, 6, QTableWidgetItem(""))
                continue
            stats = entry.stats
            count = stats["particle_count"] if stats["particle_count"] is not None else stats["voxel_count"]
            self.table.setItem(row, 2, QTableWidgetItem("-" if count is None else str(count)))
            tiles = stats["active_tiles"]
            self.table.setItem(row, 3, QTableWidgetItem("-" if tiles is None else str(tiles)))
            ms = entry.solve_ms
            self.table.setItem(row, 4, QTableWidgetItem("-" if ms is None else f"{ms:.2f}"))
            disk = entry.disk_bytes
            self.table.setItem(row, 5, QTableWidgetItem("-" if disk is None else f"{disk / 1024:.1f} KB"))
            summary = ", ".join(f"{name}: {vals['min']:.3g}..{vals['max']:.3g}"
                                for name, vals in list(stats["fields"].items())[:2])
            self.table.setItem(row, 6, QTableWidgetItem(summary))

    def _on_invalidate_clicked(self):
        if self.cache is None or self.run is None:
            return
        cacheinspector.invalidate_from_frame(self.cache, self.run, self.invalidate_spin.value())
        self.refresh()

    def _on_resolve_clicked(self):
        if self.resolve_callback is not None:
            self.resolve_callback(self.start_frame, self.end_frame)
        self.refresh()

    def _on_open_folder_clicked(self):
        if self.cache is None or self.run is None:
            return
        folder = cacheinspector.cache_folder(self.cache, self.run)
        if folder is None:
            QMessageBox.information(self, "Cache folder", "This cache has no disk tier.")
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
