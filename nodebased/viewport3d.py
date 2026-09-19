"""Interactive editor viewport for the bounded 3D foundation.

Navigation is local UI state.  It never writes Camera3D parameters, so orbiting while inspecting
a shot cannot silently change the authored Render3D camera.  The renderer is the same CPU/reference
scene3d path used by exports; this widget is not a GPU performance claim.
"""
from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

from . import scene3d


class Viewport3D(QWidget):
    def __init__(self, window=None, parent=None):
        super().__init__(parent)
        self.window = window
        self.document = None
        self.azimuth, self.elevation, self.distance = 35.0, 20.0, 7.0
        self.pan_x = self.pan_y = 0.0
        self._drag = None
        self.setMinimumSize(320, 220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_document(self, document):
        self.document = document
        self.update()

    def _camera(self):
        nodes = (self.document or {}).get("nodes", {})
        camera = next((n for n in nodes.values() if n["type"] == "Camera3D"), None)
        if camera is None:
            return scene3d.Camera()
        authored = scene3d.camera_from_node(camera)
        # Editor orbit/pan/dolly is a temporary inspection camera around the authored target.
        target = authored.target.array() + np.array((self.pan_x, self.pan_y, 0), np.float32)
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        position = target + np.array((math.sin(az) * math.cos(el), math.sin(el),
                                      math.cos(az) * math.cos(el)), np.float32) * self.distance
        return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(*position), authored.transform.rotation),
                              scene3d.Vec3(*target), authored.fov, authored.near, authored.far)

    def _scene(self):
        nodes = (self.document or {}).get("nodes", {})
        geometry = tuple(scene3d.geometry_from_node(n) for n in nodes.values()
                         if n["type"] in ("Card3D", "Cube3D"))
        return scene3d.Scene(geometry)

    def paintEvent(self, event):
        camera = self._camera()
        width, height = max(1, self.width()), max(1, self.height())
        image, depth = scene3d.render(self._scene(), camera, width, height,
                                      (0.025, 0.025, 0.03, 1.0), shade=True, return_depth=True)
        rgb = np.clip(image[..., :3] / np.maximum(image[..., 3:4], 1e-6), 0, 1)
        rgba = np.concatenate((np.sqrt(rgb) * 255, np.full((*rgb.shape[:2], 1), 255)), axis=2).astype(np.uint8)
        qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        painter = QPainter(self)
        painter.drawImage(0, 0, qimage)
        # Editor grid and axes are navigation chrome, never scene data, but they live in world
        # space: project them through the same camera and hide them behind rendered geometry.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for start, end, color in scene3d.grid_axes(width, height, scale=1.5):
            is_axis = color[:3] != (0.25, 0.25, 0.25)
            pen = QPen(QColor.fromRgbF(*color[:3], 1.0 if is_axis else 0.55), 2 if is_axis else 1)
            painter.setPen(pen)
            for a, b in self._visible_segments(camera, depth, start, end):
                painter.drawLine(a, b)
        painter.setPen(QColor("#d8d8df"))
        painter.drawText(12, 22, "3D VIEWPORT · CPU reference · orbit LMB · pan MMB · dolly wheel · F frame")
        painter.end()

    @staticmethod
    def _visible_segments(camera, depth, start, end, samples=96):
        """Split a world-space line into screen segments that are in front of the near plane
        and not occluded by the depth buffer."""
        t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
        points = np.asarray(start, np.float32) * (1 - t) + np.asarray(end, np.float32) * t
        height, width = depth.shape
        xy, z = scene3d.project(camera, width, height, points)
        ix = np.clip(xy[:, 0].astype(int), 0, width - 1)
        iy = np.clip(xy[:, 1].astype(int), 0, height - 1)
        onscreen = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        occluded = onscreen & (depth[iy, ix] < z * 0.999 - 1e-3)
        keep = (z > camera.near) & (z < camera.far) & ~occluded
        return [(QPointF(*xy[i]), QPointF(*xy[i + 1])) for i in range(samples - 1)
                if keep[i] and keep[i + 1]]

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = (event.button(), event.position())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag:
            button, previous = self._drag
            delta = event.position() - previous
            self._drag = (button, event.position())
            if button == Qt.MouseButton.LeftButton:
                self.azimuth += float(delta.x()) * 0.5
                self.elevation = max(-89.0, min(89.0, self.elevation + float(delta.y()) * 0.5))
            else:
                self.pan_x -= float(delta.x()) * self.distance / max(self.width(), 1)
                self.pan_y += float(delta.y()) * self.distance / max(self.height(), 1)
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        self.distance = max(0.1, min(10000.0, self.distance * (0.9 if event.angleDelta().y() > 0 else 1.1)))
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F:
            self.azimuth, self.elevation, self.distance = 35.0, 20.0, 7.0
            self.pan_x = self.pan_y = 0.0
            self.update()
        else:
            super().keyPressEvent(event)
