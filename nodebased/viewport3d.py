"""Interactive editor viewport for NodeBased's 3D nodes.

Navigation is local UI state. It never writes Camera3D parameters, so orbiting while inspecting
a shot cannot silently change the authored Render3D camera. Pixels come from the same CPU
reference renderer as Render3D (scene3d), with a fixed inspection headlight until the scene has
lights of its own; this widget is not a GPU performance claim.
"""
from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

from . import scene3d
from .core import GEOMETRY_TYPES

# Textures are evaluated at this proxy tier: the viewport is for placing things, and a quarter
# resolution plate is plenty to see where a card sits without stalling the UI on a 4K Read.
TEXTURE_TIER = 4
DRAG_SCALE = 0.5  # render at half size while the mouse is down, full size when it is released
HOME = (35.0, 20.0, 7.0)


class Viewport3D(QWidget):
    def __init__(self, window=None, parent=None):
        super().__init__(parent)
        self.window = window
        self.document = None
        self.azimuth, self.elevation, self.distance = HOME
        self.center = np.zeros(3, np.float32)  # orbit pivot, world space
        self.look_through = False
        self.status = ""
        self._drag = None
        self._evaluator = None
        self._scene_cache = (None, None)
        self.setMinimumSize(320, 220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_document(self, document):
        self.document = document
        self._scene_cache = (None, None)  # documents are edited in place; identity proves nothing
        if self.isVisible():
            self.update()

    # --- what to show ---------------------------------------------------------------------------

    def _render_node(self):
        """The Render3D the artist is working on: the one upstream of the viewed node, else the
        first in the document. None when the document has no Render3D yet."""
        nodes = (self.document or {}).get("nodes", {})
        stack, seen = [self.document.get("view")] if self.document else [], set()
        while stack:
            key = stack.pop()
            if key is None or key in seen or key not in nodes:
                continue
            seen.add(key)
            if nodes[key]["type"] == "Render3D":
                return nodes[key]
            stack.extend(nodes[key]["inputs"].values())
        return next((n for n in nodes.values() if n["type"] == "Render3D"), None)

    def _frame(self):
        return int((self.document or {}).get("time", {}).get("current", 1))

    def _evaluated(self):
        """(scene, authored camera) through the real evaluator, so textures, animation,
        expressions, nested scenes and disabled nodes all look the way Render3D will see them.
        Falls back to the loose geometry in the document while nothing is wired up yet."""
        document = self.document or {}
        identity = (self._frame(), document.get("view"))
        if self._scene_cache[0] == identity:
            return self._scene_cache[1]
        nodes = document.get("nodes", {})
        render = self._render_node()
        scene = camera = None
        self.status = ""
        if self._evaluator is None:
            from .imaging import Evaluator
            self._evaluator = Evaluator(cache_bytes=256 << 20)
        for slot in ("scene", "camera"):
            source = render["inputs"].get(slot) if render else None
            if source is None:
                continue
            try:
                value = self._evaluator.evaluate_raster(document, source, frame=self._frame(),
                                                        tier=TEXTURE_TIER, typed=True)
            except Exception as error:  # a broken upstream must not take the editor down
                self.status = str(error)
                continue
            scene, camera = (value, camera) if slot == "scene" else (scene, value)
        if scene is None:
            loose = [n for n in nodes.values() if not n["disabled"]]
            try:
                scene = scene3d.Scene(
                    tuple(scene3d.geometry_from_node(n) for n in loose if n["type"] in GEOMETRY_TYPES),
                    tuple(scene3d.light_from_node(n) for n in loose if n["type"] == "Light3D"))
            except ValueError as error:
                self.status, scene = str(error), scene3d.Scene()
        if camera is None:
            authored = next((n for n in nodes.values() if n["type"] == "Camera3D"), None)
            camera = scene3d.camera_from_node(authored) if authored else None
        self._scene_cache = (identity, (scene, camera))
        return scene, camera

    def _camera(self, authored=None):
        if self.look_through and authored is not None:
            return authored
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        position = self.center + np.array((math.sin(az) * math.cos(el), math.sin(el),
                                           math.cos(az) * math.cos(el)), np.float32) * self.distance
        return scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(*position)), scene3d.Vec3(*self.center),
                              45.0, max(0.01, self.distance * 0.01), max(1000.0, self.distance * 100))

    # --- painting -------------------------------------------------------------------------------

    def paintEvent(self, event):
        scene, authored = self._evaluated()
        camera = self._camera(authored)
        scale = DRAG_SCALE if self._drag else 1.0
        width, height = max(1, int(self.width() * scale)), max(1, int(self.height() * scale))
        try:
            # The interactive viewport does not show shadows yet.
            image, depth = scene3d.render(scene, camera, width, height, (0.025, 0.025, 0.03, 1.0),
                                          shade=not scene.lights, ambient=0.15, return_depth=True, shadows=False)
        except ValueError as error:
            self.status = str(error)
            image, depth = scene3d.render(scene3d.Scene(), camera, width, height,
                                          (0.025, 0.025, 0.03, 1.0), return_depth=True, shadows=False)
        rgb = np.clip(image[..., :3] / np.maximum(image[..., 3:4], 1e-6), 0, 1)
        rgba = np.concatenate((np.sqrt(rgb) * 255, np.full((*rgb.shape[:2], 1), 255)), axis=2).astype(np.uint8)
        qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.rect(), qimage)
        # Editor chrome lives in world space: projected through the same camera and hidden behind
        # rendered geometry. It is never scene data and never reaches Render3D.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(1 / scale, 1 / scale)
        for start, end, color in scene3d.grid_axes(width, height, scale=1.5):
            is_axis = color[:3] != (0.25, 0.25, 0.25)
            painter.setPen(QPen(QColor.fromRgbF(*color[:3], 1.0 if is_axis else 0.55), 2 if is_axis else 1))
            for a, b in self._visible_segments(camera, depth, start, end):
                painter.drawLine(a, b)
        if authored is not None and not self.look_through:
            painter.setPen(QPen(QColor("#8db8e8"), 1.5))
            for start, end in scene3d.frustum_lines(authored, 16 / 9):
                for a, b in self._visible_segments(camera, depth, start, end, samples=24):
                    painter.drawLine(a, b)
        painter.setPen(QPen(QColor("#e8d98d"), 1.5))
        for light in scene.lights:
            position, direction = light.world()
            for a, b in self._visible_segments(camera, depth, position, position + direction * 0.8, samples=12):
                painter.drawLine(a, b)
            xy, z = scene3d.project(camera, width, height, position[None])
            if z[0] > camera.near:
                painter.drawEllipse(QPointF(*xy[0]), 5, 5)
        painter.resetTransform()
        painter.setPen(QColor("#d8d8df"))
        mode = "through camera (C to leave)" if self.look_through and authored is not None else \
            "orbit LMB · pan MMB · dolly wheel · F frame · C camera"
        painter.drawText(12, 22, f"3D VIEWPORT · CPU reference · {mode}")
        if self.status:
            painter.setPen(QColor("#e06f6f"))
            painter.drawText(12, 42, self.status[:160])
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

    # --- navigation -----------------------------------------------------------------------------

    def frame_scene(self):
        """Home the orbit on the scene's bounds (or the origin when it is empty)."""
        scene, _camera = self._evaluated()
        points = [(g.world_matrix()[:3, :3] @ g.vertices.T + g.world_matrix()[:3, 3:4]).T
                  for g in scene.geometries]
        self.azimuth, self.elevation, self.distance = HOME
        self.center = np.zeros(3, np.float32)
        if points:
            cloud = np.concatenate(points)
            low, high = cloud.min(axis=0), cloud.max(axis=0)
            self.center = ((low + high) / 2).astype(np.float32)
            radius = max(float(np.linalg.norm(high - low)) / 2, 1e-3)
            self.distance = radius / math.sin(math.radians(45.0) / 2) * 1.1
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = (event.button(), event.position())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag and not self.look_through:
            button, previous = self._drag
            delta = event.position() - previous
            self._drag = (button, event.position())
            if button == Qt.MouseButton.LeftButton:
                self.azimuth -= float(delta.x()) * 0.5
                self.elevation = max(-89.0, min(89.0, self.elevation + float(delta.y()) * 0.5))
            else:
                # Pan in the view plane, scaled so the pivot tracks the cursor.
                _eye, view = scene3d._view_basis(self._camera())
                per_pixel = 2 * self.distance * math.tan(math.radians(45.0) / 2) / max(self.height(), 1)
                self.center = self.center + (-view[0] * float(delta.x()) + view[1] * float(delta.y())) * per_pixel
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.update()  # repaint at full resolution

    def wheelEvent(self, event):
        if not self.look_through:
            self.distance = max(0.01, min(100000.0, self.distance * (0.9 if event.angleDelta().y() > 0 else 1.1)))
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F:
            self.look_through = False
            self.frame_scene()
        elif event.key() == Qt.Key.Key_C:
            self.look_through = not self.look_through
            self.update()
        else:
            super().keyPressEvent(event)
