"""Interactive editor viewport for NodeBased's 3D nodes.

Navigation is local UI state. It never writes Camera3D parameters, so orbiting while inspecting
a shot cannot silently change the authored Render3D camera. Pixels come from the interactive wgpu
renderer (viewportgpu) when an adapter is available: meshes are uploaded once and orbiting only
moves a camera matrix. Without an adapter the widget falls back to the CPU reference renderer
(scene3d), which is correct but slow. Either way a fixed inspection headlight shades the scene
until it has lights of its own.

Gaussian splats are shown as a proxy for placing things, never as the Render3D look: discs on the
GPU (see viewportgpu), a depth-tested 2 x 2 mark per splat centre in the CPU fallback.
"""
from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

from . import scene3d, viewportgpu
from .core import GEOMETRY_TYPES

# Textures are evaluated at this proxy tier: the viewport is for placing things, and a quarter
# resolution plate is plenty to see where a card sits without stalling the UI on a 4K Read.
TEXTURE_TIER = 4
DRAG_SCALE = 0.5  # CPU fallback only: half size while the mouse is down, full size on release
CPU_SPLATS = 200_000  # CPU fallback only: splat centres drawn per cloud
HOME = (35.0, 20.0, 7.0)
BACKGROUND = (0.025, 0.025, 0.03, 1.0)


def _line_vertices(segments):
    """(N*2, 7) float32 line-list vertices from (start, end, rgba) segments.

    Editor colours are display-referred; the GPU target encodes sRGB, so they are decoded here.
    """
    if not segments:
        return np.zeros((0, 7), np.float32)
    out = np.empty((len(segments) * 2, 7), np.float32)
    for index, (start, end, color) in enumerate(segments):
        out[index * 2, :3], out[index * 2 + 1, :3] = start, end
        out[index * 2:index * 2 + 2, 3:] = color
    rgb = out[:, 3:6]
    out[:, 3:6] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return out


_GRID = _line_vertices([(start, end, color if color[:3] != (0.25, 0.25, 0.25) else (*color[:3], 0.55))
                        for start, end, color in scene3d.grid_axes(0, 0, scale=1.5)])


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
        self._splat_points = {}  # CPU fallback: id(cloud) -> (cloud, proxy rows, stride)
        self.splat_note = ""
        self.backend = "auto"  # "cpu" forces the reference renderer (tests, troubleshooting)
        self._last_frame = None
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
        gpu = viewportgpu.renderer() if self.backend != "cpu" else None
        painter = QPainter(self)
        if gpu is not None and self._paint_gpu(painter, gpu, scene, camera, authored):
            backend = "GPU"
        else:
            self._paint_cpu(painter, scene, camera, authored)
            backend = "CPU reference"
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#e8d98d"), 1.5))
        for light in scene.lights:
            xy, z = scene3d.project(camera, self.width(), self.height(), light.world()[0][None])
            if z[0] > camera.near:
                painter.drawEllipse(QPointF(*xy[0]), 5, 5)
        painter.setPen(QColor("#d8d8df"))
        mode = "through camera (C to leave)" if self.look_through and authored is not None else \
            "orbit LMB · pan MMB · dolly wheel · F frame · C camera"
        painter.drawText(12, 22, f"3D VIEWPORT · {backend} · {mode}")
        if self.splat_note:
            painter.drawText(12, self.height() - 12, self.splat_note)
        if self.status:
            painter.setPen(QColor("#e06f6f"))
            painter.drawText(12, 42, self.status[:160])
        painter.end()

    def _editor_lines(self, scene, authored):
        segments = []
        if authored is not None and not self.look_through:
            segments += [(a, b, (0.553, 0.722, 0.91, 1.0)) for a, b in scene3d.frustum_lines(authored, 16 / 9)]
        for light in scene.lights:
            position, direction = light.world()
            segments.append((position, position + direction * 0.8, (0.91, 0.851, 0.553, 1.0)))
        return np.concatenate((_GRID, _line_vertices(segments))) if segments else _GRID

    def _paint_gpu(self, painter, gpu, scene, camera, authored):
        ratio = self.devicePixelRatioF()
        width, height = max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio))
        try:
            frame = gpu.render(scene, camera, width, height, BACKGROUND,
                               lines=self._editor_lines(scene, authored),
                               headlight=not scene.lights, ambient=0.15)
        except Exception as error:  # a driver fault must not take the editor down
            self.status = f"GPU viewport failed, using the CPU renderer: {error}"
            self.backend = "cpu"
            return False
        if frame is not None:  # None: a Render3D job holds the device, keep showing the last frame
            self.splat_note = self._splat_note(scene, gpu.splat_stride, "discs")
            self._last_frame = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                                      QImage.Format.Format_RGBA8888).copy()
        if self._last_frame is None:
            return False
        painter.drawImage(self.rect(), self._last_frame)
        return True

    def _paint_cpu(self, painter, scene, camera, authored):
        scale = DRAG_SCALE if self._drag else 1.0
        width, height = max(1, int(self.width() * scale)), max(1, int(self.height() * scale))
        # The reference splat rasterizer takes seconds to minutes per frame and refuses large
        # captures outright, so the fallback renders the meshes and marks splat centres instead.
        splats, scene = scene.splats, scene3d.Scene(scene.geometries, scene.lights)
        try:
            # The interactive viewport stays on the rasterizer and does not show shadows yet.
            image, depth = scene3d.render(scene, camera, width, height, BACKGROUND,
                                          shade=not scene.lights, ambient=0.15, return_depth=True, shadows=False, mode="raster")
        except ValueError as error:
            self.status = str(error)
            image, depth = scene3d.render(scene3d.Scene(), camera, width, height,
                                          BACKGROUND, return_depth=True, shadows=False, mode="raster")
        rgb = np.clip(image[..., :3] / np.maximum(image[..., 3:4], 1e-6), 0, 1)
        depth = depth.copy() if splats else depth  # the renderer's buffer is read-only
        self.splat_note = self._splat_note(scene3d.Scene(splats=splats),
                                           self._mark_splats(splats, camera, rgb, depth), "points")
        rgba = np.concatenate((np.sqrt(rgb) * 255, np.full((*rgb.shape[:2], 1), 255)), axis=2).astype(np.uint8)
        qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
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

    def _mark_splats(self, splats, camera, rgb, depth):
        """Write each splat centre into ``rgb``/``depth`` where it is nearer than what is there.
        Returns the largest stride used."""
        height, width = depth.shape
        used, largest = set(), 1
        for instance in splats:
            if not len(instance.cloud):
                continue
            key = id(instance.cloud)
            used.add(key)
            if key not in self._splat_points:
                self._splat_points[key] = (instance.cloud, *viewportgpu.splat_proxy(instance.cloud, CPU_SPLATS))
            _cloud, rows, stride = self._splat_points[key]
            largest = max(largest, stride)
            rows = rows[rows[:, 7] * instance.opacity_scale >= viewportgpu.SPLAT_MIN_OPACITY]
            matrix = np.asarray(instance.matrix, np.float32)
            xy, z = scene3d.project(camera, width, height, rows[:, :3] @ matrix[:3, :3].T + matrix[:3, 3])
            ix, iy = xy[:, 0].astype(int), xy[:, 1].astype(int)
            seen = (z > camera.near) & (z < camera.far) & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) \
                & (ix < width) & (iy < height)
            order = np.flatnonzero(seen)
            order = order[np.argsort(-z[order], kind="stable")]  # far first: the nearest is written last
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):  # 2 x 2 marks survive display scaling
                y, x = np.minimum(iy[order] + dy, height - 1), np.minimum(ix[order] + dx, width - 1)
                front = z[order] < depth[y, x]
                rgb[y[front], x[front]] = np.clip(rows[order[front], 4:7], 0, 1)
                depth[y[front], x[front]] = z[order[front]]
        for key in [k for k in self._splat_points if k not in used]:
            del self._splat_points[key]
        return largest

    @staticmethod
    def _splat_note(scene, stride, shape):
        total = sum(len(instance.cloud) for instance in scene.splats)
        if not total:
            return ""
        shown = f"1 in {stride} of " if stride > 1 else ""
        return f"splats: {shown}{total:,} shown as {shape} (layout proxy, not the render)"

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
        for instance in scene.splats:
            if len(instance.cloud):  # captures carry stray far splats: frame the bulk of the cloud
                matrix = np.asarray(instance.matrix, np.float64)
                centres = instance.cloud.positions[::max(1, len(instance.cloud) // 100_000)]
                low, high = np.percentile(centres, (2, 98), axis=0)
                corners = np.array([(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1])
                                    for z in (low[2], high[2])])
                points.append(corners @ matrix[:3, :3].T + matrix[:3, 3])
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
