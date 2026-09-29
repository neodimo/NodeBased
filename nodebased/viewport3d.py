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
import time
from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt, QPointF, QTimer, QSettings
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QToolTip, QMenu, QLabel

from . import handles3d, progressiverender, scene3d, viewportgpu, particleinspect
from .core import GEOMETRY_TYPES
from .imaging import linear_to_srgb

# The progressive "Render" mode's own idle-repaint cadence (R6 "next", closed): fast enough to look
# responsive while converging, slow enough to leave the event loop room for mouse/keyboard input.
RENDER_MODE_TICK_MS = 16

# Textures are evaluated at this proxy tier: the viewport is for placing things, and a quarter
# resolution plate is plenty to see where a card sits without stalling the UI on a 4K Read.
TEXTURE_TIER = 4
DRAG_SCALE = 0.5  # CPU fallback only: half size while the mouse is down, full size on release
CPU_SPLATS = 200_000  # CPU fallback only: splat centres drawn per cloud
HOME = (35.0, 20.0, 7.0)
BACKGROUND = (0.025, 0.025, 0.03, 1.0)
SELECT_COLOR = (0.98, 0.7, 0.15, 1.0)
PICK_DRAG_THRESHOLD = 3  # pixels of motion before a left-button press is a navigation drag, not a click


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
        self._press_pos = None
        self._dragged = False
        self.selected_key = None
        self.show_sim_stats = str(QSettings("NodeBased", "NodeBased").value(
            "viewport3d/show_sim_stats", False)).lower() in ("true", "1")
        self._sim_stats_identity = None
        self._sim_stats_text = ""
        self.sim_stats_draw_ms = 0.0
        self.sim_stats_overlay_ms = 0.0
        self._sim_stats_label = QLabel(self)
        self._sim_stats_label.setObjectName("viewport-sim-stats")
        self._sim_stats_label.setStyleSheet("color: #ecf3fa; background: rgba(10, 10, 12, 190); padding: 5px;")
        self._sim_stats_label.move(6, 68)
        self._sim_stats_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._sim_stats_label.hide()
        self._sim_stats_timer = QTimer(self)
        self._sim_stats_timer.setSingleShot(True)
        self._sim_stats_timer.timeout.connect(self._update_sim_stats_label)
        self.pivot_mode = False  # Q toggles: the gizmo moves pivot_x/y/z instead of tx/ty/tz
        self.gizmo_mode = "translate"  # W/E/R: translate / rotate / scale gizmo
        self._gizmo_drag = None
        self._evaluator = None
        self._scene_cache = (None, None)
        self._splat_points = {}  # CPU fallback: id(cloud) -> (cloud, proxy rows, stride)
        self.splat_note = ""
        self.volume_note = ""  # volumes: what the frame shows (GPU steps and the V toggle, or why nothing)
        self.backend = "auto"  # "cpu" forces the reference renderer (tests, troubleshooting)
        self._last_frame = None
        self.render_mode = False   # P: the progressive path-traced "Render" preview (progressiverender)
        self.render_note = ""      # what the progressive mode shows: sample count, low-res, converging
        self._progressive_state = None
        self._progressive_timer = QTimer(self)
        self._progressive_timer.setSingleShot(True)
        self._progressive_timer.timeout.connect(self.update)
        self.setMinimumSize(320, 220)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_document(self, document):
        self.document = document
        self._sim_stats_identity = None
        self.refresh_sim_stats()
        self._scene_cache = (None, None)  # documents are edited in place; identity proves nothing
        if self.selected_key is not None and self.selected_key not in (document or {}).get("nodes", {}):
            self.selected_key = None
            self._gizmo_drag = None
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

    def _pick_candidates(self):
        """[(node_key, Geometry)] for picking: the same wired-vs-loose source `_evaluated`
        renders from, but attributed back to the node that produced each shape (see
        `handles3d` for which node types that covers)."""
        document = self.document or {}
        render = self._render_node()
        source = render["inputs"].get("scene") if render else None
        if source is not None:
            return handles3d.resolve_geometries(document, source)
        return handles3d.loose_geometries(document)

    def _marker_candidates(self):
        """[(node_key, world_position, parent_matrix)] for picking and handles: the marker
        analogue of `_pick_candidates`, attributed back to the Light3D/Camera3D node the way
        `handles3d.resolve_geometries` attributes geometry (`scene.lights` from `_evaluated`
        is flattened and carries no node key back to draw from). Camera3D is scanned
        separately from the scene-graph walk that applies to Light3D; see the module note in
        `handles3d` above `resolve_light_markers`."""
        document = self.document or {}
        render = self._render_node()
        source = render["inputs"].get("scene") if render else None
        lights = (handles3d.resolve_light_markers(document, source) if source is not None
                 else handles3d.loose_light_markers(document))
        return lights + handles3d.camera_markers(document)

    def _authored_camera_key(self):
        render = self._render_node()
        return render["inputs"].get("camera") if render else None

    def _select(self, key):
        self.selected_key = key
        if self.window is not None and getattr(self.window, "graph", None) is not None:
            graph = self.window.graph
            item = graph.items_by_id.get(key) if key is not None else None
            graph.scene().clearSelection()
            if item is not None:
                item.setSelected(True)
        self.update()

    def _selected_bounds(self):
        if self.selected_key is None:
            return None
        for key, geometry in self._pick_candidates():
            if key == self.selected_key:
                bounds = handles3d.world_bounds(geometry)
                dragging_this = self._gizmo_drag is not None and self._gizmo_drag["key"] == key
                shift = self._gizmo_world_shift() if dragging_this else None
                if bounds is not None and shift is not None:
                    low, high = bounds
                    bounds = (low + shift, high + shift)
                return bounds
        return None

    def _gizmo_info(self, key):
        """(pivot world position, parent matrix, node params, gizmo scale) for ``key``, or
        None when it is not (or no longer) a pick or marker candidate."""
        node = (self.document or {}).get("nodes", {}).get(key)
        if node is None:
            return None
        for candidate_key, geometry in self._pick_candidates():
            if candidate_key == key:
                parent = np.asarray(geometry.parent, np.float64)
                scale = handles3d.gizmo_scale(handles3d.world_bounds(geometry))
                origin = handles3d.pivot_world_position(node["params"], parent)
                return origin, parent, node["params"], scale
        for candidate_key, position, parent in self._marker_candidates():
            if candidate_key == key:
                return position, parent, node["params"], handles3d.gizmo_scale(None)
        return None

    def _target_gizmo_info(self, key):
        """(target world position, parent matrix, node params, gizmo scale) for a Light3D or
        Camera3D's aim-target handle, or None when ``key`` is not a marker candidate."""
        node = (self.document or {}).get("nodes", {}).get(key)
        if node is None or node["type"] not in handles3d.MARKER_TYPES:
            return None
        for candidate_key, _position, parent in self._marker_candidates():
            if candidate_key == key:
                origin = handles3d.target_world_position(node["params"], parent)
                return origin, parent, node["params"], handles3d.gizmo_scale(None)
        return None

    def _marker_live_params(self, key, params):
        """``params`` with tx/ty/tz or target_x/y/z overridden by the in-progress drag on
        ``key``, if any -- the viewport-only preview a marker drag shows before it commits."""
        drag = self._gizmo_drag
        if drag is None or drag["key"] != key:
            return params
        values = self._gizmo_values(drag)
        return {**params, **values} if values is not None else params

    def _gizmo_hit(self, position):
        if self.selected_key is None or self.look_through:
            return None
        node = (self.document or {}).get("nodes", {}).get(self.selected_key)
        camera = self._camera()
        width, height = self.width(), self.height()
        xy = (position.x(), position.y())
        if node is not None and node["type"] in handles3d.MARKER_TYPES:
            # Light3D/Camera3D have no rx/ry/rz or scale knob at all (see the handles3d module
            # note), so W/E/R gizmo_mode never applies to them: only the position handle and
            # the target handle, tested here in that order.
            target_info = self._target_gizmo_info(self.selected_key)
            if target_info is not None:
                t_origin, _parent, _params, t_scale = target_info
                hit = handles3d.gizmo_hit(camera, width, height, t_origin, t_scale, xy)
                if hit is not None:
                    kind, part = hit
                    return (f"target_{kind}", part)
            info = self._gizmo_info(self.selected_key)
            if info is None:
                return None
            origin, _parent, _params, scale = info
            return handles3d.gizmo_hit(camera, width, height, origin, scale, xy)
        info = self._gizmo_info(self.selected_key)
        if info is None:
            return None
        origin, _parent, _params, scale = info
        if self.gizmo_mode == "rotate":
            name = handles3d.gizmo_ring_hit(camera, width, height, origin, scale, xy)
            return ("ring", name) if name else None
        if self.gizmo_mode == "scale":
            name = handles3d.gizmo_scale_hit(camera, width, height, origin, scale, xy)
            return ("cube", name) if name else None
        return handles3d.gizmo_hit(camera, width, height, origin, scale, xy)

    def _begin_gizmo_drag(self, hit, position):
        kind, part = hit
        is_target = kind.startswith("target_")
        node = (self.document or {}).get("nodes", {}).get(self.selected_key)
        is_marker = node is not None and node["type"] in handles3d.MARKER_TYPES
        info = self._target_gizmo_info(self.selected_key) if is_target else self._gizmo_info(self.selected_key)
        if info is None:
            return
        _origin, parent, params, _scale = info
        axis_kind = kind[len("target_"):] if is_target else kind
        # Light3D/Camera3D params have no rx/ry/rz or scale knob (see handles3d.MARKER_TYPES),
        # so scene3d._transform_from(params) would KeyError on them: own_linear is only ever
        # read back for the pivot-mode delta below, which markers never enter.
        own_linear = np.eye(3)
        if is_target:
            touched, mode = ("target_x", "target_y", "target_z"), "target"
        elif kind == "ring":
            touched, mode = (f"r{part}",), "rotate"
            own_linear = scene3d._transform_from(params).matrix()[:3, :3].astype(np.float64)
        elif kind == "cube":
            touched, mode = (("uscale",) if part == "center" else (f"s{part}",)), "scale"
            own_linear = scene3d._transform_from(params).matrix()[:3, :3].astype(np.float64)
        elif is_marker:
            touched, mode = ("tx", "ty", "tz"), "translate"  # markers have no pivot mode
        else:
            touched = ("pivot_x", "pivot_y", "pivot_z", "tx", "ty", "tz") if self.pivot_mode else ("tx", "ty", "tz")
            mode = "pivot" if self.pivot_mode else "translate"
            own_linear = scene3d._transform_from(params).matrix()[:3, :3].astype(np.float64)
        self._gizmo_drag = {
            "key": self.selected_key, "kind": axis_kind, "part": part, "mode": mode,
            "parent_linear": parent[:3, :3], "own_linear": own_linear,
            "start": (position.x(), position.y()), "current": (position.x(), position.y()),
            "original": {p: params[p] for p in touched},
        }
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _gizmo_values(self, drag=None):
        """Live parameter values for the in-progress drag (tx/ty/tz for translate, pivot_x/y/z +
        tx/ty/tz for pivot mode, one of rx/ry/rz for a ring, one of sx/sy/sz/uscale for a cube),
        or None once the node has vanished from underneath it."""
        drag = self._gizmo_drag if drag is None else drag
        if drag is None:
            return None
        is_target = drag["mode"] == "target"
        info = self._target_gizmo_info(drag["key"]) if is_target else self._gizmo_info(drag["key"])
        if info is None:
            return None
        origin, _parent, _params, _scale = info
        camera = self._camera()
        width, height = self.width(), self.height()
        original = drag["original"]
        if drag["kind"] == "ring":
            axis = handles3d.AXIS_VECS[drag["part"]]
            basis = handles3d.RING_AXES[drag["part"]]
            delta = handles3d.ring_drag_angle(camera, width, height, origin, axis, basis,
                                              drag["start"], drag["current"])
            param = f"r{drag['part']}"
            return {param: original[param] + delta}
        if drag["kind"] == "cube":
            param = "uscale" if drag["part"] == "center" else f"s{drag['part']}"
            origin_xy, z = scene3d.project(camera, width, height, np.asarray(origin, np.float64)[None])
            if z[0] <= camera.near:  # the pivot fell behind the camera: freeze rather than divide by garbage
                return {param: original[param]}
            value = handles3d.scale_from_drag(original[param], origin_xy[0], drag["start"], drag["current"])
            return {param: value}
        if drag["kind"] == "axis":
            axis = handles3d.AXIS_VECS[drag["part"]]
            start_pt = handles3d.axis_drag_point(camera, width, height, origin, axis, drag["start"])
            current_pt = handles3d.axis_drag_point(camera, width, height, origin, axis, drag["current"])
        else:
            normal = handles3d.PLANE_NORMALS[drag["part"]]
            start_pt = handles3d.plane_drag_point(camera, width, height, origin, normal, drag["start"])
            current_pt = handles3d.plane_drag_point(camera, width, height, origin, normal, drag["current"])
        world_delta = current_pt - start_pt
        local_delta = handles3d.world_to_local_delta(world_delta, drag["parent_linear"])
        if drag["mode"] == "pivot":
            pivot_delta, position_delta = handles3d.pivot_param_deltas(local_delta, drag["own_linear"])
            return {"pivot_x": original["pivot_x"] + pivot_delta[0], "pivot_y": original["pivot_y"] + pivot_delta[1],
                   "pivot_z": original["pivot_z"] + pivot_delta[2],
                   "tx": original["tx"] + position_delta[0], "ty": original["ty"] + position_delta[1],
                   "tz": original["tz"] + position_delta[2]}
        if is_target:
            return {"target_x": original["target_x"] + local_delta[0],
                   "target_y": original["target_y"] + local_delta[1],
                   "target_z": original["target_z"] + local_delta[2]}
        return {"tx": original["tx"] + local_delta[0], "ty": original["ty"] + local_delta[1],
               "tz": original["tz"] + local_delta[2]}

    def _gizmo_commands(self, key, values):
        frame = int((self.document or {}).get("time", {}).get("current", 1))
        curves = (self.document or {}).get("animation", {}).get("curves", {}).get(key, {})
        commands = []
        for param, value in values.items():
            if curves.get(param) is not None:
                commands.append({"op": "set_key", "id": key, "param": param, "frame": frame, "value": float(value)})
            else:
                commands.append({"op": "set", "id": key, "param": param, "value": float(value)})
        return commands

    def _end_gizmo_drag(self, commit):
        drag = self._gizmo_drag
        self._gizmo_drag = None
        self.unsetCursor()
        if drag is None:
            return
        moved = math.hypot(drag["current"][0] - drag["start"][0],
                           drag["current"][1] - drag["start"][1]) > PICK_DRAG_THRESHOLD
        if commit and moved and self.window is not None:
            values = self._gizmo_values(drag)
            if values is not None:
                self.window.command({"op": "batch", "commands": self._gizmo_commands(drag["key"], values)})

    def _gizmo_world_shift(self):
        """The world-space translation the live drag implies, or None (no drag, a rotate/scale
        drag, or a pivot-mode or target-handle drag -- none of which move the object by a plain
        translation)."""
        drag = self._gizmo_drag
        if drag is None or drag["kind"] not in ("axis", "plane") or drag["mode"] != "translate":
            return None
        values = self._gizmo_values(drag)
        if values is None:
            return None
        original = drag["original"]
        local_delta = np.array((values["tx"] - original["tx"], values["ty"] - original["ty"],
                                values["tz"] - original["tz"]), np.float64)
        return drag["parent_linear"] @ local_delta

    def _dragged_scene(self, scene):
        """``scene`` with the dragged object shifted, turned or scaled to match the live gizmo
        drag, without touching the document or re-running the evaluator (see the module
        docstring's "no full graph re-evaluation on a mouse move" rule). A pivot-mode drag never
        moves the object on screen by construction, so it leaves ``scene`` untouched.

        A translate/plane drag shifts the world position directly (adding the world-space delta
        to the candidate's parent matrix) rather than reconstructing a ``Transform3D``, since a
        `TransformGeo3D`-sourced candidate's ``.transform`` field does not describe its own node
        (see `handles3d`'s attribution notes) but its parent-relative world position still does --
        pure translation is representation-agnostic that way.

        A rotate/scale drag is not: it replaces the candidate's ``.transform`` with one rebuilt
        from the live parameter values, which is only correct when that field genuinely is the
        selected node's own transform (an ordinary geometry leaf, not a `TransformGeo3D`, whose
        baked vertices carry no such field). A `TransformGeo3D` selection still commits correctly
        on release; it just does not visibly turn or grow until then.
        """
        drag = self._gizmo_drag
        if drag is None:
            return scene
        if drag["kind"] in ("axis", "plane"):
            world_shift = self._gizmo_world_shift()
            if world_shift is None:
                return scene
            geometries = []
            for key, geometry in self._pick_candidates():
                if key == drag["key"]:
                    parent = np.array(geometry.parent, np.float64, copy=True)
                    parent[:3, 3] += world_shift
                    geometry = replace(geometry, parent=parent)
                geometries.append(geometry)
            return replace(scene, geometries=tuple(geometries))
        if drag["kind"] in ("ring", "cube"):
            node = (self.document or {}).get("nodes", {}).get(drag["key"])
            if node is None or node["type"] not in GEOMETRY_TYPES:
                return scene
            values = self._gizmo_values(drag)
            if values is None:
                return scene
            params = {**node["params"], **values}
            geometries = []
            for key, geometry in self._pick_candidates():
                if key == drag["key"]:
                    geometry = replace(geometry, transform=scene3d._transform_from(params))
                geometries.append(geometry)
            return replace(scene, geometries=tuple(geometries))
        return scene

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
        draw_started = time.perf_counter() if self.show_sim_stats else None
        scene, authored = self._evaluated()
        scene = self._dragged_scene(scene)
        camera = self._camera(authored)
        painter = QPainter(self)
        if self.render_mode and self._paint_progressive(painter, scene, camera, authored):
            backend = "Render (progressive)"
        else:
            gpu = viewportgpu.renderer() if self.backend != "cpu" else None
            if gpu is not None and self._paint_gpu(painter, gpu, scene, camera, authored):
                backend = "GPU"
            else:
                self._paint_cpu(painter, scene, camera, authored)
                backend = "CPU reference"
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_markers(painter, camera)
        self._draw_selection(painter, camera)
        self._draw_gizmo(painter, camera)
        painter.setPen(QColor("#d8d8df"))
        mode = "through camera (C to leave)" if self.look_through and authored is not None else \
            f"orbit LMB · pan MMB · dolly wheel · F frame · C camera · W/E/R gizmo [{self.gizmo_mode}] · " \
            "Q pivot mode · B background · P render"
        if self.pivot_mode and not self.look_through:
            painter.setPen(QColor("#f4ce63"))
            painter.drawText(12, 62, "PIVOT MODE")
            painter.setPen(QColor("#d8d8df"))
        if self.splat_note:  # splat discs are bright and busy: back the text so it stays readable
            painter.fillRect(0, 0, self.width(), 30, QColor(10, 10, 12, 170))
            painter.fillRect(0, self.height() - 28, self.width(), 28, QColor(10, 10, 12, 170))
            painter.drawText(12, self.height() - 10, self.splat_note)
        if self.volume_note:
            painter.fillRect(0, self.height() - 28 - (28 if self.splat_note else 0), self.width(), 28,
                             QColor(10, 10, 12, 170))
            painter.drawText(12, self.height() - 10 - (28 if self.splat_note else 0), self.volume_note)
        if self.render_note:
            painter.fillRect(0, self.height() - 28, self.width(), 28, QColor(10, 10, 12, 170))
            painter.drawText(12, self.height() - 10, self.render_note)
        painter.drawText(12, 22, f"3D VIEWPORT · {backend} · {mode}")
        if self.status:
            painter.setPen(QColor("#e06f6f"))
            painter.drawText(12, 42, self.status[:160])
        if self.show_sim_stats:
            overlay_started = time.perf_counter()
            # Cache inspection runs on a separate Qt turn. A changing simulation frame never
            # makes the viewport's draw path wait for a disk load or a voxel count.
            self.sim_stats_draw_ms = (time.perf_counter() - draw_started) * 1000.0
            self.refresh_sim_stats()
            self.sim_stats_overlay_ms = (time.perf_counter() - overlay_started) * 1000.0
        painter.end()

    def set_show_sim_stats(self, enabled):
        self.show_sim_stats = bool(enabled)
        QSettings("NodeBased", "NodeBased").setValue("viewport3d/show_sim_stats", self.show_sim_stats)
        self._sim_stats_identity = None
        if not self.show_sim_stats:
            self._sim_stats_timer.stop()
            self._sim_stats_label.hide()
        else:
            self.refresh_sim_stats()
        self.update()

    def refresh_sim_stats(self):
        if not self.show_sim_stats or self._sim_stats_timer.isActive():
            return
        key = self._selected_fluid_key()
        identity = (key, self._frame(), id(self.document), id(self._evaluator))
        if identity != self._sim_stats_identity:
            self._sim_stats_timer.start(0)

    def _update_sim_stats_label(self):
        text = self._sim_stats()
        if not self.show_sim_stats or not text:
            self._sim_stats_identity = (self._selected_fluid_key(), self._frame(),
                                        id(self.document), id(self._evaluator))
            self._sim_stats_label.hide()
            return
        draw_ms = self.sim_stats_draw_ms
        self._sim_stats_label.setText(text.format(draw=draw_ms))
        self._sim_stats_label.adjustSize()
        self._sim_stats_label.show()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        action = menu.addAction("Show simulation stats")
        action.setCheckable(True)
        action.setChecked(self.show_sim_stats)
        action.toggled.connect(self.set_show_sim_stats)
        menu.exec(event.globalPos())

    def _selected_fluid_key(self):
        key = self.window.graph.selected_id() if self.window is not None and hasattr(self.window, "graph") else self.selected_key
        node = (self.document or {}).get("nodes", {}).get(key)
        return key if node and node["type"] in ("FluidSolver3D", "FluidCache3D") else None

    def _sim_stats(self):
        """Sample the selected node's solved cache once per frame; never invoked while hidden."""
        key = self._selected_fluid_key()
        if key is None or self._evaluator is None:
            return ""
        frame = self._frame()
        identity = (key, frame, id(self.document), id(self._evaluator))
        if identity == self._sim_stats_identity:
            return self._sim_stats_text
        node = self.document["nodes"][key]
        try:
            from . import cachecontext, fluid_gpu_solver
            volume = cachecontext.volume_for_node(self._evaluator, self.document, key, frame)
            stream = getattr(volume, "stream", None)
            if stream is None:
                return ""
            if node["type"] == "FluidCache3D":
                context = cachecontext.cache_context(self._evaluator, self.document, key, frame)
                if context is None:
                    return ""
                store, run = context[:2]
            else:
                store, run = self._evaluator._sim_memory, stream.run
            state = store.get(run, frame)
            if state is None:
                return ""
            density = state.arrays.get("density")
            active = int(np.count_nonzero(density)) if density is not None else 0
            memory = state.nbytes / (1024 * 1024)
            backend = stream.backend or "cpu"
            adapter = fluid_gpu_solver.adapter_name() if backend != "cpu" else "CPU"
            # FluidCache3D stores display channels under a derived run; solve timing stays
            # with the underlying solver checkpoint.
            solve_ms = store.solve_ms(stream.run, frame)
            solve_text = f"{solve_ms:.2f} ms" if solve_ms is not None else "—"
            self._sim_stats_text = (f"{key} · {stream.shape[0]}×{stream.shape[1]}×{stream.shape[2]} grid · "
                                    f"{stream.voxel:g} voxel · {active:,} active\nSolve {solve_text}")
            self._sim_stats_text += (f" · Draw {{draw:.2f}} ms · Memory {memory:.2f} MB · "
                                     f"{len(store.frames(run))} cached frames\n{backend} · {adapter}")
            self._sim_stats_identity = identity
            return self._sim_stats_text
        except (ValueError, RuntimeError, KeyError, AttributeError):
            return ""

    def _draw_markers(self, painter, camera):
        """Camera3D and Light3D markers, attributed to their node key (`_marker_candidates`,
        unlike the flattened `scene.lights` the base render draws from) so they can be picked
        and show a selection highlight. The selected marker's own position/target tracks an
        in-progress drag on it before the drag commits (no document write, no re-evaluation).
        A camera the viewport is currently looking through draws no marker for itself: its own
        lens is the view, so a dot and aim line for it would just sit at the edge of frame."""
        width, height = self.width(), self.height()
        nodes = (self.document or {}).get("nodes", {})
        looked_through_key = self._authored_camera_key() if self.look_through else None
        for key, _position, parent in self._marker_candidates():
            if key == looked_through_key:
                continue
            node = nodes.get(key)
            if node is None:
                continue
            params = self._marker_live_params(key, node["params"])
            position = handles3d.pivot_world_position(params, parent)
            target = handles3d.target_world_position(params, parent)
            selected = key == self.selected_key
            is_camera = node["type"] == "Camera3D"
            color = SELECT_COLOR if selected else \
                ((0.553, 0.722, 0.91, 1.0) if is_camera else (0.91, 0.851, 0.553, 1.0))
            pen_color = QColor(*(round(c * 255) for c in color[:3]))
            xy, z = scene3d.project(camera, width, height, np.array((position, target)))
            if z[0] > camera.near and z[1] > camera.near:
                painter.setPen(QPen(pen_color, 2 if selected else 1.5))
                painter.drawLine(QPointF(*xy[0]), QPointF(*xy[1]))
            if z[0] > camera.near:
                painter.setPen(QPen(pen_color, 2))
                radius = 7 if selected else 5
                painter.drawEllipse(QPointF(*xy[0]), radius, radius)

    def _draw_selection(self, painter, camera):
        """Draw the selected object's world-space bounds box, straight over the finished
        frame -- a UI overlay, not scene data, the same way the light markers above are drawn
        without a depth test against the rendered geometry."""
        bounds = self._selected_bounds()
        if bounds is None:
            return
        painter.setPen(QPen(QColor(*(round(c * 255) for c in SELECT_COLOR[:3])), 2))
        for start, end in handles3d.bounds_edges(bounds):
            xy, z = scene3d.project(camera, self.width(), self.height(), np.array((start, end)))
            if z[0] > camera.near and z[1] > camera.near:
                painter.drawLine(QPointF(*xy[0]), QPointF(*xy[1]))

    _GIZMO_AXIS_COLORS = {"x": QColor(224, 90, 90), "y": QColor(120, 200, 110), "z": QColor(94, 150, 226)}
    _GIZMO_PLANE_COLORS = {"xy": QColor(94, 150, 226, 90), "yz": QColor(224, 90, 90, 90), "xz": QColor(120, 200, 110, 90)}
    _GIZMO_UNIFORM_COLOR = QColor(224, 206, 99)

    def _draw_gizmo(self, painter, camera):
        """The gizmo for the selected object's current mode (translate/rotate/scale; or the
        pivot gizmo, in pivot mode, which reuses the translate arrows), plus a numeric readout
        while a drag is in progress. A UI overlay, drawn the same way `_draw_selection` is.

        A selected Light3D/Camera3D draws two translate-style gizmos instead: a position
        handle at the node's own tx/ty/tz, and a target handle at target_x/y/z (see
        `handles3d.MARKER_TYPES` -- neither node type has a rotate or scale knob, so
        gizmo_mode never applies to them)."""
        if self.selected_key is None or self.look_through:
            return
        node = (self.document or {}).get("nodes", {}).get(self.selected_key)
        width, height = self.width(), self.height()
        drag_values = self._gizmo_values() if self._gizmo_drag is not None else None
        if node is not None and node["type"] in handles3d.MARKER_TYPES:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            info = self._gizmo_info(self.selected_key)
            if info is not None:
                origin, _parent, _params, scale = info
                self._draw_translate_arrows(painter, camera, width, height, origin, scale, mode="translate")
            target_info = self._target_gizmo_info(self.selected_key)
            if target_info is not None:
                t_origin, t_parent, _params, t_scale = target_info
                if drag_values is not None and self._gizmo_drag["mode"] == "target":
                    t_origin = handles3d.target_world_position({**node["params"], **drag_values}, t_parent)
                self._draw_translate_arrows(painter, camera, width, height, t_origin, t_scale, mode="target")
            painter.restore()
            if drag_values is not None:
                self._draw_drag_label(painter, drag_values)
            return
        info = self._gizmo_info(self.selected_key)
        if info is None:
            return
        origin, _parent, _params, scale = info
        if drag_values is not None and self._gizmo_drag["mode"] == "pivot":
            origin = origin + (drag_values["pivot_x"] - self._gizmo_drag["original"]["pivot_x"],
                               drag_values["pivot_y"] - self._gizmo_drag["original"]["pivot_y"],
                               drag_values["pivot_z"] - self._gizmo_drag["original"]["pivot_z"])
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.gizmo_mode == "rotate":
            self._draw_rotate_rings(painter, camera, width, height, origin, scale)
        elif self.gizmo_mode == "scale":
            self._draw_scale_cubes(painter, camera, width, height, origin, scale)
        else:
            self._draw_translate_arrows(painter, camera, width, height, origin, scale)
        painter.restore()
        if drag_values is not None:
            self._draw_drag_label(painter, drag_values)

    def _draw_drag_label(self, painter, values):
        painter.save()
        painter.setPen(QColor("#f4ce63"))
        cx, cy = self._gizmo_drag["current"]
        painter.drawText(cx + 14, cy - 14, self._gizmo_label(values))
        painter.restore()

    def _draw_translate_arrows(self, painter, camera, width, height, origin, scale, mode=None):
        """``mode``, when given, restricts the active-part highlight to a drag of that mode --
        needed once a marker's position and target handles (both drawn with this same method,
        at different origins) can be on screen at once, so a shared axis letter (e.g. "x")
        does not light up both simultaneously."""
        for name, corners in handles3d.gizmo_planes(origin, scale).items():
            xy, z = scene3d.project(camera, width, height, np.array(corners))
            if np.all(z > camera.near):
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self._GIZMO_PLANE_COLORS[name])
                painter.drawPolygon(QPolygonF([QPointF(*point) for point in xy]))
        for name, (start, end) in handles3d.gizmo_arrows(origin, scale).items():
            xy, z = scene3d.project(camera, width, height, np.array((start, end)))
            if z[0] > camera.near and z[1] > camera.near:
                active = (self._gizmo_drag is not None and self._gizmo_drag["part"] == name
                         and (mode is None or self._gizmo_drag["mode"] == mode))
                color = QColor(self._GIZMO_AXIS_COLORS[name])
                if active:
                    color = color.lighter(140)
                painter.setPen(QPen(color, 3 if active else 2.5))
                painter.drawLine(QPointF(*xy[0]), QPointF(*xy[1]))

    def _draw_rotate_rings(self, painter, camera, width, height, origin, scale):
        for name, points in handles3d.gizmo_rings(origin, scale).items():
            xy, z = scene3d.project(camera, width, height, np.array(points))
            if not np.all(z > camera.near):
                continue  # a ring that dips behind the near plane anywhere is skipped whole
            active = self._gizmo_drag is not None and self._gizmo_drag["part"] == name
            color = QColor(self._GIZMO_AXIS_COLORS[name])
            if active:
                color = color.lighter(140)
            painter.setPen(QPen(color, 3 if active else 2.5))
            polygon = QPolygonF([QPointF(*point) for point in xy])
            polygon.append(QPointF(*xy[0]))
            painter.drawPolyline(polygon)

    def _draw_scale_cubes(self, painter, camera, width, height, origin, scale):
        for name, center in handles3d.gizmo_cubes(origin, scale).items():
            xy, z = scene3d.project(camera, width, height, np.asarray(center, np.float64)[None])
            if z[0] <= camera.near:
                continue
            active = self._gizmo_drag is not None and self._gizmo_drag["part"] == name
            color = QColor(self._GIZMO_AXIS_COLORS.get(name, self._GIZMO_UNIFORM_COLOR))
            if active:
                color = color.lighter(140)
            half = 7.0 if active else 5.5
            painter.setPen(QPen(color.darker(130), 1.5))
            painter.setBrush(color)
            painter.drawRect(xy[0, 0] - half, xy[0, 1] - half, half * 2, half * 2)

    @staticmethod
    def _gizmo_label(values):
        if "pivot_x" in values:
            return f"pivot {values['pivot_x']:.3f}, {values['pivot_y']:.3f}, {values['pivot_z']:.3f}"
        if "tx" in values:
            return f"tx {values['tx']:.3f}  ty {values['ty']:.3f}  tz {values['tz']:.3f}"
        if "target_x" in values:
            return f"target {values['target_x']:.3f}, {values['target_y']:.3f}, {values['target_z']:.3f}"
        param, value = next(iter(values.items()))
        return f"{param} {value:.4f}"

    def _editor_lines(self, scene, authored):
        segments = []
        if authored is not None and not self.look_through:
            segments += [(a, b, (0.553, 0.722, 0.91, 1.0)) for a, b in scene3d.frustum_lines(authored, 16 / 9)]
        for light in scene.lights:
            position, direction = light.world()
            segments.append((position, position + direction * 0.8, (0.91, 0.851, 0.553, 1.0)))
        return np.concatenate((_GRID, _line_vertices(segments))) if segments else _GRID

    def _progressive_key(self, camera):
        """Whatever should restart the progressive accumulation (`progressiverender.step`'s `key`):
        the document/frame/view identity, the camera's own numbers (orbiting never edits the
        document, so `_evaluated`'s cache key alone would miss it) and any in-progress gizmo drag
        (which also changes what `_dragged_scene` shows)."""
        document = self.document or {}
        identity = (self._frame(), document.get("view"), self.look_through)
        position, target = camera.transform.position, camera.target
        pose = (round(position.x, 4), round(position.y, 4), round(position.z, 4),
               round(target.x, 4), round(target.y, 4), round(target.z, 4), round(camera.fov, 3))
        drag = (self._gizmo_drag["key"], self._gizmo_drag["current"]) if self._gizmo_drag is not None else None
        return identity, pose, drag

    def _paint_progressive(self, painter, scene, camera, authored):
        ratio = self.devicePixelRatioF()
        width, height = max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio))
        key = self._progressive_key(camera)
        try:
            self._progressive_state = progressiverender.step(
                self._progressive_state, scene, camera, width, height, BACKGROUND, 0.15, key)
        except Exception as error:  # a scene the path tracer refuses (e.g. particles slipped through)
            self.status = f"Render mode failed, using the interactive viewport: {error}"
            self.render_mode = False
            self._progressive_timer.stop()
            return False
        state = self._progressive_state
        if state.image is None:
            return False
        self.splat_note = self.volume_note = ""   # the progressive image already carries their look
        rgb = np.clip(state.image[..., :3] / np.maximum(state.image[..., 3:4], 1e-6), 0, 1)
        rgba = np.concatenate((linear_to_srgb(rgb) * 255, np.full((*rgb.shape[:2], 1), 255)), axis=2)
        rgba = np.ascontiguousarray(rgba.astype(np.uint8))
        qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                        QImage.Format.Format_RGBA8888).copy()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.rect(), qimage)
        done = progressiverender.converged(state, width, height)
        label = f"{state.samples} sample{'s' if state.samples != 1 else ''}"
        if state.low_res:
            label += " · low-res"
        self.render_note = f"RENDER · {label}" + ("" if done else " · converging")
        if done:
            self._progressive_timer.stop()
        else:
            self._progressive_timer.start(RENDER_MODE_TICK_MS)
        return True

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
            self.volume_note = self._volume_note(scene, gpu)
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
        had_volumes = bool(scene.volumes)
        splats, scene = scene.splats, scene3d.Scene(scene.geometries, scene.lights, particles=scene.particles)
        self.volume_note = "volumes need the GPU viewport (Render3D still renders them)" if had_volumes else ""
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
    def _volume_note(scene, gpu):
        if not scene.volumes:
            return ""
        if gpu.volume_note:
            return gpu.volume_note
        quality = "quality" if gpu.volume_quality else "fast"
        return f"volumes: {quality}, {gpu.volume_steps} steps across (V toggles quality)"

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
            if len(instance.cloud):
                # Captures wrap their subject in a far shell of sky and haze splats (on the Nelson
                # capture the 2nd..98th percentile box is 16x the size of the 10th..90th), so frame
                # the middle of the cloud, widened by a quarter to take in the subject's edges.
                matrix = np.asarray(instance.matrix, np.float64)
                centres = instance.cloud.positions[::max(1, len(instance.cloud) // 100_000)]
                low, high = np.percentile(centres, (10, 90), axis=0)
                low, high = low - (high - low) * 0.125, high + (high - low) * 0.125
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
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._gizmo_hit(event.position())
            if hit is not None:
                self._begin_gizmo_drag(hit, event.position())
                event.accept()
                return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = (event.button(), event.position())
            self._press_pos = event.position()
            self._dragged = False
            event.accept()

    def mouseMoveEvent(self, event):
        if self._gizmo_drag is not None:
            self._gizmo_drag["current"] = (event.position().x(), event.position().y())
            self.update()
            return
        if self._drag:
            if (event.position() - self._press_pos).manhattanLength() > PICK_DRAG_THRESHOLD:
                self._dragged = True
            if not self.look_through:
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
        if event.button() == Qt.MouseButton.LeftButton and self._gizmo_drag is not None:
            self._end_gizmo_drag(commit=True)
            self.update()
            return
        # A click (no drag) picks; a drag was orbit/pan and must not also pick on release.
        if (event.button() == Qt.MouseButton.LeftButton and self._drag is not None
                and not self._dragged and not self.look_through):
            self._pick(event.position())
        self._drag = None
        self.update()  # repaint at full resolution

    def _pick(self, position):
        camera = self._camera()
        width, height = self.width(), self.height()
        scene, _authored = self._evaluated()
        for instance in scene.particles:
            matrix = np.asarray(instance.matrix, np.float64)
            world = instance.positions @ matrix[:3, :3].T + matrix[:3, 3]
            index = particleinspect.pick_particle(camera, width, height, world,
                                                  position.x(), position.y())
            if index is not None:
                data = particleinspect.particle_readout(instance, index)
                text = (f"Particle {data['id']}\nAge: {data['age']:.3f}\n"
                        f"Velocity: ({data['velocity'][0]:.3f}, {data['velocity'][1]:.3f}, "
                        f"{data['velocity'][2]:.3f})\nSpeed: {data['speed']:.3f}\n"
                        f"Size: {data['size']:.3f}\nCustom attributes: none")
                QToolTip.showText(self.mapToGlobal(position.toPoint()), text, self)
                return
        # Markers are a small screen-space target sitting on top of the rendered frame (see
        # _draw_markers), not scene geometry with a world-space bounds box: test them first.
        marker_hit = handles3d.pick_marker(self._marker_candidates(), camera, width, height,
                                           position.x(), position.y())
        if marker_hit is not None:
            self._select(marker_hit)
            return
        hit = handles3d.pick(self._pick_candidates(), camera, width, height,
                             position.x(), position.y())
        self._select(hit[0] if hit is not None else None)

    def wheelEvent(self, event):
        if not self.look_through:
            self.distance = max(0.01, min(100000.0, self.distance * (0.9 if event.angleDelta().y() > 0 else 1.1)))
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._gizmo_drag is not None:
            self._end_gizmo_drag(commit=False)
            self.update()
        elif event.key() == Qt.Key.Key_W and not event.modifiers():
            self.gizmo_mode = "translate"
            self.update()
        elif event.key() == Qt.Key.Key_E and not event.modifiers():
            self.gizmo_mode = "rotate"
            self.update()
        elif event.key() == Qt.Key.Key_R and not event.modifiers():
            self.gizmo_mode = "scale"
            self.update()
        elif event.key() == Qt.Key.Key_V and not event.modifiers():
            gpu = viewportgpu.renderer()
            if gpu is not None:
                gpu.volume_quality = not gpu.volume_quality
                self.update()
        elif event.key() == Qt.Key.Key_Q and not event.modifiers():
            self.pivot_mode = not self.pivot_mode
            self.update()
        elif event.key() == Qt.Key.Key_B and not event.modifiers():
            gpu = viewportgpu.renderer()
            if gpu is not None:
                gpu.show_background = not gpu.show_background
                self.update()
        elif event.key() == Qt.Key.Key_P and not event.modifiers():
            self.render_mode = not self.render_mode
            if not self.render_mode:
                self._progressive_timer.stop()
                self.render_note = ""
            self._progressive_state = None   # always start the next Render pass fresh
            self.update()
        elif event.key() == Qt.Key.Key_F:
            self.look_through = False
            self.frame_scene()
        elif event.key() == Qt.Key.Key_C:
            self.look_through = not self.look_through
            self.update()
        else:
            super().keyPressEvent(event)
