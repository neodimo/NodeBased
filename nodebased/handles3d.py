"""3D viewport picking: screen-to-ray projection and a nearest-object hit test.

Object identity for picking is resolved by walking the *document graph* directly (Scene3D
grouping, Axis3D parenting, TransformGeo3D baking), not through the flattened ``Scene`` the
evaluator returns: a flattened ``Scene``'s ``Geometry`` entries carry no reference back to the
node that produced them, so there is no way to recover "which node was clicked" from the
render path alone. Only the node types that ``scene3d.geometry_from_node`` understands
(``Card3D``, ``Cube3D``, ``Sphere3D``, ``ReadGeo3D``) plus ``TransformGeo3D`` are attributed
and therefore pickable this way. Nodes that assemble a scene from an external file
(``ReadAlembic3D``, ``ReadUSD3D``, ``ReadGLTF3D``, ``ReadSplat3D``) can hold any number of
inner meshes with no per-mesh document node to select, so they are walked over (their own
transform still parents whatever is nested beneath a ``Scene3D``/``Axis3D`` that contains
one) but are not themselves pick targets: a click on one of them currently finds nothing.

The CPU path is the only one implemented: the interactive wgpu viewport (``viewportgpu``)
draws no per-pixel object-id buffer, so picking always ray-tests world-space bounds here,
whichever backend painted the frame.
"""
from __future__ import annotations

import math

import numpy as np

from . import scene3d
from .core import GEOMETRY_TYPES

# TransformGeo3D is not in core.GEOMETRY_TYPES (it bakes an upstream geometry rather than
# producing one from its own params), but it is still a pickable, attributable node.
PICKABLE_LEAF_TYPES = (*GEOMETRY_TYPES, "TransformGeo3D")


def screen_to_ray(camera, width, height, x, y):
    """World-space (origin, unit direction) for the pixel (x, y), the inverse of
    ``scene3d.project`` for a single point at the camera's near/far range."""
    eye, view = scene3d._view_basis(camera)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    aspect = width / max(height, 1)
    ndc_x = (x / max(width, 1) - 0.5) * 2.0
    ndc_y = 1.0 - 2.0 * (y / max(height, 1))
    local = np.array((ndc_x * aspect / focal, ndc_y / focal, -1.0), np.float64)
    direction = view.T.astype(np.float64) @ local
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction = direction / norm
    return eye.astype(np.float64), direction


def _resolve_single(nodes, key, seen):
    """Object-space ``Geometry`` for one geometry-producing node (no Scene3D/Axis3D parent
    applied), or None when the node is disabled, missing, or not a geometry source."""
    if key is None or key not in nodes or key in seen:
        return None
    node = nodes[key]
    kind = node["type"]
    seen = seen | {key}
    if kind in GEOMETRY_TYPES:
        if node["disabled"]:
            return None
        try:
            return scene3d.geometry_from_node(node)
        except ValueError:
            return None
    if kind == "TransformGeo3D":
        base = _resolve_single(nodes, node["inputs"].get("geo"), seen)
        if base is None:
            return None
        if node["disabled"]:
            return base
        return scene3d.transform_geometry(base, scene3d._transform_from(node["params"]))
    return None


def resolve_geometries(document, root_key):
    """[(node_key, Geometry)] for every enabled, attributable geometry reachable from
    ``root_key`` through Scene3D grouping and Axis3D parenting, in world space
    (``geometry.world_matrix()`` already carries the accumulated parent)."""
    nodes = document.get("nodes", {})
    found = []

    def walk(key, parent, seen):
        if key is None or key not in nodes or key in seen:
            return
        node = nodes[key]
        kind = node["type"]
        seen = seen | {key}
        if kind == "Scene3D":
            if node["disabled"]:  # a disabled Scene3D evaluates to an empty scene, not a passthrough
                return
            own = scene3d._transform_from(node["params"]).matrix().astype(np.float64)
            matrix = parent @ own
            for index in range(8):
                walk(node["inputs"].get(f"object{index}"), matrix, seen)
        elif kind == "Axis3D":
            params = _IDENTITY_XFORM if node["disabled"] else node["params"]
            own = scene3d._transform_from(params).matrix().astype(np.float64)
            walk(node["inputs"].get("object"), parent @ own, seen)
        elif kind in PICKABLE_LEAF_TYPES:
            geometry = _resolve_single(nodes, key, set())
            if geometry is not None:
                found.append((key, replace_parent(geometry, parent)))
        # Other scene-producing node types (ReadAlembic3D, ReadUSD3D, ReadGLTF3D,
        # ReadSplat3D, Project3D) are not attributable to one inner mesh; stop here.

    walk(root_key, np.eye(4, dtype=np.float64), set())
    return found


def replace_parent(geometry, parent):
    return type(geometry)(**{**geometry.__dict__, "parent": parent})


_IDENTITY_XFORM = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0,
                   "sx": 1.0, "sy": 1.0, "sz": 1.0, "uscale": 1.0, "rot_order": "XYZ",
                   "pivot_x": 0.0, "pivot_y": 0.0, "pivot_z": 0.0}


def loose_geometries(document):
    """[(node_key, Geometry)] for every enabled top-level geometry-leaf node, at its own
    transform -- the same candidate set ``Viewport3D._evaluated`` renders before anything is
    wired into a ``Render3D``."""
    nodes = document.get("nodes", {})
    out = []
    for key, node in nodes.items():
        if not node["disabled"] and node["type"] in GEOMETRY_TYPES:
            try:
                out.append((key, scene3d.geometry_from_node(node)))
            except ValueError:
                continue
    return out


def world_bounds(geometry):
    """(min, max) float64 world-space AABB corners, or None for an empty mesh."""
    vertices = np.asarray(geometry.vertices, np.float64)
    if not len(vertices):
        return None
    matrix = np.asarray(geometry.world_matrix(), np.float64)
    corners = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return corners.min(axis=0), corners.max(axis=0)


def bounds_edges(bounds):
    """The 12 world-space line segments of an AABB's wireframe."""
    low, high = (np.asarray(v, np.float64) for v in bounds)
    corners = [(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    edges = []
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j > i:
                edges.append((corners[i], corners[j]))
    return edges


def _ray_aabb(origin, direction, bounds):
    """Nearest non-negative hit distance along the ray, or None."""
    low, high = (np.asarray(v, np.float64) for v in bounds)
    safe = np.where(direction == 0, 1e-12, direction)
    t1, t2 = (low - origin) / safe, (high - origin) / safe
    tmin = float(np.minimum(t1, t2).max())
    tmax = float(np.maximum(t1, t2).min())
    if tmax < max(tmin, 0.0):
        return None
    return max(tmin, 0.0)


def pick(candidates, camera, width, height, x, y):
    """(node_key, Geometry, distance) for the nearest candidate hit by the screen pixel
    (x, y), or None. ``candidates`` is a [(node_key, Geometry)] list such as
    ``resolve_geometries`` or ``loose_geometries`` returns."""
    origin, direction = screen_to_ray(camera, width, height, x, y)
    best = None
    for key, geometry in candidates:
        bounds = world_bounds(geometry)
        if bounds is None:
            continue
        t = _ray_aabb(origin, direction, bounds)
        if t is not None and (best is None or t < best[2]):
            best = (key, geometry, t)
    return best


# --- translate gizmo and pivot mode (step 3a) ---------------------------------------------------
#
# The gizmo is world-axis-aligned (not the object's own rotated axes): an arrow moves the object
# along world X/Y/Z, a plane square moves it in a world plane. Dragging maps screen motion onto
# that axis/plane by finding, for both the press point and the current point, the world point the
# camera ray through that pixel implies (closest point on the axis line for an arrow, the ray/plane
# intersection for a square) and taking the difference -- the same approach Maya/Blender gizmos use,
# so the object tracks the cursor exactly regardless of camera angle.
#
# Pivot mode moves the same gizmo but writes pivot_x/y/z instead of tx/ty/tz, compensating
# tx/ty/tz so the object does not move on screen. Derivation: a node's own local-space affine map
# is M(p) = linear @ (p - pivot) + pivot + position, where linear = R @ S (rotation and scale;
# uscale folded into S). This is affine in p with a p-independent linear part (linear itself, which
# does not depend on pivot) and a constant offset C = position + pivot - linear @ pivot. Moving the
# pivot by a local delta d while holding M(p) fixed for every p requires only that C stay fixed:
#   C = position + pivot - linear @ pivot = position' + (pivot + d) - linear @ (pivot + d)
#   => position' = position - d + linear @ d = position + (linear - I) @ d
# The world position of the pivot POINT itself (p = pivot) is M(pivot) = position + pivot: notably
# free of `linear`, since rotation/scale pivot around themselves. A parent chain (Scene3D/Axis3D)
# then applies uniformly on top and does not change either derivation.

X_AXIS = np.array((1.0, 0.0, 0.0))
Y_AXIS = np.array((0.0, 1.0, 0.0))
Z_AXIS = np.array((0.0, 0.0, 1.0))
AXIS_VECS = {"x": X_AXIS, "y": Y_AXIS, "z": Z_AXIS}
PLANE_NORMALS = {"xy": Z_AXIS, "yz": X_AXIS, "xz": Y_AXIS}
PLANE_AXES = {"xy": ("x", "y"), "yz": ("y", "z"), "xz": ("x", "z")}

ARROW_INNER_GAP_FACTOR = 0.15  # a dead zone around the pivot so a click there orbits, not drags
ARROW_LENGTH_FACTOR = 1.6
PLANE_OFFSET_FACTOR = 0.45
PLANE_SIZE_FACTOR = 0.28
MIN_GIZMO_SCALE = 0.25
GIZMO_HIT_PIXELS = 10.0


def pivot_world_position(params, parent_matrix):
    """World-space position of a node's pivot point: parent @ (position + pivot_local).

    Distinct from ``geometry.world_matrix()[:3, 3]``, which is the world position of the
    node's own local *origin* (0, 0, 0), not its pivot.
    """
    local = np.array((params["tx"] + params.get("pivot_x", 0.0),
                      params["ty"] + params.get("pivot_y", 0.0),
                      params["tz"] + params.get("pivot_z", 0.0)), np.float64)
    parent_matrix = np.asarray(parent_matrix, np.float64)
    return parent_matrix[:3, :3] @ local + parent_matrix[:3, 3]


def gizmo_scale(bounds):
    """A gizmo radius sized to the selected object's world-space bounds, or a sane default
    when there is no usable bounds (an empty mesh)."""
    if bounds is None:
        return 1.0
    low, high = (np.asarray(v, np.float64) for v in bounds)
    return max(float(np.linalg.norm(high - low)) * 0.5, MIN_GIZMO_SCALE)


def gizmo_arrows(origin, scale):
    """{'x': (start, tip), 'y': ..., 'z': ...}: world-space translate-arrow segments.

    Each segment starts ``ARROW_INNER_GAP_FACTOR * scale`` out from the pivot, not at it, so a
    click exactly on the pivot (where all three arrows would otherwise coincide, and which is
    also where the object itself is likely to be under the cursor) falls through to orbit or
    picking instead of an ambiguous axis choice. The drag math still uses the axis line through
    the true pivot (see ``axis_drag_point``); only the hit-test/paint segment is inset.
    """
    origin = np.asarray(origin, np.float64)
    inner, length = scale * ARROW_INNER_GAP_FACTOR, scale * ARROW_LENGTH_FACTOR
    return {name: (origin + axis * inner, origin + axis * length) for name, axis in AXIS_VECS.items()}


def gizmo_planes(origin, scale):
    """{'xy': [4 world-space corners], ...}: small plane-translate squares, offset from the
    pivot along both of the plane's axes so they sit clear of the pivot and the arrows."""
    origin = np.asarray(origin, np.float64)
    offset, size = scale * PLANE_OFFSET_FACTOR, scale * PLANE_SIZE_FACTOR
    out = {}
    for name, (a_name, b_name) in PLANE_AXES.items():
        a, b = AXIS_VECS[a_name], AXIS_VECS[b_name]
        base = origin + a * offset + b * offset
        out[name] = [base + a * dx * size + b * dy * size for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
    return out


def _closest_point_on_line(ray_origin, ray_dir, line_point, line_dir):
    """The point on the infinite line (line_point + t*line_dir) closest to the ray
    (ray_origin + s*ray_dir). Both directions are assumed unit length. Falls back to a plain
    projection onto the line when the ray runs parallel to it (the closest-point system is
    singular there)."""
    w0 = ray_origin - line_point
    b = float(np.dot(ray_dir, line_dir))
    d = float(np.dot(ray_dir, w0))
    e = float(np.dot(line_dir, w0))
    denom = 1.0 - b * b
    t = e if abs(denom) < 1e-9 else (e - b * d) / denom
    return line_point + t * line_dir


def axis_drag_point(camera, width, height, origin, axis, screen_xy):
    """World point on the axis line through ``origin`` closest to the camera ray through
    ``screen_xy`` -- the anchor an axis-arrow drag tracks as the mouse moves."""
    ray_origin, ray_dir = screen_to_ray(camera, width, height, screen_xy[0], screen_xy[1])
    return _closest_point_on_line(ray_origin, ray_dir, np.asarray(origin, np.float64),
                                  np.asarray(axis, np.float64))


def plane_drag_point(camera, width, height, origin, normal, screen_xy):
    """World point where the camera ray through ``screen_xy`` crosses the plane through
    ``origin`` with the given ``normal`` -- the anchor a plane-square drag tracks."""
    ray_origin, ray_dir = screen_to_ray(camera, width, height, screen_xy[0], screen_xy[1])
    origin, normal = np.asarray(origin, np.float64), np.asarray(normal, np.float64)
    denom = float(np.dot(ray_dir, normal))
    if abs(denom) < 1e-9:  # the ray runs parallel to the plane: no intersection, stay put
        return origin
    t = float(np.dot(origin - ray_origin, normal)) / denom
    return ray_origin + ray_dir * t


def world_to_local_delta(world_delta, parent_linear):
    """A world-space translation delta, expressed in the node's own parent-local frame (the
    frame its tx/ty/tz and pivot_x/y/z parameters live in)."""
    try:
        return np.linalg.solve(np.asarray(parent_linear, np.float64), np.asarray(world_delta, np.float64))
    except np.linalg.LinAlgError:
        return np.asarray(world_delta, np.float64)


def pivot_param_deltas(local_pivot_delta, own_linear):
    """(pivot delta, compensating translate delta): see the module-level derivation above.
    ``local_pivot_delta`` is already in the node's own parent-local frame (see
    ``world_to_local_delta``); ``own_linear`` is the node's own rotation-and-scale matrix
    (``Transform3D.matrix()[:3, :3]``) at the values the drag started from."""
    position_delta = (np.asarray(own_linear, np.float64) - np.eye(3)) @ local_pivot_delta
    return local_pivot_delta, position_delta


def _point_segment_distance(point, a, b):
    point, a, b = (np.asarray(v, np.float64) for v in (point, a, b))
    ab = b - a
    length2 = float(np.dot(ab, ab))
    t = 0.0 if length2 < 1e-9 else max(0.0, min(1.0, float(np.dot(point - a, ab)) / length2))
    return float(np.linalg.norm(point - (a + ab * t)))


def _point_in_polygon(point, corners):
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(corners)
    for index in range(n):
        x1, y1 = corners[index]
        x2, y2 = corners[(index + 1) % n]
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def gizmo_hit(camera, width, height, origin, scale, screen_xy, threshold=GIZMO_HIT_PIXELS):
    """('plane', name) or ('axis', name) for the gizmo part under ``screen_xy``, or None.

    Planes are tested first (they are small, sit closer to the pivot, and are drawn on top of
    the arrow shafts); the nearest arrow wins among axis candidates within ``threshold`` pixels.
    """
    screen_xy = np.asarray(screen_xy, np.float64)
    for name, corners in gizmo_planes(origin, scale).items():
        xy, z = scene3d.project(camera, width, height, np.array(corners))
        if np.all(z > camera.near) and _point_in_polygon(screen_xy, xy):
            return ("plane", name)
    best = None
    for name, (start, end) in gizmo_arrows(origin, scale).items():
        xy, z = scene3d.project(camera, width, height, np.array((start, end)))
        if z[0] <= camera.near or z[1] <= camera.near:
            continue
        # An axis end-on to the camera (or very nearly so) projects to a near-zero-length
        # segment: every screen point is "on" it, which would swallow orbiting and picking
        # anywhere near the pivot. Real tools drop such a foreshortened handle; so do we.
        if float(np.linalg.norm(xy[1] - xy[0])) < 2 * threshold:
            continue
        distance = _point_segment_distance(screen_xy, xy[0], xy[1])
        if distance <= threshold and (best is None or distance < best[1]):
            best = (("axis", name), distance)
    return best[0] if best else None
