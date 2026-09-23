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
