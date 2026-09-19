"""Small, deterministic 3D foundation used by the image evaluator.

This is intentionally a bounded CPU/reference renderer: a handful of typed primitives,
one perspective camera, flat RGBA materials, depth buffering, and a grid/axis overlay for
the interactive viewport.  It is a real render path (not a node-shaped placeholder), while
leaving splats, ray tracing, particles, fluids, textures, and GPU scene evaluation for the
roadmap owned by the parent branch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def array(self):
        return np.array((self.x, self.y, self.z), dtype=np.float32)


@dataclass(frozen=True)
class Transform3D:
    position: Vec3 = Vec3()
    rotation: Vec3 = Vec3()  # degrees, XYZ Euler order
    scale: Vec3 = Vec3(1.0, 1.0, 1.0)

    def matrix(self):
        rx, ry, rz = (math.radians(v) for v in (self.rotation.x, self.rotation.y, self.rotation.z))
        cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry),
                                  math.cos(rz), math.sin(rz))
        rot = np.array(((cy * cz, -cy * sz, sy),
                        (sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy),
                        (-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy)), np.float32)
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = rot @ np.diag((self.scale.x, self.scale.y, self.scale.z))
        out[:3, 3] = self.position.array()
        return out


@dataclass(frozen=True)
class Geometry:
    vertices: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float, float]
    transform: Transform3D = Transform3D()


@dataclass(frozen=True)
class Camera:
    transform: Transform3D = Transform3D(position=Vec3(0, 0, 5))
    target: Vec3 = Vec3()
    fov: float = 45.0
    near: float = 0.1
    far: float = 1000.0


@dataclass(frozen=True)
class Scene:
    geometries: tuple[Geometry, ...] = ()


def _card(width, height, color, transform):
    w, h = float(width) / 2, float(height) / 2
    return Geometry(np.array(((-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)), np.float32),
                    np.array(((0, 1, 2), (0, 2, 3)), np.int32), color, transform)


def _cube(size, color, transform):
    h = float(size) / 2
    v = np.array(((-h,-h,-h),(h,-h,-h),(h,h,-h),(-h,h,-h),
                  (-h,-h,h),(h,-h,h),(h,h,h),(-h,h,h)), np.float32)
    t = np.array(((0,1,2),(0,2,3),(1,5,6),(1,6,2),(5,4,7),(5,7,6),
                  (4,0,3),(4,3,7),(3,2,6),(3,6,7),(4,5,1),(4,1,0)), np.int32)
    return Geometry(v, t, color, transform)


def geometry_from_node(node):
    p = node["params"]
    transform = Transform3D(Vec3(p["x"], p["y"], p["z"]), Vec3(p["rx"], p["ry"], p["rz"]),
                            Vec3(p["sx"], p["sy"], p["sz"]))
    color = (float(p["red"]), float(p["green"]), float(p["blue"]), float(p["alpha"]))
    return (_card(p["width"], p["height"], color, transform) if node["type"] == "Card3D"
            else _cube(p["size"], color, transform))


def camera_from_node(node):
    p = node["params"]
    return Camera(Transform3D(Vec3(p["x"], p["y"], p["z"]), Vec3(p["rx"], p["ry"], p["rz"])),
                  Vec3(p["target_x"], p["target_y"], p["target_z"]), p["fov"], p["near"], p["far"])


def render(scene: Scene, camera: Camera, width: int, height: int, background=(0., 0., 0., 0.)):
    """Rasterize a scene to premultiplied float32 scene-linear RGBA."""
    width, height = int(width), int(height)
    out = np.broadcast_to(np.asarray(background, np.float32), (height, width, 4)).copy()
    depth = np.full((height, width), np.inf, np.float32)
    eye = camera.transform.position.array()
    target = camera.target.array()
    forward = target - eye; forward /= max(np.linalg.norm(forward), 1e-8)
    up = np.array((0, 1, 0), np.float32)
    right = np.cross(forward, up); right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(right, forward)
    view = np.stack((right, up, -forward), axis=0)
    aspect = width / max(height, 1)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    for geometry in scene.geometries:
        points = (view @ (geometry.transform.matrix()[:3, :3] @ geometry.vertices.T
                          + geometry.transform.matrix()[:3, 3:4] - eye[:, None])).T
        z = -points[:, 2]
        visible = z > camera.near
        projected = np.empty((len(points), 2), np.float32)
        projected[:, 0] = (points[:, 0] * focal / aspect / np.maximum(z, 1e-8) * .5 + .5) * width
        projected[:, 1] = (1 - (points[:, 1] * focal / np.maximum(z, 1e-8) * .5 + .5)) * height
        rgba = np.asarray(geometry.color, np.float32); alpha = np.clip(rgba[3], 0, 1)
        premult = rgba.copy(); premult[:3] *= alpha
        for ia, ib, ic in geometry.triangles:
            if not (visible[ia] and visible[ib] and visible[ic]): continue
            a, b, c = projected[[ia, ib, ic]]
            x0, x1 = max(0, int(math.floor(min(a[0], b[0], c[0])))), min(width - 1, int(math.ceil(max(a[0], b[0], c[0]))))
            y0, y1 = max(0, int(math.floor(min(a[1], b[1], c[1])))), min(height - 1, int(math.ceil(max(a[1], b[1], c[1]))))
            if x1 < x0 or y1 < y0: continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]; px = xx + .5; py = yy + .5
            den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
            if abs(float(den)) < 1e-8: continue
            wa = ((b[1]-c[1])*(px-c[0]) + (c[0]-b[0])*(py-c[1])) / den
            wb = ((c[1]-a[1])*(px-c[0]) + (a[0]-c[0])*(py-c[1])) / den
            wc = 1 - wa - wb; inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
            zbuf = wa*z[ia] + wb*z[ib] + wc*z[ic]
            region_depth = depth[y0:y1+1, x0:x1+1]; take = inside & (zbuf < region_depth)
            if not np.any(take): continue
            region_depth[take] = zbuf[take]
            region = out[y0:y1+1, x0:x1+1]; dst_a = region[..., 3]
            region[take, :3] = premult[:3] + region[take, :3] * (1 - alpha)
            region[take, 3] = alpha + dst_a[take] * (1 - alpha)
    out.flags.writeable = False
    return out


def grid_axes(width, height, scale=10, extent=10):
    """Return line segments for a simple editor grid and RGB axes (viewport overlay input)."""
    lines = []
    for i in range(-extent, extent + 1):
        lines.extend([((-extent, 0, i), (extent, 0, i), (0.25, 0.25, 0.25, 1)),
                      ((i, 0, -extent), (i, 0, extent), (0.25, 0.25, 0.25, 1))])
    lines.extend([((0, 0, 0), (scale, 0, 0), (1, 0.2, 0.2, 1)),
                  ((0, 0, 0), (0, scale, 0), (0.2, 1, 0.2, 1)),
                  ((0, 0, 0), (0, 0, scale), (0.2, 0.4, 1, 1))])
    return lines
