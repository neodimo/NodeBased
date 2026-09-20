"""Deterministic CPU reference renderer behind NodeBased's 3D nodes.

Typed scene values (geometry, lights, cameras, scenes) are assembled by the image evaluator and
rasterized here into premultiplied, scene-linear float32 RGBA. The rasterizer is NumPy on the
CPU: perspective-correct attributes, near-plane clipping, z-buffered opaque surfaces, sorted
transparency, mip-mapped bilinear textures, Lambert lighting, supersampled antialiasing and
depth/normal outputs, primary ray tracing and an EWA Gaussian splat beauty layer.
It is the correctness reference, not a throughput claim.

Conventions: right-handed, +Y up, camera looks down -Z in view space, world units are
unitless, rotations are degrees in XYZ Euler order, UV (0,0) is the bottom-left of a texture
and image row 0 is the top of the frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
import math
import os
import tempfile

import numpy as np

from .raytrace import Bvh, TriangleSet
from .splats import SplatCloud

MAX_TRIANGLES = 250_000
SHADOW_WORK_BUDGET = 4_000_000_000
RAYTRACE_WORK_BUDGET = 4_000_000_000
_PRIMARY_RAY_CHUNK = 4096
# Bounds shaded surfaces, including alpha-zero surfaces, through termination.
MAX_HITS_PER_RAY = 64
MAX_MESH_LAYERS = 16
PEEL_BATCH = 8
# At most 64 triangles, broadcasting avoids traversal overhead. Patch for parity tests.
_SHADOW_BRUTE_THRESHOLD = 64
# 518,400 ground-to-light rays, 10,002 / 99,858 sphere+ground triangles:
# 3.130 / 3.981 s query, 61 / 590 ms build. Relative to measured brute
# throughput, c1 = 14.2 / 14.6; round up to 16 equivalent tests per level.
_SHADOW_BVH_COST = 16.0
# Build timings correspond to about 14.4 / 11.2 equivalent tests per N log N.
_SHADOW_BVH_BUILD_COST = 16.0
_SHADOW_RAY_CHUNK = 128
_SHADOW_TRIANGLE_CHUNK = 512
RENDER_OUTPUTS = ("rgba", "depth", "normals", "albedo", "diffuse", "specular",
                  "emission", "position", "uv", "object_id")
LIGHT_OUTPUTS = ("rgba", "albedo", "diffuse", "specular", "emission")
DATA_OUTPUTS = ("depth", "normals", "position", "uv", "object_id")
LIGHT_TYPES = ("Directional", "Point")
_IDENTITY = np.eye(4, dtype=np.float32)
_IDENTITY.flags.writeable = False


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

    order: str = 'XYZ'
    pivot: Vec3 = Vec3()
    uniform: float = 1.0

    def matrix(self):
        """Column vectors: T(position) @ T(pivot) @ R(order) @ S @ T(-pivot).

        R(XYZ) = Rx @ Ry @ Rz, so the rightmost rotation acts first.
        S multiplies each axis scale by uniform.
        """
        if self.order not in ('XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'):
            raise ValueError('rotation order must be a permutation of XYZ')
        rx, ry, rz = (math.radians(v) for v in (self.rotation.x, self.rotation.y, self.rotation.z))
        cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry),
                                  math.cos(rz), math.sin(rz))
        rot = np.array(((cy * cz, -cy * sz, sy),
                        (sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy),
                        (-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy)), np.float32)
        if self.order != 'XYZ':
            axes = dict(X=np.array(((1,0,0),(0,cx,-sx),(0,sx,cx))),
                        Y=np.array(((cy,0,sy),(0,1,0),(-sy,0,cy))),
                        Z=np.array(((cz,-sz,0),(sz,cz,0),(0,0,1))))
            rot = (axes[self.order[0]] @ axes[self.order[1]] @ axes[self.order[2]]).astype(np.float32)
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = rot @ np.diag((self.scale.x*self.uniform, self.scale.y*self.uniform,
                                    self.scale.z*self.uniform))
        out[:3, 3] = self.position.array()
        if self.pivot != Vec3():
            pivot = self.pivot.array()
            out[:3, 3] += pivot - out[:3, :3] @ pivot
        return out


@dataclass(frozen=True, eq=False)
class Geometry:
    vertices: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float, float]
    transform: Transform3D = Transform3D()
    uvs: np.ndarray | None = None       # per vertex, (N,2)
    normals: np.ndarray | None = None   # per vertex object-space; None means flat face normals
    texture: np.ndarray | None = None   # premultiplied float32 RGBA, row 0 at the top
    parent: np.ndarray = field(default_factory=lambda: _IDENTITY)  # enclosing Scene3D transforms
    projection: Projection | None = None

    specular: float = 0.0
    shininess: float = 32.0
    emission: float = 0.0

    def world_matrix(self):
        return self.parent @ self.transform.matrix()


@dataclass(frozen=True, eq=False)
class Light:
    kind: str = "Directional"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    position: Vec3 = Vec3(2.0, 4.0, 3.0)
    target: Vec3 = Vec3()
    parent: np.ndarray = field(default_factory=lambda: _IDENTITY)
    shadows: bool = False

    def world(self):
        """World-space (position, unit direction the light travels along)."""
        position = (self.parent @ np.append(self.position.array(), 1.0))[:3]
        target = (self.parent @ np.append(self.target.array(), 1.0))[:3]
        direction = target - position
        return position.astype(np.float32), (direction / max(np.linalg.norm(direction), 1e-8)).astype(np.float32)


@dataclass(frozen=True)
class Camera:
    transform: Transform3D = Transform3D(position=Vec3(0, 0, 5))
    target: Vec3 = Vec3()
    fov: float = 45.0   # vertical, degrees
    near: float = 0.1
    far: float = 1000.0
    roll: float = 0.0   # degrees about the view axis


@dataclass(frozen=True, eq=False)
class Projection:
    """Camera-projected premultiplied RGBA; image row zero is at the top.

    Depth occlusion approximates visibility with a texture-aspect depth map, capped
    at 512 pixels on its longer side. A conservative 3x3 comparison and a view-depth
    bias of max(0.002 * depth, 0.001) avoid acne, but soften occlusion boundaries.
    Transparent occluders count as occluders wherever their alpha is greater than zero.
    """
    camera: Camera
    texture: np.ndarray
    outside: str = "transparent"  # transparent | clamp
    backfaces: str = "project"    # project | skip
    occlusion: str = "off"        # off | depth


@dataclass(frozen=True, eq=False)
class SplatInstance:
    """A cloud under a column-vector world transform."""
    cloud: SplatCloud
    matrix: np.ndarray = field(default_factory=lambda: _IDENTITY)
    sh_degree: int | None = None
    opacity_scale: float = 1.0
    scale_scale: float = 1.0


@dataclass(frozen=True, eq=False)
class Scene:
    geometries: tuple[Geometry, ...] = ()
    lights: tuple[Light, ...] = ()
    splats: tuple = ()


def write_obj(scene, path):
    """Atomically write world-space OBJ geometry, returning object/vertex/triangle counts.

    UVs and per-vertex normals are preserved. Lights, colours, textures and projections
    are not exported. Each geometry becomes one object; transforms are baked into positions.
    """
    counts = dict(objects=len(scene.geometries),
                  vertices=sum(len(g.vertices) for g in scene.geometries),
                  triangles=sum(len(g.triangles) for g in scene.geometries))
    if not counts['objects'] or not counts['triangles']:
        raise ValueError("Cannot export an empty scene: no geometry faces")
    if counts['triangles'] > MAX_TRIANGLES:
        raise ValueError(f"Scene exceeds {MAX_TRIANGLES} triangles; OBJ export refuses it")
    destination = Path(path).expanduser()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=destination.parent, suffix='.obj.tmp', delete=False) as handle:
            temporary = handle.name
            vo = uo = no = 1
            for number, geometry in enumerate(scene.geometries, 1):
                matrix = geometry.world_matrix()
                vertices = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
                normals = geometry.normals
                if normals is not None:
                    if normals.shape != geometry.vertices.shape:
                        raise ValueError("OBJ export requires per-vertex normals")
                    normals = (np.linalg.inv(matrix[:3, :3]).T @ normals.T).T
                    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
                uvs = geometry.uvs
                if uvs is not None and uvs.shape != (len(vertices), 2):
                    raise ValueError("OBJ export requires per-vertex UVs")
                handle.write(f'o geometry_{number}\n')
                for tag, array in (('v', vertices), ('vt', uvs), ('vn', normals)):
                    if array is not None:
                        if not np.isfinite(array).all():
                            raise ValueError("OBJ export requires finite geometry attributes")
                        for row in array:
                            handle.write(tag + ' ' + ' '.join('%.9g' % x for x in row) + '\n')
                for triangle in geometry.triangles:
                    face = []
                    for index in triangle:
                        if index < 0 or index >= len(vertices):
                            raise ValueError("OBJ face references a missing vertex")
                        token = str(vo + index)
                        if uvs is not None or normals is not None:
                            token += '/' + (str(uo + index) if uvs is not None else '')
                        if normals is not None:
                            token += '/' + str(no + index)
                        face.append(token)
                    handle.write('f ' + ' '.join(face) + '\n')
                vo += len(vertices)
                uo += len(uvs) if uvs is not None else 0
                no += len(normals) if normals is not None else 0
        os.replace(temporary, destination)
    except (OSError, np.linalg.LinAlgError) as error:
        raise ValueError(f"Cannot export OBJ {destination}: {error}") from error
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return counts


def apply_projection(scene_or_geometry: Scene | Geometry, projection: Projection):
    """Return a projected copy, preserving geometry transforms and scene lights."""
    if isinstance(scene_or_geometry, Geometry):
        return replace(scene_or_geometry, projection=projection)
    return replace(scene_or_geometry, geometries=tuple(
        replace(geometry, projection=projection) for geometry in scene_or_geometry.geometries))


def _projection_uv(projection, points):
    """World positions to bottom-up UVs and projection-camera view depths."""
    h, w = projection.texture.shape[:2]
    pixels, depth = project(projection.camera, w, h, points)
    return np.column_stack((pixels[:, 0] / w, 1 - pixels[:, 1] / h)), depth


# --- primitives ---------------------------------------------------------------------------------

def _card(width, height, color, transform, texture=None):
    w, h = float(width) / 2, float(height) / 2
    return Geometry(np.array(((-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)), np.float32),
                    np.array(((0, 1, 2), (0, 2, 3)), np.int32), color, transform,
                    uvs=np.array(((0, 0), (1, 0), (1, 1), (0, 1)), np.float32), texture=texture)


def _cube(size, color, transform, texture=None):
    h = float(size) / 2
    # Four vertices per face so every face carries its own 0..1 UV square.
    faces = (((-h,-h,h),(h,-h,h),(h,h,h),(-h,h,h)), ((h,-h,-h),(-h,-h,-h),(-h,h,-h),(h,h,-h)),
             ((h,-h,h),(h,-h,-h),(h,h,-h),(h,h,h)), ((-h,-h,-h),(-h,-h,h),(-h,h,h),(-h,h,-h)),
             ((-h,h,h),(h,h,h),(h,h,-h),(-h,h,-h)), ((-h,-h,-h),(h,-h,-h),(h,-h,h),(-h,-h,h)))
    v = np.array([p for face in faces for p in face], np.float32)
    t = np.array([(i*4+a, i*4+b, i*4+c) for i in range(6) for a, b, c in ((0, 1, 2), (0, 2, 3))], np.int32)
    return Geometry(v, t, color, transform, uvs=np.tile(((0, 0), (1, 0), (1, 1), (0, 1)), (6, 1)).astype(np.float32),
                    texture=texture)


def _sphere(radius, segments, color, transform, texture=None):
    cols, rows = max(3, int(segments)), max(2, int(segments) // 2)
    u, v = np.meshgrid(np.linspace(0, 1, cols + 1), np.linspace(0, 1, rows + 1))
    theta, phi = u * 2 * np.pi, (v - 0.5) * np.pi
    unit = np.stack((np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)), -1).reshape(-1, 3)
    tris = []
    for r in range(rows):
        for c in range(cols):
            a, b = r * (cols + 1) + c, r * (cols + 1) + c + 1
            d, e = a + cols + 1, b + cols + 1
            tris += [(a, b, e), (a, e, d)]
    return Geometry((unit * float(radius)).astype(np.float32), np.array(tris, np.int32), color, transform,
                    uvs=np.stack((u, v), -1).reshape(-1, 2).astype(np.float32),
                    normals=unit.astype(np.float32), texture=texture)


@lru_cache(maxsize=8)
def _load_obj(path, _size, _mtime_ns):
    """Parse a Wavefront OBJ into (vertices, triangles, uvs|None, normals|None), fan-triangulated.
    OBJ indexes position/uv/normal separately, so corners are re-welded into single vertices."""
    positions, texcoords, normals, corners, tris = [], [], [], {}, []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                positions.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "vt":
                texcoords.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
            elif parts[0] == "vn":
                normals.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "f":
                face = []
                for token in parts[1:]:
                    ids = (token.split("/") + ["", ""])[:3]
                    key = tuple((int(i) - 1 if int(i) > 0 else n + int(i)) if i else -1
                                for i, n in zip(ids, (len(positions), len(texcoords), len(normals))))
                    face.append(corners.setdefault(key, len(corners)))
                tris += [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]
                if len(tris) > MAX_TRIANGLES:
                    raise ValueError(f"{Path(path).name}: more than {MAX_TRIANGLES} triangles; "
                                     "the CPU reference renderer refuses meshes this large")
    if not tris:
        raise ValueError(f"{Path(path).name}: no faces found (only Wavefront OBJ polygons are read)")
    keys = list(corners)
    # Mixed OBJ objects may combine smooth normals with flat faces. Split only missing-normal
    # corners per triangle so those faces stay flat without discarding the supplied normals.
    if normals and any(k[2] < 0 for k in keys):
        for face_index, triangle in enumerate(tris):
            if all(keys[i][2] >= 0 for i in triangle):
                continue
            try:
                a, b, c = np.array([positions[keys[i][0]] for i in triangle], np.float32)
            except IndexError:
                raise ValueError(f"{Path(path).name}: a face references a missing vertex") from None
            normal = np.cross(b - a, c - a)
            normal /= max(float(np.linalg.norm(normal)), 1e-8)
            normals.append(tuple(normal))
            replacement = []
            for i in triangle:
                if keys[i][2] < 0:
                    keys.append((*keys[i][:2], len(normals) - 1))
                    i = len(keys) - 1
                replacement.append(i)
            tris[face_index] = tuple(replacement)
        # Remove the original missing-normal corners, which no face references anymore.
        used = dict.fromkeys(i for triangle in tris for i in triangle)
        remap = {old: new for new, old in enumerate(used)}
        keys = [keys[i] for i in used]
        tris = [tuple(remap[i] for i in triangle) for triangle in tris]
    try:
        vertices = np.array([positions[k[0]] for k in keys], np.float32)
        uvs = (np.array([texcoords[k[1]] for k in keys], np.float32)
               if all(k[1] >= 0 for k in keys) else None)
        vertex_normals = (np.array([normals[k[2]] for k in keys], np.float32)
                          if all(k[2] >= 0 for k in keys) else None)
    except IndexError:
        raise ValueError(f"{Path(path).name}: a face references a missing vertex") from None
    for array in (vertices, uvs, vertex_normals):
        if array is not None:
            array.flags.writeable = False
    triangles = np.array(tris, np.int32); triangles.flags.writeable = False
    return vertices, triangles, uvs, vertex_normals


def obj_fingerprint(path):
    """Identity of the file on disk, for cache digests. Raises ValueError if it is unreadable."""
    if not path:
        raise ValueError("ReadGeo3D: choose an OBJ file")
    try:
        stat = Path(path).stat()
    except OSError:
        raise ValueError(f"ReadGeo3D: cannot read {path}") from None
    return [str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns]


def _transform_from(p):
    return Transform3D(Vec3(p["tx"], p["ty"], p["tz"]), Vec3(p["rx"], p["ry"], p["rz"]),
                       Vec3(p["sx"], p["sy"], p["sz"]),
                       order=p.get("rot_order", "XYZ"),
                       pivot=Vec3(*(p.get("pivot_" + axis, 0.0) for axis in "xyz")),
                       uniform=p.get("uscale", 1.0))


def geometry_from_node(node, texture=None):
    p = node["params"]
    return replace(_geometry_from_node(node, texture),
                   specular=float(p.get("spec_amount", 0.0)),
                   shininess=float(p.get("spec_shininess", 32.0)),
                   emission=float(p.get("emission", 0.0)))


def _geometry_from_node(node, texture=None):
    p = node["params"]
    color = tuple(float(p[k]) for k in ("red", "green", "blue", "alpha"))
    transform = _transform_from(p)
    kind = node["type"]
    if kind == "Card3D":
        return _card(p["card_width"], p["card_height"], color, transform, texture)
    if kind == "Cube3D":
        return _cube(p["cube_size"], color, transform, texture)
    if kind == "Sphere3D":
        return _sphere(p["sphere_radius"], p["segments"], color, transform, texture)
    if kind == "ReadGeo3D":
        resolved, size, mtime = obj_fingerprint(p["geo_path"])
        vertices, triangles, uvs, normals = _load_obj(resolved, size, mtime)
        return Geometry(vertices, triangles, color, transform, uvs=uvs, normals=normals, texture=texture)
    raise ValueError(f"{kind} is not a geometry node")


def light_from_node(node):
    p = node["params"]
    return Light(p["light_type"], (float(p["red"]), float(p["green"]), float(p["blue"])),
                 float(p["intensity"]), Vec3(p["tx"], p["ty"], p["tz"]),
                 Vec3(p["target_x"], p["target_y"], p["target_z"]),
                 shadows=p.get("shadows", "off") == "on")


def camera_from_node(node):
    p = node["params"]
    return Camera(Transform3D(Vec3(p["tx"], p["ty"], p["tz"])),
                  Vec3(p["target_x"], p["target_y"], p["target_z"]), p["fov"], p["near"], p["far"],
                  p["roll"])


def scene_from_node(node, members):
    """Assemble geometry, lights, splats and nested scenes under this node's transform."""
    matrix = _transform_from(node["params"]).matrix()
    geometries, lights, splats = [], [], []
    for member in members:
        if isinstance(member, Scene):
            items = member.geometries + member.lights + member.splats
        else:
            items = (member,)
        for item in items:
            if isinstance(item, SplatInstance):
                splats.append(replace(item, matrix=matrix @ item.matrix))
                continue
            moved = type(item)(**{**item.__dict__, "parent": matrix @ item.parent})
            (geometries if isinstance(item, Geometry) else lights).append(moved)
    return Scene(tuple(geometries), tuple(lights), tuple(splats))


# --- camera -------------------------------------------------------------------------------------

_VIEW_LIGHT = np.array((-0.45, 0.8, 0.4), np.float32) / np.linalg.norm((-0.45, 0.8, 0.4))


def _view_basis(camera):
    eye = camera.transform.position.array()
    forward = camera.target.array() - eye; forward /= max(np.linalg.norm(forward), 1e-8)
    up = np.array((0, 1, 0), np.float32)
    if abs(float(forward @ up)) > 0.9999:  # looking straight up/down: pick a stable roll axis
        up = np.array((0, 0, -1 if forward[1] < 0 else 1), np.float32)
    right = np.cross(forward, up); right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(right, forward)
    if camera.roll:
        c, s = math.cos(math.radians(camera.roll)), math.sin(math.radians(camera.roll))
        right, up = c * right + s * up, c * up - s * right
    return eye, np.stack((right, up, -forward), axis=0).astype(np.float32)


def _to_pixels(local, z, focal, aspect, width, height):
    xy = np.empty((len(local), 2), np.float32)
    safe = np.maximum(z, 1e-8)
    xy[:, 0] = (local[:, 0] * focal / aspect / safe * .5 + .5) * width
    xy[:, 1] = (1 - (local[:, 1] * focal / safe * .5 + .5)) * height
    return xy


def project(camera: Camera, width: int, height: int, points):
    """World points (N,3) -> pixel xy (N,2) and view depth (N,), matching render()."""
    eye, view = _view_basis(camera)
    local = (view @ (np.asarray(points, np.float32) - eye).T).T
    z = -local[:, 2]
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    return _to_pixels(local, z, focal, width / max(height, 1), width, height), z


def frustum_lines(camera: Camera, aspect: float, length: float = 1.5):
    """World-space line segments sketching a camera: eye-to-corner rays and the far rectangle."""
    eye, view = _view_basis(camera)
    half = math.tan(math.radians(camera.fov) / 2) * length
    corners = [eye + view.T @ np.array((sx * half * aspect, sy * half, -length), np.float32)
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return ([(eye, corner) for corner in corners]
            + [(corners[i], corners[(i + 1) % 4]) for i in range(4)])


# --- rasterizer ---------------------------------------------------------------------------------

def _mip_chain(texture):
    levels = [np.asarray(texture, np.float32)]
    while min(levels[-1].shape[:2]) > 1:
        t = levels[-1]
        h, w = t.shape[0] // 2 * 2, t.shape[1] // 2 * 2
        levels.append(t[:h, :w].reshape(h // 2, 2, w // 2, 2, 4).mean(axis=(1, 3)))
    return levels


def _sample(texture, u, v):
    """Bilinear sample with edge clamp. v=0 is the bottom row of the image."""
    h, w = texture.shape[:2]
    x = np.clip(u, 0, 1) * w - 0.5
    y = (1 - np.clip(v, 0, 1)) * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    return ((texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx) * (1 - fy)
            + (texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx) * fy)


def _clip_near(attributes, z, near):
    """Clip one triangle against z=near in view space. `attributes` is (3,K); returns a list of
    (3,K) triangles (0, 1 or 2) with every attribute interpolated along the cut edges."""
    inside = z > near
    if inside.all():
        return [attributes]
    if not inside.any():
        return []
    polygon = []
    for i in range(3):
        j = (i + 1) % 3
        if inside[i]:
            polygon.append(attributes[i])
        if inside[i] != inside[j]:
            t = (near - z[i]) / (z[j] - z[i])
            polygon.append(attributes[i] + (attributes[j] - attributes[i]) * t)
    return [np.stack((polygon[0], polygon[i], polygon[i + 1])) for i in range(1, len(polygon) - 1)]


def _shadow_cancel(cancel):
    if cancel is not None and cancel.is_set():
        from .imaging import Cancelled
        raise Cancelled()


def _shadow_cost(rays, triangles, *, build=True):
    if triangles <= _SHADOW_BRUTE_THRESHOLD:
        return rays * triangles
    levels = math.log2(triangles + 2)
    return rays * _SHADOW_BVH_COST * levels + (_SHADOW_BVH_BUILD_COST * triangles * levels if build else 0)


def _shadow_budget(work):
    if work > SHADOW_WORK_BUDGET:
        raise ValueError(f"Shadow rays exceed the CPU reference budget: {work:,.0f} estimated "
                         "ray-triangle-equivalent tests (including BVH build); "
                         "reduce resolution/samples/triangles or switch shadows off")


def _shadow_visibility(position, normal, light, light_position, direction,
                       v0, e1, e2, alpha, bias, cancel, *, triangles=None, bvh=None):
    """Chunked, two-sided Moller-Trumbore; material alpha only, never texture alpha."""
    visibility = np.ones(len(position), np.float32)
    for start in range(0, len(position), _SHADOW_RAY_CHUNK):
        _shadow_cancel(cancel)
        stop = start + _SHADOW_RAY_CHUNK
        origin = position[start:stop] + normal[start:stop] * bias
        if light.kind == "Point":
            ray = light_position - origin
            limit = np.linalg.norm(ray, axis=1)
            ray = ray / np.maximum(limit[:, None], 1e-8)
        else:
            ray = np.broadcast_to(-direction, origin.shape)
            limit = np.full(len(origin), np.inf)
        primitives = triangles if triangles is not None else TriangleSet(v0, e1, e2, alpha)
        if bvh is None:
            visibility[start:stop] = primitives.brute_transmittance(
                origin, ray, bias * .01, limit, triangle_chunk=_SHADOW_TRIANGLE_CHUNK, cancel=cancel)
        else:
            visibility[start:stop] = primitives.transmittance(
                bvh, origin, ray, bias * .01, limit, cancel=cancel)
    return visibility


@dataclass
class _ShadowContext:
    primitives: TriangleSet
    bvh: Bvh | None
    bias: float
    cancel: object
    triangle_count: int
    work: float
    raytrace: bool = False

    def visibility(self, position, normal, light, light_position, direction):
        self.work += _shadow_cost(len(position), self.triangle_count, build=False)
        (_raytrace_budget if self.raytrace else _shadow_budget)(self.work)
        p = self.primitives
        return _shadow_visibility(position, normal, light, light_position, direction,
                                  p.v0, p.e1, p.e2, p.alpha, self.bias, self.cancel,
                                  triangles=p, bvh=self.bvh)


def _triangle_mip(tri, den, mips, projection):
    if mips is None:
        return 0
    tri_uv = tri[:, 9:11]
    if projection is not None:
        tri_uv, _ = _projection_uv(projection, tri[:, 3:6])
    # One mip per clipped triangle from texel/pixel area, shared by both modes.
    h, w = mips[0].shape[:2]
    (eu, ev), (fu, fv) = tri_uv[1] - tri_uv[0], tri_uv[2] - tri_uv[0]
    uv_area = abs(float(eu * fv - ev * fu)) * w * h
    return int(np.clip(round(0.5 * math.log2(max(uv_area / max(abs(den), 1e-8), 1.0))), 0, len(mips) - 1))


def _shade_fragments(position, normal, uv, *, geometry, rgba, mips, level,
                     eye, lights, ambient, output, shade, scene,
                     projection_depth_maps, shadow_context, cancel):
    """Shared surface shader; inputs are world attributes and triangle mip information."""
    projection = geometry.projection
    lit = bool(lights) and not shade
    source = np.broadcast_to(rgba, (len(position), 4)).copy()
    source[:, :3] *= source[:, 3:4]                          # premultiply the flat colour
    if mips is not None:
        if projection is not None:
            uv, projection_depth = _projection_uv(projection, position)
        texel = _sample(mips[level], uv[:, 0], uv[:, 1])
        source = texel * np.append(rgba[:3] * rgba[3], rgba[3])  # tint premultiplied texels
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    if projection is not None:
        rejected = projection_depth <= 0
        if projection.outside == "transparent":
            rejected |= ((projection_depth <= projection.camera.near)
                         | (projection_depth >= projection.camera.far)
                         | (uv < 0).any(axis=1) | (uv > 1).any(axis=1))
        if projection.occlusion == "depth":
            # Filmback aspect is part of the camera: different texture aspects need
            # separate maps even when the authored Camera value is shared.
            th, tw = projection.texture.shape[:2]
            scale = min(1.0, 512 / max(tw, th))
            mw, mh = max(1, round(tw * scale)), max(1, round(th * scale))
            key = (projection.camera, mw, mh)
            if key not in projection_depth_maps:
                occluders = replace(scene, geometries=tuple(
                    replace(g, projection=None) for g in scene.geometries))
                _, shadow_depth = render(occluders, projection.camera, mw, mh,
                                         output="depth", return_depth=True,
                                         samples=1, cancel=cancel)
                projection_depth_maps[key] = shadow_depth
            shadow_depth = projection_depth_maps[key]
            pixels, projector_z = project(projection.camera, mw, mh, position)
            in_map = ((pixels >= 0).all(axis=1)
                      & (pixels < (mw, mh)).all(axis=1)
                      & (projector_z > projection.camera.near)
                      & (projector_z < projection.camera.far))
            xy = np.floor(pixels[in_map]).astype(int)
            xx, yy = xy[:, 0], xy[:, 1]
            limit = np.full(len(xy), -np.inf, np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = shadow_depth[np.clip(yy + dy, 0, mh - 1),
                                             np.clip(xx + dx, 0, mw - 1)]
                    limit = np.maximum(limit, np.where(np.isfinite(neighbour), neighbour, -np.inf))
            # Use the largest finite neighbour, never the minimum: sloped cards,
            # sphere facets and cube faces must not shadow themselves. Empty centre
            # pixels remain visible; this trades a thin silhouette leak for no acne.
            limit[~np.isfinite(shadow_depth[yy, xx]) | (limit == -np.inf)] = np.inf
            fragment_z = projector_z[in_map]
            rejected[in_map] |= fragment_z > limit + np.maximum(2e-3 * fragment_z, 1e-3)
        if projection.backfaces == "skip":
            toward_projector = projection.camera.transform.position.array() - position
            rejected |= np.einsum("ij,ij->i", normal, toward_projector) <= 0
        # Mask before shading and data outputs; zero alpha also prevents depth writes.
        source[rejected] = 0
    if lit or shade or output == "normals":
        toward_eye = eye - position
        normal = np.where((np.einsum("ij,ij->i", normal, toward_eye) < 0)[:, None], -normal, normal)
    albedo = source[:, :3].copy() if output == "albedo" else None
    emissive = source[:, :3] * geometry.emission if geometry.emission and output in ("rgba", "emission") else None
    specular = None
    if shade:
        source[:, :3] *= (0.25 + 0.75 * np.abs(normal @ _VIEW_LIGHT))[:, None]
    elif lit:
        radiance = np.full((len(position), 3), float(ambient), np.float32)
        specular = np.zeros_like(radiance) if geometry.specular and output in ("rgba", "specular") else None
        if specular is not None:
            to_eye = toward_eye / np.maximum(np.linalg.norm(toward_eye, axis=1, keepdims=True), 1e-8)
        for light, light_position, direction in lights:
            if light.kind == "Point":
                to_light = light_position - position
                to_light /= np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-8)
                lambert = np.einsum("ij,ij->i", normal, to_light)
            else:
                to_light = -direction
                lambert = normal @ -direction
            front = lambert > 0
            visibility = 1.0
            if shadow_context is not None and light.shadows:
                visibility = shadow_context.visibility(position, normal, light, light_position, direction)
                lambert = lambert * visibility
            radiance += np.maximum(lambert, 0)[:, None] * (np.asarray(light.color, np.float32) * light.intensity)
            if specular is not None:
                half = to_light + to_eye
                half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)
                lobe = np.maximum(np.einsum("ij,ij->i", normal, half), 0) ** geometry.shininess
                specular += (geometry.specular * lobe * front * visibility)[:, None] * (
                    np.asarray(light.color, np.float32) * light.intensity)
        source[:, :3] *= radiance
        if specular is not None:
            source[:, :3] += specular * source[:, 3:4]
    if emissive is not None:
        source[:, :3] += emissive
    if not shade:
        if output == "albedo":
            source[:, :3] = albedo
        elif output == "specular":
            source[:, :3] = specular * source[:, 3:4] if specular is not None else 0
        elif output == "emission":
            source[:, :3] = emissive if emissive is not None else 0
    return source, normal, uv


def _raytrace_budget(work):
    if work > RAYTRACE_WORK_BUDGET:
        raise ValueError(f"Ray-traced render exceeds the CPU reference budget: {work:,.0f} estimated "
                         "ray-triangle-equivalent tests (including BVH build); "
                         "reduce resolution/samples/triangles")


def _render_primary(scene, camera, width, height, out, depth, *, attributes, object_ids,
                    mip_levels, clipped_mips, materials, primitives, bvh, eye, view,
                    focal, aspect, lights, ambient, output, shade, shadow_context, cancel, mesh_layers=None):
    """Chunked primary visibility; shading is batched by geometry and mip level."""
    flat, flat_depth = out.reshape(-1, 4), depth.ravel()
    projection_depth_maps = {}
    data_output = output in DATA_OUTPUTS
    # Invert the actual float32 view basis, including roll, to undo _to_pixels.
    inverse_view = np.linalg.inv(view.astype(np.float64))
    for start in range(0, width*height, _PRIMARY_RAY_CHUNK):
        _shadow_cancel(cancel)
        stop = min(start+_PRIMARY_RAY_CHUNK, width*height)
        pixels = np.arange(start, stop)
        x, y = pixels % width + .5, pixels // width + .5
        local_dirs = np.column_stack(((2*x/width-1)*aspect/focal,
                                      (1-2*y/height)/focal, -np.ones(len(pixels))))
        dirs = local_dirs @ inverse_view.T
        origins = np.broadcast_to(eye, dirs.shape)
        # Unnormalised directions have unit view-forward depth: t is view z,
        # so near/far are planes rather than radial distances from the eye.
        alive = np.ones(len(pixels), dtype=bool)
        accumulated = np.zeros((len(pixels), 4), np.float64)
        transmission = np.ones(len(pixels), np.float64)
        composited = np.zeros(len(pixels), dtype=np.int32)
        active = np.arange(len(pixels))
        previous_t = np.full(len(pixels), -np.inf)
        previous_primitive = np.full(len(pixels), -1)
        previous_id = np.full(len(pixels), -1)
        previous_edge = np.zeros(len(pixels), bool)
        while len(active):
            _shadow_cancel(cancel)
            hit_lists = primitives.nearest_hits(
                bvh, origins[active], dirs[active], camera.near, camera.far,
                PEEL_BATCH, after_t=previous_t[active],
                after_primitive=previous_primitive[active], cancel=cancel)
            counts = np.array([len(h) for h in hit_lists])
            if not counts.sum():
                break
            hits = np.concatenate(hit_lists)
            rays = np.repeat(active, counts)
            ids = object_ids[hits['primitive']]
            keep = ids > 0
            edge = np.minimum.reduce((abs(hits['u']), abs(hits['v']),
                                      abs(1-hits['u']-hits['v']))) < 1e-10
            # Retain the previous RAW hit: exact ties and roundoff-separated
            # shared edges use the same adjacency/tolerance rule across batches.
            prior_t = np.r_[previous_t[rays[0]], hits['t'][:-1]]
            prior_id = np.r_[previous_id[rays[0]], ids[:-1]]
            prior_edge = np.r_[previous_edge[rays[0]], edge[:-1]]
            first = np.r_[True, rays[1:] != rays[:-1]]
            prior_t[first] = previous_t[rays[first]]
            prior_id[first] = previous_id[rays[first]]
            prior_edge[first] = previous_edge[rays[first]]
            duplicate = ((ids == prior_id) & edge & prior_edge
                         & (abs(hits['t']-prior_t) <= 1e-10*np.maximum(1, abs(hits['t']))))
            keep &= ~duplicate
            ends = np.cumsum(counts)[counts > 0]-1
            received = active[counts > 0]
            previous_t[received] = hits['t'][ends]
            previous_id[received] = ids[ends]
            previous_edge[received] = edge[ends]
            previous_primitive[received] = hits['primitive'][ends]
            next_active = active[counts == PEEL_BATCH]
            hits, rays = hits[keep], rays[keep]
            counts = np.bincount(rays, minlength=len(pixels))
            ranks = np.arange(len(rays)) - np.repeat(np.cumsum(counts)-counts, counts)
            for rank in range(int(counts.max(initial=0))):
                _shadow_cancel(cancel)
                selected = (ranks == rank) & alive[rays]
                if not selected.any():
                    continue
                h, rr = hits[selected], rays[selected]
                composited[rr] += 1
                if np.any(composited[rr] > MAX_HITS_PER_RAY):
                    raise ValueError(f'Ray-traced render exceeds MAX_HITS_PER_RAY ({MAX_HITS_PER_RAY}): '
                                     f'more than {MAX_HITS_PER_RAY} surfaces composited along a ray')
                if mesh_layers is not None and np.any(composited[rr] > MAX_MESH_LAYERS):
                    count = int(composited[rr].max())
                    raise ValueError(f'Ray-traced render exceeds MAX_MESH_LAYERS ({MAX_MESH_LAYERS}): {count} surfaces along a ray')
                pp = h['primitive']
                weights = np.column_stack((1-h['u']-h['v'], h['u'], h['v']))
                attr = np.einsum('ij,ijk->ik', weights, attributes[pp])
                levels = mip_levels[pp].copy()
                for primitive in np.unique(pp):
                    if primitive not in clipped_mips:
                        continue
                    at = np.flatnonzero(pp == primitive)
                    second, second_level = clipped_mips[primitive]
                    a, b, c = second[:, 3:6].astype(np.float64)
                    e, f = b-a, c-a
                    delta = attr[at, 3:6]-a
                    ee, ef, ff = e@e, e@f, f@f
                    den = ee*ff-ef*ef
                    if abs(den) > 1e-20:
                        u = (ff*(delta@e)-ef*(delta@f))/den
                        v = (ee*(delta@f)-ef*(delta@e))/den
                        levels[at[(u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)]] = second_level
                groups = np.column_stack((object_ids[pp], levels))
                for object_id, level in np.unique(groups, axis=0):
                    take = (groups[:, 0] == object_id) & (groups[:, 1] == level)
                    r = rr[take]
                    geometry, rgba, mips = materials[object_id-1]
                    position = attr[take, 3:6]
                    source, normal, uv = _shade_fragments(
                        position, attr[take, 6:9].copy(), attr[take, 9:11],
                        geometry=geometry, rgba=rgba, mips=mips, level=int(level),
                        eye=eye, lights=lights, ambient=ambient, output=output, shade=shade,
                        scene=scene, projection_depth_maps=projection_depth_maps,
                        shadow_context=shadow_context, cancel=cancel)
                    alpha = source[:, 3]
                    z = h['t'][take]
                    if data_output:
                        covered = alpha > 0
                        if output == 'depth':
                            values = np.repeat(z[:, None], 3, axis=1)
                        elif output == 'normals':
                            values = normal
                        elif output == 'position':
                            values = position
                        elif output == 'uv':
                            values = np.column_stack((uv, np.zeros(len(uv))))
                        else:
                            values = np.broadcast_to((object_id, 0, 0), (len(r), 3))
                        flat[start+r[covered]] = np.column_stack((values[covered], np.ones(covered.sum())))
                        flat_depth[start+r[covered]] = z[covered]
                        alive[r[covered]] = False
                    else:
                        if mesh_layers is not None:
                            ld, lc, la = mesh_layers
                            slot = composited[r]-1
                            ld.reshape(-1, MAX_MESH_LAYERS)[start+r, slot] = z
                            lc.reshape(-1, MAX_MESH_LAYERS, 3)[start+r, slot] = source[:, :3]
                            la.reshape(-1, MAX_MESH_LAYERS)[start+r, slot] = alpha
                        else:
                            accumulated[r] += transmission[r, None]*source
                        transmission[r] *= 1-alpha
                        solid = alpha >= .999
                        alive[r[solid]] = False
                        flat_depth[start+r[solid]] = z[solid]
            active = next_active[alive[next_active]]
        if not data_output and mesh_layers is None:
            flat[start:stop] = accumulated + transmission[:, None]*flat[start:stop]


def _opaque_meshes(scene):
    """Conservatively prove that the opaque depth-buffer shortcut is sufficient."""
    return all(g.color[3] >= .999 and g.projection is None
               and (g.texture is None or np.all(g.texture[..., 3] >= .999))
               for g in scene.geometries)


def _render_mesh_layers(scene, camera, width, height, out, depth, **kwargs):
    """Record primary-ray shaded surfaces, including the terminating surface."""
    shape = (height, width, MAX_MESH_LAYERS)
    layers = (np.full(shape, np.inf, np.float64),
              np.zeros((*shape, 3), np.float32), np.zeros(shape, np.float32))
    _render_primary(scene, camera, width, height, out, depth, mesh_layers=layers, **kwargs)
    return layers


def render(scene: Scene, camera: Camera, width: int, height: int, background=(0., 0., 0., 0.),
           shade=False, return_depth=False, ambient=0.0, samples=1, output="rgba", cancel=None, *, shadows=True, mode="raster"):
    """Render a scene to premultiplied float32 RGBA using raster or raytrace visibility.

    Surfaces are unlit (their authored colour/texture) until the scene has lights; then they are
    Lambert-shaded by those lights plus ``ambient``. ``shade`` instead applies the viewport's
    fixed inspection headlight. ``samples`` is supersampling per axis for beauty and light outputs.
    ``output`` "depth" writes view-space distance and "normals" world-space normals into RGB with
    coverage in alpha; both ignore the background and are never antialiased, because averaging
    depths or normals across an edge invents values that exist nowhere in the scene.
    Position stores world xyz, uv stores mesh/projected (u, v, 0), and object_id stores
    the 1-based scene geometry index in red. All data outputs use first-hit coverage:
    transparent geometry with alpha > 0 counts as a hit, without antialiasing.
    Light outputs exclude background and retain beauty alpha for premultiplied over.
    Their RGB identity is diffuse + specular + emission == transparent-background beauty.
    Unlit scenes have diffuse == albedo and specular == 0, hence beauty == albedo + emission.
    Alpha is shared coverage/transparency, not an additive lighting component.
    With ``shadows=True``, enabled lights trace two-sided world-space triangle rays.
    Every geometry casts and receives shadows, including projected geometry. Visibility
    multiplies (1 - geometry alpha) over hits; texture alpha is NOT considered. Ambient,
    data outputs and inspection shading are unaffected.
    Splats contribute only to rgba; all other outputs ignore them. Splat planes
    depth-test per pixel. With splats and transparent meshes, mesh visibility
    comes from primary rays in raster mode too, and all fragments are depth merged.
    ``return_depth`` also returns the depth buffer (inf where empty).
    """
    _shadow_cancel(cancel)
    if mode not in ("raster", "raytrace"):
        raise ValueError(f"Unknown 3D render mode {mode!r}")
    if output not in RENDER_OUTPUTS:
        raise ValueError(f"Unknown 3D render output {output!r}")
    width, height = int(width), int(height)
    data_output = output in DATA_OUTPUTS
    samples = max(1, min(int(samples), 4)) if not data_output else 1
    shadow_count = sum(light.shadows and light.intensity > 0 for light in scene.lights)
    shadow_active = shadows and not shade and output in ("rgba", "diffuse", "specular") and shadow_count > 0
    triangle_count = sum(len(g.triangles) for g in scene.geometries)
    if triangle_count > MAX_TRIANGLES:
        raise ValueError(f"Scene exceeds {MAX_TRIANGLES} triangles; the CPU reference renderer refuses it")
    layered = bool(scene.splats) and output == "rgba" and not _opaque_meshes(scene)
    if mode == "raytrace" or layered:
        rays = width * height * samples ** 2
        _raytrace_budget(_shadow_cost(rays, triangle_count)
                         + _shadow_cost(rays * shadow_count, triangle_count, build=False) * shadow_active)
    elif shadow_active:
        # Estimate before framebuffer allocation; a running counter also bounds overdraw.
        _shadow_budget(_shadow_cost(width * height * samples ** 2 * shadow_count, triangle_count))
    if samples > 1:
        big = render(scene, camera, width * samples, height * samples, background, shade,
                     return_depth, ambient, 1, output, cancel, shadows=shadows, mode=mode)
        image, depth = big if return_depth else (big, None)
        image = image.reshape(height, samples, width, samples, 4).mean(axis=(1, 3)).astype(np.float32)
        image.flags.writeable = False
        if return_depth:
            depth = depth.reshape(height, samples, width, samples).min(axis=(1, 3))
            depth.flags.writeable = False
            return image, depth
        return image
    bg = np.asarray(background, np.float32).copy()
    bg[3] = np.clip(bg[3], 0, 1)
    bg[:3] *= bg[3]
    if output != "rgba":
        bg[:] = 0
    out = np.broadcast_to(bg, (height, width, 4)).copy()
    depth = np.full((height, width), np.inf, np.float32)
    eye, view = _view_basis(camera)
    aspect = width / max(height, 1)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    lights = [(light, *light.world()) for light in scene.lights if light.intensity > 0]
    # Collect clipped triangles, then rasterize far-to-near. Opaque pixels write the z buffer;
    # transparent ones only test it, so they reveal what is behind them and composite in depth
    # order. That is sorted transparency, not order-independent transparency: interpenetrating
    # transparent surfaces can still sort wrongly.
    # Per-vertex attribute layout: view xyz (3) | world xyz (3) | world normal (3) | uv (2).
    queue = []
    # Transparent mesh/splat visibility uses identical primary rays in both modes.
    ray_mode = mode == "raytrace" or layered
    mesh_layers = None
    if ray_mode:
        ray_attributes = np.zeros((triangle_count, 3, 11), np.float32)
        ray_object_ids = np.zeros(triangle_count, np.int32)
        ray_mip_levels = np.zeros(triangle_count, np.int32)
        clipped_mips, materials = {}, []
        primitive_index = -1
    projection_depth_maps = {}
    shadow_triangles, shadow_alphas = [], []
    shadow_work = _shadow_cost(0, triangle_count) if shadow_active else 0
    for object_id, geometry in enumerate(scene.geometries, 1):
        _shadow_cancel(cancel)
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        if (shadow_active or ray_mode) and len(geometry.triangles):
            shadow_triangles.append(world[geometry.triangles])
            shadow_alphas.append(np.full(len(geometry.triangles), np.clip(geometry.color[3], 0, 1), np.float32))
        local = (view @ (world - eye).T).T
        if geometry.normals is not None:
            normal_matrix = np.linalg.inv(matrix[:3, :3]).T
            normals = (normal_matrix @ geometry.normals.T).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        uvs = geometry.uvs if geometry.uvs is not None else np.zeros((len(world), 2), np.float32)
        rgba = np.asarray(geometry.color, np.float32)
        mips = _mip_chain(geometry.texture) if geometry.texture is not None and geometry.uvs is not None else None
        projection = geometry.projection
        if projection is not None:
            mips = _mip_chain(projection.texture)
        if ray_mode:
            materials.append((geometry, rgba, mips))
        for tri in geometry.triangles:
            if ray_mode:
                primitive_index += 1
            zs = -local[tri, 2]
            if (zs <= camera.near).all() or (zs >= camera.far).all():
                continue
            if geometry.normals is not None:
                tri_normals = normals[tri]
            else:
                face = np.cross(world[tri[1]] - world[tri[0]], world[tri[2]] - world[tri[0]])
                tri_normals = np.broadcast_to(face / max(float(np.linalg.norm(face)), 1e-8), (3, 3))
            attributes = np.concatenate((local[tri], world[tri], tri_normals, uvs[tri]), axis=1)
            clipped_triangles = _clip_near(attributes.astype(np.float32), zs, camera.near)
            if ray_mode:
                ray_attributes[primitive_index] = attributes
                ray_object_ids[primitive_index] = object_id
            for piece, clipped in enumerate(clipped_triangles):
                z = -clipped[:, 2]
                if ray_mode:
                    a, b, c = _to_pixels(clipped[:, :3], z, focal, aspect, width, height)
                    den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
                    level = _triangle_mip(clipped, den, mips, projection)
                    if piece == 0:
                        ray_mip_levels[primitive_index] = level
                    else:
                        clipped_mips[primitive_index] = (clipped, level)
                else:
                    queue.append((float(z.mean()), clipped, z, rgba, mips, projection, geometry, object_id))
    shadow_context = None
    if shadow_triangles or ray_mode:
        triangles = np.concatenate(shadow_triangles) if shadow_triangles else np.empty((0, 3, 3), np.float32)
        if ray_mode:
            triangles = triangles.astype(np.float64)
        v0 = triangles[:, 0]
        e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
        alphas = np.concatenate(shadow_alphas) if shadow_alphas else np.empty(0, np.float32)
        primitives = TriangleSet(v0, e1, e2, alphas)
        bvh = (Bvh.build(*primitives.aabbs(), cancel=cancel)
               if ray_mode or triangle_count > _SHADOW_BRUTE_THRESHOLD else None)
        bias = 1e-3 * max(1.0, float(np.ptp(triangles.reshape(-1, 3), axis=0).max())) if len(triangles) else .001
        if shadow_active:
            work = _shadow_cost(width*height, triangle_count) if ray_mode else shadow_work
            shadow_context = _ShadowContext(primitives, bvh, bias, cancel, triangle_count, work, ray_mode)
    if ray_mode:
        primary = _render_mesh_layers if layered else _render_primary
        mesh_layers = primary(scene, camera, width, height, out, depth,
                        attributes=ray_attributes, object_ids=ray_object_ids,
                        mip_levels=ray_mip_levels, clipped_mips=clipped_mips, materials=materials,
                        primitives=primitives, bvh=bvh, eye=eye, view=view, focal=focal, aspect=aspect,
                        lights=lights, ambient=ambient, output=output, shade=shade,
                        shadow_context=shadow_context, cancel=cancel)
    for index, (_mean_z, tri, z, rgba, mips, projection, geometry, object_id) in enumerate(sorted(queue, key=lambda item: item[0], reverse=True)):
        if cancel is not None and index % 256 == 0 and cancel.is_set():
            from .imaging import Cancelled
            raise Cancelled()
        (a, b, c) = _to_pixels(tri[:, :3], z, focal, aspect, width, height)
        x0, x1 = max(0, int(math.floor(min(a[0], b[0], c[0])))), min(width - 1, int(math.ceil(max(a[0], b[0], c[0]))))
        y0, y1 = max(0, int(math.floor(min(a[1], b[1], c[1])))), min(height - 1, int(math.ceil(max(a[1], b[1], c[1]))))
        if x0 > x1 or y0 > y1:
            continue
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-8:
            continue
        ys, xs = np.mgrid[y0:y1+1, x0:x1+1]; px, py = xs + 0.5, ys + 0.5
        wa = ((b[1]-c[1])*(px-c[0]) + (c[0]-b[0])*(py-c[1])) / den
        wb = ((c[1]-a[1])*(px-c[0]) + (a[0]-c[0])*(py-c[1])) / den
        wc = 1 - wa - wb
        inside = np.ones(px.shape, dtype=bool)
        for start, end in ((b, c), (c, a), (a, b)):
            # Canonical endpoints make the edge value and its relative roundoff band
            # identical for neighbours, regardless of their third vertex or winding.
            forward = tuple(start) < tuple(end)
            lo, hi = (start, end) if forward else (end, start)
            lx, ly = map(float, lo)
            dx, dy = float(hi[0]) - lx, float(hi[1]) - ly
            edge = dx * (py - ly) - dy * (px - lx)
            tolerance = (8 * np.finfo(np.float64).eps * (abs(dx) + abs(dy)) *
                         max(width, height, abs(lx), abs(ly), abs(float(hi[0])), abs(float(hi[1]))))
            orientation = (1 if forward else -1) * (1 if den > 0 else -1)
            edge *= orientation
            ex, ey = dx * orientation, dy * orientation
            top_left = ey < 0 or (ey == 0 and ex > 0)  # screen Y points down
            # Outer silhouette edges obey the same rule: exact edge samples are
            # included only on top-left edges, not on bottom/right edges.
            inside &= (edge > tolerance) | ((np.abs(edge) <= tolerance) & top_left)
        if not inside.any():
            continue
        # Screen-space barycentrics interpolate 1/z linearly; dividing through recovers weights
        # that are correct in 3D, for depth and for every attribute.
        inverse = np.stack((wa / z[0], wb / z[1], wc / z[2]), -1)
        total = np.maximum(inverse.sum(-1), 1e-12)
        zbuf = 1.0 / total
        region_depth = depth[y0:y1+1, x0:x1+1]
        take = inside & (zbuf < region_depth) & (zbuf < camera.far)
        if not take.any():
            continue
        weights = (inverse / total[..., None])[take]            # (P,3)
        position = weights @ tri[:, 3:6]
        normal = weights @ tri[:, 6:9]
        uv = weights @ tri[:, 9:11]
        source, normal, uv = _shade_fragments(
            position, normal, uv, geometry=geometry, rgba=rgba, mips=mips,
            level=_triangle_mip(tri, den, mips, projection), eye=eye, lights=lights,
            ambient=ambient, output=output, shade=shade, scene=scene,
            projection_depth_maps=projection_depth_maps, shadow_context=shadow_context, cancel=cancel)
        src_alpha = source[:, 3]
        region = out[y0:y1+1, x0:x1+1]
        if output == "depth":
            opaque = src_alpha > 0
            pixels = region[take]
            pixels[opaque] = np.concatenate((np.repeat(zbuf[take][opaque, None], 3, 1),
                                             np.ones((int(opaque.sum()), 1), np.float32)), 1)
            region[take] = pixels
        elif output == "normals":
            opaque = src_alpha > 0
            pixels = region[take]
            pixels[opaque] = np.concatenate((normal[opaque], np.ones((int(opaque.sum()), 1), np.float32)), 1)
            region[take] = pixels
        elif data_output:
            opaque = src_alpha > 0
            pixels = region[take]
            if output == "position":
                values = position
            elif output == "uv":
                values = np.column_stack((uv, np.zeros(len(uv), np.float32)))
            else:  # object_id
                values = np.broadcast_to((object_id, 0, 0), (len(weights), 3))
            pixels[opaque] = np.column_stack((values[opaque], np.ones(int(opaque.sum()))))
            region[take] = pixels
        else:
            region[take] = source + region[take] * (1 - src_alpha[:, None])
        solid = take.copy()
        solid[take] = src_alpha >= 0.999
        if data_output:
            solid[take] = src_alpha > 0
        region_depth[solid] = zbuf[solid]
    if scene.splats and output == "rgba":
        from .splatraster import render_splats
        splat_rgb, splat_alpha = render_splats(
            scene.splats, camera, width, height,
            None if layered else depth, cancel=cancel, mesh_layers=mesh_layers,
            background_rgba=out if layered else None)
        if layered:
            out[:] = np.concatenate((splat_rgb, splat_alpha[..., None]), axis=2)
        else:
            out[:, :, :3] = splat_rgb + (1-splat_alpha[:, :, None])*out[:, :, :3]
            out[:, :, 3] = splat_alpha + (1-splat_alpha)*out[:, :, 3]
    out.flags.writeable = False
    if return_depth:
        depth.flags.writeable = False
        return out, depth
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
