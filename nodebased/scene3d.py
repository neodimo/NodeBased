"""Deterministic CPU reference renderer behind NodeBased's 3D nodes.

Typed scene values (geometry, lights, cameras, scenes) are assembled by the image evaluator and
rasterized here into premultiplied, scene-linear float32 RGBA. The rasterizer is NumPy on the
CPU: perspective-correct attributes, near-plane clipping, z-buffered opaque surfaces, sorted
transparency, mip-mapped bilinear textures, Lambert lighting, supersampled antialiasing and
depth/normal outputs. It is the correctness reference, not a throughput claim; ray tracing,
Gaussian splats, particles and fluids are roadmap stages (docs/3D_ROADMAP.md), not this module.

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

MAX_TRIANGLES = 250_000
SHADOW_WORK_BUDGET = 4_000_000_000
_SHADOW_RAY_CHUNK = 128
_SHADOW_TRIANGLE_CHUNK = 512
RENDER_OUTPUTS = ("rgba", "depth", "normals")
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
class Scene:
    geometries: tuple[Geometry, ...] = ()
    lights: tuple[Light, ...] = ()


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
                       Vec3(p["sx"], p["sy"], p["sz"]))


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
    """Assemble a Scene3D: geometry, lights and nested scenes, all under this node's transform."""
    matrix = _transform_from(node["params"]).matrix()
    geometries, lights = [], []
    for member in members:
        if isinstance(member, Scene):
            items = member.geometries + member.lights
        else:
            items = (member,)
        for item in items:
            moved = type(item)(**{**item.__dict__, "parent": matrix @ item.parent})
            (geometries if isinstance(item, Geometry) else lights).append(moved)
    return Scene(tuple(geometries), tuple(lights))


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


def _shadow_budget(work):
    if work > SHADOW_WORK_BUDGET:
        raise ValueError(f"Shadow rays exceed the CPU reference budget: {work} ray-triangle tests; "
                         "reduce resolution/samples/triangles or switch shadows off")


def _shadow_visibility(position, normal, light, light_position, direction,
                       v0, e1, e2, alpha, bias, cancel):
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
        transmission = np.ones(len(origin), np.float32)
        for first in range(0, len(v0), _SHADOW_TRIANGLE_CHUNK):
            _shadow_cancel(cancel)
            chunk = slice(first, first + _SHADOW_TRIANGLE_CHUNK)
            h = np.cross(ray[:, None, :], e2[chunk])
            det = np.einsum("rtj,tj->rt", h, e1[chunk])
            valid = np.abs(det) > 1e-10
            inverse = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
            delta = origin[:, None, :] - v0[chunk]
            u = np.einsum("rtj,rtj->rt", delta, h) * inverse
            q = np.cross(delta, e1[chunk])
            v = np.einsum("rj,rtj->rt", ray, q) * inverse
            t = np.einsum("tj,rtj->rt", e2[chunk], q) * inverse
            hit = valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > bias * .01) & (t < limit[:, None])
            transmission *= np.prod(np.where(hit, 1 - alpha[chunk], 1), axis=1)
        visibility[start:stop] = transmission
    return visibility


def render(scene: Scene, camera: Camera, width: int, height: int, background=(0., 0., 0., 0.),
           shade=False, return_depth=False, ambient=0.0, samples=1, output="rgba", cancel=None, *, shadows=True):
    """Rasterize a scene to premultiplied float32 scene-linear RGBA.

    Surfaces are unlit (their authored colour/texture) until the scene has lights; then they are
    Lambert-shaded by those lights plus ``ambient``. ``shade`` instead applies the viewport's
    fixed inspection headlight. ``samples`` is supersampling per axis for the rgba output.
    ``output`` "depth" writes view-space distance and "normals" world-space normals into RGB with
    coverage in alpha; both ignore the background and are never antialiased, because averaging
    depths or normals across an edge invents values that exist nowhere in the scene.
    With ``shadows=True``, enabled lights trace two-sided world-space triangle rays.
    Every geometry casts and receives shadows, including projected geometry. Visibility
    multiplies (1 - geometry alpha) over hits; texture alpha is NOT considered. Ambient,
    data outputs and inspection shading are unaffected.
    ``return_depth`` also returns the depth buffer (inf where empty).
    """
    _shadow_cancel(cancel)
    if output not in RENDER_OUTPUTS:
        raise ValueError(f"Unknown 3D render output {output!r}")
    width, height = int(width), int(height)
    samples = max(1, min(int(samples), 4)) if output == "rgba" else 1
    shadow_count = sum(light.shadows and light.intensity > 0 for light in scene.lights)
    shadow_active = shadows and not shade and output == "rgba" and shadow_count > 0
    triangle_count = sum(len(g.triangles) for g in scene.geometries)
    if shadow_active:
        # Estimate before framebuffer allocation; a running counter also bounds overdraw.
        _shadow_budget(width * height * samples ** 2 * shadow_count * triangle_count)
    if samples > 1:
        big = render(scene, camera, width * samples, height * samples, background, shade,
                     return_depth, ambient, 1, output, cancel, shadows=shadows)
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
    lit = bool(lights) and not shade
    # Collect clipped triangles, then rasterize far-to-near. Opaque pixels write the z buffer;
    # transparent ones only test it, so they reveal what is behind them and composite in depth
    # order. That is sorted transparency, not order-independent transparency: interpenetrating
    # transparent surfaces can still sort wrongly.
    # Per-vertex attribute layout: view xyz (3) | world xyz (3) | world normal (3) | uv (2).
    queue = []
    projection_depth_maps = {}
    shadow_triangles, shadow_alphas = [], []
    shadow_work = 0
    if sum(len(g.triangles) for g in scene.geometries) > MAX_TRIANGLES:
        raise ValueError(f"Scene exceeds {MAX_TRIANGLES} triangles; the CPU reference renderer refuses it")
    for geometry in scene.geometries:
        _shadow_cancel(cancel)
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        if shadow_active and len(geometry.triangles):
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
        for tri in geometry.triangles:
            zs = -local[tri, 2]
            if (zs <= camera.near).all() or (zs >= camera.far).all():
                continue
            if geometry.normals is not None:
                tri_normals = normals[tri]
            else:
                face = np.cross(world[tri[1]] - world[tri[0]], world[tri[2]] - world[tri[0]])
                tri_normals = np.broadcast_to(face / max(float(np.linalg.norm(face)), 1e-8), (3, 3))
            attributes = np.concatenate((local[tri], world[tri], tri_normals, uvs[tri]), axis=1)
            for clipped in _clip_near(attributes.astype(np.float32), zs, camera.near):
                z = -clipped[:, 2]
                queue.append((float(z.mean()), clipped, z, rgba, mips, projection, geometry))
    if shadow_triangles:
        triangles = np.concatenate(shadow_triangles)
        v0 = triangles[:, 0]
        e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
        alphas = np.concatenate(shadow_alphas)
        bias = 1e-3 * max(1.0, float(np.ptp(triangles.reshape(-1, 3), axis=0).max()))
    for index, (_mean_z, tri, z, rgba, mips, projection, geometry) in enumerate(sorted(queue, key=lambda item: item[0], reverse=True)):
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
        source = np.broadcast_to(rgba, (len(weights), 4)).copy()
        source[:, :3] *= source[:, 3:4]                          # premultiply the flat colour
        if mips is not None:
            uv = weights @ tri[:, 9:11]
            tri_uv = tri[:, 9:11]
            if projection is not None:
                uv, projection_depth = _projection_uv(projection, weights @ tri[:, 3:6])
                tri_uv, _ = _projection_uv(projection, tri[:, 3:6])
            # One mip level per triangle from its texel/pixel area ratio: a bounded, stable
            # approximation of footprint filtering that stops distant cards from shimmering.
            h, w = mips[0].shape[:2]
            (eu, ev), (fu, fv) = tri_uv[1] - tri_uv[0], tri_uv[2] - tri_uv[0]
            uv_area = abs(float(eu * fv - ev * fu)) * w * h
            level = int(np.clip(round(0.5 * math.log2(max(uv_area / max(abs(den), 1e-8), 1.0))), 0, len(mips) - 1))
            texel = _sample(mips[level], uv[:, 0], uv[:, 1])
            source = texel * np.append(rgba[:3] * rgba[3], rgba[3])  # tint premultiplied texels
        normal = weights @ tri[:, 6:9]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        position = weights @ tri[:, 3:6]
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
        emissive = source[:, :3] * geometry.emission if geometry.emission and output == "rgba" else None
        if shade:
            source[:, :3] *= (0.25 + 0.75 * np.abs(normal @ _VIEW_LIGHT))[:, None]
        elif lit:
            radiance = np.full((len(weights), 3), float(ambient), np.float32)
            specular = np.zeros_like(radiance) if geometry.specular and output == "rgba" else None
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
                if shadow_active and light.shadows and shadow_triangles:
                    shadow_work += len(position) * triangle_count
                    _shadow_budget(shadow_work)
                    visibility = _shadow_visibility(position, normal, light, light_position, direction,
                                                    v0, e1, e2, alphas, bias, cancel)
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
        else:
            region[take] = source + region[take] * (1 - src_alpha[:, None])
        solid = take.copy()
        solid[take] = src_alpha >= 0.999
        if output != "rgba":
            solid[take] = src_alpha > 0
        region_depth[solid] = zbuf[solid]
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
