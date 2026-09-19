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

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import math

import numpy as np

MAX_TRIANGLES = 250_000
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
class Scene:
    geometries: tuple[Geometry, ...] = ()
    lights: tuple[Light, ...] = ()


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
    color = (float(p["red"]), float(p["green"]), float(p["blue"]), float(p["alpha"]))
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
                 Vec3(p["target_x"], p["target_y"], p["target_z"]))


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


def render(scene: Scene, camera: Camera, width: int, height: int, background=(0., 0., 0., 0.),
           shade=False, return_depth=False, ambient=0.0, samples=1, output="rgba", cancel=None):
    """Rasterize a scene to premultiplied float32 scene-linear RGBA.

    Surfaces are unlit (their authored colour/texture) until the scene has lights; then they are
    Lambert-shaded by those lights plus ``ambient``. ``shade`` instead applies the viewport's
    fixed inspection headlight. ``samples`` is supersampling per axis for the rgba output.
    ``output`` "depth" writes view-space distance and "normals" world-space normals into RGB with
    coverage in alpha; both ignore the background and are never antialiased, because averaging
    depths or normals across an edge invents values that exist nowhere in the scene.
    ``return_depth`` also returns the depth buffer (inf where empty).
    """
    if output not in RENDER_OUTPUTS:
        raise ValueError(f"Unknown 3D render output {output!r}")
    width, height = int(width), int(height)
    samples = max(1, min(int(samples), 4)) if output == "rgba" else 1
    if samples > 1:
        big = render(scene, camera, width * samples, height * samples, background, shade,
                     return_depth, ambient, 1, output, cancel)
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
    if sum(len(g.triangles) for g in scene.geometries) > MAX_TRIANGLES:
        raise ValueError(f"Scene exceeds {MAX_TRIANGLES} triangles; the CPU reference renderer refuses it")
    for geometry in scene.geometries:
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        local = (view @ (world - eye).T).T
        if geometry.normals is not None:
            normal_matrix = np.linalg.inv(matrix[:3, :3]).T
            normals = (normal_matrix @ geometry.normals.T).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        uvs = geometry.uvs if geometry.uvs is not None else np.zeros((len(world), 2), np.float32)
        rgba = np.asarray(geometry.color, np.float32)
        mips = _mip_chain(geometry.texture) if geometry.texture is not None and geometry.uvs is not None else None
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
                queue.append((float(z.mean()), clipped, z, rgba, mips))
    for index, (_mean_z, tri, z, rgba, mips) in enumerate(sorted(queue, key=lambda item: item[0], reverse=True)):
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
        inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
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
            # One mip level per triangle from its texel/pixel area ratio: a bounded, stable
            # approximation of footprint filtering that stops distant cards from shimmering.
            h, w = mips[0].shape[:2]
            (eu, ev), (fu, fv) = tri[1, 9:11] - tri[0, 9:11], tri[2, 9:11] - tri[0, 9:11]
            uv_area = abs(float(eu * fv - ev * fu)) * w * h
            level = int(np.clip(round(0.5 * math.log2(max(uv_area / max(abs(den), 1e-8), 1.0))), 0, len(mips) - 1))
            texel = _sample(mips[level], uv[:, 0], uv[:, 1])
            source = texel * np.append(rgba[:3] * rgba[3], rgba[3])  # tint premultiplied texels
        normal = weights @ tri[:, 6:9]
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        position = weights @ tri[:, 3:6]
        if lit or shade or output == "normals":
            toward_eye = eye - position
            normal = np.where((np.einsum("ij,ij->i", normal, toward_eye) < 0)[:, None], -normal, normal)
        if shade:
            source[:, :3] *= (0.25 + 0.75 * np.abs(normal @ _VIEW_LIGHT))[:, None]
        elif lit:
            radiance = np.full((len(weights), 3), float(ambient), np.float32)
            for light, light_position, direction in lights:
                if light.kind == "Point":
                    to_light = light_position - position
                    to_light /= np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-8)
                    lambert = np.einsum("ij,ij->i", normal, to_light)
                else:
                    lambert = normal @ -direction
                radiance += np.maximum(lambert, 0)[:, None] * (np.asarray(light.color, np.float32) * light.intensity)
            source[:, :3] *= radiance
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
