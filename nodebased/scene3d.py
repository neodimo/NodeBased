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

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
import hashlib
import itertools
import math
import os
import tempfile
import threading
import weakref

import numpy as np

from .raytrace import Bvh, TriangleSet, SplatSet
from .splats import SplatCloud
from . import filmback as _fb

MAX_TRIANGLES = 250_000
# Default ray-origin offset, as a fraction of the scene extent (at least one unit): the epsilon the
# shadow code has always used. A light's `shadow_bias` replaces it; the default leaves renders unchanged.
SHADOW_BIAS_DEFAULT = 1e-3
SHADOW_SAMPLES_MAX = 64
SHADOW_WORK_BUDGET = 4_000_000_000
RAYTRACE_WORK_BUDGET = 4_000_000_000
SPLAT_SHADOW_BUDGET = RAYTRACE_WORK_BUDGET
# Shadows darker than 0.1% are treated as fully dark.
SPLAT_SHADOW_CUTOFF = 1e-3
_PRIMARY_RAY_CHUNK = 4096
# Bounds shaded surfaces, including alpha-zero surfaces, through termination.
MAX_HITS_PER_RAY = 64
MAX_MESH_LAYERS = 16
LAYER_BAND_BYTES = 64 * 1024 * 1024
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
# Splat shadow rays are queried in blocks this size; SplatSet.transmittance chunks internally as well.
_SPLAT_SHADOW_QUERY_CHUNK = 1024
_SHADOW_TRIANGLE_CHUNK = 512
RENDER_OUTPUTS = ("rgba", "depth", "normals", "albedo", "diffuse", "specular",
                  "emission", "position", "uv", "object_id", "relight", "splats")
LIGHT_OUTPUTS = ("rgba", "albedo", "diffuse", "specular", "emission", "splats")
DATA_OUTPUTS = ("depth", "normals", "position", "uv", "object_id")
LIGHT_TYPES = ("Directional", "Point", "Spot")
FALLOFF_TYPES = ("No falloff", "Linear", "Quadratic", "Cubic")
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
    # Spot cone and distance falloff (Nuke's Light knobs); see `light_attenuation`.
    cone_angle: float = 30.0            # full cone, degrees, at full intensity
    cone_penumbra_angle: float = 5.0    # extra degrees on each side over which the edge fades out
    cone_falloff: float = 1.0           # >1 fades faster across the penumbra, <1 slower
    falloff_type: str = "No falloff"    # distance falloff for Point and Spot
    # Shadow knobs (Nuke's Light: shadow bias and the soft-shadow samples/size); see `_shadow_trace`.
    shadow_bias: float = SHADOW_BIAS_DEFAULT   # ray origin offset along the normal, x scene extent (min 1 unit)
    shadow_blur: float = 0.0            # light half-angle in degrees as seen from the surface; 0 = hard
    shadow_samples: int = 1             # jittered shadow rays per shading point when blur > 0

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
    fov: float = 45.0   # vertical, degrees; what every renderer reads
    near: float = 0.1
    far: float = 1000.0
    roll: float = 0.0   # degrees about the view axis
    # Film back in millimetres (Nuke's model). `fov` is the one stored lens value, so the focal
    # length below is derived from it and the vertical aperture and can never disagree.
    haperture: float = _fb.DEFAULT_HAPERTURE
    vaperture: float = _fb.DEFAULT_VAPERTURE

    @property
    def focal(self):
        return _fb.focal_from_fov(self.fov, self.vaperture)

    @property
    def hfov(self):
        """Horizontal field of view in degrees across the horizontal aperture."""
        return _fb.fov_from_aperture(self.focal, self.haperture)


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
    relight: float = 0.0
    shadow_catch: float = 0.0  # meshes darken the captured colour; the capture's own look is kept
    cast_shadows: bool = True  # off for environments: a capture's sky shell otherwise blocks every light
    specular: float = 0.0      # keep the capture's own highlights (SH beyond DC) through relighting and catching


@dataclass(frozen=True, eq=False)
class ParticleInstance:
    """One solved frame of a particle system, in the space of `matrix` (docs/SIMULATION.md).

    `positions` (N,3), `sizes` (N,) world-space diameters and premultiplied `colors` (N,4) are what
    the renderer reads; the rest is solver data kept for later passes. `stream` and `frame` name the
    run this frame came from so ParticleCache3D can re-solve it through a persistent cache.
    """
    positions: np.ndarray
    sizes: np.ndarray
    colors: np.ndarray
    matrix: np.ndarray = field(default_factory=lambda: _IDENTITY)
    render_as: str = "points"        # "points" flat discs, "spheres" shaded discs, "cards" camera-facing squares
    size_scale: float = 1.0          # multiplies `sizes` at draw time (ParticleRender3D), never the solve
    texture: np.ndarray | None = None  # premultiplied float32 RGBA sprite for cards, row 0 at the top
    velocities: np.ndarray | None = None
    ages: np.ndarray | None = None       # frames since birth
    lifetimes: np.ndarray | None = None  # frames from birth to death
    ids: np.ndarray | None = None
    stream: object | None = None
    frame: int = 0

    def __len__(self):
        return len(self.positions)


@dataclass(frozen=True, eq=False)
class Scene:
    geometries: tuple[Geometry, ...] = ()
    lights: tuple[Light, ...] = ()
    splats: tuple = ()
    particles: tuple = ()


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

def _card(width, height, color, transform, texture=None, rows=1, columns=1):
    """A flat XY card, optionally subdivided into a `rows` x `columns` grid of quads.

    `rows`/`columns` both at 1 (the default) returns the exact single-quad geometry NodeBased
    has always produced, so every existing Card3D document and pixel test is untouched.
    """
    w, h = float(width) / 2, float(height) / 2
    rows, columns = max(1, int(rows)), max(1, int(columns))
    if rows == 1 and columns == 1:
        return Geometry(np.array(((-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)), np.float32),
                        np.array(((0, 1, 2), (0, 2, 3)), np.int32), color, transform,
                        uvs=np.array(((0, 0), (1, 0), (1, 1), (0, 1)), np.float32), texture=texture)
    xs, ys = np.linspace(-w, w, columns + 1), np.linspace(-h, h, rows + 1)
    xu, yu = np.meshgrid(xs, ys)
    vertices = np.stack((xu, yu, np.zeros_like(xu)), -1).reshape(-1, 3).astype(np.float32)
    uu, vu = np.meshgrid(np.linspace(0, 1, columns + 1), np.linspace(0, 1, rows + 1))
    uvs = np.stack((uu, vu), -1).reshape(-1, 2).astype(np.float32)
    tris = []
    for r in range(rows):
        for c in range(columns):
            a, b = r * (columns + 1) + c, r * (columns + 1) + c + 1
            d, e = a + columns + 1, b + columns + 1
            tris += [(a, b, e), (a, e, d)]
    return Geometry(vertices, np.array(tris, np.int32), color, transform, uvs=uvs, texture=texture)


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


def _sphere_grid(radius, rows, columns, color, transform, texture=None):
    """Sphere3D's Nuke-style rows (latitude bands) / columns (longitude segments) knobs.

    Same vertex/UV/normal generation as `_sphere` (so the two agree vertex-for-vertex at
    matching resolutions), but the pole rings emit only the one valid triangle per column
    instead of two: at the south pole (r == 0) the first two corners of every quad coincide,
    and at the north pole (r == rows - 1) the last two do, so one of the two triangles a
    plain quad-split would emit there always has zero area. Skipping it leaves no degenerate
    or duplicated pole triangles while the vertex positions and unit normals are unchanged.
    """
    cols, rows = max(3, int(columns)), max(2, int(rows))
    u, v = np.meshgrid(np.linspace(0, 1, cols + 1), np.linspace(0, 1, rows + 1))
    theta, phi = u * 2 * np.pi, (v - 0.5) * np.pi
    unit = np.stack((np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)), -1).reshape(-1, 3)
    tris = []
    for r in range(rows):
        for c in range(cols):
            a, b = r * (cols + 1) + c, r * (cols + 1) + c + 1
            d, e = a + cols + 1, b + cols + 1
            if r > 0:
                tris.append((a, b, e))
            if r < rows - 1:
                tris.append((a, e, d))
    return Geometry((unit * float(radius)).astype(np.float32), np.array(tris, np.int32), color, transform,
                    uvs=np.stack((u, v), -1).reshape(-1, 2).astype(np.float32),
                    normals=unit.astype(np.float32), texture=texture)


def _cylinder(radius, height, rows, columns, closed, color, transform, texture=None):
    """A cylinder along +Y: `columns` radial segments, `rows` segments along the height,
    optional flat end caps (`closed`) fanned from a centre vertex at each end."""
    cols, rows = max(3, int(columns)), max(1, int(rows))
    h = float(height) / 2
    theta, y = np.meshgrid(np.linspace(0, 2 * np.pi, cols + 1), np.linspace(-h, h, rows + 1))
    x, z = radius * np.cos(theta), radius * np.sin(theta)
    vertices = np.stack((x, y, z), -1).reshape(-1, 3).astype(np.float32)
    uu, vv = np.meshgrid(np.linspace(0, 1, cols + 1), np.linspace(0, 1, rows + 1))
    uvs = np.stack((uu, vv), -1).reshape(-1, 2).astype(np.float32)
    normals = np.stack((np.cos(theta), np.zeros_like(theta), np.sin(theta)), -1).reshape(-1, 3).astype(np.float32)
    tris = []
    for r in range(rows):
        for c in range(cols):
            a, b = r * (cols + 1) + c, r * (cols + 1) + c + 1
            d, e = a + cols + 1, b + cols + 1
            tris += [(a, b, e), (a, e, d)]
    if closed:
        bottom_ring = [c for c in range(cols)]
        top_ring = [rows * (cols + 1) + c for c in range(cols)]
        bottom_center, top_center = len(vertices), len(vertices) + 1
        vertices = np.vstack((vertices, [[0, -h, 0], [0, h, 0]])).astype(np.float32)
        normals = np.vstack((normals, [[0, -1, 0], [0, 1, 0]])).astype(np.float32)
        uvs = np.vstack((uvs, [[0.5, 0.5], [0.5, 0.5]])).astype(np.float32)
        for c in range(cols):
            tris.append((bottom_center, bottom_ring[(c + 1) % cols], bottom_ring[c]))
            tris.append((top_center, top_ring[c], top_ring[(c + 1) % cols]))
    return Geometry(vertices, np.array(tris, np.int32), color, transform, uvs=uvs, normals=normals, texture=texture)


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


def node_parent_chain(document, key):
    """Ancestor node keys for `key`, nearest first, found by walking every Scene3D's
    object0..7 slots and Axis3D's object slot. A node wired into more than one parent keeps
    whichever parent is discovered first (dict iteration order): a read-only matrix readout
    is a graph-structure preview, not a claim that a fanned-out node has one true parent."""
    nodes = document.get("nodes", {})
    parent_of = {}
    for pkey, node in nodes.items():
        kind = node.get("type")
        if kind == "Scene3D":
            for index in range(8):
                child = node.get("inputs", {}).get(f"object{index}")
                if child is not None and child not in parent_of:
                    parent_of[child] = pkey
        elif kind == "Axis3D":
            child = node.get("inputs", {}).get("object")
            if child is not None and child not in parent_of:
                parent_of[child] = pkey
    chain, current, seen = [], key, {key}
    while current in parent_of and parent_of[current] not in seen:
        current = parent_of[current]
        seen.add(current)
        chain.append(current)
    return chain


def _own_matrix(node):
    """One node's own local transform matrix, for the read-only matrix readouts.

    Camera3D and Light3D have no rx/ry/rz/scale/pivot knobs (they aim through target_x/y/z
    instead), so their local matrix is translation-only, matching `camera_from_node` and
    `light_from_node`. A disabled Axis3D parents at the identity, matching the evaluator
    (`imaging.py`'s `_IDENTITY_XFORM if node["disabled"]`); other kinds show their transform
    as authored regardless of `disabled`, since this is a structural preview, not a render.
    """
    params = node.get("params", {})
    kind = node.get("type")
    if kind == "Axis3D" and node.get("disabled"):
        return _IDENTITY.copy()
    if kind in ("Camera3D", "Light3D"):
        return Transform3D(Vec3(params.get("tx", 0.0), params.get("ty", 0.0), params.get("tz", 0.0))).matrix()
    return _transform_from(params).matrix()


def local_and_world_matrix(document, key):
    """(local, world) float32 4x4 matrices for a transform-carrying node, for the properties
    panel's read-only matrix readouts. `local` is the node's own transform; `world`
    multiplies in every Scene3D/Axis3D ancestor's own transform (nearest first), matching how
    `scene_from_node` accumulates a parent matrix at render time. Pass a frame-resolved
    document (curves/expressions applied) so the readout matches what actually renders."""
    nodes = document.get("nodes", {})
    node = nodes.get(key)
    if node is None:
        return _IDENTITY.copy(), _IDENTITY.copy()
    local = _own_matrix(node)
    parent = _IDENTITY.copy()
    for ancestor_key in reversed(node_parent_chain(document, key)):
        parent = parent @ _own_matrix(nodes[ancestor_key])
    return local, parent @ local


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
        return _card(p["card_width"], p["card_height"], color, transform, texture,
                    rows=p.get("rows", 1), columns=p.get("columns", 1))
    if kind == "Cube3D":
        return _cube(p["cube_size"], color, transform, texture)
    if kind == "Sphere3D":
        return _sphere_grid(p["sphere_radius"], p.get("rows", 16), p.get("columns", 32), color, transform, texture)
    if kind == "Cylinder3D":
        return _cylinder(p["cyl_radius"], p["cyl_height"], p.get("rows", 1), p.get("columns", 24),
                         p.get("cyl_caps", "closed") == "closed", color, transform, texture)
    if kind == "ReadGeo3D":
        resolved, size, mtime = obj_fingerprint(p["geo_path"])
        vertices, triangles, uvs, normals = _load_obj(resolved, size, mtime)
        return Geometry(vertices, triangles, color, transform, uvs=uvs, normals=normals, texture=texture)
    raise ValueError(f"{kind} is not a geometry node")


def transform_geometry(geometry: Geometry, transform: Transform3D) -> Geometry:
    """Bake `transform` into `geometry`'s own vertices and normals (TransformGeo3D).

    Unlike Axis3D, which only ever adds another parent matrix, this rewrites the geometry's own
    object-space vertices, so it acts before the geometry's existing `transform`/`parent` at
    render time. Normals go through the inverse transpose (the same rule `write_obj` uses) and are
    renormalized; positions go through the full affine matrix, translation included.
    """
    matrix = transform.matrix()
    linear = matrix[:3, :3]
    vertices = (linear @ geometry.vertices.T).T + matrix[:3, 3]
    normals = geometry.normals
    if normals is not None:
        normals = (np.linalg.inv(linear).T @ normals.T).T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        normals = normals.astype(np.float32)
    return replace(geometry, vertices=vertices.astype(np.float32), normals=normals)


def empty_geometry() -> Geometry:
    """A geometry with no vertices: what MergeGeo3D produces when nothing is wired into it."""
    return Geometry(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), (0.8, 0.8, 0.8, 1.0))


def merge_geometry(geometries, transform: Transform3D | None = None) -> Geometry:
    """Concatenate `geometries` into ONE geometry (MergeGeo3D).

    Each input's full world matrix (its own `transform` and any enclosing `parent`) is baked into
    its vertices, so the result sits in world space under an identity transform; `transform` (the
    MergeGeo3D node's own) is then baked on top exactly as TransformGeo3D would. Normals go through
    the inverse transpose; a mirroring matrix (negative determinant) reverses that input's winding
    so its faces keep facing outward. Indices are offset per input and UVs concatenated. If some
    inputs carry normals and others do not, the ones without get smooth normals from
    `recompute_normals`; if none carry normals the result has none (flat face normals, as before).
    A Geometry holds ONE colour, texture, material and projection, so those come from the first
    input; the others' are dropped. No inputs gives an empty geometry.
    """
    geometries = [g for g in geometries if g is not None]
    if not geometries:
        return empty_geometry()
    any_normals = any(g.normals is not None for g in geometries)
    any_uvs = any(g.uvs is not None for g in geometries)
    vertices, triangles, normals, uvs, offset = [], [], [], [], 0
    for g in geometries:
        matrix = g.world_matrix().astype(np.float64)
        linear = matrix[:3, :3]
        verts = (linear @ g.vertices.astype(np.float64).T).T + matrix[:3, 3]
        tris = g.triangles.astype(np.int64)
        if np.linalg.det(linear) < 0:
            tris = tris[:, ::-1]
        if any_normals:
            source = g.normals if g.normals is not None else recompute_normals(g).normals
            n = (np.linalg.inv(linear).T @ source.astype(np.float64).T).T if len(source) else source
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8) if len(n) else n
            normals.append(n)
        if any_uvs:
            uvs.append(g.uvs if g.uvs is not None else np.zeros((len(g.vertices), 2), np.float32))
        vertices.append(verts)
        triangles.append(tris + offset)
        offset += len(g.vertices)
    first = geometries[0]
    merged = replace(
        first, vertices=np.concatenate(vertices).astype(np.float32),
        triangles=np.concatenate(triangles).astype(np.int32),
        normals=np.concatenate(normals).astype(np.float32) if any_normals else None,
        uvs=np.concatenate(uvs).astype(np.float32) if any_uvs else None,
        transform=Transform3D(), parent=_IDENTITY)
    return merged if transform is None else transform_geometry(merged, transform)


def _face_cross(vertices, triangles):
    v = vertices.astype(np.float64)
    return np.cross(v[triangles[:, 1]] - v[triangles[:, 0]], v[triangles[:, 2]] - v[triangles[:, 0]])


def _position_groups(vertices):
    """Index of the welded position each vertex sits on (seams and poles share one)."""
    if not len(vertices):
        return np.zeros(0, np.int64)
    extent = float(np.ptp(vertices, axis=0).max())
    step = max(extent, 1e-6) * 1e-5
    _, inverse = np.unique(np.round(vertices.astype(np.float64) / step).astype(np.int64),
                           axis=0, return_inverse=True)
    return inverse.reshape(-1)


def recompute_normals(geometry: Geometry, crease_degrees: float = 50.0) -> Geometry:
    """Smooth per-vertex normals from face normals, area weighted (Normals3D "recompute").

    Faces meeting at one vertex contribute in proportion to their area. Vertices that share a
    position (a UV seam, a sphere pole) are smoothed together, but only with faces within
    `crease_degrees` of that vertex's own faces, so a cube built from per-face vertices stays
    flat while a sphere's seam and poles come out radial.
    """
    vertices, triangles = geometry.vertices, geometry.triangles.astype(np.int64)
    count = len(vertices)
    if not len(triangles):
        return replace(geometry, normals=np.zeros((count, 3), np.float32) + (0, 0, 1))
    cross = _face_cross(vertices, triangles)          # length = 2 x area, direction = face normal
    own = np.zeros((count, 3))
    for corner in range(3):
        np.add.at(own, triangles[:, corner], cross)
    result = own.copy()
    group = _position_groups(vertices)
    sizes = np.bincount(group)
    shared = np.flatnonzero(sizes > 1)
    if len(shared):
        corner_vertex = triangles.reshape(-1)
        corner_face = np.repeat(np.arange(len(triangles)), 3)
        corner_group = group[corner_vertex]
        corner_order = np.argsort(corner_group, kind="stable")
        corner_bounds = np.searchsorted(corner_group[corner_order], np.arange(len(sizes) + 1))
        vertex_order = np.argsort(group, kind="stable")
        vertex_bounds = np.searchsorted(group[vertex_order], np.arange(len(sizes) + 1))
        cosine = math.cos(math.radians(crease_degrees))
        lengths = np.linalg.norm(cross, axis=1)
        for g in shared:
            faces = np.unique(corner_face[corner_order[corner_bounds[g]:corner_bounds[g + 1]]])
            faces = faces[lengths[faces] > 1e-12]
            if not len(faces):
                continue
            units = cross[faces] / lengths[faces, None]
            for vertex in vertex_order[vertex_bounds[g]:vertex_bounds[g + 1]]:
                reference = own[vertex]
                norm = np.linalg.norm(reference)
                if norm < 1e-12:
                    continue
                keep = units @ (reference / norm) >= cosine
                if keep.any():
                    result[vertex] = cross[faces[keep]].sum(axis=0)
    length = np.linalg.norm(result, axis=1, keepdims=True)
    fallback = geometry.normals.astype(np.float64) if geometry.normals is not None else np.tile((0.0, 0.0, 1.0), (count, 1))
    result = np.where(length > 1e-12, result / np.maximum(length, 1e-12), fallback)
    return replace(geometry, normals=result.astype(np.float32))


def flip_geometry(geometry: Geometry) -> Geometry:
    """Reverse every face's winding and negate the normals, so back faces stay consistent."""
    normals = None if geometry.normals is None else (-geometry.normals).astype(np.float32)
    return replace(geometry, triangles=np.ascontiguousarray(geometry.triangles[:, ::-1]), normals=normals)


def unify_winding(geometry: Geometry) -> Geometry:
    """Make every connected surface's winding agree across shared edges, then recompute normals.

    Adjacency follows welded positions (a UV seam does not split a surface). Each connected
    component is grown from its first face; a closed component is then turned outward by its
    signed volume, an open one keeps the orientation of its first face.
    """
    triangles = geometry.triangles.astype(np.int64).copy()
    if not len(triangles):
        return recompute_normals(geometry)
    group = _position_groups(geometry.vertices)[triangles]      # (M, 3) welded corner ids
    edges = {}
    for face in range(len(triangles)):
        for k in range(3):
            a, b = int(group[face, k]), int(group[face, (k + 1) % 3])
            if a != b:
                edges.setdefault((min(a, b), max(a, b)), []).append((face, a < b))
    neighbours = [[] for _ in range(len(triangles))]
    for uses in edges.values():
        for i, (face, forward) in enumerate(uses):
            for other, other_forward in uses[i + 1:]:
                # Two faces traversing a shared edge in the same direction disagree.
                neighbours[face].append((other, forward == other_forward))
                neighbours[other].append((face, forward == other_forward))
    flipped = np.zeros(len(triangles), bool)
    seen = np.zeros(len(triangles), bool)
    volumes = _face_cross(geometry.vertices, triangles)
    centre = geometry.vertices.astype(np.float64).mean(axis=0) if len(geometry.vertices) else 0.0
    corner0 = geometry.vertices.astype(np.float64)[triangles[:, 0]] - centre
    for start in range(len(triangles)):
        if seen[start]:
            continue
        component, stack = [start], [start]
        seen[start] = True
        while stack:
            face = stack.pop()
            for other, disagree in neighbours[face]:
                if not seen[other]:
                    seen[other] = True
                    flipped[other] = flipped[face] ^ disagree
                    component.append(other)
                    stack.append(other)
        members = np.array(component)
        closed = all(len(edges[(min(int(group[f, k]), int(group[f, (k + 1) % 3])),
                                max(int(group[f, k]), int(group[f, (k + 1) % 3])))]) == 2
                     for f in members for k in range(3) if group[f, k] != group[f, (k + 1) % 3])
        if closed:
            sign = np.where(flipped[members], -1.0, 1.0)
            volume = float((sign * np.einsum("ij,ij->i", corner0[members], volumes[members])).sum())
            if volume < 0:
                flipped[members] = ~flipped[members]
    triangles[flipped] = triangles[flipped][:, ::-1]
    return recompute_normals(replace(geometry, triangles=triangles.astype(np.int32)))


def normals_from_node(geometry: Geometry, params) -> Geometry:
    """Apply Normals3D: the `normals_mode`, then the optional `flip_winding` reversal."""
    mode = params.get("normals_mode", "recompute")
    if mode == "recompute":
        geometry = recompute_normals(geometry)
    elif mode == "flip":
        geometry = flip_geometry(geometry)
    elif mode == "unify":
        geometry = unify_winding(geometry)
    if params.get("flip_winding", 0):
        geometry = flip_geometry(geometry)
    return geometry


def displace_geometry(geometry: Geometry, texture, scale: float, offset: float,
                      channel: str = "luminance", recompute: bool = True) -> Geometry:
    """Move each vertex along its normal by `scale` * (image channel at the vertex UV) + `offset`.

    `texture` is a premultiplied float RGBA array (row 0 at the top) or None; with None, or a
    geometry without UVs, only `offset` applies. Colour channels are un-premultiplied first and
    luminance is Rec. 709. The normals used are the geometry's own, or smooth recomputed ones when
    it has none. With `recompute` the result's normals are recomputed from the displaced surface.
    """
    if not len(geometry.vertices):
        return geometry
    normals = geometry.normals if geometry.normals is not None else recompute_normals(geometry).normals
    amount = np.zeros(len(geometry.vertices), np.float64)
    if texture is not None and geometry.uvs is not None:
        texels = _sample(np.asarray(texture, np.float32), geometry.uvs[:, 0], geometry.uvs[:, 1]).astype(np.float64)
        alpha = texels[:, 3]
        if channel == "alpha":
            values = alpha
        else:
            colour = texels[:, :3] / np.maximum(alpha, 1e-8)[:, None]
            colour = np.where(alpha[:, None] > 1e-8, colour, 0.0)
            values = (colour @ (0.2126, 0.7152, 0.0722) if channel == "luminance"
                      else colour[:, ("red", "green", "blue").index(channel)])
        amount = values
    moved = geometry.vertices.astype(np.float64) + normals.astype(np.float64) * (
        float(scale) * amount + float(offset))[:, None]
    result = replace(geometry, vertices=moved.astype(np.float32),
                     normals=geometry.normals if geometry.normals is not None else None)
    return recompute_normals(result) if recompute else result


def light_from_node(node):
    p = node["params"]
    return Light(p["light_type"], (float(p["red"]), float(p["green"]), float(p["blue"])),
                 float(p["intensity"]), Vec3(p["tx"], p["ty"], p["tz"]),
                 Vec3(p["target_x"], p["target_y"], p["target_z"]),
                 shadows=p.get("shadows", "off") == "on",
                 cone_angle=float(p.get("cone_angle", 30.0)),
                 cone_penumbra_angle=float(p.get("cone_penumbra_angle", 5.0)),
                 cone_falloff=float(p.get("cone_falloff", 1.0)),
                 falloff_type=p.get("falloff_type", "No falloff"),
                 shadow_bias=float(p.get("shadow_bias", SHADOW_BIAS_DEFAULT)),
                 shadow_blur=float(p.get("shadow_blur", 0.0)),
                 shadow_samples=int(p.get("shadow_samples", 1)))


def light_attenuation(light, world_point):
    """Scalar factor in [0, 1] a light's cone and distance falloff apply at world point(s).

    `world_point` is (3,) or (N, 3); the result is a float or an (N,) float32 array. A Directional
    light is 1 everywhere. A Point light applies only the distance falloff. A Spot light multiplies
    the falloff by its cone: 1 inside the inner half-angle (`cone_angle / 2`), 0 beyond
    `cone_angle / 2 + cone_penumbra_angle`, a smoothstep between them raised to `cone_falloff`.
    Distance falloff is 1 / d, 1 / d^2 or 1 / d^3 for Linear, Quadratic and Cubic, capped at 1 for
    distances under one unit so the factor never brightens a light. Pure geometry: no shading, no
    shadows.
    """
    points = np.atleast_2d(np.asarray(world_point, np.float64))
    if light.kind == "Directional":
        result = np.ones(len(points), np.float32)
    else:
        position, direction = light.world()
        offset = points - position.astype(np.float64)
        distance = np.linalg.norm(offset, axis=1)
        power = int(_falloff_power(light))
        result = np.minimum(1.0, np.maximum(distance, 1e-8) ** -power) if power else np.ones(len(points))
        if light.kind == "Spot":
            cosine = (offset @ direction.astype(np.float64)) / np.maximum(distance, 1e-12)
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            inner = float(light.cone_angle) / 2
            outer = inner + max(float(light.cone_penumbra_angle), 0.0)
            if outer > inner:
                t = np.clip((outer - angle) / (outer - inner), 0.0, 1.0)
                cone = t * t * (3.0 - 2.0 * t)
                cone = np.where(t > 0.0, cone ** max(float(light.cone_falloff), 0.0), 0.0)
            else:
                cone = (angle <= inner).astype(np.float64)
            result = result * cone
        result = result.astype(np.float32)
    return float(result[0]) if np.ndim(world_point) == 1 else result


_POSITIONAL = ("Point", "Spot")   # lights whose direction is toward a position, not a constant


def _falloff_power(light):
    return float({"No falloff": 0, "Linear": 1, "Quadratic": 2, "Cubic": 3}[light.falloff_type])


def _cone_terms(light):
    """(is spot, inner half-angle, outer half-angle, exponent): the cone as the GPU shaders take it."""
    if light.kind != "Spot":
        return (0.0, 0.0, 0.0, 1.0)
    inner = float(light.cone_angle) / 2
    return (1.0, inner, inner + max(float(light.cone_penumbra_angle), 0.0), max(float(light.cone_falloff), 0.0))


def _light_factor(light, points):
    """`light_attenuation` for shading, or None when it is 1 everywhere (Directional, plain Point)."""
    if light.kind == "Directional" or (light.kind == "Point" and light.falloff_type == "No falloff"):
        return None
    return light_attenuation(light, points)


def camera_from_node(node):
    p = node["params"]
    return Camera(Transform3D(Vec3(p["tx"], p["ty"], p["tz"])),
                  Vec3(p["target_x"], p["target_y"], p["target_z"]),
                  _fb.fov_from_aperture(p["focal"], p["vaperture"]), p["near"], p["far"],
                  p["roll"], float(p["haperture"]), float(p["vaperture"]))


def scene_from_node(node, members):
    """Assemble geometry, lights, splats and nested scenes under this node's transform."""
    matrix = _transform_from(node["params"]).matrix()
    geometries, lights, splats, particles = [], [], [], []
    for member in members:
        if isinstance(member, Scene):
            items = member.geometries + member.lights + member.splats + member.particles
        else:
            items = (member,)
        for item in items:
            if isinstance(item, SplatInstance):
                splats.append(replace(item, matrix=matrix @ item.matrix))
                continue
            if isinstance(item, ParticleInstance):
                particles.append(replace(item, matrix=matrix @ item.matrix))
                continue
            moved = type(item)(**{**item.__dict__, "parent": matrix @ item.parent})
            (geometries if isinstance(item, Geometry) else lights).append(moved)
    return Scene(tuple(geometries), tuple(lights), tuple(splats), tuple(particles))


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


def _point_seeds(points):
    """Stable per-point uint32 hash of float32 world positions (independent of chunking or tiling)."""
    bits = np.ascontiguousarray(np.asarray(points, np.float32)).view(np.uint32).reshape(len(points), 3)
    h = bits[:, 0] * np.uint32(73856093) ^ bits[:, 1] * np.uint32(19349663) ^ bits[:, 2] * np.uint32(83492791)
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x85ebca6b)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0xc2b2ae35)
    h ^= h >> np.uint32(16)
    return h


def _disc_samples(points, samples):
    """(samples, N, 2) offsets on the unit disc: a Vogel spiral turned by a per-point hash angle, so a
    point always gets the same pattern and neighbouring points get different ones."""
    phi = _point_seeds(points).astype(np.float64) * (2 * math.pi / 2 ** 32)
    k = np.arange(samples, dtype=np.float64)[:, None]
    r = np.sqrt((k + .5) / samples)
    theta = k * 2.399963229728653 + phi[None, :]
    return np.stack((r * np.cos(theta), r * np.sin(theta)), axis=-1)


def _shadow_trace(light, origin, light_position, ray, limit, trace):
    """Transmittance toward `light` from `origin` along `ray` (unit, toward the light) up to `limit`.

    `trace(ray, limit)` returns one hard-shadow transmittance array. With `shadow_blur` 0 this is a
    single call with the given ray, exactly the hard shadow. Otherwise the light is a disc of angular
    radius `shadow_blur` degrees facing the shaded point, sampled with `shadow_samples` deterministic
    jittered rays (seeded by the point's position) and the visibilities are averaged.
    """
    if light.shadow_blur <= 0:
        return trace(ray, limit)
    samples = int(np.clip(light.shadow_samples, 1, SHADOW_SAMPLES_MAX))
    tan = math.tan(math.radians(min(float(light.shadow_blur), 89.0)))
    ray = np.asarray(ray)
    axis = np.where((np.abs(ray[:, 1]) < .9)[:, None], np.array((0., 1., 0.)), np.array((1., 0., 0.)))
    a = np.cross(ray, axis)
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(ray, a)
    offsets = _disc_samples(origin, samples)
    total = np.zeros(len(origin), np.float64)
    for k in range(samples):
        spread = offsets[k, :, 0:1] * a + offsets[k, :, 1:2] * b
        if light.kind in _POSITIONAL:
            target = light_position + spread * (limit[:, None] * tan)
            jray = target - origin
            jlimit = np.linalg.norm(jray, axis=1)
            jray = jray / np.maximum(jlimit[:, None], 1e-30)
        else:
            jray = ray + spread * tan
            jray = jray / np.linalg.norm(jray, axis=1)[:, None]
            jlimit = limit
        total += trace(jray.astype(origin.dtype, copy=False), jlimit)
    return (total / samples).astype(np.float32)


def _shadow_terms(light):
    """(bias scale, tan of the blur half-angle or 0 for a hard shadow, sample count) for the GPU light tables."""
    blur = float(light.shadow_blur)
    return (light.shadow_bias / SHADOW_BIAS_DEFAULT,
            math.tan(math.radians(min(blur, 89.0))) if blur > 0 else 0.0,
            int(np.clip(light.shadow_samples, 1, SHADOW_SAMPLES_MAX)))


def _light_bias(bias, light):
    """The context's scene-scaled epsilon, rescaled by the light's own `shadow_bias`."""
    return bias * (light.shadow_bias / SHADOW_BIAS_DEFAULT)


def _shadow_visibility(position, normal, light, light_position, direction,
                       v0, e1, e2, alpha, bias, cancel, *, triangles=None, bvh=None, splat_shadows=None):
    """Chunked, two-sided Moller-Trumbore; material alpha only, never texture alpha."""
    visibility = np.ones(len(position), np.float32)
    bias = _light_bias(bias, light)
    for start in range(0, len(position), _SHADOW_RAY_CHUNK):
        _shadow_cancel(cancel)
        stop = start + _SHADOW_RAY_CHUNK
        origin = position[start:stop] + normal[start:stop] * bias
        if light.kind in _POSITIONAL:
            ray = light_position - origin
            limit = np.linalg.norm(ray, axis=1)
            ray = ray / np.maximum(limit[:, None], 1e-8)
        else:
            ray = np.broadcast_to(-direction, origin.shape)
            limit = np.full(len(origin), np.inf)
        primitives = triangles if triangles is not None else TriangleSet(v0, e1, e2, alpha)

        def trace(ray, limit):
            if bvh is None:
                value = primitives.brute_transmittance(
                    origin, ray, bias * .01, limit, triangle_chunk=_SHADOW_TRIANGLE_CHUNK, cancel=cancel)
            else:
                value = primitives.transmittance(bvh, origin, ray, bias * .01, limit, cancel=cancel)
            if splat_shadows is not None:
                value = value * splat_shadows.primitives.transmittance(
                    splat_shadows.bvh, origin, ray, bias * .01, limit,
                    cutoff=SPLAT_SHADOW_CUTOFF, cancel=cancel)
            return value
        visibility[start:stop] = _shadow_trace(light, origin, light_position, ray, limit, trace)
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
    splat_shadows: object = None

    def visibility(self, position, normal, light, light_position, direction):
        self.work += _shadow_cost(len(position), self.triangle_count, build=False)
        (_raytrace_budget if self.raytrace else _shadow_budget)(self.work)
        p = self.primitives
        return _shadow_visibility(position, normal, light, light_position, direction,
                                  p.v0, p.e1, p.e2, p.alpha, self.bias, self.cancel,
                                  triangles=p, bvh=self.bvh, splat_shadows=self.splat_shadows)


# Shadow visibility at a splat's centre depends on the casters, the mesh occluders and where the
# light is. It does not depend on the camera, the light's colour or intensity, ambient or the
# Relight amount, so it is kept between renders: a camera move over a lit capture traces nothing.
_SPLAT_CASTER_SETS = 2
_SPLAT_CASTER_BYTES = 1024 * 1024 * 1024   # about 206 bytes per caster; the newest set always stays
_SPLAT_VISIBILITY_BYTES = 256 * 1024 * 1024
_splat_cache_lock = threading.Lock()
_splat_casters = OrderedDict()      # caster key -> _SplatCasters
_splat_visibility = OrderedDict()   # (caster key, mesh key, light key, instance) -> float64, NaN = untraced
_splat_cloud_tokens = {}            # id(cloud) -> (weakref, token); clouds are immutable by convention
_splat_token_counter = itertools.count(1)
splat_shadow_stats = {"caster_builds": 0, "rays_traced": 0, "rays_reused": 0}


def clear_splat_shadow_cache():
    with _splat_cache_lock:
        _splat_casters.clear()
        _splat_visibility.clear()


def _cloud_token(cloud):
    entry = _splat_cloud_tokens.get(id(cloud))
    if entry is not None and entry[0]() is cloud:
        return entry[1]
    key, token = id(cloud), next(_splat_token_counter)
    _splat_cloud_tokens[key] = (weakref.ref(cloud, lambda _ref, key=key, token=token: (
        _splat_cloud_tokens.pop(key, None) if _splat_cloud_tokens.get(key, (None, None))[1] == token else None)), token)
    return token


def _instance_parts(instance):
    cloud, matrix = (instance.cloud, instance.matrix) if hasattr(instance, 'cloud') else instance
    return cloud, np.asarray(matrix, dtype=np.float64)


class _SplatCasters:
    """World-space shadow casters for one set of splat instances, with their BVH.

    Casters with maximum alpha min(.99, opacity) < 1/255 are omitted,
    matching the raster alpha threshold. Relight never controls casting.
    Only positions and scales are kept per instance; the transformed SH is dropped.
    """
    def __init__(self, instances, cancel=None):
        from .splats import _rotation
        self.positions, self.scales, rotations, opacities, offsets = [], [], [], [], [0]
        for instance in instances:
            _shadow_cancel(cancel)
            cloud, matrix = _instance_parts(instance)
            world = cloud.transformed(matrix)
            self.positions.append(world.positions)
            self.scales.append(world.scales * abs(getattr(instance, 'scale_scale', 1.)))
            rotations.append(_rotation(world.rotations))
            casts = 1. if getattr(instance, 'cast_shadows', True) else 0.
            opacities.append(np.clip(world.opacity * getattr(instance, 'opacity_scale', 1.), 0, 1) * casts)
            offsets.append(offsets[-1]+len(world))
        opacity = np.concatenate(opacities)
        keep = np.minimum(.99, opacity) >= 1/255
        self.empty = not keep.any()
        self.offsets = offsets
        self.ids = np.full(len(keep), -1, dtype=np.int64)
        self.ids[keep] = np.arange(keep.sum())
        self.primitives = SplatSet(np.concatenate(self.positions)[keep], np.concatenate(rotations)[keep],
                                   np.concatenate(self.scales)[keep], opacity[keep])
        self.bvh = Bvh.build(*self.primitives.aabbs(), cancel=cancel)
        arrays = {}
        for holder in (self, self.primitives, self.bvh):
            for value in vars(holder).values():
                for array in (value if isinstance(value, list) else (value,)):
                    if isinstance(array, np.ndarray):
                        base = array if array.base is None else array.base
                        arrays[id(base)] = getattr(base, "nbytes", array.nbytes)
        self.nbytes = sum(arrays.values())

    @staticmethod
    def key(instances):
        return tuple((_cloud_token(cloud), matrix.tobytes(), float(getattr(instance, 'scale_scale', 1.)),
                      float(getattr(instance, 'opacity_scale', 1.)), bool(getattr(instance, 'cast_shadows', True)))
                     for instance in instances for cloud, matrix in (_instance_parts(instance),))

    @classmethod
    def shared(cls, instances, cancel=None):
        key = cls.key(instances)
        with _splat_cache_lock:
            casters = _splat_casters.get(key)
            if casters is not None:
                _splat_casters.move_to_end(key)
                return key, casters
        casters = cls(instances, cancel)
        with _splat_cache_lock:
            splat_shadow_stats["caster_builds"] += 1
            _splat_casters[key] = casters
            while len(_splat_casters) > 1 and (len(_splat_casters) > _SPLAT_CASTER_SETS or
                    sum(c.nbytes for c in _splat_casters.values()) > _SPLAT_CASTER_BYTES):
                dropped, _ = _splat_casters.popitem(last=False)
                for stale in [k for k in _splat_visibility if k[0] == dropped]:
                    del _splat_visibility[stale]
        return key, casters


def _mesh_key(mesh, bias):
    if mesh is None:
        return None
    digest = hashlib.blake2b(digest_size=16)
    for array in (mesh.v0, mesh.e1, mesh.e2, mesh.alpha):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.digest(), float(bias)


def _light_key(light):
    position, direction = light.world()
    where = position if light.kind in _POSITIONAL else direction
    # The soft-shadow knobs change the rays, so they are part of the key; the bias only scales the
    # mesh epsilon, which the mesh key already carries per render.
    return (light.kind, np.asarray(where, dtype=np.float64).tobytes(),
            float(light.shadow_bias), float(light.shadow_blur),
            int(light.shadow_samples) if light.shadow_blur > 0 else 1)


class _SplatShadows:
    """One world-space caster set per render, shared by meshes and splats.

    Closest-density alpha accumulation is not a volume integral.
    """
    def __init__(self, instances, lights, mesh=None, mesh_bvh=None, bias=.001, cancel=None):
        self.instances, self.lights = instances, lights
        self.mesh, self.mesh_bvh, self.bias, self.cancel = mesh, mesh_bvh, bias, cancel
        self._mesh_key = _mesh_key(mesh, bias)
        self._shared = None
        self.relit_shadows = True  # False when the render only catches mesh shadows

    def _casters(self):
        # Built on first use: a render that only catches mesh shadows never needs the splat BVH.
        if self._shared is None:
            self._shared = _SplatCasters.shared(self.instances, self.cancel)
        return self._shared

    positions = property(lambda self: self._casters()[1].positions)
    scales = property(lambda self: self._casters()[1].scales)
    offsets = property(lambda self: self._casters()[1].offsets)
    ids = property(lambda self: self._casters()[1].ids)
    primitives = property(lambda self: self._casters()[1].primitives)
    empty = property(lambda self: self._casters()[1].empty)
    bvh = property(lambda self: self._casters()[1].bvh)

    def _store(self, index, light, catch_key=None, count=None):
        key = (catch_key or self._casters()[0], self._mesh_key, _light_key(light), index)
        count = len(self.positions[index]) if count is None else count
        with _splat_cache_lock:
            store = _splat_visibility.get(key)
            if store is None:
                store = _splat_visibility[key] = np.full(count, np.nan)
                total = sum(a.nbytes for a in _splat_visibility.values())
                while total > _SPLAT_VISIBILITY_BYTES and len(_splat_visibility) > 1:
                    _, dropped = _splat_visibility.popitem(last=False)
                    total -= dropped.nbytes
            else:
                _splat_visibility.move_to_end(key)
        return store

    def _trace_catch(self, light, cloud, matrix, ids):
        """Mesh transmittance from the splat centres `ids` toward `light` (the CPU reference)."""
        position, direction = light.world()
        out = np.empty(len(ids))
        for start in range(0, len(ids), _SPLAT_SHADOW_QUERY_CHUNK):
            _shadow_cancel(self.cancel)
            chunk = ids[start:start+_SPLAT_SHADOW_QUERY_CHUNK]
            origin = (matrix[:3, :3] @ np.asarray(cloud.positions[chunk], dtype=np.float64).T).T + matrix[:3, 3]
            if light.kind in _POSITIONAL:
                ray = position-origin
                limit = np.linalg.norm(ray, axis=1)
                ray /= np.maximum(limit[:, None], 1e-30)
            else:
                ray = np.broadcast_to(-direction, origin.shape)
                limit = np.full(len(origin), np.inf)
            bias = _light_bias(self.bias, light)
            out[start:start+len(chunk)] = _shadow_trace(light, origin, position, ray, limit, lambda r, l: self.mesh.transmittance(
                self.mesh_bvh, origin, r, bias*.01, l, cancel=self.cancel))
        return out

    def _trace_relit(self, index, light, ids):
        """Mesh and splat transmittance from the splat centres `ids` of one instance toward `light`."""
        positions, scales = self.positions[index], self.scales[index]
        position, direction = light.world()
        out = np.empty(len(ids))
        for start in range(0, len(ids), _SPLAT_SHADOW_QUERY_CHUNK):
            _shadow_cancel(self.cancel)
            chunk = ids[start:start+_SPLAT_SHADOW_QUERY_CHUNK]
            origin = positions[chunk].astype(float)
            if light.kind in _POSITIONAL:
                ray = position-origin
                limit = np.linalg.norm(ray, axis=1)
                ray /= np.maximum(limit[:, None], 1e-30)
            else:
                ray = np.broadcast_to(-direction, origin.shape)
                limit = np.full(len(origin), np.inf)
            bias = _light_bias(self.bias, light)

            def trace(ray, limit, ids=chunk, origin=origin, bias=bias):
                # Exclude emitter; skip its surface thickness to prevent acne.
                value = np.ones(len(ids)) if self.empty else self.primitives.transmittance(
                    self.bvh, origin, ray, 2.5*np.max(scales[ids], axis=1), limit,
                    exclude=self.ids[self.offsets[index]+ids], cancel=self.cancel,
                    cutoff=SPLAT_SHADOW_CUTOFF)
                if self.mesh is not None:
                    value = value * self.mesh.transmittance(self.mesh_bvh, origin, ray,
                        bias*.01, limit, cancel=self.cancel)
                return value
            out[start:start+len(chunk)] = _shadow_trace(light, origin, position, ray, limit, trace)
        return out

    def catch_for_indices(self, index, indices):
        """Mesh-only visibility of splat centres toward each shadowed light, (len(indices), lights).

        Splats never occlude here, so no splat BVH is built. Cached like for_indices.
        """
        indices = np.asarray(indices)
        visibility = np.ones((len(indices), len(self.lights)))
        if self.mesh is None:
            return visibility
        cloud, matrix = _instance_parts(self.instances[index])
        catch_key = ('catch', _cloud_token(cloud), matrix.tobytes())
        for j, light in enumerate(self.lights):
            if not light.shadows or light.intensity <= 0:
                continue
            store = self._store(index, light, catch_key, len(cloud))
            missing = np.unique(indices[np.isnan(store[indices])])
            splat_shadow_stats["rays_traced"] += len(missing)
            splat_shadow_stats["rays_reused"] += len(indices) - len(missing)
            if len(missing):
                store[missing] = self._trace_catch(light, cloud, matrix, missing)
            visibility[:, j] = store[indices]
        return visibility

    def for_indices(self, index, indices):
        indices = np.asarray(indices)
        visibility = np.ones((len(indices), len(self.lights)))
        if getattr(self.instances[index], 'relight', 0) <= 0 or not self.relit_shadows:
            return visibility
        positions, scales = self.positions[index], self.scales[index]
        for j, light in enumerate(self.lights):
            if not light.shadows or light.intensity <= 0:
                continue
            store = self._store(index, light)
            missing = np.unique(indices[np.isnan(store[indices])])
            splat_shadow_stats["rays_traced"] += len(missing)
            splat_shadow_stats["rays_reused"] += len(indices) - len(missing)
            if len(missing):
                store[missing] = self._trace_relit(index, light, missing)
            visibility[:, j] = store[indices]
        return visibility


def _splat_shadow_visibility(instances, lights, mesh=None, mesh_bvh=None, bias=.001, cancel=None):
    """Compatibility helper returning full per-instance centre visibility."""
    context = _SplatShadows(instances, lights, mesh, mesh_bvh, bias, cancel)
    return [context.for_indices(i, np.arange(len(p))) for i, p in enumerate(context.positions)]


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
    if output == "relight":
        # Keep the single-output shader below unchanged. Bundle data normals face the
        # eye even without lights, just as the standalone normals output does.
        toward_eye = eye - position
        normal = np.where((np.einsum("ij,ij->i", normal, toward_eye) < 0)[:, None], -normal, normal)
        alpha = source[:, 3:4]

        def channel(rgb):
            result = np.empty((len(position), 4), np.float32)
            result[:, :3] = rgb
            result[:, 3:4] = alpha
            return result

        channels = dict(albedo=channel(source[:, :3]), normals=channel(normal),
                        position=channel(position))
        # Existing unlit diffuse is albedo, independent of ambient.
        radiance = np.full((len(position), 3), float(ambient) if lights else 1., np.float32)
        specular_total = np.zeros_like(radiance)
        to_eye = toward_eye / np.maximum(np.linalg.norm(toward_eye, axis=1, keepdims=True), 1e-8)
        for i, (light, light_position, direction) in enumerate(lights):
            if light.kind in _POSITIONAL:
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
            attenuation = _light_factor(light, position)
            if attenuation is not None:
                lambert = lambert * attenuation
            diffuse_response = np.maximum(lambert, 0)
            half = to_light + to_eye
            half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)
            lobe = np.maximum(np.einsum("ij,ij->i", normal, half), 0) ** geometry.shininess
            specular_response = geometry.specular * lobe * front * visibility
            if attenuation is not None:
                specular_response = specular_response * attenuation
            colour = np.asarray(light.color, np.float32) * light.intensity
            radiance += diffuse_response[:, None] * colour
            specular_total += specular_response[:, None] * colour
            channels[f"diffuse_L{i}"] = channel(diffuse_response[:, None] * alpha)
            channels[f"specular_L{i}"] = channel(specular_response[:, None] * alpha)
        channels["diffuse"] = channel(source[:, :3] * radiance)
        channels["specular"] = channel(specular_total * alpha)
        channels["emission"] = channel(source[:, :3] * geometry.emission)
        return channels, normal, uv
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
            if light.kind in _POSITIONAL:
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
            attenuation = _light_factor(light, position)
            if attenuation is not None:
                lambert = lambert * attenuation
            radiance += np.maximum(lambert, 0)[:, None] * (np.asarray(light.color, np.float32) * light.intensity)
            if specular is not None:
                half = to_light + to_eye
                half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)
                lobe = np.maximum(np.einsum("ij,ij->i", normal, half), 0) ** geometry.shininess
                specular += (geometry.specular * lobe * front * visibility
                              * (1.0 if attenuation is None else attenuation))[:, None] * (
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
                    focal, aspect, lights, ambient, output, shade, shadow_context, cancel, mesh_layers=None, rows=None):
    """Chunked primary visibility; shading is batched by geometry and mip level."""
    flat, flat_depth = out.reshape(-1, 4), depth.ravel()
    projection_depth_maps = {}
    data_output = output in DATA_OUTPUTS
    # Invert the actual float32 view basis, including roll, to undo _to_pixels.
    inverse_view = np.linalg.inv(view.astype(np.float64))
    y0, y1 = (0, height) if rows is None else rows
    for start in range(0, width*(y1-y0), _PRIMARY_RAY_CHUNK):
        _shadow_cancel(cancel)
        stop = min(start+_PRIMARY_RAY_CHUNK, width*(y1-y0))
        pixels = np.arange(start, stop)
        x, y = pixels % width + .5, pixels // width + y0 + .5
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


def _render_mesh_layers(scene, camera, width, height, out, depth, rows=None, **kwargs):
    """Record shaded surfaces for full-frame rows=(y0, y1), including termination.

    out, depth and returned layers use band-local rows; rays retain the full
    width/height projection. Omitting rows records the full frame.
    """
    y0, y1 = (0, height) if rows is None else rows
    shape = (y1-y0, width, MAX_MESH_LAYERS)
    layers = (np.full(shape, np.inf, np.float64),
              np.zeros((*shape, 3), np.float32), np.zeros(shape, np.float32))
    _render_primary(scene, camera, width, height, out, depth, mesh_layers=layers, rows=rows, **kwargs)
    return layers


def render(scene: Scene, camera: Camera, width: int, height: int, background=(0., 0., 0., 0.),
           shade=False, return_depth=False, ambient=0.0, samples=1, output="rgba", cancel=None, *, shadows=True, mode="raster", progress=None):
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
    Shading outputs exclude background and retain mesh beauty alpha for premultiplied over.
    Without splats, diffuse + specular + emission == transparent-background beauty.
    Unlit scenes have diffuse == albedo and specular == 0, hence beauty == albedo + emission.
    Alpha is shared coverage/transparency, not an additive lighting component.
    With ``shadows=True``, enabled lights trace two-sided world-space triangle rays.
    Every geometry casts and receives shadows, including projected geometry. Visibility
    multiplies (1 - geometry alpha) over hits; texture alpha is NOT considered. Ambient,
    data outputs and inspection shading are unaffected.
    Splats contribute to rgba and the premultiplied, background-free splats layer.
    Data outputs select the first mesh or accumulated splat opacity >= 0.5, with
    binary coverage. Splat object IDs follow geometry IDs, one per instance.
    Albedo, diffuse, specular and emission ignore splats until relighting exists. Splat planes
    depth-test per pixel. Raster mesh visibility uses the raster depth buffer, except
    transparent beauty/layer surfaces, which use depth-merged primary-ray layers.
    ``return_depth`` also returns the depth buffer (inf where empty).
    CPU splat progress(stage, fraction, info) spans all accumulation bands.
    Stages are prepare/splats/done; info includes tile_work and estimate_seconds
    once prepared, plus eta_seconds on updates. A callback disables the splat
    tile-work budget; callback exceptions abort rendering.
    """
    if output == "relight":
        if mode != "raster":
            raise ValueError("the relight bundle output is raster-only for now")
        if scene.splats:
            raise ValueError("the relight bundle output does not support scenes with splats yet")
        if return_depth:
            raise ValueError("the relight bundle output does not support return_depth=True")
    _shadow_cancel(cancel)
    if mode not in ("raster", "raytrace"):
        raise ValueError(f"Unknown 3D render mode {mode!r}")
    if output not in RENDER_OUTPUTS:
        raise ValueError(f"Unknown 3D render output {output!r}")
    width, height = int(width), int(height)
    data_output = output in DATA_OUTPUTS
    samples = max(1, min(int(samples), 4)) if not data_output and output != "relight" else 1
    shadow_count = sum(light.shadows and light.intensity > 0 for light in scene.lights)
    shadow_active = shadows and not shade and output in ("rgba", "diffuse", "specular", "relight") and shadow_count > 0
    triangle_count = sum(len(g.triangles) for g in scene.geometries)
    splat_shadow_active = (shadows and shadow_count > 0 and output in ('rgba', 'splats')
                           and any(getattr(i, 'relight', 0) > 0 for i in scene.splats))
    casting = [i for i in scene.splats if getattr(i, 'cast_shadows', True)]
    splat_cast_active = bool(scene.splats) and ((shadow_active and bool(casting)) or splat_shadow_active)
    # Shadow catching: meshes shadow the CAPTURED colour of a splat instance. Only meshes cast here;
    # the capture already contains the shadows its own splats threw when it was photographed.
    catching = [i for i in scene.splats if getattr(i, 'shadow_catch', 0) > 0 and getattr(i, 'relight', 0) < 1]
    splat_catch_active = bool(shadows and shadow_count > 0 and triangle_count and catching
                              and output in ('rgba', 'splats'))
    if splat_catch_active:
        work = _shadow_cost(sum(len(i.cloud) for i in catching)*shadow_count, triangle_count)
        if work > SPLAT_SHADOW_BUDGET:
            raise ValueError(f'Splat shadow-catch rays exceed the CPU reference budget: {work:,.0f} estimated tests > {SPLAT_SHADOW_BUDGET:,}')
    if splat_cast_active:
        counts = [len(i.cloud if hasattr(i, 'cloud') else i[0]) for i in scene.splats]
        rays = sum(n for n, i in zip(counts, scene.splats) if getattr(i, 'relight', 0) > 0)*shadow_count
        mesh_rays = width*height*samples**2*shadow_count if shadow_active and triangle_count and casting else 0
        casters = sum(n for n, i in zip(counts, scene.splats) if getattr(i, 'cast_shadows', True))
        work = _shadow_cost(rays, triangle_count) + (_shadow_cost(rays+mesh_rays, casters) if casters else 0)
        if work > SPLAT_SHADOW_BUDGET:
            raise ValueError(f'Splat shadow rays exceed the CPU reference budget: {work:,.0f} estimated tests > {SPLAT_SHADOW_BUDGET:,}')
    if triangle_count > MAX_TRIANGLES:
        raise ValueError(f"Scene exceeds {MAX_TRIANGLES} triangles; the CPU reference renderer refuses it")
    layered = bool(scene.splats) and output in ("rgba", "splats") and not _opaque_meshes(scene)
    splat_visibility = bool(scene.splats) and (data_output or output in ("rgba", "splats"))
    ray_mode = mode == "raytrace" or layered
    if ray_mode:
        rays = width * height * samples ** 2
        _raytrace_budget(_shadow_cost(rays, triangle_count)
                         + _shadow_cost(rays * shadow_count, triangle_count, build=False) * shadow_active)
    elif shadow_active:
        # Estimate before framebuffer allocation; a running counter also bounds overdraw.
        _shadow_budget(_shadow_cost(width * height * samples ** 2 * shadow_count, triangle_count))
    if samples > 1:
        big = render(scene, camera, width * samples, height * samples, background, shade,
                     return_depth, ambient, 1, output, cancel, shadows=shadows, mode=mode,
                     progress=progress)
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
    if output == "relight":
        names = ("albedo", "normals", "position", "diffuse", "specular", "emission")
        names += tuple(f"{kind}_L{i}" for i in range(len(lights)) for kind in ("diffuse", "specular"))
        channels = {name: np.full((height, width, 4), 0, np.float32) for name in names}
        # Data hits include transparent surfaces; beauty depth only includes solids.
        # Keep this selection buffer separate so intersecting transparent triangles
        # cannot overwrite a nearer data hit in the far-to-near beauty draw order.
        first_hit_depth = np.full((height, width), np.inf, np.float32)
    # Collect clipped triangles, then rasterize far-to-near. Opaque pixels write the z buffer;
    # transparent ones only test it, so they reveal what is behind them and composite in depth
    # order. That is sorted transparency, not order-independent transparency: interpenetrating
    # transparent surfaces can still sort wrongly.
    # Per-vertex attribute layout: view xyz (3) | world xyz (3) | world normal (3) | uv (2).
    queue = []
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
        if (shadow_active or splat_shadow_active or splat_catch_active or ray_mode) and len(geometry.triangles):
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
               if triangle_count and (ray_mode or splat_shadow_active or splat_catch_active
                                      or triangle_count > _SHADOW_BRUTE_THRESHOLD) else None)
        if ray_mode and bvh is None:
            empty = np.empty(0, np.int32)
            bvh = Bvh(np.empty((0, 3)), np.empty((0, 3)), empty, empty, empty, empty, empty)
        bias = 1e-3 * max(1.0, float(np.ptp(triangles.reshape(-1, 3), axis=0).max())) if len(triangles) else .001
        if shadow_active:
            work = _shadow_cost(width*height, triangle_count) if ray_mode else shadow_work
            shadow_context = _ShadowContext(primitives, bvh, bias, cancel, triangle_count, work, ray_mode)
    splat_shadows = None
    if splat_cast_active or splat_catch_active:
        splat_shadows = _SplatShadows(scene.splats, scene.lights,
            primitives if triangle_count else None, bvh if triangle_count else None,
            bias if triangle_count else .001, cancel)
        splat_shadows.relit_shadows = splat_shadow_active
        if shadow_context is not None and splat_cast_active and casting:
            shadow_context.splat_shadows = splat_shadows
    if ray_mode:
        primary_kwargs = dict(attributes=ray_attributes, object_ids=ray_object_ids,
                        mip_levels=ray_mip_levels, clipped_mips=clipped_mips, materials=materials,
                        primitives=primitives, bvh=bvh, eye=eye, view=view, focal=focal, aspect=aspect,
                        lights=lights, ambient=ambient, output=output, shade=shade,
                        shadow_context=shadow_context, cancel=cancel)
        if not (layered or (splat_visibility and data_output)):
            _render_primary(scene, camera, width, height, out, depth, **primary_kwargs)
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
        bundle = source if output == "relight" else None
        if bundle is not None:
            source = bundle["albedo"]
        src_alpha = source[:, 3]
        region = out[y0:y1+1, x0:x1+1]
        if bundle is not None:
            opaque = src_alpha > 0
            hit_depth = first_hit_depth[y0:y1+1, x0:x1+1]
            first_hit = opaque & (zbuf[take] < hit_depth[take])
            hit_pixels = hit_depth[take]
            hit_pixels[first_hit] = zbuf[take][first_hit]
            hit_depth[take] = hit_pixels
            for name, fragments in bundle.items():
                channel_region = channels[name][y0:y1+1, x0:x1+1]
                if name in ("normals", "position"):
                    pixels = channel_region[take]
                    pixels[first_hit, :3] = fragments[first_hit, :3]
                    pixels[first_hit, 3] = 1
                    channel_region[take] = pixels
                else:
                    channel_region[take] = fragments + channel_region[take] * (1 - src_alpha[:, None])
        elif output == "depth":
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
    if output == "relight":
        # Lighting components add in RGB; alpha remains shared surface coverage.
        out = channels["diffuse"].copy()
        out[..., :3] += channels["specular"][..., :3] + channels["emission"][..., :3]
        out.flags.writeable = False
        for channel in channels.values():
            channel.flags.writeable = False
        return out, channels
    if scene.splats and (output in ("rgba", "splats") or data_output):
        from .splatraster import prepare_splats, accumulate_splats, estimate_seconds, estimate_eta_seconds
        if progress is not None:
            progress("prepare", 0.0, {})
        splat_lighting = (scene.lights, ambient)
        if splat_shadow_active or splat_catch_active:
            splat_lighting = (scene.lights, ambient, splat_shadows)
        prepared = prepare_splats(scene.splats, camera, width, height, cancel=cancel,
                                  output=output, object_id_offset=len(scene.geometries),
                                  enforce_budget=progress is None,
                                  lighting=splat_lighting if not data_output and (splat_catch_active or
                                  any(getattr(i, "relight", 0) > 0 for i in scene.splats)) else None)
        band_progress = None
        if progress is not None:
            from time import monotonic
            info = dict(tile_work=prepared.tile_work,
                        estimate_seconds=estimate_seconds(prepared.tile_work))
            progress("splats", 0.0, dict(info))
            started = monotonic()
            completed_work = band_work = 0
            last_fraction = 0.0

            def band_progress(done, total):
                nonlocal band_work, last_fraction
                band_work = total
                completed = completed_work + done
                fraction = completed / prepared.tile_work if prepared.tile_work else 0.0
                # Bound the whole render's notifications even with one-row bands.
                if fraction - last_fraction >= .005 or (fraction == 1.0 and last_fraction < 1.0):
                    last_fraction = fraction
                    progress("splats", fraction, dict(info, eta_seconds=estimate_eta_seconds(
                        completed, prepared.tile_work, monotonic() - started)))
        rows_per_band = max(1, min(height, LAYER_BAND_BYTES // (width * 384)))
        if not layered and not data_output and output != "splats":
            rows_per_band = height  # Preserve the opaque beauty shortcut.
        for y0 in range(0, height, rows_per_band):
            _shadow_cancel(cancel)
            y1 = min(height, y0+rows_per_band)
            band = (y0, y1)
            band_out, band_depth = out[y0:y1], depth[y0:y1]
            if layered:
                mesh_layers = _render_mesh_layers(scene, camera, width, height,
                    band_out, band_depth, rows=band, **primary_kwargs)
            elif data_output and ray_mode:
                _render_primary(scene, camera, width, height, band_out, band_depth,
                                rows=band, **primary_kwargs)
                mesh_layers = (band_depth[..., None], band_out[..., None, :3], band_out[..., None, 3])
            splat_rgb, splat_alpha = accumulate_splats(
                prepared, mesh_depth=None if mesh_layers is not None else band_depth, cancel=cancel,
                mesh_layers=mesh_layers,
                background_rgba=band_out if layered and output == "rgba" else None,
                output=output,
                hit_depth=band_depth if data_output else None, rows=band,
                progress=band_progress)
            if progress is not None:
                completed_work += band_work
            if data_output and not ray_mode:
                # Preserve raster mesh attributes bit-for-bit unless a splat hits.
                hit = splat_alpha > 0
                band_out[hit, :3] = splat_rgb[hit]
                band_out[hit, 3] = splat_alpha[hit]
            elif layered or data_output or output == "splats":
                band_out[..., :3], band_out[..., 3] = splat_rgb, splat_alpha
            else:
                band_out[..., :3] = splat_rgb + (1-splat_alpha[..., None])*band_out[..., :3]
                band_out[..., 3] = splat_alpha + (1-splat_alpha)*band_out[..., 3]
            # Release layers before allocating the next band.
            mesh_layers = None
        if progress is not None:
            progress("done", 1.0, dict(info, eta_seconds=0.0))
    if scene.particles and output == "rgba":
        _draw_particles(scene, camera, width, height, out, depth, eye, view, focal, aspect, cancel)
    if output == "splats" and not scene.splats:
        out[:] = 0
    out.flags.writeable = False
    if return_depth:
        depth.flags.writeable = False
        return out, depth
    return out


PARTICLE_MIN_RADIUS = 0.75      # pixels: a particle smaller than this still lights its own pixel
PARTICLE_MAX_RADIUS = 96        # pixels: a nearer particle is clamped rather than filling the frame
_PARTICLE_SHAPES = {"points": 0, "spheres": 1, "cards": 2}
_PARTICLE_FRAGMENT_CHUNK = 2_000_000


def particle_sprites(scene, camera, width, height, eye, view, focal, aspect):
    """Every ParticleInstance as screen sprites sorted far to near, shared by the CPU and GPU draws.

    Returns None when nothing is in front of the camera, else `(z, centre, radius, color, shape,
    world_radius, texture_id, textures)`: view depth, pixel centre, clamped pixel radius, premultiplied
    colour, shape code (`_PARTICLE_SHAPES`), world radius, index into `textures` (-1 for none).
    """
    order_sets, textures = [], []
    for instance in scene.particles:
        if not len(instance.positions):
            continue
        matrix = instance.matrix.astype(np.float64)
        world = (matrix[:3, :3] @ instance.positions.astype(np.float64).T).T + matrix[:3, 3]
        local = (view @ (world - eye).T).T
        z = -local[:, 2]
        keep = (z > camera.near) & (z < camera.far)
        if not keep.any():
            continue
        z = z[keep]
        centre = _to_pixels(local[keep], z, focal, aspect, width, height)
        sizes = instance.sizes[keep] * np.float32(instance.size_scale)
        radius = np.clip(0.25 * sizes * focal * height / z, PARTICLE_MIN_RADIUS, PARTICLE_MAX_RADIUS)
        shape = np.full(len(z), _PARTICLE_SHAPES.get(instance.render_as, 0), np.int8)
        texture = -1
        if instance.render_as == "cards" and instance.texture is not None:
            texture = len(textures)
            textures.append(instance.texture)
        order_sets.append((z, centre, radius, instance.colors[keep], shape, 0.5 * sizes,
                           np.full(len(z), texture, np.int32)))
    if not order_sets:
        return None
    z = np.concatenate([s[0] for s in order_sets])
    centre = np.concatenate([s[1] for s in order_sets])
    radius = np.concatenate([s[2] for s in order_sets])
    color = np.concatenate([s[3] for s in order_sets])
    shape = np.concatenate([s[4] for s in order_sets])
    world_radius = np.concatenate([s[5] for s in order_sets])
    texture_id = np.concatenate([s[6] for s in order_sets])
    far_first = np.argsort(-z, kind="stable")
    z, centre, radius, color = z[far_first], centre[far_first], radius[far_first], color[far_first]
    shape, world_radius, texture_id = shape[far_first], world_radius[far_first], texture_id[far_first]
    return z, centre, radius, color, shape, world_radius, texture_id, textures


def _draw_particles(scene, camera, width, height, out, depth, eye, view, focal, aspect, cancel):
    """Composite every ParticleInstance over `out` as size-scaled discs (beauty output only).

    The CPU reference for the particle draw: each particle is a flat disc (points), a shaded disc
    (spheres) or a square (cards) of world diameter `size` facing the camera, `colors` premultiplied,
    tested against the mesh depth buffer but never written to it. Particles composite far to near over
    what is already in `out` (meshes and splats), so overlapping translucent particles blend in depth
    order. A pixel belongs to a disc when its centre lies within the projected radius, which is at
    least PARTICLE_MIN_RADIUS pixels.
    """
    sprites = particle_sprites(scene, camera, width, height, eye, view, focal, aspect)
    if sprites is None:
        return
    z, centre, radius, color, shape, world_radius, texture_id, textures = sprites
    extra = (shape, world_radius, texture_id, textures)
    reach = np.ceil(radius).astype(np.int64)
    footprint = (2 * reach + 1) ** 2
    start = 0
    while start < len(z):
        _shadow_cancel(cancel)
        stop = start + 1
        total = int(footprint[start])
        while stop < len(z) and total + int(footprint[stop]) <= _PARTICLE_FRAGMENT_CHUNK:
            total += int(footprint[stop])
            stop += 1
        _composite_particle_chunk(slice(start, stop), z, centre, radius, reach, color, out, depth, width, height, extra)
        start = stop


def _composite_particle_chunk(rows, z, centre, radius, reach, color, out, depth, width, height, extra):
    shape, world_radius, texture_id, textures = extra
    pixel_parts, order_parts, u_parts, v_parts = [], [], [], []
    z, centre, radius, reach, color = z[rows], centre[rows], radius[rows], reach[rows], color[rows]
    shape, world_radius, texture_id = shape[rows], world_radius[rows], texture_id[rows]
    for span in np.unique(reach):
        members = np.flatnonzero(reach == span)
        offsets = np.arange(-span, span + 1)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        dx, dy = dx.ravel(), dy.ravel()
        px = np.floor(centre[members, 0])[:, None].astype(np.int64) + dx[None]
        py = np.floor(centre[members, 1])[:, None].astype(np.int64) + dy[None]
        off_x, off_y = px + 0.5 - centre[members, 0][:, None], py + 0.5 - centre[members, 1][:, None]
        disc = off_x ** 2 + off_y ** 2 <= radius[members, None] ** 2
        square = shape[members] == 2                       # cards fill their whole square
        if square.any():
            box = (np.abs(off_x) <= radius[members, None]) & (np.abs(off_y) <= radius[members, None])
            disc = np.where(square[:, None], box, disc)
        inside = disc & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        rows_index, cols_index = np.nonzero(inside)
        if not len(rows_index):
            continue
        owner = members[rows_index]
        pixel_parts.append(py[rows_index, cols_index] * width + px[rows_index, cols_index])
        order_parts.append(owner)
        u_parts.append(off_x[rows_index, cols_index] / radius[owner])      # -1 .. 1 across the particle,
        v_parts.append(off_y[rows_index, cols_index] / radius[owner])      # v growing downwards
    if not pixel_parts:
        return
    pixel, owner = np.concatenate(pixel_parts), np.concatenate(order_parts)
    frag_u, frag_v = np.concatenate(u_parts), np.concatenate(v_parts)
    frag_z = z[owner]
    sphere = shape[owner] == 1
    if sphere.any():                                        # a sphere's surface is nearer than its centre
        facing = np.sqrt(np.maximum(0.0, 1.0 - frag_u ** 2 - frag_v ** 2))
        frag_z = np.where(sphere, frag_z - facing * world_radius[owner], frag_z)
    visible = frag_z < depth.reshape(-1)[pixel]
    pixel, owner, frag_u, frag_v = pixel[visible], owner[visible], frag_u[visible], frag_v[visible]
    if not len(pixel):
        return
    # Far to near inside each pixel: order fragments by owner (already far-first), then group by
    # pixel with a stable sort, and give each fragment its rank in its pixel's stack.
    by_owner = np.argsort(owner, kind="stable")
    pixel, owner, frag_u, frag_v = pixel[by_owner], owner[by_owner], frag_u[by_owner], frag_v[by_owner]
    by_pixel = np.argsort(pixel, kind="stable")
    pixel, owner, frag_u, frag_v = pixel[by_pixel], owner[by_pixel], frag_u[by_pixel], frag_v[by_pixel]
    first = np.r_[True, pixel[1:] != pixel[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(pixel)), 0))
    rank = np.arange(len(pixel)) - group_start
    by_rank = np.argsort(rank, kind="stable")
    counts = np.bincount(rank)
    fragment = color[owner]
    sphere = shape[owner] == 1
    if sphere.any():                                        # view-space normal on the unit disc, headlight-ish
        nz = np.sqrt(np.maximum(0.0, 1.0 - frag_u ** 2 - frag_v ** 2))
        lit = np.maximum(0.0, frag_u * _VIEW_LIGHT[0] - frag_v * _VIEW_LIGHT[1] + nz * _VIEW_LIGHT[2])
        fragment[sphere, :3] *= (0.25 + 0.75 * lit[sphere])[:, None].astype(np.float32)
    textured = texture_id[owner] >= 0
    for index in np.unique(texture_id[owner][textured]):
        image = textures[index]
        pick = textured & (texture_id[owner] == index)
        column = np.clip(((frag_u[pick] + 1.0) * 0.5 * image.shape[1]).astype(np.int64), 0, image.shape[1] - 1)
        line = np.clip(((frag_v[pick] + 1.0) * 0.5 * image.shape[0]).astype(np.int64), 0, image.shape[0] - 1)
        fragment[pick] *= image[line, column]
    flat = out.reshape(-1, 4)
    begin = 0
    for count in counts:
        pick = by_rank[begin:begin + count]
        begin += count
        target, source = pixel[pick], fragment[pick]
        flat[target] = source + flat[target] * (1.0 - source[:, 3:4])


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
