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
from .splats import SplatCloud, C0
from .envlight import Environment
from . import filmback as _fb
from . import lens as _lens_module

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
                  "emission", "position", "uv", "object_id", "relight", "splats", "normals_blend",
                  "volume_density", "volume_motion", "volume_temperature", "volume_vorticity", "volume_id")
# Single-purpose control passes of the volume raymarch (nodebased/volumerender.py); they read only volumes.
VOLUME_OUTPUTS = ("volume_density", "volume_motion", "volume_temperature", "volume_vorticity", "volume_id")
# Internal to render(): the splats' normal layer that "normals_blend" composites over the mesh normals.
_SPLAT_LAYERS = ("splats", "splat_normals")
LIGHT_OUTPUTS = ("rgba", "albedo", "diffuse", "specular", "emission", "splats")
DATA_OUTPUTS = ("depth", "normals", "position", "uv", "object_id")
LIGHT_TYPES = ("Directional", "Point", "Spot", "Rect", "Disc", "Sphere", "Environment")   # Environment builds an envlight.Environment, not a Light
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
    # Physically based material (materials 1, R1: docs/3D_FOUNDATION.md "Materials"). Only read when
    # `material` is "pbr": Cook-Torrance GGX, the same BRDF `splatshade` shades splats with
    # (`_shade_pbr_mesh` below), instead of the Blinn-Phong `specular`/`shininess` above.
    metallic: float = 0.0        # 0 dielectric .. 1 conductor (tints specular by base_color, kills diffuse)
    pbr_roughness: float = 0.5   # 0 mirror .. 1 fully rough
    pbr_specular: float = 0.5    # dielectric F0 knob, 0..1; 0.5 is F0 0.04, the same default splats use
    # Liquid material (plan 3 step D). `material` "standard" is every surface before liquids existed; "liquid" refracts
    # and reflects in the ray-traced modes (see `_LiquidTracer`) and is approximated in raster.
    material: str = "standard"
    ior: float = 1.333
    absorption_color: tuple[float, float, float] = (0.55, 0.8, 0.95)  # fraction that survives `absorption_distance`
    absorption_distance: float = 1.0
    reflection: float = 1.0             # scales the Fresnel reflection; total internal reflection is always full
    roughness: float = 0.0              # blurs the environment reflection (a prefiltered lookup level)
    # Cryptomatte identity (nodebased/cryptomatte3d.py). `name` is the node that created this
    # geometry (CryptoObject); `asset` is the outermost Scene3D/Axis3D parent's name, stamped by
    # `scene_from_node`, or `name` itself when nothing wraps it (CryptoAsset).
    name: str = ""
    asset: str = ""
    # Per-vertex velocity in object space, units per frame (FluidSurface3D fills it from its particles). Motion
    # blur (nodebased/motionblur.py) moves the vertices along it across the shutter; nothing else reads it.
    velocities: np.ndarray | None = None

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
    # Rect/Disc/Sphere (R2, area lights, see `_AREA`): a real-size emitter facing `target` (Sphere is
    # isotropic and ignores it); see `_area_light_contribution`. `shadow_blur`/`shadow_samples` above
    # are unused for these kinds: softness comes from sampling the light's own surface.
    area_width: float = 1.0             # Rect, world units
    area_height: float = 1.0            # Rect, world units
    area_radius: float = 0.5            # Disc/Sphere, world units
    area_normalize: bool = False        # on: intensity is power, independent of area; off: a radiance
    two_sided: bool = False             # Rect/Disc emit from both faces
    light_samples: int = 4              # light-surface samples per shading point

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
    # Thin-lens depth of field (nodebased/lens.py; one scene unit is one metre). `fstop` 0 is a pinhole, the
    # only camera every renderer drew before the lens existed. The path tracer and the ray-traced mode sample
    # the aperture; raster, splat-only and data passes stay sharp.
    fstop: float = 0.0
    focus_distance: float = 5.0        # scene units from the camera along the view axis to the sharp plane
    aperture_blades: int = 0           # 0 (or fewer than 3) is a round aperture, else a regular polygon
    blade_rotation: float = 0.0        # degrees
    anamorphic_squeeze: float = 1.0    # 2 makes the bokeh twice as tall as wide

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
    normal_smoothing: int = 0  # average each splat's estimated normal over this many nearest splats (0 = off)
    use_intrinsics: bool = True  # relight from the cloud's de-lit albedo and BRDF when it has them (ReadSplat3D Delight)
    metallic: float = 0.0      # 0 dielectric .. 1 conductor: constant over the cloud, a capture cannot show it
    roughness_scale: float = 1.0   # multiplies the de-lit roughness (0 is a mirror)
    intrinsics_mix: float = 1.0    # 1 the de-lit PBR shading, 0 the captured colour lit as before
    reflection_samples: int = 0    # mesh reflection rays per splat; 0 reads the prefiltered environment only
    indirect_samples: int = 0      # hemisphere rays per splat for occlusion and one diffuse bounce; 0 is off (splatindirect)
    indirect_distance: float = 1.0 # how far those rays look, in world units
    denoise: float = 0.0           # guided smoothing of the bounce and traced reflections only, 0 off .. 1 full
    quality: str = "medium"        # preview | medium | final: scales indirect and reflection sample counts
    name: str = ""                 # the ReadSplat3D that produced this cloud (Cryptomatte CryptoObject/CryptoAsset)


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
    render_as: str = "points"        # "points" flat discs, "spheres" shaded discs, "cards" camera-facing squares,
                                     # "foam" white lit soft discs (see `foam_density`, `spray_size`)
    size_scale: float = 1.0          # multiplies `sizes` at draw time (ParticleRender3D), never the solve
    foam_density: float = 1.0        # foam only: the fraction of the particles drawn (a fixed subset by particle id)
    spray_size: float = 1.0          # foam only: multiplies the disc size, on top of `size_scale`
    texture: np.ndarray | None = None  # premultiplied float32 RGBA sprite for cards, row 0 at the top
    velocities: np.ndarray | None = None
    ages: np.ndarray | None = None       # frames since birth
    lifetimes: np.ndarray | None = None  # frames from birth to death
    ids: np.ndarray | None = None
    stream: object | None = None
    frame: int = 0
    surface: object | None = None    # a liquid's signed-distance Volume (FluidLiquidSolver3D): negative inside
    whitewater_type: np.ndarray | None = None  # uint8: 0 foam, 1 spray, 2 bubbles (FluidWhitewater3D)
    # Material and attribute ramps (R7 of 7, ParticleRender3D). "standard" keeps the old fake headlight
    # look on "spheres" pixel-identical; "pbr" shades every particle with the scene's own lights and
    # dome through the same Cook-Torrance GGX BRDF `_shade_pbr_mesh` shades meshes with.
    material: str = "standard"
    metallic: float = 0.0
    pbr_roughness: float = 0.5
    pbr_specular: float = 0.5
    cast_shadows: bool = True        # off: this instance's particles no longer occlude scene lights
    emission: np.ndarray | None = None  # (N,) self-glow multiplier from `particle_emission`/emission ramp, 0 is off

    def __len__(self):
        return len(self.positions)


@dataclass(frozen=True, eq=False)
class InstanceSet:
    """N copies of one of up to eight `sources` meshes (Instance3D), in the space of `parent`.

    `matrices` is (N,4,4) float64: each instance's own object-to-parent transform (position,
    orientation, scale already baked in by `instance_transforms`). `variant` is (N,) int32,
    indexing `sources`. `sources` keeps each variant mesh's vertices and triangles exactly once;
    `expand_instances` hands out Geometry objects that reference those same arrays (never copies
    them), so N stays cheap regardless of how many triangles a source mesh has.
    """
    sources: tuple
    matrices: np.ndarray
    variant: np.ndarray
    colors: np.ndarray | None = None   # (N,4) premultiplied tint from `color_from_points`, or None
    ids: np.ndarray | None = None
    parent: np.ndarray = field(default_factory=lambda: _IDENTITY)
    velocities: np.ndarray | None = None   # (N,3) point velocity in the space of `matrices`, units per frame (motion blur)

    def __len__(self):
        return len(self.matrices)


@dataclass(frozen=True, eq=False)
class Volume:
    """A regular voxel grid of smoke-like data (docs/FLUIDS_SPIKE.md), in the space of `matrix`.

    Arrays are indexed [ix, iy, iz], so `density` is (nx, ny, nz) float32 and `velocity` (nx, ny, nz, 3)
    is a cell-centred vector field in the volume's own space, in units per second (multiply by the
    matrix to get world). `voxel_size` is the edge of one cubic voxel; `origin` is the object-space
    position of the minimum corner of voxel (0, 0, 0), so cell (i, j, k) is centred at
    `origin + (i + .5, j + .5, k + .5) * voxel_size`. `matrix` is the column-vector object-to-world
    transform (Scene3D and Axis3D multiply their own onto it, like SplatInstance). `temperature` is an
    optional (nx, ny, nz) float32 scalar field.
    """
    density: np.ndarray
    voxel_size: float = 1.0
    origin: tuple = (0.0, 0.0, 0.0)
    matrix: np.ndarray = field(default_factory=lambda: _IDENTITY)
    temperature: np.ndarray | None = None
    velocity: np.ndarray | None = None
    flame: np.ndarray | None = None     # optional (nx, ny, nz) burn rate of a fire solve (fuel per frame)
    stream: object | None = None        # the fluid run this frame came from (FluidCache3D re-solves through it)
    frame: int = 0
    sparse: object | None = None        # a sparsevol.SparseGrid of the same fields when the frame came from sparse tiles
    fuel: np.ndarray | None = None      # optional unburnt fuel carried for later combustion

    @classmethod
    def from_sparse(cls, grid, voxel_size=1.0, origin=(0.0, 0.0, 0.0), matrix=None, stream=None, frame=0):
        """A Volume from a `sparsevol.SparseGrid` with a `density` field (and optionally `temperature`, `velocity`,
        `flame`). The dense arrays the ray marcher reads are built here, on demand; the sparse grid stays on `.sparse`."""
        dense = grid.to_dense()
        kwargs = {} if matrix is None else {"matrix": matrix}
        return cls(dense["density"], voxel_size=voxel_size, origin=origin, temperature=dense.get("temperature"),
                   velocity=dense.get("velocity"), flame=dense.get("flame"), fuel=dense.get("fuel"), stream=stream, frame=frame, sparse=grid,
                   **kwargs)

    def to_sparse(self, tile=8, threshold=0.0):
        """The tiles of this volume that hold anything (a `sparsevol.SparseGrid`); the stored one when there is one."""
        if self.sparse is not None:
            return self.sparse
        from .sparsevol import SparseGrid
        fields = {"density": self.density}
        for name in ("temperature", "velocity", "flame", "fuel"):
            if getattr(self, name) is not None:
                fields[name] = getattr(self, name)
        return SparseGrid.from_dense(fields, tile, threshold=threshold)

    def __post_init__(self):
        density = np.ascontiguousarray(self.density, np.float32)
        if density.ndim != 3 or 0 in density.shape:
            raise ValueError("Volume density must be a non-empty (nx, ny, nz) array")
        if not float(self.voxel_size) > 0:
            raise ValueError("Volume voxel_size must be positive")
        object.__setattr__(self, "density", density)
        if self.temperature is not None:
            temperature = np.ascontiguousarray(self.temperature, np.float32)
            if temperature.shape != density.shape:
                raise ValueError("Volume temperature must match the density shape")
            object.__setattr__(self, "temperature", temperature)
        if self.velocity is not None:
            velocity = np.ascontiguousarray(self.velocity, np.float32)
            if velocity.shape != density.shape + (3,):
                raise ValueError("Volume velocity must be (nx, ny, nz, 3), matching the density")
            object.__setattr__(self, "velocity", velocity)
        if self.flame is not None:
            flame = np.ascontiguousarray(self.flame, np.float32)
            if flame.shape != density.shape:
                raise ValueError("Volume flame must match the density shape")
            object.__setattr__(self, "flame", flame)
        if self.fuel is not None:
            fuel = np.ascontiguousarray(self.fuel, np.float32)
            if fuel.shape != density.shape:
                raise ValueError("Volume fuel must match the density shape")
            object.__setattr__(self, "fuel", fuel)
        object.__setattr__(self, "voxel_size", float(self.voxel_size))
        object.__setattr__(self, "origin", tuple(float(v) for v in self.origin))
        object.__setattr__(self, "matrix", np.asarray(self.matrix, np.float32))

    @property
    def shape(self):
        return self.density.shape

    def fingerprint(self):
        """Hex digest of every field, so a cache keys on content: equal volumes agree, an edit changes it."""
        h = hashlib.sha256()
        for name, array in (("density", self.density), ("temperature", self.temperature),
                            ("velocity", self.velocity), ("matrix", self.matrix), ("flame", self.flame),
                            ("fuel", self.fuel)):
            h.update(name.encode())
            if array is not None:
                h.update(str(array.shape).encode())
                h.update(np.ascontiguousarray(array).tobytes())
        h.update(repr((self.voxel_size, self.origin)).encode())
        return h.hexdigest()


def analytic_plume(resolution=32, seed=0):
    """A deterministic rising smoke plume with a matching velocity field, for tests and demos.

    The domain is a unit cube (`voxel_size` 1 / resolution) with x and z centred on zero and y from 0
    to 1. Density is a Gaussian column that widens and thins with height, broken up by seeded low
    frequency noise; temperature is density fading with height; velocity rises, swirls about the y
    axis and spreads with height. The same (resolution, seed) always gives identical arrays.
    """
    n = int(resolution)
    if n < 4:
        raise ValueError("analytic_plume resolution must be at least 4")
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    coarse = rng.rand(5, 5, 5).astype(np.float64)
    axis = (np.arange(n) + .5) / n
    x, y, z = np.meshgrid(axis - .5, axis, axis - .5, indexing="ij")
    # Trilinear upsample of the coarse noise: exact and free of library dependence.
    pos = np.stack((axis, axis, axis)) * 4      # coarse-grid coordinates in [0, 4]
    i0 = np.minimum(pos.astype(int), 3)
    f = pos - i0
    noise = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wx = (f[0] if dx else 1 - f[0])[:, None, None]
                wy = (f[1] if dy else 1 - f[1])[None, :, None]
                wz = (f[2] if dz else 1 - f[2])[None, None, :]
                noise = noise + wx * wy * wz * coarse[np.ix_(i0[0] + dx, i0[1] + dy, i0[2] + dz)]
    radius = .06 + .16 * y
    r2 = x * x + z * z
    density = np.exp(-r2 / (2 * radius * radius)) * (1.0 - .55 * y) * (1.0 + .9 * (noise - .5))
    density = np.clip(density, 0, None) * (y > 0)
    temperature = density * (1.0 - .7 * y)
    swirl = 1.2 * np.exp(-r2 / (2 * (.2 * .2)))
    velocity = np.stack((-swirl * z + .3 * x * y, np.full_like(x, .8) + .4 * y, swirl * x + .3 * z * y), -1)
    return Volume(density.astype(np.float32), 1.0 / n, (-.5, 0.0, -.5), _IDENTITY,
                  temperature.astype(np.float32), velocity.astype(np.float32))


@dataclass(frozen=True, eq=False)
class Scene:
    geometries: tuple[Geometry, ...] = ()
    lights: tuple[Light, ...] = ()
    splats: tuple = ()
    particles: tuple = ()
    volumes: tuple = ()
    environments: tuple = ()   # envlight.Environment items: image-based light for meshes and splats
    instances: tuple = ()      # InstanceSet items (Instance3D); expand_instances turns them into geometries


def write_obj(scene, path):
    """Atomically write world-space OBJ geometry, returning object/vertex/triangle counts.

    UVs and per-vertex normals are preserved. Lights, colours, textures and projections
    are not exported. Each geometry becomes one object; transforms are baked into positions.
    """
    scene = resolve_instances(scene)
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


def material_fields(p):
    """The liquid-material Geometry fields of a node's params; a document saved before liquids has none of them."""
    return dict(material=str(p.get("material", "standard")),
                ior=float(p.get("ior", 1.333)),
                absorption_color=tuple(float(p.get(f"absorption_{c}", d))
                                       for c, d in (("red", 0.55), ("green", 0.8), ("blue", 0.95))),
                absorption_distance=float(p.get("absorption_distance", 1.0)),
                reflection=float(p.get("reflection", 1.0)),
                roughness=float(p.get("roughness", 0.0)))


def geometry_from_node(node, texture=None):
    p = node["params"]
    name = node.get("name", "")
    return replace(_geometry_from_node(node, texture),
                   specular=float(p.get("spec_amount", 0.0)),
                   shininess=float(p.get("spec_shininess", 32.0)),
                   emission=float(p.get("emission", 0.0)),
                   metallic=float(p.get("metallic", 0.0)),
                   pbr_roughness=float(p.get("pbr_roughness", 0.5)),
                   pbr_specular=float(p.get("pbr_specular", 0.5)),
                   name=name, asset=name, **material_fields(p))


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


MAX_INSTANCE_SOURCES = 8


def _points_from_value(value):
    """(positions, velocities, normals, colors, ages, ids) float64/int64 arrays, any of them None
    except positions, describing the points an Instance3D copies onto (docs/3D_ROADMAP.md).

    A `ParticleInstance` supplies whatever it solved. A `Geometry` (or a `Scene`'s geometries)
    supplies its world-space vertices as points, with world-space normals when it has them; it has
    no velocity, colour, age or id, so those come back None.
    """
    if isinstance(value, ParticleInstance):
        matrix = value.matrix.astype(np.float64)
        linear = matrix[:3, :3]
        positions = (linear @ np.asarray(value.positions, np.float64).T).T + matrix[:3, 3]
        velocities = None if value.velocities is None else (linear @ np.asarray(value.velocities, np.float64).T).T
        colors = None if value.colors is None else np.asarray(value.colors, np.float64)
        ages = None if value.ages is None else np.asarray(value.ages, np.float64)
        ids = None if value.ids is None else np.asarray(value.ids, np.int64)
        return positions, velocities, None, colors, ages, ids
    geometries = value.geometries if isinstance(value, Scene) else (() if value is None else (value,))
    positions, normals = [], []
    any_normals = True
    for geometry in geometries:
        if not len(geometry.vertices):
            continue
        matrix = geometry.world_matrix().astype(np.float64)
        linear = matrix[:3, :3]
        positions.append((linear @ geometry.vertices.astype(np.float64).T).T + matrix[:3, 3])
        if geometry.normals is not None:
            n = (np.linalg.inv(linear).T @ geometry.normals.astype(np.float64).T).T
            normals.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8))
        else:
            any_normals = False
    if not positions:
        return np.zeros((0, 3)), None, None, None, None, None
    return np.concatenate(positions), None, (np.concatenate(normals) if any_normals else None), None, None, None


def _sources_from_value(value):
    """Up to `MAX_INSTANCE_SOURCES` variant meshes (Instance3D's `instance` input)."""
    if value is None:
        return ()
    geometries = value.geometries if isinstance(value, Scene) else (value,)
    return tuple(g for g in geometries if g is not None and len(g.triangles))[:MAX_INSTANCE_SOURCES]


def _basis_from_axis(axis):
    """(N,3,3) rotation matrices whose local +Z is `axis`; a near-zero axis (a still particle)
    keeps the identity rather than pick an arbitrary orientation."""
    axis = np.asarray(axis, np.float64)
    n = len(axis)
    length = np.linalg.norm(axis, axis=1)
    still = length <= 1e-9
    safe_length = np.where(still, 1.0, length)
    z = np.where(still[:, None], np.array([0., 0., 1.]), axis / safe_length[:, None])
    world_up = np.tile(np.array([0., 1., 0.]), (n, 1))
    world_up[np.abs(z[:, 1]) > 0.999] = (1., 0., 0.)
    x = np.cross(world_up, z)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    y = np.cross(z, x)
    basis = np.stack((x, y, z), axis=2)
    basis[still] = np.eye(3)
    return basis


def _random_unit_axis(u, v):
    """A uniformly distributed unit vector per row from two independent [0, 1) values."""
    z = u * 2.0 - 1.0
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = v * 2.0 * np.pi
    return np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1)


def _rodrigues(axis, degrees):
    """(N,3,3) rotation matrices by `degrees` about each row's (not necessarily unit) `axis`."""
    axis = np.asarray(axis, np.float64)
    axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    theta = np.radians(np.asarray(degrees, np.float64))
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros_like(x)
    k = np.stack((np.stack((zero, -z, y), axis=1),
                  np.stack((z, zero, -x), axis=1),
                  np.stack((-y, x, zero), axis=1)), axis=1)
    eye = np.tile(np.eye(3), (len(axis), 1, 1))
    return eye + s[:, None, None] * k + (1.0 - c)[:, None, None] * (k @ k)


def instances_from_node(points_value, instance_value, params):
    """Instance3D: an `InstanceSet` copying `instance_value`'s meshes onto `points_value`'s points.

    `scale`/`scale_random` are a uniform factor and a seeded +/- fraction of it (docs/3D_ROADMAP.md
    "Instancing" test: the result always stays within `scale * (1 +/- scale_random)`). `orient`
    aligns each instance's local +Z to the point's velocity or normal (falling back to no rotation
    when the point set has neither); `rotate_random` then jitters that orientation by up to that many
    degrees about a random axis, and `spin` adds a further rotation about the instance's own (now
    final) local Z of `spin` degrees per frame of `age`. `variant` picks a source mesh per point:
    "cycle" round-robins in point order, "random" is seeded per point, "attribute" uses the point's
    particle id (a geometry's points have no other integer attribute, so it falls back to point
    index). `color_from_points` tints each instance's material colour by the point's colour, when
    the points carry one. Every random draw is seeded by `seed`, so the same seed reproduces the
    same instances bit-for-bit (`particles._hash_unit`, the same generator ParticleEmitter3D uses).
    """
    from .particles import _hash_unit
    positions, velocities, normals, colors, ages, ids = _points_from_value(points_value)
    sources = _sources_from_value(instance_value)
    n = len(positions)
    if not sources or not n:
        return InstanceSet((), np.zeros((0, 4, 4)), np.zeros(0, np.int32))
    seed = int(params.get("seed", 0))
    index = np.arange(n, dtype=np.int64)
    orient = params.get("inst_orient", "none")
    if orient == "velocity" and velocities is not None:
        rotation = _basis_from_axis(velocities)
    elif orient == "normal" and normals is not None:
        rotation = _basis_from_axis(normals)
    elif orient == "random":
        axis = _random_unit_axis(_hash_unit(seed, 521, index), _hash_unit(seed, 522, index))
        rotation = _rodrigues(axis, _hash_unit(seed, 523, index) * 360.0)
    else:
        rotation = np.tile(np.eye(3), (n, 1, 1))
    rotate_random = float(params.get("inst_rotate_random", 0.0))
    if rotate_random > 0:
        axis = _random_unit_axis(_hash_unit(seed, 531, index), _hash_unit(seed, 532, index))
        jitter = (_hash_unit(seed, 533, index) * 2.0 - 1.0) * rotate_random
        rotation = _rodrigues(axis, jitter) @ rotation
    spin = float(params.get("inst_spin", 0.0))
    if spin and ages is not None:
        rotation = _rodrigues(rotation[:, :, 2], spin * ages) @ rotation
    scale = float(params.get("inst_scale", 1.0))
    scale_random = float(params.get("inst_scale_random", 0.0))
    factor = scale if scale_random <= 0 else scale * (1.0 + (_hash_unit(seed, 541, index) * 2.0 - 1.0) * scale_random)
    factor = np.maximum(np.broadcast_to(factor, (n,)).astype(np.float64), 0.0)
    matrices = np.tile(np.eye(4), (n, 1, 1))
    matrices[:, :3, :3] = rotation * factor[:, None, None]
    matrices[:, :3, 3] = positions
    variant_mode = params.get("inst_variant", "cycle")
    count = len(sources)
    if variant_mode == "random":
        variant = np.floor(_hash_unit(seed, 551, index) * count).astype(np.int64) % count
    elif variant_mode == "attribute":
        variant = np.mod((ids if ids is not None else index).astype(np.int64), count)
    else:
        variant = index % count
    tint = colors.astype(np.float32) if (params.get("inst_color_from_points") and colors is not None) else None
    return InstanceSet(sources, matrices, variant.astype(np.int32), colors=tint,
                       ids=None if ids is None else ids.astype(np.int64),
                       velocities=None if velocities is None else np.asarray(velocities, np.float64))


def expand_instances(instance_set):
    """Geometry per instance (Instance3D), each sharing its source mesh's vertex and triangle
    arrays by reference: N instances of one mesh cost O(1) triangles in memory, not O(N)."""
    if not len(instance_set) or not instance_set.sources:
        return ()
    parent = instance_set.parent.astype(np.float64)
    bases = [source.world_matrix().astype(np.float64) for source in instance_set.sources]
    out = []
    for i in range(len(instance_set)):
        variant = int(instance_set.variant[i])
        source = instance_set.sources[variant]
        matrix = (parent @ instance_set.matrices[i] @ bases[variant]).astype(np.float32)
        color = source.color
        if instance_set.colors is not None:
            tint = instance_set.colors[i]
            color = tuple(float(c) * float(t) for c, t in zip(color, tint))
        # Each instance is its own Cryptomatte object: the source mesh's name plus the instance's
        # id (or its index when the points carry none), so repeats of one mesh still separate.
        instance_id = int(instance_set.ids[i]) if instance_set.ids is not None else i
        name = f"{source.name or 'instance'}_{instance_id}"
        out.append(replace(source, transform=Transform3D(), parent=matrix, color=color, name=name))
    return tuple(out)


def resolve_instances(scene):
    """Expand every `scene.instances` into ordinary geometries, for consumers (render, WriteGeo3D,
    the USD/OBJ exporters) that only know `Scene.geometries`. A scene with none is unchanged."""
    if not isinstance(scene, Scene) or not scene.instances:
        return scene
    expanded = list(scene.geometries)
    for instance_set in scene.instances:
        expanded.extend(expand_instances(instance_set))
    return replace(scene, geometries=tuple(expanded), instances=())


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


# --- Shrinkwrap3D --------------------------------------------------------------------------------

def _target_triangles(value):
    """World-space triangles (T,3,3) of a geometry or a scene (Shrinkwrap3D's `target`).

    Each geometry's full `world_matrix` (its own transform and any enclosing Scene3D/Axis3D parent)
    is baked in, matching how `merge_geometry` and `particles.collider_triangles` flatten a wired
    geometry-or-scene input into one world-space triangle soup.
    """
    geometries = value.geometries if isinstance(value, Scene) else (() if value is None else (value,))
    parts = []
    for geometry in geometries:
        if not len(geometry.vertices) or not len(geometry.triangles):
            continue
        matrix = geometry.world_matrix().astype(np.float64)
        world = (matrix[:3, :3] @ geometry.vertices.astype(np.float64).T).T + matrix[:3, 3]
        parts.append(world[geometry.triangles])
    return np.concatenate(parts) if parts else np.zeros((0, 3, 3), np.float64)


def _closest_on_triangles(points, triangles):
    """The closest point and that triangle's flat normal, per query point, over a triangle soup.

    Exact per-triangle closest point (the Voronoi-region case analysis of Ericson's "Real-Time
    Collision Detection"), brute-forced over every triangle and chunked over query points to bound
    memory: O(P x T), a correctness reference like the rest of this module, not a throughput claim.
    Degenerate (zero-area) triangles fall back to vertex `a` with a zero normal; `np.argmin` below
    never selects them over a real triangle unless every triangle is degenerate.
    """
    if not len(triangles):
        raise ValueError("Shrinkwrap3D: the target has no faces to wrap onto")
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    cross = np.cross(ab, ac)
    area = np.linalg.norm(cross, axis=1)
    face_normal = cross / np.maximum(area, 1e-300)[:, None]
    points = np.asarray(points, np.float64)
    closest = np.empty((len(points), 3))
    normals = np.empty((len(points), 3))
    chunk = max(1, min(512, 4_000_000 // max(len(triangles), 1)))
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        ap = p[:, None, :] - a[None, :, :]
        d1 = np.einsum('pij,ij->pi', ap, ab)
        d2 = np.einsum('pij,ij->pi', ap, ac)
        region_a = (d1 <= 0) & (d2 <= 0)

        bp = p[:, None, :] - b[None, :, :]
        d3 = np.einsum('pij,ij->pi', bp, ab)
        d4 = np.einsum('pij,ij->pi', bp, ac)
        region_b = (d3 >= 0) & (d4 <= d3)

        vc = d1 * d4 - d3 * d2
        v_ab = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 - d3) != 0)
        region_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)

        cp = p[:, None, :] - c[None, :, :]
        d5 = np.einsum('pij,ij->pi', cp, ab)
        d6 = np.einsum('pij,ij->pi', cp, ac)
        region_c = (d6 >= 0) & (d5 <= d6)

        vb = d5 * d2 - d1 * d6
        w_ac = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 - d6) != 0)
        region_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)

        va = d3 * d6 - d5 * d4
        denom_bc = (d4 - d3) + (d5 - d6)
        w_bc = np.divide(d4 - d3, denom_bc, out=np.zeros_like(d4), where=denom_bc != 0)
        region_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

        denom = va + vb + vc
        vv = np.divide(vb, denom, out=np.zeros_like(vb), where=denom != 0)
        ww = np.divide(vc, denom, out=np.zeros_like(vc), where=denom != 0)

        pts = np.broadcast_to(a, (p.shape[0],) + a.shape).copy()
        seen = region_a.copy()
        pts[region_b & ~seen] = np.broadcast_to(b, pts.shape)[region_b & ~seen]
        seen |= region_b
        edge_ab = a[None] + v_ab[..., None] * ab[None]
        take = region_ab & ~seen
        pts[take] = edge_ab[take]
        seen |= region_ab
        take = region_c & ~seen
        pts[take] = np.broadcast_to(c, pts.shape)[take]
        seen |= region_c
        edge_ac = a[None] + w_ac[..., None] * ac[None]
        take = region_ac & ~seen
        pts[take] = edge_ac[take]
        seen |= region_ac
        edge_bc = b[None] + w_bc[..., None] * (c - b)[None]
        take = region_bc & ~seen
        pts[take] = edge_bc[take]
        seen |= region_bc
        face_pt = a[None] + vv[..., None] * ab[None] + ww[..., None] * ac[None]
        pts[~seen] = face_pt[~seen]

        distance2 = np.sum((p[:, None, :] - pts) ** 2, axis=-1)
        best = np.argmin(distance2, axis=1)
        rows = np.arange(len(p))
        closest[start:start + chunk] = pts[rows, best]
        normals[start:start + chunk] = face_normal[best]
    return closest, normals


def _project_onto_triangles(origins, directions, triangles):
    """First hit walking `directions` from `origins` into a triangle soup, and that triangle's normal.

    Origins with no hit (a proxy that does not fully enclose a concave target along that vertex's
    normal) fall back to `_closest_on_triangles`, so `Shrinkwrap3D` "project" mode is always defined.
    """
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    cross = np.cross(e1, e2)
    face_normal = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-300)
    tri_set = TriangleSet(v0, e1, e2, 1.0)
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    bvh = Bvh.build(lo, hi)
    t, prim, _, _ = tri_set.closest_hit(bvh, origins, directions, tmin=1e-6)
    hit = prim >= 0
    landing = origins + directions * np.where(hit, t, 0.0)[:, None]
    normal = np.where(hit[:, None], face_normal[np.clip(prim, 0, None)], 0.0)
    if not hit.all():
        fallback_points, fallback_normals = _closest_on_triangles(origins[~hit], triangles)
        landing[~hit] = fallback_points
        normal[~hit] = fallback_normals
    return landing, normal


def _laplacian_smooth(vertices, triangles):
    """One pass of uniform Laplacian smoothing over the mesh's own edges (positions only)."""
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
    edges = np.concatenate((edges, edges[:, ::-1]))
    v = vertices.astype(np.float64)
    sums = np.zeros_like(v)
    counts = np.zeros(len(v))
    np.add.at(sums, edges[:, 0], v[edges[:, 1]])
    np.add.at(counts, edges[:, 0], 1)
    counts = np.maximum(counts, 1)
    return (sums / counts[:, None]).astype(np.float32)


def _proxy_sphere(center, triangles, resolution, color):
    points = triangles.reshape(-1, 3)
    radius = float(np.linalg.norm(points - center, axis=1).max()) if len(points) else 1.0
    geometry = _sphere_grid(max(radius, 1e-6), max(2, resolution), max(3, resolution * 2), color, Transform3D())
    return replace(geometry, vertices=(geometry.vertices + center).astype(np.float32))


def _proxy_cylinder(center, triangles, resolution, color):
    points = triangles.reshape(-1, 3)
    cx, cy, cz = center
    radial = points[:, (0, 2)] - center[[0, 2]] if len(points) else np.zeros((1, 2))
    radius = max(float(np.sqrt((radial ** 2).sum(axis=1)).max()) if len(radial) else 1.0, 1e-6)
    height = max(float(points[:, 1].max() - points[:, 1].min()) if len(points) else 2.0, 1e-6)
    cols, rows = max(3, int(resolution)), max(1, int(resolution) // 2)
    h = height / 2.0
    theta, yy = np.meshgrid(np.linspace(0, 2 * np.pi, cols + 1), np.linspace(-h, h, rows + 1))
    side_vertices = np.stack((cx + radius * np.cos(theta), cy + yy, cz + radius * np.sin(theta)), -1
                             ).reshape(-1, 3)
    uu, vv = np.meshgrid(np.linspace(0, 0.7, cols + 1), np.linspace(0, 1, rows + 1))
    side_uvs = np.stack((uu, vv), -1).reshape(-1, 2)
    side_tris = []
    for r in range(rows):
        for c in range(cols):
            va, vb = r * (cols + 1) + c, r * (cols + 1) + c + 1
            vd, ve = va + cols + 1, vb + cols + 1
            side_tris += [(va, vb, ve), (va, ve, vd)]

    def cap(y_value, uv_cy, top):
        angle = np.linspace(0, 2 * np.pi, cols, endpoint=False)
        ring = np.stack((cx + radius * np.cos(angle), np.full(cols, y_value), cz + radius * np.sin(angle)), -1)
        vertices = np.vstack(([[cx, y_value, cz]], ring))
        uvs = np.vstack(([[0.85, uv_cy]],
                         np.stack((0.85 + 0.13 * np.cos(angle), uv_cy + 0.13 * np.sin(angle)), -1)))
        tris = []
        for c in range(cols):
            n = (c + 1) % cols + 1
            tris.append((0, n, c + 1) if top else (0, c + 1, n))
        return vertices.astype(np.float32), uvs.astype(np.float32), np.array(tris, np.int32)

    bottom_v, bottom_uv, bottom_t = cap(cy - h, 0.25, False)
    top_v, top_uv, top_t = cap(cy + h, 0.75, True)
    vertices = np.concatenate((side_vertices, bottom_v, top_v)).astype(np.float32)
    uvs = np.concatenate((side_uvs, bottom_uv, top_uv)).astype(np.float32)
    triangles_out = np.concatenate((
        np.array(side_tris, np.int32),
        bottom_t + len(side_vertices),
        top_t + len(side_vertices) + len(bottom_v))).astype(np.int32)
    return Geometry(vertices, triangles_out, color, Transform3D(), uvs=uvs)


def _proxy_box(center, triangles, resolution, color):
    points = triangles.reshape(-1, 3)
    lo, hi = (points.min(axis=0), points.max(axis=0)) if len(points) else (center - 1, center + 1)
    hx, hy, hz = ((hi - lo) / 2).tolist()
    hx, hy, hz = max(hx, 1e-6), max(hy, 1e-6), max(hz, 1e-6)
    cx, cy, cz = center
    resolution = max(1, int(resolution))
    # A six-face cross: left/front/right/back in the middle row, top above front, bottom below it.
    faces = ((lambda u, v: (cx + u * hx, cy + v * hy, cz + hz), (0.25, 1 / 3, 0.5, 2 / 3)),    # front +Z
            (lambda u, v: (cx - u * hx, cy + v * hy, cz - hz), (0.75, 1 / 3, 1.0, 2 / 3)),     # back -Z
            (lambda u, v: (cx + hx, cy + v * hy, cz - u * hz), (0.5, 1 / 3, 0.75, 2 / 3)),     # right +X
            (lambda u, v: (cx - hx, cy + v * hy, cz + u * hz), (0.0, 1 / 3, 0.25, 2 / 3)),     # left -X
            (lambda u, v: (cx + u * hx, cy + hy, cz - v * hz), (0.25, 2 / 3, 0.5, 1.0)),       # top +Y
            (lambda u, v: (cx + u * hx, cy - hy, cz + v * hz), (0.25, 0.0, 0.5, 1 / 3)))       # bottom -Y
    steps = np.linspace(-1, 1, resolution + 1)
    vertices, uvs, triangles_out, offset = [], [], [], 0
    for position, (u0, v0, u1, v1) in faces:
        uu, vv = np.meshgrid(steps, steps)
        xs, ys, zs = (np.broadcast_to(a, uu.shape) for a in position(uu, vv))
        vertices.append(np.stack((xs, ys, zs), -1).reshape(-1, 3))
        gu, gv = np.meshgrid(np.linspace(u0, u1, resolution + 1), np.linspace(v0, v1, resolution + 1))
        uvs.append(np.stack((gu, gv), -1).reshape(-1, 2))
        for r in range(resolution):
            for c in range(resolution):
                va, vb = offset + r * (resolution + 1) + c, offset + r * (resolution + 1) + c + 1
                vd, ve = va + resolution + 1, vb + resolution + 1
                triangles_out += [(va, vb, ve), (va, ve, vd)]
        offset += (resolution + 1) ** 2
    return Geometry(np.concatenate(vertices).astype(np.float32), np.array(triangles_out, np.int32),
                    color, Transform3D(), uvs=np.concatenate(uvs).astype(np.float32))


def shrinkwrap_geometry(target, proxy, params) -> Geometry:
    """Shrinkwrap3D: fit `proxy` (or a generated one) onto `target`'s surface.

    Computed in world space throughout: `target` (a Geometry or Scene) is flattened to world-space
    triangles (`_target_triangles`); a wired `proxy` is baked to world space too, keeping its own UVs
    exactly as authored, while an unwired one gets a fresh enclosing primitive (`wrap_shape` at
    `wrap_resolution`) with clean UVs instead. Each proxy vertex then finds the target, either the
    closest point over the whole surface (`wrap_mode` "nearest") or the first hit walking inward
    along the proxy's own vertex normal (`wrap_mode` "project"). `wrap_offset` moves that landing
    point along the target's (flat, per-triangle) normal, `wrap_falloff` blends between the original
    proxy position and the wrapped one, and `wrap_smooth_iterations` Laplacian-smooths the result
    over the proxy's own edges afterward, a pass that only ever touches positions, never UVs.
    """
    triangles = _target_triangles(target)
    if not len(triangles):
        raise ValueError("Shrinkwrap3D: the target has no faces to wrap onto")
    if proxy is not None:
        matrix = proxy.world_matrix().astype(np.float64)
        base = replace(proxy, vertices=((matrix[:3, :3] @ proxy.vertices.astype(np.float64).T).T +
                                        matrix[:3, 3]).astype(np.float32),
                      transform=Transform3D(), parent=_IDENTITY.copy())
    else:
        lo, hi = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
        center = (lo + hi) / 2
        shape = str(params.get("wrap_shape", "sphere"))
        resolution = max(2, int(params.get("wrap_resolution", 16)))
        color = tuple(float(params.get(k, d)) for k, d in
                      (("red", 0.8), ("green", 0.8), ("blue", 0.8), ("alpha", 1.0)))
        builder = {"sphere": _proxy_sphere, "cylinder": _proxy_cylinder, "box": _proxy_box}[shape]
        base = builder(center, triangles, resolution, color)
    base = recompute_normals(base)
    original = base.vertices.astype(np.float64)
    mode = str(params.get("wrap_mode", "nearest"))
    if mode == "project":
        landing, normal = _project_onto_triangles(original, -base.normals.astype(np.float64), triangles)
    else:
        landing, normal = _closest_on_triangles(original, triangles)
    offset = float(params.get("wrap_offset", 0.0))
    wrapped = landing + normal * offset
    falloff = float(np.clip(params.get("wrap_falloff", 1.0), 0.0, 1.0))
    vertices = original + (wrapped - original) * falloff
    for _ in range(max(0, int(params.get("wrap_smooth_iterations", 0)))):
        vertices = _laplacian_smooth(vertices, base.triangles)
    return recompute_normals(replace(base, vertices=vertices.astype(np.float32)))


def _kelvin_to_rgb(kelvin):
    """Approximate blackbody colour, linear 0..1 RGB normalised so 6500K is near-white (Tanner
    Helland's fit); `kelvin` is clamped to the LIMITS range (1000..40000)."""
    t = float(np.clip(kelvin, 1000.0, 40000.0)) / 100.0
    red = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    green = (99.4708025861 * math.log(t) - 161.1195681661 if t <= 66
             else 288.1221695283 * (t - 60) ** -0.0755148492)
    blue = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    return tuple(float(c) for c in np.clip((red, green, blue), 0, 255) / 255.0)


_AREA = ("Rect", "Disc", "Sphere")   # R2: real-size emitters, see `_area_light_contribution`


def light_from_node(node, image=None):
    """A `Light` for a Directional, Point, Spot, Rect, Disc or Sphere Light3D; an `envlight.Environment`
    for an Environment one.

    `image` is the scene-linear RGB (H, W, 3) equirectangular map wired to an Environment light; without
    one the light is a uniform sky of its colour, which lights everything evenly.
    """
    p = node["params"]
    kind = p["light_type"]
    if kind == "Environment":
        from . import envlight
        rgb = np.ones((16, 32, 3), np.float32) if image is None else np.asarray(image, np.float32)
        return envlight.Environment(rgb, envlight.fingerprint_of(rgb), float(p["intensity"]),
                                    float(p.get("env_rotation", 0.0)), float(p.get("env_blur", 0.0)),
                                    (float(p["red"]), float(p["green"]), float(p["blue"])))
    color = (float(p["red"]), float(p["green"]), float(p["blue"]))
    intensity = float(p["intensity"])
    if kind in _AREA:
        if p.get("light_color_mode", "RGB") == "Kelvin":
            color = _kelvin_to_rgb(float(p.get("kelvin", 6500.0)))
        intensity *= 2.0 ** float(p.get("exposure", 0.0))
    return Light(kind, color,
                 intensity, Vec3(p["tx"], p["ty"], p["tz"]),
                 Vec3(p["target_x"], p["target_y"], p["target_z"]),
                 shadows=p.get("shadows", "off") == "on",
                 cone_angle=float(p.get("cone_angle", 30.0)),
                 cone_penumbra_angle=float(p.get("cone_penumbra_angle", 5.0)),
                 cone_falloff=float(p.get("cone_falloff", 1.0)),
                 falloff_type=p.get("falloff_type", "No falloff"),
                 shadow_bias=float(p.get("shadow_bias", SHADOW_BIAS_DEFAULT)),
                 shadow_blur=float(p.get("shadow_blur", 0.0)),
                 shadow_samples=int(p.get("shadow_samples", 1)),
                 area_width=float(p.get("area_width", 1.0)),
                 area_height=float(p.get("area_height", 1.0)),
                 area_radius=float(p.get("area_radius", 0.5)),
                 area_normalize=p.get("area_normalize", "off") == "on",
                 two_sided=p.get("two_sided", "off") == "on",
                 light_samples=int(p.get("light_samples", 4)))


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
                  p["roll"], float(p["haperture"]), float(p["vaperture"]),
                  float(p.get("fstop", 0.0)), float(p.get("focus_distance", 5.0)),
                  int(p.get("aperture_blades", 0)), float(p.get("blade_rotation", 0.0)),
                  float(p.get("anamorphic_squeeze", 1.0)))


def scene_from_node(node, members):
    """Assemble geometry, lights, splats and nested scenes under this node's transform."""
    matrix = _transform_from(node["params"]).matrix()
    geometries, lights, splats, particles, volumes, environments, instances = [], [], [], [], [], [], []
    for member in members:
        if isinstance(member, Scene):
            items = (member.geometries + member.lights + member.splats + member.particles
                     + member.volumes + member.environments + member.instances)
        else:
            items = (member,)
        for item in items:
            if isinstance(item, Environment):
                environments.append(replace(item, parent=matrix @ item.parent))
                continue
            if isinstance(item, SplatInstance):
                splats.append(replace(item, matrix=matrix @ item.matrix))
                continue
            if isinstance(item, ParticleInstance):
                particles.append(replace(item, matrix=matrix @ item.matrix))
                continue
            if isinstance(item, Volume):
                volumes.append(replace(item, matrix=matrix @ item.matrix))
                continue
            if isinstance(item, InstanceSet):
                instances.append(replace(item, parent=matrix @ item.parent))
                continue
            fields = {**item.__dict__, "parent": matrix @ item.parent}
            if isinstance(item, Geometry):
                # The outermost Scene3D/Axis3D names the asset: nesting evaluates inside-out, so
                # the last (outermost) node to pass through here overwrites what an inner one set.
                node_name = node.get("name", "")
                if node_name:
                    fields["asset"] = node_name
            moved = type(item)(**fields)
            (geometries if isinstance(item, Geometry) else lights).append(moved)
    return Scene(tuple(geometries), tuple(lights), tuple(splats), tuple(particles), tuple(volumes),
                 tuple(environments), tuple(instances))


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


def _indirect_rays(scene):
    """Hemisphere rays the indirect-light pass casts for `scene`, after the quality presets."""
    from .splatindirect import effective_samples
    return sum(len(i.cloud) * effective_samples(i, 'indirect_samples') for i in scene.splats
               if getattr(i, 'relight', 0) > 0 and getattr(i, 'indirect_distance', 1.0) > 0)


def _indirect_instances(scene):
    return sum(1 for i in scene.splats if _indirect_rays(Scene(splats=(i,))) > 0)


def _indirect_budget(scene, triangle_count):
    rays = _indirect_rays(scene)
    casters = sum(len(i.cloud) for i in scene.splats if getattr(i, 'cast_shadows', True))
    work = _shadow_cost(rays, triangle_count) + (_shadow_cost(rays, casters) if casters else 0)
    if work > SPLAT_SHADOW_BUDGET:
        raise ValueError(f'Splat indirect-light rays exceed the CPU reference budget: {work:,.0f} estimated tests > '
                         f'{SPLAT_SHADOW_BUDGET:,}; lower Indirect samples or the Quality preset')


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
    volume_shadows: object = None    # volumerender.ShadowCasters: the scene's smoke darkens what is under it

    def visibility(self, position, normal, light, light_position, direction):
        self.work += _shadow_cost(len(position), self.triangle_count, build=False)
        (_raytrace_budget if self.raytrace else _shadow_budget)(self.work)
        p = self.primitives
        visibility = _shadow_visibility(position, normal, light, light_position, direction,
                                        p.v0, p.e1, p.e2, p.alpha, self.bias, self.cancel,
                                        triangles=p, bvh=self.bvh, splat_shadows=self.splat_shadows)
        if self.volume_shadows:
            visibility = (visibility * self.volume_shadows.transmittance(light, position)).astype(np.float32)
        return visibility

    def area_visibility(self, position, normal, targets, light):
        """Hard hit/miss transmittance from `position` straight to each of `targets` (N,3), one light
        sample per shading point. The caller (`_area_light_contribution`) already varies `targets`
        across its own `light_samples` loop, so no angular jitter is added here."""
        self.work += _shadow_cost(len(position), self.triangle_count, build=False)
        (_raytrace_budget if self.raytrace else _shadow_budget)(self.work)
        bias = _light_bias(self.bias, light)
        p = self.primitives
        origin = position + normal * bias
        ray = targets - origin
        limit = np.linalg.norm(ray, axis=1)
        ray = ray / np.maximum(limit[:, None], 1e-8)
        if self.bvh is None:
            value = p.brute_transmittance(origin, ray, bias * .01, limit,
                                          triangle_chunk=_SHADOW_TRIANGLE_CHUNK, cancel=self.cancel)
        else:
            value = p.transmittance(self.bvh, origin, ray, bias * .01, limit, cancel=self.cancel)
        if self.splat_shadows is not None:
            value = value * self.splat_shadows.primitives.transmittance(
                self.splat_shadows.bvh, origin, ray, bias * .01, limit,
                cutoff=SPLAT_SHADOW_CUTOFF, cancel=self.cancel)
        return value


# --- area lights (R2): Rect, Disc, Sphere -------------------------------------------------------

_R2_A, _R2_B = 0.7548776662466927, 0.5698402909980532   # the plastic-constant low-discrepancy 2D sequence


def _light_basis(direction):
    """Orthonormal (right, up) spanning the plane perpendicular to unit `direction`."""
    direction = np.asarray(direction, np.float64)
    up_ref = np.array((0., 1., 0.)) if abs(direction[1]) < .9 else np.array((1., 0., 0.))
    right = np.cross(direction, up_ref)
    right = right / max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, direction)
    return right, up


def _area_light_radiance(light):
    """(emitted radiance (3,), surface area) of a Rect/Disc/Sphere light from its intensity and
    `area_normalize` (power, independent of size, vs. a plain radiance that dims as the light shrinks
    the way a real emitter's would)."""
    if light.kind == "Rect":
        area = max(float(light.area_width), 0.0) * max(float(light.area_height), 0.0)
    elif light.kind == "Disc":
        area = math.pi * max(float(light.area_radius), 0.0) ** 2
    else:  # Sphere
        area = 4 * math.pi * max(float(light.area_radius), 0.0) ** 2
    radiance = np.asarray(light.color, np.float64) * float(light.intensity)
    if light.area_normalize and area > 1e-12:
        radiance = radiance / (area * math.pi)
    return radiance, area


def _area_light_samples(light, count):
    """(count, 3) world sample points on the light's surface and their outward normals (Sphere: the
    radial direction at each point), a fixed low-discrepancy set reused by every shading point, the
    way a real light's finite set of quadrature points would be."""
    position, direction = light.world()
    position = position.astype(np.float64)
    k = np.arange(count, dtype=np.float64)
    u, v = (k * _R2_A) % 1.0, (k * _R2_B) % 1.0
    if light.kind == "Sphere":
        z = 1 - 2 * u
        r = np.sqrt(np.maximum(1 - z * z, 0))
        phi = 2 * np.pi * v
        local = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)
        points = position + local * float(light.area_radius)
        return points.astype(np.float32), local.astype(np.float32)
    right, up = _light_basis(direction.astype(np.float64))
    if light.kind == "Rect":
        ox = (u - .5) * max(float(light.area_width), 0.0)
        oy = (v - .5) * max(float(light.area_height), 0.0)
    else:  # Disc: Shirley-Chiu concentric square-to-disc map, then scale by radius
        a, b = 2 * u - 1, 2 * v - 1
        both_zero = (a == 0) & (b == 0)
        a_safe, b_safe = np.where(a == 0, 1e-12, a), np.where(b == 0, 1e-12, b)
        r = np.where(np.abs(a) > np.abs(b), a, b)
        theta = np.where(np.abs(a) > np.abs(b), (np.pi / 4) * (b / a_safe),
                         np.pi / 2 - (np.pi / 4) * (a / b_safe))
        theta = np.where(both_zero, 0.0, theta)
        ox = r * np.cos(theta) * float(light.area_radius)
        oy = r * np.sin(theta) * float(light.area_radius)
    points = position + right * ox[:, None] + up * oy[:, None]
    normals = np.broadcast_to(direction.astype(np.float64), points.shape)
    return points.astype(np.float32), normals.astype(np.float32)


def _area_light_contribution(position, normal, light, shadow_context):
    """(shadowed diffuse irradiance (N,3), representative to-light direction (N,3), average visibility
    (N,)) for a Rect/Disc/Sphere light: a Monte Carlo estimate of

        E(x) = (Area / count) * sum_k radiance * cos(light_k) * cos(surface_k) / dist_k^2 * visibility_k

    over `light.light_samples` fixed points on the light (`_area_light_samples`), exact in the limit
    and already softened (a bigger light or a longer `light_samples` count both narrow the estimator's
    error, and the shadow itself softens because each sample's ray lands somewhere else on the light).
    """
    count = int(np.clip(light.light_samples, 1, SHADOW_SAMPLES_MAX * 4))
    radiance, area = _area_light_radiance(light)
    light_points, light_normals = _area_light_samples(light, count)
    n = len(position)
    position64, normal64 = position.astype(np.float64), normal.astype(np.float64)
    total = np.zeros((n, 3), np.float64)
    rep_dir_sum = np.zeros((n, 3), np.float64)
    vis_sum = np.zeros(n, np.float64)
    for k in range(count):
        q, ln = light_points[k].astype(np.float64), light_normals[k].astype(np.float64)
        to_recv = position64 - q
        dist2 = np.maximum(np.einsum('ij,ij->i', to_recv, to_recv), 1e-10)
        dist = np.sqrt(dist2)
        wi = to_recv / dist[:, None]                       # light -> receiver, unit
        cos_light = wi @ ln
        cos_light = np.abs(cos_light) if light.two_sided else np.maximum(cos_light, 0.0)
        cos_surface = np.maximum(-np.einsum('ij,ij->i', wi, normal64), 0.0)
        weight = cos_light * cos_surface / dist2
        active = weight > 0
        vis = np.ones(n, np.float32)
        if shadow_context is not None and light.shadows and active.any():
            targets = np.broadcast_to(q.astype(np.float32), position.shape)
            vis[active] = shadow_context.area_visibility(position[active], normal[active], targets[active], light)
        total += (weight * vis)[:, None] * radiance
        rep_dir_sum += -wi
        vis_sum += vis
    irradiance = (total * (area / count)).astype(np.float32)
    if shadow_context is not None and getattr(shadow_context, 'volume_shadows', None) and light.shadows:
        irradiance = irradiance * shadow_context.volume_shadows.transmittance(light, position).astype(np.float32)[:, None]
    rep_dir = rep_dir_sum / count
    rep_dir = rep_dir / np.maximum(np.linalg.norm(rep_dir, axis=1, keepdims=True), 1e-8)
    return irradiance, rep_dir.astype(np.float32), (vis_sum / count).astype(np.float32)


def _area_light_shading(position, normal, light, shadow_context):
    """(diffuse irradiance (N,3), a to-light direction (N,3), specular colour (N,3)) for a Rect/Disc/
    Sphere light: the Monte Carlo diffuse irradiance above, and a centre-point, inverse-square
    approximation of the light for the specular highlight (no test in this step exercises area-light
    specular directly; the diffuse estimator is the one held to the analytic and penumbra tests)."""
    irradiance, _, vis_avg = _area_light_contribution(position, normal, light, shadow_context)
    center, _ = light.world()
    to_light = center.astype(np.float64) - position.astype(np.float64)
    dist2 = np.maximum(np.einsum('ij,ij->i', to_light, to_light), 1e-6)
    to_light = to_light / np.sqrt(dist2)[:, None]
    radiance, area = _area_light_radiance(light)
    specular_colour = (vis_avg / dist2)[:, None] * (radiance[None, :] * area)
    return irradiance, to_light.astype(np.float32), specular_colour.astype(np.float32)


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
        self.volume_shadows = None  # volumerender.ShadowCasters: smoke shadows the splats it hangs over

    def _volume_factor(self, light, points):
        return self.volume_shadows.transmittance(light, points) if self.volume_shadows else 1.0

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
        if self.mesh is None and not self.volume_shadows:
            return visibility
        cloud, matrix = _instance_parts(self.instances[index])
        catch_key = ('catch', _cloud_token(cloud), matrix.tobytes())
        for j, light in enumerate(self.lights):
            if not light.shadows or light.intensity <= 0:
                continue
            if self.mesh is not None:
                store = self._store(index, light, catch_key, len(cloud))
                missing = np.unique(indices[np.isnan(store[indices])])
                splat_shadow_stats["rays_traced"] += len(missing)
                splat_shadow_stats["rays_reused"] += len(indices) - len(missing)
                if len(missing):
                    store[missing] = self._trace_catch(light, cloud, matrix, missing)
                visibility[:, j] = store[indices]
            if self.volume_shadows:
                centres = (matrix[:3, :3] @ np.asarray(cloud.positions[indices], np.float64).T).T + matrix[:3, 3]
                visibility[:, j] *= self._volume_factor(light, centres)
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
            if self.volume_shadows:
                visibility[:, j] *= self._volume_factor(light, positions[indices].astype(np.float64))
        return visibility


def _radical_inverse(k):
    """Base-2 van der Corput sequence of the integers `k`, in [0, 1)."""
    k = np.asarray(k, dtype=np.uint64)
    result = np.zeros(len(k))
    scale = 0.5
    while k.any():
        result += scale * (k & 1)
        k = k >> np.uint64(1)
        scale *= 0.5
    return result


def _ggx_reflection_directions(mirror, normals, roughness, samples):
    """`samples` reflection directions per row, importance-sampled from the GGX lobe of `roughness`.

    Deterministic: a Hammersley set whose azimuth is turned by a golden-ratio hash of the row. One sample is
    the mirror direction itself. Directions that fall under the surface fall back to the mirror direction.
    """
    m = len(mirror)
    if samples <= 1:
        return mirror[:, None, :]
    alpha = np.maximum(np.asarray(roughness, dtype=np.float64), 0.05) ** 2
    k = np.arange(samples)
    u1 = ((k + 0.5) / samples)[None, :]
    turn = ((np.arange(m) * 0.6180339887498949) % 1.0)[:, None]
    phi = 2 * np.pi * ((_radical_inverse(k + 1)[None, :] + turn) % 1.0)
    a2 = (alpha ** 2)[:, None]
    cos_t = np.sqrt((1 - u1) / (1 + (a2 - 1) * u1))
    sin_t = np.sqrt(np.maximum(1 - cos_t ** 2, 0))
    helper = np.where(np.abs(normals[:, 1:2]) > 0.99, np.array((1.0, 0, 0)), np.array((0, 1.0, 0)))
    tangent = np.cross(helper, normals)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals, tangent)
    h = (sin_t * np.cos(phi))[..., None] * tangent[:, None] + (sin_t * np.sin(phi))[..., None] * bitangent[:, None] \
        + cos_t[..., None] * normals[:, None]
    view = 2 * np.sum(normals * mirror, axis=1, keepdims=True) * normals - mirror
    out = 2 * np.sum(view[:, None] * h, axis=2, keepdims=True) * h - view[:, None]
    below = np.sum(out * normals[:, None], axis=2) <= 0
    return np.where(below[..., None], mirror[:, None, :], out)


class _MeshReflector:
    """Ray-traced reflections of the meshes for de-lit splats, the CPU reference.

    Called as `reflect(positions, directions, normals, roughness, samples) -> (N,3)`: per splat, `samples`
    closest-hit rays along the mirror direction (one sample) or the GGX lobe (several) against the mesh set;
    a hit returns the surface colour lit by the scene's lights (no shadows), `ambient` and the environments'
    diffuse light, a miss returns the environment, read unblurred when the rays already span the lobe and at
    the splat's roughness when there is only the mirror ray. Splats do not reflect splats.
    """
    CHUNK = 16384

    def __init__(self, primitives, bvh, colors, lights, ambient, environments, bias, cancel):
        self.primitives, self.bvh, self.colors = primitives, bvh, colors
        self.lights, self.ambient, self.environments = lights, float(ambient), environments
        self.bias, self.cancel = bias, cancel

    def _shade_hits(self, points, normals, colors):
        radiance = np.full((len(points), 3), self.ambient)
        for light, light_position, direction in self.lights:
            if light.kind in _AREA:
                continue  # R2: mesh reflections seen by de-lit splats do not carry area lights yet
            if light.kind in _POSITIONAL:
                toward = light_position - points
                toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
            else:
                toward = np.broadcast_to(-direction.astype(np.float64), points.shape)
            lambert = np.maximum(np.sum(normals * toward, axis=1), 0)
            attenuation = _light_factor(light, points)
            if attenuation is not None:
                lambert = lambert * attenuation
            radiance += lambert[:, None] * (np.asarray(light.color, np.float64) * light.intensity)
        for environment in self.environments:
            radiance += environment.diffuse(normals)
        return colors * radiance

    def hit(self, origins, rays, tmax):
        """Closest mesh hit within `tmax` for `splatindirect`: `(t (N,), radiance (N,3))`, t inf on a miss.

        The radiance is the surface colour lit by the scene's lights (no shadows), `ambient` and the
        environments' diffuse light, on the side of the face that looks at the ray."""
        t, primitive, _, _ = self.primitives.closest_hit(self.bvh, origins, rays, self.bias, tmax, cancel=self.cancel)
        hit = primitive >= 0
        radiance = np.zeros((len(rays), 3))
        t = np.where(hit, t, np.inf)
        if hit.any():
            p = primitive[hit]
            face = np.cross(self.primitives.e1[p], self.primitives.e2[p])
            face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
            face = np.where((np.sum(face * rays[hit], axis=1) > 0)[:, None], -face, face)
            points = origins[hit] + t[hit, None] * rays[hit]
            radiance[hit] = self._shade_hits(points, face, self.colors[p])
        return t, radiance

    def __call__(self, positions, directions, normals, roughness, samples):
        positions = np.asarray(positions, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        roughness = np.asarray(roughness, dtype=np.float64)
        samples = max(1, int(samples))
        out = np.zeros((len(positions), 3))
        step = max(1, self.CHUNK // samples)
        for start in range(0, len(positions), step):
            _shadow_cancel(self.cancel)
            rows = slice(start, start + step)
            sampled = _ggx_reflection_directions(directions[rows], normals[rows], roughness[rows], samples)
            m = len(sampled)
            origins = np.repeat(positions[rows], samples, axis=0)
            rays = sampled.reshape(-1, 3)
            t, primitive, _, _ = self.primitives.closest_hit(self.bvh, origins, rays, self.bias, np.inf,
                                                             cancel=self.cancel)
            hit = primitive >= 0
            radiance = np.zeros((len(rays), 3))
            miss = ~hit
            if miss.any() and self.environments:
                level = np.repeat(roughness[rows], samples) if samples == 1 else np.zeros(len(rays))
                radiance[miss] = sum(e.specular(rays[miss], level[miss]) for e in self.environments)
            if hit.any():
                p = primitive[hit]
                face = np.cross(self.primitives.e1[p], self.primitives.e2[p])
                face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
                face = np.where((np.sum(face * rays[hit], axis=1) > 0)[:, None], -face, face)
                points = origins[hit] + t[hit, None] * rays[hit]
                radiance[hit] = self._shade_hits(points, face, self.colors[p])
            out[rows] = radiance.reshape(m, samples, 3).mean(axis=1)
        return out


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


def _mesh_environment_specular(environments, normal, toward_eye, geometry):
    """Environment reflection on a mesh material: the prefiltered light along the mirror direction, scaled
    by `specular`. The Blinn-Phong `shininess` maps to a GGX roughness with the usual sqrt(2 / (s + 2))."""
    v = toward_eye / np.maximum(np.linalg.norm(toward_eye, axis=1, keepdims=True), 1e-8)
    reflected = 2 * np.sum(normal * v, axis=1, keepdims=True) * normal - v
    roughness = np.full(len(normal), math.sqrt(2.0 / (float(geometry.shininess) + 2.0)))
    total = np.zeros((len(normal), 3), np.float32)
    for environment in environments:
        total += environment.specular(reflected, roughness).astype(np.float32) * float(geometry.specular)
    return total


def _mesh_pbr_environment(environments, normal, toward_eye, base_rgb, metallic, roughness, f0_dielectric):
    """Split-sum image-based light for a PBR mesh material: `(diffuse, specular)`, each (N,3).

    Mirrors `splatshade.environment_terms`'s dielectric/conductor blend and multiple-scattering
    compensation for a mesh's one constant metallic/roughness/F0 (meshes have no per-splat
    decomposition, and there are no traced mesh-to-mesh reflections here, unlike splats)."""
    from .envlight import dfg
    n = len(normal)
    zero = np.zeros((n, 3), np.float32)
    if not environments:
        return zero, zero
    v = toward_eye / np.maximum(np.linalg.norm(toward_eye, axis=1, keepdims=True), 1e-8)
    nv = np.einsum("ij,ij->i", normal, v)
    a, b = dfg(nv, roughness)
    compensation = 1 / np.maximum(a + b, 1e-4)

    def weight(f0):
        return (f0 * a[:, None] + b[:, None]) * (1 + f0 * (compensation[:, None] - 1))
    dielectric = weight(np.full((n, 3), f0_dielectric))
    m = float(np.clip(metallic, 0, 1))
    total = (1 - m) * dielectric + m * weight(base_rgb)
    kd = (1 - m) * (1 - dielectric[:, :1])
    direction = 2 * nv[:, None] * normal - v
    diffuse = sum((e.diffuse(normal).astype(np.float32) for e in environments), zero) * kd
    lookup = sum((e.specular(direction, roughness).astype(np.float32) for e in environments), zero)
    return diffuse, total * lookup


def _particle_shadow_scale(position, to_light, distance, particles, bias):
    """Hard-edged light visibility (N,) from `cast_shadows` ParticleInstance spheres between `position`
    and a light at `distance` along the unit direction `to_light` (R7 of 7, "particles cast a shadow"):
    an analytic ray-sphere test against every particle. None when nothing is occluded (the common case,
    so the caller's `scale` stays untouched). Chunked over `position` to bound peak memory at large
    fragment counts times particle counts.
    """
    casters = [i for i in particles if getattr(i, "cast_shadows", True) and len(i.positions)]
    total = sum(len(i.positions) for i in casters)
    if not total:
        return None
    n = len(position)
    blocked = np.zeros(n, bool)
    chunk = max(1, min(n, 200_000 // total))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        pos, dirn, dist = position[start:stop], to_light[start:stop], distance[start:stop]
        for instance in casters:
            matrix = instance.matrix.astype(np.float64)
            centers = (matrix[:3, :3] @ instance.positions.astype(np.float64).T).T + matrix[:3, 3]
            radii = 0.5 * np.asarray(instance.sizes, np.float64) * float(instance.size_scale)
            oc = pos[:, None, :] - centers[None, :, :]
            b = np.einsum("nmc,nc->nm", oc, dirn)
            c = np.einsum("nmc,nmc->nm", oc, oc) - radii[None, :] ** 2
            disc = b * b - c
            hit_dist = -b - np.sqrt(np.maximum(disc, 0))
            hits = (disc > 0) & (hit_dist > bias) & (hit_dist < dist[:, None])
            blocked[start:stop] |= hits.any(axis=1)
    if not blocked.any():
        return None
    scale = np.ones(n, np.float64)
    scale[blocked] = 0.0
    return scale


def _shade_pbr_mesh(position, normal, toward_eye, base_rgb, lights, ambient, environments,
                    metallic, roughness, f0_dielectric, shadow_context, need_specular, particle_occluders=()):
    """Cook-Torrance GGX shading of a mesh fragment for every scene light and the environment.

    Shares `splatshade._cook_torrance` with splat shading, so a mesh and a splat under one light
    with the same metallic/roughness/base colour match. Returns `(diffuse_radiance, specular)`,
    both (N,3); the caller premultiplies `diffuse_radiance` by `base_rgb` and alpha itself, the way
    the Blinn-Phong path already does, so unlit outputs (`albedo`, `depth`, ...) are untouched.
    `particle_occluders` (R7 of 7) are `scene.particles`: a `cast_shadows` instance between a fragment
    and a shadowed light darkens it, on top of `shadow_context`'s mesh shadows.
    """
    from .splatshade import _cook_torrance
    n = len(position)
    metallic = float(np.clip(metallic, 0, 1))
    roughness_arr = np.full(n, float(np.clip(roughness, 0, 1)))
    f0 = f0_dielectric * (1 - metallic) + base_rgb * metallic
    to_eye = toward_eye / np.maximum(np.linalg.norm(toward_eye, axis=1, keepdims=True), 1e-8)
    diffuse_radiance = np.full((n, 3), float(ambient), np.float32) * (1 - metallic)
    specular = np.zeros((n, 3), np.float32) if need_specular else None
    for light, light_position, direction in lights:
        if light.kind in _AREA:
            irradiance, to_light, spec_colour = _area_light_shading(position, normal, light, shadow_context)
            half = to_light + to_eye
            half = half / np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)
            vh = np.maximum(np.einsum("ij,ij->i", to_eye, half), 0)
            kd = (1 - metallic) * (1 - (0.04 + 0.96 * (1 - vh) ** 5))
            diffuse_radiance += kd[:, None] * irradiance
            if specular is not None:
                response, _ = _cook_torrance(normal, to_eye, to_light, roughness_arr, f0)
                specular += response * spec_colour
            continue
        if light.kind in _POSITIONAL:
            raw_to_light = light_position - position
            light_distance = np.linalg.norm(raw_to_light, axis=1)
            to_light = raw_to_light / np.maximum(light_distance, 1e-8)[:, None]
        else:
            to_light = np.broadcast_to(-direction, position.shape)
            light_distance = np.full(n, np.inf)
        scale = np.ones(n)
        if shadow_context is not None and light.shadows:
            scale = shadow_context.visibility(position, normal, light, light_position, direction)
        if particle_occluders and light.shadows:
            particle_scale = _particle_shadow_scale(position, to_light, light_distance,
                                                     particle_occluders, SHADOW_BIAS_DEFAULT)
            if particle_scale is not None:
                scale = scale * particle_scale
        attenuation = _light_factor(light, position)
        if attenuation is not None:
            scale = scale * attenuation
        response, _ = _cook_torrance(normal, to_eye, to_light, roughness_arr, f0)
        nl = np.maximum(np.einsum("ij,ij->i", normal, to_light), 0)
        half = to_light + to_eye
        half = half / np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)
        vh = np.maximum(np.einsum("ij,ij->i", to_eye, half), 0)
        kd = (1 - metallic) * (1 - (0.04 + 0.96 * (1 - vh) ** 5))
        colour = np.asarray(light.color, np.float32) * light.intensity
        diffuse_radiance += (nl * kd * scale)[:, None] * colour
        if specular is not None:
            specular += response * scale[:, None] * colour
    env_diffuse, env_spec = _mesh_pbr_environment(environments, normal, toward_eye, base_rgb,
                                                  metallic, roughness_arr, f0_dielectric)
    diffuse_radiance += env_diffuse
    if specular is not None:
        specular += env_spec
    return diffuse_radiance, specular


def _shade_fragments(position, normal, uv, *, geometry, rgba, mips, level,
                     eye, lights, ambient, output, shade, scene,
                     projection_depth_maps, shadow_context, cancel):
    """Shared surface shader; inputs are world attributes and triangle mip information."""
    projection = geometry.projection
    environments = getattr(scene, 'environments', ())
    lit = (bool(lights) or bool(environments)) and not shade
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
        radiance = np.full((len(position), 3), float(ambient) if lights or environments else 1., np.float32)
        for environment in environments:
            radiance += environment.diffuse(normal).astype(np.float32)
        specular_total = np.zeros_like(radiance)
        to_eye = toward_eye / np.maximum(np.linalg.norm(toward_eye, axis=1, keepdims=True), 1e-8)
        for i, (light, light_position, direction) in enumerate(lights):
            if light.kind in _AREA:
                # R2: not lit yet here (see splatshade.instance_passes' matching note); a present,
                # zeroed pair of channels keeps every other light's index unchanged.
                channels[f"diffuse_L{i}"] = channel(np.zeros((len(position), 1), np.float32))
                channels[f"specular_L{i}"] = channel(np.zeros((len(position), 1), np.float32))
                continue
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
        if environments and geometry.specular:
            specular_total += _mesh_environment_specular(environments, normal, toward_eye, geometry)
        channels["diffuse"] = channel(source[:, :3] * radiance)
        channels["specular"] = channel(specular_total * alpha)
        channels["emission"] = channel(source[:, :3] * geometry.emission)
        return channels, normal, uv
    albedo = source[:, :3].copy() if output == "albedo" else None
    emissive = source[:, :3] * geometry.emission if geometry.emission and output in ("rgba", "emission") else None
    specular = None
    if shade:
        source[:, :3] *= (0.25 + 0.75 * np.abs(normal @ _VIEW_LIGHT))[:, None]
    elif lit and geometry.material == "pbr":
        # materials 1, R1: Cook-Torrance GGX, the splat BRDF shared through `_shade_pbr_mesh`.
        base_rgb = source[:, :3] / np.maximum(source[:, 3:4], 1e-6)
        need_specular = output in ("rgba", "specular")
        diffuse_radiance, specular = _shade_pbr_mesh(
            position, normal, toward_eye, base_rgb, lights, ambient, environments,
            geometry.metallic, geometry.pbr_roughness, 0.08 * float(np.clip(geometry.pbr_specular, 0, 1)),
            shadow_context, need_specular, particle_occluders=getattr(scene, "particles", ()))
        source[:, :3] = base_rgb * diffuse_radiance * source[:, 3:4]
        if specular is not None:
            source[:, :3] += specular * source[:, 3:4]
    elif lit:
        radiance = np.full((len(position), 3), float(ambient), np.float32)
        for environment in environments:
            radiance += environment.diffuse(normal).astype(np.float32)
        specular = np.zeros_like(radiance) if geometry.specular and output in ("rgba", "specular") else None
        if specular is not None:
            to_eye = toward_eye / np.maximum(np.linalg.norm(toward_eye, axis=1, keepdims=True), 1e-8)
        for light, light_position, direction in lights:
            if light.kind in _AREA:
                irradiance, to_light, spec_colour = _area_light_shading(position, normal, light, shadow_context)
                radiance += irradiance
                if specular is not None:
                    half = to_light + to_eye
                    half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)
                    lobe = np.maximum(np.einsum("ij,ij->i", normal, half), 0) ** geometry.shininess
                    specular += geometry.specular * lobe[:, None] * spec_colour
                continue
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
        if specular is not None and environments:
            specular += _mesh_environment_specular(environments, normal, toward_eye, geometry)
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
                    focal, aspect, lights, ambient, output, shade, shadow_context, cancel, mesh_layers=None, rows=None,
                    liquid=None, lens=None):
    """Chunked primary visibility; shading is batched by geometry and mip level.

    `lens` is (ax, ay, focus): the rays start that far right and up of the eye and aim at the pixel's point on the
    plane `focus` in front of the camera (one lens sample of the ray-traced depth of field)."""
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
        if lens is not None:
            local_dirs[:, 0] -= lens[0] / lens[2]
            local_dirs[:, 1] -= lens[1] / lens[2]
        dirs = local_dirs @ inverse_view.T
        eye_ray = eye if lens is None else eye + np.array((lens[0], lens[1], 0.0)) @ inverse_view.T
        origins = np.broadcast_to(eye_ray, dirs.shape)
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
                        eye=eye_ray, lights=lights, ambient=ambient, output=output, shade=shade,
                        scene=scene, projection_depth_maps=projection_depth_maps,
                        shadow_context=shadow_context, cancel=cancel)
                    if liquid is not None and liquid.table["liquid"][object_id]:
                        view_rays = dirs[r] / np.linalg.norm(dirs[r], axis=1, keepdims=True)
                        source = liquid.shade_primary(position, attr[take, 6:9], view_rays,
                                                      np.full(len(r), object_id)).astype(np.float32)
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


def _render_depth_of_field(scene, camera, width, height, background, shade, return_depth, ambient, samples, output,
                           cancel, shadows, progress, volume):
    """The ray-traced mode's thin lens: the mean of one full render per lens sample, each a pinhole render whose
    primary rays start at a Hammersley point of the aperture and aim at the same focal-plane point of their pixel
    (`_render_primary(lens=...)`). Meshes, instances and liquids blur; splats, smoke and particles are composited
    from the pinhole view and stay sharp here (the path tracer blurs everything it traces). The depth returned with
    `return_depth` is the pinhole one."""
    pinhole = replace(camera, fstop=0.0)
    count = 16 * samples
    focus = _lens_module.focus_plane(camera)
    total = None
    for ax, ay in zip(*_lens_module.hammersley_offsets(camera, count)):
        _shadow_cancel(cancel)
        image = render(scene, pinhole, width, height, background, shade, False, ambient, samples, output, cancel,
                       shadows=shadows, mode="raytrace", progress=None, volume=volume, _lens=(float(ax), float(ay), focus))
        total = image.astype(np.float64) if total is None else total + image
    result = (total / count).astype(np.float32)
    result.flags.writeable = False
    if not return_depth:
        return result
    _, depth = render(scene, pinhole, width, height, background, shade, True, ambient, samples, output, cancel,
                      shadows=shadows, mode="raytrace", progress=progress, volume=volume)
    return result, depth


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
           shade=False, return_depth=False, ambient=0.0, samples=1, output="rgba", cancel=None, *, shadows=True, mode="raster", progress=None, volume=None, path=None, _lens=None):
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
    Volumes (`scene.volumes`) are raymarched by nodebased.volumerender with the `volume` settings
    (a VolumeSettings, defaults when None), composited over the mesh image and cut at the mesh depth
    buffer; the `depth` output merges their first-hit depth and the `volume_density`, `volume_motion`,
    `volume_temperature` and `volume_vorticity` outputs are their control passes (never antialiased).
    ``mode`` "pathtrace" hands the scene to the path tracer (nodebased/pathtrace.py, CPU reference) with
    the `PathSettings` in ``path``; ``samples`` is then unused (the settings carry samples per pixel).
    CPU splat progress(stage, fraction, info) spans all accumulation bands.
    Stages are prepare/splats/done; info includes tile_work and estimate_seconds
    once prepared, plus eta_seconds on updates. A callback disables the splat
    tile-work budget; callback exceptions abort rendering.
    """
    if mode == "pathtrace":
        from . import pathtrace
        return pathtrace.render_scene3d(scene, camera, width, height, background, ambient, output, cancel,
                                        progress, return_depth, path, volume)
    if output == "denoise":
        raise ValueError("the denoise output needs Render3D's path tracer mode (render_mode pathtrace)")
    scene = resolve_instances(scene)
    if output == "relight":
        if mode != "raster":
            raise ValueError("the relight bundle output is raster-only for now")
        if return_depth:
            raise ValueError("the relight bundle output does not support return_depth=True")
        if scene.splats:
            if scene.geometries or scene.particles:
                raise ValueError("the relight bundle output does not support scenes with splats that also "
                                 "hold geometry or particles yet")
            return _render_splat_bundle(scene, camera, int(width), int(height), ambient, cancel, progress)
    _shadow_cancel(cancel)
    if mode not in ("raster", "raytrace"):
        raise ValueError(f"Unknown 3D render mode {mode!r}")
    if output not in RENDER_OUTPUTS and output != "splat_normals":
        raise ValueError(f"Unknown 3D render output {output!r}")
    if output == "normals_blend":
        return _render_normals_blend(scene, camera, width, height, return_depth=return_depth,
                                     cancel=cancel, mode=mode, progress=progress)
    if output in VOLUME_OUTPUTS:
        if return_depth:
            raise ValueError("volume control passes do not support return_depth=True")
        return _render_volume_pass(scene, camera, width, height, output, volume, cancel, mode)
    width, height = int(width), int(height)
    data_output = output in DATA_OUTPUTS
    samples = max(1, min(int(samples), 4)) if not data_output and output != "relight" else 1
    if (mode == "raytrace" and _lens is None and _lens_module.active(camera) and not data_output
            and output in ("rgba", "diffuse", "specular", "emission", "splats")):
        return _render_depth_of_field(scene, camera, width, height, background, shade, return_depth, ambient,
                                      samples, output, cancel, shadows, progress, volume)
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
    volume_casters = None
    if scene.volumes and shadows and shadow_count > 0 and output in ("rgba", "splats"):
        # The smoke shadows meshes, relit splats and the shadow catcher (its optical depth toward each shadowed light).
        from . import volumerender
        volume_casters = volumerender.ShadowCasters(
            scene.volumes, volume if volume is not None else volumerender.VolumeSettings()) or None
    splat_catch_active = bool(shadows and shadow_count > 0 and (triangle_count or volume_casters) and catching
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
    layered = bool(scene.splats) and output in ("rgba", *_SPLAT_LAYERS) and not _opaque_meshes(scene)
    splat_visibility = bool(scene.splats) and (data_output or output in ("rgba", "splats"))
    # Mesh reflections on de-lit splats: a closest-hit ray per splat and sample against the meshes.
    from .splatindirect import effective_samples
    reflect_active = bool(triangle_count) and output in ("rgba", "splats") and any(
        getattr(i, 'relight', 0) > 0 and effective_samples(i, 'reflection_samples') > 0 for i in scene.splats)
    if reflect_active:
        work = _shadow_cost(sum(len(i.cloud) * effective_samples(i, 'reflection_samples') for i in scene.splats
                                if getattr(i, 'relight', 0) > 0 and effective_samples(i, 'reflection_samples') > 0),
                            triangle_count)
        if work > SPLAT_SHADOW_BUDGET:
            raise ValueError(f'Splat reflection rays exceed the CPU reference budget: {work:,.0f} estimated tests > {SPLAT_SHADOW_BUDGET:,}')
    # Ambient occlusion and one diffuse bounce (splatindirect): hemisphere rays per splat through the splat
    # casters and the meshes.
    indirect_active = output in ("rgba", "splats") and _indirect_instances(scene) > 0
    if indirect_active:
        _indirect_budget(scene, triangle_count)
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
                     progress=progress, volume=volume, _lens=_lens)
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
    shadow_triangles, shadow_alphas, triangle_colors = [], [], []
    shadow_work = _shadow_cost(0, triangle_count) if shadow_active else 0
    for object_id, geometry in enumerate(scene.geometries, 1):
        _shadow_cancel(cancel)
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        if (shadow_active or splat_shadow_active or splat_catch_active or ray_mode or reflect_active
                or indirect_active) and len(geometry.triangles):
            triangle_colors.append(np.tile(np.asarray(geometry.color[:3], np.float64), (len(geometry.triangles), 1)))
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
            culled = (zs <= camera.near).all() or (zs >= camera.far).all()
            if culled and not ray_mode:
                continue
            if geometry.normals is not None:
                tri_normals = normals[tri]
            else:
                face = np.cross(world[tri[1]] - world[tri[0]], world[tri[2]] - world[tri[0]])
                tri_normals = np.broadcast_to(face / max(float(np.linalg.norm(face)), 1e-8), (3, 3))
            attributes = np.concatenate((local[tri], world[tri], tri_normals, uvs[tri]), axis=1)
            if ray_mode:
                # Culled triangles keep their attributes: no primary ray hits them, but reflected and refracted
                # rays (liquids) can reach surfaces behind the camera.
                ray_attributes[primitive_index] = attributes
                ray_object_ids[primitive_index] = object_id
            if culled:
                continue
            clipped_triangles = _clip_near(attributes.astype(np.float32), zs, camera.near)
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
               if triangle_count and (ray_mode or splat_shadow_active or splat_catch_active or reflect_active
                                      or indirect_active or triangle_count > _SHADOW_BRUTE_THRESHOLD) else None)
        if ray_mode and bvh is None:
            empty = np.empty(0, np.int32)
            bvh = Bvh(np.empty((0, 3)), np.empty((0, 3)), empty, empty, empty, empty, empty)
        bias = 1e-3 * max(1.0, float(np.ptp(triangles.reshape(-1, 3), axis=0).max())) if len(triangles) else .001
        if shadow_active:
            work = _shadow_cost(width*height, triangle_count) if ray_mode else shadow_work
            shadow_context = _ShadowContext(primitives, bvh, bias, cancel, triangle_count, work, ray_mode)
    splat_shadows = None
    if splat_cast_active or splat_catch_active or indirect_active:
        splat_shadows = _SplatShadows(scene.splats, scene.lights,
            primitives if triangle_count else None, bvh if triangle_count else None,
            bias if triangle_count else .001, cancel)
        splat_shadows.relit_shadows = splat_shadow_active
        if shadow_context is not None and splat_cast_active and casting:
            shadow_context.splat_shadows = splat_shadows
    if volume_casters is not None:
        if shadow_context is not None and output == "rgba":
            shadow_context.volume_shadows = volume_casters
        if splat_shadows is not None:
            splat_shadows.volume_shadows = volume_casters
    if ray_mode:
        primary_kwargs = dict(attributes=ray_attributes, object_ids=ray_object_ids,
                        mip_levels=ray_mip_levels, clipped_mips=clipped_mips, materials=materials,
                        primitives=primitives, bvh=bvh, eye=eye, view=view, focal=focal, aspect=aspect,
                        lights=lights, ambient=ambient, output=output, shade=shade,
                        shadow_context=shadow_context, cancel=cancel, lens=_lens)
        if output == "rgba" and not shade and any(g.material == "liquid" for g in scene.geometries):
            from .liquid_render import LiquidTracer
            primary_kwargs["liquid"] = LiquidTracer(
                primitives=primitives, bvh=bvh, object_ids=ray_object_ids, attributes=ray_attributes,
                mip_levels=ray_mip_levels, materials=materials, eye=eye, lights=lights, ambient=ambient,
                scene=scene, background=np.asarray(background, np.float64), cancel=cancel)
        if not (layered or (splat_visibility and data_output)):
            _render_primary(scene, camera, width, height, out, depth, **primary_kwargs)
    # The raster mode's liquid stand-in (liquid_render.raster_liquid); the viewport's inspection headlight glints too.
    raster_liquid = output == "rgba" and any(g.material == "liquid" for g in scene.geometries)
    liquid_closed, behind_frame = {}, None
    raster_lights = lights if not shade else [(Light(), None, -_VIEW_LIGHT)]
    # Liquid triangles come last, far to near: the picture they refract is everything else, drawn first.
    liquid_ids = {i for i, g in enumerate(scene.geometries, 1) if g.material == "liquid"} if raster_liquid else set()
    ordered = sorted(queue, key=lambda item: (item[7] not in liquid_ids, item[0]), reverse=True)
    for index, (_mean_z, tri, z, rgba, mips, projection, geometry, object_id) in enumerate(ordered):
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
        wet = raster_liquid and geometry.material == "liquid"
        if wet:
            from .liquid_render import is_closed
            if object_id not in liquid_closed:
                liquid_closed[object_id] = is_closed(geometry)
            if liquid_closed[object_id]:
                # a closed liquid draws its front faces only: the far side would be drawn into the picture the
                # refraction samples
                facing = np.einsum("ij,ij->i", normal, eye - position) > 0
                if not facing.all():
                    kept = take.copy()
                    kept[take] = facing
                    take = kept
                    if not take.any():
                        continue
                    weights, position, normal, uv = weights[facing], position[facing], normal[facing], uv[facing]
        normal_before = normal
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
        elif wet:
            from .liquid_render import raster_liquid as liquid_fragments
            if behind_frame is None:
                behind_frame = out.copy()
            source = liquid_fragments(
                position=position, normal=normal_before, geometry=geometry, eye=eye, view=view,
                lights=raster_lights, environments=getattr(scene, "environments", ()), background=background,
                out=behind_frame, xs=xs[take], ys=ys[take])
            src_alpha = source[:, 3]
            region[take] = source
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
    if scene.splats and (output in ("rgba", *_SPLAT_LAYERS) or data_output):
        from .splatraster import prepare_splats, accumulate_splats, estimate_seconds, estimate_eta_seconds
        if progress is not None:
            progress("prepare", 0.0, {})
        splat_lighting = (scene.lights, ambient)
        splat_extras = None
        if scene.environments or reflect_active or indirect_active:
            from .envlight import SplatLighting
            from .splatindirect import IndirectLight
            reflector = (_MeshReflector(primitives, bvh, np.concatenate(triangle_colors), lights, ambient,
                                        scene.environments, bias, cancel) if reflect_active or (
                                        indirect_active and triangle_count) else None)
            splat_extras = SplatLighting(
                scene.environments, reflector if reflect_active else None,
                IndirectLight(splat_shadows, ambient, scene.environments, reflector, cancel=cancel)
                if indirect_active else None)
        if splat_shadow_active or splat_catch_active or indirect_active:
            splat_lighting = (scene.lights, ambient, splat_shadows)
        if splat_extras is not None:
            splat_lighting = (*splat_lighting[:2], splat_lighting[2] if len(splat_lighting) > 2 else None,
                              splat_extras)
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
        if not layered and not data_output and output not in _SPLAT_LAYERS:
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
            elif layered or data_output or output in _SPLAT_LAYERS:
                band_out[..., :3], band_out[..., 3] = splat_rgb, splat_alpha
            else:
                band_out[..., :3] = splat_rgb + (1-splat_alpha[..., None])*band_out[..., :3]
                band_out[..., 3] = splat_alpha + (1-splat_alpha)*band_out[..., 3]
            # Release layers before allocating the next band.
            mesh_layers = None
        if progress is not None:
            progress("done", 1.0, dict(info, eta_seconds=0.0))
    if scene.particles and output == "rgba":
        _draw_particles(scene, camera, width, height, out, depth, eye, view, focal, aspect, cancel,
                        lights=lights, ambient=ambient, environments=scene.environments,
                        shadow_context=shadow_context)
    if scene.volumes and output in ("rgba", "depth"):
        from . import volumerender
        settings = volume if volume is not None else volumerender.VolumeSettings()
        if output == "rgba":
            volumerender.composite_beauty(scene, camera, width, height, out, depth, settings, ambient, cancel,
                                          _volume_occluders(scene, cancel) if shadows and shadow_count > 0 else None)
        else:
            first = volumerender.first_hit_depth(scene, camera, width, height, depth, settings, cancel)
            hit = first < depth
            out[hit, :3] = first[hit, None]
            out[hit, 3] = 1
            depth[hit] = first[hit]
    if output in _SPLAT_LAYERS and not scene.splats:
        out[:] = 0
    out.flags.writeable = False
    if return_depth:
        depth.flags.writeable = False
        return out, depth
    return out


def _render_splat_bundle(scene, camera, width, height, ambient, cancel, progress):
    """The relight bundle of a splat-only scene: `(beauty_rgba, {layer: rgba})`.

    Layers: `albedo` (the de-lit layer when a splat instance has one and `use_intrinsics` is on, else the
    captured colour), `normals` (the splat-aware blend, eye-facing), `position`, `roughness`, `occlusion`
    (1 open), `diffuse_L{i}` / `specular_L{i}` per positive-intensity light (unitless responses, as the
    Relight node expects; a shadowed light's response carries the traced visibility), the environment
    layers `environment_diffuse`, `environment_specular`, `reflections` and the direct-light `visibility`
    (see `splatshade.instance_passes`), and the `diffuse`, `specular`, `emission` sums the scene's own
    lights, environments and `ambient` give. Every layer is premultiplied by splat coverage, alpha is
    coverage, and every one is a per-splat value drawn through the same accumulation as the beauty, so
    they blend exactly as colour does. Visibility is traced through the splat BVH (splat-on-splat
    included) for lights with Shadows on; mesh reflections need geometry, which a splat-only bundle has none of.
    """
    from .splatshade import instance_passes
    if progress is not None:
        progress("prepare", 0.0, {})
    lights = [light for light in scene.lights if light.intensity > 0]
    eye, _ = _view_basis(camera)
    shadowed = [light for light in lights if light.shadows]
    visibility = [None] * len(scene.splats)
    if shadowed:
        rays = sum(len(i.cloud) for i in scene.splats) * len(shadowed)
        work = _shadow_cost(rays, 0) + _shadow_cost(rays, sum(len(i.cloud) for i in scene.splats
                                                              if getattr(i, 'cast_shadows', True)))
        if work > SPLAT_SHADOW_BUDGET:
            raise ValueError(f'Splat shadow rays exceed the CPU reference budget: {work:,.0f} estimated tests > {SPLAT_SHADOW_BUDGET:,}')
        forced = tuple(replace(i, relight=max(float(getattr(i, 'relight', 0)), 1.0)) for i in scene.splats)
        context = _SplatShadows(forced, tuple(lights), None, None, .001, cancel)
        visibility = [context.for_indices(index, np.arange(len(instance.cloud)))
                      for index, instance in enumerate(scene.splats)]
    from .envlight import SplatLighting
    extras = SplatLighting(scene.environments) if scene.environments else None
    if _indirect_instances(scene) > 0:
        _indirect_budget(scene, 0)
        from .splatindirect import IndirectLight
        forced = tuple(replace(i, relight=max(float(getattr(i, 'relight', 0)), 1.0)) for i in scene.splats)
        context = _SplatShadows(forced, tuple(lights), None, None, .001, cancel)
        context.relit_shadows = bool(shadowed)
        extras = SplatLighting(scene.environments, None,
                               IndirectLight(context, ambient, scene.environments, cancel=cancel))
    per_instance = [instance_passes(instance, eye, lights, ambient, seen, extras)
                    for instance, seen in zip(scene.splats, visibility)]

    def layer(name):
        instances = []
        for instance, (passes, _cloud) in zip(scene.splats, per_instance):
            source = instance.cloud
            values = np.asarray(passes[name], dtype=np.float32)
            plain = SplatCloud(source.positions, source.scales, source.rotations, source.opacity,
                               ((values - 0.5) / C0)[:, None, :], 0, colorspace='linear')
            instances.append(replace(instance, cloud=plain, sh_degree=None, relight=0.0, shadow_catch=0.0,
                                     specular=0.0))
        return render(replace(scene, splats=tuple(instances), lights=()), camera, width, height,
                      output="splats", cancel=cancel)

    names = ("albedo", "roughness", "occlusion", "diffuse", "specular", "environment_diffuse",
             "environment_specular", "reflections", "indirect", "visibility")
    names += tuple(f"{kind}_L{i}" for i in range(len(lights)) for kind in ("diffuse", "specular"))
    channels = {}
    for number, name in enumerate(names):
        _shadow_cancel(cancel)
        channels[name] = np.array(layer(name))
        if progress is not None:
            progress("splats", (number + 1) / (len(names) + 2), {})
    channels["normals"] = np.array(_render_normals_blend(scene, camera, width, height, return_depth=False,
                                                         cancel=cancel, mode="raster", progress=None))
    channels["position"] = np.array(render(scene, camera, width, height, output="position", cancel=cancel))
    channels["emission"] = np.zeros((height, width, 4), np.float32)
    coverage = channels["albedo"][..., 3]
    for name in ("diffuse", "specular", "emission"):
        channels[name][..., 3] = coverage
    out = channels["diffuse"].copy()
    out[..., :3] += channels["specular"][..., :3]
    out.flags.writeable = False
    for channel in channels.values():
        channel.flags.writeable = False
    if progress is not None:
        progress("done", 1.0, {})
    return out, channels


def _volume_occluders(scene, cancel=None):
    """`occluders(points, light)` for the volume march: the transmittance of the scene's meshes (material alpha) and
    casting splats from world `points` toward `light`, hard shadows, so a card between the light and a plume shadows
    the plume. None when nothing can cast (no shadowed light, or no meshes and no casting splats)."""
    if not any(light.shadows and light.intensity > 0 for light in scene.lights):
        return None
    corners, alphas = [], []
    for geometry in scene.geometries:
        if not len(geometry.triangles):
            continue
        matrix = geometry.world_matrix()
        world = (matrix[:3, :3] @ geometry.vertices.T + matrix[:3, 3:4]).T
        corners.append(world[geometry.triangles])
        alphas.append(np.full(len(geometry.triangles), np.clip(geometry.color[3], 0, 1), np.float32))
    casting = [i for i in scene.splats if getattr(i, "cast_shadows", True)]
    if not corners and not casting:
        return None
    triangles = np.concatenate(corners) if corners else np.empty((0, 3, 3), np.float32)
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    primitives = TriangleSet(v0, e1, e2, np.concatenate(alphas) if alphas else np.empty(0, np.float32))
    bvh = Bvh.build(*primitives.aabbs(), cancel=cancel) if len(triangles) > _SHADOW_BRUTE_THRESHOLD else None
    bias = 1e-3 * max(1.0, float(np.ptp(triangles.reshape(-1, 3), axis=0).max())) if len(triangles) else .001
    splat_shadows = _SplatShadows(scene.splats, scene.lights, None, None, bias, cancel) if casting else None

    def occluders(points, light):
        position, direction = light.world()
        points = np.asarray(points, np.float64)
        return _shadow_visibility(points, np.zeros_like(points), replace(light, shadow_blur=0.0), position, direction,
                                  v0, e1, e2, primitives.alpha, bias, cancel, triangles=primitives, bvh=bvh,
                                  splat_shadows=splat_shadows)
    return occluders


def _render_volume_pass(scene, camera, width, height, output, volume, cancel, mode):
    """A volume control pass: the raymarch integral cut at the opaque mesh depth (nodebased.volumerender)."""
    from . import volumerender
    width, height = int(width), int(height)
    if not scene.volumes:
        return np.zeros((height, width, 4), np.float32)
    meshes = Scene(scene.geometries)
    _, mesh_depth = render(meshes, camera, width, height, output="depth", return_depth=True,
                           cancel=cancel, mode=mode)
    out = volumerender.render_pass(scene, camera, width, height, mesh_depth,
                                   volume if volume is not None else volumerender.VolumeSettings(),
                                   output, cancel)
    out.flags.writeable = False
    return out


def _render_normals_blend(scene, camera, width, height, *, return_depth, cancel, mode, progress):
    """World-space normals with splats blended the way colour is: RGB unit normal, alpha coverage.

    Meshes give their first-hit normal, as the `normals` output. Each splat contributes its estimated
    normal, flipped toward the eye and smoothed by the instance's `normal_smoothing`, composited front
    to back with the alpha weights the beauty pass uses, over whatever meshes lie behind; the blend
    is divided by coverage and renormalised, so a pixel covered by two splats facing different ways
    gets the direction between them (`normals` reports the first one only). Without splats this is
    exactly `normals`. Not antialiased.
    """
    if return_depth:
        raise ValueError("the normals_blend output does not support return_depth=True")
    if not scene.splats:
        return render(scene, camera, width, height, output="normals", cancel=cancel, mode=mode, progress=progress)
    meshes = render(replace(scene, splats=()), camera, width, height, output="normals", cancel=cancel, mode=mode)
    layer = render(scene, camera, width, height, output="splat_normals", cancel=cancel, mode=mode,
                   progress=progress)
    a = layer[..., 3:4]
    total = layer[..., :3] + (1 - a) * meshes[..., :3] * meshes[..., 3:4]
    coverage = a + (1 - a) * meshes[..., 3:4]
    length = np.linalg.norm(total, axis=-1, keepdims=True)
    out = np.zeros((height, width, 4), np.float32)
    good = length[..., 0] > 1e-9
    out[good, :3] = total[good] / length[good]
    out[..., 3] = np.where(good, coverage[..., 0], 0)
    out.flags.writeable = False
    return out


MULTICHANNEL_PASSES = ("beauty", "normals", "depth", "relight", "albedo", "denoise", "motion") + VOLUME_OUTPUTS
DEFAULT_PASSES = "beauty,normals,depth"


def parse_passes(text):
    """The `passes` knob (a comma-separated list, order and repeats ignored) as a tuple in
    `MULTICHANNEL_PASSES` order. An unknown name is an error that lists the valid ones."""
    wanted = {part.strip().lower() for part in str(text).split(",") if part.strip()}
    unknown = sorted(wanted - set(MULTICHANNEL_PASSES))
    if unknown:
        raise ValueError(f"Unknown Render3D pass {unknown[0]!r}; choose from {', '.join(MULTICHANNEL_PASSES)}")
    return tuple(name for name in MULTICHANNEL_PASSES if name in wanted)


def _volume_layer(scene, camera, width, height, name, volume, cancel, mode, backend):
    """One volume control pass for the multichannel output, on the GPU when `backend` allows it."""
    if backend != "cpu":
        from . import gpu3d
        from .cancellation import Cancelled
        if gpu3d.available():
            try:
                return gpu3d.render(scene, camera, width, height, output=name, volume=volume, cancel=cancel, mode=mode)
            except Cancelled:
                raise
            except Exception as exc:     # unsupported scene, device loss, out of memory
                if backend == "gpu":
                    raise ValueError(f"GPU Render3D {'unsupported' if isinstance(exc, gpu3d.Unsupported) else 'failed'}: {exc}") from exc
        elif backend == "gpu":
            raise ValueError(f"GPU Render3D unavailable: {gpu3d.describe()}")
    return render(scene, camera, width, height, output=name, volume=volume, cancel=cancel, mode=mode)


def render_multichannel(scene, camera, width, height, background=(0., 0., 0., 0.), *, passes=DEFAULT_PASSES,
                        ambient=0.0, samples=1, cancel=None, mode="raster", progress=None, volume=None, backend="cpu",
                        path=None, motion_layer=None):
    """One frame with several passes as named layers: returns `(beauty_rgba, {layer: rgba})`.

    Layer names follow Nuke's `layer.channel` scheme once written to EXR (nodebased.media):
    `normals` (the splat-aware normals_blend, exactly `normals` without splats), `depth`, and per
    enabled light `relight_light1_diffuse` and `relight_light1_specular`, ... in `Scene3D` wiring
    order (the bundle's unitless response terms, see the relight output). `beauty` is the returned
    rgba; without it that array is transparent black. Every pass is the single-purpose output of
    the same name, so its pixels equal a `Render3D` set to that Output. The relight layers keep the
    relight bundle's limits (raster mode; splat scenes only without geometry or particles). The four
    volume control passes (`volume_density`, `volume_motion`, `volume_temperature`, `volume_vorticity`)
    are layers of the same names (plus `volume_id`, the number of the nearest Volume member with smoke on the ray,
    1 for the first member, 0 for none), raymarched with the `volume` settings (nodebased.volumerender). With
    `backend` "gpu" or "auto" they are raymarched on the GPU (nodebased.gpuvolume; "auto" falls back to the CPU
    reference, "gpu" reports why not); the other layers are always the CPU reference.
    """
    chosen = parse_passes(passes)
    if not chosen:
        raise ValueError("Render3D multichannel needs at least one pass")
    if "motion" in chosen:
        if motion_layer is None:
            raise ValueError("the motion pass needs Render3D's scene and camera evaluated a frame later")
        beauty, layers = render_multichannel(scene, camera, width, height, background,
                                             passes=",".join(name for name in chosen if name != "motion") or "beauty",
                                             ambient=ambient, samples=samples, cancel=cancel, mode=mode,
                                             progress=progress, volume=volume, backend=backend, path=path)
        if "beauty" not in chosen:
            beauty = np.zeros_like(beauty)
        return beauty, {**layers, "motion": motion_layer}
    if mode == "pathtrace":
        # the path tracer's own passes (`path` carries its settings, backend its device): the beauty, its guides
        # (normals, depth and albedo, the passes an external denoiser reads) and `denoise`, the beauty filtered
        # with them by nodebased/ptdenoise.py
        from . import pathtrace
        allowed = ("beauty", "normals", "depth", "albedo", "denoise")
        unsupported = [name for name in chosen if name not in allowed]
        if unsupported:
            raise ValueError(f"the path tracer's multichannel output has beauty, normals, depth, albedo and denoise; "
                             f"not {', '.join(unsupported)}")
        st = {}
        need = "beauty" in chosen or "denoise" in chosen
        raw = (pathtrace.render(scene, camera, width, height, (0, 0, 0, 0), ambient, "rgba", path, cancel=cancel,
                                progress=progress, stats=st, backend=backend, volume=volume)
               if need else np.zeros((int(height), int(width), 4), np.float32))
        beauty = pathtrace.over_background(raw.astype(np.float64), background).astype(np.float32) \
            if "beauty" in chosen else np.zeros((int(height), int(width), 4), np.float32)
        wanted = {"albedo", "normals", "depth"} & set(chosen)
        guides = (pathtrace.guide_aovs(scene, camera, width, height, path, cancel, backend, volume, st)
                  if wanted or "denoise" in chosen else {})
        layers = {name: guides[name] for name in chosen if name in wanted}
        if "denoise" in chosen:
            layers["denoise"] = pathtrace.denoised(raw, guides, st.get("variance"), background)
        return beauty, layers
    beauty = (render(scene, camera, width, height, background, ambient=ambient, samples=samples,
                     cancel=cancel, mode=mode, progress=progress, volume=volume)
              if "beauty" in chosen else np.zeros((int(height), int(width), 4), np.float32))
    if "denoise" in chosen:
        raise ValueError("the denoise pass needs Render3D's path tracer mode (render_mode pathtrace)")
    layers = {}
    if "normals" in chosen:
        layers["normals"] = _render_normals_blend(scene, camera, width, height, return_depth=False,
                                                  cancel=cancel, mode=mode, progress=None)
    if "depth" in chosen:
        layers["depth"] = render(scene, camera, width, height, output="depth", cancel=cancel, mode=mode)
    for name in VOLUME_OUTPUTS:
        if name in chosen:
            layers[name] = _volume_layer(scene, camera, width, height, name, volume, cancel, mode, backend)
    if "relight" in chosen:
        _, bundle = render(scene, camera, width, height, ambient=ambient, output="relight",
                           cancel=cancel, mode=mode)
        lights = sum(1 for name in bundle if name.startswith("diffuse_L"))
        for index in range(lights):
            for kind in ("diffuse", "specular"):
                layers[f"relight_light{index + 1}_{kind}"] = bundle[f"{kind}_L{index}"]
    if "albedo" in chosen:
        layers["albedo"] = render(scene, camera, width, height, output="albedo", cancel=cancel, mode=mode)
    beauty.flags.writeable = False
    return beauty, layers


PARTICLE_MIN_RADIUS = 0.75      # pixels: a particle smaller than this still lights its own pixel
PARTICLE_MAX_RADIUS = 96        # pixels: a nearer particle is clamped rather than filling the frame
_PARTICLE_SHAPES = {"points": 0, "spheres": 1, "cards": 2, "foam": 3}
_PARTICLE_FRAGMENT_CHUNK = 2_000_000


def _parse_value_ramp(text, channels):
    """The stops of a "t:v[,v...];t:v[,v...]" ramp string, sorted by t; None when blank (docs/3D_FOUNDATION.md
    "Particles", following `volumerender.parse_fire_ramp`'s stop grammar). Raises ValueError naming the
    offending stop when a stop does not parse or does not have exactly `channels` values."""
    text = (text or "").strip()
    if not text:
        return None
    stops = []
    for part in text.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            t, rest = part.split(":")
            values = [float(v) for v in rest.replace(" ", "").split(",")]
            if len(values) != channels:
                raise ValueError
            stops.append((float(t), values))
        except ValueError:
            raise ValueError(f"particle ramp stop {part!r} is not 't:{','.join(['v'] * channels)}'") from None
    if not stops:
        return None
    stops.sort(key=lambda stop: stop[0])
    return np.array([t for t, _ in stops], np.float64), np.array([v for _, v in stops], np.float64)


def evaluate_particle_ramp(text, t, channels):
    """(N, channels) linear interpolation of a `_parse_value_ramp` ramp at `t` (N,), clamped to the ramp's
    own ends; None (no override, the incoming attribute passes through unchanged) when `text` is blank."""
    ramp = _parse_value_ramp(text, channels)
    if ramp is None:
        return None
    knots, values = ramp
    t = np.clip(np.asarray(t, np.float64), knots[0], knots[-1])
    return np.stack([np.interp(t, knots, values[:, c]) for c in range(channels)], -1).astype(np.float32)


def particle_ramp_parameter(instance, ramp_by):
    """The per-particle (N,) value the age/speed ramps read: "age" is the solved fraction of life (0 at
    birth, 1 at death; a particle with no lifetime set never reaches 1), "speed" is the world-space
    velocity magnitude in units per frame. "off" (or any other value) returns None: ramps are skipped."""
    n = len(instance.positions)
    if ramp_by == "age":
        if instance.ages is None or instance.lifetimes is None:
            return np.zeros(n, np.float32)
        life = np.where(np.asarray(instance.lifetimes) > 0, instance.lifetimes, np.inf)
        return np.clip(np.asarray(instance.ages, np.float64) / life, 0.0, 1.0).astype(np.float32)
    if ramp_by == "speed":
        if instance.velocities is None:
            return np.zeros(n, np.float32)
        return np.linalg.norm(np.asarray(instance.velocities, np.float64), axis=1).astype(np.float32)
    return None


def apply_particle_look(instance, params):
    """ParticleRender3D's material and attribute ramps (R7 of 7): bakes `particle_color_ramp`,
    `particle_opacity_ramp`, `particle_size_ramp` and `particle_emission_ramp` (by age or speed,
    `particle_ramp_by`) into this frame's own `colors`/`sizes`/`emission` arrays, once, here, and never
    inside the solve (matching `render_as`/`size_scale` above). `particle_ramp_by` "off" (the default)
    and every ramp blank leaves `colors` and `sizes` untouched, so an old document renders exactly as it
    did before this step; ramps compose with `size_scale`, never replace it. `particle_material` "pbr"
    (docs/3D_FOUNDATION.md "Materials") switches lighting on for this instance.
    """
    ramp_by = params.get("particle_ramp_by", "off")
    colors, sizes, emission = instance.colors, instance.sizes, None
    t = particle_ramp_parameter(instance, ramp_by) if ramp_by in ("age", "speed") else None
    if t is not None and len(t):
        color_ramp = evaluate_particle_ramp(params.get("particle_color_ramp", ""), t, 3)
        if color_ramp is not None:
            alpha = colors[:, 3:4]
            colors = np.concatenate((color_ramp * alpha, alpha), axis=1).astype(np.float32)
        opacity_ramp = evaluate_particle_ramp(params.get("particle_opacity_ramp", ""), t, 1)
        if opacity_ramp is not None:
            alpha = np.clip(opacity_ramp[:, 0], 0.0, 1.0)
            rgb = colors[:, :3] / np.maximum(colors[:, 3:4], 1e-6)
            colors = np.concatenate((rgb * alpha[:, None], alpha[:, None]), axis=1).astype(np.float32)
        size_ramp = evaluate_particle_ramp(params.get("particle_size_ramp", ""), t, 1)
        if size_ramp is not None:
            sizes = (np.asarray(sizes, np.float32) * np.maximum(size_ramp[:, 0], 0.0)).astype(np.float32)
        emission_ramp = evaluate_particle_ramp(params.get("particle_emission_ramp", ""), t, 1)
        if emission_ramp is not None:
            emission = np.maximum(emission_ramp[:, 0], 0.0).astype(np.float32)
    if emission is None and float(params.get("particle_emission", 0.0)) > 0:
        emission = np.full(len(instance.positions), float(params.get("particle_emission", 0.0)), np.float32)
    return replace(instance, colors=colors, sizes=sizes, emission=emission,
                  material=str(params.get("particle_material", "standard")),
                  metallic=float(params.get("particle_metallic", 0.0)),
                  pbr_roughness=float(params.get("particle_pbr_roughness", 0.5)),
                  pbr_specular=float(params.get("particle_pbr_specular", 0.5)),
                  cast_shadows=bool(params.get("particle_cast_shadows", 1)))


def foam_subset(instance):
    """Boolean mask of the foam particles kept at `foam_density`: a fixed subset chosen by particle id (by position
    in the set without ids), so the same particles stay while the knob moves and a frame is deterministic."""
    ids = instance.ids if instance.ids is not None and len(instance.ids) == len(instance.positions) \
        else np.arange(len(instance.positions))
    fraction = ((np.asarray(ids).astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) / 4294967296.0
    return fraction < float(instance.foam_density)


def particle_sprites(scene, camera, width, height, eye, view, focal, aspect):
    """Every ParticleInstance as screen sprites sorted far to near, shared by the CPU and GPU draws.

    Returns None when nothing is in front of the camera, else `(z, centre, radius, color, shape,
    world_radius, texture_id, textures, world_center, pbr, metallic, roughness, specular, emission)`:
    view depth, pixel centre, clamped pixel radius, premultiplied colour, shape code
    (`_PARTICLE_SHAPES`), world radius, index into `textures` (-1 for none), world-space centre (for
    the "pbr" lighting reconstruction below), whether this particle's material is "pbr", its
    metallic/roughness/specular and its emission multiplier (0 when off).
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
        foam = instance.render_as == "foam"
        if foam and instance.foam_density < 1.0:
            keep &= foam_subset(instance)
        if not keep.any():
            continue
        z = z[keep]
        centre = _to_pixels(local[keep], z, focal, aspect, width, height)
        sizes = instance.sizes[keep] * np.float32(instance.size_scale * (instance.spray_size if foam else 1.0))
        radius = np.clip(0.25 * sizes * focal * height / z, PARTICLE_MIN_RADIUS, PARTICLE_MAX_RADIUS)
        shape = np.full(len(z), _PARTICLE_SHAPES.get(instance.render_as, 0), np.int8)
        texture = -1
        if instance.render_as == "cards" and instance.texture is not None:
            texture = len(textures)
            textures.append(instance.texture)
        n = len(z)
        pbr = np.full(n, instance.material == "pbr")
        emission = (np.asarray(instance.emission, np.float32)[keep] if instance.emission is not None
                    else np.zeros(n, np.float32))
        order_sets.append((z, centre, radius, instance.colors[keep], shape, 0.5 * sizes,
                           np.full(n, texture, np.int32), world[keep], pbr,
                           np.full(n, float(instance.metallic)), np.full(n, float(instance.pbr_roughness)),
                           np.full(n, float(instance.pbr_specular)), emission))
    if not order_sets:
        return None
    z = np.concatenate([s[0] for s in order_sets])
    centre = np.concatenate([s[1] for s in order_sets])
    radius = np.concatenate([s[2] for s in order_sets])
    color = np.concatenate([s[3] for s in order_sets])
    shape = np.concatenate([s[4] for s in order_sets])
    world_radius = np.concatenate([s[5] for s in order_sets])
    texture_id = np.concatenate([s[6] for s in order_sets])
    world_center = np.concatenate([s[7] for s in order_sets])
    pbr = np.concatenate([s[8] for s in order_sets])
    metallic = np.concatenate([s[9] for s in order_sets])
    roughness = np.concatenate([s[10] for s in order_sets])
    specular = np.concatenate([s[11] for s in order_sets])
    emission = np.concatenate([s[12] for s in order_sets])
    far_first = np.argsort(-z, kind="stable")
    z, centre, radius, color = z[far_first], centre[far_first], radius[far_first], color[far_first]
    shape, world_radius, texture_id = shape[far_first], world_radius[far_first], texture_id[far_first]
    world_center, pbr = world_center[far_first], pbr[far_first]
    metallic, roughness, specular, emission = (metallic[far_first], roughness[far_first],
                                               specular[far_first], emission[far_first])
    return (z, centre, radius, color, shape, world_radius, texture_id, textures,
           world_center, pbr, metallic, roughness, specular, emission)


def _draw_particles(scene, camera, width, height, out, depth, eye, view, focal, aspect, cancel,
                    lights=(), ambient=0.0, environments=(), shadow_context=None):
    """Composite every ParticleInstance over `out` as size-scaled discs (beauty output only).

    The CPU reference for the particle draw: each particle is a flat disc (points), a shaded disc
    (spheres) or a square (cards) of world diameter `size` facing the camera, `colors` premultiplied,
    tested against the mesh depth buffer but never written to it. Particles composite far to near over
    what is already in `out` (meshes and splats), so overlapping translucent particles blend in depth
    order. A pixel belongs to a disc when its centre lies within the projected radius, which is at
    least PARTICLE_MIN_RADIUS pixels. A "pbr" instance (R7 of 7) is shaded by `lights`, `ambient` and
    `environments` through `_shade_pbr_mesh`, and receives `shadow_context`'s mesh shadows, exactly as
    a mesh of the same material would; every other instance keeps the original fixed "headlight" look.
    """
    sprites = particle_sprites(scene, camera, width, height, eye, view, focal, aspect)
    if sprites is None:
        return
    (z, centre, radius, color, shape, world_radius, texture_id, textures,
     world_center, pbr, metallic, roughness, specular, emission) = sprites
    extra = (shape, world_radius, texture_id, textures, world_center, pbr, metallic, roughness, specular, emission)
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
        _composite_particle_chunk(slice(start, stop), z, centre, radius, reach, color, out, depth, width, height,
                                  extra, eye, view, lights, ambient, environments, shadow_context)
        start = stop


def _composite_particle_chunk(rows, z, centre, radius, reach, color, out, depth, width, height, extra,
                              eye, view, lights, ambient, environments, shadow_context):
    shape, world_radius, texture_id, textures, world_center, pbr, metallic, roughness, specular, emission = extra
    pixel_parts, order_parts, u_parts, v_parts = [], [], [], []
    z, centre, radius, reach, color = z[rows], centre[rows], radius[rows], reach[rows], color[rows]
    shape, world_radius, texture_id = shape[rows], world_radius[rows], texture_id[rows]
    world_center, pbr = world_center[rows], pbr[rows]
    metallic, roughness, specular, emission = metallic[rows], roughness[rows], specular[rows], emission[rows]
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
    sphere = (shape[owner] == 1) | (shape[owner] == 3)     # spheres and foam are round: the surface is nearer
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
    sphere = (shape[owner] == 1) | (shape[owner] == 3)
    facing = np.sqrt(np.maximum(0.0, 1.0 - frag_u ** 2 - frag_v ** 2))
    pbr_owner = pbr[owner]
    old_sphere = sphere & ~pbr_owner
    if old_sphere.any():                                    # view-space normal on the unit disc, headlight-ish
        lit = np.maximum(0.0, frag_u * _VIEW_LIGHT[0] - frag_v * _VIEW_LIGHT[1] + facing * _VIEW_LIGHT[2])
        fragment[old_sphere, :3] *= (0.25 + 0.75 * lit[old_sphere])[:, None].astype(np.float32)
    if pbr_owner.any():
        # R7 of 7: a "pbr" particle is lit like a mesh of the same material. Spheres get the curved
        # impostor normal `(u, -v, facing)`; points and cards (and foam, if ever set to "pbr") are flat
        # discs facing the camera, so their normal is the camera's own forward axis. Both are view-space
        # vectors, taken to world by `view`'s inverse (its transpose: `view` is an orthonormal rotation).
        normal_view = np.stack((np.where(sphere, frag_u, 0.0), np.where(sphere, -frag_v, 0.0),
                                np.where(sphere, facing, 1.0)), axis=-1)
        world_normal = normal_view @ view
        world_normal /= np.maximum(np.linalg.norm(world_normal, axis=1, keepdims=True), 1e-8)
        offset = np.where(sphere, world_radius[owner], 0.0)[:, None]
        position = world_center[owner] + world_normal * offset
        toward_eye = eye[None, :] - position
        base_rgb = (fragment[:, :3] / np.maximum(fragment[:, 3:4], 1e-6)).astype(np.float64)
        combo = np.stack((metallic[owner], roughness[owner], specular[owner]), axis=1)
        for m, r, sp in np.unique(combo[pbr_owner], axis=0):
            group = pbr_owner & (combo[:, 0] == m) & (combo[:, 1] == r) & (combo[:, 2] == sp)
            diffuse, spec = _shade_pbr_mesh(position[group], world_normal[group], toward_eye[group],
                                            base_rgb[group], lights, ambient, environments,
                                            m, r, 0.08 * float(np.clip(sp, 0, 1)), shadow_context, True)
            shaded = base_rgb[group] * diffuse
            if spec is not None:
                shaded = shaded + spec
            fragment[group, :3] = (shaded * fragment[group, 3:4]).astype(np.float32)
    foam = shape[owner] == 3
    if foam.any():                                          # a soft rim: premultiplied colour and alpha fade together
        fragment[foam] *= ((1.0 - np.minimum(1.0, frag_u ** 2 + frag_v ** 2)) ** 2)[foam, None].astype(np.float32)
    textured = texture_id[owner] >= 0
    for index in np.unique(texture_id[owner][textured]):
        image = textures[index]
        pick = textured & (texture_id[owner] == index)
        column = np.clip(((frag_u[pick] + 1.0) * 0.5 * image.shape[1]).astype(np.int64), 0, image.shape[1] - 1)
        line = np.clip(((frag_v[pick] + 1.0) * 0.5 * image.shape[0]).astype(np.int64), 0, image.shape[0] - 1)
        fragment[pick] *= image[line, column]
    glow = emission[owner]
    if np.any(glow > 0):                                    # self-emission (R7 of 7): brighten, never darken
        fragment[:, :3] *= (1.0 + glow)[:, None].astype(np.float32)
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
