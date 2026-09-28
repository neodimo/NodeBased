"""Unidirectional path tracer for Render3D's `pathtrace` mode: the CPU reference (NumPy, float64).

The GPU twin is `nodebased/gpupathtrace.py`; both read the scene through `build_scene` and use the
same counter-based random numbers (`pcg`, `rand`), so a pixel's sample stream is the same on both.

What it does. Each pixel receives `samples` paths accumulated in passes. Every path vertex does
next-event estimation (one sample per analytic light and per environment, the environment sampled by a
luminance CDF) and samples the BSDF, and the two are combined by multiple importance sampling (power
heuristic). The BSDF is R1's: Lambert diffuse plus GGX (Smith visibility, Schlick Fresnel, the same
`splatshade._cook_torrance` response), sampled through the visible normal distribution; the diffuse
weight is `(1 - metallic) * (1 - specular albedo)` from `envlight.dfg`, the split-sum weighting the
environment already uses, so a rough dielectric under a uniform sky neither gains nor loses energy.
The liquid material's interface (Fresnel reflection or Snell refraction, Beer-Lambert absorption in the
medium) is one more branch of the same integrator. Russian roulette starts after the third bounce.

The environment is read texel by texel (piecewise constant), the same field the luminance CDF samples, so
light sampling and BSDF sampling estimate one integral; a bilinear read would leak energy the CDF never aims at.

Units. The codebase's lights are radiance-times-pi: an intensity-1 light lights a white diffuse
surface facing it to exactly 1, with no 1/pi left over (`splatshade._cook_torrance` multiplies by pi
for the same reason). Analytic lights therefore emit pi times their `intensity * color`; an
environment's texels are radiance as they are (a uniform map of 1 lights a white surface to 1).

Bounce counts. `max_bounces` is the number of scattering events a path may have. The ray leaving the
last allowed vertex is still traced for emitters and the environment, so `max_bounces=1` is direct
lighting (light sampled at the first hit plus mirror reflections of lights and sky) and equals the
ray-traced renderer's direct lighting. `diffuse_bounces`, `specular_bounces` and `transmission_bounces`
cap the events of each kind. Ambient is a uniform sky of that radiance, found by BSDF sampling and
occluded by geometry; camera rays never see it.

Passes. Beauty components: `emission` (the surface's own emission at the first hit), `diffuse` and
`specular` (light arriving at the first hit directly), `diffuse_indirect` and `specular_indirect` (light
that reaches it after more bounces, filed under the lobe the first bounce chose; liquid interfaces count
as specular). rgba is their sum. The data passes (depth, normals, position, uv, object_id) are one
un-jittered first-hit ray per pixel, never antialiased, like every other renderer here.

Limits, stated: alpha below 0.5 does not block shadow rays and the coverage of an alpha surface is
stochastic; textures are read at their top mip; projections are ignored; emissive meshes are found by
BSDF sampling only (they are not sampled as lights); area lights are visible in reflections but not to
the camera ray itself; liquid roughness and thin sheets are not modelled.
"""
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import envlight, raytrace, scene3d as s
from .liquid_render import fresnel, sigma_of

PI = math.pi
LIGHT_UNIT = PI
MAX_BOUNCES_LIMIT = 64
TILE = 16
MIN_ADAPTIVE_SAMPLES = 16
AOV_OUTPUTS = ("rgba", "diffuse", "specular", "emission", "albedo", "diffuse_indirect", "specular_indirect")
DATA_OUTPUTS = s.DATA_OUTPUTS
PATH_OUTPUTS = AOV_OUTPUTS + DATA_OUTPUTS
_BRUTE_TRIANGLES = 64      # up to this many triangles a shape is intersected by broadcast, above it by its BVH
_DELTA_ROUGHNESS = 0.02    # at or below this a specular lobe is a perfect mirror
_CHUNK = 65536             # paths traced together: bounds the NumPy working set
_DIM_BASE = 8              # random dimensions 0..7 of a path key are the pixel jitter
_DIM_STRIDE = 64           # dimensions one vertex may use
_MAX_LIGHT_SAMPLES = 22    # light-sampling dimensions fit the stride: 2 per area light or environment
_BSDF_DIM = 50             # offset of the BSDF-sampling dimensions inside a vertex's stride


@dataclass(frozen=True)
class PathSettings:
    samples: int = 64
    max_bounces: int = 8
    diffuse_bounces: int = 4
    specular_bounces: int = 8
    transmission_bounces: int = 8
    time_limit: float = 0.0        # seconds of rendering; 0 = no limit
    noise_threshold: float = 0.0   # relative standard error a tile must reach to stop; 0 = every tile takes all samples
    seed: int = 1
    pass_samples: int = 0          # samples per pass; 0 = choose from the image size

    def clamped(self):
        return replace(self, samples=int(np.clip(self.samples, 1, 65536)),
                       max_bounces=int(np.clip(self.max_bounces, 0, MAX_BOUNCES_LIMIT)),
                       diffuse_bounces=int(np.clip(self.diffuse_bounces, 0, MAX_BOUNCES_LIMIT)),
                       specular_bounces=int(np.clip(self.specular_bounces, 0, MAX_BOUNCES_LIMIT)),
                       transmission_bounces=int(np.clip(self.transmission_bounces, 0, MAX_BOUNCES_LIMIT)),
                       time_limit=max(float(self.time_limit), 0.0),
                       noise_threshold=max(float(self.noise_threshold), 0.0),
                       seed=int(self.seed) & 0x7FFFFFFF)


def settings_from_params(params):
    """Render3D's knobs as a `PathSettings`; a document saved before the mode has none of them."""
    return PathSettings(samples=int(params.get("pt_samples", 64)), max_bounces=int(params.get("max_bounces", 8)),
                        diffuse_bounces=int(params.get("diffuse_bounces", 4)),
                        specular_bounces=int(params.get("specular_bounces", 8)),
                        transmission_bounces=int(params.get("transmission_bounces", 8)),
                        time_limit=float(params.get("time_limit", 0.0)),
                        noise_threshold=float(params.get("noise_threshold", 0.0)),
                        seed=int(params.get("pt_seed", 1))).clamped()


# --- random numbers -----------------------------------------------------------------------------------------
# PCG output hash on 32-bit words; gpupathtrace.py carries the same code in WGSL.

def pcg(v):
    v = np.asarray(v, np.uint32)
    with np.errstate(over="ignore"):
        state = v * np.uint32(747796405) + np.uint32(2891336453)
        word = ((state >> ((state >> np.uint32(28)) + np.uint32(4))) ^ state) * np.uint32(277803737)
    return (word >> np.uint32(22)) ^ word


def path_key(pixel, sample, seed):
    """One 32-bit stream key per (pixel, sample, seed)."""
    with np.errstate(over="ignore"):
        return pcg(np.asarray(pixel, np.uint32) + pcg(np.asarray(sample, np.uint32) + pcg(np.uint32(seed))))


def rand(key, dim):
    """Uniform [0, 1) number `dim` of the stream `key`."""
    with np.errstate(over="ignore"):
        mixed = np.asarray(key, np.uint32) ^ (np.uint32(dim) * np.uint32(2654435769))
    return (pcg(mixed) >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)


# --- the scene ----------------------------------------------------------------------------------------------

@dataclass
class _Blas:
    """One mesh in its own space: triangles, their bounding tree and the attributes shading reads."""
    tris: raytrace.TriangleSet
    bvh: object
    v0: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    normals: np.ndarray          # (T, 3, 3) per-vertex, or the face normal repeated
    smooth: bool
    uvs: np.ndarray | None       # (T, 3, 2)
    lo: np.ndarray
    hi: np.ndarray


def _blas_of(geometry):
    v = geometry.vertices.astype(np.float64)
    tri = geometry.triangles
    v0, v1, v2 = (v[tri[:, i]] for i in range(3)) if len(tri) else (np.zeros((0, 3)),) * 3
    e1, e2 = v1 - v0, v2 - v0
    face = np.cross(e1, e2)
    face = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-300)
    if geometry.normals is not None and len(tri):
        n = geometry.normals.astype(np.float64)[tri]
        smooth = True
    else:
        n = np.repeat(face[:, None, :], 3, axis=1)
        smooth = False
    uvs = geometry.uvs.astype(np.float64)[tri] if geometry.uvs is not None and len(tri) else None
    tris = raytrace.TriangleSet(v0, e1, e2, np.ones(len(tri), np.float32))
    bvh = raytrace.Bvh.build(*tris.aabbs()) if len(tri) > _BRUTE_TRIANGLES else None
    lo = np.minimum(np.minimum(v0, v1), v2).min(0) if len(tri) else np.zeros(3)
    hi = np.maximum(np.maximum(v0, v1), v2).max(0) if len(tri) else np.zeros(3)
    return _Blas(tris, bvh, v0, e1, e2, n, smooth, uvs, lo, hi)


@dataclass
class _Env:
    rgb: np.ndarray              # (H, W, 3) decimated map
    gain: np.ndarray             # intensity * tint
    local: object                # world direction -> map frame, `Environment._local`
    lum: np.ndarray              # (H, W) texel luminance
    marginal: np.ndarray         # (H,) cumulative row weight, ends at 1
    conditional: np.ndarray      # (H, W) cumulative within each row, each row ends at 1
    scale: float                 # W * H / (2 pi^2 * sum(lum * sin theta))


def _env_of(environment):
    rgb = envlight._decimate(np.asarray(environment.rgb, np.float64), envlight.MAX_SIDE)
    h, w = rgb.shape[:2]
    lum = rgb @ np.array((0.2126, 0.7152, 0.0722))
    lum = np.maximum(lum, 0.0) * float(np.mean(environment._gain()))
    sin_theta = np.sin((np.arange(h) + 0.5) / h * PI)
    weight = lum * sin_theta[:, None]
    rows = weight.sum(1)
    total = float(rows.sum())
    if total <= 0:
        weight = np.broadcast_to(sin_theta[:, None], (h, w)).copy()
        lum = np.ones((h, w))
        rows = weight.sum(1)
        total = float(rows.sum())
    marginal = np.cumsum(rows) / total
    marginal[-1] = 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        conditional = np.cumsum(weight, axis=1) / np.where(rows > 0, rows, 1.0)[:, None]
    conditional[:, -1] = 1.0
    return _Env(rgb.astype(np.float64), environment._gain().astype(np.float64), environment._local, lum,
                marginal, conditional, w * h / (2 * PI * PI * total))


@dataclass
class _AreaLight:
    kind: str
    position: np.ndarray
    normal: np.ndarray
    right: np.ndarray
    up: np.ndarray
    half_w: float
    half_h: float
    radius: float
    area: float
    radiance: np.ndarray
    two_sided: bool


@dataclass
class _PointLight:
    light: object
    kind: str
    position: np.ndarray
    direction: np.ndarray
    irradiance: np.ndarray


@dataclass
class PathScene:
    """Everything a path needs, with world-space shapes that share one bottom-level tree per source mesh."""
    blases: list
    shape_blas: np.ndarray
    matrix: np.ndarray           # (S, 4, 4) local -> world
    inverse: np.ndarray          # (S, 4, 4) world -> local
    world_lo: np.ndarray
    world_hi: np.ndarray
    object_id: np.ndarray
    base: np.ndarray             # (S, 3)
    alpha: np.ndarray
    kind: np.ndarray             # 0 standard, 1 pbr, 2 liquid
    metallic: np.ndarray
    roughness: np.ndarray
    f0: np.ndarray               # dielectric F0
    emission: np.ndarray
    ior: np.ndarray
    reflection: np.ndarray
    sigma: np.ndarray            # (S, 3)
    texture: list                # per shape: premultiplied RGBA top mip or None
    area_lights: list
    point_lights: list
    envs: list
    ambient: float
    extent: float
    eps: float
    shadow_eps: float            # shadow rays start this far along the shading normal, like the ray-traced renderer's bias
    geometries: tuple = ()

    @property
    def shapes(self):
        return len(self.shape_blas)


def _material_row(geometry):
    if geometry.material == "liquid":
        return (2, 0.0, 0.5, 0.0, max(float(geometry.ior), 1.0), float(np.clip(geometry.reflection, 0, 1)),
                sigma_of(geometry.absorption_color, geometry.absorption_distance))
    if geometry.material == "pbr":
        return (1, float(np.clip(geometry.metallic, 0, 1)), float(np.clip(geometry.pbr_roughness, 0, 1)),
                0.08 * float(np.clip(geometry.pbr_specular, 0, 1)), 1.0, 1.0, np.zeros(3))
    spec = float(np.clip(geometry.specular, 0, 1))
    return (0, 0.0, math.sqrt(2.0 / (max(float(geometry.shininess), 0.0) + 2.0)), 0.08 * spec, 1.0, 1.0, np.zeros(3))


def _area_light(light):
    position, direction = light.world()
    right, up = s._light_basis(direction.astype(np.float64))
    radiance, area = s._area_light_radiance(light)
    return _AreaLight(light.kind, position.astype(np.float64), direction.astype(np.float64), right, up,
                      max(float(light.area_width), 0.0) / 2, max(float(light.area_height), 0.0) / 2,
                      max(float(light.area_radius), 0.0), max(area, 1e-12), radiance * LIGHT_UNIT,
                      bool(light.two_sided) and light.kind != "Sphere")


def build_scene(scene, ambient=0.0):
    """Pack `scene` (instances kept as instances) for tracing."""
    geometries = list(scene.geometries)
    blases, cache = [], {}

    def blas_index(geometry):
        key = (id(geometry.vertices), id(geometry.triangles), id(geometry.normals), id(geometry.uvs))
        if key not in cache:
            cache[key] = len(blases)
            blases.append(_blas_of(geometry))
        return cache[key]

    entries = []   # (geometry, matrix, tint or None)
    for geometry in geometries:
        entries.append((geometry, geometry.world_matrix().astype(np.float64), None))
    for instance_set in scene.instances:
        if not len(instance_set) or not instance_set.sources:
            continue
        parent = instance_set.parent.astype(np.float64)
        bases = [source.world_matrix().astype(np.float64) for source in instance_set.sources]
        for i in range(len(instance_set)):
            variant = int(instance_set.variant[i])
            source = instance_set.sources[variant]
            tint = None if instance_set.colors is None else instance_set.colors[i]
            entries.append((source, parent @ instance_set.matrices[i] @ bases[variant], tint))
    n = len(entries)
    shape_blas = np.zeros(n, np.int32)
    matrix, inverse = np.zeros((n, 4, 4)), np.zeros((n, 4, 4))
    world_lo, world_hi = np.zeros((n, 3)), np.zeros((n, 3))
    base, alpha, sigma = np.zeros((n, 3)), np.ones(n), np.zeros((n, 3))
    kind, metallic, roughness, f0 = np.zeros(n, np.int32), np.zeros(n), np.zeros(n), np.zeros(n)
    emission, ior, reflection = np.zeros(n), np.ones(n), np.ones(n)
    textures = []
    for i, (geometry, m, tint) in enumerate(entries):
        index = blas_index(geometry)
        shape_blas[i] = index
        blas = blases[index]
        matrix[i] = m
        inverse[i] = np.linalg.inv(m) if abs(np.linalg.det(m[:3, :3])) > 1e-18 else np.eye(4)
        corners = np.array([[x, y, z] for x in (blas.lo[0], blas.hi[0]) for y in (blas.lo[1], blas.hi[1])
                            for z in (blas.lo[2], blas.hi[2])])
        world = corners @ m[:3, :3].T + m[:3, 3]
        world_lo[i], world_hi[i] = world.min(0) - 1e-9, world.max(0) + 1e-9
        color = np.asarray(geometry.color, np.float64)
        if tint is not None:
            color = color * np.asarray(tint, np.float64)
        base[i], alpha[i] = color[:3], float(np.clip(color[3], 0, 1))
        row = _material_row(geometry)
        kind[i], metallic[i], roughness[i], f0[i], ior[i], reflection[i], sigma[i] = row
        emission[i] = float(geometry.emission)
        textures.append(s._mip_chain(geometry.texture)[0].astype(np.float64)
                        if geometry.texture is not None and geometry.uvs is not None else None)
    area_lights, point_lights = [], []
    for light in scene.lights:
        if light.intensity <= 0:
            continue
        if light.kind in s._AREA:
            area_lights.append(_area_light(light))
        else:
            position, direction = light.world()
            colour = np.asarray(light.color, np.float64) * float(light.intensity) * LIGHT_UNIT
            point_lights.append(_PointLight(light, light.kind, position.astype(np.float64),
                                            direction.astype(np.float64), colour))
    all_lo = world_lo.min(0) if n else np.zeros(3)
    all_hi = world_hi.max(0) if n else np.zeros(3)
    extent = max(float(np.ptp(np.stack((all_lo, all_hi)), axis=0).max()) if n else 1.0, 1e-6)
    return PathScene(blases, shape_blas, matrix, inverse, world_lo, world_hi,
                     np.arange(1, n + 1, dtype=np.int32), base, alpha, kind, metallic, roughness, f0, emission,
                     ior, reflection, sigma, textures, area_lights, point_lights,
                     [_env_of(e) for e in scene.environments], float(ambient), extent,
                     1e-4 * max(1.0, extent), s.SHADOW_BIAS_DEFAULT * max(1.0, extent), tuple(geometries))


# --- ray queries --------------------------------------------------------------------------------------------

def _brute_closest(blas, o, d, tmin, tmax):
    """Closest hit of rays (o, d) against every triangle of `blas` by broadcast (small meshes)."""
    n = len(o)
    t_best = np.full(n, np.inf)
    prim = np.full(n, -1, np.int64)
    uu, vv = np.zeros(n), np.zeros(n)
    count = len(blas.v0)
    if not count or not n:
        return t_best, prim, uu, vv
    for start in range(0, count, 32):
        sl = slice(start, min(start + 32, count))
        e1, e2, v0 = blas.e1[sl], blas.e2[sl], blas.v0[sl]
        h = np.cross(d[:, None, :], e2[None])
        det = np.einsum("ntj,tj->nt", h, e1)
        valid = np.abs(det) > 1e-12
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
        delta = o[:, None, :] - v0[None]
        u = np.einsum("ntj,ntj->nt", delta, h) * inv
        q = np.cross(delta, e1[None])
        v = np.einsum("nj,ntj->nt", d, q) * inv
        t = np.einsum("tj,ntj->nt", e2, q) * inv
        hit = valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > tmin[:, None]) & (t < np.minimum(tmax, t_best)[:, None])
        t = np.where(hit, t, np.inf)
        best = np.argmin(t, axis=1)
        tb = t[np.arange(n), best]
        better = tb < t_best
        t_best = np.where(better, tb, t_best)
        prim = np.where(better, best + sl.start, prim)
        uu = np.where(better, u[np.arange(n), best], uu)
        vv = np.where(better, v[np.arange(n), best], vv)
    return t_best, prim, uu, vv


def _slab(lo, hi, o, d, tmax):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        a, b = (lo - o) * inv, (hi - o) * inv
    near = np.nanmax(np.minimum(a, b), axis=1)
    far = np.nanmin(np.maximum(a, b), axis=1)
    return (near <= np.minimum(far, tmax)) & (far >= 0)


def closest(ps, o, d, tmin, tmax, cancel=None):
    """(t, shape, prim, u, v) of the nearest hit along each ray; shape -1 for a miss."""
    n = len(o)
    tmin = np.broadcast_to(np.asarray(tmin, np.float64), (n,))
    best = np.broadcast_to(np.asarray(tmax, np.float64), (n,)).copy()
    shape = np.full(n, -1, np.int64)
    prim = np.full(n, -1, np.int64)
    us, vs = np.zeros(n), np.zeros(n)
    for i in range(ps.shapes):
        raytrace._cancel(cancel)
        rays = np.flatnonzero(_slab(ps.world_lo[i], ps.world_hi[i], o, d, best))
        if not len(rays):
            continue
        inverse = ps.inverse[i]
        lo_o = o[rays] @ inverse[:3, :3].T + inverse[:3, 3]
        lo_d = d[rays] @ inverse[:3, :3].T
        blas = ps.blases[ps.shape_blas[i]]
        if blas.bvh is None:
            t, p, u, v = _brute_closest(blas, lo_o, lo_d, tmin[rays], best[rays])
        else:
            t, p, u, v = blas.tris.closest_hit(blas.bvh, lo_o, lo_d, tmin[rays], best[rays])
        got = p >= 0
        if got.any():
            r = rays[got]
            best[r], shape[r], prim[r], us[r], vs[r] = t[got], i, p[got], u[got], v[got]
    return best, shape, prim, us, vs


def occluded(ps, o, d, tmax, cancel=None):
    """True where something opaque lies along (o, d) before `tmax`. Shapes with alpha < 0.5 never block."""
    n = len(o)
    blocked = np.zeros(n, bool)
    tmax = np.broadcast_to(np.asarray(tmax, np.float64), (n,))
    for i in range(ps.shapes):
        if ps.alpha[i] < 0.5:
            continue
        raytrace._cancel(cancel)
        rays = np.flatnonzero(~blocked & _slab(ps.world_lo[i], ps.world_hi[i], o, d, tmax))
        if not len(rays):
            continue
        inverse = ps.inverse[i]
        lo_o = o[rays] @ inverse[:3, :3].T + inverse[:3, 3]
        lo_d = d[rays] @ inverse[:3, :3].T
        blas = ps.blases[ps.shape_blas[i]]
        zero = np.zeros(len(rays))
        if blas.bvh is None:
            t, p, _, _ = _brute_closest(blas, lo_o, lo_d, zero, tmax[rays])
        else:
            t, p, _, _ = blas.tris.closest_hit(blas.bvh, lo_o, lo_d, zero, tmax[rays])
        blocked[rays[p >= 0]] = True
    return blocked


# --- surface data -------------------------------------------------------------------------------------------

def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def _surface(ps, shape, prim, u, v):
    """World-space shading normal `ns`, geometric normal `ng` (same side as `ns`), uv and base colour of hits."""
    n = len(shape)
    ns, ng, uv = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 2))
    base = ps.base[shape].copy()
    alpha = ps.alpha[shape].copy()
    w = np.stack((1 - u - v, u, v), axis=1)
    for index in np.unique(ps.shape_blas[shape]):
        take = np.flatnonzero(ps.shape_blas[shape] == index)
        blas = ps.blases[index]
        p = prim[take]
        local_ns = np.einsum("nk,nkj->nj", w[take], blas.normals[p])
        local_ng = np.cross(blas.e1[p], blas.e2[p])
        inv3 = ps.inverse[shape[take]][:, :3, :3]
        ns[take] = np.einsum("nji,nj->ni", inv3, local_ns)
        ng[take] = np.einsum("nji,nj->ni", inv3, local_ng)
        if blas.uvs is not None:
            uv[take] = np.einsum("nk,nkj->nj", w[take], blas.uvs[p])
    ns, ng = _unit(ns), _unit(ng)
    ng = np.where((np.einsum("ij,ij->i", ng, ns) < 0)[:, None], -ng, ng)
    for i in np.unique(shape):
        if ps.texture[i] is None:
            continue
        take = np.flatnonzero(shape == i)
        texel = s._sample(ps.texture[i], uv[take, 0], uv[take, 1])
        a = np.maximum(texel[:, 3:4], 1e-6)
        base[take] = ps.base[i] * (texel[:, :3] / a)
        alpha[take] = ps.alpha[i] * texel[:, 3]
    return ns, ng, uv, base, alpha


# --- BSDF ---------------------------------------------------------------------------------------------------

def _alpha_of(roughness):
    return np.maximum(roughness, 0.05) ** 2


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _lobes(ps, shape, base, nv):
    """Per-hit lobe constants: diffuse colour (N,3), specular F0 (N,3), compensation (N,3), spec albedo (N,3), delta mask."""
    m = ps.metallic[shape][:, None]
    r = ps.roughness[shape]
    f0d = ps.f0[shape]
    a, b = envlight.dfg(nv, r)
    comp = 1 / np.maximum(a + b, 1e-4)
    f0 = f0d[:, None] * (1 - m) + base * m
    k = 1 + f0 * (comp[:, None] - 1)
    spec_albedo = (f0 * a[:, None] + b[:, None]) * k
    dielectric = (f0d * a + b) * (1 + f0d * (comp - 1))
    # a legacy (Blinn-Phong) surface keeps its whole albedo, as it always has; a PBR one gives the diffuse
    # lobe what the specular lobe does not take
    legacy = (ps.kind[shape] == 0)[:, None]
    diffuse = base * np.where(legacy, 1.0, (1 - m) * (1 - dielectric[:, None]))
    has_spec = (f0.max(axis=1) > 0)
    return diffuse, f0, k, spec_albedo, has_spec


def _select_probability(diffuse, spec_albedo, has_spec):
    ws = spec_albedo.mean(axis=1) * has_spec
    wd = diffuse.mean(axis=1)
    total = np.maximum(ws + wd, 1e-12)
    p = np.clip(ws / total, 0.05, 0.95)
    return np.where(wd <= 1e-9, np.where(has_spec, 1.0, 0.0), np.where(ws <= 1e-9, 0.0, p))


def _ggx_terms(n, v, wi, r):
    h = _unit(wi + v)
    nl = np.maximum(_dot(n, wi), 0)
    nv = np.maximum(_dot(n, v), 1e-4)
    nh = np.maximum(_dot(n, h), 0)
    vh = np.maximum(_dot(v, h), 1e-6)
    alpha = _alpha_of(r)
    a2 = alpha * alpha
    d = a2 / (PI * (nh * nh * (a2 - 1) + 1) ** 2)
    g1 = 2 * nv / (nv + np.sqrt(a2 + (1 - a2) * nv * nv))
    return h, nl, nv, vh, d, g1


def bsdf_eval(ps, shape, base, n, v, wi):
    """(diffuse*cos (N,3), specular*cos (N,3), pdf of sampling `wi` (N,)) at hits.

    `specular` is the GGX response of `splatshade._cook_torrance` (divided by pi: it carries the pi of the
    codebase's lights) with the multiple-scattering compensation of the environment split sum.
    """
    from .splatshade import _cook_torrance
    nv = np.maximum(_dot(n, v), 1e-4)
    diffuse, f0, k, spec_albedo, has_spec = _lobes(ps, shape, base, nv)
    r = ps.roughness[shape]
    nl = np.maximum(_dot(n, wi), 0)
    f_diff = diffuse * (nl / PI)[:, None]
    delta = r <= _DELTA_ROUGHNESS
    response, _ = _cook_torrance(n, v, wi, r, f0)
    f_spec = response / PI * k * (has_spec & ~delta)[:, None]
    _, _, _, vh, d, g1 = _ggx_terms(n, v, wi, r)
    pdf_spec = g1 * d / (4 * nv)
    p_spec = _select_probability(diffuse, spec_albedo, has_spec)
    p_spec_eff = np.where(delta, 0.0, p_spec)
    pdf = (1 - p_spec) * nl / PI + p_spec_eff * pdf_spec
    return f_diff, f_spec, pdf


def _frame(n):
    sign = np.where(n[:, 2] >= 0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t = np.stack((1 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]), axis=1)
    bt = np.stack((b, sign + n[:, 1] ** 2 * a, -n[:, 1]), axis=1)
    return t, bt


def _cosine_sample(u1, u2):
    r = np.sqrt(u1)
    phi = 2 * PI * u2
    return np.stack((r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0, 1 - u1))), axis=1)


def _vndf_sample(vl, alpha, u1, u2):
    """Heitz 2018: a visible microfacet normal in the local frame for view `vl` (z up)."""
    vh = _unit(np.stack((alpha * vl[:, 0], alpha * vl[:, 1], vl[:, 2]), axis=1))
    lensq = vh[:, 0] ** 2 + vh[:, 1] ** 2
    t1 = np.where(lensq[:, None] > 1e-12,
                  np.stack((-vh[:, 1], vh[:, 0], np.zeros(len(vh))), axis=1) / np.sqrt(np.maximum(lensq, 1e-12))[:, None],
                  np.array((1.0, 0.0, 0.0)))
    t2 = np.cross(vh, t1)
    r = np.sqrt(u1)
    phi = 2 * PI * u2
    a, b = r * np.cos(phi), r * np.sin(phi)
    s_ = 0.5 * (1 + vh[:, 2])
    b = (1 - s_) * np.sqrt(np.maximum(0, 1 - a * a)) + s_ * b
    nh = a[:, None] * t1 + b[:, None] * t2 + np.sqrt(np.maximum(0, 1 - a * a - b * b))[:, None] * vh
    return _unit(np.stack((alpha * nh[:, 0], alpha * nh[:, 1], np.maximum(nh[:, 2], 0)), axis=1))


# --- lights -------------------------------------------------------------------------------------------------

def _texel(env, dirs):
    u, v = envlight._uv(env.local(dirs))
    h, w = env.lum.shape
    return np.minimum((v * h).astype(np.int64), h - 1), np.minimum((u * w).astype(np.int64), w - 1)


def env_radiance(env, dirs):
    """The map read texel by texel: the same piecewise-constant field the CDF samples, so light sampling and
    BSDF sampling estimate one integral (a bilinear read would leak energy the CDF never aims at)."""
    y, x = _texel(env, dirs)
    return env.rgb[y, x] * env.gain


def env_pdf(env, dirs):
    y, x = _texel(env, dirs)
    return env.lum[y, x] * env.scale


def env_sample(env, u1, u2):
    """World direction, its solid-angle pdf and its radiance for uniform numbers (u1, u2)."""
    h, w = env.lum.shape
    row = np.minimum(np.searchsorted(env.marginal, u1, side="right"), h - 1)
    prev = np.where(row > 0, env.marginal[np.maximum(row - 1, 0)], 0.0)
    fu = np.clip((u1 - prev) / np.maximum(env.marginal[row] - prev, 1e-300), 0, 1 - 1e-12)
    cdf_rows = env.conditional[row]
    col = np.minimum((cdf_rows <= u2[:, None]).sum(1), w - 1)
    cprev = np.where(col > 0, cdf_rows[np.arange(len(col)), np.maximum(col - 1, 0)], 0.0)
    fv = np.clip((u2 - cprev) / np.maximum(cdf_rows[np.arange(len(col)), col] - cprev, 1e-300), 0, 1 - 1e-12)
    theta = (row + fu) / h * PI
    phi = ((col + fv) / w - 0.5) * 2 * PI
    local = np.stack((np.sin(theta) * np.sin(phi), np.cos(theta), -np.sin(theta) * np.cos(phi)), axis=1)
    world = _to_world(env, local)
    pdf = env.lum[row, col] * env.scale
    return world, pdf, env.rgb[row, col] * env.gain


def _to_world(env, local):
    """Inverse of `env.local` for directions; the map turn is a rotation so its inverse is its transpose."""
    probe = np.eye(3)
    forward = env.local(probe)          # rows: images of the world axes in the map frame
    return local @ forward.T


def _area_sample(light, u1, u2):
    """(points on the light (N,3), outward normals (N,3), area pdf) by uniform area sampling."""
    n = len(u1)
    if light.kind == "Sphere":
        z = 1 - 2 * u1
        r = np.sqrt(np.maximum(1 - z * z, 0))
        phi = 2 * PI * u2
        local = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)
        return light.position + local * light.radius, local
    if light.kind == "Rect":
        ox, oy = (u1 * 2 - 1) * light.half_w, (u2 * 2 - 1) * light.half_h
    else:
        r = np.sqrt(u1) * light.radius
        phi = 2 * PI * u2
        ox, oy = r * np.cos(phi), r * np.sin(phi)
    points = light.position + light.right * ox[:, None] + light.up * oy[:, None]
    return points, np.broadcast_to(light.normal, (n, 3))


def _area_hits(light, o, d, tmax):
    """Distance along each ray to an emitting face of `light` (inf for none), and the front-facing mask."""
    n = len(o)
    t = np.full(n, np.inf)
    if light.kind == "Sphere":
        oc = o - light.position
        b = _dot(oc, d)
        c = _dot(oc, oc) - light.radius ** 2
        disc = b * b - c
        root = np.sqrt(np.maximum(disc, 0))
        t0 = -b - root
        hit = (disc > 0) & (t0 > 1e-9)
        return np.where(hit & (t0 < tmax), t0, np.inf)
    denom = d @ light.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = ((light.position - o) @ light.normal) / denom
    ok = (np.abs(denom) > 1e-12) & (tt > 1e-9) & (tt < tmax) & ((denom < 0) | light.two_sided)
    p = o + d * np.where(ok, tt, 0)[:, None] - light.position
    x, y = p @ light.right, p @ light.up
    if light.kind == "Rect":
        inside = (np.abs(x) <= light.half_w) & (np.abs(y) <= light.half_h)
    else:
        inside = x * x + y * y <= light.radius ** 2
    return np.where(ok & inside, tt, np.inf)


def _area_pdf(light, dist, cos_l):
    return dist * dist / np.maximum(cos_l * light.area, 1e-30)


def _mis(pa, pb):
    a, b = pa * pa, pb * pb
    return a / np.maximum(a + b, 1e-300)


# --- the integrator -----------------------------------------------------------------------------------------

class _Accum:
    """Per-path radiance buckets."""
    def __init__(self, n):
        z = lambda: np.zeros((n, 3))
        self.emission, self.diffuse, self.specular = z(), z(), z()
        self.diffuse_indirect, self.specular_indirect = z(), z()

    def total(self):
        return self.emission + self.diffuse + self.specular + self.diffuse_indirect + self.specular_indirect


def _add_class(acc, rows, cls, vertex_depth, value):
    """File `value` (M,3) of paths `rows` by the first bounce's lobe `cls` (0 diffuse, 1 specular); light
    that reaches the first vertex from depth 1 is direct, anything deeper is indirect."""
    direct = np.broadcast_to(np.asarray(vertex_depth) <= 1, (len(rows),))
    for c, direct_bucket, indirect_bucket in ((0, acc.diffuse, acc.diffuse_indirect),
                                              (1, acc.specular, acc.specular_indirect)):
        pick = cls == c
        for is_direct, bucket in ((True, direct_bucket), (False, indirect_bucket)):
            sel = pick & (direct == is_direct)
            if sel.any():
                bucket[rows[sel]] += value[sel]


def trace_paths(ps, o, d, keys, settings, tmin, tmax, cancel=None):
    """Radiance buckets and first-hit information for `len(o)` camera rays (unit directions `d`).

    `tmin`/`tmax` bound the camera ray only (the near and far planes); every later ray starts at zero.
    Returns (`_Accum`, first) where `first` holds alpha (coverage), albedo and the first hit's shape, t,
    shading normal, position and uv for the data passes.
    """
    n = len(o)
    acc = _Accum(n)
    first = dict(alpha=np.zeros(n), albedo=np.zeros((n, 3)), shape=np.full(n, -1), t=np.zeros(n),
                 ns=np.zeros((n, 3)), pos=np.zeros((n, 3)), uv=np.zeros((n, 2)))
    throughput = np.ones((n, 3))
    alive = np.ones(n, bool)
    bounces = np.zeros(n, np.int32)
    counts = np.zeros((n, 3), np.int32)        # diffuse, specular, transmission scatters so far
    cls = np.full(n, -1, np.int32)             # the first scatter's lobe: 0 diffuse, 1 specular (liquid counts)
    prev_delta = np.ones(n, bool)              # the last scatter was a perfect mirror or refraction (no MIS)
    capped = np.zeros(n, bool)                 # the last scatter reached its kind's cap: the next ray sees emitters only
    prev_pdf = np.zeros(n)
    medium = np.full(n, -1, np.int64)
    vertex_depth = np.zeros(n, np.int32)       # scatters so far: the depth of the vertex the ray reaches next
    o, d = o.copy(), d.copy()
    tmin, tmax = tmin.copy(), tmax.copy()
    if len(ps.area_lights) + len(ps.envs) > _MAX_LIGHT_SAMPLES:
        raise ValueError(f"the path tracer takes at most {_MAX_LIGHT_SAMPLES} area lights and environments together")
    for turn in range(settings.max_bounces + 40):
        raytrace._cancel(cancel)
        rows = np.flatnonzero(alive)
        if not len(rows):
            break
        ro, rd = o[rows], d[rows]
        t, shape, prim, u, v = closest(ps, ro, rd, tmin[rows], tmax[rows], cancel)
        tmin[:], tmax[:] = 0.0, np.inf
        vd = vertex_depth[rows]
        dim = _DIM_BASE + turn * _DIM_STRIDE
        # emitters: the analytic area lights, seen by every ray but the camera's own
        t_light = np.full(len(rows), np.inf)
        light_id = np.full(len(rows), -1)
        for li, light in enumerate(ps.area_lights):
            tl = _area_hits(light, ro, rd, t)
            better = tl < t_light
            t_light[better], light_id[better] = tl[better], li
        lit = (light_id >= 0) & (vd >= 1)
        if lit.any():
            sel = np.flatnonzero(lit)
            contrib = np.zeros((len(sel), 3))
            for li, light in enumerate(ps.area_lights):
                pick = light_id[sel] == li
                if not pick.any():
                    continue
                idx = sel[pick]
                point = ro[idx] + rd[idx] * t_light[idx][:, None]
                normal = _unit(point - light.position) if light.kind == "Sphere" else np.broadcast_to(light.normal, point.shape)
                facing = _dot(normal, rd[idx])
                cos_l = np.abs(facing) if light.two_sided else np.maximum(-facing, 0)
                pdf_l = _area_pdf(light, t_light[idx], cos_l)
                weight = np.where(prev_delta[rows[idx]], 1.0, _mis(prev_pdf[rows[idx]], pdf_l))
                contrib[pick] = light.radiance * weight[:, None]
            _add_class(acc, rows[sel], cls[rows[sel]], vd[sel], throughput[rows[sel]] * contrib)
            alive[rows[sel]] = False
        gone = lit
        # rays that leave the scene see the ambient sky and the environments (the camera ray sees neither)
        miss = (shape < 0) & ~gone
        if miss.any():
            m = np.flatnonzero(miss)
            r = rows[m]
            sky = np.full((len(m), 3), ps.ambient)
            for env in ps.envs:
                radiance = env_radiance(env, rd[m])
                weight = np.where(prev_delta[r], 1.0, _mis(prev_pdf[r], env_pdf(env, rd[m])))
                sky += radiance * weight[:, None]
            deep = vd[m] >= 1
            if deep.any():
                _add_class(acc, r[deep], cls[r[deep]], vd[m][deep], throughput[r[deep]] * sky[deep])
            alive[r] = False
        hit = (shape >= 0) & ~gone
        if not hit.any():
            continue
        h = np.flatnonzero(hit)
        r = rows[h]
        sh = shape[h]
        ns, ng, uv, base, alpha = _surface(ps, sh, prim[h], u[h], v[h])
        wd = rd[h]
        wo = -wd
        pos = ro[h] + wd * t[h][:, None]
        inside = medium[r] >= 0
        if inside.any():
            throughput[r[inside]] *= np.exp(-ps.sigma[medium[r[inside]]] * t[h][inside][:, None])
        liquid = ps.kind[sh] == 2
        # two-sided surfaces face the ray; a liquid keeps its outward normal to know which side it is entering
        flip = (_dot(ns, wo) < 0) & ~liquid
        ns = np.where(flip[:, None], -ns, ns)
        ng = np.where(flip[:, None], -ng, ng)
        # stochastic coverage of an alpha surface: the ray passes straight through the missing part
        cover = np.where(alpha < 1.0, rand(keys[r], dim) < alpha, True)
        through = np.flatnonzero(~cover)
        if len(through):
            o[r[through]] = pos[through] + wd[through] * ps.eps
        vd_h = vertex_depth[r]
        f = np.flatnonzero((vd_h == 0) & cover)
        if len(f):
            fr = r[f]
            first["alpha"][fr] = alpha[f]
            first["albedo"][fr] = base[f] * alpha[f][:, None]
            first["shape"][fr], first["t"][fr] = sh[f], t[h][f]
            first["ns"][fr], first["pos"][fr], first["uv"][fr] = ns[f], pos[f], uv[f]
        keep = np.flatnonzero(cover)
        if not len(keep):
            continue
        r, sh, ns, ng, base, pos, wo, wd = (x[keep] for x in (r, sh, ns, ng, base, pos, wo, wd))
        vd = vertex_depth[r]
        has_emit = ps.emission[sh] > 0
        if has_emit.any():
            e = np.flatnonzero(has_emit)
            gathered = throughput[r[e]] * base[e] * ps.emission[sh[e]][:, None]
            first_vertex = vd[e] == 0
            acc.emission[r[e][first_vertex]] += gathered[first_vertex]
            if (~first_vertex).any():
                _add_class(acc, r[e][~first_vertex], cls[r[e][~first_vertex]], vd[e][~first_vertex],
                           gathered[~first_vertex])
        # the ray that left the last allowed vertex was traced for emitters only
        can_scatter = (bounces[r] < settings.max_bounces) & ~capped[r]
        alive[r[~can_scatter]] = False
        sc = np.flatnonzero(can_scatter)
        if not len(sc):
            continue
        r, sh, ns, ng, base, pos, wo, wd, vd = (x[sc] for x in (r, sh, ns, ng, base, pos, wo, wd, vd))
        liquid = ps.kind[sh] == 2
        wet = np.flatnonzero(liquid)
        if len(wet):
            _liquid_event(ps, keys, r[wet], sh[wet], ns[wet], wd[wet], pos[wet], o, d, medium, alive, counts,
                          cls, prev_delta, capped, bounces, vertex_depth, settings, dim)
        dry = np.flatnonzero(~liquid)
        if len(dry):
            _surface_event(ps, keys, r[dry], sh[dry], ns[dry], ng[dry], base[dry], pos[dry], wo[dry], vd[dry],
                           o, d, throughput, alive, counts, cls, prev_delta, prev_pdf, capped, bounces, vertex_depth,
                           acc, settings, dim, cancel)
    return acc, first


def _liquid_event(ps, keys, r, sh, ns, wd, pos, o, d, medium, alive, counts, cls, prev_delta, capped, bounces,
                  vertex_depth, settings, dim):
    """One interface event at liquid vertices `r`: Fresnel picks reflection, else the ray refracts."""
    entering = _dot(wd, ns) < 0
    n = np.where(entering[:, None], ns, -ns)
    ior = ps.ior[sh]
    eta_i, eta_t = np.where(entering, 1.0, ior), np.where(entering, ior, 1.0)
    cos_i = np.clip(-_dot(wd, n), 0.0, 1.0)
    f, tir, cos_t = fresnel(cos_i, eta_i, eta_t)
    weight_r = np.where(tir, 1.0, ps.reflection[sh] * f)
    reflect = rand(keys[r], dim + 1) < weight_r
    ratio = eta_i / eta_t
    refl = wd + 2.0 * cos_i[:, None] * n
    trans = _unit(ratio[:, None] * wd + (ratio * cos_i - cos_t)[:, None] * n)
    new_dir = np.where(reflect[:, None], refl, trans)
    side = np.where(reflect[:, None], n, -n)
    kind_index = np.where(reflect, 1, 2)
    limit = np.where(reflect, settings.specular_bounces, settings.transmission_bounces)
    ok = counts[r, kind_index] < limit
    alive[r[~ok]] = False
    go = np.flatnonzero(ok)
    g = r[go]
    o[g], d[g] = pos[go] + side[go] * ps.eps, new_dir[go]
    counts[g, kind_index[go]] += 1
    capped[g] = counts[g, kind_index[go]] >= limit[go]
    medium[g] = np.where(reflect[go], medium[g], np.where(entering[go], sh[go], -1))
    cls[g[cls[g] < 0]] = 1
    prev_delta[g] = True
    bounces[g] += 1
    vertex_depth[g] += 1


def _surface_event(ps, keys, r, sh, ns, ng, base, pos, wo, vd, o, d, throughput, alive, counts, cls,
                   prev_delta, prev_pdf, capped, bounces, vertex_depth, acc, settings, dim, cancel):
    """Light sampling and BSDF sampling at the surface vertices `r` (path indices)."""
    n = len(r)
    nv = np.maximum(_dot(ns, wo), 1e-4)
    diffuse, f0, k, spec_albedo, has_spec = _lobes(ps, sh, base, nv)
    rough = ps.roughness[sh]
    delta = (rough <= _DELTA_ROUGHNESS) & has_spec
    origin_side = lambda w: np.where((_dot(w, ng) >= 0)[:, None], ng, -ng)
    # ---- next-event estimation ----
    direct_d = np.zeros((n, 3))
    direct_s = np.zeros((n, 3))
    key = keys[r]
    slot = 0

    def add(f_diff, f_spec, weight):
        nonlocal direct_d, direct_s
        direct_d += f_diff * weight[:, None]
        direct_s += f_spec * weight[:, None]

    for light in ps.point_lights:
        wi, dist, irr = _point_wi(light, pos)
        nl = _dot(ns, wi)
        front = (nl > 0) & (irr.max(axis=1) > 0)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        po = pos[idx] + ns[idx] * ps.shadow_eps
        hidden = occluded(ps, po, wi[idx], dist[idx] - ps.shadow_eps, cancel)
        idx = idx[~hidden]
        if not len(idx):
            continue
        f_diff, f_spec, _ = bsdf_eval(ps, sh[idx], base[idx], ns[idx], wo[idx], wi[idx])
        direct_d[idx] += f_diff * irr[idx]
        direct_s[idx] += f_spec * irr[idx]
    for li, light in enumerate(ps.area_lights):
        u1, u2 = rand(key, dim + 2 + slot), rand(key, dim + 3 + slot)
        slot += 2
        q, nq = _area_sample(light, u1, u2)
        to = q - pos
        dist = np.linalg.norm(to, axis=1)
        wi = to / np.maximum(dist, 1e-12)[:, None]
        cos_l = _dot(-wi, nq)
        cos_l = np.abs(cos_l) if light.two_sided else np.maximum(cos_l, 0)
        nl = _dot(ns, wi)
        front = (nl > 0) & (cos_l > 1e-9) & (dist > 1e-9)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        po = pos[idx] + ns[idx] * ps.shadow_eps
        hidden = occluded(ps, po, wi[idx], dist[idx] * (1 - 1e-4) - ps.shadow_eps, cancel)
        idx = idx[~hidden]
        if not len(idx):
            continue
        pdf_l = _area_pdf(light, dist[idx], cos_l[idx])
        f_diff, f_spec, pdf_b = bsdf_eval(ps, sh[idx], base[idx], ns[idx], wo[idx], wi[idx])
        weight = _mis(pdf_l, pdf_b) / pdf_l
        direct_d[idx] += f_diff * light.radiance * weight[:, None]
        direct_s[idx] += f_spec * light.radiance * weight[:, None]
    for env in ps.envs:
        u1, u2 = rand(key, dim + 2 + slot), rand(key, dim + 3 + slot)
        slot += 2
        wi, pdf_l, radiance = env_sample(env, u1, u2)
        nl = _dot(ns, wi)
        front = (nl > 0) & (pdf_l > 0)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        po = pos[idx] + ns[idx] * ps.shadow_eps
        hidden = occluded(ps, po, wi[idx], np.inf, cancel)
        idx = idx[~hidden]
        if not len(idx):
            continue
        f_diff, f_spec, pdf_b = bsdf_eval(ps, sh[idx], base[idx], ns[idx], wo[idx], wi[idx])
        weight = _mis(pdf_l[idx], pdf_b) / pdf_l[idx]
        direct_d[idx] += f_diff * radiance[idx] * weight[:, None]
        direct_s[idx] += f_spec * radiance[idx] * weight[:, None]
    tp = throughput[r]
    first_vertex = vd == 0
    if first_vertex.any():
        f = np.flatnonzero(first_vertex)
        acc.diffuse[r[f]] += tp[f] * direct_d[f]
        acc.specular[r[f]] += tp[f] * direct_s[f]
    if (~first_vertex).any():
        f = np.flatnonzero(~first_vertex)
        gathered = tp[f] * (direct_d[f] + direct_s[f])
        _add_class(acc, r[f], cls[r[f]], vd[f] + 1, gathered)
    # ---- BSDF sampling ----
    u_lobe, u1, u2 = rand(key, dim + _BSDF_DIM), rand(key, dim + _BSDF_DIM + 1), rand(key, dim + _BSDF_DIM + 2)
    p_spec = _select_probability(diffuse, spec_albedo, has_spec)
    choose_spec = u_lobe < p_spec
    frame_t, frame_b = _frame(ns)
    to_local = lambda w: np.stack((_dot(w, frame_t), _dot(w, frame_b), _dot(w, ns)), axis=1)
    from_local = lambda w: w[:, :1] * frame_t + w[:, 1:2] * frame_b + w[:, 2:3] * ns
    wi_diffuse = from_local(_cosine_sample(u1, u2))
    alpha = _alpha_of(rough)
    hl = _vndf_sample(to_local(wo), alpha, u1, u2)
    wi_spec = from_local(2 * np.sum(to_local(wo) * hl, axis=1, keepdims=True) * hl - to_local(wo))
    mirror = 2 * nv[:, None] * ns - wo
    wi_spec = np.where(delta[:, None], mirror, wi_spec)
    wi = np.where(choose_spec[:, None], wi_spec, wi_diffuse)
    cos = np.maximum(_dot(ns, wi), 0)
    f_diff, f_spec, pdf = bsdf_eval(ps, sh, base, ns, wo, wi)
    weight = np.zeros((n, 3))
    safe = pdf > 1e-12
    weight[safe] = (f_diff[safe] + f_spec[safe]) / pdf[safe][:, None]
    # a perfect mirror lobe: a discrete choice with probability p_spec and the Fresnel weight
    schlick = f0 + (1 - f0) * ((1 - nv) ** 5)[:, None]
    mirror_weight = schlick * k / np.maximum(p_spec, 1e-12)[:, None]
    is_delta = choose_spec & delta
    weight = np.where(is_delta[:, None], mirror_weight, weight)
    valid = (cos > 0) | is_delta
    kind_index = np.where(choose_spec, 1, 0)
    ok = valid & (counts[r, kind_index] < np.where(choose_spec, settings.specular_bounces, settings.diffuse_bounces))
    new_tp = tp * weight
    ok &= new_tp.max(axis=1) > 0
    alive[r[~ok]] = False
    go = np.flatnonzero(ok)
    g = r[go]
    throughput[g] = new_tp[go]
    side = origin_side(wi)
    o[g] = pos[go] + side[go] * ps.eps
    d[g] = wi[go]
    counts[g, kind_index[go]] += 1
    capped[g] = counts[g, kind_index[go]] >= np.where(choose_spec[go], settings.specular_bounces, settings.diffuse_bounces)
    first_bounce = cls[g] < 0
    cls[g[first_bounce]] = np.where(choose_spec[go][first_bounce], 1, 0)
    prev_delta[g] = is_delta[go]
    prev_pdf[g] = pdf[go]
    bounces[g] += 1
    vertex_depth[g] += 1
    # Russian roulette from the fourth scatter on
    roulette = go[bounces[g] >= 3]
    if len(roulette):
        rg = r[roulette]
        q = np.clip(throughput[rg].max(axis=1), 0.05, 0.95)
        survive = rand(key[roulette], dim + _BSDF_DIM + 3) < q
        throughput[rg[survive]] /= q[survive][:, None]
        alive[rg[~survive]] = False


def _point_wi(light, pos):
    if light.kind == "Directional":
        wi = np.broadcast_to(-light.direction, pos.shape).copy()
        irr = np.broadcast_to(light.irradiance, pos.shape).copy()
        return wi, np.full(len(pos), np.inf), irr
    to = light.position - pos
    dist = np.linalg.norm(to, axis=1)
    wi = to / np.maximum(dist, 1e-12)[:, None]
    atten = s._light_factor(light.light, pos)
    irr = np.broadcast_to(light.irradiance, pos.shape).copy()
    if atten is not None:
        irr = irr * np.asarray(atten, np.float64)[:, None]
    return wi, dist, irr


# --- rendering ----------------------------------------------------------------------------------------------

def camera_rays(camera, width, height, x, y):
    """Unit world directions through pixel positions (x, y) (fractions of a pixel), the eye, the cosine of
    each ray with the view axis (view depth is t / c) and the near/far bounds along each unit ray."""
    eye, view = s._view_basis(camera)
    inverse_view = np.linalg.inv(view.astype(np.float64))
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    aspect = width / max(height, 1)
    local = np.column_stack(((2 * x / width - 1) * aspect / focal, (1 - 2 * y / height) / focal, -np.ones(len(x))))
    dirs = local @ inverse_view.T
    c = np.linalg.norm(dirs, axis=1)
    return dirs / c[:, None], eye.astype(np.float64), 1.0 / c, camera.near * c, camera.far * c


def _channels(output):
    return {"rgba": "total", "diffuse": "diffuse", "specular": "specular", "emission": "emission",
            "albedo": "albedo", "diffuse_indirect": "diffuse_indirect",
            "specular_indirect": "specular_indirect"}[output]


def _read_only(a):
    a.flags.writeable = False
    return a


def render_data(ps, camera, width, height, output, cancel=None):
    """The data passes: one un-jittered first-hit ray per pixel, coverage in alpha, never antialiased."""
    n = width * height
    out = np.zeros((n, 4), np.float32)
    pix = np.arange(n)
    x, y = pix % width + 0.5, pix // width + 0.5
    d, eye, cos, tmin, tmax = camera_rays(camera, width, height, x, y)
    for start in range(0, n, _CHUNK):
        raytrace._cancel(cancel)
        sl = slice(start, min(start + _CHUNK, n))
        o = np.broadcast_to(eye, (sl.stop - sl.start, 3))
        t, shape, prim, u, v = closest(ps, o, d[sl], tmin[sl], tmax[sl], cancel)
        got = shape >= 0
        # an alpha surface is a hit for the data passes, like every other renderer here
        ns, _, uv, _, _ = _surface(ps, shape[got], prim[got], u[got], v[got])
        wo = -d[sl][got]
        ns = np.where((_dot(ns, wo) < 0)[:, None], -ns, ns)
        pos = o[got] + d[sl][got] * t[got][:, None]
        if output == "depth":
            values = np.repeat((t[got] * cos[sl][got])[:, None], 3, axis=1)
        elif output == "normals":
            values = ns
        elif output == "position":
            values = pos
        elif output == "uv":
            values = np.column_stack((uv, np.zeros(len(uv))))
        else:
            values = np.column_stack((ps.object_id[shape[got]], np.zeros(len(uv)), np.zeros(len(uv))))
        block = np.zeros((sl.stop - sl.start, 4), np.float32)
        block[got, :3] = values
        block[got, 3] = 1.0
        out[sl] = block
    return _read_only(out.reshape(height, width, 4))


def render(scene, camera, width, height, background=(0., 0., 0., 0.), ambient=0.0, output="rgba",
           settings=None, cancel=None, progress=None, stats=None, backend="cpu", pass_hook=None):
    """Path trace `scene` to a premultiplied float32 (height, width, 4) image.

    `backend` "cpu" is this reference; "auto" and "gpu" use `gpupathtrace` (auto falls back to the CPU
    reference when the GPU cannot take the scene, gpu reports why not). `stats`, when a dict, receives
    `samples` (per-pixel counts), `passes`, `seconds`, `backend`. `progress(stage, fraction, info)` runs
    after every pass; the render can be cancelled through `cancel` between passes and inside them.
    """
    if output not in PATH_OUTPUTS:
        raise ValueError(f"Unknown path traced output {output!r}")
    settings = (settings or PathSettings()).clamped()
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("Render dimensions must be positive")
    raytrace._cancel(cancel)
    check_scene(scene)
    if backend in ("auto", "gpu"):
        from . import gpu3d, gpupathtrace
        from .cancellation import Cancelled
        try:
            if not gpu3d.available():
                raise gpu3d.Unsupported(gpu3d.describe())
            return gpupathtrace.render(scene, camera, width, height, background, ambient, output, settings,
                                       cancel=cancel, progress=progress, stats=stats)
        except Cancelled:
            raise
        except gpu3d.Unsupported as exc:
            if backend == "gpu":
                raise ValueError(f"GPU Render3D unsupported: {exc}") from exc
            reason = str(exc)
        except Exception as exc:
            # device loss, out of memory or an adapter error after `available()` said yes
            if backend == "gpu":
                raise ValueError(f"GPU Render3D failed: {exc}") from exc
            reason = str(exc)
        if stats is not None:
            stats["fallback"] = reason
    started = time.perf_counter()
    ps = build_scene(scene, ambient)
    if stats is not None:
        stats["backend"] = "cpu"
    if output in DATA_OUTPUTS:
        return render_data(ps, camera, width, height, output, cancel)
    npix = width * height
    channel = _channels(output)
    total = np.zeros((npix, 3))
    alpha_sum = np.zeros(npix)
    lum_sum, lum_sq = np.zeros(npix), np.zeros(npix)
    count = np.zeros(npix, np.int64)
    tiles_x, tiles_y = -(-width // TILE), -(-height // TILE)
    tile_of = ((np.arange(npix) // width) // TILE) * tiles_x + (np.arange(npix) % width) // TILE
    tile_done = np.zeros(tiles_x * tiles_y, bool)
    per_pass = settings.pass_samples or int(np.clip(_CHUNK // npix, 1, 8))
    sample_index, passes = 0, 0
    while sample_index < settings.samples:
        raytrace._cancel(cancel)
        take = min(per_pass, settings.samples - sample_index)
        active = np.flatnonzero(~tile_done[tile_of])
        if not len(active):
            break
        for start in range(0, len(active), max(1, _CHUNK // take)):
            raytrace._cancel(cancel)
            pixels = active[start:start + max(1, _CHUNK // take)]
            rep = np.repeat(pixels, take)
            sample = sample_index + np.tile(np.arange(take), len(pixels))
            keys = path_key(rep, sample, settings.seed)
            x = rep % width + rand(keys, 0)
            y = rep // width + rand(keys, 1)
            d, eye, _, tmin, tmax = camera_rays(camera, width, height, x, y)
            acc, first = trace_paths(ps, np.broadcast_to(eye, d.shape), d, keys, settings, tmin, tmax, cancel)
            if channel == "total":
                value = acc.total()
            elif channel == "albedo":
                value = first["albedo"]
            else:
                value = getattr(acc, channel)
            per_pixel = value.reshape(len(pixels), take, 3).sum(axis=1)
            total[pixels] += per_pixel
            alpha_sum[pixels] += first["alpha"].reshape(len(pixels), take).sum(axis=1)
            lum = (acc.total() @ np.array((0.2126, 0.7152, 0.0722))).reshape(len(pixels), take)
            lum_sum[pixels] += lum.sum(axis=1)
            lum_sq[pixels] += (lum * lum).sum(axis=1)
            count[pixels] += take
        sample_index += take
        passes += 1
        if settings.noise_threshold > 0:
            _retire_tiles(tile_done, tile_of, lum_sum, lum_sq, count, tiles_x * tiles_y, settings.noise_threshold)
        elapsed = time.perf_counter() - started
        if progress is not None:
            progress("pathtrace", sample_index / settings.samples,
                     dict(samples=sample_index, passes=passes, seconds=elapsed, tiles_active=int((~tile_done).sum())))
        if pass_hook is not None:
            pass_hook(passes, sample_index)
        if settings.time_limit and elapsed >= settings.time_limit:
            break
    raytrace._cancel(cancel)
    n = np.maximum(count, 1)[:, None]
    rgb = total / n
    alpha = alpha_sum / np.maximum(count, 1)
    image = np.zeros((npix, 4))
    image[:, :3] = rgb
    image[:, 3] = alpha
    if output == "rgba":
        bg = np.asarray(background, np.float64).copy()
        bg[3] = np.clip(bg[3], 0, 1)
        bg[:3] *= bg[3]
        image = image + bg * (1 - image[:, 3:4])
    if stats is not None:
        stats.update(samples=count.reshape(height, width).copy(), passes=passes,
                     seconds=time.perf_counter() - started)
    return _read_only(image.reshape(height, width, 4).astype(np.float32))


def _retire_tiles(tile_done, tile_of, lum_sum, lum_sq, count, tiles, threshold):
    """Mark tiles whose relative standard error of the mean luminance is under `threshold`."""
    n = np.maximum(count, 1).astype(np.float64)
    mean = lum_sum / n
    var = np.maximum(lum_sq / n - mean * mean, 0) * n / np.maximum(n - 1, 1)
    err2 = np.bincount(tile_of, weights=var / n, minlength=tiles)
    sums = np.bincount(tile_of, weights=mean, minlength=tiles)
    sizes = np.maximum(np.bincount(tile_of, minlength=tiles), 1)
    tile_n = np.bincount(tile_of, weights=count, minlength=tiles) / sizes
    noise = np.sqrt(err2 / sizes) / (sums / sizes + 0.02)
    tile_done |= (tile_n >= MIN_ADAPTIVE_SAMPLES) & (noise < threshold)


def check_scene(scene):
    """The path tracer renders meshes and instances lit by lights and environments; it does not draw the rest."""
    for name in ("splats", "particles", "volumes"):
        if getattr(scene, name, ()):
            raise ValueError(f"the path tracer does not render {name} yet; use raytrace mode for a scene that has them")


def render_scene3d(scene, camera, width, height, background, ambient, output, cancel, progress, return_depth, path):
    """`scene3d.render`'s entry: one output, the CPU reference, `path` the `PathSettings` (defaults when None)."""
    check_scene(scene)
    if output == "normals_blend":
        output = "normals"
    if output not in PATH_OUTPUTS:
        raise ValueError(f"the path tracer does not produce the {output!r} output")
    image = render(scene, camera, width, height, background, ambient, output, path, cancel=cancel, progress=progress)
    if not return_depth:
        return image
    depth = render(scene, camera, width, height, background, ambient, "depth", path, cancel=cancel)
    covered = depth[..., 3] > 0
    return image, _read_only(np.where(covered, depth[..., 0], np.inf).astype(np.float32))
