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

Depth of field (step R5): a camera with an f-stop starts each path at a point of its aperture, aimed at the point of
the focal plane its pixel looks at (`lens.py`); the lens reads random dimensions 2 and 3 of the path key, the GPU twin the
same ones. The data passes stay pinhole.

Splats and smoke (step R4). Splats are Gaussian surfaces hit with their opacity along the ray, shaded like meshes
(`nodebased/ptsplats.py`); volumes are sampled by delta tracking, lit like surfaces, and emit fire
(`nodebased/ptvolume.py`). Both are CPU-only for now: `backend="auto"` falls back to this reference for a scene that
has them and `"gpu"` says why not. Particles are still refused.

Limits, stated: alpha below 0.5 does not block shadow rays and the coverage of an alpha surface is
stochastic; textures are read at their top mip; projections are ignored; emissive meshes are found by
BSDF sampling only (they are not sampled as lights); area lights are visible in reflections but not to
the camera ray itself; liquid roughness and thin sheets are not modelled.
"""
import math
import time
from dataclasses import dataclass, replace

import numpy as np

from . import envlight, lens, ptdenoise, ptsplats, ptvolume, raytrace, scene3d as s
from .liquid_render import fresnel, sigma_of

PI = math.pi
LIGHT_UNIT = PI
MAX_BOUNCES_LIMIT = 64
TILE = 16
MIN_ADAPTIVE_SAMPLES = 16
SAMPLING_MODES = ("fixed", "adaptive")
NOISE_FLOOR = 0.02         # added to the mean luminance before the noise estimate divides by it, so black does not read as infinitely noisy
LUMINANCE = (0.2126, 0.7152, 0.0722)
AOV_OUTPUTS = ("rgba", "diffuse", "specular", "emission", "albedo", "diffuse_indirect", "specular_indirect")
DATA_OUTPUTS = s.DATA_OUTPUTS
PATH_OUTPUTS = AOV_OUTPUTS + DATA_OUTPUTS + ("denoise",)
_PARTICLE_LIGHT_GAIN = 4.0 # emissive particle -> synthetic Point light intensity (R7 of 7 finish, `_particle_lights`)
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
    # Per-pixel adaptive sampling (plan "Rendering 6", step R1). "fixed" is everything above exactly as it always ran
    # (including the legacy tile retirement under `noise_threshold`). "adaptive" ignores `samples`: every pixel takes
    # `min_samples`, then `adaptive_pass_size` more per pass until its own noise estimate (`pixel_noise`) is under
    # `noise_threshold` or it has `max_samples`; a threshold of 0 never stops a pixel early.
    sampling: str = "fixed"
    min_samples: int = 16
    max_samples: int = 256
    adaptive_pass_size: int = 8

    @property
    def adaptive(self):
        return self.sampling == "adaptive"

    def clamped(self):
        sampling = self.sampling if self.sampling in SAMPLING_MODES else "fixed"
        max_samples = int(np.clip(self.max_samples, 1, 65536))
        min_samples = int(np.clip(self.min_samples, 1, max_samples))
        if sampling == "adaptive":
            min_samples = min(max(min_samples, 2), max_samples)   # one sample has no variance to estimate
        return replace(self, sampling=sampling, min_samples=min_samples, max_samples=max_samples,
                       adaptive_pass_size=int(np.clip(self.adaptive_pass_size, 1, 1024)),
                       samples=int(np.clip(self.samples, 1, 65536)),
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
                        seed=int(params.get("pt_seed", 1)),
                        sampling=str(params.get("sampling", "fixed")),
                        min_samples=int(params.get("min_samples", 16)),
                        max_samples=int(params.get("max_samples", 256)),
                        adaptive_pass_size=int(params.get("adaptive_pass_size", 8))).clamped()


@dataclass(frozen=True)
class DenoiseSettings:
    """Render3D's `denoise` output/pass knobs (step X2, nodebased/ptdenoise.py). `strength` 1 and every
    sensitivity 1 reproduce the filter exactly as it always ran; strength 0 is the raw beauty."""
    strength: float = 1.0
    color_sensitivity: float = 1.0
    normal_sensitivity: float = 1.0
    depth_sensitivity: float = 1.0
    iterations: int = ptdenoise.ITERATIONS
    temporal: bool = False

    def clamped(self):
        return replace(self, strength=float(np.clip(self.strength, 0.0, 1.0)),
                       color_sensitivity=max(float(self.color_sensitivity), 0.0),
                       normal_sensitivity=max(float(self.normal_sensitivity), 0.0),
                       depth_sensitivity=max(float(self.depth_sensitivity), 0.0),
                       iterations=int(np.clip(self.iterations, 0, 64)), temporal=bool(self.temporal))


def denoise_settings_from_params(params):
    """Render3D's denoiser knobs as a `DenoiseSettings`; a document saved before the knobs existed has none
    of them, and gets back the defaults that reproduce today's filter exactly."""
    return DenoiseSettings(strength=float(params.get("denoiser_strength", 1.0)),
                           color_sensitivity=float(params.get("denoise_color_sensitivity", 1.0)),
                           normal_sensitivity=float(params.get("denoise_normal_sensitivity", 1.0)),
                           depth_sensitivity=float(params.get("denoise_depth_sensitivity", 1.0)),
                           iterations=int(params.get("denoise_iterations", ptdenoise.ITERATIONS)),
                           temporal=bool(params.get("denoise_temporal", 0))).clamped()


# The temporal denoiser's history, one entry per `history_key` a caller supplies (Render3D's node key): the
# last frame's filtered image and the scene/camera it was rendered from, so the next call can reproject it
# through `motionblur.motion_vectors`. Process-local and unbounded by design (one entry per live Render3D
# node in "denoise temporal" mode is never large); a caller that wants a fresh start passes a new key.
_TEMPORAL_HISTORY = {}


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
    tangent: np.ndarray | None   # (T, 3) world/object-space, flat per triangle; None with no uvs
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
    tangent = _flat_tangents(v0, e1, e2, uvs) if uvs is not None else None
    tris = raytrace.TriangleSet(v0, e1, e2, np.ones(len(tri), np.float32))
    bvh = raytrace.Bvh.build(*tris.aabbs()) if len(tri) > _BRUTE_TRIANGLES else None
    lo = np.minimum(np.minimum(v0, v1), v2).min(0) if len(tri) else np.zeros(3)
    hi = np.maximum(np.maximum(v0, v1), v2).max(0) if len(tri) else np.zeros(3)
    return _Blas(tris, bvh, v0, e1, e2, n, smooth, uvs, tangent, lo, hi)


def _flat_tangents(v0, e1, e2, uvs):
    """One tangent per triangle (object space, not normalized) from its UV gradient, for normal mapping.
    Degenerate UVs (zero determinant) fall back to `e1`, matching a flat, untextured tangent frame."""
    duv1, duv2 = uvs[:, 1] - uvs[:, 0], uvs[:, 2] - uvs[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    safe = np.abs(det) > 1e-12
    r = np.where(safe, 1.0 / np.where(safe, det, 1.0), 0.0)
    tangent = (e1 * (duv2[:, 1] * r)[:, None]) - (e2 * (duv1[:, 1] * r)[:, None])
    return np.where(safe[:, None], tangent, e1)


@dataclass
class _Env:
    rgb: np.ndarray              # (H, W, 3) decimated map
    gain: np.ndarray             # intensity * tint
    local: object                # world direction -> map frame, `Environment._local`
    lum: np.ndarray              # (H, W) texel luminance
    marginal: np.ndarray         # (H,) cumulative row weight, ends at 1
    conditional: np.ndarray      # (H, W) cumulative within each row, each row ends at 1
    scale: float                 # W * H / (2 pi^2 * sum(lum * sin theta))
    visible_to_camera: bool = False


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
                marginal, conditional, w * h / (2 * PI * PI * total),
                visible_to_camera=bool(getattr(environment, "visible_to_camera", False)))


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
    visible_to_camera: bool = False


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
    mr_texture: list             # per shape: metallic-roughness top mip (G=rough, B=metal), linear, or None
    normal_texture: list         # per shape: tangent-space normal top mip, linear, or None
    normal_scale: np.ndarray     # (S,)
    occlusion_texture: list      # per shape: occlusion top mip (R channel), linear, or None
    occlusion_strength: np.ndarray  # (S,)
    emissive_texture: list       # per shape: emissive top mip, sRGB-decoded, or None
    emissive_color: np.ndarray   # (S, 3)
    area_lights: list
    point_lights: list
    envs: list
    ambient: float
    extent: float
    eps: float
    shadow_eps: float            # shadow rays start this far along the shading normal, like the ray-traced renderer's bias
    geometries: tuple = ()
    splats: object = None        # ptsplats.SplatLayer: splat `k` is shape `n_shapes + k` in every per-shape array above
    volumes: object = None       # ptvolume.VolumeLayer
    # Light linking (scene3d.light_reaches): per shape (splats last), the bit mask of the lights it excludes. Light
    # `i` is bit `i` in the order point lights, area lights, environments; None when no shape excludes any light.
    excl: object = None

    @property
    def shapes(self):
        return len(self.shape_blas)

    def mis(self, bit, pa, pb):
        """The power-heuristic weight of strategy `pa` against `pb` for light `bit`. A light that some shape excludes
        (light linking) is sampled by next-event estimation alone, whose shadow rays skip the excluding shapes: a
        BSDF ray that hits such a shape could not be told to see through it, so the two strategies would no longer
        estimate one integral. Weight 1 for the light sample, 0 for the BSDF hit (a perfect mirror still reads it)."""
        if self.linked_bits >> bit & 1:
            return np.ones_like(np.asarray(pa, np.float64))
        return _mis(pa, pb)

    @property
    def linked_bits(self):
        """The lights some shape excludes, as a bit mask (0 without light linking)."""
        if "_linked_bits" not in self.__dict__:
            bits = int(np.bitwise_or.reduce(self.excl)) if self.excl is not None else 0
            if self.volumes is not None and self.volumes.excl is not None:
                bits |= int(np.bitwise_or.reduce(self.volumes.excl))
            self.__dict__["_linked_bits"] = bits
        return self.__dict__["_linked_bits"]

    def hit_weight(self, bit, prev_delta, prev_pdf, pdf_light):
        """The weight of a BSDF ray's hit of light `bit` (see `mis`): 1 after a perfect mirror, else the MIS weight, 0
        for a light some shape excludes."""
        if self.linked_bits >> bit & 1:
            return np.where(prev_delta, 1.0, 0.0)
        return np.where(prev_delta, 1.0, _mis(prev_pdf, pdf_light))

    def reaches(self, shape, bit):
        """Per entry of `shape` (-1: no surface, so every light reaches), whether light `bit` lights that shape. A smoke
        vertex in volume `v` is shape `-3 - v`."""
        shape = np.asarray(shape)
        out = np.ones(len(shape), bool)
        if self.excl is not None:
            out &= (shape < 0) | (((self.excl[np.maximum(shape, 0)] >> bit) & 1) == 0)
        if self.volumes is not None and self.volumes.excl is not None:
            inside = shape <= -3
            index = np.clip(-3 - shape, 0, len(self.volumes.excl) - 1)
            out &= ~inside | (((self.volumes.excl[index] >> bit) & 1) == 0)
        return out

    def is_splat(self, shape):
        return np.asarray(shape) >= self.shapes


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
                      bool(light.two_sided) and light.kind != "Sphere",
                      visible_to_camera=bool(getattr(light, "visible_to_camera", False)))


def build_scene(scene, ambient=0.0, eye=None, volume=None):
    """Pack `scene` (instances kept as instances) for tracing; `eye` only feeds the splat normal smoothing and
    `volume` (`volumerender.VolumeSettings`) is the smoke knobs."""
    geometries = list(scene.geometries)
    blases, cache = [], {}

    def blas_index(geometry):
        key = (id(geometry.vertices), id(geometry.triangles), id(geometry.normals), id(geometry.uvs))
        if key not in cache:
            cache[key] = len(blases)
            blases.append(_blas_of(geometry))
        return cache[key]

    entries = []   # (geometry, matrix, tint or None, light link)
    for geometry in geometries:
        entries.append((geometry, geometry.world_matrix().astype(np.float64), None, geometry.light_link))
    for instance_set in scene.instances:
        if not len(instance_set) or not instance_set.sources:
            continue
        parent = instance_set.parent.astype(np.float64)
        bases = [source.world_matrix().astype(np.float64) for source in instance_set.sources]
        for i in range(len(instance_set)):
            variant = int(instance_set.variant[i])
            source = instance_set.sources[variant]
            tint = None if instance_set.colors is None else instance_set.colors[i]
            entries.append((source, parent @ instance_set.matrices[i] @ bases[variant], tint, instance_set.light_link))
    n = len(entries)
    shape_blas = np.zeros(n, np.int32)
    matrix, inverse = np.zeros((n, 4, 4)), np.zeros((n, 4, 4))
    world_lo, world_hi = np.zeros((n, 3)), np.zeros((n, 3))
    base, alpha, sigma = np.zeros((n, 3)), np.ones(n), np.zeros((n, 3))
    kind, metallic, roughness, f0 = np.zeros(n, np.int32), np.zeros(n), np.zeros(n), np.zeros(n)
    emission, ior, reflection = np.zeros(n), np.ones(n), np.ones(n)
    normal_scale, occlusion_strength = np.ones(n), np.ones(n)
    emissive_color = np.zeros((n, 3))
    textures, mr_textures, normal_textures, occlusion_textures, emissive_textures = [], [], [], [], []
    for i, (geometry, m, tint, _link) in enumerate(entries):
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
        has_uvs = geometry.uvs is not None
        textures.append(s._mip_chain(geometry.texture)[0].astype(np.float64)
                        if geometry.texture is not None and has_uvs else None)
        mr_textures.append(s._mip_chain(geometry.metallic_roughness_texture)[0].astype(np.float64)
                           if geometry.metallic_roughness_texture is not None and has_uvs else None)
        normal_textures.append(s._mip_chain(geometry.normal_texture)[0].astype(np.float64)
                               if geometry.normal_texture is not None and has_uvs else None)
        occlusion_textures.append(s._mip_chain(geometry.occlusion_texture)[0].astype(np.float64)
                                  if geometry.occlusion_texture is not None and has_uvs else None)
        emissive_textures.append(s._mip_chain(geometry.emissive_texture)[0].astype(np.float64)
                                 if geometry.emissive_texture is not None and has_uvs else None)
        normal_scale[i] = float(geometry.normal_scale)
        occlusion_strength[i] = float(np.clip(geometry.occlusion_strength, 0, 1))
        emissive_color[i] = np.asarray(geometry.emissive_color, np.float64)
    area_lights, point_lights = [], []
    area_sources = []
    for light in scene.lights:
        if light.intensity <= 0:
            continue
        if light.kind in s._AREA:
            area_lights.append(_area_light(light))
            area_sources.append(light)
        else:
            position, direction = light.world()
            colour = np.asarray(light.color, np.float64) * float(light.intensity) * LIGHT_UNIT
            point_lights.append(_PointLight(light, light.kind, position.astype(np.float64),
                                            direction.astype(np.float64), colour))
    # the lights in bit order (point, area, environment) and each shape's mask of the ones it excludes
    ordered_lights = [p.light for p in point_lights] + area_sources + list(scene.environments)
    link_masks = {}

    def mask_of(link):
        if link[0] == "all":
            return 0
        if link not in link_masks:
            if len(ordered_lights) > 62:
                raise ValueError("light linking takes at most 62 lights and environments together")
            link_masks[link] = sum(1 << k for k, light in enumerate(ordered_lights) if not s.light_reaches(link, light))
        return link_masks[link]

    excl = np.array([mask_of(entry[3]) for entry in entries], np.int64)
    layer = ptsplats.build(scene, eye) if getattr(scene, "splats", ()) else None
    smoke = ptvolume.build(scene, volume) if getattr(scene, "volumes", ()) else None
    if smoke is not None:
        volume_excl = np.array([mask_of(getattr(v, "light_link", s.LIGHT_LINK_ALL)) for v in scene.volumes], np.int64)
        smoke.excl = volume_excl if volume_excl.any() else None
    all_lo = world_lo.min(0) if n else np.full(3, np.inf)
    all_hi = world_hi.max(0) if n else np.full(3, -np.inf)
    for extra in (layer, smoke):
        if extra is not None:
            all_lo, all_hi = np.minimum(all_lo, extra.lo), np.maximum(all_hi, extra.hi)
    extent = max(float(np.ptp(np.stack((all_lo, all_hi)), axis=0).max())
                 if (n or layer is not None or smoke is not None) else 1.0, 1e-6)
    object_id = np.arange(1, n + 1, dtype=np.int32)
    if layer is not None:
        # every splat is one more shape in the per-shape material arrays (one object id per splat instance)
        m = len(layer)
        pbr = layer.pbr
        base = np.concatenate((base, layer.albedo))
        alpha = np.concatenate((alpha, np.ones(m)))
        kind = np.concatenate((kind, np.where(pbr, 1, 0).astype(np.int32)))
        metallic = np.concatenate((metallic, layer.metallic))
        roughness = np.concatenate((roughness, layer.roughness))
        f0 = np.concatenate((f0, np.where(pbr, 0.04, 0.0)))
        emission = np.concatenate((emission, np.zeros(m)))
        ior, reflection = np.concatenate((ior, np.ones(m))), np.concatenate((reflection, np.ones(m)))
        sigma = np.concatenate((sigma, np.zeros((m, 3))))
        object_id = np.concatenate((object_id, (n + 1 + layer.instance).astype(np.int32)))
        normal_scale = np.concatenate((normal_scale, np.ones(m)))
        occlusion_strength = np.concatenate((occlusion_strength, np.ones(m)))
        emissive_color = np.concatenate((emissive_color, np.zeros((m, 3))))
        mr_textures += [None] * m
        normal_textures += [None] * m
        occlusion_textures += [None] * m
        emissive_textures += [None] * m
        splat_excl = np.array([mask_of(item.light_link) for item in scene.splats], np.int64)[layer.instance]
        layer.excl = splat_excl if splat_excl.any() else None
        excl = np.concatenate((excl, splat_excl))
    return PathScene(blases, shape_blas, matrix, inverse, world_lo, world_hi,
                     object_id, base, alpha, kind, metallic, roughness, f0, emission,
                     ior, reflection, sigma, textures, mr_textures, normal_textures, normal_scale,
                     occlusion_textures, occlusion_strength, emissive_textures, emissive_color,
                     area_lights, point_lights,
                     [_env_of(e) for e in scene.environments], float(ambient), extent,
                     1e-4 * max(1.0, extent), s.SHADOW_BIAS_DEFAULT * max(1.0, extent), tuple(geometries),
                     splats=layer, volumes=smoke, excl=excl if excl.any() else None)


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


_SPLAT_DIM = 56             # the random dimension a splat hit's coin flips hash from (inside the vertex stride, unused otherwise)


def closest(ps, o, d, tmin, tmax, cancel=None, stoch=None, skip=None):
    """(t, shape, prim, u, v) of the nearest hit along each ray; shape -1 for a miss.

    Splats are hit stochastically when `stoch` is `(keys, dim)`: each candidate splat is present with its alpha,
    decided by a hash of the path key and the splat, and the nearest present one wins (front-to-back compositing
    in expectation). Without `stoch` the data passes' rule applies: the first splat where the accumulated
    opacity reaches one half. `skip` is `(splat, tmin, plane normals, plane thickness)` per ray, the surface
    the ray just left, so a splat sheet does not hit itself. A splat hit is shape `ps.shapes + splat`."""
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
    if ps.splats is not None and n:
        near = np.broadcast_to(np.asarray(tmin, np.float64), (n,))
        exclude = plane = None
        if skip is not None:
            exclude, near, plane = skip[0], np.maximum(near, skip[1]), (skip[2], skip[3])
        if stoch is not None:
            keys, dim = stoch

            def accept(ray, splat, alpha):
                return rand(pcg(keys[ray] + splat.astype(np.uint32)), dim) < alpha
            t, splat = ptsplats.nearest_accepted(ps.splats, o, d, near, best, accept, exclude, plane, cancel)
        else:
            t, splat = ptsplats.first_opaque(ps.splats, o, d, near, best, cancel=cancel)
        won = (splat >= 0) & (t < best)
        best[won], shape[won], prim[won] = t[won], ps.shapes + splat[won], -1
        us[won] = vs[won] = 0.0
    return best, shape, prim, us, vs


def occluded(ps, o, d, tmax, cancel=None, bit=None):
    """True where something opaque lies along (o, d) before `tmax`. Shapes with alpha < 0.5 never block. `bit` is
    the light the shadow ray goes to: a shape that excludes that light (light linking) casts no shadow from it."""
    n = len(o)
    blocked = np.zeros(n, bool)
    tmax = np.broadcast_to(np.asarray(tmax, np.float64), (n,))
    for i in range(ps.shapes):
        if ps.alpha[i] < 0.5:
            continue
        if bit is not None and ps.excl is not None and (int(ps.excl[i]) >> bit) & 1:
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
    """World-space shading normal `ns`, geometric normal `ng` (same side as `ns`), uv, base colour and the
    PBR texture maps (materials 3): `metallic_ovr`/`roughness_ovr` (-1 where the shape has no
    metallic-roughness texture, else the sampled value), `occlusion` (1 where there is no map) and
    `emissive` (additive, zero where there is neither an `emissive_color` nor a texture) of hits."""
    n = len(shape)
    ns, ng, uv = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 2))
    tangent = np.zeros((n, 3))
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
        if blas.tangent is not None:
            tangent[take] = np.einsum("nji,nj->ni", inv3, blas.tangent[p])
    ns, ng = _unit(ns), _unit(ng)
    ng = np.where((np.einsum("ij,ij->i", ng, ns) < 0)[:, None], -ng, ng)
    metallic_ovr, roughness_ovr = np.full(n, -1.0), np.full(n, -1.0)
    occlusion, emissive = np.ones(n), np.zeros((n, 3))
    for i in np.unique(shape):
        take = np.flatnonzero(shape == i)
        if ps.texture[i] is not None:
            texel = s._sample(ps.texture[i], uv[take, 0], uv[take, 1])
            a = np.maximum(texel[:, 3:4], 1e-6)
            base[take] = ps.base[i] * (texel[:, :3] / a)
            alpha[take] = ps.alpha[i] * texel[:, 3]
        if ps.mr_texture[i] is not None:
            texel = s._sample(ps.mr_texture[i], uv[take, 0], uv[take, 1])
            roughness_ovr[take] = texel[:, 1]
            metallic_ovr[take] = texel[:, 2]
        if ps.normal_texture[i] is not None:
            texel = s._sample(ps.normal_texture[i], uv[take, 0], uv[take, 1])
            local = (texel[:, :3] * 2.0 - 1.0) * np.array((ps.normal_scale[i], ps.normal_scale[i], 1.0))
            t = _unit(tangent[take] - ns[take] * _dot(tangent[take], ns[take])[:, None])
            degenerate = np.linalg.norm(tangent[take], axis=1) < 1e-12
            t = np.where(degenerate[:, None], _frame(ns[take])[0], t)
            b = np.cross(ns[take], t)
            ns[take] = _unit(t * local[:, 0:1] + b * local[:, 1:2] + ns[take] * local[:, 2:3])
        if ps.occlusion_texture[i] is not None:
            texel = s._sample(ps.occlusion_texture[i], uv[take, 0], uv[take, 1])
            occlusion[take] = 1.0 - ps.occlusion_strength[i] * (1.0 - texel[:, 0])
        if ps.emissive_texture[i] is not None:
            texel = s._sample(ps.emissive_texture[i], uv[take, 0], uv[take, 1])
            emissive[take] = np.asarray(ps.emissive_color[i]) * texel[:, :3]
        elif np.any(ps.emissive_color[i] != 0):
            emissive[take] = np.asarray(ps.emissive_color[i])
    return ns, ng, uv, base, alpha, metallic_ovr, roughness_ovr, occlusion, emissive


def _surface_any(ps, shape, prim, u, v, wd):
    """`_surface` for mesh hits and `ptsplats.surface` for splat hits:
    `(ns, ng, uv, base, alpha, emission (N,3), metallic_ovr, roughness_ovr, occlusion)`."""
    n = len(shape)
    ns, ng, uv = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 2))
    base, alpha, emit = np.zeros((n, 3)), np.ones(n), np.zeros((n, 3))
    metallic_ovr, roughness_ovr, occlusion = np.full(n, -1.0), np.full(n, -1.0), np.ones(n)
    splat = ps.is_splat(shape)
    mesh = np.flatnonzero(~splat)
    if len(mesh):
        (m_ns, m_ng, m_uv, m_base, m_alpha, m_metal, m_rough, m_occ,
         m_emissive) = _surface(ps, shape[mesh], prim[mesh], u[mesh], v[mesh])
        ns[mesh], ng[mesh], uv[mesh], base[mesh], alpha[mesh] = m_ns, m_ng, m_uv, m_base, m_alpha
        metallic_ovr[mesh], roughness_ovr[mesh], occlusion[mesh] = m_metal, m_rough, m_occ
        emit[mesh] = m_base * ps.emission[shape[mesh]][:, None] + m_emissive
    sp = np.flatnonzero(splat)
    if len(sp):
        index = shape[sp] - ps.shapes
        s_ns, s_base, s_emit = ptsplats.surface(ps.splats, index, -wd[sp], wd[sp])
        ns[sp], ng[sp], base[sp], emit[sp] = s_ns, s_ns, s_base, s_emit
    return ns, ng, uv, base, alpha, emit, metallic_ovr, roughness_ovr, occlusion


def _relight_of(ps, shape):
    """The splat instances' `relight` mix at hits `shape` (1 for a mesh)."""
    if ps.splats is None:
        return np.ones(len(shape))
    splat = ps.is_splat(shape)
    return np.where(splat, ps.splats.relight[np.maximum(shape - ps.shapes, 0)], 1.0)


def _splat_light(ps, origin, wi, dist, shape, ns, cancel=None, bit=None):
    """Transmittance of everything soft between shadow-ray origins and a light `dist` away: the splat casters
    and the smoke volumes. `shape` (-1 for a point in a medium) says which surface the ray leaves."""
    n = len(origin)
    out = np.ones(n)
    if not n:
        return out
    if ps.splats is not None:
        layer = ps.splats
        on_splat = ps.is_splat(shape)
        index = np.where(on_splat, shape - ps.shapes, 0)
        tmin = np.where(on_splat, ptsplats.START_SCALE * layer.scale_max[index], 0.01 * ps.eps)
        exclude = np.where(on_splat, index, -1)
        thickness = np.where(on_splat, layer.scale_min[index], -1e30)
        out = ptsplats.transmittance(layer, origin, wi, tmin, dist, exclude, (ns, thickness), cancel, bit)
    if ps.volumes is not None:
        out = out * ptvolume.transmittance(ps.volumes, origin, wi, dist, bit)
    return out


# --- BSDF ---------------------------------------------------------------------------------------------------

def _alpha_of(roughness):
    return np.maximum(roughness, 0.05) ** 2


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _lobes(ps, shape, base, nv, metallic, roughness, f0d):
    """Per-hit lobe constants: diffuse colour (N,3), specular F0 (N,3), compensation (N,3), spec albedo (N,3), delta mask.

    `metallic`/`roughness`/`f0d` are the per-hit values (materials 3): a shape's own scalar knob,
    overridden per texel where its metallic-roughness texture is present (`_surface`)."""
    m = metallic[:, None]
    r = roughness
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


def bsdf_eval(ps, shape, base, n, v, wi, metallic, roughness, f0d):
    """(diffuse*cos (N,3), specular*cos (N,3), pdf of sampling `wi` (N,)) at hits.

    `specular` is the GGX response of `splatshade._cook_torrance` (divided by pi: it carries the pi of the
    codebase's lights) with the multiple-scattering compensation of the environment split sum.
    `metallic`/`roughness`/`f0d` are the per-hit values `_surface_event` derives (materials 3).
    """
    from .splatshade import _cook_torrance
    nv = np.maximum(_dot(n, v), 1e-4)
    diffuse, f0, k, spec_albedo, has_spec = _lobes(ps, shape, base, nv, metallic, roughness, f0d)
    r = roughness
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
    prev_shape = np.full(n, -1, np.int64)      # the shape of the last surface scatter (-1: none, or a liquid or smoke vertex)
    smoke_run = np.zeros(n, np.int64)          # smoke vertices in a row so far (`volume_multi_scatter` acts at the second)
    capped = np.zeros(n, bool)                 # the last scatter reached its kind's cap: the next ray sees emitters only
    prev_pdf = np.zeros(n)
    medium = np.full(n, -1, np.int64)
    vertex_depth = np.zeros(n, np.int32)       # scatters so far: the depth of the vertex the ray reaches next
    skip = (np.full(n, -1, np.int64), np.zeros(n), np.zeros((n, 3)), np.zeros(n))   # the splat a ray just left
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
        dim = _DIM_BASE + turn * _DIM_STRIDE
        t, shape, prim, u, v = closest(ps, ro, rd, tmin[rows], tmax[rows], cancel, stoch=(keys[rows], dim + _SPLAT_DIM),
                                       skip=tuple(a[rows] for a in skip) if ps.splats is not None else None)
        tmin[:], tmax[:] = 0.0, np.inf
        vd = vertex_depth[rows]
        # emitters: the analytic area lights, seen by every ray but the camera's own
        t_light = np.full(len(rows), np.inf)
        light_id = np.full(len(rows), -1)
        for li, light in enumerate(ps.area_lights):
            tl = _area_hits(light, ro, rd, t)
            better = tl < t_light
            t_light[better], light_id[better] = tl[better], li
        evented = np.zeros(len(rows), bool)
        if ps.volumes is not None:
            # smoke: the next real collision along each ray, and the fire the ray sees on the way to it
            t_end = np.minimum(t, np.where(vd >= 1, t_light, np.inf))
            t_event, glow, in_volume = ptvolume.free_flight(ps.volumes, ro, rd, t_end, keys[rows], turn, cancel)
            evented = t_event < t_end
            smoke_run[rows[~evented]] = 0
            if glow.any():
                g_idx = np.flatnonzero(glow.max(axis=1) > 0)
                gathered = throughput[rows[g_idx]] * glow[g_idx]
                # fire seen from a smoke vertex is the fire lighting the smoke: `volume_fire_light` is its gain
                gathered = gathered * np.where(prev_shape[rows[g_idx]] <= -3, ps.volumes.settings.fire_light, 1.0)[:, None]
                seen_first = vd[g_idx] == 0
                acc.emission[rows[g_idx][seen_first]] += gathered[seen_first]
                if (~seen_first).any():
                    _add_class(acc, rows[g_idx][~seen_first], cls[rows[g_idx][~seen_first]], vd[g_idx][~seen_first],
                               gathered[~seen_first])
        # a camera ray (vd == 0) only registers a light hit when that light's `visible_to_camera` is on
        visible_flags = np.array([bool(light.visible_to_camera) for light in ps.area_lights], bool)
        camera_visible = np.zeros(len(rows), bool)
        has_light = light_id >= 0
        if has_light.any():
            camera_visible[has_light] = visible_flags[light_id[has_light]] if len(visible_flags) else False
        lit = has_light & ((vd >= 1) | camera_visible) & ~evented
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
                weight = ps.hit_weight(len(ps.point_lights) + li, prev_delta[rows[idx]], prev_pdf[rows[idx]], pdf_l)
                weight = weight * ((vd[idx] == 0) | ps.reaches(prev_shape[rows[idx]], len(ps.point_lights) + li))
                contrib[pick] = light.radiance * weight[:, None]
            value = throughput[rows[sel]] * contrib
            first_seen = vd[sel] == 0
            if first_seen.any():
                fs = sel[first_seen]
                fr = rows[fs]
                acc.emission[fr] += value[first_seen]
                first["alpha"][fr] = 1.0
                first["albedo"][fr] = contrib[first_seen]
                first["shape"][fr], first["t"][fr] = -2, t_light[fs]
                first["pos"][fr] = ro[fs] + rd[fs] * t_light[fs][:, None]
            if (~first_seen).any():
                _add_class(acc, rows[sel[~first_seen]], cls[rows[sel[~first_seen]]], vd[sel[~first_seen]],
                          value[~first_seen])
            alive[rows[sel]] = False
        gone = lit | evented
        if evented.any():
            e = np.flatnonzero(evented)
            fresh = e[vd[e] == 0]
            if len(fresh):
                fr = rows[fresh]
                first["alpha"][fr], first["albedo"][fr] = 1.0, ps.volumes.color
                first["t"][fr], first["pos"][fr] = t_event[fresh], ro[fresh] + rd[fresh] * t_event[fresh][:, None]
            _medium_event(ps, keys, rows[e], (ro[e] + rd[e] * t_event[e][:, None]), rd[e], vd[e], o, d, throughput,
                          alive, counts, cls, prev_delta, prev_pdf, capped, bounces, vertex_depth, acc, settings, dim,
                          cancel, skip, in_volume[e], smoke_run[rows[e]])
            smoke_run[rows[e]] += 1
            prev_shape[rows[e]] = -3 - in_volume[e]     # a smoke vertex: `PathScene.reaches` reads its volume's link
        # rays that leave the scene see the ambient sky and the environments; a camera ray sees an
        # environment too when that environment's `visible_to_camera` is on, replacing the flat background
        miss = (shape < 0) & ~gone
        if miss.any():
            m = np.flatnonzero(miss)
            r = rows[m]
            sky = np.full((len(m), 3), ps.ambient)
            for ei, env in enumerate(ps.envs):
                radiance = env_radiance(env, rd[m])
                weight = ps.hit_weight(len(ps.point_lights) + len(ps.area_lights) + ei, prev_delta[r], prev_pdf[r],
                                       env_pdf(env, rd[m]))
                weight = weight * ps.reaches(prev_shape[r], len(ps.point_lights) + len(ps.area_lights) + ei)
                sky += radiance * weight[:, None]
            deep = vd[m] >= 1
            if deep.any():
                _add_class(acc, r[deep], cls[r[deep]], vd[m][deep], throughput[r[deep]] * sky[deep])
            first_seen = ~deep
            if first_seen.any():
                visible_envs = [env for env in ps.envs if env.visible_to_camera]
                if visible_envs:
                    fs = m[first_seen]
                    fr = r[first_seen]
                    background_radiance = np.zeros((len(fs), 3))
                    for env in visible_envs:
                        background_radiance += env_radiance(env, rd[fs])
                    acc.emission[fr] += background_radiance
                    first["alpha"][fr] = 1.0
                    first["albedo"][fr] = background_radiance
            alive[r] = False
        hit = (shape >= 0) & ~gone
        if not hit.any():
            continue
        h = np.flatnonzero(hit)
        r = rows[h]
        sh = shape[h]
        ns, ng, uv, base, alpha, emit, metallic_ovr, roughness_ovr, occlusion = _surface_any(
            ps, sh, prim[h], u[h], v[h], rd[h])
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
            skip[0][r[through]], skip[1][r[through]], skip[3][r[through]] = -1, 0.0, -1e30
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
        r, sh, ns, ng, base, pos, wo, wd, emit, metallic_ovr, roughness_ovr, occlusion = (
            x[keep] for x in (r, sh, ns, ng, base, pos, wo, wd, emit, metallic_ovr, roughness_ovr, occlusion))
        vd = vertex_depth[r]
        has_emit = emit.max(axis=1) > 0
        if has_emit.any():
            e = np.flatnonzero(has_emit)
            gathered = throughput[r[e]] * emit[e]
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
        r, sh, ns, ng, base, pos, wo, wd, vd, metallic_ovr, roughness_ovr, occlusion = (
            x[sc] for x in (r, sh, ns, ng, base, pos, wo, wd, vd, metallic_ovr, roughness_ovr, occlusion))
        liquid = ps.kind[sh] == 2
        wet = np.flatnonzero(liquid)
        if len(wet):
            _liquid_event(ps, keys, r[wet], sh[wet], ns[wet], wd[wet], pos[wet], o, d, medium, alive, counts,
                          cls, prev_delta, capped, bounces, vertex_depth, settings, dim)
            prev_shape[r[wet]] = -1
            skip[0][r[wet]], skip[1][r[wet]], skip[3][r[wet]] = -1, 0.0, -1e30
        dry = np.flatnonzero(~liquid)
        if len(dry):
            _surface_event(ps, keys, r[dry], sh[dry], ns[dry], ng[dry], base[dry], pos[dry], wo[dry], vd[dry],
                           metallic_ovr[dry], roughness_ovr[dry], occlusion[dry],
                           o, d, throughput, alive, counts, cls, prev_delta, prev_pdf, capped, bounces, vertex_depth,
                           acc, settings, dim, cancel, skip, prev_shape)
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


_VOL_ABSORB_DIM = _BSDF_DIM + 4


def _volume_reaches(ps, owner, bit):
    """Per smoke vertex (`owner` is its volume), whether light `bit` lights it (light linking)."""
    return ps.reaches(-3 - owner, bit)


def _medium_event(ps, keys, r, pos, wd, vd, o, d, throughput, alive, counts, cls, prev_delta, prev_pdf, capped,
                  bounces, vertex_depth, acc, settings, dim, cancel, skip, owner, run):
    """A real collision at points `pos` of paths `r` (arriving along `wd`) in volumes `owner`: absorb, or scatter with
    light sampling and a phase-sampled continuation. `run` is how many smoke vertices in a row each path has had before
    this one: at the second (run 1) the path's throughput gains `1 + volume_multi_scatter`, once, so everything it gathers
    after two scatterings in smoke (this vertex's light, the sky, a light, fire) is brightened and 0 changes nothing."""
    layer = ps.volumes
    n = len(r)
    key = keys[r]
    g = float(layer.settings.anisotropy)
    scatter = rand(key, dim + _VOL_ABSORB_DIM) < layer.scatter_fraction
    can = scatter & (bounces[r] < settings.max_bounces) & ~capped[r]
    alive[r[~can]] = False
    keep = np.flatnonzero(can)
    if not len(keep):
        return
    r, pos, wd, vd, key, owner = r[keep], pos[keep], wd[keep], vd[keep], key[keep], owner[keep]
    run = run[keep]
    n = len(r)
    none = np.full(n, -1)
    zero = np.zeros((n, 3))
    direct = np.zeros((n, 3))
    slot = 0
    for bit, light in enumerate(ps.point_lights):
        wi, dist, irr = _point_wi(light, pos)
        front = (irr.max(axis=1) > 0) & _volume_reaches(ps, owner, bit)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        hidden = occluded(ps, pos[idx], wi[idx], dist[idx], cancel, bit)
        idx = idx[~hidden]
        if not len(idx):
            continue
        seen = _splat_light(ps, pos[idx], wi[idx], dist[idx], none[idx], zero[idx], cancel, bit)
        direct[idx] += ptvolume.phase(g, _dot(wi[idx], wd[idx]))[:, None] * irr[idx] * seen[:, None]
    for li, light in enumerate(ps.area_lights):
        bit = len(ps.point_lights) + li
        u1, u2 = rand(key, dim + 2 + slot), rand(key, dim + 3 + slot)
        slot += 2
        q, nq = _area_sample(light, u1, u2)
        to = q - pos
        dist = np.linalg.norm(to, axis=1)
        wi = to / np.maximum(dist, 1e-12)[:, None]
        cos_l = _dot(-wi, nq)
        cos_l = np.abs(cos_l) if light.two_sided else np.maximum(cos_l, 0)
        front = (cos_l > 1e-9) & (dist > 1e-9) & _volume_reaches(ps, owner, bit)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        hidden = occluded(ps, pos[idx], wi[idx], dist[idx] * (1 - 1e-4), cancel, bit)
        idx = idx[~hidden]
        if not len(idx):
            continue
        pdf_l = _area_pdf(light, dist[idx], cos_l[idx])
        f = ptvolume.phase(g, _dot(wi[idx], wd[idx]))
        weight = ps.mis(bit, pdf_l, f) / pdf_l
        weight = weight * _splat_light(ps, pos[idx], wi[idx], dist[idx] * (1 - 1e-4), none[idx], zero[idx], cancel, bit)
        direct[idx] += (f * weight)[:, None] * light.radiance
    for ei, env in enumerate(ps.envs):
        bit = len(ps.point_lights) + len(ps.area_lights) + ei
        u1, u2 = rand(key, dim + 2 + slot), rand(key, dim + 3 + slot)
        slot += 2
        wi, pdf_l, radiance = env_sample(env, u1, u2)
        front = (pdf_l > 0) & _volume_reaches(ps, owner, bit)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        hidden = occluded(ps, pos[idx], wi[idx], np.inf, cancel, bit)
        idx = idx[~hidden]
        if not len(idx):
            continue
        f = ptvolume.phase(g, _dot(wi[idx], wd[idx]))
        weight = ps.mis(bit, pdf_l[idx], f) / pdf_l[idx]
        weight = weight * _splat_light(ps, pos[idx], wi[idx], np.full(len(idx), np.inf), none[idx], zero[idx], cancel, bit)
        direct[idx] += (f * weight)[:, None] * radiance[idx]
    tint = layer.color
    tp = throughput[r] * np.where(run == 1, 1.0 + layer.settings.multi_scatter, 1.0)[:, None]
    gathered = tp * tint * direct
    first_vertex = vd == 0
    if first_vertex.any():
        f_ = np.flatnonzero(first_vertex)
        acc.diffuse[r[f_]] += gathered[f_]
    if (~first_vertex).any():
        f_ = np.flatnonzero(~first_vertex)
        _add_class(acc, r[f_], cls[r[f_]], vd[f_] + 1, gathered[f_])
    # the phase function's own sampling continues the path (its weight is exactly 1)
    u1, u2 = rand(key, dim + _BSDF_DIM + 1), rand(key, dim + _BSDF_DIM + 2)
    new_dir, pdf = ptvolume.sample_phase(g, wd, u1, u2)
    ok = counts[r, 0] < settings.diffuse_bounces
    alive[r[~ok]] = False
    go = np.flatnonzero(ok)
    if not len(go):
        return
    gg = r[go]
    throughput[gg] = tp[go] * tint
    o[gg], d[gg] = pos[go], new_dir[go]
    counts[gg, 0] += 1
    capped[gg] = counts[gg, 0] >= settings.diffuse_bounces
    fresh = cls[gg] < 0
    cls[gg[fresh]] = 0
    prev_delta[gg] = False
    prev_pdf[gg] = pdf[go]
    bounces[gg] += 1
    vertex_depth[gg] += 1
    skip[0][gg], skip[1][gg], skip[3][gg] = -1, 0.0, -1e30
    roulette = go[bounces[gg] >= 3]
    if len(roulette):
        rg = r[roulette]
        q = np.clip(throughput[rg].max(axis=1), 0.05, 0.95)
        survive = rand(key[roulette], dim + _BSDF_DIM + 3) < q
        throughput[rg[survive]] /= q[survive][:, None]
        alive[rg[~survive]] = False


def _surface_event(ps, keys, r, sh, ns, ng, base, pos, wo, vd, metallic_ovr, roughness_ovr, occlusion,
                   o, d, throughput, alive, counts, cls,
                   prev_delta, prev_pdf, capped, bounces, vertex_depth, acc, settings, dim, cancel, skip,
                   prev_shape):
    """Light sampling and BSDF sampling at the surface vertices `r` (path indices).

    `metallic_ovr`/`roughness_ovr` (-1 where the hit has no metallic-roughness texture) and `occlusion`
    (1 where the hit has no occlusion map) come from `_surface` (materials 3): the per-hit metallic and
    roughness fall back to the shape's own scalar knob, and occlusion attenuates the diffuse response."""
    n = len(r)
    nv = np.maximum(_dot(ns, wo), 1e-4)
    metallic_hit = np.where(metallic_ovr >= 0, metallic_ovr, ps.metallic[sh])
    roughness_hit = np.where(roughness_ovr >= 0, roughness_ovr, ps.roughness[sh])
    f0d_hit = ps.f0[sh]
    diffuse, f0, k, spec_albedo, has_spec = _lobes(ps, sh, base, nv, metallic_hit, roughness_hit, f0d_hit)
    diffuse = diffuse * occlusion[:, None]
    rough = roughness_hit
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

    for bit, light in enumerate(ps.point_lights):
        wi, dist, irr = _point_wi(light, pos)
        nl = _dot(ns, wi)
        front = (nl > 0) & (irr.max(axis=1) > 0) & ps.reaches(sh, bit)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        po = pos[idx] + ns[idx] * ps.shadow_eps
        hidden = occluded(ps, po, wi[idx], dist[idx] - ps.shadow_eps, cancel, bit)
        idx = idx[~hidden]
        if not len(idx):
            continue
        f_diff, f_spec, _ = bsdf_eval(ps, sh[idx], base[idx], ns[idx], wo[idx], wi[idx],
                              metallic_hit[idx], roughness_hit[idx], f0d_hit[idx])
        f_diff = f_diff * occlusion[idx, None]
        seen = _splat_light(ps, pos[idx] + ns[idx] * ps.shadow_eps, wi[idx], dist[idx], sh[idx], ns[idx], cancel, bit)
        direct_d[idx] += f_diff * (irr[idx] * seen[:, None])
        direct_s[idx] += f_spec * (irr[idx] * seen[:, None])
    for li, light in enumerate(ps.area_lights):
        bit = len(ps.point_lights) + li
        u1, u2 = rand(key, dim + 2 + slot), rand(key, dim + 3 + slot)
        slot += 2
        q, nq = _area_sample(light, u1, u2)
        to = q - pos
        dist = np.linalg.norm(to, axis=1)
        wi = to / np.maximum(dist, 1e-12)[:, None]
        cos_l = _dot(-wi, nq)
        cos_l = np.abs(cos_l) if light.two_sided else np.maximum(cos_l, 0)
        nl = _dot(ns, wi)
        front = (nl > 0) & (cos_l > 1e-9) & (dist > 1e-9) & ps.reaches(sh, bit)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        po = pos[idx] + ns[idx] * ps.shadow_eps
        hidden = occluded(ps, po, wi[idx], dist[idx] * (1 - 1e-4) - ps.shadow_eps, cancel, bit)
        idx = idx[~hidden]
        if not len(idx):
            continue
        pdf_l = _area_pdf(light, dist[idx], cos_l[idx])
        f_diff, f_spec, pdf_b = bsdf_eval(ps, sh[idx], base[idx], ns[idx], wo[idx], wi[idx],
                                  metallic_hit[idx], roughness_hit[idx], f0d_hit[idx])
        f_diff = f_diff * occlusion[idx, None]
        weight = ps.mis(bit, pdf_l, pdf_b) / pdf_l
        weight = weight * _splat_light(ps, pos[idx] + ns[idx] * ps.shadow_eps, wi[idx], dist[idx] * (1 - 1e-4), sh[idx],
                                       ns[idx], cancel, bit)
        direct_d[idx] += f_diff * light.radiance * weight[:, None]
        direct_s[idx] += f_spec * light.radiance * weight[:, None]
    for ei, env in enumerate(ps.envs):
        bit = len(ps.point_lights) + len(ps.area_lights) + ei
        u1, u2 = rand(key, dim + 2 + slot), rand(key, dim + 3 + slot)
        slot += 2
        wi, pdf_l, radiance = env_sample(env, u1, u2)
        nl = _dot(ns, wi)
        front = (nl > 0) & (pdf_l > 0) & ps.reaches(sh, bit)
        if not front.any():
            continue
        idx = np.flatnonzero(front)
        po = pos[idx] + ns[idx] * ps.shadow_eps
        hidden = occluded(ps, po, wi[idx], np.inf, cancel, bit)
        idx = idx[~hidden]
        if not len(idx):
            continue
        f_diff, f_spec, pdf_b = bsdf_eval(ps, sh[idx], base[idx], ns[idx], wo[idx], wi[idx],
                                  metallic_hit[idx], roughness_hit[idx], f0d_hit[idx])
        f_diff = f_diff * occlusion[idx, None]
        weight = ps.mis(bit, pdf_l[idx], pdf_b) / pdf_l[idx]
        weight = weight * _splat_light(ps, pos[idx] + ns[idx] * ps.shadow_eps, wi[idx], np.full(len(idx), np.inf),
                                       sh[idx], ns[idx], cancel, bit)
        direct_d[idx] += f_diff * radiance[idx] * weight[:, None]
        direct_s[idx] += f_spec * radiance[idx] * weight[:, None]
    mix = _relight_of(ps, sh)
    if ps.splats is not None:
        direct_d, direct_s = direct_d * mix[:, None], direct_s * mix[:, None]
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
    f_diff, f_spec, pdf = bsdf_eval(ps, sh, base, ns, wo, wi, metallic_hit, roughness_hit, f0d_hit)
    f_diff = f_diff * occlusion[:, None]
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
    new_tp = tp * weight * mix[:, None]
    ok &= new_tp.max(axis=1) > 0
    alive[r[~ok]] = False
    go = np.flatnonzero(ok)
    g = r[go]
    throughput[g] = new_tp[go]
    side = origin_side(wi)
    o[g] = pos[go] + side[go] * ps.eps
    d[g] = wi[go]
    if ps.splats is not None:
        on_splat = ps.is_splat(sh[go])
        layer_index = np.maximum(sh[go] - ps.shapes, 0)
        skip[0][g] = np.where(on_splat, layer_index, -1)
        skip[1][g] = np.where(on_splat, ptsplats.START_SCALE * ps.splats.scale_max[layer_index], 0.0)
        skip[2][g] = ns[go]
        skip[3][g] = np.where(on_splat, ps.splats.scale_min[layer_index], -1e30)
    counts[g, kind_index[go]] += 1
    capped[g] = counts[g, kind_index[go]] >= np.where(choose_spec[go], settings.specular_bounces, settings.diffuse_bounces)
    first_bounce = cls[g] < 0
    cls[g[first_bounce]] = np.where(choose_spec[go][first_bounce], 1, 0)
    prev_delta[g] = is_delta[go]
    prev_pdf[g] = pdf[go]
    prev_shape[g] = sh[go]
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

def path_time(pixel, sample, seed, count):
    """The shutter sample (0 .. count - 1) each path sees: a pixel's samples cycle through the times, from a
    per-pixel start, so every `count` consecutive samples of a pixel cover the shutter once and neighbouring pixels
    are not in step."""
    with np.errstate(over="ignore"):
        start = pcg(np.asarray(pixel, np.uint32) + pcg(np.uint32(seed) ^ np.uint32(0x5EED)))
        return ((np.asarray(sample, np.uint64) + start.astype(np.uint64)) % np.uint64(count)).astype(np.int64)


def _trace_timed(timed, times, width, height, x, y, keys, settings, cancel):
    """`trace_paths` for paths that each carry a time: the paths of every moment of `timed` (a list of
    `(PathScene, Camera)`) are traced against that moment's scene and camera, and the buckets put back in order."""
    n = len(keys)
    acc = _Accum(n)
    first = dict(alpha=np.zeros(n), albedo=np.zeros((n, 3)), shape=np.full(n, -1), t=np.zeros(n),
                 ns=np.zeros((n, 3)), pos=np.zeros((n, 3)), uv=np.zeros((n, 2)))
    for index, (ps, camera) in enumerate(timed):
        rows = np.flatnonzero(times == index)
        if not len(rows):
            continue
        d, eye, _, tmin, tmax = camera_rays(camera, width, height, x[rows], y[rows], keys[rows])
        part, seen = trace_paths(ps, np.broadcast_to(eye, d.shape), d, keys[rows], settings, tmin, tmax, cancel)
        for name in ("emission", "diffuse", "specular", "diffuse_indirect", "specular_indirect"):
            getattr(acc, name)[rows] = getattr(part, name)
        for name, value in seen.items():
            first[name][rows] = value
    return acc, first


def camera_rays(camera, width, height, x, y, keys=None):
    """Unit world directions through pixel positions (x, y) (fractions of a pixel), the ray origins, the cosine of
    each ray with the view axis (view depth is t / c) and the near/far bounds along each unit ray.

    With `keys` (the path keys) and a camera that has a lens (`fstop` > 0) each ray starts at its own point of the
    aperture and aims at the point of the focal plane its pixel looks at (nodebased/lens.py); the origins are then
    an (n, 3) array. Without either, one shared eye and the pinhole rays, exactly as before the lens existed."""
    eye, view = s._view_basis(camera)
    inverse_view = np.linalg.inv(view.astype(np.float64))
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    aspect = width / max(height, 1)
    local = np.column_stack(((2 * x / width - 1) * aspect / focal, (1 - 2 * y / height) / focal, -np.ones(len(x))))
    origins = eye.astype(np.float64)
    if keys is not None and lens.active(camera):
        ax, ay = lens.aperture_points(rand(keys, lens.DIM_U1), rand(keys, lens.DIM_U2), camera.aperture_blades,
                                      camera.blade_rotation, camera.anamorphic_squeeze)
        radius, focus = lens.aperture_radius(camera), lens.focus_plane(camera)
        ax, ay = ax * radius, ay * radius
        local = local - np.column_stack((ax, ay, np.zeros(len(x)))) / focus   # aim at the focus point from the lens point
        origins = eye.astype(np.float64) + np.column_stack((ax, ay, np.zeros(len(x)))) @ inverse_view.T
    dirs = local @ inverse_view.T
    c = np.linalg.norm(dirs, axis=1)
    return dirs / c[:, None], origins, 1.0 / c, camera.near * c, camera.far * c


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
        ns, _, uv, _, _, _, _, _, _ = _surface_any(ps, shape[got], prim[got], u[got], v[got], d[sl][got])
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


def _particle_lights(scene):
    """Synthetic Point lights (R7 of 7 finish), one per particle whose baked `emission` (ParticleRender3D's
    `particle_emission`/emission ramp) is greater than zero: an emissive particle (a spark, an ember)
    lights its surroundings the same way a Light3D would, through next-event estimation and its own
    shadow ray, instead of brightening only its own drawn pixel. Quadratic falloff, `light.world()`'s
    convention; the particle's own straight (unpremultiplied) colour tints the light.
    """
    lights = []
    for instance in scene.particles:
        if instance.emission is None or not len(instance.positions):
            continue
        glow = np.asarray(instance.emission, np.float64)
        hot = np.flatnonzero(glow > 0)
        if not len(hot):
            continue
        matrix = instance.matrix.astype(np.float64)
        world = (matrix[:3, :3] @ instance.positions.astype(np.float64).T).T + matrix[:3, 3]
        alpha = np.maximum(instance.colors[:, 3], 1e-6)
        rgb = instance.colors[:, :3] / alpha[:, None]
        for i in hot:
            lights.append(s.Light(kind="Point", color=tuple(float(c) for c in rgb[i]),
                                  intensity=float(glow[i]) * _PARTICLE_LIGHT_GAIN,
                                  position=s.Vec3(*(float(v) for v in world[i])), falloff_type="Quadratic"))
    return tuple(lights)


def render(scene, camera, width, height, background=(0., 0., 0., 0.), ambient=0.0, output="rgba",
           settings=None, cancel=None, progress=None, stats=None, backend="cpu", pass_hook=None, volume=None,
           moments=None, denoise_settings=None, history_key=None):
    """Path trace `scene` to a premultiplied float32 (height, width, 4) image.

    `moments`, the CPU reference only, is `render_motion`'s list of `(scene, camera)` across the shutter: every path
    then carries its own time (an index into the list, `path_time`) and traces that moment's scene through that
    moment's camera, inside this one sampling loop. `scene` and `camera` are then the middle moment.

    `backend` "cpu" is this reference; "auto" and "gpu" use `gpupathtrace` (auto falls back to the CPU
    reference when the GPU cannot take the scene, gpu reports why not). `stats`, when a dict, receives
    `samples` (per-pixel counts), `passes`, `seconds`, `backend`. `progress(stage, fraction, info)` runs
    after every pass; the render can be cancelled through `cancel` between passes and inside them.

    `denoise_settings` and `history_key` are output "denoise" only (a `DenoiseSettings`, defaults when None);
    `history_key`, when given, remembers this call's filtered image under that key so a later call with
    "denoise_temporal" on and the same key reprojects it through the two calls' motion (see `_render_denoised`).
    """
    if output not in PATH_OUTPUTS:
        raise ValueError(f"Unknown path traced output {output!r}")
    settings = (settings or PathSettings()).clamped()
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("Render dimensions must be positive")
    raytrace._cancel(cancel)
    check_scene(scene)
    if scene.particles:
        # R7 of 7 finish: an emissive particle lights its surroundings through the same next-event
        # estimation and shadow ray as any Light3D (`_particle_lights`); the particle itself is not a
        # traced shape (below, `_draw_particles` composites the visible sprites after the trace).
        extra_lights = _particle_lights(scene)
        if extra_lights:
            scene = replace(scene, lights=scene.lights + extra_lights)
    if output == "denoise":
        return _render_denoised(scene, camera, width, height, background, ambient, settings, cancel, progress, stats,
                                backend, pass_hook, volume, denoise_settings, history_key)
    if backend in ("auto", "gpu"):
        from . import gpu3d, gpupathtrace
        from .cancellation import Cancelled
        try:
            if not gpu3d.available():
                raise gpu3d.Unsupported(gpu3d.describe())
            # R7 of 7 finish (2): a particle is still not a shape the GPU trace itself takes (it never
            # occludes or scatters a ray there either), so the GPU traces the rest of the scene and this
            # composites the particle sprites on top afterwards, matching the CPU reference's own
            # `_draw_particles` call below bit-for-bit in approach (extra emissive-particle lights were
            # already folded into `scene.lights` above, so the GPU trace lights its surroundings too).
            gpu_scene = replace(scene, particles=()) if scene.particles else scene
            image = gpupathtrace.render(gpu_scene, camera, width, height, background, ambient, output, settings,
                                        cancel=cancel, progress=progress, stats=stats, volume=volume)
            if scene.particles and output == "rgba":
                image = _composite_gpu_particles(scene, gpu_scene, camera, width, height, ambient, image, cancel, volume)
            return image
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
    ps = build_scene(scene, ambient, eye=s._view_basis(camera)[0], volume=volume)
    timed = None
    if moments is not None and len(moments) > 1 and output not in DATA_OUTPUTS:
        timed = [(ps if index == len(moments) // 2 else
                  build_scene(sc, ambient, eye=s._view_basis(cam)[0], volume=volume), cam)
                 for index, (sc, cam) in enumerate(moments)]
    if stats is not None:
        stats["backend"] = "cpu"
    if output in DATA_OUTPUTS:
        data = render_data(ps, camera, width, height, output, cancel)
        if ps.volumes is not None:
            surface = data if output == "depth" else render_data(ps, camera, width, height, "depth", cancel)
            data = merge_volume_data(scene, camera, width, height, data, output, ps.volumes.settings, surface, cancel)
        return data
    npix = width * height
    channel = _channels(output)
    total = np.zeros((npix, 3))
    alpha_sum = np.zeros(npix)
    lum_sum, lum_sq = np.zeros(npix), np.zeros(npix)
    albedo_sum = np.zeros((npix, 3))
    count = np.zeros(npix, np.int64)
    tiles_x, tiles_y = -(-width // TILE), -(-height // TILE)
    tile_of = ((np.arange(npix) // width) // TILE) * tiles_x + (np.arange(npix) % width) // TILE
    tile_done = np.zeros(tiles_x * tiles_y, bool)
    pixel_done = np.zeros(npix, bool)      # adaptive sampling: the pixels whose own noise estimate is under the threshold
    adaptive = settings.adaptive
    total_samples = settings.max_samples if adaptive else settings.samples
    per_pass = settings.pass_samples or int(np.clip(_CHUNK // npix, 1, 8))
    sample_index, passes = 0, 0
    while sample_index < total_samples:
        raytrace._cancel(cancel)
        if adaptive:
            # the first pass reaches `min_samples` at once (no stop is allowed before it), the rest take `adaptive_pass_size`
            take = min((settings.min_samples if sample_index == 0 else settings.adaptive_pass_size),
                       total_samples - sample_index)
            active = np.flatnonzero(~pixel_done)
        else:
            take = min(per_pass, total_samples - sample_index)
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
            if timed is None:
                d, eye, _, tmin, tmax = camera_rays(camera, width, height, x, y, keys)
                acc, first = trace_paths(ps, np.broadcast_to(eye, d.shape), d, keys, settings, tmin, tmax, cancel)
            else:
                acc, first = _trace_timed(timed, path_time(rep, sample, settings.seed, len(timed)), width, height,
                                          x, y, keys, settings, cancel)
            if channel == "total":
                value = acc.total()
            elif channel == "albedo":
                value = first["albedo"]
            else:
                value = getattr(acc, channel)
            per_pixel = value.reshape(len(pixels), take, 3).sum(axis=1)
            total[pixels] += per_pixel
            albedo_sum[pixels] += first["albedo"].reshape(len(pixels), take, 3).sum(axis=1)
            alpha_sum[pixels] += first["alpha"].reshape(len(pixels), take).sum(axis=1)
            lum = (acc.total() @ np.array(LUMINANCE)).reshape(len(pixels), take)
            lum_sum[pixels] += lum.sum(axis=1)
            lum_sq[pixels] += (lum * lum).sum(axis=1)
            count[pixels] += take
        sample_index += take
        passes += 1
        if adaptive:
            if settings.noise_threshold > 0:
                pixel_done |= (count >= settings.min_samples) & (pixel_noise(lum_sum, lum_sq, count) < settings.noise_threshold)
        elif settings.noise_threshold > 0:
            _retire_tiles(tile_done, tile_of, lum_sum, lum_sq, count, tiles_x * tiles_y, settings.noise_threshold)
        elapsed = time.perf_counter() - started
        if progress is not None:
            if adaptive:
                converged = float(pixel_done.mean())
                progress("pathtrace", max(sample_index / total_samples, converged),
                         dict(samples=sample_index, passes=passes, seconds=elapsed, tiles_active=int((~pixel_done).sum()),
                              converged=converged, pixels_active=int((~pixel_done).sum())))
            else:
                progress("pathtrace", sample_index / total_samples,
                         dict(samples=sample_index, passes=passes, seconds=elapsed, tiles_active=int((~tile_done).sum()),
                              converged=float(tile_done[tile_of].mean())))
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
        image = over_background(image, background)
    if stats is not None:
        n_ = np.maximum(count, 1).astype(np.float64)
        mean_lum = lum_sum / n_
        variance = np.maximum(lum_sq / n_ - mean_lum * mean_lum, 0.0) * n_ / np.maximum(n_ - 1, 1) / n_
        stats.update(sampling=settings.sampling, converged=(pixel_done if adaptive else tile_done[tile_of]).reshape(height, width).copy(),
                     noise=pixel_noise(lum_sum, lum_sq, count).reshape(height, width),
                     samples=count.reshape(height, width).copy(), passes=passes,
                     seconds=time.perf_counter() - started, variance=variance.reshape(height, width),
                     albedo=(albedo_sum / n_[:, None]).reshape(height, width, 3),
                     alpha=alpha.reshape(height, width).astype(np.float32))
    image = image.reshape(height, width, 4).astype(np.float32)
    if scene.particles and output == "rgba":
        # The particles themselves are not traced shapes (see `_particle_lights` above); their own visible
        # sprites are composited on top exactly as `scene3d.render`'s raster/ray-traced modes do, tested
        # against the path-traced depth for correct occlusion by whatever the trace already put there.
        eye, view = s._view_basis(camera)
        focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
        aspect = width / max(height, 1)
        first_hit = render_data(ps, camera, width, height, "depth", cancel)
        particle_depth = np.where(first_hit[..., 3] > 0, first_hit[..., 0], np.inf).astype(np.float32)
        lit = [(light, *light.world()) for light in scene.lights if light.intensity > 0]
        s._draw_particles(scene, camera, width, height, image, particle_depth, eye, view, focal, aspect, cancel,
                          lights=lit, ambient=ambient, environments=scene.environments, shadow_context=None)
    return _read_only(image)


def render_motion(moments, width, height, background=(0., 0., 0., 0.), ambient=0.0, output="rgba", settings=None,
                  cancel=None, progress=None, backend="cpu", volume=None, stats=None):
    """Motion blurred path tracing (step R5): `moments` is a list of `(scene, camera)`, one per time across the shutter
    (`motionblur.shutter_times`), and every path sees one instant. On the CPU reference each path carries its time
    (`path_time`) through the one sampling loop, so the sample count, the adaptive noise stop, the time limit and the
    progress are those of the render as a whole. The GPU twin takes the times as equal shares instead: separate
    renders with their own seeds (`ceil(samples / n)` samples each, the time limit split the same way), whose mean is
    the blurred image. The data passes are sharp and read the middle time. `stats` is `render`'s (filled by the CPU
    reference only)."""
    settings = (settings or PathSettings()).clamped()
    if output in DATA_OUTPUTS:
        scene, camera = moments[len(moments) // 2]
        return render(scene, camera, width, height, background, ambient, output, settings, cancel=cancel,
                      backend=backend, volume=volume)
    count = len(moments)
    if backend == "auto":
        from . import gpu3d
        if not gpu3d.available():
            backend = "cpu"
    if backend == "cpu" and output != "denoise" and count > 1:
        scene, camera = moments[count // 2]
        return render(scene, camera, width, height, background, ambient, output, settings, cancel=cancel,
                      progress=progress, backend=backend, volume=volume, moments=moments, stats=stats)
    share = replace(settings, samples=max(1, -(-settings.samples // count)),
                    min_samples=max(1, -(-settings.min_samples // count)),
                    max_samples=max(1, -(-settings.max_samples // count)),
                    time_limit=settings.time_limit / count if settings.time_limit else 0.0).clamped()
    total = None
    for index, (scene, camera) in enumerate(moments):
        raytrace._cancel(cancel)
        image = render(scene, camera, width, height, background, ambient, output,
                       replace(share, seed=(settings.seed + 7919 * index) & 0x7FFFFFFF), cancel=cancel,
                       backend=backend, volume=volume)
        total = image.astype(np.float64) if total is None else total + image
        if progress is not None:
            progress("pathtrace", (index + 1) / count,
                     dict(samples=(share.max_samples if settings.adaptive else share.samples) * (index + 1),
                          passes=index + 1, seconds=0.0, tiles_active=0))
    return _read_only((total / count).astype(np.float32))


def merge_volume_data(scene, camera, width, height, data, output, settings, surface, cancel=None):
    """The data pass `output` (`data`, from `render_data` or the card) with the smoke put in where its first sample at
    `settings.depth_threshold` is nearer than the surface (`surface`, the surfaces' depth pass, alpha 1 where covered):
    the depth is that sample's view depth, `position` its world position, `normals` the density gradient pointing out of
    the smoke (turned toward the eye), `object_id` `scene3d.volume_id_base` plus the volume's index, `uv` zero, alpha 1.
    The raymarch finds the sample (one per pixel, never blurred), so both tracers agree on it exactly."""
    from . import volumerender
    covered = surface[..., 3] > 0
    nearest = np.where(covered, surface[..., 0], np.inf)
    hits = volumerender.first_hit_data(scene, camera, width, height, nearest, replace(settings, motion_blur=0.0), cancel)
    hits["hit"] = hits["hit"] & (hits["depth"] < nearest)
    merged = np.array(data)
    volumerender.write_data_pass(merged, hits, output, s.volume_id_base(scene))
    return _read_only(merged)


def merge_volume_depth(scene, camera, width, height, data, settings, cancel=None):
    """The depth pass with the smoke's first sample at `depth_threshold` merged in, as the other renderers do."""
    return merge_volume_data(scene, camera, width, height, data, "depth", settings, data, cancel)


def guide_aovs(scene, camera, width, height, settings=None, cancel=None, backend="cpu", volume=None, stats=None):
    """The passes an external denoiser reads next to the raw beauty: `{"albedo", "normals", "depth"}` as (H, W, 4)
    float32 (premultiplied albedo and the un-jittered normals and view depth). `stats`, the dict a beauty render
    filled, supplies the albedo (and its coverage alpha) the render already gathered; without it the albedo is
    traced at a few samples."""
    settings = (settings or PathSettings()).clamped()
    out = {}
    if stats is not None and "albedo" in stats and stats["albedo"].shape[:2] == (height, width):
        albedo = np.zeros((height, width, 4), np.float32)
        albedo[..., :3] = stats["albedo"]
        albedo[..., 3] = stats["alpha"] if "alpha" in stats else 1.0
        out["albedo"] = _read_only(albedo)
    else:
        few = replace(settings, samples=min(settings.samples, 8), time_limit=0.0, noise_threshold=0.0, sampling="fixed")
        out["albedo"] = render(scene, camera, width, height, (0, 0, 0, 0), 0.0, "albedo", few, cancel=cancel,
                               backend=backend, volume=volume)
    for name in ("normals", "depth"):
        out[name] = render(scene, camera, width, height, (0, 0, 0, 0), 0.0, name, settings, cancel=cancel,
                           backend=backend, volume=volume)
    return out


def _render_denoised(scene, camera, width, height, background, ambient, settings, cancel, progress, stats, backend,
                     pass_hook, volume, denoise_settings=None, history_key=None):
    """The beauty filtered by `ptdenoise` with its own guides and Render3D's denoiser knobs; the background goes
    on after the filter. With `denoise_settings.temporal` on and a `history_key`, the previous call under that
    key (if any, and the same size) is reprojected through this call's motion and blended in (`_TEMPORAL_HISTORY`);
    this call's own filtered image (before strength and the background) then becomes the next one's history."""
    st = {} if stats is None else stats
    beauty = render(scene, camera, width, height, (0, 0, 0, 0), ambient, "rgba", settings, cancel=cancel,
                    progress=progress, stats=st, backend=backend, pass_hook=pass_hook, volume=volume)
    st["beauty_raw"] = beauty     # the unfiltered beauty, transparent background: Render3D's `beauty_raw` layer
    guides = guide_aovs(scene, camera, width, height, settings, cancel, backend, volume, st)
    ds = (denoise_settings or DenoiseSettings()).clamped()
    history = None
    if ds.temporal and history_key is not None:
        previous = _TEMPORAL_HISTORY.get(history_key)
        if previous is not None and previous["image"].shape[:2] == (int(height), int(width)):
            from . import motionblur
            vectors = motionblur.motion_vectors(scene, camera, previous["scene"], previous["camera"], width, height)
            history = (previous["image"], vectors)
    filtered = ptdenoise.denoise(beauty, guides["albedo"], guides["normals"][..., :3], guides["depth"][..., 0],
                                 st.get("variance"), iterations=ds.iterations,
                                 phi_color=ptdenoise.PHI_COLOR * ds.color_sensitivity,
                                 phi_normal=ptdenoise.PHI_NORMAL * ds.normal_sensitivity,
                                 phi_depth=ptdenoise.PHI_DEPTH * ds.depth_sensitivity,
                                 strength=ds.strength, history=history)
    if ds.temporal and history_key is not None:
        _TEMPORAL_HISTORY[history_key] = {"image": filtered, "scene": scene, "camera": camera}
    return _read_only(over_background(filtered.astype(np.float64), background).astype(np.float32))


def over_background(image, background):
    """Premultiplied `image` (..., 4) over the straight-alpha `background` colour."""
    bg = np.asarray(background, np.float64).copy()
    bg[3] = np.clip(bg[3], 0, 1)
    bg[:3] *= bg[3]
    return image + bg * (1 - image[..., 3:4])


def denoised(beauty, guides, variance, background):
    """`beauty` (a transparent-background render) filtered with its `guides` (`guide_aovs`) and put over
    `background`, at the filter's defaults (Render3D's own knobs go through `_render_denoised` instead)."""
    image = ptdenoise.denoise(beauty, guides["albedo"], guides["normals"][..., :3], guides["depth"][..., 0], variance)
    return over_background(image.astype(np.float64), background).astype(np.float32)


def pixel_noise(lum_sum, lum_sq, count):
    """Adaptive sampling's per-pixel noise estimate: the variance of the pixel's mean luminance relative to the
    square of that mean. From the running sums of luminance and of its square, with n samples, mean m = sum / n
    and s2 = (sq / n - m^2) * n / (n - 1) the sample variance, it is (s2 / n) / (m + NOISE_FLOOR)^2. A flat
    pixel reads 0; a pixel whose samples scatter as widely as their mean reads 1 / n. `NOISE_FLOOR` keeps a
    black pixel from reading as infinitely noisy."""
    n = np.maximum(np.asarray(count, np.float64), 1.0)
    mean = lum_sum / n
    var_of_mean = np.maximum(lum_sq / n - mean * mean, 0.0) * n / np.maximum(n - 1.0, 1.0) / n
    return var_of_mean / (mean + NOISE_FLOOR) ** 2


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
    """The path tracer renders meshes, instances, splats and volumes lit by lights and environments.

    Particles (R7 of 7 finish) are not a traced shape: they do not occlude or scatter a camera or
    shadow ray, so they cast no shadow of their own and are invisible to reflections, refractions and
    other particles. What they do get: their own visible sprites are composited on top of the trace
    exactly as `scene3d.render`'s raster/ray-traced modes draw them (`render`, below), and an emissive
    one (`particle_emission`/an emission ramp) lights its surroundings as a real, shadowed Point light
    (`_particle_lights`) rather than only brightening its own drawn pixel. This holds on the GPU backend
    too (R7 of 7 finish (2), `_composite_gpu_particles`): the GPU traces the particle-free scene and the
    sprites are composited on the CPU afterwards, the same as the CPU reference's own particle pass.
    """
    return


def _composite_gpu_particles(scene, gpu_scene, camera, width, height, ambient, image, cancel, volume):
    """Composite `scene.particles`'s visible sprites over a `gpupathtrace.render` `rgba` image (R7 of 7
    finish (2)): a particle is not a shape the GPU trace itself takes, on the GPU any more than on the
    CPU reference (`check_scene` above), so occlusion is tested against a CPU first-hit depth pass over
    `gpu_scene` (the same scene minus particles the GPU actually traced) the same way `render`'s own CPU
    path gets its particle-occlusion depth from `render_data`, below. `image` already carries the
    background (`gpupathtrace.render` composites it before returning); the sprites go on top of that.
    """
    ps = build_scene(gpu_scene, ambient, eye=s._view_basis(camera)[0], volume=volume)
    eye, view = s._view_basis(camera)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    aspect = width / max(height, 1)
    first_hit = render_data(ps, camera, width, height, "depth", cancel)
    particle_depth = np.where(first_hit[..., 3] > 0, first_hit[..., 0], np.inf).astype(np.float32)
    lit = [(light, *light.world()) for light in scene.lights if light.intensity > 0]
    out = np.array(image, copy=True)
    s._draw_particles(scene, camera, width, height, out, particle_depth, eye, view, focal, aspect, cancel,
                      lights=lit, ambient=ambient, environments=scene.environments, shadow_context=None)
    return _read_only(out)


def render_scene3d(scene, camera, width, height, background, ambient, output, cancel, progress, return_depth, path,
                   volume=None, denoise_settings=None, history_key=None):
    """`scene3d.render`'s entry: one output, the CPU reference, `path` the `PathSettings` (defaults when None).
    `denoise_settings`/`history_key` are `output` "denoise" only; see `render`."""
    check_scene(scene)
    if output == "normals_blend":
        output = "normals"
    if output not in PATH_OUTPUTS:
        raise ValueError(f"the path tracer does not produce the {output!r} output")
    image = render(scene, camera, width, height, background, ambient, output, path, cancel=cancel, progress=progress,
                   volume=volume, denoise_settings=denoise_settings, history_key=history_key)
    if not return_depth:
        return image
    depth = render(scene, camera, width, height, background, ambient, "depth", path, cancel=cancel, volume=volume)
    covered = depth[..., 3] > 0
    return image, _read_only(np.where(covered, depth[..., 0], np.inf).astype(np.float32))
