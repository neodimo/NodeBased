"""CPU reference raymarch for `scene3d.Volume` members (docs/FLUIDS_SPIKE.md, docs/3D_FOUNDATION.md).

This module is the parity oracle for the GPU volume renderer: it is deliberately simple, exact where a
closed form exists and fully specified below, and it never optimises away a term the GPU version must
also compute. Pure NumPy, vectorised per ray batch, deterministic (no random numbers, fixed order of
volumes and lights).

Model
-----
One primary ray per pixel centre leaves the eye. For every volume the ray is clipped to the volume's box
(`origin` to `origin + shape * voxel_size` in object space) and to the nearest opaque mesh hit, then
marched front to back in segments of at most `step_size` world units. Each segment samples the density
at its MIDPOINT with zero-padded trilinear interpolation of the cell-centred grid, and treats it as
constant over the segment.

    sigma  = density_scale * density                extinction shape
    sigma_a = absorption * sigma                     absorbed light
    sigma_s = scattering * sigma                     scattered light
    sigma_t = sigma_a + sigma_s
    T_seg   = exp(-sigma_t * ds)
    L_seg   = (sigma_s / sigma_t) * (1 - T_seg) * S  the exact integral of sigma_s e^-tau S over the segment
    rgb    += T * L_seg ;  T *= T_seg                alpha of the volume is 1 - T

`S` is the light arriving at the sample times `color`, the "smoke color": the ambient term plus, for every
scene light, `light.color * light.intensity * light_attenuation(light, p)` times that light's shadow
transmittance. A scene without lights is unlit like the meshes: S is `color` alone. The phase function is 1
(isotropic and energy normalised: a thick, fully scattering slab lit by an intensity 1 light approaches
`color`). The shadow of a light at a sample is `exp(-shadow_density * sigma_t_unit * integral of sigma)`
along the ray from the sample toward the light, cut at the volume's exit or the light, midpoint rule with
exactly `shadow_steps` equal segments; the cone and falloff of Spot and Point lights come from
`scene3d.light_attenuation`, and a light whose factor is 0 at the sample costs no shadow ray. Meshes do not
shadow the volume and volumes do not shadow meshes; overlapping volumes do not shadow each other and are
composited whole, nearest box centre first.

Look terms (plan 3 step B), all off or neutral at their defaults so a document renders as before:

    phase      Henyey-Greenstein, normalised so isotropic is 1: P(g, c) = (1 - g^2) / (1 + g^2 - 2 g c)^1.5 with c
               the cosine between the direction the light travels and the direction from the sample to the eye, so
               a positive `anisotropy` brightens smoke that is lit from behind (forward scattering: the light
               continues toward the eye) and dims smoke lit from the camera's side.
    multiple   With `multi_scatter` m in (0, 1] the light term of a sample is (1 - m) * L0 + m * sum(w_n L_n) / sum(w_n),
    scattering L0 = P(g, c) exp(-tau), L_n = P(g 0.5^n, c) exp(-b^n tau), w_n = 0.5^(n-1), n = 1..N, b = 1 - 0.75 *
               `multi_scatter_blur`, tau the shadow ray's optical depth. Each octave sees the same shadow through a
               thinner medium (light leaks into the shadow, the edge softens) with a rounder phase (the light
               forgets its direction). It is a convex mix of terms each bounded by the unshadowed light, so it never
               makes a point brighter than the light that reaches it, and it costs no density lookups (tau is shared).
               N is 2, 3 or 4 by `quality`. m = 0 is exactly L0, the single scattering render.
    fire       Where temperature * `temperature_scale` (kelvin) exceeds `fire_threshold` the sample emits the radiance
               Le(K) = chromaticity(K) * (K / 1500)^4 (Planck spectrum through the CIE 1931 observer, in scene-linear
               ACEScg, luminance 1 before the T^4 brightness; or the piecewise-linear `fire_ramp` when one is given),
               tabulated on 64 log-spaced kelvin knots from 400 K to 8000 K and linearly interpolated (the GPU reads
               the same table). The emission coefficient is `fire_intensity` * sigma, attenuated like scattered light:
                   rgb += T * fire_intensity * Le * (1 - T_seg) / sigma_t_unit      (sigma * ds without extinction)
    fire light Emitted light also lights the smoke: the emission field fire_intensity * sigma * Le is block-averaged to
               a coarse grid (cells of at most 32 per side), blurred with a Gaussian of 15 % of the longest box side,
               and added to the incident light at every sample as `fire_light` * (that radius in world units) * the
               blurred field. It is isotropic and unshadowed. In a scene without lights the smoke source is
               `color` * (1 + that light). The field reads the sharp (unblurred by motion) grids.

The volume is composited over the mesh image with the premultiplied over operator using its own
transmittance, so a card in front of it hides it (the ray is cut at the card's depth) and a card behind is
dimmed by it. Particles, splats and transparent surfaces have no depth-tested interaction with volumes:
they are treated as behind a volume.

Control passes (`VOLUME_PASSES`) are integrals along the same rays, cut at mesh depth, in RGBA float32:

    volume_density      sum sigma * ds                              (R = G = B, alpha = coverage)
    volume_temperature  sum temperature * sigma * ds                (density weighted)
    volume_vorticity    sum |curl velocity| * ds                    (units 1/s, the volume's own space)
    volume_motion       forward vector in pixels per frame          (R = dx right, G = dy UP, as Nuke)

`volume_motion` is the screen displacement of the density-weighted mean position of the ray between this
frame and the next: project(p + v / fps) - project(p) with p = sum(sigma ds P) / sum(sigma ds) and v the
same average of the world-space velocity. Coverage (alpha) is 1 where the ray integral of density is
positive. `first_hit_depth` is the view-space depth of the first sample whose `sigma` reaches
`depth_threshold`, the value scene3d's `depth` output merges with the mesh depth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from . import scene3d

VOLUME_PASSES = scene3d.VOLUME_OUTPUTS
# The chunk of rays marched together, and the ceiling on sample evaluations one render may cost. The
# budget counts density lookups, shadow lookups included, and makes the CPU reference refuse a frame it
# cannot finish in reasonable time (lower the resolution, raise the step size, cut shadow steps).
RAY_CHUNK = 4096
SAMPLE_BUDGET = 300_000_000
_EPS = 1e-6


# `quality` presets: march segments across the longest volume diagonal, shadow segments per ray, multiple
# scattering octaves. "custom" (the default) uses `step_size` and `shadow_steps` as given.
QUALITY_PRESETS = {"preview": (48, 6, 2), "medium": (128, 12, 3), "final": (256, 24, 4)}
QUALITY_CHOICES = ("custom",) + tuple(QUALITY_PRESETS)
FIRE_KNOTS = 64
FIRE_K_LOW, FIRE_K_HIGH = 400.0, 8000.0    # kelvin range of the emission table; hotter reads the last knot
FIRE_K_REF = 1500.0                        # kelvin at which the T^4 brightness is 1
MULTI_SCATTER_DECAY = 0.5                  # octave weight and phase roundness ratio
FIRE_LIGHT_RADIUS = 0.15                   # of the longest box side
FIRE_LIGHT_CELLS = 32                      # coarse grid cells per side, at most


@dataclass(frozen=True)
class VolumeSettings:
    """The Render3D volume knobs (Houdini Pyro names in brackets)."""
    step_size: float = 0.05        # world units per march segment
    density_scale: float = 1.0     # [density_scale] multiplies the stored density
    shadow_density: float = 1.0    # [shadow_density] multiplies the density seen by shadow rays
    shadow_steps: int = 16         # equal segments per shadow ray
    scattering: float = 1.0        # [scattering]
    absorption: float = 0.2        # [absorption]
    color: tuple = (1.0, 1.0, 1.0)  # [smoke_color] tint of scattered light
    fps: float = 24.0              # frames per second, for volume_motion
    depth_threshold: float = 0.1   # scaled density at which `depth` sees the volume
    motion_blur: float = 0.0       # shutter length in frames; 0 is sharp (needs a velocity field)
    motion_samples: int = 8        # equal steps of the shutter
    anisotropy: float = 0.0        # [anisotropy] Henyey-Greenstein g, -0.99 to 0.99; 0 is isotropic
    multi_scatter: float = 0.0     # amount of the multiple scattering approximation, 0 to 1
    multi_scatter_blur: float = 0.5  # how far the octaves thin the medium the shadow ray sees, 0 to 1
    fire_intensity: float = 0.0    # [intensity scale] emission gain; 0 switches fire off
    temperature_scale: float = 1500.0  # [temperature scale] kelvin per unit of the stored temperature
    fire_threshold: float = 600.0  # kelvin below which the smoke does not glow
    fire_light: float = 1.0        # how strongly the fire lights the smoke around it
    fire_ramp: str = ""            # custom emission colour over kelvin, "K:r,g,b;K:r,g,b"; empty is blackbody
    quality: str = "custom"        # custom, preview, medium or final

    def validated(self):
        if not self.step_size > 0:
            raise ValueError("volume_step_size must be positive")
        if int(self.shadow_steps) < 1:
            raise ValueError("volume_shadow_steps must be at least 1")
        if not self.motion_blur >= 0:
            raise ValueError("volume_motion_blur must not be negative")
        if int(self.motion_samples) < 1:
            raise ValueError("volume_motion_samples must be at least 1")
        if not -0.99 <= self.anisotropy <= 0.99:
            raise ValueError("volume_anisotropy must be between -0.99 and 0.99")
        if not 0.0 <= self.multi_scatter <= 1.0:
            raise ValueError("volume_multi_scatter must be between 0 and 1")
        if not 0.0 <= self.multi_scatter_blur <= 1.0:
            raise ValueError("volume_multi_scatter_blur must be between 0 and 1")
        if not self.fire_intensity >= 0 or not self.fire_light >= 0:
            raise ValueError("volume_fire_intensity and volume_fire_light must not be negative")
        if not self.temperature_scale > 0:
            raise ValueError("volume_temperature_scale must be positive")
        if self.quality not in QUALITY_CHOICES:
            raise ValueError(f"volume_quality must be one of {', '.join(QUALITY_CHOICES)}")
        parse_fire_ramp(self.fire_ramp)
        return self

    def blurred(self, volume):
        """True when `volume` is sampled along the shutter: a shutter is open and there is a velocity field."""
        return self.motion_blur > 0 and volume.velocity is not None

    def glows(self, volume):
        """True when `volume` emits: fire is on and the volume has a temperature field."""
        return self.fire_intensity > 0 and volume.temperature is not None

    @property
    def octaves(self):
        """Multiple scattering octaves of the quality preset (3 for custom)."""
        return QUALITY_PRESETS[self.quality][2] if self.quality in QUALITY_PRESETS else 3

    def resolved(self, volumes):
        """The settings with the quality preset's step size and shadow steps filled in for `volumes` (a Scene's
        volumes): march segments across the longest world-space diagonal. Idempotent; "custom" returns self."""
        volumes = tuple(volumes or ())
        if self.quality == "custom" or self.quality not in QUALITY_PRESETS or not volumes:
            return self
        steps, shadow_steps, _ = QUALITY_PRESETS[self.quality]
        diagonal = max(float(np.linalg.norm(np.asarray(v.matrix, np.float64)[:3, :3] @ (np.array(v.shape) * v.voxel_size)))
                       for v in volumes)
        return replace(self, step_size=max(diagonal / steps, 1e-4), shadow_steps=shadow_steps)


# --- Fire: blackbody emission -------------------------------------------------------------------------------------

# XYZ (ACES, D60 white) to ACEScg (AP1) primaries.
_XYZ_TO_ACESCG = np.array([[1.6410233797, -0.3248032942, -0.2364246952],
                           [-0.6636628587, 1.6153315917, 0.0167563477],
                           [0.0117218943, -0.0082844420, 0.9883948585]])


def _lobe(wavelength, mu, sigma_low, sigma_high):
    sigma = np.where(wavelength < mu, sigma_low, sigma_high)
    return np.exp(-0.5 * ((wavelength - mu) / sigma) ** 2)


def blackbody_chromaticity(kelvin):
    """ACEScg colour of a blackbody at `kelvin` (scalar or array), scaled to luminance (CIE Y) 1: the Planck spectrum
    through the multi-lobe fit of the CIE 1931 observer (Wyman, Sloan and Shirley 2013), mapped to AP1. Negative
    components (kelvins whose chromaticity lies outside the gamut) are clipped to 0. Shape (..., 3)."""
    kelvin = np.asarray(kelvin, np.float64)
    wavelength = np.arange(380.0, 781.0, 5.0)
    x = (1.056 * _lobe(wavelength, 599.8, 37.9, 31.0) + 0.362 * _lobe(wavelength, 442.0, 16.0, 26.7)
         - 0.065 * _lobe(wavelength, 501.1, 20.4, 26.2))
    y = 0.821 * _lobe(wavelength, 568.8, 46.9, 40.5) + 0.286 * _lobe(wavelength, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(wavelength, 437.0, 11.8, 36.0) + 0.681 * _lobe(wavelength, 459.0, 26.0, 13.8)
    cmf = np.stack((x, y, z), -1)                                          # (W, 3)
    with np.errstate(over="ignore"):
        planck = wavelength ** -5.0 / np.expm1(1.4388e7 / (wavelength * np.maximum(kelvin[..., None], 1.0)))
    xyz = planck @ cmf                                                     # (..., 3)
    xyz = xyz / np.maximum(xyz[..., 1:2], 1e-300)
    return np.maximum(xyz @ _XYZ_TO_ACESCG.T, 0.0)


def blackbody_radiance(kelvin):
    """Emitted scene-linear ACEScg radiance at `kelvin`: chromaticity times (kelvin / 1500)^4."""
    kelvin = np.asarray(kelvin, np.float64)
    return blackbody_chromaticity(kelvin) * ((kelvin / FIRE_K_REF) ** 4)[..., None]


def parse_fire_ramp(text):
    """The stops of a `fire_ramp` string, "K:r,g,b;K:r,g,b" (kelvin, ACEScg radiance), sorted by kelvin; empty is
    None (blackbody). Raises ValueError with the offending stop."""
    text = (text or "").strip()
    if not text:
        return None
    stops = []
    for part in text.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            kelvin, rgb = part.split(":")
            values = [float(v) for v in rgb.replace(" ", "").split(",")]
            if len(values) != 3 or not float(kelvin) > 0:
                raise ValueError
            stops.append((float(kelvin), values))
        except ValueError:
            raise ValueError(f"volume_fire_ramp stop {part!r} is not 'kelvin:r,g,b'") from None
    if not stops:
        return None
    stops.sort(key=lambda stop: stop[0])
    return np.array([k for k, _ in stops]), np.array([v for _, v in stops])


def fire_knots():
    """The kelvin of the FIRE_KNOTS table entries (log spaced from FIRE_K_LOW to FIRE_K_HIGH)."""
    return FIRE_K_LOW * (FIRE_K_HIGH / FIRE_K_LOW) ** (np.arange(FIRE_KNOTS) / (FIRE_KNOTS - 1))


def fire_table(settings):
    """(FIRE_KNOTS, 3) float32 emission radiance table: blackbody, or the `fire_ramp` stops evaluated at the knots."""
    knots = fire_knots()
    ramp = parse_fire_ramp(settings.fire_ramp)
    if ramp is None:
        return blackbody_radiance(knots).astype(np.float32)
    kelvin, colours = ramp
    return np.stack([np.interp(knots, kelvin, colours[:, c]) for c in range(3)], -1).astype(np.float32)


def fire_radiance(table, kelvin):
    """Table lookup of the emitted radiance (N, 3) at `kelvin` (N,), linear between log-spaced knots, clamped."""
    u = np.log(np.maximum(kelvin, 1e-3) / FIRE_K_LOW) / math.log(FIRE_K_HIGH / FIRE_K_LOW) * (FIRE_KNOTS - 1)
    u = np.clip(u, 0.0, FIRE_KNOTS - 1.0)
    i = np.minimum(np.floor(u).astype(np.int64), FIRE_KNOTS - 2)
    f = (u - i)[:, None]
    return table[i].astype(np.float64) * (1 - f) + table[i + 1].astype(np.float64) * f


def _gaussian_blur(grid, sigma):
    """Separable Gaussian of a (nx, ny, nz, C) grid, `sigma` in cells, zero outside."""
    radius = max(1, int(math.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    for axis in range(3):
        n = grid.shape[axis]
        pad = [(0, 0)] * grid.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(grid, pad)
        out = np.zeros_like(grid)
        for k, weight in zip(offsets, kernel):
            index = [slice(None)] * grid.ndim
            index[axis] = slice(radius + k, radius + k + n)
            out += weight * padded[tuple(index)]
        grid = out
    return grid


def fire_light_grid(volume, settings):
    """(grid, cell): the light the fire casts on the smoke around it, or None. `grid` is (gx, gy, gz, 3) float32
    on cubic cells of `cell` object-space units starting at the volume's `origin` (a cell holds the blurred, scaled
    emission field; see the module docstring), None when the volume does not glow or `fire_light` is 0."""
    if not settings.glows(volume) or settings.fire_light <= 0:
        return None
    kelvin = (settings.temperature_scale * volume.temperature).astype(np.float64).reshape(-1)
    hot = (kelvin > settings.fire_threshold) & (volume.density.reshape(-1) > 0)
    emission = np.zeros((kelvin.size, 3))
    if hot.any():
        table = fire_table(settings)
        emission[hot] = fire_radiance(table, kelvin[hot]) * (
            settings.fire_intensity * settings.density_scale * volume.density.reshape(-1)[hot].astype(np.float64))[:, None]
    emission = emission.reshape(volume.density.shape + (3,))
    if not emission.any():
        return None
    factor = max(1, int(math.ceil(max(volume.shape) / FIRE_LIGHT_CELLS)))
    padded_shape = [-(-n // factor) * factor for n in volume.shape]
    grid = np.zeros(padded_shape + [3])
    grid[:volume.shape[0], :volume.shape[1], :volume.shape[2]] = emission
    coarse = grid.reshape(padded_shape[0] // factor, factor, padded_shape[1] // factor, factor,
                          padded_shape[2] // factor, factor, 3).mean(axis=(1, 3, 5))
    cell = factor * volume.voxel_size
    side = float(max(volume.shape) * volume.voxel_size)
    radius = FIRE_LIGHT_RADIUS * side                                    # object units
    world_scale = abs(np.linalg.det(np.asarray(volume.matrix, np.float64)[:3, :3])) ** (1 / 3)
    blurred = _gaussian_blur(coarse, radius / cell)
    return (settings.fire_light * radius * world_scale * blurred).astype(np.float32), cell


def vorticity_magnitude(volume):
    """|curl v| per cell of `volume.velocity`, central differences, float32 (1/s); zeros without velocity."""
    if volume.velocity is None:
        return np.zeros(volume.density.shape, np.float32)
    h = volume.voxel_size
    v = volume.velocity.astype(np.float64)
    grads = [[np.gradient(v[..., c], h, axis=a) if v.shape[a] > 1 else np.zeros(v.shape[:3])
              for a in range(3)] for c in range(3)]   # grads[component][axis]
    curl = np.stack((grads[2][1] - grads[1][2], grads[0][2] - grads[2][0], grads[1][0] - grads[0][1]), -1)
    return np.linalg.norm(curl, axis=-1).astype(np.float32)


def _trilinear(grid, g):
    """Zero-padded trilinear samples of a cell-centred grid at index-space points g (N, 3)."""
    shape = np.array(grid.shape[:3])
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + (dx, dy, dz)
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) \
                    * (f[:, 2] if dz else 1 - f[:, 2])
                ok = ((idx >= 0) & (idx < shape)).all(axis=1)
                idx = np.clip(idx, 0, shape - 1)
                val = grid[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
                w = np.where(ok, w, 0.0)
                out = out + (w[:, None] * val if val.ndim > 1 else w * val)
    return out


class _Volume:
    """A volume prepared for marching: inverse matrix, box and the derived vorticity grid."""

    def __init__(self, volume, want_vorticity, settings=None):
        self.volume = volume
        m = volume.matrix.astype(np.float64)
        self.m = m
        self.inv = np.linalg.inv(m)
        self.box_min = np.array(volume.origin, np.float64)
        self.box_max = self.box_min + np.array(volume.shape, np.float64) * volume.voxel_size
        self.centre = (m @ np.append((self.box_min + self.box_max) / 2, 1.0))[:3]
        self.vorticity = vorticity_magnitude(volume) if want_vorticity else None
        light = fire_light_grid(volume, settings) if settings is not None else None
        self.fire_light, self.fire_cell = light if light is not None else (None, 1.0)

    def fire_light_at(self, p_obj):
        """The fire's light (N, 3) at object-space points, zero-padded trilinear on the coarse grid."""
        if self.fire_light is None:
            return 0.0
        return _trilinear(self.fire_light, (p_obj - self.box_min) / self.fire_cell - .5)

    def to_grid(self, p_obj):
        return (p_obj - self.box_min) / self.volume.voxel_size - .5

    def to_object(self, p_world):
        return p_world @ self.inv[:3, :3].T + self.inv[:3, 3]

    def clip(self, origin, direction):
        """(t_enter, t_exit) of world rays against the box (t in world distance; enter >= 0)."""
        o = self.to_object(origin[None, :])[0]
        d = direction @ self.inv[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (self.box_min - o) / d, (self.box_max - o) / d
        near, far = np.minimum(a, b), np.maximum(a, b)
        parallel = np.abs(d) < 1e-12
        inside = ((o >= self.box_min) & (o <= self.box_max))
        near = np.where(parallel, np.where(inside, -np.inf, np.inf), near)
        far = np.where(parallel, np.where(inside, np.inf, -np.inf), far)
        return np.maximum(near.max(axis=-1), 0.0), far.min(axis=-1)


def _shutter_points(prep, settings, g):
    """Grid-space sample points of the shutter for `g`: the density at time t is the stored density moved
    forward by the velocity, so it is read at `g - v t` for the `motion_samples` mid-points t of the shutter
    (`motion_blur` frames at `fps`). One point (`g`) when the volume is not blurred."""
    volume = prep.volume
    if not settings.blurred(volume):
        return [g]
    shift = _trilinear(volume.velocity, g) * (settings.motion_blur / settings.fps / volume.voxel_size)
    n = int(settings.motion_samples)
    return [g - shift * ((s + .5) / n) for s in range(n)]


def _shutter_mean(grid, points):
    """Mean of the trilinear samples of `grid` over the shutter points."""
    total = _trilinear(grid, points[0])
    for point in points[1:]:
        total = total + _trilinear(grid, point)
    return total / len(points)


def _pixel_rays(camera, width, height):
    """Unit world directions per pixel centre (H*W, 3) and the |w| that maps view depth to ray distance."""
    eye, view = scene3d._view_basis(camera)
    aspect = width / max(height, 1)
    focal = 1.0 / math.tan(math.radians(camera.fov) / 2)
    xs = (np.arange(width) + .5) / width * 2 - 1
    ys = 1 - (np.arange(height) + .5) / height * 2
    lx = np.broadcast_to(xs[None, :] * aspect / focal, (height, width))
    ly = np.broadcast_to(ys[:, None] / focal, (height, width))
    local = np.stack((lx, ly, -np.ones_like(lx)), -1).reshape(-1, 3)      # view-space, forward depth 1
    world = local @ view.astype(np.float64)                                # rows of view are right, up, -forward
    length = np.linalg.norm(world, axis=1)
    return eye.astype(np.float64), world / length[:, None], length


def _shadow_transmittance(prep, settings, light, p_world, sigma_t_unit):
    """exp(-shadow_density * sigma_t_unit * scale * integral(density)) from each point toward the light."""
    return np.exp(-_shadow_tau(prep, settings, light, p_world, sigma_t_unit))


def _shadow_tau(prep, settings, light, p_world, sigma_t_unit):
    """shadow_density * sigma_t_unit * scale * integral(density) from each point toward the light (the optical depth)."""
    volume = prep.volume
    if light.kind == "Directional":
        _, direction = light.world()
        to_light = np.broadcast_to(-direction.astype(np.float64), p_world.shape)
        limit = np.full(len(p_world), np.inf)
    else:
        position = light.world()[0].astype(np.float64)
        offset = position - p_world
        distance = np.maximum(np.linalg.norm(offset, axis=1), 1e-12)
        to_light = offset / distance[:, None]
        limit = distance
    o = prep.to_object(p_world)
    d = to_light @ prep.inv[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (prep.box_min - o) / d, (prep.box_max - o) / d
    near, far = np.minimum(a, b), np.maximum(a, b)
    parallel = np.abs(d) < 1e-12
    inside = (o >= prep.box_min) & (o <= prep.box_max)
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), near).max(axis=1)
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), far).min(axis=1)
    # The integral runs from where the ray enters the box (0 for a point inside it, as every march sample is)
    # to where it leaves it or reaches the light. Points outside the box (meshes and splats it shadows) enter late.
    near = np.maximum(near, 0.0)
    length = np.clip(np.minimum(far, limit) - near, 0.0, None)
    near = np.where(length > 0, near, 0.0)      # a ray that misses the box integrates nothing, at a finite start
    steps = int(settings.shadow_steps)
    total = np.zeros(len(p_world))
    for j in range(steps):
        p = o + d * (near + (j + .5) / steps * length)[:, None]
        total += _trilinear(volume.density, prep.to_grid(p))
    return settings.shadow_density * sigma_t_unit * settings.density_scale * total * (length / steps)


def _henyey_greenstein(cosine, g):
    """Henyey-Greenstein phase normalised so g = 0 is exactly 1 (isotropic), as the module docstring states."""
    if g == 0.0:
        return 1.0
    return (1 - g * g) / (1 + g * g - 2 * g * cosine) ** 1.5


def _light_weight(settings, tau, cosine):
    """The phase and shadow factor of one light at a sample: L0 = P(g) exp(-tau), or with `multi_scatter` the mix of
    L0 and the octaves L_n = P(g 0.5^n) exp(-b^n tau) described in the module docstring."""
    g, m = settings.anisotropy, settings.multi_scatter
    single = _henyey_greenstein(cosine, g) * np.exp(-tau)
    if m <= 0:
        return single
    b = 1.0 - 0.75 * settings.multi_scatter_blur
    total = weights = 0.0
    weight, thin = 1.0, 1.0
    for n in range(1, settings.octaves + 1):
        thin *= b
        total = total + weight * _henyey_greenstein(cosine, g * MULTI_SCATTER_DECAY ** n) * np.exp(-tau * thin)
        weights += weight
        weight *= MULTI_SCATTER_DECAY
    return (1 - m) * single + m * total / weights


class ShadowCasters:
    """The volumes of a scene as shadow casters for surfaces (meshes, relit splats, the shadow catcher):
    `transmittance(light, points)` is exp(-shadow_density * sigma_t * integral of scaled density) along each point's
    ray to the light, through every volume (the same optical depth the volume's own shadow rays use), so a dense
    plume darkens the floor under it. `shadow_density` 0 casts nothing."""

    def __init__(self, volumes, settings):
        volumes = tuple(volumes)
        self.settings = settings.validated().resolved(volumes)
        self.preps = [_Volume(v, False) for v in volumes]

    def __bool__(self):
        return bool(self.preps) and self.settings.shadow_density > 0

    def transmittance(self, light, points):
        points = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.ones(len(points))
        if not self:
            return out
        sigma_t = self.settings.absorption + self.settings.scattering
        for prep in self.preps:
            out *= _shadow_transmittance(prep, self.settings, light, points, sigma_t)
        return out


def _light_cosine(light, p_world, view_dirs):
    """Cosine between the direction the light travels to each sample and the direction from the sample to the eye
    (the negated ray): 1 is forward scattering, -1 is back scattering."""
    if light.kind == "Directional":
        travel = np.broadcast_to(light.world()[1].astype(np.float64), p_world.shape)
    else:
        offset = p_world - light.world()[0].astype(np.float64)
        travel = offset / np.maximum(np.linalg.norm(offset, axis=1), 1e-12)[:, None]
    return -np.sum(view_dirs * travel, axis=1)


def _march(prep, settings, lights, ambient, eye, dirs, t0, t1, want, lit, cancel, occluders=None):
    """March the given rays (already clipped to [t0, t1]); returns accumulators for the rays."""
    volume = prep.volume
    n = len(dirs)
    step = float(settings.step_size)
    count = np.ceil((t1 - t0) / step - 1e-9).astype(np.int64)
    trans = np.ones(n)
    rgb = np.zeros((n, 3))
    density_sum = np.zeros(n)
    temp_sum = np.zeros(n)
    vort_sum = np.zeros(n)
    pos_sum = np.zeros((n, 3))
    vel_sum = np.zeros((n, 3))
    first_t = np.full(n, np.inf)
    color = np.asarray(settings.color, np.float64)
    fire_table_ = fire_table(settings) if settings.glows(volume) and "beauty" in want else None
    sa, ss = settings.absorption, settings.scattering
    sigma_t_unit = sa + ss
    for k in range(int(count.max()) if n else 0):
        if cancel is not None and cancel.is_set():
            from .imaging import Cancelled
            raise Cancelled()
        ds = np.clip(t1 - (t0 + k * step), 0.0, step)
        live = (ds > 0) & (trans > _EPS)
        if not live.any():
            break
        idx = np.nonzero(live)[0]
        tm = t0[idx] + k * step + ds[idx] / 2
        p_world = eye + dirs[idx] * tm[:, None]
        g = prep.to_grid(prep.to_object(p_world))
        points = _shutter_points(prep, settings, g) if want & {"beauty", "density", "temperature", "depth"} else [g]
        sigma = settings.density_scale * _shutter_mean(volume.density, points)
        seg = ds[idx]
        if "beauty" in want:
            st = sigma_t_unit * sigma
            t_seg = np.exp(-st * seg)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(st > 0, ss * sigma / np.maximum(st, 1e-300), 0.0)
            if lit:
                incident = np.full((len(idx), 3), float(ambient))
                for light in lights:
                    attn = scene3d.light_attenuation(light, p_world).astype(np.float64)
                    hit = attn > 0
                    if hit.any() and sigma.any():
                        shadow = np.ones(len(idx))
                        tau = _shadow_tau(prep, settings, light, p_world[hit], sigma_t_unit)
                        shadow[hit] = _light_weight(settings, tau, _light_cosine(light, p_world[hit], dirs[idx][hit]))
                        if occluders is not None and light.shadows:
                            # Meshes and splats between the sample and the light (hard shadows).
                            dense = hit & (sigma > 0)
                            shadow[dense] *= occluders(p_world[dense], light)
                        incident += (np.asarray(light.color, np.float64) * light.intensity)[None, :] \
                            * (attn * shadow)[:, None]
                if prep.fire_light is not None:
                    incident += prep.fire_light_at(prep.to_object(p_world))
                source = color * incident
            elif prep.fire_light is not None:
                source = color * (1.0 + prep.fire_light_at(prep.to_object(p_world)))
            else:
                source = np.broadcast_to(color, (len(idx), 3))
            rgb[idx] += trans[idx, None] * (frac * (1 - t_seg))[:, None] * source
            if fire_table_ is not None and sigma.any():
                kelvin = settings.temperature_scale * _shutter_mean(volume.temperature, points)
                glow = (kelvin > settings.fire_threshold) & (sigma > 0)
                if glow.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        weight = np.where(sigma_t_unit > 0, (1 - t_seg) / max(sigma_t_unit, 1e-300), sigma * seg)
                    emitted = fire_radiance(fire_table_, kelvin[glow]) * (settings.fire_intensity * weight[glow])[:, None]
                    rgb[idx[glow]] += trans[idx[glow], None] * emitted
            trans[idx] *= t_seg
        w = sigma * seg
        density_sum[idx] += w
        if "temperature" in want and volume.temperature is not None:
            temp_sum[idx] += _shutter_mean(volume.temperature, points) * w
        if "vorticity" in want:
            vort_sum[idx] += _trilinear(prep.vorticity, g) * seg
        if "motion" in want:
            pos_sum[idx] += p_world * w[:, None]
            if volume.velocity is not None:
                vel_sum[idx] += _trilinear(volume.velocity, g) * w[:, None]
        if "depth" in want:
            newly = (sigma >= settings.depth_threshold) & np.isinf(first_t[idx])
            first_t[idx[newly]] = tm[newly]
    vel_sum = vel_sum @ prep.m[:3, :3].T   # object-space velocity to world
    return dict(trans=trans, rgb=rgb, density=density_sum, temperature=temp_sum, vorticity=vort_sum,
                position=pos_sum, velocity=vel_sum, first_t=first_t)


def integrate(scene, camera, width, height, mesh_depth, settings, ambient, want, cancel=None, occluders=None):
    """Raymarch every volume of `scene`; returns per-pixel arrays for the requested terms.

    `mesh_depth` is the view-space depth buffer of the opaque meshes (inf where empty) or None. `want`
    is a set of "beauty", "density", "temperature", "vorticity", "motion", "depth". The result holds
    `rgb` (H, W, 3, premultiplied), `alpha` (H, W), and the accumulated terms as (H, W) arrays, with
    `position` and `velocity` (H, W, 3) sums for motion and `first_t` the ray distance of the first
    depth hit (inf for none).
    """
    settings = settings.validated().resolved(scene.volumes)
    width, height = int(width), int(height)
    lights = [light for light in scene.lights if light.intensity > 0]
    lit = bool(scene.lights)
    eye, dirs, length = _pixel_rays(camera, width, height)
    if mesh_depth is None:
        t_mesh = np.full(width * height, np.inf)
    else:
        t_mesh = np.asarray(mesh_depth, np.float64).reshape(-1) * length
    need_vort = "vorticity" in want
    ranked = sorted(((_Volume(v, need_vort, settings if "beauty" in want else None), i) for i, v in enumerate(scene.volumes)),
                    key=lambda pair: float(np.linalg.norm(pair[0].centre - eye)))
    preps = [pair[0] for pair in ranked]
    numbers = [pair[1] + 1 for pair in ranked]
    clips = []
    work = 0.0
    shadow = len(lights) * settings.shadow_steps if "beauty" in want else 0
    for prep in preps:
        t0, t1 = prep.clip(eye, dirs)
        t1 = np.minimum(t1, t_mesh)
        clips.append((t0, t1))
        span = np.clip(t1 - t0, 0.0, None)
        blur = 1 + int(settings.motion_samples) if settings.blurred(prep.volume) else 1
        work += float(np.ceil(span / settings.step_size)[span > 0].sum()) * (blur + shadow)
    if work > SAMPLE_BUDGET:
        raise ValueError(f"Volume raymarch exceeds the CPU reference budget: {work:,.0f} estimated density "
                         f"lookups > {SAMPLE_BUDGET:,}; lower the resolution or samples, raise "
                         "volume_step_size or cut volume_shadow_steps")
    total = width * height
    acc = dict(trans=np.ones(total), rgb=np.zeros((total, 3)), density=np.zeros(total),
               temperature=np.zeros(total), vorticity=np.zeros(total), position=np.zeros((total, 3)),
               velocity=np.zeros((total, 3)), first_t=np.full(total, np.inf), id=np.zeros(total))
    for prep, number, (t0, t1) in zip(preps, numbers, clips):
        rays = np.nonzero(t1 - t0 > 0)[0]
        for start in range(0, len(rays), RAY_CHUNK):
            sel = rays[start:start + RAY_CHUNK]
            got = _march(prep, settings, lights, ambient, eye, dirs[sel], t0[sel], t1[sel], want, lit, cancel, occluders)
            acc["rgb"][sel] += acc["trans"][sel, None] * got["rgb"]
            acc["trans"][sel] *= got["trans"]
            for name in ("density", "temperature", "vorticity", "position", "velocity"):
                acc[name][sel] += got[name]
            acc["first_t"][sel] = np.minimum(acc["first_t"][sel], got["first_t"])
            # Nearest volume that holds smoke on the ray (volumes are visited near to far).
            fresh = (acc["id"][sel] == 0) & (got["density"] > 0)
            acc["id"][sel[fresh]] = number
    shape2 = (height, width)
    out = {name: (value.reshape(height, width, 3) if value.ndim == 2 else value.reshape(shape2))
           for name, value in acc.items()}
    out["alpha"] = 1.0 - out["trans"]
    out["length"] = length.reshape(shape2)
    return out


def composite_beauty(scene, camera, width, height, out, depth, settings, ambient, cancel=None, occluders=None):
    """Composite the volumes over `out` (premultiplied RGBA, modified in place), cut at mesh `depth`.

    `occluders(points, light)` (scene3d._volume_occluders) is the transmittance of the scene's meshes and casting
    splats from each point toward a shadow-casting light; it darkens the smoke behind them."""
    got = integrate(scene, camera, width, height, depth, settings, ambient, {"beauty"}, cancel, occluders)
    out[..., :3] = got["rgb"] + got["trans"][..., None] * out[..., :3]
    out[..., 3] = got["alpha"] + got["trans"] * out[..., 3]


def first_hit_depth(scene, camera, width, height, mesh_depth, settings, cancel=None):
    """View-space depth (H, W) of the first sample at or above `settings.depth_threshold` (inf: none)."""
    got = integrate(scene, camera, width, height, mesh_depth, settings, 0.0, {"depth"}, cancel)
    return (got["first_t"] / got["length"]).astype(np.float32)


def render_pass(scene, camera, width, height, mesh_depth, settings, name, cancel=None):
    """One control pass as (H, W, 4) float32 RGBA; `name` is one of `VOLUME_PASSES`."""
    if name not in VOLUME_PASSES:
        raise ValueError(f"Unknown volume pass {name!r}")
    key = name.split("_", 1)[1]
    if key == "motion":
        settings = replace(settings, motion_blur=0.0)   # the pass carries the unblurred vectors
    got = integrate(scene, camera, width, height, mesh_depth, settings, 0.0, {key}, cancel)
    return finish_pass(got, key, camera, settings)


def gpu_accumulators(scene, camera, width, height, mesh_depth, key, images, far_to_near):
    """`integrate`'s accumulators for control pass `key` from the GPU draws of gpuvolume.render_passes.

    `images[target][slot]` are the per-volume (H, W, 4) images in `far_to_near` order (indices into
    `scene.volumes`, as gpuvolume draws them): 'sums' (density, temperature, vorticity, first hit) or, for
    motion, position and velocity sums with the density weight in the position's alpha. The volumes are added
    in the reference's near-to-far order, the first hit takes the minimum and the id the nearest covered one."""
    shape = (int(height), int(width))
    acc = dict(density=np.zeros(shape), temperature=np.zeros(shape), vorticity=np.zeros(shape),
               position=np.zeros(shape + (3,)), velocity=np.zeros(shape + (3,)), first_t=np.full(shape, np.inf),
               id=np.zeros(shape))
    _, _, length = _pixel_rays(camera, width, height)
    acc["length"] = length.reshape(shape)
    for slot in reversed(range(len(far_to_near))):
        number = far_to_near[slot] + 1
        first = images[0][slot]
        if key == "motion":
            acc["density"] += first[..., 3]
            acc["position"] += first[..., :3]
            acc["velocity"] += images[1][slot][..., :3]
            covered = first[..., 3] > 0
        else:
            acc["density"] += first[..., 0]
            acc["temperature"] += first[..., 1]
            acc["vorticity"] += first[..., 2]
            acc["first_t"] = np.minimum(acc["first_t"], np.where(first[..., 3] < 1e37, first[..., 3], np.inf))
            covered = first[..., 0] > 0
        acc["id"] = np.where((acc["id"] == 0) & covered, number, acc["id"])
    return acc


def finish_pass(got, key, camera, settings):
    """The (H, W, 4) float32 RGBA of control pass `key` from `integrate`-style accumulators."""
    height, width = got["density"].shape
    image = np.zeros((height, width, 4), np.float32)
    covered = got["density"] > 0
    if key == "id":
        image[..., 0] = image[..., 1] = image[..., 2] = got["id"]
        covered = got["id"] > 0
    elif key == "motion":
        weight = np.maximum(got["density"], 1e-30)[..., None]
        mean_p = (got["position"] / weight).reshape(-1, 3)
        mean_v = (got["velocity"] / weight).reshape(-1, 3)
        here, _ = scene3d.project(camera, width, height, mean_p.astype(np.float32))
        there, _ = scene3d.project(camera, width, height, (mean_p + mean_v / settings.fps).astype(np.float32))
        delta = (there - here).reshape(height, width, 2)
        image[..., 0] = np.where(covered, delta[..., 0], 0.0)
        image[..., 1] = np.where(covered, -delta[..., 1], 0.0)   # Nuke's y points up, image rows go down
    else:
        value = got[key].astype(np.float32)
        image[..., 0] = image[..., 1] = image[..., 2] = value
    image[..., 3] = covered
    return image
