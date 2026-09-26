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
from dataclasses import dataclass

import numpy as np

from . import scene3d

VOLUME_PASSES = scene3d.VOLUME_OUTPUTS
# The chunk of rays marched together, and the ceiling on sample evaluations one render may cost. The
# budget counts density lookups, shadow lookups included, and makes the CPU reference refuse a frame it
# cannot finish in reasonable time (lower the resolution, raise the step size, cut shadow steps).
RAY_CHUNK = 4096
SAMPLE_BUDGET = 300_000_000
_EPS = 1e-6


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

    def validated(self):
        if not self.step_size > 0:
            raise ValueError("volume_step_size must be positive")
        if int(self.shadow_steps) < 1:
            raise ValueError("volume_shadow_steps must be at least 1")
        return self


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

    def __init__(self, volume, want_vorticity):
        self.volume = volume
        m = volume.matrix.astype(np.float64)
        self.m = m
        self.inv = np.linalg.inv(m)
        self.box_min = np.array(volume.origin, np.float64)
        self.box_max = self.box_min + np.array(volume.shape, np.float64) * volume.voxel_size
        self.centre = (m @ np.append((self.box_min + self.box_max) / 2, 1.0))[:3]
        self.vorticity = vorticity_magnitude(volume) if want_vorticity else None

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
    far = np.maximum(a, b)
    far = np.where(np.abs(d) < 1e-12, np.inf, far).min(axis=1)
    length = np.clip(np.minimum(far, limit), 0.0, None)
    steps = int(settings.shadow_steps)
    total = np.zeros(len(p_world))
    for j in range(steps):
        p = o + d * ((j + .5) / steps * length)[:, None]
        total += _trilinear(volume.density, prep.to_grid(p))
    tau = settings.shadow_density * sigma_t_unit * settings.density_scale * total * (length / steps)
    return np.exp(-tau)


def _march(prep, settings, lights, ambient, eye, dirs, t0, t1, want, lit, cancel):
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
        sigma = settings.density_scale * _trilinear(volume.density, g)
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
                        shadow[hit] = _shadow_transmittance(prep, settings, light, p_world[hit], sigma_t_unit)
                        incident += (np.asarray(light.color, np.float64) * light.intensity)[None, :] \
                            * (attn * shadow)[:, None]
                source = color * incident
            else:
                source = np.broadcast_to(color, (len(idx), 3))
            rgb[idx] += trans[idx, None] * (frac * (1 - t_seg))[:, None] * source
            trans[idx] *= t_seg
        w = sigma * seg
        density_sum[idx] += w
        if "temperature" in want and volume.temperature is not None:
            temp_sum[idx] += _trilinear(volume.temperature, g) * w
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


def integrate(scene, camera, width, height, mesh_depth, settings, ambient, want, cancel=None):
    """Raymarch every volume of `scene`; returns per-pixel arrays for the requested terms.

    `mesh_depth` is the view-space depth buffer of the opaque meshes (inf where empty) or None. `want`
    is a set of "beauty", "density", "temperature", "vorticity", "motion", "depth". The result holds
    `rgb` (H, W, 3, premultiplied), `alpha` (H, W), and the accumulated terms as (H, W) arrays, with
    `position` and `velocity` (H, W, 3) sums for motion and `first_t` the ray distance of the first
    depth hit (inf for none).
    """
    settings = settings.validated()
    width, height = int(width), int(height)
    lights = [light for light in scene.lights if light.intensity > 0]
    lit = bool(scene.lights)
    eye, dirs, length = _pixel_rays(camera, width, height)
    if mesh_depth is None:
        t_mesh = np.full(width * height, np.inf)
    else:
        t_mesh = np.asarray(mesh_depth, np.float64).reshape(-1) * length
    need_vort = "vorticity" in want
    preps = sorted((_Volume(v, need_vort) for v in scene.volumes),
                   key=lambda p: float(np.linalg.norm(p.centre - eye)))
    clips = []
    work = 0.0
    shadow = len(lights) * settings.shadow_steps if "beauty" in want else 0
    for prep in preps:
        t0, t1 = prep.clip(eye, dirs)
        t1 = np.minimum(t1, t_mesh)
        clips.append((t0, t1))
        span = np.clip(t1 - t0, 0.0, None)
        work += float(np.ceil(span / settings.step_size)[span > 0].sum()) * (1 + shadow)
    if work > SAMPLE_BUDGET:
        raise ValueError(f"Volume raymarch exceeds the CPU reference budget: {work:,.0f} estimated density "
                         f"lookups > {SAMPLE_BUDGET:,}; lower the resolution or samples, raise "
                         "volume_step_size or cut volume_shadow_steps")
    total = width * height
    acc = dict(trans=np.ones(total), rgb=np.zeros((total, 3)), density=np.zeros(total),
               temperature=np.zeros(total), vorticity=np.zeros(total), position=np.zeros((total, 3)),
               velocity=np.zeros((total, 3)), first_t=np.full(total, np.inf))
    for prep, (t0, t1) in zip(preps, clips):
        rays = np.nonzero(t1 - t0 > 0)[0]
        for start in range(0, len(rays), RAY_CHUNK):
            sel = rays[start:start + RAY_CHUNK]
            got = _march(prep, settings, lights, ambient, eye, dirs[sel], t0[sel], t1[sel], want, lit, cancel)
            acc["rgb"][sel] += acc["trans"][sel, None] * got["rgb"]
            acc["trans"][sel] *= got["trans"]
            for name in ("density", "temperature", "vorticity", "position", "velocity"):
                acc[name][sel] += got[name]
            acc["first_t"][sel] = np.minimum(acc["first_t"][sel], got["first_t"])
    shape2 = (height, width)
    out = {name: (value.reshape(height, width, 3) if value.ndim == 2 else value.reshape(shape2))
           for name, value in acc.items()}
    out["alpha"] = 1.0 - out["trans"]
    out["length"] = length.reshape(shape2)
    return out


def composite_beauty(scene, camera, width, height, out, depth, settings, ambient, cancel=None):
    """Composite the volumes over `out` (premultiplied RGBA, modified in place), cut at mesh `depth`."""
    got = integrate(scene, camera, width, height, depth, settings, ambient, {"beauty"}, cancel)
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
    got = integrate(scene, camera, width, height, mesh_depth, settings, 0.0, {key}, cancel)
    height, width = got["density"].shape
    image = np.zeros((height, width, 4), np.float32)
    covered = got["density"] > 0
    if key == "motion":
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
