"""High-resolution pyro reconstruction driven by a cached coarse fluid stream.

The coarse cache owns the motion. This pass samples its scalar fields and velocity on a finer
grid, advects the fine scalar detail by that velocity, and adds deterministic curl-noise detail.
It deliberately has no pressure projection: it is a look/detail pass, not a second fluid domain.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import fluid3d, simcache
from .sparsevol import SparseGrid, TILE

DEFAULTS = {"upres_factor": 2, "turbulence": 0.0, "swirl_size": 1.0, "grain": 2,
            "pulse_length": 30.0, "shredding": 0.0, "seed": 0, "upres_backend": "auto",
            "swirl": 0.0, "swirl_scale": 6.0, "swirl_life": 1.0, "swirl_max": 1.5,
            "cache_memory_mb": 256, "cache_disk_mb": 2048}

# Swirl (docs/FLUIDS_SPIKE.md, "Fluids 17 B"): small-scale rotational motion added to the velocity that carries the fine field.
SWIRL_WAVES = 24         # random divergence-free waves in the band-limited curl noise
SWIRL_CONFINE = 2.0      # vorticity confinement strength per unit of `swirl`
SWIRL_LIMIT = 2.0        # the added velocity never exceeds this multiple of the coarse speed at the voxel
SWIRL_KEEP = 0.92        # share of the carried detail kept each frame (the rest relaxes back to the coarse field)
SWIRL_SPIN = 16.0        # persistent swirl (Fluids 17 C): confinement acceleration per second, per unit of `swirl`
SWIRL_INJECT = 4.0       # persistent swirl: curl-noise source per second, per unit of `swirl` (times the coarse speed)
SWIRL_PROJECT = 20       # persistent swirl: Jacobi pressure iterations that keep the carried velocity divergence-free
SWIRL_MARGIN = 2         # voxels of velocity computed around the active tiles (the confinement stencil reaches two)


def _active_fine_tiles(source, factor, previous=None, halo=1):
    """Conservative fine tile support, including interpolation at the smoke boundary."""
    fields = {"density": source.density}
    if source.fuel is not None:
        fields["fuel"] = source.fuel
    occupied = SparseGrid.from_dense(fields, tile=TILE).coords
    shape = tuple(n * factor for n in source.density.shape)
    tile_shape = tuple((n + TILE - 1) // TILE for n in shape)
    mask = np.zeros(tile_shape, dtype=bool)
    for cx, cy, cz in occupied:
        # The coarse tile spans eight cells. Fine sampling reaches one coarse
        # cell past the boundary; a one-fine-tile halo is conservative.
        lo = np.maximum(0, np.array((cx, cy, cz)) * factor - halo)
        hi = np.minimum(tile_shape, (np.array((cx, cy, cz)) + 1) * factor + halo)
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    if previous is not None:
        prior = previous.sparse if previous.sparse is not None else previous.to_sparse()
        for cx, cy, cz in prior.coords:
            lo = np.maximum(0, np.array((cx, cy, cz)) - halo)
            hi = np.minimum(tile_shape, np.array((cx, cy, cz)) + halo + 1)
            mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return np.argwhere(mask).astype(np.int32)


def swirl_waves(seed, scale, factor, frame):
    """The band-limited curl noise of the swirl: `SWIRL_WAVES` plane waves, each a divergence-free velocity field
    `C cos(K . x + phase)` with C perpendicular to K, so the sum has zero divergence by construction. Wavelengths run from
    `scale` coarse cells down to half of it, with amplitude falling as wavelength ** (5/6) (a Kolmogorov spectrum); the
    amplitudes are normalised so the field's rms speed is 1. Phases drift a little each frame, so the pattern boils slowly.
    Returns float32 `K` (n, 3, radians per fine voxel), `C` (n, 3) and `phase` (n,)."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 0x5717])
    n = SWIRL_WAVES
    wavelength = max(float(scale), 0.5) * np.exp2(-rng.random(n))
    k = 2.0 * np.pi / (wavelength * int(factor))
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    pole = rng.normal(size=(n, 3))
    pole -= np.sum(pole * direction, axis=1, keepdims=True) * direction
    pole /= np.linalg.norm(pole, axis=1, keepdims=True)
    amplitude = wavelength ** (5.0 / 6.0)
    amplitude /= np.sqrt(np.sum(amplitude ** 2) / 2.0)
    phase = rng.random(n) * 2.0 * np.pi + (rng.random(n) - .5) * .5 * float(frame)
    return ((direction * k[:, None]).astype(np.float32), (pole * amplitude[:, None]).astype(np.float32),
            phase.astype(np.float32))


def swirl_box(coords):
    """Lowest voxel and size of the dense box around the given fine tiles, with `SWIRL_MARGIN` voxels of margin."""
    low = coords.min(axis=0).astype(np.int64) * TILE - SWIRL_MARGIN
    high = (coords.max(axis=0).astype(np.int64) + 1) * TILE + SWIRL_MARGIN
    return low, high - low


def _central(field, axis, h):
    """Central difference along `axis` with edge-clamped neighbours (the GPU kernels clamp the same way)."""
    n = field.shape[axis]
    up = np.take(field, np.minimum(np.arange(n) + 1, n - 1), axis=axis)
    down = np.take(field, np.maximum(np.arange(n) - 1, 0), axis=axis)
    return (up - down) / np.float32(2.0 * h)


def swirl_dense_cpu(guide, factor, low, dims, waves, strength, voxel):
    """Reference swirl velocity (the part added to the coarse guide) on the dense box `low`/`dims`, shape (*dims, 3).

    Noise: the wave sum times `strength` times the coarse speed there, so there is no swirl in still air. Confinement: the
    curl of the coarse-plus-noise velocity, its magnitude's gradient N, and `SWIRL_CONFINE * strength * h * (N x curl)`, a
    velocity that spins each vortex up. The result is limited to `SWIRL_LIMIT` times the coarse speed."""
    K, C, phase = waves
    coarse = guide.shape[:3]
    axes = [np.arange(dims[a], dtype=np.float32) + np.float32(low[a]) for a in range(3)]
    q = [np.clip(axes[a] * np.float32((coarse[a] - 1) / (coarse[a] * factor - 1)), 0, coarse[a] - 1) for a in range(3)]
    qx, qy, qz = np.meshgrid(*q, indexing="ij")
    vc = np.stack([fluid3d.trilerp(guide[..., a], qx, qy, qz) for a in range(3)], axis=-1).astype(np.float32)
    speed = np.sqrt(np.sum(vc * vc, axis=-1))
    noise = np.zeros((*dims, 3), np.float32)
    for i in range(len(phase)):
        theta = (K[i, 0] * axes[0][:, None, None] + K[i, 1] * axes[1][None, :, None] +
                 K[i, 2] * axes[2][None, None, :] + phase[i])
        wave = np.cos(theta).astype(np.float32)
        noise += wave[..., None] * C[i]
    u = vc + np.float32(strength) * speed[..., None] * noise
    h = float(voxel)
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    omega = np.stack([_central(uz, 1, h) - _central(uy, 2, h), _central(ux, 2, h) - _central(uz, 0, h),
                      _central(uy, 0, h) - _central(ux, 1, h)], axis=-1)
    mag = np.sqrt(np.sum(omega * omega, axis=-1))
    eta = np.stack([_central(mag, a, h) for a in range(3)], axis=-1)
    unit = eta / (np.sqrt(np.sum(eta * eta, axis=-1, keepdims=True)) + np.float32(1e-20))
    u = u + np.float32(SWIRL_CONFINE * strength * h) * np.cross(unit, omega)
    added = u - vc
    cap = np.float32(SWIRL_LIMIT) * speed
    size = np.sqrt(np.sum(added * added, axis=-1))
    added *= (np.minimum(1.0, cap / np.maximum(size, np.float32(1e-20))))[..., None]
    return added.astype(np.float32)


def swirl_blocks(guide, coords, factor, params, frame, voxel, backend):
    """The swirl velocity at every voxel of the given fine tiles, shape (len(coords), 8, 8, 8, 3)."""
    strength = float(params.get("swirl", 0.0))
    waves = swirl_waves(int(params.get("seed", 0)), float(params.get("swirl_scale", 6.0)), factor, int(frame or 0))
    low, dims = swirl_box(coords)
    if backend == "gpu":
        from .fluid_upres_gpu import swirl_dense
        dense = swirl_dense(guide, factor, low, dims, waves, strength, SWIRL_CONFINE * strength, SWIRL_LIMIT, voxel)
    else:
        dense = swirl_dense_cpu(guide, factor, low, dims, waves, strength, voxel)
    local = np.indices((TILE, TILE, TILE), dtype=np.int64)
    index = [coords[:, axis, None, None, None].astype(np.int64) * TILE + local[axis] - low[axis] for axis in range(3)]
    return dense[index[0], index[1], index[2]]


def _jacobi_pressure(div, h, iters):
    """`iters` Jacobi sweeps of the 7-point Poisson equation (edge-clamped neighbours, as the GPU kernel) for `div`."""
    p = np.zeros_like(div)
    h2 = np.float32(h * h)
    for _ in range(int(iters)):
        q = np.pad(p, 1, mode="edge")
        total = (q[2:, 1:-1, 1:-1] + q[:-2, 1:-1, 1:-1] + q[1:-1, 2:, 1:-1] + q[1:-1, :-2, 1:-1] +
                 q[1:-1, 1:-1, 2:] + q[1:-1, 1:-1, :-2])
        p = ((total - h2 * div) / np.float32(6.0)).astype(np.float32)
    return p


def divergence_dense(u, h):
    """Central-difference divergence of a dense velocity field (*dims, 3), edge-clamped."""
    return (_central(u[..., 0], 0, h) + _central(u[..., 1], 1, h) + _central(u[..., 2], 2, h)).astype(np.float32)


def swirl_persist_cpu(guide, factor, low, dims, waves, uf0, strength, voxel, dt, life, vmax, iters,
                      spin=None, inject=None):
    """Reference persistent swirl step (Fluids 17 C) on the dense box `low`/`dims`: the fine velocity `uf0` of the last frame in,
    the new one (*dims, 3) out. In order: advect `uf0` by the coarse velocity plus itself (semi-Lagrangian), inject curl noise
    scaled by the coarse speed, confine vorticity of the total velocity, decay by exp(-dt / life), project with `iters` Jacobi
    sweeps, and limit to `vmax` times the coarse speed."""
    spin = SWIRL_SPIN if spin is None else spin
    inject = SWIRL_INJECT if inject is None else inject
    K, C, phase = waves
    coarse = guide.shape[:3]
    h = float(voxel)
    axes = [np.arange(dims[a], dtype=np.float32) + np.float32(low[a]) for a in range(3)]
    q = [np.clip(axes[a] * np.float32((coarse[a] - 1) / (coarse[a] * factor - 1)), 0, coarse[a] - 1) for a in range(3)]
    qx, qy, qz = np.meshgrid(*q, indexing="ij")
    vc = np.stack([fluid3d.trilerp(guide[..., a], qx, qy, qz) for a in range(3)], axis=-1).astype(np.float32)
    speed = np.sqrt(np.sum(vc * vc, axis=-1))
    step = np.float32(dt / h)
    gx, gy, gz = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in dims], indexing="ij")
    total = vc + uf0
    back = [np.clip(g - total[..., a] * step, 0, dims[a] - 1) for a, g in enumerate((gx, gy, gz))]
    u = np.stack([fluid3d.trilerp(uf0[..., a], *back) for a in range(3)], axis=-1).astype(np.float32)
    noise = np.zeros((*dims, 3), np.float32)
    for i in range(len(phase)):
        theta = (K[i, 0] * axes[0][:, None, None] + K[i, 1] * axes[1][None, :, None] +
                 K[i, 2] * axes[2][None, None, :] + phase[i])
        noise += np.cos(theta).astype(np.float32)[..., None] * C[i]
    u = u + (np.float32(inject * strength * dt) * speed)[..., None] * noise
    t = vc + u
    omega = np.stack([_central(t[..., 2], 1, h) - _central(t[..., 1], 2, h), _central(t[..., 0], 2, h) - _central(t[..., 2], 0, h),
                      _central(t[..., 1], 0, h) - _central(t[..., 0], 1, h)], axis=-1)
    mag = np.sqrt(np.sum(omega * omega, axis=-1))
    eta = np.stack([_central(mag, a, h) for a in range(3)], axis=-1)
    unit = eta / (np.sqrt(np.sum(eta * eta, axis=-1, keepdims=True)) + np.float32(1e-20))
    u = (u + np.float32(spin * strength * dt * h) * np.cross(unit, omega)) * np.float32(np.exp(-dt / life))
    if iters > 0:
        u = u - np.stack([_central(_jacobi_pressure(divergence_dense(u, h), h, iters), a, h) for a in range(3)], axis=-1)
    size = np.sqrt(np.sum(u * u, axis=-1))
    u *= (np.minimum(1.0, np.float32(vmax) * speed / np.maximum(size, np.float32(1e-20))))[..., None]
    return u.astype(np.float32)


def scatter_tiles(coords, blocks, low, dims):
    """Dense box (*dims, 3) holding the vector tile `blocks` of the fine tiles `coords`; zero elsewhere (and outside the box)."""
    dense = np.zeros((*[int(n) for n in dims], 3), np.float32)
    if len(coords) == 0:
        return dense
    local = np.indices((TILE, TILE, TILE), dtype=np.int64)
    index = [np.asarray(coords)[:, axis, None, None, None].astype(np.int64) * TILE + local[axis] - low[axis] for axis in range(3)]
    inside = np.ones(index[0].shape, bool)
    for axis in range(3):
        inside &= (index[axis] >= 0) & (index[axis] < dims[axis])
    dense[index[0][inside], index[1][inside], index[2][inside]] = np.asarray(blocks, np.float32)[inside]
    return dense


def swirl_persistent_blocks(guide, coords, factor, params, frame, voxel, backend, previous, dt):
    """The persistent fine velocity `u_fine` at every voxel of the given fine tiles, shape (len(coords), 8, 8, 8, 3).

    The previous frame's `u_fine` is read from its cached sparse grid (zero at the first frame, and where a tile was dropped)."""
    strength = float(params.get("swirl", 0.0))
    life = float(params.get("swirl_life", 1.0))
    vmax = float(params.get("swirl_max", 1.5))
    waves = swirl_waves(int(params.get("seed", 0)), float(params.get("swirl_scale", 6.0)), factor, int(frame or 0))
    low, dims = swirl_box(coords)
    prior = None
    if previous is not None:
        prior = previous.sparse if getattr(previous, "sparse", None) is not None else None
    uf0 = np.zeros((*[int(n) for n in dims], 3), np.float32)
    if prior is not None and "ufine" in prior.data and len(prior.coords):
        uf0 = scatter_tiles(prior.coords, prior.data["ufine"], low, dims)
    if backend == "gpu":
        from .fluid_upres_gpu import swirl_persist
        dense = swirl_persist(guide, factor, low, dims, waves, uf0, strength, voxel, dt, life, vmax, SWIRL_PROJECT,
                              SWIRL_SPIN, SWIRL_INJECT)
    else:
        dense = swirl_persist_cpu(guide, factor, low, dims, waves, uf0, strength, voxel, dt, life, vmax, SWIRL_PROJECT)
    local = np.indices((TILE, TILE, TILE), dtype=np.int64)
    index = [coords[:, axis, None, None, None].astype(np.int64) * TILE + local[axis] - low[axis] for axis in range(3)]
    return dense[index[0], index[1], index[2]]


def _sparse_blocks(fields, coords, shape, coarse_shape, backend, guide, previous, step, swirl=None):
    """Fine-grid tile blocks (`coords` x 8 x 8 x 8 per field, plus the sampled guide velocity when there is a guide) for the active tiles.

    With `previous` the blocks are that fine state carried back along `guide`; without it they are the coarse `fields` sampled on the fine
    grid (carried back along `guide` first when there is one). `swirl` (tiles x 8 x 8 x 8 x 3) is extra velocity added to the guide
    for the carry only; the returned velocity block stays the guide."""
    if backend == "gpu":
        from .fluid_upres_gpu import reconstruct_sparse
        return reconstruct_sparse(fields, coords, shape, guide, previous, step, swirl)
    if backend != "cpu":
        raise ValueError("sparse up-res backend must be cpu or gpu")
    local = np.indices((TILE, TILE, TILE), dtype=np.int32)
    xyz = [coords[:, axis, None, None, None] * TILE + local[axis] for axis in range(3)]
    valid = np.ones((len(coords), TILE, TILE, TILE), dtype=bool)
    for axis in range(3):
        valid &= xyz[axis] < shape[axis]
    sample = [(p.astype(np.float32) * np.float32((coarse_shape[i] - 1) / (shape[i] - 1)))
              for i, p in enumerate(xyz)]
    velocity = None
    if guide is not None:
        velocity = np.stack([fluid3d.trilerp(guide[..., axis], *sample) for axis in range(3)], axis=-1)
        back = [xyz[axis].astype(np.float32) - velocity[..., axis] * np.float32(step)
                for axis in range(3)]
        back = [np.clip(back[axis], 0, shape[axis] - 1) for axis in range(3)]
        if swirl is not None:
            # The smoke density alone is carried by the guide plus the swirl; the flame, temperature and fuel keep the guide.
            carry = velocity + swirl
            swirled = [np.clip(xyz[axis].astype(np.float32) - carry[..., axis] * np.float32(step), 0, shape[axis] - 1)
                       for axis in range(3)]
    else:
        back = [p.astype(np.float32) for p in xyz]
    blocks = {}
    for name, field in fields.items():
        prior = getattr(previous, name) if previous is not None else None
        place = swirled if swirl is not None and name == "density" else back
        if prior is not None:
            block = fluid3d.trilerp(prior, *place).astype(np.float32)
        elif guide is not None:
            coarse_back = [place[axis] * np.float32((coarse_shape[axis] - 1) /
                                                    (shape[axis] - 1)) for axis in range(3)]
            block = fluid3d.trilerp(field, *coarse_back).astype(np.float32)
        else:
            block = fluid3d.trilerp(field, *sample).astype(np.float32)
        block[~valid] = 0
        blocks[name] = block
    if velocity is not None:
        velocity[~valid] = 0
        blocks["velocity"] = velocity.astype(np.float32)
    return blocks


def _curl_y(positions, size, octaves, seed):
    """The y component of `particles.turbulence_field(positions, "curl", ...)` alone: a third of its noise evaluations."""
    from .particles import _fbm
    points = positions.astype(np.float64) / float(size)
    eps = 1e-3

    def derivative(salt, axis):
        step = np.zeros(3)
        step[axis] = eps
        return (_fbm(points + step, seed, salt, octaves) - _fbm(points - step, seed, salt, octaves)) / (2.0 * eps)
    return derivative(17, 2) - derivative(29, 0)


def _tile_noise(coords, params, seed, stride=1):
    """Curl-noise detail (zero mean) at every voxel of the given fine tiles, shape (len(coords), 8, 8, 8).

    `stride` 1 evaluates the noise at every voxel. A larger stride evaluates it on a lattice that many voxels apart over the tiles'
    bounding box and interpolates trilinearly: about `stride ** 3` times fewer evaluations, with features smaller than two lattice
    steps smoothed away (the reinjecting pass uses 2, since it draws a new pattern every frame)."""
    from .particles import turbulence_field
    local = np.indices((TILE, TILE, TILE), dtype=np.float32).reshape(3, -1).T
    out = np.empty((len(coords), TILE, TILE, TILE), np.float32)
    size = max(.001, float(params.get("swirl_size", 1.0)))
    octaves = max(1, int(params.get("grain", 2)))
    if stride <= 1:
        for first in range(0, len(coords), 128):
            group = coords[first:first + 128]
            positions = (group[:, None, :].astype(np.float32) * TILE + local[None, :, :]).reshape(-1, 3)
            noise = turbulence_field(positions, "curl", size, octaves, int(seed))[:, 1]
            out[first:first + len(group)] = noise.reshape(-1, TILE, TILE, TILE)
        return out
    low = coords.min(axis=0).astype(np.int64) * TILE // stride
    high = (coords.max(axis=0).astype(np.int64) + 1) * TILE // stride + 2
    axes = [np.arange(low[a], high[a], dtype=np.float32) * stride for a in range(3)]
    # Shifted off the integers: the noise's lattice nodes (swirl size 1, 2, 4... put the samples on them) have no curl, so a lattice
    # sitting exactly on them would read zero. The pattern is random, so a fixed shift changes nothing statistically.
    lattice = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3) + np.float32(.37)
    values = np.empty(len(lattice), np.float32)
    for first in range(0, len(lattice), 1 << 16):
        values[first:first + (1 << 16)] = _curl_y(lattice[first:first + (1 << 16)], size, octaves, int(seed))
    values = values.reshape(high - low)
    for first in range(0, len(coords), 128):
        group = coords[first:first + 128]
        positions = (group[:, None, :].astype(np.float32) * TILE + local[None, :, :]).reshape(-1, 3)
        at = [positions[:, a] / stride - low[a] for a in range(3)]
        out[first:first + len(group)] = fluid3d.trilerp(values, *at).reshape(-1, TILE, TILE, TILE)
    return out


def upres_sparse_grid(source, factor, backend="cpu", guide_velocity=None, previous=None, params=None,
                      previous_source=None, frame=None):
    """Reconstruct only smoke/fuel-bearing tiles; return compact field blocks.

    Guided transport samples the cached velocity into just the active fine
    tiles. A prior sparse frame supplies compact tiles for temporal advance.

    With `frame` given the pass reinjects the coarse cache every frame: the fine field is the coarse field sampled on the fine
    grid plus the detail left over from the previous frame carried along the guide (`previous_source` is the previous coarse
    frame, whose own sampled-and-carried copy is subtracted from the carried fine field, so only the detail is carried), and new
    detail is seeded into the smoke that arrived this frame. Without it (the older behaviour) the fine field is the previous fine
    field carried along, with nothing new entering after the first frame.
    """
    factor = int(factor)
    if factor not in (2, 4):
        raise ValueError("sparse up-res factor must be 2 or 4")
    shape = tuple(n * factor for n in source.density.shape)
    coarse_shape = tuple(source.density.shape)
    guide = source.velocity if guide_velocity is None else guide_velocity
    fps = getattr(getattr(source, "stream", None), "fps", 24.0)
    step = 1.0 / (max(float(fps), 1e-6) * float(source.voxel_size) / factor)
    strength = float((params or DEFAULTS).get("swirl", 0.0)) if guide is not None else 0.0
    reach = 1.0 + min(SWIRL_LIMIT, 1.5 * strength) if strength > 0 else 1.0
    if strength > 0 and float((params or DEFAULTS).get("swirl_life", 0.0)) > 0:
        reach = 1.0 + max(0.0, float((params or DEFAULTS).get("swirl_max", 1.5)))
    halo = 1 if guide is None else max(1, int(np.ceil(float(np.max(np.abs(guide))) * reach * step / TILE)) + 1)
    coords = _active_fine_tiles(source, factor, previous, halo)
    names = ("density", "temperature", "flame", "fuel")
    fields = {name: getattr(source, name) for name in names if getattr(source, name) is not None}
    swirl = None
    ufine = None
    if strength > 0 and len(coords):
        if float((params or DEFAULTS).get("swirl_life", 0.0)) > 0:
            swirl = ufine = swirl_persistent_blocks(guide, coords, factor, params, frame, float(source.voxel_size) / factor,
                                                    backend, previous, 1.0 / max(float(fps), 1e-6))
        else:
            swirl = swirl_blocks(guide, coords, factor, params, frame, float(source.voxel_size) / factor, backend)
    blocks = _sparse_blocks(fields, coords, shape, coarse_shape, backend, guide, previous, step, swirl)
    if ufine is not None:
        blocks["ufine"] = ufine.astype(np.float32)
    fresh = None
    keep = SWIRL_KEEP
    if ufine is not None:
        # A persistent swirl keeps the smoke's displacement as long as the swirl itself lasts (never less than step B's share).
        keep = max(SWIRL_KEEP, float(np.exp(-1.0 / (max(float(fps), 1e-6) * float(params["swirl_life"])))))
    if frame is not None:
        # Reinjection: coarse field now + (fine carried - coarse carried). With no previous frame this is just the coarse field now.
        now = _sparse_blocks(fields, coords, shape, coarse_shape, backend, None, None, step) if guide is not None or previous is not None else blocks
        if previous is not None and previous_source is not None and guide is not None:
            before = {name: getattr(previous_source, name) for name in fields if getattr(previous_source, name, None) is not None}
            carried = _sparse_blocks(before, coords, shape, coarse_shape, backend, guide, None, step)
            for name in fields:
                if name in carried and swirl is not None and name == "density":
                    # The fine field was carried by the guide plus the swirl, the coarse one by the guide alone: what is left
                    # is the swirl's displacement of the smoke plus the carried detail, kept and relaxed back to the coarse field.
                    blocks[name] = now[name] + np.float32(keep) * (blocks[name] - carried[name])
                elif name in carried:
                    blocks[name] = now[name] + blocks[name] - carried[name]
                else:
                    blocks[name] = now[name]
            fresh = np.maximum(now["density"] - carried["density"], 0.0) if "density" in carried else now["density"]
        else:
            for name in fields:
                blocks[name] = now[name]
            fresh = now["density"]
        for name in ("density", "flame", "fuel"):
            if name in blocks:
                np.maximum(blocks[name], 0.0, out=blocks[name])
    # Candidate halo tiles may be wholly empty after interpolation. Do not
    # retain or cache them, even if another channel has nonzero background.
    occupied = np.any(blocks["density"] != 0, axis=(1, 2, 3))
    if "fuel" in blocks:
        occupied |= np.any(blocks["fuel"] != 0, axis=(1, 2, 3))
    coords = coords[occupied]
    blocks = {name: block[occupied] for name, block in blocks.items()}
    if fresh is not None:
        fresh = fresh[occupied]
    params = params or DEFAULTS
    amount = float(params.get("turbulence", 0.0))
    shredding = float(params.get("shredding", 0.0))
    if (amount or shredding) and len(coords) and (previous is None or frame is not None):
        detail = max(0.0, min(amount, 2.0)) + max(0.0, min(shredding, 2.0))
        seed = int(params.get("seed", 0)) + (0 if frame is None else 7919 * int(frame))
        noise = _tile_noise(coords, params, seed, 1 if fresh is None else 2)
        if fresh is None:
            blocks["density"] *= np.maximum(0.0, 1.0 + np.float32(detail) * noise)
        else:
            blocks["density"] = np.maximum(0.0, blocks["density"] + np.float32(detail) * noise * fresh)
    if len(coords):
        mass = float(np.sum(source.density, dtype=np.float64)) * factor ** 3
        got = float(np.sum(blocks["density"], dtype=np.float64))
        if got > 1e-20:
            blocks["density"] *= np.float32(mass / got)
    return SparseGrid(shape, coords, blocks, tile=TILE)


def _resize(field, shape):
    """Trilinearly resample a cell-centred scalar field, preserving its domain extent."""
    if tuple(field.shape) == tuple(shape):
        return np.asarray(field, dtype=np.float32).copy()
    out = np.asarray(field, dtype=np.float32)
    for axis, (n, m) in enumerate(zip(field.shape, shape)):
        coords = np.linspace(0.0, n - 1.0, m, dtype=np.float32)
        lo = np.floor(coords).astype(np.intp)
        hi = np.minimum(lo + 1, n - 1)
        weight = coords - lo
        a = np.take(out, lo, axis=axis)
        b = np.take(out, hi, axis=axis)
        view = [1] * out.ndim
        view[axis] = m
        out = a + (b - a) * weight.reshape(view)
    return out


def _mass_match(field, mass):
    got = float(np.sum(field, dtype=np.float64))
    if got > 1e-20:
        field *= np.float32(mass / got)
    return field


def _advect(field, velocity, voxel_size, fps):
    """Backtrace one frame through a cell-centred high-resolution velocity field."""
    nx, ny, nz = field.shape
    x, y, z = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32),
                          np.arange(nz, dtype=np.float32), indexing="ij")
    step = np.float32(1.0 / (max(float(fps), 1e-6) * float(voxel_size)))
    return fluid3d.trilerp(field, x - velocity[..., 0] * step,
                           y - velocity[..., 1] * step, z - velocity[..., 2] * step).astype(np.float32)


def upres_volume(source, params, frame, guide_velocity=None, previous=None):
    """Advance one fine-grid frame using the cached coarse velocity and previous fine state."""
    factor = int(params["upres_factor"])
    if factor not in (1, 2, 4):
        raise ValueError("FluidUpres3D: upres_factor must be 1, 2 or 4")
    shape = tuple(int(n * factor) for n in source.density.shape)
    coarse_velocity = source.velocity if guide_velocity is None else guide_velocity
    backend = params.get("upres_backend", "auto")
    gpu_result = None
    if backend in ("auto", "gpu") and factor > 1:
        from .fluid_gpu_solver import Unsupported
        from .fluid_upres_gpu import reconstruct
        try:
            gpu_result = reconstruct(source, factor, coarse_velocity, previous)
        except Unsupported:
            if backend == "gpu":
                raise
    if gpu_result is not None:
        density, temperature, flame, fuel, velocity = gpu_result
    else:
        density = _resize(source.density, shape)
        temperature = None if source.temperature is None else _resize(source.temperature, shape)
        flame = None if source.flame is None else _resize(source.flame, shape)
        fuel = None if source.fuel is None else _resize(source.fuel, shape)
        if previous is not None:
            density = previous.density.copy()
            temperature = None if previous.temperature is None else previous.temperature.copy()
            flame = None if previous.flame is None else previous.flame.copy()
            fuel = None if previous.fuel is None else previous.fuel.copy()
        velocity = None
        if coarse_velocity is not None:
            velocity = np.stack([_resize(coarse_velocity[..., i], shape) for i in range(3)], axis=-1)
        if velocity is not None and factor > 1:
            density = _advect(density, velocity, float(source.voxel_size) / factor,
                              getattr(getattr(source, "stream", None), "fps", 24.0))
            if temperature is not None:
                temperature = _advect(temperature, velocity, float(source.voxel_size) / factor,
                                      getattr(getattr(source, "stream", None), "fps", 24.0))
            if flame is not None:
                flame = _advect(flame, velocity, float(source.voxel_size) / factor,
                                getattr(getattr(source, "stream", None), "fps", 24.0))
            if fuel is not None:
                fuel = _advect(fuel, velocity, float(source.voxel_size) / factor,
                               getattr(getattr(source, "stream", None), "fps", 24.0))
    # Disturb the transported scalar on the new voxel scale. Hash-lattice noise is deterministic
    # by seed and has no process-global random state.
    amount = float(params.get("turbulence", 0.0))
    # Seed detail once, then transport it with the fine state. Reapplying the same
    # modulation every frame compounds it and creates temporal flicker.
    if (amount or float(params.get("shredding", 0.0))) and factor > 1 and previous is None:
        from .particles import turbulence_field
        xs, ys, zs = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in shape), indexing="ij")
        positions = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        noise = turbulence_field(positions, "curl", max(0.001, float(params.get("swirl_size", 1.0))),
                                 max(1, int(params.get("grain", 2))),
                                 int(params.get("seed", 0)))[:, 1]
        noise = noise.reshape(shape).astype(np.float32)
        # Suppress modulation in near-empty cells and preserve the source's integrated mass.
        detail = max(0.0, min(amount, 2.0)) + max(0.0, min(float(params.get("shredding", 0.0)), 2.0))
        density *= np.maximum(0.0, 1.0 + np.float32(detail) * noise)
    if factor > 1:
        ratio = float(factor ** 3)
        density = _mass_match(density, float(np.sum(source.density, dtype=np.float64)) * ratio)
    # Volume density is a per-voxel value; mass is density times voxel volume. Finer voxels
    # therefore carry the original mass when their density sum remains unchanged.
    voxel = float(source.voxel_size) / factor
    return type(source)(density, voxel_size=voxel, origin=source.origin, matrix=source.matrix,
                        temperature=temperature, velocity=velocity, flame=flame, fuel=fuel, frame=int(frame))


def run_key(source_stream, params, reinject=False):
    body = [getattr(source_stream, "run", None), params]
    if reinject:
        body.append("reinject")      # frames that carry the coarse cache's new content differ from the older carry-only frames
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def cached_upres(source, params, frame, store, cancel=None, guide_velocity=None, previous=None,
                 previous_source=None, reinject=False):
    """Serve an up-res frame from its own bounded SimCache.

    `reinject` (the evaluator sets it) takes each frame's flame, temperature, fuel and new smoke from the coarse cache instead of only
    carrying the first frame's state forward, and `previous_source` is the previous coarse frame (see `upres_sparse_grid`)."""
    if cancel is not None:
        cancel.check()
    key = run_key(getattr(source, "stream", None), params, reinject)
    got = store.get(key, int(frame))
    if got is None:
        sparse_ok = int(params["upres_factor"]) in (2, 4)
        if sparse_ok:
            from .fluid_gpu_solver import Unsupported
            backend = params.get("upres_backend", "auto")
            try:
                grid = upres_sparse_grid(source, params["upres_factor"],
                                         "gpu" if backend in ("gpu", "auto") else "cpu",
                                         guide_velocity, previous, params, previous_source,
                                         int(frame) if reinject else None)
            except Unsupported:
                if backend == "gpu":
                    raise
                grid = upres_sparse_grid(source, params["upres_factor"], "cpu",
                                         guide_velocity, previous, params, previous_source,
                                         int(frame) if reinject else None)
            arrays = grid.arrays()
            meta = {"frame": int(frame), "sparse_shape": grid.shape}
        else:
            out = upres_volume(source, params, frame, guide_velocity, previous)
            arrays = {"density": out.density}
            for name in ("temperature", "velocity", "flame", "fuel"):
                value = getattr(out, name)
                if value is not None:
                    arrays[name] = value
            meta = {"frame": int(frame)}
        store.put(key, int(frame), simcache.State(arrays, meta, copy=False))
        got = store.get(key, int(frame))
    from .scene3d import Volume
    a = got.arrays
    factor = int(params["upres_factor"])
    if "sparse_shape" in got.meta:
        grid = SparseGrid.from_arrays(got.meta["sparse_shape"], a)
        return Volume.from_sparse(grid, voxel_size=float(source.voxel_size) / factor,
                                  origin=source.origin, matrix=source.matrix, frame=int(frame))
    return Volume(a["density"].astype(np.float32), voxel_size=float(source.voxel_size) / factor,
                  origin=source.origin, matrix=source.matrix,
                  temperature=None if "temperature" not in a else a["temperature"].astype(np.float32),
                  velocity=None if "velocity" not in a else a["velocity"].astype(np.float32),
                  flame=None if "flame" not in a else a["flame"].astype(np.float32),
                  fuel=None if "fuel" not in a else a["fuel"].astype(np.float32), frame=int(frame))
