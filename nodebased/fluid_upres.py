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
            "cache_memory_mb": 256, "cache_disk_mb": 2048}


def _active_fine_tiles(source, factor):
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
        lo = np.maximum(0, np.array((cx, cy, cz)) * factor - 1)
        hi = np.minimum(tile_shape, (np.array((cx, cy, cz)) + 1) * factor + 1)
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return np.argwhere(mask).astype(np.int32)


def upres_sparse_grid(source, factor, backend="cpu"):
    """Reconstruct only smoke/fuel-bearing tiles; return compact field blocks.

    This is the first-frame reconstruction path. Guided transport, previous
    fine state and turbulence still use the dense upres_volume path.
    """
    factor = int(factor)
    if factor not in (2, 4):
        raise ValueError("sparse up-res factor must be 2 or 4")
    shape = tuple(n * factor for n in source.density.shape)
    coords = _active_fine_tiles(source, factor)
    fields = {name: getattr(source, name) for name in ("density", "temperature", "flame", "fuel")
              if getattr(source, name) is not None}
    if backend == "gpu":
        from .fluid_upres_gpu import reconstruct_sparse
        blocks = reconstruct_sparse(fields, coords, shape)
    elif backend == "cpu":
        local = np.indices((TILE, TILE, TILE), dtype=np.int32)
        xyz = [coords[:, axis, None, None, None] * TILE + local[axis] for axis in range(3)]
        valid = np.ones((len(coords), TILE, TILE, TILE), dtype=bool)
        for axis in range(3):
            valid &= xyz[axis] < shape[axis]
        sample = [(p.astype(np.float32) * np.float32((source.density.shape[i] - 1) / (shape[i] - 1)))
                  for i, p in enumerate(xyz)]
        blocks = {}
        for name, field in fields.items():
            block = fluid3d.trilerp(field, *sample).astype(np.float32)
            block[~valid] = 0
            blocks[name] = block
    else:
        raise ValueError("sparse up-res backend must be cpu or gpu")
    # Candidate halo tiles may be wholly empty after interpolation. Do not
    # retain or cache them, even if another channel has nonzero background.
    occupied = np.any(blocks["density"] != 0, axis=(1, 2, 3))
    if "fuel" in blocks:
        occupied |= np.any(blocks["fuel"] != 0, axis=(1, 2, 3))
    coords = coords[occupied]
    blocks = {name: block[occupied] for name, block in blocks.items()}
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


def run_key(source_stream, params):
    body = json.dumps([getattr(source_stream, "run", None), params], sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


def cached_upres(source, params, frame, store, cancel=None, guide_velocity=None, previous=None):
    """Serve an up-res frame from its own bounded SimCache."""
    if cancel is not None:
        cancel.check()
    key = run_key(getattr(source, "stream", None), params)
    got = store.get(key, int(frame))
    if got is None:
        sparse_ok = (int(params["upres_factor"]) in (2, 4) and previous is None
                     and guide_velocity is None and source.velocity is None
                     and not float(params.get("turbulence", 0))
                     and not float(params.get("shredding", 0)))
        if sparse_ok:
            from .fluid_gpu_solver import Unsupported
            backend = params.get("upres_backend", "auto")
            try:
                grid = upres_sparse_grid(source, params["upres_factor"],
                                         "gpu" if backend in ("gpu", "auto") else "cpu")
            except Unsupported:
                if backend == "gpu":
                    raise
                grid = upres_sparse_grid(source, params["upres_factor"], "cpu")
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
