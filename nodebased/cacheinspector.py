"""Per-frame inspection of a `simcache.SimCache` run, for FluidCache3D and ParticleCache3D.

Everything here is generic over what a solved `simcache.State` actually carries (particle arrays
or fluid grids), the same way the slice viewer builds its field list from the `Volume` in hand
rather than a hard-coded one: a frame's stats come from whatever named arrays its `State` holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TILE = 8            # coarse block edge for the "active tiles" occupancy count
TILE_THRESHOLD = 1e-6


def _active_tile_count(density: np.ndarray) -> int:
    nx, ny, nz = density.shape
    count = 0
    for i in range(0, nx, TILE):
        for j in range(0, ny, TILE):
            for k in range(0, nz, TILE):
                block = density[i:i + TILE, j:j + TILE, k:k + TILE]
                if block.size and np.any(np.abs(block) > TILE_THRESHOLD):
                    count += 1
    return count


def frame_stats(state) -> dict:
    """Per-field min/max/mean plus voxel/particle/active-tile counts for one solved `State`."""
    fields = {}
    voxel_count = None
    particle_count = None
    for name, array in state.arrays.items():
        array = np.asarray(array)
        if array.size == 0:
            continue
        fields[name] = {"min": float(np.min(array)), "max": float(np.max(array)),
                        "mean": float(np.mean(array))}
        if name == "position" and array.ndim == 2:
            particle_count = int(array.shape[0])
        if array.ndim == 3 and voxel_count is None:
            voxel_count = int(array.size)
    density = state.arrays.get("density")
    active_tiles = (_active_tile_count(np.asarray(density))
                    if density is not None and np.asarray(density).ndim == 3 else None)
    return {"fields": fields, "voxel_count": voxel_count, "particle_count": particle_count,
            "active_tiles": active_tiles, "meta": dict(state.meta)}


@dataclass
class FrameEntry:
    frame: int
    present: bool
    stats: "dict | None" = None
    solve_ms: "float | None" = None
    disk_bytes: "int | None" = None


def list_frames(cache, run: str, start_frame: int, end_frame: int) -> list[FrameEntry]:
    """One `FrameEntry` per frame in `[start_frame, end_frame]`, in order. A frame is present when
    the cache has it (memory or disk); a missing one still gets a row so the artist sees the gap."""
    present = set(cache.frames(run))
    entries = []
    for frame in range(int(start_frame), int(end_frame) + 1):
        if frame not in present:
            entries.append(FrameEntry(frame, present=False))
            continue
        state = cache.get(run, frame)
        if state is None:      # a corrupt file: SimCache.get already discarded the index entry
            entries.append(FrameEntry(frame, present=False))
            continue
        entries.append(FrameEntry(frame, present=True, stats=frame_stats(state),
                                  solve_ms=cache.solve_ms(run, frame), disk_bytes=cache.disk_size(run, frame)))
    return entries


def invalidate_from_frame(cache, run: str, frame: int) -> list[int]:
    """Discard `run`'s cached frames at or after `frame`; the sorted frame numbers removed."""
    return cache.invalidate_from(run, frame)


def cache_folder(cache, run: str) -> "Path | None":
    """The on-disk directory holding `run`'s cached frames, or `None` when the cache has no disk
    tier (memory-only, e.g. `NODEBASED_SIM_CACHE=0`)."""
    if cache.root is None:
        return None
    return cache.root / run[:2] / run
