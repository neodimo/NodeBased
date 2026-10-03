"""Sparse tile representation of volume fields (Lane 6 step D, "Sparse Pyro" style).

A `SparseGrid` keeps only the TILE-cubed tiles of a grid that hold something other than the rest value, as a list of
tile coordinates and one (T, TILE, TILE, TILE[, C]) block per field. `to_dense` rebuilds the full arrays on demand
(the ray marcher and the passes read dense arrays); `from_dense` and `to_dense` round-trip exactly. The GPU-resident
solver stores its cache frames this way, so a smoke column in a large box costs the column and not the box.
"""
from __future__ import annotations

import numpy as np

TILE = 8


class SparseGrid:
    def __init__(self, shape, coords, data, tile=TILE, rest=None):
        self.shape = tuple(int(n) for n in shape)
        self.tile = int(tile)
        self.coords = np.ascontiguousarray(coords, np.int32).reshape(-1, 3)
        self.data = {name: np.asarray(block) for name, block in data.items()}
        self.rest = {name: float((rest or {}).get(name, 0.0)) for name in self.data}
        self._lookup = {tuple(int(v) for v in coord): i for i, coord in enumerate(self.coords)}
        for name, block in self.data.items():
            if block.shape[:4] != (len(self.coords), tile, tile, tile):
                raise ValueError(f"field {name!r} has shape {block.shape}, not (tiles, {tile}, {tile}, {tile}, ...)")

    # -- building -----------------------------------------------------------------------------------
    @classmethod
    def from_dense(cls, fields, tile=TILE, rest=None, threshold=0.0):
        """The tiles of `fields` (name -> (nx, ny, nz[, C]) array) where any value differs from its rest value by more
        than `threshold`. Every field shares the tile list."""
        rest = rest or {}
        first = next(iter(fields.values()))
        shape = first.shape[:3]
        nt = tuple(-(-n // tile) for n in shape)
        active = np.zeros(nt, bool)
        padded = tuple(t * tile for t in nt)
        for name, array in fields.items():
            diff = np.abs(np.asarray(array, np.float32) - np.float32(rest.get(name, 0.0)))
            if diff.ndim == 4:
                diff = diff.max(axis=3)
            if padded != shape:
                big = np.zeros(padded, np.float32)
                big[:shape[0], :shape[1], :shape[2]] = diff
                diff = big
            blocks = diff.reshape(nt[0], tile, nt[1], tile, nt[2], tile).max(axis=(1, 3, 5))
            active |= blocks > threshold
        coords = np.argwhere(active).astype(np.int32)
        data = {}
        for name, array in fields.items():
            array = np.asarray(array)
            fill = np.float32(rest.get(name, 0.0))
            extra = array.shape[3:]
            big = np.full(padded + extra, fill, array.dtype)
            big[:shape[0], :shape[1], :shape[2]] = array
            blocks = big.reshape((nt[0], tile, nt[1], tile, nt[2], tile) + extra)
            blocks = blocks.transpose((0, 2, 4, 1, 3, 5) + tuple(range(6, 6 + len(extra))))
            data[name] = np.ascontiguousarray(blocks[active])
        return cls(shape, coords, data, tile, rest)

    # -- using --------------------------------------------------------------------------------------
    @property
    def tile_count(self):
        return len(self.coords)

    @property
    def nbytes(self):
        return int(self.coords.nbytes + sum(block.nbytes for block in self.data.values()))

    def mask(self):
        """Bool array over the tile grid: True where a tile is stored."""
        nt = tuple(-(-n // self.tile) for n in self.shape)
        out = np.zeros(nt, bool)
        if len(self.coords):
            out[self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]] = True
        return out

    def to_dense(self):
        """name -> full array, the rest value everywhere no tile is stored (same dtype as the stored blocks)."""
        t = self.tile
        nt = tuple(-(-n // t) for n in self.shape)
        out = {}
        for name, block in self.data.items():
            extra = block.shape[4:]
            grid = np.full((nt[0], nt[1], nt[2], t, t, t) + extra, self.rest[name], block.dtype)
            if len(self.coords):
                grid[self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]] = block
            dense = grid.transpose((0, 3, 1, 4, 2, 5) + tuple(range(6, 6 + len(extra))))
            dense = dense.reshape((nt[0] * t, nt[1] * t, nt[2] * t) + extra)
            out[name] = np.ascontiguousarray(dense[:self.shape[0], :self.shape[1], :self.shape[2]])
        return out

    def arrays(self):
        """The grid as plain arrays for a cache: `coords` plus one block array per field."""
        return {"coords": self.coords, **self.data}

    @classmethod
    def from_arrays(cls, shape, arrays, tile=TILE, rest=None):
        return cls(shape, arrays["coords"], {k: v for k, v in arrays.items() if k != "coords"}, tile, rest)

    def sample_trilinear(self, name, points):
        """Zero-padded trilinear samples at cell-centred index positions without expanding tiles."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        base = np.floor(points).astype(np.int64)
        frac = points - base
        block = self.data[name]
        vector = block.ndim == 5
        out = np.zeros((len(points), 3), np.float64) if vector else np.zeros(len(points), np.float64)
        tile = self.tile
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    index = base + (dx, dy, dz)
                    valid = np.all((index >= 0) & (index < np.asarray(self.shape)), axis=1)
                    weights = ((frac[:, 0] if dx else 1.0 - frac[:, 0])
                               * (frac[:, 1] if dy else 1.0 - frac[:, 1])
                               * (frac[:, 2] if dz else 1.0 - frac[:, 2]))
                    rows = np.flatnonzero(valid)
                    if not len(rows):
                        continue
                    tile_coords = index[rows] // tile
                    unique, inverse = np.unique(tile_coords, axis=0, return_inverse=True)
                    for group, coord in enumerate(unique):
                        group_rows = rows[inverse == group]
                        tile_index = self._lookup.get(tuple(int(v) for v in coord))
                        if tile_index is None:
                            out[group_rows] += weights[group_rows] * self.rest[name]
                        else:
                            local = index[group_rows] % tile
                            values = block[tile_index, local[:, 0], local[:, 1], local[:, 2]]
                            out[group_rows] += weights[group_rows, None] * values if vector else weights[group_rows] * values
        return out


class SparseField:
    """Array-compatible view of one sparse tile field; NumPy conversion is an explicit dense fallback."""
    def __init__(self, grid, name):
        self.grid, self.name = grid, name
        self.shape = grid.shape + grid.data[name].shape[4:]
        self.ndim = len(self.shape)
        self.dtype = grid.data[name].dtype

    @property
    def nbytes(self):
        return self.grid.data[self.name].nbytes

    def sparse_sample(self, points):
        return self.grid.sample_trilinear(self.name, points)

    def __array__(self, dtype=None, copy=None):
        array = self.grid.to_dense()[self.name]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array.copy() if copy else array
