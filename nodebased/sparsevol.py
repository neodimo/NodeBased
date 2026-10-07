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

    # -- GPU layout ---------------------------------------------------------------------------------
    def atlas_layout(self):
        """(ax, ay, az): how many tiles the GPU atlas holds along each axis; slot i of the tile list sits at
        (i % ax, (i // ax) % ay, i // (ax * ay)). Near-cubic, with the last layer partly empty."""
        count = max(self.tile_count, 1)
        side = max(1, int(np.ceil(np.cbrt(count) - 1e-9)))
        return side, side, -(-count // (side * side))

    def gpu_index(self):
        """The indirection table as float32 (ntz, nty, ntx, 4), x fastest as a 3D texture wants it: for a stored tile the
        voxel origin of its atlas slot in xyz and 1 in w, for an empty tile zeros. Integers travel as float values
        (exact below 2^24), so the shader needs no division to find a tile."""
        nt = tuple(-(-n // self.tile) for n in self.shape)
        out = np.zeros((nt[2], nt[1], nt[0], 4), np.float32)
        if len(self.coords):
            ax, ay, _az = self.atlas_layout()
            slot = np.arange(len(self.coords))
            origin = np.stack((slot % ax, (slot // ax) % ay, slot // (ax * ay)), axis=1) * self.tile
            out[self.coords[:, 2], self.coords[:, 1], self.coords[:, 0], :3] = origin
            out[self.coords[:, 2], self.coords[:, 1], self.coords[:, 0], 3] = 1.0
        return out

    def gpu_atlas(self, name, channels=None):
        """Field `name`'s stored tiles as one float32 array (az*tile, ay*tile, ax*tile[, channels]) laid out as
        `atlas_layout` says, x fastest; no tile is expanded and the empty region is never built. A vector field is
        padded to `channels` (4 for an rgba32float texture)."""
        ax, ay, az = self.atlas_layout()
        t = self.tile
        block = self.data[name]
        extra = block.shape[4:]
        width = (channels or extra[0],) if extra else ()
        slots = np.zeros((ax * ay * az, t, t, t) + width, np.float32)
        if extra:
            slots[:len(block), ..., :extra[0]] = block
        else:
            slots[:len(block)] = block
        order = (0, 3, 2, 1) + tuple(range(4, slots.ndim))
        slots = slots.transpose(order).reshape((az, ay, ax, t, t, t) + slots.shape[4:])
        axes = (0, 3, 1, 4, 2, 5) + tuple(range(6, slots.ndim))
        return np.ascontiguousarray(slots.transpose(axes).reshape((az * t, ay * t, ax * t) + slots.shape[6:]))

    def max(self, name):
        """The largest value of field `name` over the grid (the rest value counts where a tile is absent)."""
        block = self.data[name]
        nt = -(-np.asarray(self.shape) // self.tile)
        peak = -np.inf
        if len(self.coords):
            t = self.tile
            for index, coord in enumerate(self.coords):
                limit = [min(t, n - int(c) * t) for c, n in zip(coord, self.shape)]
                peak = max(peak, float(np.max(block[index, :limit[0], :limit[1], :limit[2]])))
        if len(self.coords) < int(np.prod(nt)) or self.tile_count == 0:
            peak = max(peak, self.rest[name])
        return peak

    def arrays(self):
        """The grid as plain arrays for a cache: `coords` plus one block array per field."""
        return {"coords": self.coords, **self.data}

    @classmethod
    def from_arrays(cls, shape, arrays, tile=TILE, rest=None):
        return cls(shape, arrays["coords"], {k: v for k, v in arrays.items() if k != "coords"}, tile, rest)

    def sample_trilinear(self, name, points):
        """Zero-padded trilinear samples at cell-centred index positions without expanding tiles."""
        points = np.asarray(points, np.float64)
        output_shape = points.shape[:-1]
        points = points.reshape(-1, 3)
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
        return out.reshape(output_shape + ((3,) if vector else ()))


class SparseField:
    """Array-compatible view of one sparse tile field; NumPy conversion is an explicit dense fallback."""
    def __init__(self, grid, name):
        self.grid, self.name = grid, name
        self.shape = grid.shape + grid.data[name].shape[4:]
        self.ndim = len(self.shape)
        self.dtype = grid.data[name].dtype

    @property
    def nbytes(self):
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def storage_nbytes(self):
        """Bytes retained for this field's packed tile blocks."""
        return self.grid.data[self.name].nbytes

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        """Reduce a scalar sparse field without materializing its dense rest region."""
        if axis is not None or out is not None or keepdims or where is not True:
            return np.sum(np.asarray(self), axis=axis, dtype=dtype, out=out,
                          keepdims=keepdims, initial=initial, where=where)
        rest = self.grid.rest[self.name]
        result_dtype = np.dtype(dtype) if dtype is not None else np.result_type(self.dtype, np.intp)
        total = np.asarray(rest * int(np.prod(self.shape, dtype=np.int64)), dtype=result_dtype)[()]
        tile = self.grid.tile
        for index, (cx, cy, cz) in enumerate(self.grid.coords):
            limits = tuple(min(tile, n - int(c) * tile) for c, n in zip((cx, cy, cz), self.shape))
            block = self.grid.data[self.name][index, :limits[0], :limits[1], :limits[2]]
            total += np.sum(block - rest, dtype=result_dtype)
        return total + initial

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, where=True):
        if axis is not None or out is not None or keepdims or where is not True:
            return np.mean(np.asarray(self), axis=axis, dtype=dtype, out=out,
                           keepdims=keepdims, where=where)
        return self.sum(dtype=dtype) / int(np.prod(self.shape, dtype=np.int64))

    def sparse_sample(self, points):
        return self.grid.sample_trilinear(self.name, points)

    def max(self):
        """The largest value, from the stored tiles and the rest value, without building the dense array."""
        return self.grid.max(self.name)

    def __array__(self, dtype=None, copy=None):
        array = self.grid.to_dense()[self.name]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array.copy() if copy else array
