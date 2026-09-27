"""Read OpenVDB `.vdb` files (Houdini Pyro, Blender) into the scene's `Volume` member, in-house.

No pip wheel exists for OpenVDB (docs/FLUIDS_SPIKE.md), so this is a small reader for the subset that
smoke caches use, written from the public OpenVDB sources (`io/Archive.cc`, `io/File.cc`,
`io/GridDescriptor.cc`, `io/Compression.h`, `tree/RootNode.h`, `tree/InternalNode.h`,
`tree/LeafNode.h`, `math/Maps.h`; https://github.com/AcademySoftwareFoundation/openvdb):

* file versions 222 to 225 (OpenVDB 3.0 and later; per-grid compression flags, node-mask compression),
* the default tree layout (root, 5, 4, 3 log2 dims: 32768 and 4096 slot internal nodes, 8x8x8 leaves),
* scalar `float` and `half` (and `double`) grids and `vec3s` (and `vec3d`) grids, stored as float32 or,
  when the grid carries the `_HalfFloat` suffix, as 16-bit halves,
* uncompressed and zip-compressed value buffers, with or without active-mask compression,
* linear maps only: scale, uniform scale, translation, scale-translate, affine, unitary and compounds
  of those. A `NonlinearFrustumMap` (camera-space grids) is refused by name,
* Blosc-compressed buffers (what Blender and Houdini write by default): a pure-Python decoder for the
  LZ4 codec with byte shuffle, the only combination OpenVDB writes. Other Blosc codecs are refused,
* files written with or without the grid offset table (Blender streams its caches without one).

A grid is read into a dense float32 array over the bounding box of its active voxels: inactive voxels
are zero, constant active tiles are filled, values are never taken from inactive voxels. One grid is
decoded at a time from a seekable stream, and the result is deterministic.

The writer half (`write_vdb`) produces exactly the layout the reader consumes (dense leaves, optional
constant tiles, zip or no compression, every mask-compression form, half storage), so the tests can
build their own assets with no external tool. `write_scene` (`WriteVDB3D`, `nodebased/vdbexport.py`)
is the same writer pointed at a fluid scene: a solved `Volume`'s density, temperature, velocity and
flame as fog volumes, or a liquid's signed-distance surface as a narrow-band level set. Only `zip` and
`none` compress for real here; `blosc` writes a valid, uncompressed Blosc container (this module's own
Blosc encoder is a decoder only), so it exists for reader coverage, not smaller files.
"""
from __future__ import annotations

import hashlib
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x56444220
MIN_FILE_VERSION = 222
MAX_FILE_VERSION = 225
COMPRESS_ZIP, COMPRESS_ACTIVE_MASK, COMPRESS_BLOSC = 1, 2, 4
DEFAULT_MAX_VOXELS = 1 << 27           # 134,217,728 voxels: 512 MiB of float32 for one scalar grid

# Per-node "what was stored for the inactive values" byte (Compression.h).
NO_MASK_OR_INACTIVE_VALS, NO_MASK_AND_MINUS_BG, NO_MASK_AND_ONE_INACTIVE_VAL = 0, 1, 2
MASK_AND_NO_INACTIVE_VALS, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS = 3, 4, 5
NO_MASK_AND_ALL_VALS = 6

BLOSC_MAX_OVERHEAD = 16
SUPPORTED_TYPES = "float, half, double, vec3s, vec3d"
# tree value type name -> (numpy dtype of one component, components)
_VALUE_TYPES = {"float": ("<f4", 1), "half": ("<f2", 1), "double": ("<f8", 1),
                "vec3s": ("<f4", 3), "vec3d": ("<f8", 3)}
_INT_TYPES = {"int32": ("<i4", 1)}     # writer-only: lets tests build a grid class the reader must refuse
_TREE_NAME = re.compile(r"^Tree_(.+?)_(\d+)_(\d+)_(\d+)$")
_HALF_SUFFIX = "_HalfFloat"
_SEP = "\x1e"                          # OpenVDB's "record separator" between a grid name and its instance tag


class VdbError(ValueError):
    """A file this reader cannot or will not read. The message says which limit it hit."""


# ------------------------------------------------------------------------------------------ reading

class _Reader:
    def __init__(self, handle, path):
        self.f, self.path = handle, str(path)

    def take(self, count):
        data = self.f.read(count)
        if len(data) != count:
            raise VdbError(f"{self.path}: unexpected end of file (a truncated .vdb?)")
        return data

    def u8(self):
        return self.take(1)[0]

    def i8(self):
        return struct.unpack("<b", self.take(1))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def i64(self):
        return struct.unpack("<q", self.take(8))[0]

    def string(self):
        return self.take(self.u32()).decode("utf-8", "replace")

    def tell(self):
        return self.f.tell()

    def seek(self, position):
        self.f.seek(position)


@dataclass
class GridDescriptor:
    """One entry of the file's grid table: where the grid's bytes are and what tree type it holds."""
    name: str
    grid_type: str
    half_float: bool
    instance_parent: str
    grid_pos: int
    block_pos: int
    end_pos: int


@dataclass
class VdbHeader:
    version: int
    library: tuple
    has_offsets: bool
    metadata: dict
    grids: list
    count: int = 0            # grids in the file; a file without offsets lists them only as it is walked


@dataclass
class GridInfo:
    """What the file says about one grid without decoding its voxels."""
    name: str
    grid_type: str            # e.g. "Tree_float_5_4_3"
    value_type: str           # "float", "vec3s", "int32", ...
    half_float: bool
    supported: bool
    grid_class: str = ""      # "fog volume", "level set", "staggered" or "unknown"
    bbox: tuple | None = None  # ((imin), (imax)) of the active voxels when the file records it
    voxel_count: int | None = None

    @property
    def is_vector(self):
        return self.value_type.startswith("vec3")


@dataclass(eq=False)
class VdbGrid:
    """One grid as a dense array plus where it sits.

    `data` is float32 (nx, ny, nz) or (nx, ny, nz, 3). Voxel [i, j, k] is the grid's index voxel
    `index_min + (i, j, k)`, centred at index position `index_min + (i, j, k)`. `matrix` is the
    column-vector transform from index space to the file's world space (float64, 4x4).
    """
    name: str
    data: np.ndarray
    index_min: tuple
    matrix: np.ndarray
    transform: str
    half_float: bool = False
    active_voxels: int = 0


def _open_header(reader):
    if reader.take(8) != struct.pack("<q", MAGIC):
        raise VdbError(f"{reader.path} is not a VDB file")
    version = reader.u32()
    if version < MIN_FILE_VERSION:
        raise VdbError(f"{reader.path}: VDB file version {version} is older than {MIN_FILE_VERSION} "
                       "(OpenVDB 3.0); re-save it from Houdini or Blender to read it here")
    library = (reader.u32(), reader.u32())
    has_offsets = bool(reader.u8())
    reader.take(36)                                    # the file's uuid, as ASCII
    metadata = _read_metadata(reader)
    count = reader.i32()
    grids = []
    if has_offsets:
        for _ in range(count):
            grids.append(_read_descriptor(reader))
            reader.seek(grids[-1].end_pos)
    return VdbHeader(version, library, has_offsets, metadata, grids, count)


def _read_descriptor(reader):
    name = reader.string()
    grid_type = reader.string()
    parent = reader.string()
    half = grid_type.endswith(_HALF_SUFFIX)
    if half:
        grid_type = grid_type[:-len(_HALF_SUFFIX)]
    grid_pos, block_pos, end_pos = reader.i64(), reader.i64(), reader.i64()
    return GridDescriptor(name.split(_SEP)[0], grid_type, half, parent, grid_pos, block_pos, end_pos)


def _read_metadata(reader):
    """A metadata table: name, type name and a length-prefixed payload each. Common types are decoded."""
    out = {}
    for _ in range(reader.u32()):
        name, type_name = reader.string(), reader.string()
        payload = reader.take(reader.u32())
        try:
            if type_name == "string":
                out[name] = payload.decode("utf-8", "replace")
            elif type_name == "vec3i":
                out[name] = struct.unpack("<3i", payload)
            elif type_name == "int64":
                out[name] = struct.unpack("<q", payload)[0]
            elif type_name == "int32":
                out[name] = struct.unpack("<i", payload)[0]
            elif type_name == "bool":
                out[name] = bool(payload[0])
        except (struct.error, IndexError):
            pass
    return out


def _value_spec(grid_type):
    match = _TREE_NAME.match(grid_type)
    if not match:
        raise VdbError(f"unsupported VDB grid class {grid_type!r}: ReadVDB3D reads {SUPPORTED_TYPES} grids "
                       "with the default 5-4-3 tree")
    value_type, dims = match.group(1), tuple(int(match.group(i)) for i in (2, 3, 4))
    if value_type not in _VALUE_TYPES:
        raise VdbError(f"unsupported VDB grid class {grid_type!r}: ReadVDB3D reads {SUPPORTED_TYPES} grids")
    if dims != (5, 4, 3):
        raise VdbError(f"unsupported VDB tree layout {grid_type!r}: only the default 5-4-3 tree is read")
    return value_type, dims


def _supported(grid_type):
    try:
        _value_spec(grid_type)
        return True
    except VdbError:
        return False


def _make_info(descriptor, meta):
    match = _TREE_NAME.match(descriptor.grid_type)
    info = GridInfo(descriptor.name, descriptor.grid_type, match.group(1) if match else descriptor.grid_type,
                    descriptor.half_float, _supported(descriptor.grid_type), meta.get("class", ""))
    if "file_bbox_min" in meta and "file_bbox_max" in meta:
        info.bbox = (tuple(meta["file_bbox_min"]), tuple(meta["file_bbox_max"]))
    info.voxel_count = meta.get("file_voxel_count")
    return info


def _walk(reader, header, wanted=None, max_voxels=DEFAULT_MAX_VOXELS):
    """Yield `(info, grid)` per grid in file order; `grid` is decoded only for the grid called `wanted`.

    A file with an offset table is stepped over with seeks and yields nothing but metadata for the
    others. A streamed file has no table, so each earlier grid has to be decoded (and dropped) to
    find the next one.
    """
    if header.has_offsets:
        for descriptor in header.grids:
            reader.seek(descriptor.grid_pos)
            if descriptor.name == wanted:
                yield _decode_grid(reader, descriptor, True, max_voxels)
            else:
                reader.u32()
                yield _make_info(descriptor, _read_metadata(reader)), None
        return
    for _ in range(header.count):
        descriptor = _read_descriptor(reader)
        yield _decode_grid(reader, descriptor, descriptor.name == wanted, max_voxels, can_skip=False)


def list_grids(path):
    """`GridInfo` for every grid in the file, in file order."""
    with open(path, "rb") as handle:
        reader = _Reader(handle, path)
        header = _open_header(reader)
        return [info for info, _ in _walk(reader, header)]


def frame_path(pattern, frame):
    """The file for one frame: a plain path, or a padded pattern (`smoke.%04d.vdb`, `smoke.####.vdb`)."""
    if not pattern:
        raise VdbError("ReadVDB3D: choose a VDB file")
    from .media import resolve_source_path
    resolved, exists = resolve_source_path(pattern, frame, "error")
    if not exists:
        raise VdbError(f"VDB file not found: {resolved}")
    return resolved


def grid_choices(pattern):
    """The grid knob's choices for the node panel: "auto", "none", then the names in the file.

    The file is the first one of a sequence pattern (or the plain path); an unreadable or missing file
    just leaves the two fixed choices, so a panel never fails to open on a bad path.
    """
    choices = ["auto", "none"]
    try:
        from .media import is_sequence, nearest_sequence_path
        path = nearest_sequence_path(pattern, 0) if is_sequence(pattern) else pattern
        choices += [n for n in grid_names(path) if n not in choices] if path else []
    except (OSError, ValueError):
        pass
    return choices


def grid_names(path):
    """The grid names in the file, in file order."""
    return [info.name for info in list_grids(path)]


def fingerprint(path):
    """Cache key for a file: its resolved path, size and modification time (a sequence keys per frame)."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise VdbError(f"VDB file not found: {path}")
    stat = file.stat()
    return hashlib.sha256(f"{file.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()


# Every linear map class and how many doubles its body holds after the type name (Maps.h).
_MAP_DOUBLES = {"AffineMap": 16, "UnitaryMap": 16, "ScaleMap": 15, "UniformScaleMap": 15,
                "TranslationMap": 3, "ScaleTranslateMap": 18, "UniformScaleTranslateMap": 18}


def _read_transform(reader):
    """The grid's index-to-world map as a 4x4 column-vector matrix (float64) and its type name."""
    type_name = reader.string()
    if "NonlinearFrustumMap" in type_name:
        raise VdbError("frustum-transform VDB grids are not supported (camera-space grids need a "
                       "resample to a linear grid in Houdini first)")
    parts = type_name.split(":")
    if any(part not in _MAP_DOUBLES for part in parts):
        raise VdbError(f"unsupported VDB transform {type_name!r}: only linear maps are read")
    if len(parts) == 2:                                # a compound leads with the composed affine matrix
        body = np.frombuffer(reader.take(8 * 16), "<f8").reshape(4, 4)
        reader.take(8 * (_MAP_DOUBLES[parts[0]] + _MAP_DOUBLES[parts[1]]))
        return body.T.copy(), type_name
    doubles = np.frombuffer(reader.take(8 * _MAP_DOUBLES[type_name]), "<f8")
    matrix = np.eye(4)
    if type_name in ("AffineMap", "UnitaryMap"):
        matrix = doubles.reshape(4, 4).T.copy()        # OpenVDB stores row vectors: the translation is row 3
    elif type_name in ("ScaleMap", "UniformScaleMap"):
        matrix[range(3), range(3)] = doubles[0:3]
    elif type_name == "TranslationMap":
        matrix[:3, 3] = doubles[0:3]
    else:                                              # ScaleTranslate: translation, then the scale
        matrix[:3, 3] = doubles[0:3]
        matrix[range(3), range(3)] = doubles[3:6]
    return matrix, type_name


class _TreeReader:
    """Decodes one grid's tree: topology first, then the leaf buffers in the same depth-first order."""

    def __init__(self, reader, value_type, half_float, compression):
        self.r = reader
        self.value_type = value_type
        dtype, self.ncomp = _VALUE_TYPES[value_type]
        self.native = np.dtype(dtype)
        # `_HalfFloat` grids store float values as 16-bit halves; a native `half` grid already is one.
        self.storage = np.dtype("<f2") if half_float and value_type != "half" else self.native
        self.value_bytes = self.native.itemsize * self.ncomp     # sizeof(ValueT): inactive values, tiles
        self.compression = compression
        self.leaves = []                               # [origin, leaf_bits, values] in file order
        self.tiles = []                                # (origin, size, value)

    # -- values
    def _data(self, count):
        """`count` stored elements as float32 (count, ncomp), decoding zip or refusing blosc."""
        nbytes = count * self.ncomp * self.storage.itemsize
        r = self.r
        if nbytes == 0:
            # A zero-length zip or blosc chunk still writes its 8-byte size (0); a half-float read skips it.
            if self.compression & (COMPRESS_BLOSC | COMPRESS_ZIP) and self.storage == self.native:
                empty = r.i64()
                if empty > 0:                          # Blosc still writes its 16-byte header for zero bytes
                    r.take(empty)
            return np.zeros((0, self.ncomp), np.float32)
        if self.compression & (COMPRESS_BLOSC | COMPRESS_ZIP):
            size = r.i64()
            if size <= 0:                              # a negative chunk size marks stored (uncompressed) bytes
                if -size != nbytes:
                    raise VdbError(f"{r.path}: expected a {nbytes}-byte chunk, got {-size} bytes")
                raw = r.take(nbytes)
            elif self.compression & COMPRESS_BLOSC:
                raw = blosc_decompress(r.take(size), nbytes)
            else:
                packed = r.take(size)
                try:
                    raw = zlib.decompress(packed)
                except zlib.error as error:
                    raise VdbError(f"{r.path}: corrupt zip chunk ({error})") from None
                if len(raw) != nbytes:
                    raise VdbError(f"{r.path}: zip chunk decompressed to {len(raw)} bytes, expected {nbytes}")
        else:
            raw = r.take(nbytes)
        array = np.frombuffer(raw, self.storage).reshape(count, self.ncomp)
        return array.astype(np.float32)

    def _skip_value(self):
        self.r.take(self.value_bytes)

    def read_mask(self, nbits):
        return np.unpackbits(np.frombuffer(self.r.take(nbits // 8), np.uint8), bitorder="little").astype(bool)

    def read_values(self, count, active):
        """A node's value buffer as float32 (count, ncomp): active values in place, everything else zero."""
        r = self.r
        meta = r.i8()
        if not 0 <= meta <= NO_MASK_AND_ALL_VALS:
            raise VdbError(f"{r.path}: corrupt node (mask-compression byte {meta})")
        if meta in (NO_MASK_AND_ONE_INACTIVE_VAL, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS):
            self._skip_value()
            if meta == MASK_AND_TWO_INACTIVE_VALS:
                self._skip_value()
        if meta in (MASK_AND_NO_INACTIVE_VALS, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS):
            r.take(count // 8)                         # the inside/outside selection mask: inactive values only
        n_on = int(active.sum())
        if self.compression & COMPRESS_ACTIVE_MASK and meta != NO_MASK_AND_ALL_VALS and n_on != count:
            packed = self._data(n_on)
            values = np.zeros((count, self.ncomp), np.float32)
            values[active] = packed
            return values
        values = self._data(count)
        if n_on != count:
            values[~active] = 0.0
        return values

    # -- topology
    def read_root(self, dims):
        r = self.r
        r.take(self.value_bytes)                       # background
        n_tiles, n_children = r.u32(), r.u32()
        root_size = 1 << sum(dims)
        for _ in range(n_tiles):
            origin = struct.unpack("<3i", r.take(12))
            value = np.frombuffer(r.take(self.value_bytes), self.native).astype(np.float32)
            if r.u8():
                self.tiles.append((origin, root_size, value))
        for _ in range(n_children):
            origin = struct.unpack("<3i", r.take(12))
            self.read_internal(0, dims, origin)

    def read_internal(self, level, dims, origin):
        log2 = dims[level]
        count = 1 << (3 * log2)
        child_log2 = sum(dims[level + 1:])
        child_size = 1 << child_log2
        child_mask = self.read_mask(count)
        value_mask = self.read_mask(count)
        values = self.read_values(count, value_mask)
        mask = (1 << log2) - 1
        for pos in np.flatnonzero(value_mask & ~child_mask):
            corner = (origin[0] + (pos >> (2 * log2)) * child_size,
                      origin[1] + ((pos >> log2) & mask) * child_size,
                      origin[2] + (pos & mask) * child_size)
            self.tiles.append((corner, child_size, values[pos]))
        for pos in np.flatnonzero(child_mask):
            corner = (origin[0] + (pos >> (2 * log2)) * child_size,
                      origin[1] + ((pos >> log2) & mask) * child_size,
                      origin[2] + (pos & mask) * child_size)
            if level == len(dims) - 2:                 # the children are leaves: their topology is their mask
                self.leaves.append([corner, self.read_mask(1 << (3 * dims[-1])), None])
            else:
                self.read_internal(level + 1, dims, corner)

    def read_buffers(self, dims):
        leaf_voxels = 1 << (3 * dims[-1])
        for leaf in self.leaves:
            bits = self.read_mask(leaf_voxels)         # the leaf writes its value mask again before its values
            leaf[1] = bits
            leaf[2] = self.read_values(leaf_voxels, bits)


def _decode_grid(reader, descriptor, densify, max_voxels, can_skip=True):
    """Decode the grid whose data starts at the reader's position; returns `(info, VdbGrid or None)`."""
    compression = reader.u32()
    meta = _read_metadata(reader)
    info = _make_info(descriptor, meta)
    if not info.supported:
        if not can_skip:
            raise VdbError(f"{reader.path}: unsupported VDB grid class {descriptor.grid_type!r} in a streamed "
                           "file, which cannot be skipped")
        return info, None
    matrix, transform = _read_transform(reader)
    if descriptor.instance_parent:
        raise VdbError(f"{reader.path}: grid {descriptor.name!r} is an instance of {descriptor.instance_parent!r}; "
                       "instanced trees are not supported")
    value_type, dims = _value_spec(descriptor.grid_type)
    if reader.i32() != 1:
        raise VdbError(f"{reader.path}: multi-buffer VDB trees are not supported")
    tree = _TreeReader(reader, value_type, descriptor.half_float, compression)
    tree.read_root(dims)
    tree.read_buffers(dims)
    if not densify:
        return info, None
    return info, _densify(tree, dims, descriptor.name, matrix, transform, descriptor.half_float, max_voxels,
                          reader.path)


def read_grid(path, name, *, max_voxels=DEFAULT_MAX_VOXELS):
    """Decode the grid called `name` into a dense `VdbGrid` (see the module docstring for the rules)."""
    with open(path, "rb") as handle:
        reader = _Reader(handle, path)
        header = _open_header(reader)
        seen = []
        for info, grid in _walk(reader, header, name, max_voxels):
            seen.append(info.name)
            if info.name == name:
                if not info.supported:
                    _value_spec(info.grid_type)            # raises the "unsupported grid class" error
                return grid
        raise VdbError(f"{path}: no grid named {name!r}; the file has: {', '.join(seen) or 'no grids'}")


def _densify(tree, dims, name, matrix, transform, half_float, max_voxels, path):
    side = 1 << dims[-1]
    lows, highs, active_voxels = [], [], 0
    stacks = []
    if tree.leaves:
        bits = np.stack([leaf[1] for leaf in tree.leaves]).reshape(-1, side, side, side)
        origins = np.array([leaf[0] for leaf in tree.leaves], np.int64)
        present = bits.any(axis=(1, 2, 3))
        active_voxels += int(bits.sum())
        if present.any():
            lo, hi = [], []
            for axis in range(3):
                others = tuple(a for a in (1, 2, 3) if a != axis + 1)
                any_axis = bits.any(axis=others)
                lo.append(any_axis.argmax(axis=1))
                hi.append(side - 1 - any_axis[:, ::-1].argmax(axis=1))
            lows.append((origins + np.stack(lo, axis=1))[present].min(axis=0))
            highs.append((origins + np.stack(hi, axis=1))[present].max(axis=0))
    for corner, size, _ in tree.tiles:
        lows.append(np.array(corner, np.int64))
        highs.append(np.array(corner, np.int64) + size - 1)
        active_voxels += size ** 3
    if not lows:
        raise VdbError(f"{path}: grid {name!r} has no active voxels")
    lo, hi = np.min(lows, axis=0), np.max(highs, axis=0)
    shape = tuple(int(v) for v in (hi - lo + 1))
    if int(np.prod(shape, dtype=np.int64)) > max_voxels:
        raise VdbError(f"{path}: grid {name!r} spans {shape[0]} x {shape[1]} x {shape[2]} voxels "
                       f"({int(np.prod(shape, dtype=np.int64)):,}), over the {max_voxels:,} voxel limit")
    ncomp = tree.ncomp
    dense = np.zeros(shape + ((3,) if ncomp == 3 else ()), np.float32)
    for corner, size, value in tree.tiles:
        a = np.array(corner, np.int64) - lo
        b = a + size
        clip_a, clip_b = np.maximum(a, 0), np.minimum(b, shape)
        if (clip_b > clip_a).all():
            dense[clip_a[0]:clip_b[0], clip_a[1]:clip_b[1], clip_a[2]:clip_b[2]] = value if ncomp == 3 else value[0]
    for corner, leaf_bits, values in tree.leaves:
        if not leaf_bits.any():
            continue
        a = np.array(corner, np.int64) - lo
        b = a + side
        clip_a, clip_b = np.maximum(a, 0), np.minimum(b, shape)
        block = dense[clip_a[0]:clip_b[0], clip_a[1]:clip_b[1], clip_a[2]:clip_b[2]]
        cut = tuple(slice(int(c - o), int(c - o + (e - c))) for c, o, e in zip(clip_a, a, clip_b))
        on = leaf_bits.reshape(side, side, side)[cut]
        source = values.reshape((side, side, side) + ((3,) if ncomp == 3 else ()))[cut]
        block[on] = source[on]
    dense.flags.writeable = False
    return VdbGrid(name, dense, tuple(int(v) for v in lo), matrix, transform, half_float, active_voxels)


# --------------------------------------------------------------------------------------- Blosc

def lz4_block_decompress(source, size):
    """Decode one LZ4 block (https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md) to `size` bytes."""
    out = bytearray()
    i, n = 0, len(source)
    try:
        while i < n:
            token = source[i]
            i += 1
            literals = token >> 4
            if literals == 15:
                while True:
                    extra = source[i]
                    i += 1
                    literals += extra
                    if extra != 255:
                        break
            out += source[i:i + literals]
            i += literals
            if i >= n:
                break                                  # the last sequence carries literals only
            offset = source[i] | (source[i + 1] << 8)
            i += 2
            match = token & 15
            if match == 15:
                while True:
                    extra = source[i]
                    i += 1
                    match += extra
                    if extra != 255:
                        break
            match += 4
            start = len(out) - offset
            if offset == 0 or start < 0:
                raise VdbError("corrupt LZ4 block (bad match offset)")
            if offset >= match:
                out += out[start:start + match]
            else:                                      # an overlapping match repeats the last `offset` bytes
                out += (out[start:] * (match // offset + 1))[:match]
    except IndexError:
        raise VdbError("corrupt LZ4 block (ran past its end)") from None
    if len(out) != size:
        raise VdbError(f"corrupt LZ4 block (decoded {len(out)} bytes, expected {size})")
    return bytes(out)


def _unshuffle(data, typesize):
    """Undo Blosc's byte shuffle: byte b of every element is stored together; a ragged tail is left as is."""
    whole = len(data) - len(data) % typesize
    if typesize <= 1 or whole == 0:
        return data
    body = np.frombuffer(data, np.uint8, count=whole).reshape(typesize, whole // typesize)
    return body.T.tobytes() + data[whole:]


def blosc_decompress(chunk, expected):
    """Decode one Blosc 1.x chunk (header, block start table, per-block streams) to `expected` bytes.

    Supports the LZ4 codec (and stored streams) with or without byte shuffle, and memcpy chunks:
    every combination OpenVDB's `blosc_compress_ctx(..., "lz4", shuffle)` call produces. The block
    split count is not recorded in old chunks, so it is inferred: a block either is one stream or
    `typesize` streams, and only one of the two fits the block's recorded extent.
    """
    if len(chunk) < BLOSC_MAX_OVERHEAD:
        raise VdbError("corrupt Blosc chunk (shorter than its header)")
    flags, typesize = chunk[2], max(chunk[3], 1)
    nbytes, blocksize, cbytes = struct.unpack_from("<III", chunk, 4)
    if nbytes != expected:
        raise VdbError(f"Blosc chunk holds {nbytes} bytes, expected {expected}")
    if flags & 0x2:                                    # memcpy: the bytes follow the header unchanged
        return bytes(chunk[16:16 + nbytes])
    if flags & 0x4:
        raise VdbError("Blosc bit-shuffled VDB data is not supported (OpenVDB writes byte shuffle)")
    codec = flags >> 5
    nblocks = -(-nbytes // blocksize) if blocksize else 0
    starts = struct.unpack_from(f"<{nblocks}I", chunk, 16)
    out = []
    for index in range(nblocks):
        size = min(blocksize, nbytes - index * blocksize)
        begin = starts[index]
        end = starts[index + 1] if index + 1 < nblocks else cbytes
        block = None
        for splits in ((typesize, 1) if typesize <= 16 and size // typesize >= 128 else (1,)):
            streams = _split_streams(chunk, begin, end, size, splits, codec)
            if streams is not None:
                block = streams
                break
        if block is None:
            raise VdbError("corrupt Blosc chunk (block streams do not fit their extent)")
        out.append(_unshuffle(block, typesize) if flags & 0x1 else block)
    return b"".join(out)


def _split_streams(chunk, begin, end, size, splits, codec):
    """The decoded block if `splits` streams tile bytes [begin, end) exactly, else None."""
    each, ragged = divmod(size, splits)
    pieces, position = [], begin
    for split in range(splits):
        if position + 4 > end:
            return None
        (csize,) = struct.unpack_from("<i", chunk, position)
        position += 4
        want = each + (ragged if split == splits - 1 else 0)
        if csize < 0 or position + csize > end:
            return None
        payload = chunk[position:position + csize]
        position += csize
        if csize == want:
            pieces.append(bytes(payload))
        elif codec == 1:
            try:
                pieces.append(lz4_block_decompress(payload, want))
            except VdbError:
                return None
        else:
            raise VdbError(f"Blosc codec {codec} is not supported (OpenVDB writes LZ4)")
    return b"".join(pieces) if position == end else None


# ------------------------------------------------------------------------------- Volume assembly

_AUTO_NAMES = {"density": ("density",), "temperature": ("temperature", "heat"), "velocity": ("vel", "velocity", "v")}


def _select(role, wanted, infos, names):
    """The grid name a knob selects: "none" drops it, "auto" tries the usual names, else it must exist."""
    if wanted == "none":
        return None
    if wanted not in ("", "auto"):
        if wanted not in names:
            raise VdbError(f"no grid named {wanted!r}; the file has: {', '.join(names)}")
        return wanted
    for candidate in _AUTO_NAMES[role]:
        if candidate in names:
            return candidate
    if role == "density":                              # `density_noise`, or a single scalar grid, is the density
        prefixed = [n for n in names if n.startswith("density")]
        if len(prefixed) == 1:
            return prefixed[0]
        scalars = [i.name for i in infos if i.supported and not i.is_vector and i.grid_class != "level set"]
        if len(scalars) == 1:
            return scalars[0]
    return None


def load_volume(path, density="auto", temperature="auto", velocity="auto", voxel_scale=1.0,
                max_voxels=DEFAULT_MAX_VOXELS):
    """Read the chosen grids of one `.vdb` file as a `scene3d.Volume`.

    `density`, `temperature` and `velocity` are grid names, "none" or "auto" (the usual names:
    `density`; `temperature` or `heat`; `vel`, `velocity` or `v`). The grids share one dense box,
    the union of their active boxes. Voxels are cubes of `voxel_size` (the grid transform's scale);
    a transform with rotation or shear goes into `Volume.matrix`, and a plain scale and translation
    become `voxel_size` and `origin`. `voxel_scale` scales the whole volume about the file's own origin
    (positions, voxel size and velocities).
    """
    from .scene3d import Volume
    infos = list_grids(path)
    names = [info.name for info in infos]
    if not names:
        raise VdbError(f"{path} holds no grids")
    density_name = _select("density", density, infos, names)
    if density_name is None:
        raise VdbError(f"{path}: no density grid; the file has: {', '.join(names)}. Choose one in the node")
    chosen = [("density", density_name), ("temperature", _select("temperature", temperature, infos, names)),
              ("velocity", _select("velocity", velocity, infos, names))]
    chosen = [(role, grid_name) for role, grid_name in chosen if grid_name is not None]
    info_by_name = {i.name: i for i in infos}
    for role, grid_name in chosen:
        if role == "velocity" and not info_by_name[grid_name].is_vector:
            raise VdbError(f"grid {grid_name!r} is a scalar grid; the velocity grid must be a Vec3f grid")
        if role != "velocity" and info_by_name[grid_name].is_vector:
            raise VdbError(f"grid {grid_name!r} is a vector grid; the {role} grid must be a scalar FloatGrid")
    grids = {role: read_grid(path, grid_name, max_voxels=max_voxels) for role, grid_name in chosen}
    base = grids["density"]
    for role, grid in grids.items():
        if not np.allclose(grid.matrix, base.matrix, rtol=1e-9, atol=1e-9):
            raise VdbError(f"grids {density_name!r} and {dict(chosen)[role]!r} have different transforms; "
                           "resample them to one grid in Houdini first")
    lo = np.min([g.index_min for g in grids.values()], axis=0)
    hi = np.max([np.array(g.index_min) + g.data.shape[:3] for g in grids.values()], axis=0)
    shape = tuple(int(v) for v in hi - lo)
    if int(np.prod(shape, dtype=np.int64)) > max_voxels:
        raise VdbError(f"the chosen grids span {shape[0]} x {shape[1]} x {shape[2]} voxels together, "
                       f"over the {max_voxels:,} voxel limit")

    def place(grid):
        if grid.data.shape[:3] == shape and tuple(grid.index_min) == tuple(lo):
            return grid.data
        out = np.zeros(shape + grid.data.shape[3:], np.float32)
        a = np.array(grid.index_min) - lo
        b = a + grid.data.shape[:3]
        out[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = grid.data
        return out

    linear = base.matrix[:3, :3]
    translation = base.matrix[:3, 3]
    voxel_size = float(np.mean(np.linalg.norm(linear, axis=0)))
    if not voxel_size > 0:
        raise VdbError(f"{path}: the grid transform has a zero voxel size")
    unit = linear / voxel_size
    scale = float(voxel_scale)
    origin = (lo - 0.5) * voxel_size
    if np.allclose(unit, np.eye(3), atol=1e-9):
        origin, matrix = origin + translation, np.eye(4)
    else:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = unit, translation
    velocity_data = None
    if "velocity" in grids:
        velocity_data = place(grids["velocity"])
        if not np.allclose(unit, np.eye(3), atol=1e-9):
            velocity_data = velocity_data @ np.linalg.inv(unit).T.astype(np.float32)
        velocity_data = velocity_data * np.float32(scale)
    if scale != 1.0:
        matrix[:3, 3] *= scale
    return Volume(place(grids["density"]), voxel_size * scale, tuple(origin * scale), matrix.astype(np.float32),
                  temperature=place(grids["temperature"]) if "temperature" in grids else None,
                  velocity=velocity_data)


def load_scene(path, density="auto", temperature="auto", velocity="auto", voxel_scale=1.0,
               max_voxels=DEFAULT_MAX_VOXELS):
    """`load_volume` wrapped in a `Scene` with one volume (what the ReadVDB3D node outputs)."""
    from .scene3d import Scene
    return Scene(volumes=(load_volume(path, density, temperature, velocity, voxel_scale, max_voxels),))


# ------------------------------------------------------------------------------------------ writing

def _blosc_stored(raw, typesize=4, splits=None):
    """A valid Blosc 1.x chunk holding `raw` with byte shuffle and stored (uncompressed) streams.

    It exercises the reader's container path (header, block table, split streams, unshuffle) without
    an LZ4 encoder; the real LZ4 path is covered by real Blender caches. Under 128 bytes it is a
    memcpy chunk, as Blosc itself does.
    """
    n = len(raw)
    codec = 1 << 5                                     # LZ4
    if n < 128:
        return struct.pack("<BBBBIII", 2, 1, 0x2 | codec, typesize, n, n, n + 16) + raw
    whole = n - n % typesize
    body = np.frombuffer(raw, np.uint8, count=whole).reshape(whole // typesize, typesize)
    shuffled = body.T.tobytes() + raw[whole:]
    splits = splits or (typesize if n // typesize >= 128 else 1)   # Blosc splits a block only when it is big enough
    each, ragged = divmod(n, splits)
    streams = b""
    for split in range(splits):
        size = each + (ragged if split == splits - 1 else 0)
        streams += struct.pack("<i", size) + shuffled[split * each: split * each + size]
    start = 16 + 4
    return struct.pack("<BBBBIII", 2, 1, 0x1 | codec, typesize, n, n, start + len(streams)) \
        + struct.pack("<I", start) + streams


def _pack_mask(bits):
    return np.packbits(np.asarray(bits, bool), bitorder="little").tobytes()


def _metadata_entry(name, type_name, payload):
    def s(text):
        raw = text.encode()
        return struct.pack("<I", len(raw)) + raw
    return s(name) + s(type_name) + struct.pack("<I", len(payload)) + payload


def _value_layout(dtype, vector):
    """The tree value type name and the numpy storage dtype of one component for an input array dtype."""
    kind = np.dtype(dtype)
    if kind == np.float16:
        if vector:
            raise ValueError("write_vdb: a Vec3 grid needs float32 or float64 data (use half=True to store halves)")
        return "half", np.dtype("<f2")
    if kind == np.float32:
        return ("vec3s" if vector else "float"), np.dtype("<f4")
    if kind == np.float64:
        return ("vec3d" if vector else "double"), np.dtype("<f8")
    if kind == np.int32 and not vector:
        return "int32", np.dtype("<i4")
    raise ValueError(f"write_vdb: unsupported grid dtype {kind}")


def _same(a, b):
    return bool(np.array_equal(a, b))


class _Writer:
    """Serialises one grid in the layout `read_grid` consumes (the mirror of Compression.h's writer)."""

    def __init__(self, out, value_dtype, ncomp, half, flags):
        self.out, self.dtype, self.ncomp, self.half, self.flags = out, value_dtype, ncomp, half, flags
        self.stored = np.dtype("<f2") if half and value_dtype.kind == "f" and value_dtype.itemsize > 2 else value_dtype

    def chunk(self, array):
        raw = np.ascontiguousarray(array.astype(self.stored)).tobytes()
        if not raw and self.stored != self.dtype:
            return                                     # a half-float write of zero values writes nothing
        if self.flags & COMPRESS_BLOSC:
            packed = _blosc_stored(raw)
            self.out.write(struct.pack("<q", len(packed)) + packed)
        elif self.flags & COMPRESS_ZIP and not raw:
            self.out.write(struct.pack("<q", 0))
        elif self.flags & COMPRESS_ZIP:
            packed = zlib.compress(raw, 6)
            if len(packed) < len(raw):
                self.out.write(struct.pack("<q", len(packed)) + packed)
            else:
                self.out.write(struct.pack("<q", -len(raw)) + raw)
        else:
            self.out.write(raw)

    def truncated(self, value):
        value = np.asarray(value, self.dtype)
        return value.astype(np.float16).astype(self.dtype) if self.stored.itemsize == 2 and self.dtype.itemsize > 2 else value

    def values(self, buffer, value_mask, child_mask, background):
        """One node's value buffer with the mask-compression metadata byte Compression.h chooses."""
        buffer = np.asarray(buffer, self.dtype).reshape(len(buffer), self.ncomp)
        bg = np.asarray(background, self.dtype).reshape(self.ncomp)
        out = self.out
        if not self.flags & COMPRESS_ACTIVE_MASK:
            out.write(struct.pack("<b", NO_MASK_AND_ALL_VALS))
            self.chunk(buffer)
            return
        inactive = ~value_mask & ~child_mask
        uniques = []
        if inactive.any():                             # distinct inactive values, in order of first appearance
            found, first = np.unique(buffer[inactive], axis=0, return_index=True)
            uniques = [found[i] for i in np.argsort(first)[:3]]
        neg = -bg
        meta, slots = NO_MASK_OR_INACTIVE_VALS, [bg, bg]
        if len(uniques) == 1:
            slots[0] = uniques[0]
            if not _same(uniques[0], bg):
                meta = NO_MASK_AND_MINUS_BG if _same(uniques[0], neg) else NO_MASK_AND_ONE_INACTIVE_VAL
        elif len(uniques) == 2:
            slots = [uniques[0], uniques[1]]
            if not _same(slots[0], bg) and not _same(slots[1], bg):
                meta = MASK_AND_TWO_INACTIVE_VALS
            elif _same(slots[1], bg):
                meta = MASK_AND_NO_INACTIVE_VALS if _same(slots[0], neg) else MASK_AND_ONE_INACTIVE_VAL
            else:
                slots.reverse()
                meta = MASK_AND_NO_INACTIVE_VALS if _same(slots[0], neg) else MASK_AND_ONE_INACTIVE_VAL
        elif len(uniques) > 2:
            meta = NO_MASK_AND_ALL_VALS
        out.write(struct.pack("<b", meta))
        if meta in (NO_MASK_AND_ONE_INACTIVE_VAL, MASK_AND_ONE_INACTIVE_VAL, MASK_AND_TWO_INACTIVE_VALS):
            out.write(self.truncated(slots[0]).tobytes())
            if meta == MASK_AND_TWO_INACTIVE_VALS:
                out.write(self.truncated(slots[1]).tobytes())
        if meta == NO_MASK_AND_ALL_VALS:
            self.chunk(buffer)
            return
        if meta >= MASK_AND_NO_INACTIVE_VALS:
            selection = inactive & np.all(buffer == slots[1], axis=1)
            out.write(_pack_mask(selection))
        self.chunk(buffer[value_mask])


def write_vdb(path, grids, *, voxel_size=1.0, index_min=(0, 0, 0), matrix=None, compression="zip",
              mask_compression=True, half=False, active=None, tiles=None, background=0.0,
              transform=None, grid_class="fog volume", offsets=True):
    """Write dense arrays as a `.vdb` file this module (and OpenVDB) can read: the test-asset writer.

    `grids` maps grid name to an array indexed [i, j, k] (a trailing axis of 3 makes a Vec3 grid).
    Voxel [0, 0, 0] sits at index `index_min`. `active` maps a grid name to a boolean array (default:
    value != 0); values under an inactive voxel are written as given, which is how the tests build
    holes. `tiles` maps a grid name to `[(corner, value), ...]`: constant active 8x8x8 blocks at
    8-aligned index corners. `compression` is "zip", "none" or "blosc" (byte-shuffled, stored streams: a valid chunk, not
    LZ4-compressed); `mask_compression` toggles the
    active-mask flag; `half` stores float grids as 16-bit halves (`_HalfFloat`). `matrix` is an
    optional 4x4 column-vector index-to-world matrix written as an AffineMap; without it the map is a
    uniform scale of `voxel_size`. `transform="frustum"` writes only a NonlinearFrustumMap type tag
    (for the refusal test). `offsets=False` writes the streamed form (no grid offset table, as Blender writes it).
    `grid_class` and `background` are either one value for every grid or a `{name: value}` map (a grid
    missing from the map falls back to "fog volume" and 0.0), so one file can mix, for instance, fog
    volumes with a level-set liquid surface (`write_scene`).
    """
    if compression not in ("zip", "none", "blosc"):
        raise ValueError("write_vdb compression must be 'zip', 'blosc' or 'none'")
    flags = ({"zip": COMPRESS_ZIP, "blosc": COMPRESS_BLOSC, "none": 0}[compression]
             | (COMPRESS_ACTIVE_MASK if mask_compression else 0))
    active, tiles = active or {}, tiles or {}
    parts = [struct.pack("<q", MAGIC), struct.pack("<I", 224), struct.pack("<II", 11, 0), b"\x01" if offsets else b"\x00",
             b"00000000-0000-4000-8000-000000000000", struct.pack("<I", 0), struct.pack("<i", len(grids))]
    header = b"".join(parts)
    body = bytearray(header)
    base_offset = 0
    for name, array in grids.items():
        gc = grid_class.get(name, "fog volume") if isinstance(grid_class, dict) else grid_class
        bg = background.get(name, 0.0) if isinstance(background, dict) else background
        chunk = _grid_bytes(name, np.asarray(array), voxel_size, index_min, matrix, flags, half,
                            active.get(name), tiles.get(name, []), bg, transform, gc)
        descriptor_head, payload = chunk
        start = len(body) + len(descriptor_head) + 24
        block = start + payload[1]
        end = start + len(payload[0])
        body += descriptor_head + struct.pack("<qqq", *((start, block, end) if offsets else (0, 0, 0))) + payload[0]
    Path(path).write_bytes(bytes(body))
    return Path(path)


def _write_transform(volume):
    """The 4x4 `matrix` for `write_vdb` that makes `read_grid`/`load_volume` reconstruct `volume`'s
    own `origin`, `voxel_size` and `matrix` exactly, taking array index 0 as file index 0 (this
    module always writes with `index_min=(0, 0, 0)`, so a grid's own array index is the file index).

    OpenVDB voxel centres sit on integer index positions (`load_volume`'s docstring), so cell `i`
    of `volume.density`, centred in `volume`'s own space at `origin + (i + .5) * voxel_size`, must
    land in world space at `volume.matrix @ (origin + (i + .5) * voxel_size)`. Solving
    `A @ i + b == that`, for every `i`, gives the AffineMap below.
    """
    m = np.asarray(volume.matrix, np.float64)
    linear, translation = m[:3, :3], m[:3, 3]
    voxel_size = float(volume.voxel_size)
    origin = np.asarray(volume.origin, np.float64)
    a = linear * voxel_size
    out = np.eye(4)
    out[:3, :3] = a
    out[:3, 3] = a @ np.full(3, 0.5) + linear @ origin + translation
    return out


def _volume_grids(volume):
    """The fog-volume grids a `Volume` writes: `density` always, `temperature`, `vel` (the vector
    field) and `flame` (the burn rate) only when the solve carries them."""
    grids, classes = {"density": volume.density}, {"density": "fog volume"}
    if volume.temperature is not None:
        grids["temperature"], classes["temperature"] = volume.temperature, "fog volume"
    if volume.velocity is not None:
        grids["vel"], classes["vel"] = volume.velocity, "unknown"
    if volume.flame is not None:
        grids["flame"], classes["flame"] = volume.flame, "fog volume"
    return grids, classes


def write_scene(path, scene, *, compression="zip", half=False, narrow_band=3.0):
    """Write the one `Volume` (or the one liquid surface) of `scene` as a `.vdb` file (`WriteVDB3D`).

    A fluid volume (`scene.volumes`) becomes fog-volume grids: `density`, and `temperature`, `vel`
    (its velocity, a Vec3f grid) and `flame` when the solve carries them. A liquid's signed-distance
    surface (`instance.surface` on `scene.particles`, a `Volume` whose `density` field is really phi,
    negative inside) becomes one `level set` grid named `surface`, active only within `narrow_band`
    voxels of the zero crossing (the narrow band OpenVDB level sets themselves use); the rest reads
    back as the band's own world-space half-width, its background value. A scene with both is refused:
    write them from two WriteVDB3D nodes, since a level set and a fog volume are different caches even
    when they came from the same solve. Every grid in the file shares one voxel size and transform, so
    a scene with more than one of either is refused by name too.
    """
    volumes = [v for v in (getattr(scene, "volumes", None) or ())]
    surfaces = [i.surface for i in (getattr(scene, "particles", None) or ()) if getattr(i, "surface", None) is not None]
    if volumes and surfaces:
        raise VdbError("WriteVDB3D: the scene has both a fluid volume and a liquid surface; "
                       "write them from two WriteVDB3D nodes")
    if len(volumes) > 1:
        raise VdbError(f"WriteVDB3D writes one Volume per file; the scene has {len(volumes)}")
    if len(surfaces) > 1:
        raise VdbError(f"WriteVDB3D writes one liquid surface per file; the scene has {len(surfaces)}")
    if not volumes and not surfaces:
        raise VdbError("WriteVDB3D: the scene has no volume and no liquid surface to write")
    if volumes:
        reference = volumes[0]
        grids, classes = _volume_grids(reference)
        active, background = None, 0.0
    else:
        reference = surfaces[0]
        band = float(narrow_band) * reference.voxel_size
        grids = {"surface": reference.density}
        classes = {"surface": "level set"}
        active = {"surface": np.abs(reference.density) <= band}
        background = {"surface": band}
    write_vdb(path, grids, voxel_size=1.0, matrix=_write_transform(reference), compression=compression,
             half=half, active=active, grid_class=classes, background=background)
    return Path(path)


def _string(text):
    raw = text.encode()
    return struct.pack("<I", len(raw)) + raw


def _grid_bytes(name, array, voxel_size, index_min, matrix, flags, half, active_mask, tiles, background,
                transform, grid_class):
    vector = array.ndim == 4
    if vector and array.shape[3] != 3:
        raise ValueError("write_vdb: a Vec3 grid needs shape (nx, ny, nz, 3)")
    value_type, dtype = _value_layout(array.dtype, vector)
    ncomp = 3 if vector else 1
    is_half = bool(half) and value_type in ("float", "double", "vec3s", "vec3d")
    tree_name = f"Tree_{value_type}_5_4_3"
    active_mask = (np.any(array != 0, axis=-1) if vector else array != 0) if active_mask is None else np.asarray(active_mask, bool)
    if active_mask.shape != array.shape[:3]:
        raise ValueError("write_vdb: an active mask must match the grid shape")
    descriptor = _string(name) + _string(tree_name + (_HALF_SUFFIX if is_half else "")) + _string("")
    out = bytearray()
    out += struct.pack("<I", flags)
    imin = np.array(index_min, np.int64)
    n_active = int(active_mask.sum())
    lo, hi = imin, imin + np.array(array.shape[:3]) - 1
    entries = [_metadata_entry("class", "string", grid_class.encode()),
               _metadata_entry("file_bbox_min", "vec3i", struct.pack("<3i", *[int(v) for v in lo])),
               _metadata_entry("file_bbox_max", "vec3i", struct.pack("<3i", *[int(v) for v in hi])),
               _metadata_entry("file_voxel_count", "int64", struct.pack("<q", n_active))]
    out += struct.pack("<I", len(entries)) + b"".join(entries)
    out += _transform_bytes(voxel_size, matrix, transform)
    writer = _Writer(_Sink(out), np.dtype(dtype), ncomp, is_half, flags)
    tree = _build_tree(array, active_mask, imin, tiles, ncomp)
    bg = np.full(ncomp, background, dtype)
    out += struct.pack("<i", 1)
    out += np.asarray(bg, dtype).tobytes()
    root_children = sorted(tree)
    out += struct.pack("<II", 0, len(root_children))
    leaves = []
    for corner in root_children:
        out += struct.pack("<3i", *corner)
        _write_internal(writer, tree[corner], 0, corner, bg, leaves)
    block = len(out)
    for leaf in leaves:
        writer.out.write(_pack_mask(leaf["active"]))
        writer.values(leaf["values"], leaf["active"], np.zeros(512, bool), bg)
    return descriptor, (bytes(out), block)


class _Sink:
    """A write() target that appends to a bytearray, so the writer can be told the stream position."""

    def __init__(self, target):
        self.target = target

    def write(self, data):
        self.target += data


def _transform_bytes(voxel_size, matrix, transform):
    if transform == "frustum":
        return _string("NonlinearFrustumMap") + bytes(8 * 16)
    if matrix is not None:
        m = np.asarray(matrix, np.float64)
        return _string("AffineMap") + np.ascontiguousarray(m.T).astype("<f8").tobytes()
    scale = np.full(3, float(voxel_size))
    body = scale.tobytes() + scale.tobytes() + (1 / scale).tobytes() + (1 / scale ** 2).tobytes() + (0.5 / scale).tobytes()
    return _string("UniformScaleMap") + body


def _build_tree(array, active_mask, imin, tiles, ncomp):
    """{upper corner: {slot: {slot: leaf dict or tile dict}}} from dense arrays and tiles."""
    tree = {}
    shape = np.array(array.shape[:3])
    lo = imin
    hi = imin + shape
    leaf_lo = (lo // 8) * 8
    leaf_hi = -(-hi // 8) * 8
    flat_values = array.reshape(shape[0], shape[1], shape[2], ncomp)

    def insert(corner, node):
        c = np.array(corner)
        upper = tuple(int(v) for v in (c // 4096) * 4096)
        lower_slot = _slot((c % 4096) // 128, 5)
        leaf_slot = _slot((c % 128) // 8, 4)
        tree.setdefault(upper, {}).setdefault(lower_slot, {})[leaf_slot] = node

    for x in range(int(leaf_lo[0]), int(leaf_hi[0]), 8):
        for y in range(int(leaf_lo[1]), int(leaf_hi[1]), 8):
            for z in range(int(leaf_lo[2]), int(leaf_hi[2]), 8):
                a = np.array([x, y, z]) - lo
                cut_lo, cut_hi = np.maximum(a, 0), np.minimum(a + 8, shape)
                if (cut_hi <= cut_lo).any():
                    continue
                on = np.zeros((8, 8, 8), bool)
                values = np.zeros((8, 8, 8, ncomp), flat_values.dtype)
                dst = tuple(slice(int(c - o), int(e - o)) for c, o, e in zip(cut_lo, a, cut_hi))
                src = tuple(slice(int(c), int(e)) for c, e in zip(cut_lo, cut_hi))
                on[dst] = active_mask[src]
                values[dst] = flat_values[src]
                if on.any():
                    insert((x, y, z), {"origin": (x, y, z), "active": on.reshape(-1),
                                       "values": values.reshape(512, ncomp)})
    for corner, value in tiles:
        if any(int(v) % 8 for v in corner):
            raise ValueError("write_vdb: a tile corner must be a multiple of 8")
        insert(tuple(int(v) for v in corner), {"tile": np.full(ncomp, value, flat_values.dtype)})
    return tree


def _slot(local, log2):
    return int((int(local[0]) << (2 * log2)) | (int(local[1]) << log2) | int(local[2]))


def _write_internal(writer, node, level, origin, bg, leaves):
    """Topology of an internal node: masks, values (tiles), then the children (level 0 is the 5-log2 node)."""
    log2 = 5 - level
    count = 1 << (3 * log2)
    child = np.zeros(count, bool)
    value_mask = np.zeros(count, bool)
    values = np.tile(bg, (count, 1))
    for slot, entry in node.items():
        if level == 1 and "tile" in entry:
            value_mask[slot] = True
            values[slot] = entry["tile"]
        else:
            child[slot] = True
    out = writer.out
    out.write(_pack_mask(child))
    out.write(_pack_mask(value_mask))
    writer.values(values, value_mask, child, bg)
    for slot in sorted(node):
        entry = node[slot]
        if level == 1:
            if "tile" not in entry:
                out.write(_pack_mask(entry["active"]))
                leaves.append(entry)
        else:
            _write_internal(writer, entry, 1, None, bg, leaves)
