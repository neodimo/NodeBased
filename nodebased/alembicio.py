"""Read-only Alembic Ogawa reader, using only Python and NumPy.

Objects expose name-keyed ``children`` and a top ``properties`` compound;
compounds expose name-keyed ``properties`` and support ``compound[name]``.
Samples are independent NumPy arrays: scalars have shape (extent,), arrays
have their stored dimensions followed by extent (omitted when extent is 1).
Strings use object arrays (UTF-8 with surrogateescape, or UTF-32 for wstring).
No coordinate, winding, transform, or unit conversion is performed.

The implementation interprets the upstream Alembic BSD-3 format specification
(Ogawa and AbcCoreOgawa); it does not use Alembic bindings. Files use a read-only mapping; byte slices and decoded arrays are independent
copies. close() releases the mapping and disables archive operations.
Container depth, total references and expanded tree size are deliberately bounded.
"""
from dataclasses import dataclass, field
import math
import mmap
from pathlib import Path
import operator
import struct

import numpy as np


class AlembicError(ValueError):
    """Unsupported, unreadable, or malformed Alembic archive."""


def _check(condition, message):
    if not condition:
        raise AlembicError(message)


class _Cursor:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, size):
        _check(0 <= size <= len(self.data) - self.pos, 'Truncated/corrupt Alembic data')
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt)))[0]

    def uint(self, width):
        return int.from_bytes(self.take(width), 'little')

    def __bool__(self):
        return self.pos < len(self.data)


def _text(data):
    return data.decode('utf-8', errors='surrogateescape')


def _metadata(data):
    result = {}
    for item in _text(data).split(';'):
        if item:
            key, sep, value = item.partition('=')
            _check(bool(sep and key), 'Corrupt metadata')
            result[key] = value
    return result


@dataclass(frozen=True)
class TimeSampling:
    """Stored cycle times; lookup(t, property_sample_count) -> floor, ceil, weight.

    Endpoints clamp. Within Alembic's 1e-5 second tolerance, samples snap.
    max_samples is archive bookkeeping, not a particular property's count.
    """
    kind: str = 'uniform'
    time_per_cycle: float = 1.0
    times: tuple = (0.0,)
    max_samples: int = 0

    def __post_init__(self):
        _check(self.kind in ('uniform', 'cyclic', 'acyclic'), 'Invalid sampling kind')
        _check(bool(self.times) and all(math.isfinite(t) for t in self.times), 'Invalid sample times')
        _check(all(a < b for a, b in zip(self.times, self.times[1:])), 'Unordered sample times')
        if self.kind != 'acyclic':
            _check(math.isfinite(self.time_per_cycle) and self.time_per_cycle > 0, 'Invalid cycle duration')
            _check(self.times[-1] - self.times[0] <= self.time_per_cycle, 'Invalid cyclic times')
        _check(self.kind != 'uniform' or len(self.times) == 1, 'Invalid uniform sampling')

    def sample_time(self, index):
        index = operator.index(index)
        _check(index >= 0, 'Negative sample index')
        if self.kind == 'acyclic':
            _check(index < len(self.times), 'Acyclic sample index out of range')
            return self.times[index]
        cycle, within = divmod(index, len(self.times))
        return self.times[within] + cycle * self.time_per_cycle

    def lookup(self, time, num_samples):
        num_samples = operator.index(num_samples)
        _check(num_samples > 0 and math.isfinite(time), 'Lookup requires samples and finite time')
        if self.kind == 'acyclic':
            _check(num_samples <= len(self.times), 'Too many acyclic samples')
        end = num_samples - 1
        if time <= self.sample_time(0):
            return 0, 0, 0.0
        if time >= self.sample_time(end):
            return end, end, 0.0
        lo, hi = 0, end
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.sample_time(mid) <= time:
                lo = mid
            else:
                hi = mid
        a, b = self.sample_time(lo), self.sample_time(hi)
        if abs(b - time) <= 1e-5:
            return hi, hi, 0.0
        if abs(time - a) <= 1e-5:
            return lo, lo, 0.0
        return lo, hi, (time - a) / (b - a)

    def floor_index(self, time, num_samples):
        return self.lookup(time, num_samples)[0]

    def ceil_index(self, time, num_samples):
        return self.lookup(time, num_samples)[1]


_PODS = ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
         'uint64', 'int64', 'float16', 'float32', 'float64', 'string', 'wstring')
_DTYPES = tuple(np.dtype(x).newbyteorder('<') for x in _PODS[:12])
_DATA = 1 << 63


@dataclass
class Property:
    name: str
    metadata: dict
    kind: str = 'compound'
    pod: str | None = None
    extent: int = 0
    num_samples: int = 0
    time_sampling_index: int = 0
    is_constant: bool = True
    properties: dict = field(default_factory=dict)
    _archive: object = field(default=None, repr=False)
    _refs: tuple = field(default=(), repr=False)
    _first: int = field(default=0, repr=False)
    _last: int = field(default=0, repr=False)

    def __getitem__(self, name):
        return self.properties[name]

    @property
    def time_sampling(self):
        _check(not self._archive.closed, 'Archive is closed')
        return self._archive.time_samplings[self.time_sampling_index]

    def lookup(self, time):
        return self.time_sampling.lookup(time, self.num_samples)

    def _stored_index(self, index):
        index = operator.index(index)
        _check(0 <= index < self.num_samples, 'Property sample index out of range')
        if self.is_constant or index < self._first:
            return 0
        return min(index, self._last) - self._first + 1

    def read(self, sample_index=0):
        _check(self.kind != 'compound', 'Cannot read a compound as a sample')
        index = self._stored_index(sample_index)
        slot = index * (2 if self.kind == 'array' else 1)
        raw = self._archive._data(self._refs[slot])
        _check(not raw or len(raw) >= 16, 'Truncated sample digest')
        payload = raw[16:]
        if self.pod in ('string', 'wstring'):
            if self.pod == 'wstring':
                _check(len(payload) % 4 == 0, 'Invalid wstring size')
                try:
                    text = payload.decode('utf-32-le')
                except UnicodeError as exc:
                    raise AlembicError('Invalid wstring data') from exc
            else:
                text = _text(payload)
            _check(not text or text.endswith('\0'), 'Unterminated string sample')
            values = np.array(text[:-1].split('\0') if text else [], dtype=object)
        else:
            dtype = _DTYPES[_PODS.index(self.pod)]
            _check(len(payload) % dtype.itemsize == 0, 'Misaligned sample payload')
            values = np.frombuffer(payload, dtype=dtype).copy()
        if self.kind == 'scalar':
            dims = ()
        else:
            dimdata = self._archive._data(self._refs[slot + 1])
            _check(len(dimdata) % 8 == 0 and len(dimdata) <= 256, 'Invalid array rank')
            dims = tuple(v[0] for v in struct.iter_unpack('<Q', dimdata))
            if not dims:
                _check(values.size % self.extent == 0, 'Invalid array extent')
                dims = (values.size // self.extent,)
        _check(math.prod(dims) * self.extent == values.size, 'Sample dimensions do not match payload')
        shape = dims + ((self.extent,) if self.extent != 1 or self.kind == 'scalar' else ())
        try:
            return values.reshape(shape)
        except (ValueError, OverflowError) as exc:
            raise AlembicError('Invalid sample dimensions') from exc


@dataclass
class Object:
    name: str
    full_path: str
    metadata: dict
    properties: Property
    children: dict = field(default_factory=dict)


class Archive:
    """Validated container and eagerly decoded headers; sample payloads decode on read."""
    def __init__(self, path):
        self.closed = False
        self._blob = b''
        self._groups = {}
        self._budget = 100000
        try:
            with open(fingerprint(path)[0], 'rb') as stream:
                self._blob = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as exc:
            raise AlembicError(f'Cannot read Alembic file {path}: {exc}') from exc
        try:
            self._initialize()
        except Exception:
            self.close()
            raise

    def _slice(self, offset, size):
        _check(not self.closed, 'Archive is closed')
        _check(0 <= offset <= len(self._blob) and 0 <= size <= len(self._blob) - offset,
               'Truncated/corrupt Ogawa offset or size')
        return self._blob[offset:offset + size]

    def _u64(self, offset):
        return struct.unpack('<Q', self._slice(offset, 8))[0]

    def _data(self, ref):
        _check(ref & _DATA, 'Expected Ogawa data reference')
        offset = ref & (_DATA - 1)
        if offset == 0:
            _check(not self.closed, 'Archive is closed')
            return b''
        _check(offset >= 16, 'Invalid data offset')
        return self._slice(offset + 8, self._u64(offset))

    def _validate(self, ref, active, depth=0):
        _check(depth <= 128, 'Ogawa group tree is too deep')
        self._budget -= 1
        _check(self._budget >= 0, 'Ogawa group tree is oversized')
        if ref & _DATA:
            offset = ref & (_DATA - 1)
            if offset:
                _check(offset >= 16, 'Invalid data offset')
                size = self._u64(offset)
                _check(size <= len(self._blob) - offset - 8, 'Truncated/corrupt Ogawa data size')
            return
        _check(ref not in active, 'Cyclic Ogawa group tree')
        if ref in self._groups:
            return
        if ref == 0:
            self._groups[ref] = ()
            return
        _check(ref >= 16, 'Invalid group offset')
        count = self._u64(ref)
        _check(0 < count <= 100000, 'Invalid/oversized group child count')
        refs = tuple(x[0] for x in struct.iter_unpack('<Q', self._slice(ref + 8, count * 8)))
        active.add(ref)
        for child in refs:
            self._validate(child, active, depth + 1)
        active.remove(ref)
        self._groups[ref] = refs

    def _group(self, ref):
        _check(ref in self._groups, 'Expected Ogawa group reference')
        self._budget -= 1
        _check(self._budget >= 0, 'Expanded Alembic tree is oversized')
        return self._groups[ref]

    def _initialize(self):
        if self._blob[:8] == b'\x89HDF\r\n\x1a\n':
            raise AlembicError('HDF5-backed Alembic archives are not supported (only Ogawa)')
        _check(self._blob[:5] == b'Ogawa', 'Not an Ogawa Alembic archive')
        header = self._slice(0, 16)
        _check(header[5] != 0, 'Ogawa archive is not frozen: incomplete or being written')
        _check(header[5] == 255, 'Corrupt Ogawa frozen flag')
        # Version is two big-endian bytes; offsets and payloads are little-endian.
        self.ogawa_version = int.from_bytes(header[6:8], 'big')
        root = struct.unpack('<Q', header[8:])[0]
        _check(self.ogawa_version == 1, 'Unsupported Ogawa version')
        self._validate(root, set())
        refs = self._group(root)
        _check(len(refs) >= 6, 'Corrupt Alembic archive root')
        versions = [self._data(r) for r in refs[:2]]
        _check(all(len(v) == 4 for v in versions), 'Corrupt Alembic version')
        self.format_version, self.archive_version = [struct.unpack('<i', v)[0] for v in versions]
        _check(self.format_version == 0 and self.archive_version >= 9999, 'Unsupported Alembic version')
        self.metadata = _metadata(self._data(refs[3]))
        self.time_samplings = []
        cursor = _Cursor(self._data(refs[4]))
        while cursor:
            maximum, period, count = cursor.unpack('I'), cursor.unpack('d'), cursor.unpack('I')
            _check(0 < count <= len(cursor.data) // 8, 'Invalid time sampling count')
            times = struct.unpack('<' + str(count) + 'd', cursor.take(count * 8))
            kind = 'acyclic' if period == np.finfo(np.float64).max else ('uniform' if count == 1 else 'cyclic')
            self.time_samplings.append(TimeSampling(kind, period, times, maximum))
        # Ogawa serializes the default identity entry too; do not prepend it.
        _check(bool(self.time_samplings), 'Missing default time sampling')
        default = self.time_samplings[0]
        _check(default.kind == 'uniform' and default.times == (0.0,) and default.time_per_cycle == 1,
               'Invalid default time sampling')
        self._metadata_table = [{}]
        cursor = _Cursor(self._data(refs[5]))
        _check(len(cursor.data) <= 65536, 'Oversized metadata table')
        while cursor:
            self._metadata_table.append(_metadata(cursor.take(cursor.uint(1))))
        _check(len(self._metadata_table) <= 255, 'Oversized metadata index')
        self.root = self._object(refs[2], 'ABC', '/', self.metadata)

    def _meta(self, cursor, index, width):
        if index == 255:
            return _metadata(cursor.take(cursor.uint(width)))
        _check(index < len(self._metadata_table), 'Invalid metadata index')
        return self._metadata_table[index].copy()

    def _object(self, ref, name, path, metadata):
        refs = self._group(ref)
        if not refs:
            return Object(name, path, metadata, Property('', {}, _archive=self))
        _check(len(refs) >= 2, 'Corrupt object layout')
        result = Object(name, path, metadata, self._compound(refs[0], Property('', {}, _archive=self)))
        data = self._data(refs[-1])
        _check(len(data) >= 32, 'Truncated object hashes')
        cursor = _Cursor(data[:-32])
        i = 1
        while cursor:
            childname = _text(cursor.take(cursor.uint(4)))
            _check(bool(childname) and '/' not in childname and childname not in result.children, 'Invalid object name')
            meta = self._meta(cursor, cursor.uint(1), 4)
            _check(i < len(refs) - 1, 'Missing object group')
            result.children[childname] = self._object(refs[i], childname, path.rstrip('/') + '/' + childname, meta)
            i += 1
        _check(i == len(refs) - 1, 'Object header/group count mismatch')
        return result

    def _compound(self, ref, result):
        refs = self._group(ref)
        if not refs:
            return result
        cursor = _Cursor(self._data(refs[-1]))
        i = 0
        while cursor:
            info = cursor.unpack('I')
            ptype, hint = info & 3, (info >> 2) & 3
            _check(hint < 3, 'Invalid property size hint')
            width = 1 << hint
            pod, extent, count, first, last, ts = None, 0, 0, 0, 0, 0
            if ptype:
                podindex, extent = (info >> 4) & 15, (info >> 12) & 255
                _check(podindex < len(_PODS) and extent > 0, 'Invalid property POD/extent')
                pod = _PODS[podindex]
                count = cursor.uint(width)
                if info & 512:
                    first, last = cursor.uint(width), cursor.uint(width)
                elif not info & 2048 and count > 1:
                    first, last = 1, count - 1
                _check((first == last == 0) or (0 < first <= last < count), 'Invalid sample repetition range')
                ts = cursor.uint(width) if info & 256 else 0
                _check(ts < len(self.time_samplings), 'Invalid time sampling index')
                sampling = self.time_samplings[ts]
                _check(sampling.kind != 'acyclic' or count <= len(sampling.times), 'Invalid acyclic sample count')
            name = _text(cursor.take(cursor.uint(width)))
            _check(bool(name) and name not in result.properties, 'Invalid property name')
            meta = self._meta(cursor, (info >> 20) & 255, width)
            _check(i < len(refs) - 1, 'Missing property group')
            prop = Property(name, meta, 'compound' if not ptype else ('scalar' if ptype == 1 else 'array'),
                            pod, extent, count, ts, first == 0, _archive=self, _first=first, _last=last)
            if not ptype:
                self._compound(refs[i], prop)
            else:
                prop._refs = self._group(refs[i])
                stored = 0 if count == 0 else (1 if first == 0 else last - first + 2)
                _check(len(prop._refs) == stored * (2 if prop.kind == 'array' else 1), 'Stored sample count mismatch')
                _check(all(r & _DATA for r in prop._refs), 'Invalid sample reference')
            result.properties[name] = prop
            i += 1
        _check(i == len(refs) - 1, 'Property header/group count mismatch')
        return result

    def close(self):
        self.closed = True
        if isinstance(self._blob, mmap.mmap):
            self._blob.close()
        self._blob = b''
        self._groups.clear()

    def __enter__(self):
        _check(not self.closed, 'Archive is closed')
        return self

    def __exit__(self, *args):
        self.close()


def open_archive(path):
    """Open a frozen Ogawa archive; raises AlembicError on invalid input."""
    return Archive(path)


@dataclass
class GeometryParameter:
    values: np.ndarray
    indices: np.ndarray | None
    scope: str

    def expanded_face_corners(self, face_counts, face_indices):
        """Expand indexed values and constant/uniform/vertex/varying/facevarying scopes."""
        values = self.values
        if self.indices is not None:
            _check(np.all(self.indices >= 0) and np.all(self.indices < len(values)), 'Invalid geometry parameter indices')
            values = values[self.indices]
        scope = self.scope
        if scope in ('con', 'constant'):
            _check(len(values) == 1, 'Invalid constant geometry parameter')
            return np.repeat(values, len(face_indices), axis=0)
        if scope in ('uni', 'uniform'):
            _check(len(values) == len(face_counts) and np.all(face_counts >= 0), 'Invalid uniform geometry parameter')
            return np.repeat(values, face_counts, axis=0)
        if scope in ('vtx', 'var', 'vertex', 'varying'):
            _check(np.all(face_indices >= 0) and np.all(face_indices < len(values)), 'Invalid vertex geometry parameter')
            return values[face_indices]
        _check(scope in ('fvr', 'facevarying') and len(values) == len(face_indices), 'Invalid face-varying scope/size')
        return values.copy()


@dataclass
class PolyMesh:
    positions: np.ndarray
    face_indices: np.ndarray
    face_counts: np.ndarray
    uvs: GeometryParameter | None
    normals: GeometryParameter | None
    num_samples: int
    time_sampling: TimeSampling
    self_bounds: np.ndarray | None


def read_polymesh(obj, sample_index=0):
    """Read local-space mesh data without applying parent transforms or changing winding.

    Each component clamps to its own last logical sample (including constants).
    The mesh sample range and time sampling are those of positions.
    """
    _check(obj.metadata.get('schema') == 'AbcGeom_PolyMesh_v1', 'Object is not AbcGeom_PolyMesh_v1')
    try:
        geom = obj.properties['.geom']
        positions = geom['P']
        _check(0 <= sample_index < positions.num_samples, 'Mesh sample index out of range')

        def read(prop):
            return prop.read(min(sample_index, prop.num_samples - 1))

        def parameter(name):
            prop = geom.properties.get(name)
            if prop is None:
                return None
            if prop.kind == 'compound':
                vals = prop.properties.get('.vals', prop.properties.get('vals'))
                inds = prop.properties.get('.indices', prop.properties.get('indices'))
                _check(vals is not None, 'Missing geometry parameter values')
                return GeometryParameter(read(vals), read(inds) if inds else None,
                                         prop.metadata.get('geoScope', vals.metadata.get('geoScope', '')))
            return GeometryParameter(read(prop), None, prop.metadata.get('geoScope', ''))

        p, indices, counts = read(positions), read(geom['.faceIndices']), read(geom['.faceCounts'])
        _check(p.ndim == 2 and p.shape[1] == 3 and p.dtype == np.dtype('<f4'), 'Invalid mesh positions')
        _check(counts.ndim == indices.ndim == 1 and counts.dtype.kind in 'iu' and indices.dtype.kind in 'iu', 'Invalid mesh topology type')
        _check(np.all(counts >= 0) and sum(map(int, counts)) == len(indices), 'Invalid face counts')
        _check(np.all(indices >= 0) and np.all(indices < len(p)), 'Invalid face indices')
        bounds = geom.properties.get('.selfBnds')
        return PolyMesh(p, indices, counts, parameter('uv'), parameter('N'), positions.num_samples,
                        positions.time_sampling, read(bounds).reshape(2, 3) if bounds else None)
    except (KeyError, ValueError) as exc:
        if isinstance(exc, AlembicError):
            raise
        raise AlembicError(f'Malformed PolyMesh: {exc}') from exc


def dump(archive):
    """Return an object/property tree suitable for debugging."""
    _check(not archive.closed, 'Archive is closed')
    lines = []

    def props(prop, indent):
        for p in prop.properties.values():
            lines.append(' ' * indent + f'{p.name}: {p.kind}' +
                         (f' {p.pod}[{p.extent}] samples={p.num_samples} ts={p.time_sampling_index} constant={p.is_constant}' if p.kind != 'compound' else ''))
            props(p, indent + 2)

    def obj(o, indent):
        lines.append(' ' * indent + o.full_path + ' ' + o.metadata.get('schema', ''))
        props(o.properties, indent + 2)
        for child in o.children.values():
            obj(child, indent + 2)

    obj(archive.root, 0)
    return '\n'.join(lines)


# Object is the public tree reader; retain its original name for compatibility.
ObjectReader = Object


@dataclass
class XformOp:
    """Decoded operation; animated_channels are indices local to this operation.

    Hints are retained as their low-nibble integer codes. They describe authoring
    intent and do not change the operation's matrix. Values include static channels.
    """
    type: str
    hint: int
    values: tuple
    animated_channels: tuple


@dataclass
class Xform:
    matrix: np.ndarray
    inherits: bool
    ops: list


def _axis_rotation(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    length = np.linalg.norm(axis)
    if length == 0:
        return np.eye(3)
    x, y, z = axis / length
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    # Transposed Rodrigues formula: vectors multiply from the left.
    return c * np.eye(3) + (1-c) * np.outer(axis/length, axis/length) + s * np.array(
        [[0, z, -y], [-z, 0, x], [y, -x, 0]])


def _decode_xform_ops(encoded, values, animated_channels=()):
    names = ('scale', 'translate', 'rotate', 'matrix', 'rotateX', 'rotateY', 'rotateZ')
    sizes = (3, 3, 4, 16, 1, 1, 1)
    values = np.asarray(values, dtype=np.float64).ravel()
    animated = set(map(int, animated_channels))
    matrix, ops, offset = np.eye(4), [], 0
    for byte in np.asarray(encoded).ravel():
        code, hint = int(byte) >> 4, int(byte) & 15
        _check(0 <= code < len(names), f'Unknown xform op type {code}')
        size = sizes[code]
        channels = values[offset:offset + size]
        _check(len(channels) == size, 'Missing xform channels')
        ops.append(XformOp(names[code], hint, tuple(channels),
                           tuple(i for i in range(size) if offset+i in animated)))
        offset += size
        op = np.eye(4)
        if code == 0:
            op[:3, :3] = np.diag(channels)
        elif code == 1:
            op[3, :3] = channels
        elif code == 3:
            op = channels.reshape(4, 4).copy()
        else:
            axis = channels[:3] if code == 2 else np.eye(3)[code-4]
            op[:3, :3] = _axis_rotation(axis, channels[-1])
        # Avoid arithmetic on a single stored matrix (including signed zeros).
        matrix = op if len(ops) == 1 else op @ matrix
    _check(offset == len(values), 'Unexpected xform channels')
    _check(all(0 <= i < offset for i in animated), 'Invalid animated channel index')
    return matrix, ops


def _schema(obj, schema, compound):
    _check(obj.metadata.get('schema') == schema, f'Object is not {schema}')
    _check(compound in obj.properties.properties, f'Missing {compound} schema')
    return obj.properties[compound].properties


def _sample(prop, index):
    _check(prop.num_samples > 0, 'Property has no samples')
    return prop.read(min(index, prop.num_samples-1))


def read_xform(obj, sample_index=0):
    """Read an Alembic row-vector matrix, inheritance and decoded operation stack.

    .ops topology is static. The final .animChans sample labels animated channels;
    .vals always contains ALL channels, including static ones. Missing identity
    marker means constant identity in the upstream schema; an explicit false
    marker is also accepted as the identity shortcut.
    """
    props = _schema(obj, 'AbcGeom_Xform_v3', '.xform')
    sample_index = operator.index(sample_index)
    count = max((p.num_samples for n, p in props.items() if n in ('.vals', '.inherits')), default=1)
    _check(0 <= sample_index < count, 'Xform sample index out of range')
    inherits = bool(_sample(props['.inherits'], sample_index)[0]) if '.inherits' in props else True
    marker = props.get('isNotConstantIdentity')
    identity = marker is None or not bool(marker.read(0)[0])
    vals, encoded = props.get('.vals'), props.get('.ops')
    if vals is None or encoded is None:
        _check(identity, 'Missing xform operations or values')
        return Xform(np.eye(4), inherits, [])
    anim = props.get('.animChans')
    animated = anim.read(anim.num_samples-1).ravel() if anim is not None and anim.num_samples else ()
    matrix, ops = _decode_xform_ops(encoded.read(0), _sample(vals, sample_index), animated)
    return Xform(np.eye(4) if identity else matrix, inherits, ops)


def _decompose(matrix):
    if not np.isfinite(matrix).all() or not np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-8):
        return None
    scale = np.linalg.norm(matrix[:3, :3], axis=1)
    if np.any(scale < 1e-12):
        return None
    rotation = matrix[:3, :3] / scale[:, None]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0):
        return None
    if np.linalg.det(rotation) < 0:
        scale[0] *= -1
        rotation[0] *= -1
    # Quaternion from a column-vector rotation, choosing the largest component
    # to remain stable at 180 degrees. Components are (w, x, y, z).
    r = rotation.T
    candidates = np.array([1 + np.trace(r), 1+2*r[0, 0]-np.trace(r),
                           1+2*r[1, 1]-np.trace(r), 1+2*r[2, 2]-np.trace(r)])
    i = int(np.argmax(candidates))
    q = np.zeros(4)
    q[i] = math.sqrt(max(0, candidates[i])) / 2
    d = 4*q[i]
    if i == 0:
        q[1:] = [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]
        q[1:] /= d
    else:
        a, b, c = i-1, i % 3, (i+1) % 3
        q[0] = (r[c, b]-r[b, c])/d
        q[b+1] = (r[b, a]+r[a, b])/d
        q[c+1] = (r[c, a]+r[a, c])/d
    return matrix[3, :3], q / np.linalg.norm(q), scale


def _interpolate_matrix(a, b, weight):
    da, db = _decompose(a), _decompose(b)
    if da is None or db is None:
        return (1-weight)*a + weight*b
    ta, qa, sa = da
    tb, qb, sb = db
    dot = float(qa @ qb)
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 1 - 1e-12:
        q = (1-weight)*qa + weight*qb
    else:
        angle = math.acos(np.clip(dot, -1, 1))
        q = (math.sin((1-weight)*angle)*qa + math.sin(weight*angle)*qb)/math.sin(angle)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)],
                         [2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)],
                         [2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)]])
    result = np.eye(4)
    result[:3, :3] = ((1-weight)*sa + weight*sb)[:, None]*rotation
    result[3, :3] = (1-weight)*ta + weight*tb
    return result


def xform_at_time(obj, time):
    """Return a 4x4 row-vector matrix, clamped to the property's sampled range.

    Exact samples bypass decomposition. Translation/scale are linear, rotation
    uses shortest-path quaternion slerp. Significant shear (>1e-6 normalised
    axis dot product), singular or non-affine matrices use component-wise matrix
    lerp instead. Constant transforms return their stored matrix.
    """
    props = _schema(obj, 'AbcGeom_Xform_v3', '.xform')
    _check(math.isfinite(time), 'Time must be finite')
    vals = props.get('.vals')
    if vals is None or vals.is_constant:
        return read_xform(obj).matrix
    lo, hi, weight = vals.lookup(time)
    a = read_xform(obj, lo).matrix
    return a if weight == 0 else _interpolate_matrix(a, read_xform(obj, hi).matrix, weight)


def _object_chain(archive_or_root, path):
    _check(not getattr(archive_or_root, 'closed', False), 'Archive is closed')
    root = getattr(archive_or_root, 'root', archive_or_root)
    path = path.full_path if hasattr(path, 'full_path') else path
    _check(isinstance(path, str), 'Object path must be a string or ObjectReader')
    prefix = root.full_path.rstrip('/')
    if path.startswith('/'):
        _check(path == prefix or path.startswith(prefix+'/'), 'Path is outside supplied root')
        path = path[len(prefix):]
    chain = [root]
    for name in filter(None, path.split('/')):
        _check(name in chain[-1].children, f'Object not found: {path}')
        chain.append(chain[-1].children[name])
    return chain


def world_matrix(archive_or_root, path, time):
    """Compose ancestors child @ parent, resetting at inherits=False.

    Accept an Archive or ObjectReader root and an object path or ObjectReader.
    Non-xform objects contribute identity. Inheritance is held at its floor
    sample using its own time sampling (it is a discrete boolean).
    """
    _check(math.isfinite(time), 'Time must be finite')
    matrix = np.eye(4)
    for obj in _object_chain(archive_or_root, path):
        if obj.metadata.get('schema') != 'AbcGeom_Xform_v3':
            continue
        props = _schema(obj, 'AbcGeom_Xform_v3', '.xform')
        prop = props.get('.inherits')
        inherits = bool(prop.read(prop.lookup(time)[0])[0]) if prop is not None else True
        local = xform_at_time(obj, time)
        matrix = local @ matrix if inherits else local
    return matrix


@dataclass
class CameraSample:
    """The 16 Alembic core scalars, in storage order.

    Focal length is mm; apertures and film offsets are cm. Other distances use
    authored scene units. Alembic core has no film-fit enum; scene3d conversion
    fits the vertical aperture and lets the output image determine aspect.
    """
    focal_length: float
    horizontal_aperture: float
    horizontal_film_offset: float
    vertical_aperture: float
    vertical_film_offset: float
    lens_squeeze_ratio: float
    overscan_left: float
    overscan_right: float
    overscan_top: float
    overscan_bottom: float
    f_stop: float
    focus_distance: float
    shutter_open: float
    shutter_close: float
    near_clipping_plane: float
    far_clipping_plane: float


def _camera_core(obj):
    props = _schema(obj, 'AbcGeom_Camera_v1', '.geom')
    _check('.core' in props, 'Missing camera core')
    return props['.core']


def read_camera(obj, sample_index=0):
    """Read all 16 core scalars. Optional film-back transform ops are not applied."""
    values = _camera_core(obj).read(sample_index).ravel()
    _check(len(values) == 16, 'Invalid camera core size')
    return CameraSample(*map(float, values))


def camera_at_time(obj, time):
    """Linearly interpolate every core scalar; clamp outside the sampled range."""
    prop = _camera_core(obj)
    lo, hi, weight = prop.lookup(time)
    a = read_camera(obj, lo)
    if weight == 0 or prop.is_constant:
        return a
    b = read_camera(obj, hi)
    return CameraSample(*((1-weight)*getattr(a, name) + weight*getattr(b, name)
                          for name in CameraSample.__dataclass_fields__))


def camera_to_scene3d(archive_or_root, obj_or_path, time):
    """Convert a perspective camera using vertical film-back fit.

    Reject non-uniform scale, shear and reflection; normalise positive uniform
    scale. Ignore film offsets, lens squeeze, overscan, shutter, depth of field
    and optional film-back operations. No unit metadata exists in Alembic:
    distances are used as authored, assumed Y-up and right-handed.
    """
    from dataclasses import replace
    from . import filmback as fb
    from . import scene3d as s

    chain = _object_chain(archive_or_root, obj_or_path)
    camera = camera_at_time(chain[-1], time)
    matrix = world_matrix(archive_or_root, chain[-2], time) if len(chain) > 1 else np.eye(4)
    axes = matrix[:3, :3]
    lengths = np.linalg.norm(axes, axis=1)
    _check(np.isfinite(matrix).all() and np.all(lengths > 1e-8)
           and np.allclose(lengths, lengths[0], rtol=1e-5, atol=1e-8)
           and np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-8)
           and np.allclose((axes / lengths[:, None]) @ (axes / lengths[:, None]).T,
                           np.eye(3), atol=1e-5)
           and np.linalg.det(axes) > 0,
           'Camera non-uniform scale, shear or reflection is unsupported')
    axes = axes / lengths[:, None]
    position, forward, up = matrix[3, :3], -axes[2], axes[1]
    focus = camera.focus_distance if camera.focus_distance > 0 else 1.0
    _check(camera.focal_length > 0 and camera.vertical_aperture > 0,
           'Camera aperture and focal length must be positive')
    # The core stores apertures in centimetres and the focal length in millimetres.
    result = s.Camera(s.Transform3D(position=s.Vec3(*position)),
                      s.Vec3(*(position + forward*focus)),
                      fb.fov_from_aperture(camera.focal_length, camera.vertical_aperture*10),
                      camera.near_clipping_plane, camera.far_clipping_plane,
                      haperture=camera.horizontal_aperture*10, vaperture=camera.vertical_aperture*10)
    _, basis = s._view_basis(result)
    # The renderer rolls up toward -right: up = cos(r)*up0 - sin(r)*right0.
    # Projections on those two zero-roll axes therefore give cos(r), -sin(r).
    roll = math.degrees(math.atan2(-float(up @ basis[0]), float(up @ basis[1])))
    return replace(result, roll=roll)


def fingerprint(path) -> list:
    """Return resolved path, file size and nanosecond modification time."""
    try:
        if path is None or not str(path).strip():
            raise ValueError('empty path')
        resolved = Path(path).expanduser().resolve()
        with resolved.open('rb') as stream:
            import os
            stat = os.fstat(stream.fileno())
        return [str(resolved), stat.st_size, stat.st_mtime_ns]
    except (OSError, ValueError, TypeError):
        raise AlembicError(f'cannot read {path}') from None


def _walk(obj):
    yield obj
    for child in obj.children.values():
        yield from _walk(child)


def _is_visible(archive_or_root, obj, time):
    """Missing/deferred visibility inherits; any hidden ancestor hides its subtree."""
    for ancestor in _object_chain(archive_or_root, obj):
        prop = ancestor.properties.properties.get('visible')
        if prop is not None:
            value = prop.read(prop.lookup(time)[0])
            _check(value.size == 1 and int(value.flat[0]) in (-1, 0, 1),
                   'Invalid Alembic visibility')
            if int(value.flat[0]) == 0:
                return False
    return True


def _scene_mesh(archive, obj, time, remaining):
    from . import scene3d as s
    prop = obj.properties['.geom']['P']
    lo, hi, weight = prop.lookup(time)
    mesh = read_polymesh(obj, lo)
    points = mesh.positions
    if weight:
        other = read_polymesh(obj, hi)
        if (points.shape == other.positions.shape
                and np.array_equal(mesh.face_counts, other.face_counts)
                and np.array_equal(mesh.face_indices, other.face_indices)):
            points = (1-weight)*points + weight*other.positions
    _check(np.isfinite(points).all(), f'{obj.full_path}: non-finite points')

    def expanded(parameter, width):
        if parameter is None:
            return None
        values = parameter.expanded_face_corners(mesh.face_counts, mesh.face_indices)
        _check(values.shape == (len(mesh.face_indices), width) and np.isfinite(values).all(),
               f'{obj.full_path}: invalid geometry attribute')
        return values

    uvs = expanded(mesh.uvs, 2)
    normals = expanded(mesh.normals, 3) if mesh.normals is not None and mesh.normals.scope not in ('uni', 'uniform') else None
    corners, triangles = {}, []
    offset = 0
    for count in mesh.face_counts:
        face = mesh.face_indices[offset:offset+count]
        if count >= 3 and len(set(face)) == count:
            for j in range(1, count-1):
                local = (0, j+1, j)  # Reverse Alembic's stored winding.
                a, b, c = points[[face[k] for k in local]]
                if not np.any(np.cross(b-a, c-a)):
                    continue
                triangle = []
                for k in local:
                    key = (int(face[k]), tuple(uvs[offset+k]) if uvs is not None else None,
                           tuple(normals[offset+k]) if normals is not None else None)
                    triangle.append(corners.setdefault(key, len(corners)))
                triangles.append(triangle)
                _check(len(triangles) <= remaining,
                       f'{obj.full_path}: more than {s.MAX_TRIANGLES} triangles; '
                       'the CPU reference renderer refuses meshes this large')
        offset += count
    if not triangles:
        return None
    keys = list(corners)
    matrix = world_matrix(archive, obj, time)
    vertices = points[[k[0] for k in keys]] @ matrix[:3, :3] + matrix[3, :3]
    uv_array = np.asarray([k[1] for k in keys], np.float32) if uvs is not None else None
    normal_array = None
    if normals is not None:
        try:
            normal_array = np.asarray([k[2] for k in keys]) @ np.linalg.inv(matrix[:3, :3]).T
        except np.linalg.LinAlgError as exc:
            raise AlembicError('cannot transform normals with a singular world transform') from exc
        normal_array /= np.maximum(np.linalg.norm(normal_array, axis=1, keepdims=True), 1e-8)
        normal_array = normal_array.astype(np.float32)
    return s.Geometry(vertices.astype(np.float32), np.asarray(triangles, np.int32),
                      (0.8, 0.8, 0.8, 1.0), uvs=uv_array, normals=normal_array)


def load_scene(path, time, root='/'):
    """Load visible PolyMeshes, welding corners and baking world transforms.

    Time is seconds. Positions interpolate only across identical topology;
    attributes use the floor sample. Uniform normals request flat shading.
    Units/axes are used as authored (Alembic has no unit metadata), assumed
    Y-up right-handed as exported. Other schemas are skipped; their counts
    are available through unsupported_schemas(). The archive closes on return.
    """
    from . import scene3d as s
    _check(math.isfinite(time), 'Time must be finite')
    geometries, total = [], 0
    with open_archive(path) as archive:
        for obj in _walk(_object_chain(archive, root)[-1]):
            if obj.metadata.get('schema') != 'AbcGeom_PolyMesh_v1' or not _is_visible(archive, obj, time):
                continue
            geometry = _scene_mesh(archive, obj, time, s.MAX_TRIANGLES-total)
            if geometry is not None:
                geometries.append(geometry)
                total += len(geometry.triangles)
    return s.Scene(tuple(geometries))


def unsupported_schemas(path, root='/') -> dict:
    """Count skipped schema objects under root, including hidden objects.

    Transforms and cameras are supported and excluded; schema-less grouping
    objects are excluded. Counts include curves, points, SubD, NuPatch, face
    sets and any other unknown schema, one per object (not per sample).
    """
    counts = {}
    with open_archive(path) as archive:
        for obj in _walk(_object_chain(archive, root)[-1]):
            schema = obj.metadata.get('schema')
            if schema and schema not in ('AbcGeom_PolyMesh_v1', 'AbcGeom_Xform_v3', 'AbcGeom_Camera_v1'):
                counts[schema] = counts.get(schema, 0)+1
    return counts


def load_camera(path, time, camera_path=''):
    """Load the first depth-first camera, or a camera shape/parent transform path.

    Uses camera_to_scene3d's vertical film fit and authored units/axes.
    """
    with open_archive(path) as archive:
        if camera_path:
            try:
                obj = _object_chain(archive, camera_path)[-1]
            except AlembicError as exc:
                raise AlembicError(f'Alembic camera not found: {camera_path}') from exc
            candidates = ([obj] if obj.metadata.get('schema') == 'AbcGeom_Camera_v1'
                          else obj.children.values() if obj.metadata.get('schema') == 'AbcGeom_Xform_v3' else [])
        else:
            candidates = _walk(archive.root)
        camera = next((o for o in candidates if o.metadata.get('schema') == 'AbcGeom_Camera_v1'), None)
        _check(camera is not None, f'Alembic camera not found: {camera_path or "<first camera>"}')
        return camera_to_scene3d(archive, camera, time)
