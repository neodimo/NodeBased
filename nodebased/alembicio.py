"""Read-only Alembic Ogawa reader, using only Python and NumPy.

Objects expose name-keyed ``children`` and a top ``properties`` compound;
compounds expose name-keyed ``properties`` and support ``compound[name]``.
Samples are independent NumPy arrays: scalars have shape (extent,), arrays
have their stored dimensions followed by extent (omitted when extent is 1).
Strings use object arrays (UTF-8 with surrogateescape, or UTF-32 for wstring).
No coordinate, winding, transform, or unit conversion is performed.

The implementation interprets the upstream Alembic BSD-3 format specification
(Ogawa and AbcCoreOgawa); it does not use Alembic bindings. Files are read into
an immutable snapshot and closed during open_archive, so returned arrays never
keep OS handles alive. close() releases the snapshot and disables sample reads.
Container depth, total references and expanded tree size are deliberately bounded.
"""
from dataclasses import dataclass, field
import math
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
            with open(path, 'rb') as stream:
                self._blob = stream.read()
        except OSError as exc:
            raise AlembicError(f'Cannot read Alembic file {path!s}: {exc}') from exc
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
        if self._blob.startswith(b'\x89HDF\r\n\x1a\n'):
            raise AlembicError('HDF5-backed Alembic archives are not supported (only Ogawa)')
        _check(self._blob.startswith(b'Ogawa'), 'Not an Ogawa Alembic archive')
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
