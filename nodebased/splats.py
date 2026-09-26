"""NumPy-only 3DGS data and PLY IO; column-vector, +Y-up/-Z-forward transforms.

SH uses the reference 3DGS real basis and position-minus-camera view directions.
Display-referred SH is never linearized coefficient-wise: call to_linear_color on
its evaluated RGB at render time. No rendering or scene/node integration lives here.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from collections import OrderedDict
from threading import RLock

import numpy as np

MAX_SPLATS = 20_000_000
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435)


def _quaternions(q):
    q = np.array(q, dtype=np.float64, copy=True)
    length = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.divide(q, length, out=np.zeros_like(q), where=length != 0)
    q[length[:, 0] == 0, 0] = 1
    return q


def _rotation(q):
    w, x, y, z = _quaternions(q).T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def _matrix_quaternions(r):
    # Four stable branches, vectorized across splats (including 180-degree turns).
    q = np.empty((len(r), 4), np.float64)
    candidates = np.stack((1+np.trace(r, axis1=1, axis2=2),
                          1+r[:,0,0]-r[:,1,1]-r[:,2,2],
                          1-r[:,0,0]+r[:,1,1]-r[:,2,2],
                          1-r[:,0,0]-r[:,1,1]+r[:,2,2]), -1)
    branch = candidates.argmax(axis=1)
    for i in range(4):
        mask = branch == i
        a = r[mask]
        t = np.sqrt(np.maximum(candidates[mask, i], 0)) * 2
        if i == 0:
            values = (t/4, (a[:,2,1]-a[:,1,2])/t, (a[:,0,2]-a[:,2,0])/t, (a[:,1,0]-a[:,0,1])/t)
        else:
            j, k = ((1,2), (2,0), (0,1))[i-1]
            h = i-1
            values = [None]*4
            values[0] = (a[:,k,j]-a[:,j,k])/t
            values[i] = t/4
            values[j+1] = (a[:,j,h]+a[:,h,j])/t
            values[k+1] = (a[:,k,h]+a[:,h,k])/t
        q[mask] = np.stack(values, -1)
    return q


def covariance_upper(covariance):
    """Pack (N,3,3) as float32 (N,6): xx, xy, xz, yy, yz, zz."""
    return np.asarray(covariance)[:, (0,0,0,1,1,2), (0,1,2,1,2,2)].astype(np.float32)


@dataclass(frozen=True, eq=False)
class SplatCloud:
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    sh_degree: int
    raw_scale_log: np.ndarray | None = None
    raw_opacity_logit: np.ndarray | None = None
    colorspace: str = 'srgb'
    intrinsics: object | None = None   # nodebased.intrinsics.Intrinsics: de-lit albedo, normals, roughness (captured SH untouched)

    def __post_init__(self):
        if not isinstance(self.sh_degree, (int, np.integer)) or not 0 <= self.sh_degree <= 3:
            raise ValueError('SH degree must be 0..3')
        if self.colorspace not in ('srgb', 'linear'):
            raise ValueError('colorspace must be srgb or linear')
        n = len(self.positions)
        if self.intrinsics is not None and len(self.intrinsics) != n:
            raise ValueError('intrinsics must have one entry per splat')
        shapes = dict(positions=(n,3), scales=(n,3), rotations=(n,4), opacity=(n,),
                      sh=(n,(self.sh_degree+1)**2,3), raw_scale_log=(n,3), raw_opacity_logit=(n,))
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is None and name.startswith('raw_'):
                continue
            a = np.array(value, dtype=np.float32, copy=True)
            if a.shape != shape:
                raise ValueError(f'{name} must have shape {shape}')
            if np.isnan(a).any() or (not name.startswith('raw_') and not np.isfinite(a).all()):
                raise ValueError(f'{name} must contain finite values')
            if name == 'rotations':
                # Already rounded unit quaternions stay bit-identical on re-export.
                norms = np.linalg.norm(a.astype(np.float64), axis=1)
                mask = np.abs(norms-1) > 6e-8
                a[mask] = _quaternions(a[mask])
            if name == 'scales' and (a < 0).any():
                raise ValueError('scales must be nonnegative')
            if name == 'opacity' and ((a < 0) | (a > 1)).any():
                raise ValueError('opacity must be in 0..1')
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    def __len__(self):
        return len(self.positions)

    def covariance(self):
        """Return float64 (N,3,3) R diag(scales**2) R.T."""
        rs = _rotation(self.rotations) * self.scales.astype(np.float64)[:, None, :]
        return rs @ rs.transpose(0,2,1)

    def covariance_upper(self):
        return covariance_upper(self.covariance())

    def normals(self):
        """Shortest local axis in world space; ties choose the first scale axis."""
        r = _rotation(self.rotations)
        return r[np.arange(len(self)), :, self.scales.argmin(axis=1)].astype(np.float32)

    def aabbs(self, k=3.0):
        """Exact ellipsoid AABBs, half extent k*sqrt(diag(covariance))."""
        if not np.isfinite(k) or k < 0:
            raise ValueError('k must be finite and nonnegative')
        extent = k*np.sqrt(np.diagonal(self.covariance(), axis1=1, axis2=2))
        return self.positions-extent, self.positions+extent

    def bounds(self):
        """Union of 3-sigma bounds; an empty cloud has (+inf, -inf) bounds."""
        lo, hi = self.aabbs()
        return lo.min(axis=0, initial=np.inf), hi.max(axis=0, initial=-np.inf)

    def transformed(self, matrix4):
        """Apply a column-vector affine matrix, including nonuniform scale/shear.

        Eigenvectors/roots of M Sigma M.T become rotation/scales; axis ordering
        and normal sign may change. SH degrees 0..3 follow the polar orthogonal
        factor of M (least-squares real-basis rotation), ignoring stretch. Thus
        SH remains in its source frame for a pure positive stretch. Singular
        transforms use NumPy's SVD choice of orthogonal factor. Raw scale logs
        are discarded because the ellipsoid has changed; raw opacity is kept.
        """
        m = np.asarray(matrix4, dtype=np.float64)
        if m.shape != (4,4) or not np.isfinite(m).all() or not np.allclose(m[3], (0,0,0,1)):
            raise ValueError('matrix4 must be a finite column-vector affine 4x4 matrix')
        a = m[:3,:3]
        cov = a @ self.covariance() @ a.T
        values, axes = np.linalg.eigh(cov)
        axes[:,:,2] *= np.linalg.det(axes)[:,None]
        u, _, vt = np.linalg.svd(a)
        sh = _rotate_sh(self.sh, u @ vt)
        return SplatCloud(self.positions @ a.T + m[:3,3], np.sqrt(np.maximum(values,0)),
                          _matrix_quaternions(axes), self.opacity, sh, self.sh_degree,
                          raw_opacity_logit=self.raw_opacity_logit, colorspace=self.colorspace,
                          intrinsics=None if self.intrinsics is None else self.intrinsics.transformed(a))


def _basis(directions, degree):
    x, y, z = np.asarray(directions, dtype=np.float64).T
    b = [np.full_like(x, C0)]
    if degree >= 1:
        b += [-C1*y, C1*z, -C1*x]
    if degree >= 2:
        b += [C2[0]*x*y, C2[1]*y*z, C2[2]*(2*z*z-x*x-y*y), C2[3]*x*z, C2[4]*(x*x-y*y)]
    if degree >= 3:
        b += [C3[0]*y*(3*x*x-y*y), C3[1]*x*y*z, C3[2]*y*(4*z*z-x*x-y*y),
              C3[3]*z*(2*z*z-3*x*x-3*y*y), C3[4]*x*(4*z*z-x*x-y*y),
              C3[5]*z*(x*x-y*y), C3[6]*x*(x*x-3*y*y)]
    return np.stack(b, -1)


def eval_sh(cloud_or_sh, directions, *, offset=0.5, clamp_min=0.0):
    """Evaluate per-splat unit (position-camera) directions, returning (N,3).

    Pass offset=0, clamp_min=None for the raw SH sum. Directions are expected
    to be unit vectors; a single (3,) direction broadcasts to all splats.
    """
    sh = np.asarray(cloud_or_sh.sh if isinstance(cloud_or_sh, SplatCloud) else cloud_or_sh)
    if sh.ndim != 3 or sh.shape[2] != 3 or sh.shape[1] not in (1,4,9,16):
        raise ValueError('sh must have shape (N,K,3), K=1,4,9,16')
    d = np.broadcast_to(np.asarray(directions), (len(sh),3))
    rgb = np.einsum('nk,nkc->nc', _basis(d, int(np.sqrt(sh.shape[1]))-1), sh) + offset
    return (rgb if clamp_min is None else np.maximum(rgb, clamp_min)).astype(np.float32)


def _rotate_sh(sh, rotation):
    if np.allclose(rotation, np.eye(3), atol=1e-12, rtol=0):
        return sh
    degree = int(np.sqrt(sh.shape[1]))-1
    # Deterministic Fibonacci sphere; overdetermined, well-conditioned per degree.
    i = np.arange(64)
    z = 1-2*(i+0.5)/64
    phi = i*(np.pi*(3-np.sqrt(5)))
    d = np.stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z), -1)
    b, rotated = _basis(d, degree), _basis(d @ rotation, degree)
    result = sh.copy()
    for l in range(1, degree+1):
        sl = slice(l*l, (l+1)**2)
        t = np.linalg.lstsq(b[:,sl], rotated[:,sl], rcond=None)[0]
        result[:,sl] = np.einsum('ij,njc->nic', t, sh[:,sl])
    return result


def to_linear_color(rgb, colorspace='srgb'):
    """Render-time sRGB EOTF on evaluated RGB; linear input passes through."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if colorspace == 'linear':
        return rgb.copy()
    if colorspace != 'srgb':
        raise ValueError('colorspace must be srgb or linear')
    return np.where(rgb <= 0.04045, rgb/12.92,
                    ((np.maximum(rgb,0)+0.055)/1.055)**2.4).astype(np.float32)


def fingerprint(path):
    """Return [resolved path, size, mtime_ns], rejecting empty/unreadable files."""
    try:
        if not path:
            raise OSError('empty path')
        p = Path(path).expanduser()
        with p.open('rb') as f:
            if not f.read(1):
                raise OSError('empty file')
            stat = os.fstat(f.fileno())
        return [str(p.resolve()), stat.st_size, stat.st_mtime_ns]
    except OSError as error:
        raise ValueError(f'cannot read {path}: {error}') from error


CLOUD_CACHE_MAXSIZE = 4
CLOUD_CACHE_MAX_BYTES = 1 << 30
_cloud_cache = OrderedDict()
_cloud_cache_lock = RLock()


def load_cloud_cached(path, orientation='as_authored', colorspace='srgb'):
    """Thread-safe, size/mtime-keyed LRU of immutable decoded clouds (4 / 1 GiB)."""
    with _cloud_cache_lock:
        key = (*fingerprint(path), orientation, colorspace)
        if key in _cloud_cache:
            cloud, size = _cloud_cache.pop(key)
            _cloud_cache[key] = (cloud, size)
            return cloud
        cloud = read_ply(key[0], orientation=orientation, colorspace=colorspace)
        size = sum(a.nbytes for a in vars(cloud).values() if isinstance(a, np.ndarray))
        if size <= CLOUD_CACHE_MAX_BYTES:
            _cloud_cache[key] = (cloud, size)
        while (_cloud_cache and (len(_cloud_cache) > CLOUD_CACHE_MAXSIZE or
               sum(entry[1] for entry in _cloud_cache.values()) > CLOUD_CACHE_MAX_BYTES)):
            _cloud_cache.popitem(last=False)
        return cloud


_TYPES = dict(char='i1', uchar='u1', short='<i2', ushort='<u2', int='<i4', uint='<u4',
              float='<f4', double='<f8', int8='i1', uint8='u1', int16='<i2', uint16='<u2',
              int32='<i4', uint32='<u4', float32='<f4', float64='<f8')
_REQUIRED = ['x','y','z'] + [f'f_dc_{i}' for i in range(3)] + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]


def _header(f):
    line = f.readline(65537)
    if not line or len(line) > 65536:
        raise ValueError('truncated or invalid PLY header')
    if line.strip() != b'ply':
        raise ValueError('not a PLY file')
    elements, fmt = [], None
    while True:
        line = f.readline(65537)
        if not line or len(line) > 65536:
            raise ValueError('truncated or invalid PLY header')
        p = line.decode('ascii').split()
        if not p or p[0] in ('comment','obj_info'):
            continue
        if p[0] == 'end_header':
            break
        if p[0] == 'format' and len(p) == 3:
            fmt = p[1]
            if p[2] != '1.0':
                raise ValueError('unsupported PLY version')
        elif p[0] == 'element' and len(p) == 3:
            count = int(p[2])
            if count < 0:
                raise ValueError('negative PLY element count')
            if p[1] == 'vertex' and count > MAX_SPLATS:
                raise ValueError(f'PLY exceeds MAX_SPLATS ({MAX_SPLATS})')
            elements.append((p[1], count, []))
        elif p[0] == 'property' and elements:
            if len(p) == 3 and p[1] in _TYPES:
                prop = (p[2], _TYPES[p[1]], None)
            elif len(p) == 5 and p[1] == 'list' and p[2] in _TYPES and p[3] in _TYPES:
                prop = (p[4], _TYPES[p[3]], _TYPES[p[2]])
                if np.dtype(prop[2]).kind not in 'iu':
                    raise ValueError('PLY list count must be integer')
            else:
                raise ValueError('unsupported PLY property')
            if prop[0] in [q[0] for q in elements[-1][2]]:
                raise ValueError('duplicate PLY property')
            elements[-1][2].append(prop)
        else:
            raise ValueError('invalid PLY header')
    if fmt not in ('ascii','binary_little_endian'):
        raise ValueError('unsupported PLY format (expected ascii or binary_little_endian)')
    vertices = [e for e in elements if e[0] == 'vertex']
    if len(vertices) != 1:
        raise ValueError('PLY requires one vertex element')
    names = [p[0] for p in vertices[0][2] if p[2] is None]
    missing = sorted(set(_REQUIRED)-set(names))
    if missing:
        raise ValueError('missing required properties: '+', '.join(missing))
    rest = [n for n in names if n.startswith('f_rest_')]
    if len(rest) not in (0,9,24,45) or set(rest) != {f'f_rest_{i}' for i in range(len(rest))}:
        raise ValueError('invalid f_rest count/indexes: expected 0, 9, 24, or 45 contiguous properties')
    return fmt, elements, len(rest)


def _element(f, count, properties, binary, keep):
    dtype = np.dtype([(name, typ) for name, typ, ct in properties if ct is None])
    if not properties:
        return np.empty(count, dtype=dtype) if keep else None
    if binary and all(ct is None for _, _, ct in properties):
        size = count*dtype.itemsize
        if os.fstat(f.fileno()).st_size-f.tell() < size:
            raise ValueError('truncated binary PLY')
        if not keep:
            f.seek(size, 1)
            return None
        data = np.fromfile(f, dtype=dtype, count=count)
        if len(data) != count:
            raise ValueError('truncated binary PLY')
        return data
    data = np.empty(count, dtype=dtype) if keep else None
    for row in range(count):
        tokens = None
        if not binary:
            line = f.readline()
            if line == b'':
                raise ValueError('truncated ascii PLY')
            tokens = iter(line.split())
        def scalar(typ):
            if binary:
                raw = f.read(np.dtype(typ).itemsize)
                if len(raw) != np.dtype(typ).itemsize:
                    raise ValueError('truncated binary PLY')
                return np.frombuffer(raw, dtype=typ)[0]
            try:
                return np.asarray(next(tokens), dtype=typ).item()
            except StopIteration:
                raise ValueError('truncated ascii PLY') from None
        for name, typ, ct in properties:
            if ct is None:
                value = scalar(typ)
                if keep:
                    data[name][row] = value
            else:
                length = int(scalar(ct))
                if length < 0:
                    raise ValueError('negative PLY list count')
                if binary:
                    size = length*np.dtype(typ).itemsize
                    if os.fstat(f.fileno()).st_size-f.tell() < size:
                        raise ValueError('truncated binary PLY')
                    f.seek(size, 1)
                else:
                    for _ in range(length):
                        scalar(typ)
    return data


def _sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+e), e/(1+e))


def read_ply(path, *, orientation='as_authored', colorspace='srgb'):
    """Read standard 3DGS PLY. SH is display-referred by default, not linearized.

    COLMAP conversion rotates 180 degrees about X, including exact SH parity.
    Unknown elements/properties (including lists) are consumed in file order.
    Binary scalar vertices are loaded in one structured np.fromfile pass.
    """
    if orientation not in ('as_authored','colmap'):
        raise ValueError('orientation must be as_authored or colmap')
    if colorspace not in ('srgb','linear'):
        raise ValueError('colorspace must be srgb or linear')
    resolved, _, _ = fingerprint(path)
    try:
        with Path(resolved).open('rb') as f:
            fmt, elements, rest = _header(f)
            for name, count, props in elements:
                data = _element(f, count, props, fmt == 'binary_little_endian', name == 'vertex')
                if name == 'vertex':
                    vertex = data
    except (OSError, UnicodeError) as error:
        raise ValueError(f'cannot read {path}: {error}') from error
    def columns(names):
        return np.stack([vertex[n] for n in names], -1).astype(np.float32)
    positions = columns(['x','y','z'])
    logs = columns([f'scale_{i}' for i in range(3)])
    logits = vertex['opacity'].astype(np.float32)
    rotations = columns([f'rot_{i}' for i in range(4)])
    sh = columns([f'f_dc_{i}' for i in range(3)])[:,None,:]
    if rest:
        tail = columns([f'f_rest_{i}' for i in range(rest)]).reshape(len(vertex),3,rest//3).transpose(0,2,1)
        sh = np.concatenate((sh, tail), axis=1)
    if orientation == 'colmap':
        positions *= (1,-1,-1)
        # (0,1,0,0) Hamilton-multiplied by (w,x,y,z).
        rotations = _quaternions(rotations)[:,[1,0,3,2]] * (-1,1,-1,1)
        signs = np.array((1,-1,-1,1, -1,1,1,-1,1, -1,1,-1,-1,1,-1,1))
        sh *= signs[None,:sh.shape[1],None]
    with np.errstate(over='ignore'):
        scales = np.exp(logs)
    return SplatCloud(positions, scales, rotations, _sigmoid(logits), sh,
                      int(np.sqrt(sh.shape[1]))-1, logs, logits, colorspace)


def write_ply(cloud, path):
    """Atomically write binary little-endian 3DGS; preserve supplied raw logs/logits.

    PLY carries no colorspace/orientation metadata; pass colorspace on re-read.
    Without raw fields, zero scales and opacity endpoints encode as infinities.
    """
    names = ['x','y','z','nx','ny','nz'] + [f'f_dc_{i}' for i in range(3)]
    rest = 3*(cloud.sh.shape[1]-1)
    names += [f'f_rest_{i}' for i in range(rest)] + ['opacity']
    names += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    data = np.zeros(len(cloud), dtype=[(n,'<f4') for n in names])
    with np.errstate(divide='ignore', invalid='ignore'):
        logs = cloud.raw_scale_log if cloud.raw_scale_log is not None else np.log(cloud.scales)
        logits = cloud.raw_opacity_logit if cloud.raw_opacity_logit is not None else np.log(cloud.opacity)-np.log1p(-cloud.opacity)
    for i, name in enumerate(('x', 'y', 'z')):
        data[name] = cloud.positions[:,i]
    for i in range(3):
        data[f'f_dc_{i}'] = cloud.sh[:,0,i]
        data[f'scale_{i}'] = logs[:,i]
    coefficients = cloud.sh.shape[1]-1
    for i in range(rest):
        data[f'f_rest_{i}'] = cloud.sh[:,1+i % coefficients,i // coefficients]
    data['opacity'] = logits
    for i in range(4):
        data[f'rot_{i}'] = cloud.rotations[:,i]
    destination, temporary = Path(path).expanduser(), None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.ply.tmp', delete=False) as f:
            temporary = f.name
            header = ['ply','format binary_little_endian 1.0',f'element vertex {len(cloud)}']
            header += [f'property float {n}' for n in names] + ['end_header']
            f.write(('\n'.join(header)+'\n').encode('ascii'))
            data.tofile(f)
        os.replace(temporary, destination)
    except OSError as error:
        raise ValueError(f'cannot write {path}: {error}') from error
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
