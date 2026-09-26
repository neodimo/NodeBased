"""Environment light: an equirectangular scene-linear image, prefiltered for a physically based BRDF.

Split sum (Karis 2013): diffuse light is the cosine-convolved radiance, kept as 9 spherical-harmonic
coefficients (Ramamoorthi and Hanrahan 2001); specular light is the radiance blurred by a GGX lobe at a
few roughness levels, looked up along the reflection direction and scaled by an analytic DFG term
(`dfg`). Everything is NumPy, deterministic, and prefiltered once per image fingerprint and cached.

Conventions. The map is latitude-longitude, row 0 up. A direction `d` (unit, world) reads
`u = 0.5 + atan2(d.x, -d.z) / 2pi`, `v = acos(d.y) / pi`, so the map centre looks down -Z. `rotation`
turns the environment about +Y in degrees (positive counter-clockwise seen from above). Radiance is
returned as the environment's own units times `intensity` and `tint`: a uniform map of 1 lights a white
diffuse surface to exactly 1 (no 1/pi left over), which is what makes the white furnace test hold.
"""
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

FORMAT = 1                    # bump when the prefilter maths changes; part of the cache key
LEVEL_ROUGHNESS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PREFILTER_SIZE = (64, 32)     # width, height of the blurred levels; level 0 keeps the full map
MAX_SIDE = 2048               # the full-resolution map is decimated above this many pixels wide
_CHUNK = 256


@dataclass(frozen=True, eq=False)
class Environment:
    """One environment light as a scene item. `rgb` is (H, W, 3) linear float32 and never mutated."""
    rgb: np.ndarray
    fingerprint: str
    intensity: float = 1.0
    rotation: float = 0.0        # degrees about +Y
    blur: float = 0.0            # 0..1 added to every roughness, so 1 makes the whole light diffuse
    tint: tuple = (1.0, 1.0, 1.0)
    parent: np.ndarray = field(default_factory=lambda: np.eye(4))

    def _pre(self):
        return prefilter(self.rgb, self.fingerprint)

    def _local(self, directions):
        """World directions in the map's frame: the environment's rotation and its parent's turn undone."""
        d = np.asarray(directions, dtype=np.float64)
        a = np.radians(self.rotation)
        c, s = np.cos(a), np.sin(a)
        # rotate by -a about +Y, then by the inverse of the parent's rotation
        turn = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
        parent = np.asarray(self.parent, dtype=np.float64)[:3, :3]
        norms = np.linalg.norm(parent, axis=0)
        rot = parent / np.where(norms > 1e-12, norms, 1.0)
        return d @ (rot.T @ turn).T

    def _gain(self):
        return float(self.intensity) * np.asarray(self.tint, dtype=np.float64)

    def diffuse(self, normals):
        """Cosine-convolved radiance around `normals` (N,3), linear RGB (N,3). Uniform 1 gives 1."""
        n = self._local(normals)
        return _sh_irradiance(self._pre().sh, n) * self._gain()

    def specular(self, directions, roughness):
        """Radiance blurred by the GGX lobe of `roughness` (N,) along `directions` (N,3), (N,3)."""
        pre = self._pre()
        r = 1 - (1 - np.clip(np.asarray(roughness, dtype=np.float64), 0, 1)) * (1 - float(np.clip(self.blur, 0, 1)))
        return pre.lookup(self._local(directions), r) * self._gain()

    def background(self, directions):
        """The unblurred map along `directions`: what a camera looking out of the scene would see."""
        return self.specular(directions, np.zeros(len(directions)))


def fingerprint_of(rgb):
    """Content hash of a map, the cache key of its prefilter."""
    a = np.ascontiguousarray(rgb, dtype=np.float32)
    h = hashlib.sha1(a.tobytes())
    h.update(str(a.shape).encode())
    h.update(str(FORMAT).encode())
    return h.hexdigest()


# --- sphere maths ---------------------------------------------------------------------------

def direction_grid(width, height):
    """Unit directions of every texel centre of a `width` x `height` map, and their solid angles."""
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = (u - 0.5) * 2 * np.pi
    theta = v * np.pi
    x = np.sin(theta)[:, None] * np.sin(phi)[None, :]
    y = np.cos(theta)[:, None] * np.ones(width)[None, :]
    z = -np.sin(theta)[:, None] * np.cos(phi)[None, :]
    d = np.stack((x, y, z), axis=-1)
    omega = (np.sin(theta) * (np.pi / height) * (2 * np.pi / width))[:, None] * np.ones(width)[None, :]
    omega *= 4 * np.pi / omega.sum()          # a coarse grid still integrates a uniform map exactly
    return d, omega


def _uv(d):
    d = np.asarray(d, dtype=np.float64)
    u = 0.5 + np.arctan2(d[..., 0], -d[..., 2]) / (2 * np.pi)
    v = np.arccos(np.clip(d[..., 1], -1, 1)) / np.pi
    return u, v


def sample_map(image, d):
    """Bilinear lookup of an (H, W, 3) latitude-longitude `image` along directions `d` (N,3)."""
    h, w = image.shape[:2]
    u, v = _uv(d)
    x = u * w - 0.5
    y = np.clip(v * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0m, x1m = x0 % w, (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    top = image[y0, x0m] * (1 - fx) + image[y0, x1m] * fx
    bottom = image[y1, x0m] * (1 - fx) + image[y1, x1m] * fx
    return top * (1 - fy) + bottom * fy


# --- diffuse: spherical harmonics -----------------------------------------------------------

def _sh_basis(d):
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack((np.full_like(x, 0.282095), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
                     1.092548 * x * z, 0.546274 * (x * x - y * y)), axis=-1)


_BAND = np.array((np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3, np.pi / 4, np.pi / 4, np.pi / 4,
                  np.pi / 4, np.pi / 4)) / np.pi        # the cosine lobe per coefficient, over pi


def _sh_project(image):
    h, w = image.shape[:2]
    d, omega = direction_grid(w, h)
    basis = _sh_basis(d)                                # (H, W, 9)
    return np.einsum('hwk,hwc,hw->kc', basis, image.astype(np.float64), omega)


def _sh_irradiance(coefficients, normals):
    return (_sh_basis(np.asarray(normals, dtype=np.float64)) * _BAND) @ coefficients


# --- specular: GGX-blurred levels -------------------------------------------------------------

def _blur_level(source, source_dirs, source_omega, roughness, size):
    """`source` blurred by a GGX lobe of `roughness`, sampled at a `size` grid of reflection directions."""
    w, h = size
    dirs, _ = direction_grid(w, h)
    flat_out = dirs.reshape(-1, 3)
    flat_src = source_dirs.reshape(-1, 3)
    radiance = source.reshape(-1, 3).astype(np.float64)
    omega = source_omega.reshape(-1)
    alpha = max(roughness, 1e-3) ** 2
    out = np.empty((len(flat_out), 3))
    for start in range(0, len(flat_out), _CHUNK):
        r = flat_out[start:start + _CHUNK]
        rl = r @ flat_src.T                              # N = V = R: cos between reflection and light
        nh2 = (1 + rl) / 2                               # (N.H)^2 for H = normalise(R + L)
        d = alpha ** 2 / (np.pi * (nh2 * (alpha ** 2 - 1) + 1) ** 2)
        weight = d * np.maximum(rl, 0) * omega
        total = weight.sum(axis=1, keepdims=True)
        out[start:start + _CHUNK] = (weight @ radiance) / np.maximum(total, 1e-30)
    return out.reshape(h, w, 3)


def _decimate(rgb, max_width):
    h, w = rgb.shape[:2]
    while w > max_width and w % 2 == 0 and h % 2 == 0:
        rgb = rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
        h, w = h // 2, w // 2
    return rgb


@dataclass(frozen=True, eq=False)
class Prefiltered:
    sh: np.ndarray               # (9, 3)
    levels: tuple                # (H, W, 3) maps, roughness LEVEL_ROUGHNESS[i]

    def lookup(self, directions, roughness):
        r = np.clip(np.asarray(roughness, dtype=np.float64), 0, 1)
        position = r * (len(self.levels) - 1)
        lower = np.minimum(np.floor(position).astype(np.int64), len(self.levels) - 2)
        fraction = (position - lower)[:, None]
        out = np.zeros((len(r), 3))
        for i in np.unique(lower):
            pick = lower == i
            a = sample_map(self.levels[i], directions[pick])
            b = sample_map(self.levels[i + 1], directions[pick])
            out[pick] = a * (1 - fraction[pick]) + b * fraction[pick]
        return out


_CACHE = OrderedDict()
_CACHE_LIMIT = 4
prefilter_runs = 0               # prefilters computed in this process; the tests count them


def prefilter(rgb, fingerprint=None):
    """The diffuse SH and specular levels of `rgb`, computed once per fingerprint."""
    global prefilter_runs
    key = fingerprint or fingerprint_of(rgb)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    prefilter_runs += 1
    rgb = np.asarray(rgb, dtype=np.float32)
    full = _decimate(rgb, MAX_SIDE)
    small = _decimate(rgb, PREFILTER_SIZE[0] * 2)
    h, w = small.shape[:2]
    src_dirs, src_omega = direction_grid(w, h)
    levels = [full.astype(np.float32)]
    for roughness in LEVEL_ROUGHNESS[1:]:
        levels.append(_blur_level(small, src_dirs, src_omega, roughness, PREFILTER_SIZE).astype(np.float32))
    for level in levels:
        level.flags.writeable = False
    result = Prefiltered(_sh_project(small), tuple(levels))
    _CACHE[key] = result
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)
    return result


def clear_cache():
    _CACHE.clear()


# --- the BRDF terms of the split sum ---------------------------------------------------------

def dfg(n_dot_v, roughness):
    """Karis's analytic fit of the split-sum BRDF table: `(A, B)` with `F0 * A + B` the specular albedo."""
    nv = np.clip(np.asarray(n_dot_v, dtype=np.float64), 1e-4, 1)
    r = np.asarray(roughness, dtype=np.float64)
    c0 = np.array((-1.0, -0.0275, -0.572, 0.022))
    c1 = np.array((1.0, 0.0425, 1.04, -0.04))
    t = r[..., None] * c0 + c1
    a004 = np.minimum(t[..., 0] ** 2, 2 ** (-9.28 * nv)) * t[..., 0] + t[..., 1]
    return -1.04 * a004 + t[..., 2], 1.04 * a004 + t[..., 3]
