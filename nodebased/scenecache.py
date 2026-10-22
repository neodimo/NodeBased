"""Content-keyed reuse of the path tracer's scene preparation (lane 4, plan "Progress", step B of 3).

Preparing a scene for the path tracer is the dearest part of an interactive first look on a scene with splats: the splat layer's
arrays and tree (about 310 ms for 40,000 splats), the mesh trees (30 ms) and the pack for the card (40 ms), redone on every camera
move although a camera move changes none of it. Three caches, each keyed by what the build reads and not by which Python object
carried it:

* the prepared scene (`PathScene` and its packed buffers) of a whole scene, ambient level and smoke settings: a camera move
  reuses it whole; a material edit rebuilds the cheap per-shape arrays and the pack, but not the trees below;
* the splat layer of the scene's splat instances: reused by a material or light edit, rebuilt by an edit to a splat cloud, to
  an instance's transform or to a knob the build reads (`SPLAT_FIELDS`), and, only when a normal smoothing knob is on, by a
  change of the camera position (the smoothing flips normals toward the eye);
* each mesh's bottom-level tree, keyed by the mesh's vertices, triangles, normals and uvs: a transform or material edit keeps it.

A key is built from values: a read-only array is identified by itself (the entry keeps it alive so the identity stays unique),
a writeable one by a hash of its bytes, so an array someone edited in place never matches a stale entry. `COUNTERS` says what was
built and what was reused (the tests read the difference around a render). `clear()` drops everything.
"""
from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from dataclasses import fields, is_dataclass

import numpy as np

COUNTERS = dict(scene_builds=0, scene_hits=0, splat_builds=0, splat_hits=0, blas_builds=0, blas_hits=0)
BUILD_SECONDS = 0.0       # seconds spent preparing scenes (the build and the pack) since the process started: the first look reads the difference

# What `ptsplats.build` and the `splatshade` functions it calls read from a `SplatInstance`; any other field (the indirect and
# reflection sample counts, the quality tier, the light link, the name, ...) leaves the layer as it was.
SPLAT_FIELDS = ("cloud", "matrix", "sh_degree", "opacity_scale", "scale_scale", "relight", "cast_shadows", "normal_smoothing",
                "use_intrinsics", "metallic", "roughness_scale", "intrinsics_mix")
# fields the path tracer never reads, left out of the whole-scene key
IGNORED = {"Scene": ("particles",), "Geometry": ("velocities",)}

SCENE_BYTES = 768 << 20       # prepared scenes kept, in host bytes
SCENE_ENTRIES = 2
SPLAT_BYTES = 512 << 20
SPLAT_ENTRIES = 3
BLAS_BYTES = 512 << 20
BLAS_ENTRIES = 128
_MAX_DEPTH = 10


class _Lru:
    def __init__(self, entries, nbytes):
        self.entries, self.nbytes = entries, nbytes
        self.items = OrderedDict()          # key -> (value, size, pins)
        self.total = 0

    def get(self, key):
        item = self.items.get(key)
        if item is None:
            return None
        self.items.move_to_end(key)
        return item[0]

    def put(self, key, value, size, pins):
        if size > self.nbytes:
            return
        old = self.items.pop(key, None)
        if old is not None:
            self.total -= old[1]
        self.items[key] = (value, size, pins)
        self.total += size
        while len(self.items) > self.entries or self.total > self.nbytes:
            _key, (_value, gone, _pins) = self.items.popitem(last=False)
            self.total -= gone

    def clear(self):
        self.items.clear()
        self.total = 0


_LOCK = threading.RLock()
_SCENES = _Lru(SCENE_ENTRIES, SCENE_BYTES)
_SPLATS = _Lru(SPLAT_ENTRIES, SPLAT_BYTES)
_BLASES = _Lru(BLAS_ENTRIES, BLAS_BYTES)


def clear():
    """Drop every cached preparation (the host copies of the scenes, splat layers and mesh trees)."""
    with _LOCK:
        for cache in (_SCENES, _SPLATS, _BLASES):
            cache.clear()


def counters():
    return dict(COUNTERS)


def _frozen(array):
    """True when neither `array` nor any array it is a view of can be written through."""
    while isinstance(array, np.ndarray):
        if array.flags.writeable:
            return False
        array = array.base
    return True


SMALL_ARRAY_BYTES = 64 << 10      # hashed by content whether or not it is frozen: cheaper than a miss on a matrix made anew


def _array_key(array, pins):
    if array.nbytes > SMALL_ARRAY_BYTES and _frozen(array):
        pins.append(array)
        return ("a", id(array), array.shape, array.dtype.str)
    data = np.ascontiguousarray(array)
    digest = hashlib.blake2b(memoryview(data).cast("B") if data.size else b"", digest_size=16).digest()
    return ("h", array.shape, array.dtype.str, digest)


def fingerprint(value, pins, depth=0):
    """A hashable stand-in for `value` that is equal for equal content (see the module docstring). Objects it cannot read by
    value are identified by themselves and added to `pins`, which the cache entry holds so the identity cannot be reused."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _array_key(value, pins)
    if depth < _MAX_DEPTH:
        if isinstance(value, (tuple, list)):
            return ("t", tuple(fingerprint(v, pins, depth + 1) for v in value))
        if isinstance(value, dict):
            return ("d", tuple(sorted(((str(k), fingerprint(v, pins, depth + 1)) for k, v in value.items()))))
        if is_dataclass(value) and not isinstance(value, type):
            skip = IGNORED.get(type(value).__name__, ())
            return (type(value).__name__,
                    tuple((f.name, fingerprint(getattr(value, f.name), pins, depth + 1)) for f in fields(value)
                          if f.name not in skip))
    pins.append(value)
    return ("o", id(value))


def _nbytes(*objects):
    total = 0
    for obj in objects:
        for value in (vars(obj).values() if hasattr(obj, "__dict__") else ()):
            if isinstance(value, np.ndarray):
                total += value.nbytes
            elif isinstance(value, (list, tuple)):
                total += sum(v.nbytes for v in value if isinstance(v, np.ndarray))
    return total


# --- the mesh trees -----------------------------------------------------------------------------------------------------------

def blas(geometry, build):
    """The bottom-level tree of `geometry`'s mesh: the cached one for equal vertices, triangles, normals and uvs, else
    `build(geometry)` (kept for next time)."""
    pins = []
    key = tuple(None if a is None else _array_key(a, pins)
                for a in (geometry.vertices, geometry.triangles, geometry.normals, geometry.uvs))
    with _LOCK:
        hit = _BLASES.get(key)
    if hit is not None:
        COUNTERS["blas_hits"] += 1
        return hit
    made = build(geometry)
    COUNTERS["blas_builds"] += 1
    size = _nbytes(made, made.tris, made.bvh) + sum(a.nbytes for a in pins if isinstance(a, np.ndarray)) // 2
    with _LOCK:
        _BLASES.put(key, made, max(size, 1), pins)
    return made


# --- the splat layer ----------------------------------------------------------------------------------------------------------

def splat_layer(scene, eye, build):
    """The splat layer of `scene.splats` for the camera position `eye`: the cached one when every instance reads the same, else
    `build()` (kept for next time). A layer is handed out as a shallow copy, because the scene build writes the light-link
    masks into it and a second prepared scene must not see the first one's."""
    pins = []
    smoothing = any(int(getattr(item, "normal_smoothing", 0) or 0) > 0 for item in scene.splats)
    key = (tuple(tuple(fingerprint(getattr(item, name, None), pins) for name in SPLAT_FIELDS) for item in scene.splats),
           tuple(float(v) for v in eye) if smoothing and eye is not None else None)
    with _LOCK:
        hit = _SPLATS.get(key)
    if hit is not None:
        COUNTERS["splat_hits"] += 1
        layer = copy.copy(hit)
        layer.excl = None
        return layer
    layer = build()
    COUNTERS["splat_builds"] += 1
    if layer is not None:
        size = _nbytes(layer, layer.hit, layer.shadow, layer.bvh)
        with _LOCK:
            _SPLATS.put(key, layer, max(size, 1), [*pins, scene.splats])
        layer = copy.copy(layer)
        layer.excl = None
    return layer


# --- the whole prepared scene -------------------------------------------------------------------------------------------------

def _scene_key(scene, ambient, eye, volume):
    pins = []
    smoothing = any(int(getattr(item, "normal_smoothing", 0) or 0) > 0 for item in getattr(scene, "splats", ()))
    key = (fingerprint(scene, pins), float(ambient), tuple(float(v) for v in eye) if smoothing and eye is not None else None,
           fingerprint(volume, pins))
    pins.append(scene)
    return key, pins


def prepared_scene(scene, ambient, eye, volume):
    """(key, pins, entry): `entry` is the cached `(PathScene, packed)` for this content or None; pass `key` and `pins` to
    `remember_scene` after a build."""
    key, pins = _scene_key(scene, ambient, eye, volume)
    with _LOCK:
        hit = _SCENES.get(key)
    if hit is not None:
        COUNTERS["scene_hits"] += 1
    return key, pins, hit


def is_prepared(scene, ambient, eye, volume=None):
    """True when `prepared_scene` would hit: the first look asks before it sizes itself, so a camera move is not charged the cost
    of a build it will not make."""
    key, _pins = _scene_key(scene, ambient, eye, volume)
    with _LOCK:
        return key in _SCENES.items


def add_build_seconds(seconds):
    global BUILD_SECONDS
    BUILD_SECONDS += seconds


def remember_scene(key, pins, ps, packed):
    COUNTERS["scene_builds"] += 1
    size = _nbytes(ps, packed) + (_nbytes(ps.splats) if ps.splats is not None else 0)
    with _LOCK:
        _SCENES.put(key, (ps, packed), max(size, 1), pins)
