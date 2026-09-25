"""Deterministic particle emitter and solver behind ParticleEmitter3D and ParticleCache3D.

The solver is a `simcache` step function: `ParticleEmitter.initial_state` and
`ParticleEmitter.step` are handed to `simcache.solve_to_frame`, which owns the substep loop,
the checkpoints and cancellation. See docs/SIMULATION.md for the time model and the knob
vocabulary (Nuke's ParticleEmitter names where Nuke has one, Houdini's POP Source names otherwise).

Determinism rule: every substep draws its randomness from
`np.random.default_rng((seed, frame - start_frame, substep))` and from nothing else, so a frame
comes out bit-identical however the solve was split across sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import simcache

EMIT_FROM = ("point", "vertices", "surface", "volume")
RATE_UNITS = ("per_frame", "per_second")
# Emission count rule: births accumulate `rate / substeps` per substep in a carry and a substep
# emits floor(carry + COUNT_EPSILON). For a constant rate the total after n whole frames is
# floor(rate * n + COUNT_EPSILON), so 7.5 per frame gives 7, 15, 22, 30 ... and 100 gives 100n.
COUNT_EPSILON = 1e-9
# Solver-call counter for tests and benchmarks: one tick per substep actually solved.
SOLVER_STATS = {"steps": 0}
_VOLUME_BATCH = 1024
_VOLUME_ATTEMPTS = 64
_VOLUME_TRIANGLE_CAP = 20000

ARRAYS = ("position", "velocity", "age", "life", "size", "color", "id")


def _empty_arrays():
    return {"position": np.zeros((0, 3), np.float32), "velocity": np.zeros((0, 3), np.float32),
            "age": np.zeros(0, np.int32), "life": np.zeros(0, np.int32),
            "size": np.zeros(0, np.float32), "color": np.zeros((0, 4), np.float32),
            "id": np.zeros(0, np.int64)}


def _state(arrays, meta):
    return simcache.State(arrays, meta, copy=False)   # solver arrays are fresh every step


class ParticleSource:
    """Emission geometry in world space: vertices, and triangles with areas and face normals."""

    def __init__(self, geometry=None):
        self.vertices = np.zeros((0, 3), np.float64)
        self.triangles = np.zeros((0, 3, 3), np.float64)
        self.cumulative_area = np.zeros(0, np.float64)
        self.face_normals = np.zeros((0, 3), np.float64)
        self.total_area = 0.0
        self.low = self.high = None
        if geometry is None or not len(geometry.vertices):
            return
        matrix = geometry.world_matrix().astype(np.float64)
        world = (matrix[:3, :3] @ geometry.vertices.astype(np.float64).T).T + matrix[:3, 3]
        self.vertices = world
        self.low, self.high = world.min(axis=0), world.max(axis=0)
        if len(geometry.triangles):
            tris = world[geometry.triangles]
            cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            area = 0.5 * np.linalg.norm(cross, axis=1)
            self.triangles = tris
            self.face_normals = cross / np.maximum(2 * area, 1e-12)[:, None]
            self.cumulative_area = np.cumsum(area)
            self.total_area = float(self.cumulative_area[-1])

    @property
    def has_surface(self):
        return self.total_area > 0.0


def _inside(points, triangles):
    """Ray-parity inside test along +X (Moller-Trumbore), chunked over the triangles."""
    inside = np.zeros(len(points), np.int32)
    for start in range(0, len(triangles), 512):
        chunk = triangles[start:start + 512]
        v0, e1, e2 = chunk[:, 0], chunk[:, 1] - chunk[:, 0], chunk[:, 2] - chunk[:, 0]
        # Ray direction is +X with a small fixed tilt so axis-aligned faces are never grazed.
        direction = np.array((1.0, 0.000731, 0.000517))
        h = np.cross(direction, e2)                                   # (T,3)
        det = np.einsum("tj,tj->t", e1, h)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = points[:, None, :] - v0[None]                             # (P,T,3)
        u = np.einsum("ptj,tj->pt", s, h) * inv
        q = np.cross(s, e1[None])
        v = np.einsum("j,ptj->pt", direction, q) * inv
        t = np.einsum("tj,ptj->pt", e2, q) * inv
        hit = ok[None] & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-12)
        inside += hit.sum(axis=1).astype(np.int32)
    return (inside % 2) == 1


# --- forces (step 2b) --------------------------------------------------------------------------

FORCE_KINDS = ("ParticleGravity3D", "ParticleDrag3D", "ParticleWind3D", "ParticleTurbulence3D")
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(values):
    """splitmix64 finaliser over an unsigned 64-bit array (wraps on purpose)."""
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _hash_unit(seed, salt, *columns):
    """A repeatable uniform value in [0, 1) per row from integer columns, a seed and a salt."""
    with np.errstate(over="ignore"):
        h = _mix(np.uint64((int(seed) * 1000003 + int(salt)) & 0xFFFFFFFFFFFFFFFF) + np.zeros_like(
            np.asarray(columns[0]).astype(np.uint64)))
        for column in columns:
            h = _mix(h ^ np.asarray(column).astype(np.int64).astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def _smooth(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _value_noise(points, seed, salt):
    """Smooth lattice noise in [-1, 1] at (N,3) points: quintic-interpolated hashed lattice values."""
    base = np.floor(points)
    frac = points - base
    cell = base.astype(np.int64)
    w = _smooth(frac)
    total = np.zeros(len(points))
    for corner in range(8):
        dx, dy, dz = corner & 1, (corner >> 1) & 1, (corner >> 2) & 1
        value = _hash_unit(seed, salt, cell[:, 0] + dx, cell[:, 1] + dy, cell[:, 2] + dz) * 2.0 - 1.0
        weight = ((w[:, 0] if dx else 1.0 - w[:, 0]) * (w[:, 1] if dy else 1.0 - w[:, 1])
                  * (w[:, 2] if dz else 1.0 - w[:, 2]))
        total += weight * value
    return total


def _fbm(points, seed, salt, octaves):
    total, amplitude, scale = np.zeros(len(points)), 1.0, 1.0
    for octave in range(max(1, int(octaves))):
        total += amplitude * _value_noise(points * scale, seed, salt + 101 * octave)
        amplitude *= 0.5
        scale *= 2.0
    return total


def _noise_1d(t, seed):
    """Smooth repeatable noise in [-1, 1] on a scalar time (an array of one value per call)."""
    base = math.floor(t)
    w = _smooth(t - base)
    a = _hash_unit(seed, 7919, np.array([base]))[0] * 2.0 - 1.0
    b = _hash_unit(seed, 7919, np.array([base + 1]))[0] * 2.0 - 1.0
    return a + (b - a) * w


def turbulence_field(positions, mode, size, octaves, seed):
    """The turbulence acceleration direction at (N,3) `positions`, in units of 1 / (lattice cell).

    `curl` is the curl of a three-component noise potential (divergence free, so particles swirl
    rather than clump); `gradient` is the gradient of one scalar noise potential. Both come from
    central differences of `_fbm` and are multiplied by `size`, so `strength` does not change
    meaning with the feature size. Zero mean over any large volume by symmetry of the lattice values.
    """
    points = positions.astype(np.float64) / float(size)
    eps = 1e-3
    offsets = np.eye(3) * eps

    def derivative(salt, axis):
        return (_fbm(points + offsets[axis], seed, salt, octaves)
                - _fbm(points - offsets[axis], seed, salt, octaves)) / (2.0 * eps)
    if mode == "gradient":
        return np.stack([derivative(11, axis) for axis in range(3)], axis=1)
    dz_y, dy_z = derivative(29, 1), derivative(23, 2)
    dx_z, dz_x = derivative(17, 2), derivative(29, 0)
    dy_x, dx_y = derivative(23, 0), derivative(17, 1)
    return np.stack((dz_y - dy_z, dx_z - dz_x, dy_x - dx_y), axis=1)


class ParticleForce:
    """One force node's contribution: knobs, curves and the velocity update it makes each substep.

    Accelerations are in units per frame squared and the substep timestep is `1 / substeps` frames,
    like the emitter's speeds (docs/SIMULATION.md, "Substeps and units"). The update is semi-implicit
    Euler: `v += a * dt`, then the emitter integrates position with the new `v`.
    """

    def __init__(self, kind, params, curves=None):
        self.kind = kind
        self.params = dict(params)
        self.curves = curves or None
        self._resolved = {}

    def frame_params(self, frame):
        cached = self._resolved.get(frame)
        if cached is None:
            from .core import SPECS, LIMITS
            from .animation import resolve_params
            cached = resolve_params({"params": self.params}, self.curves, frame,
                                    SPECS[self.kind]["params"], LIMITS)
            if len(self._resolved) > 4096:
                self._resolved.clear()
            self._resolved[frame] = cached
        return cached

    def selected(self, ids, p):
        """Boolean mask of the particles this force touches: a seeded, fixed fraction by id."""
        probability = float(p["probability"])
        if probability >= 1.0:
            return np.ones(len(ids), bool)
        return _hash_unit(int(p["seed"]), 1, ids) < probability

    def apply(self, arrays, velocity, frame, substep, substeps):
        """`velocity` (N,3 float64) after this force acts for one substep."""
        p = self.frame_params(frame)
        if not (int(p["from_frame"]) <= frame <= int(p["to_frame"])) or not len(velocity):
            return velocity
        mask = self.selected(arrays["id"], p)
        if not mask.any():
            return velocity
        dt = 1.0 / substeps
        kind = self.kind
        if kind == "ParticleDrag3D":
            speed = np.linalg.norm(velocity[mask], axis=1, keepdims=True)
            rate = float(p["drag"]) + float(p["drag_quadratic"]) * speed
            velocity = velocity.copy()
            velocity[mask] *= np.exp(-rate * dt)
            return velocity
        if kind == "ParticleGravity3D":
            accel = np.array((p["gravity_x"], p["gravity_y"], p["gravity_z"]), np.float64) * float(p["strength"])
            accel = np.broadcast_to(accel, (int(mask.sum()), 3))
        elif kind == "ParticleWind3D":
            direction = np.array((p["wind_x"], p["wind_y"], p["wind_z"]), np.float64)
            length = np.linalg.norm(direction)
            direction = direction / length if length > 1e-12 else np.zeros(3)
            t = frame + substep / substeps
            gust = 1.0 + float(p["wind_gust"]) * _noise_1d(t * float(p["wind_gust_rate"]), int(p["seed"]))
            accel = np.broadcast_to(direction * float(p["strength"]) * gust, (int(mask.sum()), 3))
        elif kind == "ParticleTurbulence3D":
            field = turbulence_field(arrays["position"][mask], p["turb_mode"], p["turb_size"],
                                     p["octaves"], int(p["seed"]))
            accel = field * (float(p["turb_size"]) * float(p["strength"]))
        else:
            return velocity
        velocity = velocity.copy()
        velocity[mask] += accel * dt
        return velocity

    def identity(self, expressions=None):
        return {"kind": self.kind, "params": self.params, "curves": self.curves,
                "expressions": expressions, "format": 1}


class ParticleEmitter:
    """One deterministic run: the emitter knobs, the emission source and the timeline model.

    `params` are the node's stored (unresolved) knobs; `curves` its animation curves. The run
    constants (`start_frame`, `substeps`, `seed`, `max_particles`, `emit_from`, `emit_rate_unit`)
    are read from the stored values; every other knob is resolved per frame, so an animated rate
    or an animated emitter transform works and is part of the run's identity through `curves`.
    """

    def __init__(self, params, curves=None, source=None, fps=24.0, kind="ParticleEmitter3D"):
        self.params = dict(params)
        self.curves = curves or None
        self.source = source if source is not None else ParticleSource()
        self.fps = float(fps)
        self.kind = kind
        self.start_frame = int(self.params["start_frame"])
        self.substeps = int(self.params["substeps"])
        self.seed = int(self.params["seed"])
        self.max_particles = int(self.params["max_particles"])
        self.emit_from = self.params["emit_from"]
        self.per_second = self.params["emit_rate_unit"] == "per_second"
        self.forces = ()          # ParticleForce chain, upstream first (see `with_force`)
        self._resolved = {}

    def with_force(self, force):
        """A copy of this emitter that also applies `force` after the forces already chained."""
        import copy
        other = copy.copy(self)
        other.forces = self.forces + (force,)
        return other

    def frame_params(self, frame):
        cached = self._resolved.get(frame)
        if cached is None:
            from .core import SPECS, LIMITS
            from .animation import resolve_params
            cached = resolve_params({"params": self.params}, self.curves, frame,
                                    SPECS[self.kind]["params"], LIMITS)
            if len(self._resolved) > 4096:
                self._resolved.clear()
            self._resolved[frame] = cached
        return cached

    # --- simcache contract --------------------------------------------------------------------

    def initial_state(self, seed):
        return _state(_empty_arrays(), {"emitted": 0, "carry": 0.0, "dropped": 0})

    def step(self, state, frame, substep, seed):
        SOLVER_STATS["steps"] += 1
        arrays, meta = state.arrays, dict(state.meta)
        p = self.frame_params(frame)
        rate = float(p["emit_rate"])
        per_frame = rate / self.fps if self.per_second else rate
        carry = float(meta["carry"]) + per_frame / self.substeps
        births = max(0, int(math.floor(carry + COUNT_EPSILON)))
        carry -= births
        alive = len(arrays["id"])
        room = max(0, self.max_particles - alive)
        emit = min(births, room)
        if emit < births:
            meta["dropped"] = int(meta["dropped"]) + births - emit
        if emit:
            rng = np.random.default_rng((self.seed, frame - self.start_frame, substep))
            new = self._births(emit, int(meta["emitted"]), p, rng, (frame - self.start_frame, substep))
            arrays = {name: np.concatenate((arrays[name], new[name])) for name in ARRAYS}
            meta["emitted"] = int(meta["emitted"]) + emit
        meta["carry"] = carry
        dt = np.float32(1.0 / self.substeps)
        if self.forces and len(arrays["id"]):
            velocity = arrays["velocity"].astype(np.float64)
            for force in self.forces:
                velocity = force.apply(arrays, velocity, frame, substep, self.substeps)
            arrays = {**arrays, "velocity": velocity.astype(np.float32)}
        position = arrays["position"] + arrays["velocity"] * dt
        age = arrays["age"] + np.int32(1)
        alive_mask = age < arrays["life"]
        if alive_mask.all():
            result = {**arrays, "position": position, "age": age}
        else:
            result = {"position": position[alive_mask], "velocity": arrays["velocity"][alive_mask],
                      "age": age[alive_mask], "life": arrays["life"][alive_mask],
                      "size": arrays["size"][alive_mask], "color": arrays["color"][alive_mask],
                      "id": arrays["id"][alive_mask]}
        return _state(result, meta)

    # --- births -------------------------------------------------------------------------------

    def _births(self, count, first_id, p, rng, stream):
        from .scene3d import _transform_from
        matrix = _transform_from(p).matrix().astype(np.float64)
        linear = matrix[:3, :3]
        draws = rng.random((count, 8))
        local, normals = self._locations(count, draws, stream)
        position = local @ linear.T + matrix[:3, 3]
        direction = np.array((p["emit_dir_x"], p["emit_dir_y"], p["emit_dir_z"]), np.float64)
        if not np.linalg.norm(direction) > 1e-12:
            direction = np.array((0.0, 1.0, 0.0))
        base = np.broadcast_to(direction / np.linalg.norm(direction), (count, 3)).copy()
        if normals is not None and int(p["direction_from_normals"]):
            base = normals
        base = base @ linear.T
        base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
        # Uniform in the solid angle of a cone: cos(theta) between cos(spread) and 1.
        cos_spread = math.cos(math.radians(float(p["spread"])))
        cos_theta = 1.0 - draws[:, 3] * (1.0 - cos_spread)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta ** 2))
        phi = 2.0 * math.pi * draws[:, 4]
        helper = np.where(np.abs(base[:, 1:2]) < 0.99, np.array((0.0, 1.0, 0.0)), np.array((1.0, 0.0, 0.0)))
        side = np.cross(base, helper)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
        up = np.cross(base, side)
        aimed = (base * cos_theta[:, None] + side * (sin_theta * np.cos(phi))[:, None]
                 + up * (sin_theta * np.sin(phi))[:, None])
        signed = lambda column, variance: 1.0 + float(variance) * (2.0 * column - 1.0)  # noqa: E731
        speed = float(p["emit_speed"]) * signed(draws[:, 5], p["speed_variance"])
        life_frames = float(p["life"]) * signed(draws[:, 6], p["life_variance"])
        size = float(p["particle_size"]) * signed(draws[:, 7], p["size_variance"])
        rgba = np.array([p["red"], p["green"], p["blue"], p["alpha"]], np.float64)
        color = np.broadcast_to(np.array((rgba[0] * rgba[3], rgba[1] * rgba[3], rgba[2] * rgba[3], rgba[3])),
                                (count, 4))
        life_ticks = np.maximum(1, np.rint(np.maximum(life_frames, 0.0) * self.substeps)).astype(np.int32)
        return {"position": position.astype(np.float32),
                "velocity": (aimed * speed[:, None]).astype(np.float32),
                "age": np.zeros(count, np.int32), "life": life_ticks,
                "size": np.maximum(size, 0.0).astype(np.float32),
                "color": color.astype(np.float32),
                "id": np.arange(first_id, first_id + count, dtype=np.int64)}

    def _locations(self, count, draws, stream):
        """Emission points in the emitter's local space, and face normals when they exist."""
        source, mode = self.source, self.emit_from
        origin = np.zeros((count, 3))
        if mode == "point" or source.low is None:
            return origin, None
        if mode == "vertices" or (mode == "surface" and not source.has_surface):
            picks = np.minimum((draws[:, 0] * len(source.vertices)).astype(np.int64), len(source.vertices) - 1)
            return source.vertices[picks], None
        if mode == "surface" or (mode == "volume" and not source.has_surface):
            return self._surface(count, draws)
        return self._volume(count, draws, stream)

    def _surface(self, count, draws):
        source = self.source
        target = draws[:, 0] * source.total_area
        tri = np.minimum(np.searchsorted(source.cumulative_area, target, side="right"),
                         len(source.cumulative_area) - 1)
        r1, r2 = np.sqrt(draws[:, 1]), draws[:, 2]
        a, b, c = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
        corners = source.triangles[tri]
        points = corners[:, 0] * a[:, None] + corners[:, 1] * b[:, None] + corners[:, 2] * c[:, None]
        return points, source.face_normals[tri]

    def _volume(self, count, draws, stream):
        """Uniform inside a closed mesh (ray parity, rejection from its bounding box).

        The candidate stream is its own generator seeded from the substep's indices, so the volume
        draw never disturbs the main draws. A mesh that is not closed, or too large to test, falls
        back to surface points after `_VOLUME_ATTEMPTS` empty batches.
        """
        source = self.source
        if len(source.triangles) > _VOLUME_TRIANGLE_CAP:
            return self._surface(count, draws)
        rng = np.random.default_rng((self.seed, stream[0], stream[1], 1))
        span = np.maximum(source.high - source.low, 1e-9)
        found = []
        have = 0
        for _ in range(_VOLUME_ATTEMPTS):
            candidates = source.low + rng.random((_VOLUME_BATCH, 3)) * span
            keep = candidates[_inside(candidates, source.triangles)]
            found.append(keep)
            have += len(keep)
            if have >= count:
                break
        if have < count:
            return self._surface(count, draws)
        return np.concatenate(found)[:count], None


@dataclass(frozen=True, eq=False)
class ParticleStream:
    """A run definition: enough to solve any frame of it, carried beside the particles."""
    emitter: ParticleEmitter
    run: str
    start_frame: int
    substeps: int
    seed: int


def build_stream(evaluator, doc, key, node, cancel=None):
    """The `ParticleStream` for one ParticleEmitter3D node of `doc`.

    The emission geometry is sampled once, at the start frame, so a run does not depend on the
    frame being viewed: an animated emitter mesh is frozen at the start frame (docs/SIMULATION.md).
    The run's identity is the digest of that geometry plus the stored knobs, the curves and
    expressions on them, and the document frame rate when the rate is per second.
    """
    params = node["params"]
    start = int(params["start_frame"])
    geometry, digest = None, None
    source_key = node["inputs"].get("geo")
    if source_key is not None and params["emit_from"] != "point":
        geometry, digest = evaluator.evaluate_raster(doc, source_key, cancel, frame=start,
                                                     typed=True, return_digest=True)
    curves = doc.get("animation", {}).get("curves", {}).get(key)
    expressions = doc.get("expressions", {}).get(key)
    fps = float(doc.get("time", {}).get("fps", 24.0))
    emitter = ParticleEmitter(params, curves, ParticleSource(geometry), fps)
    identity = {"kind": node["type"], "params": params, "curves": curves, "expressions": expressions,
                "fps": fps if emitter.per_second else None, "format": 1}
    return ParticleStream(emitter, simcache.run_key(digest, identity), start, emitter.substeps, emitter.seed)


def extend_stream(stream, doc, key, node):
    """`stream` with the force node `key` chained on: same emission, a new run identity.

    The run is `run_key(old run, force identity)`, so changing any force knob, curve or expression
    (or adding, removing or reordering a force) abandons the old frames exactly like an emitter edit.
    """
    curves = doc.get("animation", {}).get("curves", {}).get(key)
    expressions = doc.get("expressions", {}).get(key)
    force = ParticleForce(node["type"], node["params"], curves)
    run = simcache.run_key(stream.run, force.identity(expressions))
    return ParticleStream(stream.emitter.with_force(force), run, stream.start_frame, stream.substeps,
                          stream.seed)


def solve_frame(stream, frame, cache, cancel=None):
    """The solved `simcache.State` at `frame`, from `cache` where possible."""
    return simcache.solve_to_frame(cache, stream.run, int(frame), stream.start_frame, stream.substeps,
                                   stream.seed, stream.emitter.initial_state, stream.emitter.step, cancel)


def instance_from_state(state, stream, frame):
    """A `scene3d.ParticleInstance` (read-only views of the solved arrays) for one frame."""
    from .scene3d import ParticleInstance

    def view(name):
        array = state.arrays[name].view()
        array.flags.writeable = False
        return array
    substeps = stream.substeps
    ages = state.arrays["age"].astype(np.float32) / np.float32(substeps)
    lifetimes = state.arrays["life"].astype(np.float32) / np.float32(substeps)
    for array in (ages, lifetimes):
        array.flags.writeable = False
    return ParticleInstance(positions=view("position"), sizes=view("size"), colors=view("color"),
                            velocities=view("velocity"), ages=ages, lifetimes=lifetimes,
                            ids=view("id"), stream=stream, frame=int(frame))


def placeholder_instance(stream, frame):
    """An empty `ParticleInstance` that only names the run, for a ParticleCache3D to solve."""
    from .scene3d import ParticleInstance
    empty = _empty_arrays()
    ages = np.zeros(0, np.float32)
    return ParticleInstance(positions=empty["position"], sizes=empty["size"], colors=empty["color"],
                            velocities=empty["velocity"], ages=ages, lifetimes=ages, ids=empty["id"],
                            stream=stream, frame=int(frame))
