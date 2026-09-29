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

FORCE_KINDS = ("ParticleGravity3D", "ParticleDrag3D", "ParticleWind3D", "ParticleTurbulence3D",
               "ParticleBounce3D", "ParticleCollide3D")
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


# --- collisions (step 2c) ----------------------------------------------------------------------

MAX_HITS_PER_SUBSTEP = 4     # bounces resolved inside one substep; a particle still hitting then holds still
SURFACE_EPS = 1e-4           # world units: where a bounced particle is placed above the surface it hit
REST_SPEED = 1e-3            # units per frame: a rebound slower than this is not a rebound, the particle rests


def collider_triangles(value):
    """World-space triangles (T,3,3) of a geometry or a scene, `Geometry.world_matrix` applied."""
    from .scene3d import Scene
    geometries = value.geometries if isinstance(value, Scene) else (() if value is None else (value,))
    parts = []
    for geometry in geometries:
        if not len(geometry.vertices) or not len(geometry.triangles):
            continue
        matrix = geometry.world_matrix().astype(np.float64)
        world = (matrix[:3, :3] @ geometry.vertices.astype(np.float64).T).T + matrix[:3, 3]
        parts.append(world[geometry.triangles])
    return np.concatenate(parts) if parts else np.zeros((0, 3, 3), np.float64)


def collider_objects(value):
    """Per-object (local, rest-pose triangles (T,3,3), world matrix (4,4)) of a geometry or a
    scene's geometries. Kept separate rather than baked into world space, because the animated
    collision sweep below needs each object's own transform at two different instants -- the mesh
    itself is assumed not to deform while it animates, only its transform, exactly the assumption
    `nodebased/fluid3d.py`'s `GeometryTrack.motion` makes for a moving collider."""
    from .scene3d import Scene
    geometries = value.geometries if isinstance(value, Scene) else (() if value is None else (value,))
    objects = []
    for geometry in geometries:
        if not len(geometry.vertices) or not len(geometry.triangles):
            continue
        local = geometry.vertices.astype(np.float64)[geometry.triangles]
        objects.append((local, geometry.world_matrix().astype(np.float64)))
    return objects


class GeometryTrack:
    """A geometry (or Scene) input, sampled per frame from `provider(frame) -> value`. A static
    track (`animated=False`) always samples `start_frame`, exactly the old "frozen at the start
    frame" behaviour. `.at(frame)` gives its world-space triangles (the frozen path);
    `.objects_at(frame)` gives each object's own rest-pose triangles and world matrix separately,
    for the animated, rigid-frame sweep (`ParticleCollider`, "Bounce and collisions (animated)").
    """

    def __init__(self, provider, animated, start_frame):
        self.provider = provider
        self.animated = bool(animated)
        self.start_frame = int(start_frame)
        self._cache = {}

    def _value(self, frame):
        key = int(frame) if self.animated else self.start_frame
        cached = self._cache.get(key)
        if cached is None:
            cached = self.provider(key)
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = cached
        return cached

    def prime(self, frame, value):
        """Seed the cache for `frame` with an already-evaluated result, so a caller that evaluated
        the geometry for another reason (its content digest, say) does not pay for it twice."""
        self._cache[int(frame)] = value

    def at(self, frame):
        return collider_triangles(self._value(frame))

    def objects_at(self, frame):
        return collider_objects(self._value(frame))


class ParticleCollider(ParticleForce):
    """A ParticleBounce3D node's contribution: knobs, the collision geometry (frozen, or tracked
    per frame when `animated`) and its BVH.

    It changes no velocity through `apply`; `collide` resolves the whole chain's colliders together
    after the emitter has moved the particles (docs/SIMULATION.md, "Bounce and collisions").

    When `animated` is off the collider is exactly the old frozen-at-the-start-frame behaviour
    (`first_hit`'s static branch), bit-identical. When it is on, each object is tracked as a rigid
    body: its rest-pose (local) triangles stay fixed and its world transform is sampled per frame
    and linearly interpolated per substep. A particle's own substep motion is tested *in that
    object's own local frame* -- transformed by the object's inverse transform at the start and end
    of the substep respectively -- which reduces "did a moving, rotating surface hit a particle,
    including one that never itself moves" to an ordinary static ray test, exact for a rigid
    translation and a good local approximation for a rotation refined by more substeps (the same
    kind of approximation `docs/SIMULATION.md`'s force integration already makes). The collider's
    own world velocity at the hit point is recovered from where that same local point sits under the
    start-of-substep and end-of-substep transforms.
    """
    collides = True

    def __init__(self, kind, params, curves=None, track=None, geometry_digest=None, animation_identity=None):
        super().__init__(kind, params, curves)
        self.geometry_digest = geometry_digest
        self.animation_identity = animation_identity
        self.animated = bool(int(params.get("animated", 0)))
        self.track = track if track is not None else GeometryTrack(lambda frame: None, False, 1)
        self._static = None            # (TriangleSet, Bvh, normals) for the frozen path
        self._object_bvh = {}          # object index -> (Bvh, triangle count), the animated path's refit cache

    def apply(self, arrays, velocity, frame, substep, substeps):
        return velocity

    def identity(self, expressions=None):
        base = {**super().identity(expressions), "collider": self.geometry_digest, "animated": self.animated}
        if self.animated:
            base["animated_geometry"] = self.animation_identity
        return base

    def _static_structures(self):
        if self._static is None:
            from . import raytrace
            triangles = self.track.at(self.track.start_frame)
            if not len(triangles):
                self._static = (None, None, None)
            else:
                v0, e1, e2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                cross = np.cross(e1, e2)
                normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-300)
                lo, hi = triangles.min(axis=1), triangles.max(axis=1)
                self._static = (raytrace.TriangleSet(v0, e1, e2, 1.0), raytrace.Bvh.build(lo, hi), normals)
        return self._static

    def _object_structures(self, index, local):
        """(TriangleSet, Bvh, normals) over one object's rest-pose triangles, refitting the Bvh in
        place when its triangle count matches the previous call and rebuilding only when it does
        not (the mesh's own vertex positions are constant in local space for a rigid transform, so
        a refit here is a no-op recompute of the same bounds; a genuinely deforming input, or one
        whose triangle count changes, still refits or rebuilds correctly)."""
        from . import raytrace
        if not len(local):
            self._object_bvh[index] = None
            return None
        v0, e1, e2 = local[:, 0], local[:, 1] - local[:, 0], local[:, 2] - local[:, 0]
        cross = np.cross(e1, e2)
        normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-300)
        lo, hi = local.min(axis=1), local.max(axis=1)
        cached = self._object_bvh.get(index)
        if cached is not None and cached[1] == len(local):
            bvh = cached[0].refit(lo, hi)
        else:
            bvh = raytrace.Bvh.build(lo, hi)
        self._object_bvh[index] = (bvh, len(local))
        return raytrace.TriangleSet(v0, e1, e2, 1.0), bvh, normals

    def _objects_at_fraction(self, fraction_frame):
        """Each object's (local triangles, world matrix) at the fractional frame `fraction_frame`,
        the matrix linearly interpolated between the two integer frames it falls between (matrices
        are only ever sampled at whole frames; see docs/SIMULATION.md, "Bounce and collisions
        (animated)"). Falls back to the earlier frame's matrix, unblended, across a frame where the
        object's own triangle count changed (a topology change is a momentary hold, not a
        interpolated jump, matching `GeometryTrack.objects_at`'s emitter-geometry counterpart)."""
        base = math.floor(fraction_frame)
        frac = fraction_frame - base
        objects_a = self.track.objects_at(int(base))
        if frac <= 0.0 or not objects_a:
            return objects_a
        objects_b = self.track.objects_at(int(base) + 1)
        if len(objects_b) != len(objects_a):
            return objects_a
        blended = []
        for (local_a, matrix_a), (local_b, matrix_b) in zip(objects_a, objects_b):
            if local_b.shape != local_a.shape:
                blended.append((local_a, matrix_a))
            else:
                # Mesh vertices can animate independently of the object transform (Alembic,
                # USD, deformed liquid surfaces and evaluated instances). Interpolate both parts
                # so the substep sees the actual deforming surface, not a frozen rest mesh.
                blended.append((local_a + frac * (local_b - local_a),
                                matrix_a + frac * (matrix_b - matrix_a)))
        return blended

    def first_hit(self, origins, displacements, frame, substep, substeps):
        """Fraction t in (0, 1] of each displacement at which it first crosses the collider, or inf,
        with the unit face normal at that hit and the collider's own world velocity there (zero when
        not animated)."""
        count = len(origins)
        zero = np.zeros((count, 3))
        if not self.animated:
            tri_set, bvh, normals = self._static_structures()
            if tri_set is None:
                return np.full(count, np.inf), zero, zero
            t, prim, _, _ = tri_set.closest_hit(bvh, origins, displacements, 0.0, 1.0)
            hit = prim >= 0
            return np.where(hit, t, np.inf), normals[np.maximum(prim, 0)], zero
        f0, f1 = frame + substep / substeps, frame + (substep + 1) / substeps
        objects0 = self._objects_at_fraction(f0)
        objects1 = self._objects_at_fraction(f1)
        objects_mid = self._objects_at_fraction((f0 + f1) * 0.5)
        best_t = np.full(count, np.inf)
        best_normal, best_velocity = zero.copy(), zero.copy()
        ends = origins + displacements
        for index, ((local0, m0), (local1, m1), (local, mm)) in enumerate(
                zip(objects0, objects1, objects_mid)):
            structures = self._object_structures(index, local)
            if structures is None:
                continue
            tri_set, bvh, normals = structures
            inv0, inv1 = np.linalg.inv(m0), np.linalg.inv(m1)
            local_start = (inv0[:3, :3] @ origins.T).T + inv0[:3, 3]
            local_end = (inv1[:3, :3] @ ends.T).T + inv1[:3, 3]
            # Transform the particle sweep into the interpolated object's local frame. Hits are
            # resolved on the midpoint deform (the same linear-in-time approximation used for
            # transforms), then the hit's barycentric coordinates recover the per-vertex surface
            # velocity. This captures a bending sheet whose average object transform is static.
            invm = np.linalg.inv(mm)
            # Keep each endpoint in the object's own frame at that endpoint. This is what makes
            # rigid rotation/translation sweeps work; the midpoint mesh supplies the tested shape.
            mid_start, mid_end = local_start.copy(), local_end.copy()
            if local0.shape == local1.shape and len(local0):
                # Work in a frame moving with the mean deform velocity so a stationary particle
                # is swept by a moving/bending sheet just as it is by a translated rigid paddle.
                # The per-hit barycentric velocity below restores the local vertex motion.
                deform = (local1 - local0).reshape(-1, 3).mean(axis=0)
                mid_start += 0.5 * deform
                mid_end -= 0.5 * deform
            t, prim, u, v = tri_set.closest_hit(bvh, mid_start, mid_end - mid_start, 0.0, 1.0)
            better = (prim >= 0) & (t < best_t)
            if not better.any():
                continue
            rows = np.flatnonzero(better)
            prim_rows = prim[rows]
            weights = np.stack((1.0 - u[rows] - v[rows], u[rows], v[rows]), axis=1)
            tri0, tri1 = local0[prim[rows]], local1[prim[rows]]
            local_hit0 = np.einsum("ni,nij->nj", weights, tri0)
            local_hit1 = np.einsum("ni,nij->nj", weights, tri1)
            world_hit0 = (m0[:3, :3] @ local_hit0.T).T + m0[:3, 3]
            world_hit1 = (m1[:3, :3] @ local_hit1.T).T + m1[:3, 3]
            normal_transform = np.linalg.inv(mm[:3, :3]).T
            normal = normals[prim_rows] @ normal_transform.T
            best_t[rows] = t[rows]
            best_normal[rows] = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            best_velocity[rows] = (world_hit1 - world_hit0) * float(substeps)
        return best_t, best_normal, best_velocity


def collide(colliders, frame, substep, substeps, ids, start, velocity, end, dt, sweep_velocity=None):
    """Resolve one substep's motion `start -> start + velocity * dt` against `colliders`.

    Swept, not sampled: every substep tests the whole segment a particle travels against the
    triangles (or, for an animated collider, the conservative swept prism `ParticleCollider` builds
    for the substep -- see docs/SIMULATION.md, "Bounce and collisions (animated)"), so no speed
    tunnels through a surface at any timestep. Inside a substep the nearest hit across all colliders
    is resolved first, the particle rebounds and spends the rest of the substep's time moving on (up
    to MAX_HITS_PER_SUBSTEP times; a particle still hitting then holds its position). The bounce and
    friction below act on the particle's velocity *relative to the collider's own velocity at the hit
    point* (zero for a static collider), so a moving surface throws a particle it hits: the rebound
    normal speed is `bounce` times the relative approach speed (a rebound slower than REST_SPEED
    becomes rest relative to the surface), and friction removes relative tangential speed by the same
    Coulomb rule as before. Returns (position, velocity, dead) with only the particles that hit
    something changed; the rest keep `end` and `velocity`.

    `sweep_velocity`, when given, drives the swept *displacement* test only (`start ->
    start + sweep_velocity * dt`); the ordinary `velocity` still drives the bounce response and is
    what an untouched particle keeps. The two differ only when `end` was pushed somewhere
    `start + velocity * dt` would not reach on its own -- `ParticleCollide3D`'s self-collision
    correction, chained before this call (docs/SIMULATION.md, "Particle-particle collisions") -- so
    the swept test still sees the whole substep's true displacement instead of silently missing a
    crossing that only the position correction produced. Omitted, it defaults to `velocity` and this
    function is bit-identical to before `sweep_velocity` existed.
    """
    count = len(ids)
    position, out_velocity = end, velocity
    dead = np.zeros(count, bool)
    if not count or not colliders:
        return position, out_velocity, dead
    params = [c.frame_params(frame) for c in colliders]
    masks = []
    for collider, p in zip(colliders, params):
        window = int(p["from_frame"]) <= frame <= int(p["to_frame"])
        masks.append(collider.selected(ids, p) if window and len(collider.track.at(frame))
                     else np.zeros(count, bool))
    candidates = np.flatnonzero(np.logical_or.reduce(masks))
    if not len(candidates):
        return position, out_velocity, dead
    bounce = np.array([float(p["bounce"]) for p in params])
    friction = np.array([float(p["friction"]) for p in params])
    kill = np.array([bool(int(p["kill_on_collision"])) for p in params])
    pos = start.astype(np.float64).copy()
    vel = velocity.astype(np.float64).copy()
    sweep_vel = vel if sweep_velocity is None else sweep_velocity.astype(np.float64).copy()
    left = np.ones(count)
    touched = np.zeros(count, bool)
    active = candidates
    for _ in range(MAX_HITS_PER_SUBSTEP):
        if not len(active):
            break
        origins = pos[active]
        disp = sweep_vel[active] * (dt * left[active])[:, None]
        best = np.full(len(active), np.inf)
        normal = np.zeros((len(active), 3))
        surface_vel = np.zeros((len(active), 3))
        owner = np.zeros(len(active), np.int64)
        for index, collider in enumerate(colliders):
            rows = np.flatnonzero(masks[index][active])
            if not len(rows):
                continue
            t, n, v = collider.first_hit(origins[rows], disp[rows], frame, substep, substeps)
            better = t < best[rows]
            chosen = rows[better]
            best[chosen], normal[chosen], surface_vel[chosen], owner[chosen] = t[better], n[better], v[better], index
        hit = np.isfinite(best)
        free = active[~hit]
        pos[free] += vel[free] * (dt * left[free])[:, None]          # nothing in the way: finish the substep
        left[free] = 0.0
        if not hit.any():
            active = active[:0]
            break
        rows = np.flatnonzero(hit)
        idx, t, who = active[rows], best[rows], owner[rows]
        d = disp[rows]
        n = normal[rows]
        n = np.where((np.einsum("ij,ij->i", n, d) > 0)[:, None], -n, n)     # face the incoming side
        point = origins[rows] + t[:, None] * d
        surface = surface_vel[rows]
        v_in = vel[idx]
        v_rel = v_in - surface                                              # velocity relative to the collider
        vn = np.einsum("ij,ij->i", v_rel, n)                                # negative: heading into the surface
        vt = v_rel - vn[:, None] * n
        vn_out = np.where(-vn * bounce[who] < REST_SPEED, 0.0, -vn * bounce[who])
        speed_t = np.linalg.norm(vt, axis=1)
        cut = friction[who] * (-vn + vn_out)
        scale = np.where(speed_t > 1e-12, np.maximum(0.0, 1.0 - cut / np.maximum(speed_t, 1e-12)), 0.0)
        vel[idx] = surface + vt * scale[:, None] + vn_out[:, None] * n
        if sweep_velocity is not None:
            sweep_vel[idx] = vel[idx]    # the rest of the substep follows the real post-bounce velocity
        removed = kill[who]
        dead[idx[removed]] = True
        pos[idx] = point + n * SURFACE_EPS
        pos[idx[removed]] = point[removed]
        left[idx] *= 1.0 - t
        touched[idx] = True
        active = idx[~removed & (left[idx] > 1e-9)]
    # Particles still hitting after MAX_HITS_PER_SUBSTEP hold their position (never pass through).
    if touched.any():
        position = end.copy()
        out_velocity = velocity.copy()
        position[touched] = pos[touched].astype(np.float32)
        out_velocity[touched] = vel[touched].astype(np.float32)
    return position, out_velocity, dead


# --- particle-particle collision (step 2d) ------------------------------------------------------

# Half of the 27-cell neighbourhood (itself plus one of every +/- pair of the other 26), so a
# uniform-grid broad phase visits every adjacent cell pair exactly once. Built from a fixed
# ordering, never a set or dict, so the stencil itself cannot introduce iteration-order drift.
_HALF_OFFSETS = ((0, 0, 0),) + tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0) and (dx, dy, dz) < (-dx, -dy, -dz))


def _grid_pairs(positions, cell_size):
    """Candidate index pairs `(a, b)`, `a < b`, whose uniform-grid cells (of `cell_size`) are the
    same or adjacent (Chebyshev distance 1), from sorted cell keys and the `_HALF_OFFSETS` stencil.

    This is the broad phase only (see `resolve_particle_collisions` for the distance test): two
    particles farther apart than `cell_size` in any axis are never returned, so every true contact
    at a radius sum up to `cell_size / 2` is found, and the search cost tracks the number of
    occupied cells and their neighbours, not the size of empty space between them. The result's
    order is a pure function of the sorted cell keys and the fixed stencil order, never a set or
    dict's iteration order, though callers still sort by particle id before summing anything
    floating-point over these pairs (see "Determinism" in docs/SIMULATION.md).
    """
    count = len(positions)
    empty = np.zeros(0, np.int64)
    if count < 2:
        return empty, empty
    cell = np.floor(positions / cell_size).astype(np.int64)
    cell = cell - cell.min(axis=0)
    dims = cell.max(axis=0) + 1
    flat = cell[:, 0] + dims[0] * (cell[:, 1] + dims[1] * cell[:, 2])
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_flat[1:] != sorted_flat[:-1])))
    unique_flat = sorted_flat[starts]
    counts = np.diff(np.concatenate((starts, [count])))
    pairs_a, pairs_b = [], []
    for dx, dy, dz in _HALF_OFFSETS:
        nx, ny, nz = cell[:, 0] + dx, cell[:, 1] + dy, cell[:, 2] + dz
        valid = (nx >= 0) & (nx < dims[0]) & (ny >= 0) & (ny < dims[1]) & (nz >= 0) & (nz < dims[2])
        neighbor = np.where(valid, nx + dims[0] * (ny + dims[1] * nz), -1)
        found_at = np.clip(np.searchsorted(unique_flat, neighbor), 0, len(unique_flat) - 1)
        found = valid & (unique_flat[found_at] == neighbor)
        rows = np.flatnonzero(found)
        if not len(rows):
            continue
        row_starts, row_counts = starts[found_at[rows]], counts[found_at[rows]]
        total = int(row_counts.sum())
        if not total:
            continue
        cumulative = np.cumsum(row_counts)
        rep_rows = np.repeat(rows, row_counts)
        within = np.arange(total) - np.repeat(cumulative - row_counts, row_counts)
        candidates = order[np.repeat(row_starts, row_counts) + within]
        keep = rep_rows < candidates if (dx, dy, dz) == (0, 0, 0) else np.ones(total, bool)
        pairs_a.append(rep_rows[keep])
        pairs_b.append(candidates[keep])
    if not pairs_a:
        return empty, empty
    return np.concatenate(pairs_a), np.concatenate(pairs_b)


class ParticleSelfCollider(ParticleForce):
    """A ParticleCollide3D node's contribution: particles collide with each other as spheres.

    Unlike a `ParticleForce` (a velocity update) or a `ParticleCollider` (particles against wired
    geometry), this changes no velocity through `apply` -- `resolve_particle_collisions` runs the
    whole substep's self-collision pass once, after the emitter has integrated position but *before*
    any geometry collision: `ParticleCollider`'s `collide()` runs last, given the self-collision
    result as an extra `sweep_velocity` so its swept geometry test still catches whatever the
    self-collision correction did (docs/SIMULATION.md, "Particle-particle collisions").
    """
    self_collides = True

    def apply(self, arrays, velocity, frame, substep, substeps):
        return velocity


def resolve_particle_collisions(force, frame, ids, position, velocity, size):
    """One substep's particle-particle contact pass for one `ParticleCollide3D` (docs/SIMULATION.md,
    "Particle-particle collisions").

    Each of `iterations` passes rebuilds the broad-phase candidate list from the current positions
    (`_grid_pairs`, a uniform grid of cell size `2 * max(radius)`), keeps only pairs that actually
    overlap, sorts them by `(id_a, id_b)` -- so every `np.add.at` scatter-add below accumulates a
    particle's several contacts in the same order every run, the determinism rule this file already
    applies to random draws, applied here to a floating-point sum instead -- then applies a
    position correction (half the overlap to each side) and a normal/tangential velocity response
    (`restitution`, Coulomb `friction`) averaged over however many contacts touched that particle
    this pass, so many simultaneous contacts do not overshoot. `sleep_threshold` zeroes the
    velocity of a touched, near-resting particle at the end, memory-free: it is re-derived from the
    current speed every substep, not a persisted flag, so it needs no new state array and stays
    exact under caching and restart. `probability`, `from_frame`, `to_frame` and `seed` select and
    window the participating particles exactly like every other force.
    """
    count = len(ids)
    if count < 2:
        return position, velocity
    p = force.frame_params(frame)
    if not (int(p["from_frame"]) <= frame <= int(p["to_frame"])):
        return position, velocity
    active = np.flatnonzero(force.selected(ids, p))
    if len(active) < 2:
        return position, velocity
    radius_from_size = bool(int(p["radius_from_size"]))
    radius = (size[active] * 0.5) if radius_from_size else np.full(len(active), float(p["collide_radius"]))
    radius = np.maximum(radius, 0.0)
    restitution, friction = float(p["restitution"]), float(p["friction"])
    sleep_threshold = max(0.0, float(p["sleep_threshold"]))
    iterations = max(1, int(p["iterations"]))
    # Large particle sets use the adapter's on-device uniform grid. The small-set CPU
    # reference remains the exact historical path, and is used when wgpu is unavailable.
    if len(active) >= 10000:
        from . import particlegpu
        if particlegpu.available():
            gpu_pos, gpu_vel = particlegpu.resolve(ids[active], position[active], velocity[active],
                radius, iterations=iterations, restitution=restitution, friction=friction,
                sleep_threshold=sleep_threshold)
            result_pos, result_vel = position.copy(), velocity.copy()
            result_pos[active], result_vel[active] = gpu_pos, gpu_vel
            return result_pos, result_vel
    pos = position[active].astype(np.float64).copy()
    vel = velocity[active].astype(np.float64).copy()
    sub_id = ids[active]
    cell_size = max(2.0 * float(radius.max()), 1e-6)
    touched = np.zeros(len(active), bool)
    for _ in range(iterations):
        pairs_a, pairs_b = _grid_pairs(pos, cell_size)
        if not len(pairs_a):
            break
        contact_order = np.lexsort((sub_id[pairs_b], sub_id[pairs_a]))
        pairs_a, pairs_b = pairs_a[contact_order], pairs_b[contact_order]
        delta = pos[pairs_a] - pos[pairs_b]
        dist = np.linalg.norm(delta, axis=1)
        overlap = radius[pairs_a] + radius[pairs_b] - dist
        hit = overlap > 0.0
        if not hit.any():
            break
        pairs_a, pairs_b = pairs_a[hit], pairs_b[hit]
        delta, dist, overlap = delta[hit], dist[hit], overlap[hit]
        normal = np.zeros((len(pairs_a), 3))
        safe = dist > 1e-9
        normal[safe] = delta[safe] / dist[safe, None]
        if (~safe).any():
            # Coincident centres: an id-derived angle breaks the tie the same way every run,
            # rather than dividing by ~0 (docs/SIMULATION.md, "Particle-particle collisions").
            angle = _hash_unit(0, 5, sub_id[pairs_a[~safe]], sub_id[pairs_b[~safe]]) * (2.0 * math.pi)
            normal[~safe] = np.stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)), axis=1)
        counts = np.zeros(len(pos))
        np.add.at(counts, pairs_a, 1.0)
        np.add.at(counts, pairs_b, 1.0)
        weight = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
        correction = 0.5 * overlap
        pos_delta = np.zeros_like(pos)
        np.add.at(pos_delta, pairs_a, correction[:, None] * normal)
        np.add.at(pos_delta, pairs_b, -correction[:, None] * normal)
        pos = pos + pos_delta * weight[:, None]
        v_rel = vel[pairs_a] - vel[pairs_b]
        vn = np.einsum("ij,ij->i", v_rel, normal)
        impulse_n = np.where(vn < 0.0, -(1.0 + restitution) * 0.5 * vn, 0.0)
        vt = v_rel - vn[:, None] * normal
        speed_t = np.linalg.norm(vt, axis=1)
        cut = friction * np.abs(impulse_n)
        scale = np.where(speed_t > 1e-12, np.maximum(0.0, 1.0 - cut / np.maximum(speed_t, 1e-12)), 1.0)
        contact_delta = impulse_n[:, None] * normal + 0.5 * vt * (scale - 1.0)[:, None]
        vel_delta = np.zeros_like(vel)
        np.add.at(vel_delta, pairs_a, contact_delta)
        np.add.at(vel_delta, pairs_b, -contact_delta)
        vel = vel + vel_delta * weight[:, None]
        touched[pairs_a] = True
        touched[pairs_b] = True
    if touched.any() and sleep_threshold > 0.0:
        speed = np.linalg.norm(vel, axis=1)
        vel[touched & (speed < sleep_threshold)] = 0.0
    position, velocity = position.copy(), velocity.copy()
    position[active] = pos.astype(np.float32)
    velocity[active] = vel.astype(np.float32)
    return position, velocity


class ParticleEmitter:
    """One deterministic run: the emitter knobs, the emission source and the timeline model.

    `params` are the node's stored (unresolved) knobs; `curves` its animation curves. The run
    constants (`start_frame`, `substeps`, `seed`, `max_particles`, `emit_from`, `emit_rate_unit`)
    are read from the stored values; every other knob is resolved per frame, so an animated rate
    or an animated emitter transform works and is part of the run's identity through `curves`.
    """

    def __init__(self, params, curves=None, source=None, fps=24.0, kind="ParticleEmitter3D", geo_provider=None):
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
        self.animated = bool(int(self.params.get("animated", 0)))
        self.geo_provider = geo_provider          # frame -> Geometry, only used when animated
        self.forces = ()          # ParticleForce chain, upstream first (see `with_force`)
        self._resolved = {}
        self._source_cache = {}

    def source_at(self, frame):
        """The emission source at `frame`: the frozen `self.source` unless `animated` and a "geo"
        input is wired, in which case the geometry is resampled per frame it is actually solved at
        (docs/SIMULATION.md, "The emitter model")."""
        if not self.animated or self.geo_provider is None:
            return self.source
        key = int(frame)
        cached = self._source_cache.get(key)
        if cached is None:
            cached = ParticleSource(self.geo_provider(key))
            if len(self._source_cache) > 8:
                self._source_cache.clear()
            self._source_cache[key] = cached
        return cached

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
            new = self._births(emit, int(meta["emitted"]), p, rng, (frame - self.start_frame, substep), frame)
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
        # Self-collision runs before the geometry sweep, not after: a particle-particle contact
        # correction can shove a particle across a wall or floor it had already legally bounced
        # off, and only the geometry sweep below tests the *whole* substep displacement (original
        # position to wherever forces and self-collision together put it) against the actual
        # boundary, so it is the one correction allowed to have the last word on staying inside a
        # collider (docs/SIMULATION.md, "Particle-particle collisions").
        self_colliders = [force for force in self.forces if getattr(force, "self_collides", False)]
        sweep_velocity = None
        if self_colliders and len(arrays["id"]) > 1:
            pre_position, velocity = arrays["position"], arrays["velocity"]
            for force in self_colliders:
                position, velocity = resolve_particle_collisions(
                    force, frame, arrays["id"], position, velocity, arrays["size"])
            arrays = {**arrays, "velocity": velocity}
            # The self-collision correction can move a particle somewhere its velocity alone would
            # not reach this substep; `sweep_velocity` lets the geometry sweep below still test the
            # *whole* true displacement (docs/SIMULATION.md, "Particle-particle collisions").
            sweep_velocity = ((position.astype(np.float64) - pre_position.astype(np.float64))
                              / float(dt)).astype(np.float32)
        colliders = [force for force in self.forces if getattr(force, "collides", False)]
        if colliders and len(arrays["id"]):
            position, velocity32, dead = collide(colliders, frame, substep, self.substeps, arrays["id"],
                                                 arrays["position"], arrays["velocity"], position, float(dt),
                                                 sweep_velocity=sweep_velocity)
            arrays = {**arrays, "velocity": velocity32}
            alive_mask = alive_mask & ~dead
        if alive_mask.all():
            result = {**arrays, "position": position, "age": age}
        else:
            result = {"position": position[alive_mask], "velocity": arrays["velocity"][alive_mask],
                      "age": age[alive_mask], "life": arrays["life"][alive_mask],
                      "size": arrays["size"][alive_mask], "color": arrays["color"][alive_mask],
                      "id": arrays["id"][alive_mask]}
        return _state(result, meta)

    # --- births -------------------------------------------------------------------------------

    def _births(self, count, first_id, p, rng, stream, frame):
        from .scene3d import _transform_from
        matrix = _transform_from(p).matrix().astype(np.float64)
        linear = matrix[:3, :3]
        draws = rng.random((count, 8))
        local, normals = self._locations(count, draws, stream, frame)
        position = local @ linear.T + matrix[:3, 3]
        inherit = float(p.get("inherit_velocity", 0.0))
        birth_velocity = None
        if inherit:
            # The emitter's own transform's rigid velocity at the birth point: the position this
            # same local point would have one frame later, minus where it is now. Zero for a static
            # transform; for a translating or rotating emitter it carries the point along with it,
            # docs/SIMULATION.md "The emitter model" (inherit_velocity).
            next_matrix = _transform_from(self.frame_params(int(frame) + 1)).matrix().astype(np.float64)
            next_position = local @ next_matrix[:3, :3].T + next_matrix[:3, 3]
            birth_velocity = (next_position - position) * inherit
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
        velocity = aimed * speed[:, None]
        if birth_velocity is not None:
            velocity = velocity + birth_velocity
        return {"position": position.astype(np.float32),
                "velocity": velocity.astype(np.float32),
                "age": np.zeros(count, np.int32), "life": life_ticks,
                "size": np.maximum(size, 0.0).astype(np.float32),
                "color": color.astype(np.float32),
                "id": np.arange(first_id, first_id + count, dtype=np.int64)}

    def _locations(self, count, draws, stream, frame):
        """Emission points in the emitter's local space, and face normals when they exist."""
        source, mode = self.source_at(frame), self.emit_from
        origin = np.zeros((count, 3))
        if mode == "point" or source.low is None:
            return origin, None
        if mode == "vertices" or (mode == "surface" and not source.has_surface):
            picks = np.minimum((draws[:, 0] * len(source.vertices)).astype(np.int64), len(source.vertices) - 1)
            return source.vertices[picks], None
        if mode == "surface" or (mode == "volume" and not source.has_surface):
            return self._surface(count, draws, source)
        return self._volume(count, draws, stream, source)

    def _surface(self, count, draws, source):
        target = draws[:, 0] * source.total_area
        tri = np.minimum(np.searchsorted(source.cumulative_area, target, side="right"),
                         len(source.cumulative_area) - 1)
        r1, r2 = np.sqrt(draws[:, 1]), draws[:, 2]
        a, b, c = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
        corners = source.triangles[tri]
        points = corners[:, 0] * a[:, None] + corners[:, 1] * b[:, None] + corners[:, 2] * c[:, None]
        return points, source.face_normals[tri]

    def _volume(self, count, draws, stream, source):
        """Uniform inside a closed mesh (ray parity, rejection from its bounding box).

        The candidate stream is its own generator seeded from the substep's indices, so the volume
        draw never disturbs the main draws. A mesh that is not closed, or too large to test, falls
        back to surface points after `_VOLUME_ATTEMPTS` empty batches.
        """
        if len(source.triangles) > _VOLUME_TRIANGLE_CAP:
            return self._surface(count, draws, source)
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


def _geometry_definition(doc, geo_key):
    """A JSON-safe snapshot of a wired geometry node's own definition (its stored params, curves and
    expressions), used as the run-identity term for a moving source or collider (see
    `ParticleEmitter3D`/`ParticleBounce3D` "animated" below). Nothing here depends on which frame is
    being viewed, so scrubbing an unedited animation keeps one run; any keyframe edit changes it and
    abandons the run, exactly like editing any other knob (docs/SIMULATION.md, "The run's identity").
    """
    if geo_key is None:
        return None
    node = doc["nodes"].get(geo_key)
    if node is None:
        return None
    return {"type": node["type"], "params": node["params"],
            "curves": doc.get("animation", {}).get("curves", {}).get(geo_key),
            "expressions": doc.get("expressions", {}).get(geo_key)}


def build_stream(evaluator, doc, key, node, cancel=None):
    """The `ParticleStream` for one ParticleEmitter3D node of `doc`.

    The emission geometry is sampled once, at the start frame, unless `animated` is on, in which
    case it is resampled at whichever frame the solver is actually advancing to (docs/SIMULATION.md,
    "The emitter model"). The run's identity is the digest of the start-frame geometry plus the
    stored knobs, the curves and expressions on them, the document frame rate when the rate is per
    second, and, only when animated, the wired geometry node's own definition (see
    `_geometry_definition`).
    """
    params = node["params"]
    start = int(params["start_frame"])
    animated = bool(int(params.get("animated", 0)))
    geometry, digest = None, None
    source_key = node["inputs"].get("geo")
    geo_provider = None
    if source_key is not None and params["emit_from"] != "point":
        geometry, digest = evaluator.evaluate_raster(doc, source_key, cancel, frame=start,
                                                     typed=True, return_digest=True)
        if animated:
            def geo_provider(frame, _key=source_key):
                return evaluator.evaluate_raster(doc, _key, cancel, frame=frame, typed=True)
    curves = doc.get("animation", {}).get("curves", {}).get(key)
    expressions = doc.get("expressions", {}).get(key)
    fps = float(doc.get("time", {}).get("fps", 24.0))
    emitter = ParticleEmitter(params, curves, ParticleSource(geometry), fps, geo_provider=geo_provider)
    identity = {"kind": node["type"], "params": params, "curves": curves, "expressions": expressions,
                "fps": fps if emitter.per_second else None, "format": 1}
    if animated:
        identity["animated_geometry"] = _geometry_definition(doc, source_key)
    return ParticleStream(emitter, simcache.run_key(digest, identity), start, emitter.substeps, emitter.seed)


def extend_stream(stream, doc, key, node, evaluator=None, cancel=None):
    """`stream` with the force node `key` chained on: same emission, a new run identity.

    The run is `run_key(old run, force identity)`, so changing any force knob, curve or expression
    (or adding, removing or reordering a force) abandons the old frames exactly like an emitter edit.
    A ParticleBounce3D also samples its `geometry` input once, at the start frame, and its digest is
    part of the identity, exactly like an emission mesh; when `animated` is on the collider is instead
    tracked per frame (docs/SIMULATION.md, "Bounce and collisions (animated)") and the geometry node's
    own definition is folded into the identity too, through `_geometry_definition`.
    """
    curves = doc.get("animation", {}).get("curves", {}).get(key)
    expressions = doc.get("expressions", {}).get(key)
    if node["type"] == "ParticleBounce3D":
        digest, track, animation_identity = None, None, None
        animated = bool(int(node["params"].get("animated", 0)))
        geo = node["inputs"].get("geometry")
        if geo is not None and evaluator is not None:
            geometry, digest = evaluator.evaluate_raster(doc, geo, cancel, frame=stream.start_frame,
                                                         typed=True, return_digest=True)

            def provider(frame, _geo=geo):
                return evaluator.evaluate_raster(doc, _geo, cancel, frame=frame, typed=True)
            track = GeometryTrack(provider, animated, stream.start_frame)
            track.prime(stream.start_frame, geometry)
            if animated:
                animation_identity = _geometry_definition(doc, geo)
        force = ParticleCollider(node["type"], node["params"], curves, track, digest, animation_identity)
    elif node["type"] == "ParticleCollide3D":
        force = ParticleSelfCollider(node["type"], node["params"], curves)
    else:
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
