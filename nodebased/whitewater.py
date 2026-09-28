"""Cached secondary liquid particles: foam, ballistic spray and buoyant bubbles.

The three emission potentials follow Ihmsen et al., *Unified spray, foam and air bubbles
for particle-based fluids* (2012, https://doi.org/10.1007/s00371-012-0697-9): trapped
air/vorticity, wave-crest curvature, and kinetic energy. This compact CPU post-pass uses the
liquid's signed-distance volume and particle velocities as its local estimates; it deliberately
leaves the FLIP solve untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FOAM, SPRAY, BUBBLE = 0, 1, 2
TYPE_NAMES = ("foam", "spray", "bubbles")


def _sample_phi(volume, points):
    """Nearest-cell SDF sample in volume object coordinates (the liquid currently has identity matrix)."""
    if volume is None or not len(points):
        return np.full(len(points), np.inf, np.float64)
    q = np.floor((points - np.asarray(volume.origin)) / volume.voxel_size).astype(np.int64)
    shape = np.asarray(volume.density.shape)
    inside = np.all((q >= 0) & (q < shape), axis=1)
    result = np.full(len(points), np.inf, np.float64)
    if np.any(inside):
        ijk = q[inside]
        result[inside] = volume.density[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return result


def _surface_curvature(volume, points):
    """Approximate signed SDF curvature, sampled at the liquid particles."""
    if volume is None or not len(points):
        return np.zeros(len(points), np.float64)
    phi = np.asarray(volume.density, np.float32)
    h = float(volume.voxel_size)
    gx, gy, gz = np.gradient(phi, h)
    grad2 = gx * gx + gy * gy + gz * gz
    lap = (np.gradient(gx, h, axis=0) + np.gradient(gy, h, axis=1)
           + np.gradient(gz, h, axis=2))
    curvature = lap / np.sqrt(np.maximum(grad2, 1e-8))
    q = np.floor((points - np.asarray(volume.origin)) / h).astype(np.int64)
    shape = np.asarray(phi.shape)
    inside = np.all((q >= 0) & (q < shape), axis=1)
    result = np.zeros(len(points), np.float64)
    if np.any(inside):
        ijk = q[inside]
        result[inside] = curvature[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return result


@dataclass(frozen=True)
class WhitewaterState:
    positions: np.ndarray
    velocities: np.ndarray
    sizes: np.ndarray
    ages: np.ndarray
    lifetimes: np.ndarray
    ids: np.ndarray
    kinds: np.ndarray
    next_id: int = 0


def empty_state():
    return WhitewaterState(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                           np.zeros(0, np.float32), np.zeros(0, np.float32),
                           np.zeros(0, np.float32), np.zeros(0, np.int64),
                           np.zeros(0, np.uint8), 0)


class FluidWhitewater3D:
    """Pure one-frame update. All random choices derive from seed, frame and stable liquid ids."""
    def __init__(self, params=None, seed=0, fps=24.0, colliders=()):
        self.p = dict(whitewater_defaults())
        self.p.update(params or {})
        self.seed, self.fps, self.colliders = int(seed), float(fps), tuple(colliders)

    def initial_state(self, seed=0):
        return empty_state()

    def checkpoint(self, state):
        return state

    def restore(self, state):
        return state

    def step(self, state, liquid, frame=0, substep=0, seed=0):
        if not len(liquid) or liquid.velocities is None:
            return empty_state()
        p = self.p
        dt = 1.0 / self.fps
        positions = np.asarray(state.positions, np.float64).copy()
        start_positions = positions.copy()
        velocity = np.asarray(state.velocities, np.float64).copy()
        sizes = np.asarray(state.sizes, np.float32).copy()
        ages = np.asarray(state.ages, np.float32) + dt
        life = np.asarray(state.lifetimes, np.float32)
        ids = np.asarray(state.ids, np.int64).copy()
        kinds = np.asarray(state.kinds, np.uint8).copy()

        # Existing whitewater: airborne spray is ballistic with drag; foam follows local
        # liquid velocity and loses density through its finite lifespan; bubbles rise in water.
        phi = _sample_phi(liquid.surface, positions)
        for kind, mask in ((SPRAY, kinds == SPRAY), (FOAM, kinds == FOAM), (BUBBLE, kinds == BUBBLE)):
            if not np.any(mask):
                continue
            if kind == SPRAY:
                velocity[mask, 1] -= float(p["gravity"]) * dt
                velocity[mask] *= max(0.0, 1.0 - float(p["spray_drag"]) * dt)
            elif kind == BUBBLE:
                velocity[mask, 1] += float(p["bubble_buoyancy"]) * dt
                velocity[mask] *= max(0.0, 1.0 - float(p["bubble_drag"]) * dt)
                risen = mask & (phi >= -float(p["surface_band"]))
                kinds[risen] = FOAM
                ages[risen] = 0.0
                life[risen] = float(p["foam_lifespan"])
            else:
                # Relax foam toward the liquid's local velocity; in the liquid particle
                # representation nearest-particle velocity is the consistent surface guide.
                nearest = _nearest_indices(positions[mask], liquid.positions)
                velocity[mask] += (np.asarray(liquid.velocities)[nearest] - velocity[mask]) * min(1.0, dt * 8.0)
            positions[mask] += velocity[mask] * dt

        if self.colliders and len(positions):
            positions, velocity = self._collide(start_positions, positions, velocity, int(frame))

        alive = (ages < life) & np.isfinite(positions).all(axis=1)
        positions, velocity, sizes, ages, life, ids, kinds = (a[alive] for a in
                                                               (positions, velocity, sizes, ages, life, ids, kinds))

        # Surface kinetic-energy / crest potential; trapped-air potential is the interior
        # vorticity proxy from neighbour velocity differences. Thresholds and rates are per type.
        liquid_pos = np.asarray(liquid.positions, np.float64)
        liquid_vel = np.asarray(liquid.velocities, np.float64)
        surf_phi = _sample_phi(liquid.surface, liquid_pos)
        surface = np.abs(surf_phi) <= float(p["surface_band"])
        speed = np.linalg.norm(liquid_vel, axis=1)
        energy = speed * speed
        curvature = np.abs(_surface_curvature(liquid.surface, liquid_pos))
        # Neighbour velocity contrast approximates trapped-air/vorticity potential without
        # introducing a grid dependency or changing the liquid solver.
        nearest = _nearest_indices(liquid_pos, liquid_pos, exclude_self=True)
        swirl = np.linalg.norm(liquid_vel - liquid_vel[nearest], axis=1)
        potential = {
            FOAM: np.where(surface, energy * (0.25 + curvature * float(getattr(liquid.surface, "voxel_size", 1.0))), 0.0),
            SPRAY: np.where(surface, energy, 0.0),
            BUBBLE: np.where(surf_phi < -float(p["surface_band"]), swirl * (0.1 + energy), 0.0),
        }
        total_cap = max(0, int(p["max_particles"]))
        per_type_cap = 0 if total_cap == 0 else max(1, (total_cap + 2) // 3)
        emitted = []
        next_id = int(state.next_id)
        for kind in (FOAM, SPRAY, BUBBLE):
            rate = max(0.0, float(p[f"{TYPE_NAMES[kind]}_emission"]))
            threshold = max(1e-9, float(p[f"{TYPE_NAMES[kind]}_threshold"]))
            score = potential[kind] * rate
            candidates = np.flatnonzero(score >= threshold)
            if not len(candidates) or per_type_cap <= 0:
                continue
            # Deterministic per-id thinning approximates the emission rate and avoids frame-order RNG.
            ids_in = np.asarray(liquid.ids if liquid.ids is not None else np.arange(len(liquid_pos)), np.int64)
            rank = _stable_rank(ids_in[candidates], self.seed ^ int(frame) ^ (kind * 0x9E3779B1))
            candidates = candidates[np.argsort(rank, kind="stable")[:per_type_cap]]
            count = min(len(candidates), per_type_cap, max(1, int(round(len(candidates) * min(rate, 1.0)))))
            candidates = candidates[:count]
            pos = liquid_pos[candidates].copy()
            vel = liquid_vel[candidates].copy()
            if kind == SPRAY:
                vel[:, 1] += float(p["spray_lift"])
                pos[:, 1] += float(p["surface_offset"])
            elif kind == BUBBLE:
                pos[:, 1] -= float(p["surface_offset"])
            n = len(candidates)
            emitted.append((pos, vel, np.full(n, float(p[f"{TYPE_NAMES[kind]}_size"])),
                            np.zeros(n), np.full(n, float(p["foam_lifespan"] if kind == FOAM else p["particle_lifespan"])),
                            np.arange(next_id, next_id + n, dtype=np.int64), np.full(n, kind, np.uint8)))
            next_id += n
        if emitted:
            combined = tuple(np.concatenate([part[i] for part in emitted], axis=0) for i in range(7))
            positions, velocity, sizes, ages, life, ids, kinds = (np.concatenate((old, new), axis=0)
                for old, new in zip((positions, velocity, sizes, ages, life, ids, kinds), combined))
        cap = total_cap
        if len(ids) > cap:
            order = np.argsort(ids, kind="stable")[-cap:] if cap else np.zeros(0, int)
            positions, velocity, sizes, ages, life, ids, kinds = (a[order] for a in
                                                                   (positions, velocity, sizes, ages, life, ids, kinds))
        return WhitewaterState(*(np.asarray(a) for a in (positions, velocity, sizes, ages, life, ids, kinds)), next_id)

    def _collide(self, start, end, velocity, frame):
        """Sweep whitewater against the same per-frame collider tracks that constrain FLIP.

        Animated tracks are sampled at the current frame and their measured per-triangle
        motion supplies the surface velocity, so a moving collider can impart momentum.
        """
        from . import raytrace
        start, end, velocity = start.copy(), end.copy(), velocity.copy()
        for collider in self.colliders:
            triangles = np.asarray(collider.track.at(frame), np.float64)
            if not len(triangles):
                continue
            v0 = triangles[:, 0]
            e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
            cross = np.cross(e1, e2)
            normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
            bvh = raytrace.Bvh.build(triangles.min(axis=1), triangles.max(axis=1))
            tri_set = raytrace.TriangleSet(v0, e1, e2, 1.0)
            displacement = end - start
            t, prim, _, _ = tri_set.closest_hit(bvh, start, displacement, 0.0, 1.0)
            hit = prim >= 0
            if not np.any(hit):
                continue
            rows = np.flatnonzero(hit)
            normal = normals[prim[rows]].copy()
            collider_velocity = np.zeros_like(normal)
            if collider.animated:
                motion = np.asarray(collider.track.motion(frame), np.float64)
                if len(motion) == len(triangles):
                    collider_velocity = motion[prim[rows]] * self.fps
            relative = velocity[rows] - collider_velocity
            flip = np.sum(relative * normal, axis=1) > 0.0
            normal[flip] *= -1.0
            vn = np.sum(relative * normal, axis=1)
            incoming = vn < 0.0
            velocity[rows[incoming]] -= (1.15 * vn[incoming, None]) * normal[incoming]
            velocity[rows[incoming]] += collider_velocity[incoming]
            impact = start[rows] + t[rows, None] * displacement[rows]
            remainder = end[rows] - impact
            into = np.minimum(np.sum(remainder * normal, axis=1), 0.0)
            end[rows] = impact + remainder - into[:, None] * normal + normal * 1e-4
        return end, velocity


def _nearest_indices(query, points, exclude_self=False):
    if not len(points):
        return np.zeros(len(query), np.int64)
    # A spatial hash keeps this linear for production liquid counts. Search adjacent bins;
    # the bin width tracks the mean particle spacing and the exact Euclidean nearest point
    # wins among candidates.
    span = np.ptp(points, axis=0)
    width = max(float(np.max(span)) / max(1.0, len(points) ** (1.0 / 3.0)), 1e-5)
    bins = np.floor(points / width).astype(np.int64)
    table = {}
    for i, key in enumerate(map(tuple, bins)):
        table.setdefault(key, []).append(i)
    result = np.empty(len(query), np.int64)
    qbins = np.floor(query / width).astype(np.int64)
    for row, (q, base) in enumerate(zip(query, qbins)):
        candidates = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    candidates.extend(table.get((base[0] + x, base[1] + y, base[2] + z), ()))
        if not candidates:
            # Sparse gaps are uncommon; a bounded fallback searches at most 4096 sampled
            # points rather than turning one empty bin into a full quadratic scan.
            candidates = np.linspace(0, len(points) - 1, min(len(points), 4096), dtype=np.int64).tolist()
        c = np.asarray(candidates, np.int64)
        if exclude_self and len(points) > 1 and row < len(points):
            c = c[c != row]
            if not len(c):
                c = np.arange(min(len(points), 4096), dtype=np.int64)
                c = c[c != row]
        d2 = np.sum((points[c] - q) ** 2, axis=1)
        result[row] = c[int(np.argmin(d2))]
    return result


def _stable_rank(ids, salt):
    x = np.asarray(ids, np.uint64) ^ np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
    x ^= x >> 30; x *= np.uint64(0xbf58476d1ce4e5b9); x ^= x >> 27
    x *= np.uint64(0x94d049bb133111eb); x ^= x >> 31
    return x


def whitewater_defaults():
    return {"foam_emission": 0.5, "spray_emission": 0.5, "bubbles_emission": 0.5,
            "foam_threshold": 0.4, "spray_threshold": 1.0, "bubbles_threshold": 0.05,
            "foam_lifespan": 4.0, "particle_lifespan": 2.0, "max_particles": 60_000,
            "foam_size": 0.04, "spray_size": 0.025, "bubbles_size": 0.025,
            "surface_band": 0.08, "surface_offset": 0.02, "spray_lift": 0.5,
            "gravity": 9.8, "spray_drag": 0.15, "bubble_buoyancy": 1.5, "bubble_drag": 2.0,
            "seed": 0, "cache_memory_mb": 256, "cache_disk_mb": 2048}


def instance_from_state(state, liquid, frame):
    from .scene3d import ParticleInstance
    colors = np.ones((len(state.ids), 4), np.float32)
    colors[:, :3] = 1.0
    arrays = [state.positions.astype(np.float32), state.sizes.astype(np.float32), colors,
              state.velocities.astype(np.float32), state.ages.astype(np.float32),
              state.lifetimes.astype(np.float32), state.ids.astype(np.int64), state.kinds.astype(np.uint8)]
    for array in arrays:
        array.flags.writeable = False
    # The whitewater node owns its simulation cache. Do not expose the upstream LiquidStream
    # as if this output were raw FLIP; a downstream ParticleCache3D would otherwise replace it.
    return ParticleInstance(positions=arrays[0], sizes=arrays[1], colors=arrays[2], velocities=arrays[3],
                            ages=arrays[4], lifetimes=arrays[5], ids=arrays[6], stream=None,
                            frame=int(frame), render_as="foam", whitewater_type=arrays[7])
