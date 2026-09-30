"""Deterministic CPU reference for small rigid-body scenes (world units, seconds)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import numpy as np


_BOX_VERTICES = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
_BOX_TRIANGLES = np.array(((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
                           (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                           (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)), dtype=np.int32)


def _poly_axes(vertices, triangles):
    points = np.asarray(vertices, dtype=np.float64)
    faces, edges = [], []
    for tri in np.asarray(triangles, dtype=np.int64):
        a, b, c = points[tri]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length > 1e-10:
            faces.append(normal / length)
        for p, q in ((a, b), (b, c), (c, a)):
            edge = q - p
            length = np.linalg.norm(edge)
            if length > 1e-10:
                edge /= length
                if not any(abs(float(np.dot(edge, old))) > 0.9999 for old in edges):
                    edges.append(edge)
    def unique(items):
        out, seen = [], set()
        for axis in items:
            axis = np.asarray(axis, dtype=np.float64)
            nz = np.flatnonzero(np.abs(axis) > 1e-8)
            if len(nz) and axis[nz[0]] < 0:
                axis = -axis
            key = tuple(np.round(axis, 5))
            if key not in seen:
                seen.add(key); out.append(axis)
        return out
    return unique(faces), unique(edges)


def convex_hull_triangles(vertices, triangle_hint=None, max_vertices=128):
    """Build supporting triangles for the convex hull of a mesh's vertices, without a native dependency."""
    points = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
    if len(points) > max_vertices and triangle_hint is not None:
        hint = np.asarray(triangle_hint, dtype=np.int64)
        scale = max(float(np.ptp(points, axis=0).max()), 1.0)
        tolerance = scale * 1e-7
        is_convex = True
        for tri in hint:
            normal = np.cross(points[tri[1]] - points[tri[0]], points[tri[2]] - points[tri[0]])
            if np.linalg.norm(normal) <= tolerance:
                continue
            distances = (points - points[tri[0]]) @ normal
            if not (np.all(distances <= tolerance) or np.all(distances >= -tolerance)):
                is_convex = False
                break
        if is_convex:
            return points, hint.astype(np.int32)
    if len(points) > max_vertices:
        raise ValueError(f"RigidBody3D hull of a non-convex mesh is limited to {max_vertices} unique vertices")
    if len(points) < 4:
        raise ValueError("RigidBody3D convex hull needs at least four non-coplanar vertices")
    keep = np.unique(np.round(points, 9), axis=0, return_index=True)[1]
    points = points[np.sort(keep)]
    if len(points) > max_vertices:
        raise ValueError(f"RigidBody3D convex hull is limited to {max_vertices} unique vertices")
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-9) < 3:
        raise ValueError("RigidBody3D convex hull points are coplanar")
    faces, planes = [], set()
    tolerance = max(float(np.ptp(points, axis=0).max()), 1.0) * 1e-8
    for i, j, k in combinations(range(len(points)), 3):
        normal = np.cross(points[j] - points[i], points[k] - points[i])
        length = np.linalg.norm(normal)
        if length <= tolerance:
            continue
        normal /= length
        signed = (points - points[i]) @ normal
        if np.all(signed <= tolerance):
            pass
        elif np.all(signed >= -tolerance):
            normal = -normal
        else:
            continue
        offset = float(np.dot(normal, points[i]))
        key = tuple(np.round(np.r_[normal, offset], 7))
        if key in planes:
            continue
        planes.add(key)
        coplanar = np.flatnonzero(np.abs(points @ normal - offset) <= tolerance)
        center = points[coplanar].mean(axis=0)
        u = points[coplanar[1]] - center
        u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(normal, u)
        angles = np.arctan2((points[coplanar] - center) @ v, (points[coplanar] - center) @ u)
        polygon = coplanar[np.argsort(angles)]
        for t in range(1, len(polygon) - 1):
            a, b, c = polygon[0], polygon[t], polygon[t + 1]
            if np.dot(np.cross(points[b] - points[a], points[c] - points[a]), normal) < 0:
                b, c = c, b
            faces.append((a, b, c))
    if not faces:
        raise ValueError("RigidBody3D convex hull points are coplanar")
    faces = np.asarray(faces, dtype=np.int32)
    used = np.unique(faces)
    remap = np.full(len(points), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return points[used], remap[faces]


def _sat_contact(av, at, bv, bt):
    """Exact SAT contact for two convex triangle meshes, returning A-to-B normal, depth and point."""
    af, ae = _poly_axes(av, at)
    bf, be = _poly_axes(bv, bt)
    axes = af + bf
    seen_axes = {tuple(np.round(axis, 5)) for axis in axes}
    for ea in ae:
        for eb in be:
            axis = np.cross(ea, eb)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                axis /= norm
                key = tuple(np.round(axis if axis[np.flatnonzero(np.abs(axis) > 1e-8)[0]] >= 0 else -axis, 5))
                if key not in seen_axes:
                    seen_axes.add(key); axes.append(axis)
    if not axes:
        return None
    best_depth, best_axis = float("inf"), None
    for axis in axes:
        pa, pb = av @ axis, bv @ axis
        overlap = min(float(pa.max() - pb.min()), float(pb.max() - pa.min()))
        if overlap <= 0:
            return None
        if overlap < best_depth:
            best_depth = overlap
            best_axis = axis if float(np.dot(bv.mean(0) - av.mean(0), axis)) >= 0 else -axis
    sa = av[av @ best_axis >= (av @ best_axis).max() - 1e-7].mean(0)
    sb = bv[bv @ best_axis <= (bv @ best_axis).min() + 1e-7].mean(0)
    return best_axis, best_depth, (sa + sb) * 0.5


@dataclass
class RigidBody3D:
    shape: str = "box"                 # box, sphere, convex, compound
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    size: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    density: float = 1000.0
    mass: float = 0.0                   # 0 means derive from density and volume
    friction: float = 0.5
    restitution: float = 0.0
    dynamic: bool = True
    children: tuple["RigidBody3D", ...] = ()
    geometry: tuple = ()                # optional visual mesh or child meshes supplied by the graph
    initial_position: np.ndarray | None = None
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    initial_rotation: np.ndarray | None = None
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    collision_parts: tuple = ()           # optional (local vertices, triangle indices) convex pieces
    sleeping: bool = False
    sleep_time: float = 0.0

    def __post_init__(self):
        self.shape = str(self.shape).lower()
        if self.shape not in {"box", "sphere", "convex", "compound"}:
            raise ValueError("shape must be box, sphere, convex or compound")
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3).copy()
        if self.initial_position is None:
            self.initial_position = self.position.copy()
        else:
            self.initial_position = np.asarray(self.initial_position, dtype=np.float64).reshape(3).copy()
        self.size = np.maximum(np.asarray(self.size, dtype=np.float64).reshape(3), 1e-6)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3).copy()
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3).copy()
        self.initial_rotation = self.rotation.copy() if self.initial_rotation is None else \
            np.asarray(self.initial_rotation, dtype=np.float64).reshape(3).copy()
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64).reshape(3).copy()
        self.torque = np.asarray(self.torque, dtype=np.float64).reshape(3).copy()
        if self.density <= 0 or self.mass < 0:
            raise ValueError("density must be positive and mass cannot be negative")
        if not 0 <= self.friction <= 100 or not 0 <= self.restitution <= 1:
            raise ValueError("friction must be nonnegative and restitution in [0, 1]")

    @property
    def volume(self):
        if self.collision_parts:
            total = 0.0
            for vertices, triangles in self.collision_parts:
                v = np.asarray(vertices, dtype=np.float64)
                t = np.asarray(triangles, dtype=np.int64)
                total += abs(float(np.einsum("ij,ij->i", v[t[:, 0]],
                                             np.cross(v[t[:, 1]], v[t[:, 2]])).sum())) / 6.0
            if total > 1e-12:
                return total
        if self.shape == "sphere":
            return float(4 * np.pi * np.prod(self.size) / 24.0)  # size stores diameters
        if self.shape == "compound":
            return sum(child.volume for child in self.children) if self.children else float(np.prod(self.size))
        return float(np.prod(self.size))

    @property
    def effective_mass(self):
        return float(self.mass or self.density * self.volume) if self.dynamic else float("inf")

    @property
    def half_extent(self):
        if self.shape == "sphere":
            return np.repeat(float(np.max(self.size)) / 2, 3)
        return self.size / 2

    def rotation_matrix(self):
        x, y, z = np.radians(self.rotation)
        cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
        return np.array(((cy*cz, -cy*sz, sy),
                         (sx*sy*cz+cx*sz, -sx*sy*sz+cx*cz, -sx*cy),
                         (-cx*sy*cz+sx*sz, cx*sy*sz+sx*cz, cx*cy)), dtype=np.float64)

    @property
    def inertia(self):
        m = self.effective_mass
        x, y, z = self.size
        if self.shape == "sphere":
            r = max(x, y, z) / 2
            value = 0.4 * m * r * r
            return np.repeat(value, 3)
        return m / 12.0 * np.array((y*y + z*z, x*x + z*z, x*x + y*y))


class RigidSolver3D:
    """Stable sequential-impulse solver with deterministic pair order and cached frame states.

    Boxes use oriented-box SAT, and connected convex meshes use SAT on their triangle hulls.
    Angular velocity is integrated in world space with a diagonal box/sphere inertia approximation.
    """
    def __init__(self, bodies=(), gravity=(0.0, -9.81, 0.0), fps=24.0, substeps=4,
                 floor_y=0.0, iterations=8, sleep_threshold=0.025, sleep_frames=0.5,
                 liquid_surface_y=None, liquid_density=1000.0):
        self.bodies = list(bodies)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.fps, self.substeps = float(fps), int(substeps)
        self.floor_y = floor_y if floor_y is None else float(floor_y)
        self.iterations = int(iterations)
        self.sleep_threshold, self.sleep_frames = float(sleep_threshold), float(sleep_frames)
        self.liquid_surface_y = None if liquid_surface_y is None else float(liquid_surface_y)
        self.liquid_density = float(liquid_density)
        if self.fps <= 0 or self.substeps < 1 or self.iterations < 1:
            raise ValueError("fps, substeps and iterations must be positive")
        self._frames = {}

    def _floor(self, body):
        if not body.dynamic or self.floor_y is None:
            return
        world = np.concatenate([vertices for vertices, _ in self._world_parts(body)], axis=0)
        low = float(world[:, 1].min())
        if low < self.floor_y:
            bottom = world[world[:, 1] <= low + max(1e-7, (world[:, 1].max() - low) * 1e-7)]
            contact = bottom.mean(axis=0)
            ground = RigidBody3D(position=(0, self.floor_y, 0), dynamic=False,
                                 friction=body.friction, restitution=body.restitution)
            self._resolve(ground, body, np.array((0.0, 1.0, 0.0)), self.floor_y - low, contact)
            if abs(body.velocity[1]) < self.sleep_threshold:
                body.velocity[1] = 0.0

    def _pair(self, a, b):
        if not a.dynamic and not b.dynamic:
            return
        if a.shape == b.shape == "sphere":
            delta = b.position - a.position
            distance = float(np.linalg.norm(delta))
            radius_a, radius_b = float(np.max(a.size)) / 2, float(np.max(b.size)) / 2
            if distance >= radius_a + radius_b or distance < 1e-12:
                return
            normal = delta / distance
            point = a.position + normal * (radius_a - (radius_a + radius_b - distance) / 2)
            contacts = [(normal, radius_a + radius_b - distance, point)]
        elif a.shape == "sphere" and b.shape != "sphere" and not b.collision_parts:
            rot = b.rotation_matrix()
            local = rot.T @ (a.position - b.position)
            closest = np.clip(local, -b.half_extent, b.half_extent)
            point = b.position + rot @ closest
            delta = point - a.position
            distance = float(np.linalg.norm(delta))
            radius = float(np.max(a.size)) / 2
            if distance >= radius or distance < 1e-12:
                return
            contacts = [(delta / distance, radius - distance, point)]
        elif b.shape == "sphere" and a.shape != "sphere" and not a.collision_parts:
            rot = a.rotation_matrix()
            local = rot.T @ (b.position - a.position)
            closest = np.clip(local, -a.half_extent, a.half_extent)
            point = a.position + rot @ closest
            delta = b.position - point
            distance = float(np.linalg.norm(delta))
            radius = float(np.max(b.size)) / 2
            if distance >= radius or distance < 1e-12:
                return
            contacts = [(delta / distance, radius - distance, point)]
        elif (not a.collision_parts and not b.collision_parts and
              np.all(np.abs(a.rotation) < 1e-10) and np.all(np.abs(b.rotation) < 1e-10)):
            delta = b.position - a.position
            overlap = a.half_extent + b.half_extent - np.abs(delta)
            if np.any(overlap <= 0):
                return
            axis = int(np.argmin(overlap))
            normal = np.zeros(3); normal[axis] = 1.0 if delta[axis] >= 0 else -1.0
            point = (a.position + b.position) * 0.5
            low_a, high_a = a.position - a.half_extent, a.position + a.half_extent
            low_b, high_b = b.position - b.half_extent, b.position + b.half_extent
            for other in range(3):
                if other != axis:
                    point[other] = (max(low_a[other], low_b[other]) + min(high_a[other], high_b[other])) * 0.5
            point[axis] = a.position[axis] + normal[axis] * a.half_extent[axis]
            contacts = [(normal, overlap[axis], point)]
        else:
            contacts = []
            for av, at in self._world_parts(a):
                for bv, bt in self._world_parts(b):
                    contact = _sat_contact(av, at, bv, bt)
                    if contact is not None:
                        contacts.append(contact)
        for normal, depth, point in contacts:
            self._resolve(a, b, normal, depth, point)

    def _world_parts(self, body):
        rotation = body.rotation_matrix()
        parts = body.collision_parts or ((
            _BOX_VERTICES * body.half_extent,
            _BOX_TRIANGLES),)
        return [((rotation @ np.asarray(vertices, dtype=np.float64).T).T + body.position,
                 np.asarray(triangles, dtype=np.int32)) for vertices, triangles in parts]

    @staticmethod
    def _inverse_inertia_world(body):
        if not body.dynamic:
            return np.zeros((3, 3), dtype=np.float64)
        rotation = body.rotation_matrix()
        return rotation @ np.diag(1.0 / np.maximum(body.inertia, 1e-12)) @ rotation.T

    def _resolve(self, a, b, normal, depth, point):
        inv_a = 1.0 / a.effective_mass if a.dynamic else 0.0
        inv_b = 1.0 / b.effective_mass if b.dynamic else 0.0
        if inv_a + inv_b <= 0:
            return
        ra, rb = point - a.position, point - b.position
        ia, ib = self._inverse_inertia_world(a), self._inverse_inertia_world(b)
        correction = max(0.0, depth - 1e-8) * 0.8 / (inv_a + inv_b)
        a.position -= normal * correction * inv_a
        b.position += normal * correction * inv_b
        relative_velocity = (b.velocity + np.cross(b.angular_velocity, rb) -
                             a.velocity - np.cross(a.angular_velocity, ra))
        vn = float(np.dot(relative_velocity, normal))
        if vn >= 0:
            return
        ran = np.cross(ra, normal)
        rbn = np.cross(rb, normal)
        denom = inv_a + inv_b + float(np.dot(normal, np.cross(ia @ ran, ra) + np.cross(ib @ rbn, rb)))
        impulse_mag = -(1 + min(a.restitution, b.restitution)) * vn / max(denom, 1e-12)
        impulse = normal * impulse_mag
        if a.dynamic:
            a.velocity -= impulse * inv_a
            a.angular_velocity -= ia @ np.cross(ra, impulse)
        if b.dynamic:
            b.velocity += impulse * inv_b
            b.angular_velocity += ib @ np.cross(rb, impulse)
        tangent = relative_velocity - vn * normal
        length = float(np.linalg.norm(tangent))
        if length > 1e-10:
            tangent /= length
            jt = -float(np.dot(relative_velocity, tangent)) / max(inv_a + inv_b, 1e-12)
            jt = float(np.clip(jt, -min(a.friction, b.friction) * impulse_mag,
                               min(a.friction, b.friction) * impulse_mag))
            friction_impulse = tangent * jt
            if a.dynamic:
                a.velocity -= friction_impulse * inv_a
                a.angular_velocity -= ia @ np.cross(ra, friction_impulse)
            if b.dynamic:
                b.velocity += friction_impulse * inv_b
                b.angular_velocity += ib @ np.cross(rb, friction_impulse)

    def _step(self, dt):
        for body in self.bodies:
            wet = (self.liquid_surface_y is not None and body.dynamic and
                   body.position[1] - body.half_extent[1] < self.liquid_surface_y)
            if wet or (body.dynamic and np.linalg.norm(body.torque) > 1e-12):
                body.sleeping = False
            if body.dynamic and not body.sleeping:
                acceleration = self.gravity.copy()
                if self.liquid_surface_y is not None and body.shape != "compound":
                    half = body.half_extent[1]
                    submerged = np.clip((self.liquid_surface_y - (body.position[1] - half)) / (2 * half), 0.0, 1.0)
                    density_ratio = self.liquid_density / (body.effective_mass / max(body.volume, 1e-12))
                    acceleration[1] += -self.gravity[1] * density_ratio * submerged
                    body.velocity[1] *= max(0.0, 1.0 - min(dt * 6.0, 0.25))
                body.velocity += acceleration * dt
                body.position += body.velocity * dt
                body.angular_velocity += body.torque / np.maximum(body.inertia, 1e-12) * dt
                body.rotation += np.degrees(body.angular_velocity * dt)
        for _ in range(self.iterations):
            for body in self.bodies:
                self._floor(body)
            for i, a in enumerate(self.bodies):
                for b in self.bodies[i + 1:]:
                    self._pair(a, b)
        for body in self.bodies:
            if not body.dynamic:
                continue
            if (np.linalg.norm(body.velocity) <= self.sleep_threshold and
                    np.linalg.norm(body.angular_velocity) <= self.sleep_threshold and
                    np.linalg.norm(body.torque) <= 1e-12):
                body.velocity[:] = 0
                body.angular_velocity[:] = 0
                body.sleep_time += dt
                if body.sleep_time >= self.sleep_frames:
                    body.sleeping = True
            else:
                body.sleep_time = 0.0

    def solve_frame(self, frame, cancel=None):
        frame = int(frame)
        if frame in self._frames:
            result = self._frames[frame].copy()
            self._restore(result)
            return result
        start = max(self._frames, default=0)
        if frame < start:
            start = max((cached for cached in self._frames if cached <= frame), default=0)
            self._restore(self._frames.get(start, self.snapshot()))
        dt = 1.0 / (self.fps * self.substeps)
        for _f in range(start + 1, frame + 1):
            for _ in range(self.substeps):
                if cancel is not None:
                    cancel.check()
                self._step(dt)
            self._frames[_f] = self.snapshot()
        if frame == 0 and frame not in self._frames:
            self._frames[frame] = self.snapshot()
        result = self._frames[frame].copy()
        self._restore(result)
        return result

    def _restore(self, snapshot):
        for body, row in zip(self.bodies, snapshot):
            body.position[:] = row[:3]
            body.velocity[:] = row[3:6]
            body.sleeping = bool(row[6])
            if len(row) > 7:
                body.sleep_time = float(row[7])
            if len(row) > 13:
                body.rotation[:] = row[8:11]
                body.angular_velocity[:] = row[11:14]

    def snapshot(self):
        return np.asarray([np.r_[b.position, b.velocity, float(b.sleeping), b.sleep_time, b.rotation,
                                 b.angular_velocity] for b in self.bodies], dtype=np.float64).reshape((-1, 14))


def liquid_reaction(instance, bodies, *, gravity=9.81, strength=0.04):
    """Return the FLIP instance with a small equal-and-opposite contact kick under immersed bodies.

    This is a render/evaluation coupling signal; the upstream FLIP checkpoint remains immutable.
    The impulse is deliberately bounded so a light floating body does not destabilise the liquid.
    """
    if instance is None or not len(getattr(instance, "positions", ())):
        return instance
    positions = np.asarray(instance.positions, dtype=np.float64)
    matrix = np.asarray(getattr(instance, "matrix", np.eye(4)), dtype=np.float64)
    world = (matrix[:3, :3] @ positions.T).T + matrix[:3, 3]
    velocities = np.zeros_like(positions) if instance.velocities is None else np.asarray(instance.velocities).copy()
    changed = False
    for body in bodies:
        low = body.position - body.half_extent
        high = body.position + body.half_extent
        mask = np.all((world >= low) & (world <= high), axis=1)
        if np.any(mask) and body.dynamic:
            local_kick = np.linalg.inv(matrix[:3, :3]) @ np.array((0.0, -abs(gravity) * strength, 0.0))
            velocities[mask] += local_kick.astype(velocities.dtype, copy=False)
            changed = True
    return replace(instance, velocities=velocities) if changed else instance
