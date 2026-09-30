"""Deterministic CPU reference for small rigid-body scenes (world units, seconds)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


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
    sleeping: bool = False
    sleep_time: float = 0.0

    def __post_init__(self):
        self.shape = str(self.shape).lower()
        if self.shape not in {"box", "sphere", "convex", "compound"}:
            raise ValueError("shape must be box, sphere, convex or compound")
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3).copy()
        self.size = np.maximum(np.asarray(self.size, dtype=np.float64).reshape(3), 1e-6)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3).copy()
        if self.density <= 0 or self.mass < 0:
            raise ValueError("density must be positive and mass cannot be negative")
        if not 0 <= self.friction <= 100 or not 0 <= self.restitution <= 1:
            raise ValueError("friction must be nonnegative and restitution in [0, 1]")

    @property
    def volume(self):
        if self.shape == "sphere":
            return float(4 * np.pi * np.prod(self.size) / 24.0)  # size stores diameters
        if self.shape == "compound":
            return sum(child.volume for child in self.children)
        return float(np.prod(self.size))

    @property
    def effective_mass(self):
        return float(self.mass or self.density * self.volume) if self.dynamic else float("inf")

    @property
    def half_extent(self):
        if self.shape == "sphere":
            return np.repeat(float(np.max(self.size)) / 2, 3)
        return self.size / 2


class RigidSolver3D:
    """Stable sequential-impulse solver with deterministic pair order and cached frame states.

    The reference implementation uses axis-aligned contact bounds (spheres use their radius for
    broad/narrow phase); rotation is intentionally omitted until the angular solver lands.
    """
    def __init__(self, bodies=(), gravity=(0.0, -9.81, 0.0), fps=24.0, substeps=4,
                 floor_y=0.0, iterations=8, sleep_threshold=0.025, sleep_frames=0.5):
        self.bodies = list(bodies)
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.fps, self.substeps = float(fps), int(substeps)
        self.floor_y = floor_y if floor_y is None else float(floor_y)
        self.iterations = int(iterations)
        self.sleep_threshold, self.sleep_frames = float(sleep_threshold), float(sleep_frames)
        if self.fps <= 0 or self.substeps < 1 or self.iterations < 1:
            raise ValueError("fps, substeps and iterations must be positive")
        self._frames = {}

    def _floor(self, body):
        if not body.dynamic or self.floor_y is None:
            return
        low = body.position[1] - body.half_extent[1]
        if low < self.floor_y:
            body.position[1] += self.floor_y - low
            vy = body.velocity[1]
            if vy < 0:
                body.velocity[1] = -vy * body.restitution
            body.velocity[[0, 2]] *= max(0.0, 1.0 - body.friction * 0.12)
            if abs(body.velocity[1]) < self.sleep_threshold:
                body.velocity[1] = 0.0

    def _pair(self, a, b):
        if not a.dynamic and not b.dynamic:
            return
        delta = b.position - a.position
        overlap = a.half_extent + b.half_extent - np.abs(delta)
        if np.any(overlap <= 0):
            return
        axis = int(np.argmin(overlap))
        sign = 1.0 if delta[axis] >= 0 else -1.0
        inv_a = 1.0 / a.effective_mass if a.dynamic else 0.0
        inv_b = 1.0 / b.effective_mass if b.dynamic else 0.0
        inv_sum = inv_a + inv_b
        if inv_sum <= 0:
            return
        correction = max(0.0, overlap[axis] - 1e-8) / inv_sum
        a.position[axis] -= sign * correction * inv_a
        b.position[axis] += sign * correction * inv_b
        relative = b.velocity[axis] - a.velocity[axis]
        if relative * sign < 0:
            impulse = -(1 + min(a.restitution, b.restitution)) * relative / inv_sum
            a.velocity[axis] -= sign * impulse * inv_a
            b.velocity[axis] += sign * impulse * inv_b
            tangent = [i for i in range(3) if i != axis]
            friction = min(a.friction, b.friction)
            a.velocity[tangent] *= max(0.0, 1.0 - friction * 0.08)
            b.velocity[tangent] *= max(0.0, 1.0 - friction * 0.08)

    def _step(self, dt):
        for body in self.bodies:
            if body.dynamic and not body.sleeping:
                body.velocity += self.gravity * dt
                body.position += body.velocity * dt
        for _ in range(self.iterations):
            for body in self.bodies:
                self._floor(body)
            for i, a in enumerate(self.bodies):
                for b in self.bodies[i + 1:]:
                    self._pair(a, b)
        for body in self.bodies:
            if not body.dynamic:
                continue
            if np.linalg.norm(body.velocity) <= self.sleep_threshold:
                body.velocity[:] = 0
                body.sleep_time += dt
                if body.sleep_time >= self.sleep_frames:
                    body.sleeping = True
            else:
                body.sleep_time = 0.0

    def solve_frame(self, frame, cancel=None):
        frame = int(frame)
        if frame in self._frames:
            return self._frames[frame].copy()
        start = max(self._frames, default=0)
        if frame < start:
            raise ValueError("backward rigid-body solve requires a cached frame")
        dt = 1.0 / (self.fps * self.substeps)
        for _f in range(start + 1, frame + 1):
            for _ in range(self.substeps):
                if cancel is not None:
                    cancel.check()
                self._step(dt)
            self._frames[_f] = self.snapshot()
        if frame == 0 and frame not in self._frames:
            self._frames[frame] = self.snapshot()
        return self._frames[frame].copy()

    def snapshot(self):
        return np.asarray([np.r_[b.position, b.velocity, float(b.sleeping)] for b in self.bodies], dtype=np.float64)
