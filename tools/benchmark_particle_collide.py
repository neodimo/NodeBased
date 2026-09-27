"""Frame time of ParticleCollide3D's self-collision pass, chained onto an emitter, on the CPU.

`python tools/benchmark_particle_collide.py [--counts 2000 20000] [--substeps 4] [--repeat 3]`

The workload is the worst case a settled pile puts on the broad phase: `count` particles packed at
a fixed volume fraction (so most of them have several live contacts every substep, not a sparse
cloud that never enters the grid's neighbour search), timed through `ParticleEmitter.step` with a
`ParticleCollide3D` chained on -- the same call path `nodebased/particles.py`'s solver uses in the
real graph. See docs/SIMULATION.md, "Particle-particle collisions", for the measured number this
tool produced and what it means (CPU only; no GPU path).
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from nodebased import particles
from nodebased.core import SPECS

PACKING_FRACTION = 0.4   # a settled pile of spheres is roughly this fraction solid


def packed_state(count, radius, seed=0):
    """`count` particles of `radius` at rest, packed at `PACKING_FRACTION` inside a cube, with a
    small random velocity so the substep does real work instead of finding everything asleep."""
    volume = count * (4.0 / 3.0) * np.pi * radius ** 3 / PACKING_FRACTION
    side = volume ** (1.0 / 3.0)
    rng = np.random.default_rng(seed)
    position = (rng.random((count, 3)).astype(np.float32) - 0.5) * side
    velocity = ((rng.random((count, 3)).astype(np.float32) - 0.5) * 0.1)
    arrays = {"position": position, "velocity": velocity,
              "age": np.zeros(count, np.int32), "life": np.full(count, 1_000_000, np.int32),
              "size": np.full(count, radius * 2.0, np.float32),
              "color": np.zeros((count, 4), np.float32), "id": np.arange(count, dtype=np.int64)}
    return particles._state(arrays, {"emitted": count, "carry": 0.0, "dropped": 0})


def make_emitter(substeps):
    params = {**SPECS["ParticleEmitter3D"]["params"], "emit_rate": 0.0, "substeps": substeps}
    emitter = particles.ParticleEmitter(params, None, None, 24.0)
    collide_params = {**SPECS["ParticleCollide3D"]["params"], "radius_from_size": 1}
    force = particles.ParticleSelfCollider("ParticleCollide3D", collide_params)
    return emitter.with_force(force)


def best(emitter, state, substeps, repeat):
    times = []
    for _ in range(repeat):
        current, start = state, time.perf_counter()
        for substep in range(substeps):
            current = emitter.step(current, 1, substep, 0)
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=int, nargs="+", default=[2_000, 20_000])
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    print(f"numpy {np.__version__}, python {platform.python_version()}, "
          f"{platform.processor() or platform.machine()}, packing {PACKING_FRACTION}")
    print(f"{'particles':>10} {'substeps':>9} {'ms/frame':>10} {'ms/substep':>11}")
    for count in args.counts:
        emitter = make_emitter(args.substeps)
        state = packed_state(count, args.radius)
        seconds = best(emitter, state, args.substeps, args.repeat)
        print(f"{count:>10,} {args.substeps:>9} {seconds * 1000:>10.1f} "
              f"{seconds * 1000 / args.substeps:>11.2f}")


if __name__ == "__main__":
    main()
