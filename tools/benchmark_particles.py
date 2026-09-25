"""Particles per second of the particle solver alone (no cache, no scene, no rendering).

`python tools/benchmark_particles.py [--sizes 10000 100000 1000000] [--repeat 3]`

Two workloads at each live-particle count N, both timed on `ParticleEmitter.step` (one substep):

* advance only: N particles alive, nothing born or dying, so the step is the integrate.
* steady state: N alive, a birth rate equal to the death rate (lifetime 100 frames, N / 100 births per
  frame), so every substep emits, advances and culls.

The figure is live particles processed per second (N divided by the best step time of `--repeat`).
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from nodebased import particles
from nodebased.core import SPECS

LIFE_FRAMES = 100


def knobs(**overrides):
    return {**SPECS["ParticleEmitter3D"]["params"], **overrides}


def filled(count):
    """An emitter and a state holding `count` particles, all born on one step."""
    emitter = particles.ParticleEmitter(knobs(emit_rate=float(count), life=1.0e6, emit_speed=1.0,
                                              spread=45.0, max_particles=count + 1), None, None, 24.0)
    state = emitter.step(emitter.initial_state(0), 1, 0, 0)
    assert len(state.arrays["id"]) == count
    return emitter, state


def steady(count):
    """An emitter and a state at the balance point where births equal deaths."""
    rate = count / LIFE_FRAMES
    emitter = particles.ParticleEmitter(knobs(emit_rate=rate, life=float(LIFE_FRAMES), emit_speed=1.0,
                                              spread=45.0, max_particles=count * 2), None, None, 24.0)
    state = emitter.initial_state(0)
    for frame in range(1, LIFE_FRAMES + 1):
        state = emitter.step(state, frame, 0, 0)
    return emitter, state, rate


def best(step, state, frame0, repeat, calls=5):
    times = []
    for attempt in range(repeat):
        current, start = state, time.perf_counter()
        for offset in range(calls):
            current = step(current, frame0 + offset, 0, 0)
        times.append((time.perf_counter() - start) / calls)
    return min(times), len(current.arrays["id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    print(f"numpy {np.__version__}, python {platform.python_version()}, {platform.processor() or platform.machine()}")
    print(f"{'live':>10} {'workload':<14} {'ms/substep':>11} {'particles/s':>14}")
    for count in args.sizes:
        emitter, state = filled(count)
        emitter.params["emit_rate"] = 0.0
        emitter._resolved.clear()
        seconds, alive = best(emitter.step, state, 2, args.repeat)
        print(f"{alive:>10,} {'advance only':<14} {seconds * 1000:>11.2f} {alive / seconds:>14,.0f}")
        emitter, state, rate = steady(count)
        seconds, alive = best(emitter.step, state, LIFE_FRAMES + 1, args.repeat)
        print(f"{alive:>10,} {'steady state':<14} {seconds * 1000:>11.2f} {alive / seconds:>14,.0f}")


if __name__ == "__main__":
    main()
