"""Live "sim stats" overlay text for the 3D viewport while a simulation solves.

`SimStatsSnapshot` is a plain value: a solve loop hands one to a callback (or a widget's
`update_snapshot`) after every substep, and it never reaches back into the solver or the cache.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SimStatsSnapshot:
    frame: int
    substep: int          # 0-based
    substeps: int
    ms_per_substep: float
    gpu_memory_mb: "float | None" = None

    def text(self) -> str:
        lines = [f"Frame {self.frame}  substep {self.substep + 1}/{self.substeps}",
                 f"{self.ms_per_substep:.2f} ms/substep"]
        if self.gpu_memory_mb is not None:
            lines.append(f"GPU mem {self.gpu_memory_mb:.0f} MB")
        return "\n".join(lines)


def solve_with_stats(cache, run, target_frame, start_frame, substeps, seed, initial_state, step,
                     step_callback, cancel=None, gpu_memory_mb=None):
    """`simcache.solve_to_frame`, but calling `step_callback(SimStatsSnapshot)` after every
    substep it actually computes (a cache hit calls back zero times). Same arguments as
    `solve_to_frame` plus the callback, so a solve node's own frame loop and the offscreen tests
    see the identical overlay text as a real solve advances."""
    from . import simcache

    def instrumented(state, frame, substep, seed):
        started = time.perf_counter()
        state = step(state, frame, substep, seed)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        step_callback(SimStatsSnapshot(int(frame), int(substep), int(substeps), elapsed_ms, gpu_memory_mb))
        return state

    return simcache.solve_to_frame(cache, run, target_frame, start_frame, substeps, seed,
                                   initial_state, instrumented, cancel)
