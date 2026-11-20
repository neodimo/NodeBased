"""Fluids 18 B: seed one vortex ring in still air, no emitter, and follow its peak vorticity and kinetic energy through 48 frames.

A correct solver keeps a finer grid's vortex alive longer. Usage (from the repository root, PYTHONPATH set to it):

    python examples/fire_shot/vortex_decay.py --cells 64 128 --backend gpu --out benchmarks/fire_shot/vortex-decay.json

Every setting is the node's own default except the ones named on the command line. World units throughout (metres, seconds);
the box is 2 m on a side, closed, 24 frames per second.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

FPS = 24.0




def run(*args, **kwargs):
    from nodebased import fluid_diagnostics
    return fluid_diagnostics.follow_ring_vortex(*args, **kwargs)


def summarise(rows):
    """Decay rates: the log-slope per second of peak vorticity and of energy over frames 0 to 48 (least squares), and the fractions left."""
    t = np.array([r["frame"] for r in rows], np.float64) / FPS
    out = {}
    for key in ("peak_vorticity", "energy"):
        y = np.log(np.maximum([r[key] for r in rows], 1e-30))
        out[key + "_rate_per_s"] = round(float(-np.polyfit(t, y, 1)[0]), 4)
        out[key + "_left"] = round(float(rows[-1][key] / rows[0][key]), 4)
    after = rows[1:]
    out["max_divergence_after_projection"] = round(float(max(r["max_divergence"] for r in after)), 5)
    out["pressure_iterations_max"] = int(max(r["pressure_iterations"] for r in after))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--backend", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--vorticity", type=float, default=0.0)
    parser.add_argument("--advection", default="maccormack")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--peak-speed", type=float, default=0.5, help="largest speed of the seeded ring, m/s")
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--core", type=float, default=0.1)
    parser.add_argument("--extra", default="{}", help="JSON of solver parameter overrides")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = []
    for cells in args.cells:
        rows, seconds = run(cells, args.frames, args.backend, peak_speed=args.peak_speed, radius=args.radius, core=args.core, substeps=args.substeps, vorticity=args.vorticity, advection=args.advection,
                            max_iterations=args.max_iterations, tolerance=args.tolerance, extra=json.loads(args.extra))
        entry = {"cells": cells, "backend": args.backend, "peak_speed": args.peak_speed, "substeps": args.substeps, "vorticity": args.vorticity, "advection": args.advection,
                 "extra": json.loads(args.extra), "seconds": round(seconds, 2), **summarise(rows), "rows": rows[::6]}
        results.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "rows"}), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
