"""Median-of-five timing of Undistort and Redistort on the CPU reference and on the card (Rendering 13 step G1).

    python tools/benchmark_lens_warp.py [cpu|gpu] [--json FILE]

Float RGBA, the bilinear filter and the lens of the distorted demo plate. Each size is warmed up once (shader build,
allocations) and then timed five times through `Evaluator._lens_warp`, so the figure is the node's own cost: the
lens maths, the resample, and for the card the upload and the read back. Redistort reads the grown image Undistort
wrote. Run it under `flock /tmp/nb-gpu.lock` when the card's number matters.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nodebased.imaging import Evaluator, Raster, Region  # noqa: E402

SIZES = ((1920, 1080), (3840, 2160))
LENS = dict(lens_preset="custom", lens_k1=-0.03145, lens_k2=0.00423, lens_k3=0.0, lens_p1=0.0, lens_p2=0.0,
            lens_cx=0.0, lens_cy=0.0, anamorphic_squeeze=1.0, filter="bilinear")


def measure(backend, repeats=5):
    params = {**LENS, "lens_backend": backend}
    results = {}
    for width, height in SIZES:
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        pixels = np.stack([0.5 + 0.4 * np.sin(xx / 37), 0.5 + 0.4 * np.cos(yy / 41),
                           0.5 + 0.3 * np.sin((xx + yy) / 53), np.ones_like(xx)], -1).astype(np.float32)
        region = Region(0, 0, width, height)
        plate = Raster(pixels, region, region)
        grown = Evaluator._lens_warp("Undistort", params, [plate])
        for kind, source in (("Undistort", plate), ("Redistort", grown)):
            Evaluator._lens_warp(kind, params, [source])
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                Evaluator._lens_warp(kind, params, [source])
                times.append(time.perf_counter() - started)
            results[f"{kind} {width}x{height}"] = round(statistics.median(times) * 1000, 1)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backend", nargs="?", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--json")
    args = parser.parse_args()
    table = measure(args.backend)
    print(args.backend, json.dumps(table))
    if args.json:
        Path(args.json).write_text(json.dumps({args.backend: table}, indent=1))
