#!/usr/bin/env python3
"""Frame time of the defocus gather behind ZDefocus and Bokeh (nodebased/defocusgather.py) at 1920x1080 with a depth ramp.

The per-pixel reference loop is far too slow for a whole frame, so it is timed on a strip of ROWS rows across the full width (the ramp, so the
mix of blur radii) and scaled by 1080 / ROWS; the vectorised gather is timed on the whole frame. The largest difference between the two is measured
on that same strip. Usage: benchmark_defocus_gather.py [--rows 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased import bokehknobs, defocusgather  # noqa: E402
from nodebased.core import SPECS  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402

W, H = 1920, 1080


def picture(h, w):
    rng = np.random.default_rng(5)
    image = rng.random((h, w, 4), dtype=np.float32)
    image[..., 3] = 1.0
    return image


def ramp(h, w):
    return np.tile(np.linspace(0.5, 6.0, w, dtype=np.float32), (h, 1))


def zdefocus(image, depth, size):
    params = {**SPECS["ZDefocus"]["params"], "shape": 1.0, "focal_plane": 3.0, "depth_of_field": 0.2, "size": float(size), "max_size": float(size)}
    return Evaluator._zdefocus(image, depth, params)


def bokeh(image, depth, size):
    params = {**SPECS["Bokeh"]["params"], "focalPlane": 3.0, "focusRegionSize": 0.2, "max_kernelsize": 2.0 * size, "multiplier": 10.0}
    planes = bokehknobs.planes_for(image, params, lambda name: None)
    return bokehknobs.render(image, depth, params, None, planes)


CASES = {"ZDefocus": zdefocus, "Bokeh": bokeh}


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()
    print(f"{'case':10s} {'size':>4s} {'old (strip x scale)':>20s} {'gather only':>12s} {'gather + FFT':>13s} {'speedup':>8s} {'max diff (strip, vs old)':>25s} {'FFT vs gather (frame)':>22s}")
    for name, fn in CASES.items():
        for size in (20, 50):
            strip_image, strip_depth = picture(args.rows, W), ramp(args.rows, W)
            Evaluator._zdefocus_reference = True
            old, t_old = timed(fn, strip_image, strip_depth, size)
            Evaluator._zdefocus_reference = False
            new_strip, _ = timed(fn, strip_image, strip_depth, size)
            diff = float(np.abs(np.asarray(old, np.float64) - np.asarray(new_strip, np.float64)).max())
            frame_image, frame_depth = picture(H, W), ramp(H, W)
            gain = defocusgather.FFT_GAIN
            try:
                defocusgather.FFT_GAIN = 0
                plain, t_plain = timed(fn, frame_image, frame_depth, size)
            finally:
                defocusgather.FFT_GAIN = gain
            fast, t_new = timed(fn, frame_image, frame_depth, size)
            fft_diff = float(np.abs(np.asarray(plain, np.float64) - np.asarray(fast, np.float64)).max())
            t_old_frame = t_old * H / args.rows
            print(f"{name:10s} {size:4d} {t_old_frame:17.1f} s {t_plain:10.2f} s {t_new:11.2f} s {t_old_frame / t_new:7.0f}x {diff:25.2e} {fft_diff:22.2e}", flush=True)


if __name__ == "__main__":
    main()
