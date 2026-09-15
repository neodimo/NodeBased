"""Benchmark for `nodebased.color.display_rgb`.

This isolates the view-transform cost from the rest of the evaluator so a change to
`display_rgb`, the OCIO processor cache, or the GPU display path can be measured on its
own instead of being buried inside a full-graph benchmark. Synthetic data is used rather
than a project's own EXRs so the benchmark has no dependency outside this repository:
the values are float32 ACEScg-shaped HDR data (some >1, some negative, matching the kind
of pixels a real scene-linear composite produces), not a plausible image.

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m nodebased.bench_display
    QT_QPA_PLATFORM=xcb DISPLAY=:0 .venv/bin/python -m nodebased.bench_display --backend all
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

RESOLUTIONS = {"hd": (1920, 1080), "4k": (3840, 2160)}
VIEWS = ("sRGB", "ACES 2.0")
SEED = 20260914


def synthetic_frame(width: int, height: int) -> np.ndarray:
    """Deterministic float32 HxWx3 ACEScg-shaped data: negatives, >1 highlights, mid-grey body."""
    rng = np.random.default_rng(SEED ^ (width * 73856093) ^ (height * 19349663))
    base = rng.random((height, width, 3), dtype=np.float32) * 0.6 + 0.02
    # Sparse bright highlights well above 1.0 and a few sub-zero specks, both legal in
    # scene-linear ACEScg and exactly the values a naive clamp-based transform would get wrong.
    highlight = rng.random((height, width, 3)) < 0.01
    base[highlight] += rng.random(int(highlight.sum())).astype(np.float32) * 8.0
    negative = rng.random((height, width, 3)) < 0.005
    base[negative] -= rng.random(int(negative.sum())).astype(np.float32) * 0.2
    return base.astype(np.float32)


def time_calls(fn, iterations: int, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return {
        "min_ms": round(ordered[0], 3),
        "median_ms": round(statistics.median(ordered), 3),
        "mean_ms": round(statistics.mean(ordered), 3),
        "max_ms": round(ordered[-1], 3),
        "iterations": iterations,
    }


def run_cpu(resolution: str, view: str, iterations: int) -> dict:
    """Times the CPU path directly (cached processor + threaded `applyRGB`).

    Deliberately bypasses `color.display_rgb`, which prefers the GPU path whenever one is
    available: this backend exists to measure the CPU path in isolation, on any machine,
    not "whichever path display_rgb happened to pick here."
    """
    from .color import display_processor, apply_threaded
    width, height = RESOLUTIONS[resolution]
    frame = synthetic_frame(width, height)
    cpu_processor = display_processor(view)
    return time_calls(lambda: apply_threaded(cpu_processor, frame.copy()), iterations)


def _ensure_qapp():
    """A QOffscreenSurface needs a QGuiApplication; bench_display has no GUI of its own."""
    from PySide6.QtWidgets import QApplication
    if QApplication.instance() is None:
        global _qapp
        _qapp = QApplication([])


def run_gpu(resolution: str, view: str, iterations: int) -> dict:
    from . import gpudisplay
    _ensure_qapp()
    width, height = RESOLUTIONS[resolution]
    frame = synthetic_frame(width, height)
    display = gpudisplay.get_display(force=True)
    if display is None:
        return {"skipped": "no GPU context available"}
    return time_calls(lambda: display.render(frame, view), iterations)


def machine_identity() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure NodeBased display_rgb latency.")
    parser.add_argument("--backend", choices=["cpu", "gpu", "all"], default="cpu")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS) + ["all"], default="all")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    resolutions = list(RESOLUTIONS) if args.resolution == "all" else [args.resolution]
    backends = ["cpu", "gpu"] if args.backend == "all" else [args.backend]

    results = {"machine": machine_identity(), "results": []}
    for resolution in resolutions:
        for view in VIEWS:
            for backend in backends:
                runner = run_cpu if backend == "cpu" else run_gpu
                entry = {"resolution": resolution, "view": view, "backend": backend,
                          **runner(resolution, view, args.iterations)}
                results["results"].append(entry)
                if not args.json:
                    if "skipped" in entry:
                        print(f"{resolution:>4} {view:<10} {backend:<4}  skipped: {entry['skipped']}")
                    else:
                        print(f"{resolution:>4} {view:<10} {backend:<4}  "
                              f"median {entry['median_ms']:>8.2f} ms  "
                              f"min {entry['min_ms']:>8.2f} ms  max {entry['max_ms']:>8.2f} ms")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
