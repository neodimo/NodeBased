"""4K viewport benchmark: docs/BENCHMARKS-v0.9-4k.md's method, rerun on the current build.

Graph: Checker -> Grade -> Blur -> Merge(A=Blur, B=a second Checker) -> target, at a full
3840x2160 canvas. Two requests are measured against it, matching the v0.9 table exactly:

* the full-frame reference evaluator (`Evaluator.evaluate_raster`) over the whole canvas;
* a centered 1920x1080 `TileExecutor` viewport request (25% of the canvas), tile-native for
  every kind in this graph (Checker/Grade/Blur/Merge are all in `tiles.SUPPORTED_TILED_KINDS`).

For each request: cold time to first pixel (fresh Evaluator/TileExecutor, no warm cache), warm
(repeat, identical request), edit p50/p95 (`Grade.exposure` set to a new value each time, then
re-evaluated -- a cache miss by construction, the interactive "drag a slider" case), and peak
process memory over the whole run.

    python tools/benchmark_4k_viewport.py [--json out.json] [--edits 40] [--label "CI runner"]

`--label` is stamped into the output table and JSON so a machine that produced the numbers
(a shared CI runner, not a fixed workstation) is never confused for one.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.tiles import TileRegion
from nodebased.tileexec import TileExecutor

WIDTH, HEIGHT = 3840, 2160
VIEWPORT_WIDTH, VIEWPORT_HEIGHT = 1920, 1080


def peak_memory_mb():
    """Peak resident/working-set memory of this process, in MB.

    `resource.getrusage` (the convention every other `tools/benchmark_*.py` in this repo
    uses) is POSIX-only and raises ImportError on Windows; this benchmark is the first one
    wired into the Windows CI job, so it falls back to the Win32 API on that platform. Linux
    reports `ru_maxrss` in KiB; Windows' `PeakWorkingSetSize` is in bytes.
    """
    if sys.platform == "win32":
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if not ok:
            return float("nan")
        return counters.PeakWorkingSetSize / (1024 * 1024)
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_document():
    d = Dispatcher()
    d.execute({"op": "create", "type": "Checker", "id": "checker",
               "params": {"width": WIDTH, "height": HEIGHT, "size": 32}})
    d.execute({"op": "create", "type": "Grade", "id": "grade", "params": {"exposure": 0.0}})
    d.execute({"op": "connect", "id": "grade", "input": "image", "source": "checker"})
    d.execute({"op": "create", "type": "Blur", "id": "blur", "params": {"radius": 4.0}})
    d.execute({"op": "connect", "id": "blur", "input": "image", "source": "grade"})
    d.execute({"op": "create", "type": "Checker", "id": "checker_b",
               "params": {"width": WIDTH, "height": HEIGHT, "size": 48}})
    d.execute({"op": "create", "type": "Merge", "id": "merge", "params": {"operation": "over"}})
    d.execute({"op": "connect", "id": "merge", "input": "A", "source": "blur"})
    d.execute({"op": "connect", "id": "merge", "input": "B", "source": "checker_b"})
    return d, "merge"


def centered_viewport(exe, document, target):
    bounds = exe.canvas_region(document, target, frame=1, tier=1)
    x = bounds.full_x + (bounds.full_width - VIEWPORT_WIDTH) // 2
    y = bounds.full_y + (bounds.full_height - VIEWPORT_HEIGHT) // 2
    return TileRegion(x=x, y=y, width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT,
                      full_width=bounds.full_width, full_height=bounds.full_height,
                      full_x=bounds.full_x, full_y=bounds.full_y)


def percentile_ms(samples, pct):
    return float(np.percentile(np.asarray(samples) * 1000.0, pct))


def measure_full_frame(edits):
    d, target = build_document()

    ev = Evaluator()
    t0 = time.perf_counter()
    ev.evaluate_raster(d.document, target, tier=1, typed=True)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    ev.evaluate_raster(d.document, target, tier=1, typed=True)
    warm = time.perf_counter() - t0

    edit_times = []
    for i in range(edits):
        d.execute({"op": "set", "id": "grade", "param": "exposure", "value": (i + 1) * 0.01})
        t0 = time.perf_counter()
        ev.evaluate_raster(d.document, target, tier=1, typed=True)
        edit_times.append(time.perf_counter() - t0)

    return {"cold_ttfp_ms": cold * 1000.0, "warm_ms": warm * 1000.0,
            "edit_p50_ms": percentile_ms(edit_times, 50), "edit_p95_ms": percentile_ms(edit_times, 95)}


def measure_tiled_viewport(edits):
    d, target = build_document()

    exe = TileExecutor()
    region = centered_viewport(exe, d.document, target)
    t0 = time.perf_counter()
    exe.compose_region(d.document, target, region, frame=1, tier=1)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    exe.compose_region(d.document, target, region, frame=1, tier=1)
    warm = time.perf_counter() - t0

    edit_times = []
    for i in range(edits):
        d.execute({"op": "set", "id": "grade", "param": "exposure", "value": (i + 1) * 0.01})
        region = centered_viewport(exe, d.document, target)
        t0 = time.perf_counter()
        exe.compose_region(d.document, target, region, frame=1, tier=1)
        edit_times.append(time.perf_counter() - t0)

    return {"cold_ttfp_ms": cold * 1000.0, "warm_ms": warm * 1000.0,
            "edit_p50_ms": percentile_ms(edit_times, 50), "edit_p95_ms": percentile_ms(edit_times, 95)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edits", type=int, default=40, help="number of edit-latency samples per request")
    parser.add_argument("--json", type=Path, default=None, help="write the results as JSON to this path")
    parser.add_argument("--label", default=platform.node() or "unknown",
                        help="identifies the machine that produced this run (e.g. a CI runner name)")
    args = parser.parse_args()

    full = measure_full_frame(args.edits)
    tiled = measure_tiled_viewport(args.edits)
    peak_mb = peak_memory_mb()

    result = {
        "label": args.label,
        "commit": None,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "edits_sampled": args.edits,
        "peak_memory_mb": peak_mb,
        "full_frame_3840x2160": full,
        "tiled_viewport_1920x1080": tiled,
    }

    print(f"[{args.label}] {result['platform']}, Python {result['python']}, NumPy {result['numpy']}")
    print(f"{'request':<38}{'cold TTFP':>12}{'warm':>10}{'edit p50':>12}{'edit p95':>12}")
    for name, row in (("full 3840x2160 reference evaluator", full),
                      ("centered 1920x1080 TileExecutor viewport", tiled)):
        print(f"{name:<38}{row['cold_ttfp_ms']:>10.1f}ms{row['warm_ms']:>8.1f}ms"
              f"{row['edit_p50_ms']:>10.1f}ms{row['edit_p95_ms']:>10.1f}ms")
    print(f"peak process memory: {peak_mb:.1f} MB")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
