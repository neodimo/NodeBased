"""4K viewport benchmark: docs/BENCHMARKS-v0.9-4k.md's method, rerun on the current build, plus
the M2 gate's own measurements (docs/VISION.md, docs/BENCHMARKS-v0.33-m2.md).

M1 method (unchanged, still what docs/BENCHMARKS-v0.33-4k.md's numbers come from): Checker ->
Grade -> Blur -> Merge(A=Blur, B=a second Checker) -> target, at a full 3840x2160 canvas, with
two requests measured against it:

* the full-frame reference evaluator (`Evaluator.evaluate_raster`) over the whole canvas;
* a centered 1920x1080 `TileExecutor` viewport request (25% of the canvas), tile-native for
  every kind in this graph (Checker/Grade/Blur/Merge are all in `tiles.SUPPORTED_TILED_KINDS`).

For each request: cold time to first pixel (fresh Evaluator/TileExecutor, no warm cache), warm
(repeat, identical request), edit p50/p95 (`Grade.exposure` set to a new value each time, then
re-evaluated -- a cache miss by construction, the interactive "drag a slider" case), and peak
process memory over the whole run.

M2 additions:

* **Process startup**, cold and warm. Both launch a fresh subprocess that imports
  `nodebased.core`/`nodebased.imaging` and builds an empty `Dispatcher()` document, timed
  end-to-end (interpreter launch included) from outside the subprocess. "Cold" and "warm" here
  mean the OS's own page/dentry cache for the interpreter and the package's files -- the first
  launch may still be paying for disk I/O the second one no longer has to -- not the evaluator's
  in-memory result cache (a new process has none). The full desktop GUI's own Qt/PySide6 import
  cost is deliberately out of scope: that is a GUI-startup number, not the data-layer readiness
  this measures.
* **Time to first pixel**, 1080p and 4K, on a `Read` -> `Viewer` graph: a real EXR is written to a
  temp file at each size and read back through `Read`, the actual node a document opens with, not
  a synthetic generator like `Checker`.
* **p50/p95 edit latency** of six representative edits, at 1080p and 4K, each sampled
  `--edit-samples` times (10 by default -- the M2 gate's own wording): `Grade.exposure`,
  `Transform.translate_x`, `Merge.mix`, `Blur.radius`, a Roto shape's point dragged
  (`set_shapes` with a moved point, the interactive "drag a control point" case), and a Tracker
  frame step (the node carries no solved track, so its own kernel still runs -- an identity
  transform -- at each new frame, exercising the per-frame recompute path without needing a
  fabricated track history).

    python tools/benchmark_4k_viewport.py [--json out.json] [--edits 40]
        [--edit-samples 10] [--label "CI runner"]

`--label` is stamped into the output table and JSON so a machine that produced the numbers
(a shared CI runner, not a fixed workstation) is never confused for one.
"""
import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.tiles import TileRegion
from nodebased.tileexec import TileExecutor

WIDTH, HEIGHT = 3840, 2160
VIEWPORT_WIDTH, VIEWPORT_HEIGHT = 1920, 1080
RESOLUTIONS = (("1080p", 1920, 1080), ("4K", 3840, 2160))
EDIT_KINDS = ("Grade", "Transform", "Merge", "Blur", "Roto shape drag", "Tracker frame step")


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


_STARTUP_PROBE = (
    "from nodebased.core import Dispatcher\n"
    "from nodebased.imaging import Evaluator\n"
    "Dispatcher()\n"
)


def _time_subprocess(script):
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-c", script], cwd=str(REPO_ROOT),
                   capture_output=True, text=True, check=True)
    return (time.perf_counter() - t0) * 1000.0


def measure_process_startup():
    """Cold and warm process startup to an empty graph. See the module docstring for scope."""
    cold_ms = _time_subprocess(_STARTUP_PROBE)
    warm_ms = _time_subprocess(_STARTUP_PROBE)
    return {"cold_start_ms": cold_ms, "warm_start_ms": warm_ms}


def measure_time_to_first_pixel(width, height, tmpdir):
    """Time to first pixel of a real `Read` -> `Viewer` graph at `width`x`height`."""
    path = Path(tmpdir) / f"ttfp_{width}x{height}.exr"
    frame = np.random.default_rng(0).random((height, width, 4)).astype(np.float32)
    frame[..., 3] = 1.0
    write_exr(path, frame)

    d = Dispatcher()
    d.execute({"op": "create", "type": "Read", "id": "read", "params": {"path": str(path)}})
    d.execute({"op": "create", "type": "Viewer", "id": "viewer", "params": {}})
    d.execute({"op": "connect", "id": "viewer", "input": "image", "source": "read"})

    ev = Evaluator()
    t0 = time.perf_counter()
    ev.evaluate_raster(d.document, "viewer", tier=1, typed=True)
    return (time.perf_counter() - t0) * 1000.0


def _edit_kind_graph(kind, width, height):
    """(`Dispatcher`, target id, `edit(i, frame)` callable) for one of `EDIT_KINDS`.

    `edit` mutates the document (or returns a frame number for Tracker) so the next
    `evaluate_raster` call is a genuine cache miss -- the interactive "drag a slider"/"drag a
    point"/"step a frame" case, the same construction `measure_full_frame`'s own edit loop uses.
    """
    d = Dispatcher()
    if kind == "Grade":
        d.execute({"op": "create", "type": "Checker", "id": "src",
                   "params": {"width": width, "height": height, "size": 32}})
        d.execute({"op": "create", "type": "Grade", "id": "n", "params": {}})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})

        def edit(i):
            d.execute({"op": "set", "id": "n", "param": "exposure", "value": (i + 1) * 0.01})
        return d, "n", edit
    if kind == "Transform":
        d.execute({"op": "create", "type": "Checker", "id": "src",
                   "params": {"width": width, "height": height, "size": 32}})
        d.execute({"op": "create", "type": "Transform", "id": "n", "params": {}})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})

        def edit(i):
            d.execute({"op": "set", "id": "n", "param": "translate_x", "value": float(i + 1)})
        return d, "n", edit
    if kind == "Merge":
        d.execute({"op": "create", "type": "Checker", "id": "a",
                   "params": {"width": width, "height": height, "size": 32}})
        d.execute({"op": "create", "type": "Checker", "id": "b",
                   "params": {"width": width, "height": height, "size": 48}})
        d.execute({"op": "create", "type": "Merge", "id": "n", "params": {"operation": "over"}})
        d.execute({"op": "connect", "id": "n", "input": "A", "source": "a"})
        d.execute({"op": "connect", "id": "n", "input": "B", "source": "b"})

        def edit(i):
            d.execute({"op": "set", "id": "n", "param": "mix", "value": 1.0 - (i + 1) * 0.001})
        return d, "n", edit
    if kind == "Blur":
        d.execute({"op": "create", "type": "Checker", "id": "src",
                   "params": {"width": width, "height": height, "size": 32}})
        d.execute({"op": "create", "type": "Blur", "id": "n", "params": {}})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})

        def edit(i):
            d.execute({"op": "set", "id": "n", "param": "radius", "value": 1.0 + i * 0.1})
        return d, "n", edit
    if kind == "Roto shape drag":
        d.execute({"op": "create", "type": "Roto", "id": "n",
                   "params": {"width": width, "height": height}})

        def edit(i):
            x = 10.0 + i
            shape = {"name": "shape1", "mode": "union", "opacity": 1.0, "feather": 0.0,
                     "points": [{"x": x, "y": 10.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                                {"x": x + 80.0, "y": 10.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                                {"x": x + 80.0, "y": 90.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                                {"x": x, "y": 90.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}]}
            d.execute({"op": "set_shapes", "id": "n", "shapes": [shape]})
        return d, "n", edit
    if kind == "Tracker frame step":
        d.execute({"op": "create", "type": "Constant", "id": "src",
                   "params": {"width": width, "height": height, "red": 0.4, "green": 0.3,
                              "blue": 0.2, "alpha": 1.0}})
        d.execute({"op": "create", "type": "Tracker", "id": "n", "params": {}})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "src"})
        # Keyframed (not constant) tracks, so the solved transform genuinely differs at every
        # frame in range -- a Tracker with no payload or a constant one would mostly cache-hit
        # across a frame step, which is not the interactive cost this edit means to measure.
        tracks = []
        for index, (x, y) in enumerate(((10.0, 10.0), (width - 10.0, 10.0),
                                        (width - 10.0, height - 10.0), (10.0, height - 10.0))):
            tracks.append({"name": f"p{index}", "enabled": 1,
                           "x": {"value": x, "curve": {"interpolation": "linear", "keys": [
                               {"frame": 1, "value": x}, {"frame": 200, "value": x + 20.0}]}},
                           "y": {"value": y, "curve": {"interpolation": "linear", "keys": [
                               {"frame": 1, "value": y}, {"frame": 200, "value": y + 15.0}]}}})
        d.execute({"op": "set_tracks", "id": "n", "tracks": tracks})

        def edit(i):
            pass
        return d, "n", edit
    raise ValueError(f"unknown edit kind {kind!r}")


def measure_edit_kind(kind, width, height, samples):
    d, target, edit = _edit_kind_graph(kind, width, height)
    ev = Evaluator()
    times = []
    for i in range(samples):
        edit(i)
        frame = i + 1 if kind == "Tracker frame step" else None
        t0 = time.perf_counter()
        ev.evaluate_raster(d.document, target, frame=frame, tier=1, typed=True)
        times.append(time.perf_counter() - t0)
    return {"edit_p50_ms": percentile_ms(times, 50), "edit_p95_ms": percentile_ms(times, 95)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edits", type=int, default=40, help="number of edit-latency samples per request")
    parser.add_argument("--edit-samples", type=int, default=10,
                        help="number of samples per M2 representative-edit kind")
    parser.add_argument("--skip-m2", action="store_true",
                        help="run only the M1 method (startup and TTFP measurements need a real "
                             "subprocess and a temp EXR, which a constrained CI step may want to skip)")
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

    if not args.skip_m2:
        startup = measure_process_startup()
        result["process_startup"] = startup
        print(f"process startup: cold {startup['cold_start_ms']:.1f}ms, warm {startup['warm_start_ms']:.1f}ms")

        with tempfile.TemporaryDirectory() as tmpdir:
            ttfp = {label: measure_time_to_first_pixel(w, h, tmpdir) for label, w, h in RESOLUTIONS}
        result["read_to_viewer_ttfp_ms"] = ttfp
        for label, ms in ttfp.items():
            print(f"Read -> Viewer time to first pixel ({label}): {ms:.1f}ms")

        edits = {label: {kind: measure_edit_kind(kind, w, h, args.edit_samples) for kind in EDIT_KINDS}
                for label, w, h in RESOLUTIONS}
        result["edit_samples"] = args.edit_samples
        result["representative_edit_latency"] = edits
        print(f"representative edit latency ({args.edit_samples} samples each):")
        print(f"{'kind':<20}{'1080p p50':>12}{'1080p p95':>12}{'4K p50':>12}{'4K p95':>12}")
        for kind in EDIT_KINDS:
            row1080, row4k = edits["1080p"][kind], edits["4K"][kind]
            print(f"{kind:<20}{row1080['edit_p50_ms']:>10.1f}ms{row1080['edit_p95_ms']:>10.1f}ms"
                  f"{row4k['edit_p50_ms']:>10.1f}ms{row4k['edit_p95_ms']:>10.1f}ms")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
