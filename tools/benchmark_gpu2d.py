#!/usr/bin/env python3
"""R6-1 part 4: measure the realistic comp with Grade, Transform, Blur and Merge resident on the GPU, and decide.

Four commands, one JSON file each (raw per-frame records are kept in the file):

``kernels``   Sources are decoded once into RAM ("predecoded"), then the four-node chain is timed on its own in
              three modes over the same frames and parameters:
                cpu        the CPU kernels (`Evaluator._windowed_kernel`, the truth for the pixel check)
                roundtrip  one GPU run per node: every node uploads its inputs and reads its result back
                resident   the island: stage the four nodes, upload the sources, run, one terminal readback
                           * ``fresh``  every source uploaded for every frame (the transfer-inclusive number)
                           * ``warm``   sources already on the device (keyed uploads, the kernel-only number)
              plus a pixel check of roundtrip and resident against the CPU chain, and the counters.
``graph``     The full node graph (Read decodes, Roto, the four nodes, ColorCorrect) through `Evaluator`, serial and
              without a window, on the CPU or with the island: per frame wall time and per-node times.
``playback``  Runs `tools/benchmark_2d_playback.py` (the real window, accepted pictures) for N fresh passes and
              joins each island's counters to the accepted frame whose node evaluation contained it.
``gates``     Reads the files above and prints the go/no-go table for every R6-1 gate.

Timing method (labelled in every file): GPU time is bounded completion timing. ``gpu_ms`` runs from the queue
submit until a 16-byte flags readback returns, which happens after every pass has finished; enqueue time is never
used. GPU timestamp queries are not used: the shared device (`gpu3d._state`) is requested with only the
``float32-blendable`` feature, so ``timestamp-query`` is not enabled and this wgpu build's queue work-done callback fails.

Examples (``$NB_RUN`` is the disk scratch directory; sources from `tools/benchmark_2d_playback.py --generate-inputs`)::

    python tools/benchmark_gpu2d.py kernels --inputs $NB_RUN/plates --adapter default --label rtx3080ti --output k.json
    python tools/benchmark_gpu2d.py graph --inputs $NB_RUN/plates --backend gpu2d --adapter default --output g.json
    python tools/benchmark_gpu2d.py playback --inputs $NB_RUN/plates --display :99 --backend gpu2d --label rtx3080ti \\
        --frame-range 48 --passes 3 --out-dir $NB_RUN/pb
    python tools/benchmark_gpu2d.py gates --tier rtx3080ti --kernels k1.json k2.json k3.json \\
        --playback pb/summary-gpu2d-48-rtx3080ti.json --cpu-playback pb/summary-cpu-48-rtx3080ti.json
"""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
FIXTURE = ROOT / "tests/fixtures/playback_comp.nbcomp"
CHAIN = (("grade", "Grade"), ("transform", "Transform"), ("blur", "Blur"), ("merge", "Merge"))
TIMING_METHOD = ("bounded completion: gpu_ms = queue submit until a 16-byte flags readback returns (every pass finished); "
                 "upload_ms = host write_buffer calls; readback_ms = copy and map of the finished image; "
                 "exec_ms = stage + encode + passes to completion (no source upload, no terminal readback); "
                 "total_ms = stage start until the Raster is on the host; no GPU timestamp queries (feature not enabled on the shared device)")
GATES = {"kernel_p95_ms": 20.0, "transfer_p95_ms": 30.0, "node_stage_speedup": 2.0,
         "intermediate_fps": {"rtx3080ti": 16.0, "rtx4070laptop": 12.0}, "intermediate_p95_ms": {"rtx3080ti": 62.5, "rtx4070laptop": 83.33},
         "final_fps": 24.0, "final_p95_ms": 41.67}


# ---------------------------------------------------------------------------------------------------------
# shared helpers

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pct(values, q):
    return round(float(np.percentile(values, q)), 3) if len(values) else None


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(float(np.mean(values)), 3), "p50": pct(values, 50), "p95": pct(values, 95),
            "p99": pct(values, 99), "max": round(max(values), 3), "min": round(min(values), 3)}


def parse_frames(spec):
    """``1-48``, ``49-240:8`` or ``1,5,9`` into a list of ints."""
    frames = []
    for part in spec.split(","):
        part = part.strip()
        step = 1
        if ":" in part:
            part, step = part.split(":")
            step = int(step)
        if "-" in part:
            a, b = part.split("-")
            frames.extend(range(int(a), int(b) + 1, step))
        else:
            frames.append(int(part))
    return frames


def parse_size(text):
    if text == "1080p":
        return 1920, 1080
    if text == "4k":
        return 3840, 2160
    w, h = text.lower().split("x")
    return int(w), int(h)


def run_text(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=ROOT).stdout.strip()
    except Exception:
        return None


def portable(text):
    """The command line without machine-specific paths (evidence files are committed)."""
    text = str(text)
    for prefix, name in ((os.environ.get("NB_RUN"), "$NB_RUN"), (str(ROOT), "$WORKTREE"), (str(ROOT.resolve()), "$WORKTREE"),
                         (os.path.dirname(sys.executable), "$VENV_BIN"), (str(pathlib.Path.home()), "$HOME"), (str(pathlib.Path.home().resolve()), "$HOME")):
        if prefix:
            text = text.replace(prefix, name)
    return text


def environment(args, backend=None, extra=None):
    cpu = None
    try:
        cpu = next((l.split(":", 1)[1].strip() for l in pathlib.Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), None)
    except OSError:
        pass
    versions = {}
    for name in ("numpy", "wgpu", "PySide6"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    nvidia = run_text("nvidia-smi", "--query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,"
                      "pcie.link.width.current,pcie.link.width.max", "--format=csv,noheader")
    record = {"revision": run_text("git", "rev-parse", "HEAD"), "working_tree_dirty": bool(run_text("git", "status", "--porcelain")),
              "fixture_sha256": __import__("hashlib").sha256(FIXTURE.read_bytes()).hexdigest(), "host": platform.node(),
              "os": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(), "python": platform.python_version(), "versions": versions,
              "nvidia_smi": nvidia, "command": portable(" ".join(sys.argv)),
              "disk_cache": os.environ.get("NODEBASED_DISK_CACHE"), "timing_method": TIMING_METHOD}
    if backend is not None:
        record["gpu"] = backend.describe()
    if extra:
        record.update(extra)
    return record


def open_backend(adapter, budget_mb):
    from nodebased import gpu2d
    backend = gpu2d.GPU2D.open(adapter, budget_mb << 20)
    if backend is None:
        raise SystemExit(f"no GPU 2D backend on adapter {adapter!r}: {gpu2d.GPU2D.open_error}")
    return backend


def pick(delta):
    keys = ("uploads", "upload_bytes", "upload_cache_hits", "downloads", "download_bytes", "intermediate_readbacks", "submissions",
            "dispatches", "allocations", "pool_hits", "nonfinite_fallbacks", "pipeline_compiles", "shader_compile_ms",
            "upload_ms", "encode_ms", "gpu_ms", "readback_ms", "readback_map_ms", "readback_host_ms")
    row = {k: (round(delta[k], 3) if isinstance(delta[k], float) else delta[k]) for k in keys}
    row["cpu_boundaries"] = list(delta["cpu_boundaries"])
    return row


# ---------------------------------------------------------------------------------------------------------
# kernels

def node_params(doc, key, frame):
    from nodebased.animation import resolve_params
    from nodebased.core import LIMITS, SPECS
    node = doc["nodes"][key]
    return resolve_params(node, doc.get("animation", {}).get("curves", {}).get(key), frame, SPECS[node["type"]]["params"], LIMITS)


class Frame:
    """One predecoded frame: the three sources and the four resolved parameter sets."""

    def __init__(self, doc, evaluator, frame):
        evaluator.clear()
        self.frame = frame
        self.plate = evaluator.evaluate_raster(doc, "plate", frame=frame)
        self.background = evaluator.evaluate_raster(doc, "background", frame=frame)
        self.matte = evaluator.evaluate_raster(doc, "roto", frame=frame)
        self.params = {key: node_params(doc, key, frame) for key, _ in CHAIN}

    def nbytes(self):
        return sum(r.pixels.nbytes for r in (self.plate, self.background, self.matte))


def chain_cpu(item, times=None):
    from nodebased.imaging import Evaluator
    kernel = Evaluator._windowed_kernel
    p = item.params
    stamps = []

    def step(kind, key, inputs):
        t = time.perf_counter()
        out = kernel(kind, p[key], inputs, item.frame, None)
        stamps.append((time.perf_counter() - t) * 1000)
        return out
    grade = step("Grade", "grade", [item.plate, None])
    transform = step("Transform", "transform", [grade])
    blur = step("Blur", "blur", [transform, None])
    merge = step("Merge", "merge", [blur, item.background, item.matte])
    if times is not None:
        times.update(grade_ms=stamps[0], transform_ms=stamps[1], blur_ms=stamps[2], merge_ms=stamps[3])
    return merge


def chain_resident(backend, item, keyed):
    """Stage the four nodes and run the island once. ``keyed`` lets the device keep the sources between runs."""
    from nodebased import gpu2d
    f = item.frame
    keys = (("plate", f), ("background", f), ("matte", f)) if keyed else (None, None, None)
    before = backend.counters.snapshot()
    t0 = time.perf_counter()
    grade = backend.stage("Grade", item.params["grade"], [item.plate, None], [keys[0], None])
    transform = backend.stage("Transform", item.params["transform"], [grade])
    blur = backend.stage("Blur", item.params["blur"], [transform, None])
    merge = backend.stage("Merge", item.params["merge"], [blur, item.background, item.matte], [None, keys[1], keys[2]])
    t1 = time.perf_counter()
    out = backend.run(merge)
    t2 = time.perf_counter()
    delta = gpu2d.Counters.delta(backend.counters.snapshot(), before)
    row = pick(delta)
    row.update(frame=f, stage_ms=round((t1 - t0) * 1000, 3), run_ms=round((t2 - t1) * 1000, 3), total_ms=round((t2 - t0) * 1000, 3))
    # Everything except moving sources up and the result down: host planning, table writes, encoding, the passes to completion.
    row["exec_ms"] = round(row["total_ms"] - row["upload_ms"] - row["readback_ms"], 3)
    return out, row


def chain_roundtrip(backend, item):
    from nodebased import gpu2d
    p = item.params
    before = backend.counters.snapshot()
    stamps = []

    def one(kind, key, images):
        t = time.perf_counter()
        out = backend.run(backend.stage(kind, p[key], images))
        stamps.append(round((time.perf_counter() - t) * 1000, 3))
        return out
    t0 = time.perf_counter()
    grade = one("Grade", "grade", [item.plate, None])
    transform = one("Transform", "transform", [grade])
    blur = one("Blur", "blur", [transform, None])
    merge = one("Merge", "merge", [blur, item.background, item.matte])
    total = (time.perf_counter() - t0) * 1000
    row = pick(gpu2d.Counters.delta(backend.counters.snapshot(), before))
    row.update(frame=item.frame, grade_ms=stamps[0], transform_ms=stamps[1], blur_ms=stamps[2], merge_ms=stamps[3], total_ms=round(total, 3))
    return merge, row


def pixel_row(ck, got, truth):
    err = ck.compare_arrays(got.pixels, truth.pixels)
    err["windows_match"] = bool(got.data == truth.data and got.display == truth.display)
    return err


def pixel_summary(rows):
    errors = [r for r in rows if "max_abs" in r]
    return {"frames": len(rows), "channels": sum(r.get("channels", 0) for r in rows),
            "max_abs": max((r["max_abs"] for r in errors), default=None),
            "mean_abs": float(np.mean([r["mean_abs"] for r in errors])) if errors else None,
            "p99_abs": max((r["p99_abs"] for r in errors), default=None),
            "max_gate_ratio": max((r["max_gate_ratio"] for r in errors), default=None),
            "violations": sum(r["violations"] for r in errors),
            "windows_match": all(r["windows_match"] for r in rows), "nonfinite_match": all(r["nonfinite_match"] for r in rows)}


def summarize_kernel_modes(modes):
    out = {}
    for name, rows in modes.items():
        if not rows:
            continue
        numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "frame"]
        out[name] = {k: stats([r[k] for r in rows]) for k in numeric if k.endswith("_ms")}
        for k in ("uploads", "upload_bytes", "upload_cache_hits", "downloads", "download_bytes", "intermediate_readbacks", "submissions",
                  "dispatches", "allocations", "pool_hits", "nonfinite_fallbacks"):
            if k in rows[0]:
                values = sorted({r[k] for r in rows})
                out[name][k] = values[0] if len(values) == 1 else {"min": values[0], "max": values[-1], "mean": round(float(np.mean([r[k] for r in rows])), 2)}
        if "cpu_boundaries" in rows[0]:
            out[name]["cpu_boundaries"] = sorted({b for r in rows for b in r["cpu_boundaries"]})
    return out


def cmd_kernels(args):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT))
    ck = load_module("check_gpu2d_fixture", TOOLS / "check_gpu2d_fixture.py")
    from nodebased import gpu2d
    from nodebased.imaging import Evaluator
    width, height = parse_size(args.size)
    frames = parse_frames(args.frames)
    inputs = pathlib.Path(args.inputs).resolve()
    doc = ck.fixture_document(inputs / "plate.%04d.exr", inputs / "background.%04d.exr", width, height, last=max(frames + [args.last or 0]) or None)
    backend = open_backend(args.adapter, args.budget_mb)
    cold_ms = backend.warm()
    cold_counters = backend.counters.snapshot()
    started = time.perf_counter()
    evaluator = Evaluator()
    items = []
    for f in frames:
        items.append(Frame(doc, evaluator, f))
    predecode_s = time.perf_counter() - started
    evaluator.clear()
    print(f"predecoded {len(items)} frames ({sum(i.nbytes() for i in items) / 1e9:.2f} GB) in {predecode_s:.1f} s; cold shader setup {cold_ms:.1f} ms", file=sys.stderr)

    # -- sacrificial warm-up: pipelines, buffer pool and driver caches; nothing from it is reported
    for item in items[:args.warmup]:
        chain_cpu(item)
        chain_resident(backend, item, keyed=False)
        chain_roundtrip(backend, item)
    backend.release_all()

    # -- pixel check: roundtrip and resident against the CPU chain, and the CPU chain against the real evaluator
    pixels = {"resident": [], "roundtrip": []}
    truth_check = []
    for index, item in enumerate(items):
        truth = chain_cpu(item)
        if index < args.truth_frames:
            reference = Evaluator().evaluate_raster(doc, "merge", frame=item.frame)
            truth_check.append(bool(np.array_equal(reference.pixels, truth.pixels) and reference.data == truth.data and reference.display == truth.display))
        out, _ = chain_resident(backend, item, keyed=False)
        pixels["resident"].append(pixel_row(ck, out, truth))
        out, _ = chain_roundtrip(backend, item)
        pixels["roundtrip"].append(pixel_row(ck, out, truth))
        del truth, out
    backend.release_all()

    modes = {"cpu": [], "roundtrip": [], "resident_fresh": [], "resident_warm": []}
    order = ["cpu", "roundtrip", "resident_fresh", "resident_warm"]
    before_all = backend.counters.snapshot()
    for mode in order:
        for item in items:
            if mode == "cpu":
                times = {}
                t = time.perf_counter()
                chain_cpu(item, times)
                times["total_ms"] = (time.perf_counter() - t) * 1000
                times["frame"] = item.frame
                modes[mode].append({k: round(v, 3) if isinstance(v, float) else v for k, v in times.items()})
            elif mode == "roundtrip":
                modes[mode].append(chain_roundtrip(backend, item)[1])
            elif mode == "resident_fresh":
                modes[mode].append(chain_resident(backend, item, keyed=False)[1])
            else:
                chain_resident(backend, item, keyed=True)              # uploads the sources (not reported)
                for _ in range(args.reps):
                    modes[mode].append(chain_resident(backend, item, keyed=True)[1])
    counters_total = gpu2d.Counters.delta(backend.counters.snapshot(), before_all)
    end = backend.counters.snapshot()
    report = {
        "tool": "tools/benchmark_gpu2d.py kernels", "label": args.label, "adapter_choice": args.adapter, "size": [width, height],
        "frames": frames, "reps_warm": args.reps, "budget_bytes": backend.budget,
        "environment": environment(args, backend),
        "predecoded": {"seconds": round(predecode_s, 2), "bytes": sum(i.nbytes() for i in items),
                       "note": "plate, background and Roto matte decoded once into RAM; excluded from every kernel timing"},
        "cold_shader_setup_ms": round(cold_ms, 3), "cold_counters": {k: cold_counters[k] for k in ("pipeline_compiles", "shader_compile_ms")},
        "cpu_reference_is_evaluator": {"frames_checked": len(truth_check), "bit_identical": all(truth_check)},
        "pixels": {name: pixel_summary(rows) for name, rows in pixels.items()},
        "pixel_gate_rule": f"every finite channel abs(gpu-cpu) <= {ck.TOLERANCE_ABS} + {ck.TOLERANCE_REL}*abs(cpu)",
        "summary": summarize_kernel_modes(modes),
        "memory": {"peak_bytes": end["peak_bytes"], "live_bytes_at_end": end["live_bytes"], "budget_bytes": backend.budget,
                   "within_budget": end["peak_bytes"] <= backend.budget},
        "counters_timed_section": pick(counters_total),
        "samples": modes,
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    brief = {name: {"total_p50": s["total_ms"]["p50"], "total_p95": s["total_ms"]["p95"]} for name, s in report["summary"].items()}
    brief["resident_warm_exec_p95"] = report["summary"]["resident_warm"]["exec_ms"]["p95"]
    print(json.dumps({"label": args.label, "pixels": {k: (v["max_gate_ratio"], v["violations"]) for k, v in report["pixels"].items()},
                      "timings": brief, "peak_bytes": end["peak_bytes"]}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------
# graph

def cmd_graph(args):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["NODEBASED_DISK_CACHE"] = "0"
    sys.path.insert(0, str(ROOT))
    ck = load_module("check_gpu2d_fixture", TOOLS / "check_gpu2d_fixture.py")
    from nodebased import gpu2d
    from nodebased.imaging import Evaluator
    width, height = parse_size(args.size)
    frames = parse_frames(args.frames)
    inputs = pathlib.Path(args.inputs).resolve()
    doc = ck.fixture_document(inputs / "plate.%04d.exr", inputs / "background.%04d.exr", width, height, last=max(frames))
    backend = cold = None
    if args.backend == "gpu2d":
        backend = open_backend(args.adapter, args.budget_mb)
        cold = backend.warm()
    evaluator = Evaluator()
    evaluator.gpu2d = backend
    if getattr(args, "source_workers", 0):        # independent sources of a frame on worker threads (nodebased/sourcepool.py)
        from nodebased.sourcepool import SourcePool
        evaluator.source_pool = SourcePool(max_workers=args.source_workers)
    for f in range(1, 1 + args.warmup):
        evaluator.evaluate_raster(doc, "colorcorrect", frame=f)
    evaluator.clear()
    if backend is not None:
        backend.release_all()
    # Per-node cost: time every kernel call by node kind (the two Reads add up), and the island's staging and run.
    spent = {}
    spent_lock = threading.Lock()

    def timed(label, function):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                with spent_lock:
                    spent[label] = spent.get(label, 0.0) + (time.perf_counter() - t0) * 1000
        return wrapper
    original_kernel = Evaluator.__dict__["_windowed_kernel"].__func__
    Evaluator._windowed_kernel = staticmethod(lambda kind, *a, **kw: timed(kind, original_kernel)(kind, *a, **kw))
    if backend is not None:
        backend.stage = timed("island staging", backend.stage)
        backend.run = timed("island run (upload, passes, readback)", backend.run)
    rows = []
    try:
        for f in frames:
            before = backend.counters.snapshot() if backend else None
            spent.clear()
            t = time.perf_counter()
            out = evaluator.evaluate_raster(doc, "colorcorrect", frame=f)
            wall = (time.perf_counter() - t) * 1000
            row = {"frame": f, "wall_ms": round(wall, 3), "nodes_ms": {k: round(v, 3) for k, v in spent.items()},
                   "finite": bool(np.isfinite(out.pixels).all())}
            if backend:
                row["island"] = pick(gpu2d.Counters.delta(backend.counters.snapshot(), before))
            rows.append(row)
    finally:
        Evaluator._windowed_kernel = staticmethod(original_kernel)
    keys = sorted({k for r in rows for k in r["nodes_ms"]})
    report = {"tool": "tools/benchmark_gpu2d.py graph", "label": args.label, "backend": args.backend, "adapter_choice": args.adapter,
              "source_workers": getattr(args, "source_workers", 0), "size": [width, height], "frames": frames, "environment": environment(args, backend),
              "cold_shader_setup_ms": None if cold is None else round(cold, 3),
              "note": "evaluation of the whole fixture to ColorCorrect (source_workers 0 is serial: Read decodes inline, both Reads add up under Read; "
                      "above 0 the sources run on worker threads and the per-kind rows are busy time summed over threads while wall_ms stays the frame), no window and no display "
                      "conversion; kernel_ms_by_node_kind times every CPU kernel call by node kind, and under the island the island rows hold the staging and "
                      "the run (upload, passes, terminal readback) in place of the Grade, Transform, Blur and Merge kernels",
              "wall_ms": stats([r["wall_ms"] for r in rows]),
              "kernel_ms_by_node_kind": {k: stats([r["nodes_ms"].get(k, 0.0) for r in rows]) for k in keys},
              "samples": rows}
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"label": args.label, "backend": args.backend, "wall_ms": report["wall_ms"],
                      "node_p50": {k: v["p50"] for k, v in report["kernel_ms_by_node_kind"].items()}}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------
# interaction

def cmd_interaction(args):
    """Interaction latency of the realistic comp through `Evaluator` (CPU nodes, no window): a Grade knob drag with the
    sources cached, a scrub to frames nobody has computed, and a Roto point drag. `--source-workers N` installs the source pool."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["NODEBASED_DISK_CACHE"] = "0"
    sys.path.insert(0, str(ROOT))
    ck = load_module("check_gpu2d_fixture", TOOLS / "check_gpu2d_fixture.py")
    from nodebased.imaging import Evaluator
    width, height = parse_size(args.size)
    inputs = pathlib.Path(args.inputs).resolve()
    base = ck.fixture_document(inputs / "plate.%04d.exr", inputs / "background.%04d.exr", width, height)

    backend = None
    if getattr(args, "backend", "cpu") == "gpu2d":      # the GPU nodes: the island on the chosen adapter (--adapter default|discrete|integrated|cpu)
        backend = open_backend(args.adapter, args.budget_mb)
        backend.warm()

    def fresh():
        evaluator = Evaluator()
        evaluator.gpu2d = backend
        if args.source_workers:
            from nodebased.sourcepool import SourcePool
            evaluator.source_pool = SourcePool(max_workers=args.source_workers)
        return evaluator

    def timed(evaluator, doc, frame):
        t = time.perf_counter()
        out = evaluator.evaluate_raster(doc, "colorcorrect", frame=frame)
        return (time.perf_counter() - t) * 1000, bool(np.isfinite(out.pixels).all())

    results = {}
    # knob drag: every sample is a new Grade multiply at one frame (the exposure is animated in the fixture), so the Grade chain recomputes and the sources are cache hits
    evaluator = fresh()
    doc = copy.deepcopy(base)
    timed(evaluator, doc, 5)
    rows = []
    for i in range(args.samples):
        doc["nodes"]["grade"]["params"]["multiply"] = 1.0 + 0.002 * (i + 1)
        rows.append(timed(evaluator, doc, 5))
    results["knob drag (Grade multiply, sources cached)"] = rows
    # Roto point drag: a new point position per sample, the Reads cached
    evaluator = fresh()
    doc = copy.deepcopy(base)
    timed(evaluator, doc, 5)
    rows = []
    point = doc["node_data"]["roto"]["shapes"][0]["points"][0]
    start_y = float(point["y"]["value"] if isinstance(point["y"], dict) else point["y"])
    for i in range(args.samples):
        point["y"] = start_y + 0.5 * (i + 1)
        rows.append(timed(evaluator, doc, 5))
    results["Roto point drag (Reads cached)"] = rows
    # scrub: a frame order that never repeats, so every sample decodes two Reads and draws a matte
    rng = random.Random(7)
    order = list(range(1, 49))
    rng.shuffle(order)
    evaluator = fresh()
    timed(evaluator, base, order[0])           # one frame to warm the process, then nothing is cached
    evaluator.clear()
    rows = []
    for frame in order[1:1 + args.samples]:
        rows.append(timed(evaluator, base, frame))
    results["scrub (new frame, nothing cached)"] = rows
    report = {"tool": "tools/benchmark_gpu2d.py interaction", "label": args.label, "source_workers": args.source_workers,
              "backend": getattr(args, "backend", "cpu"), "adapter_choice": args.adapter,
              "size": [width, height], "samples": args.samples, "environment": environment(args, backend),
              "note": "Evaluator to ColorCorrect (CPU nodes, or the island with --backend gpu2d), no window; the knob and point drags are cache misses by construction "
                      "(a new value for every sample), the scrub draws frames in a fixed shuffled order",
              "interactions": {name: {"ms": stats([r[0] for r in rows]), "finite": all(r[1] for r in rows),
                                      "samples": [round(r[0], 3) for r in rows]} for name, rows in results.items()}}
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({name: {"p50": v["ms"]["p50"], "p95": v["ms"]["p95"]} for name, v in report["interactions"].items()}, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------
# playback

def join_islands(timeline, limit=None):
    """Assign every island run to the accepted fresh frame whose node evaluation contained it. ``limit`` keeps the
    first N accepted frames only (the first traversal of the clip; later loops are not the gate)."""
    islands = (timeline.get("gpu2d") or {}).get("islands") or []
    offset = timeline.get("trace_offset_ms")
    frames = sorted((f for f in timeline["fresh_frame_timeline"]["frames"] if "shown" in f["points_ms"]), key=lambda f: f["points_ms"]["shown"])
    if limit:
        frames = frames[:limit]
    if offset is None or not islands:
        return []
    rows = []
    for fr in frames:
        begin, end = fr["points_ms"].get("eval_begin"), fr["points_ms"].get("eval_end")
        if begin is None or end is None:
            continue
        inside = [i for i in islands if begin + offset - 1 <= i["t_ms"] <= end + offset + 1]
        if len(inside) != 1:
            continue
        i = inside[0]
        evaluation = fr["stages_ms"].get("node evaluation")
        rows.append({"node_evaluation_ms": evaluation, "island_wall_ms": i["wall_ms"], "upload_ms": i["upload_ms"],
                     "encode_ms": i["encode_ms"], "gpu_ms": i["gpu_ms"], "readback_ms": i["readback_ms"],
                     "outside_island_ms": round(evaluation - i["wall_ms"], 3) if evaluation is not None else None,
                     "interval_ms": fr.get("interval_ms")})
    return rows


def summarize_playback_pass(path):
    summ = load_module("summarize_2d_playback", TOOLS / "summarize_2d_playback.py")
    data = summ.load(path)
    run = summ.summarize_run(data)
    joined = join_islands(data, data.get("first_pass_frames") or None)
    if joined:
        run["accounting"] = {"frames_joined": len(joined),
                             "island_share_of_node_evaluation": round(float(np.sum([r["island_wall_ms"] for r in joined]) /
                                                                            np.sum([r["node_evaluation_ms"] for r in joined])), 4),
                             **{k: stats([r[k] for r in joined]) for k in ("node_evaluation_ms", "island_wall_ms", "upload_ms",
                                                                          "encode_ms", "gpu_ms", "readback_ms", "outside_island_ms")}}
    run["accepted_frames_over_total"] = [run["fresh_frames_traced"], run["frames_in_first_pass"]]
    return run


def cmd_playback(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = pathlib.Path(args.inputs).resolve()
    summary = {"tool": "tools/benchmark_gpu2d.py playback", "label": args.label, "backend": args.backend, "frame_range": args.frame_range,
               "force_adapter": args.force_adapter, "cpus": args.cpus, "env": args.env, "passes": []}
    for n in range(1, args.passes + 1):
        case = f"{args.backend}-{args.frame_range}-{args.label}-{n}"
        raw = out_dir / f"{case}.json"
        timeline = out_dir / f"{case}.timeline.json"
        cmd = [sys.executable, str(TOOLS / "benchmark_2d_playback.py"), "--plate-pattern", str(inputs / "plate.%04d.exr"),
               "--background-pattern", str(inputs / "background.%04d.exr"), "--size", "1080p", "--cache-mode", "uncached",
               "--display", args.display, "--force-adapter", args.force_adapter, "--adapter", args.label, "--backend", args.backend,
               "--gpu-budget-mb", str(args.budget_mb), "--seconds", str(args.seconds), "--case", case, "--output", str(raw),
               "--timeline-output", str(timeline)]
        if args.frame_range != 48:
            cmd += ["--frame-range", str(args.frame_range)]
        if getattr(args, "leaf_sampler", False):
            cmd += ["--leaf-sampler"]
        if getattr(args, "no_sampler", False):
            cmd += ["--no-sampler"]
        if getattr(args, "stages", False):
            cmd += ["--stage-trace", str(out_dir / f"{case}.stages.json")]
        if args.cpus:                  # the laptop stand-in: four CPUs (pair with --env QT_SCALE_FACTOR=2)
            cmd = ["taskset", "-c", args.cpus] + cmd
        env = {**os.environ, "PYTHONPATH": str(ROOT), "DISPLAY": args.display, "NODEBASED_DISK_CACHE": "0"}
        env.update(item.split("=", 1) for item in args.env)
        started = time.perf_counter()
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
        if done.returncode != 0 or not timeline.exists():
            summary["passes"].append({"case": case, "error": (done.stderr or done.stdout)[-1500:]})
            print(f"{case}: FAILED\n{(done.stderr or done.stdout)[-600:]}", file=sys.stderr)
            continue
        run = summarize_playback_pass(timeline)
        run["wall_seconds"] = round(time.perf_counter() - started, 1)
        summary["passes"].append(run)
        with open(timeline, "rb") as src, gzip.open(str(timeline) + ".gz", "wb") as dst:
            dst.write(src.read())
        print(f"{case}: {run['first_pass_fps']} fps, interval p95 {run['interval_ms']['p95']} ms, node evaluation p50 "
              f"{run['node_evaluation_ms']['p50']} ms, accepted {run['accepted_frames_over_total']}", file=sys.stderr)
    good = [p for p in summary["passes"] if "error" not in p]
    summary["median"] = {
        "first_pass_fps": round(statistics.median([p["first_pass_fps"] for p in good]), 3) if good else None,
        "interval_p95_ms": round(statistics.median([p["interval_ms"]["p95"] for p in good]), 3) if good else None,
        "node_evaluation_p50_ms": round(statistics.median([p["node_evaluation_ms"]["p50"] for p in good]), 3) if good else None}
    (out_dir / f"summary-{args.backend}-{args.frame_range}-{args.label}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["median"]))
    return 0 if len(good) == args.passes else 1


def exclusive_times(spans):
    """Per-label exclusive time of one thread's spans (each span's own time less the spans nested inside it)."""
    spans = sorted(spans, key=lambda s: (s[2], -s[3]))
    totals, stack = {}, []        # stack of [label, end, child_time]
    def close(entry, end):
        label, _, children, begin = entry
        totals[label] = totals.get(label, 0.0) + (end - begin) - children
    for _, label, begin, end in spans:
        while stack and stack[-1][1] <= begin + 1e-9:
            done = stack.pop()
            close(done, done[1])
            if stack:
                stack[-1][2] += done[1] - done[3]
        stack.append([label, end, 0.0, begin])
    while stack:
        done = stack.pop()
        close(done, done[1])
        if stack:
            stack[-1][2] += done[1] - done[3]
    return totals


def stage_rows(stages_path, timeline_path):
    """One pass: per accepted fresh frame, where the evaluating thread's node stage went (exclusive ms per stage, the
    island run split by its counters) and what the other threads did meanwhile (busy ms inside the node stage)."""
    st = json.loads(pathlib.Path(stages_path).read_text())
    tl = json.loads(pathlib.Path(timeline_path).read_text())
    limit = tl.get("first_pass_frames") or None
    frames = sorted((f for f in st["frames"] if "shown" in f["points_ms"] and "eval_begin" in f["points_ms"] and "eval_end" in f["points_ms"]),
                    key=lambda f: f["points_ms"]["shown"])
    if limit:
        frames = frames[:limit]
    by_thread = {}
    for span in st["spans"]:
        by_thread.setdefault(span[0], []).append(span)
    islands = st.get("islands") or []
    rows = []
    for f in frames:
        begin, end = f["points_ms"]["eval_begin"], f["points_ms"]["eval_end"]
        top = None
        for thread, spans in by_thread.items():
            if not thread.startswith("nodebased-preview"):
                continue
            for s in spans:
                if s[1].startswith("walk:") and s[2] >= begin - 1.5 and s[3] <= end + 1.5 and (top is None or s[3] - s[2] > top[3] - top[2]):
                    top = s
        if top is None:
            continue
        evaluator = [s for s in by_thread[top[0]] if top[2] - 1e-6 <= s[2] and s[3] <= top[3] + 1e-6]
        excl = exclusive_times(evaluator)
        row = {"node_stage_ms": end - begin, "walk_ms": top[3] - top[2], "stages": {}, "other_threads": {}}
        run_total = excl.pop("island run", 0.0)
        island = next((i for i in islands if top[2] - 1 <= i["t_ms"] <= top[3] + 1), None)
        if island is not None and run_total:
            parts = {"island upload (host to card)": island["upload_ms"], "island encode (host)": island["encode_ms"],
                     "island kernels on the card (submit to done)": island["gpu_ms"],
                     "island download: copy and map": island["readback_map_ms"], "island download: host copy into the picture": island["readback_host_ms"]}
            row["stages"].update(parts)
            row["stages"]["island run: other host work (blob, submit, buffers)"] = max(0.0, run_total - sum(parts.values()))
        elif run_total:
            row["stages"]["island run (not split)"] = run_total
        row["stages"].update(excl)
        for thread, spans in by_thread.items():
            if thread == top[0]:
                continue
            clipped = [(t, l, max(a, begin), min(b, end)) for t, l, a, b in spans if b > begin and a < end]
            for label, value in exclusive_times(clipped).items():
                if value > 0:
                    row["other_threads"][f"{thread.rsplit('-', 1)[0]}: {label}"] = row["other_threads"].get(f"{thread.rsplit('-', 1)[0]}: {label}", 0.0) + value
        rows.append(row)
    return rows, st


def cmd_stages(args):
    """Average the stage files of the passes under --dir for one case prefix (<backend>-<range>-<label>-<n>)."""
    directory = pathlib.Path(args.dir)
    passes = sorted(directory.glob(f"{args.prefix}-[0-9].stages.json"))
    if not passes:
        raise SystemExit(f"no {args.prefix}-N.stages.json under {directory}")
    per_pass, all_stage, all_other, profiles = [], {}, {}, []
    for path in passes:
        timeline = path.with_name(path.name.replace(".stages.json", ".timeline.json"))
        rows, st = stage_rows(path, timeline)
        if not rows:
            continue
        n = len(rows)
        stage_means = {}
        for r in rows:
            for k, v in r["stages"].items():
                stage_means[k] = stage_means.get(k, 0.0) + v / n
        other_means = {}
        for r in rows:
            for k, v in r["other_threads"].items():
                other_means[k] = other_means.get(k, 0.0) + v / n
        per_pass.append({"case": path.name.replace(".stages.json", ""), "frames": n,
                         "node_stage_ms": round(sum(r["node_stage_ms"] for r in rows) / n, 3),
                         "walk_ms": round(sum(r["walk_ms"] for r in rows) / n, 3),
                         "stages_ms": {k: round(v, 3) for k, v in sorted(stage_means.items(), key=lambda kv: -kv[1])},
                         "other_threads_busy_ms": {k: round(v, 3) for k, v in sorted(other_means.items(), key=lambda kv: -kv[1])}})
        for k, v in stage_means.items():
            all_stage.setdefault(k, []).append(v)
        for k, v in other_means.items():
            all_other.setdefault(k, []).append(v)
        profiles.append(st.get("sampling_profile", {}).get("worker", {}))
    passes_n = len(per_pass)
    def mean_rows(source):
        out = {}
        for k, values in source.items():
            values = values + [0.0] * (passes_n - len(values))
            out[k] = {"mean_ms": round(float(np.mean(values)), 3), "min_ms": round(min(values), 3), "max_ms": round(max(values), 3)}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["mean_ms"]))
    merged = {}
    for prof in profiles:
        for k, v in prof.items():
            merged[k] = merged.get(k, 0) + v
    walk_samples = max(merged.get("imaging.py:evaluate_raster", 0), 1)
    top_python = [{"function": k, "share_of_walk_samples": round(v / walk_samples, 3)} for k, v in sorted(merged.items(), key=lambda kv: -kv[1])[:25]]
    report = {"tool": "tools/benchmark_gpu2d.py stages", "prefix": args.prefix, "passes": per_pass,
              "node_stage_ms": {"mean": round(float(np.mean([p["node_stage_ms"] for p in per_pass])), 3),
                                "min": min(p["node_stage_ms"] for p in per_pass), "max": max(p["node_stage_ms"] for p in per_pass)},
              "stages_ms": mean_rows(all_stage), "other_threads_busy_ms": mean_rows(all_other), "worker_thread_top_functions": top_python,
              "note": "exclusive milliseconds per accepted fresh frame on the thread that evaluates it, averaged over the frames of a pass and then over the passes; "
                      "'walk: ...' is the evaluator's own time between its nodes (cache keys, digests, windows, Python); the island run is split by its counters; "
                      "other_threads_busy_ms is what the source and decode workers did inside the same node stage (it overlaps, it is not added)"}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"node_stage_ms": report["node_stage_ms"], "top": list(report["stages_ms"].items())[:10]}, indent=1))
    return 0


def cmd_resummarize(args):
    """Rebuild the per-group summary from saved ``*.timeline.json`` files (``<backend>-<range>-<label>-<n>``)."""
    directory = pathlib.Path(args.dir)
    groups = {}
    for path in sorted(directory.glob("*.timeline.json")):
        stem = path.name[:-len(".timeline.json")]
        groups.setdefault(stem.rsplit("-", 1)[0], []).append(path)
    for group, paths in groups.items():
        backend, frame_range, label = group.split("-", 2)
        passes = [summarize_playback_pass(p) for p in paths]
        summary = {"tool": "tools/benchmark_gpu2d.py playback", "label": label, "backend": backend, "frame_range": int(frame_range), "passes": passes,
                   "median": {"first_pass_fps": round(statistics.median([p["first_pass_fps"] for p in passes]), 3),
                              "interval_p95_ms": round(statistics.median([p["interval_ms"]["p95"] for p in passes]), 3),
                              "node_evaluation_p50_ms": round(statistics.median([p["node_evaluation_ms"]["p50"] for p in passes]), 3)}}
        (directory / f"summary-{group}.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(group, summary["median"])
    return 0


def cmd_table(args):
    """Combine the kernel passes of every tier into one compact summary: per pass and pooled statistics for each mode."""
    tiers = {}
    for path in args.kernels:
        report = json.loads(pathlib.Path(path).read_text())
        tiers.setdefault(report["label"], []).append(report)
    out = {}
    for label, reports in tiers.items():
        row = {"device": reports[0]["environment"]["gpu"]["device"], "adapter_type": reports[0]["environment"]["gpu"]["adapter_type"],
               "size": reports[0]["size"], "frames": len(reports[0]["frames"]), "budget_mib": reports[0]["budget_bytes"] >> 20, "passes": len(reports),
               "cold_shader_setup_ms": [r["cold_shader_setup_ms"] for r in reports],
               "pixels_worst_gate_ratio": max(max(r["pixels"][m]["max_gate_ratio"] for m in ("resident", "roundtrip")) for r in reports),
               "pixel_violations": sum(r["pixels"][m]["violations"] for r in reports for m in ("resident", "roundtrip")),
               "peak_bytes": max(r["memory"]["peak_bytes"] for r in reports), "modes": {}}
        for mode in ("cpu", "roundtrip", "resident_fresh", "resident_warm"):
            samples = [[x for x in r["samples"][mode]] for r in reports]
            metrics = [k for k in samples[0][0] if k.endswith("_ms")]
            row["modes"][mode] = {k: {"pooled": stats([x[k] for xs in samples for x in xs]),
                                      "per_pass_p50": [stats([x[k] for x in xs])["p50"] for xs in samples],
                                      "per_pass_p95": [stats([x[k] for x in xs])["p95"] for xs in samples]} for k in metrics}
        counters = reports[0]["summary"]
        row["counters_per_frame"] = {m: {k: counters[m][k] for k in ("uploads", "upload_bytes", "downloads", "download_bytes", "intermediate_readbacks",
                                                                     "submissions", "dispatches", "allocations", "pool_hits", "upload_cache_hits", "cpu_boundaries")}
                                     for m in ("roundtrip", "resident_fresh", "resident_warm")}
        out[label] = row
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    lines = ["| Tier | Mode | Total p50 | Total p95 | Per-pass p95 |", "|---|---|---:|---:|---|"]
    for label, row in out.items():
        for mode, metrics in row["modes"].items():
            total = metrics["total_ms"]
            lines.append(f"| {label} | {mode} | {total['pooled']['p50']} | {total['pooled']['p95']} | {total['per_pass_p95']} |")
        lines.append(f"| {label} | resident_fresh exec | {row['modes']['resident_fresh']['exec_ms']['pooled']['p50']} | "
                     f"{row['modes']['resident_fresh']['exec_ms']['pooled']['p95']} | {row['modes']['resident_fresh']['exec_ms']['per_pass_p95']} |")
        lines.append(f"| {label} | resident_warm exec | {row['modes']['resident_warm']['exec_ms']['pooled']['p50']} | "
                     f"{row['modes']['resident_warm']['exec_ms']['pooled']['p95']} | {row['modes']['resident_warm']['exec_ms']['per_pass_p95']} |")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------------------------------------
# gates

def verdict(ok):
    return "pass" if ok else "fail"


def cmd_gates(args):
    kernels = [json.loads(pathlib.Path(p).read_text()) for p in args.kernels]
    table = []

    def add(gate, limit, value, ok, note=""):
        table.append({"gate": gate, "limit": limit, "measured": value, "result": ok if isinstance(ok, str) else verdict(ok), "note": note})
    resident = [k["summary"]["resident_warm"] for k in kernels]
    fresh = [k["summary"]["resident_fresh"] for k in kernels]
    cpu = [k["summary"]["cpu"] for k in kernels]
    pixel_ok = all(k["pixels"][m]["violations"] == 0 and k["pixels"][m]["windows_match"] and k["pixels"][m]["nonfinite_match"]
                   for k in kernels for m in ("resident", "roundtrip"))
    worst_ratio = max(k["pixels"][m]["max_gate_ratio"] for k in kernels for m in ("resident", "roundtrip"))
    add("Pixels inside 2e-5 + 2e-5*abs(cpu)", "ratio <= 1", f"worst ratio {worst_ratio:.4f}, violations 0" if pixel_ok else "violations", pixel_ok)
    readbacks = sum(k["summary"]["resident_fresh"]["intermediate_readbacks"] if isinstance(k["summary"]["resident_fresh"]["intermediate_readbacks"], int)
                    else k["summary"]["resident_fresh"]["intermediate_readbacks"]["max"] for k in kernels)
    boundaries = sorted({b for k in kernels for b in k["summary"]["resident_fresh"]["cpu_boundaries"]})
    add("Zero readbacks between the four kernels, all four on the device", "0", f"{readbacks} intermediate readbacks; boundaries {boundaries or 'none'}", readbacks == 0 and not boundaries)
    add("Allocations within the cap", f"peak <= {kernels[0]['budget_bytes'] >> 20} MiB", f"peak {max(k['memory']['peak_bytes'] for k in kernels) / 2**20:.0f} MiB",
        all(k["memory"]["within_budget"] for k in kernels))
    exec_p95 = [r["exec_ms"]["p95"] for r in resident]
    fresh_exec_p95 = [r["exec_ms"]["p95"] for r in fresh]
    add("Four-node resident execution p95", f"<= {GATES['kernel_p95_ms']} ms",
        f"sources already on the device {exec_p95} ms; sources uploaded that frame, upload and readback time left out {fresh_exec_p95} ms (passes)",
        max(exec_p95 + fresh_exec_p95) <= GATES["kernel_p95_ms"],
        "stage + encode + passes to completion; both variants must pass")
    total_p95 = [r["total_ms"]["p95"] for r in fresh]
    add("Uploads + resident execution + terminal readback p95", f"<= {GATES['transfer_p95_ms']} ms", f"{total_p95} ms (passes)", max(total_p95) <= GATES["transfer_p95_ms"],
        "completion included")
    return table


def markdown(table):
    lines = ["| Gate | Limit | Measured | Result |", "|---|---|---|---|"]
    for row in table:
        lines.append(f"| {row['gate']} | {row['limit']} | {row['measured']} | {row['result']} |")
    return "\n".join(lines)


def cmd_gates_full(args):
    table = cmd_gates(args)
    tier = args.tier
    pairs = list(zip(args.playback or [], args.cpu_playback or []))
    for gpu_path, cpu_path in pairs:
        gpu = json.loads(pathlib.Path(gpu_path).read_text())
        cpu = json.loads(pathlib.Path(cpu_path).read_text())
        good = [p for p in gpu["passes"] if "error" not in p]
        base = [p for p in cpu["passes"] if "error" not in p]
        rng = gpu["frame_range"]
        cpu_node = statistics.median([p["node_evaluation_ms"]["p50"] for p in base])
        gpu_node = statistics.median([p["node_evaluation_ms"]["p50"] for p in good])
        ratio = cpu_node / gpu_node
        accepted = [p["accepted_frames_over_total"] for p in good]
        table.append({"gate": f"Realistic graph node-stage median vs the same machine's CPU run ({rng}-frame clip)", "limit": ">= 2x",
                      "measured": f"CPU {cpu_node:.1f} ms -> island {gpu_node:.1f} ms = {ratio:.2f}x (passes: {[round(p['node_evaluation_ms']['p50'], 1) for p in good]})",
                      "result": verdict(ratio >= GATES["node_stage_speedup"]), "note": f"accepted fresh frames per pass {accepted}"})
        fps = [p["first_pass_fps"] for p in good]
        p95 = [p["interval_ms"]["p95"] for p in good]
        inter = GATES["intermediate_fps"].get(tier, GATES["intermediate_fps"]["rtx3080ti"])
        inter_p95 = GATES["intermediate_p95_ms"].get(tier, GATES["intermediate_p95_ms"]["rtx3080ti"])
        table.append({"gate": f"Intermediate: fresh playback fps and interval p95, {rng}-frame clip", "limit": f">= {inter} fps and p95 <= {inter_p95} ms",
                      "measured": f"{fps} fps, p95 {p95} ms (passes); CPU run {[p['first_pass_fps'] for p in base]} fps",
                      "result": verdict(min(fps) >= inter and max(p95) <= inter_p95), "note": ""})
        table.append({"gate": f"Final: 24 fps, {rng}-frame clip", "limit": f">= {GATES['final_fps']} fps and p95 <= {GATES['final_p95_ms']} ms",
                      "measured": f"{fps} fps, p95 {p95} ms (passes)",
                      "result": verdict(min(fps) >= GATES["final_fps"] and max(p95) <= GATES["final_p95_ms"]), "note": ""})
    print(markdown(table))
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps({"tier": tier, "gates": table}, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------------------------------------
# input generation for small fixtures (the generator is tools/benchmark_2d_playback.py's)

def cmd_make_inputs(args):
    bench = load_module("benchmark_2d_playback", TOOLS / "benchmark_2d_playback.py")
    width, height = parse_size(args.size)
    print(json.dumps(bench.make_inputs(pathlib.Path(args.out_dir), width, height, args.frames)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--inputs", required=True, type=pathlib.Path, help="directory with plate.%%04d.exr and background.%%04d.exr")
        p.add_argument("--size", default="1080p", help="1080p, 4k or WIDTHxHEIGHT matching the generated inputs")
        p.add_argument("--adapter", default="default", choices=("default", "discrete", "integrated", "cpu"))
        p.add_argument("--budget-mb", type=int, default=1024, help="2D device allocation cap")
        p.add_argument("--label", default="run")
        p.add_argument("--output", required=True, type=pathlib.Path)
        p.add_argument("--warmup", type=int, default=2, help="sacrificial frames before measuring")
    k = sub.add_parser("kernels", help="predecoded four-node chain: cpu, per-node round trip, resident")
    common(k)
    k.add_argument("--frames", default="1-48", help="frame spec, e.g. 1-48 or 49-240:8")
    k.add_argument("--last", type=int, help="extend the fixture's frame range (curves hold their last key)")
    k.add_argument("--reps", type=int, default=2, help="timed repeats per frame with the sources already on the device")
    k.add_argument("--truth-frames", type=int, default=3, help="frames on which the CPU chain is checked bit-for-bit against the evaluator")
    k.set_defaults(fn=cmd_kernels)
    g = sub.add_parser("graph", help="the whole graph through Evaluator, no window")
    common(g)
    g.add_argument("--frames", default="1-48")
    g.add_argument("--backend", choices=("cpu", "gpu2d"), default="gpu2d")
    g.add_argument("--source-workers", type=int, default=0, help="worker threads for the independent sources of a frame (0 = serial, the step C method)")
    g.set_defaults(fn=cmd_graph)
    it = sub.add_parser("interaction", help="knob drag, Roto point drag and scrub latency of the realistic comp through the evaluator")
    common(it)
    it.add_argument("--samples", type=int, default=24)
    it.add_argument("--source-workers", type=int, default=0, help="workers of the source pool (0 = none)")
    it.add_argument("--backend", choices=("cpu", "gpu2d"), default="cpu", help="cpu nodes (default) or the GPU nodes (the island) on --adapter")
    it.set_defaults(fn=cmd_interaction)
    pb = sub.add_parser("playback", help="N fresh passes of the real-window playback benchmark, summarized")
    pb.add_argument("--inputs", required=True, type=pathlib.Path)
    pb.add_argument("--display", required=True, help="private X display, e.g. :99")
    pb.add_argument("--backend", choices=("cpu", "gpu2d"), default="gpu2d")
    pb.add_argument("--force-adapter", choices=("default", "integrated", "cpu"), default="default")
    pb.add_argument("--label", required=True, help="hardware label used in the case names")
    pb.add_argument("--frame-range", type=int, default=48)
    pb.add_argument("--passes", type=int, default=3)
    pb.add_argument("--seconds", type=float, default=None)
    pb.add_argument("--budget-mb", type=int, default=1024)
    pb.add_argument("--cpus", help="restrict the run to these CPUs with taskset, e.g. 0-3")
    pb.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="extra environment for the run (repeatable), e.g. QT_SCALE_FACTOR=2")
    pb.add_argument("--leaf-sampler", action="store_true", help="with --stages: also sample the evaluating thread line by line every millisecond")
    pb.add_argument("--no-sampler", action="store_true", help="run without the harness's 2 ms stack sampler (its GIL use slows the playback it measures)")
    pb.add_argument("--stages", action="store_true", help="also write <case>.stages.json: every thread span of the run (see the `stages` command)")
    pb.add_argument("--out-dir", required=True, type=pathlib.Path)
    pb.set_defaults(fn=cmd_playback)
    sg = sub.add_parser("stages", help="where the node stage of a played frame goes, from the --stages files of one case prefix")
    sg.add_argument("--dir", required=True)
    sg.add_argument("--prefix", required=True, help="<backend>-<range>-<label>, e.g. gpu2d-48-radeon8060s-1026c")
    sg.add_argument("--output", required=True)
    sg.set_defaults(fn=cmd_stages)
    tb = sub.add_parser("table", help="combine kernel passes of each tier into one compact summary and a table")
    tb.add_argument("--kernels", nargs="+", required=True)
    tb.add_argument("--output")
    tb.set_defaults(fn=cmd_table)
    rs = sub.add_parser("resummarize", help="rebuild the playback summaries from saved timeline files")
    rs.add_argument("--dir", required=True)
    rs.set_defaults(fn=cmd_resummarize)
    ga = sub.add_parser("gates", help="the go/no-go table")
    ga.add_argument("--tier", required=True, help="rtx3080ti, rtx4070laptop, radeon8060s ...")
    ga.add_argument("--kernels", nargs="+", required=True)
    ga.add_argument("--playback", nargs="*", help="gpu2d playback summaries (summary-gpu2d-<range>-<label>.json)")
    ga.add_argument("--cpu-playback", nargs="*", help="the matching CPU summaries, same order, same machine")
    ga.add_argument("--output")
    ga.set_defaults(fn=cmd_gates_full)
    mi = sub.add_parser("make-inputs", help="generate small EXR sources (the generator is benchmark_2d_playback.make_inputs)")
    mi.add_argument("--out-dir", required=True)
    mi.add_argument("--size", default="480x270")
    mi.add_argument("--frames", type=int, default=12)
    mi.set_defaults(fn=cmd_make_inputs)
    args = parser.parse_args()
    if args.command == "playback" and args.seconds is None:
        args.seconds = 12.0 if args.frame_range <= 48 else 60.0
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
