"""Bake the Hot pour scene end to end and write down what it costs (Lane 6 step N3, docs/SIMULATION.md "Hot pour").

    flock /tmp/nb-gpu.lock python tools/benchmark_hot_pour.py [--adapter default|integrated|cpu]
        [--liquid-cells 256] [--smoke-cells 192] [--frames 120] [--workdir DIR] [--json OUT.json]
        [--playback] [--stills DIR] [--still-frames 60,120] [--still-size 1920x1080] [--still-samples 64]

The graph is `nodebased.hotpour.hot_pour_ops`, the same code that writes the shipped preset. Each fluid is baked a frame at
a time through the evaluator, the way the app's bake does (the liquid first, because the steam reads the liquid's surface
for every frame of the document's range). Recorded per frame: the liquid solve and its surface field, the mesh, the whitewater
and the steam solve (wall seconds); per run: peak resident memory of this process, the GPU memory the app's own tracker
counts (the device's live buffers and textures), the NVIDIA driver's per-process figure when there is one, the cache's
size on disk and what the same sparse volume frames would have taken stored dense. `--playback` then plays the baked range
from the cache the way the viewport does (the cache read, the sparse volume's upload as tiles and the GPU raster draw) and
reports the frame rate with and without the liquid's mesh. `--stills` renders the listed frames with the GPU path tracer.

Nothing here touches the user's cache: every cache lives under `--workdir`.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def force_adapter(kind):
    """Make the app's default adapter `kind` ('integrated' is the AMD iGPU, 'cpu' is llvmpipe); 'default' changes nothing."""
    if kind == "default":
        return
    from nodebased import gpu3d
    original = gpu3d._state
    gpu3d._state = lambda choice=None: original(kind if (choice or "default") == "default" else choice)


def rss_peak_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def rss_now_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def amd_memory_mb():
    """VRAM plus GTT in use on the AMD card(s), from sysfs (None without one)."""
    total, found = 0, False
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        device = card / "device"
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
            total += int((device / "mem_info_vram_used").read_text()) + int((device / "mem_info_gtt_used").read_text())
            found = True
        except (OSError, ValueError):
            continue
    return total / 2 ** 20 if found else None


class DriverGpuSampler(threading.Thread):
    """Peak of what the driver says is in use: the NVIDIA driver's per-process figure, or the AMD card's VRAM and GTT less
    what was in use when the run began (None where there is neither)."""

    def __init__(self, adapter="default"):
        super().__init__(daemon=True)
        self.peak = None
        self._halt = threading.Event()
        self.pid = os.getpid()
        self.adapter = adapter
        self.baseline = amd_memory_mb() if adapter == "integrated" else None

    def run(self):
        while not self._halt.is_set():
            try:
                if self.adapter == "integrated":
                    now = amd_memory_mb()
                    if now is not None and self.baseline is not None:
                        self.peak = max(self.peak or 0, now - self.baseline)
                elif self.adapter == "default":
                    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout
                    for line in out.splitlines():
                        pid, used = (v.strip() for v in line.split(","))
                        if int(pid) == self.pid:
                            self.peak = max(self.peak or 0, int(used))
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self._halt.wait(0.25)

    def stop(self):
        self._halt.set()


def build_document(args):
    from nodebased import hotpour
    from nodebased.core import Dispatcher, validate
    ops = hotpour.hot_pour_ops(args.liquid_cells, args.smoke_cells, args.frames, pressure=args.liquid_pressure,
                               smoke_pressure=args.smoke_pressure, spout_height=args.spout_height)
    plain = []
    for op in ops:
        op = dict(op)
        for field in ("id", "source"):
            if isinstance(op.get(field), str):
                op[field] = op[field].replace("$new:", "")
        plain.append(op)
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": plain})
    dispatcher.execute({"op": "time", "first": 1, "last": args.frames})
    validate(dispatcher.document)
    return dispatcher


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return time.perf_counter() - started, value


def cache_report(root, args):
    """Bytes on disk per run in the cache, classified by what its frames hold, and the dense equivalent of the served volumes."""
    import numpy as np
    runs = {}
    for run_dir in sorted(Path(root).glob("*/*")):
        files = sorted(run_dir.glob("*.npz"))
        if not files:
            continue
        with np.load(files[0], allow_pickle=False) as sample:
            names = set(sample.files)
        total = sum(f.stat().st_size for f in files)
        if "coords" in names:
            kind = "steam volume (sparse tiles)"
        elif "density" in names:
            kind = "steam solver checkpoints"
        elif "position" in names and "radius" in names:
            kind = "whitewater"
        elif "position" in names:
            kind = "liquid particles"
        else:
            kind = "other: " + ",".join(sorted(names))[:60]
        runs[run_dir.name[:10]] = {"kind": kind, "frames": len(files), "bytes": total}
    report = {"runs": runs, "total_bytes": sum(r["bytes"] for r in runs.values())}
    dense_equivalent = 0
    stored = 0
    for run_dir in sorted(Path(root).glob("*/*")):
        files = sorted(run_dir.glob("*.npz"))
        if not files:
            continue
        with np.load(files[0], allow_pickle=False) as sample:
            if "coords" not in sample.files:
                continue
            channels = [n for n in sample.files if n not in ("coords", "__meta__")]
            itemsize = sample[channels[0]].dtype.itemsize
            scalar = {"velocity": 3}
        for f in files:
            with np.load(f, allow_pickle=False) as frame:
                meta = json.loads(str(frame["__meta__"]))
                shape = meta.get("domain_shape")
                if shape:
                    dense_equivalent += int(shape[0]) * int(shape[1]) * int(shape[2]) * itemsize * sum(
                        scalar.get(c, 1) for c in channels)
                stored += f.stat().st_size
    report["steam_sparse_bytes"] = stored
    report["steam_dense_equivalent_bytes"] = dense_equivalent
    return report


def bake(args, document, evaluator, gpu):
    frames = range(1, args.frames + 1)
    rows = {f: {} for f in frames}
    peaks = {}
    tracker = gpu["memory"] if gpu else None

    def stage(name, node, key):
        if tracker is not None:
            tracker.reset_peak()
        started = time.perf_counter()
        for f in frames:
            elapsed, value = timed(evaluator.evaluate_raster, document, node, frame=f, typed=True)
            rows[f][key] = elapsed
            summarise(rows[f], key, value)
            if args.progress and f % 10 == 0:
                print(f"  {key} frame {f}: {elapsed:.2f} s", flush=True)
        peaks[name] = {"seconds": time.perf_counter() - started, "host_rss_peak_mb": rss_peak_mb(),
                       "host_rss_now_mb": rss_now_mb(),
                       "gpu_tracked_peak_mb": None if tracker is None else tracker.peak / 2 ** 20}

    stage("liquid", "liquid_cache", "liquid_s")
    stage("surface", "surface", "surface_s")
    if not args.no_whitewater:
        stage("whitewater", "whitewater", "whitewater_s")
    if not args.no_steam:
        stage("steam", "steam_cache", "steam_s")
    return rows, peaks


def summarise(row, key, value):
    if key == "liquid_s" and hasattr(value, "positions"):
        row["particles"] = int(len(value.positions))
        stream = getattr(value, "stream", None)
        row["liquid_backend"] = getattr(stream, "backend", None)
        row["liquid_fallback"] = getattr(stream, "fallback_reason", None)
    elif key == "surface_s" and hasattr(value, "triangles"):
        row["triangles"] = int(len(value.triangles))
    elif key == "whitewater_s" and hasattr(value, "positions"):
        row["whitewater_particles"] = int(len(value.positions))
    elif key == "steam_s":
        density = getattr(value, "density", None)
        sparse = getattr(value, "sparse", None)
        if density is not None:
            row["steam_shape"] = list(density.shape)
        if sparse is not None:
            row["steam_tiles"] = int(sparse.tile_count)
        stream = getattr(value, "stream", None)
        row["steam_backend"] = getattr(stream, "backend", None)


def percentile_row(values):
    values = sorted(values)
    if not values:
        return None
    return {"mean_s": round(statistics.fmean(values), 3), "median_s": round(statistics.median(values), 3),
            "max_s": round(values[-1], 3), "total_s": round(sum(values), 2)}


def playback(args, document, evaluator, gpu):
    """Frames per second of the baked range played from the cache through the app's GPU viewport."""
    import numpy as np
    from nodebased import viewportgpu
    renderer = viewportgpu.renderer()
    camera = evaluator.evaluate_raster(document, "camera", frame=1, typed=True)
    results = {}
    for label, node in (("steam volume alone", "steam_volume_scene"), ("whole scene", "scene")):
        times = []
        for f in range(1, args.frames + 1):
            started = time.perf_counter()
            scene = evaluator.evaluate_raster(document, node, frame=f, typed=True)
            fetched = time.perf_counter()
            renderer.render(scene, camera, args.play_width, args.play_height, (0.02, 0.02, 0.025, 1.0), ambient=0.2)
            done = time.perf_counter()
            times.append((fetched - started, done - fetched))
        fetch = [t[0] for t in times]
        draw = [t[1] for t in times]
        total = [a + b for a, b in times]
        results[label] = {"fps_median": round(1.0 / statistics.median(total), 2), "fps_mean": round(len(total) / sum(total), 2),
                          "fetch_ms_median": round(statistics.median(fetch) * 1000, 1),
                          "draw_ms_median": round(statistics.median(draw) * 1000, 1)}
    return results


def add_playback_scene(d):
    """A Scene3D holding only the steam volume and the lights, so the playback of the sparse cache can be told from the mesh's."""
    d.execute({"op": "batch", "commands": [
        {"op": "create", "id": "steam_volume_scene", "type": "Scene3D", "params": {}},
        {"op": "connect", "id": "steam_volume_scene", "input": "object0", "source": "steam_cache"},
        {"op": "connect", "id": "steam_volume_scene", "input": "object1", "source": "key_light"}]})


def save_png(pixels, path):
    """Linear RGBA float pixels over a dark slate, sRGB encoded, to a PNG through Qt (no other imaging library needed)."""
    import numpy as np
    from PySide6.QtGui import QImage
    rgb, alpha = np.clip(pixels[..., :3], 0.0, 1.0), pixels[..., 3:4] if pixels.shape[-1] > 3 else 1.0
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    out = srgb + np.array([0.04, 0.045, 0.055]) * (1.0 - alpha)
    rgb8 = np.ascontiguousarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
    height, width = rgb8.shape[:2]
    QImage(rgb8.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy().save(str(path))


def render_stills(args, dispatcher, evaluator):
    """The frames in `--still-frames` through the real Render3D node on the GPU path tracer."""
    out = Path(args.stills)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.still_size.split("x"))
    for name, value in {"width": width, "height": height, "render_mode": "pathtrace", "render_backend": "gpu",
                        "pt_samples": args.still_samples, "max_bounces": 8, "ambient": 0.45, "samples": 1}.items():
        dispatcher.execute({"op": "set", "id": "render", "param": name, "value": value})
    document = dispatcher.document
    results = {}
    for frame in (int(v) for v in args.still_frames.split(",")):
        # warm: the first render also builds the scene and compiles the shaders; the recorded time is the second
        started = time.perf_counter()
        image = evaluator.evaluate_raster(document, "render", frame=frame)
        first = time.perf_counter() - started
        path = out / f"hot_pour_frame_{frame:03d}.png"
        save_png(np.asarray(image.pixels), path)
        evaluator.clear()
        started = time.perf_counter()
        evaluator.evaluate_raster(document, "render", frame=frame)
        results[str(frame)] = {"first_render_s": round(first, 1), "second_render_s": round(time.perf_counter() - started, 1),
                               "size": [width, height], "samples": args.still_samples, "file": str(path)}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="default", choices=("default", "integrated", "cpu"))
    parser.add_argument("--liquid-cells", type=int, default=256)
    parser.add_argument("--smoke-cells", type=int, default=192)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--spout-height", type=float, default=0.62)
    parser.add_argument("--liquid-pressure", default="resident")
    parser.add_argument("--smoke-pressure", default="resident_sparse")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--reuse-cache", action="store_true", help="skip simulation and use the cache already under --workdir")
    parser.add_argument("--no-steam", action="store_true")
    parser.add_argument("--no-whitewater", action="store_true")
    parser.add_argument("--playback", action="store_true")
    parser.add_argument("--play-width", type=int, default=960)
    parser.add_argument("--play-height", type=int, default=540)
    parser.add_argument("--stills", default=None)
    parser.add_argument("--still-frames", default="60,120")
    parser.add_argument("--still-size", default="1920x1080")
    parser.add_argument("--still-samples", type=int, default=64)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    workdir = Path(args.workdir or f"/tmp/hot-pour-{args.adapter}-{args.liquid_cells}-{args.smoke_cells}")
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["NODEBASED_SIM_CACHE_MB"] = "400000"
    os.environ["XDG_CACHE_HOME"] = str(workdir / "xdg")
    force_adapter(args.adapter)
    from nodebased import gpu3d, simcache
    from nodebased.imaging import Evaluator
    gpu = gpu3d._state()
    adapter = str(gpu["info"].get("device"))
    sim_root = workdir / "simcache"
    store = simcache.SimCache(root=sim_root, memory_budget=2 << 30, disk_budget=400_000 << 20)
    evaluator = Evaluator(sim=store)
    dispatcher = build_document(args)
    document = dispatcher.document
    result = {"adapter": str(gpu["info"].get("device")), "adapter_request": args.adapter,
              "liquid_cells": args.liquid_cells, "smoke_cells": args.smoke_cells, "frames": args.frames,
              "passes": {}}

    def checkpoint(pass_name):
        """Persist each completed pass immediately so an interrupted later pass keeps its evidence."""
        if args.json:
            result["completed_pass"] = pass_name
            target = Path(args.json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=1, default=str))

    if args.reuse_cache:
        result["passes"]["bake"] = "reused existing cache; bake metrics unavailable"
    else:
        sampler = DriverGpuSampler(args.adapter)
        sampler.start()
        started = time.perf_counter()
        rows, peaks = bake(args, document, evaluator, gpu)
        wall = time.perf_counter() - started
        sampler.stop()
        result.update({"bake_wall_s": round(wall, 1), "stages": peaks, "driver_gpu_peak_mb": sampler.peak,
              "liquid": percentile_row([r["liquid_s"] for r in rows.values()]),
              "surface": percentile_row([r["surface_s"] for r in rows.values()]),
              "whitewater": percentile_row([r["whitewater_s"] for r in rows.values() if "whitewater_s" in r]),
              "steam": percentile_row([r["steam_s"] for r in rows.values() if "steam_s" in r]),
              "peak_particles": max((r.get("particles", 0) for r in rows.values()), default=0),
              "peak_triangles": max((r.get("triangles", 0) for r in rows.values()), default=0),
              "liquid_backend": rows[1].get("liquid_backend"), "liquid_fallback": rows[1].get("liquid_fallback"),
              "steam_backend": rows[1].get("steam_backend"),
              "peak_steam_tiles": max((r.get("steam_tiles", 0) for r in rows.values()), default=0),
              "per_frame": {str(f): rows[f] for f in rows}})
    result["cache"] = cache_report(sim_root, args)
    if not args.reuse_cache:
        result["passes"]["bake"] = "complete"
        checkpoint("bake")
    if args.playback and not args.no_steam:
        add_playback_scene(dispatcher)
        document = dispatcher.document
        evaluator = Evaluator(sim=simcache.SimCache(root=sim_root, memory_budget=2 << 30, disk_budget=400_000 << 20))
        result["playback"] = playback(args, document, evaluator, gpu)
        result["passes"]["playback"] = "complete"
        checkpoint("playback")
    if args.stills:
        result["stills"] = render_stills(args, dispatcher, evaluator)
        result["passes"]["stills"] = "complete"
        checkpoint("stills")
    text = json.dumps(result, indent=1, default=str)
    if args.json:
        Path(args.json).write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame"}, indent=1, default=str))


if __name__ == "__main__":
    main()
