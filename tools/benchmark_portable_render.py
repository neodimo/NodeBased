"""Portable final-render benchmark: the same representative scenes on every adapter that works, one JSON file per case.

    python tools/benchmark_portable_render.py --matrix [--adapters integrated,cpu] [--size 640x360] [--out DIR]
                                                       [--timeout 900] [--lock-wait 3600] [--force]
    python tools/benchmark_portable_render.py --case X1-fixed64 --adapter integrated --out DIR     (one case, in-process)
    python tools/benchmark_portable_render.py --report DIR                                         (markdown table of DIR)

Cases: the X1, Y1 and Z1 scenes of `tools/benchmark_adaptive.py` at fixed 64 samples and adaptive 0.003, plus a sparse smoke
plume (`tools/benchmark_volume_majorant.py`, a tenth of its box) with Rendering 8's two-level majorant grid and, for
contrast, the single-box majorant. Per case it records the adapter, backend, scene, settings, seed and resolution, the
warm-up and timed frame counts, the median wall time, the renderer's phase times and sample counts, the smoke cases'
collision counters and PSNR / error against the CPU path tracer rendered with the same settings and seed (cached beside the
results, so it is rendered once per case and size).

Every case is its own subprocess under a short exclusive GPU-lock turn (`/tmp/nb-gpu.lock`, all adapters), with a
timeout. Its JSON is written the moment it finishes, so an interrupted run keeps the earlier cases; a case whose file exists
is skipped unless `--force`. An adapter that is missing or hangs gives a record with status `unavailable` or `timeout`: it is
never omitted and never a pass. `discrete` (the RTX) is not in the default adapters while its eGPU is in the NVRM lock.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOCK = "/tmp/nb-gpu.lock"
SCHEMA = 1
HARDWARE = ("default", "discrete", "integrated")
DEFAULT_ADAPTERS = ("integrated", "cpu")
WARMUP, TIMED = 1, 3
BASE = dict(max_bounces=8, seed=1)


def _case_table():
    """id -> dict(scene=name, label=text, settings=dict of PathSettings fields, kind='surface'|'smoke', majorant=...)."""
    cases = {}
    for scene in ("X1", "Y1", "Z1"):
        cases[f"{scene}-fixed64"] = dict(scene=scene, kind="surface", label="fixed 64",
                                         settings=dict(sampling="fixed", samples=64, **BASE))
        cases[f"{scene}-adaptive0.003"] = dict(scene=scene, kind="surface", label="adaptive 0.003",
                                               settings=dict(sampling="adaptive", noise_threshold=0.003, min_samples=16,
                                                             max_samples=256, adaptive_pass_size=8, **BASE))
    smoke = dict(sampling="fixed", samples=32, max_bounces=4, seed=4)
    cases["smoke-grid"] = dict(scene="smoke plume 64", kind="smoke", label="fixed 32, two-level majorant grid",
                               majorant="grid", settings=smoke)
    cases["smoke-box"] = dict(scene="smoke plume 64", kind="smoke", label="fixed 32, single-box majorant",
                              majorant="box", settings=smoke)
    return cases


CASES = _case_table()


# ----------------------------------------------------------------------------------------------------------- pure helpers

def compare_images(image, reference):
    """Deterministic comparison of two renders on what the viewer shows (clipped to 0..1, sRGB encoded, peak 1):
    PSNR in dB (inf-free: capped at 200), mean and largest absolute difference. Shape mismatches raise."""
    from nodebased.imaging import linear_to_srgb

    def display(a):
        return np.clip(linear_to_srgb(np.clip(np.asarray(a)[..., :3].astype(np.float64), 0, 1)), 0, 1)
    a, b = display(image), display(reference)
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} against {b.shape}")
    diff = np.abs(a - b)
    mse = float(np.mean(diff ** 2))
    return dict(psnr_db=round(min(10 * np.log10(1.0 / max(mse, 1e-20)), 200.0), 3),
                mean_abs=float(diff.mean()), max_abs=float(diff.max()))


def result_path(out, adapter, case):
    return Path(out) / f"{adapter}__{case}.json"


def save_case(out, result):
    """Write one case's JSON atomically (temp file then rename), so a killed run never leaves half a file."""
    path = result_path(out, result["adapter_request"], result["case"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_cases(out):
    results = []
    for path in sorted(Path(out).glob("*__*.json")):
        try:
            results.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            results.append(dict(case=path.stem, status="corrupt", adapter_request=path.stem.split("__")[0]))
    return results


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def _record(case, adapter, size, status, **extra):
    spec = CASES[case]
    return dict(schema=SCHEMA, case=case, status=status, adapter_request=adapter, scene=spec["scene"], label=spec["label"],
                settings=spec["settings"], seed=spec["settings"]["seed"], size=list(size), warmup=WARMUP, timed=TIMED,
                commit=git_head(), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)


# ------------------------------------------------------------------------------------------------------------ rendering

def force_adapter(kind):
    from nodebased import gpu3d
    if kind != "default":
        original = gpu3d._state
        gpu3d._state = lambda choice=None: original(kind if (choice or "default") == "default" else choice)
    return gpu3d._state()           # raises RuntimeError when the adapter does not exist or cannot start


def adapter_info(state):
    info = state["info"]
    return dict(name=info.get("device") or info.get("description") or "unnamed", backend=str(info.get("backend_type", "unknown")),
                type=str(info.get("adapter_type", "unknown")), driver=str(info.get("description", "")),
                format=state["format"])


def build_case(case):
    """(scene, camera, ambient, background, PathSettings, volume settings or None)."""
    from nodebased import pathtrace as pt, scene3d as s
    spec = CASES[case]
    settings = pt.PathSettings(**spec["settings"])
    if spec["kind"] == "smoke":
        from tools.benchmark_volume_majorant import tenth_plume
        scene, camera, volume, _ = tenth_plume(64)
        return scene, camera, .7, (0, 0, 0, 0), settings, volume
    from tools.benchmark_adaptive import scenes
    scene, ambient = scenes()[spec["scene"]]
    return scene, s.Camera(), ambient, (0.02, 0.02, 0.03, 1.0), settings, None


def set_majorant(case):
    from nodebased import gpupathtrace, ptvolume
    mode = CASES[case].get("majorant")
    if mode:
        ptvolume.ENABLE_SKIP = gpupathtrace.ENABLE_VOLUME_SKIP = mode != "box"
        ptvolume.MAJORANT_RATIO = 4


def render_once(case, size, backend):
    from nodebased import pathtrace as pt
    scene, camera, ambient, background, settings, volume = build_case(case)
    stats = {}
    kwargs = dict(volume=volume) if volume is not None else {}
    image = pt.render(scene, camera, size[0], size[1], background, ambient, "rgba", settings, stats=stats, backend=backend, **kwargs)
    return image, stats


def reference_image(case, size, out):
    """The CPU path tracer at the case's own settings and seed, cached as an .npy in `out` (it is the slow render)."""
    path = Path(out) / "reference" / f"{case}_{size[0]}x{size[1]}.npy"
    if path.exists():
        return np.load(path)
    set_majorant(case)
    started = time.perf_counter()
    image, _ = render_once(case, size, "cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, image.astype(np.float32))
    (path.with_suffix(".txt")).write_text(f"cpu reference render took {time.perf_counter() - started:.1f} s\n")
    return image


def _summarise(stats):
    out = {}
    if "samples" in stats:
        samples = np.asarray(stats["samples"])
        out["samples"] = dict(mean=float(samples.mean()), min=int(samples.min()), max=int(samples.max()))
    if stats.get("phases"):
        out["phases_ms"] = {k: round(v * 1000, 3) for k, v in stats["phases"].items()}
    for key in ("passes", "backend", "fallback"):
        if key in stats:
            out[key] = stats[key]
    return out


def collision_counters(case, adapter_state):
    """Smoke cases only: tentative, real and hop counts per camera path from the renderer's own counters (CPU layer
    counters; the GPU counting shader on a small image). Empty for surface cases."""
    if CASES[case]["kind"] != "smoke":
        return {}
    from nodebased import gpupathtrace, pathtrace as pt
    scene, camera, _, _, _, volume = build_case(case)
    counters = {}
    if adapter_state is not None:
        gpupathtrace.COUNT_COLLISIONS = True
        try:
            image = pt.render(scene, camera, 64, 64, ambient=.7, volume=volume, backend="gpu",
                              settings=pt.PathSettings(samples=8, max_bounces=4, seed=4))
        finally:
            gpupathtrace.COUNT_COLLISIONS = False
        t, r, h = (float(v) for v in image[..., :3].reshape(-1, 3).mean(axis=0))
        counters["gpu_per_path"] = dict(tentative=t, real=r, null=t - r, hops=h)
    from tools.benchmark_volume_majorant import counts_cpu
    _, t, r, h = counts_cpu(scene, camera, volume, 24)
    counters["cpu_per_path"] = dict(tentative=t, real=r, null=t - r, hops=h)
    return counters


def run_case(case, adapter, size, out):
    """Render one case on one adapter and write its JSON. Never raises for a missing adapter: that is a record."""
    backend = "gpu"      # "cpu" as an adapter is llvmpipe, the software Vulkan device; the CPU path tracer is the reference
    state = None
    try:
        state = force_adapter(adapter)
    except Exception as exc:
        return save_case(out, _record(case, adapter, size, "unavailable", reason=f"{type(exc).__name__}: {exc}"))
    try:
        info = adapter_info(state)
        reference = reference_image(case, size, out)
        set_majorant(case)
        for _ in range(WARMUP):
            render_once(case, size, backend)
        times, stats, image = [], {}, None
        for _ in range(TIMED):
            started = time.perf_counter()
            image, stats = render_once(case, size, backend)
            times.append(time.perf_counter() - started)
        result = _record(case, adapter, size, "ok", adapter_info=info, times_s=[round(t, 6) for t in times],
                         median_s=statistics.median(times), renderer=_summarise(stats),
                         counters=collision_counters(case, state),
                         quality=dict(reference="cpu backend, same settings and seed", **compare_images(image, reference)))
    except Exception as exc:
        result = _record(case, adapter, size, "error", adapter_info=adapter_info(state), reason=f"{type(exc).__name__}: {exc}")
    return save_case(out, result)


# ---------------------------------------------------------------------------------------------------------- the matrix

class GpuTurn:
    """One short exclusive turn on the shared lock; times out instead of waiting for ever."""

    def __init__(self, wait):
        self.wait, self.fd = wait, None

    def __enter__(self):
        self.fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o666)
        deadline = time.monotonic() + self.wait
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() > deadline:
                    os.close(self.fd)
                    raise TimeoutError(f"GPU lock {LOCK} still held after {self.wait} s")
                time.sleep(2)

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def run_matrix(adapters, cases, size, out, timeout, lock_wait, force):
    for adapter in adapters:
        for case in cases:
            path = result_path(out, adapter, case)
            if path.exists() and not force:
                print(f"skip {adapter} {case}: {path.name} exists", flush=True)
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--case", case, "--adapter", adapter,
                       "--size", f"{size[0]}x{size[1]}", "--out", str(out)]
            env = dict(os.environ, PYTHONPATH=str(ROOT), QT_QPA_PLATFORM="offscreen")
            started = time.perf_counter()
            try:
                if not (Path(out) / "reference" / f"{case}_{size[0]}x{size[1]}.npy").exists():
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--prepare-reference", case,
                                    "--size", f"{size[0]}x{size[1]}", "--out", str(out)],
                                   cwd=ROOT, env=env, timeout=timeout, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                turn = GpuTurn(lock_wait)
                if turn:
                    turn.__enter__()
                try:
                    subprocess.run(command, cwd=ROOT, env=env, timeout=timeout, check=False, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                finally:
                    if turn:
                        turn.__exit__()
                if not path.exists():
                    save_case(out, _record(case, adapter, size, "error", reason="worker exited without writing a result"))
            except subprocess.CalledProcessError as exc:
                save_case(out, _record(case, adapter, size, "error", reason=f"CPU reference preparation failed with exit {exc.returncode}"))
            except subprocess.TimeoutExpired:
                save_case(out, _record(case, adapter, size, "timeout", reason=f"no result within {timeout} s"))
            except TimeoutError as exc:
                save_case(out, _record(case, adapter, size, "unavailable", reason=str(exc)))
            status = json.loads(path.read_text())["status"]
            print(f"{adapter:10} {case:18} {status:12} {time.perf_counter() - started:7.1f} s", flush=True)


def report(out):
    lines = ["| adapter | backend | case | status | median | passes | mean spp | PSNR vs CPU | max error |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in load_cases(out):
        if r.get("status") != "ok":
            lines.append(f"| {r.get('adapter_request')} | - | {r.get('case')} | {r.get('status')}: {r.get('reason', '')} | | | | | |")
            continue
        info, ren, q = r["adapter_info"], r["renderer"], r["quality"]
        spp = ren.get("samples", {}).get("mean")
        lines.append(f"| {info['name']} | {info['backend']} | {r['case']} | ok | {r['median_s'] * 1000:.0f} ms | {ren.get('passes', '')} | "
                     f"{'' if spp is None else f'{spp:.1f}'} | {q['psnr_db']:.1f} dB | {q['max_abs']:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--prepare-reference")
    parser.add_argument("--adapter", default="cpu")
    parser.add_argument("--adapters", default=",".join(DEFAULT_ADAPTERS))
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--out", default=str(ROOT / "scratch-portable-render"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--lock-wait", type=int, default=3600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    if args.report:
        print(report(args.report))
    elif args.prepare_reference:
        reference_image(args.prepare_reference, size, args.out)
    elif args.case:
        print(run_case(args.case, args.adapter, size, args.out))
    elif args.matrix:
        run_matrix(args.adapters.split(","), args.cases.split(","), size, args.out, args.timeout, args.lock_wait, args.force)
        print(report(args.out))
    else:
        parser.error("give --matrix, --case or --report")


if __name__ == "__main__":
    main()
