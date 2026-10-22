"""Run the test suite in sequential batches, each in a fresh interpreter.

Why: GitHub's hosted Ubuntu runners (4 cores, 16 GB) died with "lost communication ... starves it for
CPU/Memory" or exit code 143 on every run after 2026-09-25 6:05 PM PDT, always deep into one long
``unittest discover`` process. Each module is bounded on its own (the heaviest viewer and viewport
modules peak around 1 GB), so the growth is accumulation across the single process: Qt windows,
GPU adapters and caches that only the interpreter's exit returns. A fresh interpreter per batch
releases all of it, and a runner that dies now loses one batch's output, not the whole log.

``--jobs N`` runs up to N batches at once (default 1 locally, os.cpu_count() when the CI variable is set).
The serial path is unchanged. The parallel path exists because the serial suite took 1 h 42 min on
Linux and 1 h 56 min on Windows (run 39749648280), 48 and 52 minutes of it in one batch, and the
hosted runners have 4 cores that one batch at a time leaves idle. It schedules the heaviest work
first (COST_SECONDS, measured from that run), gives every module of COST_ISOLATE seconds or more its own
batch, splits SPLIT_MODULES down to single tests, packs the rest as before, and prints each batch's
output whole when the batch ends, so lines from different batches never interleave.

Concurrency safety: every batch is its own interpreter, and tests/isolation.py gives each interpreter
its own QSettings directory and GPU lock file; tests that need ports bind port 0; the temp files tests
write come from tempfile. SERIAL_MODULES (wall-clock budget tests) run alone after the pool drains.

Usage: python tools/run_tests.py [--batch N] [--jobs N] [-v] [pattern ...]
Exit status is non-zero if any batch fails. Serial runs go in sorted order, like discover.
"""
import argparse
import ast
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def modules(patterns):
    """Sorted module names matching any of the glob patterns (a single string is one pattern)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    found = {f"tests.{p.stem}" for pattern in patterns for p in (ROOT / "tests").glob(pattern) if p.is_file()}
    return sorted(found)


# Seconds per module on a GitHub Linux runner, from the per-test timestamps of run 39749648280
# (modules not listed are estimated at DEFAULT_COST). Only the order matters, not the accuracy.
COST_SECONDS = {
    "tests.test_desktop": 290, "tests.test_memory_ceiling_gate": 170, "tests.test_lens_shot_l3": 160,
    "tests.test_3d_shadow_catcher": 150, "tests.test_demo_shot": 120, "tests.test_3d_viewport_render_mode": 120,
    "tests.test_3d_holdout": 120, "tests.test_3d_splat_relight_benchmark": 90, "tests.test_bypass": 65,
    "tests.test_3d_first_look": 60, "tests.test_particles_collide": 50, "tests.test_3d_light_groups": 50,
    "tests.test_fluid_upres": 50, "tests.test_knob_sliders": 45,
}
DEFAULT_COST = 8
COST_ISOLATE = 100  # modules at least this slow get a batch of their own
# One test at a time: the three slow methods of test_3d_progressive_convergence are 8 to 16 minutes each.
SPLIT_MODULES = {"tests.test_3d_progressive_convergence": {
    "tests.test_3d_progressive_convergence.ConvergenceTests.test_depth_of_field_agrees_and_is_really_there": 940,
    "tests.test_3d_progressive_convergence.ConvergenceTests.test_the_viewport_picture_through_a_lens_agrees_with_render3d_through_lens": 460,
    "tests.test_3d_progressive_convergence.OutputsUntouchedTests.test_render3d_gives_the_same_picture_before_and_after_a_viewport_pass": 600,
}}
# Run alone, after the pool: these assert wall-clock budgets (the first parallel run failed
# test_m2_latency_gate's 43.8 ms time-to-first-pixel at 50.4 ms on a loaded CPU) or exercise the GPU lock
# protocol with real timing. They are cheap, a few seconds each.
SERIAL_MODULES = {
    "tests.test_gpulock", "tests.test_m2_latency_gate", "tests.test_m2_throughput_gate",
    "tests.test_graph_input_latency", "tests.test_graph_display_rate", "tests.test_perf_playback",
    "tests.test_transform_resample_perf", "tests.test_benchmark_adaptive_overhead",
    "tests.test_benchmark_hot_pour", "tests.test_benchmark_portable_render",
    "tests.test_directviewer_playback", "tests.test_playback", "tests.test_m3_vram_budget",
}


def split_units(module):
    """One dotted test id per test method of `module`, read from its source without importing it."""
    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
    units = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            units += [f"{module}.{cls.name}.{fn.name}" for fn in cls.body
                      if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test")]
    return units


def plan(mods, batch_size):
    """(cost, [unittest names]) batches for the pool and the serial group, heaviest pool batch first."""
    pooled, serial, singles = [], [], []
    for m in mods:
        if m in SERIAL_MODULES:
            serial.append([m])
        elif m in SPLIT_MODULES:
            ids = split_units(m)
            if not ids:
                singles.append((DEFAULT_COST, [m]))
                continue
            heavy = SPLIT_MODULES[m]
            singles += [(heavy.get(i, 120 if heavy else DEFAULT_COST), [i]) for i in ids]
        elif COST_SECONDS.get(m, DEFAULT_COST) >= COST_ISOLATE:
            singles.append((COST_SECONDS[m], [m]))
        else:
            pooled.append(m)
    packed = [pooled[i:i + batch_size] for i in range(0, len(pooled), batch_size)]
    batches = singles + [(sum(COST_SECONDS.get(m, DEFAULT_COST) for m in b), b) for b in packed]
    batches.sort(key=lambda cb: -cb[0])
    return [b for _, b in batches], serial


def run_captured(n, total, batch, cmd, env):
    started = time.monotonic()
    notes = ""
    while True:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = proc.stdout.decode("utf-8", "replace")
        # Every test passed and the interpreter then died on a signal at exit (seen once in two full
        # local parallel runs: SIGSEGV after "OK" in a batch that passes alone three times out of three).
        # Say so, and run the batch once more; a second death or any test failure still fails the batch.
        if proc.returncode < 0 and not notes and re.search(r"^OK\b", text, re.M):
            notes = (text + f"\n=== batch {n}: all tests passed, then the process died with signal {-proc.returncode} "
                     f"at exit; running it once more\n")
            continue
        break
    elapsed = time.monotonic() - started
    out = (f"=== batch {n} of {total}: {batch[0]} .. {batch[-1]} ({len(batch)} modules)\n{notes}{text}"
           + ("" if text.endswith("\n") or not text else "\n")
           + f"=== batch {n} exit {proc.returncode} in {elapsed:.1f} s")
    return n, proc.returncode, out, text


def run_parallel(mods, args, env):
    pool_batches, serial_batches = plan(mods, args.batch)
    pool_env = dict(env, NODEBASED_TEST_TIMEOUT_SCALE=str(args.jobs))  # see tests/progressive_widget.py
    batches = pool_batches + serial_batches
    total = len(batches)
    failed, ran = [], 0
    started = time.monotonic()

    def cmd_for(batch):
        return [sys.executable, "-X", "faulthandler", "-m", "unittest"] + (["-v"] if args.verbose else []) + batch

    def finish(result):
        nonlocal ran
        n, code, out, text = result
        print(out, flush=True)
        ran += sum(int(c) for c in re.findall(r"^Ran (\d+) tests?", text, re.M))
        if code:
            b = batches[n - 1]
            failed.append((n, b[0], b[-1], code))

    print(f"=== {total} batches over {args.jobs} jobs, heaviest first", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_captured, n, total, b, cmd_for(b), pool_env)
                   for n, b in enumerate(pool_batches, 1)]
        for f in concurrent.futures.as_completed(futures):
            finish(f.result())
    for n, b in enumerate(serial_batches, len(pool_batches) + 1):
        finish(run_captured(n, total, b, cmd_for(b), env))
    # A wall-clock assertion can miss its budget only because three other batches share the cores (CI run
    # 39792276070: test_roto's 0.25 s feather render). Each failed pool batch gets one more try alone
    # once the pool has drained; a failure that survives it fails the run, and both outputs are printed.
    retry = [n for n, *_ in sorted(failed) if n <= len(pool_batches)]
    for n in retry:
        print(f"=== batch {n} failed in the pool; running it once more alone", flush=True)
        result = run_captured(n, total, batches[n - 1], cmd_for(batches[n - 1]), env)
        failed[:] = [f for f in failed if f[0] != n]
        finish(result)
    failed.sort()
    print(f"=== {total} batches, {len(mods)} modules, {ran} tests, {time.monotonic() - started:.1f} s total", flush=True)
    for n, first, last, code in failed:
        print(f"=== FAILED batch {n} ({first} .. {last}) exit {code}", flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=12, help="test modules per interpreter")
    parser.add_argument("--jobs", type=int, default=(os.cpu_count() or 1) if os.environ.get("CI") else 1,
                        help="batches to run at once (default: 1, or the CPU count when CI is set)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("pattern", nargs="*", default=["test*.py"],
                        help="one or more globs under tests/ (tools/field_report.py passes the GPU modules by name)")
    args = parser.parse_args()
    mods = modules(args.pattern)
    if not mods:
        print("no test modules found", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    if args.jobs > 1:
        return run_parallel(mods, args, env)
    failed = []
    started = time.monotonic()
    batches = [mods[i:i + args.batch] for i in range(0, len(mods), args.batch)]
    for n, batch in enumerate(batches, 1):
        print(f"=== batch {n} of {len(batches)}: {batch[0]} .. {batch[-1]} ({len(batch)} modules)", flush=True)
        cmd = [sys.executable, "-X", "faulthandler", "-m", "unittest"] + (["-v"] if args.verbose else []) + batch
        t = time.monotonic()
        code = subprocess.call(cmd, cwd=ROOT, env=env)
        print(f"=== batch {n} exit {code} in {time.monotonic() - t:.1f} s", flush=True)
        if code:
            failed.append((n, batch[0], batch[-1], code))
    print(f"=== {len(batches)} batches, {len(mods)} modules, {time.monotonic() - started:.1f} s total", flush=True)
    for n, first, last, code in failed:
        print(f"=== FAILED batch {n} ({first} .. {last}) exit {code}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
