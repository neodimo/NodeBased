"""Run the test suite in sequential batches, each in a fresh interpreter.

Why: GitHub's hosted Ubuntu runners (4 cores, 16 GB) died with "lost communication ... starves it for
CPU/Memory" or exit code 143 on every run after 2026-09-25 6:05 PM PDT, always deep into one long
``unittest discover`` process. Each module is bounded on its own (the heaviest viewer and viewport
modules peak around 1 GB), so the growth is accumulation across the single process: Qt windows,
GPU adapters and caches that only the interpreter's exit returns. A fresh interpreter per batch
releases all of it, and a runner that dies now loses one batch's output, not the whole log.

Usage: python tools/run_tests.py [--batch N] [-v] [pattern]
Exit status is non-zero if any batch fails. Modules run in sorted order, like discover.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def modules(pattern):
    return sorted(f"tests.{p.stem}" for p in (ROOT / "tests").glob(pattern) if p.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=12, help="test modules per interpreter")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("pattern", nargs="?", default="test*.py")
    args = parser.parse_args()
    mods = modules(args.pattern)
    if not mods:
        print("no test modules found", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    failed = []
    started = time.monotonic()
    batches = [mods[i:i + args.batch] for i in range(0, len(mods), args.batch)]
    for n, batch in enumerate(batches, 1):
        print(f"=== batch {n} of {len(batches)}: {batch[0]} .. {batch[-1]} ({len(batch)} modules)", flush=True)
        cmd = [sys.executable, "-m", "unittest"] + (["-v"] if args.verbose else []) + batch
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
