"""Evaluator benchmark harness.

This is a deliverable in its own right, not a convenience script. `docs/VISION.md` says M2 chooses
its GPU abstraction through measured experiments, and `docs/EVALUATION_TIERS.md` refuses to accept
the word "faster" in place of numbers. Nothing here can decide that on its own, but it establishes
the measurement discipline and the baseline the tiers have to beat.

What is measured:

* **Cold time-to-first-pixel** — an empty cache evaluating the graph once. This is what an artist
  waits through after opening a project or changing a source.
* **Warm time-to-first-pixel** — the same evaluation with the cache populated.
* **Interaction latency p50/p95** — repeated parameter edits with the source still cached, which
  is the recompute cost the tiers exist to make bearable. Timeline scrubbing is deliberately not
  measured here: with no Read and no animated parameter in the graph, every digest is frame-stable
  and a scrub would report cache-lookup time rather than evaluation time.

Output is JSON so results are diffable across machines and commits. Machine identity and the build
commit are recorded, because a latency number without them is an anecdote.

    uv run python -m nodebased.bench --resolution 4k --frames 24
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time

from .core import SPECS
from .imaging import Evaluator

RESOLUTIONS = {"hd": (1920, 1080), "2k": (2048, 1152), "4k": (3840, 2160), "8k": (7680, 4320)}


def build_graph(width: int, height: int) -> dict:
    """A representative interactive graph: generated sources, a spatial filter, a resample, a merge.

    Deliberately built from Constant/Checker rather than a Read so the benchmark measures the
    evaluator instead of the filesystem and the image decoder. Disk-bound Read benchmarks are a
    separate question and would drown this signal.
    """

    def node(name, kind, inputs, **params):
        spec = SPECS[kind]
        slots = {slot: None for slot in spec["inputs"]}
        slots.update({slot: None for slot in spec.get("optional_inputs", [])})
        slots.update(inputs)
        return {"name": name, "type": kind, "inputs": slots,
                "params": dict(spec["params"], **params), "disabled": False,
                "position": [0, 0]}

    nodes = {
        "plate": node("plate", "Checker", {}, width=width, height=height, size=64),
        "wash": node("wash", "Constant", {}, width=width, height=height,
                     red=0.2, green=0.05, blue=0.4, alpha=0.5),
        "grade": node("grade", "Grade", {"image": "plate"}, exposure=0.75, multiply=1.1),
        "blur": node("blur", "Blur", {"image": "grade"}, radius=12.0),
        "xform": node("xform", "Transform", {"image": "blur"},
                      translate_x=17.0, translate_y=-9.0, rotate=6.0, scale=1.05,
                      center_x=width / 2, center_y=height / 2, filter="bilinear"),
        "merge": node("merge", "Merge", {"A": "wash", "B": "xform"}, operation="over", mix=0.8),
        "viewer": node("viewer", "Viewer", {"image": "merge"}),
    }
    return {"version": 5, "nodes": nodes, "view": "viewer",
            "time": {"first": 1, "last": 240, "current": 1, "fps": 24.0}}


def machine_identity() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "unknown"
    return {"commit": commit, "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(), "numpy": numpy_version}


def measure(document, frames: int, tier: int = 1) -> dict:
    evaluator = Evaluator()

    start = time.perf_counter()
    evaluator.evaluate(document, frame=1)
    cold_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    evaluator.evaluate(document, frame=1)
    warm_ms = (time.perf_counter() - start) * 1000.0

    # Interaction latency, measured by dragging a slider rather than by scrubbing time.
    #
    # A first version of this walked consecutive frames and reported 0.03 ms at HD, which is a
    # measurement artefact, not a result: this graph has no Read and no animated parameter, so
    # every digest is frame-stable and every "scrub" frame is a pure cache hit. Until animation
    # merges and a sequence Read is in the graph, a timeline scrub here measures the dictionary
    # lookup and nothing else.
    #
    # Changing a parameter each iteration measures the thing the tiers actually exist to fix: the
    # recompute cost downstream of an edit, with the upstream source still cached. That is the
    # worst case an artist feels when adjusting a grade over a 4K plate.
    samples = []
    for index in range(frames):
        document["nodes"]["grade"]["params"]["exposure"] = 0.5 + index * 0.01
        start = time.perf_counter()
        evaluator.evaluate(document, frame=1)
        samples.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples)
    return {
        "tier": tier,
        "cold_ttfp_ms": round(cold_ms, 3),
        "warm_ttfp_ms": round(warm_ms, 3),
        "interaction_edits": frames,
        "interaction_p50_ms": round(statistics.median(ordered), 3),
        "interaction_p95_ms": round(ordered[max(0, math_ceil_index(len(ordered), 0.95))], 3),
        "interaction_min_ms": round(ordered[0], 3),
        "interaction_max_ms": round(ordered[-1], 3),
        "cache_hits": evaluator.hits,
        "cache_misses": evaluator.misses,
        "cache_bytes": evaluator.bytes,
    }


def math_ceil_index(count: int, percentile: float) -> int:
    """Index of the nearest-rank percentile in a 0-based sorted list."""
    return min(count - 1, int(-(-count * percentile // 1)) - 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure NodeBased evaluator latency.")
    parser.add_argument("--resolution", default="4k", choices=sorted(RESOLUTIONS))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--tier", type=int, default=1,
                        help="Proxy tier. Only tier 1 exists until ROI execution lands.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args(argv)

    width, height = RESOLUTIONS[args.resolution]
    document = build_graph(width, height)
    result = {"resolution": args.resolution, "width": width, "height": height,
              "machine": machine_identity(), **measure(document, args.frames, args.tier)}

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(json.dumps(result, indent=2))
    print(f"\n{args.resolution} tier {result['tier']}: "
          f"cold {result['cold_ttfp_ms']:.1f} ms, warm {result['warm_ttfp_ms']:.1f} ms, "
          f"edit p50 {result['interaction_p50_ms']:.1f} ms / "
          f"p95 {result['interaction_p95_ms']:.1f} ms",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
