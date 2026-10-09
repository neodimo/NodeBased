"""The host's share of one GPU path-traced frame, by phase, one JSON line per case.

    flock /tmp/nb-gpu.lock python tools/profile_host_phases.py --adapter discrete --cases smoke-grid,Z1-fixed64 [--timed 7]

Every case of `tools/benchmark_portable_render.py` can be named. The renderer's own laps (`stats["phases"]`) give, per frame:
`scene build` and `pack` (the scene encode on the host), `upload` (static buffers, the image accumulator, the pipeline lookup),
`encode` (the per-band uniform, bind group and command encoder), `dispatch` (the submit and the wait for the card, which holds
the shader's own time), `final readback` (the accumulator copied to the host), `convert` (float32 image from the sums),
`stats` (the per-pixel variance and noise a caller that asks for statistics receives) and `postprocess`. `host_ms` is the frame
minus `dispatch`; `host_no_stats_ms` leaves out `stats`, which only callers that pass a stats dict pay for.
The medians are per phase, so they need not add up to the median frame. Run under the shared GPU lock.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOST_ONLY_SKIPS = ("dispatch",)


def run(case, adapter, size, timed, noise_maps=False, pass_samples=0):
    from tools import benchmark_portable_render as bpr
    state = bpr.force_adapter(adapter)
    bpr.set_majorant(case)
    if pass_samples:
        bpr.CASES[case]["settings"]["pass_samples"] = pass_samples      # samples one dispatch takes per pixel (0: one)
    for _ in range(2):                                    # the first render compiles the pipeline
        bpr.render_once(case, size, "gpu", not noise_maps)
    frames, phases, reads, image = [], {}, {}, None
    for _ in range(timed):
        started = time.perf_counter()
        image, stats = bpr.render_once(case, size, "gpu", not noise_maps)
        frames.append(time.perf_counter() - started)
        for key, value in stats.get("phases", {}).items():
            phases.setdefault(key, []).append(value)
        reads = stats.get("readbacks", reads)
    ms = {key: round(statistics.median(values) * 1000, 3) for key, values in phases.items()}
    frame = statistics.median(frames) * 1000
    host = frame - ms.get("dispatch", 0.0)
    return dict(case=case, adapter=adapter, device=bpr.adapter_info(state)["name"], size=list(size), timed=timed, pass_samples=pass_samples,
                frame_ms=round(frame, 2), frames_ms=[round(f * 1000, 2) for f in frames], phases_ms=ms,
                host_ms=round(host, 2), host_no_stats_ms=round(host - ms.get("stats", 0.0), 2),
                waits=reads.get("waits"), counters=reads, build=_build(), passes=None)


def _build():
    from nodebased import gpupathtrace
    return getattr(gpupathtrace, "reuse_counters", lambda: {})()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="discrete", choices=("default", "discrete", "integrated", "cpu"))
    parser.add_argument("--cases", default="smoke-grid,Z1-fixed64")
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--timed", type=int, default=7)
    parser.add_argument("--noise-maps", action="store_true", help="ask for stats[variance] and stats[noise] too (the default skips them)")
    parser.add_argument("--pass-samples", type=int, default=0, help="samples per pixel one dispatch takes (0: the default of one)")
    parser.add_argument("--out", help="append the JSON lines to this file")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    lines = []
    for case in args.cases.split(","):
        line = json.dumps(run(case, args.adapter, size, args.timed, args.noise_maps, args.pass_samples))
        print(line, flush=True)
        lines.append(line)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as handle:
            handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
