"""Where an adaptive render spends its time, per pass and per phase, next to a fixed render of the same budget (Lane 4, step Q1).

    flock /tmp/nb-gpu.lock python tools/benchmark_adaptive_overhead.py --adapter integrated|cpu|discrete [--scene Y1]
                                                                       [--size 640x360] [--out FILE.json]

Rows, for one scene of `tools/benchmark_adaptive.py` (default Y1):

* `fixed 64`: the render adaptive sampling replaces.
* `never stops, pass size P`: adaptive at a threshold no pixel reaches, 16 minimum and 64 maximum samples, so every pixel takes
  exactly 64 samples like the fixed render but in `1 + 48 / P` passes. The wall time against the pass count is the per-pass
  cost (compaction, the convergence test, submission, the wait), because the shader work is identical in every row: the
  slope of a straight-line fit is milliseconds per pass, its intercept the fixed render's cost.
* `adaptive 0.003, Path samples 65536` and `adaptive 0.003, Path samples 64`: the portable benchmark's adaptive case (16
  minimum, 256 maximum, passes of 8) with `samples` large enough that it caps nothing (the behaviour before step Q1, which
  ignored `samples`) and with `samples = 64`, the fixed render it replaces (no pixel takes more than 64 from step Q1 on).

Every row also carries its PSNR against a 1024-sample fixed render (another seed) on the same backend, on what the viewer
shows (clipped to 0..1, sRGB encoded).

Each row is the median of three renders after a warm-up, with the renderer's own phase times and the sample counts. Run it
under the exclusive GPU lock.
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_adaptive as ba                                    # noqa: E402
from nodebased import gpu3d, pathtrace as pt                       # noqa: E402

PASS_SIZES = (48, 24, 16, 12, 8, 4, 2)
NEVER = 1e-12


def rows_for(settings_base):
    rows = [("fixed 64", replace(ba.FIXED, samples=64))]
    for size in PASS_SIZES:
        rows.append((f"never stops, pass size {size}",
                     replace(ba.ADAPTIVE, noise_threshold=NEVER, min_samples=16, max_samples=64, adaptive_pass_size=size,
                             samples=65536)))
    rows.append(("adaptive 0.003, Path samples 65536", replace(ba.ADAPTIVE, noise_threshold=0.003, samples=65536)))
    rows.append(("adaptive 0.003, Path samples 64", replace(ba.ADAPTIVE, noise_threshold=0.003, samples=64)))
    return rows


def measure(scene, ambient, settings, width, height, reference=None, repeats=3):
    walls, stats_list = [], []
    for i in range(repeats + 1):
        stats = {}
        started = time.perf_counter()
        image = pt.render(scene, ba.CAMERA, width, height, (0.02, 0.02, 0.03, 1.0), ambient, "rgba", settings, stats=stats,
                          backend="gpu")
        wall = time.perf_counter() - started
        if i:
            walls.append(wall)
            stats_list.append(stats)
    order = sorted(range(len(walls)), key=lambda k: walls[k])
    median = stats_list[order[len(order) // 2]]
    phases = {}
    for st in stats_list:
        for key, value in st["phases"].items():
            phases.setdefault(key, []).append(value)
    samples = median["samples"]
    return dict(psnr_vs_1024_db=round(ba.psnr(image, reference), 2), wall_ms=round(1000 * statistics.median(walls), 2),
                passes=int(median["passes"]),
                mean_spp=round(float(samples.mean()), 2), max_spp=int(samples.max()),
                phases_ms={k: round(1000 * statistics.median(v), 2) for k, v in phases.items()},
                readbacks=median.get("readbacks", {}))


def per_pass_fit(rows):
    pts = [(r["passes"], r["wall_ms"]) for label, r in rows.items() if label.startswith("never stops")]
    if len(pts) < 2:
        return None
    x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    slope, intercept = np.polyfit(x, y, 1)
    return dict(ms_per_pass=round(float(slope), 3), intercept_ms=round(float(intercept), 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="default")
    parser.add_argument("--scene", default="Y1")
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.adapter != "default":
        gpu3d._states.setdefault("default", gpu3d._state(args.adapter))
    width, height = (int(v) for v in args.size.split("x"))
    scene, ambient = ba.scenes()[args.scene]
    result = dict(adapter=args.adapter, device=gpu3d.adapter_report(), scene=args.scene, size=[width, height], rows={})
    reference = pt.render(scene, ba.CAMERA, width, height, (0.02, 0.02, 0.03, 1.0), ambient, "rgba", ba.REFERENCE, backend="gpu")
    for label, settings in rows_for(None):
        result["rows"][label] = measure(scene, ambient, settings, width, height, reference)
        r = result["rows"][label]
        print(f"{label:36s} {r['wall_ms']:8.1f} ms  passes {r['passes']:3d}  mean {r['mean_spp']:6.1f}  "
              f"{r['psnr_vs_1024_db']:6.2f} dB  phases {r['phases_ms']}")
    result["fit"] = per_pass_fit(result["rows"])
    print("fit", result["fit"])
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
