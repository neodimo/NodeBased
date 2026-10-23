"""What the minification filter costs: draw time of a 4K picture zoomed out, filtered against nearest.

For each view (sRGB, ACES 2.0), surface size and zoom (1:2, 1:4, 1:8) a presenter draws the same 4K half-float
picture into an offscreen surface 40 times, five times over (the card is warmed first, the figure is the best
five), once with the shipped shader and once with the filter's tap count held at 1 (the old nearest sampling).
The extra time is the difference. Gate: 1 ms per frame at 4K on the RTX 3080 Ti. A new picture is drawn at least
once per frame, so the extra per frame is the extra per draw.

    PYTHONPATH=. python tools/viewer_direct_minify_bench.py OUT.json [--runs 10] [--size 3840x2160]

Run it holding /tmp/nb-gpu.lock (flock) on a machine where the card is otherwise idle. Offscreen: no display needed.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--runs", type=int, default=10, help="repeat the whole measurement this many times")
p.add_argument("--size", default="3840x2160", help="the picture")
p.add_argument("--gate-ms", type=float, default=1.0)
p.add_argument("--adapter", choices=("default", "integrated", "cpu"), default="default",
               help="the card to measure: the app's default (discrete), the integrated GPU, or llvmpipe")
p.add_argument("--surfaces", default="1920x1080,3840x2160")
p.add_argument("--warm", type=int, default=60, help="draws before timing (the card's clocks come up)")
p.add_argument("--draws", type=int, default=40, help="draws in one timed batch")
p.add_argument("--batches", type=int, default=5, help="timed batches; the best is the figure")
args = p.parse_args()

from PySide6.QtWidgets import QApplication
APP = QApplication.instance() or QApplication([])
if args.adapter != "default":            # what force-adapter.py does: the app's default adapter becomes the named kind
    from nodebased import gpu3d
    _state = gpu3d._state
    gpu3d._state = lambda choice=None: _state(args.adapter if (choice or "default") == "default" else choice)
from nodebased import carddisplay, directviewer as dv
from tests.test_carddisplay import card_adapter
from tests.test_directviewer import FakeContext

WIDTH, HEIGHT = (int(x) for x in args.size.split("x"))
SURFACES = tuple(tuple(int(v) for v in item.split("x")) for item in args.surfaces.split(","))
ZOOMS = (0.5, 0.25, 0.125)
VIEWS = ("sRGB", "ACES 2.0")
CAP = "0.35), 1.0, 8.0)"
CAP_WGSL = "0.35), 1., 8.)"


def picture():
    rng = np.random.default_rng(1)
    frame = np.ones((HEIGHT, WIDTH, 4), np.float32)
    frame[..., 0] = np.linspace(0, 1, WIDTH)[None]
    frame[..., 1] = np.linspace(0, 1, HEIGHT)[:, None]
    frame[..., 2] = 0.3
    frame[..., :3] += rng.random((HEIGHT, WIDTH, 3), np.float32) * 0.2
    return frame.astype(np.float16)


def draw_ms(presenter, view, settings, frame, surface, zoom):
    presenter.offer_linear(frame, settings, (0, 0, WIDTH, HEIGHT), (0, 0, WIDTH, HEIGHT))
    presenter.draw((WIDTH / 2, HEIGHT / 2, zoom), surface)
    presenter.wait_idle()
    for _ in range(args.warm):
        presenter.draw((WIDTH / 2, HEIGHT / 2, zoom), surface)
    presenter.wait_idle()
    best = None
    for _ in range(args.batches):
        started = time.perf_counter()
        for _ in range(args.draws):
            presenter.draw((WIDTH / 2, HEIGHT / 2, zoom), surface)
        presenter.wait_idle()
        took = (time.perf_counter() - started) / args.draws * 1000.0
        best = took if best is None else min(best, took)
    return best


def measure(frame, nearest):
    original = carddisplay.FRAGMENT, dv.SHADER
    if nearest:
        assert CAP in carddisplay.FRAGMENT and CAP_WGSL in dv.SHADER
        carddisplay.FRAGMENT = carddisplay.FRAGMENT.replace(CAP, "0.35), 1.0, 1.0)")
        dv.SHADER = dv.SHADER.replace(CAP_WGSL, "0.35), 1., 1.)")
        carddisplay.fragment_source.cache_clear()
    try:
        out = {}
        for surface in SURFACES:
            context = FakeContext(surface)
            presenter = dv.RetainedPresenter(context, adapter=ADAPTER, overlay=False, fill_thread=False)
            try:
                for view in VIEWS:
                    settings = carddisplay.settings_from(0.0, "RGB", "black", view)
                    for zoom in ZOOMS:
                        out[(surface, view, zoom)] = draw_ms(presenter, view, settings, frame, surface, zoom)
            finally:
                presenter.lose_device()
        return out
    finally:
        carddisplay.FRAGMENT, dv.SHADER = original
        carddisplay.fragment_source.cache_clear()


ADAPTER = card_adapter()
frame = picture()
rows = {}
for run in range(args.runs):
    filtered, nearest = measure(frame, False), measure(frame, True)
    for key in filtered:
        rows.setdefault(key, []).append((filtered[key], nearest[key]))
    print(f"run {run + 1}/{args.runs}", file=sys.stderr, flush=True)

table = []
for (surface, view, zoom), runs in rows.items():
    extra = [f - n for f, n in runs]
    table.append({"surface": f"{surface[0]}x{surface[1]}", "view": view, "zoom": f"1:{round(1 / zoom)}",
                  "filtered_ms_median": round(statistics.median(f for f, _ in runs), 3),
                  "nearest_ms_median": round(statistics.median(n for _, n in runs), 3),
                  "extra_ms_median": round(statistics.median(extra), 3), "extra_ms_max": round(max(extra), 3),
                  "extra_ms_runs": [round(x, 3) for x in extra]})
worst = max(row["extra_ms_median"] for row in table)
result = {"adapter": ADAPTER.info.get("device"), "picture": f"{WIDTH}x{HEIGHT} rgba16float", "runs": args.runs,
          "gate_ms": args.gate_ms, "worst_extra_median_ms": worst,
          "worst_extra_max_ms": max(row["extra_ms_max"] for row in table), "gate_met": worst <= args.gate_ms,
          "rows": table}
with open(args.out, "w") as handle:
    json.dump(result, handle, indent=1)
for row in table:
    print(f"{row['surface']:>10} {row['view']:>9} {row['zoom']}  nearest {row['nearest_ms_median']:6.3f}  "
          f"filtered {row['filtered_ms_median']:6.3f}  extra median {row['extra_ms_median']:6.3f}  max {row['extra_ms_max']:6.3f}")
print(f"{result['adapter']}: worst median extra {worst:.3f} ms, worst single {result['worst_extra_max_ms']:.3f} ms, "
      f"gate {args.gate_ms} ms {'met' if result['gate_met'] else 'NOT met'}")
