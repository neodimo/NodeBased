"""Frame time of the live 3D viewport with and without the camera's lens (Rendering 13 step G2).

    python tools/benchmark_viewport_lens.py [default|integrated|cpu] [--width 1920] [--height 1080] [--json FILE]

The viewport looks through a camera, once with no lens (the viewport's frame as it was) and once with a barrel
lens (the overscan picture drawn, then bent by Redistort on the same adapter). Each is painted a few times to warm
up (shader builds, buffers) and then fifteen times; the figure is the median of `widget.grab()`, the whole paint:
scene evaluation from cache, the GPU render and its read back, the lens warp, the Qt image and the overlays. The warp
alone is timed on the same adapter on a picture of the overscan size. The adapter is forced the way
`run/force-adapter.py` forces it for the tests: `default` is the RTX 3080 Ti on this workstation, `integrated` the
Radeon, `cpu` the software one. Run it under `flock -s /tmp/nb-gpu.lock` (shared, like a suite) unless nothing else
needs the card.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LENS = dict(lens_k1=-0.06, lens_k2=0.004)


def force(kind):
    from nodebased import gpu3d
    original = gpu3d._state
    gpu3d._state = lambda choice=None: original(kind if (choice or "default") == "default" else choice)
    return gpu3d._state()["info"].get("device", kind)


def graph(**lens):
    from nodebased.core import Dispatcher
    d = Dispatcher()
    for key, kind, params in (("ball", "Sphere3D", dict(sphere_radius=1.5, tz=-6.0)),
                              ("small", "Sphere3D", dict(sphere_radius=0.4, tx=-2.4, ty=1.2, tz=-6.5)),
                              ("light", "Light3D", {}),
                              ("cam", "Camera3D", dict(tz=0.0, target_z=-1.0, **lens)),
                              ("scene", "Scene3D", {}), ("render", "Render3D", {})):
        d.execute(dict(op="create", id=key, type=kind, params=params))
    for slot, source in (("object0", "ball"), ("object1", "small"), ("object2", "light")):
        d.execute(dict(op="connect", id="scene", input=slot, source=source))
    d.execute(dict(op="connect", id="render", input="scene", source="scene"))
    d.execute(dict(op="connect", id="render", input="camera", source="cam"))
    d.execute({"op": "view", "id": "render"})
    return d


def median_ms(call, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        times.append((time.perf_counter() - started) * 1000.0)
    return round(statistics.median(times), 1)


def measure(width, height, repeats=15):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from nodebased import scene3d, viewportlens
    from nodebased.viewport3d import Viewport3D
    results = {}
    for name, lens in (("no lens", {}), ("lens", LENS)):
        widget = Viewport3D()
        widget.setMinimumSize(0, 0)
        widget.resize(width, height)
        widget.set_document(graph(**lens).document)
        widget.look_through = True
        widget.show()
        for _ in range(3):
            widget.grab()
        app.processEvents()
        results[f"viewport {name}"] = median_ms(widget.grab, repeats)
        results[f"drew through the lens ({name})"] = widget._lens_drawn
        widget.close()
    camera = scene3d.Camera(**LENS)
    frame = viewportlens.frame_for(camera, width, height)
    picture = np.random.default_rng(1).integers(0, 256, (frame.render_height, frame.render_width, 4), dtype=np.uint8)
    for backend in ("gpu", "cpu"):
        viewportlens.redistort(frame, picture, backend=backend)
        results[f"warp only, {backend}"] = median_ms(lambda: viewportlens.redistort(frame, picture, backend=backend),
                                                     repeats if backend == "gpu" else 3)
    results["overscan picture"] = f"{frame.render_width}x{frame.render_height}"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("adapter", nargs="?", default="default", choices=("default", "integrated", "cpu"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--json")
    args = parser.parse_args()
    device = force(args.adapter)
    table = {"adapter": device, "size": f"{args.width}x{args.height}", **measure(args.width, args.height)}
    print(json.dumps(table))
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
