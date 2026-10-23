"""Where does a frame of the mixed mesh and splat scene spend its time? One JSON, both routes.

Takes the scene a played Render3D hands the renderer (the furnished room, a splat shell, relit by the room's lamp) from a real
window, then renders it at the size asked (1920 x 1080, one sample, as playback does) under an orbiting camera, once per route:

    card        the splat layer drawn over the meshes on the card, in the frame's own submission (``gpusplat.prepare_on_card``)
    processor   the layer drawn on its own, read back and composited on the processor (``gpu3d.SPLATS_ON_CARD = False``)

and splits a frame into mesh render, splat sort, splat colours, splat upload / raster / readback, compositing the layers,
readback and the display picture. ``--scene mixedglass`` adds a smoke puff and a translucent glass sphere: before 10/23 smoke together
with splats raised Unsupported in ``gpu3d.render`` (the CPU drew the frame), so its processor route is the CPU frame, timed once
(``--cpu-frames``); the card route draws the splat layer between the meshes and the smoke march. Stage times are medians in
milliseconds over the frames after a warm-up.

    PYTHONPATH=. QT_QPA_PLATFORM=offscreen flock /tmp/nb-gpu.lock \\
        python tools/profile_3d_mixed_stages.py --out benchmarks/3d-playback/mixed-stages.json
    python tools/profile_3d_mixed_stages.py --self-check      # tiny, a few seconds

``--adapter NAME`` picks the adapter as NODEBASED_ADAPTER does (RTX, Radeon, llvmpipe).
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def capture_scene(splats, name="mixed"):
    """(scene, camera, render args) of the mixed scene as a window's Render3D passes them to gpu3d.render."""
    import tests.isolation  # noqa: F401  (keeps the run out of the real Qt settings)
    import benchmark_3d_playback as bench
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from nodebased import gpu3d
    from nodebased.app import Window
    seen = {}
    real = gpu3d.render

    def grab(*args, **kwargs):
        seen.setdefault("call", (args, kwargs))
        return real(*args, **kwargs)
    gpu3d.render = grab
    try:
        with tempfile.TemporaryDirectory(prefix="nb-stages-") as tmp:
            path = bench.make_splat_file(splats, tmp)
            window = Window(agent_name="stages-" + uuid.uuid4().hex)
            window.resize(1800, 1100)
            window.show()
            ops, view = bench.build_scene(name, 1920, 1080, "auto", splat_path=path)
            for op in ops:
                if op.get("type") == "Render3D":
                    op["params"]["samples"] = 1
            window.agent_command({"op": "batch", "commands": ops})
            window.set_time(first=1, last=240, current=60, fps=24)
            window.agent_command({"op": "view", "id": view})
            end = time.time() + 120
            while (not seen or window.busy or len(window.preview_queue)) and time.time() < end:
                app.processEvents()
                time.sleep(0.01)
    finally:
        gpu3d.render = real
    if not seen:
        raise SystemExit("the window never rendered the scene")
    args, kwargs = seen["call"]
    return args[0], args[1], args[4:], kwargs


class Timers:
    """Wrap functions so each call adds its wall time (ms) to a per-frame dictionary."""

    def __init__(self):
        self.frame = {}
        self.undo = []

    def wrap(self, owner, name, label):
        function = getattr(owner, name)

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.frame[label] = self.frame.get(label, 0.0) + (time.perf_counter() - start) * 1000
        setattr(owner, name, timed)
        self.undo.append((owner, name, function))

    def restore(self):
        for owner, name, function in reversed(self.undo):
            setattr(owner, name, function)
        self.undo.clear()


def median(rows, key):
    values = [row.get(key, 0.0) for row in rows]
    return round(statistics.median(values), 2) if values else None


def profile_route(route, scene, camera, rest, kwargs, width, height, frames, view):
    import numpy as np
    from nodebased import gpu3d, gpupicture, gpusplat, imaging, playback_trace, scene3d
    gpu3d.SPLATS_ON_CARD = route == "card"
    timers = Timers()
    timers.wrap(gpu3d, "_render", "mesh passes (colour, and the depth pass on the processor route)")
    timers.wrap(gpusplat, "_geometry_and_order", "splat sort")
    timers.wrap(gpusplat, "_appearances", "splat colours (relit on the processor)")
    timers.wrap(gpusplat, "_render", "splat layer on its own: upload, raster and readback")
    timers.wrap(gpusplat, "prepare_on_card", "splat preparation on the card route (sort, colours, buffers)")
    rows = []
    kwargs = dict(kwargs, samples=1)
    if route == "processor" and scene.volumes and scene.splats:
        # smoke together with splats raises Unsupported on this route: Render3D's `auto` backend falls back to the CPU frame
        gpu3d.SPLATS_ON_CARD = True
        timers.restore()
        note, size = "the CPU frame: gpu3d.render raised Unsupported for smoke together with splats", (width, height)
        for size in ((width, height), (width // 2, height // 2), (width // 4, height // 4)):
            start = time.perf_counter()
            try:
                scene3d.render(scene, camera, *size, kwargs.get("background", (0, 0, 0, 0)), ambient=kwargs.get("ambient", 0.0),
                               samples=1, volume=kwargs.get("volume"))
            except ValueError as error:          # the CPU reference refuses a smoke march over its lookup budget
                note += f"; refused at {size[0]} x {size[1]} ({error})"[:140]
                continue
            break
        total = (time.perf_counter() - start) * 1000
        note += f"; timed at {size[0]} x {size[1]}"
        return {"frames": 1, "stages_ms": {"whole render call": round(total, 2)}, "frame_ms": round(total, 2),
                "note": note}
    try:
        for index in range(frames + 3):
            angle = index * 0.05
            position = scene3d.Vec3(6.2 * np.cos(angle), 1.6, 6.2 * np.sin(angle))
            eye = replace(camera, transform=replace(camera.transform, position=position))
            timers.frame = {}
            playback_trace.enable()
            gpupicture.clear()
            gpupicture.arm(view)
            start = time.perf_counter()
            try:
                with gpu3d.playing(True):
                    frame = gpu3d.render(scene, eye, width, height, *rest, **kwargs)
            finally:
                gpupicture.disarm()
            total = (time.perf_counter() - start) * 1000
            events = playback_trace.disable()
            marks = {}
            for stamp, _thread, label in events:
                marks.setdefault(label, stamp)
                marks[label + "_last"] = stamp
            picture_start = time.perf_counter()
            imaging.to_qimage(frame, view=view)
            picture = (time.perf_counter() - picture_start) * 1000
            row = dict(timers.frame)
            row["whole render call"] = total
            row["display picture (conversion stage)"] = picture
            if "submit" in marks and "gpu_done_last" in marks:
                row["card: first submit to the last pixels mapped"] = (marks["gpu_done_last"] - marks["submit"]) * 1000
            if "gpu_done_last" in marks and "readback_done_last" in marks:
                row["readback copy"] = (marks["readback_done_last"] - marks["gpu_done_last"]) * 1000
            row["card picture used"] = float(gpupicture.counters["taken"] > 0)
            rows.append(row)
    finally:
        timers.restore()
        gpu3d.SPLATS_ON_CARD = True
    rows = rows[3:]
    keys = sorted({key for row in rows for key in row})
    stages = {key: median(rows, key) for key in keys}
    if route == "processor":
        parts = sum(stages.get(k) or 0 for k in ("mesh passes (colour, and the depth pass on the processor route)",
                                                  "splat layer on its own: upload, raster and readback"))
        stages["compositing the layers and the rest on the processor"] = round(stages["whole render call"] - parts, 2)
    return {"frames": len(rows), "stages_ms": stages, "frame_ms": stages["whole render call"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--splats", type=int, default=20000)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--adapter", default="default")
    parser.add_argument("--scene", default="mixed", choices=("mixed", "mixedglass"))
    parser.add_argument("--routes", default="processor,card")
    parser.add_argument("--out", default="")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        args.size, args.splats, args.frames = "160x90", 400, 4
    if args.adapter != "default":
        os.environ["NODEBASED_ADAPTER"] = args.adapter
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    width, height = (int(v) for v in args.size.lower().split("x"))
    scene, camera, rest, kwargs = capture_scene(args.splats, args.scene)
    from nodebased import gpu3d
    view = "ACES 2.0"
    result = {"scene": args.scene, "size": args.size, "splats": args.splats, "adapter": gpu3d.chosen_adapter_name(),
              "routes": {route: profile_route(route, scene, camera, rest, kwargs, width, height, args.frames, view)
                         for route in args.routes.split(",")}}
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text, flush=True)
    if args.self_check:
        ok = all(route["frame_ms"] > 0 for route in result["routes"].values())
        print("self-check:", "ok" if ok else "FAILED", flush=True)
        os._exit(0 if ok else 1)
    os._exit(0)


if __name__ == "__main__":
    main()
