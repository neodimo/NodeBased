"""Release-media clips recorded from the real app window.

Run on a machine with a display: PYTHONPATH=. python tools/release_media_clips.py OUT_DIR CLIP...
The window is grabbed at a fixed 30 fps. When the GUI thread is busy (a render, a toggle) no grab
can happen, so the last frame is repeated until the clock catches up: the clip plays in real time
and a slow toggle looks slow. Each clip writes an MP4 plus a manifest entry (commit, adapter, steps).
"""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, splats
from nodebased.app import Window
from nodebased.core import Dispatcher

FPS = 30
STEP_LIMIT = 900      # seconds; a command that changes nothing never produces a frame
SIZE = (1920, 1080)


class Recorder:
    def __init__(self, widget, path):
        self.widget, self.start, self.written, self.last = widget, None, 0, None
        self.grab_ms = []
        w, h = SIZE
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)
        self.timer = QTimer(); self.timer.setInterval(1000 // FPS); self.timer.timeout.connect(self.tick)

    def frame(self):
        image = self.widget.grab().toImage().convertToFormat(QImage.Format_RGB888)
        if (image.width(), image.height()) != SIZE:
            image = image.scaled(*SIZE)
        w, h = SIZE
        rows = np.frombuffer(image.constBits(), np.uint8, image.bytesPerLine() * h).reshape(h, -1)
        return rows[:, :w * 3].tobytes()

    def tick(self):
        now = time.perf_counter()
        if self.start is None:
            self.start = now
        due = int((now - self.start) * FPS) + 1
        if self.written >= due:            # Qt's coarse timers fire up to 5% early; a frame per
            return                         # tick would stretch the clip into slow motion
        if self.last is not None:          # the GUI stalled: hold the old picture for the gap
            while self.written < due - 1:
                self.ffmpeg.stdin.write(self.last); self.written += 1
        grabbed = time.perf_counter(); self.last = self.frame()
        self.grab_ms.append((time.perf_counter() - grabbed) * 1000)
        self.ffmpeg.stdin.write(self.last); self.written += 1

    def begin(self):
        self.timer.start(); self.tick()

    def clock(self):
        return time.perf_counter() - self.start

    def finish(self):
        self.timer.stop(); self.tick()
        self.ffmpeg.stdin.close(); self.ffmpeg.wait()
        return self.written / FPS


def graph(nodes, links, view):
    d = Dispatcher()
    for key, kind, params, pos in nodes:
        d.execute(dict(op="create", id=key, type=kind, params=params))
        d.execute(dict(op="move", id=key, pos=list(pos)))
    for key, slot, source in links:
        d.execute(dict(op="connect", id=key, input=slot, source=source))
    d.execute(dict(op="view", id=view))
    return d.document


def bypass_clip():
    """#1: the graph from DiMo's 2026-09-20 bug report. Bypass the Merge (viewer shows B), then
    the Grade (viewer shows the plain checker), each toggled back on."""
    document = graph(
        (("checker", "Checker", {}, (0, 0)), ("grade", "Grade", dict(exposure=2.0), (0, 120)),
         ("fg", "Constant", dict(red=.8, green=.1, blue=.1, alpha=.5), (220, 120)),
         ("merge", "Merge", {}, (110, 240))),
        (("grade", "image", "checker"), ("merge", "A", "fg"), ("merge", "B", "grade")), "merge")
    toggle = lambda key, value: dict(op="disable", id=key, value=value)
    steps = [(None, 1.5), (toggle("merge", True), 1.5), (toggle("merge", False), 1.5),
             (toggle("merge", True), 1.5), (toggle("merge", False), 1.5), (dict(op="view", id="grade"), 1.5),
             (toggle("grade", True), 1.5), (toggle("grade", False), 1.5),
             (toggle("grade", True), 1.5), (toggle("grade", False), 1.5)]
    return document, steps


NELSON = Path("assets/splats/scene.ply")
CREDIT = "Nelson Ghost Town by Paolo Tosolini (superspl.at), CC BY 4.0"


def nelson_graph(width, height, catch=0.0):
    """The Catch shadows demo scene: a red cube and a shadowed sun dropped into the capture."""
    return graph(
        (("nelson", "ReadSplat3D", dict(splat_path=str(NELSON.resolve()), splat_orientation="colmap",
                                        splat_shadow_catch=catch), (0, 0)),
         ("cube", "Cube3D", dict(cube_size=1.0, tx=3.0, ty=.12, tz=9.0, ry=32,
                                 red=.85, green=.12, blue=.1), (200, 0)),
         ("sun", "Light3D", dict(tx=-2, ty=6, tz=14, target_x=3, target_y=.2, target_z=9,
                                 red=1, green=.95, blue=.85, shadows="on"), (400, 0)),
         ("scene", "Scene3D", {}, (200, 140)),
         ("camera", "Camera3D", dict(tx=5.6456, ty=2.1610, tz=14.2390, target_x=-.3802,
                                     target_y=-.3498, target_z=6.2046, fov=50, near=3.0, far=5000.0), (420, 140)),
         ("render", "Render3D", dict(width=width, height=height, ambient=.35, samples=1,
                                     render_backend="cpu"), (300, 280))),
        (("scene", "object0", "nelson"), ("scene", "object1", "cube"), ("scene", "object2", "sun"),
         ("render", "scene", "scene"), ("render", "camera", "camera")), "render")


def catch_clip():
    """#4: Catch shadows swept 0 -> 0.85 on the capture; the capture keeps its look (Relight 0)
    and takes the cube's shadow."""
    catch = lambda v: dict(op="set", id="nelson", param="splat_shadow_catch", value=v)
    steps = [(None, 2.0)] + [(catch(v), 1.2) for v in (.15, .3, .45, .6, .75, .85)] + [(catch(0.0), 1.5), (catch(.85), 2.0)]
    return nelson_graph(960, 540), steps


def progress_clip():
    """#5: a cold 1080p render of the capture, with the progress bar and time left running."""
    return nelson_graph(1920, 1080), [(None, 2.0)]


def galaxy(path, count=200_000, seed=7):
    """A synthetic 200k-splat spiral: dense enough that the CPU path is visibly slow, and no
    capture data involved."""
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 3, count)
    radius = rng.gamma(2.0, .55, count)
    angle = arm * 2 * np.pi / 3 + radius * 1.7 + rng.normal(0, .22, count)
    pos = np.stack([radius * np.cos(angle), rng.normal(0, .06, count) * (1 + radius * .3),
                    radius * np.sin(angle)], 1)
    core = np.exp(-radius * 1.3)[:, None]
    colour = np.clip(core * np.array([1., .85, .55]) + (1 - core) * np.array([.35, .55, 1.])
                     + rng.normal(0, .05, (count, 3)), 0, 1)
    cloud = splats.SplatCloud(pos, np.full((count, 3), .018) * (1 + radius[:, None] * .4),
                              np.tile((1., 0, 0, 0), (count, 1)), np.full(count, .75),
                              ((colour - .5) / splats.C0)[:, None, :], 0)
    splats.write_ply(cloud, path)


def splats_clip(out):
    """#2: the same 200k-splat graph on the CPU, then the GPU, orbiting the camera on each."""
    ply = out / "galaxy-200k.ply"
    if not ply.exists():
        galaxy(ply)
    document = graph(
        (("galaxy", "ReadSplat3D", dict(splat_path=str(ply.resolve()), splat_sh_degree=0), (0, 0)),
         ("scene", "Scene3D", {}, (0, 140)),
         ("camera", "Camera3D", dict(tx=0, ty=2.2, tz=5.5, fov=45), (200, 140)),
         ("render", "Render3D", dict(width=1280, height=720, render_backend="cpu", render_mode="raster"),
          (100, 280))),
        (("scene", "object0", "galaxy"), ("render", "scene", "scene"), ("render", "camera", "camera")), "render")
    orbit = [(5.5 * np.sin(a), 5.5 * np.cos(a)) for a in (.45, .9)]    # about 23 s each on the CPU
    move = lambda x, z: dict(op="batch", commands=[dict(op="set", id="camera", param="tx", value=float(x)),
                                                   dict(op="set", id="camera", param="tz", value=float(z))])
    backend = lambda b: dict(op="set", id="render", param="render_backend", value=b)
    steps = [(None, 1.5)] + [(move(x, z), .4) for x, z in orbit] + [(backend("gpu"), 1.0)]
    steps += [(move(x, z), .15) for x, z in [(5.5 * np.sin(a), 5.5 * np.cos(a)) for a in np.linspace(.9, 0, 16)[1:]]]   # [1:]: the camera is already at .9, and a no-op set renders nothing
    steps += [(move(x, z), .15) for x, z in [(5.5 * np.sin(a), 5.5 * np.cos(a)) for a in np.linspace(0, -.9, 16)[1:]]]
    return document, steps + [(None, 1.5)]


SELECT = {"04-catch-shadows": "nelson", "02-splats-cpu-vs-gpu": "render"}     # the node whose Properties panel the clip shows
CLIPS = {"02-splats-cpu-vs-gpu": (splats_clip, True), "01-bypass": (bypass_clip, True), "04-catch-shadows": (catch_clip, True),
         "05-progress-1080p": (progress_clip, False)}


def idle(window, generation):
    return window.frame_generation > generation and not window.busy


def record(app, out, name, manifest):
    build, warm = CLIPS[name]
    document, steps = build(out) if build is splats_clip else build()
    generation = -1
    window = Window(document=document, **({} if warm else {}))
    window.resize(*SIZE); window.show()
    if name in SELECT:
        window.graph.items_by_id[SELECT[name]].setSelected(True)
    pump = lambda seconds: [app.processEvents() or time.sleep(.005)
                            for _ in range(int(seconds / .005))]
    warmup = time.perf_counter()
    if warm:                         # the clip starts on a finished picture
        while not idle(window, generation):
            pump(.05)
    else:
        pump(.3)
    warmup = time.perf_counter() - warmup
    log = []
    recorder = Recorder(window, out / f"{name}.mp4")
    recorder.begin()
    def wait(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents(); time.sleep(.002)
    for cmd, hold in steps:
        started, at = time.perf_counter(), recorder.clock()
        if cmd is not None:
            generation = window.frame_generation
            window.command(cmd)
        while not idle(window, generation if cmd is not None else -1):
            app.processEvents(); time.sleep(.002)
            if time.perf_counter() - started > STEP_LIMIT:
                raise RuntimeError(f"{name}: no new frame {STEP_LIMIT} s after {cmd}")
        log.append(dict(cmd=cmd, until_picture_ms=round((time.perf_counter() - started) * 1000),
                        cmd_at_s=round(at, 3), picture_at_s=round(recorder.clock(), 3), hold_s=hold))
        print(f"{name}: step {len(log)}/{len(steps)} {log[-1]['until_picture_ms']} ms", flush=True)
        wait(hold)
    seconds = recorder.finish()
    window.saved_document = copy.deepcopy(window.dispatcher.document)  # no unsaved-changes prompt
    window.close(); app.processEvents()
    manifest["files"][f"{name}.mp4"] = dict(
        seconds=round(seconds, 2), fps=FPS, size=list(SIZE), warmup_s=round(warmup, 1), steps=log,
        grab_ms_mean=round(float(np.mean(recorder.grab_ms)), 1),
        grab_ms_p95=round(float(np.percentile(recorder.grab_ms, 95)), 1),
        errors=list(getattr(window, "render_errors", [])),
        credit=CREDIT if name.startswith(("04", "05")) else None)
    print(f"{name}.mp4: {seconds:.1f} s, warm-up {warmup:.1f} s; "
          + ", ".join(f"{(s['cmd'] or {}).get('op', 'hold')} {s['until_picture_ms']} ms" for s in log), flush=True)


def main(out, names):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest-clips.json"
    manifest = json.loads(path.read_text()) if path.exists() else dict(script="tools/release_media_clips.py", files={})
    manifest.update(
        commit=subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
        nodebased_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--", "nodebased"], text=True).strip()),
        adapter=str(gpu3d.adapter_report()))
    for name in names or CLIPS:
        record(app, out, name, manifest)
        path.write_text(json.dumps(manifest, indent=2))     # per clip: a later failure keeps these


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
