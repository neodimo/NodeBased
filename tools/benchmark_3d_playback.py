"""How fast does a 3D comp play in the real window? One JSON per scene and run.

Plays a 24 fps comp for N seconds (default 10) in the real ``Window`` and records the adapter, the
frames per second the viewer actually showed, the time between shown frames (p50, p95, p99, max), the
Render3D time the status line reports and the last status line itself.

Scenes (every camera animates; frame 1 to 240 swings the camera round the scene):

    empty      an empty scene under an animated camera
    cube       one cube under an animated camera
    furnished  the furnished room (floor, walls, spheres, box, mirror, shelf, lamp) under an animated camera
    glass      the furnished room with a liquid-glass sphere: the GPU raster path has no liquid, so Render3D falls
               back to the CPU rasteriser (about 20 s a frame); measured to show that cost, not part of "all"
    mixed      the furnished room plus a 20,000-splat shell under an animated camera
    mixedglass the mixed scene plus a smoke puff (Plume3D) and a translucent glass sphere: smoke together with splats and a
               translucent mesh together with splats, both of which the card draws since 10/23 (Lane 4, gaps step A)
    grade      a Render3D of the cube into a Grade into the Viewer, camera animated

    # private Xvfb display, real window (what the Playback table in docs/PERFORMANCE.md used)
    Xvfb :91 -screen 0 2560x1440x24 &
    DISPLAY=:91 QT_QPA_PLATFORM=xcb flock /tmp/nb-gpu.lock \\
        python tools/benchmark_3d_playback.py --scene cube --out benchmarks/3d-playback/run.json

    # offscreen platform (no display needed)
    QT_QPA_PLATFORM=offscreen python tools/benchmark_3d_playback.py --scene all --size 960x540 --seconds 4

    python tools/benchmark_3d_playback.py --self-check      # offscreen, tiny, a few seconds

``--proxy on`` plays as the app ships, with "Proxy while playing" checked: a frame that misses the comp's rate steps the picture
down a resolution tier (the JSON then says how many pictures were drawn at each tier, the rate at the tier playback ended on and what
the time row read). After the run the tool stops playback and waits for the full-quality picture: ``settle_s`` is the time from the
stop to that picture on screen. Without ``--proxy`` it plays at the size set, the raw cost.

``--adapter NAME`` runs on the adapter whose name contains NAME (RTX, Radeon, llvmpipe), as NODEBASED_ADAPTER does. ``--backend auto|gpu|cpu`` sets Render3D's ``render_backend`` (the node's own default is
``cpu``). ``--field`` prints one JSON object of the fields worth pasting back from another machine.
``--stages`` also records the per-stage timeline (see ``nodebased.playback_trace``) when the build has it.
Run it under /tmp/nb-gpu.lock when the card matters: it needs the whole card for the length of the run.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SMOKE_SCALE = 1.2    # the puff of "mixedglass", in metres across (the plume is a unit cube); 2.4 costs about 20 ms more a frame on the RTX 3080 Ti
SCENES = ("empty", "cube", "furnished", "glass", "mixed", "mixedglass", "grade")
FIELDS = ("adapter", "scene", "size", "backend", "proxy", "fps", "shown", "interval_ms", "render_ms_status", "tier_pictures",
          "fps_at_end_tier", "time_row", "settle_s", "status")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="cube", help="one of %s, a comma list of them, or 'all' (all but glass)" % ", ".join(SCENES))
    p.add_argument("--size", default="1920x1080", help="Render3D width x height")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=24.0, help="the comp's rate")
    p.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    p.add_argument("--adapter", default="default",
                   help="a part of an adapter's name (RTX, Radeon, llvmpipe), as NODEBASED_ADAPTER takes it; default: the app's choice")
    p.add_argument("--out", default="", help="JSON file (or a directory: one file per scene)")
    p.add_argument("--field", action="store_true", help="print the compact field JSON (for a machine you cannot reach)")
    p.add_argument("--stages", action="store_true", help="record the per-stage timeline")
    p.add_argument("--events", default="", help="write the raw playback trace events (monotonic seconds, thread, label) to this JSON file")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--splats", type=int, default=20000)
    p.add_argument("--once", action="store_true",
                   help="also time 12 single Render3D frames through the evaluator (no playback): render_once_ms")
    p.add_argument("--samples", type=int, default=0, help="Render3D samples (default: the node's own, 2)")
    p.add_argument("--view", default="", help="display view to play through (default: the window's own, ACES 2.0)")
    p.add_argument("--proxy", default="off", choices=("off", "on"),
                   help="on: play as the app ships, with \"Proxy while playing\" checked (a frame that misses the comp's rate steps "
                        "the picture down a tier); off: play at the size set, the raw cost (default)")
    p.add_argument("--settle-wait", type=float, default=30.0, help="seconds to wait after stopping for the full-quality picture")
    p.add_argument("--smoke-scale", type=float, default=0.0, help="\"mixedglass\": the puff's size in metres (default SMOKE_SCALE)")
    p.add_argument("--first-seconds", type=float, default=0.0,
                   help="record the start: seconds from opening the scene to the first picture, every frame interval of the first N seconds "
                        "of playback (3 is the lane's number) and the pipeline and buffer builds that fall in them; --replay repeats the "
                        "playback from frame 1 in the same process (warm)")
    p.add_argument("--play-at", default="settled", choices=("settled", "first"),
                   help="when playback starts: after the scene's own pictures are done (settled, the default) or as soon as the first picture "
                        "is on screen, as an artist presses play (first; the refinement still running is cancelled by the play)")
    p.add_argument("--replay", action="store_true", help="with --first-seconds: stop, rewind to frame 1 and play the first seconds again (warm)")
    p.add_argument("--first-frame-wait", type=float, default=120.0,
                   help="seconds to wait for the stopped first frame before playing (0: play at once; \"mixedglass\" needs it, its stopped "
                        "frame is the processor's, which is refused at 1080p)")
    return p.parse_args(argv)


def percentile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] if v else None


def distribution(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "p50": round(percentile(values, 50), 2), "p95": round(percentile(values, 95), 2),
            "p99": round(percentile(values, 99), 2), "max": round(max(values), 2),
            "mean": round(statistics.fmean(values), 2)}


# ---- scenes, as the agent command ops the window understands -------------------------------------------------

class Doc:
    def __init__(self):
        self.ops = []

    def create(self, name, kind, params=None):
        self.ops.append({"op": "create", "id": name, "type": kind, "params": params or {}})

    def connect(self, dest, slot, source):
        self.ops.append({"op": "connect", "id": dest, "input": slot, "source": source})

    def key(self, node, param, frame, value):
        self.ops.append({"op": "set_key", "id": node, "param": param, "frame": int(frame), "value": value})


def _tree(doc, name, members):
    """Scene3D nodes hold eight members each; nest them so a room of any size reaches Render3D as one scene."""
    layer, level = list(members), 0
    while len(layer) > 8:
        merged = []
        for i in range(0, len(layer), 8):
            node = f"{name}_group{level}_{i // 8}"
            doc.create(node, "Scene3D")
            for slot, member in enumerate(layer[i:i + 8]):
                doc.connect(node, f"object{slot}", member)
            merged.append(node)
        layer, level = merged, level + 1
    doc.create(name, "Scene3D")
    for slot, member in enumerate(layer):
        doc.connect(name, f"object{slot}", member)


def _camera(doc, radius, height, target_y, last):
    doc.create("camera", "Camera3D", {"tx": radius, "ty": height, "tz": 0.0, "target_y": target_y, "focal": 28.0})
    quarter = last // 4
    for step, (x, z) in enumerate(((radius, 0.0), (0.0, radius), (-radius, 0.0), (0.0, -radius), (radius, 0.0))):
        doc.key("camera", "tx", 1 + step * quarter, x)
        doc.key("camera", "tz", 1 + step * quarter, z)


def build_scene(name, width, height, backend, last=240, splat_path=None):
    """Return (ops, view node id). The camera orbits the scene over frames 1..last."""
    doc = Doc()
    render = {"width": width, "height": height, "samples": 2, "render_backend": backend, "ambient": 0.2}
    if name == "empty":
        doc.create("scene", "Scene3D")
        _camera(doc, 5.0, 1.5, 0.0, last)
    elif name in ("cube", "grade"):
        doc.create("cube", "Cube3D", {"cube_size": 2.0, "red": 0.8, "green": 0.5, "blue": 0.3})
        doc.create("lamp", "Light3D", {"light_type": "Directional", "tx": 2.0, "ty": 4.0, "tz": 3.0})
        _tree(doc, "scene", ["cube", "lamp"])
        _camera(doc, 5.0, 1.5, 0.0, last)
    else:
        members = []

        def add(node, kind, params):
            doc.create(node, kind, params)
            members.append(node)
        add("floor", "Card3D", {"card_width": 12, "card_height": 12, "rx": -90, "red": 0.75, "green": 0.72, "blue": 0.68,
                                "material": "pbr", "metallic": 0.0, "pbr_roughness": 0.25})
        add("back", "Card3D", {"card_width": 12, "card_height": 6, "ty": 3, "tz": -4, "red": 0.8, "green": 0.8, "blue": 0.78})
        add("left", "Card3D", {"card_width": 12, "card_height": 6, "tx": -6, "ty": 3, "ry": 90, "red": 0.75, "green": 0.2, "blue": 0.15})
        add("right", "Card3D", {"card_width": 12, "card_height": 6, "tx": 6, "ty": 3, "ry": 90, "red": 0.2, "green": 0.55, "blue": 0.25})
        add("metal", "Sphere3D", {"sphere_radius": 0.8, "tx": -1.6, "ty": 0.8, "tz": 0.4, "rows": 24, "columns": 48,
                                  "red": 0.95, "green": 0.75, "blue": 0.4, "material": "pbr", "metallic": 1.0, "pbr_roughness": 0.3})
        add("glass", "Sphere3D", {"sphere_radius": 0.8, "tx": 0.2, "ty": 0.8, "tz": 1.2, "rows": 24, "columns": 48,
                                  **({"material": "liquid", "ior": 1.5} if name == "glass" else
                                     {"red": 0.9, "green": 0.95, "blue": 0.95, "alpha": 0.35} if name == "mixedglass" else
                                     {"red": 0.9, "green": 0.95, "blue": 0.95})})
        add("box", "Cube3D", {"cube_size": 1.1, "tx": 2.0, "ty": 0.55, "tz": -0.2, "ry": 25, "red": 0.7, "green": 0.1, "blue": 0.1,
                              "material": "pbr", "metallic": 0.0, "pbr_roughness": 0.35})
        add("mirror", "Card3D", {"card_width": 2.6, "card_height": 1.8, "ty": 1.6, "tz": -3.9, "material": "pbr", "metallic": 1.0,
                                 "pbr_roughness": 0.0})
        add("shelf", "Card3D", {"card_width": 8, "card_height": 0.6, "ty": 0.06, "tz": -2.4, "rx": -90, "red": 0.6, "green": 0.45, "blue": 0.3})
        add("lamp", "Light3D", {"light_type": "Point", "tx": 0.0, "ty": 4.5, "tz": 0.5, "intensity": 6.0, "shadows": "on"})
        if name in ("mixed", "mixedglass"):
            add("splats", "ReadSplat3D", {"splat_path": str(splat_path), "splat_relight": 1.0, "ty": 0.6, "tz": -1.0})
        if name == "mixedglass":
            add("smoke", "Plume3D", {"plume_resolution": 32, "tx": -0.6, "ty": 0.0, "tz": 0.4, "uscale": SMOKE_SCALE})
        _tree(doc, "scene", members)
        _camera(doc, 6.2, 1.6, 0.95, last)
    doc.create("render", "Render3D", render)
    doc.connect("render", "scene", "scene")
    doc.connect("render", "camera", "camera")
    view = "render"
    if name == "grade":
        doc.create("bench_grade", "Grade", {"exposure": 0.2, "multiply": 1.1})
        doc.connect("bench_grade", "image", "render")
        doc.create("bench_view", "Viewer")
        doc.connect("bench_view", "image", "bench_grade")
        view = "bench_view"
    return doc.ops, view


def make_splat_file(count, directory):
    from benchmark_pathtrace import splat_shell
    from nodebased import splats
    path = Path(directory) / "shell.ply"
    splats.write_ply(splat_shell(count), str(path))
    return path


# ---- the run -------------------------------------------------------------------------------------------------

def check_shown_picture(window):
    """The picture on screen against what the processor makes of the float frame it was made from: the largest difference in
    8-bit levels (1 or less is agreement). Catches a picture that is not the frame's own."""
    import numpy as np
    from nodebased import gpupicture, imaging
    image, frame = getattr(window.viewer, "last_display_image", None), window.frame
    if image is None or frame is None or window.proxy.currentData() not in (1, None) or (image.width(), image.height()) != frame.shape[1::-1]:
        return {"checked": False}
    gpupicture.clear()
    saved, imaging.CARD_PICTURE = imaging.CARD_PICTURE, False
    try:
        expected = imaging.to_qimage(frame, view=window.display_view.currentText())
    finally:
        imaging.CARD_PICTURE = saved

    def rows(picture):
        raw = np.frombuffer(picture.constBits(), np.uint8, count=picture.sizeInBytes()).reshape(picture.height(), picture.bytesPerLine())
        return raw[:, :picture.width() * 3].astype(int)
    shown = image.convertToFormat(expected.format()) if image.format() != expected.format() else image
    return {"checked": True, "max_difference_levels": int(np.abs(rows(shown) - rows(expected)).max())}



class BuildTrace:
    """Times every pipeline, shader module, buffer and texture the renderer asks the card for, and every queue upload, while
    installed (monkeypatches the wgpu device and queue classes; the product code is not touched)."""
    NAMES = {"GPUDevice": ("create_render_pipeline", "create_compute_pipeline", "create_shader_module", "create_buffer",
                           "create_buffer_with_data", "create_texture"),
             "GPUQueue": ("write_buffer", "write_texture")}

    def __init__(self):
        self.rows = []            # (monotonic start, kind, milliseconds, bytes)
        self.saved = []

    def install(self):
        from wgpu.backends import wgpu_native
        for class_name, names in self.NAMES.items():
            cls = getattr(wgpu_native, class_name)
            for name in names:
                original = getattr(cls, name)
                self.saved.append((cls, name, original))

                def wrapper(this, *args, _o=original, _n=name, **kwargs):
                    started = time.monotonic()
                    try:
                        return _o(this, *args, **kwargs)
                    finally:
                        size = 0
                        data = kwargs.get("data", args[0] if _n == "create_buffer_with_data" and args else None)
                        if _n == "write_buffer":
                            data = kwargs.get("data", args[2] if len(args) > 2 else None)
                        if _n == "create_buffer":
                            size = int(kwargs.get("size", args[0] if args else 0) or 0)
                        elif data is not None:
                            try:
                                size = memoryview(data).nbytes
                            except TypeError:
                                size = 0
                        self.rows.append((started, _n, (time.monotonic() - started) * 1000, size))
                setattr(cls, name, wrapper)
        return self

    def remove(self):
        for cls, name, original in self.saved:
            setattr(cls, name, original)
        self.saved = []

    def within(self, begin, end):
        """Per kind: count, total milliseconds, total bytes, and the single slowest call, for builds between two monotonic times."""
        out = {}
        for started, kind, ms, size in self.rows:
            if begin <= started <= end:
                row = out.setdefault(kind, {"count": 0, "total_ms": 0.0, "bytes": 0, "slowest_ms": 0.0})
                row["count"] += 1
                row["total_ms"] += ms
                row["bytes"] += size
                row["slowest_ms"] = max(row["slowest_ms"], ms)
        return {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in out.items()}


def first_seconds_report(times, start, span, budget_ms, events, builds):
    """The first ``span`` seconds of one playback: when the pictures came, the frame intervals, and what each slow frame
    spent its time on (the stage marks and the builds that fell inside it)."""
    from nodebased import playback_trace
    inside = [t for t in times if start <= t <= start + span]
    intervals = [(b - a) * 1000 for a, b in zip(inside, inside[1:])]
    report = {
        "span_s": span, "shown": len(inside), "fps": round(len(inside) / span, 2),
        "first_picture_after_play_ms": round((inside[0] - start) * 1000, 1) if inside else None,
        "frame_budget_ms": round(budget_ms, 1),
        "shown_at_ms": [round((t - start) * 1000, 1) for t in inside],
        "intervals_ms": [round(v, 1) for v in intervals],
        "interval_max_ms": round(max(intervals), 1) if intervals else None,
        "intervals_over_2x_budget": sum(1 for v in intervals if v > 2 * budget_ms),
    }
    # frames of the worker, in order, with their stage durations and the builds that fell inside each
    frames, current = [], None
    for stamp, _thread, label in events:
        if label == "worker_start":
            current = {"worker_start": stamp}
            frames.append(current)
        elif current is not None:
            current.setdefault(label, stamp)
            if label in ("submit", "gpu_done", "readback_done"):
                current[label] = stamp
    rows = []
    for number, f in enumerate(frames):
        if not (start <= f["worker_start"] <= start + span) or "handover" not in f:
            continue
        marks = [(n, f[n]) for n in ("worker_start", "render_begin", "scene_ready", "upload_done", "submit", "gpu_done", "readback_done",
                                    "resolve_done", "put_begin", "cache_put", "emit", "handover") if n in f]
        stages = {f"{a} to {b}": round((tb - ta) * 1000, 1) for (a, ta), (b, tb) in zip(marks, marks[1:])}
        row = {"frame": number, "start_ms": round((f["worker_start"] - start) * 1000, 1),
               "whole_ms": round((f["handover"] - f["worker_start"]) * 1000, 1), "stages_ms": stages}
        if builds is not None:
            row["builds"] = builds.within(f["worker_start"], f["handover"])
        rows.append(row)
    report["worker_frames"] = rows
    worst = sorted(rows, key=lambda r: -r["whole_ms"])[:5]
    report["slowest_worker_frames"] = [{"frame": r["frame"], "start_ms": r["start_ms"], "whole_ms": r["whole_ms"],
                                        "top_stage": max(r["stages_ms"], key=r["stages_ms"].get) if r["stages_ms"] else "",
                                        "builds": r.get("builds", {})} for r in worst]
    if builds is not None:
        report["builds"] = builds.within(start, start + span)
    return report


def run_scene(window, app, scene, args, splat_path, width, height):
    import re
    from nodebased import gpu3d

    def spin(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.001)

    ops, view = build_scene(scene, width, height, args.backend, splat_path=splat_path)
    builds = BuildTrace().install() if args.first_seconds else None
    t_open = time.monotonic()
    if args.samples:
        for op in ops:
            if op.get("type") == "Render3D":
                op["params"]["samples"] = args.samples
    window.agent_command({"op": "batch", "commands": ops})
    last = 240
    window.set_time(first=1, last=last, current=1, fps=args.fps)
    window.playback_proxy.setChecked(args.proxy == "on")     # off: play at the size the artist set
    shown_tiers = []                      # (monotonic time, proxy tier) of every picture the viewer was handed
    real_show = window._show_image

    def show_image(image, scale, *rest, **more):
        shown_tiers.append((time.monotonic(), int(scale)))
        return real_show(image, scale, *rest, **more)
    window._show_image = show_image
    readies = []                          # (monotonic time, request.playing, request.display, request.tier, picture shown?, status)

    def ready(payload, frame, image, status, render_region=None):
        request = payload[0]
        readies.append((time.monotonic(), bool(request.playing), bool(request.display), int(request.tier), frame is not None, status))
        return window.__class__.preview_ready(window, payload, frame, image, status, render_region)
    window.signals.finished.disconnect()
    window.signals.finished.connect(ready)
    if args.view:
        window.display_view.setCurrentText(args.view)
    t_view = time.monotonic()
    window.agent_command({"op": "view", "id": view})
    t0 = time.perf_counter()
    if args.play_at == "first":
        while not any(t >= t_view for t, _tier in shown_tiers) and time.perf_counter() - t0 < args.first_frame_wait:
            spin(0.005)
    else:
        while (window.busy or len(window.preview_queue)) and time.perf_counter() - t0 < args.first_frame_wait:
            spin(0.05)
    first_picture = None
    if args.first_seconds:
        later = [t for t, _tier in shown_tiers if t >= t_view]
        first_picture = {"scene_build_s": round(t_view - t_open, 3),
                         "open_to_first_picture_s": round(later[0] - t_open, 3) if later else None,
                         "view_to_first_picture_s": round(later[0] - t_view, 3) if later else None,
                         "view_to_ready_s": round(time.perf_counter() - t0, 3), "play_at": args.play_at,
                         "builds_before_first_picture": builds.within(t_open, later[0]) if later else {}}
    warm_ms = None
    window.playback_shown_times.clear()
    shown_before = len(window.playback_shown_times)
    dropped_before = window.playback_dropped_frames
    once = None
    if args.once:
        document = window.dispatcher.document
        times = []
        for frame in range(1, 14):
            started = time.perf_counter()
            window.evaluator.evaluate(document, "render", frame=frame, tier=1)
            times.append((time.perf_counter() - started) * 1000)
        once = distribution(times[1:])       # the first frame builds pipelines
    from nodebased import playback_trace
    playback_trace.enable()
    window.toggle_playback(True)
    start = time.monotonic()
    statuses = []
    first_times = None
    while time.monotonic() - start < args.seconds:
        spin(0.1)
        statuses.append(window.statusBar().currentMessage())
        if args.first_seconds and first_times is None and time.monotonic() - start >= args.first_seconds + 0.5:
            first_times = list(window.playback_shown_times)
    elapsed = time.monotonic() - start
    playing_tier = int(window.proxy.currentData() or 1)
    picture_check = check_shown_picture(window)
    window.toggle_playback(False)
    stopped = time.monotonic()
    events = playback_trace.disable()
    cold = None
    if args.first_seconds:
        cold = first_seconds_report(first_times or list(window.playback_shown_times), start, args.first_seconds, 1000.0 / args.fps, events, builds)
        cold["first_picture"] = first_picture
    if args.events:
        Path(args.events).write_text(json.dumps(events))
    main_times = list(window.playback_shown_times)
    warm = None
    if args.first_seconds and args.replay:
        # Warm: the pipelines, buffers and caches the first pass built are resident. Rewind to frame 1 and play the first seconds again.
        spin(0.5)
        window.set_time(current=1)
        waited = time.perf_counter()
        while (window.busy or len(window.preview_queue)) and time.perf_counter() - waited < args.first_frame_wait:
            spin(0.05)
        window.playback_shown_times.clear()
        playback_trace.enable()
        window.toggle_playback(True)
        warm_start = time.monotonic()
        warm_times = None
        while time.monotonic() - warm_start < args.first_seconds + 0.5:
            spin(0.1)
        warm_times = list(window.playback_shown_times)
        window.toggle_playback(False)
        warm = first_seconds_report(warm_times, warm_start, args.first_seconds, 1000.0 / args.fps, playback_trace.disable(), builds)
        stopped = time.monotonic()
    if builds is not None:
        builds.remove()
    settle_s, settle_status = None, ""
    deadline = stopped + args.settle_wait
    while time.monotonic() < deadline:
        spin(0.02)
        # the full-quality picture: a finished request that is not a playing one, drawn at the artist's own tier
        refined = [r for r in readies if r[0] >= stopped and not r[1] and r[2] and r[3] == 1 and r[4]]
        if refined and not window.busy and not len(window.preview_queue):
            settle_s, settle_status = round(refined[-1][0] - stopped, 2), refined[-1][5]
            break
    if settle_s is None:
        settle_status = window.statusBar().currentMessage()
    spin(0.3)
    times = [t for t in main_times if start <= t <= start + elapsed]
    intervals = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    status = next((s for s in reversed(statuses) if s), "")
    render_ms = [int(m.group(1)) for m in (re.search(r"(\d+) ms", s) for s in statuses if s) if m]
    try:
        adapter = gpu3d.chosen_adapter_name() if gpu3d.available() else "none (CPU rasteriser)"
    except Exception as exc:       # an adapter error should not lose the run
        adapter = f"error: {exc}"
    result = {
        "scene": scene, "size": [width, height], "backend": args.backend, "view": window.display_view.currentText(), "seconds": round(elapsed, 2),
        "adapter": adapter, "platform": os.environ.get("QT_QPA_PLATFORM", ""), "display": os.environ.get("DISPLAY", ""),
        "comp_fps": args.fps, "shown": len(times), "fps": round(len(times) / elapsed, 2) if elapsed else 0.0,
        "interval_ms": distribution(intervals), "render_ms_status": distribution(render_ms),
        "dropped": window.playback_dropped_frames - dropped_before, "status": status,
        "proxy": args.proxy, "tier_at_end": playing_tier,
    }
    in_run = [(t, tier) for t, tier in shown_tiers if start <= t <= start + elapsed]
    result["tier_pictures"] = {str(tier): sum(1 for _, k in in_run if k == tier) for tier in sorted({k for _, k in in_run})}
    last_tier = in_run[-1][1] if in_run else 1
    final = [t for t, tier in in_run if tier == last_tier]
    if len(final) > 2:
        # the rate at the tier playback ended on: the span of the pictures drawn at it
        result["fps_at_end_tier"] = round((len(final) - 1) / (final[-1] - final[0]), 2) if final[-1] > final[0] else 0.0
    result["time_row"] = window.realtime_label.text()
    result["settle_s"] = settle_s
    result["settle_status"] = settle_status
    if once:
        result["render_once_ms"] = once
        result["render_samples"] = args.samples or "node default (2)"
    table, jobs = [], 0
    if hasattr(playback_trace, "stage_table"):         # a checkout from before the stage table still runs the tool
        table, jobs, _ = playback_trace.stage_table(events)
    result["worker_jobs"] = jobs
    if hasattr(playback_trace, "frame_summary"):     # Lane 2's per-frame stage marks: how the stages add up to the interval and how much overlaps
        result["frame_timeline"] = playback_trace.frame_summary(playback_trace.frame_timeline(events))
    if cold is not None:
        result["first_seconds"] = {"cold": cold}
        if warm is not None:
            result["first_seconds"]["warm"] = warm
    result["picture_check"] = picture_check
    try:
        from nodebased import gpupicture
        result["card_pictures"] = dict(gpupicture.counters)      # pictures the render card made / the conversion stage took
    except ImportError:
        pass
    result["stage_p50_ms"] = {row["stage"]: row["p50_ms"] for row in sorted(table, key=lambda r: -r["p50_ms"])}
    waits = getattr(getattr(playback_trace, "stage_table", None), "series", {}).get("submit to gpu_done", [])
    if waits:
        # Another process using the same card shows up here as a step: the wait for a trivial render jumps from a few
        # milliseconds to a time slice or more and back. A run where the spread is over 2.5 times is not a clean one.
        result["gpu_wait_ms"] = {"min": min(waits), "p50": statistics.median(waits), "max": max(waits),
                                 "shared_card_suspected": max(waits) > 2.5 * max(min(waits), 1.0)}
    return result


def run_one(args, scene):
    """One scene in this process; the window is never closed (closing a window with unsaved nodes asks first)."""
    import tests.isolation  # noqa: F401  (keeps the run out of the real Qt settings)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from nodebased.app import Window
    width, height = (int(v) for v in args.size.lower().split("x"))
    with tempfile.TemporaryDirectory(prefix="nb-play-") as tmp:
        splat_path = make_splat_file(args.splats, tmp) if scene in ("mixed", "mixedglass") else None
        window = Window(agent_name="play3d-" + uuid.uuid4().hex)
        window.resize(1800, 1100)
        window.move(0, 0)
        window.show()
        end = time.perf_counter() + 0.5
        while time.perf_counter() < end:
            app.processEvents()
        return run_scene(window, app, scene, args, splat_path, width, height)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke_scale:
        global SMOKE_SCALE
        SMOKE_SCALE = args.smoke_scale
    if args.self_check:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        args.scene, args.size, args.seconds, args.backend, args.splats = "cube", "128x72", 2.0, "auto", 200
    if args.adapter != "default":
        os.environ["NODEBASED_ADAPTER"] = args.adapter
    scenes = tuple(s for s in SCENES if s != "glass") if args.scene == "all" else tuple(args.scene.split(","))
    for scene in scenes:
        if scene not in SCENES:
            raise SystemExit(f"unknown scene {scene!r}; choose from {', '.join(SCENES)} or all")
    if len(scenes) > 1:      # one process per scene: a fresh window, caches and adapter state each time
        import subprocess
        results = []
        for scene in scenes:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                name = handle.name
            cmd = [sys.executable, __file__, "--scene", scene, "--size", args.size, "--seconds", str(args.seconds),
                   "--fps", str(args.fps), "--backend", args.backend, "--adapter", args.adapter,
                   "--splats", str(args.splats), "--view", args.view, "--samples", str(args.samples), "--out", name]
            cmd += ["--once"] if args.once else []
            cmd += ["--proxy", args.proxy, "--settle-wait", str(args.settle_wait), "--play-at", args.play_at]
            cmd += ["--first-seconds", str(args.first_seconds)] if args.first_seconds else []
            cmd += ["--replay"] if args.replay else []
            subprocess.run(cmd, check=False, env=dict(os.environ, PYTHONPATH=str(ROOT)))
            try:
                results.append(json.loads(Path(name).read_text()))
            except (OSError, ValueError):
                results.append({"scene": scene, "error": "the run produced no result"})
            finally:
                Path(name).unlink(missing_ok=True)
    else:
        results = [run_one(args, scenes[0])]
        print(json.dumps({k: results[0][k] for k in ("scene", "adapter", "fps", "interval_ms", "status")}), flush=True)
    if args.out:
        out = Path(args.out)
        if out.suffix == ".json":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results if len(results) > 1 else results[0], indent=1))
        else:
            out.mkdir(parents=True, exist_ok=True)
            for r in results:
                (out / f"{r['scene']}.json").write_text(json.dumps(r, indent=1))
    if args.field:
        print(json.dumps([{k: r.get(k) for k in FIELDS} for r in results]))
    if args.self_check:
        bad = [r for r in results if r.get("shown", 0) < 1]
        print("self-check:", "FAILED, no frame shown" if bad else "ok", flush=True)
        os._exit(1 if bad else 0)
    os._exit(0)


if __name__ == "__main__":
    main()
