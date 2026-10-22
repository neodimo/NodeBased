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
    grade      a Render3D of the cube into a Grade into the Viewer, camera animated

    # private Xvfb display, real window (what the Playback table in docs/PERFORMANCE.md used)
    Xvfb :91 -screen 0 2560x1440x24 &
    DISPLAY=:91 QT_QPA_PLATFORM=xcb flock /tmp/nb-gpu.lock \\
        python tools/benchmark_3d_playback.py --scene cube --out benchmarks/3d-playback/run.json

    # offscreen platform (no display needed)
    QT_QPA_PLATFORM=offscreen python tools/benchmark_3d_playback.py --scene all --size 960x540 --seconds 4

    python tools/benchmark_3d_playback.py --self-check      # offscreen, tiny, a few seconds

``--adapter default|discrete|integrated|cpu`` forces the 3D adapter the way ``run/force-adapter.py`` does
(``cpu`` is llvmpipe). ``--backend auto|gpu|cpu`` sets Render3D's ``render_backend`` (the node's own default is
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

SCENES = ("empty", "cube", "furnished", "glass", "mixed", "grade")
FIELDS = ("adapter", "scene", "size", "backend", "fps", "shown", "interval_ms", "render_ms_status", "status")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="cube", help="one of %s, or 'all'" % ", ".join(SCENES))
    p.add_argument("--size", default="1920x1080", help="Render3D width x height")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=24.0, help="the comp's rate")
    p.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    p.add_argument("--adapter", default="default", choices=("default", "discrete", "integrated", "cpu"))
    p.add_argument("--out", default="", help="JSON file (or a directory: one file per scene)")
    p.add_argument("--field", action="store_true", help="print the compact field JSON (for a machine you cannot reach)")
    p.add_argument("--stages", action="store_true", help="record the per-stage timeline")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--splats", type=int, default=20000)
    p.add_argument("--view", default="", help="display view to play through (default: the window's own, ACES 2.0)")
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
                                     {"red": 0.9, "green": 0.95, "blue": 0.95})})
        add("box", "Cube3D", {"cube_size": 1.1, "tx": 2.0, "ty": 0.55, "tz": -0.2, "ry": 25, "red": 0.7, "green": 0.1, "blue": 0.1,
                              "material": "pbr", "metallic": 0.0, "pbr_roughness": 0.35})
        add("mirror", "Card3D", {"card_width": 2.6, "card_height": 1.8, "ty": 1.6, "tz": -3.9, "material": "pbr", "metallic": 1.0,
                                 "pbr_roughness": 0.0})
        add("shelf", "Card3D", {"card_width": 8, "card_height": 0.6, "ty": 0.06, "tz": -2.4, "rx": -90, "red": 0.6, "green": 0.45, "blue": 0.3})
        add("lamp", "Light3D", {"light_type": "Point", "tx": 0.0, "ty": 4.5, "tz": 0.5, "intensity": 6.0, "shadows": "on"})
        if name == "mixed":
            add("splats", "ReadSplat3D", {"splat_path": str(splat_path), "splat_relight": 1.0, "ty": 0.6, "tz": -1.0})
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

def run_scene(window, app, scene, args, splat_path, width, height):
    import re
    from nodebased import gpu3d

    def spin(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.001)

    ops, view = build_scene(scene, width, height, args.backend, splat_path=splat_path)
    window.agent_command({"op": "batch", "commands": ops})
    last = 240
    window.set_time(first=1, last=last, current=1, fps=args.fps)
    window.playback_proxy.setChecked(False)     # play at the size the artist set
    if args.view:
        window.display_view.setCurrentText(args.view)
    window.agent_command({"op": "view", "id": view})
    t0 = time.perf_counter()
    while (window.busy or len(window.preview_queue)) and time.perf_counter() - t0 < 120:
        spin(0.05)
    warm_ms = None
    window.playback_shown_times.clear()
    shown_before = len(window.playback_shown_times)
    dropped_before = window.playback_dropped_frames
    from nodebased import playback_trace
    playback_trace.enable()
    window.toggle_playback(True)
    start = time.monotonic()
    statuses = []
    while time.monotonic() - start < args.seconds:
        spin(0.1)
        statuses.append(window.statusBar().currentMessage())
    elapsed = time.monotonic() - start
    window.toggle_playback(False)
    events = playback_trace.disable()
    spin(0.3)
    times = [t for t in list(window.playback_shown_times) if start <= t <= start + elapsed]
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
        "proxy_tier_at_end": getattr(window, "playback_tier", None),
    }
    table, jobs, _ = playback_trace.stage_table(events)
    result["worker_jobs"] = jobs
    result["stage_p50_ms"] = {row["stage"]: row["p50_ms"] for row in sorted(table, key=lambda r: -r["p50_ms"])}
    waits = playback_trace.stage_table.series.get("submit to gpu_done", [])
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
        splat_path = make_splat_file(args.splats, tmp) if scene == "mixed" else None
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
    if args.self_check:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        args.scene, args.size, args.seconds, args.backend, args.splats = "cube", "128x72", 2.0, "auto", 200
    if args.adapter != "default":
        os.environ["NODEBASED_ADAPTER"] = args.adapter
    scenes = tuple(s for s in SCENES if s != "glass") if args.scene == "all" else (args.scene,)
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
                   "--splats", str(args.splats), "--view", args.view, "--out", name]
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
