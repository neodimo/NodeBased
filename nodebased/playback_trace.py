"""Per-stage stamps for one played frame (tools/profile_3d_playback.py reads them).

``mark(label)`` appends ``(time.monotonic(), thread name, label)`` while a profiler has called ``enable()``;
it is one attribute test when nobody is looking. Stages, in the order a played 3D frame meets them:

    tick                 Window.playback_tick (Qt thread): the playhead moved
    worker_start         the worker picked the request up
    render_begin         gpu3d.render entered: the scene and camera are evaluated (scene evaluation = the time since worker_start)
    scene_ready          the per-frame CPU work for the scene is done (shadow data, vertex packing)
    upload_done          buffers, textures, pipelines and bind groups exist (scene upload or reuse)
    submit               the command buffer went to the queue (render submit)
    gpu_done             the card finished and the staging buffer is mapped (GPU done)
    readback_done        the pixels are in a NumPy array (readback)
    resolve_done         supersample average and background composite finished (resolve)
    render_end           gpu3d.render returned
    put_begin            the frame is out of the renderer and the node graph; the display cache put starts
    cache_put            the display cache took the frame
    emit                 the worker handed the frame to the Qt thread
    handover             Window._show_image returned on the Qt thread (hand-over to the display)
"""
import collections
import statistics
import threading
import time

_events = None
_lock = threading.Lock()


def enable():
    global _events
    _events = []


def disable():
    """Stop recording and return the events as (monotonic seconds, thread name, label) tuples."""
    global _events
    events, _events = _events or [], None
    return events


def active():
    return _events is not None


def mark(label):
    events = _events
    if events is not None:
        events.append((time.monotonic(), threading.current_thread().name, label))


# (stage name, from label, to label), in the order a frame meets them
STAGES = (
    ("scene evaluation", "worker_start", "render_begin"),
    ("scene prepare (shadow data, vertex packing)", "render_begin", "scene_ready"),
    ("scene upload (buffers, textures, pipelines, bind groups)", "scene_ready", "upload_done"),
    ("render encode", "upload_done", "submit"),
    ("GPU render and wait", "submit", "gpu_done"),
    ("readback copy", "gpu_done", "readback_done"),
    ("resolve (supersample, background)", "readback_done", "resolve_done"),
    ("after the render (node bookkeeping)", "resolve_done", "put_begin"),
    ("display cache put", "put_begin", "cache_put"),
    ("convert and emit", "cache_put", "emit"),
    ("hand-over to the display", "emit", "handover"),
)


def stage_table(events):
    """Split the marks into frames at each worker_start; per frame, the time between consecutive stage marks."""
    frames, current = [], None
    for stamp, thread, label in events:
        if label == "worker_start":
            current = {}
            frames.append(current)
        if current is None:
            continue
        if label in ("submit", "gpu_done", "readback_done"):
            current.setdefault(label + "_first", stamp)
            current[label] = stamp
        else:
            current.setdefault(label, stamp)
    rows = collections.defaultdict(list)
    for f in frames:
        if "handover" not in f or "render_begin" not in f:
            continue
        marks = [("worker_start", f["worker_start"]), ("render_begin", f["render_begin"]),
                 ("scene_ready", f.get("scene_ready")), ("upload_done", f.get("upload_done")),
                 ("submit", f.get("submit_first")), ("gpu_done", f.get("gpu_done")),
                 ("readback_done", f.get("readback_done")), ("resolve_done", f.get("resolve_done")),
                 ("put_begin", f.get("put_begin")), ("cache_put", f.get("cache_put")), ("emit", f.get("emit")), ("handover", f.get("handover"))]
        marks = [(n, t) for n, t in marks if t is not None]
        for (a, ta), (b, tb) in zip(marks, marks[1:]):
            rows[(a, b)].append((tb - ta) * 1000)
        rows[("total", "")].append((marks[-1][1] - marks[0][1]) * 1000)
    names = {(a, b): name for name, a, b in STAGES}
    table = []
    for (a, b), values in rows.items():
        name = names.get((a, b), f"{a} to {b}") if a != "total" else "worker start to hand-over (whole frame)"
        table.append({"stage": name, "from": a, "to": b, "n": len(values), "p50_ms": round(statistics.median(values), 2),
                      "p95_ms": round(sorted(values)[min(len(values) - 1, int(0.95 * len(values)))], 2),
                      "mean_ms": round(statistics.fmean(values), 2)})
    tick_gaps = [(b[0] - a[0]) * 1000 for a, b in zip([e for e in events if e[2] == "tick"], [e for e in events if e[2] == "tick"][1:])]
    stage_table.series = {f"{a} to {b}": [round(v, 1) for v in values] for (a, b), values in rows.items()}
    return table, len(frames), tick_gaps


