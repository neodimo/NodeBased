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
