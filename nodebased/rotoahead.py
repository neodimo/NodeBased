"""Roto matte read-ahead for playback.

A Roto matte at 1080p takes 25 to 30 ms and decides the frame time of the realistic comp. While one
frame is evaluated, converted and shown, `RotoAhead` has the source pool's workers compute the matte of
the next frame or two in playback order, so the walk of that frame finds it in the cache.

What keeps it safe:

* A read-ahead is the same nested evaluation the source pool runs for a source (`Evaluator.read_ahead_node`),
  so its result is the ordinary cache entry under the ordinary key. A walk that reaches a matte being
  computed waits for it; one that reaches a matte nobody started computes it itself.
* The lead is bounded twice: by a number of frames (`NODEBASED_ROTO_AHEAD`, default 2, 0 turns it off) and by a
  byte cap on the mattes in flight (`NODEBASED_ROTO_AHEAD_MB`, default 256), counted with the size each
  Roto produced last. The finished mattes are results in the evaluator's budgeted cache, so they are
  evicted like any other.
* `cancel()` ends the epoch: queued tasks are dropped, a running one stops before it stores. The window calls it
  when playback stops, the direction changes, the playhead is moved by hand, or the document is edited
  (every request that cancels the preview in flight).
* Only a Roto with no input is read ahead (a wired one would decode its plate on the worker), and none in a
  graph that remaps time: there the next frame's matte is not the next frame's.
* A Roto whose shapes do not animate has one matte for every frame: it is read ahead once per document state.
* It only runs for a Roto the walk waits for (`Evaluator.roto_is_critical`): where the chain beside the Roto is the long
  part of the frame (the CPU kernels) the matte is ready anyway, and drawing it early costs the chain memory bandwidth.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from . import cachetier, groups, sourcepool
from .cancellation import Cancelled

# The 1080p GPU playback trace still found 27 ms p95 Roto kernels on the
# foreground walk with a two-frame queue.  Keep one more matte in flight so
# the single preview worker has a better chance of finding it complete.
DEFAULT_FRAMES = 3
DEFAULT_BYTE_CAP = 256 << 20


def _build_matte(params, data):
    """Process entry point: build the same immutable Roto matte as Evaluator.

    This stays module-level so it is picklable under both fork and spawn.  The
    ndarray result is copied back through the executor's buffer; the evaluator
    remains the only owner of its caches.
    """
    from . import roto
    raster = roto.rasterise(data or [], params["width"], params["height"],
                            bool(params.get("invert", 0)))
    raster.flags.writeable = False
    return raster


class MatteProcessPool:
    """One optional process used only by Roto read-ahead.

    A failed or unavailable executor deliberately falls back to the existing
    source-pool thread.  The process is lazy so headless/unit-test evaluators
    never pay a child startup unless playback actually queues a matte.
    """
    def __init__(self):
        self._executor = None
        self._lock = threading.Lock()
        self.available = os.environ.get("NODEBASED_ROTO_PROCESS", "1") != "0"
        self.runs = 0
        self.fallbacks = 0

    def build(self, params, data, cancel):
        if not self.available:
            return None
        # Test/instrumentation wrappers must remain visible to the existing
        # thread-path tests and profiling hooks; child processes cannot inherit
        # their in-memory replacement safely.
        from . import roto
        if getattr(roto.rasterise, "__module__", None) != "nodebased.roto":
            self.fallbacks += 1
            return None
        with self._lock:
            if self._executor is None:
                try:
                    self._executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
                except Exception:
                    self.available = False
                    self.fallbacks += 1
                    return None
            future = self._executor.submit(_build_matte, params, data)
        while True:
            if cancel is not None and cancel.is_set():
                future.cancel()
                raise Cancelled()
            try:
                raster = future.result(timeout=0.02)
                self.runs += 1
                return raster
            except TimeoutError:
                continue
            except Exception:
                self.fallbacks += 1
                return None

    def shutdown(self):
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

# Kinds after which the frame a node sees is not the frame the walk asked for (or is fetched by a nested call).
_TIME_KINDS = ("TimeOffset", "FrameHold", "Retime", "TimeClip", "FrameRange", "AppendClip",
               "TimeBlur", "TimeEcho", "NoTimeBlur", "TimeWarp", "MotionBlur", "MotionBlur2D", "MotionBlur3D", "Kronos", "OFlow")


def default_frames() -> int:
    """`NODEBASED_ROTO_AHEAD` is how many frames ahead to compute (0 turns the read-ahead off)."""
    raw = os.environ.get("NODEBASED_ROTO_AHEAD")
    if raw:
        try:
            return max(0, min(8, int(raw)))
        except ValueError:
            pass
    return DEFAULT_FRAMES


def default_byte_cap() -> int:
    override = cachetier._env_bytes("NODEBASED_ROTO_AHEAD_MB")
    return DEFAULT_BYTE_CAP if override is None else override


def payload_animates(payload) -> bool:
    """True when a Roto's stored shapes can change from frame to frame: any value with an animation curve of
    two or more keys, any motion samples, or a link to a tracker. Read-only; a shape that tests False has
    one matte for the whole range."""
    if isinstance(payload, dict):
        curve = payload.get("curve")
        if isinstance(curve, dict) and len(curve.get("keys") or ()) > 1:
            return True
        for name, value in payload.items():
            if "motion" in str(name) or "track" in str(name):
                if value not in (None, [], {}, "", 0, False):
                    return True
            if payload_animates(value):
                return True
        return False
    if isinstance(payload, (list, tuple)):
        return any(payload_animates(item) for item in payload)
    return False


def read_ahead_sources(doc, target):
    """The Roto nodes under `target` worth reading ahead: enabled, nothing wired into them. A graph that remaps
    time returns none. Works on the flattened graph; returns `(flat_doc, flat_target, keys)`."""
    try:
        flat, flat_target = groups.flatten_groups(doc, target)
    except Exception:
        return doc, target, []
    nodes = flat["nodes"]
    from .tileexec import _all_ancestors
    try:
        ancestors = list(_all_ancestors(flat, flat_target, include_optional=True))
    except Exception:
        return flat, flat_target, []
    if any(nodes[key]["type"] in _TIME_KINDS for key in ancestors):
        return flat, flat_target, []
    keys = [key for key in ancestors
            if nodes[key]["type"] == "Roto" and not nodes[key]["disabled"]
            and all(source is None for source in nodes[key]["inputs"].values())]
    return flat, flat_target, keys


class RotoAhead:
    """Schedules Roto read-ahead on a `SourcePool` for one evaluator (see the module docstring)."""

    def __init__(self, evaluator, pool, frames=None, byte_cap=None):
        self.evaluator = evaluator
        self.pool = pool
        self.frames = default_frames() if frames is None else max(0, int(frames))
        self.byte_cap = default_byte_cap() if byte_cap is None else int(byte_cap)
        self._lock = threading.Lock()
        self._epoch = threading.Event()
        self._tasks = {}
        self._static = set()
        self.matte_pool = MatteProcessPool()
        self.scheduled = 0
        self.cancelled = 0
        self.epochs = 0

    @property
    def active(self) -> bool:
        return self.frames > 0 and self.pool is not None and self.pool.active

    def cancel(self):
        """End the epoch: queued read-ahead is dropped now, running read-ahead stops before it stores."""
        with self._lock:
            event, self._epoch = self._epoch, threading.Event()
            tasks, self._tasks = list(self._tasks.values()), {}
            self._static.clear()
            self.epochs += 1
        event.set()
        for task in tasks:
            if task.drop():
                self.cancelled += 1

    def schedule(self, doc, target, frames, tier) -> int:
        """Queue the mattes of `frames` (playback order, nearest first) for the Roto sources under `target`.
        Returns how many tasks were queued. A frame already queued or running is not queued twice."""
        if not self.active or not frames or target is None:
            return 0
        flat, flat_target, keys = read_ahead_sources(doc, target)
        keys = [key for key in keys if self.evaluator.roto_is_critical(key)]
        if not keys:
            return 0
        tier = int(tier)
        sizes = self.evaluator._source_sizes
        queued = 0
        with self._lock:
            epoch = self._epoch
            self._tasks = {name: task for name, task in self._tasks.items() if not task._done.is_set()}
            for key in keys:
                size = sizes.get(key, sourcepool.DEFAULT_SOURCE_BYTES)
                lead = max(1, min(self.frames, self.byte_cap // max(1, size)))
                wanted = list(frames)[:lead]
                payload = flat.get("node_data", {}).get(key)
                curves = flat.get("animation", {}).get("curves", {}).get(key)
                if not payload_animates(payload) and not curves:
                    signature = hashlib.sha256(json.dumps(
                        [key, tier, flat["nodes"][key]["params"], payload], sort_keys=True,
                        default=str).encode()).hexdigest()
                    if signature in self._static:
                        continue
                    self._static.add(signature)
                    wanted = wanted[:1]
                for frame in wanted:
                    name = (key, int(frame), tier)
                    if name in self._tasks:
                        continue

                    def run(key=key, frame=int(frame), cancel=epoch):
                        return self.evaluator.read_ahead_node(flat, key, frame, tier, cancel, self.matte_pool)
                    task = self.pool.submit_ahead(run, size, key)
                    if task is None:
                        return queued
                    self._tasks[name] = task
                    queued += 1
        self.scheduled += queued
        return queued

    def stats(self) -> dict:
        with self._lock:
            return {"frames": self.frames, "scheduled": self.scheduled, "cancelled": self.cancelled,
                    "epochs": self.epochs, "pending": len(self._tasks)}

    def shutdown(self):
        """Stop the optional process once the desktop window is closing."""
        self.cancel()
        self.matte_pool.shutdown()
