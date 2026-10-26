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
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

from . import cachetier, groups, sourcepool

DEFAULT_FRAMES = 2
DEFAULT_BYTE_CAP = 256 << 20

# Kinds after which the frame a node sees is not the frame the walk asked for (or is fetched by a nested call).
_TIME_KINDS = ("TimeOffset", "FrameHold", "Retime", "TimeClip", "FrameRange", "AppendClip",
               "TimeBlur", "TimeEcho", "TimeWarp", "MotionBlur", "MotionBlur2D", "MotionBlur3D", "Kronos", "OFlow")


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
                        return self.evaluator.read_ahead_node(flat, key, frame, tier, cancel)
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
