"""Bounded parallel decode-ahead cache for upcoming Read-node source frames.

Playback advances the timeline faster than a full graph evaluation can safely run more than
one at a time (docs/PLAYBACK.md keeps the retained-result `Evaluator` single-owner), but
decoding a Read node's SOURCE pixels -- OIIO decode plus color ingest -- shares no mutable
state with the graph evaluator and has no such constraint. It is also the single largest cost
in proxied 4K playback (~150-180ms of the ~460ms per frame measured in
docs/BENCHMARKS-v0.17-playback.md, and it is identical at every proxy tier: a Read's decode
does not shrink with the tier, only the decimation afterward does -- see
`tileexec._node_full_image_at_tier`). This module runs that decode on a small dedicated
thread pool, ahead of the playhead, so by the time the single preview worker actually needs a
source frame it is usually already resident and the worker only pays for decimation, graph
evaluation and the display transform.

Cache entries are addressed by the exact Read parameters (including the resolved frame), so an
edit that changes those parameters is naturally a miss under the new key -- there is no
separate invalidation path to keep in sync. The `epoch` counter exists only to avoid caching
work that a seek has already made pointless: a decode already dispatched to OIIO cannot be
interrupted mid-flight (the same limitation `PlaybackQueue.replace` documents for graph
evaluation), so a stale job still runs to completion, but its result is dropped instead of
occupying a cache slot a still-wanted frame could use.
"""
from __future__ import annotations

from collections import OrderedDict
import os
import queue
import threading
import time

from . import cachetier

# CPU-bound decode + ingest work: more workers than physical headroom just contends for the
# same cores the single preview worker and the OCIO thread pool (`color._cpu_pool`) also want.
# Four is enough to keep 3-frame read-ahead (`Window.future_frames`) moving without starving
# the render actually driving the screen.
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)

# A handful of full-resolution decoded frames -- enough for read-ahead depth without taking a
# large bite out of the machine's memory alongside the evaluator's own retained-result budget
# and the display cache (both sized from `cachetier.default_memory_bytes()`).
DEFAULT_BUDGET_FRACTION = 0.1


def default_budget_bytes() -> int:
    """`NODEBASED_DECODE_AHEAD_MB` overrides it outright, same rationale as the other caches."""
    override = cachetier._env_bytes("NODEBASED_DECODE_AHEAD_MB")
    if override is not None:
        return override
    return int(cachetier.default_memory_bytes() * DEFAULT_BUDGET_FRACTION)


def default_workers() -> int:
    """`NODEBASED_DECODE_AHEAD_WORKERS` overrides the worker count outright, mainly for
    benchmarking how much of the single preview worker's real-world overhead (beyond the
    isolated per-stage costs in docs/BENCHMARKS-v0.17-playback.md) is GIL contention with this
    pool -- see that document's "Unverified" section."""
    raw = os.environ.get("NODEBASED_DECODE_AHEAD_WORKERS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_WORKERS


class DecodeAheadPool:
    """Thread-safe bounded LRU of decoded Read-node source rasters, filled ahead of demand."""

    def __init__(self, max_workers=None, budget_bytes=None):
        self.max_workers = int(max_workers) if max_workers else default_workers()
        self.budget = default_budget_bytes() if budget_bytes is None else int(budget_bytes)
        self._lock = threading.Lock()
        self._tasks = queue.Queue()
        self._threads = []
        self._shutdown = False
        self._entries: "OrderedDict[tuple, object]" = OrderedDict()
        self._bytes = 0
        self._pending: dict = {}
        self._epoch = 0
        self.hits = 0
        self.misses = 0
        for index in range(self.max_workers):
            thread = threading.Thread(target=self._worker,
                                      name=f"nodebased-decode-ahead-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def bump_epoch(self):
        """Invalidate every not-yet-finished decode job. Call on a real seek/edit, never on a
        plain playback tick -- see docs/PLAYBACK.md criterion 3 for the same distinction
        `PlaybackQueue.replace` already draws between a tick and a content-invalidating event."""
        with self._lock:
            self._epoch += 1

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def request(self, key, decode_fn):
        """Ensure `key` is resident or being decoded. Fire-and-forget: this is read-ahead, not
        a request for the pixels themselves, so it never blocks the caller and returns nothing.
        `decode_fn` takes no arguments, returns an array, and runs only on this pool's own
        worker threads -- never on the caller's thread."""
        with self._lock:
            if self._shutdown:
                return
            if key in self._entries or key in self._pending:
                return
            epoch = self._epoch
            self._pending[key] = epoch
            self._tasks.put((key, decode_fn, epoch))

    def _worker(self):
        while True:
            task = self._tasks.get()
            try:
                if task is None:
                    return
                key, decode_fn, epoch = task
                try:
                    self._run(key, decode_fn, epoch)
                except Exception:
                    # Read-ahead is opportunistic; a missing/corrupt source must not kill the
                    # worker that should service later frames. The foreground render reports
                    # the same decode failure when it actually needs the frame.
                    pass
            finally:
                self._tasks.task_done()

    def _run(self, key, decode_fn, epoch):
        try:
            with self._lock:
                if epoch != self._epoch:
                    return  # Superseded before decode started; skip the work entirely.
            pixels = decode_fn()
            with self._lock:
                if epoch != self._epoch:
                    return  # Superseded while decoding; drop the result, don't cache it.
            self._put(key, pixels)
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def _put(self, key, pixels):
        size = int(pixels.nbytes)
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._bytes -= int(existing.nbytes)
            if size > self.budget:
                return
            self._entries[key] = pixels
            self._bytes += size
            while self._bytes > self.budget and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= int(evicted.nbytes)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def shutdown(self, wait=False, timeout=None):
        """Stop accepting work and return without waiting for native decode calls by default.

        OIIO/OCIO calls cannot be interrupted safely. Workers are daemon threads so a GUI close
        cannot be held hostage by one such call, while ``wait=True`` remains available for tests
        or callers that have already released their decode gates. Queued work is removed from
        ``_pending``; an active job observes the bumped epoch and drops its result on completion.
        """
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                self._epoch += 1
                while True:
                    try:
                        key, _decode_fn, _epoch = self._tasks.get_nowait()
                    except queue.Empty:
                        break
                    self._pending.pop(key, None)
                    self._tasks.task_done()
                for _ in self._threads:
                    self._tasks.put(None)
        if wait:
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            for thread in self._threads:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                thread.join(remaining)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "bytes": self._bytes, "budget": self.budget,
                   "hits": self.hits, "misses": self.misses, "pending": len(self._pending)}
