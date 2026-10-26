"""Worker threads that compute the independent sources of one frame at the same time.

A comp's Read, Roto and generator nodes have no input to wait for, so the Evaluator can hand all but
the first of them to this pool and compute them while it walks the graph up to, and through, the
first one. Each source is joined where the walk reaches its node, so a node never starts before its
inputs exist and a source that raises raises at its own node, as on the serial path.

What makes this safe to leave on:

* A task is claimed exactly once. The joining thread claims a task no worker has started and runs it
  itself, so nothing waits on a queue, a pool with no free worker costs nothing, and a pool that has
  shut down (or never had a worker) leaves the work with the thread that asked for it. Nothing is
  ever scheduled onto a stopped executor.
* Workers never wait for another task, and no lock is held while a source computes, so the pool
  cannot deadlock against the decode-ahead pool, the GPU lock or the evaluator's cache lock.
* Memory in flight is bounded twice: by the worker count and by a byte budget. A worker that would
  exceed the budget leaves its task for the joining thread.
* The pool is off at three CPUs or fewer and runs `FOUR_CPU_WORKERS` workers at four
  (`default_workers`), where measurement showed the stand-in gaining from them.
* Read-ahead work (`submit_ahead`) is a second, lower queue: workers take it only when no source of the
  frame being computed is waiting, at most `ahead_limit` of them at once, and a task that does not fit
  the byte budget is dropped instead of waiting. Nobody joins an ahead task; its result is a cache entry.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

from . import cachetier

# Three covers a comp's second Read, its Roto and one generator beside the thread that walks the graph.
MAX_WORKERS = 3

# What one source is assumed to hold before it has been measured: a 1080p float RGBA frame is 33 MB, a
# 4K frame 133 MB. The evaluator replaces this with the size it saw the node produce last.
DEFAULT_SOURCE_BYTES = 64 << 20

DEFAULT_BUDGET_FRACTION = 0.1

# Workers at exactly four CPUs (the laptop stand-in). Measured 10/26 on the stand-in, one and two workers with the Roto
# read-ahead: the CPU nodes play at 5.9 fps with one and 5.7 with two (4.8 with none), the Radeon island at 9.0 and 9.4 (7.1).
FOUR_CPU_WORKERS = 1


def available_cpus() -> int:
    """The CPUs this process may run on (affinity and cgroup limits included on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_workers(cpus: int | None = None) -> int:
    """`NODEBASED_SOURCE_WORKERS` overrides the count outright (0 turns the pool off); otherwise
    none at three CPUs or fewer, `FOUR_CPU_WORKERS` at four, and up to `MAX_WORKERS` beyond that."""
    raw = os.environ.get("NODEBASED_SOURCE_WORKERS")
    if raw:
        try:
            return max(0, min(MAX_WORKERS, int(raw)))
        except ValueError:
            pass
    cpus = available_cpus() if cpus is None else int(cpus)
    if cpus <= 3:
        return 0
    if cpus == 4:
        return FOUR_CPU_WORKERS
    return min(MAX_WORKERS, cpus - 3)


def default_budget_bytes() -> int:
    """`NODEBASED_SOURCE_POOL_MB` overrides the in-flight ceiling outright."""
    override = cachetier._env_bytes("NODEBASED_SOURCE_POOL_MB")
    if override is not None:
        return override
    return max(256 << 20, int(cachetier.default_memory_bytes() * DEFAULT_BUDGET_FRACTION))


class SourceTask:
    """One source evaluation: claimed once, by a worker or by the thread that joins it."""

    __slots__ = ("function", "size", "_lock", "_claimed", "_done", "_value", "_error", "pool", "key", "ahead")

    def __init__(self, function, size, pool=None, key=None):
        self.function = function
        self.size = int(size)
        self.pool = pool
        self.key = key
        self.ahead = False
        self._lock = threading.Lock()
        self._claimed = False
        self._done = threading.Event()
        self._value = None
        self._error = None

    def claim(self) -> bool:
        with self._lock:
            if self._claimed:
                return False
            self._claimed = True
            return True

    @property
    def claimed(self) -> bool:
        return self._claimed

    def execute(self):
        """Run the function on the calling thread; only the thread that won `claim` may call this."""
        try:
            self._value = self.function()
        except BaseException as error:      # re-raised, with the same type, at the joining node
            self._error = error
        finally:
            self._done.set()

    def result(self):
        """The task's value, computed here when no worker has started it; its error is re-raised."""
        if self.claim():
            self.execute()
        else:
            self._done.wait()
        if self._error is not None:
            raise self._error
        return self._value

    def wait(self, cancel=None):
        """Wait for a task that is running elsewhere; returns False when `cancel` fired first."""
        while not self._done.wait(0.02):
            if cancel is not None and cancel.is_set():
                return False
        return True

    def drop(self) -> bool:
        """Take an unstarted task out of play (it never runs); a running or finished one is left alone."""
        if self.claim():
            self._error = None
            self._done.set()
            return True
        return False

    def finish(self):
        """End of the evaluation that owns this task: drop it when unstarted, wait when running, so
        no source work outlives the call that asked for it."""
        if self.claim():
            self._error = None
            self._done.set()
        else:
            self._done.wait()


class SourcePool:
    """A small bounded pool of daemon threads for `SourceTask`s (see the module docstring)."""

    def __init__(self, max_workers=None, budget_bytes=None):
        self.max_workers = default_workers() if max_workers is None else max(0, min(MAX_WORKERS, int(max_workers)))
        self.budget = default_budget_bytes() if budget_bytes is None else int(budget_bytes)
        self._cond = threading.Condition()
        self._queue = deque()
        self._ahead = deque()
        self._threads = []
        self._shutdown = False
        self._in_flight = 0
        self.peak_in_flight = 0
        self.submitted = 0
        self.worker_runs = 0
        # Read-ahead tasks running at once: one stream keeps the lead without taking the workers the
        # frame in hand needs.
        self.ahead_limit = max(1, self.max_workers // 2)
        self._ahead_running = 0
        self.ahead_submitted = 0
        self.ahead_runs = 0
        self.ahead_dropped = 0

    @property
    def active(self) -> bool:
        return self.max_workers > 0 and not self._shutdown

    def submit(self, function, size=DEFAULT_SOURCE_BYTES, key=None) -> SourceTask:
        """Queue a task for the workers. Always returns a task the caller can join: with no worker
        (or after `shutdown`) nothing queues and `result()` runs the function on the caller."""
        task = SourceTask(function, size, self, key)
        with self._cond:
            self.submitted += 1
            if self._shutdown or self.max_workers == 0:
                return task
            self._queue.append(task)
            self._start_workers_locked()
            self._cond.notify()
        return task

    def submit_ahead(self, function, size=DEFAULT_SOURCE_BYTES, key=None):
        """Queue a read-ahead task, or return None when the pool has no worker to run it. Nobody joins an
        ahead task and a worker may drop it: its work is only ever a cache entry made early."""
        with self._cond:
            if self._shutdown or self.max_workers == 0:
                return None
            task = SourceTask(function, size, self, key)
            task.ahead = True
            self.ahead_submitted += 1
            self._ahead.append(task)
            self._start_workers_locked()
            self._cond.notify()
        return task

    def _start_workers_locked(self):
        while len(self._threads) < self.max_workers:
            thread = threading.Thread(target=self._worker, name=f"nodebased-source-{len(self._threads)}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def _take_locked(self):
        """The next task for a worker: a source of the frame in hand first, then read-ahead (one stream at
        a time, within the byte budget); None when there is nothing to start."""
        while self._queue:
            task = self._queue.popleft()
            # Hold the task back while its bytes would not fit beside what is already in flight;
            # the thread that joins it computes it instead.
            while (not self._shutdown and not task.claimed and self._in_flight
                   and self._in_flight + task.size > self.budget):
                self._cond.wait(0.005)
            if self._shutdown:
                return None
            if task.claim():
                return task
        while self._ahead and self._ahead_running < self.ahead_limit:
            task = self._ahead.popleft()
            if task.claimed:
                continue
            if self._in_flight and self._in_flight + task.size > self.budget:
                task.drop()
                self.ahead_dropped += 1
                continue
            if task.claim():
                return task
        return None

    def _worker(self):
        while True:
            with self._cond:
                task = None
                while task is None:
                    if self._shutdown:
                        return
                    task = self._take_locked()
                    if task is None:
                        if self._shutdown:
                            return
                        self._cond.wait(0.05 if (self._queue or self._ahead) else None)
                self._in_flight += task.size
                self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
                if task.ahead:
                    self._ahead_running += 1
            try:
                task.execute()
            finally:
                with self._cond:
                    self._in_flight -= task.size
                    if task.ahead:
                        self._ahead_running -= 1
                        self.ahead_runs += 1
                    else:
                        self.worker_runs += 1
                    self._cond.notify_all()

    def shutdown(self, wait=False, timeout=None):
        """Stop the workers. Tasks still unclaimed stay with whoever joins them; none is dropped."""
        with self._cond:
            self._shutdown = True
            self._queue.clear()
            for task in self._ahead:
                task.drop()
            self._ahead.clear()
            self._cond.notify_all()
            threads = list(self._threads)
        if wait:
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            for thread in threads:
                thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))

    def stats(self) -> dict:
        with self._cond:
            return {"workers": self.max_workers, "threads": len(self._threads), "budget": self.budget,
                    "in_flight": self._in_flight, "peak_in_flight": self.peak_in_flight,
                    "submitted": self.submitted, "worker_runs": self.worker_runs,
                    "queued": len(self._queue), "ahead_submitted": self.ahead_submitted,
                    "ahead_runs": self.ahead_runs, "ahead_dropped": self.ahead_dropped,
                    "ahead_queued": len(self._ahead)}
