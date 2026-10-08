"""Disk-backed cache for frame-by-frame simulation solves. See docs/SIMULATION.md."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from .cancellation import Cancelled

MIB = 1024 * 1024
DEFAULT_MEMORY_BUDGET = 256 * MIB
DEFAULT_DISK_BUDGET = 2048 * MIB


def run_key(upstream_digest, params) -> str:
    payload = json.dumps([upstream_digest, params], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def default_disk_root() -> Path:
    from .cachetier import default_disk_root as cache_root

    return cache_root().parent / "simcache"


def _env_bytes(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        megabytes = float(raw)
    except ValueError:
        return None
    return int(megabytes * MIB) if megabytes > 0 else None


class State:
    """One solved simulation frame: named arrays plus JSON-safe metadata."""

    def __init__(self, arrays: dict[str, "np.ndarray"], meta: dict | None = None, copy: bool = True):
        # copy=False is for a solver that hands over freshly allocated arrays it will not touch again.
        self.arrays = {name: (np.asarray(array).copy() if copy else np.asarray(array))
                       for name, array in arrays.items()}
        self.meta = dict(meta or {})

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.arrays.values())

    def __eq__(self, other):
        if not isinstance(other, State) or self.meta != other.meta or self.arrays.keys() != other.arrays.keys():
            return False
        return all(np.array_equal(self.arrays[name], other.arrays[name]) for name in self.arrays)


class SimCache:
    """Bounded memory and disk cache of solved simulation frames."""

    def __init__(self, root=None, memory_budget: int | None = None,
                 disk_budget: int | None = None, enabled: bool = True):
        self.enabled = bool(enabled)
        self.root = Path(root) if root is not None else None
        self.memory_budget = DEFAULT_MEMORY_BUDGET if memory_budget is None else int(memory_budget)
        self.disk_budget = DEFAULT_DISK_BUDGET if disk_budget is None else int(disk_budget)
        self._memory: OrderedDict[tuple[str, int], State] = OrderedDict()
        self._index: OrderedDict[tuple[str, int], int] = OrderedDict()
        self._scanned = False
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.writes = 0
        self.evictions = 0
        self.rejections = 0
        # Wall-clock milliseconds the most recent `solve_to_frame` call spent computing (not
        # loading) each frame, keyed the same as `_memory`/`_index`. Kept off `State.meta` so it
        # never affects a solved state's identity or equality (a cache inspector's "solve time"
        # column, not a property of the simulation): see the cache inspector, docs/SIMULATION.md.
        self.timing: dict[tuple[str, int], float] = {}

    @classmethod
    def shared(cls) -> "SimCache":
        disabled = os.environ.get("NODEBASED_SIM_CACHE", "1").strip().lower() in {"0", "false", "off", "no"}
        if disabled:
            return cls(enabled=False)
        return cls(root=default_disk_root(),
                   disk_budget=_env_bytes("NODEBASED_SIM_CACHE_MB") or DEFAULT_DISK_BUDGET)

    def _path(self, key: tuple[str, int]) -> Path:
        run, frame = key
        return self.root / run[:2] / run / f"{frame:010d}.npz"

    def _scan(self) -> None:
        if self._scanned or not self.enabled or self.root is None:
            return
        self._scanned = True
        entries = []
        try:
            for path in self.root.glob("*/*/*.npz"):
                try:
                    run = path.parent.name
                    frame = int(path.stem)
                    stat = path.stat()
                except (OSError, ValueError):
                    continue
                entries.append((stat.st_mtime_ns, (run, frame), stat.st_size))
        except OSError:
            return
        for _, key, size in sorted(entries):
            self._index[key] = size
        self._evict_disk_locked()

    def _discard_locked(self, key: tuple[str, int]) -> None:
        self._index.pop(key, None)
        try:
            self._path(key).unlink(missing_ok=True)
        except OSError:
            pass

    def _evict_disk_locked(self) -> None:
        total = sum(self._index.values())
        while self._index and total > self.disk_budget:
            key, size = self._index.popitem(last=False)
            total -= size
            try:
                self._path(key).unlink(missing_ok=True)
            except OSError:
                pass
            self.evictions += 1

    def _write_disk_locked(self, key: tuple[str, int], state: State) -> bool:
        if not self.enabled or self.root is None:
            return False
        path = self._path(key)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(handle, "wb") as stream:
                values = dict(state.arrays)
                values["__meta__"] = np.array(json.dumps(state.meta))
                np.savez(stream, **values)
            os.replace(temporary, path)
            temporary = None
            self._scan()
            size = path.stat().st_size
            self._index[key] = size
            self._index.move_to_end(key)
            self.writes += 1
            self._evict_disk_locked()
            return True
        except (OSError, ValueError, TypeError):
            return False
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_disk(self, key: tuple[str, int], state: State) -> bool:
        with self._lock:
            return self._write_disk_locked(key, state)

    def _load_disk(self, key: tuple[str, int]) -> State | None:
        try:
            with np.load(self._path(key), allow_pickle=False) as archive:
                if "__meta__" not in archive.files:
                    raise KeyError("__meta__")
                meta = json.loads(str(archive["__meta__"]))
                if not isinstance(meta, dict):
                    raise ValueError("metadata is not a dict")
                # Each `archive[name]` is already a newly allocated, independently owned
                # ndarray. Keep it directly: copying here and again in State used to make
                # two full copies of every cache frame on a disk hit.
                arrays = {name: np.asarray(archive[name])
                          for name in archive.files if name != "__meta__"}
            return State(arrays, meta, copy=False)
        except (OSError, ValueError, EOFError, KeyError, TypeError):
            return None

    def get(self, run: str, frame: int) -> "State | None":
        key = (run, int(frame))
        with self._lock:
            state = self._memory.get(key)
            if state is not None:
                self._memory.move_to_end(key)
                self.hits += 1
                return state
            if not self.enabled or self.root is None:
                self.misses += 1
                return None
            self._scan()
            if key not in self._index:
                self.misses += 1
                return None
        state = self._load_disk(key)
        if state is None:
            with self._lock:
                self._discard_locked(key)
                self.misses += 1
            return None
        with self._lock:
            self._index.move_to_end(key)
            self.disk_hits += 1
            self.misses += 0
            self._memory[key] = state
            self._memory.move_to_end(key)
            self._trim_memory_locked()
        return state

    def _trim_memory_locked(self) -> None:
        while self._memory and sum(state.nbytes for state in self._memory.values()) > self.memory_budget:
            key, state = self._memory.popitem(last=False)
            if self.enabled and self.root is not None:
                self._write_disk_locked(key, state)

    def put(self, run: str, frame: int, state: "State") -> None:
        key = (run, int(frame))
        with self._lock:
            self._memory[key] = state
            self._memory.move_to_end(key)
            self._trim_memory_locked()
        if self.enabled and self.root is not None:
            self._write_disk(key, state)

    def latest_at_or_before(self, run: str, frame: int) -> "tuple[int, State] | None":
        candidates = set()
        with self._lock:
            candidates.update(number for (name, number) in self._memory if name == run and number <= frame)
            if self.enabled and self.root is not None:
                self._scan()
                candidates.update(number for (name, number) in self._index if name == run and number <= frame)
        for number in sorted(candidates, reverse=True):
            state = self.get(run, number)
            if state is not None:
                return number, state
        return None

    def frames(self, run: str) -> list[int]:
        """The frame numbers cached for `run`, in memory or on disk, ascending. For the cache
        inspector: what a frame list marks present versus missing."""
        with self._lock:
            present = {frame for (r, frame) in self._memory if r == run}
            if self.enabled and self.root is not None:
                self._scan()
                present.update(frame for (r, frame) in self._index if r == run)
        return sorted(present)

    def disk_size(self, run: str, frame: int) -> "int | None":
        """Bytes of the on-disk `.npz` for one cached frame, or `None` when it has none (still
        memory-only, or never solved)."""
        key = (run, int(frame))
        with self._lock:
            if self.enabled and self.root is not None:
                self._scan()
            return self._index.get(key)

    def solve_ms(self, run: str, frame: int) -> "float | None":
        """Milliseconds the last `solve_to_frame` in this process spent computing `frame` of
        `run`, or `None` when it was only ever loaded from disk or never solved this session."""
        return self.timing.get((run, int(frame)))

    def invalidate_from(self, run: str, frame: int) -> list[int]:
        """Discard every cached frame of `run` at or after `frame`, in memory and on disk. Returns
        the sorted frame numbers actually removed, so a caller can report exactly what changed."""
        frame = int(frame)
        removed = set()
        with self._lock:
            for key in [k for k in self._memory if k[0] == run and k[1] >= frame]:
                del self._memory[key]
                removed.add(key[1])
            if self.enabled and self.root is not None:
                self._scan()
                for key in [k for k in self._index if k[0] == run and k[1] >= frame]:
                    self._discard_locked(key)
                    removed.add(key[1])
        for key in [k for k in self.timing if k[0] == run and k[1] >= frame]:
            del self.timing[key]
        return sorted(removed)

    def stats(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled,
                    "memory_bytes": sum(state.nbytes for state in self._memory.values()),
                    "memory_entries": len(self._memory),
                    "disk_bytes": sum(self._index.values()),
                    "disk_entries": len(self._index),
                    "hits": self.hits, "misses": self.misses, "disk_hits": self.disk_hits,
                    "writes": self.writes, "evictions": self.evictions,
                    "rejections": self.rejections}


def solve_to_frame(cache: SimCache, run: str, target_frame: int, start_frame: int,
                   substeps: int, seed: int, initial_state, step, cancel=None) -> "State":
    if target_frame < start_frame:
        return initial_state(seed)
    cached = cache.get(run, target_frame)
    if cached is not None:
        return cached
    anchor = cache.latest_at_or_before(run, target_frame - 1)
    if anchor is None or anchor[0] < start_frame - 1:
        anchor_frame, state = start_frame - 1, initial_state(seed)
    else:
        anchor_frame, state = anchor
    for frame in range(anchor_frame + 1, target_frame + 1):
        started = time.perf_counter()
        for substep in range(substeps):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            state = step(state, frame, substep, seed)
        cache.timing[(run, frame)] = (time.perf_counter() - started) * 1000.0
        cache.put(run, frame, state)
    return state
