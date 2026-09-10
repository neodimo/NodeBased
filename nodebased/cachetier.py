"""Cache sizing and the on-disk tier that sits below the evaluator's in-memory LRU.

Two problems are solved here, both measured rather than assumed.

**Sizing.** The evaluator shipped with a fixed 256 MiB budget. One 4K RGBA float32 frame is
126.6 MiB, so a six-deep chain evicted its own predecessor at every node and the cache returned
zero hits — warm time-to-first-pixel matched cold to within 0.2%. One 8K frame is 506.2 MiB, which
exceeded the whole budget, so nothing was stored at all. A fixed byte count cannot be right for
both a laptop and a workstation, so the default is now derived from the machine's physical memory
and clamped into a range that is neither useless at 4K nor hostile on a small machine.

**Spill.** Clause C4 of `docs/EVALUATION_TIERS.md`: results evicted from memory go to a bounded,
digest-keyed on-disk store with its own LRU, and a read that hits disk repopulates memory. Entries
record dtype and shape and are rejected rather than reinterpreted on mismatch; a corrupt or
truncated entry is a miss, never an error raised at the artist.

The store is per-user and per-schema-version, and `.npy` is used precisely because its header
carries dtype and shape, so validation reads the file's own claim for the things the format can
state itself. A result's data and display windows are the one thing `.npy` cannot state, so they
live in a small JSON sibling; a missing or unreadable sidecar degrades to "the array is its own
window", which is exactly how every entry written before docs/BOUNDING_BOX.md should be read.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np

MIB = 1024 * 1024

# Fraction of physical memory offered to the retained-result cache. A compositor's cache is the
# difference between an interactive session and a slideshow, so this is not shy; it stays well
# clear of the working set an artist needs for the source images themselves and the OS page cache.
MEMORY_FRACTION = 0.25
# 512 MiB holds a four-deep 4K chain. Below that the cache stops being able to keep a realistic
# graph resident, which is the failure this module exists to remove.
MEMORY_FLOOR = 512 * MIB
MEMORY_CEILING = 8192 * MIB
MEMORY_FALLBACK = 1024 * MIB

DISK_BUDGET = 8192 * MIB

# Only the evaluator's own artifact layout is accepted back off disk.
ARTIFACT_DTYPE = np.float32
ARTIFACT_CHANNELS = 4


def physical_memory_bytes() -> int | None:
    """Total installed RAM, or None when the platform will not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    return None


def _env_bytes(name: str) -> int | None:
    """Read a megabyte-denominated override. A malformed value is ignored, not fatal."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        megabytes = float(raw)
    except ValueError:
        return None
    return int(megabytes * MIB) if megabytes > 0 else None


def default_memory_bytes() -> int:
    """The in-memory cache budget for this machine.

    `NODEBASED_CACHE_MB` overrides it outright — a render farm slot with a hard memory limit needs
    to state its own number rather than inherit a share of the host's RAM.
    """
    override = _env_bytes("NODEBASED_CACHE_MB")
    if override is not None:
        return override
    installed = physical_memory_bytes()
    if installed is None:
        return MEMORY_FALLBACK
    return max(MEMORY_FLOOR, min(MEMORY_CEILING, int(installed * MEMORY_FRACTION)))


def frame_bytes(width: int, height: int) -> int:
    """Bytes held by one RGBA float32 frame at this resolution."""
    return int(width) * int(height) * ARTIFACT_CHANNELS * np.dtype(ARTIFACT_DTYPE).itemsize


def resident_results(budget: int, width: int, height: int) -> int:
    """How many full-resolution results a budget keeps resident.

    The number that matters when reading a benchmark: a chain deeper than this evicts its own
    upstream at every node and cannot hit warm.
    """
    size = frame_bytes(width, height)
    return 0 if size <= 0 else int(budget // size)


def default_disk_root() -> Path:
    """Per-user, per-schema-version cache directory following each platform's convention."""
    from .core import SCHEMA_VERSION

    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        # Sandboxed/package test environments can deliberately clear both user-profile
        # variables and HOME. A cache location must degrade to a writable process-local path,
        # never make cache configuration itself fatal.
        base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or os.getcwd())
    elif sys.platform == "darwin":  # pragma: no cover - exercised on macOS only
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "nodebased" / f"schema-{SCHEMA_VERSION}"


class DiskCache:
    """A bounded, digest-keyed store of evicted results with its own LRU.

    Thread-safe because the evaluator runs on a worker thread while the UI thread may clear the
    cache underneath it. Every filesystem failure degrades to a miss: a cache that raises is worse
    than no cache at all.
    """

    def __init__(self, root=None, budget: int = DISK_BUDGET, enabled: bool = True):
        self.enabled = bool(enabled)
        self.budget = int(budget)
        self.root = Path(root) if root is not None else None
        self._index: OrderedDict[str, int] = OrderedDict()
        self._scanned = False
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.evictions = 0
        self.rejections = 0

    @classmethod
    def shared(cls) -> "DiskCache":
        """The default store. `NODEBASED_DISK_CACHE=0` turns the tier off entirely."""
        if os.environ.get("NODEBASED_DISK_CACHE", "1").strip().lower() in {"0", "false", "off", "no"}:
            return cls(enabled=False)
        return cls(root=default_disk_root(), budget=_env_bytes("NODEBASED_DISK_CACHE_MB") or DISK_BUDGET)

    @property
    def bytes(self) -> int:
        with self._lock:
            return sum(self._index.values())

    def _path(self, digest: str) -> Path:
        # Sharded by the first two hex characters so a large store does not become one flat
        # directory with a hundred thousand entries in it.
        return self.root / digest[:2] / f"{digest}.npy"

    def _scan(self):
        """Rebuild the LRU index from the filesystem on first use.

        The store outlives the process, so the index cannot live only in memory. Ordering by mtime
        reconstructs an approximate recency order across restarts, which is what the LRU needs.
        """
        if self._scanned:
            return
        self._scanned = True
        entries = []
        try:
            for path in self.root.glob("*/*.npy"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                entries.append((stat.st_mtime_ns, path.stem, stat.st_size))
        except OSError:
            return
        for _, digest, size in sorted(entries):
            self._index[digest] = size
        self._evict_locked()

    def get(self, digest: str):
        if not self.enabled or self.root is None:
            return None
        with self._lock:
            self._scan()
            if digest not in self._index:
                self.misses += 1
                return None
            path = self._path(digest)
        try:
            pixels = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            # Truncated by a crash mid-write, or not a readable array any more. Treat as a miss and
            # remove it, per C4: the artist never learns that a cache file went bad.
            self._discard(digest)
            self.misses += 1
            return None
        if pixels.dtype != ARTIFACT_DTYPE or pixels.ndim != 3 or pixels.shape[2] != ARTIFACT_CHANNELS:
            # Rejected, never reinterpreted. A stored artifact whose layout no longer matches is
            # not evidence about anything; reshaping it would be an invented result.
            self._discard(digest)
            self.rejections += 1
            self.misses += 1
            return None
        pixels.flags.writeable = False
        with self._lock:
            if digest in self._index:
                self._index.move_to_end(digest)
            self.hits += 1
        return pixels

    def put(self, digest: str, pixels) -> bool:
        if not self.enabled or self.root is None:
            return False
        size = int(pixels.nbytes)
        if size > self.budget:
            return False
        path = self._path(digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a sibling temporary file and rename, so a crash or a concurrent reader can
            # never observe a half-written array under the real digest.
            handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(handle, "wb") as stream:
                    np.save(stream, pixels, allow_pickle=False)
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except OSError:
            return False
        with self._lock:
            self._scan()
            self._index[digest] = path.stat().st_size if path.exists() else size
            self._index.move_to_end(digest)
            self.writes += 1
            self._evict_locked()
        return True

    # -- window-aware variants ------------------------------------------------------------------
    #
    # A stored array cannot say where it lives, and since docs/BOUNDING_BOX.md a result's data
    # window is part of its identity. The two rectangles go in a tiny JSON sibling rather than
    # inside the .npy, so the array file stays a plain readable array and every existing
    # `get`/`put` caller is untouched. A missing or unreadable sidecar means "data window equals
    # the array, display window equals the array" — the pre-bounding-box interpretation, which is
    # correct for every entry written before this existed.

    def _window_path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.box"

    def get_raster(self, digest: str):
        """A `Raster` rebuilt from the stored array and its window sidecar, or None on a miss."""
        pixels = self.get(digest)
        if pixels is None:
            return None
        from .raster import Raster
        from .tiers import Region

        data = display = None
        try:
            windows = json.loads(self._window_path(digest).read_text())
            data = Region(*windows["data"])
            display = Region(*windows["display"])
        except (OSError, ValueError, KeyError, TypeError):
            data = display = None
        if data is None or (data.width, data.height) != (pixels.shape[1], pixels.shape[0]):
            return Raster(pixels)
        return Raster(pixels, data, display)

    def put_raster(self, digest: str, raster) -> bool:
        if not self.put(digest, raster.pixels):
            return False
        payload = json.dumps({"data": [raster.data.x, raster.data.y, raster.data.width, raster.data.height],
                              "display": [raster.display.x, raster.display.y,
                                          raster.display.width, raster.display.height]})
        try:
            path = self._window_path(digest)
            handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(handle, "w") as stream:
                stream.write(payload)
            os.replace(temporary, path)
        except OSError:
            # The array is already stored and reads back correctly as a windowless raster, so a
            # failed sidecar degrades this entry rather than failing the write.
            return True
        return True

    def _evict_locked(self):
        total = sum(self._index.values())
        while self._index and total > self.budget:
            digest, size = self._index.popitem(last=False)
            total -= size
            self._unlink(digest)
            self.evictions += 1

    def _discard(self, digest: str):
        with self._lock:
            self._index.pop(digest, None)
        self._unlink(digest)

    def _unlink(self, digest: str):
        for path in (self._path(digest), self._window_path(digest)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self):
        if self.root is None:
            return
        with self._lock:
            self._scan()
            digests = list(self._index)
            self._index.clear()
        for digest in digests:
            self._unlink(digest)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "budget": self.budget, "bytes": self.bytes,
                "entries": len(self._index), "hits": self.hits, "misses": self.misses,
                "writes": self.writes, "evictions": self.evictions, "rejections": self.rejections}
