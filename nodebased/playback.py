"""Small, deterministic request queue for interactive playback.

This is deliberately not an execution framework.  It owns priority, bounds,
and stale-result identity while Window owns the worker and Evaluator owns its
cache.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading


MAX_PREFETCH = 3
FULL_QUALITY = "full"


@dataclass(frozen=True)
class FrameRequest:
    generation: int
    frame: int
    display: bool
    document: dict
    quality: str = FULL_QUALITY


class PlaybackQueue:
    def __init__(self, max_prefetch=MAX_PREFETCH):
        self.max_prefetch = max_prefetch
        self._items = deque()
        self.active_cancel: threading.Event | None = None

    def replace(self, generation, frame, document, future_frames=(), quality=FULL_QUALITY):
        """Cancel obsolete work and install one display request plus bounded read-ahead."""
        if quality != FULL_QUALITY:
            raise ValueError("Proxy tiers are not implemented; playback quality must be 'full'")
        if self.active_cancel is not None:
            self.active_cancel.set()
        self._items.clear()
        self._items.append(FrameRequest(generation, int(frame), True, document, quality))
        seen = {int(frame)}
        for future in future_frames:
            future = int(future)
            if future in seen:
                continue
            self._items.append(FrameRequest(generation, future, False, document, quality))
            seen.add(future)
            if len(self._items) >= 1 + self.max_prefetch:
                break

    def cancel(self):
        if self.active_cancel is not None:
            self.active_cancel.set()
        self._items.clear()

    def take(self):
        if not self._items:
            return None
        request = self._items.popleft()
        self.active_cancel = threading.Event()
        return request, self.active_cancel

    def finish(self, cancel):
        if self.active_cancel is cancel:
            self.active_cancel = None

    def __len__(self):
        return len(self._items)
