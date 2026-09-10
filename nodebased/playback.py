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
FULL_TIER = 1


@dataclass(frozen=True)
class FrameRequest:
    generation: int
    frame: int
    display: bool
    document: dict
    # The proxy tier the request was made at. It travels with the request rather than being read
    # off the window, so a result that arrives after the artist has changed tier can be recognised
    # as stale instead of being drawn at the wrong size.
    tier: int = FULL_TIER


class PlaybackQueue:
    def __init__(self, max_prefetch=MAX_PREFETCH):
        self.max_prefetch = max_prefetch
        self._items = deque()
        self.active_cancel: threading.Event | None = None

    def replace(self, generation, frame, document, future_frames=(), tier=FULL_TIER):
        """Cancel obsolete work and install one display request plus bounded read-ahead."""
        from .tiers import PROXY_TIERS
        if int(tier) not in PROXY_TIERS:
            raise ValueError(f"Unsupported proxy tier {tier}; expected one of {PROXY_TIERS}")
        tier = int(tier)
        if self.active_cancel is not None:
            self.active_cancel.set()
        self._items.clear()
        self._items.append(FrameRequest(generation, int(frame), True, document, tier))
        seen = {int(frame)}
        for future in future_frames:
            future = int(future)
            if future in seen:
                continue
            self._items.append(FrameRequest(generation, future, False, document, tier))
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
