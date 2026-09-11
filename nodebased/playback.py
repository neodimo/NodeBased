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
    # Scene-space visible rectangle captured on the UI thread. None requests the complete
    # target data window (first preview / no prior image).
    viewport: tuple[int, int, int, int] | None = None
    # True when the transport issued this request. A playing request may still be displayed
    # after the playhead has moved past it, because during playback the useful guarantee is
    # "show the newest frame that finished" rather than "show exactly the playhead". See
    # Window.preview_ready and docs/PLAYBACK.md criterion 4.
    playing: bool = False


class PlaybackQueue:
    def __init__(self, max_prefetch=MAX_PREFETCH):
        self.max_prefetch = max_prefetch
        self._items = deque()
        self.active_cancel: threading.Event | None = None

    def replace(self, generation, frame, document, future_frames=(), tier=FULL_TIER, viewport=None,
                playing=False, cancel_active=True):
        """Install one display request plus bounded read-ahead, replacing what was queued.

        ``cancel_active=False`` leaves an already-running evaluation alone. The transport uses
        it: a playback tick invalidates the *queue*, not the frame currently being rendered.
        Cancelling on every tick is what made playback freeze outright whenever a frame cost
        more than one frame interval — each render was killed by the next tick, so none ever
        finished and the viewer kept showing the frame playback started on. Content-invalidating
        events (edits, view/channel/tier changes, scrubs, stop) still cancel, which is the set
        docs/PLAYBACK.md criterion 3 actually names.
        """
        from .tiers import PROXY_TIERS
        if int(tier) not in PROXY_TIERS:
            raise ValueError(f"Unsupported proxy tier {tier}; expected one of {PROXY_TIERS}")
        tier = int(tier)
        if cancel_active and self.active_cancel is not None:
            self.active_cancel.set()
        self._items.clear()
        self._items.append(
            FrameRequest(generation, int(frame), True, document, tier, viewport, playing))
        seen = {int(frame)}
        for future in future_frames:
            future = int(future)
            if future in seen:
                continue
            # Read-ahead intentionally requests the full target window. It warms future frames
            # without coupling background work to a viewport the artist may pan away from.
            self._items.append(
                FrameRequest(generation, future, False, document, tier, None, playing))
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
