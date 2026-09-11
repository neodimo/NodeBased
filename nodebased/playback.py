"""Small, deterministic request queue for interactive playback.

This is deliberately not an execution framework.  It owns priority, bounds,
and stale-result identity while Window owns the worker and Evaluator owns its
cache.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import threading

from . import cachetier


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


class DisplayCache:
    """A bounded cache of finished, post-view-transform display images.

    This is the AE/Nuke "RAM preview" layer, and it exists because of where the cost in this
    pipeline actually lives: composing a 4K frame from its source graph is ~0.8s, cheap next to
    the ~2.3s the ACES 2.0 view transform costs on the same frame (measured directly; the sRGB
    view is ~0.2s on identical pixels, so this is the transform, not the resolution). The
    retained-result cache one layer down already makes a *repeated* compose of the identical
    graph state ~30ms — the transform was the one thing nothing amortized. Once a frame has paid
    for the transform once, replaying it costs only the cheap raw recompute: looping playback,
    scrubbing back onto a frame already shown, or returning a paused parameter to a value it held
    before.

    Keyed on the whole document rather than just frame number: a graph edit while paused changes
    the key and is correctly a miss, and undoing that edit back to a value already seen is
    correctly a hit. There is no separate invalidation path to keep in sync with the graph's
    actual edit operations — the key simply stops matching.
    """

    def __init__(self, budget_bytes=None):
        self.budget = cachetier.default_display_memory_bytes() if budget_bytes is None else int(budget_bytes)
        self.bytes = 0
        self._entries: OrderedDict[tuple, tuple[bytes, int, int, int]] = OrderedDict()

    @staticmethod
    def key(document, target, frame, tier, view, exposure, channel, background):
        # The document never carries pixel data (a Read node is a path string), so this stays
        # cheap regardless of source resolution -- it scales with graph size, not image size.
        digest = hashlib.blake2b(json.dumps(document, sort_keys=True).encode(), digest_size=16).digest()
        return (digest, target, int(frame), int(tier), view, float(exposure), channel, background)

    def get(self, key):
        """Return (bytes, width, height, bytes_per_line) or None."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, key, data: bytes, width: int, height: int, bytes_per_line: int):
        size = len(data)
        if size > self.budget:
            return
        existing = self._entries.pop(key, None)
        if existing is not None:
            self.bytes -= len(existing[0])
        self._entries[key] = (data, width, height, bytes_per_line)
        self.bytes += size
        while self.bytes > self.budget:
            _, evicted = self._entries.popitem(last=False)
            self.bytes -= len(evicted[0])

    def clear(self):
        self._entries.clear()
        self.bytes = 0

    def __len__(self):
        return len(self._entries)
