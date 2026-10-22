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

import numpy as np

from . import cachetier
from .media import half_safe


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
    # A drag may show the last completed proxy request while newer snapshots are queued.
    live_preview: bool = False


class PlaybackQueue:
    def __init__(self, max_prefetch=MAX_PREFETCH):
        self.max_prefetch = max_prefetch
        self._items = deque()
        self.active_cancel: threading.Event | None = None

    def replace(self, generation, frame, document, future_frames=(), tier=FULL_TIER, viewport=None,
                playing=False, cancel_active=True, live_preview=False):
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
            FrameRequest(generation, int(frame), True, document, tier, viewport, playing, live_preview))
        seen = {int(frame)}
        for future in future_frames:
            future = int(future)
            if future in seen:
                continue
            # Read-ahead intentionally requests the full target window. It warms future frames
            # without coupling background work to a viewport the artist may pan away from.
            self._items.append(
                FrameRequest(generation, future, False, document, tier, None, playing, False))
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


PRESENTATION_FIELDS = ("pos", "name", "label", "thumbnail")

# Playback draws a 3D comp at the viewport's own size and one sample per pixel. A Render3D at 2 samples
# renders, reads back and averages four times the pixels (3840 x 2160 for a 1080p comp: a 133 MB staging
# buffer and about 300 ms of the 485 ms a cube frame cost on the RTX 3080 Ti, docs/PERFORMANCE.md "Lane 4
# playback profile"). The cap lives in the document the playing requests carry, so the display cache keeps
# playback frames apart from the full-quality ones: stopping re-renders the frame at the node's own samples.
PLAYBACK_SAMPLES = 1
_SAMPLED_MODES = ("raster", "raytrace")


def playing_document(document):
    """The document a *playing* request evaluates: every raster or ray-traced Render3D capped at
    ``PLAYBACK_SAMPLES`` samples and, when a GPU adapter exists, off the CPU rasteriser (a ``cpu`` backend plays as
    ``auto``: the GPU where it can draw the scene, the CPU reference where it cannot). Returns ``document`` itself
    when nothing needs changing, otherwise a copy that shares everything except those nodes. Path-traced Render3D
    nodes are left alone: they take their sample count from ``pt_samples``, not from ``samples``."""
    nodes = document.get("nodes", {})
    changed = {}
    gpu = None
    for key, node in nodes.items():
        if node.get("type") != "Render3D":
            continue
        params = node.get("params", {})
        if params.get("render_mode", "raster") not in _SAMPLED_MODES:
            continue
        edits = {}
        if int(params.get("samples", 1)) > PLAYBACK_SAMPLES:
            edits["samples"] = PLAYBACK_SAMPLES
        if params.get("render_backend", "cpu") == "cpu":
            if gpu is None:
                from . import gpu3d
                gpu = bool(gpu3d.available())
            if gpu:
                edits["render_backend"] = "auto"
        if edits:
            changed[key] = {**node, "params": {**params, **edits}}
    if not changed:
        return document
    return {**document, "nodes": {**nodes, **changed}}


# Node kinds whose output is a pure function of their parameters and inputs: no file sequence, no
# simulation, no per-frame state. This is an allow-list on purpose. A kind that is missing from it
# is treated as time-dependent, which only costs speed; a kind wrongly listed here would freeze
# playback on one picture. Add a kind only after reading that it never looks at the frame number.
STATIC_KINDS = frozenset({
    "Checker", "Constant", "Grid", "Ramp", "Rectangle", "Radial",
    "Grade", "Add", "Multiply", "Gamma", "Exposure", "Invert", "Clamp", "ColorCorrect",
    "Saturation", "Premult", "Unpremult", "Posterize", "Blur", "Sharpen", "Merge", "Blend",
    "Transform", "Crop", "Reformat", "Mirror", "Dot", "NoOp", "Switch", "Viewer",
    "Scene3D", "Camera3D", "Cube3D", "Sphere3D", "Cylinder3D", "Card3D", "Light3D", "Axis3D",
    "TransformGeo3D", "MergeGeo3D", "Render3D",
})

# Stands in for the frame number in a display key whose content does not depend on the frame.
STATIC_FRAME = "static"


def is_time_dependent(document, target):
    """True unless the picture at ``target`` is the same on every frame.

    Walks ``target`` and everything wired into it. The picture can change with the frame when an
    upstream knob carries an animation curve with more than one key, when any expression drives a
    knob (an expression may read the frame), or when a node is not on the ``STATIC_KINDS`` list
    (a Read of an image sequence, a simulation cache, a retime, a particle system and so on).
    An unknown target or a missing node counts as time-dependent.
    """
    nodes = document.get("nodes", {})
    if not target or target not in nodes:
        return True
    curves = (document.get("animation") or {}).get("curves") or {}
    expressions = document.get("expressions") or {}
    seen, pending = set(), [target]
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        node = nodes.get(key)
        if node is None or node.get("type") not in STATIC_KINDS:
            return True
        if any(len((curve or {}).get("keys", ())) > 1 for curve in (curves.get(key) or {}).values()):
            return True
        if expressions.get(key):
            return True
        inputs = node.get("inputs") or {}
        sources = [source for source in inputs.values() if source]
        if node.get("type") == "Render3D" and _is_empty_scene(nodes, inputs.get("scene")):
            sources = [inputs["scene"]]       # a camera cannot change a picture with nothing in front of it
        pending.extend(sources)
    return False


def _is_empty_scene(nodes, key):
    """True for a Scene3D with nothing wired into any slot: Render3D draws only its flat background."""
    node = nodes.get(key) if key else None
    return (node is not None and node.get("type") == "Scene3D"
            and not any((node.get("inputs") or {}).values()))


class DisplayCache:
    """A bounded cache of scene-linear, premultiplied frames.

    Frames are keyed independently of display parameters: view, exposure, channel, and
    background are display-time concerns applied after retrieval. Colour frames are stored as
    half-float because the precision loss is imperceptible; data passes such as depth, position,
    motion vectors, and UV maps are kept at full float32 precision so they are never rounded.

    This is the AE/Nuke "RAM preview" layer, and it exists because of where the cost in this
    pipeline actually lives: composing a 4K frame from its source graph is ~0.8s, cheap next to
    the ~2.3s the ACES 2.0 view transform costs on the same frame (measured directly; the sRGB
    view is ~0.2s on identical pixels, so this is the transform, not the resolution). The
    retained-result cache one layer down already makes a *repeated* compose of the identical
    graph state ~30ms. Keeping the composed scene-linear frame here avoids rebuilding that graph
    state when looping playback, scrubbing back onto a frame already shown, or returning a paused
    parameter to a value it held before; display conversion remains a separate, display-time step.

    Keyed on the whole document rather than just frame number: a graph edit while paused changes
    the key and is correctly a miss, and undoing that edit back to a value already seen is
    correctly a hit. There is no separate invalidation path to keep in sync with the graph's
    actual edit operations — the key simply stops matching.
    """

    def __init__(self, budget_bytes=None, force_float32=False):
        self.budget = cachetier.default_display_memory_bytes() if budget_bytes is None else int(budget_bytes)
        self.force_float32 = bool(force_float32)
        self.bytes = 0
        # Keyed by (frame key, region): a full-resolution preview is only the visible crop, so one
        # frame can legitimately hold a whole-frame read-ahead image and a zoomed-in crop at once.
        self._entries: OrderedDict[tuple, tuple[np.ndarray, bool]] = OrderedDict()
        self._residents: dict[tuple, int] = {}
        # Entries stored as the producer's float32 array (see put(lazy=True)) that have not been
        # rounded to half-float yet. Only the worker thread touches this.
        self._pending: set[tuple] = set()

    @staticmethod
    def identity(document, target, tier):
        """Everything that decides a scene-linear frame *except* which frame it is.

        The document never carries pixel data (a Read node is a path string), so this stays
        cheap regardless of source resolution -- it scales with graph size, not image size.

        ``time.current`` is deliberately excluded from the digest. Nothing downstream reads it:
        every evaluation path takes ``frame`` as an explicit argument (see evaluate_raster), and
        animation curves are resolved against that argument too. Leaving it in the hash meant a
        frame prefetched while current==5 hashed differently from the same frame redisplayed at
        current==6, so every single read-ahead entry was a guaranteed miss -- read-ahead defeated
        outright. Hoisting the frame out of the digest and into the key tuple (below) is what
        makes "prefetch it now, redisplay it later" land on one key, and it makes the identity
        reusable across a whole frame range in a single hash -- which is what lets the timeline
        ask "which frames are cached for this exact graph state?" without rehashing per frame.
        """
        if document.get("time", {}).get("current") is not None:
            document = {**document, "time": {**document["time"], "current": 0}}
        # Presentation-only node fields never change a pixel. Leaving them in meant dragging or
        # labelling a node silently orphaned every cached frame and blanked the timeline band.
        document = {**document, "nodes": {
            key: {name: value for name, value in node.items()
                  if name not in PRESENTATION_FIELDS}
            for key, node in document.get("nodes", {}).items()}}
        digest = hashlib.blake2b(json.dumps(document, sort_keys=True).encode(), digest_size=16).digest()
        return (digest, target, int(tier))

    @classmethod
    def key(cls, document, target, frame, tier):
        """The cache key of one frame. A target that does not depend on time shares one key
        (``STATIC_FRAME``) for every frame, so its picture is composed and converted once."""
        if not is_time_dependent(document, target):
            return (cls.identity(document, target, tier), STATIC_FRAME)
        return (cls.identity(document, target, tier), int(frame))

    def resident_frames(self, identity, frames):
        """Which of ``frames`` currently hold a scene-linear frame under ``identity``.

        Exact rather than bookkept: it reads the cache itself, so an entry the LRU evicted stops
        being reported the moment it is gone, and a document edit changes the identity so the
        answer becomes "none" without any invalidation hook to keep in sync.
        """
        if (identity, STATIC_FRAME) in self._residents:
            return {int(f) for f in frames}
        return {int(f) for f in frames if (identity, int(f)) in self._residents}

    def get(self, key, region=None, copy=True):
        """Return a copied float32 frame or None.

        ``copy=False`` lets a caller that only reads the pixels (the display conversion, the
        readout) take the stored array itself when it is already float32, saving a full-frame
        copy. A half-float entry is converted either way. Never write into the result of a
        ``copy=False`` read.

        `region` is the canvas rectangle the image covers. A full-resolution preview is only the
        visible crop, so an image cached for one pan/zoom is different pixels from the same frame
        at another; handing it back would paint the old crop at the new position. Regions must
        match exactly. None means "the whole frame" and matches only None.
        """
        entry = self._entries.get((key, region))
        if entry is None:
            return None
        self._entries.move_to_end((key, region))
        stored, _is_data = entry
        if not copy and stored.dtype == np.float32:
            return stored
        return stored.astype(np.float32, copy=True)

    def put(self, key, frame, region=None, is_data=False, lazy=False):
        """Store a frame. ``lazy=True`` keeps the producer's float32 array as it is, with no
        rounding or copy on the calling (worker) thread, and leaves the half-float conversion to
        ``compact()`` or to budget pressure. The caller must not write into ``frame`` afterwards."""
        pending = False
        if is_data or self.force_float32:
            stored = np.ascontiguousarray(frame, dtype=np.float32)
        elif lazy:
            stored = np.ascontiguousarray(frame, dtype=np.float32)
            pending = True
        else:
            stored = np.ascontiguousarray(half_safe(frame)).astype(np.float16)
        size = stored.nbytes
        if size > self.budget:
            return
        slot = (key, region)
        existing = self._entries.pop(slot, None)
        if existing is not None:
            self.bytes -= existing[0].nbytes
            self._pending.discard(slot)
        else:
            self._residents[key] = self._residents.get(key, 0) + 1
        self._entries[slot] = (stored, bool(is_data))
        if pending:
            self._pending.add(slot)
        self.bytes += size
        while self.bytes > self.budget and self._pending:
            # Rounding a waiting entry to half-float frees half its size, which is cheaper than
            # losing a whole frame.
            self._compact_one(next(iter(self._pending)))
        while self.bytes > self.budget:
            (evicted_key, evicted_region), evicted = self._entries.popitem(last=False)
            self._pending.discard((evicted_key, evicted_region))
            self.bytes -= evicted[0].nbytes
            remaining = self._residents[evicted_key] - 1
            if remaining:
                self._residents[evicted_key] = remaining
            else:
                del self._residents[evicted_key]

    def _compact_one(self, slot):
        entry = self._entries.get(slot)
        self._pending.discard(slot)
        if entry is None or entry[0].dtype != np.float32:
            return
        stored = np.ascontiguousarray(half_safe(entry[0])).astype(np.float16)
        self.bytes -= entry[0].nbytes - stored.nbytes
        self._entries[slot] = (stored, entry[1])

    @property
    def has_pending(self):
        return bool(self._pending)

    def compact(self, limit=None):
        """Round entries stored by ``put(lazy=True)`` to half-float. Run it on the worker when
        nothing else is waiting; ``limit`` caps how many entries one call converts."""
        done = 0
        for slot in list(self._pending):
            if limit is not None and done >= limit:
                break
            self._compact_one(slot)
            done += 1
        return done

    def clear(self):
        self._pending.clear()
        self._entries.clear()
        self._residents.clear()
        self.bytes = 0

    def __len__(self):
        return len(self._entries)
