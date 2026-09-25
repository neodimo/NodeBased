"""CPU reference evaluator. Internal RGBA is float32, scene-linear, premultiplied."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np
from PySide6.QtGui import QImage, QImageReader

from . import cachetier
from . import shapes
from . import tiers
from . import tracker
from .raster import Raster, as_array, scale_window
from .tiers import Region


from .cancellation import Cancelled


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((np.maximum(rgb, 0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)


def read_image(path, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0,
               frame_offset=0, missing="error", frame=None):
    """Read one timeline frame as a display-window array. Overscan is dropped at this boundary."""
    return read_image_raster(path, colorspace, alpha_mode, layer, subimage,
                             frame_offset, missing, frame).to_display()


def read_image_raster(path, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0,
                      frame_offset=0, missing="error", frame=None):
    """Read one timeline frame, data window intact. Read owns its source-time mapping — see
    docs/TIME_MODEL.md; it owns its bounding box the same way — see docs/BOUNDING_BOX.md."""
    from .media import nearest_sequence_path, read_media_raster, resolve_source_path
    source_frame = int(frame if frame is not None else 0) + int(frame_offset)
    resolved, exists = resolve_source_path(path, source_frame, missing)
    if resolved is None and not exists:
        # Preserve the sequence's actual display window. A 1x1 placeholder would make every
        # downstream Merge fail exactly when an artist asked for a harmless black gap. The gap's
        # data window is the display window: a black hole in a sequence claims no overscan.
        reference = nearest_sequence_path(path, source_frame)
        if reference is None:
            raise ValueError(f'No frames found for sequence {path}')
        display = read_media_raster(reference, colorspace, alpha_mode, layer, subimage).display
        return Raster(np.zeros((display.height, display.width, 4), np.float32), display, display)
    return read_media_raster(resolved, colorspace, alpha_mode, layer, subimage)


def read_image_region(path, region, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0,
                      frame_offset=0, missing="error", frame=None):
    """Acquire a bounded Read region at full resolution.

    The returned Raster is anchored at ``region`` even when it extends into EXR overscan or
    beyond a source data window. This is the source-side counterpart to TileExecutor's bounded
    compose API: a viewport request no longer needs a full image decode merely to slice it.
    """
    from .media import nearest_sequence_path, read_media_region, read_media_raster, resolve_source_path
    source_frame = int(frame if frame is not None else 0) + int(frame_offset)
    resolved, exists = resolve_source_path(path, source_frame, missing)
    if resolved is None and not exists:
        reference = nearest_sequence_path(path, source_frame)
        if reference is None:
            raise ValueError(f'No frames found for sequence {path}')
        display = read_media_raster(reference, colorspace, alpha_mode, layer, subimage).display
        return Raster(np.zeros((region.height, region.width, 4), np.float32), region, display)
    return read_media_region(resolved, region, colorspace, alpha_mode, layer, subimage)


def read_image_bounds(path, subimage=0, frame_offset=0, missing="error", frame=None):
    """Read only source window metadata for a timeline frame (no pixel decode)."""
    from .media import nearest_sequence_path, read_media_bounds, resolve_source_path
    source_frame = int(frame if frame is not None else 0) + int(frame_offset)
    resolved, exists = resolve_source_path(path, source_frame, missing)
    if resolved is None and not exists:
        resolved = nearest_sequence_path(path, source_frame)
        if resolved is None:
            raise ValueError(f'No frames found for sequence {path}')
    return read_media_bounds(resolved, subimage)


def to_qimage(frame, exposure=0.0, channel="RGB", background="black", view="sRGB"):
    """Compose a display image over `background` ("black" or "checker").

    Frames are premultiplied, so black is a true no-op: transparent regions stay at
    zero and a partially transparent edge keeps the value the graph produced. The
    checkerboard reads alpha at a glance but tints every pixel it shows through,
    which is why it is no longer the default.
    """
    alpha = frame[..., 3:4]
    if channel == "A":
        rgb = np.repeat(alpha, 3, axis=2)
    else:
        rgb = frame[..., :3] * (2.0 ** exposure)
        if channel in ("R", "G", "B"):
            rgb = np.repeat(rgb[..., "RGB".index(channel):"RGB".index(channel) + 1], 3, axis=2)
        from .color import display_rgb
        # The view transform is nonlinear, so it has to see straight (unassociated) colour;
        # feeding it premultiplied RGB bends every partially transparent pixel. `write_png`
        # has always unpremultiplied first, so the viewer and the exporter used to disagree
        # on exactly those pixels. Same order here: divide out alpha, transform, re-associate.
        weight = np.clip(alpha, 0, 1)
        straight = np.divide(rgb, weight, out=np.zeros_like(rgb), where=weight > 1e-8)
        rgb = display_rgb(straight, view) * weight
        if background == "checker":
            # Composited after the transform, so the checker keeps the tone it was authored
            # with under any view instead of being pushed through the display curve.
            yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
            levels = display_rgb(np.array([[[0.055] * 3, [0.095] * 3]], np.float32), view)[0, :, 0]
            bg = np.where((xx // 16 + yy // 16) % 2 == 0, levels[0], levels[1]).astype(np.float32)
            rgb = rgb + bg[..., None] * (1 - weight)
    rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    return QImage(rgb8.data, rgb8.shape[1], rgb8.shape[0], rgb8.strides[0], QImage.Format.Format_RGB888).copy()


def write_png(path, frame):
    if Path(path).suffix.lower() != ".png":
        raise ValueError("Export path must end in .png")
    alpha = np.clip(frame[..., 3:4], 0, 1)
    straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
    # Through the OCIO sRGB view, not a bare transfer curve: the working space is ACEScg, so
    # encoding needs the AP1 -> Rec.709 gamut conversion as well as the OETF. A hardcoded
    # `linear_to_srgb` here would silently desaturate every export.
    from .color import display_rgb
    rgba = np.concatenate((np.clip(display_rgb(straight, "sRGB"), 0, 1), alpha), axis=2)
    rgba8 = (rgba * 255 + 0.5).astype(np.uint8)
    image = QImage(rgba8.data, rgba8.shape[1], rgba8.shape[0], rgba8.strides[0], QImage.Format.Format_RGBA8888).copy()
    # QSaveFile preserves an existing export on write failure.
    from PySide6.QtCore import QSaveFile, QIODevice
    target = QSaveFile(str(path))
    if not target.open(QIODevice.OpenModeFlag.WriteOnly):
        raise ValueError(target.errorString())
    if not image.save(target, "PNG") or not target.commit():
        raise ValueError(f"Cannot export PNG: {target.errorString()}")


# Producers with their own time mapping (TIME_MODEL.md's "clip" shape): each remaps the timeline
# frame it is evaluated at onto a different frame for its single required input, via a nested
# `Evaluator.evaluate_raster` call rather than by reusing this walk's `values[source]`.
_TIME_REMAP_KINDS = ("TimeOffset", "FrameHold", "Retime", "TimeClip", "FrameRange", "AppendClip")


def _range_frame(frame, first, last, before, after):
    """Map ``frame`` onto ``[first, last]`` under the before/after policy, returning
    ``(frame_in_range, black)``. hold clamps; loop repeats the range (period ``last - first + 1``);
    bounce ping-pongs without repeating the end frames (period ``2 * (last - first)``, so 1..4 runs
    1 2 3 4 3 2 1 2 ...); black returns the nearest in-range frame (so the size is known) with
    ``black`` set, and the caller blanks it."""
    lo, hi = (first, last) if first <= last else (last, first)
    if lo <= frame <= hi:
        return frame, False
    policy = before if frame < lo else after
    nearest = lo if frame < lo else hi
    if policy == "black":
        return nearest, True
    if policy == "loop":
        return lo + (frame - lo) % (hi - lo + 1), False
    if policy == "bounce":
        if hi == lo:
            return lo, False
        period = 2 * (hi - lo)
        r = (frame - lo) % period
        return lo + (r if r <= hi - lo else period - r), False
    return nearest, False   # hold


def _time_clip_frame(kind, params, frame):
    """``(source_frame, black)`` for TimeClip and FrameRange. TimeClip first maps the timeline
    frame by ``frame - time_offset`` (TimeOffset's sign) and, unless ``frame_range_type`` is
    "all", applies ``[first, last]`` with ``before``/``after``. FrameRange applies
    ``[first_frame, last_frame]`` to the frame directly."""
    frame = int(frame)
    if kind == "TimeClip":
        mapped = frame - int(params.get("time_offset", 0))
        if params.get("frame_range_type", "custom") == "all":
            return mapped, False
        first, last = int(params.get("first", 1)), int(params.get("last", 100))
    else:
        mapped = frame
        first, last = int(params.get("first_frame", 1)), int(params.get("last_frame", 100))
    return _range_frame(mapped, first, last, params.get("before", "hold"), params.get("after", "hold"))


def _branch_frame_range(nodes, key):
    """The frame range a clip's branch presents downstream, or None when unknown. Known only when
    a FrameRange or a custom-range TimeClip is reached from `key` through bypassed nodes and Dots
    (the document has no other notion of a branch's range); a TimeClip presents its range shifted
    by its offset, since that is the frame span over which it maps into [first, last]."""
    from .core import SPECS, bypass_slot
    for _ in range(len(nodes) + 1):
        node = nodes[key]
        kind = node["type"]
        if node["disabled"] or kind in ("Dot", "NoOp"):
            slot = bypass_slot(node)
            key = None if slot is None else node["inputs"].get(slot)
            if key is None:
                return None
            continue
        params = node["params"]
        if kind == "FrameRange":
            first, last = int(params["first_frame"]), int(params["last_frame"])
            return (min(first, last), max(first, last))
        if kind == "TimeClip" and params.get("frame_range_type", "custom") == "custom":
            shift = int(params.get("time_offset", 0))
            first, last = int(params["first"]) + shift, int(params["last"]) + shift
            return (min(first, last), max(first, last))
        return None
    return None


def _append_clip_plan(doc_nodes, node, params, frame):
    """The nested evaluations AppendClip needs at timeline ``frame``: a list of
    ``(slot, source_frame, weight)`` with weights summing to 1 (one entry, or two inside a
    dissolve).

    Wired clips play head to tail from ``first_frame``. A clip's length is its branch's frame range
    when directly known (`_branch_frame_range`), and it is sampled from that range's first frame;
    otherwise its length is the ``length<i>`` knob (0 skips the clip) and it is sampled from frame 1.
    ``dissolve`` frames overlap each clip's tail with the next clip's head (clamped to one frame
    less than either clip); inside the overlap the outgoing clip is weighted ``1 - w`` and the
    incoming ``w = (k + 1) / (dissolve + 1)`` at overlap index ``k``. Before the first clip and
    after the last the first frame / last frame holds. If more than two clips overlap on one frame
    (dissolve longer than a middle clip), the first two are blended and the rest ignored."""
    from .core import SPECS
    clips = []
    for index, slot in enumerate(SPECS["AppendClip"]["optional_inputs"]):
        source = node["inputs"].get(slot)
        if source is None:
            continue
        span = _branch_frame_range(doc_nodes, source)
        if span is not None:
            length, start = span[1] - span[0] + 1, span[0]
        else:
            length, start = int(params.get(f"length{index}", 100)), 1
        if length > 0:
            clips.append((slot, length, start))
    if not clips:
        raise ValueError(f"{node['name']}: connect at least one clip input")
    dissolve = int(params.get("dissolve", 0))
    starts, overlaps, cursor = [], [], int(params.get("first_frame", 1))
    for i, (_, length, _) in enumerate(clips):
        starts.append(cursor)
        overlap = 0
        if i + 1 < len(clips):
            overlap = max(0, min(dissolve, length - 1, clips[i + 1][1] - 1))
        overlaps.append(overlap)
        cursor += length - overlap
    frame = int(frame)
    live = [i for i, (_, length, _) in enumerate(clips) if starts[i] <= frame < starts[i] + length]
    if not live:
        i = 0 if frame < starts[0] else len(clips) - 1
        slot, length, first = clips[i]
        return [(slot, first if i == 0 else first + length - 1, 1.0)]
    if len(live) >= 2:
        a, b = live[0], live[1]
        k = frame - starts[b]
        weight = (k + 1) / (overlaps[a] + 1)
        (slot_a, _, first_a), (slot_b, _, first_b) = clips[a], clips[b]
        return [(slot_a, first_a + frame - starts[a], 1.0 - weight),
                (slot_b, first_b + frame - starts[b], weight)]
    i = live[0]
    slot, _, first = clips[i]
    return [(slot, first + frame - starts[i], 1.0)]


def _time_remap_frame(kind, params, frame):
    """The frame a time-remapping node's input is evaluated at, given the timeline `frame` the
    node itself is being evaluated at.

    TimeOffset: the input is evaluated at ``frame - time_offset``; ``reverse`` flips which
    direction the offset is applied (``frame + time_offset``), reversing the shift rather than the
    footage itself.

    FrameHold: holds on ``first_frame`` when ``increment`` is 0 (Nuke's own default). A positive
    increment advances the hold in steps: the input is evaluated at the nearest
    ``first_frame + k * increment`` at or before ``frame`` (``k`` clamped to 0 for a `frame` before
    ``first_frame`` — there is no earlier valid hold point to reach for).

    Retime (simplified per docs/PARITY_2D.md — nearest-frame sampling, no frame blending): a frame
    maps linearly from the output range onto the input range, anchored at each range's start and
    scaled by ``speed``: ``input_range_start + (frame - output_range_start) * speed``. The range
    end knobs are carried for Nuke-parity naming/duration bookkeeping; the simplified mapping does
    not consult them.
    """
    frame = int(frame)
    if kind == "TimeOffset":
        offset = int(params.get("time_offset", 0))
        return frame + offset if params.get("reverse") else frame - offset
    if kind == "FrameHold":
        first = int(params.get("first_frame", 1))
        increment = int(params.get("increment", 0))
        if increment <= 0 or frame <= first:
            return first
        return first + ((frame - first) // increment) * increment
    if kind == "Retime":
        input_start = float(params.get("input_range_start", 0))
        output_start = float(params.get("output_range_start", 0))
        speed = float(params.get("speed", 1.0))
        return int(round(input_start + (frame - output_start) * speed))
    if kind in ("TimeClip", "FrameRange"):
        return _time_clip_frame(kind, params, frame)[0]   # the "black" flag lives on that function
    raise ValueError(f"{kind} is not a single-input time-remapping kind")


class Evaluator:
    """Retained-result evaluator with a memory tier over an optional disk tier.

    The budget is sized for the machine rather than fixed (see `nodebased/cachetier.py`), because
    the previous fixed 256 MiB could not hold a single 8K result and evicted its own upstream at
    every node of a 4K chain — measured as zero cache hits and warm time-to-first-pixel matching
    cold to within 0.2%.
    """

    def __init__(self, cache_bytes=None, disk=None, sim=None):
        # The disk tier is opt-in at construction rather than on by default: a library evaluator
        # must not start writing to a user's cache directory as a side effect of being imported.
        # The desktop app and the agent CLI pass `DiskCache.shared()` explicitly.
        self.budget = cachetier.default_memory_bytes() if cache_bytes is None else int(cache_bytes)
        self.disk = cachetier.DiskCache(enabled=False) if disk is None else disk
        self.cache = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        # Simulation frames (docs/SIMULATION.md). A ParticleEmitter3D keeps its frames in memory only,
        # so scrubbing within a session never re-solves. ParticleCache3D adds a persistent tier
        # when a `simcache.SimCache` is passed as `sim` (the app passes `SimCache.shared()`, like
        # the disk tier above); its budgets are per node, one store per distinct pair of budgets.
        from . import simcache
        self.sim_template = sim
        self._sim_memory = simcache.SimCache(enabled=False)
        self._sim_stores = {}
        # The desktop app sets this callback to show ETA and stop CPU splat budget refusals.
        self.progress = None

    def sim_store(self, memory_mb, disk_mb):
        """The ParticleCache3D store for one pair of budgets (megabytes)."""
        from . import simcache
        key = (int(memory_mb), int(disk_mb))
        store = self._sim_stores.get(key)
        if store is None:
            template = self.sim_template
            enabled = bool(template is not None and template.enabled and template.root is not None and key[1] > 0)
            store = simcache.SimCache(root=template.root if enabled else None,
                                      memory_budget=key[0] << 20, disk_budget=key[1] << 20, enabled=enabled)
            self._sim_stores[key] = store
        return store

    def clear(self):
        """Drop retained results from memory.

        The disk tier survives on purpose: it is what makes reopening a project cheap, and clearing
        it is a separate, explicit act.
        """
        self.cache.clear()
        self.bytes = 0

    def resident_results(self, width, height):
        """How many results at this resolution the current budget keeps resident."""
        return cachetier.resident_results(self.budget, width, height)

    def _store(self, digest, raster):
        """Insert into memory, spilling evicted neighbours to the disk tier (contract C4).

        An oversized single result is still stored. The budget is a ceiling on the retained *set*;
        refusing to keep the only result the artist is looking at was the 8K failure — it made the
        cache silently inert at exactly the resolution where recompute hurts most.
        """
        while self.cache and self.bytes + raster.nbytes > self.budget:
            old_digest, old = self.cache.popitem(last=False)
            self.bytes -= old.nbytes
            self.disk.put_raster(old_digest, old)
        self.cache[digest] = raster
        self.bytes += raster.nbytes

    def evaluate(self, doc, target=None, cancel: threading.Event | None = None, frame=None, tier=1):
        """Evaluate `target` and return its **display window** as an array.

        Overscan is discarded here and only here — see `evaluate_raster` when the caller needs the
        node's real data window. Keeping this signature returning a frame-shaped array is what lets
        the bounding box become a first-class concept without every existing caller changing: a
        comp with no overscan produces a byte-identical result to the pre-bounding-box evaluator.
        """
        return self.evaluate_raster(doc, target, cancel, frame, tier).to_display()

    def evaluate_raster(self, doc, target=None, cancel: threading.Event | None = None,
                        frame=None, tier=1, typed=False, return_digest=False):
        """Evaluate `target` at one timeline frame, optionally at a proxy tier.

        `frame` is an argument rather than ambient state on purpose: a clip-based timeline maps one
        timeline frame onto a different source frame per clip, so nothing may reach for a global
        playhead. See docs/TIME_MODEL.md. Omitting it uses the document's stored current frame.

        `tier` is an argument for the same reason, and additionally because export must be able to
        ask for tier 1 while the viewer is showing tier 4 (contract clause C3). It is deliberately
        not stored in the document: a proxy setting is how one artist is looking at a comp right
        now, not a property of the comp.

        `return_digest`, when true, returns `(raster, digest)` instead of `raster`. This is how
        TimeOffset/FrameHold/Retime fold a *nested* evaluation's content digest into their own
        cache key: each is a clip-shaped producer whose input is evaluated at a remapped frame (a
        recursive `evaluate_raster` call, exactly the pattern TIME_MODEL.md's "clip timeline"
        section sketches), and its digest must reflect what that nested call actually resolved to
        at that frame — not the outer walk's `hashes[source]`, which was computed at the wrong
        (document) frame and would miss an edit to a keyframe elsewhere in the curve.
        """
        tier = int(tier)
        if tier not in tiers.PROXY_TIERS:
            raise ValueError(f"Unsupported proxy tier {tier}; expected one of {tiers.PROXY_TIERS}")
        target = target or doc["view"]
        if target is None:
            raise ValueError("Select a node and press 1 to view it")
        if frame is None:
            frame = doc.get("time", {}).get("current", 1)
        frame = int(frame)
        nodes = doc["nodes"]
        # Iterative postorder traversal: execute only ancestors of the viewer.
        order, seen = [], set()
        stack = [(target, False)]
        while stack:
            key, visited = stack.pop()
            if key in seen:
                continue
            if visited:
                seen.add(key)
                order.append(key)
                continue
            stack.append((key, True))
            inputs = list(nodes[key]["inputs"].values())
            if nodes[key]["disabled"]:
                from .core import bypass_slot
                slot = bypass_slot(nodes[key])
                inputs = [] if slot is None else [nodes[key]["inputs"][slot]]
            elif nodes[key]["type"] in _TIME_REMAP_KINDS:
                # This node's input is fetched by a nested `evaluate_raster` call at a remapped
                # frame (below), not from `values[source]` computed by this walk at `frame` --
                # walking into it here would only evaluate and cache it at the wrong frame,
                # uselessly (a cache miss neither this node nor anything else consumes) and, for
                # an animated source, would defeat exactly the reuse FrameHold exists to give a
                # scrub: `hashes[source]` would churn with the outer frame even while
                # `effective_frame` — and so the real result — stays put.
                inputs = []
            stack.extend((source, False) for source in inputs if source is not None)
        values, hashes = {}, {}
        from . import particles

        def reads_lazily(key):
            """True when every reader of particle node `key` is an enabled ParticleCache3D or a force
            node, so `key` need not solve: the cache (or the force, which carries the run on and solves
            the whole chain itself when it is not read lazily too) does the solving."""
            readers = [other for other in order if key in nodes[other]["inputs"].values()]
            return bool(readers) and all(
                (nodes[other]["type"] == "ParticleCache3D" and not nodes[other]["disabled"])
                or nodes[other]["type"] in particles.FORCE_KINDS for other in readers)
        for key in order:
            if cancel and cancel.is_set():
                raise Cancelled()
            node = nodes[key]
            kind = node["type"]
            active_inputs = node["inputs"]
            if node["disabled"]:
                from .core import bypass_slot
                slot = bypass_slot(node)
                active_inputs = {} if slot is None else {slot: node["inputs"][slot]}
            sources = list(active_inputs.values())
            # Resolve animation curves onto a per-frame params copy. node["params"] is the stored
            # base value; animation is an overlay that never mutates it. Static nodes (no curves)
            # resolve to a shallow copy that compares equal under json.dumps, so existing caches
            # keep their keys. See docs/ANIMATION.md.
            from .core import (SPECS as _SPECS, LIMITS as _LIMITS, OUTPUT_TYPES, GEOMETRY_TYPES,
                               _XFORM as _IDENTITY_XFORM, DRAW_KINDS as _DRAW_KINDS)
            from .animation import resolve_params as _resolve_params
            node_curves = doc.get("animation", {}).get("curves", {}).get(key)
            params = _resolve_params(node, node_curves, frame, _SPECS[kind]["params"], _LIMITS)
            # Pixel-unit parameters are scaled in the same pass that shrinks the sources, so a blur
            # radius or a crop rectangle means the same thing at every tier (clause C3). Generated
            # sources therefore produce small pixels rather than being rendered full size and
            # shrunk, which is the difference between a tier that saves work and one that does not.
            # Scaling runs *after* curve resolution: a pixel-unit parameter must be scaled from the
            # value this frame actually uses, or an animated blur radius would proxy at its base.
            params = tiers.scale_params(kind, params, tier)
            # Structured payloads (schema v8) follow exactly the same two steps as parameters, in
            # the same order and for the same reasons: scale the pixel units to this tier, then
            # resolve every animatable scalar at this frame. `scale_node_data` scales curve key
            # values too, so scaling first leaves a curve a curve and the order is not observable.
            payload = tiers.scale_node_data(kind, doc.get("node_data", {}).get(key), tier)
            data = None
            if kind == "Roto":
                data = shapes.resolve_shapes(payload, frame)
            elif kind == "Tracker":
                # A Tracker's solved geometry is merged into params rather than carried beside
                # them, so every later stage — window, kernel, region rule, digest — sees an
                # ordinary Transform and cannot treat the two differently by accident.
                params = {**params, **tracker.solve(payload, frame, params)}
            # Only required slots (those listed in SPECS[kind]["inputs"]) must be wired; optional
            # slots — like the new "mask" input on image-filter nodes — are allowed to be None and
            # the kernel treats that as identity (mask.a = 1, no extra gating).
            required = set(_SPECS.get(kind, {}).get("inputs", []))
            for slot, source in active_inputs.items():
                if source is None and slot in required:
                    raise ValueError(f"{node['name']}: connect required input(s)")
            # 3D scene values are typed runtime objects rather than image rasters. They stay on
            # the same graph/evaluation boundary, but are deliberately reference-rendered here:
            # the only value that crosses back into the existing 2D graph is Render3D's Raster.
            if OUTPUT_TYPES.get(kind, "image") != "image" or kind == "Render3D":
                from . import scene3d
                fingerprint = None
                if kind == "ReadSplat3D" and not node["disabled"]:
                    from . import splats
                    if not params["splat_path"]:
                        raise ValueError("ReadSplat3D: choose a splat file")
                    fingerprint = splats.fingerprint(params["splat_path"])
                if kind == "ReadGeo3D" and not node["disabled"]:
                    fingerprint = scene3d.obj_fingerprint(params["geo_path"])
                if kind in ("ReadUSD3D", "ReadUSDCamera3D") and not node["disabled"]:
                    from . import usdio
                    try:
                        fingerprint = [usdio.fingerprint(params["usd_path"]), frame]
                    except RuntimeError as error:
                        raise ValueError(str(error)) from None
                if kind == "ReadGLTF3D" and not node["disabled"]:
                    from . import gltfio
                    fingerprint = gltfio.fingerprint(params["gltf_path"])
                if kind in ("ReadAlembic3D", "ReadAlembicCamera3D") and not node["disabled"]:
                    from . import alembicio
                    seconds = frame / doc["time"]["fps"]
                    fingerprint = [alembicio.fingerprint(params["abc_path"]), frame, seconds]
                stream = None
                if kind == "ParticleEmitter3D" and not node["disabled"]:
                    stream = particles.build_stream(self, doc, key, node, cancel)
                    fingerprint = [stream.run, frame]
                elif kind == "ParticleCache3D" and not node["disabled"]:
                    stream = getattr(values[node["inputs"]["particles"]], "stream", None)
                    fingerprint = [None if stream is None else stream.run, frame]
                elif kind in particles.FORCE_KINDS:
                    stream = getattr(values[node["inputs"]["particles"]], "stream", None)
                    if stream is not None and not node["disabled"]:
                        stream = particles.extend_stream(stream, doc, key, node, self, cancel)
                    fingerprint = [None if stream is None else stream.run, frame]
                digest = hashlib.sha256(json.dumps([kind, params, node["disabled"],
                                                     [hashes[s] if s is not None else None for s in sources],
                                                     fingerprint, tier, data], sort_keys=True).encode()).hexdigest()
                hashes[key] = digest
                if kind in GEOMETRY_TYPES:
                    # A disabled geometry node contributes nothing rather than passing its texture on.
                    texture = values[sources[0]] if sources and sources[0] is not None else None
                    value = None if node["disabled"] else scene3d.geometry_from_node(
                        {"type": kind, "params": params},
                        None if texture is None else texture.to_display())
                elif kind == "TransformGeo3D":
                    # Disabled bakes nothing: the geometry passes through exactly as bypass_slot
                    # says (its one required input), matching a disabled 2D Transform.
                    source = values[node["inputs"]["geo"]]
                    value = source if node["disabled"] else scene3d.transform_geometry(
                        source, scene3d._transform_from(params))
                elif kind == "MergeGeo3D":
                    # Disabled passes the first wired geometry through untouched (bypass_slot), and
                    # an empty merge is an empty geometry, never a crash.
                    if node["disabled"]:
                        value = next((v for v in (values[s] for s in sources if s is not None)
                                      if v is not None), scene3d.empty_geometry())
                    else:
                        wired = [values[s] for s in sources if s is not None]
                        value = scene3d.merge_geometry(wired, scene3d._transform_from(params))
                elif kind == "ParticleEmitter3D":
                    # Disabled passes the emission geometry (bypass_slot "geo"), or nothing when unwired.
                    if node["disabled"]:
                        source = node["inputs"].get("geo")
                        value = values[source] if source is not None else None
                    elif key != target and reads_lazily(key):
                        # Only ParticleCache3D and force nodes read this emitter: the cache solves the run
                        # through its own persistent store, so the emitter hands over the run without
                        # solving it twice.
                        value = particles.placeholder_instance(stream, frame)
                    else:
                        state = particles.solve_frame(stream, frame, self._sim_memory, cancel)
                        value = particles.instance_from_state(state, stream, frame)
                elif kind in particles.FORCE_KINDS:
                    incoming = values[node["inputs"]["particles"]]
                    if stream is None:
                        value = incoming
                    elif key != target and reads_lazily(key):
                        value = replace(incoming, stream=stream, frame=int(frame))
                    else:
                        state = particles.solve_frame(stream, frame, self._sim_memory, cancel)
                        value = replace(particles.instance_from_state(state, stream, frame),
                                        matrix=incoming.matrix, render_as=incoming.render_as,
                                        size_scale=incoming.size_scale, texture=incoming.texture)
                elif kind == "ParticleRender3D":
                    incoming = values[node["inputs"]["particles"]]
                    if node["disabled"]:
                        value = incoming
                    else:
                        image = node["inputs"].get("image")
                        value = replace(incoming, render_as=params["representation"],
                                        size_scale=params["size_scale"],
                                        texture=None if image is None else values[image].to_display())
                elif kind == "ParticleCache3D":
                    incoming = values[node["inputs"]["particles"]]
                    if node["disabled"] or stream is None:
                        value = incoming
                    else:
                        store = self.sim_store(params["cache_memory_mb"], params["cache_disk_mb"])
                        state = particles.solve_frame(stream, frame, store, cancel)
                        value = replace(particles.instance_from_state(state, stream, frame),
                                        matrix=incoming.matrix, render_as=incoming.render_as,
                                        size_scale=incoming.size_scale, texture=incoming.texture)
                elif kind == "Normals3D":
                    source = values[node["inputs"]["geo"]]
                    value = source if node["disabled"] or source is None else scene3d.normals_from_node(source, params)
                elif kind == "DisplaceGeo3D":
                    source = values[node["inputs"]["geo"]]
                    image = None
                    if not node["disabled"] and node["inputs"].get("image") is not None:
                        image = values[node["inputs"]["image"]]
                    value = source if node["disabled"] or source is None else scene3d.displace_geometry(
                        source, None if image is None else image.to_display(),
                        params["displace_scale"], params["displace_offset"],
                        params["displace_channel"], bool(params["recompute_normals"]))
                elif kind == "ReadSplat3D":
                    value = scene3d.Scene() if node["disabled"] else scene3d.Scene(splats=(
                        scene3d.SplatInstance(splats.load_cloud_cached(
                            params["splat_path"], params["splat_orientation"], params["splat_colorspace"]),
                            scene3d._transform_from(params).matrix(), params["splat_sh_degree"],
                            params["splat_opacity"], params["splat_scale"], params.get("splat_relight", 0.0),
                            params.get("splat_shadow_catch", 0.0),
                            params.get("splat_cast_shadows", "on") == "on",
                            params.get("splat_specular", 0.0)),))
                elif kind in ("ReadUSD3D", "ReadUSDCamera3D"):
                    from . import usdio
                    try:
                        if kind == "ReadUSD3D":
                            value = (scene3d.Scene() if node["disabled"] else
                                     usdio.load_scene(params["usd_path"], frame, params["usd_root"]))
                        else:
                            value = (scene3d.Camera() if node["disabled"] else
                                     usdio.load_camera(params["usd_path"], frame, params["usd_camera"]))
                    except RuntimeError as error:
                        raise ValueError(str(error)) from None
                elif kind == "ReadGLTF3D":
                    from . import gltfio
                    value = (scene3d.Scene() if node["disabled"] else
                             gltfio.load_scene(params["gltf_path"], params["gltf_root"]))
                elif kind in ("ReadAlembic3D", "ReadAlembicCamera3D"):
                    from . import alembicio
                    if kind == "ReadAlembic3D":
                        value = (scene3d.Scene() if node["disabled"] else
                                 alembicio.load_scene(params["abc_path"], seconds, params["abc_root"]))
                    else:
                        value = (scene3d.Camera() if node["disabled"] else
                                 alembicio.load_camera(params["abc_path"], seconds, params["abc_camera"]))
                elif kind == "Light3D":
                    value = None if node["disabled"] else scene3d.light_from_node({"params": params})
                elif kind == "Camera3D":
                    value = scene3d.camera_from_node({"params": params})
                elif kind == "Project3D":
                    value = values[node["inputs"]["geometry"]]
                    if isinstance(value, scene3d.Geometry):
                        value = scene3d.Scene((value,))
                    elif value is None:  # disabled upstream geometry contributes nothing
                        value = scene3d.Scene()
                    if not node["disabled"]:
                        value = scene3d.apply_projection(value, scene3d.Projection(
                            camera=values[node["inputs"]["camera"]],
                            texture=values[node["inputs"]["image"]].to_display(),
                            outside=params["project_outside"], backfaces=params["project_backfaces"],
                            occlusion=params.get("project_occlusion", "off")))
                elif kind == "WriteGeo3D":
                    value = values[node["inputs"]["scene"]]
                elif kind == "Axis3D":
                    # Chaining is ordinary scene nesting: scene_from_node already flattens a Scene
                    # member and multiplies its own matrix onto each item's parent, in order, so two
                    # chained Axis3D compose exactly as Scene3D nesting does. Disabled uses the
                    # identity transform rather than going empty (bypass_slot passes "object"
                    # through), matching how a disabled 2D Transform passes its image unchanged.
                    source = node["inputs"]["object"]
                    member = values[source] if source is not None else None
                    members = [] if member is None else [member]
                    value = scene3d.scene_from_node(
                        {"params": _IDENTITY_XFORM if node["disabled"] else params}, members)
                elif kind == "Scene3D":
                    slots = [node["inputs"].get(s) for s in _SPECS[kind]["optional_inputs"]]
                    members = [values[s] for s in slots if s is not None and values[s] is not None]
                    value = scene3d.Scene() if node["disabled"] else scene3d.scene_from_node(
                        {"params": params}, members)
                elif digest in self.cache:
                    self.hits += 1
                    value = self.cache.pop(digest)
                    self.cache[digest] = value
                else:
                    self.misses += 1
                    if node["disabled"]:  # nothing sensible to pass through: a scene is not an image
                        values[key] = Raster.of(np.zeros((params["height"], params["width"], 4), np.float32))
                        continue
                    scene, camera = (values[node["inputs"][slot]] for slot in ("scene", "camera"))
                    backend = params.get("render_backend", "cpu")
                    if params.get("render_output", "rgba") == "relight":
                        if backend == "gpu":
                            raise ValueError("GPU Render3D unsupported: the relight bundle output is CPU-only for now")
                        backend = "cpu"
                    mode = params.get("render_mode", "raster")
                    args = (scene, camera, params["width"], params["height"],
                            (params["red"], params["green"], params["blue"], params["alpha"]))
                    kwargs = dict(ambient=params["ambient"], samples=params["samples"],
                                  output=params.get("render_output", "rgba"), cancel=cancel, mode=mode)
                    rgba = None
                    if backend != "cpu":
                        from . import gpu3d
                        if gpu3d.available():
                            try:
                                rgba = gpu3d.render(*args, **kwargs)
                            except Cancelled:
                                raise
                            except gpu3d.Unsupported as exc:
                                if backend == "gpu":
                                    raise ValueError(f"GPU Render3D unsupported: {exc}") from exc
                            except Exception as exc:
                                # Device loss, out of memory or an adapter error after available()
                                # said yes: auto falls back to the CPU reference, gpu reports it.
                                if backend == "gpu":
                                    raise ValueError(f"GPU Render3D failed: {exc}") from exc
                        elif backend == "gpu":
                            raise ValueError(f"GPU Render3D unavailable: {gpu3d.describe()}")
                    if rgba is None:
                        rgba = scene3d.render(*args, shadows=True, progress=self.progress, **kwargs)
                    if params.get("render_output", "rgba") == "relight":
                        rgba, layers = rgba
                        value = Raster(rgba, layers={name: Raster.of(arr) for name, arr in layers.items()})
                    else:
                        value = Raster.of(rgba)
                    self._store(digest, value)
                values[key] = value
                continue
            # Time enters the digest only where it changes the result. A Read resolves the concrete
            # file for this frame and fingerprints *that*; a still resolves to the same path at
            # every frame and keeps its cache entry, while a sequence naturally re-keys. Downstream
            # digests already fold in their inputs' hashes, so time-dependence propagates exactly as
            # far as it really reaches. See docs/TIME_MODEL.md.
            fingerprint = None
            if kind == "Read" and params["path"]:
                from .media import nearest_sequence_path, resolve_source_path
                source_frame = frame + int(params.get("frame_offset", 0))
                resolved, exists = resolve_source_path(params["path"], source_frame, params.get("missing", "error"))
                if resolved is None:
                    reference = nearest_sequence_path(params["path"], source_frame)
                    if reference is None:
                        raise ValueError(f'No frames found for sequence {params["path"]}')
                    stat = Path(reference).stat()
                    fingerprint = ["<black>", source_frame, str(Path(reference).resolve()),
                                   stat.st_size, stat.st_mtime_ns]
                else:
                    stat = Path(resolved).stat()
                    fingerprint = [str(Path(resolved).resolve()), stat.st_size, stat.st_mtime_ns]
            if kind == "Relight":
                # Sparse light slots are paired by index, so their positions affect the result.
                fingerprint = [slot for slot, source in active_inputs.items() if source is not None]
            remap_raster = None
            if kind in _TIME_REMAP_KINDS and not node["disabled"]:
                # The fingerprint is the *nested* call's own digest, not this walk's
                # `hashes[source]` (which was computed at the wrong, outer `frame`): a still keeps
                # one cache entry across every effective frame it is asked for, exactly like a
                # Read's fingerprint above, and a curve edited on a keyframe the outer walk never
                # visits at `frame` is still picked up, because the nested call resolves the
                # source's own curves at `effective_frame`, not at `frame`.
                if kind == "AppendClip":
                    # Up to two nested evaluations (two inside a dissolve), each of a different
                    # clip at its own frame; the blend weight is a function of `frame`, so it joins
                    # the fingerprint alongside the nested digests.
                    plan = _append_clip_plan(nodes, node, params, frame)
                    parts = [self.evaluate_raster(doc, target=node["inputs"][slot], cancel=cancel,
                                                  frame=source_frame, tier=tier, typed=True,
                                                  return_digest=True)
                             for slot, source_frame, _ in plan]
                    remap_raster = parts[0][0]
                    fingerprint = ["time_remap", *(digest for _, digest in parts)]
                    if len(parts) == 2:
                        weight = plan[1][2]
                        a, b = parts[0][0], parts[1][0]
                        if a.pixels.shape != b.pixels.shape or a.data != b.data or a.display != b.display:
                            raise ValueError(f"{node['name']}: dissolving clips need matching formats")
                        remap_raster = Raster((a.pixels * np.float32(1.0 - weight)
                                               + b.pixels * np.float32(weight)), a.data, a.display)
                        fingerprint.append(round(weight, 9))
                else:
                    black = False
                    if kind in ("TimeClip", "FrameRange"):
                        effective_frame, black = _time_clip_frame(kind, params, frame)
                    else:
                        effective_frame = _time_remap_frame(kind, params, frame)
                    source_key = node["inputs"][_SPECS[kind]["inputs"][0]]
                    remap_raster, remap_digest = self.evaluate_raster(
                        doc, target=source_key, cancel=cancel, frame=effective_frame, tier=tier,
                        typed=True, return_digest=True)
                    fingerprint = ["time_remap", remap_digest]
                    if black:
                        # Outside the range with before/after "black": a transparent frame at the
                        # size of the nearest in-range frame (`_range_frame` returned that one).
                        remap_raster = Raster(np.zeros_like(remap_raster.pixels), remap_raster.data,
                                              remap_raster.display)
                        fingerprint.append("black")
            # The tier is folded in explicitly rather than left implicit in the scaled parameters:
            # a Grade has no pixel units, so its parameters are identical at every tier while its
            # pixels are not. Without this, a tier 4 result would satisfy a tier 1 request (C1).
            # `data` joins the digest explicitly: a Roto's shapes are not in params, so without
            # this an edited shape would serve the previous matte out of cache. A Tracker needs no
            # term here because its solve already landed in params above.
            #
            # A time-remapping node folds in `fingerprint` (the nested call's digest at
            # `effective_frame`) instead of `hashes[source]`: `hashes[source]` was computed by
            # this walk at the outer `frame`, which is *not* the frame this node's pixels actually
            # come from, and an animated source would make it churn on every outer frame even
            # when `effective_frame` — and so the actual result — does not change (the FrameHold
            # cache-reuse case docs/TIME_MODEL.md and this lane's own tests require).
            source_hashes = ([] if kind in _TIME_REMAP_KINDS and not node["disabled"]
                             else [hashes[s] for s in sources if s is not None])
            digest = hashlib.sha256(json.dumps([kind, params, node["disabled"], source_hashes, fingerprint, tier, data], sort_keys=True).encode()).hexdigest()
            hashes[key] = digest
            # `pixels`, not `frame`: in this module "frame" now means a position in time, and the
            # loop must not clobber the timeline frame that later Reads still need.
            if digest in self.cache:
                self.hits += 1
                raster = self.cache.pop(digest)
                self.cache[digest] = raster
            else:
                self.misses += 1
                # A memory miss consults the disk tier before recomputing. A hit there repopulates
                # memory, so the second read of a spilled result is a memory hit again (C4).
                spilled = self.disk.get_raster(digest)
                if spilled is not None:
                    self.disk_hits += 1
                    self._store(digest, spilled)
                    values[key] = spilled
                    continue
                # Build the inputs list in declared slot order (required then optional) so kernels
                # pick up `image` first and `mask` second. None for optional slots becomes None.
                if node["disabled"]:
                    # Only the passed-through input was evaluated; the other slots have no value.
                    # A Draw node's bypass slot ("image") is itself optional, so a disabled Ramp/
                    # Radial/Rectangle/Noise/Text with nothing wired into it has no upstream value
                    # to look up at all -- it passes a transparent frame at its own declared format
                    # instead, exactly as an unwired bypass on any other optional-image slot would.
                    if sources and sources[0] is not None:
                        raster = values[sources[0]]
                    elif kind in _DRAW_KINDS:
                        raster = Raster.of(np.zeros((int(params["height"]), int(params["width"]), 4),
                                                     np.float32))
                    elif sources and sources[0] is None:
                        raise ValueError(f"{node['name']}: connect at least one clip input")
                    else:
                        raster = values[sources[0]]
                elif kind in _TIME_REMAP_KINDS:
                    raster = remap_raster
                else:
                    slot_sources = [node["inputs"][s] for s in _SPECS[kind]["inputs"]]
                    slot_sources.extend(node["inputs"].get(s) for s in _SPECS[kind].get("optional_inputs", []))
                    images = [values[s] if s is not None else None for s in slot_sources]
                    raster = self._windowed_kernel(kind, params, images, frame, data)
                if tier != 1 and kind == "Read" and not node["disabled"]:
                    # A file cannot be decoded at a fraction of its size, so a Read is the one
                    # source that must decimate after the fact. Everything downstream of it still
                    # runs small, which is where the saving lives. Both windows move with the
                    # pixels; a data window left in full-resolution coordinates would place the
                    # overscan four times too far out at tier 4.
                    decimated = self._decimate(raster.pixels, tier)
                    raster = Raster(decimated,
                                    scale_window(raster.data, tier, decimated.shape[1], decimated.shape[0]),
                                    raster.display.scaled(tier))
                raster.pixels.flags.writeable = False
                self._store(digest, raster)
            values[key] = raster
        result = values[target]
        if not isinstance(result, Raster) and not typed:
            raise ValueError(f"{nodes[target]['name']} outputs {OUTPUT_TYPES.get(nodes[target]['type'])}, "
                             "not an image; view a Render3D instead")
        if return_digest:
            return result, hashes[target]
        return result

    # --- bounding box ---------------------------------------------------------------------------
    #
    # `_kernel` stays a pure array function: it is the reference math and it is tested directly.
    # Everything about *where* an image lives lives here instead, in one place, so a new kernel
    # cannot accidentally invent its own window convention. See docs/BOUNDING_BOX.md.

    @staticmethod
    def _windowed_kernel(kind, p, inputs, frame=None, data=None):
        """Run a kernel with its inputs aligned to the output's data window.

        Two rules decide every case:
          * Which rectangle does this node's output cover? (`_filter_window` for the filters;
            generators state their own; Merge takes the union of its inputs'.)
          * Are its inputs aligned into that rectangle before the array math runs? Always — no
            kernel ever sees two arrays that disagree about where their pixels are.
        """
        from .core import DRAW_KINDS, MASK_MIX_KINDS, MERGE_LIKE_KINDS, WINDOW_KINDS

        if kind == "Read":
            return read_image_raster(**p, frame=frame)
        if kind in ("Constant", "Checker"):
            # A generated source defines the frame: data window and display window coincide.
            return Raster.of(Evaluator._kernel(kind, p, [], frame))
        if kind == "Roto":
            # Also a generator, and deliberately one: a Roto that inherited an input's format
            # could disagree with it about the data window and produce a matte that silently
            # fails to line up with the thing it is masking.
            return Raster.of(Evaluator._kernel(kind, p, [], frame, data))
        if kind in DRAW_KINDS:
            # A fourth kind of generator: still states its own format like Roto, but also takes an
            # optional "image" it composites the shape over, and an optional "mask" -- so unlike
            # Roto it is not exempt from the mask/mix contract every gated filter honours.
            width, height = int(p["width"]), int(p["height"])
            out = Region(0, 0, width, height)
            image, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if image is not None and image.display != out:
                raise ValueError(f"{kind} image input must match its own format in M0")
            if mask is not None and mask.display != out:
                raise ValueError(
                    f"Mask display window {mask.display} does not match {kind}'s own format {out}; "
                    "no silent resampling is performed")
            background = image.fit(out) if image is not None else np.zeros((height, width, 4), np.float32)
            shape = Evaluator._draw_shape(kind, p, 0, 0, width, height, frame)
            composited = Evaluator._composite_shape_over(shape, background)
            pixels = Evaluator._apply_mask_mix(background, composited,
                                               mask.fit(out) if mask is not None else None,
                                               p.get("mix", 1.0))
            return Raster.of(pixels)
        if kind == "Relight":
            bundle_raster = inputs[0]
            camera = inputs[1]  # Accepted for future use; v1 uses the render's baked camera response.
            layers = bundle_raster.layers
            if layers is None:
                raise ValueError("Relight: connect a Render3D node with Output set to 'Relight passes'")
            if "albedo" not in layers:
                raise ValueError("Relight: input bundle is missing the 'albedo' layer")
            if p["mix"] == 0:
                return bundle_raster.with_pixels(bundle_raster.pixels)
            # Ambient is NOT scaled by the Diffuse knob (docs/3D_ROADMAP.md "Design: relight
            # passes"): Diffuse/Specular scale only the per-light response sums, so setting either
            # to 0 turns off that kind of light contribution without also killing the ambient fill.
            ambient = np.empty_like(bundle_raster.pixels[..., :3])
            ambient[...] = (p["red"], p["green"], p["blue"])
            diffuse_sum = np.zeros_like(ambient)
            specular_total = np.zeros_like(ambient)
            for i, light in enumerate(inputs[2:10]):
                if light is None or light.intensity <= 0:
                    continue
                colour = np.asarray(light.color, dtype=np.float32) * light.intensity
                if f"diffuse_L{i}" in layers:
                    diffuse_sum += layers[f"diffuse_L{i}"].pixels[..., :3] * colour
                if f"specular_L{i}" in layers:
                    specular_total += layers[f"specular_L{i}"].pixels[..., :3] * colour
            relit_rgb = (layers["albedo"].pixels[..., :3] * (ambient + diffuse_sum * p["diffuse"])
                         + specular_total * p["specular"])
            result_rgb = p["mix"] * relit_rgb + (1 - p["mix"]) * bundle_raster.pixels[..., :3]
            return bundle_raster.with_pixels(np.concatenate(
                (result_rgb, bundle_raster.pixels[..., 3:4]), axis=-1))
        if kind == "ChannelShuffle":
            a, b = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if b is not None and b.display != a.display:
                raise ValueError(
                    f"ChannelShuffle B display window {b.display} does not match A {a.display}; "
                    "no silent resampling is performed")
            # Pointwise over A's rectangle. B is fitted into it rather than unioned: routing a
            # channel out of B cannot enlarge the picture, it can only recolour the picture A
            # already defines. Area of B outside A's data window reads as transparent black,
            # which is what `fit` produces.
            out = a.data
            return Raster(Evaluator._kernel(kind, p, [a.pixels, None if b is None else b.fit(out)],
                                            frame), out, a.display)
        if kind in MERGE_LIKE_KINDS:
            a, b = inputs[0], inputs[1]
            if a.display != b.display:
                # Differing *display* windows is still a format mistake and still raises, exactly
                # as it did before bounding boxes existed. Differing *data* windows is now normal:
                # that is a plate with overscan merged over one without, which must work.
                raise ValueError(f"{kind} inputs must have matching formats in M0")
            out = a.data.union(b.data)
            mask = inputs[2] if len(inputs) > 2 else None
            if mask is not None and mask.display != b.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match {kind} B {b.display}; "
                    "no silent resampling is performed")
            layers = [a.fit(out), b.fit(out)] + ([] if mask is None else [mask.fit(out)])
            return Raster(Evaluator._kernel(kind, p, layers, frame), out, b.display)
        if kind == "Switch":
            chosen = Evaluator._kernel("Switch", p, [r.pixels if r is not None else None
                                                     for r in inputs[:2]], frame)
            return inputs[int(p["which"])].with_pixels(chosen)
        if kind == "Reformat":
            # Unlike every other MASK_MIX_KINDS member, Reformat's own output *display* window is
            # not `source.display` -- it is the chosen format (docs/PARITY_2D.md) -- so it cannot
            # go through the generic branch below, which always keeps the source's display. mask/
            # mix still follow the same recipe, just gated against the new format's own rectangle.
            source, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if mask is not None and mask.display != source.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match source {source.display}; "
                    "no silent resampling is performed")
            reformat_type = p["reformat_type"]
            if reformat_type == "scale":
                factor = float(p["scale"])
                target_w = max(1, int(round(source.display.width * factor)))
                target_h = max(1, int(round(source.display.height * factor)))
            else:
                target_w, target_h = int(p["width"]), int(p["height"])
            target_display = Region(0, 0, target_w, target_h)
            own_box = target_display
            if p["preserve_bbox"]:
                own_box = target_display.union(Evaluator._reformat_transformed_window(
                    source.data, source.display.width, source.display.height,
                    target_w, target_h, p))
            mix = p["mix"]
            gated = mask is not None or mix != 1.0
            out = own_box.union(source.data) if gated else own_box
            filtered = Evaluator._reformat(source.pixels, source.display.width, source.display.height,
                                           target_w, target_h, p, src_box=source.data, dst_box=out)
            pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                               None if mask is None else mask.fit(out), mix)
            return Raster(pixels, out, target_display)
        if kind in WINDOW_KINDS:
            return Evaluator._window_node(kind, p, inputs[0])
        if kind in MASK_MIX_KINDS:
            source, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if mask is not None and mask.display != source.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match source {source.display}; "
                    "no silent resampling is performed")
            mix = p.get("mix", 1.0)
            filtered_box = Evaluator._filter_window(kind, p, source)
            # A gated filter blends against the untouched source, so the source's own rectangle is
            # part of the answer. An ungated one is replaced outright and keeps only its own.
            gated = mask is not None or mix != 1.0
            out = filtered_box.union(source.data) if gated else filtered_box
            filtered = Evaluator._filtered_pixels(kind, p, source, out)
            pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                               None if mask is None else mask.fit(out), mix)
            return Raster(pixels, out, source.display)
        # Pointwise and pass-through kinds: Viewer, Write, NoOp, Dot, Shuffle, Premult, Unpremult.
        source = inputs[0]
        return source.with_pixels(Evaluator._kernel(kind, p, [source.pixels], frame))

    @staticmethod
    def _window_node(kind, p, source):
        """Position, BlackOutside and AdjustBBox: move or resize the data window, never resample.

        * Position: window and pixels shift together by whole pixels; the display window stays put.
        * BlackOutside: the window grows one pixel on every side and the new ring is 0 (black and
          transparent, the same "no pixel here" every uncovered area already is), so a filter
          downstream reads black at the old edge instead of extending the edge pixel.
        * AdjustBBox: `numpixels` grows the window (zero fill) or, negative, shrinks it (pixels
          outside are dropped); pixels that stay are byte-identical and stay where they were.
          `clip_to_format` then intersects the result with the display window.
        """
        box = source.data
        if kind == "Position":
            box = Region(box.x + int(p["translate_x"]), box.y + int(p["translate_y"]), box.width, box.height)
            return Raster(source.pixels, box, source.display)
        if box.is_empty:
            return source
        if kind == "BlackOutside":
            grown = box.expand(1, 1)
        else:
            grown = box.expand(int(p["numpixels"]), int(p["numpixels"]))
            if grown.is_empty:
                grown = Region(box.x, box.y, 0, 0)
            if p.get("clip_to_format", 0):
                grown = grown.intersect(source.display)
        return Raster(source.fit(grown), grown, source.display)

    @staticmethod
    def _filter_window(kind, p, source):
        """The rectangle a filter's own output covers, before any mask/mix blending."""
        if kind == "Crop":
            # Crop's whole meaning is "the output is this rectangle". Shrinking the data window
            # here is what stops every downstream node from computing over discarded area.
            return source.data.intersect(Region(int(p["x"]), int(p["y"]),
                                                int(p["width"]), int(p["height"])))
        if kind in ("Transform", "Tracker"):
            return Evaluator._transformed_window(source.data, p)
        if kind == "CornerPin":
            return Evaluator._cornerpin_window(source.data, p)
        # Grade, ColorCorrect and Blur are all in-place with respect to geometry. Blur notably does
        # NOT grow its box: growing it would change the box filter's edge handling from "extend the
        # edge pixel" to "average in transparent black", which is a visible change to every
        # existing comp. Recorded as a deliberate limitation in docs/BOUNDING_BOX.md.
        return source.data

    @staticmethod
    def _transformed_window(box, p):
        """Forward-map the corners of `box` and take the bounding rectangle.

        The exact inverse of `_transform`'s sampling map, so the two cannot drift: there,
        src = center + R(-theta) * (dst - center - translate) / scale.
        """
        if box.is_empty:
            return box
        theta = math.radians(float(p.get("rotate", 0.0)))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        scale = float(p.get("scale", 1.0))
        cx, cy = float(p.get("center_x", 0.0)), float(p.get("center_y", 0.0))
        tx, ty = float(p.get("translate_x", 0.0)), float(p.get("translate_y", 0.0))
        xs, ys = [], []
        for px, py in ((box.x, box.y), (box.right, box.y), (box.x, box.bottom), (box.right, box.bottom)):
            u, v = (px - cx) * scale, (py - cy) * scale
            xs.append(u * cos_t - v * sin_t + cx + tx)
            ys.append(u * sin_t + v * cos_t + cy + ty)
        # Reconstruction reach of the filter, in destination pixels. Rounding outward and adding
        # the support keeps every pixel the resampler can actually write inside the window; a
        # window one pixel short would clip a rotated edge and look like a rendering bug.
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(p.get("filter", "nearest"), 2)
        margin = int(math.ceil(support * max(1.0, abs(scale))))
        left, top = math.floor(min(xs)) - margin, math.floor(min(ys)) - margin
        right, bottom = math.ceil(max(xs)) + margin, math.ceil(max(ys)) + margin
        return Region(int(left), int(top), int(right - left), int(bottom - top))

    @staticmethod
    def _solve_homography(src_pts, dst_pts):
        """The 3x3 projective matrix mapping four `src_pts` onto four `dst_pts` exactly.

        Standard 4-point DLT with h33 fixed at 1: eight linear equations (two per
        correspondence) for the eight remaining unknowns, solved exactly rather than by least
        squares, since four correspondences fully determine a projective map. `to == from`
        (CornerPin's own default) solves to the identity matrix, which is the brief's own
        worked example.
        """
        a = np.zeros((8, 8), dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        for i, ((x, y), (u, v)) in enumerate(zip(src_pts, dst_pts)):
            a[2 * i] = [x, y, 1, 0, 0, 0, -x * u, -y * u]
            b[2 * i] = u
            a[2 * i + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v]
            b[2 * i + 1] = v
        h = np.linalg.solve(a, b)
        return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])

    @staticmethod
    def _cornerpin_forward_matrix(p):
        """The homography mapping a pre-warp canvas point to its post-warp destination point.

        `direction` "forward" (Nuke's default) warps the "from" quad onto the "to" quad;
        "inverse" swaps which quad is the pre-warp side, by using the same solved homography's
        inverse instead. Sharing this one function between the kernel (which needs its inverse,
        destination -> source) and `_cornerpin_window` (which needs it forward, source ->
        destination) is what keeps the two from drifting apart the way `_transform` and
        `_transformed_window` are kept in sync by hand -- see that pair's own docstring.
        """
        homography = Evaluator._solve_homography(
            [(p["from1_x"], p["from1_y"]), (p["from2_x"], p["from2_y"]),
             (p["from3_x"], p["from3_y"]), (p["from4_x"], p["from4_y"])],
            [(p["to1_x"], p["to1_y"]), (p["to2_x"], p["to2_y"]),
             (p["to3_x"], p["to3_y"]), (p["to4_x"], p["to4_y"])])
        return homography if p["direction"] == "forward" else np.linalg.inv(homography)

    @staticmethod
    def _cornerpin_window(box, p):
        """Forward-map the corners of `box` and take the bounding rectangle.

        Exact for a proper (non-self-intersecting, horizon-clear) quad: a projective transform
        maps straight edges to straight edges, so a rectangle's extrema are still at its four
        mapped corners, the same reasoning `_transformed_window` uses for the affine case.
        """
        if box.is_empty:
            return box
        forward = Evaluator._cornerpin_forward_matrix(p)
        xs, ys = [], []
        for px, py in ((box.x, box.y), (box.right, box.y), (box.x, box.bottom), (box.right, box.bottom)):
            denom = forward[2, 0] * px + forward[2, 1] * py + forward[2, 2]
            denom = denom if abs(denom) > 1e-9 else 1e-9
            xs.append((forward[0, 0] * px + forward[0, 1] * py + forward[0, 2]) / denom)
            ys.append((forward[1, 0] * px + forward[1, 1] * py + forward[1, 2]) / denom)
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(p.get("filter", "nearest"), 2)
        left, top = math.floor(min(xs)) - support, math.floor(min(ys)) - support
        right, bottom = math.ceil(max(xs)) + support, math.ceil(max(ys)) + support
        return Region(int(left), int(top), int(right - left), int(bottom - top))

    @staticmethod
    def _filtered_pixels(kind, p, source, out: Region):
        """The filter's result, evaluated over exactly the rectangle `out`."""
        if kind == "Crop":
            pixels = np.zeros((out.height, out.width, 4), dtype=np.float32)
            keep = out.intersect(source.data).intersect(
                Region(int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"])))
            if not keep.is_empty:
                pixels[keep.y - out.y:keep.bottom - out.y,
                       keep.x - out.x:keep.right - out.x] = source.fit(keep)
            return pixels
        if kind in ("Transform", "Tracker"):
            return Evaluator._transform(source.pixels, p["translate_x"], p["translate_y"], p["rotate"],
                                        p["scale"], p["center_x"], p["center_y"], p["filter"],
                                        src_box=source.data, dst_box=out)
        if kind == "CornerPin":
            return Evaluator._cornerpin(source.pixels, p, src_box=source.data, dst_box=out)
        if kind == "Grade":
            return Evaluator._grade(source.fit(out), p)
        if kind == "ColorCorrect":
            return Evaluator._color_correct(source.fit(out), p)
        if kind == "Blur":
            return Evaluator._blur(source.fit(out), p)
        if kind == "Invert":
            return Evaluator._invert(source.fit(out), p)
        if kind == "Clamp":
            return Evaluator._clamp(source.fit(out), p)
        if kind == "Multiply":
            return Evaluator._channel_multiply(source.fit(out), p)
        if kind == "Add":
            return Evaluator._channel_add(source.fit(out), p)
        if kind == "Gamma":
            return Evaluator._channel_gamma(source.fit(out), p)
        if kind == "Saturation":
            return Evaluator._saturation(source.fit(out), p)
        if kind == "Exposure":
            return Evaluator._exposure(source.fit(out), p)
        if kind == "HueCorrect":
            return Evaluator._hue_correct(source.fit(out), p)
        if kind == "ColorMatrix":
            return Evaluator._color_matrix(source.fit(out), p)
        if kind == "Keyer":
            return Evaluator._keyer(source.fit(out), p)
        if kind == "HueKeyer":
            return Evaluator._hue_keyer(source.fit(out), p)
        if kind == "Erode":
            return Evaluator._erode(source.fit(out), p)
        if kind == "Dilate":
            return Evaluator._dilate(source.fit(out), p)
        if kind == "Median":
            return Evaluator._median(source.fit(out), p)
        if kind == "Sharpen":
            return Evaluator._sharpen(source.fit(out), p)
        if kind == "Glow":
            return Evaluator._glow(source.fit(out), p)
        if kind == "Soften":
            return Evaluator._soften(source.fit(out), p)
        if kind == "Defocus":
            return Evaluator._defocus(source.fit(out), p)
        if kind == "DropShadow":
            return Evaluator._drop_shadow(source.fit(out), p)
        if kind == "DirBlur":
            # Radial and zoom are centred on a canvas point; the kernel sees only an array, so the
            # centre is handed over relative to this rectangle's own corner.
            local = dict(p, center_x=float(p.get("center_x", 0.0)) - out.x,
                         center_y=float(p.get("center_y", 0.0)) - out.y)
            return Evaluator._dirblur(source.fit(out), local)
        if kind == "Mirror":
            return Evaluator._mirror(source.fit(out), p)
        raise ValueError(f"No windowed filter for {kind}")

    @staticmethod
    def _decimate(pixels, tier):
        """Area-average downscale by an integer factor, matching ceil(n / tier) extents.

        Area averaging rather than point sampling because a proxy is judged by whether it predicts
        the full-resolution result; point sampling aliases a detailed plate into a different image
        and would make the tier lie about the shot.

        Averages by adding `tier` strided views and scaling once, rather than a single
        `reshape(..., tier, ..., tier, ...).mean(axis=(1, 3))` call: the reshape-and-multi-axis-
        mean forces NumPy to walk the array in an access pattern that defeats its fast reduction
        loops, measured ~8x slower than this for both tier 2 and tier 4 on a 4K frame
        (docs/BENCHMARKS-v0.17-playback.md) despite doing the identical arithmetic (a sum of the
        same `tier * tier` input values divided by `tier * tier`, just accumulated in a
        different order, so results agree with the old implementation to float32 rounding).
        """
        tier = int(tier)
        if tier == 1:
            return pixels
        height, width = pixels.shape[:2]
        rows, columns = -(-height // tier), -(-width // tier)
        pad_y, pad_x = rows * tier - height, columns * tier - width
        if pad_y or pad_x:
            # Edge padding, so a plate whose size is not a multiple of the tier keeps its border
            # value instead of averaging in black and darkening its last row.
            pixels = np.pad(pixels, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        rows_reduced = pixels[0::tier]
        for offset in range(1, tier):
            rows_reduced = rows_reduced + pixels[offset::tier]
        rows_reduced *= np.float32(1.0 / tier)
        columns_reduced = rows_reduced[:, 0::tier]
        for offset in range(1, tier):
            columns_reduced = columns_reduced + rows_reduced[:, offset::tier]
        columns_reduced *= np.float32(1.0 / tier)
        return columns_reduced.astype(np.float32)

    @staticmethod
    def _box_blur_axis(frame, radius, axis):
        # O(n) sliding-window box filter via cumulative sum; "edge" padding avoids darkening borders.
        r = int(round(radius))
        if r <= 0:
            return frame
        n = frame.shape[axis]
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (r, r)
        padded = np.pad(frame, pad, mode="edge").astype(np.float64)
        zero_shape = list(padded.shape)
        zero_shape[axis] = 1
        cumsum = np.concatenate([np.zeros(zero_shape, dtype=np.float64), np.cumsum(padded, axis=axis)], axis=axis)
        window = 2 * r + 1
        hi = [slice(None)] * frame.ndim
        hi[axis] = slice(window, window + n)
        lo = [slice(None)] * frame.ndim
        lo[axis] = slice(0, n)
        return ((cumsum[tuple(hi)] - cumsum[tuple(lo)]) / window).astype(np.float32)

    @staticmethod
    def _kernel(kind, p, inputs, frame=None, data=None):
        if kind == "Read":
            # Only Read consumes time today. Animated parameters will make `frame` matter to the
            # rest of these kernels; the argument exists so that is an addition, not a signature
            # change rippling through every node. See docs/TIME_MODEL.md.
            return read_image(**p, frame=frame)
        if kind == "Constant":
            color = np.array([p["red"] * p["alpha"], p["green"] * p["alpha"], p["blue"] * p["alpha"], p["alpha"]], dtype=np.float32)
            return np.broadcast_to(color, (p["height"], p["width"], 4)).copy()
        if kind == "Checker":
            yy, xx = np.ogrid[:p["height"], :p["width"]]
            pattern = ((xx // p["size"] + yy // p["size"]) % 2).astype(np.float32)
            frame = np.ones((p["height"], p["width"], 4), np.float32)
            frame[..., :3] = (0.06 + pattern * 0.24)[..., None]
            return frame
        if kind in ("Viewer", "Write", "NoOp"):
            # Write is a tap, not a transform: rendering it is an explicit action, and the pixels
            # continue downstream untouched so parking one mid-branch changes nothing.
            return inputs[0]
        if kind == "Grade":
            filtered = Evaluator._grade(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "ColorCorrect":
            filtered = Evaluator._color_correct(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Blur":
            filtered = Evaluator._blur(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("Transform", "Tracker"):
            filtered = Evaluator._transform(inputs[0], p["translate_x"], p["translate_y"], p["rotate"],
                                              p["scale"], p["center_x"], p["center_y"], p["filter"])
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Crop":
            filtered = Evaluator._crop(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Invert":
            filtered = Evaluator._invert(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Clamp":
            filtered = Evaluator._clamp(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Multiply":
            filtered = Evaluator._channel_multiply(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Add":
            filtered = Evaluator._channel_add(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Gamma":
            filtered = Evaluator._channel_gamma(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Saturation":
            filtered = Evaluator._saturation(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Keyer":
            filtered = Evaluator._keyer(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "HueKeyer":
            filtered = Evaluator._hue_keyer(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Erode":
            filtered = Evaluator._erode(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Dilate":
            filtered = Evaluator._dilate(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Median":
            filtered = Evaluator._median(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Sharpen":
            filtered = Evaluator._sharpen(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Glow":
            filtered = Evaluator._glow(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Exposure":
            filtered = Evaluator._exposure(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "HueCorrect":
            filtered = Evaluator._hue_correct(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "ColorMatrix":
            filtered = Evaluator._color_matrix(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Soften":
            filtered = Evaluator._soften(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Defocus":
            filtered = Evaluator._defocus(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "DropShadow":
            filtered = Evaluator._drop_shadow(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "DirBlur":
            filtered = Evaluator._dirblur(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Mirror":
            filtered = Evaluator._mirror(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Dissolve":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Dissolve inputs must have matching formats in M0")
            blended = Evaluator._dissolve(a, b, p)
            return Evaluator._apply_mask_mix(b, blended, mask=mask, mix=p.get("mix", 1.0))
        if kind == "Keymix":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Keymix inputs must have matching formats in M0")
            key_mask = mask
            if key_mask is not None and p.get("invert_mask"):
                key_mask = np.concatenate([key_mask[..., :3], 1 - key_mask[..., 3:4]], axis=-1)
            return Evaluator._apply_mask_mix(b, a, mask=key_mask, mix=p.get("mix", 1.0))
        if kind == "Copy":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Copy inputs must have matching formats in M0")
            copied = Evaluator._copy_channels(a, b, p)
            return Evaluator._apply_mask_mix(b, copied, mask=mask, mix=p.get("mix", 1.0))
        if kind == "ChannelMerge":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("ChannelMerge inputs must have matching formats in M0")
            merged = Evaluator._channel_merge(a, b, p)
            return Evaluator._apply_mask_mix(b, merged, mask=mask, mix=p.get("mix", 1.0))
        if kind == "Difference":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Difference inputs must have matching formats in M0")
            keyed = Evaluator._difference_key(a, b, p)
            return Evaluator._apply_mask_mix(b, keyed, mask=mask, mix=p.get("mix", 1.0))
        if kind == "Roto":
            from . import roto
            return roto.rasterise(data or [], p["width"], p["height"], bool(p.get("invert", 0)))
        if kind == "ChannelShuffle":
            a, b = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            index = {"r": 0, "g": 1, "b": 2, "a": 3}
            h, w = a.shape[:2]

            def pick(name):
                if name in ("0", "1"):
                    return np.full((h, w, 1), float(name), dtype=np.float32)
                which, channel = name.split(".")
                if which == "B":
                    if b is None:
                        # Black would be a plausible-looking result for a wiring mistake, and the
                        # artist would find it in a review rather than here.
                        raise ValueError(f"ChannelShuffle routes {name} but input B is not connected")
                    return b[..., index[channel]:index[channel] + 1]
                return a[..., index[channel]:index[channel] + 1]
            return np.concatenate([pick(p["out_red"]), pick(p["out_green"]),
                                   pick(p["out_blue"]), pick(p["out_alpha"])], axis=2).astype(np.float32)
        if kind == "Shuffle":
            source = inputs[0]
            index = {"R": 0, "G": 1, "B": 2, "A": 3}
            h, w = source.shape[:2]
            def pick(name):
                if name in index:
                    return source[..., index[name]:index[name] + 1]
                return np.full((h, w, 1), float(name), dtype=np.float32)
            return np.concatenate([pick(p["red_from"]), pick(p["green_from"]), pick(p["blue_from"]), pick(p["alpha_from"])], axis=2).astype(np.float32)
        if kind == "Merge":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Merge inputs must have matching formats in M0")
            return Evaluator._merge_gated(p.get("operation", "over"), a, b, p["mix"], mask)
        if kind == "Premult":
            frame = inputs[0].copy()
            alpha = frame[..., 3:4]
            frame[..., :3] = frame[..., :3] * alpha
            return frame
        if kind == "Unpremult":
            frame = inputs[0].copy()
            alpha = frame[..., 3:4]
            # Guard alpha == 0 to keep RGB untouched (avoids NaN/inf while preserving the premultiplied channel).
            safe = np.where(alpha > 0, alpha, np.float32(1.0))
            straight = np.where(alpha > 0, frame[..., :3] / safe, frame[..., :3])
            return np.concatenate([straight, alpha], axis=2).astype(np.float32)
        if kind == "Dot":
            # Graph passthrough: wire reroute, pixel data unchanged. The optional_mask/mix
            # contract does not apply; Dot is intentionally neutral.
            return inputs[0].copy()
        if kind == "Switch":
            which = int(p["which"])
            choices = inputs[:2]  # only the two declared inputs ("0" and "1") are selectable
            if which < 0 or which >= len(choices):
                raise ValueError(f"Switch 'which'={which} is out of range for {len(choices)} wired inputs")
            return choices[which].copy()
        raise ValueError(f"No kernel for {kind}")

    @staticmethod
    def _merge_gated(op, a, b, mix, mask=None):
        """Merge A onto B, gated by `mix` and, when wired, by the mask's alpha per pixel.

        gate = mix * mask.a (a scalar `mix` when the mask is unwired). `over` scales A by the gate
        before compositing; every other operation blends its result against B by the gate. With no
        mask this is byte-identical to the pre-mask Merge, and where mask.a is 0 the output is B.
        """
        if mask is not None and mask.shape[:2] != b.shape[:2]:
            raise ValueError("Merge mask must match the merged format")
        gate = np.float32(mix) if mask is None else np.float32(mix) * mask[..., 3:4]
        if op == "over":
            # Keep the v0.3.0 mix behaviour byte-identical: scale A by mix, then standard over.
            return a * gate + b * (1 - (a[..., 3:4] * gate))
        full = Evaluator._merge_op(op, a, b)
        return full * gate + b * (1 - gate)

    @staticmethod
    def _merge_op(op, a, b):
        """Nuke-style premultiplied compositing. A is the foreground, B is the background.
        Formulas verified against Nuke's documented merge math: in/out key off B's alpha (shape
        clipping the foreground), mask/stencil key off A's alpha (shape clipping the background),
        and the remaining Porter-Duff operators follow the standard premultiplied composition
        conventions. HDR-safe: plus/minus/screen produce unclamped values."""
        aa = a[..., 3:4]
        ba = b[..., 3:4]
        if op == "over":
            return a + b * (1 - aa)
        if op == "under":
            # Symmetric swap of over: B with A behind it.
            return b + a * (1 - ba)
        if op == "plus":
            return a + b
        if op == "minus":
            return a - b
        if op == "multiply":
            return a * b
        if op == "screen":
            return a + b - a * b
        if op == "max":
            return np.maximum(a, b)
        if op == "min":
            return np.minimum(a, b)
        if op == "difference":
            return np.abs(a - b)
        if op == "divide":
            # Element-wise safe divide: where |B| <= epsilon the output is zero.
            safe = np.where(np.abs(b) > 1e-6, b, np.float32(1.0))
            return np.where(np.abs(b) > 1e-6, a / safe, np.float32(0.0)).astype(np.float32)
        if op == "mask":
            # Nuke "mask": B's RGB modulated by A's alpha (the foreground's shape clips the background).
            return b * aa
        if op == "stencil":
            # Nuke "stencil": B's RGB modulated by (1 - A's alpha) — the inverse mask.
            return b * (1 - aa)
        if op == "in":
            return a * ba
        if op == "out":
            return a * (1 - ba)
        if op == "atop":
            return a * ba + b * (1 - aa)
        if op == "xor":
            return a * (1 - ba) + b * (1 - aa)
        if op == "average":
            return (a + b) * np.float32(0.5)
        if op == "from":
            return b - a
        if op == "hypot":
            return np.sqrt(a * a + b * b)
        return Evaluator._merge_op_extra(op, a, b, aa, ba)

    @staticmethod
    def _merge_op_extra(op, a, b, aa, ba):
        """The eleven Nuke operations added in step 3a, shared by `_merge_op` (aa/ba are the
        alpha slices) and `_channel_merge_op` (aa/ba are the channel values themselves).

        Formulas follow Foundry's documented merge algorithms. Division convention: every
        divisor with magnitude <= 1e-6 is treated as unusable and the guarded term is replaced
        by the limit chosen below, so zero and HDR inputs always give finite results.
        disjoint-over: the B(1-a)/b term is 0 where b is unusable. conjoint-over: A alone where
        b is unusable. geometric: 0 where A+B is unusable. color-dodge: 1 where 1-A is unusable
        and B > 0 (else 0). color-burn: 0 where A is unusable, except 1 when B >= 1.
        soft-light keeps Foundry's documented A*B < 1 branch test on the raw values.
        """
        eps = np.float32(1e-6)
        one = np.float32(1.0)
        zero = np.float32(0.0)
        if op == "matte":
            return a * aa + b * (1 - aa)
        if op == "disjoint-over":
            safe = np.where(np.abs(ba) > eps, ba, one)
            tail = np.where(np.abs(ba) > eps, b * (1 - aa) / safe, zero)
            return np.where(aa + ba < 1, a + b, a + tail).astype(np.float32)
        if op == "conjoint-over":
            safe = np.where(np.abs(ba) > eps, ba, one)
            ratio = np.where(np.abs(ba) > eps, aa / safe, one)
            return np.where((aa > ba) | (np.abs(ba) <= eps), a, a + b * (1 - ratio)).astype(np.float32)
        if op == "copy":
            return a.copy()
        if op == "exclusion":
            return a + b - 2 * a * b
        if op == "geometric":
            total = a + b
            safe = np.where(np.abs(total) > eps, total, one)
            return np.where(np.abs(total) > eps, 2 * a * b / safe, zero).astype(np.float32)
        if op == "overlay":
            return np.where(b < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b)).astype(np.float32)
        if op == "hard-light":
            return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b)).astype(np.float32)
        if op == "soft-light":
            ab = a * b
            return np.where(ab < 1, b * (2 * a + b * (1 - ab)), 2 * ab).astype(np.float32)
        if op == "color-dodge":
            gap = 1 - a
            safe = np.where(gap > eps, gap, one)
            limit = np.where(b > 0, one, zero)
            return np.where(gap > eps, b / safe, limit).astype(np.float32)
        if op == "color-burn":
            safe = np.where(a > eps, a, one)
            limit = np.where(b >= 1, one, zero)
            return np.where(a > eps, 1 - (1 - b) / safe, limit).astype(np.float32)
        raise ValueError(f"Unknown merge operation: {op}")

    @staticmethod
    def _channel_merge_op(op, a, b):
        """Single-channel counterpart of `_merge_op`.

        ChannelMerge's A and B are one selected scalar channel each, not full RGBA, so there is no
        separate alpha to key off. Nuke's own convention (and the one this mirrors) is that each
        side's own value stands in for its alpha, which is why `aa` and `ba` are just `a` and `b`
        rather than a fourth channel slice.
        """
        aa, ba = a, b
        if op == "over":
            return a + b * (1 - aa)
        if op == "under":
            return b + a * (1 - ba)
        if op == "plus":
            return a + b
        if op == "minus":
            return a - b
        if op == "multiply":
            return a * b
        if op == "screen":
            return a + b - a * b
        if op == "max":
            return np.maximum(a, b)
        if op == "min":
            return np.minimum(a, b)
        if op == "difference":
            return np.abs(a - b)
        if op == "divide":
            safe = np.where(np.abs(b) > 1e-6, b, np.float32(1.0))
            return np.where(np.abs(b) > 1e-6, a / safe, np.float32(0.0)).astype(np.float32)
        if op == "mask":
            return b * aa
        if op == "stencil":
            return b * (1 - aa)
        if op == "in":
            return a * ba
        if op == "out":
            return a * (1 - ba)
        if op == "atop":
            return a * ba + b * (1 - aa)
        if op == "xor":
            return a * (1 - ba) + b * (1 - aa)
        if op == "average":
            return (a + b) * np.float32(0.5)
        if op == "from":
            return b - a
        if op == "hypot":
            return np.sqrt(a * a + b * b)
        return Evaluator._merge_op_extra(op, a, b, aa, ba)

    # Channel sets for the "channels" knob on Invert/Clamp/Multiply/Add/Gamma. "rgba" reaches
    # alpha too; the default "rgb" leaves it untouched, matching Nuke's own default.
    _CHANNEL_SETS = {"rgb": (0, 1, 2), "rgba": (0, 1, 2, 3), "alpha": (3,)}

    @staticmethod
    def _invert(image, p):
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = 1.0 - image[..., c]
        return out

    @staticmethod
    def _clamp(image, p):
        out = image.copy()
        lo = p.get("minimum", 0.0) if p.get("clamp_min", 1) else -np.inf
        hi = p.get("maximum", 1.0) if p.get("clamp_max", 1) else np.inf
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = np.clip(image[..., c], lo, hi)
        return out

    @staticmethod
    def _channel_multiply(image, p):
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = image[..., c] * p.get("multiply", 1.0)
        return out

    @staticmethod
    def _channel_add(image, p):
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = image[..., c] + p.get("offset", 0.0)
        return out

    @staticmethod
    def _channel_gamma(image, p):
        out = image.copy()
        g = p.get("gamma", 1.0)
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            v = image[..., c]
            # sign * |x|^(1/gamma), as ColorCorrect's gamma does: avoids raising a negative HDR
            # value to a fractional power.
            out[..., c] = np.sign(v) * np.abs(v) ** (1.0 / g)
        return out

    @staticmethod
    def _saturation(image, p):
        # Unlike Multiply/Add/Gamma, Saturation has no channels knob in Nuke: it is inherently an
        # RGB-luma operation and always leaves alpha untouched.
        rgb = image[..., :3]
        luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
        out_rgb = luma + (rgb - luma) * p.get("saturation", 1.0)
        return np.concatenate([out_rgb, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _range_ramp(value, a, b, c, d):
        """Nuke Keyer's four-point range: 0 at or below `a`, ramping to 1 between `a` and `b`, 1
        from `b` through `c`, ramping back to 0 between `c` and `d`, 0 at or above `d`.

        Checked as a waterfall of exclusive conditions (each `elif` only reached once the ones
        above it are false) rather than independent masks, so a degenerate zero-width segment
        (a == b, or c == d) falls straight through to its neighbour instead of two masks
        disagreeing about who owns the boundary point. `np.where(b > a, b - a, 1.0)` (and the `d`/
        `c` equivalent) only guards the division: when the ramp is degenerate that branch's mask
        is empty, so the substitute denominator is never actually used.
        """
        rise_span = np.where(b > a, b - a, 1.0)
        fall_span = np.where(d > c, d - c, 1.0)
        return np.where(value <= a, 0.0,
                        np.where(value < b, (value - a) / rise_span,
                                 np.where(value <= c, 1.0,
                                          np.where(value < d, 1.0 - (value - c) / fall_span, 0.0))))

    @staticmethod
    def _keyer(image, p):
        # Alpha only; RGB passes through untouched (Nuke's own Keyer never recolours, it only
        # writes a new matte for whatever's downstream -- typically a Premult -- to use).
        rgb = image[..., :3]
        r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        op = p.get("keyer_operation", "luminance")
        if op == "luminance":
            value = 0.2126 * r + 0.7152 * g + 0.0722 * b
        elif op == "red":
            value = r
        elif op == "green":
            value = g
        elif op == "blue":
            value = b
        elif op == "min":
            value = np.minimum(np.minimum(r, g), b)
        elif op == "max":
            value = np.maximum(np.maximum(r, g), b)
        elif op == "saturation":
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            value = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1.0), 0.0)
        else:
            raise ValueError(f"Unknown Keyer operation: {op}")
        alpha = Evaluator._range_ramp(value, p.get("range_a", 0.0), p.get("range_b", 0.0),
                                      p.get("range_c", 1.0), p.get("range_d", 1.0))
        if p.get("invert"):
            alpha = 1.0 - alpha
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _rgb_to_hue_sat(rgb):
        """Standard HSV hue (degrees, 0..360) and saturation (0..1) for a premultiplied-agnostic
        RGB triplet -- both are ratios of the raw channel values, so premult status doesn't matter
        the way it would for an additive quantity."""
        r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        delta = mx - mn
        safe_delta = np.where(delta > 1e-12, delta, 1.0)
        hue = np.where(mx == r, ((g - b) / safe_delta) % 6.0,
                       np.where(mx == g, ((b - r) / safe_delta) + 2.0, ((r - g) / safe_delta) + 4.0))
        hue = np.where(delta > 1e-12, hue * 60.0, 0.0) % 360.0
        safe_mx = np.where(mx > 1e-12, mx, 1.0)
        sat = np.where(mx > 1e-12, delta / safe_mx, 0.0)
        return hue, sat

    @staticmethod
    def _edge_falloff(distance, plateau, softness):
        """1 at or inside `plateau`, ramping down to 0 over the next `softness`, 0 beyond. A
        continuous formula rather than a branch on `softness == 0`: the tiny `1e-6` floor on the
        span makes that case an effectively-sharp (not literally infinite-slope) cutoff, which
        keeps this differentiable-everywhere like the rest of the evaluator's math."""
        span = max(float(softness), 1e-6)
        return np.clip((plateau + span - distance) / span, 0.0, 1.0)

    @staticmethod
    def _hue_keyer(image, p):
        rgb = image[..., :3]
        hue, sat = Evaluator._rgb_to_hue_sat(rgb)
        center = float(p.get("hue_center", 0.0)) % 360.0
        raw_dist = np.abs(hue - center) % 360.0
        dist = np.minimum(raw_dist, 360.0 - raw_dist)
        hue_gate = Evaluator._edge_falloff(dist, float(p.get("hue_width", 30.0)) / 2.0,
                                           p.get("hue_softness", 15.0))
        sat_gate = ((sat >= p.get("sat_min", 0.0)) & (sat <= p.get("sat_max", 1.0))).astype(np.float32)
        alpha = (hue_gate * sat_gate).astype(np.float32)
        if p.get("invert"):
            alpha = 1.0 - alpha
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _box_extreme_axis(frame, support, axis, use_max):
        """One separable pass of a box min/max filter: like `_box_blur_axis`'s sliding window, but
        min/max cannot use a cumulative-sum shortcut, so this walks the `2*support+1` offsets
        directly. "edge" padding matches Blur's own border handling."""
        if support <= 0:
            return frame
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (support, support)
        padded = np.pad(frame, pad, mode="edge")
        n = frame.shape[axis]
        reduce_fn = np.maximum if use_max else np.minimum
        result = None
        for offset in range(2 * support + 1):
            sl = [slice(None)] * frame.ndim
            sl[axis] = slice(offset, offset + n)
            window = padded[tuple(sl)]
            result = window if result is None else reduce_fn(result, window)
        return result

    @staticmethod
    def _box_extreme(image, size, use_max):
        # A box min/max filter is separable (unlike a general convolution) because the box
        # structuring element itself separates into independent x and y passes.
        support = 0 if abs(size) < 0.5 else int(math.ceil(abs(size)))
        if support == 0:
            return image.copy()
        return Evaluator._box_extreme_axis(
            Evaluator._box_extreme_axis(image, support, axis=1, use_max=use_max),
            support, axis=0, use_max=use_max)

    @staticmethod
    def _morph(image, size, channels):
        """Shared box erode/dilate kernel: positive `size` erodes (min filter, shrinks bright
        regions), negative `size` dilates (max filter, grows them) -- exactly Nuke's own signed
        Erode (fast) "size" knob."""
        out = image.copy()
        filtered = Evaluator._box_extreme(image, size, use_max=(size < 0))
        for c in Evaluator._CHANNEL_SETS[channels]:
            out[..., c] = filtered[..., c]
        return out

    @staticmethod
    def _erode(image, p):
        return Evaluator._morph(image, p.get("erode_size", 1.0), p.get("channels", "rgba"))

    @staticmethod
    def _dilate(image, p):
        # Dilate is Erode's positive twin: a positive dilate_size must grow, so it is handed to
        # the shared kernel negated (Erode's own convention for "grow").
        return Evaluator._morph(image, -p.get("dilate_size", 1.0), p.get("channels", "rgba"))

    @staticmethod
    def _median(image, p):
        size = p.get("median_size", 1.0)
        support = 0 if size < 0.5 else int(math.ceil(size))
        out = image.copy()
        if support == 0:
            return out
        h, w = image.shape[:2]
        padded = np.pad(image, ((support, support), (support, support), (0, 0)), mode="edge")
        window = 2 * support + 1
        # A true 2D median is not separable, unlike box blur/erode/dilate, so every tap in the
        # window is stacked and reduced at once. Fine at the small radii these tests and typical
        # despeckle work use; a large radius would want a running-histogram median instead.
        stack = np.stack([padded[dy:dy + h, dx:dx + w] for dy in range(window) for dx in range(window)])
        med = np.median(stack, axis=0).astype(np.float32)
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            out[..., c] = med[..., c]
        return out

    @staticmethod
    def _sharpen(image, p):
        # Unsharp mask: add back `amount` of the high-frequency detail a box blur removed.
        # A flat image has no detail (blurred == image), so it is the identity at any amount.
        amount = p.get("sharpen_amount", 0.5)
        size = p.get("sharpen_size", 1.0)
        blurred = Evaluator._blur(image, {"radius": size})
        detail = image - blurred
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = image[..., c] + detail[..., c] * amount
        return out

    @staticmethod
    def _glow(image, p):
        # Nuke's Glow2: blur only what's above a threshold, tint and scale it, and add it back
        # over the input. A black image has nothing above a positive threshold, so it stays black.
        threshold = p.get("glow_threshold", 1.0)
        size = p.get("glow_size", 8.0)
        bright = np.clip(image[..., :3] - threshold, 0.0, None)
        blurred = Evaluator._blur(np.concatenate([bright, image[..., 3:4]], axis=2),
                                  {"radius": size})[..., :3]
        brightness = p.get("brightness", 1.0)
        tint = np.array([p.get("red", 1.0), p.get("green", 1.0), p.get("blue", 1.0)], dtype=np.float32)
        glow_rgb = (blurred * brightness * tint).astype(np.float32)
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            if c < 3:
                out[..., c] = image[..., c] + glow_rgb[..., c]
        return out

    @staticmethod
    def _exposure(image, p):
        # Nuke's Exposure: out = (in - blackpoint) * gain, per channel, so a pixel sitting at the
        # black point becomes exactly 0 and everything else scales about it. `stops` gain is
        # 2 ** exposure (the same stop math as Grade.exposure); `densities` gain is
        # 10 ** (density / 0.6), the reference guide's "log10(density) of 0.6 gamma negative
        # stock". With `gang` on the red slider drives all three channels. Alpha (channels =
        # rgba/alpha) takes the red channel's gain.
        base = 2.0 if p.get("exposure_mode", "stops") == "stops" else 10.0
        scale = 1.0 if base == 2.0 else 1.0 / 0.6
        red = float(p.get("red", 0.0))
        if p.get("gang", 1):
            amounts = (red, red, red)
        else:
            amounts = (red, float(p.get("green", 0.0)), float(p.get("blue", 0.0)))
        gains = [np.float32(base ** (a * scale)) for a in amounts]
        gains.append(gains[0])
        black = np.float32(p.get("blackpoint", 0.0))
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = (image[..., c] - black) * gains[c]
        return out

    _HUE_BANDS = ("red", "yellow", "green", "cyan", "blue", "magenta")

    @staticmethod
    def _hue_correct(image, p):
        # HueCorrect: per-hue saturation and luminance multipliers. Six anchors sit on the hue
        # circle at 0, 60 .. 300 degrees; between two neighbours the multiplier follows a
        # smoothstep (3t^2 - 2t^3), so it is continuous, flat at every anchor and wraps from
        # magenta back to red. Saturation scales the chroma about luma (as Saturation does); the
        # luminance multiplier is faded in by the pixel's HSV saturation so neutrals, whose hue is
        # undefined, are never darkened or brightened by whichever band hue 0 happens to be.
        # `hue_shift` finally rotates the chroma about the neutral axis (channel average kept).
        # Alpha is untouched.
        rgb = image[..., :3]
        hue, sat = Evaluator._rgb_to_hue_sat(rgb)
        pos = hue / 60.0
        base = np.floor(pos)
        t = (pos - base).astype(np.float32)
        t = t * t * (3.0 - 2.0 * t)
        i0 = base.astype(np.int64) % 6
        i1 = (i0 + 1) % 6

        def band(prefix):
            v = np.array([float(p.get(f"{prefix}_{b}", 1.0)) for b in Evaluator._HUE_BANDS], np.float32)
            return v[i0] * (1.0 - t) + v[i1] * t

        s_mult, l_mult = band("sat"), band("lum")
        luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
        out = rgb + (s_mult - 1.0) * (rgb - luma)
        out = out * (1.0 + (l_mult - 1.0) * np.clip(sat, 0.0, 1.0))
        shift = float(p.get("hue_shift", 0.0))
        if shift % 360.0 != 0.0:
            a = math.radians(shift)
            c, s = math.cos(a), math.sin(a)
            k = 1.0 / math.sqrt(3.0)
            # Rodrigues rotation about (1, 1, 1)/sqrt(3), written out as a 3x3 matrix.
            m = np.array([[c + (1 - c) / 3, (1 - c) / 3 - s * k, (1 - c) / 3 + s * k],
                          [(1 - c) / 3 + s * k, c + (1 - c) / 3, (1 - c) / 3 - s * k],
                          [(1 - c) / 3 - s * k, (1 - c) / 3 + s * k, c + (1 - c) / 3]], np.float32)
            out = out @ m.T
        return np.concatenate([out, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _color_matrix(image, p):
        # ColorMatrix: out = M @ (r, g, b) with M read row by row from matrix_RC. With `invert` on
        # the inverse is applied instead; a singular matrix (|det| < 1e-9) has no inverse, so the
        # node then passes the input through unchanged rather than emitting NaNs. Alpha is untouched.
        m = np.array([[float(p.get(f"matrix_{i}{j}", 1.0 if i == j else 0.0)) for j in range(3)]
                      for i in range(3)], np.float64)
        if p.get("invert"):
            if abs(np.linalg.det(m)) < 1e-9:
                return image.copy()
            m = np.linalg.inv(m)
        m = m.astype(np.float32)
        rgb = image[..., :3]
        out = np.stack([m[i, 0] * rgb[..., 0] + m[i, 1] * rgb[..., 1] + m[i, 2] * rgb[..., 2]
                        for i in range(3)], axis=2)
        return np.concatenate([out, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _gaussian_axis(frame, radius, sigma, axis, mode="edge"):
        """Separable Gaussian pass with a truncated, renormalised kernel of half-width `radius`
        (weights sum to exactly 1, so flat regions and total energy are preserved) and "edge"
        padding like `_box_blur_axis`."""
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()
        n = frame.shape[axis]
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(frame, pad, mode=mode).astype(np.float64)
        out = np.zeros(frame.shape, dtype=np.float64)
        for i, weight in enumerate(weights):
            index = [slice(None)] * frame.ndim
            index[axis] = slice(i, i + n)
            out += weight * padded[tuple(index)]
        return out.astype(np.float32)

    @staticmethod
    def _soften(image, p):
        # Nuke's Soften: a Gaussian-leaning blur. `soften_size` is the kernel's pixel reach:
        # sigma = size / 3, truncated at ceil(size) pixels (three sigma), which is exactly the
        # padded support `tiers._support_rule` declares, so tiles have every neighbour they read.
        # Below the same 0.5 cut-off the other padded filters use, it is the identity.
        size = abs(float(p.get("soften_size", 4.0)))
        if size < 0.5:
            return image.copy()
        radius = int(math.ceil(size))
        sigma = size / 3.0
        soft = Evaluator._gaussian_axis(Evaluator._gaussian_axis(image, radius, sigma, axis=1),
                                        radius, sigma, axis=0)
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            out[..., c] = soft[..., c]
        return out

    @staticmethod
    def _disc_radii(p):
        """Semi-axes (rx, ry) of Defocus's disc: `defocus` wide, `aspect` = width / height."""
        radius = abs(float(p.get("defocus", 6.0)))
        aspect = float(p.get("aspect", 1.0))
        return radius, radius / aspect if aspect > 0 else radius

    @staticmethod
    def _defocus(image, p):
        # Nuke's Defocus without depth: every output pixel is the plain average of the input pixels
        # whose centres fall inside an ellipse of semi-axes (defocus, defocus / aspect), so a
        # bright point becomes a flat-topped disc (unlike Blur's box and Soften's Gaussian) and
        # the weights sum to exactly 1. The disc is summed one row at a time: row dy contributes a
        # horizontal box of half-width floor(rx * sqrt(1 - (dy / ry) ** 2)), built from a running
        # sum, so cost grows with the radius, not its square. "edge" padding, like Blur. Below a
        # half pixel it is the identity, the cut-off the other padded filters use.
        rx, ry = Evaluator._disc_radii(p)
        if max(rx, ry) < 0.5:
            return image.copy()
        rows = int(math.floor(ry + 1e-9))
        half_widths = {}
        for dy in range(-rows, rows + 1):
            inside = max(0.0, 1.0 - (dy / ry) ** 2)
            half_widths[dy] = int(math.floor(rx * math.sqrt(inside) + 1e-9))
        reach = max(half_widths.values())
        total = float(sum(2 * w + 1 for w in half_widths.values()))
        height, width = image.shape[:2]
        padded = np.pad(image, ((rows, rows), (reach, reach), (0, 0)), mode="edge").astype(np.float64)
        running = np.concatenate([np.zeros((padded.shape[0], 1, 4)), np.cumsum(padded, axis=1)], axis=1)
        out = np.zeros(image.shape, dtype=np.float64)
        for dy, w in half_widths.items():
            band = running[rows + dy:rows + dy + height]
            out += band[:, reach + w + 1:reach + w + 1 + width] - band[:, reach - w:reach - w + width]
        out /= total
        result = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            result[..., c] = out[..., c].astype(np.float32)
        return result

    @staticmethod
    def _shadow_offset(p):
        """Whole-pixel (dx, dy) of DropShadow's offset, in array coordinates (y down)."""
        angle = math.radians(float(p.get("angle", -45.0)))
        distance = abs(float(p.get("distance", 0.0)))
        return int(round(distance * math.cos(angle))), int(round(-distance * math.sin(angle)))

    @staticmethod
    def _drop_shadow(image, p):
        # Nuke's DropShadow: the input's alpha is moved by (distance, angle), rounded to whole
        # pixels so the shadow lands exactly where the offset says, blurred by `shadow_size` (the
        # Soften kernel: sigma = size / 3, truncated at ceil(size) pixels), scaled by `opacity`,
        # tinted, and composited UNDER the input: out = input + shadow * (1 - input alpha) on
        # premultiplied RGBA, so wherever the input is opaque it is untouched. Outside the frame
        # counts as transparent (zero fill, not edge extension: a shadow must not be invented from
        # the border pixel). The result stays inside the input's window, like Blur.
        height, width = image.shape[:2]
        dx, dy = Evaluator._shadow_offset(p)
        shifted = np.zeros((height, width, 1), dtype=np.float32)
        src_x, src_y = slice(max(0, -dx), min(width, width - dx)), slice(max(0, -dy), min(height, height - dy))
        dst_x, dst_y = slice(max(0, dx), min(width, width + dx)), slice(max(0, dy), min(height, height + dy))
        if dst_x.stop > dst_x.start and dst_y.stop > dst_y.start:
            shifted[dst_y, dst_x, 0] = image[src_y, src_x, 3]
        size = abs(float(p.get("shadow_size", 0.0)))
        if size >= 0.5:
            radius, sigma = int(math.ceil(size)), size / 3.0
            shifted = Evaluator._gaussian_axis(Evaluator._gaussian_axis(shifted, radius, sigma, axis=1, mode="constant"),
                                               radius, sigma, axis=0, mode="constant")
        shadow = shifted * np.float32(p.get("opacity", 0.5))
        tint = np.array([p.get("red", 0.0), p.get("green", 0.0), p.get("blue", 0.0), 1.0], dtype=np.float32)
        return (image + shadow * tint * (1.0 - image[..., 3:4])).astype(np.float32)

    @staticmethod
    def _bilinear_gather(image, qx, qy):
        """Bilinear samples of `image` at continuous positions (`qx`, `qy`), where pixel (i, j)
        has its centre at (i + 0.5, j + 0.5). Positions past the frame clamp to the edge pixel,
        the "edge" padding the other padded filters use."""
        height, width = image.shape[:2]
        u, v = qx - 0.5, qy - 0.5
        x0, y0 = np.floor(u), np.floor(v)
        fx, fy = (u - x0)[..., None], (v - y0)[..., None]
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        x1, y1 = np.clip(x0 + 1, 0, width - 1), np.clip(y0 + 1, 0, height - 1)
        x0, y0 = np.clip(x0, 0, width - 1), np.clip(y0, 0, height - 1)
        return ((image[y0, x0] * (1 - fx) + image[y0, x1] * fx) * (1 - fy)
                + (image[y1, x0] * (1 - fx) + image[y1, x1] * fx) * fy)

    @staticmethod
    def _dirblur(image, p):
        # Nuke's DirBlur. All three types average bilinear samples taken along a path through each
        # pixel; the weights sum to 1 so flat regions and total energy are preserved.
        #   linear: a box of `length` pixels along `angle` (degrees, counter-clockwise from +x with
        #           y up, as in Nuke). Taps sit on whole-pixel steps along the line, the two end
        #           taps weighted by how much of their pixel the box covers, so length 8 is exactly
        #           eight pixels of coverage and length 1 or less is the identity. Symmetric about
        #           the pixel, and the reach is ceil(length / 2) + 1 pixels (the tile region rule).
        #   zoom:   samples on the ray through the centre, scale 1 +/- length / 200.
        #   radial: samples on the circle about the centre, +/- angle / 2 degrees.
        # The centre is in this array's own pixel coordinates (see `_filtered_pixels`).
        kind = p.get("blur_type", "linear")
        height, width = image.shape[:2]
        if kind == "linear":
            length = abs(float(p.get("length", 0.0)))
            if length <= 1.0:
                return image.copy()
            half = length / 2.0
            reach = int(math.floor(half + 0.5))
            angle = math.radians(float(p.get("angle", 0.0)))
            dx, dy = math.cos(angle), -math.sin(angle)
            pad = reach + 1
            padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float64)
            acc = np.zeros(image.shape, dtype=np.float64)
            total = 0.0
            for t in range(-reach, reach + 1):
                weight = min(t + 0.5, half) - max(t - 0.5, -half)
                if weight <= 0.0:
                    continue
                ox, oy = round(t * dx, 9), round(t * dy, 9)
                ix, iy = int(math.floor(ox)), int(math.floor(oy))
                fx, fy = ox - ix, oy - iy
                for (sx, sy, w) in ((ix, iy, (1 - fx) * (1 - fy)), (ix + 1, iy, fx * (1 - fy)),
                                    (ix, iy + 1, (1 - fx) * fy), (ix + 1, iy + 1, fx * fy)):
                    if w:
                        acc += (weight * w) * padded[pad + sy:pad + sy + height, pad + sx:pad + sx + width]
                total += weight
            acc /= total
        elif kind in ("zoom", "radial"):
            cx, cy = float(p.get("center_x", 0.0)), float(p.get("center_y", 0.0))
            gy, gx = np.mgrid[0:height, 0:width]
            rx, ry = gx + 0.5 - cx, gy + 0.5 - cy
            far = math.hypot(max(abs(0.0 - cx), abs(width - cx)), max(abs(0.0 - cy), abs(height - cy)))
            if kind == "zoom":
                span = abs(float(p.get("length", 0.0))) / 100.0
                if span <= 0.0:
                    return image.copy()
                count = int(min(96, max(2, math.ceil(far * span) + 1)))
                steps = [1.0 + s for s in np.linspace(-span / 2.0, span / 2.0, count)]
                positions = [(cx + rx * s, cy + ry * s) for s in steps]
            else:
                sweep = math.radians(abs(float(p.get("angle", 0.0))))
                if sweep <= 0.0:
                    return image.copy()
                count = int(min(96, max(2, math.ceil(far * sweep) + 1)))
                positions = []
                for a in np.linspace(-sweep / 2.0, sweep / 2.0, count):
                    c, s = math.cos(a), math.sin(a)
                    positions.append((cx + rx * c - ry * s, cy + rx * s + ry * c))
            acc = np.zeros(image.shape, dtype=np.float64)
            for qx, qy in positions:
                acc += Evaluator._bilinear_gather(image.astype(np.float64), qx, qy)
            acc /= len(positions)
        else:
            raise ValueError(f"Unknown DirBlur blur_type {kind!r}")
        result = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            result[..., c] = acc[..., c].astype(np.float32)
        return result

    @staticmethod
    def _mirror(image, p):
        # Flip about the format centre: a pixel at x moves to width-1-x (flip_x) and/or
        # y moves to height-1-y (flip_y). No mask/mix-independent box growth -- it is a pure
        # in-place reindex, like Invert, just spatial instead of per-channel.
        out = image
        if p.get("flip_x", 0):
            out = out[:, ::-1, :]
        if p.get("flip_y", 0):
            out = out[::-1, :, :]
        return np.ascontiguousarray(out, dtype=np.float32)

    @staticmethod
    def _dissolve(a, b, p):
        # which=0 -> A, which=1 -> B, values between cross-fade linearly.
        which = np.float32(p.get("which", 0.0))
        return a * (1 - which) + b * which

    @staticmethod
    def _copy_channels(a, b, p):
        out = b.copy()
        index = {"r": 0, "g": 1, "b": 2, "a": 3}
        for out_channel, param in (("r", "copy_red"), ("g", "copy_green"),
                                   ("b", "copy_blue"), ("a", "copy_alpha")):
            source = p.get(param, "none")
            if source == "none":
                continue
            _, src_channel = source.split(".")
            out[..., index[out_channel]] = a[..., index[src_channel]]
        return out

    @staticmethod
    def _difference_key(a, b, p):
        """Nuke's Difference keyer: alpha from the largest per-channel colour difference between A
        and B (max, not mean -- a change in any one channel should be enough to key, the same
        reasoning `_merge_op`'s own `difference` uses per-channel rather than collapsing to one
        number first), `offset` and `gain` shaping it, output is B's colour with the new alpha."""
        diff = np.max(np.abs(a[..., :3] - b[..., :3]), axis=-1, keepdims=True)
        alpha = np.clip((diff - p.get("offset", 0.0)) * p.get("gain", 1.0), 0.0, 1.0)
        return np.concatenate([b[..., :3], alpha], axis=2).astype(np.float32)

    @staticmethod
    def _channel_merge(a, b, p):
        index = {"r": 0, "g": 1, "b": 2, "a": 3}
        a_idx = index[p.get("a_channel", "A.a").split(".")[1]]
        b_idx = index[p.get("b_channel", "B.a").split(".")[1]]
        out_idx = {"R": 0, "G": 1, "B": 2, "A": 3}[p.get("out_channel", "A")]
        combined = Evaluator._channel_merge_op(p.get("operation", "over"),
                                               a[..., a_idx:a_idx + 1], b[..., b_idx:b_idx + 1])
        out = b.copy()
        out[..., out_idx:out_idx + 1] = combined
        return out

    @staticmethod
    def _apply_mask_mix(source, filtered, mask, mix):
        """Reusable Nuke-style mask + mix on image-filter nodes.

        Math (premultiplied RGBA, matches Nuke's "mask" + "mix" knobs):
            gate = mix * mask.a   (scalar when mask is unwired; per-pixel when wired)
            result = gate * filtered + (1 - gate) * source

        Properties:
          * mix=0                       -> result == source (filter is fully bypassed).
          * mask=None (full opacity)    -> result == mix*filtered + (1-mix)*source.
          * mask.a=0 (everywhere)       -> result == source (mask hides the filter).
          * Mask shape mismatch with source raises — no silent resampling.
        HDR-safe: never produces NaN/inf for finite inputs (mix and mask.a are in [0, 1]).
        """
        if mask is None:
            gate = np.float32(mix)
        else:
            if mask.shape != source.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match source {source.shape}; "
                    "no silent resampling is performed")
            gate = (mask[..., 3:4] * np.float32(mix)).astype(np.float32)
        return (filtered * gate + source * (1.0 - gate)).astype(np.float32)

    # --- Draw-menu generators (Ramp, Radial, Rectangle, Noise, Text) -----------------------
    #
    # `_draw_shape` is the one function both `_windowed_kernel` (full frame, at (0, 0, width,
    # height)) and `tileexec._evaluate_tile_kernel` (one tile, at (region.x, region.y, region.width,
    # region.height)) call, with no other path to a pixel: a tile seam can only be identical to the
    # full-frame reference if it runs the *same* code, not a second implementation kept in sync by
    # hand. Every shape returns straight premultiplied RGBA covering exactly (x0, y0, w, h) in
    # absolute canvas coordinates -- column x0 is pixel index x0, not a half-pixel-centred sample --
    # matching Checker's own `xx // size` integer-index convention.

    @staticmethod
    def _draw_shape(kind, p, x0, y0, w, h, frame):
        if kind == "Ramp":
            return Evaluator._ramp_shape(p, x0, y0, w, h)
        if kind == "Radial":
            return Evaluator._radial_shape(p, x0, y0, w, h)
        if kind == "Rectangle":
            return Evaluator._rectangle_shape(p, x0, y0, w, h)
        if kind == "Noise":
            return Evaluator._noise_shape(p, x0, y0, w, h)
        if kind == "Text":
            return Evaluator._text_shape(p, x0, y0, w, h)
        if kind == "Grid":
            return Evaluator._grid_shape(p, x0, y0, w, h)
        raise ValueError(f"No draw shape for {kind}")

    @staticmethod
    def _composite_shape_over(shape, background):
        """Premultiplied "over": the shape is drawn over its optional background, Nuke's own
        Draw-node convention. `shape`'s alpha is where the shape itself is opaque; the background
        shows through everywhere the shape is not."""
        shape_alpha = shape[..., 3:4]
        return (shape + background * (1.0 - shape_alpha)).astype(np.float32)

    @staticmethod
    def _ramp_shape(p, x0, y0, w, h):
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        p0x, p0y = float(p["p0_x"]), float(p["p0_y"])
        p1x, p1y = float(p["p1_x"]), float(p["p1_y"])
        dx, dy = p1x - p0x, p1y - p0y
        denom = dx * dx + dy * dy
        # p0 == p1 is a degenerate ramp with no direction; Nuke holds colour0 everywhere rather
        # than dividing by zero.
        t = np.zeros_like(xx) if denom < 1e-12 else ((xx - p0x) * dx + (yy - p0y) * dy) / denom
        t = np.clip(t, 0.0, 1.0)[..., None]
        c0 = np.array([p["color0_red"], p["color0_green"], p["color0_blue"], p["color0_alpha"]])
        c1 = np.array([p["color1_red"], p["color1_green"], p["color1_blue"], p["color1_alpha"]])
        straight = c0[None, None, :] * (1.0 - t) + c1[None, None, :] * t
        alpha = straight[..., 3:4]
        return np.concatenate([straight[..., :3] * alpha, alpha], axis=-1).astype(np.float32)

    @staticmethod
    def _box_edge_distance(p, x0, y0, w, h):
        """Signed distance (pixels) from the nearest edge of `box_*`: positive inside, negative
        outside. The box's valid columns/rows are `[box_x, box_x + box_width)`, the same half-open
        convention `Region` uses everywhere else in this codebase."""
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        bx, by = float(p["box_x"]), float(p["box_y"])
        bw, bh = float(p["box_width"]), float(p["box_height"])
        return np.minimum(np.minimum(xx - bx, bx + bw - 1.0 - xx),
                          np.minimum(yy - by, by + bh - 1.0 - yy)), bw, bh

    @staticmethod
    def _paint_coverage(p, coverage):
        """A flat colour (`red`/`green`/`blue`/`alpha`) modulated by a 0..1 coverage mask,
        premultiplied. Shared by Radial and Rectangle, which differ only in how coverage is shaped."""
        color = np.array([p["red"], p["green"], p["blue"], p["alpha"]])
        alpha = (coverage * color[3]).astype(np.float32)
        return np.concatenate([alpha[..., None] * color[:3], alpha[..., None]], axis=-1).astype(np.float32)

    @staticmethod
    def _rectangle_shape(p, x0, y0, w, h):
        dist, bw, bh = Evaluator._box_edge_distance(p, x0, y0, w, h)
        softness = float(p.get("softness", 0.0)) * min(bw, bh) / 2.0
        coverage = (dist >= 0).astype(np.float64) if softness < 1e-9 else np.clip(dist / softness, 0.0, 1.0)
        return Evaluator._paint_coverage(p, coverage)

    @staticmethod
    def _grid_lines(index, extent, spacing, number, offset, line_width):
        """Line coverage (0..1) along one axis for absolute pixel indices `index`. A line starts at
        `offset + k * step` for every integer k and covers `line_width` pixels from there, so with
        whole-pixel step, offset and width the covered indices are exactly those where
        `(index - offset) mod step < line_width`. `number` above zero replaces the spacing with
        `extent / number`, which puts `number` lines across the format."""
        step = float(extent) / float(number) if number > 0 else float(spacing)
        phase = np.mod(index - float(offset), step)
        return np.clip(float(line_width) - phase, 0.0, 1.0)

    @staticmethod
    def _grid_shape(p, x0, y0, w, h):
        cols = Evaluator._grid_lines(np.arange(x0, x0 + w, dtype=np.float64), p["width"], p["spacing_x"],
                                     p["number_x"], p["grid_offset_x"], p["line_width"])
        rows = Evaluator._grid_lines(np.arange(y0, y0 + h, dtype=np.float64), p["height"], p["spacing_y"],
                                     p["number_y"], p["grid_offset_y"], p["line_width"])
        coverage = np.maximum(rows[:, None], cols[None, :])
        return Evaluator._paint_coverage(p, coverage)

    @staticmethod
    def _radial_shape(p, x0, y0, w, h):
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        bx, by = float(p["box_x"]), float(p["box_y"])
        bw, bh = float(p["box_width"]), float(p["box_height"])
        cx, cy = bx + (bw - 1.0) / 2.0, by + (bh - 1.0) / 2.0
        rx, ry = max(bw / 2.0, 1e-9), max(bh / 2.0, 1e-9)
        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        softness = float(p.get("softness", 0.0))
        coverage = (d < 1.0).astype(np.float64) if softness < 1e-9 else np.clip((1.0 - d) / softness, 0.0, 1.0)
        return Evaluator._paint_coverage(p, coverage)

    @staticmethod
    def _hash_lattice(ix, iy, iz, seed):
        """A deterministic pseudo-random value in [0, 1) per integer lattice point. Pure integer
        arithmetic (no `random`/`np.random`, whose sequences are seed-state, not coordinate,
        driven): the same (ix, iy, iz, seed) always hashes to the same value, on any machine,
        whether it is asked for as a whole canvas or one tile at a time."""
        h = (ix.astype(np.int64) * np.int64(374761393) + iy.astype(np.int64) * np.int64(668265263)
            + np.int64(int(iz)) * np.int64(2147483647) + np.int64(int(seed)) * np.int64(2654435761))
        h = h & np.int64(0xffffffff)
        h = ((h ^ (h >> 13)) * np.int64(1274126177)) & np.int64(0xffffffff)
        h = h ^ (h >> 16)
        return h.astype(np.float64) / 4294967295.0

    @staticmethod
    def _value_noise(gx, gy, gz, seed):
        ix0, iy0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        fx, fy = gx - ix0, gy - iy0
        iz = math.floor(gz)
        # Quintic smoothstep (Perlin's improved curve): zero first and second derivative at 0/1,
        # so octave boundaries never show a slope discontinuity.
        sx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        sy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        v00 = Evaluator._hash_lattice(ix0, iy0, iz, seed)
        v10 = Evaluator._hash_lattice(ix0 + 1, iy0, iz, seed)
        v01 = Evaluator._hash_lattice(ix0, iy0 + 1, iz, seed)
        v11 = Evaluator._hash_lattice(ix0 + 1, iy0 + 1, iz, seed)
        top = v00 + sx * (v10 - v00)
        bottom = v01 + sx * (v11 - v01)
        return top + sy * (bottom - top)

    @staticmethod
    def _noise_shape(p, x0, y0, w, h):
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        size = max(float(p.get("size", 64.0)), 1e-6)
        z = float(p.get("z_slice", 0.0))
        seed = int(p.get("seed", 0))
        octaves = max(1, int(p.get("octaves", 1)))
        lacunarity = float(p.get("lacunarity", 2.0))
        gain = float(p.get("gain", 0.5))
        gamma = max(float(p.get("gamma", 1.0)), 1e-6)
        total = np.zeros_like(xx)
        amplitude, freq, amp_sum = 1.0, 1.0 / size, 0.0
        for octave in range(octaves):
            total = total + amplitude * Evaluator._value_noise(xx * freq, yy * freq, z, seed + octave * 101)
            amp_sum += amplitude
            amplitude *= gain
            freq *= lacunarity
        value = np.clip(total / max(amp_sum, 1e-9), 0.0, 1.0) ** (1.0 / gamma)
        value = value.astype(np.float32)
        alpha = np.ones_like(value)
        return np.concatenate([np.repeat(value[..., None], 3, axis=-1), alpha[..., None]], axis=-1).astype(np.float32)

    @staticmethod
    def _text_shape(p, x0, y0, w, h):
        """Qt's own text rasteriser (QPainter/QFont on an offscreen QImage), no new dependency.

        Font-availability limit: `font` is a family name resolved through Qt's font database at
        render time, exactly like Nuke's own font picker resolves a family on the machine running
        Nuke. Which families are installed is a machine property NodeBased does not control or
        embed -- a comp that names a font missing on another artist's machine renders in whatever
        Qt substitutes there, the same portability limit Nuke's text tools have always had.
        """
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor, QFont, QImage, QPainter
        from PySide6.QtWidgets import QApplication

        if QApplication.instance() is None:
            QApplication(["nodebased"])
        w, h = max(1, int(w)), max(1, int(h))
        image = QImage(w, h, QImage.Format.Format_RGBA8888)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # Paint in absolute canvas coordinates; the translate is what makes a tile's render of a
        # box that straddles its edge identical to the full-frame render cropped to that tile.
        painter.translate(-x0, -y0)
        family = p.get("font") or ""
        font = QFont(family) if family else QFont()
        font.setPointSizeF(max(1.0, float(p.get("font_size", 48.0))))
        painter.setFont(font)
        color = QColor.fromRgbF(float(np.clip(p.get("red", 1.0), 0.0, 1.0)),
                                float(np.clip(p.get("green", 1.0), 0.0, 1.0)),
                                float(np.clip(p.get("blue", 1.0), 0.0, 1.0)),
                                float(np.clip(p.get("alpha", 1.0), 0.0, 1.0)))
        painter.setPen(color)
        box = QRectF(float(p["box_x"]), float(p["box_y"]), float(p["box_width"]), float(p["box_height"]))
        justify_flags = {"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
                         "right": Qt.AlignmentFlag.AlignRight}
        align = justify_flags.get(p.get("justify", "left"), Qt.AlignmentFlag.AlignLeft)
        painter.drawText(box, int(align | Qt.AlignmentFlag.AlignVCenter) | int(Qt.TextFlag.TextWordWrap),
                         str(p.get("message", "")))
        painter.end()
        stride = image.bytesPerLine()
        raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint8).reshape(h, stride)
        straight = raw[:, :w * 4].reshape(h, w, 4).astype(np.float32) / 255.0
        alpha = straight[..., 3:4]
        return np.concatenate([straight[..., :3] * alpha, alpha], axis=-1).astype(np.float32)

    @staticmethod
    def _grade(image, p):
        frame = image.copy()
        frame[..., :3] = frame[..., :3] * (2.0 ** p["exposure"]) * p["multiply"] + p["offset"] * frame[..., 3:4]
        return frame

    @staticmethod
    def _color_correct(image, p):
        frame = image
        alpha = frame[..., 3:4]
        straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
        corrected = straight * p["gain"] + p["lift"] * (1 - straight)
        # sign * |x|^(1/gamma) avoids raising a negative base to a fractional power.
        powered = np.sign(corrected) * np.abs(corrected) ** (1.0 / p["gamma"])
        luma = 0.2126 * powered[..., 0:1] + 0.7152 * powered[..., 1:2] + 0.0722 * powered[..., 2:3]
        saturated = luma + (powered - luma) * p["saturation"]
        return np.concatenate([saturated * alpha, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _blur(image, p):
        radius = p["radius"]
        if radius < 0.5:
            return image.copy()
        return Evaluator._box_blur_axis(Evaluator._box_blur_axis(image, radius, axis=1), radius, axis=0)

    @staticmethod
    def _crop(source, p):
        # Masks to a rectangle without resizing the canvas, matching Transform's fixed-bounds format.
        frame = np.zeros_like(source)
        h, w = frame.shape[:2]
        left, top = max(p["x"], 0), max(p["y"], 0)
        right, bottom = min(w, p["x"] + p["width"]), min(h, p["y"] + p["height"])
        if left < right and top < bottom:
            frame[top:bottom, left:right] = source[top:bottom, left:right]
        return frame

    @staticmethod
    def _reformat_place(target_w, target_h, src_w, src_h, resize_type, center, turn):
        """Scale and offset placing a `src_w`x`src_h` source into a `target_w`x`target_h` format,
        before flip/flop/turn are applied. Shared by `_reformat` (the kernel) and
        `_reformat_transformed_window` (the preserve-bounding-box case) so the two formulas
        cannot drift apart, the same reason `_transform`/`_transformed_window` share their math.

        `turn` solves the resize against the *swapped* working format -- Nuke's own turn knob
        also treats width and height as exchanged before deciding the resize -- so a turned
        Reformat still fits/fills correctly instead of using the unturned format's aspect.
        """
        tw, th = (target_h, target_w) if turn else (target_w, target_h)
        sx0 = tw / src_w if src_w else 1.0
        sy0 = th / src_h if src_h else 1.0
        if resize_type == "none":
            scale_x = scale_y = 1.0
        elif resize_type == "width":
            scale_x = scale_y = sx0
        elif resize_type == "height":
            scale_x = scale_y = sy0
        elif resize_type == "fit":
            scale_x = scale_y = min(sx0, sy0)
        elif resize_type == "fill":
            scale_x = scale_y = max(sx0, sy0)
        elif resize_type == "distort":
            scale_x, scale_y = sx0, sy0
        else:
            raise ValueError(f"Unknown Reformat resize type: {resize_type}")
        if center:
            offset_x = (tw - src_w * scale_x) / 2.0
            offset_y = (th - src_h * scale_y) / 2.0
        else:
            offset_x = offset_y = 0.0
        return scale_x, scale_y, offset_x, offset_y, tw, th

    @staticmethod
    def _reformat_transformed_window(box, src_display_w, src_display_h, target_w, target_h, p):
        """Forward-map the corners of `box` through the same placement `_reformat` inverts, for
        the preserve-bounding-box case. Exact inverse of `_reformat`'s sampling map."""
        if box.is_empty:
            return box
        resize_type, center = p["resize_type"], p["center"]
        flip, flop, turn = p["flip"], p["flop"], p["turn"]
        scale_x, scale_y, offset_x, offset_y, _tw, _th = Evaluator._reformat_place(
            target_w, target_h, src_display_w, src_display_h, resize_type, center, turn)
        xs, ys = [], []
        for bx, by in ((box.x, box.y), (box.right, box.y), (box.x, box.bottom), (box.right, box.bottom)):
            px, py = bx * scale_x + offset_x, by * scale_y + offset_y
            fx, fy = (py, px) if turn else (px, py)
            if flop:
                fx = target_w - fx
            if flip:
                fy = target_h - fy
            xs.append(fx)
            ys.append(fy)
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(p.get("filter", "nearest"), 2)
        left, top = math.floor(min(xs)) - support, math.floor(min(ys)) - support
        right, bottom = math.ceil(max(xs)) + support, math.ceil(max(ys)) + support
        return Region(int(left), int(top), int(right - left), int(bottom - top))

    @staticmethod
    def _reformat(src, src_display_w, src_display_h, target_w, target_h, p, src_box=None, dst_box=None):
        """Inverse-mapped resize/reposition into a `target_w`x`target_h` format, with sub-pixel
        filtering, mirroring `_transform`'s own structure (see its docstring). Placement is
        solved against `src_display_w`/`src_display_h` -- the input's own declared format -- not
        against `src`'s array extent, which may carry overscan (`src_box`, aligning the array to
        the source's real data window, exactly as `_transform` uses it)."""
        h, w = src.shape[:2]
        if src_box is None:
            src_box = Region(0, 0, w, h)
        if dst_box is None:
            dst_box = Region(0, 0, target_w, target_h)
        resize_type, center = p["resize_type"], p["center"]
        flip, flop, turn = p["flip"], p["flop"], p["turn"]
        scale_x, scale_y, offset_x, offset_y, _tw, _th = Evaluator._reformat_place(
            target_w, target_h, src_display_w, src_display_h, resize_type, center, turn)
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float32) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float32) + dst_box.y + 0.5)
        # Undo flip/flop/turn, in reverse of the order `_reformat_transformed_window` applies them,
        # then undo the placement scale/offset to land on a source sample coordinate.
        y_a = (target_h - gy) if flip else gy
        x_b = (target_w - gx) if flop else gx
        px, py = (y_a, x_b) if turn else (x_b, y_a)
        sx = (px - offset_x) / scale_x if scale_x else px
        sy = (py - offset_y) / scale_y if scale_y else py
        sx_frac = sx - 0.5 - src_box.x
        sy_frac = sy - 0.5 - src_box.y
        return Evaluator._resample(src, sx_frac, sy_frac, p["filter"]).astype(np.float32)

    @staticmethod
    def _cornerpin(src, p, src_box=None, dst_box=None):
        """Inverse-mapped projective (four-point) warp, sharing `_transform`'s own inverse-map-
        then-`_resample` structure. `_cornerpin_forward_matrix` is the one place the homography
        is solved, so the kernel and `_cornerpin_window` cannot disagree about which quad is the
        pre-warp side."""
        h, w = src.shape[:2]
        if src_box is None:
            src_box = Region(0, 0, w, h)
        if dst_box is None:
            dst_box = src_box
        forward = Evaluator._cornerpin_forward_matrix(p)
        inverse = np.linalg.inv(forward)
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float64) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float64) + dst_box.y + 0.5)
        denom = inverse[2, 0] * gx + inverse[2, 1] * gy + inverse[2, 2]
        denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
        sx = (inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2]) / denom
        sy = (inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2]) / denom
        sx_frac = (sx - 0.5 - src_box.x).astype(np.float32)
        sy_frac = (sy - 0.5 - src_box.y).astype(np.float32)
        return Evaluator._resample(src, sx_frac, sy_frac, p["filter"]).astype(np.float32)

    @staticmethod
    def _transform(src, translate_x, translate_y, rotate, scale, center_x, center_y, filter,
                   src_box=None, dst_box=None):
        """Inverse-mapped 2D transform with sub-pixel filtering. Pixels outside the source return
        transparent black — there is no wrap, matching the v0.3.0 invariant.

        `src_box`/`dst_box` place the two arrays in image coordinates. Omitting them means "both
        arrays start at the origin", which is the pre-bounding-box behaviour and produces
        bit-identical output.
        """
        h, w = src.shape[:2]
        if src_box is None:
            src_box = Region(0, 0, w, h)
        if dst_box is None:
            dst_box = src_box
        # Identity shortcut: when every parameter is at its default and the filter is nearest,
        # copy the source. Keeps existing v0.3.0 pixel data bit-identical for upgrade-default nodes.
        if (translate_x == 0 and translate_y == 0 and (rotate % 360.0) == 0
                and scale == 1 and center_x == 0 and center_y == 0 and filter == "nearest"
                and dst_box == src_box):
            return src.copy()
        theta = math.radians(rotate)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        # Destination pixel centres (world space). Nuke convention: integer pixel indices span
        # [i, i+1) and the centre sits at i + 0.5, which makes translate=0 sample the source on its
        # own pixel centres for the identity transform.
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float32) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float32) + dst_box.y + 0.5)
        # Inverse affine: src = center + R(-θ) · (dst - center - translate) / scale
        ox = gx - center_x - translate_x
        oy = gy - center_y - translate_y
        sx = (ox * cos_t + oy * sin_t) * inv_scale + center_x
        sy = (-ox * sin_t + oy * cos_t) * inv_scale + center_y
        # Convert world sample coord to fractional pixel index inside the source array, which may
        # itself start away from the origin when the source carries overscan.
        sx_frac = sx - 0.5 - src_box.x
        sy_frac = sy - 0.5 - src_box.y
        return Evaluator._resample(src, sx_frac, sy_frac, filter).astype(np.float32)

    @staticmethod
    def _resample(src, sx_frac, sy_frac, filter):
        h, w = src.shape[:2]
        if filter == "nearest":
            xi = np.floor(sx_frac + 0.5).astype(np.int32)
            yi = np.floor(sy_frac + 0.5).astype(np.int32)
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xi_c = np.clip(xi, 0, w - 1)
            yi_c = np.clip(yi, 0, h - 1)
            sampled = src[yi_c, xi_c]
            return np.where(valid[..., None], sampled, np.float32(0.0)).astype(np.float32)
        if filter == "bilinear":
            x0 = np.floor(sx_frac).astype(np.int32)
            y0 = np.floor(sy_frac).astype(np.int32)
            fx = (sx_frac - x0).astype(np.float32)
            fy = (sy_frac - y0).astype(np.float32)
            def fetch(xi, yi):
                valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                xc = np.clip(xi, 0, w - 1)
                yc = np.clip(yi, 0, h - 1)
                return np.where(valid[..., None], src[yc, xc], np.float32(0.0))
            wx0, wx1 = (1 - fx)[..., None], fx[..., None]
            wy0, wy1 = (1 - fy)[..., None], fy[..., None]
            return (fetch(x0, y0) * wx0 * wy0 + fetch(x0 + 1, y0) * wx1 * wy0
                    + fetch(x0, y0 + 1) * wx0 * wy1 + fetch(x0 + 1, y0 + 1) * wx1 * wy1).astype(np.float32)
        if filter == "cubic":
            # Catmull-Rom (B=0, C=0.5): classic image-processing bicubic, sharper than Mitchell.
            a = -0.5
            x0 = np.floor(sx_frac).astype(np.int32)
            y0 = np.floor(sy_frac).astype(np.int32)
            fx = (sx_frac - x0).astype(np.float32)
            fy = (sy_frac - y0).astype(np.float32)
            def weight(t):
                at = np.abs(t)
                at2, at3 = at * at, at * at * at
                return np.where(at <= 1, (a + 2) * at3 - (a + 3) * at2 + 1,
                                a * at3 - 5 * a * at2 + 8 * a * at - 4 * a).astype(np.float32)
            wx = np.stack([weight(fx + 1), weight(fx), weight(fx - 1), weight(fx - 2)], axis=-1)
            wy = np.stack([weight(fy + 1), weight(fy), weight(fy - 1), weight(fy - 2)], axis=-1)
            def fetch(xi, yi):
                valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                xc = np.clip(xi, 0, w - 1)
                yc = np.clip(yi, 0, h - 1)
                return np.where(valid[..., None], src[yc, xc], np.float32(0.0))
            result = np.zeros(sx_frac.shape + (4,), dtype=np.float32)
            for dy in range(4):
                row = np.zeros(sx_frac.shape + (4,), dtype=np.float32)
                for dx in range(4):
                    row += fetch(x0 + dx - 1, y0 + dy - 1) * wx[..., dx:dx + 1]
                result += row * wy[..., dy:dy + 1]
            return result.astype(np.float32)
        raise ValueError(f"Unknown transform filter: {filter}")
