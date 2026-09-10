"""Demand-driven tile executor that complements `Evaluator`'s full-frame path.

Tile-native execution is a separate code path from `Evaluator.evaluate`. The two paths produce
the same pixel data at the same tier for the supported kernel subset (`tiles.SUPPORTED_TILED_KINDS`),
which is what makes the bounded correctness claim possible: every supported kernel has its
tile-native behaviour pinned against the full-frame reference by a golden test.

**What changed in this revision (reviewer-flagged).**

1. *Content-digest identity.* Every `TileKey` now carries a per-node content digest computed from
   (kind, scaled params, input digests, Read file fingerprint, tier, frame). Editing a Grade's
   exposure changes the digest, which changes the key, which makes the old tile a cache miss.
   The artist's edit auto-invalidates the cached pixels; the stale entry still sits in the
   cache until LRU evicts it.

2. *All ancestors traversed.* `supports_tiled` and `_canvas_size` no longer walk only the first
   wired input. They walk every evaluated ancestor, respecting Switch's selected branch and
   the disabled-flag passthrough, so a graph containing one Switch child is correctly reported
   as not-tiled.

3. *Decoded sources are retained.* Within a single `compose`, each generator is decoded once
   (or generated once) and sliced for every requesting tile. `_source_cache[(node, frame,
   tier)]` holds the full canvas. The decode-call counter on the executor makes the cost
   visible to tests and benchmarks; a missing lock-out is a regression.

4. *Transform and Crop are not in the supported set.* Their kernels are coordinate-dependent
   on the canvas origin, so reusing them on a tile buffer is silently wrong. They fall back to
   the legacy full-frame path; the tile executor never claims to have tiled them.

5. *Merge/mask inputs are aligned to the output region.* A Blur-tracked input and a Constant
   input both end up at the same `(buffered.x, buffered.y, buffered.width, buffered.height)`
   shape before the kernel sees them, so `a.shape != b.shape` cannot fire mid-render.
"""
from __future__ import annotations

from collections import OrderedDict
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from . import imaging
from . import tiers
from .core import SPECS
from .imaging import Evaluator
from .tiles import (DEFAULT_TILE_EDGE, SUPPORTED_TILED_KINDS, TileArtifact, TileCache, TileKey,
                    TileRegion, fits_in_budget, grid_for, iter_tiles, memory_budget_for_tiles,
                    resolve_halo)


def _to_tier_region(region: TileRegion) -> tiers.Region:
    """Project the tile region into the geometry the ROI rules in `tiers.py` speak."""
    return tiers.Region(int(region.x), int(region.y), int(region.width), int(region.height))


def _full_image(node, frame):
    """Decode (or generate) the full canvas for a generator. Counts as one source decode.

    `imaging.read_image` decodes file-based sources; this function passes the document's frame
    through. For Constant/Checker it builds the array from `node["params"]`. The result is
    float32 RGBA, read-only.
    """
    params = dict(node["params"])
    params["frame"] = frame
    if node["type"] == "Read":
        pixels = imaging.read_image(**params)
    elif node["type"] == "Constant":
        alpha = float(params["alpha"])
        color = np.array([float(params["red"]) * alpha, float(params["green"]) * alpha,
                          float(params["blue"]) * alpha, alpha], dtype=np.float32)
        pixels = np.broadcast_to(color, (int(params["height"]), int(params["width"]), 4)).copy()
    elif node["type"] == "Checker":
        h, w = int(params["height"]), int(params["width"])
        size = max(1, int(params["size"]))
        yy, xx = np.ogrid[:h, :w]
        pattern = ((xx // size + yy // size) % 2).astype(np.float32)
        pixels = np.ones((h, w, 4), np.float32)
        pixels[..., :3] = (0.06 + pattern * 0.24)[..., None]
    else:
        raise ValueError(f"{node['type']} is not a generator")
    pixels = np.asarray(pixels, dtype=np.float32)
    pixels.flags.writeable = False
    return pixels


def _node_content_digest(node, params, input_hashes, frame, tier, fingerprint=None):
    """Deterministic SHA-256 over the node's rendered content.

    Mirrors `imaging.Evaluator.evaluate`'s digest fields exactly so an edit that would change
    the legacy cache key also changes the tile cache key. `input_hashes` is the list of
    upstream tile digests (positional with the node's declared input slots); for generators
    it's empty.
    """
    payload = [node["type"], params, node["disabled"],
               [h for h in input_hashes if h is not None], fingerprint, int(tier),
               int(frame)]
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_node_digests(document, tier, frame):
    """Walk the graph topologically and produce a per-node content digest.

    Mirrors `Evaluator.evaluate`'s digest loop exactly. The result is a dict
    `{node_id: hex_digest}` that the tile executor passes into every cache key so an edit
    that changes the legacy digest also changes the tile cache key. Read nodes pick up the
    same file fingerprint as the legacy path.
    """
    nodes = document["nodes"]
    from .media import nearest_sequence_path, resolve_source_path

    order, seen = [], set()
    stack = [(node_id, False) for node_id in nodes]
    while stack:
        key, visited = stack.pop()
        if key in seen:
            continue
        if visited:
            seen.add(key)
            order.append(key)
            continue
        stack.append((key, True))
        node = nodes[key]
        if node["disabled"]:
            inputs = [v for v in node["inputs"].values() if v is not None][:1]
        else:
            inputs = list(node["inputs"].values())
        stack.extend((s, False) for s in inputs if s is not None)

    hashes = {}
    for key in order:
        node = nodes[key]
        kind = node["type"]
        params = tiers.scale_params(kind, node["params"], tier)
        # ALL wired input slots, required and optional (mask included) — must mirror
        # `Evaluator.evaluate`'s `list(node["inputs"].values())` exactly. The earlier draft used
        # only `SPECS[kind]["inputs"]`, which excludes optional slots such as "mask": rewiring or
        # editing a mask left the digest unchanged, so the tile cache kept serving pixels rendered
        # against the old mask. Reproduced directly: a Grade with mask A vs mask B produced
        # byte-identical tiled output before this fix.
        sources = list(node["inputs"].values())
        if node["disabled"]:
            sources = sources[:1]
        fingerprint = None
        if kind == "Read" and params["path"]:
            from .media import nearest_sequence_path as _nearest, resolve_source_path as _resolve
            source_frame = frame + int(params.get("frame_offset", 0))
            resolved, exists = _resolve(params["path"], source_frame, params.get("missing", "error"))
            if resolved is None:
                reference = _nearest(params["path"], source_frame)
                if reference is not None:
                    stat = Path(reference).stat()
                    fingerprint = ["<black>", source_frame, str(Path(reference).resolve()),
                                   stat.st_size, stat.st_mtime_ns]
            else:
                stat = Path(resolved).stat()
                fingerprint = [str(Path(resolved).resolve()), stat.st_size, stat.st_mtime_ns]
        hashes[key] = _node_content_digest(node, params, [hashes.get(s) for s in sources],
                                           frame, tier, fingerprint)
    return hashes


def _all_ancestors(document, target):
    """Yield every ancestor of `target` reachable through evaluated inputs.

    Disabled nodes pass through to their first input; Switch nodes only traverse the selected
    branch. Cycles are broken with a visited set so the caller cannot loop on a malformed graph
    (the document validator rejects cycles anyway, but defence in depth here keeps the executor
    honest about what would actually run).
    """
    nodes = document["nodes"]
    stack = [target]
    seen = set()
    while stack:
        key = stack.pop()
        if key in seen or key not in nodes:
            continue
        seen.add(key)
        node = nodes[key]
        yield key
        kind = node["type"]
        if node["disabled"] and kind not in ("Read", "Constant", "Checker"):
            # Disabled filter = passthrough of its first wired input. Disabled generator still
            # returns the same kind of source (the framework falls back to the first input).
            slots = list(SPECS[kind]["inputs"])
            for s in slots[:1]:
                src = node["inputs"].get(s)
                if src is not None:
                    stack.append(src)
            continue
        if kind == "Switch":
            which = int(node["params"].get("which", 0))
            slots = list(SPECS[kind]["inputs"])
            if 0 <= which < len(slots):
                src = node["inputs"].get(slots[which])
                if src is not None:
                    stack.append(src)
            continue
        slots = list(SPECS[kind]["inputs"])
        for s in slots:
            src = node["inputs"].get(s)
            if src is not None:
                stack.append(src)


class UnsupportedTile(Exception):
    """A kind has no tile-native implementation; fall back to `Evaluator.evaluate`."""


class CancelledTile(Exception):
    """A tile render was cancelled mid-flight (caller replaced the queue or hit the deadline)."""


@dataclass
class TileResult:
    """Result of a `TileExecutor.compose` call: full-frame pixels plus provenance."""

    pixels: np.ndarray
    canvas_width: int
    canvas_height: int
    tier: int
    tiled: bool
    tile_hits: int
    tile_misses: int
    full_frame_fallbacks: int
    source_decodes: int

    @property
    def shape(self) -> tuple:
        return tuple(self.pixels.shape[:2])


def _node_full_image_at_tier(node, frame, tier):
    """Full-frame generator output for `node` at `tier`, applying decimation for Read sources."""
    params = tiers.scale_params(node["type"], dict(node["params"]), int(tier))
    params["frame"] = frame
    full = _full_image_with_params(node, params, frame)
    if node["type"] == "Read" and int(tier) != 1:
        full = Evaluator._decimate(full, int(tier))
    return full


def _full_image_with_params(node, params, frame):
    """Build the full RGBA float32 array for a generator from already-scaled params."""
    if node["type"] == "Read":
        pixels = imaging.read_image(**params)
    elif node["type"] == "Constant":
        alpha = float(params["alpha"])
        color = np.array([float(params["red"]) * alpha, float(params["green"]) * alpha,
                          float(params["blue"]) * alpha, alpha], dtype=np.float32)
        pixels = np.broadcast_to(color, (int(params["height"]), int(params["width"]), 4)).copy()
    elif node["type"] == "Checker":
        h, w = int(params["height"]), int(params["width"])
        size = max(1, int(params["size"]))
        yy, xx = np.ogrid[:h, :w]
        pattern = ((xx // size + yy // size) % 2).astype(np.float32)
        pixels = np.ones((h, w, 4), np.float32)
        pixels[..., :3] = (0.06 + pattern * 0.24)[..., None]
    else:
        raise ValueError(f"{node['type']} is not a generator")
    pixels = np.asarray(pixels, dtype=np.float32)
    pixels.flags.writeable = False
    return pixels


class TileExecutor:
    """Tile-native evaluator that wraps the existing `Evaluator` for fallback."""

    def __init__(self, cache: TileCache | None = None, evaluator: Evaluator | None = None,
                 tile_edge: int = DEFAULT_TILE_EDGE):
        self.cache = cache or TileCache()
        self.evaluator = evaluator or Evaluator()
        self.tile_edge = int(tile_edge)
        self._lock = threading.Lock()
        # Per-compose source cache: (node_id, frame, tier) -> full float32 RGBA canvas. Held for
        # the duration of one compose so every tile render reuses the same decoded source instead
        # of re-decoding per tile (the bug the reviewer flagged).
        self._source_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        # Counters updated each compose; tests assert on these.
        self.stats = {
            "tile_renders": 0,
            "tile_hits": 0,
            "tile_misses": 0,
            "full_frame_fallbacks": 0,
            "unsupported_kinds": 0,
            "supported_kinds": 0,
            "source_decodes": 0,
        }

    # --- public API -------------------------------------------------------
    def canvas_size(self, document, target, frame=None, tier=1):
        return _canvas_size_for_chain(document, target, frame or 1, int(tier))[:2]

    def supports_tiled(self, document, target):
        """True iff every evaluated ancestor (including Switch selected branch, disabled bypass)
        is in SUPPORTED_TILED_KINDS and the chain ends at a generator."""
        nodes = document["nodes"]
        kind_seen = False
        for node_id in _all_ancestors(document, target):
            node = nodes[node_id]
            if node["type"] not in SUPPORTED_TILED_KINDS:
                return False
            kind_seen = True
        return kind_seen  # empty chain -> not tiled

    def compose(self, document, target, frame: int = 1, tier: int = 1,
                cancel: threading.Event | None = None) -> TileResult:
        """Render `target` to a full-frame RGBA float32 array, tile-by-tile."""
        tier = int(tier)
        frame = int(frame)
        if tier not in tiers.PROXY_TIERS:
            raise ValueError(f"Unsupported proxy tier {tier}; expected one of {tiers.PROXY_TIERS}")
        if not self.supports_tiled(document, target):
            self.stats["full_frame_fallbacks"] += 1
            pixels = self.evaluator.evaluate(document, target, frame=frame, tier=tier)
            return TileResult(pixels=pixels, canvas_width=int(pixels.shape[1]),
                              canvas_height=int(pixels.shape[0]), tier=tier, tiled=False,
                              tile_hits=0, tile_misses=0, full_frame_fallbacks=1,
                              source_decodes=0)

        # Per-compose transient state.
        self._source_cache.clear()
        decodes_before = self.stats["source_decodes"]
        node_digests = _compute_node_digests(document, tier, frame)

        width, height = _canvas_size_for_chain(document, target, frame, tier)
        nodes = document["nodes"]
        chain = list(_all_ancestors(document, target))
        _validate_merge_formats(document, chain, frame, tier)
        halos = [resolve_halo(nodes[key]["type"], tiers.scale_params(
            nodes[key]["type"], nodes[key]["params"], tier)) for key in chain]
        worst_halo = (max((h[0] for h in halos), default=0),
                      max((h[1] for h in halos), default=0))
        if not fits_in_budget(width, height, self.tile_edge, *worst_halo, self.cache.budget):
            self.stats["full_frame_fallbacks"] += 1
            pixels = self.evaluator.evaluate(document, target, frame=frame, tier=tier)
            return TileResult(pixels=pixels, canvas_width=width, canvas_height=height, tier=tier,
                              tiled=False,
                              tile_hits=self.cache.hits_exact + self.cache.hits_preview,
                              tile_misses=self.cache.misses, full_frame_fallbacks=1,
                              source_decodes=self.stats["source_decodes"] - decodes_before)

        output = np.zeros((height, width, 4), dtype=np.float32)
        tile_hits_before = self.cache.hits_exact + self.cache.hits_preview
        tile_misses_before = self.cache.misses
        for tile_region in iter_tiles(width, height, self.tile_edge):
            if cancel is not None and cancel.is_set():
                raise CancelledTile()
            buffered = self._render_tile(document, target, frame, tier, tile_region,
                                         node_digests, cancel)
            # The cached artifact's `region` is the BUFFERED canvas-clamped extent that the
            # pixels actually cover. The OUTPUT REGION (== `tile_region`) sits inside it,
            # offset by (tile_region.x - buffered.region.x, tile_region.y - buffered.region.y).
            # That offset may be larger than the kernel's halo when the halo is clamped to the
            # canvas edge; the offset is the right thing to use because it's the only thing
            # that aligns canvas coords to pixel coords.
            offset_y = tile_region.y - buffered.region.y
            offset_x = tile_region.x - buffered.region.x
            ys_canvas = slice(tile_region.y, tile_region.bottom)
            xs_canvas = slice(tile_region.x, tile_region.right)
            output[ys_canvas, xs_canvas] = buffered.pixels[
                offset_y:offset_y + tile_region.height,
                offset_x:offset_x + tile_region.width]
        return TileResult(pixels=output, canvas_width=width, canvas_height=height, tier=tier,
                          tiled=True,
                          tile_hits=(self.cache.hits_exact + self.cache.hits_preview) - tile_hits_before,
                          tile_misses=self.cache.misses - tile_misses_before,
                          full_frame_fallbacks=0,
                          source_decodes=self.stats["source_decodes"] - decodes_before)

    # --- internal ---------------------------------------------------------
    def _render_tile(self, document, node_id, frame, tier, region, node_digests, cancel):
        """Return a buffered tile for `node_id` at `region`, populating the cache as a side effect."""
        node = document["nodes"][node_id]
        kind = node["type"]
        if kind not in SUPPORTED_TILED_KINDS:
            raise UnsupportedTile(f"{kind} has no tile-native implementation")
        params = tiers.scale_params(kind, node["params"], tier)
        halo = resolve_halo(kind, params)
        cache_region = dataclasses.replace(region, halo_x=halo[0], halo_y=halo[1])

        content_digest = node_digests.get(node_id, "")
        key = TileKey(node_id=node_id, frame=int(frame), tier=int(tier),
                      region_x=int(cache_region.x), region_y=int(cache_region.y),
                      region_width=int(cache_region.width), region_height=int(cache_region.height),
                      tile_edge=int(self.tile_edge), halo_x=halo[0], halo_y=halo[1], exact=True,
                      content_digest=content_digest)

        cached = self.cache.get(key)
        if cached is not None:
            self.stats["tile_hits"] += 1
            return cached

        buffered_region = cache_region.buffered
        if buffered_region.is_empty:
            empty = np.zeros((max(0, buffered_region.height), max(0, buffered_region.width), 4),
                             dtype=np.float32)
            empty.flags.writeable = False
            artifact = TileArtifact(key=key, pixels=empty, region=cache_region)
            self.cache.put(artifact)
            return artifact
        if cancel is not None and cancel.is_set():
            raise CancelledTile()

        self.stats["tile_misses"] += 1
        self.stats["supported_kinds"] += 1
        self.stats["tile_renders"] += 1

        inputs = self._gather_inputs(document, node_id, node, params, frame, tier, buffered_region,
                                     node_digests, cancel)

        raw = self._evaluate_tile_kernel(kind, params, inputs, buffered_region, frame)
        bh, bw = buffered_region.height, buffered_region.width
        if raw.shape[0] == bh and raw.shape[1] == bw:
            pixels = raw
        else:
            # The kernel produced an over-fetched array (Blur returns the same shape as its input,
            # which is the input expanded by the kernel's own halo). Crop to the buffered region.
            # The offset is `buffered_region.x - input_region.x`, NOT the kernel's halo, because
            # the input might have been clamped at the canvas edge and the kernel's halo only
            # applies to the canonical unclamped region.
            crop_x0, crop_y0 = 0, 0
            if inputs and inputs[0] is not None:
                crop_x0 = max(0, buffered_region.x - inputs[0].region.x)
                crop_y0 = max(0, buffered_region.y - inputs[0].region.y)
            y1 = min(crop_y0 + bh, raw.shape[0])
            x1 = min(crop_x0 + bw, raw.shape[1])
            pixels = raw[crop_y0:y1, crop_x0:x1]
            pad_y = max(0, bh - pixels.shape[0])
            pad_x = max(0, bw - pixels.shape[1])
            if pad_y > 0 or pad_x > 0:
                pixels = np.pad(pixels, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant")
        pixels.flags.writeable = False
        # The artifact's `region` is the BUFFERED region (canvas-clamped extent the pixels
        # actually cover). The cache key uses `cache_region` (OUTPUT region with halo, possibly
        # off-canvas) so two requests that ask for the same buffered pixels but differ in their
        # cache_region share a cache entry — see the placement code in `compose` for how the
        # offset between tile_region and buffered_region is computed to extract the OUTPUT pixels.
        artifact = TileArtifact(key=key, pixels=pixels, region=buffered_region)
        self.cache.put(artifact)
        return artifact

    def _gather_inputs(self, document, node_id, node, params, frame, tier, buffered_region,
                       node_digests, cancel):
        kind = node["type"]
        if kind in ("Read", "Constant", "Checker"):
            return [self._generator_tile(node_id, kind, node, params, frame, tier,
                                         buffered_region, node_digests)]
        # Disabled filter: passthrough to the first wired input (only). The legacy evaluator
        # reads inputs[:1] in this case, so we do the same.
        if node["disabled"]:
            slots = list(SPECS[kind]["inputs"])
            source_id = node["inputs"].get(slots[0]) if slots else None
            if source_id is None:
                return [None]
            return [self._render_tile(document, source_id, frame, tier,
                                     dataclasses.replace(buffered_region, halo_x=0, halo_y=0),
                                     node_digests, cancel)]

        slots = list(SPECS[kind]["inputs"]) + list(SPECS[kind].get("optional_inputs", []))
        arity = len(slots)
        rule = tiers.input_regions(kind, params, _to_tier_region(buffered_region), arity)
        source_ids = [node["inputs"].get(s) for s in slots]
        full_w, full_h = buffered_region.full_width, buffered_region.full_height
        gathered = []
        for slot, source_id, needed_region in zip(slots, source_ids, rule):
            if source_id is None or needed_region is None or needed_region.is_empty:
                gathered.append(None)
                continue
            clamped = needed_region.clamp(full_w, full_h)
            if clamped.is_empty:
                gathered.append(None)
                continue
            needed = TileRegion(clamped.x, clamped.y, clamped.width, clamped.height,
                                halo_x=0, halo_y=0, full_width=full_w, full_height=full_h)
            if cancel is not None and cancel.is_set():
                raise CancelledTile()
            gathered.append(self._render_tile(document, source_id, frame, tier, needed,
                                              node_digests, cancel))
        return gathered

    def _generator_tile(self, node_id, kind, node, params, frame, tier, buffered_region,
                        node_digests) -> TileArtifact:
        """Render a buffered tile of a Read/Constant/Checker, with source retention per compose.

        Keyed on the node's actual document key (`node_id`), which is unique by construction —
        not on `name`/`path`. An earlier draft derived identity from `node.get("name")`, and node
        names default to the node's kind (e.g. "Constant") when the artist doesn't rename them, so
        two unrenamed Constant nodes with different colours collided on the same synthetic id and
        the same (type, name)-searched digest, and the tile executor rendered one node's colour for
        both. Reproduced directly: two default-named Constant nodes with red vs. green params
        produced byte-identical tiled output before this fix.
        """
        real_digest = node_digests.get(node_id, "")
        key = TileKey(node_id=f"@{kind}:{node_id}", frame=int(frame), tier=int(tier),
                      region_x=int(buffered_region.x), region_y=int(buffered_region.y),
                      region_width=int(buffered_region.width),
                      region_height=int(buffered_region.height),
                      tile_edge=int(self.tile_edge), halo_x=0, halo_y=0, exact=True,
                      content_digest=real_digest)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # Decode once per (node_id, frame, tier) for the lifetime of this compose call.
        cache_key = (node_id, int(frame), int(tier))
        full = self._source_cache.get(cache_key)
        if full is None:
            full = _node_full_image_at_tier(node, frame, tier)
            self._source_cache[cache_key] = full
            self.stats["source_decodes"] += 1
        full_h, full_w = full.shape[:2]
        bx0 = max(0, buffered_region.x - buffered_region.halo_x)
        by0 = max(0, buffered_region.y - buffered_region.halo_y)
        bx1 = min(full_w, buffered_region.right + buffered_region.halo_x)
        by1 = min(full_h, buffered_region.bottom + buffered_region.halo_y)
        if bx1 <= bx0 or by1 <= by0:
            tile_pixels = np.zeros((max(0, buffered_region.height),
                                    max(0, buffered_region.width), 4), dtype=np.float32)
        else:
            tile_pixels = np.ascontiguousarray(full[by0:by1, bx0:bx1])
        tile_pixels.flags.writeable = False
        artifact = TileArtifact(key=key, pixels=tile_pixels, region=buffered_region)
        self.cache.put(artifact)
        return artifact

    def _evaluate_tile_kernel(self, kind, params, inputs, buffered_region, frame):
        """Apply `kind`'s kernel on one tile's worth of inputs.

        `inputs` are passed in the shape the kernel actually needs:
        * Pointwise and Merge kernels expect inputs at the OUTPUT region (== buffered_region,
          because their halo is zero).
        * Blur expects its image input at BUFFERED.expand(blur_halo) shape, so it can read
          neighbours. The cached input for a Blur is already at that expanded region; passing it
          through unchanged is correct.

        The caller is responsible for the OUTPUT-region alignment of Merge inputs (see the
        `Merge` branch below) so the legacy `a.shape != b.shape` invariant is upheld.
        """
        if kind in ("Read", "Constant", "Checker"):
            return inputs[0].pixels.copy()
        if kind == "Viewer":
            return inputs[0].pixels.copy()
        if kind == "Dot":
            return inputs[0].pixels.copy()
        if kind in ("Grade", "ColorCorrect", "Blur"):
            image_artifact = inputs[0]
            image = image_artifact.pixels
            mask_artifact = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
            if kind == "Grade":
                filtered = imaging.Evaluator._grade(image, params)
            elif kind == "ColorCorrect":
                filtered = imaging.Evaluator._color_correct(image, params)
            else:
                filtered = imaging.Evaluator._blur(image, params)
            if mask_artifact is not None and mask_artifact.pixels.shape != image.shape:
                # `tiers._blur_rule` (and the identity rule for Grade/ColorCorrect) declares the
                # mask's needed ROI as the plain, unexpanded output region — "the mask gates the
                # *result*, so it is only ever sampled at the output pixels themselves" — while
                # `image` may be halo-expanded (Blur). `_apply_mask_mix` requires matching shapes,
                # so crop image/filtered down to the mask's shape using the same region-to-region
                # offset the post-kernel crop in `_render_tile` uses for the same purpose.
                # Reproduced directly: a Blur with a wired mask raised "Mask shape ... does not
                # match source" on every call before this fix — Blur+mask was unusable tiled.
                mh, mw = mask_artifact.pixels.shape[:2]
                offset_x = max(0, mask_artifact.region.x - image_artifact.region.x)
                offset_y = max(0, mask_artifact.region.y - image_artifact.region.y)
                image = image[offset_y:offset_y + mh, offset_x:offset_x + mw]
                filtered = filtered[offset_y:offset_y + mh, offset_x:offset_x + mw]
            mask = mask_artifact.pixels if mask_artifact is not None else None
            return imaging.Evaluator._apply_mask_mix(image, filtered, mask=mask,
                                                     mix=params.get("mix", 1.0))
        if kind in ("Shuffle", "Premult", "Unpremult"):
            return _run_full_kernel_on_array(kind, params, [inputs[0].pixels], frame)
        if kind == "Merge":
            # Merge's halo is zero, so buffered_region IS the output region. Both inputs need to
            # be at that shape; the cached artifacts are at their own buffered extents (different
            # halo-driven sizes), so we crop both into the common output region here.
            target_region = buffered_region
            a_pixels = _align_artifact_to(inputs[0], target_region) if inputs[0] is not None else None
            b_pixels = _align_artifact_to(inputs[1], target_region) if inputs[1] is not None else None
            op = params.get("operation", "over")
            mix = params["mix"]
            if op == "over":
                gate = a_pixels[..., 3:4] * np.float32(mix)
                return (a_pixels * np.float32(mix) + b_pixels * (1 - gate)).astype(np.float32)
            merged = imaging.Evaluator._merge_op(op, a_pixels, b_pixels)
            return (merged * np.float32(mix) + b_pixels * (1 - np.float32(mix))).astype(np.float32)
        raise UnsupportedTile(f"{kind} has no tile-native implementation")


def _align_artifact_to(artifact: TileArtifact, target_region: TileRegion) -> np.ndarray:
    """Crop (and pad, if the artifact's buffered region was clamped to canvas) the artifact to
    exactly the target region's shape.

    Each artifact pixel sits at canvas `(region.x + j, region.y + i)`. To produce a tile at
    `target_region` we have to find, for each target canvas position, the corresponding source
    pixel: `target[i, j] = artifact[i + (target.y - region.y), j + (target.x - region.x)]`.
    Source pixels outside the artifact's bounds are filled with transparent black.
    """
    src = artifact.pixels
    src_region = artifact.region
    target_h, target_w = target_region.height, target_region.width
    src_h, src_w = src.shape[:2]
    if src_h == target_h and src_w == target_w and src_region.x == target_region.x \
            and src_region.y == target_region.y:
        return src
    # Inside the source's footprint, slice the right chunk of the artifact.
    overlap_x0 = max(src_region.x, target_region.x)
    overlap_y0 = max(src_region.y, target_region.y)
    overlap_x1 = min(src_region.right, target_region.right)
    overlap_y1 = min(src_region.bottom, target_region.bottom)
    if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
        return np.zeros((target_h, target_w, 4), dtype=np.float32)
    src_x0 = overlap_x0 - src_region.x
    src_y0 = overlap_y0 - src_region.y
    src_x1 = overlap_x1 - src_region.x
    src_y1 = overlap_y1 - src_region.y
    cropped = src[src_y0:src_y1, src_x0:src_x1]
    out_h, out_w = cropped.shape[:2]
    pad_top = overlap_y0 - target_region.y
    pad_left = overlap_x0 - target_region.x
    if pad_top == 0 and pad_left == 0 and out_h == target_h and out_w == target_w:
        return cropped
    padded = np.zeros((target_h, target_w, cropped.shape[2]), dtype=cropped.dtype)
    padded[pad_top:pad_top + out_h, pad_left:pad_left + out_w] = cropped
    return padded


def _run_full_kernel_on_array(kind, params, inputs, frame):
    """Invoke the existing full-frame kernel as if it were on a full image, returning the array.

    Some kernels (Shuffle/Premult/Unpremult) read pixel-for-pixel and produce the same output
    shape as the input, so a tile of the input is just the tile of the output. This wrapper
    keeps their handling alongside the other pointwise filters without re-implementing the
    maths.
    """
    image = inputs[0]
    if kind == "Shuffle":
        index = {"R": 0, "G": 1, "B": 2, "A": 3}
        h, w = image.shape[:2]
        def pick(name):
            if name in index:
                return image[..., index[name]:index[name] + 1]
            return np.full((h, w, 1), float(name), dtype=np.float32)
        return np.concatenate([pick(params["red_from"]), pick(params["green_from"]),
                               pick(params["blue_from"]), pick(params["alpha_from"])],
                              axis=2).astype(np.float32)
    if kind == "Premult":
        out = image.copy()
        alpha = image[..., 3:4]
        out[..., :3] = image[..., :3] * alpha
        return out
    if kind == "Unpremult":
        alpha = image[..., 3:4]
        safe = np.where(alpha > 0, alpha, np.float32(1.0))
        straight = np.where(alpha > 0, image[..., :3] / safe, image[..., :3])
        return np.concatenate([straight, alpha], axis=2).astype(np.float32)
    raise UnsupportedTile(f"{kind} has no tile-native full-kernel path")


def _canvas_size_for_chain(document, target, frame, tier):
    """Walk back through EVALUATED ancestors to a generator and return (width, height, chain).

    Respects Switch's selected branch and the disabled passthrough; verifies that all reachable
    generators agree on the canvas size, since M0 requires it.
    """
    nodes = document["nodes"]
    seen = set()
    chain = []
    cursor = target
    while cursor is not None and cursor not in seen:
        seen.add(cursor)
        chain.append(cursor)
        node = nodes[cursor]
        if node["type"] in ("Constant", "Checker"):
            params = tiers.scale_params(node["type"], node["params"], tier)
            return int(params["width"]), int(params["height"])
        if node["type"] == "Read":
            params = dict(node["params"])
            params["frame"] = frame
            full = imaging.read_image(**params)
            if int(tier) != 1:
                full = Evaluator._decimate(full, int(tier))
            return int(full.shape[1]), int(full.shape[0])
        # Pick the next upstream node through the relevant branch.
        if node["disabled"] and node["type"] not in ("Read", "Constant", "Checker"):
            slots = list(SPECS[node["type"]]["inputs"])
            cursor = node["inputs"].get(slots[0]) if slots else None
            continue
        if node["type"] == "Switch":
            which = int(node["params"].get("which", 0))
            slots = list(SPECS["Switch"]["inputs"])
            cursor = node["inputs"].get(slots[which]) if 0 <= which < len(slots) else None
            continue
        slots = list(SPECS[node["type"]]["inputs"])
        cursor = next((node["inputs"][s] for s in slots if node["inputs"].get(s) is not None), None)
    raise ValueError(f"Cannot determine a canvas for {target!r}: no generator reached")


def _validate_merge_formats(document, chain, frame, tier):
    """Reject a Merge whose A and B branches disagree on canvas size, matching the reference
    evaluator's `a.shape != b.shape` check (`imaging.py` "Merge inputs must have matching formats
    in M0").

    `_evaluate_tile_kernel`'s Merge branch aligns both inputs to a common tile-sized array before
    comparing them, so `a.shape != b.shape` can never fire there — both are always forced to the
    same shape by construction, whatever the underlying full-canvas sizes actually are. That let a
    64x64 Constant merge with a 16x16 Constant render silently at tile-executor level while the
    full-frame evaluator correctly raised on the identical graph. Checked once per `compose()`,
    against the two branches' true canvas sizes, before any tile is rendered.
    """
    nodes = document["nodes"]
    for node_id in chain:
        node = nodes[node_id]
        if node["type"] != "Merge" or node["disabled"]:
            continue
        a_id, b_id = node["inputs"].get("A"), node["inputs"].get("B")
        if a_id is None or b_id is None:
            continue
        if _canvas_size_for_chain(document, a_id, frame, tier) != \
                _canvas_size_for_chain(document, b_id, frame, tier):
            raise ValueError("Merge inputs must have matching formats in M0")
