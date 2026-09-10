"""Tile artifact identity, the disjoint tile cache, and tile-grid arithmetic.

**Why this exists.** The v0.8 evaluator caches full-frame results by a digest that mixes the
node's kind, scaled parameters, input digests, file fingerprints, frame and tier. That digest is
correct for "did the artist get the comp back identical" — but the tier cache is too coarse to
exploit the fact that a Blur of region (2000, 4000, 400, 400) only needs a 432×432 patch of its
input. A 4K plate has ~216 of those patches; recomputing them one by one for a Grade tweak on a
crop-aware workspace was the actual bottleneck the v0.8 cache could not address.

The v0.9 work introduces an explicit tile artifact with every fidelity-affecting state in its
identity, so two tiles that should render identically have the same key and two tiles that should
render differently never collide. Concretely:

* `tier` is folded in, so a tier 4 proxy tile never satisfies a tier 1 export request (contract C1).
* `exact` distinguishes an evaluated tile from a preview that was rebuilt from a higher tier — for
  the supported kernel subset the tile-native evaluator and the full-frame evaluator produce
  identical pixels at the same tier, so this is not "preview as second-class data" but rather a
  forward-compatibility flag for the day a future lossless tier (e.g., a tier-2 render requested
  while only a tier-4 one lives in cache) decides explicitly what to do.
* `halo` is the support reservation needed by the kernel — blur radius or transform filter wings.
  An exact tile's identity includes the halo it was evaluated with, because a halo-4 tile does
  not legitimately satisfy a halo-8 request: the boundary pixels were filtered using only four
  neighbours and would put a faint dark seam in the centre of a halo-8 composite. This is the
  difference between an LRU of "checkers" and an LRU of "true tile artefacts".
* `provenance_version` is the digit that lets the cache reject an entry written by a previous
  implementation that meant the same field to mean something different. 1 is the version this
  module ships; on bump, old entries are misses, never re-interpreted (see cachetier.py C4).

**Disjoint keys.** Exact and preview tile caches live under separate namespaces so the cache
returning a preview for an exact request — or vice versa — is syntactically impossible rather
than a runtime mistake to avoid.

**Working-set arithmetic.** The tile-grid size and halo are configured from the host's memory
budget rather than fixed. At 4K with the default 256×256 tile, a "complete tile retile" is the
working set; this module exports the arithmetic (`memory_budget_for_tiles`) so the bench harness
and the app can budget the executor's pool without sizing by hand.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Iterable

import numpy as np

from . import cachetier

# Schema version of the tile artifact identity itself. Bumping this invalidates every existing
# tile; the new value does not have to mean anything new, but downstream code MUST re-validate
# its assumptions about which fields are in the digest.
PROVENANCE_VERSION = 1

# Default tile edge in pixels. Configurable per-call; this is what the bench and the app settle
# on when there is no other signal.
DEFAULT_TILE_EDGE = 256


@dataclass(frozen=True)
class TileKey:
    """Identity of one tile of one node at one frame and one tier.

    All fields are part of the deterministic digest. Two tiles that should render to the same
    pixel data MUST collide on every field; two tiles that should render differently MUST differ
    in at least one. The frozen dataclass gives us hashing out of the box; the dedicated
    `digest()` method forces a canonical field ordering so the cache never gets fooled by Python
    dict order.

    `region_x`/`region_y`/`region_width`/`region_height` are the canvas-relative output region of
    this tile (the central area; not the buffer). Two distinct requests on the same node at the
    same frame and tier must address their own buffered regions explicitly — the cached buffer's
    bounding box is implied by `region`, `halo_x`/`halo_y`, and the canvas extent carried by the
    holding `TileArtifact.region`.

    `tile_edge` is the canonical tile edge in pixels — the size the executor would have asked for
    if it had been free to choose. A tile's on-disk shape can be smaller (canvas edge) or larger
    (a blur output that includes halo); the key still records the canonical edge so two tile
    requests under the same configuration hit the same cache entry.

    `content_digest` is the deterministic hash of the resolved node content (kind + scaled params
    + input hashes + Read file fingerprint + tier + frame). Without it, an edit to a Grade's
    exposure would return stale pixels under the same (node_id, region): the document's identity
    is the chain of parameter values, not the slot the artist happened to put it in. The
    content digest makes edits auto-invalidate: the old key no longer exists in the cache, so a
    miss triggers a fresh render. The old entry sits until LRU evicts it.
    """

    node_id: str
    frame: int
    tier: int
    region_x: int
    region_y: int
    region_width: int
    region_height: int
    tile_edge: int
    halo_x: int = 0
    halo_y: int = 0
    exact: bool = True
    content_digest: str = ""
    provenance_version: int = PROVENANCE_VERSION

    def digest(self) -> str:
        payload = {
            "node_id": self.node_id,
            "frame": int(self.frame),
            "tier": int(self.tier),
            "region": [int(self.region_x), int(self.region_y),
                       int(self.region_width), int(self.region_height)],
            "tile_edge": int(self.tile_edge),
            "halo": [int(self.halo_x), int(self.halo_y)],
            "exact": bool(self.exact),
            "content_digest": str(self.content_digest),
            "provenance_version": int(self.provenance_version),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class TileArtifact:
    """One tile of pixels with its identity. Pixels are read-only float32 RGBA."""

    key: TileKey
    pixels: "np.ndarray"   # shape (h, w, 4), float32, read-only
    region: "TileRegion"   # the output region covered by this tile (after halo strip)

    @property
    def shape(self) -> tuple:
        return tuple(self.pixels.shape[:2])

    @property
    def bytes(self) -> int:
        return int(self.pixels.nbytes)


@dataclass(frozen=True)
class TileRegion:
    """An integer pixel rectangle in node output space, anchored to a tile-grid layout.

    Coordinates are inclusive-on-the-output / halo-inclusive-on-the-buffer: a TileRegion is the
    part of the output the artist asked for; the buffer it lives in is bigger by `halo_x`/`halo_y`
    on each side so kernels that need neighbour reads can run on it.
    """

    x: int
    y: int
    width: int
    height: int
    halo_x: int = 0
    halo_y: int = 0
    full_width: int = 0    # total canvas width (so adjacent tiles can be sanity-checked)
    full_height: int = 0

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def buffered(self) -> "TileRegion":
        """Region including halo, clamped to the canvas so a halo-overflow never indexes out."""
        x0 = max(0, self.x - self.halo_x)
        y0 = max(0, self.y - self.halo_y)
        x1 = min(self.full_width, self.right + self.halo_x)
        y1 = min(self.full_height, self.bottom + self.halo_y)
        return TileRegion(x0, y0, x1 - x0, y1 - y0, 0, 0, self.full_width, self.full_height)

    def as_output_slices(self):
        """Slices into the buffer addressing just the output pixels (halo stripped)."""
        ox = self.x - max(0, self.x - self.halo_x)
        oy = self.y - max(0, self.y - self.halo_y)
        return (slice(oy, oy + self.height), slice(ox, ox + self.width))

    def as_buffered_slices(self):
        bx0 = max(0, self.x - self.halo_x)
        by0 = max(0, self.y - self.halo_y)
        bx1 = min(self.full_width, self.right + self.halo_x)
        by1 = min(self.full_height, self.bottom + self.halo_y)
        return (slice(by0, by1), slice(bx0, bx1))


def grid_for(width: int, height: int, edge: int = DEFAULT_TILE_EDGE) -> tuple:
    """Number of tiles in (columns, rows) to cover a canvas with the given edge.

    Both axes ceil out so a tile whose left/top edge starts inside the canvas is still drawn —
    matching the half-open `[x, right)` convention the evaluator uses elsewhere.
    """
    edge = max(1, int(edge))
    return -(-int(width) // edge), -(-int(height) // edge)


def iter_tiles(width: int, height: int, edge: int = DEFAULT_TILE_EDGE,
               halo_x: int = 0, halo_y: int = 0) -> Iterable[TileRegion]:
    """Walk the tile grid, yielding one TileRegion per tile.

    Yields regions whose `x`/`y` are multiples of `edge`, with extents clipped to the canvas. The
    returned regions include `halo`; `buffered`/`as_output_slices` is what downstream code uses
    to split the buffer from the output.
    """
    columns, rows = grid_for(width, height, edge)
    for ty in range(rows):
        for tx in range(columns):
            x = tx * edge
            y = ty * edge
            w = min(edge, width - x)
            h = min(edge, height - y)
            yield TileRegion(x, y, w, h, halo_x, halo_y, width, height)


def tile_at(region: TileRegion, edge: int = DEFAULT_TILE_EDGE) -> tuple:
    """Return `(tile_x, tile_y)` indices for a region's top-left pixel."""
    edge = max(1, int(edge))
    return int(region.x) // edge, int(region.y) // edge


def memory_budget_for_tiles(width: int, height: int, tile_edge: int = DEFAULT_TILE_EDGE,
                            halo_x: int = 0, halo_y: int = 0) -> int:
    """Memory held by one full retile of a canvas at the given halo, in bytes.

    This is the working set an in-progress tile render must reserve. Each tile costs
    `(edge + 2 halo_x) * (edge + 2 halo_y) * 4 (channels) * 4 (float32 bytes)` plus a few
    scratch buffers; the formula below is the conservative upper bound for the tile data alone.
    """
    tile_edge = max(1, int(tile_edge))
    halo_x = max(0, int(halo_x))
    halo_y = max(0, int(halo_y))
    bw = tile_edge + 2 * halo_x
    bh = tile_edge + 2 * halo_y
    columns, rows = grid_for(width, height, tile_edge)
    return int(columns) * int(rows) * bw * bh * 4 * np.dtype(np.float32).itemsize


def fits_in_budget(width: int, height: int, tile_edge: int, halo_x: int, halo_y: int,
                   budget_bytes: int) -> bool:
    """Whether a full retile of `width`×`height` at this tile edge and halo fits the budget.

    `0` budget is interpreted as "only ask for individual tiles rather than a full retile"; the
    call returns True so eager scheduling does not deadlock when the host chooses zero-headroom
    ingest for sources that are not on screen.
    """
    if budget_bytes <= 0:
        return True
    return memory_budget_for_tiles(width, height, tile_edge, halo_x, halo_y) <= budget_bytes


# --- Supported kernel subset -------------------------------------------------------------------
#
# Tile-native execution is opt-in per kernel. A kernel appears here if and only if either:
# 1. it has no spatial support (pointwise), in which case it reads exactly what it was asked for
#    and needs no halo; or
# 2. its ROI rule in `tiers.py` declares the precise support in pixels, so the tile executor can
#    reserve the matching halo and compose tiles exactly.
#
# A kernel not in this table is rendered by the full-frame evaluator and must never pretend to be
# tiled. The TileExecutor surfaces this distinction so a caller can see whether the result came
# from a tile-native execution.
#
# **Transform and Crop are deliberately excluded.** Both are coordinate-dependent on the canvas
# origin: `_transform` reads destination pixel centres (`gx, gy` with i+0.5 origin), and `_crop`
# reads its rectangle in canvas space. Reusing the legacy kernels on a tile buffer produces pixels
# that are not byte-identical to a crop of the full-frame render; the reviewer flagged this as a
# silent-correctness issue. Rather than ship a half-right implementation, the v0.9 tile executor
# falls back to `Evaluator.evaluate` for any graph that contains Transform or Crop. Adding them
# back requires origin-aware kernel variants + golden tests on nonzero tiles — a v0.10 task.

SUPPORTED_TILED_KINDS = frozenset({
    "Read", "Constant", "Checker",          # generators or sources whose downsampled form is exact
    "Grade", "ColorCorrect",                # pointwise, halo = (0, 0)
    "Shuffle", "Premult", "Unpremult",      # pointwise, halo = (0, 0)
    "Dot",                                  # passthrough, halo = (0, 0)
    "Blur",                                 # halo = (radius, radius), declared by tiers._blur_rule
    "Merge",                                # halo = (0, 0); both inputs demand the same output region
    "Viewer",                               # passthrough, halo = (0, 0)
})


# Default halo per supported kernel. The kernel-declared halo (from REGION_RULES) takes
# precedence at request time; this is the floor used when the params don't pin it down.
DEFAULT_HALO_PER_KIND = {
    "Read": (0, 0), "Constant": (0, 0), "Checker": (0, 0),
    "Grade": (0, 0), "ColorCorrect": (0, 0),
    "Shuffle": (0, 0), "Premult": (0, 0), "Unpremult": (0, 0),
    "Dot": (0, 0),
    "Blur": (0, 0),       # resolved at request time from params["radius"]
    "Merge": (0, 0), "Viewer": (0, 0),
}


def resolve_halo(kind: str, params: dict | None) -> tuple:
    """Halo (halo_x, halo_y) this kernel reserves when producing a tile.

    Falls back to the kernel's default if no param-derived halo is known. The result is in the
    *kernel's own output space*, which is also its tile space at any given tier; tier scaling of
    the halo is the caller's job when the request is in tier space.
    """
    params = params or {}
    if kind == "Blur":
        import math
        radius = float(params.get("radius", 0.0))
        # Mirror `_blur_rule`: the kernel reaches `ceil(radius)` pixels each way (the box-blur
        # window is 2r+1, with r=round(radius); the round-up is what makes the rule slightly
        # conservative, which matches the actual filter support).
        support = 0 if radius < 0.5 else int(math.ceil(radius))
        return (support, support)
    if kind == "Transform":
        # Nearest samples 1 pixel; bilinear 1 (its neighbours; covered by +1 below); cubic 2.
        # +1 on every side covers bilinear's neighbour access and absorbs sub-pixel sampling at
        # the destination edge.
        return {"nearest": (1, 1), "bilinear": (1, 1), "cubic": (2, 2)}.get(params.get("filter", "nearest"), (2, 2))
    return DEFAULT_HALO_PER_KIND.get(kind, (0, 0))


# --- Disjoint-key tile cache -------------------------------------------------------------------
#
# Two caches, both LRU and counted against a single byte budget. An exact key lookup never
# returns a preview entry and a preview key lookup never returns an exact entry — the namespacing
# is by prefix, not by analogy.

_EXACT_PREFIX = "exact"
_PREVIEW_PREFIX = "preview"


class TileCache:
    """Bounded LRU of TileArtifact instances, with disjoint exact/preview namespaces.

    Thread-safe so the tile executor's worker threads and the calling UI thread can interact
    without explicit locking on the calling side. Each entry records its identity so the cache
    can answer "do you have this tile?" without an external key map, and so eviction is debug
    rather than mysterious.

    **Disjoint keys, shared budget.** The cache keeps one byte budget across both namespaces so
    a preview-heavy render cannot push exact tiles out of memory by starving the exact namespace,
    nor can an exact-only render leave preview tiles resident forever. Recency is tracked in a
    single global ledger: when a put needs to free space, the oldest entry from either namespace
    is the one evicted. The namespaces remain disjoint by key prefix — an exact request never
    reads a preview entry and vice versa.
    """

    def __init__(self, budget_bytes: int | None = None):
        self.budget = int(cachetier.default_memory_bytes()) if budget_bytes is None else int(budget_bytes)
        # One OrderedDict, one recency order. The key is the namespace-prefixed digest; the
        # namespace is encoded in the key prefix and consulted on every get, never in the
        # ordering. Eviction walks this single dictionary by insertion order.
        self._entries: OrderedDict[str, TileArtifact] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits_exact = 0
        self.hits_preview = 0
        self.misses = 0
        self.evictions = 0
        self.evictions_exact = 0
        self.evictions_preview = 0
        self.inserts = 0

    @staticmethod
    def _namespaced(key: TileKey) -> str:
        namespace = _EXACT_PREFIX if key.exact else _PREVIEW_PREFIX
        return f"{namespace}:{key.digest()}"

    @staticmethod
    def _is_exact(namespaced_key: str) -> bool:
        return namespaced_key.startswith(f"{_EXACT_PREFIX}:")

    def get(self, key: TileKey) -> TileArtifact | None:
        """Return a tile with this identity, or None.

        Exact requests consult the exact cache only; preview requests consult the preview cache
        only. The disjoint-key invariant is enforced here by checking the key's `exact` flag
        against the cached entry's namespace prefix at lookup time — a wrong-namespace match is
        treated as a miss rather than handed back.
        """
        namespaced = self._namespaced(key)
        with self._lock:
            artifact = self._entries.get(namespaced)
            if artifact is None or self._is_exact(namespaced) != bool(key.exact):
                # The second clause is paranoia against any future caller passing a stale key;
                # today `_namespaced` ensures they match.
                self.misses += 1
                return None
            self._entries.move_to_end(namespaced)
            if key.exact:
                self.hits_exact += 1
            else:
                self.hits_preview += 1
            return artifact

    def put(self, artifact: TileArtifact) -> bool:
        """Insert (or refresh) a tile. False when the single tile would blow the budget.

        The byte budget is shared across exact and preview namespaces; eviction walks the global
        recency order so the same memory accounting governs both.
        """
        size = artifact.bytes
        if size > self.budget:
            return False
        namespaced = self._namespaced(artifact.key)
        with self._lock:
            existing = self._entries.get(namespaced)
            if existing is not None and self._is_exact(namespaced) == bool(artifact.key.exact):
                self._bytes -= existing.bytes
                self._entries.pop(namespaced, None)
            elif existing is not None:
                # Wrong namespace — should be impossible, treat as missing rather than corrupt.
                self.misses += 1
            while self._bytes + size > self.budget and self._entries:
                evicted_key, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.bytes
                if self._is_exact(evicted_key):
                    self.evictions_exact += 1
                else:
                    self.evictions_preview += 1
                self.evictions += 1
            self._entries[namespaced] = artifact
            self._bytes += size
            self.inserts += 1
        return True

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    @property
    def bytes(self) -> int:
        return self._bytes

    @property
    def entries(self) -> int:
        # Lock-free read: OrderedDict.__len__ is itself thread-unsafe in isolation, but a stale
        # count here is only ever diagnostic. Hot-path callers should go through `stats()`.
        return len(self._entries)

    def stats(self) -> dict:
        # All counts are computed inside the same lock acquisition so we never see a torn view.
        # Do NOT call `self.entries` from inside this method — `entries` re-reads `_entries`
        # without the lock, and a previous revision of this method deadlocked on that pattern.
        with self._lock:
            count = len(self._entries)
            exact_count = sum(1 for k in self._entries if self._is_exact(k))
            preview_count = count - exact_count
            return {"budget": self.budget, "bytes": self._bytes,
                    "entries": count, "exact_entries": exact_count,
                    "preview_entries": preview_count,
                    "hits_exact": self.hits_exact, "hits_preview": self.hits_preview,
                    "misses": self.misses, "evictions": self.evictions,
                    "evictions_exact": self.evictions_exact,
                    "evictions_preview": self.evictions_preview,
                    "inserts": self.inserts}
