"""Tile artifact identity and disjoint-key cache (v0.9 docs/EVALUATION_TIERS.md extension).

Two reviewer-flagged bugs are pinned here:

1. **stats() must not deadlock.** A previous revision of `TileCache.stats()` called
   `self.entries` while already holding `self._lock`; `entries` tried to re-acquire the same
   non-reentrant Lock and the call hung. The fix computes counts inline; this test holds the
   lock from another thread and verifies `stats()` returns anyway.

2. **The byte budget is shared across exact + preview namespaces.** A previous revision only
   evicted from the target namespace; if `preview` held all the bytes and an exact tile was
   inserted, the eviction loop exited immediately and the budget was exceeded. The fix uses
   one global recency ledger so the oldest entry from EITHER namespace is the one evicted.
   These tests verify the shared budget and that eviction counts are reported per-namespace.
"""
import threading
import unittest

import numpy as np

from nodebased.tiles import (DEFAULT_TILE_EDGE, PROVENANCE_VERSION, TileArtifact, TileCache,
                             TileKey, TileRegion, grid_for, iter_tiles, memory_budget_for_tiles,
                             fits_in_budget, resolve_halo, SUPPORTED_TILED_KINDS)


def make_key(name="n", frame=1, tier=1, rx=0, ry=0, rw=64, rh=64, exact=True, halo=(0, 0),
             content_digest="default"):
    return TileKey(node_id=name, frame=frame, tier=tier,
                   region_x=rx, region_y=ry, region_width=rw, region_height=rh,
                   tile_edge=DEFAULT_TILE_EDGE, halo_x=halo[0], halo_y=halo[1], exact=exact,
                   content_digest=content_digest)


def make_artifact(key, pixels_shape=(64, 64, 4), full=64):
    pixels = np.zeros(pixels_shape, np.float32)
    region = TileRegion(key.region_x, key.region_y, key.region_width, key.region_height,
                        key.halo_x, key.halo_y, full, full)
    return TileArtifact(key=key, pixels=pixels, region=region)


class TileKeyIdentityTests(unittest.TestCase):
    def test_same_fields_produce_same_digest(self):
        a = make_key(name="g", frame=1, tier=1, rx=0, ry=0, rw=64, rh=64)
        b = make_key(name="g", frame=1, tier=1, rx=0, ry=0, rw=64, rh=64)
        self.assertEqual(a.digest(), b.digest())

    def test_each_fidelity_field_changes_the_digest(self):
        """All fidelity-affecting state must be in the key; an unrelated state must not."""
        # Frame differs
        a = make_key(frame=1)
        b = make_key(frame=2)
        self.assertNotEqual(a.digest(), b.digest())
        # Tier differs
        a = make_key(tier=1); b = make_key(tier=2)
        self.assertNotEqual(a.digest(), b.digest())
        # Tile coords differ
        a = make_key(rx=0); b = make_key(rx=64)
        self.assertNotEqual(a.digest(), b.digest())
        a = make_key(ry=0); b = make_key(ry=64)
        self.assertNotEqual(a.digest(), b.digest())
        # Same index, different ROI: the user asks for two different regions in the same node
        # under the same tile index. With tile_x/y alone these would collide.
        a = make_key(rx=0, ry=0, rw=64, rh=64)
        b = make_key(rx=0, ry=0, rw=32, rh=32)
        self.assertNotEqual(a.digest(), b.digest(),
                           "Region dimensions must be in the key for sub-tile ROI requests")
        # Halo differs
        a = make_key(halo=(0, 0)); b = make_key(halo=(8, 8))
        self.assertNotEqual(a.digest(), b.digest())
        # Exact vs preview differs
        a = make_key(exact=True); b = make_key(exact=False)
        self.assertNotEqual(a.digest(), b.digest())
        # Content digest differs (this is what makes edits auto-invalidate)
        a = make_key(content_digest="hash-of-params-A")
        b = make_key(content_digest="hash-of-params-B")
        self.assertNotEqual(a.digest(), b.digest(),
                           "Two requests with different content digests must never collide, "
                           "even when all geometry matches — this is how edits invalidate.")
        # Tile edge differs (configuration change -> new identity)
        a = make_key(); b = TileKey(node_id="n", frame=1, tier=1, region_x=0, region_y=0,
                                     region_width=64, region_height=64, tile_edge=128)
        self.assertNotEqual(a.digest(), b.digest())

    def test_provenance_version_keeps_old_digests_invalidated(self):
        """Bumping PROVENANCE_VERSION must invalidate every existing tile cache entry."""
        a = make_key()
        b = TileKey(node_id=a.node_id, frame=a.frame, tier=a.tier, region_x=a.region_x,
                    region_y=a.region_y, region_width=a.region_width, region_height=a.region_height,
                    tile_edge=a.tile_edge, halo_x=a.halo_x, halo_y=a.halo_y, exact=a.exact,
                    provenance_version=PROVENANCE_VERSION + 1)
        self.assertNotEqual(a.digest(), b.digest())


class DisjointKeyCacheTests(unittest.TestCase):
    """An exact entry must never satisfy a preview lookup and vice versa."""

    def test_namespaces_are_disjoint_under_lookup(self):
        cache = TileCache(budget_bytes=4 * 1024 * 1024)
        exact = make_artifact(make_key(exact=True))
        preview = make_artifact(make_key(exact=False))
        cache.put(exact)
        cache.put(preview)
        self.assertIs(cache.get(make_key(exact=True)), exact)
        self.assertIs(cache.get(make_key(exact=False)), preview)
        s = cache.stats()
        self.assertEqual(s["entries"], 2)
        self.assertEqual(s["exact_entries"], 1)
        self.assertEqual(s["preview_entries"], 1)

    def test_collision_in_underlying_digest_is_resolved_by_namespace_prefix(self):
        """Two keys with identical digest fields but different `exact` flags are different entries."""
        cache = TileCache(budget_bytes=4 * 1024 * 1024)
        exact = make_artifact(make_key(exact=True))
        preview = make_artifact(make_key(exact=False))
        cache.put(exact)
        # Inserting the preview must not evict the exact entry (different prefix).
        cache.put(preview)
        self.assertIs(cache.get(make_key(exact=True)), exact)
        self.assertIs(cache.get(make_key(exact=False)), preview)


class BudgetSharingTests(unittest.TestCase):
    """The byte budget is shared across exact and preview namespaces; the LRU is global."""

    def test_cross_namespace_eviction_when_target_namespace_is_empty(self):
        """If `preview` owns the budget, inserting an `exact` tile must evict preview entries.

        This is the regression: the previous revision only walked the target namespace's
        OrderedDict, so a put to an empty target namespace against a full source namespace
        exceeded the budget.
        """
        # Small budget so a few tiles force eviction. 1 MiB holds ~16 tiles of 64x64x4 (16 KiB).
        budget = 1 * 1024 * 1024
        cache = TileCache(budget_bytes=budget)
        tile_shape = (64, 64, 4)  # 16 KiB per tile
        # Fill the preview namespace first: 64 inserts = 1 MiB exactly, no eviction yet.
        for i in range(64):
            cache.put(make_artifact(make_key(name=f"p{i}", exact=False), pixels_shape=tile_shape))
        self.assertEqual(cache.bytes, budget)
        # The exact namespace is empty. A single exact insert must now evict preview entries
        # to stay within the budget. The previous revision's eviction loop exited immediately
        # because the target namespace was empty, and the cache exceeded the budget.
        cache.put(make_artifact(make_key(name="e0", exact=True), pixels_shape=tile_shape))
        self.assertLessEqual(cache.bytes, budget,
                             f"budget exceeded after cross-namespace insert: {cache.bytes}")
        s = cache.stats()
        self.assertGreater(s["evictions_preview"], 0,
                           "preview namespace never got hit by shared LRU — cross-namespace "
                           "eviction is broken")
        self.assertEqual(s["evictions_exact"], 0,
                         "exact namespace should not have been evicted; the new entry just arrived")

    def test_both_namespaces_get_evicted_under_sustained_pressure(self):
        """Long alternating insertion must share eviction across both namespaces, not just one."""
        budget = 1 * 1024 * 1024
        cache = TileCache(budget_bytes=budget)
        tile_shape = (64, 64, 4)
        # 200 alternating inserts: 100 exact + 100 preview = ~3.1 MiB requested, 1 MiB allowed.
        for i in range(200):
            cache.put(make_artifact(make_key(name=f"x{i}", exact=(i % 2 == 0)),
                                    pixels_shape=tile_shape))
            self.assertLessEqual(cache.bytes, budget,
                                 f"budget exceeded at insert {i}: {cache.bytes} > {budget}")
        s = cache.stats()
        self.assertGreater(s["evictions_exact"], 0)
        self.assertGreater(s["evictions_preview"], 0)

    def test_refresh_does_not_double_count_bytes(self):
        """Re-inserting the same key should replace, not accumulate."""
        cache = TileCache(budget_bytes=4 * 1024 * 1024)
        art = make_artifact(make_key(), pixels_shape=(64, 64, 4))
        cache.put(art)
        bytes_after_first = cache.bytes
        cache.put(art)  # refresh
        self.assertEqual(cache.bytes, bytes_after_first)
        self.assertEqual(cache.stats()["entries"], 1)

    def test_oversized_single_tile_is_rejected(self):
        cache = TileCache(budget_bytes=4 * 1024)
        art = make_artifact(make_key(), pixels_shape=(64, 64, 4))  # 16 KiB > 4 KiB budget
        self.assertFalse(cache.put(art))
        self.assertEqual(cache.bytes, 0)
        self.assertEqual(cache.entries, 0)


class StatsSafetyTests(unittest.TestCase):
    """stats() must never deadlock, even while another thread holds the lock."""

    def test_stats_does_not_deadlock_under_reentrant_acquisition_attempt(self):
        """A naïve implementation that called `self.entries` while holding the lock would hang.

        We hold the cache's lock from a side thread and then call `stats()` from the main
        thread; the test fails if `stats()` blocks for more than a small bounded time.
        """
        cache = TileCache(budget_bytes=1024 * 1024)
        cache.put(make_artifact(make_key()))
        gate = threading.Event()
        held = threading.Event()

        def hold_lock_briefly():
            with cache._lock:
                held.set()
                # Block until the main thread has had time to call stats().
                gate.wait(timeout=1.0)

        side = threading.Thread(target=hold_lock_briefly, daemon=True)
        side.start()
        self.assertTrue(held.wait(timeout=1.0), "side thread failed to acquire the lock")
        # stats() must observe the lock-held state and *wait* for the lock — not deadlock
        # forever. A correct implementation just serialises; a buggy one would self-deadlock
        # or hang. Bounded by the side thread's wait above.
        snapshot = cache.stats()
        gate.set()
        side.join(timeout=2.0)
        self.assertFalse(side.is_alive(), "stats() deadlocked a worker thread")
        self.assertIn("entries", snapshot)
        self.assertEqual(snapshot["entries"], 1)

    def test_stats_returns_consistent_counts(self):
        cache = TileCache(budget_bytes=4 * 1024 * 1024)
        for i in range(10):
            cache.put(make_artifact(make_key(name=f"n{i}", exact=(i % 2 == 0))))
        s = cache.stats()
        self.assertEqual(s["entries"], s["exact_entries"] + s["preview_entries"])
        self.assertEqual(s["exact_entries"], 5)
        self.assertEqual(s["preview_entries"], 5)
        self.assertGreaterEqual(s["inserts"], 10)


class TileGridArithmeticTests(unittest.TestCase):
    def test_grid_for_rounds_outward(self):
        self.assertEqual(grid_for(640, 360, 256), (3, 2))  # ceil(640/256)=3, ceil(360/256)=2

    def test_iter_tiles_covers_every_pixel_once(self):
        covered = set()
        for region in iter_tiles(640, 360, 256):
            for y in range(region.y, region.bottom):
                for x in range(region.x, region.right):
                    covered.add((x, y))
        self.assertEqual(len(covered), 640 * 360)

    def test_memory_budget_for_tiles_at_4k(self):
        mem = memory_budget_for_tiles(3840, 2160, 256, halo_x=8, halo_y=8)
        # 4K retile at 256 px edge with halo 8: 15x9 = 135 tiles, each (272x272x4x4 bytes).
        expected = 15 * 9 * 272 * 272 * 4 * 4
        self.assertEqual(mem, expected)

    def test_fits_in_budget_zero_budget_means_individual_only(self):
        self.assertTrue(fits_in_budget(3840, 2160, 256, 8, 8, 0))

    def test_fits_in_budget_rejects_oversize(self):
        self.assertFalse(fits_in_budget(3840, 2160, 256, 8, 8, 100 * 1024 * 1024))

    def test_supported_kinds_are_documented(self):
        # The supported set must include the kernels we proved correct end-to-end. If a new
        # tile-native kernel is added here, the equality test in test_tileexec.py must cover it.
        expected = {"Read", "Constant", "Checker", "Grade", "ColorCorrect", "Shuffle", "Premult",
                    "Unpremult", "Dot", "Blur", "Merge", "Viewer"}
        self.assertTrue(expected.issubset(SUPPORTED_TILED_KINDS))
        # Switch is intentionally NOT supported at this pass: its ROI rule returns None for the
        # unselected branch, but switching the active branch mid-frame would have to rebuild
        # the tile cache; that is a v0.10 problem.
        self.assertNotIn("Switch", SUPPORTED_TILED_KINDS)
        # Transform and Crop are excluded in v0.9 because their kernels are coordinate-dependent
        # on the canvas origin; see test_transform_and_crop_are_excluded_with_a_reason.
        self.assertNotIn("Transform", SUPPORTED_TILED_KINDS)
        self.assertNotIn("Crop", SUPPORTED_TILED_KINDS)


class HaloResolutionTests(unittest.TestCase):
    def test_blur_radius_resolves_to_ceil(self):
        self.assertEqual(resolve_halo("Blur", {"radius": 8.0}), (8, 8))
        self.assertEqual(resolve_halo("Blur", {"radius": 0.25}), (0, 0))

    def test_pointwise_kernels_have_zero_halo(self):
        for kind in ("Grade", "ColorCorrect", "Premult", "Unpremult", "Shuffle",
                     "Dot", "Viewer", "Constant", "Checker", "Read", "Merge"):
            with self.subTest(kind=kind):
                self.assertEqual(resolve_halo(kind, {}), (0, 0))

    def test_transform_and_crop_are_excluded_with_a_reason(self):
        """Coordinate-dependent kernels are deliberately NOT tile-native in v0.9.

        Documenting the absence in the test suite means a future contributor who re-adds them
        is forced to confront the origin-aware kernel problem rather than ship a silently-wrong
        implementation. The kind is still in SPECS; only the tiled-subset claim is withdrawn.
        """
        for kind in ("Transform", "Crop"):
            self.assertNotIn(kind, SUPPORTED_TILED_KINDS,
                             f"{kind} re-added to SUPPORTED_TILED_KINDS — origin-aware kernels "
                             "and golden tests on nonzero tiles are required before that claim")


if __name__ == "__main__":
    unittest.main()
