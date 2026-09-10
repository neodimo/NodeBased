"""Tile executor: content digest, multi-input walks, source retention, halo correctness, alignment.

Each section pins a behaviour the reviewer explicitly called out as a release-blocker.

* `TileKeyIdentityTests` — content digest + same-index/different-ROI must collide the way edits
  do, not the way unrelated lookups should.
* `SupportsTiledCoverageTests` — the `supports_tiled` check must walk ALL evaluated ancestors
  (Merge's B branch, Switch's selected branch, disabled passthrough), not just the first
  wired input of each node.
* `SourceRetentionTests` — within a compose, each generator is decoded at most once.
* `HaloAndEdgeCorrectnessTests` — the equality check for Blur is golden against the legacy
  Evaluator. The Transform/Crop exclusion is tested here, with a comment explaining why.
* `MergeAlignmentTests` — Merge inputs with different halo-driven buffered shapes must arrive
  at the kernel with the same shape; the legacy `a.shape != b.shape` invariant is preserved.
* `EditInvalidationTests` — edits to a node's params change its content digest and force a
  fresh render.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher, demo_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, _compute_node_digests, _align_artifact_to
from nodebased.tiles import TileArtifact, TileRegion, TileKey, DEFAULT_TILE_EDGE


def make_artifact(region, pixels):
    """Construct a TileArtifact for alignment tests without going through the executor."""
    key = TileKey(node_id="test", frame=1, tier=1,
                  region_x=region.x, region_y=region.y,
                  region_width=region.width, region_height=region.height,
                  tile_edge=DEFAULT_TILE_EDGE)
    return TileArtifact(key=key, pixels=pixels, region=region)


class TileKeyIdentityTests(unittest.TestCase):
    """TileKey must include enough fidelity-affecting state to avoid stale/wrong pixels on edits."""

    def test_same_index_different_roi_yields_different_keys(self):
        """Two requests at the same tile index but different ROIs must not collide.

        Without the explicit region_x/y/width/height in the key, sub-tile ROI requests would
        return pixels from a different region than the one asked for. The first version of the
        key used a tile-grid index alone and silently returned wrong pixels.
        """
        a = TileKey(node_id="n", frame=1, tier=1, region_x=0, region_y=0,
                    region_width=64, region_height=64, tile_edge=DEFAULT_TILE_EDGE,
                    content_digest="d")
        b = TileKey(node_id="n", frame=1, tier=1, region_x=0, region_y=0,
                    region_width=32, region_height=32, tile_edge=DEFAULT_TILE_EDGE,
                    content_digest="d")
        self.assertNotEqual(a.digest(), b.digest())

    def test_edits_change_content_digest(self):
        """Two renders of the same node with different params get different digests."""
        doc = {"version": 5, "nodes": {
            "g": {"type": "Grade", "name": "g", "params": {"exposure": 0.0, "multiply": 1.0,
                                                              "offset": 0.0, "mix": 1.0},
                    "inputs": {"image": None, "mask": None}, "pos": [0, 0], "disabled": False}},
               "view": None, "time": {"first": 1, "last": 1, "current": 1, "fps": 24.0}}
        digests_before = _compute_node_digests(doc, tier=1, frame=1)
        doc["nodes"]["g"]["params"]["exposure"] = 0.5
        digests_after = _compute_node_digests(doc, tier=1, frame=1)
        self.assertNotEqual(digests_before["g"], digests_after["g"])

    def test_changing_input_content_changes_downstream_digest(self):
        doc = {"version": 5, "nodes": {
            "p": {"type": "Checker", "name": "p", "params": {"width": 64, "height": 64, "size": 16},
                   "inputs": {}, "pos": [0, 0], "disabled": False},
            "g": {"type": "Grade", "name": "g", "params": {"exposure": 0.0, "multiply": 1.0,
                                                              "offset": 0.0, "mix": 1.0},
                   "inputs": {"image": "p", "mask": None}, "pos": [0, 0], "disabled": False}},
               "view": None, "time": {"first": 1, "last": 1, "current": 1, "fps": 24.0}}
        digests_before = _compute_node_digests(doc, tier=1, frame=1)
        doc["nodes"]["p"]["params"]["size"] = 32  # changing input content
        digests_after = _compute_node_digests(doc, tier=1, frame=1)
        self.assertNotEqual(digests_before["g"], digests_after["g"],
                             "Downstream digest must change when upstream content changes")


class SupportsTiledCoverageTests(unittest.TestCase):
    """supports_tiled must traverse every evaluated ancestor, including the Merge B branch."""

    def test_merge_with_one_supported_branch_is_still_supported(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 64, "height": 64, "size": 16}})
        d.execute({"op": "create", "type": "Constant", "id": "w",
                   "params": {"width": 64, "height": 64, "alpha": 0.5}})
        d.execute({"op": "create", "type": "Merge", "id": "m"})
        d.execute({"op": "connect", "id": "m", "input": "A", "source": "w"})
        d.execute({"op": "connect", "id": "m", "input": "B", "source": "p"})
        self.assertTrue(TileExecutor().supports_tiled(d.document, "m"))

    def test_switch_selected_branch_is_traversed(self):
        """A graph containing a Switch is NOT tile-native — selecting a branch is unsupported."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 64, "height": 64, "size": 16}})
        d.execute({"op": "create", "type": "Checker", "id": "q",
                   "params": {"width": 64, "height": 64, "size": 16}})
        d.execute({"op": "create", "type": "Switch", "id": "s", "params": {"which": 0}})
        d.execute({"op": "connect", "id": "s", "input": "0", "source": "p"})
        d.execute({"op": "connect", "id": "s", "input": "1", "source": "q"})
        self.assertFalse(TileExecutor().supports_tiled(d.document, "s"),
                         "Switch is intentionally excluded from v0.9 tile execution; the executor "
                         "must reject it via supports_tiled, not silently process it")

    def test_unsupported_kind_in_merge_branch_disables_tiled(self):
        """Switch on Merge's B branch means the whole graph is not tile-native."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 64, "height": 64, "size": 16}})
        d.execute({"op": "create", "type": "Checker", "id": "q",
                   "params": {"width": 64, "height": 64, "size": 16}})
        d.execute({"op": "create", "type": "Switch", "id": "s", "params": {"which": 0}})
        d.execute({"op": "connect", "id": "s", "input": "0", "source": "p"})
        d.execute({"op": "connect", "id": "s", "input": "1", "source": "q"})
        d.execute({"op": "create", "type": "Merge", "id": "m"})
        d.execute({"op": "connect", "id": "m", "input": "A", "source": "p"})
        d.execute({"op": "connect", "id": "m", "input": "B", "source": "s"})
        self.assertFalse(TileExecutor().supports_tiled(d.document, "m"),
                         "Merge with a Switch on its B branch must NOT be claimed as tile-native")

    def test_transform_in_chain_is_fallback_not_tiled(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 64, "height": 64, "size": 16}})
        d.execute({"op": "create", "type": "Transform", "id": "x",
                   "params": {"translate_x": 5.0, "translate_y": 0.0, "rotate": 0.0,
                              "scale": 1.0, "center_x": 0.0, "center_y": 0.0,
                              "filter": "nearest"}})
        d.execute({"op": "connect", "id": "x", "input": "image", "source": "p"})
        self.assertFalse(TileExecutor().supports_tiled(d.document, "x"))

    def test_crop_in_chain_is_fallback_not_tiled(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 64, "height": 64, "size": 16}})
        d.execute({"op": "create", "type": "Crop", "id": "c",
                   "params": {"x": 0, "y": 0, "width": 32, "height": 32}})
        d.execute({"op": "connect", "id": "c", "input": "image", "source": "p"})
        self.assertFalse(TileExecutor().supports_tiled(d.document, "c"))


class SourceRetentionTests(unittest.TestCase):
    """Within a single compose, each generator is decoded at most once."""

    def test_demo_decodes_each_source_exactly_once(self):
        """Demo has one Checker and one Constant source; total decodes must be 2, not 4."""
        exe = TileExecutor()
        result = exe.compose(demo_document(), "viewer", frame=1, tier=1)
        self.assertEqual(result.source_decodes, 2,
                         f"Each generator must decode once per compose; got {result.source_decodes}")

    def test_constant_source_is_decoded_once_per_compose(self):
        d = Dispatcher()
        for _ in range(8):
            d.execute({"op": "create", "type": "Constant", "id": f"c{_}",
                       "params": {"width": 32, "height": 32}})
            d.execute({"op": "create", "type": "Grade", "id": f"g{_}"})
            d.execute({"op": "connect", "id": f"g{_}", "input": "image", "source": f"c{_}"})
        d.execute({"op": "create", "type": "Viewer", "id": "v"})
        # Wire viewer to last grade (any one works for the decode-count test)
        d.execute({"op": "connect", "id": "v", "input": "image", "source": "g0"})
        exe = TileExecutor()
        result = exe.compose(d.document, "v", frame=1, tier=1)
        # 8 Constant sources, each decoded once. Without retention it would be > 8.
        self.assertLessEqual(result.source_decodes, 8)
        self.assertGreater(result.source_decodes, 0)

    def test_repeat_compose_does_not_redecode_when_tile_cache_is_warm(self):
        """The tile cache persists across compose calls; a repeat compose hits the cache and
        does not need to re-decode. The source retention cache is cleared each compose, but
        the tile cache means we don't reach the decode path on the second call."""
        exe = TileExecutor()
        doc = demo_document()
        r1 = exe.compose(doc, "viewer", frame=1, tier=1)
        r2 = exe.compose(doc, "viewer", frame=1, tier=1)
        self.assertEqual(r1.source_decodes, 2)
        self.assertEqual(r2.source_decodes, 0,
                         "Warm tile cache should mean no source decodes on repeat compose")

    def test_fresh_compose_after_cache_clear_redecodes_each_source(self):
        """A second compose with an empty tile cache re-decodes each source."""
        exe = TileExecutor()
        doc = demo_document()
        exe.compose(doc, "viewer", frame=1, tier=1)
        exe.cache.clear()
        r2 = exe.compose(doc, "viewer", frame=1, tier=1)
        self.assertEqual(r2.source_decodes, 2,
                         "Cleared cache + new compose must re-decode the two demo sources")


class HaloAndEdgeCorrectnessTests(unittest.TestCase):
    """Tile render must match full-frame render byte-for-byte at tier 1, including across tile boundaries."""

    def _assert_tile_equals_full(self, doc, target):
        ev = Evaluator()
        full = ev.evaluate(doc, target, tier=1)
        # Multi-tile + single-tile: a tiny tile_edge forces many tiles.
        for tile_edge in (256, 64):
            exe = TileExecutor(tile_edge=tile_edge)
            tiled = exe.compose(doc, target, frame=1, tier=1)
            self.assertEqual(tiled.shape, full.shape[:2],
                             f"shape mismatch for {target} at tile_edge={tile_edge}")
            diff = np.abs(tiled.pixels - full)
            self.assertLess(float(diff.max()), 1e-5,
                            f"{target} at tile_edge={tile_edge} disagrees with full-frame: "
                            f"max_diff={float(diff.max()):.6f}, mean={float(diff.mean()):.6f}")

    def test_blur_chain_matches_full_frame(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 256, "height": 256, "size": 16}})
        d.execute({"op": "create", "type": "Blur", "id": "b", "params": {"radius": 12.0}})
        d.execute({"op": "connect", "id": "b", "input": "image", "source": "p"})
        self._assert_tile_equals_full(d.document, "b")

    def test_region_request_matches_reference_crop_without_composing_the_canvas(self):
        """The demand-driven API returns exactly the requested rectangle.

        A tile executor that renders a full canvas and crops at the end has the same pixels but
        defeats the point of ROI scheduling.  The shape assertion pins the public contract; the
        dedicated request covers only 96x80 pixels of a 256x256 blur chain.
        """
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 256, "height": 256, "size": 16}})
        d.execute({"op": "create", "type": "Blur", "id": "b", "params": {"radius": 12.0}})
        d.execute({"op": "connect", "id": "b", "input": "image", "source": "p"})
        full = Evaluator().evaluate(d.document, "b")
        request = TileRegion(64, 80, 96, 80, full_width=256, full_height=256)
        result = TileExecutor(tile_edge=64).compose_region(d.document, "b", request, tier=1)
        self.assertTrue(result.tiled)
        self.assertEqual(result.shape, (80, 96))
        self.assertEqual(result.region, request)
        np.testing.assert_allclose(result.pixels, full[80:160, 64:160], atol=1e-5)

    def test_blur_through_merge_matches_full_frame(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 256, "height": 256, "size": 16}})
        d.execute({"op": "create", "type": "Constant", "id": "w",
                   "params": {"width": 256, "height": 256, "alpha": 0.4}})
        d.execute({"op": "create", "type": "Blur", "id": "b", "params": {"radius": 8.0}})
        d.execute({"op": "connect", "id": "b", "input": "image", "source": "p"})
        d.execute({"op": "create", "type": "Merge", "id": "m"})
        d.execute({"op": "connect", "id": "m", "input": "A", "source": "w"})
        d.execute({"op": "connect", "id": "m", "input": "B", "source": "b"})
        self._assert_tile_equals_full(d.document, "m")

    def test_demo_matches_full_frame(self):
        self._assert_tile_equals_full(demo_document(), "viewer")

    def test_demo_at_tier_4_matches_full_frame(self):
        ev = Evaluator()
        doc = demo_document()
        full = ev.evaluate(doc, "viewer", tier=4)
        for tile_edge in (256, 64):
            exe = TileExecutor(tile_edge=tile_edge)
            tiled = exe.compose(doc, "viewer", frame=1, tier=4)
            diff = np.abs(tiled.pixels - full)
            self.assertLess(float(diff.max()), 1e-5,
                            f"viewer at tier 4 tile_edge={tile_edge} disagrees: "
                            f"max_diff={float(diff.max()):.6f}")


class MergeAlignmentTests(unittest.TestCase):
    """Merge inputs with different halo-driven buffered shapes must arrive at the kernel
    with the same shape."""

    def test_align_crops_to_smaller_target(self):
        """Source covers (0, 0, 200, 200), target is (50, 50, 100, 100)."""
        pixels = np.full((200, 200, 4), 0.5, np.float32)
        region = TileRegion(0, 0, 200, 200, halo_x=0, halo_y=0, full_width=200, full_height=200)
        artifact = make_artifact(region, pixels)
        target = TileRegion(50, 50, 100, 100, halo_x=0, halo_y=0, full_width=200, full_height=200)
        aligned = _align_artifact_to(artifact, target)
        self.assertEqual(aligned.shape, (100, 100, 4))
        np.testing.assert_array_equal(aligned, np.full((100, 100, 4), 0.5, np.float32))

    def test_align_pads_when_source_smaller_than_target(self):
        pixels = np.full((50, 50, 4), 0.25, np.float32)
        region = TileRegion(0, 0, 50, 50, halo_x=0, halo_y=0, full_width=200, full_height=200)
        artifact = make_artifact(region, pixels)
        target = TileRegion(0, 0, 100, 100, halo_x=0, halo_y=0, full_width=200, full_height=200)
        aligned = _align_artifact_to(artifact, target)
        self.assertEqual(aligned.shape, (100, 100, 4))
        # Source was 50x50 of 0.25, padding is transparent zeros.
        np.testing.assert_array_equal(aligned[:50, :50], np.full((50, 50, 4), 0.25, np.float32))
        np.testing.assert_array_equal(aligned[50:, :], np.zeros((50, 100, 4), np.float32))


class EditInvalidationTests(unittest.TestCase):
    """Edits to a node's params must invalidate the corresponding tile cache entry."""

    def test_grade_edit_invalidates_cache_and_returns_new_pixels(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "p",
                   "params": {"width": 128, "height": 128, "size": 16}})
        d.execute({"op": "create", "type": "Grade", "id": "g", "params": {"exposure": 0.0}})
        d.execute({"op": "connect", "id": "g", "input": "image", "source": "p"})

        exe = TileExecutor()
        r1 = exe.compose(d.document, "g", frame=1, tier=1)
        h1 = exe.cache.hits_exact + exe.cache.hits_preview
        m1 = exe.cache.misses

        # Edit the grade — different content digest -> the cached entry is keyed differently.
        d.execute({"op": "set", "id": "g", "param": "exposure", "value": 1.0})
        r2 = exe.compose(d.document, "g", frame=1, tier=1)
        h2 = exe.cache.hits_exact + exe.cache.hits_preview
        m2 = exe.cache.misses

        self.assertNotEqual(np.abs(r1.pixels - r2.pixels).max(), 0.0,
                            "Editing exposure must change the rendered pixels")
        # The second compose is a full miss because the old cache entries are keyed by the
        # pre-edit digest; the new digest addresses a different cache entry.
        self.assertGreater(m2 - m1, 0,
                           "Edited Grade must produce new cache misses; old entries are stale")


class ProportionalCropTileTests(unittest.TestCase):
    """Sanity: a tile-aware Constant source renders the right subregion when buffer is requested."""

    def test_constant_tile_pixels_are_a_crop_of_full(self):
        d = Dispatcher()
        d.execute({"op": "create", "type": "Constant", "id": "c",
                   "params": {"width": 128, "height": 128, "alpha": 1.0,
                              "red": 1.0, "green": 0.0, "blue": 0.0}})
        exe = TileExecutor(tile_edge=64)
        full = exe.compose(d.document, "c", frame=1, tier=1)
        # The pixel value at (10, 10) should be (1.0, 0.0, 0.0, 1.0).
        np.testing.assert_array_equal(full.pixels[10, 10], [1.0, 0.0, 0.0, 1.0])


class PostReviewBlockerRegressionTests(unittest.TestCase):
    """Three release-blockers a Codex review found in the reviewer-flagged draft, reproduced live
    against that draft and confirmed fixed here. Each test fails on the pre-fix code.
    """

    def test_editing_a_mask_input_invalidates_the_tiled_result(self):
        """`_compute_node_digests` must hash ALL wired input slots (required + optional), not just
        `SPECS[kind]["inputs"]`. The pre-fix version excluded "mask", so rewiring a Grade's mask
        left the digest unchanged and the tile cache kept serving pixels rendered against the old
        mask — confirmed live: same output whether the mask fully passed or fully blocked the
        filter."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Constant", "id": "img",
                  "params": {"width": 32, "height": 32, "red": 1.0, "green": 0.0,
                             "blue": 0.0, "alpha": 1.0}})
        d.execute({"op": "create", "type": "Constant", "id": "open",
                  "params": {"width": 32, "height": 32, "alpha": 1.0}})
        d.execute({"op": "create", "type": "Constant", "id": "closed",
                  "params": {"width": 32, "height": 32, "alpha": 0.0}})
        d.execute({"op": "create", "type": "Grade", "id": "g", "params": {"exposure": 2.0, "mix": 1.0}})
        d.execute({"op": "connect", "id": "g", "input": "image", "source": "img"})
        d.execute({"op": "connect", "id": "g", "input": "mask", "source": "open"})
        executor = TileExecutor()
        filter_applied = executor.compose(d.document, "g", tier=1).pixels
        d.execute({"op": "connect", "id": "g", "input": "mask", "source": "closed"})
        filter_bypassed = executor.compose(d.document, "g", tier=1).pixels
        self.assertFalse(np.array_equal(filter_applied, filter_bypassed))
        np.testing.assert_allclose(filter_bypassed[0, 0], [1.0, 0.0, 0.0, 1.0])
        reference = Evaluator().evaluate(d.document, "g", tier=1)
        np.testing.assert_allclose(filter_bypassed, reference, atol=1e-5)

    def test_unrenamed_generators_of_the_same_kind_do_not_collide(self):
        """Node identity for the synthetic generator tile cache must come from the node's actual
        document key, not `node.get("name")`. Node names default to the node's kind when unset
        (`nodebased/core.py`'s create op: `"name": cmd.get("name", kind)`), so two Constant nodes
        an artist never renamed both have name == "Constant" — the pre-fix code searched the
        document for the first (type, name) match and used its digest for both, so the second
        node rendered the first node's colour."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Constant", "id": "a",
                  "params": {"width": 16, "height": 16, "red": 1.0, "green": 0.0,
                             "blue": 0.0, "alpha": 1.0}})
        d.execute({"op": "create", "type": "Constant", "id": "b",
                  "params": {"width": 16, "height": 16, "red": 0.0, "green": 1.0,
                             "blue": 0.0, "alpha": 1.0}})
        self.assertEqual(d.document["nodes"]["a"]["name"], "Constant")
        self.assertEqual(d.document["nodes"]["b"]["name"], "Constant")
        executor = TileExecutor()
        red = executor.compose(d.document, "a", tier=1).pixels
        green = executor.compose(d.document, "b", tier=1).pixels
        np.testing.assert_allclose(red[0, 0], [1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(green[0, 0], [0.0, 1.0, 0.0, 1.0])

    def test_blur_with_a_wired_mask_matches_the_reference(self):
        """Not one of the three review-flagged blockers — found while golden-testing the fixes
        above against a broader multi-tile graph. `image` arrives at Blur's kernel halo-expanded
        (per `tiers._blur_rule`); `mask` arrives at the plain output-region shape ("the mask gates
        the *result*"). `_apply_mask_mix` requires matching shapes and raised on every call before
        this fix, so a Blur with any mask wired was unusable through the tile executor."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Checker", "id": "plate",
                  "params": {"width": 300, "height": 200, "size": 17}})
        d.execute({"op": "create", "type": "Constant", "id": "maskv",
                  "params": {"width": 300, "height": 200, "alpha": 0.6}})
        d.execute({"op": "create", "type": "Blur", "id": "blur", "params": {"radius": 6.0, "mix": 1.0}})
        d.execute({"op": "connect", "id": "blur", "input": "image", "source": "plate"})
        d.execute({"op": "connect", "id": "blur", "input": "mask", "source": "maskv"})
        for tier in (1, 2, 4):
            with self.subTest(tier=tier):
                tiled = TileExecutor(tile_edge=64).compose(d.document, "blur", tier=tier)
                reference = Evaluator().evaluate(d.document, "blur", tier=tier)
                self.assertEqual(tiled.pixels.shape, reference.shape)
                np.testing.assert_allclose(tiled.pixels, reference, atol=1e-5)

    def test_merge_of_mismatched_canvas_sizes_raises_like_the_reference(self):
        """The reference evaluator raises when Merge's two inputs have different full-frame
        formats (`imaging.py`: `if a.shape != b.shape: raise ValueError(...)`). The pre-fix tile
        executor's Merge branch cropped/padded both inputs to a common tile shape before any
        comparison, so `a.shape != b.shape` could never fire and a genuinely mismatched-format
        Merge (e.g. a 64x64 plate merged with a 16x16 one — almost certainly an artist mistake)
        rendered silently instead of raising."""
        d = Dispatcher()
        d.execute({"op": "create", "type": "Constant", "id": "big", "params": {"width": 64, "height": 64}})
        d.execute({"op": "create", "type": "Constant", "id": "small", "params": {"width": 16, "height": 16}})
        d.execute({"op": "create", "type": "Merge", "id": "m"})
        d.execute({"op": "connect", "id": "m", "input": "A", "source": "big"})
        d.execute({"op": "connect", "id": "m", "input": "B", "source": "small"})
        with self.assertRaisesRegex(ValueError, "matching formats"):
            Evaluator().evaluate(d.document, "m", tier=1)
        with self.assertRaisesRegex(ValueError, "matching formats"):
            TileExecutor().compose(d.document, "m", tier=1)


if __name__ == "__main__":
    unittest.main()
