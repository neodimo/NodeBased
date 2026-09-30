"""Memory ceiling gate (docs/VISION.md's M2 gate, docs/BENCHMARKS-v0.33-m2.md's "Memory ceiling"
section): the evaluator's raster cache and the tile executor's tile cache, tied together by
`cachetier.SharedMemoryBudget` (`TileExecutor(memory_budget=...)`), stay under a combined byte
budget on an 8K graph, evicting by least recent use, and the pixels a frame produces after its
predecessor has been evicted are still correct -- an eviction must only cost a recompute, never
corrupt one.

Slow: four 7680x4320 frames through the ten-node graph `tools.benchmark_memory_throughput` builds
(Read, Grade, Transform, Merge, Blur, Roto, Tracker, ColorCorrect, Reformat, Write) is real,
non-trivial pixel work at 8K.
"""
import tempfile
import unittest

import numpy as np

from nodebased import cachetier
from nodebased.tileexec import TileExecutor
from tools.benchmark_memory_throughput import build_document, write_sequence, _full_canvas_region

WIDTH, HEIGHT = 7680, 4320
# One 8K RGBA float32 raster alone is ~506 MB (cachetier.frame_bytes(7680, 4320)). The evaluator's
# own budget contract keeps an oversized *single* result resident rather than refusing it
# (`Evaluator._store`'s docstring: this is what fixed the 8K cache going silently inert at exactly
# the resolution where recompute hurts most) -- so a budget smaller than one frame cannot be held
# to at all, and is not the case this gate is for. 1.3x one frame holds exactly one 8K result
# resident and forces a real eviction on every subsequent frame, which is the case that matters.
ONE_FRAME_BYTES = cachetier.frame_bytes(WIDTH, HEIGHT)
BUDGET_BYTES = int(ONE_FRAME_BYTES * 1.3)


class MemoryCeilingGateTests(unittest.TestCase):
    def test_8k_playback_stays_under_combined_budget_with_correct_eviction_recompute(self):
        with tempfile.TemporaryDirectory() as seq_dir:
            sequence_path = write_sequence(seq_dir, WIDTH, HEIGHT, 4)
            d, target = build_document(sequence_path, WIDTH, HEIGHT)
            exe = TileExecutor(memory_budget=BUDGET_BYTES)
            region = _full_canvas_region(exe, d.document, target)

            first_pixels = None
            for frame in range(1, 5):
                result = exe.compose_region(d.document, target, region, frame=frame, tier=1)
                if frame == 1:
                    first_pixels = np.array(result.pixels, copy=True)
                total = exe.shared_budget.bytes_total()
                self.assertLessEqual(
                    total, BUDGET_BYTES * 1.1,
                    f"combined cache bytes {total} exceeded the {BUDGET_BYTES}-byte budget plus "
                    f"10 percent after frame {frame} (cache.bytes={exe.cache.bytes}, "
                    f"evaluator.bytes={exe.evaluator.bytes})")

            # Frame 1's result is long evicted by now (budget holds well under one frame). A
            # recompute of it must still be pixel-identical to what was produced live, never a
            # stale or corrupted read of an evicted slot.
            recomputed = exe.compose_region(d.document, target, region, frame=1, tier=1)
            np.testing.assert_array_equal(recomputed.pixels, first_pixels)


if __name__ == "__main__":
    unittest.main()
