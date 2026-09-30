"""M2 gate (docs/VISION.md, docs/BENCHMARKS-v0.33-m2.md's "Throughput" section): the 4K tile-path
playback of the ten-node graph (Read, Grade, Transform, Merge, Blur, Roto, Tracker, ColorCorrect,
Reformat, Write) reaches a frames-per-second budget set from the measurement in that doc plus 30
percent headroom.

Slow (four real 3840x2160 frames through the tile path, warm on-disk cache) and skipped on a
software/no-GPU adapter, the same convention `tests/test_m2_latency_gate.py` already uses: these
are wall-clock budgets calibrated on the workstation named in docs/BENCHMARKS-v0.33-m2.md, not a
CI runner's own number.
"""
import tempfile
import unittest

from nodebased.tileexec import TileExecutor
from tools.benchmark_memory_throughput import build_document, write_sequence, _full_canvas_region, _playback

WIDTH, HEIGHT = 3840, 2160
FRAMES = 4
# Measured warm 4K tile-path fps in docs/BENCHMARKS-v0.33-m2.md's "Throughput" section: 0.29 fps.
# Budget is that measurement minus 30 percent headroom (a throughput floor gets *looser* the more
# headroom it carries, the mirror image of the latency gate's own ceiling-plus-headroom).
FPS_BUDGET = 0.20


def _on_software_or_missing_adapter():
    try:
        from nodebased import gpu3d
        info = gpu3d._state()["info"]
        return str(info.get("adapter_type", "")).lower() == "cpu"
    except Exception:
        return True


@unittest.skipIf(_on_software_or_missing_adapter(),
                 "software/no-GPU adapter: not the workstation this wall-clock budget is calibrated on")
class M2ThroughputGateTests(unittest.TestCase):
    def test_4k_tile_path_playback_meets_fps_budget(self):
        with tempfile.TemporaryDirectory() as seq_dir:
            sequence_path = write_sequence(seq_dir, WIDTH, HEIGHT, FRAMES)
            d, target = build_document(sequence_path, WIDTH, HEIGHT)
            exe = TileExecutor()
            region = _full_canvas_region(exe, d.document, target)

            def evaluate_frame(frame):
                exe.compose_region(d.document, target, region, frame=frame, tier=1)

            # One warm-up pass so the measured loop is the steady-state cost the doc's own
            # numbers are, not a cold first frame's extra one-time overhead.
            evaluate_frame(1)
            fps, _ = _playback(evaluate_frame, FRAMES)
            self.assertGreaterEqual(
                fps, FPS_BUDGET,
                f"4K tile-path playback ran at {fps:.3f} fps, under the {FPS_BUDGET} fps budget "
                f"(docs/BENCHMARKS-v0.33-m2.md's Throughput section)")


if __name__ == "__main__":
    unittest.main()
