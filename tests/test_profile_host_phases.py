"""tools/profile_host_phases.py (lane 4 Rendering, step R2): one small case profiled on the default adapter gives the host laps
of a frame by phase, the host cost is the laps other than the card's own, and the frames after the first reuse the scene,
accumulator and bind groups instead of building them."""
import unittest

from nodebased import gpu3d, gpupathtrace
from tests.test_benchmark_portable_render import RestoresGlobals
from tools import profile_host_phases as profile


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class ProfileTests(RestoresGlobals):
    def setUp(self):
        super().setUp()
        gpupathtrace.release_frame_cache()
        self.addCleanup(gpupathtrace.release_frame_cache)

    def test_a_profile_lists_the_phases_and_the_host_cost_leaves_out_the_card(self):
        line = profile.run("Z1-fixed64", "default", (64, 36), timed=3)
        phases = line["phases_ms"]
        for name in ("scene build", "pack", "upload", "encode", "dispatch", "final readback"):
            self.assertIn(name, phases)
        host = sum(v for k, v in phases.items() if k != "dispatch")
        self.assertAlmostEqual(line["host_ms"], host, delta=0.5 + 0.05 * host)     # medians per phase, so only about the sum
        self.assertLess(line["host_ms"], line["frame_ms"] + 1.0)
        self.assertEqual(line["case"], "Z1-fixed64")
        self.assertGreaterEqual(line["waits"], 1)

    def test_the_timed_frames_reuse_what_the_first_ones_built(self):
        before = gpupathtrace.reuse_counters()
        line = profile.run("Z1-fixed64", "default", (64, 36), timed=4)
        after = gpupathtrace.reuse_counters()
        built = {k: after[k] - before[k] for k in after}
        self.assertEqual(built["scene_uploads"], 1)
        self.assertEqual(built["accum_buffers"], 1)
        self.assertGreaterEqual(built["scene_reuses"], 5)        # 2 warm-up and 4 timed frames, the first builds
        self.assertLessEqual(built["pipelines"], 1)             # one compile at most (the process may have it already)
        self.assertEqual(built["bind_groups"], line["build"]["bind_groups"] - before["bind_groups"])


if __name__ == "__main__":
    unittest.main()
