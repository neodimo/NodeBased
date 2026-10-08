"""The portable final-render benchmark (lane 4 Rendering, step P1): a case's result survives an interrupted run, a missing
adapter is recorded as unavailable, the image-quality comparison is deterministic and ranks a noisier image lower, and one
small case runs end to end against the CPU path tracer."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import gpupathtrace, gpu3d, ptvolume
from tools import benchmark_portable_render as b


class RestoresGlobals(unittest.TestCase):
    """run_case forces the adapter and the majorant for its process; the test runner shares one process."""

    def setUp(self):
        saved = (gpu3d._state, ptvolume.ENABLE_SKIP, gpupathtrace.ENABLE_VOLUME_SKIP, ptvolume.MAJORANT_RATIO)

        def restore():
            gpu3d._state, ptvolume.ENABLE_SKIP, gpupathtrace.ENABLE_VOLUME_SKIP, ptvolume.MAJORANT_RATIO = saved
        self.addCleanup(restore)


class Comparison(unittest.TestCase):
    def test_identical_images_have_no_error_and_noise_lowers_psnr_and_repeats_exactly(self):
        rng = np.random.default_rng(3)
        reference = rng.uniform(0.05, 0.9, (12, 16, 4)).astype(np.float32)
        same = b.compare_images(reference, reference)
        self.assertEqual((same["mean_abs"], same["max_abs"], same["psnr_db"]), (0.0, 0.0, 200.0))
        small = reference + rng.normal(0, 0.005, reference.shape).astype(np.float32)
        large = reference + rng.normal(0, 0.05, reference.shape).astype(np.float32)
        a, c = b.compare_images(small, reference), b.compare_images(large, reference)
        self.assertGreater(a["psnr_db"], c["psnr_db"] + 10)
        self.assertGreater(c["max_abs"], a["max_abs"])
        self.assertEqual(b.compare_images(small, reference), a)

    def test_psnr_is_measured_on_the_clipped_srgb_image_and_alpha_is_ignored(self):
        reference = np.full((4, 4, 4), 0.5, np.float32)
        brighter = reference.copy()
        brighter[..., 3] = 0.0
        self.assertEqual(b.compare_images(brighter, reference)["max_abs"], 0.0)
        over = np.full((4, 4, 4), 7.0, np.float32)
        self.assertEqual(b.compare_images(over, np.ones((4, 4, 4), np.float32))["max_abs"], 0.0)

    def test_shapes_must_match(self):
        with self.assertRaises(ValueError):
            b.compare_images(np.zeros((4, 4, 4)), np.zeros((4, 5, 4)))


class Persistence(RestoresGlobals):
    def test_each_case_is_on_disk_the_moment_it_is_saved_and_a_later_failure_keeps_it(self):
        with tempfile.TemporaryDirectory() as out:
            first = b._record("Z1-fixed64", "cpu", (8, 8), "ok", median_s=0.5)
            b.save_case(out, first)
            with self.assertRaises(TypeError):         # the next case dies while serialising: not JSON
                b.save_case(out, b._record("Y1-fixed64", "cpu", (8, 8), "ok", median_s=object()))
            loaded = b.load_cases(out)
            self.assertEqual([r["case"] for r in loaded], ["Z1-fixed64"])
            self.assertEqual(loaded[0]["median_s"], 0.5)
            self.assertEqual(sorted(p.name for p in Path(out).iterdir()), ["cpu__Z1-fixed64.json"])    # no temp file left

    def test_a_record_names_the_scene_settings_seed_size_and_frame_counts(self):
        record = b._record("smoke-grid", "integrated", (64, 36), "ok")
        for key in ("scene", "settings", "seed", "size", "warmup", "timed", "commit", "adapter_request"):
            self.assertIn(key, record)
        self.assertEqual((record["size"], record["seed"], record["warmup"], record["timed"]), ([64, 36], 4, b.WARMUP, b.TIMED))
        json.dumps(record)

    def test_a_missing_adapter_is_recorded_as_unavailable_and_shows_in_the_report(self):
        with tempfile.TemporaryDirectory() as out:
            path = b.run_case("Z1-fixed64", "no-such-adapter", (8, 8), out)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "unavailable")
            self.assertIn("no-such-adapter", record["reason"])
            self.assertIn("unavailable", b.report(out))

    def test_the_matrix_lists_the_x1_y1_z1_and_both_smoke_cases(self):
        self.assertEqual(set(b.CASES), {f"{s}-{m}" for s in ("X1", "Y1", "Z1") for m in ("fixed64", "adaptive0.003")}
                         | {"smoke-grid", "smoke-box"})


class EndToEnd(RestoresGlobals):
    def test_a_small_case_on_the_software_adapter_matches_the_cpu_path_tracer(self):
        with tempfile.TemporaryDirectory() as out:
            record = json.loads(b.run_case("Z1-fixed64", "cpu", (32, 18), out).read_text())
            if record["status"] == "unavailable":
                self.skipTest(record["reason"])
            self.assertEqual(record["status"], "ok", record)
            self.assertEqual(len(record["times_s"]), b.TIMED)
            self.assertGreater(record["quality"]["psnr_db"], 40)
            self.assertEqual(record["renderer"]["samples"]["mean"], 64.0)
            self.assertTrue((Path(out) / "reference" / "Z1-fixed64_32x18.npy").exists())


if __name__ == "__main__":
    unittest.main()
