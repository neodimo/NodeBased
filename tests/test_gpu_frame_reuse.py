"""The GPU path tracer keeps what does not change between frames of one scene on the device (step R2).

A second render of an unchanged scene reuses its pipelines, its scene buffers, its image accumulator, its bind groups and
its readback buffers (counted by `gpupathtrace.reuse_counters`), a changed knob or scene still changes the image, and the
cheaper ways of finishing a frame (several passes in one submission, the divide on the card) give the image the plain ones do."""
import threading
import unittest
from unittest import mock

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace as pt, scene3d as s
from nodebased.cancellation import Cancelled
from tests.test_3d_adaptive_sampling import SHADOW_CAMERA, POINT, shadow_scene
from tests.test_3d_pathtrace import box

SIZE = (48, 32)
LEAN = {"skip_noise_maps": True}


def settings(**kw):
    return pt.PathSettings(**{"sampling": "fixed", "samples": 12, "max_bounces": 2, "seed": 3, **kw})


def render(scene=None, camera=SHADOW_CAMERA, ambient=0.0, output="rgba", stats=None, background=(0, 0, 0, 0), progress=None,
           cancel=None, **kw):
    return pt.render(scene or shadow_scene(), camera, SIZE[0], SIZE[1], background, ambient, output, settings(**kw),
                     stats=stats if stats is not None else dict(LEAN), backend="gpu", progress=progress, cancel=cancel)


def counted(function):
    before = gpupathtrace.reuse_counters()
    result = function()
    after = gpupathtrace.reuse_counters()
    return result, {key: after[key] - before[key] for key in after}


def cold():
    gpupathtrace.release_frame_cache()


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class FrameReuseTests(unittest.TestCase):
    def setUp(self):
        cold()
        self.addCleanup(cold)

    def test_the_second_render_of_an_unchanged_scene_reuses_everything_static(self):
        first, built = counted(render)
        second, reused = counted(render)
        self.assertGreaterEqual(built["scene_uploads"], 1)
        self.assertEqual(built["scene_reuses"], 0)
        self.assertEqual((built["accum_buffers"], built["accum_reuses"]), (1, 0))
        self.assertGreaterEqual(built["bind_groups"], 2)       # one per dispatch slot in use and the resolve pass
        self.assertEqual(reused["scene_uploads"], 0)
        self.assertEqual(reused["scene_reuses"], 1)
        self.assertEqual((reused["accum_buffers"], reused["accum_reuses"]), (0, 1))
        for name in ("pipelines", "resolve_pipelines", "bind_groups", "uniform_buffers", "staging_buffers"):
            self.assertEqual(reused[name], 0, name)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(first[..., :3].mean()), 0)

    def test_a_changed_knob_changes_the_image_and_keeps_the_static_buffers(self):
        base = render()
        for name, change in (("seed", dict(seed=9)), ("samples", dict(samples=20)), ("ambient", dict(ambient=0.4)),
                             ("bounces", dict(max_bounces=1))):
            with self.subTest(name):
                warm, counts = counted(lambda: render(**change))
                self.assertEqual((counts["scene_uploads"], counts["scene_reuses"]), (0, 1))
                self.assertFalse(np.array_equal(warm, base), name)
                cold()
                np.testing.assert_array_equal(warm, render(**change))      # and it is the image a fresh start gives
                render()                                                    # put the base scene back for the next knob
        moved = s.Camera(s.Transform3D(position=s.Vec3(1, 4, 6)), s.Vec3(0, -1, 0), 45.0)
        warm, counts = counted(lambda: render(camera=moved))
        self.assertEqual(counts["scene_uploads"], 0)
        self.assertFalse(np.array_equal(warm, base))

    def test_a_changed_scene_uploads_new_buffers_and_renders_the_new_scene(self):
        render()
        ground = shadow_scene().geometries[0]
        other = s.Scene((ground, box((1, 1, 1), (.5, .5, .5, 1), (1.2, -.5, 0))), (POINT,))
        changed, counts = counted(lambda: render(other))
        self.assertEqual((counts["scene_uploads"], counts["scene_reuses"]), (1, 0))
        cold()
        np.testing.assert_array_equal(changed, render(other))
        self.assertFalse(np.array_equal(changed, render(shadow_scene())))

    def test_a_new_image_size_makes_a_new_accumulator(self):
        render()
        _, counts = counted(lambda: pt.render(shadow_scene(), SHADOW_CAMERA, 40, 24, (0, 0, 0, 0), 0.0, "rgba", settings(),
                                              stats=dict(LEAN), backend="gpu"))
        self.assertEqual((counts["accum_buffers"], counts["accum_reuses"], counts["scene_reuses"]), (1, 0, 1))


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class FinishOnTheCardTests(unittest.TestCase):
    def setUp(self):
        cold()
        self.addCleanup(cold)

    def test_the_image_finished_on_the_card_matches_the_host_conversion(self):
        for output in ("rgba", "albedo", "depth", "normals"):
            for background in ((0, 0, 0, 0), (0.2, 0.4, 0.1, 0.5)):
                with self.subTest(output=output, background=background):
                    lean_stats, rich_stats = dict(LEAN), {}
                    lean = render(output=output, stats=lean_stats, background=background)
                    rich = render(output=output, stats=rich_stats, background=background)
                    np.testing.assert_allclose(lean, rich, rtol=2e-6, atol=1e-7)
                    np.testing.assert_array_equal(lean_stats["samples"], rich_stats["samples"])
                    np.testing.assert_array_equal(lean_stats["converged"], rich_stats["converged"])
                    self.assertEqual(lean_stats["readbacks"]["image"], 1)

    def test_adaptive_sampling_finishes_on_the_card_with_the_same_counts_and_flags(self):
        adaptive = dict(sampling="adaptive", noise_threshold=0.02, min_samples=8, max_samples=64, adaptive_pass_size=8)
        lean_stats, rich_stats = dict(LEAN), {}
        lean = render(stats=lean_stats, **adaptive)
        rich = render(stats=rich_stats, **adaptive)
        np.testing.assert_allclose(lean, rich, rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(lean_stats["samples"], rich_stats["samples"])
        np.testing.assert_array_equal(lean_stats["converged"], rich_stats["converged"])
        self.assertEqual(lean_stats["passes"], rich_stats["passes"])
        self.assertGreater(int(lean_stats["samples"].max()), int(lean_stats["samples"].min()))      # the mask did its work
        self.assertNotIn("variance", lean_stats)
        self.assertIn("variance", rich_stats)

    def test_a_caller_that_asks_for_the_noise_maps_gets_them_and_the_others_do_not(self):
        lean, rich = dict(LEAN), {}
        render(stats=lean)
        render(stats=rich)
        self.assertNotIn("variance", lean)
        self.assertNotIn("noise", lean)
        self.assertEqual(rich["variance"].shape, (SIZE[1], SIZE[0]))
        self.assertEqual(rich["noise"].shape, (SIZE[1], SIZE[0]))
        self.assertGreater(float(rich["variance"].max()), 0.0)
        self.assertEqual(lean["samples"].shape, (SIZE[1], SIZE[0]))
        self.assertTrue(np.all(lean["samples"] == 12))
        image = pt.render(shadow_scene(), SHADOW_CAMERA, SIZE[0], SIZE[1], (0, 0, 0, 0), 0.0, "rgba", settings(), backend="gpu")
        self.assertEqual(image.shape, (SIZE[1], SIZE[0], 4))                    # and no stats at all is the same lean path

    def test_the_legacy_noise_test_reads_each_pass_and_still_works_with_the_card_finish(self):
        stats = dict(LEAN)
        image = render(stats=stats, samples=16, pass_samples=4, noise_threshold=0.02)
        self.assertEqual(stats["readbacks"]["mask"], stats["passes"])
        self.assertGreater(float(image[..., :3].mean()), 0)
        self.assertEqual(stats["samples"].shape, (SIZE[1], SIZE[0]))


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class BatchingTests(unittest.TestCase):
    def setUp(self):
        cold()
        self.addCleanup(cold)

    def test_passes_share_submissions_and_the_image_is_the_same(self):
        single = dict(LEAN)
        with mock.patch.object(gpupathtrace, "GPU_PATHS_PER_SUBMISSION", SIZE[0] * SIZE[1]), \
             mock.patch.object(gpupathtrace, "BAND_TARGET_SECONDS", 0.0):
            one_by_one = render(samples=24, stats=single)
        batched_stats = dict(LEAN)
        batched = render(samples=24, stats=batched_stats)
        self.assertEqual(single["readbacks"]["waits"], 24)                      # a wait per pass
        self.assertLess(batched_stats["readbacks"]["waits"], 24)
        self.assertGreaterEqual(batched_stats["readbacks"]["waits"], 1)
        np.testing.assert_array_equal(one_by_one, batched)                      # the sums do not depend on the grouping

    def test_progress_reports_whole_passes_and_ends_at_the_last_sample(self):
        seen = []
        render(samples=16, pass_samples=2, progress=lambda stage, fraction, info: seen.append(info["samples"]))
        self.assertTrue(seen)
        self.assertEqual(seen, sorted(seen))
        self.assertTrue(all(n % 2 == 0 for n in seen), seen)
        self.assertEqual(seen[-1], 16)

    def test_a_time_limit_stops_between_passes_with_every_pixel_at_the_same_count(self):
        stats = dict(LEAN)
        with mock.patch.object(gpupathtrace, "GPU_PATHS_PER_SUBMISSION", SIZE[0] * SIZE[1]):
            image = render(samples=4000, time_limit=0.01, stats=stats)
        counts = np.unique(stats["samples"])
        self.assertEqual(len(counts), 1)
        self.assertLess(int(counts[0]), 4000)
        again = render(samples=int(counts[0]))
        np.testing.assert_allclose(image, again, rtol=1e-6, atol=1e-7)

    def test_a_cancelled_render_leaves_the_next_one_correct(self):
        expected = render()
        event = threading.Event()

        def stop(stage, fraction, info):
            event.set()
        with self.assertRaises(Cancelled):
            render(samples=400, pass_samples=1, progress=stop, cancel=event)
        np.testing.assert_array_equal(render(), expected)

    def test_renders_from_two_threads_do_not_share_an_accumulator(self):
        expected = render()
        results, errors = [], []

        def work():
            try:
                for _ in range(3):
                    results.append(render())
            except Exception as exc:       # noqa: BLE001 - reported below
                errors.append(exc)
        threads = [threading.Thread(target=work) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 6)
        for image in results:
            np.testing.assert_array_equal(image, expected)


if __name__ == "__main__":
    unittest.main()
