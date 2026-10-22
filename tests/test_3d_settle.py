"""Lane 4, plan "Render mode responsiveness", step C: Render mode settles in seconds by sampling only where the picture is noisy.

Render mode used to trace every refinement step from nothing (8, 16, 32, 64 samples, each a new render) and filter each picture on the
processor. Now the full-size steps continue one accumulation: the pixels whose noise estimate is under the threshold stay as they are and the
rest take their next samples (`gpupathtrace` and `pathtrace` keep and take back their running sums through ``stats["keep_accumulation"]``
and ``stats["resume"]``), and the filter runs on the graphics card (`gpudenoise`). The tests assert pixels and sample counts:

* a continued render is the render that was never stopped (image, counts and done flags equal), on the CPU reference and on whatever
  adapter the process has;
* a flat scene finishes in fewer samples than a noisy one, the result for a fixed seed on the CPU path is the same every time, and a
  cancelled pass or a new view restarts the noise estimate;
* the card filter agrees with the numpy filter, refuses what it cannot take, honours a cancel, and the viewport falls back to numpy;
* the viewport note names the share of the picture still refining, and the debug tint marks exactly those pixels.
"""
import os
import threading
import unittest
from dataclasses import replace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpu3d, gpudenoise, pathtrace as pt, progressivejob, progressiverender as P, ptdenoise
from nodebased.cancellation import Cancelled
from nodebased.progressivejob import ProgressiveJob, Request
from tests.test_3d_adaptive_sampling import FLAT_CAMERA, SHADOW_CAMERA, flat_scene, shadow_scene
from tests.test_3d_progressive_render import _SKIP

SIZE = (40, 30)
BACKGROUND = (0.0, 0.0, 0.0, 1.0)


def settings(cap, threshold=0.02, seed=1):
    return pt.PathSettings(samples=cap, max_bounces=2, seed=seed, sampling="adaptive", noise_threshold=threshold, min_samples=8,
                           max_samples=cap, adaptive_pass_size=8)


def trace(scene, camera, cap, backend, **stats_in):
    stats = {"skip_noise_maps": True, **stats_in}
    image = pt.render(scene, camera, SIZE[0], SIZE[1], BACKGROUND, 0.1, "rgba", settings(cap), backend=backend, stats=stats)
    return image, stats


def _gpu_available():
    return gpu3d.available()


class ContinuedRenderTests(unittest.TestCase):
    def check_equal(self, backend):
        scene = shadow_scene()
        whole, whole_stats = trace(scene, SHADOW_CAMERA, 32, backend)
        _, first = trace(scene, SHADOW_CAMERA, 16, backend, keep_accumulation=True)
        continued, second = trace(scene, SHADOW_CAMERA, 32, backend, keep_accumulation=True, resume=first["accumulation"])
        np.testing.assert_array_equal(continued, whole)
        np.testing.assert_array_equal(second["samples"], whole_stats["samples"])
        np.testing.assert_array_equal(second["converged"], whole_stats["converged"])
        self.assertEqual(second["backend"], whole_stats["backend"])
        return first, second

    def test_a_continued_cpu_render_is_the_one_that_was_never_stopped(self):
        first, second = self.check_equal("cpu")
        self.assertEqual(first["accumulation"]["kind"], "cpu")
        self.assertGreater(second["samples"].max(), first["samples"].max())     # the second call really took more samples

    @unittest.skipUnless(_gpu_available(), "no graphics adapter")
    def test_a_continued_gpu_render_is_the_one_that_was_never_stopped(self):
        first, second = self.check_equal("auto")
        self.assertEqual(first["accumulation"]["kind"], "gpu")
        self.assertGreater(second["samples"].max(), first["samples"].max())

    def test_pixels_under_the_threshold_take_no_more_samples_when_the_render_is_continued(self):
        scene = shadow_scene()
        _, first = trace(scene, SHADOW_CAMERA, 16, "cpu", keep_accumulation=True)
        _, second = trace(scene, SHADOW_CAMERA, 64, "cpu", keep_accumulation=True, resume=first["accumulation"])
        settled = first["converged"]
        self.assertTrue(settled.any() and not settled.all(), "the scene needs both quiet and noisy pixels")
        np.testing.assert_array_equal(second["samples"][settled], first["samples"][settled])
        self.assertTrue((second["samples"][~settled] > first["samples"][~settled]).all())

    def test_an_accumulation_of_another_size_or_tracer_is_ignored(self):
        scene = shadow_scene()
        _, first = trace(scene, SHADOW_CAMERA, 16, "cpu", keep_accumulation=True)
        fresh, fresh_stats = trace(scene, SHADOW_CAMERA, 32, "cpu")
        wrong_size = dict(first["accumulation"], pixels=first["accumulation"]["pixels"] + 1)
        wrong_kind = dict(first["accumulation"], kind="gpu")
        for resume in (wrong_size, wrong_kind):
            image, stats = trace(scene, SHADOW_CAMERA, 32, "cpu", resume=resume)
            np.testing.assert_array_equal(image, fresh)
            np.testing.assert_array_equal(stats["samples"], fresh_stats["samples"])


class SettleTests(unittest.TestCase):
    def settle(self, scene, camera, backend="cpu", rounds=4):
        """Drive `progressiverender.step` the way Render mode does after its preview stages (8, 16, 32, 64 samples, accumulating)."""
        state = P.ProgressiveState(key="k", samples=4, width=SIZE[0], height=SIZE[1], stage=2)
        states = []
        for _ in range(rounds):
            state = P.step(state, scene, camera, SIZE[0], SIZE[1], BACKGROUND, 0.1, "k", backend=backend, max_bounces=2, accumulate=True)
            states.append(state)
            if P.converged(state, *SIZE):
                break
        return states

    def test_a_flat_scene_settles_in_fewer_samples_than_a_noisy_one(self):
        flat = self.settle(flat_scene(), FLAT_CAMERA)
        noisy = self.settle(shadow_scene(), SHADOW_CAMERA)
        self.assertTrue(flat[-1].finished)
        self.assertEqual(flat[-1].samples, P.ADAPTIVE_FROM)           # every pixel was clean at the minimum
        self.assertEqual(len(flat), 1)
        flat_mean = float(flat[-1].accumulation["count"].mean())
        noisy_mean = float(noisy[-1].accumulation["count"].mean())
        self.assertLess(flat_mean, noisy_mean)
        self.assertGreater(noisy[-1].samples, flat[-1].samples)
        self.assertLess(noisy[-1].converged_fraction, 1.0 + 1e-9)

    def test_the_converged_fraction_and_mask_grow_as_the_view_refines(self):
        states = self.settle(shadow_scene(), SHADOW_CAMERA)
        fractions = [state.converged_fraction for state in states]
        self.assertEqual(fractions, sorted(fractions))
        for state in states:
            self.assertEqual(state.converged_mask.shape, (SIZE[1], SIZE[0]))
            self.assertAlmostEqual(float(state.converged_mask.mean()), state.converged_fraction)

    def test_the_cpu_result_for_a_fixed_seed_is_the_same_every_time(self):
        first = self.settle(shadow_scene(), SHADOW_CAMERA)
        second = self.settle(shadow_scene(), SHADOW_CAMERA)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.image, b.image)
            np.testing.assert_array_equal(a.accumulation["count"], b.accumulation["count"])
            self.assertEqual(a.converged_fraction, b.converged_fraction)

    def test_a_cancelled_pass_leaves_the_previous_accumulation_and_a_new_view_starts_over(self):
        scene, camera = shadow_scene(), SHADOW_CAMERA
        state = self.settle(scene, camera, rounds=2)[-1]
        counts = state.accumulation["count"].copy()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            P.step(state, scene, camera, SIZE[0], SIZE[1], BACKGROUND, 0.1, "k", backend="cpu", max_bounces=2, cancel=cancel, accumulate=True)
        np.testing.assert_array_equal(state.accumulation["count"], counts)           # the pass that stopped stored nothing
        resumed = P.step(state, scene, camera, SIZE[0], SIZE[1], BACKGROUND, 0.1, "k", backend="cpu", max_bounces=2, accumulate=True)
        self.assertGreater(float(resumed.accumulation["count"].sum()), float(counts.sum()))
        # a new key (the camera moved, the scene changed) discards the noise estimate: nothing is carried into it
        again = P.step(state, scene, camera, SIZE[0], SIZE[1], BACKGROUND, 0.1, "other", backend="cpu", max_bounces=2, accumulate=True)
        self.assertIsNone(again.accumulation)
        self.assertIsNone(again.converged_fraction)
        self.assertEqual(again.passes, 1)

    def test_a_superseded_view_restarts_from_its_first_look_and_reaches_the_same_picture(self):
        scene, camera = shadow_scene(), SHADOW_CAMERA

        def request(identity):
            return Request(identity, scene, camera, SIZE[0], SIZE[1], BACKGROUND, 0.1, backend="cpu", max_bounces=2)
        job = ProgressiveJob(settle=0.0, denoise_passes=False, convert=False)
        self.addCleanup(job.close)
        job.submit(request("a"))
        job.submit(request("b"))                  # supersedes "a" wherever its worker is
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 120))
        picture = job.view("b")[0]
        self.assertEqual([frame.state.passes for frame in job.frames][:2], [1, 2])      # b's own first passes, none of a's
        other = ProgressiveJob(settle=0.0, denoise_passes=False, convert=False)
        self.addCleanup(other.close)
        other.submit(request("b"))
        self.assertTrue(other.wait(lambda state, phase: phase == "settled", 120))
        np.testing.assert_array_equal(picture.image, other.view("b")[0].image)       # the interrupted job and a clean one agree

    def test_an_earlier_frame_keeps_no_accumulation(self):
        states = []
        job = ProgressiveJob(settle=0.0, denoise_passes=False, convert=False)
        self.addCleanup(job.close)
        job.submit(Request("v", shadow_scene(), SHADOW_CAMERA, SIZE[0], SIZE[1], BACKGROUND, 0.1, backend="cpu", max_bounces=2))
        self.assertTrue(job.wait(lambda state, phase: phase == "settled", 120))
        older = [frame.state for frame in job.frames[:-1]]
        self.assertTrue(older)
        self.assertTrue(all(state.accumulation is None and state.image is None for state in older))
        self.assertIsNotNone(job.frames[-1].state.accumulation)


class SecondStageTests(unittest.TestCase):
    def test_a_second_stage_the_first_stages_cost_says_is_slow_is_skipped(self):
        stages = P.PREVIEW_STAGES
        cheap = P.ProgressiveState(key="k", samples=stages[0][1], width=1920, height=1080, preview=True, stage=1, seconds=0.2,
                                   cost_per_pixel=0.3e-6)
        dear = replace(cheap, cost_per_pixel=5e-6)
        self.assertFalse(P._ladder_over(cheap, 1920, 1080, stages))
        self.assertTrue(P._ladder_over(dear, 1920, 1080, stages))
        self.assertEqual(P.next_samples(dear, 1920, 1080), 8)         # straight to the first full-size pass
        self.assertEqual(P.next_samples(cheap, 1920, 1080), stages[1][1])
        # a state with no measured cost (a scripted one) keeps the stage
        self.assertFalse(P._ladder_over(replace(cheap, cost_per_pixel=0.0), 1920, 1080, stages))


def noisy_picture(width=64, height=48, seed=3):
    """A smooth picture with its guides plus noise: what the filter is for, without the tracer."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    albedo = np.stack([0.3 + 0.4 * (x > width // 2), 0.5 + 0.0 * x, 0.6 - 0.3 * (y > height // 2)], axis=-1).astype(np.float32)
    normals = np.zeros((height, width, 3), np.float32)
    normals[..., 2] = 1.0
    normals[:, width // 2:, 0] = 0.6
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    depth = (2.0 + 0.01 * x + 0.5 * (x > width // 2)).astype(np.float32)
    clean = albedo * (0.5 + 0.5 * np.sin(y / 9.0))[..., None]
    color = np.concatenate([clean + 0.12 * rng.standard_normal(clean.shape).astype(np.float32), np.ones((height, width, 1), np.float32)], -1)
    color[:4, :4] = 0.0           # a corner with no coverage: the filter leaves it alone
    return color.astype(np.float32), albedo, normals, depth


@unittest.skipUnless(_gpu_available(), "no graphics adapter")
class CardFilterTests(unittest.TestCase):
    def test_the_card_filter_agrees_with_the_numpy_filter(self):
        color, albedo, normals, depth = noisy_picture()
        expected = ptdenoise.denoise(color, albedo, normals, depth, None, dtype=np.float32)
        got = gpudenoise.denoise(color, albedo, normals, depth)
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.isfinite(got).all())
        np.testing.assert_allclose(got, expected, atol=2e-3, rtol=2e-3)
        np.testing.assert_array_equal(got[..., 3], color[..., 3])                      # coverage is the tracer's own
        self.assertLess(float(np.abs(got - expected).mean()), 2e-5)
        self.assertLess(float(np.abs(got[..., :3] - expected[..., :3]).max()), 0.01)  # and it did filter: the noise is far larger
        self.assertGreater(float(np.abs(color - expected)[..., :3].mean()), 0.01)

    def test_a_repeat_with_the_same_guides_gives_the_same_picture(self):
        color, albedo, normals, depth = noisy_picture()
        first = gpudenoise.denoise(color, albedo, normals, depth)
        second = gpudenoise.denoise(color, albedo, normals, depth)
        np.testing.assert_array_equal(first, second)
        other = gpudenoise.denoise(color * 0.5, albedo, normals, depth)           # the guides stay uploaded, the picture does not
        self.assertFalse(np.array_equal(first, other))

    def test_a_changed_guide_is_uploaded_again(self):
        color, albedo, normals, depth = noisy_picture()
        base = gpudenoise.denoise(color, albedo, normals, depth)
        flat_normals = np.zeros_like(normals)
        flat_normals[..., 2] = 1.0
        changed = gpudenoise.denoise(color, albedo, flat_normals, depth)
        expected = ptdenoise.denoise(color, albedo, flat_normals, depth, None, dtype=np.float32)
        np.testing.assert_allclose(changed, expected, atol=2e-3, rtol=2e-3)
        self.assertFalse(np.array_equal(base, changed))

    def test_a_cancelled_filter_raises_and_the_next_one_still_works(self):
        color, albedo, normals, depth = noisy_picture()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            gpudenoise.denoise(color, albedo, normals, depth, cancel=cancel)
        np.testing.assert_allclose(gpudenoise.denoise(color, albedo, normals, depth),
                                   ptdenoise.denoise(color, albedo, normals, depth, None, dtype=np.float32), atol=2e-3, rtol=2e-3)

    def test_a_picture_too_small_for_a_gradient_is_refused(self):
        color, albedo, normals, depth = noisy_picture(width=1, height=8)
        with self.assertRaises(gpu3d.Unsupported):
            gpudenoise.denoise(color, albedo, normals, depth)


class FilterChoiceTests(unittest.TestCase):
    def test_the_cpu_backend_keeps_the_numpy_filter(self):
        color, albedo, normals, depth = noisy_picture()
        got = P.denoise_picture(color, albedo, normals, depth, backend="cpu")
        np.testing.assert_array_equal(got, ptdenoise.denoise_strips(color, albedo, normals, depth, None, workers=2, dtype=np.float32))

    def test_a_card_that_refuses_the_picture_leaves_it_to_the_processor(self):
        color, albedo, normals, depth = noisy_picture()
        with mock.patch.object(gpudenoise, "available", return_value=True), \
                mock.patch.object(gpudenoise, "denoise", side_effect=gpu3d.Unsupported("too large")):
            got = P.denoise_picture(color, albedo, normals, depth, backend="auto")
        self.assertFalse(P._CARD_FILTER["broken"])
        np.testing.assert_array_equal(got, ptdenoise.denoise_strips(color, albedo, normals, depth, None, workers=P.DENOISE_WORKERS,
                                                                   dtype=np.float32))

    def test_a_card_failure_turns_the_card_filter_off_for_the_session(self):
        color, albedo, normals, depth = noisy_picture()
        self.addCleanup(P._CARD_FILTER.update, broken=False)
        with mock.patch.object(gpudenoise, "available", return_value=True), \
                mock.patch.object(gpudenoise, "denoise", side_effect=RuntimeError("device lost")) as card:
            P.denoise_picture(color, albedo, normals, depth, backend="auto")
            P.denoise_picture(color, albedo, normals, depth, backend="auto")
        self.assertEqual(card.call_count, 1)            # the second picture did not ask the card again
        self.assertTrue(P._CARD_FILTER["broken"])

    def test_a_cancel_passes_through_the_card_filter(self):
        color, albedo, normals, depth = noisy_picture()
        with mock.patch.object(gpudenoise, "available", return_value=True), \
                mock.patch.object(gpudenoise, "denoise", side_effect=Cancelled()):
            with self.assertRaises(Cancelled):
                P.denoise_picture(color, albedo, normals, depth, backend="auto")


class WorkgroupTests(unittest.TestCase):
    def test_nvidia_keeps_the_8x8_workgroup_for_splats_with_smoke_and_the_others_use_4x4(self):
        from nodebased import gpupathtrace
        nvidia = {"info": {"vendor": "NVIDIA Corporation", "device": "NVIDIA GeForce RTX 3080 Ti", "vendor_id": 4318}}
        amd = {"info": {"vendor": "AMD", "device": "AMD Radeon Graphics (RADV GFX1151)", "vendor_id": 4098}}
        software = {"info": {"vendor": "Mesa", "device": "llvmpipe (LLVM 22.1.8, 256 bits)", "vendor_id": 65541}}
        self.assertTrue(gpupathtrace.wide_workgroups(nvidia))
        self.assertFalse(gpupathtrace.wide_workgroups(amd))
        self.assertFalse(gpupathtrace.wide_workgroups(software))
        for wide, expected in ((True, 8), (False, 4)):
            self.assertEqual(gpupathtrace._wg_size(True, True, False, wide), expected)
            self.assertEqual(gpupathtrace._wg_size(True, False, True, wide), expected)      # splats with light links
        for wide in (True, False):
            for kinds in ((False, False), (True, False), (False, True)):
                self.assertEqual(gpupathtrace._wg_size(*kinds, False, wide), 8)             # every other variant stays 8x8
        self.assertIn("@workgroup_size(8, 8, 1)", gpupathtrace.shader_source(splats=True, volumes=True, wide=True))
        self.assertIn("@workgroup_size(4, 4, 1)", gpupathtrace.shader_source(splats=True, volumes=True))


class RefiningViewTests(unittest.TestCase):
    def test_the_tint_marks_exactly_the_pixels_still_above_the_threshold(self):
        rgba = np.full((6, 8, 4), 100, np.uint8)
        rgba[..., 3] = 255
        mask = np.ones((6, 8), bool)
        mask[2:4, 3:6] = False
        tinted = P.tint_refining(rgba, mask)
        self.assertIsNot(tinted, rgba)
        np.testing.assert_array_equal(rgba[..., 0], 100)                      # the picture itself is untouched
        np.testing.assert_array_equal(tinted[mask], rgba[mask])
        self.assertTrue((tinted[~mask][:, 0] > 100).all() and (tinted[~mask][:, 2] < 100).all())
        np.testing.assert_array_equal(tinted[..., 3], 255)
        np.testing.assert_array_equal(P.tint_refining(rgba, None), rgba)       # no measurement: no tint
        np.testing.assert_array_equal(P.tint_refining(rgba, np.ones((2, 2), bool)), rgba)

    def test_the_state_carries_the_mask_its_fraction_was_taken_from(self):
        states = SettleTests().settle(shadow_scene(), SHADOW_CAMERA)
        last = states[-1]
        self.assertAlmostEqual(float(last.converged_mask.mean()), last.converged_fraction)
        self.assertFalse(last.converged_mask.all())


if __name__ == "__main__":
    unittest.main()
