"""Lane 4 step H1: the denoised first look of the viewport's progressive Render mode (`progressiverender.step_interactive`).

Covers first-image delivery (output size, nonempty, finite, honestly marked as an enlarged denoised preview), that the
preview is cleaner than the raw pass it came from, the stage ladder and the hand-over to the unchanged doubling steps, reset on
a new key, cancellation, that the scene preparation reuse never leaks into a normal render, and that the threaded denoiser
gives the single-call result. The one-second timing is measured by `tools/benchmark_first_look.py`, not asserted here: a
test machine's load would make a time limit flaky.
"""
import os
import unittest
from dataclasses import replace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpupathtrace, pathtrace, progressiverender as P, ptdenoise, scenecache, scene3d as s
from nodebased.cancellation import Cancelled
from tests.test_3d_pathtrace import CORNELL_CAMERA, cornell, gpu_ready
from tests.progressive_widget import first_picture, paint_until
from tests.test_3d_progressive_render import BACKGROUND, CAMERA, _SKIP, _scene

WIDTH, HEIGHT = 160, 90
STAGES = ((120 * 68, 4), (140 * 79, 8))       # small stand-ins for PREVIEW_STAGES (the filter's reach needs a few dozen pixels)


def look(state=None, key="a", scene=None, camera=CAMERA, **more):
    return P.step_interactive(state, scene or _scene(), camera, WIDTH, HEIGHT, BACKGROUND, 0.1, key, stages=STAGES, **more)


class DenoiseStripsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        h, w = 131, 47
        self.albedo = rng.uniform(0.2, 0.9, (h, w, 3))
        self.albedo[:, : w // 2] *= 0.5
        self.normal = np.zeros((h, w, 3))
        self.normal[..., 2] = 1.0
        self.normal[60:, 30:] = (1.0, 0.0, 0.0)
        self.depth = np.linspace(2.0, 8.0, h)[:, None] * np.ones((1, w))
        self.color = np.concatenate([self.albedo * 0.6, np.ones((h, w, 1))], -1) * np.exp(rng.normal(0, 0.4, (h, w, 1)))
        self.color[..., 3] = 1.0

    def test_strips_give_the_single_call_result(self):
        whole = ptdenoise.denoise(self.color, self.albedo, self.normal, self.depth, dtype=np.float32)
        for workers in (2, 5, 8):
            strips = ptdenoise.denoise_strips(self.color, self.albedo, self.normal, self.depth, workers=workers,
                                              min_rows=8, dtype=np.float32)
            np.testing.assert_allclose(strips, whole, atol=1e-6)

    def test_float32_follows_float64(self):
        a = ptdenoise.denoise(self.color, self.albedo, self.normal, self.depth)
        b = ptdenoise.denoise(self.color, self.albedo, self.normal, self.depth, dtype=np.float32)
        self.assertEqual(b.dtype, np.float32)
        np.testing.assert_allclose(b, a, atol=2e-5)

    def test_a_short_image_falls_through_to_one_call(self):
        out = ptdenoise.denoise_strips(self.color[:20], self.albedo[:20], self.normal[:20], self.depth[:20], workers=8)
        ref = ptdenoise.denoise(self.color[:20], self.albedo[:20], self.normal[:20], self.depth[:20])
        np.testing.assert_array_equal(out, ref)


class GuideAmbientTests(unittest.TestCase):
    def test_the_guide_passes_do_not_depend_on_the_ambient_level(self):
        """`guide_aovs(ambient=...)` only lets a GPU render share the beauty's prepared scene; the passes must stay the same."""
        room = cornell()
        settings = pathtrace.PathSettings(samples=2, max_bounces=3)
        dark = pathtrace.guide_aovs(room, CORNELL_CAMERA, 16, 12, settings, backend="cpu")
        lit = pathtrace.guide_aovs(room, CORNELL_CAMERA, 16, 12, settings, backend="cpu", ambient=0.5)
        for name in dark:
            np.testing.assert_array_equal(dark[name], lit[name])


class DisplayHelperTests(unittest.TestCase):
    def test_enlarge_keeps_a_flat_image_and_the_size(self):
        flat = np.full((9, 16, 4), 0.25, np.float32)
        big = P.enlarge(flat, 160, 90)
        self.assertEqual(big.shape, (90, 160, 4))
        np.testing.assert_allclose(big, 0.25, atol=1e-6)
        self.assertIs(P.enlarge(big, 160, 90).dtype, np.dtype(np.float32))

    def test_enlarge_interpolates_between_pixels(self):
        ramp = np.zeros((2, 2, 1), np.float32)
        ramp[:, 1] = 1.0
        row = P.enlarge(ramp, 8, 2)[0, :, 0]
        self.assertTrue(np.all(np.diff(row) >= 0))
        self.assertAlmostEqual(float(row[0]), 0.0, places=5)
        self.assertAlmostEqual(float(row[-1]), 1.0, places=5)

    def test_display_rgba8_is_within_a_level_of_the_exact_curve(self):
        value = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
        image = np.stack([value, value, value, np.ones_like(value)], -1)[None]
        shown = P.display_rgba8(image)
        exact = np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055) * 255
        self.assertLessEqual(float(np.abs(shown[0, :, 0] - exact).max()), 1.0)
        self.assertTrue(np.all(shown[..., 3] == 255))

    def test_display_rgba8_unpremultiplies(self):
        half = np.array([[[0.25, 0.25, 0.25, 0.5]]], np.float32)
        full = np.array([[[0.5, 0.5, 0.5, 1.0]]], np.float32)
        np.testing.assert_array_equal(P.display_rgba8(half), P.display_rgba8(full))


@unittest.skipIf(_SKIP, f"no working path tracer backend: {_SKIP}")
class FirstLookTests(unittest.TestCase):
    def setUp(self):
        P._COST.clear()
        patch = mock.patch.object(P, "STAGE_TWO_MAX_FIRST_SECONDS", 1e9)      # a loaded test machine must not skip stage two
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(P._COST.clear)

    def test_the_first_look_is_an_output_sized_nonempty_denoised_preview(self):
        state = look()
        self.assertEqual(state.image.shape, (HEIGHT, WIDTH, 4))
        self.assertEqual(state.image.dtype, np.float32)
        self.assertTrue(np.isfinite(state.image).all())
        self.assertGreater(float(state.image[..., :3].std()), 0.0)
        self.assertTrue(state.preview and state.denoised and state.low_res)
        self.assertEqual((state.width, state.height), (WIDTH, HEIGHT))
        self.assertLess(state.traced_width * state.traced_height, WIDTH * HEIGHT)
        self.assertEqual(state.stage, 1)
        self.assertEqual(state.samples, STAGES[0][1])
        self.assertEqual([name for name, _ in state.phases], ["render", "guides", "denoise", "enlarge"])
        self.assertGreater(state.seconds, 0.0)

    def test_the_preview_is_cleaner_than_the_raw_pass_it_came_from(self):
        room = cornell()           # lit by a small rectangle light: noisy at 4 samples, unlike the plain floor and sphere
        state = look(scene=room, camera=CORNELL_CAMERA)
        size = (state.traced_width, state.traced_height)
        trace = lambda samples: P.enlarge(pathtrace.render(
            room, CORNELL_CAMERA, *size, (0, 0, 0, 0), 0.1, "rgba", pathtrace.PathSettings(samples=samples, max_bounces=3),
            backend="auto"), WIDTH, HEIGHT)
        reference = pathtrace.over_background(trace(256), BACKGROUND)       # same resolution, converged: only noise differs
        raw = pathtrace.over_background(trace(state.samples), BACKGROUND)
        shown = lambda image: P.display_rgba8(image)[..., :3].astype(np.float64) / 255      # a lone bright sample is clipped on screen
        error = lambda image: float(np.mean((shown(image) - shown(reference)) ** 2))
        self.assertLess(error(state.image), 0.6 * error(raw))
        np.testing.assert_allclose(shown(state.image).mean(), shown(reference).mean(), rtol=0.05)

    def test_the_background_shows_where_nothing_is_hit(self):
        state = look()
        np.testing.assert_allclose(state.image[0, 0], BACKGROUND, atol=1e-5)

    def test_stages_climb_then_the_doubling_steps_continue_at_full_size(self):
        one = look()
        two = look(one)
        self.assertEqual((one.stage, two.stage), (1, 2))
        self.assertGreater(two.traced_width, one.traced_width)
        self.assertTrue(two.preview and two.denoised)
        three = look(two)
        self.assertFalse(three.preview or three.denoised or three.low_res)
        self.assertEqual((three.width, three.height), (WIDTH, HEIGHT))
        self.assertEqual(three.samples, STAGES[1][1] * 2)
        self.assertEqual(three.image.shape, (HEIGHT, WIDTH, 4))
        four = look(three)                           # the ladder does not start over while the key stays
        self.assertFalse(four.preview)
        self.assertEqual(four.samples, three.samples * 2)

    def test_a_view_no_larger_than_the_first_stage_starts_the_doubling_directly(self):
        state = P.step_interactive(None, _scene(), CAMERA, 64, 48, BACKGROUND, 0.1, "a", stages=((64 * 48, 4),))
        self.assertFalse(state.preview)
        self.assertTrue(state.low_res)              # `step`'s own reset pass, unchanged
        self.assertEqual(state.samples, P.FIRST_PASS_SAMPLES)

    def test_a_new_key_starts_over_and_a_changed_scene_changes_the_picture(self):
        done = look(look())
        moved = look(done, key="b", camera=replace(CAMERA, transform=replace(CAMERA.transform, position=s.Vec3(3, 4, 8))))
        self.assertEqual((moved.stage, moved.passes), (1, 1))
        self.assertTrue(moved.preview)
        self.assertGreater(float(np.abs(moved.image - look().image).mean()), 1e-4)

    def test_cancellation_raises_and_leaves_the_state_and_later_renders_alone(self):
        import threading
        first = look()
        stop = threading.Event()
        stop.set()
        with self.assertRaises(Cancelled):
            look(first, cancel=stop)
        self.assertEqual(first.stage, 1)
        self.assertIsNone(getattr(gpupathtrace._PREPARED, "entry", None))
        again = look(first)
        self.assertEqual(again.stage, 2)

    @unittest.skipUnless(gpu_ready(), "the GPU path tracer is not available")
    def test_a_first_look_prepares_the_scene_once_for_the_beauty_and_all_three_guides(self):
        scenecache.clear()
        self.addCleanup(scenecache.clear)
        calls = []
        real = pathtrace.build_scene
        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        with mock.patch.object(pathtrace, "build_scene", counting):
            P.step_interactive(None, _scene(), CAMERA, WIDTH, HEIGHT, BACKGROUND, 0.1, "a", stages=STAGES, backend="gpu")
        self.assertEqual(len(calls), 1)

    def test_the_final_render_is_the_same_with_and_without_scene_reuse(self):
        scene = _scene()
        settings = pathtrace.PathSettings(samples=8, max_bounces=3)
        plain = pathtrace.render(scene, CAMERA, 48, 32, BACKGROUND, 0.1, "rgba", settings, backend="auto")
        with gpupathtrace.reusing_scene():
            pathtrace.render(scene, CAMERA, 48, 32, BACKGROUND, 0.1, "normals", settings, backend="auto")
            reused = pathtrace.render(scene, CAMERA, 48, 32, BACKGROUND, 0.1, "rgba", settings, backend="auto")
        after = pathtrace.render(scene, CAMERA, 48, 32, BACKGROUND, 0.1, "rgba", settings, backend="auto")
        np.testing.assert_array_equal(plain, reused)
        np.testing.assert_array_equal(plain, after)

    @unittest.skipUnless(gpu_ready(), "the GPU path tracer is not available")
    def test_scene_reuse_builds_once_per_scene_content_and_not_again_for_a_camera_move(self):
        # the preparation is kept by content (`scenecache`): one build serves the four renders of a view, a camera move, an equal
        # scene made again, and the renders outside the block (the thread-local reuse inside it is only the first, cheaper lookup)
        scenecache.clear()
        self.addCleanup(scenecache.clear)
        scene = _scene()
        settings = pathtrace.PathSettings(samples=1, max_bounces=2)
        calls = []
        real = pathtrace.build_scene
        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        def trace(sc, camera, output="rgba"):
            pathtrace.render(sc, camera, 16, 12, BACKGROUND, 0.1, output, settings, backend="gpu")
        with mock.patch.object(pathtrace, "build_scene", counting):
            with gpupathtrace.reusing_scene():
                for output in ("rgba", "albedo", "normals", "depth"):
                    trace(scene, CAMERA, output)
                self.assertEqual(len(calls), 1)
                trace(scene, replace(CAMERA, transform=replace(CAMERA.transform, position=s.Vec3(1, 4, 9))))
                self.assertEqual(len(calls), 1)            # another camera position: the same preparation
                trace(_scene(), CAMERA)
                self.assertEqual(len(calls), 1)            # another scene object of equal content: the same preparation
            trace(scene, CAMERA)
            trace(scene, CAMERA)
            self.assertEqual(len(calls), 1)                # outside the block the content cache serves every render
            scenecache.clear()
            trace(scene, CAMERA)
            self.assertEqual(len(calls), 2)                # dropped: prepared again


@unittest.skipIf(_SKIP, f"no working path tracer backend: {_SKIP}")
class ViewportTests(unittest.TestCase):
    """The viewport's Render mode shows the denoised first look and labels it as an enlarged preview."""

    def _widget(self):
        from tests.test_3d_viewport_render_mode import APP, graph      # APP: the one QApplication
        from nodebased.viewport3d import Viewport3D
        widget = Viewport3D()
        self.addCleanup(widget.close)
        widget.resize(640, 360)           # more pixels than the first stage traces, so the first look applies
        widget.set_document(graph().document)
        widget.render_mode = True
        return widget

    def setUp(self):
        P._COST.clear()
        self.addCleanup(P._COST.clear)

    def test_the_first_picture_is_a_labelled_denoised_preview_then_the_doubling_steps_follow(self):
        widget = self._widget()
        widget.grab()
        self.assertIn("preparing", widget.render_note)               # the interactive picture shows until the first look is ready
        self.assertTrue(paint_until(widget, lambda state, phase: state is not None and not state.preview))
        frames = [frame.state for frame in widget._progressive_job.frames]
        state = frames[0]
        self.assertTrue(state.preview and state.denoised)
        self.assertEqual((state.height, state.width), (360, 640))
        later = next(frame for frame in frames if not frame.preview)
        self.assertGreater(later.samples, state.samples)
        self.assertEqual((later.height, later.width), (360, 640))

    def test_orbiting_starts_a_new_first_look(self):
        widget = self._widget()
        self.assertTrue(paint_until(widget, lambda state, phase: state is not None and not state.preview))
        widget.azimuth += 30.0
        widget.grab()
        self.assertIsNone(widget._progressive_state)                 # not the old camera's picture
        self.assertTrue(paint_until(widget, first_picture))
        first = widget._progressive_job.frames[0].state
        self.assertTrue(first.preview)
        self.assertEqual(first.stage, 1)


class BudgetTests(unittest.TestCase):
    """The first stage sizes itself to the one-second target from what the last first look cost per pixel."""

    def setUp(self):
        P._COST.clear()
        self.addCleanup(P._COST.clear)
        self.surface, self.heavy = s.Scene(), s.Scene(volumes=(object(),))
        self.key = P._cost_key(self.surface)

    def test_an_unseen_surface_scene_starts_at_the_stage_size_and_a_heavy_one_smaller(self):
        stage = 480 * 270
        self.assertEqual(P.first_stage_pixels(self.surface, stage), stage)
        self.assertEqual(P.first_stage_pixels(self.heavy, stage), P.HEAVY_FIRST_PIXELS)

    def test_a_costly_scene_shrinks_the_next_first_look_to_fit_the_target(self):
        stage = 480 * 270
        P._record_cost(self.surface, 1.2, stage)                       # 1.2 s for the full stage: too slow
        pixels = P.first_stage_pixels(self.surface, stage)
        self.assertLess(pixels, stage)
        predicted = pixels * (1.2 / stage) + P.ENLARGE_ALLOWANCE
        self.assertAlmostEqual(predicted, P.FIRST_LOOK_SECONDS * P.FIT_FRACTION, delta=0.01)

    def test_a_cached_preparation_sizes_for_the_warm_target_and_an_uncached_one_pays_for_its_build(self):
        stage = 480 * 270
        P._record_cost(self.surface, 2.0, stage, built=0.4)            # 0.4 of the 2 s was the scene build
        self.assertAlmostEqual(P._COST[self.key], 1.6 / stage)
        self.assertAlmostEqual(P._BUILD[self.key], 0.4)
        warm = P.first_stage_pixels(self.surface, stage, prepared=True)
        rebuilt = P.first_stage_pixels(self.surface, stage, prepared=False)
        unknown = P.first_stage_pixels(self.surface, stage)
        cost = 1.6 / stage
        self.assertAlmostEqual(warm * cost + P.ENLARGE_ALLOWANCE, P.WARM_FIRST_LOOK_SECONDS * P.FIT_FRACTION, delta=0.01)
        self.assertAlmostEqual(rebuilt * cost + P.ENLARGE_ALLOWANCE + 0.4, P.FIRST_LOOK_SECONDS * P.FIT_FRACTION, delta=0.01)
        self.assertAlmostEqual(unknown * cost + P.ENLARGE_ALLOWANCE, P.FIRST_LOOK_SECONDS * P.FIT_FRACTION, delta=0.01)
        self.assertLess(rebuilt, warm)          # a build of 0.4 s costs more pixels than the warm target holds back
        self.assertLess(warm, unknown)

    def test_a_cheap_scene_never_grows_past_the_stage_and_a_dear_one_never_below_the_floor(self):
        stage = 480 * 270
        P._record_cost(self.surface, 0.01, stage)
        self.assertEqual(P.first_stage_pixels(self.surface, stage), stage)
        P._COST[self.key] = 1.0                                         # a second per pixel
        self.assertEqual(P.first_stage_pixels(self.surface, stage), P.MIN_FIRST_PIXELS)

    def test_heavy_and_surface_scenes_keep_separate_costs(self):
        P._record_cost(self.heavy, 2.0, 100000)
        self.assertEqual(P.first_stage_pixels(self.surface, 480 * 270), 480 * 270)

    @unittest.skipIf(_SKIP, f"no working path tracer backend: {_SKIP}")
    def test_a_slow_first_stage_goes_straight_to_the_doubling_steps(self):
        one = look()
        slow = replace(one, seconds=P.STAGE_TWO_MAX_FIRST_SECONDS + 0.5)
        state = look(slow)
        self.assertFalse(state.preview)
        self.assertEqual(state.samples, STAGES[0][1] * 2)
        self.assertEqual((state.width, state.height), (WIDTH, HEIGHT))


if __name__ == "__main__":
    unittest.main()
