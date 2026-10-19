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

from nodebased import gpupathtrace, pathtrace, progressiverender as P, ptdenoise, scene3d as s
from nodebased.cancellation import Cancelled
from tests.test_3d_pathtrace import CORNELL_CAMERA, cornell
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

    def test_scene_reuse_builds_once_per_scene_and_camera_and_not_outside_the_block(self):
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
                self.assertEqual(len(calls), 2)            # another camera position: prepared again
                trace(_scene(), CAMERA)
                self.assertEqual(len(calls), 3)            # another scene object: prepared again
            before = len(calls)
            trace(scene, CAMERA)
            trace(scene, CAMERA)
            self.assertEqual(len(calls) - before, 2)       # outside the block every render prepares its own


if __name__ == "__main__":
    unittest.main()
