"""R6 (docs/SPLAT_RELIGHTING.md, "Production look"): the viewport's progressive "Render" mode
(`progressiverender`) -- a low-res first pass, doubling samples toward `SAMPLE_CAP`, a reset on a
new key, and convergence toward the path tracer's own output. GPU-accelerated (`backend="auto"`);
these tests need a real trace so they are not GPU-gated the way the rasterised viewport tests are,
but they do skip if the path tracer has no working backend at all.
"""
import os
import time
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import pathtrace, progressiverender as P, scene3d as s

BACKGROUND = (0.02, 0.02, 0.03, 1.0)
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 4, 9)), s.Vec3(0, 0, 0), 45.0, 0.1, 100.0)


def _scene():
    floor = replace(s._card(8, 8, (0.8, 0.8, 0.8, 1.0), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0))),
                    material="pbr", metallic=0.0, pbr_roughness=0.9, pbr_specular=0.5)
    sphere = replace(s._sphere(1.0, 16, (0.8, 0.2, 0.2, 1.0), s.Transform3D(s.Vec3(0, 0.2, 0))),
                     material="pbr", metallic=0.0, pbr_roughness=0.15, pbr_specular=0.5)
    light = s.Light(kind="Directional", intensity=1.5, position=s.Vec3(4, 6, 2), target=s.Vec3(0, 0, 0))
    return s.Scene((floor, sphere), (light,))


def _skip_reason():
    try:
        pathtrace.render(s.Scene(), CAMERA, 4, 3, BACKGROUND, output="rgba",
                         settings=pathtrace.PathSettings(samples=1), backend="auto")
    except Exception as error:  # pragma: no cover - environment without any path tracer backend
        return str(error)
    return None


_SKIP = _skip_reason()


@unittest.skipIf(_SKIP, f"no working path tracer backend: {_SKIP}")
class ProgressiveSteps(unittest.TestCase):
    def test_the_reset_step_is_low_resolution_and_one_sample(self):
        state = P.step(None, _scene(), CAMERA, 200, 150, BACKGROUND, 0.1, key="a")
        self.assertEqual(state.samples, P.FIRST_PASS_SAMPLES)
        self.assertTrue(state.low_res)
        self.assertEqual((state.width, state.height), (200 // P.FIRST_PASS_SCALE, 150 // P.FIRST_PASS_SCALE))
        self.assertEqual(state.image.shape, (state.height, state.width, 4))

    def test_later_steps_are_full_size_and_double_the_sample_count(self):
        state = P.step(None, _scene(), CAMERA, 80, 60, BACKGROUND, 0.1, key="a")
        state = P.step(state, _scene(), CAMERA, 80, 60, BACKGROUND, 0.1, key="a")
        self.assertEqual(state.samples, P.FIRST_PASS_SAMPLES * 2)
        self.assertFalse(state.low_res)
        self.assertEqual((state.width, state.height), (80, 60))
        state = P.step(state, _scene(), CAMERA, 80, 60, BACKGROUND, 0.1, key="a")
        self.assertEqual(state.samples, P.FIRST_PASS_SAMPLES * 4)

    def test_sample_count_stops_doubling_at_the_cap(self):
        state = P.step(None, _scene(), CAMERA, 60, 45, BACKGROUND, 0.1, key="a")
        for _ in range(20):
            state = P.step(state, _scene(), CAMERA, 60, 45, BACKGROUND, 0.1, key="a")
        self.assertEqual(state.samples, P.SAMPLE_CAP)
        self.assertTrue(P.converged(state, 60, 45))

    def test_a_different_key_resets_to_the_low_res_first_pass(self):
        state = P.step(None, _scene(), CAMERA, 80, 60, BACKGROUND, 0.1, key="a")
        state = P.step(state, _scene(), CAMERA, 80, 60, BACKGROUND, 0.1, key="a")
        self.assertGreater(state.samples, P.FIRST_PASS_SAMPLES)
        moved = P.step(state, _scene(), CAMERA, 80, 60, BACKGROUND, 0.1, key="b")
        self.assertEqual(moved.samples, P.FIRST_PASS_SAMPLES)
        self.assertTrue(moved.low_res)

    def test_not_converged_before_the_cap_or_while_still_low_res(self):
        state = P.step(None, _scene(), CAMERA, 60, 45, BACKGROUND, 0.1, key="a")
        self.assertFalse(P.converged(state, 60, 45))
        state = P.step(state, _scene(), CAMERA, 60, 45, BACKGROUND, 0.1, key="a")
        self.assertFalse(P.converged(state, 60, 45))   # samples=2, far under the cap

    def test_particles_are_dropped_instead_of_raising(self):
        particles = s.ParticleInstance(np.zeros((4, 3), "f4"), np.full(4, 0.1, "f4"), np.ones((4, 4), "f4"))
        scene = replace(_scene(), particles=(particles,))
        state = P.step(None, scene, CAMERA, 40, 30, BACKGROUND, 0.1, key="a")
        self.assertIsNotNone(state.image)

    def test_more_samples_measurably_converge_toward_a_high_sample_reference(self):
        """A rough sphere is noisy at one sample; the progressive steps should get closer to a
        high-sample reference as they go, not just louder or unrelated."""
        reference = pathtrace.render(_scene(), CAMERA, 96, 72, BACKGROUND, ambient=0.1, output="rgba",
                                     settings=pathtrace.PathSettings(samples=256, max_bounces=P.MAX_BOUNCES),
                                     backend="auto")
        state = P.step(None, _scene(), CAMERA, 96, 72, BACKGROUND, 0.1, key="a")
        errors = []
        for _ in range(6):
            state = P.step(state, _scene(), CAMERA, 96, 72, BACKGROUND, 0.1, key="a")
            if state.low_res:
                continue
            errors.append(float(np.abs(state.image - reference).mean()))
        self.assertGreaterEqual(len(errors), 3)
        self.assertLess(errors[-1], errors[0])
