"""Lane L6 fluids plan, step A part 3: the volume control passes and the depth pass with volumes."""
import math
import unittest
from dataclasses import replace

import numpy as np

from nodebased import scene3d, volumerender
from nodebased.core import CHOICES, Dispatcher
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from tests.test_volume_render import CAMERA, FLAT, SIZE, card
from tests.test_volume_scene import box, make, wire

CENTRE = (SIZE // 2, SIZE // 2)


def render(scene, output, settings=FLAT, size=SIZE):
    return scene3d.render(scene, CAMERA, size, size, output=output, volume=settings)


class DensityTests(unittest.TestCase):
    def test_volume_density_is_the_integrated_scaled_density(self):
        image = render(scene3d.Scene(volumes=(box(32, 2.0),)), "volume_density", replace(FLAT, density_scale=1.5))
        self.assertAlmostEqual(float(image[CENTRE][0]), 2.0 * 1.5, delta=.03)
        np.testing.assert_array_equal(image[CENTRE][:3], (image[CENTRE][0],) * 3)
        self.assertEqual(float(image[CENTRE][3]), 1.0)
        self.assertEqual(float(np.abs(image[0, 0]).max()), 0.0)

    def test_a_card_in_front_removes_the_smoke_it_hides(self):
        smoke = box(32, 2.0)
        bare = render(scene3d.Scene(volumes=(smoke,)), "volume_density")
        hidden = render(scene3d.Scene((card(2.0),), volumes=(smoke,)), "volume_density")
        self.assertEqual(float(np.abs(hidden[CENTRE]).max()), 0.0)
        half = render(scene3d.Scene((card(0.0),), volumes=(smoke,)), "volume_density")
        self.assertAlmostEqual(float(half[CENTRE][0]), float(bare[CENTRE][0]) / 2, delta=.03)

    def test_temperature_is_density_weighted(self):
        smoke = box(16, 2.0, temperature=5.0)
        temperature = render(scene3d.Scene(volumes=(smoke,)), "volume_temperature")
        density = render(scene3d.Scene(volumes=(smoke,)), "volume_density")
        # Both fields ramp to zero over the outer half voxel, so the product loses a hair at the faces.
        np.testing.assert_allclose(temperature[CENTRE][0], 5.0 * density[CENTRE][0], rtol=.02)
        self.assertEqual(float(render(scene3d.Scene(volumes=(box(),)), "volume_temperature")[CENTRE][0]), 0.0)


class VorticityTests(unittest.TestCase):
    def test_curl_magnitude_of_a_rigid_rotation(self):
        n = 24
        axis = (np.arange(n) + .5) / n - .5
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        omega = 2.0
        velocity = np.stack((-omega * y, omega * x, np.zeros_like(x)), -1)      # curl = 2 * omega along z
        smoke = scene3d.Volume(np.ones((n,) * 3), 1.0 / n, (-.5,) * 3, velocity=velocity)
        grid = volumerender.vorticity_magnitude(smoke)
        np.testing.assert_allclose(grid, 2 * omega, atol=1e-4)
        image = render(scene3d.Scene(volumes=(smoke,)), "volume_vorticity")
        self.assertAlmostEqual(float(image[CENTRE][0]), 2 * omega * (1 - .5 / n), delta=.15)
        self.assertEqual(float(image[0, 0][0]), 0.0)

    def test_no_velocity_means_no_vorticity_or_motion(self):
        scene = scene3d.Scene(volumes=(box(),))
        for name in ("volume_vorticity", "volume_motion"):
            image = render(scene, name)
            self.assertEqual(float(np.abs(image[..., :3]).max()), 0.0)
            self.assertEqual(float(image[CENTRE][3]), 1.0)


class MotionTests(unittest.TestCase):
    def expected(self, point, velocity, fps=24.0):
        here, _ = scene3d.project(CAMERA, SIZE, SIZE, [point])
        there, _ = scene3d.project(CAMERA, SIZE, SIZE, [np.array(point) + np.array(velocity) / fps])
        return (there - here)[0] * (1, -1)          # Nuke: +x right, +y up

    def test_a_uniformly_moving_volume_gives_the_projected_velocity(self):
        for velocity in ((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (12.0, -12.0, 0.0)):
            with self.subTest(velocity=velocity):
                smoke = box(16, 2.0, velocity=velocity)
                image = render(scene3d.Scene(volumes=(smoke,)), "volume_motion")
                point = np.array((0.0, 0.0, 0.0))
                want = self.expected(point, velocity)
                np.testing.assert_allclose(image[CENTRE][:2], want, atol=1.0)
                self.assertGreater(float(np.abs(want).max()), 1.0)      # a real displacement, not zero
                self.assertEqual(float(image[CENTRE][2]), 0.0)
                self.assertEqual(float(image[CENTRE][3]), 1.0)

    def test_motion_follows_the_volume_matrix_and_frame_rate(self):
        turn = np.eye(4, dtype=np.float32)
        turn[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))         # object +x is world +y
        smoke = box(16, 2.0, velocity=(6.0, 0.0, 0.0), matrix=turn)
        image = render(scene3d.Scene(volumes=(smoke,)), "volume_motion", replace(FLAT, fps=12.0))
        np.testing.assert_allclose(image[CENTRE][:2], self.expected((0, 0, 0), (0.0, 6.0, 0.0), 12.0), atol=1.0)
        self.assertGreater(float(image[CENTRE][1]), 1.0)
        self.assertLess(abs(float(image[CENTRE][0])), 1.0)

    def test_motion_is_zero_where_there_is_no_smoke(self):
        image = render(scene3d.Scene(volumes=(box(16, 2.0, velocity=(12.0, 0, 0)),)), "volume_motion")
        self.assertEqual(float(np.abs(image[0, 0]).max()), 0.0)


class DepthTests(unittest.TestCase):
    def test_depth_includes_the_first_hit_of_the_volume(self):
        settings = replace(FLAT, step_size=.05, depth_threshold=.1)
        image = render(scene3d.Scene(volumes=(box(16, 2.0),)), "depth", settings)
        self.assertAlmostEqual(float(image[CENTRE][0]), 4.5, delta=.06)      # the front face is 4.5 from the eye
        self.assertEqual(float(image[CENTRE][3]), 1.0)
        self.assertEqual(float(image[0, 0][3]), 0.0)

    def test_depth_takes_the_nearer_of_mesh_and_volume(self):
        smoke = box(16, 2.0)
        front = render(scene3d.Scene((card(2.0),), volumes=(smoke,)), "depth")
        self.assertAlmostEqual(float(front[CENTRE][0]), 3.0, places=3)
        behind = render(scene3d.Scene((card(-2.0),), volumes=(smoke,)), "depth")
        self.assertAlmostEqual(float(behind[CENTRE][0]), 4.5, delta=.06)
        # A card wider than the volume still shows at its own depth outside the smoke's silhouette.
        self.assertAlmostEqual(float(behind[2, 2][0]), 7.0, delta=.01)

    def test_smoke_below_the_threshold_leaves_depth_alone(self):
        settings = replace(FLAT, depth_threshold=5.0)
        image = render(scene3d.Scene(volumes=(box(16, 2.0),)), "depth", settings)
        self.assertEqual(float(image[CENTRE][3]), 0.0)
        raised = render(scene3d.Scene(volumes=(box(16, 2.0),)), "depth", replace(FLAT, depth_threshold=1.5))
        self.assertEqual(float(raised[CENTRE][3]), 1.0)

    def test_return_depth_buffer_carries_the_volume_hit(self):
        image, depth = scene3d.render(scene3d.Scene(volumes=(box(16, 2.0),)), CAMERA, SIZE, SIZE,
                                      output="depth", return_depth=True, volume=FLAT)
        self.assertAlmostEqual(float(depth[CENTRE]), float(image[CENTRE][0]), places=6)


class GraphTests(unittest.TestCase):
    def test_the_passes_are_render3d_outputs_with_knobs(self):
        names = ("volume_density", "volume_motion", "volume_temperature", "volume_vorticity")
        for name in names:
            self.assertIn(name, CHOICES["render_output"])
            self.assertIn(name, scene3d.RENDER_OUTPUTS)
        groups = {p for g in knob_layout("Render3D") for p in g.params}
        for param in ("volumes", "volume_step_size", "volume_density_scale", "volume_shadow_density",
                      "volume_shadow_steps", "volume_scattering", "volume_absorption", "volume_red",
                      "volume_green", "volume_blue", "volume_fps", "volume_depth_threshold"):
            self.assertIn(param, groups)

    def test_a_pass_renders_through_the_graph_and_cpu_only(self):
        d = Dispatcher()
        make(d, p=("Plume3D", {"plume_resolution": 12}), s=("Scene3D", {}),
             c=("Camera3D", {"ty": .5, "tz": 3, "target_y": .5}),
             r=("Render3D", {"width": 24, "height": 24, "samples": 2, "render_output": "volume_density",
                             "volume_density_scale": 4.0}))
        wire(d, "s", "object0", "p")
        wire(d, "r", "scene", "s")
        wire(d, "r", "camera", "c")
        image = Evaluator().evaluate(dict(d.document, view="r"), frame=1)
        self.assertGreater(float(image[..., 0].max()), .1)
        self.assertEqual(image.shape, (24, 24, 4))
        d.execute({"op": "set", "id": "r", "param": "render_backend", "value": "gpu"})
        with self.assertRaisesRegex(ValueError, "volumes are CPU-only"):
            Evaluator().evaluate(dict(d.document, view="r"), frame=1)

    def test_passes_are_deterministic(self):
        scene = scene3d.Scene(volumes=(scene3d.analytic_plume(12, 5),))
        for name in scene3d.VOLUME_OUTPUTS:
            np.testing.assert_array_equal(render(scene, name, size=16), render(scene, name, size=16))


class MultichannelTests(unittest.TestCase):
    """Runs once lane 4's multichannel EXR (Render3D `passes`) is in the tree; skipped before that."""

    @unittest.skipUnless("volume_density" in getattr(scene3d, "MULTICHANNEL_PASSES", ()),
                         "Render3D `passes` does not carry the volume layers yet")
    def test_volume_layers_land_in_the_exr_with_their_names(self):
        import tempfile
        from pathlib import Path
        from nodebased import media
        beauty, layers = scene3d.render_multichannel(
            scene3d.Scene(volumes=(box(16, 2.0, velocity=(1, 0, 0), temperature=1.0),)), CAMERA, 16, 16,
            passes="beauty,depth,volume_density,volume_motion,volume_temperature,volume_vorticity")
        self.assertEqual(set(layers), {"depth", "volume_density", "volume_motion", "volume_temperature",
                                       "volume_vorticity"})
