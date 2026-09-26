"""Lane L6 fluids plan, step A part 2: the CPU reference raymarch through scene3d.render and Render3D."""
import math
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, scene3d, volumerender
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_volume_scene import box, make, wire

SIZE = 48
CAMERA = scene3d.Camera()          # at (0, 0, 5) looking at the origin
FLAT = volumerender.VolumeSettings(absorption=0.0)   # every bit of extinction scatters


def render(scene, settings=FLAT, size=SIZE, camera=CAMERA, **kwargs):
    return scene3d.render(scene, camera, size, size, samples=1, volume=settings, **kwargs)


def card(z, color=(.2, .4, .8, 1.0), width=6.0):
    return scene3d._card(width, width, color, scene3d.Transform3D(scene3d.Vec3(0, 0, z)))


class RaymarchTests(unittest.TestCase):
    def test_uniform_slab_alpha_matches_beer_lambert(self):
        image = render(scene3d.Scene(volumes=(box(32, 2.0),)))
        centre = image[SIZE // 2, SIZE // 2]
        expected = 1 - math.exp(-2.0)
        self.assertAlmostEqual(float(centre[3]), expected, delta=.01)
        np.testing.assert_allclose(centre[:3], centre[3], atol=1e-5)    # unlit, white smoke: rgb == alpha
        self.assertEqual(float(image[0, 0, 3]), 0.0)                      # rays that miss the box see nothing

    def test_step_size_barely_changes_a_smooth_volume(self):
        scene = scene3d.Scene(volumes=(box(32, 1.5),))
        fine = render(scene, replace(FLAT, step_size=.02))[..., 3]
        coarse = render(scene, replace(FLAT, step_size=.25))[..., 3]
        np.testing.assert_allclose(fine, coarse, atol=.03)

    def test_density_scale_zero_renders_nothing(self):
        scene = scene3d.Scene(volumes=(box(),), lights=(scene3d.Light(),))
        image = render(scene, replace(FLAT, density_scale=0.0), ambient=.2)
        self.assertEqual(float(np.abs(image).max()), 0.0)
        np.testing.assert_array_equal(image, render(scene3d.Scene(lights=scene.lights), ambient=.2))

    def test_scattering_and_absorption_split_the_extinction(self):
        scene = scene3d.Scene(volumes=(box(32, 3.0),))
        both = render(scene, replace(FLAT, scattering=1.0, absorption=1.0))[SIZE // 2, SIZE // 2]
        scatter = render(scene, replace(FLAT, scattering=1.0, absorption=0.0))[SIZE // 2, SIZE // 2]
        # Same extinction gives the same alpha only when the coefficients sum equal; here 2 vs 1.
        self.assertGreater(both[3], scatter[3])
        np.testing.assert_allclose(both[0], both[3] * .5, rtol=1e-4)       # half of the extinction scatters
        tint = render(scene, replace(FLAT, color=(1.0, .5, 0.0)))[SIZE // 2, SIZE // 2]
        np.testing.assert_allclose(tint[:3], scatter[:3] * (1.0, .5, 0.0), atol=1e-5)

    def test_the_lit_side_is_brighter_than_the_far_side(self):
        light = scene3d.Light("Directional", position=scene3d.Vec3(6, 0, 0), target=scene3d.Vec3())
        scene = scene3d.Scene(volumes=(box(32, 4.0),), lights=(light,))
        settings = replace(FLAT, absorption=1.0, shadow_steps=24)
        image = render(scene, settings, ambient=0.0)
        row = SIZE // 2
        near_light, far_side = (image[row, col, :3].sum() for col in (SIZE // 2 + 4, SIZE // 2 - 4))
        # The camera looks along -z; +x is to the right, so the light comes from the right-hand side.
        self.assertGreater(near_light, far_side * 1.5)
        # With shadows off (zero shadow density) both sides are equally bright.
        flat = render(scene, replace(settings, shadow_density=0.0), ambient=0.0)
        np.testing.assert_allclose(flat[row, SIZE // 2 + 4, :3].sum(), flat[row, SIZE // 2 - 4, :3].sum(), rtol=.05)

    def test_a_spot_aimed_away_leaves_the_volume_dark(self):
        away = scene3d.Light("Spot", position=scene3d.Vec3(0, 0, 4), target=scene3d.Vec3(0, 0, 9),
                             cone_angle=30, cone_penumbra_angle=5)
        toward = replace(away, target=scene3d.Vec3(0, 0, 0))
        smoke = box(16, 2.0)
        dark = render(scene3d.Scene(volumes=(smoke,), lights=(away,)), ambient=0.0)
        lit = render(scene3d.Scene(volumes=(smoke,), lights=(toward,)), ambient=0.0)
        self.assertEqual(float(np.abs(dark[..., :3]).max()), 0.0)
        self.assertGreater(float(dark[..., 3].max()), .5)         # the smoke is still there, just unlit
        self.assertGreater(float(lit[SIZE // 2, SIZE // 2, :3].sum()), .05)

    def test_light_cone_and_falloff_reach_the_volume(self):
        near = scene3d.Light("Point", (1, 1, 1), 1.0, scene3d.Vec3(0, 0, 3), falloff_type="Quadratic")
        far = replace(near, position=scene3d.Vec3(0, 0, 8))
        smoke = scene3d.Scene(volumes=(box(16, .5),))
        bright = render(replace(smoke, lights=(near,)), replace(FLAT, absorption=1.0), ambient=0.0)
        dim = render(replace(smoke, lights=(far,)), replace(FLAT, absorption=1.0), ambient=0.0)
        self.assertGreater(float(bright[SIZE // 2, SIZE // 2, :3].sum()), 3 * float(dim[SIZE // 2, SIZE // 2, :3].sum()))

    def test_a_card_in_front_hides_the_volume_and_one_behind_is_dimmed(self):
        smoke = box(16, 2.0)
        alone = render(scene3d.Scene(volumes=(smoke,)))
        centre = (SIZE // 2, SIZE // 2)
        front = render(scene3d.Scene((card(2.0),), volumes=(smoke,)))
        np.testing.assert_allclose(front[centre], (.2, .4, .8, 1.0), atol=1e-6)
        behind = render(scene3d.Scene((card(-2.0),), volumes=(smoke,)))
        a = float(alone[centre + (3,)])
        expected = alone[centre] + (1 - a) * np.array((.2, .4, .8, 1.0))
        np.testing.assert_allclose(behind[centre], expected, atol=1e-5)
        self.assertLess(float(behind[centre + (0,)]), .2 + a)             # dimmed: the card shows less than it would bare

    def test_a_card_inside_the_volume_cuts_the_ray(self):
        smoke = box(32, 2.0)
        full = render(scene3d.Scene(volumes=(smoke,)))[SIZE // 2, SIZE // 2, 3]
        half = render(scene3d.Scene((card(0.0, (0, 0, 0, 1)),), volumes=(smoke,)))[SIZE // 2, SIZE // 2]
        # Only the front half of the smoke lies in front of the card; black card, so rgb is all smoke.
        self.assertAlmostEqual(float(half[3]), 1.0, places=6)
        front_alpha = 1 - math.exp(-2.0 * .5)
        self.assertAlmostEqual(float(half[0]), front_alpha, delta=.01)
        self.assertLess(front_alpha, full)

    def test_rendering_is_deterministic_and_chunk_independent(self):
        scene = scene3d.Scene((card(-1.0),), volumes=(scene3d.analytic_plume(16, 2),),
                              lights=(scene3d.Light(shadows=True),))
        first, second = render(scene, ambient=.1), render(scene, ambient=.1)
        np.testing.assert_array_equal(first, second)
        old = volumerender.RAY_CHUNK
        volumerender.RAY_CHUNK = 37
        try:
            np.testing.assert_allclose(render(scene, ambient=.1), first, atol=1e-6)
        finally:
            volumerender.RAY_CHUNK = old

    def test_supersampling_still_renders_volumes_once(self):
        scene = scene3d.Scene(volumes=(box(16, 1.0),))
        image = scene3d.render(scene, CAMERA, 24, 24, samples=2, volume=FLAT)
        self.assertEqual(image.shape, (24, 24, 4))
        self.assertGreater(float(image[12, 12, 3]), .3)

    def test_the_budget_refuses_a_frame_it_cannot_finish(self):
        old = volumerender.SAMPLE_BUDGET
        volumerender.SAMPLE_BUDGET = 1000
        try:
            with self.assertRaisesRegex(ValueError, "budget"):
                render(scene3d.Scene(volumes=(box(),), lights=(scene3d.Light(),)))
        finally:
            volumerender.SAMPLE_BUDGET = old

    def test_a_moved_volume_moves_on_screen(self):
        centre = np.eye(4, dtype=np.float32)
        centre[0, 3] = 1.0
        image = render(scene3d.Scene(volumes=(box(16, 2.0, matrix=centre),)))
        columns = np.nonzero(image[..., 3].max(axis=0) > .1)[0]
        self.assertGreater(columns.mean(), SIZE / 2 + 4)


class Render3DTests(unittest.TestCase):
    def graph(self, **render):
        d = Dispatcher()
        make(d, p=("Plume3D", {"plume_resolution": 12}), s=("Scene3D", {}), c=("Camera3D", {"ty": .5, "tz": 3, "target_y": .5}),
             r=("Render3D", {"width": 32, "height": 32, "samples": 1, "volume_density_scale": 4.0, **render}))
        wire(d, "s", "object0", "p")
        wire(d, "r", "scene", "s")
        wire(d, "r", "camera", "c")
        return d

    def image(self, d):
        return Evaluator().evaluate(dict(d.document, view="r"), frame=1)

    def test_render3d_knobs_are_wired_to_the_raymarch(self):
        base = self.image(self.graph())
        self.assertGreater(float(base[..., 3].max()), .05)
        dense = self.image(self.graph(volume_density_scale=8.0))
        self.assertGreater(float(dense[..., 3].max()), float(base[..., 3].max()))
        none = self.image(self.graph(volume_density_scale=0.0))
        self.assertEqual(float(none[..., 3].max()), 0.0)
        tinted = self.image(self.graph(volume_red=1.0, volume_green=0.0, volume_blue=0.0))
        self.assertEqual(float(tinted[..., 1:3].max()), 0.0)
        self.assertGreater(float(tinted[..., 0].max()), 0.0)

    def test_volumes_off_ignores_them(self):
        off = self.image(self.graph(volumes="off"))
        self.assertEqual(float(np.abs(off).max()), 0.0)

    def test_the_gpu_backend_refuses_volumes_and_auto_falls_back(self):
        d = self.graph(render_backend="gpu")
        with self.assertRaisesRegex(ValueError, "volumes are CPU-only"):
            self.image(d)
        cpu = self.image(self.graph(render_backend="cpu"))
        d.execute({"op": "set", "id": "r", "param": "render_backend", "value": "auto"})
        np.testing.assert_array_equal(self.image(d), cpu)

    def test_gpu3d_render_raises_unsupported_for_a_volume_scene(self):
        with self.assertRaisesRegex(gpu3d.Unsupported, "volumes are CPU-only until lane 4 wires them"):
            gpu3d.render(scene3d.Scene(volumes=(box(4),)), CAMERA, 8, 8)
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene3d.Scene(), CAMERA, 8, 8, output="volume_density")

    def test_old_documents_gain_the_volume_knobs(self):
        d = self.graph()
        doc = d.document
        for name in ("volumes", "volume_step_size", "volume_density_scale", "volume_shadow_density",
                     "volume_shadow_steps", "volume_scattering", "volume_absorption", "volume_red",
                     "volume_fps", "volume_depth_threshold"):
            del doc["nodes"]["r"]["params"][name]
        from nodebased.core import upgrade_document
        upgraded = upgrade_document(doc)
        self.assertEqual(upgraded["nodes"]["r"]["params"]["volumes"], "on")
        self.assertEqual(upgraded["nodes"]["r"]["params"]["volume_step_size"], 0.05)
