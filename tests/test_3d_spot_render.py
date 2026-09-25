"""Spot cone and distance falloff applied by every renderer (lane L4 step A).

The model is `scene3d.light_attenuation` (tests/test_3d_spot_light.py); these tests render a lit
Card3D and check the CPU reference against hand-computed factors, then each GPU path against the CPU."""
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, scene3d as s
from nodebased.scene3d import Light, Vec3

SIZE = (65, 49)            # odd, so the centre pixel sits exactly on the axis
WHITE = (1.0, 1.0, 1.0, 1.0)


def card():
    return s._card(4, 4, WHITE, s.Transform3D())


def spot(**kw):
    args = dict(kind="Spot", position=Vec3(0, 0, 3), target=Vec3(0, 0, 0), cone_angle=20.0,
                cone_penumbra_angle=8.0)
    args.update(kw)
    return Light(**args)


def render(light, mode=None, size=SIZE, **kw):
    scene = s.Scene((card(),), (light,))
    if mode is None:
        return s.render(scene, s.Camera(), *size, ambient=0.0, **kw)
    return gpu3d.render(scene, s.Camera(), *size, ambient=0.0, mode=mode, **kw)


def luminance(image):
    return image[..., :3].mean(axis=-1)


class CpuSpotTests(unittest.TestCase):
    def test_bright_at_the_centre_dark_outside_the_cone(self):
        image = render(spot())
        h, w = image.shape[:2]
        self.assertGreater(image[h // 2, w // 2, 0], 0.95)
        # The card's top-left corner is about 34 degrees off axis, well past 14 degrees.
        corner = image[int(h * .32), int(w * .30), 0]
        self.assertEqual(corner, 0.0)
        # An unshaped Directional light lights that corner, so the cone is what darkens it.
        flat = render(Light("Directional", position=Vec3(0, 0, 3), target=Vec3()))
        self.assertGreater(flat[int(h * .32), int(w * .30), 0], 0.5)

    def test_penumbra_is_a_monotonic_ramp(self):
        row = luminance(render(spot(cone_penumbra_angle=10.0), size=(129, 49)))[24, 64:]
        self.assertTrue(np.all(np.diff(row) <= 1e-6), "brightness never rises going outward")
        ramp = row[(row > 0.02) & (row < 0.98)]
        self.assertGreaterEqual(len(np.unique(np.round(ramp, 4))), 4, "the edge is a ramp, not a step")
        self.assertEqual(row[-1], 0.0)

    def test_distance_falloff_matches_the_hand_computed_factor(self):
        # Card centre, light straight above at distance d: lambert 1, colour 1, so pixel == factor.
        for falloff, power in (("Linear", 1), ("Quadratic", 2), ("Cubic", 3)):
            for d in (2.0, 4.0):
                for kind in ("Point", "Spot"):
                    light = Light(kind, position=Vec3(0, 0, d), target=Vec3(), cone_angle=90.0,
                                  falloff_type=falloff)
                    image = render(light)
                    self.assertAlmostEqual(float(image[24, 32, 0]), d ** -power, places=5,
                                           msg=f"{kind} {falloff} at {d}")

    def test_falloff_never_brightens_inside_one_unit(self):
        light = Light("Point", position=Vec3(0, 0, 0.5), target=Vec3(), falloff_type="Quadratic")
        self.assertLessEqual(float(render(light)[24, 32, 0]), 1.0 + 1e-6)

    def test_spot_shadows_use_the_position_direction(self):
        blocker = s._card(1, 1, WHITE, s.Transform3D(position=Vec3(0, 0, 1.5)))
        scene = s.Scene((card(), blocker), (spot(shadows=True, cone_angle=60.0),))
        shadowed = s.render(scene, s.Camera(), *SIZE, ambient=0.0, shadows=True)
        # The blocker sits between the light and the centre of the card and hides it from the light.
        # (The blocker itself is what the camera sees there; look just past it, where the card is lit.)
        lit = s.render(s.Scene((card(),), scene.lights), s.Camera(), *SIZE, ambient=0.0)
        self.assertGreater(float(lit[24, 32, 0]), 0.9)
        self.assertGreater(float(lit[24, 44, 0]), 0.0)
        self.assertAlmostEqual(float(shadowed[24, 44, 0]), float(lit[24, 44, 0]), places=3)

    def test_directional_and_default_point_render_identically_to_before(self):
        # `_light_factor` returning None for these is the whole guarantee; force it and compare bytes.
        lights = (Light("Directional", position=Vec3(2, 3, 4), target=Vec3(), color=(1, .8, .6), intensity=1.3),
                  Light("Point", position=Vec3(1, 1, 3), target=Vec3(), intensity=1.5))
        for light in lights:
            for kwargs in ({}, {"mode": "raytrace"}):
                scene = s.Scene((card(),), (light,))
                now = s.render(scene, s.Camera(), *SIZE, ambient=.1, **kwargs)
                with patch.object(s, "_light_factor", lambda *_: None):
                    before = s.render(scene, s.Camera(), *SIZE, ambient=.1, **kwargs)
                self.assertEqual(now.tobytes(), before.tobytes(), light.kind)
            self.assertIsNone(s._light_factor(light, np.zeros((1, 3))))


class SplatSpotTests(unittest.TestCase):
    def test_relit_splats_take_the_cone_and_falloff(self):
        from nodebased.splatshade import shade_splats
        positions = np.array(((0, 0, 0), (2.5, 0, 0), (0, 0, 0)), float)   # on axis, outside cone, on axis
        normals = np.array((0, 0, 1.0))[None].repeat(3, 0)
        args = (np.zeros((3, 3)), np.ones((3, 3)), positions, normals, np.ones(3), (0, 0, 5), )
        lit = shade_splats(*args, [spot()], 0.0, 1.0)
        self.assertAlmostEqual(float(lit[0, 0]), 1.0, places=5)
        self.assertEqual(float(lit[1, 0]), 0.0)
        far = shade_splats(*args, [spot(position=Vec3(0, 0, 4), falloff_type="Quadratic")], 0.0, 1.0)
        self.assertAlmostEqual(float(far[0, 0]), 1 / 16, places=5)


@unittest.skipUnless(gpu3d.available(), "wgpu adapter unavailable")
class GpuSpotTests(unittest.TestCase):
    def check(self, light, mode, centre=1.0):
        cpu = render(light)
        try:
            gpu = render(light, mode)
        except gpu3d.Unsupported as exc:
            self.skipTest(str(exc))
        self.assertLess(float(np.abs(gpu - cpu).mean()), 5e-3)
        self.assertLess(float(np.abs(gpu - cpu).max()), 0.03)
        h, w = cpu.shape[:2]
        self.assertAlmostEqual(float(gpu[h // 2, w // 2, 0]), centre, delta=0.03)
        self.assertLess(gpu[int(h * .32), int(w * .30), 0], 0.02)

    def test_raster_and_raytrace_match_the_cpu_spot(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                self.check(spot(), mode)
                self.check(spot(cone_falloff=2.0, falloff_type="Linear", position=Vec3(0, 0, 2)), mode, centre=0.5)

    def test_gpu_falloff_types_match_the_cpu(self):
        for mode in ("raster", "raytrace"):
            for falloff in ("Linear", "Quadratic", "Cubic"):
                with self.subTest(mode=mode, falloff=falloff):
                    light = Light("Point", position=Vec3(0.5, 0.5, 3), target=Vec3(), falloff_type=falloff)
                    cpu, gpu = render(light), render(light, mode)
                    self.assertLess(float(np.abs(gpu - cpu).max()), 0.01)

    def test_gpu_penumbra_is_monotonic(self):
        for mode in ("raster", "raytrace"):
            row = luminance(render(spot(cone_penumbra_angle=10.0), mode, size=(129, 49)))[24, 64:]
            self.assertTrue(np.all(np.diff(row) <= 2e-3), mode)


if __name__ == "__main__":
    unittest.main()
