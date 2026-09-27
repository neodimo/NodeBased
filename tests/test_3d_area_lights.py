"""Rect/Disc/Sphere area lights (R2): real-size emitters, soft shadows from light-sample integration.

docs/3D_FOUNDATION.md "Area lights". `test_3d_shadows.py` and `test_3d_spot_light.py` pin the legacy
Directional/Point/Spot path this step must not disturb; this file only covers the new light kinds.
"""
import math
import unittest

import numpy as np

from nodebased import scene3d as s
from nodebased.core import CHOICES, LIMITS, SPECS
from nodebased.scene3d import Light, Vec3, light_from_node


def _analytic_rect_irradiance(radiance, half_width, half_height, distance, grid=2000):
    """Irradiance from a coaxial Lambertian rectangle directly above a point whose normal points
    straight at the rectangle's centre: E = integral over the rectangle of radiance * cos(light) *
    cos(surface) / distance^2 dA, both cosines equal to distance/r on this axis. A converged Riemann
    sum (the physical integral itself, not the renderer's own Monte Carlo estimator) is the analytic
    value here to within float precision; `grid` is chosen well past its own convergence."""
    a, b, d = half_width, half_height, distance
    xs = (np.arange(grid) + 0.5) / grid * 2 * a - a
    ys = (np.arange(grid) + 0.5) / grid * 2 * b - b
    r2 = xs[None, :] ** 2 + ys[:, None] ** 2 + d * d
    return float(radiance * (d * d / (r2 * r2)).sum() * (2 * a / grid) * (2 * b / grid))


class AreaLightRegistryTests(unittest.TestCase):
    def test_light_type_choices_and_limits(self):
        for kind in ("Rect", "Disc", "Sphere"):
            self.assertIn(kind, CHOICES["light_type"])
        for name in ("area_width", "area_height", "area_radius", "light_samples", "kelvin"):
            self.assertIn(name, LIMITS)
        self.assertIn("area_normalize", CHOICES)
        self.assertIn("two_sided", CHOICES)

    def test_light_from_node_defaults_without_new_keys(self):
        # An old document's Light3D params dict, before this step's setdefault migration ran.
        node = {"params": {"light_type": "Rect", "tx": 0.0, "ty": 3.0, "tz": 0.0,
                           "target_x": 0.0, "target_y": 0.0, "target_z": 0.0,
                           "red": 1.0, "green": 1.0, "blue": 1.0, "intensity": 1.0, "shadows": "off"}}
        light = light_from_node(node)
        self.assertEqual(light.kind, "Rect")
        self.assertEqual(light.area_width, 1.0)
        self.assertEqual(light.light_samples, 4)
        self.assertFalse(light.area_normalize)

    def test_kelvin_color_mode(self):
        node = {"params": {"light_type": "Sphere", "tx": 0.0, "ty": 0.0, "tz": 0.0,
                           "target_x": 0.0, "target_y": 0.0, "target_z": -1.0,
                           "red": 1.0, "green": 1.0, "blue": 1.0, "intensity": 1.0, "shadows": "off",
                           "light_color_mode": "Kelvin", "kelvin": 3000.0, "area_radius": 0.5}}
        warm = light_from_node(node)
        node["params"]["kelvin"] = 12000.0
        cool = light_from_node(node)
        # A warm (low Kelvin) light is redder than blue; a cool (high Kelvin) one is the opposite.
        self.assertGreater(warm.color[0] - warm.color[2], 0)
        self.assertLess(cool.color[0] - cool.color[2], 0)


class RectIrradianceTests(unittest.TestCase):
    """A Rect light directly above a point, both facing each other (coaxial): `_area_light_contribution`
    evaluated at the exact point, so the check is of the light-sample integration itself, not the
    rasterizer's pixel grid (a fixed camera and resolution would add its own discretization error on
    top of the one under test, tightest exactly where this light falls off fastest)."""

    def irradiance(self, half_width, half_height, distance, samples=64):
        light = Light("Rect", position=Vec3(0, distance, 0), target=Vec3(0, 0, 0),
                      area_width=2 * half_width, area_height=2 * half_height,
                      light_samples=samples, shadows=False)
        position = np.array([[0.0, 0.0, 0.0]], np.float32)
        normal = np.array([[0.0, 1.0, 0.0]], np.float32)
        result, _, _ = s._area_light_contribution(position, normal, light, None)
        return result[0]

    def test_matches_analytic_within_two_percent(self):
        for half_width, half_height, distance in ((1.0, 1.0, 3.0), (2.0, 0.5, 4.0), (0.3, 0.3, 1.5)):
            with self.subTest(half_width=half_width, half_height=half_height, distance=distance):
                got = self.irradiance(half_width, half_height, distance)
                expected = _analytic_rect_irradiance(1.0, half_width, half_height, distance)
                np.testing.assert_allclose(got, expected, rtol=0.02)

    def test_normalize_keeps_power_independent_of_size(self):
        # With `area_normalize` on, doubling the rect (4x the area) at the same distance redistributes
        # the same total power over a wider solid angle: the on-axis irradiance drops (radiance =
        # power / (area * pi)), unlike the un-normalized case where it grows with the rect's size.
        position = np.array([[0.0, 0.0, 0.0]], np.float32)
        normal = np.array([[0.0, 1.0, 0.0]], np.float32)
        light = Light("Rect", position=Vec3(0, 3, 0), target=Vec3(0, 0, 0),
                      area_width=1.0, area_height=1.0, area_normalize=True, light_samples=64, shadows=False)
        normalized, _, _ = s._area_light_contribution(position, normal, light, None)
        big_light = Light("Rect", position=Vec3(0, 3, 0), target=Vec3(0, 0, 0),
                          area_width=2.0, area_height=2.0, area_normalize=True, light_samples=64, shadows=False)
        big_normalized, _, _ = s._area_light_contribution(position, normal, big_light, None)
        self.assertGreater(normalized[0, 0], 0)
        self.assertLess(big_normalized[0, 0], normalized[0, 0])


class AreaShadowPenumbraTests(unittest.TestCase):
    """A card blocker under a Disc light: the penumbra (the pixel band between fully lit and fully
    shadowed) must widen as the light grows, matching similar-triangles geometry."""

    camera = s.Camera(s.Transform3D(s.Vec3(0, 6, 9)))
    ground = s._card(12, 12, (1, 1, 1, 1), s.Transform3D(rotation=s.Vec3(-90, 0, 0)))
    blocker = s._card(1, 1, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, 2, 0), s.Vec3(-90, 0, 0)))

    def _penumbra_width(self, radius, samples=48):
        # `area_normalize` keeps the total emitted power (not the radiance) fixed as the disc grows,
        # so the three radii stay comparably bright and only the shadow's softness changes with size.
        light = Light("Disc", position=Vec3(0, 5, 0), target=Vec3(0, 0, 0), intensity=20.0,
                      area_radius=radius, area_normalize=True, light_samples=samples, shadows=True)
        scene = s.Scene((self.ground, self.blocker), (light,))
        image = s.render(scene, self.camera, 96, 72, ambient=0.0)
        xy, _ = s.project(self.camera, 96, 72, [(x, 0.0, 0.0) for x in np.linspace(-2.5, 2.5, 200)])
        row = np.clip(np.floor(xy[:, 1]).astype(int), 0, 71)
        col = np.clip(np.floor(xy[:, 0]).astype(int), 0, 95)
        profile = image[row, col, 0]
        lit, dark = profile.max(), profile.min()
        band = (profile > dark + 0.1 * (lit - dark)) & (profile < dark + 0.9 * (lit - dark))
        return band.sum(), lit, dark

    def test_penumbra_grows_with_light_size(self):
        widths = []
        for radius in (0.05, 0.5, 1.5):
            band_pixels, lit, dark = self._penumbra_width(radius)
            self.assertGreater(lit - dark, 0.01, "the blocker must actually cast a visible shadow")
            widths.append(band_pixels)
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])


if __name__ == "__main__":
    unittest.main()
