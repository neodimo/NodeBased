"""Light3D spot cone and distance falloff: the light model and `light_attenuation`.

The CPU renderer does not apply the factor yet (that hook belongs to lane 4; see
docs/3D_FOUNDATION.md), so these tests pin the model itself."""
import copy
import math
import unittest

import numpy as np

from nodebased import scene3d
from nodebased.core import CHOICES, Dispatcher, LIMITS, SPECS, upgrade_document, validate
from nodebased.scene3d import Light, Vec3, light_attenuation


def spot(z=4.0, cone=40.0, penumbra=10.0, falloff=1.0, distance="No falloff"):
    return Light("Spot", position=Vec3(0, 0, z), target=Vec3(0, 0, 0), cone_angle=cone,
                 cone_penumbra_angle=penumbra, cone_falloff=falloff, falloff_type=distance)


def at_angle(degrees, z=4.0):
    """A point on the plane z=0 that sits `degrees` off the axis of a spot at (0, 0, z)."""
    return (z * math.tan(math.radians(degrees)), 0.0, 0.0)


class ConeTests(unittest.TestCase):
    def test_one_on_the_axis_and_zero_well_outside(self):
        light = spot()
        self.assertAlmostEqual(light_attenuation(light, (0, 0, 0)), 1.0, places=6)
        self.assertEqual(light_attenuation(light, (0, 0, 2.5)), 1.0)      # nearer along the axis
        self.assertEqual(light_attenuation(light, (50, 0, 0)), 0.0)
        self.assertEqual(light_attenuation(light, (0, 0, 40)), 0.0)       # behind the light

    def test_full_inside_the_inner_cone_and_zero_beyond_the_penumbra(self):
        light = spot()   # inner half angle 20 degrees, penumbra out to 30
        self.assertAlmostEqual(light_attenuation(light, at_angle(19.9)), 1.0, places=6)
        self.assertAlmostEqual(light_attenuation(light, at_angle(30.1)), 0.0, places=6)

    def test_penumbra_midpoint_is_the_smoothstep_half(self):
        self.assertAlmostEqual(light_attenuation(spot(), at_angle(25.0)), 0.5, places=5)
        # cone_falloff raises the fade: 0.5 ** 2 at the midpoint, 0.5 ** 0.5 for a softer one.
        self.assertAlmostEqual(light_attenuation(spot(falloff=2.0), at_angle(25.0)), 0.25, places=5)
        self.assertAlmostEqual(light_attenuation(spot(falloff=0.5), at_angle(25.0)), 0.5 ** 0.5, places=5)

    def test_monotonic_across_the_penumbra(self):
        light = spot()
        angles = np.linspace(0.0, 45.0, 181)
        values = light_attenuation(light, np.array([at_angle(a) for a in angles]))
        self.assertTrue(np.all(np.diff(values) <= 1e-7))
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[-1], 0.0)
        self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_no_penumbra_is_a_hard_edge(self):
        light = spot(penumbra=0.0)
        self.assertEqual(light_attenuation(light, at_angle(19.0)), 1.0)
        self.assertEqual(light_attenuation(light, at_angle(21.0)), 0.0)

    def test_batch_matches_single_points(self):
        light = spot()
        points = np.array([(0, 0, 0), at_angle(25.0), (50, 0, 0)])
        batch = light_attenuation(light, points)
        self.assertEqual(batch.shape, (3,))
        for point, value in zip(points, batch):
            self.assertAlmostEqual(float(value), light_attenuation(light, point), places=6)


class FalloffTests(unittest.TestCase):
    def test_distance_falloff_matches_hand_values_at_two_distances(self):
        # Light at z=2 aimed at the origin; the origin is 2 away, z=-2 is 4 away.
        expected = {"No falloff": (1.0, 1.0), "Linear": (1 / 2, 1 / 4),
                    "Quadratic": (1 / 4, 1 / 16), "Cubic": (1 / 8, 1 / 64)}
        for kind in ("Point", "Spot"):
            for name, (near, far) in expected.items():
                light = Light(kind, position=Vec3(0, 0, 2), target=Vec3(), cone_angle=60.0,
                              falloff_type=name)
                self.assertAlmostEqual(light_attenuation(light, (0, 0, 0)), near, places=6, msg=(kind, name))
                self.assertAlmostEqual(light_attenuation(light, (0, 0, -2)), far, places=6, msg=(kind, name))

    def test_falloff_never_brightens_a_close_light(self):
        light = Light("Point", position=Vec3(0, 0, 2), falloff_type="Quadratic")
        self.assertEqual(light_attenuation(light, (0, 0, 1.5)), 1.0)   # 0.5 away: 1/0.25 capped at 1

    def test_spot_multiplies_cone_by_falloff(self):
        light = spot(z=4.0, distance="Quadratic")
        point = at_angle(25.0)
        distance = math.hypot(point[0], 4.0)
        self.assertAlmostEqual(light_attenuation(light, point), 0.5 / distance ** 2, places=5)

    def test_directional_light_is_unaffected(self):
        for name in scene3d.FALLOFF_TYPES:
            light = Light("Directional", position=Vec3(2, 4, 3), target=Vec3(), falloff_type=name,
                          cone_angle=1.0, cone_penumbra_angle=0.0)
            for point in ((0, 0, 0), (100, -50, 30), (0.01, 0, 0)):
                self.assertEqual(light_attenuation(light, point), 1.0)
        self.assertTrue(np.all(light_attenuation(light, np.zeros((5, 3))) == 1.0))

    def test_parent_transform_moves_the_cone(self):
        parent = np.eye(4, dtype=np.float32)
        parent[0, 3] = 10.0
        light = Light("Spot", position=Vec3(0, 0, 4), target=Vec3(), cone_angle=20.0,
                      cone_penumbra_angle=0.0, parent=parent)
        self.assertEqual(light_attenuation(light, (10, 0, 0)), 1.0)
        self.assertEqual(light_attenuation(light, (0, 0, 0)), 0.0)


class KnobAndDocumentTests(unittest.TestCase):
    def test_light3d_knobs_choices_and_limits(self):
        params = SPECS["Light3D"]["params"]
        self.assertEqual(CHOICES["light_type"], ["Directional", "Point", "Spot"])
        self.assertEqual(CHOICES["falloff_type"], ["No falloff", "Linear", "Quadratic", "Cubic"])
        self.assertEqual(scene3d.LIGHT_TYPES, tuple(CHOICES["light_type"]))
        self.assertEqual(scene3d.FALLOFF_TYPES, tuple(CHOICES["falloff_type"]))
        for name in ("cone_angle", "cone_penumbra_angle", "cone_falloff"):
            self.assertIn(name, LIMITS)
            self.assertIn(name, params)
        self.assertEqual(params["falloff_type"], "No falloff")

    def test_light_from_node_reads_the_spot_knobs(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "l", "type": "Light3D", "params": {
            "light_type": "Spot", "cone_angle": 50.0, "cone_penumbra_angle": 12.0,
            "cone_falloff": 2.0, "falloff_type": "Cubic"}})
        light = scene3d.light_from_node(d.document["nodes"]["l"])
        self.assertEqual((light.kind, light.cone_angle, light.cone_penumbra_angle,
                          light.cone_falloff, light.falloff_type), ("Spot", 50.0, 12.0, 2.0, "Cubic"))

    def test_old_light_document_loads_with_defaults(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "l", "type": "Light3D", "params": {"light_type": "Point"}})
        old = copy.deepcopy(d.document)
        for name in ("cone_angle", "cone_penumbra_angle", "cone_falloff", "falloff_type"):
            del old["nodes"]["l"]["params"][name]
        upgraded = upgrade_document(old)
        validate(upgraded)
        light = scene3d.light_from_node(upgraded["nodes"]["l"])
        self.assertEqual((light.kind, light.falloff_type), ("Point", "No falloff"))
        self.assertEqual(light_attenuation(light, (5, 5, 5)), 1.0)


if __name__ == "__main__":
    unittest.main()
