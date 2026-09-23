"""Qt-free tests for 3D viewport picking: screen-to-ray, node attribution, nearest-hit."""
import unittest

import numpy as np

from nodebased import handles3d as h
from nodebased import scene3d as s
from nodebased.core import Dispatcher


def _card_graph():
    """Two Card3D nodes ('near' at z=2, 'far' at z=-2) grouped under a Scene3D, wired
    into a Render3D so `resolve_geometries` has to walk through the grouping node."""
    d = Dispatcher()
    for key, kind in (("near", "Card3D"), ("far", "Card3D"), ("scene", "Scene3D"),
                      ("cam", "Camera3D"), ("render", "Render3D")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "set", "id": "near", "param": "tz", "value": 2.0})
    d.execute({"op": "set", "id": "far", "param": "tz", "value": -2.0})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "near"})
    d.execute({"op": "connect", "id": "scene", "input": "object1", "source": "far"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    d.execute({"op": "view", "id": "render"})
    return d.document


CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 10)), s.Vec3(), 45.0, 0.1, 1000.0)
WIDTH, HEIGHT = 640, 360


class ScreenToRayTests(unittest.TestCase):
    def test_round_trips_through_scene3d_project(self):
        points = np.array(((0, 0, 0), (1, 2, -1), (-2, 0.5, 3), (0.3, -1.4, -4)), np.float64)
        xy, _z = s.project(CAMERA, WIDTH, HEIGHT, points)
        for point, pixel in zip(points, xy):
            origin, direction = h.screen_to_ray(CAMERA, WIDTH, HEIGHT, pixel[0], pixel[1])
            t = float(np.dot(point - origin, direction))
            closest = origin + direction * t
            self.assertLess(np.linalg.norm(closest - point), 1e-4)

    def test_direction_is_a_unit_vector(self):
        _origin, direction = h.screen_to_ray(CAMERA, WIDTH, HEIGHT, 100, 40)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=9)

    def test_origin_is_the_camera_eye(self):
        origin, _direction = h.screen_to_ray(CAMERA, WIDTH, HEIGHT, 320, 180)
        np.testing.assert_allclose(origin, (0, 0, 10), atol=1e-9)


class ResolveGeometriesTests(unittest.TestCase):
    def test_walks_through_scene3d_to_both_cards(self):
        document = _card_graph()
        render = document["nodes"]["render"]
        candidates = h.resolve_geometries(document, render["inputs"]["scene"])
        self.assertEqual({key for key, _geometry in candidates}, {"near", "far"})

    def test_disabled_card_is_not_a_candidate(self):
        document = _card_graph()
        document["nodes"]["far"]["disabled"] = True
        render = document["nodes"]["render"]
        candidates = h.resolve_geometries(document, render["inputs"]["scene"])
        self.assertEqual({key for key, _geometry in candidates}, {"near"})

    def test_disabled_scene3d_contributes_nothing(self):
        document = _card_graph()
        document["nodes"]["scene"]["disabled"] = True
        render = document["nodes"]["render"]
        candidates = h.resolve_geometries(document, render["inputs"]["scene"])
        self.assertEqual(candidates, [])

    def test_world_matrix_carries_the_scene3d_parent(self):
        document = _card_graph()
        document["nodes"]["scene"]["params"]["tx"] = 5.0
        render = document["nodes"]["render"]
        candidates = dict(h.resolve_geometries(document, render["inputs"]["scene"]))
        matrix = candidates["near"].world_matrix()
        np.testing.assert_allclose(matrix[:3, 3], (5.0, 0.0, 2.0), atol=1e-5)

    def test_loose_geometries_matches_the_unwired_fallback(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "card", "type": "Card3D"})
        candidates = h.loose_geometries(d.document)
        self.assertEqual([key for key, _geometry in candidates], ["card"])


class PickTests(unittest.TestCase):
    def test_nearer_of_two_overlapping_cards_wins(self):
        document = _card_graph()
        render = document["nodes"]["render"]
        candidates = h.resolve_geometries(document, render["inputs"]["scene"])
        xy, _z = s.project(CAMERA, WIDTH, HEIGHT, ((0, 0, 2),))
        hit = h.pick(candidates, CAMERA, WIDTH, HEIGHT, xy[0, 0], xy[0, 1])
        self.assertEqual(hit[0], "near")

    def test_empty_space_picks_nothing(self):
        document = _card_graph()
        render = document["nodes"]["render"]
        candidates = h.resolve_geometries(document, render["inputs"]["scene"])
        hit = h.pick(candidates, CAMERA, WIDTH, HEIGHT, 5, 5)
        self.assertIsNone(hit)

    def test_bounds_edges_count_and_extent(self):
        edges = h.bounds_edges(((0, 0, 0), (1, 2, 3)))
        self.assertEqual(len(edges), 12)
        xs = {c[0] for edge in edges for c in edge}
        self.assertEqual(xs, {0, 1})


if __name__ == "__main__":
    unittest.main()
