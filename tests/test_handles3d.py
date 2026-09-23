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


# A camera off every principal axis, so no gizmo arrow projects to a degenerate zero-length
# segment (CAMERA above looks straight down Z, which the gizmo tests below must avoid).
ANGLED_CAMERA = s.Camera(s.Transform3D(s.Vec3(5.0, 4.0, 6.0)), s.Vec3(), 45.0, 0.1, 1000.0)


class PivotWorldPositionTests(unittest.TestCase):
    def test_identity_parent_adds_translate_and_pivot(self):
        params = {"tx": 3.0, "ty": -1.0, "tz": 2.0, "pivot_x": 0.5, "pivot_y": 0.0, "pivot_z": -0.5}
        np.testing.assert_allclose(h.pivot_world_position(params, np.eye(4)), (3.5, -1.0, 1.5))

    def test_parent_transform_is_applied(self):
        params = {"tx": 1.0, "ty": 0.0, "tz": 0.0, "pivot_x": 0.0, "pivot_y": 0.0, "pivot_z": 0.0}
        parent = np.eye(4)
        parent[:3, 3] = (10.0, 20.0, 30.0)
        parent[:3, :3] *= 2.0  # a uniform-2x parent scale
        np.testing.assert_allclose(h.pivot_world_position(params, parent), (12.0, 20.0, 30.0))

    def test_missing_pivot_keys_default_to_zero(self):
        params = {"tx": 1.0, "ty": 2.0, "tz": 3.0}
        np.testing.assert_allclose(h.pivot_world_position(params, np.eye(4)), (1.0, 2.0, 3.0))


class GizmoScaleTests(unittest.TestCase):
    def test_none_bounds_uses_a_default(self):
        self.assertEqual(h.gizmo_scale(None), 1.0)

    def test_scales_with_the_bounds_diagonal(self):
        self.assertAlmostEqual(h.gizmo_scale(((0, 0, 0), (2, 0, 0))), 1.0)

    def test_never_smaller_than_the_minimum(self):
        self.assertEqual(h.gizmo_scale(((0, 0, 0), (0.01, 0, 0))), h.MIN_GIZMO_SCALE)


class GizmoGeometryTests(unittest.TestCase):
    def test_arrows_are_inset_from_the_pivot_not_touching_it(self):
        origin = np.array((1.0, 2.0, 3.0))
        for start, end in h.gizmo_arrows(origin, 2.0).values():
            self.assertGreater(np.linalg.norm(start - origin), 0.0)
            self.assertGreater(np.linalg.norm(end - origin), np.linalg.norm(start - origin))

    def test_planes_sit_clear_of_the_pivot(self):
        origin = np.array((0.0, 0.0, 0.0))
        for corners in h.gizmo_planes(origin, 2.0).values():
            for corner in corners:
                self.assertGreater(np.linalg.norm(np.asarray(corner) - origin), 0.5)

    def test_plane_normals_match_their_named_axes(self):
        # The xy square's corners must all share the same z (its normal is Z), etc.
        origin = np.array((0.0, 0.0, 0.0))
        for name, corners in h.gizmo_planes(origin, 2.0).items():
            normal = h.PLANE_NORMALS[name]
            axis_index = int(np.argmax(np.abs(normal)))
            values = {round(float(c[axis_index]), 9) for c in corners}
            self.assertEqual(len(values), 1)


class AxisPlaneDragPointTests(unittest.TestCase):
    def test_axis_drag_point_recovers_a_known_point_on_the_axis(self):
        origin, axis = np.array((0.0, 0.0, 0.0)), h.Z_AXIS
        target = origin + axis * 1.75
        xy, _z = s.project(ANGLED_CAMERA, WIDTH, HEIGHT, target[None])
        point = h.axis_drag_point(ANGLED_CAMERA, WIDTH, HEIGHT, origin, axis, (xy[0, 0], xy[0, 1]))
        np.testing.assert_allclose(point, target, atol=1e-3)

    def test_axis_drag_point_ignores_off_axis_camera_motion(self):
        # A point exactly on the axis always maps back onto that same axis, regardless of which
        # pixel on the projected line it came from -- the property a "drag along X" handle needs.
        origin, axis = np.array((1.0, -2.0, 0.5)), h.X_AXIS
        for t in (-1.0, 0.5, 3.0):
            point = origin + axis * t
            xy, _z = s.project(ANGLED_CAMERA, WIDTH, HEIGHT, point[None])
            recovered = h.axis_drag_point(ANGLED_CAMERA, WIDTH, HEIGHT, origin, axis, (xy[0, 0], xy[0, 1]))
            # The recovered point must itself lie on the axis line (a pure t*axis + origin).
            offset = recovered - origin
            cross = np.cross(offset, axis)
            self.assertLess(float(np.linalg.norm(cross)), 1e-6)

    def test_plane_drag_point_recovers_a_known_point_in_the_plane(self):
        origin, normal = np.array((0.0, 0.0, 0.0)), h.Z_AXIS
        target = np.array((1.2, -0.8, 0.0))  # in the z=0 plane
        xy, _z = s.project(ANGLED_CAMERA, WIDTH, HEIGHT, target[None])
        point = h.plane_drag_point(ANGLED_CAMERA, WIDTH, HEIGHT, origin, normal, (xy[0, 0], xy[0, 1]))
        np.testing.assert_allclose(point, target, atol=1e-3)


class DeltaConversionTests(unittest.TestCase):
    def test_world_to_local_delta_identity_parent_is_a_no_op(self):
        delta = np.array((1.0, 2.0, 3.0))
        np.testing.assert_allclose(h.world_to_local_delta(delta, np.eye(3)), delta)

    def test_world_to_local_delta_inverts_a_scaled_parent(self):
        parent_linear = np.diag((2.0, 1.0, 4.0))
        world_delta = np.array((4.0, 3.0, 8.0))
        local = h.world_to_local_delta(world_delta, parent_linear)
        np.testing.assert_allclose(parent_linear @ local, world_delta, atol=1e-9)

    def test_pivot_param_deltas_match_the_derivation(self):
        # own_linear = 2x uniform scale: position_delta = (linear - I) @ pivot_delta = pivot_delta.
        own_linear = np.eye(3) * 2.0
        pivot_delta = np.array((1.0, 0.0, 0.0))
        moved_pivot, position_delta = h.pivot_param_deltas(pivot_delta, own_linear)
        np.testing.assert_allclose(moved_pivot, pivot_delta)
        np.testing.assert_allclose(position_delta, pivot_delta)

    def test_pivot_param_deltas_are_zero_for_an_identity_transform(self):
        # No rotation or scale: moving the pivot needs no translate compensation at all.
        _pivot_delta, position_delta = h.pivot_param_deltas(np.array((1.0, 2.0, 3.0)), np.eye(3))
        np.testing.assert_allclose(position_delta, (0.0, 0.0, 0.0))


class GizmoHitTests(unittest.TestCase):
    def test_hit_on_the_x_arrow_shaft(self):
        origin, scale = np.array((0.0, 0.0, 0.0)), 2.0
        start, end = h.gizmo_arrows(origin, scale)["x"]
        midpoint = (np.asarray(start) + np.asarray(end)) / 2
        xy, _z = s.project(ANGLED_CAMERA, WIDTH, HEIGHT, midpoint[None])
        hit = h.gizmo_hit(ANGLED_CAMERA, WIDTH, HEIGHT, origin, scale, (xy[0, 0], xy[0, 1]))
        self.assertEqual(hit, ("axis", "x"))

    def test_hit_on_the_xy_plane_square(self):
        origin, scale = np.array((0.0, 0.0, 0.0)), 2.0
        corners = h.gizmo_planes(origin, scale)["xy"]
        center = np.mean(np.asarray(corners), axis=0)
        xy, _z = s.project(ANGLED_CAMERA, WIDTH, HEIGHT, center[None])
        hit = h.gizmo_hit(ANGLED_CAMERA, WIDTH, HEIGHT, origin, scale, (xy[0, 0], xy[0, 1]))
        self.assertEqual(hit, ("plane", "xy"))

    def test_click_at_the_pivot_itself_hits_nothing(self):
        # The dead zone around the pivot: clicking exactly on the object must fall through to
        # picking/orbit, not an arbitrary axis choice among three that all start there.
        origin, scale = np.array((0.0, 0.0, 0.0)), 2.0
        xy, _z = s.project(ANGLED_CAMERA, WIDTH, HEIGHT, origin[None])
        hit = h.gizmo_hit(ANGLED_CAMERA, WIDTH, HEIGHT, origin, scale, (xy[0, 0], xy[0, 1]))
        self.assertIsNone(hit)

    def test_click_far_from_the_gizmo_hits_nothing(self):
        origin, scale = np.array((0.0, 0.0, 0.0)), 2.0
        hit = h.gizmo_hit(ANGLED_CAMERA, WIDTH, HEIGHT, origin, scale, (5, 5))
        self.assertIsNone(hit)

    def test_an_axis_end_on_to_the_camera_is_not_a_hit_target(self):
        # CAMERA (module-level) looks straight down Z: the z-arrow foreshortens to a point, and
        # must not swallow clicks anywhere near the pivot (see the orbit-preserving guard).
        origin, scale = np.array((0.0, 0.0, 2.0)), 1.4142135623730951
        hit = h.gizmo_hit(CAMERA, WIDTH, HEIGHT, origin, scale, (WIDTH / 2, HEIGHT / 2))
        self.assertIsNone(hit)


if __name__ == "__main__":
    unittest.main()
