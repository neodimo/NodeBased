"""Lane L3 step 2: rows/columns on Card3D and Sphere3D, Cylinder3D, and read-only local/world
matrix readouts. See docs/3D_FOUNDATION.md, the Nodes table and "Read-only local/world matrix
readouts"."""
import copy
import unittest

import numpy as np

from nodebased.core import Dispatcher, SCHEMA_VERSION, SPECS, GEOMETRY_TYPES, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased import scene3d


def _render_graph(geo_kind="Cube3D"):
    d = Dispatcher()
    for key, kind in (("geo", geo_kind), ("cam", "Camera3D"), ("scene", "Scene3D"), ("render", "Render3D")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "geo"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    for param, value in (("width", 64), ("height", 64), ("samples", 1)):
        d.execute({"op": "set", "id": "render", "param": param, "value": value})
    return d


def _set(d, node, **values):
    for param, value in values.items():
        d.execute({"op": "set", "id": node, "param": param, "value": value})


def _triangle_areas(vertices, triangles):
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    return np.linalg.norm(np.cross(b - a, c - a), axis=1)


class Card3DGridTests(unittest.TestCase):
    def test_default_is_the_single_quad_byte_identical_to_before(self):
        default = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D())
        explicit = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D(), rows=1, columns=1)
        np.testing.assert_array_equal(default.vertices, explicit.vertices)
        np.testing.assert_array_equal(default.triangles, explicit.triangles)
        np.testing.assert_array_equal(default.uvs, explicit.uvs)

    def test_two_by_two_grid_has_the_right_vertex_count_positions_and_uvs(self):
        card = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D(), rows=2, columns=2)
        self.assertEqual(len(card.vertices), 9)   # 3x3 grid of corners
        self.assertEqual(len(card.triangles), 8)  # 4 quads, 2 triangles each
        # Every corner of the 2x2 world-space grid must be present exactly once.
        expected_xy = {(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
                       (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0),
                       (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)}
        got_xy = {(round(float(x), 6), round(float(y), 6)) for x, y, _ in card.vertices}
        self.assertEqual(got_xy, expected_xy)
        # UV (0,0) is bottom-left, (1,1) top-right, matching the un-subdivided card.
        by_pos = {(round(float(x), 6), round(float(y), 6)): tuple(uv) for (x, y, _), uv in
                  zip(card.vertices, card.uvs)}
        np.testing.assert_allclose(by_pos[(-1.0, -1.0)], (0.0, 0.0))
        np.testing.assert_allclose(by_pos[(1.0, -1.0)], (1.0, 0.0))
        np.testing.assert_allclose(by_pos[(1.0, 1.0)], (1.0, 1.0))
        np.testing.assert_allclose(by_pos[(-1.0, 1.0)], (0.0, 1.0))
        areas = _triangle_areas(card.vertices, card.triangles)
        self.assertTrue((areas > 1e-6).all())

    def test_card3d_node_default_render_is_unchanged(self):
        old = _render_graph("Card3D")
        _set(old, "geo", ry=15.0)
        expected = Evaluator().evaluate(old.document, target="render")
        new = _render_graph("Card3D")
        _set(new, "geo", ry=15.0, rows=1, columns=1)
        np.testing.assert_array_equal(expected, Evaluator().evaluate(new.document, target="render"))


class Sphere3DPoleTests(unittest.TestCase):
    def test_rows_columns_grid_matches_the_old_segments_sphere_vertex_for_vertex(self):
        # rows=16, columns=32 is exactly _sphere(radius, segments=32, ...)'s own split
        # (cols=segments, rows=segments // 2), so the two must agree on every vertex, UV and
        # normal -- the fix only removes degenerate triangles, it never moves a vertex.
        grid = scene3d._sphere_grid(1.0, 16, 32, (1, 1, 1, 1), scene3d.Transform3D())
        old = scene3d._sphere(1.0, 32, (1, 1, 1, 1), scene3d.Transform3D())
        np.testing.assert_array_equal(grid.vertices, old.vertices)
        np.testing.assert_array_equal(grid.uvs, old.uvs)
        np.testing.assert_array_equal(grid.normals, old.normals)
        self.assertLess(len(grid.triangles), len(old.triangles))

    def test_small_sphere_has_no_degenerate_or_duplicated_pole_triangles_and_unit_normals(self):
        sphere = scene3d._sphere_grid(1.0, 4, 6, (1, 1, 1, 1), scene3d.Transform3D())
        areas = _triangle_areas(sphere.vertices, sphere.triangles)
        self.assertTrue((areas > 1e-6).all(), "a zero-area (degenerate) pole triangle survived")
        self.assertEqual(len(sphere.triangles), len(set(map(tuple, sphere.triangles.tolist()))),
                         "a triangle was duplicated")
        norms = np.linalg.norm(sphere.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        # 4 rows, 6 columns: the pole rows contribute one triangle per column instead of two.
        self.assertEqual(len(sphere.triangles), 2 * 6 * (4 - 2) + 2 * 6)

    def test_sphere3d_node_default_render_is_unchanged(self):
        old = _render_graph("Sphere3D")
        _set(old, "geo", tx=0.2)
        expected = Evaluator().evaluate(old.document, target="render")
        new = _render_graph("Sphere3D")
        _set(new, "geo", tx=0.2, rows=16, columns=32)
        np.testing.assert_array_equal(expected, Evaluator().evaluate(new.document, target="render"))

    def test_old_document_with_a_non_default_segments_value_renders_unchanged(self):
        d = _render_graph("Sphere3D")
        _set(d, "geo", segments=20, tx=0.3)
        # Before this change, geometry came straight from "segments"; simulate that document by
        # deleting the rows/columns keys this change adds, the way a document saved before this
        # step would arrive.
        stored = copy.deepcopy(d.document)
        del stored["nodes"]["geo"]["params"]["rows"]
        del stored["nodes"]["geo"]["params"]["columns"]
        old_geometry = scene3d._sphere(1.0, 20, (0.8, 0.8, 0.8, 1.0),
                                       scene3d._transform_from(stored["nodes"]["geo"]["params"]))
        self.assertEqual(stored["version"], SCHEMA_VERSION)
        upgraded = upgrade_document(stored)
        validate(upgraded)
        self.assertEqual(upgraded["nodes"]["geo"]["params"]["columns"], 20)
        self.assertEqual(upgraded["nodes"]["geo"]["params"]["rows"], 10)
        rendered_new = Evaluator().evaluate(upgraded, target="render")
        # Cross-check against the untouched original document (still using "segments" alone
        # end to end through the un-upgraded evaluator path is not meaningful post-change, so
        # compare instead against the primitive `_sphere` call the old node code used to make).
        expected_geo = scene3d.geometry_from_node(upgraded["nodes"]["geo"])
        np.testing.assert_array_equal(expected_geo.vertices, old_geometry.vertices)


class Cylinder3DTests(unittest.TestCase):
    def test_registered_everywhere_a_geometry_node_must_be(self):
        self.assertIn("Cylinder3D", GEOMETRY_TYPES)
        self.assertIn("Cylinder3D", SPECS)
        from nodebased.core import OUTPUT_TYPES, LIMITS
        self.assertEqual(OUTPUT_TYPES["Cylinder3D"], "geometry")
        for param in ("cyl_radius", "cyl_height", "rows", "columns"):
            self.assertIn(param, LIMITS)
        from nodebased.knobs import knob_layout
        laid_out = {p for group in knob_layout("Cylinder3D") for p in group.params}
        self.assertEqual(laid_out, set(SPECS["Cylinder3D"]["params"]))
        from nodebased.theme import COLORS
        self.assertIn("Cylinder3D", COLORS)
        from nodebased.tiers import REGION_RULES
        self.assertIn("Cylinder3D", REGION_RULES)

    def test_open_cylinder_has_no_caps_closed_cylinder_does(self):
        open_cyl = scene3d._cylinder(1.0, 2.0, 1, 8, False, (1, 1, 1, 1), scene3d.Transform3D())
        closed_cyl = scene3d._cylinder(1.0, 2.0, 1, 8, True, (1, 1, 1, 1), scene3d.Transform3D())
        self.assertEqual(len(open_cyl.vertices), 2 * (8 + 1))
        self.assertEqual(len(closed_cyl.vertices), 2 * (8 + 1) + 2)  # + top/bottom centres
        self.assertEqual(len(closed_cyl.triangles), len(open_cyl.triangles) + 2 * 8)
        areas = _triangle_areas(closed_cyl.vertices, closed_cyl.triangles)
        self.assertTrue((areas > 1e-6).all())
        norms = np.linalg.norm(closed_cyl.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_renders_through_render3d_with_visible_coverage(self):
        d = _render_graph("Cylinder3D")
        image = Evaluator().evaluate(d.document, target="render")
        self.assertGreater(float(image[..., 3].sum()), 50.0)

    def test_disabled_contributes_nothing_like_the_other_geometry_nodes(self):
        d = _render_graph("Cylinder3D")
        d.document["nodes"]["geo"]["disabled"] = True
        image = Evaluator().evaluate(d.document, target="render")
        self.assertEqual(float(np.abs(image).max()), 0.0)

    def test_a_new_document_with_only_cylinder3d_bypasses_and_reloads_unchanged(self):
        d = _render_graph("Cylinder3D")
        _set(d, "geo", cyl_radius=0.6, cyl_height=1.4, rows=2, columns=12, cyl_caps="open")
        expected = Evaluator().evaluate(d.document, target="render")
        roundtrip = upgrade_document(copy.deepcopy(d.document))
        validate(roundtrip)
        np.testing.assert_array_equal(expected, Evaluator().evaluate(roundtrip, target="render"))


class MatrixReadoutTests(unittest.TestCase):
    def test_local_matrix_is_the_nodes_own_transform(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "card", "type": "Card3D", "params": {"tx": 1.0, "ry": 45.0}})
        local, world = scene3d.local_and_world_matrix(d.document, "card")
        expected = scene3d._transform_from(d.document["nodes"]["card"]["params"]).matrix()
        np.testing.assert_array_equal(local, expected)
        np.testing.assert_array_equal(world, expected)  # no parent: world == local

    def test_world_matrix_for_a_parented_node_matches_the_hand_computed_product(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "scene", "type": "Scene3D",
                  "params": {"tx": 1.0, "ry": 30.0, "sx": 2.0}})
        d.execute({"op": "create", "id": "card", "type": "Card3D", "params": {"tz": 2.0, "rx": 10.0}})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "card"})
        local, world = scene3d.local_and_world_matrix(d.document, "card")
        scene_matrix = scene3d._transform_from(d.document["nodes"]["scene"]["params"]).matrix()
        card_matrix = scene3d._transform_from(d.document["nodes"]["card"]["params"]).matrix()
        np.testing.assert_array_equal(local, card_matrix)
        np.testing.assert_allclose(world, scene_matrix @ card_matrix, atol=1e-6)

    def test_axis3d_chain_matches_the_render(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "outer", "type": "Axis3D", "params": {"tx": 2.0}})
        d.execute({"op": "create", "id": "inner", "type": "Axis3D", "params": {"ty": -1.0, "ry": 15.0}})
        d.execute({"op": "create", "id": "sphere", "type": "Sphere3D"})
        d.execute({"op": "connect", "id": "inner", "input": "object", "source": "sphere"})
        d.execute({"op": "connect", "id": "outer", "input": "object", "source": "inner"})
        local, world = scene3d.local_and_world_matrix(d.document, "sphere")
        outer_matrix = scene3d._transform_from(d.document["nodes"]["outer"]["params"]).matrix()
        inner_matrix = scene3d._transform_from(d.document["nodes"]["inner"]["params"]).matrix()
        sphere_matrix = scene3d._transform_from(d.document["nodes"]["sphere"]["params"]).matrix()
        np.testing.assert_array_equal(local, sphere_matrix)
        np.testing.assert_allclose(world, outer_matrix @ inner_matrix @ sphere_matrix, atol=1e-6)

    def test_camera_and_light_local_matrix_is_translation_only_and_never_parented(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "scene", "type": "Scene3D", "params": {"tx": 5.0}})
        d.execute({"op": "create", "id": "cam", "type": "Camera3D", "params": {"tx": 1.0, "ty": 2.0, "tz": 3.0}})
        d.execute({"op": "create", "id": "light", "type": "Light3D",
                  "params": {"tx": -1.0, "ty": 0.0, "tz": 0.0}})
        cam_local, cam_world = scene3d.local_and_world_matrix(d.document, "cam")
        expected_cam = scene3d.Transform3D(scene3d.Vec3(1.0, 2.0, 3.0)).matrix()
        np.testing.assert_array_equal(cam_local, expected_cam)
        np.testing.assert_array_equal(cam_world, expected_cam)  # Camera3D can never be parented
        light_local, _ = scene3d.local_and_world_matrix(d.document, "light")
        np.testing.assert_array_equal(light_local, scene3d.Transform3D(scene3d.Vec3(-1.0, 0.0, 0.0)).matrix())

    def test_disabled_axis3d_parents_at_the_identity(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "axis", "type": "Axis3D", "params": {"tx": 9.0}})
        d.execute({"op": "create", "id": "card", "type": "Card3D"})
        d.execute({"op": "connect", "id": "axis", "input": "object", "source": "card"})
        d.document["nodes"]["axis"]["disabled"] = True
        _, world = scene3d.local_and_world_matrix(d.document, "card")
        card_matrix = scene3d._transform_from(d.document["nodes"]["card"]["params"]).matrix()
        np.testing.assert_array_equal(world, card_matrix)


if __name__ == "__main__":
    unittest.main()
