"""Axis3D (a pure Nuke-style parenting transform) and TransformGeo3D (bakes a transform into an
incoming geometry's own vertices and normals). See docs/3D_FOUNDATION.md, "Two ways to move
geometry"."""
import copy
import unittest

import numpy as np

from nodebased.core import Dispatcher, SCHEMA_VERSION, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased import scene3d


def _render_graph(geo_kind="Cube3D"):
    """geo -> scene -> render, the smallest tree a geometry node can render through."""
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


def _render(d, target="render"):
    return Evaluator().evaluate(d.document, target=target)


def _typed(d, target):
    return Evaluator().evaluate_raster(d.document, target, typed=True)


class Axis3DTests(unittest.TestCase):
    def _chain(self):
        """cube -> a2 -> a1 -> scene -> render, two chained Axis3D nodes."""
        d = _render_graph()
        d.execute({"op": "create", "id": "a2", "type": "Axis3D"})
        d.execute({"op": "create", "id": "a1", "type": "Axis3D"})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": None})
        d.execute({"op": "connect", "id": "a2", "input": "object", "source": "geo"})
        d.execute({"op": "connect", "id": "a1", "input": "object", "source": "a2"})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "a1"})
        return d

    def test_nothing_wired_is_an_empty_scene(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "axis", "type": "Axis3D"})
        scene = _typed(d, "axis")
        self.assertIsInstance(scene, scene3d.Scene)
        self.assertEqual((len(scene.geometries), len(scene.lights), len(scene.splats)), (0, 0, 0))

    def test_parents_a_single_geometry(self):
        d = _render_graph()
        d.execute({"op": "create", "id": "axis", "type": "Axis3D"})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": None})
        d.execute({"op": "connect", "id": "axis", "input": "object", "source": "geo"})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "axis"})
        _set(d, "axis", tx=1.0, ry=30.0)
        scene = _typed(d, "axis")
        self.assertEqual(len(scene.geometries), 1)
        geometry = scene.geometries[0]
        expected = scene3d._transform_from(d.document["nodes"]["axis"]["params"]).matrix()
        np.testing.assert_allclose(geometry.parent, expected, atol=1e-6)
        # The Cube3D's own transform is untouched: Axis3D only ever adds a parent.
        np.testing.assert_allclose(geometry.transform.matrix(), np.eye(4), atol=1e-6)

    def test_chain_of_two_composes_in_order_against_the_hand_computed_product(self):
        d = self._chain()
        _set(d, "a2", tz=2.0, rx=15.0)
        _set(d, "a1", tx=1.0, ry=30.0)
        scene = _typed(d, "a1")
        geometry = scene.geometries[0]
        m1 = scene3d._transform_from(d.document["nodes"]["a1"]["params"]).matrix()
        m2 = scene3d._transform_from(d.document["nodes"]["a2"]["params"]).matrix()
        np.testing.assert_allclose(geometry.parent, m1 @ m2, atol=1e-6)
        # Order matters: the reverse product must disagree once the two transforms differ.
        self.assertFalse(np.allclose(geometry.parent, m2 @ m1, atol=1e-6))

    def test_parents_a_light_and_flattens_a_nested_scene(self):
        d = Dispatcher()
        for key, kind in (("ball", "Sphere3D"), ("key", "Light3D"), ("inner", "Scene3D"), ("axis", "Axis3D")):
            d.execute({"op": "create", "id": key, "type": kind})
        d.execute({"op": "connect", "id": "inner", "input": "object0", "source": "ball"})
        d.execute({"op": "connect", "id": "inner", "input": "object1", "source": "key"})
        d.execute({"op": "connect", "id": "axis", "input": "object", "source": "inner"})
        _set(d, "axis", tx=3.0)
        scene = _typed(d, "axis")
        self.assertEqual((len(scene.geometries), len(scene.lights)), (1, 1))
        np.testing.assert_allclose(scene.geometries[0].world_matrix()[:3, 3], (3, 0, 0), atol=1e-6)
        np.testing.assert_allclose(scene.lights[0].world()[0], (5, 4, 3), atol=1e-5)

    def test_disabled_uses_the_identity_and_still_passes_the_object_through(self):
        d = self._chain()
        _set(d, "a2", tz=2.0)
        _set(d, "a1", tx=5.0, ry=45.0)  # would move the geometry if applied
        # Axis3D's one input is optional, so SPECS["Axis3D"]["inputs"] is empty and the interactive
        # "disable" op refuses it as a source node, exactly like Scene3D. Old documents and agent
        # authored ones can still carry disabled=True, so set it directly, as the Scene3D/Card3D
        # bypass tests in test_3d_foundation.py do.
        d.document["nodes"]["a1"]["disabled"] = True
        scene = _typed(d, "a1")
        m2 = scene3d._transform_from(d.document["nodes"]["a2"]["params"]).matrix()
        # a1 contributes nothing (identity) but a2's geometry still arrives.
        np.testing.assert_allclose(scene.geometries[0].parent, m2, atol=1e-6)

    def test_at_rest_renders_byte_identical_to_no_axis3d_at_all(self):
        plain = _render_graph()
        image = _render(plain)
        via_axis = _render_graph()
        via_axis.execute({"op": "create", "id": "axis", "type": "Axis3D"})
        via_axis.execute({"op": "connect", "id": "scene", "input": "object0", "source": None})
        via_axis.execute({"op": "connect", "id": "axis", "input": "object", "source": "geo"})
        via_axis.execute({"op": "connect", "id": "scene", "input": "object0", "source": "axis"})
        np.testing.assert_array_equal(image, _render(via_axis))

    def test_translating_the_axis_changes_the_render(self):
        d = self._chain()
        near = _render(d)
        _set(d, "a1", tx=1.0)
        far = _render(d)
        self.assertFalse(np.array_equal(near, far))

    def test_rejects_a_camera_input_without_mutating_the_document(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "cam", "type": "Camera3D"})
        d.execute({"op": "create", "id": "axis", "type": "Axis3D"})
        before = d.document
        with self.assertRaisesRegex(ValueError, "expects geometry or light or scene"):
            d.execute({"op": "connect", "id": "axis", "input": "object", "source": "cam"})
        self.assertEqual(before, d.document)

    def test_old_document_without_axis3d_loads_and_renders_unchanged(self):
        d = _render_graph()
        _set(d, "geo", tx=0.5, ry=20.0)
        expected = _render(d)
        old = copy.deepcopy(d.document)
        self.assertEqual(old["version"], SCHEMA_VERSION)
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        np.testing.assert_array_equal(expected, Evaluator().evaluate(upgraded, target="render"))


class TransformGeo3DTests(unittest.TestCase):
    def _xf_graph(self, geo_kind="Cube3D"):
        """geo -> xf -> scene -> render."""
        d = _render_graph(geo_kind)
        d.execute({"op": "create", "id": "xf", "type": "TransformGeo3D"})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": None})
        d.execute({"op": "connect", "id": "xf", "input": "geo", "source": "geo"})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "xf"})
        return d

    def test_bakes_translation_into_vertices_leaving_everything_else_alone(self):
        d = self._xf_graph()
        _set(d, "xf", tx=1.0, ty=-2.0, tz=0.5)
        before = scene3d.geometry_from_node(d.document["nodes"]["geo"])
        after = _typed(d, "xf")
        np.testing.assert_allclose(after.vertices, before.vertices + (1.0, -2.0, 0.5), atol=1e-5)
        np.testing.assert_array_equal(after.triangles, before.triangles)
        self.assertEqual(after.color, before.color)
        np.testing.assert_allclose(after.transform.matrix(), np.eye(4), atol=1e-6)

    def test_rotates_normals_through_the_inverse_transpose_and_keeps_them_unit_length(self):
        d = self._xf_graph("Sphere3D")
        _set(d, "xf", ry=90.0)
        before = scene3d.geometry_from_node(d.document["nodes"]["geo"])
        after = _typed(d, "xf")
        c, s = 0.0, 1.0  # cos(90), sin(90)
        rotate_y = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), np.float32)
        expected = (rotate_y @ before.normals.T).T
        np.testing.assert_allclose(after.normals, expected, atol=1e-4)
        lengths = np.linalg.norm(after.normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)

    def test_pivot_case_matches_the_hand_computed_matrix_and_moves_the_far_vertex(self):
        d = self._xf_graph()
        _set(d, "xf", tx=0.5, ry=70.0, rz=15.0, uscale=1.3, pivot_x=1.0, pivot_y=-0.5, pivot_z=0.25)
        matrix = scene3d._transform_from(d.document["nodes"]["xf"]["params"]).matrix()
        pivot = np.array((1.0, -0.5, 0.25, 1.0), np.float32)
        np.testing.assert_allclose(matrix @ pivot, pivot + (0.5, 0, 0, 0), atol=1e-5)
        moved = _render(d)
        _set(d, "xf", pivot_x=0.0, pivot_y=0.0, pivot_z=0.0)
        self.assertFalse(np.array_equal(moved, _render(d)))

    def test_acts_before_the_geometrys_own_transform_and_further_parenting(self):
        # card's own tx, then xf's transform baked into vertices, then the enclosing Scene3D's tx:
        # world = scene_matrix @ card.transform.matrix() @ (xf_matrix @ local_vertex).
        d = Dispatcher()
        for key, kind in (("card", "Card3D"), ("xf", "TransformGeo3D"), ("scene", "Scene3D"),
                          ("cam", "Camera3D"), ("render", "Render3D")):
            d.execute({"op": "create", "id": key, "type": kind})
        d.execute({"op": "connect", "id": "xf", "input": "geo", "source": "card"})
        d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "xf"})
        d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
        d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
        _set(d, "card", tx=2.0)
        _set(d, "xf", tx=1.0)
        _set(d, "scene", tx=0.5)
        result = _typed(d, "scene")
        geometry = result.geometries[0]
        card_before = scene3d.geometry_from_node({"type": "Card3D", "params": {
            **d.document["nodes"]["card"]["params"]}})
        xf_matrix = scene3d._transform_from(d.document["nodes"]["xf"]["params"]).matrix()
        baked_vertex = (xf_matrix[:3, :3] @ card_before.vertices[0]) + xf_matrix[:3, 3]
        world = geometry.world_matrix()
        world_vertex = world[:3, :3] @ baked_vertex + world[:3, 3]
        card_matrix = scene3d._transform_from(d.document["nodes"]["card"]["params"]).matrix()
        scene_matrix = scene3d._transform_from(d.document["nodes"]["scene"]["params"]).matrix()
        expected = (scene_matrix @ card_matrix)[:3, :3] @ baked_vertex + (scene_matrix @ card_matrix)[:3, 3]
        np.testing.assert_allclose(world_vertex, expected, atol=1e-5)

    def test_disabled_passes_geometry_through_unbaked(self):
        d = self._xf_graph()
        _set(d, "xf", tx=5.0, ry=45.0)
        d.execute({"op": "disable", "id": "xf", "value": True})
        before = scene3d.geometry_from_node(d.document["nodes"]["geo"])
        after = _typed(d, "xf")
        np.testing.assert_array_equal(after.vertices, before.vertices)

    def test_renders_through_render3d_and_translating_it_changes_pixels(self):
        d = self._xf_graph()
        near = _render(d)
        _set(d, "xf", tz=2.0)
        far = _render(d)
        self.assertFalse(np.array_equal(near, far))
        self.assertGreater(float(near[..., 3].max()), 0.9)

    def test_requires_its_input(self):
        d = self._xf_graph()
        d.execute({"op": "connect", "id": "xf", "input": "geo", "source": None})
        with self.assertRaisesRegex(ValueError, "connect required input"):
            _render(d)

    def test_rejects_a_scene_input_without_mutating_the_document(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "scene", "type": "Scene3D"})
        d.execute({"op": "create", "id": "xf", "type": "TransformGeo3D"})
        before = d.document
        with self.assertRaisesRegex(ValueError, "expects geometry"):
            d.execute({"op": "connect", "id": "xf", "input": "geo", "source": "scene"})
        self.assertEqual(before, d.document)

    def test_old_document_without_transformgeo3d_loads_and_renders_unchanged(self):
        d = _render_graph()
        _set(d, "geo", sx=1.5, rz=10.0)
        expected = _render(d)
        old = copy.deepcopy(d.document)
        upgraded = upgrade_document(old)
        validate(upgraded)
        np.testing.assert_array_equal(expected, Evaluator().evaluate(upgraded, target="render"))


if __name__ == "__main__":
    unittest.main()
