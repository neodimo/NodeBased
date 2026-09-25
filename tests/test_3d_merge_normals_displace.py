"""MergeGeo3D, Normals3D and DisplaceGeo3D (lane L3 step 4). Vertices, normals, indices and pixels."""
import unittest

import numpy as np

from nodebased.core import Dispatcher, SPECS, bypass_slot, OUTPUT_TYPES, INPUT_TYPES, validate
from nodebased.imaging import Evaluator
from nodebased import scene3d


def _make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def _wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def _value(d, key):
    return Evaluator().evaluate_raster(d.document, key, typed=True)


def _world(g):
    m = g.world_matrix().astype(np.float64)
    return (m[:3, :3] @ g.vertices.astype(np.float64).T).T + m[:3, 3]


class MergeGeo3DTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        _make(self.d, a=("Card3D", {}), b=("Card3D", {"tx": 3.0}), m=("MergeGeo3D", {}))
        _wire(self.d, "m", "geo0", "a")
        _wire(self.d, "m", "geo1", "b")

    def test_two_cards_concatenate_with_transforms_baked_and_indices_offset(self):
        g = _value(self.d, "m")
        self.assertEqual(g.vertices.shape, (8, 3))
        np.testing.assert_allclose(g.vertices[:4], _world(_value(self.d, "a")), atol=1e-6)
        np.testing.assert_allclose(g.vertices[4:, 0], (2, 4, 4, 2), atol=1e-6)
        self.assertEqual(g.triangles.tolist(), [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        self.assertEqual(g.uvs.shape, (8, 2))
        np.testing.assert_array_equal(g.uvs[:4], g.uvs[4:])
        np.testing.assert_allclose(g.world_matrix(), np.eye(4), atol=1e-7)

    def test_a_translated_input_lands_translated(self):
        self.d.execute({"op": "set", "id": "a", "param": "ty", "value": 5.0})
        g = _value(self.d, "m")
        np.testing.assert_allclose(g.vertices[:4, 1], (4, 4, 6, 6), atol=1e-6)

    def test_inputs_under_a_scene_parent_bake_the_parent_too(self):
        _make(self.d, axis=("Axis3D", {"tz": 2.0}), s=("Scene3D", {}))
        _wire(self.d, "axis", "object", "a")
        _wire(self.d, "s", "object0", "axis")
        # An Axis3D outputs a scene, which a geometry slot refuses; the parent lives on the geometry.
        with self.assertRaises(ValueError):
            _wire(self.d, "m", "geo2", "axis")
        parented = _value(self.d, "s").geometries[0]
        merged = scene3d.merge_geometry([parented])
        np.testing.assert_allclose(merged.vertices[:, 2], 2.0, atol=1e-6)

    def test_merging_one_input_equals_transform_geo3d_of_it(self):
        _make(self.d, solo=("MergeGeo3D", {"tx": 1.0, "ry": 30.0, "uscale": 2.0}),
              t=("TransformGeo3D", {"tx": 1.0, "ry": 30.0, "uscale": 2.0}),
              sph=("Sphere3D", {"rows": 6, "columns": 8}))
        _wire(self.d, "solo", "geo0", "sph")
        _wire(self.d, "t", "geo", "sph")
        merged, transformed = _value(self.d, "solo"), _value(self.d, "t")
        np.testing.assert_allclose(merged.vertices, _world(transformed), atol=1e-5)
        m = transformed.world_matrix().astype(np.float64)[:3, :3]
        expected = (np.linalg.inv(m).T @ transformed.normals.T.astype(np.float64)).T
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(merged.normals, expected, atol=1e-5)
        np.testing.assert_array_equal(merged.triangles, transformed.triangles)

    def test_normals_use_the_inverse_transpose_under_nonuniform_scale(self):
        _make(self.d, sph=("Sphere3D", {"rows": 6, "columns": 8, "sx": 4.0}), one=("MergeGeo3D", {}))
        _wire(self.d, "one", "geo0", "sph")
        g = _value(self.d, "one")
        plane = np.linalg.norm(g.normals, axis=1)
        np.testing.assert_allclose(plane, 1.0, atol=1e-5)
        # An ellipsoid stretched along X: the true normal is (x/16, y, z), not the scaled sphere normal.
        expected = g.vertices * np.array((1 / 16, 1, 1))
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(g.normals, expected, atol=0.05)

    def test_mixed_normals_still_gives_one_unit_normal_per_vertex(self):
        _make(self.d, sph=("Sphere3D", {"rows": 4, "columns": 6}))
        _wire(self.d, "m", "geo2", "sph")
        g = _value(self.d, "m")
        self.assertEqual(len(g.normals), len(g.vertices))
        np.testing.assert_allclose(np.linalg.norm(g.normals, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(g.normals[:8], np.tile((0, 0, 1), (8, 1)), atol=1e-6)   # the two cards face +Z

    def test_mirroring_input_keeps_faces_outward(self):
        # No knob can mirror (scale is positive), but an imported parent matrix can.
        mirrored = scene3d.replace(_value(self.d, "a"), transform=scene3d.Transform3D(scale=scene3d.Vec3(-1, 1, 1)))
        g = scene3d.merge_geometry([mirrored])
        np.testing.assert_allclose(g.vertices[:, 0], (1, -1, -1, 1), atol=1e-6)
        cross = np.cross(g.vertices[g.triangles[0, 1]] - g.vertices[g.triangles[0, 0]],
                         g.vertices[g.triangles[0, 2]] - g.vertices[g.triangles[0, 0]])
        self.assertGreater(cross[2], 0)

    def test_empty_merge_is_an_empty_geometry_and_renders(self):
        _make(self.d, empty=("MergeGeo3D", {}), cam=("Camera3D", {}), s=("Scene3D", {}),
              r=("Render3D", {"width": 16, "height": 16, "samples": 1}))
        g = _value(self.d, "empty")
        self.assertEqual((len(g.vertices), len(g.triangles)), (0, 0))
        _wire(self.d, "s", "object0", "empty")
        _wire(self.d, "r", "scene", "s")
        _wire(self.d, "r", "camera", "cam")
        pixels = Evaluator().evaluate(self.d.document, target="r")
        self.assertEqual(float(np.asarray(pixels)[..., 3].max()), 0.0)

    def test_output_feeds_scene3d_and_renders_both_cards(self):
        _make(self.d, cam=("Camera3D", {"tz": 8.0}), s=("Scene3D", {}),
              r=("Render3D", {"width": 64, "height": 32, "samples": 1}))
        _wire(self.d, "s", "object0", "m")
        _wire(self.d, "r", "scene", "s")
        _wire(self.d, "r", "camera", "cam")
        pixels = np.asarray(Evaluator().evaluate(self.d.document, target="r"))
        covered = pixels[..., 3] > 0.5
        self.assertTrue(covered[:, :32].any() and covered[:, 32:].any())

    def test_bypass_passes_the_first_wired_input(self):
        self.d.execute({"op": "disable", "id": "m", "value": True})
        np.testing.assert_array_equal(_value(self.d, "m").vertices, _value(self.d, "a").vertices)
        _wire(self.d, "m", "geo0", None)
        np.testing.assert_array_equal(_value(self.d, "m").vertices, _value(self.d, "b").vertices)
        node = self.d.document["nodes"]["m"]
        self.assertEqual(bypass_slot(node), "geo1")

    def test_registered_types_and_old_documents(self):
        self.assertEqual(OUTPUT_TYPES["MergeGeo3D"], "geometry")
        self.assertEqual(SPECS["MergeGeo3D"]["inputs"], [])
        self.assertEqual(len(SPECS["MergeGeo3D"]["optional_inputs"]), 8)
        self.assertEqual(INPUT_TYPES["geo3"], ("geometry",))
        validate(self.d.document)


class Normals3DTests(unittest.TestCase):
    def _graph(self, kind, mode, **params):
        d = Dispatcher()
        _make(d, g=(kind, params), n=("Normals3D", {"normals_mode": mode}))
        _wire(d, "n", "geo", "g")
        return d

    def test_recompute_on_a_card_gives_unit_normals_facing_plus_z(self):
        g = _value(self._graph("Card3D", "recompute", rows=3, columns=3), "n")
        np.testing.assert_allclose(g.normals, np.tile((0, 0, 1), (16, 1)), atol=1e-6)

    def test_recompute_follows_winding(self):
        d = self._graph("Card3D", "recompute")
        d.execute({"op": "create", "id": "f", "type": "Normals3D", "params": {"normals_mode": "flip"}})
        _wire(d, "f", "geo", "g")
        _wire(d, "n", "geo", "f")
        np.testing.assert_allclose(_value(d, "n").normals, np.tile((0, 0, -1), (4, 1)), atol=1e-6)

    def test_recompute_on_a_sphere_matches_the_normalised_positions(self):
        g = _value(self._graph("Sphere3D", "recompute", rows=16, columns=32), "n")
        expected = g.vertices / np.linalg.norm(g.vertices, axis=1, keepdims=True)
        self.assertLess(np.abs(g.normals - expected).max(), 0.05)
        np.testing.assert_allclose(np.linalg.norm(g.normals, axis=1), 1.0, atol=1e-5)

    def test_recompute_keeps_a_cube_flat(self):
        g = _value(self._graph("Cube3D", "recompute"), "n")
        # Every normal is axis aligned: per-face vertices are not smoothed across the hard edges.
        np.testing.assert_allclose(np.abs(g.normals).max(axis=1), 1.0, atol=1e-5)

    def test_unchanged_is_the_input(self):
        d = self._graph("Sphere3D", "unchanged", rows=4, columns=6)
        np.testing.assert_array_equal(_value(d, "n").normals, _value(d, "g").normals)
        np.testing.assert_array_equal(_value(d, "n").triangles, _value(d, "g").triangles)

    def test_flip_negates_normals_and_reverses_winding(self):
        d = self._graph("Sphere3D", "flip", rows=4, columns=6)
        before, after = _value(d, "g"), _value(d, "n")
        np.testing.assert_allclose(after.normals, -before.normals, atol=1e-7)
        np.testing.assert_array_equal(after.triangles, before.triangles[:, ::-1])
        np.testing.assert_array_equal(after.vertices, before.vertices)

    def test_flip_winding_toggle_reverses_faces(self):
        d = self._graph("Card3D", "unchanged")
        d.execute({"op": "set", "id": "n", "param": "flip_winding", "value": 1})
        g = _value(d, "n")
        np.testing.assert_array_equal(g.triangles, [[2, 1, 0], [3, 2, 0]])
        self.assertIsNone(g.normals)   # flat face normals follow the reversed winding by themselves

    def test_unify_repairs_mixed_winding_on_a_closed_cube_outward(self):
        d = self._graph("Cube3D", "unify")
        cube = _value(d, "g")
        broken = cube.triangles.copy()
        broken[[0, 3, 7]] = broken[[0, 3, 7]][:, ::-1]
        fixed = scene3d.unify_winding(scene3d.replace(cube, triangles=broken))
        centre = fixed.vertices.mean(axis=0)
        cross = scene3d._face_cross(fixed.vertices, fixed.triangles.astype(np.int64))
        face_centres = fixed.vertices[fixed.triangles].mean(axis=1) - centre
        self.assertTrue((np.einsum("ij,ij->i", cross, face_centres) > 0).all())
        # Even an inside-out closed surface is turned outward.
        inside_out = scene3d.unify_winding(scene3d.flip_geometry(cube))
        cross = scene3d._face_cross(inside_out.vertices, inside_out.triangles.astype(np.int64))
        self.assertTrue((np.einsum("ij,ij->i", cross, inside_out.vertices[inside_out.triangles].mean(axis=1) - centre) > 0).all())

    def test_two_sided_shader_means_flip_does_not_change_pixels(self):
        # Render3D shades both sides toward the eye, so a flipped card renders identically.
        d = Dispatcher()
        _make(d, c=("Card3D", {"ty": 0.2, "rx": -30.0}), n=("Normals3D", {"normals_mode": "recompute"}),
              f=("Normals3D", {"normals_mode": "flip"}), cam=("Camera3D", {"tz": 5.0}), s=("Scene3D", {}),
              light=("Light3D", {"tx": 1.0, "ty": 2.0, "tz": 4.0}),
              r=("Render3D", {"width": 32, "height": 32, "samples": 1}))
        _wire(d, "n", "geo", "c")
        _wire(d, "f", "geo", "n")
        _wire(d, "s", "object0", "n")
        _wire(d, "s", "object1", "light")
        _wire(d, "r", "scene", "s")
        _wire(d, "r", "camera", "cam")
        plain = np.asarray(Evaluator().evaluate(d.document, target="r"))
        _wire(d, "s", "object0", "f")
        flipped = np.asarray(Evaluator().evaluate(d.document, target="r"))
        self.assertGreater(float(plain[..., 3].max()), 0.5)
        np.testing.assert_allclose(plain, flipped, atol=1e-5)
        # The evaluated normals themselves are negated, which is the state Normals3D owns.
        np.testing.assert_allclose(_value(d, "f").normals, -_value(d, "n").normals, atol=1e-6)

    def test_bypass_passes_the_geometry(self):
        d = self._graph("Card3D", "flip")
        d.execute({"op": "disable", "id": "n", "value": True})
        np.testing.assert_array_equal(_value(d, "n").triangles, _value(d, "g").triangles)


class DisplaceGeo3DTests(unittest.TestCase):
    def _graph(self, image=None, primitive=("Card3D", {"rows": 4, "columns": 4}), **params):
        d = Dispatcher()
        _make(d, g=primitive, disp=("DisplaceGeo3D", params))
        _wire(d, "disp", "geo", "g")
        if image == "white":
            _make(d, img=("Constant", {"width": 16, "height": 16, "red": 1.0, "green": 1.0, "blue": 1.0}))
            _wire(d, "disp", "image", "img")
        elif image == "ramp":
            _make(d, img=("Ramp", {"width": 64, "height": 16, "p1_x": 63.0}))
            _wire(d, "disp", "image", "img")
        return d

    def test_constant_white_moves_every_vertex_by_scale_plus_offset_along_z(self):
        d = self._graph("white", displace_scale=2.0, displace_offset=0.25)
        before, after = _value(d, "g"), _value(d, "disp")
        self.assertEqual(after.vertices.shape, (25, 3))
        np.testing.assert_allclose(after.vertices[:, :2], before.vertices[:, :2], atol=1e-7)
        np.testing.assert_allclose(after.vertices[:, 2], 2.25, atol=1e-5)

    def test_horizontal_ramp_displaces_the_right_edge_more_than_the_left(self):
        d = self._graph("ramp", displace_scale=1.0)
        g = _value(d, "disp")
        left, right = g.vertices[g.vertices[:, 0] < -0.99], g.vertices[g.vertices[:, 0] > 0.99]
        self.assertGreater(right[:, 2].mean(), left[:, 2].mean() + 0.8)
        # Monotone across a row.
        row = g.vertices[:5, 2]
        self.assertTrue((np.diff(row) > 0).all())

    def test_no_image_is_a_flat_offset(self):
        d = self._graph(displace_scale=5.0, displace_offset=0.5)
        np.testing.assert_allclose(_value(d, "disp").vertices[:, 2], 0.5, atol=1e-6)

    def test_channels_pick_the_right_component(self):
        d = Dispatcher()
        _make(d, g=("Card3D", {}), img=("Constant", {"width": 8, "height": 8, "red": 0.2, "green": 0.4, "blue": 0.6}),
              disp=("DisplaceGeo3D", {"displace_channel": "green"}))
        _wire(d, "disp", "geo", "g")
        _wire(d, "disp", "image", "img")
        for channel, expected in (("red", 0.2), ("green", 0.4), ("blue", 0.6), ("alpha", 1.0),
                                  ("luminance", 0.2126 * 0.2 + 0.7152 * 0.4 + 0.0722 * 0.6)):
            d.execute({"op": "set", "id": "disp", "param": "displace_channel", "value": channel})
            np.testing.assert_allclose(_value(d, "disp").vertices[:, 2], expected, atol=1e-5, err_msg=channel)

    def test_recomputed_normals_tilt_with_the_slope(self):
        d = self._graph("ramp", displace_scale=1.0)
        g = _value(d, "disp")
        # Height rises with +x, so the surface normal leans toward -x.
        self.assertTrue((g.normals[:, 0] < -0.1).all())
        self.assertTrue((g.normals[:, 2] > 0).all())
        np.testing.assert_allclose(np.linalg.norm(g.normals, axis=1), 1.0, atol=1e-5)
        d.execute({"op": "set", "id": "disp", "param": "recompute_normals", "value": 0})
        stale = _value(d, "disp")
        self.assertIsNone(stale.normals)   # a Card3D has none; with recompute off none are invented

    def test_displacement_runs_along_the_normal_not_always_z(self):
        d = Dispatcher()
        _make(d, g=("Sphere3D", {"rows": 6, "columns": 8}), disp=("DisplaceGeo3D", {"displace_offset": 0.5}))
        _wire(d, "disp", "geo", "g")
        before, after = _value(d, "g"), _value(d, "disp")
        np.testing.assert_allclose(np.linalg.norm(after.vertices, axis=1), 1.5, atol=1e-5)

    def test_recompute_setting_changes_shading_pixels(self):
        def render(recompute):
            d = self._graph("ramp", ("Sphere3D", {"rows": 16, "columns": 32}), displace_scale=0.6,
                            recompute_normals=recompute)
            _make(d, cam=("Camera3D", {"tz": 5.0}), s=("Scene3D", {}),
                  light=("Light3D", {"tx": -3.0, "ty": 0.0, "tz": 3.0, "light_type": "Point"}),
                  r=("Render3D", {"width": 48, "height": 48, "samples": 1}))
            _wire(d, "s", "object0", "disp")
            _wire(d, "s", "object1", "light")
            _wire(d, "r", "scene", "s")
            _wire(d, "r", "camera", "cam")
            return np.asarray(Evaluator().evaluate(d.document, target="r"))
        on, off = render(1), render(0)
        self.assertGreater(float(on[..., 3].max()), 0.5)
        self.assertGreater(float(np.abs(on - off).max()), 0.02)

    def test_bypass_passes_the_geometry(self):
        d = self._graph("white", displace_scale=3.0)
        d.execute({"op": "disable", "id": "disp", "value": True})
        np.testing.assert_array_equal(_value(d, "disp").vertices, _value(d, "g").vertices)

    def test_registered(self):
        self.assertEqual(OUTPUT_TYPES["DisplaceGeo3D"], "geometry")
        self.assertEqual(SPECS["DisplaceGeo3D"]["inputs"], ["geo"])
        self.assertEqual(SPECS["DisplaceGeo3D"]["optional_inputs"], ["image"])


class RegistrationTests(unittest.TestCase):
    KINDS = ("MergeGeo3D", "Normals3D", "DisplaceGeo3D")

    def test_every_registry_knows_the_new_nodes(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from nodebased import knobs, tiers, theme
        from nodebased.app import CIRCLE_TYPES, MATRIX_READOUT_TYPES, node_form
        from nodebased.core import CHOICES, LIMITS
        for kind in self.KINDS:
            self.assertIn(kind, theme.COLORS)
            self.assertEqual(node_form({"type": kind}), "round")
            self.assertNotIn(kind, CIRCLE_TYPES)
            covered = {p for group in knobs.KNOB_LAYOUT[kind] for p in group.params}
            self.assertEqual(covered, set(SPECS[kind]["params"]), kind)
            arity = len(SPECS[kind]["inputs"]) + len(SPECS[kind].get("optional_inputs", []))
            self.assertEqual(tiers.input_regions(kind, {}, tiers.Region(0, 0, 8, 8), arity),
                             [None] * arity)
        self.assertIn("MergeGeo3D", MATRIX_READOUT_TYPES)
        self.assertEqual(CHOICES["normals_mode"], ["unchanged", "recompute", "flip", "unify"])
        self.assertIn("displace_scale", LIMITS)

    def test_matrix_readouts_for_merge_geo(self):
        d = Dispatcher()
        _make(d, m=("MergeGeo3D", {"tx": 2.0}), s=("Scene3D", {"ty": 3.0}))
        _wire(d, "s", "object0", "m")
        local, world = scene3d.local_and_world_matrix(d.document, "m")
        np.testing.assert_allclose(local[:3, 3], (2, 0, 0))
        np.testing.assert_allclose(world[:3, 3], (2, 3, 0))


if __name__ == "__main__":
    unittest.main()
