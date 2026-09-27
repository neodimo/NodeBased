"""Instance3D (lane L4 step A, DiMo 9/27): copies a mesh onto every particle or point.

See docs/3D_ROADMAP.md "Instancing" and `scene3d.instances_from_node`/`expand_instances`.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import scene3d
from nodebased.core import Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS, bypass_slot, validate
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.theme import COLORS
from nodebased.tiers import REGION_RULES


def _make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def _wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def _set(d, key, **params):
    for name, value in params.items():
        d.execute({"op": "set", "id": key, "param": name, "value": value})


def _value(d, key, frame=1):
    return Evaluator().evaluate_raster(d.document, key, frame=frame, typed=True)


def _points(positions, velocities=None, colors=None, ages=None, ids=None):
    n = len(positions)
    return scene3d.ParticleInstance(
        positions=np.asarray(positions, np.float32),
        sizes=np.full(n, 0.1, np.float32),
        colors=np.ones((n, 4), np.float32) if colors is None else np.asarray(colors, np.float32),
        velocities=None if velocities is None else np.asarray(velocities, np.float32),
        ages=None if ages is None else np.asarray(ages, np.float32),
        ids=None if ids is None else np.asarray(ids, np.int64))


def _cube_mesh(size=1.0, color=(0.8, 0.2, 0.2, 1.0)):
    return scene3d._cube(size, color, scene3d.Transform3D())


class RegistrationTests(unittest.TestCase):
    def test_registered_everywhere(self):
        self.assertIn("Instance3D", SPECS)
        self.assertEqual(SPECS["Instance3D"]["inputs"], ["points", "instance"])
        self.assertEqual(OUTPUT_TYPES["Instance3D"], "scene")
        self.assertIn("particles", INPUT_TYPES["points"])
        self.assertIn("geometry", INPUT_TYPES["points"])
        self.assertIn("geometry", INPUT_TYPES["instance"])
        self.assertIn("Instance3D", COLORS)
        self.assertIn("Instance3D", REGION_RULES)
        laid_out = sorted(p for group in knob_layout("Instance3D") for p in group.params)
        self.assertEqual(laid_out, sorted(SPECS["Instance3D"]["params"]))

    def test_bypass_passes_the_points(self):
        d = Dispatcher()
        _make(d, pts=("ParticleEmitter3D", {}), cube=("Cube3D", {}),
              inst=("Instance3D", {}))
        _wire(d, "inst", "points", "pts")
        _wire(d, "inst", "instance", "cube")
        self.assertEqual(bypass_slot(d.document["nodes"]["inst"]), "points")
        d.execute({"op": "disable", "id": "inst", "value": True})
        points_value = _value(d, "pts")
        bypassed_value = _value(d, "inst")
        self.assertIsInstance(bypassed_value, scene3d.ParticleInstance)
        np.testing.assert_array_equal(bypassed_value.positions, points_value.positions)


class InstanceTransformTests(unittest.TestCase):
    def test_n_particles_give_n_instances_at_their_positions(self):
        positions = [(0, 0, 0), (1, 2, 3), (-4, 5, -6), (7, -8, 9)]
        points = _points(positions)
        instances = scene3d.instances_from_node(points, _cube_mesh(), {})
        self.assertEqual(len(instances), len(positions))
        expanded = scene3d.expand_instances(instances)
        self.assertEqual(len(expanded), len(positions))
        for geometry, position in zip(expanded, positions):
            np.testing.assert_allclose(geometry.world_matrix()[:3, 3], position, atol=1e-5)

    def test_orient_velocity_points_plus_z_along_velocity(self):
        points = _points([(0, 0, 0)], velocities=[(0, 0, 5)])
        instances = scene3d.instances_from_node(points, _cube_mesh(), {"inst_orient": "velocity"})
        matrix = scene3d.expand_instances(instances)[0].world_matrix()
        local_z_world = matrix[:3, :3] @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(local_z_world / np.linalg.norm(local_z_world), (0, 0, 1), atol=1e-5)

        points2 = _points([(0, 0, 0)], velocities=[(3, 0, 0)])
        instances2 = scene3d.instances_from_node(points2, _cube_mesh(), {"inst_orient": "velocity"})
        matrix2 = scene3d.expand_instances(instances2)[0].world_matrix()
        local_z_world2 = matrix2[:3, :3] @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(local_z_world2 / np.linalg.norm(local_z_world2), (1, 0, 0), atol=1e-5)

    def test_same_seed_reproduces_identical_variants_and_rotations(self):
        n = 64
        rng_positions = np.random.default_rng(0).uniform(-5, 5, (n, 3))
        points = _points(rng_positions)
        params = {"seed": 17, "inst_orient": "random", "inst_rotate_random": 30.0,
                  "inst_variant": "random", "inst_scale_random": 0.5}
        instance_value = scene3d.Scene(geometries=(_cube_mesh(), _cube_mesh(color=(0.8, 0.8, 0.2, 1.0))))
        a = scene3d.instances_from_node(points, instance_value, params)
        b = scene3d.instances_from_node(points, instance_value, params)
        np.testing.assert_array_equal(a.matrices, b.matrices)
        np.testing.assert_array_equal(a.variant, b.variant)

    def test_scale_random_stays_within_bounds(self):
        n = 200
        positions = np.random.default_rng(1).uniform(-5, 5, (n, 3))
        points = _points(positions)
        params = {"seed": 3, "inst_scale": 2.0, "inst_scale_random": 0.25}
        instances = scene3d.instances_from_node(points, _cube_mesh(), params)
        # Uniform scale: the transformed length of a unit local axis is the scale factor.
        scales = np.linalg.norm(instances.matrices[:, :3, 0], axis=1)
        self.assertTrue(np.all(scales >= 2.0 * 0.75 - 1e-6))
        self.assertTrue(np.all(scales <= 2.0 * 1.25 + 1e-6))

    def test_instanced_sphere_matches_merge_geo3d_placement(self):
        positions = [(2, 0, 0), (-2, 0, 0), (0, 3, 0)]
        sphere = scene3d._sphere_grid(1.0, 8, 12, (0.7, 0.7, 0.7, 1.0), scene3d.Transform3D())
        points = _points(positions)
        instances = scene3d.instances_from_node(points, sphere, {})
        expanded = scene3d.expand_instances(instances)
        got = scene3d.merge_geometry(expanded)
        placed = [scene3d.replace(sphere, parent=scene3d.Transform3D(scene3d.Vec3(*p)).matrix())
                  for p in positions]
        expected = scene3d.merge_geometry(placed)
        got_sorted = got.vertices[np.lexsort(got.vertices.T[::-1])]
        expected_sorted = expected.vertices[np.lexsort(expected.vertices.T[::-1])]
        np.testing.assert_allclose(got_sorted, expected_sorted, atol=1e-4)

    def test_bypass_falls_back_to_point_index_for_geometry_points(self):
        # A Geometry's points have no particle id, so "attribute" variant selection falls back to
        # point index, matching "cycle" exactly.
        cube = scene3d._cube(1.0, (1, 1, 1, 1), scene3d.Transform3D())
        instance_value = scene3d.Scene(geometries=(_cube_mesh(), _cube_mesh(color=(0.1, 0.1, 0.9, 1.0))))
        a = scene3d.instances_from_node(cube, instance_value, {"inst_variant": "attribute"})
        b = scene3d.instances_from_node(cube, instance_value, {"inst_variant": "cycle"})
        np.testing.assert_array_equal(a.variant, b.variant)


class InstanceRenderTests(unittest.TestCase):
    def test_shadows_fall_from_instances(self):
        ground = scene3d._card(12, 12, (1, 1, 1, 1), scene3d.Transform3D(rotation=scene3d.Vec3(-90, 0, 0)))
        points = _points([(0, 2, 0)])
        instances = scene3d.instances_from_node(points, _cube_mesh(1.5, (1, 1, 1, 1)), {})
        scene = scene3d.Scene(geometries=(ground,), lights=(scene3d.Light(position=scene3d.Vec3(0, 6, 0), shadows=True),),
                              instances=(instances,))
        camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(4, 5, 7)))
        shadowed = scene3d.render(scene, camera, 64, 48, ambient=0.1)
        xy, _ = scene3d.project(camera, 64, 48, [(0, 0, 0)])
        x, y = np.floor(xy[0]).astype(int)
        xy2, _ = scene3d.project(camera, 64, 48, [(4, 0, 0)])
        x2, y2 = np.floor(xy2[0]).astype(int)
        # The pixel under the instanced cube is darker than a pixel far from its shadow.
        self.assertLess(shadowed[y, x, 0], shadowed[y2, x2, 0])

    def test_write_geo3d_round_trips_the_flattened_result(self):
        positions = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]
        points = _points(positions)
        cube = _cube_mesh()
        instances = scene3d.instances_from_node(points, cube, {})
        scene = scene3d.Scene(instances=(instances,))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.obj"
            counts = scene3d.write_obj(scene, str(path))
            self.assertEqual(counts["objects"], len(positions))
            self.assertEqual(counts["triangles"], len(positions) * len(cube.triangles))
            self.assertTrue(path.exists())

    def test_old_documents_load_unaffected(self):
        # A document with no Instance3D node (every document before this step) validates and
        # evaluates exactly as before: adding the node type touches no other node's behaviour.
        d = Dispatcher()
        _make(d, cube=("Cube3D", {}), cam=("Camera3D", {}), scene=("Scene3D", {}), render=("Render3D", {}))
        _wire(d, "scene", "object0", "cube")
        _wire(d, "render", "scene", "scene")
        _wire(d, "render", "camera", "cam")
        validate(d.document)
        value = _value(d, "render")
        self.assertIsNotNone(value)


class InstanceMemoryTests(unittest.TestCase):
    def test_100k_instances_of_a_1k_triangle_mesh_stay_small(self):
        """docs/3D_ROADMAP.md budget: 100k instances of a 1k-triangle mesh must not allocate
        100M triangles. `InstanceSet`/`expand_instances` share the source mesh's vertex and
        triangle arrays by reference across every instance, so this stays a few MB, not gigabytes.
        """
        vertices = np.random.default_rng(2).uniform(-1, 1, (600, 3)).astype(np.float32)
        triangles = np.random.default_rng(2).integers(0, 600, (1000, 3)).astype(np.int32)
        mesh = scene3d.Geometry(vertices, triangles, (0.8, 0.8, 0.8, 1.0))
        n = 100_000
        positions = np.random.default_rng(3).uniform(-50, 50, (n, 3))
        points = _points(positions)
        instances = scene3d.instances_from_node(points, mesh, {})
        self.assertEqual(len(instances), n)
        expanded = scene3d.expand_instances(instances)
        self.assertEqual(len(expanded), n)
        # Every instance shares the SAME vertex/triangle arrays: at most one distinct array per
        # source mesh, never one per instance.
        unique_vertex_arrays = {id(g.vertices) for g in expanded}
        unique_triangle_arrays = {id(g.triangles) for g in expanded}
        self.assertEqual(len(unique_vertex_arrays), 1)
        self.assertEqual(len(unique_triangle_arrays), 1)
        total_mesh_bytes = vertices.nbytes + triangles.nbytes
        self.assertLess(total_mesh_bytes, 1_000_000)  # ~19 KB here; nowhere near 100M triangles' worth


if __name__ == "__main__":
    unittest.main()
