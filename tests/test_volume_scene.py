"""Lane L6 fluids plan, step A part 1: the Volume scene member, analytic_plume, and how graphs carry it."""
import unittest

import numpy as np

from nodebased import scene3d
from nodebased.core import INPUT_TYPES, OUTPUT_TYPES, SPECS, Dispatcher, bypass_slot
from nodebased.imaging import Evaluator


def box(n=16, density=1.0, temperature=None, velocity=None, size=1.0, matrix=None):
    """A uniform cube of smoke centred on the origin, `size` units on a side."""
    kwargs = {}
    if matrix is not None:
        kwargs["matrix"] = matrix
    return scene3d.Volume(np.full((n, n, n), density, np.float32), size / n, (-size / 2,) * 3,
                          temperature=None if temperature is None else np.full((n, n, n), temperature, np.float32),
                          velocity=None if velocity is None else np.broadcast_to(np.float32(velocity), (n, n, n, 3)),
                          **kwargs)


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


class VolumeTests(unittest.TestCase):
    def test_fields_are_validated_and_stored_as_float32(self):
        volume = scene3d.Volume(np.ones((3, 4, 5)), .5, (0, 1, 2), velocity=np.zeros((3, 4, 5, 3)),
                                temperature=np.ones((3, 4, 5)))
        self.assertEqual(volume.shape, (3, 4, 5))
        self.assertEqual(volume.density.dtype, np.float32)
        self.assertEqual(volume.velocity.shape, (3, 4, 5, 3))
        self.assertEqual(volume.origin, (0.0, 1.0, 2.0))
        np.testing.assert_array_equal(volume.matrix, np.eye(4))
        with self.assertRaises(ValueError):
            scene3d.Volume(np.ones((3, 4)))
        with self.assertRaises(ValueError):
            scene3d.Volume(np.ones((2, 2, 2)), velocity=np.zeros((2, 2, 2)))
        with self.assertRaises(ValueError):
            scene3d.Volume(np.ones((2, 2, 2)), temperature=np.ones((3, 2, 2)))
        with self.assertRaises(ValueError):
            scene3d.Volume(np.ones((2, 2, 2)), voxel_size=0)

    def test_fingerprint_follows_content_only(self):
        a, b = scene3d.analytic_plume(12, 3), scene3d.analytic_plume(12, 3)
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), scene3d.analytic_plume(12, 4).fingerprint())
        density = a.density.copy()
        density[3, 3, 3] += 1e-3
        edited = scene3d.Volume(density, a.voxel_size, a.origin, a.matrix, a.temperature, a.velocity)
        self.assertNotEqual(a.fingerprint(), edited.fingerprint())
        moved = scene3d.Volume(a.density, a.voxel_size, a.origin, np.diag((2, 2, 2, 1)), a.temperature, a.velocity)
        self.assertNotEqual(a.fingerprint(), moved.fingerprint())
        self.assertNotEqual(a.fingerprint(), scene3d.Volume(a.density, a.voxel_size, a.origin, a.matrix).fingerprint())

    def test_analytic_plume_is_deterministic_and_smoke_like(self):
        a, b = scene3d.analytic_plume(16, 7), scene3d.analytic_plume(16, 7)
        for name in ("density", "temperature", "velocity"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        self.assertEqual(a.shape, (16, 16, 16))
        self.assertEqual(a.velocity.shape, (16, 16, 16, 3))
        self.assertGreater(float(a.density.max()), .3)
        self.assertGreaterEqual(float(a.density.min()), 0.0)
        self.assertGreater(float(a.velocity[..., 1].mean()), .5)          # it rises
        self.assertFalse(np.array_equal(a.density, scene3d.analytic_plume(16, 8).density))
        # Narrow at the bottom, wider at the top: the column's mass moves outward with height.
        low, high = a.density[:, 2, :], a.density[:, 12, :]
        self.assertGreater(int((high > .02).sum()), int((low > .02).sum()))
        with self.assertRaises(ValueError):
            scene3d.analytic_plume(2)


class SceneCarryTests(unittest.TestCase):
    def test_scene_from_node_applies_the_node_matrix_and_nests(self):
        volume = box()
        inner = scene3d.scene_from_node({"params": {**SPECS["Scene3D"]["params"], "tx": 1.0}}, [volume])
        outer = scene3d.scene_from_node({"params": {**SPECS["Scene3D"]["params"], "ty": 2.0}}, [inner])
        self.assertEqual(len(outer.volumes), 1)
        np.testing.assert_allclose(outer.volumes[0].matrix[:3, 3], (1, 2, 0))
        np.testing.assert_array_equal(outer.volumes[0].density, volume.density)

    def test_graph_carries_the_volume_through_scene3d_and_axis3d(self):
        d = Dispatcher()
        make(d, p=("Plume3D", {"plume_resolution": 8, "tx": 1.0}), a=("Axis3D", {"ty": 3.0}),
             s=("Scene3D", {"tz": -4.0}))
        wire(d, "a", "object", "p")
        wire(d, "s", "object0", "a")
        result = Evaluator().evaluate_raster(dict(d.document, view="s"), target="s", typed=True)
        self.assertEqual(len(result.volumes), 1)
        np.testing.assert_allclose(result.volumes[0].matrix[:3, 3], (1, 3, -4))

    def test_node_types_route_volumes(self):
        self.assertEqual(OUTPUT_TYPES["Plume3D"], "volume")
        for slot in ("object", *(f"object{i}" for i in range(8))):
            self.assertIn("volume", INPUT_TYPES[slot])
        for slot in (*(f"geo{i}" for i in range(8)), "geo", "geometry", "scene", "light0", "particles"):
            self.assertNotIn("volume", INPUT_TYPES[slot])

    def test_merge_geo3d_refuses_a_volume_by_type(self):
        d = Dispatcher()
        make(d, p=("Plume3D", {}), m=("MergeGeo3D", {}))
        with self.assertRaises(ValueError):
            wire(d, "m", "geo0", "p")
        self.assertIsNone(d.document["nodes"]["m"]["inputs"].get("geo0"))

    def test_plume3d_is_a_source_like_light3d(self):
        d = Dispatcher()
        make(d, p=("Plume3D", {}))
        self.assertIsNone(bypass_slot(d.document["nodes"]["p"]))
        with self.assertRaises(ValueError):
            d.execute({"op": "disable", "id": "p", "value": True})
