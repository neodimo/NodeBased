"""Shrinkwrap3D (lane L4, DiMo 9/27): fits a proxy mesh onto a target's surface.

See docs/3D_FOUNDATION.md, the `Shrinkwrap3D` row.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SPECS, bypass_slot, validate
from nodebased.imaging import Evaluator
from nodebased import scene3d


def _make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def _wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def _value(d, key):
    return Evaluator().evaluate_raster(d.document, key, typed=True)


def _torus(major_radius, minor_radius, u_segments=32, v_segments=16):
    """A raw torus Geometry, lying flat in the XZ plane: no `Torus3D` node exists to build one."""
    u = np.linspace(0, 2 * np.pi, u_segments + 1)
    v = np.linspace(0, 2 * np.pi, v_segments + 1)
    uu, vv = np.meshgrid(u, v)
    x = (major_radius + minor_radius * np.cos(vv)) * np.cos(uu)
    z = (major_radius + minor_radius * np.cos(vv)) * np.sin(uu)
    y = minor_radius * np.sin(vv)
    vertices = np.stack((x, y, z), -1).reshape(-1, 3).astype(np.float32)
    cols = u_segments + 1
    tris = []
    for i in range(v_segments):
        for j in range(u_segments):
            a, b = i * cols + j, i * cols + j + 1
            c, d = a + cols, b + cols
            tris += [(a, b, d), (a, d, c)]
    return scene3d.Geometry(vertices, np.array(tris, np.int32), (0.8, 0.8, 0.8, 1.0))


def _torus_distance(points, major_radius, minor_radius):
    radial = np.sqrt(points[:, 0] ** 2 + points[:, 2] ** 2) - major_radius
    return np.abs(np.sqrt(radial ** 2 + points[:, 1] ** 2) - minor_radius)


def _row_sorted(vertices):
    """Row order does not survive an OBJ round trip; sort whole rows lexicographically to compare sets."""
    return vertices[np.lexsort(vertices.T[::-1])]


class Shrinkwrap3DTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()

    def test_registered_types_and_old_documents(self):
        self.assertEqual(OUTPUT_TYPES["Shrinkwrap3D"], "geometry")
        self.assertEqual(SPECS["Shrinkwrap3D"]["inputs"], ["target"])
        self.assertEqual(SPECS["Shrinkwrap3D"]["optional_inputs"], ["proxy"])
        self.assertEqual(INPUT_TYPES["target"], ("geometry", "scene"))
        self.assertEqual(INPUT_TYPES["proxy"], ("geometry",))
        validate(self.d.document)

    def test_bypass_passes_the_target(self):
        _make(self.d, sph=("Sphere3D", {"sphere_radius": 2.0}),
              wrap=("Shrinkwrap3D", {"wrap_resolution": 8}))
        _wire(self.d, "wrap", "target", "sph")
        self.assertEqual(bypass_slot(self.d.document["nodes"]["wrap"]), "target")
        self.d.execute({"op": "disable", "id": "wrap", "value": True})
        np.testing.assert_array_equal(_value(self.d, "wrap").vertices, _value(self.d, "sph").vertices)

    def test_wrap_onto_a_sphere_lands_on_the_surface(self):
        _make(self.d, sph=("Sphere3D", {"sphere_radius": 2.0, "rows": 24, "columns": 48}),
              wrap=("Shrinkwrap3D", {"wrap_shape": "sphere", "wrap_resolution": 10}))
        _wire(self.d, "wrap", "target", "sph")
        g = _value(self.d, "wrap")
        distances = np.linalg.norm(g.vertices.astype(np.float64), axis=1)
        self.assertLess(np.abs(distances - 2.0).max(), 0.02)
        self.assertEqual(g.uvs.shape, g.vertices.shape[:1] + (2,))
        self.assertEqual(g.normals.shape, g.vertices.shape)

    def test_wrap_onto_a_torus_lands_on_the_surface(self):
        target = _torus(2.0, 0.5, u_segments=32, v_segments=16)
        params = {**SPECS["Shrinkwrap3D"]["params"],
                  "wrap_shape": "sphere", "wrap_resolution": 12,
                  "red": 0.8, "green": 0.8, "blue": 0.8, "alpha": 1.0}
        wrapped = scene3d.shrinkwrap_geometry(target, None, params)
        errors = _torus_distance(wrapped.vertices.astype(np.float64), 2.0, 0.5)
        self.assertLess(errors.max(), 0.05)

    def test_project_mode_also_lands_on_the_surface(self):
        _make(self.d, sph=("Sphere3D", {"sphere_radius": 2.0, "rows": 24, "columns": 48}),
              wrap=("Shrinkwrap3D", {"wrap_shape": "sphere", "wrap_resolution": 12, "wrap_mode": "project"}))
        _wire(self.d, "wrap", "target", "sph")
        g = _value(self.d, "wrap")
        distances = np.linalg.norm(g.vertices.astype(np.float64), axis=1)
        self.assertLess(np.abs(distances - 2.0).max(), 0.05)

    def test_uvs_are_unchanged_by_wrapping(self):
        _make(self.d, target=("Cube3D", {"cube_size": 6.0}),
              proxy=("Sphere3D", {"sphere_radius": 1.0, "rows": 6, "columns": 8}),
              wrap=("Shrinkwrap3D", {}))
        _wire(self.d, "wrap", "target", "target")
        _wire(self.d, "wrap", "proxy", "proxy")
        wrapped, proxy = _value(self.d, "wrap"), _value(self.d, "proxy")
        np.testing.assert_array_equal(wrapped.uvs, proxy.uvs)
        self.assertEqual(wrapped.vertices.shape, proxy.vertices.shape)

    def test_smoothing_moves_positions_but_keeps_the_uvs(self):
        _make(self.d, target=("Cube3D", {"cube_size": 6.0}),
              proxy=("Sphere3D", {"sphere_radius": 1.0, "rows": 6, "columns": 8}),
              flat=("Shrinkwrap3D", {}), smoothed=("Shrinkwrap3D", {"wrap_smooth_iterations": 3}))
        for key in ("flat", "smoothed"):
            _wire(self.d, key, "target", "target")
            _wire(self.d, key, "proxy", "proxy")
        flat, smoothed = _value(self.d, "flat"), _value(self.d, "smoothed")
        np.testing.assert_array_equal(flat.uvs, smoothed.uvs)
        self.assertGreater(np.abs(flat.vertices - smoothed.vertices).max(), 1e-6)

    def test_offset_moves_each_vertex_by_that_distance_along_the_normal(self):
        _make(self.d, sph=("Sphere3D", {"sphere_radius": 2.0}),
              zero=("Shrinkwrap3D", {"wrap_resolution": 10, "wrap_offset": 0.0}),
              offset=("Shrinkwrap3D", {"wrap_resolution": 10, "wrap_offset": 0.3}))
        for key in ("zero", "offset"):
            _wire(self.d, key, "target", "sph")
        base, moved = _value(self.d, "zero"), _value(self.d, "offset")
        distance = np.linalg.norm((moved.vertices - base.vertices).astype(np.float64), axis=1)
        np.testing.assert_allclose(distance, 0.3, atol=1e-4)

    def test_falloff_blends_toward_the_original_proxy_shape(self):
        _make(self.d, target=("Cube3D", {"cube_size": 6.0}),
              proxy=("Sphere3D", {"sphere_radius": 1.0, "rows": 6, "columns": 8}),
              full=("Shrinkwrap3D", {"wrap_falloff": 1.0}), half=("Shrinkwrap3D", {"wrap_falloff": 0.5}))
        for key in ("full", "half"):
            _wire(self.d, key, "target", "target")
            _wire(self.d, key, "proxy", "proxy")
        original, wrapped, blended = (_value(self.d, "proxy"), _value(self.d, "full"), _value(self.d, "half"))
        expected = original.vertices.astype(np.float64) + (
            wrapped.vertices.astype(np.float64) - original.vertices.astype(np.float64)) * 0.5
        np.testing.assert_allclose(blended.vertices.astype(np.float64), expected, atol=1e-5)

    def test_renders_and_round_trips_through_write_geo3d(self):
        _make(self.d, sph=("Sphere3D", {"sphere_radius": 2.0}),
              wrap=("Shrinkwrap3D", {"wrap_resolution": 8, "wrap_shape": "box"}),
              cam=("Camera3D", {"tz": 8.0}), scene=("Scene3D", {}),
              render=("Render3D", {"width": 32, "height": 32, "samples": 1}))
        _wire(self.d, "wrap", "target", "sph")
        _wire(self.d, "scene", "object0", "wrap")
        _wire(self.d, "render", "scene", "scene")
        _wire(self.d, "render", "camera", "cam")
        pixels = np.asarray(Evaluator().evaluate(self.d.document, target="render"))
        self.assertGreater(float(pixels[..., 3].max()), 0.5)

        wrapped = _value(self.d, "wrap")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrap.obj"
            scene3d.write_obj(scene3d.Scene((wrapped,)), path)
            _make(self.d, read=("ReadGeo3D", {"geo_path": str(path)}))
            reread = _value(self.d, "read")
        self.assertEqual(len(reread.vertices), len(wrapped.vertices))
        np.testing.assert_allclose(_row_sorted(reread.vertices), _row_sorted(wrapped.vertices), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
