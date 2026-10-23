"""Lane 4, step A: translucent meshes draw far to near from a buffer of their own, one draw per run of one material.

Drawing the shared vertex buffer in depth order was a draw call per triangle (2,300 for a glass sphere, 14 ms of every played
frame on the RTX 3080 Ti). The test fails on that code: the number of draws would equal the number of triangles."""
import dataclasses
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, scene3d as s
from tests.test_3d_playback_rate import ball, camera, need_gpu, room


def glass_room():
    base = room()
    glass = ball(0.8, (0.9, 0.95, 0.95, 0.35), (0.2, 0.8, 1.2), rows=24)
    return dataclasses.replace(base, geometries=base.geometries + (glass,)), len(glass.triangles)


def _prepare_all(scene, order):
    return gpu3d._prepare_static(scene, order, None)[0]


class SortedTranslucentTests(unittest.TestCase):
    def render(self, split):
        scene, _ = glass_room()
        real = gpu3d._sorted_translucent
        seen = []

        def spy(*args):
            runs, buffer = real(*args)
            seen.append(runs)
            if split:       # the same buffer, a draw per triangle
                runs = [(first + 3*i, 3, material) for first, count, material in runs for i in range(count // 3)]
            return runs, buffer
        with unittest.mock.patch.object(gpu3d, "_sorted_translucent", spy):
            frame = gpu3d.render(scene, camera(1.0), 160, 90, (0, 0, 0, 0), 0.2, 1)
        return frame, seen

    def test_a_glass_sphere_is_one_draw_not_one_per_triangle(self):
        need_gpu(self)
        frame, seen = self.render(False)
        _, triangles = glass_room()
        self.assertGreater(triangles, 1000)
        self.assertEqual(len(seen[0]), 1)
        self.assertEqual(seen[0][0][1], 3*triangles)

    def test_the_picture_equals_the_per_triangle_draws(self):
        need_gpu(self)
        one, _ = self.render(False)
        many, _ = self.render(True)
        np.testing.assert_array_equal(np.asarray(one), np.asarray(many))

    def test_the_static_scene_keeps_the_translucent_vertices_one_row_per_triangle(self):
        scene, _ = glass_room()
        order = gpu3d.link_order(scene)
        _, _, _, translucent, _ = gpu3d._prepare_static(scene, order, None)
        self.assertEqual(translucent["vertices"].shape, (len(translucent["start"]), 3, 38))
        vertices = _prepare_all(scene, order)
        np.testing.assert_array_equal(translucent["vertices"][5], vertices[translucent["start"][5]:translucent["start"][5] + 3])


if __name__ == "__main__":
    unittest.main()
