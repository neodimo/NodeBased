"""CPU coverage at shared edges and vertices, without a GPU dependency."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import scene3d as s


class FillRuleTests(unittest.TestCase):
    camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 5)), fov=90)
    splits = (((0, 1, 2), (0, 2, 3)), ((0, 1, 3), (1, 2, 3)))
    square = ((0, 0), (64, 0), (64, 64), (0, 64))
    color = (.2, .4, .8, .25)

    def geometry(self, points, triangles, alpha=.25):
        # World coordinates chosen to project to the given 64x64 screen points.
        points = np.asarray(points, dtype=np.float32)
        vertices = np.column_stack(((points[:, 0]/32-1)*5,
                                    (1-points[:, 1]/32)*5, np.zeros(len(points))))
        return s.Geometry(vertices.astype('f4'), np.asarray(triangles), (*self.color[:3], alpha))

    def render(self, *geometries, samples=1):
        return s.render(s.Scene(geometries), self.camera, 64, 64, samples=samples)

    def uniform(self, image, alpha=.25):
        expected = np.empty_like(image)
        expected[...] = (*np.multiply(self.color[:3], alpha), alpha)
        np.testing.assert_allclose(image, expected, atol=1e-6, rtol=0)

    def test_card_splits_windings_samples_and_draw_order(self):
        for samples in (1, 2, 3):
            for split in self.splits:
                # Also cover mixed winding: ownership must be local to each triangle.
                for reverse_a in (False, True):
                    for reverse_b in (False, True):
                        with self.subTest(samples=samples, split=split, winding=(reverse_a, reverse_b)):
                            triangles = [t[::-1] if reverse else t
                                         for t, reverse in zip(split, (reverse_a, reverse_b))]
                            card = self.geometry(self.square, triangles)
                            image = self.render(card, samples=samples)
                            self.uniform(image)
                            np.testing.assert_array_equal(image, self.render(
                                replace(card, triangles=card.triangles[::-1]), samples=samples))

    def test_abutting_cards(self):
        # The shared horizontal/vertical edge passes through pixel centres.
        for vertical in (False, True):
            left = ((0, 0), (32.5, 0), (32.5, 64), (0, 64))
            right = ((32.5, 0), (64, 0), (64, 64), (32.5, 64))
            if not vertical:
                left, right = [tuple((y, x) for x, y in points) for points in (left, right)]
            for alpha in (.25, 1):
                for samples in (1, 2, 3):
                    for split in self.splits:
                        with self.subTest(vertical=vertical, alpha=alpha, samples=samples, split=split):
                            a = self.geometry(left, split, alpha)
                            b = self.geometry(right, [t[::-1] for t in split], alpha)
                            image = self.render(a, b, samples=samples)
                            self.uniform(image, alpha)
                            np.testing.assert_array_equal(image, self.render(b, a, samples=samples))

    def test_triangle_fan_shared_vertex(self):
        points = (*self.square, (32.5, 32.5))
        for winding in (1, -1):
            triangles = [tuple((i, (i+1) % 4, 4))[::winding] for i in range(4)]
            for samples in (1, 2, 3):
                with self.subTest(winding=winding, samples=samples):
                    card = self.geometry(points, triangles)
                    image = self.render(card, samples=samples)
                    self.uniform(image)
                    np.testing.assert_array_equal(image, self.render(
                        replace(card, triangles=card.triangles[::-1]), samples=samples))

    def test_asymmetric_triangles_roundoff_and_side_ownership(self):
        # Non-power-of-two areas give the old 1-wa-wb residual on the diagonal.
        # The unequal third vertices must not change either edge's tolerance.
        points = ((.125, .125), (63.875, .125), (63.875, 63.875), (.125, 49.875))
        for split in self.splits:
            for winding in (1, -1):
                card = self.geometry(points, [t[::winding] for t in split])
                image = self.render(card)
                self.uniform(image[1:48, 1:63])
                self.uniform(image[image[..., 3] > 0])
                np.testing.assert_array_equal(image, self.render(replace(card, triangles=card.triangles[::-1])))

    def test_outer_edges_top_left_and_opaque_interior(self):
        for split in self.splits:
            for winding in (1, -1):
                card = self.geometry(((8.5, 8.5), (55.5, 8.5), (55.5, 55.5), (8.5, 55.5)),
                                     [t[::winding] for t in split], alpha=1)
                image = self.render(card)
                expected = np.zeros((64, 64), 'f4')
                expected[8:55, 8:55] = 1  # top/left included; bottom/right excluded
                np.testing.assert_array_equal(image[..., 3], expected)
                self.uniform(image[9:55, 9:55], alpha=1)
                # Former inclusive coverage agrees everywhere away from the edges.
                old = np.zeros_like(expected)
                old[8:56, 8:56] = 1
                away = np.ones_like(expected, dtype=bool)
                away[[8, 55], :] = False
                away[:, [8, 55]] = False
                np.testing.assert_array_equal(image[..., 3][away], old[away])

    def test_sub_roundoff_edge_offsets_have_complementary_ownership(self):
        # Exercise the tolerance band directly at projection precision, on both
        # sides of zero, without letting float32 projection round away the offset.
        card = self.geometry(self.square, self.splits[0])
        project = s._to_pixels
        for offset in (-1e-14, 1e-14):
            def shifted(*args):
                pixels = project(*args).astype('f8')
                pixels[:, 0] += offset
                return pixels
            with self.subTest(offset=offset), patch.object(s, '_to_pixels', side_effect=shifted):
                self.uniform(self.render(card))


if __name__ == '__main__':
    unittest.main()
