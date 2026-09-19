"""Pixel-level camera projection tests, using only NumPy and the scene3d API."""
from dataclasses import replace
import unittest

import numpy as np

from nodebased import scene3d as s


def camera(z=4, x=0, target=None, fov=90, **kwargs):
    return s.Camera(s.Transform3D(s.Vec3(x, 0, z)), target or s.Vec3(), fov=fov, **kwargs)


def card(size=8, z=0, **kwargs):
    return replace(s._card(size, size, (1, 1, 1, 1),
                           s.Transform3D(s.Vec3(0, 0, z))), **kwargs)


def quadrants(height=32, width=32):
    colours = np.array([[(1, 0, 0, 1), (0, 1, 0, 1)],
                        [(0, 0, 1, 1), (1, 1, 1, 1)]], np.float32)
    return colours.repeat(height // 2, 0).repeat(width // 2, 1)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.camera = camera()
        self.texture = quadrants()
        self.projection = s.Projection(self.camera, self.texture)

    def render(self, geometry, render_camera=None, **kwargs):
        return s.render(s.Scene((geometry,)), render_camera or self.camera,
                        64, 64, samples=1, **kwargs)

    def test_same_camera_is_upright_and_not_mirrored(self):
        geometry = s.apply_projection(card(), self.projection)
        image = self.render(geometry)
        for y in (16, 48):
            for x in (16, 48):
                np.testing.assert_array_equal(image[y, x], self.texture[y // 2, x // 2])

    def test_projection_sticks_to_world_points_with_moving_render_camera(self):
        moved = camera(x=3)
        geometry = s.apply_projection(card(16, uvs=None), self.projection)
        image = self.render(geometry, moved)
        points = np.array([(-2, 2, 0), (2, 2, 0), (-2, -2, 0), (2, -2, 0)], np.float32)
        projected, _ = s.project(self.camera, 32, 32, points)
        rendered, _ = s.project(moved, 64, 64, points)
        for (u, v), (x, y) in zip(projected.astype(int), rendered.astype(int)):
            np.testing.assert_array_equal(image[y, x], self.texture[v, u])

    def test_perspective_fixed_card_samples_half_the_texture_span(self):
        # A horizontal ramp makes the texture coordinates observable in rendered pixels.
        texture = np.ones((64, 64, 4), np.float32)
        texture[..., :3] = ((np.arange(64) + .5) / 64)[None, :, None]
        projection = s.Projection(self.camera, texture)
        near = self.render(s.apply_projection(card(), projection))
        # Move the render camera with the card to keep its on-screen size fixed.
        far = self.render(s.apply_projection(card(z=-4), projection),
                          camera(z=0, target=s.Vec3(0, 0, -4)))
        near_span = near[24, 48, 0] - near[24, 16, 0]
        far_span = far[24, 48, 0] - far[24, 16, 0]
        self.assertAlmostEqual(float(far_span), float(near_span) / 2, places=6)
        # The full frustum footprint conversely doubles in world size at twice the depth.
        extents = []
        for z in (0, -4):
            image = self.render(s.apply_projection(card(40, z=z), self.projection),
                                camera(z=z + 16, target=s.Vec3(0, 0, z)))
            extents.append(np.count_nonzero(image[32, :, 3]))
        self.assertEqual(extents, [16, 32])

    def test_outside_transparent_or_clamp_and_no_depth_write(self):
        render_camera = camera(z=8)
        for mode in ('transparent', 'clamp'):
            geometry = s.apply_projection(card(20), replace(self.projection, outside=mode))
            image, depth = self.render(geometry, render_camera, return_depth=True)
            if mode == 'transparent':
                np.testing.assert_array_equal(image[8, 8], 0)
                self.assertTrue(np.isinf(depth[8, 8]))
                back = card(24, z=-1, color=(.25, .5, .75, 1))
                composited = s.render(s.Scene((geometry, back)), render_camera, 64, 64)
                np.testing.assert_array_equal(composited[8, 8], back.color)
            else:
                np.testing.assert_array_equal(image[8, 8], (1, 0, 0, 1))
                self.assertTrue(np.isfinite(depth[8, 8]))

    def test_behind_near_and_far_projection_planes_are_transparent(self):
        projector = replace(self.projection, camera=replace(self.camera, near=1, far=6))
        for z in (5, 3, -2, -3):
            with self.subTest(z=z):
                geometry = s.apply_projection(card(20, z=z), projector)
                image, depth = self.render(geometry, camera(z=10), return_depth=True)
                np.testing.assert_array_equal(image, 0)
                self.assertTrue(np.isinf(depth).all())

    def test_clamp_rejects_at_and_behind_projector(self):
        for z in (4, 5):
            geometry = s.apply_projection(card(20, z=z), replace(self.projection, outside='clamp'))
            image, depth = self.render(geometry, camera(z=10), return_depth=True)
            np.testing.assert_array_equal(image, 0)
            self.assertTrue(np.isinf(depth).all())

    def test_backface_skip_uses_unflipped_flat_and_interpolated_normals(self):
        for normals in (None, np.tile((0, 0, -1), (4, 1)).astype(np.float32)):
            geometry = card(normals=normals)
            if normals is None:
                geometry = replace(geometry, triangles=geometry.triangles[:, ::-1])
            for mode in ('skip', 'project'):
                projected = s.apply_projection(geometry, replace(self.projection, backfaces=mode))
                for output in ('rgba', 'normals', 'depth'):
                    image, depth = self.render(projected, shade=True, output=output, return_depth=True)
                    self.assertEqual(float(image[16, 16, 3]), 0 if mode == 'skip' else 1)
                    self.assertEqual(bool(np.isinf(depth[16, 16])), mode == 'skip')

    def test_alpha_and_colour_tint_remain_premultiplied(self):
        texture = np.full((16, 16, 4), (.5, .25, 0, .5), np.float32)
        texture[:8, :8] = 0
        geometry = s.apply_projection(card(color=(1, .5, 1, .5)),
                                      s.Projection(self.camera, texture))
        image, depth = self.render(geometry, return_depth=True)
        np.testing.assert_array_equal(image[16, 16], 0)
        np.testing.assert_array_equal(image[16, 48], (.25, .0625, 0, .25))
        self.assertTrue(np.isinf(depth).all())
        self.assertEqual(image.dtype, np.float32)

    def test_projected_mips_override_mesh_texture_and_uvs(self):
        checks = (np.indices((256, 256)).sum(0) % 2).astype(np.float32)
        texture = np.repeat(checks[..., None], 4, -1)
        texture[..., 3] = 1
        geometry = card(uvs=None, texture=quadrants())
        image = self.render(s.apply_projection(geometry, s.Projection(self.camera, texture)))
        np.testing.assert_allclose(image[16, 16], (.5, .5, .5, 1), atol=1e-6)

    def test_texture_aspect_is_projection_filmback(self):
        projection = s.Projection(self.camera, quadrants(32, 64))
        image = self.render(s.apply_projection(card(32), projection), camera(z=16))
        self.assertEqual(np.count_nonzero(image[32, :, 3]), 32)
        self.assertEqual(np.count_nonzero(image[:, 32, 3]), 16)
        np.testing.assert_array_equal(image[28, 20], (1, 0, 0, 1))

    def test_apply_projection_preserves_nested_transforms_without_mutation(self):
        def node(**changes):
            params = dict(tx=0, ty=0, tz=0, rx=0, ry=0, rz=0, sx=1, sy=1, sz=1)
            params.update(changes)
            return {'params': params}
        original = card(2)
        light = s.Light(intensity=0)
        inner = s.scene_from_node(node(tx=1), (original, light))
        outer = s.scene_from_node(node(sx=2), (inner,))
        projected = s.apply_projection(outer, self.projection)
        self.assertIsNot(projected, outer)
        self.assertIsNot(projected.geometries[0], outer.geometries[0])
        for geometry in (original, inner.geometries[0], outer.geometries[0]):
            self.assertIs(geometry.projection, None)
        self.assertIs(projected.geometries[0].projection, self.projection)
        self.assertIs(projected.lights, outer.lights)
        self.assertIs(projected.geometries[0].parent, outer.geometries[0].parent)
        np.testing.assert_array_equal(projected.geometries[0].world_matrix(),
                                      outer.geometries[0].world_matrix())
        np.testing.assert_array_equal(projected.geometries[0].world_matrix()[:3, 3], (2, 0, 0))
        # A projected geometry also survives subsequent scene assembly.
        regrouped = s.scene_from_node(node(ty=1), (projected,))
        self.assertIs(regrouped.geometries[0].projection, self.projection)
        image = s.render(projected, self.camera, 64, 64)
        np.testing.assert_array_equal(image[24, 48], (0, 1, 0, 1))


if __name__ == '__main__':
    unittest.main()
