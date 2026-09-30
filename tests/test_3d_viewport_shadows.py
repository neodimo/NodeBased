"""R6 (docs/SPLAT_RELIGHTING.md, "Production look") and Y2 of 2 (issue #4): the interactive
viewport's shadow atlas, up to `viewportgpu.MAX_SHADOW_LIGHTS` lights at once
(`viewportgpu.shadow_view_proj`, `_shadow_lights`, `_mesh_bounds`, `_render_shadow_map`).
Directional and Spot only (Point needs a cube map, not built); opaque meshes cast, meshes and
splats both receive. GPU cases skip without an adapter.
"""
import os
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import gpu3d, intrinsics as I, scene3d as s, splats, viewportgpu

BACKGROUND = (0.02, 0.02, 0.03, 1.0)
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 4, 9)), s.Vec3(0, 0, 0), 45.0, 0.1, 100.0)


def _floor():
    return replace(s._card(8, 8, (0.8, 0.8, 0.8, 1.0), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0))),
                   material="pbr", metallic=0.0, pbr_roughness=0.9, pbr_specular=0.5)


def _cube():
    return replace(s._cube(1.5, (0.8, 0.2, 0.2, 1.0), s.Transform3D(s.Vec3(0, 0.75, 0))),
                   material="pbr", metallic=0.0, pbr_roughness=0.5, pbr_specular=0.5)


def _splat_floor(n=24):
    """A flat grid of splats standing in for a floor, so shadow receiving on splats can be checked
    against the same occluder as the mesh floor above."""
    xs, zs = np.meshgrid(np.linspace(-4, 4, n), np.linspace(-4, 4, n))
    positions = np.stack((xs.ravel(), np.full(xs.size, -1.0), zs.ravel()), axis=1).astype(np.float32)
    count = len(positions)
    quats = np.tile((1.0, 0.0, 0.0, 0.0), (count, 1)).astype(np.float32)
    scales = np.tile((0.12, 0.12, 0.02), (count, 1)).astype(np.float32)
    sh_dc = np.zeros((count, 1, 3), np.float32)
    sh_dc[:, 0, :] = (0.8 - 0.5) / splats.C0
    cloud = splats.SplatCloud(positions, scales, quats, np.full(count, 0.99, np.float32), sh_dc, 0,
                              colorspace="linear")
    normals = cloud.normals()
    layer = I.Intrinsics(np.tile((0.8, 0.8, 0.8), (count, 1)), np.full(count, 0.9), normals, np.ones(count),
                         np.ones(count), np.ones(count), np.zeros(3), np.zeros((0, 3)), np.zeros((0, 3)), 0)
    return I.attach(cloud, layer)


class ShadowGeometryIsCPUOnly(unittest.TestCase):
    """`_shadow_lights`, `_mesh_bounds` and `shadow_view_proj`: no GPU needed, so these always run."""

    def test_the_brightest_shadow_enabled_lights_win_first(self):
        dim = s.Light(kind="Directional", intensity=0.5, shadows=True)
        bright = s.Light(kind="Spot", intensity=4.0, shadows=True)
        off = s.Light(kind="Directional", intensity=10.0, shadows=False)   # brightest, but not enabled
        point = s.Light(kind="Point", intensity=20.0, shadows=True)        # brightest of all, but unsupported
        lights = [(light, *light.world()) for light in (dim, bright, off, point)]
        chosen = viewportgpu._shadow_lights(lights)
        self.assertEqual(len(chosen), 2)
        (first_index, first_light, _p1, _d1), (second_index, second_light, _p2, _d2) = chosen
        self.assertEqual(first_index, 1)
        self.assertIs(first_light, bright)
        self.assertEqual(second_index, 0)
        self.assertIs(second_light, dim)

    def test_no_qualifying_light_returns_empty(self):
        lights = [(s.Light(kind="Directional", shadows=False), *s.Light().world()),
                 (s.Light(kind="Point", shadows=True), *s.Light().world())]
        self.assertEqual(viewportgpu._shadow_lights(lights), [])

    def test_more_than_max_shadow_lights_keeps_only_the_brightest(self):
        lights = [(s.Light(kind="Directional", intensity=float(i + 1), shadows=True), *s.Light().world())
                 for i in range(viewportgpu.MAX_SHADOW_LIGHTS + 3)]
        chosen = viewportgpu._shadow_lights(lights)
        self.assertEqual(len(chosen), viewportgpu.MAX_SHADOW_LIGHTS)
        intensities = [light.intensity for _index, light, _p, _d in chosen]
        self.assertEqual(intensities, sorted(intensities, reverse=True))
        self.assertEqual(min(intensities), len(lights) - viewportgpu.MAX_SHADOW_LIGHTS + 1)

    def test_mesh_bounds_covers_the_scenes_geometry_and_ignores_splats_only_scenes(self):
        scene = s.Scene((_floor(), _cube()))
        low, high = viewportgpu._mesh_bounds(scene)
        np.testing.assert_allclose(low, (-4, -1, -4), atol=1e-5)
        np.testing.assert_allclose(high, (4, 1.5, 4), atol=1e-5)
        self.assertIsNone(viewportgpu._mesh_bounds(s.Scene()))

    def test_a_point_behind_the_caster_along_the_light_projects_inside_its_own_footprint(self):
        """A basic sanity check on `shadow_view_proj`'s convention: a point straight behind the
        cube's centre (along the light's own travel direction) lands at very nearly the same NDC
        x/y as the cube's own centre, and strictly deeper (larger NDC z, farther from the light)."""
        light = s.Light(kind="Directional", intensity=1.5, position=s.Vec3(4, 6, 2), target=s.Vec3(0, 0, 0),
                        shadows=True)
        position, direction = light.world()
        bounds = (np.array((-3.0, -1.2, -3.0)), np.array((3.0, 3.0, 3.0)))
        vp = viewportgpu.shadow_view_proj(light, position, direction, bounds)

        def ndc(point):
            clip = vp @ np.array((*point, 1.0))
            return clip[:3] / clip[3]

        centre = ndc((0.0, 0.75, 0.0))
        behind = ndc((0.0, 0.75, 0.0) + direction)
        np.testing.assert_allclose(behind[:2], centre[:2], atol=1e-6)   # same light-space x/y
        self.assertGreater(behind[2], centre[2])                       # farther from the light

    def test_spot_projection_uses_the_cone_as_its_field_of_view(self):
        light = s.Light(kind="Spot", intensity=2.0, position=s.Vec3(0, 5, 0), target=s.Vec3(0, 0, 0),
                        shadows=True, cone_angle=20.0, cone_penumbra_angle=5.0)
        position, direction = light.world()
        vp = viewportgpu.shadow_view_proj(light, position, direction, (np.array((-1., -1., -1.)),
                                                                        np.array((1., 1., 1.))))
        # A point on the ground at the edge of the (wide) cone must still fall inside the map, and a
        # point well outside the cone (a Directional-sized ortho box would still show it) must not.
        inside = vp @ np.array((0.6, 0.0, 0.0, 1.0))
        outside = vp @ np.array((20.0, 0.0, 0.0, 1.0))
        self.assertLess(abs(inside[0] / inside[3]), 1.0)
        self.assertGreater(abs(outside[0] / outside[3]), 1.0)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class ShadowMapRendering(unittest.TestCase):
    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def _render(self, geometries, splats_=(), shadows=True, kind="Directional", **light_kw):
        light = s.Light(kind=kind, intensity=light_kw.pop("intensity", 1.5), shadows=shadows,
                        position=s.Vec3(4, 6, 2), target=s.Vec3(0, 0, 0), **light_kw)
        scene = s.Scene(geometries, (light,), splats=splats_)
        return self.gpu.render(scene, CAMERA, 200, 150, BACKGROUND, headlight=False, ambient=0.1)

    def test_a_mesh_floor_darkens_behind_an_occluding_mesh(self):
        on = self._render((_floor(), _cube()), shadows=True)
        off = self._render((_floor(), _cube()), shadows=False)
        diff = np.abs(on[..., :3].astype(int) - off[..., :3].astype(int))
        self.assertGreater(int((diff.max(axis=2) > 20).sum()), 200)

    def test_a_spot_light_also_casts(self):
        on = self._render((_floor(), _cube()), shadows=True, kind="Spot", intensity=8.0,
                          cone_angle=60.0, cone_penumbra_angle=10.0)
        off = self._render((_floor(), _cube()), shadows=False, kind="Spot", intensity=8.0,
                           cone_angle=60.0, cone_penumbra_angle=10.0)
        diff = np.abs(on[..., :3].astype(int) - off[..., :3].astype(int))
        self.assertGreater(int((diff.max(axis=2) > 20).sum()), 100)

    def test_a_point_light_never_casts(self):
        on = self._render((_floor(), _cube()), shadows=True, kind="Point", intensity=6.0)
        off = self._render((_floor(), _cube()), shadows=False, kind="Point", intensity=6.0)
        np.testing.assert_array_equal(on, off)

    def test_splats_receive_the_same_shadow_a_mesh_floor_would(self):
        on = self._render((_cube(),), splats_=(s.SplatInstance(_splat_floor(), relight=1.0),), shadows=True)
        off = self._render((_cube(),), splats_=(s.SplatInstance(_splat_floor(), relight=1.0),), shadows=False)
        diff = np.abs(on[..., :3].astype(int) - off[..., :3].astype(int))
        self.assertGreater(int((diff.max(axis=2) > 20).sum()), 200)

    def test_two_lights_at_right_angles_cast_shadows_in_different_directions(self):
        """Y2 of 2, deliverable 3: two shadow-casting lights at right angles must each get their own
        cell in the atlas and shade independently, so the floor darkens in two different places, one
        per light's own direction, not just one light's shadow winning."""
        light_x = s.Light(kind="Directional", intensity=1.5, shadows=True,
                          position=s.Vec3(6, 5, 0), target=s.Vec3(0, 0, 0))
        light_z = s.Light(kind="Directional", intensity=1.5, shadows=True,
                          position=s.Vec3(0, 5, 6), target=s.Vec3(0, 0, 0))

        def render(x_shadows, z_shadows):
            # Both lights are always present and lit; only whether each one's shadow map is on
            # changes, so every image here shares the same illumination and a diff isolates exactly
            # one light's shadow (or both), never a brightness change from adding/removing a light.
            lights = (replace(light_x, shadows=x_shadows), replace(light_z, shadows=z_shadows))
            return self.gpu.render(s.Scene((_floor(), _cube()), lights), CAMERA, 200, 150, BACKGROUND,
                                   headlight=False, ambient=0.1)

        def diffmask(a, b):
            # A lower threshold than the single-light tests above: with a second, unshadowed light
            # also lighting the scene, one light's own shadow is a smaller share of the pixel.
            diff = np.abs(a[..., :3].astype(int) - b[..., :3].astype(int))
            return diff.max(axis=2) > 15

        neither, only_x, only_z, both = (render(False, False), render(True, False),
                                         render(False, True), render(True, True))
        mask_x, mask_z, mask_both = diffmask(only_x, neither), diffmask(only_z, neither), diffmask(both, neither)
        self.assertGreater(int(mask_x.sum()), 100)
        self.assertGreater(int(mask_z.sum()), 100)
        # The two lights' own shadows fall in mostly different places (right angles, same caster).
        self.assertGreater(int((mask_x & ~mask_z).sum()), 50)
        self.assertGreater(int((mask_z & ~mask_x).sum()), 50)
        # Shadowing both lights at once darkens strictly more of the floor than either alone.
        self.assertGreater(int(mask_both.sum()), max(int(mask_x.sum()), int(mask_z.sum())))

    def test_more_shadow_lights_than_fit_reports_the_brightest_four_on_the_status_line(self):
        lights = [s.Light(kind="Directional", intensity=float(i + 1), shadows=True,
                          position=s.Vec3(i, 5, 0), target=s.Vec3(0, 0, 0))
                 for i in range(viewportgpu.MAX_SHADOW_LIGHTS + 2)]
        self.gpu.render(s.Scene((_floor(), _cube()), tuple(lights)), CAMERA, 200, 150, BACKGROUND,
                        headlight=False, ambient=0.1)
        self.assertIn(str(viewportgpu.MAX_SHADOW_LIGHTS), self.gpu.shadow_note)
        self.assertIn(str(len(lights)), self.gpu.shadow_note)

    def test_default_shadows_off_leaves_the_render_unchanged_from_before_r6_finish(self):
        never_shadowed = self._render((_floor(), _cube()), shadows=False)
        # A light with `shadows` unset (the dataclass default, False) must look identical to one
        # explicitly turned off: the common, pre-existing case is untouched by this feature.
        default = self.gpu.render(s.Scene((_floor(), _cube()),
                                          (s.Light(intensity=1.5, position=s.Vec3(4, 6, 2), target=s.Vec3(0, 0, 0)),)),
                                  CAMERA, 200, 150, BACKGROUND, headlight=False, ambient=0.1)
        np.testing.assert_array_equal(never_shadowed, default)
