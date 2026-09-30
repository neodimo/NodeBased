"""CPU-reference comparisons; optional wgpu is never required for this suite."""
from dataclasses import replace
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import envlight as E, gpu3d, scene3d as s


def card(color=(.2, .6, .9, 1), position=s.Vec3(), texture=None):
    return s._card(2.7, 2.1, color, s.Transform3D(position), texture)


def gradient(alpha=False, shape=(32, 64)):
    y, x = np.mgrid[:shape[0], :shape[1]]
    tex = np.stack((x/(shape[1]-1), y/(shape[0]-1), .2+.5*x/(shape[1]-1), np.ones_like(x)), -1).astype('f4')
    if alpha:
        tex[..., 3] = np.where(x < shape[1]//3, 0, np.where(x < shape[1]*2//3, .4, 1))
        tex[..., :3] *= tex[..., 3:4]
    return tex


def sun_map(width=128, height=64, sky=0.1, sun=20.0):
    d, _ = E.direction_grid(width, height)
    rgb = np.full((height, width, 3), sky, np.float32)
    rgb[d[..., 0] > 0.95] = sun                     # a sun toward +X
    return rgb


def env_of(rgb, **kw):
    return E.Environment(rgb, E.fingerprint_of(rgb), **kw)


def smooth_map(width=128, height=64):
    """A low-frequency environment (no hard edges): the atlas's resampled level 0 (`_environment_resources`
    resamples the CPU reference's full-resolution level 0 down to `envlight.PREFILTER_SIZE`) then agrees
    with the CPU reference much more tightly than a hard-edged map like `sun_map`, whose sharp disc is
    resampled to a different edge position at each resolution -- a real mismatch between the two
    resamplings, not a shading bug, and not representative of a real (photographed) HDRI."""
    d, _ = E.direction_grid(width, height)
    return (np.clip(2.0 + 3.0 * d[..., 0], 0.2, None)[..., None] * np.array([1.0, 0.9, 0.7], np.float32))


class NoGPURequired(unittest.TestCase):
    def test_flat_attributes_agree_at_every_triangle_vertex(self):
        ground = s.Geometry(np.array([[-8, -1, 6], [8, -1, 6], [8, -1, -12], [-8, -1, -12]], 'f4'),
                            np.array([[0, 1, 2], [0, 2, 3]], 'i4'), (.3, .6, .2, 1))
        _, _, vertices, _, _ = gpu3d._prepare(s.Scene((ground,)), s.Camera(), 64, 48, None)
        self.assertTrue(vertices)
        for triangle in vertices:
            np.testing.assert_array_equal(triangle[:, 11:16], np.broadcast_to(triangle[0, 11:16], (3, 5)))

    def test_missing_dependency_is_cached_and_clear(self):
        with patch.dict(sys.modules, {'wgpu': None}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
            self.assertFalse(gpu3d.available())
            self.assertFalse(gpu3d.available())
            self.assertIn('wgpu unavailable', gpu3d.describe())
            with self.assertRaisesRegex(RuntimeError, 'wgpu unavailable'):
                gpu3d.render(s.Scene((card(),)), s.Camera(), 64, 48)

    def test_missing_adapter_and_device(self):
        for failure in (None, RuntimeError('device unavailable')):
            from unittest.mock import Mock
            module = Mock()
            if failure is None:
                module.gpu.request_adapter_sync.return_value = None
            else:
                module.gpu.request_adapter_sync.return_value.features = set()
                module.gpu.request_adapter_sync.return_value.request_device_sync.side_effect = failure
            with patch.dict(sys.modules, {'wgpu': module}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
                self.assertFalse(gpu3d.available())
                self.assertIn('unavailable', gpu3d.describe())
                self.assertFalse(gpu3d.available())
                self.assertEqual(module.gpu.request_adapter_sync.call_count, 1)

    def test_downlevel_adapter_without_float32_targets_is_unavailable(self):
        from unittest.mock import Mock
        module = Mock()
        adapter = module.gpu.request_adapter_sync.return_value
        adapter.features = set()
        adapter.limits = {'max-storage-buffer-binding-size': 1 << 27}
        adapter.request_device_sync.return_value.create_texture.side_effect = RuntimeError('downlevel restrictions')
        with patch.dict(sys.modules, {'wgpu': module}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
            self.assertFalse(gpu3d.available())
            self.assertIn('cannot render to rgba32float', gpu3d.describe())
            self.assertFalse(gpu3d.available())
            self.assertEqual(module.gpu.request_adapter_sync.call_count, 1)

    def test_adapter_with_too_few_storage_buffers_is_unavailable(self):
        from unittest.mock import Mock
        for limit in (0, 1):
            module = Mock()
            adapter = module.gpu.request_adapter_sync.return_value
            adapter.features = set()
            adapter.limits = {'max-storage-buffer-binding-size': 1 << 27,
                              'max-storage-buffers-per-shader-stage': limit}
            with patch.dict(sys.modules, {'wgpu': module}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
                self.assertFalse(gpu3d.available())
                reason = 'adapter allows fewer than 2 storage buffers per shader stage'
                self.assertIn(reason, gpu3d.describe())
                with self.assertRaisesRegex(RuntimeError, reason):
                    gpu3d._state()
                self.assertFalse(gpu3d.available())
                self.assertEqual(module.gpu.request_adapter_sync.call_count, 1)
                adapter.request_device_sync.assert_not_called()

    def test_unsupported_without_gpu(self):
        projected = replace(card(), projection=s.Projection(s.Camera(), gradient()))
        with self.assertRaisesRegex(gpu3d.Unsupported, 'projected'):
            gpu3d.render(s.Scene((projected,)), s.Camera(), 64, 48)
        with patch.object(s, 'MAX_TRIANGLES', 1):
            with self.assertRaises(gpu3d.Unsupported):
                gpu3d.render(s.Scene((card(),)), s.Camera(), 64, 48)
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene(), s.Camera(), 64, 48, output='shade')

    def test_cancel_before_device(self):
        from nodebased.imaging import Cancelled
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            gpu3d.render(s.Scene((card(),)), s.Camera(), 64, 48, cancel=event)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUComparison(unittest.TestCase):
    def compare(self, scene, camera=None, **kwargs):
        camera = camera or s.Camera()
        expected = s.render(scene, camera, 64, 48, **kwargs)
        actual = gpu3d.render(scene, camera, 64, 48, **kwargs)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (48, 64, 4))
        self.assertFalse(actual.flags.writeable)
        np.testing.assert_array_equal(actual, gpu3d.render(scene, camera, 64, 48, **kwargs))
        a, b = actual[..., 3] > 0, expected[..., 3] > 0
        union = (a | b).sum()
        self.assertGreater((a & b).sum()/max(union, 1), .99)
        mask = expected[..., 3] == 1
        # Erode by two pixels, with no scipy dependency or wraparound at borders.
        padded = np.pad(mask, 2)
        interior = np.logical_and.reduce([padded[y:y+48, x:x+64] for y in range(5) for x in range(5)])
        self.assertTrue(interior.any(), 'comparison must have opaque interior pixels')
        np.testing.assert_array_equal(actual[interior, 3], 1.0)
        delta = np.abs(actual[interior]-expected[interior])
        if kwargs.get('output') == 'depth':
            self.assertLess(float((delta[:, :3]/np.maximum(expected[interior, :3], 1e-8)).mean()), 1e-3)
        else:
            self.assertLess(float(delta.mean()), 5e-3)
        return actual, expected

    def test_half_precision_fallback(self):
        state = gpu3d._state()
        # Exercise the portable target even on float32-blendable adapters.
        with patch.dict(state, {'format': 'rgba16float', 'pipelines': {}}):
            self.compare(s.Scene((card(),)), background=(.1, .2, .3, 1))
            self.compare(s.Scene((card((.7, .2, .4, .3), s.Vec3(0, 0, 1)), card())))
            self.compare(s.Scene((card(),)), output='depth')

    def test_transformed_normals_and_camera(self):
        sphere = s._sphere(1.1, 32, (.3, .6, .7, 1),
                           s.Transform3D(s.Vec3(), s.Vec3(15, 30, 4), s.Vec3(1.2, .8, 1)))
        camera = s.Camera(s.Transform3D(s.Vec3(1, 1, 5)), roll=13)
        self.compare(s.Scene((sphere,), (s.Light(),)), camera, ambient=.2)
        self.compare(s.Scene((sphere,)), camera, output='normals')
        backface = replace(card(), triangles=np.array([[2, 1, 0], [3, 2, 0]], 'i4'))
        self.compare(s.Scene((backface,), (s.Light(),)), ambient=.2)

    def test_flat_card_cube(self):
        cube = s._cube(1.3, (.8, .3, .1, 1), s.Transform3D(s.Vec3(.65, .1, .6), s.Vec3(10, 23, 7)))
        self.compare(s.Scene((card(position=s.Vec3(-.4, 0, -.5)), cube)))

    def test_textures_and_mips(self):
        for shape in ((32, 64), (129, 257), (1, 64)):
            with self.subTest(shape=shape):
                tex = gradient(shape=shape) if shape[0] > 1 else np.ones((1, 64, 4), 'f4')
                self.compare(s.Scene((card(texture=tex),)))

    def test_lighting(self):
        sphere = s._sphere(1.3, 32, (.7, .4, .2, 1), s.Transform3D())
        lights = (s.Light(intensity=.7), s.Light('Point', (.3, .8, 1), 1.1, s.Vec3(-2, 1, 3)))
        self.compare(s.Scene((sphere,), lights), ambient=.17)
        # Ambient alone must not light an otherwise unlit scene.
        self.compare(s.Scene((sphere,), (s.Light(intensity=0),)), ambient=.9)

    def test_transparency(self):
        scene = s.Scene((card((.8, .1, .3, .35), s.Vec3(.4, .2, 1)),
                         card((.1, .8, .2, .55), s.Vec3(-.3, 0, .5)),
                         card((.2, .3, .8, 1), s.Vec3(0, 0, -.6))))
        a, b = self.compare(scene)
        # Include partially covered transparent regions, not just opaque interiors.
        common = (a[..., 3] > 0) & (b[..., 3] > 0)
        self.assertLess(float(np.abs(a[common]-b[common]).mean()), 5e-3)
        self.compare(s.Scene((card(texture=gradient(True), position=s.Vec3(0, 0, 1)), card())))

    def test_ground_crosses_near(self):
        ground = s.Geometry(np.array([[-8, -1, 6], [8, -1, 6], [8, -1, -12], [-8, -1, -12]], 'f4'),
                            np.array([[0, 1, 2], [0, 2, 3]], 'i4'), (.3, .6, .2, 1))
        a, _ = self.compare(s.Scene((ground,)))
        self.assertGreater(a[-1, :, 3].sum(), 60)

    def test_data_outputs(self):
        scene = s.Scene((s._sphere(1.3, 32, (.7, .4, .2, .2), s.Transform3D()),
                         card(position=s.Vec3(0, 0, -1))))
        for output in ('depth', 'normals'):
            with self.subTest(output=output):
                self.compare(scene, output=output, samples=4, background=(1, 0, 1, 1))
                self.compare(s.Scene((card(texture=gradient(True)),)), output=output)

    def test_supersampling_and_background(self):
        self.compare(s.Scene((card(),)), samples=2)
        self.compare(s.Scene((card((.3, .7, .2, .4)),)), samples=2, background=(.1, .3, .6, 1))
        for output in ('rgba', 'depth', 'normals'):
            a = gpu3d.render(s.Scene(), s.Camera(), 64, 48, background=(.8, .4, .2, .5), output=output)
            np.testing.assert_array_equal(a, s.render(s.Scene(), s.Camera(), 64, 48, background=(.8, .4, .2, .5), output=output))
            self.assertFalse(a.flags.writeable)

    def test_environment_lights_a_mesh(self):
        # Y1 of 2 finish (2): a scene with a single Environment and no analytic lights used to be
        # CPU-only outright; now the raster shader's `env_diffuse` (an SH lookup) agrees with the
        # CPU reference's `environment.diffuse`.
        sphere = s._sphere(1.1, 32, (.6, .6, .6, 1), s.Transform3D())
        self.compare(s.Scene((sphere,), environments=(env_of(sun_map()),)), ambient=.05)
        # A uniform environment leaves the authored colour alone on both backends alike.
        flat = s.Scene((card(),), environments=(env_of(np.full((16, 32, 3), .4, np.float32)),))
        self.compare(flat)

    def test_environment_specular_reflects_off_a_shiny_mesh(self):
        # `_mesh_environment_specular`'s Blinn-Phong-shininess-to-GGX-roughness mapping (`env_specular`
        # in `_SHADER`), on top of an analytic light so both the light and environment specular terms
        # are exercised together.
        shiny = replace(s._sphere(1.0, 32, (.5, .5, .5, 1), s.Transform3D()), specular=.8, shininess=60.0)
        scene = s.Scene((shiny,), (s.Light(intensity=.6),), environments=(env_of(smooth_map()),))
        self.compare(scene, ambient=.1)

    def test_environment_rotation_and_parent_transform_match_the_cpu_reference(self):
        # `env_local`'s combined rotation matrix (the Environment's own turn and its parent's) must
        # agree with `envlight.Environment._local`, not just the no-parent case `viewportgpu.py`'s
        # dome assumes.
        sphere = s._sphere(1.0, 24, (.6, .6, .6, 1), s.Transform3D())
        rotated = env_of(sun_map(), rotation=115.0)
        self.compare(s.Scene((sphere,), environments=(rotated,)), ambient=.05)
        parent = np.asarray(s.Transform3D(rotation=s.Vec3(0, 40, 0)).matrix())
        parented = replace(rotated, parent=parent)
        self.compare(s.Scene((sphere,), environments=(parented,)), ambient=.05)

    def test_environment_visible_to_camera_replaces_the_background(self):
        # Y1 of 2 finish (2), the "background image" part: `scene3d._visible_background` reused
        # directly on the GPU readback.
        sphere = s._sphere(.6, 24, (.6, .6, .6, 1), s.Transform3D(s.Vec3(-1.4, 0, 0)))
        visible = env_of(sun_map(), visible_to_camera=True)
        self.compare(s.Scene((sphere,), environments=(visible,)), ambient=.05, background=(0, 0, 0, 0))

    def test_data_outputs_are_unaffected_by_the_environment(self):
        sphere = s._sphere(1.1, 32, (.6, .6, .6, 1), s.Transform3D())
        scene = s.Scene((sphere,), environments=(env_of(sun_map()),))
        for output in ('depth', 'normals', 'albedo', 'emission'):
            with self.subTest(output=output):
                self.compare(scene, output=output)

    def test_pbr_material_matches_the_cpu_reference(self):
        # Y3 of 3, part 1: `pbr` metallic/roughness/dielectric-specular factors now shade on the GPU
        # raster path (Cook-Torrance GGX, `ggx_response` in `_SHADER`), lit by Directional/Point/Spot
        # lights the same way `scene3d._shade_pbr_mesh` does. No Environment and no texture maps yet
        # (see `EnvironmentRefusalBoundaries` below for what still falls back to the CPU reference).
        sphere = replace(s._sphere(1.1, 32, (.7, .3, .2, 1), s.Transform3D()),
                         material='pbr', metallic=.15, pbr_roughness=.35)
        lights = (s.Light(intensity=.8), s.Light('Point', (.5, .7, 1), 1.0, s.Vec3(-2, 1.5, 2)))
        self.compare(s.Scene((sphere,), lights), ambient=.12)
        self.compare(s.Scene((sphere,), lights), ambient=.12, output='specular')
        metal = replace(sphere, metallic=.9, pbr_roughness=.2, color=(.8, .7, .3, 1))
        self.compare(s.Scene((metal,), lights), ambient=.05)
        # A low dielectric-specular knob (`pbr_specular`) changes the highlight, not just its default.
        dim = replace(sphere, pbr_specular=.1)
        self.compare(s.Scene((dim,), lights), ambient=.12, output='specular')

    def test_pbr_material_data_outputs_are_unaffected(self):
        sphere = replace(s._sphere(1.1, 32, (.7, .3, .2, 1), s.Transform3D()), material='pbr', metallic=.4)
        scene = s.Scene((sphere,), (s.Light(intensity=.8),))
        for output in ('depth', 'normals', 'albedo', 'emission'):
            with self.subTest(output=output):
                self.compare(scene, output=output)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class EnvironmentRefusalBoundaries(unittest.TestCase):
    """What the raster path still leaves to the CPU reference (docs/3D_FOUNDATION.md "Left out"):
    more than one Environment, the ray-traced render mode, a `pbr` material together with an
    Environment or with any of its five texture maps, `pbr` on the ray-traced mode at all, and
    Rect/Disc/Sphere area lights. Each must raise `gpu3d.Unsupported` rather than silently drawing an
    unlit or wrong picture. Plain `pbr` metallic/roughness factors (no maps, no Environment) shade on
    the GPU raster path now (Y3 of 3, part 1); see `GPUComparison.test_pbr_material_matches_the_cpu_reference`."""

    def test_more_than_one_environment_is_cpu_only(self):
        scene = s.Scene((card(),), environments=(env_of(sun_map()), env_of(sun_map(), rotation=1)))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, s.Camera(), 16, 16)

    def test_raytrace_mode_with_an_environment_is_still_cpu_only(self):
        scene = s.Scene((card(),), environments=(env_of(sun_map()),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, s.Camera(), 16, 16, mode='raytrace')

    def test_pbr_material_with_an_environment_is_still_cpu_only(self):
        pbr_card = replace(card(), material='pbr', metallic=.4)
        scene = s.Scene((pbr_card,), environments=(env_of(sun_map()),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, s.Camera(), 16, 16)

    def test_pbr_material_on_the_raytrace_mode_is_still_cpu_only(self):
        pbr_card = replace(card(), material='pbr')
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene((pbr_card,)), s.Camera(), 16, 16, mode='raytrace')

    def test_pbr_texture_maps_are_still_cpu_only(self):
        for field in ('metallic_roughness_texture', 'normal_texture', 'occlusion_texture', 'emissive_texture'):
            with self.subTest(field=field):
                pbr_card = replace(card(), material='pbr', **{field: gradient()})
                with self.assertRaises(gpu3d.Unsupported):
                    gpu3d.render(s.Scene((pbr_card,)), s.Camera(), 16, 16)
        emissive = replace(card(), material='pbr', emissive_color=(.2, .1, 0))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene((emissive,)), s.Camera(), 16, 16)

    def test_area_light_is_still_cpu_only(self):
        light = s.Light('Rect', (1, 1, 1), 1.0, s.Vec3(0, 2, 0), s.Vec3(0, 0, 0))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene((card(),), (light,)), s.Camera(), 16, 16)

    def test_pbr_material_with_an_area_light_is_still_cpu_only(self):
        pbr_card = replace(card(), material='pbr', metallic=.4)
        light = s.Light('Rect', (1, 1, 1), 1.0, s.Vec3(0, 2, 0), s.Vec3(0, 0, 0))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene((pbr_card,), (light,)), s.Camera(), 16, 16)


if __name__ == '__main__':
    unittest.main()
