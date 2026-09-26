"""Physically based shading of de-lit splats: energy, environment, reflections, visibility, GPU parity, defaults."""
from dataclasses import replace
import unittest

import numpy as np

from nodebased import envlight as E, gpu3d, intrinsics as I, scene3d as s, splats, splatshade as sh
from tests import gpu_precision
from tests.test_3d_environment_light import env_of, sun_map


def _quaternion_about_y(theta):
    return np.array((np.cos(theta / 2), 0, np.sin(theta / 2), 0))


def layer(albedo, roughness, normals):
    n = len(normals)
    return I.Intrinsics(np.broadcast_to(albedo, (n, 3)), np.full(n, roughness), normals, np.ones(n), np.ones(n),
                        np.ones(n), np.zeros(3), np.zeros((0, 3)), np.zeros((0, 3)), 0)


def sphere_cloud(count=1400, radius=1.0, albedo=(1, 1, 1), roughness=0.5):
    """Fibonacci-sphere discs facing outward, with a de-lit layer of the given albedo and roughness."""
    k = np.arange(count) + 0.5
    y = 1 - 2 * k / count
    ring = np.sqrt(1 - y ** 2)
    phi = np.pi * (3 - np.sqrt(5)) * k
    normals = np.stack((ring * np.cos(phi), y, ring * np.sin(phi)), axis=1)
    quats = np.empty((count, 4))
    for i, n in enumerate(normals):                 # rotate local +z onto n
        axis = np.cross((0, 0, 1), n)
        norm = np.linalg.norm(axis)
        angle = np.arctan2(norm, n[2])
        axis = axis / norm if norm > 1e-9 else np.array((0, 1, 0))
        quats[i] = (np.cos(angle / 2), *(np.sin(angle / 2) * axis))
    spacing = radius * np.sqrt(4 * np.pi / count)
    sh_dc = np.zeros((count, 1, 3))
    cloud = splats.SplatCloud(normals * radius, np.tile((spacing * .9, spacing * .9, spacing * .02), (count, 1)),
                              quats, np.full(count, 0.99), sh_dc, 0, colorspace='linear')
    return I.attach(cloud, layer(np.asarray(albedo, float), roughness, normals.astype(np.float32)))


CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 4.5)), s.Vec3(0, 0, 0))


def shade(albedo, roughness, metallic, environment, normal=(0, 0, 1), eye=(0, 0, 5), samples_n=1, **kw):
    n = np.array([normal], float)
    n /= np.linalg.norm(n)
    extras = E.SplatLighting((environment,)) if environment is not None else None
    return sh.shade_splats(np.zeros((1, 3)), np.array([albedo], float), np.zeros((1, 3)), n, np.ones(1), np.array(eye, float),
                           (), 0.0, 1.0, roughness=np.full(1, roughness), occlusion=np.ones(1), metallic=metallic,
                           extras=extras, **kw)[0]


class EnergyTests(unittest.TestCase):
    """White furnace: a white surface under a uniform light of 1 returns 1, whatever its roughness."""
    FURNACE = env_of(np.ones((32, 64, 3), np.float32))

    def test_white_dielectric_and_white_metal_reflect_what_they_receive(self):
        for roughness in (0.05, 0.3, 0.6, 1.0):
            for metallic in (0.0, 0.5, 1.0):
                for normal in ((0, 0, 1), (1, 0, 1), (1, 0, 0.15)):
                    got = shade((1, 1, 1), roughness, metallic, self.FURNACE, normal=normal)
                    np.testing.assert_allclose(got, 1.0, atol=0.005, err_msg=f'{roughness} {metallic} {normal}')

    def test_a_darker_surface_returns_less_and_never_more_than_the_light(self):
        for albedo in ((0.5, 0.5, 0.5), (0.2, 0.6, 0.9)):
            got = shade(albedo, 0.5, 0.0, self.FURNACE)
            self.assertTrue(np.all(got < 1.0 + 0.03))
            self.assertTrue(np.all(got > np.array(albedo) * 0.85))

    def test_direct_light_conserves_energy(self):
        # A directional light of intensity 1 on a white dielectric: diffuse plus specular stays at or under
        # the incoming light (n.l), since the Fresnel share moves from the diffuse to the specular lobe.
        from types import SimpleNamespace
        light = s.Light('Directional', (1, 1, 1), 1.0, s.Vec3(0, 3, 3), s.Vec3(0, 0, 0))
        for roughness in (0.1, 0.5, 1.0):
            n = np.array([[0, 0, 1.0]])
            eye = np.array([0, 0, 5.0])
            got = sh.shade_splats(np.zeros((1, 3)), np.ones((1, 3)), np.zeros((1, 3)), n, np.ones(1), eye, (light,), 0.0,
                                  1.0, roughness=np.full(1, roughness), occlusion=np.ones(1))[0]
            toward = -light.world()[1]
            self.assertLess(float(got[0]), float(toward @ n[0]) * 1.05 + 0.0, roughness)
            self.assertGreater(float(got[0]), 0.4 * float(toward @ n[0]))


class SphereRenderTests(unittest.TestCase):
    def render(self, cloud, environment, size=48, **instance):
        scene = s.Scene(splats=(s.SplatInstance(cloud, relight=1.0, **instance),), environments=(environment,))
        return s.render(scene, CAMERA, size, size)

    def test_white_sphere_under_a_uniform_environment_is_uniform_without_seams(self):
        image = self.render(sphere_cloud(roughness=0.7), env_of(np.ones((32, 64, 3), np.float32)))
        yy, xx = np.mgrid[:48, :48]
        inside = ((xx - 23.5) ** 2 + (yy - 23.5) ** 2) < 14 ** 2       # well inside the silhouette
        rgb = image[..., :3][inside] / np.maximum(image[..., 3][inside], 1e-6)[:, None]
        self.assertGreater(image[..., 3][inside].min(), 0.9)
        self.assertLess(float(rgb.std()), 0.03)
        self.assertAlmostEqual(float(rgb.mean()), 1.0, delta=0.04)

    def test_rotating_the_environment_moves_the_highlight(self):
        cloud = sphere_cloud(albedo=(0.05, 0.05, 0.05), roughness=0.25)

        def peak(rotation):
            image = self.render(cloud, env_of(sun_map(128, 64, sky=0.02, sun=60.0), rotation=rotation))
            luma = image[..., :3].sum(axis=2)
            y, x = np.unravel_index(int(np.argmax(luma)), luma.shape)
            return x, y, float(luma.max())
        right = peak(0)      # the sun is at +X, the highlight sits right of centre
        front = peak(270)    # +X turned to +Z, toward the camera: the highlight is in the middle
        self.assertGreater(right[0], 30)
        self.assertLess(abs(front[0] - 23.5), 5)
        self.assertGreater(right[2], 0.5)
        self.assertGreater(front[2], 0.5)


class ReflectionTests(unittest.TestCase):
    def scene(self, with_mesh, samples=1, metallic=1.0, roughness_scale=0.0, environments=()):
        theta = np.pi / 4                                # the mirror direction of the eye is +X
        n = np.array(((np.sin(theta), 0, np.cos(theta)),), np.float32)
        cloud = splats.SplatCloud(np.zeros((1, 3)), np.array(((0.6, 0.6, 0.01),)), _quaternion_about_y(theta)[None],
                                  np.array((0.99,)), np.zeros((1, 1, 3)), 0, colorspace='linear')
        cloud = I.attach(cloud, layer(np.array((0.9, 0.9, 0.9)), 0.05, n))
        instance = s.SplatInstance(cloud, relight=1.0, metallic=metallic, roughness_scale=roughness_scale,
                                   reflection_samples=samples)
        wall = s._card(4, 4, (1.0, 0.1, 0.05, 1.0), s.Transform3D(s.Vec3(3, 0, 0), s.Vec3(0, 90, 0)))
        return s.Scene((wall,) if with_mesh else (), splats=(instance,), environments=tuple(environments))

    def pixel(self, scene):
        image = s.render(scene, CAMERA, 32, 32, ambient=1.0)
        return image[16, 16, :3] / max(float(image[16, 16, 3]), 1e-6)

    def test_a_smooth_metal_splat_reflects_a_mesh_beside_it(self):
        lit = self.pixel(self.scene(True))
        dark = self.pixel(self.scene(False))
        self.assertGreater(lit[0], 0.5)
        self.assertGreater(lit[0], 4 * lit[1])          # the wall is red
        self.assertLess(float(dark.max()), 0.05)        # no wall, no environment: nothing to reflect
        # A roughness lobe of several rays averages the wall with its surroundings but still sees it.
        rough = self.pixel(self.scene(True, samples=16, roughness_scale=4.0))
        self.assertGreater(rough[0], 0.05)

    def test_a_reflection_miss_reads_the_environment(self):
        sky = env_of(np.full((16, 32, 3), 0.25, np.float32))
        scene = self.scene(False, environments=(sky,))
        np.testing.assert_allclose(self.pixel(scene)[0], 0.25 * 0.9, atol=0.05)   # albedo-tinted F0 of the metal

    def test_reflections_are_deterministic(self):
        scene = self.scene(True, samples=8, roughness_scale=2.0)
        a, b = s.render(scene, CAMERA, 24, 24, ambient=1.0), s.render(scene, CAMERA, 24, 24, ambient=1.0)
        self.assertEqual(a.tobytes(), b.tobytes())


class VisibilityTests(unittest.TestCase):
    def test_a_mesh_shadow_and_a_splat_occluder_darken_the_delit_shading(self):
        # A floor of splats facing up, lit from straight above; a splat occluder hovers over its middle.
        x, z = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
        floor = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        occluder = np.array(((0.0, 0.5, 0.0),))
        positions = np.vstack((floor, occluder))
        n = len(positions)
        up = np.tile((0, 1.0, 0), (n, 1)).astype(np.float32)
        quats = np.tile(_quaternion_about_y(0)[None], (n, 1))
        quats[:] = (np.sqrt(.5), -np.sqrt(.5), 0, 0)      # local +z onto +y
        scales = np.tile((0.12, 0.12, 0.005), (n, 1))
        scales[-1] = (0.35, 0.35, 0.005)
        opacity = np.full(n, 0.99)
        cloud = splats.SplatCloud(positions, scales, quats, opacity, np.zeros((n, 1, 3)), 0, colorspace='linear')
        cloud = I.attach(cloud, layer(np.array((0.8, 0.8, 0.8)), 0.6, up))
        light = s.Light('Directional', (1, 1, 1), 1.0, s.Vec3(0, 5, 0), s.Vec3(0, 0, 0), shadows=True)
        instance = s.SplatInstance(cloud, relight=1.0)
        eye = np.array((0, 3.0, 4.0))
        context = s._SplatShadows((instance,), (light,))
        visibility = context.for_indices(0, np.arange(n))
        colours = sh.instance_colors(instance, eye, (light,), 0.0, visibility)
        centre = int(np.argmin(np.linalg.norm(floor, axis=1)))
        corner = 0
        self.assertLess(visibility[centre, 0], 0.2)
        self.assertGreater(visibility[corner, 0], 0.9)
        self.assertLess(float(colours[centre].mean()), 0.25 * float(colours[corner].mean()))
        # Without the traced visibility the two floor splats shade alike.
        flat = sh.instance_colors(instance, eye, (light,), 0.0)
        self.assertAlmostEqual(float(flat[centre].mean()), float(flat[corner].mean()), delta=0.05)

    def test_a_mesh_casts_onto_delit_splats_in_a_render(self):
        cloud = sphere_cloud(count=900, roughness=0.6)
        light = s.Light('Directional', (1, 1, 1), 1.0, s.Vec3(0, 6, 0), s.Vec3(0, 0, 0), shadows=True)
        lid = s._card(4, 4, (0.5, 0.5, 0.5, 1.0), s.Transform3D(s.Vec3(0, 1.6, 0), s.Vec3(-90, 0, 0)))
        base = s.Scene(splats=(s.SplatInstance(cloud, relight=1.0),), lights=(light,))
        shadowed = replace(base, geometries=(lid,))
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0.0, 4.5)), s.Vec3(0, 0, 0))
        open_sky = s.render(base, camera, 32, 32)
        under = s.render(shadowed, camera, 32, 32)
        top = (slice(6, 11), slice(13, 19))
        self.assertGreater(float(open_sky[top][..., :3].mean()), 3 * float(under[top][..., :3].mean()) + 0.01)


class BlendAndDefaultTests(unittest.TestCase):
    def setUp(self):
        self.cloud = sphere_cloud(count=500, roughness=0.4, albedo=(0.7, 0.4, 0.3))
        self.light = s.Light('Directional', (1, 1, 1), 1.0, s.Vec3(2, 4, 3), s.Vec3(0, 0, 0))
        self.eye = np.array((0, 0, 4.5))

    def colours(self, **kw):
        return sh.instance_colors(s.SplatInstance(self.cloud, relight=1.0, **kw), self.eye, (self.light,), 0.05)

    def test_defaults_are_the_dielectric_full_intrinsics_shading(self):
        d = s.SplatInstance(self.cloud)
        self.assertEqual((d.metallic, d.roughness_scale, d.intrinsics_mix, d.reflection_samples), (0.0, 1.0, 1.0, 0))
        np.testing.assert_array_equal(self.colours(), self.colours(intrinsics_mix=1.0, metallic=0.0))

    def test_the_captured_blend_ends_at_the_captured_shading(self):
        captured = self.colours(use_intrinsics=False)
        np.testing.assert_array_equal(self.colours(intrinsics_mix=0.0), captured)
        full, half = self.colours(intrinsics_mix=1.0), self.colours(intrinsics_mix=0.5)
        np.testing.assert_allclose(half, 0.5 * full + 0.5 * captured, atol=1e-6)
        self.assertGreater(float(np.abs(full - captured).max()), 0.05)

    def test_metallic_and_roughness_change_the_shading(self):
        base = self.colours()
        self.assertGreater(float(np.abs(self.colours(metallic=1.0) - base).max()), 0.05)
        self.assertGreater(float(np.abs(self.colours(roughness_scale=0.2) - base).max()), 0.02)

    def test_an_instance_without_a_layer_ignores_the_new_knobs(self):
        plain = replace(self.cloud, intrinsics=None)
        a = sh.instance_colors(s.SplatInstance(plain, relight=1.0), self.eye, (self.light,), 0.05)
        b = sh.instance_colors(s.SplatInstance(plain, relight=1.0, metallic=1.0, roughness_scale=0.0,
                                               intrinsics_mix=0.2, reflection_samples=4), self.eye, (self.light,), 0.05)
        np.testing.assert_array_equal(a, b)


class BundleTests(unittest.TestCase):
    def scene(self):
        cloud = sphere_cloud(count=800, roughness=0.4, albedo=(0.7, 0.5, 0.4))
        sun = s.Light('Directional', (1, 0.9, 0.8), 1.2, s.Vec3(2, 3, 4), s.Vec3(0, 0, 0), shadows=True)
        fill = s.Light('Point', (0.3, 0.4, 0.8), 0.8, s.Vec3(-2, 1, 3), s.Vec3(0, 0, 0))
        return s.Scene(splats=(s.SplatInstance(cloud, relight=1.0, metallic=0.2),), lights=(sun, fill),
                       environments=(env_of(sun_map(64, 32, sky=0.15, sun=6.0), rotation=20),))

    def test_the_bundle_carries_the_environment_layers_and_sums_to_the_beauty(self):
        scene = self.scene()
        beauty = s.render(scene, CAMERA, 40, 40, ambient=0.05)
        out, layers = s.render(scene, CAMERA, 40, 40, ambient=0.05, output='relight')
        np.testing.assert_allclose(out, beauty, atol=2e-4)
        for name in ('environment_diffuse', 'environment_specular', 'reflections', 'visibility',
                     'diffuse_L0', 'specular_L1'):
            self.assertIn(name, layers)
        covered = layers['albedo'][..., 3] > 0.9
        self.assertGreater(float(layers['environment_diffuse'][..., :3][covered].mean()), 0.05)
        self.assertGreater(float(layers['environment_specular'][..., :3][covered].max()), 0.05)
        self.assertLess(float(np.abs(layers['reflections'][..., :3]).max()), 1e-6)   # no meshes to reflect
        visible = layers['visibility'][..., 0][covered]
        self.assertLess(float(visible.min()), 0.5)          # the sun's shadow falls on the sphere itself
        self.assertGreater(float(visible.max()), 0.99)

    def test_shadowed_light_responses_carry_the_traced_visibility(self):
        scene = self.scene()
        _, layers = s.render(scene, CAMERA, 40, 40, ambient=0.05, output='relight')
        unshadowed = replace(scene, lights=(replace(scene.lights[0], shadows=False), scene.lights[1]))
        _, open_layers = s.render(unshadowed, CAMERA, 40, 40, ambient=0.05, output='relight')
        self.assertLess(float(layers['diffuse_L0'][..., :3].sum()), 0.995 * float(open_layers['diffuse_L0'][..., :3].sum()))
        np.testing.assert_allclose(layers['diffuse_L1'], open_layers['diffuse_L1'], atol=1e-6)

    def test_the_relight_node_rebalances_the_environment(self):
        import tempfile
        from pathlib import Path
        from nodebased.core import Dispatcher
        from nodebased.imaging import Evaluator
        cloud = replace(sphere_cloud(count=600, albedo=(0.6, 0.6, 0.6)), intrinsics=None)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'ball.ply')
            splats.write_ply(cloud, path)
            d, e = Dispatcher(), Evaluator()
            for key, kind, params in (
                    ('read', 'ReadSplat3D', dict(splat_path=path, splat_colorspace='linear', splat_relight=1.0)),
                    ('sky', 'Constant', dict(width=64, height=32, red=0.5, green=0.5, blue=0.5)),
                    ('env', 'Light3D', dict(light_type='Environment')),
                    ('scene', 'Scene3D', {}), ('camera', 'Camera3D', dict(tz=4.5)),
                    ('render', 'Render3D', dict(width=24, height=24, samples=1, render_output='relight', ambient=0.0)),
                    ('relight', 'Relight', dict(red=0.0, green=0.0, blue=0.0))):
                d.execute(dict(op='create', id=key, type=kind, params=params))
            for target, slot, source in (('env', 'image', 'sky'), ('scene', 'object0', 'read'),
                                         ('scene', 'object1', 'env'), ('render', 'scene', 'scene'),
                                         ('render', 'camera', 'camera'), ('relight', 'image', 'render')):
                d.execute(dict(op='connect', id=target, input=slot, source=source))
            full = e.evaluate_raster(d.document, 'relight').pixels
            d.execute(dict(op='set', id='relight', param='environment', value=0.0))
            none = e.evaluate_raster(d.document, 'relight').pixels
            d.execute(dict(op='set', id='relight', param='environment', value=0.5))
            half = e.evaluate_raster(d.document, 'relight').pixels
        centre = (12, 12)
        self.assertGreater(float(full[centre][0]), 0.2)      # albedo 0.6 under a sky of 0.5, near 0.3
        self.assertAlmostEqual(float(none[centre][0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(half[centre][0]), 0.5 * float(full[centre][0]), delta=1e-4)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GpuParityTests(unittest.TestCase):
    def test_environment_and_light_shading_matches_the_cpu_reference(self):
        cloud = sphere_cloud(count=700, roughness=0.35, albedo=(0.6, 0.5, 0.4))
        light = s.Light('Directional', (1, 0.9, 0.8), 1.5, s.Vec3(2, 3, 4), s.Vec3(0, 0, 0))
        scene = s.Scene(splats=(s.SplatInstance(cloud, relight=1.0, metallic=0.3),), lights=(light,),
                        environments=(env_of(sun_map(64, 32, sky=0.2, sun=8.0), rotation=40),))
        cpu = s.render(scene, CAMERA, 40, 40, ambient=0.05)
        gpu = gpu3d.render(scene, CAMERA, 40, 40, ambient=0.05)
        difference = np.abs(cpu - gpu)
        self.assertLess(float(difference.max()), gpu_precision.tolerance(3e-3))
        self.assertGreater(float(cpu[..., :3].max()), 0.3)

    def test_meshes_with_an_environment_fall_back_to_the_cpu(self):
        scene = s.Scene((s._card(1, 1, (0.5, 0.5, 0.5, 1.0), s.Transform3D()),),
                        environments=(env_of(np.ones((8, 16, 3), np.float32)),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, CAMERA, 16, 16)


if __name__ == '__main__':
    unittest.main()
