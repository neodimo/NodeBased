"""Z3 of 3: the ray-traced mode's packed material table and texture atlas (gpurt_render.py).

Host tests check the packing without an adapter: every record the compute shader reads lives in the
`table` or the `texels` atlas, so the shader still needs only eight storage bindings. GPU tests compare
the ray-traced mode with the CPU reference on pbr meshes (all five maps, flat emissive), the Environment's
diffuse and specular and Rect/Disc/Sphere area lights with shadows, and on documents that use none of it.
"""
import re
import unittest
from dataclasses import replace

import numpy as np

from nodebased import envlight as E, gpu3d, gpurt_render as r, scene3d as s
from tests.test_3d_gpu import card, env_of, gradient, smooth_map, sun_map


def textured_pbr_sphere():
    mr = np.zeros((2, 2, 4), np.float32)
    mr[..., 1], mr[..., 2], mr[..., 3] = .3, .6, 1
    return replace(s._sphere(1.0, 16, (.7, .4, .3, 1), s.Transform3D()), material='pbr', texture=gradient(),
                   metallic=.2, pbr_roughness=.5, pbr_specular=.7, metallic_roughness_texture=mr,
                   normal_texture=np.full((4, 4, 4), (.6, .5, .8, 1.), np.float32), normal_scale=.5,
                   occlusion_texture=np.full((2, 2, 4), (.4, .4, .4, 1.), np.float32), occlusion_strength=.8,
                   emissive_texture=np.full((2, 4, 4), (.5, .25, .75, 1.), np.float32), emissive_color=(.3, .2, .1))


class PackingTests(unittest.TestCase):
    def prepare(self, scene):
        return r._prepare(scene, s.Camera(), 64, 48)

    def test_shader_still_needs_only_eight_storage_bindings(self):
        self.assertEqual(len(re.findall(r'var<storage', r._SHADER)), 8)
        self.assertEqual(len(re.findall(r'var<uniform', r._SHADER)), 1)
        self.assertEqual(r.PARAM_WORDS*4, 64)
        self.assertIn('areas: u32, area_offset: u32, env_offset: u32', r._SHADER)

    def test_material_table_holds_the_factors_and_map_descriptors(self):
        sphere = textured_pbr_sphere()
        prepared = self.prepare(s.Scene((sphere,)))
        table, texels = prepared[3], prepared[4]
        np.testing.assert_allclose(table[2], (.2, .5, .08*.7, 1))
        np.testing.assert_allclose(table[3, :2], (.5, .8))
        np.testing.assert_allclose(table[4, :3], (.3, .2, .1))
        maps = (sphere.metallic_roughness_texture, sphere.normal_texture, sphere.occlusion_texture, sphere.emissive_texture)
        for slot, texture in enumerate(maps):
            offset, w, h, present = (int(v) for v in table[5+slot])
            self.assertEqual((w, h, present), (texture.shape[1], texture.shape[0], 1))
            np.testing.assert_array_equal(texels[offset:offset+w*h].reshape(h, w, 4), texture)
        # The base-colour mip chain follows the header, one descriptor per level.
        mips = s._mip_chain(gradient())
        for level, mip in enumerate(mips):
            offset, w, h, _ = (int(v) for v in table[r.MATERIAL_HEADER+level])
            np.testing.assert_array_equal(texels[offset:offset+w*h].reshape(h, w, 4), mip)

    def test_a_plain_material_has_no_pbr_flag_and_absent_maps(self):
        table = self.prepare(s.Scene((card(),)))[3]
        self.assertEqual(table[2, 3], 0)
        self.assertTrue(np.all(table[5:9, 3] == 0))
        np.testing.assert_array_equal(table[4], 0)

    def test_the_flat_uv_tangent_rides_in_the_spare_attribute_components(self):
        sphere = textured_pbr_sphere()
        attrs = self.prepare(s.Scene((sphere,)))[2]
        matrix = sphere.world_matrix()
        world = (matrix[:3, :3] @ sphere.vertices.T + matrix[:3, 3:4]).T
        tri = sphere.triangles[5]
        e1, e2 = world[tri[1]]-world[tri[0]], world[tri[2]]-world[tri[0]]
        duv1, duv2 = sphere.uvs[tri[1]]-sphere.uvs[tri[0]], sphere.uvs[tri[2]]-sphere.uvs[tri[0]]
        det = duv1[0]*duv2[1]-duv2[0]*duv1[1]
        expected = (e1*duv2[1]-e2*duv1[1])/det if abs(det) > 1e-12 else e1
        got = np.array((attrs[5, 1, 3], attrs[5, 2, 3], attrs[5, 6, 3]))
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)

    def test_environment_block_matches_the_prefiltered_levels(self):
        env = env_of(sun_map(), rotation=40., blur=.2, intensity=2., tint=(1., .9, .8))
        prepared = self.prepare(s.Scene((card(),), environments=(env,)))
        table, texels, env_offset = prepared[3], prepared[4], prepared[-1]
        self.assertGreater(env_offset, 0)
        pre = env._pre()
        block = table[env_offset:env_offset+19]
        np.testing.assert_allclose(block[:9, :3], pre.sh, rtol=1e-6)
        np.testing.assert_allclose(block[12, :3], 2.*np.array((1., .9, .8)), rtol=1e-6)
        self.assertAlmostEqual(float(block[12, 3]), .2, places=6)
        for index, level in enumerate(pre.levels):
            offset, w, h, _ = (int(v) for v in block[13+index])
            self.assertEqual((h, w), level.shape[:2])
            np.testing.assert_array_equal(texels[offset:offset+w*h, :3].reshape(h, w, 3), level)
        # The matrix columns undo the environment's turn: `Environment._local(d)` is matrix @ d.
        d = np.array((.3, .5, -.8)); d /= np.linalg.norm(d)
        matrix = block[9:12, :3].T.astype('f8')
        np.testing.assert_allclose(matrix @ d, env._local(d[None])[0], atol=1e-6)

    def test_no_environment_and_no_area_lights_cost_nothing(self):
        prepared = self.prepare(s.Scene((card(),), (s.Light(),)))
        self.assertEqual(prepared[-3:], (0, 0, 0))

    def test_area_light_records_reference_the_cpu_sample_points(self):
        rect = s.Light('Rect', (1, .9, .8), 2., s.Vec3(0, 2, 0), s.Vec3(0, 0, 0), light_samples=12, shadows=True)
        sphere_light = s.Light('Sphere', (1, 1, 1), 1., s.Vec3(2, 1, 1), area_radius=.4, light_samples=5, two_sided=True)
        prepared = self.prepare(s.Scene((card(),), (rect, s.Light(), sphere_light)))
        table, lights, area_count, area_offset = prepared[3], prepared[5], prepared[-3], prepared[-2]
        self.assertEqual((lights, area_count), (1, 2))      # the Directional light stays an analytic light
        for number, light in enumerate((rect, sphere_light)):
            record = table[area_offset+3*number:area_offset+3*number+3]
            radiance, area = s._area_light_radiance(light)
            np.testing.assert_allclose(record[1], (*radiance, area), rtol=1e-5)
            self.assertEqual(float(record[0, 3]), float(light.two_sided))
            first, count = int(record[2, 0]), int(record[2, 1])
            points, normals = s._area_light_samples(light, count)
            self.assertEqual(count, light.light_samples)
            np.testing.assert_array_equal(table[first:first+2*count:2, :3], points)
            np.testing.assert_array_equal(table[first+1:first+2*count:2, :3], normals)
        self.assertEqual(float(table[area_offset+2, 3]), 1.0)
        self.assertEqual(float(table[area_offset+5, 3]), 0.0)

    def test_an_oversized_atlas_refuses_cleanly(self):
        limits = {'max-storage-buffers-per-shader-stage': 8, 'max-storage-buffer-binding-size': 1 << 20,
                  'max-buffer-size': 1 << 20}
        with self.assertRaisesRegex(ValueError, 'needs about.*MiB'):
            r.gpurt._memory_check({'limits': limits}, 5 << 20)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class RenderTests(unittest.TestCase):
    def compare(self, scene, tolerance=1e-3, **kwargs):
        camera = kwargs.pop('camera', s.Camera())
        cpu = s.render(scene, camera, 64, 48, mode='raytrace', **kwargs)
        gpu = gpu3d.render(scene, camera, 64, 48, mode='raytrace', **kwargs)
        np.testing.assert_array_equal(gpu, gpu3d.render(scene, camera, 64, 48, mode='raytrace', **kwargs))
        interior = cpu[..., 3] == 1
        self.assertGreater(interior.sum(), 100)
        delta = np.abs(gpu-cpu)[interior]
        self.assertLess(float(delta.mean()), tolerance, f'mean error {delta.mean():.5f}')
        self.assertLess(float(delta.max()), 20*tolerance, f'max error {delta.max():.5f}')
        return gpu, cpu

    def test_every_pbr_map_with_an_environment_and_lights(self):
        rect = s.Light('Rect', (1, 1, 1), 2., s.Vec3(0, 2, 1), s.Vec3(0, 0, 0), shadows=True)
        point = s.Light('Point', (.6, .8, 1), .8, s.Vec3(-2, 1, 3), shadows=True)
        scene = s.Scene((textured_pbr_sphere(),), (rect, point), environments=(env_of(smooth_map()),))
        for output in ('rgba', 'diffuse', 'specular', 'emission', 'albedo', 'normals'):
            with self.subTest(output=output):
                self.compare(scene, ambient=.05, output=output)

    def test_translucent_pbr_surface_composites_over_what_is_behind_it(self):
        glass = replace(s._sphere(.9, 24, (.8, .5, .2, .55), s.Transform3D(s.Vec3(.3, 0, .6))), material='pbr',
                        metallic=.4, pbr_roughness=.3)
        scene = s.Scene((glass, card(position=s.Vec3(0, 0, -1))), (s.Light(intensity=.8),),
                        environments=(env_of(smooth_map()),))
        self.compare(scene, ambient=.1, tolerance=2e-3)

    def test_area_light_shadows_on_a_pbr_ground(self):
        ground = replace(s.Geometry(np.array([[-8, -1, 6], [8, -1, 6], [8, -1, -12], [-8, -1, -12]], 'f4'),
                                    np.array([[0, 1, 2], [0, 2, 3]], 'i4'), (.5, .5, .5, 1)), material='pbr')
        cube = s._cube(1.0, (.3, .3, .3, 1), s.Transform3D())
        lights = (s.Light('Disc', (1, 1, 1), 3., s.Vec3(0, 3, 0), s.Vec3(0, 0, 0), area_radius=.8,
                          light_samples=16, shadows=True),
                  s.Light('Sphere', (1, .8, .6), 2., s.Vec3(2, 2, 2), area_radius=.4, light_samples=8))
        self.compare(s.Scene((ground, cube), lights), ambient=.05)

    def test_documents_without_any_of_it_render_as_before(self):
        sphere = s._sphere(1.1, 32, (.6, .5, .4, 1), s.Transform3D())
        shiny = replace(sphere, specular=.7, shininess=40.)
        lights = (s.Light(intensity=.8), s.Light('Point', (.5, .7, 1), 1., s.Vec3(-2, 1.5, 2), shadows=True),
                  s.Light('Spot', (1, .8, .6), 1.5, s.Vec3(1, 3, 2), s.Vec3(-20, 0, 0), shadows=True,
                          shadow_blur=3., shadow_samples=8))
        ground = s.Geometry(np.array([[-8, -1, 6], [8, -1, 6], [8, -1, -12], [-8, -1, -12]], 'f4'),
                            np.array([[0, 1, 2], [0, 2, 3]], 'i4'), (.5, .5, .5, 1))
        self.compare(s.Scene((shiny, ground), lights), ambient=.1)
        self.compare(s.Scene((card(texture=gradient(True)), card((.7, .2, .4, .3), s.Vec3(0, 0, 1))), lights[:2]), ambient=.1)
        self.compare(s.Scene((replace(sphere, emission=.6),), lights[:1]), ambient=.05)
        for output in ('depth', 'normals', 'albedo', 'diffuse', 'specular', 'emission', 'position', 'uv'):
            with self.subTest(output=output):
                self.compare(s.Scene((shiny, ground), lights), ambient=.1, output=output)

    def test_scene_with_every_feature_stays_within_eight_storage_bindings(self):
        # The pipeline is created with layout='auto', so a ninth storage binding would fail to compile.
        state = gpu3d._state()
        self.assertIsNone(r.check_capability(state))
        self.assertGreaterEqual(r.gpurt._limits(state)['max-storage-buffers-per-shader-stage'], 8)


if __name__ == '__main__':
    unittest.main()
