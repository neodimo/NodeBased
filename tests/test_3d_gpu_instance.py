"""GPU ray tracer instancing (lane L4 step B of 2, DiMo 9/27): a top-level tree of instance
transforms over one bottom-level tree per source mesh, uploaded once (nodebased.gpuinstance).

See docs/3D_ROADMAP.md "Instancing" for the CPU/GPU memory numbers and the scope this narrows to.
"""
import dataclasses
import unittest

import numpy as np

from nodebased import gpu3d, gpuinstance, raytrace, scene3d as s


def _points(positions):
    return s.Geometry(np.asarray(positions, 'f4'), np.zeros((0, 3), 'i4'), (1, 1, 1, 1))


def _sphere_scene(n=10, radius=15, sphere_radius=.4, seed=3, tint=False, variants=1,
                  lights=None, shadow_variant=True):
    sources = [s._sphere(sphere_radius, 8, (.7, .3 + .1 * i, .2, 1), s.Transform3D())
              for i in range(variants)]
    instance = sources[0] if variants == 1 else s.Scene(tuple(sources))
    rng = np.random.RandomState(seed)
    pts = np.stack([rng.uniform(-radius, radius, n), np.zeros(n),
                    rng.uniform(-radius, radius, n)], axis=1)
    points = _points(pts)
    params = {'inst_scale': 1.0}
    if variants > 1:
        params['inst_variant'] = 'cycle'
    if tint:
        params['inst_color_from_points'] = True
    inst_set = s.instances_from_node(points, instance, params)
    if tint:
        rng2 = np.random.RandomState(seed + 1)
        colors = np.concatenate([rng2.uniform(.2, 1, (n, 3)), np.ones((n, 1))], axis=1).astype('f4')
        inst_set = dataclasses.replace(inst_set, colors=colors)
    default_lights = (s.Light(position=s.Vec3(2, 3, 4), shadows=shadow_variant),)
    return s.Scene(instances=(inst_set,), lights=lights if lights is not None else default_lights)


CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 10, 22), s.Vec3(-22, 0, 0)))


class HostTests(unittest.TestCase):
    def test_capability_reasons(self):
        limits = {'max-storage-buffers-per-shader-stage': 8,
                  'max-storage-buffer-binding-size': 1024, 'max-buffer-size': 1024,
                  'max-compute-invocations-per-workgroup': 64,
                  'max-compute-workgroup-size-x': 64, 'max-compute-workgroups-per-dimension': 65535}
        self.assertIsNone(gpuinstance.check_capability({'limits': limits}))
        for key in limits:
            with self.subTest(key=key):
                self.assertIn(key, gpuinstance.check_capability({'limits': dict(limits, **{key: 0})}))

    def test_empty_instance_scene_returns_background(self):
        empty = s.Scene(instances=(s.InstanceSet((), np.zeros((0, 4, 4)), np.zeros(0, 'i4')),))
        prepared = gpuinstance._prepare(empty, None)
        self.assertIsNone(prepared)

    def test_unsupported_scope_raised_before_touching_the_gpu(self):
        texture = np.zeros((2, 2, 4), 'f4')
        textured = dataclasses.replace(s._card(1, 1, (1, 1, 1, 1), s.Transform3D()), texture=texture)
        liquid = dataclasses.replace(s._card(1, 1, (1, 1, 1, 1), s.Transform3D()), material='liquid')
        projected = dataclasses.replace(s._card(1, 1, (1, 1, 1, 1), s.Transform3D()),
                                        projection=s.Projection(s.Camera(), texture))
        points = _points([[0, 0, 0]])
        for source, reason in ((textured, 'textured'), (liquid, 'liquid'), (projected, 'projected')):
            inst_set = s.instances_from_node(points, source, {})
            scene = s.Scene(instances=(inst_set,))
            with self.subTest(reason=reason):
                with self.assertRaises(gpu3d.Unsupported):
                    gpuinstance._prepare(scene, None)

    def test_mixed_scene_is_unsupported_not_silently_dropped(self):
        # Before this step, gpu3d.render never read scene.instances at all (roadmap: "GPU renderer
        # and 3D viewport do not draw particles or instances"), which silently drew nothing for
        # them; a scene combining instances with ordinary geometry must refuse rather than repeat
        # that silent drop now that a real instanced path exists.
        points = _points([[0, 0, 0], [2, 0, 0]])
        sphere = s._sphere(.4, 8, (1, 1, 1, 1), s.Transform3D())
        inst_set = s.instances_from_node(points, sphere, {})
        mixed = s.Scene((s._card(2, 2, (1, 1, 1, 1), s.Transform3D()),), instances=(inst_set,))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(mixed, CAMERA, 8, 8, mode='raytrace')
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(mixed, CAMERA, 8, 8, mode='raster')

    def test_memory_stays_per_source_mesh_not_per_instance(self):
        """docs/3D_ROADMAP.md budget, GPU side: the triangle/attribute buffers this module uploads
        hold each unique source mesh once; only the (small, fixed-size) per-instance table grows
        with instance count, so 100k instances of a 1k-triangle mesh upload ~1k triangles, not 100M.
        """
        vertices = np.random.default_rng(2).uniform(-1, 1, (600, 3)).astype(np.float32)
        triangles = np.random.default_rng(2).integers(0, 600, (1000, 3)).astype(np.int32)
        mesh = s.Geometry(vertices, triangles, (.8, .8, .8, 1.0))
        n = 100_000
        positions = np.random.default_rng(3).uniform(-50, 50, (n, 3))
        inst_set = s.instances_from_node(_points(positions), mesh, {})
        self.assertEqual(len(inst_set), n)
        scene = s.Scene(instances=(inst_set,))
        prepared = gpuinstance._prepare(scene, None)
        _, _, packed_triangles, attrs, instances, table, *_ = prepared
        self.assertEqual(len(packed_triangles), len(triangles))    # one copy of the mesh, not n
        self.assertEqual(len(attrs), len(triangles))
        self.assertEqual(len(instances), n)                        # the only array that grows with n
        self.assertLess(packed_triangles.nbytes + attrs.nbytes, 1_000_000)


@unittest.skipUnless(gpu3d.available(), 'wgpu adapter unavailable')
class GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = gpu3d._state()
        reason = gpuinstance.check_capability(cls.state)
        if reason:
            raise unittest.SkipTest(reason)

    def compare(self, scene, camera=CAMERA, size=(48, 36), ambient=.12, output='rgba', samples=1):
        kwargs = {} if output in s.DATA_OUTPUTS else dict(ambient=ambient)
        a = gpuinstance.render(self.state, scene, camera, *size, (0, 0, 0, 0),
                               ambient if output not in s.DATA_OUTPUTS else 0.0, output=output, samples=samples)
        b = s.render(scene, camera, *size, output=output, mode='raytrace', samples=samples, **kwargs)
        self.assertEqual(a.dtype, np.float32); self.assertFalse(a.flags.writeable)
        error = np.abs(a.astype('f8') - b.astype('f8')).max(initial=0)
        self.assertLessEqual(error, 2e-3, f'{output}: max error {error:.6g}')
        self.assertGreater((b[..., 3] > 0).sum(), 5)
        np.testing.assert_array_equal(a[..., 3] > 0, b[..., 3] > 0)
        return a, b

    def test_beauty_and_shadows_parity(self):
        # Overlapping instances (dense points, several variants) so shadows fall between them too.
        scene = _sphere_scene(n=14, radius=4, sphere_radius=.5, variants=2, tint=True,
                              lights=(s.Light(position=s.Vec3(2, 4, 3), shadows=True),
                                      s.Light('Point', (.4, .7, .9), .6, s.Vec3(-2, 3, 2), shadows=True)))
        self.compare(scene)

    def test_per_instance_tint_reaches_the_gpu(self):
        scene = _sphere_scene(n=6, radius=3, tint=True, lights=())
        a, _ = self.compare(scene, ambient=1.0)
        # Six differently tinted, well-separated spheres: expect several distinct colours on screen.
        visible = a[a[..., 3] > 0][:, :3]
        distinct = {tuple(np.round(px, 2)) for px in visible[::max(1, len(visible)//200)]}
        self.assertGreater(len(distinct), 2)

    def test_aov_parity(self):
        scene = _sphere_scene(n=8, radius=3, tint=True)
        for output in gpuinstance.SUPPORTED_OUTPUTS:
            with self.subTest(output=output):
                self.compare(scene, output=output)

    def test_supersampling(self):
        self.compare(_sphere_scene(n=5, radius=2.5), samples=2)

    def test_unsupported_output_and_unsupported_scope_fall_back_cleanly(self):
        scene = _sphere_scene(n=2)
        with self.assertRaises(gpu3d.Unsupported):
            gpuinstance.render(self.state, scene, CAMERA, 8, 8, (0, 0, 0, 0), 0, output='uv')
        with self.assertRaises(ValueError):
            gpuinstance.render(self.state, scene, CAMERA, 8, 8, (0, 0, 0, 0), 0, output='nonsense')

    def test_routes_through_gpu3d_render(self):
        scene = _sphere_scene(n=6, radius=3, tint=True)
        a = gpu3d.render(scene, CAMERA, 40, 30, ambient=.12, mode='raytrace')
        b = s.render(scene, CAMERA, 40, 30, ambient=.12, mode='raytrace')
        error = np.abs(a.astype('f8') - b.astype('f8')).max(initial=0)
        self.assertLessEqual(error, 2e-3)

    def test_step_a_instance_memory_test_stays_green(self):
        from tests.test_3d_instance import InstanceMemoryTests
        result = unittest.TestResult()
        InstanceMemoryTests('test_100k_instances_of_a_1k_triangle_mesh_stay_small').run(result)
        self.assertFalse(result.errors or result.failures, result.errors or result.failures)


if __name__ == '__main__':
    unittest.main()
