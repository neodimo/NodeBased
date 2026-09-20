"""GPU f32 parity: relative depth 2e-4, barycentrics 1e-3, edge band 1e-6."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch, Mock
import numpy as np
from nodebased import gpu3d, gpurt, raytrace, scene3d as s
from nodebased.cancellation import Cancelled


def triangles(vertices):
    v = np.asarray(vertices, float).reshape(-1, 3, 3)
    return raytrace.TriangleSet(v[:, 0], v[:, 1]-v[:, 0], v[:, 2]-v[:, 0], .5)


def cards(z):
    return triangles([[[-2, -2, a], [2, -2, a], [0, 2, a]] for a in z])


def scene_triangles(scene):
    vertices = []
    for geometry in scene.geometries:
        m = geometry.world_matrix()
        world = (m[:3, :3] @ geometry.vertices.T + m[:3, 3:4]).T
        vertices.extend(world[geometry.triangles])
    return triangles(vertices)


class HostTests(unittest.TestCase):
    def test_capability_without_gpu(self):
        limits = {'max-storage-buffers-per-shader-stage': 4,
                  'max-storage-buffer-binding-size': 1024, 'max-buffer-size': 1024,
                  'max-compute-invocations-per-workgroup': 64,
                  'max-compute-workgroup-size-x': 64, 'max-compute-workgroups-per-dimension': 65535}
        self.assertIsNone(gpurt.check_capability({'limits': limits}))
        for key in limits:
            with self.subTest(limit=key):
                self.assertIn(key, gpurt.check_capability({'limits': dict(limits, **{key: 0})}))

    def test_guards_and_destroy_once_without_gpu(self):
        tri = cards(range(10)); bvh = raytrace.Bvh.build(*tri.aabbs(), leaf_size=1)
        with patch.object(gpurt, 'STACK_SIZE', 1):
            with self.assertRaisesRegex(ValueError, 'depth.*stack'):
                gpurt.GpuTriangleScene({}, tri, bvh)
        with self.assertRaisesRegex(ValueError, 'needs about.*MiB.*adapter allows.*MiB'):
            gpurt.GpuTriangleScene({'limits': {'max-buffer-size': 64, 'max-storage-buffer-binding-size': 64}}, tri, bvh)
        state = {'device': Mock(), 'wgpu': Mock()}
        buffers = [Mock() for _ in range(3)]
        state['device'].create_buffer_with_data.side_effect = buffers
        with patch.object(gpurt, '_memory_check'), patch.object(gpurt, 'check_capability', return_value=None):
            scene = gpurt.GpuTriangleScene(state, tri, bvh)
            scene.close(); scene.close()
        for b in buffers:
            b.destroy.assert_called_once()

    def test_primary_matches_actual_cpu_rays(self):
        camera = s.Camera(roll=27, near=.3, far=37)
        scene = s.Scene((s._sphere(1, 12, (1, 1, 1, 1), s.Transform3D()),))
        captured = []
        original = raytrace.TriangleSet.nearest_hits
        def query(self, bvh, origins, dirs, tmin, tmax, k, **kwargs):
            captured.append((origins.copy(), dirs.copy(), tmin, tmax))
            return original(self, bvh, origins, dirs, tmin, tmax, k, **kwargs)
        with patch.object(raytrace.TriangleSet, 'nearest_hits', query):
            s.render(scene, camera, 24, 18, mode='raytrace', output='depth')
        o, d, lo, hi = gpurt.primary_rays(camera, 24, 18)
        self.assertTrue(captured)
        np.testing.assert_array_equal(o, captured[0][0])
        np.testing.assert_array_equal(d, captured[0][1])
        np.testing.assert_array_equal(lo, np.broadcast_to(captured[0][2], lo.shape))
        np.testing.assert_array_equal(hi, np.broadcast_to(captured[0][3], hi.shape))
        ro, rd, _, _ = gpurt.primary_rays(camera, 24, 18, rows=(3, 7))
        np.testing.assert_array_equal(ro, o[72:168]); np.testing.assert_array_equal(rd, d[72:168])


@unittest.skipUnless(gpu3d.available(), 'wgpu adapter unavailable')
class GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = gpu3d._state()
        reason = gpurt.check_capability(cls.state)
        if reason:
            raise unittest.SkipTest(reason)

    def compare(self, actual, expected):
        cpu = np.zeros(actual.shape, gpurt.HIT_DTYPE)
        cpu['t'], cpu['primitive'] = np.inf, -1
        for i, row in enumerate(expected):
            cpu[i, :len(row)] = row
        a, b = actual['primitive'] >= 0, cpu['primitive'] >= 0
        self.assertGreaterEqual(np.mean(a == b), .9999)
        agree = a & b
        if not agree.any():
            return
        dt = abs(actual['t'][agree]-cpu['t'][agree])/np.maximum(1, abs(cpu['t'][agree]))
        du = abs(actual['u'][agree]-cpu['u'][agree]); dv = abs(actual['v'][agree]-cpu['v'][agree])
        message = f'max relative t={dt.max():.3g}, u={du.max():.3g}, v={dv.max():.3g}'
        self.assertLessEqual(dt.max(), 2e-4, message)
        edge = np.minimum.reduce((cpu['u'][agree], cpu['v'][agree], 1-cpu['u'][agree]-cpu['v'][agree])) < 1e-3
        same = actual['primitive'][agree] == cpu['primitive'][agree]
        self.assertTrue(np.all(same | (edge & (dt <= 2e-4))), message)
        # Barycentric coordinates belong to different bases when edge ties select
        # different triangles; compare u/v only for matching primitives.
        if same.any():
            self.assertLessEqual(du[same].max(), 1e-3, message)
            self.assertLessEqual(dv[same].max(), 1e-3, message)

    def test_soups_closest_and_nearest(self):
        for count in (500, 5000):
            rng = np.random.default_rng(count)
            v0 = rng.uniform(-3, 3, (count, 3))
            sizes = rng.uniform(.02, .7, (count, 1))
            tri = raytrace.TriangleSet(v0, rng.normal(size=(count, 3))*sizes,
                                      rng.normal(size=(count, 3))*sizes, 1)
            bvh = raytrace.Bvh.build(*tri.aabbs())
            o = rng.uniform(-4, 4, (2048, 3)); d = rng.normal(size=o.shape)
            # Exactly parallel slabs, origins within the soup, both ray signs.
            d[:384] = np.tile(np.concatenate((np.eye(3), -np.eye(3))), (64, 1))
            o[:384] *= .25
            with gpurt.GpuTriangleScene(self.state, tri, bvh) as scene:
                for k in (1, 3, 8):
                    with self.subTest(triangles=count, k=k):
                        actual = gpurt.nearest_hits(scene, o, d, 0, 30, k)
                        expected = tri.nearest_hits(bvh, o, d, 0, 30, k)
                        self.compare(actual, expected)

    def test_peeling_coincident_stacked_and_cursor(self):
        o = np.array([[.1, .2, 20], [-.2, .3, 20]])
        d = np.tile([0, 0, -1], (2, 1))
        for levels in ([0]*12, list(range(12)), [0, 0, 1, 1, 2, 3], np.arange(70)/8):
            tri = cards(levels); bvh = raytrace.Bvh.build(*tri.aabbs(), leaf_size=1)
            expected = tri.all_hits(bvh, o, d, 0, 100, max_hits=80)
            with gpurt.GpuTriangleScene(self.state, tri, bvh) as scene:
                for k in (1, 3, 8):
                    actual = gpurt.all_hits(scene, o, d, 0, 100, max_hits=80, k=k)
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a['primitive'], b['primitive'])
                        np.testing.assert_allclose(a['t'], b['t'], rtol=2e-4)
                ct, cp = np.array([expected[i][2]['t'] for i in range(2)]), np.array([expected[i][2]['primitive'] for i in range(2)])
                self.compare(gpurt.nearest_hits(scene, o, d, 0, 100, 8, after_t=ct, after_primitive=cp),
                             tri.nearest_hits(bvh, o, d, 0, 100, 8, after_t=ct, after_primitive=cp))
                if len(levels) == 70:
                    with self.assertRaisesRegex(ValueError, 'MAX_HITS_PER_RAY'):
                        gpurt.all_hits(scene, o, d)
                self.compare(gpurt.nearest_hits(scene, o, d, 18, 20, 8), tri.nearest_hits(bvh, o, d, 18, 20, 8))

    def test_grid_edges_vertices_no_cracks(self):
        vertices = []
        for y in range(16):
            for x in range(16):
                vertices.extend(([[x, y, 0], [x+1, y, 0], [x+1, y+1, 0]],
                                 [[x, y, 0], [x+1, y+1, 0], [x, y+1, 0]]))
        tri = triangles(vertices); bvh = raytrace.Bvh.build(*tri.aabbs())
        x, y = np.meshgrid(np.linspace(0, 16, 129), np.arange(17))
        xy = np.column_stack((x.ravel(), y.ravel()))
        diagonal = np.concatenate([np.column_stack((np.arange(16)+a, np.arange(16)+a)) for a in np.linspace(0, 1, 129)])
        xy = np.concatenate((xy, xy[:, ::-1], diagonal))
        o = np.column_stack((xy, np.ones(len(xy)))); d = np.tile([0, 0, -1], (len(o), 1))
        self.assertTrue(all(len(h) for h in tri.nearest_hits(bvh, o, d, 0, 10, 1)))
        with gpurt.GpuTriangleScene(self.state, tri, bvh) as scene:
            self.assertTrue(np.all(gpurt.nearest_hits(scene, o, d, 0, 10, 1)['primitive'] >= 0))

    def test_chunks_determinism_empty_and_2d_dispatch(self):
        tri = cards(range(12)); bvh = raytrace.Bvh.build(*tri.aabbs())
        o = np.tile([.1, .2, 20], (333, 1)); d = np.tile([0, 0, -1], (333, 1))
        with gpurt.GpuTriangleScene(self.state, tri, bvh) as scene:
            expected = gpurt.nearest_hits(scene, o, d, 0, 100, 8)
            for chunk in (7, 333):
                np.testing.assert_array_equal(expected, gpurt.nearest_hits(scene, o, d, 0, 100, 8, chunk=chunk))
            limits = dict(self.state['device'].limits, **{'max-compute-workgroups-per-dimension': 3})
            with patch.dict(self.state, limits=limits):
                np.testing.assert_array_equal(expected, gpurt.nearest_hits(scene, o, d, 0, 100, 8))
            self.assertEqual(gpurt.nearest_hits(scene, o[:0], d[:0], 0, 100, 3).shape, (0, 3))
        tri = triangles([])
        with gpurt.GpuTriangleScene(self.state, tri, raytrace.Bvh.build(*tri.aabbs())) as scene:
            result = gpurt.nearest_hits(scene, o, d, 0, 100, 3)
            self.assertTrue(np.all(result['primitive'] == -1)); self.assertTrue(np.isinf(result['t']).all())

    def test_cancellation_cleanup(self):
        tri = cards([0]); bvh = raytrace.Bvh.build(*tri.aabbs())
        scene = gpurt.GpuTriangleScene(self.state, tri, bvh)
        device = self.state['device']; original = device.create_buffer_with_data
        # Check both post-readback and next-chunk cancellation. Native buffers
        # remain native; only destroy is wrapped so binding validation still runs.
        for checks in ([False, False, False, True], [False, False, False, False, True]):
            event = Mock(); event.is_set.side_effect = checks
            destroyed = []
            with ExitStack() as patches:
                def create(**kwargs):
                    b = original(**kwargs)
                    destroyed.append(patches.enter_context(patch.object(b, 'destroy', wraps=b.destroy)))
                    return b
                patches.enter_context(patch.object(device, 'create_buffer_with_data', side_effect=create))
                with self.assertRaises(Cancelled):
                    gpurt.nearest_hits(scene, np.tile([0, 0, 1], (4, 1)), np.tile([0, 0, -1], (4, 1)),
                                       0, 10, 1, chunk=2, cancel=event)
                self.assertEqual(len(destroyed), 2)
                for destroy in destroyed:
                    destroy.assert_called_once()
        scene.close(); scene.close()

    def test_depth_render(self):
        geometry = (s._sphere(1, 24, (1, 1, 1, 1), s.Transform3D()),
                    s._card(8, 8, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, -1.2, 0), s.Vec3(-90, 0, 0))))
        cpu_scene = s.Scene(geometry); camera = s.Camera()
        depth = s.render(cpu_scene, camera, 64, 48, output='depth', mode='raytrace')
        tri = scene_triangles(cpu_scene); bvh = raytrace.Bvh.build(*tri.aabbs())
        o, d, lo, hi = gpurt.primary_rays(camera, 64, 48)
        with gpurt.GpuTriangleScene(self.state, tri, bvh) as scene:
            actual = gpurt.nearest_hits(scene, o, d, lo, hi, 1)['t'].reshape(48, 64)
        covered = depth[..., 3] > 0
        mask = covered.copy(); mask[[0, -1], :] = False; mask[:, [0, -1]] = False
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            mask &= np.roll(covered, (dy, dx), (0, 1))
        self.assertGreater(mask.sum(), 100)
        np.testing.assert_allclose(actual[mask], depth[..., 0][mask], rtol=2e-4, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
