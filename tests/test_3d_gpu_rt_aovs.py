"""GPU ray-traced AOV parity and first-hit/compositing contracts."""
from dataclasses import replace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nodebased import gpu3d, gpurt_render as r, scene3d as s, splats
from nodebased.cancellation import Cancelled
from tests.test_3d_gpu_rt_render import card, gradient
from tests.test_3d_gpu_rt_integration import GraphFixture

OUTPUTS = tuple(name for name in s.RENDER_OUTPUTS if name != 'splats')


def fixtures():
    textured = replace(card(texture=gradient()), specular=.4, emission=.2)
    stack = s.Scene((card(-.5, .3), card(.3, .5), card(1, .2)))
    sphere = replace(s._sphere(.9, 20, (.6, .3, .1, 1), s.Transform3D()),
                     specular=.7, shininess=16, emission=.2)
    blocker = replace(card(1.3, .5, size=.6),
                      transform=s.Transform3D(position=s.Vec3(.5, .2, 1.3)))
    lights = (s.Light('Point', position=s.Vec3(2, 2, 4), shadows=True),)
    parent = s.Transform3D(s.Vec3(.2, .1, -.5), s.Vec3(7, 18, 4),
                           s.Vec3(1.2, .8, 1)).matrix()
    zero_texture = np.zeros((8, 8, 4), 'f4')
    return (
        s.Scene((textured,)),
        stack,
        s.Scene((sphere, blocker), lights),
        s.Scene((card(-.4), replace(card(.5, size=1.3),
            transform=s.Transform3D(s.Vec3(.4, .1, .5), s.Vec3(0, 20, 0))))),
        s.Scene((card(1, 0), card())),
        s.Scene((card(1, texture=zero_texture), card())),
        s.Scene((replace(textured, parent=parent),)),
        s.Scene((card(4, rotation=s.Vec3(80, 0, 0), texture=gradient(), size=8),)),
    )


class HostTests(unittest.TestCase):
    def test_output_validation(self):
        for entry in (r.render,):
            with self.assertRaises(gpu3d.Unsupported):
                entry({}, s.Scene(), s.Camera(), 8, 8, (0, 0, 0, 0), 0, 'splats')
            with self.assertRaisesRegex(ValueError, 'Unknown 3D render output'):
                entry({}, s.Scene(), s.Camera(), 8, 8, (0, 0, 0, 0), 0, 'bad')

    def test_routing_all_outputs(self):
        for output in OUTPUTS:
            target = 'render_beauty' if output == 'rgba' else 'render'
            with self.subTest(output=output), patch.object(gpu3d, '_state', return_value={}), \
                    patch.object(r, 'check_capability', return_value=None), \
                    patch.object(r, target, return_value='sentinel') as render:
                self.assertEqual(gpu3d.render(s.Scene(), s.Camera(), 8, 8,
                    output=output, samples=2, mode='raytrace'), 'sentinel')
                render.assert_called_once()
                if output != 'rgba':
                    self.assertEqual(render.call_args.args[7:9], (output, 2))

    def test_unsupported_before_device_access(self):
        cloud = splats.SplatCloud(np.zeros((1, 3)), np.full((1, 3), .15),
            np.array([[1, 0, 0, 0]]), np.array([.8]), np.zeros((1, 1, 3)), 0)
        cases = (
            (s.Scene((card(),)), 'splats'),
            (s.Scene(splats=(s.SplatInstance(cloud),)), 'depth'),
            (s.Scene((replace(card(), projection=s.Projection(
                s.Camera(), gradient())),)), 'diffuse'))
        for scene, output in cases:
            with self.subTest(output=output), patch.object(gpu3d, '_state',
                    side_effect=AssertionError('must reject before device')):
                with self.assertRaises(gpu3d.Unsupported):
                    gpu3d.render(scene, s.Camera(), 16, 12, output=output, mode='raytrace')

    def test_object_ids_are_geometry_indices(self):
        # Different mip counts make material offsets unsuitable as object IDs.
        attrs = r._prepare(s.Scene((card(texture=gradient()), card(.5))),
                           s.Camera(), 32, 24)[2]
        np.testing.assert_array_equal(attrs[:, 0, 3], (1, 1, 2, 2))

    def test_cpu_data_overflow_contract(self):
        scene = s.Scene(tuple(card(-i*.01, 0) for i in range(4)))
        with patch.object(s, 'MAX_HITS_PER_RAY', 3):
            with self.assertRaisesRegex(ValueError,
                    r'^Ray-traced render exceeds MAX_HITS_PER_RAY \(3\): more than 3 surfaces composited along a ray$'):
                s.render(scene, s.Camera(), 9, 9, output='depth', mode='raytrace')


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUAOVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = gpu3d._state()
        reason = r.check_capability(cls.state)
        if reason:
            raise unittest.SkipTest(reason)

    def render(self, scene, output, samples=1, background=(0, 0, 0, 0),
               cancel=None, size=(48, 36)):
        result = r.render(self.state, scene, s.Camera(), *size, background,
                          .13, output, samples, cancel)
        self.assertEqual(result.dtype, np.float32)
        self.assertFalse(result.flags.writeable)
        return result

    def compare(self, scene, output, samples=1):
        actual = self.render(scene, output, samples)
        expected = s.render(scene, s.Camera(), 48, 36, ambient=.13,
                            output=output, samples=samples, mode='raytrace')
        covered, reference = actual[..., 3] > 0, expected[..., 3] > 0
        union = (covered | reference).sum()
        self.assertGreaterEqual((covered & reference).sum()/max(1, union), .99)
        # Exclude silhouettes only, retaining internal/shared triangle edges.
        padded = np.pad(reference, 1)
        interior = np.logical_and.reduce(
            [padded[y:y+36, x:x+48] for y in range(3) for x in range(3)])
        self.assertTrue(interior.any())
        # Honest f32 tolerances: lights 2e-3 absolute, depth 2e-4 relative,
        # position/UV/normals 1e-3 absolute, IDs exact.
        if output == 'object_id':
            np.testing.assert_array_equal(actual, expected)
        elif output == 'depth':
            np.testing.assert_allclose(actual[interior], expected[interior],
                                       atol=0, rtol=2e-4)
        else:
            tolerance = 1e-3 if output in s.DATA_OUTPUTS else 2e-3
            np.testing.assert_allclose(actual[interior], expected[interior],
                                       atol=tolerance, rtol=0)

    def test_every_output_on_all_scenes(self):
        for index, scene in enumerate(fixtures()):
            for output in OUTPUTS:
                with self.subTest(scene=index, output=output):
                    self.compare(scene, output)

    def test_light_supersampling_and_identity(self):
        for scene in (fixtures()[1], fixtures()[2]):
            for output in s.LIGHT_OUTPUTS:
                if output == 'splats':
                    continue
                with self.subTest(output=output):
                    self.compare(scene, output, samples=2)
                    big = self.render(scene, output, size=(96, 72))
                    expected = big.reshape(36, 2, 48, 2, 4).mean(axis=(1, 3))
                    np.testing.assert_array_equal(self.render(scene, output, 2), expected)
            for samples in (1, 2):
                beauty = self.render(scene, 'rgba', samples)
                parts = [self.render(scene, name, samples)
                         for name in ('diffuse', 'specular', 'emission')]
                np.testing.assert_allclose(sum(p[..., :3] for p in parts),
                                           beauty[..., :3], atol=1e-5, rtol=0)
                for part in parts:
                    np.testing.assert_array_equal(part[..., 3], beauty[..., 3])

    def test_background_samples_and_bands(self):
        scene = fixtures()[1]
        for output in OUTPUTS:
            with self.subTest(output=output):
                samples = 1 if output in s.DATA_OUTPUTS else 2
                expected = self.render(scene, output, samples)
                with patch.object(r, 'GPU_RT_RAYS_PER_SUBMISSION', 192):
                    actual = self.render(scene, output, samples)
                np.testing.assert_array_equal(actual, expected)
                if output != 'rgba':
                    np.testing.assert_array_equal(
                        self.render(scene, output, samples, (.8, .2, .6, 1)), expected)
                if output in s.DATA_OUTPUTS:
                    np.testing.assert_array_equal(
                        self.render(scene, output, 4, (.8, .2, .6, 1)), expected)

    def test_first_positive_alpha_and_ids(self):
        for front in (card(1, 0), card(1, texture=np.zeros((4, 4, 4), 'f4'))):
            scene = s.Scene((front, card()))
            for output in s.DATA_OUTPUTS:
                actual = self.render(scene, output)
                expected = self.render(s.Scene((card(),)), output)
                if output == 'object_id':
                    expected = expected.copy()
                    expected[..., 0] *= 2
                np.testing.assert_array_equal(actual, expected)
        scene = fixtures()[3]
        ids = self.render(scene, 'object_id')
        self.assertEqual(set(np.unique(ids[..., 0])), {0, 1, 2})
        np.testing.assert_array_equal(ids[ids[..., 3] == 0], 0)
        positive = s.Scene((card(1, .01), card()))
        np.testing.assert_array_equal(self.render(positive, 'depth'),
                                      self.render(s.Scene((card(1),)), 'depth'))

    def test_normals_face_eye_and_missing_uv(self):
        scene = s.Scene((replace(card(rotation=s.Vec3(0, 180, 0)), uvs=None),))
        normals = self.render(scene, 'normals')
        np.testing.assert_allclose(normals[normals[..., 3] > 0, :3],
                                   np.broadcast_to((0, 0, 1), (int((normals[..., 3] > 0).sum()), 3)),
                                   atol=1e-3)
        np.testing.assert_array_equal(self.render(scene, 'uv')[..., :3], 0)

    def test_shadows_only_affect_lit_components(self):
        scene = fixtures()[2]
        off = replace(scene, lights=tuple(replace(light, shadows=False)
                                         for light in scene.lights))
        for output in OUTPUTS:
            if output in ('rgba', 'diffuse', 'specular'):
                continue
            with self.subTest(output=output):
                np.testing.assert_array_equal(self.render(scene, output),
                                              self.render(off, output))

    def test_partial_texture_alpha_first_hit(self):
        texture = gradient()
        texture[:, :32] = 0
        scene = s.Scene((card(1, texture=texture), card()))
        for output in s.DATA_OUTPUTS:
            with self.subTest(output=output):
                self.compare(scene, output)

    def test_empty_all_outputs(self):
        for output in OUTPUTS:
            np.testing.assert_array_equal(self.render(s.Scene(), output), 0)

    def test_data_overflow(self):
        scene = s.Scene(tuple(card(-i*.01, 0) for i in range(65)))
        errors = []
        for render in (lambda: self.render(scene, 'depth', size=(9, 9)),
                       lambda: s.render(scene, s.Camera(), 9, 9,
                                        output='depth', mode='raytrace')):
            with self.assertRaises(ValueError) as error:
                render()
            errors.append(str(error.exception))
        self.assertEqual(errors[0], errors[1])
        # A positive-alpha first surface terminates before overflow.
        self.render(s.Scene((card(1, .01), *scene.geometries)), 'depth', size=(9, 9))

    def test_data_cancellation_between_bands(self):
        event = threading.Event()
        device = self.state['device']
        proxy = Mock(wraps=device)
        proxy.limits = device.limits
        proxy.queue = Mock(wraps=device.queue)
        def read(*args, **kwargs):
            result = device.queue.read_buffer(*args, **kwargs)
            event.set()
            return result
        proxy.queue.read_buffer.side_effect = read
        with patch.object(r, 'GPU_RT_RAYS_PER_SUBMISSION', 96):
            with self.assertRaises(Cancelled):
                r.render(dict(self.state, device=proxy), fixtures()[0], s.Camera(),
                         48, 36, (0, 0, 0, 0), .13, 'depth', cancel=event)
        self.assertEqual(proxy.queue.submit.call_count, 1)
        self.render(fixtures()[0], 'depth')


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GraphAOVTests(GraphFixture, unittest.TestCase):
    def test_unsupported_auto_is_exact_cpu(self):
        cloud = splats.SplatCloud(np.zeros((1, 3)), np.full((1, 3), .15),
            np.array([[1, 0, 0, 0]]), np.array([.8]), np.zeros((1, 1, 3)), 0)
        cases = (
            (s.Scene((card(),)), 'splats'),
            (s.Scene(splats=(s.SplatInstance(cloud),)), 'depth'),
            (s.Scene((replace(card(), projection=s.Projection(
                s.Camera(), gradient())),)), 'diffuse'))
        for scene, output in cases:
            with self.subTest(output=output), patch.object(s, 'scene_from_node', return_value=scene):
                self.set('render', 'render_output', output)
                expected = self.render('cpu')
                with patch.object(s, 'render', wraps=s.render) as cpu:
                    np.testing.assert_array_equal(self.render('auto'), expected)
                    self.assertTrue(cpu.called)
                with self.assertRaisesRegex(ValueError, 'GPU Render3D unsupported'):
                    self.render('gpu')

    def test_aovs_use_gpu_and_nested_scene_transform(self):
        self.create('nested', 'Scene3D', tx=.2, ry=18, sx=1.2)
        self.connect('nested', 'object0', 'card')
        self.connect('scene', 'object0', 'nested')
        for output in OUTPUTS:
            self.set('render', 'render_output', output)
            expected = self.render('cpu')
            for backend in ('gpu', 'auto'):
                with self.subTest(output=output, backend=backend), \
                        patch.object(s, 'render', side_effect=AssertionError('CPU fallback')):
                    actual = self.render(backend)
                if output == 'object_id':
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
