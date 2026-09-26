"""Environment light (Light3D type Environment): prefilter, cache, rotation, mesh shading and the node."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import envlight as E, scene3d as s
from nodebased.core import Dispatcher, SPECS, load_document
from nodebased.imaging import Evaluator


def sun_map(width=128, height=64, sky=0.1, sun=20.0):
    d, _ = E.direction_grid(width, height)
    rgb = np.full((height, width, 3), sky, np.float32)
    rgb[d[..., 0] > 0.95] = sun                     # a sun toward +X
    return rgb


def env_of(rgb, **kw):
    return E.Environment(rgb, E.fingerprint_of(rgb), **kw)


class PrefilterTests(unittest.TestCase):
    def test_uniform_map_lights_a_white_surface_to_its_own_value(self):
        env = env_of(np.full((32, 64, 3), 0.7, np.float32))
        n = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1], [0.3, -0.5, 0.8]], float)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        np.testing.assert_allclose(env.diffuse(n), 0.7, atol=2e-3)
        for roughness in (0.0, 0.3, 1.0):
            np.testing.assert_allclose(env.specular(n, np.full(4, roughness)), 0.7, atol=2e-3)

    def test_diffuse_faces_the_sun_and_specular_blurs_with_roughness(self):
        env = env_of(sun_map())
        toward, away = np.array([[1.0, 0, 0]]), np.array([[-1.0, 0, 0]])
        self.assertGreater(env.diffuse(toward)[0, 0], 4 * env.diffuse(away)[0, 0])
        sharp = env.specular(toward, np.array([0.0]))[0, 0]
        blurred = env.specular(toward, np.array([0.5]))[0, 0]
        self.assertGreater(sharp, 15)
        self.assertLess(blurred, 0.7 * sharp)
        self.assertGreater(blurred, env.specular(away, np.array([0.5]))[0, 0])

    def test_rotation_moves_the_highlight(self):
        rgb = sun_map()
        probes = np.array([[1.0, 0, 0], [0, 0, -1.0], [-1.0, 0, 0], [0, 0, 1.0]])
        for degrees, index in ((0, 0), (90, 1), (180, 2), (270, 3)):
            got = env_of(rgb, rotation=degrees).specular(probes, np.zeros(4))[:, 0]
            self.assertEqual(int(np.argmax(got)), index, degrees)
            self.assertGreater(got[index], 15)

    def test_intensity_and_tint_scale_the_light(self):
        rgb = sun_map()
        base = env_of(rgb).diffuse(np.array([[1.0, 0, 0]]))
        got = env_of(rgb, intensity=2.0, tint=(1.0, 0.5, 0.0)).diffuse(np.array([[1.0, 0, 0]]))
        np.testing.assert_allclose(got, base * 2 * np.array((1, .5, 0)), rtol=1e-6)

    def test_blur_raises_the_effective_roughness(self):
        rgb = sun_map()
        toward = np.array([[1.0, 0, 0]])
        sharp = env_of(rgb).specular(toward, np.array([0.0]))[0, 0]
        blurred = env_of(rgb, blur=1.0).specular(toward, np.array([0.0]))[0, 0]
        self.assertLess(blurred, 0.2 * sharp)

    def test_prefilter_runs_once_per_fingerprint(self):
        E.clear_cache()
        rgb = sun_map(64, 32)
        before = E.prefilter_runs
        for degrees in (0, 30, 60):                 # rotation, intensity and blur reuse the prefilter
            env_of(rgb, rotation=degrees, intensity=1 + degrees / 30).diffuse(np.array([[0, 1.0, 0]]))
        self.assertEqual(E.prefilter_runs, before + 1)
        env_of(rgb * 2).diffuse(np.array([[0, 1.0, 0]]))
        self.assertEqual(E.prefilter_runs, before + 2)

    def test_dfg_terms_are_bounded(self):
        a, b = E.dfg(np.linspace(0.05, 1, 20), np.full(20, 0.5))
        self.assertTrue(np.all(a > 0) and np.all(b >= 0) and np.all(a + b <= 1.05))


class MeshShadingTests(unittest.TestCase):
    def test_environment_lights_a_mesh_and_moves_with_rotation(self):
        tilt = np.array((0.7071, 0, 0.7071), np.float32)     # faces the camera and leans toward +X
        card = s.Geometry(np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32),
                          np.array(((0, 1, 2), (0, 2, 3)), np.int32), (1.0, 1.0, 1.0, 1.0),
                          normals=np.tile(tilt, (4, 1)))
        rgb = sun_map()
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0))

        def lit(rotation):
            scene = s.Scene((card,), environments=(env_of(rgb, rotation=rotation),))
            return s.render(scene, camera, 24, 24)[12, 12, :3]
        facing = lit(0)                              # the card's normal is +X, the sun is at +X
        turned = lit(180)
        self.assertGreater(facing[0], 3 * turned[0])
        # A uniform environment leaves the authored colour alone.
        flat = s.Scene((card,), environments=(env_of(np.ones((16, 32, 3), np.float32)),))
        np.testing.assert_allclose(s.render(flat, camera, 24, 24)[12, 12], (1, 1, 1, 1), atol=6e-3)

    def test_no_environment_is_unchanged(self):
        card = s.Geometry(np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0)), np.float32), np.array(((0, 1, 2),), np.int32),
                          (0.5, 0.25, 0.75, 1.0))
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0))
        a = s.render(s.Scene((card,)), camera, 16, 16)
        b = s.render(s.Scene((card,), environments=()), camera, 16, 16)
        np.testing.assert_array_equal(a, b)


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.d, self.e = Dispatcher(), Evaluator()

    def test_spec_and_defaults(self):
        spec = SPECS['Light3D']
        self.assertIn('image', spec['optional_inputs'])
        self.assertEqual((spec['params']['env_rotation'], spec['params']['env_blur']), (0.0, 0.0))

    def test_old_document_gets_the_new_defaults(self):
        d = Dispatcher()
        d.execute(dict(op='create', id='lamp', type='Light3D', params={}))
        doc = copy.deepcopy(d.document)
        for name in ('env_rotation', 'env_blur'):
            del doc['nodes']['lamp']['params'][name]
        with tempfile.TemporaryDirectory() as folder:
            old_file = Path(folder) / 'old.nbcomp'
            old_file.write_text(json.dumps(doc))
            loaded = load_document(old_file)
        self.assertEqual(loaded['nodes']['lamp']['params']['env_rotation'], 0.0)
        self.assertEqual(loaded['nodes']['lamp']['params']['env_blur'], 0.0)

    def test_environment_node_evaluates_to_an_environment_item(self):
        for key, kind, params in (('sky', 'Constant', dict(width=64, height=32, red=0.5, green=0.5, blue=0.5)),
                                  ('env', 'Light3D', dict(light_type='Environment', intensity=2.0,
                                                          env_rotation=45.0, env_blur=0.25)),
                                  ('scene', 'Scene3D', {})):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        self.d.execute(dict(op='connect', id='env', input='image', source='sky'))
        self.d.execute(dict(op='connect', id='scene', input='object0', source='env'))
        scene = self.e.evaluate_raster(self.d.document, 'scene', frame=1, typed=True)
        self.assertEqual(len(scene.environments), 1)
        self.assertEqual(scene.lights, ())
        env = scene.environments[0]
        self.assertEqual((env.intensity, env.rotation, env.blur), (2.0, 45.0, 0.25))
        self.assertEqual(env.rgb.shape, (32, 64, 3))
        np.testing.assert_allclose(env.rgb, 0.5, atol=1e-5)

    def test_disabled_environment_contributes_nothing_and_no_image_is_a_uniform_sky(self):
        self.d.execute(dict(op='create', id='env', type='Light3D', params=dict(light_type='Environment', red=0.2)))
        self.d.execute(dict(op='create', id='scene', type='Scene3D', params={}))
        self.d.execute(dict(op='connect', id='scene', input='object0', source='env'))
        scene = self.e.evaluate_raster(self.d.document, 'scene', frame=1, typed=True)
        np.testing.assert_allclose(scene.environments[0].diffuse(np.array([[0, 1.0, 0]])), [(0.2, 1.0, 1.0)], atol=4e-3)


if __name__ == '__main__':
    unittest.main()
