"""Named AOV contracts: analytic CPU probes, graph caching and optional GPU parity.

Lighting identities sum RGB; each component independently carries beauty alpha.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, scene3d as s
from nodebased.core import CHOICES, Dispatcher, load_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout, resolve_kind
from tests.test_3d_gpu_shadows import boundary, dilate


# 'relight' is the multichannel relight bundle (tests/test_3d_relight_bundle.py); it is neither a
# LIGHT_OUTPUTS nor a DATA_OUTPUTS single-channel image, so it is intentionally exercised there.
OUTPUTS = ('rgba', 'depth', 'normals', 'albedo', 'diffuse', 'specular',
           'emission', 'position', 'uv', 'object_id', 'relight', 'splats', 'normals_blend')


def scenes():
    texture = np.empty((8, 8, 4), 'f4')
    texture[:] = (.125, .25, .375, .5)
    card = replace(s._card(3.5, 3, (.6, .4, .8, .5), s.Transform3D(), texture),
                   specular=.4, shininess=12, emission=.3)
    blocker = replace(s._card(.8, .9, (.2, .7, .4, .5),
                             s.Transform3D(s.Vec3(.1, .1, .8))), specular=.2, emission=.1)
    lights = (s.Light(position=s.Vec3(-2, 3, 5), shadows=True),
              s.Light('Point', (.3, .6, .9), .7, s.Vec3(2, 1, 4)))
    sphere = replace(s._sphere(.9, 16, (.6, .3, .1, .7), s.Transform3D()),
                     specular=.7, shininess=16, emission=.2)
    ground = s._card(6, 6, (.4, .5, .6, 1),
                     s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0)))
    return (s.Scene((card, blocker), lights), s.Scene((sphere, ground), lights),
            s.Scene((card, blocker)))


def plane_hit(camera, x, y, width, height, geometry):
    """Independent ray/plane intersection using the documented camera convention."""
    eye = camera.transform.position.array().astype(float)
    forward = camera.target.array() - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 1, 0)); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    roll = math.radians(camera.roll)
    right, up = math.cos(roll)*right + math.sin(roll)*up, math.cos(roll)*up - math.sin(roll)*right
    scale = math.tan(math.radians(camera.fov)/2)
    ray = forward + right*(2*(x+.5)/width-1)*(width/height)*scale + up*(1-2*(y+.5)/height)*scale
    matrix = geometry.world_matrix()
    normal = np.linalg.inv(matrix[:3, :3]).T @ np.array((0., 0., 1.))
    return eye + ray * (np.dot(matrix[:3, 3]-eye, normal) / np.dot(ray, normal))


class AOVTests(unittest.TestCase):
    def test_lighting_identity_and_supersampling(self):
        for number, scene in enumerate(scenes()):
            for samples in (1, 2, 3):
                with self.subTest(scene=number, samples=samples):
                    args = dict(ambient=.13, samples=samples)
                    beauty = s.render(scene, s.Camera(), 24, 24, **args)
                    parts = [s.render(scene, s.Camera(), 24, 24, output=name,
                                      background=(.9, .7, .3, 1), **args)
                             for name in ('diffuse', 'specular', 'emission')]
                    np.testing.assert_allclose(sum(p[..., :3] for p in parts), beauty[..., :3], atol=1e-5, rtol=0)
                    for part in parts:
                        np.testing.assert_array_equal(part[..., 3], beauty[..., 3])
                    if not scene.lights:
                        albedo = s.render(scene, s.Camera(), 24, 24, output='albedo', **args)
                        np.testing.assert_array_equal(parts[0], albedo)
                        np.testing.assert_array_equal(parts[1][..., :3], 0)

    def test_box_filter_is_exact_for_every_light_output(self):
        scene = scenes()[0]
        for output in s.LIGHT_OUTPUTS:
            for samples in (2, 3):
                big = s.render(scene, s.Camera(), 20*samples, 16*samples, output=output)
                expected = big.reshape(16, samples, 20, samples, 4).mean(axis=(1, 3))
                actual = s.render(scene, s.Camera(), 20, 16, output=output, samples=samples)
                np.testing.assert_array_equal(actual, expected)

    def test_shadows_affect_only_diffuse_and_specular(self):
        scene = scenes()[0]
        for output in ('albedo', 'diffuse', 'specular', 'emission'):
            on = s.render(scene, s.Camera(), 48, 48, output=output, ambient=.1)
            off = s.render(scene, s.Camera(), 48, 48, output=output, ambient=.1, shadows=False)
            if output in ('albedo', 'emission'):
                np.testing.assert_array_equal(on, off)
            else:
                self.assertGreater(float(np.max(off[..., :3]-on[..., :3])), .001)
                self.assertTrue(np.all(on[..., :3] <= off[..., :3] + 1e-7))

    def test_analytic_lighting_components(self):
        texture = np.full((4, 4, 4), (.125, .25, .375, .5), 'f4')
        card = replace(s._card(8, 8, (.4, .6, .8, .5), s.Transform3D(), texture),
                       specular=.7, shininess=16, emission=2)
        light = s.Light(position=s.Vec3(0, 0, 5), color=(1, .5, .25), intensity=2)
        scene = s.Scene((card,), (light,))
        albedo = np.array(card.color[:3])*.5*texture[0, 0, :3]
        alpha = .25
        colour = np.array(light.color)*light.intensity
        for x, y in ((32, 32), (46, 21)):
            point = plane_hit(s.Camera(), x, y, 65, 65, card)
            view = np.array((0, 0, 5))-point; view /= np.linalg.norm(view)
            half = view + (0, 0, 1); half /= np.linalg.norm(half)
            expected = dict(albedo=albedo, diffuse=albedo*(.15+colour),
                            specular=.7*half[2]**16*colour*alpha, emission=albedo*2)
            for output, rgb in expected.items():
                with self.subTest(output=output, pixel=(x, y)):
                    image = s.render(scene, s.Camera(), 65, 65, ambient=.15, output=output)
                    np.testing.assert_allclose(image[y, x], (*rgb, alpha), atol=2e-6, rtol=0)
        zero = replace(scene, geometries=(replace(card, specular=0),))
        np.testing.assert_array_equal(s.render(zero, s.Camera(), 17, 17, output='specular')[..., :3], 0)

    def test_position_and_uv_analytic_transformed_plane(self):
        camera = s.Camera(s.Transform3D(s.Vec3(.4, .3, 5)), roll=7)
        card = s._card(5, 4, (1, 1, 1, .2),
                       s.Transform3D(s.Vec3(.2, -.1, .3), s.Vec3(12, 25, 5), s.Vec3(.9, 1.1, 1)))
        scene = s.Scene((card,))
        position = s.render(scene, camera, 65, 49, output='position')
        uv = s.render(scene, camera, 65, 49, output='uv')
        for x, y in ((32, 24), (17, 13), (46, 34), (42, 16)):
            world = plane_hit(camera, x, y, 65, 49, card)
            local = np.linalg.inv(card.world_matrix()) @ np.append(world, 1)
            np.testing.assert_allclose(position[y, x], (*world, 1), atol=2e-6, rtol=0)
            np.testing.assert_allclose(uv[y, x], (local[0]/5+.5, local[1]/4+.5, 0, 1), atol=2e-6, rtol=0)
        # Card fills the frame with corners approaching UV (0,0) and (1,1).
        flat = s._card(10, 10, (1, 1, 1, 1), s.Transform3D())
        camera = s.Camera(fov=90)
        image = s.render(s.Scene((flat,)), camera, 65, 65, output='uv')
        for x, y in ((0, 0), (64, 0), (0, 64), (64, 64), (32, 32)):
            np.testing.assert_allclose(image[y, x], ((x+.5)/65, 1-(y+.5)/65, 0, 1), atol=1e-6)
        no_uv = replace(flat, uvs=None)
        np.testing.assert_array_equal(s.render(s.Scene((no_uv,)), camera, 17, 17, output='uv')[..., :3], 0)

    def test_projected_uv_and_shading_components(self):
        scene = scenes()[0]
        projector = s.Camera(s.Transform3D(s.Vec3(1, .5, 5)), fov=55)
        texture = np.ones((12, 24, 4), 'f4')
        texture[..., :3] = np.linspace(.1, .8, 24)[None, :, None]
        projection = s.Projection(projector, texture)
        card = replace(scene.geometries[0], projection=projection)
        scene = replace(scene, geometries=(card, scene.geometries[1]))
        uv = s.render(s.Scene((card,)), s.Camera(), 65, 65, output='uv')
        for x, y in ((32, 32), (21, 19), (44, 41)):
            world = plane_hit(s.Camera(), x, y, 65, 65, card)
            # Projection filmback is texture aspect, independent of render aspect.
            eye, view = s._view_basis(projector)
            local = view @ (world-eye)
            focal = 1/math.tan(math.radians(projector.fov)/2)
            expected = (local[0]*focal/2/-local[2]*.5+.5, local[1]*focal/-local[2]*.5+.5, 0, 1)
            np.testing.assert_allclose(uv[y, x], expected, atol=2e-6, rtol=0)
        parts = [s.render(scene, s.Camera(), 32, 32, output=o, samples=2, ambient=.1)
                 for o in ('diffuse', 'specular', 'emission')]
        beauty = s.render(scene, s.Camera(), 32, 32, samples=2, ambient=.1)
        np.testing.assert_allclose(sum(p[..., :3] for p in parts), beauty[..., :3], atol=1e-5, rtol=0)
        projected_albedo = s.render(s.Scene((card,)), s.Camera(), 32, 32, output='albedo')
        unlit = s.render(s.Scene((replace(card, emission=0),)), s.Camera(), 32, 32)
        np.testing.assert_array_equal(projected_albedo, unlit)

    def test_data_first_hit_ids_no_background_or_antialiasing(self):
        far = s._card(3, 3, (1, 1, 1, 1), s.Transform3D())
        near = s._card(1, 1, (1, 1, 1, .1), s.Transform3D(s.Vec3(0, 0, 1)))
        invisible = replace(near, color=(1, 1, 1, 0), transform=s.Transform3D(s.Vec3(0, 0, 2)))
        scene = s.Scene((far, near, invisible))
        ids = s.render(scene, s.Camera(), 65, 65, output='object_id')
        self.assertEqual(set(np.unique(ids[..., 0])), {0, 1, 2})
        np.testing.assert_array_equal(ids[32, 32], (2, 0, 0, 1))
        np.testing.assert_array_equal(ids[32, 15], (1, 0, 0, 1))
        np.testing.assert_array_equal(ids[0, 0], 0)
        # IDs follow scene order rather than draw/depth order.
        reverse = s.render(replace(scene, geometries=(near, far)), s.Camera(), 65, 65, output='object_id')
        np.testing.assert_array_equal(reverse[32, 32], (1, 0, 0, 1))
        for output in s.DATA_OUTPUTS:
            base = s.render(scene, s.Camera(), 65, 65, output=output)
            other = s.render(scene, s.Camera(), 65, 65, output=output, samples=3, background=(1, .4, .3, 1))
            np.testing.assert_array_equal(base, other)
            self.assertEqual(set(np.unique(other[..., 3])), {0, 1})
            np.testing.assert_array_equal(other[ids[..., 3] == 0], 0)
            self.assertEqual(other.dtype, np.float32)
            self.assertFalse(other.flags.writeable)
        position = s.render(scene, s.Camera(), 65, 65, output='position')
        self.assertEqual(set(np.unique(position[..., 2])), {0, 1})
        # Texture alpha, including fully invisible front texels, controls data hits.
        transparent = replace(near, texture=np.zeros((2, 2, 4), 'f4'))
        hit = s.render(s.Scene((far, transparent)), s.Camera(), 65, 65, output='object_id')
        np.testing.assert_array_equal(hit[32, 32], (1, 0, 0, 1))

    def test_existing_outputs_pixel_exact(self):
        card = s._card(2, 2, (.25, .5, .75, .5), s.Transform3D())
        scene = s.Scene((card,))
        expected = dict(rgba=(.125, .25, .375, .5), depth=(5, 5, 5, 1), normals=(0, 0, 1, 1))
        for output, value in expected.items():
            image = s.render(scene, s.Camera(), 65, 65, output=output)
            np.testing.assert_array_equal(image[32, 32], value)
            np.testing.assert_array_equal(image[0, 0], 0)
        back = replace(card, transform=s.Transform3D(s.Vec3(0, 0, -1)), color=(0, 0, 1, 1))
        image = s.render(s.Scene((card, back)), s.Camera(), 65, 65)
        np.testing.assert_array_equal(image[32, 32], (.125, .25, .875, 1))

    def graph(self):
        d = Dispatcher()
        for key, kind, params in (('card', 'Card3D', dict(spec_amount=.5, emission=.2, alpha=.5)),
                                  ('light', 'Light3D', {}), ('camera', 'Camera3D', {}),
                                  ('scene', 'Scene3D', {}),
                                  ('render', 'Render3D', dict(width=32, height=24, samples=2))):
            d.execute(dict(op='create', id=key, type=kind, params=params))
        for slot, source in (('object0', 'card'), ('object1', 'light')):
            d.execute(dict(op='connect', id='scene', input=slot, source=source))
        for slot in ('scene', 'camera'):
            d.execute(dict(op='connect', id='render', input=slot, source=slot))
        return d

    def test_choices_knob_graph_cache_and_old_documents(self):
        self.assertEqual(s.RENDER_OUTPUTS, OUTPUTS + s.VOLUME_OUTPUTS)
        # 'multichannel' is a graph-level output (tests/test_3d_multichannel_exr.py), not a scene3d.render output.
        self.assertEqual(CHOICES['render_output'], list(OUTPUTS + s.VOLUME_OUTPUTS) + ['multichannel'])
        knob = next(g for g in knob_layout('Render3D') if 'render_output' in g.params)
        self.assertEqual((knob.kind, knob.label), ('enum', 'Output'))
        self.assertEqual(resolve_kind('Render3D', 'render_output', 'uv'), 'enum')
        d, evaluator = self.graph(), Evaluator()
        scene = evaluator.evaluate_raster(d.document, 'scene', typed=True)
        with patch.object(s, 'render', wraps=s.render) as render:
            for output in OUTPUTS:
                d.execute(dict(op='set', id='render', param='render_output', value=output))
                calls = render.call_count
                actual = evaluator.evaluate(d.document, 'render')
                self.assertGreater(render.call_count, calls)
                calls = render.call_count
                np.testing.assert_array_equal(evaluator.evaluate(d.document, 'render'), actual)
                self.assertEqual(render.call_count, calls)
                expected = s.render(scene, s.Camera(), 32, 24, samples=2, output=output, ambient=.1)
                if output == 'relight':
                    # scene3d.render(output='relight') returns (rgba, layers); the graph's cached
                    # value is a Raster whose .pixels is that same rgba array (the layers live in
                    # Raster.layers, checked in tests/test_3d_relight_bundle.py).
                    expected = expected[0]
                np.testing.assert_array_equal(actual, expected)
                if output in OUTPUTS[:3]:
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder)/'old.json'
                        old = json.loads(json.dumps(d.document))
                        del old['nodes']['render']['params']['render_backend']
                        path.write_text(json.dumps(old))
                        loaded = load_document(path)
                    self.assertEqual(loaded['nodes']['render']['params']['render_output'], output)
                    np.testing.assert_array_equal(Evaluator().evaluate(loaded, 'render'), actual)
        for renderer in (s.render, gpu3d.render):
            with self.assertRaisesRegex(ValueError, 'Unknown 3D render output'):
                renderer(scene, s.Camera(), 8, 8, output='not_an_aov')

    def test_disabled_and_projected_auto_graph(self):
        d, evaluator = self.graph(), Evaluator()
        for output in OUTPUTS:
            d.execute(dict(op='set', id='render', param='render_output', value=output))
            expected = evaluator.evaluate(d.document, 'render')
            d.execute(dict(op='disable', id='render', value=True))
            np.testing.assert_array_equal(evaluator.evaluate(d.document, 'render'), 0)
            d.execute(dict(op='disable', id='render', value=False))
            np.testing.assert_array_equal(evaluator.evaluate(d.document, 'render'), expected)
        d.execute(dict(op='create', id='texture', type='Checker', params=dict(width=16, height=16)))
        d.execute(dict(op='create', id='project', type='Project3D'))
        for slot, source in (('image', 'texture'), ('camera', 'camera'), ('geometry', 'card')):
            d.execute(dict(op='connect', id='project', input=slot, source=source))
        d.execute(dict(op='connect', id='scene', input='object0', source='project'))
        for output in OUTPUTS[3:]:
            d.execute(dict(op='set', id='render', param='render_output', value=output))
            d.execute(dict(op='set', id='render', param='render_backend', value='cpu'))
            expected = evaluator.evaluate(d.document, 'render')
            d.execute(dict(op='set', id='render', param='render_backend', value='auto'))
            with patch.object(gpu3d, 'available', return_value=True):
                np.testing.assert_array_equal(evaluator.evaluate(d.document, 'render'), expected)

    def test_gpu_object_id_packing_without_device(self):
        scene = scenes()[0]
        _, _, vertices, _, _ = gpu3d._prepare(scene, s.Camera(), 32, 32, None)
        packed = np.concatenate(vertices)
        self.assertEqual(packed.shape[1], 20)
        self.assertEqual(packed.strides[0], 80)
        self.assertEqual(set(packed[:, 19]), {1, 2})
        for triangle in vertices:
            np.testing.assert_array_equal(triangle[:, 19], triangle[0, 19])


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUAOVTests(unittest.TestCase):
    def test_existing_outputs_pixel_exact_and_large_ids(self):
        card = s._card(2, 2, (.25, .5, .75, .5), s.Transform3D())
        expected = dict(rgba=(.125, .25, .375, .5), depth=(5, 5, 5, 1), normals=(0, 0, 1, 1))
        for output, value in expected.items():
            image = gpu3d.render(s.Scene((card,)), s.Camera(), 65, 65, output=output)
            np.testing.assert_array_equal(image[32, 32], value)
            np.testing.assert_array_equal(image[0, 0], 0)
        # Empty geometries still occupy an ID. 2049 cannot be represented in float16.
        empty = replace(card, triangles=np.empty((0, 3), 'i4'))
        scene = s.Scene((empty,)*2048 + (card,))
        with patch.dict(gpu3d._state(), {'format': 'rgba16float', 'pipelines': {}}):
            image = gpu3d.render(scene, s.Camera(), 17, 17, output='object_id')
        np.testing.assert_array_equal(image[8, 8], (2049, 0, 0, 1))

    def test_all_aovs_parity(self):
        for index, scene in enumerate(scenes()):
            ids = s.render(scene, s.Camera(), 48, 48, output='object_id')
            beauty = s.render(scene, s.Camera(), 48, 48, ambient=.13)
            unshadowed = s.render(scene, s.Camera(), 48, 48, ambient=.13, shadows=False)
            edges = boundary(np.concatenate((ids, beauty-unshadowed), axis=2))
            interior = (ids[..., 3] > 0) & ~dilate(edges, 2)
            self.assertGreater(interior.sum(), 10)
            # 'relight' returns (rgba, layers), not a single array, and is CPU-only: excluded from
            # this array-shaped GPU-vs-CPU parity sweep (see tests/test_3d_relight_bundle.py).
            for output in (o for o in OUTPUTS[3:-2] if o != 'relight'):
                with self.subTest(scene=index, output=output):
                    args = dict(output=output, samples=2, ambient=.13, background=(.7, .4, .2, 1))
                    cpu = s.render(scene, s.Camera(), 48, 48, **args)
                    gpu = gpu3d.render(scene, s.Camera(), 48, 48, **args)
                    self.assertEqual(gpu.dtype, np.float32)
                    self.assertFalse(gpu.flags.writeable)
                    if output == 'object_id':
                        np.testing.assert_array_equal(gpu[interior], cpu[interior])
                    elif output in s.DATA_OUTPUTS:
                        # Measured on an RTX 3080 Ti: GPU vs CPU position/uv differ by up to ~3.5e-4 (float32
                        # perspective interpolation order); object_id is compared exactly above.
                        np.testing.assert_allclose(gpu[interior], cpu[interior], atol=1e-3, rtol=0)
                    else:
                        self.assertLess(float(np.abs(gpu[interior]-cpu[interior]).mean()), 5e-3)
                    if output in s.DATA_OUTPUTS:
                        np.testing.assert_array_equal(gpu, gpu3d.render(scene, s.Camera(), 48, 48, output=output))

    def test_identity_and_half_precision_path(self):
        for half in (False, True):
            state = gpu3d._state()
            overrides = {'format': 'rgba16float', 'pipelines': {}} if half else {}
            with patch.dict(state, overrides):
                for scene in scenes():
                    for samples in (2, 3):
                        beauty = gpu3d.render(scene, s.Camera(), 24, 24, samples=samples, ambient=.1)
                        parts = [gpu3d.render(scene, s.Camera(), 24, 24, samples=samples,
                                             ambient=.1, output=o) for o in ('diffuse', 'specular', 'emission')]
                        tolerance = 5e-3 if state['format'] == 'rgba16float' else 1e-5
                        np.testing.assert_allclose(sum(p[..., :3] for p in parts), beauty[..., :3], atol=tolerance, rtol=0)
                        for part in parts:
                            np.testing.assert_array_equal(part[..., 3], beauty[..., 3])
                # Data remains float32 even when light targets are forced to half precision.
                card = s._card(8, 8, (1, 1, 1, .5), s.Transform3D(s.Vec3(0, 0, .1234567)))
                for output in ('position', 'uv', 'object_id'):
                    scene = s.Scene((card,))
                    cpu = s.render(scene, s.Camera(), 33, 33, output=output)
                    gpu = gpu3d.render(scene, s.Camera(), 33, 33, output=output, samples=3)
                    np.testing.assert_allclose(gpu[2:-2, 2:-2], cpu[2:-2, 2:-2], atol=1e-3, rtol=0)
