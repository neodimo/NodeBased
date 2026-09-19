"""Small CPU-only primary-ray, shading parity and Render3D integration probes."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, scene3d as s
from nodebased.core import CHOICES, Dispatcher, load_document
from nodebased.imaging import Cancelled, Evaluator
from nodebased.knobs import knob_layout
from nodebased.raytrace import Bvh, TriangleSet
from nodebased.viewport3d import Viewport3D
from tests import test_3d_shadows as shadow_reference


def interior(image):
    """Exclude two pixels around coverage, transparency and object boundaries."""
    edge = np.zeros(image.shape[:2], bool)
    dx = np.any(abs(np.diff(image, axis=1)) > .01, axis=2)
    dy = np.any(abs(np.diff(image, axis=0)) > .01, axis=2)
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:] |= dy
    edge[:-1] |= dy
    h, w = edge.shape
    padded = np.pad(edge, 2, constant_values=True)
    edge = np.logical_or.reduce([padded[y:y+h, x:x+w] for y in range(5) for x in range(5)])
    return ~edge & (image[..., -1] > 0)


def scenes():
    # Material/AOV reference recipes: textured transparency, smooth sphere and floor.
    texture = np.full((8, 8, 4), (.125, .25, .375, .5), 'f4')
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


def card(z=0, alpha=.5, color=(.2, .4, .6)):
    return s._card(8, 8, (*color, alpha), s.Transform3D(s.Vec3(0, 0, z)))


class PrimaryRenderTests(unittest.TestCase):
    def compare(self, scene, camera=s.Camera(), **kwargs):
        a = s.render(scene, camera, 48, 36, **kwargs)
        b = s.render(scene, camera, 48, 36, mode='raytrace', **kwargs)
        ids = s.render(scene, camera, 48, 36, output='object_id')
        coverage = s.render(scene, camera, 48, 36)[..., 3:4]
        mask = interior(np.concatenate((ids, coverage), axis=2))
        self.assertGreater(mask.sum(), 10)
        np.testing.assert_allclose(b[mask], a[mask], atol=1e-4, rtol=0)
        ca, cb = a[..., 3] > 0, b[..., 3] > 0
        self.assertGreater((ca & cb).sum()/max(1, (ca | cb).sum()), .99)
        self.assertEqual(b.dtype, np.float32)
        self.assertFalse(b.flags.writeable)
        return a, b

    def test_all_outputs_parity(self):
        for index, scene in enumerate(scenes()):
            for samples in (1, 2):
                for output in s.RENDER_OUTPUTS:
                    with self.subTest(scene=index, samples=samples, output=output):
                        self.compare(scene, output=output, samples=samples, ambient=.13)

    def test_transparent_stack_opaque_stop_and_background(self):
        stack = (card(1, .25, (1, 0, 0)), card(0, .5, (0, 1, 0)), card(-1, .25, (0, 0, 1)))
        for background in ((0, 0, 0, 0), (.8, .4, .2, .5)):
            a, b = self.compare(s.Scene(stack), background=background, samples=2)
            np.testing.assert_allclose(b, a, atol=1e-7, rtol=0)
            expected = np.array(background, float)
            expected[:3] *= expected[3]
            for g in reversed(stack):
                c = np.array(g.color)
                c[:3] *= c[3]
                expected = c + expected*(1-c[3])
            np.testing.assert_allclose(b[18, 24], expected, atol=1e-7)
        for alpha in (1, .999):
            front = card(1, alpha)
            scene = s.Scene((front, card(0, .5), card(-1, 1)))
            # Stops shading at the opaque threshold even when more hits were collected.
            with patch.object(s, '_shade_fragments', wraps=s._shade_fragments) as shader:
                b = s.render(scene, s.Camera(), 16, 12, mode='raytrace')
                self.assertEqual(shader.call_count, 1)
            np.testing.assert_array_equal(b, s.render(s.Scene((front,)), s.Camera(), 16, 12, mode='raytrace'))

    def test_near_far_planes_roll_and_texture_mips(self):
        floor = s._card(40, 40, (1, 1, 1, 1),
                        s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0)))
        _, b = self.compare(s.Scene((floor,)))
        self.assertEqual(b[34, 24, 3], 1)
        self.assertEqual(b[2, 24, 3], 0)
        self.compare(s.Scene((floor,)), s.Camera(near=2.8), output='depth')
        checker = (np.indices((64, 64)).sum(0) % 2).astype('f4')
        texture = np.repeat(checker[..., None], 4, axis=2)
        texture[..., 3] = 1
        self.compare(s.Scene((replace(floor, texture=texture),)), s.Camera(near=2.8), samples=2)
        tilted = replace(card(), texture=texture,
                          transform=s.Transform3D(rotation=s.Vec3(15, 35, 12)))
        for output in ('rgba', 'position', 'uv', 'normals'):
            self.compare(s.Scene((tilted,)), s.Camera(roll=31), output=output)
        for z in (4.95, -20):
            result = s.render(s.Scene((card(z),)), s.Camera(far=10), 12, 10, mode='raytrace')
            self.assertFalse(result.any())

    def test_light_identity_box_filter_and_data_contract(self):
        scene = scenes()[0]
        for samples in (1, 2):
            images = {o: s.render(scene, s.Camera(), 24, 18, mode='raytrace', output=o,
                                  samples=samples, ambient=.13) for o in s.LIGHT_OUTPUTS}
            np.testing.assert_allclose(sum(images[o][..., :3] for o in ('diffuse', 'specular', 'emission')),
                                       images['rgba'][..., :3], atol=1e-6, rtol=0)
            for output in s.LIGHT_OUTPUTS:
                np.testing.assert_array_equal(images[output][..., 3], images['rgba'][..., 3])
                big = s.render(scene, s.Camera(), 24*samples, 18*samples, mode='raytrace',
                               output=output, ambient=.13)
                np.testing.assert_array_equal(images[output], big.reshape(18, samples, 24, samples, 4).mean((1, 3)))
        scene = s.Scene((card(-1, 1), card(0, .1), replace(card(1), texture=np.zeros((2, 2, 4), 'f4'))))
        for output in s.DATA_OUTPUTS:
            a = s.render(scene, s.Camera(), 24, 18, mode='raytrace', output=output)
            b = s.render(scene, s.Camera(), 24, 18, mode='raytrace', output=output,
                         samples=3, background=(1, 1, 1, 1))
            np.testing.assert_array_equal(a, b)
            self.assertTrue((a[..., 3] == 1).all())
            if output == 'depth':
                np.testing.assert_array_equal(a[..., :3], 5)
            if output == 'object_id':
                np.testing.assert_array_equal(a[..., 0], 2)
        empty, depth = s.render(s.Scene(), s.Camera(), 4, 3, (.8, .4, .2, .5),
                                mode='raytrace', return_depth=True, samples=2)
        np.testing.assert_allclose(empty[0, 0], (.4, .2, .1, .5))
        self.assertTrue(np.isinf(depth).all())
        self.assertFalse(depth.flags.writeable)

    def test_analytic_shadow_and_projected_occlusion(self):
        ref = shadow_reference.ShadowTests()
        light = ref.light(s.Vec3(-4, 4, 0))
        scene = s.Scene((ref.ground, ref.blocker()), (light,))
        a = s.render(scene, ref.camera, 64, 48, ambient=.1)
        b = s.render(scene, ref.camera, 64, 48, ambient=.1, mode='raytrace')
        for point, value in (((1, 0, 0), .1), ((0, 0, 0), .1+1/np.sqrt(2))):
            np.testing.assert_allclose(ref.pixel(b, point)[:3], value, atol=1e-5)
            np.testing.assert_allclose(ref.pixel(b, point), ref.pixel(a, point), atol=1e-5)
        texture = np.ones((24, 32, 4), 'f4')
        texture[..., :3] = np.linspace(.1, .8, 32)[None, :, None]
        projector = s.Camera(s.Transform3D(s.Vec3(2, 1, 5)))
        geometries = (card(0, 1), s._card(1, 1, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, 0, 1))))
        for occlusion in ('off', 'depth'):
            projected = s.apply_projection(s.Scene(geometries), s.Projection(projector, texture, occlusion=occlusion))
            for output in ('rgba', 'uv', 'object_id'):
                self.compare(projected, output=output)
        off = s.render(s.apply_projection(s.Scene(geometries), s.Projection(projector, texture)),
                       s.Camera(), 48, 36, mode='raytrace')
        on = s.render(projected, s.Camera(), 48, 36, mode='raytrace')
        self.assertGreater(np.max(abs(off-on)), .1)

    def test_large_scene_one_bvh_and_determinism(self):
        sphere = s._sphere(1, 142, (.6, .4, .2, 1), s.Transform3D())
        self.assertGreaterEqual(len(sphere.triangles), 20000)
        scene = s.Scene((sphere,), (s.Light(position=s.Vec3(2, 4, 5), shadows=True),))
        with patch.object(Bvh, 'build', wraps=Bvh.build) as build:
            a = s.render(scene, s.Camera(), 48, 36, samples=2, mode='raytrace')
            self.assertEqual(build.call_count, 1)
        raster = s.render(scene, s.Camera(), 48, 36, samples=2)
        mask = interior(raster[..., 3:4])
        self.assertGreater(mask.sum(), 0)
        np.testing.assert_allclose(a[mask], raster[mask], atol=1e-4, rtol=0)
        small = scenes()[0]
        base = s.render(small, s.Camera(), 24, 18, samples=2, mode='raytrace')
        np.testing.assert_array_equal(base, s.render(small, s.Camera(), 24, 18, samples=2, mode='raytrace'))
        with patch.object(s, '_PRIMARY_RAY_CHUNK', 37):
            np.testing.assert_array_equal(base, s.render(small, s.Camera(), 24, 18, samples=2, mode='raytrace'))

    def test_budget_cancellation_hit_cap_and_validation(self):
        scene = s.Scene((card(), card(-1)))
        with patch.object(s, 'RAYTRACE_WORK_BUDGET', 1), patch.object(
                s.np, 'broadcast_to', side_effect=AssertionError('heavy allocation')), patch.object(
                Bvh, 'build', side_effect=AssertionError('build')):
            with self.assertRaisesRegex(ValueError, 'Ray-traced render exceeds the CPU reference budget:'):
                s.render(scene, s.Camera(), 10000, 10000, samples=2, mode='raytrace')
        with patch.object(s, 'MAX_HITS_PER_RAY', 1):
            with self.assertRaisesRegex(ValueError, 'MAX_HITS_PER_RAY'):
                s.render(scene, s.Camera(), 12, 10, mode='raytrace')
        with self.assertRaisesRegex(ValueError, 'render mode'):
            s.render(scene, s.Camera(), 12, 10, mode='unknown')
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            s.render(scene, s.Camera(), 12, 10, mode='raytrace', cancel=event)
        event.clear()
        original = TriangleSet._intersect
        def intersect(*args, **kwargs):
            result = original(*args, **kwargs)
            event.set()
            return result
        with patch.object(TriangleSet, '_intersect', new=intersect):
            with self.assertRaises(Cancelled):
                s.render(scene, s.Camera(), 12, 10, mode='raytrace', cancel=event)

    def test_all_hits_sorted_chunks_shared_edges_and_limits(self):
        vertices = np.array([[[0, 0, z], [1, 0, z], [1, 1, z]] for z in (0, 2, 1)], float)
        # Add the neighbour on the diagonal: a shared edge must not become a hole.
        vertices = np.concatenate((vertices, [[[0, 0, 0], [1, 1, 0], [0, 1, 0]]]))
        tri = TriangleSet(vertices[:, 0], vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0], .5)
        bvh = Bvh.build(*tri.aabbs(), leaf_size=1)
        q = np.linspace(0, 1, 17)
        origins = np.column_stack((q, q, np.full(len(q), 3)))
        dirs = np.tile((0, 0, -1), (len(q), 1))
        expected = tri.all_hits(bvh, origins, dirs)
        for chunk in (1, 7):
            for a, b in zip(expected, tri.all_hits(bvh, origins, dirs, chunk=chunk)):
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a['t'], (1, 2, 3, 3))
                np.testing.assert_array_equal(a['primitive'], (1, 2, 0, 3))
        for h in tri.all_hits(bvh, origins, dirs, 1, 3):
            np.testing.assert_array_equal(h['t'], (2,))
        # The renderer composites an exact shared diagonal only once.
        image = s.render(s.Scene((card(),)), s.Camera(), 17, 17, mode='raytrace')
        np.testing.assert_array_equal(image[..., 3], .5)

    def test_transparent_shadow_overdraw_respects_running_budget(self):
        scene = s.Scene((card(), card(-1)),
                        (s.Light(position=s.Vec3(0, 0, 5), shadows=True),))
        rays, triangles = 8 * 6, 4
        # Preflight allows one shaded layer per ray. The second transparent
        # layer must still count its shadow work and refuse before tracing it.
        estimate = (s._shadow_cost(rays, triangles)
                    + s._shadow_cost(rays, triangles, build=False))
        with patch.object(s, 'RAYTRACE_WORK_BUDGET', estimate + 1):
            with self.assertRaisesRegex(ValueError, 'Ray-traced render exceeds the CPU reference budget:'):
                s.render(scene, s.Camera(), 8, 6, mode='raytrace')
        for scene in (s.Scene(), s.Scene((card(),))):
            with patch.object(Bvh, 'build', wraps=Bvh.build) as build:
                s.render(scene, s.Camera(), 8, 6, mode='raytrace', samples=2)
                self.assertEqual(build.call_count, 1)


class RenderModeGraphTests(unittest.TestCase):
    def graph(self):
        d = Dispatcher()
        for key, kind, params in [('card', 'Card3D', dict(spec_amount=.4, emission=.2)),
                                 ('camera', 'Camera3D', {}), ('scene', 'Scene3D', {}),
                                 ('render', 'Render3D', dict(width=24, height=18, samples=2))]:
            d.execute(dict(op='create', id=key, type=kind, params=params))
        d.execute(dict(op='connect', id='scene', input='object0', source='card'))
        for slot in ('scene', 'camera'):
            d.execute(dict(op='connect', id='render', input=slot, source=slot))
        return d

    def test_graph_cache_upgrade_choices_and_backends(self):
        d, evaluator = self.graph(), Evaluator()
        self.assertEqual(CHOICES['render_mode'], ['raster', 'raytrace'])
        knob = next(k for k in knob_layout('Render3D') if 'render_mode' in k.params)
        self.assertEqual((knob.kind, knob.label), ('enum', 'Mode'))
        raster = evaluator.evaluate(d.document, 'render')
        scene = evaluator.evaluate_raster(d.document, 'scene', typed=True)
        d.execute(dict(op='set', id='render', param='render_mode', value='raytrace'))
        with patch.object(s, 'render', wraps=s.render) as renderer:
            actual = evaluator.evaluate(d.document, 'render')
            self.assertTrue(renderer.called)
            renderer.reset_mock()
            np.testing.assert_array_equal(actual, evaluator.evaluate(d.document, 'render'))
            renderer.assert_not_called()
        np.testing.assert_array_equal(actual, s.render(scene, s.Camera(), 24, 18, samples=2, ambient=.1, mode='raytrace'))
        with patch.object(gpu3d, 'available', side_effect=AssertionError('GPU probed')), patch.object(
                gpu3d, 'render', side_effect=AssertionError('GPU rendered')):
            d.execute(dict(op='set', id='render', param='render_backend', value='auto'))
            np.testing.assert_array_equal(actual, evaluator.evaluate(d.document, 'render'))
            d.execute(dict(op='set', id='render', param='render_backend', value='gpu'))
            with self.assertRaisesRegex(ValueError, 'GPU Render3D unsupported: ray-traced mode is CPU-only for now'):
                evaluator.evaluate(d.document, 'render')
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, s.Camera(), 24, 18, mode='raytrace')
        d.execute(dict(op='set', id='render', param='render_backend', value='cpu'))
        del d.document['nodes']['render']['params']['render_mode']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.json'
            path.write_text(json.dumps(d.document))
            old = load_document(path)
        self.assertEqual(old['nodes']['render']['params']['render_mode'], 'raster')
        np.testing.assert_array_equal(raster, Evaluator().evaluate(old, 'render'))

    def test_viewport_stays_raster(self):
        app = QApplication.instance() or QApplication([])
        d = self.graph()
        d.execute(dict(op='set', id='render', param='render_mode', value='raytrace'))
        widget = Viewport3D()
        widget.resize(48, 36)
        widget.set_document(d.document)
        with patch.object(s, 'render', wraps=s.render) as renderer:
            widget.grab()
            self.assertTrue(renderer.called)
            self.assertTrue(all(c.kwargs.get('mode') == 'raster' for c in renderer.call_args_list))
        widget.close()


if __name__ == '__main__':
    unittest.main()
