"""CPU reference shadow pixels and graph/backend/viewport contracts."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, scene3d as s
from nodebased.core import Dispatcher, load_document
from nodebased.imaging import Cancelled, Evaluator
from nodebased.knobs import knob_layout, resolve_kind
from nodebased.viewport3d import Viewport3D


class ShadowTests(unittest.TestCase):
    camera = s.Camera(s.Transform3D(s.Vec3(4, 5, 7)))
    ground = s._card(12, 12, (1, 1, 1, 1), s.Transform3D(rotation=s.Vec3(-90, 0, 0)))

    def blocker(self, alpha=1, height=1):
        return s._card(1, 1, (1, 1, 1, alpha),
                       s.Transform3D(s.Vec3(0, height, 0), s.Vec3(-90, 0, 0)))

    def light(self, position=s.Vec3(0, 4, 0), **kwargs):
        return s.Light(position=position, shadows=True, **kwargs)

    def render(self, light=None, alpha=1, **kwargs):
        scene = s.Scene((self.ground, self.blocker(alpha)), (light or self.light(),))
        return s.render(scene, self.camera, 64, 48, ambient=.1, **kwargs)

    def pixel(self, image, point):
        xy, _ = s.project(self.camera, 64, 48, [point])
        x, y = np.floor(xy[0]).astype(int)
        return image[y, x]

    def ground_at_pixel(self, point):
        xy, _ = s.project(self.camera, 64, 48, [point])
        x, y = np.floor(xy[0]) + .5
        eye, view = s._view_basis(self.camera)
        f = 1 / np.tan(np.deg2rad(self.camera.fov / 2))
        ray = view.T @ np.array(((x / 64 * 2 - 1) * (64 / 48) / f,
                                 (1 - y / 48 * 2) / f, -1))
        return eye - ray * eye[1] / ray[1]

    def test_off_is_bit_identical_and_disabled_modes_skip_rays(self):
        light = replace(self.light(), shadows=False)
        with patch.object(s, '_shadow_visibility', side_effect=AssertionError('unexpected ray')):
            np.testing.assert_array_equal(self.render(light, shadows=False), self.render(light, shadows=True))
            for output, shade in [('depth', False), ('normals', False), ('rgba', True)]:
                np.testing.assert_array_equal(self.render(output=output, shade=shade, shadows=False),
                                              self.render(output=output, shade=shade, shadows=True))

    def test_vertical_directional_shadow_and_full_lambert(self):
        image = self.render()
        np.testing.assert_allclose(self.pixel(image, (0, 0, 0)), (.1, .1, .1, 1), atol=1e-5)
        np.testing.assert_allclose(self.pixel(image, (2, 0, 0)), (1.1, 1.1, 1.1, 1), atol=1e-5)

    def test_tilted_directional_shadow_is_shifted_one_unit(self):
        image = self.render(self.light(s.Vec3(-4, 4, 0)))
        np.testing.assert_allclose(self.pixel(image, (1, 0, 0))[:3], .1, atol=1e-5)
        np.testing.assert_allclose(self.pixel(image, (0, 0, 0))[:3], .1 + 1 / np.sqrt(2), atol=1e-5)

    def test_point_shadow_similar_triangles_and_distance_limit(self):
        light = self.light(s.Vec3(-2, 4, 0), kind='Point')
        image = self.render(light)
        np.testing.assert_allclose(self.pixel(image, (2 / 3, 0, 0))[:3], .1, atol=1e-5)
        point = (2, 0, 0)
        ray = light.position.array() - self.ground_at_pixel(point)
        np.testing.assert_allclose(self.pixel(image, point)[:3], .1 + ray[1] / np.linalg.norm(ray), atol=1e-5)
        # A card beyond the point light must not occlude the ground.
        scene = s.Scene((self.ground, self.blocker(height=5)), (self.light(kind='Point'),))
        on = s.render(scene, self.camera, 64, 48)
        off = s.render(scene, self.camera, 64, 48, shadows=False)
        np.testing.assert_allclose(self.pixel(on, (0, 0, 0)), self.pixel(off, (0, 0, 0)), atol=1e-5)

    def test_material_alpha_transmission(self):
        for alpha, expected in [(1, .1), (.5, .6), (0, 1.1)]:
            with self.subTest(alpha=alpha):
                np.testing.assert_allclose(self.pixel(self.render(alpha=alpha), (0, 0, 0))[:3], expected, atol=1e-5)

    def test_projected_and_textured_blockers_ignore_texture_alpha(self):
        transparent = np.zeros((2, 2, 4), np.float32)
        blocker = self.blocker()
        for hidden in [replace(blocker, texture=transparent),
                       replace(blocker, projection=s.Projection(self.camera, transparent))]:
            scene = s.Scene((self.ground, hidden), (self.light(),))
            image = s.render(scene, self.camera, 64, 48, ambient=.1)
            np.testing.assert_allclose(self.pixel(image, (0, 0, 0))[:3], .1, atol=1e-5)

    def test_multiple_transparent_blockers_multiply_and_are_two_sided(self):
        lower = self.blocker(.5)
        upper = self.blocker(.5, height=2)
        upper = replace(upper, triangles=upper.triangles[:, ::-1])
        scene = s.Scene((self.ground, lower, upper), (self.light(),))
        image = s.render(scene, self.camera, 64, 48, ambient=.1)
        np.testing.assert_allclose(self.pixel(image, (0, 0, 0))[:3], .35, atol=1e-5)

    def test_no_self_shadow_on_card_sphere_cube(self):
        shapes = [s._card(2, 2, (1, 1, 1, 1), s.Transform3D()),
                  s._sphere(1, 32, (1, 1, 1, 1), s.Transform3D()),
                  s._cube(2, (1, 1, 1, 1), s.Transform3D())]
        for shape in shapes:
            scene = s.Scene((shape,), (self.light(s.Vec3(4, 1, 2)),))
            off = s.render(scene, s.Camera(), 64, 48, shadows=False)
            on = s.render(scene, s.Camera(), 64, 48)
            lit = (off[..., 0] > .3) & (off[..., 3] == 1)
            self.assertTrue(lit.any())
            np.testing.assert_allclose(on[lit], off[lit], atol=1e-5)
            dark = (off[..., 0] == 0) & (off[..., 3] == 1)
            np.testing.assert_array_equal(on[dark], off[dark])
            if shape is shapes[1]:
                self.assertTrue(dark.any())

    def test_two_lights_add_independent_contributions(self):
        shadowed = self.light(color=(1, .5, .25), intensity=2)
        unshadowed = replace(
            self.light(s.Vec3(-4, 4, 0), color=(.2, .4, .8), intensity=3), shadows=False)
        scene = s.Scene((self.ground, self.blocker(.5)), (shadowed, unshadowed))
        image = s.render(scene, self.camera, 64, 48, ambient=.1)
        expected = .1 + np.array(shadowed.color) * 2 * .5 + np.array(unshadowed.color) * 3 / np.sqrt(2)
        np.testing.assert_allclose(self.pixel(image, (0, 0, 0))[:3], expected, atol=1e-5)

    def test_supersampling_matches_per_sample_render(self):
        scene = s.Scene((self.ground, self.blocker()), (self.light(s.Vec3(-4, 4, 0)),))
        expected = s.render(scene, self.camera, 128, 96).reshape(48, 2, 64, 2, 4).mean(axis=(1, 3))
        np.testing.assert_array_equal(s.render(scene, self.camera, 64, 48, samples=2), expected)

    def test_budget_and_cancellation(self):
        # BVH estimate includes one N log N build and ray cost, before buffers.
        count = 10000
        rays = 64 * 48 * 4 * 2
        expected = (rays * s._SHADOW_BVH_COST + count * s._SHADOW_BVH_BUILD_COST) * np.log2(count + 2)
        self.assertAlmostEqual(s._shadow_cost(rays, count), expected)
        self.assertEqual(s._shadow_cost(12, 4), 48)
        sphere = s._sphere(1, 100, (1, 1, 1, 1), s.Transform3D())
        scene = s.Scene((sphere,), (self.light(), self.light()))
        with patch.object(s, 'SHADOW_WORK_BUDGET', expected - 1), patch.object(
                s.np, 'broadcast_to', side_effect=AssertionError('framebuffer allocated')):
            with self.assertRaisesRegex(ValueError, 'estimated ray-triangle-equivalent tests'):
                s.render(scene, self.camera, 64, 48, samples=2)
        with patch.object(s, 'SHADOW_WORK_BUDGET', 1):
            with self.assertRaisesRegex(ValueError, 'Shadow rays exceed the CPU reference budget:.*switch shadows off'):
                self.render()
            self.render(shadows=False)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            self.render(cancel=cancel)
        # Cancellation after tracing has started is checked between triangle chunks.
        cancel.clear()
        original = np.cross
        def cross(*args, **kwargs):
            cancel.set()
            return original(*args, **kwargs)
        with patch.object(s, '_SHADOW_TRIANGLE_CHUNK', 1), patch.object(s.np, 'cross', side_effect=cross):
            with self.assertRaises(Cancelled):
                s._shadow_visibility(np.zeros((1, 3)), np.array([[0, 1, 0]]), self.light(),
                                     np.array([0, 4, 0]), np.array([0, -1, 0]),
                                     np.zeros((2, 3)), np.ones((2, 3)), np.ones((2, 3)),
                                     np.ones(2), .001, cancel)

    def graph(self):
        d = Dispatcher()
        for key, kind, params in [
            ('ground', 'Card3D', dict(card_width=12, card_height=12, rx=-90)),
            ('blocker', 'Card3D', dict(card_width=1, card_height=1, rx=-90, ty=1)),
            ('light', 'Light3D', dict(tx=0, ty=4, tz=0, shadows='on')),
            ('camera', 'Camera3D', dict(tx=4, ty=5, tz=7)),
            ('scene', 'Scene3D', {}), ('render', 'Render3D', dict(width=64, height=48, samples=1))]:
            d.execute(dict(op='create', id=key, type=kind, params=params))
        for slot, source in [('object0', 'ground'), ('object1', 'blocker'), ('object2', 'light')]:
            d.execute(dict(op='connect', id='scene', input=slot, source=source))
        for slot in ['scene', 'camera']:
            d.execute(dict(op='connect', id='render', input=slot, source=slot))
        return d

    def test_graph_and_old_document(self):
        d, e = self.graph(), Evaluator()
        scene = e.evaluate_raster(d.document, 'scene', typed=True)
        self.assertTrue(scene.lights[0].shadows)
        expected = s.render(scene, self.camera, 64, 48, ambient=.1)
        np.testing.assert_array_equal(e.evaluate(d.document, 'render'), expected)
        del d.document['nodes']['light']['params']['shadows']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.json'
            path.write_text(json.dumps(d.document), encoding='utf-8')
            loaded = load_document(path)
        self.assertEqual(loaded['nodes']['light']['params']['shadows'], 'off')
        np.testing.assert_array_equal(Evaluator().evaluate(loaded, 'render'),
                                      s.render(scene, self.camera, 64, 48, ambient=.1, shadows=False))

    def test_shadowed_auto_fallback_and_gpu_unavailable(self):
        d = self.graph()
        expected = Evaluator().evaluate(d.document, 'render')
        with patch.dict(sys.modules, {'wgpu': None}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
            for probe in (False, None):
                context = patch.object(gpu3d, 'available', return_value=False) if probe is False else patch.dict(sys.modules, {'wgpu': None})
                with context:
                    d.execute(dict(op='set', id='render', param='render_backend', value='auto'))
                    np.testing.assert_array_equal(Evaluator().evaluate(d.document, 'render'), expected)
                    d.execute(dict(op='set', id='render', param='render_backend', value='gpu'))
                    with self.assertRaisesRegex(ValueError, 'GPU Render3D unavailable'):
                        Evaluator().evaluate(d.document, 'render')

    def test_viewport_disables_shadows(self):
        app = QApplication.instance() or QApplication([])
        widget = Viewport3D()
        widget.resize(64, 48)
        widget.set_document(self.graph().document)
        with patch.object(s, 'render', wraps=s.render) as render, patch.object(
                s, '_shadow_visibility', side_effect=AssertionError('viewport shadow')):
            widget.grab()
            self.assertTrue(render.called)
            self.assertTrue(all(call.kwargs.get('shadows') is False for call in render.call_args_list))
        widget.close()

    def test_shadow_knob(self):
        group = next(g for g in knob_layout('Light3D') if 'shadows' in g.params)
        self.assertEqual((group.kind, group.label), ('enum', 'Shadows'))
        self.assertEqual(resolve_kind('Light3D', 'shadows', 'off'), 'enum')


if __name__ == '__main__':
    unittest.main()
