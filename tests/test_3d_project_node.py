"""Project3D's typed graph, cache, animation and viewport contracts."""
import copy
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.viewport3d import Viewport3D
from nodebased import scene3d as s

APP = QApplication.instance() or QApplication([])


class ProjectNodeTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        self.e = Evaluator()
        for key, kind, params in (
            ('image', 'Checker', dict(width=32, height=32, size=16)),
            ('projector', 'Camera3D', dict(tz=4, focal=9.336)),
            ('camera', 'Camera3D', dict(tz=4, focal=9.336)),
            ('card', 'Card3D', dict(card_width=8, card_height=8, red=1, green=1, blue=1)),
            ('project', 'Project3D', {}), ('scene', 'Scene3D', {}),
            ('render', 'Render3D', dict(width=64, height=64, samples=1))):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        for key, slot, source in (('project', 'image', 'image'), ('project', 'camera', 'projector'),
                                  ('project', 'geometry', 'card'), ('scene', 'object0', 'project'),
                                  ('render', 'scene', 'scene'), ('render', 'camera', 'camera')):
            self.connect(key, slot, source)

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def set(self, key, param, value):
        self.d.execute(dict(op='set', id=key, param=param, value=value))

    def value(self, target='project', frame=1):
        return self.e.evaluate_raster(self.d.document, target, typed=True, frame=frame)

    def render(self, frame=1):
        return self.e.evaluate(self.d.document, 'render', frame=frame)

    def test_pixels_match_image_and_stick_with_second_render_camera(self):
        texture = self.e.evaluate(self.d.document, 'image')
        image = self.render()
        for y in (16, 48):
            for x in (16, 48):
                np.testing.assert_allclose(image[y, x], texture[y // 2, x // 2])
        self.set('camera', 'tx', 3)
        moved = self.render()
        points = np.array([(-2, 2, 0), (2, 2, 0), (-2, -2, 0), (2, -2, 0)], np.float32)
        xy, _ = s.project(self.value('camera'), 64, 64, points)
        for (x, y), expected in zip(xy.astype(int), texture[[8, 8, 24, 24], [8, 24, 8, 24]]):
            np.testing.assert_allclose(moved[y, x], expected)

    def test_all_dependencies_invalidate_render_digest(self):
        for key, param, value in [('image', 'size', 8), ('projector', 'tx', 2),
                                  ('card', 'tx', 1), ('project', 'project_outside', 'clamp')]:
            with self.subTest(key=key):
                before = self.render().copy()
                misses = self.e.misses
                self.set(key, param, value)
                after = self.render()
                self.assertGreater(self.e.misses, misses)
                self.assertFalse(np.array_equal(before, after))

    def test_typed_rejections_are_atomic(self):
        for slot, sources in [('image', ['projector', 'card', 'scene']),
                              ('camera', ['image', 'card', 'scene']),
                              ('geometry', ['image', 'projector'])]:
            for source in sources:
                before = copy.deepcopy(self.d.document)
                with self.assertRaises(ValueError):
                    self.connect('project', slot, source)
                self.assertEqual(before, self.d.document)

    def test_missing_required_inputs(self):
        for slot in ('image', 'camera', 'geometry'):
            self.d.execute(dict(op='connect', id='project', input=slot, source=None))
            with self.assertRaisesRegex(ValueError, 'connect required input'):
                self.render()
            self.d.execute(dict(op='undo'))

    def test_disabled_passes_geometry_or_scene_without_projection(self):
        self.d.execute(dict(op='disable', id='project', value=True))
        # Bypass must not evaluate the disconnected image or camera.
        for slot in ('image', 'camera'):
            self.d.execute(dict(op='connect', id='project', input=slot, source=None))
        value = self.value()
        self.assertIsInstance(value, s.Scene)
        self.assertIsNone(value.geometries[0].projection)
        self.d.execute(dict(op='create', id='group', type='Scene3D', params={'tx': 1}))
        self.connect('group', 'object0', 'card')
        self.connect('project', 'geometry', 'group')
        self.assertIsNone(self.value().geometries[0].projection)
        np.testing.assert_array_equal(self.value().geometries[0].world_matrix()[:3, 3], (1, 0, 0))
        self.connect('project', 'image', 'image')
        self.connect('project', 'camera', 'projector')
        self.d.execute(dict(op='disable', id='project', value=False))
        self.assertIsNotNone(self.value().geometries[0].projection)

    def test_connect_disconnect_undo_redo(self):
        connected = copy.deepcopy(self.d.document)
        self.d.execute(dict(op='connect', id='project', input='geometry', source=None))
        disconnected = copy.deepcopy(self.d.document)
        self.d.execute(dict(op='undo'))
        self.assertEqual(connected, self.d.document)
        self.d.execute(dict(op='redo'))
        self.assertEqual(disconnected, self.d.document)
        self.connect('project', 'geometry', 'card')
        self.d.execute(dict(op='undo'))
        self.assertEqual(disconnected, self.d.document)
        self.d.execute(dict(op='redo'))
        self.assertEqual(connected, self.d.document)
        self.assertIsNotNone(self.value().geometries[0].projection)

    def test_animated_projector_moves_texture_without_moving_geometry(self):
        self.set('project', 'project_outside', 'clamp')
        self.set('card', 'card_width', 6)
        self.set('card', 'card_height', 6)
        self.d.document['animation']['curves']['projector'] = {'tx': {'keys': [
            {'frame': 1, 'value': 0.0, 'interpolation': 'linear'},
            {'frame': 10, 'value': 2.0, 'interpolation': 'linear'}]}}
        before, after = self.render(1), self.render(10)
        self.assertFalse(np.array_equal(before, after))
        np.testing.assert_array_equal(before[..., 3], after[..., 3])
        np.testing.assert_array_equal(before[:8], after[:8])
        np.testing.assert_array_equal(self.value(frame=1).geometries[0].world_matrix(),
                                      self.value(frame=10).geometries[0].world_matrix())

    def test_viewport_evaluates_projected_scene(self):
        widget = Viewport3D()
        try:
            widget.set_document(self.d.document)
            scene, camera = widget._evaluated()
            self.assertIsNotNone(scene.geometries[0].projection)
            self.assertEqual(camera, self.value('camera'))
            self.assertEqual(widget.status, '')
        finally:
            widget.close()


class ProjectOcclusionNodeTests(ProjectNodeTests):
    def setup_blocker(self):
        self.set('project', 'project_occlusion', 'depth')
        self.set('camera', 'tz', 1)
        self.set('camera', 'focal', 2.5)
        self.d.execute(dict(op='create', id='blocker', type='Card3D',
                            params=dict(card_width=1, card_height=1, tz=2)))
        self.connect('scene', 'object1', 'blocker')

    def test_graph_matches_projection_api(self):
        self.setup_blocker()
        texture = self.e.evaluate(self.d.document, 'image')
        receiver = s.apply_projection(self.value('card'), s.Projection(
            self.value('projector'), texture, occlusion='depth'))
        scene = s.Scene((receiver, self.value('blocker')))
        expected = s.render(scene, self.value('camera'), 64, 64, ambient=.1)
        np.testing.assert_array_equal(self.render(), expected)
        self.assertEqual(expected[32, 32, 3], 0)
        self.assertEqual(expected[32, 48, 3], 1)

    def test_animated_blocker_moves_shadow(self):
        self.setup_blocker()
        self.d.document['animation']['curves']['blocker'] = {'tx': {'keys': [
            {'frame': 1, 'value': -.75, 'interpolation': 'linear'},
            {'frame': 10, 'value': .75, 'interpolation': 'linear'}]}}
        before, after = self.render(1), self.render(10)
        xy, _ = s.project(self.value('camera'), 64, 64, [(-1.5, 0, 0), (1.5, 0, 0)])
        (x, y), (u, v) = xy.astype(int)
        self.assertEqual(before[y, x, 3], 0)
        self.assertEqual(after[y, x, 3], 1)
        self.assertEqual(before[v, u, 3], 1)
        self.assertEqual(after[v, u, 3], 0)

    def test_old_document_loads_and_evaluates_as_off(self):
        import json
        from pathlib import Path
        import tempfile
        from nodebased.core import load_document
        expected = self.render().copy()
        del self.d.document['nodes']['project']['params']['project_occlusion']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.json'
            path.write_text(json.dumps(self.d.document))
            loaded = load_document(path)
        self.assertEqual(loaded['nodes']['project']['params']['project_occlusion'], 'off')
        np.testing.assert_array_equal(self.e.evaluate(loaded, 'render'), expected)
