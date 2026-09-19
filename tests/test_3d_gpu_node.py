"""Render3D backend selection, compatibility and cache contracts."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, scene3d
from nodebased.core import Dispatcher, load_document
from nodebased.imaging import Evaluator, Cancelled


class GPUNodeTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        self.e = Evaluator()
        for key, kind, params in (
            ('card', 'Card3D', dict(red=.2, green=.6, blue=.9)),
            ('camera', 'Camera3D', {}), ('scene', 'Scene3D', {}),
            ('render', 'Render3D', dict(width=64, height=48, samples=1))):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        self.connect('scene', 'object0', 'card')
        self.connect('render', 'scene', 'scene')
        self.connect('render', 'camera', 'camera')

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def backend(self, value):
        self.d.execute(dict(op='set', id='render', param='render_backend', value=value))

    def render(self):
        return self.e.evaluate(self.d.document, 'render')

    def test_cpu_default_is_exact_reference(self):
        self.assertEqual(self.d.document['nodes']['render']['params']['render_backend'], 'cpu')
        scene = self.e.evaluate_raster(self.d.document, 'scene', typed=True)
        camera = self.e.evaluate_raster(self.d.document, 'camera', typed=True)
        with patch.object(gpu3d, 'available', side_effect=AssertionError('CPU must not probe')):
            np.testing.assert_array_equal(self.render(), scene3d.render(scene, camera, 64, 48, ambient=.1, samples=1))

    def test_auto_without_adapter(self):
        expected = self.render().copy()
        self.backend('auto')
        with patch.object(gpu3d, 'available', return_value=False):
            np.testing.assert_array_equal(self.render(), expected)

    def test_optional_dependency(self):
        expected = self.render().copy()
        self.backend('auto')
        with patch.dict(sys.modules, {'wgpu': None}), patch.object(gpu3d, '_states', {}), patch.object(gpu3d, '_errors', {}):
            np.testing.assert_array_equal(self.render(), expected)
            self.backend('gpu')
            with self.assertRaisesRegex(ValueError, 'GPU Render3D unavailable.*wgpu unavailable'):
                self.render()

    def test_gpu_without_adapter(self):
        self.backend('gpu')
        with patch.object(gpu3d, 'available', return_value=False), patch.object(gpu3d, 'describe', return_value='no test adapter'):
            with self.assertRaisesRegex(ValueError, 'GPU Render3D unavailable: no test adapter'):
                self.render()

    def test_projected_fallback_and_required_gpu_error(self):
        self.d.execute(dict(op='create', id='image', type='Checker', params=dict(width=16, height=16)))
        self.d.execute(dict(op='create', id='project', type='Project3D'))
        self.connect('project', 'image', 'image')
        self.connect('project', 'camera', 'camera')
        self.connect('project', 'geometry', 'card')
        self.connect('scene', 'object0', 'project')
        expected = self.render().copy()
        with patch.object(gpu3d, 'available', return_value=True):
            self.backend('auto')
            np.testing.assert_array_equal(self.render(), expected)
            self.backend('gpu')
            with self.assertRaisesRegex(ValueError, 'GPU Render3D unsupported:.*projected'):
                self.render()

    def test_gpu_receives_cancel_event(self):
        self.backend('gpu')
        event = threading.Event()
        def render(*args, cancel=None, **kwargs):
            self.assertIs(cancel, event)
            event.set()
            gpu3d._cancel(cancel)
        with patch.object(gpu3d, 'available', return_value=True), patch.object(gpu3d, 'render', side_effect=render):
            with self.assertRaises(Cancelled):
                self.e.evaluate(self.d.document, 'render', cancel=event)

    def test_old_document_loads_and_renders(self):
        expected = self.render().copy()
        del self.d.document['nodes']['render']['params']['render_backend']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.json'
            path.write_text(json.dumps(self.d.document))
            loaded = load_document(path)
        self.assertEqual(loaded['nodes']['render']['params']['render_backend'], 'cpu')
        np.testing.assert_array_equal(Evaluator().evaluate(loaded, 'render'), expected)

    @unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
    def test_auto_matches_cpu_and_switch_invalidates_cache(self):
        expected = self.render().copy()
        misses = self.e.misses
        self.backend('auto')
        with patch.object(gpu3d, 'render', wraps=gpu3d.render) as render:
            actual = self.render()
            self.assertEqual(render.call_count, 1)
            self.assertEqual(self.e.misses, misses + 1)
            np.testing.assert_array_equal(self.render(), actual)
            self.assertEqual(render.call_count, 1)
        mask = expected[..., 3] == 1
        padded = np.pad(mask, 2)
        interior = np.logical_and.reduce([padded[y:y+48, x:x+64] for y in range(5) for x in range(5)])
        self.assertTrue(interior.any())
        np.testing.assert_allclose(actual[interior], expected[interior], atol=5e-3, rtol=0)
        self.backend('cpu')
        np.testing.assert_array_equal(self.render(), expected)


if __name__ == '__main__':
    unittest.main()
