"""Alembic graph integration uses the bundled reader without optional packages."""
import copy
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import alembicio, scene3d
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

FIXTURE = Path(__file__).parent / 'fixtures' / 'abc' / 'probe.abc'


class AlembicNodeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.d = Dispatcher()
        self.e = Evaluator()
        for key, kind, params in (
            ('read', 'ReadAlembic3D', {'abc_path': str(FIXTURE)}),
            ('camera', 'ReadAlembicCamera3D', {'abc_path': str(FIXTURE)}),
            ('scene', 'Scene3D', {}),
            ('render', 'Render3D', {'width': 128, 'height': 96, 'samples': 1})):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        self.connect('scene', 'object0', 'read')
        self.connect('render', 'scene', 'scene')
        self.connect('render', 'camera', 'camera')

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def set(self, key, param, value):
        self.d.execute(dict(op='set', id=key, param=param, value=value))

    def value(self, key, frame=1):
        return self.e.evaluate_raster(self.d.document, key, frame=frame, typed=True)

    def render(self, frame=1):
        return self.e.evaluate(self.d.document, 'render', frame=frame)

    def test_render_pixels_without_optional_packages(self):
        with patch.dict(sys.modules, {'pxr': None, 'alembic': None, 'imath': None, 'wgpu': None}):
            image = self.render()
        scene = alembicio.load_scene(FIXTURE, 1/24)
        camera = alembicio.load_camera(FIXTURE, 1/24)
        np.testing.assert_array_equal(image, scene3d.render(scene, camera, 128, 96))
        c = math.cos(math.pi/6)
        centroid = np.array([1, 0, -1.5]) @ np.array([[c, 0, -.5], [0, 1, 0], [.5, 0, c]]) + [1, 3, -2]
        pixels, _ = scene3d.project(camera, 128, 96, [centroid])
        x, y = np.floor(pixels[0]).astype(int)
        self.assertEqual(image[y, x, 3], 1)
        self.assertEqual(image[-1, -1, 3], 0)

    def test_animation_and_fps_mapping(self):
        first = self.render(1)
        self.assertFalse(np.array_equal(first, self.render(5)))
        np.testing.assert_array_equal(first, self.render(1))
        self.d.document['time']['fps'] = 12.0
        scene = self.value('read')
        expected = alembicio.load_scene(FIXTURE, 1/12)
        np.testing.assert_array_equal(scene.geometries[0].vertices, expected.geometries[0].vertices)
        # Blender's baked easing at u=1/4: 3*u**2 - 2*u**3 = 5/32.
        # Translation x = 1 + 4*(5/32); apex y = 3 + 2*(5/32).
        self.assertLess(np.linalg.norm(scene.geometries[0].vertices - [1.625, 3, -2], axis=1).min(), 1e-6)
        self.assertAlmostEqual(float(scene.geometries[0].vertices[:, 1].max()), 3.3125)
        self.assertEqual(self.value('camera'), alembicio.load_camera(FIXTURE, 1/12))
        np.testing.assert_array_equal(self.render(), scene3d.render(expected, self.value('camera'), 128, 96))
        self.assertFalse(np.array_equal(first, self.render()))
        self.d.document['time']['fps'] = 24.0
        np.testing.assert_array_equal(first, self.render())

    def test_replacement_invalidates_render_and_unsupported_loads(self):
        path = self.root / 'scene.abc'
        shutil.copyfile(FIXTURE, path)
        self.set('read', 'abc_path', str(path))
        before = self.render()
        misses = self.e.misses
        # Only one archive is bundled. Make a second fixture with an unknown mesh
        # schema, preserving Ogawa byte offsets and all camera/transform data.
        other = self.root / 'unsupported.abc'
        other.write_bytes(FIXTURE.read_bytes().replace(b'AbcGeom_PolyMesh_v1', b'AbcGeom_TestMesh_v1'))
        self.assertTrue(alembicio.unsupported_schemas(other))
        stamp = path.stat()
        shutil.copyfile(other, path)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
        after = self.render()
        self.assertFalse(np.array_equal(before, after))
        self.assertEqual(float(after[..., 3].sum()), 0)
        self.assertGreater(self.e.misses, misses)

    def test_roots_and_disabled(self):
        self.set('read', 'abc_root', '/cam')
        self.assertEqual(self.value('read').geometries, ())
        self.set('read', 'abc_root', '/rig')
        self.assertEqual(len(self.value('read').geometries), 1)
        for key in ('read', 'camera'):
            self.set(key, 'abc_path', '/missing.abc')
            self.d.document['nodes'][key]['disabled'] = True
        self.assertEqual(self.value('read').geometries, ())
        self.assertEqual(self.value('read').lights, ())
        self.assertEqual(self.value('camera'), scene3d.Camera())

    def test_errors_surface_through_evaluator(self):
        path = self.root / 'bad.abc'
        for payload, message in ((None, 'cannot read'), (b'garbage', 'Ogawa|Alembic|corrupt'),
                                 (b'\x89HDF\r\n\x1a\n' + bytes(32), 'HDF5')):
            if payload is not None:
                path.write_bytes(payload)
            for key in ('read', 'camera'):
                self.set(key, 'abc_path', str(path))
                with self.assertRaisesRegex(ValueError, message):
                    self.value(key)
        self.set('camera', 'abc_path', str(FIXTURE))
        for camera in ('/missing', '/rig/probe/probe'):
            self.set('camera', 'abc_camera', camera)
            with self.assertRaisesRegex(ValueError, 'camera not found'):
                self.value('camera')

    def test_types_and_undo_redo(self):
        self.d.execute(dict(op='create', id='image', type='Constant'))
        for kind, target, slot in (('ReadAlembic3D', 'scene', 'object1'),
                                   ('ReadAlembicCamera3D', 'render', 'camera')):
            before = copy.deepcopy(self.d.document)
            self.d.execute(dict(op='create', id=kind, type=kind))
            created = copy.deepcopy(self.d.document)
            self.connect(target, slot, kind)
            connected = copy.deepcopy(self.d.document)
            for op, expected in [('undo', created), ('undo', before), ('redo', created), ('redo', connected)]:
                self.d.execute(dict(op=op))
                self.assertEqual(self.d.document, expected)
        for target, slot, source in [('read', 'image', 'image'), ('camera', 'image', 'image'),
                                     ('render', 'camera', 'read')]:
            before = copy.deepcopy(self.d.document)
            with self.assertRaises(ValueError):
                self.connect(target, slot, source)
            self.assertEqual(self.d.document, before)


if __name__ == '__main__':
    unittest.main()
