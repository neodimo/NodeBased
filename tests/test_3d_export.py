"""OBJ geometry export and deterministic animation round trips."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nodebased.core import Dispatcher, INPUT_TYPES, OUTPUT_TYPES, SCHEMA_VERSION, upgrade_document
from nodebased.geoexport import export_obj
from nodebased.imaging import Evaluator
from nodebased import scene3d as s


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.d = Dispatcher()
        self.e = Evaluator()
        self.add('cube', 'Cube3D', cube_size=1, red=0.7, green=0.7, blue=0.7)
        self.add('scene', 'Scene3D')
        self.add('camera', 'Camera3D', tz=7)
        self.add('render', 'Render3D', width=80, height=64, samples=1)
        self.add('write', 'WriteGeo3D')
        self.connect('scene', 'object0', 'cube')
        self.connect('write', 'scene', 'scene')
        self.connect('render', 'scene', 'write')
        self.connect('render', 'camera', 'camera')

    def add(self, key, kind, **params):
        self.d.execute(dict(op='create', id=key, type=kind, params=params))

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def set(self, key, **params):
        for param, value in params.items():
            self.d.execute(dict(op='set', id=key, param=param, value=value))

    def value(self, key='scene', frame=1):
        return self.e.evaluate_raster(self.d.document, key, frame=frame, tier=1, typed=True)

    def render(self, frame=1, document=None):
        return self.e.evaluate(document or self.d.document, 'render', frame=frame)

    def read(self, path):
        if 'read' not in self.d.document['nodes']:
            self.add('read', 'ReadGeo3D', red=0.7, green=0.7, blue=0.7)
        self.set('read', geo_path=str(path))
        return self.value('read')

    def curve(self, key, param, first, last):
        self.d.document['animation']['curves'].setdefault(key, {})[param] = {'interpolation': 'linear', 'keys': [
            {'frame': 1, 'value': first},
            {'frame': 10, 'value': last}]}

    def test_nested_primitives_round_trip_flat_and_lit(self):
        self.set('cube', tx=-1.3, ry=23, rz=11)
        self.add('sphere', 'Sphere3D', segments=20, sphere_radius=0.65,
                 tx=0.9, ry=19, sx=0.8, sy=1.2, red=0.7, green=0.7, blue=0.7)
        self.add('card', 'Card3D', card_width=0.8, card_height=1.1,
                 ty=-1.2, rx=17, rz=-13, red=0.7, green=0.7, blue=0.7)
        self.add('group', 'Scene3D', tx=0.2, ty=0.3, rz=9, sx=1.1)
        self.connect('group', 'object0', 'sphere')
        self.connect('scene', 'object1', 'group')
        self.connect('scene', 'object2', 'card')
        self.add('light', 'Light3D', tx=-2, ty=4, tz=5)
        path = self.root / 'combined.obj'
        for lit in (False, True):
            with self.subTest(lit=lit):
                self.connect('scene', 'object3', 'light' if lit else None)
                original = self.value()
                counts = s.write_obj(original, path)
                imported = self.read(path)
                self.assertEqual(counts, dict(objects=3,
                    vertices=sum(len(g.vertices) for g in original.geometries),
                    triangles=sum(len(g.triangles) for g in original.geometries)))
                self.assertIsNotNone(imported.normals)
                a = self.render()
                b = s.render(s.Scene((imported,), original.lights), self.value('camera'), 80, 64,
                             ambient=0.1)
                self.assertGreater(float(a[..., 3].sum()), 100)
                np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)

    def test_textured_card_uv_round_trip(self):
        self.add('image', 'Checker', width=32, height=32, size=8)
        self.add('card', 'Card3D', ry=27, rz=13, red=0.7, green=0.7, blue=0.7)
        self.connect('card', 'image', 'image')
        self.connect('scene', 'object0', 'card')
        original = self.render()
        path = self.root / 'texture.obj'
        export_obj(self.d.document, 'write', [1], path)
        self.read(path)
        self.connect('read', 'image', 'image')
        self.connect('scene', 'object0', 'read')
        self.assertGreater(len(np.unique(original[..., 0])), 3)
        np.testing.assert_allclose(original, self.render(), atol=1e-5, rtol=0)

    def test_deterministic_text_counts_and_all_index_forms(self):
        base = s._card(1, 1, (1, 1, 1, 1), s.Transform3D())
        normal = np.tile(np.array((0, 0, 1), np.float32), (4, 1))
        geometries = tuple(replace(base, uvs=base.uvs if uv else None,
                                   normals=normal if n else None)
                           for uv, n in ((False, False), (False, True), (True, False), (True, True)))
        path, second = self.root / 'one.obj', self.root / 'two.obj'
        self.assertEqual(s.write_obj(s.Scene(geometries), path),
                         dict(objects=4, vertices=16, triangles=8))
        s.write_obj(s.Scene(geometries), second)
        self.assertEqual(path.read_bytes(), second.read_bytes())
        self.assertNotIn(b'\r', path.read_bytes())
        faces = [line for line in path.read_text(encoding='utf-8').splitlines() if line.startswith('f ')]
        self.assertEqual(faces[::2], ['f 1 2 3', 'f 5//1 6//2 7//3',
                                     'f 9/1 10/2 11/3', 'f 13/5/5 14/6/6 15/7/7'])
        for index, geometry in enumerate(geometries):
            with self.subTest(index=index):
                target = self.root / f'form{index}.obj'
                s.write_obj(s.Scene((geometry,)), target)
                v, t, uv, n = s._load_obj(*s.obj_fingerprint(target))
                np.testing.assert_array_equal(v, geometry.vertices)
                np.testing.assert_array_equal(t, geometry.triangles)
                if geometry.uvs is None:
                    self.assertIsNone(uv)
                else:
                    np.testing.assert_array_equal(uv, geometry.uvs)
                if geometry.normals is None:
                    self.assertIsNone(n)
                else:
                    np.testing.assert_array_equal(n, geometry.normals)
        self.assertEqual(len(s._load_obj(*s.obj_fingerprint(path))[1]), 8)

    def test_animated_geometry_exports_world_positions_and_pixels(self):
        self.curve('cube', 'tx', 0, 1.5)
        pattern = self.root / 'geo.%04d.obj'
        written = export_obj(self.d.document, 'write', [1, 10], pattern, self.e)
        self.assertEqual(written, [str(self.root / 'geo.0001.obj'), str(self.root / 'geo.0010.obj')])
        def positions(path):
            return np.array([list(map(float, line.split()[1:]))
                             for line in Path(path).read_text().splitlines() if line.startswith('v ')])
        np.testing.assert_array_equal(positions(written[1])[:, 0] - positions(written[0])[:, 0], 1.5)
        np.testing.assert_array_equal(positions(written[1])[:, 1:], positions(written[0])[:, 1:])
        original = self.render(10)
        self.read(written[1])
        self.connect('scene', 'object0', 'read')
        np.testing.assert_allclose(original, self.render(10), atol=1e-5, rtol=0)

    def test_animated_camera_json_round_trip_is_deterministic(self):
        for param, first, last in (('tx', 0, 1.2), ('ty', 0, 0.6), ('fov', 45, 60),
                                   ('roll', 0, 25), ('target_x', 0, 0.4),
                                   ('target_y', 0, -0.2), ('target_z', 0, 0.3)):
            self.curve('camera', param, first, last)
        frames = (1, 5, 10)
        images = [self.render(f) for f in frames]
        for i in range(3):
            self.assertGreater(float(images[i][..., 3].sum()), 0)
            for j in range(i):
                self.assertFalse(np.array_equal(images[i], images[j]))
        path = self.root / 'animation.json'
        path.write_text(json.dumps(self.d.document), encoding='utf-8')
        restored = upgrade_document(json.loads(path.read_text(encoding='utf-8')))
        self.assertEqual(restored['version'], SCHEMA_VERSION)
        for frame, expected in zip(frames, images):
            np.testing.assert_array_equal(expected, self.render(frame))
            np.testing.assert_array_equal(expected, Evaluator().evaluate(restored, 'render', frame=frame))

    def test_export_errors_leave_no_files(self):
        for key, frames, path, message in (
            ('write', [1], ' ', 'output path'),
            ('cube', [1], self.root / 'bad.obj', 'WriteGeo3D'),
            ('absent', [1], self.root / 'bad.obj', 'WriteGeo3D'),
            ('write', [1, 10], self.root / 'bad.obj', 'single file')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                export_obj(self.d.document, key, frames, path)
        self.connect('write', 'scene', None)
        with self.assertRaisesRegex(ValueError, 'upstream scene'):
            export_obj(self.d.document, 'write', [1], self.root / 'bad.obj')
        self.assertEqual(list(self.root.iterdir()), [])

    def test_empty_limit_and_atomic_failure(self):
        path = self.root / 'atomic.obj'
        with self.assertRaisesRegex(ValueError, 'empty scene'):
            s.write_obj(s.Scene(), path)
        with patch.object(s, 'MAX_TRIANGLES', 1), self.assertRaisesRegex(ValueError, 'exceeds 1'):
            s.write_obj(self.value(), path)
        self.assertFalse(path.exists())
        path.write_bytes(b'previous content\n')
        with patch.object(s.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(ValueError, 'replace failed'):
                s.write_obj(self.value(), path)
        self.assertEqual(path.read_bytes(), b'previous content\n')
        broken = replace(self.value().geometries[0], triangles=np.array(((0, 1, 999),)))
        with self.assertRaisesRegex(ValueError, 'missing vertex'):
            s.write_obj(s.Scene((broken,)), self.root / 'broken.obj')
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_read_deleted_file_and_rewrite_invalidates_render_cache(self):
        path = self.root / 'changing.obj'
        export_obj(self.d.document, 'write', [1], path)
        self.read(path)
        self.connect('scene', 'object0', 'read')
        before = self.render()
        misses = self.e.misses
        self.set('cube', tx=1.2)
        s.write_obj(s.Scene((self.value('cube'),)), path)
        # Explicit timestamp avoids relying on the host filesystem's clock resolution.
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        after = self.render()
        self.assertFalse(np.array_equal(before, after))
        self.assertGreater(self.e.misses, misses)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'cannot read'):
            self.render()

    def test_typed_passthrough_disabled_and_undo_redo(self):
        self.assertEqual(OUTPUT_TYPES['WriteGeo3D'], 'scene')
        self.assertEqual(INPUT_TYPES['scene'], ('scene',))
        self.add('image', 'Constant')
        before = copy.deepcopy(self.d.document)
        with self.assertRaisesRegex(ValueError, 'expects scene'):
            self.connect('write', 'scene', 'image')
        self.assertEqual(before, self.d.document)
        expected = self.render()
        for disabled in (False, True):
            self.d.document['nodes']['write']['disabled'] = disabled
            self.assertEqual(len(self.value('write').geometries), len(self.value().geometries))
            np.testing.assert_array_equal(expected, self.render())
        before = copy.deepcopy(self.d.document)
        self.add('extra', 'WriteGeo3D')
        created = copy.deepcopy(self.d.document)
        self.connect('extra', 'scene', 'write')
        connected = copy.deepcopy(self.d.document)
        self.d.execute(dict(op='undo'))
        self.assertEqual(created, self.d.document)
        self.d.execute(dict(op='undo'))
        self.assertEqual(before, self.d.document)
        self.d.execute(dict(op='redo'))
        self.assertEqual(created, self.d.document)
        self.d.execute(dict(op='redo'))
        self.assertEqual(connected, self.d.document)
        self.assertEqual(len(self.value('extra').geometries), 1)
