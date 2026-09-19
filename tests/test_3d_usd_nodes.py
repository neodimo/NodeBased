"""USD graph integration, cache identity, export and optional-dependency contracts."""
import copy
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import usdio
from nodebased.core import Dispatcher, load_document
from nodebased.geoexport import export_obj
from nodebased.imaging import Evaluator


class GraphFixture:
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.d = Dispatcher()
        self.e = Evaluator()
        for key, kind, params in (
            ('cube', 'Cube3D', dict(cube_size=1)), ('scene', 'Scene3D', {}),
            ('camera', 'Camera3D', dict(tz=7)),
            ('render', 'Render3D', dict(width=80, height=64, samples=1)),
            ('write', 'WriteGeo3D', {})):
            self.add(key, kind, **params)
        self.connect('scene', 'object0', 'cube')
        self.connect('render', 'scene', 'scene')
        self.connect('render', 'camera', 'camera')
        self.connect('write', 'scene', 'scene')

    def add(self, key, kind, **params):
        self.d.execute(dict(op='create', id=key, type=kind, params=params))

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def value(self, key, frame=1):
        return self.e.evaluate_raster(self.d.document, key, frame=frame, typed=True)

    def render(self, frame=1):
        return self.e.evaluate(self.d.document, 'render', frame=frame)

    def read(self, path):
        self.add('read', 'ReadUSD3D', usd_path=str(path))
        self.connect('scene', 'object0', 'read')

    def animate(self):
        self.d.document['animation']['curves']['cube'] = {'tx': {
            'interpolation': 'linear', 'keys': [dict(frame=1, value=0), dict(frame=10, value=1.5)]}}


class DegradationTests(GraphFixture, unittest.TestCase):
    def test_dependency_errors_through_evaluator(self):
        for kind in ('ReadUSD3D', 'ReadUSDCamera3D'):
            self.add(kind, kind, usd_path='absent.usda')
            with patch.object(usdio, 'available', return_value=False):
                with self.assertRaisesRegex(ValueError, r'pip install nodebased\[usd\]'):
                    self.value(kind)
            usdio.available.cache_clear()
            try:
                with patch.dict(sys.modules, {'pxr': None}):
                    with self.assertRaisesRegex(ValueError, r'pip install nodebased\[usd\]'):
                        self.value(kind)
            finally:
                usdio.available.cache_clear()
        with patch.object(usdio, 'available', return_value=False):
            with self.assertRaisesRegex(ValueError, r'pip install nodebased\[usd\]'):
                export_obj(self.d.document, 'write', [1], self.root / 'out.usda')

    def test_disabled_empty_scene_without_dependency(self):
        self.add('read', 'ReadUSD3D')
        self.d.document['nodes']['read']['disabled'] = True
        with patch.object(usdio, 'available', return_value=False):
            self.assertEqual(self.value('read').geometries, ())

    def test_typed_rejection_and_undo_redo(self):
        self.add('image', 'Constant')
        before = copy.deepcopy(self.d.document)
        self.add('read', 'ReadUSD3D')
        created = copy.deepcopy(self.d.document)
        self.connect('scene', 'object0', 'read')
        connected = copy.deepcopy(self.d.document)
        for op, expected in [('undo', created), ('undo', before), ('redo', created), ('redo', connected)]:
            self.d.execute(dict(op=op))
            self.assertEqual(self.d.document, expected)
        self.add('camread', 'ReadUSDCamera3D')
        for target, slot, source in [('read', 'image', 'image'), ('camread', 'image', 'image'),
                                     ('render', 'camera', 'read')]:
            before = copy.deepcopy(self.d.document)
            with self.assertRaises(ValueError):
                self.connect(target, slot, source)
            self.assertEqual(self.d.document, before)

    def test_old_document_and_unknown_extension(self):
        import json
        path = self.root / 'old.nbcomp'
        path.write_text(json.dumps(self.d.document))
        self.assertEqual(load_document(path), self.d.document)
        with self.assertRaisesRegex(ValueError, r'\.obj, \.usd, \.usda, \.usdc, \.usdz'):
            export_obj(self.d.document, 'write', [1], self.root / 'bad.ply')


@unittest.skipUnless(usdio.available(), 'usd-core not installed')
class USDNodeTests(GraphFixture, unittest.TestCase):
    def test_static_export_render_round_trip(self):
        from pxr import Usd, UsdGeom
        expected = self.render()
        self.assertGreater(float(expected[..., 3].sum()), 50)
        path = self.root / 'static.usda'
        self.assertEqual(export_obj(self.d.document, 'write', [1], path), [str(path)])
        stage = Usd.Stage.Open(str(path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/World/geometry_1'))
        self.assertEqual(mesh.GetPointsAttr().GetTimeSamples(), [])
        self.read(path)
        np.testing.assert_array_equal(expected, self.render())
        np.testing.assert_array_equal(expected, self.render(10))
        self.d.document['nodes']['read']['disabled'] = True
        self.assertEqual(self.value('read').geometries, ())
        self.assertEqual(float(self.render()[..., 3].sum()), 0)

    def test_sampled_export_and_frame_cache(self):
        from pxr import Usd, UsdGeom
        self.animate()
        expected = {f: self.render(f) for f in (1, 10)}
        self.assertFalse(np.array_equal(expected[1], expected[10]))
        path = self.root / 'animated.usda'
        self.assertEqual(export_obj(self.d.document, 'write', [1, 10], path), [str(path)])
        stage = Usd.Stage.Open(str(path))
        self.assertEqual(UsdGeom.Mesh(stage.GetPrimAtPath('/World/geometry_1')).GetPointsAttr().GetTimeSamples(), [1, 10])
        self.read(path)
        for frame in (1, 10, 1, 10):
            np.testing.assert_array_equal(expected[frame], self.render(frame))

    def test_patterns_and_obj(self):
        self.animate()
        expected = {f: self.render(f) for f in (1, 10)}
        for suffix in ('usda', 'obj'):
            paths = export_obj(self.d.document, 'write', [1, 10], self.root / ('geo.%04d.' + suffix))
            self.assertEqual([Path(p).name for p in paths], [f'geo.{f:04d}.{suffix}' for f in (1, 10)])
            self.assertTrue(all(Path(p).exists() for p in paths))
            if suffix == 'usda':
                for frame, path in zip((1, 10), paths):
                    self.add(f'read{frame}', 'ReadUSD3D', usd_path=path)
                    self.connect('render', 'scene', f'read{frame}')
                    np.testing.assert_array_equal(expected[frame], self.render(frame))

    def test_sublayer_disk_edit_invalidates_render(self):
        from pxr import Usd
        layer = self.root / 'layer.usda'
        usdio.write_usd(self.value('scene'), layer)
        root = self.root / 'root.usda'
        stage = Usd.Stage.CreateNew(str(root))
        stage.GetRootLayer().subLayerPaths = ['layer.usda']
        stage.GetRootLayer().Save()
        del stage
        self.read(root)
        before = self.render()
        misses = self.e.misses
        self.d.execute(dict(op='set', id='cube', param='tx', value=1.5))
        from nodebased.scene3d import Scene
        usdio.write_usd(Scene((self.value('cube'),)), layer)
        stat = layer.stat()
        os.utime(layer, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        self.assertFalse(np.array_equal(before, self.render()))
        self.assertGreater(self.e.misses, misses)

    def test_root_filter_and_missing_file(self):
        from pxr import Usd, UsdGeom
        path = self.root / 'roots.usda'
        stage = Usd.Stage.CreateNew(str(path))
        for root, offset in (('/Left', -1), ('/Right', 1)):
            mesh = UsdGeom.Mesh.Define(stage, root + '/mesh')
            mesh.CreatePointsAttr([(offset-0.4,-0.4,0), (offset+0.4,-0.4,0), (offset,0.4,0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0,1,2])
        stage.GetRootLayer().Save()
        self.read(path)
        self.assertEqual(len(self.value('read').geometries), 2)
        self.d.execute(dict(op='set', id='read', param='usd_root', value='/Left'))
        self.assertEqual(len(self.value('read').geometries), 1)
        self.assertLess(self.value('read').geometries[0].vertices[:, 0].max(), 0)
        for kind in ('ReadUSD3D', 'ReadUSDCamera3D'):
            self.add(kind, kind, usd_path=str(self.root / 'missing.usda'))
            with self.assertRaisesRegex(ValueError, 'cannot read'):
                self.value(kind)

    def test_camera_drives_render_and_changes_with_frame(self):
        from pxr import Usd, UsdGeom
        expected = self.render()
        path = self.root / 'camera.usda'
        stage = Usd.Stage.CreateNew(str(path))
        camera = UsdGeom.Camera.Define(stage, '/Camera')
        camera.CreateFocalLengthAttr(50)
        camera.CreateVerticalApertureAttr(100 * math.tan(math.radians(45 / 2)))
        camera.CreateClippingRangeAttr((0.1, 1000))
        camera.CreateFocusDistanceAttr(7)
        move = camera.AddTranslateOp()
        move.Set((0,0,7), 1)
        move.Set((1.5,0,7), 10)
        stage.GetRootLayer().Save()
        self.add('usd_camera', 'ReadUSDCamera3D', usd_path=str(path), usd_camera='/Camera')
        self.connect('render', 'camera', 'usd_camera')
        np.testing.assert_allclose(expected, self.render(), atol=1e-6)
        later = self.render(10)
        self.assertFalse(np.array_equal(expected, later))
        np.testing.assert_array_equal(later, self.render(10))

    def test_topology_error_surfaces(self):
        self.add('sphere', 'Sphere3D')
        self.connect('scene', 'object0', 'sphere')
        self.d.document['animation']['curves']['sphere'] = {'segments': {
            'interpolation': 'linear', 'keys': [dict(frame=1, value=8), dict(frame=10, value=16)]}}
        with self.assertRaisesRegex(ValueError, 'topology'):
            export_obj(self.d.document, 'write', [1, 10], self.root / 'topology.usda')
