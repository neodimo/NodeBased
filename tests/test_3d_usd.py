"""USD fixtures authored with the optional bindings, without a Qt dependency."""
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import scene3d as s, usdio


class DegradationTests(unittest.TestCase):
    def test_missing_optional_dependency(self):
        usdio.available.cache_clear()
        self.addCleanup(usdio.available.cache_clear)
        with patch.dict(sys.modules, {'pxr': None}):
            self.assertFalse(usdio.available())
            for call in (usdio.require, lambda: usdio.load_scene('absent.usda', 1),
                         lambda: usdio.load_camera('absent.usda', 1),
                         lambda: usdio.fingerprint('absent.usda'),
                         lambda: usdio.write_usd(s.Scene(), 'absent.usda')):
                with self.assertRaisesRegex(RuntimeError, "USD support needs the optional 'usd-core' package") as caught:
                    call()
                self.assertEqual(str(caught.exception),
                    "USD support needs the optional 'usd-core' package: pip install nodebased[usd]")


@unittest.skipUnless(usdio.available(), 'usd-core not installed')
class USDTests(unittest.TestCase):
    def setUp(self):
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
        self.Gf, self.Sdf, self.Usd, self.U, self.Vt = Gf, Sdf, Usd, UsdGeom, Vt
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def stage(self, name='scene.usda'):
        path = self.root / name
        return self.Usd.Stage.CreateNew(str(path)), path

    def mesh(self, stage, path='/World/mesh', points=None, counts=None, indices=None):
        mesh = self.U.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(points if points is not None else [(-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0)])
        mesh.CreateFaceVertexCountsAttr(counts if counts is not None else [4])
        mesh.CreateFaceVertexIndicesAttr(indices if indices is not None else [0,1,2,3])
        return mesh

    def save(self, stage):
        stage.GetRootLayer().Save()

    def test_cube_face_varying_pixels_and_indexed_attributes(self):
        stage, path = self.stage()
        cube = s._cube(2, (0.3, 0.6, 0.9, 1), s.Transform3D())
        # Eight shared positions, 24 independent UV/normal corners.
        positions, ids = np.unique(cube.vertices, axis=0, return_inverse=True)
        mesh = self.mesh(stage, points=positions.tolist(), counts=[4]*6, indices=ids.tolist())
        uv = self.U.PrimvarsAPI(mesh).CreatePrimvar('st', self.Sdf.ValueTypeNames.TexCoord2fArray,
                                                 self.U.Tokens.faceVarying)
        uv.Set([(0,0), (1,0), (1,1), (0,1)])
        uv.SetIndices([0,1,2,3]*6)
        normals = []
        for face in cube.vertices.reshape(6,4,3):
            normal = np.cross(face[1]-face[0], face[2]-face[0])
            normals.append((normal / np.linalg.norm(normal)).tolist())
        pn = self.U.PrimvarsAPI(mesh).CreatePrimvar('normals', self.Sdf.ValueTypeNames.Normal3fArray,
                                                 self.U.Tokens.faceVarying)
        pn.Set(normals)
        pn.SetIndices(np.repeat(np.arange(6), 4).tolist())
        mesh.CreateDisplayColorPrimvar().Set([cube.color[:3]])
        self.save(stage)
        loaded = usdio.load_scene(path, 1)
        geometry = loaded.geometries[0]
        self.assertEqual(len(geometry.vertices), 24)
        np.testing.assert_array_equal(geometry.vertices, cube.vertices)
        np.testing.assert_array_equal(geometry.uvs, cube.uvs)
        np.testing.assert_array_equal(geometry.normals, np.repeat(normals, 4, axis=0))
        camera = s.Camera(s.Transform3D(position=s.Vec3(3,2,6)))
        for lights in ((), (s.Light(),)):
            a = s.render(s.Scene((cube,), lights), camera, 80, 64)
            b = s.render(replace(loaded, lights=lights), camera, 80, 64)
            self.assertGreater(np.count_nonzero(b[...,3]), 300)
            self.assertLess(np.count_nonzero(b[...,3]), 2000)
            np.testing.assert_allclose(a, b, atol=1e-6)
        image = s.render(loaded, s.Camera(), 64, 64)
        np.testing.assert_allclose(image[32,32], cube.color, atol=1e-6)

    def test_polygons_holes_degeneracy_orientation_and_bad_indices(self):
        stage, path = self.stage()
        mesh = self.mesh(stage, points=[(0,0,0),(2,0,0),(3,1,0),(1,2,0),(0,1,0)],
                         counts=[4,5,3,2], indices=[0,1,2,3, 0,1,2,3,4, 0,0,1, 0,1])
        self.save(stage)
        self.assertEqual(len(usdio.load_scene(path, 1).geometries[0].triangles), 5)
        mesh.CreateHoleIndicesAttr([0])
        mesh.CreateOrientationAttr(self.U.Tokens.leftHanded)
        self.save(stage)
        g = usdio.load_scene(path, 1).geometries[0]
        self.assertEqual(len(g.triangles), 3)
        for tri in g.vertices[g.triangles]:
            self.assertLess(np.cross(tri[1]-tri[0], tri[2]-tri[0])[2], 0)
        mesh.GetFaceVertexIndicesAttr().Set([0,1,2,99, 0,1,2,3,4, 0,0,1, 0,1])
        self.save(stage)
        with self.assertRaisesRegex(ValueError, '/World/mesh.*index out of range'):
            usdio.load_scene(path, 1)

    def test_hierarchy_transform_and_inverse_transpose(self):
        stage, path = self.stage()
        parent = self.U.Xform.Define(stage, '/World')
        parent.AddTranslateOp().Set((3,4,5))
        parent.AddRotateZOp().Set(90)
        mesh = self.mesh(stage)
        mesh.AddScaleOp().Set((2,3,4))
        normal = np.array((1,1,1), dtype=float) / math.sqrt(3)
        mesh.CreateNormalsAttr([normal.tolist()]*4)
        mesh.SetNormalsInterpolation(self.U.Tokens.vertex)
        self.save(stage)
        g = usdio.load_scene(path, 1).geometries[0]
        np.testing.assert_allclose(g.vertices, [(6,2,5),(6,6,5),(0,6,5),(0,2,5)], atol=1e-6)
        expected = np.array((-1/3,1/2,1/4))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(g.normals, np.tile(expected, (4,1)), atol=1e-6)
        np.testing.assert_array_equal(g.world_matrix(), np.eye(4))

    def test_animation_visibility_purpose_and_root(self):
        stage, path = self.stage()
        group = self.U.Xform.Define(stage, '/World')
        translate = group.AddTranslateOp()
        translate.Set((0,0,0), 1)
        translate.Set((9,18,-9), 10)
        self.mesh(stage)
        self.U.Xform.Define(stage, '/hidden')
        self.mesh(stage, '/hidden/mesh')
        self.U.Imageable(stage.GetPrimAtPath('/hidden')).CreateVisibilityAttr(self.U.Tokens.invisible)
        self.U.Xform.Define(stage, '/guide')
        self.mesh(stage, '/guide/mesh')
        self.U.Imageable(stage.GetPrimAtPath('/guide')).CreatePurposeAttr(self.U.Tokens.guide)
        self.save(stage)
        first = usdio.load_scene(path, 1)
        self.assertEqual(len(first.geometries), 1)
        for frame in (5.5,10):
            vertices = usdio.load_scene(path, frame).geometries[0].vertices
            np.testing.assert_allclose(vertices-first.geometries[0].vertices,
                                       np.tile((frame-1,2*(frame-1),1-frame), (4,1)))
        self.assertEqual(len(usdio.load_scene(path,1,root='/World/mesh').geometries),1)
        self.assertEqual(len(usdio.load_scene(path,1,purposes=('guide',)).geometries),1)
        with self.assertRaisesRegex(ValueError, 'root not found'):
            usdio.load_scene(path,1,root='/absent')

    def test_uniform_normals_and_fallback_texcoords(self):
        stage, path = self.stage()
        mesh = self.mesh(stage)
        mesh.CreateNormalsAttr([(0,0,1)])
        mesh.SetNormalsInterpolation(self.U.Tokens.uniform)
        uv = self.U.PrimvarsAPI(mesh).CreatePrimvar('otherUV', self.Sdf.ValueTypeNames.TexCoord2fArray,
                                                 self.U.Tokens.vertex)
        uv.Set([(0,0),(1,0),(1,1),(0,1)])
        mesh.CreateDisplayColorPrimvar().Set([(0.1,0.2,0.3)])
        mesh.CreateDisplayOpacityPrimvar().Set([0.25])
        self.save(stage)
        geometry = usdio.load_scene(path,1).geometries[0]
        self.assertIsNone(geometry.normals)
        np.testing.assert_array_equal(geometry.uvs, [(0,0),(1,0),(1,1),(0,1)])
        np.testing.assert_allclose(geometry.color, (0.1,0.2,0.3,0.25))

    def test_normals_attribute_face_varying_seams_and_precedence(self):
        stage, path = self.stage()
        mesh = self.mesh(stage, counts=[3,3], indices=[0,1,2,0,2,3])
        mesh.CreateNormalsAttr([(0,0,1)]*3 + [(0,1,1)]*3)
        mesh.SetNormalsInterpolation(self.U.Tokens.faceVarying)
        pv = self.U.PrimvarsAPI(mesh).CreatePrimvar('normals', self.Sdf.ValueTypeNames.Normal3fArray,
                                                 self.U.Tokens.vertex)
        pv.Set([(1,0,0)]*4)
        self.save(stage)
        g = usdio.load_scene(path,1).geometries[0]
        self.assertEqual(len(g.vertices),6)
        np.testing.assert_array_equal(g.vertices, [(-1,-1,0),(1,-1,0),(1,1,0),
                                                  (-1,-1,0),(1,1,0),(-1,1,0)])
        np.testing.assert_allclose(g.normals, [(0,0,1)]*3 + [(0,2**-0.5,2**-0.5)]*3,atol=1e-7)
        self.assertEqual(g.color, (0.8,0.8,0.8,1))
        # A separate uniformly shaded mesh stays flat without discarding smooth normals.
        flat = self.mesh(stage, '/flat')
        flat.CreateNormalsAttr([(0,0,1)])
        flat.SetNormalsInterpolation(self.U.Tokens.uniform)
        self.save(stage)
        scene = usdio.load_scene(path,1)
        self.assertIsNotNone(scene.geometries[0].normals)
        self.assertIsNone(scene.geometries[1].normals)

    def test_sublayer_strength_and_fingerprint(self):
        weak, weak_path = self.stage('weak.usda')
        self.mesh(weak)
        self.save(weak)
        strong, strong_path = self.stage('strong.usda')
        override = self.U.Mesh(strong.OverridePrim('/World/mesh'))
        override.CreatePointsAttr([(-2,-2,0),(2,-2,0),(2,2,0),(-2,2,0)])
        override.AddTranslateOp().Set((3,0,0))
        self.save(strong)
        stage, path = self.stage()
        # Earlier sublayers are stronger; root opinions would beat both sublayers.
        stage.GetRootLayer().subLayerPaths = [strong_path.name, weak_path.name]
        self.save(stage)
        g = usdio.load_scene(path,1).geometries[0]
        np.testing.assert_array_equal(g.vertices, [(1,-2,0),(5,-2,0),(5,2,0),(1,2,0)])
        before = usdio.fingerprint(path)
        identities = before[3:]
        self.assertTrue(any(row[0] == str(strong_path.resolve()) for row in identities))
        with strong_path.open('a') as handle:
            handle.write('\n# fingerprint edit\n')
        stat = strong_path.stat()
        os.utime(strong_path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
        self.assertNotEqual(before, usdio.fingerprint(path))
        self.U.Mesh(stage.OverridePrim('/World/mesh')).CreatePointsAttr(
            [(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)])
        self.save(stage)
        np.testing.assert_array_equal(usdio.load_scene(path,1).geometries[0].vertices,
                                      [(2,-1,0),(4,-1,0),(4,1,0),(2,1,0)])

    def test_reference_variants_and_native_instances(self):
        asset, asset_path = self.stage('asset.usda')
        model = self.U.Xform.Define(asset, '/Model')
        asset.SetDefaultPrim(model.GetPrim())
        mesh = self.mesh(asset, '/Model/mesh')
        variants = model.GetPrim().GetVariantSets().AddVariantSet('shape')
        for name, extent in (('small',1), ('large',2)):
            variants.AddVariant(name)
            variants.SetVariantSelection(name)
            with variants.GetVariantEditContext():
                mesh.GetPointsAttr().Set([(-extent,-extent,0),(extent,-extent,0),
                                          (extent,extent,0),(-extent,extent,0)])
        # Points authored outside the variants would override their opinions.
        mesh.GetPointsAttr().Clear()
        variants.SetVariantSelection('small')
        self.save(asset)
        stage, path = self.stage()
        for name, x in (('A',0),('B',5)):
            ref = self.U.Xform.Define(stage, '/'+name)
            ref.GetPrim().GetReferences().AddReference(asset_path.name)
            ref.GetPrim().GetVariantSets().GetVariantSet('shape').SetVariantSelection('large')
            ref.AddTranslateOp().Set((x,0,0))
            ref.GetPrim().SetInstanceable(True)
        self.save(stage)
        self.assertTrue(stage.GetPrimAtPath('/A').IsInstance())
        scene = usdio.load_scene(path,1)
        self.assertEqual(len(scene.geometries),2)
        np.testing.assert_array_equal(scene.geometries[0].vertices, [(-2,-2,0),(2,-2,0),(2,2,0),(-2,2,0)])
        np.testing.assert_array_equal(scene.geometries[1].vertices-scene.geometries[0].vertices,
                                      np.tile((5,0,0),(4,1)))
        self.assertEqual(len(usdio.load_scene(path,1,root='/B').geometries),1)
        self.assertTrue(any(row[0] == str(asset_path.resolve()) for row in usdio.fingerprint(path)[3:]))

    def test_camera_pinhole_roll_fov_and_scale(self):
        stage, path = self.stage()
        rig = self.U.Xform.Define(stage, '/Rig')
        rig.AddTranslateOp().Set((2,3,5))
        camera = self.U.Camera.Define(stage, '/Rig/camera')
        # Independent orthonormal basis: tilt around X then roll around local Z.
        tilt, roll = np.radians((31,27))
        rx = np.array(((1,0,0),(0,np.cos(tilt),-np.sin(tilt)),(0,np.sin(tilt),np.cos(tilt))))
        rz = np.array(((np.cos(roll),-np.sin(roll),0),(np.sin(roll),np.cos(roll),0),(0,0,1)))
        axes = rx @ rz
        matrix = np.eye(4)
        matrix[:3,:3] = axes
        camera.AddTransformOp().Set(self.Gf.Matrix4d(matrix.T.tolist()))
        camera.CreateVerticalApertureAttr(24)
        camera.CreateFocalLengthAttr(35)
        camera.CreateFocusDistanceAttr(4)
        camera.CreateClippingRangeAttr((0.2,200))
        self.save(stage)
        imported = usdio.load_camera(path,1)
        self.assertAlmostEqual(imported.fov, math.degrees(2*math.atan(24/70)))
        self.assertAlmostEqual(imported.near,0.2)
        self.assertEqual(imported.far,200)
        eye = np.array((2,3,5))
        np.testing.assert_allclose(imported.target.array(), eye-4*axes[:,2], atol=1e-6)
        local = np.array(((0,0,-3),(1,0.5,-4),(-0.4,1,-6),(0.7,-1,-5)))
        world = local @ axes.T + eye
        pixels, depth = s.project(imported, 100,80,world)
        expected = np.column_stack((50+local[:,0]/-local[:,2]*(35/24)*80,
                                    40-local[:,1]/-local[:,2]*(35/24)*80))
        np.testing.assert_allclose(pixels,expected,atol=1e-5)
        np.testing.assert_allclose(depth,-local[:,2],atol=1e-6)
        self.assertGreater(abs(imported.roll),10)
        np.testing.assert_allclose(s._view_basis(imported)[1][1], axes[:,1], atol=1e-6)
        self.assertEqual(usdio.load_camera(path,1,'/Rig/camera'), imported)
        for name in ('/Rig','/missing'):
            with self.assertRaisesRegex(ValueError,'camera not found'):
                usdio.load_camera(path,1,name)
        camera.CreateFocusDistanceAttr(0)
        rig.AddScaleOp().Set((2,2,2))
        self.save(stage)
        normalized = usdio.load_camera(path,1)
        np.testing.assert_allclose(normalized.target.array(), eye-axes[:,2],atol=1e-6)
        rig.GetOrderedXformOps()[-1].Set((1,2,1))
        self.save(stage)
        with self.assertRaisesRegex(ValueError,'non-uniform scale'):
            usdio.load_camera(path,1)

    def test_limits_and_unreadable(self):
        stage,path = self.stage()
        self.mesh(stage)
        self.mesh(stage,'/World/second')
        self.save(stage)
        with patch.object(s,'MAX_TRIANGLES',3), self.assertRaisesRegex(ValueError,'more than 3 triangles'):
            usdio.load_scene(path,1)
        for bad in (self.root/'absent.usda', self.root):
            for fn in (lambda p: usdio.load_scene(p,1),usdio.fingerprint):
                with self.assertRaisesRegex(ValueError,'cannot read'):
                    fn(bad)
        for fn in (lambda p: usdio.load_scene(p,1),usdio.fingerprint):
            with self.assertRaisesRegex(ValueError,'empty path'):
                fn('')
        with patch.object(Path, 'open', side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(ValueError,'cannot read'):
                usdio.fingerprint(path)
        broken = self.root/'broken.usda'
        broken.write_text('this is not USD')
        with self.assertRaisesRegex(ValueError,'cannot read'):
            usdio.load_scene(broken,1)

    def scene_for_export(self):
        cube = s._cube(1.5,(0.3,0.6,0.8,1),s.Transform3D(position=s.Vec3(-1,0,0),
                         rotation=s.Vec3(13,27,5), scale=s.Vec3(1,1.2,0.8)))
        sphere = s._sphere(0.7,12,(0.8,0.2,0.1,1),s.Transform3D(position=s.Vec3(1,0,0),
                             scale=s.Vec3(0.8,1.2,1)))
        parent = s.Transform3D(rotation=s.Vec3(0,0,7)).matrix()
        return s.Scene((replace(cube,parent=parent),replace(sphere,parent=parent)))

    def assert_geometry_roundtrip(self, original, imported):
        self.assertEqual(len(original.geometries),len(imported.geometries))
        for a,b in zip(original.geometries,imported.geometries):
            matrix = a.world_matrix()
            vertices = a.vertices @ matrix[:3,:3].T + matrix[:3,3]
            # Compare triangle corners, allowing the importer to re-order/weld vertices.
            # Sphere poles contain degenerate triangles, omitted by the importer.
            keep = np.linalg.norm(np.cross(vertices[a.triangles[:,1]]-vertices[a.triangles[:,0]],
                                           vertices[a.triangles[:,2]]-vertices[a.triangles[:,0]]),axis=1)>0
            triangles = a.triangles[keep]
            np.testing.assert_allclose(vertices[triangles],b.vertices[b.triangles],atol=1e-6)
            np.testing.assert_allclose(a.uvs[triangles],b.uvs[b.triangles],atol=1e-6)
            if a.normals is not None:
                normals = a.normals @ np.linalg.inv(matrix[:3,:3])
                normals /= np.linalg.norm(normals,axis=1,keepdims=True)
                np.testing.assert_allclose(normals[triangles],b.normals[b.triangles],atol=1e-6)
            else:
                self.assertIsNone(b.normals)
        for lights in ((),(s.Light(),)):
            camera = s.Camera(s.Transform3D(position=s.Vec3(0,1,6)))
            a = s.render(replace(original,lights=lights),camera,80,64)
            b = s.render(replace(imported,lights=lights),camera,80,64)
            self.assertGreater(a[...,3].sum(),100)
            np.testing.assert_allclose(a,b,atol=2e-6)

    def test_usda_usdc_usd_roundtrip(self):
        original = self.scene_for_export()
        for suffix in ('.usda','.usdc','.usd'):
            with self.subTest(suffix=suffix):
                path = self.root/('roundtrip'+suffix)
                usdio.write_usd(original,path)
                self.assert_geometry_roundtrip(original,usdio.load_scene(path,1))
                stage = self.Usd.Stage.Open(str(path))
                self.assertEqual(self.U.GetStageUpAxis(stage),self.U.Tokens.y)
                self.assertEqual(self.U.GetStageMetersPerUnit(stage),1)
                if suffix == '.usdc':
                    self.assertEqual(path.read_bytes()[:8], b'PXR-USDC')

    def test_multiframe_samples_topology_and_atomic_failure(self):
        first = self.scene_for_export()
        delta = s.Transform3D(position=s.Vec3(0.5,0.2,0)).matrix()
        last = s.Scene(tuple(replace(g,parent=delta@g.parent) for g in first.geometries))
        path = self.root/'animated.usdc'
        usdio.write_usd([first,last],path,frames=[1,10])
        self.assert_geometry_roundtrip(first,usdio.load_scene(path,1))
        self.assert_geometry_roundtrip(last,usdio.load_scene(path,10))
        a,b = usdio.load_scene(path,1),usdio.load_scene(path,5.5)
        for ga,gb in zip(a.geometries,b.geometries):
            np.testing.assert_allclose(gb.vertices-ga.vertices,np.tile((0.25,0.1,0),(len(ga.vertices),1)),atol=1e-6)
        stage = self.Usd.Stage.Open(str(path))
        self.assertEqual((stage.GetStartTimeCode(),stage.GetEndTimeCode()),(1,10))
        mesh = self.U.Mesh(stage.GetPrimAtPath('/World/geometry_2'))
        self.assertEqual(mesh.GetNormalsAttr().GetTimeSamples(),[1,10])
        stage = mesh = None
        before = path.read_bytes()
        broken = replace(last, geometries=(replace(last.geometries[0],triangles=np.array([[0,1,3]])),last.geometries[1]))
        for changed in (broken,replace(last,geometries=last.geometries[:1])):
            with self.assertRaisesRegex(ValueError,'topology'):
                usdio.write_usd([first,changed],path,[1,10])
        with patch.object(usdio.os,'replace',side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(ValueError,'replace failed'):
                usdio.write_usd(first,path)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(list(self.root.iterdir()),[path])
        with self.assertRaisesRegex(ValueError,'one finite frame'):
            usdio.write_usd([first,last],path,[1])
        with patch.object(s,'MAX_TRIANGLES',1), self.assertRaisesRegex(ValueError,'exceeds 1'):
            usdio.write_usd(first,path)
        with self.assertRaisesRegex(ValueError,'empty scene'):
            usdio.write_usd(s.Scene(),path)

    def test_deforming_positions_and_normals_samples(self):
        card = s._card(2,2,(0.4,0.5,0.6,0.75),s.Transform3D())
        card = replace(card, normals=np.tile((0,0,1),(4,1)).astype(np.float32))
        last = replace(card, vertices=card.vertices + np.array((0,0,2),np.float32),
                       normals=np.tile((0,0.6,0.8),(4,1)).astype(np.float32),
                       uvs=card.uvs*0.5)
        path = self.root/'deform.usda'
        usdio.write_usd([s.Scene((card,)),s.Scene((last,))],path,[1,10])
        midpoint = usdio.load_scene(path,5.5).geometries[0]
        np.testing.assert_allclose(midpoint.vertices,card.vertices+(0,0,1))
        expected = np.array((0,0.3,0.9))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(midpoint.normals,np.tile(expected,(4,1)),atol=1e-6)
        np.testing.assert_allclose(midpoint.uvs,card.uvs*0.75)
        np.testing.assert_allclose(midpoint.color,card.color)
        self.assert_geometry_roundtrip(s.Scene((last,)),usdio.load_scene(path,10))

    def test_invalid_export_preserves_destination_and_cleans_temporary(self):
        original = self.scene_for_export()
        path = self.root/'existing.usda'
        path.write_bytes(b'previous contents')
        first = original.geometries[0]
        for invalid, message in (
                (replace(first,triangles=np.array(((0,1,999),))), 'missing vertex'),
                (replace(first,normals=np.ones((1,3))), 'per-vertex normals'),
                (replace(first,uvs=np.ones((1,2))), 'per-vertex UVs'),
                (replace(first,vertices=first.vertices*np.nan), 'finite geometry')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError,message):
                usdio.write_usd(s.Scene((invalid,)),path)
            self.assertEqual(path.read_bytes(),b'previous contents')
            self.assertEqual(list(self.root.iterdir()),[path])

    def test_usdz_roundtrip(self):
        original = self.scene_for_export()
        path = self.root/'packed.usdz'
        try:
            usdio.write_usd(original,path)
        except ValueError as error:
            if str(error) == '.usdz export not supported':
                self.skipTest(f'USDZ packaging unavailable in this sandbox: {error.__cause__}')
            raise
        self.assert_geometry_roundtrip(original,usdio.load_scene(path,1))
        self.assertEqual(usdio.fingerprint(path)[0],str(path.resolve()))


if __name__ == '__main__':
    unittest.main()
