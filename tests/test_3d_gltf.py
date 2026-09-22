"""glTF 2.0 reader: file structure, accessors, hierarchy, materials, textures and the ReadGLTF3D node."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nodebased import gltfio, scene3d as s
from nodebased.color import to_working
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.gltf_fixture import (Builder, QUAD, QUAD_INDICES, QUAD_NORMALS, QUAD_UVS, png_bytes,
                                quad_builder)


def working(r, g, b, space='sRGB'):
    return to_working(np.array([[[r, g, b, 1.0]]], np.float32), space)[0, 0, :3]


class Folder:
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def path(self, name):
        return self.root / name

    def glb(self, builder, name='scene.glb'):
        return builder.write_glb(self.path(name))


class StructureTests(Folder, unittest.TestCase):
    def test_glb_quad_with_transform(self):
        b = Builder()
        half = np.sin(np.pi / 4)
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, QUAD_NORMALS, QUAD_UVS)),
               translation=(1, 2, 3), rotation=(0, half, 0, half), scale=(2, 2, 2))
        scene = gltfio.load_scene(self.glb(b))
        self.assertEqual(len(scene.geometries), 1)
        g = scene.geometries[0]
        # Rotate 90 degrees about Y, double, then move: (-1, -1, 0) -> (0, -2, 2) -> (1, 0, 5).
        np.testing.assert_allclose(g.vertices[0], (1, 0, 5), atol=1e-6)
        np.testing.assert_allclose(g.vertices[2], (1, 4, 1), atol=1e-6)
        np.testing.assert_allclose(g.normals, [(1, 0, 0)] * 4, atol=1e-6)
        self.assertEqual(g.triangles.tolist(), [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(g.triangles.dtype, np.int32)
        # glTF's V runs downwards; scene3d's runs upwards.
        self.assertEqual(g.uvs.tolist(), [[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertEqual(g.color, (1.0, 1.0, 1.0, 1.0))
        self.assertIsNone(g.texture)
        self.assertEqual(g.transform, s.Transform3D())

    def test_matrix_node_is_column_major(self):
        b = Builder()
        matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 7, 8, 9, 1]  # translation in the last column
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES)), matrix=matrix)
        g = gltfio.load_scene(self.glb(b)).geometries[0]
        np.testing.assert_allclose(g.vertices[0], (6, 7, 9), atol=1e-6)

    def test_gltf_with_external_buffer_and_data_uri(self):
        b = quad_builder(factor=(0.2, 0.4, 0.6, 1.0))
        external = b.write_gltf(self.path('scene.gltf'))
        inline = b.write_gltf(self.path('inline.gltf'), external=None)
        for path in (external, inline):
            with self.subTest(path=path.name):
                scene = gltfio.load_scene(path)
                self.assertEqual(len(scene.geometries), 1)
                np.testing.assert_allclose(scene.geometries[0].color[:3], working(0.2, 0.4, 0.6, 'Linear Rec.709'),
                                           atol=1e-5)
        # The fingerprint of a .gltf covers the buffer it points at.
        before = gltfio.fingerprint(external)
        self.assertIn('buffer.bin', before)
        (self.root / 'buffer.bin').write_bytes(b'\0' * 8)
        self.assertNotEqual(gltfio.fingerprint(external), before)
        with self.assertRaisesRegex(ValueError, 'shorter than its byteLength'):
            gltfio.load_scene(external)
        os.remove(self.root / 'buffer.bin')
        self.assertEqual(gltfio.fingerprint(external)[-1], None)
        with self.assertRaisesRegex(ValueError, 'cannot read buffer.bin'):
            gltfio.load_scene(external)

    def test_bad_files(self):
        cases = {
            'v3.glb': (b'glTF' + b'\x03\x00\x00\x00' + b'\x00' * 8, 'version 3'),
            'short.glb': (b'glTF\x02', 'truncated'),
            'text.glb': (b'nope{', 'not a glTF'),
            'nojson.glb': (b'glTF' + (2).to_bytes(4, 'little') + (20).to_bytes(4, 'little')
                           + (0).to_bytes(4, 'little') + (0x004E4942).to_bytes(4, 'little'), 'no JSON chunk'),
        }
        for name, (data, message) in cases.items():
            with self.subTest(name=name):
                self.path(name).write_bytes(data)
                with self.assertRaisesRegex(ValueError, message):
                    gltfio.load_scene(self.path(name))
        with self.assertRaisesRegex(ValueError, 'choose a glTF'):
            gltfio.fingerprint('')
        with self.assertRaisesRegex(ValueError, 'cannot read'):
            gltfio.fingerprint(self.path('absent.glb'))

    def test_required_extension_is_refused_by_name(self):
        b = quad_builder()
        b.doc['extensionsRequired'] = ['KHR_mesh_quantization', 'KHR_draco_mesh_compression']
        with self.assertRaisesRegex(ValueError, 'KHR_draco_mesh_compression'):
            gltfio.load_scene(self.glb(b))
        b.doc['extensionsRequired'] = ['KHR_mesh_quantization']
        self.assertEqual(len(gltfio.load_scene(self.glb(b, 'ok.glb')).geometries), 1)

    def test_empty_scene_and_unused_meshes(self):
        b = Builder()
        b.mesh(b.primitive(QUAD, QUAD_INDICES))  # no node references it
        self.assertEqual(gltfio.load_scene(self.glb(b)).geometries, ())
        b = Builder()
        b.node(name='empty')
        self.assertEqual(gltfio.load_scene(self.glb(b, 'b.glb')).geometries, ())
        b = Builder()
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES)))
        del b.doc['scenes'], b.doc['scene']  # no scene list: every parentless node is a root
        self.assertEqual(len(gltfio.load_scene(self.glb(b, 'c.glb')).geometries), 1)


class AccessorTests(Folder, unittest.TestCase):
    def test_index_types_and_unindexed(self):
        for dtype in (np.uint8, np.uint16, np.uint32):
            b = Builder()
            b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES.astype(dtype))))
            self.assertEqual(gltfio.load_scene(self.glb(b, f'{dtype.__name__}.glb')).geometries[0].triangles.tolist(),
                             [[0, 1, 2], [0, 2, 3]])
        b = Builder()
        b.node(mesh=b.mesh(b.primitive(QUAD[[0, 1, 2, 0, 2, 3]], None)))
        g = gltfio.load_scene(self.glb(b)).geometries[0]
        self.assertEqual(len(g.vertices), 6)
        self.assertEqual(g.triangles.tolist(), [[0, 1, 2], [3, 4, 5]])

    def test_strip_fan_points_and_lines(self):
        b = Builder()
        strip = b.primitive(QUAD, np.array([0, 1, 3, 2], np.uint16), mode=5)
        fan = b.primitive(QUAD, None, mode=6)
        points = b.primitive(QUAD, None, mode=0)
        lines = b.primitive(QUAD, np.array([0, 1, 1, 2], np.uint16), mode=1)
        b.node(mesh=b.mesh(strip, fan, points, lines))
        scene = gltfio.load_scene(self.glb(b))
        self.assertEqual(len(scene.geometries), 2)
        self.assertEqual(scene.geometries[0].triangles.tolist(), [[0, 1, 3], [3, 1, 2]])
        self.assertEqual(scene.geometries[1].triangles.tolist(), [[0, 1, 2], [0, 2, 3]])

    def test_interleaved_and_normalized_attributes(self):
        # One buffer view holding position (float32 x3) and uv (uint16 x2, normalized) per vertex,
        # padded to a 20 byte stride, as KHR_mesh_quantization exporters write.
        b = Builder()
        rows = bytearray()
        for (x, y, z), (u, v) in zip(QUAD, QUAD_UVS):
            rows += np.array((x, y, z), np.float32).tobytes()
            rows += np.array((u * 65535, v * 65535), np.uint16).tobytes()
            rows += b'\0' * 4
        view = b.view(bytes(rows), stride=20)
        positions = b.accessor(QUAD, view=view, offset=0)
        uvs = b.accessor(np.zeros((4, 2), np.uint16), view=view, offset=12, normalized=True)
        b.node(mesh=b.mesh(b.primitive(None, QUAD_INDICES, uvs=uvs, position_accessor=positions)))
        g = gltfio.load_scene(self.glb(b)).geometries[0]
        np.testing.assert_allclose(g.vertices, QUAD)
        np.testing.assert_allclose(g.uvs, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=2e-5)

    def test_accessor_errors(self):
        def build(mutate):
            b = quad_builder()
            mutate(b)
            return self.glb(b, f'{len(os.listdir(self.root))}.glb')

        def overrun(b):
            b.doc['accessors'][0]['count'] = 400

        def sparse(b):
            b.doc['accessors'][0]['sparse'] = {'count': 1}

        def out_of_range(b):
            view = b.doc['bufferViews'][b.doc['accessors'][b.doc['meshes'][0]['primitives'][0]['indices']]['bufferView']]
            b.blob[view['byteOffset']:view['byteOffset'] + 2] = (9).to_bytes(2, 'little')

        def nan(b):
            b.blob[b.doc['bufferViews'][0]['byteOffset']:b.doc['bufferViews'][0]['byteOffset'] + 4] = np.float32('nan').tobytes()

        def no_position(b):
            del b.doc['meshes'][0]['primitives'][0]['attributes']['POSITION']

        def bad_stride(b):
            b.doc['bufferViews'][0]['byteStride'] = 8

        for mutate, message in ((overrun, 'overruns'), (sparse, 'sparse'), (out_of_range, 'missing vertex'),
                                (nan, 'non-finite'), (no_position, 'no POSITION'), (bad_stride, 'byteStride')):
            with self.subTest(case=mutate.__name__):
                with self.assertRaisesRegex(ValueError, message):
                    gltfio.load_scene(build(mutate))

    def test_degenerate_triangles_dropped_and_triangle_cap(self):
        b = Builder()
        b.node(mesh=b.mesh(b.primitive(QUAD, np.array([0, 1, 2, 1, 1, 2], np.uint16))))
        self.assertEqual(gltfio.load_scene(self.glb(b)).geometries[0].triangles.tolist(), [[0, 1, 2]])
        b = Builder()
        b.node(mesh=b.mesh(b.primitive(QUAD, np.array([1, 1, 2, 3, 3, 3], np.uint16))))
        self.assertEqual(gltfio.load_scene(self.glb(b, 'none.glb')).geometries, ())
        with patch.object(s, 'MAX_TRIANGLES', 1):
            with self.assertRaisesRegex(ValueError, 'more than 1 triangles'):
                gltfio.load_scene(self.glb(quad_builder(), 'cap.glb'))


class HierarchyTests(Folder, unittest.TestCase):
    def build(self):
        b = Builder()
        mesh = b.mesh(b.primitive(QUAD, QUAD_INDICES, QUAD_NORMALS))
        leaf = b.node(mesh=mesh, name='leaf', translation=(0, 0, 1), root=False)
        b.node(children=[leaf], name='parent', translation=(5, 0, 0))
        b.node(mesh=mesh, name='mirror', scale=(-1, 1, 1))
        return self.glb(b)

    def test_parents_compose_and_a_mesh_can_be_instanced(self):
        scene = gltfio.load_scene(self.build())
        self.assertEqual(len(scene.geometries), 2)
        np.testing.assert_allclose(scene.geometries[0].vertices[0], (4, -1, 1))
        np.testing.assert_allclose(scene.geometries[1].vertices[0], (1, -1, 0))

    def test_mirroring_flips_winding_and_normals_survive(self):
        scene = gltfio.load_scene(self.build())
        self.assertEqual(scene.geometries[0].triangles[0].tolist(), [0, 1, 2])
        self.assertEqual(scene.geometries[1].triangles[0].tolist(), [0, 2, 1])
        np.testing.assert_allclose(scene.geometries[1].normals[0], (0, 0, 1), atol=1e-6)

    def test_root_filter(self):
        path = self.build()
        scene = gltfio.load_scene(path, 'leaf')
        self.assertEqual(len(scene.geometries), 1)
        np.testing.assert_allclose(scene.geometries[0].vertices[0], (4, -1, 1))
        self.assertEqual(len(gltfio.load_scene(path, 'parent').geometries), 1)
        with self.assertRaisesRegex(ValueError, "no node named 'absent'"):
            gltfio.load_scene(path, 'absent')

    def test_cycle_is_an_error(self):
        b = Builder()
        first = b.node(name='a')
        second = b.node(name='b', children=[first], root=False)
        b.doc['nodes'][first]['children'] = [second]
        with self.assertRaisesRegex(ValueError, 'own ancestor'):
            gltfio.load_scene(self.glb(b))

    def test_singular_transform_with_normals(self):
        b = Builder()
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, QUAD_NORMALS)), scale=(1, 1, 0))
        with self.assertRaisesRegex(ValueError, 'singular'):
            gltfio.load_scene(self.glb(b))


class MaterialTests(Folder, unittest.TestCase):
    def texture(self):
        rgba = np.zeros((2, 4, 4), np.uint8)
        rgba[0] = (255, 0, 0, 255)
        rgba[1] = (0, 0, 255, 128)
        return rgba

    def test_factor_alpha_follows_alpha_mode(self):
        for mode, alpha in ((None, 1.0), ('OPAQUE', 1.0), ('BLEND', 0.25), ('MASK', 0.25)):
            with self.subTest(mode=mode):
                b = quad_builder(factor=(0.5, 0.25, 0.125, 0.25), alpha_mode=mode)
                color = gltfio.load_scene(self.glb(b, f'{mode}.glb')).geometries[0].color
                np.testing.assert_allclose(color[:3], working(0.5, 0.25, 0.125, 'Linear Rec.709'), atol=1e-5)
                self.assertEqual(color[3], alpha)

    def test_embedded_texture_is_premultiplied_working_space_top_row_first(self):
        b = Builder()
        material = b.material(texture=b.image(png_bytes(self.texture())))
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, QUAD_NORMALS, QUAD_UVS, material=material)))
        g = gltfio.load_scene(self.glb(b)).geometries[0]
        self.assertEqual(g.texture.shape, (2, 4, 4))
        self.assertEqual(g.texture.dtype, np.float32)
        np.testing.assert_allclose(g.texture[0, 0], (*working(1, 0, 0), 1.0), atol=1e-5)
        np.testing.assert_allclose(g.texture[1, 0], (*working(0, 0, 1), 1.0), atol=1e-5)  # opaque: alpha forced

    def test_blend_keeps_texture_alpha_and_external_image(self):
        b = Builder()
        self.path('tex.png').write_bytes(png_bytes(self.texture()))
        material = b.material(texture=b.image(None, mime=None, uri='tex.png'), alpha_mode='BLEND')
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, QUAD_NORMALS, QUAD_UVS, material=material)))
        path = b.write_gltf(self.path('scene.gltf'))
        g = gltfio.load_scene(path).geometries[0]
        alpha = 128 / 255
        np.testing.assert_allclose(g.texture[1, 0], (*(working(0, 0, 1) * alpha), alpha), atol=1e-5)
        self.assertIn('tex.png', gltfio.fingerprint(path))
        os.remove(self.path('tex.png'))
        with self.assertRaisesRegex(ValueError, 'image 0: cannot read tex.png'):
            gltfio.load_scene(path)

    def test_texture_without_uvs_is_dropped_and_texcoord_set_is_honoured(self):
        b = Builder()
        material = b.material(texture=b.image(png_bytes(self.texture())))
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, material=material)))
        g = gltfio.load_scene(self.glb(b)).geometries[0]
        self.assertIsNone(g.texture)
        self.assertIsNone(g.uvs)
        b = Builder()
        material = b.material(texture=b.image(png_bytes(self.texture())), texcoord=1)
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, uvs=QUAD_UVS, uvs1=QUAD_UVS * 0.5, material=material)))
        g = gltfio.load_scene(self.glb(b, 'set1.glb')).geometries[0]
        self.assertIsNotNone(g.texture)
        np.testing.assert_allclose(g.uvs[1], (0.5, 0.5))

    def test_specular_glossiness_and_texture_transform(self):
        b = quad_builder(factor=(0.5, 0.5, 0.5, 1.0), glossy=True)
        np.testing.assert_allclose(gltfio.load_scene(self.glb(b)).geometries[0].color[:3],
                                   working(0.5, 0.5, 0.5, 'Linear Rec.709'), atol=1e-5)
        b = Builder()
        material = b.material(texture=b.image(png_bytes(self.texture())),
                              transform={'offset': [0.25, 0.0], 'scale': [2.0, 1.0]})
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, uvs=QUAD_UVS, material=material)))
        g = gltfio.load_scene(self.glb(b, 'transform.glb')).geometries[0]
        np.testing.assert_allclose(g.uvs[1], (2.25, 0.0), atol=1e-6)  # (1, 1) -> scaled, shifted, V flipped

    def test_large_textures_are_box_filtered(self):
        rgba = np.zeros((4, 8, 4), np.uint8)
        rgba[:, :4] = (255, 255, 255, 255)
        rgba[:, 4:] = (0, 0, 0, 255)
        b = Builder()
        material = b.material(texture=b.image(png_bytes(rgba)))
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, uvs=QUAD_UVS, material=material)))
        with patch.object(gltfio, 'MAX_TEXTURE', 4):
            g = gltfio.load_scene(self.glb(b)).geometries[0]
        self.assertEqual(g.texture.shape, (2, 4, 4))
        np.testing.assert_allclose(g.texture[0, 0], (1, 1, 1, 1), atol=1e-4)
        np.testing.assert_allclose(g.texture[0, 3], (0, 0, 0, 1), atol=1e-4)

    def test_undecodable_texture(self):
        b = Builder()
        material = b.material(texture=b.image(b'not a png'))
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, uvs=QUAD_UVS, material=material)))
        with self.assertRaisesRegex(ValueError, 'image 0: cannot decode'):
            gltfio.load_scene(self.glb(b))


class NodeTests(Folder, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.d = Dispatcher()
        self.e = Evaluator()
        for key, kind, params in (('read', 'ReadGLTF3D', {}), ('scene', 'Scene3D', {}),
                                  ('camera', 'Camera3D', dict(tz=3)),
                                  ('render', 'Render3D', dict(width=64, height=64, samples=1))):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        for key, slot, source in (('scene', 'object0', 'read'), ('render', 'scene', 'scene'),
                                  ('render', 'camera', 'camera')):
            self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def set(self, param, value):
        self.d.execute(dict(op='set', id='read', param=param, value=value))

    def textured_quad(self, name='quad.glb'):
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[:4] = (255, 0, 0, 255)
        rgba[4:] = (0, 0, 255, 255)
        b = Builder()
        material = b.material(texture=b.image(png_bytes(rgba)))
        b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, QUAD_NORMALS, QUAD_UVS, material=material)))
        return str(b.write_glb(self.path(name)))

    def test_render_shows_the_texture_the_right_way_up(self):
        self.set('gltf_path', self.textured_quad())
        frame = np.asarray(self.e.evaluate(self.d.document, 'render', frame=1))
        np.testing.assert_allclose(frame[10, 32, :3], working(1, 0, 0), atol=1e-4)
        np.testing.assert_allclose(frame[54, 32, :3], working(0, 0, 1), atol=1e-4)
        value = self.e.evaluate_raster(self.d.document, 'read', frame=1, typed=True)
        self.assertIsInstance(value, s.Scene)
        self.assertEqual(len(value.geometries), 1)

    def test_missing_path_disabled_and_root(self):
        with self.assertRaisesRegex(ValueError, 'ReadGLTF3D: choose a glTF or GLB file'):
            self.e.evaluate(self.d.document, 'render', frame=1)
        self.set('gltf_path', str(self.path('absent.glb')))
        with self.assertRaisesRegex(ValueError, 'ReadGLTF3D: cannot read'):
            self.e.evaluate(self.d.document, 'render', frame=1)
        self.d.document['nodes']['read']['disabled'] = True
        self.assertEqual(self.e.evaluate_raster(self.d.document, 'read', frame=1, typed=True).geometries, ())
        self.d.document['nodes']['read']['disabled'] = False
        self.set('gltf_path', self.textured_quad())
        self.set('gltf_root', 'nothing')
        with self.assertRaisesRegex(ValueError, "no node named 'nothing'"):
            self.e.evaluate(self.d.document, 'render', frame=1)

    def test_edit_on_disk_invalidates_the_render(self):
        path = self.textured_quad()
        self.set('gltf_path', path)
        before = np.asarray(self.e.evaluate(self.d.document, 'render', frame=1)).copy()
        b = quad_builder(factor=(0.0, 1.0, 0.0, 1.0))
        stat = os.stat(path)
        b.write_glb(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        after = np.asarray(self.e.evaluate(self.d.document, 'render', frame=1))
        self.assertFalse(np.allclose(before, after))
        np.testing.assert_allclose(after[32, 32, :3], working(0, 1, 0, 'Linear Rec.709'), atol=1e-4)

    def test_typed_rejection(self):
        self.d.execute(dict(op='create', id='image', type='Constant', params={}))
        for target, slot, source in (('read', 'image', 'image'), ('render', 'camera', 'read')):
            with self.assertRaises(ValueError):
                self.d.execute(dict(op='connect', id=target, input=slot, source=source))


if __name__ == '__main__':
    unittest.main()
