"""Independent Blender fixture checks plus handcrafted format/error cases."""
from collections import Counter
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from nodebased.alembicio import AlembicError, TimeSampling, dump, open_archive, read_polymesh

FIXTURE = Path(__file__).parent / 'fixtures' / 'abc' / 'probe.abc'
POSITIONS = [(0, 0, 0), (2, 0, 0), (2, 0, -3), (0, 0, -3), (1, 0, -5)]
DATA = 1 << 63


class BlenderFixtureTests(unittest.TestCase):
    def setUp(self):
        self.archive = open_archive(FIXTURE)
        self.addCleanup(self.archive.close)
        self.obj = self.archive.root.children['rig'].children['probe'].children['probe']

    def test_hierarchy_metadata_and_dump(self):
        a = self.archive
        self.assertEqual(set(a.root.children), {'rig', 'cam'})
        self.assertEqual(set(a.root.children['rig'].children), {'probe'})
        self.assertEqual(set(a.root.children['cam'].children), {'cam'})
        self.assertEqual(self.obj.full_path, '/rig/probe/probe')
        self.assertEqual(self.obj.metadata['schema'], 'AbcGeom_PolyMesh_v1')
        self.assertIn('schemaObjTitle', self.obj.metadata)
        self.assertEqual(a.metadata['_ai_Application'], 'Blender')
        self.assertEqual(a.ogawa_version, 1)
        self.assertIn('P: array float32[3] samples=5', dump(a))
        self.assertEqual(a.time_samplings[0].sample_time(2), 2)

    def test_positions_topology_and_bounds(self):
        first, last = read_polymesh(self.obj), read_polymesh(self.obj, 4)
        np.testing.assert_array_equal(first.positions, POSITIONS)
        np.testing.assert_array_equal(last.positions, POSITIONS[:4] + [(1, 2, -5)])
        self.assertEqual(first.positions.dtype, np.float32)
        self.assertEqual(first.face_indices.dtype, np.int32)
        np.testing.assert_array_equal(first.face_counts, [4, 3])
        self.assertEqual(set(first.face_indices[:4]), {0, 1, 2, 3})
        self.assertEqual(set(first.face_indices[4:]), {2, 3, 4})
        # Blender reverses each authored face for Alembic; preserve this winding.
        np.testing.assert_array_equal(first.face_indices, [3, 2, 1, 0, 4, 2, 3])
        np.testing.assert_array_equal(last.face_indices, [3, 2, 1, 0, 4, 2, 3])
        np.testing.assert_array_equal(first.self_bounds, [(0, 0, -5), (2, 0, 0)])
        np.testing.assert_array_equal(last.self_bounds, [(0, 0, -5), (2, 2, 0)])
        self.assertEqual(first.num_samples, 5)

    def test_uvs_and_normals(self):
        mesh = read_polymesh(self.obj)
        uv = mesh.uvs
        self.assertEqual(uv.scope, 'fvr')
        np.testing.assert_array_equal(uv.indices, [0, 1, 2, 3, 4, 1, 0])
        expanded = uv.expanded_face_corners(mesh.face_counts, mesh.face_indices)
        self.assertEqual(Counter(map(tuple, expanded[:4])), Counter([(0, 0), (1, 0), (1, .5), (0, .5)]))
        self.assertEqual(Counter(map(tuple, expanded[4:])), Counter([(0, .5), (1, .5), (.5, 1)]))
        expected_pairs = {(0, 0, 0): (0, 0), (2, 0, 0): (1, 0),
                          (2, 0, -3): (1, .5), (0, 0, -3): (0, .5), (1, 0, -5): (.5, 1)}
        for position, coord in zip(mesh.positions[mesh.face_indices], expanded):
            self.assertEqual(tuple(coord), expected_pairs[tuple(position)])
        self.assertEqual(mesh.normals.scope, 'fvr')
        np.testing.assert_array_equal(mesh.normals.values, [(0, 1, 0)] * 7)
        np.testing.assert_allclose(read_polymesh(self.obj, 4).normals.values,
                                   [(0, 1, 0)] * 4 + [(0, 2**-.5, 2**-.5)] * 3)

    def test_actual_fixture_times(self):
        # Contrary to the generator docstring, this Blender file starts at 1/24.
        # Archive times must not be silently rebased to zero.
        sampling = read_polymesh(self.obj).time_sampling
        np.testing.assert_allclose([sampling.sample_time(i) for i in range(5)],
                                   [1/24, 2/24, 3/24, 4/24, 5/24])
        self.assertEqual(sampling.lookup(2/24, 5), (1, 1, 0))
        lo, hi, weight = sampling.lookup(2.5/24, 5)
        self.assertEqual((lo, hi), (1, 2))
        self.assertAlmostEqual(weight, .5)
        prop = self.obj.properties['.geom']['.faceCounts']
        self.assertTrue(prop.is_constant)
        # Five logical samples, one physical sample, per the property header.
        self.assertEqual(prop.num_samples, 5)
        np.testing.assert_array_equal(prop.read(4), [4, 3])

    def test_every_fixture_sample_decodes(self):
        def props(p):
            for child in p.properties.values():
                if child.kind == 'compound':
                    props(child)
                else:
                    for i in range(child.num_samples):
                        child.read(i)
        def visit(obj):
            props(obj.properties)
            for child in obj.children.values():
                visit(child)
        visit(self.archive.root)

    def test_close_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'copy.abc'
            path.write_bytes(FIXTURE.read_bytes())
            with open_archive(path) as archive:
                prop = archive.root.children['rig'].properties['.xform']['.vals']
                values = prop.read()
            archive.close()
            path.unlink()
            self.assertEqual(values.shape, (16,))
            with self.assertRaisesRegex(AlembicError, 'closed'):
                prop.read()


class TimeTests(unittest.TestCase):
    def test_uniform_zero_based_and_constant(self):
        sampling = TimeSampling('uniform', 1/24, (0.,))
        self.assertEqual(sampling.lookup(2/24, 5), (2, 2, 0))
        lo, hi, weight = sampling.lookup(2.5/24, 5)
        self.assertEqual((lo, hi), (2, 3))
        self.assertAlmostEqual(weight, .5)
        self.assertEqual(sampling.lookup(-10, 5), (0, 0, 0))
        self.assertEqual(sampling.lookup(10, 5), (4, 4, 0))
        self.assertEqual(sampling.lookup(10, 1), (0, 0, 0))
        self.assertEqual(sampling.lookup(2/24 - 1e-6, 5), (2, 2, 0))

    def test_cyclic_and_acyclic(self):
        cyclic = TimeSampling('cyclic', 1., (.1, .3, .7))
        np.testing.assert_allclose([cyclic.sample_time(i) for i in range(7)], [.1, .3, .7, 1.1, 1.3, 1.7, 2.1])
        self.assertEqual(cyclic.lookup(.9, 7)[:2], (2, 3))
        self.assertAlmostEqual(cyclic.lookup(.9, 7)[2], .5)
        acyclic = TimeSampling('acyclic', np.finfo(float).max, (0., .2, .9))
        self.assertEqual(acyclic.lookup(.55, 3)[:2], (1, 2))
        self.assertAlmostEqual(acyclic.lookup(.55, 3)[2], .5)
        self.assertEqual(acyclic.lookup(.8, 2), (1, 1, 0))
        with self.assertRaises(AlembicError):
            acyclic.sample_time(3)


class FormatTests(unittest.TestCase):
    def check_bad(self, data, message=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.abc'
            path.write_bytes(data)
            with self.assertRaisesRegex(AlembicError, message or '.'):
                open_archive(path)
            path.unlink()

    def test_errors(self):
        self.check_bad(b'\x89HDF\r\n\x1a\n', r'HDF5-backed Alembic archives are not supported \(only Ogawa\)')
        self.check_bad(b'garbage', 'Not an Ogawa')
        raw = bytearray(FIXTURE.read_bytes())
        raw[5] = 0
        self.check_bad(raw, 'not frozen')
        raw = bytearray(FIXTURE.read_bytes())
        raw[8:16] = struct.pack('<Q', len(raw) + 100)
        self.check_bad(raw)
        for length in (0, 5, 8, 15, 16, 64, 256, len(raw)//2, len(raw)-1):
            with self.subTest(length=length):
                self.check_bad(FIXTURE.read_bytes()[:length])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AlembicError, 'Cannot read Alembic file'):
                open_archive(Path(directory) / 'missing.abc')
            with self.assertRaisesRegex(AlembicError, 'Cannot read Alembic file'):
                open_archive(directory)

    def test_cycle_and_oversize(self):
        header = b'Ogawa\xff\x00\x01' + struct.pack('<Q', 16)
        self.check_bad(header + struct.pack('<QQ', 1, 16), 'Cyclic')
        self.check_bad(header + struct.pack('<Q', 2**63), 'oversized')
        # A chain exceeding the explicit container depth limit.
        self.check_bad(header + b''.join(struct.pack('<QQ', 1, 32 + i*16) for i in range(130)) + struct.pack('<QQ', 1, 0), 'too deep')

    def test_handcrafted_pods_dimensions_and_repeats(self):
        # Minimal independent byte assembly, not a writer from the reader module.
        blob = bytearray(b'Ogawa\xff\x00\x01' + bytes(8))
        def data(payload):
            offset = len(blob)
            blob.extend(struct.pack('<Q', len(payload)) + payload)
            return DATA | offset
        def group(refs):
            if not refs:
                return 0
            offset = len(blob)
            blob.extend(struct.pack('<Q', len(refs)) + struct.pack('<' + 'Q'*len(refs), *refs))
            return offset
        headers, refs, expectations = [], [], {}
        dtypes = ['?', 'u1', 'i1', '<u2', '<i2', '<u4', '<i4', '<u8', '<i8', '<f2', '<f4', '<f8']
        for pod in range(14):
            name = f'p{pod}'.encode()
            if pod < 12:
                expected = np.array([1, 0], dtype=dtypes[pod])
                payload = expected.tobytes()
            else:
                expected = np.array(['hello', 'é'], dtype=object)
                payload = 'hello\0é\0'.encode('utf-8' if pod == 12 else 'utf-32-le')
            expectations[name.decode()] = expected
            refs.append(group([data(bytes(16) + payload), data(struct.pack('<QQ', 1, 2))]))
            # Array, extent 1, constant; explicit two-dimensional shape.
            hint = pod % 3
            width = 1 << hint
            inline = b'interpretation=test'
            info = 2 | hint << 2 | pod << 4 | 1 << 12 | 0x800 | 255 << 20
            headers.append(struct.pack('<I', info) + (1).to_bytes(width, 'little') +
                           len(name).to_bytes(width, 'little') + name +
                           len(inline).to_bytes(width, 'little') + inline)
        # First two logical samples repeat; then 20, 30, and a repeated end.
        refs.append(group([data(bytes(16) + struct.pack('<i', v)) for v in (10, 20, 30)]))
        headers.append(struct.pack('<I', 1 | 6 << 4 | 1 << 12 | 0x200) + bytes([5, 2, 3, 6]) + b'repeat')
        compound = group(refs + [data(b''.join(headers))])
        obj = group([compound, data(bytes(32))])
        times = data(struct.pack('<IdId', 5, 1., 1, 0.) +
                     struct.pack('<IdIdd', 6, 1., 2, .1, .4) +
                     struct.pack('<IdIddd', 3, np.finfo(float).max, 3, 0., .2, .9))
        root = group([data(struct.pack('<i', 0)), data(struct.pack('<i', 10803)), obj, data(b''), times, data(b'')])
        blob[8:16] = struct.pack('<Q', root)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pods.abc'
            path.write_bytes(blob)
            with open_archive(path) as archive:
                self.assertEqual([t.kind for t in archive.time_samplings], ['uniform', 'cyclic', 'acyclic'])
                self.assertAlmostEqual(archive.time_samplings[1].sample_time(3), 1.4)
                self.assertEqual(archive.time_samplings[2].sample_time(2), .9)
                for name, expected in expectations.items():
                    self.assertEqual(archive.root.properties[name].metadata, {'interpretation': 'test'})
                    actual = archive.root.properties[name].read()
                    self.assertEqual(actual.shape, (1, 2))
                    self.assertEqual(actual.dtype, expected.dtype)
                    np.testing.assert_array_equal(actual[0], expected)
                prop = archive.root.properties['repeat']
                self.assertEqual([prop.read(i)[0] for i in range(5)], [10, 10, 20, 30, 30])
                with self.assertRaises(AlembicError):
                    prop.read(5)


if __name__ == '__main__':
    unittest.main()
