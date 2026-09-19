"""Independent Blender fixture checks plus handcrafted format/error cases."""
from collections import Counter
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from nodebased.alembicio import (AlembicError, TimeSampling, dump, open_archive, read_polymesh,
                                ObjectReader, Property, read_xform, xform_at_time,
                                world_matrix, read_camera, camera_at_time,
                                camera_to_scene3d, _decode_xform_ops)

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






class TransformCameraFixtureTests(unittest.TestCase):
    def setUp(self):
        self.archive = open_archive(FIXTURE)
        self.addCleanup(self.archive.close)
        self.rig = self.archive.root.children['rig']
        self.camera = self.archive.root.children['cam'].children['cam']

    def test_rig_endpoints_identity_and_world_apex(self):
        c, s = math.cos(math.pi/6), .5
        for index, x, y in [(0, 1, 0), (4, 5, 2)]:
            expected = np.array([[c, 0, -s, 0], [0, 1, 0, 0],
                                 [s, 0, c, 0], [x, 3, -2, 1]])
            actual = read_xform(self.rig, index)
            self.assertEqual(actual.matrix.dtype, np.float64)
            self.assertTrue(actual.inherits)
            self.assertEqual(actual.ops[0].type, 'matrix')
            self.assertEqual(actual.ops[0].animated_channels, (12,))
            np.testing.assert_allclose(actual.matrix, expected, atol=1e-7)
            np.testing.assert_array_equal(actual.matrix,
                self.rig.properties['.xform']['.vals'].read(index).reshape(4, 4))
            time = (index+1)/24
            world = world_matrix(self.archive, '/rig/probe/probe', time)
            # (1,y,-5) @ R + (x,3,-2), independently expanded by column.
            np.testing.assert_allclose(np.array([1, y, -5, 1]) @ world,
                                       [c-5*s+x, y+3, -s-5*c-2, 1], atol=2e-7)
        probe = self.rig.children['probe']
        np.testing.assert_array_equal(read_xform(probe).matrix, np.eye(4))
        np.testing.assert_array_equal(xform_at_time(probe, 100), np.eye(4))
        for time, index in [(-1, 0), (1/24, 0), (5/24, 4), (100, 4)]:
            np.testing.assert_array_equal(xform_at_time(self.rig, time),
                                          self.rig.properties['.xform']['.vals'].read(index).reshape(4, 4))

    def test_interpolation_uses_stored_bezier_neighbours(self):
        prop = self.rig.properties['.xform']['.vals']
        a, b = (prop.read(i).reshape(4, 4) for i in (1, 2))
        # Exporter samples Blender Bezier keys. Our interpolation is between
        # these stored neighbours, not between the authored frame-1/5 keys.
        actual = xform_at_time(self.rig, 2.5/24)
        np.testing.assert_allclose(actual[3, :3], (a[3, :3]+b[3, :3])/2)
        np.testing.assert_allclose(actual[:3, :3], a[:3, :3], atol=1e-7)
        core = self.camera.properties['.geom']['.core']
        expected = (core.read(1)+core.read(2))/2
        result = camera_at_time(self.camera, 2.5/24)
        np.testing.assert_allclose(list(vars(result).values()), expected)

    def test_camera_core_world_and_projection(self):
        from nodebased import scene3d
        c, s = math.cos(math.radians(5)), math.sin(math.radians(5))
        # Blender's +85 X camera plus exporter camera-axis conversion yields
        # -5 X here: stored row Y has negative Z, row Z has positive Y.
        expected = [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 1.5, 6, 1]]
        stored = self.archive.root.children['cam'].properties['.xform']['.vals'].read().reshape(4, 4)
        np.testing.assert_allclose(stored, expected, atol=2e-7)
        np.testing.assert_allclose(world_matrix(self.archive.root, self.camera, 1/24), expected, atol=2e-7)
        for time, index, focal in [(1/24, 0, 35), (5/24, 4, 70), (-10, 0, 35), (10, 4, 70)]:
            core = read_camera(self.camera, index)
            self.assertAlmostEqual(core.focal_length, focal)
            self.assertAlmostEqual(camera_at_time(self.camera, time).focal_length, focal)
            for name, value in [('horizontal_aperture', 3.6), ('vertical_aperture', 2),
                                ('near_clipping_plane', .25), ('far_clipping_plane', 300),
                                ('focus_distance', 8), ('f_stop', 2.8)]:
                self.assertAlmostEqual(getattr(core, name), value, places=6)
            camera = camera_to_scene3d(self.archive, self.camera, time)
            np.testing.assert_allclose(camera.transform.position.array(), [0, 1.5, 6])
            np.testing.assert_allclose(camera.target.array(), [0, 1.5-8*s, 6-8*c], atol=2e-6)
            self.assertAlmostEqual(camera.fov, math.degrees(2*math.atan(20/(2*focal))))
            self.assertEqual((camera.near, camera.far), (.25, 300))
            self.assertAlmostEqual(camera.roll, 0)
            pixels, depth = scene3d.project(camera, 800, 600, [camera.target.array()])
            np.testing.assert_allclose(pixels, [[400, 300]], atol=1e-4)
            np.testing.assert_allclose(depth, [8], atol=1e-6)


class SampleProperty:
    """Independent in-memory property stub, exercising the public tree interface."""
    def __init__(self, *samples, constant=False, times=None):
        self.samples = [np.asarray(s) for s in samples]
        self.num_samples = len(samples)
        self.is_constant = constant or len(samples) == 1
        self.time_sampling = times or TimeSampling()

    def read(self, index=0):
        return self.samples[index].copy()

    def lookup(self, time):
        return self.time_sampling.lookup(time, self.num_samples)


def synthetic_xform(path, matrices, inherits=None):
    props = {'.ops': SampleProperty([0x30]),
             '.vals': SampleProperty(*(np.asarray(m).ravel() for m in matrices)),
             'isNotConstantIdentity': SampleProperty([True])}
    if inherits is not None:
        props['.inherits'] = inherits
    return ObjectReader(path.rsplit('/', 1)[-1], path, {'schema': 'AbcGeom_Xform_v3'},
                        Property('', {}, properties={'.xform': Property('.xform', {}, properties=props)}))


class SyntheticTransformTests(unittest.TestCase):
    def test_stack_hints_channels_order_and_errors(self):
        # Listed T, Rz, S produces S @ Rz @ T (not T @ Rz @ S).
        matrix, ops = _decode_xform_ops([0x11, 0x62, 0x00], [4, 5, 6, 90, 2, 3, 4], [0, 3, 6])
        np.testing.assert_allclose(matrix, [[0, 2, 0, 0], [-3, 0, 0, 0],
                                           [0, 0, 4, 0], [4, 5, 6, 1]], atol=1e-15)
        self.assertEqual([op.hint for op in ops], [1, 2, 0])
        self.assertEqual([op.animated_channels for op in ops], [(0,), (0,), (2,)])
        for code, axis, expected in [
                (0x40, [1, 0, 0], [[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                (0x50, [0, 1, 0], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                (0x60, [0, 0, 1], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])]:
            for encoded, values in [([code], [90]), ([0x20], axis+[90])]:
                matrix, _ = _decode_xform_ops(encoded, values)
                np.testing.assert_allclose(matrix[:3, :3], expected, atol=1e-15)
        for encoded, values in [([0xf0], []), ([0x30], [1]), ([0x00], [1, 2, 3, 4])]:
            with self.assertRaises(AlembicError):
                _decode_xform_ops(encoded, values)

    def test_full_channel_samples_and_final_animation_labels(self):
        obj = synthetic_xform('/stack', [np.eye(4)])
        props = obj.properties['.xform'].properties
        props['.ops'] = SampleProperty([0x10, 0x00])
        # Full channel arrays, not just the two animated values. Animation
        # labels accumulate over time, so only the final label sample is used.
        props['.vals'] = SampleProperty([1, 2, 3, 2, 2, 2], [5, 2, 3, 2, 4, 2])
        props['.animChans'] = SampleProperty([0], [0, 4])
        result = read_xform(obj, 1)
        np.testing.assert_array_equal(result.matrix,
                                      [[2, 0, 0, 0], [0, 4, 0, 0],
                                       [0, 0, 2, 0], [5, 2, 3, 1]])
        self.assertEqual([o.animated_channels for o in result.ops], [(0,), (1,)])
        self.assertEqual(result.ops[0].values, (5, 2, 3))
        np.testing.assert_allclose(xform_at_time(obj, .5),
                                   [[2, 0, 0, 0], [0, 3, 0, 0],
                                    [0, 0, 2, 0], [3, 2, 3, 1]])

    def test_quaternion_shortest_path_scale_and_shear_fallback(self):
        def rz(degrees, scale, translation):
            c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
            return np.array([[scale*c, scale*s, 0, 0], [-scale*s, scale*c, 0, 0],
                             [0, 0, scale, 0], [translation, 0, 0, 1]])
        a, b = rz(170, 1, 0), rz(-170, 3, 4)
        obj = synthetic_xform('/x', [a, b])
        np.testing.assert_allclose(xform_at_time(obj, .5), rz(180, 2, 2), atol=1e-14)
        # Exercise each quaternion largest-component branch, including near zero.
        for axis in range(3):
            a, b = np.eye(4), np.eye(4)
            other = [i for i in range(3) if i != axis]
            b[other, other] = -1
            actual = xform_at_time(synthetic_xform('/x', [a, b]), .5)
            np.testing.assert_allclose(actual @ actual, b, atol=1e-14)
        a, b = np.eye(4), np.eye(4)
        a[0, 1], b[0, 1] = .2, .8
        obj = synthetic_xform('/x', [a, b])
        np.testing.assert_array_equal(xform_at_time(obj, 0), a)
        np.testing.assert_allclose(xform_at_time(obj, .5), (a+b)/2)

    def test_inheritance_and_identity_shortcuts(self):
        a, b = np.eye(4), np.eye(4)
        a[3, :3], b[3, :3] = [10, 0, 0], [0, 2, 0]
        parent = synthetic_xform('/a', [a])
        child = synthetic_xform('/a/b', [b], SampleProperty([True], [False]))
        parent.children['b'] = child
        root = ObjectReader('ABC', '/', {}, Property('', {}), {'a': parent})
        np.testing.assert_allclose(world_matrix(root, child, 0)[3, :3], [10, 2, 0])
        np.testing.assert_allclose(world_matrix(root, '/a/b', 1)[3, :3], [0, 2, 0])
        props = child.properties['.xform'].properties
        props['isNotConstantIdentity'] = SampleProperty([False])
        np.testing.assert_array_equal(read_xform(child).matrix, np.eye(4))
        del props['isNotConstantIdentity']
        del props['.vals']
        del props['.ops']
        np.testing.assert_array_equal(read_xform(child).matrix, np.eye(4))

    def test_camera_roll_focus_and_rejected_transforms(self):
        from nodebased import scene3d
        # Camera looks down -Z with local +Y pointing left: scene3d roll +90.
        matrix = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
        parent = synthetic_xform('/p', [matrix])
        core = [50, 3.6, 0, 2, 0, 1, 0, 0, 0, 0, 2.8, 0, 0, .02, .1, 100]
        cam = ObjectReader('c', '/p/c', {'schema': 'AbcGeom_Camera_v1'},
            Property('', {}, properties={'.geom': Property('.geom', {}, properties={'.core': SampleProperty(core)})}))
        parent.children['c'] = cam
        camera = camera_to_scene3d(parent, cam, 0)
        self.assertAlmostEqual(camera.roll, 90)
        np.testing.assert_allclose(camera.target.array(), [0, 0, -1])
        np.testing.assert_allclose(scene3d._view_basis(camera)[1][1], [-1, 0, 0], atol=1e-7)
        pixels, _ = scene3d.project(camera, 800, 600, [[-1, 0, -5]])
        self.assertAlmostEqual(pixels[0, 0], 400)
        self.assertLess(pixels[0, 1], 300)
        for axes in [np.diag([1, 2, 1]), np.diag([-1, 1, 1]),
                     np.array([[1, .2, 0], [0, 1, 0], [0, 0, 1]])]:
            bad = np.eye(4)
            bad[:3, :3] = axes
            parent.properties['.xform'].properties['.vals'] = SampleProperty(bad.ravel())
            with self.assertRaisesRegex(AlembicError, 'scale, shear or reflection'):
                camera_to_scene3d(parent, cam, 0)
        uniform = np.diag([2., 2, 2, 1])
        parent.properties['.xform'].properties['.vals'] = SampleProperty(uniform.ravel())
        np.testing.assert_allclose(camera_to_scene3d(parent, cam, 0).target.array(), [0, 0, -1])


if __name__ == '__main__':
    unittest.main()
