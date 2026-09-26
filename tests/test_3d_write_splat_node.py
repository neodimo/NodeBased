"""WriteSplat3D: baked transforms, round trip through ReadSplat3D, overwrite guard, bypass."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nodebased import splats
from nodebased.core import Dispatcher, OUTPUT_TYPES, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.splatexport import export_splats, scene_cloud
from test_3d_read_splat_node import cloud


def rich_cloud(n=40, degree=3, seed=5):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return splats.SplatCloud(rng.normal(size=(n, 3)), rng.uniform(.01, .2, (n, 3)), q,
                             rng.uniform(.05, .95, n), rng.normal(size=(n, (degree + 1) ** 2, 3)) * .3, degree)


class WriteSplatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        splats._cloud_cache.clear()
        self.source = cloud()
        self.src_path = self.root / 'source.ply'
        splats.write_ply(self.source, self.src_path)
        self.d = Dispatcher()
        self.e = Evaluator()
        self.add('read', 'ReadSplat3D', splat_path=str(self.src_path))
        self.add('write', 'WriteSplat3D', splat_write_path=str(self.root / 'out.ply'))
        self.connect('write', 'scene', 'read')

    def add(self, key, kind, **params):
        self.d.execute(dict(op='create', id=key, type=kind, params=params))

    def connect(self, key, slot, source):
        self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def set(self, key, **params):
        for name, value in params.items():
            self.d.execute(dict(op='set', id=key, param=name, value=value))

    def export(self, frames=(1,)):
        return export_splats(self.d.document, 'write', frames, self.e)

    def read_back(self, path=None):
        return splats.read_ply(path or self.root / 'out.ply')

    def assert_same(self, a, b, atol=1e-6):
        self.assertEqual(len(a), len(b))
        self.assertEqual(a.sh_degree, b.sh_degree)
        np.testing.assert_allclose(a.positions, b.positions, atol=atol)
        np.testing.assert_allclose(a.scales, b.scales, atol=atol)
        np.testing.assert_allclose(a.opacity, b.opacity, atol=atol)
        np.testing.assert_allclose(a.sh, b.sh, atol=atol)
        np.testing.assert_allclose(a.covariance(), b.covariance(), atol=atol)

    def test_registration(self):
        self.assertEqual(OUTPUT_TYPES['WriteSplat3D'], 'scene')
        self.assertEqual(SPECS['WriteSplat3D']['inputs'], ['scene'])

    def test_round_trip_reader_cloud(self):
        self.assertEqual(self.export(), [str(self.root / 'out.ply')])
        back = self.read_back()
        self.assertEqual(len(back), 579)
        self.assert_same(back, self.source)
        # No transform and no knob changes: the stored numbers come back bit for bit.
        np.testing.assert_array_equal(back.raw_scale_log, self.read_back(self.src_path).raw_scale_log)
        np.testing.assert_array_equal(back.rotations, self.read_back(self.src_path).rotations)

    def test_round_trip_all_sh_degrees(self):
        for degree in range(4):
            with self.subTest(degree=degree):
                original = rich_cloud(degree=degree)
                path = self.root / f'd{degree}.ply'
                splats.write_ply(original, path)
                self.set('read', splat_path=str(path))
                self.set('write', splat_write_path=str(self.root / f'w{degree}.ply'))
                self.export()
                self.assert_same(splats.read_ply(self.root / f'w{degree}.ply'), original)

    def test_node_transform_is_baked(self):
        self.set('read', tx=1.5, ty=-.5, ry=90, uscale=2.0)
        self.export()
        back = self.read_back()
        # +90 degrees about Y sends (x, y, z) to (z, y, -x); then x2 and the translate.
        p = self.source.positions.astype(np.float64)
        expected = np.stack((p[:, 2], p[:, 1], -p[:, 0]), -1) * 2 + (1.5, -.5, 0)
        np.testing.assert_allclose(back.positions, expected, atol=1e-5)
        np.testing.assert_allclose(np.sort(back.scales, axis=1), np.sort(self.source.scales * 2, axis=1), atol=1e-6)
        # Baked quaternions: the covariance is the source covariance rotated and scaled.
        r = np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)), float)
        expected_cov = 4 * r @ self.source.covariance() @ r.T
        np.testing.assert_allclose(back.covariance(), expected_cov, atol=1e-6)

    def test_axis3d_parent_reads_back_at_transformed_positions(self):
        self.add('axis', 'Axis3D', tx=3.0, tz=-2.0, rz=30)
        self.connect('axis', 'object', 'read')
        self.connect('write', 'scene', 'axis')
        self.export()
        back = self.read_back()
        c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
        p = self.source.positions.astype(np.float64)
        expected = np.stack((c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]), -1) + (3, 0, -2)
        np.testing.assert_allclose(back.positions, expected, atol=1e-5)
        # Reading the written file into a fresh ReadSplat3D gives the same world positions.
        self.add('again', 'ReadSplat3D', splat_path=str(self.root / 'out.ply'))
        scene = self.e.evaluate_raster(self.d.document, 'again', frame=1, typed=True)
        np.testing.assert_allclose(scene_cloud(scene).positions, expected, atol=1e-5)

    def test_display_knobs_are_baked(self):
        self.set('read', splat_sh_degree=0, splat_opacity=.5, splat_scale=2.0)
        self.export()
        back = self.read_back()
        self.assertEqual(back.sh_degree, 0)
        np.testing.assert_allclose(back.opacity, self.source.opacity * .5, atol=1e-6)
        np.testing.assert_allclose(back.scales, self.source.scales * 2, atol=1e-6)

    def test_two_sources_merge_into_one_file(self):
        other = rich_cloud(n=10, degree=1)
        splats.write_ply(other, self.root / 'other.ply')
        self.add('read2', 'ReadSplat3D', splat_path=str(self.root / 'other.ply'), tx=10.0)
        self.add('scene', 'Scene3D')
        self.connect('scene', 'object0', 'read')
        self.connect('scene', 'object1', 'read2')
        self.connect('write', 'scene', 'scene')
        self.export()
        back = self.read_back()
        self.assertEqual(len(back), 589)
        self.assertEqual(back.sh_degree, 1)
        np.testing.assert_allclose(back.positions[579:], other.positions + (10, 0, 0), atol=1e-5)
        np.testing.assert_allclose(back.sh[579:], other.sh, atol=1e-6)
        # The degree-one source is padded with zero bands only where it has none.
        self.assertEqual(back.sh.shape[1], 4)

    def test_overwrite_refused_unless_allowed(self):
        self.export()
        stamp = (self.root / 'out.ply').read_bytes()
        self.set('read', tx=4.0)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.export()
        self.assertEqual((self.root / 'out.ply').read_bytes(), stamp)
        self.set('write', splat_write_overwrite=1)
        self.export()
        self.assertNotEqual((self.root / 'out.ply').read_bytes(), stamp)
        np.testing.assert_allclose(self.read_back().positions[:, 0], self.source.positions[:, 0] + 4, atol=1e-5)

    def test_sequence_pattern_and_single_file_range(self):
        with self.assertRaisesRegex(ValueError, 'padded pattern'):
            self.export((1, 2))
        self.set('write', splat_write_path=str(self.root / 'seq' / 's.%03d.ply'))
        written = self.export((3, 4))
        self.assertEqual([Path(w).name for w in written], ['s.003.ply', 's.004.ply'])
        (self.root / 'seq' / 's.003.ply').unlink()
        with self.assertRaisesRegex(ValueError, 's.004.ply already exists'):
            self.export((3, 4, 5))
        self.assertFalse((self.root / 'seq' / 's.003.ply').exists())
        # The check runs before any write: frame 5 was not produced.
        self.assertFalse((self.root / 'seq' / 's.005.ply').exists())

    def test_refusals(self):
        self.set('write', splat_write_path='')
        with self.assertRaisesRegex(ValueError, 'output path'):
            self.export()
        self.set('write', splat_write_path=str(self.root / 'x.obj'))
        with self.assertRaisesRegex(ValueError, '.ply'):
            self.export()
        self.set('write', splat_write_path=str(self.root / 'x.ply'))
        self.add('empty', 'Scene3D')
        self.connect('write', 'scene', 'empty')
        with self.assertRaisesRegex(ValueError, 'no splats'):
            self.export()
        self.assertFalse((self.root / 'x.ply').exists())
        self.connect('write', 'scene', None)
        with self.assertRaisesRegex(ValueError, 'upstream scene'):
            self.export()

    def test_bypass_passes_the_scene(self):
        self.d.execute(dict(op='disable', id='write', value=True))
        self.assertEqual(bypass_slot(self.d.document['nodes']['write']), 'scene')
        passed = self.e.evaluate_raster(self.d.document, 'write', frame=1, typed=True)
        source = self.e.evaluate_raster(self.d.document, 'read', frame=1, typed=True)
        self.assertEqual(len(passed.splats), 1)
        self.assertIs(passed.splats[0].cloud, source.splats[0].cloud)
        self.assertFalse((self.root / 'out.ply').exists())  # evaluating never writes

    def test_evaluating_passes_scene_without_writing(self):
        value = self.e.evaluate_raster(self.d.document, 'write', frame=1, typed=True)
        self.assertEqual(len(value.splats), 1)
        self.assertFalse((self.root / 'out.ply').exists())

    def test_no_home_path_in_written_file(self):
        self.export()
        data = (self.root / 'out.ply').read_bytes()
        header = data[:data.index(b'end_header')]
        self.assertNotIn(str(Path.home()).encode(), header)
        self.assertNotIn(str(self.root).encode(), data)


if __name__ == '__main__':
    unittest.main()
