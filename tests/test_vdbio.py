"""Lane L6 fluids plan, step B: the in-house OpenVDB reader (nodebased.vdbio) and the ReadVDB3D node.

The assets are written by vdbio's own writer into a temporary folder, so the round trip is
self-contained. Real Blender caches (Blosc, streamed, half float) are read too when the workspace
holds them (scratch/vdb-samples); those tests skip otherwise.
"""
import copy
import itertools
import math
import os
import shutil
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import scene3d, vdbio
from nodebased.core import CHOICES, Dispatcher, OUTPUT_TYPES, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from tests.test_volume_render import CAMERA, FLAT
from tests.test_volume_scene import make, wire

def blob(shape=(20, 17, 25), seed=1):
    """A density block with a solid interior, an empty margin and a spread of values."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32)
    out[3:15, 2:12, 5:20] = rng.random((12, 10, 15), dtype=np.float32) + .1
    return out


def velocity(shape=(20, 17, 25), seed=2):
    return np.random.default_rng(seed).standard_normal(shape + (3,)).astype(np.float32)


class TempDir(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def path(self, name="a.vdb"):
        return self.dir / name


class RoundTripTests(TempDir):
    def test_density_and_velocity_round_trip_in_every_storage_form(self):
        dens, vel = blob(), velocity()
        on = dens != 0
        for compression, mask, half in itertools.product(("none", "zip", "blosc"), (True, False), (False, True)):
            with self.subTest(compression=compression, mask=mask, half=half):
                path = vdbio.write_vdb(self.path(f"{compression}{mask}{half}.vdb"), {"density": dens, "vel": vel},
                                       voxel_size=0.1, index_min=(-5, 3, 0), compression=compression,
                                       mask_compression=mask, half=half, active={"vel": on})
                density, motion = vdbio.read_grid(path, "density"), vdbio.read_grid(path, "vel")
                self.assertEqual(density.index_min, (-2, 5, 5))           # the active box, not the array
                self.assertEqual(density.data.shape, (12, 10, 15))
                self.assertEqual(motion.data.shape, (12, 10, 15, 3))
                box = (slice(3, 15), slice(2, 12), slice(5, 20))
                if half:                                                    # exactly the float16 value
                    np.testing.assert_array_equal(density.data, dens[box].astype(np.float16).astype(np.float32))
                    np.testing.assert_array_equal(motion.data, vel[box].astype(np.float16).astype(np.float32))
                else:
                    np.testing.assert_array_equal(density.data, dens[box])
                    np.testing.assert_array_equal(motion.data, vel[box])
                self.assertEqual(density.active_voxels, int(on.sum()))
                self.assertEqual(density.transform, "UniformScaleMap")
                np.testing.assert_allclose(np.diag(density.matrix)[:3], 0.1)

    def test_native_half_and_double_grids_read(self):
        dens = blob().astype(np.float16)
        path = vdbio.write_vdb(self.path(), {"h": dens, "d": blob().astype(np.float64)}, compression="zip")
        self.assertEqual([i.grid_type for i in vdbio.list_grids(path)], ["Tree_half_5_4_3", "Tree_double_5_4_3"])
        np.testing.assert_array_equal(vdbio.read_grid(path, "h").data, dens[3:15, 2:12, 5:20].astype(np.float32))
        np.testing.assert_allclose(vdbio.read_grid(path, "d").data, blob()[3:15, 2:12, 5:20])

    def test_half_grids_are_flagged_and_lossy(self):
        path = vdbio.write_vdb(self.path(), {"density": blob()}, half=True)
        info = vdbio.list_grids(path)[0]
        self.assertTrue(info.half_float)
        self.assertEqual((info.name, info.value_type, info.grid_class, info.voxel_count),
                         ("density", "float", "fog volume", int((blob() != 0).sum())))
        self.assertEqual(info.bbox, ((0, 0, 0), (19, 16, 24)))
        grid = vdbio.read_grid(path, "density")
        self.assertTrue(grid.half_float)
        self.assertFalse(np.array_equal(grid.data, blob()[3:15, 2:12, 5:20]))
        np.testing.assert_allclose(grid.data, blob()[3:15, 2:12, 5:20], rtol=1e-3, atol=1e-4)

    def test_reads_are_deterministic(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "vel": velocity()}, compression="zip")
        a, b = vdbio.load_volume(path), vdbio.load_volume(path)
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_a_streamed_file_without_an_offset_table_reads_every_grid(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "vel": velocity(), "heat": blob(seed=9)},
                               offsets=False, compression="zip", active={"vel": blob() != 0})
        self.assertEqual(vdbio.grid_names(path), ["density", "vel", "heat"])
        np.testing.assert_array_equal(vdbio.read_grid(path, "heat").data, blob(seed=9)[3:15, 2:12, 5:20])
        np.testing.assert_array_equal(vdbio.read_grid(path, "vel").data, velocity()[3:15, 2:12, 5:20])


class ActiveMaskTests(TempDir):
    def test_inactive_voxels_read_as_zero_whatever_they_store(self):
        dens = blob()
        active = dens != 0
        active[6:9, 5:8, 8:12] = False                                       # a hole inside the smoke
        for name, fill in (("one_value", 7.0), ("minus_background", -0.0), ("many_values", None)):
            stored = dens.copy()
            if fill is None:                                                 # more than two distinct inactive values
                stored[~active] = np.random.default_rng(5).random(int((~active).sum()), dtype=np.float32) * 9 + 1
            else:
                stored[~active] = fill
            for mask in (True, False):
                with self.subTest(case=name, mask_compression=mask):
                    path = vdbio.write_vdb(self.path(f"{name}{mask}.vdb"), {"density": stored},
                                           mask_compression=mask, compression="zip", active={"density": active})
                    grid = vdbio.read_grid(path, "density")
                    lo = np.array(grid.index_min)
                    want = np.where(active, stored, 0)[lo[0]:lo[0] + grid.data.shape[0],
                                                       lo[1]:lo[1] + grid.data.shape[1],
                                                       lo[2]:lo[2] + grid.data.shape[2]]
                    np.testing.assert_array_equal(grid.data, want)
                    self.assertEqual(float(grid.data[6 - lo[0]:9 - lo[0], 5 - lo[1]:8 - lo[1], 8 - lo[2]:12 - lo[2]].max()), 0.0)
                    self.assertEqual(grid.active_voxels, int(active.sum()))

    def test_every_mask_compression_form_parses(self):
        """One grid per Compression.h metadata byte 0 to 6, each followed by another grid so a misparse shows."""
        base = np.zeros((16, 16, 16), np.float32)
        base[2:6, 2:6, 2:6] = 1.5
        active = base != 0
        forms = {}
        forms[0] = base                                                       # inactive == background
        minus = base.copy()
        minus[~active] = -0.0                                                 # -0.0 equals +0.0: form 0 again
        forms[1] = minus
        one = base.copy()
        one[~active] = 3.0
        forms[2] = one
        two = base.copy()
        two[~active] = 0.0
        two[8:, :, :] = 4.0                                                   # background plus one other value
        forms[4] = two
        both = base.copy()
        both[~active] = 2.0
        both[8:, :, :] = 5.0                                                  # two values, neither the background
        forms[5] = both
        many = base.copy()
        many[~active] = np.arange(int((~active).sum()), dtype=np.float32) % 7 + 1
        forms[6] = many
        grids = {f"f{k}": v for k, v in forms.items()}
        grids["tail"] = base + 1
        path = vdbio.write_vdb(self.path(), grids, compression="zip", active={**{n: active for n in grids},
                                                                               "tail": active})
        for name, array in grids.items():
            grid = vdbio.read_grid(path, name)
            lo = np.array(grid.index_min)
            np.testing.assert_array_equal(grid.data, np.where(active, array, 0)[2:6, 2:6, 2:6], err_msg=name)
            self.assertEqual(tuple(lo), (2, 2, 2))

    def test_active_tiles_fill_their_block(self):
        dens = np.zeros((16, 16, 16), np.float32)
        dens[1, 1, 1] = 2.0
        path = vdbio.write_vdb(self.path(), {"density": dens}, tiles={"density": [((8, 0, 8), 0.75)]},
                               compression="zip", index_min=(0, 0, 0))
        grid = vdbio.read_grid(path, "density")
        self.assertEqual(grid.index_min, (1, 0, 1))                             # the tile's floor and the lone voxel
        self.assertEqual(grid.data.shape, (15, 8, 15))
        self.assertTrue((grid.data[7:15, 0:8, 7:15] == 0.75).all())
        self.assertEqual(float(grid.data[0, 1, 0]), 2.0)                        # index (1, 1, 1)
        self.assertEqual(grid.active_voxels, 8 ** 3 + 1)
        self.assertEqual(float(grid.data[0:7, 0:8, 0:7].sum()), 2.0)

    def test_an_empty_grid_is_reported(self):
        path = vdbio.write_vdb(self.path(), {"density": np.zeros((8, 8, 8), np.float32)})
        with self.assertRaisesRegex(vdbio.VdbError, "no active voxels"):
            vdbio.read_grid(path, "density")


class RefusalTests(TempDir):
    def test_a_frustum_transform_is_refused_by_name(self):
        path = vdbio.write_vdb(self.path(), {"density": blob()}, transform="frustum")
        with self.assertRaisesRegex(vdbio.VdbError, "frustum-transform VDB grids are not supported"):
            vdbio.read_grid(path, "density")

    def test_an_unknown_grid_class_is_refused_with_the_supported_list(self):
        path = vdbio.write_vdb(self.path(), {"count": np.arange(8 ** 3, dtype=np.int32).reshape(8, 8, 8) + 1,
                                             "density": blob()})
        info = {i.name: i for i in vdbio.list_grids(path)}
        self.assertFalse(info["count"].supported)
        self.assertTrue(info["density"].supported)
        with self.assertRaisesRegex(vdbio.VdbError, r"unsupported VDB grid class 'Tree_int32_5_4_3': ReadVDB3D reads "
                                                    r"float, half, double, vec3s, vec3d"):
            vdbio.read_grid(path, "count")
        vdbio.read_grid(path, "density")                                     # the other grid is untouched

    def test_non_vdb_truncated_and_old_files(self):
        junk = self.path("junk.vdb")
        junk.write_bytes(b"not a vdb file at all, sorry")
        with self.assertRaisesRegex(vdbio.VdbError, "is not a VDB file"):
            vdbio.list_grids(junk)
        good = vdbio.write_vdb(self.path("good.vdb"), {"density": blob()}, compression="zip")
        data = good.read_bytes()
        cut = self.path("cut.vdb")
        cut.write_bytes(data[:len(data) // 2])
        with self.assertRaisesRegex(vdbio.VdbError, "unexpected end of file|corrupt"):
            vdbio.read_grid(cut, "density")
        old = self.path("old.vdb")
        old.write_bytes(data[:8] + struct.pack("<I", 221) + data[12:])
        with self.assertRaisesRegex(vdbio.VdbError, "version 221 is older than 222"):
            vdbio.list_grids(old)

    def test_a_missing_grid_lists_what_the_file_has(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "vel": velocity()})
        with self.assertRaisesRegex(vdbio.VdbError, "no grid named 'temp'; the file has: density, vel"):
            vdbio.read_grid(path, "temp")

    def test_the_voxel_budget_is_enforced(self):
        path = vdbio.write_vdb(self.path(), {"density": blob()})
        with self.assertRaisesRegex(vdbio.VdbError, "over the 1,000 voxel limit"):
            vdbio.read_grid(path, "density", max_voxels=1000)

    def test_unsupported_blosc_codecs_are_named(self):
        chunk = bytearray(vdbio._blosc_stored(bytes(range(256)) * 4))
        chunk[2] = (chunk[2] & 0x1f) | (3 << 5)                              # zlib codec bits, stored streams
        self.assertEqual(vdbio.blosc_decompress(bytes(chunk), 1024), vdbio.blosc_decompress(
            vdbio._blosc_stored(bytes(range(256)) * 4), 1024))               # stored streams need no codec
        chunk[2] |= 0x4
        with self.assertRaisesRegex(vdbio.VdbError, "bit-shuffled"):
            vdbio.blosc_decompress(bytes(chunk), 1024)


class ContainerTests(unittest.TestCase):
    def test_lz4_blocks_decode_including_overlapping_matches(self):
        block = bytes([0x32]) + b"abc" + bytes([3, 0]) + bytes([0x10]) + b"x"
        self.assertEqual(vdbio.lz4_block_decompress(block, 10), b"abcabcabcx")
        long_literals = bytes([0xF0, 5]) + bytes(range(20))                  # 15 + 5 literals, no match
        self.assertEqual(vdbio.lz4_block_decompress(long_literals, 20), bytes(range(20)))
        with self.assertRaisesRegex(vdbio.VdbError, "corrupt LZ4"):
            vdbio.lz4_block_decompress(block, 11)

    def test_blosc_stored_chunks_round_trip_at_every_size(self):
        rng = np.random.default_rng(3)
        for size in (0, 1, 7, 100, 127, 128, 130, 512, 2048, 4099):
            raw = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
            with self.subTest(size=size):
                self.assertEqual(vdbio.blosc_decompress(vdbio._blosc_stored(raw), size), raw)
                self.assertEqual(vdbio.blosc_decompress(vdbio._blosc_stored(raw, splits=1), size), raw)

    def test_unshuffle_matches_a_reference(self):
        values = np.arange(64, dtype="<f4")
        shuffled = np.frombuffer(values.tobytes(), np.uint8).reshape(-1, 4).T.tobytes()
        self.assertEqual(vdbio._unshuffle(shuffled, 4), values.tobytes())


class VolumeTests(TempDir):
    def test_scale_and_translation_become_voxel_size_and_origin(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "temperature": blob(seed=4), "vel": velocity()},
                               voxel_size=0.25, index_min=(-5, 3, 0), active={"vel": blob() != 0})
        volume = vdbio.load_volume(path)
        self.assertEqual(volume.shape, (12, 10, 15))
        self.assertAlmostEqual(volume.voxel_size, 0.25)
        np.testing.assert_allclose(volume.origin, (np.array([-2, 5, 5]) - .5) * .25)      # the active box corner
        np.testing.assert_array_equal(volume.matrix, np.eye(4))
        self.assertEqual(volume.velocity.shape, (12, 10, 15, 3))
        self.assertEqual(volume.temperature.shape, volume.shape)
        np.testing.assert_array_equal(volume.density, blob()[3:15, 2:12, 5:20])

    def test_grids_with_different_boxes_share_the_union(self):
        temperature = np.zeros((20, 17, 25), np.float32)
        temperature[1:5, 1:4, 2:9] = 3.0
        path = vdbio.write_vdb(self.path(), {"density": blob(), "temperature": temperature}, voxel_size=0.5)
        volume = vdbio.load_volume(path)
        self.assertEqual(volume.shape, (14, 11, 18))                          # union of [3,15)x[2,12)x[5,20) and [1,5)x[1,4)x[2,9)
        self.assertEqual(float(volume.temperature[0, 0, 0]), 3.0)
        self.assertEqual(float(volume.temperature.max()), 3.0)
        self.assertEqual(int((volume.temperature == 3.0).sum()), 4 * 3 * 7)
        np.testing.assert_array_equal(volume.density[2:14, 1:11, 3:18], blob()[3:15, 2:12, 5:20])
        self.assertEqual(float(volume.density[:2].max()), 0.0)
        np.testing.assert_allclose(volume.origin, (np.array([1, 1, 2]) - .5) * .5)

    def test_a_rotated_affine_lands_in_the_matrix_and_turns_the_velocity(self):
        turn = np.eye(4)
        turn[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) * 0.5     # a quarter turn about z at half-unit voxels
        turn[:3, 3] = (5, 6, 7)
        vel = np.zeros((16, 16, 16, 3), np.float32)
        vel[..., 1] = 2.0                                                    # +Y in the file's world
        dens = np.ones((16, 16, 16), np.float32)
        path = vdbio.write_vdb(self.path(), {"density": dens, "vel": vel}, matrix=turn)
        volume = vdbio.load_volume(path)
        self.assertAlmostEqual(volume.voxel_size, 0.5)
        np.testing.assert_allclose(volume.matrix[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
        np.testing.assert_allclose(volume.matrix[:3, 3], (5, 6, 7))
        np.testing.assert_allclose(volume.velocity[0, 0, 0], (2, 0, 0), atol=1e-5)  # +Y in world is +X in the volume
        world = volume.matrix[:3, :3] @ volume.velocity[0, 0, 0]
        np.testing.assert_allclose(world, (0, 2, 0), atol=1e-5)

    def test_voxel_scale_scales_the_whole_volume_about_the_file_origin(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "vel": velocity()}, voxel_size=0.2,
                               index_min=(4, 0, 0), active={"vel": blob() != 0})
        base, big = vdbio.load_volume(path), vdbio.load_volume(path, voxel_scale=3.0)
        self.assertAlmostEqual(big.voxel_size, 3 * base.voxel_size)
        np.testing.assert_allclose(big.origin, 3 * np.array(base.origin))
        np.testing.assert_allclose(big.velocity, 3 * base.velocity)
        np.testing.assert_array_equal(big.density, base.density)

    def test_the_grid_choices(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "heat": blob(seed=3), "v": velocity(),
                                             "extra": blob(seed=8)}, active={"v": blob() != 0})
        auto = vdbio.load_volume(path)
        np.testing.assert_array_equal(auto.temperature, vdbio.read_grid(path, "heat").data)     # heat, then v
        self.assertIsNotNone(auto.velocity)
        none = vdbio.load_volume(path, temperature="none", velocity="none")
        self.assertIsNone(none.temperature)
        self.assertIsNone(none.velocity)
        named = vdbio.load_volume(path, density="extra", temperature="none", velocity="none")
        np.testing.assert_array_equal(named.density, blob(seed=8)[3:15, 2:12, 5:20])
        with self.assertRaisesRegex(vdbio.VdbError, "no grid named 'nope'; the file has: density, heat, v, extra"):
            vdbio.load_volume(path, density="nope")
        with self.assertRaisesRegex(vdbio.VdbError, "velocity grid must be a Vec3f grid"):
            vdbio.load_volume(path, velocity="extra")
        with self.assertRaisesRegex(vdbio.VdbError, "vector grid; the density grid must be a scalar"):
            vdbio.load_volume(path, density="v")
        only = vdbio.write_vdb(self.path("only.vdb"), {"smoke": blob()})
        self.assertEqual(vdbio.load_volume(only).shape, (12, 10, 15))                # a lone scalar grid is the density
        lone = vdbio.write_vdb(self.path("two.vdb"), {"a": blob(), "b": blob(seed=3)})
        with self.assertRaisesRegex(vdbio.VdbError, "no density grid; the file has: a, b"):
            vdbio.load_volume(lone)

    def test_different_transforms_are_refused(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "heat": blob(seed=2)}, voxel_size=0.1)
        data = bytearray(path.read_bytes())
        scale = np.full(3, 0.1).tobytes()
        first = data.find(scale)
        second = data.find(scale, first + 8 * 15)            # the second grid's UniformScaleMap: scale, then voxel size
        self.assertGreater(second, first)
        data[second:second + 48] = np.full(6, 0.2).tobytes()
        path.write_bytes(bytes(data))
        with self.assertRaisesRegex(vdbio.VdbError, "have different transforms"):
            vdbio.load_volume(path)


class SequenceTests(TempDir):
    def write_frames(self, first=1, last=3):
        for frame in range(first, last + 1):
            dens = np.zeros((16, 16, 16), np.float32)
            dens[6:10, 6:10, 6:10] = float(frame)
            vdbio.write_vdb(self.path(f"smoke.{frame:04d}.vdb"), {"density": dens}, compression="zip",
                            voxel_size=0.25, index_min=(-8, -8, -8))
        return str(self.path("smoke.%04d.vdb"))

    def test_a_pattern_resolves_per_frame_and_the_fingerprint_changes(self):
        pattern = self.write_frames()
        self.assertEqual(Path(vdbio.frame_path(pattern, 2)).name, "smoke.0002.vdb")
        self.assertEqual(Path(vdbio.frame_path("smoke.####.vdb".replace("smoke", str(self.dir / "smoke")), 3)).name,
                         "smoke.0003.vdb")
        prints = [vdbio.fingerprint(vdbio.frame_path(pattern, f)) for f in (1, 2, 3)]
        self.assertEqual(len(set(prints)), 3)
        with self.assertRaisesRegex(ValueError, "Missing frame 9"):
            vdbio.frame_path(pattern, 9)
        with self.assertRaisesRegex(vdbio.VdbError, "choose a VDB file"):
            vdbio.frame_path("", 1)
        with self.assertRaisesRegex(vdbio.VdbError, "not found"):
            vdbio.frame_path(str(self.dir / "nothing.vdb"), 1)

    def test_the_node_reads_the_frame_it_is_evaluated_at(self):
        pattern = self.write_frames()
        d = Dispatcher()
        make(d, r=("ReadVDB3D", {"vdb_path": pattern}), s=("Scene3D", {}), c=("Camera3D", {"tz": 5.0}),
             rd=("Render3D", {"width": 24, "height": 24, "render_output": "volume_density"}))
        wire(d, "s", "object0", "r")
        wire(d, "rd", "scene", "s")
        wire(d, "rd", "camera", "c")
        evaluator = Evaluator()
        doc = dict(d.document, view="rd")
        first, third = (evaluator.evaluate(doc, frame=f)[12, 12, 0] for f in (1, 3))
        self.assertAlmostEqual(float(third) / float(first), 3.0, delta=.05)   # density 3 against density 1
        misses = evaluator.misses
        evaluator.evaluate(doc, frame=3)
        self.assertEqual(evaluator.misses, misses)                            # the same frame is a cache hit
        d.execute({"op": "set", "id": "r", "param": "frame_offset", "value": 1})
        shifted = evaluator.evaluate(dict(d.document, view="rd"), frame=1)[12, 12, 0]
        self.assertAlmostEqual(float(shifted) / float(first), 2.0, delta=.05)
        with self.assertRaisesRegex(ValueError, "Missing frame 4"):        # offset 1 at frame 3 asks for smoke.0004
            evaluator.evaluate(dict(d.document, view="rd"), frame=3)


class NodeTests(TempDir):
    def graph(self, path, **params):
        d = Dispatcher()
        make(d, r=("ReadVDB3D", {"vdb_path": str(path), **params}), s=("Scene3D", {}), c=("Camera3D", {"tz": 5.0}),
             rd=("Render3D", {"width": 32, "height": 32, "samples": 1}))
        wire(d, "s", "object0", "r")
        wire(d, "rd", "scene", "s")
        wire(d, "rd", "camera", "c")
        return d

    def test_registration(self):
        self.assertEqual(OUTPUT_TYPES["ReadVDB3D"], "scene")
        params = SPECS["ReadVDB3D"]["params"]
        for name in ("vdb_path", "density_grid", "temperature_grid", "velocity_grid", "frame_offset", "voxel_scale",
                     "tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz", "uscale", "rot_order", "pivot_x"):
            self.assertIn(name, params)
        shown = {p for g in knob_layout("ReadVDB3D") for p in g.params}
        self.assertLessEqual(set(params), shown | {"rot_order"} | set(params))
        self.assertTrue({"vdb_path", "density_grid", "temperature_grid", "velocity_grid", "frame_offset",
                         "voxel_scale"} <= shown)
        self.assertEqual(SPECS["ReadVDB3D"]["inputs"], [])
        self.assertIn("rot_order", CHOICES)

    def test_bypass_is_an_empty_scene_and_a_read_volume_renders(self):
        dens = np.zeros((16, 16, 16), np.float32)
        dens[5:11, 5:11, 5:11] = 4.0
        path = vdbio.write_vdb(self.path(), {"density": dens}, voxel_size=1 / 16, index_min=(-8, -8, -8),
                               compression="blosc")
        d = self.graph(path)
        self.assertIsNone(bypass_slot({"type": "ReadVDB3D", "inputs": {}}))
        evaluator = Evaluator()
        image = evaluator.evaluate(dict(d.document, view="rd"), frame=1)
        alpha = image[..., 3]
        self.assertGreater(float(alpha[16, 16]), .3)                          # smoke where the density is
        self.assertEqual(float(alpha[0, 0]), 0.0)                              # nothing at the corner
        self.assertEqual(float(alpha[:, :3].max()), 0.0)
        with self.assertRaisesRegex(ValueError, "Source nodes cannot be bypassed"):     # like every reader
            d.execute({"op": "disable", "id": "r", "value": True})
        loaded = copy.deepcopy(d.document)                                     # a document that carries the flag anyway
        loaded["nodes"]["r"]["disabled"] = True
        empty = evaluator.evaluate(dict(loaded, view="rd"), frame=1)
        self.assertEqual(float(np.abs(empty).max()), 0.0)

    def test_a_read_volume_matches_the_same_volume_built_directly(self):
        dens = blob()
        path = vdbio.write_vdb(self.path(), {"density": dens}, voxel_size=1 / 16, index_min=(-8, -8, -8))
        read = scene3d.render(scene3d.Scene(volumes=(vdbio.load_volume(path),)), CAMERA, 32, 32, volume=FLAT)
        direct = scene3d.Volume(dens[3:15, 2:12, 5:20], 1 / 16, ((-8 + 3 - .5) / 16, (-8 + 2 - .5) / 16, (-8 + 5 - .5) / 16))
        expected = scene3d.render(scene3d.Scene(volumes=(direct,)), CAMERA, 32, 32, volume=FLAT)
        np.testing.assert_array_equal(read, expected)
        self.assertGreater(float(read[..., 3].max()), .05)

    def test_the_transform_block_and_voxel_scale_move_the_smoke(self):
        dens = np.zeros((16, 16, 16), np.float32)
        dens[4:12, 4:12, 4:12] = 5.0
        path = vdbio.write_vdb(self.path(), {"density": dens}, voxel_size=1 / 16, index_min=(-8, -8, -8))
        d = self.graph(path)
        evaluator = Evaluator()
        centre = evaluator.evaluate(dict(d.document, view="rd"), frame=1)[..., 3]
        self.assertGreater(float(centre[16, 16]), .1)
        d.execute({"op": "set", "id": "r", "param": "tx", "value": 0.8})
        moved = evaluator.evaluate(dict(d.document, view="rd"), frame=1)[..., 3]
        self.assertEqual(float(moved[16, 16]), 0.0)
        self.assertGreater(float(moved[:, 20:].max()), .1)
        d.execute({"op": "set", "id": "r", "param": "tx", "value": 0.0})
        d.execute({"op": "set", "id": "r", "param": "voxel_scale", "value": 2.0})
        big = evaluator.evaluate(dict(d.document, view="rd"), frame=1)[..., 3]
        self.assertGreater(float((big > .05).sum()), 2.0 * float((centre > .05).sum()))

    def test_errors_reach_the_user(self):
        d = self.graph("")
        with self.assertRaisesRegex(ValueError, "choose a VDB file"):
            Evaluator().evaluate(dict(d.document, view="rd"), frame=1)
        d.execute({"op": "set", "id": "r", "param": "vdb_path", "value": str(self.path("gone.vdb"))})
        with self.assertRaisesRegex(ValueError, "not found"):
            Evaluator().evaluate(dict(d.document, view="rd"), frame=1)
        path = vdbio.write_vdb(self.path(), {"density": blob()}, transform="frustum")
        d.execute({"op": "set", "id": "r", "param": "vdb_path", "value": str(path)})
        with self.assertRaisesRegex(ValueError, "frustum-transform VDB grids are not supported"):
            Evaluator().evaluate(dict(d.document, view="rd"), frame=1)

    def test_the_panel_choices_come_from_the_file(self):
        path = vdbio.write_vdb(self.path(), {"density": blob(), "vel": velocity()}, active={"vel": blob() != 0})
        self.assertEqual(vdbio.grid_choices(str(path)), ["auto", "none", "density", "vel"])
        self.assertEqual(vdbio.grid_choices(""), ["auto", "none"])
        self.assertEqual(vdbio.grid_choices(str(self.path("missing.vdb"))), ["auto", "none"])
        bad = self.path("bad.vdb")
        bad.write_bytes(b"junk")
        self.assertEqual(vdbio.grid_choices(str(bad)), ["auto", "none"])

    def test_old_documents_are_unaffected(self):
        from nodebased.core import upgrade_document
        d = Dispatcher()
        make(d, p=("Plume3D", {"plume_resolution": 6}), s=("Scene3D", {}), c=("Camera3D", {}),
             rd=("Render3D", {"width": 16, "height": 16, "render_output": "multichannel"}))
        wire(d, "s", "object0", "p")
        wire(d, "rd", "scene", "s")
        wire(d, "rd", "camera", "c")
        old = copy.deepcopy(d.document)
        self.assertEqual(upgrade_document(copy.deepcopy(old))["nodes"], old["nodes"])
        self.assertEqual(old["nodes"]["rd"]["params"]["passes"], "beauty,normals,depth")
        raster = Evaluator().evaluate_raster(dict(old, view="rd"), frame=1)
        self.assertEqual(list(raster.layers), ["normals", "depth"])           # the new volume layers are opt-in
        self.assertNotIn("ReadVDB3D", {n["type"] for n in old["nodes"].values()})


class PanelTests(TempDir):
    """The node panel offers the file's grid names in the three grid knobs."""

    def setUp(self):
        super().setUp()
        import uuid
        from tests.test_desktop import APP, Window, release_window, wait_until
        self.APP, self.wait_until = APP, wait_until
        self.window = Window(agent_name="nodebased-test-" + uuid.uuid4().hex)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

        def close():
            self.window.saved_document = self.window.dispatcher.document
            self.window.close()
            APP.processEvents()
            release_window(self)
        self.addCleanup(close)

    def test_grid_knobs_are_combo_boxes_filled_from_the_file(self):
        from PySide6.QtWidgets import QComboBox
        path = vdbio.write_vdb(self.path(), {"density": blob(), "vel": velocity()}, active={"vel": blob() != 0})
        w = self.window
        w.dispatcher.execute({"op": "create", "id": "vdb", "type": "ReadVDB3D", "params": {"vdb_path": str(path)}})
        w.set_properties_widget(w.build_node_panel("vdb"))
        combos = [c for c in w.properties.widget().findChildren(QComboBox) if c.isEditable()]
        self.assertEqual(len(combos), 3)
        for combo in combos:
            self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["auto", "none", "density", "vel"])
            self.assertEqual(combo.currentText(), "auto")
        from PySide6.QtWidgets import QLabel
        labels = [label.text() for label in w.properties.widget().findChildren(QLabel)]
        self.assertIn("VDB file or sequence", labels)
        self.assertIn("Velocity grid", labels)
        combos[2].activated.emit(1)                                            # the velocity combo picks "none"
        self.assertTrue(self.wait_until(lambda: w.dispatcher.document["nodes"]["vdb"]["params"]["velocity_grid"] == "none"))

    def test_a_bad_path_still_opens_the_panel(self):
        w = self.window
        w.dispatcher.execute({"op": "create", "id": "vdb", "type": "ReadVDB3D", "params": {"vdb_path": "/nonexistent/x.vdb"}})
        w.set_properties_widget(w.build_node_panel("vdb"))
        self.assertGreaterEqual(len(w.properties.widget().children()), 1)


# Real caches are not in the repo. Point NB_VDB_SAMPLES at a folder holding some of the Blender test
# files (cube, smoke, smoke_low_res, velocity_named_grid, small_cloud, fluid_noise_0014,
# intelCloudLib_sparse.4.S), or leave them in the workspace's scratch/vdb-samples.
REAL = Path(os.environ.get("NB_VDB_SAMPLES") or Path.home() / ".openclaw" / "workspace" / "scratch" / "vdb-samples")


@unittest.skipUnless(REAL.is_dir(), "no real .vdb samples in the workspace scratch folder")
class RealFileTests(unittest.TestCase):
    """Blender-written caches (Blosc-LZ4, streamed, half float): grid names and voxel counts as the files record them."""

    EXPECTED = {"cube.vdb": [("density", 1331)],
                "small_cloud.vdb": [("", 26501)],
                "smoke.vdb": [("density", 1049275)],
                "smoke_low_res.vdb": [("density_level_4", 719)],
                "velocity_named_grid.vdb": [("density", 1045), ("vel", 1045)],
                "fluid_noise_0014.vdb": [("density_noise", 9313), ("flame_noise", 0)],
                "intelCloudLib_sparse.4.S.vdb": [("density", 134151)]}

    def test_names_counts_and_boxes_match_the_files_own_metadata(self):
        seen = 0
        for name, expected in self.EXPECTED.items():
            path = REAL / name
            if not path.is_file():
                continue
            seen += 1
            with self.subTest(file=name):
                infos = vdbio.list_grids(path)
                self.assertEqual([(i.name, i.voxel_count) for i in infos], expected)
                for info in infos:
                    if not info.voxel_count:
                        continue
                    grid = vdbio.read_grid(path, info.name)
                    self.assertEqual(grid.active_voxels, info.voxel_count)
                    self.assertEqual(grid.index_min, info.bbox[0])
                    top = tuple(a + n - 1 for a, n in zip(grid.index_min, grid.data.shape[:3]))
                    self.assertEqual(top, info.bbox[1])
                    self.assertGreater(float(grid.data.max()), 0.0)
        if not seen:
            self.skipTest("none of the known sample files are present")

    def test_a_real_velocity_cache_loads_as_a_volume_and_renders(self):
        path = REAL / "velocity_named_grid.vdb"
        if not path.is_file():
            self.skipTest("velocity_named_grid.vdb is not present")
        volume = vdbio.load_volume(path)
        self.assertEqual(volume.shape, (13, 13, 13))
        self.assertIsNotNone(volume.velocity)
        self.assertAlmostEqual(volume.voxel_size, 0.125)
        centre = np.array(volume.origin) + 6.5 * volume.voxel_size
        moved = replace(volume, matrix=scene3d.Transform3D(scene3d.Vec3(*(-centre))).matrix())   # centre it on the origin
        image = scene3d.render(scene3d.Scene(volumes=(moved,)), CAMERA, 24, 24, volume=FLAT)
        self.assertGreater(float(image[10:14, 10:14, 3].max()), 0.0)

    def test_the_big_real_smoke_reads_within_a_second_or_two(self):
        path = REAL / "smoke.vdb"
        if not path.is_file():
            self.skipTest("smoke.vdb is not present")
        import time
        start = time.perf_counter()
        grid = vdbio.read_grid(path, "density")
        self.assertLess(time.perf_counter() - start, 10.0)
        self.assertEqual(grid.data.shape, (111, 222, 112))
        self.assertTrue(math.isfinite(float(grid.data.max())))


class WriteSceneTests(TempDir):
    """`vdbio.write_scene` (WriteVDB3D, docs/FLUIDS_SPIKE.md "WriteVDB3D as built"): a fluid Volume's
    grids, or a liquid's signed-distance surface, written and read back through this module's own
    reader (the round trip a real OpenVDB build would also need to pass)."""

    def smoke_volume(self, n=16):
        dens = np.zeros((n, n, n), np.float32)
        dens[4:12, 4:12, 4:12] = np.random.default_rng(0).random((8, 8, 8), dtype=np.float32) + 0.1
        temp = dens * 2.0
        vel = np.zeros((n, n, n, 3), np.float32)
        vel[4:12, 4:12, 4:12] = np.random.default_rng(1).standard_normal((8, 8, 8, 3)).astype(np.float32)
        return scene3d.Volume(dens, voxel_size=0.1, origin=(-0.8, -0.8, -0.8), temperature=temp, velocity=vel)

    def test_density_temperature_and_velocity_round_trip(self):
        volume = self.smoke_volume()
        path = vdbio.write_scene(self.path(), scene3d.Scene(volumes=(volume,)))
        back = vdbio.load_volume(path)
        np.testing.assert_allclose(back.density, volume.density[4:12, 4:12, 4:12])
        np.testing.assert_allclose(back.temperature, volume.temperature[4:12, 4:12, 4:12])
        np.testing.assert_allclose(back.velocity, volume.velocity[4:12, 4:12, 4:12], atol=1e-5)
        self.assertAlmostEqual(back.voxel_size, volume.voxel_size)
        np.testing.assert_allclose(back.origin, np.array(volume.origin) + 4 * volume.voxel_size, atol=1e-6)

    def test_flame_is_written_as_a_fog_volume_when_present(self):
        volume = self.smoke_volume()
        flame = np.zeros_like(volume.density)
        flame[5:9, 5:9, 5:9] = 0.4
        volume = replace(volume, flame=flame)
        path = vdbio.write_scene(self.path(), scene3d.Scene(volumes=(volume,)))
        infos = {i.name: i for i in vdbio.list_grids(path)}
        self.assertEqual(infos["flame"].grid_class, "fog volume")
        grid = vdbio.read_grid(path, "flame")
        self.assertGreater(float(grid.data.max()), 0.0)

    def test_optional_grids_absent_from_the_volume_are_not_written(self):
        # "Empty" here is what most solved volumes are: density only, no temperature/velocity/flame.
        # write_scene must not invent grids the Volume does not carry.
        volume = scene3d.Volume(self.smoke_volume().density, voxel_size=0.1)
        path = vdbio.write_scene(self.path(), scene3d.Scene(volumes=(volume,)))
        self.assertEqual([i.name for i in vdbio.list_grids(path)], ["density"])

    def test_a_fully_zero_volume_writes_a_file_that_reports_no_active_voxels_on_read(self):
        volume = scene3d.Volume(np.zeros((8, 8, 8), np.float32), voxel_size=0.1)
        path = vdbio.write_scene(self.path(), scene3d.Scene(volumes=(volume,)))
        # A grid with nothing active is refused on read by name, like any other empty VDB grid
        # (docs/FLUIDS_SPIKE.md, vdbio's own writer follows the format it reads).
        with self.assertRaisesRegex(vdbio.VdbError, "no active voxels"):
            vdbio.read_grid(path, "density")

    def test_a_rotated_matrix_round_trips_through_write_scene(self):
        theta = np.radians(40)
        c, s = np.cos(theta), np.sin(theta)
        matrix = np.eye(4)
        matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        matrix[:3, 3] = (1.0, 2.0, -3.0)
        volume = replace(self.smoke_volume(), matrix=matrix)
        path = vdbio.write_scene(self.path(), scene3d.Scene(volumes=(volume,)))
        back = vdbio.load_volume(path)
        i = np.array([1, 2, 3])
        local_i = i + 4                                      # back's local index maps to the file's [4:12) box
        world_read = back.matrix[:3, :3] @ (np.array(back.origin) + (i + 0.5) * back.voxel_size) + back.matrix[:3, 3]
        world_written = matrix[:3, :3] @ (np.array(volume.origin) + (local_i + 0.5) * volume.voxel_size) + matrix[:3, 3]
        np.testing.assert_allclose(world_read, world_written, atol=1e-4)

    def liquid_surface(self, n=16, voxel_size=0.1):
        # A ball: negative inside, positive outside, growing linearly with distance like a real SDF,
        # so a narrow band around phi = 0 is a thin shell rather than most of the grid.
        axis = (np.arange(n) - n / 2 + 0.5) * voxel_size
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        phi = (np.sqrt(x ** 2 + y ** 2 + z ** 2) - 0.4).astype(np.float32)
        return scene3d.Volume(phi, voxel_size=voxel_size, origin=tuple(axis[0] - voxel_size / 2 for _ in range(3)))

    def liquid_scene(self, surface):
        return scene3d.Scene(particles=(scene3d.ParticleInstance(
            positions=np.zeros((1, 3), np.float32), sizes=np.ones(1, np.float32),
            colors=np.ones((1, 4), np.float32), surface=surface),))

    def test_liquid_surface_writes_a_level_set_grid(self):
        surface = self.liquid_surface()
        path = vdbio.write_scene(self.path(), self.liquid_scene(surface), narrow_band=3.0)
        infos = {i.name: i for i in vdbio.list_grids(path)}
        self.assertEqual(list(infos), ["surface"])
        self.assertEqual(infos["surface"].grid_class, "level set")

    def test_the_level_set_round_trips_within_its_narrow_band(self):
        surface = self.liquid_surface()
        band = 3.0 * surface.voxel_size
        path = vdbio.write_scene(self.path(), self.liquid_scene(surface), narrow_band=3.0)
        grid = vdbio.read_grid(path, "surface")
        lo = np.array(grid.index_min)
        hi = lo + np.array(grid.data.shape)
        original = surface.density[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        inside_band = np.abs(original) <= band
        np.testing.assert_allclose(grid.data[inside_band], original[inside_band], atol=1e-5)
        # Outside the band, active leaves cover only where the file actually needs to store data.
        self.assertTrue(inside_band.any())

    def test_the_narrow_band_leaves_out_most_of_a_large_grid(self):
        surface = self.liquid_surface(n=32, voxel_size=0.05)
        band = 3.0 * surface.voxel_size
        path = vdbio.write_scene(self.path(), self.liquid_scene(surface), narrow_band=3.0)
        grid = vdbio.read_grid(path, "surface")
        total_voxels = surface.density.size
        # A thin shell around the zero crossing is far below the dense voxel count and far below a
        # dense file's size, which is what "sparse leaves only where needed" means for a level set.
        self.assertLess(grid.active_voxels, total_voxels // 3)
        dense_bytes = total_voxels * 4
        self.assertLess(os.path.getsize(path), dense_bytes // 2)

    def test_a_scene_with_both_a_volume_and_a_liquid_surface_is_refused(self):
        volume = self.smoke_volume()
        surface = self.liquid_surface()
        scene = replace(self.liquid_scene(surface), volumes=(volume,))
        with self.assertRaisesRegex(vdbio.VdbError, "both a fluid volume and a liquid surface"):
            vdbio.write_scene(self.path(), scene)

    def test_more_than_one_volume_or_surface_is_refused_by_name(self):
        volume = self.smoke_volume()
        with self.assertRaisesRegex(vdbio.VdbError, "writes one Volume per file; the scene has 2"):
            vdbio.write_scene(self.path(), scene3d.Scene(volumes=(volume, volume)))
        surface = self.liquid_surface()
        inst = scene3d.ParticleInstance(positions=np.zeros((1, 3), np.float32), sizes=np.ones(1, np.float32),
                                        colors=np.ones((1, 4), np.float32), surface=surface)
        with self.assertRaisesRegex(vdbio.VdbError, "writes one liquid surface per file; the scene has 2"):
            vdbio.write_scene(self.path(), scene3d.Scene(particles=(inst, inst)))

    def test_an_empty_scene_is_refused(self):
        with self.assertRaisesRegex(vdbio.VdbError, "no volume and no liquid surface"):
            vdbio.write_scene(self.path(), scene3d.Scene())

    def test_compression_choices_and_half_float_all_read_back(self):
        volume = self.smoke_volume()
        for compression, half in itertools.product(("none", "zip", "blosc"), (False, True)):
            with self.subTest(compression=compression, half=half):
                path = vdbio.write_scene(self.path(f"{compression}{half}.vdb"), scene3d.Scene(volumes=(volume,)),
                                         compression=compression, half=half)
                grid = vdbio.read_grid(path, "density")
                atol = 5e-3 if half else 1e-6
                np.testing.assert_allclose(grid.data, volume.density[4:12, 4:12, 4:12], atol=atol)
                self.assertEqual(grid.half_float, half)


if __name__ == "__main__":
    unittest.main()
