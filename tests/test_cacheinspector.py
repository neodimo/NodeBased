import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import cacheinspector, simcache


def _put_particle_frame(cache, run, frame, count):
    positions = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    state = simcache.State({"position": positions, "id": np.arange(count, dtype=np.int64)},
                           {"substep_count": 1})
    cache.put(run, frame, state)


def _put_volume_frame(cache, run, frame, fill):
    density = np.zeros((16, 16, 16), np.float32)
    density[:8, :8, :8] = fill   # exactly one occupied 8-cube tile out of eight
    state = simcache.State({"density": density}, {})
    cache.put(run, frame, state)


class ListFramesTests(unittest.TestCase):
    def test_particle_frames_report_particle_count_and_field_stats(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            _put_particle_frame(cache, "runA", 1, 5)
            _put_particle_frame(cache, "runA", 2, 7)
            entries = cacheinspector.list_frames(cache, "runA", 1, 2)
            self.assertEqual([e.frame for e in entries], [1, 2])
            self.assertTrue(all(e.present for e in entries))
            self.assertEqual(entries[0].stats["particle_count"], 5)
            self.assertEqual(entries[1].stats["particle_count"], 7)
            self.assertAlmostEqual(entries[0].stats["fields"]["position"]["min"], 0.0)
            self.assertAlmostEqual(entries[0].stats["fields"]["position"]["max"], 14.0)

    def test_missing_frames_are_reported_as_a_gap(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            _put_particle_frame(cache, "runA", 1, 3)
            _put_particle_frame(cache, "runA", 3, 3)
            entries = cacheinspector.list_frames(cache, "runA", 1, 3)
            self.assertEqual([e.frame for e in entries], [1, 2, 3])
            self.assertTrue(entries[0].present)
            self.assertFalse(entries[1].present)
            self.assertIsNone(entries[1].stats)
            self.assertTrue(entries[2].present)

    def test_volume_frames_report_voxel_count_and_active_tiles(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            _put_volume_frame(cache, "runB", 1, fill=2.0)
            entries = cacheinspector.list_frames(cache, "runB", 1, 1)
            stats = entries[0].stats
            self.assertEqual(stats["voxel_count"], 16 * 16 * 16)
            self.assertEqual(stats["active_tiles"], 1)
            self.assertAlmostEqual(stats["fields"]["density"]["max"], 2.0)

    def test_disk_size_is_reported_once_written(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            _put_particle_frame(cache, "runA", 1, 3)
            entries = cacheinspector.list_frames(cache, "runA", 1, 1)
            self.assertIsNotNone(entries[0].disk_bytes)
            self.assertGreater(entries[0].disk_bytes, 0)

    def test_solve_ms_is_recorded_by_solve_to_frame_not_by_a_plain_put(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)

            def initial_state(seed):
                return simcache.State({"x": np.zeros(1, np.float32)}, {})

            def step(state, frame, substep, seed):
                return simcache.State({"x": state.arrays["x"] + 1}, {})

            simcache.solve_to_frame(cache, "runC", 2, 1, 1, 0, initial_state, step)
            entries = cacheinspector.list_frames(cache, "runC", 1, 2)
            self.assertIsNotNone(entries[0].solve_ms)
            self.assertGreaterEqual(entries[0].solve_ms, 0.0)

            # A frame that was only ever `put` directly (never solved) has no recorded time.
            _put_particle_frame(cache, "runA", 9, 1)
            direct = cacheinspector.list_frames(cache, "runA", 9, 9)
            self.assertIsNone(direct[0].solve_ms)


class InvalidateFromFrameTests(unittest.TestCase):
    def test_invalidate_removes_exactly_the_requested_frames(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            for frame in range(1, 6):
                _put_particle_frame(cache, "runA", frame, 2)
            removed = cacheinspector.invalidate_from_frame(cache, "runA", 3)
            self.assertEqual(removed, [3, 4, 5])
            remaining = cacheinspector.list_frames(cache, "runA", 1, 5)
            self.assertEqual([e.frame for e in remaining if e.present], [1, 2])

    def test_invalidate_removes_disk_files_too(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            _put_particle_frame(cache, "runA", 1, 2)
            folder = cacheinspector.cache_folder(cache, "runA")
            # Frames are packed .nbc files since Fluids 8 (P1); older caches hold .npz. Either kind must go.
            frames = lambda: [p for p in folder.iterdir() if p.suffix in (".nbc", ".npz")]
            self.assertTrue(frames())
            cacheinspector.invalidate_from_frame(cache, "runA", 1)
            self.assertFalse(frames())


class CacheFolderTests(unittest.TestCase):
    def test_folder_is_none_for_a_memory_only_cache(self):
        cache = simcache.SimCache(root=None)
        self.assertIsNone(cacheinspector.cache_folder(cache, "runA"))

    def test_folder_path_matches_the_cache_layout(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            folder = cacheinspector.cache_folder(cache, "abcdef")
            self.assertEqual(folder, Path(root) / "ab" / "abcdef")


if __name__ == "__main__":
    unittest.main()
