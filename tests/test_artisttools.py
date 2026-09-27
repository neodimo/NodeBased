import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from nodebased import simcache, simstats
from nodebased.artisttools import CacheInspectorPanel, SimStatsOverlay, SliceView
from nodebased.scene3d import Volume


def _analytic_volume():
    nx, ny, nz = 6, 6, 6
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    density = (ii + 10 * jj + 100 * kk).astype(np.float32)
    velocity = np.zeros((nx, ny, nz, 3), np.float32)
    velocity[..., 0] = ii
    return Volume(density, voxel_size=1.0, velocity=velocity)


class DesktopTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])


class SliceViewTests(DesktopTestCase):
    def setUp(self):
        self.view = SliceView()
        self.view.resize(240, 240)
        self.volume = _analytic_volume()

    def test_renders_each_available_field_with_a_readout_matching_the_voxel(self):
        self.view.set_volume(self.volume)
        for field in ("density", "speed"):
            self.view.field_combo.setCurrentText(field)
            self.assertFalse(self.view.image_label.pixmap().isNull())
            for (row, col) in [(0, 0), (3, 4), (5, 5)]:
                got = self.view.value_at(row, col)
                expected = float(getattr(self.volume, "density")[row, col, self.view.index]) if field == "density" \
                    else float(np.linalg.norm(self.volume.velocity[row, col, self.view.index]))
                self.assertAlmostEqual(got, expected, places=4)

    def test_moving_the_slice_position_changes_the_rendered_image(self):
        self.view.set_volume(self.volume)
        self.view.field_combo.setCurrentText("density")
        image_a = self.view.image_label.pixmap().toImage().copy()
        self.view.set_index(5)
        image_b = self.view.image_label.pixmap().toImage()
        self.assertNotEqual(image_a, image_b)

    def test_clearing_the_volume_clears_the_image(self):
        self.view.set_volume(self.volume)
        self.view.set_volume(None)
        self.assertTrue(self.view.image_label.pixmap() is None or self.view.image_label.pixmap().isNull())


def _put_volume_frame(cache, run, frame, fill):
    density = np.zeros((16, 16, 16), np.float32)
    density[:8, :8, :8] = fill
    state = simcache.State({"density": density}, {})
    cache.put(run, frame, state)


class CacheInspectorPanelTests(DesktopTestCase):
    def test_lists_frames_and_stats_matching_the_cache(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            _put_volume_frame(cache, "runA", 1, fill=2.0)
            _put_volume_frame(cache, "runA", 2, fill=3.0)
            panel = CacheInspectorPanel()
            panel.set_cache(cache, "runA", 1, 3)
            self.assertEqual(panel.table.rowCount(), 3)
            self.assertEqual(panel.table.item(0, 1).text(), "cached")
            self.assertEqual(panel.table.item(1, 1).text(), "cached")
            self.assertEqual(panel.table.item(2, 1).text(), "missing")
            self.assertEqual(panel.table.item(0, 2).text(), str(16 * 16 * 16))

    def test_invalidate_from_frame_removes_exactly_those_frames(self):
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            for frame in range(1, 5):
                _put_volume_frame(cache, "runA", frame, fill=1.0)
            panel = CacheInspectorPanel()
            panel.set_cache(cache, "runA", 1, 4)
            panel.invalidate_spin.setValue(3)
            panel._on_invalidate_clicked()
            statuses = [panel.table.item(row, 1).text() for row in range(panel.table.rowCount())]
            self.assertEqual(statuses, ["cached", "cached", "missing", "missing"])


class SimStatsOverlayTests(DesktopTestCase):
    def test_overlay_text_updates_during_a_short_solve(self):
        overlay = SimStatsOverlay()
        self.assertTrue(overlay.isHidden())

        def initial_state(seed):
            return simcache.State({"x": np.zeros(1, np.float32)}, {})

        def step(state, frame, substep, seed):
            return simcache.State({"x": state.arrays["x"] + 1}, {})

        seen = []

        def on_snapshot(snapshot):
            overlay.update_snapshot(snapshot)
            seen.append(overlay.text())

        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            simstats.solve_with_stats(cache, "run", 3, 1, 2, 0, initial_state, step, on_snapshot)
        self.assertFalse(overlay.isHidden())
        self.assertGreater(len(set(seen)), 1)


if __name__ == "__main__":
    unittest.main()
