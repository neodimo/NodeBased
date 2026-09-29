"""The live viewport's fluid cache readout and its saved display toggle."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from nodebased import cachecontext, scene3d
from nodebased.imaging import Evaluator
from nodebased.viewport3d import Viewport3D
from tests.test_fluid3d_nodes import plume

APP = QApplication.instance() or QApplication([])


class FluidStatsViewportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        settings_path = str(Path(self.tmp.name) / "preferences.ini")
        self.settings = QSettings(settings_path, QSettings.Format.IniFormat)
        self.patcher = patch("nodebased.viewport3d.QSettings", return_value=self.settings)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_known_plume_cache_numbers_and_hidden_cost(self):
        document = plume(pressure="cpu").document
        document["time"]["current"] = 1
        evaluator = Evaluator()
        volume = evaluator.evaluate_raster(document, "c", frame=1, typed=True)
        store, run, *_ = cachecontext.cache_context(evaluator, document, "c", 1)
        state = store.get(run, 1)
        widget = Viewport3D()
        self.addCleanup(widget.close)
        widget.set_document(document)
        widget.selected_key = "c"
        widget._evaluator = evaluator
        self.assertFalse(widget.show_sim_stats)
        with patch.object(widget, "_sim_stats", side_effect=AssertionError("hidden overlay sampled")):
            widget.show()
            APP.processEvents()
        widget.set_show_sim_stats(True)
        text = widget._sim_stats()
        self.assertIn("16×24×16 grid", text)
        self.assertIn("0.125 voxel", text)
        self.assertIn(f"{np.count_nonzero(state.arrays['density']):,} active", text)
        self.assertIn(f"{len(store.frames(run))} cached frames", text)
        self.assertIn("cpu · CPU", text)
        self.assertIn(f"Solve {store.solve_ms(volume.stream.run, 1):.2f} ms", text)
        self.assertIn(f"Memory {state.nbytes / 1048576:.2f} MB", text)
        widget.repaint()
        widget.repaint()
        self.assertLess(widget.sim_stats_draw_ms, 1.0)

    def test_toggle_persists_and_rendered_scene_pixels_do_not_change(self):
        widget = Viewport3D()
        self.addCleanup(widget.close)
        self.assertFalse(widget.show_sim_stats)
        scene = scene3d.Scene()
        camera = widget._camera()
        before = scene3d.render(scene, camera, 8, 8, (0.025, 0.025, 0.03, 1.0))
        widget.set_show_sim_stats(True)
        self.settings.sync()
        reopened = Viewport3D()
        self.addCleanup(reopened.close)
        self.assertTrue(reopened.show_sim_stats)
        after = scene3d.render(scene, camera, 8, 8, (0.025, 0.025, 0.03, 1.0))
        np.testing.assert_array_equal(before, after)
        reopened.set_show_sim_stats(False)
        self.settings.sync()
        again = Viewport3D()
        self.addCleanup(again.close)
        self.assertFalse(again.show_sim_stats)


if __name__ == "__main__":
    unittest.main()
