"""Lane L2, artist tools 1, part 4: the slice viewer and cache inspector docks wired to the
selected node in the real desktop `Window` (not just the standalone widgets)."""
import os
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from nodebased.app import Window
from tests.test_desktop import WAIT_TIMEOUT, release_window
from tests.waiting import wait_until


class ArtistToolsWiringTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = "nodebased-test-" + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f"no first frame cooked within {WAIT_TIMEOUT:.0f}s")
        self.window.slice_dock.show()
        self.window.cache_inspector_dock.show()

    def tearDown(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()

    def test_selecting_a_fluid_cache_node_fills_the_slice_view_and_cache_inspector(self):
        w = self.window
        w.dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "fsrc", "type": "FluidSource3D", "params": {}},
            {"op": "create", "id": "fsol", "type": "FluidSolver3D",
             "params": {"division_size": 0.25, "bounds_min_x": -1.0, "bounds_min_y": 0.0,
                        "bounds_min_z": -1.0, "bounds_max_x": 1.0, "bounds_max_y": 2.0,
                        "bounds_max_z": 1.0, "start_frame": 1, "substeps": 1}},
            {"op": "create", "id": "fcache", "type": "FluidCache3D", "params": {}},
            {"op": "connect", "id": "fsol", "input": "fluid", "source": "fsrc"},
            {"op": "connect", "id": "fcache", "input": "volume", "source": "fsol"},
        ]})
        w.inspect("fcache")
        self.assertIsNotNone(w.slice_view.volume)
        self.assertGreater(w.cache_inspector.table.rowCount(), 0)
        # A frame that has just been evaluated for the panel shows as cached, not missing.
        statuses = [w.cache_inspector.table.item(row, 1).text()
                   for row in range(w.cache_inspector.table.rowCount())]
        self.assertIn("cached", statuses)

    def test_selecting_a_particle_cache_node_fills_the_cache_inspector(self):
        w = self.window
        w.dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "pe", "type": "ParticleEmitter3D",
             "params": {"emit_rate": 20.0, "life": 24.0, "start_frame": 1}},
            {"op": "create", "id": "pc", "type": "ParticleCache3D", "params": {}},
            {"op": "connect", "id": "pc", "input": "particles", "source": "pe"},
        ]})
        w.inspect("pc")
        self.assertGreater(w.cache_inspector.table.rowCount(), 0)

    def test_selecting_an_unrelated_node_clears_the_slice_view(self):
        w = self.window
        w.dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "fsrc2", "type": "FluidSource3D", "params": {}},
            {"op": "create", "id": "fsol2", "type": "FluidSolver3D",
             "params": {"division_size": 0.25, "bounds_min_x": -1.0, "bounds_min_y": 0.0,
                        "bounds_min_z": -1.0, "bounds_max_x": 1.0, "bounds_max_y": 2.0,
                        "bounds_max_z": 1.0, "start_frame": 1, "substeps": 1}},
            {"op": "create", "id": "fcache2", "type": "FluidCache3D", "params": {}},
            {"op": "connect", "id": "fsol2", "input": "fluid", "source": "fsrc2"},
            {"op": "connect", "id": "fcache2", "input": "volume", "source": "fsol2"},
        ]})
        w.inspect("fcache2")
        self.assertIsNotNone(w.slice_view.volume)
        w.inspect("grade")   # the demo document's Grade node: not a volume-producing kind
        self.assertIsNone(w.slice_view.volume)


if __name__ == "__main__":
    unittest.main()
