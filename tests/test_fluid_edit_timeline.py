"""Progressive simulation edit state and its artist-facing timeline marks."""
import unittest
import threading
from types import MethodType, SimpleNamespace

from PySide6.QtWidgets import QApplication, QLabel
from nodebased.app import Window
from nodebased.core import demo_document
from nodebased.playback import FrameRequest, PlaybackQueue
from nodebased.timeline import TimelineBar

APP = QApplication.instance() or QApplication([])


class FluidEditTimelineTests(unittest.TestCase):
    def test_only_a_view_reached_from_a_fluid_node_uses_the_bake_lane(self):
        document = demo_document()
        document["nodes"]["fluid"] = {"type": "FluidSolver3D", "inputs": {}, "params": {},
                                        "disabled": False}
        document["nodes"]["render"] = {"type": "Render3D", "inputs": {"scene": "fluid"},
                                         "params": {}, "disabled": False}
        document["view"] = "render"
        self.assertTrue(Window._view_uses_fluid(document))
        document["view"] = "grade"
        self.assertFalse(Window._view_uses_fluid(document))

    def test_only_edits_in_the_solver_dependency_scope_restart_the_bake(self):
        document = demo_document()
        document["nodes"]["fluid"] = {"type": "FluidSolver3D", "inputs": {}, "params": {},
                                        "disabled": False}
        document["nodes"]["render"] = {"type": "Render3D", "inputs": {"scene": "fluid"},
                                         "params": {}, "disabled": False}
        document["view"] = "render"
        self.assertTrue(Window._command_changes_fluid_simulation(
            document, {"op": "set", "id": "fluid", "param": "iterations", "value": 3}))
        document["nodes"]["light"] = {"type": "Light3D", "inputs": {}, "params": {}, "disabled": False}
        document["nodes"]["render"]["inputs"]["camera"] = "light"
        self.assertFalse(Window._command_changes_fluid_simulation(
            document, {"op": "set", "id": "light", "param": "intensity", "value": 2}))

    def test_bake_progress_is_a_contiguous_prefix(self):
        state = {"first": 1, "last": 5, "baked": {1, 2, 4}}
        self.assertEqual(Window._contiguous_fluid_frames(state), 2)
        state["baked"].add(3)
        self.assertEqual(Window._contiguous_fluid_frames(state), 4)

    def test_ruler_keeps_simulation_frames_separate_from_display_cache(self):
        bar = TimelineBar()
        bar.setRange(1, 5)
        bar.set_marks({1}, set())
        bar.set_simulation_frames({1, 2, 3})
        self.assertEqual(bar.cached_frames, {1})
        self.assertEqual(bar.simulation_frames, {1, 2, 3})

    def test_visible_edit_frame_starts_an_ordered_non_display_bake(self):
        document = demo_document()
        document["time"].update(first=1, last=3, current=1)
        fake = SimpleNamespace(dispatcher=SimpleNamespace(document=document), generation=7,
                               fluid_bake_status=QLabel(), frame_slider=TimelineBar(),
                               preview_queue=PlaybackQueue(), playing=False, _fluid_edit_bake=None,
                               _fluid_stale=False, _fluid_bake_identity=Window._fluid_bake_identity,
                               _contiguous_fluid_frames=Window._contiguous_fluid_frames)
        fake._set_fluid_stale = MethodType(Window._set_fluid_stale, fake)
        Window._begin_fluid_edit_bake(fake)
        Window._advance_fluid_edit_bake(
            fake, FrameRequest(7, 1, True, document, 1), object(), threading.Event())
        self.assertEqual(fake._fluid_edit_bake["baked"], {1})
        self.assertEqual(fake.frame_slider.simulation_frames, {1})
        request, _ = fake.preview_queue.take()
        self.assertEqual((request.frame, request.display), (2, False))
        Window._advance_fluid_edit_bake(fake, request, object(), threading.Event())
        request, _ = fake.preview_queue.take()
        self.assertEqual((request.frame, request.display), (3, False))
        self.assertIn("SIM 2/3", fake.fluid_bake_status.text())


if __name__ == "__main__":
    unittest.main()
