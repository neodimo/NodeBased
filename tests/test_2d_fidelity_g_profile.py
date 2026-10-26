"""Fidelity step G, part 2: Profile takes Nuke's frame range, channels and profile panel. The range text is read by
nodebased/framerange.py (the Viewer's frame range uses it too)."""
import copy
import unittest

import numpy as np

from nodebased import framerange, profilenode
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_c4 import Graph


class FrameRangeTests(unittest.TestCase):
    def test_the_forms_of_range_text(self):
        for text, want in (("1-5", [1, 2, 3, 4, 5]), ("3", [3]), ("1-10x3", [1, 4, 7, 10]), ("2-4, 9", [2, 3, 4, 9]),
                           ("1-3 7-8", [1, 2, 3, 7, 8]), ("5-1", [5, 4, 3, 2, 1]), ("10-2x4", [10, 6, 2]), ("-3--1", [-3, -2, -1]),
                           ("1,1,2", [1, 2]), ("4-6X2", [4, 6])):
            with self.subTest(text=text):
                self.assertEqual(framerange.parse(text), want)

    def test_empty_text_is_the_default_and_nonsense_is_an_error(self):
        self.assertEqual(framerange.parse("", [1, 2, 3]), [1, 2, 3])
        self.assertEqual(framerange.parse("   "), [])
        for text in ("a-b", "1-", "1--", "1-5x0", "1.5"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    framerange.parse(text)
        self.assertFalse(framerange.is_valid("x"))

    def test_describe_round_trips(self):
        for frames in ([1, 2, 3], [1, 4, 7], [2, 5, 6], [9], [5, 3, 1]):
            self.assertEqual(framerange.parse(framerange.describe(frames)), frames)


class KnobTests(unittest.TestCase):
    def test_new_profile_has_an_empty_range_and_rgba(self):
        params = SPECS["Profile"]["params"]
        self.assertEqual((params["frame_range"], params["channels"]), ("", "rgba"))

    def test_the_range_runs_the_node_range_or_the_project_range(self):
        time = {"first": 3, "last": 6, "current": 3}
        self.assertEqual(profilenode.frames_to_profile({"frame_range": ""}, time), [3, 4, 5, 6])
        self.assertEqual(profilenode.frames_to_profile({"frame_range": "4-5"}, time), [4, 5])
        self.assertEqual(profilenode.frames_to_profile({}, time), [3, 4, 5, 6])

    def test_each_row_says_which_channels_it_was_taken_for(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=8, height=6))
        g.add("probe", "Profile", dict(channels="alpha"), image="plate")
        evaluator = Evaluator()
        evaluator.evaluate_raster(dict(g.doc, view="probe"), target="probe", frame=2)
        self.assertEqual(evaluator.profile_log["probe"][-1]["channels"], "alpha")

    def test_the_picture_passes_through_whatever_the_knobs_say(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=16, height=12, size=4))
        g.add("probe", "Profile", dict(channels="red", frame_range="1-3"), image="plate")
        np.testing.assert_array_equal(Evaluator().evaluate(dict(g.doc, view="probe"), "probe", frame=1),
                                      Evaluator().evaluate(dict(g.doc, view="plate"), "plate", frame=1))

    def test_an_old_document_loads_with_the_new_knobs(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=8, height=6))
        g.add("probe", "Profile", image="plate")
        document = copy.deepcopy(g.doc)
        for name in profilenode.NEW:
            document["nodes"]["probe"]["params"].pop(name, None)
        loaded = upgrade_document(document)
        validate(loaded)
        evaluator = Evaluator()
        evaluator.evaluate_raster(dict(loaded, view="probe"), target="probe", frame=1)
        self.assertEqual(evaluator.profile_log["probe"][-1]["channels"], "rgba")


class PanelTests(unittest.TestCase):
    def window(self):
        from tests.test_desktop import APP, Window, release_window
        window = Window()
        self.addCleanup(release_window, self)
        window.show()
        APP.processEvents()
        d = window.dispatcher
        d.execute({"op": "create", "id": "src1", "type": "Constant", "pos": [0, 0], "params": {"width": 8, "height": 6}})
        d.execute({"op": "create", "id": "probe1", "type": "Profile", "pos": [0, 100]})
        d.execute({"op": "connect", "id": "probe1", "input": "image", "source": "src1"})
        return window, APP

    def test_profile_frame_range_button_records_one_row_per_frame(self):
        from PySide6.QtWidgets import QPushButton, QTableWidget
        window, APP = self.window()
        window.dispatcher.execute({"op": "set", "id": "probe1", "param": "frame_range", "value": "2-6x2"})
        window.dispatcher.execute({"op": "set", "id": "probe1", "param": "channels", "value": "rgb"})
        window.evaluator.profile_log.pop("probe1", None)
        panel = window.build_node_panel("probe1")
        panel.findChild(QPushButton, "profile-run").click()
        APP.processEvents()
        rows = window.evaluator.profile_log["probe1"]
        self.assertEqual([row["frame"] for row in rows], [2, 4, 6])
        self.assertEqual({row["channels"] for row in rows}, {"rgb"})
        panel = window.build_node_panel("probe1")
        table = panel.findChild(QTableWidget, "profile-table")
        self.assertEqual(table.rowCount(), 3)
        self.assertEqual(table.item(0, 3).text(), "rgb")
        window.saved_document = window.dispatcher.document
        window.close()

    def test_an_empty_range_runs_the_project_range(self):
        from PySide6.QtWidgets import QPushButton
        window, APP = self.window()
        window.dispatcher.document["time"].update(first=1, last=4)
        window.evaluator.profile_log.pop("probe1", None)
        panel = window.build_node_panel("probe1")
        panel.findChild(QPushButton, "profile-run").click()
        APP.processEvents()
        self.assertEqual([row["frame"] for row in window.evaluator.profile_log["probe1"]], [1, 2, 3, 4])
        window.saved_document = window.dispatcher.document
        window.close()

    def test_open_profile_panel_shows_the_table_in_its_own_window(self):
        from PySide6.QtWidgets import QDialog, QPushButton, QTableWidget
        window, APP = self.window()
        window.evaluator.profile_log.pop("probe1", None)
        window.dispatcher.execute({"op": "set", "id": "probe1", "param": "frame_range", "value": "1-3"})
        panel = window.build_node_panel("probe1")
        panel.findChild(QPushButton, "profile-open-panel").click()
        APP.processEvents()
        dialog = window.findChild(QDialog, "profile-panel")
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.isVisible())
        self.assertEqual(dialog.findChild(QTableWidget, "profile-table").rowCount(), 0)
        # Running the range from the window's own button fills the same table.
        dialog.findChild(QPushButton, "profile-run").click()
        APP.processEvents()
        again = window.findChild(QDialog, "profile-panel")
        self.assertEqual(again.findChild(QTableWidget, "profile-table").rowCount(), 3)
        again.close()
        window.saved_document = window.dispatcher.document
        window.close()

    def test_a_bad_range_runs_nothing(self):
        from PySide6.QtWidgets import QPushButton
        window, APP = self.window()
        window.dispatcher.execute({"op": "set", "id": "probe1", "param": "frame_range", "value": "a-b"})
        window.evaluator.profile_log.pop("probe1", None)
        window._show_command_error = lambda error: setattr(window, "_last_error", str(error))
        panel = window.build_node_panel("probe1")
        panel.findChild(QPushButton, "profile-run").click()
        APP.processEvents()
        self.assertNotIn("probe1", window.evaluator.profile_log)
        self.assertIn("frame range", window._last_error)
        window.saved_document = window.dispatcher.document
        window.close()


if __name__ == "__main__":
    unittest.main()
