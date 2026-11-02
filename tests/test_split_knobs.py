"""Per-channel ("split") knobs: the value form, split_values, validation, schema 29 and the panel's split toggle.

Part 1 tests run on a registration made here (Grade.multiply as an r g b a knob) so they hold whatever the real nodes declare;
the nodes that really are splittable are tested further down.
"""
import contextlib
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QMessageBox, QToolButton, QDoubleSpinBox

from nodebased import core, knobs
from nodebased.core import (Dispatcher, SCHEMA_VERSION, SPECS, SPLITTABLE, demo_document, load_document, atomic_save,
                            split_count, split_values, validate)


@contextlib.contextmanager
def split_registered(kind="Grade", name="multiply", channels="rgba"):
    """Declare `kind`.`name` splittable, and lay it out as a split knob, for the length of the block."""
    layout = knobs.KNOB_LAYOUT[kind]
    patched = []
    for group in layout:
        if group.params == (name,):
            patched.append(knobs.KnobGroup("split", (name,), label=group.label, soft_range=group.soft_range,
                                           section=group.section, help=group.help, channels=channels))
        else:
            patched.append(group)
    SPLITTABLE.setdefault(kind, {})[name] = channels
    knobs.KNOB_LAYOUT[kind] = patched
    try:
        yield
    finally:
        knobs.KNOB_LAYOUT[kind] = layout
        del SPLITTABLE[kind][name]
        if not SPLITTABLE[kind]:
            del SPLITTABLE[kind]


def grade_dispatcher(**params):
    d = Dispatcher()
    d.execute({"op": "create", "id": "g", "type": "Grade", "pos": [0, 0], "params": params})
    return d


class SplitValuesTests(unittest.TestCase):
    def test_a_number_is_repeated_and_a_list_is_taken_as_given(self):
        values = split_values({"gain": 1.5}, "gain", 4)
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(values, np.full(4, 1.5, np.float32))
        np.testing.assert_array_equal(split_values({"gain": [1.0, 2.0, 3.0, 4.0]}, "gain", 4),
                                      np.array([1, 2, 3, 4], np.float32))
        np.testing.assert_array_equal(split_values({"size": [3.0, 5.0]}, "size", 2), np.array([3, 5], np.float32))
        self.assertEqual(split_values({"size": 7}, "size", 2).shape, (2,))

    def test_a_list_of_the_wrong_length_is_an_error_never_padded(self):
        with self.assertRaises(ValueError):
            split_values({"gain": [1.0, 2.0]}, "gain", 4)
        with self.assertRaises(ValueError):
            split_values({"gain": [1.0, 2.0, 3.0]}, "gain", 2)

    def test_a_missing_param_reads_its_default(self):
        np.testing.assert_array_equal(split_values({}, "gain", 3, default=2.0), np.full(3, 2.0, np.float32))
        with self.assertRaises(KeyError):
            split_values({}, "gain", 3)


class SplitValidationTests(unittest.TestCase):
    def test_a_param_that_is_not_splittable_rejects_a_list(self):
        d = grade_dispatcher()
        with self.assertRaises(ValueError):
            d.execute({"op": "set", "id": "g", "param": "multiply", "value": [1.0, 1.0, 1.0, 1.0]})

    def test_a_split_value_is_one_entry_per_channel(self):
        with split_registered():
            self.assertEqual(split_count("Grade", "multiply"), 4)
            d = grade_dispatcher()
            d.execute({"op": "set", "id": "g", "param": "multiply", "value": [1, 2.5, 0.5, 1]})
            self.assertEqual(d.document["nodes"]["g"]["params"]["multiply"], [1.0, 2.5, 0.5, 1.0])
            self.assertTrue(all(type(v) is float for v in d.document["nodes"]["g"]["params"]["multiply"]))

    def test_the_wrong_length_is_refused_at_set_time(self):
        with split_registered():
            d = grade_dispatcher()
            for bad in ([1.0], [1.0, 1.0, 1.0], [1.0] * 5, []):
                with self.assertRaises(ValueError, msg=str(bad)):
                    d.execute({"op": "set", "id": "g", "param": "multiply", "value": bad})
            self.assertEqual(d.document["nodes"]["g"]["params"]["multiply"], 1.0)      # the document is untouched

    def test_limits_and_types_apply_to_each_entry(self):
        with split_registered():
            d = grade_dispatcher()
            lo, hi = core.parameter_limits("Grade", "multiply")
            for bad in ([1.0, 1.0, hi * 2, 1.0], [1.0, float("nan"), 1.0, 1.0], [1.0, "a", 1.0, 1.0], [1.0, True, 1.0, 1.0],
                        [1.0, None, 1.0, 1.0]):
                with self.assertRaises(ValueError, msg=str(bad)):
                    d.execute({"op": "set", "id": "g", "param": "multiply", "value": bad})
            d.execute({"op": "set", "id": "g", "param": "multiply", "value": [lo, hi, 0.0, 1.0]})

    def test_a_split_knob_cannot_be_animated_or_driven(self):
        with split_registered():
            d = grade_dispatcher()
            d.execute({"op": "set", "id": "g", "param": "multiply", "value": [1.0, 2.0, 1.0, 1.0]})
            with self.assertRaises(ValueError):
                d.execute({"op": "set_key", "id": "g", "param": "multiply", "frame": 1, "value": 1.0})
            with self.assertRaises(ValueError):
                d.execute({"op": "set_expression", "id": "g", "param": "multiply", "expression": "2"})
            # the other order: an animated knob refuses a list
            d2 = grade_dispatcher()
            d2.execute({"op": "set_key", "id": "g", "param": "multiply", "frame": 1, "value": 1.0})
            with self.assertRaises(ValueError):
                d2.execute({"op": "set", "id": "g", "param": "multiply", "value": [1.0, 2.0, 1.0, 1.0]})

    def test_an_expression_reading_a_split_knob_sees_its_first_entry(self):
        with split_registered():
            d = grade_dispatcher()
            d.execute({"op": "create", "id": "h", "type": "Grade", "pos": [0, 0]})
            d.execute({"op": "set", "id": "g", "param": "multiply", "value": [3.0, 1.0, 1.0, 1.0]})
            d.execute({"op": "set_expression", "id": "h", "param": "offset", "expression": "g.multiply * 2"})
            from nodebased.animation import resolve_document
            self.assertEqual(resolve_document(d.document, 1)["nodes"]["h"]["params"]["offset"], 6.0)

    def test_describe_tells_an_agent_which_knobs_split(self):
        with split_registered():
            info = Dispatcher().execute({"op": "describe"})
            self.assertEqual(info["split"]["params"], {"Grade": {"multiply": "rgba"}})
            self.assertEqual(info["split"]["channels"]["rgba"], ["r", "g", "b", "a"])
            self.assertEqual(info["split"]["channels"]["wh"], ["w", "h"])


class SplitDocumentTests(unittest.TestCase):
    def test_a_split_value_round_trips_through_save_load_and_undo(self):
        with split_registered():
            d = grade_dispatcher()
            before = copy.deepcopy(d.document)
            d.execute({"op": "set", "id": "g", "param": "multiply", "value": [1.0, 2.0, 0.5, 1.0]})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "split.nodebased.json"
                d.execute({"op": "save", "path": str(path)})
                self.assertEqual(json.loads(path.read_text())["version"], SCHEMA_VERSION)
                loaded = load_document(path)
            self.assertEqual(loaded["nodes"]["g"]["params"]["multiply"], [1.0, 2.0, 0.5, 1.0])
            d.execute({"op": "undo"})
            self.assertEqual(d.document, before)
            d.execute({"op": "redo"})
            self.assertEqual(d.document["nodes"]["g"]["params"]["multiply"], [1.0, 2.0, 0.5, 1.0])

    def test_the_schema_is_past_28_so_an_older_build_refuses_a_split_document(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 29)

    def test_a_schema_28_document_loads_and_keeps_its_scalars(self):
        d = grade_dispatcher(multiply=1.5, offset=0.25)
        old = copy.deepcopy(d.document)
        old["version"] = 28
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(old))
            loaded = load_document(path)
        self.assertEqual(loaded["version"], SCHEMA_VERSION)
        self.assertEqual(loaded["nodes"]["g"]["params"]["multiply"], 1.5)
        self.assertEqual(loaded["nodes"]["g"]["params"]["offset"], 0.25)
        validate(loaded)

    def test_a_scalar_document_is_saved_exactly_as_before(self):
        d = grade_dispatcher(multiply=1.5)
        saved = json.dumps(d.document["nodes"]["g"]["params"], sort_keys=True)
        with split_registered():
            d2 = grade_dispatcher(multiply=1.5)
            self.assertEqual(json.dumps(d2.document["nodes"]["g"]["params"], sort_keys=True), saved)
            self.assertEqual(d2.document["nodes"]["g"]["params"]["multiply"], 1.5)


class SplitLayoutTests(unittest.TestCase):
    def test_split_group_needs_one_param_and_a_known_split_set(self):
        with self.assertRaises(ValueError):
            knobs.KnobGroup("split", ("multiply",))
        with self.assertRaises(ValueError):
            knobs.KnobGroup("split", ("multiply",), channels="xyz")
        with self.assertRaises(ValueError):
            knobs.KnobGroup("split", ("a", "b"), channels="rgba")
        with self.assertRaises(ValueError):
            knobs.KnobGroup("float", ("multiply",), channels="rgba")
        self.assertIn("split", knobs.KNOB_KINDS)
        self.assertEqual(knobs.KnobGroup("split", ("size",), channels="wh").channels, "wh")

    def test_the_core_registry_and_the_layouts_agree(self):
        self.assertEqual(knobs.split_layout_problems(), [])
        with split_registered():
            self.assertEqual(knobs.split_layout_problems(), [])
        SPLITTABLE.setdefault("Grade", {})["gamma"] = "rgba"
        try:
            self.assertTrue(any("Grade.gamma" in problem for problem in knobs.split_layout_problems()))
        finally:
            del SPLITTABLE["Grade"]["gamma"]
            if not SPLITTABLE["Grade"]:
                del SPLITTABLE["Grade"]


from tests.test_desktop import APP, release_window                                  # noqa: E402  (builds the Qt application)


class SplitPanelTests(unittest.TestCase):
    def setUp(self):
        from nodebased.app import Window
        self.window = Window()
        self.window.show()
        APP.processEvents()
        self.window.command({"op": "create", "id": "g", "type": "Grade", "pos": [0, 0]})

    def tearDown(self):
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def panel(self):
        panel = self.window.build_node_panel("g")
        self.window.set_properties_widget(panel)
        APP.processEvents()
        return panel

    def param(self, name="multiply"):
        return self.window.graph_nodes()["g"]["params"][name]

    def click(self, panel, name="multiply"):
        button = panel.findChild(QToolButton, f"{name}-split")
        self.assertIsNotNone(button, f"no split button for {name}")
        button.click()
        APP.processEvents()
        APP.processEvents()
        return self.panel()

    def test_an_unsplit_knob_is_todays_slider_plus_a_split_button(self):
        with split_registered():
            panel = self.panel()
            button = panel.findChild(QToolButton, "multiply-split")
            self.assertEqual(button.text(), "4")
            self.assertFalse(button.isChecked())
            self.assertIsNotNone(panel.findChild(QDoubleSpinBox, "multiply-field"))
            self.assertIsNone(panel.findChild(QDoubleSpinBox, "multiply-r-field"))

    def test_the_toggle_writes_the_scalar_repeated_and_back_in_one_undo_step_each(self):
        with split_registered():
            self.window.command({"op": "set", "id": "g", "param": "multiply", "value": 1.75})
            revision = self.window.dispatcher.revision
            panel = self.click(self.panel())
            self.assertEqual(self.param(), [1.75] * 4)
            self.assertEqual(self.window.dispatcher.revision, revision + 1)
            for channel in "rgba":
                self.assertIsNotNone(panel.findChild(QDoubleSpinBox, f"multiply-{channel}-field"), channel)
            self.assertIsNone(panel.findChild(QDoubleSpinBox, "multiply-field"))
            self.assertEqual(panel.findChild(QToolButton, "multiply-split").text(), "1")
            self.click(panel)                                                       # equal entries collapse with no question
            self.assertEqual(self.param(), 1.75)
            self.assertEqual(self.window.dispatcher.revision, revision + 2)
            self.window.command({"op": "undo"})
            self.assertEqual(self.param(), [1.75] * 4)
            self.window.command({"op": "undo"})
            self.assertEqual(self.param(), 1.75)

    def test_editing_one_channel_field_changes_only_that_entry(self):
        with split_registered():
            panel = self.click(self.panel())
            field = panel.findChild(QDoubleSpinBox, "multiply-g-field")
            field.setValue(2.5)
            field.editingFinished.emit()
            APP.processEvents()
            APP.processEvents()
            self.assertEqual(self.param(), [1.0, 2.5, 1.0, 1.0])

    def test_collapsing_different_entries_asks_and_keeps_the_first(self):
        with split_registered():
            self.window.command({"op": "set", "id": "g", "param": "multiply", "value": [2.0, 3.0, 4.0, 1.0]})
            panel = self.panel()
            with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No) as ask:
                panel = self.click(panel)
                ask.assert_called_once()
            self.assertEqual(self.param(), [2.0, 3.0, 4.0, 1.0])                   # declined: nothing changed
            with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                self.click(panel)
            self.assertEqual(self.param(), 2.0)

    def test_an_animated_knob_cannot_be_split_and_says_why(self):
        with split_registered():
            self.window.command({"op": "set_key", "id": "g", "param": "multiply", "frame": 1, "value": 1.0})
            self.window.command({"op": "set_key", "id": "g", "param": "multiply", "frame": 10, "value": 2.0})
            panel = self.panel()
            button = panel.findChild(QToolButton, "multiply-split")
            self.assertFalse(button.isEnabled())
            self.assertIn("nimated", button.toolTip())

    def test_an_expression_driven_knob_cannot_be_split_and_says_why(self):
        with split_registered():
            self.window.command({"op": "set_expression", "id": "g", "param": "multiply", "expression": "frame"})
            button = self.panel().findChild(QToolButton, "multiply-split")
            self.assertFalse(button.isEnabled())
            self.assertIn("expression", button.toolTip())

    def test_a_split_grade_panel_builds_with_its_wheels(self):
        with split_registered():
            self.window.command({"op": "set", "id": "g", "param": "multiply", "value": [1.0, 2.0, 3.0, 1.0]})
            panel = self.panel()
            self.assertIsNotNone(panel.findChild(QDoubleSpinBox, "multiply-b-field"))


if __name__ == "__main__":
    unittest.main()
