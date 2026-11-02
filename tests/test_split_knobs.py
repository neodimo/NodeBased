"""Per-channel ("split") knobs: the value form, split_values, validation, schema 29 and the panel's split toggle.

Part 1 tests run on a registration made here (Grade.exposure as an r g b a knob) so they hold whatever the real nodes declare;
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
def split_registered(kind="Grade", name="exposure", channels="rgba"):
    """Declare `kind`.`name` splittable, and lay it out as a split knob, for the length of the block. Grade's exposure stands in
    for any knob: no kernel reads it as a list, so these tests are about the mechanism (value form, validation, panel), not a node."""
    layout = knobs.KNOB_LAYOUT[kind]
    patched = []
    for group in layout:
        if group.params == (name,):
            patched.append(knobs.KnobGroup("split", (name,), label=group.label, soft_range=group.soft_range,
                                           section=group.section, help=group.help, channels=channels))
        else:
            patched.append(group)
    before = copy.deepcopy(SPLITTABLE)
    SPLITTABLE.setdefault(kind, {})[name] = channels
    knobs.KNOB_LAYOUT[kind] = patched
    try:
        yield
    finally:
        knobs.KNOB_LAYOUT[kind] = layout
        SPLITTABLE.clear()
        SPLITTABLE.update(before)


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
            d.execute({"op": "set", "id": "g", "param": "exposure", "value": [1.0, 1.0, 1.0, 1.0]})

    def test_a_split_value_is_one_entry_per_channel(self):
        with split_registered():
            self.assertEqual(split_count("Grade", "exposure"), 4)
            d = grade_dispatcher()
            d.execute({"op": "set", "id": "g", "param": "exposure", "value": [1, 2.5, 0.5, 1]})
            self.assertEqual(d.document["nodes"]["g"]["params"]["exposure"], [1.0, 2.5, 0.5, 1.0])
            self.assertTrue(all(type(v) is float for v in d.document["nodes"]["g"]["params"]["exposure"]))

    def test_the_wrong_length_is_refused_at_set_time(self):
        with split_registered():
            d = grade_dispatcher()
            for bad in ([1.0], [1.0, 1.0, 1.0], [1.0] * 5, []):
                with self.assertRaises(ValueError, msg=str(bad)):
                    d.execute({"op": "set", "id": "g", "param": "exposure", "value": bad})
            self.assertEqual(d.document["nodes"]["g"]["params"]["exposure"], 0.0)      # the document is untouched

    def test_limits_and_types_apply_to_each_entry(self):
        with split_registered():
            d = grade_dispatcher()
            lo, hi = core.parameter_limits("Grade", "exposure")
            for bad in ([1.0, 1.0, hi * 2, 1.0], [1.0, float("nan"), 1.0, 1.0], [1.0, "a", 1.0, 1.0], [1.0, True, 1.0, 1.0],
                        [1.0, None, 1.0, 1.0]):
                with self.assertRaises(ValueError, msg=str(bad)):
                    d.execute({"op": "set", "id": "g", "param": "exposure", "value": bad})
            d.execute({"op": "set", "id": "g", "param": "exposure", "value": [lo, hi, 0.0, 1.0]})

    def test_a_split_knob_cannot_be_animated_or_driven(self):
        with split_registered():
            d = grade_dispatcher()
            d.execute({"op": "set", "id": "g", "param": "exposure", "value": [1.0, 2.0, 1.0, 1.0]})
            with self.assertRaises(ValueError):
                d.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 1.0})
            with self.assertRaises(ValueError):
                d.execute({"op": "set_expression", "id": "g", "param": "exposure", "expression": "2"})
            # the other order: an animated knob refuses a list
            d2 = grade_dispatcher()
            d2.execute({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 1.0})
            with self.assertRaises(ValueError):
                d2.execute({"op": "set", "id": "g", "param": "exposure", "value": [1.0, 2.0, 1.0, 1.0]})

    def test_an_expression_reading_a_split_knob_sees_its_first_entry(self):
        with split_registered():
            d = grade_dispatcher()
            d.execute({"op": "create", "id": "h", "type": "Grade", "pos": [0, 0]})
            d.execute({"op": "set", "id": "g", "param": "exposure", "value": [3.0, 1.0, 1.0, 1.0]})
            d.execute({"op": "set_expression", "id": "h", "param": "offset", "expression": "g.exposure * 2"})
            from nodebased.animation import resolve_document
            self.assertEqual(resolve_document(d.document, 1)["nodes"]["h"]["params"]["offset"], 6.0)

    def test_describe_tells_an_agent_which_knobs_split(self):
        with split_registered():
            info = Dispatcher().execute({"op": "describe"})
            self.assertEqual(info["split"]["params"]["Grade"]["exposure"], "rgba")
            self.assertEqual(info["split"]["params"]["ColorCorrect"]["shadows_gain"], "rgba")
            self.assertEqual(info["split"]["channels"]["rgba"], ["r", "g", "b", "a"])
            self.assertEqual(info["split"]["channels"]["wh"], ["w", "h"])


class SplitAgentTests(unittest.TestCase):
    """The agent's knob API takes a split knob as a list and tells the agent which knobs split (describe)."""

    def proposal(self, **command):
        from nodebased import agentloop
        d = grade_dispatcher()
        describe = d.execute({"op": "describe"})
        return agentloop.validate_proposal({"commands": [{"op": "set", "id": "g", **command}], "done": False}, describe, d.document), agentloop

    def test_a_list_for_a_split_knob_is_accepted_and_applied(self):
        commands, _agentloop = self.proposal(param="multiply", value=[1.0, 2.0, 0.5, 1.0])
        d = grade_dispatcher()
        d.execute(commands[0])
        self.assertEqual(d.document["nodes"]["g"]["params"]["multiply"], [1.0, 2.0, 0.5, 1.0])

    def test_the_wrong_length_or_an_unsplittable_knob_is_refused_before_it_reaches_the_document(self):
        from nodebased import agentloop
        for param, value in (("multiply", [1.0, 2.0]), ("multiply", [1.0, "a", 1.0, 1.0]), ("exposure", [1.0, 1.0, 1.0, 1.0]),
                             ("multiply", [1.0, 1.0, 1.0, 1e9])):
            with self.subTest(param=param, value=value), self.assertRaises(agentloop.ProtocolError):
                self.proposal(param=param, value=value)

    def test_create_with_a_split_value(self):
        from nodebased import agentloop
        d = Dispatcher()
        describe = d.execute({"op": "describe"})
        agentloop.validate_proposal({"commands": [{"op": "create", "type": "ColorCorrect", "id": "c", "pos": [0, 0],
                                                   "params": {"gain": [1.0, 1.5, 2.0, 1.0]}}], "done": False}, describe, d.document)


class SplitDocumentTests(unittest.TestCase):
    def test_a_split_value_round_trips_through_save_load_and_undo(self):
        with split_registered():
            d = grade_dispatcher()
            before = copy.deepcopy(d.document)
            d.execute({"op": "set", "id": "g", "param": "exposure", "value": [1.0, 2.0, 0.5, 1.0]})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "split.nodebased.json"
                d.execute({"op": "save", "path": str(path)})
                self.assertEqual(json.loads(path.read_text())["version"], SCHEMA_VERSION)
                loaded = load_document(path)
            self.assertEqual(loaded["nodes"]["g"]["params"]["exposure"], [1.0, 2.0, 0.5, 1.0])
            d.execute({"op": "undo"})
            self.assertEqual(d.document, before)
            d.execute({"op": "redo"})
            self.assertEqual(d.document["nodes"]["g"]["params"]["exposure"], [1.0, 2.0, 0.5, 1.0])

    def test_the_schema_is_past_28_so_an_older_build_refuses_a_split_document(self):
        self.assertGreaterEqual(SCHEMA_VERSION, 29)

    def test_a_schema_28_document_loads_and_keeps_its_scalars(self):
        d = grade_dispatcher(exposure=1.5, offset=0.25)
        old = copy.deepcopy(d.document)
        old["version"] = 28
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(old))
            loaded = load_document(path)
        self.assertEqual(loaded["version"], SCHEMA_VERSION)
        self.assertEqual(loaded["nodes"]["g"]["params"]["exposure"], 1.5)
        self.assertEqual(loaded["nodes"]["g"]["params"]["offset"], 0.25)
        validate(loaded)

    def test_a_scalar_document_is_saved_exactly_as_before(self):
        d = grade_dispatcher(exposure=1.5)
        saved = json.dumps(d.document["nodes"]["g"]["params"], sort_keys=True)
        with split_registered():
            d2 = grade_dispatcher(exposure=1.5)
            self.assertEqual(json.dumps(d2.document["nodes"]["g"]["params"], sort_keys=True), saved)
            self.assertEqual(d2.document["nodes"]["g"]["params"]["exposure"], 1.5)


class SplitLayoutTests(unittest.TestCase):
    def test_split_group_needs_one_param_and_a_known_split_set(self):
        with self.assertRaises(ValueError):
            knobs.KnobGroup("split", ("exposure",))
        with self.assertRaises(ValueError):
            knobs.KnobGroup("split", ("exposure",), channels="xyz")
        with self.assertRaises(ValueError):
            knobs.KnobGroup("split", ("a", "b"), channels="rgba")
        with self.assertRaises(ValueError):
            knobs.KnobGroup("float", ("exposure",), channels="rgba")
        self.assertIn("split", knobs.KNOB_KINDS)
        self.assertEqual(knobs.KnobGroup("split", ("size",), channels="wh").channels, "wh")

    def test_the_core_registry_and_the_layouts_agree(self):
        self.assertEqual(knobs.split_layout_problems(), [])
        with split_registered():
            self.assertEqual(knobs.split_layout_problems(), [])
        SPLITTABLE["Grade"]["exposure"] = "rgba"
        try:
            self.assertTrue(any("Grade.exposure" in problem for problem in knobs.split_layout_problems()))
        finally:
            del SPLITTABLE["Grade"]["exposure"]


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

    def param(self, name="exposure"):
        return self.window.graph_nodes()["g"]["params"][name]

    def click(self, panel, name="exposure"):
        button = panel.findChild(QToolButton, f"{name}-split")
        self.assertIsNotNone(button, f"no split button for {name}")
        button.click()
        APP.processEvents()
        APP.processEvents()
        return self.panel()

    def test_an_unsplit_knob_is_todays_slider_plus_a_split_button(self):
        with split_registered():
            panel = self.panel()
            button = panel.findChild(QToolButton, "exposure-split")
            self.assertEqual(button.text(), "4")
            self.assertFalse(button.isChecked())
            self.assertIsNotNone(panel.findChild(QDoubleSpinBox, "exposure-field"))
            self.assertIsNone(panel.findChild(QDoubleSpinBox, "exposure-r-field"))

    def test_the_toggle_writes_the_scalar_repeated_and_back_in_one_undo_step_each(self):
        with split_registered():
            self.window.command({"op": "set", "id": "g", "param": "exposure", "value": 1.75})
            revision = self.window.dispatcher.revision
            panel = self.click(self.panel())
            self.assertEqual(self.param(), [1.75] * 4)
            self.assertEqual(self.window.dispatcher.revision, revision + 1)
            for channel in "rgba":
                self.assertIsNotNone(panel.findChild(QDoubleSpinBox, f"exposure-{channel}-field"), channel)
            self.assertIsNone(panel.findChild(QDoubleSpinBox, "exposure-field"))
            self.assertEqual(panel.findChild(QToolButton, "exposure-split").text(), "1")
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
            field = panel.findChild(QDoubleSpinBox, "exposure-g-field")
            field.setValue(2.5)
            field.editingFinished.emit()
            APP.processEvents()
            APP.processEvents()
            self.assertEqual(self.param(), [0.0, 2.5, 0.0, 0.0])

    def test_collapsing_different_entries_asks_and_keeps_the_first(self):
        with split_registered():
            self.window.command({"op": "set", "id": "g", "param": "exposure", "value": [2.0, 3.0, 4.0, 1.0]})
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
            self.window.command({"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 1.0})
            self.window.command({"op": "set_key", "id": "g", "param": "exposure", "frame": 10, "value": 2.0})
            panel = self.panel()
            button = panel.findChild(QToolButton, "exposure-split")
            self.assertFalse(button.isEnabled())
            self.assertIn("nimated", button.toolTip())

    def test_an_expression_driven_knob_cannot_be_split_and_says_why(self):
        with split_registered():
            self.window.command({"op": "set_expression", "id": "g", "param": "exposure", "expression": "frame"})
            button = self.panel().findChild(QToolButton, "exposure-split")
            self.assertFalse(button.isEnabled())
            self.assertIn("expression", button.toolTip())

    def test_a_split_grade_panel_builds_with_its_wheels(self):
        with split_registered():
            self.window.command({"op": "set", "id": "g", "param": "exposure", "value": [1.0, 2.0, 3.0, 1.0]})
            panel = self.panel()
            self.assertIsNotNone(panel.findChild(QDoubleSpinBox, "exposure-b-field"))


# ---------------------------------------------------------------------------------------------- the real nodes (part 2)
import sys                                                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import nuke_fidelity_cases_color as colour_cases                                     # noqa: E402
from nodebased import gradeknobs                                                      # noqa: E402
from nodebased.imaging import Evaluator                                               # noqa: E402
from tests.test_2d_behaviour_color_a import plate_graph, tiled                        # noqa: E402
from tests.test_2d_fidelity_b_chrome import image, matte, render                      # noqa: E402
from tests.test_2d_parity_group_3b import evaluator_pixels                            # noqa: E402

GRADE_SPLIT = {"blackpoint": [0.05, 0.0, 0.1, 0.0], "whitepoint": [2.0, 1.5, 1.0, 1.0], "black": [0.1, 0.0, -0.05, 0.02],
               "white": [0.9, 1.0, 1.2, 0.8], "multiply": [1.5, 0.8, 2.0, 1.3], "offset": [0.1, -0.1, 0.0, 0.05],
               "gamma": [1.8, 1.0, 0.7, 1.4]}


def nuke_grade(x, blackpoint=0.0, whitepoint=1.0, black=0.0, white=1.0, multiply=1.0, offset=0.0, gamma=1.0, black_clamp=True,
               white_clamp=False, reverse=False):
    """The Grade formula the fidelity cases state, on one channel: A = multiply * (white - black) / (whitepoint - blackpoint),
    B = offset + black - A * blackpoint, out = A * in + B, then ** (1 / gamma) where out > 0, then the clamps."""
    x = np.asarray(x, np.float64)
    scale = multiply * (white - black) / (whitepoint - blackpoint)
    shift = offset + black - scale * blackpoint
    if reverse:
        r = np.where(x > 0, np.power(np.maximum(x, 0), gamma), x)
        r = (r - shift) / scale
    else:
        r = scale * x + shift
        r = np.where(r > 0, np.power(np.maximum(r, 0), 1.0 / gamma), r)
    if black_clamp:
        r = np.maximum(r, 0.0)
    if white_clamp:
        r = np.minimum(r, 1.0)
    return r


def entry(params, c):
    """Every split knob of `params` at channel c, as the scalar a node with no split would hold."""
    return {name: value[c] for name, value in params.items()}


class SplitGradeTests(unittest.TestCase):
    def test_every_grade_knob_the_nuke_aco_knob_splits_is_registered_and_laid_out(self):
        self.assertEqual(sorted(SPLITTABLE["Grade"]), sorted(gradeknobs.SPLIT_NAMES))
        self.assertEqual(sorted(SPLITTABLE["Grade"]), sorted(["blackpoint", "whitepoint", "black", "white", "multiply", "offset", "gamma"]))
        self.assertEqual(knobs.split_layout_problems(), [])

    def test_each_channel_is_graded_with_its_own_entries_and_alpha_is_kept(self):
        for clamps in ({}, {"black_clamp": 0}, {"white_clamp": 1}):
            src = image()
            out = render("Grade", {**GRADE_SPLIT, **clamps}, source=src).pixels
            for c in range(3):
                with self.subTest(channel="rgb"[c], clamps=clamps):
                    expected = nuke_grade(src[..., c], black_clamp=bool(clamps.get("black_clamp", 1)),
                                          white_clamp=bool(clamps.get("white_clamp", 0)), **entry(GRADE_SPLIT, c))
                    np.testing.assert_allclose(out[..., c], expected, rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(out[..., 3], src[..., 3])

    def test_a_split_value_equals_a_grade_per_channel(self):
        """The independence property: channel c of the split node is channel c of an unsplit node holding entry c."""
        src = image(5)
        split = render("Grade", {**GRADE_SPLIT, "reverse": 0}, source=src).pixels
        for c in range(3):
            with self.subTest(channel="rgb"[c]):
                single = render("Grade", entry(GRADE_SPLIT, c), source=src).pixels
                np.testing.assert_allclose(split[..., c], single[..., c], rtol=2e-6, atol=2e-6)

    def test_reverse_undoes_a_split_grade_channel_by_channel(self):
        src = np.abs(image(6)) + 0.05
        params = {**{k: v for k, v in GRADE_SPLIT.items()}, "black_clamp": 0}
        graded = render("Grade", params, source=src).pixels
        back = render("Grade", {**params, "reverse": 1}, source=graded).pixels
        np.testing.assert_allclose(back[..., :3], src[..., :3], rtol=3e-4, atol=3e-4)
        for c in range(3):
            expected = nuke_grade(graded[..., c], black_clamp=False, reverse=True, **entry(GRADE_SPLIT, c))
            np.testing.assert_allclose(back[..., c], expected, rtol=3e-4, atol=3e-4)

    def test_equal_entries_equal_the_unsplit_grade(self):
        src = image(7)
        for params in ({"multiply": 1.7, "offset": 0.2}, {"gamma": 1.6, "blackpoint": 0.05, "whitepoint": 1.5},
                       {"multiply": 1.2, "gamma": 1.3, "reverse": 1, "black_clamp": 0}):
            split = {k: ([v] * 4 if k in SPLITTABLE["Grade"] else v) for k, v in params.items()}
            with self.subTest(params=params):
                np.testing.assert_allclose(render("Grade", split, source=src).pixels, render("Grade", params, source=src).pixels,
                                           rtol=3e-6, atol=3e-6)

    def test_alpha_takes_its_entry_only_when_the_channels_include_alpha(self):
        src = image(8)
        entries = {"multiply": [1.5, 1.5, 1.5, 0.5], "offset": [0.0, 0.0, 0.0, 0.1]}
        alpha_graded = nuke_grade(src[..., 3], multiply=0.5, offset=0.1)
        rgb_graded = nuke_grade(src[..., 0], multiply=1.5)
        base = render("Grade", entries, source=src).pixels                              # channels rgb: alpha is untouched
        np.testing.assert_array_equal(base[..., 3], src[..., 3])
        both = render("Grade", {**entries, "channels": "rgba"}, source=src).pixels
        np.testing.assert_allclose(both[..., 3], alpha_graded, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(both[..., 0], rgb_graded, rtol=2e-5, atol=2e-5)
        only = render("Grade", {**entries, "channels": "alpha"}, source=src).pixels
        np.testing.assert_allclose(only[..., 3], alpha_graded, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(only[..., :3], src[..., :3])
        red = render("Grade", {**entries, "channels": "red"}, source=src).pixels
        np.testing.assert_allclose(red[..., 0], rgb_graded, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(red[..., 1:], src[..., 1:])

    def test_scalar_grades_run_the_unchanged_kernel_bit_for_bit(self):
        src = image(9)
        legacy = {"exposure": 0.5, "multiply": 1.5, "offset": 0.1, "black_clamp": 0, "offset_by_alpha": 1}
        expected = src.copy()
        expected[..., :3] *= 2.0 ** 0.5
        expected[..., :3] *= 1.5
        expected[..., :3] += src[..., 3:4] * 0.1
        full = {**SPECS["Grade"]["params"], **legacy}
        self.assertFalse(gradeknobs.has_split(full))
        np.testing.assert_array_equal(Evaluator._grade(src, full), expected)
        newer = {**SPECS["Grade"]["params"], "multiply": 1.5, "offset": 0.1, "gamma": 1.4, "blackpoint": 0.05}
        out = np.empty_like(src)
        out[..., :3] = gradeknobs.grade_rgb(src[..., :3], src[..., 3:4], newer)
        out[..., 3] = src[..., 3]
        np.testing.assert_array_equal(Evaluator._grade(src, newer), out)

    def test_tiles_equal_the_whole_frame(self):
        for extra in ({}, {"channels": "rgba"}):
            params = {**GRADE_SPLIT, **extra}
            for masked in (False, True):
                g = plate_graph("Grade", params, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(extra=extra, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)

    def test_a_split_grade_has_no_gpu_kernel_and_says_so(self):
        from nodebased import gpu2d
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_grade({**SPECS["Grade"]["params"], **GRADE_SPLIT})
        gpu2d.map_grade({**SPECS["Grade"]["params"], "multiply": 1.5})                 # unsplit still maps

    def test_the_balance_wheels_of_a_split_grade_are_disabled_with_a_reason(self):
        from nodebased.app import Window
        window = Window()
        try:
            window.show()
            window.command({"op": "create", "id": "g", "type": "Grade", "pos": [0, 0]})
            window.command({"op": "set", "id": "g", "param": "multiply", "value": [1.0, 2.0, 3.0, 1.0]})
            panel = window.build_node_panel("g")
            from nodebased.app import PropertiesColorWheel
            wheel = panel.findChild(PropertiesColorWheel, "multiply-color-wheel")
            self.assertFalse(wheel.isEnabled())
            self.assertIn("split", wheel.toolTip())
        finally:
            window.deleteLater()


CC_NAMES = ("saturation", "contrast", "gamma", "gain", "offset")
CC_SPLIT = {
    "saturation": [0.5, 1.0, 1.5, 1.0], "contrast": [1.3, 1.0, 0.8, 1.2], "gamma": [1.4, 1.0, 0.8, 1.1],
    "gain": [1.5, 1.0, 0.7, 1.2], "offset": [0.05, 0.0, -0.03, 0.04],
    "shadows_gain": [1.8, 1.0, 1.0, 1.0], "shadows_offset": [0.0, 0.02, 0.0, 0.0],
    "midtones_gamma": [1.0, 1.3, 1.0, 1.0], "midtones_contrast": [1.0, 1.0, 1.2, 1.0],
    "highlights_gain": [0.7, 0.8, 0.9, 1.0], "highlights_saturation": [1.0, 0.6, 1.0, 1.0],
}


def cc_reference(src, params, c):
    """Channel c of the ColorCorrect the fidelity cases state (tools/nuke_fidelity_cases_color.color_correct_reference), with every
    knob at its entry for c."""
    master = {name: params[name][c] for name in CC_NAMES if name in params}
    zones = {zone: {name: params[f"{zone}_{name}"][c] for name in CC_NAMES if f"{zone}_{name}" in params}
             for zone in ("shadows", "midtones", "highlights")}
    return colour_cases.color_correct_reference(src[..., :3].astype(np.float64), master, zones)[..., c]


class SplitColorCorrectTests(unittest.TestCase):
    def test_all_twenty_knobs_are_registered_and_laid_out(self):
        self.assertEqual(len(SPLITTABLE["ColorCorrect"]), 20)
        self.assertEqual(knobs.split_layout_problems(), [])

    def test_each_channel_is_corrected_with_its_own_entries(self):
        src = image(11)
        out = render("ColorCorrect", CC_SPLIT, source=src).pixels
        for c in range(3):
            with self.subTest(channel="rgb"[c]):
                np.testing.assert_allclose(out[..., c], cc_reference(src, CC_SPLIT, c), rtol=5e-4, atol=5e-4)
        np.testing.assert_array_equal(out[..., 3], src[..., 3])

    def test_a_split_value_equals_a_correction_per_channel(self):
        src = image(12)
        for extra in ({}, {"divide_by_alpha": 1}):
            split = render("ColorCorrect", {**CC_SPLIT, **extra}, source=src).pixels
            for c in range(3):
                with self.subTest(channel="rgb"[c], extra=extra):
                    single = render("ColorCorrect", {**{k: v[c] for k, v in CC_SPLIT.items()}, **extra}, source=src).pixels
                    np.testing.assert_allclose(split[..., c], single[..., c], rtol=3e-5, atol=3e-5)

    def test_equal_entries_equal_the_unsplit_node(self):
        src = image(13)
        params = {"saturation": 0.6, "contrast": 1.3, "gamma": 1.2, "gain": 1.4, "offset": 0.05, "shadows_gain": 1.8,
                  "highlights_offset": -0.02, "midtones_contrast": 1.1}
        split = {k: [v] * 4 for k, v in params.items()}
        np.testing.assert_allclose(render("ColorCorrect", split, source=src).pixels, render("ColorCorrect", params, source=src).pixels,
                                   rtol=3e-5, atol=3e-5)

    def test_alpha_takes_its_entry_but_not_saturation_and_only_with_alpha_in_the_channels(self):
        src = image(14)
        params = {"gain": [1.0, 1.0, 1.0, 2.0], "offset": [0.0, 0.0, 0.0, 0.1], "saturation": [1.0, 1.0, 1.0, 0.0]}
        rgb_only = render("ColorCorrect", params, source=src).pixels
        np.testing.assert_array_equal(rgb_only[..., 3], src[..., 3])
        both = render("ColorCorrect", {**params, "channels": "rgba"}, source=src).pixels
        np.testing.assert_allclose(both[..., 3], src[..., 3] * 2.0 + 0.1, rtol=1e-5, atol=1e-5)    # the saturation entry is not applied to alpha
        np.testing.assert_allclose(both[..., :3], src[..., :3], rtol=1e-6, atol=1e-6)
        only = render("ColorCorrect", {**params, "channels": "alpha"}, source=src).pixels
        np.testing.assert_allclose(only[..., 3], both[..., 3], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(only[..., :3], src[..., :3])

    def test_tiles_equal_the_whole_frame(self):
        for extra in ({}, {"channels": "rgba", "divide_by_alpha": 1}):
            params = {**CC_SPLIT, **extra}
            for masked in (False, True):
                g = plate_graph("ColorCorrect", params, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(extra=extra, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)

    def test_a_split_correction_has_no_gpu_kernel_and_says_so(self):
        from nodebased import gpu2d
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_color_correct({**SPECS["ColorCorrect"]["params"], **CC_SPLIT})
        gpu2d.map_color_correct({**SPECS["ColorCorrect"]["params"], "gain": 1.5})


class SplitChannelNodesTests(unittest.TestCase):
    """Multiply, Add and Gamma: Nuke's `value` is an r g b a knob; their `channels` choose which entries land."""

    VALUES = {"Multiply": ("multiply", [2.0, 3.0, 0.5, 4.0], lambda x, v: x * v),
              "Add": ("offset", [0.1, -0.2, 0.3, 0.4], lambda x, v: x + v),
              "Gamma": ("gamma", [2.0, 0.5, 1.5, 3.0], lambda x, v: np.sign(x) * np.abs(x) ** (1.0 / v))}

    def test_each_channel_lands_its_own_entry(self):
        src = image(15)
        for kind, (name, value, formula) in self.VALUES.items():
            self.assertEqual(SPLITTABLE[kind], {name: "rgba"})
            for channels, picked in (("rgb", (0, 1, 2)), ("rgba", (0, 1, 2, 3)), ("alpha", (3,))):
                out = render(kind, {name: value, "channels": channels}, source=src).pixels
                for c in range(4):
                    with self.subTest(kind=kind, channels=channels, channel=c):
                        expected = formula(src[..., c], value[c]) if c in picked else src[..., c]
                        np.testing.assert_allclose(out[..., c], expected, rtol=2e-6, atol=2e-6)

    def test_equal_entries_equal_the_unsplit_node(self):
        src = image(16)
        for kind, (name, value, _formula) in self.VALUES.items():
            np.testing.assert_allclose(render(kind, {name: [value[0]] * 4, "channels": "rgba"}, source=src).pixels,
                                       render(kind, {name: value[0], "channels": "rgba"}, source=src).pixels, rtol=1e-6, atol=1e-6)

    def test_tiles_equal_the_whole_frame(self):
        for kind, (name, value, _formula) in self.VALUES.items():
            for masked in (False, True):
                g = plate_graph(kind, {name: value, "channels": "rgba"}, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


class SplitPanelRealNodesTests(unittest.TestCase):
    def test_every_splittable_knob_has_a_split_button_in_its_nodes_panel(self):
        from nodebased.app import Window
        window = Window()
        try:
            window.show()
            for kind, names in SPLITTABLE.items():
                window.command({"op": "create", "id": f"n_{kind}", "type": kind, "pos": [0, 0]})
                panel = window.build_node_panel(f"n_{kind}")
                for name in names:
                    with self.subTest(kind=kind, name=name):
                        self.assertIsNotNone(panel.findChild(QToolButton, f"{name}-split"))
        finally:
            window.deleteLater()

    def test_a_split_color_correct_panel_builds_with_every_range(self):
        from nodebased.app import Window
        window = Window()
        try:
            window.show()
            window.command({"op": "create", "id": "c", "type": "ColorCorrect", "pos": [0, 0]})
            window.command({"op": "set", "id": "c", "param": "shadows_gain", "value": [1.0, 2.0, 3.0, 1.0]})
            panel = window.build_node_panel("c")
            self.assertIsNotNone(panel.findChild(QDoubleSpinBox, "shadows_gain-g-field"))
            self.assertIsNone(panel.findChild(QDoubleSpinBox, "shadows_gain-field"))
        finally:
            window.deleteLater()


# ------------------------------------------------------------------------------------------------ layouts (part 3)
import audit_nuke_fidelity as audit                                                    # noqa: E402

NUKE_JSON = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))["nodes"]
MAPPING = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]

# The main tab's labels in order, as the panel lays them out (docs/parity/nuke_nodes.json through docs/parity/nodebased_to_nuke.json).
# Knobs NodeBased added (Exposure, Offset follows alpha, Divide by alpha, ...) sit after Nuke's own and before the mask strip. HistEQ,
# MinColor and ScreenKeyer keep NodeBased's layout: Nuke's knobs for them are a different control set (HistEQ's histogram entries,
# MinColor's search and delta) or the node is not in the Reference Guide's tables (ScreenKeyer).
LAYOUTS = {
    'Grade': [
        'Channels', 'Blackpoint', 'Whitepoint', 'Lift', 'Gain', 'Multiply', 'Offset', 'Gamma', 'Reverse', 'Black clamp',
        'White clamp', 'Exposure', 'Offset follows alpha', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert',
        'Mix'
    ],
    'ColorCorrect': [
        'Channels', 'Saturation', 'Contrast', 'Gamma', 'Gain', 'Offset', 'Saturation', 'Contrast', 'Gamma', 'Gain', 'Offset',
        'Saturation', 'Contrast', 'Gamma', 'Gain', 'Offset', 'Saturation', 'Contrast', 'Gamma', 'Gain', 'Offset',
        'Shadows curve', 'Highlights curve', 'Divide by alpha', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert',
        'Mix'
    ],
    'Saturation': ['Channels', 'Saturation', 'Luminance math', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'],
    'HueCorrect': [
        'Channels', 'Sat', 'Lum', 'Red', 'Green', 'Blue', 'R Sup', 'G Sup', 'B Sup', 'Saturation threshold', 'Hue shift',
        'Older curve maths', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'
    ],
    'HSVTool': [
        'Color replacement', 'Source color', 'Destination color', 'Hue range', 'Hue rotation', 'Hue rolloff', 'Saturation range',
        'Saturation adjustment', 'Saturation rolloff', 'Force saturation', 'Brightness range', 'Brightness adjustment',
        'Brightness rolloff', 'Force brightness', 'Output to alpha', 'Adjust by scaling', 'Mix'
    ],
    'Toe': ['Channels', 'Lift', 'Toe width', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'],
    'SoftClip': ['Conversion', 'Softclip min', 'Softclip max', 'Older grey', 'Mix'],
    'Posterize': ['Channels', 'Colors', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'],
    'Log2Lin': [
        'Channels', 'Operation', 'Black', 'Ignore', 'White', 'Gamma', 'Older log maths', 'Mask', 'Inject', 'Invert', 'Fringe',
        '(Un)premult by', 'Invert', 'Mix'
    ],
    'PLogLin': [
        'Channels', 'Operation', 'Linear reference value', 'Log reference value', 'Negative gamma', 'Density per code value',
        'Older log maths', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'
    ],
    'Colorspace': [
        'Channels', 'Transfer In', 'Whitepoint In', 'Primaries In', 'Transfer Out', 'Whitepoint Out', 'Primaries Out',
        'Bradford matrix', 'Divide by alpha', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'
    ],
    'ColorLookup': [
        'Channels', 'Use precomputed table', 'Range', 'Master', 'Red', 'Green', 'Blue', 'Alpha', 'Source', 'Target',
        'Older range scaling', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'
    ],
    'CrossTalk': [
        'R from R', 'R from G', 'R from B', 'G from R', 'G from G', 'G from B', 'B from R', 'B from G', 'B from B', 'Mask',
        'Inject', 'Invert', 'Fringe', 'Fringe', 'Unpremult by', '(Un)premult by', 'Invert', 'Mix'
    ],
    'Expression': [
        'R expression', 'G expression', 'B expression', 'A expression', 'Mask', 'Inject', 'Invert', 'Fringe', '(Un)premult by',
        'Invert', 'Mix'
    ],
    'Keyer': ['Input', 'Output', 'Combine', 'Invert', 'Operation', 'Range', 'Mix'],
    'HueKeyer': [
        'Channels', 'Hue center', 'Hue width', 'Hue softness', 'Saturation min', 'Saturation max', 'Output', 'Output channel',
        'Invert', 'Mix'
    ],
    'ChromaKeyer': [
        'View', 'Key colour', 'Screen gain', 'Screen balance', 'Chroma gain', 'White point', 'Black point', 'Alpha bias',
        'Inside mask', 'Use custom despill bias', 'Despill bias', 'Replace mode', 'Add in matte fix', 'Replace colour',
        'Replace amount', 'Premultiply', 'Tolerance', 'Softness', 'Luminance gain', 'Shadow level', 'Highlight level', 'Despill',
        'Invert', 'Mix'
    ],
    'Difference': ['Offset', 'Gain', 'Output', 'Output channel', 'Mix'],
    'IBKColor': ['Screen type', 'Size', 'Darks', 'Lights', 'Erode', 'Patch black', 'Mix'],
    'IBKGizmo': [
        'Screen type', 'Red weight', 'Blue/green weight', 'Luminance match', 'Luminance level', 'Screen subtraction',
        'Use BG luminance', 'Mix'
    ],
    'Copy': [
        'Copy Red', 'Copy Green', 'Copy Blue', 'Copy Alpha', 'Layer copy', 'Metadata from', 'Range from', 'Mask', 'Inject',
        'Invert', 'Fringe', '(Un)premult by', 'Invert', 'Mix'
    ],
    'ChannelMerge': ['A channel', 'Operation', 'B channel', 'Output', 'Mask', 'Inject', 'Invert', 'Fringe', 'Mix'],
    'Shuffle': [
        'Input 1', 'Layer 1', 'Input 2', 'Layer 2', 'Red From', 'Green From', 'Blue From', 'Alpha From', 'Second output layer',
        'Red from (second)', 'Green from (second)', 'Blue from (second)', 'Alpha from (second)'
    ],
    'HistEQ': ['Equalise', 'Mix'],
    'MinColor': ['Frame start', 'Frame end', 'Find', 'Box origin', 'Box size', 'Result R', 'Result G', 'Result B', 'Result A'],
    'ScreenKeyer': [
        'Screen colour', 'Screen gain', 'Screen balance', 'Despill bias', 'Alpha bias', 'Use bias colours', 'Alpha bias colour',
        'Despill bias colour', 'Clip black', 'Clip white', 'Clip rollback', 'Screen shrink/grow', 'Screen softness',
        'Screen pre-blur', 'Edge grade gain', 'Edge grade gamma', 'Edge grade saturation', 'Edge width', 'View',
        'Unpremultiply result', 'Use alpha bias for despill', 'Screen despot black', 'Screen despot white', 'Screen replace',
        'Screen replace colour', 'Midtones at', 'Shadow gain', 'Midtone gain', 'Highlight gain', 'Source alpha',
        'Inside replace', 'Inside replace colour', 'Inside component', 'Outside component', 'Crop x method', 'Crop y method',
        'Crop edge colour', 'Crop left', 'Crop right', 'Crop bottom', 'Crop top', 'Mix'
    ],
}


def nuke_rank(kind):
    """{param: position in Nuke's knob table} for the params the mapping pairs with a Nuke knob of `kind` (the table's order, all tabs)."""
    nuke, pairs = NUKE_JSON.get(kind), MAPPING.get(kind)
    if not nuke or not pairs:
        return {}
    by_nuke = {}
    for entry in pairs["knobs"]:
        for name in entry.get("nuke") or []:
            by_nuke.setdefault(name, []).extend(entry["nodebased"])
    seen, rank = {}, {}
    for knob in nuke["knobs"]:
        seen[knob["name"]] = seen.get(knob["name"], 0) + 1
        if knob["role"] == "ui_only":
            continue
        name = knob["name"] if seen[knob["name"]] == 1 else f"{knob['name']}#{seen[knob['name']]}"
        for param in by_nuke.get(name, []):
            rank.setdefault(param, len(rank))
    return rank


def displayed(kind):
    return [group.label for group in knobs.knob_layout(kind) if group.kind != "legacy"]


class LayoutTests(unittest.TestCase):
    def test_every_node_of_the_step_has_a_layout_test(self):
        self.assertEqual(sorted(LAYOUTS), sorted([*knobs.NUKE_LAYOUT, "HistEQ", "MinColor", "ScreenKeyer"]))
        self.assertEqual(len(LAYOUTS), 26)      # Grade, step A's seven, step B's eight and step C's ten

    def test_the_mask_strip_and_mix_close_every_node_that_has_them(self):
        tail = ["Mask", "Inject", "Invert", "Fringe", "(Un)premult by", "Invert", "Mix"]
        for kind in ("Grade", "ColorCorrect", "Saturation", "HueCorrect", "Toe", "Posterize", "Log2Lin", "PLogLin", "Colorspace",
                     "ColorLookup", "Expression", "Copy"):
            with self.subTest(kind=kind):
                labels = displayed(kind)
                self.assertEqual(labels[-len(tail):], tail if "(Un)premult by" in labels else labels[-len(tail):])
                self.assertEqual(labels[-1], "Mix")
                self.assertEqual(labels[-7:-1], tail[:-1])

    def test_channels_come_first_where_nuke_puts_them_first(self):
        for kind, table in NUKE_JSON.items():
            if kind not in LAYOUTS or not table["knobs"] or table["knobs"][0]["role"] != "chrome:channels":
                continue
            if not any(group.params == ("channels",) for group in knobs.knob_layout(kind)):
                continue
            with self.subTest(kind=kind):
                self.assertEqual(displayed(kind)[0].lower(), "channels")

    def test_nodebased_only_knobs_sit_between_nuke_knobs_and_the_mask_strip(self):
        self.assertEqual(displayed("Grade")[10:14], ["White clamp", "Exposure", "Offset follows alpha", "Mask"])
        self.assertEqual(displayed("ColorCorrect")[21:25], ["Shadows curve", "Highlights curve", "Divide by alpha", "Mask"])

    def test_keyer_range_and_the_hsv_ranges_are_one_row_each(self):
        keyer = [g for g in knobs.knob_layout("Keyer") if g.label == "Range"][0]
        self.assertEqual((keyer.kind, keyer.params, keyer.captions), ("xy", ("range_a", "range_b", "range_c", "range_d"), ("A", "B", "C", "D")))
        for label, params in (("Hue range", ("hue_range_min", "hue_range_max")), ("Saturation range", ("saturation_range_min", "saturation_range_max")),
                              ("Brightness range", ("brightness_range_min", "brightness_range_max"))):
            group = [g for g in knobs.knob_layout("HSVTool") if g.label == label][0]
            self.assertEqual((group.kind, group.params, group.captions), ("xy", params, ("min", "max")))

    def test_the_panel_lists_the_labels_in_that_order(self):
        from nodebased.app import Window
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QLabel
        window = Window()
        try:
            window.show()
            for kind in ("Grade", "ColorCorrect", "Keyer", "HSVTool", "Copy"):
                window.command({"op": "create", "id": f"n_{kind}", "type": kind, "pos": [0, 0]})
                panel = window.build_node_panel(f"n_{kind}")
                window.set_properties_widget(panel)
                APP.processEvents()
                shown = [text for _y, text in sorted((label.mapTo(panel, QPoint(0, 0)).y(), label.text())
                                                     for label in panel.findChildren(QLabel) if label.isVisibleTo(panel) and label.text())]
                remaining = iter(shown)
                with self.subTest(kind=kind):
                    for expected in LAYOUTS[kind]:
                        self.assertTrue(any(text == expected for text in remaining), f"{kind}: {expected!r} out of order in {shown}")
        finally:
            window.deleteLater()


def _layout_test(kind):
    def test(self):
        self.assertEqual(displayed(kind), LAYOUTS[kind], kind)
        rank = nuke_rank(kind)
        last = -1
        for group in knobs.knob_layout(kind):
            if group.kind == "legacy":
                continue
            ranks = [rank[param] for param in group.params if param in rank]
            if not ranks:
                continue
            with self.subTest(group=group.label):
                self.assertGreater(min(ranks), last, f"{kind}: {group.label} is before a knob Nuke lists earlier")
            last = max(last, min(ranks))
    return test


for _kind in LAYOUTS:
    setattr(LayoutTests, f"test_layout_{_kind}", _layout_test(_kind))


if __name__ == "__main__":
    unittest.main()
