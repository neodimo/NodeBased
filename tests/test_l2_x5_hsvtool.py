"""Lane 2 step X5, part 3: HSVTool's three Reset buttons (the Nuke 17.0 Reference Guide, Color nodes, HSVTool: `huereset` in the Hue section after Rotation,
`satreset` in Saturation after Adjustment, `brtreset` in Brightness after Adjustment, each before Range Rolloff). Each puts its section's range,
adjustment and rolloff back to the defaults in one undo step."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nodebased import hsvtoolknobs, knobs  # noqa: E402
from nodebased.core import SPECS, upgrade_document, validate  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402

SECTIONS = {"huereset": ("hue_rotation", "hue_rolloff"), "satreset": ("sat_adjust", "saturation_rolloff"),
            "brtreset": ("brt_adjust", "brightness_rolloff")}
MOVED = {"hue_range_min": 40.0, "hue_range_max": 200.0, "hue_rotation": 33.0, "hue_rolloff": 12.0,
         "saturation_range_min": 0.2, "saturation_range_max": 0.7, "sat_adjust": 0.5, "saturation_rolloff": 0.1,
         "brightness_range_min": 0.1, "brightness_range_max": 0.9, "brt_adjust": -0.4, "brightness_rolloff": 0.2}


class LayoutTests(unittest.TestCase):
    def test_the_guide_puts_each_reset_between_its_adjustment_and_its_rolloff(self):
        nodes = json.loads((ROOT / "docs" / "parity" / "nuke_nodes.json").read_text())["nodes"]["HSVTool"]["knobs"]
        names = [row["name"] for row in nodes]
        for reset, (before, after) in {"huereset": ("huerotation", "huerolloffrange"), "satreset": ("saturation", "satrolloffrange"),
                                       "brtreset": ("brightness", "brtrolloffrange")}.items():
            self.assertEqual(names.index(before) + 1, names.index(reset), reset)
            self.assertEqual(names.index(reset) + 1, names.index(after), reset)

    def test_the_panel_has_them_in_those_rows(self):
        order = [group.params[0] for group in knobs.KNOB_LAYOUT["HSVTool"]]
        for reset, (before, after) in SECTIONS.items():
            self.assertEqual(order.index(before) + 1, order.index(reset), reset)
            self.assertEqual(order.index(reset) + 1, order.index(after), reset)
        for group in knobs.KNOB_LAYOUT["HSVTool"]:
            if group.params[0] in SECTIONS:
                self.assertEqual((group.kind, group.label), ("button", "Reset"))

    def test_the_counters_are_on_a_new_node_at_zero(self):
        for name in SECTIONS:
            self.assertEqual(SPECS["HSVTool"]["params"][name], 0)


class ResetTests(unittest.TestCase):
    def window(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from nodebased.app import Window
        window = Window()
        window.dispatcher.execute({"op": "create", "id": "hsv", "type": "HSVTool", "pos": [0, 0]})
        for name, value in MOVED.items():
            window.dispatcher.execute({"op": "set", "id": "hsv", "param": name, "value": value})
        return app, window

    def test_each_button_restores_only_its_own_section_in_one_undo_step(self):
        from PySide6.QtWidgets import QPushButton
        app, window = self.window()
        try:
            defaults = SPECS["HSVTool"]["params"]
            for counter, names in hsvtoolknobs.RESETS.items():
                before = copy.deepcopy(window.dispatcher.document["nodes"]["hsv"]["params"])
                depth = len(window.dispatcher.undo_stack)
                panel = window.build_node_panel("hsv")
                button = panel.findChild(QPushButton, f"{counter}-button")
                self.assertIsNotNone(button, counter)
                self.assertEqual(button.text(), "Reset")
                button.click()
                app.processEvents()
                params = window.dispatcher.document["nodes"]["hsv"]["params"]
                for name in names:
                    self.assertEqual(params[name], defaults[name], name)
                for other, others in hsvtoolknobs.RESETS.items():
                    if other == counter:
                        continue
                    for name in others:
                        self.assertEqual(params[name], before[name], f"{counter} moved {name}")
                self.assertEqual(params[counter], before[counter] + 1)
                self.assertEqual(len(window.dispatcher.undo_stack), depth + 1, counter)
                window.dispatcher.execute({"op": "undo"})
                params = window.dispatcher.document["nodes"]["hsv"]["params"]
                for name in names:
                    self.assertEqual(params[name], MOVED[name], f"undo {name}")
                self.assertEqual(params[counter], before[counter])
                window.dispatcher.execute({"op": "redo"})
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()

    def test_a_reset_leaves_the_other_knobs_alone(self):
        app, window = self.window()
        try:
            window.dispatcher.execute({"op": "set", "id": "hsv", "param": "set_saturation", "value": 1})
            window.dispatcher.execute({"op": "set", "id": "hsv", "param": "HMask", "value": "rgba.alpha"})
            window.hsvtool_reset("hsv", "satreset")
            params = window.dispatcher.document["nodes"]["hsv"]["params"]
            self.assertEqual((params["set_saturation"], params["HMask"]), (1, "rgba.alpha"))
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()


class PixelAndDocumentTests(unittest.TestCase):
    def kernel(self, **params):
        image = np.random.default_rng(3).random((6, 6, 4)).astype(np.float32)
        return Evaluator._kernel("HSVTool", dict(SPECS["HSVTool"]["params"], **params), [image])

    def test_the_press_counter_moves_no_pixel(self):
        np.testing.assert_array_equal(self.kernel(), self.kernel(huereset=3, satreset=1, brtreset=7))

    def test_a_reset_gives_the_pixels_of_a_node_never_moved(self):
        plain = self.kernel()
        moved = self.kernel(**MOVED)
        self.assertGreater(float(np.abs(plain - moved).max()), 1e-3)
        reset = dict(MOVED)
        for names in hsvtoolknobs.RESETS.values():
            for name in names:
                reset[name] = SPECS["HSVTool"]["params"][name]
        np.testing.assert_array_equal(self.kernel(**reset), plain)

    def test_a_saved_document_loads_with_the_counters(self):
        from tests.test_2d_parity_group_c4 import Graph
        g = Graph()
        g.add("plate", "Constant", dict(width=4, height=4, red=0.5, green=0.5, blue=0.5, alpha=1.0))
        g.add("node", "HSVTool", {}, image="plate")
        document = copy.deepcopy(g.doc)
        for name in SECTIONS:
            document["nodes"]["node"]["params"].pop(name)
        loaded = upgrade_document(document)
        validate(loaded)
        for name in SECTIONS:
            self.assertEqual(loaded["nodes"]["node"]["params"][name], 0)

    def test_the_audit_pairs_the_three_buttons_as_panel_rows(self):
        pairs = json.loads((ROOT / "docs" / "parity" / "nodebased_to_nuke.json").read_text())["pairs"]["HSVTool"]["knobs"]
        rows = {row["nuke"][0]: row for row in pairs if row.get("nuke") and row["nuke"][0] in SECTIONS}
        self.assertEqual(set(rows), set(SECTIONS))
        for name, row in rows.items():
            self.assertEqual(row["kind"], "panel")


if __name__ == "__main__":
    unittest.main()
