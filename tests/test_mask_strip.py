"""One mask strip for every node (lane 10, step T2): the mask channel picker with its on/off box, inject, invert and fringe on one
row, (un)premult by with its invert on a second, mix last, in Nuke's words (docs/parity/nuke_nodes.json: mask, inject, invert,
fringe, (un)premult by, invert, mix; Calling Channels, Nuke 17.0 User Guide)."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import unittest

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)
from nodebased import knobs
from nodebased.core import SPECS

STRIP = set(knobs.MASK_ROW) | set(knobs.PREMULT_ROW)


class LayoutTests(unittest.TestCase):
    def test_no_panel_reads_the_old_strip_words(self):
        """The chrome's "Mask channel", "Inject mask", "Invert mask", "Mask fringe" are gone from the layout the audit reads,
        unless a lane set the node's own label on purpose (NUKE_LABELS)."""
        for kind in SPECS:
            own = {**knobs.NUKE_LABELS.get(kind, {}), **knobs.GUIDE_STRIP_WORDS.get(kind, {})}
            for group in knobs.knob_layout(kind):
                for param in group.params:
                    if param in STRIP and param not in own:
                        with self.subTest(kind=kind, param=param):
                            self.assertNotEqual(group.label, knobs._CHROME_LABELS[param])

    def test_the_strip_is_one_row_in_nukes_order(self):
        rows = 0
        for kind in SPECS:
            layout = knobs.panel_layout(kind)
            for group in layout:
                if group.params and group.params[0] == "maskChannelInput" and group.kind == "row":
                    rows += 1
                    with self.subTest(kind=kind):
                        order = [knobs.MASK_ROW.index(p) for p in group.params]
                        self.assertEqual(order, sorted(order))
                        self.assertEqual(group.members[0], "string")
                        self.assertTrue(all(c for c in group.captions[1:]))
        self.assertGreater(rows, 50)

    def test_the_panel_rows_hold_the_same_params_in_the_same_order(self):
        for kind in SPECS:
            ours = [p for g in knobs.knob_layout(kind) for p in g.params]
            drawn = [p for g in knobs.panel_layout(kind) for p in g.params]
            with self.subTest(kind=kind):
                self.assertEqual(sorted(ours), sorted(drawn))
                self.assertEqual(len(drawn), len(set(drawn)))


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nodebased.app import Window
        cls.window = Window()
        cls.window.dispatcher.execute({"op": "create", "id": "g", "type": "Grade", "pos": [0, 0]})

    @classmethod
    def tearDownClass(cls):
        cls.window.saved_document = cls.window.dispatcher.document
        cls.window.close()
        cls.window.deleteLater()

    def test_mask_row_widgets_and_names(self):
        from PySide6.QtWidgets import QCheckBox
        panel = self.window.build_node_panel("g")
        texts = {name: panel.findChild(QCheckBox, f"{name}-switch").text() for name in ("inject", "invert_mask", "fringe", "invert_unpremult")}
        self.assertEqual(texts, {"inject": "Inject", "invert_mask": "Invert", "fringe": "Fringe", "invert_unpremult": "Invert"})
        def row_of(widget):
            while widget is not None and type(widget).__name__ != "WrappingRow":
                widget = widget.parentWidget()
            return widget
        row = row_of(panel.findChild(QCheckBox, "inject-switch"))
        self.assertIsNotNone(row)
        for name in ("maskChannelInput-enabled", "invert_mask-switch", "fringe-switch"):
            self.assertIs(row_of(panel.findChild(QCheckBox, name)), row, name)

    def test_the_mask_box_turns_the_mask_on_and_off(self):
        from PySide6.QtWidgets import QCheckBox
        panel = self.window.build_node_panel("g")
        box = panel.findChild(QCheckBox, "maskChannelInput-enabled")
        self.assertFalse(box.isChecked())
        box.click()
        APP.processEvents()
        self.assertEqual(self.window.dispatcher.document["nodes"]["g"]["params"]["maskChannelInput"], "rgba.alpha")
        panel = self.window.build_node_panel("g")
        box = panel.findChild(QCheckBox, "maskChannelInput-enabled")
        self.assertTrue(box.isChecked())
        box.click()
        APP.processEvents()
        self.assertEqual(self.window.dispatcher.document["nodes"]["g"]["params"]["maskChannelInput"], "none")


if __name__ == "__main__":
    unittest.main()
