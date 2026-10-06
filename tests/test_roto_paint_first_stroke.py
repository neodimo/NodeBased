"""Lane 2 step X2: RotoPaint opens with its paint controls first, and an unwired one explains itself."""
import os
import unittest
from tests.waiting import wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QGraphicsTextItem, QLabel, QToolButton, QWidget

from nodebased.app import STYLE, Window
from nodebased.core import SPECS, Dispatcher, empty_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)


def unwired_document(params=None):
    dispatcher = Dispatcher(empty_document())
    command = {"op": "create", "id": "fad513c22990", "type": "RotoPaint", "name": "Scratch cleanup"}
    if params:
        command["params"] = params
    dispatcher.execute({"op": "batch", "commands": [
        command, {"op": "view", "id": "fad513c22990"},
        {"op": "create", "id": "gr", "type": "Grade", "name": "Warm"},
        {"op": "connect", "id": "gr", "input": "image", "source": "fad513c22990"}]})
    return dispatcher.document


class EmptySourceMessageTests(unittest.TestCase):
    def check(self, message, name, slot="image"):
        self.assertNotIn("fad513c22990", message)
        self.assertNotIn("no generator reached", message)
        self.assertIn(name, message)
        self.assertIn(slot, message)
        self.assertIn("connect", message.lower())

    def test_tile_path_names_the_unwired_rotopaint_and_its_input(self):
        document = unwired_document()
        for target in ("fad513c22990", "gr"):
            for call in (lambda t=target: TileExecutor(Evaluator()).canvas_region(document, t, 1),
                         lambda t=target: TileExecutor(Evaluator()).canvas_size(document, t, 1)):
                with self.assertRaises(ValueError) as caught:
                    call()
                self.check(str(caught.exception), "Scratch cleanup")

    def test_reference_evaluator_names_the_unwired_rotopaint_and_its_input(self):
        document = unwired_document()
        for target in ("fad513c22990", "gr"):
            with self.assertRaises(ValueError) as caught:
                Evaluator().evaluate_raster(document, target=target, frame=1)
            self.check(str(caught.exception), "Scratch cleanup")
            self.assertIn("connect required input", str(caught.exception))

    def test_default_named_node_reads_naturally(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "p", "type": "RotoPaint"}, {"op": "view", "id": "p"}]})
        with self.assertRaises(ValueError) as caught:
            TileExecutor(Evaluator()).canvas_region(dispatcher.document, "p", 1)
        message = str(caught.exception)
        self.assertTrue(message.startswith("RotoPaint has nothing connected to its image input"), message)

    def test_unwired_node_never_renders_a_made_up_image(self):
        document = unwired_document()
        with self.assertRaises(ValueError):
            Evaluator().evaluate_raster(document, target="fad513c22990", frame=1)

    def test_switch_names_the_selected_empty_input(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "id": "s", "type": "Switch", "name": "Pick", "params": {"which": 1}},
            {"op": "create", "id": "c", "type": "Constant", "params": {"width": 8, "height": 8}},
            {"op": "connect", "id": "s", "input": SPECS["Switch"]["inputs"][0], "source": "c"}]})
        with self.assertRaises(ValueError) as caught:
            TileExecutor(Evaluator()).canvas_region(dispatcher.document, "s", 1)
        self.assertIn(SPECS["Switch"]["inputs"][1], str(caught.exception))
        self.assertIn("Pick", str(caught.exception))


class RotoPaintPanelOrderTests(unittest.TestCase):
    def open(self, params=None):
        self.window = Window(unwired_document(params))
        self.window.show()
        self.window.graph.items_by_id["fad513c22990"].setSelected(True)
        APP.processEvents()
        panel = self.window.findChild(QWidget, "node-properties-panel")
        self.assertIsNotNone(panel)
        return panel

    def tearDown(self):
        window = getattr(self, "window", None)
        if window is not None:
            window.saved_document = window.dispatcher.document
            window.close(); APP.processEvents()

    @staticmethod
    def top(widget):
        return widget.mapTo(widget.window(), widget.rect().topLeft()).y()

    def test_paint_tool_and_brush_come_before_the_detect_controls(self):
        panel = self.open()
        tool = panel.findChild(QComboBox, "rotopaint-tool")
        toggle = panel.findChild(QToolButton, "rotopaint-advanced-toggle")
        self.assertIsNotNone(tool); self.assertIsNotNone(toggle)
        self.assertLess(self.top(tool), self.top(toggle))
        labels = [w.text() for w in panel.findChildren(QToolButton) if w.objectName() == "rotopaint-advanced-toggle"]
        self.assertIn("Dust removal settings (advanced)", labels[0])

    def test_advanced_section_is_collapsed_until_opened_and_keeps_every_knob(self):
        panel = self.open()
        body = panel.findChild(QWidget, "rotopaint-advanced-body")
        toggle = panel.findChild(QToolButton, "rotopaint-advanced-toggle")
        self.assertFalse(body.isVisibleTo(panel))
        toggle.click(); APP.processEvents()
        self.assertTrue(body.isVisibleTo(panel))
        shown = {label.text() for label in body.findChildren(QLabel)}
        for label in ("Detect from frame", "Detect to frame", "Detect sensitivity", "Patch blend"):
            self.assertIn(label, shown)
        toggle.click(); APP.processEvents()
        self.assertFalse(body.isVisibleTo(panel))

    def test_saved_non_default_detect_values_open_the_section_and_survive(self):
        panel = self.open({"dustbust_sensitivity": 0.8, "dustbust_frame_end": 40})
        body = panel.findChild(QWidget, "rotopaint-advanced-body")
        self.assertTrue(body.isVisibleTo(panel))
        params = self.window.graph_nodes()["fad513c22990"]["params"]
        self.assertEqual(params["dustbust_sensitivity"], 0.8)
        self.assertEqual(params["dustbust_frame_end"], 40)

    def test_mix_is_still_offered_after_the_paint_controls(self):
        panel = self.open()
        tool = panel.findChild(QComboBox, "rotopaint-tool")
        mix = panel.findChild(QWidget, "rotopaint-mix-body")
        self.assertIsNotNone(mix)
        self.assertGreater(self.top(mix), self.top(tool))

    def test_every_rotopaint_param_is_still_a_knob(self):
        from nodebased.knobs import knob_layout
        laid_out = [p for group in knob_layout("RotoPaint") for p in group.params]
        self.assertEqual(sorted(laid_out), sorted(SPECS["RotoPaint"]["params"]))


class UnwiredViewerTests(unittest.TestCase):
    def test_viewer_tells_the_artist_which_input_needs_a_source(self):
        window = Window(unwired_document())
        try:
            window.show()
            self.assertTrue(wait_until(lambda: window.viewer_info.text() == "Evaluation error"))
            texts = [item.toPlainText() for item in window.viewer.scene().items()
                     if isinstance(item, QGraphicsTextItem)]
            self.assertTrue(texts, "an error message is drawn in the viewer")
            message = texts[0]
            self.assertIn("Scratch cleanup", message)
            self.assertIn("image input", message)
            self.assertNotIn("fad513c22990", message)
            self.assertIsNone(window.frame)
        finally:
            window.saved_document = window.dispatcher.document
            window.close(); APP.processEvents()


if __name__ == "__main__":
    unittest.main()
