"""Focused behavior checks for the Nuke-style Roto tool settings strip."""
import os
import tempfile
import unittest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication
from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, atomic_save, empty_document, load_document
from nodebased.imaging import Evaluator
from nodebased import paintknobs
from nodebased.roto_settings import TOOL_CONTROLS
from tests.test_roto_ui import roto_document
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)


class RotoSettingsR2Tests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(roto_document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close(); APP.processEvents()

    def test_strip_follows_tool_controls_and_reserves_autokey(self):
        viewer = self.window.viewer
        groups = {"Select All":"Select", "Select Splines":"Select", "Select Points":"Select",
                  "Select Feather Points":"Select", "Add Points":"Points", "Remove Points":"Points",
                  "Cusp Points":"Points", "Smooth Points":"Points", "Remove Feather":"Points",
                  "Open/Close Curve":"Points"}
        groups.update({name:"Shapes" for name in ("Bezier","Cusped Bezier","B-Spline","Ellipse","Rectangle","Cusped Rectangle","Open Spline")})
        groups.update({name:"Brushes" for name in ("Brush","Eraser")})
        groups.update({name:"Clone/Reveal" for name in ("Clone","Reveal")})
        groups.update({name:"Blur/Sharpen/Smear" for name in ("Blur","Sharpen","Smear")})
        groups.update({name:"Dodge/Burn" for name in ("Dodge","Burn")})
        for tool, controls in TOOL_CONTROLS.items():
            viewer.roto_toolbar.choose(groups[tool], tool)
            actual = list(viewer.roto_toolbar.settings_widgets)
            expected = [viewer.roto_toolbar._setting_key(c["label"]) for c in controls]
            self.assertEqual(actual, expected, tool)
            self.assertEqual(viewer.roto_toolbar.settings_strip.property("toolName"), tool)

    def test_shape_creation_copies_strip_values_and_settings_survive_payload_edits(self):
        viewer = self.window.viewer
        viewer.select_roto_tool("Shapes", "Bezier")
        for key, value in (("opacity", 0.35), ("feather", 4.0), ("color", [0.2, 0.4, 0.6, 1.0])):
            viewer.roto_toolbar.set_setting("r", "Bezier", key, value)
        self.assertTrue(viewer.begin_roto_draw("r"))
        viewer.roto_draw_points = [{"x":float(x), "y":float(y), "in_x":0.0, "in_y":0.0,
                                    "out_x":0.0, "out_y":0.0, "scene":viewer.mapToScene(x,y)}
                                   for x,y in ((20,20),(80,20),(80,80),(20,80))]
        self.assertTrue(viewer.finish_roto_draw(closed=True))
        payload = self.window.dispatcher.document["node_data"]["r"]
        shape = payload["shapes"][-1]
        self.assertEqual((shape["opacity"], shape["feather"], shape["color"]),
                         (0.35, 4.0, [0.2, 0.4, 0.6, 1.0]))
        self.assertEqual(payload["tool_settings"]["Bezier"]["opacity"], 0.35)

    def test_brush_strip_values_copy_to_new_stroke(self):
        self.window.command({"op":"create", "id":"p", "type":"RotoPaint", "pos":[0,0]})
        self.window.command({"op":"view", "id":"p"})
        self.window.graph.scene().clearSelection(); self.window.graph.items_by_id["p"].setSelected(True)
        APP.processEvents()
        viewer = self.window.viewer
        viewer.select_roto_tool("Brushes", "Brush")
        for key, value in (("brush_size", 37.0), ("brush_hardness", 0.72), ("opacity", 0.42)):
            viewer.roto_toolbar.set_setting("p", "Brush", key, value)
        viewer.paint_drag = {"key":"p", "points":[{"x":10.0,"y":10.0,"pressure":1.0}]}
        self.assertTrue(viewer._finish_paint())
        stroke = self.window.dispatcher.document["node_data"]["p"]["items"][-1]
        self.assertEqual(stroke["brush"]["size"], 37.0)
        self.assertEqual(stroke["brush"]["hardness"], 0.72)
        self.assertEqual(stroke["opacity"], 0.42)

    def test_legacy_document_has_no_settings_and_validates_unchanged(self):
        legacy = empty_document()
        dispatcher = Dispatcher(legacy)
        dispatcher.execute({"op":"create", "id":"r", "type":"Roto"})
        dispatcher.execute({"op":"set", "id":"r", "param":"width", "value":32})
        dispatcher.execute({"op":"set", "id":"r", "param":"height", "value":32})
        dispatcher.execute({"op":"view", "id":"r"})
        old = dispatcher.document
        self.assertNotIn("tool_settings", old["node_data"].get("r", {}))
        loaded = Dispatcher(old)
        self.assertEqual(loaded.document, old)

    def test_tool_values_round_trip_and_view_toggles_leave_rendered_pixels_unchanged(self):
        viewer = self.window.viewer
        shape = {"name":"test", "mode":"union", "opacity":1.0, "feather":0.0,
                 "points":[{"x":float(x), "y":float(y), "in_x":0.0, "in_y":0.0,
                            "out_x":0.0, "out_y":0.0} for x,y in ((10,10),(90,10),(90,90),(10,90))]}
        self.window.command({"op":"set_shapes", "id":"r", "shapes":[shape]}, render=False)
        before = Evaluator().evaluate_raster(self.window.dispatcher.document, target="r").pixels.copy()
        viewer.roto_toolbar.set_setting("r", "Select All", "hide_points", True)
        viewer.roto_toolbar.set_setting("r", "Select All", "hide_splines", True)
        after = Evaluator().evaluate_raster(self.window.dispatcher.document, target="r").pixels
        self.assertTrue((before == after).all())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "roto.json")
            atomic_save(path, self.window.dispatcher.document)
            reloaded = load_document(path)
        self.assertEqual(reloaded["node_data"]["r"]["tool_settings"]["Select All"]["hide_points"], True)
        self.assertEqual(reloaded["node_data"]["r"]["tool_settings"]["Select All"]["hide_splines"], True)

    def test_clone_transform_and_clone_reveal_time_offsets_are_stored_and_resolve(self):
        self.window.command({"op":"create", "id":"p", "type":"RotoPaint", "pos":[0,0]})
        self.window.command({"op":"view", "id":"p"})
        self.window.graph.scene().clearSelection(); self.window.graph.items_by_id["p"].setSelected(True)
        APP.processEvents()
        viewer = self.window.viewer
        viewer.select_roto_tool("Clone/Reveal", "Clone")
        values = {"source_offset":[4.0,-3.0], "source_rotate":25.0, "source_scale":1.5,
                  "source_time_offset":2.0, "source_time_offset_type":"relative"}
        for key, value in values.items(): viewer.roto_toolbar.set_setting("p", "Clone", key, value)
        viewer.paint_drag = {"key":"p", "points":[{"x":10.0,"y":10.0,"pressure":1.0}]}
        self.assertTrue(viewer._finish_paint())
        stroke = self.window.dispatcher.document["node_data"]["p"]["items"][-1]
        self.assertEqual(stroke["source_offset"], [4.0,-3.0])
        self.assertEqual(stroke["source_transform"]["rotate"], 25.0)
        self.assertEqual(stroke["source_transform"]["scale_x"], 1.5)
        self.assertEqual(paintknobs.source_time_frame(stroke, 10, 1, 100, clone=True), 11)
        viewer.select_roto_tool("Clone/Reveal", "Reveal")
        viewer.roto_toolbar.set_setting("p", "Reveal", "source_time_offset", -3.0)
        viewer.roto_toolbar.set_setting("p", "Reveal", "source_time_offset_type", "relative")
        viewer.paint_drag = {"key":"p", "points":[{"x":10.0,"y":10.0,"pressure":1.0}]}
        self.assertTrue(viewer._finish_paint())
        reveal = self.window.dispatcher.document["node_data"]["p"]["items"][-1]
        self.assertEqual(paintknobs.source_time_frame(reveal, 10, 1, 100), 7)
