"""Focused toolbar behavior checks for the Lane 9 R1 viewer work."""
import os
import unittest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt, QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from tests.waiting import wait_until
from nodebased.app import STYLE, Window
from tests.test_roto_ui import roto_document
from nodebased import roto, shapes
import numpy as np

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)

class RotoToolbarTests(unittest.TestCase):
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

    def put_shape(self, feather=0.0, handles=False):
        points = []
        for x, y in ((80, 80), (260, 80), (260, 240), (80, 240)):
            points.append({"x": float(x), "y": float(y), "in_x": -12.0 if handles else 0.0,
                           "in_y": 0.0, "out_x": 12.0 if handles else 0.0, "out_y": 0.0,
                           "feather_x": feather, "feather_y": 0.0})
        shape = {"name":"box", "mode":"union", "opacity":1.0, "feather":0.0,
                 "closed":True, "points":points}
        self.window.command({"op":"set_shapes", "id":"r", "shapes":[shape]})
        return shape

    def click_data(self, x, y):
        viewer = self.window.viewer
        context = viewer._roto_context()
        scene = viewer._roto_scene_point({"x":float(x),"y":float(y)}, context[3])
        return viewer.hit_press(scene, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)

    def test_roto_toolbar_order_context_and_shape_cycle(self):
        viewer = self.window.viewer
        viewer.roto_toolbar.refresh()
        self.assertTrue(viewer.roto_toolbar.isVisible())
        self.assertEqual(list(viewer.roto_toolbar.buttons), ["Select", "Points", "Shapes", "Brushes",
                         "Clone/Reveal", "Blur/Sharpen/Smear", "Dodge/Burn"])
        self.assertEqual([a.text() for a in viewer.roto_toolbar.buttons["Shapes"].menu().actions()],
                         ["Bezier", "Cusped Bezier", "B-Spline", "Ellipse", "Rectangle", "Cusped Rectangle", "Open Spline"])
        self.assertFalse(viewer.roto_toolbar.buttons["Brushes"].isVisible())
        viewer.setFocus()
        QTest.keyClick(viewer, Qt.Key.Key_Q)
        self.assertEqual(viewer.roto_selection_tool, "Select Points")
        QTest.keyClick(viewer, Qt.Key.Key_Q)
        self.assertEqual(viewer.roto_selection_tool, "Select Feather Points")
        QTest.keyClick(viewer, Qt.Key.Key_V)
        self.assertEqual(viewer.roto_shape_tool, "Bezier")
        QTest.keyClick(viewer, Qt.Key.Key_V)
        self.assertEqual(viewer.roto_shape_tool, "Cusped Bezier")
        QTest.keyClick(viewer, Qt.Key.Key_V)
        self.assertEqual(viewer.roto_toolbar.tool_by_group["Shapes"], "B-Spline")
        self.window.command({"op": "create", "id": "grade", "type": "Grade"})
        self.window.command({"op": "view", "id": "grade"})
        APP.processEvents(); viewer.roto_toolbar.refresh()
        self.assertFalse(viewer.roto_toolbar.isVisible())

    def test_ellipse_and_cusped_rectangle_create_four_point_bezier_shapes(self):
        viewer = self.window.viewer
        for tool in ("Ellipse", "Rectangle", "Cusped Rectangle"):
            viewer.select_roto_tool("Shapes", tool)
            self.assertTrue(viewer.begin_roto_draw("r"))
            viewer.roto_draw_press = viewer.mapToScene(100, 100)
            viewer.roto_draw_cursor = viewer.mapToScene(500, 300)
            viewer.hit_release(viewer.roto_draw_cursor, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            shape = self.window.dispatcher.document["node_data"]["r"]["shapes"][-1]
            self.assertEqual(len(shape["points"]), 4)
            self.assertTrue(shape["closed"])
            self.assertEqual((shape["name"], tool), (f"shape{1+('Ellipse','Rectangle','Cusped Rectangle').index(tool)}", tool))
            if tool == "Ellipse":
                xs = [point["x"] for point in shape["points"]]
                ys = [point["y"] for point in shape["points"]]
                width, height = max(xs)-min(xs), max(ys)-min(ys)
                copy_shape = {**shape, "points": [dict(point) for point in shape["points"]]}
                for point in copy_shape["points"]: point["x"] += 32-min(xs); point["y"] += 32-min(ys)
                mask = roto.rasterise(shapes.resolve_shapes({"shapes": [copy_shape]}, 1),
                                      int(width+64), int(height+64))[..., 3]
                expected = np.pi * width * height / 4
                self.assertLess(abs(float(mask.sum()) - expected) / expected, 0.01)
            else:
                for point in shape["points"]:
                    self.assertEqual((point["in_x"], point["in_y"], point["out_x"], point["out_y"]),
                                     (0.0, 0.0, 0.0, 0.0))

    def test_open_spline_escape_commits_an_open_shape_and_cusped_bezier_breaks_handles(self):
        viewer = self.window.viewer
        viewer.select_roto_tool("Shapes", "Open Spline")
        self.assertTrue(viewer.begin_roto_draw("r"))
        viewer.roto_draw_points = [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
                                    "out_x": 0.0, "out_y": 0.0, "scene": viewer.mapToScene(x, y)}
                                   for x, y in ((20, 20), (70, 50))]
        viewer.setFocus(); QTest.keyClick(viewer, Qt.Key.Key_Escape)
        saved = self.window.dispatcher.document["node_data"]["r"]["shapes"][-1]
        self.assertFalse(saved["closed"])
        viewer.select_roto_tool("Shapes", "Cusped Bezier")
        self.assertTrue(viewer.begin_roto_draw("r"))
        viewer.roto_draw_press = viewer.mapToScene(80, 80)
        viewer.roto_draw_cursor = viewer.mapToScene(120, 120)
        viewer.hit_release(viewer.roto_draw_cursor, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        point = viewer.roto_draw_points[0]
        self.assertEqual((point["in_x"], point["in_y"], point["out_x"], point["out_y"]),
                         (0.0, 0.0, 0.0, 0.0))
        viewer.roto_draw_points.extend([{"x":float(x),"y":float(y),"in_x":0.0,"in_y":0.0,
                                         "out_x":0.0,"out_y":0.0,"scene":viewer.mapToScene(x,y)}
                                        for x,y in ((170,90),(220,140))])
        self.assertTrue(viewer.finish_roto_draw(closed=True))
        cusped = self.window.dispatcher.document["node_data"]["r"]["shapes"][-1]
        self.assertTrue(all(not any(p[k] for k in ("in_x","in_y","out_x","out_y")) for p in cusped["points"]))
        viewer.select_roto_tool("Shapes", "Bezier"); self.assertTrue(viewer.begin_roto_draw("r"))
        viewer.roto_draw_press = viewer.mapToScene(240, 170)
        viewer.roto_draw_cursor = viewer.mapToScene(290, 210)
        viewer.hit_release(viewer.roto_draw_cursor, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.assertTrue(any(viewer.roto_draw_points[0][key] for key in ("in_x","in_y","out_x","out_y")))

    def test_point_tools_add_remove_cusp_smooth_feather_and_toggle_curve(self):
        viewer = self.window.viewer
        self.put_shape(handles=True, feather=7.0)
        viewer.select_roto_tool("Points", "Add Points"); self.assertTrue(self.click_data(170, 80))
        self.assertEqual(len(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"]), 5)
        viewer.select_roto_tool("Points", "Remove Points"); self.assertTrue(self.click_data(80, 80))
        self.assertEqual(len(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"]), 4)
        viewer.select_roto_tool("Points", "Cusp Points"); self.assertTrue(self.click_data(260, 80))
        point = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][1]
        self.assertEqual((point["in_x"],point["out_x"]),(0.0,0.0))
        viewer.select_roto_tool("Points", "Smooth Points"); self.assertTrue(self.click_data(260, 80))
        point = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][1]
        self.assertEqual(point["in_x"], -point["out_x"])
        viewer.select_roto_tool("Points", "Remove Feather"); self.assertTrue(self.click_data(260, 80))
        point = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][1]
        self.assertEqual((point["feather_x"],point["feather_y"]),(0.0,0.0))

    def test_select_splines_points_all_and_feather_modes_pick_only_their_targets(self):
        viewer=self.window.viewer
        shape=self.put_shape(feather=5.0)
        shape["feather_model"]="edge"
        self.window.command({"op":"set_shapes","id":"r","shapes":[shape]})
        viewer.roto_toolbar.choose("Select","Select Splines")
        self.assertTrue(self.click_data(170,80))
        self.assertEqual(viewer.roto_selected_shape_index,0); self.assertEqual(viewer.roto_selected_points,set())
        viewer.roto_toolbar.choose("Select","Select Points")
        self.assertTrue(self.click_data(80,80)); self.assertEqual(viewer.roto_selected_points,{(0,0)})
        viewer.roto_toolbar.choose("Select","Select All")
        self.assertTrue(self.click_data(170,80)); self.assertEqual(viewer.roto_selected_points,{(0,i) for i in range(4)})
        viewer.roto_toolbar.choose("Select","Select Feather Points")
        resolved=viewer._roto_resolved(viewer._roto_context())
        handle=shapes.feather_handle_points(resolved[0]["points"],resolved[0]["feather"])[0]
        scene=viewer._roto_scene_point(handle,1)
        self.assertTrue(viewer.hit_press(scene,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
        self.assertEqual(viewer.roto_selected_points,{(0,0)})
        viewer.select_roto_tool("Points", "Open/Close Curve"); self.assertTrue(self.click_data(260, 80))
        self.assertFalse(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["closed"])
        viewer.setFocus(); QTest.keyClick(viewer, Qt.Key.Key_Return)
        self.assertTrue(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["closed"])
        self.put_shape(feather=5.0)
        viewer.roto_selected_points = {(0, 0)}
        QTest.keyClick(viewer, Qt.Key.Key_E, Qt.KeyboardModifier.ShiftModifier)
        point = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][0]
        self.assertEqual((point["feather_x"],point["feather_y"]),(0.0,0.0))

if __name__ == "__main__":
    unittest.main()
