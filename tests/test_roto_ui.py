"""Offscreen artist-facing Roto overlay tests.

These deliberately drive the Viewer widget, while checking the resulting document through the
same Dispatcher-owned state that an agent sees.  The overlay itself remains outside the graphics
scene so format/data-window bounds cannot be changed by its handles.
"""
import os
import time
import unittest
from tests.waiting import wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import empty_document, Dispatcher


APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)




def roto_document(shapes=None, frame=1):
    document = empty_document()
    document["nodes"] = {}
    document["view"] = None
    document["time"].update(first=1, last=2, current=frame)
    dispatcher = Dispatcher(document)
    dispatcher.execute({"op": "create", "id": "r", "type": "Roto", "pos": [0, 0]})
    dispatcher.execute({"op": "set", "id": "r", "param": "width", "value": 100})
    dispatcher.execute({"op": "set", "id": "r", "param": "height", "value": 100})
    dispatcher.execute({"op": "view", "id": "r"})
    if shapes:
        dispatcher.execute({"op": "set_shapes", "id": "r", "shapes": shapes})
    return dispatcher.document


def triangle():
    return {"name": "triangle", "mode": "union", "opacity": 1.0, "feather": 0.0,
            "points": [{"x": 10.0, "y": 10.0, "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0},
                       {"x": 90.0, "y": 10.0, "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0},
                       {"x": 50.0, "y": 90.0, "in_x": 0.0, "in_y": 0.0,
                        "out_x": 0.0, "out_y": 0.0}]}


class RotoOverlayTests(unittest.TestCase):
    def setUp(self):
        # A saved workspace/state from a previous Window in this process restores verbatim
        # instead of running the default split, leaving the viewer at whatever size that
        # earlier window happened to save.
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(roto_document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def scene_pos(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def scene_pixel_tolerance(self):
        return 1.0 / max(abs(self.window.viewer.transform().m11()), 0.05) + 0.05

    def test_drawing_commits_one_validated_payload_and_keeps_scene_bounds(self):
        viewer = self.window.viewer
        original_bounds = viewer.scene().itemsBoundingRect()
        self.assertTrue(self.window.begin_roto_draw("r"))
        for x, y in ((10, 10), (90, 10), (50, 90)):
            QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton,
                              pos=self.scene_pos(x, y))
        viewer.setFocus()
        QTest.keyClick(viewer, Qt.Key.Key_Return)
        self.assertTrue(wait_until(lambda: "r" in self.window.dispatcher.document["node_data"]))
        shapes = self.window.dispatcher.document["node_data"]["r"]["shapes"]
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["name"], "shape1")
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)
        self.assertEqual(viewer.scene().itemsBoundingRect(), original_bounds)

    def test_dragging_a_point_auto_keys_it_when_another_point_in_the_shape_is_animated(self):
        shape = triangle()
        for point in shape["points"][:1]:
            for field, value in (("x", 10.0), ("y", 10.0)):
                point[field] = {"value": value,
                                "curve": {"interpolation": "linear",
                                          "keys": [{"frame": 1, "value": value},
                                                   {"frame": 2, "value": value + 5.0}]}}
        self.window.close()
        APP.processEvents()
        self.window = Window(roto_document([shape], frame=2))
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        viewer = self.window.viewer
        start = self.scene_pos(90, 10)
        target = self.scene_pos(80, 40)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewer.viewport(), target, 20)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=target)
        # The drop lands on a whole viewport pixel, so the exact value depends on the viewer's
        # zoom, which follows the window's size. Compare against where that pixel really is.
        dropped = viewer.mapToScene(target)
        self.assertAlmostEqual(dropped.x(), 80.0, delta=self.scene_pixel_tolerance())
        self.assertAlmostEqual(dropped.y(), 40.0, delta=self.scene_pixel_tolerance())
        self.assertTrue(wait_until(lambda: isinstance(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][1]["x"], dict)))
        point = self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][1]
        self.assertEqual(point["x"]["curve"]["keys"][-1]["frame"], 2)
        self.assertAlmostEqual(point["y"]["curve"]["keys"][-1]["value"], dropped.y(), delta=0.01)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_overlay_is_not_editable_when_viewing_a_downstream_node(self):
        shape = triangle()
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [shape]})
        self.window.command({"op": "create", "id": "c", "type": "Constant", "pos": [150, 0]})
        self.window.command({"op": "view", "id": "c"})
        self.window.graph.items_by_id["r"].setSelected(True)
        self.assertIsNone(self.window.viewer._roto_context())
        self.assertFalse(self.window.viewer.begin_roto_draw("r"))

    def test_proxy_display_keeps_full_format_coordinate_mapping(self):
        viewer = self.window.viewer
        self.window.proxy.setCurrentIndex(1)  # 1/2 preview is upscaled to the full scene rectangle.
        self.assertTrue(wait_until(lambda: viewer.format_rect is not None))
        context = viewer._roto_context()
        self.assertEqual(viewer._roto_data_point(QPointF(30, 40), context[1], 2), [30.0, 40.0])

    def test_drag_draws_a_smooth_point_and_closes_by_clicking_the_first(self):
        viewer=self.window.viewer
        self.assertTrue(self.window.begin_roto_draw("r"))
        start=self.scene_pos(10,10); handle=self.scene_pos(20,10)
        QTest.mousePress(viewer.viewport(),Qt.MouseButton.LeftButton,pos=start)
        QTest.mouseMove(viewer.viewport(),handle,20)
        QTest.mouseRelease(viewer.viewport(),Qt.MouseButton.LeftButton,pos=handle)
        for x,y in ((90,10),(50,90)):
            QTest.mouseClick(viewer.viewport(),Qt.MouseButton.LeftButton,pos=self.scene_pos(x,y))
        QTest.mouseClick(viewer.viewport(),Qt.MouseButton.LeftButton,pos=start)
        self.assertTrue(wait_until(lambda: "r" in self.window.dispatcher.document["node_data"]))
        points=self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"]
        self.assertEqual(len(points),3)
        pixel_tolerance = self.scene_pixel_tolerance()
        self.assertAlmostEqual(points[0]["out_x"],10.0,delta=pixel_tolerance)
        self.assertAlmostEqual(points[0]["in_x"],-10.0,delta=pixel_tolerance)
        self.assertEqual(len(self.window.dispatcher.undo_stack),1)

    def test_ctrl_alt_outline_insert_preserves_rendered_matte_and_is_one_undo(self):
        from nodebased import roto
        shape=triangle()
        self.window.close(); APP.processEvents()
        self.window=Window(roto_document([shape])); self.window.show()
        self.assertTrue(wait_until(lambda:self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True); APP.processEvents()
        viewer=self.window.viewer; context=viewer._roto_context()
        before=roto.rasterise([shape],100,100)
        self.assertTrue(viewer._insert_roto_point(viewer.mapToScene(self.scene_pos(50,10)),context))
        after=roto.rasterise(self.window.dispatcher.document["node_data"]["r"]["shapes"],100,100)
        self.assertLess(float(abs(before-after).max()),1e-4)
        self.assertEqual(len(self.window.dispatcher.undo_stack),1)

    def test_handle_drag_mirrors_unless_ctrl_breaks_it_and_undoes_once(self):
        shape=triangle(); shape["points"][0].update(out_x=5.0,out_y=0.0,in_x=-5.0,in_y=0.0)
        self.window.close(); APP.processEvents(); self.window=Window(roto_document([shape])); self.window.show()
        self.assertTrue(wait_until(lambda:self.window.frame is not None)); self.window.graph.items_by_id["r"].setSelected(True)
        viewer=self.window.viewer; context=viewer._roto_context()
        viewer.roto_drag={"point":(0,0),"kind":"tangent","handle":"out","scene":viewer.mapToScene(self.scene_pos(20,10)),"broken":False}
        viewer._commit_roto_drag(); point=self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][0]
        pixel_tolerance = self.scene_pixel_tolerance()
        self.assertAlmostEqual(point["out_x"],10.0,delta=pixel_tolerance); self.assertAlmostEqual(point["in_x"],-10.0,delta=pixel_tolerance)
        self.assertEqual(len(self.window.dispatcher.undo_stack),1)
        viewer.roto_drag={"point":(0,0),"kind":"tangent","handle":"out","scene":viewer.mapToScene(self.scene_pos(30,10)),"broken":True}
        viewer._commit_roto_drag(); point=self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"][0]
        self.assertAlmostEqual(point["out_x"],20.0,delta=pixel_tolerance); self.assertAlmostEqual(point["in_x"],-10.0,delta=pixel_tolerance)
        self.assertEqual(len(self.window.dispatcher.undo_stack),2)

    def test_deleting_selected_point_enforces_minimum_and_is_one_undo(self):
        shape=triangle(); shape["points"].append({"x":10.,"y":90.,"in_x":0.,"in_y":0.,"out_x":0.,"out_y":0.})
        self.window.close(); APP.processEvents(); self.window=Window(roto_document([shape])); self.window.show()
        self.assertTrue(wait_until(lambda:self.window.frame is not None)); self.window.graph.items_by_id["r"].setSelected(True); viewer=self.window.viewer
        viewer.roto_selected_points={(0,0)}
        self.assertTrue(viewer._delete_roto_points()); self.assertEqual(len(self.window.dispatcher.undo_stack),1)
        self.assertEqual(len(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["points"]),3)
        viewer.roto_selected_points={(0,0)}
        self.assertFalse(viewer._delete_roto_points()); self.assertEqual(len(self.window.dispatcher.undo_stack),1)

    def test_animated_outline_insert_preserves_each_key_frame_render(self):
        from nodebased import roto, shapes
        shape=triangle()
        for point in shape["points"]:
            for field,value in tuple(point.items()):
                point[field]={"value":value,"curve":{"interpolation":"linear","keys":[{"frame":1,"value":value},{"frame":2,"value":value}]}}
        self.window.close(); APP.processEvents(); self.window=Window(roto_document([shape],frame=1)); self.window.show()
        self.assertTrue(wait_until(lambda:self.window.frame is not None)); self.window.graph.items_by_id["r"].setSelected(True)
        viewer=self.window.viewer; context=viewer._roto_context(); before={f:roto.rasterise(shapes.resolve_shapes({"shapes":[shape]},f),100,100) for f in (1,2)}
        self.assertTrue(viewer._insert_roto_point(viewer.mapToScene(self.scene_pos(50,10)),context))
        updated=self.window.dispatcher.document["node_data"]["r"]["shapes"][0]
        self.assertEqual([k["frame"] for k in updated["points"][1]["x"]["curve"]["keys"]],[1,2])
        for f in (1,2):
            after=roto.rasterise(shapes.resolve_shapes({"shapes":[updated]},f),100,100)
            self.assertLess(float(abs(before[f]-after).max()),1e-4)
        self.assertEqual(len(self.window.dispatcher.undo_stack),1)


if __name__ == "__main__":
    unittest.main()
