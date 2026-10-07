"""Lane 8 R1: Roto artist controls stay in the validated shape/render path."""
import copy
import os
import unittest

import numpy as np
from PySide6.QtCore import QSettings, Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QListWidget, QDoubleSpinBox, QWidget, QPushButton

from tests.waiting import wait_until, settle_layout
from tests.test_roto_ui import APP, roto_document, triangle
from nodebased.app import Window
from nodebased import roto, shapes

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def box(name, x0, y0, x1, y1, **settings):
    value = triangle(); value["name"] = name
    corners = ((x0,y0),(x1,y0),(x1,y1),(x0,y1))
    value["points"] = [{"x":float(x),"y":float(y),"in_x":0.0,"in_y":0.0,"out_x":0.0,"out_y":0.0} for x,y in corners]
    value.update(settings)
    return value


class RotoArtistPanelTests(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.shapes=[box("red",10,10,60,60,color=[1,0,0,1],opacity=.5,feather=2),
                     box("blue",30,30,80,80,color=[0,0,1,1],opacity=.25,feather=1)]
        self.window=Window(roto_document(self.shapes)); self.window.show()
        self.assertTrue(wait_until(lambda:self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True); APP.processEvents()
        settle_layout(self.window,self.window.properties.widget())

    def tearDown(self):
        self.window.saved_document=self.window.dispatcher.document
        self.window.close(); APP.processEvents()

    def panel(self): return self.window.properties.widget().findChild(QWidget,"roto-shapes-panel")

    def test_two_shapes_knobs_render_same_as_evaluator_and_hand_checked_feather_opacity_pixel(self):
        panel=self.panel(); self.assertIsNotNone(panel)
        listing=panel.findChild(QListWidget,"roto-shape-list")
        self.assertEqual(listing.count(),2)
        listing.setCurrentRow(0); APP.processEvents()
        feather=panel.findChild(QDoubleSpinBox,"roto-shape-feather")
        opacity=panel.findChild(QDoubleSpinBox,"roto-shape-opacity")
        self.assertIsNotNone(feather); self.assertIsNotNone(opacity)
        feather.setValue(0.0); feather.editingFinished.emit()
        opacity.setValue(0.5); opacity.editingFinished.emit()
        doc=self.window.dispatcher.document
        resolved=shapes.resolve_shapes(doc["node_data"]["r"],1)
        expected=roto.rasterise(resolved,100,100)
        actual=self.window.evaluator.evaluate_raster(doc,target="r").pixels
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        np.testing.assert_allclose(actual[20,20],[.5,0,0,.5],atol=1e-6)

    def test_visibility_reorder_and_panel_viewer_selection(self):
        listing=self.panel().findChild(QListWidget,"roto-shape-list")
        listing.setCurrentRow(1); APP.processEvents()
        self.assertEqual(self.window.viewer.roto_selected_shape_index,1)
        self.window.command({"op":"set_shapes","id":"r","shapes":copy.deepcopy(self.shapes)})
        before=self.window.evaluator.evaluate_raster(self.window.dispatcher.document,target="r").pixels.copy()
        panel=self.panel(); panel.findChild(QListWidget,"roto-shape-list").setCurrentRow(0)
        panel.findChild(QPushButton,"roto-shape-down").click()
        after=self.window.evaluator.evaluate_raster(self.window.dispatcher.document,target="r").pixels
        self.assertFalse(np.allclose(before[40,40],after[40,40]))
        panel=self.panel(); cb=panel.findChild(QCheckBox,"roto-shape-row-visible-0")
        cb.setChecked(False)
        hidden=self.window.evaluator.evaluate_raster(self.window.dispatcher.document,target="r").pixels
        self.assertFalse(np.allclose(after,hidden))

    def test_each_knob_edit_is_one_undo_and_redo_step(self):
        panel=self.panel(); feather=panel.findChild(QDoubleSpinBox,"roto-shape-feather")
        old=float(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["feather"])
        feather.setValue(7.0); feather.editingFinished.emit()
        self.assertTrue(wait_until(lambda: self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["feather"] == 7.0))
        self.window.command({"op":"undo"})
        self.assertEqual(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["feather"],old)
        self.window.command({"op":"redo"})
        self.assertEqual(self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["feather"],7.0)

    def test_viewer_pick_updates_shape_list_selection(self):
        self.window.viewer.roto_selected_shape_index=0
        self.window.rebuild_properties_dock(); APP.processEvents()
        viewer=self.window.viewer
        target=viewer.mapFromScene(QPointF(80,80))
        QTest.mouseClick(viewer.viewport(),Qt.MouseButton.LeftButton,pos=target)
        self.assertEqual(viewer.roto_selected_shape_index,1)
        listing=self.panel().findChild(QListWidget,"roto-shape-list")
        self.assertEqual(listing.currentRow(),1)

    def test_motion_blur_uses_adjacent_animation_samples(self):
        animated=box("moving",10,10,35,35,motion_blur=True)
        for point in animated["points"]:
            x=point["x"]
            point["x"]={"value":x,"curve":{"interpolation":"linear","keys":[{"frame":1,"value":x},{"frame":2,"value":x+20}]}}
        blurred=roto.rasterise(shapes.resolve_shapes({"shapes":[animated]},1),100,100)
        animated["motion_blur"]=False
        sharp=roto.rasterise(shapes.resolve_shapes({"shapes":[animated]},1),100,100)
        self.assertFalse(np.allclose(blurred,sharp))

    def test_old_shape_payload_defaults_render_identically(self):
        legacy=box("legacy",10,10,60,60)
        legacy.pop("color",None)
        doc=roto_document([legacy])
        actual=self.window.evaluator.evaluate_raster(doc,target="r").pixels
        expected=roto.rasterise(shapes.resolve_shapes({"shapes":[legacy]},1),100,100)
        np.testing.assert_allclose(actual,expected,atol=1e-6)
