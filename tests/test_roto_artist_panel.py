"""Lane 8 R1: Roto artist controls stay in the validated shape/render path."""
import copy
import os
import unittest

import numpy as np
from PySide6.QtCore import QSettings, Qt, QPointF
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QListWidget, QDoubleSpinBox, QWidget, QPushButton, QLabel

import tests.isolation
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

    def test_feather_key_states_use_the_transform_key_button_and_field_style(self):
        w = self.window
        w.set_time(first=1, last=20, current=1)
        w.command({"op": "create", "id": "t", "type": "Transform", "pos": [100, 0]}, render=False)
        animated = box("animated", 10, 10, 60, 60, feather={"value": 2.0, "curve": {
            "interpolation": "linear", "keys": [{"frame": 1, "value": 2.0},
                                                    {"frame": 11, "value": 12.0}]}})
        w.command({"op": "set_shapes", "id": "r", "shapes": [animated]}, render=False)
        for frame, value in ((1, 2.0), (11, 12.0)):
            w.command({"op": "set_key", "id": "t", "param": "translate_x", "frame": frame,
                       "value": value, "interpolation": "linear"}, render=False)
        w.graph.items_by_id["r"].setSelected(True)
        for frame, expected in ((1, "keyed"), (6, "between")):
            w.set_time(current=frame)
            self.assertTrue(wait_until(lambda: self.panel() is not None
                                       and self.panel().findChild(QPushButton, "roto-shape-key-feather")
                                       .property("keyState") == expected))
            roto_panel = self.panel()
            roto_key = roto_panel.findChild(QPushButton, "roto-shape-key-feather")
            roto_field = roto_panel.findChild(QDoubleSpinBox, "roto-shape-feather")
            transform_panel = w.build_node_panel("t")
            transform_key = next(button for button in transform_panel.findChildren(QPushButton, "key-button")
                                 if button.accessibleName() == "translate_x key")
            transform_field = transform_panel.findChild(QDoubleSpinBox, "translate_x-field")
            self.assertEqual(roto_key.property("keyState"), expected)
            self.assertEqual(roto_key.property("keyState"), transform_key.property("keyState"))
            for property_name in ("animated", "keyedHere"):
                self.assertEqual(roto_field.property(property_name), transform_field.property(property_name))
            self.assertEqual(roto_field.styleSheet(), transform_field.styleSheet())
        w.command({"op": "set_shapes", "id": "r", "shapes": [box("plain", 10, 10, 60, 60)]},
                  render=False)
        plain_panel = self.panel()
        plain_key = plain_panel.findChild(QPushButton, "roto-shape-key-feather")
        plain_field = plain_panel.findChild(QDoubleSpinBox, "roto-shape-feather")
        transform_panel = w.build_node_panel("t")
        transform_key = next(button for button in transform_panel.findChildren(QPushButton, "key-button")
                             if button.accessibleName() == "translate_y key")
        transform_field = transform_panel.findChild(QDoubleSpinBox, "translate_y-field")
        self.assertEqual(plain_key.property("keyState"), "none")
        self.assertEqual(plain_key.property("keyState"), transform_key.property("keyState"))
        self.assertEqual(plain_field.styleSheet(), transform_field.styleSheet())

    def test_shape_list_names_fit_and_controls_are_labeled_and_described(self):
        listing = self.panel().findChild(QListWidget, "roto-shape-list")
        settle_layout(self.window, listing)
        row = listing.itemWidget(listing.item(0))
        name = row.findChild(QWidget, "roto-shape-row-name-0")
        self.assertGreaterEqual(name.width(), name.fontMetrics().horizontalAdvance(name.text()))
        for object_name in ("roto-shape-key-0", "roto-shape-row-visible-0",
                            "roto-shape-row-locked-0", "roto-shape-row-invert-0"):
            control = row.findChild(QWidget, object_name)
            self.assertIsNotNone(control)
            self.assertTrue(control.toolTip().strip(), object_name)
        self.assertEqual([row.findChild(QCheckBox, f"roto-shape-row-{field}-0").text()
                          for field in ("visible", "locked", "invert")],
                         ["Visible", "Lock", "Invert"])

    def test_colour_swatch_tracks_the_shape_colour_after_a_change(self):
        from unittest.mock import patch
        chooser = self.panel().findChild(QPushButton, "roto-shape-color")
        with patch("nodebased.rotopanel.QColorDialog.getColor", return_value=QColor(32, 96, 160, 255)):
            chooser.click()
        expected = [32 / 255, 96 / 255, 160 / 255, 1.0]
        self.assertTrue(wait_until(lambda: np.allclose(
            self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["color"], expected)))
        swatch = self.panel().findChild(QLabel, "roto-shape-color-swatch")
        self.assertTrue(np.allclose(swatch.property("rgba"), expected))

    def test_properties_scroll_and_focus_survive_roto_and_transform_edits(self):
        w = self.window
        # The test needs a panel taller than its viewport. Whether a Transform panel overflows the default
        # dock depends on the platform's font and screen (10/8 CI Ubuntu: it did not), so make the viewport short.
        w.properties.setMaximumHeight(200)
        settle_layout(w, w.properties.widget())
        for node_id, field_name, value in (("r", "roto-shape-feather", 8.0),
                                           ("t", "mix-field", 0.9)):
            if node_id == "t":
                w.command({"op": "create", "id": "t", "type": "Transform", "pos": [100, 0]},
                          render=False)
                # Select only t: with r also selected, which panel Qt's unordered selectedItems()
                # puts first differs by platform, and the rebuild then showed the Roto panel.
                w.graph.scene().clearSelection()
                w.graph.items_by_id["t"].setSelected(True)
                w.inspect("t")
            bar = w.properties.verticalScrollBar()
            bar.setValue(bar.maximum())
            settle_layout(w, w.properties.widget())
            self.assertGreater(bar.maximum(), 0, f"{node_id} panel should scroll in a short dock")
            field = w.properties.widget().findChild(QDoubleSpinBox, field_name)
            self.assertIsNotNone(field, f"missing {field_name} on {node_id}")
            w.properties.ensureWidgetVisible(field)
            settle_layout(w, w.properties.widget())
            before = bar.value()
            self.assertGreater(before, 0, f"{node_id} field should be focused at a scrolled position")
            field.setFocus()
            APP.processEvents()
            focused = w.focusWidget()
            while focused is not None and focused.objectName() != field_name:
                focused = focused.parentWidget()
            self.assertIsNotNone(focused, f"could not focus {node_id} control before edit")
            field.setValue(value)
            field.editingFinished.emit()
            def restored():
                focused = w.focusWidget()
                while focused is not None and focused.objectName() != field_name:
                    focused = focused.parentWidget()
                return bar.value() == before and focused is not None
            self.assertTrue(wait_until(restored),
                            f"{node_id} panel state changed: scroll {before}->{bar.value()}, "
                            f"focus={getattr(w.focusWidget(), 'objectName', lambda: '')()}, "
                            f"remembered={getattr(w, '_properties_last_focus_path', [])}")

    def test_shape_key_button_keys_all_scalars_and_marks_the_selected_timeline(self):
        self.panel().findChild(QPushButton,"roto-shape-key-0").click()
        shape=self.window.dispatcher.document["node_data"]["r"]["shapes"][0]
        self.assertEqual(shape["points"][0]["x"]["curve"]["keys"], [{"frame":1,"value":10.0}])
        self.assertEqual(shape["opacity"]["curve"]["interpolation"], "smooth")
        self.window.refresh_timeline_marks()
        self.assertIn(1,self.window.frame_slider.key_frames)
        self.assertEqual(self.panel().findChild(QPushButton,"roto-shape-key-0").property("keyState"),"keyed")
        self.window.set_time(current=2)
        APP.processEvents()
        self.assertEqual(self.panel().findChild(QPushButton,"roto-shape-key-0").property("keyState"),"between")
        self.assertIn(1,self.window.frame_slider.key_frames)
        opacity=self.panel().findChild(QDoubleSpinBox,"roto-shape-opacity")
        opacity.setValue(0.7); opacity.editingFinished.emit()
        curve=self.window.dispatcher.document["node_data"]["r"]["shapes"][0]["opacity"]["curve"]
        self.assertEqual(curve["keys"], [{"frame":1,"value":0.5},{"frame":2,"value":0.7}])

    def test_viewer_overlay_follows_the_same_named_tracker_as_the_render(self):
        self.window.command({"op":"create","id":"t","type":"Tracker"})
        self.window.command({"op":"set_tracks","id":"t","tracks":[{
            "name":"feature","enabled":1.0,
            "x":{"value":10.0,"curve":{"interpolation":"linear","keys":[{"frame":1,"value":10.0},{"frame":2,"value":30.0}]}},
            "y":{"value":10.0,"curve":{"interpolation":"linear","keys":[{"frame":1,"value":10.0},{"frame":2,"value":25.0}]}}}]})
        shapes_for_node=copy.deepcopy(self.window.dispatcher.document["node_data"]["r"]["shapes"])
        shapes_for_node[0]["track_link"]={"tracker_id":"t","track_name":"feature"}
        self.window.command({"op":"set_shapes","id":"r","shapes":shapes_for_node})
        self.window.set_time(current=2); APP.processEvents()
        context=self.window.viewer._roto_context()
        resolved=self.window.viewer._roto_resolved(context)
        self.assertAlmostEqual(resolved[0]["points"][0]["x"],30.0)
        self.assertAlmostEqual(resolved[0]["points"][0]["y"],25.0)

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
