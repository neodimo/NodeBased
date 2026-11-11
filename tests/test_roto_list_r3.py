"""Roto step R3: Nuke's stroke/shape list as a tree table in the Roto and RotoPaint panels.

Columns and their order, one undo step per cell edit, selection both ways, the right-click commands, layers as folders
(drag/reorder, a hidden layer hides its children), and old documents unchanged."""
import copy
import os
import unittest

import numpy as np
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPushButton, QTreeWidget, QWidget

import tests.isolation
from tests.waiting import wait_until
from tests.test_roto_ui import APP, roto_document, triangle
from tests.test_roto_artist_panel import box
from tests.test_roto_paint_ui import document as paint_document
from nodebased.app import Window
from nodebased import roto, rotolist, shapes

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
COLUMNS = ["Name", "Visible", "Lock", "Overlay color", "Color", "Invert", "Blending", "Motion blur", "Lifetime", "Source"]


def paint_shape(name, x0=10, **extra):
    return {"kind": "shape", "name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "blend": "over", "visible": True,
            "points": [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
                       for x, y in ((x0, 10), (x0 + 40, 10), (x0 + 20, 60))], **extra}


def paint_stroke(name):
    return {"kind": "stroke", "name": name, "points": [{"x": 20.0, "y": 20.0, "pressure": 1.0}],
            "brush": {"size": 12.0, "hardness": 0.8, "opacity": 1.0, "spacing": 0.2, "strength": 0.2},
            "tool": "paint", "lifetime": {"mode": "all"}, "color": [1.0, 0.0, 0.0, 1.0], "source_offset": [0.0, 0.0],
            "source_frame": "relative", "opacity": 1.0, "blend": "over", "visible": True, "follow_track": None,
            "patch_blend": 0.0}


class RotoListBase(unittest.TestCase):
    key = "r"

    def start(self, window_document):
        self.window = Window(window_document); self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id[self.key].setSelected(True); APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close(); APP.processEvents()

    def doc(self):
        return self.window.dispatcher.document

    def payload(self):
        return self.doc()["node_data"][self.key]

    def tree(self):
        panel = self.window.build_node_panel(self.key)
        self.window.set_properties_widget(panel)
        return panel.findChild(QTreeWidget, self.list_name)

    def undo_depth(self):
        return len(self.window.dispatcher.undo_stack)

    def one_step(self, action):
        """Run an action that edits the document and check it took exactly one undo slot, that undoing restores it."""
        before = copy.deepcopy(self.payload()) if self.key in self.doc().get("node_data", {}) else None
        depth = self.undo_depth()
        action()
        self.assertEqual(self.undo_depth(), depth + 1)
        self.window.command({"op": "undo"})
        after = self.doc().get("node_data", {}).get(self.key)
        self.assertEqual(after, before)
        self.window.command({"op": "redo"})


class RotoListTests(RotoListBase):
    list_name = "roto-shape-list"

    def setUp(self):
        self.shapes = [box("red", 10, 10, 60, 60, color=[1, 0, 0, 1], opacity=.5),
                       box("blue", 30, 30, 80, 80, color=[0, 0, 1, 1], opacity=.25),
                       box("green", 5, 5, 20, 20, color=[0, 1, 0, 1])]
        self.start(roto_document(copy.deepcopy(self.shapes)))

    def test_columns_and_their_order(self):
        tree = self.tree()
        self.assertEqual([tree.headerItem().text(c) for c in range(tree.columnCount())], COLUMNS)
        for name in ("name", "visible", "locked", "overlay", "color", "invert", "blend", "motion_blur", "lifetime", "source"):
            self.assertIsNotNone(tree.findChild(QWidget, f"roto-shape-row-{name}-1"), name)

    def test_each_cell_edits_its_field_and_undoes_in_one_step(self):
        tree = self.tree()
        stored = lambda: self.payload()["shapes"][1]
        self.one_step(lambda: tree.findChild(QCheckBox, "roto-shape-row-visible-1").setChecked(False))
        self.assertFalse(stored()["visible"])
        tree = self.tree()
        self.one_step(lambda: tree.findChild(QCheckBox, "roto-shape-row-locked-1").setChecked(True))
        self.assertTrue(stored()["locked"])
        tree = self.tree()
        self.one_step(lambda: tree.findChild(QCheckBox, "roto-shape-row-invert-1").setChecked(True))
        self.assertTrue(stored()["invert"])
        tree = self.tree()
        self.one_step(lambda: tree.findChild(QCheckBox, "roto-shape-row-motion_blur-1").setChecked(True))
        self.assertTrue(stored()["motion_blur"])
        tree = self.tree()
        blend = tree.findChild(QComboBox, "roto-shape-row-blend-1")
        self.assertEqual([blend.itemData(i) for i in range(blend.count())], list(shapes.BLEND_MODES))
        self.one_step(lambda: blend.setCurrentIndex(blend.findData("multiply")))
        self.assertEqual(stored()["blend_mode"], "multiply")
        tree = self.tree()
        source = tree.findChild(QComboBox, "roto-shape-row-source-1")
        self.one_step(lambda: source.setCurrentIndex(source.findData("foreground")))
        self.assertEqual(stored()["source"], "foreground")
        tree = self.tree()
        name = tree.findChild(QLineEdit, "roto-shape-row-name-1")
        def rename():
            name.setText("azure"); name.editingFinished.emit()
        self.one_step(rename)
        self.assertEqual(stored()["name"], "azure")

    def test_colour_and_overlay_colour_cells_edit_one_step(self):
        tree = self.tree()
        tree.set_color(1, [0.2, 0.4, 0.6, 1.0])
        self.assertEqual([round(c, 3) for c in self.payload()["shapes"][1]["color"]], [0.2, 0.4, 0.6, 1.0])
        self.window.command({"op": "undo"})
        self.assertEqual(self.payload()["shapes"][1]["color"], [0, 0, 1, 1])
        self.one_step(lambda: self.tree().edit_entry(1, "overlay_color", [1.0, 0.5, 0.0]))
        self.assertEqual(self.payload()["shapes"][1]["overlay_color"], [1.0, 0.5, 0.0])

    def test_lifetime_cell_shows_its_type_and_opens_the_editor(self):
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [
            {**self.shapes[0], "lifetime": {"type": "range", "start": 3, "end": 9}}, *self.shapes[1:]]})
        tree = self.tree()
        self.assertEqual(tree.findChild(QPushButton, "roto-shape-row-lifetime-0").text(), "Range 3-9")
        self.assertEqual(tree.findChild(QPushButton, "roto-shape-row-lifetime-1").text(), "All frames")
        tree.findChild(QPushButton, "roto-shape-row-lifetime-1").click(); APP.processEvents()
        self.assertEqual(self.window.viewer.roto_selected_shape_index, 1)
        self.assertIsNotNone(self.window.properties.widget().findChild(QComboBox, "roto-shape-lifetime-type"))

    def test_selection_syncs_both_ways(self):
        tree = self.tree()
        tree.setCurrentRow(2); APP.processEvents()
        self.assertEqual(self.window.viewer.roto_selected_shape_index, 2)
        self.window.viewer.roto_selected_shape_index = 0
        self.window.rebuild_properties_dock(); APP.processEvents()
        panel = self.window.properties.widget().findChild(QWidget, "roto-shapes-panel")
        self.assertEqual(panel.findChild(QTreeWidget, "roto-shape-list").currentRow(), 0)

    def test_multi_select_collects_every_selected_item(self):
        tree = self.tree()
        tree.select_ids([("item", 0), ("item", 2)])
        self.assertEqual(tree.selected_ids(), [("item", 0), ("item", 2)])

    def test_old_document_loads_and_renders_bit_identical(self):
        before = roto.rasterise(shapes.resolve_shapes(self.payload(), 1), 100, 100)
        self.assertNotIn("layers", self.payload())
        self.assertTrue(all("layer" not in s and "overlay_color" not in s for s in self.payload()["shapes"]))
        actual = self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels
        np.testing.assert_array_equal(actual, before)

    def test_overlay_colour_reaches_the_overlay_data_and_changes_no_pixels(self):
        before = self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels.copy()
        self.tree().edit_entry(1, "overlay_color", [1.0, 0.0, 1.0])
        context = self.window.viewer._roto_context()
        resolved = self.window.viewer._roto_resolved(context)
        self.assertEqual(resolved[1]["overlay_color"], [1.0, 0.0, 1.0])
        self.assertNotIn("overlay_color", resolved[0])
        after = self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels
        np.testing.assert_array_equal(before, after)

    def test_context_menu_lists_nukes_commands(self):
        tree = self.tree(); tree.select_ids([("item", 0)])
        menu = tree.build_menu()
        texts = [a.text() for a in menu.actions() if a.text()]
        # Nuke's commands, then "Fit to edges" (lane 9 step I1), "Track points" (step I2) and "Fill in-betweens"
        # (lane 2 step I3) and "Shapes from matte" (lane 2 step I8), past Nuke, last.
        self.assertEqual(texts, ["Add new layer", "Duplicate", "Copy", "Cut", "Paste", "Delete",
                                 "Set lifetime / frame range", "Planar-track", "Fit to edges", "Track points",
                                 "Fill in-betweens", "Shapes from matte"])
        self.assertEqual([a.text() for a in menu.findChild(type(menu), "roto-list-copy-menu").actions()], ["Curve"])
        self.assertEqual([a.text() for a in menu.findChild(type(menu), "roto-list-paste-menu").actions()], ["Spline"])

    def test_add_new_layer(self):
        tree = self.tree()
        self.one_step(tree.cmd_add_layer)
        self.assertEqual(self.payload()["layers"], [{"name": "Layer1"}])

    def test_duplicate(self):
        tree = self.tree(); tree.select_ids([("item", 0)])
        self.one_step(tree.cmd_duplicate)
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["red", "red copy", "blue", "green"])

    def test_copy_cut_and_paste_a_curve(self):
        tree = self.tree(); tree.select_ids([("item", 1)])
        self.assertTrue(tree.cmd_copy())
        self.assertEqual(len(self.payload()["shapes"]), 3)
        tree.select_ids([("item", 2)])
        self.one_step(tree.cmd_paste)
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["red", "blue", "green", "blue 2"])
        tree = self.tree(); tree.select_ids([("item", 0)])
        self.one_step(tree.cmd_cut)
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["blue", "green", "blue 2"])
        self.assertEqual(self.window.roto_list_clipboard["entries"][0]["name"], "red")

    def test_delete(self):
        tree = self.tree(); tree.select_ids([("item", 1)])
        self.one_step(tree.cmd_delete)
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["red", "green"])

    def test_set_lifetime_frame_range(self):
        tree = self.tree(); tree.select_ids([("item", 0), ("item", 2)])
        self.one_step(lambda: tree.cmd_set_lifetime(4, 8))
        for index in (0, 2):
            self.assertEqual(self.payload()["shapes"][index]["lifetime"], {"type": "range", "start": 4, "end": 8})
        self.assertNotIn("lifetime", self.payload()["shapes"][1])

    def test_planar_track_command_calls_the_panels_track_action(self):
        calls = []
        self.window.track_roto_shape = lambda key: calls.append(key) or True
        tree = self.tree(); tree.select_ids([("item", 0)])
        tree.cmd_planar_track()
        self.assertEqual(calls, ["r"])

    # -- layers and folders --------------------------------------------------------------------------------------------------
    def test_drag_into_a_layer_reorder_and_out_again(self):
        tree = self.tree(); tree.cmd_add_layer()
        tree = self.tree()
        self.one_step(lambda: tree.move_ids([("item", 2)], ("layer", "Layer1"), "into"))
        shapes_ = self.payload()["shapes"]
        self.assertEqual([(s["name"], s.get("layer")) for s in shapes_],
                         [("red", None), ("blue", None), ("green", "Layer1")])
        tree = self.tree()
        # the list shows the front-most item first: dropping red above blue puts red in front of blue
        tree.move_ids([("item", 0)], ("item", 1), "above")
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["blue", "red", "green"])
        tree = self.tree()
        tree.move_ids([("item", 2)], ("item", 0), "above")
        self.assertEqual([(s["name"], s.get("layer")) for s in self.payload()["shapes"]],
                         [("blue", None), ("green", None), ("red", None)])
        self.assertEqual(self.payload()["layers"], [{"name": "Layer1"}])

    def test_reordering_changes_stacking(self):
        before = self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels.copy()
        tree = self.tree(); tree.setCurrentRow(0)
        tree.cmd_move(-1)       # up the list: red moves in front of blue
        after = self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["blue", "red", "green"])
        self.assertFalse(np.allclose(before[40, 40], after[40, 40]))

    def test_a_hidden_layer_hides_its_children_and_a_locked_one_locks_them(self):
        tree = self.tree(); tree.cmd_add_layer(); tree = self.tree()
        tree.move_ids([("item", 0), ("item", 1)], ("layer", "Layer1"), "into")
        visible = self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels.copy()
        tree = self.tree()
        self.one_step(lambda: tree.edit_layer("Layer1", "visible", False))
        hidden = self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels
        self.assertFalse(np.allclose(visible, hidden))
        only_green = roto.rasterise(shapes.resolve_shapes({"shapes": [self.shapes[2]]}, 1), 100, 100)
        np.testing.assert_allclose(hidden, only_green, atol=1e-6)
        resolved = shapes.resolve_shapes(self.payload(), 1)
        # the shapes moved into a layer sit in front of green in the stack, red above blue as dropped
        self.assertEqual([s["name"] for s in resolved], ["green", "blue", "red"])
        self.assertEqual([s["visible"] for s in resolved], [True, False, False])
        self.tree().edit_layer("Layer1", "locked", True)
        self.assertEqual([s["locked"] for s in shapes.resolve_shapes(self.payload(), 1)], [False, True, True])

    def test_layer_rename_follows_its_children_and_a_duplicate_layer_copies_them(self):
        tree = self.tree(); tree.cmd_add_layer(); tree = self.tree()
        tree.move_ids([("item", 0)], ("layer", "Layer1"), "into")
        tree = self.tree(); tree.rename_layer("Layer1", "Faces")
        self.assertEqual({s["name"]: s.get("layer") for s in self.payload()["shapes"]}["red"], "Faces")
        tree = self.tree(); tree.select_ids([("layer", "Faces")])
        self.one_step(tree.cmd_duplicate)
        self.assertEqual([layer["name"] for layer in self.payload()["layers"]], ["Faces", "Faces copy"])
        self.assertEqual([(s["name"], s.get("layer")) for s in self.payload()["shapes"]],
                         [("blue", None), ("green", None), ("red", "Faces"), ("red copy", "Faces copy")])

    def test_viewer_edits_keep_the_layers(self):
        tree = self.tree(); tree.cmd_add_layer(); tree = self.tree()
        tree.move_ids([("item", 0)], ("layer", "Layer1"), "into")
        # a Viewer command names no layers; the node's own are kept
        shapes_ = copy.deepcopy(self.payload()["shapes"])
        shapes_[1]["opacity"] = 0.9
        self.window.command({"op": "set_shapes", "id": "r", "shapes": shapes_})
        self.assertEqual(self.payload()["layers"], [{"name": "Layer1"}])
        self.assertEqual({s["name"]: s.get("layer") for s in self.payload()["shapes"]}["red"], "Layer1")

    def test_invalid_layers_are_refused(self):
        bad = copy.deepcopy(self.shapes); bad[0]["layer"] = "Nowhere"
        self.window.command({"op": "set_shapes", "id": "r", "shapes": bad})
        self.assertNotIn("layer", self.payload()["shapes"][0])
        self.assertIsNotNone(self.window.last_command_error)


class RotoPaintListTests(RotoListBase):
    key = "paint"
    list_name = "rotopaint-layer-list"

    def setUp(self):
        self.start(paint_document())
        self.items = [paint_shape("shape1"), paint_stroke("stroke1"), paint_shape("shape2", 40)]
        self.window.command({"op": "set_paint_items", "id": "paint", "items": copy.deepcopy(self.items)})

    def test_columns_and_their_order(self):
        tree = self.tree()
        self.assertEqual([tree.headerItem().text(c) for c in range(tree.columnCount())], COLUMNS)
        self.assertEqual(tree.count(), 3)
        for name in ("name", "visible", "locked", "overlay", "color", "invert", "blend", "motion_blur", "lifetime", "source"):
            self.assertIsNotNone(tree.findChild(QWidget, f"rotopaint-row-{name}-0"), name)
        # the guide's list inverts "shape/stroke/group" and every item has a rendered colour, so these cells are live on both kinds
        for index in (0, 1):
            self.assertTrue(tree.findChild(QCheckBox, f"rotopaint-row-invert-{index}").isEnabled())
            self.assertTrue(tree.findChild(QPushButton, f"rotopaint-row-color-{index}").isEnabled())

    def test_each_cell_edits_its_field_and_undoes_in_one_step(self):
        tree = self.tree()
        stored = lambda i: self.payload()["items"][i]
        self.one_step(lambda: tree.findChild(QCheckBox, "rotopaint-row-visible-1").setChecked(False))
        self.assertFalse(stored(1)["visible"])
        tree = self.tree()
        self.one_step(lambda: tree.findChild(QCheckBox, "rotopaint-row-locked-0").setChecked(True))
        self.assertTrue(stored(0)["locked"])
        tree = self.tree()
        self.one_step(lambda: tree.findChild(QCheckBox, "rotopaint-row-invert-0").setChecked(True))
        self.assertTrue(stored(0)["invert"])
        tree = self.tree()
        blend = tree.findChild(QComboBox, "rotopaint-row-blend-1")
        self.assertEqual([blend.itemData(i) for i in range(blend.count())], list(shapes.PAINT_BLEND_MODES))
        self.assertEqual([blend.itemData(i) for i in range(4)], ["over", "add", "multiply", "screen"])
        self.one_step(lambda: blend.setCurrentIndex(blend.findData("screen")))
        self.assertEqual(stored(1)["blend"], "screen")
        tree = self.tree()
        self.one_step(lambda: tree.findChild(QCheckBox, "rotopaint-row-motion_blur-1").setChecked(True))
        self.assertTrue(stored(1)["motion_blur"])
        tree = self.tree()
        source = tree.findChild(QComboBox, "rotopaint-row-source-1")
        self.one_step(lambda: source.setCurrentIndex(source.findData("background")))
        self.assertEqual(stored(1)["source"], "background")
        tree = self.tree()
        self.one_step(lambda: tree.set_color(1, [0.1, 0.2, 0.3, 1.0]))
        self.assertEqual(stored(1)["color"], [0.1, 0.2, 0.3, 1.0])
        self.one_step(lambda: self.tree().set_color(0, [0.9, 0.2, 0.1, 1.0]))
        self.assertEqual(stored(0)["color"], [0.9, 0.2, 0.1, 1.0])
        self.one_step(lambda: self.tree().findChild(QCheckBox, "rotopaint-row-invert-1").setChecked(True))
        self.assertTrue(stored(1)["invert"])
        self.one_step(lambda: self.tree().edit_entry(0, "overlay_color", [0.0, 1.0, 0.0]))
        self.assertEqual(stored(0)["overlay_color"], [0.0, 1.0, 0.0])

    def test_lifetime_cell_shows_the_stroke_mode(self):
        entries = copy.deepcopy(self.items)
        entries[1]["lifetime"] = {"mode": "range", "first": 2, "last": 5}
        self.window.command({"op": "set_paint_items", "id": "paint", "items": entries})
        tree = self.tree()
        self.assertEqual(tree.findChild(QPushButton, "rotopaint-row-lifetime-1").text(), "Range 2-5")
        self.assertEqual(tree.findChild(QPushButton, "rotopaint-row-lifetime-0").text(), "All frames")

    def test_selection_syncs_both_ways(self):
        tree = self.tree()
        tree.setCurrentRow(2); APP.processEvents()
        self.assertEqual(self.window.viewer.paint_selected_item_index, 2)
        self.window.viewer.paint_selected_item_index = 1
        self.assertEqual(self.tree().currentRow(), 1)

    def test_old_document_renders_bit_identical(self):
        self.assertNotIn("layers", self.payload())
        before = self.window.evaluator.evaluate_raster(self.doc(), target="paint").pixels.copy()
        shapes_ = shapes.resolve_paint_items(self.payload(), 1)
        self.assertTrue(all("layer" not in item and "overlay_color" not in item for item in shapes_))
        self.assertTrue(np.isfinite(before).all())

    def test_overlay_colour_changes_no_pixels(self):
        before = self.window.evaluator.evaluate_raster(self.doc(), target="paint").pixels.copy()
        self.tree().edit_entry(0, "overlay_color", [1.0, 0.0, 0.0])
        resolved = shapes.resolve_paint_items(self.payload(), 1)
        self.assertEqual(resolved[0]["overlay_color"], [1.0, 0.0, 0.0])
        after = self.window.evaluator.evaluate_raster(self.doc(), target="paint").pixels
        np.testing.assert_array_equal(before, after)

    def test_commands_each_take_one_undo_step(self):
        tree = self.tree(); tree.select_ids([("item", 0)])
        self.one_step(tree.cmd_duplicate)
        self.assertEqual([i["name"] for i in self.payload()["items"]], ["shape1", "shape1 copy", "stroke1", "shape2"])
        tree = self.tree(); tree.select_ids([("item", 0)])
        self.one_step(tree.cmd_add_layer)
        tree = self.tree(); tree.select_ids([("item", 3)])
        self.one_step(tree.cmd_delete)
        self.assertEqual(len(self.payload()["items"]), 3)
        tree = self.tree(); tree.select_ids([("item", 0)])
        tree.cmd_copy()
        tree = self.tree(); tree.select_ids([("item", 2)])
        self.one_step(tree.cmd_paste)
        self.assertEqual(self.payload()["items"][-1]["kind"], "shape")
        stroke = next(i for i, item in enumerate(self.payload()["items"]) if item["kind"] == "stroke")
        tree = self.tree(); tree.select_ids([("item", stroke)])
        self.one_step(lambda: tree.cmd_set_lifetime(2, 6))
        self.assertEqual(self.payload()["items"][stroke]["lifetime"], {"mode": "range", "first": 2, "last": 6})

    def test_strokes_have_no_curve_to_copy_but_shapes_paste_across_node_kinds(self):
        tree = self.tree(); tree.select_ids([("item", 1)])
        self.assertFalse(tree.cmd_copy())
        tree.select_ids([("item", 0)])
        self.assertTrue(tree.cmd_copy())
        converted = rotolist.convert_entry(self.window.roto_list_clipboard["entries"][0], to_roto=True)
        shapes.validate_shape(converted, "x")
        self.assertEqual(converted["blend_mode"], "over")
        back = rotolist.convert_entry(converted, to_roto=False)
        self.assertEqual(back["kind"], "shape"); self.assertEqual(back["blend"], "over")

    def test_layers_reorder_and_a_hidden_layer_hides_its_children(self):
        tree = self.tree(); tree.cmd_add_layer(); tree = self.tree()
        tree.move_ids([("item", 0), ("item", 1)], ("layer", "Layer1"), "into")
        items = self.payload()["items"]
        self.assertEqual([(i["name"], i.get("layer")) for i in items],
                         [("shape2", None), ("stroke1", "Layer1"), ("shape1", "Layer1")])
        tree = self.tree()
        tree.move_ids([("item", 0)], ("item", 1), "above")      # next to stroke1 puts it in the layer, above stroke1
        self.assertEqual([(i["name"], i.get("layer")) for i in self.payload()["items"]],
                         [("stroke1", "Layer1"), ("shape2", "Layer1"), ("shape1", "Layer1")])
        self.tree().edit_layer("Layer1", "visible", False)
        resolved = shapes.resolve_paint_items(self.payload(), 1)
        self.assertEqual([i["visible"] for i in resolved], [False, False, False])

    def test_layers_survive_viewer_item_edits_and_cannot_nest_into_themselves(self):
        tree = self.tree(); tree.cmd_add_layer(); tree = self.tree(); tree.cmd_add_layer(); tree = self.tree()
        self.assertTrue(tree.move_ids([("layer", "Layer2")], ("layer", "Layer1"), "into"))
        tree = self.tree()
        self.assertFalse(tree.move_ids([("layer", "Layer1")], ("layer", "Layer2"), "into"))
        items = copy.deepcopy(self.payload()["items"]); items[0]["opacity"] = 0.5
        self.window.command({"op": "set_paint_items", "id": "paint", "items": items})
        self.assertEqual([l["name"] for l in self.payload()["layers"]], ["Layer1", "Layer2"])
        self.assertEqual(self.payload()["layers"][1]["layer"], "Layer1")


class FrontFirstListTests(RotoListBase):
    """Nuke's order: the top row of the list is the item composited last, in front; the stored order stays render order."""
    list_name = "roto-shape-list"

    def setUp(self):
        self.shapes = [box("back", 10, 10, 60, 60, color=[1, 0, 0, 1]),
                       box("middle", 30, 30, 80, 80, color=[0, 0, 1, 1]),
                       box("front", 50, 50, 95, 95, color=[0, 1, 0, 1])]
        self.start(roto_document(copy.deepcopy(self.shapes)))

    def rows(self):
        tree = self.tree()
        order = []
        iterator = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        for item in iterator:
            order.append(tree.parse_id(item.data(0, 0x100)))
        return order

    def pixel(self, x, y):
        return self.window.evaluator.evaluate_raster(self.doc(), target="r").pixels[y, x][:3]

    def names_top_down(self):
        entries = self.payload()["shapes"]
        return [entries[ref]["name"] for kind, ref in self.rows() if kind == "item"]

    def test_the_top_row_is_the_front_most_item_and_the_stored_order_is_unchanged(self):
        self.assertEqual(self.names_top_down(), ["front", "middle", "back"])
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["back", "middle", "front"])
        np.testing.assert_allclose(self.pixel(55, 55), [0, 1, 0], atol=1e-4)      # front covers the others
        np.testing.assert_allclose(self.pixel(40, 40), [0, 0, 1], atol=1e-4)      # middle covers back

    def test_dropping_above_an_item_puts_the_dragged_one_in_front_of_it(self):
        np.testing.assert_allclose(self.pixel(40, 40), [0, 0, 1], atol=1e-4)
        tree = self.tree()
        self.one_step(lambda: tree.move_ids([("item", 0)], ("item", 1), "above"))      # back dropped above middle
        self.assertEqual(self.names_top_down(), ["front", "back", "middle"])
        np.testing.assert_allclose(self.pixel(40, 40), [1, 0, 0], atol=1e-4)      # back is now in front of middle
        np.testing.assert_allclose(self.pixel(55, 55), [0, 1, 0], atol=1e-4)      # still behind front

    def test_dropping_below_an_item_puts_the_dragged_one_behind_it(self):
        tree = self.tree()
        tree.move_ids([("item", 2)], ("item", 0), "below")      # front dropped below back
        self.assertEqual(self.names_top_down(), ["middle", "back", "front"])
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["front", "back", "middle"])
        np.testing.assert_allclose(self.pixel(55, 55), [0, 0, 1], atol=1e-4)      # middle now covers front

    def test_move_up_and_down_follow_the_list(self):
        tree = self.tree(); tree.setCurrentRow(0)          # back, bottom row
        tree.cmd_move(-1)
        self.assertEqual(self.names_top_down(), ["front", "back", "middle"])
        np.testing.assert_allclose(self.pixel(40, 40), [1, 0, 0], atol=1e-4)
        tree = self.tree()
        tree.cmd_move(1)
        self.assertEqual(self.names_top_down(), ["front", "middle", "back"])
        self.assertFalse(self.tree().cmd_move(1))      # already at the bottom

    def test_a_dropped_item_lands_at_the_top_of_its_layer(self):
        tree = self.tree(); tree.cmd_add_layer(); tree = self.tree()
        self.assertEqual(self.rows()[0], ("layer", "Layer1"))      # a new layer is the front-most thing in the list
        tree.move_ids([("item", 0)], ("layer", "Layer1"), "into")
        tree = self.tree()
        tree.move_ids([("item", 0)], ("layer", "Layer1"), "into")      # middle (now item 0) dropped into the layer after back
        names = [(s["name"], s.get("layer")) for s in self.payload()["shapes"]]
        self.assertEqual(names, [("front", None), ("back", "Layer1"), ("middle", "Layer1")])
        # middle is the layer's top row, so it renders in front of back
        np.testing.assert_allclose(self.pixel(40, 40), [0, 0, 1], atol=1e-4)

    def test_duplicate_appears_directly_above_the_original_in_front_of_it(self):
        tree = self.tree(); tree.select_ids([("item", 1)])
        self.one_step(tree.cmd_duplicate)
        self.assertEqual(self.names_top_down(), ["front", "middle copy", "middle", "back"])
        self.assertEqual([s["name"] for s in self.payload()["shapes"]], ["back", "middle", "middle copy", "front"])

    def test_paste_goes_in_front_of_the_selected_item(self):
        tree = self.tree(); tree.select_ids([("item", 0)])
        self.assertTrue(tree.cmd_copy())
        tree = self.tree(); tree.select_ids([("item", 1)])
        self.one_step(tree.cmd_paste)
        self.assertEqual(self.names_top_down(), ["front", "back 2", "middle", "back"])

    def test_a_new_shape_from_the_viewer_is_the_top_row(self):
        shapes_ = copy.deepcopy(self.payload()["shapes"]) + [box("newest", 0, 0, 20, 20)]
        self.window.command({"op": "set_shapes", "id": "r", "shapes": shapes_})
        self.assertEqual(self.names_top_down()[0], "newest")


class TreeArithmeticTests(unittest.TestCase):
    def test_tree_round_trips_and_keeps_empty_layers(self):
        entries = [{"name": "a"}, {"name": "b", "layer": "L"}, {"name": "c", "layer": "M"}]
        layers = [{"name": "L"}, {"name": "M", "layer": "L"}, {"name": "E"}]
        top = rotolist.build_tree(entries, layers)
        self.assertEqual([n["id"] for n in top], [("item", 0), ("layer", "L"), ("layer", "E")])
        new_entries, new_layers, order = rotolist.flatten(top, entries, layers)
        self.assertEqual(new_entries, entries); self.assertEqual(order, [0, 1, 2])
        self.assertEqual([l["name"] for l in new_layers], ["L", "M", "E"])

    def test_move_above_below_into_end(self):
        entries = [{"name": n} for n in "abcd"]
        layers = [{"name": "L"}]
        top = rotolist.build_tree(entries, layers)
        rotolist.move_nodes(top, [("item", 3)], ("item", 0), "above")
        rotolist.move_nodes(top, [("item", 0)], ("layer", "L"), "into")
        new_entries, new_layers, order = rotolist.flatten(top, entries, layers)
        self.assertEqual([e["name"] for e in new_entries], ["d", "b", "c", "a"])
        self.assertEqual(new_entries[3]["layer"], "L")


if __name__ == "__main__":
    unittest.main()
