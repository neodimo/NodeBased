"""Roto step RW2: the S key, Nuke's full blending list, layer transforms and keyable motion blur in the list."""
import copy
import os
import tempfile
import unittest

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QGroupBox

import tests.isolation
from tests.test_roto_walkthrough import load_walkthrough
from nodebased import roto, shapes
from nodebased.imaging import Evaluator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def square(name, x0, y0, x1, y1, color=(1.0, 1.0, 1.0, 1.0), **extra):
    points = [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
              for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return {"name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": points,
            "color": list(color), **extra}


MERGE_MODES = roto.MERGE_BLEND_MODES


class BlendingModeTests(unittest.TestCase):
    def render(self, blend):
        payload = {"shapes": [square("below", 4, 4, 24, 24, (0.8, 0.4, 0.2, 1.0)),
                              square("above", 14, 14, 34, 34, (0.3, 0.6, 0.9, 1.0), blend_mode=blend)]}
        shapes.validate_shape(payload["shapes"][1], "above")
        return roto.rasterise(shapes.resolve_shapes(payload, 1), 40, 40)

    def test_the_list_holds_nukes_modes_and_each_is_valid(self):
        self.assertEqual(len(shapes.BLEND_MODES), len(set(shapes.BLEND_MODES)))
        self.assertTrue(set(MERGE_MODES) <= set(shapes.BLEND_MODES))
        self.assertTrue(set(shapes.BLEND_MODES) - set(MERGE_MODES) >= {"over", "plus", "minus", "multiply", "screen"})
        self.assertTrue(set(shapes.BLEND_MODES) >= set(__import__("nodebased.core", fromlist=["x"]).MERGE_OPERATIONS))

    def test_each_merge_mode_is_the_merge_nodes_formula_on_the_overlap(self):
        a = np.asarray([[[0.3, 0.6, 0.9, 1.0]]], dtype=np.float32)       # the shape: colour with its coverage 1 as alpha
        b = np.asarray([[[0.8, 0.4, 0.2, 1.0]]], dtype=np.float32)       # what is below
        for mode in MERGE_MODES:
            image = self.render(mode)
            expected = np.clip(Evaluator._merge_op(mode, a, b), 0.0, 1.0)[0, 0]
            self.assertTrue(np.allclose(image[18, 18], expected, atol=1e-6), (mode, image[18, 18], expected))
            self.assertTrue(np.isfinite(image).all(), mode)

    def test_each_merge_mode_over_empty_background_and_beside_the_shape(self):
        a = np.asarray([[[0.3, 0.6, 0.9, 1.0]]], dtype=np.float32)
        empty = np.zeros((1, 1, 4), dtype=np.float32)
        for mode in MERGE_MODES:
            image = self.render(mode)
            self.assertTrue(np.allclose(image[30, 30], np.clip(Evaluator._merge_op(mode, a, empty), 0, 1)[0, 0], atol=1e-6), mode)
            self.assertTrue(np.array_equal(image[8, 8], np.asarray([0.8, 0.4, 0.2, 1.0], dtype=np.float32)), mode)   # below only

    def test_the_first_ten_modes_render_as_they_did(self):
        over = self.render("over")
        self.assertTrue(np.allclose(over[18, 18], [0.3, 0.6, 0.9, 1.0]))
        self.assertTrue(np.allclose(self.render("plus")[18, 18], [1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(self.render("multiply")[18, 18], [0.24, 0.24, 0.18, 1.0]))

    def test_list_cell_offers_the_whole_list(self):
        walk_module = load_walkthrough()
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            w = walk_module.Walk(app, folder, walk_module.Result(0, "t", [], "t"))
            try:
                w.fresh("roto", shapes=[square("one", 100, 100, 200, 200)])
                combo = w.panel().findChild(QComboBox, "roto-shape-row-blend-0")
                self.assertIsNotNone(combo)
                self.assertEqual([combo.itemData(i) for i in range(combo.count())], list(shapes.BLEND_MODES))
                combo.setCurrentIndex(combo.findData("atop"))
                w.settle()
                self.assertEqual(w.shape_entries()[0]["blend_mode"], "atop")
            finally:
                w.close()


def pixel_set(image, threshold=0.5):
    return np.argwhere(image[..., 3] > threshold)


class LayerTransformTests(unittest.TestCase):
    def payload(self, transform=None, nested=False):
        layers = [{"name": "group"}]
        shape_list = [square("a", 10, 10, 20, 20, layer="group"), square("b", 40, 10, 50, 20, layer="group"),
                      square("c", 10, 40, 20, 50)]
        if transform is not None:
            layers[0]["transform"] = transform
        if nested:
            layers = [{"name": "outer", "transform": {"translate_x": 5.0}}, {**layers[0], "layer": "outer"}]
        return {"shapes": shape_list, "layers": layers}

    def centroid(self, resolved, index):
        points = resolved[index]["points"]
        return (sum(p["x"] for p in points) / len(points), sum(p["y"] for p in points) / len(points))

    def test_no_transform_and_identity_transform_render_bit_identical(self):
        plain = roto.rasterise(shapes.resolve_shapes(self.payload(), 1), 64, 64)
        with_identity = roto.rasterise(shapes.resolve_shapes(self.payload({"scale_x": 1.0, "center_x": 33.0}), 1), 64, 64)
        self.assertTrue(np.array_equal(plain, with_identity))
        legacy = self.payload()
        legacy["layers"][0].pop("transform", None)
        self.assertTrue(np.array_equal(plain, roto.rasterise(shapes.resolve_shapes(legacy, 1), 64, 64)))

    def test_translate_moves_both_children_and_nothing_else(self):
        base = shapes.resolve_shapes(self.payload(), 1)
        moved = shapes.resolve_shapes(self.payload({"translate_x": 7.0, "translate_y": 3.0}), 1)
        for index, delta in ((0, (7.0, 3.0)), (1, (7.0, 3.0)), (2, (0.0, 0.0))):
            before, after = self.centroid(base, index), self.centroid(moved, index)
            self.assertAlmostEqual(after[0] - before[0], delta[0]); self.assertAlmostEqual(after[1] - before[1], delta[1])
        image = roto.rasterise(moved, 64, 64)
        self.assertGreater(image[14, 22, 3], 0.9)          # a: x 10..20 moved to 17..27
        self.assertLess(image[14, 12, 3], 0.1)
        self.assertGreater(image[16, 52, 3], 0.9)          # b: x 40..50 moved to 47..57
        self.assertGreater(image[45, 15, 3], 0.9)          # c, outside the layer, did not move

    def test_scale_and_rotate_work_about_the_default_centre_of_the_layers_contents(self):
        # The contents span x 10..50, y 10..20: centre (30, 15).
        scaled = shapes.resolve_shapes(self.payload({"scale_x": 2.0, "scale_y": 2.0}), 1)
        xs = [p["x"] for p in scaled[0]["points"]] + [p["x"] for p in scaled[1]["points"]]
        self.assertAlmostEqual(min(xs), -10.0); self.assertAlmostEqual(max(xs), 70.0)
        rotated = shapes.resolve_shapes(self.payload({"rotate": 180.0}), 1)
        a = self.centroid(rotated, 0); b = self.centroid(rotated, 1)
        self.assertAlmostEqual(a[0], 45.0); self.assertAlmostEqual(b[0], 15.0)
        self.assertAlmostEqual(a[1], 15.0, places=6)
        explicit = shapes.resolve_shapes(self.payload({"rotate": 180.0, "center_x": 0.0, "center_y": 0.0}), 1)
        self.assertAlmostEqual(self.centroid(explicit, 0)[0], -15.0)

    def test_skew_and_nested_layers_compose_innermost_out(self):
        nested = shapes.resolve_shapes(self.payload({"translate_x": 1.0}, nested=True), 1)
        base = shapes.resolve_shapes(self.payload(), 1)
        self.assertAlmostEqual(self.centroid(nested, 0)[0] - self.centroid(base, 0)[0], 6.0)
        skewed = shapes.resolve_shapes(self.payload({"skew_x": 45.0}), 1)
        self.assertNotAlmostEqual(skewed[0]["points"][0]["x"], base[0]["points"][0]["x"])

    def test_layer_transform_is_animatable_and_validated(self):
        animated = self.payload({"translate_x": {"value": 0.0, "curve": {"interpolation": "linear",
                                 "keys": [{"frame": 1, "value": 0.0}, {"frame": 11, "value": 10.0}]}}})
        shapes.validate_layers(animated["layers"], animated["shapes"], "p", "shapes")
        at6 = shapes.resolve_shapes(animated, 6)
        base = shapes.resolve_shapes(self.payload(), 6)
        self.assertAlmostEqual(self.centroid(at6, 1)[0] - self.centroid(base, 1)[0], 5.0)
        with self.assertRaises(ValueError):
            shapes.validate_layers([{"name": "g", "transform": {"bogus": 1}}], [], "p", "shapes")
        with self.assertRaises(ValueError):
            shapes.validate_layers([{"name": "g", "transform": {"scale_x": 1e9}}], [], "p", "shapes")

    def test_hidden_layer_and_motion_blur_children_still_resolve(self):
        payload = self.payload({"translate_x": 4.0})
        payload["layers"][0]["visible"] = False
        resolved = shapes.resolve_shapes(payload, 1)
        self.assertFalse(resolved[0]["visible"])
        blurred = self.payload({"translate_x": 4.0}); blurred["shapes"][0]["motion_blur"] = True
        shapes.resolve_shapes(blurred, 1)
        roto.rasterise(shapes.resolve_shapes(blurred, 1), 64, 64)


class LayerTransformUiTests(unittest.TestCase):
    def open(self, w):
        w.fresh("roto", shapes=[square("a", 100, 100, 200, 200), square("b", 300, 150, 400, 250), square("c", 100, 320, 200, 400)])
        inside = w.shape_entries()
        inside[0]["layer"] = inside[1]["layer"] = "group"
        w.window.command({"op": "set_shapes", "id": "r", "shapes": inside, "layers": [{"name": "group"}]})
        w.window.rebuild_properties_dock(); w.settle()

    def test_layer_row_shows_transform_controls_and_the_box_edits_the_layer(self):
        walk_module = load_walkthrough()
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            w = walk_module.Walk(app, folder, walk_module.Result(0, "t", [], "t"))
            try:
                self.open(w)
                tree = w.panel().findChild(__import__("PySide6.QtWidgets", fromlist=["x"]).QTreeWidget, "roto-shape-list")
                self.assertIsNotNone(tree)
                self.assertIsNone(w.panel().findChild(QGroupBox, "roto-layer-transform"))
                tree.setCurrentItem(tree.tree_items[("layer", "group")]); w.settle(); w.wait(lambda: False, 0.2)
                panel = w.panel()
                self.assertIsNotNone(panel.findChild(QGroupBox, "roto-layer-transform"))
                self.assertEqual(w.viewer.roto_list_layer, "group")
                self.assertEqual(len(w.viewer.roto_selected_points), 8)           # every point of both children
                # Edit Translate X in the panel: one undo step, both children move, the third shape does not.
                spin = panel.findChild(QDoubleSpinBox, "roto-layer-transform-translate_x")
                self.assertIsNotNone(spin)
                depth = w.undo_depth()
                spin.setValue(25.0); spin.editingFinished.emit(); w.settle()
                self.assertEqual(w.undo_depth(), depth + 1)
                layer = w.payload["layers"][0]
                self.assertEqual(layer["transform"]["translate_x"], 25.0)
                self.assertNotIn("transform", w.shape_entries()[0])
                resolved = shapes.resolve_shapes(w.payload, 1)
                self.assertAlmostEqual(resolved[0]["points"][0]["x"], 125.0)
                self.assertAlmostEqual(resolved[1]["points"][0]["x"], 325.0)
                self.assertAlmostEqual(resolved[2]["points"][0]["x"], 100.0)
                # The transform box, with the layer row selected, moves the layer the same way (autokey on: no key yet, so a value).
                tree.setCurrentItem(tree.tree_items[("layer", "group")]); w.settle()
                w.viewer.roto_transform_box_enabled = True
                depth = w.undo_depth()
                w.drag((275, 125), (305, 145))
                self.assertEqual(w.undo_depth(), depth + 1)
                layer = w.payload["layers"][0]
                translate_x = shapes.resolve_scalar(layer["transform"]["translate_x"], 1, "translate_x")
                translate_y = shapes.resolve_scalar(layer["transform"]["translate_y"], 1, "translate_y")
                self.assertGreater(translate_x, 25.0 + 20.0)
                self.assertGreater(translate_y, 10.0)
                self.assertNotIn("transform", w.shape_entries()[0])               # the shapes themselves are untouched
                # Scaling by the box corner (the drag state the mouse builds) edits the layer's scale, again as one undo step.
                from PySide6.QtCore import QPointF
                context = w.viewer._roto_context(); resolved = w.viewer._roto_resolved(context)
                selected, bounds, _, pivot = w.viewer._roto_box_geometry(resolved, context[3])
                self.assertEqual(len(selected), 8)
                corner = bounds.bottomRight()
                action = w.viewer._roto_box_action(corner, bounds, abs(w.viewer.transform().m11()), pivot)[0]
                self.assertTrue(action.startswith("scale"))
                depth = w.undo_depth()
                w.viewer.roto_drag = {"kind": "box", "kind_name": action, "handle": action[6:], "selected": selected, "bounds": bounds,
                                      "pivot": pivot, "start": corner,
                                      "scene": QPointF(pivot.x() + 2.0 * (corner.x() - pivot.x()), pivot.y() + 2.0 * (corner.y() - pivot.y())),
                                      "tier": context[3], "alt_layer": "group",
                                      "initial": {(si, pi): dict(resolved[si]["points"][pi]) for si, pi in selected}}
                w.viewer._commit_roto_drag(); w.settle()
                self.assertEqual(w.undo_depth(), depth + 1)
                layer = w.payload["layers"][0]
                self.assertAlmostEqual(shapes.resolve_scalar(layer["transform"]["scale_x"], 1, "scale_x"), 2.0, places=3)
                self.assertNotIn("transform", w.shape_entries()[0])
            finally:
                w.close()


class MotionBlurCellTests(unittest.TestCase):
    def cell(self, w, index=0):
        from PySide6.QtWidgets import QCheckBox, QTreeWidget
        tree = w.panel().findChild(QTreeWidget, "roto-shape-list")
        return tree.findChild(QCheckBox, f"roto-shape-row-motion_blur-{index}")

    def test_cell_shows_and_sets_shutter_keys_with_autokey_on(self):
        walk_module = load_walkthrough()
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            w = walk_module.Walk(app, folder, walk_module.Result(0, "t", [], "t"))
            try:
                w.fresh("roto", shapes=[square("a", 100, 100, 200, 200)], last=24)
                self.assertEqual(self.cell(w).property("keyState"), "none")
                depth = w.undo_depth()
                self.cell(w).setChecked(True); w.settle()
                self.assertEqual(w.undo_depth(), depth + 1)                       # one commit, one undo step
                shape = w.shape_entries()[0]
                self.assertTrue(shape["motion_blur"])
                keys = shape["motion_blur_shutter"]["curve"]["keys"]
                self.assertEqual([(k["frame"], k["value"]) for k in keys], [(1, 0.5)])
                self.assertEqual(self.cell(w).property("keyState"), "key")        # the cell shows the key at this frame
                self.assertIn("keyed at frame 1", self.cell(w).toolTip())
                w.goto(10)
                self.assertEqual(self.cell(w).property("keyState"), "animated")   # keyed elsewhere
                self.assertTrue(self.cell(w).isChecked())                         # a lone key holds
                self.cell(w).setChecked(False); w.settle()                        # blur off from frame 10
                keys = w.shape_entries()[0]["motion_blur_shutter"]["curve"]["keys"]
                self.assertEqual([(k["frame"], k["value"]) for k in keys], [(1, 0.5), (10, 0.0)])
                self.assertEqual(self.cell(w).property("keyState"), "key")
                self.assertFalse(self.cell(w).isChecked())
                at1 = shapes.resolve_shapes(w.payload, 1)[0]
                at10 = shapes.resolve_shapes(w.payload, 10)[0]
                self.assertTrue(at1["motion_blur"] and at1.get("_motion_samples"))
                self.assertFalse(at10.get("_motion_samples"))                     # the shutter is 0 there: sharp
                w.goto(1)
                self.assertTrue(self.cell(w).isChecked())
                self.assertEqual(self.cell(w).property("keyState"), "key")
            finally:
                w.close()

    def test_autokey_off_only_sets_the_flag(self):
        walk_module = load_walkthrough()
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            w = walk_module.Walk(app, folder, walk_module.Result(0, "t", [], "t"))
            try:
                w.fresh("roto", shapes=[square("a", 100, 100, 200, 200)], last=24)
                w.viewer.roto_toolbar.set_setting("r", "Select Points", "autokey", False)
                w.window.rebuild_properties_dock(); w.settle()
                self.cell(w).setChecked(True); w.settle()
                shape = w.shape_entries()[0]
                self.assertTrue(shape["motion_blur"])
                self.assertNotIn("motion_blur_shutter", shape)
                self.cell(w).setChecked(False); w.settle()
                self.assertFalse(w.shape_entries()[0]["motion_blur"])
            finally:
                w.close()

    def test_a_cleared_cell_on_an_unanimated_item_clears_the_flag_with_autokey_on(self):
        walk_module = load_walkthrough()
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            w = walk_module.Walk(app, folder, walk_module.Result(0, "t", [], "t"))
            try:
                w.fresh("roto", shapes=[square("a", 100, 100, 200, 200, motion_blur=True)], last=24)
                self.assertTrue(self.cell(w).isChecked())
                self.cell(w).setChecked(False); w.settle()
                shape = w.shape_entries()[0]
                self.assertFalse(shape["motion_blur"])
                self.assertNotIn("motion_blur_shutter", shape)
            finally:
                w.close()


class SelectionModeKeyTests(unittest.TestCase):
    def test_s_cycles_the_modes_of_the_selected_tool(self):
        walk_module = load_walkthrough()
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            w = walk_module.Walk(app, folder, walk_module.Result(0, "t", [], "t"))
            try:
                w.fresh("roto", shapes=[square("a", 100, 100, 200, 200)])
                w.tool("Select", "Select All")
                seen = []
                for _ in range(5):
                    seen.append(w.viewer.roto_selection_tool)
                    w.key_press(Qt.Key.Key_S)
                self.assertEqual(seen, ["Select All", "Select Splines", "Select Points", "Select Feather Points", "Select All"])
                w.tool("Shapes", "Bezier")
                w.key_press(Qt.Key.Key_S)
                self.assertEqual(w.viewer.roto_toolbar.tool_by_group["Shapes"], "Cusped Bezier")
                w.key_press(Qt.Key.Key_S, walk_module.SHIFT)                     # a modified S is not the cycle
                self.assertEqual(w.viewer.roto_toolbar.tool_by_group["Shapes"], "Cusped Bezier")
            finally:
                w.close()


if __name__ == "__main__":
    unittest.main()
