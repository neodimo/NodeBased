import unittest

from nodebased.core import LIMITS, SPECS
from nodebased.knobs import knob_layout, resolve_kind


class KnobLayoutTests(unittest.TestCase):
    def test_every_schema_param_is_grouped_once(self):
        for node_type, spec in SPECS.items():
            params = [param for group in knob_layout(node_type) for param in group.params]
            self.assertEqual(set(params), set(spec["params"]), node_type)
            self.assertEqual(len(params), len(set(params)), node_type)

    def test_compound_groups(self):
        transform = knob_layout("Transform")
        self.assertIn(("translate_x", "translate_y"), [group.params for group in transform])
        self.assertEqual(resolve_kind("Transform", "translate_x", 0.0), "xy")
        constant = knob_layout("Constant")
        self.assertIn(("red", "green", "blue", "alpha"), [group.params for group in constant])

    def test_float_soft_ranges_are_within_hard_ranges(self):
        for node_type in ("ColorCorrect", "Grade", "Blur", "Crop"):
            for group in knob_layout(node_type):
                if group.kind == "float_slider":
                    hard = LIMITS[group.params[0]]
                    self.assertIsNotNone(group.soft_range)
                    self.assertGreaterEqual(group.soft_range[0], hard[0])
                    self.assertLessEqual(group.soft_range[1], hard[1])

    def test_special_kinds(self):
        for param in ("invert", "apply_translate", "apply_rotate", "apply_scale"):
            self.assertEqual(resolve_kind("Roto" if param == "invert" else "Tracker", param, 1), "bool")
        self.assertEqual(resolve_kind("Switch", "which", 0), "int")
        self.assertEqual(resolve_kind("Read", "subimage", 0), "int")
        self.assertEqual(resolve_kind("Read", "path", ""), "file_read")
        self.assertEqual(resolve_kind("Write", "path", ""), "file_write")

    def test_enums(self):
        enum_params = ("colorspace", "alpha_mode", "operation", "filter", "red_from",
                       "green_from", "blue_from", "alpha_from", "missing", "mode",
                       "out_red", "out_green", "out_blue", "out_alpha", "file_type", "bit_depth")
        nodes = {param: node_type for node_type, spec in SPECS.items() for param in spec["params"]}
        for param in enum_params:
            self.assertEqual(resolve_kind(nodes[param], param, ""), "enum")


if __name__ == "__main__":
    unittest.main()


class Knobs3DTests(unittest.TestCase):
    """3D panels: vectors are one row of typed fields, and sliders are kept for bounded scalars."""

    def test_transforms_are_xyz_rows_not_nine_sliders(self):
        from nodebased.knobs import knob_layout
        for kind in ("Card3D", "Cube3D", "Sphere3D", "ReadGeo3D", "Scene3D"):
            rows = {group.label: group for group in knob_layout(kind)}
            for label, params in (("Translate", ("tx", "ty", "tz")), ("Rotate", ("rx", "ry", "rz")),
                                  ("Scale", ("sx", "sy", "sz"))):
                self.assertEqual((rows[label].kind, rows[label].params), ("xyz", params), kind)
        for kind in ("Light3D", "Camera3D"):
            rows = {group.label: group for group in knob_layout(kind)}
            self.assertEqual(rows["Translate"].kind, "xyz")
            self.assertEqual(rows["Look at"].params, ("target_x", "target_y", "target_z"))

    def test_sliders_on_3d_nodes_are_only_bounded_scalars(self):
        from nodebased.core import SPECS
        from nodebased.knobs import knob_layout
        allowed = {"spec_amount", "intensity", "roll", "fov", "ambient", "splat_relight", "splat_shadow_catch", "splat_specular",
                   "splat_delight_smoothness"}
        for kind in (k for k in SPECS if k.endswith("3D")):
            sliders = {group.params[0] for group in knob_layout(kind) if group.kind == "float_slider"}
            self.assertLessEqual(sliders, allowed, kind)

    def test_every_3d_param_is_laid_out_exactly_once(self):
        from nodebased.core import SPECS
        from nodebased.knobs import knob_layout
        for kind in (k for k in SPECS if k.endswith("3D")):
            laid_out = [param for group in knob_layout(kind) for param in group.params]
            self.assertEqual(sorted(laid_out), sorted(SPECS[kind]["params"]), kind)


class Knob3DWidgetTests(unittest.TestCase):
    def test_sphere_panel_has_typed_xyz_fields_that_commit(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSlider
        from nodebased.app import Window
        app = QApplication.instance() or QApplication([])
        window = Window()
        window.show()
        try:
            for _ in range(2000):
                if window.frame is not None:
                    break
                QTest.qWait(10)
            window.command({"op": "create", "id": "sph", "type": "Sphere3D", "pos": [3000, 3000], "params": {}},
                           render=False)
            window.graph.items_by_id["sph"].setSelected(True)
            app.processEvents()
            fields = {f.objectName(): f for f in window.findChildren(QDoubleSpinBox) if f.objectName()}
            for name in ("tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz", "sphere_radius"):
                self.assertIn(f"{name}-field", fields)
            # Specular is the only slider left on the panel.
            self.assertEqual(len([s for s in window.findChildren(QSlider) if s.isVisible()]), 1)
            fields["ty-field"].setValue(2.5)
            fields["ty-field"].editingFinished.emit()
            app.processEvents()
            self.assertEqual(window.dispatcher.document["nodes"]["sph"]["params"]["ty"], 2.5)
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()
