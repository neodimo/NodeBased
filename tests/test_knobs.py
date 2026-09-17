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
