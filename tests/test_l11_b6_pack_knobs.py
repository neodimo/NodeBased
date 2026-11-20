"""Lane 11 step B6: every setting a case makes reaches the Nuke script, so a render shows the case and never Nuke's defaults.

These run where Nuke is not installed: they read the scripts' text. The Shuffle2 and ShuffleCopy wiring below was read in Nuke 17.0v6
on 11/20 (two Constants into each node, every route written to a 32-bit EXR and read back).
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_nuke_reference_pack as pack  # noqa: E402
import nuke_fidelity_cases as cases  # noqa: E402
from nuke17_knob_table import load_table  # noqa: E402


def _case(case_id):
    return next(c for c in cases.CASES if c.id == case_id)


class EverySettingReachesTheScriptTests(unittest.TestCase):
    """The per-case check: the knob values a case lists are the values its script sets, and a case that sets NodeBased parameters
    with no Nuke knob is listed as not renderable instead of rendering Nuke's defaults."""

    @classmethod
    def setUpClass(cls):
        cls.table = load_table()
        cls.renderable = [c for c in cases.CASES if not c.id.endswith(pack.SKIPPED_SUFFIXES) and not pack.unrenderable(c, cls.table)]

    def test_every_listed_knob_is_set_in_the_script_with_the_case_value(self):
        for case in self.renderable:
            script = pack.py_script(case, self.table)
            cls = pack.nuke17_class(case)
            for knob, value in case.nuke_knobs:
                statement = pack.knob_statement(cls, pack.NUKE17_KNOB.get((case.nuke_class, knob), knob), value, self.table,
                                                pack.case_files(case))
                with self.subTest(case=case.id, knob=knob):
                    self.assertIn(statement, script)

    def test_no_renderable_case_leaves_a_setting_out_of_the_script(self):
        for case in self.renderable:
            with self.subTest(case=case.id):
                self.assertEqual(pack.unwritten_settings(case), [])
                if case.nuke_class in pack.ROUTING_WRITERS:
                    script = pack.py_script(case, self.table)
                    for line in pack.ROUTING_WRITERS[case.nuke_class](case):
                        self.assertIn(line, script)

    def test_a_case_with_settings_and_no_knobs_is_not_renderable(self):
        for case_id in ("huekeyer_amount_green_to_black", "crosstalk_bleed_red_into_green", "gridwarp_interior_key_moves_the_pixel_under_it",
                        "addmix_moved_curves", "splinewarp_curve_moves_the_pixels_on_it"):
            with self.subTest(case=case_id):
                reason = pack.unrenderable(_case(case_id), self.table)
                self.assertIn("lists no Nuke knob", reason)

    def test_legacy_flags_and_pack_keys_are_not_settings(self):
        case = _case("shuffle_default_passes_through")
        self.assertEqual(pack.unwritten_settings(case), [])


class ShuffleRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = load_table()

    def test_shuffle_is_nuke_17s_shuffle2(self):
        self.assertEqual(pack.nuke17_class(_case("shuffle_swap_red_and_green")), "Shuffle2")
        self.assertIn("mappings", self.table["Shuffle2"]["knobs"])

    def test_one_input_shuffle_maps_each_channel_from_input_zero(self):
        lines = pack.shuffle2_lines(_case("shuffle_swap_red_and_green"))
        self.assertEqual(lines[-1], "node['mappings'].setValue([(0, 'rgba.green', 'rgba.red'), (0, 'rgba.red', 'rgba.green'), "
                                    "(0, 'rgba.blue', 'rgba.blue'), (0, 'rgba.alpha', 'rgba.alpha')])")
        self.assertNotIn("node['fromInput2'].setValue(1)  # A: input 1", lines)

    def test_black_and_white_are_constants(self):
        line = pack.shuffle2_lines(_case("shuffle_black_and_white"))[-1]
        self.assertIn("(-1, 'black', 'rgba.red')", line)
        self.assertIn("(-1, 'white', 'rgba.green')", line)

    def test_second_input_routes_read_input_one(self):
        lines = pack.shuffle2_lines(_case("shuffle_from_the_second_input"))
        self.assertIn("node['fromInput2'].setValues(['B', 'A'])", lines)   # a script's menu lists only B until it is set
        self.assertIn("(1, 'rgba.red', 'rgba.red')", lines[-1])
        self.assertIn("(0, 'rgba.green', 'rgba.green')", lines[-1])

    def test_shufflecopy_input_zero_is_nukes_layer_two(self):
        lines = pack.shufflecopy_lines(_case("shufflecopy_mixed_routing"))     # in2.b in1.a in2.r in1.g
        self.assertEqual(lines[2:], ["node['red'].setValue('blue')", "node['green'].setValue('alpha2')",
                                     "node['blue'].setValue('red')", "node['alpha'].setValue('green2')"])
        lines = pack.shufflecopy_lines(_case("shufflecopy_constant_black_and_white"))
        self.assertIn("node['green'].setValue('black')", lines)
        self.assertIn("node['blue'].setValue('white')", lines)

    def test_shuffle_scripts_carry_the_routing(self):
        script = pack.py_script(_case("shufflecopy_swap_inputs"), self.table)
        self.assertIn("node = nuke.nodes.ShuffleCopy()", script)
        self.assertIn("node['alpha'].setValue('alpha')", script)
        script = pack.py_script(_case("shuffle_rotate_channels"), self.table)
        self.assertIn("node = nuke.nodes.Shuffle2()", script)


if __name__ == "__main__":
    unittest.main()
