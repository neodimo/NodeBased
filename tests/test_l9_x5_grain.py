"""2D behaviour, lane 9 step X5: Grain's `presets` dropdown (Nuke 17.0 Reference Guide, Grain: presets, "Choose the film stock you want to match":
Kodak 5248, Kodak 5279, Kodak FX214, Kodak GT5274, Kodak 5217, Kodak 5218 and other).

The guide gives no default and no size, irregularity or intensity values for any stock, so NodeBased lists the stocks in the guide's order after its own
`none`, records the choice, and sets no other knob (nothing is invented). A saved node loads with `none` and draws what it drew.
"""
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import CHOICES, SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import old_document


def grain(**knob_values):
    params = {**SPECS["Grain"]["params"], **knob_values}
    picture = np.full((16, 20, 4), 0.5, np.float32)
    return np.asarray(Evaluator._kernel("Grain", params, [picture, None]), np.float64)


class GrainPresetTests(unittest.TestCase):
    def test_the_list_is_the_guides_after_none(self):
        self.assertEqual(CHOICES["presets"], ["none", "Kodak 5248", "Kodak 5279", "Kodak FX214", "Kodak GT5274", "Kodak 5217", "Kodak 5218", "other"])
        self.assertEqual(SPECS["Grain"]["params"]["presets"], "none")

    def test_the_guides_defaults_for_the_grain_knobs_are_untouched(self):
        params = SPECS["Grain"]["params"]
        self.assertEqual((params["red_size"], params["green_size"], params["blue_size"]), (3.3, 2.9, 2.5))
        self.assertEqual((params["red_intensity"], params["green_intensity"], params["blue_intensity"]), (0.416, 0.46, 0.85))

    def test_choosing_a_stock_changes_no_pixel(self):
        base = grain()
        for stock in CHOICES["presets"][1:]:
            np.testing.assert_array_equal(grain(presets=stock), base)

    def test_it_follows_the_seed_in_the_panel(self):
        names = [g.label for g in knobs.KNOB_LAYOUT["Grain"]]
        self.assertEqual(names[names.index("Seed") + 1], "Presets")

    def test_a_saved_node_loads_with_none(self):
        doc = old_document("Grain", {}, drop=["presets"])
        self.assertEqual(doc["nodes"]["n"]["params"]["presets"], "none")


if __name__ == "__main__":
    unittest.main()
