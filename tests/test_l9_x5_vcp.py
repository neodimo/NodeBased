"""2D behaviour, lane 9 step X5: VectorCornerPin Frame Distance (Nuke 17.0 Reference Guide, VectorCornerPin: frameDistance, "Frame Distance",
SmartVector tab, default 1 frame). Held on the node as VectorDistort's is: stored, listed in the guide's place, and moving no pixel, because
NodeBased's SmartVector keeps one accumulated set of vectors per frame."""
import unittest

import numpy as np

from nodebased.core import CHOICES, SPECS
from tests.test_2d_behaviour_filter_g import FRAME, labels, pin_params, pinned
from tests.test_2d_behaviour_filter_a import old_document


class VectorCornerPinFrameDistanceTests(unittest.TestCase):
    def test_default_and_choices(self):
        self.assertEqual(SPECS["VectorCornerPin"]["params"]["frame_distance"], "1 frame")
        self.assertEqual(CHOICES["frame_distance"][0], "1 frame")
        self.assertEqual(CHOICES["frame_distance"][-1], "64 frames")

    def test_the_value_is_held_and_moves_no_pixel(self):
        shifted = tuple((x + 3, y + 2) for x, y in FRAME)
        base = pinned(**pin_params(shifted))
        for value in ("2 frames", "16 frames", "64 frames"):
            np.testing.assert_array_equal(pinned(frame_distance=value, **pin_params(shifted)), base)

    def test_it_sits_between_the_from_pins_and_output(self):
        names = labels("VectorCornerPin")
        self.assertEqual(names[names.index("Frame distance") - 1], "From4 xy")
        self.assertEqual(names[names.index("Frame distance") + 1], "Output")

    def test_a_saved_node_loads_with_one_frame(self):
        doc = old_document("VectorCornerPin", {"reference_frame": 1}, drop=["frame_distance"])
        self.assertEqual(doc["nodes"]["n"]["params"]["frame_distance"], "1 frame")


if __name__ == "__main__":
    unittest.main()
