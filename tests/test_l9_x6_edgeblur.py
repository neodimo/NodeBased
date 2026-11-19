"""2D behaviour, lane 9 step X6: EdgeBlur's output (Nuke 17.0 Reference Guide, EdgeBlur: `output`, default none, "Sets the output channel for any blur applied";
its check box "is the same as setting the channel to none").

The blur is applied to `channels`; `output` names the one channel it is written to (the list Emboss and EdgeDetect's output uses), and none writes it to every
channel the blur was applied to. A document saved before the knob loads with none and draws what it drew.
"""
import copy
import unittest

import numpy as np

from nodebased import knobs
from nodebased.core import CHOICES, SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import differs, evaluator_pixels, old_document, source_graph, tile_pixels

CHANNELS = {"rgba.red": 0, "rgba.green": 1, "rgba.blue": 2, "rgba.alpha": 3}


def hard_matte(h=24, w=30):
    """A picture whose every channel steps at the middle (0.1 to 0.9), with alpha stepping 0 to 1, so the edge band holds all four channels."""
    image = np.zeros((h, w, 4), np.float32)
    right = np.arange(w) >= w // 2
    for c in range(3):
        image[..., c] = np.where(right, 0.9, 0.1)[None, :] * (1.0 - 0.1 * c)
    image[..., 3] = np.where(right, 1.0, 0.0)[None, :]
    return image


def edge_blur(picture, **knob_values):
    params = {**SPECS["EdgeBlur"]["params"], "edgeblur_size": 3.0, **knob_values}
    return np.asarray(Evaluator._edge_blur(picture, params), np.float64)


class OutputKnobTests(unittest.TestCase):
    def test_the_default_is_none_and_the_choices_are_the_channel_list(self):
        self.assertEqual(SPECS["EdgeBlur"]["params"]["edgeblur_output"], "none")
        self.assertEqual(CHOICES["edgeblur_output"], ["none", "rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"])

    def test_none_blurs_every_channel_channels_names(self):
        picture = hard_matte().astype(np.float64)
        out = edge_blur(hard_matte(), edgeblur_output="none")
        for c in range(4):
            self.assertTrue(differs(out[..., c], picture[..., c]), c)

    def test_each_channel_choice_writes_the_blur_to_that_channel_only(self):
        picture = hard_matte().astype(np.float64)
        everywhere = edge_blur(hard_matte(), edgeblur_output="none")
        for token, index in CHANNELS.items():
            with self.subTest(output=token):
                out = edge_blur(hard_matte(), edgeblur_output=token)
                np.testing.assert_allclose(out[..., index], everywhere[..., index], atol=1e-7)     # the same blur lands there
                for other in range(4):
                    if other != index:
                        np.testing.assert_array_equal(out[..., other], picture[..., other])        # and nowhere else

    def test_an_output_outside_channels_writes_nothing(self):
        picture = hard_matte().astype(np.float64)
        out = edge_blur(hard_matte(), channels="rgb", edgeblur_output="rgba.alpha")
        np.testing.assert_array_equal(out, picture)

    def test_the_check_box_off_is_none(self):
        self.assertEqual(SPECS["EdgeBlur"]["params"]["edgeblur_output"], "none")      # the guide's disabled box: the same as none, which the default is

    def test_the_panel_lists_output_after_brightness_in_the_guides_order(self):
        labels = [g.label for g in knobs.knob_layout("EdgeBlur")]
        self.assertEqual(labels[labels.index("Brightness") + 1], "Output")


class RenderTests(unittest.TestCase):
    def graph(self, masked=False, **knob_values):
        return source_graph("EdgeBlur", {"edgeblur_size": 3.0, "edge_matte": "rgba.red", **knob_values}, masked)

    def test_the_node_renders_the_channel_choice(self):
        red = np.asarray(evaluator_pixels(self.graph(edgeblur_output="rgba.red").doc, "node"))
        none = np.asarray(evaluator_pixels(self.graph(edgeblur_output="none").doc, "node"))
        self.assertTrue(differs(none, red))                         # none blurs the other channels as well
        np.testing.assert_allclose(none[..., 0], red[..., 0], atol=1e-6)

    def test_tiles_of_7_16_and_32_equal_the_frame_with_and_without_a_mask(self):
        for output in ("none", "rgba.green"):
            for masked in (False, True):
                g = self.graph(masked, edgeblur_output=output)
                whole = np.asarray(evaluator_pixels(g.doc, "node"))
                for edge in (7, 16, 32):
                    with self.subTest(output=output, masked=masked, edge=edge):
                        np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), whole, atol=1e-6)


class SavedDocumentTests(unittest.TestCase):
    def test_a_saved_node_loads_with_none_and_renders_as_before(self):
        g = self.build()
        before = np.asarray(evaluator_pixels(g, "n"))
        old = copy.deepcopy(g)
        old["nodes"]["n"]["params"].pop("edgeblur_output")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["edgeblur_output"], "none")
        np.testing.assert_array_equal(np.asarray(evaluator_pixels(loaded, "n")), before)

    def build(self):
        return old_document("EdgeBlur", {"edgeblur_size": 3.0, "edge_matte": "rgba.red"}, ())


if __name__ == "__main__":
    unittest.main()
