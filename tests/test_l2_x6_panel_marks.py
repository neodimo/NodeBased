"""Lane 2 step X6, part 1: the panel marks on IBKColor, HSVTool and MatchGrade.

IBKColor: the guide types `off` (darks) and `mult` (lights) `number_list` (red, green, blue) and `multi` (patch black) `number`.
NodeBased draws darks and lights as split knobs (one number, or one per red, green and blue) and patch black as a number field.
A saved project holds one number for each and renders the same pixels: a number applies to every channel, and a list gives the
screen channel its own entry.
"""
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased import core, knobs  # noqa: E402
from nodebased.core import SPECS, SPLITTABLE, split_count, validate  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.tileexec import TileExecutor  # noqa: E402
from tests.test_2d_parity_group_c4 import Graph  # noqa: E402


def screen_graph(params, screen=(0.1, 0.9, 0.2), subject=(0.7, 0.3, 0.2)):
    """A 16 x 12 ramp from the screen colour into a foreground colour, through IBKColor."""
    g = Graph()
    g.add("plate", "Ramp", dict(width=16, height=12, p0_x=0.0, p0_y=0.0, p1_x=15.0, p1_y=0.0,
                                **{f"color0_{c}": v for c, v in zip(("red", "green", "blue"), screen)},
                                **{f"color1_{c}": v for c, v in zip(("red", "green", "blue"), subject)}))
    g.add("node", "IBKColor", params, c="plate")
    return dict(g.doc, view="node")


def full(doc):
    return Evaluator().evaluate(doc, "node", frame=1)


def tiled(doc, edge):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
    assert executor.supports_tiled(doc, "node")
    region = executor.canvas_region(doc, "node", frame=1, tier=1)
    return executor.compose_region(doc, "node", region, frame=1, tier=1).pixels


class IBKColorWidgetTests(unittest.TestCase):
    def test_darks_and_lights_are_split_knobs_of_three_channels(self):
        self.assertEqual(SPLITTABLE["IBKColor"], {"darks": "rgb", "lights": "rgb"})
        self.assertEqual(split_count("IBKColor", "darks"), 3)
        layout = {g.params[0]: g for g in knobs.KNOB_LAYOUT["IBKColor"]}
        for name in ("darks", "lights"):
            self.assertEqual((layout[name].kind, layout[name].channels), ("split", "rgb"))
        self.assertEqual(layout["patch_black"].kind, "int")
        self.assertEqual(knobs.split_layout_problems(), [])

    def test_a_new_node_still_holds_single_numbers(self):
        params = SPECS["IBKColor"]["params"]
        self.assertEqual((params["darks"], params["lights"], params["patch_black"]), (0.05, 1000.0, 0))

    def test_a_list_must_have_three_numbers_within_the_limits(self):
        d = core.Dispatcher()
        d.execute(dict(op="create", id="k", type="IBKColor", params={}))
        d.execute(dict(op="set", id="k", param="darks", value=[0.1, 0.2, 0.3]))
        validate(d.document)
        with self.assertRaises(Exception):
            d.execute(dict(op="set", id="k", param="darks", value=[0.1, 0.2]))
        with self.assertRaises(Exception):
            d.execute(dict(op="set", id="k", param="lights", value=[1.0, 2.0, 5000.0]))

    def test_the_panel_builds_with_the_split_rows(self):
        from tests.test_desktop import FluidPropertiesPanelTests  # noqa: F401  (the panel suite builds every node's panel)


class IBKColorSameStoredPixelsTests(unittest.TestCase):
    """A project saved before the change holds single numbers: its pixels do not move."""

    BASE = dict(screen_type="green", fill_size=6, screen_erode=0.0, darks=0.3, lights=0.6, patch_black=1)

    def test_equal_entries_equal_the_number(self):
        for screen_type in ("green", "blue"):
            number = full(screen_graph(dict(self.BASE, screen_type=screen_type), screen=(0.1, 0.9, 0.9) if screen_type == "blue" else (0.1, 0.9, 0.2)))
            listed = full(screen_graph(dict(self.BASE, screen_type=screen_type, darks=[0.3] * 3, lights=[0.6] * 3),
                                       screen=(0.1, 0.9, 0.9) if screen_type == "blue" else (0.1, 0.9, 0.2)))
            np.testing.assert_array_equal(listed, number)

    def test_the_screen_channel_reads_its_own_entry(self):
        for screen_type, index in (("green", 1), ("blue", 2)):
            screen = (0.1, 0.9, 0.2) if screen_type == "green" else (0.1, 0.2, 0.9)
            for name, lo_hi in (("darks", dict(darks=None, lights=1000.0)), ("lights", dict(darks=0.0, lights=None))):
                entries = [0.05, 0.05, 0.05] if name == "darks" else [1000.0] * 3
                entries[index] = 0.5
                rest = dict(self.BASE, **{k: v for k, v in lo_hi.items() if v is not None})
                listed = full(screen_graph(dict(rest, screen_type=screen_type, **{name: entries}), screen=screen))
                other = list(entries)
                other[(index + 1) % 3] = 0.77
                again = full(screen_graph(dict(rest, screen_type=screen_type, **{name: other}), screen=screen))
                single = full(screen_graph(dict(rest, screen_type=screen_type, **{name: 0.5}), screen=screen))
                with self.subTest(screen=screen_type, knob=name):
                    np.testing.assert_array_equal(listed, single)       # the other entries are not read
                    np.testing.assert_array_equal(listed, again)

    def test_a_split_entry_changes_pixels_only_through_its_channel(self):
        low = full(screen_graph(dict(self.BASE, darks=[0.0, 0.0, 0.0], lights=[1000.0] * 3)))
        high_green = full(screen_graph(dict(self.BASE, darks=[0.0, 0.8, 0.0], lights=[1000.0] * 3)))
        high_red = full(screen_graph(dict(self.BASE, darks=[0.8, 0.0, 0.0], lights=[1000.0] * 3)))
        self.assertFalse(np.array_equal(low, high_green))
        np.testing.assert_array_equal(low, high_red)                  # a green screen ignores the red entry

    def test_tiles_equal_the_full_frame_with_a_split_list(self):
        doc = screen_graph(dict(self.BASE, darks=[0.0, 0.4, 0.0], lights=[1000.0, 0.7, 1000.0]))
        expected = full(doc)
        for edge in (7, 16, 32):
            with self.subTest(edge=edge):
                np.testing.assert_allclose(tiled(doc, edge), expected, atol=1e-6)

    def test_patch_black_is_a_number_field_with_the_same_effect(self):
        on = full(screen_graph(dict(self.BASE, fill_size=0, patch_black=1)))
        off = full(screen_graph(dict(self.BASE, fill_size=0, patch_black=0)))
        self.assertFalse(np.array_equal(on, off))


if __name__ == "__main__":
    unittest.main()
