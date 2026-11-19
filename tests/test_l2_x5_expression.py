"""Lane 2 step X5, part 4: Expression's four channel rows are channel sets in the Nuke 17.0 Reference Guide (Color nodes, Expression: `channel0` to
`channel3`, "channels"), so the panel gives each a channel menu with a check box per channel, as it does Grade's `channels`. What the expressions
compute is unchanged: the same words, read the same way."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from nodebased import channelpicker, knobs, nukeexpr  # noqa: E402
from nodebased.core import SPECS, upgrade_document, validate  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402

PARAMS = [f"expr_channels_{c}" for c in "rgba"]


class WidgetTypeTests(unittest.TestCase):
    def test_the_guide_types_the_four_knobs_as_channel_sets(self):
        rows = {r["name"]: r for r in json.loads((ROOT / "docs" / "parity" / "nuke_nodes.json").read_text())["nodes"]["Expression"]["knobs"]}
        self.assertEqual([rows[f"channel{i}"]["type"] for i in range(4)], ["channels"] * 4)

    def test_the_panel_lays_each_out_once_as_a_menu(self):
        for name in PARAMS:
            groups = [g for g in knobs.KNOB_LAYOUT["Expression"] if name in g.params]
            self.assertEqual(len(groups), 1, name)
            self.assertEqual(groups[0].kind, "enum", name)
            self.assertEqual(groups[0].label, "Channels")

    def test_each_is_a_channel_set_picker_in_the_window(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from nodebased.app import Window
        window = Window()
        try:
            window.dispatcher.execute({"op": "create", "id": "ex", "type": "Expression", "pos": [0, 0]})
            panel = window.build_node_panel("ex")
            for name in PARAMS:
                self.assertIsNotNone(panel.findChild(channelpicker.ChannelSetPicker, f"{name}-channel-set"), name)
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()

    def test_a_pick_is_stored_and_read_the_way_it_was(self):
        for name, word in ((PARAMS[0], "rgb"), (PARAMS[1], "rgba.red rgba.blue"), (PARAMS[3], "alpha")):
            self.assertIn(name, channelpicker.CHANNEL_SET_PARAMS)
        self.assertEqual(nukeexpr.channel_set("rgba.red rgba.blue", "red"), [0, 2])
        self.assertEqual(nukeexpr.channel_set("rgb", "red"), [0, 1, 2])
        self.assertEqual(channelpicker.join_set("rgba", ("red", "blue")), "rgba.red rgba.blue")


class UnchangedMathTests(unittest.TestCase):
    def kernel(self, **params):
        image = np.random.default_rng(5).random((5, 6, 4)).astype(np.float32)
        base = dict(SPECS["Expression"]["params"], expr_r="r * 2", expr_g="g + 0.1", expr_b="1 - b", expr_a="a")
        return Evaluator._kernel("Expression", dict(base, **params), [image])

    def test_the_defaults_and_stored_words_still_route_the_rows(self):
        out = self.kernel()
        image = np.random.default_rng(5).random((5, 6, 4)).astype(np.float32)
        np.testing.assert_allclose(out[..., 0], image[..., 0] * 2, atol=1e-6)
        np.testing.assert_allclose(out[..., 1], image[..., 1] + 0.1, atol=1e-6)
        # a later row overrides an earlier one, so the two rows after the first are switched off to see where the first one wrote
        routed = self.kernel(expr_channels_r="rgba.green rgba.blue", expr_channels_g="none", expr_channels_b="none")
        np.testing.assert_allclose(routed[..., 1], image[..., 0] * 2, atol=1e-6)
        np.testing.assert_allclose(routed[..., 2], image[..., 0] * 2, atol=1e-6)

    def test_a_saved_document_with_these_words_validates(self):
        from tests.test_2d_parity_group_c4 import Graph
        g = Graph()
        g.add("plate", "Constant", dict(width=4, height=4, red=0.5, green=0.5, blue=0.5, alpha=1.0))
        g.add("node", "Expression", dict(expr_channels_r="rgb", expr_channels_g="rgba.red rgba.blue", expr_channels_a="alpha"), image="plate")
        validate(upgrade_document(copy.deepcopy(g.doc)))


if __name__ == "__main__":
    unittest.main()
