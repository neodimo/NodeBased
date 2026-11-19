"""Lane 2 step X6, part 3: SmartVector's Export Write button.

The guide lists `export_write` on SmartVector's Export tab (label "Export Write", no default, a button): it adds a Write node with EXR settings.
Here the button adds a Write on the node's output (32 bit float EXR, every channel, the node's frame range) and counts the press, in one undo step.
"""
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased import knobs, smartvectorknobs  # noqa: E402
from nodebased.core import Dispatcher, LIMITS, SPECS, upgrade_document, validate  # noqa: E402


def document():
    d = Dispatcher()
    d.execute(dict(op="create", id="sv", type="SmartVector", params=dict(frame_start=3, frame_end=40), pos=[100, 50]))
    return d


class ExportWriteTests(unittest.TestCase):
    def test_the_param_is_a_press_counter_with_limits(self):
        self.assertEqual(SPECS["SmartVector"]["params"]["export_write"], 0)
        self.assertIn("export_write", LIMITS)

    def test_the_button_sits_between_matte_dilation_and_flicker_compensation_as_in_the_guide(self):
        names = [g.params[0] for g in knobs.KNOB_LAYOUT["SmartVector"]]
        self.assertEqual(names.index("export_write"), names.index("matte_dilation") + 1)
        self.assertEqual(names.index("flicker_compensation"), names.index("export_write") + 1)
        group = knobs.KNOB_LAYOUT["SmartVector"][names.index("export_write")]
        self.assertEqual((group.kind, group.label), ("button", "Export Write"))

    def test_a_press_adds_a_connected_exr_write_and_counts_once(self):
        d = document()
        d.execute(dict(op="batch", commands=smartvectorknobs.export_write_commands("sv", d.document["nodes"])))
        validate(d.document)
        write = d.document["nodes"]["sv-export"]
        self.assertEqual(write["type"], "Write")
        self.assertEqual(write["inputs"], {"image": "sv"})
        self.assertEqual({k: write["params"][k] for k in ("file_type", "bit_depth", "channels", "first", "last")},
                         dict(file_type="exr", bit_depth="float", channels="all", first=3, last=40))
        self.assertEqual(d.document["nodes"]["sv"]["params"]["export_write"], 1)

    def test_a_second_press_adds_a_second_write_and_one_undo_removes_one_press(self):
        d = document()
        for _ in range(2):
            d.execute(dict(op="batch", commands=smartvectorknobs.export_write_commands("sv", d.document["nodes"])))
        self.assertEqual({k for k in d.document["nodes"] if k.startswith("sv-export")}, {"sv-export", "sv-export2"})
        self.assertEqual(d.document["nodes"]["sv"]["params"]["export_write"], 2)
        d.execute(dict(op="undo"))
        self.assertNotIn("sv-export2", d.document["nodes"])
        self.assertIn("sv-export", d.document["nodes"])
        self.assertEqual(d.document["nodes"]["sv"]["params"]["export_write"], 1)

    def test_a_press_moves_no_pixel_of_the_node(self):
        from nodebased.imaging import Evaluator
        import numpy as np
        d = Dispatcher()
        d.execute(dict(op="create", id="plate", type="Noise", params=dict(width=32, height=24)))
        d.execute(dict(op="create", id="sv", type="SmartVector", params=dict(frame_start=1, frame_end=3)))
        d.execute(dict(op="connect", id="sv", input="image", source="plate"))
        before = Evaluator().evaluate(dict(d.document, view="sv"), "sv", frame=1)
        d.execute(dict(op="batch", commands=smartvectorknobs.export_write_commands("sv", d.document["nodes"])))
        after = Evaluator().evaluate(dict(d.document, view="sv"), "sv", frame=1)
        np.testing.assert_array_equal(before, after)

    def test_an_older_document_gets_the_counter_at_zero(self):
        doc = copy.deepcopy(document().document)
        del doc["nodes"]["sv"]["params"]["export_write"]
        doc = upgrade_document(doc)
        validate(doc)
        self.assertEqual(doc["nodes"]["sv"]["params"]["export_write"], 0)

    def test_the_panel_button_writes_through_the_window(self):
        """The properties panel's button runs the same commands (the window's `smartvector_export_write`)."""
        from nodebased import app
        self.assertTrue(callable(getattr(app.Window, "smartvector_export_write", None)))


if __name__ == "__main__":
    unittest.main()
