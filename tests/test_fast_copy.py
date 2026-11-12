import copy
import unittest

import numpy as np

from nodebased.core import Dispatcher, empty_document, fast_copy


class FastCopyTests(unittest.TestCase):
    def test_equal_and_independent(self):
        data = {"a": [1, 2.5, "x", None, True, {"b": [3, (4, 5)]}], "c": {"d": []}, "n": np.float64(1.5)}
        clone = fast_copy(data)
        self.assertEqual(clone, copy.deepcopy(data))
        self.assertIsNot(clone["a"], data["a"])
        self.assertIsNot(clone["a"][5]["b"], data["a"][5]["b"])
        self.assertIsNot(clone["c"], data["c"])
        clone["a"][5]["b"].append(9)
        clone["c"]["d"].append(1)
        self.assertEqual(data["a"][5]["b"], [3, (4, 5)])
        self.assertEqual(data["c"]["d"], [])
        self.assertIsInstance(clone["a"][5]["b"][1], tuple)

    def test_unknown_objects_fall_back_to_deepcopy(self):
        class Box:
            def __init__(self):
                self.items = [1]
        box = Box()
        clone = fast_copy({"box": box})["box"]
        self.assertIsNot(clone, box)
        self.assertIsNot(clone.items, box.items)

    def test_adopt_leaves_the_live_document_alone(self):
        live = Dispatcher(empty_document())
        live.execute({"op": "create", "id": "g", "type": "Constant"})
        before = copy.deepcopy(live.document)
        preview = Dispatcher.adopt(live.document)
        preview.execute({"op": "set", "id": "g", "param": "width", "value": 640})
        self.assertEqual(live.document, before)
        self.assertEqual(preview.document["nodes"]["g"]["params"]["width"], 640)
        self.assertIsNot(preview.document["nodes"]["g"], live.document["nodes"]["g"])

    def test_set_shapes_stores_its_own_copy(self):
        dispatcher = Dispatcher(empty_document())
        dispatcher.execute({"op": "create", "id": "r", "type": "Roto"})
        shapes = [{"name": "s", "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True,
                   "points": [dict(x=x, y=y, in_x=0.0, in_y=0.0, out_x=0.0, out_y=0.0)
                              for x, y in ((0.0, 0.0), (10.0, 0.0), (5.0, 8.0))]}]
        dispatcher.execute({"op": "set_shapes", "id": "r", "shapes": shapes})
        stored = dispatcher.document["node_data"]["r"]["shapes"]
        shapes[0]["points"][0]["x"] = 99.0
        self.assertEqual(stored[0]["points"][0]["x"], 0.0)


if __name__ == "__main__":
    unittest.main()


class KnobPreviewTests(unittest.TestCase):
    """A properties slider drag previews the knob's edit on a copy: the live document and the undo history stay as they were, and the
    picture request carries exactly the document the committed edit produces."""

    def test_knob_preview_is_the_committed_edit_on_its_own_copy(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from nodebased.app import Window
        from tests.test_roto_ui import roto_document, triangle
        QApplication.instance() or QApplication([])
        QSettings("NodeBased", "NodeBased").clear()
        window = Window(roto_document([triangle()]))
        try:
            window.timer.stop()
            window.timer.start = lambda *a, **k: None
            seen = []
            real = window.request_preview
            window.request_preview = lambda *a, **k: seen.append((a, k))
            live = window.dispatcher.document
            before = copy.deepcopy(live)
            undo_before = len(window.dispatcher.undo_stack)
            window.preview_param_drag("r", "pt1_x", 12.5)
            (args, kwargs), = seen
            self.assertIs(window.dispatcher.document, live)
            self.assertEqual(live, before)
            self.assertEqual(len(window.dispatcher.undo_stack), undo_before)
            shown = kwargs["document_override"]
            self.assertIsNot(shown, live)
            self.assertTrue(kwargs["document_owned"])
            self.assertEqual(shown["nodes"]["r"]["params"]["pt1_x"], 12.5)
            committed = Dispatcher(copy.deepcopy(before))
            committed.execute({"op": "set", "id": "r", "param": "pt1_x", "value": 12.5})
            self.assertEqual(shown, committed.document)
            seen.clear()
            window.preview_param_drag("r", "pt1_x", before["nodes"]["r"]["params"]["pt1_x"])     # an edit that changes nothing
            (args, kwargs), = seen
            self.assertIsNot(kwargs["document_override"], live)
            self.assertEqual(kwargs["document_override"], before)
        finally:
            window.request_preview = real
            window.saved_document = window.dispatcher.document
            window.close()
