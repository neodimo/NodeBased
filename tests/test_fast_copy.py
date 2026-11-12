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
