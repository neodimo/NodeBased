"""Roto's plate path and first-draw viewer cue."""
import os
import unittest

import numpy as np
from tests.test_roto_ui import triangle
from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Evaluator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class RotoPlateTests(unittest.TestCase):
    def plate_document(self, shapes=None, premultiply_mode="none"):
        document = empty_document()
        dispatcher = Dispatcher(document)
        dispatcher.execute({"op": "create", "id": "plate", "type": "Checker"})
        dispatcher.execute({"op": "set", "id": "plate", "param": "width", "value": 32})
        dispatcher.execute({"op": "set", "id": "plate", "param": "height", "value": 32})
        dispatcher.execute({"op": "create", "id": "r", "type": "Roto"})
        dispatcher.execute({"op": "connect", "id": "r", "input": "bg", "source": "plate"})
        dispatcher.execute({"op": "set", "id": "r", "param": "premultiply_mode", "value": premultiply_mode})
        dispatcher.execute({"op": "view", "id": "r"})
        if shapes:
            dispatcher.execute({"op": "set_shapes", "id": "r", "shapes": shapes})
        return dispatcher.document

    def test_empty_roto_preserves_checker_and_shape_only_changes_alpha_by_default(self):
        evaluator = Evaluator()
        empty = self.plate_document()
        plate = evaluator.evaluate_raster(empty, target="plate").pixels
        result = evaluator.evaluate_raster(empty, target="r").pixels
        np.testing.assert_array_equal(result, plate)

        shape = triangle()
        shape["points"] = [dict(point, x=point["x"] * 0.25, y=point["y"] * 0.25)
                           for point in shape["points"]]
        with_shape = self.plate_document([shape])
        plate = evaluator.evaluate_raster(with_shape, target="plate").pixels
        result = evaluator.evaluate_raster(with_shape, target="r").pixels
        np.testing.assert_array_equal(result[..., :3], plate[..., :3])
        self.assertEqual(float(result[0, 0, 3]), 0.0)
        self.assertGreater(float(result[5, 5, 3]), 0.0)
        self.assertEqual(with_shape["nodes"]["r"]["params"]["premultiply_mode"], "none")

    def test_rgb_premultiply_is_applied_only_when_selected(self):
        shape = triangle()
        shape["points"] = [dict(point, x=point["x"] * 0.25, y=point["y"] * 0.25)
                           for point in shape["points"]]
        document = self.plate_document([shape], premultiply_mode="rgb")
        result = Evaluator().evaluate_raster(document, target="r").pixels
        coverage = result[..., 3:4]
        plate = Evaluator().evaluate_raster(document, target="plate").pixels
        np.testing.assert_allclose(result[..., :3], plate[..., :3] * coverage, atol=1e-7)

if __name__ == "__main__":
    unittest.main()
