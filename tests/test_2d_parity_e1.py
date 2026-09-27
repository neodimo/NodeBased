import unittest

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator, curve_tool_metrics


class MotionBlurTests(unittest.TestCase):
    def test_motionblur2d_static_transform_is_identity(self):
        d = Dispatcher()
        d.execute({"op":"create", "id":"src", "type":"Constant", "params":{"width":8,"height":4,"red":.4,"green":.2,"blue":.1,"alpha":1}})
        d.execute({"op":"create", "id":"move", "type":"Transform"})
        d.execute({"op":"connect", "id":"move", "input":"image", "source":"src"})
        d.execute({"op":"create", "id":"blur", "type":"MotionBlur2D", "params":{"samples":4}})
        d.execute({"op":"connect", "id":"blur", "input":"image", "source":"move"})
        expected = Evaluator().evaluate(dict(d.document, view="move"))
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="blur")), expected)
        d.execute({"op":"disable", "id":"blur", "value":True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="blur")), expected)

    def test_motionblur2d_animated_translate_streaks_without_losing_energy(self):
        d = Dispatcher()
        d.execute({"op":"create", "id":"shape", "type":"Rectangle", "params":{"width":40,"height":5,"box_x":10,"box_y":2,"box_width":1,"box_height":1,"red":1,"green":1,"blue":1,"alpha":1,"softness":0}})
        d.execute({"op":"create", "id":"move", "type":"Transform"})
        d.execute({"op":"connect", "id":"move", "input":"image", "source":"shape"})
        d.execute({"op":"set_key", "id":"move", "param":"translate_x", "frame":0, "value":0})
        d.execute({"op":"set_key", "id":"move", "param":"translate_x", "frame":2, "value":20})
        d.execute({"op":"create", "id":"blur", "type":"MotionBlur2D", "params":{"shutter":1,"samples":4}})
        d.execute({"op":"connect", "id":"blur", "input":"image", "source":"move"})
        out = Evaluator().evaluate(dict(d.document, view="blur"))
        before = Evaluator().evaluate(dict(d.document, view="move"), frame=1)
        self.assertAlmostEqual(float(out[...,0].sum()), float(before[...,0].sum()), places=4)
        xs = np.where(out[2,:,0] > 1e-5)[0]
        self.assertGreaterEqual(int(xs[-1]-xs[0]+1), 9)
        self.assertLessEqual(int(xs[-1]-xs[0]+1), 12)

    def test_motionblur3d_still_input_is_identity_and_bypasses(self):
        d = Dispatcher()
        d.execute({"op":"create", "id":"src", "type":"Constant", "params":{"width":5,"height":3,"red":.25,"green":.5,"blue":.75,"alpha":1}})
        d.execute({"op":"create", "id":"blur", "type":"MotionBlur3D", "params":{"samples":4}})
        d.execute({"op":"connect", "id":"blur", "input":"image", "source":"src"})
        expected = Evaluator().evaluate(dict(d.document, view="src"))
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="blur")), expected)
        d.execute({"op":"disable", "id":"blur", "value":True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="blur")), expected)


class CurveToolTests(unittest.TestCase):
    def test_metrics_match_flicker_average_and_planted_alpha_bounds(self):
        image = np.zeros((4, 6, 4), np.float32)
        image[1:3, 2:5] = [0.2, 0.4, 0.6, 1.0]
        m1 = curve_tool_metrics(image, (2, 1, 3, 2))
        image[1:3, 2:5, :3] *= 2
        m2 = curve_tool_metrics(image, (2, 1, 3, 2))
        self.assertAlmostEqual(m1["average_r"], 0.2, places=6)
        self.assertAlmostEqual(m2["average_r"], 0.4, places=6)
        self.assertEqual((m1["crop_x"], m1["crop_y"], m1["crop_width"], m1["crop_height"]), (2.0, 1.0, 3.0, 2.0))
        self.assertEqual((m1["max_x"], m1["max_y"]), (2.0, 1.0))

    def test_curve_tool_is_a_bypassable_analysis_tap(self):
        d = Dispatcher()
        d.execute({"op":"create", "id":"src", "type":"Constant", "params":{"width":3,"height":2,"red":.2,"green":.4,"blue":.6,"alpha":1}})
        d.execute({"op":"create", "id":"curve", "type":"CurveTool"})
        d.execute({"op":"connect", "id":"curve", "input":"image", "source":"src"})
        expected = Evaluator().evaluate(dict(d.document, view="src"))
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="curve")), expected)
        d.execute({"op":"disable", "id":"curve", "value":True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="curve")), expected)


class ContactSheetTests(unittest.TestCase):
    def test_contact_sheet_places_two_inputs_in_expected_cells_and_bypasses(self):
        d = Dispatcher()
        d.execute({"op":"create", "id":"red", "type":"Constant", "params":{"width":2,"height":2,"red":2,"green":0,"blue":0,"alpha":1}})
        d.execute({"op":"create", "id":"blue", "type":"Constant", "params":{"width":2,"height":2,"red":0,"green":0,"blue":1,"alpha":1}})
        d.execute({"op":"create", "id":"sheet", "type":"ContactSheet", "params":{"rows":1,"columns":2,"gap":0,"width":80,"height":40,"labels":"name","fit":"fill"}})
        d.execute({"op":"connect", "id":"sheet", "input":"clip0", "source":"red"})
        d.execute({"op":"connect", "id":"sheet", "input":"clip1", "source":"blue"})
        out = Evaluator().evaluate(dict(d.document, view="sheet"))
        np.testing.assert_allclose(out[1,0], [2,0,0,1], atol=1e-6)
        np.testing.assert_allclose(out[1,79], [0,0,1,1], atol=1e-6)
        np.testing.assert_allclose(out[4,5], [1,1,1,1], atol=1e-6)
        d.execute({"op":"disable", "id":"sheet", "value":True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="sheet")), Evaluator().evaluate(dict(d.document, view="red")))


if __name__ == "__main__":
    unittest.main()
