import unittest
from unittest.mock import patch
import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator


class FlowNodesV1Tests(unittest.TestCase):
    def graph(self):
        d = Dispatcher()
        d.execute({"op":"create", "id":"src", "type":"Constant", "params":{"width":24,"height":12,"red":.3,"green":.2,"blue":.1,"alpha":1}})
        for ident, kind, params in (("vectors", "VectorGenerator", {}), ("retime", "Kronos", {"speed":1}),
                                    ("blur", "MotionBlur", {})):
            d.execute({"op":"create", "id":ident, "type":kind, "params":params})
            d.execute({"op":"connect", "id":ident, "input":"image", "source":"src"})
        return d

    def test_vector_generator_emits_pair_layers_and_bypasses(self):
        d = self.graph(); ev = Evaluator()
        from nodebased import opticalflow
        original = opticalflow.flow_pair
        with patch.object(opticalflow, "flow_pair", wraps=original) as solve:
            out = ev.evaluate_raster(d.document, target="vectors", frame=1)
            ev.evaluate_raster(d.document, target="vectors", frame=1)
            self.assertEqual(solve.call_count, 1)
        self.assertIn("vector.forward", out.layers)
        self.assertIn("vector.backward", out.layers)
        self.assertIn("vector.occlusion", out.layers)
        np.testing.assert_array_equal(out.layers["vector.forward"].pixels[..., :2], 0)
        expected = ev.evaluate_raster(d.document, target="src", frame=1)
        d.execute({"op":"disable", "id":"vectors", "value":True})
        np.testing.assert_array_equal(ev.evaluate_raster(d.document, target="vectors", frame=1).pixels, expected.pixels)

    def test_kronos_speed_one_is_input_and_frame_blend_mode_runs(self):
        d = self.graph(); ev = Evaluator()
        a = ev.evaluate_raster(d.document, target="src", frame=1).pixels
        out = ev.evaluate_raster(d.document, target="retime", frame=1).pixels
        np.testing.assert_array_equal(out, a)
        d.execute({"op":"set", "id":"retime", "param":"interpolation", "value":"frame"})
        np.testing.assert_array_equal(ev.evaluate_raster(d.document, target="retime", frame=1).pixels, a)
        d.execute({"op":"disable", "id":"retime", "value":True})
        np.testing.assert_array_equal(ev.evaluate_raster(d.document, target="retime", frame=1).pixels, a)

    def test_motion_blur_static_is_identity_and_bypasses(self):
        d = self.graph(); ev = Evaluator()
        source = ev.evaluate_raster(d.document, target="src", frame=1).pixels
        out = ev.evaluate_raster(d.document, target="blur", frame=1).pixels
        np.testing.assert_array_equal(out, source)
        d.execute({"op":"disable", "id":"blur", "value":True})
        np.testing.assert_array_equal(ev.evaluate_raster(d.document, target="blur", frame=1).pixels, source)
