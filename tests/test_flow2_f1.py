import unittest

import numpy as np

from nodebased.core import Dispatcher
from nodebased.flow_nodes import vector_layers_to_motion
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tiers import Region


class OFlowAndVectorToMotionTests(unittest.TestCase):
    def test_oflow_speed_one_is_identity(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker",
                  "params": {"width": 24, "height": 12, "size": 3}})
        d.execute({"op": "create", "id": "retime", "type": "OFlow",
                  "params": {"input_start": 1, "input_end": 5, "output_start": 1, "speed": 1}})
        d.execute({"op": "connect", "id": "retime", "input": "image", "source": "src"})
        d.execute({"op": "time", "first": 1, "last": 5, "current": 1})
        ev = Evaluator()
        source = ev.evaluate_raster(d.document, target="src", frame=3).pixels
        output = ev.evaluate_raster(d.document, target="retime", frame=3).pixels
        np.testing.assert_array_equal(output, source)

    def test_oflow_half_speed_tracks_known_pan_midpoint(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker",
                  "params": {"width": 32, "height": 16, "size": 2}})
        d.execute({"op": "create", "id": "pan", "type": "Transform", "params": {"filter": "nearest"}})
        d.execute({"op": "connect", "id": "pan", "input": "image", "source": "src"})
        for f, x in ((1, 0.0), (5, 8.0)):
            d.execute({"op": "set_key", "id": "pan", "param": "translate_x", "frame": f, "value": x})
        d.execute({"op": "create", "id": "retime", "type": "OFlow",
                  "params": {"input_start": 1, "input_end": 5, "output_start": 1, "speed": .5}})
        d.execute({"op": "connect", "id": "retime", "input": "image", "source": "pan"})
        d.execute({"op": "time", "first": 1, "last": 5, "current": 3})
        ev = Evaluator()
        expected = ev.evaluate_raster(d.document, target="pan", frame=2).pixels
        actual = ev.evaluate_raster(d.document, target="retime", frame=3).pixels
        np.testing.assert_allclose(actual, expected, atol=.25)

    def test_vector_to_motion_drives_same_blur_as_analytic_layer(self):
        h, w = 9, 17
        display = Region(0, 0, w, h)
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[4, 5] = (1, 1, 1, 1)
        field = np.zeros((h, w, 4), np.float32)
        field[..., 0] = 4.0
        field[..., 3] = 1.0
        layers = {"smartvector.forward": Raster.of(field, display),
                  "smartvector.backward": Raster.of(-field, display)}
        converted = vector_layers_to_motion(layers)
        source = Raster.of(rgba, display)
        converted_uv = converted["vector.forward"]
        params = {"uv_layer": "", "u_channel": "R", "v_channel": "G",
                  "vector_scale": 1.0, "vector_offset": 0.0, "vector_method": "forward",
                  "vector_alpha": "none", "max_length": 100.0, "samples": 5, "mix": 1.0}
        got = Evaluator._uv_node("VectorBlur", params, [source, converted_uv, None]).pixels
        analytic = Evaluator._uv_node("VectorBlur", params,
            [source, Raster.of(field, display), None]).pixels
        np.testing.assert_array_equal(got, analytic)
        np.testing.assert_allclose(got[..., 0].sum(), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
