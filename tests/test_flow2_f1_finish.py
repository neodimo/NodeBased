import unittest

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator


class MotionBlurFinishTests(unittest.TestCase):
    def test_temporal_motion_blurs_accept_more_than_four_samples(self):
        d = Dispatcher()
        for kind in ("MotionBlur", "MotionBlur2D", "MotionBlur3D"):
            d.execute({"op": "create", "id": kind, "type": kind, "params": {"samples": 24}})
            self.assertEqual(d.document["nodes"][kind]["params"]["samples"], 24)

    def test_motionblur2d_uses_requested_high_sample_count(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Rectangle",
                  "params": {"width": 40, "height": 8, "box_x": 8, "box_y": 3,
                             "box_width": 1, "box_height": 1, "softness": 0}})
        d.execute({"op": "create", "id": "move", "type": "Transform"})
        d.execute({"op": "connect", "id": "move", "input": "image", "source": "src"})
        for frame, x in ((1, 0.0), (5, 16.0)):
            d.execute({"op": "set_key", "id": "move", "param": "translate_x",
                       "frame": frame, "value": x})
        d.execute({"op": "create", "id": "blur", "type": "MotionBlur2D",
                  "params": {"shutter": 4.0, "samples": 4}})
        d.execute({"op": "connect", "id": "blur", "input": "image", "source": "move"})
        d.execute({"op": "time", "first": 1, "last": 5, "current": 3})
        ev = Evaluator()
        low = ev.evaluate_raster(d.document, target="blur", frame=3).pixels
        d.execute({"op": "set", "id": "blur", "param": "samples", "value": 24})
        high = ev.evaluate_raster(d.document, target="blur", frame=3).pixels
        self.assertFalse(np.array_equal(low, high))
        sharp = ev.evaluate_raster(d.document, target="move", frame=3).pixels
        self.assertAlmostEqual(float(low[..., 0].sum()), float(sharp[..., 0].sum()), places=4)
        self.assertAlmostEqual(float(high[..., 0].sum()), float(sharp[..., 0].sum()), places=4)

    def test_vectorblur_exposes_automatic_and_fixed_samples(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker",
                  "params": {"width": 20, "height": 8, "size": 2}})
        d.execute({"op": "create", "id": "vectors", "type": "Constant",
                  "params": {"width": 20, "height": 8, "red": 6, "green": 0, "blue": 0}})
        d.execute({"op": "create", "id": "blur", "type": "VectorBlur", "params": {"samples": 0}})
        d.execute({"op": "connect", "id": "blur", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "blur", "input": "uv", "source": "vectors"})
        ev = Evaluator()
        auto = ev.evaluate_raster(d.document, target="blur", frame=1).pixels
        d.execute({"op": "set", "id": "blur", "param": "samples", "value": 3})
        fixed = ev.evaluate_raster(d.document, target="blur", frame=1).pixels
        self.assertFalse(np.array_equal(auto, fixed))
        d.execute({"op": "set", "id": "blur", "param": "samples", "value": 64})

    def test_vectorblur_destination_sampling_reads_vectors_at_destination(self):
        from nodebased.imaging import Evaluator
        src = np.zeros((3, 8, 4), np.float32)
        src[1, 4] = 1.0
        field = np.full((3, 8), 4.0, np.float32)
        field[:, 1] = 1.0
        params = {"vector_scale": 1.0, "max_length": 100.0, "vector_offset": 0.0,
                  "vector_method": "forward", "vector_sampling": "source", "vector_alpha": "none",
                  "samples": 2}
        source = Evaluator._vector_blur(src, field, np.zeros_like(field), params)
        params["vector_sampling"] = "destination"
        destination = Evaluator._vector_blur(src, field, np.zeros_like(field), params)
        self.assertEqual(float(source[1, 5, 0]), 0.0)
        self.assertGreater(float(destination[1, 5, 0]), 0.45)

    def test_motionblur_consumes_vector_to_motion_layer(self):
        from nodebased.raster import Raster
        from nodebased.tiers import Region
        box = Region(0, 0, 8, 3)
        pixels = np.zeros((3, 8, 4), np.float32)
        rgba = np.zeros_like(pixels)
        rgba[..., 0] = 2.5
        rgba[..., 3] = 1.0
        field = Raster.of(rgba, box)
        source = Raster(pixels, box, box, {"vector.forward": field})
        actual = Evaluator._motion_flow(source, pixels, pixels, box, {"vector_layer": "vector.forward"})
        np.testing.assert_array_equal(actual[..., 0], 2.5)
        np.testing.assert_array_equal(actual[..., 1], 0.0)
        params = {"vector_scale": 1.0, "max_length": 0.0, "vector_offset": 0.0,
                  "vector_method": "forward", "vector_sampling": "source", "vector_alpha": "none",
                  "samples": 5}
        from_vector = Evaluator._vector_blur(pixels, actual[..., 0], actual[..., 1], params)
        analytic = Evaluator._vector_blur(pixels, np.full((3, 8), 2.5, np.float32),
                                           np.zeros((3, 8), np.float32), params)
        np.testing.assert_array_equal(from_vector, analytic)

    def test_motionblur3d_uses_camera_and_depth_for_still_beauty(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "beauty", "type": "Checker",
                  "params": {"width": 64, "height": 32, "size": 4}})
        d.execute({"op": "create", "id": "depth", "type": "Constant",
                  "params": {"width": 64, "height": 32, "red": 5, "green": 5, "blue": 5}})
        d.execute({"op": "create", "id": "camera", "type": "Camera3D", "params": {}})
        for frame, x in ((1, -1.0), (5, 1.0)):
            d.execute({"op": "set_key", "id": "camera", "param": "tx", "frame": frame, "value": x})
        d.execute({"op": "create", "id": "blur", "type": "MotionBlur3D",
                  "params": {"shutter": 4.0, "samples": 9}})
        for slot, source in (("image", "beauty"), ("depth", "depth"), ("camera", "camera")):
            d.execute({"op": "connect", "id": "blur", "input": slot, "source": source})
        d.execute({"op": "time", "first": 1, "last": 5, "current": 3})
        ev = Evaluator()
        beauty = ev.evaluate_raster(d.document, target="beauty", frame=3)
        sharp = beauty.pixels
        blurred = ev.evaluate_raster(d.document, target="blur", frame=3).pixels
        self.assertGreater(float(np.max(np.abs(blurred - sharp))), 0.02)
        self.assertTrue(np.all(np.isfinite(blurred)))
        current = ev.evaluate_raster(d.document, target="camera", frame=3, typed=True)
        start = ev.evaluate_raster(d.document, target="camera", frame=1, typed=True)
        end = ev.evaluate_raster(d.document, target="camera", frame=5, typed=True)
        depth = np.full((32, 64), 5.0, np.float32)
        flow = ev._camera_depth_flow(depth, current, start, end, beauty.display, beauty.data)
        self.assertGreater(float(np.max(np.abs(flow[..., 0]))), 1.0)


if __name__ == "__main__":
    unittest.main()
