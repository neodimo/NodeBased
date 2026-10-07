"""Lane 8 Q2: analysis regions follow keyed animation across a measured frame range."""
import unittest

import numpy as np

from nodebased.analysisregion import min_color_result, region_params_at_frame
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator


class AnimatedAnalysisRegionTests(unittest.TestCase):
    def setUp(self):
        self.document = Dispatcher().document
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "source", "type": "Constant",
                            "params": {"width": 5, "height": 1, "red": 0, "green": 0, "blue": 0, "alpha": 1}})
        for frame, value in enumerate((0.0, 0.25, 0.5, 0.75, 1.0), 1):
            dispatcher.execute({"op": "set_key", "id": "source", "param": "red", "frame": frame, "value": value})
        dispatcher.execute({"op": "create", "id": "analysis", "type": "MinColor"})
        dispatcher.execute({"op": "connect", "id": "analysis", "input": "image", "source": "source"})
        self.document = dispatcher.document
        for frame, x in ((1, 0), (2, 1), (3, 2)):
            dispatcher.execute({"op": "set_key", "id": "analysis", "param": "box_x", "frame": frame, "value": x})
            dispatcher.execute({"op": "set_key", "id": "analysis", "param": "box_width", "frame": frame, "value": 1})
            dispatcher.execute({"op": "set_key", "id": "analysis", "param": "box_height", "frame": frame, "value": 1})
        self.document = dispatcher.document

    def test_box_measurements_follow_interpolated_region_on_three_frames(self):
        evaluator = Evaluator()
        measured = []
        for frame in (1, 2, 3):
            p = region_params_at_frame(self.document, "analysis", frame)
            raster = evaluator.evaluate_raster(self.document, target="source", frame=frame)
            rgba = min_color_result(raster.pixels, (raster.data.x, raster.data.y),
                                    (p["box_x"], p["box_y"], p["box_width"], p["box_height"]))
            measured.append(float(rgba[0]))
        np.testing.assert_allclose(measured, (0.0, 0.25, 0.5), atol=1e-6)
        self.assertEqual(region_params_at_frame(self.document, "analysis", 2)["box_x"], 1)

    def test_sampler_line_measurements_follow_keyed_endpoints_across_frames(self):
        from nodebased.imaging import Evaluator
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "source", "type": "Constant",
                            "params": {"width": 5, "height": 1, "red": 0, "green": 0, "blue": 0, "alpha": 1}})
        for frame, value in enumerate((0.0, 0.25, 0.5, 0.75, 1.0), 1):
            dispatcher.execute({"op": "set_key", "id": "source", "param": "red", "frame": frame, "value": value})
        dispatcher.execute({"op": "create", "id": "sampler", "type": "Sampler"})
        dispatcher.execute({"op": "connect", "id": "sampler", "input": "image", "source": "source"})
        for frame, x in ((1, 0), (2, 1), (3, 2)):
            for name in ("sample_x0", "sample_x1"):
                dispatcher.execute({"op": "set_key", "id": "sampler", "param": name, "frame": frame, "value": x})
            for name in ("sample_y0", "sample_y1"):
                dispatcher.execute({"op": "set_key", "id": "sampler", "param": name, "frame": frame, "value": 0})
        measured = []
        evaluator = Evaluator()
        for frame in (1, 2, 3):
            p = region_params_at_frame(dispatcher.document, "sampler", frame)
            raster = evaluator.evaluate_raster(dispatcher.document, target="source", frame=frame)
            values = evaluator._sample_line(raster.pixels, (p["sample_x0"], p["sample_y0"]),
                                            (p["sample_x1"], p["sample_y1"]), (raster.data.x, raster.data.y))
            measured.append(float(values[:, 0].mean()))
        np.testing.assert_allclose(measured, (0.0, 0.25, 0.5), atol=1e-6)

    def test_curve_tool_metrics_use_region_at_each_sampled_frame(self):
        from nodebased.imaging import curve_tool_metrics
        evaluator = Evaluator()
        results = []
        for frame in (1, 2, 3):
            p = region_params_at_frame(self.document, "analysis", frame)
            raster = evaluator.evaluate_raster(self.document, target="source", frame=frame).to_display()
            metrics = curve_tool_metrics(raster, (p["box_x"], p["box_y"], p["box_width"], p["box_height"]))
            results.append(metrics["average_r"])
        np.testing.assert_allclose(results, (0.0, 0.25, 0.5), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
