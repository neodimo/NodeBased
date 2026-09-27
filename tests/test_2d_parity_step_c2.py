import unittest

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator


class ColorC2Tests(unittest.TestCase):
    def test_histogram_levels_hand_values(self):
        image = np.zeros((1, 5, 4), np.float32)
        image[0, :, :3] = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], np.float32)[:, None]
        image[..., 3] = 1
        out = Evaluator._histogram_levels(image, {
            "black": 0.0, "white": 1.0, "black_out": 0.1, "white_out": 0.9, "gamma": 1.0})
        np.testing.assert_allclose(out[0, :, 0], [0.1, 0.1, 0.5, 0.9, 0.9], atol=1e-7)

    def test_histogram_of_known_ramp_is_flat(self):
        ramp = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, 256)
        image = np.repeat(ramp[..., None], 4, axis=-1)
        hist, _ = np.histogram(image[..., 0], bins=16, range=(0, 1))
        np.testing.assert_array_equal(hist, np.full(16, 16))

    def test_hist_eq_flattens_a_skewed_distribution(self):
        values = np.linspace(0, 1, 4096, dtype=np.float32) ** 2
        image = np.zeros((1, len(values), 4), np.float32)
        image[..., :3] = values[None, :, None]
        image[..., 3] = 1
        out = Evaluator._hist_eq(image, {"hist_eq_mode": "channels"})
        before = np.histogram(image[..., 0], bins=32, range=(0, 1))[0]
        after = np.histogram(out[..., 0], bins=32, range=(0, 1))[0]
        self.assertLess(float(np.var(after)), float(np.var(before)))

    def test_matchgrade_matches_known_gain_offset_and_bypasses(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 16, "height": 8, "size": 1}})
        d.execute({"op": "create", "id": "target", "type": "Grade", "params": {
            "exposure": 0, "multiply": 2, "offset": 0.1}})
        d.execute({"op": "connect", "id": "target", "input": "image", "source": "src"})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "match", "input": "reference", "source": "target"})
        doc = dict(d.document, view="match")
        np.testing.assert_allclose(Evaluator().evaluate(doc), Evaluator().evaluate(dict(d.document, view="target")), atol=1e-6)
        d.execute({"op": "disable", "id": "match", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="match")),
                                      Evaluator().evaluate(dict(d.document, view="src")))

    def test_min_color_finds_planted_darkest_pixel(self):
        image = np.ones((4, 5, 4), np.float32)
        image[..., 3] = 1
        image[2, 3, :3] = [0.01, 0.02, 0.03]
        rgba, xy = Evaluator._min_color(image)
        self.assertEqual(xy, (3, 2))
        np.testing.assert_allclose(rgba[:3], [0.01, 0.02, 0.03])

    def test_sampler_samples_line_endpoints(self):
        image = np.zeros((4, 5, 4), np.float32)
        image[..., 3] = 1
        image[..., 0] = np.arange(5, dtype=np.float32)[None, :]
        values = Evaluator._sample_line(image, (0, 0), (4, 3))
        self.assertEqual(values.shape[0], 5)
        self.assertEqual(float(values[0, 0]), 0.0)
        self.assertEqual(float(values[-1, 0]), 4.0)

    def test_analysis_nodes_pass_image_through(self):
        for kind in ("MinColor", "Sampler"):
            with self.subTest(kind=kind):
                d = Dispatcher()
                d.execute({"op": "create", "id": "src", "type": "Constant", "params": {
                    "width": 3, "height": 2, "red": 0.2, "green": 0.4, "blue": 0.6, "alpha": 1}})
                d.execute({"op": "create", "id": "analysis", "type": kind})
                d.execute({"op": "connect", "id": "analysis", "input": "image", "source": "src"})
                np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="analysis")),
                                              Evaluator().evaluate(dict(d.document, view="src")))


if __name__ == "__main__":
    unittest.main()
