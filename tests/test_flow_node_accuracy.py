"""Node-level midpoint and panning checks for the flow retimer and motion blur."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.opticalflow import _sample


class FlowNodeAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pan.%04d.exr"
        h, w = 128, 144
        rng = np.random.default_rng(19)
        texture = rng.random((h, w), dtype=np.float32)
        for _ in range(4):
            texture = (texture + np.roll(texture, 1, 0) + np.roll(texture, -1, 0)
                       + np.roll(texture, 1, 1) + np.roll(texture, -1, 1)) / 5
        self.first = np.zeros((h, w, 4), np.float32)
        self.first[..., :3] = texture[..., None] * 0.75
        self.first[..., 3] = 1
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        self.second = _sample(self.first, xx-4.0, yy-2.0).astype(np.float32)
        write_exr(str(self.path).replace("%04d", "0001"), self.first, bits="float")
        write_exr(str(self.path).replace("%04d", "0002"), self.second, bits="float")

    def graph(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Read",
                   "params": {"path": str(self.path), "missing": "hold"}})
        d.document["time"].update(first=1, last=2, current=1)
        return d

    def test_kronos_motion_midpoint_beats_frame_blend(self):
        d = self.graph()
        d.execute({"op": "create", "id": "k", "type": "Kronos",
                   "params": {"speed": 0.5, "interpolation": "motion", "shutter_samples": 1}})
        d.execute({"op": "connect", "id": "k", "input": "image", "source": "src"})
        ev = Evaluator()
        base = ev.evaluate_raster(d.document, "src", frame=1).pixels
        yy, xx = np.mgrid[:base.shape[0], :base.shape[1]].astype(np.float32)
        truth = _sample(base, xx-2.0, yy-1.0)
        motion = ev.evaluate_raster(d.document, "k", frame=3).pixels
        d.execute({"op": "set", "id": "k", "param": "interpolation", "value": "frame"})
        blend = Evaluator().evaluate_raster(d.document, "k", frame=3).pixels
        region = np.s_[12:-12, 12:-12, :3]
        motion_error = float(np.mean(np.abs(motion[region] - truth[region])))
        blend_error = float(np.mean(np.abs(blend[region] - truth[region])))
        self.assertLess(motion_error, blend_error * 0.8,
                        f"motion midpoint error {motion_error:.6f}, frame blend {blend_error:.6f}")

    def test_motion_blur_panning_matches_vector_blur_with_true_vectors(self):
        d = self.graph()
        d.execute({"op": "create", "id": "auto", "type": "MotionBlur",
                   "params": {"shutter": 1.0, "samples": 4, "shutter_offset": "centred"}})
        d.execute({"op": "connect", "id": "auto", "input": "image", "source": "src"})
        d.execute({"op": "create", "id": "true", "type": "Constant",
                   "params": {"width": 144, "height": 128, "red": 4.0, "green": 2.0,
                               "blue": 0.0, "alpha": 1.0}})
        d.execute({"op": "create", "id": "reference", "type": "VectorBlur"})
        d.execute({"op": "connect", "id": "reference", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "reference", "input": "uv", "source": "true"})
        ev = Evaluator()
        actual = ev.evaluate_raster(d.document, "auto", frame=1).pixels
        expected = ev.evaluate_raster(d.document, "reference", frame=1).pixels
        region = np.s_[12:-12, 12:-12, :3]
        error = float(np.mean(np.abs(actual[region] - expected[region])))
        self.assertLess(error, 0.05, f"automatic-vs-true-vector blur MAE {error:.6f}")


if __name__ == "__main__":
    unittest.main()
