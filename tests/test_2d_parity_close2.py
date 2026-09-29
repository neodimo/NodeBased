"""Step D2: close the partial rows MatchGrade/ColorTransfer, Blend, ContactSheet,
Encryptomatte, Erode (filter) and the HistEQ/Histogram/MinColor/Sampler group."""
import unittest

import numpy as np

from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Evaluator


class MatchGradeBakeTests(unittest.TestCase):
    def test_matchgrade_bakes_and_keeps_working_without_a_reference(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 16, "height": 8, "size": 1}})
        d.execute({"op": "create", "id": "target", "type": "Grade", "params": {
            "exposure": 0, "multiply": 2, "offset": 0.1}})
        d.execute({"op": "connect", "id": "target", "input": "image", "source": "src"})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        d.execute({"op": "connect", "id": "match", "input": "reference", "source": "target"})
        live = Evaluator().evaluate(dict(d.document, view="match"))

        frame = int(d.document["time"]["current"])
        src_px = Evaluator().evaluate_raster(d.document, target="src", frame=frame).pixels
        dst_px = Evaluator().evaluate_raster(d.document, target="target", frame=frame).pixels
        bake = []
        for channel, c in zip("rgb", range(3)):
            sm, tm = float(src_px[..., c].mean()), float(dst_px[..., c].mean())
            ss, ts = float(src_px[..., c].std()), float(dst_px[..., c].std())
            gain = ts / ss
            bake.append({"op": "set", "id": "match", "param": f"grade_gain_{channel}", "value": gain})
            bake.append({"op": "set", "id": "match", "param": f"grade_offset_{channel}", "value": tm - sm * gain})
        bake.append({"op": "set", "id": "match", "param": "match_analyzed", "value": 1})
        d.execute({"op": "batch", "commands": bake})

        baked_with_reference = Evaluator().evaluate(dict(d.document, view="match"))
        np.testing.assert_allclose(baked_with_reference, live, atol=1e-5)

        d.execute({"op": "connect", "id": "match", "input": "reference"})
        self.assertIsNone(d.document["nodes"]["match"]["inputs"].get("reference"))
        baked_without_reference = Evaluator().evaluate(dict(d.document, view="match"))
        np.testing.assert_allclose(baked_without_reference, live, atol=1e-5)

    def test_matchgrade_without_reference_or_bake_raises_a_clear_error(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 8, "height": 4, "size": 1}})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        with self.assertRaises(ValueError):
            Evaluator().evaluate(dict(d.document, view="match"))

    def test_matchgrade_bypass_still_passes_image_with_reference_unwired(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "src", "type": "Checker", "params": {
            "width": 8, "height": 4, "size": 1}})
        d.execute({"op": "create", "id": "match", "type": "MatchGrade"})
        d.execute({"op": "connect", "id": "match", "input": "image", "source": "src"})
        d.execute({"op": "disable", "id": "match", "value": True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="match")),
                                      Evaluator().evaluate(dict(d.document, view="src")))
