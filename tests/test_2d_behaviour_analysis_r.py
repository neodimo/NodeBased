"""2D behaviour, lane 2 step R: Sampler's Normalize X, and the mask strip of Histogram and HistEQ.

The Sampler is an analysis node: its picture passes through and its knobs change the readout, so the readout is asserted here (the pixel cases in
tools/nuke_fidelity_cases_color.py cover the pass-through). Histogram gains Nuke's mask strip (mask channel, invert, inject, fringe, unpremult and its
invert) and HistEQ the mask channel and its invert, through the shared chrome of `ops2d_chrome`.
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import ops2d_chrome, samplerknobs
from nodebased.core import SPECS, LIMITS, Dispatcher, empty_document
from nodebased.imaging import Evaluator

H, W = 12, 16


def picture():
    rng = np.random.default_rng(21)
    out = rng.random((H, W, 4)).astype(np.float32)
    out[..., 3] = 1.0
    return out


class SamplerReadout(unittest.TestCase):
    params = dict(SPECS["Sampler"]["params"], sample_x0=1.0, sample_y0=1.0, sample_x1=13.0, sample_y1=6.0)

    def test_the_default_is_the_guides_and_older_documents_read_on(self):
        self.assertEqual(SPECS["Sampler"]["params"]["normalize_x"], 1)
        self.assertEqual(LIMITS["normalize_x"], (0, 1))
        node = {"type": "Sampler", "params": {"sample_channels": "rgb"}}
        ops2d_chrome.upgrade_nodes({"n": node})
        self.assertEqual(node["params"]["normalize_x"], 1)

    def test_normalized_x_runs_zero_to_one_and_off_it_runs_in_pixels(self):
        on = samplerknobs.readout(picture(), dict(self.params, normalize_x=1))
        off = samplerknobs.readout(picture(), dict(self.params, normalize_x=0))
        length = math.hypot(12.0, 5.0)
        self.assertAlmostEqual(on["x"][0], 0.0)
        self.assertAlmostEqual(on["x"][-1], 1.0)
        self.assertAlmostEqual(off["x"][-1], length)
        np.testing.assert_allclose(np.asarray(off["x"]), np.asarray(on["x"]) * length, atol=1e-9)
        self.assertEqual(on["labels"], ("0", "1"))
        self.assertEqual(off["labels"], ("0", f"{length:.0f} px"))
        self.assertTrue(np.array_equal(on["values"], off["values"]))              # the sampled values do not depend on the axis

    def test_the_knob_changes_the_readout_and_not_the_picture(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "plate", "type": "Constant", "params": {"width": W, "height": H, "red": 0.3, "green": 0.4, "blue": 0.5, "alpha": 1.0}})
        shown = {}
        for flag in (0, 1):
            d.execute({"op": "create", "id": f"s{flag}", "type": "Sampler", "params": {"normalize_x": flag}})
            d.execute({"op": "connect", "id": f"s{flag}", "input": "image", "source": "plate"})
            shown[flag] = np.asarray(Evaluator().evaluate_raster(dict(d.document, view=f"s{flag}"), target=f"s{flag}", frame=1).pixels)
        base = np.asarray(Evaluator().evaluate_raster(dict(d.document, view="plate"), target="plate", frame=1).pixels)
        self.assertTrue(np.array_equal(shown[0], base) and np.array_equal(shown[1], base))
        a = samplerknobs.readout(base, dict(self.params, normalize_x=1))
        b = samplerknobs.readout(base, dict(self.params, normalize_x=0))
        self.assertNotEqual(a["x"][-1], b["x"][-1])

    def test_the_channels_filter_the_readout(self):
        for name, wanted in (("rgb", (0, 1, 2)), ("rgba", (0, 1, 2, 3)), ("alpha", (3,))):
            read = samplerknobs.readout(picture(), dict(self.params, sample_channels=name))
            self.assertEqual(read["channels"], wanted)
            self.assertEqual(read["values"].shape[1], len(wanted))

    def test_a_single_sample_does_not_divide_by_zero(self):
        self.assertEqual(samplerknobs.x_values(1, (0, 0), (5, 5), True), [0.0])


def masked_picture(kind, params, mask_alpha, **chrome):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "plate", "type": "Constant", "params": {"width": W, "height": H, "red": 0.5, "green": 0.25, "blue": 0.75, "alpha": 1.0}})
    d.execute({"op": "create", "id": "mask", "type": "Constant", "params": {"width": W, "height": H, "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": mask_alpha}})
    d.execute({"op": "create", "id": "n", "type": kind, "params": dict(params, **chrome)})
    d.execute({"op": "connect", "id": "n", "input": "image", "source": "plate"})
    d.execute({"op": "connect", "id": "n", "input": "mask", "source": "mask"})
    return np.asarray(Evaluator().evaluate_raster(dict(d.document, view="n"), target="n", frame=1).pixels, np.float64)


def unmasked_picture(kind, params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "plate", "type": "Constant", "params": {"width": W, "height": H, "red": 0.5, "green": 0.25, "blue": 0.75, "alpha": 1.0}})
    d.execute({"op": "create", "id": "n", "type": kind, "params": dict(params)})
    d.execute({"op": "connect", "id": "n", "input": "image", "source": "plate"})
    return np.asarray(Evaluator().evaluate_raster(dict(d.document, view="n"), target="n", frame=1).pixels, np.float64)


class MaskStrip(unittest.TestCase):
    def test_the_nodes_carry_nukes_rows(self):
        histogram = SPECS["Histogram"]["params"]
        for name, default in (("maskChannelInput", "none"), ("invert_mask", 0), ("inject", 0), ("fringe", 0), ("unpremult", "none"), ("invert_unpremult", 0)):
            self.assertEqual(histogram[name], default, name)
        histeq = SPECS["HistEQ"]["params"]
        self.assertEqual((histeq["maskChannelInput"], histeq["invert_mask"]), ("none", 0))
        self.assertNotIn("channels", histogram)             # the guide's Histogram and HistEQ tables have no channels knob
        self.assertNotIn("channels", histeq)

    def test_the_mask_gates_histogram(self):
        params = {"black": 0.2, "white": 0.8}
        full = unmasked_picture("Histogram", params)
        plate = unmasked_picture("Histogram", {})
        self.assertFalse(np.allclose(full, plate, atol=1e-4))                 # the stretch does something
        np.testing.assert_allclose(masked_picture("Histogram", params, 1.0), full, atol=1e-5)          # a full mask lets all of it through
        np.testing.assert_allclose(masked_picture("Histogram", params, 0.0), plate, atol=1e-5)         # an empty mask leaves the input as it was
        np.testing.assert_allclose(masked_picture("Histogram", params, 0.0, invert_mask=1), full, atol=1e-5)   # inverted, the empty mask is the full one

    def test_the_mask_gates_histeq(self):
        # a constant picture is already flat: give the equaliser something to change
        params = {"hist_eq_mode": "luminance"}
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "plate", "type": "Ramp", "params": {"width": W, "height": H}})
        d.execute({"op": "create", "id": "mask", "type": "Constant", "params": {"width": W, "height": H, "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 0.0}})
        d.execute({"op": "create", "id": "n", "type": "HistEQ", "params": params})
        d.execute({"op": "connect", "id": "n", "input": "image", "source": "plate"})
        free = np.asarray(Evaluator().evaluate_raster(dict(d.document, view="n"), target="n", frame=1).pixels, np.float64)
        plate = np.asarray(Evaluator().evaluate_raster(dict(d.document, view="plate"), target="plate", frame=1).pixels, np.float64)
        d.execute({"op": "connect", "id": "n", "input": "mask", "source": "mask"})
        gated = np.asarray(Evaluator().evaluate_raster(dict(d.document, view="n"), target="n", frame=1).pixels, np.float64)
        np.testing.assert_allclose(gated, plate, atol=1e-5)          # an empty mask leaves the input as it was
        d.execute({"op": "set", "id": "n", "param": "invert_mask", "value": 1})
        inverted = np.asarray(Evaluator().evaluate_raster(dict(d.document, view="n"), target="n", frame=1).pixels, np.float64)
        np.testing.assert_allclose(inverted, free, atol=1e-5)


class ProbesAndPairs(unittest.TestCase):
    def test_every_new_param_is_in_the_mapping(self):
        import json
        mapping = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity", "nodebased_to_nuke.json")))["pairs"]
        sampler = {n for e in mapping["Sampler"]["knobs"] if e["kind"] in ("same", "renamed", "components") for n in e["nodebased"]}
        self.assertIn("normalize_x", sampler)
        histogram = {n for e in mapping["Histogram"]["knobs"] if e["kind"] == "same" for n in e["nodebased"]}
        self.assertTrue({"maskChannelInput", "invert_mask", "inject", "fringe", "unpremult", "invert_unpremult"} <= histogram)
        histeq = {n for e in mapping["HistEQ"]["knobs"] if e["kind"] == "same" for n in e["nodebased"]}
        self.assertTrue({"maskChannelInput", "invert_mask"} <= histeq)

    def test_the_probe_finds_every_sampler_and_histogram_knob_acting(self):
        import nuke_fidelity_requirements as req
        from nuke_fidelity_cases import make_fixtures
        fx = make_fixtures()
        acts = req.probe_acts("Sampler", ["sample_channels", "sample_x0", "sample_y0", "normalize_x"], fx)
        self.assertTrue(all(acts.values()), acts)
        acts = req.probe_acts("Histogram", ["black", "white", "gamma", "black_out", "white_out", "mix"], fx)
        self.assertIsNotNone(acts)
        self.assertTrue(all(acts.values()), acts)


if __name__ == "__main__":
    unittest.main()
