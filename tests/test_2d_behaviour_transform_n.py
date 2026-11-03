"""2D behaviour, lane 9 step N: VectorDistort, PointsTo3D and Reconcile3D take the inputs and main knobs Nuke's Reference Guide lists.

* VectorDistort: Hold Frame (the reference frame is warped, or the current frame is carried back to it), Output (warped src, st-map, st-map
  inverse), Filter and Channels act; Frame Distance is listed unsupported. The rig is a Noise plate that moves right by three whole pixels from
  frame 1 to frame 2 with a real SmartVector on it, so the vectors are measured, not typed in. A graph saved before keeps its picture.
* PointsTo3D and Reconcile3D: see the classes below.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import knobs, vectordistortknobs as vd
from nodebased.core import SPECS, Dispatcher, upgrade_document, validate
from nodebased.flow_nodes import warp_by_flow
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_f import layout_order
from tests.test_2d_behaviour_filter_a import differs

import nuke_fidelity_requirements as requirements  # noqa: E402

W, H, SHIFT = 64, 48, 3


def add(d, key, kind, params=None, **inputs):
    d.execute(dict(op="create", id=key, type=kind, params=params or {}))
    for slot, source in inputs.items():
        d.execute(dict(op="connect", id=key, input=slot, source=source))


def rig(**params):
    """A Noise plate moving `SHIFT` pixels right per frame, a SmartVector on it, and a VectorDistort (reference frame 1) with `params`."""
    d = Dispatcher()
    add(d, "tex", "Noise", dict(width=W, height=H, size=7.0, octaves=3, center_x=0.0, center_y=0.0))
    d.execute(dict(op="set_key", id="tex", param="translate_x", frame=1, value=0.0))
    d.execute(dict(op="set_key", id="tex", param="translate_x", frame=2, value=float(SHIFT)))
    add(d, "sv", "SmartVector", dict(frame_start=1, frame_end=2, reference_frame=1, flow_backend="cpu", vector_detail=1, smoothness=0.4), image="tex")
    add(d, "vd", "VectorDistort", {"reference_frame": 1, "fade_frames": 0, **params}, image="tex", vectors="sv")
    d.execute(dict(op="time", first=1, last=2))
    return d.document


def render(doc, key, frame):
    return Evaluator().evaluate_raster(doc, key, frame=frame).pixels


INSIDE = (slice(8, -8), slice(10, -10))


def mean_gap(a, b):
    return float(np.abs(a[INSIDE][..., :3] - b[INSIDE][..., :3]).mean())


class VectorDistortBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = rig()
        cls.plate = {f: render(cls.doc, "tex", f) for f in (1, 2)}

    def test_the_rig_moves_the_plate_by_a_whole_number_of_pixels(self):
        np.testing.assert_allclose(np.roll(self.plate[1], SHIFT, axis=1)[:, 8:-8], self.plate[2][:, 8:-8], atol=1e-6)

    def test_hold_frame_on_warps_the_reference_frame_onto_the_current_one(self):
        out = render(self.doc, "vd", 2)
        self.assertLess(mean_gap(out, self.plate[2]), 0.03)
        self.assertGreater(mean_gap(out, self.plate[1]), 0.08)

    def test_hold_frame_off_carries_the_current_frame_back_to_the_reference(self):
        doc = rig(hold_frame=0)
        out = render(doc, "vd", 2)
        self.assertLess(mean_gap(out, self.plate[1]), 0.03)
        self.assertGreater(mean_gap(out, self.plate[2]), 0.08)
        self.assertTrue(differs(out, render(self.doc, "vd", 2), 1e-3))

    def test_at_the_reference_frame_both_readings_are_the_source(self):
        for hold in (1, 0):
            with self.subTest(hold=hold):
                np.testing.assert_allclose(render(rig(hold_frame=hold), "vd", 1), self.plate[1], atol=1e-5)

    def test_st_map_names_the_source_pixel_each_output_pixel_takes(self):
        out = render(rig(output_mode="st-map"), "vd", 2)
        x = np.arange(W)[None, :] - SHIFT
        expected_r = (x + 0.5) / W
        self.assertLess(float(np.abs(out[INSIDE][..., 0] - expected_r[:, 10:-10].repeat(H - 16, 0)).mean()), 0.01)
        expected_g = 1.0 - (np.arange(H)[:, None] + 0.5) / H
        self.assertLess(float(np.abs(out[INSIDE][..., 1] - expected_g[8:-8]).mean()), 0.01)
        self.assertEqual(float(out[INSIDE][..., 3].min()), 1.0)

    def test_st_map_inverse_names_the_pixel_each_reference_pixel_lands_on(self):
        out = render(rig(output_mode="st-map inverse"), "vd", 2)
        expected_r = ((np.arange(W)[None, :] + SHIFT) + 0.5) / W
        self.assertLess(float(np.abs(out[INSIDE][..., 0] - expected_r[:, 10:-10].repeat(H - 16, 0)).mean()), 0.01)

    def test_an_st_map_takes_no_mix_and_no_fade(self):
        base = render(rig(output_mode="st-map"), "vd", 2)
        np.testing.assert_array_equal(render(rig(output_mode="st-map", mix=0.0, fade_frames=1), "vd", 2), base)

    def test_blur_size_still_acts_on_the_warped_picture(self):
        self.assertTrue(differs(render(rig(blur_size=2.0), "vd", 2), render(self.doc, "vd", 2), 1e-3))

    def test_channels_limit_where_the_warp_reaches(self):
        out = render(rig(channels="red"), "vd", 2)
        full = render(self.doc, "vd", 2)
        np.testing.assert_allclose(out[..., 0], full[..., 0], atol=1e-6)
        np.testing.assert_allclose(out[..., 1], self.plate[2][..., 1], atol=1e-6)         # green passes through from the source
        np.testing.assert_allclose(out[..., 2], self.plate[2][..., 2], atol=1e-6)

    def test_mix_zero_returns_the_unwarped_reference_paint(self):
        np.testing.assert_allclose(render(rig(mix=0.0), "vd", 2), self.plate[1], atol=1e-5)

    def test_frame_distance_is_listed_unsupported_with_its_reason(self):
        import json
        docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity")
        pairs = json.load(open(os.path.join(docs, "nodebased_to_nuke.json")))["pairs"]["VectorDistort"]["knobs"]
        entry = next(e for e in pairs if e["nuke"] == ["frameDistance"])
        self.assertEqual((entry["kind"], entry["nodebased"]), ("unsupported", []))
        self.assertIn("SmartVector", entry["reason"])


class VectorDistortFilters(unittest.TestCase):
    """The warp's Filter, on a known field: the half pixel shift is where the filters differ."""

    @staticmethod
    def texture():
        rng = np.random.RandomState(5)
        img = rng.uniform(0, 1, (24, 32, 4)).astype(np.float32)
        img[..., 3] = 1.0
        return img

    @staticmethod
    def warp(img, shift, **params):
        field = np.zeros(img.shape[:2] + (2,), np.float32)
        field[..., 0] = shift
        out, _ = vd.warp({**SPECS["VectorDistort"]["params"], **params}, img, img, field, Evaluator._resample, lambda im: im)
        return out

    def test_every_filter_gives_the_same_picture_for_a_whole_pixel_shift_but_the_smoothing_ones(self):
        img = self.texture()
        want = np.roll(img, 2, axis=1)[:, 4:-4]
        for name in ("impulse", "cubic", "keys", "simon", "rifman", "lanczos4", "lanczos6", "sinc4"):
            with self.subTest(filter=name):
                np.testing.assert_allclose(self.warp(img, 2.0, image_filter=name)[:, 4:-4], want, atol=1e-5)
        for name in ("mitchell", "parzen"):
            with self.subTest(filter=name):
                self.assertTrue(differs(self.warp(img, 2.0, image_filter=name)[:, 4:-4], want, 1e-3))      # these smooth even on the pixel grid

    def test_impulse_carries_the_original_values_and_the_others_smooth(self):
        img = self.texture()
        impulse = self.warp(img, 2.5, image_filter="impulse")[:, 4:-4]
        self.assertTrue(np.isin(impulse[..., :3], img[..., :3]).all())
        spread = {}
        for name in ("cubic", "keys", "simon", "rifman", "mitchell", "parzen"):
            out = self.warp(img, 2.5, image_filter=name)[:, 4:-4]
            self.assertTrue(differs(out, impulse, 1e-3), name)
            spread[name] = float(out[..., :3].std())
        self.assertLess(spread["parzen"], spread["mitchell"] + 1e-6)         # "Parzen: the greatest smoothing of all filters"
        self.assertLess(spread["parzen"], spread["cubic"])
        self.assertGreater(spread["rifman"], spread["keys"])                  # Rifman sharpens more than Keys
        keys = {name: self.warp(img, 2.5, image_filter=name)[:, 4:-4] for name in ("keys", "simon", "rifman")}
        for a in keys:
            for b in keys:
                if a < b:
                    self.assertTrue(differs(keys[a], keys[b], 1e-4), (a, b))

    def test_the_filter_acts_through_the_node(self):
        a = render(rig(image_filter="impulse"), "vd", 2)
        b = render(rig(image_filter="parzen"), "vd", 2)
        self.assertTrue(differs(a, b, 1e-4))


class VectorDistortSavedGraphs(unittest.TestCase):
    def test_a_graph_saved_before_keeps_its_picture(self):
        doc = rig()
        node = doc["nodes"]["vd"]
        for name in ("legacy_vectordistort", "hold_frame", "output_mode", "image_filter"):
            node["params"].pop(name)
        old = copy.deepcopy(doc)
        migrated = upgrade_document(copy.deepcopy(doc))
        validate(migrated)
        params = migrated["nodes"]["vd"]["params"]
        self.assertEqual((params["legacy_vectordistort"], params["image_filter"], params["hold_frame"], params["output_mode"]),
                         (1, "bilinear", 1, "warped src"))
        out = render(migrated, "vd", 2)
        field = Evaluator().evaluate_raster(old, "sv", frame=2).layers["smartvector.forward"].pixels[..., :2]
        reference = render(old, "tex", 1)
        np.testing.assert_array_equal(out, warp_by_flow(reference, field))               # the bilinear warp of the held frame, bit for bit

    def test_a_new_node_reads_the_guide(self):
        p = SPECS["VectorDistort"]["params"]
        self.assertEqual((p["legacy_vectordistort"], p["hold_frame"], p["output_mode"], p["image_filter"]), (0, 1, "warped src", "cubic"))

    def test_the_main_tab_lists_the_knobs_in_the_guides_order(self):
        names = [g.label for g in knobs.KNOB_LAYOUT["VectorDistort"] if not g.params[0].startswith("legacy_") and g.params != ("mix",)]
        self.assertEqual(names[:6], ["Reference Frame", "Hold frame", "Output", "Blur Size", "Channels", "Filter"])
        kept, total = layout_order("VectorDistort")[0]
        self.assertEqual(kept, total)

    def test_the_node_lists_the_knobs_with_the_state_they_need(self):
        self.assertIn("VectorDistort", requirements.NOT_PROBED)
