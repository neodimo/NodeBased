"""2D behaviour, lane 9 step L (finish): MotionBlur's estimation knobs, BgVecs and Matte, and VectorBlur's preprocessing and blur knobs.

* MotionBlur: Resampling (the Lanczos filters overshoot a step, bilinear does not); Matte Channel with a Matte input and BgVecs (the vector
  is matte * FgVecs + (1 - matte) * BgVecs, exact on constants); Method (Regularized vectors vary less than Local ones); Vector Detail
  (a vector at every round(1 / detail) pixels); the matte keeps the foreground's vectors from dragging the background's; a saved node keeps
  the older estimate (`legacy_motionblur`);
* VectorBlur: uv offset, blur uv, blur type with motion falloff, expand blur bbox, soft lines and blur inside each change the picture the way
  the guide states, on constant-vector plates; a saved node keeps its picture.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import motionblurknobs, vectorblurknobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels
from tests.test_2d_behaviour_filter_f import layout_order
from tests.test_2d_behaviour_filter_l import H, W, interior, labels, ramp_plate, vector_plate


def blur_params(**knob):
    return {**SPECS["VectorBlur"]["params"], "vector_blur_type": "uniform", **knob}


# ---------------------------------------------------------------------------------------------------------------------- MotionBlur
class MotionBlurFgVecsKnobTests(unittest.TestCase):
    def graph(self, u=4.0, plate="ramp", **knob):
        g = ramp_plate() if plate == "ramp" else Graph()
        if plate != "ramp":
            g.add("plate", "Checker", dict(width=W, height=H, size=3))
        vector_plate(g, u)
        g.add("node", "MotionBlur", dict(knob), image="plate", fgvecs="vec")
        return g

    def test_a_new_node_reads_the_guides_defaults(self):
        params = SPECS["MotionBlur"]["params"]
        self.assertEqual((params["motion_method"], params["motion_vector_detail"], params["motion_resampling"],
                          params["motion_matte_channel"], params["legacy_motionblur"]), ("Regularized", 0.2, "Bilinear", "None", 0))
        self.assertEqual(SPECS["MotionBlur"]["optional_inputs"], ["mask", "fgvecs", "bgvecs", "matte"])

    def test_lanczos_overshoots_a_step_that_bilinear_keeps_in_range(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[:, W // 2:, :] = 1.0
        u = np.full((H, W), 2.5)
        zero = np.zeros_like(u)
        results = {}
        for name in motionblurknobs.RESAMPLING:
            p = motionblurknobs.NEW | {"shutter": 1.0, "samples": 3, "shutter_offset": "start", "motion_resampling": name}
            from nodebased import motionblur
            results[name] = Evaluator._vector_blur(picture, u, zero, motion_params(p))[..., 0]
        self.assertLessEqual(float(results["Bilinear"].max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(results["Bilinear"].min()), -1e-6)
        for name in ("Lanczos4", "Lanczos6"):
            self.assertTrue(float(results[name].max()) > 1.0005 or float(results[name].min()) < -0.0005, name)
            self.assertTrue(differs(results[name], results["Bilinear"]))

    def test_a_flat_plate_is_the_same_under_every_filter(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=W, height=H, red=0.4, green=0.5, blue=0.6, alpha=1.0))
        vector_plate(g, 3.5)
        for name in motionblurknobs.RESAMPLING:
            g.add(name, "MotionBlur", dict(shutter=1.0, samples=5, motion_resampling=name), image="plate", fgvecs="vec")
            np.testing.assert_allclose(interior(evaluator_pixels(g.doc, name), 8), 0.0 + interior(evaluator_pixels(g.doc, "plate"), 8), atol=1e-5)

    def test_the_resampling_knob_changes_the_node_picture(self):
        a = evaluator_pixels(self.graph(2.5, plate="checker", shutter=1.0, samples=4).doc, "node")
        b = evaluator_pixels(self.graph(2.5, plate="checker", shutter=1.0, samples=4, motion_resampling="Lanczos4").doc, "node")
        self.assertTrue(differs(a, b))


def motion_params(p):
    from nodebased import motionblur
    return motionblur.fgvecs_blur_params({**SPECS["MotionBlur"]["params"], **p})


class MotionBlurBgVecsTests(unittest.TestCase):
    """The guide: separate foreground and background vectors, joined by the matte that made them."""

    def graph(self, fg=4.0, bg=0.0, matte_alpha=1.0, matte_rgb=0.5, **knob):
        g = ramp_plate()
        vector_plate(g, fg)
        g.add("bgvec", "Constant", dict(width=W, height=H, red=bg, green=0.0, blue=0.0, alpha=1.0))
        g.add("matte", "Constant", dict(width=W, height=H, red=matte_rgb, green=matte_rgb, blue=matte_rgb, alpha=matte_alpha))
        g.add("node", "MotionBlur", dict(shutter=1.0, samples=5, shutter_offset="start", **knob),
              image="plate", fgvecs="vec", bgvecs="bgvec", matte="matte")
        return g

    def single(self, u):
        g = ramp_plate()
        vector_plate(g, u)
        g.add("node", "MotionBlur", dict(shutter=1.0, samples=5, shutter_offset="start"), image="plate", fgvecs="vec")
        return evaluator_pixels(g.doc, "node")

    def test_the_vector_is_matte_times_fg_plus_one_minus_matte_times_bg(self):
        for token, kwargs, mixed in (("Matte Alpha", dict(matte_alpha=0.25), 0.25), ("Matte Inverted Alpha", dict(matte_alpha=0.25), 0.75),
                                     ("Matte Luminance", dict(matte_rgb=0.5), 0.5), ("Matte Inverted Luminance", dict(matte_rgb=0.25), 0.75),
                                     ("Source Alpha", {}, 1.0), ("Source Inverted Alpha", {}, 0.0)):
            with self.subTest(token=token):
                got = evaluator_pixels(self.graph(fg=8.0, bg=2.0, motion_matte_channel=token, **kwargs).doc, "node")
                np.testing.assert_allclose(got, self.single(8.0 * mixed + 2.0 * (1.0 - mixed)), atol=1e-5)

    def test_without_a_matte_channel_the_foreground_vectors_are_used(self):
        got = evaluator_pixels(self.graph(fg=8.0, bg=2.0).doc, "node")
        np.testing.assert_allclose(got, self.single(8.0), atol=1e-6)

    def test_a_matte_channel_without_its_input_is_no_matte(self):
        g = ramp_plate()
        vector_plate(g, 8.0)
        g.add("bgvec", "Constant", dict(width=W, height=H, red=2.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("node", "MotionBlur", dict(shutter=1.0, samples=5, shutter_offset="start", motion_matte_channel="Matte Alpha"),
              image="plate", fgvecs="vec", bgvecs="bgvec")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), self.single(8.0), atol=1e-6)

    def test_the_matte_is_ignored_with_one_set_of_vectors(self):
        g = ramp_plate()
        vector_plate(g, 8.0)
        g.add("matte", "Constant", dict(width=W, height=H, red=0.0, green=0.0, blue=0.0, alpha=0.0))
        g.add("node", "MotionBlur", dict(shutter=1.0, samples=5, shutter_offset="start", motion_matte_channel="Matte Alpha"),
              image="plate", fgvecs="vec", matte="matte")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), self.single(8.0), atol=1e-6)

    def test_bgvecs_alone_are_the_vectors(self):
        g = ramp_plate()
        g.add("bgvec", "Constant", dict(width=W, height=H, red=6.0, green=0.0, blue=0.0, alpha=1.0))
        g.add("node", "MotionBlur", dict(shutter=1.0, samples=5, shutter_offset="start"), image="plate", bgvecs="bgvec")
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node"), self.single(6.0), atol=1e-6)


def moving_pair(dx=3, h=64, w=96, seed=1):
    """Two smooth-textured frames, a 32 x 32 square of texture moving dx pixels right over a static background, and the true vectors."""
    rng = np.random.RandomState(seed)

    def smooth(a):
        for axis in (0, 1):
            a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, "valid"), axis, a)
        return a
    background, texture = smooth(rng.rand(h, w)), smooth(rng.rand(h, w))

    def frame(shift):
        mask = np.zeros((h, w))
        mask[16:48, 30 + shift:62 + shift] = 1
        image = background * (1 - mask) + np.roll(texture, shift, axis=1) * mask
        out = np.zeros((h, w, 4), np.float32)
        out[..., :3], out[..., 3] = image[..., None], 1.0
        return out, mask
    (a, matte), (b, _) = frame(0), frame(dx)
    truth = np.zeros((h, w), np.float32)
    truth[16:48, 30:62] = dx
    return a, b, matte.astype(np.float32), truth


class MotionBlurEstimationTests(unittest.TestCase):
    P = {"flow_backend": "cpu"}

    def edge_error(self, flow, truth):
        band = np.zeros(truth.shape, bool)
        band[16:48, 24:36] = band[16:48, 56:68] = True
        return float(np.abs(flow[..., 0] - truth)[band].mean())

    def test_the_vector_density_is_one_vector_in_round_one_over_detail_pixels(self):
        for detail, step in ((1.0, 1), (0.5, 2), (0.2, 5), (0.34, 3)):
            self.assertEqual(motionblurknobs.density_step({"motion_vector_detail": detail}, 200, 300), step)
        self.assertEqual(motionblurknobs.density_step({"motion_vector_detail": 0.02}, 40, 300), 3)      # never below a 12 pixel field

    def test_a_vector_at_every_other_pixel_is_the_half_size_field_scaled_up(self):
        from nodebased.opticalflow import flow_pair
        a, b, _, _ = moving_pair()
        half = motionblurknobs.estimate_flow(a, b, {**self.P, "motion_method": "Local", "motion_vector_detail": 0.5})
        low_a, low_b = motionblurknobs._down(a, 2), motionblurknobs._down(b, 2)
        flow, _, _ = flow_pair(low_a, low_b, vector_detail=4, smoothness=0.0, backend="cpu")
        np.testing.assert_allclose(half, motionblurknobs._up(np.asarray(flow, np.float32), 2, 64, 96), atol=1e-6)

    def test_lower_detail_changes_the_vectors(self):
        a, b, _, _ = moving_pair()
        fine = motionblurknobs.estimate_flow(a, b, {**self.P, "motion_vector_detail": 1.0})
        coarse = motionblurknobs.estimate_flow(a, b, {**self.P, "motion_vector_detail": 0.2})
        self.assertTrue(differs(fine, coarse, 1e-3))

    def test_regularized_vectors_vary_less_than_local_ones(self):
        a, b, _, _ = moving_pair()
        local = motionblurknobs.estimate_flow(a, b, {**self.P, "motion_method": "Local", "motion_vector_detail": 1.0})
        regular = motionblurknobs.estimate_flow(a, b, {**self.P, "motion_method": "Regularized", "motion_vector_detail": 1.0})
        variation = lambda f: float(np.abs(np.diff(f[..., 0], axis=1)).mean() + np.abs(np.diff(f[..., 0], axis=0)).mean())
        self.assertLess(variation(regular), 0.5 * variation(local))
        self.assertTrue(differs(local, regular, 1e-3))

    def test_the_matte_keeps_the_foreground_from_dragging_the_background(self):
        a, b, matte, truth = moving_pair()
        for method in motionblurknobs.METHODS:
            with self.subTest(method=method):
                p = {**self.P, "motion_method": method, "motion_vector_detail": 1.0}
                plain = motionblurknobs.estimate_flow(a, b, p)
                split = motionblurknobs.estimate_flow(a, b, p, matte)
                self.assertLess(self.edge_error(split, truth), 0.9 * self.edge_error(plain, truth))

    def test_a_constant_field_is_the_same_under_any_matte(self):
        flow = np.zeros((24, 30, 2), np.float32)
        flow[..., 0], flow[..., 1] = 2.0, -1.0
        for matte in (np.ones((24, 30), np.float32), np.zeros((24, 30), np.float32), np.full((24, 30), 0.3, np.float32)):
            np.testing.assert_allclose(motionblurknobs.split_by_matte(flow, matte), flow, atol=1e-4)

    def test_the_split_averages_inside_and_outside_the_matte_apart(self):
        flow = np.zeros((24, 30, 2), np.float32)
        flow[:, 15:, 0] = 4.0
        matte = np.zeros((24, 30), np.float32)
        matte[:, 15:] = 1.0
        split = motionblurknobs.split_by_matte(flow, matte)
        np.testing.assert_allclose(split, flow, atol=1e-4)                  # a matte that follows the edge leaves both sides as they were


def moving_graph(**knob):
    g = Graph()
    g.add("plate", "Noise", dict(width=96, height=64, size=16, octaves=3))
    g.add("move", "Transform", dict(filter="nearest"), image="plate")
    g.d.execute(dict(op="set_curve", id="move", param="translate_x",
                     curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(4 * (f - 1))) for f in range(-5, 30)])))
    g.add("matte", "Constant", dict(width=96, height=64, red=0.5, green=0.5, blue=0.5, alpha=0.5))
    inputs = dict(image="move")
    if knob.pop("with_matte", False):
        inputs["matte"] = "matte"
    g.add("node", "MotionBlur", dict(shutter=1.0, samples=4, flow_backend="cpu", **knob), **inputs)
    g.d.execute(dict(op="time", first=1, last=8, current=3))
    return g


def moving_pixels(g, frame=3):
    return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame).pixels


class MotionBlurNodeTests(unittest.TestCase):
    def test_each_estimation_knob_changes_the_picture_of_a_moving_plate(self):
        base = moving_pixels(moving_graph())
        for knob, value in (("motion_method", "Local"), ("motion_vector_detail", 1.0), ("motion_resampling", "Lanczos4")):
            with self.subTest(knob=knob):
                self.assertTrue(differs(base, moving_pixels(moving_graph(**{knob: value})), 1e-5))

    def test_the_matte_channel_acts_with_its_matte_and_not_without_it(self):
        base = moving_pixels(moving_graph())
        for token in ("Source Alpha", "Source Inverted Alpha", "Matte Alpha", "Matte Luminance"):
            with self.subTest(token=token):
                wired = moving_pixels(moving_graph(with_matte=True, motion_matte_channel=token))
                self.assertTrue(differs(base, wired, 1e-6))
        # the Matte input is not wired: a channel that names it is no matte
        np.testing.assert_array_equal(moving_pixels(moving_graph(motion_matte_channel="Matte Alpha")), base)

    def test_a_still_plate_stays_still_under_the_new_estimate(self):
        g = ramp_plate()
        g.add("node", "MotionBlur", dict(shutter=1.0, samples=3, flow_backend="cpu"), image="plate")
        np.testing.assert_allclose(interior(evaluator_pixels(g.doc, "node")), interior(evaluator_pixels(g.doc, "plate")), atol=1e-5)

    def test_a_saved_node_keeps_the_older_estimate(self):
        from nodebased.opticalflow import flow_pair
        g = moving_graph()
        old = copy.deepcopy(g.doc)
        for name in list(motionblurknobs.NEW):
            old["nodes"]["node"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual((params["legacy_motionblur"], params["motion_method"]), (1, "Local"))
        ev = Evaluator()
        a = ev.evaluate_raster(dict(loaded, view="move"), target="move", frame=2.5)
        b = ev.evaluate_raster(dict(loaded, view="move"), target="move", frame=3.5)
        window = a.data.union(b.data)
        flow, _, _ = flow_pair(a.fit(window), b.fit(window), vector_detail=4, smoothness=1.0, backend=params["flow_backend"])
        expected = Evaluator._vector_blur(a.fit(window), flow[..., 0], flow[..., 1],
                                          {"vector_scale": 1.0, "max_length": 100.0, "vector_offset": 0.0, "vector_method": "forward",
                                           "vector_alpha": "none", "samples": int(params["samples"])})
        np.testing.assert_array_equal(moving_pixels(type("G", (), {"doc": loaded})()), expected)

    def test_the_main_tab_follows_the_guide(self):
        self.assertEqual(labels("MotionBlur")[:6], ["Shutter samples", "Shutter time", "Method", "Vector detail", "Resampling", "Matte channel"])
        kept, total = layout_order("MotionBlur")[0]
        self.assertEqual(kept, total)


# ---------------------------------------------------------------------------------------------------------------------- VectorBlur
class VectorBlurKnobTests(unittest.TestCase):
    def blurred(self, u=4.0, v=0.0, **knob):
        g = ramp_plate()
        vector_plate(g, u, v)
        g.add("node", "VectorBlur", dict(knob), image="plate", uv="vec")
        return evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate")

    def test_uv_offset_is_added_to_both_vectors(self):
        shifted, _ = self.blurred(u=3.0, v=-1.0, vector_scale=1.0, vector_offset=0.0, vector_blur_type="uniform", uv_offset=1.0)
        plain, _ = self.blurred(u=4.0, v=0.0, vector_scale=1.0, vector_offset=0.0, vector_blur_type="uniform")
        np.testing.assert_allclose(shifted, plain, atol=1e-6)
        cancelled, plate = self.blurred(u=3.0, v=3.0, vector_scale=1.0, vector_offset=0.0, vector_blur_type="uniform", uv_offset=-3.0)
        np.testing.assert_allclose(cancelled, plate, atol=1e-6)                                    # (0, 0): nothing moves

    def test_uv_offset_shows_in_the_uvs_output(self):
        out, _ = self.blurred(u=1.0, v=2.0, vector_output="uvs", uv_offset=-0.5)
        np.testing.assert_allclose(out[..., 0], 0.5, atol=1e-6)
        np.testing.assert_allclose(out[..., 1], 1.5, atol=1e-6)

    def test_blur_uv_leaves_constant_vectors_and_softens_a_step(self):
        constant_u, constant_v = np.full((H, W), 3.0), np.full((H, W), -1.0)
        for mode in ("uniform", "linear"):
            u, v = vectorblurknobs.blur_uvs(constant_u, constant_v, mode)
            np.testing.assert_allclose(u, constant_u, atol=1e-12)
            np.testing.assert_allclose(v, constant_v, atol=1e-12)
        step = np.zeros((H, W))
        step[:, W // 2:] = 8.0
        reach = {}
        for mode in ("none", "uniform", "linear"):
            blurred, _ = vectorblurknobs.blur_uvs(step, step, mode)
            reach[mode] = int((np.abs(blurred - step)[H // 2] > 1e-9).sum())
        self.assertEqual(reach["none"], 0)
        self.assertEqual(reach["uniform"], 2)            # one pixel each side of the step
        self.assertEqual(reach["linear"], 8)             # four each side: the stronger effect the guide states
        self.assertGreater(reach["linear"], reach["uniform"])

    def test_blur_uv_changes_the_picture_where_the_vectors_change(self):
        g = ramp_plate()
        g.add("uvs", "Ramp", dict(width=W, height=H, p0_x=0.0, p0_y=0.0, p1_x=float(W - 1), p1_y=0.0, color0_red=0.0, color0_green=0.0,
                                  color0_blue=0.0, color0_alpha=1.0, color1_red=8.0, color1_green=0.0, color1_blue=0.0, color1_alpha=1.0))
        g.add("none", "VectorBlur", dict(blur_uv="none"), image="plate", uv="uvs")
        g.add("soft", "VectorBlur", dict(blur_uv="linear"), image="plate", uv="uvs")
        self.assertTrue(differs(evaluator_pixels(g.doc, "none"), evaluator_pixels(g.doc, "soft")))

    def test_the_gaussian_profile_weights_the_middle_of_the_line(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[..., 3] = 1.0
        picture[5, 20, :3] = 1.0
        u = np.full((H, W), 4.0)
        zero = np.zeros_like(u)
        for falloff in (0.33, 1.0):
            with self.subTest(falloff=falloff):
                p = blur_params(vector_blur_type="gaussian", motion_falloff=falloff, vector_scale=1.0, vector_offset=0.0, samples=5, max_length=0.0)
                row = Evaluator._vector_blur(picture, u, zero, p)[5, :, 0]
                sigma = max(0.001, falloff / 2.0)
                weights = np.exp(-((np.arange(5) / 4.0 - 0.5) ** 2) / (2 * sigma * sigma))
                np.testing.assert_allclose(row[20:25], (weights / weights.sum())[::-1], atol=1e-6)
        sharp = Evaluator._vector_blur(picture, u, zero, blur_params(vector_blur_type="gaussian", motion_falloff=0.0, vector_scale=1.0, vector_offset=0.0, samples=5, max_length=0.0))[5, :, 0]
        np.testing.assert_allclose(sharp[22], 1.0, atol=1e-6)                      # falloff 0: only the middle of the line is left
        uniform = Evaluator._vector_blur(picture, u, zero, blur_params(vector_scale=1.0, vector_offset=0.0, samples=5, max_length=0.0))[5, :, 0]
        np.testing.assert_allclose(uniform[20:25], 0.2, atol=1e-6)

    def test_vector_blur_type_and_falloff_change_the_node_picture(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=W, height=H, size=3))
        vector_plate(g, 6.0)
        g.add("uniform", "VectorBlur", dict(vector_blur_type="uniform"), image="plate", uv="vec")
        g.add("gauss", "VectorBlur", dict(vector_blur_type="gaussian"), image="plate", uv="vec")
        g.add("sharp", "VectorBlur", dict(vector_blur_type="gaussian", motion_falloff=0.05), image="plate", uv="vec")
        base = evaluator_pixels(g.doc, "gauss")
        self.assertTrue(differs(evaluator_pixels(g.doc, "uniform"), base))
        self.assertTrue(differs(evaluator_pixels(g.doc, "sharp"), base))

    def test_expand_blur_bbox_grows_the_picture_and_the_blur_runs_into_the_new_ring(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[..., 3] = 1.0
        g = Graph()
        g.add("plate", "Constant", dict(width=W, height=H, red=1.0, green=1.0, blue=1.0, alpha=1.0))
        vector_plate(g, -4.0)
        g.add("node", "VectorBlur", dict(vector_blur_type="uniform", grow_bbox=3, vector_offset=0.0), image="plate", uv="vec")
        g.add("plain", "VectorBlur", dict(vector_blur_type="uniform", vector_offset=0.0), image="plate", uv="vec")
        grown = Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=1)
        plain = Evaluator().evaluate_raster(dict(g.doc, view="plain"), target="plain", frame=1)
        self.assertEqual((plain.data.width, plain.data.height), (W, H))
        self.assertEqual((grown.data.width, grown.data.height, grown.data.x, grown.data.y), (W + 6, H + 6, -3, -3))
        ring = grown.pixels[H // 2 + 3, 0:3]                     # left of the picture: the line reaches back into it
        self.assertGreater(float(ring[2, 0]), 0.0)
        self.assertGreater(float(ring[2, 0]), float(ring[0, 0]))
        np.testing.assert_allclose(grown.pixels[3:3 + H, 3 + 8:3 + W - 8, :3], plain.pixels[:, 8:W - 8, :3], atol=1e-6)

    def test_soft_lines_draw_a_wider_line(self):
        stripes = np.zeros((H, W, 4), np.float32)
        stripes[..., 3] = 1.0
        stripes[::2, :, :3] = 1.0
        u = np.full((H, W), 4.0)
        zero = np.zeros_like(u)
        hard = Evaluator._vector_blur(stripes, u, zero, blur_params(vector_scale=1.0, samples=3, max_length=0.0))
        soft = Evaluator._vector_blur(stripes, u, zero, blur_params(vector_scale=1.0, samples=3, max_length=0.0, soft_lines=1))
        np.testing.assert_allclose(hard[2:-2, 8:-8, 0], stripes[2:-2, 8:-8, 0], atol=1e-6)         # horizontal stripes along a horizontal blur
        np.testing.assert_allclose(soft[2:-2, 8:-8, 0], 0.5, atol=1e-6)                           # one pixel either side at a quarter: 0.5 everywhere

    def test_blur_inside_blurs_where_nothing_moves_when_soft_lines_are_on(self):
        picture = np.zeros((H, W, 4), np.float32)
        picture[..., 3] = 1.0
        picture[6, 20, :3] = 1.0
        zero = np.zeros((H, W))
        still = Evaluator._vector_blur(picture, zero, zero, blur_params(vector_scale=1.0, samples=3, max_length=0.0, soft_lines=1, blur_inside=0))
        inside = Evaluator._vector_blur(picture, zero, zero, blur_params(vector_scale=1.0, samples=3, max_length=0.0, soft_lines=1, blur_inside=1))
        hard = Evaluator._vector_blur(picture, zero, zero, blur_params(vector_scale=1.0, samples=3, max_length=0.0, soft_lines=0, blur_inside=1))
        np.testing.assert_allclose(still[..., 0], picture[..., 0], atol=1e-6)
        np.testing.assert_allclose(hard[..., 0], picture[..., 0], atol=1e-6)                      # blur inside waits for soft lines
        np.testing.assert_allclose(inside[5:8, 19:22, 0], 1.0 / 9.0, atol=1e-6)

    def test_a_saved_node_keeps_its_picture(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=W, height=H, size=3))
        vector_plate(g, 5.0)
        g.add("node", "VectorBlur", {}, image="plate", uv="vec")
        old = copy.deepcopy(g.doc)
        for name in list(vectorblurknobs.NEW):
            old["nodes"]["node"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual((params["vector_blur_type"], params["blur_uv"], params["uv_offset"], params["grow_bbox"], params["soft_lines"]),
                         ("uniform", "none", 0.0, 0, 0))
        plate = evaluator_pixels(loaded, "plate")
        older = {k: v for k, v in params.items() if k not in vectorblurknobs.NEW}
        expected = Evaluator._vector_blur(plate, np.full((H, W), 5.0), np.zeros((H, W)), older)
        np.testing.assert_array_equal(evaluator_pixels(loaded, "node"), expected)

    def test_the_main_tab_follows_the_guides_order(self):
        names = labels("VectorBlur")
        for first, second in (("Output", "Blur UV"), ("Blur UV", "Invert UV"), ("Invert UV", "UV offset"), ("UV offset", "UV alpha"),
                              ("UV alpha", "Blur type"), ("Blur type", "Scale"), ("Scale", "Motion falloff"), ("Motion falloff", "Offset"),
                              ("Offset", "Expand blur bbox"), ("Expand blur bbox", "Soft lines"), ("Soft lines", "Blur inside")):
            self.assertLess(names.index(first), names.index(second), (first, second))
        kept, total = layout_order("VectorBlur")[0]
        self.assertEqual(kept, total)


if __name__ == "__main__":
    unittest.main()
