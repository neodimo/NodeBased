"""2D behaviour, lane 9 step L: Denoise, MotionBlur and VectorBlur take the knobs and inputs Nuke's Reference Guide lists.

* Denoise: Denoise Amount lowers the residual variance of a plate with noise of a known amplitude; Output Noise is the source minus the
  result (so the result and the noise rebuild the source); Roll Off, Smoothness, Luminance Blend, Preserve Edges, Lift Blacks, Source,
  Noise Model, Profile and the Noise input each change the picture; graphs saved before keep their pictures (legacy_denoise); the tile
  path equals the evaluator, with a mask;
* VectorBlur: on a plate of constant vectors the streak is the vector times Scale, Offset slides it, Method points it back, UV alpha keeps
  the matte from smearing, Invert UV flips the vectors and Output UVs hands them back;
* MotionBlur: with FgVecs wired a plate moving a whole number of pixels per frame is streaked over Shutter Time times that many pixels,
  at Shutter Samples points, centred on the frame by default.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import bilateralknobs, denoiseknobs, knobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, texture, tile_pixels
from tests.test_2d_behaviour_filter_f import layout_order

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402


def labels(kind):
    return [g.label for g in knobs.KNOB_LAYOUT[kind] if not g.params[0].startswith("legacy_") and g.params != ("mix",)]


def denoise_cases():
    return [c for c in ruler.CASES if c.nuke == "Denoise" and c.id.startswith("denoise_")]


def noisy_flat(level=0.5, amount=0.03, h=40, w=40, seed=3):
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = level + rng.normal(0, amount, (h, w, 3))
    img[..., 3] = 1.0
    return img


def denoise(img, noise=None, **knob):
    return Evaluator._denoise(img, {**SPECS["Denoise"]["params"], **knob}, noise)


def residual(img, level=0.5):
    return float(np.var(img[..., :3].astype(np.float64) - level))


class DenoiseCaseTests(unittest.TestCase):
    def test_every_scored_case_passes(self):
        scored = [c for c in denoise_cases() if not c.unscored]
        self.assertGreaterEqual(len(scored), 3)
        for case in scored:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], case.statement))

    def test_every_unscored_case_states_its_reason(self):
        unscored = [c for c in denoise_cases() if c.unscored]
        self.assertGreaterEqual(len(unscored), 1)
        for case in unscored:
            with self.subTest(case=case.id):
                self.assertGreater(len(case.unscored), 40)
                self.assertEqual(audit.run_case(case)["status"], "not scored")


class DenoiseBehaviourTests(unittest.TestCase):
    def test_a_higher_amount_lowers_the_residual_variance(self):
        noisy = noisy_flat()
        variances = [residual(denoise(noisy, denoise_strength=a)) for a in (0.0, 0.05, 0.2, 0.6)]
        self.assertAlmostEqual(variances[0], residual(noisy), places=8)
        self.assertTrue(all(b < a for a, b in zip(variances, variances[1:])), variances)
        self.assertLess(variances[-1], 0.2 * variances[0])

    def test_amount_0_removes_nothing(self):
        picture = texture(20, 24)
        np.testing.assert_array_equal(denoise(picture, denoise_strength=0.0, denoise_lumablend=1.0), picture)

    def test_output_noise_is_the_source_minus_the_result(self):
        picture = texture(20, 24)
        for knob in ({}, {"denoise_strength": 0.5, "denoise_preserve_edges": 1}, {"denoise_lift_blacks": 1, "denoise_lumablend": 0.2}):
            with self.subTest(knob=knob):
                result = denoise(picture, **knob)
                noise = denoise(picture, denoise_output="Noise", **knob)
                np.testing.assert_array_equal(picture - result, noise)
                np.testing.assert_allclose(result + noise, picture, atol=1e-6)

    def test_the_noise_output_holds_noise_and_little_picture(self):
        picture = noisy_flat(amount=0.03)
        picture[:, 20:, :3] += 0.4                                               # an edge the filter must keep out of the noise
        noise = denoise(picture, denoise_output="Noise", denoise_strength=0.1)
        edge_energy = float(np.abs(noise[:, 18:22, :3]).mean())
        flat_energy = float(np.abs(noise[:, 2:14, :3]).mean())
        self.assertLess(edge_energy, 3.0 * flat_energy)

    def test_roll_off_1_is_a_hard_threshold_and_softens_above(self):
        picture = texture(24, 28)
        hard, soft, softer = (denoise(picture, denoise_strength=0.3, denoise_rolloff=r) for r in (1.0, 2.0, 4.0))
        self.assertTrue(differs(hard, soft))
        self.assertTrue(differs(soft, softer))
        # a hard threshold never lets in a pixel beyond the amount: the values stay inside the window's range
        isolated = np.zeros((9, 9, 4), np.float32)
        isolated[..., 3] = 1.0
        isolated[4, 4, :3] = 1.0
        kept = denoise(isolated, denoise_strength=0.3, denoise_rolloff=1.0, denoise_model="Constant")
        np.testing.assert_array_equal(kept[4, 4, :3], isolated[4, 4, :3])        # a lone bright pixel stays when the threshold is below its step

    def test_smoothness_scales_the_window(self):
        noisy = noisy_flat()
        none, normal, wide = (residual(denoise(noisy, denoise_strength=0.2, denoise_smoothness=s)) for s in (0.0, 1.0, 2.0))
        self.assertAlmostEqual(none, residual(noisy), places=8)
        self.assertLess(normal, none)
        self.assertLess(wide, normal)

    def test_luminance_blend_runs_from_the_original_to_the_denoised_luminance(self):
        picture = texture(20, 24)
        picture[..., 3] = 1.0
        full = denoise(picture, denoise_strength=0.4, denoise_lumablend=1.0)
        none = denoise(picture, denoise_strength=0.4, denoise_lumablend=0.0)
        half = denoise(picture, denoise_strength=0.4, denoise_lumablend=0.5)
        np.testing.assert_allclose(denoiseknobs.luminance(none), denoiseknobs.luminance(picture), atol=1e-5)
        np.testing.assert_allclose(denoiseknobs.luminance(half), 0.5 * (denoiseknobs.luminance(picture) + denoiseknobs.luminance(full)), atol=1e-5)
        np.testing.assert_allclose(denoise(picture, denoise_strength=0.4, denoise_lumablend=1.0), full, atol=0)

    def test_preserve_edges_sharpens_a_step(self):
        step = np.zeros((10, 16, 4), np.float32)
        step[:, 8:, :3] = 0.5
        step[..., 3] = 1.0
        plain = denoise(step, denoise_strength=0.8, denoise_preserve_edges=0, denoise_lumablend=1.0)
        sharp = denoise(step, denoise_strength=0.8, denoise_preserve_edges=1, denoise_lumablend=1.0)
        self.assertTrue(differs(plain, sharp))
        self.assertGreater(float(sharp[5, 8, 0]) - float(sharp[5, 7, 0]), float(plain[5, 8, 0]) - float(plain[5, 7, 0]))

    def test_lift_blacks_lifts_sub_blacks_to_0(self):
        picture = texture(20, 24)
        self.assertLess(float(denoise(picture, denoise_lift_blacks=0)[..., :3].min()), 0.0)
        lifted = denoise(picture, denoise_lift_blacks=1)
        self.assertGreaterEqual(float(lifted[..., :3].min()), 0.0)

    def test_the_model_and_the_source_set_where_the_threshold_is_larger(self):
        picture = np.zeros((40, 40, 4), np.float32)
        picture[..., 3] = 1.0
        rng = np.random.RandomState(5)
        picture[:, :20, :3] = 0.15 + rng.normal(0, 0.035, (40, 20, 3))           # a dark half
        picture[:, 20:, :3] = 0.85 + rng.normal(0, 0.035, (40, 20, 3))           # a light half, the same noise

        def removed(**knob):
            out = denoise(picture, denoise_strength=0.12, **knob)
            dark = 1.0 - float(np.var(out[:, 3:17, :3] - 0.15)) / float(np.var(picture[:, 3:17, :3] - 0.15))
            light = 1.0 - float(np.var(out[:, 23:37, :3] - 0.85)) / float(np.var(picture[:, 23:37, :3] - 0.85))
            return dark, light

        d_dark, d_light = removed(denoise_model="Modulated", denoise_type="Digital")
        f_dark, f_light = removed(denoise_model="Modulated", denoise_type="Film")
        c_dark, c_light = removed(denoise_model="Constant")
        self.assertGreater(d_light, d_dark)                                      # digital noise lives in the light areas
        self.assertGreater(f_dark, f_light)                                      # film noise in the dark ones
        self.assertLess(abs(c_dark - c_light), 0.5 * abs(d_light - d_dark))      # Constant treats both halves alike

    def test_the_profile_measures_the_level_per_brightness_band(self):
        picture = np.zeros((40, 40, 4), np.float32)
        picture[..., 3] = 1.0
        rng = np.random.RandomState(9)
        picture[:, :20, :3] = 0.15 + rng.normal(0, 0.05, (40, 20, 3))
        picture[:, 20:, :3] = 0.85 + rng.normal(0, 0.01, (40, 20, 3))
        centres, levels = denoiseknobs.noise_profile(picture)
        self.assertGreater(levels[1], 3.0 * levels[6])                           # the dark band is the noisy one
        self.assertTrue(differs(denoise(picture, denoise_profile="Automatic"), denoise(picture, denoise_profile="Constant")))

    def test_the_noise_level_estimate_tracks_the_noise_amplitude(self):
        for amplitude in (0.01, 0.03, 0.08):
            with self.subTest(amplitude=amplitude):
                level = denoiseknobs.noise_level(noisy_flat(amount=amplitude, h=64, w=64))
                self.assertAlmostEqual(level, amplitude * 0.749, delta=0.15 * amplitude)

    def test_a_noisier_noise_input_removes_more(self):
        noisy = noisy_flat(amount=0.05)
        quiet, loud = noisy_flat(amount=0.005, seed=11), noisy_flat(amount=0.08, seed=12)
        with_quiet = denoise(noisy, quiet, denoise_strength=0.4, denoise_model="Constant")
        with_loud = denoise(noisy, loud, denoise_strength=0.4, denoise_model="Constant")
        self.assertLess(residual(with_loud), residual(with_quiet))
        self.assertTrue(differs(with_loud, denoise(noisy, None, denoise_strength=0.4, denoise_model="Constant")))

    def test_every_new_knob_changes_the_picture(self):
        picture = texture(24, 28)
        base = denoise(picture, denoise_strength=0.3)
        for knob, value in (("denoise_type", "Film"), ("denoise_model", "Constant"), ("denoise_profile", "Automatic"), ("denoise_output", "Noise"),
                            ("denoise_lift_blacks", 1), ("denoise_rolloff", 1.0), ("denoise_smoothness", 0.5), ("denoise_lumablend", 1.0),
                            ("denoise_preserve_edges", 1)):
            with self.subTest(knob=knob):
                self.assertTrue(differs(base, denoise(picture, denoise_strength=0.3, **{knob: value})))


class DenoiseGraphTests(unittest.TestCase):
    def graph(self, masked=False, noise=False, **params):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("shade", "Grade", dict(exposure=-0.5), image="plate")
        inputs = {"image": "shade"}
        if masked:
            g.add("matte", "Constant", dict(width=45, height=37, red=1, green=1, blue=1, alpha=0.5))
            inputs["mask"] = "matte"
        if noise:
            g.add("sample", "Checker", dict(width=45, height=37, size=9))
            inputs["noise"] = "sample"
        g.add("node", "Denoise", {"denoise_strength": 0.3, **params}, **inputs)
        return g

    def test_the_tile_path_equals_the_evaluator(self):
        for masked in (False, True):
            for params in ({}, {"denoise_preserve_edges": 1, "denoise_lumablend": 0.3}, {"denoise_output": "Noise", "denoise_rolloff": 1.0}):
                doc = self.graph(masked, **params).doc
                full = evaluator_pixels(doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(masked=masked, params=params, edge=edge):
                        np.testing.assert_allclose(tile_pixels(doc, "node", edge), full, atol=1e-6)

    def test_a_measured_noise_level_runs_on_the_full_frame(self):
        from nodebased.tileexec import TileExecutor
        for params, noise in (({"denoise_profile": "Automatic"}, False), ({}, True)):
            doc = dict(self.graph(False, noise, **params).doc, view="node")
            self.assertFalse(TileExecutor(evaluator=Evaluator(), tile_edge=16).supports_tiled(doc, "node"))
        doc = dict(self.graph(False, False).doc, view="node")
        self.assertTrue(TileExecutor(evaluator=Evaluator(), tile_edge=16).supports_tiled(doc, "node"))

    def test_wiring_the_noise_input_changes_the_picture(self):
        plain = evaluator_pixels(self.graph(False, False, denoise_model="Constant").doc, "node")
        wired = evaluator_pixels(self.graph(False, True, denoise_model="Constant").doc, "node")
        self.assertTrue(differs(plain, wired))

    def test_a_saved_graph_keeps_its_picture(self):
        g = self.graph()
        old = copy.deepcopy(g.doc)
        for name in list(denoiseknobs.NEW):
            old["nodes"]["node"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual(params["legacy_denoise"], 1)
        source = evaluator_pixels(loaded, "shade")
        strength = params["denoise_strength"]
        expected = bilateralknobs.bilateral(source, {"spatial_size": 2.0 + 5.0 * strength, "colour_sigma": max(0.01, strength)})
        np.testing.assert_array_equal(evaluator_pixels(loaded, "node"), expected)

    def test_a_new_node_is_not_legacy(self):
        self.assertEqual(SPECS["Denoise"]["params"]["legacy_denoise"], 0)
        self.assertIn("noise", SPECS["Denoise"]["optional_inputs"])


class DenoiseLayoutTests(unittest.TestCase):
    def test_the_main_tab_follows_the_guide(self):
        self.assertEqual(labels("Denoise")[:10], ["Source", "Noise model", "Lift blacks", "Profile", "Output", "Denoise amount", "Roll off",
                                                  "Smoothness", "Luminance blend", "Preserve edges"])
        kept, total = layout_order("Denoise")[0]
        self.assertEqual(kept, total)


# ------------------------------------------------------------------------------------------------------ VectorBlur and MotionBlur
W, H = 40, 12


def ramp_plate():
    """A horizontal ramp, opaque: a linear picture, so a symmetric streak leaves it alone and a shifted one moves it by the shift."""
    g = Graph()
    g.add("plate", "Ramp", dict(width=W, height=H, p0_x=0.0, p0_y=0.0, p1_x=float(W - 1), p1_y=0.0,
                                color0_red=0.0, color0_green=0.0, color0_blue=0.0, color0_alpha=1.0,
                                color1_red=1.0, color1_green=1.0, color1_blue=1.0, color1_alpha=1.0))
    return g


def vector_plate(g, u, v=0.0):
    g.add("vec", "Constant", dict(width=W, height=H, red=u, green=v, blue=0.0, alpha=1.0))


def interior(a, margin=8):
    return np.asarray(a, np.float64)[:, margin:W - margin, :3]


class VectorBlurConstantVectorTests(unittest.TestCase):
    def blurred(self, u=4.0, v=0.0, **knob):
        g = ramp_plate()
        vector_plate(g, u, v)
        g.add("node", "VectorBlur", dict(knob), image="plate", uv="vec")
        return evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate")

    def slope(self, plate):
        return float(plate[5, 21, 0] - plate[5, 20, 0])

    def test_the_streak_is_the_vector_times_the_scale(self):
        for scale, shift in ((1.0, 2.0), (0.5, 1.0), (2.0, 4.0)):
            with self.subTest(scale=scale):
                out, plate = self.blurred(vector_scale=scale, vector_offset=0.0)
                np.testing.assert_allclose(interior(out), interior(plate) - shift * self.slope(plate), atol=1e-5)

    def test_scale_0_leaves_the_picture(self):
        out, plate = self.blurred(vector_scale=0.0)
        np.testing.assert_allclose(interior(out), interior(plate), atol=1e-6)

    def test_offset_slides_the_streak_and_minus_a_half_centres_it(self):
        centred, plate = self.blurred(vector_scale=1.0, vector_offset=-0.5)
        np.testing.assert_allclose(interior(centred), interior(plate), atol=1e-5)                 # p - 0.5v .. p + 0.5v averages to p
        slid, _ = self.blurred(vector_scale=1.0, vector_offset=0.5)
        np.testing.assert_allclose(interior(slid), interior(plate) - 4.0 * self.slope(plate), atol=1e-5)

    def test_the_method_points_the_streak_the_other_way(self):
        forward, plate = self.blurred(vector_scale=1.0, vector_offset=0.0, vector_method="forward")
        backward, _ = self.blurred(vector_scale=1.0, vector_offset=0.0, vector_method="backward")
        np.testing.assert_allclose(interior(backward), interior(plate) + 2.0 * self.slope(plate), atol=1e-5)
        self.assertTrue(differs(forward, backward))

    def test_invert_uv_flips_the_vectors(self):
        inverted, plate = self.blurred(vector_scale=1.0, vector_offset=0.0, invert_uv=1)
        np.testing.assert_allclose(interior(inverted), interior(plate) + 2.0 * self.slope(plate), atol=1e-5)

    def test_output_uvs_hands_back_the_vectors(self):
        out, _ = self.blurred(u=4.0, v=-1.5, vector_output="uvs")
        np.testing.assert_allclose(out[..., 0], 4.0, atol=1e-6)
        np.testing.assert_allclose(out[..., 1], -1.5, atol=1e-6)
        flipped, _ = self.blurred(u=4.0, v=-1.5, vector_output="uvs", invert_uv=1)
        np.testing.assert_allclose(flipped[..., 0], -4.0, atol=1e-6)

    def test_uv_alpha_weighted_keeps_the_matte_from_smearing(self):
        picture = np.ones((H, W, 4), np.float32)
        picture[:, W // 2:, 3] = 0.0
        picture[:, W // 2:, :3] = 0.0
        u = np.full((H, W), 4.0)
        plain = Evaluator._vector_blur(picture, u, np.zeros_like(u), {**SPECS["VectorBlur"]["params"], "vector_alpha": "none"})
        weighted = Evaluator._vector_blur(picture, u, np.zeros_like(u), {**SPECS["VectorBlur"]["params"], "vector_alpha": "weighted"})
        np.testing.assert_array_equal(weighted[..., 3], picture[..., 3])                          # the matte stays where it was
        self.assertTrue(differs(plain[..., 3], picture[..., 3]))                                   # unweighted, the matte smears
        self.assertTrue(differs(plain, weighted))

    def test_the_main_tab_lists_the_new_knobs_in_the_guides_order(self):
        names = labels("VectorBlur")
        self.assertLess(names.index("Output"), names.index("UV alpha"))
        self.assertIn("Invert UV", names)


class MotionBlurFgVecsTests(unittest.TestCase):
    def graph(self, u=4.0, **knob):
        g = ramp_plate()
        vector_plate(g, u)
        g.add("node", "MotionBlur", dict(knob), image="plate", fgvecs="vec")
        return g

    def run_node(self, **knob):
        g = self.graph(**knob)
        return evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate")

    def slope(self, plate):
        return float(plate[5, 21, 0] - plate[5, 20, 0])

    def test_a_centred_shutter_leaves_a_linear_plate_and_the_streak_is_shutter_time_times_the_motion(self):
        centred, plate = self.run_node(shutter=0.5, samples=3, shutter_offset="centred")
        np.testing.assert_allclose(interior(centred), interior(plate), atol=1e-5)
        start, _ = self.run_node(shutter=0.5, samples=3, shutter_offset="start")
        # 4 px per frame x 0.5 = a 2 px streak from the pixel back along the motion, sampled at p, p-1, p-2
        np.testing.assert_allclose(interior(start), interior(plate) - 1.0 * self.slope(plate), atol=1e-5)
        end, _ = self.run_node(shutter=0.5, samples=3, shutter_offset="end")
        np.testing.assert_allclose(interior(end), interior(plate) + 1.0 * self.slope(plate), atol=1e-5)

    def test_the_shutter_time_scales_the_streak(self):
        half, plate = self.run_node(shutter=0.5, samples=5, shutter_offset="start")
        full, _ = self.run_node(shutter=1.0, samples=5, shutter_offset="start")
        np.testing.assert_allclose(interior(half), interior(plate) - 1.0 * self.slope(plate), atol=1e-5)
        np.testing.assert_allclose(interior(full), interior(plate) - 2.0 * self.slope(plate), atol=1e-5)

    def test_the_samples_are_points_along_the_vector(self):
        # a lone bright pixel under a 4 px streak sampled at 5 points (1 px apart) becomes five pixels at a fifth each
        picture = np.zeros((H, W, 4), np.float32)
        picture[..., 3] = 1.0
        picture[5, 20, :3] = 1.0
        u = np.full((H, W), 4.0)
        from nodebased import motionblur
        p = motionblur.fgvecs_blur_params({**SPECS["MotionBlur"]["params"], "shutter": 1.0, "samples": 5, "shutter_offset": "start"})
        out = Evaluator._vector_blur(picture, u, np.zeros_like(u), p)
        row = out[5, :, 0]
        np.testing.assert_allclose(row[20:25], 0.2, atol=1e-6)                                    # the streak trails the pixel along the motion
        self.assertEqual(float(row[:20].sum() + row[25:].sum()), 0.0)

    def test_more_samples_change_the_picture_of_a_streaked_detail(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=W, height=H, size=3))
        vector_plate(g, 8.0)
        g.add("few", "MotionBlur", dict(shutter=1.0, samples=2), image="plate", fgvecs="vec")
        g.add("many", "MotionBlur", dict(shutter=1.0, samples=9), image="plate", fgvecs="vec")
        self.assertTrue(differs(evaluator_pixels(g.doc, "few"), evaluator_pixels(g.doc, "many")))

    def test_without_fgvecs_the_node_still_estimates_its_motion(self):
        self.assertIn("fgvecs", SPECS["MotionBlur"]["optional_inputs"])
        g = ramp_plate()
        g.add("node", "MotionBlur", dict(shutter=1.0, samples=3), image="plate")
        out = evaluator_pixels(g.doc, "node")
        np.testing.assert_allclose(interior(out), interior(evaluator_pixels(g.doc, "plate")), atol=1e-5)   # a still plate has no motion to blur


if __name__ == "__main__":
    unittest.main()
