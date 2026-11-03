"""2D behaviour, lane 9 step L (finish): the rest of Denoise's tabs.

* Tune Channels: Luminance Gain and Chrominance Gain scale the threshold for the luminance and the colour apart;
* Tune Frequencies: each band can be left alone or taken at its own gain (0 removes none of that band's noise);
* Profile tab: the curve and the three gains scale the threshold by brightness, the Profile Frame feeds the Automatic profile;
* Noise Analysis: the analysis box (y from the bottom, 80 by 80 pixels at least), the analysis frame, Lock, Analyze Noise, Export and Import;
* Temporal Processing and the Motion input: neighbours moved onto the frame along the vectors are blended, the Frame Blending says how much
  difference is allowed;
* the tile path equals the evaluator with the bands on, and a saved graph keeps its picture.
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import denoiseknobs, knobs
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs, evaluator_pixels, tile_pixels
from tests.test_2d_behaviour_filter_f import layout_order
from tests.test_2d_behaviour_filter_l import labels

LUMA = np.array([0.2126, 0.7152, 0.0722])
# a base that isolates one tab at a time: no luminance blend, every band on, the Constant model, no Modulated curve
BASE = {"denoise_lumablend": 1.0, "denoise_process_very_low": 1, "denoise_model": "Constant", "denoise_strength": 0.5}


def run(img, noise=None, refs=None, **knob):
    return denoiseknobs.denoise(img, {**SPECS["Denoise"]["params"], **BASE, **knob}, noise, refs)


def luma_of(img):
    return img[..., :3].astype(np.float64) @ LUMA


def chroma_of(img):
    return img[..., :3].astype(np.float64) - luma_of(img)[..., None]


def flat(level=0.5, h=48, w=48):
    img = np.zeros((h, w, 4), np.float32)
    img[..., :3] = level
    img[..., 3] = 1.0
    return img


def luma_noise(amount=0.03, seed=3, **kw):
    img = flat(**kw)
    img[..., :3] += np.random.RandomState(seed).normal(0, amount, img.shape[:2])[..., None]
    return img


def chroma_noise(amount=0.03, seed=4, **kw):
    """Noise that leaves the luminance alone: a colour difference with zero luminance."""
    img = flat(**kw)
    n = np.random.RandomState(seed).normal(0, amount, img.shape[:2])
    direction = np.array([1.0, -LUMA[0] / LUMA[1], 0.0])
    img[..., :3] += n[..., None] * direction
    return img


def spread(a):
    return float(np.var(np.asarray(a, np.float64)))


class TuneChannelsTests(unittest.TestCase):
    def test_luminance_gain_scales_the_threshold_for_the_luminance(self):
        img = luma_noise()
        low, high = run(img, denoise_luma_gain=0.3), run(img, denoise_luma_gain=3.0)
        self.assertLess(spread(luma_of(high)), spread(luma_of(low)))
        self.assertLess(spread(luma_of(low)), spread(luma_of(img)))

    def test_chrominance_gain_scales_the_threshold_for_the_colour(self):
        img = chroma_noise()
        low, high = run(img, denoise_chroma_gain=0.3, denoise_strength=0.03), run(img, denoise_chroma_gain=5.0, denoise_strength=0.03)
        self.assertLess(spread(chroma_of(high)), 0.5 * spread(chroma_of(low)))

    def test_each_gain_leaves_the_other_channel_alone(self):
        img = luma_noise()
        a, b = run(img, denoise_chroma_gain=0.3), run(img, denoise_chroma_gain=8.0)
        np.testing.assert_allclose(luma_of(a), luma_of(b), atol=1e-6)               # a luminance-only noise: the colour gain cannot touch it
        c, d = run(chroma_noise(), denoise_luma_gain=0.3), run(chroma_noise(), denoise_luma_gain=8.0)
        np.testing.assert_allclose(chroma_of(c), chroma_of(d), atol=1e-6)

    def test_a_gain_of_0_removes_none_of_that_channels_noise(self):
        img = luma_noise()
        out = run(img, denoise_luma_gain=0.0)
        np.testing.assert_allclose(luma_of(out), luma_of(img), atol=1e-5)
        img = chroma_noise()
        np.testing.assert_allclose(chroma_of(run(img, denoise_chroma_gain=0.0)), chroma_of(img), atol=1e-5)

    def test_the_defaults_are_1_and_5(self):
        params = SPECS["Denoise"]["params"]
        self.assertEqual((params["denoise_luma_gain"], params["denoise_chroma_gain"]), (1.0, 5.0))


class TuneFrequenciesTests(unittest.TestCase):
    def high_pass(self, img):
        return luma_of(img) - denoiseknobs._gauss(luma_of(img), 1.0)

    def test_the_bands_sum_to_the_noise(self):
        noise = np.random.RandomState(1).normal(0, 1, (30, 40, 4))
        np.testing.assert_allclose(sum(denoiseknobs.split_bands(noise)), noise, atol=1e-12)

    def test_every_band_at_gain_1_is_the_untuned_picture(self):
        img = luma_noise()
        tuned = run(img, denoise_process_very_low=1)
        untuned = denoiseknobs._channel_average(img, {**SPECS["Denoise"]["params"], **BASE},
                                                denoiseknobs.threshold_map(img, {**SPECS["Denoise"]["params"], **BASE})).astype(np.float32)
        np.testing.assert_allclose(tuned, untuned, atol=1e-6)

    def test_a_band_left_unprocessed_keeps_its_noise(self):
        img = luma_noise()
        everything = run(img)
        no_high = run(img, denoise_process_high=0)
        self.assertGreater(spread(self.high_pass(no_high)), 3 * spread(self.high_pass(everything)))
        np.testing.assert_allclose(run(img, denoise_high_gain=0.0), no_high, atol=1e-6)         # gain 0 is the same as not processing

    def test_a_higher_gain_removes_more_of_that_band(self):
        img = luma_noise(0.06)
        gentle, strong = run(img, denoise_high_gain=0.2, denoise_strength=0.2), run(img, denoise_high_gain=3.0, denoise_strength=0.2)
        self.assertLess(spread(self.high_pass(strong)), spread(self.high_pass(gentle)))

    def test_each_band_has_its_own_switch_and_gain(self):
        img = luma_noise(0.05)
        base = run(img)
        for band in denoiseknobs.FREQUENCY_BANDS:
            with self.subTest(band=band):
                self.assertTrue(differs(base, run(img, **{f"denoise_process_{band}": 0}), 1e-4), f"switching {band} off changed nothing")
                self.assertTrue(differs(base, run(img, **{f"denoise_{band}_gain": 0.0}), 1e-4), f"{band} gain 0 changed nothing")

    def test_very_low_is_off_by_default_and_moves_the_picture_when_on(self):
        self.assertEqual(SPECS["Denoise"]["params"]["denoise_process_very_low"], 0)
        img = luma_noise(0.05)
        params = {**SPECS["Denoise"]["params"], "denoise_lumablend": 1.0}
        off = denoiseknobs.denoise(img, {**params, "denoise_process_very_low": 0})
        on = denoiseknobs.denoise(img, {**params, "denoise_process_very_low": 1})
        self.assertTrue(differs(off, on, 1e-7))


class ProfileTabTests(unittest.TestCase):
    def two_tone(self):
        """Dark left, bright right, the same noise on both."""
        img = flat()
        img[:, :24, :3] = 0.0
        img[:, 24:, :3] = 1.0
        img[..., :3] += np.random.RandomState(5).normal(0, 0.03, img.shape[:2])[..., None]
        return img

    def test_the_profile_gains_scale_the_threshold_by_brightness(self):
        img = self.two_tone()
        out = run(img, denoise_tune_profile=1, denoise_profile_low_gain=0.0, denoise_profile_high_gain=2.0)
        dark, bright = (slice(None), slice(4, 20)), (slice(None), slice(28, 44))
        kept = lambda part: spread(luma_of(out)[part]) / spread(luma_of(img)[part])
        self.assertGreater(kept(dark), 0.5)                       # gain 0 at black: the noise near black (the darkest values) mostly stays
        self.assertLess(kept(bright), 0.2)                        # gain 2 at white: the noise goes

    def test_the_gains_wait_for_tune_profile(self):
        img = self.two_tone()
        np.testing.assert_array_equal(run(img, denoise_tune_profile=0, denoise_profile_low_gain=0.0), run(img))
        self.assertTrue(differs(run(img, denoise_tune_profile=1, denoise_profile_low_gain=0.0), run(img)))

    def test_the_curve_is_the_relative_amount_removed_by_intensity(self):
        img = self.two_tone()
        curve = '{"interpolation":"linear","points":[[0,0],[0.45,0],[0.55,1],[1,1]]}'
        out = run(img, denoise_profile_curve=curve)
        dark, bright = (slice(None), slice(4, 20)), (slice(None), slice(28, 44))
        self.assertAlmostEqual(spread(luma_of(out)[dark]), spread(luma_of(img)[dark]), delta=1e-5)
        self.assertLess(spread(luma_of(out)[bright]), 0.5 * spread(luma_of(img)[bright]))

    def test_the_profile_frame_feeds_the_automatic_profile(self):
        img = luma_noise(0.02)
        quiet, loud = luma_noise(0.001, seed=9), luma_noise(0.15, seed=9)
        own = run(img, denoise_profile="Automatic")
        with_quiet = run(img, None, {"profile": quiet}, denoise_profile="Automatic")
        with_loud = run(img, None, {"profile": loud}, denoise_profile="Automatic")
        self.assertLess(spread(luma_of(with_loud)), spread(luma_of(own)))
        self.assertLess(spread(luma_of(own)), spread(luma_of(with_quiet)))
        self.assertTrue(denoiseknobs.wants_profile_frame({**SPECS["Denoise"]["params"], "denoise_profile": "Automatic"}))
        self.assertFalse(denoiseknobs.wants_profile_frame(SPECS["Denoise"]["params"]))


class NoiseAnalysisTests(unittest.TestCase):
    def two_zone(self, h=200, w=200):
        """The top half nearly clean, the bottom half noisy."""
        img = flat(h=h, w=w)
        rng = np.random.RandomState(6)
        img[: h // 2, :, :3] += rng.normal(0, 0.002, (h // 2, w))[..., None]
        img[h // 2:, :, :3] += rng.normal(0, 0.08, (h - h // 2, w))[..., None]
        return img

    def p(self, **k):
        return {**SPECS["Denoise"]["params"], **BASE, **k}

    def test_the_box_counts_y_from_the_bottom(self):
        img = self.two_zone()
        bottom = denoiseknobs.threshold_map(img, self.p(denoise_analysis_x=0, denoise_analysis_y=0, denoise_analysis_r=200, denoise_analysis_t=100))
        top = denoiseknobs.threshold_map(img, self.p(denoise_analysis_x=0, denoise_analysis_y=100, denoise_analysis_r=200, denoise_analysis_t=200))
        self.assertGreater(float(bottom.mean()), 5 * float(top.mean()))                  # the bottom rows are the noisy ones

    def test_no_box_is_the_nominal_level_and_a_box_measures_the_picture(self):
        img = self.two_zone()
        nominal = denoiseknobs.threshold_map(img, self.p())
        boxed = denoiseknobs.threshold_map(img, self.p(denoise_analysis_x=0, denoise_analysis_y=100, denoise_analysis_r=200, denoise_analysis_t=200))
        self.assertTrue(differs(nominal, boxed, 1e-4))

    def test_a_box_under_80_pixels_removes_no_noise(self):
        img = luma_noise()
        for box in (dict(denoise_analysis_r=79, denoise_analysis_t=80), dict(denoise_analysis_r=80, denoise_analysis_t=79)):
            small = run(img, **box)
            np.testing.assert_array_equal(small, img)
            np.testing.assert_array_equal(run(img, denoise_output="Noise", **box), np.zeros_like(img))
        big = luma_noise(h=100, w=100)
        self.assertTrue(differs(run(big, denoise_analysis_r=80, denoise_analysis_t=80), big, 1e-4))

    def test_the_box_is_measured_on_the_noise_input_when_one_is_wired(self):
        img = self.two_zone()
        loud = self.two_zone()
        loud[:, :, :3] += np.random.RandomState(8).normal(0, 0.2, (200, 200))[..., None]
        box = dict(denoise_analysis_x=0, denoise_analysis_y=100, denoise_analysis_r=200, denoise_analysis_t=200)
        own = denoiseknobs.threshold_map(img, self.p(**box))
        measured_elsewhere = denoiseknobs.threshold_map(img, self.p(**box), noise=loud)
        self.assertGreater(float(measured_elsewhere.mean()), 3 * float(own.mean()))

    def test_the_analysis_frame_picture_is_what_the_box_measures(self):
        img = self.two_zone()
        box = dict(denoise_analysis_x=0, denoise_analysis_y=100, denoise_analysis_r=200, denoise_analysis_t=200)
        here = denoiseknobs.threshold_map(img, self.p(**box))
        elsewhere = denoiseknobs.threshold_map(img, self.p(**box), refs={"analysis": self.two_zone()[::-1].copy()})
        self.assertGreater(float(elsewhere.mean()), 5 * float(here.mean()))
        self.assertTrue(denoiseknobs.wants_analysis_frame(self.p(**box)))
        self.assertFalse(denoiseknobs.wants_analysis_frame(self.p()))
        self.assertFalse(denoiseknobs.wants_analysis_frame(self.p(denoise_analysis_r=40, denoise_analysis_t=40)))      # too small to analyze

    def test_lock_uses_the_held_analysis_and_measures_nothing(self):
        img = self.two_zone()
        held = self.p(denoise_lock_analysis=1, denoise_analysis_level=0.25)
        thresh = denoiseknobs.threshold_map(img, held)
        np.testing.assert_allclose(thresh, 3 * 0.5 * 0.25, atol=1e-12)                      # 3 x level x amount, whatever the picture holds
        np.testing.assert_allclose(denoiseknobs.threshold_map(luma_noise(), held), thresh[0, 0], atol=1e-12)
        unlocked_nothing_held = self.p(denoise_lock_analysis=1)
        self.assertFalse(denoiseknobs.is_locked(unlocked_nothing_held))                    # locked with nothing held reads the picture
        profile = json.dumps([0.1] * 8)
        automatic = self.p(denoise_lock_analysis=1, denoise_profile="Automatic", denoise_analysis_profile=profile)
        np.testing.assert_allclose(denoiseknobs.threshold_map(img, automatic), 3 * 0.5 * 0.1, atol=1e-9)

    def test_analyze_export_and_import_round_trip(self):
        img = self.two_zone()
        p = self.p(denoise_analysis_x=0, denoise_analysis_y=0, denoise_analysis_r=200, denoise_analysis_t=100)
        found = denoiseknobs.analyze(img, img, p)
        self.assertAlmostEqual(found["level"], denoiseknobs.noise_level(img[100:]), places=9)
        self.assertEqual(len(found["profile"]), denoiseknobs.BANDS)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "analysis.json")
            denoiseknobs.export_analysis(path, found, automatic=False)
            self.assertNotIn("profile", json.load(open(path)))                              # a Constant profile saves the analysis only
            back = denoiseknobs.import_analysis(path)
            self.assertAlmostEqual(back["level"], found["level"], places=12)
            self.assertIsNone(back["profile"])
            denoiseknobs.export_analysis(path, found, automatic=True)
            back = denoiseknobs.import_analysis(path)
            np.testing.assert_allclose(back["profile"], found["profile"])
            params = denoiseknobs.imported_params(back)
            self.assertEqual(params["denoise_lock_analysis"], 1)                            # the guide: Import locks the analysis
            self.assertTrue(denoiseknobs.is_locked({**SPECS["Denoise"]["params"], **params}))
            with open(os.path.join(folder, "other.txt"), "w") as handle:
                handle.write("{}")
            with self.assertRaises(ValueError):
                denoiseknobs.import_analysis(os.path.join(folder, "other.txt"))

    def test_analyze_noise_on_a_node_holds_and_locks_the_analysis(self):
        g = Graph()
        g.add("grain", "Noise", dict(width=100, height=100, size=2, octaves=1))
        g.add("node", "Denoise", dict(denoise_analysis_x=0, denoise_analysis_y=0, denoise_analysis_r=100, denoise_analysis_t=100), image="grain")
        updates = denoiseknobs.analyze_node(Evaluator(), g.doc, "node", 1)
        self.assertEqual(updates["denoise_lock_analysis"], 1)
        self.assertGreater(updates["denoise_analysis_level"], 0.0)
        self.assertEqual(len(json.loads(updates["denoise_analysis_profile"])), denoiseknobs.BANDS)
        held = {**SPECS["Denoise"]["params"], **updates}
        self.assertTrue(denoiseknobs.is_locked(held))


class TemporalTests(unittest.TestCase):
    def ramp(self, shift=0.0, w=64, h=32, slope=0.1):
        x = (np.arange(w, dtype=np.float32)[None, :] - shift) * slope
        img = np.zeros((h, w, 4), np.float32)
        img[..., :3] = np.broadcast_to(x[..., None], (h, w, 3))
        img[..., 3] = 1.0
        return img

    def noisy(self, img, amount, seed):
        out = img.copy()
        out[..., :3] += np.random.RandomState(seed).normal(0, amount, img.shape[:2])[..., None]
        return out

    def test_aligned_neighbours_average_the_noise_away(self):
        clean = self.ramp()
        current = self.noisy(clean, 0.03, 1)
        before, after = self.noisy(self.ramp(-3.0), 0.03, 2), self.noisy(self.ramp(3.0), 0.03, 3)
        motion = np.zeros(clean.shape[:2] + (2,), np.float32)
        motion[..., 0] = 3.0                                               # this frame to the next: three pixels right
        p = {**SPECS["Denoise"]["params"], "denoise_frame_blending": 0.2}
        blended = denoiseknobs.blend_frames(current, [(-1, before), (1, after)], p, motion)
        inner = (slice(None), slice(8, 56))
        self.assertLess(spread(blended[inner][..., 0] - clean[inner][..., 0]), 0.6 * spread(current[inner][..., 0] - clean[inner][..., 0]))

    def test_neighbours_the_vectors_leave_out_of_place_are_not_blended(self):
        clean = self.ramp()
        current = self.noisy(clean, 0.03, 1)
        before, after = self.noisy(self.ramp(-6.0), 0.03, 2), self.noisy(self.ramp(6.0), 0.03, 3)
        wrong = np.zeros(clean.shape[:2] + (2,), np.float32)               # no motion: the ramp has moved 6 pixels, 0.6 apart, far beyond twice 0.1
        p = {**SPECS["Denoise"]["params"], "denoise_frame_blending": 0.1}
        blended = denoiseknobs.blend_frames(current, [(-1, before), (1, after)], p, wrong)
        inner = (slice(None), slice(8, 56))
        np.testing.assert_allclose(blended[inner], current[inner], atol=1e-6)

    def test_a_larger_frame_blending_lets_more_regions_blend(self):
        clean = self.ramp()
        current = self.noisy(clean, 0.03, 1)
        after = self.noisy(self.ramp(3.0), 0.03, 3)
        wrong = np.zeros(clean.shape[:2] + (2,), np.float32)
        tight = denoiseknobs.blend_frames(current, [(1, after)], {**SPECS["Denoise"]["params"], "denoise_frame_blending": 0.05}, wrong)
        loose = denoiseknobs.blend_frames(current, [(1, after)], {**SPECS["Denoise"]["params"], "denoise_frame_blending": 0.6}, wrong)
        inner = (slice(None), slice(8, 56))
        self.assertLess(float(np.abs(tight[inner] - current[inner]).max()), float(np.abs(loose[inner] - current[inner]).max()))

    def test_without_a_motion_input_the_vectors_are_estimated(self):
        clean = self.ramp()
        current = self.noisy(clean, 0.02, 1)
        after = self.noisy(self.ramp(3.0), 0.02, 3)
        p = {**SPECS["Denoise"]["params"], "denoise_frame_blending": 0.3, "flow_backend": "cpu"}
        estimated = denoiseknobs.blend_frames(current, [(1, after)], p, None)
        self.assertTrue(differs(estimated, current, 1e-4))

    def test_temporal_is_off_until_enabled_and_counts_frames_each_way(self):
        p = {**SPECS["Denoise"]["params"], "denoise_frames": 3}
        self.assertEqual(denoiseknobs.temporal_frames(p), 0)
        self.assertEqual(denoiseknobs.temporal_frames({**p, "temporal": 1}), 3)
        self.assertEqual(denoiseknobs.temporal_frames({**p, "temporal": 1, "legacy_denoise": 1}), 0)       # a saved node keeps its own three-frame average

    def moving_graph(self, motion_u=None, **knob):
        g = Graph()
        g.add("plate", "Noise", dict(width=64, height=32, size=6, octaves=3))
        g.add("move", "Transform", dict(filter="nearest"), image="plate")
        g.d.execute(dict(op="set_curve", id="move", param="translate_x",
                         curve=dict(interpolation="linear", keys=[dict(frame=f, value=float(3 * (f - 1))) for f in range(-5, 30)])))
        inputs = dict(image="move")
        if motion_u is not None:
            g.add("vec", "Constant", dict(width=64, height=32, red=motion_u, green=0.0, blue=0.0, alpha=1.0))
            inputs["motion"] = "vec"
        g.add("node", "Denoise", dict(denoise_strength=0.0, denoise_lumablend=1.0, **knob), **inputs)
        g.d.execute(dict(op="time", first=1, last=8, current=4))
        return g

    def frame4(self, g):
        return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=4).pixels

    def test_the_motion_input_aligns_the_neighbours_on_the_node(self):
        off = self.frame4(self.moving_graph(motion_u=3.0, temporal=0))
        aligned = self.frame4(self.moving_graph(motion_u=3.0, temporal=1, denoise_frame_blending=0.6))
        np.testing.assert_allclose(aligned[:, 16:48], off[:, 16:48], atol=1e-5)                   # neighbours moved onto the frame add nothing new
        misaligned = self.frame4(self.moving_graph(motion_u=0.0, temporal=1, denoise_frame_blending=0.6))
        self.assertTrue(differs(misaligned[:, 16:48], off[:, 16:48], 1e-3))                       # vectors that miss the motion smear the ramp

    def test_the_motion_input_is_a_slot_of_the_node(self):
        self.assertEqual(SPECS["Denoise"]["optional_inputs"], ["mask", "noise", "motion"])


class DenoiseTabsOnTheNodeTests(unittest.TestCase):
    def graph(self, **knob):
        g = Graph()
        g.add("plate", "Checker", dict(width=45, height=37, size=5))
        g.add("shade", "Grade", dict(multiply=[1.0, 0.4, 0.15, 1.0], exposure=-0.5), image="plate")
        g.add("node", "Denoise", dict(knob), image="shade")
        return g

    def test_the_tile_path_equals_the_evaluator_with_the_bands_on(self):
        for knob in ({}, {"denoise_process_high": 0, "denoise_high_gain": 2.0}, {"denoise_tune_profile": 1, "denoise_profile_low_gain": 0.5},
                     {"denoise_luma_gain": 2.0, "denoise_chroma_gain": 0.5, "denoise_process_very_low": 1}):
            g = self.graph(**knob)
            full = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(knob=knob, edge=edge):
                    np.testing.assert_allclose(tile_pixels(g.doc, "node", edge), full, atol=1e-6)

    def test_every_tab_knob_changes_the_node_picture(self):
        base = evaluator_pixels(self.graph(denoise_strength=0.6).doc, "node")
        for knob, value in (("denoise_luma_gain", 3.0), ("denoise_chroma_gain", 0.5), ("denoise_process_high", 0), ("denoise_high_gain", 0.0),
                            ("denoise_process_medium", 0), ("denoise_medium_gain", 0.0), ("denoise_process_low", 0), ("denoise_low_gain", 0.0),
                            ("denoise_process_very_low", 1)):
            with self.subTest(knob=knob):
                got = evaluator_pixels(self.graph(denoise_strength=0.6, **{knob: value}).doc, "node")
                self.assertTrue(differs(base, got, 1e-7), knob)
        # the very low band's gain acts once that band is processed
        on = evaluator_pixels(self.graph(denoise_strength=0.6, denoise_process_very_low=1).doc, "node")
        gained = evaluator_pixels(self.graph(denoise_strength=0.6, denoise_process_very_low=1, denoise_very_low_gain=0.0).doc, "node")
        self.assertTrue(differs(on, gained, 1e-7))
        tuned = evaluator_pixels(self.graph(denoise_strength=0.6, denoise_tune_profile=1, denoise_profile_high_gain=0.0, denoise_profile_low_gain=0.0,
                                            denoise_profile_mid_gain=0.0).doc, "node")
        self.assertTrue(differs(base, tuned))

    def test_the_main_tab_follows_the_guide(self):
        names = labels("Denoise")
        order = ["Preserve edges", "Enable", "Frame blending", "Frames to blend", "Lock noise analysis", "Analysis region x", "Analysis frame",
                 "Analysis file", "Process high frequencies", "High gain", "Process medium frequencies", "Medium gain", "Process low frequencies",
                 "Low gain", "Process very low frequencies", "Very low gain", "Luminance gain", "Chrominance gain", "Profile curve", "Tune profile",
                 "Profile low gain", "Profile mid gain", "Profile high gain", "Profile frame"]
        positions = [names.index(label) for label in order]
        self.assertEqual(positions, sorted(positions))
        kept, total = layout_order("Denoise")[0]
        self.assertEqual(kept, total)


class DenoiseButtonTests(unittest.TestCase):
    """The Noise Analysis buttons of the Denoise panel: Analyze noise, Export and Import."""

    def setUp(self):
        from tests.test_desktop import APP, Window, release_window
        self.app = APP
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.window = Window()
        self.addCleanup(release_window, self)
        self.window.show()
        APP.processEvents()
        d = self.window.dispatcher
        d.execute({"op": "create", "id": "grain", "type": "Noise", "pos": [0, 0], "params": {"width": 100, "height": 100, "size": 2, "octaves": 1}})
        d.execute({"op": "create", "id": "dn", "type": "Denoise", "pos": [0, 0],
                   "params": {"denoise_analysis_r": 100, "denoise_analysis_t": 100,
                              "denoise_analysis_file": os.path.join(self.tmp.name, "profile.json")}})
        d.execute({"op": "connect", "id": "dn", "input": "image", "source": "grain"})

    def button(self, name):
        from PySide6.QtWidgets import QPushButton
        self.panel = self.window.build_node_panel("dn")             # kept, or Qt deletes the buttons with it
        found = self.panel.findChild(QPushButton, name)
        self.assertIsNotNone(found, name)
        return found

    def params(self):
        return self.window.graph_nodes()["dn"]["params"]

    def test_the_panel_has_the_three_buttons(self):
        for name in ("denoise-analyze", "denoise-export", "denoise-import"):
            self.button(name)

    def test_analyze_holds_and_locks_then_export_and_import_carry_it(self):
        self.button("denoise-analyze").click()
        self.app.processEvents()
        held = self.params()["denoise_analysis_level"]
        self.assertGreater(held, 0.0)
        self.assertEqual(self.params()["denoise_lock_analysis"], 1)
        self.button("denoise-export").click()
        self.app.processEvents()
        self.assertTrue(os.path.isfile(self.params()["denoise_analysis_file"]))
        for param, value in (("denoise_analysis_level", 0.0), ("denoise_lock_analysis", 0)):
            self.window.dispatcher.execute({"op": "set", "id": "dn", "param": param, "value": value})
        self.button("denoise-import").click()
        self.app.processEvents()
        self.assertAlmostEqual(self.params()["denoise_analysis_level"], held, places=9)
        self.assertEqual(self.params()["denoise_lock_analysis"], 1)

    def test_export_with_no_file_named_does_nothing(self):
        self.window.dispatcher.execute({"op": "set", "id": "dn", "param": "denoise_analysis_file", "value": ""})
        self.assertIsNone(self.window.denoise_export("dn"))
        self.assertEqual(self.window.denoise_import("dn"), {})


if __name__ == "__main__":
    unittest.main()
