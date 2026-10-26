"""Fidelity step G, part 3: the Tracker takes Nuke's matching options, transform modes, live-link transform, per-track T, R and S,
and the transform tab's own knobs (skew, clamp, black outside, motion blur). Pixel assertions on synthetic moving patterns."""
import copy
import math
import unittest
from unittest import mock

import numpy as np

from nodebased import shapes, tracker, trackerknobs
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from nodebased.tracker import AnalysisError, analyse, match_pattern
from tests.test_2d_parity_group_c4 import Graph


# ------------------------------------------------------------------------------------------------- fixtures
def texture(size=96, seed=7, blur=0):
    rng = np.random.default_rng(seed)
    gray = rng.random((size, size), dtype=np.float32)
    if blur:
        kernel = np.exp(-0.5 * (np.arange(-3 * blur, 3 * blur + 1) / blur) ** 2)
        kernel /= kernel.sum()
        gray = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, gray)
        gray = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, gray)
        gray = (gray - gray.min()) / (gray.max() - gray.min())
    image = np.ones((size, size, 4), dtype=np.float32)
    image[..., :3] = gray[..., None]
    return image


def moved(image, dx, dy):
    out = np.zeros_like(image)
    y0, y1 = max(0, dy), min(image.shape[0], image.shape[0] + dy)
    x0, x1 = max(0, dx), min(image.shape[1], image.shape[1] + dx)
    out[y0:y1, x0:x1] = image[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def keyed(values):
    """An animatable scalar with a linear key per `{frame: value}`."""
    frames = sorted(values)
    return {"value": float(values[frames[0]]),
            "curve": {"interpolation": "linear", "keys": [{"frame": f, "value": float(values[f])} for f in frames]}}


def track(name, positions, **extra):
    """A track through `positions`, {frame: (x, y)}."""
    return {"name": name, "enabled": 1, "x": keyed({f: p[0] for f, p in positions.items()}),
            "y": keyed({f: p[1] for f, p in positions.items()}), **extra}


def rotated(point, degrees, about=(0.0, 0.0), shift=(0.0, 0.0)):
    angle = math.radians(degrees)
    dx, dy = point[0] - about[0], point[1] - about[1]
    return (about[0] + dx * math.cos(angle) - dy * math.sin(angle) + shift[0],
            about[1] + dx * math.sin(angle) + dy * math.cos(angle) + shift[1])


REFERENCE = {"a": (100.0, 100.0), "b": (200.0, 100.0), "c": (100.0, 200.0), "d": (200.0, 200.0)}
PARAMS = dict(SPECS["Tracker"]["params"], reference_frame=1)


# ------------------------------------------------------------------------------------------------- matching
class MatchingTests(unittest.TestCase):
    def test_new_tracker_has_nukes_matching_defaults(self):
        p = SPECS["Tracker"]["params"]
        self.assertEqual((p["pretrack_filter"], p["clamp_footage"], p["max_error"], p["grab_behavior"], p["grab_error_above"]),
                         ("median", 1, 0.2, "if error above", 0.05))
        self.assertEqual((p["adjust_for_luminance_changes"], p["predict_track"], p["snap_to_markers"]), (1, 0, 0))
        self.assertEqual((p["mode"], p["livelink_transform"]), ("match_move", 1))

    def test_the_filters_do_what_their_names_say(self):
        gray = np.full((7, 7), 0.4)
        gray[3, 3] = 9.0
        np.testing.assert_allclose(trackerknobs.prefiltered(gray, "median"), np.full((7, 7), 0.4))
        np.testing.assert_array_equal(trackerknobs.prefiltered(gray, "none"), gray)
        stretched = trackerknobs.prefiltered(np.array([[0.2, 0.3], [0.4, 0.6]]), "adjust contrast")
        self.assertEqual((float(stretched.min()), float(stretched.max())), (0.0, 1.0))

    def test_a_median_pre_filter_lets_a_noisy_frame_match(self):
        first = texture(96, 7, 2)
        second = moved(first, 3, 2)
        mask = np.random.default_rng(1).random(second.shape[:2]) < 0.30
        second[..., :3][mask] = np.random.default_rng(3).choice([0.0, 1.0], size=int(mask.sum()))[:, None]
        with self.assertRaisesRegex(AnalysisError, "no reliable match"):
            match_pattern(first, second, (48.5, 48.5), 8, 10, prefilter="none")
        x, y, score = match_pattern(first, second, (48.5, 48.5), 8, 10, prefilter="median")
        self.assertAlmostEqual(x, 51.5, delta=0.3)
        self.assertAlmostEqual(y, 50.5, delta=0.3)
        self.assertGreater(score, 0.8)

    def test_adjust_for_luminance_changes_keeps_the_match_on_a_dimmed_pattern(self):
        first = texture()
        second = moved(first, 3, 0)
        second[..., :3] *= 0.5                                   # the true pattern is half as bright
        patch = first[42:55, 42:55].copy()                       # a copy of the original, at full brightness, on another place
        patch[..., :3] += np.random.default_rng(2).normal(0, 0.03, patch[..., :3].shape).astype(np.float32)
        second[42:55, 30:43] = patch
        x, _, _ = match_pattern(first, second, (48.5, 48.5), 6, 12, metric="ncc")
        self.assertAlmostEqual(x, 51.5, delta=0.3)
        x, _, _ = match_pattern(first, second, (48.5, 48.5), 6, 12, metric="ssd")
        self.assertAlmostEqual(x, 36.5, delta=0.6)               # the levels themselves match the bright copy

    def test_clamping_the_footage_keeps_a_super_white_spike_from_deciding_the_match(self):
        first = texture()
        first[..., :3] *= 0.5
        first[48, 48, :3] = 100.0
        second = moved(first, 4, 0)
        second[48, 52, :3] = 0.3                                 # the spike is gone from where the pattern went
        second[48, 44, :3] = 100.0                               # and another one sits where it would have to be
        x, _, _ = match_pattern(first, second, (48.5, 48.5), 6, 8, clamp=False)
        self.assertAlmostEqual(x, 44.5, delta=0.3)
        x, _, _ = match_pattern(first, second, (48.5, 48.5), 6, 8, clamp=True)
        self.assertAlmostEqual(x, 52.5, delta=0.6)

    def test_predict_track_follows_a_point_that_speeds_up(self):
        first = texture()
        frames = {1: first, 2: moved(first, 6, 0), 3: moved(first, 18, 0)}
        with self.assertRaisesRegex(AnalysisError, "no reliable match"):
            analyse(frames, 1, (30.5, 48.5), 6, 8, last_frame=3, predict=False)
        result = analyse(frames, 1, (30.5, 48.5), 6, 8, last_frame=3, predict=True)
        self.assertAlmostEqual(result[3][0], 48.5, delta=0.5)

    def crossfade(self):
        a, b = texture(96, 7, 2), texture(96, 11, 2)
        frames = {}
        for k in range(1, 7):
            t = (k - 1) / 5
            image = (1 - t) * a + t * b
            image[..., 3] = 1
            frames[k] = moved(image, 2 * (k - 1), 0)
        return frames

    def test_the_pattern_grab_behaviours_decide_how_far_a_changing_pattern_is_followed(self):
        frames = self.crossfade()
        reached = {}
        for behaviour, grab in (("on first frame", trackerknobs.Grab("on first frame")),
                                ("every frame", trackerknobs.Grab("every frame")),
                                ("every n frames", trackerknobs.Grab("every n frames", interval=2)),
                                ("if error above", trackerknobs.Grab("if error above", above=0.05)),
                                ("if error below", trackerknobs.Grab("if error below", below=0.5))):
            info = {}
            result = analyse(frames, 1, (40.5, 48.5), 8, 10, last_frame=6, grab=grab, max_error=0.25, info=info)
            reached[behaviour] = max(result)
        self.assertEqual(reached["on first frame"], 3)           # the first pattern has faded from the picture by frame 4
        for behaviour in ("every frame", "every n frames", "if error above", "if error below"):
            self.assertEqual(reached[behaviour], 6, behaviour)

    def test_grab_decisions(self):
        for grab, step, error, due in ((trackerknobs.Grab("on first frame"), 3, 0.9, False), (trackerknobs.Grab("every frame"), 1, 0.0, True),
                                       (trackerknobs.Grab("every n frames", interval=3), 3, 0.0, True),
                                       (trackerknobs.Grab("every n frames", interval=3), 4, 0.0, False),
                                       (trackerknobs.Grab("every n frames", interval=0), 4, 0.0, False),
                                       (trackerknobs.Grab("if error above", above=0.1), 1, 0.2, True),
                                       (trackerknobs.Grab("if error above", above=0.1), 1, 0.05, False),
                                       (trackerknobs.Grab("if error below", below=0.1), 1, 0.05, True),
                                       (trackerknobs.Grab("if error below", below=0.1), 1, 0.2, False)):
            self.assertEqual(grab.due(step, error), due, (grab, step, error))

    def test_max_error_stops_tracking_and_keeps_the_frames_before_it(self):
        frames = self.crossfade()
        info = {}
        result = analyse(frames, 1, (40.5, 48.5), 8, 10, last_frame=6, max_error=0.25, info=info)
        self.assertEqual(sorted(result), [1, 2, 3])
        self.assertEqual(info["stopped_at"], 4)
        self.assertGreater(info["error"], 0.25)
        self.assertEqual(max(analyse(frames, 1, (40.5, 48.5), 8, 10, last_frame=6)), 6)     # without it the old rule: on to the end

    def test_the_knobs_choose_the_analysis_options(self):
        options = trackerknobs.analysis_options({**PARAMS, "adjust_for_luminance_changes": 0, "pretrack_filter": "adjust contrast",
                                                 "max_error": 1.0, "predict_track": 1, "clamp_footage": 0})
        self.assertEqual((options["metric"], options["prefilter"], options["max_error"], options["predict"], options["clamp"]),
                         ("ssd", "adjust contrast", None, True, False))
        self.assertEqual(trackerknobs.analysis_options({**PARAMS, "adaptive_update": 1})["grab"].behavior, "every frame")
        self.assertEqual(trackerknobs.analysis_options(PARAMS)["max_error"], 0.2)

    def test_snap_to_marker_goes_to_the_strongest_feature_near_the_point(self):
        gray = np.full((40, 40), 0.2)
        gray[19:22, 29:32] = 0.9                                  # a 3 x 3 marker centred on pixel (30, 20)
        x, y = trackerknobs.snap_to_marker(gray, (32.5, 21.5))
        self.assertAlmostEqual(x, 30.5)
        self.assertAlmostEqual(y, 20.5)
        self.assertEqual(trackerknobs.snap_to_marker(gray, (5.5, 5.5)), (5.5, 5.5))        # flat: stays
        self.assertEqual(trackerknobs.snap_to_marker(gray, (60.5, 30.5)), (60.5, 30.5))     # nothing within reach

    def test_per_track_areas_override_the_nodes(self):
        self.assertEqual(trackerknobs.track_radii({"pattern_radius": 8, "search_radius": 24}, {}), (8, 24))
        self.assertEqual(trackerknobs.track_radii({"pattern_radius": 8, "search_radius": 24}, {"pattern_radius": 5, "search_radius": 12}), (5, 12))
        self.assertEqual(trackerknobs.track_radii({"pattern_radius": 8, "search_radius": 24}, {"pattern_radius": 30}), (30, 30))


# ------------------------------------------------------------------------------------------------- the solve
class SolveTests(unittest.TestCase):
    def payload(self, rotation=10.0, shift=(5.0, 3.0), flags=None, about=(0.0, 0.0)):
        tracks = []
        for name, point in REFERENCE.items():
            tracks.append(track(name, {1: point, 2: rotated(point, rotation, about, shift)}, **(flags or {}).get(name, {})))
        return {"tracks": tracks}

    def test_all_tracks_with_no_switches_give_the_similarity_they_always_did(self):
        solved = tracker.solve(self.payload(about=(150.0, 150.0)), 2, PARAMS)
        self.assertAlmostEqual(solved["rotate"], 10.0, places=6)
        self.assertAlmostEqual(solved["scale"], 1.0, places=9)
        self.assertAlmostEqual(solved["translate_x"], 5.0, places=6)
        self.assertAlmostEqual(solved["translate_y"], 3.0, places=6)

    def test_only_the_tracks_with_t_translate_and_only_the_tracks_with_r_rotate(self):
        # Tracks a and d give translation; b and c give rotation; nobody gives scale.
        flags = {"a": {"use_t": 1, "use_r": 0, "use_s": 0}, "d": {"use_t": 1, "use_r": 0, "use_s": 0},
                 "b": {"use_t": 0, "use_r": 1, "use_s": 0}, "c": {"use_t": 0, "use_r": 1, "use_s": 0}}
        payload = self.payload(flags=flags, about=(150.0, 150.0))
        # Break the tracks that do not rotate: they only slide, so a rotation read from them would be wrong.
        for name in ("a", "d"):
            payload["tracks"][list(REFERENCE).index(name)] = track(name, {1: REFERENCE[name], 2: (REFERENCE[name][0] + 7.0, REFERENCE[name][1] - 2.0)},
                                                                   **flags[name])
        solved = tracker.solve(payload, 2, PARAMS)
        self.assertAlmostEqual(solved["translate_x"], 7.0, places=6)
        self.assertAlmostEqual(solved["translate_y"], -2.0, places=6)
        self.assertAlmostEqual(solved["rotate"], 10.0, places=5)          # from b and c, which turn about (150, 150)
        self.assertEqual(solved["scale"], 1.0)

    def test_a_track_without_s_does_not_scale(self):
        payload = {"tracks": [track(name, {1: point, 2: (150 + (point[0] - 150) * 1.5, 150 + (point[1] - 150) * 1.5)}, use_t=1, use_r=0, use_s=0)
                              for name, point in REFERENCE.items()]}
        solved = tracker.solve(payload, 2, PARAMS)
        self.assertEqual((solved["rotate"], solved["scale"]), (0.0, 1.0))
        for track_ in payload["tracks"]:
            track_["use_s"] = 1
        self.assertAlmostEqual(tracker.solve(payload, 2, PARAMS)["scale"], 1.5, places=6)

    def test_the_flags_default_to_all_three(self):
        self.assertEqual(trackerknobs.track_flags({}), (True, True, True))
        self.assertEqual(trackerknobs.track_flags({"use_t": 1, "use_r": 0, "use_s": 0}), (True, False, False))
        self.assertEqual(trackerknobs.track_flags({"use_r": 1}), (False, True, False))

    def test_one_point_modes_use_the_first_track_alone(self):
        payload = self.payload(about=(150.0, 150.0))
        full = tracker.solve(payload, 2, PARAMS)
        one = tracker.solve(payload, 2, {**PARAMS, "mode": "match_move_1pt"})
        first_moved = rotated(REFERENCE["a"], 10.0, (150.0, 150.0), (5.0, 3.0))
        self.assertNotEqual(full["rotate"], 0.0)
        self.assertEqual((one["rotate"], one["scale"]), (0.0, 1.0))
        self.assertAlmostEqual(one["translate_x"], first_moved[0] - REFERENCE["a"][0], places=6)
        self.assertAlmostEqual(one["translate_y"], first_moved[1] - REFERENCE["a"][1], places=6)
        stabilise = tracker.solve(payload, 2, {**PARAMS, "mode": "stabilise_1pt"})
        self.assertAlmostEqual(stabilise["translate_x"], -one["translate_x"], places=9)
        self.assertEqual(stabilise["rotate"], 0.0)

    def test_none_and_live_link_off_give_the_nodes_own_transform(self):
        payload = self.payload()
        knobs = dict(PARAMS, translate_x=4.0, translate_y=-2.0, rotate=15.0, scale=1.2, center_x=10.0, center_y=20.0)
        want = {"translate_x": 4.0, "translate_y": -2.0, "rotate": 15.0, "scale": 1.2, "center_x": 10.0, "center_y": 20.0}
        self.assertEqual(tracker.solve(payload, 2, {**knobs, "mode": "none"}), want)
        self.assertEqual(tracker.solve(payload, 2, {**knobs, "livelink_transform": 0}), want)
        solved = tracker.solve(payload, 2, {**knobs, "livelink_transform": 1})
        self.assertNotEqual(solved, want)
        # A Stabilize is always solved.
        self.assertNotEqual(tracker.solve(payload, 2, {**knobs, "mode": "none"}, force_stabilise=True), want)

    def jitter(self):
        shake = {1: 0.0, 2: 3.0, 3: -3.0, 4: 3.0, 5: -3.0, 6: 3.0}
        return {"tracks": [track("a", {f: (100.0 + 10.0 * f + s, 50.0) for f, s in shake.items()}),
                           track("b", {f: (200.0 + 10.0 * f + s, 80.0) for f, s in shake.items()})]}, shake

    def test_remove_jitter_moves_the_picture_along_the_mean_path_and_add_jitter_along_the_shake(self):
        payload, shake = self.jitter()
        for frame in (3, 4):
            samples = [100.0 + 10.0 * f + shake[f] for f in (frame - 1, frame, frame + 1)]
            mean = sum(samples) / 3
            raw = 100.0 + 10.0 * frame + shake[frame]
            removed = tracker.solve(payload, frame, {**PARAMS, "mode": "remove_jitter", "jitter_period": 2})
            added = tracker.solve(payload, frame, {**PARAMS, "mode": "add_jitter", "jitter_period": 2})
            self.assertAlmostEqual(removed["translate_x"], mean - raw, places=6)
            self.assertAlmostEqual(added["translate_x"], raw - mean, places=6)
            self.assertAlmostEqual(removed["translate_y"], 0.0, places=9)
            self.assertAlmostEqual(removed["rotate"], 0.0, places=6)

    def test_a_longer_jitter_period_smooths_over_more_frames(self):
        payload, shake = self.jitter()
        short = tracker.solve(payload, 4, {**PARAMS, "mode": "remove_jitter", "jitter_period": 2})
        long = tracker.solve(payload, 4, {**PARAMS, "mode": "remove_jitter", "jitter_period": 4})
        self.assertNotAlmostEqual(short["translate_x"], long["translate_x"], places=3)

    def test_fractional_frames_interpolate_for_motion_blur(self):
        payload = self.payload(rotation=0.0, shift=(10.0, 0.0))
        solved = tracker.solve(payload, 1.5, PARAMS)
        self.assertAlmostEqual(solved["translate_x"], 5.0, places=9)


# ------------------------------------------------------------------------------------------------ the pixels
W, H = 64, 48


def build(tracks, **params):
    g = Graph()
    g.add("plate", "Checker", dict(width=W, height=H, size=4))
    g.add("node", "Tracker", {"filter": "nearest", **params}, image="plate")
    g.d.execute({"op": "set_tracks", "id": "node", "tracks": tracks})
    return g


def at_frame(g, frame, target="node"):
    return Evaluator().evaluate_raster(dict(g.doc, view=target), target=target, frame=frame)


SLIDING = [track("a", {1: (20.0, 20.0), 3: (28.0, 24.0)})]


class PixelTests(unittest.TestCase):
    def transform_reference(self, g, frame, **params):
        g.add("ref", "Transform", {"filter": "nearest", **params}, image="plate")
        return at_frame(g, frame, "ref")

    def test_match_move_equals_the_transform_it_solves(self):
        g = build(SLIDING)
        got = at_frame(g, 3)
        want = self.transform_reference(g, 3, translate_x=8.0, translate_y=4.0)
        self.assertEqual(got.data, want.data)
        np.testing.assert_array_equal(got.pixels, want.pixels)

    def test_none_is_the_nodes_own_transform(self):
        g = build(SLIDING, mode="none", translate_x=5.0, translate_y=-3.0)
        want = self.transform_reference(g, 3, translate_x=5.0, translate_y=-3.0)
        np.testing.assert_array_equal(at_frame(g, 3).pixels, want.pixels)
        np.testing.assert_array_equal(at_frame(g, 1).pixels, want.pixels)          # tracks play no part

    def test_live_link_off_keeps_the_knobs_and_turning_it_on_follows_the_tracks(self):
        off = at_frame(build(SLIDING, livelink_transform=0, translate_x=2.0), 3)
        on = at_frame(build(SLIDING, livelink_transform=1, translate_x=2.0), 3)
        self.assertNotEqual(off.data, on.data)
        want = self.transform_reference(build(SLIDING), 3, translate_x=8.0, translate_y=4.0)
        self.assertEqual(on.data, want.data)
        np.testing.assert_array_equal(on.pixels, want.pixels)
        self.assertEqual(off.data, self.transform_reference(build(SLIDING), 3, translate_x=2.0).data)

    def test_skew_comes_from_the_transform_tab(self):
        g = build(SLIDING, mode="none", skew_x=0.3, filter="bilinear")
        g.add("ref", "Transform", dict(skew_x=0.3, filter="bilinear"), image="plate")
        np.testing.assert_allclose(at_frame(g, 1).pixels, at_frame(g, 1, "ref").pixels, atol=1e-6)

    def test_clamp_and_black_outside_reach_the_resampler(self):
        shifted = dict(mode="none", translate_x=0.4, filter="cubic")
        free = at_frame(build(SLIDING, clamp=0, **shifted), 1).pixels
        held = at_frame(build(SLIDING, clamp=1, **shifted), 1).pixels
        self.assertFalse(np.array_equal(free, held))
        self.assertGreaterEqual(float(held[..., :3].min()), 0.0)
        g = build(SLIDING, mode="none", translate_x=10.0, black_outside=1)
        black = at_frame(g, 1)
        np.testing.assert_array_equal(black.pixels[:, 0], np.broadcast_to(np.float32([0, 0, 0, 1]), black.pixels[:, 0].shape))
        plain = at_frame(build(SLIDING, mode="none", translate_x=10.0, black_outside=0), 1)
        self.assertFalse(np.array_equal(black.pixels, plain.pixels))

    def test_motion_blur_is_the_mean_of_the_shutter_samples(self):
        g = build(SLIDING, motionblur=1, samples=4, shutter=1.0, shutter_offset="centred")
        blurred = at_frame(g, 2)
        sharp = build(SLIDING)
        frames = [1.5 + (i + 0.5) * 1.0 / 4 for i in range(4)]
        samples = [at_frame(sharp, f) for f in frames]
        union = samples[0].data
        for s in samples[1:]:
            union = union.union(s.data)
        np.testing.assert_allclose(blurred.pixels, np.mean(np.stack([s.fit(union) for s in samples]), axis=0), atol=1e-5)
        self.assertFalse(np.array_equal(blurred.pixels, at_frame(sharp, 2).fit(blurred.data)))

    def test_the_tile_path_equals_the_evaluator_and_blur_falls_back(self):
        for params in (dict(), dict(mode="none", translate_x=3.5, skew_y=0.1, filter="bilinear"),
                       dict(mode="remove_jitter", jitter_period=2), dict(mode="match_move_1pt"), dict(pretrack_filter="none", smoothing=1)):
            with self.subTest(params=params):
                g = build(SLIDING, **{"filter": "bilinear", **params})
                document = dict(g.doc, view="node")
                executor = TileExecutor(evaluator=Evaluator())
                self.assertTrue(executor.supports_tiled(document, "node"))
                tiled = executor.compose(document, "node", frame=3, tier=1)
                self.assertTrue(tiled.tiled)
                np.testing.assert_allclose(tiled.pixels, Evaluator().evaluate(document, "node", frame=3), atol=1e-6)
        for name in ("motionblur", "clamp", "black_outside"):
            with self.subTest(excluded=name):
                excluded = build(SLIDING, **{name: 1})
                self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(excluded.doc, view="node"), "node"))

    def test_bypass_passes_the_plate(self):
        g = build(SLIDING, motionblur=1)
        g.bypass("node")
        np.testing.assert_array_equal(at_frame(g, 3).pixels, at_frame(g, 3, "plate").pixels)

    def test_an_old_document_loads_and_draws_as_it_did(self):
        g = build(SLIDING, filter="bilinear")
        before = at_frame(g, 3).pixels
        document = copy.deepcopy(g.doc)
        for name in trackerknobs.NEW:
            document["nodes"]["node"]["params"].pop(name, None)
        loaded = upgrade_document(document)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual((params["pretrack_filter"], params["clamp_footage"], params["max_error"], params["grab_behavior"],
                          params["livelink_transform"]), ("none", 0, 1.0, "on first frame", 1))
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="node"), target="node", frame=3).pixels, before)
        self.assertEqual(trackerknobs.analysis_options(params)["max_error"], None)         # the older score-under-0.5 rule

    def test_track_options_are_validated_and_stored(self):
        g = build(SLIDING)
        tracks = copy.deepcopy(g.doc["node_data"]["node"]["tracks"])
        tracks[0].update(use_t=1, use_r=0, use_s=1, pattern_radius=6, search_radius=20)
        g.d.execute({"op": "set_tracks", "id": "node", "tracks": tracks})
        self.assertEqual(g.doc["node_data"]["node"]["tracks"][0]["search_radius"], 20)
        for bad in ({"use_t": 2}, {"pattern_radius": 0}, {"search_radius": 1.5}, {"colour": 1}):
            with self.assertRaises(ValueError):
                g.d.execute({"op": "set_tracks", "id": "node", "tracks": [{**tracks[0], **bad}]})


# ------------------------------------------------------------------------------------------------ the panel
def _open_window(setup=None):
    from PySide6.QtWidgets import QApplication
    from nodebased.app import Window
    from nodebased.core import Dispatcher, empty_document
    app = QApplication.instance() or QApplication([])
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "n1", "type": "Constant", "pos": [0, 0], "params": {"width": 64, "height": 48}})
    d.execute({"op": "create", "id": "t1", "type": "Tracker", "pos": [0, 100]})
    d.execute({"op": "connect", "id": "t1", "input": "image", "source": "n1"})
    d.execute({"op": "view", "id": "t1"})
    if setup is not None:
        setup(d)
    window = Window(d.document)
    window.graph.items_by_id["t1"].setSelected(True)
    window.inspect("t1")
    app.processEvents()
    return window, app


def _close_window(window, app):
    window.saved_document = window.dispatcher.document
    window.close()
    window.executor.shutdown(wait=True, cancel_futures=True)
    app.processEvents()


def _moving_setup(d):
    d.execute({"op": "create", "id": "tex", "type": "Noise", "pos": [0, 0], "params": {"width": 160, "height": 120, "size": 6.0, "octaves": 3}})
    d.execute({"op": "create", "id": "slide", "type": "Transform", "pos": [0, 50], "params": {"filter": "nearest"}})
    d.execute({"op": "connect", "id": "slide", "input": "image", "source": "tex"})
    for frame, value in ((1, 0.0), (2, 10.0), (3, 20.0)):
        d.execute({"op": "set_key", "id": "slide", "param": "translate_x", "frame": frame, "value": value})
    d.execute({"op": "connect", "id": "t1", "input": "image", "source": "slide"})
    d.document["time"].update(first=1, last=3, current=1)


# Opening a Window takes seconds (it probes the graphics adapters), so the panel tests share two of them: one on a plain picture
# for the buttons, one on a texture that slides ten pixels a frame for the analysis.
class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain_window, cls.app = _open_window()
        cls.moving_window_, _ = _open_window(_moving_setup)

    @classmethod
    def tearDownClass(cls):
        _close_window(cls.plain_window, cls.app)
        _close_window(cls.moving_window_, cls.app)

    NODE_DEFAULTS = {name: SPECS["Tracker"]["params"][name] for name in
                     ("snap_to_markers", "create_key_on_move", "hide_progress_bar", "autotracks_delete_keyframes", "pattern_radius", "search_radius")}

    def reset(self, window, tracks, **params):
        d = window.dispatcher
        for name, value in {**self.NODE_DEFAULTS, **params}.items():
            d.execute({"op": "set", "id": "t1", "param": name, "value": value})
        d.execute({"op": "set_tracks", "id": "t1", "tracks": tracks})
        d.document["time"].update(first=1, last=3, current=1)
        window._tracker_selected_index = None
        window._tracker_seed = window._tracker_index = window._tracker_key = None
        window.inspect("t1")
        self.app.processEvents()
        return window, self.app

    def window(self, tracks=None):
        return self.reset(self.plain_window, [] if tracks is None else tracks)

    def panel(self, window, app):
        from PySide6.QtWidgets import QWidget
        app.processEvents()
        return window.properties.widget().findChild(QWidget, "tracker-points-panel")

    def tracks(self, window):
        return window.dispatcher.document.get("node_data", {}).get("t1", {"tracks": []})["tracks"]

    def three(self):
        return [track("a", {1: (10.0, 10.0), 3: (20.0, 10.0)}), track("b", {1: (30.0, 20.0), 3: (40.0, 20.0)}),
                {"name": "c", "enabled": 1, "x": 50.0, "y": 40.0}]

    def test_add_track_puts_a_new_track_in_the_middle_of_the_picture(self):
        from PySide6.QtWidgets import QPushButton
        window, app = self.window(tracks=[])
        self.panel(window, app).findChild(QPushButton, "tracker-add-track").click()
        app.processEvents()
        self.assertEqual([(t["name"], t["x"], t["y"], t["enabled"]) for t in self.tracks(window)], [("track1", 32.0, 24.0, 1)])
        self.panel(window, app).findChild(QPushButton, "tracker-add-track").click()
        self.assertEqual([t["name"] for t in self.tracks(window)], ["track1", "track2"])

    def test_select_all_then_delete_tracks_removes_every_track(self):
        from PySide6.QtWidgets import QListWidget, QPushButton
        window, app = self.window(tracks=self.three())
        panel = self.panel(window, app)
        panel.findChild(QPushButton, "tracker-select-all").click()
        self.assertEqual(len(panel.findChild(QListWidget, "tracker-track-list").selectedItems()), 3)
        panel.findChild(QPushButton, "tracker-del-tracks").click()
        app.processEvents()
        self.assertEqual(self.tracks(window), [])

    def test_delete_tracks_removes_the_selected_ones_only(self):
        from PySide6.QtWidgets import QListWidget, QPushButton
        window, app = self.window(tracks=self.three())
        panel = self.panel(window, app)
        listing = panel.findChild(QListWidget, "tracker-track-list")
        listing.item(0).setSelected(True)
        listing.item(2).setSelected(True)
        panel.findChild(QPushButton, "tracker-del-tracks").click()
        app.processEvents()
        self.assertEqual([t["name"] for t in self.tracks(window)], ["b"])

    def test_average_tracks_adds_a_track_at_the_mean_of_the_selected_ones(self):
        from PySide6.QtWidgets import QListWidget, QPushButton
        window, app = self.window(tracks=self.three())
        window.dispatcher.document["time"].update(first=1, last=3)
        panel = self.panel(window, app)
        listing = panel.findChild(QListWidget, "tracker-track-list")
        listing.item(0).setSelected(True)
        listing.item(1).setSelected(True)
        panel.findChild(QPushButton, "tracker-average-tracks").click()
        app.processEvents()
        tracks = self.tracks(window)
        self.assertEqual([t["name"] for t in tracks], ["a", "b", "c", "average"])
        for frame, (x, y) in ((1, (20.0, 15.0)), (2, (25.0, 15.0)), (3, (30.0, 15.0))):
            got = shapes.resolve_tracks({"tracks": [tracks[3]]}, frame)[0]
            self.assertAlmostEqual(got["x"], x, places=6)
            self.assertAlmostEqual(got["y"], y, places=6)

    def test_average_needs_two_tracks(self):
        window, app = self.window(tracks=self.three())
        errors = []
        patch = mock.patch.object(window, "_show_command_error", lambda error: errors.append(str(error)))
        patch.start()
        self.addCleanup(patch.stop)
        self.assertFalse(window.average_selected_tracks("t1", [1]))
        self.assertIn("at least two", errors[0])
        self.assertEqual(len(self.tracks(window)), 3)

    def test_t_r_and_s_boxes_write_the_tracks_own_switches(self):
        from PySide6.QtWidgets import QCheckBox
        window, app = self.window(tracks=self.three())
        panel = self.panel(window, app)
        for letter in "trs":
            self.assertTrue(panel.findChild(QCheckBox, f"tracker-track-{letter}-0").isChecked())
        panel.findChild(QCheckBox, "tracker-track-r-0").setChecked(False)
        app.processEvents()
        first = self.tracks(window)[0]
        self.assertEqual((first["use_t"], first["use_r"], first["use_s"]), (1, 0, 1))
        panel = self.panel(window, app)
        self.assertFalse(panel.findChild(QCheckBox, "tracker-track-r-0").isChecked())
        self.assertTrue(panel.findChild(QCheckBox, "tracker-track-t-1").isChecked())

    def test_a_tracks_own_pattern_and_search_areas_are_stored_with_it(self):
        from PySide6.QtWidgets import QSpinBox
        window, app = self.window(tracks=self.three())
        panel = self.panel(window, app)
        self.assertEqual(panel.findChild(QSpinBox, "tracker-track-pattern-1").value(), 8)
        panel.findChild(QSpinBox, "tracker-track-search-1").setValue(40)
        panel.findChild(QSpinBox, "tracker-track-search-1").editingFinished.emit()
        app.processEvents()
        second = self.tracks(window)[1]
        self.assertEqual((second["pattern_radius"], second["search_radius"]), (8, 40))
        self.assertNotIn("pattern_radius", self.tracks(window)[0])

    # --- tracking again, with a track's own search area, through the real analysis
    def moving_window(self, **node_params):
        window, app = self.reset(self.moving_window_, [{"name": "track1", "enabled": 1, "x": 80.5, "y": 60.5}],
                                 **{"pattern_radius": 6, "search_radius": 8, **node_params})
        window._tracker_selected_index = 0
        return window, app

    def record_status(self, window):
        messages = []
        bar = window.statusBar()
        original = type(bar).showMessage
        bar.showMessage = lambda text, *args: (messages.append(text), original(bar, text, *args))[1]
        self.addCleanup(lambda: bar.__dict__.pop("showMessage", None))
        return messages

    def run_retrack(self, window, app):
        self.assertTrue(window.retrack_selected("t1", "forward", last_frame=3))
        window._tracker_future.result(timeout=120)
        window._poll_tracker_analysis()
        app.processEvents()

    def test_the_node_areas_lose_a_fast_point_and_the_tracks_own_area_follows_it(self):
        window, app = self.moving_window()
        messages = self.record_status(window)
        self.run_retrack(window, app)
        self.assertTrue(any("stopped at frame 2" in text for text in messages), messages[-3:])
        lost = shapes.resolve_tracks({"tracks": self.tracks(window)}, 3)[0]
        self.assertAlmostEqual(lost["x"], 80.5, places=3)                 # kept the one frame it tracked
        tracks = copy.deepcopy(self.tracks(window))
        tracks[0].update(pattern_radius=6, search_radius=14)
        window.dispatcher.execute({"op": "set_tracks", "id": "t1", "tracks": tracks})
        window._tracker_selected_index = 0
        self.run_retrack(window, app)
        found = shapes.resolve_tracks({"tracks": self.tracks(window)}, 3)[0]
        self.assertAlmostEqual(found["x"], 100.5, delta=0.6)
        self.assertAlmostEqual(found["y"], 60.5, delta=0.6)
        self.assertAlmostEqual(shapes.resolve_tracks({"tracks": self.tracks(window)}, 2)[0]["x"], 90.5, delta=0.6)

    def test_hide_progress_bar_keeps_the_analysis_off_the_status_bar(self):
        for hide, expect in ((0, True), (1, False)):
            with self.subTest(hide=hide):
                window, app = self.moving_window(hide_progress_bar=hide, search_radius=14)
                messages = self.record_status(window)
                self.run_retrack(window, app)
                self.assertEqual(any(text.startswith("Tracker forward") for text in messages), expect, messages)
                self.assertTrue(any("complete" in text for text in messages))

    def test_tracking_again_keeps_or_replaces_the_old_keys_outside_the_tracked_frames(self):
        for delete, expect_frame_1 in ((1, False), (0, True)):
            with self.subTest(delete=delete):
                window, app = self.moving_window(autotracks_delete_keyframes=delete, search_radius=14)
                old = track("track1", {1: (80.5, 60.5), 5: (500.0, 500.0)})
                window.dispatcher.execute({"op": "set_tracks", "id": "t1", "tracks": [old]})
                window.dispatcher.document["time"].update(first=1, last=5)
                self.run_retrack(window, app)
                keys = {key["frame"] for key in self.tracks(window)[0]["x"]["curve"]["keys"]}
                self.assertEqual(5 in keys, expect_frame_1, keys)
                self.assertTrue({1, 2, 3} <= keys)

    def test_snap_to_markers_moves_a_picked_point_to_the_marker(self):
        from nodebased.raster import Raster
        from nodebased.tiers import Region
        window, app = self.window(tracks=[])
        pixels = np.full((48, 64, 4), 0.2, dtype=np.float32)
        pixels[19:22, 29:32, :3] = 0.9
        raster = Raster(pixels, Region(0, 0, 64, 48), Region(0, 0, 64, 48))
        patch = mock.patch.object(window.evaluator, "evaluate_raster", lambda *args, **kwargs: raster)
        patch.start()
        self.addCleanup(patch.stop)
        context = window._tracker_context("t1")
        self.assertEqual(window._snap_tracker_point("t1", context, (32.5, 21.5))[0], 31.5 - 1.0)   # the marker's centre, 30.5
        self.assertAlmostEqual(window._snap_tracker_point("t1", context, (32.5, 21.5))[1], 20.5)
        window.dispatcher.execute({"op": "set", "id": "t1", "param": "snap_to_markers", "value": 1})
        window.add_tracker_point((32.5, 21.5))
        self.assertAlmostEqual(window._tracker_seed["x"], 30.5)
        self.assertAlmostEqual(window._tracker_seed["y"], 20.5)
        window.dispatcher.execute({"op": "set", "id": "t1", "param": "snap_to_markers", "value": 0})
        window._tracker_seed = window._tracker_index = window._tracker_key = None
        window.add_tracker_point((32.5, 21.5))
        self.assertEqual((window._tracker_seed["x"], window._tracker_seed["y"]), (32.5, 21.5))

    def drag(self, window, create_key):
        from PySide6.QtCore import QPointF
        window.dispatcher.execute({"op": "set", "id": "t1", "param": "create_key_on_move", "value": create_key})
        window.dispatcher.document["time"].update(first=1, last=3, current=2)
        window.viewer.tracker_drag = {"kind": "point", "index": 0, "scene": QPointF(40.0, 30.0), "moved": True, "frame": 2}
        window.viewer._commit_tracker_drag()
        return shapes.resolve_tracks({"tracks": self.tracks(window)}, 1)[0], shapes.resolve_tracks({"tracks": self.tracks(window)}, 2)[0], \
            shapes.resolve_tracks({"tracks": self.tracks(window)}, 3)[0]

    def test_a_drag_keys_the_frame_or_moves_the_whole_track(self):
        for create_key in (1, 0):
            with self.subTest(create_key=create_key):
                window, app = self.window(tracks=[track("a", {1: (10.0, 10.0), 2: (15.0, 12.0), 3: (20.0, 10.0)})])
                one, two, three = self.drag(window, create_key)
                self.assertAlmostEqual(two["x"], 40.0)
                self.assertAlmostEqual(two["y"], 30.0)
                if create_key:
                    self.assertAlmostEqual(one["x"], 10.0)            # one key replaced, the others stay
                    self.assertAlmostEqual(three["x"], 20.0)
                else:
                    self.assertAlmostEqual(one["x"], 35.0)            # every key moved by the drag (+25, +18)
                    self.assertAlmostEqual(one["y"], 28.0)
                    self.assertAlmostEqual(three["x"], 45.0)

    def test_a_drag_of_the_search_box_resizes_the_tracks_own_area(self):
        from PySide6.QtCore import QPointF
        tracks = [{**track("a", {1: (30.0, 20.0)}), "pattern_radius": 5, "search_radius": 12}]
        window, app = self.window(tracks=tracks)
        window.viewer.tracker_drag = {"kind": "search", "center": (30.0, 20.0), "index": 0, "scene": QPointF(30.0 + 20.0, 20.0), "moved": True, "frame": 1}
        window.viewer._commit_tracker_drag()
        self.assertEqual((self.tracks(window)[0]["pattern_radius"], self.tracks(window)[0]["search_radius"]), (5, 20))
        self.assertEqual(window.dispatcher.document["nodes"]["t1"]["params"]["search_radius"], 24)       # the node's own is untouched


class MergedKeysTests(unittest.TestCase):
    def test_merged_keys(self):
        old = [{"frame": 1, "value": 1.0}, {"frame": 5, "value": 5.0}, {"frame": 9, "value": 9.0}]
        new = [{"frame": 4, "value": 40.0}, {"frame": 6, "value": 60.0}]
        self.assertEqual([k["frame"] for k in trackerknobs.merged_keys(old, new, True)], [4, 6])
        self.assertEqual([k["frame"] for k in trackerknobs.merged_keys(old, new, False)], [1, 4, 6, 9])
        self.assertEqual([k["frame"] for k in trackerknobs.merged_keys([], new, False)], [4, 6])


if __name__ == "__main__":
    unittest.main()
