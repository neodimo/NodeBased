"""Lane 2, step O: SmartVector's Strength, Flicker Compensation, luminance weights, Inpaint Matte Region and Matte Dilation."""
import copy
import unittest

import numpy as np

from nodebased import smartvectorknobs as svk
from nodebased.core import LIMITS, SCHEMA_VERSION, SPECS, Dispatcher, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from tests.test_2d_behaviour_smartvector_l import FRAMES, H, W, MatteFixture, texture

NEW = ("match_strength", "inpainting", "matte_dilation", "flicker_compensation", "weight_red", "weight_green", "weight_blue")


def build(extra=None, channel="None", output="Background"):
    d = Dispatcher()
    d.execute({"op": "time", "first": 1, "last": FRAMES})
    d.execute({"op": "create", "id": "src", "type": "Constant", "params": {"width": W, "height": H}})
    d.execute({"op": "create", "id": "mat", "type": "Constant", "params": {"width": W, "height": H}})
    params = {"frame_start": 1, "frame_end": FRAMES, "reference_frame": 1, "vector_detail": 3, "smoothness": 0.5,
              "reanchor_interval": 5, "flow_backend": "cpu", "matte_channel": channel, "matte_output": output}
    params.update(extra or {})
    d.execute({"op": "create", "id": "sv", "type": "SmartVector", "params": params})
    d.execute({"op": "connect", "id": "sv", "input": "image", "source": "src"})
    d.execute({"op": "connect", "id": "sv", "input": "matte", "source": "mat"})
    return d


class Plated(Evaluator):
    """An evaluator whose `src` plate is `plate(frame)` and whose `mat` plate is the matte fixture."""

    def __init__(self, plate):
        super().__init__()
        self.plate = plate

    def evaluate_raster(self, doc, key, *a, **k):
        r = super().evaluate_raster(doc, key, *a, **k)
        if key not in ("src", "mat"):
            return r
        raster = r[0] if isinstance(r, tuple) else r
        frame = int(k.get("frame", 1))
        pixels = self.plate(frame) if key == "src" else MatteFixture.matte(frame)
        raster = type(raster)(pixels, raster.data, raster.display, raster.layers, raster.meta)
        return (raster, r[1]) if isinstance(r, tuple) else raster


def vectors(doc, plate, frame=3, layer="smartvector.forward"):
    return Plated(plate).evaluate_raster(doc, "sv", frame=frame, typed=True).layers[layer].pixels


def moving(frame):
    return texture(2 * (frame - 1))


class ShapeTests(unittest.TestCase):
    def test_the_node_has_the_seven_new_params_with_the_guide_defaults(self):
        params = SPECS["SmartVector"]["params"]
        self.assertEqual([params[n] for n in NEW], [1.5, 0, 50, 0, 0.3, 0.6, 0.1])
        for name in NEW:
            self.assertIn(name, LIMITS)

    def test_the_panel_lists_the_knobs_in_the_guide_s_order(self):
        names = [name for g in knob_layout("SmartVector") for name in g.params]
        order = ["vector_detail", "match_strength", "matte_channel", "matte_output", "inpainting", "matte_dilation",
                 "flicker_compensation", "weight_red", "weight_green", "weight_blue"]
        self.assertEqual([x for x in names if x in order], order)
        labels = {g.params[0]: g.label for g in knob_layout("SmartVector")}
        self.assertEqual([labels[n] for n in order[1:]], ["Strength", "Matte Channel", "Output", "Inpaint Matte Region",
                         "Matte Dilation", "Flicker Compensation", "Weight Red", "Weight Green", "Weight Blue"])

    def test_a_document_saved_before_keeps_rec709_weights(self):
        old = copy.deepcopy(build().document)
        old["version"] = 30
        for name in NEW:
            old["nodes"]["sv"]["params"].pop(name)
        up = upgrade_document(old)
        self.assertEqual(up["version"], SCHEMA_VERSION)
        p = up["nodes"]["sv"]["params"]
        self.assertEqual((p["weight_red"], p["weight_green"], p["weight_blue"]), svk.REC709)
        self.assertEqual((p["match_strength"], p["inpainting"], p["flicker_compensation"]), (1.5, 0, 0))

    def test_saved_nodes_estimate_exactly_as_before(self):
        # Rec. 709 weights hand the frames to the solver untouched, strength 1.5 leaves smoothness alone, and the switches are off
        frames = [moving(f) for f in (1, 2, 3)]
        self.assertIs(svk.weighted(frames, {"weight_red": .2126, "weight_green": .7152, "weight_blue": .0722}), frames)
        self.assertEqual(svk.effective_smoothness(0.5, 1.5), 0.5)
        old = copy.deepcopy(build().document)
        old["version"] = 30
        for name in NEW:
            old["nodes"]["sv"]["params"].pop(name)
        explicit = build({"weight_red": .2126, "weight_green": .7152, "weight_blue": .0722}).document
        np.testing.assert_array_equal(vectors(upgrade_document(old), moving), vectors(explicit, moving))


class WeightTests(unittest.TestCase):
    @staticmethod
    def red_only(frame):
        """Only the red channel moves; green and blue hold one still texture."""
        out = texture(0)
        out[..., 0] = texture(2 * (frame - 1))[..., 0]
        return out

    def test_a_plate_that_moves_in_one_channel_is_seen_only_through_that_weight(self):
        seen = vectors(build({"weight_red": 1.0, "weight_green": 0.0, "weight_blue": 0.0}).document, self.red_only)
        blind = vectors(build({"weight_red": 0.0, "weight_green": 0.6, "weight_blue": 0.1}).document, self.red_only)
        seen_motion = float(np.abs(seen[4:-4, 8:-8, 0]).mean())
        blind_motion = float(np.abs(blind[4:-4, 8:-8, 0]).mean())
        self.assertGreater(seen_motion, 1.5)
        self.assertLess(blind_motion, 0.5 * seen_motion)

    def test_the_default_weights_are_the_guide_s_and_differ_from_rec709(self):
        new = vectors(build().document, self.red_only)
        old = vectors(build({"weight_red": .2126, "weight_green": .7152, "weight_blue": .0722}).document, self.red_only)
        self.assertFalse(np.array_equal(new, old))


class FlickerTests(unittest.TestCase):
    @staticmethod
    def flickering(frame):
        out = moving(frame)
        out[..., :3] *= {1: 1.0, 2: 1.6, 3: 0.55}[frame]
        return out

    def test_a_brightness_step_pulls_the_vectors_away_from_the_motion_unless_compensated(self):
        truth = vectors(build().document, moving)[4:-4, 8:-8, :2]
        off = vectors(build().document, self.flickering)[4:-4, 8:-8, :2]
        on = vectors(build({"flicker_compensation": 1}).document, self.flickering)[4:-4, 8:-8, :2]
        error_off = float(np.abs(off - truth).mean())
        error_on = float(np.abs(on - truth).mean())
        self.assertLess(error_on, 0.5 * error_off)

    def test_compensation_scales_each_frame_to_the_reference_mean(self):
        frames = [self.flickering(f) for f in (1, 2, 3)]
        out = svk.flicker_compensated(frames, 0)
        means = [float(f[..., :3].mean()) for f in out]
        self.assertAlmostEqual(means[1], means[0], places=4)
        self.assertAlmostEqual(means[2], means[0], places=4)
        self.assertIs(out[0], frames[0])

    def test_with_no_brightness_change_the_switch_leaves_the_vectors_close(self):
        a = vectors(build().document, moving)
        b = vectors(build({"flicker_compensation": 1}).document, moving)
        self.assertLess(float(np.abs(a - b).mean()), 0.1)


class StrengthTests(unittest.TestCase):
    @staticmethod
    def noisy(frame):
        rng = np.random.default_rng(3)
        out = moving(frame)
        out[..., :3] += rng.normal(0, 0.15, (H, W, 1)).astype(np.float32) * (1 + 0.2 * frame)
        return out

    @staticmethod
    def roughness(v):
        field = v[4:-4, 6:-6, :2]
        return float(np.abs(np.diff(field, axis=0)).mean() + np.abs(np.diff(field, axis=1)).mean())

    def test_a_lower_strength_gives_a_smoother_field(self):
        rough = {s: self.roughness(vectors(build({"match_strength": s, "smoothness": 2.0}).document, self.noisy)) for s in (0.3, 1.5)}
        self.assertLess(rough[0.3], rough[1.5])

    def test_a_higher_strength_changes_the_vectors_too(self):
        # the solver's smoothing floors out below a smoothness of about one, so the field does not keep getting rougher above 1.5; it moves
        a = vectors(build({"match_strength": 1.5, "smoothness": 2.0}).document, self.noisy)
        b = vectors(build({"match_strength": 6.0, "smoothness": 2.0}).document, self.noisy)
        self.assertFalse(np.array_equal(a, b))

    def test_the_default_strength_is_the_solver_as_it_was(self):
        self.assertEqual(svk.effective_smoothness(1.25, 1.5), 1.25)
        self.assertGreater(svk.effective_smoothness(1.0, 0.75), 1.0)
        self.assertLess(svk.effective_smoothness(1.0, 3.0), 1.0)
        self.assertLessEqual(svk.effective_smoothness(1.0, 0.1), svk.MAX_SMOOTHNESS)


class InpaintTests(unittest.TestCase):
    """MatteFixture: the left half (columns below W/2) is the matte. Output Background leaves it absent."""

    @staticmethod
    def go(extra, output="Background", frame=3, plate=MatteFixture.plate):
        return vectors(build(extra, "Matte Alpha", output).document, plate, frame=frame)

    def test_inpainting_off_leaves_the_absent_side_empty_whatever_the_dilation(self):
        a = self.go({"matte_dilation": 50})
        b = self.go({"matte_dilation": 3})
        np.testing.assert_array_equal(a, b)
        self.assertEqual(float(a[:, :W // 2, 3].max()), 0.0)

    def test_inpainting_fills_the_absent_side_from_the_nearest_present_vectors(self):
        plain = self.go({})
        filled = self.go({"inpainting": 1, "matte_dilation": 0})
        self.assertEqual(float(filled[..., 3].min()), 1.0)                       # nothing is absent now
        np.testing.assert_array_equal(filled[:, W // 2:], plain[:, W // 2:])    # the present side is untouched
        expected = np.broadcast_to(plain[:, W // 2:W // 2 + 1, :2], (H, W // 2, 2))
        np.testing.assert_array_equal(filled[:, :W // 2, :2], expected)

    def test_matte_dilation_grows_the_infilled_region_by_that_many_pixels(self):
        n = 3
        zero = self.go({"inpainting": 1, "matte_dilation": 0}, plate=moving)       # the whole plate moves, so every column's vector differs
        grown = self.go({"inpainting": 1, "matte_dilation": n}, plate=moving)
        # the matte's last column is W/2 - 1, so a dilation of n absorbs columns W/2 to W/2 + n - 1
        np.testing.assert_array_equal(grown[:, W // 2 + n:], zero[:, W // 2 + n:])           # beyond the growth: as before
        edge = zero[:, W // 2 + n:W // 2 + n + 1, :2]                                       # the first untouched column
        np.testing.assert_array_equal(grown[:, W // 2:W // 2 + n, :2], np.broadcast_to(edge, (H, n, 2)))
        np.testing.assert_array_equal(grown[:, :W // 2, :2], np.broadcast_to(edge, (H, W // 2, 2)))
        self.assertFalse(np.array_equal(grown[:, W // 2], zero[:, W // 2]))

    def test_the_foreground_output_fills_the_background_the_same_way(self):
        plain = self.go({}, "Foreground")
        filled = self.go({"inpainting": 1, "matte_dilation": 0}, "Foreground")
        self.assertEqual(float(filled[..., 3].min()), 1.0)
        expected = np.broadcast_to(plain[:, W // 2 - 1:W // 2, :2], (H, W - W // 2, 2))
        np.testing.assert_array_equal(filled[:, W // 2:, :2], expected)

    def test_without_a_matte_channel_inpainting_has_nothing_to_fill(self):
        a = vectors(build({"inpainting": 1, "matte_dilation": 5}).document, moving)
        b = vectors(build().document, moving)
        np.testing.assert_array_equal(a, b)

    def test_the_plan_is_none_when_nothing_is_absent_or_present(self):
        calc = Evaluator._distance_transform
        self.assertIsNone(svk.inpaint_plan(np.ones((4, 4), np.float32), 3, calc))
        self.assertIsNone(svk.inpaint_plan(np.zeros((4, 4), np.float32), 3, calc))


if __name__ == "__main__":
    unittest.main()
