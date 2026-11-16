"""2D behaviour, lane 9 step Q: Sharpen, Soften, Erode (filter), Convolve, EdgeDetect, Emboss, Noise, STMap and IDistort.

The scored pixel cases live in tools/nuke_fidelity_cases_filter.py; these tests hold what a case cannot say with one reference picture: the
direction of an overshoot, which of two detectors responds more, what turns over when the angle turns, a statistic over a large sample, a
document saved before a kernel changed, and (for the two uv nodes) that every paired knob changes pixels.
"""
import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import filterbehaviour
from nodebased.core import SPECS, Dispatcher, upgrade_document, validate
from nodebased.imaging import Evaluator

H, W = 12, 16


def kern(kind, *images, **knobs):
    params = dict(SPECS[kind]["params"], **knobs)
    slots = list(images) + [None] * (len(SPECS[kind]["inputs"]) + len(SPECS[kind].get("optional_inputs", [])) - len(images))
    return np.asarray(Evaluator._kernel(kind, params, slots), np.float64)


def step():
    out = np.ones((H, W, 4), np.float32)
    out[:, :W // 2, :3] = 0.2
    out[:, W // 2:, :3] = 0.8
    return out


def square():
    out = np.zeros((H, W, 4), np.float32)
    out[3:9, 4:12, :] = 1.0
    return out


class SharpenSoftenTests(unittest.TestCase):
    def test_sharpen_overshoots_either_side_of_a_step_and_leaves_the_flat_parts(self):
        out = kern("Sharpen", step(), sharpen_size=1.0, kernel_filter="box", sharpen_amount=1.0, minimum=-1e6, maximum=1e6)
        row = out[5, :, 0]
        self.assertLess(row[7], 0.2)                    # the dark side next to the edge goes darker
        self.assertGreater(row[8], 0.8)                 # the bright side goes brighter
        np.testing.assert_allclose(row[:6], 0.2, atol=1e-6)
        np.testing.assert_allclose(row[10:], 0.8, atol=1e-6)

    def test_soften_moves_the_two_sides_toward_each_other_and_never_past(self):
        out = kern("Soften", step(), soften_size=1.0, kernel_filter="box", amount=1.0, minimum=-1e6, maximum=1e6)
        row = out[5, :, 0]
        self.assertGreater(row[7], 0.2)
        self.assertLess(row[7], 0.8)
        self.assertLess(row[8], 0.8)
        self.assertGreater(row[8], 0.2)

    def test_sharpen_amount_grows_the_overshoot(self):
        small = kern("Sharpen", step(), sharpen_size=1.0, kernel_filter="box", sharpen_amount=0.5, minimum=-1e6, maximum=1e6)
        large = kern("Sharpen", step(), sharpen_size=1.0, kernel_filter="box", sharpen_amount=1.5, minimum=-1e6, maximum=1e6)
        self.assertLess(large[5, 7, 0], small[5, 7, 0])
        self.assertGreater(large[5, 8, 0], small[5, 8, 0])


class ErodeFilterTests(unittest.TestCase):
    def test_a_positive_size_shrinks_a_white_square_and_a_negative_size_grows_it(self):
        base = square()[..., 0].sum()
        shrunk = kern("ErodeFilter", square(), filter_size=1.0, filter_type="box", channels="all")[..., 0].sum()
        grown = kern("ErodeFilter", square(), filter_size=-1.0, filter_type="box", channels="all")[..., 0].sum()
        self.assertLess(shrunk, base)
        self.assertGreater(grown, base)

    def test_gaussian_is_the_smoothest_filter_at_a_fractional_size(self):
        """The guide: 'Box is the fastest to render, whereas Gaussian blur is the smoothest.'"""
        def roughness(kind):
            out = kern("ErodeFilter", square(), filter_size=2.5, filter_type=kind, channels="all")[..., 0]
            return float(np.abs(np.diff(out, axis=1)).max())
        self.assertLess(roughness("gaussian"), roughness("box"))


class ConvolveTests(unittest.TestCase):
    def test_a_filter_that_is_one_bright_pixel_at_its_middle_is_the_identity(self):
        flt = np.zeros((H, W, 4), np.float32)
        flt[H // 2, W // 2, 3] = 1.0
        rng = np.random.RandomState(3)
        image = rng.uniform(0, 1, (H, W, 4)).astype(np.float32)
        from nodebased.raster import Raster
        out = Evaluator._windowed_kernel("Convolve", dict(SPECS["Convolve"]["params"]), [Raster.of(image), Raster.of(flt)]).pixels
        np.testing.assert_allclose(out, image, atol=1e-6)


class EdgeDetectTests(unittest.TestCase):
    def response(self, image, method="Sobel", **knobs):
        return kern("EdgeDetect", image, edge_type=method, channels="all", **knobs)[..., 0]

    def test_a_step_gives_an_edge_on_both_sides_and_none_elsewhere(self):
        out = self.response(step())
        self.assertGreater(out[5, 7], 0.0)
        self.assertAlmostEqual(out[5, 7], out[5, 8], places=6)
        np.testing.assert_allclose(out[5, :6], 0.0, atol=1e-7)
        np.testing.assert_allclose(out[5, 10:], 0.0, atol=1e-7)

    def test_turning_the_picture_turns_the_response(self):
        image = step()
        turned = np.ascontiguousarray(image.transpose(1, 0, 2))      # the step now runs across the rows
        for method in ("Sobel", "Prewitt"):
            np.testing.assert_allclose(self.response(turned, method), self.response(image, method).T, atol=1e-6)

    def test_sobel_answers_more_strongly_than_prewitt_on_a_step(self):
        self.assertGreater(self.response(step(), "Sobel")[5, 7], self.response(step(), "Prewitt")[5, 7])
        self.assertAlmostEqual(self.response(step(), "Sobel")[5, 7] / self.response(step(), "Prewitt")[5, 7], 4.0 / 3.0, places=5)

    def test_the_threshold_takes_that_much_off_the_response(self):
        plain = self.response(step())[5, 7]
        self.assertAlmostEqual(self.response(step(), threshold=0.5)[5, 7], plain - 0.5, places=5)
        self.assertEqual(float(self.response(step(), threshold=100.0).max()), 0.0)

    def test_an_older_prewitt_node_keeps_the_response_divided_by_three(self):
        new = self.response(step(), "Prewitt")
        old = self.response(step(), "Prewitt", legacy_edge_prewitt=1)
        np.testing.assert_allclose(old * 3.0, new, atol=1e-6)

    def test_a_saved_node_without_the_flag_loads_with_it_on_and_a_new_node_has_it_off(self):
        self.assertEqual(SPECS["EdgeDetect"]["params"]["legacy_edge_prewitt"], 0)
        d = Dispatcher()
        d.execute(dict(op="create", id="src", type="Checker", params=dict(width=45, height=37, size=5)))
        d.execute(dict(op="create", id="e", type="EdgeDetect", params=dict(edge_type="Prewitt")))
        d.execute(dict(op="connect", id="e", input="image", source="src"))
        old = copy.deepcopy(d.document)
        old["nodes"]["e"]["params"].pop("legacy_edge_prewitt")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["e"]["params"]["legacy_edge_prewitt"], 1)
        self.assertEqual(d.document["nodes"]["e"]["params"]["legacy_edge_prewitt"], 0)


class EmbossTests(unittest.TestCase):
    def relief(self, **knobs):
        base = dict(emboss_type="traditional", edge_detector="sobel", edge_channels="rgb", width=1.0)
        return kern("Emboss", step(), **{**base, **knobs})[..., 0] - step()[..., 0]

    def test_turning_the_angle_half_a_turn_turns_the_relief_over(self):
        np.testing.assert_allclose(self.relief(angle=0.0), -self.relief(angle=180.0), atol=1e-6)
        self.assertGreater(float(np.abs(self.relief(angle=0.0)).max()), 0.0)

    def test_width_scales_the_relief(self):
        np.testing.assert_allclose(self.relief(angle=0.0, width=2.0), 2.0 * self.relief(angle=0.0, width=1.0), atol=1e-6)

    def test_the_relief_sits_on_the_two_pixels_beside_the_step(self):
        out = self.relief(angle=0.0)[5]
        self.assertNotEqual(float(out[7]), 0.0)
        self.assertNotEqual(float(out[8]), 0.0)
        np.testing.assert_allclose(out[:6], 0.0, atol=1e-7)
        np.testing.assert_allclose(out[10:], 0.0, atol=1e-7)

    def test_the_effect_type_does_not_contain_the_picture(self):
        light = kern("Emboss", step(), emboss_type="effect", edge_detector="sobel", angle=0.0, width=1.0)[..., 0]
        np.testing.assert_allclose(light[5, :6], light[5, 10:][:6], atol=1e-6)     # the flat parts are alike whatever the picture's brightness


class NoiseSampleTests(unittest.TestCase):
    def sample(self, **knobs):
        from tests.test_2d_behaviour_draw import shape
        base = dict(size=12.0, octaves=4, seed=3, center_x=48.0, center_y=48.0)
        return np.asarray(shape("Noise", w=96, h=96, **{**base, **knobs}), np.float64)[..., 3]

    def test_the_coverage_lies_between_0_and_1_and_has_a_spread(self):
        out = self.sample()
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreater(float(out.std()), 0.02)

    def test_a_lower_gamma_darkens_the_noise_and_a_higher_one_brightens_it(self):
        low, mid, high = (self.sample(gamma=g).mean() for g in (0.5, 1.0, 2.0))
        self.assertLess(low, mid)
        self.assertLess(mid, high)

    def test_inverting_swaps_the_noise_and_the_non_noise(self):
        np.testing.assert_allclose(self.sample(invert=1), 1.0 - self.sample(), atol=1e-6)

    def test_zoffset_moves_through_the_pattern(self):
        a, b = self.sample(z_slice=0.0), self.sample(z_slice=5.0)
        self.assertGreater(float(np.abs(a - b).mean()), 0.01)
        near = self.sample(z_slice=0.01)
        self.assertLess(float(np.abs(a - near).mean()), float(np.abs(a - b).mean()))


# ---------------------------------------------------------------------------------------------------------- STMap and IDistort
def uv_fixture(u, v, a=1.0):
    out = np.empty((H, W, 4), np.float32)
    out[:] = (u, v, 0.0, a)
    return out


def texture():
    rng = np.random.RandomState(5)
    return rng.uniform(0.0, 1.0, (H, W, 4)).astype(np.float32)


def run_uv(kind, image, uv, **knobs):
    from nodebased.raster import Raster
    params = dict(SPECS[kind]["params"], **knobs)
    return np.asarray(Evaluator._windowed_kernel(kind, params, [Raster.of(image), Raster.of(uv)]).pixels, np.float64)


class IDistortTests(unittest.TestCase):
    def test_the_guides_example_pixel_comes_from_one_left_and_five_up_in_nukes_axes(self):
        """'If pixel 51, 23 has a U and V value of -1, 5, the pixel's value will come from 50, 28': Nuke's y runs up, so five rows up the array."""
        src = texture()
        out = run_uv("IDistort", src, uv_fixture(-1.0, 5.0), uv_layer="", filter="nearest")
        np.testing.assert_allclose(out[8, 6], src[3, 5], atol=1e-6)

    def test_an_older_node_moves_the_picture_the_other_way_along_x_and_adds_the_offset(self):
        src = texture()
        out = run_uv("IDistort", src, uv_fixture(1.0, 0.0), uv_layer="", filter="nearest", legacy_idistort=1)
        np.testing.assert_allclose(out[5, 6], src[5, 5], atol=1e-6)          # a positive u carried the picture right
        offset = run_uv("IDistort", src, uv_fixture(0.0, 0.0), uv_layer="", filter="nearest", legacy_idistort=1, uv_offset_x=1.0)
        np.testing.assert_allclose(offset[5, 6], src[5, 5], atol=1e-6)       # and the offset was added
        new = run_uv("IDistort", src, uv_fixture(0.0, 0.0), uv_layer="", filter="nearest", uv_offset_x=1.0)
        np.testing.assert_allclose(new[5, 6], src[5, 5], atol=1e-6)          # the guide subtracts it: u = -1, a pixel comes from the left

    def test_a_saved_node_without_the_flag_loads_with_it_on_and_a_new_node_has_it_off(self):
        self.assertEqual(SPECS["IDistort"]["params"]["legacy_idistort"], 0)
        self.assertEqual(SPECS["IDistort"]["params"]["premultiplied"], 0)
        d = Dispatcher()
        d.execute(dict(op="create", id="src", type="Checker", params=dict(width=45, height=37, size=5)))
        d.execute(dict(op="create", id="n", type="IDistort", params=dict(uv_layer="x")))
        old = copy.deepcopy(d.document)
        old["nodes"]["n"]["params"].pop("legacy_idistort")
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["n"]["params"]["legacy_idistort"], 1)

    def test_premultiplied_divides_the_vectors_by_the_maps_alpha_and_the_older_node_ignores_it(self):
        src = texture()
        uv = uv_fixture(-1.0, 0.0, a=0.5)                    # stored premultiplied, so the true u is -2
        plain = run_uv("IDistort", src, uv, uv_layer="", filter="nearest")
        undone = run_uv("IDistort", src, uv, uv_layer="", filter="nearest", premultiplied=1)
        np.testing.assert_allclose(plain[5, 6], src[5, 5], atol=1e-6)
        np.testing.assert_allclose(undone[5, 6], src[5, 4], atol=1e-6)
        older = run_uv("IDistort", src, uv, uv_layer="", filter="nearest", premultiplied=1, legacy_idistort=1)
        np.testing.assert_allclose(older, run_uv("IDistort", src, uv, uv_layer="", filter="nearest", premultiplied=0, legacy_idistort=1))

    def test_the_blur_channel_times_blur_scale_blurs_the_result(self):
        src = texture()
        uv = uv_fixture(0.0, 0.0)
        uv[..., 2] = 1.0
        sharp = run_uv("IDistort", src, uv_fixture(0.0, 0.0), uv_layer="", filter="nearest")
        blurred = run_uv("IDistort", src, uv, uv_layer="", filter="nearest", blur="rgba.blue", blur_scale=1.0)
        self.assertGreater(float(np.abs(blurred - sharp).max()), 0.01)
        np.testing.assert_allclose(run_uv("IDistort", src, uv, uv_layer="", filter="nearest", blur="rgba.blue", blur_scale=0.0), sharp, atol=1e-6)


class STMapTests(unittest.TestCase):
    def test_each_pixel_takes_the_source_at_the_position_the_map_holds(self):
        src = texture()
        out = run_uv("STMap", src, uv_fixture(5.5 / W, 1.0 - 4.5 / H), uv_layer="", filter="nearest")
        np.testing.assert_allclose(out, np.broadcast_to(src[4, 5], out.shape), atol=1e-6)

    def test_premultiplied_divides_the_map_by_its_alpha(self):
        src = texture()
        out = run_uv("STMap", src, uv_fixture(0.5 * 5.5 / W, 0.5 * (1.0 - 4.5 / H), a=0.5), uv_layer="", filter="nearest", premultiplied=1)
        np.testing.assert_allclose(out, np.broadcast_to(src[4, 5], out.shape), atol=1e-6)


class ProbeTests(unittest.TestCase):
    def test_every_paired_knob_of_the_two_uv_nodes_changes_pixels(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        import audit_nuke_fidelity as audit
        import nuke_fidelity_cases as ruler
        import nuke_fidelity_requirements as req
        mapping = audit.json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]
        fx = ruler.make_fixtures()
        for key in ("STMap", "IDistort"):
            pair = mapping[key]
            entries = [e for e in pair["knobs"] if e["kind"] in req.PAIRED]
            wanted = sorted({p for e in entries for p in e["nodebased"]} - {"uv_layer", "invert_mask", "maskChannel", "channels", "mix"})        # the layer name and the shared mask chrome are not the node's own knobs
            acts = req.probe_acts(pair["nodebased"], wanted, fx)
            self.assertIsNotNone(acts, key)
            self.assertEqual([n for n, ok in acts.items() if not ok], [], key)


if __name__ == "__main__":
    unittest.main()
