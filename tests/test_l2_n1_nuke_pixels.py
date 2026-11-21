"""Lane 2 step N1: Grade's gamma and ChannelMerge's union, xor, multiply and divide match Nuke 17.0v6's own renders.

The numbers below were read from Nuke's renders of the fidelity pack (the EXRs are not in git). Schema 39 holds older documents on the
arithmetic they had (`legacy_gamma_power` 1, `older_channel_maths` 1)."""
import copy
import unittest

import numpy as np

from nodebased.core import SCHEMA_VERSION, SPECS, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from tests.test_2d_behaviour_keyer import evaluator_pixels, two_plate_graph, two_plate_tiles
from tests.test_2d_parity_group_c4 import Graph


def grade(pixels, **knobs):
    return Evaluator._windowed_kernel("Grade", {**SPECS["Grade"]["params"], **knobs}, [Raster.of(np.asarray(pixels, np.float32))]).pixels


def row(*reds):
    """A one-row image whose red channel is `reds` (alpha 1)."""
    image = np.zeros((1, len(reds), 4), np.float32)
    image[..., 0] = reds
    image[..., 3] = 1.0
    return image


class GradeGammaMatchesNukeTests(unittest.TestCase):
    """Nuke's Grade follows the gamma power up to 1 and continues along the tangent at 1 above it (render of grade_reverse and probes)."""

    def test_forward_gamma_above_one_is_linear(self):
        out = grade(row(0.4, 1.0, 12.0, -0.5), gamma=1.8, black_clamp=0)[0, :, 0]
        np.testing.assert_allclose(out, [0.60107, 1.0, 7.11111, -0.5], rtol=1e-4)           # Nuke: 0.60107, 1, 7.11111, -0.5
        out = grade(row(12.0), gamma=0.5, black_clamp=0)[0, :, 0]
        np.testing.assert_allclose(out, [23.0], rtol=1e-5)                                  # Nuke: 1 + 11 / 0.5

    def test_reverse_gamma_above_one_is_linear(self):
        out = grade(row(0.4, 1.0, 12.0, -0.5), gamma=1.8, reverse=1, black_clamp=0)[0, :, 0]
        np.testing.assert_allclose(out, [0.19218, 1.0, 20.8, -0.5], rtol=1e-4)              # Nuke: 0.19218, 1, 20.8, -0.5
        np.testing.assert_allclose(grade(row(12.0), gamma=0.5, reverse=1, black_clamp=0)[0, :, 0], [6.5], rtol=1e-5)

    def test_grade_reverse_case_value(self):
        out = grade(row(12.0), multiply=1.5, offset=0.1, blackpoint=0.05, gamma=1.8, reverse=1, black_clamp=0)[0, 0, 0]
        self.assertAlmostEqual(float(out), 13.16, places=3)                                  # Nuke render of grade_reverse: 13.159999

    def test_split_gamma_follows_the_same_curve(self):
        out = grade(row(12.0), gamma=[1.8, 1.8, 1.8, 1.0], black_clamp=0)[0, 0, 0]
        self.assertAlmostEqual(float(out), 7.11111, places=3)

    def test_an_older_node_keeps_the_plain_power(self):
        out = grade(row(12.0), gamma=1.8, black_clamp=0, legacy_gamma_power=1)[0, 0, 0]
        self.assertAlmostEqual(float(out), 12.0 ** (1 / 1.8), places=4)

    def test_tiles_equal_the_evaluator(self):
        from tests.test_2d_fidelity_e_grade import GradeTileTests
        probe = GradeTileTests("test_tiles_equal_the_evaluator_with_a_mask_and_a_mix")
        for knobs in ({"gamma": 1.9, "black_clamp": 0}, {"gamma": 0.6, "reverse": 1, "multiply": 1.4, "offset": 0.1, "black_clamp": 0},
                      {"gamma": 1.5, "legacy_gamma_power": 1}):
            with self.subTest(knobs=knobs):
                probe.compare(knobs)
        probe.compare({"gamma": 1.5, "white_clamp": 1}, with_mask=True, mix=0.6)

    def test_gpu_mapping_declines_the_new_gamma(self):
        from nodebased import gpu2d
        base = {**SPECS["Grade"]["params"], "gamma": 1.8}
        with self.assertRaises(gpu2d.Unsupported):
            gpu2d.map_grade(base)
        self.assertEqual(gpu2d.map_grade({**base, "legacy_gamma_power": 1})["mode"], 1)


A_R = [-0.5, 1.5, -0.5, 0.0, 1.0, 1.5]
B_R = [0.05, 0.05, -0.5, -0.5, 0.75, 4.0]
NUKE = {            # Nuke's render of channelmerge_<op> at the same pixels (A in, B in, red out)
    "union": [0.05, 1.0, -0.5, -0.5, 1.0, 1.0],
    "xor": [0.05, 0.9499999, 0.0, 0.0, 0.25, 0.0],
    "multiply": [-0.025, 0.075, 0.25, -0.0, 0.75, 6.0],
    "divide": [-10.0, 30.0, -0.5, 0.0, 1.3333334, 0.375],
}


class ChannelMergeMatchesNukeTests(unittest.TestCase):
    def merge(self, op, **extra):
        a, b = row(*A_R), row(*B_R)
        params = {**SPECS["ChannelMerge"]["params"], "channel_operation": op, "a_channel": "A.r", "b_channel": "B.r", "out_channel": "R", **extra}
        return Evaluator._channel_merge(a, b, params)[0, :, 0]

    def test_each_operation_gives_nukes_pixels(self):
        for op, expected in NUKE.items():
            with self.subTest(op=op):
                np.testing.assert_allclose(self.merge(op), expected, rtol=1e-5, atol=1e-6)

    def test_an_older_node_keeps_its_formulas(self):
        a, b = np.array(A_R, np.float32), np.array(B_R, np.float32)
        old = {"union": a + b - a * b, "xor": a + b - 2 * a * b, "multiply": np.where((a < 0) & (b < 0), a, a * b)}
        for op, expected in old.items():
            with self.subTest(op=op):
                np.testing.assert_allclose(self.merge(op, older_channel_maths=1), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(float(self.merge("divide", older_channel_maths=1)[2]), 0.0)         # both negative: the old divide gave 0

    def test_new_node_defaults_and_old_defaults(self):
        self.assertEqual(SPECS["ChannelMerge"]["params"]["older_channel_maths"], 0)
        self.assertEqual(SPECS["Grade"]["params"]["legacy_gamma_power"], 0)

    def test_tiles_equal_the_whole_frame(self):
        two_plate_tiles("ChannelMerge", [{"channel_operation": op, "a_channel": "A.r", "b_channel": "B.g", "out_channel": "R"} for op in NUKE], self)


class SchemaMigrationTests(unittest.TestCase):
    def old_document(self):
        g = Graph()
        g.add("src", "Ramp", dict(width=20, height=12, p1_x=19.0, color0_red=-0.5, color1_red=3.0, color0_alpha=0.4, color1_alpha=1.0))
        g.add("grade", "Grade", dict(gamma=1.7, black_clamp=0), image="src")
        g.add("other", "Ramp", dict(width=20, height=12, color0_red=0.8, color1_red=-0.3))
        g.add("cm", "ChannelMerge", dict(channel_operation="union", a_channel="A.r", b_channel="B.r", out_channel="R"), A="src", B="other")
        doc = copy.deepcopy(g.doc)
        for name in ("grade", "cm"):
            for flag in ("legacy_gamma_power", "older_channel_maths"):
                doc["nodes"][name]["params"].pop(flag, None)
        doc["version"] = 38
        return doc

    def test_a_document_saved_at_schema_38_is_flagged_and_renders_the_old_pixels(self):
        doc = upgrade_document(self.old_document())
        self.assertEqual(doc["version"], SCHEMA_VERSION)
        self.assertEqual(doc["nodes"]["grade"]["params"]["legacy_gamma_power"], 1)
        self.assertEqual(doc["nodes"]["cm"]["params"]["older_channel_maths"], 1)
        validate(doc)
        source = evaluator_pixels(doc, "src")
        graded = evaluator_pixels(doc, "grade")
        expected = np.where(source[..., :3] > 0, np.maximum(source[..., :3], 0) ** (1 / 1.7), source[..., :3])
        np.testing.assert_allclose(graded[..., :3], expected, rtol=1e-5, atol=1e-6)
        a, b = evaluator_pixels(doc, "src")[..., 0], evaluator_pixels(doc, "other")[..., 0]
        np.testing.assert_allclose(evaluator_pixels(doc, "cm")[..., 0], a + b - a * b, rtol=1e-5, atol=1e-6)

    def test_a_node_made_now_follows_nuke(self):
        doc = self.old_document()
        doc["version"] = SCHEMA_VERSION
        for name, flag in (("grade", "legacy_gamma_power"), ("cm", "older_channel_maths")):
            doc["nodes"][name]["params"][flag] = 0
        graded = evaluator_pixels(doc, "grade")
        source = evaluator_pixels(doc, "src")
        high = source[..., :3] > 1
        self.assertTrue(high.any())
        np.testing.assert_allclose(graded[..., :3][high], (1 + (source[..., :3] - 1) / 1.7)[high], rtol=1e-5)


# ================================================================================================================ Gamma, Toe, Exposure, PLogLin
def pixels_of(reds, green=0.0, blue=0.0, alpha=1.0):
    image = np.zeros((1, len(reds), 4), np.float32)
    image[..., 0], image[..., 1], image[..., 2], image[..., 3] = reds, green, blue, alpha
    return image


class NegativeValuesMatchNukeTests(unittest.TestCase):
    def test_gamma_leaves_a_negative_value_alone(self):
        out = Evaluator._channel_gamma(pixels_of([-0.5, 0.4, 12.0]), {**SPECS["Gamma"]["params"], "gamma": 1.8, "channels": "rgb"})[0, :, 0]
        np.testing.assert_allclose(out, [-0.5, 0.60107, 3.9769], rtol=1e-4)                 # Nuke: -0.5, 0.60107, 3.9769 (a plain power above 1)

    def test_an_older_gamma_keeps_sign_times_the_power(self):
        out = Evaluator._channel_gamma(pixels_of([-0.5]), {**SPECS["Gamma"]["params"], "gamma": 2.2, "channels": "rgb", "older_negatives": 1})[0, 0, 0]
        self.assertAlmostEqual(float(out), -(0.5 ** (1 / 2.2)), places=5)

    def test_toe_sets_a_negative_value_to_zero(self):
        for lift in (0.0, 0.1):
            with self.subTest(lift=lift):
                out = Evaluator._toe(pixels_of([-0.5, 0.18]), {**SPECS["Toe"]["params"], "toe_lift": lift})[0, :, 0]
                self.assertEqual(float(out[0]), 0.0)                                        # Nuke: -0.5 becomes 0, with or without lift
        self.assertAlmostEqual(float(out[1]), 0.181, places=5)                              # Nuke at lift 0.1: 0.181

    def test_an_older_toe_keeps_a_negative_value(self):
        out = Evaluator._toe(pixels_of([-0.5]), {**SPECS["Toe"]["params"], "toe_lift": 0.1, "older_negatives": 1})[0, 0, 0]
        self.assertEqual(float(out), -0.5)


class ExposureBlackpointMatchesNukeTests(unittest.TestCase):
    def exposure(self, reds, **knobs):
        return Evaluator._exposure(pixels_of(reds), {**SPECS["Exposure"]["params"], "red": 1.0, "gang": 1, **knobs})[0, :, 0]

    def test_the_black_point_is_lifted_by_the_gain(self):
        out = self.exposure([-0.5, 0.0, 0.4, 1.5], blackpoint=0.3)
        np.testing.assert_allclose(out, [-0.57143, 0.42857, 1.22857, 3.42857], atol=2e-5)     # Nuke's render at one stop, black point 0.3
        np.testing.assert_allclose(self.exposure([-0.5, 0.18, 0.75], red=-1.0, blackpoint=0.1), [-0.3056, 0.0344, 0.3194], atol=1e-4)

    def test_at_a_gain_of_one_the_black_point_does_nothing(self):
        np.testing.assert_allclose(self.exposure([-0.5, 0.4, 12.0], red=0.0, blackpoint=0.3), [-0.5, 0.4, 12.0], atol=1e-6)    # Nuke's render

    def test_cineon_steps_ninety_code_values_a_stop(self):
        out = self.exposure([-0.5, 0.18], exposure_colorspace="Cineon")
        np.testing.assert_allclose(out, [-0.412, 0.268], atol=1e-4)                          # Nuke: -0.412 and 0.268

    def test_an_older_exposure_subtracts_the_black_point_first(self):
        out = self.exposure([0.4], blackpoint=0.3, older_exposure_maths=1)
        self.assertAlmostEqual(float(out[0]), (0.4 - 0.3) * 2.0, places=5)


class PLogLinAlphaMatchesNukeTests(unittest.TestCase):
    def run_node(self, **knobs):
        return Evaluator._ploglin(pixels_of([0.5], 0.2, 0.3, 0.25), {**SPECS["PLogLin"]["params"], **knobs})[0, 0]

    def test_a_new_node_converts_rgb_and_leaves_alpha_alone(self):
        self.assertEqual(SPECS["PLogLin"]["params"]["channels"], "rgb")
        out = self.run_node()
        self.assertEqual(float(out[3]), 0.25)                                                # Nuke's render: alpha passes through
        self.assertNotAlmostEqual(float(out[0]), 0.5, places=3)

    def test_channels_all_leaves_alpha_alone_too(self):
        self.assertEqual(float(self.run_node(channels="all")[3]), 0.25)               # Nuke's render with channels rgba or all: alpha 0.25 in, 0.25 out

    def test_an_older_node_converts_alpha(self):
        self.assertNotAlmostEqual(float(self.run_node(channels="all", older_alpha=1)[3]), 0.25, places=3)


class SchemaMigrationNegativesTests(unittest.TestCase):
    KINDS = {"Gamma": {"gamma": 2.2, "channels": "rgb"}, "Toe": {"toe_lift": 0.1}, "Exposure": {"red": 1.0, "gang": 1, "blackpoint": 0.3},
             "PLogLin": {}}

    def test_a_schema_38_document_keeps_its_pixels(self):
        g = Graph()
        g.add("src", "Ramp", dict(width=20, height=12, p1_x=19.0, color0_red=-0.5, color1_red=2.0, color0_green=0.1, color1_green=0.9,
                                  color0_alpha=0.3, color1_alpha=1.0,
                                  legacy_draw_blend=1))      # the older Ramp (before lane 9's Draw change, schema 40) keeps the negative red this test needs
        for kind, params in self.KINDS.items():
            g.add(kind.lower(), kind, params, image="src")
        doc = copy.deepcopy(g.doc)
        for kind in self.KINDS:
            for flag in ("older_negatives", "older_exposure_maths", "older_alpha"):
                doc["nodes"][kind.lower()]["params"].pop(flag, None)
            if kind == "PLogLin":
                doc["nodes"]["plogLin".lower()]["params"].pop("channels", None)
        doc["version"] = 38
        doc = upgrade_document(doc)
        validate(doc)
        self.assertEqual(doc["nodes"]["gamma"]["params"]["older_negatives"], 1)
        self.assertEqual(doc["nodes"]["toe"]["params"]["older_negatives"], 1)
        self.assertEqual(doc["nodes"]["exposure"]["params"]["older_exposure_maths"], 1)
        self.assertEqual(doc["nodes"]["ploglin"]["params"]["channels"], "all")
        self.assertEqual(doc["nodes"]["ploglin"]["params"]["older_alpha"], 1)
        src = evaluator_pixels(doc, "src")
        np.testing.assert_allclose(evaluator_pixels(doc, "gamma")[..., 0], np.sign(src[..., 0]) * np.abs(src[..., 0]) ** (1 / 2.2), rtol=1e-5, atol=1e-6)
        self.assertTrue((evaluator_pixels(doc, "toe")[..., 0] < 0).any())
        np.testing.assert_allclose(evaluator_pixels(doc, "exposure")[..., 0], (src[..., 0] - 0.3) * 2.0, rtol=1e-5, atol=1e-6)
        self.assertFalse(np.allclose(evaluator_pixels(doc, "ploglin")[..., 3], src[..., 3]))      # an older PLogLin converted alpha

    def test_tiles_equal_the_whole_frame(self):
        from tests.test_2d_behaviour_color_a import TilesMatchTests, plate_graph, tiled
        for kind, params in (("Gamma", {"gamma": 1.7, "channels": "rgb"}), ("Toe", {"toe_lift": 0.2}), ("Exposure", {"red": 1.0, "gang": 1, "blackpoint": 0.2}),
                             ("PLogLin", {})):
            for masked in (False, True):
                g = plate_graph(kind, params, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


class KeyerAndExpressionMatchNukeTests(unittest.TestCase):
    def keyer(self, greens, rng):
        image = np.zeros((1, len(greens), 4), np.float32)
        image[..., 1] = greens
        params = {**SPECS["Keyer"]["params"], "keyer_operation": "green", "range_a": rng[0], "range_b": rng[1], "range_c": rng[2], "range_d": rng[3]}
        return Evaluator._keyer(image, params)[0, :, 3]

    def test_a_falling_edge_that_starts_above_one_is_dropped(self):
        # Nuke's render (constant green plates): range {-0.25 0.5 4 12.5} gives 0.73333 at 0.3 and 1 at 0.9, 2, 5 and 12;
        # {0.1 0.5 4 12.5} and {0.1 0.5 4 5} give 1 at 3 and 5; {0 0.1 2 100} gives 1 at 3 and 5.
        np.testing.assert_allclose(self.keyer([0.3, 0.9, 2.0, 5.0, 12.0], (-0.25, 0.5, 4.0, 12.5)), [0.73333, 1, 1, 1, 1], atol=1e-5)
        np.testing.assert_allclose(self.keyer([3.0, 5.0], (0.1, 0.5, 4.0, 5.0)), [1, 1])
        np.testing.assert_allclose(self.keyer([3.0, 5.0], (0.0, 0.1, 2.0, 100.0)), [1, 1])

    def test_a_falling_edge_at_or_below_one_runs_to_d(self):
        np.testing.assert_allclose(self.keyer([0.5, 1.0, 8.0], (0.0, 0.1, 0.2, 100.0)), [0.99699, 0.99198, 0.92184], atol=1e-5)       # Nuke's render
        np.testing.assert_allclose(self.keyer([3.0, 5.0], (-0.25, 0.5, 0.9, 12.5)), [0.81897, 0.64655], atol=1e-5)      # Nuke's render

    def test_an_older_keyer_runs_the_falling_edge_wherever_it_is(self):
        image = np.zeros((1, 1, 4), np.float32)
        image[..., 1] = 5.0
        params = {**SPECS["Keyer"]["params"], "keyer_operation": "green", "range_a": 0.1, "range_b": 0.5, "range_c": 4.0, "range_d": 12.5, "older_unclamped_range": 1}
        self.assertAlmostEqual(float(Evaluator._keyer(image, params)[0, 0, 3]), (12.5 - 5.0) / 8.5, places=5)

    def test_a_blank_expression_row_leaves_the_channel_alone(self):
        from nodebased import nukeexpr
        image = pixels_of([-0.5, 0.4], 0.18, 0.75, 0.3)
        params = {**SPECS["Expression"]["params"], "expr_r": "", "expr_g": "g", "expr_b": "", "expr_a": "a"}
        out = nukeexpr.evaluate_rows(image, None, params, 1)
        np.testing.assert_array_equal(out, image)                                              # Nuke's render: nothing changes
        zero = nukeexpr.evaluate_rows(image, None, {**params, "older_blank_expression": 1}, 1)
        self.assertEqual(float(zero[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
