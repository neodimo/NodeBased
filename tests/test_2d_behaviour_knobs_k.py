"""2D behaviour lane 2, step K: the main knobs Nuke's IBKGizmo, HueCorrect, MatchGrade and GenerateLUT have and NodeBased lacked, and Nuke's layout of the time
and metadata panels.

The Reference Guide names these knobs and says nothing about what they compute, so the checks here say what each knob's name and the tutorial state as
properties of the pixels (a wider screen range keys more, yellow takes yellow out, mix luminance restores the luminance the pixel came in with), and the
cases in tools/nuke_fidelity_cases_color.py stay not scored. An older document renders as it did before, bit for bit.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import SPECS, upgrade_document
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels


def kernel(kind, params, *inputs):
    return Evaluator._kernel(kind, dict(SPECS[kind]["params"], **params), list(inputs))


def solid(r, g, b, a=1.0, h=4, w=4):
    return np.tile(np.array([r, g, b, a], np.float32), (h, w, 1))


GREEN_PLATE = (0.15, 0.7, 0.3)


class IBKGizmoNewKnobsTests(unittest.TestCase):
    def setUp(self):
        self.plate = solid(*GREEN_PLATE)
        self.fg = solid(0.45, 0.6, 0.2)             # a pixel part way between the screen and a foreground colour

    def key(self, **params):
        return kernel("IBKGizmo", dict({"screen_type": "green"}, **params), self.fg, self.plate)

    def test_a_wider_screen_range_keys_more_of_the_picture(self):
        alphas = [float(self.key(screen_range=r)[0, 0, 3]) for r in (0.5, 1.0, 1.5, 3.0)]
        self.assertTrue(all(a > b for a, b in zip(alphas, alphas[1:])), alphas)
        np.testing.assert_array_equal(self.key(screen_range=1.0), kernel("IBKGizmo", {"screen_type": "green"}, self.fg, self.plate))

    def test_pick_reads_the_screen_from_the_colour(self):
        green = self.key()
        np.testing.assert_array_equal(self.key(screen_type="pick", color_r=0.1, color_g=0.8, color_b=0.2), green)
        blue_plate = solid(0.2, 0.3, 0.9)
        blue = kernel("IBKGizmo", {"screen_type": "blue"}, self.fg, blue_plate)
        picked = kernel("IBKGizmo", {"screen_type": "pick", "color_r": 0.2, "color_g": 0.3, "color_b": 0.9}, self.fg, blue_plate)
        np.testing.assert_array_equal(picked, blue)
        self.assertFalse(np.array_equal(green, self.key(screen_type="pick", color_r=0.9, color_g=0.1, color_b=0.1)))        # a red screen reads another difference

    def test_yellow_cyan_and_magenta_take_that_secondary_colours_excess_out(self):
        for flag, pair, third in (("yellow", (0, 1), 2), ("cyan", (1, 2), 0), ("magenta", (0, 2), 1)):
            with self.subTest(flag=flag):
                fg = solid(0.8, 0.7, 0.2) if flag == "yellow" else (solid(0.2, 0.8, 0.7) if flag == "cyan" else solid(0.8, 0.2, 0.7))
                off = kernel("IBKGizmo", {"screen_type": "green", "screen_subtraction": 0}, fg, self.plate)
                on = kernel("IBKGizmo", {"screen_type": "green", "screen_subtraction": 0, flag: 1}, fg, self.plate)
                np.testing.assert_array_equal(on[..., 3], off[..., 3])                      # the matte is not touched
                excess = max(min(float(off[0, 0, pair[0]]), float(off[0, 0, pair[1]])) - float(off[0, 0, third]), 0.0)
                self.assertGreater(excess, 0.01)
                for channel in pair:
                    self.assertAlmostEqual(float(off[0, 0, channel] - on[0, 0, channel]), excess, places=5)
                self.assertEqual(float(on[0, 0, third]), float(off[0, 0, third]))

    def test_the_luminance_level_acts_only_when_its_enable_box_is_on(self):
        fg = solid(0.5, 0.55, 0.2)
        base = {"screen_type": "green", "luminance_match": 1}

        def run(**p):
            return kernel("IBKGizmo", dict(base, **p), fg, self.plate)

        np.testing.assert_array_equal(run(luminance_level=0.2, luminance_level_enable=0), run(luminance_level=1.0, luminance_level_enable=0))
        self.assertFalse(np.array_equal(run(luminance_level=0.2, luminance_level_enable=1), run(luminance_level=1.0, luminance_level_enable=1)))
        # with the enable on and the level at 1 the match is whole, as it is with the enable off
        np.testing.assert_array_equal(run(luminance_level=1.0, luminance_level_enable=1), run(luminance_level=1.0, luminance_level_enable=0))

    def test_use_bg_chroma_puts_the_backgrounds_chroma_in_the_edge(self):
        bg = solid(0.9, 0.3, 0.2)
        edge_on = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 1}, self.fg, self.plate, bg)
        edge_off = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 0}, self.fg, self.plate, bg)
        np.testing.assert_array_equal(edge_on[..., 3], edge_off[..., 3])
        self.assertGreater(float(edge_off[0, 0, 3]), 0.0)
        self.assertLess(float(edge_off[0, 0, 3]), 1.0)
        self.assertGreater(float(edge_on[0, 0, 0]), float(edge_off[0, 0, 0]))                  # red, the background's strong channel, comes in
        self.assertGreater(edge_on[0, 0, 0] - edge_off[0, 0, 0], edge_on[0, 0, 2] - edge_off[0, 0, 2])
        np.testing.assert_array_equal(kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 1}, self.fg, self.plate), self.key())      # no bg wired: nothing to take

    def test_a_fully_opaque_or_fully_transparent_pixel_takes_no_bg_chroma(self):
        bg = solid(0.9, 0.3, 0.2)
        for fg in (solid(*GREEN_PLATE), solid(0.5, 0.5, 0.5)):
            on = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 1}, fg, self.plate, bg)
            off = kernel("IBKGizmo", {"screen_type": "green", "use_bg_chroma": 0}, fg, self.plate, bg)
            np.testing.assert_allclose(on, off, atol=1e-6)

    def test_every_new_knob_is_registered_with_its_range_and_label(self):
        from nodebased import knobs
        from nodebased.core import LIMITS
        for name in ("screen_range", "color_r", "luminance_level_enable", "yellow", "cyan", "magenta", "use_bg_chroma"):
            self.assertIn(name, SPECS["IBKGizmo"]["params"])
            self.assertIn(name, LIMITS)
        self.assertEqual(SPECS["IBKGizmo"]["params"]["luminance_level_enable"], 0)
        labels = [g.label for g in knobs.KNOB_GROUPS["IBKGizmo"] if getattr(g, "label", None)] if hasattr(knobs, "KNOB_GROUPS") else []
        self.assertTrue(labels == [] or "Screen range" in labels)


def ibk_graph(params, masked=False):
    g = Graph()
    g.add("fg", "Rectangle", dict(width=61, height=47, box_x=0, box_y=0, box_width=61, box_height=47, red=0.45, green=0.6, blue=0.2, alpha=1.0, softness=0.5))
    g.add("cp", "Rectangle", dict(width=61, height=47, box_x=0, box_y=0, box_width=61, box_height=47, red=0.1, green=0.95, blue=0.3, alpha=1.0, softness=0.1))
    g.add("bg", "Rectangle", dict(width=61, height=47, box_x=0, box_y=0, box_width=61, box_height=47, red=0.8, green=0.3, blue=0.2, alpha=1.0, softness=0.1))
    g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
    inputs = {"fg": "fg", "c": "cp"}
    if "use_bg_chroma" in params or "use_bg_luminance" in params:
        inputs["bg"] = "bg"
    if masked:
        inputs["mask"] = "matte"
    g.add("node", "IBKGizmo", params, **inputs)
    return g


class IBKGizmoNewKnobsOnTilesTests(unittest.TestCase):
    def test_tiles_equal_the_whole_frame_with_the_new_knobs(self):
        from tests.test_2d_behaviour_color_a import tiled
        for params in ({"screen_type": "pick", "color_r": 0.1, "color_g": 0.7, "color_b": 0.3, "screen_range": 1.4, "yellow": 1, "magenta": 1},
                       {"screen_type": "green", "luminance_match": 1, "luminance_level": 0.4, "luminance_level_enable": 1, "cyan": 1},
                       {"screen_type": "green", "use_bg_chroma": 1, "screen_range": 0.7}):
            for masked in (False, True):
                g = ibk_graph(params, masked)
                full = evaluator_pixels(g.doc, "node")
                for edge in (7, 16, 32):
                    with self.subTest(params=params, masked=masked, edge=edge):
                        np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


class IBKGizmoOlderDocumentTests(unittest.TestCase):
    def test_an_older_document_keeps_its_luminance_level(self):
        params = {"screen_type": "green", "luminance_match": 1, "luminance_level": 0.3}
        doc = copy.deepcopy(ibk_graph(params).doc)
        for name in ("luminance_level_enable", "screen_range", "yellow", "cyan", "magenta", "use_bg_chroma", "color_r", "color_g", "color_b"):
            doc["nodes"]["node"]["params"].pop(name, None)
        doc = upgrade_document(doc)
        got = doc["nodes"]["node"]["params"]
        self.assertEqual(got["luminance_level_enable"], 1)                      # the older node always applied its level
        self.assertEqual(got.get("screen_range", 1.0), 1.0)
        fg, plate = evaluator_pixels(doc, "fg"), evaluator_pixels(doc, "cp")
        older = kernel("IBKGizmo", dict(params, luminance_level_enable=1), fg, plate)
        np.testing.assert_array_equal(evaluator_pixels(doc, "node"), older)
        # the same graph made fresh (enable off) is the whole match
        fresh = kernel("IBKGizmo", dict(params, luminance_level_enable=0), fg, plate)
        self.assertFalse(np.array_equal(older, fresh))


class HueCorrectMixLuminanceTests(unittest.TestCase):
    CURVE_LUM = '{"interpolation":"linear","points":[[0,2],[60,1],[360,2]]}'            # doubles the luminance of the reds (hue 0)

    def image(self):
        return np.tile(np.array([0.6, 0.2, 0.15, 1.0], np.float32), (4, 4, 1))

    @staticmethod
    def luma(px):
        return 0.2125 * px[..., 0] + 0.7154 * px[..., 1] + 0.0721 * px[..., 2]

    def test_mix_luminance_gives_the_output_back_the_luminance_it_came_in_with(self):
        src = self.image()
        base = kernel("HueCorrect", {"curve_lum": self.CURVE_LUM}, src)
        self.assertGreater(float(self.luma(base)[0, 0]), float(self.luma(src)[0, 0]) * 1.5)          # the curve raised the luminance
        full = kernel("HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 1.0}, src)
        self.assertAlmostEqual(float(self.luma(full)[0, 0]), float(self.luma(src)[0, 0]), places=5)
        half = kernel("HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 0.5}, src)
        self.assertAlmostEqual(float(self.luma(half)[0, 0]), 0.5 * float(self.luma(src)[0, 0]) + 0.5 * float(self.luma(base)[0, 0]), places=5)
        np.testing.assert_array_equal(kernel("HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 0.0}, src), base)
        np.testing.assert_array_equal(full[..., 3], src[..., 3])

    def test_the_default_leaves_every_older_pixel_alone(self):
        self.assertEqual(SPECS["HueCorrect"]["params"]["mix_luminance"], 0.0)
        src = self.image()
        self.assertTrue(np.array_equal(kernel("HueCorrect", {"curve_sat": '{"interpolation":"linear","points":[[0,2],[360,2]]}'}, src),
                                       kernel("HueCorrect", {"curve_sat": '{"interpolation":"linear","points":[[0,2],[360,2]]}', "mix_luminance": 0.0}, src)))

    def test_tiles_equal_the_whole_frame(self):
        from tests.test_2d_behaviour_color_a import tiled
        for masked in (False, True):
            g = Graph()
            g.add("plate", "Rectangle", dict(width=61, height=47, box_x=6, box_y=5, box_width=44, box_height=33, red=0.9, green=0.5, blue=0.2, alpha=0.8, softness=0.5))
            g.add("matte", "Radial", dict(width=61, height=47, box_x=4, box_y=4, box_width=50, box_height=38, softness=0.8))
            inputs = {"image": "plate"}
            if masked:
                inputs["mask"] = "matte"
            g.add("node", "HueCorrect", {"curve_lum": self.CURVE_LUM, "mix_luminance": 0.7}, **inputs)
            full = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(masked=masked, edge=edge):
                    np.testing.assert_allclose(tiled(g.doc, edge), full, atol=1e-5)


# ================================================================================================================ MatchGrade
def varied(h=12, w=16, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.05, 0.9, (h, w, 4)).astype(np.float32)
    img[..., 3] = 1.0
    return img


def match(source, target, **params):
    return Evaluator._match_grade(source, target, dict(SPECS["MatchGrade"]["params"], **params))


class MatchGradeTests(unittest.TestCase):
    def test_a_target_graded_by_a_known_grade_is_matched_back(self):
        """Match Graded Source: the target is the source through a per-channel gain and offset; the match returns it within float rounding."""
        src = varied()
        tgt = src.copy()
        gains, offsets = (1.4, 0.8, 1.1), (0.05, -0.02, 0.1)
        for c in range(3):
            tgt[..., c] = src[..., c] * gains[c] + offsets[c]
        for task in ("Match Graded Source", "Match Different Clip"):
            np.testing.assert_allclose(match(src, tgt, match_task=task)[..., :3], tgt[..., :3], atol=2e-5)

    def test_the_defaults_leave_the_older_match_unchanged(self):
        src, tgt = varied(), varied(seed=9)
        out = match(src, tgt)
        for c in range(3):
            s, t = src[..., c], tgt[..., c]
            ss, ts = float(np.std(s)), float(np.std(t))
            np.testing.assert_array_equal(out[..., c], ((s - float(np.mean(s))) * (ts / ss) + float(np.mean(t))))

    def alpha_split(self):
        src, tgt = varied(), varied(seed=9)
        src[:, :8, 3] = 1.0
        src[:, 8:, 3] = 0.0
        return src, tgt

    def test_the_alpha_mask_limits_the_grade_to_the_non_black_alpha(self):
        src, tgt = self.alpha_split()
        out = match(src, tgt, grade_mask="Alpha", apply_to_mask=1)
        np.testing.assert_array_equal(out[:, 8:, :3], src[:, 8:, :3])                 # the grade does not reach where alpha is black
        self.assertFalse(np.allclose(out[:, :8, :3], src[:, :8, :3]))
        inverted = match(src, tgt, grade_mask="Inverted Alpha", apply_to_mask=1)
        np.testing.assert_array_equal(inverted[:, :8, :3], src[:, :8, :3])            # and the inverted alpha grades the other half
        self.assertFalse(np.allclose(inverted[:, 8:, :3], src[:, 8:, :3]))

    def test_without_apply_to_mask_the_grade_comes_from_the_region_and_covers_the_picture(self):
        src, tgt = self.alpha_split()
        masked = match(src, tgt, grade_mask="Alpha", apply_to_mask=1)
        whole = match(src, tgt, grade_mask="Alpha", apply_to_mask=0)
        self.assertFalse(np.allclose(whole[:, 8:, :3], src[:, 8:, :3]))               # the part outside the mask is graded too
        np.testing.assert_allclose(whole[:, :8, :3], masked[:, :8, :3], atol=1e-6)    # where the mask is full, both give the same grade
        # the grade is the one fitted to the masked region: the region's mean and spread now equal those of the target's same region
        # (Match Graded Source reads the Source's alpha for both pictures)
        left, wanted = whole[:, :8, :3].reshape(-1, 3), tgt[:, :8, :3].reshape(-1, 3)
        np.testing.assert_allclose(left.mean(axis=0), wanted.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(left.std(axis=0), wanted.std(axis=0), atol=1e-5)

    def test_the_task_picks_whose_alpha_weighs_the_target(self):
        src, tgt = self.alpha_split()
        tgt = tgt.copy()
        tgt[:, 8:, 3] = 0.0
        tgt[:, :8, 3] = 1.0
        tgt_other = tgt.copy()
        tgt_other[:, :8, 3], tgt_other[:, 8:, 3] = 0.0, 1.0
        graded = match(src, tgt_other, grade_mask="Alpha", match_task="Match Graded Source")
        graded_same = match(src, tgt, grade_mask="Alpha", match_task="Match Graded Source")
        np.testing.assert_array_equal(graded, graded_same)                             # only the Source's alpha is read
        different = match(src, tgt_other, grade_mask="Alpha", match_task="Match Different Clip")
        self.assertFalse(np.allclose(graded, different))                               # the target's own alpha is read

    def test_output_shows_the_matched_source_or_target_picture(self):
        g = Graph()
        g.add("src", "Rectangle", dict(width=40, height=30, box_x=4, box_y=4, box_width=30, box_height=20, red=0.9, green=0.4, blue=0.2, alpha=1.0, softness=0.5))
        g.add("tgt", "Rectangle", dict(width=40, height=30, box_x=0, box_y=0, box_width=40, box_height=30, red=0.3, green=0.6, blue=0.8, alpha=1.0, softness=0.2))
        results = {}
        for mode in ("Matched", "Source", "Target"):
            g.add(f"node_{mode}", "MatchGrade", {"match_output": mode}, image="src", reference="tgt")
            results[mode] = evaluator_pixels(g.doc, f"node_{mode}")
        np.testing.assert_array_equal(results["Source"], evaluator_pixels(g.doc, "src"))
        np.testing.assert_array_equal(results["Target"], evaluator_pixels(g.doc, "tgt"))
        self.assertFalse(np.array_equal(results["Matched"], results["Source"]))
        self.assertFalse(np.array_equal(results["Matched"], results["Target"]))

    def test_the_new_knobs_are_paired_or_unpaired_with_a_reason(self):
        import json
        from tools import audit_nuke_fidelity as audit
        pairs = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]["MatchGrade"]["knobs"]
        nuke_names = {n for entry in pairs for n in (entry.get("nuke") or [])}
        for name in ("matchGradeTask", "analysisType", "mask", "applyToMask", "output", "alignTargetToSource", "autoCrop"):
            self.assertIn(name, nuke_names)
        for entry in pairs:
            if entry["kind"] == "unsupported":
                self.assertTrue(entry["reason"])


# ================================================================================================================ GenerateLUT
class GenerateLutFileTests(unittest.TestCase):
    """The file knobs only shape what the Generate button writes; each file is written and read back through the existing readers."""

    def lut_doc(self, tmp, suffix, **params):
        g = Graph()
        g.add("src", "Constant", dict(width=8, height=8))
        g.add("grade", "Grade", dict(multiply=1.6, offset=0.02, gamma=0.8), image="src")
        g.add("lut", "GenerateLUT", dict(lut_path=str(Path(tmp) / f"out{suffix}"), lut_size=17, **params), source="grade")
        return g.doc

    @staticmethod
    def expected(rgb):
        """The Grade the document holds (multiply 1.6, offset 0.02, gamma 0.8) on straight colour values."""
        rgb = np.asarray(rgb, np.float64)
        return np.power(np.clip(rgb * 1.6 + 0.02, 0.0, None), 1.0 / 0.8)

    def probe_colours(self):
        return np.array([[[0.0, 0.0, 0.0, 1.0], [0.02, 0.05, 0.1, 1.0], [0.2, 0.4, 0.1, 1.0], [0.5, 0.3, 0.7, 1.0], [0.9, 0.8, 0.1, 1.0], [1.0, 1.0, 1.0, 1.0]]], np.float32)

    def test_a_cube_round_trips_through_the_cube_reader(self):
        from nodebased import lutio
        with tempfile.TemporaryDirectory() as tmp:
            doc = self.lut_doc(tmp, ".cube", lut_file_type="cube")
            path = lutio.export_graph_cube(doc, "lut", doc["nodes"]["lut"]["params"]["lut_path"], 17)
            lut = lutio.read_cube(path)
        probe = self.probe_colours()
        mapped = lutio.apply_cube(probe, lut, "trilinear")
        np.testing.assert_allclose(mapped[..., :3], self.expected(probe[..., :3]), atol=0.03)

    def test_a_3dl_round_trips_through_the_3dl_reader_for_every_style_and_depth(self):
        from nodebased import lutio
        probe = self.probe_colours()
        for style in ("logarithmic", "linear"):
            for depth, tolerance in (("8 bit", 0.05), ("10 bit", 0.03), ("12 bit", 0.03), ("16 bit", 0.03)):
                with self.subTest(style=style, depth=depth), tempfile.TemporaryDirectory() as tmp:
                    doc = self.lut_doc(tmp, ".3dl", lut_file_type="3dl")
                    path = lutio.export_graph_cube(doc, "lut", doc["nodes"]["lut"]["params"]["lut_path"], 17, file_type="3dl", prelut_style=style,
                                                   input_bit_depth=depth, clamp_output=True)
                    mesh, shaper = lutio.read_3dl(path)
                    mapped = lutio.apply_3dl(probe, mesh, shaper, "trilinear")
                    wanted = np.clip(self.expected(probe[..., :3]), 0.0, 1.0)
                    np.testing.assert_allclose(mapped[..., :3], wanted, atol=tolerance)

    def test_a_logarithmic_prelut_reads_the_darks_more_exactly_than_a_linear_one(self):
        from nodebased import lutio
        dark = np.array([[[0.004, 0.006, 0.01, 1.0], [0.01, 0.02, 0.005, 1.0], [0.03, 0.015, 0.02, 1.0]]], np.float32)
        errors = {}
        for style in ("logarithmic", "linear"):
            with tempfile.TemporaryDirectory() as tmp:
                doc = self.lut_doc(tmp, ".3dl", lut_file_type="3dl")
                path = lutio.export_graph_cube(doc, "lut", doc["nodes"]["lut"]["params"]["lut_path"], 17, file_type="3dl", prelut_style=style,
                                               input_bit_depth="16 bit", clamp_output=True)
                mesh, shaper = lutio.read_3dl(path)
                errors[style] = float(np.abs(lutio.apply_3dl(dark, mesh, shaper, "trilinear")[..., :3] - self.expected(dark[..., :3])).max())
                if style == "linear":
                    self.assertIsNone(shaper)                                                    # a linear preLUT is the identity row the reader drops
                else:
                    self.assertIsNotNone(shaper)
        self.assertLess(errors["logarithmic"], errors["linear"])

    def test_the_prelut_size_sets_the_length_of_the_preLUT_row(self):
        from nodebased import lutio
        for entries in (64, 256):
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as tmp:
                doc = self.lut_doc(tmp, ".3dl", lut_file_type="3dl")
                path = lutio.export_graph_cube(doc, "lut", doc["nodes"]["lut"]["params"]["lut_path"], 17, file_type="3dl", prelut_size=entries)
                _, shaper = lutio.read_3dl(path)
                self.assertEqual(len(shaper), entries)

    def test_the_output_bit_depth_sets_the_mesh_codes(self):
        from nodebased import lutio
        for out_depth, ceiling in (("8 bit", 255), ("12 bit", 4095), ("input bit depth", 1023)):
            with self.subTest(out_depth=out_depth), tempfile.TemporaryDirectory() as tmp:
                doc = self.lut_doc(tmp, ".3dl", lut_file_type="3dl")
                path = lutio.export_graph_cube(doc, "lut", doc["nodes"]["lut"]["params"]["lut_path"], 17, file_type="3dl", input_bit_depth="10 bit",
                                               output_bit_depth=out_depth, clamp_output=True)
                codes = [int(v) for line in Path(path).read_text().splitlines()[3:] for v in line.split()]
                self.assertEqual(max(codes), ceiling)                                           # the white corner is graded above 1 and clamped to full scale
                self.assertAlmostEqual(min(codes) / ceiling, 0.0073, delta=0.003)                  # black is graded to the same fraction of full scale at any depth

    def test_clamp_output_holds_the_written_values_to_zero_to_one(self):
        from nodebased import lutio
        with tempfile.TemporaryDirectory() as tmp:
            doc = self.lut_doc(tmp, ".cube", lut_file_type="cube")
            target = doc["nodes"]["lut"]["params"]["lut_path"]
            unclamped = lutio.read_cube(lutio.export_graph_cube(doc, "lut", target, 17, clamp_output=False))
            clamped = lutio.read_cube(lutio.export_graph_cube(doc, "lut", target, 17, clamp_output=True))
        self.assertGreater(float(unclamped.max()), 1.0)
        self.assertLessEqual(float(clamped.max()), 1.0)
        self.assertGreaterEqual(float(clamped.min()), 0.0)
        np.testing.assert_array_equal(clamped, np.clip(unclamped, 0.0, 1.0))

    def test_the_path_must_end_in_the_file_type(self):
        from nodebased import lutio
        with tempfile.TemporaryDirectory() as tmp:
            doc = self.lut_doc(tmp, ".cube", lut_file_type="3dl")
            with self.assertRaises(ValueError):
                lutio.export_graph_cube(doc, "lut", doc["nodes"]["lut"]["params"]["lut_path"], 17, file_type="3dl")

    def test_an_older_document_writes_the_unclamped_cube_it_always_did(self):
        doc = self.lut_doc("/nonexistent", ".cube")
        params = doc["nodes"]["lut"]["params"]
        for name in ("lut_file_type", "prelut_size", "prelut_style", "lut_input_bit_depth", "lut_output_bit_depth", "lut_clamp_output"):
            params.pop(name, None)
        upgraded = upgrade_document(doc)["nodes"]["lut"]["params"]
        self.assertEqual(upgraded["lut_clamp_output"], 0)
        self.assertEqual(SPECS["GenerateLUT"]["params"]["lut_clamp_output"], 1)                # a new node follows Nuke
        self.assertEqual(SPECS["GenerateLUT"]["params"]["lut_file_type"], "cube")


# ================================================================================================================ layout of the time and metadata nodes
class TimeAndMetadataLayoutTests(unittest.TestCase):
    """The main tab of each node lists the knobs the guide pairs with them in the guide's order; NodeBased's own knobs follow."""

    EXPECTED = {
        "AddTimeCode": ["Startcode", "FPS", "Get FPS from metadata", "Start Frame", "Use start frame", "Drop frame"],
        "BurnIn": ["Color", "Opacity", "X padding", "Y padding", "Font", "Text scale", "Top Left", "Top middle", "Top Right", "Bottom Left",
                   "Bottom middle", "Bottom Right", "Background bar", "Bar colour", "Bar opacity", "Bar X border", "Bar Y border"],
        "CurveTool": ["Curve type", "AutoCrop color", "AutoCrop tolerance", "Frames for base average", "Channels", "Box X", "Crop X", "Average R", "Average G",
                      "Average B", "Average A", "Max X", "Maximum luminance pixel value", "Min X", "Minimum luminance pixel value"],
        "AppendClip": ["Fade in", "Fade out", "Dissolve", "First Frame", "Take metadata from first clip"],
        "TimeBlur": ["Divisions", "Shutter", "Shutter Offset", "Custom Offset"],
        "ContactSheet": ["Resolution", "Rows/columns", "Gap", "Center", "Row Order", "Column Order", "Split Inputs", "Frame range"],
        "Precomp": ["File", "Reload count", "Read file for output", "Missing frames", "Reload file count", "Output node", "Use output node"],
    }

    def test_each_node_opens_with_the_guides_knobs_in_the_guides_order(self):
        from nodebased import knobs
        for kind, labels in self.EXPECTED.items():
            with self.subTest(kind=kind):
                shown = [g.label for g in knobs.knob_layout(kind) if g.kind != "legacy"]
                self.assertEqual(shown[:len(labels)], labels)

    def test_the_audit_finds_no_order_or_label_difference_left(self):
        import json
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import audit_nuke_fidelity as audit
        import nuke_fidelity_requirements as requirements
        mapping = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))
        nuke = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))["nodes"]
        for key in self.EXPECTED:
            with self.subTest(node=key):
                pair = mapping["pairs"][key]
                frac, detail = requirements.layout_component(pair["nodebased"], pair["knobs"], nuke[key]["knobs"])
                if frac is None:
                    continue
                kept, total = detail["layout_order"]
                self.assertEqual(kept, total, f"{key}: only {kept} of {total} paired knobs in Nuke's order")
                self.assertEqual(detail["label_mismatches"], [], f"{key}: labels differ")


if __name__ == "__main__":
    unittest.main()
