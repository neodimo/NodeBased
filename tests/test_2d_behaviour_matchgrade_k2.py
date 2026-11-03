"""2D behaviour lane 2, step K finish: MatchGrade's reference frames, transform, CDL, locks, iterations, export and Align Target to Source.

The Reference Guide names these knobs without saying what they compute, so each check states a property of the pixels or of the file: a CDL fitted to
a target graded by a known CDL recovers it within a stated error, more iterations never fit worse, a lock keeps its value, a written LUT reads back
through the existing readers and maps a colour as the node does. A node saved before keeps its older match, bit for bit.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import lutio
from nodebased import matchgradeknobs as mg
from nodebased.core import SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels

SLOPE, OFFSET, POWER, SATURATION = np.array([1.3, 0.85, 1.1]), np.array([0.04, -0.02, 0.06]), np.array([0.9, 1.15, 1.0]), 1.4


def varied(h=24, w=32, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.05, 0.9, (h, w, 4)).astype(np.float32)
    img[..., 3] = 1.0
    return img


def match(source, target, **params):
    return Evaluator._match_grade(source, target, dict(SPECS["MatchGrade"]["params"], **params))


def held(**params):
    return dict(SPECS["MatchGrade"]["params"], **params)


def pixels(image):
    return image[..., :3].reshape(-1, 3).astype(np.float64)


class ReferenceFrameListTests(unittest.TestCase):
    def test_a_list_names_frames_and_ranges(self):
        self.assertEqual(mg.parse_frames("9, 1; 5 9-11"), [1, 5, 9, 10, 11])
        self.assertEqual(mg.parse_frames("", 7), [7])           # an empty list reads the current frame
        self.assertEqual(mg.parse_frames("junk", None), [])

    def test_add_puts_the_current_frame_in_and_delete_takes_it_out(self):
        self.assertEqual(mg.add_frame("1, 5", 3), "1, 3, 5")
        self.assertEqual(mg.add_frame("1, 3, 5", 3), "1, 3, 5")     # no repeats
        self.assertEqual(mg.delete_frame("1, 3, 5", 3), "1, 5")
        self.assertEqual(mg.delete_frame("1, 5", 3), "1, 5")

    def test_analyze_pools_the_listed_frames(self):
        """Two source frames and two target frames graded by one known gain and offset: the baked values recover it; one frame alone differs."""
        first, second = varied(seed=3), varied(seed=4) * 0.6
        gain, offset = np.array([1.5, 0.7, 1.2]), np.array([0.05, -0.03, 0.1])
        grade = lambda image: np.concatenate([image[..., :3] * gain + offset, image[..., 3:]], axis=2)
        source = np.concatenate([pixels(first), pixels(second)])
        target = np.concatenate([pixels(grade(first)), pixels(grade(second))])
        baked = mg.stat_bake(source, target)
        for c, channel in enumerate("rgb"):
            self.assertAlmostEqual(baked[f"grade_gain_{channel}"], gain[c], places=5)
            self.assertAlmostEqual(baked[f"grade_offset_{channel}"], offset[c], places=5)
        # the frames differ in brightness, so one frame's pooled mean alone is a different bake than both frames pooled
        alone = mg.stat_bake(pixels(first), pixels(grade(second)))
        self.assertGreater(abs(alone["grade_gain_r"] - baked["grade_gain_r"]), 0.01)

    def test_a_baked_match_reads_no_target(self):
        source, target = varied(), varied(seed=9)
        baked = mg.stat_bake(pixels(source), pixels(target))
        out = match(source, None, match_analyzed=1, **baked)
        live = match(source, target)
        np.testing.assert_allclose(out[..., :3], live[..., :3], atol=2e-5)


class CdlTransformTests(unittest.TestCase):
    def test_the_defaults_leave_the_older_match_unchanged(self):
        src, tgt = varied(), varied(seed=9)
        out = match(src, tgt)
        for c in range(3):
            s, t = src[..., c], tgt[..., c]
            np.testing.assert_array_equal(out[..., c], ((s - float(np.mean(s))) * (float(np.std(t)) / float(np.std(s))) + float(np.mean(t))))

    def test_the_cdl_transform_applies_the_held_cdl_and_reads_no_target(self):
        src = varied()
        params = dict(match_transform="CDL", grade_gain_r=1.3, grade_gain_g=0.85, grade_gain_b=1.1, grade_offset_r=0.04, grade_offset_g=-0.02,
                      grade_offset_b=0.06, grade_gamma_r=1 / 0.9, grade_gamma_g=1 / 1.15, grade_gamma_b=1.0, match_saturation=SATURATION)
        out = match(src, None, **params)                                       # no target wired at all
        wanted = mg.apply_cdl(pixels(src), SLOPE, OFFSET, POWER, SATURATION)
        np.testing.assert_allclose(pixels(out), wanted, atol=2e-6)
        other = match(src, varied(seed=11), **params)
        np.testing.assert_array_equal(out, other)                              # a target changes nothing

    def test_the_cdl_transform_works_without_a_reference_in_a_graph(self):
        g = Graph()
        g.add("src", "Rectangle", dict(width=40, height=30, box_x=4, box_y=4, box_width=30, box_height=20, red=0.9, green=0.4, blue=0.2, alpha=1.0))
        g.add("mg", "MatchGrade", {"match_transform": "CDL", "grade_gain_r": 0.5}, image="src")
        base = evaluator_pixels(g.doc, "src")
        out = evaluator_pixels(g.doc, "mg")
        np.testing.assert_allclose(out[..., 0], base[..., 0] * 0.5, atol=1e-6)

    def test_saturation_moves_colours_from_their_luma(self):
        src = varied()
        luma = (pixels(src) * mg.LUMA).sum(axis=1, keepdims=True)
        none = match(src, None, match_transform="CDL", match_saturation=0.0)
        np.testing.assert_allclose(pixels(none), np.repeat(luma, 3, axis=1), atol=2e-6)       # no saturation: every channel is the luma
        more = match(src, None, match_transform="CDL", match_saturation=2.0)
        np.testing.assert_allclose(pixels(more) - luma, 2.0 * (pixels(src) - luma), atol=2e-6)
        np.testing.assert_array_equal(match(src, None, match_transform="CDL", match_saturation=1.0), src)

    def test_saturation_also_follows_the_live_match(self):
        src, tgt = varied(), varied(seed=9)
        plain, dull = match(src, tgt), match(src, tgt, match_saturation=0.5)
        luma = (pixels(plain) * mg.LUMA).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(pixels(dull) - luma, 0.5 * (pixels(plain) - luma), atol=2e-6)

    def test_the_mask_limits_the_saturation_with_the_grade(self):
        src = varied()
        src[:, :16, 3] = 1.0
        src[:, 16:, 3] = 0.0
        out = match(src, None, match_transform="CDL", match_saturation=0.0, grade_mask="Alpha", apply_to_mask=1)
        np.testing.assert_array_equal(out[:, 16:, :3], src[:, 16:, :3])
        self.assertFalse(np.allclose(out[:, :16, :3], src[:, :16, :3]))


class RecalculateCdlTests(unittest.TestCase):
    def setUp(self):
        self.src = pixels(varied())
        self.tgt = mg.apply_cdl(self.src, SLOPE, OFFSET, POWER, SATURATION)

    def test_a_target_graded_by_a_known_cdl_is_matched_back(self):
        fit = mg.fit_cdl(self.src, self.tgt, held(match_iterations=40))
        self.assertLess(fit["error"], 1e-4)
        np.testing.assert_allclose(fit["slope"], SLOPE, atol=2e-3)
        np.testing.assert_allclose(fit["offset"], OFFSET, atol=2e-3)
        np.testing.assert_allclose(fit["power"], POWER, atol=2e-3)
        self.assertAlmostEqual(fit["saturation"], SATURATION, delta=2e-3)
        # and the fitted CDL, applied by the node, lands on the target
        updates = mg.cdl_updates(fit, held())
        out = match(varied(), None, match_transform="CDL", **updates)
        np.testing.assert_allclose(pixels(out), self.tgt, atol=5e-4)

    def test_the_default_six_iterations_match_within_a_stated_error(self):
        fit = mg.fit_cdl(self.src, self.tgt, held())
        self.assertLess(fit["error"], 5e-3)

    def test_more_iterations_never_fit_worse(self):
        errors = [mg.fit_cdl(self.src, self.tgt, held(), iterations=n)["error"] for n in (0, 1, 2, 3, 6, 12, 40)]
        self.assertTrue(all(b <= a + 1e-15 for a, b in zip(errors, errors[1:])), errors)
        self.assertLess(errors[4], errors[0])               # six iterations beat the starting point

    def test_a_lock_keeps_its_value(self):
        for name, key in (("slope", "slope"), ("offset", "offset"), ("power", "power")):
            with self.subTest(lock=name):
                params = held(match_iterations=20, grade_gain_r=1.0, grade_offset_g=0.5, grade_gamma_b=2.0, **{f"match_lock_{name}": 1})
                fit = mg.fit_cdl(self.src, self.tgt, params)
                wanted = {"slope": mg.cdl_of(params)[0], "offset": mg.cdl_of(params)[1], "power": mg.cdl_of(params)[2]}[key]
                np.testing.assert_array_equal(fit[key], wanted)
                free = [n for n in ("slope", "offset", "power") if n != name]
                self.assertTrue(any(not np.allclose(fit[n], dict(slope=mg.cdl_of(params)[0], offset=mg.cdl_of(params)[1], power=mg.cdl_of(params)[2])[n])
                                    for n in free))

    def test_a_locked_saturation_stays(self):
        fit = mg.fit_cdl(self.src, self.tgt, held(match_iterations=20, match_saturation=0.7, match_lock_saturation=1))
        self.assertEqual(fit["saturation"], 0.7)
        free = mg.fit_cdl(self.src, self.tgt, held(match_iterations=20, match_saturation=0.7))
        self.assertGreater(free["saturation"], 1.2)         # unlocked, it climbs to the target's 1.4

    def test_the_lift_the_node_adds_stays_out_of_the_offset(self):
        params = held(grade_lift_r=0.1)
        fit = mg.fit_cdl(self.src, self.tgt, params, iterations=40)
        updates = mg.cdl_updates(fit, params)
        self.assertAlmostEqual(updates["grade_offset_r"] + 0.1, fit["offset"][0], places=9)
        out = match(varied(), None, match_transform="CDL", **dict(updates, grade_lift_r=0.1))
        np.testing.assert_allclose(pixels(out), self.tgt, atol=5e-4)

    def test_a_different_clip_is_paired_by_quantiles(self):
        """Match Different Clip: two unrelated pictures; the fit pairs their quantiles and finds the per-channel CDL between their distributions."""
        other = pixels(varied(seed=21))
        graded = mg.apply_cdl(other, SLOPE, OFFSET, POWER, 1.0)
        fit = mg.fit_cdl(self.src, graded, held(match_iterations=40), aligned=False)
        # the sources are two uniform draws of the same range, so the fitted CDL carries the source's distribution onto the graded one
        carried = mg.apply_cdl(self.src, fit["slope"], fit["offset"], fit["power"], 1.0)
        np.testing.assert_allclose(carried.mean(axis=0), graded.mean(axis=0), atol=0.01)
        np.testing.assert_allclose(carried.std(axis=0), graded.std(axis=0), atol=0.01)
        self.assertAlmostEqual(fit["saturation"], 1.0, delta=0.06)
        dull = (graded - (graded * mg.LUMA).sum(axis=1, keepdims=True)) * 0.5 + (graded * mg.LUMA).sum(axis=1, keepdims=True)
        self.assertLess(mg.fit_cdl(self.src, dull, held(match_iterations=40), aligned=False)["saturation"], 0.9)   # an estimate: the per-channel quantile fit takes some of the change

    def test_the_mask_weights_the_fit(self):
        """A mask that holds only the first half: the fit follows the half whose grade differs from the rest."""
        tgt = self.tgt.copy()
        half = len(tgt) // 2
        tgt[half:] = self.src[half:] * 0.2                    # the unmasked half is graded another way
        weights = np.zeros(len(tgt))
        weights[:half] = 1.0
        fit = mg.fit_cdl(self.src, tgt, held(match_iterations=40), weights=weights)
        np.testing.assert_allclose(fit["slope"], SLOPE, atol=2e-2)


class ExportTests(unittest.TestCase):
    PROBE = np.array([[[0.1, 0.4, 0.2, 1.0], [0.5, 0.3, 0.7, 1.0], [0.8, 0.6, 0.15, 1.0], [0.3, 0.3, 0.3, 1.0]]], np.float32)

    def params(self, **extra):
        return held(grade_gain_r=1.3, grade_gain_g=0.85, grade_gain_b=1.1, grade_offset_r=0.04, grade_offset_g=-0.02, grade_offset_b=0.06,
                    grade_gamma_r=1 / 0.9, grade_gamma_g=1 / 1.15, match_saturation=SATURATION, **extra)

    def wanted(self, p):
        return mg.apply_cdl(self.PROBE[..., :3], *mg.cdl_of(p))

    def test_a_cube_reads_back_to_the_match(self):
        p = self.params(match_lut_res=33)
        with tempfile.TemporaryDirectory() as tmp:
            path = mg.write_lut(p, Path(tmp) / "m.cube")
            text = Path(path).read_text().splitlines()
            lut = lutio.read_cube(path)
        self.assertIn("LUT_3D_SIZE 33", text[1])
        np.testing.assert_allclose(lutio.apply_cube(self.PROBE, lut, "trilinear")[..., :3], self.wanted(p), atol=0.01)

    def test_lut_resolution_sets_the_size_and_a_finer_lut_reads_closer(self):
        errors = {}
        for size in (5, 17, 65):
            p = self.params(match_lut_res=size)
            with tempfile.TemporaryDirectory() as tmp:
                lut = lutio.read_cube(mg.write_lut(p, Path(tmp) / "m.cube"))
            self.assertEqual(lut["size"] if isinstance(lut, dict) and "size" in lut else np.asarray(lut["table"] if isinstance(lut, dict) else lut).shape[0], size)
            errors[size] = float(np.abs(lutio.apply_cube(self.PROBE, lut, "trilinear")[..., :3] - self.wanted(p)).max())
        self.assertLess(errors[65], errors[5])

    def test_a_3dl_reads_back_for_each_pre_lut(self):
        for choice in ("Auto Detect", "Linear", "Logarithmic"):
            with self.subTest(prelut=choice), tempfile.TemporaryDirectory() as tmp:
                p = self.params(match_lut_res=33, match_prelut=choice)
                mesh, shaper = lutio.read_3dl(mg.write_lut(p, Path(tmp) / "m.3dl"))
                np.testing.assert_allclose(lutio.apply_3dl(self.PROBE, mesh, shaper, "trilinear")[..., :3], self.wanted(p), atol=0.03)

    def test_auto_detect_gives_a_3dl_the_logarithmic_shaper_and_a_cube_the_linear_one(self):
        p = self.params()
        self.assertEqual(mg.prelut_style(p, "3dl"), "logarithmic")
        self.assertEqual(mg.prelut_style(p, "cube"), "linear")
        self.assertEqual(mg.prelut_style(self.params(match_prelut="Linear"), "3dl"), "linear")

    def test_a_logarithmic_pre_lut_reads_the_darks_closer_than_a_linear_one(self):
        dark = np.array([[[0.004, 0.006, 0.01, 1.0], [0.01, 0.02, 0.005, 1.0], [0.03, 0.015, 0.02, 1.0]]], np.float32)
        errors = {}
        for choice in ("Linear", "Logarithmic"):
            p = self.params(match_lut_res=17, match_prelut=choice)
            with tempfile.TemporaryDirectory() as tmp:
                mesh, shaper = lutio.read_3dl(mg.write_lut(p, Path(tmp) / "m.3dl"))
            errors[choice] = float(np.abs(lutio.apply_3dl(dark, mesh, shaper, "trilinear")[..., :3] - np.clip(mg.apply_cdl(dark[..., :3], *mg.cdl_of(p)), 0, 1)).max())
        self.assertLess(errors["Logarithmic"], errors["Linear"])

    def test_the_colorspace_is_the_space_the_lut_takes_in_and_gives_out(self):
        p = self.params(match_lut_res=33, match_colorspace="sRGB")
        with tempfile.TemporaryDirectory() as tmp:
            lut = lutio.read_cube(mg.write_lut(p, Path(tmp) / "m.cube"))
        encoded = self.PROBE
        working = lutio.convert_colorspace(encoded[..., :3], "sRGB", "ACEScg")
        wanted = lutio.convert_colorspace(mg.apply_cdl(working, *mg.cdl_of(p)).astype(np.float32), "ACEScg", "sRGB")
        np.testing.assert_allclose(lutio.apply_cube(encoded, lut, "trilinear")[..., :3], wanted, atol=0.02)
        plain = mg.lut_values(self.params(match_lut_res=9), 9)
        self.assertFalse(np.allclose(plain, mg.lut_values(self.params(match_lut_res=9, match_colorspace="sRGB"), 9)))

    def test_a_bad_file_or_size_is_refused(self):
        with self.assertRaises(ValueError):
            mg.write_lut(self.params(), "match.txt")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            mg.write_lut(self.params(match_lut_res=32), Path(tmp) / "m.3dl")

    def test_the_cdl_file_holds_the_nodes_cdl(self):
        p = self.params()
        with tempfile.TemporaryDirectory() as tmp:
            path = mg.write_cdl(p, Path(tmp) / "m.cc")
            slope, offset, power, saturation = mg.read_cdl(path)
        wanted = mg.cdl_of(p)
        np.testing.assert_allclose(slope, wanted[0])
        np.testing.assert_allclose(offset, wanted[1])
        np.testing.assert_allclose(power, wanted[2], rtol=1e-8)
        self.assertAlmostEqual(saturation, SATURATION)
        self.assertAlmostEqual(power[0], 0.9, places=7)                       # power is 1 / gamma


class AlignTargetTests(unittest.TestCase):
    def graph(self, crop, tw=100, th=100):
        g = Graph()
        g.add("src", "Rectangle", dict(width=60, height=40, box_x=4, box_y=4, box_width=40, box_height=30, red=0.9, green=0.4, blue=0.2, alpha=1.0))
        g.add("tgt", "Rectangle", dict(width=tw, height=th, box_x=10, box_y=20, box_width=tw - 30, box_height=th - 40, red=0.3, green=0.6, blue=0.8, alpha=1.0, softness=0.0))
        g.add("mg", "MatchGrade", {"match_crop_target": crop}, image="src", reference="tgt")
        return g

    def run_button(self, g, tw, th):
        commands, last = mg.align_commands("mg", g.doc["nodes"], 60, 40, (tw, th))
        for command in commands:
            g.d.execute(command)
        return last

    def test_the_formats_must_match_before_the_button_and_agree_after(self):
        g = self.graph(0)
        with self.assertRaises(ValueError):
            evaluator_pixels(g.doc, "mg")
        last = self.run_button(g, 100, 100)
        self.assertEqual(g.doc["nodes"]["mg"]["inputs"]["reference"], last)
        self.assertEqual(g.doc["nodes"][last]["inputs"]["image"], "tgt")
        self.assertEqual(evaluator_pixels(g.doc, "mg").shape[:2], (40, 60))

    def test_crop_target_fills_the_format_with_the_targets_centre(self):
        """With Crop Target on the aligned target is the centred format-sized part of the filled target, for a taller and for a wider target."""
        from nodebased.imaging import Evaluator
        for tw, th in ((100, 100), (100, 50)):
            with self.subTest(target=(tw, th)):
                g = self.graph(1, tw, th)
                g.add("filled", "Reformat", dict(reformat_type="to_box", width=60, height=40, resize_type="fill"), image="tgt")
                last = self.run_button(g, tw, th)
                evaluator = Evaluator()
                filled = evaluator.evaluate_raster(dict(g.doc, view="filled"))
                aligned = evaluator.evaluate_raster(dict(g.doc, view=last))
                self.assertEqual((aligned.display.width, aligned.display.height, aligned.display.x, aligned.display.y), (60, 40, 0, 0))
                oy, ox = (filled.display.height - 40) // 2, (filled.display.width - 60) // 2
                np.testing.assert_array_equal(aligned.pixels, filled.pixels[oy:oy + 40, ox:ox + 60])
                self.assertEqual(evaluator_pixels(g.doc, "mg").shape[:2], (40, 60))

    def test_aligning_needs_a_target(self):
        g = Graph()
        g.add("mg", "MatchGrade", {})
        with self.assertRaises(ValueError):
            mg.align_commands("mg", g.doc["nodes"], 60, 40)
        g2 = self.graph(1)
        with self.assertRaises(ValueError):
            mg.align_commands("mg", g2.doc["nodes"], 60, 40)           # Crop Target needs the target's size


class PanelAndPairingTests(unittest.TestCase):
    def test_every_new_knob_is_a_param_with_a_limit_or_choice(self):
        from nodebased.core import CHOICES, LIMITS
        params = SPECS["MatchGrade"]["params"]
        for name in ("match_source_frames", "match_target_frames", "match_transform", "match_prelut", "match_lut_res", "match_colorspace", "match_iterations",
                     "match_saturation", "match_lock_slope", "match_lock_offset", "match_lock_power", "match_lock_saturation", "match_cdl_error",
                     "match_lut_file", "match_crop_target"):
            self.assertIn(name, params)
        self.assertEqual(params["match_transform"], "3D LUT")       # an older document keeps the live match
        self.assertEqual(params["match_saturation"], 1.0)
        self.assertEqual(CHOICES["match_transform"], ["3D LUT", "CDL"])

    def test_every_nuke_knob_of_the_node_is_paired_or_unpaired_with_a_reason(self):
        from tools import audit_nuke_fidelity as audit
        nuke = json.loads(audit.NUKE_JSON.read_text(encoding="utf-8"))["nodes"]["MatchGrade"]["knobs"]
        entries = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]["MatchGrade"]["knobs"]
        covered = {n for entry in entries for n in (entry.get("nuke") or [])}
        for knob in nuke:
            if knob["counted"]:
                self.assertIn(knob["name"], covered, knob["name"])
        for entry in entries:
            if entry["kind"] in ("unsupported", "panel"):
                self.assertTrue(entry["reason"])

    def test_the_panel_builds_with_its_buttons(self):
        from PySide6.QtWidgets import QPushButton
        from tests.test_desktop import APP, Window
        window = Window()
        try:
            window.dispatcher.execute({"op": "create", "id": "mgn", "type": "MatchGrade", "pos": [0, 0]})
            panel = window.build_node_panel("mgn")
            names = {b.objectName() for b in panel.findChildren(QPushButton)}
            APP.processEvents()
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
        for name in ("matchgrade-add-source", "matchgrade-delete-source", "matchgrade-add-target", "matchgrade-delete-target", "matchgrade-recalculate-cdl",
                     "matchgrade-align", "matchgrade-write-lut", "matchgrade-create-file-transform", "matchgrade-create-cdl-file"):
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()
