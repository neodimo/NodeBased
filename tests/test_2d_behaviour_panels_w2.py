"""Lane 2 step W2 (November 17, 2026): the rest of the colour and time worklist.

ColorCorrect's `test` and `mix_luminance`, Retime's `warp` curve and the other left-out rows of step W1. Every new knob has a neutral default
and a document saved before keeps its pixels bit for bit.
"""
import copy
import unittest

import numpy as np

from nodebased import timeknobs
from nodebased.core import SPECS, upgrade_document
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, kernel, tile_pixels

REC709 = np.array([0.2125, 0.7154, 0.0721], np.float32)


def plate(seed=3):
    rng = np.random.RandomState(seed)
    out = rng.rand(12, 16, 4).astype(np.float32)
    out[..., 3] = 0.8
    return out


def luminance(rgb):
    return (rgb[..., :3] * REC709).sum(axis=-1)


class ColorCorrectMixLuminanceTests(unittest.TestCase):
    def test_zero_keeps_the_corrected_luminance_and_one_gives_back_the_input_s(self):
        src = plate()
        corrected = kernel("ColorCorrect", src, gain=1.8, saturation=1.4, mix_luminance=0.0)
        self.assertFalse(np.allclose(luminance(corrected), luminance(src), atol=1e-3))
        kept = kernel("ColorCorrect", src, gain=1.8, saturation=1.4, mix_luminance=1.0)
        np.testing.assert_allclose(luminance(kept), luminance(src), atol=1e-5)
        self.assertFalse(np.allclose(kept[..., :3], src[..., :3], atol=1e-3))      # the colour still changed

    def test_half_is_half_way(self):
        src = plate()
        zero = kernel("ColorCorrect", src, gain=1.8, mix_luminance=0.0)
        one = kernel("ColorCorrect", src, gain=1.8, mix_luminance=1.0)
        half = kernel("ColorCorrect", src, gain=1.8, mix_luminance=0.5)
        np.testing.assert_allclose(half, (zero + one) / 2.0, atol=1e-5)

    def test_split_knobs_keep_the_luminance_too(self):
        src = plate()
        out = kernel("ColorCorrect", src, gain=[1.5, 1.0, 0.6, 1.0], mix_luminance=1.0)
        np.testing.assert_allclose(luminance(out), luminance(src), atol=1e-5)

    def test_the_default_changes_nothing(self):
        src = plate()
        self.assertEqual(SPECS["ColorCorrect"]["params"]["mix_luminance"], 0.0)
        np.testing.assert_array_equal(kernel("ColorCorrect", src, gain=1.8), kernel("ColorCorrect", src, gain=1.8, mix_luminance=0.0))

    def test_tiles_equal_the_full_frame(self):
        g = Graph()
        g.add("src", "Checker", {})
        g.add("node", "ColorCorrect", {"gain": 1.8, "saturation": 1.4, "mix_luminance": 0.7}, image="src")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)


class ColorCorrectTestOverlayTests(unittest.TestCase):
    @staticmethod
    def ramp():
        out = np.zeros((1, 6, 4), np.float32)
        out[0, :, :3] = np.array([0.0, 0.3, 0.3, 1.0, 0.05, 0.7], np.float32)[:, None]     # shadow, mid, mid, highlight, mixed, mixed
        out[..., 3] = 0.6
        return out

    def test_pure_ranges_are_black_grey_and_white(self):
        out = kernel("ColorCorrect", self.ramp(), test=1)
        np.testing.assert_allclose(out[0, 0, :3], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[0, 1:3, :3], 0.5, atol=1e-6)
        np.testing.assert_allclose(out[0, 3, :3], 1.0, atol=1e-6)

    def test_a_mixture_of_neighbouring_ranges_is_green(self):
        out = kernel("ColorCorrect", self.ramp(), test=1)
        np.testing.assert_allclose(out[0, 4, :3], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[0, 5, :3], [0.0, 1.0, 0.0], atol=1e-6)

    def test_overlapping_curves_are_magenta(self):
        both = '{"interpolation":"linear","points":[[0,1],[1,1]]}'
        out = kernel("ColorCorrect", self.ramp(), test=1, shadows_curve=both, highlights_curve=both)
        np.testing.assert_allclose(out[0, 1, :3], [1.0, 0.0, 1.0], atol=1e-6)

    def test_alpha_is_kept_and_the_correction_is_not_drawn(self):
        src = self.ramp()
        out = kernel("ColorCorrect", src, test=1, gain=3.0)
        np.testing.assert_array_equal(out[..., 3], src[..., 3])
        np.testing.assert_allclose(out[0, 1, :3], 0.5, atol=1e-6)       # the gain moved nothing in the overlay

    def test_off_is_the_correction(self):
        src = plate()
        np.testing.assert_array_equal(kernel("ColorCorrect", src, gain=1.8, test=0), kernel("ColorCorrect", src, gain=1.8))

    def test_test_with_every_knob_at_default_still_draws(self):
        out = kernel("ColorCorrect", self.ramp(), test=1)
        self.assertFalse(np.allclose(out[..., :3], self.ramp()[..., :3]))

    def test_the_mix_and_the_mask_apply_to_the_overlay(self):
        src = self.ramp()
        half = Evaluator._apply_mask_mix(src, kernel("ColorCorrect", src, test=1), mask=None, mix=0.5)
        np.testing.assert_allclose(half[0, 3, :3], 1.0, atol=1e-6)            # a white pixel stays white half way to white
        np.testing.assert_allclose(half[0, 1, :3], (0.3 + 0.5) / 2, atol=1e-6)

    def test_tiles_equal_the_full_frame(self):
        g = Graph()
        g.add("src", "Checker", {})
        g.add("node", "ColorCorrect", {"test": 1}, image="src")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_split_knobs_draw_the_same_overlay(self):
        src = plate()
        np.testing.assert_array_equal(kernel("ColorCorrect", src, test=1), kernel("ColorCorrect", src, test=1, gain=[1.0, 1.0, 1.0, 1.0]))


class ColorCorrectOlderDocumentTests(unittest.TestCase):
    def test_a_node_saved_before_loads_with_both_off_and_its_pixels_unchanged(self):
        g = Graph()
        g.add("src", "Checker", {})
        g.add("node", "ColorCorrect", {"gain": 1.6, "saturation": 0.8}, image="src")
        before = evaluator_pixels(g.doc, "node")
        doc = copy.deepcopy(g.doc)
        for name in ("test", "mix_luminance"):
            doc["nodes"]["node"]["params"].pop(name, None)
        doc = upgrade_document(doc)
        params = doc["nodes"]["node"]["params"]
        self.assertEqual((params["test"], params["mix_luminance"]), (0, 0.0))
        np.testing.assert_array_equal(evaluator_pixels(doc, "node"), before)

    def test_the_gpu_island_leaves_both_to_the_processor(self):
        from nodebased import gpu2d
        for params in ({"test": 1}, {"mix_luminance": 0.5}):
            with self.assertRaises(gpu2d.Unsupported):
                gpu2d.map_color_correct(dict(SPECS["ColorCorrect"]["params"], **params))


class RetimeWarpTests(unittest.TestCase):
    LOCKED = {"input_range_start": 1, "input_range_start_lock": 1, "input_range_end": 17, "input_range_end_lock": 1,
              "output_range_start": 1, "output_range_start_lock": 1, "output_range_end": 17, "output_range_end_lock": 1, "retime_filter": "none"}

    @staticmethod
    def curve(*points):
        return '{"interpolation":"linear","points":[' + ",".join(f"[{x:g},{y:g}]" for x, y in points) + "]}"

    def frame(self, frame, **params):
        return timeknobs.plan("Retime", dict(self.LOCKED, **params), frame, (1, 20)).size_frame

    def test_the_default_curve_is_a_straight_line_and_warps_nothing(self):
        self.assertEqual(SPECS["Retime"]["params"]["warp"], timeknobs.WARP_IDENTITY)
        for frame in range(1, 18):
            self.assertEqual(self.frame(frame), self.frame(frame, warp=timeknobs.WARP_IDENTITY))
            self.assertEqual(self.frame(frame), frame)

    def test_a_curve_through_the_ends_keeps_the_first_and_last_frame(self):
        warp = self.curve((0, 0), (0.5, 0.25), (1, 1))
        self.assertEqual(self.frame(1, warp=warp), 1)
        self.assertEqual(self.frame(17, warp=warp), 17)

    def test_a_shallower_slope_slows_the_motion_and_a_steeper_one_speeds_it_up(self):
        warp = self.curve((0, 0), (0.5, 0.25), (1, 1))
        steps = [self.frame(f + 1, warp=warp) - self.frame(f, warp=warp) for f in range(1, 17)]
        self.assertTrue(all(s > 0 for s in steps))
        self.assertAlmostEqual(steps[0], 0.5, places=6)        # 0.25 of the range over the first half: half speed
        self.assertAlmostEqual(steps[-1], 1.5, places=6)       # 0.75 over the second half: one and a half times

    def test_a_dip_runs_the_clip_backwards(self):
        warp = self.curve((0, 0), (0.5, 0.6), (0.75, 0.4), (1, 1))
        self.assertAlmostEqual(self.frame(9, warp=warp), 1 + 16 * 0.6, places=6)
        self.assertAlmostEqual(self.frame(13, warp=warp), 1 + 16 * 0.4, places=6)
        self.assertLess(self.frame(13, warp=warp), self.frame(9, warp=warp))

    def test_reverse_flips_the_warped_time_about_the_input_range(self):
        warp = self.curve((0, 0), (0.5, 0.25), (1, 1))
        self.assertAlmostEqual(self.frame(9, warp=warp, reverse=1), 17 - 16 * 0.25, places=6)

    def test_before_and_after_still_find_the_frames_outside_the_output_range(self):
        warp = self.curve((0, 0), (0.5, 0.25), (1, 1))
        for frame in (-2, 0, 18, 25):
            self.assertEqual(self.frame(frame, warp=warp, retime_before="continue", retime_after="continue"),
                             self.frame(frame, retime_before="continue", retime_after="continue"))

    def test_the_box_widens_where_the_curve_is_steep(self):
        warp = self.curve((0, 0), (0.5, 0.25), (1, 1))
        wide = timeknobs.plan("Retime", dict(self.LOCKED, warp=warp, retime_filter="box"), 15, (1, 20))
        narrow = timeknobs.plan("Retime", dict(self.LOCKED, warp=warp, retime_filter="box"), 3, (1, 20))
        self.assertGreater(len(wide.samples), len(narrow.samples))
        self.assertAlmostEqual(sum(w for _f, w in wide.samples), 1.0, places=6)

    def test_a_node_saved_before_keeps_the_straight_line(self):
        g = Graph()
        g.add("node", "Retime", {"speed": 2.0})
        doc = copy.deepcopy(g.doc)
        doc["nodes"]["node"]["params"].pop("warp", None)
        doc = upgrade_document(doc)
        self.assertEqual(doc["nodes"]["node"]["params"]["warp"], timeknobs.WARP_IDENTITY)
        old = timeknobs.plan("Retime", {k: v for k, v in doc["nodes"]["node"]["params"].items() if k != "warp"}, 7, (1, 20))
        new = timeknobs.plan("Retime", doc["nodes"]["node"]["params"], 7, (1, 20))
        self.assertEqual(old, new)


class ExposureColorspaceTests(unittest.TestCase):
    def test_linear_is_the_multiplication_it_always_was(self):
        self.assertEqual(SPECS["Exposure"]["params"]["exposure_colorspace"], "Linear")
        src = plate()
        out = kernel("Exposure", src, red=1.0, gang=1, exposure_colorspace="Linear")
        np.testing.assert_allclose(out[..., :3], src[..., :3] * 2.0, atol=1e-6)

    def test_cineon_adds_instead_of_multiplying(self):
        src = plate()
        out = kernel("Exposure", src, red=1.0, gang=1, exposure_colorspace="Cineon")
        shift = np.log10(2.0) * 0.6 / (0.002 * 1023.0)
        np.testing.assert_allclose(out[..., :3], src[..., :3] + shift, atol=1e-6)
        np.testing.assert_array_equal(out[..., 3], src[..., 3])

    def test_cineon_through_log2lin_at_gamma_06_with_ignore_matches_linear_exactly(self):
        # The guide: "You can match Linear exactly if you set gamma to 0.6 and enable ignore in the Log2Lin node".
        src = plate()
        cineon = kernel("Exposure", src, red=1.5, gang=1, exposure_colorspace="Cineon")
        linear = kernel("Exposure", src, red=1.5, gang=1, exposure_colorspace="Linear")
        through = kernel("Log2Lin", cineon, gamma=0.6, ignore_black=1)
        reference = kernel("Log2Lin", src, gamma=0.6, ignore_black=1)
        np.testing.assert_allclose(through[..., :3], reference[..., :3] * linear[..., :3] / src[..., :3], rtol=2e-4)

    def test_each_channel_has_its_own_addition_when_not_ganged(self):
        src = plate()
        out = kernel("Exposure", src, gang=0, red=-1.0, green=2.0, blue=0.0, exposure_colorspace="Cineon")
        step = 0.6 / (0.002 * 1023.0)
        for channel, stops in enumerate((-1.0, 2.0, 0.0)):
            np.testing.assert_allclose(out[..., channel], src[..., channel] + np.log10(2.0 ** stops) * step, atol=1e-6)

    def test_a_node_saved_before_is_linear(self):
        g = Graph()
        g.add("src", "Checker", {})
        g.add("node", "Exposure", {"red": 1.0}, image="src")
        before = evaluator_pixels(g.doc, "node")
        doc = copy.deepcopy(g.doc)
        doc["nodes"]["node"]["params"].pop("exposure_colorspace", None)
        doc = upgrade_document(doc)
        self.assertEqual(doc["nodes"]["node"]["params"]["exposure_colorspace"], "Linear")
        np.testing.assert_array_equal(evaluator_pixels(doc, "node"), before)

    def test_tiles_equal_the_full_frame(self):
        g = Graph()
        g.add("src", "Checker", {})
        g.add("node", "Exposure", {"red": 1.0, "exposure_colorspace": "Cineon"}, image="src")
        np.testing.assert_allclose(tile_pixels(g.doc, "node"), evaluator_pixels(g.doc, "node"), atol=1e-6)

    def test_the_menu_is_the_guide_s_two_words(self):
        from nodebased.core import CHOICES
        self.assertEqual(CHOICES["exposure_colorspace"], ["Linear", "Cineon"])


class ColorspaceMatrixReadoutTests(unittest.TestCase):
    def params(self, **changes):
        return dict(SPECS["Colorspace"]["params"], **changes)

    def test_the_matrix_is_what_the_node_applies_between_its_curves(self):
        from nodebased.ocio_nodes import colorspace_matrix, transform
        src = plate()
        for changes in ({"primaries_out": "Rec.2020"}, {"primaries_in": "Rec.2020", "primaries_out": "P3-D65"},
                        {"whitepoint_out": "D60", "bradford_matrix": 1}, {"primaries_out": "Rec.2020", "swap": 1}):
            with self.subTest(**changes):
                p = self.params(**changes)
                out = transform("Colorspace", p, src)
                expected = src[..., :3] @ colorspace_matrix(p).T.astype(np.float32)
                np.testing.assert_allclose(out[..., :3], expected, atol=2e-6)

    def test_the_matrix_makes_the_same_picture_as_a_colormatrix_node(self):
        from nodebased.ocio_nodes import colorspace_matrix, transform
        src = plate()
        p = self.params(primaries_out="Rec.2020")
        m = colorspace_matrix(p)
        names = {f"matrix_{r}{c}": float(m[r, c]) for r in range(3) for c in range(3)}
        np.testing.assert_allclose(kernel("ColorMatrix", src, **names)[..., :3], transform("Colorspace", p, src)[..., :3], atol=2e-6)

    def test_equal_primaries_and_illuminants_give_the_identity(self):
        from nodebased.ocio_nodes import colorspace_matrix
        np.testing.assert_allclose(colorspace_matrix(self.params()), np.eye(3), atol=1e-9)

    def test_the_panel_shows_it_and_follows_the_choices(self):
        from PySide6.QtWidgets import QApplication, QLabel
        app = QApplication.instance() or QApplication([])
        from nodebased.app import Window
        from nodebased.ocio_nodes import colorspace_matrix_text
        window = Window()
        try:
            window.dispatcher.execute({"op": "create", "id": "cs", "type": "Colorspace", "pos": [0, 0]})
            window.dispatcher.execute({"op": "set", "id": "cs", "param": "primaries_out", "value": "Rec.2020"})
            panel = window.build_node_panel("cs")
            label = panel.findChild(QLabel, "colorspace-matrix-readout")
            self.assertIsNotNone(label)
            self.assertEqual(label.text(), colorspace_matrix_text(window.dispatcher.document["nodes"]["cs"]["params"]))
            self.assertEqual(len(label.text().splitlines()), 3)
            self.assertNotEqual(label.text(), colorspace_matrix_text(self.params()))
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()

    def test_the_pairing_names_the_panel(self):
        import json
        from pathlib import Path
        mapping = json.loads((Path(__file__).resolve().parents[1] / "docs" / "parity" / "nodebased_to_nuke.json").read_text())
        rows = [r for r in mapping["pairs"]["Colorspace"]["knobs"] if "colormatrix" in r.get("nuke", [])]
        self.assertEqual([r["kind"] for r in rows], ["panel"])


class GenerateLutFormatTests(unittest.TestCase):
    @staticmethod
    def document(tmp, **params):
        import pathlib
        g = Graph()
        g.add("src", "Constant", dict(width=8, height=8))
        g.add("grade", "Grade", dict(multiply=1.6, offset=0.02, gamma=0.8), image="src")
        g.add("lut", "GenerateLUT", dict(lut_path=str(pathlib.Path(tmp) / "out.3dl"), lut_size=17, lut_file_type="3dl", **params), source="grade")
        return g.doc

    def write(self, tmp, **kwargs):
        from nodebased import lutio
        doc = self.document(tmp)
        return lutio.export_graph_cube(doc, "lut", doc["nodes"]["lut"]["params"]["lut_path"], 17, file_type="3dl", clamp_output=True, **kwargs)

    def test_the_default_is_flame_and_smoke_and_a_saved_node_keeps_the_lustre_header(self):
        self.assertEqual(SPECS["GenerateLUT"]["params"]["lut_3dl_format"], "autodesk flame/smoke")
        g = Graph()
        g.add("lut", "GenerateLUT", {})
        doc = copy.deepcopy(g.doc)
        doc["nodes"]["lut"]["params"].pop("lut_3dl_format", None)
        self.assertEqual(upgrade_document(doc)["nodes"]["lut"]["params"]["lut_3dl_format"], "autodesk lustre")

    def test_lustre_opens_with_the_3dmesh_header_and_flame_does_not(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            lustre = open(self.write(tmp, format_3dl="autodesk lustre"), encoding="utf-8").read().splitlines()
        with tempfile.TemporaryDirectory() as tmp:
            flame = open(self.write(tmp, format_3dl="autodesk flame/smoke"), encoding="utf-8").read().splitlines()
        self.assertEqual(lustre[0], "3DMESH")
        self.assertTrue(lustre[1].startswith("Mesh 4 "))
        self.assertFalse(any(line.startswith(("3DMESH", "Mesh")) for line in flame))
        self.assertEqual(lustre[2:], flame)           # the preLUT row and the mesh are the same

    def test_both_read_back_to_the_same_look(self):
        from nodebased import lutio
        probe = np.array([[[0.02, 0.05, 0.1, 1.0], [0.2, 0.4, 0.1, 1.0], [0.9, 0.8, 0.1, 1.0]]], np.float32)
        import tempfile
        mapped = []
        for name in ("autodesk lustre", "autodesk flame/smoke"):
            with tempfile.TemporaryDirectory() as tmp:
                mesh, shaper = lutio.read_3dl(self.write(tmp, format_3dl=name))
                mapped.append(lutio.apply_3dl(probe, mesh, shaper, "trilinear"))
        np.testing.assert_array_equal(mapped[0], mapped[1])

    def test_an_unwritten_format_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.write(tmp, format_3dl="assimilate scratch")

    def test_the_panel_names_the_button_and_the_menu_as_the_guide_does(self):
        from PySide6.QtWidgets import QApplication, QPushButton
        app = QApplication.instance() or QApplication([])
        from nodebased.app import Window
        window = Window()
        try:
            window.dispatcher.execute({"op": "create", "id": "lut", "type": "GenerateLUT", "pos": [0, 0]})
            panel = window.build_node_panel("lut")
            button = panel.findChild(QPushButton, "generate-lut-file")
            self.assertEqual(button.text(), "Generate and Write LUT File")
            from nodebased import knobs
            labels = [g.label for g in knobs.knob_layout("GenerateLUT")]
            self.assertIn("3dl file format", labels)
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()


class HistogramPlotTests(unittest.TestCase):
    @staticmethod
    def picture():
        out = np.zeros((10, 10, 4), np.float32)
        out[:, :5, :3] = 0.1          # shadows
        out[:, 5:, :3] = 0.9          # highlights
        out[0, 0, :3] = -2.0          # a superblack and a superwhite
        out[0, 9, :3] = 3.0
        out[..., 3] = 1.0
        return out

    def test_each_channel_is_a_percentage_of_the_pixels_over_brightness_zero_to_one(self):
        from nodebased import histogramplot
        shown = histogramplot.percentages(self.picture())
        self.assertEqual(shown.shape, (3, histogramplot.BINS))
        np.testing.assert_allclose(shown.sum(axis=1), 100.0, atol=1e-9)

    def test_shadows_are_on_the_left_and_highlights_on_the_right(self):
        from nodebased import histogramplot
        shown = histogramplot.percentages(self.picture())
        half = histogramplot.BINS // 2
        self.assertAlmostEqual(float(shown[0, :half].sum()), 50.0, places=6)
        self.assertAlmostEqual(float(shown[0, half:].sum()), 50.0, places=6)
        self.assertAlmostEqual(float(shown[0, 0]), 1.0, places=6)               # the superblack counts in the first column
        self.assertAlmostEqual(float(shown[0, -1]), 1.0, places=6)              # the superwhite in the last

    def test_the_plot_follows_the_controls(self):
        from nodebased import histogramplot
        before, after = histogramplot.plot_data(self.picture(), {"black_out": 0.5, "white_out": 1.0})
        np.testing.assert_array_equal(before, histogramplot.percentages(self.picture()))
        self.assertEqual(float(after[0, :histogramplot.BINS // 2].sum()), 0.0)          # nothing is left below the output black
        self.assertAlmostEqual(float(after.sum(axis=1)[0]), 100.0, places=6)

    def test_a_black_point_moves_the_mass_down(self):
        from nodebased import histogramplot
        _, plain = histogramplot.plot_data(self.picture(), {})
        _, raised = histogramplot.plot_data(self.picture(), {"black": 0.3})
        self.assertGreater(float(raised[0, :4].sum()), float(plain[0, :4].sum()))

    def test_the_panel_draws_the_plot_when_analyzed(self):
        from PySide6.QtWidgets import QApplication, QLabel, QPushButton
        app = QApplication.instance() or QApplication([])
        from nodebased.app import Window
        window = Window()
        try:
            window.dispatcher.execute({"op": "create", "id": "src", "type": "Constant", "pos": [0, 0], "params": {"width": 16, "height": 12}})
            window.dispatcher.execute({"op": "create", "id": "hist", "type": "Histogram", "pos": [100, 0]})
            window.dispatcher.execute({"op": "connect", "id": "hist", "input": "image", "source": "src"})
            panel = window.build_node_panel("hist")
            button = next(b for b in panel.findChildren(QPushButton) if b.text() == "Analyze / refresh")
            button.click()
            app.processEvents()
            plots = [label for label in panel.findChildren(QLabel) if label.pixmap() is not None and not label.pixmap().isNull()]
            self.assertTrue(plots)
            self.assertIn("percentage of pixels", plots[0].toolTip())
        finally:
            window.saved_document = window.dispatcher.document
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
