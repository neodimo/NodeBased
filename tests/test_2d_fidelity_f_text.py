"""Fidelity step F, part 2: Text takes Nuke's vertical justify, font style and scales, kerning, tracking, baseline shift, leading,
the Groups transform, the colour ramp and drop shadows. Pixel assertions per option, the tile path against the evaluator, bypass
and old documents."""
import copy
import unittest

import numpy as np

from nodebased import textknobs
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

W, H = 200, 120


def params(**extra):
    p = dict(SPECS["Text"]["params"], width=W, height=H, message="Ab", font_size=14.0, box_x=10.0, box_y=10.0,
             box_width=180.0, box_height=100.0, red=1.0, green=1.0, blue=1.0, alpha=1.0)
    p.update(extra)
    return p


def draw(**extra):
    # shadow_distance is inert while the shadows are off, but any step F knob off its default moves the node onto the new drawing
    # path, so a baseline for comparing the other knobs against is drawn on that path too.
    extra.setdefault("shadow_distance", 9.0)
    p = params(**extra)
    return Evaluator._draw_shape("Text", p, 0, 0, W, H, 1)


def ink(image, threshold=0.5):
    """(top, bottom, left, right) of the pixels whose alpha passes the threshold, bottom and right exclusive."""
    rows, cols = np.nonzero(image[..., 3] > threshold)
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def total(image):
    return float(image[..., 3].sum())


class DefaultsTests(unittest.TestCase):
    def test_a_new_text_starts_at_the_top_like_nuke_and_an_old_one_stays_centred(self):
        self.assertEqual(SPECS["Text"]["params"]["yjustify"], "top")
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "t", "type": "Text", "params": {"width": W, "height": H}})
        old = copy.deepcopy(d.document)
        for name in textknobs.NEW:
            old["nodes"]["t"]["params"].pop(name, None)
        self.assertEqual(upgrade_document(old)["nodes"]["t"]["params"]["yjustify"], "center")

    def test_an_old_text_draws_through_the_original_call_and_gives_the_same_pixels(self):
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor, QFont, QImage, QPainter
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication(["nodebased"])
        p = params(yjustify="center", justify="center", font_size=30.0, box_width=140.0, box_height=80.0, red=0.9, green=0.5, blue=0.2, alpha=0.8)
        self.assertTrue(textknobs.is_legacy(p))
        image = QImage(W, H, QImage.Format.Format_RGBA8888)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = QFont()
        font.setPointSizeF(30.0)
        painter.setFont(font)
        painter.setPen(QColor.fromRgbF(0.9, 0.5, 0.2, 0.8))
        painter.drawText(QRectF(10, 10, 140, 80), int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter) | int(Qt.TextFlag.TextWordWrap), "Ab")
        painter.end()
        raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint8).reshape(H, image.bytesPerLine())[:, :W * 4].reshape(H, W, 4)
        straight = raw.astype(np.float32) / 255.0
        want = np.concatenate([straight[..., :3] * straight[..., 3:4], straight[..., 3:4]], axis=-1)
        np.testing.assert_array_equal(Evaluator._text_shape(p, 0, 0, W, H), want)

    def test_any_step_f_knob_off_its_default_leaves_the_original_call(self):
        for name, value in (("yjustify", "top"), ("index", "Bold"), ("global_font_scale", 2.0), ("font_width", 120.0), ("kerning", 1.0),
                            ("tracking", 1.0), ("baseline_shift", 1.0), ("leading", 1.0), ("rotate", 5.0), ("ramp", "linear"),
                            ("enable_shadows", 1)):
            with self.subTest(name=name):
                self.assertFalse(textknobs.is_legacy(params(**{"yjustify": "center", name: value})))

    def test_the_new_knobs_have_nukes_defaults_on_a_new_node(self):
        p = SPECS["Text"]["params"]
        self.assertEqual((p["index"], p["global_font_scale"], p["font_width"], p["font_height"], p["kerning"], p["tracking"],
                          p["baseline_shift"], p["leading"]), ("Regular", 1.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual((p["ramp"], p["enable_shadows"], p["shadow_opacity"], p["shadow_angle"], p["shadow_distance"]), ("none", 0, 0.7, 225.0, 5.0))


class LayoutTests(unittest.TestCase):
    def test_vertical_justify_puts_a_line_at_the_top_middle_or_bottom_of_the_box(self):
        top, mid, bottom = (ink(draw(yjustify=v)) for v in ("top", "center", "bottom"))
        self.assertLess(top[0] - 10, 10)                                            # starts within a font's ascent of the box top (row 10)
        self.assertGreater(bottom[1], 110 - 8)                                      # ends within a font's descent of the box bottom (row 110)
        self.assertLessEqual(bottom[1], 111)
        self.assertTrue(top[0] < mid[0] < bottom[0])
        self.assertAlmostEqual((mid[0] + mid[1]) / 2, 60, delta=8)

    def test_global_font_scale_multiplies_the_font_size(self):
        np.testing.assert_array_equal(draw(global_font_scale=2.0, font_size=15.0), draw(font_size=30.0))

    def test_font_width_and_height_stretch_the_glyphs(self):
        base = ink(draw())
        wide = ink(draw(font_width=200.0))
        tall = ink(draw(font_height=200.0))
        base_h, base_w = base[1] - base[0], base[3] - base[2]
        self.assertAlmostEqual((wide[3] - wide[2]) / base_w, 2.0, delta=0.15)
        self.assertAlmostEqual((wide[1] - wide[0]) / base_h, 1.0, delta=0.1)
        self.assertAlmostEqual((tall[1] - tall[0]) / base_h, 2.0, delta=0.2)

    def test_the_font_style_changes_the_weight_and_slant(self):
        regular, bold, italic = (draw(message="Hello", index=v, yjustify="center") for v in ("Regular", "Bold", "Italic"))
        self.assertGreater(total(bold), total(regular) * 1.1)
        self.assertGreater(float(np.abs(italic[..., 3] - regular[..., 3]).sum()), 5.0)
        both = draw(message="Hello", index="Bold Italic", yjustify="center")
        self.assertGreater(total(both), total(regular) * 1.1)
        self.assertGreater(float(np.abs(both[..., 3] - bold[..., 3]).sum()), 5.0)

    def test_tracking_spaces_every_pair_and_kerning_only_those_inside_a_word(self):
        base = ink(draw(message="ABC DEF"))
        gap = 6.0
        tracked = ink(draw(message="ABC DEF", tracking=gap))
        kerned = ink(draw(message="ABC DEF", kerning=gap))
        # Seven characters: tracking opens six gaps; kerning leaves the one after the space alone, so five.
        self.assertAlmostEqual((tracked[3] - tracked[2]) - (base[3] - base[2]), 6 * gap, delta=2.5)
        self.assertAlmostEqual((kerned[3] - kerned[2]) - (base[3] - base[2]), 5 * gap, delta=2.5)

    def test_negative_tracking_closes_the_letters_up(self):
        base = ink(draw(message="ABCD"))
        tight = ink(draw(message="ABCD", tracking=-2.0))
        self.assertLess(tight[3] - tight[2], base[3] - base[2] - 4)

    def test_baseline_shift_raises_the_text_by_whole_pixels(self):
        base = draw(yjustify="center")
        up = draw(yjustify="center", baseline_shift=7.0)
        np.testing.assert_allclose(up[:-7], base[7:], atol=1e-6)
        down = draw(yjustify="center", baseline_shift=-5.0)
        np.testing.assert_allclose(down[5:], base[:-5], atol=1e-6)

    def test_leading_opens_the_space_between_lines(self):
        two = draw(message="A\nB", yjustify="top")
        spaced = draw(message="A\nB", yjustify="top", leading=12.0)
        first, second = ink(two), ink(spaced)
        self.assertAlmostEqual((second[1] - second[0]) - (first[1] - first[0]), 12.0, delta=1.5)
        np.testing.assert_allclose(spaced[: first[0] + 8, :], two[: first[0] + 8, :], atol=1e-6)        # the first line has not moved

    def test_a_long_message_wraps_inside_the_box_and_justify_places_the_lines(self):
        words = "wrap these words into the narrow box"
        narrow = draw(message=words, box_width=60.0, yjustify="top", font_size=10.0)
        rows = ink(narrow)
        self.assertLessEqual(rows[3], 10 + 60 + 2)
        self.assertGreater(rows[1] - rows[0], 25)                                    # several lines
        left, right = (ink(draw(message="Ab", justify=v, yjustify="top")) for v in ("left", "right"))
        self.assertLess(left[2], right[2])


def centred(**extra):
    return draw(yjustify="center", justify="center", **extra)


class TransformTests(unittest.TestCase):
    def test_translate_moves_the_text(self):
        base = centred()
        moved = centred(translate_x=6.0, translate_y=-4.0)
        np.testing.assert_allclose(moved[:-4, 6:], base[4:, :-6], atol=1e-6)

    def test_rotate_by_180_about_the_box_centre_flips_the_text_through_it(self):
        base = ink(draw(yjustify="top"))                                             # left justified: well away from the pivot
        turned = ink(draw(yjustify="top", rotate=180.0))
        self.assertAlmostEqual((base[0] + base[1]) / 2 + (turned[0] + turned[1]) / 2, 120, delta=1.5)           # centre rows mirror about row 60
        self.assertAlmostEqual((base[2] + base[3]) / 2 + (turned[2] + turned[3]) / 2, 200, delta=1.5)           # and columns about column 100

    def test_the_centre_offset_moves_the_pivot(self):
        about_box = ink(draw(yjustify="top", rotate=180.0))
        about_offset = ink(draw(yjustify="top", rotate=180.0, center_x=-20.0, center_y=-10.0))
        self.assertAlmostEqual((about_offset[2] + about_offset[3]) / 2 - (about_box[2] + about_box[3]) / 2, 2 * -20.0, delta=1.5)
        self.assertAlmostEqual((about_offset[0] + about_offset[1]) / 2 - (about_box[0] + about_box[1]) / 2, 2 * -10.0, delta=1.5)

    def test_scale_grows_the_text_about_the_box_centre(self):
        base, big = ink(centred()), ink(centred(scale=2.0))
        self.assertAlmostEqual((big[3] - big[2]) / (base[3] - base[2]), 2.0, delta=0.15)
        self.assertAlmostEqual((big[1] - big[0]) / (base[1] - base[0]), 2.0, delta=0.2)
        self.assertAlmostEqual((big[2] + big[3]) / 2, 100, delta=3)

    def test_skew_shears_the_text_and_the_order_matters_only_with_both(self):
        def lean(**extra):
            """Slope of the stroke's centre column against the row, for a tall letter I."""
            image = centred(message="I", font_size=40.0, **extra)[..., 3]
            rows = np.nonzero(image.sum(axis=1) > 0.5)[0]
            centres = np.array([(image[r] * np.arange(image.shape[1])).sum() / image[r].sum() for r in rows])
            return float(np.polyfit(rows, centres, 1)[0])
        self.assertAlmostEqual(lean(), 0.0, delta=0.02)
        self.assertAlmostEqual(lean(skew_x=0.5), 0.5, delta=0.03)                 # x moves by 0.5 per row down
        self.assertAlmostEqual(lean(skew_x=-0.25), -0.25, delta=0.03)
        xy = centred(skew_x=0.4, skew_y=0.3, skew_order="XY")
        yx = centred(skew_x=0.4, skew_y=0.3, skew_order="YX")
        self.assertGreater(float(np.abs(xy[..., 3] - yx[..., 3]).sum()), 3.0)
        np.testing.assert_allclose(centred(skew_x=0.4, skew_order="XY"), centred(skew_x=0.4, skew_order="YX"), atol=1e-6)

    def test_the_block_matrix_composes_rotate_skew_scale_then_translate(self):
        m = textknobs.block_matrix(params(rotate=90.0, scale=2.0, translate_x=3.0, translate_y=4.0))
        cx, cy = 100.0, 60.0
        x, y = cx + 10.0, cy                                                         # 10 px right of the pivot
        out = m @ np.array([x, y, 1.0])
        np.testing.assert_allclose(out[:2], [cx + 3.0, cy + 20.0 + 4.0], atol=1e-9)        # scaled to 20 px, turned clockwise (y down), then moved


class RampTests(unittest.TestCase):
    def opaque_pixels(self, image):
        ys, xs = np.nonzero(image[..., 3] > 0.9999)
        return ys, xs

    def test_a_linear_ramp_blends_color0_into_the_text_colour_along_p0_to_p1(self):
        image = draw(message="MMMM", font_size=28.0, ramp="linear", p0_x=20.0, p0_y=0.0, p1_x=180.0, p1_y=0.0,
                     color0_red=1.0, color0_green=0.0, color0_blue=0.0, color0_alpha=1.0, red=0.0, green=0.0, blue=1.0)
        ys, xs = self.opaque_pixels(image)
        self.assertGreater(len(xs), 100)
        t = np.clip((xs - 20.0) / 160.0, 0.0, 1.0)
        np.testing.assert_allclose(image[ys, xs, 0], 1.0 - t, atol=1e-5)
        np.testing.assert_allclose(image[ys, xs, 2], t, atol=1e-5)
        np.testing.assert_allclose(image[ys, xs, 1], 0.0, atol=1e-6)

    def test_the_smooth_ramps_weigh_the_text_colour_as_stated(self):
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(textknobs.ramp_weight("linear", t), t)
        np.testing.assert_allclose(textknobs.ramp_weight("smooth", t), t * t * (3 - 2 * t))
        np.testing.assert_allclose(textknobs.ramp_weight("smooth0", t), t * t)
        np.testing.assert_allclose(textknobs.ramp_weight("smooth1", t), 1 - (1 - t) ** 2)
        for kind in ("smooth", "smooth0", "smooth1"):
            image = draw(message="MMMM", font_size=28.0, ramp=kind, p0_x=20.0, p0_y=0.0, p1_x=180.0, p1_y=0.0,
                         color0_red=1.0, color0_green=0.0, color0_blue=0.0, color0_alpha=1.0, red=0.0, green=0.0, blue=1.0)
            ys, xs = self.opaque_pixels(image)
            weight = textknobs.ramp_weight(kind, np.clip((xs - 20.0) / 160.0, 0.0, 1.0))
            np.testing.assert_allclose(image[ys, xs, 2], weight, atol=1e-5)

    def test_a_ramp_colour0_alpha_fades_the_text_in(self):
        image = draw(message="MMMM", font_size=28.0, ramp="linear", p0_x=20.0, p0_y=0.0, p1_x=180.0, p1_y=0.0,
                     color0_alpha=0.0)
        left, right = image[..., 3][:, :50], image[..., 3][:, 120:]
        self.assertLess(float(left.max()), float(right.max()))

    def test_none_leaves_the_flat_colour(self):
        flat = draw(message="MMMM", red=0.2, green=0.4, blue=0.6)
        ys, xs = self.opaque_pixels(flat)
        np.testing.assert_allclose(flat[ys, xs, :3], np.broadcast_to([0.2, 0.4, 0.6], (len(xs), 3)), atol=1e-5)


class ShadowTests(unittest.TestCase):
    def shadowed(self, **extra):
        return draw(message="MM", font_size=40.0, yjustify="center", enable_shadows=1, shadow_distance=8.0, shadow_angle=0.0,
                    shadow_opacity=1.0, shadow_red=0.0, shadow_green=1.0, shadow_blue=0.0, red=1.0, green=0.0, blue=0.0, **extra)

    def test_the_shadow_lies_under_the_text_along_the_angle_and_distance(self):
        plain = draw(message="MM", font_size=40.0, yjustify="center", red=1.0, green=0.0, blue=0.0)
        shadow = self.shadowed()
        solid = plain[..., 3] > 0.999
        np.testing.assert_allclose(shadow[solid], plain[solid], atol=1e-6)               # opaque text is untouched
        moved = np.zeros_like(plain[..., 3])
        moved[:, 8:] = plain[:, :-8, 3]                                                 # angle 0: to the right
        expect = moved * (1.0 - plain[..., 3])
        np.testing.assert_allclose(shadow[..., 1], expect, atol=1e-5)                    # green shadow colour
        np.testing.assert_allclose(shadow[..., 3], plain[..., 3] + expect, atol=1e-5)

    def test_the_angle_turns_the_offset(self):
        for angle, (dx, dy) in ((90.0, (0, -6)), (180.0, (-6, 0)), (270.0, (0, 6))):
            plain = draw(message="MM", font_size=40.0, yjustify="center")
            shadow = draw(message="MM", font_size=40.0, yjustify="center", enable_shadows=1, shadow_distance=6.0, shadow_angle=angle,
                          shadow_opacity=1.0)
            moved = np.roll(np.roll(plain[..., 3], dy, axis=0), dx, axis=1)
            np.testing.assert_allclose(shadow[..., 3], plain[..., 3] + moved * (1 - plain[..., 3]), atol=1e-5)

    def test_opacity_and_the_colour_alpha_scale_the_shadow(self):
        full, half = self.shadowed(), draw(message="MM", font_size=40.0, yjustify="center", enable_shadows=1, shadow_distance=8.0,
                                           shadow_angle=0.0, shadow_opacity=0.5, shadow_red=0.0, shadow_green=1.0, shadow_blue=0.0,
                                           red=1.0, green=0.0, blue=0.0)
        plain = draw(message="MM", font_size=40.0, yjustify="center", red=1.0, green=0.0, blue=0.0)
        gap = (full[..., 3] - plain[..., 3]) > 0.5
        self.assertTrue(gap.any())
        np.testing.assert_allclose((half[..., 3] - plain[..., 3])[gap], 0.5 * (full[..., 3] - plain[..., 3])[gap], atol=1e-5)
        dim = draw(message="MM", font_size=40.0, yjustify="center", enable_shadows=1, shadow_distance=8.0, shadow_angle=0.0,
                   shadow_opacity=1.0, shadow_alpha=0.25)
        np.testing.assert_allclose(float((dim[..., 3] - plain[..., 3])[gap].max()), 0.25, atol=1e-5)

    def test_off_and_zero_opacity_draw_the_text_alone(self):
        plain = draw(message="MM", font_size=40.0, yjustify="center")
        np.testing.assert_array_equal(draw(message="MM", font_size=40.0, yjustify="center", enable_shadows=0, shadow_distance=9.0), plain)
        np.testing.assert_allclose(draw(message="MM", font_size=40.0, yjustify="center", enable_shadows=1, shadow_opacity=0.0), plain, atol=1e-7)

    def test_softness_blurs_the_shadow_and_expand_grows_or_shrinks_it(self):
        hard = self.shadowed()[..., 3]
        soft = self.shadowed(shadow_softness=4.0)[..., 3]
        self.assertGreater(float(np.abs(soft - hard).sum()), 5.0)
        self.assertAlmostEqual(float(soft.sum()), float(hard.sum()), delta=float(hard.sum()) * 0.03)       # blur moves energy; it does not make any
        bigger = self.shadowed(shadow_expand=2.0)[..., 3]
        smaller = self.shadowed(shadow_expand=-2.0)[..., 3]
        self.assertGreater(float(bigger.sum()), float(hard.sum()))
        self.assertLess(float(smaller.sum()), float(hard.sum()))

    def test_inherit_input_colour_uses_the_text_colour(self):
        shadow = draw(message="MM", font_size=40.0, yjustify="center", enable_shadows=1, shadow_inherit_input_color=1, shadow_distance=8.0,
                      shadow_angle=0.0, shadow_opacity=1.0, red=0.2, green=0.4, blue=0.8, shadow_red=1.0, shadow_green=0.0, shadow_blue=0.0)
        plain = draw(message="MM", font_size=40.0, yjustify="center", red=0.2, green=0.4, blue=0.8)
        gap = (plain[..., 3] == 0.0) & (shadow[..., 3] > 0.999)
        self.assertTrue(gap.any())
        np.testing.assert_allclose(shadow[gap][:, :3], np.broadcast_to([0.2, 0.4, 0.8], (int(gap.sum()), 3)), atol=1e-5)

    def test_a_shadow_near_the_edge_of_a_window_is_the_same_as_in_the_whole_frame(self):
        p = params(message="MM", font_size=40.0, yjustify="center", enable_shadows=1, shadow_distance=8.0, shadow_angle=315.0,
                   shadow_softness=3.0, shadow_opacity=0.8)
        whole = Evaluator._draw_shape("Text", p, 0, 0, W, H, 1)
        for x0, y0, w, h in ((0, 0, 60, 50), (60, 20, 50, 40), (100, 50, 60, 50)):
            np.testing.assert_allclose(Evaluator._draw_shape("Text", p, x0, y0, w, h, 1), whole[y0:y0 + h, x0:x0 + w], atol=5e-3)


class TileTests(unittest.TestCase):
    def test_tiles_equal_the_evaluator_for_the_new_knobs(self):
        variants = (dict(yjustify="bottom", kerning=2.0, tracking=1.0), dict(rotate=12.0, skew_x=0.2, scale=1.3, translate_x=5.0),
                    dict(ramp="smooth", p0_x=10.0, p1_x=140.0, p1_y=100.0, color0_red=1.0, color0_alpha=1.0),
                    dict(enable_shadows=1, shadow_softness=3.0, shadow_distance=7.0), dict(font_width=150.0, font_height=80.0, leading=4.0, index="Bold"))
        for extra in variants:
            with self.subTest(extra=extra):
                d = Dispatcher(empty_document())
                d.execute({"op": "create", "id": "t", "type": "Text", "params": {"width": W, "height": H, "message": "Tile\nseams",
                           "font_size": 24.0, "box_x": 8.0, "box_y": 6.0, "box_width": 140.0, "box_height": 86.0, **extra}})
                d.execute({"op": "view", "id": "t"})
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=32)
                self.assertTrue(executor.supports_tiled(d.document, "t"))
                region = executor.canvas_region(d.document, "t", frame=1, tier=1)
                got = executor.compose_region(d.document, "t", region, frame=1, tier=1).pixels
                want = Evaluator().evaluate_raster(d.document, target="t").pixels
                np.testing.assert_allclose(got, want, atol=5e-3)


class OldDocumentAndBypassTests(unittest.TestCase):
    def test_a_text_saved_before_these_knobs_loads_and_draws_the_same(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "t", "type": "Text", "params": {"width": W, "height": H, "message": "Old", "font_size": 30.0,
                   "box_x": 10.0, "box_y": 10.0, "box_width": 140.0, "box_height": 80.0, "yjustify": "center"}})
        d.execute({"op": "view", "id": "t"})
        old = copy.deepcopy(d.document)
        for name in textknobs.NEW:
            old["nodes"]["t"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(loaded, target="t").pixels, Evaluator().evaluate_raster(d.document, target="t").pixels)

    def test_a_disabled_text_passes_its_image_through(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "bg", "type": "Constant", "params": {"width": W, "height": H, "red": 0.1, "green": 0.2, "blue": 0.3, "alpha": 1.0}})
        d.execute({"op": "create", "id": "t", "type": "Text", "params": {"width": W, "height": H, "enable_shadows": 1, "rotate": 10.0}})
        d.execute({"op": "connect", "id": "t", "input": "image", "source": "bg"})
        d.execute({"op": "view", "id": "t"})
        doc = copy.deepcopy(d.document)
        doc["nodes"]["t"]["disabled"] = True
        np.testing.assert_array_equal(Evaluator().evaluate_raster(doc, target="t").pixels, Evaluator().evaluate_raster(doc, target="bg").pixels)

    def test_bad_values_are_refused(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "t", "type": "Text", "params": {}})
        for name, value in (("yjustify", "middle"), ("ramp", "wavy"), ("font_width", 0.0), ("shadow_opacity", 2.0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                d.execute({"op": "set", "id": "t", "param": name, "value": value})


if __name__ == "__main__":
    unittest.main()
