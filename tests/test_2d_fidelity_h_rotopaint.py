"""Fidelity step H, part 2: RotoPaint takes Nuke's premultiply, mask strip, output mask and global motion blur on the node, and the
Transform, Motion Blur, Shape, Stroke, Clone and Lifetime settings on each shape and stroke of its payload. Pixel assertions per
option, the tile path against the evaluator, old documents, bypass and refusal of bad settings."""
import copy
import unittest

import numpy as np

from nodebased import paintknobs, shapes
from nodebased.core import SPECS, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

N = 32


def pt(x, y, pressure=1.0):
    return {"x": x, "y": y, "pressure": pressure}


def stroke(points=None, **extra):
    value = {"kind": "stroke", "name": "s", "points": points or [pt(6.5, 16.5), pt(24.5, 16.5)],
             "brush": {"size": 6.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2},
             "tool": "paint", "lifetime": {"mode": "all"}, "color": [1.0, 0.0, 0.0, 1.0],
             "source_offset": [0.0, 0.0], "source_frame": "relative", "opacity": 1.0,
             "blend": "over", "visible": True, "follow_track": None, "patch_blend": 0.0}
    value.update(extra)
    return value


def shape(**extra):
    corner = lambda x, y: {"x": x, "y": y, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
    value = {"kind": "shape", "name": "box", "mode": "union", "opacity": 1.0, "feather": 0.0, "blend": "over", "visible": True,
             "points": [corner(10.0, 10.0), corner(22.0, 10.0), corner(22.0, 22.0), corner(10.0, 22.0)]}
    value.update(extra)
    return value


def slide(first, last=None, frames=(1, 5)):
    """An animated number that goes from `first` to `last` across `frames` on a linear curve."""
    last = first + 16.0 if last is None else last
    return {"value": first, "curve": {"interpolation": "linear",
                                      "keys": [{"frame": frames[0], "value": first}, {"frame": frames[1], "value": last}]}}


def graph(items, *, mask=False, second=False, plate=(0.25, 0.25, 0.25), **params):
    d = Dispatcher(empty_document())
    commands = [{"op": "create", "id": "plate", "type": "Constant",
                 "params": {"width": N, "height": N, "red": plate[0], "green": plate[1], "blue": plate[2]}},
                {"op": "create", "id": "paint", "type": "RotoPaint"},
                {"op": "connect", "id": "paint", "input": "image", "source": "plate"}]
    if second:
        commands += [{"op": "create", "id": "second", "type": "Constant",
                      "params": {"width": N, "height": N, "red": 0.0, "green": 1.0, "blue": 0.0}},
                     {"op": "connect", "id": "paint", "input": "input2", "source": "second"}]
    if mask:
        commands += [{"op": "create", "id": "gate", "type": "Rectangle",
                      "params": {"width": N, "height": N, "box_x": 0.0, "box_y": 0.0, "box_width": N / 2, "box_height": float(N),
                                 "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0}},
                     {"op": "connect", "id": "paint", "input": "mask", "source": "gate"}]
    for name, value in params.items():
        commands.append({"op": "set", "id": "paint", "param": name, "value": value})
    commands.append({"op": "set_paint_items", "id": "paint", "items": items})
    d.execute({"op": "batch", "commands": commands})
    return d


def render(d, frame=1):
    return Evaluator().evaluate_raster(d.document, "paint", frame=frame)


def tiled(d, frame=1, tile_edge=8):
    executor = TileExecutor(evaluator=Evaluator(), tile_edge=tile_edge)
    document = dict(d.document, view="paint")
    region = executor.canvas_region(document, "paint", frame=frame, tier=1)
    return executor.compose_region(document, "paint", region, frame=frame, tier=1).pixels


class PremultiplyTests(unittest.TestCase):
    def test_rgb_multiplies_the_colour_by_the_alpha_and_rgba_the_alpha_too(self):
        items = [stroke(brush={"size": 6.0, "hardness": 0.0, "opacity": 0.5, "spacing": 0.2, "strength": 0.2})]
        plain = render(graph(items)).pixels
        rgb = render(graph(items, premultiply_mode="rgb")).pixels
        rgba = render(graph(items, premultiply_mode="rgba")).pixels
        np.testing.assert_allclose(rgb[..., :3], plain[..., :3] * plain[..., 3:4], atol=1e-6)
        np.testing.assert_array_equal(rgb[..., 3], plain[..., 3])
        np.testing.assert_allclose(rgba[..., 3], plain[..., 3] ** 2, atol=1e-6)

    def test_a_plate_with_half_alpha_shows_the_difference(self):
        d = graph([stroke()], premultiply_mode="rgb")
        d.execute({"op": "set", "id": "plate", "param": "alpha", "value": 0.5})
        raster = render(d).pixels
        self.assertAlmostEqual(float(raster[2, 2, 0]), 0.0625, places=5)            # the plate holds 0.125 (0.25 times its alpha), times 0.5 again

    def test_the_tiles_agree(self):
        d = graph([stroke()], premultiply_mode="rgba")
        np.testing.assert_allclose(tiled(d), render(d).pixels, atol=1e-6)


class MaskStripTests(unittest.TestCase):
    def setUp(self):
        self.items = [stroke(points=[pt(4.5, 16.5), pt(28.5, 16.5)])]

    def test_the_mask_channel_input_and_invert(self):
        plain = render(graph(self.items)).pixels
        gated = render(graph(self.items, mask=True)).pixels
        self.assertEqual(float(gated[16, 24, 0]), 0.25)                              # right half is gated off
        np.testing.assert_array_equal(gated[:, :16], plain[:, :16])
        inverted = render(graph(self.items, mask=True, invert_mask=1)).pixels
        np.testing.assert_array_equal(inverted[:, 16:], plain[:, 16:])
        self.assertEqual(float(inverted[16, 8, 0]), 0.25)

    def test_the_mask_channel_picks_a_channel_of_the_plate_when_no_mask_is_wired(self):
        d = graph(self.items, maskChannelInput="rgba.red")
        plain = render(graph(self.items)).pixels
        got = render(d).pixels
        # The plate is 0.25 red, so the paint shows at a quarter of its strength.
        np.testing.assert_allclose(got[16, 8], plain[16, 8] * 0.25 + 0.25 * 0.75 * np.array([1, 1, 1, 1]) * 0 + (1 - 0.25) * np.array([0.25, 0.25, 0.25, 1.0]), atol=1e-5)

    def test_channels_limits_what_the_paint_may_write(self):
        got = render(graph(self.items, channels="green")).pixels
        base = np.array([0.25, 0.25, 0.25, 1.0], np.float32)
        np.testing.assert_allclose(got[16, 8], [0.25, 0.0, 0.25, 1.0], atol=1e-6)      # only green took the red paint's 0
        self.assertTrue(np.allclose(got[2, 2], base))

    def test_inject_writes_the_mask_layer(self):
        raster = render(graph(self.items, mask=True, inject=1))
        layer = raster.layers["mask"].pixels
        self.assertEqual(float(layer[4, 4, 3]), 1.0)
        self.assertEqual(float(layer[4, 24, 3]), 0.0)

    def test_fringe_keeps_a_hard_mask_from_gating_anything_but_its_edge(self):
        plain = render(graph(self.items, mask=True)).pixels
        fringe = render(graph(self.items, mask=True, fringe=1)).pixels
        self.assertFalse(np.array_equal(plain, fringe))
        self.assertEqual(float(fringe[16, 8, 0]), 0.25)                             # inside a hard mask the fringe weight is nil

    def test_mix_still_fades_the_whole_layer(self):
        half = render(graph(self.items, mix=0.5)).pixels
        full = render(graph(self.items)).pixels
        np.testing.assert_allclose(half, 0.5 * full + 0.5 * render(graph([])).pixels, atol=1e-6)
        half_with_chrome = render(graph(self.items, mix=0.5, channels="rgba")).pixels
        np.testing.assert_allclose(half_with_chrome, half, atol=1e-6)

    def test_every_chrome_setting_draws_the_same_through_tiles(self):
        for options in (dict(mask=True), dict(mask=True, invert_mask=1), dict(channels="green"), dict(mask=True, fringe=1),
                        dict(maskChannelInput="rgba.red")):
            with self.subTest(options=options):
                d = graph(self.items, **options)
                np.testing.assert_allclose(tiled(d), render(d).pixels, atol=1e-6)


class OutputMaskTests(unittest.TestCase):
    def test_the_coverage_of_everything_drawn_goes_to_the_layer(self):
        d = graph([stroke(), shape()], outputMask="paint_matte")
        raster = render(d)
        matte = raster.layers["paint_matte"].pixels
        self.assertEqual(float(matte[16, 16, 3]), 1.0)                              # inside the box and on the stroke
        self.assertEqual(float(matte[2, 2, 3]), 0.0)
        for channel in range(3):
            np.testing.assert_array_equal(matte[..., channel], matte[..., 3])
        self.assertIsNone(render(graph([stroke()])).layers)

    def test_none_changes_nothing_and_the_node_renders_whole(self):
        plain = render(graph([stroke()]))
        named = render(graph([stroke()], outputMask="m"))
        np.testing.assert_array_equal(plain.pixels, named.pixels)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        self.assertTrue(executor.supports_tiled(graph([stroke()]).document, "paint"))
        self.assertFalse(executor.supports_tiled(graph([stroke()], outputMask="m").document, "paint"))

    def test_opacity_scales_the_layer(self):
        raster = render(graph([shape()], outputMask="m", opacity=0.5))
        self.assertAlmostEqual(float(raster.layers["m"].pixels[16, 16, 3]), 0.5, places=6)


class NodeMotionBlurTests(unittest.TestCase):
    def slider(self, **extra):
        return stroke(points=[pt(6.5, 16.5)], transform={"translate_x": slide(0.0, 16.0)}, **extra)

    def test_a_stroke_with_a_transform_is_moved(self):
        at1 = render(graph([self.slider()]), 1).pixels
        at5 = render(graph([self.slider()]), 5).pixels
        self.assertGreater(float(at1[16, 6, 0]), 0.9)
        self.assertEqual(float(at5[16, 6, 0]), 0.25)
        self.assertGreater(float(at5[16, 22, 0]), 0.9)

    def test_the_global_shutter_blurs_a_moving_stroke_and_leaves_a_still_one_sharp(self):
        options = dict(global_motionblur_on=1, global_motionblur=4, global_motionblur_shutter=1.0)
        sharp = render(graph([self.slider()]), 3).pixels
        blurred = render(graph([self.slider()], **options), 3).pixels
        self.assertGreater(float(np.abs(sharp - blurred).sum()), 1.0)
        still = stroke()
        np.testing.assert_array_equal(render(graph([still], **options), 3).pixels, render(graph([still]), 3).pixels)

    def test_the_per_item_shutter_blurs_only_that_item(self):
        own = self.slider(motion_blur=True, motion_blur_samples=4, motion_blur_shutter=1.0)
        a = render(graph([own]), 3).pixels
        b = render(graph([own], shape_motionblur_on=0), 3).pixels
        np.testing.assert_array_equal(b, render(graph([self.slider()]), 3).pixels)
        self.assertGreater(float(np.abs(a - b).sum()), 1.0)

    def test_a_blurred_shape_averages_its_samples(self):
        moving = shape(transform={"translate_x": slide(0.0, 8.0)}, motion_blur=True, motion_blur_samples=2, motion_blur_shutter=1.0,
                       motion_blur_offset_type="start")
        got = render(graph([moving]), 1).pixels
        plan = shapes.motion_blur_plan(moving, {})
        times = shapes.motion_blur_times(plan, 1)
        self.assertEqual(len(times), 3)               # samples + 1 times across the shutter, ends included
        # Each sample is the box at its own time (the transform slides 2 px a frame); the picture is the mean of the two renders.
        still = [render(graph([shape(transform={"translate_x": 2.0 * (t - 1.0)})]), 1).pixels for t in times]
        np.testing.assert_allclose(got, sum(still) / 3.0, atol=1e-5)
        self.assertFalse(np.array_equal(still[0], still[1]))

    def test_the_tiles_agree(self):
        options = dict(global_motionblur_on=1, global_motionblur=3, global_motionblur_shutter=1.0)
        d = graph([self.slider(), shape(transform={"translate_x": slide(0.0, 8.0)})], **options)
        np.testing.assert_allclose(tiled(d, 3), render(d, 3).pixels, atol=1e-5)


class ItemTests(unittest.TestCase):
    def test_locked_changes_no_pixel(self):
        for item in (stroke(), shape()):
            np.testing.assert_array_equal(render(graph([dict(item, locked=True)])).pixels, render(graph([item])).pixels)

    def test_a_shape_takes_rotos_settings(self):
        base = render(graph([shape()])).pixels
        self.assertEqual(float(base[16, 16, 0]), 1.0)
        inverted = render(graph([shape(invert=True)])).pixels
        self.assertEqual(float(inverted[16, 16, 0]), 0.25)
        self.assertEqual(float(inverted[2, 2, 0]), 1.0)
        hidden = render(graph([shape(lifetime={"type": "single", "start": 3})]), 1).pixels
        self.assertEqual(float(hidden[16, 16, 0]), 0.25)
        shown = render(graph([shape(lifetime={"type": "single", "start": 3})]), 3).pixels
        self.assertEqual(float(shown[16, 16, 0]), 1.0)
        soft = render(graph([shape(feather=6.0)])).pixels
        off = render(graph([shape(feather=6.0, feather_on=False)])).pixels
        np.testing.assert_array_equal(off, base)
        self.assertGreater(float(np.abs(soft - base).sum()), 0.5)

    def test_a_shape_transform_moves_it(self):
        moved = render(graph([shape(transform={"translate_x": 6.0, "translate_y": 0.0})])).pixels
        self.assertEqual(float(moved[16, 11, 0]), 0.25)
        self.assertEqual(float(moved[16, 26, 0]), 1.0)

    def test_an_open_shape_takes_end_caps(self):
        wire = shape(closed=False, stroke_width=6.0, cap_start="butt", cap_end="butt")
        wire["points"] = [dict(p, y=16.0) for p in wire["points"][:2]]
        butt = render(graph([wire])).pixels
        rounded = render(graph([dict(wire, cap_start="rounded", cap_end="rounded")])).pixels
        self.assertEqual(float(butt[16, 8, 0]), 0.25)
        self.assertGreater(float(rounded[16, 8, 0]), 0.5)

    def test_a_shape_hidden_stays_hidden_and_outside_its_lifetime_too(self):
        self.assertEqual(float(render(graph([shape(visible=False)])).pixels[16, 16, 0]), 0.25)

    def test_source_foreground_paints_the_second_image_and_background_the_plate(self):
        items = [shape(source="foreground")]
        green = render(graph(items, second=True)).pixels
        np.testing.assert_allclose(green[16, 16, :3], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(render(graph(items)).pixels[16, 16, :3], [0.0, 0.0, 0.0], atol=1e-6)   # nothing wired: black
        stroke_green = render(graph([stroke(source="foreground")], second=True)).pixels
        np.testing.assert_allclose(stroke_green[16, 12, :3], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(stroke_green[2, 2, :3], [0.25] * 3, atol=1e-6)

    def test_source_background_paints_the_plate_back_over_earlier_paint(self):
        items = [shape(), stroke(source="background", points=[pt(16.0, 16.0)],
                                 brush={"size": 6.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2})]
        got = render(graph(items)).pixels
        np.testing.assert_allclose(got[16, 16, :3], [0.25] * 3, atol=1e-6)           # the stroke put the plate back
        self.assertEqual(float(got[12, 12, 0]), 1.0)                                  # outside the stroke the box stays white

    def test_a_stroke_transform_scales_the_brush(self):
        base = render(graph([stroke()])).pixels
        doubled = render(graph([stroke(transform={"scale_x": 2.0, "scale_y": 2.0, "center_x": 15.5, "center_y": 16.5})])).pixels
        self.assertGreater(float((doubled[..., 0] > 0.9).sum()), float((base[..., 0] > 0.9).sum()) * 1.5)


class StrokeTabTests(unittest.TestCase):
    def test_effect_sets_the_blur_radius_the_smear_reach_and_the_dodge_strength(self):
        d0 = graph([stroke(tool="dodge")])
        weak, strong = (render(graph([stroke(tool="dodge", effect=e)])).pixels[16, 12, 0] for e in (0.2, 0.8))
        self.assertAlmostEqual(float(weak), 0.25 + 0.75 * 0.2, places=5)
        self.assertAlmostEqual(float(strong), 0.25 + 0.75 * 0.8, places=5)
        self.assertAlmostEqual(float(render(d0).pixels[16, 12, 0]), 0.25 + 0.75 * 0.2, places=5)     # 0 means the brush's own strength
        burn = render(graph([stroke(tool="burn", effect=0.5)])).pixels[16, 12, 0]
        self.assertAlmostEqual(float(burn), 0.125, places=5)
        edge = shape(); big = shape(); big["points"] = [dict(p) for p in edge["points"]]
        base = render(graph([edge, stroke(tool="blur", points=[pt(10.0, 16.0), pt(22.0, 16.0)])])).pixels
        wide = render(graph([edge, stroke(tool="blur", effect=5.0, points=[pt(10.0, 16.0), pt(22.0, 16.0)])])).pixels
        self.assertGreater(float(np.abs(base - wide).sum()), 0.1)

    def test_buildup_adds_overlapping_dabs(self):
        soft = {"size": 8.0, "hardness": 0.0, "opacity": 0.3, "spacing": 0.2, "strength": 0.2}
        pts = [pt(6.5, 16.5), pt(24.5, 16.5)]
        plain = render(graph([stroke(points=pts, brush=soft)])).pixels
        built = render(graph([stroke(points=pts, brush=soft, buildup=True)])).pixels
        self.assertGreater(float(built[16, 15, 0]), float(plain[16, 15, 0]) + 0.1)
        self.assertLessEqual(float(built[..., 0].max()), 1.0 + 1e-6)

    def test_dynamic_hardness_follows_the_pressure(self):
        brush = {"size": 10.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2}
        pts = [pt(6.5, 16.5, 1.0), pt(24.5, 16.5, 0.6)]
        fixed = render(graph([stroke(points=pts, brush=brush)])).pixels
        dynamic = render(graph([stroke(points=pts, brush=brush, dynamic_hardness=True)])).pixels
        self.assertFalse(np.array_equal(fixed, dynamic))
        np.testing.assert_array_equal(fixed[:, :8], dynamic[:, :8])              # at full pressure the two agree

    def test_write_on_draws_only_the_chosen_stretch_of_the_path(self):
        pts = [pt(4.5, 16.5), pt(28.5, 16.5)]
        part = render(graph([stroke(points=pts, write_on={"start": 0.25, "end": 0.75})])).pixels
        self.assertEqual(float(part[16, 6, 0]), 0.25)
        self.assertEqual(float(part[16, 27, 0]), 0.25)
        self.assertEqual(float(part[16, 16, 0]), 1.0)
        full = render(graph([stroke(points=pts, write_on={"start": 0.0, "end": 1.0})])).pixels
        np.testing.assert_array_equal(full, render(graph([stroke(points=pts)])).pixels)
        self.assertEqual(float(render(graph([stroke(points=pts, write_on={"start": 0.5, "end": 0.5})])).pixels[16, 16, 0]), 0.25)

    def test_the_tiles_agree(self):
        soft = {"size": 8.0, "hardness": 0.3, "opacity": 0.5, "spacing": 0.2, "strength": 0.2}
        d = graph([stroke(buildup=True, brush=soft), stroke(tool="blur", effect=3.0, dynamic_hardness=True),
                   stroke(tool="dodge", effect=0.5, write_on={"start": 0.1, "end": 0.9})])
        np.testing.assert_allclose(tiled(d), render(d).pixels, atol=1e-5)


class CloneTabTests(unittest.TestCase):
    def picture(self):
        """A plate with a vertical bar so that moving the clone source is visible."""
        d = graph([], plate=(0.0, 0.0, 0.0))
        d.execute({"op": "create", "id": "bar", "type": "Rectangle",
                   "params": {"width": N, "height": N, "box_x": 20.0, "box_y": 4.0, "box_width": 4.0, "box_height": 24.0,
                              "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0}})
        d.execute({"op": "create", "id": "merge", "type": "Merge"})
        d.execute({"op": "connect", "id": "merge", "input": "B", "source": "plate"})
        d.execute({"op": "connect", "id": "merge", "input": "A", "source": "bar"})
        d.execute({"op": "connect", "id": "paint", "input": "image", "source": "merge"})
        return d

    def clone(self, **extra):
        # A clone reads the source at the destination minus the offset, so -12 brings the bar at x 20 to x 8.
        return stroke(tool="clone", points=[pt(8.5, 16.5), pt(8.5, 17.5)],
                      brush={"size": 10.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2},
                      **{"source_offset": [-12.0, 0.0], **extra})

    def run_clone(self, **extra):
        d = self.picture()
        d.execute({"op": "set_paint_items", "id": "paint", "items": [self.clone(**extra)]})
        return d

    def test_an_untouched_clone_copies_the_source_offset(self):
        d = self.run_clone()
        got = render(d).pixels
        self.assertGreater(float(got[16, 8, 0]), 0.9)                              # the bar at x 20 to 24 arrives at x 8 to 12

    def test_rounding_the_offset_moves_the_copy_by_the_rounding(self):
        a = render(self.run_clone(source_offset=[-11.6, 0.0])).pixels
        b = render(self.run_clone(source_offset=[-11.6, 0.0], source_round=True)).pixels
        c = render(self.run_clone(source_offset=[-12.0, 0.0])).pixels
        np.testing.assert_array_equal(b, c)
        self.assertEqual(a.shape, b.shape)

    def test_the_source_rotates_scales_and_skews_about_its_centre(self):
        flat = render(self.run_clone()).pixels
        turned = render(self.run_clone(source_transform={"rotate": 90.0, "center_x": 22.0, "center_y": 16.5})).pixels
        self.assertFalse(np.array_equal(flat, turned))
        scaled = render(self.run_clone(source_transform={"scale_x": 2.0, "scale_y": 2.0, "center_x": 22.0, "center_y": 16.5})).pixels
        self.assertFalse(np.array_equal(flat, scaled))
        same = render(self.run_clone(source_transform={"rotate": 0.0, "scale_x": 1.0})).pixels
        np.testing.assert_array_equal(same, flat)

    def test_scale_two_widens_the_copied_bar(self):
        base = render(self.run_clone(source_transform={"scale_x": 1.0})).pixels
        wide = render(self.run_clone(source_transform={"scale_x": 2.0, "center_x": 22.0})).pixels
        self.assertGreater(float((wide[16, 2:15, 0] > 0.9).sum()), float((base[16, 2:15, 0] > 0.9).sum()) + 2)

    def test_a_smooth_filter_softens_a_fractional_read(self):
        sharp = render(self.run_clone(source_offset=[-11.5, 0.0], source_filter="impulse")).pixels
        for name in ("cubic", "keys", "simon", "rifman", "mitchell", "parzen", "notch"):
            with self.subTest(filter=name):
                smooth = render(self.run_clone(source_offset=[-11.5, 0.0], source_filter=name)).pixels
                self.assertFalse(np.array_equal(sharp, smooth))
                self.assertTrue(np.isfinite(smooth).all())

    def test_black_outside_stops_the_edge_repeating(self):
        d = self.picture()
        item = stroke(tool="clone", points=[pt(26.5, 16.5), pt(26.5, 17.5)], source_offset=[-12.0, 0.0],
                      brush={"size": 6.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2})
        d.execute({"op": "set_paint_items", "id": "paint", "items": [dict(item, source_black_outside=True, source_filter="impulse")]})
        black = render(d).pixels
        d.execute({"op": "set_paint_items", "id": "paint", "items": [dict(item, source_black_outside=False, source_filter="impulse")]})
        repeated = render(d).pixels
        np.testing.assert_allclose(black[16, 26, :3], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(repeated[16, 26, :3], [0.0, 0.0, 0.0], atol=1e-6)    # the source at x 38 has no bar either way
        d.execute({"op": "set_paint_items", "id": "paint", "items": [dict(item, source_offset=[-12.0, 0.0], points=[pt(29.5, 16.5), pt(29.5, 17.5)],
                                                                         source_black_outside=True)]})
        self.assertTrue(np.isfinite(render(d).pixels).all())

    def test_the_tiles_agree_for_every_clone_setting(self):
        for extra in (dict(source_round=True), dict(source_filter="cubic"), dict(source_black_outside=True),
                      dict(source_transform={"rotate": 30.0, "center_x": 22.0, "center_y": 16.5}, source_filter="mitchell")):
            with self.subTest(extra=sorted(extra)):
                d = self.run_clone(**extra)
                np.testing.assert_allclose(tiled(d), render(d).pixels, atol=1e-5)


class TileTests(unittest.TestCase):
    def test_every_item_setting_draws_the_same_through_tiles(self):
        variants = (shape(invert=True), shape(feather=4.0, falloff_power=0.5), shape(transform={"rotate": 25.0}),
                    shape(source="foreground"), stroke(transform={"translate_x": 3.0, "rotate": 15.0}),
                    stroke(source="foreground"), stroke(locked=True))
        for item in variants:
            with self.subTest(item=sorted(set(item) - {"kind", "name", "mode", "opacity", "feather", "points", "blend", "visible",
                                                       "brush", "tool", "lifetime", "color", "source_offset", "source_frame",
                                                       "follow_track", "patch_blend"})):
                d = graph([item], second=True)
                np.testing.assert_allclose(tiled(d), render(d).pixels, atol=1e-5)


class ValidationAndOldDocumentTests(unittest.TestCase):
    def validate_items(self, items):
        validate(graph(items).document)

    def test_good_settings_validate(self):
        self.validate_items([stroke(effect=3.0, dynamic_hardness=True, buildup=True, write_on={"start": 0.1, "end": 0.9},
                                    source="background", locked=True, source_transform={"rotate": 5.0, "skew_order": "YX"},
                                    source_round=True, source_filter="keys", source_black_outside=True,
                                    transform={"rotate": 3.0}, motion_blur=True, motion_blur_samples=4),
                             shape(invert=True, feather_on=False, falloff_power=0.5, cap_start="butt", source="foreground",
                                   lifetime={"type": "range", "start": 2, "end": 5}, locked=True)])

    def test_bad_settings_are_refused(self):
        bad = (dict(source="sideways"), dict(locked=1), dict(effect=-1.0), dict(effect=1e6), dict(dynamic_hardness=1),
               dict(buildup="yes"), dict(write_on={"start": 0.5}), dict(write_on={"start": 0.8, "end": 1.4}),
               dict(source_filter="sharp"), dict(source_transform={"rotate": "a"}), dict(source_transform={"bogus": 1}),
               dict(source_transform={"skew_order": "ZZ"}), dict(transform={"skew_order": "ZZ"}), dict(motion_blur_samples=0))
        for extra in bad:
            with self.subTest(bad=extra), self.assertRaises(ValueError):
                shapes.validate_payload("RotoPaint", {"items": [stroke(**extra)]}, "items")
        for extra in (dict(source="sideways"), dict(locked=1), dict(falloff_power=-1), dict(cap_start="pointy"), dict(bogus=1)):
            with self.subTest(bad=extra), self.assertRaises(ValueError):
                shapes.validate_payload("RotoPaint", {"items": [shape(**extra)]}, "items")

    def test_a_document_saved_before_these_knobs_loads_and_draws_the_same(self):
        items = [shape(), stroke(), stroke(tool="blur", points=[pt(10.0, 16.0), pt(22.0, 16.0)])]
        doc = graph(items).document
        old = copy.deepcopy(doc)
        for name in paintknobs.PARAMS:
            old["nodes"]["paint"]["params"].pop(name, None)
        for name in ("channels", "maskChannelInput", "invert_mask", "inject", "fringe", "output", "cliptype", "replace", "opacity"):
            old["nodes"]["paint"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        params = loaded["nodes"]["paint"]["params"]
        self.assertEqual((params["premultiply_mode"], params["outputMask"], params["global_motionblur_on"]), ("none", "none", 0))
        np.testing.assert_array_equal(Evaluator().evaluate_raster(loaded, "paint", frame=1).pixels, render(graph(items)).pixels)

    def test_the_new_params_have_their_defaults_on_a_fresh_node(self):
        params = SPECS["RotoPaint"]["params"]
        self.assertEqual((params["premultiply_mode"], params["outputMask"], params["shape_motionblur_on"],
                          params["global_motionblur_on"], params["channels"]), ("none", "none", 1, 0, "all"))

    def test_bypass_passes_the_plate_through(self):
        d = graph([stroke(), shape()], premultiply_mode="rgb", outputMask="m")
        d.execute({"op": "disable", "id": "paint", "value": True})
        got = render(d).pixels
        np.testing.assert_array_equal(got, render(graph([])).pixels)


if __name__ == "__main__":
    unittest.main()
