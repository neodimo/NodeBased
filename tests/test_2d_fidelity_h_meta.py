"""Fidelity step H, part 4: AddTimeCode takes Nuke's get-FPS-from-metadata and use-start-frame, AppendClip its fade in, fade out,
metadata-from-first and last frame, and BurnIn its opacity, padding, text scale, font style, top and bottom middle slots and
background colour and borders. Pixel and value assertions for each, old documents, bypass and bad settings."""
import copy
import unittest

import numpy as np

from nodebased import metaknobs
from nodebased import metadata as md
from nodebased.core import LIMITS, SPECS, upgrade_document
from nodebased.imaging import Evaluator, append_clip_layout
from tests.test_2d_parity_group_3b import Graph
from tests.test_2d_parity_step_4b import Graph as TimeGraph, frame_plate

W, H = 200, 60


def meta_of(g, key, frame=1, evaluator=None):
    return (evaluator or Evaluator()).evaluate_raster(dict(g.doc, view=key), frame=frame).meta


def constant(g, key="src", **params):
    return g.add(key, "Constant", dict(width=W, height=H, red=0.2, green=0.2, blue=0.2, alpha=1.0, **params))


class AddTimeCodeTests(unittest.TestCase):
    def stamped(self, frame, rate=None, **params):
        g = Graph()
        constant(g)
        source = "src"
        if rate is not None:
            g.add("set", "ModifyMetaData", {"edits": f"set input/frame_rate {rate}"}, image="src")
            source = "set"
        g.add("tc", "AddTimeCode", params, image=source)
        return meta_of(g, "tc", frame)["timecode"]

    def test_metafps_reads_the_rate_from_the_inputs_metadata(self):
        base = dict(timecode="00:00:00:00", fps=24.0, start_frame=1)
        self.assertEqual(self.stamped(26, **base), "00:00:01:01")                                  # 24 fps
        self.assertEqual(self.stamped(26, rate=25, **base), "00:00:01:01")                          # metadata ignored while off
        self.assertEqual(self.stamped(26, rate=25, metafps=1, **base), "00:00:01:00")               # 25 fps from the metadata
        self.assertEqual(self.stamped(26, rate="50", metafps=1, **base), "00:00:00:25")

    def test_metafps_falls_back_to_the_fps_knob_and_reads_fractions(self):
        base = dict(timecode="00:00:00:00", fps=24.0, start_frame=1, metafps=1)
        self.assertEqual(self.stamped(26, **base), "00:00:01:01")                                   # no key: the knob
        self.assertEqual(self.stamped(26, rate="nonsense", **base), "00:00:01:01")
        self.assertAlmostEqual(metaknobs.parse_rate("24000/1001"), 23.976, places=3)
        self.assertEqual(metaknobs.parse_rate("0"), None)
        self.assertEqual(metaknobs.parse_rate("-4"), None)
        self.assertEqual(metaknobs.time_code_rate({"metafps": 1, "fps": 30.0}, {"exr/framesPerSecond": "25"}), 25.0)

    def test_use_frame_off_counts_from_the_documents_first_frame(self):
        g = Graph()
        constant(g)
        g.add("tc", "AddTimeCode", dict(timecode="01:00:00:00", start_frame=10, use_frame=0), image="src")
        g.d.execute({"op": "time", "first": 3, "last": 40, "current": 3})
        self.assertEqual(meta_of(g, "tc", 3)["timecode"], "01:00:00:00")
        self.assertEqual(meta_of(g, "tc", 5)["timecode"], "01:00:00:02")
        g.d.execute({"op": "set", "id": "tc", "param": "use_frame", "value": 1})
        self.assertEqual(meta_of(g, "tc", 10)["timecode"], "01:00:00:00")                           # the start frame itself
        self.assertEqual(meta_of(g, "tc", 3)["timecode"], "00:59:59:17")                            # seven frames before it

    def test_the_documents_range_is_part_of_the_cache_key(self):
        g = Graph()
        constant(g)
        g.add("tc", "AddTimeCode", dict(timecode="01:00:00:00", start_frame=10, use_frame=0), image="src")
        evaluator = Evaluator()
        g.d.execute({"op": "time", "first": 1, "last": 40, "current": 1})
        self.assertEqual(meta_of(g, "tc", 4, evaluator)["timecode"], "01:00:00:03")
        g.d.execute({"op": "time", "first": 2, "last": 40, "current": 2})
        self.assertEqual(meta_of(g, "tc", 4, evaluator)["timecode"], "01:00:00:02")

    def test_an_older_node_stamps_as_before(self):
        g = Graph()
        constant(g)
        g.add("tc", "AddTimeCode", dict(timecode="00:00:00:00", start_frame=0), image="src")
        doc = copy.deepcopy(g.doc)
        for name in ("metafps", "use_frame"):
            del doc["nodes"]["tc"]["params"][name]
        loaded = upgrade_document(doc)
        self.assertEqual((loaded["nodes"]["tc"]["params"]["metafps"], loaded["nodes"]["tc"]["params"]["use_frame"]), (0, 1))
        self.assertEqual(Evaluator().evaluate_raster(dict(loaded, view="tc"), frame=24).meta["timecode"], "00:00:01:00")


class AppendClipTests(unittest.TestCase):
    def graph(self, **params):
        g = TimeGraph()
        frame_plate(g, "a", base=0.0)                      # red is 0.05 * frame
        frame_plate(g, "b", base=0.5)
        g.add("node", "AppendClip", dict(length0=4, length1=4, **params), clip0="a", clip1="b")
        return g

    @staticmethod
    def red(g, frame):
        return float(Evaluator().evaluate_raster(dict(g.doc, view="node"), frame=frame).pixels[0, 0, 0])

    def test_last_frame_is_the_end_of_the_appended_clips(self):
        clips, starts, overlaps, first, last = append_clip_layout(self.graph().doc["nodes"], self.graph().doc["nodes"]["node"],
                                                                  self.graph().doc["nodes"]["node"]["params"])
        self.assertEqual((first, last, starts, overlaps), (1, 8, [1, 5], [0, 0]))
        g = self.graph(first_frame=10, dissolve=2)
        node = g.doc["nodes"]["node"]
        self.assertEqual(append_clip_layout(g.doc["nodes"], node, node["params"])[3:], (10, 15))   # 4 + 4 - 2 overlap, from 10
        # The frames after the last one hold it: the output really ends there.
        self.assertAlmostEqual(self.red(g, 15), self.red(g, 40), places=6)
        self.assertNotAlmostEqual(self.red(g, 14), self.red(g, 15), places=6)

    def test_fade_in_and_out_dim_the_head_and_the_tail(self):
        plain = self.graph()
        faded = self.graph(fade_in=2, fade_out=3)
        for frame in range(1, 9):
            gain = 1.0
            if frame - 1 < 2:
                gain *= (frame - 1 + 1) / 3.0
            if 8 - (frame - 1) <= 3:
                gain *= (8 - (frame - 1)) / 4.0
            self.assertAlmostEqual(self.red(faded, frame), self.red(plain, frame) * gain, places=6, msg=frame)
        self.assertAlmostEqual(self.red(faded, 4), self.red(plain, 4), places=6)                  # the middle is untouched
        self.assertAlmostEqual(self.red(faded, 20), self.red(plain, 20), places=6)                # past the end nothing fades

    def test_a_fade_follows_the_first_frame_and_the_dissolve(self):
        faded = self.graph(first_frame=10, dissolve=2, fade_in=1)
        plain = self.graph(first_frame=10, dissolve=2)
        self.assertAlmostEqual(self.red(faded, 10), self.red(plain, 10) * 0.5, places=6)
        self.assertAlmostEqual(self.red(faded, 11), self.red(plain, 11), places=6)

    def test_metadata_from_the_first_clip(self):
        g = TimeGraph()
        frame_plate(g, "a", base=0.0)
        frame_plate(g, "b", base=0.5)
        g.add("ma", "ModifyMetaData", {"edits": "set show/clip first"}, image="a")
        g.add("mb", "ModifyMetaData", {"edits": "set show/clip second"}, image="b")
        g.add("node", "AppendClip", dict(length0=4, length1=4), clip0="ma", clip1="mb")
        self.assertEqual(meta_of(g, "node", 6)["show/clip"], "second")
        g.d.execute({"op": "set", "id": "node", "param": "meta_from_first", "value": 1})
        self.assertEqual(meta_of(g, "node", 6)["show/clip"], "first")
        self.assertEqual(meta_of(g, "node", 2)["show/clip"], "first")
        pixels = Evaluator().evaluate_raster(dict(g.doc, view="node"), frame=6).pixels
        g.d.execute({"op": "set", "id": "node", "param": "meta_from_first", "value": 0})
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(g.doc, view="node"), frame=6).pixels, pixels)

    def test_an_older_node_plays_as_before_and_bypass_passes_the_first_clip(self):
        g = self.graph()
        doc = copy.deepcopy(g.doc)
        for name in ("fade_in", "fade_out", "meta_from_first"):
            del doc["nodes"]["node"]["params"][name]
        loaded = upgrade_document(doc)
        self.assertEqual(loaded["nodes"]["node"]["params"]["fade_in"], 0)
        self.assertAlmostEqual(float(Evaluator().evaluate_raster(dict(loaded, view="node"), frame=6).pixels[0, 0, 0]),
                               self.red(g, 6), places=6)
        g.bypass("node")
        self.assertAlmostEqual(self.red(g, 6), 0.05 * 6, places=6)                              # clip a, whatever the frame

    def test_the_panel_shows_the_last_frame(self):
        from PySide6.QtWidgets import QLabel
        from tests.test_desktop import APP, Window, release_window
        g = self.graph(dissolve=1)
        window = Window(g.doc)
        self.addCleanup(release_window, self)
        window.show()
        APP.processEvents()
        panel = window.build_node_panel("node")
        label = panel.findChild(QLabel, "appendclip-last-frame")
        self.assertEqual(label.text(), "7")                                                       # 4 + 4 - 1 overlap


class BurnInTests(unittest.TestCase):
    FONT = 12.0

    def burn(self, **params):
        g = Graph()
        constant(g)
        base = {"top_left": "", "top_right": "", "bottom_left": "", "bottom_right": "", "center": "", "font_size": self.FONT,
                "bar": 0}
        base.update(params)
        g.add("burn", "BurnIn", base, image="src")
        return g

    @staticmethod
    def pixels(g, frame=1):
        return Evaluator().evaluate_raster(dict(g.doc, view="burn"), frame=frame).pixels

    def changed(self, g):
        source = Evaluator().evaluate_raster(dict(g.doc, view="src")).pixels
        mask = np.any(self.pixels(g) != source, axis=2)
        ys, xs = np.nonzero(mask)
        return mask, (xs.min(), ys.min(), xs.max(), ys.max()) if mask.any() else None

    def test_opacity_blends_the_whole_overlay(self):
        full = self.pixels(self.burn(top_left="Hello", bar=1, bar_opacity=1.0))
        half = self.pixels(self.burn(top_left="Hello", bar=1, bar_opacity=1.0, opacity=0.5))
        plate = self.pixels(self.burn())
        np.testing.assert_allclose(half, plate + 0.5 * (full - plate), atol=1e-6)
        np.testing.assert_array_equal(self.pixels(self.burn(top_left="Hello", opacity=0.0)), plate)

    def test_the_padding_moves_the_text_in_from_the_edges(self):
        _, near = self.changed(self.burn(top_right="Hi", margin=10.0, y_padding=0.0))
        _, moved = self.changed(self.burn(top_right="Hi", margin=40.0, y_padding=15.0))
        self.assertEqual(near[2] - moved[2], 30)                                  # 30 px further from the right edge
        self.assertEqual(moved[1] - near[1], 15)                                  # and 15 px lower
        _, low = self.changed(self.burn(bottom_left="Hi", y_padding=0.0))
        _, raised = self.changed(self.burn(bottom_left="Hi", y_padding=9.0))
        self.assertEqual(low[3] - raised[3], 9)                                   # a bottom slot rises by the padding
        _, inset = self.changed(self.burn(top_left="Hi", margin=25.0))
        _, flush = self.changed(self.burn(top_left="Hi", margin=5.0))
        self.assertEqual(inset[0] - flush[0], 20)

    def test_text_scale_multiplies_the_font_size(self):
        _, one = self.changed(self.burn(top_left="Hi", text_scale=1.0))
        _, two = self.changed(self.burn(top_left="Hi", text_scale=2.0))
        self.assertGreater(two[2] - two[0], 1.6 * (one[2] - one[0]))
        np.testing.assert_array_equal(self.pixels(self.burn(top_left="Hi", font_size=24.0)),
                                      self.pixels(self.burn(top_left="Hi", font_size=12.0, text_scale=2.0)))

    def test_the_font_style_changes_the_letters(self):
        regular = self.pixels(self.burn(top_left="Hello world", font_style="Regular"))
        bold = self.pixels(self.burn(top_left="Hello world", font_style="Bold"))
        italic = self.pixels(self.burn(top_left="Hello world", font_style="Italic"))
        self.assertFalse(np.array_equal(regular, bold))
        self.assertFalse(np.array_equal(regular, italic))
        np.testing.assert_array_equal(regular, self.pixels(self.burn(top_left="Hello world")))

    def test_top_and_bottom_middle_are_centred_in_their_bands(self):
        _, top = self.changed(self.burn(top_middle="MM"))
        _, bottom = self.changed(self.burn(bottom_middle="MM"))
        for found in (top, bottom):
            self.assertLess(abs((found[0] + found[2]) / 2.0 - W / 2.0), 3.0)
        self.assertLess(top[3], H / 2)
        self.assertGreater(bottom[1], H / 2)

    def test_the_bar_takes_its_colour_and_grows_with_the_y_border(self):
        base = self.pixels(self.burn())
        g = self.burn(top_left="Hi", bar=1, bar_opacity=1.0, bar_red=1.0, bar_green=0.0, bar_blue=0.0)
        got = self.pixels(g)
        np.testing.assert_allclose(got[0, W - 1, :3], [1.0, 0.0, 0.0], atol=1e-6)               # the bar, off the text
        half = self.pixels(self.burn(top_left="Hi", bar=1, bar_opacity=1.0, bar_red=1.0, bar_green=0.0, bar_blue=0.0, bar_alpha=0.5))
        np.testing.assert_allclose(half[0, W - 1, :3], 0.5 * np.array([1.0, 0.0, 0.0]) + 0.5 * base[0, W - 1, :3], atol=1e-6)
        rows = lambda image: int(np.count_nonzero(np.any(image[:, W - 1] != base[:, W - 1], axis=1)))
        taller = self.pixels(self.burn(top_left="Hi", bar=1, bar_opacity=1.0, bar_y_border=7.0))
        self.assertEqual(rows(taller) - rows(self.pixels(self.burn(top_left="Hi", bar=1, bar_opacity=1.0))), 7)

    def test_fit_draws_a_box_around_the_text_grown_by_the_borders(self):
        base = self.pixels(self.burn())
        text_only = self.burn(top_left="Hi")
        _, text_box = self.changed(text_only)
        boxed = self.burn(top_left="Hi", bar=1, bar_opacity=1.0, bar_fit=1, bar_x_border=6.0, bar_y_border=4.0)
        mask, found = self.changed(boxed)
        self.assertEqual((found[0], found[1], found[2], found[3]),
                         (text_box[0] - 6, text_box[1] - 4, text_box[2] + 6, text_box[3] + 4))   # the borders, on every side
        self.assertEqual(float(self.pixels(boxed)[found[1], found[0], 0]), 0.0)                  # filled with the (black) bar
        self.assertFalse(mask[:, W - 20:].any())                                                  # and no full-width bar
        self.assertFalse(np.array_equal(self.pixels(boxed), base))

    def test_an_older_burn_in_draws_the_same_pixels(self):
        g = self.burn(top_left="Hi", top_right="[frame]", bottom_left="Low", center="Mid", bar=1)
        before = self.pixels(g)
        doc = copy.deepcopy(g.doc)
        for name in metaknobs.BURN_IN:
            del doc["nodes"]["burn"]["params"][name]
        loaded = upgrade_document(doc)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(dict(loaded, view="burn"), frame=1).pixels, before)
        g.bypass("burn")
        np.testing.assert_array_equal(self.pixels(g), self.pixels(self.burn()))

    def test_the_new_params_have_their_defaults_and_limits(self):
        params = SPECS["BurnIn"]["params"]
        self.assertEqual((params["opacity"], params["y_padding"], params["text_scale"], params["font_style"]), (1.0, 0.0, 1.0, "Regular"))
        for name in metaknobs.BURN_IN:
            self.assertIn(name, params)
        self.assertEqual(LIMITS["text_scale"][0] > 0.0, True)


if __name__ == "__main__":
    unittest.main()
