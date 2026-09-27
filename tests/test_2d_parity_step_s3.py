"""Lane L2 step S3: image metadata, ViewMetaData, ModifyMetaData, CopyMetaData, CompareMetaData, AddTimeCode
and BurnIn. Metadata rides in `Raster.meta` (string keys, Nuke's names); the tests write real EXRs with header
keys, read them through Read, push them through ordinary nodes, and assert the exact key/value dicts and the
exact BurnIn pixels (against the Text node drawing the same string)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import metadata as md
from nodebased.cachetier import DiskCache
from nodebased.core import CHOICES, LIMITS, METADATA_KINDS, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.media import raster_layer_arrays, write_exr
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from nodebased.tiers import REGION_RULES
from tests.test_2d_parity_group_3b import Graph

W, H = 640, 48
KINDS = (*METADATA_KINDS, "BurnIn")


def plate():
    frame = np.zeros((H, W, 4), np.float32)
    frame[..., :3] = np.linspace(0.05, 0.5, W, dtype=np.float32)[None, :, None]
    frame[..., 3] = 1.0
    return frame


class FileCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def exr(self, name="plate.exr", metadata=None):
        path = self.dir / name
        write_exr(path, plate(), bits="float", metadata=metadata)
        return str(path.resolve())

    def read_graph(self, metadata=None):
        g = Graph()
        self.path = self.exr(metadata=metadata)
        g.add("r", "Read", {"path": self.path})
        return g

    @staticmethod
    def meta(g, key, frame=1, evaluator=None):
        return (evaluator or Evaluator()).evaluate_raster(dict(g.doc, view=key), frame=frame).meta


class HeaderTests(FileCase):
    def test_custom_header_keys_read_back(self):
        g = self.read_graph({"exr/lens": "omid", "show/shot": "sh010", "timecode": "01:00:00:12"})
        meta = self.meta(g, "r")
        self.assertEqual(meta["exr/lens"], "omid")
        self.assertEqual(meta["show/shot"], "sh010")
        self.assertEqual(meta["timecode"], "01:00:00:12")

    def test_read_adds_where_the_pixels_came_from(self):
        meta = self.meta(self.read_graph(), "r")
        self.assertEqual(meta["input/filename"], self.path)
        self.assertEqual((meta["input/width"], meta["input/height"]), (str(W), str(H)))
        self.assertNotIn("input/frame", meta)

    def test_a_sequence_read_reports_its_frame(self):
        for n in (3, 4):
            write_exr(self.dir / f"seq.{n:04d}.exr", plate(), bits="float")
        g = Graph()
        g.add("r", "Read", {"path": str(self.dir / "seq.%04d.exr")})
        e = Evaluator()
        self.assertEqual(self.meta(g, "r", 3, e)["input/frame"], "3")
        self.assertEqual(self.meta(g, "r", 4, e)["input/frame"], "4")
        self.assertTrue(self.meta(g, "r", 4, e)["input/filename"].endswith("seq.0004.exr"))

    def test_layout_attributes_are_not_metadata(self):
        meta = self.meta(self.read_graph({"exr/lens": "x"}), "r")
        for key in meta:
            self.assertNotIn(key.split("/")[-1], ("compression", "PixelAspectRatio", "screenWindowWidth"))
            self.assertNotIn(":", key)

    def test_metadata_passes_through_ordinary_nodes_unchanged(self):
        g = self.read_graph({"exr/lens": "omid"})
        g.add("grade", "Grade", {"multiply": 2.0}, image="r")
        g.add("blur", "Blur", {"radius": 3.0}, image="grade")
        g.add("shuffle", "Shuffle", {}, image="blur")
        g.add("merge", "Merge", {}, A="shuffle", B="r")
        e = Evaluator()
        expected = self.meta(g, "r", evaluator=e)
        for key in ("grade", "blur", "shuffle", "merge"):
            self.assertEqual(self.meta(g, key, evaluator=e), expected, key)

    def test_merge_takes_the_background_input_metadata(self):
        g = self.read_graph({"exr/lens": "background"})
        other = self.exr("other.exr", {"exr/lens": "foreground"})
        g.add("fg", "Read", {"path": other})
        g.add("merge", "Merge", {}, A="fg", B="r")
        self.assertEqual(self.meta(g, "merge")["exr/lens"], "background")

    def test_metadata_survives_a_proxy_tier(self):
        g = self.read_graph({"exr/lens": "omid"})
        raster = Evaluator().evaluate_raster(dict(g.doc, view="r"), tier=2)
        self.assertEqual(raster.meta["exr/lens"], "omid")

    def test_disk_tier_keeps_metadata(self):
        cache = DiskCache(self.dir / "spill", 1 << 24)
        raster = Evaluator().evaluate_raster(dict(self.read_graph({"exr/lens": "omid"}).doc, view="r"))
        self.assertTrue(cache.put_raster("d" * 64, raster))
        self.assertEqual(cache.get_raster("d" * 64).meta, raster.meta)


class WriteTests(FileCase):
    def render(self, g, key, name, frame=1):
        """What the Write node's render does per frame (app.render_write): evaluate, write the EXR."""
        raster = Evaluator().evaluate_raster(dict(g.doc, view=key), frame=frame)
        target = self.dir / name
        write_exr(target, raster.to_display(), bits="float", layers=raster_layer_arrays(raster),
                  metadata=raster.meta)
        return str(target.resolve())

    def test_write_then_read_round_trips_metadata(self):
        g = self.read_graph({"exr/lens": "omid", "show/shot": "sh010"})
        g.add("mod", "ModifyMetaData", {"edits": "set show/take 4\nset timecode 01:02:03:04"}, image="r")
        g.add("w", "Write", {}, image="mod")
        out = self.render(g, "w", "out.exr")
        again = Graph()
        again.add("r", "Read", {"path": out})
        meta = self.meta(again, "r")
        self.assertEqual({k: v for k, v in meta.items() if not k.startswith("input/")},
                         {"exr/lens": "omid", "show/shot": "sh010", "show/take": "4", "timecode": "01:02:03:04"})
        self.assertEqual(meta["input/filename"], out)   # regenerated for the new file, not carried

    def test_write_keeps_metadata_a_node_removed_out_of_the_file(self):
        g = self.read_graph({"exr/lens": "omid", "show/shot": "sh010"})
        g.add("mod", "ModifyMetaData", {"edits": "remove show/shot"}, image="r")
        out = self.render(g, "mod", "out.exr")
        again = Graph()
        again.add("r", "Read", {"path": out})
        meta = self.meta(again, "r")
        self.assertEqual(meta["exr/lens"], "omid")
        self.assertNotIn("show/shot", meta)


class ModifyTests(FileCase):
    def modified(self, edits, frame=1, **header):
        g = self.read_graph({"exr/lens": "omid", "show/shot": "sh010", **header})
        g.add("m", "ModifyMetaData", {"edits": edits}, image="r")
        return g, self.meta(g, "m", frame)

    def test_set_adds_and_replaces(self):
        _, meta = self.modified("set show/take 4\nset exr/lens someone else")
        self.assertEqual(meta["show/take"], "4")
        self.assertEqual(meta["exr/lens"], "someone else")
        self.assertEqual(meta["show/shot"], "sh010")

    def test_remove_and_rename(self):
        _, meta = self.modified("remove exr/lens\nrename show/shot show/plate\nremove not/there")
        self.assertNotIn("exr/lens", meta)
        self.assertNotIn("show/shot", meta)
        self.assertEqual(meta["show/plate"], "sh010")

    def test_values_take_the_frame_and_other_keys(self):
        g, meta = self.modified("set show/label [metadata show/shot]_f[frame]", frame=17)
        self.assertEqual(meta["show/label"], "sh010_f17")
        self.assertEqual(self.meta(g, "m", 18)["show/label"], "sh010_f18")

    def test_frame_expression_is_part_of_the_cache_key(self):
        g, _ = self.modified("set show/label frame[frame]")
        e = Evaluator()
        self.assertEqual(self.meta(g, "m", 5, e)["show/label"], "frame5")
        self.assertEqual(self.meta(g, "m", 6, e)["show/label"], "frame6")
        self.assertEqual(self.meta(g, "m", 5, e)["show/label"], "frame5")

    def test_edits_are_applied_in_order_and_do_not_touch_the_upstream_dict(self):
        g, meta = self.modified("rename exr/lens a/b\nset a/b [metadata a/b]!")
        self.assertEqual(meta["a/b"], "omid!")
        self.assertEqual(self.meta(g, "r")["exr/lens"], "omid")

    def test_pixels_are_untouched(self):
        g, _ = self.modified("remove exr/lens")
        e = Evaluator()
        np.testing.assert_array_equal(e.evaluate(dict(g.doc, view="m")), e.evaluate(dict(g.doc, view="r")))

    def test_removing_every_key_leaves_no_metadata(self):
        g = Graph()
        g.add("c", "Constant", {"width": 8, "height": 8})
        g.add("m", "ModifyMetaData", {"edits": "set a/b 1"}, image="c")
        g.add("n", "ModifyMetaData", {"edits": "remove a/b"}, image="m")
        self.assertEqual(self.meta(g, "n"), {})

    def test_a_malformed_edit_is_reported(self):
        with self.assertRaises(ValueError):
            md.parse_edits("set")
        with self.assertRaises(ValueError):
            md.parse_edits("frobnicate a b")
        with self.assertRaises(ValueError):
            md.parse_edits("rename only_one")
        self.assertEqual(md.parse_edits("# note\n\nset a b c d"), [("set", "a", "b c d")])


class CopyCompareTests(FileCase):
    def two(self):
        g = self.read_graph({"exr/lens": "omid", "show/shot": "sh010"})
        g.add("other", "Read", {"path": self.exr("other.exr", {"exr/lens": "them", "show/take": "9"})})
        return g

    def test_copy_all_keys_lays_them_over_the_image(self):
        g = self.two()
        g.add("c", "CopyMetaData", {}, image="r", meta="other")
        meta = self.meta(g, "c")
        self.assertEqual((meta["exr/lens"], meta["show/shot"], meta["show/take"]), ("them", "sh010", "9"))
        self.assertEqual(meta["input/filename"], self.meta(g, "other")["input/filename"])

    def test_copy_listed_keys_only(self):
        g = self.two()
        g.add("c", "CopyMetaData", {"keys": "show/take, exr/lens"}, image="r", meta="other")
        meta = self.meta(g, "c")
        self.assertEqual((meta["exr/lens"], meta["show/take"]), ("them", "9"))
        self.assertEqual(meta["input/filename"], self.path)

    def test_copy_with_no_meta_input_passes_the_image(self):
        g = self.two()
        g.add("c", "CopyMetaData", {}, image="r")
        self.assertEqual(self.meta(g, "c"), self.meta(g, "r"))

    def test_copy_keeps_the_images_pixels(self):
        g = self.two()
        g.add("c", "CopyMetaData", {}, image="r", meta="other")
        e = Evaluator()
        np.testing.assert_array_equal(e.evaluate(dict(g.doc, view="c")), e.evaluate(dict(g.doc, view="r")))

    def test_compare_lists_only_the_keys_that_differ(self):
        a = {"a": "1", "same": "x", "only_a": "2"}
        b = {"a": "9", "same": "x", "only_b": "3"}
        self.assertEqual(md.compare(a, b), [("a", "1", "9"), ("only_a", "2", None), ("only_b", None, "3")])
        self.assertEqual(md.compare(a, a), [])
        self.assertEqual(md.compare(None, {}), [])

    def test_compare_node_passes_the_image_and_its_metadata(self):
        g = self.two()
        g.add("cmp", "CompareMetaData", {}, image="r", other="other")
        self.assertEqual(self.meta(g, "cmp"), self.meta(g, "r"))
        differing = md.compare(self.meta(g, "r"), self.meta(g, "other"))
        self.assertEqual({key for key, _, _ in differing},
                         {"exr/lens", "show/shot", "show/take", "input/filename"})

    def test_view_metadata_is_a_pass_through(self):
        g = self.two()
        g.add("v", "ViewMetaData", {}, image="r")
        e = Evaluator()
        self.assertIs(e.evaluate_raster(dict(g.doc, view="v")), e.evaluate_raster(dict(g.doc, view="r")))


class TimeCodeTests(FileCase):
    def stamped(self, frame, **params):
        g = self.read_graph()
        g.add("t", "AddTimeCode", params, image="r")
        return self.meta(g, "t", frame).get("timecode")

    def test_24_fps_frame_24_from_zero_is_one_second(self):
        self.assertEqual(self.stamped(24, timecode="00:00:00:00", fps=24.0, start_frame=0), "00:00:01:00")

    def test_the_start_frame_shows_the_start_timecode(self):
        self.assertEqual(self.stamped(1, timecode="01:00:00:00"), "01:00:00:00")
        self.assertEqual(self.stamped(25, timecode="01:00:00:00"), "01:00:01:00")
        self.assertEqual(self.stamped(1001, timecode="10:00:00:00", start_frame=1001), "10:00:00:00")

    def test_other_rates_and_rollover(self):
        self.assertEqual(md.frames_to_timecode(25 * 3600 + 25 * 60 + 12, 25), "01:01:00:12")
        self.assertEqual(self.stamped(30, timecode="00:00:00:00", fps=30.0, start_frame=0), "00:00:01:00")
        self.assertEqual(md.timecode_at("00:00:00:23", 24, 0, 1), "00:00:01:00")
        self.assertEqual(md.timecode_at("23:59:59:23", 24, 0, 1), "00:00:00:00")

    def test_drop_frame_matches_the_smpte_table(self):
        # Frames 1798, 1799 are 00:00:59;28 / ;29; 1800 skips ;00 and ;01 and reads 00:01:00;02.
        self.assertEqual(md.frames_to_timecode(1799, 29.97, True), "00:00:59;29")
        self.assertEqual(md.frames_to_timecode(1800, 29.97, True), "00:01:00;02")
        self.assertEqual(md.frames_to_timecode(17982, 29.97, True), "00:10:00;00")   # every tenth minute keeps them
        self.assertEqual(md.frames_to_timecode(107892, 29.97, True), "01:00:00;00")
        for count in (0, 1, 1799, 1800, 17981, 17982, 200000):
            code = md.frames_to_timecode(count, 29.97, True)
            self.assertEqual(md.timecode_to_frames(code, 29.97, True), count, code)
        self.assertEqual(self.stamped(1800, fps=29.97, start_frame=0, drop_frame=1), "00:01:00;02")

    def test_bad_timecodes_are_refused(self):
        for text in ("1:2:3", "00:00:00:24", "abc"):
            with self.assertRaises(ValueError, msg=text):
                md.timecode_to_frames(text, 24)

    def test_timecode_reaches_the_file_and_the_frame_is_in_the_cache_key(self):
        g = self.read_graph()
        g.add("t", "AddTimeCode", {"start_frame": 0}, image="r")
        e = Evaluator()
        self.assertEqual(self.meta(g, "t", 24, e)["timecode"], "00:00:01:00")
        self.assertEqual(self.meta(g, "t", 48, e)["timecode"], "00:00:02:00")
        self.assertEqual(self.meta(g, "t", 24, e)["timecode"], "00:00:01:00")
        write_exr(self.dir / "tc.exr", plate(), bits="float", metadata=self.meta(g, "t", 48, e))
        again = Graph()
        again.add("r", "Read", {"path": str(self.dir / "tc.exr")})
        self.assertEqual(self.meta(again, "r")["timecode"], "00:00:02:00")


class BurnInTests(FileCase):
    FONT = 10.0

    def text_node(self, g, message, justify, **box):
        params = {"width": W, "height": H, "message": message, "font_size": self.FONT, "justify": justify,
                  "box_x": 12.0, "box_width": W - 24.0, "box_y": 0.0, "box_height": 20.0}
        params.update(box)
        return g.add("text", "Text", params, image="r")

    def burn(self, g, **params):
        base = {"top_left": "", "top_right": "", "bottom_left": "", "bottom_right": "", "center": "",
                "font_size": self.FONT, "bar": 0}
        base.update(params)
        return g.add("burn", "BurnIn", base, image="r")

    def test_filename_and_frame_in_the_top_right_match_the_text_node(self):
        g = self.read_graph()
        self.burn(g, top_right="[metadata input/filename] [frame]")
        message = f"{self.path} 7"
        self.text_node(g, message, "right")
        e = Evaluator()
        burned = e.evaluate(dict(g.doc, view="burn"), frame=7)
        drawn = e.evaluate(dict(g.doc, view="text"), frame=7)
        np.testing.assert_array_equal(burned, drawn)
        source = e.evaluate(dict(g.doc, view="r"), frame=7)
        changed = np.any(burned != source, axis=2)
        self.assertTrue(changed.any())
        ys, xs = np.nonzero(changed)
        self.assertLess(ys.max(), 20)                 # inside the top band
        self.assertGreater(xs.min(), W // 2)          # right-justified: the left half is untouched
        self.assertGreater(xs.max(), W - 40)          # ends at the right margin

    def test_each_slot_lands_in_its_own_corner(self):
        g = self.read_graph()
        self.burn(g, top_left="TL", top_right="TR", bottom_left="BL", bottom_right="BR", center="CC")
        e = Evaluator()
        burned = e.evaluate(dict(g.doc, view="burn"))
        changed = np.any(burned != e.evaluate(dict(g.doc, view="r")), axis=2)

        def inside(y0, y1, x0, x1):
            return changed[y0:y1, x0:x1].any()
        self.assertTrue(inside(0, 20, 0, 60) and inside(0, 20, W - 60, W))
        self.assertTrue(inside(H - 20, H, 0, 60) and inside(H - 20, H, W - 60, W))
        self.assertTrue(inside(14, H - 14, W // 2 - 20, W // 2 + 20))
        self.assertFalse(inside(0, 20, 100, W - 100))
        # And each equals a Text node drawing the same string in the same box.
        for slot, message, justify, box in (("top_left", "TL", "left", {}), ("top_right", "TR", "right", {}),
                                            ("bottom_left", "BL", "left", {"box_y": H - 20.0}),
                                            ("bottom_right", "BR", "right", {"box_y": H - 20.0})):
            with self.subTest(slot=slot):
                h = self.read_graph()
                self.burn(h, **{slot: message})
                self.text_node(h, message, justify, **box)
                np.testing.assert_array_equal(e.evaluate(dict(h.doc, view="burn")),
                                              e.evaluate(dict(h.doc, view="text")))

    def test_background_bar_darkens_the_band_only_where_there_is_text(self):
        g = self.read_graph()
        self.burn(g, top_left="hello", bar=1, bar_opacity=0.5)
        e = Evaluator()
        source = e.evaluate(dict(g.doc, view="r"))
        burned = e.evaluate(dict(g.doc, view="burn"))
        band = 20
        # Far right of the top band carries no glyphs: exactly half the plate.
        np.testing.assert_allclose(burned[2:band - 2, W - 40:, :3], source[2:band - 2, W - 40:, :3] * 0.5, atol=1e-6)
        np.testing.assert_array_equal(burned[band:], source[band:])   # no bar without bottom text
        g2 = self.read_graph()
        self.burn(g2, bottom_right="x", bar=1)
        again = Evaluator().evaluate(dict(g2.doc, view="burn"))
        np.testing.assert_array_equal(again[:H - 20], source[:H - 20])
        self.assertTrue(np.all(again[H - 18:H - 2, :100, :3] < source[H - 18:H - 2, :100, :3] + 1e-6))

    def test_colour_and_size_change_the_pixels(self):
        e = Evaluator()
        results = []
        for params in ({}, {"red": 0.0, "green": 0.0, "blue": 1.0}, {"font_size": 20.0}):
            g = self.read_graph()
            self.burn(g, top_left="Hello", **params)
            results.append(e.evaluate(dict(g.doc, view="burn")))
        self.assertFalse(np.array_equal(results[0], results[1]))
        self.assertFalse(np.array_equal(results[0], results[2]))

    def test_frame_substitution_is_part_of_the_cache_key(self):
        g = self.read_graph()
        self.burn(g, top_left="frame [frame]")
        e = Evaluator()
        first = e.evaluate(dict(g.doc, view="burn"), frame=1)
        second = e.evaluate(dict(g.doc, view="burn"), frame=2)
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, e.evaluate(dict(g.doc, view="burn"), frame=1))

    def test_a_timecode_key_can_be_burned_in(self):
        g = self.read_graph()
        g.add("tc", "AddTimeCode", {"start_frame": 0}, image="r")
        g.add("burn", "BurnIn", {"top_right": "", "bottom_right": "[metadata timecode]", "bar": 0,
                                 "top_left": "", "font_size": self.FONT}, image="tc")
        g.add("text", "Text", {"width": W, "height": H, "message": "00:00:01:00", "font_size": self.FONT,
                               "justify": "right", "box_x": 12.0, "box_width": W - 24.0, "box_y": H - 20.0,
                               "box_height": 20.0}, image="r")
        e = Evaluator()
        np.testing.assert_array_equal(e.evaluate(dict(g.doc, view="burn"), frame=24),
                                      e.evaluate(dict(g.doc, view="text")))

    def test_metadata_and_layers_pass_through(self):
        g = self.read_graph({"exr/lens": "omid"})
        self.burn(g, top_left="x")
        self.assertEqual(self.meta(g, "burn"), self.meta(g, "r"))

    def test_proxy_tier_scales_the_text(self):
        g = self.read_graph()
        self.burn(g, top_left="Hello", margin=12.0, font_size=20.0)
        e = Evaluator()
        full = e.evaluate(dict(g.doc, view="burn"), tier=1)
        half = e.evaluate(dict(g.doc, view="burn"), tier=2)
        self.assertEqual(half.shape[:2], (H // 2, W // 2))
        self.assertTrue(np.any(half != e.evaluate(dict(g.doc, view="r"), tier=2)))
        self.assertEqual(full.shape[:2], (H, W))


class RegistrationTests(FileCase):
    def test_every_new_kind_is_registered(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertIn(kind, SPECS)
                self.assertIn(kind, REGION_RULES)
                self.assertEqual(bypass_slot({"type": kind, "inputs": {s: None for s in
                                                                        SPECS[kind]["inputs"]}}), "image")
                for name, default in SPECS[kind]["params"].items():
                    if not isinstance(default, str):
                        self.assertIn(name, LIMITS)

    def test_they_are_excluded_from_the_tile_path_and_fall_back(self):
        for kind in KINDS:
            self.assertNotIn(kind, SUPPORTED_TILED_KINDS)
        g = self.read_graph({"exr/lens": "omid"})
        g.add("m", "ModifyMetaData", {"edits": "set a/b 1"}, image="r")
        self.assertFalse(TileExecutor(evaluator=Evaluator()).supports_tiled(dict(g.doc, view="m"), "m"))

    def test_bypass_passes_the_input_pixels_and_metadata(self):
        for kind, params in (("ModifyMetaData", {"edits": "remove exr/lens"}), ("AddTimeCode", {}),
                             ("CopyMetaData", {}), ("CompareMetaData", {}), ("ViewMetaData", {}),
                             ("BurnIn", {"top_left": "hi", "center": "there"})):
            with self.subTest(kind=kind):
                g = self.read_graph({"exr/lens": "omid"})
                g.add("n", kind, params, image="r")
                g.bypass("n")
                e = Evaluator()
                np.testing.assert_array_equal(e.evaluate(dict(g.doc, view="n")), e.evaluate(dict(g.doc, view="r")))
                self.assertEqual(self.meta(g, "n"), self.meta(g, "r"))

    def test_documents_without_these_nodes_evaluate_as_before(self):
        g = Graph()
        g.add("c", "Constant", {"width": 8, "height": 4})
        g.add("gr", "Grade", {"multiply": 0.5}, image="c")
        raster = Evaluator().evaluate_raster(dict(g.doc, view="gr"))
        self.assertIsNone(raster.meta)
        self.assertEqual(raster.pixels.shape, (4, 8, 4))

    def test_choices_are_untouched(self):
        self.assertIn("justify", CHOICES)


if __name__ == "__main__":
    unittest.main()
