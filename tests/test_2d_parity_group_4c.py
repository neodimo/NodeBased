"""Lane L2 step 4c: the document-wide format registry, Grid and NoOp. Registry tests assert the
windows two Reformats resolve to; Grid asserts the exact pixel rows and columns it draws; NoOp
asserts output equals input on the evaluator and the tile path."""
import copy
import unittest

import numpy as np

from nodebased.core import (Dispatcher, REFORMAT_FORMATS, SCHEMA_VERSION, bypass_slot, builtin_formats,
                            empty_document, upgrade_document, validate)
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    def fmt(self, **cmd):
        return self.d.execute(dict(op="format", **cmd))

    def params(self, key):
        return self.d.document["nodes"][key]["params"]

    @property
    def doc(self):
        return self.d.document


def evaluator_pixels(document, target):
    return Evaluator().evaluate(dict(document, view=target))


def tile_pixels(document, target, tile_edge=None):
    kwargs = {} if tile_edge is None else {"tile_edge": tile_edge}
    executor = TileExecutor(evaluator=Evaluator(), **kwargs)
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def window(params):
    return (params["width"], params["height"], params["pixel_aspect"])


class FormatRegistryTests(unittest.TestCase):
    def test_new_documents_carry_the_builtin_list(self):
        doc = empty_document()
        self.assertEqual(doc["settings"]["formats"], builtin_formats())
        self.assertEqual(set(doc["settings"]["formats"]), set(REFORMAT_FORMATS))
        validate(doc)

    def test_two_reformats_naming_the_same_registry_format_get_identical_windows(self):
        g = Graph()
        g.fmt(action="set", name="Delivery", width=1600, height=900)
        g.add("plate", "Checker", dict(width=64, height=48, size=8))
        g.add("a", "Reformat", dict(format="Delivery"), image="plate")
        g.add("b", "Reformat", dict(format="Custom", width=5, height=5), image="plate")
        g.d.execute(dict(op="set", id="b", param="format", value="Delivery"))
        self.assertEqual(window(g.params("a")), (1600, 900, 1.0))
        self.assertEqual(window(g.params("a")), window(g.params("b")))
        out_a, out_b = evaluator_pixels(g.doc, "a"), evaluator_pixels(g.doc, "b")
        self.assertEqual(out_a.shape[:2], (900, 1600))
        np.testing.assert_array_equal(out_a, out_b)

    def test_the_registry_wins_over_the_node_local_list(self):
        g = Graph()
        g.fmt(action="set", name="HD_720", width=1000, height=500, pixel_aspect=2.0)
        g.add("node", "Reformat", dict(format="HD_720"))
        self.assertEqual(window(g.params("node")), (1000, 500, 2.0))

    def test_editing_an_entry_updates_every_reformat_that_names_it(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=64, height=48, size=8))
        g.add("a", "Reformat", dict(format="HD_720"), image="plate")
        g.add("b", "Reformat", dict(format="HD_720"))
        g.add("c", "Reformat", dict(format="UHD_4K"))
        self.assertEqual(window(g.params("a")), (1280, 720, 1.0))
        g.fmt(action="set", name="HD_720", width=1440, height=810)
        self.assertEqual(window(g.params("a")), (1440, 810, 1.0))
        self.assertEqual(window(g.params("b")), (1440, 810, 1.0))
        self.assertEqual(window(g.params("c")), (3840, 2160, 1.0))
        out = evaluator_pixels(g.doc, "a")
        self.assertEqual(out.shape[:2], (810, 1440))

    def test_renaming_an_entry_carries_its_reformats_and_keeps_the_window(self):
        g = Graph()
        g.add("a", "Reformat", dict(format="2K_DCP"))
        g.add("b", "Reformat", dict(format="2K_DCP"))
        g.fmt(action="rename", name="2K_DCP", new_name="Cinema")
        self.assertEqual(g.params("a")["format"], "Cinema")
        self.assertEqual(g.params("b")["format"], "Cinema")
        self.assertNotIn("2K_DCP", g.doc["settings"]["formats"])
        g.fmt(action="set", name="Cinema", width=4096, height=2160)
        self.assertEqual(window(g.params("b")), (4096, 2160, 1.0))

    def test_deleting_an_entry_turns_its_reformats_custom_at_their_last_window(self):
        g = Graph()
        g.add("a", "Reformat", dict(format="Square_1K"))
        g.fmt(action="delete", name="Square_1K")
        self.assertEqual(g.params("a")["format"], "Custom")
        self.assertEqual(window(g.params("a")), (1024, 1024, 1.0))

    def test_bad_registry_edits_are_rejected_and_leave_the_document_alone(self):
        g = Graph()
        before = copy.deepcopy(g.doc)
        for cmd in (dict(action="set", name="", width=10, height=10),
                    dict(action="set", name="Custom", width=10, height=10),
                    dict(action="set", name="X", width=0, height=10),
                    dict(action="set", name="X", width=10.5, height=10),
                    dict(action="set", name="X", width=10, height=10, pixel_aspect=0.0),
                    dict(action="rename", name="nope", new_name="Y"),
                    dict(action="rename", name="HD_720", new_name="HD_1080"),
                    dict(action="delete", name="nope"),
                    dict(action="explode", name="HD_720")):
            with self.subTest(cmd=cmd), self.assertRaises(ValueError):
                g.fmt(**cmd)
        self.assertEqual(g.doc, before)

    def test_registry_edits_are_undoable(self):
        g = Graph()
        g.add("a", "Reformat", dict(format="HD_720"))
        g.fmt(action="set", name="HD_720", width=100, height=50)
        g.d.execute(dict(op="undo"))
        self.assertEqual(window(g.params("a")), (1280, 720, 1.0))
        self.assertEqual(g.doc["settings"]["formats"]["HD_720"]["width"], 1280)

    def test_a_reformat_may_not_name_an_unknown_format(self):
        g = Graph()
        g.add("a", "Reformat")
        with self.assertRaises(ValueError):
            g.d.execute(dict(op="set", id="a", param="format", value="Nonexistent"))

    def test_an_old_document_loads_with_the_builtin_list_and_renders_identically(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=64, height=48, size=8))
        g.add("node", "Reformat", dict(format="HD_720"), image="plate")
        current = copy.deepcopy(g.doc)
        old = copy.deepcopy(current)
        del old["settings"]["formats"]
        self.assertEqual(old["version"], SCHEMA_VERSION)
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["settings"]["formats"], builtin_formats())
        np.testing.assert_array_equal(evaluator_pixels(upgraded, "node"), evaluator_pixels(current, "node"))
        # A registry-less dict that skips the upgrade still validates and resolves to the built-ins.
        bare = copy.deepcopy(current)
        del bare["settings"]["formats"]
        validate(bare)
        np.testing.assert_array_equal(evaluator_pixels(bare, "node"), evaluator_pixels(current, "node"))

    def test_the_describe_reply_lists_the_format_operation_and_registry(self):
        reply = Dispatcher().execute(dict(op="describe"))
        self.assertIn("format", reply["operations"])
        self.assertEqual(set(reply["settings"]["formats"]), set(REFORMAT_FORMATS))


class GridTests(unittest.TestCase):
    def grid(self, **params):
        g = Graph()
        base = dict(width=32, height=20, spacing_x=8.0, spacing_y=5.0, line_width=1.0,
                    red=1.0, green=0.5, blue=0.25, alpha=1.0)
        base.update(params)
        g.add("node", "Grid", base)
        return g

    @staticmethod
    def lit(out):
        on = out[..., 3] > 0
        return sorted(set(np.flatnonzero(on.all(axis=0)))), sorted(set(np.flatnonzero(on.all(axis=1))))

    def test_lines_fall_on_the_computed_columns_and_rows(self):
        out = evaluator_pixels(self.grid().doc, "node")
        cols, rows = self.lit(out)
        self.assertEqual(cols, [0, 8, 16, 24])
        self.assertEqual(rows, [0, 5, 10, 15])
        np.testing.assert_allclose(out[0, 3], [1.0, 0.5, 0.25, 1.0])
        # Between the lines the frame is transparent.
        np.testing.assert_array_equal(out[2:5, 1:8], 0.0)

    def test_line_width_and_offset(self):
        out = evaluator_pixels(self.grid(line_width=3.0, grid_offset_x=2.0, grid_offset_y=1.0).doc, "node")
        cols, rows = self.lit(out)
        self.assertEqual(cols, [2, 3, 4, 10, 11, 12, 18, 19, 20, 26, 27, 28])
        self.assertEqual(rows, [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18])

    def test_negative_offset_wraps_to_the_earlier_period(self):
        out = evaluator_pixels(self.grid(grid_offset_x=-3.0).doc, "node")
        self.assertEqual(self.lit(out)[0], [5, 13, 21, 29])

    def test_number_of_lines_overrides_spacing(self):
        out = evaluator_pixels(self.grid(number_x=4, number_y=2, spacing_x=99.0, spacing_y=99.0).doc, "node")
        cols, rows = self.lit(out)
        self.assertEqual(cols, [0, 8, 16, 24])
        self.assertEqual(rows, [0, 10])

    def test_zero_width_draws_nothing_and_alpha_scales_coverage(self):
        self.assertFalse(evaluator_pixels(self.grid(line_width=0.0).doc, "node").any())
        out = evaluator_pixels(self.grid(alpha=0.5).doc, "node")
        self.assertAlmostEqual(float(out[0, 0, 3]), 0.5)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.5)  # premultiplied

    def test_composites_over_an_image_input_and_mask_mix_apply(self):
        g = self.grid(red=0.0, green=0.0, blue=0.0)
        g.add("plate", "Constant", dict(width=32, height=20, red=0.2, green=0.4, blue=0.6, alpha=1.0))
        g.d.execute(dict(op="connect", id="node", input="image", source="plate"))
        out = evaluator_pixels(g.doc, "node")
        np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(out[2, 1], [0.2, 0.4, 0.6, 1.0], atol=1e-6)
        g.d.execute(dict(op="set", id="node", param="mix", value=0.0))
        np.testing.assert_allclose(evaluator_pixels(g.doc, "node")[0, 0], [0.2, 0.4, 0.6, 1.0], atol=1e-6)

    def test_tile_path_matches_the_evaluator_including_across_tile_seams(self):
        g = self.grid(width=100, height=90, spacing_x=7.5, spacing_y=11.0, line_width=2.5,
                      grid_offset_x=3.0, grid_offset_y=-2.0)
        reference = evaluator_pixels(g.doc, "node")
        np.testing.assert_array_equal(tile_pixels(g.doc, "node"), reference)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node", tile_edge=32), reference)

    def test_bypassed_grid_passes_its_image_or_a_transparent_frame(self):
        g = self.grid()
        g.add("plate", "Constant", dict(width=32, height=20, red=0.2, green=0.4, blue=0.6, alpha=1.0))
        g.d.execute(dict(op="connect", id="node", input="image", source="plate"))
        g.d.execute(dict(op="disable", id="node", value=True))
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "node"), evaluator_pixels(g.doc, "plate"))


class NoOpTests(unittest.TestCase):
    def build(self):
        g = Graph()
        g.add("plate", "Checker", dict(width=64, height=48, size=8))
        g.add("node", "NoOp", dict(note="hold for review"), image="plate")
        return g

    def test_output_equals_input_on_both_paths(self):
        g = self.build()
        plate = evaluator_pixels(g.doc, "plate")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "node"), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node"), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node", tile_edge=16), plate)

    def test_bypassed_and_enabled_both_pass_the_input(self):
        g = self.build()
        plate = evaluator_pixels(g.doc, "plate")
        g.d.execute(dict(op="disable", id="node", value=True))
        self.assertEqual(bypass_slot(g.doc["nodes"]["node"]), "image")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "node"), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, "node"), plate)

    def test_it_keeps_its_note_and_sits_mid_chain(self):
        g = self.build()
        self.assertEqual(g.params("node")["note"], "hold for review")
        g.add("grade", "Grade", dict(exposure=1.0), image="node")
        direct = Graph()
        direct.add("plate", "Checker", dict(width=64, height=48, size=8))
        direct.add("grade", "Grade", dict(exposure=1.0), image="plate")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "grade"), evaluator_pixels(direct.doc, "grade"))


if __name__ == "__main__":
    unittest.main()
