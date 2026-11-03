"""Lane 2, step M: a new Read, Write, BurnIn, IBKColor or Keyer starts with Nuke's defaults, and IBKColor's `pick` input.

The defaults of a saved node are the values it was saved with; an older document that lacks a knob gets the value of before
(readknobs.DEFAULTS, writenode.DEFAULTS), so none of them renders differently.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import core, readknobs, writenode
from nodebased.tileexec import TileExecutor
from nodebased.core import SPECS, Dispatcher, upgrade_document
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_color_a import tiled
from tests.test_2d_parity_step_k1 import Graph, evaluator_pixels, screen_graph


def new_params(kind):
    d = Dispatcher()
    d.execute(dict(op="create", id="n", type=kind, params={}))
    return d.document["nodes"]["n"]["params"]


class NewNodeDefaults(unittest.TestCase):
    def test_read_starts_with_auto_alpha_off_and_write_on_rgb(self):
        self.assertEqual(new_params("Read")["auto_alpha"], 0)
        self.assertEqual(new_params("Write")["channels"], "rgb")

    def test_burnin_starts_with_no_background_and_an_opaque_one_when_turned_on(self):
        p = new_params("BurnIn")
        self.assertEqual((p["bar"], p["bar_opacity"]), (0, 1.0))

    def test_ibkcolor_starts_at_nukes_blue_screen_with_no_erode_and_no_multi(self):
        p = new_params("IBKColor")
        self.assertEqual((p["screen_type"], p["screen_erode"], p["patch_black"]), ("blue", 0.0, 0))

    def test_keyer_keeps_the_a_b_c_d_default_of_nuke(self):
        # The guide's knob table lists every range point as 1; Nuke's range knob starts at 0 0 1 1, which is what the node has.
        p = new_params("Keyer")
        self.assertEqual([p[k] for k in ("range_a", "range_b", "range_c", "range_d")], [0.0, 0.0, 1.0, 1.0])


class OlderDocumentsRenderAsBefore(unittest.TestCase):
    def doc_without(self, kind, **missing):
        d = Dispatcher()
        d.execute(dict(op="create", id="n", type=kind, params={}))
        doc = copy.deepcopy(d.document)
        doc["version"] = 28
        for key in missing:
            doc["nodes"]["n"]["params"].pop(key)
        return upgrade_document(doc)["nodes"]["n"]["params"]

    def test_a_read_saved_before_auto_alpha_keeps_its_alpha_of_one(self):
        self.assertEqual(self.doc_without("Read", auto_alpha=0)["auto_alpha"], 1)

    def test_a_write_saved_before_channels_keeps_all_channels(self):
        self.assertEqual(self.doc_without("Write", channels=0)["channels"], "all")

    def test_saved_values_are_not_replaced(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="b", type="BurnIn", params={"bar": 1, "bar_opacity": 0.5}))
        d.execute(dict(op="create", id="i", type="IBKColor", params={"screen_type": "green", "screen_erode": 1.0, "patch_black": 1}))
        d.execute(dict(op="create", id="w", type="Write", params={"channels": "all"}))
        doc = upgrade_document(copy.deepcopy(d.document))
        self.assertEqual((doc["nodes"]["b"]["params"]["bar"], doc["nodes"]["b"]["params"]["bar_opacity"]), (1, 0.5))
        self.assertEqual([doc["nodes"]["i"]["params"][k] for k in ("screen_type", "screen_erode", "patch_black")], ["green", 1.0, 1])
        self.assertEqual(doc["nodes"]["w"]["params"]["channels"], "all")

    def test_burnin_pixels_follow_the_saved_bar_and_the_new_default_has_none(self):
        def burn(**params):
            g = Graph()
            g.add("src", "Constant", dict(width=160, height=48, red=0.5, green=0.5, blue=0.5, alpha=1.0))
            g.add("burn", "BurnIn", dict({"top_left": "x", "top_right": "", "bottom_left": "", "bottom_right": ""}, **params), image="src")
            return Evaluator().evaluate_raster(dict(g.doc, view="burn"), frame=1).pixels
        saved = burn(bar=1, bar_opacity=0.5)
        fresh = burn()
        self.assertAlmostEqual(float(saved[0, -1, 0]), 0.25, places=3)     # half a black bar over the grey, where no text reaches
        self.assertEqual(float(fresh[0, -1, 0]), 0.5)                      # a new BurnIn draws no bar

    def test_a_new_burnin_with_its_background_on_is_opaque(self):
        g = Graph()
        g.add("src", "Constant", dict(width=160, height=48, red=0.5, green=0.5, blue=0.5, alpha=1.0))
        g.add("burn", "BurnIn", dict(top_left="x", top_right="", bottom_left="", bottom_right="", bar=1), image="src")
        out = Evaluator().evaluate_raster(dict(g.doc, view="burn"), frame=1).pixels
        self.assertAlmostEqual(float(out[0, -1, 0]), 0.0, places=3)


class IBKColorPick(unittest.TestCase):
    OLD = dict(screen_type="green", screen_erode=1.0, patch_black=1, fill_size=12)

    def graph(self, pick=True, mask=False, **params):
        g = Graph()
        plate = screen_graph(g, 97, 71)
        g.add("pickc", "Constant", dict(width=97, height=71, red=0.1, green=0.95, blue=0.3, alpha=1.0))
        wires = dict(c=plate)
        if pick:
            wires["pick"] = "pickc"
        if mask:
            g.add("matte", "Constant", dict(width=97, height=71, red=1, green=1, blue=1, alpha=0.5))
            wires["mask"] = "matte"
        g.add("node", "IBKColor", dict(self.OLD, **params), **wires)
        return g

    def test_the_node_has_a_pick_slot_after_the_mask(self):
        self.assertEqual(SPECS["IBKColor"]["inputs"], ["c"])
        self.assertEqual(SPECS["IBKColor"]["optional_inputs"], ["mask", "pick"])

    def test_the_pick_replaces_the_screen_colour_in_the_plate(self):
        without = evaluator_pixels(self.graph(pick=False).doc, "node")
        with_pick = evaluator_pixels(self.graph().doc, "node")
        # The screen is green 0.8 in the picture and 0.95 in the pick: the plate's screen is the pick's, brighter.
        self.assertAlmostEqual(float(without[2, 2, 1]), 0.8, places=3)
        self.assertAlmostEqual(float(with_pick[2, 2, 1]), 0.95, places=3)
        self.assertGreater(float(with_pick[2, 2, 1]), float(without[2, 2, 1]))
        self.assertAlmostEqual(float(with_pick[2, 2, 2]), 0.3, places=3)

    def test_the_pick_does_not_make_screen_where_the_picture_has_none(self):
        # Inside the red box the picture holds no screen; the plate there is filled from the screen around it, so the pick's
        # colour reaches it only through that fill, never as the box's own colour.
        with_pick = evaluator_pixels(self.graph(fill_size=0).doc, "node")
        without = evaluator_pixels(self.graph(pick=False, fill_size=0).doc, "node")
        np.testing.assert_array_equal(with_pick[34, 50], without[34, 50])

    def test_no_pick_renders_as_it_did(self):
        g = self.graph(pick=False)
        image = evaluator_pixels(g.doc, "plate")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "node"), Evaluator._ibk_color(image, dict(self.OLD)))

    def test_tiles_of_7_16_and_32_equal_the_frame_with_and_without_a_mask(self):
        for mask in (False, True):
            g = self.graph(mask=mask, mix=0.9)
            whole = evaluator_pixels(g.doc, "node")
            for edge in (7, 16, 32):
                with self.subTest(mask=mask, edge=edge):
                    np.testing.assert_allclose(tiled(g.doc, edge), whole, atol=1e-6)

    def test_bypass_still_passes_the_c_input(self):
        self.assertEqual(core.bypass_slot(dict(type="IBKColor", inputs={"c": "x", "mask": None, "pick": "y"})), "c")


if __name__ == "__main__":
    unittest.main()


def rgb_exr(path, colour=(0.2, 0.4, 0.6), size=(24, 16)):
    """An EXR with three channels and no alpha."""
    import OpenImageIO as oiio
    width, height = size
    spec = oiio.ImageSpec(width, height, 3, oiio.FLOAT)
    spec.channelnames = ("R", "G", "B")
    out = oiio.ImageOutput.create(str(path))
    out.open(str(path), spec)
    out.write_image(np.broadcast_to(np.array(colour, np.float32), (height, width, 3)).copy())
    out.close()


class ReadAutoAlpha(unittest.TestCase):
    """Nuke's Read starts with auto alpha off: a file with no alpha channel reads as alpha 0. A saved Read keeps what it was saved with."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = str(Path(cls.directory.name) / "rgb.exr")
        rgb_exr(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def document(self, **knobs):
        d = Dispatcher()
        d.execute(dict(op="create", id="node", type="Read", params={"path": self.path, **knobs}))
        return dict(d.document, view="node")

    def evaluated(self, document):
        return np.asarray(Evaluator().evaluate(document, "node", frame=1))

    def tile(self, document, edge):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=edge)
        region = executor.canvas_region(document, "node", frame=1, tier=1)
        return executor.compose_region(document, "node", region, frame=1, tier=1).pixels

    def test_a_new_read_of_a_file_without_alpha_has_alpha_zero_and_the_colour_of_the_file(self):
        pixels = self.evaluated(self.document())
        self.assertEqual(float(pixels[..., 3].max()), 0.0)
        self.assertGreater(float(pixels[..., 0].max()), 0.0)

    def test_a_saved_read_with_auto_alpha_on_or_a_read_saved_before_the_knob_has_alpha_one(self):
        on = self.document(auto_alpha=1)
        self.assertEqual(float(self.evaluated(on)[..., 3].min()), 1.0)
        old = self.document()
        old["nodes"]["node"]["params"].pop("auto_alpha")
        self.assertEqual(float(self.evaluated(old)[..., 3].min()), 1.0)
        np.testing.assert_array_equal(self.evaluated(old), self.evaluated(on))

    def test_the_knob_alone_does_not_force_a_whole_decode(self):
        self.assertFalse(readknobs.active({"auto_alpha": 0}))
        self.assertTrue(readknobs.active({"auto_alpha": 0, "raw": 1}))

    def test_tiles_equal_the_frame_for_both_settings(self):
        for knobs in ({}, {"auto_alpha": 1}):
            document = self.document(**knobs)
            frame = self.evaluated(document)
            for edge in (7, 16, 32):
                np.testing.assert_array_equal(self.tile(document, edge), frame)


class WriteChannelsDefault(unittest.TestCase):
    """Nuke's Write starts on rgb: the file holds no alpha. A saved Write keeps its channels; an older one without the knob writes all."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def channels_of(self, name, params, file_type="exr"):
        import OpenImageIO as oiio
        from tests.test_2d_fidelity_c_write import make_raster
        target = str(Path(self.directory.name) / name)
        writenode.write_raster(make_raster(), target, file_type, "half", params)
        return oiio.ImageBuf(target).nchannels

    def test_a_new_write_writes_three_channels_and_all_writes_four(self):
        new = new_params("Write")
        for file_type in ("exr", "png"):
            self.assertEqual(self.channels_of(f"new.{file_type}", new, file_type), 3, file_type)
            self.assertEqual(self.channels_of(f"all.{file_type}", {**new, "channels": "all"}, file_type), 4, file_type)

    def test_a_write_saved_without_channels_writes_all_of_them_as_before(self):
        d = Dispatcher()
        d.execute(dict(op="create", id="n", type="Write", params={}))
        doc = copy.deepcopy(d.document)
        doc["version"] = 28
        doc["nodes"]["n"]["params"].pop("channels")
        old = upgrade_document(doc)["nodes"]["n"]["params"]
        self.assertEqual(self.channels_of("old.exr", old), 4)
        self.assertEqual(self.channels_of("old.png", old, "png"), 4)
