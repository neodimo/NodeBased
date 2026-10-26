"""Fidelity step C, part 1: Roto takes Nuke's mask input and the shared mask and channels controls. Pixel
assertions per option, the tile path (which renders a masked Roto whole) against the evaluator, bypass, and
old documents."""
import copy
import unittest

import numpy as np

from nodebased import ops2d_chrome as chrome
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_roto_ui import triangle

N = 32


def document(mask=0.25, bg=True, wire_mask=True, **params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "plate", "type": "Constant",
               "params": {"width": N, "height": N, "red": 0.2, "green": 0.4, "blue": 0.6, "alpha": 0.8}})
    d.execute({"op": "create", "id": "m", "type": "Constant",
               "params": {"width": N, "height": N, "red": 0.5, "green": 0.1, "blue": 0.9, "alpha": mask}})
    d.execute({"op": "create", "id": "r", "type": "Roto", "params": params})
    if bg:
        d.execute({"op": "connect", "id": "r", "input": "bg", "source": "plate"})
    if wire_mask:
        d.execute({"op": "connect", "id": "r", "input": "mask", "source": "m"})
    shape = triangle()
    shape["points"] = [dict(point, x=point["x"] * 0.25, y=point["y"] * 0.25) for point in shape["points"]]
    d.execute({"op": "set_shapes", "id": "r", "shapes": [shape]})
    d.execute({"op": "view", "id": "r"})
    return d.document


def render(doc, target="r"):
    return Evaluator().evaluate_raster(doc, target=target)


class MaskInputTests(unittest.TestCase):
    def test_roto_has_the_mask_socket_and_the_shared_controls(self):
        self.assertEqual(SPECS["Roto"]["optional_inputs"], ["bg", "mask"])
        params = SPECS["Roto"]["params"]
        for name, default in (("channels", "all"), ("maskChannelInput", "none"), ("invert_mask", 0), ("inject", 0)):
            self.assertEqual(params[name], default)
        self.assertEqual(params["output"], "alpha")                      # the draw strip is untouched
        validate(document())                                              # a wired mask is a valid document

    def test_the_mask_alpha_gates_the_result_against_the_bg(self):
        ungated = render(document(wire_mask=False)).pixels
        plate = render(document(), "plate").pixels
        gated = render(document(mask=0.25)).pixels
        np.testing.assert_allclose(gated, ungated * 0.25 + plate * 0.75, atol=1e-6)

    def test_invert_flips_the_mask(self):
        ungated = render(document(wire_mask=False)).pixels
        plate = render(document(), "plate").pixels
        gated = render(document(mask=0.25, invert_mask=1)).pixels
        np.testing.assert_allclose(gated, ungated * 0.75 + plate * 0.25, atol=1e-6)

    def test_a_mask_channel_other_than_alpha(self):
        ungated = render(document(wire_mask=False)).pixels
        plate = render(document(), "plate").pixels
        held = render(document(), "m").pixels[0, 0]            # the mask Constant's own (premultiplied) pixel
        for token, value in (("rgba.red", float(held[0])), ("rgba.green", float(held[1])), ("rgba.blue", float(held[2]))):
            with self.subTest(token=token):
                gated = render(document(maskChannelInput=token)).pixels
                np.testing.assert_allclose(gated, ungated * value + plate * (1 - value), atol=1e-6)

    def test_the_default_roto_pixels_do_not_change_with_no_mask_wired(self):
        base = render(document(wire_mask=False)).pixels
        again = render(document(wire_mask=False, channels="all", maskChannelInput="none", invert_mask=0, inject=0)).pixels
        np.testing.assert_array_equal(base, again)

    def test_no_bg_gates_against_transparent_black(self):
        ungated = render(document(bg=False, wire_mask=False, width=N, height=N)).pixels
        gated = render(document(bg=False, mask=0.5, width=N, height=N)).pixels
        np.testing.assert_allclose(gated, ungated * 0.5, atol=1e-6)

    def test_channels_limits_what_the_roto_may_write(self):
        free = render(document(wire_mask=False)).pixels
        plate = render(document(), "plate").pixels
        rgb_only = render(document(wire_mask=False, channels="rgb")).pixels
        np.testing.assert_array_equal(rgb_only[..., :3], free[..., :3])
        np.testing.assert_array_equal(rgb_only[..., 3], plate[..., 3])        # alpha stays the bg's
        alpha_only = render(document(wire_mask=False, channels="alpha")).pixels
        np.testing.assert_array_equal(alpha_only[..., 3], free[..., 3])
        np.testing.assert_array_equal(alpha_only[..., :3], plate[..., :3])

    def test_inject_writes_the_mask_layer(self):
        raster = render(document(mask=0.25, inject=1))
        np.testing.assert_allclose(raster.layers["mask"].pixels[..., 3], 0.25, atol=1e-7)
        self.assertIsNone(render(document(mask=0.25)).layers and render(document(mask=0.25)).layers.get("mask"))

    def test_the_draw_strip_still_applies_inside_the_mask(self):
        ungated = render(document(wire_mask=False, opacity=0.5)).pixels
        plate = render(document(), "plate").pixels
        gated = render(document(mask=0.5, opacity=0.5)).pixels
        np.testing.assert_allclose(gated, ungated * 0.5 + plate * 0.5, atol=1e-6)


class TileAndBypassTests(unittest.TestCase):
    def test_a_masked_roto_is_rendered_whole_and_an_unmasked_one_stays_tiled(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        self.assertFalse(executor.supports_tiled(document(), "r"))
        self.assertTrue(executor.supports_tiled(document(wire_mask=False), "r"))
        self.assertFalse(executor.supports_tiled(document(wire_mask=False, channels="rgb"), "r"))
        self.assertFalse(executor.supports_tiled(document(wire_mask=False, maskChannelInput="rgba.red"), "r"))

    def test_tiles_equal_the_evaluator_for_every_option(self):
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=16)
        for options in (dict(), dict(invert_mask=1), dict(channels="rgb"), dict(inject=1), dict(maskChannelInput="rgba.blue", opacity=0.5)):
            for wired in (True, False):
                with self.subTest(options=options, mask=wired):
                    doc = document(wire_mask=wired, **options)
                    expected = render(doc).pixels
                    region = executor.canvas_region(doc, "r", frame=1, tier=1)
                    got = executor.render_region(doc, "r", region, frame=1, tier=1) if hasattr(executor, "render_region") else \
                        executor.compose_region(doc, "r", region, frame=1, tier=1).pixels
                    np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_bypass_passes_the_bg(self):
        doc = document(mask=0.25, invert_mask=1)
        doc["nodes"]["r"]["disabled"] = True
        plate = render(doc, "plate").pixels
        np.testing.assert_array_equal(render(doc).pixels, plate)


class OldDocumentTests(unittest.TestCase):
    def test_a_roto_saved_without_the_mask_slot_loads_and_draws_the_same(self):
        doc = document(wire_mask=False)
        old = copy.deepcopy(doc)
        old["nodes"]["r"]["inputs"].pop("mask", None)
        for name in ("channels", "maskChannelInput", "invert_mask", "inject"):
            old["nodes"]["r"]["params"].pop(name, None)
        loaded = upgrade_document(old)
        self.assertIn("mask", loaded["nodes"]["r"]["inputs"])
        self.assertIsNone(loaded["nodes"]["r"]["inputs"]["mask"])
        self.assertEqual(loaded["nodes"]["r"]["params"]["channels"], "all")
        np.testing.assert_array_equal(render(loaded).pixels, render(doc).pixels)


if __name__ == "__main__":
    unittest.main()
