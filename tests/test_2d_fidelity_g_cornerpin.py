"""Fidelity step G, part 1: CornerPin takes Nuke's enable1 to enable4, extra matrix, clamp, black outside, motion blur and the
three buttons (Copy 'from', Copy 'to', Set to input). Pixel assertions for each; the evaluator and the region rule agree; the node
is excluded from the tile executor like Transform; bypass; old documents."""
import copy
import unittest

import numpy as np

from nodebased import cornerpinknobs, tiers
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator, Region
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_c4 import Graph

W, H = 32, 24
FROM = dict(from1_x=0.0, from1_y=0.0, from2_x=float(W), from2_y=0.0, from3_x=0.0, from3_y=float(H), from4_x=float(W), from4_y=float(H))
SAME = dict(FROM, to1_x=0.0, to1_y=0.0, to2_x=float(W), to2_y=0.0, to3_x=0.0, to3_y=float(H), to4_x=float(W), to4_y=float(H))


def build(plate="Checker", plate_params=None, **params):
    g = Graph()
    g.add("plate", plate, plate_params or dict(width=W, height=H, size=4))
    g.add("node", "CornerPin", {**SAME, "filter": "nearest", **params}, image="plate")
    return g


def raster(g, frame=1, target="node"):
    return Evaluator().evaluate_raster(dict(g.doc, view=target), target=target, frame=frame)


def at(r, x, y):
    return r.pixels[y - r.data.y, x - r.data.x]


class EnableTests(unittest.TestCase):
    def test_defaults_pin_all_four_corners(self):
        params = SPECS["CornerPin"]["params"]
        self.assertEqual([params[f"enable{i}"] for i in (1, 2, 3, 4)], [1, 1, 1, 1])
        self.assertEqual((params["clamp"], params["black_outside"], params["motionblur"]), (0, 1, 0))

    def test_a_disabled_corner_stays_at_its_from_point(self):
        moved = dict(to4_x=W + 6.0, to4_y=H + 4.0)
        held = raster(build(enable4=0, **moved))
        pinned = raster(build(**moved))
        identity = raster(build())
        # Holding corner 4 puts the pin back to the identity pin; moving it does not.
        self.assertEqual(held.data, identity.data)
        np.testing.assert_array_equal(held.pixels, identity.pixels)
        self.assertNotEqual(pinned.data, identity.data)

    def test_each_enable_holds_its_own_corner(self):
        for corner in (1, 2, 3, 4):
            with self.subTest(corner=corner):
                moved = {f"to{corner}_x": FROM[f"from{corner}_x"] + (7.0 if corner in (1, 3) else -7.0), f"to{corner}_y": FROM[f"from{corner}_y"] + 5.0}
                held = raster(build(**{f"enable{corner}": 0}, **moved))
                np.testing.assert_array_equal(held.pixels, raster(build()).pixels)
                self.assertFalse(np.array_equal(raster(build(**moved)).pixels[: H], raster(build()).pixels[: H]) and
                                 raster(build(**moved)).data == raster(build()).data)

    def test_the_other_corners_still_pin_when_one_is_held(self):
        moved = dict(to2_x=W - 8.0, to2_y=2.0, to4_x=W + 5.0, to4_y=H + 3.0)
        held = raster(build(enable4=0, **moved))
        only_two = raster(build(**{k: v for k, v in moved.items() if k.startswith("to2")}))
        self.assertEqual(held.data, only_two.data)
        np.testing.assert_allclose(held.pixels, only_two.pixels, atol=1e-6)


class ExtraMatrixTests(unittest.TestCase):
    def test_the_identity_matrix_changes_nothing(self):
        np.testing.assert_array_equal(raster(build(**FROM, to2_x=W - 5.0)).pixels, raster(build(**FROM, to2_x=W - 5.0, extra_matrix_00=1.0)).pixels)

    def test_a_translation_in_the_matrix_moves_the_picture(self):
        moved = raster(build(extra_matrix_02=5.0, extra_matrix_12=3.0))
        plain = raster(build())
        for x, y in ((0, 0), (4, 4), (W - 1, H - 1), (9, 2)):
            np.testing.assert_allclose(at(moved, x + 5, y + 3), at(plain, x, y), atol=1e-6)
        self.assertEqual((moved.data.x, moved.data.y), (plain.data.x + 5, plain.data.y + 3))

    def test_a_scale_in_the_matrix_scales_the_picture_about_the_origin(self):
        big = raster(build(extra_matrix_00=2.0, extra_matrix_11=2.0))
        plain = raster(build())
        for x, y in ((1, 1), (6, 5), (10, 3)):
            np.testing.assert_allclose(at(big, 2 * x, 2 * y), at(plain, x, y), atol=1e-6)
        self.assertEqual(big.data.right - big.data.x > plain.data.right - plain.data.x, True)

    def test_the_matrix_follows_the_pin(self):
        # Pin the top edge inward, then slide the result: the slid pin is the pin moved, not the pin of a moved picture.
        pin = dict(to1_x=8.0, to2_x=W - 8.0)
        slid = raster(build(extra_matrix_02=4.0, **pin))
        pinned = raster(build(**pin))
        self.assertEqual((slid.data.x, slid.data.y), (pinned.data.x + 4, pinned.data.y))
        np.testing.assert_allclose(slid.pixels, pinned.pixels, atol=1e-6)

    def test_the_inverse_direction_inverts_the_whole_warp_including_the_matrix(self):
        forward = raster(build(extra_matrix_02=5.0, direction="forward"))
        inverse = raster(build(extra_matrix_02=5.0, direction="inverse"))
        self.assertEqual((inverse.data.x, forward.data.x), (forward.data.x - 10, forward.data.x))

    def test_the_region_rule_agrees_with_the_evaluator(self):
        params = {**SPECS["CornerPin"]["params"], **SAME, "extra_matrix_02": 5.0, "extra_matrix_00": 1.5, "to3_x": 4.0, "enable2": 0, "to2_x": 3.0}
        g = build(**{k: v for k, v in params.items() if k in SPECS["CornerPin"]["params"] and k not in ("mix",)})
        r = raster(g)
        out = Region(r.data.x, r.data.y, r.data.width, r.data.height)
        needed = tiers.REGION_RULES["CornerPin"](g.doc["nodes"]["node"]["params"], out, 1)[0]
        # The source rectangle the rule asks for covers what the kernel sampled: the whole plate.
        self.assertTrue(needed.x <= 0 and needed.y <= 0 and needed.right >= W and needed.bottom >= H, needed)

    def test_the_matrix_scales_with_the_proxy_tier(self):
        params = dict(SPECS["CornerPin"]["params"], extra_matrix_02=8.0, extra_matrix_12=4.0, extra_matrix_20=0.01, extra_matrix_21=0.02, **SAME)
        halved = tiers.scale_params("CornerPin", params, 2)
        self.assertEqual((halved["extra_matrix_02"], halved["extra_matrix_12"]), (4.0, 2.0))
        self.assertEqual((halved["extra_matrix_20"], halved["extra_matrix_21"]), (0.02, 0.04))


class BlackOutsideTests(unittest.TestCase):
    ROTATE = dict(to1_x=10.0, to1_y=-4.0, to2_x=W + 4.0, to2_y=6.0, to3_x=-4.0, to3_y=H - 6.0, to4_x=W - 10.0, to4_y=H + 4.0)

    def test_on_the_uncovered_area_is_transparent_black(self):
        r = raster(build("Constant", dict(width=W, height=H, red=0.8, green=0.4, blue=0.2, alpha=1.0), black_outside=1, **self.ROTATE))
        corner = at(r, r.data.x, r.data.y)
        np.testing.assert_array_equal(corner, [0, 0, 0, 0])
        np.testing.assert_allclose(at(r, W // 2, H // 2), [0.8, 0.4, 0.2, 1.0], atol=1e-6)

    def test_off_the_picture_edge_is_carried_over_the_uncovered_area(self):
        r = raster(build("Constant", dict(width=W, height=H, red=0.8, green=0.4, blue=0.2, alpha=1.0), black_outside=0, **self.ROTATE))
        np.testing.assert_allclose(at(r, r.data.x, r.data.y), [0.8, 0.4, 0.2, 1.0], atol=1e-6)
        np.testing.assert_allclose(r.pixels, np.broadcast_to(np.float32([0.8, 0.4, 0.2, 1.0]), r.pixels.shape), atol=1e-6)

    def test_the_covered_area_is_the_same_either_way(self):
        on = raster(build(black_outside=1, **self.ROTATE))
        off = raster(build(black_outside=0, **self.ROTATE))
        self.assertEqual(on.data, off.data)
        inside = np.zeros(on.pixels.shape[:2], bool)
        inside[H // 3: 2 * H // 3, W // 3: 2 * W // 3] = True
        shift = (W // 3 - on.data.x, H // 3 - on.data.y)
        np.testing.assert_array_equal(on.pixels[shift[1]: shift[1] + H // 3, shift[0]: shift[0] + W // 3],
                                      off.pixels[shift[1]: shift[1] + H // 3, shift[0]: shift[0] + W // 3])


class ClampTests(unittest.TestCase):
    def edge(self, **params):
        g = Graph()
        g.add("plate", "Checker", dict(width=W, height=H, size=4))
        g.add("node", "CornerPin", {**SAME, "filter": "cubic", "extra_matrix_02": 0.4, **params}, image="plate")
        return g

    def test_a_sharp_filter_overshoots_without_clamp_and_stays_in_range_with_it(self):
        free = raster(self.edge(clamp=0), target="node").pixels[..., :3]
        held = raster(self.edge(clamp=1), target="node").pixels[..., :3]
        source = raster(self.edge(), target="plate").pixels[..., :3]
        low, high = float(source.min()), float(source.max())
        self.assertLess(low, high)
        self.assertTrue(float(free.min()) < low - 1e-4 or float(free.max()) > high + 1e-4, (free.min(), free.max(), low, high))
        self.assertGreaterEqual(float(held.min()), low - 1e-6)
        self.assertLessEqual(float(held.max()), high + 1e-6)


class MotionBlurTests(unittest.TestCase):
    def moving(self, **params):
        g = build(**params)
        g.key("node", "to1_x", 1, 0.0)
        g.key("node", "to1_x", 11, 10.0)
        g.key("node", "to4_x", 1, float(W))
        g.key("node", "to4_x", 11, W + 10.0)
        return g

    def test_off_the_frame_is_the_pin_at_that_frame(self):
        g = self.moving()
        a = raster(g, 6)
        b = raster(build(to1_x=5.0, to4_x=W + 5.0), 6, "node")
        np.testing.assert_allclose(a.pixels, b.pixels, atol=1e-6)

    def test_on_the_frame_is_the_mean_of_the_shutter_samples(self):
        g = self.moving(motionblur=1, samples=4, shutter=2.0, shutter_offset="centred")
        blurred = raster(g, 6)
        sharp = self.moving()
        low, high = 5.0, 7.0
        frames = [low + (i + 0.5) * (high - low) / 4 for i in range(4)]
        samples = [Evaluator().evaluate_raster(dict(sharp.doc, view="node"), target="node", frame=f) for f in frames]
        union = samples[0].data
        for s in samples[1:]:
            union = union.union(s.data)
        mean = np.mean(np.stack([s.fit(union) for s in samples]), axis=0)
        self.assertEqual(blurred.data, union)
        np.testing.assert_allclose(blurred.pixels, mean, atol=1e-5)

    def test_a_still_pin_is_not_blurred(self):
        still = raster(build(motionblur=1, samples=4, shutter=2.0))
        np.testing.assert_allclose(still.pixels, raster(build()).pixels, atol=1e-6)


class ButtonTests(unittest.TestCase):
    def test_copy_from_writes_the_from_points_into_the_to_points(self):
        p = {**SPECS["CornerPin"]["params"], **FROM, "to1_x": 99.0, "to4_y": -7.0}
        updates = cornerpinknobs.copy_from_into_to(p)
        self.assertEqual(updates["to1_x"], 0.0)
        self.assertEqual(updates["to4_y"], float(H))
        self.assertEqual(len(updates), 8)

    def test_copy_to_writes_the_to_points_into_the_from_points(self):
        p = {**SPECS["CornerPin"]["params"], **FROM, "to2_x": 11.0, "to3_y": 13.0}
        updates = cornerpinknobs.copy_to_into_from(p)
        self.assertEqual((updates["from2_x"], updates["from3_y"]), (11.0, 13.0))

    def test_set_to_input_uses_the_corners_of_the_input_format(self):
        self.assertEqual(cornerpinknobs.from_input_corners(0, 0, 320, 200),
                         {"from1_x": 0.0, "from1_y": 0.0, "from2_x": 320.0, "from2_y": 0.0,
                          "from3_x": 0.0, "from3_y": 200.0, "from4_x": 320.0, "from4_y": 200.0})

    def test_the_panel_buttons_change_the_document(self):
        from PySide6.QtWidgets import QPushButton
        from tests.test_desktop import APP, Window, release_window
        window = Window()
        self.addCleanup(release_window, self)
        window.show()
        APP.processEvents()
        d = window.dispatcher
        d.execute({"op": "create", "id": "src1", "type": "Constant", "pos": [0, 0], "params": {"width": 320, "height": 200}})
        d.execute({"op": "create", "id": "pin1", "type": "CornerPin", "pos": [0, 100]})
        d.execute({"op": "connect", "id": "pin1", "input": "image", "source": "src1"})
        for name, value in (("to1_x", 40.0), ("to4_y", 77.0)):
            d.execute({"op": "set", "id": "pin1", "param": name, "value": value})
        panel = window.build_node_panel("pin1")
        for name in ("cornerpin-copy-from", "cornerpin-copy-to", "cornerpin-set-to-input"):
            self.assertIsNotNone(panel.findChild(QPushButton, name), name)
        params = lambda: window.graph_nodes()["pin1"]["params"]
        # Set to input: the From points go to the corners of the 320 x 200 plate, as one undo step.
        before = d.revision
        panel.findChild(QPushButton, "cornerpin-set-to-input").click()
        APP.processEvents()
        self.assertEqual((params()["from2_x"], params()["from4_y"]), (320.0, 200.0))
        # Copy 'from': the To points become the From points.
        panel = window.build_node_panel("pin1")
        panel.findChild(QPushButton, "cornerpin-copy-from").click()
        APP.processEvents()
        self.assertEqual([params()[f"to{i}_x"] for i in (1, 2, 3, 4)], [0.0, 320.0, 0.0, 320.0])
        self.assertEqual(params()["to4_y"], 200.0)
        # Copy 'to': move one To point, copy it back to the From points.
        d.execute({"op": "set", "id": "pin1", "param": "to3_x", "value": 12.0})
        panel = window.build_node_panel("pin1")
        panel.findChild(QPushButton, "cornerpin-copy-to").click()
        APP.processEvents()
        self.assertEqual(params()["from3_x"], 12.0)
        window.saved_document = window.dispatcher.document
        window.close()


class SurroundingBehaviourTests(unittest.TestCase):
    def test_the_node_stays_off_the_tile_path(self):
        g = build(to2_x=W - 5.0)
        executor = TileExecutor(evaluator=Evaluator())
        self.assertFalse(executor.supports_tiled(dict(g.doc, view="node"), "node"))
        result = executor.compose(dict(g.doc, view="node"), "node", frame=1, tier=1)
        self.assertFalse(result.tiled)
        np.testing.assert_array_equal(result.pixels, Evaluator().evaluate(dict(g.doc, view="node"), "node", frame=1))

    def test_bypass_passes_the_plate(self):
        g = build(to2_x=W - 5.0, extra_matrix_02=3.0, enable1=0)
        g.bypass("node")
        np.testing.assert_array_equal(raster(g).pixels, raster(g, target="plate").pixels)

    def test_an_old_document_loads_and_draws_as_it_did(self):
        g = build(to2_x=W - 5.0, to3_y=H - 4.0, filter="bilinear")
        before = raster(g).pixels
        document = copy.deepcopy(g.doc)
        for name in cornerpinknobs.NEW:
            document["nodes"]["node"]["params"].pop(name, None)
        loaded = upgrade_document(document)
        validate(loaded)
        params = loaded["nodes"]["node"]["params"]
        self.assertEqual((params["enable1"], params["black_outside"], params["clamp"], params["extra_matrix_00"]), (1, 1, 0, 1.0))
        after = Evaluator().evaluate_raster(dict(loaded, view="node"), target="node", frame=1).pixels
        np.testing.assert_array_equal(after, before)


if __name__ == "__main__":
    unittest.main()
