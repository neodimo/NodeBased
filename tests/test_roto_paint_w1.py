"""2D parity plan 19, step W1: RotoPaint's brush profile, mask/mix, tile path and bypass."""
import copy
import unittest

import numpy as np

from nodebased.core import SCHEMA_VERSION, Dispatcher, empty_document, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_roto_paint import stroke

SIZE = 32


def shape(points, **kw):
    value = {"kind": "shape", "name": "matte", "mode": "union", "opacity": 1.0, "feather": 0.0,
             "blend": "over", "visible": True,
             "points": [{"x": x, "y": y, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
                        for x, y in points]}
    value.update(kw)
    return value


def brush(**kw):
    value = {"size": 12.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2, "strength": 0.2}
    value.update(kw)
    return value


def graph(items, *, mask=False, second=False, mix=None):
    d = Dispatcher(empty_document())
    commands = [{"op": "create", "id": "plate", "type": "Constant",
                 "params": {"width": SIZE, "height": SIZE, "red": 0.25, "green": 0.25, "blue": 0.25}},
                {"op": "create", "id": "paint", "type": "RotoPaint"},
                {"op": "connect", "id": "paint", "input": "image", "source": "plate"}]
    if second:
        commands += [{"op": "create", "id": "second", "type": "Constant",
                      "params": {"width": SIZE, "height": SIZE, "red": 0.0, "green": 1.0, "blue": 0.0}},
                     {"op": "connect", "id": "paint", "input": "input2", "source": "second"}]
    if mask:
        # Left half of the frame passes, right half is gated off.
        commands += [{"op": "create", "id": "gate", "type": "Rectangle",
                      "params": {"width": SIZE, "height": SIZE, "box_x": 0.0, "box_y": 0.0,
                                 "box_width": SIZE / 2, "box_height": float(SIZE),
                                 "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0}},
                     {"op": "connect", "id": "paint", "input": "mask", "source": "gate"}]
    if mix is not None:
        commands.append({"op": "set", "id": "paint", "param": "mix", "value": mix})
    commands.append({"op": "set_paint_items", "id": "paint", "items": items})
    d.execute({"op": "batch", "commands": commands})
    return d


def render(d, frame=1):
    return Evaluator().evaluate_raster(d.document, "paint", frame=frame).pixels


def tiled(d, tile_edge=None, frame=1):
    kwargs = {} if tile_edge is None else {"tile_edge": tile_edge}
    executor = TileExecutor(evaluator=Evaluator(), **kwargs)
    document = dict(d.document, view="paint")
    assert executor.supports_tiled(document, "paint")
    region = executor.canvas_region(document, "paint", frame=frame, tier=1)
    return executor.compose_region(document, "paint", region, frame=frame, tier=1).pixels


class BrushProfileTests(unittest.TestCase):
    def row(self, **brush_kw):
        item = stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush(**brush_kw))
        return render(graph([item]))[16, 16:, 0]

    def test_hard_brush_is_solid_to_its_rim_and_a_soft_one_falls_off_inside_it(self):
        hard, soft = self.row(hardness=1.0), self.row(hardness=0.0)
        # radius 6 around x=16: pixel centres at 17.5 .. 21.5 are inside the rim.
        np.testing.assert_allclose(hard[:5], 1.0, atol=1e-5)
        self.assertEqual(float(hard[8]), 0.25)
        self.assertGreater(float(soft[0]), 0.9)
        self.assertLess(float(soft[4]), 0.6)
        self.assertTrue(np.all(np.diff(soft[:9]) <= 1e-6), "a soft brush never gets brighter outward")

    def test_hardness_sets_how_far_the_solid_core_reaches(self):
        mid, soft = self.row(hardness=0.5), self.row(hardness=0.0)
        self.assertGreater(float(mid[3]), float(soft[3]))
        self.assertAlmostEqual(float(mid[1]), 1.0, places=4)

    def test_brush_and_stroke_opacity_scale_the_paint(self):
        item = stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}],
                      brush=brush(opacity=0.5), opacity=0.5)
        # 25% of red over a 0.25 grey plate.
        np.testing.assert_allclose(render(graph([item]))[16, 16, :3],
                                   [0.25 * 0.75 + 0.25, 0.25 * 0.75, 0.25 * 0.75], atol=1e-5)

    def test_pressure_widens_the_stroke(self):
        thin = stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 0.4}], brush=brush())
        wide = stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush())
        self.assertLess(float((render(graph([thin]))[..., 0] > 0.3).sum()),
                        float((render(graph([wide]))[..., 0] > 0.3).sum()))


class MigrationTests(unittest.TestCase):
    def test_a_v18_stroke_stays_hard_edged_after_hardness_gained_a_falloff(self):
        item = stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush(hardness=0.5))
        document = copy.deepcopy(graph([item]).document)
        document["version"] = 18
        upgraded = upgrade_document(document)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["node_data"]["paint"]["items"][0]["brush"]["hardness"], 1.0)
        hard = Evaluator().evaluate_raster(upgraded, "paint").pixels[16, 16:22, 0]
        np.testing.assert_allclose(hard[:5], 1.0, atol=1e-5)


class ToolTests(unittest.TestCase):
    def test_eraser_over_a_shape_restores_only_its_footprint(self):
        square = shape([(4, 4), (28, 4), (28, 28), (4, 28)])
        eraser = stroke("eraser", points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush())
        image = render(graph([square, eraser]))
        np.testing.assert_allclose(image[16, 16], [0.25, 0.25, 0.25, 1.0], atol=1e-5)
        np.testing.assert_allclose(image[8, 8, :3], [1, 1, 1], atol=1e-5)

    def test_clone_with_an_offset_copies_the_plate_from_that_offset(self):
        patch = shape([(4, 4), (9, 4), (9, 9), (4, 9)])
        # Plate with a white square at (4..9); clone it 12 px right and 12 down.
        d = graph([patch, stroke("clone", points=[{"x": 18.0, "y": 18.0, "pressure": 1.0}],
                                 brush=brush(size=6.0), source_offset=[12.0, 12.0])])
        # The clone reads the plate (input), not the shapes, so rebuild with a real white square.
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "sq", "type": "Rectangle",
             "params": {"width": SIZE, "height": SIZE, "box_x": 4.0, "box_y": 4.0,
                        "box_width": 12.0, "box_height": 12.0,
                        "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0}},
            {"op": "connect", "id": "sq", "input": "image", "source": "plate"},
            {"op": "connect", "id": "paint", "input": "image", "source": "sq"},
            {"op": "set_paint_items", "id": "paint", "items": [
                stroke("clone", points=[{"x": 18.0, "y": 18.0, "pressure": 1.0}],
                       brush=brush(size=6.0), source_offset=[12.0, 12.0])]}]})
        image = render(d)
        # (18, 18) shows what sits at (6, 6): the white square. Outside the dab nothing changes.
        np.testing.assert_allclose(image[18, 18, :3], [1, 1, 1], atol=1e-5)
        np.testing.assert_allclose(image[18, 26, :3], [0.25, 0.25, 0.25], atol=1e-5)

    def test_reveal_shows_the_second_input_inside_the_stroke_only(self):
        d = graph([stroke("reveal", points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush())],
                  second=True)
        image = render(d)
        np.testing.assert_allclose(image[16, 16, :3], [0, 1, 0], atol=1e-5)
        np.testing.assert_allclose(image[2, 2, :3], [0.25, 0.25, 0.25], atol=1e-5)

    def test_single_frame_stroke_is_gone_one_frame_later(self):
        d = graph([stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush(),
                          lifetime={"mode": "single", "first": 5})])
        self.assertGreater(float(render(d, 5)[16, 16, 0]), 0.9)
        np.testing.assert_allclose(render(d, 6)[16, 16, :3], [0.25] * 3, atol=1e-6)
        np.testing.assert_allclose(render(d, 4)[16, 16, :3], [0.25] * 3, atol=1e-6)

    def test_range_lifetime_covers_its_frames_inclusive(self):
        d = graph([stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush(),
                          lifetime={"mode": "range", "first": 3, "last": 5})])
        shown = [float(render(d, f)[16, 16, 0]) > 0.9 for f in (2, 3, 5, 6)]
        self.assertEqual(shown, [False, True, True, False])


class MaskAndTileTests(unittest.TestCase):
    ITEMS = [stroke(points=[{"x": 4.0, "y": 16.0, "pressure": 1.0}, {"x": 28.0, "y": 16.0, "pressure": 0.6}],
                    brush=brush(size=10.0, hardness=0.4)),
             stroke("blur", points=[{"x": 8.0, "y": 8.0, "pressure": 1.0}, {"x": 24.0, "y": 24.0, "pressure": 1.0}],
                    brush=brush(size=14.0, hardness=0.5)),
             stroke("clone", points=[{"x": 20.0, "y": 6.0, "pressure": 1.0}], brush=brush(size=8.0),
                    source_offset=[-9.0, 5.0])]

    def test_mask_keeps_the_plate_outside_it(self):
        d = graph(self.ITEMS, mask=True)
        gated = render(d)
        ungated = render(graph(self.ITEMS))
        np.testing.assert_array_equal(gated[:, :16], ungated[:, :16])
        np.testing.assert_allclose(gated[:, 16:, :3], 0.25, atol=1e-6)

    def test_mix_fades_the_paint_layer(self):
        item = stroke(points=[{"x": 16.0, "y": 16.0, "pressure": 1.0}], brush=brush())
        half = render(graph([item], mix=0.5))[16, 16, 0]
        self.assertAlmostEqual(float(half), 0.5 * 1.0 + 0.5 * 0.25, places=5)

    def test_tiles_equal_the_full_frame_across_seams_with_a_mask(self):
        for mask in (False, True):
            d = graph(self.ITEMS, mask=mask, second=True)
            full = render(d)
            for edge in (None, 5, 8, 11):
                with self.subTest(mask=mask, edge=edge):
                    np.testing.assert_array_equal(tiled(d, edge), full)

    def test_editing_a_stroke_invalidates_the_tiles(self):
        d = graph(self.ITEMS)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        document = dict(d.document, view="paint")
        region = executor.canvas_region(document, "paint", frame=1, tier=1)
        before = executor.compose_region(document, "paint", region, frame=1, tier=1).pixels.copy()
        items = copy.deepcopy(self.ITEMS)
        items[0]["color"] = [0.0, 0.0, 1.0, 1.0]
        d.execute({"op": "set_paint_items", "id": "paint", "items": items})
        document = dict(d.document, view="paint")
        after = executor.compose_region(document, "paint", region, frame=1, tier=1).pixels
        self.assertFalse(np.array_equal(before, after))
        np.testing.assert_array_equal(after, render(d))

    def test_bypass_passes_the_input_on_both_paths(self):
        d = graph(self.ITEMS, mask=True)
        d.execute({"op": "disable", "id": "paint", "value": True})
        plate = Evaluator().evaluate_raster(d.document, "plate").pixels
        np.testing.assert_array_equal(render(d), plate)
        np.testing.assert_array_equal(tiled(d, 8), plate)


if __name__ == "__main__":
    unittest.main()
