"""Behaviour checks for the v13 RotoPaint payload and reference renderer."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased.core import Dispatcher, SCHEMA_VERSION, empty_document, upgrade_document
from nodebased.imaging import Evaluator, write_png
from nodebased import tiles, shapes


def stroke(tool="paint", *, points=None, **kw):
    value = {"kind": "stroke", "name": "stroke1",
             "points": points or [{"x": 8.5, "y": 8.5, "pressure": 1.0}],
             "brush": {"size": 4.0, "hardness": 1.0, "opacity": 1.0, "spacing": 0.2},
             "tool": tool, "lifetime": {"mode": "all"}, "color": [1., 0., 0., 1.],
             "source_offset": [0., 0.], "source_frame": "relative", "opacity": 1.0,
             "blend": "over", "visible": True, "follow_track": None}
    value.update(kw)
    return value


class RotoPaintRenderTests(unittest.TestCase):
    def make_graph(self, items, *, second=False):
        d = Dispatcher(empty_document())
        commands = [{"op": "create", "id": "plate", "type": "Constant"},
                    {"op": "set", "id": "plate", "param": "width", "value": 16},
                    {"op": "set", "id": "plate", "param": "height", "value": 16},
                    {"op": "set", "id": "plate", "param": "blue", "value": 1.0},
                    {"op": "create", "id": "paint", "type": "RotoPaint"},
                    {"op": "connect", "id": "paint", "input": "image", "source": "plate"}]
        if second:
            commands += [{"op": "create", "id": "second", "type": "Constant"},
                         {"op": "set", "id": "second", "param": "width", "value": 16},
                         {"op": "set", "id": "second", "param": "height", "value": 16},
                         {"op": "set", "id": "second", "param": "red", "value": 1.0},
                         {"op": "set", "id": "paint", "op2": "connect", "input": "input2", "source": "second"}]
        # Keep the normal command shape for optional second-input wiring.
        for command in commands:
            if command.get("op2"):
                command["op"] = command.pop("op2")
        commands.append({"op": "set_paint_items", "id": "paint", "items": items})
        d.execute({"op": "batch", "commands": commands})
        return d

    def render(self, document, frame=1):
        return Evaluator().evaluate_raster(document, "paint", frame=frame)

    def test_paint_stroke_rasterises_coverage_along_its_path(self):
        item = stroke(points=[{"x": 4.5, "y": 8.5, "pressure": 1.0},
                              {"x": 12.5, "y": 8.5, "pressure": 1.0}])
        image = self.render(self.make_graph([item]).document).pixels
        self.assertGreater(image[8, 8, 0], 0.9)
        self.assertGreater(image[8, 8, 3], 0.9)
        self.assertEqual(float(image[0, 0, 2]), 1.0)

    def test_per_point_pressure_changes_local_coverage(self):
        full = stroke(points=[{"x": 8.5, "y": 8.5, "pressure": 1.0}])
        light = stroke(points=[{"x": 8.5, "y": 8.5, "pressure": 0.25}])
        full_pixel = self.render(self.make_graph([full]).document).pixels[8, 8, 0]
        light_pixel = self.render(self.make_graph([light]).document).pixels[8, 8, 0]
        self.assertGreater(float(full_pixel), float(light_pixel))

    def test_eraser_restores_the_plate(self):
        painted = stroke()
        erased = stroke("eraser", points=copy.deepcopy(painted["points"]))
        image = self.render(self.make_graph([painted, erased]).document).pixels
        np.testing.assert_allclose(image[8, 8], [0.12, 0.3, 1, 1], atol=1e-6)

    def test_clone_samples_the_spatial_offset(self):
        d = self.make_graph([stroke("clone", points=[{"x": 8.5, "y": 8.5, "pressure": 1.0}],
                                    source_offset=[2.0, 2.0])])
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "patch", "type": "Rectangle"},
            {"op": "set", "id": "patch", "param": "width", "value": 16},
            {"op": "set", "id": "patch", "param": "height", "value": 16},
            {"op": "set", "id": "patch", "param": "box_x", "value": 5.0},
            {"op": "set", "id": "patch", "param": "box_y", "value": 5.0},
            {"op": "set", "id": "patch", "param": "box_width", "value": 3.0},
            {"op": "set", "id": "patch", "param": "box_height", "value": 3.0},
            {"op": "set", "id": "patch", "param": "red", "value": 1.0},
            {"op": "set", "id": "patch", "param": "blue", "value": 0.0},
            {"op": "connect", "id": "patch", "input": "image", "source": "plate"},
            {"op": "connect", "id": "paint", "input": "image", "source": "patch"}]})
        image = self.render(d.document).pixels
        self.assertGreater(image[8, 8, 0], 0.9)

    def test_reveal_uses_the_second_input(self):
        image = self.render(self.make_graph([stroke("reveal")], second=True).document).pixels
        self.assertGreater(image[8, 8, 0], 0.9)

    def test_lifetime_limits_stroke_to_its_frames(self):
        item = stroke(lifetime={"mode": "single", "first": 3})
        d = self.make_graph([item])
        self.assertAlmostEqual(float(self.render(d.document, 2).pixels[8, 8, 0]), 0.12)
        self.assertGreater(self.render(d.document, 3).pixels[8, 8, 0], 0.9)

    def test_shapes_and_strokes_composite_in_the_shared_list_order(self):
        shape = {"kind": "shape", "name": "matte", "mode": "union", "opacity": 1.0,
                 "feather": 0.0, "blend": "over", "visible": True,
                 "points": [{"x": x, "y": y, "in_x": 0.0, "in_y": 0.0,
                             "out_x": 0.0, "out_y": 0.0}
                            for x, y in ((4, 4), (12, 4), (12, 12), (4, 12))]}
        painted_then_shape = self.render(self.make_graph([stroke(), shape]).document).pixels[8, 8]
        shape_then_paint = self.render(self.make_graph([shape, stroke()]).document).pixels[8, 8]
        np.testing.assert_allclose(painted_then_shape, [1, 1, 1, 1], atol=1e-6)
        np.testing.assert_allclose(shape_then_paint, [1, 0, 0, 1], atol=1e-6)

    def test_set_paint_items_is_one_undo_step_and_changes_only_its_digest(self):
        d = self.make_graph([stroke()])
        evaluator = Evaluator()
        _, before = evaluator.evaluate_raster(d.document, "paint", return_digest=True)
        changed = copy.deepcopy(d.document["node_data"]["paint"]["items"])
        changed[0]["points"][0]["x"] = 3.5
        d.execute({"op": "set_paint_items", "id": "paint", "items": changed})
        _, after = evaluator.evaluate_raster(d.document, "paint", return_digest=True)
        self.assertNotEqual(before, after)
        d.execute({"op": "undo"})
        self.assertEqual(d.document["node_data"]["paint"]["items"][0]["points"][0]["x"], 8.5)

    def test_roto_paint_is_excluded_from_tiles(self):
        self.assertNotIn("RotoPaint", tiles.SUPPORTED_TILED_KINDS)

    def test_proxy_scaling_covers_stroke_points_brush_size_and_clone_offset(self):
        item = stroke(points=[{"x": 8.0, "y": 10.0, "pressure": 0.5}], source_offset=[4.0, -2.0])
        from nodebased.tiers import scale_node_data
        result = scale_node_data("RotoPaint", {"items": [item]}, 2)["items"][0]
        self.assertEqual((result["points"][0]["x"], result["points"][0]["y"]), (4.0, 5.0))
        self.assertEqual(result["brush"]["size"], 2.0)
        self.assertEqual(result["source_offset"], [2.0, -1.0])

    def test_a_stroke_can_follow_a_tracker(self):
        d = self.make_graph([])
        items = [stroke(points=[{"x": 6.5, "y": 8.5, "pressure": 1.0}],
                        follow_track={"node_id": "tracker", "track_index": 0})]
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "tracker", "type": "Tracker"},
            {"op": "connect", "id": "tracker", "input": "image", "source": "plate"},
            {"op": "set_tracks", "id": "tracker", "tracks": [{"name": "point", "enabled": 1.0,
                "x": {"value": 6.0, "curve": {"interpolation": "linear", "keys": [
                    {"frame": 1, "value": 6.0}, {"frame": 2, "value": 8.0}]}}, "y": 8.0}]},
            {"op": "set_paint_items", "id": "paint", "items": items}]})
        image = self.render(d.document, 2).pixels
        self.assertGreater(image[8, 8, 0], 0.9)
        self.assertLess(image[8, 5, 0], 0.2)

    def test_dustbust_replaces_a_one_frame_speck_from_the_previous_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            pattern = str(Path(temp) / "plate.####.png")
            clean = np.zeros((16, 16, 4), np.float32); clean[..., 2:] = 1.0
            speck = clean.copy(); speck[8, 8] = [1.0, 0.0, 0.0, 1.0]
            write_png(pattern.replace("####", "0001"), clean)
            write_png(pattern.replace("####", "0002"), speck)
            d = Dispatcher(empty_document())
            item = stroke("clone", points=[{"x": 8.5, "y": 8.5, "pressure": 1.0}],
                          source_offset=[0.0, 0.0], source_frame="relative",
                          lifetime={"mode": "single", "first": 2})
            d.execute({"op": "batch", "commands": [
                {"op": "create", "id": "read", "type": "Read", "params": {"path": pattern}},
                {"op": "create", "id": "paint", "type": "RotoPaint"},
                {"op": "connect", "id": "paint", "input": "image", "source": "read"},
                {"op": "set_paint_items", "id": "paint", "items": [item]}]})
            evaluator = Evaluator()
            result = evaluator.evaluate_raster(d.document, "paint", frame=2).pixels
            previous = evaluator.evaluate_raster(d.document, "read", frame=1).pixels
            np.testing.assert_allclose(result[8, 8], previous[8, 8], atol=0.01)

    def test_legacy_document_upgrades_to_current_without_touching_roto_payload(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto"})
        old = copy.deepcopy(d.document)
        old["version"] = 12
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["node_data"], old["node_data"])
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        shapes.validate_node_data(upgraded["node_data"], upgraded["nodes"])

    def test_old_roto_document_renders_identically_after_migration(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            {"op": "set", "id": "r", "param": "width", "value": 32},
            {"op": "set", "id": "r", "param": "height", "value": 32},
            {"op": "set_shapes", "id": "r", "shapes": [{"name": "box", "mode": "union",
             "opacity": 0.8, "feather": 1.0, "points": [
                {"x": x, "y": y, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
                for x, y in ((4, 4), (28, 4), (28, 28), (4, 28))]}]}]})
        old = copy.deepcopy(d.document); old["version"] = 12
        upgraded = upgrade_document(old)
        before = Evaluator().evaluate_raster(old, "r").pixels
        after = Evaluator().evaluate_raster(upgraded, "r").pixels
        np.testing.assert_array_equal(before, after)

    def test_static_stroke_digest_does_not_churn_when_scrubbing(self):
        d = self.make_graph([stroke()])
        evaluator = Evaluator()
        _, first = evaluator.evaluate_raster(d.document, "paint", frame=1, return_digest=True)
        _, later = evaluator.evaluate_raster(d.document, "paint", frame=2, return_digest=True)
        self.assertEqual(first, later)


if __name__ == "__main__":
    unittest.main()
