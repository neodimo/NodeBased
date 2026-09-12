"""Schema v8 node payloads: roto shapes, tracker tracks, and the nodes that consume them.

The invariants worth pinning here are the ones that are invisible when they break. A shape whose
pixel units are not scaled at tier 2 still renders *something*; a Tracker whose region rule and
kernel disagree about the sign of the inverse map still renders *something*. Each of those is
asserted numerically rather than by eyeball.
"""
import unittest

import numpy as np

from nodebased import roto, shapes, tiers, tiles, tracker
from nodebased.core import SCHEMA_VERSION, Dispatcher, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator


def square(x0, y0, x1, y1, **overrides):
    """An axis-aligned quad. Tangents are relative offsets, so zero handles make it a polygon."""
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    shape = {"name": "box", "mode": "union", "opacity": 1.0, "feather": 0.0,
             "points": [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0,
                         "out_x": 0.0, "out_y": 0.0} for x, y in corners]}
    shape.update(overrides)
    return shape


def sets(key, **params):
    """The dispatcher sets one parameter per command; this keeps the batches readable."""
    return [{"op": "set", "id": key, "param": name, "value": value}
            for name, value in params.items()]


def track(name, x, y, enabled=1.0):
    return {"name": name, "enabled": enabled, "x": x, "y": y}


def keyed(base, pairs, interpolation="linear"):
    return {"value": base,
            "curve": {"interpolation": interpolation,
                      "keys": [{"frame": f, "value": v} for f, v in pairs]}}


class ShapeValidationTests(unittest.TestCase):
    def test_a_well_formed_shape_validates(self):
        shapes.validate_shape(square(0, 0, 10, 10), "s")

    def test_fewer_than_three_points_is_rejected(self):
        bad = square(0, 0, 10, 10)
        bad["points"] = bad["points"][:2]
        with self.assertRaisesRegex(ValueError, "between 3 and"):
            shapes.validate_shape(bad, "s")

    def test_unknown_shape_field_is_rejected(self):
        bad = square(0, 0, 10, 10)
        bad["blend"] = "screen"
        with self.assertRaisesRegex(ValueError, "defines exactly"):
            shapes.validate_shape(bad, "s")

    def test_opacity_outside_its_bounds_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "opacity must be between"):
            shapes.validate_shape(square(0, 0, 10, 10, opacity=1.5), "s")

    def test_a_bad_curve_inside_a_shape_is_rejected_by_the_animation_rules(self):
        bad = square(0, 0, 10, 10)
        bad["points"][0]["x"] = {"value": 0.0, "curve": {"interpolation": "linear", "keys": []}}
        with self.assertRaises(ValueError) as caught:
            shapes.validate_shape(bad, "s")
        # The location must survive, or an artist cannot find which point is wrong.
        self.assertIn("points[0].x", str(caught.exception))

    def test_track_requires_exactly_its_four_fields(self):
        with self.assertRaisesRegex(ValueError, "defines exactly"):
            shapes.validate_track({"name": "t", "x": 1.0, "y": 2.0}, "t")

    def test_payload_must_match_the_node_type(self):
        with self.assertRaisesRegex(ValueError, "do not carry node_data"):
            shapes.validate_payload("Grade", {"shapes": []}, "p")
        with self.assertRaisesRegex(ValueError, "defines exactly 'tracks'"):
            shapes.validate_payload("Tracker", {"shapes": []}, "p")

    def test_orphan_node_data_is_an_error_rather_than_a_silent_drop(self):
        with self.assertRaisesRegex(ValueError, "does not name a node"):
            shapes.validate_node_data({"ghost": {"shapes": []}}, {})


class ScalarResolutionTests(unittest.TestCase):
    def test_a_plain_number_resolves_to_itself_at_every_frame(self):
        self.assertEqual(shapes.resolve_scalar(4.5, 1, "x"), 4.5)
        self.assertEqual(shapes.resolve_scalar(4.5, 999, "x"), 4.5)

    def test_a_curve_beats_the_base_value_and_holds_its_endpoints(self):
        value = keyed(-1.0, [(10, 100.0), (20, 200.0)])
        self.assertEqual(shapes.resolve_scalar(value, 15, "x"), 150.0)
        # Endpoint hold, matching schema v6 and Nuke -- not the base value.
        self.assertEqual(shapes.resolve_scalar(value, 1, "x"), 100.0)
        self.assertEqual(shapes.resolve_scalar(value, 999, "x"), 200.0)

    def test_a_resolved_value_clamps_to_its_declared_bounds(self):
        self.assertEqual(shapes.resolve_scalar(keyed(0.0, [(1, -5.0), (2, 5.0)]), 1, "opacity"), 0.0)

    def test_enabled_thresholds_at_a_half(self):
        payload = {"tracks": [track("a", 0.0, 0.0, enabled=keyed(1.0, [(1, 1.0), (3, 0.0)]))]}
        self.assertTrue(shapes.resolve_tracks(payload, 1)[0]["enabled"])
        self.assertFalse(shapes.resolve_tracks(payload, 3)[0]["enabled"])


class RasteriseTests(unittest.TestCase):
    def test_a_hard_square_covers_exactly_its_area(self):
        matte = roto.rasterise([square(16, 16, 48, 48)], 64, 64)
        self.assertEqual(matte.shape, (64, 64, 4))
        self.assertAlmostEqual(float(matte[..., 3].sum()), 32.0 * 32.0, places=3)
        np.testing.assert_allclose(matte[32, 32], [1, 1, 1, 1])
        np.testing.assert_allclose(matte[0, 0], [0, 0, 0, 0])

    def test_the_matte_stays_premultiplied(self):
        matte = roto.rasterise([square(8, 8, 24, 24, opacity=0.5)], 32, 32)
        np.testing.assert_allclose(matte[16, 16], [0.5, 0.5, 0.5, 0.5], atol=1e-6)

    def test_invert_complements_coverage(self):
        plain = roto.rasterise([square(16, 16, 48, 48)], 64, 64)
        inverted = roto.rasterise([square(16, 16, 48, 48)], 64, 64, invert=True)
        np.testing.assert_allclose(plain[..., 3] + inverted[..., 3], np.ones((64, 64)), atol=1e-6)

    def test_subtract_punches_a_hole(self):
        matte = roto.rasterise([square(8, 8, 56, 56), square(24, 24, 40, 40, mode="subtract")],
                               64, 64)
        self.assertAlmostEqual(float(matte[32, 32, 3]), 0.0, places=5)
        self.assertAlmostEqual(float(matte[12, 12, 3]), 1.0, places=5)

    def test_no_shapes_is_an_empty_matte_rather_than_an_error(self):
        matte = roto.rasterise([], 16, 16)
        self.assertEqual(float(matte.sum()), 0.0)


class TrackerSolveTests(unittest.TestCase):
    def payload(self, *positions):
        """One track per position pair, keyed frame 1 -> frame 2."""
        return {"tracks": [track(f"t{i}", keyed(a[0], [(1, a[0]), (2, b[0])]),
                                 keyed(a[1], [(1, a[1]), (2, b[1])]))
                           for i, (a, b) in enumerate(positions)]}

    def test_no_payload_solves_to_identity(self):
        self.assertEqual(tracker.solve(None, 5, {"reference_frame": 1}), dict(tracker.IDENTITY))

    def test_pure_translation_is_recovered_without_rotation_or_scale(self):
        payload = self.payload(((10.0, 10.0), (20.0, 15.0)), ((30.0, 10.0), (40.0, 15.0)))
        solved = tracker.solve(payload, 2, {"reference_frame": 1})
        self.assertAlmostEqual(solved["translate_x"], 10.0)
        self.assertAlmostEqual(solved["translate_y"], 5.0)
        self.assertAlmostEqual(solved["rotate"], 0.0)
        self.assertAlmostEqual(solved["scale"], 1.0)

    def test_a_doubling_separation_is_a_scale_of_two(self):
        payload = self.payload(((0.0, 0.0), (0.0, 0.0)), ((10.0, 0.0), (20.0, 0.0)))
        solved = tracker.solve(payload, 2, {"reference_frame": 1})
        self.assertAlmostEqual(solved["scale"], 2.0, places=6)
        self.assertAlmostEqual(solved["rotate"], 0.0, places=6)

    def test_a_quarter_turn_is_ninety_degrees(self):
        payload = self.payload(((0.0, 0.0), (0.0, 0.0)), ((10.0, 0.0), (0.0, 10.0)))
        solved = tracker.solve(payload, 2, {"reference_frame": 1})
        self.assertAlmostEqual(abs(solved["rotate"]), 90.0, places=4)
        self.assertAlmostEqual(solved["scale"], 1.0, places=6)

    def test_one_track_measures_translation_only(self):
        payload = self.payload(((10.0, 10.0), (25.0, 40.0)))
        solved = tracker.solve(payload, 2, {"reference_frame": 1})
        self.assertAlmostEqual(solved["translate_x"], 15.0)
        self.assertAlmostEqual(solved["translate_y"], 30.0)
        self.assertEqual(solved["rotate"], 0.0)
        self.assertEqual(solved["scale"], 1.0)

    def test_coincident_reference_points_yield_translation_rather_than_spin(self):
        payload = self.payload(((5.0, 5.0), (9.0, 5.0)), ((5.0, 5.0), (9.0, 5.0)))
        solved = tracker.solve(payload, 2, {"reference_frame": 1})
        self.assertEqual(solved["rotate"], 0.0)
        self.assertEqual(solved["scale"], 1.0)
        self.assertAlmostEqual(solved["translate_x"], 4.0)

    def test_a_track_disabled_at_either_frame_leaves_the_solve(self):
        payload = self.payload(((10.0, 10.0), (20.0, 10.0)), ((30.0, 10.0), (99.0, 99.0)))
        payload["tracks"][1]["enabled"] = keyed(1.0, [(1, 1.0), (2, 0.0)])
        solved = tracker.solve(payload, 2, {"reference_frame": 1})
        self.assertAlmostEqual(solved["translate_x"], 10.0)
        self.assertAlmostEqual(solved["translate_y"], 0.0)

    def test_every_track_dropped_is_identity_not_an_undefined_region(self):
        payload = self.payload(((10.0, 10.0), (20.0, 10.0)))
        payload["tracks"][0]["enabled"] = 0.0
        self.assertEqual(tracker.solve(payload, 2, {"reference_frame": 1}), dict(tracker.IDENTITY))

    def test_switching_a_component_off_removes_it(self):
        payload = self.payload(((0.0, 0.0), (5.0, 0.0)), ((10.0, 0.0), (25.0, 0.0)))
        params = {"reference_frame": 1, "apply_scale": 0, "apply_translate": 0}
        solved = tracker.solve(payload, 2, params)
        self.assertEqual(solved["scale"], 1.0)
        self.assertEqual(solved["translate_x"], 0.0)

    def test_stabilise_inverts_the_gated_transform(self):
        payload = self.payload(((0.0, 0.0), (0.0, 0.0)), ((10.0, 0.0), (20.0, 0.0)))
        match = tracker.solve(payload, 2, {"reference_frame": 1, "mode": "match_move"})
        stab = tracker.solve(payload, 2, {"reference_frame": 1, "mode": "stabilise"})
        self.assertAlmostEqual(stab["translate_x"], -match["translate_x"])
        self.assertAlmostEqual(stab["rotate"], -match["rotate"])
        self.assertAlmostEqual(stab["scale"], 1.0 / match["scale"], places=6)

    def test_stabilise_does_not_invert_a_component_that_was_gated_off(self):
        payload = self.payload(((0.0, 0.0), (0.0, 0.0)), ((10.0, 0.0), (20.0, 0.0)))
        stab = tracker.solve(payload, 2, {"reference_frame": 1, "mode": "stabilise",
                                          "apply_scale": 0})
        self.assertEqual(stab["scale"], 1.0)


class NodeDataSchemaTests(unittest.TestCase):
    def test_a_fresh_document_is_v8_with_an_empty_node_data_section(self):
        doc = empty_document()
        self.assertEqual(SCHEMA_VERSION, 8)
        self.assertEqual(doc["version"], 8)
        self.assertEqual(doc["node_data"], {})
        validate(doc)

    def test_a_v7_document_upgrades_and_keeps_its_settings(self):
        old = empty_document()
        old.pop("node_data")
        old["version"] = 7
        old["settings"]["color"]["view"] = "sRGB"
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["version"], 8)
        self.assertEqual(upgraded["node_data"], {})
        self.assertEqual(upgraded["settings"]["color"]["view"], "sRGB")
        validate(upgraded)

    def test_set_shapes_round_trips_through_the_dispatcher(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            {"op": "set_shapes", "id": "r", "shapes": [square(0, 0, 10, 10)]}]})
        self.assertEqual(len(d.document["node_data"]["r"]["shapes"]), 1)
        validate(d.document)
        d.execute({"op": "undo"})
        self.assertNotIn("r", d.document["node_data"])

    def test_an_empty_list_clears_the_entry_rather_than_storing_a_husk(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            {"op": "set_shapes", "id": "r", "shapes": [square(0, 0, 10, 10)]},
            {"op": "set_shapes", "id": "r", "shapes": []}]})
        self.assertNotIn("r", d.document["node_data"])

    def test_the_wrong_slot_for_a_node_type_is_refused(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Roto"})
        with self.assertRaisesRegex(ValueError, "do not carry tracks"):
            d.execute({"op": "set_tracks", "id": "r", "tracks": [track("a", 1.0, 1.0)]})

    def test_deleting_a_node_takes_its_payload_with_it(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            {"op": "set_shapes", "id": "r", "shapes": [square(0, 0, 10, 10)]},
            {"op": "delete", "id": "r"}]})
        self.assertEqual(d.document["node_data"], {})
        validate(d.document)

    def test_describe_exposes_the_payload_surface(self):
        described = Dispatcher(empty_document()).execute({"op": "describe"})
        self.assertEqual(described["node_data"]["payloads"]["Roto"], "shapes")
        self.assertIn("set_shapes", described["operations"])


class PayloadProxyScalingTests(unittest.TestCase):
    def test_pixel_unit_scalars_scale_and_unitless_ones_do_not(self):
        shape = square(16, 16, 48, 48, opacity=0.5, feather=4.0)
        shape["points"][1]["out_y"] = 16.0
        scaled = tiers.scale_node_data("Roto", {"shapes": [shape]}, 2)
        point = scaled["shapes"][0]["points"][1]
        self.assertAlmostEqual(point["x"], 24.0)
        # A tangent handle is a pixel offset, so it has to shrink with the point it belongs to
        # or the curve changes shape at proxy.
        self.assertAlmostEqual(point["out_y"], 8.0)
        self.assertAlmostEqual(scaled["shapes"][0]["feather"], 2.0)
        self.assertEqual(scaled["shapes"][0]["opacity"], 0.5)

    def test_curve_keys_move_with_their_scalar(self):
        payload = {"tracks": [track("a", keyed(40.0, [(1, 40.0), (2, 80.0)]), 0.0)]}
        scaled = tiers.scale_node_data("Tracker", payload, 4)
        keys = scaled["tracks"][0]["x"]["curve"]["keys"]
        self.assertEqual([k["value"] for k in keys], [10.0, 20.0])
        self.assertEqual([k["frame"] for k in keys], [1, 2])

    def test_tier_one_is_left_alone(self):
        payload = {"shapes": [square(16, 16, 48, 48)]}
        self.assertIs(tiers.scale_node_data("Roto", payload, 1), payload)


class TilePathTests(unittest.TestCase):
    def test_payload_carrying_kinds_stay_off_the_tiled_path(self):
        """The tile digest has no node_data term, so a payload kind must not tile.

        This guard is the thing that fails the day someone adds Roto to the tiled set: the tile
        cache key would then be blind to the shapes, and a scrub would serve the previous frame's
        matte. Whoever makes that change has to fold the payload into the digest first.
        """
        overlap = sorted(set(shapes.NODE_DATA_SCHEMA) & set(tiles.SUPPORTED_TILED_KINDS))
        self.assertEqual(overlap, [], f"tiled kinds carrying node_data: {overlap}")


class RenderTests(unittest.TestCase):
    def render(self, doc, key, frame=1):
        return Evaluator().evaluate_raster(doc, key, frame=frame)

    def test_a_roto_node_renders_its_matte_through_the_evaluator(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            *sets("r", width=64, height=64),
            {"op": "set_shapes", "id": "r", "shapes": [square(16, 16, 48, 48)]}]})
        raster = self.render(d.document, "r")
        self.assertAlmostEqual(float(raster.pixels[..., 3].sum()), 1024.0, places=3)

    def test_an_animated_shape_moves_between_frames(self):
        d = Dispatcher(empty_document())
        moving = square(8, 8, 24, 24)
        for point in moving["points"]:
            point["x"] = keyed(point["x"], [(1, point["x"]), (2, point["x"] + 20.0)])
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            *sets("r", width=64, height=64),
            {"op": "set_shapes", "id": "r", "shapes": [moving]}]})
        one = self.render(d.document, "r", 1).pixels[..., 3]
        two = self.render(d.document, "r", 2).pixels[..., 3]
        self.assertAlmostEqual(float(one.sum()), float(two.sum()), places=3)
        self.assertGreater(float(one[16, 16]), 0.5)
        self.assertLess(float(one[16, 40]), 0.5)
        self.assertGreater(float(two[16, 36]), 0.5)

    def test_channel_shuffle_routes_an_alpha_from_b_into_a(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "c", "type": "Constant"},
            # Opaque, because a Constant premultiplies: red 1 at alpha 0 stores black.
            *sets("c", width=64, height=64, red=1.0, green=0.0, blue=0.0, alpha=1.0),
            {"op": "create", "id": "r", "type": "Roto"},
            *sets("r", width=64, height=64),
            {"op": "set_shapes", "id": "r", "shapes": [square(16, 16, 48, 48)]},
            {"op": "create", "id": "s", "type": "ChannelShuffle"},
            {"op": "connect", "id": "s", "input": "A", "source": "c"},
            {"op": "connect", "id": "s", "input": "B", "source": "r"},
            *sets("s", out_alpha="B.a")]})
        pixels = self.render(d.document, "s").pixels
        np.testing.assert_allclose(pixels[32, 32], [1, 0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(pixels[0, 0], [1, 0, 0, 0], atol=1e-6)

    def test_naming_b_without_wiring_b_is_an_error_rather_than_black(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "c", "type": "Constant"},
            {"op": "create", "id": "s", "type": "ChannelShuffle"},
            {"op": "connect", "id": "s", "input": "A", "source": "c"},
            *sets("s", out_alpha="B.a")]})
        with self.assertRaises(ValueError):
            self.render(d.document, "s")

    def test_a_tracker_shifts_its_input_by_the_solved_translation(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "c", "type": "Constant"},
            *sets("c", width=64, height=64),
            {"op": "create", "id": "t", "type": "Tracker"},
            {"op": "connect", "id": "t", "input": "image", "source": "c"},
            {"op": "set_tracks", "id": "t", "tracks": [
                track("a", keyed(10.0, [(1, 10.0), (2, 20.0)]),
                      keyed(10.0, [(1, 10.0), (2, 15.0)]))]}]})
        first = self.render(d.document, "t", 1)
        second = self.render(d.document, "t", 2)
        # Frame 1 is the reference frame, so the solve is identity: the 64x64 input plus the
        # one-pixel bilinear margin the Transform kernel always carries.
        self.assertEqual((first.data.x, first.data.y), (-1, -1))
        self.assertEqual((first.data.width, first.data.height), (66, 66))
        # Frame 2 carries the solved +10/+5 shift. The window moves by exactly that and keeps its
        # size, which is what proves the shift is a translation and not a resize in disguise.
        self.assertEqual(second.data.x - first.data.x, 10)
        self.assertEqual(second.data.y - first.data.y, 5)
        self.assertEqual(second.data.width, first.data.width)
        self.assertEqual(second.data.height, first.data.height)
        # The display window is the format and must not drift with the solve.
        self.assertEqual(first.display, second.display)


if __name__ == "__main__":
    unittest.main()
