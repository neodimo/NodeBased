"""Schema v8 node payloads: roto shapes, tracker tracks, and the nodes that consume them.

The invariants worth pinning here are the ones that are invisible when they break. A shape whose
pixel units are not scaled at tier 2 still renders *something*; a Tracker whose region rule and
kernel disagree about the sign of the inverse map still renders *something*. Each of those is
asserted numerically rather than by eyeball.
"""
import unittest
import tempfile
import copy
import math
import sys
import os
import time
from pathlib import Path

import numpy as np

from nodebased import roto, shapes, tiers, tiles, tracker
from nodebased.tileexec import TileExecutor
from nodebased.tiles import TileRegion
from nodebased.core import (SCHEMA_VERSION, Dispatcher, atomic_save, empty_document,
                            load_document, upgrade_document, validate)
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

    def test_open_shape_accepts_two_points_and_animatable_width(self):
        line = square(0, 0, 10, 10)
        line.update(closed=False, stroke_width=keyed(4.0, [(1, 4.0), (2, 8.0)]))
        line["points"] = line["points"][:2]
        shapes.validate_shape(line, "s")

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


class ShapeTransformTests(unittest.TestCase):
    def test_group_scale_doubles_distance_and_tangent_and_ninety_degree_rotate_is_exact(self):
        source = [{"x": 2.0, "y": 1.0, "in_x": 1.0, "in_y": 0.0,
                   "out_x": 0.0, "out_y": 2.0}]
        scaled = shapes.transform_point_group(source, center=(1.0, 1.0), scale_x=2, scale_y=2)
        self.assertEqual((scaled[0]["x"], scaled[0]["y"]), (3.0, 1.0))
        self.assertEqual((scaled[0]["in_x"], scaled[0]["in_y"]), (2.0, 0.0))
        rotated = shapes.transform_point_group(source, center=(1.0, 1.0), rotate=90)
        self.assertAlmostEqual(rotated[0]["x"], 1.0, places=12)
        self.assertAlmostEqual(rotated[0]["y"], 2.0, places=12)
        self.assertAlmostEqual(rotated[0]["out_x"], -2.0, places=12)
        self.assertAlmostEqual(rotated[0]["out_y"], 0.0, places=12)

    def test_identity_transform_raster_is_byte_identical(self):
        plain = square(4, 5, 28, 31)
        explicit = copy.deepcopy(plain)
        explicit["transform"] = dict(shapes.TRANSFORM_DEFAULTS)
        a = roto.rasterise(shapes.resolve_shapes({"shapes": [plain]}, 1), 40, 40)
        b = roto.rasterise(shapes.resolve_shapes({"shapes": [explicit]}, 1), 40, 40)
        self.assertEqual(a.tobytes(), b.tobytes())
        explicit_closed = copy.deepcopy(plain); explicit_closed["closed"] = True
        c = roto.rasterise(shapes.resolve_shapes({"shapes": [explicit_closed]}, 1), 40, 40)
        self.assertEqual(a.tobytes(), c.tobytes())

    def test_pre_flag_shape_resolves_as_closed(self):
        old_shape = square(4, 5, 28, 31)
        resolved = shapes.resolve_shapes({"shapes": [old_shape]}, 1)[0]
        self.assertTrue(resolved["closed"])

    def test_open_three_point_stroke_width_and_round_caps_at_proxy_tiers(self):
        points = [{"x": x, "y": y, "in_x": 0.0, "in_y": 0.0,
                   "out_x": 0.0, "out_y": 0.0}
                  for x, y in zip((10.5, 30.5, 50.5), (20.5, 20.5, 30.5))]
        shape = {"name": "wire", "mode": "union", "opacity": 1.0, "feather": 0.0,
                 "closed": False, "stroke_width": 6.0, "points": points}
        full = roto.rasterise(shapes.resolve_shapes({"shapes": [shape]}, 1), 64, 40)[..., 3]
        self.assertEqual(float(full[20, 10]), 1.0)
        self.assertEqual(float(full[30, 50]), 1.0)
        self.assertEqual(float(full[16, 30]), 0.0)
        self.assertEqual(float(full[10, 30]), 0.0)
        for tier in (2, 4):
            scaled = tiers.scale_node_data("Roto", {"shapes": [shape]}, tier)["shapes"][0]
            image = roto.rasterise(shapes.resolve_shapes({"shapes": [scaled]}, 1), 64//tier, 40//tier)
            self.assertGreater(float(image[20//tier, 30//tier, 3]), 0.85)

    def test_stroke_feather_expands_outward_on_both_sides(self):
        line = {"name": "wire", "mode": "union", "opacity": 1.0, "feather": 3.0,
                "closed": False, "stroke_width": 4.0,
                "points": [{"x": 5.5, "y": 16.5, "in_x": 0.0, "in_y": 0.0,
                            "out_x": 0.0, "out_y": 0.0},
                           {"x": 26.5, "y": 16.5, "in_x": 0.0, "in_y": 0.0,
                            "out_x": 0.0, "out_y": 0.0}]}
        alpha = roto.rasterise([line], 32, 32)[..., 3]
        self.assertGreater(alpha[16, 15], 0.0)
        self.assertGreater(alpha[16, 15], alpha[12, 15])
        self.assertGreater(alpha[16, 15], alpha[20, 15])

    def test_open_stroke_uses_per_point_feather_offsets(self):
        line = {"name": "wire", "mode": "union", "opacity": 1.0, "feather": 0.0,
                "closed": False, "stroke_width": 2.0,
                "points": [{"x": 5.5, "y": 16.5, "in_x": 0.0, "in_y": 0.0,
                            "out_x": 0.0, "out_y": 0.0, "feather_x": 0.0, "feather_y": 2.0},
                           {"x": 26.5, "y": 16.5, "in_x": 0.0, "in_y": 0.0,
                            "out_x": 0.0, "out_y": 0.0, "feather_x": 0.0, "feather_y": 2.0}]}
        alpha = roto.rasterise([line], 32, 32)[..., 3]
        self.assertGreater(alpha[14, 15], 0.0)

    def test_viewer_edit_inverse_round_trips_shape_and_full_tracker_transforms(self):
        shape = square(0, 0, 2, 2, track_link={"tracker_id": "t", "track_name": "a"},
                       track_follow="full", transform={"translate_x": 3.0, "rotate": 17.0,
                           "scale_x": 1.2, "scale_y": .8, "center_x": 1.0, "center_y": 1.0})
        tracks = [track("a", keyed(0, [(1, 0), (2, 4)]), keyed(0, [(1, 0), (2, 3)])),
                  track("b", keyed(2, [(1, 2), (2, 4+3*math.cos(math.radians(30)))]),
                        keyed(0, [(1, 0), (2, 3+3*math.sin(math.radians(30)))]))]
        doc = {"nodes": {"t": {"type": "Tracker", "params": {"reference_frame": 1,
                    "apply_rotate": 1, "apply_scale": 1, "apply_translate": 1, "smoothing": 0,
                    "mode": "match_move"}}}, "node_data": {"t": {"tracks": tracks}}}
        view = shapes.resolve_roto_shapes({"shapes": [shape]}, 2, doc)[0]["points"][0]
        local = shapes.unresolve_roto_point(view, shape, doc, 2)
        for field in shapes.POINT_FIELDS:
            self.assertAlmostEqual(local[field], shape["points"][0][field], places=9)

    def test_shape_transform_scales_points_and_tangents_and_rotates_about_center(self):
        shape = square(0, 0, 2, 2, transform={"center_x": 1.0, "center_y": 1.0,
                                             "scale_x": 2.0, "scale_y": 2.0,
                                             "rotate": 90.0})
        resolved = shapes.resolve_shapes({"shapes": [shape]}, 1)[0]
        self.assertAlmostEqual(resolved["points"][0]["x"], 3.0)
        self.assertAlmostEqual(resolved["points"][0]["y"], -1.0)
        shape["points"][0]["out_x"] = 3.0
        resolved = shapes.resolve_shapes({"shapes": [shape]}, 1)[0]
        self.assertAlmostEqual(resolved["points"][0]["out_y"], 6.0)

    def test_animated_shape_translate_interpolates_at_midpoint(self):
        shape = square(0, 0, 2, 2, transform={"translate_x": keyed(0, [(1, 0), (3, 10)])})
        resolved = shapes.resolve_shapes({"shapes": [shape]}, 2)[0]
        self.assertAlmostEqual(resolved["points"][0]["x"], 5.0)

    def test_full_tracker_follow_applies_rotation_and_scale(self):
        shape = square(0, 0, 2, 2, track_link={"tracker_id": "t", "track_name": "a"},
                       track_follow="full", transform={"center_x": 1.0, "center_y": 1.0})
        tracks = [track("a", keyed(0, [(1, 0), (2, 10)]), keyed(0, [(1, 0), (2, 10)])),
                  track("b", keyed(2, [(1, 2), (2, 10)]), keyed(0, [(1, 0), (2, 14)]))]
        doc = {"nodes": {"t": {"type": "Tracker", "params": {"reference_frame": 1,
                    "apply_rotate": 1, "apply_scale": 1, "apply_translate": 1, "smoothing": 0,
                    "mode": "match_move"}}}, "node_data": {"t": {"tracks": tracks}}}
        resolved = shapes.resolve_roto_shapes({"shapes": [shape]}, 2, doc)[0]
        # 90 degrees and 2x scale about the shape centre, then centroid translation (9, 12).
        self.assertAlmostEqual(resolved["points"][0]["x"], 12.0, places=6)
        self.assertAlmostEqual(resolved["points"][0]["y"], 11.0, places=6)

    def test_full_tracker_follow_matches_hand_derived_thirty_degree_similarity(self):
        shape = square(0, 0, 2, 2, track_link={"tracker_id": "t", "track_name": "a"},
                       track_follow="full", transform={"center_x": 1.0, "center_y": 1.0})
        cosine, sine = math.cos(math.radians(30)), math.sin(math.radians(30))
        tracks = [track("a", keyed(0, [(1, 0), (2, 4)]), keyed(0, [(1, 0), (2, 3)])),
                  track("b", keyed(2, [(1, 2), (2, 4+3*cosine)]),
                        keyed(0, [(1, 0), (2, 3+3*sine)]))]
        doc = {"nodes": {"t": {"type": "Tracker", "params": {"reference_frame": 1,
                    "apply_rotate": 1, "apply_scale": 1, "apply_translate": 1, "smoothing": 0,
                    "mode": "match_move"}}}, "node_data": {"t": {"tracks": tracks}}}
        result = shapes.resolve_roto_shapes({"shapes": [shape]}, 2, doc)[0]["points"][0]
        expected_x = 1.0 + 1.5*(-cosine + sine) + (4.0+3*cosine/2-1.0)
        expected_y = 1.0 - 1.5*(sine + cosine) + (3.0+3*sine/2)
        self.assertAlmostEqual(result["x"], expected_x, delta=1e-3)
        self.assertAlmostEqual(result["y"], expected_y, delta=1e-3)


class RotoTileOverheadTests(unittest.TestCase):
    def _document(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            *sets("r", width=64, height=64),
            {"op": "set_shapes", "id": "r", "shapes": [square(4, 4, 60, 60)]}]})
        return d.document

    def test_whole_frame_uses_reference_path_without_tile_rasterization_at_each_tier(self):
        from unittest.mock import patch
        doc = self._document()
        for tier in (1, 8):
            executor = TileExecutor(tile_edge=16)
            reference = Evaluator().evaluate(doc, "r", tier=tier)
            with patch.object(executor, "_roto_tile", wraps=executor._roto_tile) as tiled_roto:
                result = executor.compose(doc, "r", tier=tier)
            self.assertFalse(result.tiled)
            tiled_roto.assert_not_called()
            self.assertEqual(result.pixels.tobytes(), reference.tobytes())

    def test_shape_geometry_is_flattened_once_across_every_tile_it_touches(self):
        from unittest.mock import patch
        doc = self._document()
        executor = TileExecutor(tile_edge=16)
        request = TileRegion(16, 16, 32, 32, full_x=0, full_y=0, full_width=64, full_height=64)
        original = roto.flatten_points
        with patch.object(roto, "flatten_points", wraps=original) as flatten:
            result = executor.compose_region(doc, "r", request)
        self.assertTrue(result.tiled)
        self.assertGreater(result.tile_misses, 1)
        self.assertEqual(flatten.call_count, 1)


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
    def test_a_fresh_document_carries_an_empty_node_data_section(self):
        doc = empty_document()
        # node_data arrived in v8 and must survive every later bump. The literal head-version
        # tripwire lives in test_core/test_animation; duplicating it here would mean three
        # edits per schema bump and tempts a future me to bump without reading why.
        self.assertGreaterEqual(SCHEMA_VERSION, 8)
        self.assertEqual(doc["version"], SCHEMA_VERSION)
        self.assertEqual(doc["node_data"], {})
        validate(doc)

    def test_a_v7_document_upgrades_and_keeps_its_settings(self):
        old = empty_document()
        old.pop("node_data")
        old.pop("expressions")
        old["version"] = 7
        old["settings"]["color"]["view"] = "sRGB"
        upgraded = upgrade_document(old)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["node_data"], {})
        self.assertEqual(upgraded["settings"]["color"]["view"], "sRGB")
        validate(upgraded)

    def test_set_shapes_round_trips_through_the_dispatcher(self):
        d = Dispatcher(empty_document())
        open_shape = {"name": "wire", "mode": "union", "opacity": 1.0, "feather": 0.0,
                      "closed": False, "stroke_width": keyed(4.0, [(1, 4.0), (2, 8.0)]),
                      "points": square(0, 0, 10, 10)["points"][:2]}
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            {"op": "set_shapes", "id": "r", "shapes": [open_shape]}]})
        self.assertEqual(len(d.document["node_data"]["r"]["shapes"]), 1)
        self.assertFalse(d.document["node_data"]["r"]["shapes"][0]["closed"])
        validate(d.document)
        d.execute({"op": "undo"})
        self.assertNotIn("r", d.document["node_data"])

    def test_save_and_load_round_trip_every_shape_and_tracker_key(self):
        d = Dispatcher(empty_document())
        shape = square(4, 5, 30, 35, closed=False,
                       stroke_width=keyed(4.0, [(1, 4.0), (24, 8.0)]),
                       opacity=keyed(0.5, [(1, 0.5), (24, 0.8)], "smooth"),
                       feather=keyed(1.0, [(1, 1.0), (24, 3.0)]),
                       color=[keyed(1.0, [(1, 1.0), (24, 0.2)]), 0.25, 0.5, 1.0],
                       track_link={"tracker_id": "t", "track_name": "feature"})
        for point in shape["points"]:
            point["x"] = keyed(point["x"], [(1, point["x"]), (24, point["x"] + 5)])
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            {"op": "create", "id": "t", "type": "Tracker"},
            {"op": "set_tracks", "id": "t", "tracks": [track("feature",
                keyed(10.0, [(1, 10.0), (24, 20.0)]), keyed(11.0, [(1, 11.0), (24, 15.0)]))]},
            {"op": "set_shapes", "id": "r", "shapes": [shape]}]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "roto.nodebased"
            atomic_save(path, d.document)
            restored = load_document(path)
        self.assertEqual(restored["node_data"], d.document["node_data"])
        self.assertEqual(restored["version"], SCHEMA_VERSION)

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
        self.assertIn("open shapes render as strokes", described["node_data"]["set_shapes"]["shapes"])


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
    def test_roto_is_tile_sliced_with_payload_in_digest(self):
        self.assertIn("Roto", tiles.SUPPORTED_TILED_KINDS)

    def test_region_tiles_match_full_renders_with_feather_and_open_stroke_at_each_tier(self):
        doc = empty_document()
        dispatcher = Dispatcher(doc)
        dispatcher.execute({"op": "create", "id": "r", "type": "Roto"})
        dispatcher.execute({"op": "batch", "commands": sets("r", width=128, height=96)})
        closed = square(12, 14, 50, 52, feather=7.0)
        open_shape = square(70, 20, 108, 68, closed=False, stroke_width=9.0,
                            feather=2.0)
        open_shape["points"] = open_shape["points"][:2]
        dispatcher.execute({"op": "set_shapes", "id": "r", "shapes": [closed, open_shape]})
        for tier in tiers.PROXY_TIERS:
            reference = Evaluator().evaluate_raster(dispatcher.document, "r", tier=tier).pixels
            # Keep tiles smaller than the lowest-tier image so every tier exercises pruning.
            executor = TileExecutor(tile_edge=4)
            bounds = executor.canvas_region(dispatcher.document, "r", tier=tier)
            half_width = bounds.width // 2
            request = TileRegion(bounds.x, bounds.y, half_width, bounds.height,
                                 full_x=bounds.x, full_y=bounds.y,
                                 full_width=bounds.width, full_height=bounds.height)
            result = executor.compose_region(dispatcher.document, "r", request, tier=tier)
            self.assertTrue(result.tiled)
            self.assertEqual(result.pixels.tobytes(), reference[:, :half_width].tobytes(), f"tier {tier}")
            self.assertGreater(executor.stats["skipped_shapes"], 0)


class RenderTests(unittest.TestCase):
    def test_region_result_matches_full_render_and_skips_shapes_outside_its_bounds(self):
        shapes_to_draw = [square(4, 4, 20, 20), square(36, 36, 55, 55),
                          square(70, 70, 90, 90, feather=3.0)]
        full = roto.rasterise(shapes_to_draw, 100, 100)
        stats = {}
        roi = roto.rasterise(shapes_to_draw, 100, 100, region=(0, 0, 28, 28), stats=stats)
        self.assertEqual(roi.tobytes(), full[:28, :28].copy().tobytes())
        self.assertEqual(stats, {"rasterised_shapes": 1, "skipped_shapes": 2})

    def test_edge_feather_keeps_interior_solid_and_ends_at_the_outline_distance(self):
        shape = square(10, 10, 30, 30, feather=10.0, feather_model="edge")
        alpha = roto.rasterise([shape], 50, 50)[15, :, 3]
        self.assertEqual(float(alpha[15]), 1.0)
        self.assertEqual(float(alpha[40]), 0.0)
        self.assertGreater(alpha[30], alpha[34])
        self.assertGreater(alpha[34], alpha[39])
        self.assertEqual(float(alpha[41]), 0.0)

    def test_negative_edge_feather_moves_the_solid_edge_inward(self):
        alpha = roto.rasterise([square(10, 10, 30, 30, feather=-4.0, feather_model="edge")], 40, 40)[20, :, 3]
        self.assertEqual(float(alpha[16]), 1.0)
        self.assertEqual(float(alpha[14]), 1.0)
        self.assertLess(alpha[13], 1.0)
        self.assertLess(alpha[10], 0.5)

    def test_wide_feather_does_not_thin_a_narrow_shape(self):
        alpha = roto.rasterise([square(10, 10, 16, 30, feather=20.0, feather_model="edge")], 40, 40)[20, :, 3]
        self.assertEqual(float(alpha[13]), 1.0)

    def test_vertex_feather_offset_is_local_to_its_outline_span(self):
        shape = square(10, 10, 30, 30, feather=4.0, feather_model="edge")
        shape["points"][0]["feather_y"] = -12.0
        alpha = roto.rasterise([shape], 45, 45)[:, :, 3]
        self.assertGreater(alpha[6, 11], alpha[6, 25])
        self.assertEqual(float(alpha[25, 11]), 1.0)

    def test_all_edge_falloffs_are_monotonic(self):
        for falloff in ("linear", "smooth", "gaussian"):
            alpha = roto.rasterise([square(10, 10, 30, 30, feather=10.0, feather_model="edge",
                                            feather_falloff=falloff)], 50, 50)[15, 30:, 3]
            self.assertTrue(np.all(np.diff(alpha) <= 1e-6), falloff)

    def test_1080p_edge_feather_50_completes_within_quarter_second(self):
        shape = square(400, 200, 1100, 900, feather=50.0, feather_model="edge")
        times = []
        for _ in range(3):   # the best of three: a shared runner's noise only ever adds time
            start = time.perf_counter()
            result = roto.rasterise([shape], 1920, 1080)
            times.append(time.perf_counter() - start)
        elapsed = min(times)
        self.assertEqual(result.shape, (1080, 1920, 4))
        # The quarter second is the workstation figure; the Windows runner measures 0.29 to 0.33 s and the
        # Linux release runner measured 0.264 s on the 0.40.0 tag.
        if sys.platform == "win32":
            limit = 0.5
        elif os.environ.get("GITHUB_ACTIONS"):
            limit = 0.4
        else:
            limit = 0.25
        self.assertLess(elapsed, limit, f"edge feather render took {elapsed:.4f}s")

    def test_uniform_shape_transform_scales_feather_outline_and_point_offsets(self):
        shape = square(10, 10, 30, 30, feather=5.0, feather_model="edge",
                       transform={"scale_x": 2.0, "scale_y": 2.0, "center_x": 20.0, "center_y": 20.0})
        shape["points"][0].update(feather_x=3.0, feather_y=4.0)
        resolved = shapes.resolve_shapes({"shapes": [shape]}, 1)[0]
        self.assertEqual(resolved["feather"], 10.0)
        self.assertEqual((resolved["points"][0]["feather_x"], resolved["points"][0]["feather_y"]), (6.0, 8.0))

    def test_pre_edge_documents_migrate_to_box_feather(self):
        doc = empty_document()
        doc["version"] = 21   # the literal: feather_model is the v21 -> v22 step, whatever the current schema is
        doc["node_data"] = {"r": {"shapes": [square(8, 8, 24, 24, feather=3.0)]}}
        migrated = upgrade_document(doc)
        self.assertEqual(migrated["node_data"]["r"]["shapes"][0]["feather_model"], "box")
        legacy = square(8, 8, 24, 24, feather=3.0)
        expected = roto.rasterise([legacy], 36, 36)
        actual = roto.rasterise(shapes.resolve_shapes(migrated["node_data"]["r"], 1), 36, 36)
        np.testing.assert_array_equal(actual, expected)

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

    def test_smoothly_keyed_square_has_hand_interpolated_midpoint_corner_coverage(self):
        d = Dispatcher(empty_document())
        moving = square(8, 8, 24, 24)
        for point in moving["points"]:
            for field in ("x", "y"):
                point[field] = keyed(point[field], [(1, point[field]),
                                                     (24, point[field] + 23.0)], "smooth")
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            *sets("r", width=64, height=64),
            {"op": "set_shapes", "id": "r", "shapes": [moving]}]})
        # Smoothstep is exactly 0.5 at the midpoint: the hand-computed square is [19,35]^2.
        alpha = self.render(d.document, "r", 12).pixels[19, 19, 3]
        self.assertAlmostEqual(float(alpha), 1.0, delta=1e-3)

    def test_track_link_moves_shape_by_named_track_and_matches_tile_path(self):
        d = Dispatcher(empty_document())
        moving = square(10, 10, 20, 20, track_link={"tracker_id": "t", "track_name": "feature"})
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"},
            *sets("r", width=64, height=64),
            {"op": "create", "id": "t", "type": "Tracker"},
            {"op": "set_tracks", "id": "t", "tracks": [track("feature",
                keyed(10.0, [(1, 10.0), (24, 30.0)]),
                keyed(10.0, [(1, 10.0), (24, 25.0)]))]},
            {"op": "set_shapes", "id": "r", "shapes": [moving]}]})
        full = Evaluator().evaluate_raster(d.document, "r", frame=24).pixels
        self.assertAlmostEqual(float(full[30, 35, 3]), 1.0, delta=1e-3)
        executor = TileExecutor(tile_edge=16)
        self.assertTrue(executor.supports_tiled(d.document, "r"))
        tiled = executor.compose(d.document, "r", frame=24)
        self.assertFalse(tiled.tiled)
        np.testing.assert_allclose(tiled.pixels, full, atol=1e-6)

    def test_full_follow_render_matches_the_shared_resolved_overlay_points(self):
        d = Dispatcher(empty_document())
        moving = square(16, 16, 24, 24, track_link={"tracker_id": "t", "track_name": "a"},
                        track_follow="full", transform={"center_x": 20.0, "center_y": 20.0})
        cosine, sine = math.cos(math.radians(30)), math.sin(math.radians(30))
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "r", "type": "Roto"}, *sets("r", width=64, height=64),
            {"op": "create", "id": "t", "type": "Tracker"},
            {"op": "set_tracks", "id": "t", "tracks": [
                track("a", keyed(0, [(1,0),(2,4)]), keyed(0, [(1,0),(2,3)])),
                track("b", keyed(2, [(1,2),(2,4+3*cosine)]),
                      keyed(0, [(1,0),(2,3+3*sine)]))]},
            {"op": "set_shapes", "id": "r", "shapes": [moving]}]})
        resolved = shapes.resolve_roto_shapes(d.document["node_data"]["r"], 2, d.document)
        expected = roto.rasterise(resolved, 64, 64)
        actual = self.render(d.document, "r", 2).pixels
        np.testing.assert_allclose(actual[..., 3], expected[..., 3], atol=1e-3)
        np.testing.assert_allclose(actual[..., :3], expected[..., :3], atol=1e-3)

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


class BezierSubdivisionTests(unittest.TestCase):
    def test_de_casteljau_split_preserves_the_cubic(self):
        a = {"x": 0., "y": 0., "in_x": 0., "in_y": 0., "out_x": 12., "out_y": 25.}
        b = {"x": 30., "y": 4., "in_x": -8., "in_y": 11., "out_x": 0., "out_y": 0.}
        left, middle, right = shapes.split_cubic_point(a, b, .37)
        def cubic(p, q, t):
            u=1-t
            return tuple(u**3*p[k]+3*u*u*t*(p[k]+p["out_"+k])+3*u*t*t*(q[k]+q["in_"+k])+t**3*q[k] for k in ("x","y"))
        for i in range(101):
            t=i/100
            expected=cubic(a,b,t)
            actual=cubic(left,middle,t/.37) if t <= .37 else cubic(middle,right,(t-.37)/.63)
            self.assertAlmostEqual(expected[0],actual[0],places=8)
            self.assertAlmostEqual(expected[1],actual[1],places=8)


if __name__ == "__main__":
    unittest.main()
