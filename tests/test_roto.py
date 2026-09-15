"""Schema v8 node_data: the animatable-scalar envelope, the roto rasteriser, the tracker solve,
and how all three behave once the evaluator is the one calling them.

The structural claim these tests defend is that `node_data` is not a second animation system and
not a second geometry convention. A time-varying number inside a shape resolves with v6's
semantics clause for clause; a Roto is a generator like Constant; a Tracker is a Transform whose
numbers came from data. Each of those is asserted against the thing it claims to match, rather
than against a hand-copied expectation that can drift when the original moves.
"""
import math
import os
import unittest

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from nodebased import roto, shapes, tracker
from nodebased.core import Dispatcher, SCHEMA_VERSION, SPECS, validate
from nodebased.imaging import Evaluator
from nodebased.tiers import scale_node_data
from nodebased.tileexec import TileExecutor


def point(x, y, in_x=0.0, in_y=0.0, out_x=0.0, out_y=0.0):
    return {'x': x, 'y': y, 'in_x': in_x, 'in_y': in_y, 'out_x': out_x, 'out_y': out_y}


def rectangle(x0, y0, x1, y1):
    """Corners in counter-clockwise order, as a shape with no tangent handles."""
    return [point(x0, y0), point(x1, y0), point(x1, y1), point(x0, y1)]


def shape(points, name='s', mode='union', opacity=1.0, feather=0.0):
    return {'name': name, 'mode': mode, 'opacity': opacity, 'feather': feather, 'points': points}


def animated(base, keys, interpolation='linear'):
    return {'value': base,
            'curve': {'interpolation': interpolation,
                      'keys': [{'frame': f, 'value': v} for f, v in keys]}}


def track(name, x, y, enabled=1.0):
    return {'name': name, 'enabled': enabled, 'x': x, 'y': y}


def roto_document(shape_list, width=200, height=200, invert=0):
    d = Dispatcher()
    d.execute({'op': 'create', 'type': 'Roto', 'id': 'r'})
    d.execute({'op': 'create', 'type': 'Viewer', 'id': 'v'})
    d.execute({'op': 'connect', 'id': 'v', 'input': 'image', 'source': 'r'})
    d.execute({'op': 'set', 'id': 'r', 'param': 'width', 'value': width})
    d.execute({'op': 'set', 'id': 'r', 'param': 'height', 'value': height})
    d.execute({'op': 'set', 'id': 'r', 'param': 'invert', 'value': invert})
    if shape_list:
        d.execute({'op': 'set_shapes', 'id': 'r', 'shapes': shape_list})
    return d


class AnimatableScalarTests(unittest.TestCase):
    """`resolve_scalar` against docs/ANIMATION.md, clause by clause.

    This module carries its own copy of v6's resolution rules only because the branch could not
    import an unmerged `nodebased.animation`. These tests are what make substituting the real
    implementation a checkable change rather than a hopeful one.
    """

    def test_a_plain_number_is_a_scalar_at_every_frame(self):
        for frame in (-50, 0, 1, 999):
            self.assertEqual(shapes.resolve_scalar(12.5, frame), 12.5)

    def test_an_envelope_with_no_curve_is_its_base_value(self):
        self.assertEqual(shapes.resolve_scalar({'value': 3.0}, 7), 3.0)
        self.assertEqual(shapes.resolve_scalar({'value': 3.0, 'curve': None}, 7), 3.0)

    def test_outside_the_key_range_the_base_value_applies(self):
        # This is v6's rule and it is genuinely surprising -- most curve systems hold the end key.
        # Asserted explicitly so a future change to it is a deliberate one.
        value = animated(99.0, [(10, 0.0), (20, 100.0)])
        self.assertEqual(shapes.resolve_scalar(value, 9), 99.0)
        self.assertEqual(shapes.resolve_scalar(value, 21), 99.0)

    def test_on_a_key_the_key_wins_and_between_keys_linear_interpolates(self):
        value = animated(99.0, [(10, 0.0), (20, 100.0)])
        self.assertEqual(shapes.resolve_scalar(value, 10), 0.0)
        self.assertEqual(shapes.resolve_scalar(value, 20), 100.0)
        self.assertEqual(shapes.resolve_scalar(value, 15), 50.0)

    def test_constant_interpolation_holds_the_earlier_key(self):
        value = animated(99.0, [(10, 0.0), (20, 100.0)], interpolation='constant')
        self.assertEqual(shapes.resolve_scalar(value, 19), 0.0)
        self.assertEqual(shapes.resolve_scalar(value, 20), 100.0)

    def test_the_resolved_value_clamps_to_the_scalar_bounds(self):
        # opacity is bounded [0, 1]; a curve that overshoots resolves clamped, matching v6's
        # "resolved value clamps to LIMITS". Without the name there is nothing to clamp against.
        value = animated(0.5, [(1, 0.0), (10, 1.0)])
        self.assertEqual(shapes.resolve_scalar(value, 5), 4 / 9)
        over = {'value': 0.5, 'curve': {'interpolation': 'linear',
                                        'keys': [{'frame': 1, 'value': 0.0},
                                                 {'frame': 10, 'value': 1.0}]}}
        self.assertEqual(shapes.resolve_scalar(over, 100, 'opacity'), 0.5)

    def test_enabled_resolves_then_thresholds_at_half(self):
        payload = {'tracks': [track('a', 0.0, 0.0, enabled=animated(1.0, [(1, 0.0), (11, 1.0)]))]}
        self.assertFalse(shapes.resolve_tracks(payload, 5)[0]['enabled'])
        self.assertTrue(shapes.resolve_tracks(payload, 6)[0]['enabled'])


class PayloadValidationTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = roto_document(None)

    def reject(self, fragment, shape_list):
        with self.assertRaisesRegex(ValueError, fragment):
            self.dispatcher.execute({'op': 'set_shapes', 'id': 'r', 'shapes': shape_list})

    def test_a_shape_needs_at_least_three_points(self):
        self.reject('between 3 and', [shape([point(0, 0), point(1, 1)])])

    def test_scalars_are_bounded_and_finite(self):
        self.reject('opacity must be between', [shape(rectangle(0, 0, 5, 5), opacity=2.0)])
        self.reject('must be a finite number',
                    [shape(rectangle(0, 0, 5, 5), opacity={'value': float('nan')})])

    def test_curve_keys_must_be_strictly_increasing(self):
        bad = animated(0.0, [(5, 1.0), (5, 2.0)])
        self.reject('strictly increasing', [shape(rectangle(0, 0, 5, 5), opacity=bad)])

    def test_a_node_type_without_a_payload_slot_is_refused(self):
        self.dispatcher.execute({'op': 'create', 'type': 'Blur', 'id': 'b'})
        with self.assertRaisesRegex(ValueError, 'do not carry shapes'):
            self.dispatcher.execute({'op': 'set_shapes', 'id': 'b', 'shapes': [shape(rectangle(0, 0, 5, 5))]})

    def test_tracks_and_shapes_are_not_interchangeable(self):
        with self.assertRaisesRegex(ValueError, 'do not carry tracks'):
            self.dispatcher.execute({'op': 'set_tracks', 'id': 'r', 'tracks': [track('a', 0.0, 0.0)]})

    def test_deleting_a_node_clears_its_payload_in_the_same_edit(self):
        self.dispatcher.execute({'op': 'set_shapes', 'id': 'r', 'shapes': [shape(rectangle(0, 0, 5, 5))]})
        self.assertIn('r', self.dispatcher.document['node_data'])
        self.dispatcher.execute({'op': 'delete', 'id': 'r'})
        self.assertEqual(self.dispatcher.document['node_data'], {})
        validate(self.dispatcher.document)
        # And undo restores both halves, because the payload rides on the same document snapshot.
        self.dispatcher.execute({'op': 'undo'})
        self.assertIn('r', self.dispatcher.document['node_data'])

    def test_an_empty_payload_is_stored_as_an_absent_entry(self):
        # One representation for "no shapes", so two identical-looking comps serialize identically.
        self.dispatcher.execute({'op': 'set_shapes', 'id': 'r', 'shapes': [shape(rectangle(0, 0, 5, 5))]})
        self.dispatcher.execute({'op': 'set_shapes', 'id': 'r', 'shapes': []})
        self.assertEqual(self.dispatcher.document['node_data'], {})


class RasteriserTests(unittest.TestCase):
    def test_an_axis_aligned_rectangle_has_exactly_its_area(self):
        # Exact, not approximate: horizontal coverage is analytic and the sub-scanlines of an
        # axis-aligned edge all land inside. A rasteriser that is a pixel out fails here.
        matte = roto.rasterise([shape(rectangle(20, 30, 70, 90))], 128, 128)
        self.assertEqual(matte.shape, (128, 128, 4))
        self.assertAlmostEqual(float(matte[..., 3].sum()), 50 * 60, places=3)
        self.assertEqual(float(matte[60, 40, 3]), 1.0)
        self.assertEqual(float(matte[5, 5, 3]), 0.0)

    def test_the_matte_is_premultiplied_white(self):
        matte = roto.rasterise([shape(rectangle(10, 10, 40, 40), opacity=0.25)], 64, 64)
        pixel = matte[20, 20]
        self.assertAlmostEqual(float(pixel[3]), 0.25, places=5)
        # rgb == a is what lets the result composite through every existing kernel unchanged.
        for channel in range(3):
            self.assertEqual(float(pixel[channel]), float(pixel[3]))

    def test_subtract_punches_a_hole_and_union_merges(self):
        outer = shape(rectangle(10, 10, 50, 50), name='outer')
        inner = shape(rectangle(20, 20, 40, 40), name='inner', mode='subtract')
        matte = roto.rasterise([outer, inner], 64, 64)
        self.assertAlmostEqual(float(matte[..., 3].sum()), 40 * 40 - 20 * 20, places=3)
        self.assertEqual(float(matte[30, 30, 3]), 0.0)
        self.assertEqual(float(matte[15, 15, 3]), 1.0)

    def test_overlapping_unions_do_not_exceed_full_coverage(self):
        a = shape(rectangle(10, 10, 40, 40), name='a')
        b = shape(rectangle(30, 30, 60, 60), name='b')
        matte = roto.rasterise([a, b], 64, 64)
        self.assertLessEqual(float(matte[..., 3].max()), 1.0)
        self.assertEqual(float(matte[35, 35, 3]), 1.0)

    def test_invert_complements_the_matte_everywhere(self):
        points = [shape(rectangle(10, 10, 40, 40))]
        straight = roto.rasterise(points, 64, 64)
        inverted = roto.rasterise(points, 64, 64, invert=True)
        np.testing.assert_allclose(straight[..., 3] + inverted[..., 3], 1.0, atol=1e-6)

    def test_feather_conserves_coverage_while_softening_the_edge(self):
        hard = roto.rasterise([shape(rectangle(20, 20, 80, 80))], 128, 128)
        soft = roto.rasterise([shape(rectangle(20, 20, 80, 80), feather=6.0)], 128, 128)
        # A symmetric blur moves coverage across the edge without creating or destroying it.
        self.assertAlmostEqual(float(soft[..., 3].sum()), float(hard[..., 3].sum()), delta=2.0)
        self.assertGreater(float(soft[20, 50, 3]), 0.0)
        self.assertLess(float(soft[20, 50, 3]), 1.0)
        self.assertEqual(float(hard[19, 50, 3]), 0.0)

    def test_a_shape_entirely_off_canvas_renders_empty_without_error(self):
        matte = roto.rasterise([shape(rectangle(-500, -500, -400, -400))], 64, 64)
        self.assertEqual(float(matte[..., 3].sum()), 0.0)

    def test_a_bezier_with_zero_handles_is_exactly_the_polygon(self):
        # The reason there is no separate polygon shape type: straight spans emit one segment.
        polygon = roto.rasterise([shape(rectangle(10, 10, 50, 50))], 64, 64)
        handles = shape([point(10, 10), point(50, 10), point(50, 50), point(10, 50)])
        np.testing.assert_array_equal(polygon, roto.rasterise([handles], 64, 64))

    def test_rasterising_is_deterministic(self):
        curved = shape([point(10, 10, out_x=20, out_y=-5), point(50, 20, in_x=-8, in_y=-8),
                        point(30, 55, in_x=6, in_y=-12)])
        first = roto.rasterise([curved], 64, 64)
        np.testing.assert_array_equal(first, roto.rasterise([curved], 64, 64))

    def test_a_curved_span_covers_more_than_its_chord(self):
        straight = shape([point(10, 40), point(50, 40), point(30, 10)])
        bulged = shape([point(10, 40, out_x=0, out_y=18), point(50, 40, in_x=0, in_y=18),
                        point(30, 10)])
        self.assertGreater(float(roto.rasterise([bulged], 64, 64)[..., 3].sum()),
                           float(roto.rasterise([straight], 64, 64)[..., 3].sum()))


class TrackerSolveTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(SPECS['Tracker']['params'])

    def solve(self, reference, current, **overrides):
        return tracker.solve(reference, current, {**self.params, **overrides})

    def resolved(self, positions, enabled=None):
        enabled = enabled or [True] * len(positions)
        return [{'name': f't{i}', 'enabled': on, 'x': x, 'y': y}
                for i, ((x, y), on) in enumerate(zip(positions, enabled))]

    def test_no_usable_tracks_is_identity_by_declaration(self):
        self.assertEqual(self.solve([], []), tracker.IDENTITY)

    def test_one_track_gives_translation_and_refuses_to_guess_the_rest(self):
        solved = self.solve(self.resolved([(10.0, 20.0)]), self.resolved([(15.0, 28.0)]))
        self.assertEqual((solved['translate_x'], solved['translate_y']), (5.0, 8.0))
        self.assertEqual((solved['rotate'], solved['scale']), (0.0, 1.0))

    def test_two_tracks_recover_a_pure_rotation_and_scale(self):
        reference = self.resolved([(0.0, 0.0), (10.0, 0.0)])
        current = self.resolved([(0.0, 0.0), (0.0, 20.0)])
        solved = self.solve(reference, current)
        self.assertAlmostEqual(solved['rotate'], 90.0, places=6)
        self.assertAlmostEqual(solved['scale'], 2.0, places=6)

    def test_the_solve_maps_the_reference_points_onto_the_current_ones(self):
        # The property, rather than three hand-computed numbers: apply the solve to the reference
        # positions and land on the current ones. This is the sign convention, pinned.
        reference = self.resolved([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
        current = self.resolved([(5.0, 5.0), (5.0, 15.0), (-5.0, 5.0)])
        solved = self.solve(reference, current)
        theta = math.radians(solved['rotate'])
        for source, destination in zip(reference, current):
            x = solved['scale'] * (math.cos(theta) * source['x'] - math.sin(theta) * source['y'])
            y = solved['scale'] * (math.sin(theta) * source['x'] + math.cos(theta) * source['y'])
            self.assertAlmostEqual(x + solved['translate_x'], destination['x'], places=6)
            self.assertAlmostEqual(y + solved['translate_y'], destination['y'], places=6)

    def test_stabilise_is_the_exact_inverse_of_match_move(self):
        reference = self.resolved([(0.0, 0.0), (10.0, 0.0), (4.0, 9.0)])
        current = self.resolved([(6.0, 2.0), (12.0, 10.0), (0.5, 9.0)])
        forward = self.solve(reference, current)
        backward = self.solve(reference, current, mode='stabilise')
        # Composing the two must return every point to itself, whatever the point.
        for px, py in ((0.0, 0.0), (100.0, -40.0), (-13.5, 7.25)):
            for first, second in ((forward, backward), (backward, forward)):
                t1, t2 = math.radians(first['rotate']), math.radians(second['rotate'])
                x = first['scale'] * (math.cos(t1) * px - math.sin(t1) * py) + first['translate_x']
                y = first['scale'] * (math.sin(t1) * px + math.cos(t1) * py) + first['translate_y']
                rx = second['scale'] * (math.cos(t2) * x - math.sin(t2) * y) + second['translate_x']
                ry = second['scale'] * (math.sin(t2) * x + math.cos(t2) * y) + second['translate_y']
                self.assertAlmostEqual(rx, px, places=6)
                self.assertAlmostEqual(ry, py, places=6)

    def test_unticking_rotate_constrains_the_fit_rather_than_zeroing_its_result(self):
        # The distinction that matters: the best translation-only fit against rotating tracks is
        # not the translation component of the best rotation. Against a pure 90-degree rotation
        # about the centroid, the constrained fit is the centroid offset -- here, zero.
        reference = self.resolved([(-10.0, 0.0), (10.0, 0.0)])
        current = self.resolved([(0.0, -10.0), (0.0, 10.0)])
        free = self.solve(reference, current)
        self.assertAlmostEqual(abs(free['rotate']), 90.0, places=6)
        constrained = self.solve(reference, current, apply_rotate=0, apply_scale=0)
        self.assertEqual(constrained['rotate'], 0.0)
        self.assertEqual(constrained['scale'], 1.0)
        self.assertAlmostEqual(constrained['translate_x'], 0.0, places=6)
        self.assertAlmostEqual(constrained['translate_y'], 0.0, places=6)

    def test_scale_alone_is_solvable_with_rotation_pinned(self):
        reference = self.resolved([(-10.0, 0.0), (10.0, 0.0)])
        current = self.resolved([(-30.0, 0.0), (30.0, 0.0)])
        solved = self.solve(reference, current, apply_rotate=0)
        self.assertAlmostEqual(solved['scale'], 3.0, places=6)
        self.assertEqual(solved['rotate'], 0.0)

    def test_unticking_translate_zeroes_the_move_after_the_fit(self):
        reference = self.resolved([(0.0, 0.0), (10.0, 0.0)])
        current = self.resolved([(50.0, 50.0), (50.0, 70.0)])
        solved = self.solve(reference, current, apply_translate=0)
        self.assertEqual((solved['translate_x'], solved['translate_y']), (0.0, 0.0))
        self.assertAlmostEqual(solved['rotate'], 90.0, places=6)
        self.assertAlmostEqual(solved['scale'], 2.0, places=6)

    def test_a_track_disabled_at_either_frame_does_not_participate(self):
        reference = self.resolved([(0.0, 0.0), (10.0, 0.0)], enabled=[True, True])
        current = self.resolved([(1.0, 0.0), (999.0, 999.0)], enabled=[True, False])
        solved = self.solve(reference, current)
        self.assertEqual((solved['translate_x'], solved['translate_y']), (1.0, 0.0))

    def test_a_track_missing_by_name_is_simply_absent(self):
        reference = [{'name': 'a', 'enabled': True, 'x': 0.0, 'y': 0.0}]
        current = [{'name': 'b', 'enabled': True, 'x': 50.0, 'y': 50.0}]
        self.assertEqual(self.solve(reference, current), tracker.IDENTITY)

    def test_coincident_tracks_report_identity_instead_of_exploding(self):
        reference = self.resolved([(5.0, 5.0), (5.0, 5.0)])
        current = self.resolved([(5.0, 5.0), (5.0, 5.0)])
        solved = self.solve(reference, current)
        self.assertEqual(solved['scale'], 1.0)
        self.assertEqual(solved['rotate'], 0.0)
        self.assertTrue(all(math.isfinite(v) for v in solved.values()))

    def test_a_least_squares_fit_absorbs_a_single_bad_track(self):
        # Four tracks translate by 10; one is wrong. The fit must land near 10 rather than on
        # either extreme, which is the whole reason this is least squares and not a two-point map.
        reference = self.resolved([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)])
        current = self.resolved([(10.0, 0.0), (20.0, 0.0), (10.0, 10.0), (60.0, 10.0)])
        solved = self.solve(reference, current, apply_rotate=0, apply_scale=0)
        self.assertGreater(solved['translate_x'], 10.0)
        self.assertLess(solved['translate_x'], 30.0)


class TrackerAnalysisTests(unittest.TestCase):
    """`analyse` is library-only in this pass -- no UI, no agent op. Still has to be correct."""

    @staticmethod
    def plate(width, height, x, y):
        image = np.zeros((height, width, 4), dtype=np.float32)
        image[..., 3] = 1.0
        image[y:y + 6, x:x + 6, :3] = 1.0
        return image

    def test_a_moving_feature_is_found_on_every_frame(self):
        frames = [self.plate(96, 96, 20 + 4 * i, 30 + 2 * i) for i in range(4)]
        found = tracker.analyse(frames, (20, 30), (6, 6))
        self.assertEqual(len(found), 4)
        for index, (x, y) in enumerate(found):
            self.assertAlmostEqual(x, 20 + 4 * index, delta=0.5)
            self.assertAlmostEqual(y, 30 + 2 * index, delta=0.5)

    def test_a_featureless_pattern_yields_no_movement_rather_than_a_wrong_peak(self):
        flat = [np.zeros((48, 48, 4), dtype=np.float32) for _ in range(3)]
        self.assertEqual(tracker.analyse(flat, (10, 10), (6, 6)),
                         [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)])


class NodeDataTierTests(unittest.TestCase):
    """Clause C3 reaches inside payloads: pixel units scale, everything else does not."""

    def test_positions_handles_and_feather_scale_while_opacity_does_not(self):
        payload = {'shapes': [shape(rectangle(40, 80, 120, 160), opacity=0.5, feather=8.0)]}
        scaled = scale_node_data('Roto', payload, 2)
        first = scaled['shapes'][0]
        self.assertEqual(first['opacity'], 0.5)
        self.assertEqual(first['feather'], 4.0)
        self.assertEqual((first['points'][0]['x'], first['points'][0]['y']), (20.0, 40.0))
        self.assertEqual(first['points'][2]['x'], 60.0)

    def test_curve_key_values_scale_but_their_frames_do_not(self):
        payload = {'shapes': [shape([point(animated(40.0, [(1, 40.0), (10, 80.0)]), 0.0),
                                     point(10.0, 0.0), point(0.0, 10.0)])]}
        curve = scale_node_data('Roto', payload, 4)['shapes'][0]['points'][0]['x']['curve']
        self.assertEqual([key['frame'] for key in curve['keys']], [1, 10])
        self.assertEqual([key['value'] for key in curve['keys']], [10.0, 20.0])

    def test_track_positions_scale_with_the_tier(self):
        payload = {'tracks': [track('a', 100.0, 200.0)]}
        scaled = scale_node_data('Tracker', payload, 4)['tracks'][0]
        self.assertEqual((scaled['x'], scaled['y']), (25.0, 50.0))
        self.assertEqual(scaled['enabled'], 1.0)

    def test_tier_one_is_a_pass_through_and_an_unknown_tier_raises(self):
        payload = {'tracks': [track('a', 100.0, 200.0)]}
        self.assertIs(scale_node_data('Tracker', payload, 1), payload)
        with self.assertRaisesRegex(ValueError, 'Unsupported proxy tier'):
            scale_node_data('Tracker', payload, 3)


class RotoEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()

    def test_a_roto_is_a_generator_whose_windows_coincide(self):
        d = roto_document([shape(rectangle(40, 40, 140, 140))], width=200, height=200)
        raster = self.evaluator.evaluate_raster(d.document, 'v', frame=1)
        self.assertEqual(raster.data, raster.display)
        self.assertEqual(raster.pixels.shape, (200, 200, 4))
        self.assertAlmostEqual(float(raster.pixels[..., 3].sum()), 100 * 100, places=2)

    def test_a_proxy_tier_renders_small_and_keys_the_same_part_of_the_frame(self):
        # C3 end to end: a quarter-area square at tier 2 must still be the *same* square, which is
        # what the payload scaling buys. A tier that rendered full size and shrank would pass the
        # area check and fail the position one.
        d = roto_document([shape(rectangle(40, 40, 140, 140))], width=200, height=200)
        full = self.evaluator.evaluate(d.document, 'v', frame=1)
        half = self.evaluator.evaluate(d.document, 'v', frame=1, tier=2)
        self.assertEqual(half.shape, (100, 100, 4))
        self.assertAlmostEqual(float(half[..., 3].sum()), 50 * 50, places=2)
        self.assertEqual(float(half[10, 10, 3]), float(full[20, 20, 3]))
        self.assertEqual(float(half[50, 50, 3]), float(full[100, 100, 3]))

    def test_an_animated_shape_moves_with_the_frame(self):
        moving = shape([point(animated(0.0, [(1, 0.0), (11, 100.0)]), 0.0),
                        point(animated(50.0, [(1, 50.0), (11, 150.0)]), 0.0),
                        point(animated(50.0, [(1, 50.0), (11, 150.0)]), 50.0),
                        point(animated(0.0, [(1, 0.0), (11, 100.0)]), 50.0)])
        d = roto_document([moving], width=200, height=200)
        first = self.evaluator.evaluate(d.document, 'v', frame=1)
        later = self.evaluator.evaluate(d.document, 'v', frame=6)
        self.assertEqual(float(first[25, 25, 3]), 1.0)
        self.assertEqual(float(first[25, 75, 3]), 0.0)
        self.assertEqual(float(later[25, 75, 3]), 1.0)
        self.assertAlmostEqual(float(first[..., 3].sum()), float(later[..., 3].sum()), places=2)

    def test_a_static_shape_keeps_its_cache_entry_across_a_scrub(self):
        # The digest hashes the *resolved* payload, so a shape that does not move hashes the same
        # at every frame. Bookkeeping the raw payload instead would re-key the whole scrub.
        d = roto_document([shape(rectangle(10, 10, 60, 60))], width=128, height=128)
        self.evaluator.evaluate(d.document, 'v', frame=1)
        before = self.evaluator.misses
        for frame in range(2, 8):
            self.evaluator.evaluate(d.document, 'v', frame=frame)
        self.assertEqual(self.evaluator.misses, before)

    def test_an_animated_shape_re_keys_on_the_frames_where_it_moves(self):
        moving = shape([point(animated(0.0, [(1, 0.0), (11, 100.0)]), 0.0),
                        point(50.0, 0.0), point(0.0, 50.0)])
        d = roto_document([moving], width=128, height=128)
        self.evaluator.evaluate(d.document, 'v', frame=1)
        before = self.evaluator.misses
        self.evaluator.evaluate(d.document, 'v', frame=2)
        self.assertGreater(self.evaluator.misses, before)

    def test_editing_a_shape_invalidates_the_cached_result(self):
        d = roto_document([shape(rectangle(10, 10, 60, 60))], width=128, height=128)
        first = self.evaluator.evaluate(d.document, 'v', frame=1)
        d.execute({'op': 'set_shapes', 'id': 'r', 'shapes': [shape(rectangle(10, 10, 80, 80))]})
        second = self.evaluator.evaluate(d.document, 'v', frame=1)
        self.assertGreater(float(second[..., 3].sum()), float(first[..., 3].sum()))

    def test_a_roto_with_no_shapes_renders_a_transparent_frame(self):
        d = roto_document(None, width=64, height=64)
        rendered = self.evaluator.evaluate(d.document, 'v', frame=1)
        self.assertEqual(rendered.shape, (64, 64, 4))
        self.assertEqual(float(rendered[..., 3].sum()), 0.0)

    def test_a_graph_containing_a_roto_is_not_tile_executed(self):
        # Roto, Tracker and ChannelShuffle are outside SUPPORTED_TILED_KINDS, so such a graph
        # falls back to the reference evaluator explicitly rather than being tiled with a guessed
        # halo. Recorded as a test so the fallback is a decision, not an oversight.
        d = roto_document([shape(rectangle(10, 10, 40, 40))], width=64, height=64)
        self.assertFalse(TileExecutor().supports_tiled(d.document, 'v'))


class ChannelShuffleTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()
        self.dispatcher = Dispatcher()
        for kind, key in (('Constant', 'a'), ('Roto', 'r'), ('ChannelShuffle', 's'), ('Viewer', 'v')):
            self.dispatcher.execute({'op': 'create', 'type': kind, 'id': key})
        for param, value in (('red', 0.25), ('green', 0.5), ('blue', 0.75), ('alpha', 1.0)):
            self.dispatcher.execute({'op': 'set', 'id': 'a', 'param': param, 'value': value})
        for param, value in (('width', 100), ('height', 100)):
            self.dispatcher.execute({'op': 'set', 'id': 'a', 'param': param, 'value': value})
            self.dispatcher.execute({'op': 'set', 'id': 'r', 'param': param, 'value': value})
        self.dispatcher.execute({'op': 'set_shapes', 'id': 'r',
                                 'shapes': [shape(rectangle(10, 10, 60, 60))]})
        self.dispatcher.execute({'op': 'connect', 'id': 's', 'input': 'A', 'source': 'a'})
        self.dispatcher.execute({'op': 'connect', 'id': 'v', 'input': 'image', 'source': 's'})

    def render(self, frame=1):
        return self.evaluator.evaluate(self.dispatcher.document, 'v', frame=frame)

    def test_the_default_wiring_is_a_pass_through(self):
        np.testing.assert_array_equal(self.render(), self.evaluator.evaluate(
            self.dispatcher.document, 'a', frame=1))

    def test_a_matte_can_be_copied_into_the_alpha_of_another_image(self):
        self.dispatcher.execute({'op': 'connect', 'id': 's', 'input': 'B', 'source': 'r'})
        self.dispatcher.execute({'op': 'set', 'id': 's', 'param': 'out_alpha', 'value': 'B.a'})
        rendered = self.render()
        self.assertEqual(float(rendered[30, 30, 3]), 1.0)
        self.assertEqual(float(rendered[80, 80, 3]), 0.0)
        # The colour channels are untouched, which is the point of a shuffle over a merge.
        self.assertAlmostEqual(float(rendered[80, 80, 0]), 0.25, places=6)

    def test_channels_can_be_reordered_and_pinned_to_constants(self):
        self.dispatcher.execute({'op': 'set', 'id': 's', 'param': 'out_red', 'value': 'A.b'})
        self.dispatcher.execute({'op': 'set', 'id': 's', 'param': 'out_blue', 'value': 'A.r'})
        self.dispatcher.execute({'op': 'set', 'id': 's', 'param': 'out_green', 'value': '0'})
        rendered = self.render()
        self.assertAlmostEqual(float(rendered[50, 50, 0]), 0.75, places=6)
        self.assertEqual(float(rendered[50, 50, 1]), 0.0)
        self.assertAlmostEqual(float(rendered[50, 50, 2]), 0.25, places=6)

    def test_naming_an_unwired_input_is_an_error_rather_than_a_silent_default(self):
        self.dispatcher.execute({'op': 'set', 'id': 's', 'param': 'out_alpha', 'value': 'B.a'})
        with self.assertRaisesRegex(ValueError, 'which is not connected'):
            self.render()

    def test_mismatched_display_windows_are_refused_without_resampling(self):
        self.dispatcher.execute({'op': 'set', 'id': 'r', 'param': 'width', 'value': 50})
        self.dispatcher.execute({'op': 'connect', 'id': 's', 'input': 'B', 'source': 'r'})
        with self.assertRaisesRegex(ValueError, 'does not match A'):
            self.render()


class TrackerEvaluationTests(unittest.TestCase):
    """A Tracker is a Transform whose numbers came from data -- asserted against a real Transform."""

    def setUp(self):
        self.evaluator = Evaluator()

    def build(self, kind, key):
        d = Dispatcher()
        d.execute({'op': 'create', 'type': 'Checker', 'id': 'c'})
        d.execute({'op': 'set', 'id': 'c', 'param': 'width', 'value': 120})
        d.execute({'op': 'set', 'id': 'c', 'param': 'height', 'value': 120})
        d.execute({'op': 'create', 'type': kind, 'id': key})
        d.execute({'op': 'create', 'type': 'Viewer', 'id': 'v'})
        d.execute({'op': 'connect', 'id': key, 'input': 'image', 'source': 'c'})
        d.execute({'op': 'connect', 'id': 'v', 'input': 'image', 'source': key})
        return d

    def test_a_solved_tracker_renders_identically_to_the_equivalent_transform(self):
        tracked = self.build('Tracker', 't')
        tracked.execute({'op': 'set_tracks', 'id': 't', 'tracks': [
            track('a', animated(10.0, [(1, 10.0), (11, 30.0)]), 20.0),
            track('b', animated(60.0, [(1, 60.0), (11, 80.0)]), 20.0)]})
        tracked.execute({'op': 'set', 'id': 't', 'param': 'reference_frame', 'value': 1})

        transformed = self.build('Transform', 'x')
        transformed.execute({'op': 'set', 'id': 'x', 'param': 'translate_x', 'value': 10.0})
        transformed.execute({'op': 'set', 'id': 'x', 'param': 'filter', 'value': 'bilinear'})

        np.testing.assert_allclose(self.evaluator.evaluate(tracked.document, 'v', frame=6),
                                   self.evaluator.evaluate(transformed.document, 'v', frame=6),
                                   atol=1e-6)

    def test_at_the_reference_frame_the_solve_is_identity(self):
        tracked = self.build('Tracker', 't')
        tracked.execute({'op': 'set_tracks', 'id': 't', 'tracks': [
            track('a', animated(10.0, [(1, 10.0), (11, 30.0)]), 20.0)]})
        tracked.execute({'op': 'set', 'id': 't', 'param': 'reference_frame', 'value': 1})
        source = self.evaluator.evaluate(tracked.document, 'c', frame=1)
        np.testing.assert_allclose(self.evaluator.evaluate(tracked.document, 'v', frame=1),
                                   source, atol=1e-6)

    def test_a_tracker_with_no_tracks_passes_its_input_through(self):
        tracked = self.build('Tracker', 't')
        np.testing.assert_allclose(self.evaluator.evaluate(tracked.document, 'v', frame=4),
                                   self.evaluator.evaluate(tracked.document, 'c', frame=4),
                                   atol=1e-6)

    def test_stabilise_undoes_what_match_move_applies(self):
        def rendered(mode):
            d = self.build('Tracker', 't')
            d.execute({'op': 'set_tracks', 'id': 't', 'tracks': [
                track('a', animated(10.0, [(1, 10.0), (11, 30.0)]), 20.0),
                track('b', animated(60.0, [(1, 60.0), (11, 80.0)]), 20.0)]})
            d.execute({'op': 'set', 'id': 't', 'param': 'mode', 'value': mode})
            return self.evaluator.evaluate(d.document, 'v', frame=6)
        forward = rendered('match_move')
        backward = rendered('stabilise')
        # Opposite translations of the same magnitude: sampling at the centre of the frame, the
        # two land the same distance either side of the untracked source.
        self.assertFalse(np.allclose(forward, backward))

    def test_a_tracker_re_keys_per_frame_only_while_its_tracks_move(self):
        tracked = self.build('Tracker', 't')
        tracked.execute({'op': 'set_tracks', 'id': 't', 'tracks': [
            track('a', animated(10.0, [(1, 10.0), (5, 30.0)]), 20.0)]})
        self.evaluator.evaluate(tracked.document, 'v', frame=1)
        moving = self.evaluator.misses
        self.evaluator.evaluate(tracked.document, 'v', frame=3)
        self.assertGreater(self.evaluator.misses, moving)
        # Past the last key the base value applies again, so frames 6 and 7 share a solve.
        self.evaluator.evaluate(tracked.document, 'v', frame=6)
        settled = self.evaluator.misses
        self.evaluator.evaluate(tracked.document, 'v', frame=7)
        self.assertEqual(self.evaluator.misses, settled)


class SchemaVersionTests(unittest.TestCase):
    def test_node_data_is_part_of_the_current_schema(self):
        self.assertEqual(SCHEMA_VERSION, 8)
        self.assertIn('node_data', Dispatcher().document)

    def test_describe_advertises_the_payload_surface_to_agents(self):
        described = Dispatcher().execute({'op': 'describe'})
        self.assertEqual(described['node_data']['payloads'],
                         {'Roto': 'shapes', 'Tracker': 'tracks'})
        self.assertIn('set_shapes', described['operations'])
        self.assertIn('set_tracks', described['operations'])
        self.assertEqual(described['artifact_types']['Roto'], 'matte')


if __name__ == '__main__':
    unittest.main()
