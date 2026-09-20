"""CPU splat work accounting, interactive budgets and evaluator integration."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import scene3d as s, splatraster as r, splats
from nodebased.cancellation import Cancelled
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_3d_splat_render import cloud


def random_scene(layered=False):
    rng = np.random.default_rng(73)
    c = replace(cloud(25), positions=rng.uniform(-1, 1, (25, 3)),
                scales=rng.uniform(.03, .5, (25, 3)))
    cards = (s._card(3, 3, (.2, .3, .4, .5), s.Transform3D()),) if layered else ()
    return s.Scene(cards, splats=(s.SplatInstance(c),))


class ProgressTests(unittest.TestCase):
    def check_events(self, events):
        self.assertEqual(events[0], ('prepare', 0.0, {}))
        self.assertEqual(events[1][:2], ('splats', 0.0))
        self.assertEqual(events[-1][:2], ('done', 1.0))
        fractions = [event[1] for event in events]
        self.assertEqual(fractions, sorted(fractions))
        self.assertTrue(all(0 <= f <= 1 for f in fractions))
        self.assertLessEqual(len(events), 203)
        total = events[1][2]['tile_work']
        for _, _, info in events[1:]:
            self.assertEqual(info['tile_work'], total)
            self.assertEqual(info['estimate_seconds'], r.estimate_seconds(total))

    def test_accumulation_and_band_shares(self):
        prepared = r.prepare_splats(random_scene().splats, s.Camera(), 321, 239)
        events = []
        actual = r.accumulate_splats(prepared, progress=lambda *a: events.append(a))
        expected = r.accumulate_splats(prepared)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertGreater(len(events), 1)
        self.assertLessEqual(len(events), 200)
        self.assertEqual(events[-1], (prepared.tile_work, prepared.tile_work))
        self.assertEqual([a for a, _ in events], sorted(a for a, _ in events))
        self.assertTrue(all(b == prepared.tile_work for _, b in events))
        totals = []
        for lo, hi in ((0, 1), (1, 19), (19, 117), (117, 239)):
            events = []
            r.accumulate_splats(prepared, rows=(lo, hi), progress=lambda *a: events.append(a))
            self.assertEqual(events[-1][0], events[-1][1])
            totals.append(events[-1][1])
        self.assertEqual(sum(totals), prepared.tile_work)

    def test_global_progress_and_bit_identity(self):
        for layered in (False, True):
            for mode in ('raster', 'raytrace'):
                for output in ('rgba', 'splats', 'depth'):
                    with self.subTest(layered=layered, mode=mode, output=output), \
                         patch.object(s, 'LAYER_BAND_BYTES', 37 * 384 * 3):
                        scene = random_scene(layered)
                        events = []
                        kwargs = dict(output=output, mode=mode, samples=2)
                        expected = s.render(scene, s.Camera(), 37, 29, **kwargs)
                        actual = s.render(scene, s.Camera(), 37, 29,
                                          progress=lambda *a: events.append(a), **kwargs)
                        np.testing.assert_array_equal(actual, expected)
                        self.check_events(events)
                        scale = 1 if output == 'depth' else 2
                        prepared = r.prepare_splats(scene.splats, s.Camera(), 37*scale, 29*scale,
                                                    output=output)
                        self.assertEqual(events[1][2]['tile_work'], prepared.tile_work)

    def test_interactive_bypasses_both_guards(self):
        scene = s.Scene(splats=(s.SplatInstance(cloud(scale=100)),))
        with patch.object(r, 'SPLAT_WORK_BUDGET', 1):
            with self.assertRaisesRegex(ValueError, 'Splat render exceeds the CPU reference budget: tile work at least'):
                s.render(scene, s.Camera(), 32, 32)
            prepared = r.prepare_splats(scene.splats, s.Camera(), 32, 32, enforce_budget=False)
            self.assertEqual(prepared.tile_work, 1024)
            events = []
            actual = s.render(scene, s.Camera(), 32, 32, progress=lambda *a: events.append(a))
        with patch.object(r, 'SPLAT_WORK_BUDGET', 10**12):
            expected = s.render(scene, s.Camera(), 32, 32)
        np.testing.assert_array_equal(actual, expected)
        self.check_events(events)
        # A small bbox with wide tile coverage reaches the second refusal guard.
        scene = s.Scene(splats=(s.SplatInstance(cloud(64, scale=.0001)),))
        with patch.object(r, 'SPLAT_WORK_BUDGET', 2048):
            with self.assertRaisesRegex(ValueError, 'tile work 65,536 evaluations > budget 2,048'):
                s.render(scene, s.Camera(), 32, 32)
            s.render(scene, s.Camera(), 32, 32, progress=lambda *a: None)

    def test_cancel_callback_and_recovery(self):
        scene = random_scene(True)
        def abort(*args):
            raise Cancelled()
        prepared = r.prepare_splats(scene.splats, s.Camera(), 37, 29)
        with self.assertRaises(Cancelled):
            r.accumulate_splats(prepared, progress=abort)
        r.accumulate_splats(prepared)
        for stage in ('prepare', 'splats', 'done'):
            def callback(current, fraction, info):
                if current == stage and (stage != 'splats' or fraction > 0):
                    raise Cancelled()
            with self.assertRaises(Cancelled):
                s.render(scene, s.Camera(), 37, 29, progress=callback)
        np.testing.assert_array_equal(s.render(scene, s.Camera(), 37, 29),
                                      s.render(scene, s.Camera(), 37, 29, progress=lambda *a: None))

    def test_empty_and_non_splat_outputs(self):
        events = []
        for output in ('rgba', 'splats', 'depth'):
            s.render(s.Scene(), s.Camera(), 17, 19, output=output, progress=lambda *a: events.append(a))
        s.render(random_scene(), s.Camera(), 17, 19, output='albedo', progress=lambda *a: events.append(a))
        self.assertEqual(events, [])
        scene = s.Scene(splats=(s.SplatInstance(cloud(0)),))
        s.render(scene, s.Camera(), 17, 19, progress=lambda *a: events.append(a))
        self.check_events(events)
        self.assertEqual(events[-1][2]['tile_work'], 0)
        prepared = r.prepare_splats(scene.splats, s.Camera(), 17, 19)
        calls = []
        r.accumulate_splats(prepared, progress=lambda *a: calls.append(a))
        self.assertEqual(calls[-1], (0, 0))

    def test_estimator(self):
        rate = r.SPLAT_REFERENCE_EVALS_PER_SECOND
        self.assertEqual(r.estimate_seconds(2*rate), 2)
        self.assertEqual(r.estimate_eta_seconds(0, rate, 99), 1)
        self.assertEqual(r.estimate_eta_seconds(rate*.01, rate, 99), .99)
        self.assertEqual(r.estimate_eta_seconds(2, 100, 10), 490)
        self.assertEqual(r.estimate_eta_seconds(50, 100, 10), 10)
        self.assertEqual(r.estimate_eta_seconds(100, 100, 10), 0)
        self.assertEqual(r.estimate_eta_seconds(0, 0, 10), 0)
        self.assertEqual(r.estimate_eta_seconds(110, 100, -10), 0)
        values = [r.estimate_eta_seconds(done, 100, done/rate) for done in range(101)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_evaluator_progress_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cloud.ply'
            splats.write_ply(cloud(scale=1), path)
            d, evaluator = Dispatcher(), Evaluator()
            self.assertIsNone(evaluator.progress)
            for key, kind, params in (
                ('read', 'ReadSplat3D', dict(splat_path=str(path))),
                ('scene', 'Scene3D', {}), ('camera', 'Camera3D', {}),
                ('render', 'Render3D', dict(width=33, height=33, render_backend='cpu'))):
                d.execute(dict(op='create', id=key, type=kind, params=params))
            for key, slot, source in (('scene', 'object0', 'read'),
                                      ('render', 'scene', 'scene'), ('render', 'camera', 'camera')):
                d.execute(dict(op='connect', id=key, input=slot, source=source))
            with patch.object(r, 'SPLAT_WORK_BUDGET', 1):
                with self.assertRaisesRegex(ValueError, 'Splat render exceeds the CPU reference budget:'):
                    evaluator.evaluate(d.document, 'render')
                events = []
                evaluator.progress = lambda *a: events.append(a)
                actual = evaluator.evaluate(d.document, 'render')
                self.check_events(events)
                events.clear()
                np.testing.assert_array_equal(actual, evaluator.evaluate(d.document, 'render'))
                self.assertEqual(events, [])
            with patch.object(r, 'SPLAT_WORK_BUDGET', 10**12):
                np.testing.assert_array_equal(actual, Evaluator().evaluate(d.document, 'render'))


if __name__ == '__main__':
    unittest.main()
