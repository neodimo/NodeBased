"""Bypassing (disabling) a node: what passes through, on both evaluation paths, and how fast.

DiMo, 2026-09-20: bypassing a Merge put the text 'grade' in the viewer (a KeyError), and bypassing a
Grade changed nothing. Both were real. The evaluator looked up inputs it had deliberately not
evaluated, and the tile path gathered the passed-through input and then ran the node's kernel on it
anyway.
"""
import time
import unittest
import uuid

import numpy as np

from nodebased.core import Dispatcher, SPECS, OUTPUT_TYPES, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, SUPPORTED_TILED_KINDS

# Parameters that make each filter visibly change its input, so "bypassed" cannot pass by accident.
VISIBLE = {'Grade': dict(exposure=2.0), 'ColorCorrect': dict(saturation=0.0), 'Blur': dict(radius=6.0),
           'Transform': dict(translate_x=17.0), 'Premult': {}, 'Unpremult': {}, 'Shuffle': dict(red_from='A'),
           'Crop': dict(x=8, y=8, width=32, height=32)}


def evaluator_pixels(document, target):
    return Evaluator().evaluate(dict(document, view=target))


def tile_pixels(document, target):
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    assert executor.supports_tiled(document, target), target
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op='create', id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op='connect', id=key, input=slot, source=source))
        return key

    def bypass(self, key, value=True):
        self.d.execute(dict(op='disable', id=key, value=value))

    @property
    def doc(self):
        return self.d.document


class RuleTests(unittest.TestCase):
    def test_which_input_a_bypassed_node_passes(self):
        def node(kind, **wired):
            slots = SPECS[kind]['inputs'] + SPECS[kind].get('optional_inputs', [])
            return dict(type=kind, inputs={slot: wired.get(slot) for slot in slots})
        self.assertEqual(bypass_slot(node('Grade', image='x', mask='m')), 'image')
        # Merge passes B, the background, as in Nuke; A only when B is not wired.
        self.assertEqual(bypass_slot(node('Merge', A='fg', B='bg')), 'B')
        self.assertEqual(bypass_slot(node('Merge', B='bg')), 'B')
        self.assertEqual(bypass_slot(node('Merge', A='fg')), 'A')
        self.assertEqual(bypass_slot(node('Merge')), 'B')
        self.assertEqual(bypass_slot(node('Project3D', geometry='g')), 'geometry')
        self.assertEqual(bypass_slot(node('Switch', **{'1': 'x'})), '0')
        self.assertIsNone(bypass_slot(node('Checker')))


class PixelTests(unittest.TestCase):
    def test_every_bypassed_filter_equals_its_input_on_both_paths(self):
        kinds = [k for k in SPECS if OUTPUT_TYPES.get(k, 'image') == 'image'
                 and SPECS[k]['inputs'] == ['image']
                 # Relight's image input must carry a Render3D relight-bundle (Raster.layers), not
                 # an arbitrary plate, so it fails the generic "any image in" graph this test
                 # builds; test_3d_relight_node.test_disabled_passthrough covers its bypass instead.
                 and k not in ('Viewer', 'Write', 'Tracker', 'Relight')]
        self.assertGreaterEqual(len(kinds), 7, kinds)
        for kind in kinds:
            with self.subTest(kind=kind):
                g = Graph(); g.add('plate', 'Checker')
                g.add('half', 'Grade', dict(exposure=-1.0), image='plate')
                g.add('node', kind, VISIBLE.get(kind), image='half')
                upstream = evaluator_pixels(g.doc, 'half')
                enabled = evaluator_pixels(g.doc, 'node')
                if kind in VISIBLE and kind not in ('Premult', 'Unpremult'):
                    self.assertFalse(np.array_equal(enabled, upstream), 'the filter must do something')
                g.bypass('node')
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, 'node'), upstream))
                if kind in SUPPORTED_TILED_KINDS:
                    self.assertTrue(np.array_equal(tile_pixels(g.doc, 'node'), upstream))
                g.bypass('node', False)
                self.assertTrue(np.array_equal(evaluator_pixels(g.doc, 'node'), enabled))
                if kind in SUPPORTED_TILED_KINDS:
                    self.assertTrue(np.array_equal(tile_pixels(g.doc, 'node'), tile_pixels(g.doc, 'node')))
                    np.testing.assert_allclose(tile_pixels(g.doc, 'node'), enabled, atol=1e-6)

    def merge_graph(self):
        g = Graph()
        g.add('bg_plate', 'Checker'); g.add('bg', 'Grade', dict(exposure=1.0), image='bg_plate')
        g.add('fg_plate', 'Constant', dict(red=.8, green=.1, blue=.1, alpha=.5))
        g.add('fg', 'ColorCorrect', dict(saturation=.5), image='fg_plate')
        g.add('merge', 'Merge', A='fg', B='bg')
        return g

    def test_bypassed_merge_passes_b_and_never_touches_a(self):
        g = self.merge_graph()
        background, foreground = evaluator_pixels(g.doc, 'bg'), evaluator_pixels(g.doc, 'fg')
        merged = evaluator_pixels(g.doc, 'merge')
        self.assertFalse(np.array_equal(merged, background))
        g.bypass('merge')
        for label, pixels in (('evaluator', evaluator_pixels), ('tiles', tile_pixels)):
            with self.subTest(path=label):
                self.assertTrue(np.array_equal(pixels(g.doc, 'merge'), background))
        # A is not an ancestor of a bypassed merge: a broken A branch cannot break the picture.
        g.d.execute(dict(op='connect', id='fg', input='image', source=None))
        for label, pixels in (('evaluator', evaluator_pixels), ('tiles', tile_pixels)):
            with self.subTest(path=label, a='unwired upstream'):
                self.assertTrue(np.array_equal(pixels(g.doc, 'merge'), background))
        del foreground

    def test_bypassed_merge_with_only_a_wired_passes_a(self):
        g = self.merge_graph()
        foreground = evaluator_pixels(g.doc, 'fg')
        g.d.execute(dict(op='connect', id='merge', input='B', source=None))
        g.bypass('merge')
        self.assertTrue(np.array_equal(evaluator_pixels(g.doc, 'merge'), foreground))
        self.assertTrue(np.array_equal(tile_pixels(g.doc, 'merge'), foreground))

    def test_a_bypass_in_the_middle_of_a_chain_and_downstream_of_a_merge(self):
        g = self.merge_graph()
        g.add('after', 'Grade', dict(exposure=-.5), image='merge')
        g.add('blur', 'Blur', dict(radius=4.0), image='after')
        reference = Graph()
        reference.add('bg_plate', 'Checker'); reference.add('bg', 'Grade', dict(exposure=1.0), image='bg_plate')
        reference.add('after', 'Grade', dict(exposure=-.5), image='bg')
        reference.add('blur', 'Blur', dict(radius=4.0), image='after')
        expected = evaluator_pixels(reference.doc, 'blur')
        g.bypass('merge')
        np.testing.assert_allclose(evaluator_pixels(g.doc, 'blur'), expected, atol=1e-6)
        np.testing.assert_allclose(tile_pixels(g.doc, 'blur'), expected, atol=1e-6)

    def test_mask_is_ignored_when_bypassed(self):
        g = Graph(); g.add('plate', 'Checker'); g.add('matte', 'Constant', dict(alpha=.25))
        g.add('grade', 'Grade', dict(exposure=2.0), image='plate', mask='matte')
        plate = evaluator_pixels(g.doc, 'plate')
        g.bypass('grade')
        self.assertTrue(np.array_equal(evaluator_pixels(g.doc, 'grade'), plate))
        self.assertTrue(np.array_equal(tile_pixels(g.doc, 'grade'), plate))

    def test_toggling_reuses_cached_results(self):
        g = Graph(); g.add('plate', 'Checker'); g.add('blur', 'Blur', dict(radius=12.0), image='plate')
        g.add('grade', 'Grade', dict(exposure=1.0), image='blur')
        evaluator = Evaluator()
        evaluator.evaluate(dict(g.doc, view='grade'))
        for value in (True, False, True):
            g.bypass('grade', value)
            before = evaluator.misses
            evaluator.evaluate(dict(g.doc, view='grade'))
            # Off: only the bypassed node itself is new, and it costs no kernel. On: all cached.
            self.assertLessEqual(evaluator.misses - before, 1, value)


class WindowTests(unittest.TestCase):
    def setUp(self):
        from tests.test_desktop import APP, wait_until, WAIT_TIMEOUT
        from nodebased.app import Window
        self.app, self.wait_until = APP, wait_until
        self.window = Window(agent_name='nodebased-test-' + uuid.uuid4().hex)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f'no first frame cooked within {WAIT_TIMEOUT:.0f}s')

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.app.processEvents()

    def same(self, frame, expected, label):
        # The display cache keeps half floats, so a frame served from it differs from the first
        # cook by about 1e-4. The pictures compared here differ by whole stops.
        self.assertEqual(frame.shape, expected.shape, label)
        np.testing.assert_allclose(frame, expected, atol=5e-4, err_msg=label)

    def settle(self, command):
        window = self.window
        generation, start = window.generation, time.perf_counter()
        window.command(command)
        self.assertTrue(self.wait_until(
            lambda: window.frame_generation > generation and not window.busy))
        return time.perf_counter() - start

    def test_bypass_in_the_viewer_is_right_and_quick(self):
        window = self.window
        commands = [dict(op='create', id=key, type=kind, params=params) for key, kind, params in (
            ('b_bg', 'Checker', {}), ('b_grade', 'Grade', dict(exposure=2.0)),
            ('b_fg', 'Constant', dict(red=.8, green=.1, blue=.1, alpha=.5)), ('b_merge', 'Merge', {}))]
        commands += [dict(op='connect', id='b_grade', input='image', source='b_bg'),
                     dict(op='connect', id='b_merge', input='A', source='b_fg'),
                     dict(op='connect', id='b_merge', input='B', source='b_grade')]
        window.command(dict(op='batch', commands=commands), render=False)
        self.settle(dict(op='view', id='b_merge'))
        merged = np.array(window.frame)
        self.settle(dict(op='view', id='b_grade'))
        graded = np.array(window.frame)
        self.settle(dict(op='view', id='b_bg'))
        plain = np.array(window.frame)
        self.assertGreater(float(np.abs(graded - plain).max()), .1)
        self.assertGreater(float(np.abs(graded - merged).max()), .1)

        self.settle(dict(op='view', id='b_merge'))
        off = self.settle(dict(op='disable', id='b_merge', value=True))
        self.assertIsNotNone(window.frame, window.statusBar().currentMessage())
        self.same(window.frame, graded, 'a bypassed Merge shows B')
        self.assertEqual(list(window.render_errors), [])
        on = self.settle(dict(op='disable', id='b_merge', value=False))
        self.same(window.frame, merged, 're-enabled Merge')

        self.settle(dict(op='view', id='b_grade'))
        grade_off = self.settle(dict(op='disable', id='b_grade', value=True))
        self.same(window.frame, plain, 'a bypassed Grade shows its input')
        grade_on = self.settle(dict(op='disable', id='b_grade', value=False))
        self.same(window.frame, graded, 're-enabled Grade')
        # Everything upstream is cached, so a toggle is a lookup plus the display conversion.
        # The bound is loose for CI; the measured figures are in TASKLOG.md.
        for label, seconds in (('merge off', off), ('merge on', on), ('grade off', grade_off), ('grade on', grade_on)):
            self.assertLess(seconds, 1.5, label)
        print(f'\nbypass toggle latency: merge off {off*1000:.0f} ms, on {on*1000:.0f} ms; '
              f'grade off {grade_off*1000:.0f} ms, on {grade_on*1000:.0f} ms')


    def test_an_internal_failure_says_so_instead_of_printing_a_bare_name(self):
        from unittest.mock import patch
        window = self.window
        with patch.object(window.tile_executor, 'compose_region', side_effect=KeyError('grade')), \
                patch.object(window.evaluator, 'evaluate', side_effect=KeyError('grade')):
            window.display_cache.clear() if hasattr(window.display_cache, 'clear') else None
            self.settle(dict(op='set', id='grade', param='exposure', value=1.2345))
        self.assertIsNone(window.frame)
        message = window.statusBar().currentMessage()
        self.assertIn('Internal error', message)
        self.assertIn("KeyError: 'grade'", message)
        self.assertIn('NodeBased bug', message)


if __name__ == '__main__':
    unittest.main()
