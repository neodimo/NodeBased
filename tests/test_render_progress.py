"""Progress and time left inside one slow frame: the routing, the wording, and the desktop window."""
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

from nodebased import splatraster, splats
from nodebased.renderprogress import ThreadProgress, format_duration, progress_text
from tests.test_3d_splat_render import cloud


class WordingTests(unittest.TestCase):
    def test_durations_round_to_what_a_person_would_say(self):
        cases = {0: '1 s', 0.4: '1 s', 7.4: '7 s', 12: '10 s', 43: '45 s', 88: '90 s',
                 95: '1 min 40 s', 119: '2 min', 125: '2 min 10 s', 3600: '60 min'}
        for seconds, text in cases.items():
            self.assertEqual(format_duration(seconds), text, seconds)

    def test_text_for_each_stage(self):
        self.assertEqual(progress_text('prepare', 0.0, {}), 'Preparing splats…')
        self.assertIsNone(progress_text('done', 1.0, {'eta_seconds': 0.0}))
        self.assertIsNone(progress_text('anything else', .5, {}))
        # Early on, the reference-rate total; the word "roughly" marks it as the weaker figure.
        self.assertEqual(progress_text('splats', 0.0, {'estimate_seconds': 142.0}),
                         'Rendering splats  ·  0%  ·  roughly 2 min 20 s in total')
        self.assertEqual(progress_text('splats', .03, {'estimate_seconds': 142.0, 'eta_seconds': 119.6}),
                         'Rendering splats  ·  3%  ·  roughly 2 min 20 s in total')
        self.assertEqual(progress_text('splats', .5, {'estimate_seconds': 142.0, 'eta_seconds': 42.3}),
                         'Rendering splats  ·  50%  ·  about 40 s left')
        # A frame that takes no time to speak of gets a percentage and nothing else.
        self.assertEqual(progress_text('splats', 0.0, {'estimate_seconds': .3}), 'Rendering splats  ·  0%')
        self.assertEqual(progress_text('splats', 1.7, {}), 'Rendering splats  ·  100%')


class RoutingTests(unittest.TestCase):
    def test_each_thread_sees_only_its_own_handler(self):
        router, seen = ThreadProgress(), {'main': [], 'worker': []}
        router('splats', .1, {})  # no handler: silent, not an error
        ready, release = threading.Event(), threading.Event()

        def worker():
            with router.handler(lambda *event: seen['worker'].append(event)):
                router('splats', .2, {'who': 'worker'})
                ready.set(); release.wait(5)
                router('splats', .4, {'who': 'worker'})

        thread = threading.Thread(target=worker); thread.start()
        self.assertTrue(ready.wait(5))
        with router.handler(lambda *event: seen['main'].append(event)):
            router('splats', .3, {'who': 'main'})
            with router.handler(None):
                router('splats', .35, {'who': 'nobody'})
            router('splats', .5, {'who': 'main'})
        release.set(); thread.join(5)
        router('splats', .9, {'who': 'nobody'})
        self.assertEqual([e[1] for e in seen['main']], [.3, .5])
        self.assertEqual([e[1] for e in seen['worker']], [.2, .4])

    def test_handler_is_removed_when_the_block_raises(self):
        router, seen = ThreadProgress(), []
        with self.assertRaises(RuntimeError):
            with router.handler(lambda *event: seen.append(event)):
                raise RuntimeError('render failed')
        router('splats', .5, {})
        self.assertEqual(seen, [])


class WindowTests(unittest.TestCase):
    def setUp(self):
        from tests.test_desktop import APP, wait_until, WAIT_TIMEOUT
        from nodebased.app import Window
        self.app, self.wait_until = APP, wait_until
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = self.tmp.name + '/cloud.ply'
        splats.write_ply(cloud(400), self.path)
        self.window = Window(agent_name='nodebased-test-' + uuid.uuid4().hex)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f'no first frame cooked within {WAIT_TIMEOUT:.0f}s')

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.app.processEvents()

    def build(self):
        commands = [dict(op='create', id=key, type=kind, params=params) for key, kind, params in (
            ('p_read', 'ReadSplat3D', dict(splat_path=self.path, splat_colorspace='linear')),
            ('p_scene', 'Scene3D', {}), ('p_camera', 'Camera3D', {}),
            ('p_render', 'Render3D', dict(width=96, height=64, samples=1, render_backend='cpu')))]
        commands += [dict(op='connect', id='p_scene', input='object0', source='p_read'),
                     dict(op='connect', id='p_render', input='scene', source='p_scene'),
                     dict(op='connect', id='p_render', input='camera', source='p_camera')]
        self.window.command(dict(op='batch', commands=commands), render=False)

    def test_a_frame_over_the_budget_renders_with_progress_instead_of_being_refused(self):
        window, events, shown = self.window, [], []
        window.signals.progress.connect(lambda payload, stage, fraction, info: events.append((stage, fraction)))
        original = window._show_render_progress

        def spy(stage, fraction, info):
            original(stage, fraction, info)
            shown.append((stage, window.render_progress.isVisible(), window.statusBar().currentMessage()))

        window._show_render_progress = spy
        self.build()
        # One tile evaluation of budget: every splat frame is "too big". A render without a
        # progress callback is refused at this budget (tests/test_3d_splat_progress.py).
        with patch.object(splatraster, 'SPLAT_WORK_BUDGET', 1):
            generation = window.generation
            window.command(dict(op='view', id='p_render'))
            self.assertTrue(self.wait_until(
                lambda: window.frame_generation > generation and not window.busy))
        self.assertIsNotNone(window.frame, window.statusBar().currentMessage())
        self.assertEqual(window.frame.shape[:2], (64, 96))
        self.assertGreater(float(window.frame[..., 3].max()), 0)
        self.assertEqual(events[0], ('prepare', 0.0))
        self.assertEqual(events[-1], ('done', 1.0))
        self.assertIn('splats', [stage for stage, _ in events])
        self.assertTrue(any(stage == 'splats' and visible and text.startswith('Rendering splats')
                            for stage, visible, text in shown), shown)
        self.assertFalse(window.render_progress.isVisible())
        self.assertNotIn('Rendering splats', window.statusBar().currentMessage())

    def test_a_superseded_request_cannot_move_the_bar(self):
        window = self.window
        stale = (type('Request', (), dict(generation=window.generation - 1, display=True))(), threading.Event())
        window.preview_progress(stale, 'splats', .5, {'eta_seconds': 30.0})
        self.assertFalse(window.render_progress.isVisible())
        cancelled = (type('Request', (), dict(generation=window.generation + 1, display=True))(), threading.Event())
        cancelled[1].set()
        window.preview_progress(cancelled, 'splats', .5, {'eta_seconds': 30.0})
        self.assertFalse(window.render_progress.isVisible())
        live = (type('Request', (), dict(generation=window.generation + 1, display=True))(), threading.Event())
        window.generation += 1
        window.preview_progress(live, 'splats', .5, {'eta_seconds': 30.0})
        self.assertTrue(window.render_progress.isVisible())
        self.assertEqual(window.render_progress.value(), 500)
        self.assertEqual(window.statusBar().currentMessage(), 'Rendering splats  ·  50%  ·  about 30 s left')
        window.preview_progress(live, 'done', 1.0, {})
        self.assertFalse(window.render_progress.isVisible())


if __name__ == '__main__':
    unittest.main()
