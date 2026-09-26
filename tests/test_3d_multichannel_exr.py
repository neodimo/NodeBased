"""Multichannel EXR out of Render3D: named passes, the Write node's one-part EXR, and Read-back.

Layer naming follows Nuke's layer.channel scheme (normals.X, depth.Z, relight_light1_diffuse.R).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import OpenImageIO as oiio

from nodebased import scene3d as s
from nodebased.core import CHOICES, Dispatcher, load_document, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.media import read_media_raster, write_exr, raster_layer_arrays
from tests.test_3d_aovs import scenes
from tests.test_desktop import APP, wait_until
from nodebased.app import Window

BEAUTY_CHANNELS = {'R', 'G', 'B', 'A'}
HALF_STEP = 2.0 ** -10   # relative rounding step of a half float


def channel_names(path):
    source = oiio.ImageInput.open(str(path))
    try:
        return list(source.spec().channelnames), str(source.spec().format), source.spec().nchannels
    finally:
        source.close()


def graph(**render):
    d = Dispatcher()
    for key, kind, params in (('card', 'Card3D', dict(spec_amount=.5, emission=.2, alpha=.5)),
                              ('cube', 'Cube3D', {}),
                              ('light', 'Light3D', dict(shadows='on')), ('light2', 'Light3D', dict(light_type='Point')),
                              ('camera', 'Camera3D', {}), ('scene', 'Scene3D', {}),
                              ('render', 'Render3D', dict(width=32, height=24, samples=1,
                                                          render_output='multichannel', **render)),
                              ('write', 'Write', dict(bit_depth='float'))):
        d.execute(dict(op='create', id=key, type=kind, params=params))
    for slot, source in (('object0', 'card'), ('object1', 'cube'), ('object2', 'light'), ('object3', 'light2')):
        d.execute(dict(op='connect', id='scene', input=slot, source=source))
    for slot in ('scene', 'camera'):
        d.execute(dict(op='connect', id='render', input=slot, source=slot))
    d.execute(dict(op='connect', id='write', input='image', source='render'))
    return d


class MultichannelRenderTests(unittest.TestCase):
    def test_passes_parse_and_reject_unknown_names(self):
        self.assertEqual(s.parse_passes('depth, Beauty ,depth'), ('beauty', 'depth'))
        self.assertEqual(s.parse_passes(s.DEFAULT_PASSES), ('beauty', 'normals', 'depth'))
        with self.assertRaisesRegex(ValueError, "Unknown Render3D pass 'albedo'.*beauty, normals, depth, relight"):
            s.parse_passes('beauty,albedo')
        with self.assertRaisesRegex(ValueError, 'at least one pass'):
            s.render_multichannel(scenes()[0], s.Camera(), 8, 8, passes=' ')

    def test_every_layer_equals_the_single_purpose_output(self):
        scene = scenes()[0]
        beauty, layers = s.render_multichannel(scene, s.Camera(), 40, 30, passes='beauty,normals,depth,relight',
                                               ambient=.13, samples=2)
        self.assertEqual(list(layers), ['normals', 'depth', 'relight_light1_diffuse', 'relight_light1_specular',
                                        'relight_light2_diffuse', 'relight_light2_specular'])
        np.testing.assert_array_equal(beauty, s.render(scene, s.Camera(), 40, 30, ambient=.13, samples=2))
        np.testing.assert_array_equal(layers['normals'], s.render(scene, s.Camera(), 40, 30, output='normals'))
        np.testing.assert_array_equal(layers['depth'], s.render(scene, s.Camera(), 40, 30, output='depth'))
        _, bundle = s.render(scene, s.Camera(), 40, 30, ambient=.13, output='relight')
        for i in range(2):
            for kind in ('diffuse', 'specular'):
                np.testing.assert_array_equal(layers[f'relight_light{i + 1}_{kind}'], bundle[f'{kind}_L{i}'])
        self.assertGreater(float(layers['relight_light1_diffuse'][..., :3].max()), 0)

    def test_beauty_is_optional_and_unlit_scenes_have_no_light_layers(self):
        beauty, layers = s.render_multichannel(scenes()[2], s.Camera(), 16, 12, passes='depth,relight')
        np.testing.assert_array_equal(beauty, 0)
        self.assertEqual(list(layers), ['depth'])

    def test_relight_pass_keeps_the_bundle_limits(self):
        with self.assertRaisesRegex(ValueError, 'raster-only'):
            s.render_multichannel(scenes()[0], s.Camera(), 8, 8, passes='relight', mode='raytrace')


class MultichannelGraphTests(unittest.TestCase):
    def test_choice_knob_and_defaults(self):
        self.assertEqual(CHOICES['render_output'][-1], 'multichannel')
        self.assertEqual(CHOICES['render_output'][:-1], list(s.RENDER_OUTPUTS))
        self.assertTrue(any('passes' in g.params for g in knob_layout('Render3D')))
        d = Dispatcher()
        d.execute(dict(op='create', id='r', type='Render3D'))
        self.assertEqual(d.document['nodes']['r']['params']['passes'], 'beauty,normals,depth')

    def test_render3d_raster_carries_named_layers_and_is_cached(self):
        d, evaluator = graph(passes='beauty,normals,depth,relight'), Evaluator()
        with patch.object(s, 'render_multichannel', wraps=s.render_multichannel) as call:
            raster = evaluator.evaluate_raster(d.document, 'render')
            again = evaluator.evaluate_raster(d.document, 'render')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(list(raster.layers), ['normals', 'depth', 'relight_light1_diffuse', 'relight_light1_specular',
                                               'relight_light2_diffuse', 'relight_light2_specular'])
        self.assertIs(raster.layers, again.layers)
        self.assertGreater(raster.nbytes, raster.pixels.nbytes)
        d.execute(dict(op='set', id='render', param='passes', value='beauty'))
        self.assertEqual(list(evaluator.evaluate_raster(d.document, 'render').layers), [])

    def test_write_and_bypassed_write_hand_the_layers_on(self):
        d, evaluator = graph(), Evaluator()
        through = evaluator.evaluate_raster(d.document, 'write')
        self.assertEqual(list(through.layers), ['normals', 'depth'])
        d.execute(dict(op='disable', id='write', value=True))
        bypassed = evaluator.evaluate_raster(d.document, 'write')
        self.assertEqual(list(bypassed.layers), ['normals', 'depth'])
        np.testing.assert_array_equal(bypassed.pixels, through.pixels)

    def test_gpu_backend_is_refused_and_disabled_render_is_black(self):
        d, evaluator = graph(render_backend='gpu'), Evaluator()
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            evaluator.evaluate_raster(d.document, 'render')
        d.execute(dict(op='set', id='render', param='render_backend', value='cpu'))
        d.execute(dict(op='disable', id='render', value=True))
        np.testing.assert_array_equal(evaluator.evaluate(d.document, 'render'), 0)

    def test_old_documents_gain_the_passes_knob(self):
        d = graph()
        old = json.loads(json.dumps(d.document))
        del old['nodes']['render']['params']['passes']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.json'
            path.write_text(json.dumps(old))
            loaded = load_document(path)
        self.assertEqual(loaded['nodes']['render']['params']['passes'], 'beauty,normals,depth')
        self.assertEqual(upgrade_document(old)['nodes']['render']['params'].get('passes', 'beauty,normals,depth'),
                         'beauty,normals,depth')


class MultichannelFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_layers_write_as_nuke_named_channels_in_one_part(self):
        frame = np.random.default_rng(1).random((6, 8, 4), dtype=np.float32)
        layers = {'normals': frame[..., ::-1] * 2 - 1, 'depth': frame * 5, 'relight_light1_diffuse': frame * .5}
        path = self.dir / 'layers.exr'
        write_exr(path, frame, bits='float', layers=layers)
        names, fmt, count = channel_names(path)
        self.assertEqual(sorted(names), sorted(['R', 'G', 'B', 'A', 'normals.X', 'normals.Y', 'normals.Z', 'depth.Z',
                                                'relight_light1_diffuse.R', 'relight_light1_diffuse.G',
                                                'relight_light1_diffuse.B']))
        self.assertEqual((count, fmt), (11, 'float'))
        source = oiio.ImageInput.open(str(path))
        try:
            self.assertFalse(source.seek_subimage(1, 0), 'all layers live in a single part')
        finally:
            source.close()

    def test_bad_layers_are_rejected(self):
        frame = np.zeros((4, 4, 4), np.float32)
        for name, layer, message in (('R', frame, 'Cannot write a layer'), ('bad name', frame, 'Cannot write a layer'),
                                     ('depth', np.zeros((3, 4, 4), np.float32), 'not the frame size')):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                write_exr(self.dir / 'bad.exr', frame, layers={name: layer})
        self.assertEqual(list(self.dir.iterdir()), [])

    def render_document(self, window, target, bits, passes='beauty,normals,depth'):
        window.command({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'writer3d', 'type': 'Write'},
            {'op': 'connect', 'id': 'writer3d', 'input': 'image', 'source': 'render'},
            {'op': 'set', 'id': 'writer3d', 'param': 'path', 'value': str(target)},
            {'op': 'set', 'id': 'writer3d', 'param': 'bit_depth', 'value': bits}]})
        window.render_write('writer3d', single=True)

    def window_with_render(self, passes='beauty,normals,depth'):
        window = Window()
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None))
        self.addCleanup(lambda: (setattr(window, 'saved_document', window.dispatcher.document), window.close(),
                                 APP.processEvents()))
        commands = []
        for key, kind, params in (('card', 'Card3D', dict(spec_amount=.5, emission=.2, alpha=.5)),
                                  ('cube', 'Cube3D', {}), ('light', 'Light3D', dict(shadows='on')),
                                  ('camera', 'Camera3D', {}), ('scene', 'Scene3D', {}),
                                  ('render', 'Render3D', dict(width=32, height=24, samples=1,
                                                              render_output='multichannel', passes=passes))):
            commands.append({'op': 'create', 'id': key, 'type': kind, 'params': params})
        for slot, source in (('object0', 'card'), ('object1', 'cube'), ('object2', 'light')):
            commands.append({'op': 'connect', 'id': 'scene', 'input': slot, 'source': source})
        commands += [{'op': 'connect', 'id': 'render', 'input': 'scene', 'source': 'scene'},
                     {'op': 'connect', 'id': 'render', 'input': 'camera', 'source': 'camera'}]
        window.command({'op': 'batch', 'commands': commands})
        return window

    def test_render3d_write_reads_back_with_exactly_the_expected_channels(self):
        window = self.window_with_render()
        target = self.dir / 'mc.exr'
        self.render_document(window, target, 'float')
        names, fmt, count = channel_names(target)
        self.assertEqual(sorted(names), sorted(['R', 'G', 'B', 'A', 'normals.X', 'normals.Y', 'normals.Z', 'depth.Z']))
        self.assertEqual((count, fmt), (8, 'float'))
        raster = window.evaluator.evaluate_raster(window.dispatcher.document, 'render')
        back = read_media_raster(str(target))
        np.testing.assert_array_equal(back.pixels, raster.pixels)   # float: exact
        self.assertEqual(sorted(back.layers), ['depth', 'normals'])
        np.testing.assert_array_equal(back.layers['normals'].pixels[..., :3], raster.layers['normals'].pixels[..., :3])
        np.testing.assert_array_equal(back.layers['depth'].pixels[..., 0], raster.layers['depth'].pixels[..., 0])
        self.assertGreater(float(back.layers['depth'].pixels[..., 0].max()), 1.0)
        self.assertGreater(float(np.abs(back.layers['normals'].pixels[..., :3]).max()), 0.5)

    def test_half_write_round_trips_within_half_float_tolerance(self):
        window = self.window_with_render('beauty,normals,depth,relight')
        target = self.dir / 'half.exr'
        self.render_document(window, target, 'half')
        names, fmt, count = channel_names(target)
        self.assertEqual(fmt, 'half')
        expected = {'R', 'G', 'B', 'A', 'normals.X', 'normals.Y', 'normals.Z', 'depth.Z'}
        for kind in ('diffuse', 'specular'):
            expected |= {f'relight_light1_{kind}.{c}' for c in 'RGB'}
        self.assertEqual(set(names), expected)
        self.assertEqual(count, len(expected))
        raster = window.evaluator.evaluate_raster(window.dispatcher.document, 'render')
        back = read_media_raster(str(target))
        np.testing.assert_allclose(back.pixels, raster.pixels, rtol=HALF_STEP, atol=1e-6)
        self.assertEqual(sorted(back.layers), sorted(raster.layers))
        for name, layer in raster.layers.items():
            channels = 1 if name == 'depth' else 3
            np.testing.assert_allclose(back.layers[name].pixels[..., :channels], layer.pixels[..., :channels],
                                       rtol=HALF_STEP, atol=1e-6, err_msg=name)

    def test_png_write_keeps_only_the_beauty(self):
        window = self.window_with_render()
        target = self.dir / 'mc.png'
        self.render_document(window, target, 'half')
        source = oiio.ImageInput.open(str(target))
        try:
            self.assertEqual((source.spec().nchannels, list(source.spec().channelnames)), (4, ['R', 'G', 'B', 'A']))
            self.assertFalse(source.seek_subimage(1, 0))
        finally:
            source.close()

    def test_plain_input_still_writes_plain_rgba_exr(self):
        window = self.window_with_render()
        window.command({'op': 'set', 'id': 'render', 'param': 'render_output', 'value': 'rgba'})
        target = self.dir / 'plain.exr'
        self.render_document(window, target, 'half')
        self.assertEqual(channel_names(target)[0], ['R', 'G', 'B', 'A'])
        self.assertIsNone(read_media_raster(str(target)).layers)

    def test_read_hands_the_layers_to_a_downstream_write_unchanged(self):
        window = self.window_with_render()
        first = self.dir / 'first.exr'
        self.render_document(window, first, 'float')
        d = Dispatcher()
        d.execute(dict(op='create', id='read', type='Read', params=dict(path=str(first))))
        raster = Evaluator().evaluate_raster(d.document, 'read')
        self.assertEqual(sorted(raster.layers), ['depth', 'normals'])
        # The pass-aware code sees the layers; ordinary 2D nodes work on the beauty and drop them.
        d.execute(dict(op='create', id='grade', type='Grade'))
        d.execute(dict(op='connect', id='grade', input='image', source='read'))
        self.assertIsNone(Evaluator().evaluate_raster(d.document, 'grade').layers)
        second = self.dir / 'second.exr'
        write_exr(second, raster.to_display(), bits='float', layers=raster_layer_arrays(raster))
        self.assertEqual(sorted(channel_names(second)[0]), sorted(channel_names(first)[0]))
        again = read_media_raster(str(second))
        for name in ('normals', 'depth'):
            np.testing.assert_array_equal(again.layers[name].pixels, raster.layers[name].pixels)
        # A named layer can still be picked as the image, as before.
        picked = read_media_raster(str(first), layer='normals')
        np.testing.assert_array_equal(picked.pixels[..., :3], raster.layers['normals'].pixels[..., :3])
        self.assertIsNone(picked.layers)


if __name__ == '__main__':
    unittest.main()
