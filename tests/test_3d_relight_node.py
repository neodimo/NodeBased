"""Relight graph contracts and the specified v1 recombination formula."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import scene3d
from nodebased.core import Dispatcher, INPUT_TYPES, LIMITS, OUTPUT_TYPES, SPECS
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.raster import Raster
from nodebased.tiers import Region


def graph():
    d = Dispatcher()
    for key, kind, params in (
        ('card', 'Card3D', dict(red=.3, green=.6, blue=.8, spec_amount=.5, spec_shininess=12)),
        ('camera', 'Camera3D', {}), ('scene', 'Scene3D', {}),
        ('l0', 'Light3D', dict(red=.8, green=.4, blue=.2, intensity=1.3)),
        ('l1', 'Light3D', dict(light_type='Point', tx=-2., ty=1., tz=4.,
                             red=.2, green=.5, blue=.9, intensity=.7)),
        ('render', 'Render3D', dict(width=24, height=20, samples=1,
                                  render_output='relight', ambient=.13)),
        ('relight', 'Relight', dict(red=.13, green=.13, blue=.13))):
        d.execute(dict(op='create', id=key, type=kind, params=params))
    for target, slot, source in (
        ('scene', 'object0', 'card'), ('scene', 'object1', 'l0'), ('scene', 'object2', 'l1'),
        ('render', 'scene', 'scene'), ('render', 'camera', 'camera'),
        ('relight', 'image', 'render'), ('relight', 'camera', 'camera'),
        ('relight', 'light0', 'l0'), ('relight', 'light1', 'l1')):
        d.execute(dict(op='connect', id=target, input=slot, source=source))
    return d


def expected(beauty, layers, lights, p):
    # docs/3D_ROADMAP.md "Design: relight passes": Diffuse/Specular scale only the per-light
    # response sums, not the ambient term, so either knob at 0 turns off that light kind without
    # also killing the ambient fill.
    ambient = np.zeros_like(beauty[..., :3]) + [p['red'], p['green'], p['blue']]
    diffuse_sum = np.zeros_like(ambient)
    specular = np.zeros_like(ambient)
    for i, light in enumerate(lights):
        if light is None or light.intensity <= 0:
            continue
        colour = np.asarray(light.color) * light.intensity
        if f'diffuse_L{i}' in layers:
            diffuse_sum += layers[f'diffuse_L{i}'][..., :3] * colour
        if f'specular_L{i}' in layers:
            specular += layers[f'specular_L{i}'][..., :3] * colour
    rgb = layers['albedo'][..., :3] * (ambient + diffuse_sum * p['diffuse']) + specular * p['specular']
    return np.concatenate((p['mix'] * rgb + (1 - p['mix']) * beauty[..., :3],
                           beauty[..., 3:4]), axis=-1)


class RelightNodeTests(unittest.TestCase):
    def setUp(self):
        self.d, self.e = graph(), Evaluator()

    def set_param(self, key, param, value):
        self.d.execute(dict(op='set', id=key, param=param, value=value))

    def connect(self, slot, source):
        self.d.execute(dict(op='connect', id='relight', input=slot, source=source))

    def evaluate(self):
        return self.e.evaluate_raster(self.d.document, 'relight')

    def reference(self):
        scene = self.e.evaluate_raster(self.d.document, 'scene', typed=True)
        camera = self.e.evaluate_raster(self.d.document, 'camera', typed=True)
        beauty, layers = scene3d.render(scene, camera, 24, 20, output='relight', ambient=.13)
        return beauty, layers, scene.lights

    def test_graph_formula_and_cache(self):
        for name, value in dict(red=.21, green=.37, blue=.09, diffuse=.4, specular=.65, mix=.73).items():
            self.set_param('relight', name, value)
        beauty, layers, lights = self.reference()
        result = self.evaluate()
        np.testing.assert_allclose(result.pixels, expected(beauty, layers, lights,
            self.d.document['nodes']['relight']['params']), atol=1e-6, rtol=0)
        self.assertIsNone(result.layers)
        self.assertIs(self.evaluate(), result)
        # An independent relighting light can change without rendering the scene again.
        self.d.execute(dict(op='create', id='newlight', type='Light3D', params=dict(intensity=2.)))
        self.connect('light0', 'newlight')
        bundle = self.e.evaluate_raster(self.d.document, 'render')
        with patch('nodebased.scene3d.render', side_effect=AssertionError('rerendered')):
            changed = self.evaluate()
        self.assertFalse(np.array_equal(changed.pixels, result.pixels))
        self.assertIs(self.e.evaluate_raster(self.d.document, 'render'), bundle)

    def test_mix_zero_exact_and_opaque_render_parity(self):
        beauty, layers, _ = self.reference()
        # Opaque surfaces: premultiplied response/albedo products reproduce diffuse.
        result = self.evaluate()
        np.testing.assert_allclose(result.pixels[..., :3],
            layers['diffuse'][..., :3] + layers['specular'][..., :3], atol=1e-5, rtol=0)
        np.testing.assert_allclose(result.pixels, beauty, atol=1e-5, rtol=0)
        self.set_param('relight', 'mix', 0.)
        self.assertEqual(self.evaluate().pixels.tobytes(), beauty.tobytes())
        self.assertIsNone(self.evaluate().layers)

    def test_emission_is_excluded(self):
        self.set_param('card', 'emission', .4)
        beauty, layers, _ = self.reference()
        np.testing.assert_allclose(self.evaluate().pixels[..., :3],
            beauty[..., :3] - layers['emission'][..., :3], atol=1e-5, rtol=0)
        self.assertGreater(float(layers['emission'][..., :3].max()), 0)

    def test_unwired_lights_and_diffuse_does_not_scale_ambient(self):
        self.connect('light0', None)
        self.connect('light1', None)
        self.connect('camera', None)
        self.set_param('relight', 'diffuse', .4)
        beauty, layers, _ = self.reference()
        # No lights contribute, so the Diffuse knob (which scales only the per-light sum, not
        # ambient) has no effect here: albedo * ambient, unscaled.
        np.testing.assert_allclose(self.evaluate().pixels[..., :3],
            layers['albedo'][..., :3] * .13, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(self.evaluate().pixels[..., 3], beauty[..., 3])

    def test_extra_light_and_sparse_slot_cache(self):
        baseline = self.evaluate().pixels.copy()
        self.d.execute(dict(op='create', id='extra', type='Light3D', params=dict(intensity=999.)))
        self.connect('light2', 'extra')
        np.testing.assert_array_equal(self.evaluate().pixels, baseline)
        self.connect('light2', None)
        self.connect('light1', None)
        first = self.evaluate().pixels.copy()
        # Same ordered source hashes, different light index: must miss the old cache.
        self.connect('light0', None)
        self.connect('light1', 'l0')
        second = self.evaluate().pixels
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(second, Evaluator().evaluate_raster(self.d.document, 'relight').pixels)

    def test_disabled_and_zero_intensity_lights(self):
        self.d.execute(dict(op='create', id='off', type='Light3D', params=dict(intensity=0.)))
        self.connect('light0', 'off')
        zero = self.evaluate().pixels.copy()
        self.set_param('off', 'intensity', 3.)
        # Dispatcher disallows bypassing source nodes; exercise the evaluator's
        # existing disabled-Light3D -> None convention with a document snapshot.
        disabled = copy.deepcopy(self.d.document)
        disabled['nodes']['off']['disabled'] = True
        np.testing.assert_array_equal(self.e.evaluate_raster(disabled, 'relight').pixels, zero)
        self.connect('light0', None)
        np.testing.assert_array_equal(self.evaluate().pixels, zero)

    def test_missing_bundle_and_albedo(self):
        self.set_param('render', 'render_output', 'rgba')
        with self.assertRaisesRegex(ValueError,
                "^Relight: connect a Render3D node with Output set to 'Relight passes'$"):
            self.evaluate()
        malformed = Raster(np.zeros((2, 3, 4), np.float32), layers={})
        with patch('nodebased.imaging.read_image_raster', return_value=malformed):
            self.d.execute(dict(op='create', id='read', type='Read'))
            self.connect('image', 'read')
            with self.assertRaisesRegex(ValueError, "^Relight: input bundle is missing the 'albedo' layer$"):
                self.evaluate()

    def test_disabled_passthrough(self):
        bundle = self.e.evaluate_raster(self.d.document, 'render')
        self.d.execute(dict(op='disable', id='relight', value=True))
        self.assertIs(self.evaluate(), bundle)
        # Bypass also accepts an ordinary image without inspecting layers.
        self.set_param('render', 'render_output', 'rgba')
        self.assertIs(self.evaluate(), self.e.evaluate_raster(self.d.document, 'render'))

    def test_windows_minimal_bundle_and_translucent_literal_formula(self):
        pixels = np.full((2, 3, 4), .5, np.float32)
        source = Raster(pixels, Region(-2, -1, 3, 2), Region(0, 0, 8, 6),
                        layers={'albedo': Raster.of(pixels), 'diffuse_L0': Raster.of(pixels)})
        light = scene3d.Light()
        params = SPECS['Relight']['params']
        result = Evaluator._windowed_kernel('Relight', params, [source, None, light])
        np.testing.assert_allclose(result.pixels, expected(pixels,
            {k: v.pixels for k, v in source.layers.items()}, [light], params), atol=1e-7)
        self.assertEqual(result.data, source.data)
        self.assertEqual(result.display, source.display)
        self.assertIsNone(result.layers)
        # No normals/position/specular channels are required; input arrays stay untouched.
        np.testing.assert_array_equal(source.pixels, .5)

    def test_schema_knobs_and_atomic_typed_rejection(self):
        self.assertEqual(SPECS['Relight']['params'], dict(red=.8, green=.8, blue=.8,
                                                        diffuse=1., specular=1., mix=1.))
        self.assertEqual(SPECS['Relight']['optional_inputs'], ['camera'] + [f'light{i}' for i in range(8)])
        self.assertEqual(OUTPUT_TYPES['Relight'], 'image')
        for i in range(8):
            self.assertEqual(INPUT_TYPES[f'light{i}'], ('light',))
            self.assertEqual(INPUT_TYPES[f'object{i}'], ('geometry', 'light', 'scene'))
        for name in ('diffuse', 'specular', 'mix'):
            self.assertEqual(LIMITS[name], (0, 1))
        groups = knob_layout('Relight')
        self.assertEqual([(g.kind, g.params, g.label, g.soft_range) for g in groups], [
            ('color', ('red', 'green', 'blue'), 'Ambient', None),
            ('float_slider', ('diffuse',), 'Diffuse', (0, 1)),
            ('float_slider', ('specular',), 'Specular', (0, 1)),
            ('float_slider', ('mix',), 'Mix', (0, 1))])
        for slot, source, message in [('image', 'camera', 'expects image'),
                                      ('light0', 'render', 'expects light')]:
            before, undo = copy.deepcopy(self.d.document), copy.deepcopy(self.d.undo_stack)
            with self.assertRaisesRegex(ValueError, message):
                self.connect(slot, source)
            self.assertEqual(self.d.document, before)
            self.assertEqual(self.d.undo_stack, undo)

    def test_create_connect_undo_redo(self):
        before = copy.deepcopy(self.d.document)
        self.d.execute(dict(op='create', id='new', type='Relight'))
        created = copy.deepcopy(self.d.document)
        self.d.execute(dict(op='connect', id='new', input='image', source='render'))
        connected = copy.deepcopy(self.d.document)
        for document in (created, before):
            self.d.execute(dict(op='undo'))
            self.assertEqual(self.d.document, document)
        for document in (created, connected):
            self.d.execute(dict(op='redo'))
            self.assertEqual(self.d.document, document)
        result = self.e.evaluate_raster(self.d.document, 'new')
        self.assertIsNone(result.layers)


if __name__ == '__main__':
    unittest.main()
