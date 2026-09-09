"""EXR/OCIO ingest and output. Every check runs real OpenImageIO and OpenColorIO."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import OpenImageIO as oiio
from nodebased.color import display_rgb, to_working
from nodebased.core import Dispatcher, upgrade_document, validate
from nodebased.imaging import Evaluator, srgb_to_linear
from nodebased.media import read_media, write_exr


def write_exr_raw(path, pixels, channelnames, x=0, y=0, full=None, attributes=()):
    """Author an EXR directly so the reader is tested against foreign files."""
    height, width = pixels.shape[:2]
    spec = oiio.ImageSpec(width, height, pixels.shape[2], oiio.FLOAT)
    spec.channelnames = list(channelnames)
    spec.x, spec.y = x, y
    spec.full_x, spec.full_y = 0, 0
    spec.full_width, spec.full_height = full or (width, height)
    for name, value in attributes:
        spec.attribute(name, value)
    output = oiio.ImageOutput.create(str(path))
    assert output and output.open(str(path), spec), oiio.geterror()
    assert output.write_image(np.ascontiguousarray(pixels, np.float32)), output.geterror()
    assert output.close()


class ColorTests(unittest.TestCase):
    def test_srgb_decode_matches_transfer_function(self):
        encoded = np.array([[[0.0, 0.25, 0.5, 1.0], [1.0, 0.75, 0.05, 1.0]]], np.float32)
        linear = to_working(encoded, 'sRGB')
        np.testing.assert_allclose(linear[..., :3], srgb_to_linear(encoded[..., :3]), atol=2e-3)
        np.testing.assert_array_equal(linear[..., 3], encoded[..., 3])

    def test_linear_and_raw_skip_the_transform_and_never_mutate_the_input(self):
        source = np.array([[[4.5, -0.3, 0.2, 1.0]]], np.float32)
        original = source.copy()
        for space in ('Linear Rec.709', 'Raw'):
            np.testing.assert_array_equal(to_working(source, space), original)
        np.testing.assert_array_equal(source, original)

    def test_straight_alpha_input_is_premultiplied_into_the_working_format(self):
        straight = np.array([[[4.0, -0.4, 0.2, 0.25]]], np.float32)
        np.testing.assert_allclose(to_working(straight, 'Linear Rec.709'), [[[1.0, -0.1, 0.05, 0.25]]], atol=1e-6)

    def test_acescg_primaries_convert_to_rec709(self):
        acescg = np.array([[[1.0, 0.0, 0.0, 1.0]]], np.float32)
        result = to_working(acescg, 'ACEScg')
        self.assertGreater(result[0, 0, 0], 1.0)  # wider gamut red maps outside Rec.709
        self.assertLess(result[0, 0, 1], 0.0)
        self.assertLess(result[0, 0, 2], 0.05)

    def test_associated_alpha_is_unpremultiplied_then_restored(self):
        # Encoded value 0.5 at alpha 0.5: transform must see 0.5, not 0.25.
        premultiplied = np.array([[[0.25, 0.25, 0.25, 0.5], [0.0, 0.0, 0.0, 0.0]]], np.float32)
        result = to_working(premultiplied, 'sRGB', associated=True)
        expected = srgb_to_linear(np.float32(0.5)) * 0.5
        np.testing.assert_allclose(result[0, 0, :3], [expected] * 3, atol=2e-3)
        np.testing.assert_array_equal(result[0, 1], [0, 0, 0, 0])  # zero alpha stays finite

    def test_unknown_spaces_and_views_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown input color space'):
            to_working(np.zeros((1, 1, 4), np.float32), 'Rec.2020')
        with self.assertRaisesRegex(ValueError, 'Unknown display view'):
            display_rgb(np.zeros((1, 1, 3), np.float32), 'Filmic')

    def test_display_views_differ_and_linear_is_a_passthrough(self):
        linear = np.array([[[0.18, 0.5, 4.0]]], np.float32)
        self.assertIs(display_rgb(linear, 'Linear'), linear)
        srgb = display_rgb(linear, 'sRGB')
        np.testing.assert_allclose(srgb[0, 0, 0], 0.4613, atol=2e-3)
        aces = display_rgb(linear, 'ACES 2.0')
        self.assertLess(aces[0, 0, 2], srgb[0, 0, 2])  # tone mapping pulls the highlight down
        np.testing.assert_array_equal(linear, np.array([[[0.18, 0.5, 4.0]]], np.float32))


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_exr_roundtrip_preserves_hdr_and_negative_values(self):
        path = self.dir / 'scene.exr'
        source = np.array([[[12.5, -0.4, 0.18, 1.0], [0.0, 0.0, 0.0, 0.0]]], np.float32)
        write_exr(path, source)
        np.testing.assert_array_equal(read_media(str(path)), source)

    def test_exr_output_is_linear_float_rgba(self):
        path = self.dir / 'spec.exr'
        write_exr(path, np.zeros((2, 3, 4), np.float32))
        spec = oiio.ImageInput.open(str(path)).spec()
        self.assertEqual(list(spec.channelnames), ['R', 'G', 'B', 'A'])
        self.assertEqual(str(spec.format), 'float')
        # OpenImageIO stores its own canonical name for the working space we request.
        self.assertEqual(spec.get_string_attribute('oiio:ColorSpace'), 'lin_rec709_scene')

    def test_exr_write_is_atomic_and_extension_checked(self):
        with self.assertRaisesRegex(ValueError, 'must end in .exr'):
            write_exr(self.dir / 'frame.png', np.zeros((1, 1, 4), np.float32))
        path = self.dir / 'kept.exr'
        write_exr(path, np.zeros((1, 1, 4), np.float32))
        with self.assertRaises(Exception):
            write_exr(path, np.zeros((0, 0, 4), np.float32))
        self.assertEqual(read_media(str(path)).shape, (1, 1, 4))
        self.assertEqual([p.name for p in self.dir.iterdir()], ['kept.exr'])

    def test_named_layer_selection_and_missing_layer_lists_channels(self):
        path = self.dir / 'layers.exr'
        pixels = np.zeros((1, 1, 7), np.float32)
        pixels[0, 0] = [1, 0, 0, 0, 1, 0, 0.5]
        write_exr_raw(path, pixels, ['R', 'G', 'B', 'diffuse.R', 'diffuse.G', 'diffuse.B', 'diffuse.A'])
        np.testing.assert_allclose(read_media(str(path))[0, 0], [1, 0, 0, 1])
        # EXR alpha defaults to associated, so the layer's stored RGB survives intact.
        layer = read_media(str(path), layer='diffuse')
        np.testing.assert_allclose(layer[0, 0], [0, 1, 0, 0.5], atol=1e-6)
        np.testing.assert_allclose(read_media(str(path), layer='diffuse', alpha_mode='Straight')[0, 0],
                                   [0, 0.5, 0, 0.5], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'diffuse.R'):
            read_media(str(path), layer='specular')

    def test_single_channel_depth_is_readable_as_luminance(self):
        path = self.dir / 'depth.exr'
        write_exr_raw(path, np.full((1, 1, 1), 42.0, np.float32), ['Z'])
        np.testing.assert_array_equal(read_media(str(path), layer='Z')[0, 0], [42, 42, 42, 1])

    def test_data_window_is_placed_inside_the_display_window(self):
        path = self.dir / 'crop.exr'
        write_exr_raw(path, np.ones((2, 2, 4), np.float32), ['R', 'G', 'B', 'A'], x=1, y=1, full=(4, 3))
        frame = read_media(str(path))
        self.assertEqual(frame.shape, (3, 4, 4))
        self.assertEqual(frame.sum(), 16)  # 2x2 ones, nothing wrapped or lost
        np.testing.assert_array_equal(frame[1:3, 1:3], np.ones((2, 2, 4), np.float32))

    def test_data_window_outside_the_display_window_is_clipped(self):
        path = self.dir / 'offscreen.exr'
        write_exr_raw(path, np.ones((2, 2, 4), np.float32), ['R', 'G', 'B', 'A'], x=-8, y=-8, full=(4, 4))
        np.testing.assert_array_equal(read_media(str(path)), np.zeros((4, 4, 4), np.float32))

    def test_explicit_overrides_beat_extension_defaults(self):
        path = self.dir / 'encoded.exr'
        write_exr_raw(path, np.full((1, 1, 4), 0.5, np.float32), ['R', 'G', 'B', 'A'])
        default = read_media(str(path))  # EXR defaults: linear, premultiplied
        np.testing.assert_allclose(default[0, 0, :3], 0.5, atol=1e-6)
        straight = read_media(str(path), alpha_mode='Straight')
        np.testing.assert_allclose(straight[0, 0, :3], 0.25, atol=1e-6)
        decoded = read_media(str(path), colorspace='sRGB', alpha_mode='Straight')
        np.testing.assert_allclose(decoded[0, 0, :3], srgb_to_linear(np.float32(0.5)) * 0.5, atol=2e-3)

    def test_unreadable_and_unsupported_inputs_report_actionable_errors(self):
        with self.assertRaisesRegex(ValueError, 'Read properties'):
            read_media('')
        with self.assertRaisesRegex(ValueError, 'EXR, PNG, JPEG and TIFF'):
            read_media(str(self.dir / 'clip.mov'))
        broken = self.dir / 'broken.exr'
        broken.write_bytes(b'not an exr')
        with self.assertRaisesRegex(ValueError, 'Unable to read image'):
            read_media(str(broken))
        path = self.dir / 'parts.exr'
        write_exr(path, np.zeros((1, 1, 4), np.float32))
        with self.assertRaisesRegex(ValueError, 'No subimage'):
            read_media(str(path), subimage=3)


class DocumentUpgradeTests(unittest.TestCase):
    def test_version_one_read_nodes_gain_color_defaults(self):
        old = {'version': 1, 'view': 'r', 'nodes': {'r': {'type': 'Read', 'name': 'Read1', 'pos': [0, 0],
                                                         'disabled': False, 'inputs': {}, 'params': {'path': '/tmp/a.png'}}}}
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded['version'], 3)
        self.assertEqual(upgraded['nodes']['r']['params'],
                         {'path': '/tmp/a.png', 'colorspace': 'Auto', 'alpha_mode': 'Auto', 'layer': '', 'subimage': 0})
        self.assertEqual(old['nodes']['r']['params'], {'path': '/tmp/a.png'})  # input untouched

    def test_invalid_choices_are_rejected(self):
        d = Dispatcher()
        d.execute({'op': 'create', 'type': 'Read', 'id': 'r', 'params': {'path': ''}})
        with self.assertRaisesRegex(ValueError, 'Invalid colorspace'):
            d.execute({'op': 'set', 'id': 'r', 'param': 'colorspace', 'value': 'Rec.2020'})
        self.assertEqual(d.document['nodes']['r']['params']['colorspace'], 'Auto')

    def test_read_node_passes_color_controls_through_the_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'half.exr'
            write_exr_raw(path, np.full((1, 1, 4), 0.5, np.float32), ['R', 'G', 'B', 'A'])
            d = Dispatcher()
            d.execute({'op': 'create', 'type': 'Read', 'id': 'r', 'params': {'path': str(path)}})
            evaluator = Evaluator()
            np.testing.assert_allclose(evaluator.evaluate(d.document, 'r')[0, 0, :3], 0.5, atol=1e-6)
            d.execute({'op': 'set', 'id': 'r', 'param': 'alpha_mode', 'value': 'Straight'})
            np.testing.assert_allclose(evaluator.evaluate(d.document, 'r')[0, 0, :3], 0.25, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
