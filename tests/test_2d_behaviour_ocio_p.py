"""2D behaviour lane 2, step P, part 1: the OCIO nodes' and Colorspace's remaining knobs act.

Context variables, swap, gain, gamma, saturation, channel view, ignore errors, working space, interpolation and the Colorspace choices (Rec.709
curve, Cineon, the CIE illuminants) each change the pixels the way the guide or its long-standing convention says. The config is the small one of
tools/nuke_fidelity_ocio_fixture.py; every expected value is computed from the transforms it states.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import nuke_fidelity_ocio_fixture as fixture  # noqa: E402
from nodebased import core  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from tests.test_2d_behaviour_knobs_color import assert_in_order  # noqa: E402
from tests.test_2d_behaviour_filter_a import evaluator_pixels, source_graph, tile_pixels  # noqa: E402
from tests.test_2d_parity_group_3b import Graph  # noqa: E402

TOL = dict(rtol=2e-4, atol=1e-4)
COLOUR = (0.25, 0.5, 0.75)
LUMA = np.array([0.2126, 0.7152, 0.0722])

CONTEXT_CONFIG = """ocio_profile_version: 2

environment:
  LUT: ident
search_path: "{directory}"
strictparsing: true
luma: [0.2126, 0.7152, 0.0722]
name: nodebased-context-fixture
description: A colorspace whose LUT file is named by the context variable LUT

roles:
  scene_linear: lin
  default: lin
  reference: lin

file_rules:
  - !<Rule> {{name: Default, colorspace: lin}}

displays:
  Mon:
    - !<View> {{name: Plain, colorspace: lutspace}}

active_displays: [Mon]
active_views: [Plain]

colorspaces:
  - !<ColorSpace>
    name: lin
    isdata: false

  - !<ColorSpace>
    name: lutspace
    isdata: false
    from_scene_reference: !<FileTransform> {{src: "$LUT.cube", interpolation: linear}}
"""
IDENT_CUBE = "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n"


class Config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.config, cls.cube = fixture.write(cls._dir.name)
        directory = Path(cls._dir.name)
        (directory / "ident.cube").write_text(IDENT_CUBE, encoding="utf-8")
        (directory / "bent.cube").write_text(fixture.CUBE, encoding="utf-8")
        cls.context_config = str(directory / "context.ocio")
        Path(cls.context_config).write_text(CONTEXT_CONFIG.format(directory=cls._dir.name), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def run_node(self, kind, params, alpha=1.0, colour=COLOUR, config=None):
        g = Graph()
        g.add("flat", "Constant", dict(width=4, height=3, red=float(colour[0]), green=float(colour[1]), blue=float(colour[2]), alpha=alpha))
        g.add("node", kind, dict(config=config or self.config, **params), image="flat")
        return np.asarray(evaluator_pixels(g.doc, "node")[0, 0], np.float64)


class ContextTests(Config):
    """key1 / value1 ... key4 / value4 set variables the config's file transforms read (Nuke: 'specifying the variable in the to_reference file transform')."""

    def test_the_value_picks_the_file_a_colorspace_reads(self):
        x = np.array(COLOUR)
        for kind, params in (("OCIOColorspace", dict(src="lin", dst="lutspace")),
                             ("OCIOLookTransform", dict(src="lin", dst="lutspace")),
                             ("OCIODisplay", dict(src="lin", display="Mon", view="Plain"))):
            with self.subTest(kind=kind):
                default = self.run_node(kind, params, config=self.context_config)[:3]
                np.testing.assert_allclose(default, x, **TOL)       # the environment's LUT=ident
                bent = self.run_node(kind, dict(params, key1="LUT", value1="bent"), config=self.context_config)[:3]
                np.testing.assert_allclose(bent, fixture.lut(x), **TOL)
                self.assertGreater(np.abs(bent - default).max(), 0.05)

    def test_every_one_of_the_four_slots_sets_its_variable(self):
        for slot in range(1, 5):
            got = self.run_node("OCIOColorspace", {"src": "lin", "dst": "lutspace", f"key{slot}": "LUT", f"value{slot}": "bent"}, config=self.context_config)
            np.testing.assert_allclose(got[:3], fixture.lut(np.array(COLOUR)), **TOL)

    def test_a_blank_key_sets_nothing(self):
        got = self.run_node("OCIOColorspace", dict(src="lin", dst="lutspace", key1="", value1="bent"), config=self.context_config)
        np.testing.assert_allclose(got[:3], COLOUR, **TOL)


class SwapTests(Config):
    def test_swap_trades_the_two_colorspaces(self):
        a = self.run_node("OCIOColorspace", dict(src="lin", dst="encoded", swap=1))
        b = self.run_node("OCIOColorspace", dict(src="encoded", dst="lin"))
        np.testing.assert_allclose(a, b, **TOL)
        self.assertGreater(np.abs(a - self.run_node("OCIOColorspace", dict(src="lin", dst="encoded"))).max(), 0.05)

    def test_colorspace_swap_trades_every_input_and_output_choice(self):
        base = dict(transfer_in="sRGB", transfer_out="Linear", primaries_in="Rec.709", primaries_out="Rec.2020", whitepoint_in="D65", whitepoint_out="D65")
        swapped = {"transfer_in": "Linear", "transfer_out": "sRGB", "primaries_in": "Rec.2020", "primaries_out": "Rec.709", "whitepoint_in": "D65", "whitepoint_out": "D65"}
        a = Evaluator._kernel("Colorspace", dict(core.SPECS["Colorspace"]["params"], **base, swap=1), [np.full((2, 2, 4), 0.4, np.float32)])
        b = Evaluator._kernel("Colorspace", dict(core.SPECS["Colorspace"]["params"], **swapped), [np.full((2, 2, 4), 0.4, np.float32)])
        np.testing.assert_allclose(a, b, atol=1e-6)


class DisplayTests(Config):
    """gain, gamma, saturation and the channel view around the display transform (the `Plain` view of `Mon` is the 2.2th root)."""

    def display(self, **knobs):
        return self.run_node("OCIODisplay", dict(src="lin", display="Mon", view="Plain", **knobs))

    def test_defaults_leave_the_display_transform_alone(self):
        np.testing.assert_allclose(self.display()[:3], fixture.encode(np.array(COLOUR)), **TOL)

    def test_gain_multiplies_before_the_transform(self):
        np.testing.assert_allclose(self.display(gain=2.0)[:3], fixture.encode(2.0 * np.array(COLOUR)), **TOL)
        np.testing.assert_allclose(self.display(gain=0.5)[:3], fixture.encode(0.5 * np.array(COLOUR)), **TOL)

    def test_gamma_applies_after_the_transform_and_above_one_brightens(self):
        base = fixture.encode(np.array(COLOUR))
        np.testing.assert_allclose(self.display(gamma=2.0)[:3], base ** 0.5, **TOL)
        self.assertTrue(np.all(self.display(gamma=2.0)[:3] > base))

    def test_saturation_moves_the_colour_toward_or_away_from_grey_before_the_transform(self):
        x = np.array(COLOUR)
        grey = float(x @ LUMA)
        np.testing.assert_allclose(self.display(saturation=0.0)[:3], fixture.encode(np.full(3, grey)), **TOL)
        np.testing.assert_allclose(self.display(saturation=1.5)[:3], fixture.encode(grey + 1.5 * (x - grey)), **TOL)

    def test_channel_view_shows_one_channel_as_grey(self):
        x = np.array(COLOUR)
        for name, index in (("R", 0), ("G", 1), ("B", 2)):
            np.testing.assert_allclose(self.display(channel_selector=name)[:3], fixture.encode(np.full(3, x[index])), **TOL)
        np.testing.assert_allclose(self.display(channel_selector="Luminance")[:3], fixture.encode(np.full(3, float(x @ LUMA))), **TOL)
        got = np.asarray(self.run_node("OCIODisplay", dict(src="lin", display="Mon", view="Plain", channel_selector="A"), alpha=0.25))
        np.testing.assert_allclose(got[:3], fixture.encode(np.full(3, 0.25)), **TOL)

    def test_inverse_undoes_the_forward_chain(self):
        knobs = dict(gain=2.0, gamma=1.5, saturation=0.5)
        forward = self.display(**knobs)
        back = self.run_node("OCIODisplay", dict(src="lin", display="Mon", view="Plain", transform_direction="inverse", **knobs), colour=tuple(forward[:3]))
        np.testing.assert_allclose(back[:3], COLOUR, rtol=2e-3, atol=2e-3)


class LookAndFileTests(Config):
    def test_ignore_errors_turns_a_missing_look_into_a_plain_conversion(self):
        params = dict(src="lin", dst="encoded", look="no-such-look")
        with self.assertRaises(Exception):
            self.run_node("OCIOLookTransform", params)
        got = self.run_node("OCIOLookTransform", dict(params, ignore_errors=1))
        np.testing.assert_allclose(got[:3], fixture.encode(np.array(COLOUR)), **TOL)

    def test_reload_changes_no_pixels(self):
        a = self.run_node("OCIOLookTransform", dict(src="lin", dst="encoded", look="double"))
        b = self.run_node("OCIOLookTransform", dict(src="lin", dst="encoded", look="double", reload=3))
        np.testing.assert_array_equal(a, b)

    def test_working_space_applies_the_file_there_and_comes_back(self):
        x = np.array(COLOUR)
        got = self.run_node("OCIOFileTransform", dict(path=self.cube, working_space="encoded"))
        np.testing.assert_allclose(got[:3], fixture.decode(fixture.lut(fixture.encode(x))), **TOL)
        scene = self.run_node("OCIOFileTransform", dict(path=self.cube, working_space="scene_linear"))
        np.testing.assert_allclose(scene[:3], fixture.lut(x), **TOL)

    def test_interpolation_changes_a_3d_lut(self):
        """tetrahedral and best interpolate a 3D LUT differently from linear; OpenColorIO renders nearest as linear, so Nuke (which runs OpenColorIO) does too."""
        cube3d = Path(self._dir.name) / "corners3d.cube"
        rng = np.random.default_rng(1)
        cube3d.write_text("LUT_3D_SIZE 2\n" + "\n".join(" ".join(f"{v:.3f}" for v in rng.random(3)) for _ in range(8)) + "\n", encoding="utf-8")
        out = {name: self.run_node("OCIOFileTransform", dict(path=str(cube3d), file_interpolation=name), colour=(0.3, 0.6, 0.45))[:3]
               for name in ("nearest", "linear", "tetrahedral", "best")}
        self.assertGreater(np.abs(out["tetrahedral"] - out["linear"]).max(), 0.01)
        np.testing.assert_allclose(out["best"], out["tetrahedral"], atol=1e-6)
        np.testing.assert_allclose(out["nearest"], out["linear"], atol=1e-6)


class ColorspaceChoiceTests(unittest.TestCase):
    def convert(self, value, **choices):
        params = dict(core.SPECS["Colorspace"]["params"], **choices)
        return float(Evaluator._kernel("Colorspace", params, [np.full((1, 1, 4), value, np.float32)])[0, 0, 0])

    def test_rec709_curve_is_the_camera_transfer_function_not_the_srgb_one(self):
        for linear in (0.005, 0.01, 0.05, 0.18, 0.5, 0.9):
            want = 4.5 * linear if linear < 0.018 else 1.099 * linear ** 0.45 - 0.099
            self.assertAlmostEqual(self.convert(linear, transfer_out="Rec.709"), want, places=5)
            self.assertAlmostEqual(self.convert(want, transfer_in="Rec.709"), linear, places=5)
        self.assertGreater(abs(self.convert(0.18, transfer_out="Rec.709") - self.convert(0.18, transfer_out="sRGB")), 0.005)

    def test_cineon_is_the_kodak_density_curve(self):
        offset = 10 ** ((95 - 685) * 0.002 / 0.6)
        for code in (95, 300, 470, 685, 900):
            linear = (10 ** ((code - 685) * 0.002 / 0.6) - offset) / (1 - offset)
            self.assertAlmostEqual(self.convert(code / 1023.0, transfer_in="Cineon"), linear, places=5)
            self.assertAlmostEqual(self.convert(linear, transfer_out="Cineon"), code / 1023.0, places=4)
        self.assertAlmostEqual(self.convert(0.0, transfer_out="Cineon") * 1023.0, 95.0, places=2)      # black sits at code 95

    def test_new_illuminants_keep_the_xyz_values(self):
        for name, xy in (("A", (0.44757, 0.40745)), ("D50", (0.34567, 0.35850)), ("D55", (0.33242, 0.34743)), ("D75", (0.29902, 0.31485)),
                         ("C", (0.31006, 0.31616)), ("B", (0.34842, 0.35161)), ("E", (1 / 3, 1 / 3))):
            # white (1,1,1) of Rec.709 primaries on D65 has XYZ of D65; with the illuminant changed and the Bradford matrix off the RGB values change so
            # the XYZ stays: the white therefore maps to RGB values other than 1 unless the illuminant is D65
            out = self.convert(1.0, whitepoint_out=name)
            self.assertNotAlmostEqual(out, 1.0, places=3, msg=name)
            back = self.convert(out, whitepoint_in=name)
            self.assertAlmostEqual(back, 1.0, places=4, msg=name)

    def test_the_choices_are_listed_and_older_choices_are_unchanged(self):
        self.assertEqual(core.CHOICES["transfer_in"][:4], ["Linear", "sRGB", "Gamma 2.2", "Gamma 2.4"])
        self.assertEqual(core.CHOICES["whitepoint_out"][:3], ["D65", "D60", "DCI"])
        for name in ("Rec.709", "Cineon"):
            self.assertIn(name, core.CHOICES["transfer_in"])
        for name in ("A", "B", "C", "D50", "D55", "D75", "E"):
            self.assertIn(name, core.CHOICES["whitepoint_in"])


class OlderDocumentTests(Config):
    def test_a_node_saved_without_the_new_params_renders_bit_for_bit_the_same(self):
        pixels = np.random.default_rng(3).random((5, 6, 4)).astype(np.float32)
        for kind, params in (("OCIOColorspace", dict(src="lin", dst="encoded")), ("OCIODisplay", dict(src="lin", display="Mon", view="Plain")),
                             ("OCIOLookTransform", dict(src="lin", dst="encoded", look="double")), ("OCIOFileTransform", dict(path=self.cube)),
                             ("Colorspace", dict(transfer_in="sRGB", primaries_out="Rec.2020"))):
            old = dict(core.SPECS[kind]["params"], config=self.config, **params)
            new_names = set(core.SPECS[kind]["params"]) & (set(__import__("nodebased.ocioknobs", fromlist=["NEW"]).NEW[kind]))
            saved = {k: v for k, v in old.items() if k not in new_names}
            np.testing.assert_array_equal(Evaluator._kernel(kind, old, [pixels]), Evaluator._kernel(kind, saved, [pixels]), kind)


class TileTests(Config):
    def test_tiles_equal_the_whole_frame_with_the_new_knobs_moved(self):
        requests = {
            "OCIODisplay": dict(display="Mon", view="Plain", gain=1.5, gamma=1.2, saturation=0.7, channel_selector="Luminance"),
            "OCIOColorspace": dict(src="lin", dst="gain2", swap=1),
            "OCIOLookTransform": dict(src="lin", dst="encoded", look="no-such-look", ignore_errors=1),
            "OCIOFileTransform": dict(path=None, working_space="encoded"),
        }
        for kind, params in requests.items():
            params = dict(params, config=self.config)
            if "path" in params:
                params["path"] = self.cube
            for masked in (False, True):
                g = source_graph(kind, params, masked)
                full = np.asarray(evaluator_pixels(g.doc, "node"))
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, masked=masked, edge=edge):
                        np.testing.assert_allclose(np.asarray(tile_pixels(g.doc, "node", edge)), full, atol=1e-6)


class LayoutTests(unittest.TestCase):
    def test_main_tab_order_follows_the_guide(self):
        assert_in_order(self, "OCIOColorspace", ["Channels", "Input", "Output", "Swap input/output", "Key1", "Value1", "Key4", "Value4"])
        assert_in_order(self, "OCIODisplay", ["Input colorspace", "Display", "View", "Invert direction", "Gain", "Gamma", "Saturation", "Channel view", "Key1", "Value4"])
        assert_in_order(self, "OCIOLookTransform", ["Channels", "Input", "Look", "Reload", "Invert direction", "Output", "Ignore errors", "Key1", "Value4"])
        assert_in_order(self, "OCIOFileTransform", ["Channels", "File", "Invert direction", "Interpolation", "Working space"])


if __name__ == "__main__":
    unittest.main()
