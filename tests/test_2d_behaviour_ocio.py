"""2D behaviour lane 2, step F, part 2: the five OCIO nodes hand OpenColorIO the request the knobs state.

The config is the small one of tools/nuke_fidelity_ocio_fixture.py: a 2x gain matrix (`gain2`), a power law of 2.2 (`encoded`, and `cmplog` for the
compositing_log role), a look `double` and a bent three-point LUT file. Every expected value is computed from those statements, not from OpenColorIO.
Nuke converts the stored colour, so a picture with alpha below 1 is transformed as it is; `divide_by_alpha` keeps the older divide-first behaviour for
graphs saved before.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import audit_nuke_fidelity as audit  # noqa: E402
import nuke_fidelity_cases as ruler  # noqa: E402
import nuke_fidelity_ocio_fixture as fixture  # noqa: E402
import nuke_fidelity_requirements as requirements  # noqa: E402
from nodebased import core  # noqa: E402
from nodebased.core import upgrade_document  # noqa: E402
from nodebased.knobs import knob_layout  # noqa: E402
from tests.test_2d_behaviour_filter_a import evaluator_pixels, source_graph, tile_pixels  # noqa: E402
from tests.test_2d_parity_group_3b import Graph  # noqa: E402

OCIO_NODES = ("OCIOColorspace", "OCIOLogConvert", "OCIOLookTransform", "OCIODisplay", "OCIOFileTransform")
TOL = dict(rtol=2e-4, atol=1e-4)      # OpenColorIO's power law is a fast approximation (relative error up to about 2e-5)
COLOUR = (0.25, 0.5, 0.75)


class Config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.config, cls.cube = fixture.write(cls._dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def run_node(self, kind, params, alpha=1.0, colour=COLOUR, mix=1.0):
        g = Graph()
        g.add("flat", "Constant", dict(width=4, height=3, red=float(colour[0]), green=float(colour[1]), blue=float(colour[2]), alpha=alpha))
        g.add("node", kind, dict(config=self.config, mix=mix, **params), image="flat")
        return evaluator_pixels(g.doc, "node")[0, 0]


class CaseTests(unittest.TestCase):
    def test_every_ocio_case_passes(self):
        cases = [c for c in ruler.CASES if c.nuke in OCIO_NODES]
        self.assertGreaterEqual(len(cases), 15)
        self.assertEqual({c.nuke for c in cases}, set(OCIO_NODES))
        for case in cases:
            with self.subTest(case=case.id):
                result = audit.run_case(case)
                self.assertEqual(result["status"], "pass", (case.id, result["max_abs_error"], result["worst"], case.statement))


class RequestTests(Config):
    """Through the graph: the config reaches OpenColorIO, and each knob picks the part of the request it names."""

    def test_colorspace_goes_in_to_out_and_back(self):
        x = np.array(COLOUR)
        np.testing.assert_allclose(self.run_node("OCIOColorspace", dict(src="lin", dst="gain2"))[:3], fixture.gain(x), **TOL)
        encoded = self.run_node("OCIOColorspace", dict(src="lin", dst="encoded"))[:3]
        np.testing.assert_allclose(encoded, fixture.encode(x), **TOL)
        np.testing.assert_allclose(self.run_node("OCIOColorspace", dict(src="encoded", dst="lin"), colour=tuple(encoded))[:3], x, **TOL)

    def test_colorspace_inverse_swaps_the_two_spaces(self):
        got = self.run_node("OCIOColorspace", dict(src="lin", dst="encoded", transform_direction="inverse"))[:3]
        np.testing.assert_allclose(got, fixture.decode(np.array(COLOUR)), **TOL)

    def test_logconvert_uses_the_compositing_log_role_both_ways(self):
        x = np.array(COLOUR)
        np.testing.assert_allclose(self.run_node("OCIOLogConvert", dict(ocio_log_operation="log to lin"))[:3], fixture.decode(x), **TOL)
        np.testing.assert_allclose(self.run_node("OCIOLogConvert", dict(ocio_log_operation="lin to log"))[:3], fixture.encode(x), **TOL)

    def test_look_on_and_off_and_inverted(self):
        x = np.array(COLOUR)
        np.testing.assert_allclose(self.run_node("OCIOLookTransform", dict(src="lin", dst="lin", look=""))[:3], x, **TOL)
        np.testing.assert_allclose(self.run_node("OCIOLookTransform", dict(src="lin", dst="lin", look="double"))[:3], fixture.gain(x), **TOL)
        np.testing.assert_allclose(self.run_node("OCIOLookTransform", dict(src="lin", dst="lin", look="double", transform_direction="inverse"))[:3], x / 2, **TOL)

    def test_display_and_view_each_pick_the_space(self):
        x = np.array(COLOUR)
        plain = self.run_node("OCIODisplay", dict(display="Mon", view="Plain", look=""))[:3]
        bright = self.run_node("OCIODisplay", dict(display="Mon", view="Bright", look=""))[:3]
        alt = self.run_node("OCIODisplay", dict(display="Alt", view="Plain", look=""))[:3]
        np.testing.assert_allclose(plain, fixture.encode(x), **TOL)
        np.testing.assert_allclose(bright, fixture.gain(x), **TOL)
        np.testing.assert_allclose(alt, fixture.gain(x), **TOL)
        back = self.run_node("OCIODisplay", dict(display="Mon", view="Plain", look="", transform_direction="inverse"), colour=tuple(plain))[:3]
        np.testing.assert_allclose(back, x, **TOL)

    def test_display_starts_from_the_input_colorspace_or_the_scene_linear_role(self):
        x = np.array(COLOUR)
        for name in ("linear", "", "lin", "scene_linear"):
            with self.subTest(src=name):
                np.testing.assert_allclose(self.run_node("OCIODisplay", dict(src=name, display="Mon", view="Plain", look=""))[:3], fixture.encode(x), **TOL)
        got = self.run_node("OCIODisplay", dict(src="gain2", display="Mon", view="Plain", look=""))[:3]
        np.testing.assert_allclose(got, fixture.encode(x / 2), **TOL)

    def test_file_transform_forward_and_inverse(self):
        x = np.array(COLOUR)
        forward = self.run_node("OCIOFileTransform", dict(path=self.cube))[:3]
        np.testing.assert_allclose(forward, fixture.lut(x), **TOL)
        back = self.run_node("OCIOFileTransform", dict(path=self.cube, transform_direction="inverse"), colour=tuple(forward))[:3]
        np.testing.assert_allclose(back, x, **TOL)

    def test_mix_blends_with_the_input(self):
        x = np.array(COLOUR)
        got = self.run_node("OCIOColorspace", dict(src="lin", dst="gain2"), mix=0.5)[:3]
        np.testing.assert_allclose(got, 0.5 * x + 0.5 * fixture.gain(x), **TOL)

    def test_a_missing_config_or_space_is_explained(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            g = Graph()
            g.add("flat", "Constant", dict(width=2, height=2))
            g.add("node", "OCIOColorspace", dict(config=self.config + ".missing", src="lin", dst="gain2"), image="flat")
            evaluator_pixels(g.doc, "node")


class StoredColourTests(Config):
    """Nuke transforms the stored colour and leaves alpha alone; the older divide-by-alpha behaviour stays for saved graphs."""

    def test_the_stored_colour_is_converted_and_alpha_is_kept(self):
        stored = np.array(COLOUR) * 0.5          # Constant stores its colour times its alpha
        got = self.run_node("OCIOColorspace", dict(src="lin", dst="encoded"), alpha=0.5)
        np.testing.assert_allclose(got[:3], fixture.encode(stored), **TOL)
        self.assertEqual(float(got[3]), 0.5)

    def test_divide_by_alpha_is_the_older_behaviour(self):
        x = np.array(COLOUR)
        got = self.run_node("OCIOColorspace", dict(src="lin", dst="encoded", divide_by_alpha=1), alpha=0.5)
        np.testing.assert_allclose(got[:3], 0.5 * fixture.encode(x), **TOL)

    def test_a_new_node_does_not_divide_and_a_saved_one_does(self):
        for kind in OCIO_NODES:
            self.assertEqual(core.SPECS[kind]["params"]["divide_by_alpha"], 0, kind)
            document = {"version": 29, "nodes": {"n": {"type": kind, "name": kind, "params": {}, "inputs": {}, "pos": [0, 0], "disabled": False}}}
            params = upgrade_document(document)["nodes"]["n"]["params"]
            self.assertEqual(params["divide_by_alpha"], 1, kind)

    def test_opaque_pictures_are_the_same_either_way(self):
        for flag in (0, 1):
            a = self.run_node("OCIOColorspace", dict(src="lin", dst="encoded", divide_by_alpha=flag))
            np.testing.assert_array_equal(a, self.run_node("OCIOColorspace", dict(src="lin", dst="encoded", divide_by_alpha=0)))


class TileTests(Config):
    def test_tiles_of_7_16_and_32_equal_the_whole_frame_with_and_without_a_mask(self):
        requests = {
            "OCIOColorspace": dict(src="lin", dst="gain2"),
            "OCIOLogConvert": dict(ocio_log_operation="lin to log"),
            "OCIOLookTransform": dict(src="lin", dst="encoded", look="double"),
            "OCIODisplay": dict(display="Mon", view="Plain", look=""),
            "OCIOFileTransform": dict(path=self.cube),
        }
        for kind, params in requests.items():
            for masked in (False, True):
                g = source_graph(kind, dict(config=self.config, **params), masked)
                full = np.asarray(evaluator_pixels(g.doc, "node"))
                for edge in (7, 16, 32):
                    with self.subTest(kind=kind, masked=masked, edge=edge):
                        np.testing.assert_allclose(np.asarray(tile_pixels(g.doc, "node", edge)), full, atol=1e-6)


class KnobActionTests(Config):
    """Every main-tab knob the mapping pairs acts with that config (the audit probe runs the same way)."""

    def test_each_paired_knob_changes_the_pixels(self):
        mapping = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]
        fx = ruler.make_fixtures()
        core_chrome = ("maskChannelInput", "inject", "invert_mask", "fringe", "unpremult", "invert_unpremult")      # the mask strip acts with a mask wired
        for kind in OCIO_NODES:
            paired = sorted({n for e in mapping[kind]["knobs"] if e["kind"] in ("same", "renamed") for n in e["nodebased"]}
                            & set(core.SPECS[kind]["params"]))
            acts = requirements.probe_acts(kind, paired, fx)
            for name in paired:
                if name in ("config", "mix", "channels", "divide_by_alpha", "file_interpolation", *core_chrome):
                    continue
                with self.subTest(kind=kind, knob=name):
                    self.assertTrue(acts.get(name), f"{kind}.{name} did not change the pixels with the fixture config")


class LayoutTests(unittest.TestCase):
    def test_main_tab_order_and_labels(self):
        want = {
            "OCIOColorspace": [("channels", "Channels"), ("src", "Input"), ("dst", "Output")],
            "OCIOLogConvert": [("channels", "Channels"), ("ocio_log_operation", "Operation")],
            "OCIOLookTransform": [("channels", "Channels"), ("src", "Input"), ("look", "Look"), ("transform_direction", "Invert direction"), ("dst", "Output")],
            "OCIODisplay": [("src", "Input colorspace"), ("display", "Display"), ("view", "View"), ("look", "Look"), ("transform_direction", "Invert direction")],
            "OCIOFileTransform": [("channels", "Channels"), ("path", "File"), ("transform_direction", "Invert direction"), ("file_interpolation", "File Interpolation")],
        }
        for kind, rows in want.items():
            with self.subTest(kind=kind):
                got = [(g.params[0], g.label) for g in knob_layout(kind)]
                self.assertEqual(got[:len(rows)], rows)


if __name__ == "__main__":
    unittest.main()
