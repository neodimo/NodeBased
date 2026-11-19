"""2D behaviour, lane 9 step X4: MotionBlur2D Output UV, VectorDistort Frame Distance, Vectorfield GPU extrapolate and 3DL file format,
Encryptomatte Export Write (Nuke 17.0 Reference Guide pages of the four nodes).

* MotionBlur2D Output UV: with a 2D transf connected the motion across the shutter is written to the named layer (u red, v green, in pixels), and
  the blurred picture is the one it was; none writes no layer; the layer leaves the picture alone.
* VectorDistort Frame Distance and the two Vectorfield knobs are held on the node as Nuke holds them: the stored value is in the document, listed in the
  panel in the guide's order, and (said plainly in `default_notes`) moves no pixel.
* Encryptomatte Export Write: the panel's button adds a Write below the node, wired to it and set for .exr output.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from nodebased import core, knobs
from nodebased.core import CHOICES, SPECS, Dispatcher
from nodebased.imaging import Evaluator
from tests.test_2d_behaviour_filter_a import Graph, differs
from tests.test_2d_behaviour_hsv_vf_p import run_vectorfield, write_3dl
from tests.test_2d_behaviour_knobs_color import assert_in_order
from tests.test_2d_behaviour_transform_n import render as vd_render, rig


class MotionBlur2DUvTests(unittest.TestCase):
    SHIFT = 8.0

    def graph(self, with_transf=True, **params):
        g = Graph()
        g.add("src", "Checker", dict(width=64, height=32, size=8))
        g.add("tex", "Transform", dict(translate_x=0.0, filter="bilinear"), image="src")
        g.d.execute(dict(op="set_key", id="tex", param="translate_x", frame=1, value=0.0))
        g.d.execute(dict(op="set_key", id="tex", param="translate_x", frame=2, value=self.SHIFT))
        g.d.execute(dict(op="set_key", id="tex", param="translate_x", frame=3, value=3 * self.SHIFT))
        wires = dict(image="tex")
        if with_transf:
            wires["transf"] = "tex"
        g.add("node", "MotionBlur2D", {**dict(shutter=1.0, shutter_offset="start", samples=8), **params}, **wires)
        return g

    @staticmethod
    def raster(g, frame=2):
        return Evaluator().evaluate_raster(dict(g.doc, view="node"), target="node", frame=frame)

    def test_the_default_layer_is_motion_as_the_guide_lists(self):
        self.assertEqual(SPECS["MotionBlur2D"]["params"]["uv"], "motion")

    def test_the_transforms_motion_is_written_to_the_layer_in_pixels(self):
        raster = self.raster(self.graph())
        layer = raster.layers["motion"].pixels
        self.assertGreater(float(layer[..., 0].mean()), 1.0)                      # the picture moves right across the shutter
        np.testing.assert_allclose(layer[..., 1], 0.0, atol=1e-4)                  # and not up or down
        np.testing.assert_allclose(layer[..., 3], 1.0)
        # a pure translation moves every pixel by the same amount: the Transform moves 8 then 24 pixels over frames 2 to 3, the first and last
        # of the eight samples sit at 2 + 1/16 and 3 - 1/16
        self.assertAlmostEqual(float(layer[0, 0, 0]), float(layer[-1, -1, 0]), places=3)

    def test_a_shorter_shutter_moves_less(self):
        long_ = self.raster(self.graph(shutter=1.0)).layers["motion"].pixels[..., 0].mean()
        short = self.raster(self.graph(shutter=0.25)).layers["motion"].pixels[..., 0].mean()
        self.assertGreater(float(long_), float(short))

    def test_the_layer_takes_the_name_typed_in(self):
        raster = self.raster(self.graph(uv="smartvector.forward"))
        self.assertIn("smartvector.forward", raster.layers)
        self.assertNotIn("motion", raster.layers)

    def test_none_writes_no_layer_and_a_node_without_a_transf_has_none_to_write(self):
        self.assertNotIn("motion", self.raster(self.graph(uv="none")).layers or {})
        self.assertNotIn("motion", self.raster(self.graph(with_transf=False)).layers or {})

    def test_the_layer_leaves_the_picture_as_it_was(self):
        with_layer = self.raster(self.graph(uv="motion")).pixels
        without = self.raster(self.graph(uv="none")).pixels
        np.testing.assert_array_equal(with_layer, without)

    def test_the_panel_leads_with_output_uv(self):
        assert_in_order(self, "MotionBlur2D", ["Output UV", "Shutter"])


class VectorDistortFrameDistanceTests(unittest.TestCase):
    def test_the_default_is_one_frame_and_the_choices_run_one_to_sixty_four(self):
        self.assertEqual(SPECS["VectorDistort"]["params"]["frame_distance"], "1 frame")
        self.assertEqual(CHOICES["frame_distance"][0], "1 frame")
        self.assertEqual(CHOICES["frame_distance"][-1], "64 frames")

    def test_the_value_is_held_on_the_node_and_moves_no_pixel(self):
        base = vd_render(rig(), "vd", 2)
        for value in ("8 frames", "64 frames"):
            np.testing.assert_array_equal(vd_render(rig(frame_distance=value), "vd", 2), base)

    def test_a_saved_node_loads_with_one_frame(self):
        from tests.test_2d_behaviour_filter_a import old_document
        doc = old_document("VectorDistort", {"reference_frame": 1}, drop=["frame_distance"])
        self.assertEqual(doc["nodes"]["n"]["params"]["frame_distance"], "1 frame")

    def test_the_panel_lists_it_after_hold_frame_as_the_guide_does(self):
        assert_in_order(self, "VectorDistort", ["Reference Frame", "Hold frame", "Frame distance", "Output"])


class VectorfieldKnobTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)
        self.path = self.dir / "mesh.3dl"
        write_3dl(self.path)

    def tearDown(self):
        self._dir.cleanup()

    def test_the_defaults_are_the_guides(self):
        params = SPECS["Vectorfield"]["params"]
        self.assertEqual((params["vector_gpu_extrapolate"], params["vector_format_3dl"]), (1, "auto"))
        self.assertEqual(CHOICES["vector_format_3dl"], ["auto", "autodesk flame/smoke", "autodesk lustre", "assimilate scratch"])

    def test_gpu_extrapolate_and_the_3dl_format_move_no_pixel(self):
        base = run_vectorfield(self.path)
        np.testing.assert_array_equal(run_vectorfield(self.path, vector_gpu_extrapolate=0), base)
        for value in CHOICES["vector_format_3dl"]:
            np.testing.assert_array_equal(run_vectorfield(self.path, vector_format_3dl=value), base)

    def test_a_document_saved_without_the_knobs_loads_with_them_at_the_guides_defaults(self):
        from tests.test_2d_behaviour_filter_a import old_document
        doc = old_document("Vectorfield", {"cube_path": ""}, drop=["vector_gpu_extrapolate", "vector_format_3dl"])
        params = doc["nodes"]["n"]["params"]
        self.assertEqual((params["vector_gpu_extrapolate"], params["vector_format_3dl"]), (1, "auto"))

    def test_a_node_saved_without_the_knobs_reads_the_same(self):
        omit = ("vector_gpu_extrapolate", "vector_format_3dl")
        np.testing.assert_array_equal(run_vectorfield(self.path, omit=omit), run_vectorfield(self.path))

    def test_the_panel_lists_them_in_the_guides_order(self):
        assert_in_order(self, "Vectorfield", ["Interpolation", "GPU extrapolate", "Colorspace in", "Colorspace out", "Output bit depth (.3dl)",
                                              "Normalize output", "3DL file format"])


class EncryptomatteExportWriteTests(unittest.TestCase):
    def test_the_button_adds_an_exr_write_wired_to_the_node(self):
        import types
        from nodebased.app import Window
        d = Dispatcher()
        d.execute(dict(op="create", id="crypto", type="Encryptomatte", pos=[10, 20]))
        sent = []
        window = types.SimpleNamespace(dispatcher=d, statusBar=lambda: types.SimpleNamespace(showMessage=lambda text: None))
        window.command = lambda command: (sent.append(command), d.execute(command))
        self.assertTrue(Window._encryptomatte_export_write(window, "crypto"))
        writes = [k for k, n in d.document["nodes"].items() if n["type"] == "Write"]
        self.assertEqual(len(writes), 1)
        write = d.document["nodes"][writes[0]]
        self.assertEqual(write["inputs"]["image"], "crypto")
        self.assertEqual((write["params"]["file_type"], write["params"]["bit_depth"], write["params"]["channels"]), ("exr", "float", "all"))
        core.validate(d.document)

    def test_a_node_that_is_not_an_encryptomatte_is_refused(self):
        import types
        from nodebased.app import Window
        d = Dispatcher()
        d.execute(dict(op="create", id="blur", type="Blur"))
        window = types.SimpleNamespace(dispatcher=d, command=lambda c: self.fail("no command expected"))
        self.assertFalse(Window._encryptomatte_export_write(window, "blur"))
        self.assertFalse(Window._encryptomatte_export_write(window, "nothing"))


class EncryptomattePanelTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])
        from tests.test_desktop import release_window
        from nodebased.app import Window
        self.window = Window()
        self.window.show()
        self.app.processEvents()
        self.addCleanup(release_window, self)

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.app.processEvents()

    def test_the_panel_has_an_export_write_button(self):
        from PySide6.QtWidgets import QPushButton
        self.window.dispatcher.execute(dict(op="create", id="enc", type="Encryptomatte"))
        panel = self.window.build_node_panel("enc")
        button = panel.findChild(QPushButton, "encryptomatte-export-write")
        self.assertIsNotNone(button)
        button.click()
        self.app.processEvents()
        kinds = [n["type"] for n in self.window.dispatcher.document["nodes"].values()]
        self.assertIn("Write", kinds)


if __name__ == "__main__":
    unittest.main()
