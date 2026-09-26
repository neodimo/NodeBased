"""Step 5c part 4: the conditioning bundle. `Write` with `bundle` writes a multichannel EXR and a
JSON manifest per frame; `ReadBundle` brings a model's output for the same frame back and refuses a
manifest for another frame or size. See docs/3D_FOUNDATION.md, "The conditioning bundle"."""
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from unittest.mock import patch

from nodebased import bundle
from nodebased.app import Window
from nodebased.core import SPECS, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.media import write_exr
from tests.test_2d_parity_group_3b import Graph
from tests.test_2d_parity_step_5c import H, W, constant_layer, texture
from tests.test_3d_multichannel_exr import graph as render_graph
from tests.test_desktop import APP, wait_until


def channels(path):
    source = oiio.ImageInput.open(str(path))
    try:
        return list(source.spec().channelnames)
    finally:
        source.close()


class Scratch(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)


class WriteBundleTests(Scratch):
    def rendered(self, **camera):
        d = render_graph()
        for name, value in camera.items():
            d.execute(dict(op="set", id="camera", param=name, value=value))
        return d

    def test_manifest_names_every_layer_the_frame_the_camera_and_a_fingerprint(self):
        d = self.rendered(tz=6.5, focal=40.0)
        target = self.dir / "shot.0003.exr"
        raster, manifest_file = bundle.write_frame(Evaluator(), d.document, "write", 3, str(target), "float")
        self.assertEqual(manifest_file, self.dir / "shot.0003.bundle.json")
        manifest = json.loads(manifest_file.read_text())
        self.assertEqual((manifest["format"], manifest["version"], manifest["frame"]), ("nodebased-bundle", 1, 3))
        self.assertEqual((manifest["image"], manifest["file_type"], manifest["bit_depth"]), ("shot.0003.exr", "exr", "float"))
        self.assertEqual((manifest["width"], manifest["height"]), (32, 24))
        self.assertEqual([layer["name"] for layer in manifest["layers"]], ["normals", "depth"])
        written = channels(target)
        for layer in manifest["layers"]:
            for name in layer["channels"]:
                self.assertIn(name, written)
        by_name = {layer["name"]: layer for layer in manifest["layers"]}
        self.assertEqual(by_name["normals"]["channels"], ["normals.X", "normals.Y", "normals.Z"])
        self.assertEqual(by_name["normals"]["convention"]["range"], [-1.0, 1.0])
        self.assertEqual(by_name["normals"]["convention"]["space"], "world")
        self.assertEqual(by_name["depth"]["channels"], ["depth.Z"])
        self.assertIn("away from the camera", by_name["depth"]["convention"]["direction"])
        camera = manifest["camera"]
        self.assertEqual(camera["type"], "Camera3D")
        self.assertEqual((camera["position"][2], camera["focal"]), (6.5, 40.0))
        self.assertEqual((camera["width"], camera["height"]), (32, 24))
        for field in ("target", "roll", "fov", "haperture", "vaperture", "near", "far"):
            self.assertIn(field, camera)
        self.assertRegex(manifest["fingerprint"], r"^[0-9a-f]{64}$")

    def test_fingerprint_follows_the_document(self):
        d = self.rendered()
        first = json.loads(bundle.write_frame(Evaluator(), d.document, "write", 1, str(self.dir / "a.exr"), "float")[1].read_text())
        again = json.loads(bundle.write_frame(Evaluator(), d.document, "write", 1, str(self.dir / "b.exr"), "float")[1].read_text())
        self.assertEqual(first["fingerprint"], again["fingerprint"])
        d.execute(dict(op="set", id="camera", param="tz", value=9.0))
        moved = json.loads(bundle.write_frame(Evaluator(), d.document, "write", 1, str(self.dir / "c.exr"), "float")[1].read_text())
        self.assertNotEqual(first["fingerprint"], moved["fingerprint"])
        self.assertEqual(moved["camera"]["position"][2], 9.0)

    def test_a_motion_layer_states_its_units_and_direction_and_no_render_means_no_camera(self):
        path = self.dir / "flow.exr"
        write_exr(path, texture(), bits="float", layers=dict(motion=constant_layer(4.0, -1.0), swirl=constant_layer(1.0, 0.0)))
        g = Graph()
        g.add("src", "Read", {"path": str(path)})
        g.add("w", "Write", {"bundle": 1}, image="src")
        _, manifest_file = bundle.write_frame(Evaluator(), g.doc, "w", 7, str(self.dir / "out.0007.exr"), "half")
        manifest = json.loads(manifest_file.read_text())
        self.assertIsNone(manifest["camera"])
        by_name = {layer["name"]: layer for layer in manifest["layers"]}
        self.assertEqual(by_name["motion"]["channels"], ["motion.X", "motion.Y"])
        self.assertEqual(by_name["motion"]["convention"]["units"], "pixels per frame")
        self.assertIn("forward", by_name["motion"]["convention"]["direction"])
        self.assertEqual(by_name["swirl"]["convention"]["units"], "unspecified")     # never guessed
        # the EXR carries the motion values exactly (float) under the names the manifest gives
        source = oiio.ImageInput.open(str(self.dir / "out.0007.exr"))
        names = list(source.spec().channelnames)
        pixels = source.read_image(oiio.FLOAT)
        source.close()
        np.testing.assert_array_equal(pixels[..., names.index("motion.X")], np.full((H, W), 4.0, np.float32))
        np.testing.assert_array_equal(pixels[..., names.index("motion.Y")], np.full((H, W), -1.0, np.float32))

    def test_write_carries_a_bundle_option_and_old_writes_without_it_still_validate(self):
        self.assertEqual(SPECS["Write"]["params"]["bundle"], 0)
        groups = knob_layout("Write")
        self.assertIn("bundle", [param for group in groups for param in group.params])
        g = Graph()
        g.add("c", "Checker")
        g.add("w", "Write", image="c")
        g.add("s", "Shuffle", image="c")
        for key, name in (("w", "bundle"), ("s", "layer")):
            del g.doc["nodes"][key]["params"][name]                # a document written before step 5c
        with self.assertRaises(ValueError):
            validate(g.doc)                                         # strict about unknown shapes...
        old = upgrade_document(g.doc)                               # ...so the upgrade fills the defaults
        validate(old)
        self.assertEqual((old["nodes"]["w"]["params"]["bundle"], old["nodes"]["s"]["params"]["layer"]), (0, ""))
        Evaluator().evaluate(dict(old, view="w"))


class ReadBundleTests(Scratch):
    def make(self, frame=3, model=None):
        d = render_graph()
        target = self.dir / f"shot.{frame:04d}.exr"
        raster, manifest_file = bundle.write_frame(Evaluator(), d.document, "write", frame, str(target), "float")
        image = model if model is not None else self.dir / f"model.{frame:04d}.exr"
        if model is None:
            write_exr(image, raster.to_display(), bits="float")
        return raster, manifest_file, image

    def read_node(self, image, manifest_file, **params):
        g = Graph()
        g.add("rb", "ReadBundle", dict(path=str(image), bundle=str(manifest_file), **params))
        return g

    def test_model_output_comes_back_at_the_same_frame(self):
        raster, manifest_file, image = self.make(frame=3)
        g = self.read_node(image, manifest_file)
        out = Evaluator().evaluate_raster(dict(g.doc, view="rb"), frame=3)
        np.testing.assert_array_equal(out.pixels, raster.to_display())
        self.assertEqual((out.display.width, out.display.height), (32, 24))

    def test_a_different_frame_is_refused(self):
        _, manifest_file, image = self.make(frame=3)
        g = self.read_node(image, manifest_file)
        with self.assertRaisesRegex(ValueError, r"manifest is for frame 3 but the graph is at frame 4"):
            Evaluator().evaluate_raster(dict(g.doc, view="rb"), frame=4)

    def test_a_different_size_is_refused(self):
        _, manifest_file, _ = self.make(frame=3)
        wrong = self.dir / "wrong.exr"
        write_exr(wrong, np.zeros((10, 12, 4), np.float32), bits="float")
        g = self.read_node(wrong, manifest_file)
        with self.assertRaisesRegex(ValueError, r"image is 12x10 but the bundle was written at 32x24"):
            Evaluator().evaluate_raster(dict(g.doc, view="rb"), frame=3)

    def test_sequences_pair_each_frame_with_its_own_manifest(self):
        for frame in (1, 2):
            self.make(frame=frame)
        g = self.read_node(self.dir / "model.%04d.exr", self.dir / "shot.%04d.bundle.json")
        evaluator = Evaluator()
        first = evaluator.evaluate_raster(dict(g.doc, view="rb"), frame=1).pixels.copy()
        second = evaluator.evaluate_raster(dict(g.doc, view="rb"), frame=2).pixels
        np.testing.assert_array_equal(first, second)           # same scene, so equal pixels, read per frame
        with self.assertRaisesRegex(ValueError, "Missing frame 3|missing file"):
            evaluator.evaluate_raster(dict(g.doc, view="rb"), frame=3)

    def test_unreadable_or_foreign_manifests_are_errors(self):
        _, _, image = self.make(frame=3)
        for content, message in (("not json", "not valid JSON"), ('{"format": "other"}', "not a NodeBased bundle manifest"),
                                 (json.dumps({"format": "nodebased-bundle", "version": 9}), "version 9")):
            manifest = self.dir / "bad.bundle.json"
            manifest.write_text(content)
            g = self.read_node(image, manifest)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Evaluator().evaluate_raster(dict(g.doc, view="rb"), frame=3)
        g = Graph()
        g.add("rb", "ReadBundle", dict(path=str(image)))
        with self.assertRaisesRegex(ValueError, "choose the bundle manifest"):
            Evaluator().evaluate_raster(dict(g.doc, view="rb"), frame=3)

    def test_editing_the_image_re_reads_it(self):
        raster, manifest_file, image = self.make(frame=3)
        g = self.read_node(image, manifest_file)
        evaluator = Evaluator()
        first = evaluator.evaluate_raster(dict(g.doc, view="rb"), frame=3).pixels.copy()
        write_exr(image, np.zeros((24, 32, 4), np.float32), bits="float")
        second = evaluator.evaluate_raster(dict(g.doc, view="rb"), frame=3).pixels
        self.assertGreater(float(np.abs(first).max()), 0.0)
        self.assertEqual(float(np.abs(second).max()), 0.0)


class WriteNodeUiTests(Scratch):
    def window(self):
        window = Window()
        window.show()
        self.assertTrue(wait_until(lambda: window.frame is not None))
        self.addCleanup(lambda: (setattr(window, "saved_document", window.dispatcher.document), window.close(),
                                 APP.processEvents()))
        commands = [{"op": "create", "id": "cube", "type": "Cube3D"}, {"op": "create", "id": "camera", "type": "Camera3D"},
                    {"op": "create", "id": "scene", "type": "Scene3D"},
                    {"op": "create", "id": "render", "type": "Render3D",
                     "params": dict(width=32, height=24, samples=1, render_output="multichannel")},
                    {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
                    {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
                    {"op": "connect", "id": "render", "input": "camera", "source": "camera"},
                    {"op": "create", "id": "writer", "type": "Write"},
                    {"op": "connect", "id": "writer", "input": "image", "source": "render"}]
        window.command({"op": "batch", "commands": commands})
        return window

    def test_render_write_with_the_bundle_option_writes_the_manifest_beside_the_frame(self):
        window = self.window()
        target = self.dir / "mc.%04d.exr"
        window.command({"op": "set", "id": "writer", "param": "path", "value": str(target)})
        window.command({"op": "set", "id": "writer", "param": "bit_depth", "value": "float"})
        window.command({"op": "set", "id": "writer", "param": "bundle", "value": 1})
        window.render_write("writer", single=True)
        frame = window.dispatcher.document["time"]["current"]
        manifest = json.loads((self.dir / f"mc.{frame:04d}.bundle.json").read_text())
        self.assertEqual(manifest["frame"], frame)
        self.assertEqual([layer["name"] for layer in manifest["layers"]], ["normals", "depth"])
        self.assertEqual(manifest["camera"]["type"], "Camera3D")
        self.assertTrue((self.dir / manifest["image"]).is_file())

    def test_without_the_option_no_manifest_is_written_and_png_with_it_is_refused(self):
        window = self.window()
        window.command({"op": "set", "id": "writer", "param": "path", "value": str(self.dir / "plain.exr")})
        window.render_write("writer", single=True)
        self.assertEqual(sorted(path.name for path in self.dir.iterdir()), ["plain.exr"])
        window.command({"op": "set", "id": "writer", "param": "path", "value": str(self.dir / "x.png")})
        window.command({"op": "set", "id": "writer", "param": "bundle", "value": 1})
        with patch("nodebased.app.QMessageBox.warning") as warned:
            window.render_write("writer", single=True)
        self.assertIn("needs an EXR", warned.call_args.args[2])
        self.assertFalse((self.dir / "x.png").exists())


if __name__ == "__main__":
    unittest.main()
