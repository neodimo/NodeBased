"""Camera3D film back: focal length and apertures, the derived field of view, old documents and
the Alembic, USD and glTF readers."""
import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import alembicio, filmback, gltfio, scene3d, usdio
from nodebased.core import Dispatcher, SCHEMA_VERSION, SPECS, upgrade_document
from nodebased.imaging import Evaluator

ABC = Path(__file__).parent / "fixtures" / "abc" / "probe.abc"


def hand_fov(focal, aperture):
    return math.degrees(2 * math.atan(aperture / (2 * focal)))


def graph(**camera):
    d = Dispatcher()
    for key, kind, params in (("card", "Card3D", {"tx": 0.4, "rows": 1}), ("cube", "Cube3D", {"tx": -0.8, "ty": 0.3}),
                              ("cam", "Camera3D", camera), ("scene", "Scene3D", {}),
                              ("render", "Render3D", {"width": 64, "height": 48, "samples": 1})):
        d.execute({"op": "create", "id": key, "type": kind, "params": params})
    for slot, source in (("object0", "card"), ("object1", "cube")):
        d.execute({"op": "connect", "id": "scene", "input": slot, "source": source})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    return d


class DerivedFieldOfViewTests(unittest.TestCase):
    def test_default_film_back_is_exactly_forty_five_degrees(self):
        params = SPECS["Camera3D"]["params"]
        self.assertEqual(params["haperture"], 24.576)
        self.assertEqual(params["vaperture"], 18.672)
        self.assertAlmostEqual(params["focal"], 22.5390978, places=6)
        self.assertEqual(filmback.fov_from_aperture(params["focal"], params["vaperture"]), 45.0)
        self.assertNotIn("fov", params)
        self.assertEqual(scene3d.Camera().fov, 45.0)
        self.assertAlmostEqual(scene3d.Camera().focal, params["focal"], places=9)

    def test_three_focal_lengths_match_hand_computed_fov(self):
        for focal in (18.0, 35.0, 85.0):
            d = graph(focal=focal)
            camera = scene3d.camera_from_node(d.document["nodes"]["cam"])
            self.assertAlmostEqual(camera.fov, hand_fov(focal, 18.672), places=8)
            self.assertAlmostEqual(camera.hfov, hand_fov(focal, 24.576), places=8)
            self.assertAlmostEqual(camera.focal, focal, places=8)
        # 35 mm on 18.672 mm: 2 * atan(0.26674...) = 29.871 degrees.
        self.assertAlmostEqual(hand_fov(35.0, 18.672), 29.871, places=3)

    def test_aperture_changes_the_field_of_view(self):
        camera = scene3d.camera_from_node(graph(focal=20.0, vaperture=36.0, haperture=48.0).document["nodes"]["cam"])
        self.assertAlmostEqual(camera.fov, hand_fov(20.0, 36.0), places=8)
        self.assertAlmostEqual(camera.hfov, hand_fov(20.0, 48.0), places=8)


class RenderIdentityTests(unittest.TestCase):
    def test_default_camera_renders_like_the_pre_film_back_camera(self):
        d = graph()
        image = Evaluator().evaluate(d.document, "render")
        scene = Evaluator().evaluate_raster(d.document, "scene", frame=1, typed=True)
        pinned = scene3d.render(scene, scene3d.Camera(), 64, 48)  # fov=45.0 as it always was
        np.testing.assert_array_equal(image, pinned)
        self.assertGreater(float(image[..., 3].sum()), 0)

    def test_old_document_with_only_fov_loads_and_renders_identically(self):
        for fov in (45.0, 30.0, 90.0):
            d = graph()
            old = copy.deepcopy(d.document)
            params = old["nodes"]["cam"]["params"]
            for name in ("focal", "haperture", "vaperture"):
                del params[name]
            params["fov"] = fov
            upgraded = upgrade_document(old)
            self.assertEqual(upgraded["version"], SCHEMA_VERSION)
            self.assertNotIn("fov", upgraded["nodes"]["cam"]["params"])
            from nodebased.core import validate
            validate(upgraded)
            image = Evaluator().evaluate(upgraded, "render")
            scene = Evaluator().evaluate_raster(upgraded, "scene", frame=1, typed=True)
            np.testing.assert_array_equal(image, scene3d.render(scene, scene3d.Camera(fov=fov), 64, 48))

    def test_old_animated_fov_curve_keeps_its_keys(self):
        d = graph()
        old = copy.deepcopy(d.document)
        params = old["nodes"]["cam"]["params"]
        for name in ("focal", "haperture", "vaperture"):
            del params[name]
        params["fov"] = 45.0
        old["animation"]["curves"]["cam"] = {"fov": {"interpolation": "linear", "keys": [
            {"frame": 1, "value": 30.0}, {"frame": 10, "value": 60.0}]}}
        upgraded = upgrade_document(old)
        from nodebased.core import validate
        validate(upgraded)
        for frame, fov in ((1, 30.0), (10, 60.0)):
            image = Evaluator().evaluate(upgraded, "render", frame=frame)
            scene = Evaluator().evaluate_raster(upgraded, "scene", frame=frame, typed=True)
            np.testing.assert_array_equal(image, scene3d.render(scene, scene3d.Camera(fov=fov), 64, 48))


class ReaderTests(unittest.TestCase):
    def test_alembic_camera_keeps_focal_length_and_apertures(self):
        # The fixture stores 3.6 x 2 cm apertures (a 36 x 20 mm film back), 35 mm then 70 mm.
        for time, focal in ((1 / 24, 35.0), (5 / 24, 70.0)):
            camera = alembicio.load_camera(ABC, time)
            self.assertAlmostEqual(camera.haperture, 36.0, places=5)
            self.assertAlmostEqual(camera.vaperture, 20.0, places=5)
            self.assertAlmostEqual(camera.focal, focal, places=5)
            self.assertAlmostEqual(camera.fov, hand_fov(focal, 20.0), places=6)

    def test_usd_camera_keeps_focal_length_and_apertures(self):
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest("usd-core is not installed")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lens.usda"
            stage = Usd.Stage.CreateNew(str(path))
            camera = UsdGeom.Camera.Define(stage, "/Camera")
            camera.CreateFocalLengthAttr(35)
            camera.CreateHorizontalApertureAttr(36)
            camera.CreateVerticalApertureAttr(24)
            camera.CreateClippingRangeAttr((0.1, 500))
            stage.GetRootLayer().Save()
            lens = usdio.load_camera(path, 1)
        self.assertAlmostEqual(lens.focal, 35.0, places=6)
        self.assertAlmostEqual(lens.haperture, 36.0, places=6)
        self.assertAlmostEqual(lens.vaperture, 24.0, places=6)
        self.assertAlmostEqual(lens.fov, hand_fov(35.0, 24.0), places=6)

    def gltf(self, folder, camera):
        path = Path(folder) / "camera.gltf"
        path.write_text(json.dumps({
            "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"name": "cam", "camera": 0, "translation": [1, 2, 3]}],
            "cameras": [camera]}))
        return path

    def test_gltf_camera_becomes_an_equivalent_film_back(self):
        yfov = math.radians(50.0)
        with tempfile.TemporaryDirectory() as folder:
            lens = gltfio.load_camera(self.gltf(folder, {"type": "perspective", "perspective": {
                "yfov": yfov, "aspectRatio": 2.0, "znear": 0.2, "zfar": 90.0}}))
            bare = gltfio.load_camera(self.gltf(folder, {"type": "perspective", "perspective": {
                "yfov": yfov, "znear": 0.1}}))
            with self.assertRaisesRegex(ValueError, "only perspective"):
                gltfio.load_camera(self.gltf(folder, {"type": "orthographic", "orthographic": {
                    "xmag": 1, "ymag": 1, "znear": 0.1, "zfar": 10}}))
        self.assertAlmostEqual(lens.fov, 50.0, places=6)
        self.assertAlmostEqual(lens.vaperture, 18.672)
        self.assertAlmostEqual(lens.haperture, 37.344)      # vertical aperture * aspect ratio
        self.assertAlmostEqual(lens.hfov, hand_fov(lens.focal, 37.344), places=6)
        self.assertAlmostEqual(lens.focal, 18.672 / (2 * math.tan(yfov / 2)), places=8)
        self.assertEqual((lens.near, lens.far), (0.2, 90.0))
        np.testing.assert_allclose(lens.transform.position.array(), (1, 2, 3))
        # No aspectRatio in the file: the documented default horizontal aperture.
        self.assertEqual(bare.haperture, filmback.DEFAULT_HAPERTURE)
        self.assertAlmostEqual(bare.fov, 50.0, places=6)


if __name__ == "__main__":
    unittest.main()
