"""Lane L4 step K1: Render3D writes Cryptomatte layers (docs/PARITY_2D.md Keyer row 8, the reader
side; step K3, 2026-09-26). Render3D's `cryptomatte` knob (off by default) writes CryptoObject
(the node name, and an instance id for Instance3D copies), CryptoMaterial (the geometry's material
string) and CryptoAsset (the outermost Scene3D/Axis3D parent's name) into the same EXR the beauty
pass writes, with a manifest per set, and a Read + Cryptomatte graph isolates them exactly the way
a Nuke Cryptomatte node would.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import cryptomatte as cm
from nodebased import scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import raster_layer_arrays, read_media_raster, write_exr


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, name=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        if name:
            self.d.execute(dict(op="rename", id=key, name=name))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    @property
    def doc(self):
        return self.d.document


def scene_graph(**render):
    """card (named 'RedCard') and cube (named 'GreenCube') side by side, lit flat (no shading), a
    camera looking straight down -Z, and a Render3D with the given extra params."""
    g = Graph()
    g.add("card", "Card3D", dict(card_width=1.4, card_height=1.4, tx=-0.6, red=1.0, green=0.0, blue=0.0),
         name="RedCard")
    g.add("cube", "Cube3D", dict(cube_size=1.0, tx=0.6, red=0.0, green=1.0, blue=0.0), name="GreenCube")
    g.add("camera", "Camera3D")
    g.add("scene", "Scene3D", object0="card", object1="cube")
    g.add("render", "Render3D", dict(width=32, height=24, samples=2, **render), scene="scene", camera="camera")
    return g


def evaluate(g, key="render"):
    return Evaluator().evaluate_raster(g.doc, key)


class CryptomatteOffTests(unittest.TestCase):
    def test_off_by_default_and_leaves_the_exr_byte_identical(self):
        g = scene_graph()
        self.assertEqual(g.doc["nodes"]["render"]["params"]["cryptomatte"], 0)
        raster = evaluate(g)
        self.assertNotIn("CryptoObject00", raster.layers or {})
        direct = s.render(Evaluator().evaluate_raster(g.doc, "scene", typed=True),
                          Evaluator().evaluate_raster(g.doc, "camera", typed=True), 32, 24, samples=2)
        np.testing.assert_array_equal(raster.pixels, direct)

    def test_turning_it_on_adds_layers_without_changing_the_beauty(self):
        off, on = evaluate(scene_graph()), evaluate(scene_graph(cryptomatte=1))
        np.testing.assert_array_equal(off.pixels, on.pixels)
        self.assertIn("CryptoObject00", on.layers)
        self.assertIn("CryptoMaterial00", on.layers)
        self.assertIn("CryptoAsset00", on.layers)


class CryptomatteReadBackTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def write(self, g, key="render"):
        raster = evaluate(g, key)
        path = Path(self._dir.name) / "render.exr"
        write_exr(path, raster.to_display() if hasattr(raster, "to_display") else raster.pixels,
                  bits="float", layers=raster_layer_arrays(raster), metadata=raster.meta)
        return str(path), raster

    def matte(self, path, matte_list, crypto_layer=""):
        g = Graph()
        g.add("src", "Read", {"path": path})
        g.add("key", "Cryptomatte", dict(matte_list=matte_list, crypto_layer=crypto_layer), image="src")
        return Evaluator().evaluate(g.doc, "key")[..., 3]

    def test_isolating_a_name_matches_its_rendered_coverage(self):
        path, raster = self.write(scene_graph(cryptomatte=1))
        card_alpha = self.matte(path, "RedCard", "CryptoObject")
        # Render the card alone at the same resolution/samples: its own alpha is its own coverage.
        solo = Graph()
        solo.add("card", "Card3D", dict(card_width=1.4, card_height=1.4, tx=-0.6, red=1.0, green=0.0, blue=0.0),
                name="RedCard")
        solo.add("camera", "Camera3D")
        solo.add("scene", "Scene3D", object0="card")
        solo.add("render", "Render3D", dict(width=32, height=24, samples=2), scene="scene", camera="camera")
        solo_alpha = evaluate(solo).pixels[..., 3]
        np.testing.assert_allclose(card_alpha, solo_alpha, atol=1e-3)

    def test_two_overlapping_objects_share_coverage_summing_to_one(self):
        # A narrower gap so the card and cube share antialiased edge pixels between them.
        g = Graph()
        g.add("card", "Card3D", dict(card_width=1.4, card_height=1.4, tx=-0.35, red=1.0, green=0.0, blue=0.0),
             name="RedCard")
        g.add("cube", "Cube3D", dict(cube_size=1.0, tx=0.35, red=0.0, green=1.0, blue=0.0), name="GreenCube")
        g.add("camera", "Camera3D")
        g.add("scene", "Scene3D", object0="card", object1="cube")
        g.add("render", "Render3D", dict(width=32, height=24, samples=4, cryptomatte=1),
             scene="scene", camera="camera")
        path, raster = self.write(g)
        both = self.matte(path, "RedCard, GreenCube", "CryptoObject")
        beauty_alpha = evaluate(g).pixels[..., 3]
        # Wherever the beauty pass has any surface coverage at all, the two names together account
        # for all of it: nothing hides a third, unnamed contributor.
        np.testing.assert_allclose(both, beauty_alpha, atol=1e-3)
        edge = np.argwhere((both > 0.05) & (both < 0.95))
        self.assertGreater(len(edge), 0, "expected at least one antialiased edge pixel")

    def test_ids_match_the_readers_hash_for_the_node_names(self):
        path, raster = self.write(scene_graph(cryptomatte=1))
        manifest = cm.crypto_metadata(raster.meta)
        by_name = {entry["name"]: cm.parse_manifest(entry["manifest"]) for entry in manifest.values()}
        objects = by_name[next(k for k in by_name if k == "CryptoObject")]
        self.assertEqual(objects["RedCard"], cm.name_to_bits("RedCard"))
        self.assertEqual(objects["GreenCube"], cm.name_to_bits("GreenCube"))
        materials = by_name["CryptoMaterial"]
        self.assertEqual(materials["standard"], cm.name_to_bits("standard"))

    def test_manifest_round_trips_through_read_and_write(self):
        path, raster = self.write(scene_graph(cryptomatte=1))
        g = Graph()
        g.add("src", "Read", {"path": path})
        g.add("dst", "Write", {"path": str(Path(self._dir.name) / "again.exr"), "bit_depth": "float"}, image="src")
        again_raster = Evaluator().evaluate_raster(g.doc, "src")
        write_exr(Path(self._dir.name) / "again.exr", again_raster.to_display(),
                  bits="float", layers=raster_layer_arrays(again_raster), metadata=again_raster.meta)
        reread = read_media_raster(str(Path(self._dir.name) / "again.exr"))
        self.assertEqual(cm.crypto_metadata(reread.meta), cm.crypto_metadata(raster.meta))

    def test_instances_get_distinct_object_ids(self):
        g = Graph()
        g.add("card", "Card3D", dict(card_width=0.3, card_height=0.3), name="Dot")
        g.add("points", "Cube3D", dict(cube_size=1.6), name="Points")
        g.add("inst", "Instance3D", dict(inst_variant="cycle"), points="points", instance="card")
        g.add("camera", "Camera3D")
        g.add("scene", "Scene3D", object0="inst")
        g.add("render", "Render3D", dict(width=24, height=18, samples=2, cryptomatte=1),
             scene="scene", camera="camera")
        path, raster = self.write(g)
        entries = cm.crypto_metadata(raster.meta)
        objects = next(cm.parse_manifest(e["manifest"]) for e in entries.values() if e["name"] == "CryptoObject")
        instance_names = [name for name in objects if name.startswith("Dot_")]
        self.assertGreaterEqual(len(instance_names), 2, "expected at least two distinctly-named instances")
        self.assertEqual(len(set(instance_names)), len(instance_names))

    def test_raster_and_raytrace_modes_agree(self):
        raster_mode = evaluate(scene_graph(cryptomatte=1, render_mode="raster"))
        ray_mode = evaluate(scene_graph(cryptomatte=1, render_mode="raytrace"))
        for layer in ("CryptoObject00", "CryptoMaterial00", "CryptoAsset00"):
            np.testing.assert_allclose(raster_mode.layers[layer].pixels, ray_mode.layers[layer].pixels, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
