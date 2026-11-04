"""2D behaviour lane 2, step F, part 1: Cryptomatte's manifest source, sidecar and preview, and Encryptomatte's matte name, merge operation and layer.

The Nuke 17.0 Reference Guide (keyer_nodes/cryptomatte.html) says of the Manifest Source knob: Metadata - the manifest is embedded in the input image
file's metadata; Sidecar - the manifest is a separate JSON file whose location the sidecar field holds. Of Preview: enabled, the matte boundaries are drawn
and each separate matte can be seen; disabled, the original input is displayed. Of Encryptomatte (keyer_nodes/encryptomatte.html): Matte Name auto names the
matte after the matte source node (for example Roto1), manual takes the name typed; Merge Operation: the new matte behaves as the A input of a standard Merge
and the existing set as the B input.

The fixture is the two-ID Cryptomatte EXR of tests/test_2d_parity_step_k3.py (real file, real Read), with the sphere's id moved so the file's embedded
manifest and the sidecar disagree about it.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from nodebased import cryptomatte as cm  # noqa: E402
from nodebased.core import upgrade_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, evaluator_raster
from tests.test_2d_parity_step_k3 import H, NAMES, SET, W, Scratch, beauty, crypto_layers, manifest_metadata, truth

MOVED = cm.name_to_bits("sphere_v2")        # the id the file's layers hold for the sphere


def moved_layers():
    return crypto_layers({"sphere": MOVED, "cube": cm.name_to_bits("cube"), "plane": cm.name_to_bits("plane")})


def labels(kind):
    return [(g.params[0], g.label) for g in knob_layout(kind)]


class ManifestSourceTests(Scratch):
    """The embedded manifest holds sphere -> hash("sphere"), which no pixel carries; the sidecar holds the id the layers do carry."""

    def sidecar(self, table=None):
        path = self._dir.name + "/manifest.json"
        table = table if table is not None else {"sphere": "%08x" % MOVED, "cube": "%08x" % cm.name_to_bits("cube"), "plane": "%08x" % cm.name_to_bits("plane")}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(table, handle)
        return path

    def test_metadata_reads_the_embedded_manifest(self):
        matte = self.matte({"matte_list": "sphere", "manifest_source": "metadata", "sidecar_filepath": self.sidecar()}, layers=moved_layers())
        np.testing.assert_array_equal(matte, np.zeros((H, W), np.float32))      # the embedded table points at an id the pixels do not hold

    def test_sidecar_reads_the_json_file_instead(self):
        matte = self.matte({"matte_list": "sphere", "manifest_source": "sidecar", "sidecar_filepath": self.sidecar()}, layers=moved_layers())
        np.testing.assert_array_equal(matte, truth("sphere"))

    def test_sidecar_needs_no_embedded_manifest(self):
        matte = self.matte({"matte_list": "cube", "manifest_source": "sidecar", "sidecar_filepath": self.sidecar()}, layers=moved_layers(), metadata={})
        np.testing.assert_array_equal(matte, truth("cube"))

    def test_sidecar_is_the_same_table_the_embedded_path_reads(self):
        table = json.loads(manifest_metadata()[f"cryptomatte/{cm.set_key(SET)}/manifest"])
        matte = self.matte({"matte_list": "plane", "manifest_source": "sidecar", "sidecar_filepath": self.sidecar(table)})
        np.testing.assert_array_equal(matte, truth("plane"))
        np.testing.assert_array_equal(matte, self.matte({"matte_list": "plane"}))

    def test_sidecar_without_a_file_is_explained(self):
        g = self.graph({"matte_list": "cube", "manifest_source": "sidecar"})
        with self.assertRaisesRegex(ValueError, "no sidecar file is set"):
            evaluator_pixels(g.doc, "key")
        g.d.execute({"op": "set", "id": "key", "param": "sidecar_filepath", "value": self._dir.name + "/missing.json"})
        with self.assertRaisesRegex(ValueError, "sidecar manifest does not exist"):
            evaluator_pixels(g.doc, "key")


class PreviewTests(Scratch):
    def test_preview_on_paints_the_id_colours_and_leaves_the_matte(self):
        g = self.graph({"matte_list": "cube", "preview_enabled": 1})
        out = evaluator_pixels(g.doc, "key")
        source = evaluator_raster(g.doc, "src")
        _, members = cm.choose_set(source)
        np.testing.assert_array_equal(out[..., :3], cm.preview(source, members))
        self.assertFalse(np.array_equal(out[..., :3], beauty()[..., :3]))
        np.testing.assert_array_equal(out[..., 3], truth("cube"))

    def test_preview_off_shows_the_input_and_the_same_matte(self):
        g = self.graph({"matte_list": "cube", "preview_enabled": 0})
        out = evaluator_pixels(g.doc, "key")
        np.testing.assert_array_equal(out[..., :3], beauty()[..., :3])
        np.testing.assert_array_equal(out[..., 3], truth("cube"))

    def test_preview_changes_nothing_in_the_matte_view(self):
        g = self.graph({"matte_list": "cube", "crypto_view": "matte", "preview_enabled": 1})
        on = evaluator_pixels(g.doc, "key")
        g.d.execute({"op": "set", "id": "key", "param": "preview_enabled", "value": 0})
        np.testing.assert_array_equal(on, evaluator_pixels(g.doc, "key"))

    def test_a_saved_node_keeps_the_input_colour(self):
        g = self.graph({"matte_list": "cube"})
        document = json.loads(json.dumps(g.doc))
        for node in document["nodes"].values():
            if node["type"] == "Cryptomatte":
                del node["params"]["preview_enabled"], node["params"]["manifest_source"], node["params"]["sidecar_filepath"]
        upgraded = upgrade_document(document)
        params = upgraded["nodes"]["key"]["params"]
        self.assertEqual(params["preview_enabled"], 0)
        out = evaluator_pixels(upgraded, "key")
        np.testing.assert_array_equal(out[..., :3], beauty()[..., :3])
        np.testing.assert_array_equal(out[..., 3], truth("cube"))


class CryptomatteLayoutTests(unittest.TestCase):
    def test_main_tab_order_and_labels(self):
        self.assertEqual(labels("Cryptomatte")[:4], [("crypto_layer", "Layer Selection"), ("manifest_source", "Manifest Source"),
                                                    ("sidecar_filepath", "Sidecar"), ("preview_enabled", "Preview")])

    def test_encryptomatte_main_tab_order_and_labels(self):
        # the matte name type and the first matte name share one row (panel sweep S4), labelled with the guide's "Matte Name"
        self.assertEqual(labels("Encryptomatte")[:3], [("matte_name_type", "Matte Name"),
                                                      ("merge_operation", "Merge Operation"), ("layer_name", "Cryptomatte Layer")])


def graph_with_mattes(mattes, params=None, names=None):
    """beauty -> enc, and one Constant per matte coverage, each named `names[i]` (a node name, as Roto1 would be)."""
    g = Graph()
    g.add("beauty", "Constant", dict(width=W, height=H, red=0.2, green=0.4, blue=0.6, alpha=1))
    g.add("enc", "Encryptomatte", dict(params or {}), image="beauty")
    for i, alpha in enumerate(mattes):
        g.d.execute({"op": "create", "id": f"m{i}", "type": "Constant", "name": (names or ["Roto1", "Roto2", "Roto3"])[i],
                     "params": dict(width=W, height=H, red=1, green=1, blue=1, alpha=alpha)})
        g.d.execute({"op": "connect", "id": "enc", "input": f"matte{i}", "source": f"m{i}"})
    return g


def read_matte(g, name, source="enc", layer=""):
    key = f"read_{len(g.doc['nodes'])}"
    g.add(key, "Cryptomatte", {"matte_list": name, "crypto_layer": layer}, image=source)
    return evaluator_pixels(g.doc, key)[..., 3]


class MatteNameTests(unittest.TestCase):
    def test_auto_names_each_matte_after_its_source_node(self):
        g = graph_with_mattes([0.4, 0.7], {"matte_name_type": "auto"})
        np.testing.assert_allclose(read_matte(g, "Roto1"), 0.4, atol=1e-6)
        np.testing.assert_allclose(read_matte(g, "Roto2"), 0.7, atol=1e-6)
        np.testing.assert_array_equal(read_matte(g, "typed"), 0.0)

    def test_auto_ignores_the_typed_names(self):
        g = graph_with_mattes([0.4], {"matte_name_type": "auto", "id0": "typed"})
        np.testing.assert_allclose(read_matte(g, "Roto1"), 0.4, atol=1e-6)
        np.testing.assert_array_equal(read_matte(g, "typed"), 0.0)

    def test_manual_uses_the_typed_names(self):
        g = graph_with_mattes([0.4, 0.7], {"matte_name_type": "manual", "id0": "typed", "id1": "other"})
        np.testing.assert_allclose(read_matte(g, "typed"), 0.4, atol=1e-6)
        np.testing.assert_allclose(read_matte(g, "other"), 0.7, atol=1e-6)
        np.testing.assert_array_equal(read_matte(g, "Roto1"), 0.0)

    def test_manual_with_no_name_writes_nothing(self):
        g = graph_with_mattes([0.4], {"matte_name_type": "manual"})
        self.assertFalse(any(name.startswith("crypto") for name in (evaluator_raster(g.doc, "enc").layers or {})))

    def test_a_saved_node_keeps_typed_names_and_a_fresh_set(self):
        g = graph_with_mattes([0.4], {"id0": "typed"})
        document = json.loads(json.dumps(g.doc))
        del document["nodes"]["enc"]["params"]["matte_name_type"], document["nodes"]["enc"]["params"]["merge_operation"]
        params = upgrade_document(document)["nodes"]["enc"]["params"]
        self.assertEqual((params["matte_name_type"], params["merge_operation"]), ("manual", "replace"))


class MergeOperationTests(unittest.TestCase):
    """The new matte is the Merge's A input and the set already on the image its B input."""

    def two_steps(self, operation, first=0.4, second=0.5):
        g = graph_with_mattes([first], {"matte_name_type": "auto"}, names=["objA"])
        g.d.execute({"op": "create", "id": "m9", "type": "Constant", "name": "objB",
                     "params": dict(width=W, height=H, red=1, green=1, blue=1, alpha=second)})
        g.add("enc2", "Encryptomatte", {"matte_name_type": "auto", "merge_operation": operation}, image="enc", matte0="m9")
        return g

    def coverage(self, g, name):
        return read_matte(g, name, source="enc2")

    def test_over_takes_the_old_set_down_by_the_new_matte(self):
        g = self.two_steps("over")
        np.testing.assert_allclose(self.coverage(g, "objA"), 0.4 * (1 - 0.5), atol=1e-6)
        np.testing.assert_allclose(self.coverage(g, "objB"), 0.5, atol=1e-6)

    def test_under_takes_the_new_matte_down_by_the_old_set(self):
        g = self.two_steps("under")
        np.testing.assert_allclose(self.coverage(g, "objA"), 0.4, atol=1e-6)
        np.testing.assert_allclose(self.coverage(g, "objB"), 0.5 * (1 - 0.4), atol=1e-6)

    def test_plus_keeps_both(self):
        g = self.two_steps("plus")
        np.testing.assert_allclose(self.coverage(g, "objA"), 0.4, atol=1e-6)
        np.testing.assert_allclose(self.coverage(g, "objB"), 0.5, atol=1e-6)

    def test_replace_writes_the_set_afresh(self):
        g = self.two_steps("replace")
        np.testing.assert_array_equal(self.coverage(g, "objA"), 0.0)
        np.testing.assert_allclose(self.coverage(g, "objB"), 0.5, atol=1e-6)

    def test_the_manifest_lists_both_names(self):
        g = self.two_steps("over")
        raster = evaluator_raster(g.doc, "enc2")
        self.assertEqual(set(cm.manifest_for(raster, SET)), {"objA", "objB"})

    def test_a_set_that_is_not_there_is_just_written(self):
        g = graph_with_mattes([0.3], {"matte_name_type": "auto", "merge_operation": "over"}, names=["objA"])
        np.testing.assert_allclose(read_matte(g, "objA"), 0.3, atol=1e-6)

    def test_another_layer_name_is_a_set_of_its_own(self):
        g = self.two_steps("over")
        g.d.execute({"op": "set", "id": "enc2", "param": "layer_name", "value": "crypto_extra"})
        raster = evaluator_raster(g.doc, "enc2")
        self.assertIn("crypto_object00", raster.layers)
        self.assertIn("crypto_extra00", raster.layers)
        np.testing.assert_allclose(read_matte(g, "objA", "enc2", "crypto_object"), 0.4, atol=1e-6)     # the first set's objects are as they were
        np.testing.assert_allclose(read_matte(g, "objB", "enc2", "crypto_extra"), 0.5, atol=1e-6)

    def test_tiles_equal_the_whole_frame_for_every_operation(self):
        for operation in ("over", "under", "plus", "replace"):
            with self.subTest(operation=operation):
                g = self.two_steps(operation)
                raster = Evaluator().evaluate_raster(dict(g.doc, view="enc2"))
                tiles = TileExecutor(evaluator=Evaluator(), tile_edge=7)
                self.assertTrue(tiles.supports_tiled(dict(g.doc, view="enc2"), "enc2"))
                tiled = tiles.compose(dict(g.doc, view="enc2"), "enc2", frame=1, tier=1)
                self.assertEqual(set(tiled.layers), set(raster.layers))
                for name, layer in raster.layers.items():
                    np.testing.assert_array_equal(tiled.layers[name], layer.pixels)


class ExportTests(unittest.TestCase):
    def test_the_export_button_is_unpaired_with_its_reason(self):
        import audit_nuke_fidelity as audit
        mapping = json.loads(audit.MAP_JSON.read_text(encoding="utf-8"))["pairs"]["Encryptomatte"]["knobs"]
        entry = [e for e in mapping if "exportWrite" in e.get("nuke", [])]
        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0]["kind"], "unsupported")
        self.assertIn("Write", entry[0]["reason"])


if __name__ == "__main__":
    unittest.main()
