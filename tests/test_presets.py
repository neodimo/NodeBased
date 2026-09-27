import json
import unittest
from pathlib import Path

from nodebased import presets


def _write(directory, stem, name="A preset", category="Particles", description="", ops=None,
          thumbnail=None):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "category": category, "description": description,
                "ops": ops if ops is not None else [{"op": "create", "id": "$new:a", "type": "Sphere3D"}]}
    if thumbnail:
        manifest["thumbnail"] = thumbnail
    (directory / f"{stem}.json").write_text(json.dumps(manifest))


class PresetLoadingTests(unittest.TestCase):
    def test_loads_presets_from_a_shipped_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            shipped_dir = Path(tmp)
            _write(shipped_dir, "one", name="One")
            loaded, errors = presets.load_all(user_directory=Path("/nonexistent-user-dir"),
                                              shipped_directory=shipped_dir)
            self.assertFalse(errors)
            self.assertTrue(loaded)
            self.assertTrue(all(preset.category for preset in loaded))
            self.assertTrue(all(preset.ops for preset in loaded))

    def test_user_and_shipped_directories_both_load(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as tmp2:
            user_dir, shipped_dir = Path(tmp), Path(tmp2)
            _write(user_dir, "mine", name="Mine")
            _write(shipped_dir, "theirs", name="Theirs")
            loaded, errors = presets.load_all(user_directory=user_dir, shipped_directory=shipped_dir)
            self.assertFalse(errors)
            names = sorted(preset.name for preset in loaded)
            self.assertEqual(names, ["Mine", "Theirs"])
            by_name = {preset.name: preset for preset in loaded}
            self.assertFalse(by_name["Theirs"].user)
            self.assertTrue(by_name["Mine"].user)

    def test_bad_manifest_is_an_error_not_a_crash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "broken.json").write_text("{not json")
            (directory / "missing_name.json").write_text(json.dumps({"category": "X", "ops": []}))
            loaded, errors = presets.load_all(user_directory=directory,
                                              shipped_directory=Path("/nonexistent-shipped-dir"))
            self.assertFalse(loaded)
            self.assertEqual(len(errors), 2)

    def test_thumbnail_resolved_next_to_manifest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write(directory, "withthumb", thumbnail="pic.png")
            loaded, errors = presets.load_all(user_directory=directory,
                                              shipped_directory=Path("/nonexistent-shipped-dir"))
            self.assertFalse(errors)
            self.assertEqual(loaded[0].thumbnail, directory / "pic.png")


class PresetSearchTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        directory = Path(self._tmp.name)
        _write(directory, "sparks", name="Sparks", category="Particles", description="fast and hot")
        _write(directory, "dust", name="Dust puff", category="Particles", description="a soft cloud")
        _write(directory, "wrap", name="Shrinkwrap", category="Modeling", description="wraps a mesh")
        self.loaded, errors = presets.load_all(user_directory=Path("/nonexistent"), shipped_directory=directory)
        self.assertFalse(errors)

    def tearDown(self):
        self._tmp.cleanup()

    def test_categories_first_seen_order(self):
        self.assertEqual(presets.categories(self.loaded), ["Particles", "Modeling"])

    def test_search_by_category(self):
        found = presets.search(self.loaded, category="Modeling")
        self.assertEqual([p.name for p in found], ["Shrinkwrap"])

    def test_search_by_text_matches_name_or_description(self):
        found = {p.name for p in presets.search(self.loaded, text="soft")}
        self.assertEqual(found, {"Dust puff"})
        found = {p.name for p in presets.search(self.loaded, text="sparks")}
        self.assertEqual(found, {"Sparks"})

    def test_empty_search_matches_everything(self):
        self.assertEqual(len(presets.search(self.loaded, text="")), 3)


class BuildOpsTests(unittest.TestCase):
    def test_placeholders_resolved_against_selection(self):
        preset = presets.Preset(
            id="wrap", path=Path("/x/wrap.json"), name="Wrap", category="Modeling", description="",
            thumbnail=None, user=False,
            ops=[{"op": "create", "id": "$new:w", "type": "Shrinkwrap3D"},
                 {"op": "connect", "id": "$new:w", "input": "target", "source": "$selected[0]"}])
        nodes = {"sel1": {"type": "Sphere3D", "params": {}, "inputs": {}, "pos": [0, 0]}}
        ops = presets.build_ops(preset, ["sel1"], nodes, _Pos(10, 20))
        self.assertEqual(ops[0]["op"], "create")
        self.assertEqual(ops[0]["type"], "Shrinkwrap3D")
        self.assertEqual(ops[0]["pos"], [10, 20])
        self.assertEqual(ops[1]["source"], "sel1")

    def test_two_new_id_placeholders_resolve_the_same_within_one_build(self):
        preset = presets.Preset(
            id="chain", path=Path("/x/chain.json"), name="Chain", category="Particles", description="",
            thumbnail=None, user=False,
            ops=[{"op": "create", "id": "$new:a", "type": "ParticleEmitter3D"},
                 {"op": "create", "id": "$new:b", "type": "ParticleGravity3D"},
                 {"op": "connect", "id": "$new:b", "input": "particles", "source": "$new:a"}])
        ops = presets.build_ops(preset, [], {}, _Pos(0, 0))
        self.assertEqual(ops[2]["source"], ops[0]["id"])


class _Pos:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


class SaveSelectionAsPresetTests(unittest.TestCase):
    def test_round_trips_into_the_browser(self):
        import tempfile
        nodes = {
            "n1": {"type": "Sphere3D", "params": {"sphere_radius": 1.0}, "inputs": {}, "pos": [0, 0]},
            "n2": {"type": "ParticleEmitter3D", "params": {"emit_from": "surface"},
                  "inputs": {"geo": "n1"}, "pos": [100, 0]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = presets.save_selection_as_preset(
                directory, "my_combo", "My combo", "Particles", nodes, ["n1", "n2"],
                description="a sphere feeding an emitter")
            self.assertTrue(path.is_file())
            loaded, errors = presets.load_all(user_directory=directory,
                                              shipped_directory=Path("/nonexistent"))
            self.assertFalse(errors)
            self.assertEqual(len(loaded), 1)
            saved = loaded[0]
            self.assertEqual(saved.name, "My combo")
            self.assertTrue(saved.user)
            create_ops = [op for op in saved.ops if op["op"] == "create"]
            self.assertEqual({op["type"] for op in create_ops}, {"Sphere3D", "ParticleEmitter3D"})
            connect_ops = [op for op in saved.ops if op["op"] == "connect"]
            self.assertEqual(len(connect_ops), 1)
            self.assertEqual(connect_ops[0]["input"], "geo")

    def test_root_input_from_outside_selection_becomes_selected_placeholder(self):
        import tempfile
        nodes = {
            "outside": {"type": "Sphere3D", "params": {}, "inputs": {}, "pos": [0, 0]},
            "n1": {"type": "ParticleBounce3D", "params": {}, "inputs": {"geometry": "outside"},
                  "pos": [100, 0]},
        }
        ops = presets.capture_selection_ops(nodes, ["n1"])
        connect_ops = [op for op in ops if op["op"] == "connect"]
        self.assertEqual(len(connect_ops), 1)
        self.assertEqual(connect_ops[0]["source"], "$selected[0]")


if __name__ == "__main__":
    unittest.main()
