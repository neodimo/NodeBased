"""Lane 2 step N finish: Root's project format, proxy type and format, read-proxy-files setting and script directory.

Each field lives in `settings.project` (nodebased/projectsettings.py), is stored only when it is off its default, and changes
behaviour: the size a new Constant, Draw generator or Reformat takes, the viewer's proxy tier, which file a Read with a proxy
reads, and where a relative Read path resolves. Documents that never set one render and serialize as before."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from nodebased import projectsettings, readproxy
from nodebased.core import SCHEMA_VERSION, Dispatcher, demo_document, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_fidelity_f_readwrite import FolderCase, H, W, red_of

ROOT = Path(__file__).resolve().parents[1]


def project_set(dispatcher, **fields):
    return dispatcher.execute({"op": "settings", "settings": {"project": fields}})


class StoredOnlyWhenSetTests(unittest.TestCase):
    def test_new_fields_are_absent_from_a_document_that_never_sets_them(self):
        doc = demo_document()
        self.assertNotIn("project", doc["settings"])
        self.assertEqual({k: projectsettings.project(doc)[k] for k in
                          ("script_directory", "format", "proxy_type", "proxy_format", "read_proxy_files")},
                         {"script_directory": "", "format": "", "proxy_type": "scale", "proxy_format": "",
                          "read_proxy_files": "always"})

    def test_an_older_document_loads_with_the_defaults_and_no_schema_change_is_needed(self):
        old = copy.deepcopy(demo_document())
        old["settings"]["project"] = {"name": "old", "label": "", "project_directory": "", "lock_connections": False,
                                      "lock_range": False}
        upgraded = upgrade_document(copy.deepcopy(old))
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        validate(upgraded)
        self.assertEqual(projectsettings.project(upgraded)["read_proxy_files"], "always")

    def test_setting_a_field_back_to_its_default_drops_the_group_again(self):
        d = Dispatcher(empty_document())
        project_set(d, proxy_type="format")
        self.assertEqual(d.document["settings"]["project"]["proxy_type"], "format")
        project_set(d, proxy_type="scale")
        self.assertNotIn("project", d.document["settings"])

    def test_bad_values_are_refused(self):
        d = Dispatcher(empty_document())
        for fields in ({"proxy_type": "ratio"}, {"read_proxy_files": "sometimes"}, {"format": 3}, {"script_directory": 1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                project_set(d, **fields)
        with self.assertRaises(ValueError):
            project_set(d, format="Nowhere")
        with self.assertRaises(ValueError):
            project_set(d, proxy_format="Nowhere")
        self.assertNotIn("project", d.document["settings"])


class ProjectFormatTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher(empty_document())

    def size(self, key):
        return (self.d.document["nodes"][key]["params"]["width"], self.d.document["nodes"][key]["params"]["height"])

    def test_a_new_generator_takes_the_project_format(self):
        project_set(self.d, format="HD_720")
        for kind in ("Constant", "Checker", "Ramp", "Radial", "Rectangle", "Noise", "Grid", "Text"):
            with self.subTest(kind=kind):
                self.d.execute({"op": "create", "id": kind, "type": kind})
                self.assertEqual(self.size(kind), (1280, 720))

    def test_a_size_the_caller_names_is_left_alone(self):
        project_set(self.d, format="HD_720")
        self.d.execute({"op": "create", "id": "c", "type": "Constant", "params": {"width": 64}})
        self.assertEqual(self.size("c"), (64, 540))             # one named side: the node keeps its own default for the other

    def test_without_a_project_format_a_generator_keeps_its_own_size(self):
        self.d.execute({"op": "create", "id": "c", "type": "Constant"})
        self.assertEqual(self.size("c"), (960, 540))

    def test_a_new_reformat_goes_to_the_project_format_unless_it_names_one(self):
        project_set(self.d, format="UHD_4K")
        self.d.execute({"op": "create", "id": "a", "type": "Reformat"})
        self.d.execute({"op": "create", "id": "b", "type": "Reformat", "params": {"format": "HD_720"}})
        self.assertEqual(self.d.document["nodes"]["a"]["params"]["format"], "UHD_4K")
        self.assertEqual(self.size("a"), (3840, 2160))
        self.assertEqual(self.size("b"), (1280, 720))

    def test_an_existing_node_is_not_resized(self):
        self.d.execute({"op": "create", "id": "c", "type": "Constant"})
        project_set(self.d, format="HD_720")
        self.assertEqual(self.size("c"), (960, 540))

    def test_the_frame_a_new_constant_renders_has_the_project_formats_size(self):
        project_set(self.d, format="Square_1K")
        self.d.execute({"op": "create", "id": "c", "type": "Constant"})
        self.d.execute({"op": "view", "id": "c"})
        self.assertEqual(Evaluator().evaluate(self.d.document, "c", frame=1, tier=8).shape[:2], (128, 128))

    def test_the_format_follows_a_registry_rename_and_clears_on_delete(self):
        self.d.execute({"op": "format", "action": "set", "name": "Mine", "width": 100, "height": 50})
        project_set(self.d, format="Mine", proxy_format="Mine")
        self.d.execute({"op": "format", "action": "rename", "name": "Mine", "new_name": "Yours"})
        fields = projectsettings.project(self.d.document)
        self.assertEqual((fields["format"], fields["proxy_format"]), ("Yours", "Yours"))
        self.d.execute({"op": "format", "action": "delete", "name": "Yours"})
        fields = projectsettings.project(self.d.document)
        self.assertEqual((fields["format"], fields["proxy_format"]), ("", ""))
        validate(self.d.document)


class ProxyFormatTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher(empty_document())
        self.d.execute({"op": "create", "id": "c", "type": "Constant", "params": {"width": 64, "height": 32}})
        self.d.execute({"op": "view", "id": "c"})

    def tier(self, pick):
        return self.d.execute({"op": "viewer_proxy", "tier": pick})["result"]["tier"]

    def test_the_scale_type_keeps_the_pick(self):
        project_set(self.d, format="UHD_4K", proxy_format="Square_1K")
        self.assertEqual(self.tier(2), 2)

    def test_the_format_type_turns_any_proxy_pick_into_the_proxy_formats_tier(self):
        project_set(self.d, format="UHD_4K", proxy_type="format", proxy_format="HD_1080")      # 3840 / 1920
        self.assertEqual(self.tier(8), 2)
        project_set(self.d, proxy_format="Square_1K")                                          # 3840 / 1024 = 3.75
        self.assertEqual(self.d.document["settings"]["viewer"]["proxy"], 4)
        self.assertEqual(self.tier(1), 1)

    def test_the_picture_is_asked_for_at_that_tier(self):
        project_set(self.d, format="UHD_4K", proxy_type="format", proxy_format="HD_1080")
        tier = self.tier(4)
        from nodebased.core import viewer_proxy
        self.assertEqual(Evaluator().evaluate(self.d.document, "c", frame=1, tier=viewer_proxy(self.d.document)).shape[:2], (16, 32))
        self.assertEqual(tier, 2)

    def test_a_proxy_format_that_is_not_smaller_means_full_size(self):
        project_set(self.d, format="HD_1080", proxy_type="format", proxy_format="HD_1080")
        self.assertEqual(self.tier(2), 1)

    def test_without_a_project_format_the_proxy_format_is_measured_against_hd_1080(self):
        project_set(self.d, proxy_type="format", proxy_format="HD_720")                       # 1920 / 1280 = 1.5: nearest is 2
        self.assertEqual(projectsettings.proxy_tier(self.d.document, 2), 2)
        project_set(self.d, proxy_format="Square_1K")                                          # 1920 / 1024 = 1.9
        self.assertEqual(projectsettings.proxy_tier(self.d.document, 8), 2)

    def test_the_format_type_with_no_proxy_format_leaves_the_pick(self):
        project_set(self.d, proxy_type="format")
        self.assertEqual(self.tier(4), 4)


class ViewerProxyComboTests(unittest.TestCase):
    def test_the_proxy_combo_follows_the_tier_the_project_proxy_format_decides(self):
        from tests.test_2d_parity_feedback_y3 import close_window, make_window
        window = make_window(self)
        self.addCleanup(close_window, self)
        window.command({"op": "settings", "settings": {"project": {"format": "UHD_4K", "proxy_type": "format",
                                                                   "proxy_format": "HD_1080"}}})
        window.proxy.setCurrentIndex(window.proxy.findData(8))
        window.set_viewer_proxy(8)
        self.assertEqual(window.proxy.currentData(), 2)
        from nodebased.core import viewer_proxy
        self.assertEqual(viewer_proxy(window.dispatcher.document), 2)


class ReadProxyFilesTests(FolderCase):
    """Main file 24 x 16 (red 0.2); proxy file as wide as the test says (red 0.8)."""

    def doc(self, setting, proxy_width, tier_pick=None):
        self.exr("full.exr", 0.2)
        self.exr("small.exr", 0.8, width=proxy_width, height=proxy_width * H // W)
        d = Dispatcher(__import__("nodebased.core", fromlist=["empty_document"]).empty_document())
        d.execute({"op": "create", "id": "r", "type": "Read",
                   "params": {"path": str(self.dir / "full.exr"), "proxy": str(self.dir / "small.exr")}})
        d.execute({"op": "view", "id": "r"})
        if setting is not None:
            project_set(d, read_proxy_files=setting)
        return d.document

    def red(self, doc, tier):
        return red_of(doc, tier=tier)[0]

    def test_never_reads_the_main_file(self):
        self.assertAlmostEqual(self.red(self.doc("never", W // 2), 2), 0.2, places=6)

    def test_always_and_the_old_default_read_the_proxy_file_at_any_proxy_tier(self):
        for setting in (None, "always"):
            with self.subTest(setting=setting):
                self.assertAlmostEqual(self.red(self.doc(setting, 6), 2), 0.8, places=6)         # even a proxy smaller than the tier needs

    def test_if_larger_reads_the_proxy_file_only_when_it_is_at_least_the_size_the_tier_needs(self):
        self.assertAlmostEqual(self.red(self.doc("if larger", W // 2), 2), 0.8, places=6)           # 12 wide, 12 needed
        self.assertAlmostEqual(self.red(self.doc("if larger", W // 2), 4), 0.8, places=6)           # 12 wide, 6 needed
        self.assertAlmostEqual(self.red(self.doc("if larger", 6), 2), 0.2, places=6)                # 6 wide, 12 needed: main file

    def test_if_nearest_reads_whichever_file_is_nearer_the_size_the_tier_needs(self):
        self.assertAlmostEqual(self.red(self.doc("if nearest", 6), 2), 0.8, places=6)               # |6 - 12| < |24 - 12|
        self.assertAlmostEqual(self.red(self.doc("if nearest", 6), 1), 0.2, places=6)               # tier 1: no proxy at all
        self.assertAlmostEqual(self.red(self.doc("if nearest", 36), 2), 0.2, places=6)              # |36 - 12| > |24 - 12|: main file

    def test_the_decimated_main_file_has_the_tier_size_when_the_setting_skips_the_proxy(self):
        _value, raster = red_of(self.doc("never", W // 2), tier=2)
        self.assertEqual(raster.pixels.shape[:2], (H // 2, W // 2))

    def test_the_tile_path_gives_the_evaluators_pixels_for_every_setting(self):
        for setting in ("never", "if larger", "if nearest", "always"):
            with self.subTest(setting=setting):
                doc = self.doc(setting, 6)
                executor = TileExecutor(evaluator=Evaluator(), tile_edge=4)
                region = executor.canvas_region(doc, "r", frame=1, tier=2)
                got = executor.compose_region(doc, "r", region, frame=1, tier=2)
                np.testing.assert_allclose(got.pixels, red_of(doc, tier=2)[1].pixels, atol=1e-6)

    def test_active_is_the_one_decision(self):
        self.doc(None, 6)
        params = {"path": str(self.dir / "full.exr"), "proxy": str(self.dir / "small.exr")}
        self.assertTrue(readproxy.active(params, 2))
        self.assertFalse(readproxy.active(params, 2, "never"))
        self.assertFalse(readproxy.active(params, 1, "always"))
        self.assertFalse(readproxy.active({**params, "proxy": ""}, 2, "always"))
        self.assertFalse(readproxy.active(params, 2, "if larger"))
        self.assertTrue(readproxy.active(params, 2, "if nearest"))


class ScriptDirectoryTests(FolderCase):
    def test_a_relative_read_path_resolves_against_the_script_directory(self):
        self.exr("shots/plate.exr", 0.4)
        d = Dispatcher(empty_document())
        project_set(d, script_directory=str(self.dir))
        d.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": "shots/plate.exr"}})
        self.assertEqual(d.document["nodes"]["r"]["params"]["path"], str(self.dir / "shots" / "plate.exr"))
        d.execute({"op": "view", "id": "r"})
        self.assertAlmostEqual(red_of(d.document)[0], 0.4, places=6)

    def test_the_project_directory_wins_and_an_absolute_path_is_left_alone(self):
        d = Dispatcher(empty_document())
        other = self.dir / "other"
        other.mkdir()
        project_set(d, script_directory=str(self.dir), project_directory=str(other))
        d.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": "plate.exr"}})
        self.assertEqual(d.document["nodes"]["r"]["params"]["path"], str(other / "plate.exr"))
        d.execute({"op": "create", "id": "s", "type": "Read", "params": {"path": str(self.dir / "abs.exr")}})
        self.assertEqual(d.document["nodes"]["s"]["params"]["path"], str(self.dir / "abs.exr"))


class RootMappingTests(unittest.TestCase):
    def test_the_mapping_pairs_the_new_fields_and_lists_the_rest_as_unsupported(self):
        pair = json.loads((ROOT / "docs/parity/nodebased_to_nuke.json").read_text())["pairs"]["Root"]
        by_knob = {e["nuke"][0]: e for e in pair["knobs"]}
        expected = {"Script_directory": "settings.project.script_directory", "format": "settings.project.format",
                    "proxy_type": "settings.project.proxy_type", "proxy_format": "settings.project.proxy_format",
                    "proxySetting": "settings.project.read_proxy_files"}
        for knob, field in expected.items():
            self.assertEqual(by_knob[knob]["nodebased"], [field], knob)
            self.assertNotEqual(by_knob[knob]["kind"], "unsupported", knob)
        for knob in ("is_live", "render_mode"):
            self.assertEqual(by_knob[knob]["kind"], "unsupported")
            self.assertTrue(by_knob[knob]["reason"])


if __name__ == "__main__":
    unittest.main()
