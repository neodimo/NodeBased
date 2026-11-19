"""Lane 2 step X3 part 1: the OCIO config of Project Settings is a menu, like Nuke 17.0's Root `OCIO config` knob.

The menu lists the configs Nuke ships, the built-in configs NodeBased can load, then `custom` with its path field. A document that
stored a path keeps it and opens on `custom` (schema 37). A config NodeBased cannot load warns, and the colour-space menus fall back
to the built-in list.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QWidget

import tests.isolation  # noqa: F401
from nodebased import core, ocioconfigs, rootpanel
from nodebased.app import Window
from nodebased.core import Dispatcher, demo_document, load_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor
from tests.test_2d_parity_group_c4 import Graph
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


def write_config(directory):
    import PyOpenColorIO as ocio
    path = Path(directory) / "custom.ocio"
    ocio.Config.CreateFromBuiltinConfig(ocioconfigs.DEFAULT).serialize(str(path))
    return str(path)


class MenuEntriesTests(unittest.TestCase):
    def test_nukes_shipped_configs_come_first_then_the_built_ins_then_custom(self):
        ids = ocioconfigs.entry_ids()
        self.assertEqual(ids[:len(ocioconfigs.NUKE_SHIPPED)], ocioconfigs.NUKE_SHIPPED)
        self.assertEqual(ids[-1], "custom")
        self.assertIn("nuke-default", ids)                               # the Reference Guide's default for the knob
        self.assertEqual(ocioconfigs.NUKE_SHIPPED[-1], ocioconfigs.NUKE_DEFAULT)
        self.assertEqual(len(ids), len(set(ids)))
        built_in = ids[len(ocioconfigs.NUKE_SHIPPED):-1]
        self.assertTrue(built_in and all(uri.startswith("ocio://") for uri in built_in))
        self.assertEqual(built_in[0], ocioconfigs.DEFAULT)

    def test_the_new_project_opens_on_a_built_in_with_no_warning(self):
        color = demo_document()["settings"]["color"]
        self.assertEqual(ocioconfigs.selection(color), (ocioconfigs.DEFAULT, ""))
        self.assertIsNone(ocioconfigs.problem(color))
        self.assertNotIn("custom_config", color)

    def test_every_built_in_entry_loads(self):
        for uri in ocioconfigs.builtin_names():
            with self.subTest(uri=uri):
                self.assertIsNone(ocioconfigs.problem({"config": uri}))

    def test_a_config_nodebased_cannot_load_warns_and_the_menus_fall_back(self):
        for name in ocioconfigs.NUKE_SHIPPED:
            color = {"config": name}
            with self.subTest(name=name):
                self.assertIn(name, ocioconfigs.problem(color))
                self.assertEqual(ocioconfigs.choices_config(color), ocioconfigs.DEFAULT)

    def test_custom_warns_until_it_names_a_loadable_file(self):
        self.assertIn("Enter the path", ocioconfigs.problem({"config": "custom"}))
        self.assertIn("does not exist", ocioconfigs.problem({"config": "custom", "custom_config": "/nowhere/x.ocio"}))
        with tempfile.TemporaryDirectory() as directory:
            good = write_config(directory)
            self.assertIsNone(ocioconfigs.problem({"config": "custom", "custom_config": good}))
            self.assertEqual(ocioconfigs.choices_config({"config": "custom", "custom_config": good}), good)
            broken = Path(directory) / "broken.ocio"
            broken.write_text("this is not a config")
            self.assertIn("Could not load", ocioconfigs.problem({"config": "custom", "custom_config": str(broken)}))


class DocumentMigrationTests(unittest.TestCase):
    def old_document(self, config):
        doc = copy.deepcopy(demo_document())
        doc["version"] = 36
        doc["settings"]["color"]["config"] = config
        return doc

    def test_a_stored_path_keeps_its_value_and_opens_on_custom(self):
        upgraded = core.upgrade_document(self.old_document("/studio/show.ocio"))
        self.assertEqual(upgraded["version"], core.SCHEMA_VERSION)
        self.assertEqual(upgraded["settings"]["color"]["config"], "custom")
        self.assertEqual(upgraded["settings"]["color"]["custom_config"], "/studio/show.ocio")
        core.validate(upgraded)
        self.assertEqual(ocioconfigs.selection(upgraded["settings"]["color"]), ("custom", "/studio/show.ocio"))
        self.assertEqual(ocioconfigs.effective(upgraded["settings"]["color"]), "/studio/show.ocio")

    def test_an_unknown_uri_becomes_a_custom_path_too(self):
        upgraded = core.upgrade_document(self.old_document("ocio://not-a-built-in"))
        self.assertEqual((upgraded["settings"]["color"]["config"], upgraded["settings"]["color"]["custom_config"]),
                         ("custom", "ocio://not-a-built-in"))

    def test_a_built_in_document_is_untouched_by_the_migration(self):
        for config in (ocioconfigs.DEFAULT, "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5"):
            upgraded = core.upgrade_document(self.old_document(config))
            self.assertEqual(upgraded["settings"]["color"]["config"], config)
            self.assertNotIn("custom_config", upgraded["settings"]["color"])

    def test_an_old_file_loads_from_disk_and_saves_in_the_new_form(self):
        with tempfile.TemporaryDirectory() as directory:
            old = self.old_document("/studio/show.ocio")
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(old))
            doc = load_document(path)
            self.assertEqual(doc["settings"]["color"]["custom_config"], "/studio/show.ocio")
            again = Path(directory) / "again.json"
            again.write_text(json.dumps(doc))
            self.assertEqual(load_document(again)["settings"]["color"], doc["settings"]["color"])

    def test_the_custom_path_pixels_match_the_old_path_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_config(directory)
            g = Graph()
            g.add("plate", "Constant", dict(width=8, height=8, red=0.4, green=0.2, blue=0.1, alpha=1.0))
            g.add("cs", "OCIOColorspace", dict(src="ACEScg", dst="sRGB - Texture"), image="plate")
            old = json.loads(json.dumps(dict(g.doc, view="cs")))
            old["version"] = 36
            old["settings"] = copy.deepcopy(demo_document()["settings"])
            old["settings"]["color"]["config"] = config
            before = Evaluator().evaluate(json.loads(json.dumps(old)), "cs", frame=1)
            upgraded = core.upgrade_document(old)
            self.assertEqual(upgraded["settings"]["color"]["config"], "custom")
            after = Evaluator().evaluate(upgraded, "cs", frame=1)
            np.testing.assert_array_equal(before, after)
            tiled = TileExecutor(evaluator=Evaluator()).compose(dict(upgraded, view="cs"), "cs", frame=1, tier=1).pixels
            np.testing.assert_allclose(tiled, after, atol=1e-6)      # the tile path reads the same custom config

    def test_the_project_config_reaches_a_node_and_a_nuke_config_fails_it_clearly(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=4, height=4, red=0.5, green=0.5, blue=0.5, alpha=1.0))
        g.add("cs", "OCIOColorspace", dict(), image="plate")
        doc = dict(g.doc, view="cs")
        Evaluator().evaluate(doc, "cs", frame=1)
        broken = json.loads(json.dumps(doc))
        broken["settings"]["color"]["config"] = "nuke-default"
        with self.assertRaisesRegex(Exception, "NodeBased does not ship"):
            Evaluator().evaluate(broken, "cs", frame=1)

    def test_the_settings_operation_edits_the_menu_and_the_path(self):
        d = Dispatcher()
        d.execute({"op": "settings", "settings": {"color": {"config": "custom", "custom_config": "/studio/show.ocio"}}})
        self.assertEqual(d.document["settings"]["color"]["custom_config"], "/studio/show.ocio")
        d.execute({"op": "settings", "settings": {"color": {"config": ocioconfigs.DEFAULT, "custom_config": ""}}})
        self.assertNotIn("custom_config", d.document["settings"]["color"])
        d.execute({"op": "undo"})
        self.assertEqual(d.document["settings"]["color"]["custom_config"], "/studio/show.ocio")


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = Window()
        cls.window.show()
        assert wait_until(lambda: cls.window.frame is not None)

    @classmethod
    def tearDownClass(cls):
        cls.window.saved_document = cls.window.dispatcher.document
        cls.window.close()
        cls.window.deleteLater()
        APP.processEvents()

    def setUp(self):
        self.window.clear_panels()
        self.window.command({"op": "settings", "settings": {"color": {"config": ocioconfigs.DEFAULT, "custom_config": ""}}})
        APP.processEvents()

    def panel(self):
        self.window.open_project_settings_panel()
        return self.window.properties.findChild(QWidget, "project-settings-panel")

    def set_config(self, entry):
        combo = self.panel().findChild(QComboBox, "root-OCIO_config")
        combo.setCurrentIndex(combo.findData(entry))
        combo.activated.emit(combo.currentIndex())
        wait_until(lambda: rootpanel.get(self.window.dispatcher.document, rootpanel.COLOR_CONFIG) == entry)
        APP.processEvents()

    def test_the_menu_lists_the_entries_in_order_and_opens_on_the_projects_config(self):
        combo = self.panel().findChild(QComboBox, "root-OCIO_config")
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], list(ocioconfigs.entry_ids()))
        self.assertEqual(combo.currentData(), ocioconfigs.DEFAULT)
        self.assertIsNone(self.panel().findChild(QLabel, "root-OCIO_config-warning"))
        self.assertIsNone(self.panel().findChild(QLineEdit, "root-customOCIOConfigPath"))

    def test_choosing_custom_shows_the_path_field_below_the_menu_and_the_path_is_stored(self):
        self.set_config("custom")
        panel = self.panel()
        field = panel.findChild(QLineEdit, "root-customOCIOConfigPath")
        self.assertIsNotNone(field)
        self.assertIsNotNone(panel.findChild(QLabel, "root-OCIO_config-warning"))      # no path yet
        with tempfile.TemporaryDirectory() as directory:
            config = write_config(directory)
            field.setText(config)
            field.editingFinished.emit()
            wait_until(lambda: self.window.dispatcher.document["settings"]["color"].get("custom_config") == config)
            APP.processEvents()
            self.assertEqual(self.window.dispatcher.document["settings"]["color"]["custom_config"], config)
            self.assertIsNone(self.panel().findChild(QLabel, "root-OCIO_config-warning"))
        self.set_config(ocioconfigs.DEFAULT)
        self.assertIsNone(self.panel().findChild(QLineEdit, "root-customOCIOConfigPath"))
        self.assertIn("custom_config", self.window.dispatcher.document["settings"]["color"])     # the path is remembered

    def test_a_config_nodebased_cannot_load_shows_a_warning(self):
        self.set_config("aces_1.2")
        note = self.panel().findChild(QLabel, "root-OCIO_config-warning")
        self.assertIsNotNone(note)
        self.assertIn("aces_1.2", note.text())

    def test_a_menu_edit_is_one_undo_step(self):
        start = self.window.dispatcher.revision
        self.set_config("nuke-default")
        self.assertEqual(self.window.dispatcher.revision, start + 1)
        self.window.command({"op": "undo"})
        APP.processEvents()
        self.assertEqual(self.window.dispatcher.document["settings"]["color"]["config"], ocioconfigs.DEFAULT)

    def test_the_colour_space_menus_fall_back_to_the_built_in_list(self):
        w = self.window
        w.dispatcher.execute({"op": "create", "id": "x3cs", "type": "OCIOColorspace", "pos": [0, 0]})
        try:
            w.command({"op": "settings", "settings": {"color": {"config": "nuke-default"}}})
            APP.processEvents()
            w.set_properties_widget(w.build_node_panel("x3cs"))
            source = w.properties.widget().findChild(QComboBox, "ocio-src-choice")
            self.assertIn("ACEScg", [source.itemText(i) for i in range(source.count())])
        finally:
            w.command({"op": "settings", "settings": {"color": {"config": ocioconfigs.DEFAULT}}})
            w.dispatcher.execute({"op": "delete", "id": "x3cs"})


if __name__ == "__main__":
    unittest.main()
