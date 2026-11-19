"""Lane 2 step X3 part 2: the Views tab of Project Settings (Nuke's Root Views tab) over `settings.views`.

A project starts with one view, `main`. The tab lists the views, adds, removes and moves them, names the hero view, sets up left and
right for stereo, and has `View selection uses buttons?` and `Use colors in UI?`. The views are recorded and saved; rendering stays mono.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QListWidget, QPushButton, QTabWidget, QWidget

import tests.isolation  # noqa: F401
from nodebased import core, projectviews, rootpanel
from nodebased.app import Window
from nodebased.core import Dispatcher, demo_document, load_document
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_c4 import Graph
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]


class ViewsModelTests(unittest.TestCase):
    def test_a_new_project_has_one_view_main_and_stores_nothing(self):
        doc = demo_document()
        self.assertNotIn("views", doc["settings"])
        views = projectviews.views(doc)
        self.assertEqual((views["names"], views["hero"], views["buttons"], views["use_colours"]), (["main"], "main", True, False))

    def test_the_defaults_are_the_guides(self):
        guide = json.loads((ROOT / "docs/parity/nuke_nodes.json").read_text())["nodes"]["Root"]["knobs"]
        default = {k["name"]: k["default"] for k in guide}
        self.assertEqual(default["views#5"], "main")
        self.assertEqual(default["hero_view"], projectviews.DEFAULTS["hero"])
        self.assertIs(default["views_button"], projectviews.DEFAULTS["buttons"])
        self.assertIs(default["views_colours"], projectviews.DEFAULTS["use_colours"])

    def test_add_remove_move_and_the_only_view_stays(self):
        d = Dispatcher()
        d.execute(rootpanel.edit_views(d.document, "add", name="alt"))
        self.assertEqual(projectviews.views(d.document)["names"], ["main", "alt"])
        d.execute(rootpanel.edit_views(d.document, "move", name="alt", step=-1))
        self.assertEqual(projectviews.views(d.document)["names"], ["alt", "main"])
        self.assertEqual(projectviews.views(d.document)["colours"][1], "#ffffff")           # main keeps its own colour through the move
        self.assertIsNone(rootpanel.edit_views(d.document, "move", name="alt", step=-1))     # already at the top
        self.assertIsNone(rootpanel.edit_views(d.document, "add", name="alt"))               # a name is used once
        d.execute(rootpanel.edit_views(d.document, "remove", name="alt"))
        self.assertEqual(projectviews.views(d.document)["names"], ["main"])
        self.assertIsNone(rootpanel.edit_views(d.document, "remove", name="main"))           # the only view cannot be removed
        self.assertNotIn("views", d.document["settings"])                                    # back at the defaults: stored as before

    def test_removing_the_hero_hands_the_role_to_the_first_view_left(self):
        d = Dispatcher()
        d.execute(rootpanel.edit_views(d.document, "add", name="alt"))
        d.execute(rootpanel.edit(d.document, "settings.views.hero", "alt"))
        self.assertEqual(projectviews.views(d.document)["hero"], "alt")
        d.execute(rootpanel.edit_views(d.document, "remove", name="alt"))
        self.assertEqual(projectviews.views(d.document)["hero"], "main")

    def test_set_up_views_for_stereo(self):
        d = Dispatcher()
        d.execute(rootpanel.edit_views(d.document, "stereo"))
        views = projectviews.views(d.document)
        self.assertEqual((views["names"], views["hero"]), (["left", "right"], "left"))
        self.assertEqual(len(views["colours"]), 2)
        self.assertIsNone(rootpanel.edit_views(d.document, "stereo"))

    def test_the_flags_edit_through_the_one_operation(self):
        d = Dispatcher()
        command = rootpanel.edit(d.document, "settings.views.use_colours", True)
        self.assertEqual(command, {"op": "settings", "settings": {"views": {"use_colours": True}}})
        d.execute(command)
        self.assertTrue(projectviews.views(d.document)["use_colours"])
        self.assertIsNone(rootpanel.edit(d.document, "settings.views.use_colours", True))
        d.execute(rootpanel.edit(d.document, "settings.views.buttons", False))
        self.assertFalse(projectviews.views(d.document)["buttons"])
        d.execute({"op": "undo"})
        self.assertTrue(projectviews.views(d.document)["buttons"])

    def test_bad_values_are_refused(self):
        d = Dispatcher()
        for changes in ({"names": []}, {"names": ["a", "a"], "colours": ["#ffffff", "#ffffff"]}, {"names": ["a"], "colours": ["red"]},
                        {"hero": "nobody"}, {"use_colours": "yes"}, {"surprise": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                d.execute({"op": "settings", "settings": {"views": changes}})
        self.assertNotIn("views", d.document["settings"])

    def test_the_views_save_and_load(self):
        d = Dispatcher()
        d.execute(rootpanel.edit_views(d.document, "stereo"))
        d.execute(rootpanel.edit(d.document, "settings.views.use_colours", True))
        d.execute(rootpanel.edit(d.document, "settings.views.hero", "right"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.json"
            path.write_text(json.dumps(d.document))
            loaded = load_document(path)
        self.assertEqual(loaded["settings"]["views"], d.document["settings"]["views"])
        self.assertEqual(projectviews.views(loaded)["hero"], "right")

    def test_an_older_document_loads_with_the_default_views(self):
        old = copy.deepcopy(demo_document())
        old["version"] = 36
        upgraded = core.upgrade_document(old)
        core.validate(upgraded)
        self.assertEqual(projectviews.views(upgraded)["names"], ["main"])
        self.assertEqual(json.dumps(upgraded["settings"], sort_keys=True), json.dumps(demo_document()["settings"], sort_keys=True))

    def test_views_change_no_pixels(self):
        g = Graph()
        g.add("plate", "Constant", dict(width=8, height=8, red=0.3, green=0.5, blue=0.7, alpha=1.0))
        g.add("blur", "Blur", dict(), image="plate")
        doc = dict(g.doc, view="blur")
        mono = Evaluator().evaluate(json.loads(json.dumps(doc)), "blur", frame=1)
        staged = json.loads(json.dumps(doc))
        staged["settings"] = copy.deepcopy(staged["settings"])
        projectviews.merge(staged["settings"], {"names": ["left", "right"], "colours": ["#ff0000", "#00ff00"], "hero": "right", "use_colours": True})
        np.testing.assert_array_equal(Evaluator().evaluate(staged, "blur", frame=1), mono)


class ViewsTabTests(unittest.TestCase):
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
        self.window.command({"op": "settings", "settings": {"views": {"names": ["main"], "colours": ["#ffffff"], "hero": "main",
                                                                      "buttons": True, "use_colours": False}}})
        APP.processEvents()

    def panel(self):
        self.window.open_project_settings_panel()
        return self.window.properties.findChild(QWidget, "project-settings-panel")

    def views(self):
        return projectviews.views(self.window.dispatcher.document)

    def settle(self, check):
        wait_until(check)
        APP.processEvents()

    def test_the_tab_follows_the_colour_tab_and_lists_main(self):
        panel = self.panel()
        tabs = panel.findChild(QTabWidget, "project-settings-tabs")
        self.assertEqual([tabs.tabText(i) for i in range(tabs.count())], ["Root", "Color", "Views"])
        listing = panel.findChild(QListWidget, "root-views-list")
        self.assertEqual([listing.item(i).text() for i in range(listing.count())], ["main"])
        self.assertEqual(panel.findChild(QComboBox, "root-hero_view").currentText(), "main")
        for name in ("root-views-add", "root-views-remove", "root-views-up", "root-views-down", "root-setlr"):
            self.assertIsNotNone(panel.findChild(QPushButton, name), name)
        self.assertTrue(panel.findChild(QCheckBox, "root-views_button").isChecked())
        self.assertFalse(panel.findChild(QCheckBox, "root-views_colours").isChecked())

    def test_the_tooltip_says_the_views_are_recorded_and_rendering_is_mono(self):
        panel = self.panel()
        for name in ("root-views", "root-hero_view", "root-setlr", "root-views_button", "root-views_colours"):
            tip = panel.findChild(QWidget, name).toolTip()
            self.assertIn("mono", tip, name)
            self.assertIn("recorded", tip, name)

    def test_the_buttons_add_move_and_remove_in_one_undo_step_each(self):
        w = self.window
        self.panel()
        start = w.dispatcher.revision
        w._root_views("add", name="alt")
        self.settle(lambda: self.views()["names"] == ["main", "alt"])
        self.assertEqual(w.dispatcher.revision, start + 1)
        listing = self.panel().findChild(QListWidget, "root-views-list")
        self.assertEqual([listing.item(i).text() for i in range(listing.count())], ["main", "alt"])
        listing.setCurrentRow(1)
        self.panel().findChild(QPushButton, "root-views-up").click()
        self.settle(lambda: self.views()["names"] == ["alt", "main"])
        listing = self.panel().findChild(QListWidget, "root-views-list")
        listing.setCurrentRow(0)
        self.panel().findChild(QPushButton, "root-views-remove").click()
        self.settle(lambda: self.views()["names"] == ["main"])
        self.assertEqual(w.dispatcher.revision, start + 3)
        w.command({"op": "undo"})
        self.settle(lambda: self.views()["names"] == ["alt", "main"])

    def test_the_hero_menu_lists_the_views_and_sets_the_hero(self):
        w = self.window
        w._root_views("add", name="alt")
        self.settle(lambda: len(self.views()["names"]) == 2)
        combo = self.panel().findChild(QComboBox, "root-hero_view")
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["main", "alt"])
        combo.setCurrentIndex(1)
        combo.activated.emit(1)
        self.settle(lambda: self.views()["hero"] == "alt")
        self.assertEqual(self.views()["hero"], "alt")

    def test_use_colours_shows_each_views_colour_in_the_list(self):
        w = self.window
        w._root_views("stereo")
        self.settle(lambda: self.views()["names"] == ["left", "right"])
        listing = self.panel().findChild(QListWidget, "root-views-list")
        plain = listing.item(0).foreground().color()
        self.assertNotEqual(plain, QColor("#ff0000"))                      # colours stay off until `Use colors in UI?` is on
        box = self.panel().findChild(QCheckBox, "root-views_colours")
        box.setChecked(True)
        self.settle(lambda: self.views()["use_colours"])
        listing = self.panel().findChild(QListWidget, "root-views-list")
        self.assertEqual(listing.item(0).foreground().color(), QColor("#ff0000"))
        self.assertEqual(listing.item(1).foreground().color(), QColor("#00ff00"))

    def test_the_stereo_button_sets_up_left_and_right(self):
        self.panel().findChild(QPushButton, "root-setlr").click()
        self.settle(lambda: self.views()["names"] == ["left", "right"])
        self.assertEqual(self.views()["hero"], "left")


class ViewsLayoutTests(unittest.TestCase):
    def test_rows_follow_the_guides_views_tab(self):
        guide = json.loads((ROOT / "docs/parity/nuke_nodes.json").read_text())["nodes"]["Root"]["knobs"]
        views_tab = [k for k in guide if k["tab"] == "Views Tab"]
        label = {k["name"]: k["label"] for k in views_tab}
        rows = [r for r in rootpanel.ROWS if r.tab == "Views"]
        self.assertEqual([r.names[0] for r in rows], ["views#5", "hero_view", "setlr", "views_button", "views_colours"])
        for row in rows:
            self.assertEqual(label[row.names[0]], row.label)
        order = [k["name"] for k in guide]
        self.assertEqual(sorted(r.names[0] for r in rows), sorted(r.names[0] for r in rows))
        self.assertEqual([order.index(r.names[0]) for r in rows], sorted(order.index(r.names[0]) for r in rows))
        self.assertEqual(rootpanel.TABS, ("Root", "Color", "Views"))

    def test_every_views_field_exists_in_a_new_document(self):
        document = demo_document()
        for row in rootpanel.ROWS:
            for path in row.fields:
                rootpanel.get(document, path)


if __name__ == "__main__":
    unittest.main()
