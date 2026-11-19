"""Lane 2 step X2 part 1: the Project Settings panel (Nuke's Root panel) in the Properties bin.

S in the graph and Edit > Project Settings open it; its tabs, rows and widget kinds are `nodebased/rootpanel.py`'s, in the Reference
Guide's order; each edit goes through the operation the setting already used and is one undo step.
"""
import json
import unittest
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QTabWidget, QWidget

import tests.isolation  # noqa: F401
from nodebased import rootpanel
from nodebased.app import Window
from nodebased.core import Dispatcher, demo_document
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]


class RootPanelLayoutTests(unittest.TestCase):
    def test_rows_follow_the_guides_order_and_labels(self):
        nodes = json.loads((ROOT / "docs/parity/nuke_nodes.json").read_text())
        nodes = nodes["nodes"] if "nodes" in nodes else nodes
        guide = nodes["Root"]["knobs"]
        order = {k["name"]: i for i, k in enumerate(guide)}
        label = {k["name"]: k["label"] for k in guide}
        tab = {k["name"]: k["tab"] for k in guide}
        for row in rootpanel.ROWS:
            for name in row.names:
                self.assertIn(name, order, name)
                self.assertEqual(label[name], row.label, name)
        for first, second in zip(rootpanel.ROWS, rootpanel.ROWS[1:]):
            self.assertLess(order[first.names[0]], order[second.names[0]], (first.label, second.label))
        self.assertEqual(rootpanel.TABS[0], "Root")
        self.assertEqual([r.label for r in rootpanel.ROWS if r.tab == "Root"][:5],
                         ["name", "project directory", "Script Directory", "comment", "lock all connections"])
        self.assertTrue(all(tab[n] in ("Root Tab", "Color Tab", "default LUT settings", "Views Tab") for r in rootpanel.ROWS for n in r.names))

    def test_every_row_field_exists_in_a_new_document(self):
        document = demo_document()
        for row in rootpanel.ROWS:
            for path in row.fields:
                rootpanel.get(document, path)

    def test_edits_are_single_operations_and_no_ops_return_none(self):
        doc = demo_document()
        self.assertIsNone(rootpanel.edit(doc, "settings.project.name", ""))
        self.assertEqual(rootpanel.edit(doc, "settings.project.name", "shot"), {"op": "settings", "settings": {"project": {"name": "shot"}}})
        self.assertEqual(rootpanel.edit(doc, "time.fps", 25.0), {"op": "time", "fps": 25.0})
        self.assertEqual(rootpanel.edit(doc, "settings.color.view", "Linear"), {"op": "settings", "settings": {"color": {"view": "Linear"}}})
        self.assertEqual(rootpanel.edit_proxy(doc, on=True), {"op": "viewer_proxy", "tier": 2})
        self.assertEqual(rootpanel.edit_proxy(doc, scale=0.25), {"op": "viewer_proxy", "tier": 4})
        self.assertIsNone(rootpanel.edit_proxy(doc, on=False))


class RootPanelWindowTests(unittest.TestCase):
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

    def panel(self):
        return self.window.properties.findChild(QWidget, "project-settings-panel")

    def test_s_and_the_edit_menu_open_the_panel_in_the_bin(self):
        w = self.window
        self.assertIsNone(self.panel())
        w.graph.setFocus()
        QTest.keyClick(w.graph, Qt.Key.Key_S)
        APP.processEvents()
        self.assertIsNotNone(self.panel())
        w.clear_panels()
        action = next(a for a in w.findChildren(QAction) if a.objectName() == "project-settings-action")
        action.trigger()
        APP.processEvents()
        self.assertIsNotNone(self.panel())
        tabs = self.panel().findChild(QTabWidget, "project-settings-tabs")
        self.assertEqual([tabs.tabText(i) for i in range(tabs.count())], list(rootpanel.TABS))

    def test_widget_kinds_follow_the_guide(self):
        self.window.open_project_settings_panel()
        panel = self.panel()
        kinds = {"name": QLineEdit, "lock_connections": QCheckBox, "first_frame": QSpinBox, "last_frame": QSpinBox, "fps": QDoubleSpinBox,
                 "format": QComboBox, "proxy": QCheckBox, "proxy_type": QComboBox, "proxy_scale": QDoubleSpinBox,
                 "proxySetting": QComboBox, "monitorLUT": QComboBox, "OCIO_config": QComboBox}
        for name, kind in kinds.items():
            self.assertIsInstance(panel.findChild(QWidget, f"root-{name}"), kind, name)

    def test_an_edit_is_one_undo_step(self):
        w = self.window
        w.open_project_settings_panel()
        start = w.dispatcher.revision
        field = self.panel().findChild(QLineEdit, "root-name")
        field.setText("my shot")
        field.editingFinished.emit()
        wait_until(lambda: w.dispatcher.document["settings"].get("project", {}).get("name") == "my shot")
        APP.processEvents()
        self.assertEqual(w.dispatcher.document["settings"]["project"]["name"], "my shot")
        self.assertEqual(w.dispatcher.revision, start + 1)
        w.command({"op": "undo"})
        APP.processEvents()
        self.assertNotIn("project", w.dispatcher.document["settings"])
        self.assertEqual(self.panel().findChild(QLineEdit, "root-name").text(), "")     # the panel follows the undo

    def test_frame_range_fps_lock_and_proxy_controls_edit_the_document(self):
        w = self.window
        w.open_project_settings_panel()
        spin = self.panel().findChild(QSpinBox, "root-last_frame")
        spin.setValue(48)
        spin.editingFinished.emit()
        wait_until(lambda: w.dispatcher.document["time"]["last"] == 48)
        self.assertEqual(w.dispatcher.document["time"]["last"], 48)
        box = self.panel().findChild(QCheckBox, "root-lock_range")
        box.setChecked(True)
        wait_until(lambda: w.dispatcher.document["settings"].get("project", {}).get("lock_range") is True)
        self.assertTrue(w.dispatcher.document["settings"]["project"]["lock_range"])
        proxy = self.panel().findChild(QCheckBox, "root-proxy")
        proxy.setChecked(True)
        wait_until(lambda: rootpanel.get(w.dispatcher.document, "settings.viewer.proxy") == 2)
        self.assertEqual(rootpanel.get(w.dispatcher.document, "settings.viewer.proxy"), 2)
        scale = self.panel().findChild(QDoubleSpinBox, "root-proxy_scale")
        self.assertAlmostEqual(scale.value(), 0.5)


if __name__ == "__main__":
    unittest.main()
