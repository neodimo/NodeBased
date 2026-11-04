import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
import uuid
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit, QPushButton, QSlider, QTabWidget, QWidget
from PySide6.QtTest import QTest
from nodebased.app import Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])

class ReadPanelLayoutTests(unittest.TestCase):
    def setUp(self):
        self.window = Window(agent_name="read-panel-" + uuid.uuid4().hex)
        self.window.resize(1200, 1200)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None), "first frame did not arrive")
        self.window.command({"op": "create", "id": "read-layout", "type": "Read", "pos": [0, 0]})
        self.window.inspect("read-layout")
        QTest.qWait(150)
        self.panel = self.window.properties.widget()
        self.tabs = self.panel.findChild(QTabWidget, "node-tabs")

    def tearDown(self):
        # Every test edits the document; without this, closing asks "Save changes?" and the modal box never returns.
        self.window.saved_document = self.window.dispatcher.document
        self.window.close(); self.window.deleteLater(); APP.processEvents()

    def named(self, name):
        return self.panel.findChild(QWidget, name)

    def test_read_tabs_and_tab_ownership(self):
        self.assertEqual([self.tabs.tabText(i) for i in range(self.tabs.count())],
                         ["Read", "Context", "Sequence", "Metadata", "Node"])
        read = self.tabs.widget(0)
        for name in ("key1-field", "value1-field", "key4-field", "value4-field", "read-import-sequence",
                     "read-sequence-state", "read-metadata-filter", "read-shown-metadata"):
            self.assertIsNone(read.findChild(QWidget, name), name)
        for name in ("key1-field", "value4-field"):
            self.assertIsNotNone(self.tabs.widget(1).findChild(QWidget, name))
        for name in ("read-import-sequence", "read-sequence-state"):
            self.assertIsNotNone(self.tabs.widget(2).findChild(QWidget, name))
        for name in ("read-metadata-filter", "read-shown-metadata"):
            self.assertIsNotNone(self.tabs.widget(3).findChild(QWidget, name))

    def test_read_rows_and_control_types(self):
        for name, cls in (("path-field", QLineEdit), ("proxy-field", QLineEdit),
                          ("localizationPolicy-choice", QComboBox), ("before-choice", QComboBox),
                          ("after-choice", QComboBox), ("frame_mode-choice", QComboBox),
                          ("premultiplied-switch", QCheckBox), ("raw-switch", QCheckBox),
                          ("auto_alpha-switch", QCheckBox)):
            self.assertIsInstance(self.panel.findChild(cls, name), cls, name)
        self.assertIsNotNone(self.panel.findChild(QPushButton, "read-reload"))
        self.assertIsNone(self.panel.findChild(QWidget, "reload-field"))
        for name in ("first-field", "last-field", "origfirst-field", "origlast-field"):
            field = self.panel.findChild(QWidget, name)
            self.assertIsNotNone(field, name)
            self.assertFalse(any(isinstance(w, QSlider) for w in field.parentWidget().findChildren(QSlider)))
        self.assertIsNone(self.panel.findChild(QWidget, "frame_offset-field"))
        policy = self.panel.findChild(QComboBox, "localizationPolicy-choice")
        self.assertGreaterEqual(policy.sizeHint().width(), policy.fontMetrics().horizontalAdvance("from auto-localize path"))
        for names in (("first-field", "before-choice", "last-field", "after-choice"),
                      ("frame_mode-choice", "frame_expr-field"), ("origfirst-field", "origlast-field")):
            widgets = [self.panel.findChild(QWidget, name) for name in names]
            ancestor = widgets[0].parentWidget()
            while ancestor and not all(w is ancestor or ancestor.isAncestorOf(w) for w in widgets[1:]):
                ancestor = ancestor.parentWidget()
            self.assertIsNotNone(ancestor, names)

    def test_boolean_controls_commit_and_legacy_offset_is_conditional(self):
        toggle = self.panel.findChild(QCheckBox, "raw-switch")
        toggle.click()
        # The checkbox commits through defer_command, so the value lands on a later event-loop turn.
        self.assertTrue(wait_until(lambda: self.window.graph_nodes()["read-layout"]["params"]["raw"] == 1))
        self.window.command({"op": "set", "id": "read-layout", "param": "frame_offset", "value": 3})
        self.window.inspect("read-layout")
        QTest.qWait(100)
        self.panel = self.window.properties.widget()
        self.assertIsNotNone(self.panel.findChild(QWidget, "frame_offset-field"))
