"""Lane L2 (2D parity) plan 23, step AA1 part 1: S creates a Shuffle (hands-on pass 10/7, finding 1).
The Edit menu's Settings used to own bare S and swallowed the graph's Shuffle key."""
import unittest

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401
from nodebased.app import SHORTCUT_SECTIONS, Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])

GRAPH_LETTER_KEYS = "RGMTBCSOPUW" + "DF"


class ShortcutTests(unittest.TestCase):
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

    def test_s_with_a_checker_selected_makes_a_wired_shuffle_and_opens_no_settings(self):
        w = self.window
        w.command({"op": "batch", "commands": [{"op": "delete", "id": k} for k in list(w.graph_nodes())]})
        w.command({"op": "create", "id": "chk", "type": "Checker", "pos": [0, 0], "params": {}})
        APP.processEvents()
        w.graph.scene().clearSelection()
        w.graph.items_by_id["chk"].setSelected(True)
        before = set(w.graph.items_by_id)
        opened = []
        original = w.project_settings
        w.project_settings = lambda *a, **k: opened.append(1)
        try:
            w.graph.setFocus()
            QTest.keyClick(w.graph, Qt.Key.Key_S)
            APP.processEvents()
        finally:
            w.project_settings = original
        (new,) = [k for k in w.graph.items_by_id if k not in before]
        node = w.graph_nodes()[new]
        self.assertEqual(node["type"], "Shuffle")
        self.assertEqual(node["inputs"].get("image"), "chk")
        self.assertEqual(opened, [])

    def test_settings_is_ctrl_comma_and_listed(self):
        keys = dict((text, key) for key, text in dict(SHORTCUT_SECTIONS)["Edit"])
        self.assertEqual(keys["Settings"], "Ctrl+,")
        settings = [a for a in self.window.findChildren(QAction) if a.text() == "Settings…"]
        self.assertEqual([a.shortcut().toString() for a in settings], ["Ctrl+,"])

    def test_no_menu_shortcut_shadows_a_bare_letter_graph_key(self):
        letters = {QKeySequence(c).toString() for c in GRAPH_LETTER_KEYS}
        clashes = []
        for action in self.window.findChildren(QAction):
            for seq in action.shortcuts():
                if seq.toString() in letters:
                    clashes.append((action.text(), seq.toString()))
        self.assertEqual(clashes, [])
        # the on-screen reference agrees: no menu row (File, Edit, Time) uses a bare graph letter
        for section in ("File", "Edit", "Time"):
            for key, text in dict(SHORTCUT_SECTIONS)[section]:
                self.assertNotIn(key, letters, text)

    def test_graph_letter_keys_match_the_graphs_handler(self):
        import inspect
        from nodebased import app
        source = inspect.getsource(app.Graph.keyPressEvent)
        for letter in GRAPH_LETTER_KEYS:
            self.assertIn(f"Key_{letter}", source, letter)


if __name__ == "__main__":
    unittest.main()
