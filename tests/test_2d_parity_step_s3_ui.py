"""Lane L2 step S3: the metadata inspector on the ViewMetaData and CompareMetaData panels (offscreen Qt)."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QTableWidget

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document
from nodebased.media import write_exr

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def plate():
    frame = np.zeros((8, 16, 4), np.float32)
    frame[..., 3] = 1.0
    return frame


class MetadataPanelTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        first, second = (Path(self._dir.name) / name for name in ("a.exr", "b.exr"))
        write_exr(first, plate(), bits="float", metadata={"exr/lens": "35mm", "show/shot": "sh010", "same/key": "x"})
        write_exr(second, plate(), bits="float", metadata={"exr/lens": "50mm", "show/take": "9", "same/key": "x"})
        document = empty_document()
        document["nodes"] = {}
        d = Dispatcher(document)
        d.execute({"op": "create", "id": "a", "type": "Read", "params": {"path": str(first)}, "pos": [0, 0]})
        d.execute({"op": "create", "id": "b", "type": "Read", "params": {"path": str(second)}, "pos": [0, 80]})
        d.execute({"op": "create", "id": "v", "type": "ViewMetaData", "pos": [150, 0]})
        d.execute({"op": "connect", "id": "v", "input": "image", "source": "a"})
        d.execute({"op": "create", "id": "c", "type": "CompareMetaData", "pos": [150, 80]})
        d.execute({"op": "connect", "id": "c", "input": "image", "source": "a"})
        d.execute({"op": "connect", "id": "c", "input": "other", "source": "b"})
        d.execute({"op": "create", "id": "loose", "type": "ViewMetaData", "pos": [150, 160]})
        self.window = Window(d.document)
        self.window.show()
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def panel(self, key):
        panel = self.window.build_node_panel(key)
        return panel, panel.findChild(QTableWidget, "metadata-table")

    @staticmethod
    def cells(table):
        return {(r, c): table.item(r, c).text() for r in range(table.rowCount()) for c in range(table.columnCount())}

    def test_view_lists_every_key_and_value(self):
        _, table = self.panel("v")
        keys = {table.item(r, 0).text(): table.item(r, 1).text() for r in range(table.rowCount())}
        self.assertEqual(keys["exr/lens"], "35mm")
        self.assertEqual(keys["show/shot"], "sh010")
        self.assertIn("input/filename", keys)
        self.assertTrue(keys["input/filename"].endswith("a.exr"))

    def test_search_filters_by_key_or_value(self):
        panel, table = self.panel("v")
        search = panel.findChild(QLineEdit, "metadata-search")
        search.setText("SH010")
        shown = [table.item(r, 0).text() for r in range(table.rowCount()) if not table.isRowHidden(r)]
        self.assertEqual(shown, ["show/shot"])
        search.setText("lens")
        shown = [table.item(r, 0).text() for r in range(table.rowCount()) if not table.isRowHidden(r)]
        self.assertEqual(shown, ["exr/lens"])
        search.setText("")
        self.assertFalse(any(table.isRowHidden(r) for r in range(table.rowCount())))

    def test_compare_lists_the_differing_keys(self):
        _, table = self.panel("c")
        rows = {table.item(r, 0).text(): (table.item(r, 1).text(), table.item(r, 2).text())
                for r in range(table.rowCount())}
        self.assertEqual(rows["exr/lens"], ("35mm", "50mm"))
        self.assertEqual(rows["show/shot"], ("sh010", ""))
        self.assertEqual(rows["show/take"], ("", "9"))
        self.assertNotIn("same/key", rows)

    def test_an_unwired_node_says_what_is_missing(self):
        panel, table = self.panel("loose")
        self.assertEqual(table.rowCount(), 0)
        self.assertIn("Connect", panel.findChild(QLabel, "metadata-summary").text())


if __name__ == "__main__":
    unittest.main()
