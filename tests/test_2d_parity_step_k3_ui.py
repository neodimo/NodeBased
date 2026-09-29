"""Lane L2 step K3: picking Cryptomatte objects by clicking in the viewer (offscreen Qt). The click
lands on the viewer showing the finished matte; the name comes from the Cryptomatte layers of the
node's input."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document
from nodebased.media import write_exr
from tests.test_2d_parity_step_k3 import beauty, crypto_layers, manifest_metadata
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


class CryptomattePickTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        path = Path(self._dir.name) / "crypto.exr"
        write_exr(path, beauty(), bits="float", layers=crypto_layers(), metadata=manifest_metadata())
        document = empty_document()
        document["nodes"] = {}
        dispatcher = Dispatcher(document)
        dispatcher.execute({"op": "create", "id": "r", "type": "Read", "params": {"path": str(path)}, "pos": [0, 0]})
        dispatcher.execute({"op": "create", "id": "k", "type": "Cryptomatte", "pos": [150, 0]})
        dispatcher.execute({"op": "connect", "id": "k", "input": "image", "source": "r"})
        dispatcher.execute({"op": "view", "id": "k"})
        self.window = Window(dispatcher.document)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None and self.window.viewer.format_rect is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def matte_list(self):
        return self.window.dispatcher.document["nodes"]["k"]["params"]["matte_list"]

    def click(self, x, y):
        viewer = self.window.viewer
        QTest.mouseClick(viewer.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         viewer.mapFromScene(QPointF(x + 0.5, y + 0.5)))

    def test_clicks_add_names_until_escape(self):
        viewer = self.window.viewer
        self.assertTrue(self.window.begin_crypto_pick("k"))
        self.assertEqual(viewer.crypto_picking, "k")
        self.click(1, 2)
        self.assertEqual(self.matte_list(), "sphere")
        self.click(6, 2)
        self.assertEqual(self.matte_list(), "sphere, cube")
        self.click(6, 3)                       # already listed: unchanged
        self.assertEqual(self.matte_list(), "sphere, cube")
        self.assertEqual(len(self.window.dispatcher.undo_stack), 2)      # one undo step per addition
        QTest.keyClick(viewer, Qt.Key.Key_Escape)
        self.assertIsNone(viewer.crypto_picking)
        self.click(12, 2)                      # picking is off: a click adds nothing
        self.assertEqual(self.matte_list(), "sphere, cube")

    def test_a_click_outside_the_image_adds_nothing(self):
        self.assertTrue(self.window.crypto_pick("k", 500, 500) is False)
        self.assertEqual(self.matte_list(), "")

    def test_picking_needs_a_cryptomatte_node(self):
        self.window.command({"op": "create", "id": "g", "type": "Grade", "pos": [300, 0]})
        self.assertFalse(self.window.begin_crypto_pick("g"))
        self.assertIsNone(self.window.viewer.crypto_picking)


if __name__ == "__main__":
    unittest.main()
