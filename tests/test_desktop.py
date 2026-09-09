import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox
from nodebased.app import Window, STYLE

APP = QApplication.instance() or QApplication([])
APP.setStyle('Fusion')
APP.setStyleSheet(STYLE)


def wait_until(condition, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if condition():
            return True
        QTest.qWait(10)
    return False


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = 'nodebased-test-' + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_live_agent_edit_and_undo(self):
        client = QLocalSocket()
        client.connectToServer(self.endpoint)
        self.assertTrue(client.waitForConnected(2000))
        def rpc(cmd):
            client.write(json.dumps(cmd).encode() + b'\n'); client.flush()
            self.assertTrue(wait_until(lambda: client.canReadLine()))
            return json.loads(bytes(client.readLine()))
        self.assertTrue(rpc({'op': 'set', 'id': 'grade', 'param': 'exposure', 'value': 2})['ok'])
        self.assertEqual(self.window.dispatcher.document['nodes']['grade']['params']['exposure'], 2)
        self.assertTrue(rpc({'op': 'undo'})['ok'])
        self.assertEqual(self.window.dispatcher.document['nodes']['grade']['params']['exposure'], 0.35)
        self.assertFalse(rpc({'op': 'connect', 'id': 'grade', 'input': 'image', 'source': 'viewer'})['ok'])
        client.disconnectFromServer()

    def test_keyboard_view_and_inspector_edit(self):
        w = self.window
        w.graph.items_by_id['grade'].setSelected(True)
        w.graph.setFocus()
        QTest.keyClick(w.graph, Qt.Key.Key_1)
        self.assertEqual(w.dispatcher.document['view'], 'grade')
        editor = w.properties.findChildren(QDoubleSpinBox)[0]
        editor.setValue(1.5)
        editor.editingFinished.emit()
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['params']['exposure'] == 1.5))
        self.assertTrue(wait_until(lambda: w.frame_generation == w.generation))

    def test_wire_ports_and_disconnect(self):
        w = self.window
        graph = w.graph
        source = graph.mapFromScene(graph.items_by_id['wash'].output.scenePos())
        dest = graph.mapFromScene(graph.items_by_id['grade'].inputs['image'].scenePos())
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=source)
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=dest)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['inputs']['image'] == 'wash'))
        dest = graph.mapFromScene(graph.items_by_id['grade'].inputs['image'].scenePos())
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.RightButton, pos=dest)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['inputs']['image'] is None))

    def test_stale_preview_cannot_win(self):
        w = self.window
        for exposure in [1, 2, 3, -1]:
            w.command({'op': 'set', 'id': 'grade', 'param': 'exposure', 'value': exposure})
            APP.processEvents()
        self.assertTrue(wait_until(lambda: w.frame_generation == w.generation))
        from nodebased.imaging import Evaluator
        import numpy as np
        np.testing.assert_array_equal(w.frame, Evaluator().evaluate(w.dispatcher.document))

    def test_screenshot_artifact(self):
        target = os.environ.get('NODEBASED_SCREENSHOT')
        if target:
            self.window.graph.items_by_id['grade'].setSelected(True)
            APP.processEvents()
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            self.assertTrue(self.window.grab().save(target))

    def test_export_uses_snapshot_across_modal_dialog(self):
        from unittest.mock import patch
        from nodebased.imaging import read_image
        w = self.window
        original_shape = w.frame.shape
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'export.png')
            def dialog(*args):
                w.frame = None  # A newer graph evaluation can fail while dialog is open.
                return path, 'PNG (*.png)'
            with patch('nodebased.app.QFileDialog.getSaveFileName', side_effect=dialog):
                w.export()
            self.assertEqual(read_image(path).shape, original_shape)

    def test_update_button_flow_and_unsaved_cancel(self):
        from unittest.mock import patch
        w = self.window
        w.updater.changed.emit('available', '0.2.0', 0)
        self.assertEqual(w.update_button.text(), 'Download v0.2.0')
        with patch.object(w.updater, 'fetch') as fetch:
            w.update_button.click()
            fetch.assert_called_once()
        w.updater.changed.emit('downloading', '0.2.0', 42)
        self.assertEqual(w.update_button.text(), 'Downloading 42%')
        self.assertFalse(w.update_button.isEnabled())
        w.updater.changed.emit('ready', '0.2.0', 100)
        with patch.object(w, 'confirm_discard', return_value=False), patch.object(w.updater, 'install') as install:
            w.update_button.click()
            install.assert_not_called()
        self.assertTrue(w.isVisible())
