import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QPointF, QEvent
from PySide6.QtGui import QCursor, QKeyEvent
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox
from nodebased.app import Window, STYLE, NodeSearch

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

    def test_dragging_an_output_noodle_connects_to_an_input(self):
        w = self.window
        graph = w.graph
        source = graph.mapFromScene(graph.items_by_id['wash'].output.scenePos())
        dest = graph.mapFromScene(graph.items_by_id['grade'].inputs['image'].scenePos())
        QTest.mousePress(graph.viewport(), Qt.MouseButton.LeftButton, pos=source)
        QTest.mouseMove(graph.viewport(), dest, 30)
        QTest.mouseRelease(graph.viewport(), Qt.MouseButton.LeftButton, pos=dest)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['inputs']['image'] == 'wash'))

    def test_dragging_a_top_input_snaps_to_an_output(self):
        w = self.window
        graph = w.graph
        source = graph.mapFromScene(graph.items_by_id['grade'].inputs['image'].scenePos())
        # Deliberately release 18 physical pixels from the output centre. This is
        # inside the magnetic zone and proves input -> output reverse wiring.
        output_scene = graph.items_by_id['wash'].output.scenePos() + QPointF(18, 0)
        dest = graph.mapFromScene(output_scene)
        QTest.mousePress(graph.viewport(), Qt.MouseButton.LeftButton, pos=source)
        QTest.mouseMove(graph.viewport(), dest, 30)
        QTest.mouseRelease(graph.viewport(), Qt.MouseButton.LeftButton, pos=dest)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['inputs']['image'] == 'wash'))

    def test_ports_are_centred_and_new_nodes_do_not_overlap(self):
        w = self.window
        merge = w.graph.items_by_id['merge']
        self.assertEqual(merge.output.pos().x(), 95)
        self.assertEqual((merge.inputs['A'].pos().x() + merge.inputs['B'].pos().x()) / 2, 95)
        desired = merge.pos()
        merge_rect = merge.sceneBoundingRect()
        w.add_node('Grade', position=desired)
        created = next(item for key, item in w.graph.items_by_id.items() if key not in {'plate', 'wash', 'grade', 'merge', 'viewer'})
        self.assertFalse(created.sceneBoundingRect().intersects(merge_rect))

    def test_tab_search_filters_node_types(self):
        picker = NodeSearch(self.window, ['Grade', 'ColorCorrect', 'Transform'], self.window.pos())
        picker.query.setText('color')
        self.assertEqual([picker.list.item(i).text() for i in range(picker.list.count())], ['ColorCorrect'])
        picker.close()

    def test_tab_is_captured_when_pointer_is_over_graph(self):
        from unittest.mock import patch
        point = self.window.graph.viewport().rect().center()
        QCursor.setPos(self.window.graph.viewport().mapToGlobal(point))
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier)
        with patch.object(self.window, 'node_search') as search:
            self.assertTrue(self.window.eventFilter(self.window.properties, event))
            search.assert_called_once()

    def test_rewire_picks_up_a_connected_input_and_reconnects_it(self):
        w = self.window
        graph = w.graph
        # 'grade' starts wired to 'plate'. Picking it up never disconnects until
        # a valid output is chosen, so Esc/missed drops cannot damage the comp.
        dest = graph.mapFromScene(graph.items_by_id['grade'].inputs['image'].scenePos())
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=dest)
        self.assertEqual(w.dispatcher.document['nodes']['grade']['inputs']['image'], 'plate')
        self.assertEqual(graph.wire_input, ('grade', 'image'))
        self.assertIsNotNone(graph.pending_edge)
        new_source = graph.mapFromScene(graph.items_by_id['wash'].output.scenePos())
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=new_source)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['inputs']['image'] == 'wash'))
        self.assertIsNone(graph.wire_input)

    def test_clicking_empty_canvas_drops_a_picked_up_wire(self):
        w = self.window
        graph = w.graph
        dest = graph.mapFromScene(graph.items_by_id['grade'].inputs['image'].scenePos())
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=dest)
        self.assertEqual(w.dispatcher.document['nodes']['grade']['inputs']['image'], 'plate')
        self.assertIsNotNone(graph.wire_input)
        empty = graph.mapFromScene(QPointF(-3000, -3000))
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=empty)
        self.assertEqual(w.dispatcher.document['nodes']['grade']['inputs']['image'], 'plate')
        self.assertIsNone(graph.wire_input)
        self.assertIsNone(graph.pending_edge)

    def test_ctrl_dragging_a_noodle_midpoint_inserts_dot_without_breaking_flow(self):
        w = self.window
        graph = w.graph
        edge, source, destination, slot = next(edge for edge in graph.edges
                                               if edge[1:] == ('plate', 'grade', 'image'))
        handle = graph.mapFromScene(edge.handle)
        target = handle + QPointF(80, 50).toPoint()
        QTest.mousePress(graph.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, handle)
        self.assertIsNotNone(graph.dot_preview)
        QTest.mouseMove(graph.viewport(), target, 30)
        self.assertEqual(graph.dot_preview.pos(), graph.mapToScene(target))
        QTest.mouseRelease(graph.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, target)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['inputs']['image'] != 'plate'))
        dot_id = w.dispatcher.document['nodes']['grade']['inputs']['image']
        self.assertEqual(w.dispatcher.document['nodes'][dot_id]['type'], 'Dot')
        self.assertEqual(w.dispatcher.document['nodes'][dot_id]['inputs']['input'], 'plate')

    def test_creating_a_dot_never_clears_existing_graph_items(self):
        w = self.window
        graph = w.graph
        before_edges = len(graph.edges)
        w.add_node('Dot', position=QPointF(300, 300))
        dot = next(item for item in graph.items_by_id.values() if item.is_dot)
        self.assertEqual(dot.rect().size().width(), 20)
        self.assertEqual(len(graph.edges), before_edges)
        self.assertIn('viewer', graph.items_by_id)

    def test_dot_center_selects_and_drags_without_hitting_its_ports(self):
        w = self.window
        graph = w.graph
        w.add_node('Dot', position=QPointF(300, 300))
        key, dot = next((key, item) for key, item in graph.items_by_id.items() if item.is_dot)
        center = graph.mapFromScene(dot.sceneBoundingRect().center())
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, pos=center)
        self.assertTrue(dot.isSelected())
        target = center + QPointF(70, 40).toPoint()
        QTest.mousePress(graph.viewport(), Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(graph.viewport(), target, 30)
        QTest.mouseRelease(graph.viewport(), Qt.MouseButton.LeftButton, pos=target)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes'][key]['pos'] != [300, 300]))

    def test_control_key_reveals_graph_handles_under_pointer(self):
        graph = self.window.graph
        point = graph.viewport().rect().center()
        QCursor.setPos(graph.viewport().mapToGlobal(point))
        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier)
        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Control, Qt.KeyboardModifier.NoModifier)
        self.window.eventFilter(graph, press)
        self.assertTrue(graph.ctrl_handles_visible)
        self.window.eventFilter(graph, release)
        self.assertFalse(graph.ctrl_handles_visible)

    def test_window_uses_the_nodebased_application_icon(self):
        self.assertFalse(self.window.windowIcon().isNull())

    def test_viewer_channel_and_framing_shortcuts(self):
        w = self.window
        viewer = w.viewer
        viewer.setFocus()
        QTest.keyClick(viewer, Qt.Key.Key_R)
        self.assertEqual(w.channels.currentText(), 'R')
        QTest.keyClick(viewer, Qt.Key.Key_R)
        self.assertEqual(w.channels.currentText(), 'RGB')
        for key, channel in ((Qt.Key.Key_G, 'G'), (Qt.Key.Key_B, 'B'), (Qt.Key.Key_A, 'A')):
            QTest.keyClick(viewer, key)
            self.assertEqual(w.channels.currentText(), channel)
        viewer.scale(1.5, 1.5)
        QTest.keyClick(viewer, Qt.Key.Key_F)
        fitted = viewer.transform().m11()
        viewer.scale(1.5, 1.5)
        QTest.keyClick(viewer, Qt.Key.Key_H)
        self.assertAlmostEqual(viewer.transform().m11(), fitted, places=5)

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
