import json
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid
import threading

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QPointF, QEvent
from PySide6.QtGui import QCursor, QKeyEvent
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QPushButton
from nodebased.app import Window, STYLE, NodeSearch, ProjectSettingsDialog
from nodebased.imaging import to_qimage
from nodebased.playback import FrameRequest, MAX_PREFETCH

APP = QApplication.instance() or QApplication([])
APP.setStyle('Fusion')
APP.setStyleSheet(STYLE)


# The whole harness's clock. `wait_until` returns the instant its condition holds, so a
# generous budget costs nothing on a fast machine — it only spends wall time when a test is
# already failing. A tight budget turns a slow machine into a false failure: the 5 s value
# that used to be here is exactly what produced the 9 Windows conformance failures in run
# 34578372838. All nine were `setUp` waiting for the first frame, which a cold
# windows-latest runner cannot reliably cook in 5 s; the byte-identical tree passed on a
# warm runner in 34614864732. Deliberately one value on every platform, so "passes locally,
# fails on CI" can never be a timing artefact of the harness itself.
WAIT_TIMEOUT = 30.0


def wait_until(condition, timeout=WAIT_TIMEOUT):
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
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f'no first frame cooked within {WAIT_TIMEOUT:.0f}s')

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

    def test_agent_errors_op_surfaces_live_render_failures(self):
        # The Dispatcher only knows about document edits, never the async render pipeline, so an
        # attached agent had no way to see "this frame failed to evaluate" except by asking a
        # human to read the screen. The errors op reaches into Window's live render state instead.
        w = self.window
        client = QLocalSocket()
        client.connectToServer(self.endpoint)
        self.assertTrue(client.waitForConnected(2000))
        def rpc(cmd):
            client.write(json.dumps(cmd).encode() + b'\n'); client.flush()
            self.assertTrue(wait_until(lambda: client.canReadLine()))
            return json.loads(bytes(client.readLine()))
        baseline = rpc({'op': 'errors'})
        self.assertTrue(baseline['ok'])
        self.assertEqual(baseline['result']['errors'], [])
        w.dispatcher.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'broken', 'type': 'Read', 'params': {'path': ''}},
            {'op': 'connect', 'id': 'viewer', 'input': 'image', 'source': 'broken'},
        ]})
        w.request_preview()
        self.assertTrue(wait_until(lambda: w.viewer_info.text() == 'Evaluation error'))
        result = rpc({'op': 'errors'})['result']
        self.assertGreaterEqual(len(result['errors']), 1)
        self.assertEqual(result['errors'][-1]['target'], 'viewer')
        self.assertTrue(result['errors'][-1]['message'])  # a real, non-empty exception message
        self.assertEqual(result['status'], 'Evaluation error')
        # `since` excludes anything already seen, so polling doesn't re-report old failures.
        future = rpc({'op': 'errors', 'since': time.time() + 5})['result']
        self.assertEqual(future['errors'], [])
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

    def test_project_settings_surface_and_sync_to_viewer(self):
        w = self.window
        self.assertEqual(w.display_view.currentText(), 'ACES 2.0')
        dialog = ProjectSettingsDialog(copy.deepcopy(w.dispatcher.document['settings']), w)
        self.assertEqual(dialog.view.currentText(), 'ACES 2.0')
        self.assertEqual(dialog.background.currentData(), 'black')
        dialog.view.setCurrentText('Linear')
        dialog.background.setCurrentIndex(dialog.background.findData('checker'))
        w.command({'op': 'settings', 'settings': dialog.changes()}, render=False)
        self.assertEqual(w.display_view.currentText(), 'Linear')
        self.assertEqual(w.dispatcher.document['settings']['viewer']['background'], 'checker')
        dialog.close()

    def test_reconnecting_viewer_to_a_larger_source_requests_the_full_canvas(self):
        # Reported live against a 4K sequence: the image appeared stuck zoomed into its top-left
        # corner with no way to recenter. The demo graph is 960x540; swapping the viewer onto a
        # much larger source reproduces the same viewport-mapped-scene-rect the bug depended on,
        # because the OLD 960x540 image's fitted viewport, mapped to scene coordinates, is a small
        # rectangle near the origin -- intersecting that against the NEW canvas's bounds (rather
        # than requesting the new canvas outright) is exactly the wrong crop the old code shipped.
        w = self.window
        w.dispatcher.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'big', 'type': 'Constant',
             'params': {'width': 4000, 'height': 3000, 'red': 0.2, 'green': 0.4, 'blue': 0.6, 'alpha': 1.0}},
            {'op': 'connect', 'id': 'viewer', 'input': 'image', 'source': 'big'},
        ]})
        w.request_preview()
        self.assertTrue(wait_until(lambda: w.frame_generation == w.generation))
        extent = w.viewer.scene().itemsBoundingRect()
        scene_rect = w.viewer.sceneRect()
        self.assertEqual((extent.width(), extent.height()), (4000.0, 3000.0),
                         'first frame of a resized source must render the whole new canvas, not '
                         'a crop inherited from the previous viewport')
        self.assertEqual((extent.width(), extent.height()), (scene_rect.width(), scene_rect.height()))

    def test_format_overlay_tracks_the_frame_without_growing_scene_bounds(self):
        # The dotted border and resolution readout are drawForeground-painted, not scene
        # items, precisely so they never perturb itemsBoundingRect() -- the property the
        # test above depends on to prove a resize wasn't cropped. Guard that invariant
        # directly, plus that the overlay actually picks up the new canvas size.
        w = self.window
        w.dispatcher.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'big', 'type': 'Constant',
             'params': {'width': 4000, 'height': 3000, 'red': 0.2, 'green': 0.4, 'blue': 0.6, 'alpha': 1.0}},
            {'op': 'connect', 'id': 'viewer', 'input': 'image', 'source': 'big'},
        ]})
        w.request_preview()
        self.assertTrue(wait_until(lambda: w.frame_generation == w.generation))
        self.assertEqual((w.viewer.format_rect.width(), w.viewer.format_rect.height()), (4000.0, 3000.0))
        extent = w.viewer.scene().itemsBoundingRect()
        self.assertEqual((extent.width(), extent.height()), (4000.0, 3000.0))

    def test_format_overlay_clears_on_an_evaluation_error(self):
        # A dotted box left over from the last good frame, floating above an error
        # message with nothing behind it, would misreport a format that isn't there.
        w = self.window
        w.dispatcher.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'broken', 'type': 'Read', 'params': {'path': ''}},
            {'op': 'connect', 'id': 'viewer', 'input': 'image', 'source': 'broken'},
        ]})
        w.request_preview()
        self.assertTrue(wait_until(lambda: w.viewer_info.text() == 'Evaluation error'))
        self.assertIsNone(w.viewer.format_rect)

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

    def test_adding_a_node_with_a_selection_wires_into_its_branch(self):
        # 'grade' feeds merge's 'B' input in the demo graph. Selecting it before adding a
        # node should splice the new node inline -- wired from grade's output, and taking
        # over grade's existing downstream connection -- rather than dropping an
        # unconnected node whose position happens to depend on a stale last click.
        w = self.window
        w.graph.items_by_id['grade'].setSelected(True)
        w.add_node('Blur')
        blur_id = next(key for key in w.graph.items_by_id if key not in {'plate', 'wash', 'grade', 'merge', 'viewer'})
        doc = w.dispatcher.document
        self.assertEqual(doc['nodes'][blur_id]['inputs']['image'], 'grade')
        self.assertEqual(doc['nodes']['merge']['inputs']['B'], blur_id)
        self.assertEqual(doc['nodes']['grade']['inputs']['image'], 'plate')

    def test_adding_a_generator_with_a_selection_still_uses_click_position(self):
        # Read/Constant/Checker have no input slot, so selecting a node beforehand must not
        # try to wire a connection that doesn't exist -- it should fall back to ordinary
        # click-position placement exactly like no selection was made.
        w = self.window
        w.graph.items_by_id['grade'].setSelected(True)
        desired = QPointF(400, 400)
        w.add_node('Checker', position=desired)
        doc = w.dispatcher.document
        new_id = next(key for key in w.graph.items_by_id if key not in {'plate', 'wash', 'grade', 'merge', 'viewer'})
        self.assertEqual(doc['nodes']['grade']['inputs']['image'], 'plate')
        self.assertEqual(doc['nodes'][new_id]['inputs'], {})

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

    def test_timeline_edits_scrub_step_and_follow_undo(self):
        w = self.window
        w.set_time(first=1001, last=1003, current=1002, fps=24.0)
        self.assertEqual(w.frame_slider.minimum(), 1001)
        self.assertEqual(w.frame_slider.maximum(), 1003)
        self.assertEqual(w.frame_slider.value(), 1002)
        w.step_frame(1)
        self.assertEqual(w.dispatcher.document['time']['current'], 1003)
        w.step_frame(1)
        self.assertEqual(w.dispatcher.document['time']['current'], 1003)
        w.frame_slider.setValue(1001)
        self.assertEqual(w.dispatcher.document['time']['current'], 1001)
        w.command({'op': 'undo'})
        self.assertEqual(w.dispatcher.document['time']['current'], 1003)
        self.assertEqual(w.frame_slider.value(), 1003)

    def test_playback_rate_control_defaults_to_24_and_drives_the_transport(self):
        w = self.window
        self.assertEqual(w.dispatcher.document['time']['fps'], 24.0)
        self.assertEqual(w.frame_fps.value(), 24.0)
        self.assertEqual(w.fps_presets.currentText(), '24')

        w.set_time(first=1, last=100, current=1)
        w.frame_fps.setValue(48.0)
        self.assertEqual(w.dispatcher.document['time']['fps'], 48.0)
        # The rate is a document edit, so it undoes like any other.
        w.command({'op': 'undo'})
        self.assertEqual(w.dispatcher.document['time']['fps'], 24.0)
        self.assertEqual(w.frame_fps.value(), 24.0)

        # A preset selects a broadcast rate exactly rather than a rounded one.
        w.fps_presets.setCurrentIndex(w.fps_presets.findText('23.976'))
        self.assertAlmostEqual(w.dispatcher.document['time']['fps'], 24000.0 / 1001.0, places=6)
        self.assertIn('4.17 s', w.frame_info.text())

        # The transport follows the document rate: at 50 fps the same elapsed wall clock advances
        # twice as far as it did at 25.
        for rate, expected in ((25.0, 4), (50.0, 7)):
            w.set_time(fps=rate)
            w.toggle_playback(True)
            w.playback_origin_frame = 1
            w.playback_origin_time = time.monotonic() - (3.25 / 25.0)
            # This isolates the wall-clock math itself; the Nuke-style render-pacing cap (see
            # playback_tick) is exercised separately below and must not gate this assertion.
            w.playback_frames_rendered = expected
            w.playback_tick()
            self.assertEqual(w.dispatcher.document['time']['current'], expected)
            w.toggle_playback(False)
            w.set_time(current=1)

    def test_changing_rate_mid_playback_reanchors_instead_of_jumping(self):
        w = self.window
        w.set_time(first=1, last=200, current=1, fps=24.0)
        w.toggle_playback(True)
        w.playback_origin_frame = 1
        w.playback_origin_time = time.monotonic() - (10.0 / 24.0)
        # Isolates the wall-clock/re-anchoring math from the Nuke-style render-pacing cap.
        w.playback_frames_rendered = 200
        w.playback_tick()
        landed = w.dispatcher.document['time']['current']
        w.set_time(fps=60.0)
        self.assertEqual(w.playback_origin_frame, landed)
        w.playback_tick()
        # Re-anchored: the playhead continues from where it was rather than teleporting to the
        # position the new rate would have reached from the old origin.
        self.assertLess(w.dispatcher.document['time']['current'] - landed, 3)
        w.toggle_playback(False)

    def test_playback_uses_wall_clock_and_bounded_read_ahead(self):
        w = self.window
        w.set_time(first=1, last=8, current=1, fps=24.0)
        undo_slots = len(w.dispatcher.undo_stack)
        w.toggle_playback(True)
        self.assertTrue(w.playing)
        self.assertEqual(w.play_button.text(), '■')
        self.assertLessEqual(len(w.preview_queue), 4)
        w.playback_origin_frame = 1
        w.playback_origin_time = time.monotonic() - (3.25 / 24.0)
        # Isolates the wall-clock math from the Nuke-style render-pacing cap (see playback_tick),
        # which is what test_playback_falls_back_to_sequential_order_when_rendering_cannot_keep_up
        # exercises directly.
        w.playback_frames_rendered = 4
        w.playback_tick()
        self.assertEqual(w.dispatcher.document['time']['current'], 4)
        self.assertEqual(len(w.dispatcher.undo_stack), undo_slots)
        self.assertLessEqual(len(w.preview_queue), 4)
        w.toggle_playback(False)
        self.assertFalse(w.playing)
        self.assertEqual(w.play_button.text(), '▶')
        self.assertEqual(len(w.preview_queue), 0)

    def test_playback_falls_back_to_sequential_order_when_rendering_cannot_keep_up(self):
        # DiMo's report against the real 4K/ACES noise sequence: playback didn't just look slow,
        # it visibly jumped around non-sequentially (62, 14, 64, 17, 68 ...) because the wall
        # clock raced far ahead of a multi-second render and the transport showed "whatever
        # finished" rather than "the next frame in order" -- a documented tradeoff from the
        # original hard-freeze fix, but broken-looking at ACES 2.0/4K costs. DiMo chose Nuke's
        # fallback: play in order, slower than real time, never backward.
        w = self.window
        w.set_time(first=1, last=100, current=1, fps=24.0)
        w.toggle_playback(True)
        w.playback_origin_frame = 1
        # Simulate a wall clock that has raced 50 frames ahead while nothing has rendered yet --
        # exactly the native-4K-ACES scenario. The old wall-clock-chasing code would jump straight
        # to frame 51; the playhead must instead stay at the origin until something finishes.
        w.playback_origin_time = time.monotonic() - (50.0 / 24.0)
        w.playback_tick()
        self.assertEqual(w.dispatcher.document['time']['current'], 1)
        self.assertGreater(w.playback_dropped_frames, 0, 'still measures how far behind schedule this is')
        # The origin frame's own render finishes -- pacing may advance by exactly one frame, not
        # by however far the wall clock has since raced ahead.
        w.playback_frames_rendered = 1
        w.playback_tick()
        self.assertEqual(w.dispatcher.document['time']['current'], 2)
        # A second slow frame finishing unlocks exactly one more step, still strictly in order.
        w.playback_frames_rendered = 2
        w.playback_tick()
        self.assertEqual(w.dispatcher.document['time']['current'], 3)
        w.toggle_playback(False)

    def test_cancelled_render_does_not_advance_playback_pacing(self):
        w = self.window
        w.set_time(first=1, last=100, current=1, fps=24.0)
        w.toggle_playback(True)
        rendered_before = w.playback_frames_rendered
        request = FrameRequest(w.generation, 1, True, copy.deepcopy(w.dispatcher.document), playing=True)
        cancel = threading.Event()
        cancel.set()  # superseded before it finished -- must not count as render progress
        w.preview_ready((request, cancel), None, None, 'Cancelled', None)
        self.assertEqual(w.playback_frames_rendered, rendered_before)
        w.toggle_playback(False)

    def test_stale_preview_cannot_win(self):
        w = self.window
        for exposure in [1, 2, 3, -1]:
            w.command({'op': 'set', 'id': 'grade', 'param': 'exposure', 'value': exposure})
            APP.processEvents()
        self.assertTrue(wait_until(lambda: w.frame_generation == w.generation))
        from nodebased.imaging import Evaluator
        import numpy as np
        np.testing.assert_array_equal(w.frame, Evaluator().evaluate(w.dispatcher.document))

    def test_replaying_a_seen_frame_hits_the_display_cache(self):
        """After the ACES 2.0 view transform has paid its cost once for a given document, frame,
        and display setting, redisplaying it -- looping, scrubbing back, or returning a paused
        parameter to a value it already held -- must not pay it again."""
        # setUp's own wait for a first frame already primes one entry, so this checks that a
        # second, distinct request (a genuinely new document generation) is itself a hit rather
        # than starting the cache from empty.
        w = self.window
        entries_after_boot = len(w.display_cache)
        self.assertGreaterEqual(entries_after_boot, 1)
        w.generation += 1
        w.request_preview()
        self.assertTrue(wait_until(lambda: w.frame_generation == w.generation))
        self.assertIn('display cache hit', w.viewer_info.text())
        self.assertEqual(len(w.display_cache), entries_after_boot)

    def test_playback_auto_drops_proxy_above_hd_and_restores_it_on_stop(self):
        """Standard proxy-resolution playback: a source above HD is too slow for the ACES 2.0
        transform to sustain in real time, so playback temporarily drops quality and restores
        the artist's own choice -- Full -- the moment playback stops."""
        w = self.window
        w.dispatcher.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'big', 'type': 'Constant',
             'params': {'width': 3840, 'height': 2160, 'red': 0.2, 'green': 0.4, 'blue': 0.6, 'alpha': 1.0}},
            {'op': 'connect', 'id': 'viewer', 'input': 'image', 'source': 'big'},
        ]})
        self.assertEqual(w.proxy.currentData(), 1, 'starts at the artist default of Full')
        w.toggle_playback(True)
        self.assertGreater(w.proxy.currentData(), 1, 'a 4K source must auto-drop below Full to play')
        w.toggle_playback(False)
        self.assertEqual(w.proxy.currentData(), 1, 'stopping must restore the artist\'s own choice')

    def test_playback_never_overrides_a_manually_chosen_proxy(self):
        w = self.window
        w.dispatcher.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'big', 'type': 'Constant',
             'params': {'width': 3840, 'height': 2160, 'red': 0.2, 'green': 0.4, 'blue': 0.6, 'alpha': 1.0}},
            {'op': 'connect', 'id': 'viewer', 'input': 'image', 'source': 'big'},
        ]})
        for index in range(w.proxy.count()):
            if w.proxy.itemData(index) == 4:
                w.proxy.setCurrentIndex(index)
                break
        self.assertEqual(w.proxy.currentData(), 4, 'artist manually chose the smallest tier')
        w.toggle_playback(True)
        self.assertEqual(w.proxy.currentData(), 4, 'playback must not second-guess a manual choice')
        w.toggle_playback(False)
        self.assertEqual(w.proxy.currentData(), 4, 'stopping must not touch a tier it never changed')

    def test_same_generation_wrong_frame_cannot_enter_viewer(self):
        import numpy as np
        w = self.window
        original = w.frame.copy()
        wrong = np.zeros_like(original)
        current = w.dispatcher.document['time']['current']
        request = FrameRequest(w.generation, current + 1, True,
                               copy.deepcopy(w.dispatcher.document))
        cancel = threading.Event()
        w.preview_ready((request, cancel), wrong, to_qimage(wrong), 'wrong frame')
        np.testing.assert_array_equal(w.frame, original)

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

    def test_exr_export_filter_selects_half_or_float(self):
        """The chosen dialog filter must decide the pixel type on disk.

        The half/float choice is carried only by the filter string the dialog hands back,
        so a reworded filter would otherwise silently downgrade a deliberate 32-bit
        export to half with no test noticing.
        """
        from unittest.mock import patch
        import OpenImageIO as oiio
        w = self.window
        with tempfile.TemporaryDirectory() as folder:
            for chosen, expected in [('OpenEXR half RGBA ZIPS (*.exr)', 'half'),
                                     ('OpenEXR 32-bit float RGBA ZIPS (*.exr)', 'float')]:
                path = str(Path(folder) / f'{expected}.exr')
                with patch('nodebased.app.QFileDialog.getSaveFileName',
                           return_value=(path, chosen)):
                    w.export()
                spec = oiio.ImageInput.open(path).spec()
                self.assertEqual(str(spec.format), expected, chosen)
                self.assertTrue(spec.get_string_attribute('compression').startswith('zips'))

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


class SlowPlaybackTests(unittest.TestCase):
    """Playback must drop frames, never stall.

    Reported against v0.10.0: an EXR sequence froze on whatever frame play was pressed on
    while the timeline kept advancing, and scrubbing the same sequence was fine. The cause
    was not the sequence — it was that every playback tick cancelled the render in flight
    and the display gate additionally required the finished frame to still be the playhead.
    A frame costing more than one frame interval could therefore never be shown, so the
    viewer sat on its last image no matter how long playback ran. The same failure
    reproduces on v0.9.1, so this is a latent transport defect rather than an animation
    regression. These tests pin the behaviour with a render deliberately slower than the
    frame interval.
    """

    def setUp(self):
        self.endpoint = 'nodebased-slow-' + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f'no first frame cooked within {WAIT_TIMEOUT:.0f}s')
        # Keep the real graph path but make its cost negligible beside the deliberate 120 ms
        # delay. Hosted Linux runners vary wildly in raster speed; letting the 960x540 demo frame
        # dominate made this transport test fail or pass based on runner load.
        self.window.dispatcher.execute({'op': 'batch', 'commands': [
            {'op': 'set', 'id': 'plate', 'param': 'width', 'value': 64},
            {'op': 'set', 'id': 'plate', 'param': 'height', 'value': 64},
            {'op': 'set', 'id': 'wash', 'param': 'width', 'value': 64},
            {'op': 'set', 'id': 'wash', 'param': 'height', 'value': 64},
        ]})
        self.displayed = []
        self.displayed_generations = []
        original = self.window.preview_ready

        def spy(payload, frame, image, status, render_region=None):
            request = payload[0]
            before = self.window.frame_generation
            original(payload, frame, image, status, render_region)
            if request.display and self.window.frame_generation != before:
                self.displayed.append(request.frame)
                self.displayed_generations.append(request.generation)

        self.window.signals.finished.disconnect()
        self.window.signals.finished.connect(spy)

    def tearDown(self):
        self.window.toggle_playback(False)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def _slow_down(self, seconds=0.12):
        """Make every evaluation cost more than one frame interval at 24 fps (~42 ms)."""
        evaluator = self.window.evaluator
        original = evaluator.evaluate
        tile_executor = self.window.tile_executor
        original_compose = tile_executor.compose_region

        def slow(*args, **kwargs):
            time.sleep(seconds)
            return original(*args, **kwargs)

        evaluator.evaluate = slow
        def slow_compose(*args, **kwargs):
            time.sleep(seconds)
            return original_compose(*args, **kwargs)

        tile_executor.compose_region = slow_compose
        self.addCleanup(lambda: setattr(evaluator, 'evaluate', original))
        self.addCleanup(lambda: setattr(tile_executor, 'compose_region', original_compose))

    def _play_for(self, seconds):
        # The range must be long enough that a 2.5 s run can never wrap back onto a frame it has
        # already shown. It used to be 1-8: harmless before the display cache existed, but once a
        # repeated frame became a cheap cache hit, an 8-frame loop replayed too easily inside the
        # window, sometimes catching up to every distinct frame and defeating the very drop
        # behaviour this class exists to verify. A range this long guarantees every request here
        # is a genuine first-time miss, independent of whatever caching lives further down.
        self.window.set_time(first=1, last=200, current=1, fps=24.0)
        self.displayed.clear()
        self.displayed_generations.clear()
        self.visited = []
        self.window.toggle_playback(True)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            APP.processEvents()
            self.visited.append(self.window.dispatcher.document['time']['current'])
            QTest.qWait(10)
        self.window.toggle_playback(False)
        APP.processEvents()

    def test_slow_frames_still_reach_the_viewer_instead_of_freezing(self):
        self._slow_down()
        self._play_for(2.5)
        self.assertGreater(self.window.dispatcher.document['time']['current'], 1,
                           'transport did not advance, so this does not test the freeze')
        self.assertGreater(len(set(self.displayed)), 1,
                           'viewer froze: a render slower than one frame interval never '
                           'reached the viewer, which is the reported bug')

    def test_slow_playback_drops_frames_rather_than_queueing_them(self):
        self._slow_down()
        self._play_for(2.5)
        self.assertGreater(len(set(self.displayed)), 1,
                           'viewer drew nothing, so "dropped rather than queued" is vacuous here')
        self.assertLessEqual(len(self.window.preview_queue), 1 + MAX_PREFETCH,
                             'read-ahead grew past its bound while rendering fell behind')
        # The transport follows the wall clock, so at 24 fps against a ~120 ms render it must
        # pass through more timeline positions than the viewer manages to draw. Frames being
        # skipped is the correct outcome; frames being queued up (or none drawn at all) is not.
        self.assertGreater(len(set(self.visited)), len(set(self.displayed)),
                           'nothing was dropped, so this run never exercised falling behind')

    def test_displayed_frames_never_go_backwards_during_playback(self):
        self._slow_down()
        self._play_for(2.5)
        # Catching-up display accepts a result the playhead has passed, so it must still
        # refuse anything older than what is already on screen.
        self.assertTrue(all(b > a for a, b in zip(self.displayed_generations,
                                                  self.displayed_generations[1:])),
                        f'viewer accepted stale generations: {self.displayed_generations}')

    def test_transport_tick_does_not_cancel_the_render_in_flight(self):
        w = self.window
        w.set_time(first=1, last=8, current=1, fps=24.0)
        w.toggle_playback(True)
        w.request_preview(playhead_only=True)
        request, active = w.preview_queue.take()
        self.assertFalse(active.is_set())
        w.request_preview(playhead_only=True)
        self.assertFalse(active.is_set(),
                         'a playback tick cancelled the frame being rendered, which is what '
                         'made playback freeze instead of dropping frames')
        w.toggle_playback(False)

    def test_content_change_during_playback_still_cancels(self):
        w = self.window
        w.set_time(first=1, last=8, current=1, fps=24.0)
        w.toggle_playback(True)
        w.request_preview(playhead_only=True)
        _, active = w.preview_queue.take()
        self.assertFalse(active.is_set())
        # A graph edit invalidates content, so it must still cancel even mid-playback.
        w.request_preview()
        self.assertTrue(active.is_set(),
                        'a content change mid-playback must still cancel in-flight work')
        w.toggle_playback(False)


class KeyframeUiTests(unittest.TestCase):
    """The curve engine and its three atomic ops already existed and were agent-only. These
    cover the half that was missing: reaching them from the properties panel, and reporting the
    result on the timeline strip."""

    def setUp(self):
        self.endpoint = 'nodebased-test-' + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f'no first frame cooked within {WAIT_TIMEOUT:.0f}s')
        self.window.set_time(first=1, last=20, current=1)
        self.select('grade')

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def select(self, node_id):
        self.window.graph.items_by_id[node_id].setSelected(True)
        APP.processEvents()

    def editor(self):
        return self.window.properties.findChildren(QDoubleSpinBox)[0]

    def key_button(self):
        """The keyframe button belonging to the first numeric knob."""
        buttons = [b for b in self.window.properties.findChildren(QPushButton)
                   if b.text() in ('○', '◇', '◆')]
        self.assertTrue(buttons, 'no keyframe button on a numeric knob')
        return buttons[0]

    def curve(self):
        return (self.window.dispatcher.document.get('animation') or {}) \
            .get('curves', {}).get('grade', {}).get('exposure')

    def test_key_button_sets_a_key_at_the_playhead_and_toggles_it_off(self):
        w = self.window
        self.assertEqual(self.key_button().text(), '○')
        self.editor().setValue(1.5)
        self.key_button().click()
        self.assertTrue(wait_until(lambda: self.curve() is not None))
        self.assertEqual([(k['frame'], k['value']) for k in self.curve()['keys']], [(1, 1.5)])
        # The panel is rebuilt from the document, so the button now reports the key it made.
        self.assertTrue(wait_until(lambda: self.key_button().text() == '◆'))
        self.key_button().click()
        self.assertTrue(wait_until(lambda: self.curve() is None))

    def test_a_key_is_an_ordinary_undoable_document_edit(self):
        w = self.window
        self.editor().setValue(1.5)
        self.key_button().click()
        self.assertTrue(wait_until(lambda: self.curve() is not None))
        w.command({'op': 'undo'})
        self.assertIsNone(self.curve())

    def test_editing_an_animated_knob_keys_the_frame_instead_of_the_base(self):
        # The confusing failure this prevents: typing into an animated knob writes the base
        # parameter, the curve immediately overrides it, and the viewer does not move -- which
        # reads as "the comp is ignoring my input".
        w = self.window
        base = w.dispatcher.document['nodes']['grade']['params']['exposure']
        self.editor().setValue(1.5)
        self.key_button().click()
        self.assertTrue(wait_until(lambda: self.curve() is not None))
        w.set_time(current=10)
        APP.processEvents()
        editor = self.editor()
        editor.setValue(3.0)
        editor.editingFinished.emit()
        self.assertTrue(wait_until(lambda: len(self.curve()['keys']) == 2))
        self.assertEqual([(k['frame'], k['value']) for k in self.curve()['keys']],
                         [(1, 1.5), (10, 3.0)])
        self.assertEqual(w.dispatcher.document['nodes']['grade']['params']['exposure'], base)

    def test_an_animated_knob_shows_the_value_in_use_at_the_current_frame(self):
        w = self.window
        w.command({'op': 'set_key', 'id': 'grade', 'param': 'exposure', 'frame': 1, 'value': 0.0,
                   'interpolation': 'linear'}, render=False)
        w.command({'op': 'set_key', 'id': 'grade', 'param': 'exposure', 'frame': 11,
                   'value': 10.0}, render=False)
        self.select('grade')
        w.set_time(current=6)
        self.assertTrue(wait_until(lambda: abs(self.editor().value() - 5.0) < 1e-6),
                        f'knob shows {self.editor().value()} at the midpoint of a 0->10 ramp')

    def test_keyed_frames_are_reported_on_the_timeline_and_follow_the_selection(self):
        w = self.window
        for frame in (3, 7):
            w.command({'op': 'set_key', 'id': 'grade', 'param': 'exposure', 'frame': frame,
                       'value': 1.0}, render=False)
        self.assertTrue(wait_until(lambda: w.frame_slider.key_frames == {3, 7}))
        # Selecting an unanimated node scopes the band to it, the way Nuke's does.
        self.select('wash')
        w.graph.items_by_id['grade'].setSelected(False)
        APP.processEvents()
        self.assertTrue(wait_until(lambda: w.frame_slider.key_frames == set()),
                        f'band still showing another node\'s keys: {w.frame_slider.key_frames}')

    def test_cached_frames_are_reported_on_the_timeline(self):
        w = self.window
        self.assertTrue(wait_until(lambda: w.frame_slider.cached_frames == {1}),
                        f'first cooked frame not reported as cached: {w.frame_slider.cached_frames}')
        w.set_time(current=2)
        self.assertTrue(wait_until(lambda: w.frame_slider.cached_frames >= {1, 2}))
        # A graph edit changes the viewer identity, so nothing already rendered still applies.
        w.command({'op': 'set', 'id': 'grade', 'param': 'exposure', 'value': 0.75})
        self.assertTrue(wait_until(lambda: w.frame_slider.cached_frames == {2}),
                        f'stale cache reported after an edit: {w.frame_slider.cached_frames}')


class PlaybackProxyToggleTests(unittest.TestCase):
    """Proxy-while-playing used to be unconditional. DiMo asked for it to be the artist's
    choice, so the toggle has to actually gate the auto-switch in both positions."""

    def setUp(self):
        self.endpoint = 'nodebased-test-' + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f'no first frame cooked within {WAIT_TIMEOUT:.0f}s')
        self.window.set_time(first=1, last=8, current=1, fps=24.0)
        # The auto-switch only fires above HD, so a demo-sized 960x540 graph would make every
        # assertion below pass no matter how the toggle is wired. Push the source past the
        # threshold first; 2560x1440 lands on tier 2 without paying 4K render time per test.
        for node in ('plate', 'wash'):
            self.window.command({'op': 'set', 'id': node, 'param': 'width', 'value': 2560},
                                render=False)
            self.window.command({'op': 'set', 'id': node, 'param': 'height', 'value': 1440},
                                render=False)

    def tearDown(self):
        if self.window.playing:
            self.window.toggle_playback(False)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_defaults_to_on(self):
        self.assertTrue(self.window.playback_proxy.isChecked())

    def test_checked_auto_switches_for_the_duration_of_playback_and_restores_after(self):
        # The negative control for the test below: same graph, toggle on, tier really does move.
        w = self.window
        self.assertEqual(w.proxy.currentData(), 1)
        w.toggle_playback(True)
        APP.processEvents()
        self.assertEqual(w.proxy.currentData(), 2,
                         'above-HD source did not drop to a proxy tier for playback')
        self.assertIsNotNone(w.playback_auto_proxy_index)
        w.toggle_playback(False)
        self.assertEqual(w.proxy.currentData(), 1, 'the artist\'s tier was not restored on stop')
        self.assertIsNone(w.playback_auto_proxy_index)

    def test_unchecked_plays_at_the_selected_tier_and_never_auto_switches(self):
        w = self.window
        w.playback_proxy.setChecked(False)
        self.assertEqual(w.proxy.currentData(), 1)
        w.toggle_playback(True)
        APP.processEvents()
        self.assertIsNone(w.playback_auto_proxy_index,
                          'proxy was auto-switched with the toggle off')
        self.assertEqual(w.proxy.currentData(), 1,
                         'viewer silently changed resolution with the toggle off')
        w.toggle_playback(False)
        self.assertEqual(w.proxy.currentData(), 1)

    def test_an_explicit_proxy_tier_is_never_overridden_even_with_the_toggle_on(self):
        w = self.window
        self.assertTrue(w.playback_proxy.isChecked())
        quarter = w.proxy.findData(4)
        self.assertNotEqual(quarter, -1)
        w.proxy.setCurrentIndex(quarter)
        w.toggle_playback(True)
        APP.processEvents()
        self.assertIsNone(w.playback_auto_proxy_index)
        self.assertEqual(w.proxy.currentData(), 4)
        w.toggle_playback(False)
        self.assertEqual(w.proxy.currentData(), 4)
