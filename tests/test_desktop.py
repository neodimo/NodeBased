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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QLineEdit, QPushButton,
                               QGraphicsSimpleTextItem, QToolBar, QMenu, QMessageBox, QCheckBox, QPlainTextEdit)
from nodebased.app import (Window, thumbnail_key, STYLE, NodeSearch, ProjectSettingsDialog, Preferences,
                           SequenceBrowser, ElidedLabel)
from nodebased.theme import COLORS, THEMES, DEFAULT_THEME, build_style
import unittest.mock
from nodebased.imaging import to_qimage
from nodebased.playback import DisplayCache
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
        self.assertEqual(merge.inputs['A'].pos().x(), 0)
        self.assertEqual(merge.inputs['B'].pos().x(), 95)
        desired = merge.pos()
        merge_rect = merge.sceneBoundingRect()
        w.add_node('Grade', position=desired)
        created = next(item for key, item in w.graph.items_by_id.items() if key not in {'plate', 'wash', 'grade', 'merge', 'viewer'})
        self.assertFalse(created.sceneBoundingRect().intersects(merge_rect))

    def test_node_titles_are_large_and_centred(self):
        node = self.window.graph.items_by_id['grade']
        title = next(item for item in node.childItems() if isinstance(item, QGraphicsSimpleTextItem))
        self.assertGreaterEqual(title.font().pointSize(), 14)
        self.assertAlmostEqual(title.pos().x() + title.boundingRect().width() / 2, 95, places=3)

    def test_disabled_nodes_are_dimmed_and_keep_merge_b_input_visible(self):
        w = self.window
        w.command({'op': 'disable', 'id': 'merge', 'value': True}, render=False)
        merge = w.graph.items_by_id['merge']
        self.assertTrue(merge.disabled)
        self.assertLess(merge.opacity(), 1.0)
        self.assertIn('B', merge.inputs)
        self.assertEqual(merge.inputs['B'].pos(), QPointF(95, 0))

    def test_mask_is_on_right_and_a_is_on_left(self):
        grade = self.window.graph.items_by_id['grade']
        self.assertEqual(grade.inputs['image'].pos(), QPointF(95, 0))
        self.assertEqual(grade.inputs['mask'].pos(), QPointF(190, 26))
        merge = self.window.graph.items_by_id['merge']
        self.assertEqual(merge.inputs['A'].pos(), QPointF(0, 26))
        self.assertEqual(merge.inputs['mask'].pos(), QPointF(190, 26))

    def test_merge_b_is_the_top_centre_trunk(self):
        # In Nuke the B stream runs straight down through a Merge; A and mask join from the sides.
        merge = self.window.graph.items_by_id['merge']
        self.assertEqual(merge.inputs['B'].pos(), QPointF(95, 0))
        self.assertEqual(set(merge.inputs), {'A', 'B', 'mask'})

    def test_only_sources_show_thumbnails_by_default_and_the_node_tab_toggles_them(self):
        w = self.window
        w.set_show_thumbnails(True)
        w.thumbnail_timer.start(0)
        self.assertTrue(wait_until(lambda: {'plate', 'wash'} <= set(w.thumbnails)),
                        'source thumbnails never arrived')
        plate = w.graph.items_by_id['plate']
        self.assertFalse(plate.thumbnail.pixmap().isNull())
        self.assertEqual(plate.output.pos().y(), plate.rect().height())
        for key in ('grade', 'merge', 'viewer'):
            self.assertIsNone(w.graph.items_by_id[key].thumbnail, key)
            self.assertEqual(w.graph.items_by_id[key].rect().height(), 52)
        # The Node tab switches a filter's stamp on; the document stores only the override.
        w.graph.items_by_id['grade'].setSelected(True)
        w.inspect('grade')
        tabs = w.properties.widget()
        self.assertEqual([tabs.tabText(i) for i in range(tabs.count())], ['Grade', 'Node'])
        tabs.findChild(QCheckBox, 'node-thumbnail').setChecked(True)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade'].get('thumbnail') is True))
        self.assertTrue(wait_until(lambda: 'grade' in w.thumbnails), 'grade thumbnail never arrived')
        self.assertIsNotNone(w.graph.items_by_id['grade'].thumbnail)
        # Moving a node cannot change its picture, so it keeps the same thumbnail identity.
        before = w.thumbnails['grade'][0]
        w.command({'op': 'move', 'id': 'grade', 'pos': [-400, -160]}, render=False)
        doc = w.dispatcher.document
        self.assertEqual(before, thumbnail_key(doc, 'grade', doc['time']['current'],
                                               w.display_view.currentText()))
        w.set_show_thumbnails(False)
        self.assertIsNone(w.graph.items_by_id['plate'].thumbnail)
        self.assertEqual(w.graph.items_by_id['plate'].rect().height(), 52)
        w.set_show_thumbnails(True)

    def test_the_node_tab_labels_and_disables_a_node(self):
        w = self.window
        # Edits rebuild the panel for the graph selection, so select the node being edited.
        w.graph.items_by_id['grade'].setSelected(True)
        w.inspect('grade')
        tabs = w.properties.widget()
        label = tabs.findChild(QPlainTextEdit, 'node-label')
        label.setPlainText('key light\nsecond line')
        label.finished.emit()
        self.assertTrue(wait_until(
            lambda: w.dispatcher.document['nodes']['grade'].get('label') == 'key light\nsecond line'))
        self.assertIn('key light', [child.text() for child in w.graph.items_by_id['grade'].childItems()
                                    if hasattr(child, 'text')])
        enabled = w.properties.widget().findChild(QCheckBox, 'node-enabled')
        self.assertTrue(enabled.isChecked())
        enabled.setChecked(False)
        self.assertTrue(wait_until(lambda: w.dispatcher.document['nodes']['grade']['disabled']))
        # The Node tab stays the open tab across the rebuild the edit caused.
        w.properties.widget().setCurrentIndex(1)
        w.inspect('grade')
        self.assertEqual(w.properties.widget().currentIndex(), 1)
        w.inspect('plate')
        self.assertFalse(w.properties.widget().findChild(QCheckBox, 'node-enabled').isEnabled())

    def test_the_cache_band_counts_frames_cached_at_any_proxy_tier(self):
        w = self.window
        document = w.dispatcher.document
        frame = document['time']['current']
        identity = DisplayCache.identity(document, document['view'], 2, w.display_view.currentText(),
                                         w.exposure.value(), w.channels.currentText(),
                                         document['settings']['viewer']['background'])
        w.display_cache.put((identity, frame), b'\x00' * 64, 4, 4, 16)
        seen = {}
        w.frame_slider.set_marks = lambda cached, keyed: seen.update(cached=set(cached))
        w.refresh_timeline_marks()
        self.assertIn(frame, seen['cached'])

    def test_moving_or_labelling_a_node_does_not_rerender_the_viewer(self):
        w = self.window
        self.assertTrue(wait_until(lambda: not w.busy and not len(w.preview_queue)))
        generation = w.generation
        w.command({'op': 'move', 'id': 'grade', 'pos': [-420, -160]})
        w.command({'op': 'rename', 'id': 'grade', 'name': 'Hero grade'})
        w.command({'op': 'label', 'id': 'merge', 'value': 'comp'})
        # A branch the viewer does not see is not a reason to render either.
        w.command({'op': 'create', 'id': 'loose', 'type': 'Blur'})
        w.command({'op': 'set', 'id': 'loose', 'param': 'radius', 'value': 4.0})
        self.assertEqual(w.generation, generation)
        w.command({'op': 'set', 'id': 'grade', 'param': 'exposure', 'value': 1.25})
        self.assertGreater(w.generation, generation)

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

    def test_a_node_added_to_a_selection_lands_underneath_it(self):
        w = self.window
        rect = lambda key: w.graph.items_by_id[key].sceneBoundingRect()

        def add_below(parent, kind):
            known = set(w.graph.items_by_id)
            w.graph.scene().clearSelection()
            w.graph.items_by_id[parent].setSelected(True)
            w.add_node(kind)
            # Edits rebuild the scene items, so everything is looked up by id afterwards.
            return next(key for key in w.graph.items_by_id if key not in known)

        blur = add_below('grade', 'Blur')
        above, below = rect('grade'), rect(blur)
        self.assertAlmostEqual(below.center().x(), above.center().x(), delta=1)
        self.assertGreater(below.top(), above.bottom())
        others = [rect(key) for key in w.graph.items_by_id if key != blur]
        self.assertFalse(any(below.intersects(other) for other in others))
        # Adding again stacks further down the same column instead of stepping sideways.
        second = add_below(blur, 'Grade')
        self.assertAlmostEqual(rect(second).center().x(), rect(blur).center().x(), delta=1)
        self.assertGreater(rect(second).top(), rect(blur).bottom())

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
        # Zoomed out, a Dot's socket hit areas used to cover its centre. The Dot must still win.
        graph.resetTransform()
        graph.scale(0.45, 0.45)
        graph.centerOn(dot)
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
        # Falling behind means the wall clock outran rendering: at 24 fps against a ~120 ms render
        # far more frame intervals elapse than frames finish. Assert that directly off the
        # transport's own counters.
        #
        # Not off sampled playhead positions, which is what this used to do. `playback_tick`
        # deliberately clamps the playhead to `playback_frames_rendered` so a slow pipeline
        # degrades to strictly in-order playback rather than racing ahead and jumping back, so
        # under that clamp the playhead visits about as many positions as get drawn -- by design.
        # The old comparison therefore asserted the opposite of the documented behaviour and
        # failed whenever the machine was loaded enough to make the clamp bite.
        self.assertGreater(self.window.playback_elapsed_frames,
                           self.window.playback_frames_rendered,
                           'rendering kept up, so this run never exercised falling behind')

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


class ExpressionUiTests(unittest.TestCase):
    """The expression engine must be reachable from the same inspector artists use for knobs."""

    def setUp(self):
        self.endpoint = 'nodebased-test-' + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        # Expression editing is a document/inspector interaction; it does not need to wait for
        # the demo image's asynchronous preview. Avoid coupling these tests to OCIO availability
        # or a cold CI renderer when the assertions below only inspect Dispatcher state and Qt
        # controls.
        self.window.set_time(first=1, last=20, current=1)
        self.window.graph.items_by_id['grade'].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def expression_editor(self):
        editors = self.window.properties.findChildren(QLineEdit)
        self.assertTrue(editors, 'numeric inspector has no expression editor')
        for editor in editors:
            if editor.accessibleName() == 'grade.exposure expression':
                return editor
        self.fail('grade exposure expression editor was not exposed by the inspector')

    def expression_button(self, object_name):
        row = self.expression_editor().parentWidget()
        button = row.findChild(QPushButton, object_name)
        self.assertIsNotNone(button, f'missing {object_name} button')
        return button

    def test_set_expression_displays_driven_state_and_resolved_value(self):
        w = self.window
        editor = self.expression_editor()
        editor.setText('frame * 0.5')
        self.expression_button('set-expression').click()
        self.assertTrue(wait_until(lambda: w.dispatcher.document['expressions']['grade']['exposure']
                                   == 'frame * 0.5'))
        knob = w.properties.findChildren(QDoubleSpinBox)[0]
        self.assertFalse(knob.isEnabled(), 'expression-driven knob should not invite base edits')
        key_buttons = [b for b in w.properties.findChildren(QPushButton) if b.text() == 'ƒ']
        self.assertEqual(len(key_buttons), 1)
        self.assertFalse(key_buttons[0].isEnabled())
        w.set_time(current=6)
        self.assertTrue(wait_until(lambda: abs(w.properties.findChildren(QDoubleSpinBox)[0].value() - 3)
                                   < 1e-6))

    def test_clear_expression_restores_base_and_is_undoable(self):
        w = self.window
        editor = self.expression_editor()
        editor.setText('frame * 0.5')
        self.expression_button('set-expression').click()
        self.assertTrue(wait_until(lambda: 'exposure' in w.dispatcher.document['expressions']['grade']))
        self.expression_button('clear-expression').click()
        self.assertTrue(wait_until(lambda: 'grade' not in w.dispatcher.document['expressions']))
        self.assertTrue(w.properties.findChildren(QDoubleSpinBox)[0].isEnabled())
        w.command({'op': 'undo'})
        self.assertEqual(w.dispatcher.document['expressions']['grade']['exposure'], 'frame * 0.5')

    def test_invalid_expression_stays_out_of_document_and_reports_error(self):
        w = self.window
        editor = self.expression_editor()
        editor.setText('unknown_name + 1')
        self.expression_button('set-expression').click()
        # The command is deferred and the status-bar message is transient: an in-flight preview
        # may replace it before the event loop returns. The persistent command-error surface is
        # the deterministic UI contract; validation still has to reject the edit atomically.
        self.assertTrue(wait_until(lambda: 'Unknown name' in w.command_error_label.text()))
        self.assertEqual(w.dispatcher.document['expressions'], {})


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


class DecodeAheadPlaybackTests(unittest.TestCase):
    """`Window.decode_pool` (decodepool.DecodeAheadPool): playback must warm it for the Read
    ancestors of the viewed target, a plain playhead tick must never invalidate it (that would
    defeat read-ahead every 40ms), and a real seek/edit must -- see docs/PLAYBACK.md criterion 3
    for the same tick-vs-content-change distinction `PlaybackQueue.replace` already draws."""

    def setUp(self):
        self.endpoint = 'nodebased-test-' + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None),
                        f'no first frame cooked within {WAIT_TIMEOUT:.0f}s')
        self.temp = tempfile.TemporaryDirectory()
        from nodebased.imaging import write_png
        self.plate_path = str(Path(self.temp.name) / 'plate.####.png')
        # Above the HD threshold `tiers.auto_playback_tier` gates on: the decode-ahead pool is
        # only ever consulted by the tier != 1 Read path (see the tier == 1 gate in
        # Window.request_preview), so a source at or below HD would never engage it and this
        # whole test class would be exercising nothing.
        for frame in range(1, 6):
            write_png(self.plate_path.replace('####', f'{frame:04d}'),
                     __import__('numpy').full((1080, 2000, 4), 0.5, 'float32'))
        self.window.command({'op': 'create', 'type': 'Read', 'id': 'qa_read',
                            'params': {'path': self.plate_path}}, render=False)
        self.window.command({'op': 'view', 'id': 'qa_read'}, render=False)
        self.window.set_time(first=1, last=5, current=1, fps=24.0)
        self.assertTrue(wait_until(lambda: self.window.frame_generation == self.window.generation))

    def tearDown(self):
        if self.window.playing:
            self.window.toggle_playback(False)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        # Product shutdown is deliberately non-blocking because a native OIIO decode cannot be
        # interrupted safely. This test owns temporary source files, though, and Windows refuses
        # to unlink a file while OIIO still has it open. Join the already-shutting-down daemon
        # workers here before deleting their fixture directory; this is test resource cleanup,
        # not a change to the GUI's prompt-close contract.
        self.window.decode_pool.shutdown(wait=True, timeout=WAIT_TIMEOUT)
        self.assertEqual(self.window.decode_pool.stats()['pending'], 0,
                         'decode workers still hold temporary source files after bounded join')
        self.temp.cleanup()

    def test_playback_warms_the_decode_pool_within_its_memory_bound(self):
        w = self.window
        w.toggle_playback(True)
        self.assertTrue(wait_until(lambda: w.decode_pool.stats()['entries'] > 0))
        stats = w.decode_pool.stats()
        self.assertLessEqual(stats['bytes'], w.decode_pool.budget)
        w.toggle_playback(False)

    def test_a_plain_playhead_tick_does_not_bump_the_epoch(self):
        w = self.window
        w.toggle_playback(True)
        epoch_before = w.decode_pool.epoch
        w.playback_frames_rendered = 1
        w.playback_origin_frame = w.dispatcher.document['time']['current']
        w.playback_origin_time = time.monotonic() - (1 / 24.0)
        w.playback_tick()
        self.assertEqual(w.decode_pool.epoch, epoch_before,
                         'a playback tick must not invalidate in-flight read-ahead decodes')
        w.toggle_playback(False)

    def test_a_scrub_during_playback_bumps_the_epoch(self):
        w = self.window
        w.toggle_playback(True)
        epoch_before = w.decode_pool.epoch
        w.frame_slider.setValue(3)  # A real seek, not a transport-driven tick.
        self.assertGreater(w.decode_pool.epoch, epoch_before,
                           'a seek must invalidate read-ahead decodes queued for the old position')
        w.toggle_playback(False)

    def test_stopping_playback_shuts_down_cleanly_with_a_warm_pool(self):
        w = self.window
        w.toggle_playback(True)
        wait_until(lambda: w.decode_pool.stats()['entries'] > 0)
        w.toggle_playback(False)
        self.assertFalse(w.playing)  # No hang or exception tearing down a populated pool.


class InspectorMenuTests(unittest.TestCase):
    """A right-click in the properties panel must leave a menu on screen.

    The original bug was not in the menu code at all: `editingFinished` fires on focus-out,
    including focus lost to a popup, so opening a menu re-submitted the value the document
    already held -- and the unconditional panel rebuild that followed deleted the widget the
    menu was parented to. The menu appeared and vanished a frame later.
    """
    def setUp(self):
        self.endpoint = 'nodebased-test-' + uuid.uuid4().hex
        self.window = Window(agent_name=self.endpoint)
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.key = next(k for k, n in self.window.dispatcher.document['nodes'].items()
                        if n['type'] == 'Grade')
        self.window.graph.scene().clearSelection()
        self.window.graph.items_by_id[self.key].setSelected(True)
        APP.processEvents()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def settle(self):
        for _ in range(12):
            APP.processEvents()
            QTest.qWait(5)

    def test_a_noop_knob_focus_out_keeps_the_properties_panel_alive(self):
        w = self.window
        panel = w.properties.widget()
        revision = w.dispatcher.revision
        spin = panel.findChildren(QDoubleSpinBox)[0]
        spin.editingFinished.emit()      # exactly what focus moving to a popup menu does
        self.settle()
        self.assertEqual(w.dispatcher.revision, revision, 'a focus-out must not edit the document')
        self.assertIs(w.properties.widget(), panel,
                      'the panel was rebuilt for an edit that changed nothing, which is what '
                      'destroyed the context menu parented inside it')

    def test_a_noop_name_focus_out_keeps_the_properties_panel_alive(self):
        w = self.window
        panel = w.properties.widget()
        name = next(e for e in panel.findChildren(QLineEdit)
                    if e.text() == w.dispatcher.document['nodes'][self.key]['name'])
        name.editingFinished.emit()
        self.settle()
        self.assertIs(w.properties.widget(), panel)

    def test_a_real_knob_edit_still_rebuilds_the_panel(self):
        w = self.window
        panel = w.properties.widget()
        spin = panel.findChildren(QDoubleSpinBox)[0]
        spin.setValue(spin.value() + 0.5)
        spin.editingFinished.emit()
        self.settle()
        self.assertIsNot(w.properties.widget(), panel,
                         'an edit that changes the document must still refresh the inspector')

    def test_knobs_carry_the_animation_menu_themselves(self):
        # Nuke parity: right-clicking the number is how a knob is keyed; the diamond is a shortcut.
        panel = self.window.properties.widget()
        spins = panel.findChildren(QDoubleSpinBox)
        self.assertTrue(spins)
        for spin in spins:
            self.assertEqual(spin.contextMenuPolicy(), Qt.ContextMenuPolicy.CustomContextMenu)

    def test_the_curve_menu_is_owned_by_the_window_not_the_panel(self):
        # A menu parented into the inspector dies with it. Proven by construction: build the menu
        # the same way curve_menu does and check its parent survives a panel rebuild.
        w = self.window
        panel = w.properties.widget()
        menu = QMenu(w)
        w.inspect(self.key)
        self.settle()
        self.assertIsNot(w.properties.widget(), panel)
        self.assertIs(menu.parent(), w)


class LayoutStabilityTests(unittest.TestCase):
    """The viewer must never resize the windows around it. Only the user dragging a splitter
    handle is allowed to change the layout."""
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def splitter(self):
        return self.window.centralWidget()

    def test_a_long_playback_status_does_not_resize_the_layout(self):
        w = self.window
        before_window = w.size()
        before_sizes = self.splitter().sizes()
        before_minimum = w.minimumSizeHint().width()
        # The exact shape playback appends: "ahead 8/8 · dropped 137".
        w.viewer_info.setText('3840 × 2160  ·  412 ms  ·  tiles 240 hit/18 miss  ·  '
                              'cache 812.5 / 1024.0 MB  ·  ahead 8/8  ·  dropped 137')
        w.command_error_label.setText('connect: input "mask" references a missing node '
                                      'and the document was left untouched')
        APP.processEvents()
        QTest.qWait(20)
        APP.processEvents()
        self.assertEqual(w.size(), before_window, 'status text resized the main window')
        self.assertEqual(self.splitter().sizes(), before_sizes,
                         'status text redistributed the splitter panels')
        self.assertLessEqual(w.minimumSizeHint().width(), before_minimum,
                             'status text raised the window minimum width')

    def test_the_full_status_is_still_readable_after_elision(self):
        # Eliding is a painting decision. The stored status stays whole, because the agent bridge
        # and the artist both read it as the real result of the last render.
        status = 'x' * 400
        self.window.viewer_info.setText(status)
        self.assertEqual(self.window.viewer_info.text(), status)
        self.assertEqual(self.window.viewer_info.toolTip(), status)

    def test_a_larger_comp_format_does_not_resize_the_layout(self):
        w = self.window
        before_window, before_sizes = w.size(), self.splitter().sizes()
        # Every source feeding the viewed Merge has to grow together: resizing one of them alone
        # is a format mismatch, and the resulting evaluation error would never reach the viewer,
        # so the test would pass or fail on the error path instead of on the layout.
        sources = [k for k, n in w.dispatcher.document['nodes'].items()
                   if n['type'] in ('Constant', 'Checker')]
        # 1080p rather than 4K on purpose. What is under test is that *growing* the format leaves
        # the layout alone, and any enlargement past the viewport proves that equally well. A 4K
        # render costs 21s of real evaluation on the development machine against a 30s budget, so
        # it passed here and timed out on the slower CI runners -- a test that measures runner
        # speed rather than layout. This size renders in about 6s and keeps the margin honest.
        w.command({'op': 'batch', 'commands': [
            {'op': 'set', 'id': key, 'param': param, 'value': value}
            for key in sources for param, value in (('width', 1920), ('height', 1080))]})
        self.assertTrue(wait_until(lambda: w.viewer.sceneRect().width() == 1920),
                        'the viewer never picked up the larger format')
        self.assertEqual(w.size(), before_window, 'a larger format resized the main window')
        self.assertEqual(self.splitter().sizes(), before_sizes,
                         'a larger format redistributed the splitter panels')


class ChromeTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_check_for_updates_is_the_last_thing_on_the_toolbar(self):
        toolbar = self.window.findChildren(QToolBar)[0]
        widgets = [toolbar.widgetForAction(a) for a in toolbar.actions()]
        widgets = [x for x in widgets if x is not None]
        self.assertIs(widgets[-1], self.window.update_button,
                      'the update button must be the trailing item')
        spacers = [x for x in widgets if x.objectName() == 'toolbarSpacer']
        self.assertTrue(spacers, 'right-justification needs an expanding spacer before the button')
        self.assertGreater(widgets.index(self.window.update_button), widgets.index(spacers[0]))
        # Right-justified in practice, not only in widget order.
        button = self.window.update_button
        self.assertGreater(button.mapTo(toolbar, button.rect().center()).x(), toolbar.width() // 2)

    def test_the_theme_choice_restyles_the_application_and_persists(self):
        w = self.window
        original = w.theme_name
        other = next(name for name in THEMES if name != original)
        w.apply_theme_name(other)
        self.assertEqual(w.theme_name, other)
        self.assertEqual(APP.styleSheet(), build_style(other))
        w.preferences.set_theme(other)
        self.assertEqual(Preferences().theme(), other)
        # The node-family colours are identity, not theme: they must not move with it.
        self.assertEqual(COLORS['Grade'], '#83cbb7')
        w.apply_theme_name(original)
        w.preferences.set_theme(original)

    def test_an_accent_colour_restyles_over_any_theme_and_persists(self):
        w = self.window
        original_theme, original_accent = w.theme_name, w.accent_color
        w.apply_theme_name(original_theme, '#E592C0')
        self.assertEqual(w.accent_color, '#e592c0')
        self.assertIn('#e592c0', APP.styleSheet())
        self.assertEqual(APP.styleSheet(), build_style(original_theme, '#e592c0'))
        w.preferences.set_accent('#e592c0')
        self.assertEqual(Preferences().accent(), '#e592c0')
        dialog = ProjectSettingsDialog(copy.deepcopy(w.dispatcher.document['settings']), w,
                                       theme=w.theme_name, accent=w.accent_color)
        self.assertEqual(dialog.chosen_accent(), '#e592c0')
        self.assertEqual(set(dialog.changes()), {'color', 'viewer'})
        dialog.accent.setCurrentIndex(dialog.accent.findText('Theme default'))
        self.assertIsNone(dialog.chosen_accent())
        dialog.deleteLater()
        # A garbage preference is ignored rather than breaking the stylesheet.
        w.apply_theme_name(original_theme, 'not-a-colour')
        self.assertIsNone(w.accent_color)
        w.preferences.set_accent(original_accent)
        w.apply_theme_name(original_theme, original_accent)

    def test_an_unknown_stored_theme_falls_back_instead_of_failing(self):
        self.assertEqual(self.window.apply_theme_name('Chartreuse') or self.window.theme_name,
                         DEFAULT_THEME)

    def test_the_settings_dialog_offers_the_theme_without_putting_it_in_the_document(self):
        w = self.window
        dialog = ProjectSettingsDialog(copy.deepcopy(w.dispatcher.document['settings']), w,
                                       theme=w.theme_name)
        self.assertEqual(dialog.theme.currentText(), w.theme_name)
        self.assertEqual(set(dialog.changes()), {'color', 'viewer'},
                         'a machine preference must not travel inside the comp')
        dialog.deleteLater()


class ViewerNodeGraphTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_a_viewer_noodle_is_faint_dashed_and_arrowless(self):
        w = self.window
        viewer_key = next(k for k, n in w.dispatcher.document['nodes'].items()
                          if n['type'] == 'Viewer')
        target = next(k for k, n in w.dispatcher.document['nodes'].items() if n['type'] == 'Grade')
        w.command({'op': 'view', 'id': target})
        viewer_edges = [edge for edge, source, key, slot in w.graph.edges if key == viewer_key]
        other_edges = [edge for edge, source, key, slot in w.graph.edges if key != viewer_key]
        self.assertTrue(viewer_edges, 'viewing a node must draw the Viewer connection')
        self.assertTrue(other_edges, 'the comp itself must still have ordinary noodles to compare')
        for edge in viewer_edges:
            self.assertEqual(edge.pen().style(), Qt.PenStyle.DashLine)
            self.assertFalse(edge.arrow, 'a view tap must not claim a processing direction')
            self.assertLess(edge.pen().widthF(), other_edges[0].pen().widthF())
            self.assertLess(edge.zValue(), other_edges[0].zValue())

    def test_viewing_a_node_moves_the_viewer_node_input_in_the_graph(self):
        w = self.window
        viewer_key = next(k for k, n in w.dispatcher.document['nodes'].items()
                          if n['type'] == 'Viewer')
        target = next(k for k, n in w.dispatcher.document['nodes'].items() if n['type'] == 'Constant')
        w.command({'op': 'view', 'id': target})
        self.assertEqual(w.dispatcher.document['nodes'][viewer_key]['inputs']['image'], target)
        self.assertIn((viewer_key, 'image'),
                      [(key, slot) for _, source, key, slot in w.graph.edges if source == target])


class WriteRenderTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        w = self.window
        source = next(k for k, n in w.dispatcher.document['nodes'].items() if n['type'] == 'Constant')
        w.command({'op': 'batch', 'commands': [
            {'op': 'set', 'id': source, 'param': 'width', 'value': 16},
            {'op': 'set', 'id': source, 'param': 'height', 'value': 16},
            {'op': 'create', 'id': 'writer', 'type': 'Write'},
            {'op': 'connect', 'id': 'writer', 'input': 'image', 'source': source}]})

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_a_write_node_renders_a_padded_frame_range(self):
        w = self.window
        with tempfile.TemporaryDirectory() as temp:
            pattern = str(Path(temp) / 'render.%04d.exr')
            w.command({'op': 'set', 'id': 'writer', 'param': 'path', 'value': pattern})
            w.set_time(first=1, last=3, current=1)
            w.render_write('writer', single=False)
            written = sorted(p.name for p in Path(temp).iterdir())
            self.assertEqual(written, ['render.0001.exr', 'render.0002.exr', 'render.0003.exr'])

    def test_a_write_node_renders_one_frame_to_a_still_path(self):
        w = self.window
        with tempfile.TemporaryDirectory() as temp:
            target = str(Path(temp) / 'single.png')
            w.command({'op': 'set', 'id': 'writer', 'param': 'path', 'value': target})
            w.render_write('writer', single=True)
            self.assertTrue(Path(target).is_file())

    def test_a_range_render_refuses_an_unpadded_path_instead_of_overwriting(self):
        w = self.window
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'flat.exr'
            w.command({'op': 'set', 'id': 'writer', 'param': 'path', 'value': str(target)})
            w.set_time(first=1, last=3, current=1)
            with unittest.mock.patch.object(QMessageBox, 'warning') as warning:
                w.render_write('writer', single=False)
            self.assertTrue(warning.called, 'a 3-frame range into one file must be refused')
            self.assertFalse(target.exists())

    def test_an_empty_write_path_is_refused_with_a_reason(self):
        with unittest.mock.patch.object(QMessageBox, 'warning') as warning:
            self.window.render_write('writer', single=True)
        self.assertTrue(warning.called)

    def test_an_explicit_file_type_wins_over_a_disagreeing_extension(self):
        w = self.window
        w.command({'op': 'batch', 'commands': [
            {'op': 'set', 'id': 'writer', 'param': 'path', 'value': '/tmp/out.exr'},
            {'op': 'set', 'id': 'writer', 'param': 'file_type', 'value': 'png'}]})
        path, file_type, bits = w.write_target('writer')
        self.assertEqual((Path(path).suffix, file_type), ('.png', 'png'))
