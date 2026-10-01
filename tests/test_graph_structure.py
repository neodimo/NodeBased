"""Node-graph structure items: Backdrop and PostageStamp (step S2 of the script-structure plan)."""
import os
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QPointF, QEvent, QSettings
from PySide6.QtTest import QTest

from nodebased.core import Dispatcher, OUTPUT_TYPES, SPECS
from nodebased.imaging import Evaluator
from tests.test_2d_parity_group_4c import Graph, evaluator_pixels, tile_pixels
from tests.test_desktop import APP, Window
from tests.waiting import wait_until, pause


class BackdropPostageStampTests(unittest.TestCase):
    """One window for the whole class; every test uses node ids of its own."""

    @classmethod
    def setUpClass(cls):
        # A saved workspace/state from a previous Window in this process restores verbatim
        # instead of running the default split, leaving the graph dock at whatever size that
        # earlier window happened to save, which moves on-screen drag positions.
        QSettings("NodeBased", "NodeBased").clear()
        cls.window = Window(agent_name='nodebased-test-' + uuid.uuid4().hex)
        cls.window.show()
        assert wait_until(lambda: cls.window.frame is not None)

    @classmethod
    def tearDownClass(cls):
        cls.window.saved_document = cls.window.dispatcher.document
        cls.window.close()
        APP.processEvents()
        cls.window.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        del cls.window

    def command(self, *commands):
        result = self.window.command({'op': 'batch', 'commands': list(commands)})
        self.assertIsNotNone(result, self.window.last_command_error)

    def doc(self):
        return self.window.dispatcher.document

    def press_drag_release(self, start_scene, end_scene):
        graph = self.window.graph
        graph.centerOn(start_scene)
        APP.processEvents()
        start, end = graph.mapFromScene(start_scene), graph.mapFromScene(end_scene)
        QTest.mousePress(graph.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(graph.viewport(), end, 20)
        QTest.mouseRelease(graph.viewport(), Qt.MouseButton.LeftButton, pos=end)
        wait_until(lambda: True)
        pause(50)
        APP.processEvents()

    def drag_rounding_delta(self):
        """One device pixel's worth of scene units at the graph's current zoom: `mapFromScene`
        rounds each endpoint of a drag to the nearest device pixel, so the delivered drag can be
        off by up to about one device pixel's worth of scene distance at either end. How many
        scene units that is depends on the graph view's current zoom, which depends on the
        window's dock split -- not a fixed pixel count."""
        zoom = max(abs(self.window.graph.transform().m11()), 0.01)
        return max(4.0, 1.0 / zoom)

    # -- Backdrop ----------------------------------------------------------------------------

    def test_a_backdrop_is_not_an_image_node(self):
        self.assertEqual(SPECS['Backdrop']['inputs'], [])
        self.assertNotIn('optional_inputs', SPECS['Backdrop'])
        self.assertEqual(OUTPUT_TYPES['Backdrop'], 'none')
        self.command({'op': 'create', 'id': 'bd_kind', 'type': 'Backdrop', 'pos': [3000, 3000]})
        with self.assertRaises(ValueError):
            self.window.dispatcher.execute({'op': 'view', 'id': 'bd_kind'})
        with self.assertRaises(ValueError):
            self.window.dispatcher.execute({'op': 'connect', 'id': 'viewer', 'input': 'image', 'source': 'bd_kind'})
        item = self.window.graph.items_by_id['bd_kind']
        self.assertEqual((item.inputs, item.output), ({}, None))
        self.assertLess(item.zValue(), 0)

    def test_dragging_a_backdrop_title_moves_the_nodes_inside_it_and_no_others(self):
        self.command(
            {'op': 'create', 'id': 'bd_move', 'type': 'Backdrop', 'name': 'Sky', 'pos': [1000, 1000],
             'params': {'width': 500, 'height': 400}},
            {'op': 'label', 'id': 'bd_move', 'value': 'Sky work'},
            {'op': 'create', 'id': 'in_a', 'type': 'Constant', 'pos': [1060, 1100]},
            {'op': 'create', 'id': 'in_b', 'type': 'Grade', 'pos': [1250, 1200]},
            {'op': 'create', 'id': 'out_c', 'type': 'Constant', 'pos': [1700, 1100]},
            {'op': 'create', 'id': 'out_d', 'type': 'Constant', 'pos': [1000, 1500]})
        before = {key: list(self.doc()['nodes'][key]['pos']) for key in ('in_a', 'in_b', 'out_c', 'out_d')}
        self.press_drag_release(QPointF(1100, 1010), QPointF(1180, 1090))
        nodes = self.doc()['nodes']
        # The drag is 80 scene units each way, give or take the view's pixel rounding.
        moved = [nodes['bd_move']['pos'][0] - 1000, nodes['bd_move']['pos'][1] - 1000]
        delta = self.drag_rounding_delta()
        self.assertAlmostEqual(moved[0], 80, delta=delta)
        self.assertAlmostEqual(moved[1], 80, delta=delta)
        for key in ('in_a', 'in_b'):
            self.assertEqual(nodes[key]['pos'], [before[key][0] + moved[0], before[key][1] + moved[1]], key)
        for key in ('out_c', 'out_d'):
            self.assertEqual(nodes[key]['pos'], before[key], key)
        # The properties panel stays on the backdrop rather than jumping to a contained node.
        self.assertEqual(self.window.graph.selected_id(), 'bd_move')

    def test_the_grip_resizes_a_backdrop_and_the_size_is_saved(self):
        self.command({'op': 'create', 'id': 'bd_size', 'type': 'Backdrop', 'pos': [2000, 2000],
                      'params': {'width': 300, 'height': 200}})
        self.press_drag_release(QPointF(2000 + 300 - 8, 2000 + 200 - 8), QPointF(2000 + 500 - 8, 2000 + 350 - 8))
        params = self.doc()['nodes']['bd_size']['params']
        delta = self.drag_rounding_delta()
        self.assertAlmostEqual(params['width'], 500, delta=delta)
        self.assertAlmostEqual(params['height'], 350, delta=delta)
        item = self.window.graph.items_by_id['bd_size']
        self.assertEqual((item.rect().width(), item.rect().height()), (params['width'], params['height']))

    def test_a_backdrop_survives_save_and_reload(self):
        self.command({'op': 'create', 'id': 'bd_save', 'type': 'Backdrop', 'name': 'Keep', 'pos': [500, 4000],
                      'params': {'width': 610, 'height': 240, 'red': 0.9, 'green': 0.1, 'blue': 0.2}},
                     {'op': 'label', 'id': 'bd_save', 'value': 'Keep me'})
        path = str(Path(tempfile.mkdtemp(prefix='nb-backdrop-')) / 'doc.json')
        self.window.dispatcher.execute({'op': 'save', 'path': path})
        expected = self.doc()['nodes']['bd_save']
        other = Dispatcher()
        other.execute({'op': 'load', 'path': path})
        self.assertEqual(other.document['nodes']['bd_save'], expected)
        self.assertEqual(expected['label'], 'Keep me')
        self.assertIsNotNone(self.window.command({'op': 'load', 'path': path}))
        item = self.window.graph.items_by_id['bd_save']
        self.assertEqual((item.rect().width(), item.rect().height()), (610, 240))
        self.assertEqual(item.caption, 'Keep me')

    def test_a_backdrop_made_with_nodes_selected_frames_them(self):
        self.command({'op': 'create', 'id': 'fr_a', 'type': 'Constant', 'pos': [6000, 6000]},
                     {'op': 'create', 'id': 'fr_b', 'type': 'Constant', 'pos': [6300, 6200]})
        graph = self.window.graph
        graph.scene().clearSelection()
        graph.items_by_id['fr_a'].setSelected(True)
        graph.items_by_id['fr_b'].setSelected(True)
        before = set(self.doc()['nodes'])
        self.window.add_node('Backdrop')
        made = (set(self.doc()['nodes']) - before).pop()
        item = graph.items_by_id[made]
        area = item.sceneBoundingRect()
        for key in ('fr_a', 'fr_b'):
            self.assertTrue(area.contains(graph.items_by_id[key].sceneBoundingRect()), key)
        self.assertEqual({key for key in ('fr_a', 'fr_b')},
                         {i.key for i in item.enclosed_items() if i.key in ('fr_a', 'fr_b')})

    # -- PostageStamp ------------------------------------------------------------------------

    def test_a_postage_stamp_passes_its_input_through_and_wears_a_thumbnail(self):
        self.command({'op': 'create', 'id': 'ps_src', 'type': 'Constant', 'pos': [-2000, 0],
                      'params': {'width': 64, 'height': 32}},
                     {'op': 'create', 'id': 'ps', 'type': 'PostageStamp', 'pos': [-2000, 200]},
                     {'op': 'connect', 'id': 'ps', 'input': 'image', 'source': 'ps_src'})
        doc = self.doc()
        self.assertTrue(np.array_equal(Evaluator().evaluate(doc, 'ps'), Evaluator().evaluate(doc, 'ps_src')))
        self.assertIsNotNone(self.window.graph.items_by_id['ps'].thumbnail)

    def test_the_stamp_thumbnail_follows_its_input_and_hide_input_hides_only_the_wire(self):
        w = self.window
        w.set_show_thumbnails(True)
        self.command({'op': 'create', 'id': 'pt_src', 'type': 'Constant', 'pos': [-2600, 0],
                      'params': {'width': 64, 'height': 32, 'red': 0.9, 'green': 0.1, 'blue': 0.1}},
                     {'op': 'create', 'id': 'pt', 'type': 'PostageStamp', 'pos': [-2600, 200]},
                     {'op': 'connect', 'id': 'pt', 'input': 'image', 'source': 'pt_src'})
        w.thumbnail_timer.start(0)
        self.assertTrue(wait_until(lambda: 'pt' in w.thumbnails), 'stamp thumbnail never arrived')
        first_key, first_image = w.thumbnails['pt']
        red = first_image.pixelColor(first_image.width() // 2, first_image.height() // 2)
        self.assertGreater(red.red(), red.blue())
        self.command({'op': 'set', 'id': 'pt_src', 'param': 'red', 'value': 0.1},
                     {'op': 'set', 'id': 'pt_src', 'param': 'blue', 'value': 0.9})
        self.assertTrue(wait_until(lambda: w.thumbnails['pt'][0] != first_key), 'stamp never refreshed')
        image = w.thumbnails['pt'][1]
        blue = image.pixelColor(image.width() // 2, image.height() // 2)
        self.assertGreater(blue.blue(), blue.red())
        # Hiding the input drops the noodle and keeps the connection.
        edges = len(w.graph.edges)
        self.command({'op': 'set', 'id': 'pt', 'param': 'hide_input', 'value': 1})
        self.assertEqual(len(w.graph.edges), edges - 1)
        self.assertEqual(self.doc()['nodes']['pt']['inputs']['image'], 'pt_src')
        self.command({'op': 'set', 'id': 'pt', 'param': 'hide_input', 'value': 0})
        self.assertEqual(len(w.graph.edges), edges)


class PostageStampKernelTests(unittest.TestCase):
    def test_output_equals_input_on_both_paths_bypassed_or_not(self):
        g = Graph()
        g.add('plate', 'Checker', dict(width=64, height=48, size=8))
        g.add('stamp', 'PostageStamp', dict(hide_input=1), image='plate')
        plate = evaluator_pixels(g.doc, 'plate')
        np.testing.assert_array_equal(evaluator_pixels(g.doc, 'stamp'), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, 'stamp'), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, 'stamp', tile_edge=16), plate)
        g.d.execute(dict(op='disable', id='stamp', value=True))
        np.testing.assert_array_equal(evaluator_pixels(g.doc, 'stamp'), plate)
        np.testing.assert_array_equal(tile_pixels(g.doc, 'stamp'), plate)


if __name__ == '__main__':
    unittest.main()
