"""The Node tab every node shares: Nuke's Node-tab knobs, what they store and what they change."""
import copy
import os
import tempfile
import unittest
import unittest.mock

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QObject
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QPushButton, QSpinBox, QTabWidget, QToolButton)

from nodebased.app import Window, STYLE
from nodebased.core import (Dispatcher, NODE_TAB_DEFAULTS, SPECS, apply_lifetime, load_document, node_tab_pixels,
                            validate)
from nodebased.imaging import Evaluator
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle('Fusion')
APP.setStyleSheet(STYLE)

# Nuke's order, and the object names the panel gives them.
EXPECTED_CONTROLS = ['node-label', 'node-font', 'node-font-size', 'node-bold', 'node-italic', 'node-font-color',
                     'node-font-color-clear', 'node-tile-color', 'node-tile-color-clear', 'node-thumbnail',
                     'node-hide-input', 'node-cached', 'node-disable', 'node-dope-sheet', 'node-bookmark',
                     'node-use-lifetime', 'node-lifetime-start', 'node-lifetime-end']


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher()
        self.d.execute({'op': 'create', 'id': 'g', 'type': 'Grade', 'pos': [0, 0]})

    def tab(self):
        return self.d.document['nodes']['g'].get('tab')

    def test_untouched_node_has_no_tab_key(self):
        self.assertNotIn('tab', self.d.document['nodes']['g'])

    def test_a_knob_sets_stores_and_clears(self):
        self.d.execute({'op': 'node_tab', 'id': 'g', 'name': 'bold', 'value': True})
        self.d.execute({'op': 'node_tab', 'id': 'g', 'name': 'tile_color', 'value': '#AA3300'})
        self.assertEqual(self.tab(), {'bold': True, 'tile_color': '#aa3300'})
        self.d.execute({'op': 'node_tab', 'id': 'g', 'name': 'bold', 'value': False})
        self.d.execute({'op': 'node_tab', 'id': 'g', 'name': 'tile_color', 'value': ''})
        self.assertNotIn('tab', self.d.document['nodes']['g'])

    def test_undo_and_redo_restore_the_knob(self):
        self.d.execute({'op': 'node_tab', 'id': 'g', 'name': 'font_size', 'value': 20})
        self.d.execute({'op': 'undo'})
        self.assertNotIn('tab', self.d.document['nodes']['g'])
        self.d.execute({'op': 'redo'})
        self.assertEqual(self.tab(), {'font_size': 20})

    def test_bad_values_are_refused(self):
        for name, value in (('font', 'Comic'), ('font_size', 2), ('font_size', 1.5), ('bold', 1),
                            ('tile_color', 'red'), ('lifetime_start', True), ('nope', 1)):
            with self.assertRaises(ValueError, msg=name):
                self.d.execute({'op': 'node_tab', 'id': 'g', 'name': name, 'value': value})

    def test_the_knobs_survive_save_and_load(self):
        for name, value in (('cached', True), ('use_lifetime', True), ('lifetime_end', 40), ('bookmark', True)):
            self.d.execute({'op': 'node_tab', 'id': 'g', 'name': name, 'value': value})
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'comp.json')
            self.d.execute({'op': 'save', 'path': path})
            reloaded = load_document(path)
        self.assertEqual(reloaded['nodes']['g']['tab'],
                         {'cached': True, 'use_lifetime': True, 'lifetime_end': 40, 'bookmark': True})

    def test_an_older_document_loads_with_todays_look(self):
        document = copy.deepcopy(self.d.document)
        validate(document)
        self.assertNotIn('tab', document['nodes']['g'])

    def test_a_stored_default_is_refused(self):
        document = copy.deepcopy(self.d.document)
        document['nodes']['g']['tab'] = {'bold': False}
        with self.assertRaises(ValueError):
            validate(document)

    def test_a_node_inside_a_group_takes_the_knob(self):
        self.d.execute({'op': 'create', 'id': 'h', 'type': 'Group', 'pos': [0, 0]})
        inner = {'op': 'create', 'id': 'i', 'type': 'Grade', 'pos': [0, 0], 'path': ['h']}
        self.d.execute(inner)
        self.d.execute({'op': 'node_tab', 'id': 'i', 'name': 'cached', 'value': True, 'path': ['h']})
        self.assertEqual(self.d.document['nodes']['h']['graph']['nodes']['i']['tab'], {'cached': True})


class LifetimeTests(unittest.TestCase):
    """Lifetime bypasses a node outside [start, end], on the evaluator and the tile path."""

    def build(self, **tab):
        d = Dispatcher()
        d.execute({'op': 'create', 'id': 'c', 'type': 'Constant', 'pos': [0, 0],
                   'params': {'red': 0.2, 'green': 0.2, 'blue': 0.2}})
        d.execute({'op': 'create', 'id': 'g', 'type': 'Grade', 'pos': [0, 0], 'params': {'exposure': 2.0}})
        d.execute({'op': 'connect', 'id': 'g', 'input': SPECS['Grade']['inputs'][0], 'source': 'c'})
        for name, value in tab.items():
            d.execute({'op': 'node_tab', 'id': 'g', 'name': name, 'value': value})
        return d.document

    def pixel(self, document, frame):
        raster = Evaluator().evaluate(dict(document, view='g'), frame=frame)
        return float(np.asarray(raster.pixels if hasattr(raster, 'pixels') else raster)[0, 0, 0])

    def test_inside_the_lifetime_the_grade_works_and_outside_it_passes_through(self):
        document = self.build(use_lifetime=True, lifetime_start=10, lifetime_end=20)
        inside, outside = self.pixel(document, 15), self.pixel(document, 30)
        self.assertAlmostEqual(outside, 0.2, places=3)
        self.assertGreater(inside, outside * 2)

    def test_lifetime_off_ignores_the_range(self):
        document = self.build(lifetime_start=10, lifetime_end=20)
        self.assertEqual(apply_lifetime(document, 30), document)
        self.assertGreater(self.pixel(document, 30), 0.3)

    def test_the_document_is_not_changed_by_the_bypass(self):
        document = self.build(use_lifetime=True, lifetime_start=10, lifetime_end=20)
        before = copy.deepcopy(document)
        self.pixel(document, 30)
        self.assertEqual(document, before)
        self.assertTrue(apply_lifetime(document, 30)['nodes']['g']['disabled'])
        self.assertFalse(apply_lifetime(document, 15)['nodes']['g']['disabled'])

    def test_a_source_has_nothing_to_pass_through_and_is_left_alone(self):
        document = self.build()
        document['nodes']['c']['tab'] = {'use_lifetime': True, 'lifetime_start': 1, 'lifetime_end': 2}
        self.assertFalse(apply_lifetime(document, 50)['nodes']['c']['disabled'])

    def test_a_lifetime_change_is_a_pixel_change_but_a_colour_is_not(self):
        self.assertEqual(node_tab_pixels({'tab': {'tile_color': '#112233', 'bold': True}}), {})
        self.assertEqual(node_tab_pixels({'tab': {'cached': True, 'bookmark': True}}), {'cached': True})


class TilePathAndPlaybackTests(unittest.TestCase):
    def test_the_tile_path_bypasses_outside_the_lifetime(self):
        from nodebased.tileexec import TileExecutor
        document = LifetimeTests().build(use_lifetime=True, lifetime_start=10, lifetime_end=20)
        document = dict(document, view='g')
        executor = TileExecutor(evaluator=Evaluator())
        values = {}
        for frame in (15, 30):
            region = executor.canvas_region(document, 'g', frame=frame, tier=1)
            result = executor.compose_region(document, 'g', region, frame=frame, tier=1)
            values[frame] = float(np.asarray(result.pixels)[0, 0, 0])
        self.assertAlmostEqual(values[30], 0.2, places=3)
        self.assertGreater(values[15], 0.3)

    def test_playback_identity_ignores_colours_but_not_the_lifetime(self):
        from nodebased.playback import DisplayCache
        document = LifetimeTests().build()
        base = DisplayCache.identity(document, 'g', 1)
        styled = copy.deepcopy(document)
        styled['nodes']['g']['tab'] = {'tile_color': '#112233', 'bold': True, 'bookmark': True}
        self.assertEqual(DisplayCache.identity(styled, 'g', 1), base)
        timed = copy.deepcopy(document)
        timed['nodes']['g']['tab'] = {'use_lifetime': True}
        self.assertNotEqual(DisplayCache.identity(timed, 'g', 1), base)


class CachedTests(unittest.TestCase):
    def test_the_flag_persists_and_reaches_the_evaluator(self):
        d = Dispatcher()
        d.execute({'op': 'create', 'id': 'c', 'type': 'Constant', 'pos': [0, 0]})
        d.execute({'op': 'create', 'id': 'g', 'type': 'Grade', 'pos': [0, 0]})
        d.execute({'op': 'connect', 'id': 'g', 'input': SPECS['Grade']['inputs'][0], 'source': 'c'})
        d.execute({'op': 'node_tab', 'id': 'g', 'name': 'cached', 'value': True})
        evaluator = Evaluator()
        evaluator.evaluate(dict(d.document, view='g'))
        self.assertEqual(list(evaluator._pinned), ['g'])
        pinned = evaluator._pinned['g']
        self.assertIn(pinned, evaluator.cache)

    def test_a_pinned_result_outlives_eviction(self):
        d = Dispatcher()
        d.execute({'op': 'create', 'id': 'c', 'type': 'Constant', 'pos': [0, 0]})
        d.execute({'op': 'create', 'id': 'g', 'type': 'Grade', 'pos': [0, 0]})
        d.execute({'op': 'connect', 'id': 'g', 'input': SPECS['Grade']['inputs'][0], 'source': 'c'})
        d.execute({'op': 'node_tab', 'id': 'g', 'name': 'cached', 'value': True})
        evaluator = Evaluator()
        evaluator.evaluate(dict(d.document, view='g'))
        pinned = evaluator._pinned['g']
        evaluator._fixed_budget = 1   # everything else must go; the pinned entry stays
        from nodebased.raster import Raster
        for index in range(3):
            evaluator._store(f'other-{index}', Raster(np.zeros((8, 8, 4), np.float32)))
        self.assertIn(pinned, evaluator.cache)

    def test_turning_it_off_unpins(self):
        d = Dispatcher()
        d.execute({'op': 'create', 'id': 'c', 'type': 'Constant', 'pos': [0, 0]})
        d.execute({'op': 'create', 'id': 'g', 'type': 'Grade', 'pos': [0, 0]})
        d.execute({'op': 'connect', 'id': 'g', 'input': SPECS['Grade']['inputs'][0], 'source': 'c'})
        d.execute({'op': 'node_tab', 'id': 'g', 'name': 'cached', 'value': True})
        evaluator = Evaluator()
        evaluator.evaluate(dict(d.document, view='g'))
        d.execute({'op': 'node_tab', 'id': 'g', 'name': 'cached', 'value': False})
        evaluator.evaluate(dict(d.document, view='g'))
        self.assertEqual(evaluator._pinned, {})


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()

    def node(self):
        return self.window.dispatcher.document['nodes']['grade']

    def open_panel(self):
        w = self.window
        w.graph.items_by_id['grade'].setSelected(True)
        w.inspect('grade')
        tabs = w.properties.widget().findChild(QTabWidget, 'node-tabs')
        tabs.setCurrentIndex(1)
        return w.properties.widget()

    def test_every_control_sets_its_field(self):
        panel = self.open_panel()
        panel.findChild(QCheckBox, 'node-bold').setChecked(True)
        self.assertTrue(wait_until(lambda: self.node().get('tab', {}).get('bold') is True))
        panel = self.open_panel()
        panel.findChild(QCheckBox, 'node-italic').setChecked(True)
        self.assertTrue(wait_until(lambda: self.node()['tab'].get('italic') is True))
        panel = self.open_panel()
        font = panel.findChild(QComboBox, 'node-font')
        font.setCurrentText('Serif')
        font.activated.emit(font.currentIndex())
        self.assertTrue(wait_until(lambda: self.node()['tab'].get('font') == 'Serif'))
        panel = self.open_panel()
        size = panel.findChild(QSpinBox, 'node-font-size')
        size.setValue(18)
        size.editingFinished.emit()
        self.assertTrue(wait_until(lambda: self.node()['tab'].get('font_size') == 18))
        for name, field in (('node-hide-input', 'hide_input'), ('node-cached', 'cached'),
                            ('node-dope-sheet', 'dope_sheet'), ('node-bookmark', 'bookmark'),
                            ('node-use-lifetime', 'use_lifetime')):
            panel = self.open_panel()
            panel.findChild(QCheckBox, name).setChecked(True)
            self.assertTrue(wait_until(lambda f=field: self.node()['tab'].get(f) is True), field)
        panel = self.open_panel()
        end = panel.findChild(QSpinBox, 'node-lifetime-end')
        self.assertTrue(end.isEnabled())
        end.setValue(60)
        end.editingFinished.emit()
        self.assertTrue(wait_until(lambda: self.node()['tab'].get('lifetime_end') == 60))

    def test_disable_is_nukes_sense_and_the_stored_field_is_unchanged(self):
        panel = self.open_panel()
        box = panel.findChild(QCheckBox, 'node-disable')
        self.assertFalse(box.isChecked())
        box.setChecked(True)
        self.assertTrue(wait_until(lambda: self.node()['disabled'] is True))
        self.assertTrue(self.open_panel().findChild(QCheckBox, 'node-disable').isChecked())

    def test_the_colours_come_from_the_dialog_and_clear(self):
        panel = self.open_panel()
        with unittest.mock.patch('nodebased.app.QColorDialog.getColor', return_value=QColor('#33aa55')):
            panel.findChild(QPushButton, 'node-tile-color').click()
        self.assertTrue(wait_until(lambda: self.node().get('tab', {}).get('tile_color') == '#33aa55'))
        panel = self.open_panel()
        with unittest.mock.patch('nodebased.app.QColorDialog.getColor', return_value=QColor('#ff0000')):
            panel.findChild(QPushButton, 'node-font-color').click()
        self.assertTrue(wait_until(lambda: self.node()['tab'].get('font_color') == '#ff0000'))
        panel = self.open_panel()
        panel.findChild(QToolButton, 'node-tile-color-clear').click()
        self.assertTrue(wait_until(lambda: 'tile_color' not in self.node()['tab']))

    def test_the_graph_item_reflects_tile_colour_and_hidden_input(self):
        w = self.window
        before = QColor(w.graph.items_by_id['grade'].accent)
        w.command({'op': 'node_tab', 'id': 'grade', 'name': 'tile_color', 'value': '#12ab34'})
        self.assertTrue(wait_until(lambda: w.graph.items_by_id['grade'].accent.name() == '#12ab34'))
        self.assertNotEqual(before.name(), '#12ab34')
        w.command({'op': 'node_tab', 'id': 'grade', 'name': 'tile_color', 'value': ''})
        self.assertTrue(wait_until(lambda: w.graph.items_by_id['grade'].accent.name() == before.name()))

        def wires():
            return len([item for item in w.graph.scene().items() if type(item).__name__ == 'Edge'])
        wires_before = wires()
        connections = sum(1 for source in w.graph_nodes()['grade']['inputs'].values() if source)
        w.command({'op': 'node_tab', 'id': 'grade', 'name': 'hide_input', 'value': True})
        self.assertTrue(wait_until(lambda: wires() == wires_before - connections))
        # The connection itself stays.
        self.assertEqual(sum(1 for source in w.graph_nodes()['grade']['inputs'].values() if source), connections)

    def test_the_label_styling_reaches_the_graph_item(self):
        w = self.window
        w.command({'op': 'label', 'id': 'grade', 'value': 'key light'})
        w.command({'op': 'node_tab', 'id': 'grade', 'name': 'bold', 'value': True})
        w.command({'op': 'node_tab', 'id': 'grade', 'name': 'font_size', 'value': 17})
        w.command({'op': 'node_tab', 'id': 'grade', 'name': 'font_color', 'value': '#ff8800'})

        def styled():
            for child in w.graph.items_by_id['grade'].childItems():
                if hasattr(child, 'text') and 'key light' in child.text():
                    return child.font().bold() and child.font().pixelSize() == 17 \
                        and child.brush().color().name() == '#ff8800'
            return False
        self.assertTrue(wait_until(styled))

    def test_every_node_type_has_the_same_node_tab(self):
        w = self.window
        reference = None
        drift = []
        for index, kind in enumerate(sorted(SPECS)):
            key = f'tab_scan_{index}'
            w.dispatcher.execute({'op': 'create', 'id': key, 'type': kind, 'pos': [0, 0]})
            page = w.node_tab(key, w.graph_nodes()[key])
            names = [child.objectName() for child in page.findChildren(QObject)
                     if child.objectName().startswith('node-')]
            # A group keeps its note on the main tab, so it alone has no label here.
            comparable = [name for name in names if not (kind == 'Group' and name == 'node-label')]
            if reference is None and kind != 'Group':
                reference = names
                self.assertEqual(reference, EXPECTED_CONTROLS)
            elif kind != 'Group' and names != reference:
                drift.append(kind)
            elif kind == 'Group' and comparable != [n for n in EXPECTED_CONTROLS if n != 'node-label']:
                drift.append(kind)
        self.assertEqual(drift, [])


if __name__ == '__main__':
    unittest.main()
