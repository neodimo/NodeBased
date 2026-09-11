import copy
import tempfile
import unittest
from pathlib import Path

from nodebased.core import Dispatcher, atomic_save, load_document, demo_document, empty_document, validate


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.d = Dispatcher(demo_document())

    def test_cycle_rejection_is_atomic(self):
        before = copy.deepcopy(self.d.document)
        with self.assertRaisesRegex(ValueError, 'cycles'):
            self.d.execute({'op': 'connect', 'id': 'grade', 'input': 'image', 'source': 'viewer'})
        self.assertEqual(before, self.d.document)
        self.assertEqual(self.d.undo_stack, [])

    def test_batch_rollback(self):
        before = copy.deepcopy(self.d.document)
        with self.assertRaises(ValueError):
            self.d.execute({'op': 'batch', 'commands': [
                {'op': 'set', 'id': 'grade', 'param': 'exposure', 'value': 2},
                {'op': 'set', 'id': 'wash', 'param': 'alpha', 'value': 8}]})
        self.assertEqual(before, self.d.document)

    def test_undo_redo_deletion_restores_connections(self):
        before = copy.deepcopy(self.d.document)
        self.d.execute({'op': 'delete', 'id': 'grade'})
        self.assertIsNone(self.d.document['nodes']['merge']['inputs']['B'])
        self.d.execute({'op': 'undo'})
        self.assertEqual(before, self.d.document)
        self.d.execute({'op': 'redo'})
        self.assertNotIn('grade', self.d.document['nodes'])

    def test_invalid_parameters_rejected(self):
        for param, value in [('exposure', float('nan')), ('exposure', True), ('bogus', 1)]:
            with self.assertRaises(ValueError):
                self.d.execute({'op': 'set', 'id': 'grade', 'param': param, 'value': value})

    def test_portable_project(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.d.execute({'op': 'create', 'id': 'read', 'type': 'Read', 'params': {'path': str(folder / 'plate.png')}})
            project = folder / 'comp.nbcomp'
            atomic_save(project, self.d.document)
            self.assertIn('"path": "plate.png"', project.read_text())
            self.assertEqual(load_document(project), self.d.document)

    def test_inspect_is_a_copy(self):
        result = self.d.execute({'op': 'inspect'})
        result['document']['nodes'].clear()
        self.assertTrue(self.d.document['nodes'])

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            self.d.execute({'op': 'connect', 'id': 'grade', 'input': 'image', 'source': 'missing'})

    def test_new_node_types_create_and_connect(self):
        for kind in ('Blur', 'ColorCorrect', 'Crop', 'Shuffle'):
            key = kind.lower()
            self.d.execute({'op': 'batch', 'commands': [
                {'op': 'create', 'id': key, 'type': kind},
                {'op': 'connect', 'id': key, 'input': 'image', 'source': 'plate'}]})
            self.assertEqual(self.d.document['nodes'][key]['inputs']['image'], 'plate')

    def test_shuffle_rejects_invalid_channel_choice(self):
        self.d.execute({'op': 'create', 'id': 'shuffle', 'type': 'Shuffle'})
        with self.assertRaises(ValueError):
            self.d.execute({'op': 'set', 'id': 'shuffle', 'param': 'red_from', 'value': 'Z'})

    def test_project_settings_are_validated_undoable_and_described(self):
        d = Dispatcher()
        self.assertEqual(d.document['settings']['color']['working_space'], 'ACEScg')
        self.assertEqual(d.document['settings']['color']['view'], 'ACES 2.0')
        d.execute({'op': 'settings', 'settings': {
            'color': {'view': 'Linear'}, 'viewer': {'background': 'checker'}}})
        self.assertEqual(d.document['settings']['color']['view'], 'Linear')
        self.assertEqual(d.document['settings']['viewer']['background'], 'checker')
        self.assertEqual(d.execute({'op': 'describe'})['settings'], d.document['settings'])
        d.execute({'op': 'undo'})
        self.assertEqual(d.document['settings']['color']['view'], 'ACES 2.0')
        with self.assertRaisesRegex(ValueError, 'Working space must be ACEScg'):
            d.execute({'op': 'settings', 'settings': {'color': {'working_space': 'Linear Rec.709'}}})

    def test_schema_six_upgrade_preserves_old_display_appearance(self):
        old = empty_document()
        old.pop('settings')
        old['version'] = 6
        upgraded = Dispatcher(old).document
        validate(upgraded)
        self.assertEqual(upgraded['version'], 7)
        self.assertEqual(upgraded['settings']['color']['working_space'], 'ACEScg')
        self.assertEqual(upgraded['settings']['color']['view'], 'sRGB')
