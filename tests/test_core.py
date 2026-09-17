import copy
import tempfile
import unittest
from pathlib import Path

from nodebased.core import (SCHEMA_VERSION, Dispatcher, atomic_save, load_document, demo_document,
                            empty_document, validate)


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
        old.pop('node_data')
        old.pop('expressions')
        old['version'] = 6
        upgraded = Dispatcher(old).document
        validate(upgraded)
        self.assertEqual(upgraded['version'], SCHEMA_VERSION)
        self.assertEqual(upgraded['settings']['color']['working_space'], 'ACEScg')
        self.assertEqual(upgraded['settings']['color']['view'], 'sRGB')


class ViewerNodeTests(unittest.TestCase):
    """The `view` op is Nuke's "press 1": the Viewer node's input follows what is being viewed,
    so the graph never shows a Viewer wired somewhere the artist is not actually looking."""

    def build(self):
        d = Dispatcher()
        d.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'c', 'type': 'Constant'},
            {'op': 'create', 'id': 'g', 'type': 'Grade'},
            {'op': 'create', 'id': 'v', 'type': 'Viewer'},
            {'op': 'connect', 'id': 'g', 'input': 'image', 'source': 'c'},
        ]})
        return d

    def test_viewing_a_node_rewires_every_viewer_to_it(self):
        d = self.build()
        d.execute({'op': 'view', 'id': 'g'})
        self.assertEqual(d.document['nodes']['v']['inputs']['image'], 'g')
        d.execute({'op': 'view', 'id': 'c'})
        self.assertEqual(d.document['nodes']['v']['inputs']['image'], 'c')

    def test_viewing_nothing_disconnects_the_viewer(self):
        d = self.build()
        d.execute({'op': 'view', 'id': 'g'})
        d.execute({'op': 'view', 'id': None})
        self.assertIsNone(d.document['nodes']['v']['inputs']['image'])
        self.assertIsNone(d.document['view'])

    def test_a_viewer_is_never_rewired_into_a_cycle(self):
        # A Grade fed *from* the Viewer means viewing that Grade would close a loop. The view
        # target still changes; the connection is the part that is refused.
        d = self.build()
        d.execute({'op': 'create', 'id': 'downstream', 'type': 'Grade'})
        d.execute({'op': 'connect', 'id': 'downstream', 'input': 'image', 'source': 'v'})
        d.execute({'op': 'view', 'id': 'downstream'})
        self.assertEqual(d.document['view'], 'downstream')
        self.assertNotEqual(d.document['nodes']['v']['inputs']['image'], 'downstream')
        validate(d.document)

    def test_a_viewer_never_views_itself(self):
        d = self.build()
        d.execute({'op': 'view', 'id': 'v'})
        self.assertIsNone(d.document['nodes']['v']['inputs']['image'])
        validate(d.document)

    def test_the_rewire_is_one_undo_step_with_the_view_change(self):
        d = self.build()
        d.execute({'op': 'view', 'id': 'g'})
        d.execute({'op': 'undo'})
        self.assertIsNone(d.document['view'])
        self.assertIsNone(d.document['nodes']['v']['inputs']['image'])


class WriteNodeTests(unittest.TestCase):
    """Write states a destination; it never alters the pixels flowing through it."""

    def test_write_passes_its_input_through_unchanged(self):
        import numpy as np
        from nodebased.imaging import Evaluator
        d = Dispatcher()
        d.execute({'op': 'batch', 'commands': [
            {'op': 'create', 'id': 'c', 'type': 'Constant',
             'params': {'width': 4, 'height': 4, 'red': 0.25, 'green': 0.5, 'blue': 0.75}},
            {'op': 'create', 'id': 'w', 'type': 'Write'},
            {'op': 'connect', 'id': 'w', 'input': 'image', 'source': 'c'},
        ]})
        evaluator = Evaluator()
        np.testing.assert_array_equal(evaluator.evaluate(d.document, 'w'),
                                      evaluator.evaluate(d.document, 'c'))

    def test_write_can_be_bypassed_like_any_other_processing_node(self):
        d = Dispatcher()
        d.execute({'op': 'create', 'id': 'w', 'type': 'Write'})
        d.execute({'op': 'disable', 'id': 'w', 'value': True})
        self.assertTrue(d.document['nodes']['w']['disabled'])

    def test_write_params_are_validated_against_the_format_choices(self):
        d = Dispatcher()
        d.execute({'op': 'create', 'id': 'w', 'type': 'Write'})
        with self.assertRaises(ValueError):
            d.execute({'op': 'set', 'id': 'w', 'param': 'file_type', 'value': 'tga'})
        d.execute({'op': 'set', 'id': 'w', 'param': 'file_type', 'value': 'png'})
        self.assertEqual(d.document['nodes']['w']['params']['file_type'], 'png')


class NodeTabTests(unittest.TestCase):
    def test_label_and_thumbnail_store_only_overrides(self):
        d = Dispatcher(demo_document())
        d.execute({"op": "label", "id": "grade", "value": "hero"})
        d.execute({"op": "thumbnail", "id": "grade", "value": True})
        d.execute({"op": "thumbnail", "id": "plate", "value": True})
        nodes = d.document["nodes"]
        self.assertEqual((nodes["grade"]["label"], nodes["grade"]["thumbnail"]), ("hero", True))
        self.assertNotIn("thumbnail", nodes["plate"], "a default value is not stored")
        d.execute({"op": "label", "id": "grade", "value": ""})
        self.assertNotIn("label", d.document["nodes"]["grade"])
        with self.assertRaises(ValueError):
            d.execute({"op": "thumbnail", "id": "grade", "value": "yes"})
        d.execute({"op": "undo"})
        self.assertEqual(d.document["nodes"]["grade"]["label"], "hero")

    def test_a_v11_document_upgrades_unchanged(self):
        from nodebased.core import upgrade_document, validate
        old = copy.deepcopy(demo_document())
        old["version"] = 11
        upgraded = upgrade_document(copy.deepcopy(old))
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["nodes"], old["nodes"])
        validate(upgraded)
