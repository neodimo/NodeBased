import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_roto_ui import ROOT, SECTIONS, score

class NukeRotoReferenceTests(unittest.TestCase):
    def test_reference_and_measured_entries_are_sourced_and_nonempty(self):
        ref = json.loads((ROOT / 'docs/parity/nuke_roto_ui.json').read_text())
        measured = json.loads((ROOT / 'docs/parity/nodebased_roto_ui.json').read_text())
        self.assertTrue(ref['source_pages'])
        source = set(ref['source_pages'])
        for section in SECTIONS:
            self.assertTrue(ref[section])
            self.assertEqual(len(ref[section]), len(measured[section]))
            names = []
            for entry in ref[section]:
                self.assertIn(entry['source_page'], source)
                if section == 'toolbar':
                    for tool in entry['tools']:
                        self.assertIn(tool['source_page'], source)
                        names.append(tool['name'])
            if section == 'toolbar':
                self.assertEqual(len(names), len(set(names)))
                measured_entries = [tool for group in measured[section] for tool in group['tools']]
            else:
                measured_entries = measured[section]
            for entry in measured_entries:
                self.assertIn(entry['status'], ('match', 'differs', 'missing'))
                self.assertTrue(entry['location'])

    def test_score_is_recomputed_from_status(self):
        measured = json.loads((ROOT / 'docs/parity/nodebased_roto_ui.json').read_text())
        before = score({}, measured)
        first = measured['toolbar'][0]['tools'][0]
        old = first['status']
        first['status'] = 'missing' if old != 'missing' else 'match'
        after = score({}, measured)
        self.assertNotEqual(before, after)
