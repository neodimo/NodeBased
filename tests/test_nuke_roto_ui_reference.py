import json
import unittest

from tools.audit_roto_ui import ROOT, SECTIONS, entries_for, score


def walk(value):
    if isinstance(value, dict):
        if 'source_page' in value:
            yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


class NukeRotoReferenceTests(unittest.TestCase):
    def test_reference_is_sourced_nonempty_and_tools_unique_per_node(self):
        ref = json.loads((ROOT / 'docs/parity/nuke_roto_ui.json').read_text())
        measured = json.loads((ROOT / 'docs/parity/nodebased_roto_ui.json').read_text())
        sources = set(ref['source_pages'])
        self.assertTrue(sources)
        for section in SECTIONS:
            self.assertTrue(ref[section])
            self.assertEqual(len(entries_for(section, ref)), len(entries_for(section, measured)))
            for item in walk(ref[section]):
                self.assertIn(item['source_page'], sources)
            for item in entries_for(section, measured):
                self.assertIn(item['status'], ('match', 'differs', 'missing', 'not_applicable'))
                self.assertTrue(item.get('location'))
        for node in ('Roto', 'RotoPaint'):
            names = [tool['name'] for group in ref['toolbar'] for tool in group['tools']
                     if tool['availability'] in (node, 'Roto and RotoPaint')]
            self.assertEqual(len(names), len(set(names)))

    def test_score_recomputes_and_changes_when_a_status_changes(self):
        ref = json.loads((ROOT / 'docs/parity/nuke_roto_ui.json').read_text())
        measured = json.loads((ROOT / 'docs/parity/nodebased_roto_ui.json').read_text())
        before = score(ref, measured)
        first = measured['toolbar'][0]['tools'][0]
        old = first['status']
        first['status'] = 'missing' if old != 'missing' else 'match'
        self.assertNotEqual(before, score(ref, measured))

    def test_not_applicable_rows_are_counted_but_never_scored(self):
        ref = json.loads((ROOT / 'docs/parity/nuke_roto_ui.json').read_text())
        measured = json.loads((ROOT / 'docs/parity/nodebased_roto_ui.json').read_text())
        rows = [e for section in SECTIONS for e in entries_for(section, measured) if e['status'] == 'not_applicable']
        self.assertEqual(sorted(e.get('label', e.get('name')) for e in rows), ['View', 'View'])
        for row in rows:
            self.assertIn('multi-view', row['difference'])
        before = score(ref, measured)
        counted = sum(before[s]['counts']['not_applicable'] for s in SECTIONS)
        self.assertEqual(counted, 2)
        for section in SECTIONS:
            self.assertEqual(before[section]['entries'], before[section]['scored_entries'] + before[section]['counts']['not_applicable'])
        for row in rows:                       # scoring one of them as a miss would lower the score; leaving it out does not
            row['status'] = 'missing'
        self.assertLess(score(ref, measured)['overall'], before['overall'])

