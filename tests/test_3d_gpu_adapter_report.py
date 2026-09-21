"""Names the wgpu adapter in the test log.

GPU parity numbers depend on the adapter, its backend and whether the rgba
target is float32 or half float; a CI log has to say which one produced them.
"""
import unittest
from nodebased import gpu3d


class AdapterReport(unittest.TestCase):
    def test_report_names_the_adapter(self):
        report = gpu3d.adapter_report()
        print('\n' + report, flush=True)
        if not gpu3d.available():
            self.assertIn('none', report)
            self.skipTest('No wgpu adapter')
        state = gpu3d._state()
        for field in ('type ', 'backend ', 'rgba target ' + state['format'], 'wgpu '):
            self.assertIn(field, report)
        self.assertEqual('MISSING' in report, state['format'] == 'rgba16float')


if __name__ == '__main__':
    unittest.main()
