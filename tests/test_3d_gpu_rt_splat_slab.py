"""The splat shadow slab test must not index the packed node vec4s.

``_pack_casters`` stores each node as three ``vec4<f32>``: bounds in ``xyz`` and a
child link bitcast into ``w``. A leaf's link is ``-1``, whose bits are a NaN. When
the slab loop indexed ``low[a]`` and ``high[a]`` directly, D3D12 let that NaN reach
the axis comparison and rejected every leaf, so a light whose direction had an
exactly-zero component cast no splat shadow at all. Vulkan and the CPU tracer were
unaffected, which is why only the Windows runner ever saw it.

No adapter available here reproduces the miscompile, so this guards the shape of
the shader instead: the loop reads ``vec3`` copies, and ``.w`` is only ever read
whole, through ``bitcast``.
"""
import re
import unittest
from nodebased import gpurt_render


class SplatSlabTest(unittest.TestCase):
    def setUp(self):
        match = re.search(r'fn splat_visibility\(.*?\n}\n', gpurt_render._SHADER, re.S)
        self.assertIsNotNone(match, 'splat_visibility not found in the shader')
        self.body = match.group(0)

    def test_slab_loop_indexes_vec3_copies(self):
        self.assertIn('let lo=low.xyz; let hi=high.xyz;', self.body)
        for axis in ('lo[a]', 'hi[a]'):
            self.assertIn(axis, self.body)

    def test_packed_vec4s_are_never_indexed_dynamically(self):
        for name in ('low', 'high'):
            found = re.findall(rf'\b{name}\[', self.body)
            self.assertEqual(found, [], f'{name} is indexed dynamically: {found}')

    def test_child_links_are_read_whole(self):
        for name in ('low', 'high'):
            for use in re.findall(rf'\b{name}\.w\b', self.body):
                self.assertIn(f'bitcast<u32>({name}.w)', self.body)
            self.assertIn(f'bitcast<u32>({name}.w)', self.body)


if __name__ == '__main__':
    unittest.main()
