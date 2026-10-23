"""The path tracer's traversal stacks are sized to the trees, and a smoke vertex casts all its shadow rays through one call.

The splats-and-smoke shader of Render mode's settle took 4.5 s for one 8-sample pass at 1080p on the RTX 3080 Ti (3.7 million paths a
second; the same scene without the smoke 22, without the splats 31). Two things made it slow, both about how much code and local
memory one thread carries, and neither about the work it does: the three traversal stacks (32, 64 and 64 entries per thread) were
sized for a tree of any depth, and the smoke vertex's light sampling had three inlined copies of the shadow ray (the splat tree walk
and the smoke march), one for point and directional lights, one for area lights and one for the environment. Now each stack holds what
its tree's depth needs (rounded up to a few sizes so scenes share compiled shaders) and one call serves the lights and the
environment: 0.74 s for the same pass. Each test here fails without that."""
import re
import unittest

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, gpupathtrace as gp, pathtrace, raytrace, scene3d as s


def boxes(count, seed=0):
    rng = np.random.default_rng(seed)
    low = rng.uniform(-10, 10, (count, 3))
    return low, low + rng.uniform(.05, .5, (count, 3))


class StackSizes(unittest.TestCase):
    def test_a_balanced_tree_is_as_deep_as_its_primitive_count_says(self):
        for count in (1, 3, 4, 5, 9, 64, 1000, 20000):
            with self.subTest(count=count):
                depth = gp.tree_depth(raytrace.Bvh.build(*boxes(count)))
                leaves = -(-count // 4)
                self.assertEqual(depth, max(0, int(np.ceil(np.log2(leaves)))))

    def test_the_stack_a_tree_gets_always_holds_its_traversal(self):
        for count in (1, 5, 300, 5000, 40000):
            bvh = raytrace.Bvh.build(*boxes(count))
            depth = gp.tree_depth(bvh)
            size = gp.stack_sizes(depth, depth, depth)
            for entries in size:
                self.assertGreaterEqual(entries, depth + 2, 'a push writes one past the top: depth + 2 entries are the least')
                self.assertLessEqual(entries, 64)

    def test_sizes_come_in_a_few_buckets_and_never_past_the_old_lengths_unless_the_tree_is_deeper(self):
        self.assertEqual(gp.stack_sizes(0, 0, 0), (16, 16, 16))
        self.assertEqual(gp.stack_sizes(14, 14, 14), (16, 16, 16))
        self.assertEqual(gp.stack_sizes(15, 15, 15), (24, 24, 24))
        self.assertEqual(gp.stack_sizes(40, 40, 40)[1:], (48, 48))
        self.assertEqual(gp.stack_sizes(25, 25, 25)[0], 32)      # the top-level stack never grows past its old 32 for a tree that fits it
        self.assertEqual(gp.stack_sizes(40, 0, 0)[0], 42)        # and a deeper one gets the room it needs (it overflowed before)
        self.assertEqual(gp.stack_sizes(61, 61, 61)[1:], (64, 64))
        self.assertEqual(gp.stack_sizes(70, 70, 70)[1], 72)

    def test_the_shader_declares_the_sizes_it_is_given(self):
        code = gp.shader_source(splats=True, volumes=True, wide=True, stacks=(16, 24, 32))
        self.assertIn('var<private> tlas_stack: array<i32, 16>;', code)
        self.assertIn('var<private> blas_stack: array<i32, 24>;', code)
        self.assertIn('var<private> splat_stack: array<i32, 32>;', code)
        default = gp.shader_source(splats=True, volumes=True, wide=True)
        self.assertIn('array<i32, 32>;', default.split('var<private> tlas_stack')[1].split('\n')[0])
        self.assertIn('var<private> blas_stack: array<i32, 64>;', default)
        self.assertNotIn('STACK_SIZE', default)

    def test_a_packed_scene_carries_the_sizes_of_its_trees(self):
        from tests.test_3d_pathtrace_splats import instance, plane_cloud
        scene = s.Scene(geometries=(s._card(2, 2, (.5, .5, .5, 1), s.Transform3D(s.Vec3(0, 0, -1)), None),),
                        splats=(instance(plane_cloud(400)),))
        packed = gp.pack(pathtrace.build_scene(scene, ambient=0.05))
        self.assertEqual(len(packed.stacks), 3)
        self.assertTrue(all(16 <= v <= 64 for v in packed.stacks))


class ShadowRaysOfASmokeVertex(unittest.TestCase):
    def test_the_event_block_has_one_call_to_the_shadow_ray_not_one_per_kind_of_light(self):
        code = gp.shader_source(splats=True, volumes=True, wide=True)
        start = code.index('// a real collision in the smoke')
        end = code.index('let gathered = thr * params.vp1.xyz * direct;')
        self.assertEqual(len(re.findall(r'soft_visibility\(', code[start:end])), 1)

    def test_every_shader_variant_still_has_the_function_once(self):
        for splats, volumes in ((True, True), (True, False), (False, True)):
            code = gp.shader_source(splats=splats, volumes=volumes)
            self.assertEqual(len(re.findall(r'fn soft_visibility\(', code)), 1 if (splats or volumes) else 0)


if __name__ == '__main__':
    unittest.main()
