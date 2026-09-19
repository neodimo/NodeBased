"""CPU-only BVH structure, query and renderer parity tests."""
import threading
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import scene3d as s
from nodebased.imaging import Cancelled
from nodebased.raytrace import Bvh, TriangleSet, traverse


def soup(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.uniform(-10, 10, (n, 3)).astype('f4')
    return TriangleSet(v, rng.uniform(-.5, .5, (n, 3)).astype('f4'),
                       rng.uniform(-.5, .5, (n, 3)).astype('f4'),
                       rng.choice([0., .25, 1.], n))


def reference_hits(tri, origins, dirs, tmin, tmax):
    # Independent broadcast Moller-Trumbore, primitive order supplies tie rule.
    h = np.cross(dirs[:, None], tri.e2)
    det = np.einsum('rtj,tj->rt', h, tri.e1)
    valid = abs(det) > 1e-10
    inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
    delta = origins[:, None]-tri.v0
    u = np.einsum('rtj,rtj->rt', delta, h)*inv
    q = np.cross(delta, tri.e1)
    v = np.einsum('rj,rtj->rt', dirs, q)*inv
    t = np.einsum('tj,rtj->rt', tri.e2, q)*inv
    valid &= (u >= 0) & (v >= 0) & (u+v <= 1) & (t > np.broadcast_to(tmin, (len(origins),))[:, None]) & (t < np.broadcast_to(tmax, (len(origins),))[:, None])
    t = np.where(valid, t, np.inf)
    p = t.argmin(1)
    rows = np.arange(len(origins))
    best = t[rows, p]
    return best, np.where(np.isfinite(best), p, -1), u[rows,p], v[rows,p]


class RaytraceTests(unittest.TestCase):
    def test_structure_and_determinism(self):
        for n in (0, 1, 2, 37, 2000):
            for degenerate in (False, True):
                lo = np.random.default_rng(4).normal(size=(n, 3))
                hi = lo + .25
                if degenerate:
                    lo[:] = 2
                    hi[:] = 2
                a, b = Bvh.build(lo, hi), Bvh.build(lo, hi)
                for name in a.__dataclass_fields__:
                    np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                seen = []
                for i, count in enumerate(a.prim_count):
                    if count:
                        ids = a.prim_order[a.prim_offset[i]:a.prim_offset[i]+count]
                        seen.extend(ids)
                        self.assertTrue((a.node_lo[i] <= lo[ids]).all())
                        self.assertTrue((a.node_hi[i] >= hi[ids]).all())
                    else:
                        for child in (a.left[i], a.right[i]):
                            self.assertGreater(child, i)
                            self.assertTrue((a.node_lo[i] <= a.node_lo[child]).all())
                            self.assertTrue((a.node_hi[i] >= a.node_hi[child]).all())
                self.assertEqual(sorted(seen), list(range(n)))

    def test_queries_random_and_limits(self):
        for seed, n in enumerate((200, 777, 2000)):
            tri = soup(n, seed)
            bvh = Bvh.build(*tri.aabbs())
            rng = np.random.default_rng(seed+10)
            origins = rng.uniform(-10, 10, (100, 3)).astype('f4')
            dirs = rng.normal(size=(100, 3)).astype('f4')
            dirs[:12] = np.tile(np.eye(3), (4, 1))
            dirs[12] = 0
            # Guarantee many hits, with origins inside the overall scene bounds.
            origins[20:60] = tri.v0[:40]+.25*(tri.e1[:40]+tri.e2[:40])-dirs[20:60]*.1
            for tmin, tmax in ((0., np.inf), (.1, 4.), (-1., np.linspace(.01, 20, 100))):
                expected = reference_hits(tri, origins, dirs, tmin, tmax)
                actual = tri.closest_hit(bvh, origins, dirs, tmin, tmax)
                np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=0)
                np.testing.assert_array_equal(actual[1], expected[1])
                hit = expected[1] >= 0
                for a, e in zip(actual[2:], expected[2:]):
                    np.testing.assert_allclose(a[hit], e[hit], rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(tri.any_hit(bvh, origins, dirs, tmin, tmax), hit)
                np.testing.assert_allclose(tri.transmittance(bvh, origins, dirs, tmin, tmax),
                                           tri.brute_transmittance(origins, dirs, tmin, tmax), atol=1e-7)

    def test_edges_vertices_duplicates_and_chunking(self):
        # Binary-exact shared edge/vertex samples avoid changing MT's strict policy.
        v = np.array([[0,0,0], [0,0,0], [0,0,0]], dtype='f4')
        tri = TriangleSet(v, np.array([[1,0,0], [1,1,0], [1,0,0]], dtype='f4'),
                          np.array([[1,1,0], [0,1,0], [1,1,0]], dtype='f4'), [.25]*3)
        bvh = Bvh.build(*tri.aabbs(), leaf_size=1)
        q = np.linspace(0, 1, 257, dtype='f4')
        origins = np.column_stack((q,q,np.ones_like(q)))
        dirs = np.tile([0.,0.,-1.], (len(q),1)).astype('f4')
        base = tri.closest_hit(bvh, origins, dirs)
        np.testing.assert_array_equal(base[1], 0)  # exact ties choose lowest index
        for chunk in (1, 17, 1024):
            for actual, expected in zip(tri.closest_hit(bvh, origins, dirs, chunk=chunk), base):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(tri.transmittance(bvh, origins, dirs, chunk=chunk), .75**3)
        for lo, hi in ((1., np.inf), (0., 1.)):
            self.assertFalse(tri.any_hit(bvh, origins, dirs, lo, hi).any())

    def test_stats_and_empty(self):
        tri = soup(5000)
        bvh = Bvh.build(*tri.aabbs())
        origins = np.random.default_rng(8).uniform(-12,12,(300,3)).astype('f4')
        dirs = np.tile([0.,0.,1.], (300,1)).astype('f4')
        stats = {}
        transmission = tri.transmittance(bvh, origins, dirs, stats=stats)
        for chunk in (19, 512):
            np.testing.assert_array_equal(
                tri.transmittance(bvh, origins, dirs, chunk=chunk), transmission)
        self.assertGreater(stats['node_tests'], 0)
        self.assertLess(stats['primitive_tests'], .1*len(origins)*5000)
        empty = soup(0)
        bvh = Bvh.build(*empty.aabbs())
        self.assertTrue(np.isinf(empty.closest_hit(bvh, origins, dirs)[0]).all())
        np.testing.assert_array_equal(empty.closest_hit(bvh, origins, dirs)[1], -1)
        self.assertFalse(empty.any_hit(bvh, origins, dirs).any())
        np.testing.assert_array_equal(empty.transmittance(bvh, origins, dirs), 1)
        np.testing.assert_array_equal(empty.brute_transmittance(origins, dirs), 1)

    def test_cancellation_between_chunks(self):
        tri = soup(10)
        bvh = Bvh.build(*tri.aabbs())
        event = threading.Event()
        origins = tri.v0.copy()
        dirs = np.ones_like(origins)
        def callback(r, p):
            event.set()
        with self.assertRaises(Cancelled):
            traverse(bvh, origins, dirs, np.inf, callback, chunk=1, cancel=event)
        with self.assertRaises(Cancelled):
            tri.transmittance(bvh, origins, dirs, cancel=event)
        with self.assertRaises(Cancelled):
            Bvh.build(*tri.aabbs(), cancel=event)

    def test_large_shadow_render(self):
        sphere = s._sphere(1, 142, (1,1,1,.25), s.Transform3D())
        self.assertGreaterEqual(len(sphere.triangles), 20000)
        ground = s._card(8,8,(1,1,1,1),s.Transform3D(s.Vec3(0,-1.2,0),s.Vec3(-90,0,0)))
        light = s.Light(position=s.Vec3(-2,4,2), shadows=True)
        scene = s.Scene((sphere,ground),(light,))
        camera = s.Camera(s.Transform3D(s.Vec3(3,3,5)))
        with patch.object(s.Bvh, 'build', wraps=Bvh.build) as build:
            actual = s.render(scene,camera,8,6,ambient=.1)
            self.assertEqual(build.call_count, 1)
        with patch.object(s, '_SHADOW_BRUTE_THRESHOLD', 100000):
            expected = s.render(scene,camera,8,6,ambient=.1)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        self.assertTrue(np.any(actual[...,3]))


if __name__ == '__main__':
    unittest.main()


class NearestHitsTests(unittest.TestCase):
    def test_sorted_truncated_reference(self):
        tri = soup(90, 21)
        # Repeat primitives to exercise deterministic equal-depth ties.
        tri = TriangleSet(np.tile(tri.v0, (2, 1)), np.tile(tri.e1, (2, 1)),
                          np.tile(tri.e2, (2, 1)), .5)
        rng = np.random.default_rng(42)
        dirs = rng.normal(size=(90, 3))
        origins = tri.v0[:90]+.25*(tri.e1[:90]+tri.e2[:90])-dirs
        bvh = Bvh.build(*tri.aabbs(), leaf_size=2)
        for lo, hi in ((0., 20.), (.5, np.linspace(.8, 4, 90))):
            # Single leaf tests every primitive, independent of BVH pruning.
            brute = tri.all_hits(Bvh.build(*tri.aabbs(), leaf_size=200),
                                 origins, dirs, lo, hi, max_hits=200)
            for k in (1, 3, 8, 200):
                for chunk in (7, 128):
                    actual = tri.nearest_hits(bvh, origins, dirs, lo, hi, k, chunk=chunk)
                    for a, b in zip(actual, brute):
                        np.testing.assert_array_equal(a, b[:k])

    def test_cancellation_and_empty(self):
        tri = soup(0)
        self.assertEqual(len(tri.nearest_hits(Bvh.build(*tri.aabbs()),
                                             np.zeros((1, 3)), np.ones((1, 3)), 0, 10, 1)[0]), 0)
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            tri.nearest_hits(Bvh.build(*tri.aabbs()), np.zeros((1, 3)),
                             np.ones((1, 3)), 0, 10, 1, cancel=event)
