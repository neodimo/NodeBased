"""Lane 6 step N3: what broke when the fluid features met in the Hot pour scene, one test per fix."""
import types
import unittest

import numpy as np

from nodebased import flip3d, fluid3d
from tests.test_fluid3d import box_collider, box_triangles


def stub(shape, origin=(0.0, 0.0, 0.0), voxel=1.0):
    return types.SimpleNamespace(shape=tuple(shape), origin=np.asarray(origin, float), voxel=float(voxel))


def reference_voxelize(triangles, shape, values=None):
    """The one-triangle-at-a-time voxeliser this step replaced, kept as the oracle for the batched one."""
    mask = np.zeros(shape, bool)
    total = None if values is None else np.zeros(tuple(shape) + (3,), np.float64)
    count = None if values is None else np.zeros(shape, np.int32)
    hi_limit = np.array(shape) - 1
    for t, tri in enumerate(np.asarray(triangles, np.float64)):
        lo = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(tri.max(axis=0)).astype(int), hi_limit)
        if np.any(hi < lo):
            continue
        ii, jj, kk = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1),
                                 np.arange(lo[2], hi[2] + 1), indexing="ij")
        cells = np.stack((ii.ravel(), jj.ravel(), kk.ravel()), axis=1)
        keep = fluid3d._triangle_box_overlap(tri, cells + 0.5)
        cells = cells[keep]
        if not len(cells):
            continue
        mask[cells[:, 0], cells[:, 1], cells[:, 2]] = True
        if total is not None:
            total[cells[:, 0], cells[:, 1], cells[:, 2]] += values[t]
            count[cells[:, 0], cells[:, 1], cells[:, 2]] += 1
    if total is None:
        return mask
    return mask, total / np.maximum(count, 1)[..., None]


def sphere_triangles(centre, radius, rings=14, segments=20):
    pts = []
    for r in range(rings + 1):
        phi = np.pi * r / rings
        pts.append([(centre[0] + radius * np.sin(phi) * np.cos(2 * np.pi * s / segments), centre[1] + radius * np.cos(phi),
                     centre[2] + radius * np.sin(phi) * np.sin(2 * np.pi * s / segments)) for s in range(segments)])
    tri = []
    for r in range(rings):
        for s in range(segments):
            a, b = pts[r][s], pts[r][(s + 1) % segments]
            c, d = pts[r + 1][s], pts[r + 1][(s + 1) % segments]
            tri += [[a, c, b], [b, c, d]]
    return np.array(tri, float)


class BatchedVoxeliserTests(unittest.TestCase):
    def test_the_batched_voxeliser_matches_the_one_triangle_loop_on_a_sphere_and_on_random_triangles(self):
        rng = np.random.default_rng(7)
        cases = [sphere_triangles((9.3, 8.1, 10.6), 6.4), box_triangles((2.2, 3.7, 4.1), (11.3, 9.9, 12.5)),
                 rng.uniform(-3.0, 21.0, size=(300, 3, 3)), np.repeat(rng.uniform(0, 16, size=(40, 1, 3)), 3, axis=1)]
        for tri in cases:
            motion = rng.normal(size=(len(tri), 3))
            self.assertTrue(np.array_equal(fluid3d.voxelize_surface(tri, (18, 18, 18)),
                                           reference_voxelize(tri, (18, 18, 18))))
            got_mask, got = fluid3d.voxelize_surface(tri, (18, 18, 18), motion)
            want_mask, want = reference_voxelize(tri, (18, 18, 18), motion)
            self.assertTrue(np.array_equal(got_mask, want_mask))
            self.assertTrue(np.array_equal(got, want))

    def test_chunks_do_not_change_the_result(self):
        tri = sphere_triangles((20.0, 20.0, 20.0), 15.0, 30, 40)
        whole = fluid3d.voxelize_surface(tri, (40, 40, 40))
        old = fluid3d.PAIR_CHUNK
        try:
            fluid3d.PAIR_CHUNK = 500
            self.assertTrue(np.array_equal(fluid3d.voxelize_surface(tri, (40, 40, 40)), whole))
        finally:
            fluid3d.PAIR_CHUNK = old

    def test_a_dense_liquid_sized_mesh_voxelises_in_a_second_or_two(self):
        import time
        tri = sphere_triangles((60.0, 60.0, 60.0), 50.0, 140, 280)          # 78,000 triangles
        started = time.perf_counter()
        surface = fluid3d.voxelize_surface(tri, (120, 120, 120))
        self.assertLess(time.perf_counter() - started, 4.0)
        self.assertGreater(int(surface.sum()), 20000)


class ColliderCrossingTheDomainTests(unittest.TestCase):
    def test_a_table_that_runs_off_the_bottom_of_the_domain_is_solid_down_to_the_floor(self):
        # a slab whose top is at y = 6.5 and whose underside is below the domain: before the fix only its top
        # face's cells (one layer) were solid, because the flood fill escaped through the cut underside
        collider = box_collider((2.2, -5.0, 2.2), (12.8, 6.5, 12.8))
        solid, _velocity = collider.mask(stub((16, 16, 16)), 1)
        self.assertTrue(solid[3:12, 0:6, 3:12].all(), "the slab must be solid from the domain floor up to its top")
        self.assertFalse(solid[3:12, 7:, 3:12].any())

    def test_a_box_inside_the_domain_fills_exactly_as_before(self):
        collider = box_collider((4.2, 4.2, 4.2), (9.8, 9.8, 9.8))
        solid, _velocity = collider.mask(stub((16, 16, 16)), 1)
        self.assertEqual(int(solid.sum()), 6 ** 3)

    def test_a_wall_that_runs_off_the_side_is_solid_up_to_the_edge(self):
        collider = box_collider((-9.0, 2.2, 2.2), (4.8, 9.8, 9.8))
        solid, _velocity = collider.mask(stub((16, 16, 16)), 1)
        self.assertTrue(solid[0:4, 3:9, 3:9].all())


class WideColliderDoesNotInflateTheDomainTests(unittest.TestCase):
    """A table far wider than the liquid used to pull the adaptive box out to the table's whole width."""

    def test_liquid_domain_stays_near_the_liquid_when_a_wide_table_is_in_the_scene(self):
        table = box_collider((-120.0, -6.0, -120.0), (120.0, 0.5, 120.0))
        source = fluid3d.Source(center=(0.0, 14.0, 0.0), radius=3.0, fluid_type="liquid", end_frame=1)
        solver = flip3d.Liquid3D({"nx": 32, "ny": 32, "nz": 32, "origin_x": -16.0, "origin_y": -2.0, "origin_z": -16.0,
                                  "voxel_size": 1.0, "auto_resize": 1, "padding": 8, "max_size": 256,
                                  "particles_per_cell": 2, "substeps": 1, "gravity": 0.02, "start_frame": 1,
                                  "max_iterations": 40}, sources=[source], colliders=[table])
        state = solver.initial_state()
        for frame in range(1, 4):
            state = solver.step(state, frame=frame)
        shape = state.meta["domain_shape"]
        self.assertLessEqual(shape[0], 64, f"the box grew to {shape} to cover the table")
        self.assertLessEqual(shape[2], 64)
        solid = solver._solid_for(3)[0]
        self.assertTrue(solid[:, :2, :].any(), "the table is still in the box below the liquid")

    def test_smoke_domain_stays_near_the_smoke_when_a_wide_table_is_in_the_scene(self):
        table = box_collider((-120.0, -6.0, -120.0), (120.0, 0.5, 120.0))
        solver = fluid3d.Smoke3D({"nx": 16, "ny": 16, "nz": 16, "default_source": 0, "auto_resize": 1, "padding": 8,
                                  "max_size": 256, "boundary_x": "open", "boundary_y": "open", "boundary_z": "open"},
                                 colliders=[table])
        state = solver.initial_state()
        state.arrays["density"][6:10, 3:7, 6:10] = 1.0
        resized = solver._resize_active_domain(state, frame=1)
        shape = resized.meta["domain_shape"]
        self.assertLessEqual(max(shape), 48, f"the box grew to {shape} to cover the table")


class SparseTileMaskSurvivesAResizeTests(unittest.TestCase):
    """A resident sparse smoke checkpoint carries `tile_mask`, one value per 8-cell tile. The adaptive resize treated it
    as a cell field and padded it out to the cell grid, so resuming the next substep raised "Invalid data_length"."""

    def test_the_tile_mask_follows_the_box_in_whole_tiles(self):
        solver = fluid3d.Smoke3D({"nx": 32, "ny": 32, "nz": 32, "default_source": 0, "auto_resize": 1, "padding": 0,
                                  "max_size": 128, "boundary_x": "open", "boundary_y": "open", "boundary_z": "open"})
        state = solver.initial_state()
        state.arrays["density"][16:24, 8:16, 8:16] = 1.0               # one tile, at tile index (2, 1, 1)
        mask = np.zeros((4, 4, 4), np.uint8)
        mask[2, 1, 1] = 1
        state.arrays["tile_mask"] = mask
        resized = solver._resize_active_domain(state, frame=1)
        shape = tuple(resized.meta["domain_shape"])
        start = np.round((np.asarray(resized.meta["domain_origin"]) - solver.origin) / solver.voxel).astype(int)
        got = resized.arrays["tile_mask"]
        self.assertEqual(got.shape, tuple(n // 8 for n in shape))
        self.assertEqual(int(got.sum()), 1)
        self.assertEqual(tuple(int(i) for i in np.argwhere(got)[0]), tuple(int(v) for v in (np.array((2, 1, 1)) - start // 8)))


class SteamFromAColliderSurfaceTests(unittest.TestCase):
    """Steam rises off a liquid that is also the smoke's collider: the emitting cells are the collider's own cells, and
    the solver clears density in every collider cell, so the steam never appeared."""

    def solve(self, frames=6, **source_args):
        tri = box_triangles((3.2, 3.2, 3.2), (12.8, 8.0, 12.8))
        track = fluid3d.GeometryTrack(lambda frame: tri, animated=False)
        source = fluid3d.Source("surface", density=1.0, temperature=1.0, track=track, **source_args)
        collider = fluid3d.Collider(track)
        solver = fluid3d.Smoke3D({"nx": 16, "ny": 24, "nz": 16, "default_source": 0, "substeps": 1, "max_iterations": 60,
                                  "boundary_x": "open", "boundary_y": "open", "boundary_z": "open"},
                                 sources=[source], colliders=[collider])
        state = solver.initial_state()
        for frame in range(1, frames + 1):
            state = solver.step(state, frame=frame)
        return solver, state

    def test_steam_appears_just_outside_the_liquid_and_never_inside_it(self):
        solver, state = self.solve(dilate=1)
        density = state.arrays["density"]
        solid = solver._solid_for(1)[0]
        self.assertEqual(float(density[solid].max()), 0.0)
        self.assertGreater(float(density[:, 8:12, :].sum()), 0.5, "no steam above the liquid's top face")

    def test_without_the_dilation_nothing_changes(self):
        solver, state = self.solve()
        solid = solver._solid_for(1)[0]
        self.assertEqual(float(state.arrays["density"][solid].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
