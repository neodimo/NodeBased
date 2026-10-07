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


class SourceGeometryThatAppearsLaterTests(unittest.TestCase):
    """A surface source was sampled once, at its start frame, unless Inherit velocity was above zero: steam off a liquid
    that did not exist yet at frame 1 never came out. `animated` samples the geometry every frame."""

    def graph(self, animated):
        from nodebased.core import Dispatcher
        d = Dispatcher()
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "ball", "type": "Cylinder3D", "params": {"cyl_radius": 0.3, "cyl_height": 0.3, "ty": 9.0}},
            {"op": "create", "id": "src", "type": "FluidSource3D", "params": {"fluid_emit_from": "surface",
                                                                              "animated": animated, "src_density": 1.0}},
            {"op": "connect", "id": "src", "input": "geo", "source": "ball"},
            {"op": "create", "id": "smoke", "type": "FluidSolver3D", "params": {
                "division_size": 0.1, "bounds_min_x": -1.0, "bounds_max_x": 1.0, "bounds_min_z": -1.0, "bounds_max_z": 1.0,
                "bounds_min_y": 0.0, "bounds_max_y": 2.0, "pressure": "cpu", "max_iterations": 30}},
            {"op": "connect", "id": "smoke", "input": "fluid", "source": "src"},
            {"op": "set_key", "id": "ball", "param": "ty", "frame": 1, "value": 9.0},
            {"op": "set_key", "id": "ball", "param": "ty", "frame": 3, "value": 0.5}]})
        d.execute({"op": "time", "first": 1, "last": 6})
        return d

    def smoke_at(self, animated, frame=6):
        from nodebased.imaging import Evaluator
        d = self.graph(animated)
        volume = Evaluator().evaluate_raster(d.document, "smoke", frame=frame, typed=True)
        return float(np.asarray(volume.density).sum())

    def test_an_animated_source_emits_once_its_geometry_arrives(self):
        self.assertGreater(self.smoke_at(1), 1.0)

    def test_a_static_source_keeps_its_start_frame_footprint(self):
        self.assertEqual(self.smoke_at(0), 0.0)


class DigestOnlyTests(unittest.TestCase):
    """Smoke that collides with, or is emitted from, a liquid's surface hashes the surface of every frame in the
    document's range. It used to build that surface (a solve, a level set and a mesh) once per frame in the range, for
    every frame it was asked to solve: the cost of a bake grew with the square of its length."""

    def graph(self):
        from nodebased.core import Dispatcher
        d = Dispatcher()
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "src", "type": "FluidSource3D", "params": {
                "fluid_type": "liquid", "src_center_y": 0.7, "src_radius": 0.3, "src_vel_y": -0.5}},
            {"op": "create", "id": "liq", "type": "FluidLiquidSolver3D", "params": {
                "division_size": 0.1, "bounds_min_x": -0.5, "bounds_max_x": 0.5, "bounds_min_z": -0.5, "bounds_max_z": 0.5,
                "bounds_max_y": 1.2, "pressure": "cpu", "particles_per_cell": 2, "substeps": 1, "max_iterations": 30}},
            {"op": "connect", "id": "liq", "input": "fluid", "source": "src"},
            {"op": "create", "id": "cache", "type": "ParticleCache3D", "params": {}},
            {"op": "connect", "id": "cache", "input": "particles", "source": "liq"},
            {"op": "create", "id": "surface", "type": "FluidSurface3D", "params": {}},
            {"op": "connect", "id": "surface", "input": "particles", "source": "cache"}]})
        d.execute({"op": "time", "first": 1, "last": 12})
        return d

    def test_the_digest_alone_equals_the_digest_of_the_built_surface_and_builds_nothing(self):
        from nodebased.imaging import Evaluator
        from nodebased import flip3d
        d = self.graph()
        evaluator = Evaluator()
        for frame in (3, 7):
            before = flip3d.SOLVER_STATS["steps"]
            none, quick = evaluator.evaluate_raster(d.document, "surface", frame=frame, typed=True, return_digest=True,
                                                    digest_only=True)
            self.assertIsNone(none)
            self.assertEqual(flip3d.SOLVER_STATS["steps"], before, "the digest solved the liquid")
            geometry, full = evaluator.evaluate_raster(d.document, "surface", frame=frame, typed=True, return_digest=True)
            self.assertEqual(quick, full)
            self.assertGreater(flip3d.SOLVER_STATS["steps"], before)

    def test_a_smoke_that_collides_with_the_surface_does_not_mesh_the_range_to_solve_a_frame(self):
        from nodebased.imaging import Evaluator
        from nodebased import liquid_surface
        d = self.graph()
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "gsrc", "type": "FluidSource3D", "params": {"src_center_y": 0.2, "src_radius": 0.1}},
            {"op": "create", "id": "col", "type": "FluidCollide3D", "params": {"animated": 1}},
            {"op": "connect", "id": "col", "input": "fluid", "source": "gsrc"},
            {"op": "connect", "id": "col", "input": "geometry", "source": "surface"},
            {"op": "create", "id": "gas", "type": "FluidSolver3D", "params": {
                "division_size": 0.1, "bounds_min_x": -0.5, "bounds_max_x": 0.5, "bounds_min_z": -0.5, "bounds_max_z": 0.5,
                "pressure": "cpu", "max_iterations": 20, "bounds_max_y": 1.2}},
            {"op": "connect", "id": "gas", "input": "fluid", "source": "col"}]})
        calls = {"n": 0}
        original = liquid_surface.marching_tetrahedra

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)
        liquid_surface.marching_tetrahedra = counting
        try:
            Evaluator().evaluate_raster(d.document, "gas", frame=2, typed=True)
        finally:
            liquid_surface.marching_tetrahedra = original
        # the collider samples the surface at frames 2 and 3 (and the solves of 1 and 2 sample theirs): a handful, not
        # one per frame of the 12-frame range for every solve
        self.assertLessEqual(calls["n"], 8)


class WhitewaterAtFineResolutionTests(unittest.TestCase):
    """The kinetic-energy potential was 0.5 * m * v^2 with m the particle's volume, so it shrank with the cube of the
    resolution: at 128 cells and finer no liquid, however violent, reached the knob's range (its smallest allowed
    maximum is 1e-6), and Hot pour's whitewater never emitted a particle."""

    def impact(self, spacing, **params):
        from types import SimpleNamespace
        from nodebased import whitewater
        from tests.test_whitewater import liquid
        angle = np.linspace(-.5, .5, 12)
        points = np.column_stack((.2 + .15 * np.cos(angle), .3 + .15 * np.sin(angle), np.full(12, .1)))
        axes = [np.arange(12, dtype=np.float32) * .1 - .1 for _ in range(3)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        phi = np.sqrt((x - .2) ** 2 + (y - .3) ** 2 + (z - .1) ** 2) - .15
        normals = points - np.array([.2, .3, .1])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        velocity = normals * 3.0
        velocity[:, 2] += np.where(np.arange(12) % 2, -1.0, 1.0)
        source = liquid(points, velocity, phi)
        from dataclasses import replace
        source = replace(source, stream=SimpleNamespace(spacing=spacing))
        solver = whitewater.FluidWhitewater3D({"max_particles": 60, "spray_threshold": .01, "foam_threshold": .01,
                                               "spray_emission": 1., "foam_emission": 1., "wave_crest_min": 0.0,
                                               "wave_crest_max": .2, "kinetic_energy_min": 1e-4,
                                               "kinetic_energy_max": 3e-3, "whitewater_backend": "cpu", **params})
        return len(solver.step(solver.initial_state(), source, 1).ids)

    def test_a_fine_liquid_emits_nothing_by_default_and_emits_with_the_energy_per_unit_mass(self):
        self.assertEqual(self.impact(0.003), 0)
        self.assertGreater(self.impact(0.003, kinetic_energy_per_mass=1), 0)

    def test_the_default_leaves_a_coarse_liquid_exactly_as_it_was(self):
        self.assertEqual(self.impact(0.15), self.impact(0.15, kinetic_energy_per_mass=0))


def reference_neighbour_lists(positions, h):
    """The loop over particles and bins that built the whitewater's neighbour lists before this step."""
    n = len(positions)
    bins = np.floor(positions / h).astype(np.int64)
    table = {}
    for i, key in enumerate(map(tuple, bins)):
        table.setdefault(key, []).append(i)
    starts = np.zeros(n + 1, np.uint32)
    members = []
    for i, base in enumerate(bins):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    members.extend(table.get((base[0] + dx, base[1] + dy, base[2] + dz), ()))
        starts[i + 1] = len(members)
    return starts, np.asarray(members or [0], np.uint32)


class WhitewaterNeighbourListTests(unittest.TestCase):
    """The whitewater's neighbour search ran in Python at one pass per particle and bin: seconds a frame at 100,000
    particles, and the largest cost of the Hot pour bake (296 of 435 seconds at 128 cells)."""

    def test_the_array_build_is_the_loop_build_bit_for_bit(self):
        from nodebased.fluid_gpu_whitewater import neighbor_lists
        rng = np.random.default_rng(3)
        for n, h in ((1, 0.1), (2, 0.5), (50, 0.2), (3000, 0.05), (4000, 0.5)):
            points = rng.uniform(-0.3, 0.3, (n, 3))
            want, got = reference_neighbour_lists(points, h), neighbor_lists(points, h)
            np.testing.assert_array_equal(got[0], want[0], err_msg=f"starts, n={n}")
            np.testing.assert_array_equal(got[1], want[1], err_msg=f"members, n={n}")

    def test_it_is_several_times_faster_on_a_pool_sized_set(self):
        import time
        from nodebased.fluid_gpu_whitewater import neighbor_lists
        points = np.random.default_rng(4).uniform(0.0, 0.2, (30000, 3))
        started = time.perf_counter()
        reference_neighbour_lists(points, 0.02)
        loop = time.perf_counter() - started
        started = time.perf_counter()
        neighbor_lists(points, 0.02)
        arrays = time.perf_counter() - started
        self.assertLess(arrays * 3.0, loop, f"loop {loop:.2f} s, arrays {arrays:.2f} s")


class WhitewaterNearestParticleTests(unittest.TestCase):
    """Foam and bubbles follow the nearest liquid particle. The search hashed every liquid particle in Python on each call:
    1.7 of the 2.1 seconds of a whitewater frame at 256 cells."""

    def test_the_array_search_returns_the_loops_indices_including_ties_and_empty_neighbourhoods(self):
        from nodebased import whitewater
        rng = np.random.default_rng(5)
        cases = [(rng.uniform(-1, 1, (n, 3)), rng.uniform(-1.3, 1.3, (m, 3)))
                 for n, m in ((10, 5), (500, 300), (5000, 2000))]
        far = rng.uniform(-1, 1, (400, 3))
        cases.append((far, np.vstack([far[:50] + 0.01, np.full((3, 3), 40.0)])))                  # queries far outside
        cases.append((np.repeat(rng.uniform(0, 1, (100, 3)), 5, axis=0), rng.uniform(0, 1, (300, 3))))   # exact ties
        for points, query in cases:
            np.testing.assert_array_equal(whitewater._nearest_indices(query, points),
                                          whitewater._nearest_indices_loop(query, points))

    def test_it_is_much_faster_on_a_pool_sized_liquid(self):
        import time
        from nodebased import whitewater
        rng = np.random.default_rng(6)
        points, query = rng.uniform(0, 1, (100000, 3)), rng.uniform(0, 1, (10000, 3))
        started = time.perf_counter()
        whitewater._nearest_indices_loop(query, points)
        loop = time.perf_counter() - started
        started = time.perf_counter()
        whitewater._nearest_indices(query, points)
        arrays = time.perf_counter() - started
        self.assertLess(arrays * 4.0, loop, f"loop {loop:.2f} s, arrays {arrays:.2f} s")


if __name__ == "__main__":
    unittest.main()
