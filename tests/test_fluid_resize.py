"""Adaptive smoke-domain checkpoints (Lane 6, M1)."""
import unittest
import tempfile

import numpy as np

from nodebased import core, fluid3d, flip3d, fluid_gpu_solver, simcache, viewport3d


class SmokeResizeTests(unittest.TestCase):
    def test_shrinking_domain_preserves_active_field_mass(self):
        solver = fluid3d.Smoke3D({"nx": 32, "ny": 32, "nz": 32, "default_source": 0,
                                 "auto_resize": 1, "padding": 1, "max_size": 64})
        state = solver.initial_state()
        state.arrays["density"][14:18, 14:18, 14:18] = 0.25
        before = float(state.arrays["density"].sum())
        shrunk = solver._resize_active_domain(state)
        self.assertLess(shrunk.arrays["density"].size, state.arrays["density"].size)
        self.assertAlmostEqual(float(shrunk.arrays["density"].sum()), before, places=6)

    def test_resize_preserves_fields_and_checkpoint_shape(self):
        solver = fluid3d.Smoke3D({"nx": 16, "ny": 16, "nz": 16, "default_source": 0,
                                 "auto_resize": 1, "padding": 2, "max_size": 32,
                                 "buoyancy_density": 0.0, "buoyancy_temperature": 0.0,
                                 "vorticity": 0.0, "boundary_x": "open", "boundary_y": "open",
                                 "boundary_z": "open", "max_iterations": 60})
        state = solver.initial_state()
        state.arrays["density"][11:14, 7:10, 7:10] = 1.0
        before = float(state.arrays["density"].sum())
        result = solver.step(state, frame=1)
        self.assertNotEqual(tuple(result.arrays["density"].shape), (16, 16, 16))
        self.assertEqual(result.meta["domain_shape"], list(result.arrays["density"].shape))
        self.assertGreaterEqual(float(result.arrays["density"].sum()), before - 1.0e-5)
        saved = solver.checkpoint(result)
        restored = solver.restore(saved)
        self.assertEqual(restored.meta["domain_shape"], result.meta["domain_shape"])
        np.testing.assert_array_equal(restored.arrays["density"], result.arrays["density"])

    def test_simcache_serializes_a_resized_domain_checkpoint(self):
        solver = fluid3d.Smoke3D({"nx": 16, "ny": 16, "nz": 16, "default_source": 0,
                                 "auto_resize": 1, "padding": 1, "max_size": 32,
                                 "buoyancy_density": 0.0, "buoyancy_temperature": 0.0,
                                 "vorticity": 0.0, "boundary_x": "open", "boundary_y": "open",
                                 "boundary_z": "open", "max_iterations": 60})
        initial = solver.initial_state()
        initial.arrays["density"][12:15, 7:10, 7:10] = 1.0
        first = solver.step(initial, frame=1)
        with tempfile.TemporaryDirectory() as root:
            cache = simcache.SimCache(root=root)
            self.assertTrue(cache._write_disk(("resize-test", 1), first))
            loaded = cache.get("resize-test", 1)
        self.assertEqual(tuple(loaded.arrays["density"].shape), tuple(first.meta["domain_shape"]))
        np.testing.assert_array_equal(loaded.arrays["density"], first.arrays["density"])

    def test_served_volume_cache_restores_per_frame_shape_and_origin(self):
        from types import SimpleNamespace
        stream = SimpleNamespace(run="adaptive-run", backend="cpu", shape=(16, 16, 16),
                                 origin=(0.0, 0.0, 0.0), voxel=0.25)
        cache = simcache.SimCache(enabled=False)
        out_run = simcache.run_key(stream.run, {"out": ["float32", "density"]})
        density = np.ones((8, 16, 24), np.float32)
        cache.put(out_run, 2, simcache.State({"density": density},
                                             {"domain_shape": [8, 16, 24],
                                              "domain_origin": [-1.0, 2.0, 0.5]}))
        volume = fluid3d.cached_volume(stream, 2, cache, None, "float32", "density")
        self.assertEqual(volume.density.shape, (8, 16, 24))
        self.assertEqual(volume.origin, (-1.0, 2.0, 0.5))

    @unittest.skipUnless(fluid_gpu_solver.available(), "no wgpu compute adapter")
    def test_sparse_gpu_domain_reallocates_on_tile_aligned_resize(self):
        solver = fluid_gpu_solver.GpuSmoke3D({"nx": 16, "ny": 16, "nz": 16, "default_source": 0,
                                             "auto_resize": 1, "padding": 4, "max_size": 64,
                                             "buoyancy_density": 0.0, "buoyancy_temperature": 0.0,
                                             "vorticity": 0.0, "boundary_x": "open", "boundary_y": "open",
                                             "boundary_z": "open", "max_iterations": 80}, sparse=True)
        initial = solver.initial_state()
        initial.arrays["density"][7:9, 13:15, 7:9] = 1.0
        initial.arrays["v"][:] = 4.0
        first = solver.step(initial, frame=1)
        self.assertTrue(all(n % 8 == 0 for n in first.meta["domain_shape"]))
        second = solver.step(first, frame=2)
        self.assertEqual(tuple(second.arrays["density"].shape), tuple(second.meta["domain_shape"]))

    def test_adaptive_domain_keeps_animated_collider_masks_on_the_frame_grid(self):
        from tests.test_fluid3d import box_triangles
        def provider(frame):
            y = 10.0 + float(frame)
            return box_triangles((10.0, y, 2.0), (12.0, y + 2.0, 6.0))
        track = fluid3d.GeometryTrack(provider, animated=True, start_frame=1)
        collider = fluid3d.Collider(track, animated=True)
        solver = fluid3d.Smoke3D({"nx": 8, "ny": 8, "nz": 8, "origin_x": 0.0, "origin_y": 0.0,
                                  "origin_z": 0.0, "voxel_size": 1.0, "default_source": 0,
                                  "auto_resize": 1, "padding": 0, "max_size": 64,
                                  "boundary_x": "open", "boundary_y": "open", "boundary_z": "open"},
                                 colliders=[collider])
        resized = solver._resize_active_domain(solver.initial_state(), frame=1)
        solver._sync_domain(resized)
        solid, velocity, _ = solver._solid_for(1)
        self.assertEqual(solid.shape, tuple(resized.meta["domain_shape"]))
        self.assertEqual(velocity.shape, solid.shape + (3,))

    def test_adaptive_liquid_keeps_its_world_floor_anchored(self):
        source = fluid3d.Source(center=(0.0, 1.0, 0.0), radius=0.45, fluid_type="liquid", end_frame=1)
        solver = flip3d.Liquid3D({"nx": 16, "ny": 16, "nz": 16, "origin_x": -1.0,
                                  "origin_y": 0.0, "origin_z": -1.0, "voxel_size": 0.125,
                                  "auto_resize": 1, "padding": 8, "max_size": 64,
                                  "particles_per_cell": 2, "substeps": 2, "gravity": 0.02,
                                  "start_frame": 1, "max_iterations": 100}, sources=[source])
        state = solver.initial_state()
        for frame in range(1, 25):
            state = solver.step(state, frame=frame)
        self.assertGreater(float(state.arrays["position"][:, 1].min()), -1.0e-4)
        self.assertGreaterEqual(state.meta["domain_origin"][1], solver.floor_y - 1.0e-8)

    def test_source_above_old_top_gets_a_frame_box_before_advection(self):
        adaptive = fluid3d.Smoke3D({"nx": 16, "ny": 16, "nz": 16, "default_source": 0,
                                   "auto_resize": 1, "padding": 4, "max_size": 64,
                                   "buoyancy_density": 0.0, "buoyancy_temperature": 0.0,
                                   "vorticity": 0.0, "boundary_x": "open", "boundary_y": "open",
                                   "boundary_z": "open", "max_iterations": 80})
        initial = adaptive.initial_state()
        initial.arrays["density"][7:9, 13:15, 7:9] = 1.0
        initial.arrays["v"][:] = 4.0
        result = adaptive.step(initial, frame=1)
        origin_y = result.meta["domain_origin"][1]
        self.assertGreater(origin_y + result.arrays["density"].shape[1], 16)
        self.assertEqual(float(result.arrays["density"][:, 15 - int(origin_y), :].max()), 0.0)

    def test_checkpoint_restart_across_resize_matches_uninterrupted(self):
        params = {"nx": 16, "ny": 16, "nz": 16, "default_source": 0,
                  "auto_resize": 1, "padding": 2, "max_size": 32,
                  "buoyancy_density": 0.0, "buoyancy_temperature": 0.0,
                  "vorticity": 0.0, "boundary_x": "open", "boundary_y": "open",
                  "boundary_z": "open", "max_iterations": 60}
        uninterrupted = fluid3d.Smoke3D(params)
        initial = uninterrupted.initial_state()
        initial.arrays["density"][12:15, 7:10, 7:10] = 1.0
        first = uninterrupted.step(initial, frame=1)
        expected = uninterrupted.step(first, frame=2)
        resumed_solver = fluid3d.Smoke3D(params)
        cache = simcache.SimCache(enabled=False)
        cache.put("resized", 1, first)
        actual = simcache.solve_to_frame(cache, "resized", 2, 1, 1, 0,
                                         resumed_solver.initial_state, resumed_solver.step)
        self.assertEqual(actual, expected)

    def test_resized_solve_matches_a_fixed_oversized_reference(self):
        params = {"nx": 16, "ny": 16, "nz": 16, "default_source": 0,
                  "auto_resize": 1, "padding": 4, "max_size": 64,
                  "buoyancy_density": 0.0, "buoyancy_temperature": 0.0,
                  "vorticity": 0.0, "boundary_x": "open", "boundary_y": "open",
                  "boundary_z": "open", "max_iterations": 80}
        adaptive = fluid3d.Smoke3D(params)
        initial = adaptive.initial_state()
        initial.arrays["density"][7:9, 13:15, 7:9] = 1.0
        initial.arrays["v"][:] = 4.0
        prepared = adaptive._resize_active_domain(initial)
        origin = np.asarray(prepared.meta["domain_origin"], np.float64)
        shape = tuple(prepared.arrays["density"].shape)
        fixed_shape = tuple(n + 32 for n in shape)
        fixed_params = {**params, "nx": fixed_shape[0], "ny": fixed_shape[1], "nz": fixed_shape[2],
                        "auto_resize": 0, "origin_x": origin[0], "origin_y": origin[1],
                        "origin_z": origin[2]}
        fixed = fluid3d.Smoke3D(fixed_params)
        fixed_state = fixed.initial_state()
        for name, array in prepared.arrays.items():
            slices = tuple(slice(0, n) for n in array.shape)
            fixed_state.arrays[name][slices] = array
        actual = adaptive.step(initial, frame=1)
        reference = fixed.step(fixed_state, frame=1)
        offset = np.rint((np.asarray(actual.meta["domain_origin"]) - origin)).astype(int)
        crop = tuple(slice(int(offset[a]), int(offset[a] + actual.arrays["density"].shape[a])) for a in range(3))
        np.testing.assert_allclose(actual.arrays["density"], reference.arrays["density"][crop], atol=2.0e-4, rtol=2.0e-4)


class LiquidResizeTests(unittest.TestCase):
    def test_particles_drive_growth_and_are_retained_in_world_space(self):
        source = fluid3d.Source(center=(4.0, 7.0, 4.0), radius=1.5, fluid_type="liquid")
        solver = flip3d.Liquid3D({"nx": 8, "ny": 8, "nz": 8, "origin_x": 0.0,
                                  "origin_y": 0.0, "origin_z": 0.0, "voxel_size": 1.0,
                                  "auto_resize": 1, "padding": 2, "max_size": 32,
                                  "particles_per_cell": 1, "substeps": 1, "gravity": 0.0,
                                  "start_frame": 1, "max_iterations": 60}, sources=[source])
        result = solver.step(solver.initial_state(), frame=1)
        self.assertGreater(result.arrays["position"].shape[0], 0)
        self.assertNotEqual(result.meta["domain_shape"], [8, 8, 8])
        self.assertTrue(np.all(result.arrays["position"][:, 1] < 16.0))
        self.assertGreaterEqual(result.meta["domain_origin"][1] + result.meta["domain_shape"][1], 16)

    def test_restart_and_surface_outputs_follow_the_liquid_frame_box(self):
        source = fluid3d.Source(center=(4.0, 7.0, 4.0), radius=1.5, fluid_type="liquid", start_frame=1, end_frame=1)
        params = {"nx": 8, "ny": 8, "nz": 8, "origin_x": 0.0, "origin_y": 0.0, "origin_z": 0.0,
                  "voxel_size": 1.0, "auto_resize": 1, "padding": 2, "max_size": 32,
                  "particles_per_cell": 1, "substeps": 1, "gravity": 0.0, "start_frame": 1,
                  "max_iterations": 60, "flip_ratio": 0.95, "viscosity": 0.0, "narrow_band": 2.0,
                  "backend": "cpu"}
        uninterrupted = flip3d.Liquid3D(params, sources=[source])
        first = uninterrupted.step(uninterrupted.initial_state(), frame=1)
        expected = uninterrupted.step(first, frame=2)
        resumed = flip3d.Liquid3D(params, sources=[source])
        cache = simcache.SimCache(enabled=False)
        cache.put("liquid-resized", 1, first)
        actual = simcache.solve_to_frame(cache, "liquid-resized", 2, 1, 1, 0,
                                         resumed.initial_state, resumed.step)
        self.assertEqual(actual, expected)
        chain = fluid3d.FluidChain(sources=[source])
        stream = flip3d.LiquidStream(chain, {**params, "start_frame": 1, "seed": 0,
                                              "division_size": 1.0, "bounds_min_x": 0.0,
                                              "bounds_min_y": 0.0, "bounds_min_z": 0.0,
                                              "bounds_max_x": 8.0, "bounds_max_y": 8.0, "bounds_max_z": 8.0,
                                              "particles_per_cell": 1}, "liquid-resized", 24.0)
        instance = flip3d.instance_from_state(first, stream, 1)
        self.assertEqual(instance.surface.density.shape, tuple(first.meta["domain_shape"]))
        self.assertEqual(instance.surface.origin, tuple(first.meta["domain_origin"]))
        geom = flip3d.surface_geometry(instance, {"surface_resolution": 1, "detail_ratio": 1,
                                                   "particle_radius": 0.0, "smoothing": 0,
                                                   "thin_sheet_preservation": 0})
        self.assertGreater(len(geom.vertices), 0)
        foam = flip3d.foam_instance(instance, {"foam_speed": 0.0, "foam_curvature": 0.0, "foam_size": 1.0})
        self.assertTrue(np.all(foam.positions >= np.asarray(first.meta["domain_origin"])))


class FluidDomainViewportTests(unittest.TestCase):
    def test_viewport_box_uses_that_frames_voxel_dimensions_and_origin(self):
        from nodebased.scene3d import Volume
        volume = Volume(np.zeros((2, 3, 4), np.float32), voxel_size=0.5, origin=(-1.0, 2.0, 3.0))
        edges = viewport3d.fluid_domain_edges(volume)
        self.assertEqual(len(edges), 12)
        points = np.asarray(edges).reshape(-1, 3)
        np.testing.assert_allclose(points.min(axis=0), (-1.0, 2.0, 3.0))
        np.testing.assert_allclose(points.max(axis=0), (0.0, 3.5, 5.0))


class FluidDomainDefaultsTests(unittest.TestCase):
    def test_v19_migration_and_new_nodes_keep_explicit_bounds_fixed_by_default(self):
        legacy = core.empty_document()
        legacy["version"] = core.SCHEMA_VERSION - 1
        old_params = dict(core.SPECS["FluidSolver3D"]["params"])
        old_params.pop("auto_resize")
        old_params.pop("padding")
        old_params.pop("max_size")
        legacy["nodes"] = {"old": {"type": "FluidSolver3D", "params": old_params}}
        upgraded = core.upgrade_document(legacy)
        self.assertEqual(upgraded["version"], core.SCHEMA_VERSION)
        self.assertEqual(upgraded["nodes"]["old"]["params"]["auto_resize"], 0)
        self.assertEqual(core.SPECS["FluidSolver3D"]["params"]["auto_resize"], 0)
        self.assertEqual(core.SPECS["FluidLiquidSolver3D"]["params"]["auto_resize"], 0)


if __name__ == "__main__":
    unittest.main()
