"""Lane L5 step 2a, part 1: the deterministic particle solver (nodebased/particles.py).

Driven through `simcache.solve_to_frame` exactly as the nodes drive it, without any node or scene:
counts, ages, deaths, positions and velocities are asserted on the solved arrays.
"""
import math
import tempfile
import unittest

import numpy as np

from nodebased import particles, simcache
from nodebased.core import SPECS
from nodebased.scene3d import Transform3D, Vec3, _card, _cube, merge_geometry


def knobs(**overrides):
    return {**SPECS["ParticleEmitter3D"]["params"], **overrides}


def solve(params, frame, source=None, cache=None, fps=24.0, curves=None):
    emitter = particles.ParticleEmitter(params, curves, source, fps)
    run = simcache.run_key(None, params)
    cache = cache or simcache.SimCache(enabled=False)
    return simcache.solve_to_frame(cache, run, frame, emitter.start_frame, emitter.substeps,
                                   emitter.seed, emitter.initial_state, emitter.step), emitter


def same(a, b):
    return a.meta == b.meta and all(np.array_equal(a.arrays[n], b.arrays[n]) for n in particles.ARRAYS)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_arrays_bit_for_bit(self):
        params = knobs(emit_rate=13.5, spread=40.0, speed_variance=0.5, life_variance=0.5, size_variance=0.5,
                       substeps=3, seed=11)
        a, _ = solve(params, 30)
        b, _ = solve(params, 30)
        self.assertGreater(len(a.arrays["id"]), 50)
        self.assertTrue(same(a, b))
        for name in particles.ARRAYS:
            self.assertEqual(a.arrays[name].tobytes(), b.arrays[name].tobytes(), name)

    def test_different_seeds_differ(self):
        base = dict(emit_rate=20.0, spread=40.0, speed_variance=0.5, life_variance=0.5)
        a, _ = solve(knobs(seed=1, **base), 12)
        b, _ = solve(knobs(seed=2, **base), 12)
        self.assertEqual(a.meta["emitted"], b.meta["emitted"])         # the count rule ignores the seed
        self.assertFalse(np.array_equal(a.arrays["velocity"], b.arrays["velocity"]))
        self.assertFalse(np.array_equal(a.arrays["life"], b.arrays["life"]))

    def test_a_split_solve_across_two_caches_matches_a_direct_solve(self):
        params = knobs(emit_rate=9.0, spread=25.0, life_variance=0.4, substeps=2, seed=5)
        direct, _ = solve(params, 40)
        with tempfile.TemporaryDirectory() as root:
            first, _ = solve(params, 17, cache=simcache.SimCache(root))
            resumed, _ = solve(params, 40, cache=simcache.SimCache(root))   # a new "process": disk only
            self.assertGreaterEqual(len(first.arrays["id"]), 1)
        self.assertTrue(same(direct, resumed))


class EmissionTests(unittest.TestCase):
    def test_emission_count_is_floor_of_rate_times_frames(self):
        for rate, substeps in ((100.0, 1), (7.5, 1), (7.5, 4), (0.3, 1), (13.37, 5)):
            for frames in (1, 2, 3, 10, 40):
                params = knobs(emit_rate=rate, substeps=substeps, life=100000.0, max_particles=10_000_000)
                state, _ = solve(params, frames)
                expected = math.floor(rate * frames + particles.COUNT_EPSILON)
                self.assertEqual(state.meta["emitted"], expected, (rate, substeps, frames))
                self.assertEqual(len(state.arrays["id"]), expected)

    def test_per_second_rate_uses_the_document_frame_rate(self):
        params = knobs(emit_rate=30.0, emit_rate_unit="per_second", life=100000.0)
        for fps, per_frame in ((24.0, 1.25), (30.0, 1.0), (60.0, 0.5)):
            state, _ = solve(params, 24, fps=fps)
            self.assertEqual(state.meta["emitted"], math.floor(per_frame * 24 + particles.COUNT_EPSILON), fps)

    def test_ids_are_sequential_in_emission_order(self):
        state, _ = solve(knobs(emit_rate=6.0, life=100000.0), 5)
        self.assertEqual(state.arrays["id"].tolist(), list(range(30)))

    def test_nothing_exists_before_the_start_frame_and_a_negative_start_pre_rolls(self):
        params = knobs(start_frame=10, emit_rate=5.0, life=100000.0)
        self.assertEqual(len(solve(params, 9)[0].arrays["id"]), 0)
        self.assertEqual(len(solve(params, 10)[0].arrays["id"]), 5)
        self.assertEqual(len(solve(params, 12)[0].arrays["id"]), 15)
        rolled = knobs(start_frame=-9, emit_rate=5.0, life=100000.0)
        self.assertEqual(len(solve(rolled, 1)[0].arrays["id"]), 55)      # frames -9 through 1 is eleven frames

    def test_the_maximum_particle_budget_caps_the_live_count_and_counts_drops(self):
        state, _ = solve(knobs(emit_rate=100.0, life=100000.0, max_particles=250), 10)
        self.assertEqual(len(state.arrays["id"]), 250)
        self.assertEqual(state.meta["dropped"], 1000 - 250)


class LifeTests(unittest.TestCase):
    def test_ages_and_deaths_are_exact(self):
        # One particle per frame, life five frames, one substep: born in frame f it is alive at the end of
        # frames f .. f+3 with ages 1..4 and gone at f+4 (its age reaches its life).
        state, emitter = solve(knobs(emit_rate=1.0, life=5.0), 30)
        self.assertEqual(state.arrays["age"].tolist(), [4, 3, 2, 1])
        self.assertEqual(state.arrays["id"].tolist(), [26, 27, 28, 29])
        self.assertTrue((state.arrays["life"] == 5).all())

    def test_a_particle_dies_on_the_exact_substep_its_age_reaches_its_life(self):
        # Two particles per frame in eight substeps; life 2.5 frames is exactly 20 substeps.
        params = knobs(emit_rate=2.0, life=2.5, substeps=8)
        alive_ids = {}
        for frame in range(1, 12):
            state, _ = solve(params, frame)
            alive_ids[frame] = set(state.arrays["id"].tolist())
            self.assertTrue((state.arrays["age"] < state.arrays["life"]).all())
            self.assertTrue((state.arrays["life"] == 20).all())
            self.assertTrue((state.arrays["age"] >= 1).all())
        # Births land on substeps 3 and 7 of every frame (a quarter particle per substep), so once
        # the system is warm the ages at a frame end are exactly these, and 21 never appears.
        for frame in range(4, 12):
            ages = solve(params, frame)[0].arrays["age"]
            self.assertEqual(sorted(set(ages.tolist())), [1, 5, 9, 13, 17], frame)

    def test_lifetime_variance_stays_inside_its_bounds_and_quantises_to_substeps(self):
        state, _ = solve(knobs(emit_rate=200.0, life=10.0, life_variance=0.5, substeps=4), 3)
        life = state.arrays["life"] / 4.0
        self.assertGreaterEqual(float(life.min()), 5.0 - 0.25)
        self.assertLessEqual(float(life.max()), 15.0 + 0.25)
        self.assertGreater(float(life.max() - life.min()), 3.0)


class MotionTests(unittest.TestCase):
    def test_a_point_emitter_moves_particles_along_its_direction_at_its_speed(self):
        state, _ = solve(knobs(emit_rate=1.0, life=100.0, emit_speed=2.0, emit_dir_x=1.0, emit_dir_y=0.0,
                               emit_dir_z=0.0), 4)
        # Born at frame f the particle has advanced (4 - f + 1) frames at 2 units per frame.
        np.testing.assert_allclose(state.arrays["position"][:, 0], [8.0, 6.0, 4.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(state.arrays["position"][:, 1:], 0.0, atol=1e-7)
        np.testing.assert_allclose(state.arrays["velocity"], [[2.0, 0.0, 0.0]] * 4, atol=1e-6)

    def test_births_get_sub_frame_ages_from_the_substep_they_land_on(self):
        # One particle per frame: with S substeps it is born on the last substep of frame 1, so at the
        # end of frame 2 it has moved 1/S + 1 frames at 3 units per frame.
        for substeps in (1, 2, 5, 8):
            state, _ = solve(knobs(emit_rate=1.0, life=100.0, emit_speed=3.0, substeps=substeps), 2)
            np.testing.assert_allclose(state.arrays["position"][0, 1], 3.0 * (1 + 1.0 / substeps), atol=1e-5)
            self.assertEqual(int(state.arrays["age"][0]), substeps + 1)

    def test_spread_is_a_cone_half_angle_about_the_direction(self):
        state, _ = solve(knobs(emit_rate=500.0, life=100.0, emit_speed=1.0, spread=30.0), 1)
        velocity = state.arrays["velocity"].astype(np.float64)
        angle = np.degrees(np.arccos(np.clip(velocity[:, 1] / np.linalg.norm(velocity, axis=1), -1, 1)))
        self.assertLessEqual(float(angle.max()), 30.0 + 1e-3)
        self.assertGreater(float(angle.max()), 27.0)                  # the cone is actually filled
        np.testing.assert_allclose(np.linalg.norm(velocity, axis=1), 1.0, atol=1e-5)

    def test_zero_spread_is_exactly_the_direction(self):
        state, _ = solve(knobs(emit_rate=50.0, life=100.0, emit_speed=2.0, emit_dir_y=0.0, emit_dir_z=2.0), 1)
        np.testing.assert_allclose(state.arrays["velocity"], [[0, 0, 2.0]] * 50, atol=1e-6)

    def test_speed_and_size_variance_stay_inside_their_bounds(self):
        state, _ = solve(knobs(emit_rate=400.0, life=100.0, emit_speed=2.0, speed_variance=0.25,
                               particle_size=0.4, size_variance=0.5), 1)
        speed = np.linalg.norm(state.arrays["velocity"], axis=1)
        self.assertGreaterEqual(float(speed.min()), 1.5 - 1e-4)
        self.assertLessEqual(float(speed.max()), 2.5 + 1e-4)
        self.assertGreaterEqual(float(state.arrays["size"].min()), 0.2 - 1e-6)
        self.assertLessEqual(float(state.arrays["size"].max()), 0.6 + 1e-6)

    def test_the_emitter_transform_places_and_aims_new_particles(self):
        params = knobs(emit_rate=4.0, life=100.0, emit_speed=1.0, tx=5.0, rz=90.0)
        state, _ = solve(params, 1)
        # +Y rotated 90 degrees about Z is -X; particles start at the translate and move one unit.
        np.testing.assert_allclose(state.arrays["position"], [[4.0, 0.0, 0.0]] * 4, atol=1e-5)
        np.testing.assert_allclose(state.arrays["velocity"], [[-1.0, 0.0, 0.0]] * 4, atol=1e-5)

    def test_colour_is_premultiplied_and_alpha_is_kept(self):
        state, _ = solve(knobs(emit_rate=3.0, red=1.0, green=0.5, blue=0.25, alpha=0.5), 1)
        np.testing.assert_allclose(state.arrays["color"], [[0.5, 0.25, 0.125, 0.5]] * 3, atol=1e-7)

    def test_an_animated_emission_rate_is_resolved_per_frame(self):
        curves = {"emit_rate": {"keys": [{"frame": 1, "value": 0.0}, {"frame": 5, "value": 8.0}],
                                "interpolation": "linear"}}
        state, _ = solve(knobs(life=100000.0), 5, curves=curves)
        self.assertEqual(state.meta["emitted"], 0 + 2 + 4 + 6 + 8)


class SourceTests(unittest.TestCase):
    def _geo(self, geometry, transform=Transform3D()):
        return particles.ParticleSource(geometry)

    def test_vertices_emit_only_from_the_geometry_points(self):
        cube = _cube(2.0, (1, 1, 1, 1), Transform3D())
        state, _ = solve(knobs(emit_from="vertices", emit_rate=300.0, life=100.0, emit_speed=0.0),
                         1, source=self._geo(cube))
        corners = {tuple(np.round(v, 5)) for v in cube.vertices}
        found = {tuple(np.round(v, 5)) for v in state.arrays["position"]}
        self.assertTrue(found <= corners)
        self.assertGreater(len(found), 4)

    def test_surface_emits_on_the_triangles_and_can_follow_face_normals(self):
        card = _card(4.0, 2.0, (1, 1, 1, 1), Transform3D(), None)
        state, _ = solve(knobs(emit_from="surface", emit_rate=400.0, life=100.0, emit_speed=1.0,
                               direction_from_normals=1), 1, source=self._geo(card))
        position = state.arrays["position"] - state.arrays["velocity"]     # undo the one advance
        np.testing.assert_allclose(position[:, 2], 0.0, atol=1e-5)
        self.assertLessEqual(float(np.abs(position[:, 0]).max()), 2.0 + 1e-5)
        self.assertLessEqual(float(np.abs(position[:, 1]).max()), 1.0 + 1e-5)
        self.assertGreater(float(np.abs(position[:, 0]).max()), 1.7)       # covers the whole card
        np.testing.assert_allclose(np.abs(state.arrays["velocity"][:, 2]), 1.0, atol=1e-5)

    def test_surface_emission_is_area_weighted(self):
        big = _card(4.0, 4.0, (1, 1, 1, 1), Transform3D(), None)
        small = _card(1.0, 1.0, (1, 1, 1, 1), Transform3D(position=Vec3(10, 0, 0)), None)
        merged = merge_geometry([big, small])
        state, _ = solve(knobs(emit_from="surface", emit_rate=4000.0, life=100.0, emit_speed=0.0),
                         1, source=self._geo(merged))
        share_small = float((state.arrays["position"][:, 0] > 5).mean())
        self.assertAlmostEqual(share_small, 1.0 / 17.0, delta=0.02)

    def test_volume_emits_inside_a_closed_mesh(self):
        cube = _cube(2.0, (1, 1, 1, 1), Transform3D(), None)
        state, _ = solve(knobs(emit_from="volume", emit_rate=600.0, life=100.0, emit_speed=0.0),
                         1, source=self._geo(cube))
        position = state.arrays["position"]
        self.assertEqual(len(position), 600)
        self.assertLessEqual(float(np.abs(position).max()), 1.0 + 1e-6)
        self.assertLess(float(np.abs(position).max(axis=0).min()), 1.0)
        self.assertLess(float(np.abs(position.mean(axis=0)).max()), 0.15)        # centred, not on a shell
        interior = (np.abs(position) < 0.9).all(axis=1).mean()
        self.assertGreater(float(interior), 0.6)                                 # 0.9**3 = 0.73 of the volume

    def test_no_source_falls_back_to_a_point_emitter(self):
        state, _ = solve(knobs(emit_from="surface", emit_rate=3.0, life=100.0, emit_speed=0.0), 1)
        np.testing.assert_allclose(state.arrays["position"], 0.0)


if __name__ == "__main__":
    unittest.main()
