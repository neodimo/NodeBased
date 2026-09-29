"""Lane L6 step E, part 3: the liquid surface (nodebased/liquid_surface.py, FluidSurface3D) and foam (FluidFoam3D).

The level set is a signed distance that is negative inside; the mesh is closed, watertight, wound and shaded outward,
its volume tracks the particles', and it stays inside the domain; the surface knobs change it the way they say;
foam appears in the splash and not in a liquid at rest.
"""
import unittest

import numpy as np

from nodebased import flip3d, liquid_surface as ls, scene3d
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_liquid_nodes import GRID, at
from tests.test_particles_nodes import make, set_, wire

CELL = 0.125


def blob(radius, centre=(0.0, 1.0, 0.0), spacing=CELL / 2, seed=0):
    """Particles filling a ball, on a jittered lattice of `spacing`."""
    rng = np.random.default_rng(seed)
    axis = np.arange(-radius, radius + spacing, spacing)
    pts = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    pts = pts + (rng.random(pts.shape) - 0.5) * spacing * 0.5
    return (pts[np.linalg.norm(pts, axis=1) <= radius] + np.array(centre)), spacing


def mesh_of(positions, spacing, resolution=1, smoothing=1, radius=None):
    radius = radius or flip3d.RADIUS_SPACINGS * spacing
    (v, t, n), phi = ls.surface_from_particles(positions, (-1.0, 0.0, -1.0), CELL, (16, 16, 16), radius,
                                               max(flip3d.SUPPORT_SPACINGS * spacing, 2.2 * radius), resolution, smoothing)
    return v, t, n, phi


class LevelSetTests(unittest.TestCase):
    def test_one_particle_is_a_sphere_and_the_field_is_negative_inside(self):
        pos = np.array([[0.0, 1.0, 0.0]])
        phi = ls.level_set(pos, (-1.0, 0.0, -1.0), CELL, (16, 16, 16), 0.2, 0.5)
        centre = np.array([8, 8, 8])                                  # cell centre (0.0625, 1.0625, 0.0625) is the nearest
        c = (centre + 0.5) * CELL + np.array((-1.0, 0.0, -1.0))
        self.assertAlmostEqual(float(phi[tuple(centre)]), float(np.linalg.norm(c - pos[0]) - 0.2), places=5)
        self.assertLess(float(phi[tuple(centre)]), 0.0)
        self.assertGreater(float(phi[0, 0, 0]), 0.0)                  # far away: positive
        self.assertEqual(phi.dtype, np.float32)

    def test_a_ball_of_particles_has_a_negative_core_and_a_positive_outside(self):
        pos, spacing = blob(0.4)
        phi = ls.level_set(pos, (-1.0, 0.0, -1.0), CELL, (16, 16, 16), 0.8 * spacing, 3 * spacing)
        self.assertLess(float(phi[8, 8, 8]), 0.0)
        self.assertGreater(float(phi[1, 1, 1]), 0.0)
        self.assertLess(float(phi.min()), -0.5 * spacing)

    def test_curvature_of_a_sphere_is_two_over_the_radius(self):
        x = (np.arange(24) + 0.5) * 0.05 - 0.6
        phi = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2) - 0.4
        kappa = ls.curvature(phi, 0.05)
        i = np.argmin(np.abs(x - 0.4 * 0.0))                          # sample the shell along +x
        shell = kappa[np.abs(phi) < 0.03]
        self.assertAlmostEqual(float(np.median(shell)), 2.0 / 0.4, delta=0.6)


class MeshTests(unittest.TestCase):
    @staticmethod
    def components(triangles):
        parent = np.arange(int(triangles.max()) + 1)

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for triangle in triangles:
            root = find(int(triangle[0]))
            for vertex in triangle[1:]:
                parent[find(int(vertex))] = root
        return len({find(int(vertex)) for vertex in np.unique(triangles)})

    def test_thin_splash_bridge_survives_only_with_preservation_enabled(self):
        voxel, spacing, radius = 0.025, 0.1, 0.05
        axis = (np.arange(48) + 0.5) * voxel - 0.6
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        left = np.sqrt((x + 0.07) ** 2 + y ** 2 + z ** 2) - radius
        right = np.sqrt((x - 0.07) ** 2 + y ** 2 + z ** 2) - radius
        beads = np.minimum(left, right).astype(np.float32)
        plain = ls.marching_tetrahedra(beads, (-0.6, -0.6, -0.6), voxel)
        preserved = ls.marching_tetrahedra(flip3d.preserve_thin_sheet_gaps(beads, spacing),
                                           (-0.6, -0.6, -0.6), voxel)
        self.assertEqual(self.components(plain[1]), 2)
        self.assertEqual(self.components(preserved[1]), 1)
        self.assertTrue(ls.is_closed(preserved[1]))
        self.assertGreater(ls.signed_volume(preserved[0], preserved[1]), 0.0)

    def test_the_mesh_is_closed_outward_and_close_to_the_particle_volume(self):
        pos, spacing = blob(0.45)
        v, t, n, _ = mesh_of(pos, spacing)
        self.assertTrue(ls.is_closed(t))
        volume = ls.signed_volume(v, t)
        self.assertGreater(volume, 0.0)                                # outward winding
        particle_volume = len(pos) * spacing ** 3
        self.assertGreater(volume, 0.8 * particle_volume)
        self.assertLess(volume, 1.25 * particle_volume)

    def test_normals_are_unit_outward_and_agree_with_the_winding(self):
        pos, spacing = blob(0.45)
        v, t, n, _ = mesh_of(pos, spacing)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-3)
        radial = v - v.mean(axis=0)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        self.assertGreater(float(np.median((radial * n).sum(axis=1))), 0.9)
        self.assertGreater(float(((radial * n).sum(axis=1)).min()), 0.0)
        tri = v[t]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        agree = (face * n[t].sum(axis=1)).sum(axis=1)
        self.assertTrue(np.all(agree >= 0.0))

    def test_two_separate_drops_make_two_closed_shells(self):
        a, spacing = blob(0.25, (-0.5, 1.0, 0.0))
        b, _ = blob(0.25, (0.5, 1.0, 0.0), seed=1)
        v, t, n, _ = mesh_of(np.concatenate((a, b)), spacing)
        self.assertTrue(ls.is_closed(t))
        self.assertGreater(ls.signed_volume(v, t), 0.0)
        left = np.unique(t[v[t][:, :, 0].max(axis=1) < 0.0])
        self.assertGreater(len(left), 0)
        self.assertLess(float(v[left][:, 0].max()), 0.0)
        self.assertGreater(float(v[np.setdiff1d(np.arange(len(v)), left)][:, 0].min()), 0.0)

    def test_liquid_against_a_wall_is_capped_at_the_wall_and_stays_in_the_domain(self):
        rng = np.random.default_rng(2)
        spacing = CELL / 2
        axis = np.arange(spacing / 2, 0.5, spacing)
        pts = np.stack(np.meshgrid(axis - 1.0, axis, axis - 1.0, indexing="ij"), axis=-1).reshape(-1, 3)   # a corner block
        pts = pts + (rng.random(pts.shape) - 0.5) * 0.02
        pts = np.clip(pts, (-1.0 + 1e-3, 1e-3, -1.0 + 1e-3), None)
        v, t, n, _ = mesh_of(pts, spacing)
        self.assertTrue(ls.is_closed(t))
        self.assertGreater(ls.signed_volume(v, t), 0.0)
        self.assertGreaterEqual(float(v.min(axis=0).min()), -1.0 - 1e-6)
        self.assertGreaterEqual(float(v[:, 1].min()), -1e-6)
        self.assertLessEqual(float(v.max(axis=0).max()), 2.0 + 1e-6)
        self.assertAlmostEqual(float(v[:, 0].min()), -1.0, places=5)   # the block touches the wall: the mesh reaches it
        self.assertAlmostEqual(float(v[:, 1].min()), 0.0, places=5)

    def test_empty_and_far_particles_make_no_mesh(self):
        v, t, n = ls.marching_tetrahedra(np.full((6, 6, 6), 1.0, np.float32), (0, 0, 0), 0.1)
        self.assertEqual((len(v), len(t), len(n)), (0, 0, 0))
        phi = ls.level_set(np.zeros((0, 3)), (0, 0, 0), 0.1, (6, 6, 6), 0.05, 0.2)
        self.assertGreater(float(phi.min()), 0.0)

    def test_resolution_smoothing_and_radius_do_what_they_say(self):
        pos, spacing = blob(0.45)
        v1, t1, _, _ = mesh_of(pos, spacing, resolution=1)
        v2, t2, _, _ = mesh_of(pos, spacing, resolution=2)
        self.assertGreater(len(t2), 2.5 * len(t1))                     # a finer grid, more triangles
        small = ls.signed_volume(*mesh_of(pos, spacing, radius=0.5 * spacing)[:2])
        big = ls.signed_volume(*mesh_of(pos, spacing, radius=1.2 * spacing)[:2])
        self.assertGreater(big, small)
        rough = mesh_of(pos, spacing, smoothing=0)
        smooth = mesh_of(pos, spacing, smoothing=4)
        self.assertTrue(ls.is_closed(smooth[1]))                         # smoothing keeps the topology
        bumps = lambda v: float(np.linalg.norm(v - v.mean(axis=0), axis=1).std())     # a ball: spread of the radius
        self.assertLess(bumps(smooth[0]), 0.9 * bumps(rough[0]))          # lumps are ironed out
        self.assertAlmostEqual(ls.signed_volume(*smooth[:2]) / ls.signed_volume(*rough[:2]), 1.0, delta=0.04)


def tank(**foam):
    """A pool with a ball above it, one liquid source over a merged pool and ball, dropped into a 16 cubed grid."""
    d = Dispatcher()
    make(d, cube=("Cube3D", {"cube_size": 1.9, "sy": 0.3, "ty": 0.35}), ball=("Sphere3D", {"sphere_radius": 0.3, "ty": 1.3}),
         m=("MergeGeo3D", {}), src=("FluidSource3D", {"fluid_type": "liquid", "fluid_emit_from": "volume", "end_frame": 1}),
         sol=("FluidLiquidSolver3D", dict(GRID)), foam=("FluidFoam3D", foam), c=("ParticleCache3D", {}),
         sf=("FluidSurface3D", {}))
    wire(d, "m", "geo0", "cube")
    wire(d, "m", "geo1", "ball")
    wire(d, "src", "geo", "m")
    wire(d, "sol", "fluid", "src")
    wire(d, "foam", "particles", "sol")
    wire(d, "c", "particles", "sol")
    wire(d, "sf", "particles", "sol")
    return d


class NodeTests(unittest.TestCase):
    def test_surface_node_outputs_a_closed_geometry_with_normals(self):
        d = tank()
        g = at(Evaluator(), d, "sf", 3)
        self.assertIsInstance(g, scene3d.Geometry)
        self.assertEqual(g.normals.shape, g.vertices.shape)
        self.assertTrue(ls.is_closed(g.triangles))
        self.assertGreater(ls.signed_volume(g.vertices, g.triangles), 0.0)
        inst = at(Evaluator(), d, "sol", 3)
        self.assertGreater(ls.signed_volume(g.vertices, g.triangles), 0.7 * len(inst) / 8 * CELL ** 3)
        self.assertLess(ls.signed_volume(g.vertices, g.triangles), 1.2 * len(inst) / 8 * CELL ** 3)

    def test_the_knobs_reach_the_mesh(self):
        ev = Evaluator()
        d = tank()
        base = at(ev, d, "sf", 3)
        set_(d, "sf", surface_resolution=2)
        self.assertGreater(len(at(ev, d, "sf", 3).triangles), 2 * len(base.triangles))
        smooth = at(ev, (set_(d, "sf", surface_resolution=1, smoothing=4) or d), "sf", 3)
        self.assertFalse(np.array_equal(smooth.vertices, base.vertices))
        set_(d, "sf", smoothing=1, particle_radius=0.09)
        self.assertGreater(ls.signed_volume(*[at(ev, d, "sf", 3).vertices, at(ev, d, "sf", 3).triangles]),
                           ls.signed_volume(base.vertices, base.triangles))

    def test_detail_ratio_refines_and_keeps_the_surface_closed(self):
        d = tank()
        ev = Evaluator()
        base = at(ev, d, "sf", 8)
        set_(d, "sf", detail_ratio=2)
        refined = at(ev, d, "sf", 8)
        self.assertGreater(len(refined.triangles), 2 * len(base.triangles))
        self.assertTrue(ls.is_closed(refined.triangles))
        self.assertGreater(ls.signed_volume(refined.vertices, refined.triangles), 0.0)

    def test_thin_sheet_knob_expands_the_level_set_and_keeps_mesh_closed(self):
        d = tank()
        ev = Evaluator()
        instance = at(ev, d, "sol", 8)
        params = d.document["nodes"]["sf"]["params"]
        plain, _ = flip3d.surface_level_set(instance, params)
        set_(d, "sf", thin_sheet_preservation=1)
        kept, _ = flip3d.surface_level_set(instance, d.document["nodes"]["sf"]["params"])
        np.testing.assert_allclose(plain - kept, 0.5 * instance.stream.spacing, atol=1e-7)
        mesh = at(ev, d, "sf", 8)
        self.assertTrue(ls.is_closed(mesh.triangles))
        self.assertGreater(ls.signed_volume(mesh.vertices, mesh.triangles), 0.0)

    def test_all_surface_resolutions_are_closed_and_outward(self):
        d = tank()
        ev = Evaluator()
        for resolution in range(1, 5):
            set_(d, "sf", surface_resolution=resolution, detail_ratio=1)
            mesh = at(ev, d, "sf", 8)
            self.assertTrue(ls.is_closed(mesh.triangles), resolution)
            self.assertGreater(ls.signed_volume(mesh.vertices, mesh.triangles), 0.0, resolution)

    def test_temporal_sdf_blend_accepts_neighboring_frames_and_keeps_mesh_closed(self):
        d = tank()
        ev = Evaluator()
        set_(d, "sf", temporal_smoothing=1)
        g = at(ev, d, "sf", 8)
        self.assertTrue(ls.is_closed(g.triangles))
        self.assertGreater(ls.signed_volume(g.vertices, g.triangles), 0.0)
        self.assertEqual(g.velocities.shape, g.vertices.shape)

    def test_still_tank_temporal_smoothing_reduces_surface_change_to_near_zero(self):
        from dataclasses import replace
        d = tank()
        set_(d, "sol", liquid_gravity=0.0)
        ev = Evaluator()
        frames = [at(ev, d, "sol", f) for f in (3, 4, 5, 6, 7)]
        # Model the frame-to-frame particle noise that the filter targets while keeping the solved tank still.
        rng = np.random.default_rng(99)
        noisy = [replace(item, positions=item.positions + rng.normal(0, 0.01, item.positions.shape).astype(np.float32))
                 for item in frames]
        raw = [flip3d.surface_level_set(item, {"surface_resolution": 1, "particle_radius": 0.0})[0]
               for item in noisy]
        smoothed = []
        for index, item in enumerate(noisy):
            neighbors = noisy[max(0, index - 1):min(len(noisy), index + 2)]
            smoothed.append(flip3d.surface_level_set(item, {"surface_resolution": 1, "particle_radius": 0.0},
                                                       neighbors)[0])
        raw_change = float(np.mean(np.abs(raw[3] - raw[2])))
        smoothed_change = float(np.mean(np.abs(smoothed[3] - smoothed[2])))
        self.assertLess(smoothed_change, 0.5 * raw_change)
        self.assertLess(smoothed_change, 0.002)

    def test_foam_appears_only_in_the_splash(self):
        d = tank()
        ev = Evaluator()
        counts = {frame: len(at(ev, d, "foam", frame)) for frame in (1, 2, 6, 8, 10, 12, 14, 80)}
        self.assertEqual(counts[1], 0)                                 # a liquid at rest has none
        peak = max(counts[frame] for frame in (6, 8, 10, 12, 14))
        self.assertGreater(peak, 20)                                   # the impact throws some off
        self.assertLess(counts[80], 0.2 * peak)                        # and it is gone once the liquid has settled
        foam = at(ev, d, "foam", 10)
        self.assertIsInstance(foam, scene3d.ParticleInstance)
        self.assertLess(len(foam), 0.05 * len(at(ev, d, "sol", 10)))   # a few particles, not the liquid
        self.assertIsNone(foam.stream)                                  # no stream: a cache after it passes it on
        self.assertTrue(np.all(foam.colors == 1.0))

    def test_foam_thresholds_gate_it(self):
        ev = Evaluator()
        loose = len(at(ev, tank(foam_speed=0.2, foam_curvature=0.5), "foam", 10))
        default = len(at(ev, tank(), "foam", 10))
        strict = len(at(ev, tank(foam_speed=50.0), "foam", 10))
        self.assertGreater(loose, default)
        self.assertEqual(strict, 0)

    def test_foam_size_scales_the_particles(self):
        ev = Evaluator()
        small = at(ev, tank(foam_size=0.25), "foam", 10)
        big = at(ev, tank(foam_size=1.0), "foam", 10)
        np.testing.assert_allclose(big.sizes, 4.0 * small.sizes, rtol=1e-5)

    def test_a_non_liquid_input_gives_an_empty_geometry_and_no_foam(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {}), sf=("FluidSurface3D", {}), foam=("FluidFoam3D", {}))
        wire(d, "sf", "particles", "e")
        wire(d, "foam", "particles", "e")
        ev = Evaluator()
        self.assertEqual(len(at(ev, d, "sf", 3).vertices), 0)
        self.assertEqual(len(at(ev, d, "foam", 3)), 0)


if __name__ == "__main__":
    unittest.main()
