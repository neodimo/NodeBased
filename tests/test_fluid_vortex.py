"""Fluids 18 B: a vortex ring in still air, no emitter, through the 3D smoke solver (docs/FLUIDS_SPIKE.md, "Fluids 18 B").

A correct solver keeps the same physical vortex alive at least as long on a finer grid. The ring has a 0.3 m radius and a 0.1 m core in a
2 m closed box; the peak vorticity and kinetic energy come from the face velocities in world units (`nodebased.fluid_diagnostics`). The bounds
below are the measured numbers with room for the adapters' rounding, and they hold on the CPU reference and on every GPU adapter.
"""
from __future__ import annotations

import unittest

import numpy as np

from nodebased import fluid3d, fluid_diagnostics as diag

try:
    from nodebased import fluid_gpu_solver as fgs
    HAVE_GPU = fgs.available()
except Exception:        # pragma: no cover - no wgpu on this machine
    HAVE_GPU = False


def peak_left(rows):
    return rows[-1]["peak_vorticity"] / rows[0]["peak_vorticity"]


def energy_left(rows):
    return rows[-1]["energy"] / rows[0]["energy"]


class RingSeedTests(unittest.TestCase):
    def test_seed_is_divergence_free_and_the_size_asked_for(self):
        for cells in (24, 40):
            voxel = diag.BOX / cells
            arrays = diag.ring_vortex_arrays((cells,) * 3, voxel, diag.FPS, (1.0, 0.8, 1.0), 0.3, 0.1, 0.5)
            self.assertLess(diag.max_divergence(arrays, voxel, diag.FPS), 1e-3, "the staggered curl is divergence-free to round-off")
            velocity = diag.cell_centred_velocity(arrays, voxel, diag.FPS)
            self.assertAlmostEqual(float(np.sqrt((velocity ** 2).sum(-1)).max()), 0.5, places=3)
            for name, dims in (("u", (cells + 1, cells, cells)), ("v", (cells, cells + 1, cells)), ("w", (cells, cells, cells + 1))):
                self.assertEqual(arrays[name].shape, dims)
            # nothing reaches the closed walls
            self.assertLess(float(np.abs(arrays["u"][0]).max() + np.abs(arrays["u"][-1]).max()), 1e-6)

    def test_the_ring_travels_along_the_requested_direction(self):
        voxel = diag.BOX / 32
        up = diag.ring_vortex_arrays((32,) * 3, voxel, diag.FPS, (1.0, 0.8, 1.0), 0.3, 0.1, 0.5, direction=1.0)
        down = diag.ring_vortex_arrays((32,) * 3, voxel, diag.FPS, (1.0, 0.8, 1.0), 0.3, 0.1, 0.5, direction=-1.0)
        centre = 16
        self.assertGreater(float(up["v"][centre, 12:20, centre].mean()), 0.0)
        self.assertLess(float(down["v"][centre, 12:20, centre].mean()), 0.0)


class VortexDecayCpuTests(unittest.TestCase):
    """The CPU reference solver (no confinement, the node's MacCormack advection, 24 frames per second, 48 frames)."""

    @classmethod
    def setUpClass(cls):
        cls.rows = {cells: diag.follow_ring_vortex(cells, frames=48, backend="cpu")[0] for cells in (32, 48)}

    def test_a_finer_grid_keeps_the_vortex_longer(self):
        coarse, fine = self.rows[32], self.rows[48]
        self.assertGreater(peak_left(fine), peak_left(coarse), "peak vorticity left after 48 frames must grow with resolution")
        self.assertGreater(energy_left(fine), energy_left(coarse))

    def test_decay_bounds(self):
        # measured: 32 cells 16.5 % of the peak vorticity and 8.5 % of the energy left, 48 cells 20.0 % and 12.8 %
        self.assertGreater(peak_left(self.rows[32]), 0.12)
        self.assertGreater(peak_left(self.rows[48]), 0.16)
        self.assertGreater(energy_left(self.rows[48]), 0.09)
        self.assertLess(peak_left(self.rows[48]), 1.0)

    def test_energy_never_grows_without_confinement(self):
        for cells, rows in self.rows.items():
            energy = [r["energy"] for r in rows]
            for before, after in zip(energy, energy[1:]):
                self.assertLessEqual(after, before * 1.001, f"{cells} cells gained kinetic energy in still air")

    def test_projection_leaves_less_than_the_tolerance(self):
        for cells, rows in self.rows.items():
            # tolerance 1e-3 cells per frame per cell is 0.024 per second at 24 frames per second
            self.assertLess(max(r["max_divergence"] for r in rows[1:]), 0.03, f"{cells} cells")


@unittest.skipUnless(HAVE_GPU, "no wgpu compute adapter")
class VortexDecayGpuTests(unittest.TestCase):
    """The resident GPU solver, the one the fire shot bakes on at 128 cells."""

    @classmethod
    def setUpClass(cls):
        cls.rows = {cells: diag.follow_ring_vortex(cells, frames=48, backend="gpu")[0] for cells in (48, 64)}

    def test_a_finer_grid_keeps_the_vortex_longer(self):
        self.assertGreater(peak_left(self.rows[64]), peak_left(self.rows[48]))
        self.assertGreater(energy_left(self.rows[64]), energy_left(self.rows[48]))

    def test_decay_bounds(self):
        # the CPU reference keeps 20.0 % (48 cells) and 23.8 % (64 cells) of the peak vorticity
        self.assertGreater(peak_left(self.rows[48]), 0.15)
        self.assertGreater(peak_left(self.rows[64]), 0.18)
        self.assertLess(peak_left(self.rows[64]), 0.4)

    def test_agrees_with_the_cpu_reference(self):
        cpu, _ = diag.follow_ring_vortex(48, frames=24, backend="cpu")
        gpu, _ = diag.follow_ring_vortex(48, frames=24, backend="gpu")
        self.assertAlmostEqual(peak_left(gpu), peak_left(cpu), delta=0.03)
        self.assertAlmostEqual(energy_left(gpu), energy_left(cpu), delta=0.03)


if __name__ == "__main__":
    unittest.main()
