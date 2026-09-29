"""Lane 6 K1: GPU collision path compared with the CPU reference."""
import unittest

import numpy as np

from nodebased import particles, particlegpu
from nodebased.core import SPECS


@unittest.skipUnless(particlegpu.available(), "wgpu adapter unavailable")
class ParticleCollisionGpuTests(unittest.TestCase):
    def test_gpu_contacts_match_cpu_overlap_and_are_seed_repeatable(self):
        rng = np.random.default_rng(31)
        ids = np.arange(1024, dtype=np.int64)
        # Dense, seeded packing exercises grid collisions and coincident-particle handling.
        position = rng.normal(0.0, 0.16, (len(ids), 3)).astype(np.float32)
        velocity = rng.normal(0.0, 0.1, (len(ids), 3)).astype(np.float32)
        radius = np.full(len(ids), 0.08, np.float32)
        gpu_a = particlegpu.resolve(ids, position, velocity, radius, iterations=4,
                                    restitution=0.3, friction=0.4, sleep_threshold=0.0)
        gpu_b = particlegpu.resolve(ids, position, velocity, radius, iterations=4,
                                    restitution=0.3, friction=0.4, sleep_threshold=0.0)
        np.testing.assert_array_equal(gpu_a[0], gpu_b[0])
        np.testing.assert_array_equal(gpu_a[1], gpu_b[1])
        force = particles.ParticleSelfCollider("ParticleCollide3D", {
            "from_frame": -100, "to_frame": 100, "probability": 1.0, "seed": 0,
            "collide_radius": 0.08, "radius_from_size": 0, "iterations": 4,
            "restitution": 0.3, "friction": 0.4, "sleep_threshold": 0.0,
        })
        cpu_pos, _ = particles.resolve_particle_collisions(force, 1, ids, position, velocity,
                                                            np.full(len(ids), 0.16, np.float32))
        gpu_pairs_a, gpu_pairs_b = particles._grid_pairs(gpu_a[0], 0.16)
        cpu_pairs_a, cpu_pairs_b = particles._grid_pairs(cpu_pos, 0.16)
        gpu_overlap = int((np.linalg.norm(gpu_a[0][gpu_pairs_a] - gpu_a[0][gpu_pairs_b], axis=1) < 0.16).sum())
        cpu_overlap = int((np.linalg.norm(cpu_pos[cpu_pairs_a] - cpu_pos[cpu_pairs_b], axis=1) < 0.16).sum())
        self.assertLessEqual(abs(gpu_overlap - cpu_overlap), max(1, int(cpu_overlap * 0.03)))
        gpu_height = float(np.ptp(gpu_a[0][:, 1]))
        cpu_height = float(np.ptp(cpu_pos[:, 1]))
        self.assertLessEqual(abs(gpu_height - cpu_height), max(1e-5, cpu_height * 0.03))


if __name__ == "__main__":
    unittest.main()
