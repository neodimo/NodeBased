"""Adaptive smoke-domain checkpoints (Lane 6, M1)."""
import unittest
import tempfile

import numpy as np

from nodebased import fluid3d, simcache


class SmokeResizeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
