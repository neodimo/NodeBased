import unittest

from nodebased import cachecontext, scene3d
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

from tests.test_particles_nodes import make, wire
from tests.test_fluid3d_nodes import GRID


class FluidCacheContextTests(unittest.TestCase):
    def test_resolves_the_store_run_and_start_frame(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {"src_center_y": 0.4, "src_radius": 0.3}),
            sol=("FluidSolver3D", {**GRID, "start_frame": 1, "substeps": 2}),
            c=("FluidCache3D", {}))
        wire(d, "sol", "fluid", "src")
        wire(d, "c", "volume", "sol")
        evaluator = Evaluator()
        context = cachecontext.cache_context(evaluator, d.document, "c", frame=1)
        self.assertIsNotNone(context)
        store, run, start_frame, substeps = context
        self.assertEqual(start_frame, 1)
        self.assertEqual(substeps, 2)
        self.assertTrue(run)
        # The volume this same FluidCache3D node evaluates to is inspectable too.
        volume = cachecontext.volume_for_node(evaluator, d.document, "c", frame=1)
        self.assertIsInstance(volume, scene3d.Volume)

    def test_disabled_cache_node_resolves_to_none(self):
        d = Dispatcher()
        make(d, src=("FluidSource3D", {}), sol=("FluidSolver3D", {**GRID}), c=("FluidCache3D", {}))
        wire(d, "sol", "fluid", "src")
        wire(d, "c", "volume", "sol")
        d.execute({"op": "disable", "id": "c", "value": True})
        evaluator = Evaluator()
        self.assertIsNone(cachecontext.cache_context(evaluator, d.document, "c", frame=1))

    def test_missing_node_resolves_to_none(self):
        d = Dispatcher()
        evaluator = Evaluator()
        self.assertIsNone(cachecontext.cache_context(evaluator, d.document, "nope", frame=1))
        self.assertIsNone(cachecontext.volume_for_node(evaluator, d.document, "nope", frame=1))


class ParticleCacheContextTests(unittest.TestCase):
    def test_resolves_the_store_and_run(self):
        d = Dispatcher()
        make(d, e=("ParticleEmitter3D", {"emit_rate": 10.0, "life": 100.0, "start_frame": 1}),
            c=("ParticleCache3D", {}))
        wire(d, "c", "particles", "e")
        evaluator = Evaluator()
        context = cachecontext.cache_context(evaluator, d.document, "c", frame=1)
        self.assertIsNotNone(context)
        store, run, start_frame, substeps = context
        self.assertEqual(start_frame, 1)
        self.assertTrue(run)


if __name__ == "__main__":
    unittest.main()
