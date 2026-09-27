"""Particle artist tools 3: the shelf tools acting on selected 3D nodes."""
import itertools
import unittest

from nodebased import particletools
from nodebased.core import Dispatcher, validate


def _sphere_doc():
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "create", "id": "sphere", "type": "Sphere3D", "params": {}})
    return dispatcher


def _id_factory():
    counter = itertools.count()
    return lambda: f"new{next(counter)}"


class EmitParticlesFromSelectedTests(unittest.TestCase):
    def test_wires_an_emitter_reading_the_selected_spheres_surface(self):
        dispatcher = _sphere_doc()
        ops = particletools.emit_particles_from_selected(dispatcher.document["nodes"], ["sphere"], _id_factory())
        dispatcher.execute({"op": "batch", "commands": ops})
        validate(dispatcher.document)
        nodes = dispatcher.document["nodes"]
        emitters = [n for n in nodes.values() if n["type"] == "ParticleEmitter3D"]
        self.assertEqual(len(emitters), 1)
        self.assertEqual(emitters[0]["params"]["emit_from"], "surface")
        self.assertEqual(emitters[0]["inputs"]["geo"], "sphere")

    def test_one_emitter_per_selected_node(self):
        dispatcher = _sphere_doc()
        dispatcher.execute({"op": "create", "id": "sphere2", "type": "Sphere3D", "params": {}})
        ops = particletools.emit_particles_from_selected(dispatcher.document["nodes"], ["sphere", "sphere2"],
                                                          _id_factory())
        dispatcher.execute({"op": "batch", "commands": ops})
        emitters = [n for n in dispatcher.document["nodes"].values() if n["type"] == "ParticleEmitter3D"]
        self.assertEqual(len(emitters), 2)
        self.assertEqual({e["inputs"]["geo"] for e in emitters}, {"sphere", "sphere2"})


class MakeSelectedColliderTests(unittest.TestCase):
    def test_wires_a_bounce_node_against_the_selected_sphere(self):
        dispatcher = _sphere_doc()
        ops = particletools.make_selected_a_particle_collider(dispatcher.document["nodes"], ["sphere"], _id_factory())
        dispatcher.execute({"op": "batch", "commands": ops})
        validate(dispatcher.document)
        bounces = [n for n in dispatcher.document["nodes"].values() if n["type"] == "ParticleBounce3D"]
        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0]["inputs"]["geometry"], "sphere")
        self.assertEqual(bounces[0]["params"].get("animated", 0), 0)

    def test_animated_flag_set_when_node_is_in_animated_ids(self):
        dispatcher = _sphere_doc()
        ops = particletools.make_selected_a_particle_collider(
            dispatcher.document["nodes"], ["sphere"], _id_factory(), animated_ids=frozenset({"sphere"}))
        dispatcher.execute({"op": "batch", "commands": ops})
        bounces = [n for n in dispatcher.document["nodes"].values() if n["type"] == "ParticleBounce3D"]
        self.assertEqual(bounces[0]["params"]["animated"], 1)


class ScatterInstancesTests(unittest.TestCase):
    def test_wires_instance3d_over_the_selected_sphere_with_a_default_mesh(self):
        dispatcher = _sphere_doc()
        ops = particletools.scatter_instances_on_selected(dispatcher.document["nodes"], ["sphere"], _id_factory())
        dispatcher.execute({"op": "batch", "commands": ops})
        validate(dispatcher.document)
        nodes = dispatcher.document["nodes"]
        instances = [n for n in nodes.values() if n["type"] == "Instance3D"]
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["inputs"]["points"], "sphere")
        instance_source = instances[0]["inputs"]["instance"]
        self.assertEqual(nodes[instance_source]["type"], "Sphere3D")
        self.assertNotEqual(instance_source, "sphere")


class AddForceToSelectedParticlesTests(unittest.TestCase):
    def _emitter_and_cache_doc(self):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "create", "id": "emit", "type": "ParticleEmitter3D", "params": {}})
        dispatcher.execute({"op": "create", "id": "cache", "type": "ParticleCache3D", "params": {}})
        dispatcher.execute({"op": "connect", "id": "cache", "input": "particles", "source": "emit"})
        return dispatcher

    def test_rejects_a_kind_that_is_not_a_force(self):
        with self.assertRaises(ValueError):
            particletools.add_force_to_selected_particles({}, ["x"], "Sphere3D", _id_factory())

    def test_splices_the_force_between_the_emitter_and_its_consumer(self):
        dispatcher = self._emitter_and_cache_doc()
        ops = particletools.add_force_to_selected_particles(
            dispatcher.document["nodes"], ["emit"], "ParticleWind3D", _id_factory())
        dispatcher.execute({"op": "batch", "commands": ops})
        validate(dispatcher.document)
        nodes = dispatcher.document["nodes"]
        winds = [key for key, n in nodes.items() if n["type"] == "ParticleWind3D"]
        self.assertEqual(len(winds), 1)
        wind_id = winds[0]
        self.assertEqual(nodes[wind_id]["inputs"]["particles"], "emit")
        self.assertEqual(nodes["cache"]["inputs"]["particles"], wind_id)

    def test_each_force_kind_is_accepted(self):
        for kind in particletools.FORCE_KINDS:
            with self.subTest(kind=kind):
                dispatcher = self._emitter_and_cache_doc()
                ops = particletools.add_force_to_selected_particles(
                    dispatcher.document["nodes"], ["emit"], kind, _id_factory())
                dispatcher.execute({"op": "batch", "commands": ops})
                self.assertTrue(any(n["type"] == kind for n in dispatcher.document["nodes"].values()))


if __name__ == "__main__":
    unittest.main()
