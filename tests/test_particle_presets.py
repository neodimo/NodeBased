"""Particle artist tools 2: the shipped particle presets. Every preset must validate, solve ten
frames and end up with a non-empty particle count through its own ParticleCache3D."""
import unittest
from pathlib import Path

from nodebased import presets, scene3d
from nodebased.core import Dispatcher, validate
from nodebased.imaging import Evaluator

_SHIPPED_PARTICLES_DIR = presets._SHIPPED_ROOT / "particles"


class _Pos:
    def x(self):
        return 0

    def y(self):
        return 0


def _load_particle_presets():
    return presets.load_all(user_directory=Path("/nonexistent"), shipped_directory=_SHIPPED_PARTICLES_DIR)


class ShippedParticlePresetTests(unittest.TestCase):
    def setUp(self):
        self.presets, errors = _load_particle_presets()
        self.assertFalse(errors, errors)
        expected = {"Sparks", "Debris burst", "Dust puff", "Rain", "Snow", "Leaves in wind",
                    "Fountain", "Magic trail"}
        self.assertEqual({p.name for p in self.presets}, expected)

    def _build(self, preset):
        ops = presets.build_ops(preset, [], {}, _Pos())
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "batch", "commands": ops})
        validate(dispatcher.document)
        cache_ids = [op["id"] for op in ops if op["op"] == "create" and op["type"] == "ParticleCache3D"]
        self.assertEqual(len(cache_ids), 1, f"{preset.name} must have exactly one ParticleCache3D")
        return dispatcher, cache_ids[0]

    def test_every_preset_validates_and_solves_with_particles(self):
        for preset in self.presets:
            with self.subTest(preset=preset.name):
                dispatcher, cache_id = self._build(preset)
                evaluator = Evaluator()
                value = evaluator.evaluate_raster(dispatcher.document, cache_id, frame=10, typed=True)
                self.assertIsInstance(value, scene3d.ParticleInstance)
                self.assertGreater(len(value), 0, f"{preset.name} produced no particles by frame 10")

    def test_sparks_uses_instance3d_for_streaks(self):
        preset = next(p for p in self.presets if p.name == "Sparks")
        ops = presets.build_ops(preset, [], {}, _Pos())
        types = {op["id"]: op["type"] for op in ops if op["op"] == "create"}
        self.assertIn("Instance3D", types.values())

    def test_rain_kills_particles_on_collision(self):
        preset = next(p for p in self.presets if p.name == "Rain")
        ops = presets.build_ops(preset, [], {}, _Pos())
        bounce = next(op for op in ops if op["op"] == "create" and op["type"] == "ParticleBounce3D")
        self.assertEqual(bounce["params"]["kill_on_collision"], 1)

    def test_magic_trail_animates_the_emitters_own_transform(self):
        preset = next(p for p in self.presets if p.name == "Magic trail")
        ops = presets.build_ops(preset, [], {}, _Pos())
        set_key_ops = [op for op in ops if op["op"] == "set_key"]
        self.assertEqual(len(set_key_ops), 2)
        self.assertTrue(all(op["param"] == "tx" for op in set_key_ops))


if __name__ == "__main__":
    unittest.main()
