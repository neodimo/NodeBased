"""Playback quick wins: the static-frame check, the lazy display cache, and the idle repaint loop."""
import unittest

import numpy as np

from nodebased.core import Dispatcher
from nodebased.playback import DisplayCache, STATIC_FRAME, is_time_dependent


def build(*commands):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": list(commands)})
    return dispatcher


def checker_grade():
    return [{"op": "create", "type": "Checker", "id": "c", "params": {}},
            {"op": "create", "type": "Grade", "id": "g", "params": {}},
            {"op": "connect", "id": "g", "input": "image", "source": "c"},
            {"op": "view", "id": "g"}]


class TimeDependenceTests(unittest.TestCase):
    def test_checker_and_constant_grade_are_static(self):
        document = build(*checker_grade()).document
        self.assertFalse(is_time_dependent(document, "c"))
        self.assertFalse(is_time_dependent(document, "g"))

    def test_animated_knob_upstream_is_time_dependent(self):
        commands = checker_grade() + [
            {"op": "set_key", "id": "c", "param": "size", "frame": 1, "value": 32},
            {"op": "set_key", "id": "c", "param": "size", "frame": 10, "value": 64}]
        document = build(*commands).document
        self.assertTrue(is_time_dependent(document, "g"))
        self.assertTrue(is_time_dependent(document, "c"))

    def test_a_curve_on_an_unrelated_node_does_not_count(self):
        commands = checker_grade() + [
            {"op": "create", "type": "Checker", "id": "other", "params": {}},
            {"op": "set_key", "id": "other", "param": "size", "frame": 1, "value": 32},
            {"op": "set_key", "id": "other", "param": "size", "frame": 10, "value": 64}]
        document = build(*commands).document
        self.assertFalse(is_time_dependent(document, "g"))
        self.assertTrue(is_time_dependent(document, "other"))

    def test_single_key_curve_is_constant(self):
        commands = checker_grade() + [
            {"op": "set_key", "id": "g", "param": "exposure", "frame": 1, "value": 1.0}]
        self.assertFalse(is_time_dependent(build(*commands).document, "g"))

    def test_expression_is_time_dependent(self):
        commands = checker_grade() + [
            {"op": "set_expression", "id": "g", "param": "exposure", "expression": "frame / 24"}]
        self.assertTrue(is_time_dependent(build(*commands).document, "g"))

    def test_read_with_a_frame_range_is_time_dependent(self):
        document = build({"op": "create", "type": "Read", "id": "r",
                          "params": {"path": "plate.####.exr"}},
                         {"op": "view", "id": "r"}).document
        self.assertTrue(is_time_dependent(document, "r"))

    def test_fluid_cache_and_time_nodes_are_time_dependent(self):
        for kind in ("FluidCache3D", "TimeOffset", "Retime", "ParticleCache3D"):
            document = build({"op": "create", "type": kind, "id": "n", "params": {}}).document
            self.assertTrue(is_time_dependent(document, "n"), kind)

    def test_static_scene3d_render_is_static(self):
        document = build(
            {"op": "create", "type": "Camera3D", "id": "cam", "params": {}},
            {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
            {"op": "create", "type": "Render3D", "id": "r", "params": {}},
            {"op": "connect", "id": "r", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "r", "input": "camera", "source": "cam"},
            {"op": "view", "id": "r"}).document
        self.assertFalse(is_time_dependent(document, "r"))

    def test_animated_camera_makes_the_render_time_dependent(self):
        document = build(
            {"op": "create", "type": "Camera3D", "id": "cam", "params": {}},
            {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
            {"op": "create", "type": "Render3D", "id": "r", "params": {}},
            {"op": "connect", "id": "r", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "r", "input": "camera", "source": "cam"},
            {"op": "set_key", "id": "cam", "param": "tx", "frame": 1, "value": 0.0},
            {"op": "set_key", "id": "cam", "param": "tx", "frame": 24, "value": 2.0}).document
        self.assertTrue(is_time_dependent(document, "r"))

    def test_missing_target_is_time_dependent(self):
        document = build(*checker_grade()).document
        self.assertTrue(is_time_dependent(document, None))
        self.assertTrue(is_time_dependent(document, "nope"))


class StaticKeyTests(unittest.TestCase):
    def test_static_target_shares_one_key_across_frames(self):
        document = build(*checker_grade()).document
        keys = {DisplayCache.key(document, "g", frame, 1) for frame in (1, 2, 50)}
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys.pop()[1], STATIC_FRAME)

    def test_animated_target_keeps_one_key_per_frame(self):
        commands = checker_grade() + [
            {"op": "set_key", "id": "c", "param": "size", "frame": 1, "value": 32},
            {"op": "set_key", "id": "c", "param": "size", "frame": 10, "value": 64}]
        document = build(*commands).document
        keys = {DisplayCache.key(document, "g", frame, 1) for frame in (1, 2, 50)}
        self.assertEqual(len(keys), 3)

    def test_a_static_entry_shows_every_frame_as_cached(self):
        document = build(*checker_grade()).document
        cache = DisplayCache(budget_bytes=10_000_000)
        key = DisplayCache.key(document, "g", 1, 1)
        cache.put(key, np.ones((4, 4, 4), np.float32), is_data=True)
        identity = DisplayCache.identity(document, "g", 1)
        self.assertEqual(cache.resident_frames(identity, range(1, 6)), {1, 2, 3, 4, 5})

    def test_an_edit_orphans_the_static_key(self):
        document = build(*checker_grade()).document
        edited = build(*checker_grade(), {"op": "set", "id": "g", "param": "exposure", "value": 1.0}).document
        self.assertNotEqual(DisplayCache.key(document, "g", 1, 1), DisplayCache.key(edited, "g", 1, 1))


class LazyCacheTests(unittest.TestCase):
    def frame(self, value=0.25):
        return np.full((8, 8, 4), value, np.float32)

    def test_lazy_put_keeps_the_producer_array_until_compacted(self):
        cache = DisplayCache(budget_bytes=10_000_000)
        frame = self.frame()
        cache.put("k", frame, lazy=True)
        self.assertIs(cache.get("k", copy=False), frame)
        self.assertTrue(cache.has_pending)
        self.assertEqual(cache.bytes, frame.nbytes)

    def test_default_get_still_copies(self):
        cache = DisplayCache(budget_bytes=10_000_000)
        frame = self.frame()
        cache.put("k", frame, lazy=True)
        copy = cache.get("k")
        self.assertIsNot(copy, frame)
        copy[...] = 0
        self.assertEqual(float(cache.get("k", copy=False)[0, 0, 0]), 0.25)

    def test_compact_rounds_to_half_and_frees_half_the_bytes(self):
        cache = DisplayCache(budget_bytes=10_000_000)
        frame = self.frame()
        cache.put("k", frame, lazy=True)
        self.assertEqual(cache.compact(), 1)
        self.assertFalse(cache.has_pending)
        self.assertEqual(cache.bytes, frame.nbytes // 2)
        restored = cache.get("k", copy=False)
        self.assertEqual(restored.dtype, np.float32)
        self.assertTrue(np.allclose(restored, frame))

    def test_budget_pressure_compacts_before_evicting(self):
        one = self.frame()
        cache = DisplayCache(budget_bytes=int(one.nbytes * 1.6))
        cache.put("a", one, lazy=True)
        cache.put("b", self.frame(0.5), lazy=True)
        self.assertEqual(len(cache), 2)
        self.assertLessEqual(cache.bytes, cache.budget)

    def test_eviction_forgets_pending_state(self):
        one = self.frame()
        cache = DisplayCache(budget_bytes=one.nbytes // 2)
        cache.put("a", one, lazy=True)   # larger than the budget: not stored
        self.assertEqual(len(cache), 0)
        self.assertFalse(cache.has_pending)

    def test_data_frames_stay_float32_and_never_pending(self):
        cache = DisplayCache(budget_bytes=10_000_000)
        cache.put("d", self.frame(1 / 3), is_data=True, lazy=True)
        self.assertFalse(cache.has_pending)
        self.assertEqual(float(cache.get("d")[0, 0, 0]), np.float32(1 / 3))

    def test_clear_resets_pending(self):
        cache = DisplayCache(budget_bytes=10_000_000)
        cache.put("k", self.frame(), lazy=True)
        cache.clear()
        self.assertFalse(cache.has_pending)
        self.assertEqual(cache.bytes, 0)


if __name__ == "__main__":
    unittest.main()


class SupersampleResolveTests(unittest.TestCase):
    def test_box_resolve_matches_the_block_mean(self):
        from nodebased import scene3d
        rng = np.random.default_rng(3)
        for samples in (2, 3, 4):
            big = rng.random((6 * samples, 5 * samples, 4), dtype=np.float32) * 4
            expected = big.reshape(6, samples, 5, samples, 4).mean(axis=(1, 3))
            got = scene3d.box_resolve(big, samples)
            self.assertEqual(got.dtype, np.float32)
            self.assertEqual(got.shape, (6, 5, 4))
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_box_resolve_does_not_touch_its_input(self):
        from nodebased import scene3d
        big = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
        before = big.copy()
        scene3d.box_resolve(big, 2)
        np.testing.assert_array_equal(big, before)

    def test_empty_scene_is_detected(self):
        from nodebased import scene3d
        self.assertTrue(scene3d.is_empty_scene(scene3d.Scene()))

    def test_scene_with_geometry_is_not_empty(self):
        from nodebased import scene3d
        scene = scene3d.Scene(geometries=(object(),))
        self.assertFalse(scene3d.is_empty_scene(scene))

    def test_empty_scene_renders_the_flat_background_with_supersampling(self):
        from nodebased import scene3d
        camera = scene3d.Camera()
        background = (0.1, 0.2, 0.3, 1.0)
        plain = scene3d.render(scene3d.Scene(), camera, 16, 9, background, samples=1)
        supersampled = scene3d.render(scene3d.Scene(), camera, 16, 9, background, samples=3)
        self.assertEqual(supersampled.shape, (9, 16, 4))
        np.testing.assert_allclose(supersampled, plain, rtol=1e-6)
