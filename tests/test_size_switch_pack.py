"""Lane 4, step C (10/25): the scene is packed once and drawn at both sizes.

A play pressed after the card rests draws its first frames at half size and then returns to the chosen size (`nodebased/cardclock.py`).
The Evaluator makes the half-size scene and the full-size scene as two sets of array objects with the same content (a tier is part of
every node's digest), and the cache keys of `gpu3d` told read-only arrays apart by identity, so the first full-size frame packed the
shadow triangles and built the resident scene again (18 + 10 ms for the furnished room, docs/PERFORMANCE.md). Each test fails on the
code before the change it names."""
import dataclasses
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import cardclock, gpu3d, imaging, scene3d as s
from nodebased.core import Dispatcher
from tests.test_3d_playback_rate import camera, need_gpu, room


def read_only_copy(scene):
    """The same scene made again, as the Evaluator does for another tier: new array objects, read-only, equal content."""
    geometries = []
    for g in scene.geometries:
        fields = {}
        for name in ("vertices", "triangles", "normals", "uvs"):
            array = getattr(g, name, None)
            if isinstance(array, np.ndarray):
                array = array.copy()
                array.flags.writeable = False
                fields[name] = array
        geometries.append(dataclasses.replace(g, **fields))
    return dataclasses.replace(scene, geometries=tuple(geometries))


class ArrayKeyTests(unittest.TestCase):
    def test_equal_read_only_arrays_have_one_key_and_different_ones_do_not(self):
        a = np.arange(30, dtype="f4").reshape(10, 3)
        b, c = a.copy(), a.copy()
        c[4, 1] += 1
        for array in (a, b, c):
            array.flags.writeable = False
        self.assertEqual(gpu3d._array_key(a), gpu3d._array_key(b))
        self.assertNotEqual(gpu3d._array_key(a), gpu3d._array_key(c))

    def test_a_read_only_array_keeps_its_key_when_asked_again_without_hashing_it_again(self):
        a = np.arange(3000, dtype="f4").reshape(1000, 3).copy()
        a.flags.writeable = False
        first = gpu3d._array_key(a)
        with unittest.mock.patch.object(gpu3d.hashlib, "blake2b", side_effect=AssertionError("hashed again")):
            self.assertEqual(gpu3d._array_key(a), first)

    def test_a_read_only_view_of_an_array_that_can_change_is_hashed_every_time(self):
        base = np.zeros((10, 3), "f4")
        view = base[:]
        view.flags.writeable = False
        before = gpu3d._array_key(view)
        base[2, 1] = 5.0
        self.assertNotEqual(gpu3d._array_key(view), before)

    def test_an_array_over_the_hash_limit_is_still_told_by_identity(self):
        with unittest.mock.patch.object(gpu3d, "HASH_BYTES", 64):
            a = np.zeros((100, 3), "f4")
            b = a.copy()
            a.flags.writeable = b.flags.writeable = False
            self.assertEqual(gpu3d._array_key(a), gpu3d._array_key(a))
            self.assertNotEqual(gpu3d._array_key(a), gpu3d._array_key(b))

    def test_the_memo_holds_a_bounded_number_of_bytes(self):
        arrays = []
        with unittest.mock.patch.object(gpu3d, "DIGEST_MEMO_BYTES", 4096):
            for index in range(20):
                a = np.full((100, 3), index, "f4")        # 1200 bytes each
                a.flags.writeable = False
                arrays.append(a)
                gpu3d._array_key(a)
            self.assertLessEqual(sum(held[0].nbytes for held in gpu3d._digests.values()), 4096)


class SharedPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_gpu(cls)

    def setUp(self):
        gpu3d.drop_resident()
        self.state = gpu3d._state()
        self.state.pop("shadow_cache", None)

    def counters(self):
        state = self.state
        return (state.get("resident_builds", 0), state.get("resident_reuses", 0),
                state.get("shadow_cache_misses", 0), state.get("shadow_cache_hits", 0))

    def draw(self, scene, size, x=0.5):
        with unittest.mock.patch.object(gpu3d, "_adapter_kind", return_value="discrete"):
            return gpu3d.render(scene, camera(x), *size, (0, 0, 0, 0), 0.2, 1)

    def test_the_full_size_frame_reuses_the_pack_the_half_size_frame_made(self):
        scene = room()
        self.draw(scene, (48, 27))
        builds, reuses, misses, hits = self.counters()
        self.draw(read_only_copy(scene), (96, 54))
        after = self.counters()
        self.assertEqual(after[0], builds, "the resident scene was built again")
        self.assertEqual(after[1], reuses + 1)
        self.assertEqual(after[2], misses, "the shadow triangles were packed again")
        self.assertEqual(after[3], hits + 1)

    def test_going_back_to_the_half_size_scene_and_forward_again_packs_nothing(self):
        scene = room()
        self.draw(scene, (48, 27))
        again = read_only_copy(scene)
        self.draw(again, (96, 54))
        builds = self.counters()[0]
        for size, current in (((48, 27), scene), ((96, 54), again), ((48, 27), read_only_copy(scene))):
            self.draw(current, size)
        self.assertEqual(self.counters()[0], builds)

    def test_an_edit_during_play_packs_again_and_draws_the_edit(self):
        scene = room()
        before = self.draw(scene, (96, 54))
        counts = self.counters()
        edited = read_only_copy(scene)
        moved = edited.geometries[2].vertices.copy()
        moved[:, 0] += 0.75
        moved.flags.writeable = False
        geometries = list(edited.geometries)
        geometries[2] = dataclasses.replace(geometries[2], vertices=moved)
        edited = dataclasses.replace(edited, geometries=tuple(geometries))
        picture = self.draw(edited, (96, 54))
        after = self.counters()
        self.assertEqual(after[0], counts[0] + 1, "the edit did not build a new resident scene")
        self.assertEqual(after[2], counts[2] + 1, "the edit did not pack the shadow triangles again")
        self.assertGreater(float(np.abs(picture - before).max()), 0.05)
        gpu3d.drop_resident()
        self.state.pop("shadow_cache", None)
        np.testing.assert_allclose(picture, self.draw(edited, (96, 54)), atol=2e-3, rtol=0)

    def test_a_light_edit_with_the_same_geometry_packs_again(self):
        scene = room()
        self.draw(scene, (96, 54))
        builds = self.counters()[0]
        lamp = scene.lights[0]
        dimmer = dataclasses.replace(scene, lights=(dataclasses.replace(lamp, intensity=lamp.intensity * 0.5),))
        self.draw(read_only_copy(dimmer), (96, 54))
        self.assertEqual(self.counters()[0], builds + 1)

    def test_half_size_and_full_size_frames_equal_a_fresh_render_at_their_size(self):
        scene = room()
        half = self.draw(scene, (48, 27))
        full = self.draw(read_only_copy(scene), (96, 54))
        for size, picture in (((48, 27), half), ((96, 54), full)):
            with self.subTest(size=size):
                gpu3d.drop_resident()
                self.state.pop("shadow_cache", None)
                fresh = self.draw(read_only_copy(scene), size)
                self.assertGreater(float((picture[..., 3] > 0).mean()), 0.5)
                np.testing.assert_allclose(picture, fresh, atol=2e-3, rtol=0)


class StagingBufferTests(unittest.TestCase):
    """The readback buffer of the chosen size is made by the half-size frames, not by the first full-size frame."""

    @classmethod
    def setUpClass(cls):
        need_gpu(cls)

    def setUp(self):
        self.state = gpu3d._state()
        self.saved = self.state.get("staging"), self.state.get("staging_wanted")
        self.state["staging"] = None
        self.state.pop("staging_wanted", None)
        self.addCleanup(self.restore)

    def restore(self):
        held = self.state.get("staging")
        if held is not None and held is not self.saved[0]:
            held.destroy()
        self.state["staging"], wanted = self.saved
        if wanted is None:
            self.state.pop("staging_wanted", None)
        else:
            self.state["staging_wanted"] = wanted

    def test_releasing_a_big_buffer_remembers_its_size(self):
        gpu3d._staging_buffer(self.state, 20 << 20, lambda b: b)
        gpu3d._release_staging(self.state)
        self.assertEqual(self.state["staging_wanted"], 20 << 20)
        self.assertIsNone(self.state["staging"])

    def test_a_ramp_frame_makes_the_buffer_the_chosen_size_wanted_and_the_switch_frame_reuses_it(self):
        self.state["staging_wanted"] = 40 << 20
        ramp = cardclock.Ramp(1 / 24)
        self.addCleanup(ramp.finish)
        small = gpu3d._staging_buffer(self.state, 10 << 20, lambda b: b)
        self.assertGreaterEqual(small.size, 40 << 20)
        creates = self.state.get("staging_creates", 0)
        big = gpu3d._staging_buffer(self.state, 40 << 20, lambda b: b)
        self.assertIs(big, small)
        self.assertEqual(self.state.get("staging_creates", 0), creates)

    def test_a_ramp_that_never_saw_the_chosen_size_grows_the_buffer_by_its_size_ratio_squared(self):
        ramp = cardclock.Ramp(1 / 24, scale=2)
        self.addCleanup(ramp.finish)
        small = gpu3d._staging_buffer(self.state, 10 << 20, lambda b: b)
        self.assertGreaterEqual(small.size, 40 << 20)
        self.assertLess(small.size, 45 << 20)
        creates = self.state.get("staging_creates", 0)
        self.assertIs(gpu3d._staging_buffer(self.state, 40 << 20, lambda b: b), small)
        self.assertEqual(self.state.get("staging_creates", 0), creates)

    def test_a_buffer_the_state_would_not_keep_is_not_made_early(self):
        ramp = cardclock.Ramp(1 / 24, scale=2)
        self.addCleanup(ramp.finish)
        size = (gpu3d.STAGING_KEEP_BYTES // 4) + (1 << 20)
        self.assertEqual(gpu3d._staging_buffer(self.state, size, lambda b: b).size, size)

    def test_without_a_ramp_a_frame_makes_only_the_buffer_it_needs(self):
        self.state["staging_wanted"] = 40 << 20
        small = gpu3d._staging_buffer(self.state, 10 << 20, lambda b: b)
        self.assertEqual(small.size, 10 << 20)

    def test_a_ramp_that_has_ended_does_not_enlarge_the_buffer(self):
        self.state["staging_wanted"] = 40 << 20
        cardclock.Ramp(1 / 24).finish()
        self.assertFalse(cardclock.active())
        self.assertEqual(gpu3d._staging_buffer(self.state, 10 << 20, lambda b: b).size, 10 << 20)


class EvaluatorTierTests(unittest.TestCase):
    """The real source of the two sets of arrays: one document evaluated at tier 1 and at tier 2 by the Evaluator."""

    @classmethod
    def setUpClass(cls):
        need_gpu(cls)

    def document(self, **ball):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "type": "Sphere3D", "id": "ball", "params": {"rows": 8, "columns": 16, **ball}},
            {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
            {"op": "create", "type": "Camera3D", "id": "camera", "params": {}},
            {"op": "create", "type": "Render3D", "id": "render", "params": {"width": 96, "height": 54, "samples": 1,
                                                                         "render_backend": "gpu"}},
            {"op": "connect", "id": "scene", "input": "object0", "source": "ball"},
            {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "render", "input": "camera", "source": "camera"}]})
        return dispatcher.document

    def test_one_document_at_two_tiers_builds_one_resident_scene_and_an_edit_builds_another(self):
        gpu3d.drop_resident()
        state = gpu3d._state()
        evaluator = imaging.Evaluator()
        document = self.document()
        sizes = []
        real = gpu3d.render

        def spy(scene, cam, width, height, *args, **kwargs):
            sizes.append((width, height))
            return real(scene, cam, width, height, *args, **kwargs)
        with unittest.mock.patch.object(gpu3d, "render", spy):
            evaluator.evaluate(document, "render", frame=1, tier=2)
            builds = state.get("resident_builds", 0)
            evaluator.evaluate(document, "render", frame=1, tier=1)
            self.assertEqual(state.get("resident_builds", 0), builds, "the full-size scene was packed again")
            evaluator.evaluate(self.document(sphere_radius=1.5), "render", frame=1, tier=1)
            self.assertEqual(state.get("resident_builds", 0), builds + 1)
        self.assertGreater(sizes[1][0], sizes[0][0])


if __name__ == "__main__":
    unittest.main()
