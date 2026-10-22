"""Lane 4 playback profile fixes (docs/PERFORMANCE.md, "Lane 4: 3D playback profile").

Each test fails on the code before its fix: the playing document keeps its sample count, an empty scene's
animated camera counts as time, the readback buffer is created again for every render, `read_mapped()` copies into
a bytearray, an evicted played frame goes to the disk tier, and the display conversion divides the whole frame."""
import os
import unittest
import unittest.mock
import uuid

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, imaging, playback, playback_trace, scene3d as s
from nodebased.core import Dispatcher


def build(*commands):
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": list(commands)})
    return dispatcher.document


def render_commands(samples=2, mode="raster", scene_members=("cube",)):
    commands = [{"op": "create", "type": "Cube3D", "id": "cube", "params": {}},
                {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
                {"op": "create", "type": "Camera3D", "id": "camera", "params": {}},
                {"op": "create", "type": "Render3D", "id": "render",
                 "params": {"width": 64, "height": 36, "samples": samples, "render_mode": mode}},
                {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
                {"op": "connect", "id": "render", "input": "camera", "source": "camera"},
                {"op": "view", "id": "render"}]
    for slot, member in enumerate(scene_members):
        commands.append({"op": "connect", "id": "scene", "input": f"object{slot}", "source": member})
    return commands


class PlayingDocumentTests(unittest.TestCase):
    def test_raster_render3d_is_capped_at_one_sample_and_the_input_is_untouched(self):
        document = build(*render_commands(samples=2))
        playing = playback.playing_document(document)
        self.assertEqual(playing["nodes"]["render"]["params"]["samples"], 1)
        self.assertEqual(document["nodes"]["render"]["params"]["samples"], 2)

    def test_raytrace_is_capped_and_pathtrace_is_not(self):
        self.assertEqual(playback.playing_document(build(*render_commands(4, "raytrace")))["nodes"]["render"]["params"]["samples"], 1)
        pathtrace = build(*render_commands(4, "pathtrace"))
        self.assertIs(playback.playing_document(pathtrace), pathtrace)

    def test_nothing_to_cap_returns_the_same_document(self):
        document = build(*render_commands(samples=1))
        self.assertIs(playback.playing_document(document), document)

    def test_playing_and_paused_documents_do_not_share_display_cache_frames(self):
        document = build(*render_commands(samples=2))
        paused = playback.DisplayCache.identity(document, "render", 1)
        playing = playback.DisplayCache.identity(playback.playing_document(document), "render", 1)
        self.assertNotEqual(paused, playing)


class EmptySceneStaticTests(unittest.TestCase):
    def animated_camera(self):
        return [{"op": "set_key", "id": "camera", "param": "tx", "frame": 1, "value": 5},
                {"op": "set_key", "id": "camera", "param": "tx", "frame": 24, "value": -5}]

    def test_an_empty_scene_under_an_animated_camera_is_static(self):
        document = build(*render_commands(scene_members=()), *self.animated_camera())
        self.assertFalse(playback.is_time_dependent(document, "render"))

    def test_the_same_camera_in_front_of_a_cube_is_time_dependent(self):
        document = build(*render_commands(scene_members=("cube",)), *self.animated_camera())
        self.assertTrue(playback.is_time_dependent(document, "render"))

    def test_an_animated_background_of_an_empty_scene_still_counts(self):
        document = build(*render_commands(scene_members=()),
                         {"op": "set_key", "id": "render", "param": "red", "frame": 1, "value": 0},
                         {"op": "set_key", "id": "render", "param": "red", "frame": 24, "value": 1})
        self.assertTrue(playback.is_time_dependent(document, "render"))


class StagingBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not gpu3d.available():
            raise unittest.SkipTest("no GPU adapter")

    def scene(self):
        cube = s._cube(2.0, (0.8, 0.5, 0.3, 1), s.Transform3D())
        light = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(2, 4, 3), s.Vec3(0, 0, 0))
        return s.Scene((cube,), lights=(light,))

    def camera(self, x=5.0):
        return s.Camera(s.Transform3D(s.Vec3(x, 1.5, 0)), s.Vec3(0, 0, 0), 40.0)

    def test_the_readback_buffer_is_made_once_and_reused(self):
        state = gpu3d._state()
        gpu3d.release_staging(state)
        first = gpu3d.render(self.scene(), self.camera(), 96, 54, (0, 0, 0, 0), 0.2, 1)
        created = state["staging_creates"]
        reused = state.get("staging_reuses", 0)
        again = gpu3d.render(self.scene(), self.camera(4.0), 96, 54, (0, 0, 0, 0), 0.2, 1)
        self.assertEqual(state["staging_creates"], created, "a second render of the same size made a new readback buffer")
        self.assertEqual(state["staging_reuses"], reused + 1)
        self.assertEqual(first.shape, again.shape)

    def test_a_reused_buffer_gives_the_same_picture(self):
        gpu3d.release_staging()
        cold = gpu3d.render(self.scene(), self.camera(), 80, 45, (0.1, 0.2, 0.3, 1), 0.2, 2)
        warm = gpu3d.render(self.scene(), self.camera(), 80, 45, (0.1, 0.2, 0.3, 1), 0.2, 2)
        np.testing.assert_array_equal(cold, warm)

    def test_a_bigger_picture_replaces_the_held_buffer(self):
        state = gpu3d._state()
        gpu3d.release_staging(state)
        gpu3d.render(self.scene(), self.camera(), 64, 36, (0, 0, 0, 0), 0.2, 1)
        small = state["staging"].size
        created = state["staging_creates"]
        gpu3d.render(self.scene(), self.camera(), 128, 72, (0, 0, 0, 0), 0.2, 1)
        self.assertGreater(state["staging"].size, small)
        self.assertEqual(state["staging_creates"], created + 1)

    def test_a_transparent_background_leaves_the_picture_as_rendered(self):
        image = gpu3d.render(self.scene(), self.camera(), 64, 36, (0, 0, 0, 0), 0.2, 1)
        self.assertEqual(float(image[0, 0, 3]), 0.0)
        self.assertGreater(float(image[18, 32, 3]), 0.0)

    def test_stage_marks_are_recorded_only_while_a_profiler_is_on(self):
        gpu3d.render(self.scene(), self.camera(), 64, 36, (0, 0, 0, 0), 0.2, 1)
        self.assertFalse(playback_trace.active())
        playback_trace.enable()
        gpu3d.render(self.scene(), self.camera(), 64, 36, (0, 0, 0, 0), 0.2, 1)
        labels = [label for _, _, label in playback_trace.disable()]
        for expected in ("render_begin", "scene_ready", "upload_done", "submit", "gpu_done", "readback_done", "resolve_done"):
            self.assertIn(expected, labels)
        self.assertLess(labels.index("submit"), labels.index("gpu_done"))


class ReadbackCopyTests(unittest.TestCase):
    def test_a_band_is_copied_straight_from_the_mapping_not_through_a_bytearray(self):
        stride = 256
        raw = np.arange(4 * stride // 4, dtype="f4")
        staging = unittest.mock.Mock()
        staging.read_mapped.return_value = memoryview(raw).cast("B")
        result = np.zeros((4, 8, 4), "f4")
        gpu3d._copy_band(staging, result, 0, 4, np.dtype("f4"), stride, 8)
        staging.read_mapped.assert_called_once_with(copy=False)
        expected = raw.reshape(4, stride // 4)[:, :32].reshape(4, 8, 4)
        np.testing.assert_array_equal(result, expected)


class EvictionSpillTests(unittest.TestCase):
    def evaluator_with_one_slot(self):
        evaluator = imaging.Evaluator()
        evaluator.disk = unittest.mock.Mock()
        evaluator.cache.clear()
        evaluator.bytes = 0
        evaluator._fixed_budget = 100
        evaluator._shared_budget = None
        return evaluator

    def raster(self, n=60):
        from nodebased.raster import Raster
        return Raster.of(np.zeros((1, n // 4, 4), "f4"))

    def test_an_evicted_frame_goes_to_the_disk_tier_when_paused(self):
        evaluator = self.evaluator_with_one_slot()
        evaluator._store("a", self.raster())
        evaluator._store("b", self.raster())
        evaluator.disk.put_raster.assert_called_once()

    def test_an_evicted_frame_is_not_written_while_playing(self):
        evaluator = self.evaluator_with_one_slot()
        evaluator.disk_spills_enabled = False
        evaluator._store("a", self.raster())
        evaluator._store("b", self.raster())
        evaluator.disk.put_raster.assert_not_called()
        self.assertEqual(list(evaluator.cache), ["b"])


class DisplayConversionTests(unittest.TestCase):
    def test_unpremultiply_equals_the_masked_divide_bit_for_bit(self):
        for seed in range(4):
            generator = np.random.RandomState(seed)
            frame = generator.uniform(-1, 3, (48, 64, 4)).astype(np.float32)
            frame[..., 3] = generator.choice([0, 1, 0.5, 1e-9, 0.2, 2.0, -0.3], (48, 64))
            frame[5, 5, 0] = np.inf
            frame[6, 6, 1] = np.nan
            rgb, weight = frame[..., :3], np.clip(frame[..., 3:4], 0, 1)
            expected = np.divide(rgb, weight, out=np.zeros_like(rgb), where=weight > 1e-8)
            np.testing.assert_array_equal(imaging._unpremultiply(rgb, weight), expected)

    def test_an_object_on_transparent_black_is_divided_only_inside_its_box(self):
        frame = np.zeros((90, 160, 4), np.float32)
        frame[30:50, 40:90, :3] = 0.25
        frame[30:50, 40:90, 3] = 0.5
        rgb, weight = frame[..., :3], np.clip(frame[..., 3:4], 0, 1)
        straight = imaging._unpremultiply(rgb, weight)
        self.assertEqual(float(straight[40, 60, 0]), 0.5)
        self.assertEqual(float(straight[0, 0, 0]), 0.0)
        self.assertEqual(float(straight.sum()), float(0.5 * 3 * 20 * 50))

    def test_an_empty_frame_converts_to_zeros(self):
        frame = np.zeros((9, 16, 4), np.float32)
        np.testing.assert_array_equal(imaging._unpremultiply(frame[..., :3], frame[..., 3:4]), 0)

    def test_to_qimage_with_and_without_exposure_still_agree_at_zero_stops(self):
        generator = np.random.RandomState(3)
        frame = generator.uniform(0, 1, (12, 20, 4)).astype(np.float32)
        frame[..., :3] *= frame[..., 3:4]
        a = imaging.to_qimage(frame, 0.0, "RGB", "black", "sRGB")
        b = imaging.to_qimage(frame.copy(), 0, "RGB", "black", "sRGB")
        self.assertEqual(a, b)
        self.assertEqual(a.width(), 20)


class WindowPlaybackRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from nodebased.app import Window
        self.window = Window(agent_name="playcost-" + uuid.uuid4().hex)
        self.window.agent_command({"op": "batch", "commands": render_commands(samples=2)})
        self.window.set_time(first=1, last=24, current=1, fps=24.0)

    def tearDown(self):
        self.window.playing = False
        self.window.close = lambda: True

    def queued_samples(self):
        request, _ = self.window.preview_queue.take()
        return request.document["nodes"]["render"]["params"]["samples"]

    def test_a_playing_request_carries_the_capped_document(self):
        self.window.toggle_playback(True)
        self.window.request_preview(playhead_only=True)
        self.assertEqual(self.queued_samples(), 1)
        self.window.toggle_playback(False)

    def test_a_paused_request_keeps_the_nodes_own_samples(self):
        self.window.request_preview()
        self.assertEqual(self.queued_samples(), 2)

    def test_the_evaluator_stops_spilling_while_playing_and_starts_again_when_paused(self):
        self.window.toggle_playback(True)
        self.window.request_preview(playhead_only=True)
        self.assertFalse(self.window.evaluator.disk_spills_enabled)
        self.window.toggle_playback(False)
        self.window.request_preview()
        self.assertTrue(self.window.evaluator.disk_spills_enabled)


class TransportPacingTests(WindowPlaybackRequestTests):
    """A finished played frame looks at the clock itself instead of waiting for the next transport tick."""

    def test_the_next_frame_is_requested_when_a_played_frame_lands(self):
        import threading
        from nodebased.playback import FrameRequest
        w = self.window
        w.toggle_playback(True)
        w.playback_timer.stop()                  # no tick is coming
        w.playback_origin_time -= 1.0            # a second of the wall clock has gone by: the next frame is owed
        w.preview_queue.take()
        origin = w.playback_origin_frame
        request = FrameRequest(w.generation, origin, True, w.dispatcher.document, 1, None, True)
        cancel = threading.Event()
        w.busy = True
        frame = np.zeros((36, 64, 4), np.float32)
        with unittest.mock.patch.object(w, "start_preview") as start:
            w.preview_ready((request, cancel), frame, imaging.to_qimage(frame), "64 x 36  -  5 ms")
        start.assert_called_once()
        queued = list(w.preview_queue._items)
        self.assertEqual([item.frame for item in queued if item.display], [origin + 1])
        w.toggle_playback(False)

    def test_a_paused_finish_does_not_advance_anything(self):
        import threading
        from nodebased.playback import FrameRequest
        w = self.window
        request = FrameRequest(w.generation, 1, True, w.dispatcher.document, 1, None, False)
        cancel = threading.Event()
        w.busy = True
        frame = np.zeros((36, 64, 4), np.float32)
        with unittest.mock.patch.object(w, "playback_tick") as tick:
            w.preview_ready((request, cancel), frame, imaging.to_qimage(frame), "64 x 36  -  5 ms")
        tick.assert_not_called()


class BenchmarkToolTests(unittest.TestCase):
    def test_every_scene_builds_valid_operations(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import benchmark_3d_playback as bench
        from nodebased.core import SPECS
        for scene in bench.SCENES:
            ops, view = bench.build_scene(scene, 64, 36, "auto", splat_path="x.ply")
            made = set()
            for op in ops:
                if op["op"] == "create":
                    self.assertNotIn(op["id"], made, scene)
                    self.assertIn(op["type"], SPECS)
                    made.add(op["id"])
                elif op["op"] == "connect":
                    self.assertIn(op["source"], made, scene)
                    self.assertIn(op["id"], made, scene)
            self.assertIn(view, made)

    def test_the_stage_table_splits_frames_at_the_worker_start(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import profile_3d_playback as profile
        events = []
        for frame in range(3):
            t = frame * 1.0
            for step, label in enumerate(("tick", "worker_start", "render_begin", "scene_ready", "upload_done", "submit",
                                          "gpu_done", "readback_done", "resolve_done", "put_begin", "cache_put", "emit", "handover")):
                events.append((t + step * 0.01, "w", label))
        table, frames, _ = profile.stage_table(events)
        self.assertEqual(frames, 3)
        gpu = next(row for row in table if row["from"] == "submit")
        self.assertAlmostEqual(gpu["p50_ms"], 10.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()


class CoveredBoxConversionTests(unittest.TestCase):
    """to_qimage converts only the rectangle with coverage; the picture is the full-frame conversion's, byte for byte."""

    def frame(self, seed=0, box=(20, 60, 30, 110), partial=True):
        generator = np.random.RandomState(seed)
        frame = np.zeros((100, 160, 4), np.float32)
        y0, y1, x0, x1 = box
        frame[y0:y1, x0:x1, :3] = generator.uniform(0, 2, (y1 - y0, x1 - x0, 3))
        frame[y0:y1, x0:x1, 3] = generator.choice([1.0, 0.5, 0.25], (y1 - y0, x1 - x0)) if partial else 1.0
        frame[y0:y1, x0:x1, :3] *= frame[y0:y1, x0:x1, 3:4]
        return frame

    def both(self, frame, **kwargs):
        args = dict(exposure=0.0, channel="RGB", background="black", view="sRGB", look=None)
        args.update(kwargs)
        full = unittest.mock.patch.object(imaging, "CROP_TRANSPARENT", False)
        with full:
            reference = imaging.to_qimage(frame, **args)
        return imaging.to_qimage(frame, **args), reference

    def test_the_cropped_conversion_equals_the_full_one(self):
        for view in ("sRGB", "Linear"):
            for exposure in (0.0, 1.5):
                with self.subTest(view=view, exposure=exposure):
                    cropped, reference = self.both(self.frame(), view=view, exposure=exposure)
                    self.assertEqual(cropped, reference)

    def test_the_crop_is_actually_taken(self):
        frame = self.frame()
        self.assertIsNotNone(imaging._covered_box(frame, "RGB", "black", None))
        calls = []
        real = imaging.to_qimage
        with unittest.mock.patch.object(imaging, "to_qimage", side_effect=lambda f, *a, **k: calls.append(f.shape) or real(f, *a, **k)):
            real(frame)
        self.assertEqual(calls, [(40, 80, 4)])

    def test_edges_of_the_frame_and_single_pixels(self):
        for box in ((0, 30, 0, 50), (70, 100, 110, 160), (50, 51, 80, 81)):
            with self.subTest(box=box):
                cropped, reference = self.both(self.frame(box=box))
                self.assertEqual(cropped, reference)

    def test_whole_frame_cases_are_not_cropped(self):
        frame = self.frame(box=(0, 100, 0, 160))
        self.assertIsNone(imaging._covered_box(frame, "RGB", "black", None))
        self.assertIsNone(imaging._covered_box(self.frame(), "A", "black", None))
        self.assertIsNone(imaging._covered_box(self.frame(), "RGB", "checker", None))
        self.assertIsNone(imaging._covered_box(self.frame(), "RGB", "black", {"zebra": True}))
        self.assertIsNone(imaging._covered_box(np.zeros((100, 160, 4), np.float32), "RGB", "black", None))

    def test_other_channels_and_the_checker_background_match_too(self):
        for kwargs in ({"channel": "A"}, {"channel": "G"}, {"background": "checker"}, {"look": {"zebra": True}}):
            with self.subTest(**kwargs):
                cropped, reference = self.both(self.frame(), **kwargs)
                self.assertEqual(cropped, reference)


class SceneReuseTests(unittest.TestCase):
    """A camera that moves leaves the scene's shadow triangles and BVH alone, and the vertex packing is the loop's."""

    @classmethod
    def setUpClass(cls):
        if not gpu3d.available():
            raise unittest.SkipTest("no GPU adapter")

    def room(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from benchmark_pathtrace import furnished_scene
        scene = furnished_scene()
        lights = tuple(__import__("dataclasses").replace(light, shadows=True) for light in scene.lights)
        # the raster renderer draws meshes and a lamp; the room's glass ball, instanced shelf and sky are for the ray tracers
        meshes = tuple(g for g in scene.geometries if g.material != "liquid")
        return __import__("dataclasses").replace(scene, geometries=meshes, lights=lights, instances=(), environments=())

    def camera(self, x):
        return s.Camera(s.Transform3D(s.Vec3(x, 1.6, 6.2)), s.Vec3(0, 0.95, 0), 40.0)

    def test_a_moving_camera_builds_the_shadow_bvh_once(self):
        from nodebased import raytrace
        scene = self.room()
        state = gpu3d._state()
        state.pop("shadow_cache", None)
        with unittest.mock.patch.object(raytrace.Bvh, "build", wraps=raytrace.Bvh.build) as build, \
                unittest.mock.patch.object(gpu3d, "_shadow_data", wraps=gpu3d._shadow_data) as pack, \
                unittest.mock.patch.object(gpu3d, "SHADOW_BVH_THRESHOLD", 0):
            first = gpu3d.render(scene, self.camera(0.0), 96, 54, (0, 0, 0, 0), 0.2, 1)
            for x in (0.4, 0.8, -0.4):
                gpu3d.render(scene, self.camera(x), 96, 54, (0, 0, 0, 0), 0.2, 1)
        self.assertEqual(build.call_count, 1, "the BVH was rebuilt for a camera move")
        self.assertEqual(pack.call_count, 1, "the shadow triangles were repacked for a camera move")
        self.assertTrue(np.isfinite(first).all())

    def test_a_moved_object_rebuilds_them(self):
        import dataclasses
        from nodebased import raytrace
        scene = self.room()
        moved = dataclasses.replace(scene, geometries=(dataclasses.replace(
            scene.geometries[4], transform=s.Transform3D(s.Vec3(-1.0, 0.8, 0.4))),) + scene.geometries[1:])
        state = gpu3d._state()
        state.pop("shadow_cache", None)
        with unittest.mock.patch.object(raytrace.Bvh, "build", wraps=raytrace.Bvh.build) as build, \
                unittest.mock.patch.object(gpu3d, "SHADOW_BVH_THRESHOLD", 0):
            gpu3d.render(scene, self.camera(0.0), 64, 36, (0, 0, 0, 0), 0.2, 1)
            gpu3d.render(moved, self.camera(0.0), 64, 36, (0, 0, 0, 0), 0.2, 1)
        self.assertEqual(build.call_count, 2)

    def test_reused_shadows_give_the_picture_a_fresh_build_gives(self):
        scene = self.room()
        state = gpu3d._state()
        with unittest.mock.patch.object(gpu3d, "SHADOW_BVH_THRESHOLD", 0):
            gpu3d.render(scene, self.camera(0.0), 80, 45, (0, 0, 0, 0), 0.2, 1)
            reused = gpu3d.render(scene, self.camera(0.5), 80, 45, (0, 0, 0, 0), 0.2, 1)
            state.pop("shadow_cache", None)
            fresh = gpu3d.render(scene, self.camera(0.5), 80, 45, (0, 0, 0, 0), 0.2, 1)
        np.testing.assert_array_equal(reused, fresh)

    def test_vector_packing_equals_the_triangle_loop(self):
        scene = self.room()
        for camera in (self.camera(0.0), s.Camera(s.Transform3D(s.Vec3(-3.0, 2.0, 5.0)), s.Vec3(0, 1, 0), 40.0),
                       s.Camera(s.Transform3D(s.Vec3(0.2, 0.5, 2.0)), s.Vec3(0, 1, 0), 70.0)):
            with unittest.mock.patch.object(gpu3d, "VECTOR_PREPARE", False):
                loop = gpu3d._prepare(scene, camera, 320, 180, None)
            vector = gpu3d._prepare(scene, camera, 320, 180, None)
            np.testing.assert_array_equal(np.concatenate(loop[2]), np.concatenate(vector[2]))
            self.assertEqual(loop[3], vector[3])

    def test_a_geometry_that_crosses_the_near_plane_keeps_the_loop(self):
        scene = self.room()
        camera = s.Camera(s.Transform3D(s.Vec3(0.0, 0.3, 0.2)), s.Vec3(0, 0.3, -3.0), 90.0)     # inside the room, floor and walls close
        with unittest.mock.patch.object(gpu3d, "VECTOR_PREPARE", False):
            loop = gpu3d._prepare(scene, camera, 160, 90, None)
        vector = gpu3d._prepare(scene, camera, 160, 90, None)
        np.testing.assert_array_equal(np.concatenate(loop[2]), np.concatenate(vector[2]))
        self.assertEqual(loop[3], vector[3])

    def test_draws_are_merged_but_every_triangle_is_still_drawn(self):
        scene = self.room()
        with unittest.mock.patch.object(gpu3d, "SHADOW_BVH_THRESHOLD", 0):
            image = gpu3d.render(scene, self.camera(0.0), 120, 68, (0, 0, 0, 0), 0.2, 1)
        covered = float((image[..., 3] > 0).mean())
        self.assertGreater(covered, 0.5)
