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
