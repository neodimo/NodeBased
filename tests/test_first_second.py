"""The first second of a 3D scene (docs/PERFORMANCE.md, "Lane 4: the first second").

Part 2, the first played frames: a stopped-frame render that cannot be cancelled kept the first played frame waiting (the 100 s
processor render of the smoke, splats and glass scene started the moment playback stopped; pressing play again showed nothing for
a minute and a half). Two causes, each with its test below: the direct Render3D path of `TileExecutor.compose_region` did not hand its
cancel event to the evaluator, and `PlaybackQueue` cancelled only the newest taken job, not the one still running ahead of it.

Part 3, builds before playback: the first picture builds every pipeline and shader the scene needs, and the display pass the played
frames use (`gpupicture.warm`), so the first played frames build nothing. `tools/benchmark_3d_playback.py --first-seconds` reports the
builds in the first seconds; the tests here count them.

Part 1, the first picture: the coarse tiers of a progressive preview (the first look after opening a scene or editing it) draw
a Render3D with splats and meshes on the card, as playback does. A scene with smoke, splats and a see-through sphere took 7 s
for its first picture because the exact picture comes from the processor; each test here fails without `first_look_document`
and the draft documents in `Window.request_preview`."""
import copy
import os
import unittest
import unittest.mock
import uuid

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, imaging, playback, scene3d
from nodebased import scene3d as s
from nodebased.core import Dispatcher
from tests.test_3d_playback_cost import build, render_commands


def splat_commands(smoke=False, glass=False, lamp=False, backend="auto", mode="raster", size=(64, 36), path=""):
    commands = render_commands(samples=2, mode=mode, backend=backend)
    for command in commands:
        if command.get("id") == "render" and command["op"] == "create":
            command["params"].update(width=size[0], height=size[1])
    commands.insert(0, {"op": "create", "type": "ReadSplat3D", "id": "splats", "params": {"splat_path": path, "splat_relight": 1.0}})
    members = ["cube", "splats"]
    if glass:
        commands.insert(0, {"op": "create", "type": "Sphere3D", "id": "glass",
                            "params": {"sphere_radius": 0.5, "red": 0.9, "green": 0.95, "blue": 0.95, "alpha": 0.35}})
        members.append("glass")
    if lamp:
        commands.insert(0, {"op": "create", "type": "Light3D", "id": "lamp",
                            "params": {"light_type": "Point", "ty": 3.0, "intensity": 6.0, "shadows": "on"}})
        members.append("lamp")
    if smoke:
        commands.insert(0, {"op": "create", "type": "Plume3D", "id": "smoke", "params": {"plume_resolution": 16, "uscale": 1.2}})
        members.append("smoke")
    commands = [c for c in commands if not (c["op"] == "connect" and c.get("id") == "scene")]
    for slot, member in enumerate(members):
        commands.insert(len(commands) - 1, {"op": "connect", "id": "scene", "input": f"object{slot}", "source": member})
    return commands


class FirstLookDocumentTests(unittest.TestCase):
    def test_a_splat_render_is_marked_as_the_playing_route_and_the_input_is_untouched(self):
        document = build(*splat_commands())
        draft = playback.first_look_document(document)
        self.assertEqual(draft["nodes"]["render"]["params"]["pt_seed"], playback.PLAYING_SPLAT_SEED)
        self.assertNotEqual(document["nodes"]["render"]["params"].get("pt_seed"), playback.PLAYING_SPLAT_SEED)

    def test_the_draft_keeps_the_samples_the_artist_set(self):
        document = build(*splat_commands())
        self.assertEqual(playback.first_look_document(document)["nodes"]["render"]["params"]["samples"], 2)

    def test_scenes_without_splats_the_cpu_backend_and_other_modes_are_left_alone(self):
        for name, commands in (("no splats", render_commands(samples=2)),
                               ("cpu backend", splat_commands(backend="cpu")),
                               ("path tracer", splat_commands(mode="pathtrace")),
                               ("ray tracer", splat_commands(mode="raytrace"))):
            with self.subTest(case=name):
                document = build(*commands)
                self.assertIs(playback.first_look_document(document), document)

    def test_a_document_that_is_already_a_draft_comes_back_as_it_is(self):
        draft = playback.first_look_document(build(*splat_commands()))
        self.assertIs(playback.first_look_document(draft), draft)


APP = None


def _window():
    global APP
    from PySide6.QtWidgets import QApplication
    APP = QApplication.instance() or QApplication([])
    from nodebased.app import Window
    return Window(agent_name="first-second-" + uuid.uuid4().hex)


class ProgressiveDraftTests(unittest.TestCase):
    def setUp(self):
        self.window = _window()
        self.window.agent_command({"op": "batch", "commands": splat_commands()})
        self.window.agent_command({"op": "view", "id": "render"})
        held = unittest.mock.patch.object(self.window, "start_preview")     # the queued tiers are read here, not rendered
        held.start()
        self.addCleanup(held.stop)
        self.addCleanup(self.close)

    def close(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()

    def seed(self, request):
        return request.document["nodes"]["render"]["params"].get("pt_seed")

    def test_the_coarse_tiers_are_drafts_and_the_selected_tier_is_the_exact_document(self):
        w = self.window
        w.proxy.setCurrentIndex(w.proxy.findData(1))
        w.request_preview(progressive=True)
        request, cancel = w.preview_queue.take()
        self.assertEqual(request.tier, 8)
        self.assertEqual(self.seed(request), playback.PLAYING_SPLAT_SEED)
        picture = np.zeros((135, 240, 4), dtype=np.float32)
        w.preview_ready((request, cancel), picture, imaging.to_qimage(picture), "20 ms")
        request, cancel = w.preview_queue.take()
        self.assertEqual(request.tier, 4)
        self.assertEqual(self.seed(request), playback.PLAYING_SPLAT_SEED)
        w.preview_ready((request, cancel), picture, imaging.to_qimage(picture), "20 ms")
        request, cancel = w.preview_queue.take()
        self.assertEqual(request.tier, 1)
        self.assertNotEqual(self.seed(request), playback.PLAYING_SPLAT_SEED, "the settled picture must be the exact one")

    def test_a_tier_the_artist_selected_is_never_a_draft(self):
        w = self.window
        w.proxy.setCurrentIndex(w.proxy.findData(4))
        w.request_preview(progressive=True)
        request, cancel = w.preview_queue.take()
        self.assertEqual(request.tier, 8)
        self.assertEqual(self.seed(request), playback.PLAYING_SPLAT_SEED)
        picture = np.zeros((135, 240, 4), dtype=np.float32)
        w.preview_ready((request, cancel), picture, imaging.to_qimage(picture), "20 ms")
        request, cancel = w.preview_queue.take()
        self.assertEqual(request.tier, 4)
        self.assertNotEqual(self.seed(request), playback.PLAYING_SPLAT_SEED)

    def test_a_plain_request_is_not_a_draft(self):
        w = self.window
        w.request_preview()
        request, _ = w.preview_queue.take()
        self.assertNotEqual(self.seed(request), playback.PLAYING_SPLAT_SEED)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class DraftIsDrawnOnTheCard(unittest.TestCase):
    """Smoke that shades relit splats under a shadowing light went to the processor outside playback (7 s at the coarsest tier)."""
    def setUp(self):
        import tempfile
        from tools.benchmark_pathtrace import splat_shell
        from nodebased import splats
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "shell.ply")
        splats.write_ply(splat_shell(400), self.path)
        self.document = build(*splat_commands(smoke=True, glass=True, lamp=True, path=self.path))

    def evaluate(self, document, tier):
        calls = []

        def processor(scene, camera, width, height, *args, **kwargs):
            calls.append((width, height))
            return np.zeros((height, width, 4), np.float32)
        with unittest.mock.patch.object(scene3d, "render", side_effect=processor):
            imaging.Evaluator().evaluate(document, "render", frame=1, tier=tier)
        return calls

    def test_the_draft_of_smoke_splats_and_glass_is_drawn_on_the_card(self):
        self.assertEqual(self.evaluate(playback.first_look_document(self.document), 8), [],
                         "the draft went to the processor rasteriser")

    def test_the_exact_picture_still_comes_from_the_processor(self):
        self.assertEqual(len(self.evaluate(self.document, 8)), 1)


class CancelReachesTheRunningRender(unittest.TestCase):
    def test_a_cancel_set_before_the_direct_render3d_path_runs_stops_it(self):
        import threading
        from nodebased import tileexec
        from nodebased.cancellation import Cancelled
        document = build(*render_commands(samples=1, backend="cpu"))
        executor = tileexec.TileExecutor()
        bounds = executor.canvas_region(document, "render", frame=1, tier=1)
        cancel = threading.Event()
        cancel.set()
        with unittest.mock.patch.object(tileexec, "RENDER3D_DIRECT", True), self.assertRaises(Cancelled):
            executor.compose_region(document, "render", bounds, frame=1, tier=1, cancel=cancel)

    def test_the_queue_cancels_every_job_still_running_not_only_the_newest(self):
        queue = playback.PlaybackQueue()
        document = {"nodes": {}}
        queue.replace(1, 1, document)
        _, running = queue.take()                  # the stopped frame's render: it runs for a long time
        queue.replace(2, 1, document, cancel_active=False)
        _, waiting = queue.take()                  # taken while the first still runs; it waits behind it on the one worker
        self.assertIsNot(running, waiting)
        queue.replace(3, 2, document)              # the artist presses play
        self.assertTrue(running.is_set(), "the render running ahead of the newest job was left to finish")
        self.assertTrue(waiting.is_set())

    def test_a_retired_job_is_not_touched_again(self):
        queue = playback.PlaybackQueue()
        document = {"nodes": {}}
        queue.replace(1, 1, document)
        _, done = queue.take()
        queue.retire(done)
        queue.active_cancel = None
        queue.replace(2, 1, document)
        self.assertFalse(done.is_set())

    def test_the_worker_retires_every_job_it_ran(self):
        import time
        w = _window()
        self.addCleanup(lambda: (setattr(w, "saved_document", w.dispatcher.document), w.close()))
        w.agent_command({"op": "batch", "commands": render_commands(samples=1, backend="cpu")})
        w.agent_command({"op": "view", "id": "render"})
        end = time.monotonic() + 60
        while time.monotonic() < end and (w.busy or len(w.preview_queue) or w.frame is None):
            APP.processEvents()
            time.sleep(0.01)
        self.assertIsNotNone(w.frame)
        self.assertEqual(w.preview_queue._running, set(), "jobs the worker finished are still listed as running")

    def test_cancel_reaches_a_running_job_too(self):
        queue = playback.PlaybackQueue()
        queue.replace(1, 1, {"nodes": {}})
        _, running = queue.take()
        queue.replace(2, 1, {"nodes": {}}, cancel_active=False)
        queue.take()
        queue.cancel()
        self.assertTrue(running.is_set())


PIPELINE_CALLS = ("create_render_pipeline", "create_compute_pipeline", "create_shader_module")


def pipeline_builds(trace):
    return [(kind, round(ms, 1)) for _started, kind, ms, _size in trace.rows if kind in PIPELINE_CALLS]


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class FirstFramesBuildNothingNew(unittest.TestCase):
    def setUp(self):
        from nodebased import gpupicture
        from tools.benchmark_3d_playback import BuildTrace
        self.gpupicture = gpupicture
        patcher = unittest.mock.patch.object(gpupicture, "ARMED_KINDS", ("discrete", "integrated", "cpu", "other"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.trace = BuildTrace()
        self.addCleanup(self.trace.remove)

    def render(self, scene, camera, *, view=None, warm=False, playing=False, **more):
        self.gpupicture.clear()
        if view is not None:
            self.gpupicture.arm(view)
        elif warm:
            self.gpupicture.arm_warm(warm)
        try:
            with gpu3d.playing(playing):
                return gpu3d.render(scene, camera, 96, 54, **more)
        finally:
            self.gpupicture.disarm()

    def test_a_picture_before_playback_builds_the_display_pass_the_played_frames_use(self):
        from tests.test_3d_card_picture import HDR_SCENE
        state = gpu3d._state()
        state.pop("picture", None)                      # a cold display pass
        self.trace.install()
        self.render(HDR_SCENE, s.Camera(), warm="ACES 2.0")        # the first picture, still, from the coarse tier
        self.trace.rows.clear()
        for step in range(3):                           # the first played frames
            self.render(HDR_SCENE, s.Camera(), view="ACES 2.0")
        self.assertEqual(pipeline_builds(self.trace), [], "a played frame built a shader or pipeline the first picture could have")

    def test_the_played_frames_of_smoke_splats_and_glass_rebuild_no_pipeline_the_first_picture_built(self):
        from dataclasses import replace
        from tests.test_3d_splats_smoke_on_card import smoke_scene, SETTINGS
        from tests.test_3d_splats_on_card import card, many
        from tests.test_3d_gpu_volumes import CAMERA
        scene = smoke_scene(many(300, 3), geometries=(card(-.5),), relight=1.0)
        self.trace.install()
        self.render(scene, CAMERA, playing=True, volume=SETTINGS, ambient=.2)      # the first picture (a draft is drawn as a played frame)
        built = pipeline_builds(self.trace)
        self.assertGreater(len(built), 0, "the first frame must have built the scene's pipelines")
        self.trace.rows.clear()
        position = CAMERA.transform.position
        for step in range(1, 5):                        # the camera moves, as it does in the first seconds of playback
            moved = replace(CAMERA, transform=replace(CAMERA.transform, position=s.Vec3(position.x + .02 * step, position.y, position.z)))
            self.render(scene, moved, playing=True, volume=SETTINGS, ambient=.2)
        self.assertEqual(pipeline_builds(self.trace), [], "the first frames after a cold start rebuilt pipelines the scene had built")


if __name__ == "__main__":
    unittest.main()
