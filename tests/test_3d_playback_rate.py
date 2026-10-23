"""Lane 4, 3D playback at the comp's rate (docs/PERFORMANCE.md, "3D playback at the comp's rate").

Each test fails on the code before its change: the shadow BVH was not chosen for a few hundred triangles on a card, a
camera move repacked and re-uploaded the whole scene, a scene with nothing translucent drew a second pass over every
triangle, and the display conversion divided and multiplied by an alpha of 1 on one thread."""
import dataclasses
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, gpurt_render, imaging, playback, scene3d as s, tileexec
from nodebased.core import Dispatcher
from tests.test_new_look_viewer_strip_nl5 import Base
from tests.waiting import wait_until


def card(width, height, color, position, rotation=(0, 0, 0), **fields):
    g = s._card(width, height, color, s.Transform3D(s.Vec3(*position), s.Vec3(*rotation)))
    return s.Geometry(g.vertices, g.triangles, color, g.transform, uvs=g.uvs, **fields)


def ball(radius, color, position, rows=12, **fields):
    g = s._sphere_grid(radius, rows, rows * 2, color, s.Transform3D(s.Vec3(*position)))
    return s.Geometry(g.vertices, g.triangles, color, g.transform, normals=g.normals, **fields)


def room(translucent=False):
    geometries = [card(8, 8, (0.7, 0.7, 0.65, 1), (0, 0, 0), (-90, 0, 0), material="pbr", metallic=0.0, pbr_roughness=0.3),
                  card(8, 4, (0.8, 0.3, 0.2, 1), (0, 2, -3)),
                  ball(0.8, (0.9, 0.7, 0.4, 1), (-1.0, 0.8, 0.2), material="pbr", metallic=1.0, pbr_roughness=0.3),
                  ball(0.6, (0.4, 0.6, 0.9, 1), (1.2, 0.6, 0.6))]
    if translucent:
        geometries.append(card(2, 2, (0.9, 0.2, 0.2, 0.5), (0.2, 1.0, 1.5)))
        geometries.append(card(2, 2, (0.2, 0.9, 0.2, 0.5), (0.5, 1.2, 2.0)))
    lamp = s.Light("Point", (1, 1, 1), 6.0, s.Vec3(0, 4.0, 1.0), shadows=True)
    return s.Scene(tuple(geometries), lights=(lamp,))


def camera(x, y=1.6, z=6.0):
    return s.Camera(s.Transform3D(s.Vec3(x, y, z)), s.Vec3(0, 0.9, 0), 40.0)


def need_gpu(test):
    if not gpu3d.available():
        raise unittest.SkipTest("no GPU adapter")


class ShadowPathTests(unittest.TestCase):
    def test_a_card_traces_shadows_through_the_bvh_from_a_few_hundred_triangles(self):
        need_gpu(self)
        scene = room()
        count = sum(len(g.triangles) for g in scene.geometries)
        self.assertGreater(count, 300)
        self.assertLess(count, 2000)
        with unittest.mock.patch.object(gpu3d, "_adapter_kind", return_value="discrete"):
            gpu3d.render(scene, camera(0.0), 32, 18, (0, 0, 0, 0), 0.2, 1)
        self.assertEqual(gpu3d.last_shadow_path, "bvh")


class ResidentSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_gpu(cls)

    def render(self, scene, x, size=(96, 54), **kwargs):
        return gpu3d.render(scene, camera(x), *size, (0, 0, 0, 0), 0.2, 1, **kwargs)

    def counters(self):
        state = gpu3d._state()
        return state.get("resident_builds", 0), state.get("resident_reuses", 0)

    def test_a_moving_camera_builds_the_resident_scene_once(self):
        scene = room()
        gpu3d.drop_resident()
        before = self.counters()
        for x in (0.0, 0.5, 1.0, -0.5):
            self.render(scene, x)
        after = self.counters()
        self.assertEqual((after[0] - before[0], after[1] - before[1]), (1, 3))

    def test_a_reused_frame_uploads_no_buffer_and_makes_no_bind_group(self):
        scene = room()
        self.render(scene, 0.0)
        device = gpu3d._state()["device"]
        names = ("create_buffer_with_data", "create_bind_group", "create_texture")
        counts = dict.fromkeys(names, 0)
        originals = {name: getattr(device, name) for name in names}

        def counting(name):
            def call(*args, **kwargs):
                counts[name] += 1
                return originals[name](*args, **kwargs)
            return call
        for name in names:
            setattr(device, name, counting(name))
        try:
            self.render(scene, 0.7)
        finally:
            for name in names:
                delattr(device, name)
        self.assertEqual(counts, dict.fromkeys(names, 0))

    def test_the_picture_equals_the_per_frame_path(self):
        for translucent in (False, True):
            scene = room(translucent)
            for x in (0.0, 1.5, -2.0):
                with self.subTest(translucent=translucent, x=x):
                    resident = self.render(scene, x)
                    with unittest.mock.patch.object(gpu3d, "RESIDENT", False):
                        per_frame = self.render(scene, x)
                    self.assertGreater(float((resident[..., 3] > 0).mean()), 0.5)
                    np.testing.assert_allclose(resident, per_frame, atol=2e-3, rtol=0)

    def test_the_data_passes_come_from_resident_buffers_too_and_equal_the_per_frame_path(self):
        scene = room(True)
        gpu3d.drop_resident()
        self.render(scene, 0.0)
        builds = self.counters()[0]
        for output in ("depth", "normals", "position", "uv", "object_id", "albedo", "diffuse"):
            with self.subTest(output=output):
                resident = gpu3d.render(scene, camera(1.0), 96, 54, (0, 0, 0, 0), 0.2, 1, output=output)
                with unittest.mock.patch.object(gpu3d, "RESIDENT", False):
                    per_frame = gpu3d.render(scene, camera(1.0), 96, 54, (0, 0, 0, 0), 0.2, 1, output=output)
                np.testing.assert_allclose(resident, per_frame, atol=2e-3, rtol=1e-4)
        # the first data pass builds its entry (it draws without shadows, so it is not the beauty's); the rest reuse it
        self.assertLessEqual(self.counters()[0] - builds, 2)
        before = self.counters()[0]
        gpu3d.render(scene, camera(2.0), 96, 54, (0, 0, 0, 0), 0.2, 1, output="depth")
        gpu3d.render(scene, camera(3.0), 96, 54, (0, 0, 0, 0), 0.2, 1, output="depth")
        self.assertEqual(self.counters()[0], before)

    def test_a_camera_inside_the_room_that_crosses_the_near_plane_still_draws(self):
        scene = room()
        inside = s.Camera(s.Transform3D(s.Vec3(0.0, 0.4, 0.2)), s.Vec3(0, 0.4, -3.0), 90.0)
        resident = gpu3d.render(scene, inside, 96, 54, (0, 0, 0, 0), 0.2, 1)
        with unittest.mock.patch.object(gpu3d, "RESIDENT", False):
            per_frame = gpu3d.render(scene, inside, 96, 54, (0, 0, 0, 0), 0.2, 1)
        np.testing.assert_allclose(resident, per_frame, atol=2e-2, rtol=0)

    def test_an_opaque_scene_draws_one_pass_and_a_translucent_one_two(self):
        self.render(room(), 0.0)
        self.assertEqual(sorted(list(gpu3d._state()["residents"].values())[-1]["binds"]), [0])
        self.render(room(True), 0.0)
        self.assertEqual(sorted(list(gpu3d._state()["residents"].values())[-1]["binds"]), [0, 1])

    def test_translucent_triangles_blend_far_to_near_from_either_side(self):
        scene = room(True)
        for x in (3.0, -3.0):
            with self.subTest(x=x):
                resident = self.render(scene, x)
                with unittest.mock.patch.object(gpu3d, "RESIDENT", False):
                    per_frame = self.render(scene, x)
                np.testing.assert_allclose(resident, per_frame, atol=2e-3, rtol=0)

    def test_a_changed_light_or_object_builds_a_new_resident_scene_and_shows_it(self):
        scene = room()
        first = self.render(scene, 0.0)
        builds, _ = self.counters()
        dimmer = dataclasses.replace(scene, lights=(dataclasses.replace(scene.lights[0], intensity=1.0),))
        second = self.render(dimmer, 0.0)
        self.assertEqual(self.counters()[0], builds + 1)
        self.assertGreater(float(np.abs(first - second).max()), 0.01)
        moved = dataclasses.replace(scene, geometries=(scene.geometries[0], scene.geometries[1],
                                                       dataclasses.replace(scene.geometries[2], transform=s.Transform3D(s.Vec3(1.0, 0.8, 0.2))))
                                    + scene.geometries[3:])
        third = self.render(moved, 0.0)
        self.assertEqual(self.counters()[0], builds + 2)
        self.assertGreater(float(np.abs(first - third).max()), 0.01)

    def test_a_textured_scene_keeps_the_per_frame_path(self):
        texture = np.random.default_rng(3).random((16, 16, 4), dtype=np.float32)
        texture[..., 3] = 1.0
        textured = dataclasses.replace(card(2, 2, (1, 1, 1, 1), (0, 1, 0)), texture=texture)
        scene = s.Scene((textured,))
        self.assertFalse(gpu3d._resident_ok(scene, "rgba", None))
        self.assertTrue(gpu3d._resident_ok(room(), "rgba", None))
        self.assertTrue(gpu3d._resident_ok(room(), "depth", None))

    def test_the_static_vertices_do_not_depend_on_the_camera_and_match_the_camera_path(self):
        scene = room(True)
        order = gpu3d.link_order(scene)
        vertices, materials, opaque, translucent, everything = gpu3d._prepare_static(scene, order, None)
        _, _, packed, _, per_frame_materials = gpu3d._prepare(scene, camera(0.0), 96, 54, None)
        reference = np.concatenate(packed)
        self.assertEqual(vertices.shape, reference.shape)
        np.testing.assert_array_equal(vertices[:, 3:], reference[:, 3:])      # everything but the camera-space position
        self.assertFalse(vertices[:, :3].any())
        self.assertEqual(len(materials), len(per_frame_materials))
        self.assertEqual(sum(count for _, count, _ in opaque) // 3 + len(translucent["start"]), len(vertices) // 3)
        self.assertEqual(sum(count for _, count, _ in everything), len(vertices))


class DisplayConversionTests(unittest.TestCase):
    def frame(self, alpha=1.0, size=(90, 160), seed=5):
        rng = np.random.default_rng(seed)
        frame = rng.random(size + (4,), dtype=np.float32) * 1.4
        frame[..., 3] = alpha
        frame[..., :3] *= frame[..., 3:4]
        return frame

    def both(self, frame, **kwargs):
        args = dict(exposure=0.0, channel="RGB", background="black", view="sRGB", look=None)
        args.update(kwargs)
        with unittest.mock.patch.object(imaging, "OPAQUE_SHORTCUT", False), \
                unittest.mock.patch.object(imaging, "THREADED_QUANTIZE", False):
            reference = imaging.to_qimage(frame, **args)
        return imaging.to_qimage(frame, **args), reference

    def test_an_opaque_frame_converts_to_the_same_bytes_without_the_divide(self):
        for view in ("sRGB", "Linear", "Raw", "ACES 2.0"):
            for kwargs in ({}, {"exposure": 1.0}, {"look": {"gamma": 2.2}}, {"look": {"zebra": True}}, {"channel": "G"},
                           {"background": "checker"}):
                with self.subTest(view=view, **kwargs):
                    fast, reference = self.both(self.frame(), view=view, **kwargs)
                    self.assertEqual(fast, reference)

    def test_a_translucent_frame_still_takes_the_full_path(self):
        frame = self.frame(alpha=0.6)
        with unittest.mock.patch.object(imaging, "_unpremultiply", wraps=imaging._unpremultiply) as divide:
            imaging.to_qimage(frame)
        self.assertEqual(divide.call_count, 1)
        fast, reference = self.both(frame)
        self.assertEqual(fast, reference)

    def test_an_opaque_frame_skips_the_divide(self):
        with unittest.mock.patch.object(imaging, "_unpremultiply", wraps=imaging._unpremultiply) as divide:
            imaging.to_qimage(self.frame())
        self.assertEqual(divide.call_count, 0)

    def test_linear_and_raw_never_write_into_the_callers_frame(self):
        for view in ("Linear", "Raw"):
            frame = self.frame()
            before = frame.copy()
            imaging.to_qimage(frame, view=view)
            np.testing.assert_array_equal(frame, before)

    def test_the_banded_quantize_equals_the_single_pass(self):
        rng = np.random.default_rng(9)
        for shape in ((600, 700, 3), (513, 515, 3), (4, 5, 3)):
            values = (rng.random(shape, dtype=np.float32) * 1.6 - 0.3)
            threaded = imaging._quantize(values.copy())
            with unittest.mock.patch.object(imaging, "THREADED_QUANTIZE", False):
                single = imaging._quantize(values.copy())
            np.testing.assert_array_equal(threaded, single)

    def test_the_banded_unpremultiply_equals_the_single_pass(self):
        rng = np.random.default_rng(4)
        weight = rng.random((700, 800, 1), dtype=np.float32)
        weight[:100] = 0.0
        weight[:, :150] = 0.0
        weight[300:320, 300:340] = 1.0
        rgb = rng.random((700, 800, 3), dtype=np.float32) * weight
        threaded = imaging._unpremultiply(rgb, weight)
        with unittest.mock.patch.object(imaging, "THREADED_QUANTIZE", False):
            single = imaging._unpremultiply(rgb, weight)
        np.testing.assert_array_equal(threaded, single)

    def test_the_covered_box_is_looked_for_once(self):
        frame = np.zeros((300, 400, 4), np.float32)
        frame[100:140, 120:200] = (0.2, 0.3, 0.4, 1.0)
        with unittest.mock.patch.object(imaging, "_covered_box", wraps=imaging._covered_box) as probe:
            imaging.to_qimage(frame)
        self.assertEqual(probe.call_count, 1)

    def test_an_rgba_frame_goes_to_the_card_whole_and_the_cpu_path_takes_three_channels(self):
        from nodebased import color, gpudisplay
        frame = self.frame()
        seen = []

        class Card:
            def render(self, rgb, view, copy=True):
                seen.append(rgb.shape)
                return np.ascontiguousarray(rgb[..., :3])
        with unittest.mock.patch.object(gpudisplay, "get_display", return_value=Card()), \
                unittest.mock.patch.object(imaging, "CARD_PICTURE", False):
            fast = imaging.to_qimage(frame, view="ACES 2.0")
        self.assertEqual(seen, [frame.shape])
        with unittest.mock.patch.object(gpudisplay, "get_display", return_value=None):
            cpu = imaging.to_qimage(frame, view="ACES 2.0")
        self.assertEqual(color.display_rgb(frame, "ACES 2.0").shape, frame.shape[:2] + (3,))
        self.assertEqual(fast.size(), cpu.size())
        # anything that changes the colour first keeps the three-channel input
        seen.clear()
        with unittest.mock.patch.object(gpudisplay, "get_display", return_value=Card()), \
                unittest.mock.patch.object(imaging, "CARD_PICTURE", False):
            imaging.to_qimage(frame, exposure=1.0, view="ACES 2.0")
            imaging.to_qimage(frame, view="ACES 2.0", look={"gamma": 2.0})
        self.assertEqual(seen, [frame.shape[:2] + (3,)] * 2)

    def test_the_card_gives_the_same_picture_from_an_rgba_frame_as_from_three_channels(self):
        from PySide6.QtWidgets import QApplication
        from nodebased import gpudisplay
        QApplication.instance() or QApplication([])
        gpudisplay.reset_for_testing()
        self.addCleanup(gpudisplay.reset_for_testing)
        display = gpudisplay.get_display(force=True)
        if display is None:
            self.skipTest(f"no GPU display context: {gpudisplay.status()}")
        frame = self.frame()
        three = display.render(np.ascontiguousarray(frame[..., :3]), "ACES 2.0")
        four = display.render(frame, "ACES 2.0")
        self.assertEqual(four.shape, three.shape)
        np.testing.assert_array_equal(four, three)
        again = display.render(np.ascontiguousarray(frame[..., :3]), "ACES 2.0")      # the texture changes back
        np.testing.assert_array_equal(again, three)

    def test_the_card_makes_the_8_bit_picture_of_an_opaque_frame_within_one_level_of_the_cpu_quantise(self):
        from PySide6.QtWidgets import QApplication
        from nodebased import gpudisplay
        QApplication.instance() or QApplication([])
        gpudisplay.reset_for_testing()
        self.addCleanup(gpudisplay.reset_for_testing)
        if gpudisplay.get_display(force=True) is None:
            self.skipTest(f"no GPU display context: {gpudisplay.status()}")
        frame = self.frame(size=(72, 128))
        frame[..., :3] *= 1.5
        frame[0, :4, :3] = ((-0.5, 0.2, 0.9), (30.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1e-6, 1e-6, 1e-6))      # out of range and dark
        calls = []
        real = imaging._card_picture
        with unittest.mock.patch.object(imaging, "_card_picture", side_effect=lambda *a: calls.append(1) or real(*a)):
            card = imaging.to_qimage(frame, view="ACES 2.0")
        self.assertEqual(len(calls), 1)
        with unittest.mock.patch.object(imaging, "CARD_PICTURE", False):
            cpu = imaging.to_qimage(frame, view="ACES 2.0")
        self.assertEqual((card.width(), card.height(), card.format()), (cpu.width(), cpu.height(), cpu.format()))

        def pixels(image):
            return np.frombuffer(image.constBits(), np.uint8, count=image.bytesPerLine() * image.height()).reshape(
                image.height(), image.bytesPerLine())[:, :image.width() * 3].astype(int)
        difference = np.abs(pixels(card) - pixels(cpu))
        self.assertLessEqual(int(difference.max()), 1)
        self.assertLess(float((difference > 0).mean()), 0.02)

    def test_a_frame_with_any_translucency_or_an_edited_look_does_not_go_to_the_card_picture(self):
        translucent = self.frame(alpha=0.5)
        with unittest.mock.patch.object(imaging, "_card_picture", return_value=None) as card:
            imaging.to_qimage(translucent, view="ACES 2.0")
            imaging.to_qimage(self.frame(), exposure=1.0, view="ACES 2.0")
            imaging.to_qimage(self.frame(), view="ACES 2.0", look={"gamma": 2.0})
            imaging.to_qimage(self.frame(), view="ACES 2.0", background="checker")
            imaging.to_qimage(self.frame(), view="ACES 2.0", channel="G")
            imaging.to_qimage(self.frame(), view="sRGB")
        card.assert_not_called()
        with unittest.mock.patch.object(imaging, "_card_picture", return_value=None) as card:
            imaging.to_qimage(self.frame(), view="ACES 2.0")
        card.assert_called_once()

    def test_a_large_frame_is_quantized_on_the_pool(self):
        from nodebased import color
        calls = []
        real = color._cpu_pool.map
        with unittest.mock.patch.object(color._cpu_pool, "map", side_effect=lambda *a, **k: calls.append(1) or real(*a, **k)):
            imaging._quantize(np.zeros((600, 700, 3), np.float32))
        self.assertEqual(len(calls), 1)


class SplatTierTests(unittest.TestCase):
    """Meshes and splats under a shadowing light: ray traced when still, raster plus a splat layer while playing."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from benchmark_3d_playback import build_scene, make_splat_file
        cls.directory = tempfile.TemporaryDirectory()
        cls.splat_path = make_splat_file(400, cls.directory.name)
        ops, cls.view = build_scene("mixed", 96, 54, "auto", splat_path=cls.splat_path)
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "batch", "commands": ops})
        cls.document = dispatcher.document

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_only_a_render3d_with_splats_upstream_is_marked(self):
        playing = playback.playing_document(self.document)
        self.assertEqual(playing["nodes"]["render"]["params"]["pt_seed"], playback.PLAYING_SPLAT_SEED)
        self.assertNotEqual(self.document["nodes"]["render"]["params"]["pt_seed"], playback.PLAYING_SPLAT_SEED)
        plain = Dispatcher()
        plain.execute({"op": "batch", "commands": [
            {"op": "create", "type": "Cube3D", "id": "cube", "params": {}},
            {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
            {"op": "create", "type": "Camera3D", "id": "camera", "params": {}},
            {"op": "create", "type": "Render3D", "id": "render", "params": {"samples": 1, "render_backend": "auto"}},
            {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
            {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "render", "input": "camera", "source": "camera"}]})
        self.assertIs(playback.playing_document(plain.document), plain.document)

    def test_the_marked_document_has_its_own_cache_identity(self):
        playing = playback.playing_document(self.document)
        before = playback.DisplayCache.identity(self.document, self.view, 1)
        self.assertNotEqual(before, playback.DisplayCache.identity(playing, self.view, 1))

    def test_playing_skips_the_ray_tracer_and_a_still_frame_uses_it(self):
        if not gpu3d.available():
            raise unittest.SkipTest("no GPU adapter")
        playing = playback.playing_document(self.document)
        with unittest.mock.patch.object(gpurt_render, "render", wraps=gpurt_render.render) as traced:
            still = imaging.Evaluator().evaluate(self.document, self.view, frame=1, tier=1)
            self.assertGreaterEqual(traced.call_count, 1)
            traced.reset_mock()
            moving = imaging.Evaluator().evaluate(playing, self.view, frame=1, tier=1)
            self.assertEqual(traced.call_count, 0)
        self.assertEqual(still.shape, moving.shape)
        self.assertGreater(float((moving[..., 3] > 0).mean()), 0.3)
        self.assertGreater(float((still[..., 3] > 0).mean()), 0.3)

    def test_the_flag_is_per_thread_and_restored(self):
        self.assertFalse(gpu3d.is_playing())
        with gpu3d.playing(True):
            self.assertTrue(gpu3d.is_playing())
            with gpu3d.playing(False):
                self.assertFalse(gpu3d.is_playing())
            self.assertTrue(gpu3d.is_playing())
        self.assertFalse(gpu3d.is_playing())


class DisplayCacheCompactionTests(unittest.TestCase):
    """A full display cache rounds old frames to half-float off the preview worker."""

    def frames(self, count, size=(64, 96)):
        rng = np.random.default_rng(11)
        return [rng.random(size + (4,), dtype=np.float32) * 3.0 for _ in range(count)]

    def fill(self, cache, frames):
        for number, frame in enumerate(frames):
            cache.put(("scene", number), frame, lazy=True)

    def test_a_put_over_budget_rounds_on_another_thread(self):
        import threading
        frames = self.frames(8)
        cache = playback.DisplayCache(budget_bytes=int(frames[0].nbytes * 4.5))
        cache.background_compaction = True
        threads = []
        real = cache._compact_one
        slots = []
        original_slots = cache._compact_slots

        def spy(batch):
            threads.append(threading.current_thread().name)
            return original_slots(batch)
        with unittest.mock.patch.object(cache, "_compact_slots", side_effect=spy), \
                unittest.mock.patch.object(cache, "_compact_one", side_effect=lambda slot: slots.append(slot) or real(slot)):
            self.fill(cache, frames)
            cache.drain()
        self.assertEqual(slots, [], "a put rounded a frame on the calling thread")
        self.assertTrue(threads and all(name.startswith("nodebased-compact") for name in threads))
        cache.close()

    def test_the_cache_ends_inside_its_budget_with_the_frames_it_kept_intact(self):
        frames = self.frames(8)
        cache = playback.DisplayCache(budget_bytes=int(frames[0].nbytes * 4.5))
        cache.background_compaction = True
        self.fill(cache, frames)
        cache.drain()
        self.assertLessEqual(cache.bytes, cache.budget)
        self.assertEqual(cache.bytes, sum(entry[0].nbytes for entry in cache._entries.values()))
        self.assertGreater(len(cache), 4, "rounding to half-float should keep more frames than the float32 budget holds")
        key = ("scene", 7)
        np.testing.assert_allclose(cache.get(key), frames[7], rtol=1e-3, atol=1e-3)
        cache.close()

    def test_without_the_background_the_rounding_is_still_inline(self):
        frames = self.frames(8)
        cache = playback.DisplayCache(budget_bytes=int(frames[0].nbytes * 4.5))
        self.fill(cache, frames)
        self.assertLessEqual(cache.bytes, cache.budget)
        self.assertIsNone(cache._pool)

    def test_a_frame_replaced_while_it_is_being_rounded_is_not_overwritten(self):
        frames = self.frames(3)
        cache = playback.DisplayCache(budget_bytes=int(frames[0].nbytes * 10))
        cache.put("a", frames[0], lazy=True)
        slot = ("a", None)
        cache._compacting.add(slot)
        old = cache._entries[slot][0]
        cache.put("a", frames[1], lazy=True)
        cache._compacting.add(slot)
        original = playback.half_safe

        def slow(frame):
            cache.put("a", frames[2], lazy=True)       # a newer frame lands while this one is being rounded
            return original(frame)
        with unittest.mock.patch.object(playback, "half_safe", side_effect=slow):
            cache._compact_slots([slot])
        np.testing.assert_array_equal(cache.get("a"), frames[2])
        self.assertIsNot(cache._entries[slot][0], old)


class GeometryMemoTests(unittest.TestCase):
    """A 3D shape node makes its geometry once while its parameters and inputs stay the same."""

    def document(self, **cube):
        dispatcher = Dispatcher()
        dispatcher.execute({"op": "batch", "commands": [
            {"op": "create", "type": "Sphere3D", "id": "ball", "params": {"rows": 8, "columns": 16, **cube}},
            {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
            {"op": "create", "type": "Camera3D", "id": "camera", "params": {}},
            {"op": "create", "type": "Render3D", "id": "render", "params": {"width": 32, "height": 18, "samples": 1,
                                                                         "render_backend": "cpu"}},
            {"op": "connect", "id": "scene", "input": "object0", "source": "ball"},
            {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
            {"op": "connect", "id": "render", "input": "camera", "source": "camera"}]})
        return dispatcher.document

    def made(self, evaluator, document):
        seen = []
        real = s.geometry_from_node
        with unittest.mock.patch.object(s, "geometry_from_node", side_effect=lambda *a, **k: seen.append(1) or real(*a, **k)):
            evaluator.evaluate(document, "render", frame=1, tier=1)
        return len(seen)

    def test_the_shape_is_built_once_and_again_only_after_an_edit(self):
        evaluator = imaging.Evaluator()
        document = self.document()
        self.assertEqual(self.made(evaluator, document), 1)
        evaluator.cache.clear()
        evaluator.bytes = 0
        self.assertEqual(self.made(evaluator, document), 0)
        self.assertEqual(self.made(evaluator, self.document(sphere_radius=1.25)), 1)

    def test_the_arrays_are_read_only_and_a_clear_forgets_the_geometry(self):
        evaluator = imaging.Evaluator()
        document = self.document()
        self.made(evaluator, document)
        geometry = next(iter(evaluator._geometry_memo.values()))
        for array in (geometry.vertices, geometry.triangles, geometry.normals):
            self.assertFalse(array.flags.writeable)
        evaluator.clear()
        self.assertEqual(self.made(evaluator, document), 1)


class ComposeShortcutTests(unittest.TestCase):
    """A Render3D that is the whole target comes back as the renderer's own frame, not cut into 256-pixel tiles and put back."""

    def document(self, target_grade=False):
        dispatcher = Dispatcher()
        commands = [{"op": "create", "type": "Cube3D", "id": "cube", "params": {}},
                    {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
                    {"op": "create", "type": "Camera3D", "id": "camera", "params": {}},
                    {"op": "create", "type": "Render3D", "id": "render",
                     "params": {"width": 600, "height": 340, "samples": 1, "render_backend": "cpu"}},
                    {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
                    {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
                    {"op": "connect", "id": "render", "input": "camera", "source": "camera"}]
        if target_grade:
            commands += [{"op": "create", "type": "Grade", "id": "grade", "params": {"exposure": 0.5}},
                         {"op": "connect", "id": "grade", "input": "image", "source": "render"}]
        dispatcher.execute({"op": "batch", "commands": commands})
        return dispatcher.document

    def compose(self, document, target, direct, region=None):
        executor = tileexec.TileExecutor()
        bounds = executor.canvas_region(document, target, frame=1, tier=1)
        with unittest.mock.patch.object(tileexec, "RENDER3D_DIRECT", direct):
            return executor.compose_region(document, target, region or bounds, frame=1, tier=1), executor

    def test_the_direct_frame_equals_the_tiled_one(self):
        document = self.document()
        direct, _ = self.compose(document, "render", True)
        tiled, _ = self.compose(document, "render", False)
        np.testing.assert_array_equal(direct.pixels, tiled.pixels)
        self.assertEqual((direct.canvas_width, direct.canvas_height), (tiled.canvas_width, tiled.canvas_height))
        self.assertEqual(direct.tile_misses, 0)
        self.assertGreater(tiled.tile_misses, 4)

    def test_a_crop_or_a_node_after_the_render_still_tiles(self):
        document = self.document(target_grade=True)
        executor = tileexec.TileExecutor()
        bounds = executor.canvas_region(document, "render", frame=1, tier=1)
        crop = tileexec.TileRegion(100, 50, 200, 120, full_x=0, full_y=0, full_width=bounds.width, full_height=bounds.height)
        cropped, _ = self.compose(document, "render", True, crop)
        self.assertGreater(cropped.tile_misses, 0)
        self.assertEqual(cropped.pixels.shape[:2], (120, 200))
        graded, _ = self.compose(document, "grade", True)
        self.assertGreater(graded.tile_misses, 0)


class NextFrameBeforeHandOverTests(Base):
    """While playing, the next frame is requested before the finished one is put on screen."""

    def test_the_worker_is_started_before_the_picture_is_shown(self):
        w = self.open_window((1280, 720))
        w.agent_command({"op": "batch", "commands": [
            {"op": "create", "type": "Constant", "id": "flat", "params": {"width": 1920, "height": 1080}},
            {"op": "set_key", "id": "flat", "param": "red", "frame": 1, "value": 0.1},
            {"op": "set_key", "id": "flat", "param": "red", "frame": 48, "value": 0.9},
            {"op": "view", "id": "flat"}]})
        w.set_time(first=1, last=48, current=1, fps=120.0)       # frames slower than the clock: the next one is always due
        self.assertTrue(wait_until(lambda: not w.busy and not len(w.preview_queue)))
        log = []
        real_ready, real_start, real_show = w.preview_ready, w.start_preview, w._show_image

        def ready(*args, **kwargs):
            log.append("ready")
            return real_ready(*args, **kwargs)

        def start(*args, **kwargs):
            before = w.busy
            result = real_start(*args, **kwargs)
            if w.busy and not before:
                log.append("start")
            return result

        def show(*args, **kwargs):
            log.append("show")
            return real_show(*args, **kwargs)
        w.preview_ready, w.start_preview, w._show_image = ready, start, show
        w.signals.finished.disconnect()
        w.signals.finished.connect(ready)
        w.playback_proxy.setChecked(False)
        # The early start is the serial path's: with the played-frame pipeline (10/22) the worker is freed when a frame
        # is evaluated, before its "ready", which test_playback_pipeline covers. Hold this one to the serial path.
        w.pipeline_playback = False
        self.addCleanup(setattr, w, "pipeline_playback", True)
        w.toggle_playback(True)
        self.assertTrue(wait_until(lambda: log.count("show") >= 6, timeout=20))
        w.toggle_playback(False)
        # between a "ready" and the "show" it leads to, the next "start" has already happened (once playback is under way)
        turns = []
        for index, entry in enumerate(log):
            if entry == "ready":
                after = log[index + 1:]
                shown = after.index("show") if "show" in after else None
                if shown is not None:
                    turns.append("start" in after[:shown])
        self.assertGreaterEqual(len(turns), 4)
        self.assertTrue(sum(turns) >= len(turns) - 2, turns)


class PlaybackTierAdaptationTests(Base):
    """The automatic playback tier ignores cold frames, steps down when frames miss the rate and climbs back when they have room."""

    def start(self, fps=24.0):
        w = self.open_window((1280, 720))
        w.set_time(first=1, last=240, current=1, fps=fps)
        w.playback_proxy.setChecked(True)
        w.playing = True
        w._adapt_floor_tier = 1
        w._adapt_warm = w.ADAPT_WARM_FRAMES
        w._adapt_headroom = 0
        return w

    def frame(self, w, ms):
        from types import SimpleNamespace
        return w._adapt_playback_proxy(SimpleNamespace(tier=int(w.proxy.currentData() or 1)), ms)

    def test_cold_frames_do_not_move_the_tier(self):
        w = self.start()
        self.assertEqual([self.frame(w, 900), self.frame(w, 400)], [False, False])
        self.assertEqual(int(w.proxy.currentData()), 1)
        self.assertTrue(self.frame(w, 400))
        self.assertEqual(int(w.proxy.currentData()), 2)
        w.playing = False

    def test_each_new_tier_gets_one_cold_frame_and_a_miss_steps_down_again(self):
        w = self.start()
        w._adapt_warm = 0
        self.assertTrue(self.frame(w, 60))
        self.assertFalse(self.frame(w, 300))            # the first frame at the new tier: not counted
        self.assertTrue(self.frame(w, 60))
        self.assertEqual(int(w.proxy.currentData()), 4)
        w.playing = False

    def test_the_tier_climbs_back_after_a_run_of_frames_with_room_and_stops_at_the_floor(self):
        w = self.start()
        w._adapt_warm = 0
        self.assertTrue(self.frame(w, 60))
        self.assertTrue(self.frame(w, 60) is False)      # cold frame at tier 2
        results = [self.frame(w, 10) for _ in range(w.ADAPT_UP_FRAMES)]
        self.assertTrue(results[-1])
        self.assertEqual(int(w.proxy.currentData()), 1)
        for _ in range(3 * w.ADAPT_UP_FRAMES):
            self.assertFalse(self.frame(w, 5))           # already at the floor: nothing above it
        self.assertEqual(int(w.proxy.currentData()), 1)
        w.playing = False

    def test_frames_without_enough_room_keep_the_tier_and_a_slow_one_resets_the_count(self):
        w = self.start()
        w._adapt_warm = 0
        self.assertTrue(self.frame(w, 60))
        self.frame(w, 10)
        for _ in range(3 * w.ADAPT_UP_FRAMES):
            self.assertFalse(self.frame(w, 30))          # inside the frame time but over 1/2.5 of it
        for _ in range(w.ADAPT_UP_FRAMES - 1):
            self.frame(w, 10)
        self.assertFalse(self.frame(w, 35))              # resets the run
        for _ in range(w.ADAPT_UP_FRAMES - 1):
            self.assertFalse(self.frame(w, 10))
        self.assertEqual(int(w.proxy.currentData()), 2)
        w.playing = False

    def test_a_tier_the_artist_chose_is_left_alone(self):
        w = self.start()
        w.proxy.setCurrentIndex(w.proxy.findData(2))
        w._adapt_warm = 0
        for ms in (500, 500, 5, 5):
            self.assertFalse(self.frame(w, ms))
        self.assertEqual(int(w.proxy.currentData()), 2)
        w.playing = False


class RefineOnStopTests(Base):
    """A stop draws the frame at the node's own settings when the picture on screen was a playback frame."""

    def stop_requests(self, commands, tier=None):
        w = self.open_window((1280, 720))
        w.agent_command({"op": "batch", "commands": commands})
        self.assertTrue(wait_until(lambda: not w.busy and not len(w.preview_queue)))
        if tier is not None:
            w.proxy.setCurrentIndex(w.proxy.findData(tier))
            self.assertTrue(wait_until(lambda: not w.busy and not len(w.preview_queue)))
        w.set_time(first=1, last=24, current=1, fps=24.0)
        w.toggle_playback(True)
        seen = []
        real = w.request_preview
        w.request_preview = lambda *a, **k: seen.append((a, k)) or real(*a, **k)
        w.toggle_playback(False)
        return seen

    def test_a_render3d_at_two_samples_is_redrawn_when_playback_stops(self):
        commands = [{"op": "create", "type": "Cube3D", "id": "cube", "params": {}},
                    {"op": "create", "type": "Scene3D", "id": "scene", "params": {}},
                    {"op": "create", "type": "Camera3D", "id": "camera", "params": {}},
                    {"op": "create", "type": "Render3D", "id": "render", "params": {"width": 64, "height": 36, "samples": 2,
                                                                                     "render_backend": "auto"}},
                    {"op": "connect", "id": "scene", "input": "object0", "source": "cube"},
                    {"op": "connect", "id": "render", "input": "scene", "source": "scene"},
                    {"op": "connect", "id": "render", "input": "camera", "source": "camera"},
                    {"op": "view", "id": "render"}]
        self.assertEqual(len(self.stop_requests(commands)), 1)

    def test_a_comp_with_no_playback_frames_of_its_own_asks_for_nothing(self):
        commands = [{"op": "create", "type": "Constant", "id": "flat", "params": {}}, {"op": "view", "id": "flat"}]
        self.assertEqual(self.stop_requests(commands), [])


class TimeRowTierTests(Base):
    """The time row says which resolution tier is on screen while playing."""

    def play_at(self, tier, fps_shown):
        w = self.open_window()
        w.set_time(fps=24.0)
        w.playing = True
        w.playback_origin_time = time.monotonic() - 1.0
        now = time.monotonic()
        w.playback_shown_times.clear()
        w.playback_shown_times.extend(now - 1.0 + i * (1.0 / fps_shown) for i in range(1, int(fps_shown) + 1))
        w.proxy.setCurrentIndex(w.proxy.findData(tier))
        return w

    def test_full_resolution_adds_nothing(self):
        w = self.play_at(1, 24)
        w.update_realtime_light()
        self.assertEqual(w.realtime_label.text(), "real time · 24 fps")
        w.playing = False

    def test_a_proxy_tier_is_named_when_the_rate_holds_and_when_it_does_not(self):
        for tier, note in ((2, "half res"), (4, "quarter res"), (8, "eighth res")):
            with self.subTest(tier=tier):
                w = self.play_at(tier, 24)
                w.update_realtime_light()
                self.assertEqual(w.realtime_label.text(), f"real time · 24 fps · {note}")
                w.playback_shown_times.clear()
                w.playback_shown_times.extend(time.monotonic() - 0.95 + i * 0.0865 for i in range(11))
                w.update_realtime_light()
                self.assertTrue(w.realtime_label.text().endswith(f"of 24 fps · {note}"), w.realtime_label.text())
                w.playing = False
                w.update_realtime_light()
                self.assertEqual(w.realtime_label.text(), "real time")


if __name__ == "__main__":
    unittest.main()
