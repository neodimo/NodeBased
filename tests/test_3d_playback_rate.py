"""Lane 4, 3D playback at the comp's rate (docs/PERFORMANCE.md, "3D playback at the comp's rate").

Each test fails on the code before its change: the shadow BVH was not chosen for a few hundred triangles on a card, a
camera move repacked and re-uploaded the whole scene, a scene with nothing translucent drew a second pass over every
triangle, and the display conversion divided and multiplied by an alpha of 1 on one thread."""
import dataclasses
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, imaging, scene3d as s


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

    def test_a_camera_inside_the_room_that_crosses_the_near_plane_still_draws(self):
        scene = room()
        inside = s.Camera(s.Transform3D(s.Vec3(0.0, 0.4, 0.2)), s.Vec3(0, 0.4, -3.0), 90.0)
        resident = gpu3d.render(scene, inside, 96, 54, (0, 0, 0, 0), 0.2, 1)
        with unittest.mock.patch.object(gpu3d, "RESIDENT", False):
            per_frame = gpu3d.render(scene, inside, 96, 54, (0, 0, 0, 0), 0.2, 1)
        np.testing.assert_allclose(resident, per_frame, atol=2e-2, rtol=0)

    def test_an_opaque_scene_draws_one_pass_and_a_translucent_one_two(self):
        self.render(room(), 0.0)
        self.assertEqual(sorted(gpu3d._state()["resident"]["binds"]), [0])
        self.render(room(True), 0.0)
        self.assertEqual(sorted(gpu3d._state()["resident"]["binds"]), [0, 1])

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
        self.assertFalse(gpu3d._resident_ok(room(), "depth", None))

    def test_the_static_vertices_do_not_depend_on_the_camera_and_match_the_camera_path(self):
        scene = room(True)
        order = gpu3d.link_order(scene)
        vertices, materials, opaque, translucent = gpu3d._prepare_static(scene, order, None)
        _, _, packed, _, per_frame_materials = gpu3d._prepare(scene, camera(0.0), 96, 54, None)
        reference = np.concatenate(packed)
        self.assertEqual(vertices.shape, reference.shape)
        np.testing.assert_array_equal(vertices[:, 3:], reference[:, 3:])      # everything but the camera-space position
        self.assertFalse(vertices[:, :3].any())
        self.assertEqual(len(materials), len(per_frame_materials))
        self.assertEqual(sum(count for _, count, _ in opaque) // 3 + len(translucent["start"]), len(vertices) // 3)


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

    def test_a_large_frame_is_quantized_on_the_pool(self):
        from nodebased import color
        calls = []
        real = color._cpu_pool.map
        with unittest.mock.patch.object(color._cpu_pool, "map", side_effect=lambda *a, **k: calls.append(1) or real(*a, **k)):
            imaging._quantize(np.zeros((600, 700, 3), np.float32))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
