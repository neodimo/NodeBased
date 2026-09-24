import os
import tempfile
import math
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased.core import Dispatcher, SCHEMA_VERSION, upgrade_document
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.viewport3d import Viewport3D
from nodebased import scene3d
from PySide6.QtWidgets import QApplication


APP = QApplication.instance() or QApplication([])


def graph():
    d = Dispatcher()
    for key, kind in (("card", "Card3D"), ("cube", "Cube3D"), ("cam", "Camera3D"),
                      ("scene", "Scene3D"), ("render", "Render3D"), ("write", "Write")):
        d.execute({"op": "create", "id": key, "type": kind})
    d.execute({"op": "connect", "id": "scene", "input": "object0", "source": "card"})
    d.execute({"op": "connect", "id": "scene", "input": "object1", "source": "cube"})
    d.execute({"op": "connect", "id": "render", "input": "scene", "source": "scene"})
    d.execute({"op": "connect", "id": "render", "input": "camera", "source": "cam"})
    d.execute({"op": "connect", "id": "write", "input": "image", "source": "render"})
    d.execute({"op": "view", "id": "write"})
    return d


class FoundationTests(unittest.TestCase):
    def test_near_and_far_planes_reject_geometry(self):
        camera = scene3d.Camera()
        geometry = scene3d.Geometry(np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0)), np.float32),
                                    np.array(((0, 1, 2),), np.int32), (1, 0, 0, 1),
                                    scene3d.Transform3D(scene3d.Vec3(0, 0, 4.95)))
        image = scene3d.render(scene3d.Scene((geometry,)), camera, 32, 32)
        self.assertEqual(float(image[..., 3].max()), 0.0)
        camera = scene3d.Camera(camera.transform, camera.target, camera.fov, camera.near, 0.01)
        geometry = scene3d.Geometry(geometry.vertices, geometry.triangles, geometry.color,
                                    scene3d.Transform3D(scene3d.Vec3(0, 0, -10)))
        image = scene3d.render(scene3d.Scene((geometry,)), camera, 32, 32)
        self.assertEqual(float(image[..., 3].max()), 0.0)

    def test_transparent_geometry_does_not_occlude_and_composites_in_depth_order(self):
        vertices = np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
        triangles = np.array(((0, 1, 2), (0, 2, 3)), np.int32)
        front = scene3d.Geometry(vertices, triangles, (1, 0, 0, 0.5), scene3d.Transform3D())
        back = scene3d.Geometry(vertices, triangles, (0, 0, 1, 1),
                                scene3d.Transform3D(scene3d.Vec3(0, 0, -1)))
        image = scene3d.render(scene3d.Scene((front, back)), scene3d.Camera(), 32, 32)
        center = image[16, 16]
        np.testing.assert_allclose(center, (0.5, 0.0, 0.5, 1.0), atol=1e-5)
        invisible = scene3d.Geometry(vertices, triangles, (1, 0, 0, 0), scene3d.Transform3D())
        image = scene3d.render(scene3d.Scene((invisible, back)), scene3d.Camera(), 32, 32)
        np.testing.assert_allclose(image[16, 16], (0, 0, 1, 1), atol=1e-5)

    def test_viewport_shading_is_opt_in_and_depth_matches_projection(self):
        camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(3, 2, 5)))
        cube = scene3d.Scene((scene3d._cube(2, (1, 1, 1, 1), scene3d.Transform3D()),))
        flat = scene3d.render(cube, camera, 80, 64)
        self.assertEqual(float(flat[..., :3][flat[..., 3] > 0].min()), 1.0)
        shaded, depth = scene3d.render(cube, camera, 80, 64, shade=True, return_depth=True)
        covered = shaded[..., 3] > 0
        self.assertGreater(len(np.unique(np.round(shaded[..., 0][covered], 3))), 1)
        self.assertTrue(np.array_equal(covered, np.isfinite(depth)))
        # The cube's centre projects behind its own front faces; a point beside it is unoccluded.
        xy, z = scene3d.project(camera, 80, 64, np.array(((0, 0, 0), (-6, 0, -1)), np.float32))
        x, y = xy.astype(int)[0]
        self.assertLess(float(depth[y, x]), float(z[0]))
        x, y = xy.astype(int)[1]
        self.assertTrue(0 <= x < 80 and 0 <= y < 64 and np.isinf(depth[y, x]))

    def test_viewport_grid_is_hidden_behind_geometry(self):
        camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0, 0.5, 6)))
        wall = scene3d.Scene((scene3d._card(4, 4, (1, 1, 1, 1), scene3d.Transform3D()),))
        _image, depth = scene3d.render(wall, camera, 64, 64, return_depth=True)
        behind = Viewport3D._visible_segments(camera, depth, (-1, 0, -2), (1, 0, -2))
        before = Viewport3D._visible_segments(camera, depth, (-1, 0, 2), (1, 0, 2))
        self.assertEqual(behind, [])
        self.assertGreater(len(before), 0)

    def test_background_rgb_is_premultiplied(self):
        image = scene3d.render(scene3d.Scene(), scene3d.Camera(), 4, 4, (0.8, 0.4, 0.2, 0.5))
        np.testing.assert_allclose(image[0, 0], (0.4, 0.2, 0.1, 0.5), atol=1e-6)

    def test_card_cube_camera_scene_render_is_real_premultiplied_float_output(self):
        d = graph()
        d.execute({"op": "set", "id": "render", "param": "width", "value": 96})
        d.execute({"op": "set", "id": "render", "param": "height", "value": 64})
        d.execute({"op": "set", "id": "card", "param": "red", "value": 1.0})
        image = Evaluator().evaluate(d.document, target="write")
        self.assertEqual((64, 96, 4), image.shape)
        self.assertEqual(np.float32, image.dtype)
        self.assertFalse(image.flags.writeable)
        self.assertGreater(float(image[..., 3].max()), 0.9)
        self.assertGreater(float(image[..., 0].max()), 0.1)
        self.assertTrue(np.all(image[..., :3] <= image[..., 3:4] + 1e-6))

    def test_depth_and_transform_change_pixels(self):
        d = graph()
        e = Evaluator()
        near = e.evaluate(d.document, target="render")
        d.execute({"op": "set", "id": "cube", "param": "tz", "value": 2.0})
        far = e.evaluate(d.document, target="render")
        self.assertFalse(np.array_equal(near, far))
        d.execute({"op": "set", "id": "cam", "param": "focal", "value": 50.0})
        narrow = e.evaluate(d.document, target="render")
        self.assertFalse(np.array_equal(far, narrow))

    def test_typed_wiring_rejects_without_mutating_document(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "cam", "type": "Camera3D"})
        d.execute({"op": "create", "id": "render", "type": "Render3D"})
        before = d.document
        with self.assertRaisesRegex(ValueError, "expects scene"):
            d.execute({"op": "connect", "id": "render", "input": "scene", "source": "cam"})
        self.assertEqual(before, d.document)

    def test_write_path_receives_render_tree(self):
        d = graph()
        image = Evaluator().evaluate(d.document, target="write")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "foundation.exr")
            write_exr(path, image)
            self.assertGreater(os.path.getsize(path), 128)

    def test_old_document_upgrade_and_undo_redo(self):
        old = {"version": 12, "nodes": {}, "view": None, "time": {"first": 1, "last": 1,
               "current": 1, "fps": 24.0}, "animation": {"curves": {}},
               "settings": {"color": {"config": "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5",
               "working_space": "ACEScg", "display": "sRGB - Display", "view": "sRGB"},
               "viewer": {"background": "black"}}, "node_data": {}, "expressions": {}, "references": []}
        self.assertEqual(upgrade_document(old)["version"], SCHEMA_VERSION)
        d = Dispatcher()
        d.execute({"op": "create", "id": "card", "type": "Card3D"})
        self.assertEqual(d.execute({"op": "undo"})["revision"], 2)
        self.assertEqual(d.execute({"op": "redo"})["revision"], 3)

    def test_editor_navigation_is_local_and_frame_resets(self):
        d = graph()
        widget = Viewport3D()
        widget.set_document(d.document)
        authored = d.document["nodes"]["cam"]["params"].copy()
        widget.azimuth += 40
        widget.distance = 3
        self.assertEqual(authored, d.document["nodes"]["cam"]["params"])
        widget.keyPressEvent(type("Event", (), {"key": lambda self: 0x46})())
        # F frames the scene's bounds: the 2-unit cube at the origin, seen whole at 45 degrees.
        np.testing.assert_allclose(widget.center, (0, 0, 0), atol=1e-6)
        self.assertAlmostEqual(widget.distance, math.sqrt(3) / math.sin(math.radians(22.5)) * 1.1, places=4)
        self.assertEqual((widget.azimuth, widget.elevation), (35.0, 20.0))
        self.assertEqual(authored, d.document["nodes"]["cam"]["params"])


def lit_sphere(light=None, **render_args):
    sphere = scene3d._sphere(1.0, 48, (1, 1, 1, 1), scene3d.Transform3D())
    lights = (light,) if light else ()
    return scene3d.render(scene3d.Scene((sphere,), lights), scene3d.Camera(), 96, 96, **render_args)


class RenderingTests(unittest.TestCase):
    def test_textured_card_is_perspective_correct_and_upright(self):
        # 2x2 texture: top row red|green, bottom row blue|white. Row 0 is the top of the image.
        texture = np.array((((1, 0, 0, 1), (0, 1, 0, 1)), ((0, 0, 1, 1), (1, 1, 1, 1))), np.float32)
        card = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D(), np.repeat(np.repeat(texture, 8, 0), 8, 1))
        image = scene3d.render(scene3d.Scene((card,)), scene3d.Camera(), 64, 64)
        np.testing.assert_allclose(image[24, 24], (1, 0, 0, 1), atol=1e-5)   # top-left stays top-left
        np.testing.assert_allclose(image[24, 40], (0, 1, 0, 1), atol=1e-5)
        np.testing.assert_allclose(image[40, 24], (0, 0, 1, 1), atol=1e-5)
        # Swing the card 60 degrees: the texture's midline (u=0.5) is the card's centre line, which
        # perspective moves off the centre of its screen footprint. Affine UVs would put the
        # red/green boundary at the footprint's middle; correct ones put it at the projected x=0.
        turned = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D(rotation=scene3d.Vec3(0, 60, 0)),
                               np.repeat(np.repeat(texture, 8, 0), 8, 1))
        image = scene3d.render(scene3d.Scene((turned,)), scene3d.Camera(), 256, 256)
        row = image[100]
        covered = np.flatnonzero(row[:, 3] > 0.5)
        boundary = covered[np.argmax(row[covered, 1] > 0.5)]
        self.assertAlmostEqual(boundary, 128, delta=1)
        self.assertGreater(abs((covered[0] + covered[-1]) / 2 - 128), 4)

    def test_texture_alpha_and_tint_stay_premultiplied(self):
        texture = np.full((4, 4, 4), (0.5, 0.25, 0.0, 0.5), np.float32)  # premultiplied half-alpha
        card = scene3d._card(2, 2, (1, 0.5, 1, 0.5), scene3d.Transform3D(), texture)
        image = scene3d.render(scene3d.Scene((card,)), scene3d.Camera(), 32, 32)
        np.testing.assert_allclose(image[16, 16], (0.25, 0.0625, 0, 0.25), atol=1e-6)

    def test_distant_textured_card_uses_a_mip_instead_of_aliasing(self):
        checker = np.indices((64, 64)).sum(0) % 2
        texture = np.repeat(checker[..., None], 4, -1).astype(np.float32); texture[..., 3] = 1
        card = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D(scene3d.Vec3(0, 0, -40)), texture)
        image = scene3d.render(scene3d.Scene((card,)), scene3d.Camera(), 64, 64)
        values = image[..., 0][image[..., 3] > 0.99]
        self.assertGreater(len(values), 0)
        np.testing.assert_allclose(values, 0.5, atol=0.05)   # 1-texel checks average to grey

    def test_lights_shade_lambert_and_unlit_scenes_keep_authored_colour(self):
        unlit = lit_sphere()
        self.assertEqual(float(unlit[..., 0][unlit[..., 3] > 0].min()), 1.0)
        key = scene3d.Light("Directional", (1, 0.5, 0.25), 2.0, scene3d.Vec3(0, 0, 5), scene3d.Vec3())
        lit = lit_sphere(key)
        np.testing.assert_allclose(lit[48, 48], (2, 1, 0.5, 1), atol=0.02)      # facing the light
        # Along the middle row, compare with the analytic answer: ray-sphere hit, N.L = n_z.
        for x in (28, 36, 60, 68):
            ray = np.array(((x + 0.5) / 48 - 1, (48.5 / 48 - 1) * -1, -1 / math.tan(math.radians(22.5))))
            ray[:2] *= 1.0; ray /= np.linalg.norm(ray)
            origin = np.array((0, 0, 5.0)); b = origin @ ray
            hit = origin + ray * (-b - math.sqrt(b * b - (origin @ origin - 1)))
            self.assertAlmostEqual(float(lit[48, x, 0]), 2 * hit[2], delta=0.03)
        ambient = lit_sphere(scene3d.Light(intensity=0.0), ambient=0.3)
        self.assertEqual(float(ambient[48, 48, 0]), 1.0)  # a zero light is no light: still unlit
        side = scene3d.Light("Point", (1, 1, 1), 1.0, scene3d.Vec3(5, 0, 0))
        lit = lit_sphere(side, ambient=0.1)
        self.assertGreater(float(lit[48, 66, 0]), float(lit[48, 30, 0]) + 0.3)
        np.testing.assert_allclose(lit[48, 27, :3], 0.1, atol=0.02)               # ambient only

    def test_near_plane_clips_triangles_instead_of_dropping_them(self):
        ground = scene3d._card(40, 40, (1, 1, 1, 1), scene3d.Transform3D(scene3d.Vec3(0, -1, 0),
                                                                      scene3d.Vec3(-90, 0, 0)))
        image = scene3d.render(scene3d.Scene((ground,)), scene3d.Camera(), 64, 64)
        self.assertEqual(float(image[60, 32, 3]), 1.0)   # floor runs under and behind the camera
        self.assertEqual(float(image[4, 32, 3]), 0.0)    # nothing above the horizon

    def test_supersampling_antialiases_edges_only(self):
        card = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D(rotation=scene3d.Vec3(0, 0, 30)))
        hard = scene3d.render(scene3d.Scene((card,)), scene3d.Camera(), 64, 64)
        soft = scene3d.render(scene3d.Scene((card,)), scene3d.Camera(), 64, 64, samples=4)
        self.assertEqual(set(np.unique(hard[..., 3])), {0.0, 1.0})
        partial = (soft[..., 3] > 0) & (soft[..., 3] < 1)
        self.assertGreater(int(partial.sum()), 20)
        self.assertEqual(float(soft[32, 32, 3]), 1.0)
        self.assertAlmostEqual(float(soft[..., 3].sum()), float(hard[..., 3].sum()), delta=12)

    def test_depth_and_normal_outputs(self):
        card = scene3d._card(2, 2, (1, 1, 1, 1), scene3d.Transform3D(scene3d.Vec3(0, 0, 1)))
        scene = scene3d.Scene((card,))
        depth = scene3d.render(scene, scene3d.Camera(), 32, 32, (1, 1, 1, 1), output="depth", samples=4)
        np.testing.assert_allclose(depth[16, 16], (4, 4, 4, 1), atol=1e-4)
        np.testing.assert_allclose(depth[0, 0], 0)      # background never leaks into a data pass
        normals = scene3d.render(scene, scene3d.Camera(), 32, 32, output="normals")
        np.testing.assert_allclose(normals[16, 16], (0, 0, 1, 1), atol=1e-5)

    def test_camera_roll_rotates_the_frame(self):
        marker = scene3d._card(0.5, 0.5, (1, 1, 1, 1), scene3d.Transform3D(scene3d.Vec3(1, 0, 0)))
        rolled = scene3d.Camera(roll=90.0)
        image = scene3d.render(scene3d.Scene((marker,)), rolled, 64, 64)
        ys, xs = np.nonzero(image[..., 3])
        self.assertAlmostEqual(float(xs.mean()), 31.5, delta=1.5)
        self.assertLess(abs(float(ys.mean()) - 31.5) - 10, 40)
        self.assertGreater(abs(float(ys.mean()) - 31.5), 8)

    def test_obj_import_welds_corners_triangulates_quads_and_rejects_garbage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "quad.obj")
            with open(path, "w") as handle:
                handle.write("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n"
                             "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
                             "f 1/1/1 2/2/1 3/3/1 4/4/1\n")
            d = Dispatcher()
            d.execute({"op": "create", "id": "geo", "type": "ReadGeo3D"})
            d.execute({"op": "set", "id": "geo", "param": "geo_path", "value": path})
            geometry = Evaluator().evaluate_raster(d.document, "geo", typed=True)
            self.assertEqual((geometry.vertices.shape, geometry.triangles.shape), ((4, 3), (2, 3)))
            np.testing.assert_allclose(geometry.uvs[2], (1, 1))
            image = scene3d.render(scene3d.Scene((geometry,)), scene3d.Camera(), 32, 32)
            self.assertEqual(float(image[16, 16, 3]), 1.0)
            with self.assertRaisesRegex(ValueError, "not an image"):
                Evaluator().evaluate(d.document, "geo")
            empty = os.path.join(folder, "empty.obj")
            with open(empty, "w") as handle:
                handle.write("v 0 0 0\n")
            d.execute({"op": "set", "id": "geo", "param": "geo_path", "value": empty})
            with self.assertRaisesRegex(ValueError, "no faces"):
                Evaluator().evaluate_raster(d.document, "geo", typed=True)
            d.execute({"op": "set", "id": "geo", "param": "geo_path", "value": os.path.join(folder, "gone.obj")})
            with self.assertRaisesRegex(ValueError, "cannot read"):
                Evaluator().evaluate_raster(d.document, "geo", typed=True)


class GraphTests(unittest.TestCase):
    def textured(self):
        d = graph()
        d.execute({"op": "create", "id": "plate", "type": "Constant"})
        for name, value in (("width", 16), ("height", 16), ("red", 0.0), ("green", 1.0), ("blue", 0.0)):
            d.execute({"op": "set", "id": "plate", "param": name, "value": value})
        d.execute({"op": "connect", "id": "card", "input": "image", "source": "plate"})
        d.execute({"op": "set", "id": "card", "param": "tz", "value": 2.0})   # in front of the cube
        for name in ("red", "green", "blue"):
            d.execute({"op": "set", "id": "card", "param": name, "value": 1.0})
        d.execute({"op": "set", "id": "render", "param": "samples", "value": 1})
        return d

    def test_image_input_textures_a_card_and_upstream_edits_rerender(self):
        d, e = self.textured(), Evaluator()
        image = e.evaluate(d.document, target="render")
        np.testing.assert_allclose(image[270, 480], (0, 1, 0, 1), atol=1e-5)
        d.execute({"op": "set", "id": "plate", "param": "blue", "value": 1.0})
        np.testing.assert_allclose(e.evaluate(d.document, target="render")[270, 480], (0, 1, 1, 1), atol=1e-5)

    def test_render_is_cached_and_proxy_tiers_shrink_it(self):
        d, e = self.textured(), Evaluator()
        e.evaluate(d.document, target="render")
        misses = e.misses
        e.evaluate(d.document, target="render")
        self.assertEqual(e.misses, misses)
        self.assertEqual(e.evaluate(d.document, target="render", tier=4).shape, (135, 240, 4))

    def test_nested_scenes_inherit_transforms_and_carry_lights(self):
        d = graph()
        for key, kind in (("group", "Scene3D"), ("ball", "Sphere3D"), ("key", "Light3D")):
            d.execute({"op": "create", "id": key, "type": kind})
        d.execute({"op": "connect", "id": "group", "input": "object0", "source": "ball"})
        d.execute({"op": "connect", "id": "group", "input": "object1", "source": "key"})
        d.execute({"op": "connect", "id": "scene", "input": "object7", "source": "group"})
        d.execute({"op": "set", "id": "group", "param": "tx", "value": 3.0})
        d.execute({"op": "set", "id": "scene", "param": "sx", "value": 2.0})
        scene = Evaluator().evaluate_raster(d.document, "scene", typed=True)
        self.assertEqual((len(scene.geometries), len(scene.lights)), (3, 1))
        ball = scene.geometries[-1]
        np.testing.assert_allclose(ball.world_matrix()[:3, 3], (6, 0, 0), atol=1e-6)
        np.testing.assert_allclose(scene.lights[0].world()[0], (10, 4, 3), atol=1e-5)
        with self.assertRaisesRegex(ValueError, "expects geometry or light or scene"):
            d.execute({"op": "connect", "id": "scene", "input": "object2", "source": "cam"})
        with self.assertRaisesRegex(ValueError, "expects image"):
            d.execute({"op": "connect", "id": "card", "input": "image", "source": "cube"})

    def test_disabled_members_leave_the_scene_and_disabled_render_is_empty(self):
        d, e = graph(), Evaluator()
        d.document["nodes"]["cube"]["disabled"] = True
        self.assertEqual(len(e.evaluate_raster(d.document, "scene", typed=True).geometries), 1)
        d.document["nodes"]["render"]["disabled"] = True
        self.assertEqual(float(np.abs(e.evaluate(d.document, target="render")).max()), 0.0)

    def test_animated_transform_renders_per_frame(self):
        d, e = graph(), Evaluator()
        d.document["nodes"]["card"]["disabled"] = True
        first = e.evaluate(d.document, target="render", frame=1)
        d.document["animation"]["curves"]["cube"] = {"tx": {"keys": [
            {"frame": 1, "value": 0.0, "interpolation": "linear"},
            {"frame": 10, "value": 1.5, "interpolation": "linear"}]}}
        np.testing.assert_array_equal(first, e.evaluate(d.document, target="render", frame=1))
        self.assertFalse(np.array_equal(first, e.evaluate(d.document, target="render", frame=10)))

    def test_viewport_shows_what_render3d_will_see(self):
        d = self.textured()
        widget = Viewport3D()
        widget.set_document(d.document)
        scene, camera = widget._evaluated()
        self.assertEqual(len(scene.geometries), 2)
        self.assertEqual(scene.geometries[0].texture.shape, (4, 4, 4))   # proxy-tier plate
        self.assertEqual(camera, scene3d.camera_from_node(d.document["nodes"]["cam"]))
        widget.look_through = True
        self.assertEqual(widget._camera(camera), camera)
        widget.resize(320, 220)
        self.assertFalse(widget.grab().toImage().isNull())
        self.assertEqual(widget.status, "")


if __name__ == "__main__":
    unittest.main()
