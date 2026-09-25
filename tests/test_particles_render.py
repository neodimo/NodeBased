"""Lane L5 step 2a, part 3: particles drawn as size-scaled discs by the CPU raster path of Render3D."""
import math
import unittest

import numpy as np

from nodebased import gpu3d, scene3d
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

SIZE = 200


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def graph(*emitters, cam=None, **extra):
    """A Render3D of one Scene3D holding the given (key, params) emitters and extra nodes."""
    d = Dispatcher()
    make(d, s=("Scene3D", {}), cam=("Camera3D", cam or {}),
         r=("Render3D", {"width": SIZE, "height": SIZE, "samples": 1}))
    wire(d, "r", "scene", "s")
    wire(d, "r", "camera", "cam")
    for slot, (key, params) in enumerate(emitters):
        make(d, **{key: ("ParticleEmitter3D", {"emit_speed": 0.0, "life": 1000.0, **params})})
        wire(d, "s", f"object{slot}", key)
    for slot, (key, (kind, params)) in enumerate(extra.items(), start=len(emitters)):
        make(d, **{key: (kind, params)})
        wire(d, "s", f"object{slot}", key)
    return d


def render(d, frame=1, key="r"):
    return Evaluator().evaluate(dict(d.document, view=key), frame=frame)


def pixel_of(point, camera=None):
    pixels, _ = scene3d.project(camera or scene3d.Camera(), SIZE, SIZE, [point])
    return int(pixels[0, 1]), int(pixels[0, 0])          # row, column


class PointRenderTests(unittest.TestCase):
    def test_a_rendered_frame_has_pixels_where_the_particle_is_and_background_elsewhere(self):
        d = graph(("e", {"emit_rate": 1.0, "tx": 0.5, "particle_size": 0.2, "green": 0.5, "blue": 0.25}))
        image = render(d)
        row, col = pixel_of((0.5, 0.0, 0.0))
        np.testing.assert_allclose(image[row, col], [1.0, 0.5, 0.25, 1.0], atol=1e-6)
        alpha = image[..., 3]
        self.assertEqual(float(alpha[0, 0] + alpha[-1, -1] + alpha[0, -1] + alpha[-1, 0]), 0.0)
        ys, xs = np.nonzero(alpha)
        self.assertLess(abs(ys.mean() - row), 1.0)
        self.assertLess(abs(xs.mean() - col), 1.0)
        # The disc is centred on the projected point.
        self.assertEqual(image.shape, (SIZE, SIZE, 4))

    def test_the_disc_radius_follows_the_size_and_the_distance(self):
        focal = 1.0 / math.tan(math.radians(45.0) / 2)
        for size in (0.2, 0.4, 0.8):
            image = render(graph(("e", {"emit_rate": 1.0, "particle_size": size})))
            area = int((image[..., 3] > 0).sum())
            radius = 0.25 * size * focal * SIZE / 5.0
            self.assertAlmostEqual(area / (math.pi * radius ** 2), 1.0, delta=0.12, msg=size)
        near = int((render(graph(("e", {"emit_rate": 1.0, "particle_size": 0.4, "tz": 2.5})))[..., 3] > 0).sum())
        far = int((render(graph(("e", {"emit_rate": 1.0, "particle_size": 0.4, "tz": -5.0})))[..., 3] > 0).sum())
        self.assertAlmostEqual(near / far, (10.0 / 2.5) ** 2, delta=1.5)          # (z ratio) squared

    def test_every_particle_lands_on_its_projected_pixel(self):
        d = graph(("e", {"emit_rate": 40.0, "emit_speed": 0.05, "spread": 60.0, "particle_size": 0.06}))
        frame = 3
        value = Evaluator().evaluate_raster(d.document, "e", frame=frame, typed=True)
        self.assertEqual(len(value), 120)
        image = render(d, frame=frame)
        pixels, _ = scene3d.project(scene3d.Camera(), SIZE, SIZE, value.positions)
        rows, cols = pixels[:, 1].astype(int), pixels[:, 0].astype(int)
        self.assertTrue(((rows >= 0) & (rows < SIZE) & (cols >= 0) & (cols < SIZE)).all())
        self.assertTrue((image[rows, cols, 3] > 0).all())
        self.assertLess(int((image[..., 3] > 0).sum()), 120 * 30)                  # small discs, not a wash

    def test_particles_move_between_frames(self):
        d = graph(("e", {"emit_rate": 1.0, "emit_speed": 0.5, "emit_dir_x": 1.0, "emit_dir_y": 0.0,
                         "particle_size": 0.1}))
        a, b = render(d, 2), render(d, 4)
        self.assertFalse(np.array_equal(a, b))
        cols_a = np.nonzero(a[..., 3])[1].mean()
        cols_b = np.nonzero(b[..., 3])[1].mean()
        self.assertGreater(cols_b, cols_a)

    def test_a_particle_under_an_axis_moves_with_it(self):
        d = graph(("e", {"emit_rate": 1.0, "particle_size": 0.1}))
        d2 = Dispatcher()
        make(d2, e=("ParticleEmitter3D", {"emit_rate": 1.0, "particle_size": 0.1, "emit_speed": 0.0,
                                          "life": 1000.0}),
             ax=("Axis3D", {"tx": 1.0}), s=("Scene3D", {}), cam=("Camera3D", {}),
             r=("Render3D", {"width": SIZE, "height": SIZE, "samples": 1}))
        wire(d2, "ax", "object", "e")
        wire(d2, "s", "object0", "ax")
        wire(d2, "r", "scene", "s")
        wire(d2, "r", "camera", "cam")
        row, col = pixel_of((1.0, 0.0, 0.0))
        image = render(d2)
        self.assertGreater(image[row, col, 3], 0.99)
        self.assertEqual(float(render(d)[row, col, 3]), 0.0)

    def test_translucent_particles_composite_over_each_other(self):
        one = render(graph(("e", {"emit_rate": 1.0, "alpha": 0.5, "particle_size": 0.3})))
        two = render(graph(("e", {"emit_rate": 2.0, "alpha": 0.5, "particle_size": 0.3})))
        row, col = pixel_of((0.0, 0.0, 0.0))
        np.testing.assert_allclose(one[row, col], [0.5, 0.5, 0.5, 0.5], atol=1e-6)     # premultiplied
        np.testing.assert_allclose(two[row, col], [0.75, 0.75, 0.75, 0.75], atol=1e-6)  # 1 - 0.5 * 0.5

    def test_nearer_particles_cover_farther_ones_whatever_the_slot_order(self):
        red = ("red", {"emit_rate": 1.0, "particle_size": 0.4, "green": 0.0, "blue": 0.0, "tz": 0.0})
        green = ("green", {"emit_rate": 1.0, "particle_size": 0.4, "red": 0.0, "blue": 0.0, "tz": 1.0})
        row, col = pixel_of((0.0, 0.0, 0.0))
        for order in ((red, green), (green, red)):
            image = render(graph(*order))
            np.testing.assert_allclose(image[row, col], [0.0, 1.0, 0.0, 1.0], atol=1e-6)

    def test_meshes_occlude_particles_behind_them_and_particles_show_in_front(self):
        card = {"card_width": 4.0, "card_height": 4.0, "red": 0.0, "green": 0.0, "blue": 1.0}
        row, col = pixel_of((0.0, 0.0, 0.0))
        behind = render(graph(("e", {"emit_rate": 1.0, "particle_size": 0.5, "tz": -1.0}),
                              card=("Card3D", card)))
        np.testing.assert_allclose(behind[row, col], [0.0, 0.0, 1.0, 1.0], atol=1e-6)
        front = render(graph(("e", {"emit_rate": 1.0, "particle_size": 0.5, "tz": 1.0}),
                             card=("Card3D", card)))
        np.testing.assert_allclose(front[row, col], [1.0, 1.0, 1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(front[row, col + 60], [0.0, 0.0, 1.0, 1.0], atol=1e-6)   # card still there

    def test_antialiasing_softens_the_disc_edge(self):
        d = graph(("e", {"emit_rate": 1.0, "particle_size": 0.3}))
        d.execute({"op": "set", "id": "r", "param": "samples", "value": 2})
        alpha = render(d)[..., 3]
        self.assertGreater(int(((alpha > 0) & (alpha < 1)).sum()), 8)
        self.assertGreater(float(alpha.max()), 0.99)

    def test_particles_behind_or_beside_the_camera_and_empty_sets_draw_nothing(self):
        for params in ({"tz": 8.0}, {"tx": 500.0}, {"emit_rate": 0.0}):
            image = render(graph(("e", {"emit_rate": 1.0, "particle_size": 0.3, **params})))
            self.assertEqual(float(image[..., 3].max()), 0.0, params)

    def test_a_cache_node_renders_the_same_frame_as_its_emitter(self):
        d = graph(("e", {"emit_rate": 5.0, "emit_speed": 0.05, "spread": 50.0, "particle_size": 0.08}))
        direct = render(d, 6)
        make(d, c=("ParticleCache3D", {}))
        wire(d, "c", "particles", "e")
        wire(d, "s", "object0", "c")                       # the cache node now feeds the scene
        self.assertGreater(float(direct[..., 3].max()), 0.0)
        np.testing.assert_array_equal(render(d, 6), direct)

    def test_scenes_without_particles_render_exactly_as_before(self):
        card = scene3d._card(2.0, 2.0, (0.8, 0.2, 0.2, 1.0), scene3d.Transform3D(), None)
        plain = scene3d.render(scene3d.Scene((card,)), scene3d.Camera(), 64, 48, (0, 0, 0, 0))
        again = scene3d.render(scene3d.Scene((card,), (), (), ()), scene3d.Camera(), 64, 48, (0, 0, 0, 0))
        np.testing.assert_array_equal(plain, again)

    def test_data_outputs_ignore_particles(self):
        d = graph(("e", {"emit_rate": 1.0, "particle_size": 0.4}))
        d.execute({"op": "set", "id": "r", "param": "render_output", "value": "depth"})
        self.assertEqual(float(render(d)[..., 3].max()), 0.0)


class BackendTests(unittest.TestCase):
    def test_the_gpu_renderer_refuses_particle_scenes_and_auto_falls_back_to_the_cpu(self):
        d = graph(("e", {"emit_rate": 1.0, "particle_size": 0.3}))
        scene = Evaluator().evaluate_raster(d.document, "s", typed=True)
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, scene3d.Camera(), 32, 32)
        cpu = render(d)
        d.execute({"op": "set", "id": "r", "param": "render_backend", "value": "auto"})
        np.testing.assert_array_equal(render(d), cpu)
        d.execute({"op": "set", "id": "r", "param": "render_backend", "value": "gpu"})
        with self.assertRaises(ValueError):
            render(d)


if __name__ == "__main__":
    unittest.main()
