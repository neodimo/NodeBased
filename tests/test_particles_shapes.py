"""Lane L5 step 2c, part 2: particles drawn as shaded spheres and camera-facing cards.

ParticleRender3D picks the representation, scales the size and can texture the cards; it changes how
the solved frame is drawn and never the solve (no re-solve, same run).
"""
import math
import unittest

import numpy as np

from nodebased import particles, scene3d
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

SIZE = 200
FOCAL = 1.0 / math.tan(math.radians(45.0) / 2)


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


def graph(representation="spheres", emitter=None, render=None, cam=None, image=None, extra=()):
    """Render3D of a Scene3D holding one resting particle at the origin, drawn by a ParticleRender3D."""
    d = Dispatcher()
    make(d, s=("Scene3D", {}), cam=("Camera3D", cam or {}),
         r=("Render3D", {"width": SIZE, "height": SIZE, "samples": 1}),
         e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.0, "life": 1000.0, "particle_size": 0.4,
                                 **(emitter or {})}),
         p=("ParticleRender3D", {"representation": representation, **(render or {})}))
    wire(d, "r", "scene", "s")
    wire(d, "r", "camera", "cam")
    wire(d, "p", "particles", "e")
    wire(d, "s", "object0", "p")
    if image is not None:
        make(d, img=image)
        wire(d, "p", "image", "img")
    for slot, (key, spec) in enumerate(extra, start=1):
        make(d, **{key: spec})
        wire(d, "s", f"object{slot}", key)
    return d


def render(d, frame=1, evaluator=None):
    return (evaluator or Evaluator()).evaluate(dict(d.document, view="r"), frame=frame)


def covered(image):
    return image[..., 3] > 0


class SphereTests(unittest.TestCase):
    def test_a_sphere_is_a_disc_of_the_expected_radius_at_a_known_camera(self):
        for size in (0.4, 0.8):
            mask = covered(render(graph(emitter={"particle_size": size})))
            radius = 0.25 * size * FOCAL * SIZE / 5.0                 # world radius size / 2 at distance 5
            self.assertAlmostEqual(mask.sum() / (math.pi * radius ** 2), 1.0, delta=0.06, msg=size)
            ys, xs = np.nonzero(mask)
            self.assertAlmostEqual((xs.max() - xs.min() + 1) / 2.0, radius, delta=1.0)
            self.assertAlmostEqual((ys.max() - ys.min() + 1) / 2.0, radius, delta=1.0)

    def test_a_sphere_is_shaded_lit_on_the_light_side_and_darker_away(self):
        image = render(graph(emitter={"particle_size": 0.8}))
        mask = covered(image)
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        radius = 0.5 * (xs.max() - xs.min() + 1)
        lit = image[int(cy - 0.5 * radius), int(cx - 0.4 * radius), 0]      # towards the upper left light
        dark = image[int(cy + 0.5 * radius), int(cx + 0.4 * radius), 0]
        self.assertGreater(lit, dark + 0.3)
        self.assertGreater(lit, 0.8)
        self.assertGreater(dark, 0.2)                                       # the 0.25 ambient floor
        self.assertLess(dark, 0.45)
        # A flat disc is one value; a sphere varies a lot over its face.
        values = image[..., 0][mask]
        self.assertGreater(values.max() - values.min(), 0.5)
        np.testing.assert_allclose(image[..., 3][mask], 1.0)

    def test_the_sphere_takes_the_particle_colour_per_particle(self):
        camera = scene3d.Camera()
        colours = np.array([[1, 0, 0, 1], [0, 0, 1, 1]], np.float32)
        instance = scene3d.ParticleInstance(positions=np.array([[-0.6, 0, 0], [0.6, 0, 0]], np.float32),
                                            sizes=np.array([0.5, 0.5], np.float32), colors=colours,
                                            render_as="spheres")
        image = scene3d.render(scene3d.Scene(particles=(instance,)), camera, SIZE, SIZE)
        pixels, _ = scene3d.project(camera, SIZE, SIZE, instance.positions)
        left, right = (int(pixels[0, 1]), int(pixels[0, 0])), (int(pixels[1, 1]), int(pixels[1, 0]))
        self.assertGreater(image[left][0], 0.3)
        self.assertEqual(float(image[left][2]), 0.0)
        self.assertGreater(image[right][2], 0.3)
        self.assertEqual(float(image[right][0]), 0.0)

    def test_a_sphere_is_occluded_by_a_nearer_mesh_and_shows_in_front_of_a_farther_one(self):
        card = ("card", ("Card3D", {"card_width": 4.0, "card_height": 4.0, "red": 0.0, "green": 0.0,
                                     "blue": 1.0, "tz": 1.0}))
        behind = render(graph(emitter={"tz": 0.0}, extra=[card]))
        np.testing.assert_allclose(behind[SIZE // 2, SIZE // 2], [0.0, 0.0, 1.0, 1.0], atol=1e-6)
        front = render(graph(emitter={"tz": 2.0}, extra=[card]))
        self.assertGreater(front[SIZE // 2, SIZE // 2, 0], 0.2)              # white-ish sphere over the card

    def test_intersecting_spheres_show_the_nearer_surface_not_the_nearer_centre(self):
        camera = scene3d.Camera()
        big = scene3d.ParticleInstance(positions=np.array([[0, 0, 0.0]], np.float32),
                                       sizes=np.array([2.0], np.float32),
                                       colors=np.array([[1, 0, 0, 1]], np.float32), render_as="spheres")
        small = scene3d.ParticleInstance(positions=np.array([[0, 0, 0.9]], np.float32),
                                         sizes=np.array([0.05], np.float32),
                                         colors=np.array([[0, 1, 0, 1]], np.float32), render_as="spheres")
        # The small sphere sits in front of the big one's centre plane but inside its surface bulge.
        image = scene3d.render(scene3d.Scene(particles=(big, small)), camera, SIZE, SIZE)
        self.assertGreater(image[SIZE // 2, SIZE // 2, 1], 0.2)


class CardTests(unittest.TestCase):
    def check_full_square(self, image, distance, size):
        mask = covered(image)
        half = 0.25 * size * FOCAL * SIZE / distance
        ys, xs = np.nonzero(mask)
        width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        self.assertAlmostEqual(width, 2 * half, delta=1.5)
        self.assertAlmostEqual(height, 2 * half, delta=1.5)
        self.assertEqual(int(mask.sum()), int(width * height))               # every pixel of the box is filled
        return width

    def test_a_card_is_a_full_square_of_the_expected_size(self):
        for size in (0.4, 0.8):
            self.check_full_square(render(graph("cards", emitter={"particle_size": size})), 5.0, size)

    def test_cards_face_the_camera_from_any_direction_and_roll(self):
        angle = math.radians(40.0)
        cams = [{"tx": 5.0 * math.sin(angle), "tz": 5.0 * math.cos(angle)},                  # orbited 40 degrees
                {"tx": 0.0, "ty": 4.0, "tz": 3.0},                                            # from above
                {"roll": 30.0}]                                                               # rolled about the view
        for cam in cams:
            distance = math.sqrt(cam.get("tx", 0.0) ** 2 + cam.get("ty", 0.0) ** 2 + cam.get("tz", 5.0) ** 2)
            self.check_full_square(render(graph("cards", cam=cam)), distance, 0.4)

    def test_size_scale_multiplies_the_drawn_size_without_changing_the_solve(self):
        def area(representation, scale):
            d = graph(representation, emitter={"particle_size": 1.0}, render={"size_scale": scale})
            return covered(render(d)).sum()
        for representation in ("points", "spheres", "cards"):
            one = area(representation, 1.0)
            self.assertAlmostEqual(area(representation, 2.0) / one, 4.0, delta=0.15, msg=representation)
            self.assertAlmostEqual(area(representation, 0.5) / one, 0.25, delta=0.02, msg=representation)

    def test_the_sprite_image_shows_on_the_card(self):
        texture = np.zeros((2, 2, 4), np.float32)
        texture[0, 0] = (1, 0, 0, 1)
        texture[0, 1] = (0, 1, 0, 1)
        texture[1, 0] = (0, 0, 1, 1)
        texture[1, 1] = (1, 1, 1, 1)
        camera = scene3d.Camera()
        instance = scene3d.ParticleInstance(positions=np.zeros((1, 3), np.float32),
                                            sizes=np.array([2.0], np.float32),
                                            colors=np.ones((1, 4), np.float32), render_as="cards",
                                            texture=texture)
        image = scene3d.render(scene3d.Scene(particles=(instance,)), camera, SIZE, SIZE)
        c, o = SIZE // 2, int(0.25 * 2.0 * FOCAL * SIZE / 5.0 * 0.5)
        np.testing.assert_allclose(image[c - o, c - o], (1, 0, 0, 1), atol=1e-6)      # top left
        np.testing.assert_allclose(image[c - o, c + o], (0, 1, 0, 1), atol=1e-6)      # top right
        np.testing.assert_allclose(image[c + o, c - o], (0, 0, 1, 1), atol=1e-6)      # bottom left
        np.testing.assert_allclose(image[c + o, c + o], (1, 1, 1, 1), atol=1e-6)      # bottom right

    def test_an_image_input_textures_the_cards_and_is_tinted_by_the_particle_colour(self):
        image_node = ("Constant", {"width": 8, "height": 8, "red": 0.5, "green": 0.25, "blue": 1.0, "alpha": 1.0})
        plain = render(graph("cards", image=image_node))
        centre = plain[SIZE // 2, SIZE // 2]
        self.assertGreater(abs(float(centre[0]) - float(centre[1])), 0.1)              # the constant, not white
        self.assertAlmostEqual(float(centre[2]) / float(centre[0]), 2.0, delta=0.15)
        tinted = render(graph("cards", image=image_node, emitter={"red": 0.0}))
        self.assertLess(float(tinted[SIZE // 2, SIZE // 2, 0]), 1e-6)                   # red particle channel is zero
        untextured = render(graph("cards"))
        np.testing.assert_allclose(untextured[SIZE // 2, SIZE // 2], [1, 1, 1, 1], atol=1e-6)

    def test_a_transparent_sprite_pixel_leaves_what_is_behind(self):
        texture = np.zeros((2, 2, 4), np.float32)
        texture[:, 0] = (1, 1, 1, 1)                                                    # left half opaque white
        camera = scene3d.Camera()
        instance = scene3d.ParticleInstance(positions=np.zeros((1, 3), np.float32),
                                            sizes=np.array([2.0], np.float32),
                                            colors=np.ones((1, 4), np.float32), render_as="cards",
                                            texture=texture)
        image = scene3d.render(scene3d.Scene(particles=(instance,)), camera, SIZE, SIZE,
                               background=(0.0, 0.0, 0.0, 0.0))
        o = int(0.25 * 2.0 * FOCAL * SIZE / 5.0 * 0.5)
        self.assertEqual(float(image[SIZE // 2, SIZE // 2 + o, 3]), 0.0)
        self.assertEqual(float(image[SIZE // 2, SIZE // 2 - o, 3]), 1.0)


class NodeBehaviourTests(unittest.TestCase):
    def test_points_are_unchanged_and_the_default_representation_is_points(self):
        a = render(graph("points"))
        b = Evaluator().evaluate(dict(graph("points").document, view="r"), frame=1)
        np.testing.assert_array_equal(a, b)
        d = graph("points")
        d.document["nodes"]["p"]["disabled"] = True
        np.testing.assert_array_equal(render(d), a)

    def test_a_bypassed_render_node_draws_points(self):
        d = graph("cards")
        d.document["nodes"]["p"]["disabled"] = True
        flat = covered(render(d))
        ys, xs = np.nonzero(flat)
        radius = 0.25 * 0.4 * FOCAL * SIZE / 5.0
        self.assertAlmostEqual(flat.sum() / (math.pi * radius ** 2), 1.0, delta=0.12)     # a disc, not a square

    def test_changing_the_representation_does_not_re_solve_or_change_the_run(self):
        d = graph("points", emitter={"emit_rate": 5.0, "emit_speed": 0.1, "spread": 40.0})
        make(d, c=("ParticleCache3D", {}))
        d.document["nodes"]["p"]["inputs"]["particles"] = "c"
        wire(d, "c", "particles", "e")
        ev = Evaluator()
        first = render(d, frame=6, evaluator=ev)
        run = ev.evaluate_raster(d.document, "p", frame=6, typed=True).stream.run
        steps = particles.SOLVER_STATS["steps"]
        for representation in ("spheres", "cards"):
            d.execute({"op": "set", "id": "p", "param": "representation", "value": representation})
            other = render(d, frame=6, evaluator=ev)
            self.assertFalse(np.array_equal(first, other))
            self.assertEqual(ev.evaluate_raster(d.document, "p", frame=6, typed=True).stream.run, run)
        self.assertEqual(particles.SOLVER_STATS["steps"], steps)

    def test_the_representation_survives_a_cache_or_force_after_it(self):
        d = graph("spheres")
        make(d, c=("ParticleCache3D", {}), f=("ParticleDrag3D", {}))
        wire(d, "f", "particles", "p")
        wire(d, "c", "particles", "f")
        d.document["nodes"]["s"]["inputs"]["object0"] = "c"
        self.assertEqual(ev := Evaluator().evaluate_raster(d.document, "c", frame=3, typed=True).render_as, "spheres")
        del ev

    def test_a_sphere_scene_renders_with_bounce_and_is_deterministic(self):
        d = graph("spheres", emitter={"emit_rate": 8.0, "emit_speed": 0.3, "spread": 60.0, "ty": 0.5, "seed": 5,
                                      "particle_size": 0.15})
        make(d, g=("ParticleGravity3D", {"strength": 0.03}), fl=("Card3D", {"card_width": 8.0, "card_height": 8.0,
                                                                             "rx": -90.0, "ty": -0.5}),
             b=("ParticleBounce3D", {"bounce": 0.5}))
        wire(d, "g", "particles", "e")
        wire(d, "b", "particles", "g")
        wire(d, "b", "geometry", "fl")
        wire(d, "p", "particles", "b")
        a, b = render(d, frame=25), render(d, frame=25)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(int(covered(a).sum()), 200)


if __name__ == "__main__":
    unittest.main()
