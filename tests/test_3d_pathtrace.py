"""Path tracer (lane L4 step R3 of 7): the CPU reference, then the GPU against it.

docs/3D_FOUNDATION.md "Path tracing". The CPU tests use tiny images so a whole file runs in seconds; the
GPU tests are guarded by `gpu3d.available()` like every other GPU test.
"""
import dataclasses
import math
import threading
import unittest

import numpy as np

from nodebased import core, gpu3d, pathtrace as pt, scene3d as s
from nodebased.cancellation import Cancelled
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.envlight import Environment, fingerprint_of

PBR = dict(material="pbr")


def uniform_env(value=1.0):
    rgb = np.full((16, 32, 3), value, np.float32)
    return Environment(rgb, fingerprint_of(rgb))


def sun_env(value=0.0, spot=50.0, y=6, x=8):
    rgb = np.full((16, 32, 3), value, np.float32)
    rgb[y, x] = spot
    return Environment(rgb, fingerprint_of(rgb))


def geometry(shape, color=(1, 1, 1, 1), **fields):
    return dataclasses.replace(shape, color=color, **fields)


def sphere(radius=1.0, color=(1, 1, 1, 1), position=(0, 0, 0), **fields):
    g = s._sphere_grid(radius, 16, 32, color, s.Transform3D(position=s.Vec3(*position)))
    return dataclasses.replace(g, **fields)


def card(width, height, color, position, rotation=(0, 0, 0), **fields):
    g = s._card(width, height, color, s.Transform3D(position=s.Vec3(*position), rotation=s.Vec3(*rotation)))
    return dataclasses.replace(g, **fields)


def box(size, color, position, **fields):
    """Closed box with outward per-face normals; `size` is (x, y, z)."""
    sx, sy, sz = (float(v) / 2 for v in size)
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    vertices, triangles, normals = [], [], []
    half = np.array((sx, sy, sz))
    for n, u, v in faces:
        n, u, v = (np.array(a, float) for a in (n, u, v))
        base = len(vertices)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append((n + su * u + sv * v) * half)
            normals.append(n)
        triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    tri = np.array(triangles, np.int32)
    verts = np.array(vertices, np.float32)
    # flip winding where it disagrees with the outward normal
    for i, t in enumerate(tri):
        a, b, c = verts[t]
        if np.dot(np.cross(b - a, c - a), np.array(normals[t[0]])) < 0:
            tri[i] = t[[0, 2, 1]]
    g = s.Geometry(verts, tri, color, s.Transform3D(position=s.Vec3(*position)), normals=np.array(normals, np.float32))
    return dataclasses.replace(g, **fields)


def cornell(left=(0.9, 0.1, 0.1), light_intensity=4.0, lights=None):
    """A 2 x 2 x 2 room open toward the camera: red left wall, white everywhere else, a rect light in the ceiling."""
    white = (0.8, 0.8, 0.8, 1)
    walls = (card(2, 2, (*left, 1), (-1, 0, 0), (0, 90, 0)),
             card(2, 2, (0.1, 0.8, 0.1, 1), (1, 0, 0), (0, 90, 0)),
             card(2, 2, white, (0, -1, 0), (90, 0, 0)),
             card(2, 2, white, (0, 1, 0), (90, 0, 0)),
             card(2, 2, white, (0, 0, -1), (0, 0, 0)))
    light = s.Light("Rect", (1, 1, 1), light_intensity, s.Vec3(0, 0.97, 0), s.Vec3(0, -1, 0),
                    area_width=0.6, area_height=0.6, light_samples=1)
    return s.Scene(walls, lights=lights if lights is not None else (light,))


CORNELL_CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 3.6)), s.Vec3(0, 0, 0), 45.0)
FRONT = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0), 30.0)


def trace(scene, camera=FRONT, size=(16, 16), samples=32, output="rgba", **settings):
    settings.setdefault("max_bounces", 8)
    return pt.render(scene, camera, size[0], size[1], (0, 0, 0, 0), settings.pop("ambient", 0.0), output,
                     pt.PathSettings(samples=samples, **settings))


def center(img, half=2):
    h, w = img.shape[:2]
    return img[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half, :3]


class RandomNumberTests(unittest.TestCase):
    def test_uniform_and_reproducible(self):
        keys = pt.path_key(np.arange(4096), 3, 7)
        a, b = pt.rand(keys, 5), pt.rand(keys, 5)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((a >= 0) & (a < 1)))
        self.assertAlmostEqual(float(a.mean()), 0.5, delta=0.02)
        self.assertAlmostEqual(float(np.corrcoef(a, pt.rand(keys, 6))[0, 1]), 0.0, delta=0.05)
        self.assertNotEqual(float(pt.rand(pt.path_key(1, 3, 7), 5)), float(pt.rand(pt.path_key(1, 3, 8), 5)))


class FurnaceTests(unittest.TestCase):
    """A closed object under a uniform sky of radiance 1 that loses no light must show 1."""

    def test_rough_dielectric_neither_gains_nor_loses(self):
        for roughness in (0.15, 0.5, 1.0):
            with self.subTest(roughness=roughness):
                scene = s.Scene((sphere(**PBR, metallic=0.0, pbr_roughness=roughness),), environments=(uniform_env(),))
                value = center(trace(scene, samples=96)).mean()
                self.assertAlmostEqual(float(value), 1.0, delta=0.04)

    def test_diffuse_under_bounces(self):
        scene = s.Scene((sphere(),), environments=(uniform_env(),))
        self.assertAlmostEqual(float(center(trace(scene, samples=64)).mean()), 1.0, delta=0.03)

    def test_a_grey_sphere_shows_its_albedo(self):
        scene = s.Scene((sphere(color=(0.5, 0.5, 0.5, 1)),), environments=(uniform_env(),))
        self.assertAlmostEqual(float(center(trace(scene, samples=64)).mean()), 0.5, delta=0.03)

    def test_furnace_with_interreflection(self):
        # two white spheres in a white furnace: every bounce between them still adds up to exactly 1
        scene = s.Scene((sphere(0.7, position=(-0.75, 0, 0)), sphere(0.7, position=(0.75, 0, 0))),
                        environments=(uniform_env(),))
        img = trace(scene, size=(24, 12), samples=64, max_bounces=16, diffuse_bounces=16)
        mask = img[..., 3] > 0.99
        self.assertGreater(int(mask.sum()), 20)
        self.assertAlmostEqual(float(img[..., :3][mask].mean()), 1.0, delta=0.04)


class BounceTests(unittest.TestCase):
    def test_colour_bleeds_from_a_red_wall(self):
        scene = cornell()
        direct = trace(scene, CORNELL_CAMERA, (24, 24), 48, max_bounces=1)
        bounced = trace(scene, CORNELL_CAMERA, (24, 24), 48, max_bounces=4)
        # the back wall, next to the red one: its left third against its right third
        left, right = bounced[8:16, 7:10, :3].mean(axis=(0, 1)), bounced[8:16, 14:17, :3].mean(axis=(0, 1))
        self.assertGreater(left[0] - left[1], 0.02)          # red tint on the wall beside the red wall
        self.assertGreater(left[0] / left[1], right[0] / right[1] * 1.15)
        # without indirect light the white back wall is not tinted by the red one
        d_left = direct[8:16, 7:10, :3].mean(axis=(0, 1))
        self.assertAlmostEqual(float(d_left[0] / d_left[1]), 1.0, delta=0.03)
        self.assertGreater(float(bounced.mean()), float(direct.mean()))


class ConvergenceTests(unittest.TestCase):
    def test_error_falls_as_one_over_root_samples(self):
        scene = cornell()
        size = (12, 12)
        reference = trace(scene, CORNELL_CAMERA, size, 1024, seed=99)
        errors = {}
        for samples in (4, 16, 64):
            runs = [trace(scene, CORNELL_CAMERA, size, samples, seed=seed) for seed in (1, 2, 3)]
            errors[samples] = float(np.mean([np.sqrt(np.mean((r - reference) ** 2)) for r in runs]))
        # sixteen times the samples is four times less noise (the reference's own noise adds a floor)
        ratio = errors[4] / errors[64]
        self.assertGreater(ratio, 2.6, errors)
        self.assertLess(ratio, 6.0, errors)
        self.assertLess(errors[64], errors[16])
        self.assertLess(errors[16], errors[4])


class DirectLightingTests(unittest.TestCase):
    """`max_bounces` 1 is direct light: it agrees with the ray-traced renderer's own direct lighting."""

    def _scene(self, **material):
        floor = card(6, 6, (0.7, 0.7, 0.7, 1), (0, -1, 0), (-90, 0, 0))
        ball = sphere(0.8, (0.8, 0.4, 0.2, 1), **material)
        lights = (s.Light("Directional", (1, 1, 1), 0.8, s.Vec3(2, 4, 3), s.Vec3(0, 0, 0), shadows=True),
                  s.Light("Point", (1, 0.9, 0.7), 0.6, s.Vec3(-2, 2, 2), s.Vec3(0, 0, 0), shadows=True))
        return s.Scene((floor, ball), lights=lights)

    def _compare(self, scene, tolerance):
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 1.5, 4.5)), s.Vec3(0, -0.3, 0), 40.0)
        reference = s.render(scene, camera, 32, 24, (0, 0, 0, 0), ambient=0.0, samples=2, mode="raytrace")
        traced = trace(scene, camera, (32, 24), 24, max_bounces=1)
        both = (reference[..., 3] > 0.99) & (traced[..., 3] > 0.99)
        self.assertGreater(int(both.sum()), 400)
        a, b = reference[..., :3][both], traced[..., :3][both]
        self.assertLess(abs(float(a.mean()) - float(b.mean())) / float(a.mean()), tolerance)
        self.assertLess(float(np.abs(a - b).mean()) / float(a.mean()), tolerance * 2.5)

    def test_lambert_lights_and_shadows(self):
        self._compare(self._scene(), 0.03)

    def test_pbr_dielectric_lights_and_shadows(self):
        self._compare(self._scene(material="pbr", metallic=0.0, pbr_roughness=0.4, pbr_specular=0.5), 0.05)


class MirrorAndGlassTests(unittest.TestCase):
    def test_a_mirror_shows_the_sky_tinted_by_its_colour(self):
        mirror = card(8, 8, (0.8, 0.5, 0.2, 1), (0, 0, 0), material="pbr", metallic=1.0, pbr_roughness=0.0)
        img = trace(s.Scene((mirror,), environments=(uniform_env(),)), samples=8)
        np.testing.assert_allclose(center(img).reshape(-1, 3).mean(0), (0.8, 0.5, 0.2), atol=0.03)

    def test_a_mirror_places_a_reflected_light_where_optics_says(self):
        mirror = card(8, 8, (1, 1, 1, 1), (0, 0, 0), material="pbr", metallic=1.0, pbr_roughness=0.0)
        # a red emitter 6 behind the camera: the mirror shows it straight ahead, 10 away
        lamp = card(1, 1, (1, 0, 0, 1), (0, 0, 6), emission=2.0)
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0), 30.0)
        img = trace(s.Scene((mirror, lamp)), camera, (24, 24), 4)
        np.testing.assert_allclose(img[12, 12, :3], (2.0, 0, 0), atol=0.02)
        self.assertLess(float(img[2, 2, :3].max()), 1e-6)
        # 1 / 10 of the card's size against a 30 degree field of view: a 0.1 rad wide square, about 4 pixels of 24
        lit = (img[..., 0] > 1.0).sum()
        self.assertTrue(4 <= lit <= 20, lit)

    def _slab(self, absorption=(1, 1, 1)):
        slab = box((4, 4, 0.2), (1, 1, 1, 1), (0, 0, 0), material="liquid", ior=1.5, absorption_color=absorption,
                   absorption_distance=0.2, reflection=1.0)
        backdrop = card(6, 6, (1, 1, 1, 1), (0, 0, -1.5), emission=1.0)
        return s.Scene((slab, backdrop))

    def test_glass_slab_transmits_the_fresnel_sum(self):
        # (1 - F) / (1 + F) with F = ((n - 1) / (n + 1))^2 for a clear parallel slab, all inner reflections counted
        f = ((1.5 - 1) / (1.5 + 1)) ** 2
        img = trace(self._slab(), samples=128, size=(8, 8), diffuse_bounces=0)
        self.assertAlmostEqual(float(center(img, 1).mean()), (1 - f) / (1 + f), delta=0.03)

    def test_absorbing_slab_follows_beer_lambert(self):
        f, a = ((1.5 - 1) / (1.5 + 1)) ** 2, 0.5
        expected = (1 - f) ** 2 * a / (1 - f * f * a * a)
        img = trace(self._slab((0.5, 0.5, 0.5)), samples=128, size=(8, 8), diffuse_bounces=0)
        self.assertAlmostEqual(float(center(img, 1).mean()), expected, delta=0.03)

    def test_glass_sphere_in_a_furnace_is_invisible(self):
        ball = sphere(1.0, material="liquid", ior=1.5, absorption_color=(1, 1, 1), reflection=1.0)
        img = trace(s.Scene((ball,), environments=(uniform_env(),)), samples=64, size=(12, 12))
        mask = img[..., 3] > 0.99
        self.assertGreater(int(mask.sum()), 20)
        self.assertAlmostEqual(float(img[..., :3][mask].mean()), 1.0, delta=0.04)

    def test_glass_sphere_inverts_the_backdrop(self):
        # a ball lens turns the picture behind it round: left of its axis it shows the backdrop's right half
        ball = sphere(1.0, material="liquid", ior=1.5, absorption_color=(1, 1, 1), reflection=1.0)
        backdrop = card(30, 30, (1, 1, 1, 1), (0, 0, -6), emission=1.0)
        red_half = card(15, 30, (1, 0, 0, 1), (-7.5, 0, -5.9), emission=1.0)
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 6)), s.Vec3(0, 0, 0), 60.0)
        img = trace(s.Scene((ball, backdrop, red_half)), camera, (24, 24), 48, diffuse_bounces=0)
        red = lambda x: float(img[10:14, x, 0].mean() - img[10:14, x, 1].mean())
        self.assertGreater(red(1), 0.9)             # plain backdrop: red on the left ...
        self.assertLess(red(22), 0.05)              # ... white on the right
        self.assertLess(red(10), 0.2)               # through the lens, just left of the axis: white
        self.assertGreater(red(13), 0.5)            # just right of the axis: red


class PassTests(unittest.TestCase):
    def test_components_add_up_and_indirect_appears(self):
        scene = cornell()
        beauty = trace(scene, CORNELL_CAMERA, (12, 12), 16, seed=4)
        parts = {name: trace(scene, CORNELL_CAMERA, (12, 12), 16, seed=4, output=name)
                 for name in ("emission", "diffuse", "specular", "diffuse_indirect", "specular_indirect")}
        total = sum(p[..., :3] for p in parts.values())
        np.testing.assert_allclose(total, beauty[..., :3], atol=2e-5)
        self.assertGreater(float(parts["diffuse"][..., :3].mean()), 0.01)
        self.assertGreater(float(parts["diffuse_indirect"][..., :3].mean()), 0.005)
        self.assertEqual(float(parts["specular"][..., :3].max()), 0.0)           # Lambert walls have no highlight
        self.assertEqual(float(parts["specular_indirect"][..., :3].max()), 0.0)

    def test_glossy_floor_has_specular_and_specular_indirect(self):
        floor = card(6, 6, (0.8, 0.8, 0.8, 1), (0, -1, 0), (-90, 0, 0), material="pbr", metallic=0.0, pbr_roughness=0.25)
        wall = card(6, 6, (0.6, 0.2, 0.2, 1), (0, 0, -1.5))
        light = s.Light("Rect", (1, 1, 1), 3.0, s.Vec3(0, 2, 0), s.Vec3(0, -1, 0), area_width=1.0, area_height=1.0)
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0.6, 3)), s.Vec3(0, -1.0, 1.5), 40.0)   # looks down at the light's mirror image
        scene = s.Scene((floor, wall), lights=(light,))
        spec = trace(scene, camera, (16, 16), 48, output="specular")
        indirect = trace(scene, camera, (16, 16), 48, output="specular_indirect")
        self.assertGreater(float(spec[..., :3].max()), 0.05)
        self.assertGreater(float(indirect[..., :3].max()), 0.005)

    def test_data_passes(self):
        ball = sphere(1.0, (0.2, 0.6, 0.9, 1))
        scene = s.Scene((ball,), environments=(uniform_env(),))
        depth = trace(scene, output="depth")
        self.assertAlmostEqual(float(depth[8, 8, 0]), 3.0, delta=0.05)          # the near side of a unit ball 4 away
        self.assertEqual(float(depth[0, 0, 3]), 0.0)
        normals = trace(scene, output="normals")
        np.testing.assert_allclose(normals[8, 8, :3], (0, 0, 1), atol=0.08)
        ids = trace(scene, output="object_id")
        self.assertEqual(float(ids[8, 8, 0]), 1.0)
        albedo = trace(scene, output="albedo")
        np.testing.assert_allclose(albedo[8, 8, :3], (0.2, 0.6, 0.9), atol=0.02)
        position = trace(scene, output="position")
        self.assertAlmostEqual(float(position[8, 8, 2]), 1.0, delta=0.05)
        uv = trace(scene, output="uv")
        self.assertEqual(uv.shape, (16, 16, 4))

    def test_alpha_and_background(self):
        scene = s.Scene((sphere(0.5),), environments=(uniform_env(),))
        img = pt.render(scene, FRONT, 12, 12, (0.1, 0.2, 0.3, 1.0), 0.0, "rgba", pt.PathSettings(samples=8))
        np.testing.assert_allclose(img[0, 0], (0.1, 0.2, 0.3, 1.0), atol=1e-6)
        self.assertGreater(float(img[6, 6, 0]), 0.5)


class SamplingControlTests(unittest.TestCase):
    def test_noise_threshold_stops_quiet_tiles_early(self):
        scene = s.Scene((sphere(1.0),), environments=(uniform_env(),))
        stats = {}
        pt.render(scene, FRONT, 32, 32, (0, 0, 0, 0), 0.0, "rgba",
                  pt.PathSettings(samples=64, pass_samples=8, noise_threshold=0.05), stats=stats)
        counts = stats["samples"]
        self.assertLess(int(counts.min()), 64)
        self.assertGreaterEqual(int(counts.min()), pt.MIN_ADAPTIVE_SAMPLES)

    def test_noisy_tiles_keep_going(self):
        stats = {}
        pt.render(cornell(), CORNELL_CAMERA, 16, 16, (0, 0, 0, 0), 0.0, "rgba",
                  pt.PathSettings(samples=32, pass_samples=8, noise_threshold=0.0001), stats=stats)
        self.assertEqual(int(stats["samples"].min()), 32)

    def test_time_limit_ends_after_a_pass(self):
        stats = {}
        pt.render(cornell(), CORNELL_CAMERA, 12, 12, (0, 0, 0, 0), 0.0, "rgba",
                  pt.PathSettings(samples=4096, pass_samples=2, time_limit=1e-6), stats=stats)
        self.assertEqual(stats["passes"], 1)
        self.assertEqual(int(stats["samples"].max()), 2)

    def test_bounce_caps(self):
        scene = cornell()
        none = trace(scene, CORNELL_CAMERA, (12, 12), 16, max_bounces=8, diffuse_bounces=1)
        direct = trace(scene, CORNELL_CAMERA, (12, 12), 16, max_bounces=1)
        np.testing.assert_allclose(none, direct, atol=1e-6)       # one diffuse event is direct light only
        more = trace(scene, CORNELL_CAMERA, (12, 12), 16, max_bounces=8, diffuse_bounces=4)
        self.assertGreater(float(more.mean()), float(direct.mean()))


class InstanceTests(unittest.TestCase):
    def _instanced(self, count=6):
        source = sphere(0.35, (0.8, 0.3, 0.2, 1), material="pbr", pbr_roughness=0.4)
        rng = np.random.RandomState(2)
        matrices = np.tile(np.eye(4), (count, 1, 1))
        matrices[:, :3, 3] = np.column_stack((np.linspace(-1.2, 1.2, count), rng.uniform(-.3, .3, count), np.zeros(count)))
        instances = s.InstanceSet(sources=(source,), matrices=matrices, variant=np.zeros(count, np.int32))
        light = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(2, 4, 3), s.Vec3(0, 0, 0))
        return s.Scene(instances=(instances,), lights=(light,), environments=(uniform_env(0.3),))

    def test_instances_are_not_flattened(self):
        scene = self._instanced(40)
        built = pt.build_scene(scene)
        self.assertEqual(built.shapes, 40)
        self.assertEqual(len(built.blases), 1)      # forty instances, one bottom-level mesh

    def test_same_picture_as_flattened(self):
        scene = self._instanced()
        a = trace(scene, FRONT, (16, 12), 8)
        b = trace(s.resolve_instances(scene), FRONT, (16, 12), 8)
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertGreater(float(a[..., 3].max()), 0.99)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_image_and_a_new_seed_a_new_one(self):
        scene = cornell()
        a = trace(scene, CORNELL_CAMERA, (12, 12), 8, seed=5)
        b = trace(scene, CORNELL_CAMERA, (12, 12), 8, seed=5)
        c = trace(scene, CORNELL_CAMERA, (12, 12), 8, seed=6)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(float(np.abs(a - c).mean()), 1e-4)

    def test_pass_size_does_not_change_the_image(self):
        scene = cornell()
        a = trace(scene, CORNELL_CAMERA, (12, 12), 8, pass_samples=1)
        b = trace(scene, CORNELL_CAMERA, (12, 12), 8, pass_samples=8)
        np.testing.assert_allclose(a, b, atol=1e-6)


class CancelTests(unittest.TestCase):
    def test_cancel_between_passes(self):
        event = threading.Event()
        seen = []

        def progress(stage, fraction, info):
            seen.append(info["samples"])
            event.set()
        with self.assertRaises(Cancelled):
            pt.render(cornell(), CORNELL_CAMERA, 12, 12, (0, 0, 0, 0), 0.0, "rgba",
                      pt.PathSettings(samples=64, pass_samples=2), cancel=event, progress=progress)
        self.assertEqual(seen, [2])

    def test_already_cancelled(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            pt.render(cornell(), CORNELL_CAMERA, 8, 8, cancel=event)


class NodeTests(unittest.TestCase):
    def _graph(self, **render):
        d = Dispatcher()
        for key, kind, params in (('ball', 'Sphere3D', dict(red=.8, green=.4, blue=.2, sphere_radius=1.0)),
                                  ('camera', 'Camera3D', dict(tz=4.0)), ('scene', 'Scene3D', {}),
                                  ('sky', 'Light3D', dict(light_type='Environment')),
                                  ('render', 'Render3D', dict(width=16, height=12, render_mode='pathtrace',
                                                              pt_samples=4, **render))):
            d.execute(dict(op='create', id=key, type=kind, params=params))
        d.execute(dict(op='connect', id='scene', input='object0', source='ball'))
        d.execute(dict(op='connect', id='scene', input='object1', source='sky'))
        d.execute(dict(op='connect', id='render', input='scene', source='scene'))
        d.execute(dict(op='connect', id='render', input='camera', source='camera'))
        return d

    def test_render3d_in_pathtrace_mode(self):
        d = self._graph()
        image = Evaluator().evaluate(d.document, 'render')
        self.assertEqual(image.shape, (12, 16, 4))
        self.assertGreater(float(image[6, 8, 3]), 0.99)
        self.assertGreater(float(image[6, 8, :3].max()), 0.05)
        again = Evaluator().evaluate(self._graph().document, 'render')
        np.testing.assert_array_equal(image, again)

    def test_outputs_and_settings_reach_the_renderer(self):
        d = self._graph(render_output='diffuse_indirect', max_bounces=1)
        image = Evaluator().evaluate(d.document, 'render')
        self.assertEqual(float(np.abs(image[..., :3]).max()), 0.0)     # one bounce is direct light: nothing indirect
        d.execute(dict(op='set', id='render', param='max_bounces', value=4))
        indirect = Evaluator().evaluate(d.document, 'render')
        self.assertEqual(indirect.shape, (12, 16, 4))

    def test_the_knobs_are_registered(self):
        for name, default in core._PATHTRACE_DEFAULTS.items():
            self.assertEqual(core.SPECS["Render3D"]["params"][name], default)
            self.assertIn(name, core.LIMITS)
        self.assertIn("pathtrace", core.CHOICES["render_mode"])
        self.assertIn("diffuse_indirect", core.CHOICES["render_output"])
        settings = pt.settings_from_params(core.SPECS["Render3D"]["params"])
        self.assertEqual((settings.samples, settings.max_bounces), (64, 8))

    def test_an_old_document_loads_with_the_defaults(self):
        d = Dispatcher()
        d.execute(dict(op='create', id='render', type='Render3D', params={}))
        doc = d.document
        for name in core._PATHTRACE_DEFAULTS:
            doc["nodes"]["render"]["params"].pop(name)
        upgraded = core.upgrade_document(doc)
        for name, default in core._PATHTRACE_DEFAULTS.items():
            self.assertEqual(upgraded["nodes"]["render"]["params"][name], default)

    def test_scenes_it_cannot_draw_are_refused(self):
        splat = s.Scene(particles=(object(),))
        with self.assertRaisesRegex(ValueError, "does not render particles"):
            pt.render(splat, FRONT, 4, 4)


if __name__ == "__main__":
    unittest.main()
