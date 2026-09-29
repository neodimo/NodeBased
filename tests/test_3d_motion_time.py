"""Motion blur (lane L4 step R5, finish 1): every path of the CPU path tracer carries its own time.

docs/3D_FOUNDATION.md "Motion blur", "Sampling". The times are the shutter samples of `render_motion`'s moments; a
path picks one (`pathtrace.path_time`) and traces that moment's scene through that moment's camera, all inside the one
sampling loop of `pathtrace.render`.
"""
import dataclasses
import math
import unittest

import numpy as np

from nodebased import motionblur, pathtrace as pt, scene3d as s
from nodebased.imaging import Evaluator
from tests.test_3d_motion_blur import extent, graph
from tests.test_3d_pathtrace import sphere

CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 10)), s.Vec3(0, 0, 0), 30.0)
SETTINGS = dict(max_bounces=0)


def dot_at(x, radius=0.4):
    dot = sphere(radius, (1, 1, 1, 1), emission=1.0)
    return s.Scene((dataclasses.replace(dot, transform=s.Transform3D(position=s.Vec3(x, 0, 0))),))


def lit_columns(picture, threshold=0.02):
    lit = np.flatnonzero((picture[..., 3] > threshold).any(axis=0))
    return int(lit.min()), int(lit.max())


class PathTimeTests(unittest.TestCase):
    def test_a_pixels_consecutive_samples_cover_every_time_once(self):
        pixels = np.repeat(np.arange(50), 8)
        samples = np.tile(np.arange(8), 50)
        times = pt.path_time(pixels, samples, 1, 8).reshape(50, 8)
        for row in times:
            self.assertEqual(sorted(row), list(range(8)))

    def test_neighbouring_pixels_start_at_different_times(self):
        starts = pt.path_time(np.arange(400), np.zeros(400, int), 1, 8)
        counts = np.bincount(starts, minlength=8)
        self.assertGreater(int(counts.min()), 20)          # roughly 50 each: no time is favoured

    def test_a_different_seed_changes_the_order(self):
        a = pt.path_time(np.arange(64), np.zeros(64, int), 1, 8)
        b = pt.path_time(np.arange(64), np.zeros(64, int), 2, 8)
        self.assertFalse(np.array_equal(a, b))


class TimedRenderTests(unittest.TestCase):
    def moments(self, xs):
        return [(dot_at(x), CAMERA) for x in xs]

    def render(self, moments, samples, stats=None, **more):
        settings = pt.PathSettings(samples=samples, **SETTINGS, **more)
        return pt.render_motion(moments, 64, 64, (0, 0, 0, 0), 0.0, "rgba", settings)

    def test_each_moment_lights_its_own_place_in_equal_measure(self):
        picture = self.render(self.moments([-1.5, 1.5]), 64)
        left, right = picture[..., 3][:, :32].sum(), picture[..., 3][:, 32:].sum()
        self.assertGreater(float(left), 20.0)
        self.assertAlmostEqual(float(left), float(right), delta=0.03 * float(left))

    def test_the_sample_count_is_the_one_asked_for(self):
        stats = {}
        moments = self.moments(np.linspace(-1, 1, 9))
        scene, camera = moments[4]
        pt.render(scene, camera, 32, 32, (0, 0, 0, 0), 0.0, "rgba", pt.PathSettings(samples=70, **SETTINGS),
                  stats=stats, moments=moments)
        self.assertTrue((stats["samples"] == 70).all())    # equal shares would have cost ceil(70 / 9) * 9 = 72

    def test_the_noise_stop_and_the_time_are_those_of_the_whole_render(self):
        stats = {}
        moments = self.moments(np.linspace(-1, 1, 8))
        scene, camera = moments[4]
        pt.render(scene, camera, 64, 64, (0, 0, 0, 0), 0.0, "rgba",
                  pt.PathSettings(samples=256, noise_threshold=0.05, **SETTINGS), stats=stats, moments=moments)
        counts = stats["samples"]
        self.assertEqual(int(counts.min()), pt.MIN_ADAPTIVE_SAMPLES)     # empty tiles retire at the first look
        self.assertLess(float(counts.mean()), 256)
        self.assertGreater(int(counts.max()), pt.MIN_ADAPTIVE_SAMPLES)   # the moving dot keeps sampling

    def test_the_blur_is_the_length_of_the_motion_and_a_moment_camera_moves_the_picture(self):
        times = motionblur.shutter_times(-0.5, 0.5, 8)
        moments = [(dot_at(2.0 * t), CAMERA) for t in times]          # two units per frame across one frame
        blurred = self.render(moments, 128)
        sharp = self.render([(dot_at(0.0), CAMERA)] * 2, 128)
        scale = 64 / (2 * 10 * math.tan(math.radians(15)))
        grown = (lit_columns(blurred)[1] - lit_columns(blurred)[0]) - (lit_columns(sharp)[1] - lit_columns(sharp)[0])
        self.assertAlmostEqual(grown, 2.0 * scale, delta=3.0)
        panning = [(dot_at(0.0), s.Camera(s.Transform3D(position=s.Vec3(x, 0, 10)), s.Vec3(x, 0, 0), 30.0))
                   for x in (-1.0, 1.0)]
        moved = self.render(panning, 128)
        self.assertGreater(lit_columns(moved)[1] - lit_columns(moved)[0],
                           lit_columns(sharp)[1] - lit_columns(sharp)[0] + 1.5 * scale)

    def test_one_moment_or_a_repeated_moment_is_the_plain_render(self):
        moments = [(dot_at(0.3), CAMERA)] * 4
        timed = self.render(moments, 32, seed=5)
        plain = pt.render(dot_at(0.3), CAMERA, 64, 64, (0, 0, 0, 0), 0.0, "rgba",
                          pt.PathSettings(samples=32, seed=5, **SETTINGS))
        self.assertLess(float(np.abs(timed - plain).max()), 1e-6)

    def test_seeded_and_reproducible(self):
        moments = self.moments([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(self.render(moments, 24), self.render(moments, 24))


def layers_of(d, frame=0):
    raster = Evaluator().evaluate_raster(d.document, "render", frame=frame)
    return raster.to_display(), {name: layer.pixels for name, layer in raster.layers.items()}


class BundleOutputTests(unittest.TestCase):
    """The multichannel and relight outputs blur: the beauty and the light layers across the shutter, the
    measurements (normals, depth, position) stay those of the middle time."""

    def multichannel(self, mode, passes, **more):
        return layers_of(graph(mode=mode, render_output="multichannel", passes=passes, samples=9, shutter=1.0,
                               **more))

    def sharp(self, mode, passes, **more):
        return layers_of(graph(mode=mode, render_output="multichannel", passes=passes, samples=9, motion=0, **more))

    def test_multichannel_blurs_the_beauty_and_keeps_depth_sharp(self):
        for mode in ("raytrace", "raster"):
            with self.subTest(mode=mode):
                beauty, layers = self.multichannel(mode, "beauty,depth,normals")
                sharp_beauty, sharp_layers = self.sharp(mode, "beauty,depth,normals")
                self.assertGreater(extent(beauty), extent(sharp_beauty) + 10)
                for name in ("depth", "normals"):
                    np.testing.assert_array_equal(layers[name], sharp_layers[name])

    def test_multichannel_beauty_is_the_beauty_output(self):
        beauty, _ = self.multichannel("raytrace", "beauty")
        single = Evaluator().evaluate(graph(mode="raytrace", samples=9, shutter=1.0).document, "render", frame=0)
        np.testing.assert_allclose(beauty, single, atol=1e-6)

    def test_multichannel_albedo_blurs_and_the_motion_pass_is_kept(self):
        _, layers = self.multichannel("raytrace", "albedo,motion")
        _, sharp = self.sharp("raytrace", "albedo,motion")
        self.assertGreater(extent(layers["albedo"]), extent(sharp["albedo"]) + 10)
        np.testing.assert_array_equal(layers["motion"], sharp["motion"])

    def test_path_traced_multichannel_blurs_beauty_and_albedo_and_denoises_the_blur(self):
        beauty, layers = self.multichannel("pathtrace", "beauty,albedo,normals,denoise")
        sharp_beauty, sharp_layers = self.sharp("pathtrace", "beauty,albedo,normals,denoise")
        self.assertGreater(extent(beauty), extent(sharp_beauty) + 10)
        self.assertGreater(extent(layers["albedo"]), extent(sharp_layers["albedo"]) + 10)
        self.assertGreater(extent(layers["denoise"]), extent(sharp_layers["denoise"]) + 10)
        np.testing.assert_array_equal(layers["normals"], sharp_layers["normals"])

    def test_the_relight_bundle_blurs_its_beauty_and_lighting_and_keeps_normals_and_position(self):
        light = ("light", "Light3D", {})
        d = graph(mode="raster", render_output="relight", samples=9, shutter=1.0)
        d.execute(dict(op="create", id="light", type="Light3D", params={}))
        d.execute(dict(op="connect", id="scene", input="object1", source="light"))
        sharp_d = graph(mode="raster", render_output="relight", samples=9, motion=0)
        sharp_d.execute(dict(op="create", id="light", type="Light3D", params={}))
        sharp_d.execute(dict(op="connect", id="scene", input="object1", source="light"))
        beauty, layers = layers_of(d)
        sharp_beauty, sharp_layers = layers_of(sharp_d)
        self.assertGreater(extent(beauty), extent(sharp_beauty) + 10)
        for name in ("albedo", "diffuse", "emission"):
            self.assertGreater(extent(layers[name]), extent(sharp_layers[name]) + 10, name)
        for name in ("normals", "position"):
            np.testing.assert_array_equal(layers[name], sharp_layers[name])


if __name__ == "__main__":
    unittest.main()
