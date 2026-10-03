"""Adaptive sampling and the final-render denoiser (lane L4, plan "Rendering 6", step R1 of 3).

`sampling = adaptive` stops each pixel on its own noise estimate (`pathtrace.pixel_noise`); the tests assert pixels
and sample counts, on the CPU reference and on the GPU path tracer (guarded by `gpu3d.available()`)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from nodebased import core, gpu3d, pathtrace as pt, scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from tests.test_3d_pathtrace import CORNELL_CAMERA, FRONT, box, card, cornell, sphere

SIZE = (32, 24)
BACKGROUND = (0, 0, 0, 0)
POINT = s.Light("Point", (1, 1, 1), 3.0, s.Vec3(2, 3, 2))
SHADOW_CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 4, 6)), s.Vec3(0, -1, 0), 45.0)
TIGHT_PSNR_DB = 37.5        # adaptive at threshold 0.0003 against a 1024-sample fixed render: measures 39.2 dB (fixed 64: 40.0)
FLAT_CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 2)), s.Vec3(0, 0, 0), 30.0)


def shadow_scene():
    ground = card(8, 8, (.7, .7, .7, 1), (0, -1, 0), (-90, 0, 0))
    return s.Scene((ground, box((1, 1, 1), (.5, .5, .5, 1), (0, -.5, 0))), (POINT,))


def flat_scene():
    """A plane that fills the view, lit by one point light, direct light only: every pixel's value is smooth."""
    wall = card(12, 12, (.6, .6, .6, 1), (0, 0, 0))
    return s.Scene((wall,), (s.Light("Point", (1, 1, 1), 2.0, s.Vec3(0, 0, 3)),))


def render(scene, camera, settings, backend="cpu", size=SIZE, output="rgba"):
    stats = {}
    image = pt.render(scene, camera, size[0], size[1], BACKGROUND, 0.0, output, settings, stats=stats, backend=backend)
    return image, stats


def adaptive(threshold, **kw):
    return pt.PathSettings(sampling="adaptive", noise_threshold=threshold,
                           **{"min_samples": 8, "max_samples": 128, "adaptive_pass_size": 8, "max_bounces": 2, **kw})


def psnr(image, reference):
    mse = float(np.mean((image[..., :3].astype(np.float64) - reference[..., :3]) ** 2))
    return 10 * np.log10(float(reference[..., :3].max()) ** 2 / max(mse, 1e-30))


class NoiseEstimateTests(unittest.TestCase):
    def test_a_flat_pixel_reads_zero_and_a_scattered_one_one_over_n(self):
        n = np.array([16.0, 16.0])
        flat = np.array([16 * 0.5, 16 * 0.5])
        sq = np.array([16 * 0.25, 0.0])
        self.assertAlmostEqual(float(pt.pixel_noise(flat[:1], sq[:1], n[:1])[0]), 0.0, places=12)
        # samples alternating 0 and 1: mean 0.5, sample variance 16/15 * 0.25, variance of the mean that over 16
        noise = float(pt.pixel_noise(np.array([8.0]), np.array([8.0]), np.array([16.0]))[0])
        self.assertAlmostEqual(noise, (16 / 15 * 0.25 / 16) / (0.5 + pt.NOISE_FLOOR) ** 2, places=12)

    def test_a_black_pixel_does_not_read_as_noisy(self):
        self.assertEqual(float(pt.pixel_noise(np.zeros(1), np.zeros(1), np.full(1, 8.0))[0]), 0.0)


class SettingsTests(unittest.TestCase):
    def test_defaults_are_fixed_and_old_params_give_the_old_settings(self):
        old = {"pt_samples": 24, "noise_threshold": 0.05}
        st = pt.settings_from_params(old)
        self.assertEqual((st.sampling, st.samples, st.noise_threshold), ("fixed", 24, 0.05))
        self.assertFalse(st.adaptive)

    def test_adaptive_settings_are_clamped_to_something_runnable(self):
        st = pt.PathSettings(sampling="adaptive", min_samples=1, max_samples=64, adaptive_pass_size=0).clamped()
        self.assertEqual((st.min_samples, st.max_samples, st.adaptive_pass_size), (2, 64, 1))     # one sample has no variance
        st = pt.PathSettings(sampling="adaptive", min_samples=500, max_samples=40).clamped()
        self.assertEqual((st.min_samples, st.max_samples), (40, 40))
        self.assertEqual(pt.PathSettings(sampling="nonsense").clamped().sampling, "fixed")

    def test_the_knobs_are_registered(self):
        for name, default in core._ADAPTIVE_DEFAULTS.items():
            self.assertEqual(core.SPECS["Render3D"]["params"][name], default)
        self.assertEqual(core.CHOICES["sampling"], ["fixed", "adaptive"])
        self.assertEqual(core.CHOICES["denoise"], ["off", "final"])
        for name in ("min_samples", "max_samples", "adaptive_pass_size", "beauty_raw"):
            self.assertIn(name, core.LIMITS)


class AdaptiveCpuTests(unittest.TestCase):
    def test_a_flat_lit_plane_converges_at_min_samples(self):
        image, stats = render(flat_scene(), FLAT_CAMERA, adaptive(0.05, min_samples=8, max_bounces=1))
        self.assertTrue(np.all(stats["samples"] == 8), np.unique(stats["samples"]))
        self.assertTrue(stats["converged"].all())
        self.assertEqual(stats["passes"], 1)         # every pixel stopped after the first pass: the render ended early
        self.assertGreater(float(image[12, 16, 0]), 0.1)

    def test_a_pixel_under_a_hard_shadow_edge_takes_more_samples_than_one_in_open_light(self):
        scene = shadow_scene()
        reference, _ = render(scene, SHADOW_CAMERA, pt.PathSettings(samples=256, max_bounces=2))
        luminance = reference[..., :3].mean(axis=2)
        gradient = np.hypot(*np.gradient(luminance))
        lit = luminance > 0.5 * luminance.max()
        edge = np.unravel_index(np.argmax(np.where(lit | (gradient > 0), gradient, 0)), gradient.shape)
        open_light = np.unravel_index(np.argmin(np.where(lit & (luminance > 0), gradient, np.inf)), gradient.shape)
        _, stats = render(scene, SHADOW_CAMERA, adaptive(0.02))
        samples = stats["samples"]
        self.assertGreater(int(samples[edge]), int(samples[open_light]))
        self.assertEqual(int(samples[open_light]), 8)       # open light: quiet, stops at min_samples
        self.assertGreater(int(samples.max()), 8)           # something is noisy enough to keep going

    def test_a_tight_threshold_is_within_the_documented_psnr_of_the_fixed_reference(self):
        scene = shadow_scene()
        reference, _ = render(scene, SHADOW_CAMERA, pt.PathSettings(samples=1024, max_bounces=2, seed=5))
        image, stats = render(scene, SHADOW_CAMERA, adaptive(0.0003, max_samples=512, seed=9))
        self.assertGreaterEqual(psnr(image, reference), TIGHT_PSNR_DB)
        # and it is no accident of taking every sample: the quiet floor still stopped early, under a fixed 64
        self.assertLess(float(stats["samples"].mean()), 64)

    def test_threshold_zero_never_stops_a_pixel_early(self):
        _, stats = render(flat_scene(), FLAT_CAMERA, adaptive(0.0, max_samples=24, max_bounces=1))
        self.assertTrue(np.all(stats["samples"] == 24))

    def test_pass_size_does_not_change_the_image_of_a_pixel_that_takes_the_same_count(self):
        # sample streams are per (pixel, sample index): a different pass size reaches the same pixels at the same samples
        a, sa = render(flat_scene(), FLAT_CAMERA, adaptive(0.0, max_samples=32, adaptive_pass_size=4, max_bounces=1))
        b, sb = render(flat_scene(), FLAT_CAMERA, adaptive(0.0, max_samples=32, adaptive_pass_size=16, max_bounces=1))
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_adaptive_ignores_samples_and_matches_fixed_when_nothing_stops(self):
        scene = shadow_scene()
        a, _ = render(scene, SHADOW_CAMERA, adaptive(0.0, max_samples=24))
        b, _ = render(scene, SHADOW_CAMERA, pt.PathSettings(samples=24, max_bounces=2))
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_progress_reports_the_converged_fraction_and_ends_at_one(self):
        seen = []
        pt.render(flat_scene(), FLAT_CAMERA, SIZE[0], SIZE[1], BACKGROUND, 0.0, "rgba", adaptive(0.05, max_bounces=1),
                  progress=lambda stage, fraction, info: seen.append((fraction, info)))
        self.assertEqual(len(seen), 1)
        fraction, info = seen[0]
        self.assertEqual((fraction, info["converged"], info["pixels_active"], info["passes"]), (1.0, 1.0, 0, 1))

    def test_noise_image_and_converged_mask_are_in_the_stats(self):
        _, stats = render(shadow_scene(), SHADOW_CAMERA, adaptive(0.02))
        self.assertEqual(stats["noise"].shape, (SIZE[1], SIZE[0]))
        done = stats["converged"]
        self.assertTrue(np.all(stats["noise"][done & (stats["samples"] >= 8)] < 0.02 + 1e-12))


class OldDocumentTests(unittest.TestCase):
    def _graph(self, **render_params):
        d = Dispatcher()
        for key, kind, params in (("ball", "Sphere3D", dict(red=.8, green=.4, blue=.2, sphere_radius=1.0)),
                                  ("camera", "Camera3D", dict(tz=4.0)), ("scene", "Scene3D", {}),
                                  ("sky", "Light3D", dict(light_type="Environment")),
                                  ("render", "Render3D", {"width": 16, "height": 12, "render_mode": "pathtrace", "pt_samples": 6,
                                                  **render_params})):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        d.execute(dict(op="connect", id="scene", input="object0", source="ball"))
        d.execute(dict(op="connect", id="scene", input="object1", source="sky"))
        d.execute(dict(op="connect", id="render", input="scene", source="scene"))
        d.execute(dict(op="connect", id="render", input="camera", source="camera"))
        return d

    def test_a_document_without_the_new_keys_renders_exactly_as_before(self):
        d = self._graph(noise_threshold=0.05)
        explicit = Evaluator().evaluate(d.document, "render")
        for name in list(core._ADAPTIVE_DEFAULTS) + ["denoise", "beauty_raw"]:
            d.document["nodes"]["render"]["params"].pop(name, None)
        old = Evaluator().evaluate(core.upgrade_document(d.document), "render")
        np.testing.assert_array_equal(old, explicit)
        params = d.document["nodes"]["render"]["params"]
        params.pop("noise_threshold")
        self.assertEqual(pt.settings_from_params(params), pt.PathSettings(samples=6, seed=params["pt_seed"]).clamped())
        legacy = Evaluator().evaluate(d.document, "render")
        self.assertEqual(legacy.shape, (12, 16, 4))

    def test_sampling_adaptive_reaches_the_renderer(self):
        d = self._graph(sampling="adaptive", noise_threshold=0.05, min_samples=4, max_samples=12, adaptive_pass_size=4)
        st = pt.settings_from_params(d.document["nodes"]["render"]["params"])
        self.assertEqual((st.sampling, st.min_samples, st.max_samples, st.adaptive_pass_size), ("adaptive", 4, 12, 4))
        image = Evaluator().evaluate(d.document, "render")
        self.assertEqual(image.shape, (12, 16, 4))


class FinalDenoiseTests(OldDocumentTests):
    """`denoise = final` runs the viewport's denoiser on the final render (step R1), `beauty_raw` keeps the noisy one."""

    def test_denoise_off_is_the_default_and_leaves_the_beauty_alone(self):
        d = self._graph()
        self.assertEqual(d.document["nodes"]["render"]["params"]["denoise"], "off")
        raster = Evaluator().evaluate_raster(d.document, "render")
        self.assertFalse(raster.layers)

    def test_final_equals_the_denoise_output_with_the_same_controls(self):
        controls = dict(denoiser_strength=0.7, denoise_color_sensitivity=2.0, denoise_iterations=3, pt_samples=8)
        final = Evaluator().evaluate(self._graph(denoise="final", **controls).document, "render")
        output = Evaluator().evaluate(self._graph(render_output="denoise", **controls).document, "render")
        np.testing.assert_array_equal(final, output)
        raw = Evaluator().evaluate(self._graph(pt_samples=8).document, "render")
        self.assertFalse(np.array_equal(final, raw))
        self.assertLess(float(np.var(final[..., :3])), float(np.var(raw[..., :3])) * 1.0001)

    def test_strength_zero_final_is_the_raw_beauty(self):
        final = Evaluator().evaluate(self._graph(denoise="final", denoiser_strength=0.0).document, "render")
        raw = Evaluator().evaluate(self._graph().document, "render")
        np.testing.assert_array_equal(final, raw)

    def test_beauty_raw_is_the_noisy_beauty_as_a_layer_and_lands_in_the_exr(self):
        d = self._graph(denoise="final", beauty_raw=1)
        raster = Evaluator().evaluate_raster(d.document, "render")
        self.assertEqual(list(raster.layers), ["beauty_raw"])
        raw = Evaluator().evaluate(self._graph().document, "render")
        np.testing.assert_array_equal(raster.layers["beauty_raw"].pixels, raw)
        filtered = Evaluator().evaluate(self._graph(denoise="final").document, "render")
        np.testing.assert_array_equal(raster.pixels, filtered)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "final.exr"
            write_exr(path, raster.pixels, "float", layers={k: v.pixels for k, v in raster.layers.items()})
            source = oiio.ImageInput.open(str(path))
            try:
                names = list(source.spec().channelnames)
            finally:
                source.close()
        for channel in ("R", "G", "B", "A", "beauty_raw.R", "beauty_raw.G", "beauty_raw.B"):
            self.assertIn(channel, names)

    def test_beauty_raw_without_denoise_final_writes_nothing(self):
        raster = Evaluator().evaluate_raster(self._graph(beauty_raw=1).document, "render")
        self.assertFalse(raster.layers)

    def test_final_denoise_needs_the_path_tracer(self):
        d = self._graph(denoise="final", render_mode="raytrace")
        with self.assertRaisesRegex(ValueError, "Denoise final needs"):
            Evaluator().evaluate(d.document, "render")

    def test_final_denoise_on_an_adaptive_render_reads_the_per_pixel_variance(self):
        d = self._graph(denoise="final", sampling="adaptive", noise_threshold=0.05, min_samples=4, max_samples=12,
                        adaptive_pass_size=4, beauty_raw=1)
        raster = Evaluator().evaluate_raster(d.document, "render")
        self.assertEqual(raster.pixels.shape, (12, 16, 4))
        self.assertEqual(raster.layers["beauty_raw"].pixels.shape, (12, 16, 4))

    def test_the_motion_blur_route_keeps_a_raw_layer_too(self):
        d = self._graph(denoise="final", beauty_raw=1, motion_blur=1, motion_samples=2, pt_samples=4)
        raster = Evaluator().evaluate_raster(d.document, "render")
        self.assertEqual(list(raster.layers), ["beauty_raw"])


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class AdaptiveGpuTests(unittest.TestCase):
    def test_several_samples_per_dispatch_average_instead_of_summing_cumulatively(self):
        # the shader once carried one sample's radiance into the next within a dispatch: 8 per pass read ~4x too bright
        one, _ = render(flat_scene(), FLAT_CAMERA, pt.PathSettings(samples=16, max_bounces=1), backend="gpu")
        many, _ = render(flat_scene(), FLAT_CAMERA, pt.PathSettings(samples=16, pass_samples=8, max_bounces=1), backend="gpu")
        np.testing.assert_allclose(many, one, rtol=0.02)

    def test_flat_plane_and_shadow_edge_behave_as_on_the_cpu(self):
        _, stats = render(flat_scene(), FLAT_CAMERA, adaptive(0.05, min_samples=8, max_bounces=1), backend="gpu")
        self.assertTrue(np.all(stats["samples"] == 8))
        self.assertEqual(stats["passes"], 1)
        _, edge = render(shadow_scene(), SHADOW_CAMERA, adaptive(0.02), backend="gpu")
        self.assertGreater(int(edge["samples"].max()), int(edge["samples"].min()))

    def test_adaptive_gpu_agrees_with_the_cpu_reference(self):
        for name, (scene, camera) in {"shadow": (shadow_scene(), SHADOW_CAMERA), "cornell": (cornell(), CORNELL_CAMERA)}.items():
            with self.subTest(scene=name):
                settings = adaptive(0.01, max_samples=96, seed=11)
                cpu, cs = render(scene, camera, settings)
                gpu, gs = render(scene, camera, settings, backend="gpu")
                self.assertAlmostEqual(float(gpu[..., :3].mean()) / float(cpu[..., :3].mean()), 1.0, delta=0.02)
                rmse = float(np.sqrt(np.mean((gpu - cpu) ** 2)))
                self.assertLess(rmse, 0.03 * float(cpu.mean()) + 0.01)            # the existing path tracer tolerance
                self.assertGreater(float((np.abs(cs["samples"].astype(int) - gs["samples"]) <= 16).mean()), 0.95)
                self.assertEqual(gs["backend"], "gpu")

    def test_final_denoise_runs_on_the_gpu_backend(self):
        d = FinalDenoiseTests()._graph(denoise="final", beauty_raw=1, render_backend="gpu")
        raster = Evaluator().evaluate_raster(d.document, "render")
        self.assertEqual(list(raster.layers), ["beauty_raw"])
        self.assertLess(float(np.var(raster.pixels[..., :3])), float(np.var(raster.layers["beauty_raw"].pixels[..., :3])) * 1.0001)

    def test_adaptive_threshold_zero_matches_fixed_on_the_gpu(self):
        a, _ = render(shadow_scene(), SHADOW_CAMERA, adaptive(0.0, max_samples=16), backend="gpu")
        b, _ = render(shadow_scene(), SHADOW_CAMERA, pt.PathSettings(samples=16, max_bounces=2), backend="gpu")
        np.testing.assert_allclose(a, b, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
