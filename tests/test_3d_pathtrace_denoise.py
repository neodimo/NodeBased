"""The path tracer's denoiser and its guide passes (lane L4 step R4 of 7).

The measured claim is in `DenoiseAgainstAReferenceTests`: at 16 samples per pixel the filtered beauty is a stated
factor closer to a many-sample reference than the raw one, and its mean is where the raw mean was."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from nodebased import gpu3d, pathtrace as pt, ptdenoise, scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from tests.test_3d_pathtrace import CORNELL_CAMERA, cornell, gpu_ready

SIZE = 32
MSE_FACTOR = 2.2            # required; the filter measures about 3 on this scene at 16 samples
MEAN_SHIFT = 0.015          # relative, required; it measures a few tenths of a percent


def mse(a, b):
    return float(np.mean((a[..., :3].astype(np.float64) - b[..., :3]) ** 2))


class SyntheticTests(unittest.TestCase):
    """The filter on constructed images, where the right answer is known."""

    def setUp(self):
        self.rng = np.random.default_rng(11)
        h = w = 48
        self.h, self.w = h, w
        self.normal = np.zeros((h, w, 3))
        self.normal[..., 2] = 1.0
        self.depth = np.full((h, w), 4.0)
        self.albedo = np.full((h, w, 3), 0.5)

    def noisy(self, level, noise=0.25):
        clean = np.broadcast_to(np.asarray(level, np.float64), (self.h, self.w, 3)).copy()
        color = np.empty((self.h, self.w, 4))
        color[..., :3] = clean * self.albedo * (1 + noise * self.rng.standard_normal((self.h, self.w, 1)))
        color[..., 3] = 1.0
        return clean, color

    def test_a_flat_noisy_wall_is_smoothed_and_its_mean_kept(self):
        clean, color = self.noisy((0.8, 0.6, 0.4))
        variance = np.full((self.h, self.w), (0.25 * 0.3) ** 2)
        out = ptdenoise.denoise(color, self.albedo, self.normal, self.depth, variance)
        target = clean * self.albedo
        self.assertLess(mse(out, target), 0.15 * mse(color, target))
        np.testing.assert_allclose(out[..., :3].mean(axis=(0, 1)), color[..., :3].mean(axis=(0, 1)), rtol=0.02)
        np.testing.assert_array_equal(out[..., 3], color[..., 3])          # coverage is left as the tracer had it

    def test_a_geometric_edge_is_not_blurred_across(self):
        clean, color = self.noisy(1.0, noise=0.05)
        left = np.arange(self.w) < self.w // 2
        normal = self.normal.copy()
        normal[:, ~left] = (1.0, 0.0, 0.0)
        # the two faces are lit differently: 1.0 on the left, 0.2 on the right
        color[:, ~left, :3] *= 0.2
        variance = np.full((self.h, self.w), 0.01 ** 2)
        out = ptdenoise.denoise(color, self.albedo, normal, self.depth, variance)
        row = out[self.h // 2, :, 0]
        edge = self.w // 2
        self.assertGreater(float(row[edge - 1]), 0.9 * float(color[..., 0][:, :edge].mean()))
        self.assertLess(float(row[edge]), 1.1 * float(color[..., 0][:, edge:].mean()))

    def test_a_depth_edge_is_not_blurred_across(self):
        clean, color = self.noisy(1.0, noise=0.05)
        far = np.arange(self.w) >= self.w // 2
        depth = self.depth.copy()
        depth[:, far] = 9.0
        color[:, far, :3] *= 0.3
        variance = np.full((self.h, self.w), 0.01 ** 2)
        out = ptdenoise.denoise(color, self.albedo, self.normal, depth, variance)
        edge = self.w // 2
        self.assertGreater(float(out[self.h // 2, edge - 1, 0]), 0.9 * float(color[:, :edge, 0].mean()))
        self.assertLess(float(out[self.h // 2, edge, 0]), 1.15 * float(color[:, edge:, 0].mean()))

    def test_texture_detail_survives_because_the_albedo_is_divided_out(self):
        # a checkerboard albedo under noisy but flat irradiance: the checks stay, the noise goes
        checks = ((np.arange(self.h)[:, None] // 4 + np.arange(self.w)[None, :] // 4) % 2).astype(np.float64)
        albedo = np.repeat(np.where(checks > 0, 0.8, 0.1)[..., None], 3, axis=2)
        irradiance = 1.0 + 0.3 * self.rng.standard_normal((self.h, self.w, 1))
        color = np.concatenate((albedo * irradiance, np.ones((self.h, self.w, 1))), axis=2)
        variance = np.full((self.h, self.w), (0.3 * 0.5) ** 2)
        out = ptdenoise.denoise(color, albedo, self.normal, self.depth, variance)
        target = np.concatenate((albedo, np.ones((self.h, self.w, 1))), axis=2)
        self.assertLess(mse(out, target), 0.15 * mse(color, target))
        contrast = float(out[..., 0][checks > 0].mean() / out[..., 0][checks == 0].mean())
        self.assertGreater(contrast, 6.0)          # the true ratio is 8

    def test_a_converged_image_is_left_alone(self):
        clean, color = self.noisy((0.8, 0.6, 0.4), noise=0.0)
        color[:, self.w // 2:, :3] *= 0.4                              # a hard shadow edge, no noise, no variance
        out = ptdenoise.denoise(color, self.albedo, self.normal, self.depth, np.zeros((self.h, self.w)))
        np.testing.assert_allclose(out, color, atol=2e-3)

    def test_background_and_empty_pixels_stay_empty(self):
        clean, color = self.noisy(1.0)
        color[:10] = 0.0
        albedo = self.albedo.copy()
        albedo[:10] = 0.0
        out = ptdenoise.denoise(color, albedo, self.normal, self.depth, np.full((self.h, self.w), 0.02))
        self.assertEqual(float(np.abs(out[:10]).max()), 0.0)
        self.assertTrue(np.isfinite(out).all())

    def test_without_a_variance_it_is_estimated_from_the_picture(self):
        clean, color = self.noisy(0.7, noise=0.3)
        out = ptdenoise.denoise(color, self.albedo, self.normal, self.depth)
        target = clean * self.albedo
        self.assertLess(mse(out, target), 0.5 * mse(color, target))


class DenoiseAgainstAReferenceTests(unittest.TestCase):
    """The Cornell box at 16 samples, filtered, against 768 samples of the same scene."""

    @classmethod
    def setUpClass(cls):
        cls.scene = cornell()
        cls.settings16 = pt.PathSettings(samples=16, max_bounces=6, seed=5)
        cls.reference = pt.render(cls.scene, CORNELL_CAMERA, SIZE, SIZE,
                                  settings=pt.PathSettings(samples=768, max_bounces=6, seed=99))
        cls.stats = {}
        cls.raw = pt.render(cls.scene, CORNELL_CAMERA, SIZE, SIZE, settings=cls.settings16, stats=cls.stats)
        cls.guides = pt.guide_aovs(cls.scene, CORNELL_CAMERA, SIZE, SIZE, cls.settings16, stats=cls.stats)
        cls.filtered = pt.denoised(cls.raw, cls.guides, cls.stats["variance"], (0, 0, 0, 0))

    def test_the_filtered_beauty_is_closer_to_the_reference_by_a_measured_factor(self):
        raw_error, filtered_error = mse(self.raw, self.reference), mse(self.filtered, self.reference)
        factor = raw_error / filtered_error
        print(f"\ndenoise, Cornell {SIZE}x{SIZE} at 16 spp: mse raw {raw_error:.3e}, filtered {filtered_error:.3e}, "
              f"factor {factor:.2f}")
        self.assertGreater(factor, MSE_FACTOR)

    def test_the_mean_does_not_move(self):
        raw_mean, filtered_mean = float(self.raw[..., :3].mean()), float(self.filtered[..., :3].mean())
        self.assertLess(abs(filtered_mean / raw_mean - 1.0), MEAN_SHIFT)
        # per channel too: the red and green walls keep their bleed
        np.testing.assert_allclose(self.filtered[..., :3].mean(axis=(0, 1)), self.raw[..., :3].mean(axis=(0, 1)),
                                   rtol=MEAN_SHIFT)
        # and against the reference, no more off than the raw picture was
        off = lambda a: abs(float(a[..., :3].mean()) / float(self.reference[..., :3].mean()) - 1.0)
        self.assertLess(off(self.filtered), off(self.raw) + MEAN_SHIFT)

    def test_it_is_not_just_a_blur(self):
        # a box blur of the same reach with no guides does worse, and the walls' edge is still there
        blurred = np.array(self.raw, np.float64)
        for _ in range(3):
            blurred[..., :3] = sum(ptdenoise._tap(blurred[..., :3], dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
        self.assertLess(mse(self.filtered, self.reference), mse(blurred, self.reference))
        red_wall, floor = self.filtered[SIZE // 2, 1, :3], self.filtered[SIZE // 2, SIZE // 2, :3]
        self.assertGreater(float(red_wall[0] - red_wall[1]), 0.05)         # the left wall is still red

    def test_the_filter_is_deterministic(self):
        again = pt.denoised(self.raw, self.guides, self.stats["variance"], (0, 0, 0, 0))
        np.testing.assert_array_equal(again, self.filtered)

    def test_the_denoise_output_is_the_same_thing(self):
        image = pt.render(self.scene, CORNELL_CAMERA, SIZE, SIZE, output="denoise", settings=self.settings16)
        np.testing.assert_array_equal(image, self.filtered)
        # a background goes on after the filter, like the beauty's
        over = pt.render(self.scene, CORNELL_CAMERA, SIZE, SIZE, (0.2, 0.0, 0.0, 1.0), output="denoise",
                         settings=self.settings16)
        np.testing.assert_allclose(over[..., 0], self.filtered[..., 0] + 0.2 * (1 - self.filtered[..., 3]), atol=1e-6)

    def test_the_variance_estimate_falls_with_the_sample_count(self):
        more = {}
        pt.render(self.scene, CORNELL_CAMERA, SIZE, SIZE, settings=pt.PathSettings(samples=64, max_bounces=6, seed=5),
                  stats=more)
        self.assertLess(float(more["variance"].mean()), 0.4 * float(self.stats["variance"].mean()))


class GuideTests(unittest.TestCase):
    def test_the_guides_are_albedo_normals_and_depth_of_the_first_hit(self):
        scene = cornell()
        guides = pt.guide_aovs(scene, CORNELL_CAMERA, 16, 16, pt.PathSettings(samples=4, max_bounces=1))
        self.assertEqual(sorted(guides), ["albedo", "depth", "normals"])
        for name, image in guides.items():
            self.assertEqual(image.shape, (16, 16, 4), name)
        # the back wall faces the camera: normal +z, depth 3.6 - (-1) at the centre
        np.testing.assert_allclose(guides["normals"][8, 8, :3], (0, 0, 1), atol=1e-6)
        self.assertAlmostEqual(float(guides["depth"][8, 8, 0]), 4.6, delta=0.02)
        np.testing.assert_allclose(guides["albedo"][8, 8, :3], (0.8, 0.8, 0.8), atol=1e-6)
        # the left wall is the red one (0.9, 0.1, 0.1)
        np.testing.assert_allclose(guides["albedo"][8, 3, :3], (0.9, 0.1, 0.1), atol=1e-6)
        np.testing.assert_allclose(guides["normals"][8, 3, :3], (1, 0, 0), atol=1e-6)

    def _graph(self, **render):
        d = Dispatcher()
        for key, kind, params in (('ball', 'Sphere3D', dict(red=.8, green=.4, blue=.2, sphere_radius=1.0)),
                                  ('camera', 'Camera3D', dict(tz=4.0)), ('scene', 'Scene3D', {}),
                                  ('sky', 'Light3D', dict(light_type='Environment')),
                                  ('render', 'Render3D', dict(width=16, height=12, render_mode='pathtrace',
                                                              pt_samples=8, **render)),
                                  ('write', 'Write', dict(bit_depth='float'))):
            d.execute(dict(op='create', id=key, type=kind, params=params))
        d.execute(dict(op='connect', id='scene', input='object0', source='ball'))
        d.execute(dict(op='connect', id='scene', input='object1', source='sky'))
        d.execute(dict(op='connect', id='render', input='scene', source='scene'))
        d.execute(dict(op='connect', id='render', input='camera', source='camera'))
        d.execute(dict(op='connect', id='write', input='image', source='render'))
        return d

    def test_render3d_outputs_denoise_and_the_multichannel_passes_carry_the_guides(self):
        d = self._graph(render_output="denoise")
        image = Evaluator().evaluate(d.document, "render")
        self.assertEqual(image.shape, (12, 16, 4))
        d.execute(dict(op="set", id="render", param="render_output", value="multichannel"))
        d.execute(dict(op="set", id="render", param="passes", value="beauty,albedo,normals,depth,denoise"))
        raster = Evaluator().evaluate_raster(d.document, "render")
        self.assertEqual(list(raster.layers), ["normals", "depth", "albedo", "denoise"])
        for name in ("albedo", "normals", "depth", "denoise"):
            self.assertEqual(raster.layers[name].pixels.shape, (12, 16, 4), name)
        # the layers are the single-purpose outputs
        d.execute(dict(op="set", id="render", param="render_output", value="denoise"))
        np.testing.assert_array_equal(raster.layers["denoise"].pixels, Evaluator().evaluate(d.document, "render"))
        np.testing.assert_allclose(raster.layers["albedo"].pixels[6, 8, :3], (0.8, 0.4, 0.2), atol=1e-6)

    def test_the_exr_holds_the_raw_beauty_next_to_the_guides_and_the_filtered_layer(self):
        d = self._graph(render_output="multichannel", passes="beauty,albedo,normals,depth,denoise")
        raster = Evaluator().evaluate_raster(d.document, "render")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "look.exr"
            write_exr(path, raster.pixels, "float", layers={k: v.pixels for k, v in raster.layers.items()})
            source = oiio.ImageInput.open(str(path))
            try:
                names = list(source.spec().channelnames)
            finally:
                source.close()
        for channel in ("R", "G", "B", "A", "albedo.R", "albedo.G", "albedo.B", "normals.X", "normals.Y", "normals.Z",
                        "depth.Z", "denoise.R", "denoise.G", "denoise.B"):
            self.assertIn(channel, names)

    def test_the_denoise_output_is_path_tracer_only(self):
        scene = s.Scene()
        with self.assertRaisesRegex(ValueError, "denoise output needs"):
            s.render(scene, s.Camera(), 8, 8, output="denoise")
        with self.assertRaisesRegex(ValueError, "denoise pass needs"):
            s.render_multichannel(scene, s.Camera(), 8, 8, passes="beauty,denoise")

    def test_the_albedo_pass_is_a_layer_in_the_other_modes_too(self):
        card = s._card(2, 2, (0.3, 0.6, 0.9, 1), s.Transform3D())
        beauty, layers = s.render_multichannel(s.Scene((card,)), s.Camera(), 12, 12, passes="beauty,albedo")
        np.testing.assert_allclose(layers["albedo"][6, 6, :3], (0.3, 0.6, 0.9), atol=1e-6)


@unittest.skipUnless(gpu_ready(), "wgpu adapter unavailable for the path tracer")
class GpuDenoiseTests(unittest.TestCase):
    def test_the_gpu_beauty_denoises_toward_the_cpu_reference(self):
        scene = cornell()
        reference = pt.render(scene, CORNELL_CAMERA, SIZE, SIZE, settings=pt.PathSettings(samples=768, max_bounces=6, seed=99))
        settings = pt.PathSettings(samples=16, max_bounces=6, seed=5)
        raw = pt.render(scene, CORNELL_CAMERA, SIZE, SIZE, settings=settings, backend="gpu")
        filtered = pt.render(scene, CORNELL_CAMERA, SIZE, SIZE, output="denoise", settings=settings, backend="gpu")
        factor = mse(raw, reference) / mse(filtered, reference)
        print(f"\ndenoise on the GPU beauty: factor {factor:.2f}")
        self.assertGreater(factor, 1.8)
        self.assertLess(abs(float(filtered[..., :3].mean()) / float(raw[..., :3].mean()) - 1.0), 0.02)

    def test_the_gpu_guides_match_the_cpu(self):
        scene = cornell()
        settings = pt.PathSettings(samples=4, max_bounces=1)
        cpu = pt.guide_aovs(scene, CORNELL_CAMERA, 16, 16, settings)
        gpu = pt.guide_aovs(scene, CORNELL_CAMERA, 16, 16, settings, backend="gpu")
        for name in ("normals", "depth"):
            # one un-jittered ray per pixel on both, in f32 against f64: a pixel on an edge or a corner may pick the
            # neighbouring surface, so allow the few that do
            differ = np.abs(gpu[name] - cpu[name]).max(axis=-1) > 1e-3
            self.assertLess(float(differ.mean()), 0.03, name)


if __name__ == "__main__":
    unittest.main()
