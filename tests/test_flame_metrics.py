"""Lane 6, fire shape step B: the flame-shape measurements in examples/fire_shot/flame_metrics.py, on volumes built to have a known answer."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "fire_shot"
sys.path.insert(0, str(EXAMPLE))
try:
    import flame_metrics as fm
finally:
    sys.path.remove(str(EXAMPLE))

from nodebased import scene3d, volumerender


def volume(flame, temperature=None, density=None, velocity=None):
    flame = np.asarray(flame, np.float32)
    return scene3d.Volume(density=np.zeros_like(flame) if density is None else np.asarray(density, np.float32),
                          voxel_size=.1, origin=(0.0, 0.0, 0.0), flame=flame,
                          temperature=None if temperature is None else np.asarray(temperature, np.float32),
                          velocity=velocity)


class RegionTests(unittest.TestCase):
    def test_label_regions_counts_separate_blobs_and_ignores_specks(self):
        mask = np.zeros((12, 12), bool)
        mask[1:4, 1:4] = True
        mask[7:10, 6:9] = True
        mask[11, 11] = True                    # one cell: below TONGUE_MIN_CELLS
        self.assertEqual(fm.label_regions(mask), 2)
        mask[4, 3] = mask[5, 3] = mask[6, 3] = mask[6, 4] = mask[6, 5] = mask[6, 6] = True   # a bridge joins them
        self.assertEqual(fm.label_regions(mask), 1)

    def test_runs_in_row_needs_the_minimum_width(self):
        row = np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], bool)
        self.assertEqual(fm.runs_in_row(row, 4), 2)
        self.assertEqual(fm.runs_in_row(row, 2), 3)

    def test_flame_span_is_where_the_slice_peak_reaches_the_live_fraction(self):
        peaks = np.array([0, .01, .2, 1.0, .5, .04, 0])
        self.assertEqual(fm.flame_span(peaks), (2, 4))
        self.assertIsNone(fm.flame_span(np.zeros(5)))


class ProfileTests(unittest.TestCase):
    def luma(self, peaks):
        field = np.zeros((4, len(peaks), 4), np.float32)
        field[1, :, 1] = peaks
        return field

    def test_a_flame_that_cools_with_height_has_its_core_in_the_lowest_third(self):
        profile = fm.height_profile(self.luma([0, 9, 8, 6, 4, 2, 1, .5, .3]))
        self.assertTrue(profile["core_in_lowest_third"])
        self.assertLess(profile["top_over_bottom"], .2)
        self.assertEqual(profile["span"], [1, 7])

    def test_a_flame_brightest_at_the_top_is_not(self):
        profile = fm.height_profile(self.luma([0, 1, 2, 3, 4, 5, 6, 7, 9]))
        self.assertFalse(profile["core_in_lowest_third"])
        self.assertGreater(profile["top_over_bottom"], 4)

    def test_cells_below_the_floor_are_ignored(self):
        v = volume(np.zeros((4, 9, 4)))
        field = self.luma([9, 9, 1, 1, 1, 1, 1, 1, 1])
        raised = fm.above(field, v, floor_height=.2)
        self.assertEqual(float(raised[:, :2].max()), 0.0)
        self.assertEqual(float(raised[:, 2:].max()), 1.0)
        self.assertIs(fm.above(field, v, 0.0), field)


class TongueTests(unittest.TestCase):
    def test_two_tongues_in_one_slice_count_as_two_and_a_bridge_as_one(self):
        flame = np.zeros((12, 6, 12), np.float32)
        flame[1:4, 2:4, 1:4] = 1.0
        flame[7:10, 2:4, 7:10] = 1.0
        self.assertEqual(fm.tongue_counts(flame)["mean"], 2.0)
        flame[4:7, 2:4, 3] = 1.0
        flame[6, 2:4, 4:8] = 1.0
        self.assertEqual(fm.tongue_counts(flame)["mean"], 1.0)

    def test_a_dim_tongue_beside_a_bright_one_is_not_counted(self):
        flame = np.zeros((12, 3, 12), np.float32)
        flame[1:4, 1, 1:4] = 1.0
        flame[7:10, 1, 7:10] = .2
        self.assertEqual(fm.tongue_counts(flame)["mean"], 1.0)


class EmissionTests(unittest.TestCase):
    SETTINGS = volumerender.VolumeSettings(flame_intensity=1.0, temperature_scale=1000.0, fire_threshold=500.0, flame_follow=1)

    def test_the_emission_cools_up_the_column_when_the_glow_follows_temperature(self):
        flame = np.zeros((4, 8, 4), np.float32)
        flame[1:3, :, 1:3] = 1.0
        temperature = np.zeros_like(flame)
        temperature[1:3, :, 1:3] = np.linspace(2.5, .8, 8)[None, :, None]
        v = volume(flame, temperature)
        profile = fm.height_profile(fm.emission_luminance(v, self.SETTINGS))
        self.assertTrue(profile["core_in_lowest_third"])
        self.assertLess(profile["top_over_bottom"], .2)
        fixed = fm.height_profile(fm.emission_luminance(v, replace(self.SETTINGS, flame_follow=0)))
        self.assertAlmostEqual(fixed["top_over_bottom"], 1.0, places=3)

    def test_colour_gets_redder_with_height_and_summarise_averages_frames(self):
        flame = np.zeros((4, 9, 4), np.float32)
        flame[1:3, :, 1:3] = 1.0
        temperature = np.zeros_like(flame)
        temperature[1:3, :, 1:3] = np.linspace(2.5, 1.0, 9)[None, :, None]
        v = volume(flame, temperature)
        colour = fm.colour_by_height(v, self.SETTINGS)
        self.assertGreater(colour["bottom_g_over_r"], colour["top_g_over_r"])
        entry = fm.frame_metrics(v, self.SETTINGS)
        summary = fm.summarise({"1": entry, "2": entry})
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["core_in_lowest_third_frames"], 2)


class EddyTests(unittest.TestCase):
    def test_a_rotating_flow_has_vorticity_and_a_uniform_one_has_none(self):
        n = 12
        x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        density = np.ones((n, n, n), np.float32)
        spin = np.stack((-(z - n / 2), np.zeros_like(x), (x - n / 2)), -1).astype(np.float32)
        calm = np.zeros_like(spin)
        spinning = fm.smoke_eddies(volume(np.zeros_like(density), density=density, velocity=spin))
        still = fm.smoke_eddies(volume(np.zeros_like(density), density=density, velocity=calm))
        self.assertGreater(spinning["vorticity_mean"], 1.0)
        self.assertEqual(still["vorticity_mean"], 0.0)
        self.assertEqual(still["density_contrast"], 0.0)

    def test_burning_cells_do_not_count_as_smoke(self):
        noise = np.random.default_rng(1).uniform(0, 2, (6, 12, 12)).astype(np.float32)
        density = np.ones((12, 12, 12), np.float32)
        in_flame, in_smoke = density.copy(), density.copy()
        in_flame[:6] = noise
        in_smoke[6:] = noise
        flame = np.zeros_like(density)
        flame[:6] = 1.0
        self.assertLess(fm.smoke_eddies(volume(flame, density=in_flame))["density_contrast"],
                        .5 * fm.smoke_eddies(volume(flame, density=in_smoke))["density_contrast"])

if __name__ == "__main__":
    unittest.main()


class UpresScaleTests(unittest.TestCase):
    """Lane 6, smoke up-res step: the measures read an up-res volume (2 or 4 times finer) over the same physical size."""

    def test_cell_scale_follows_the_voxel_size(self):
        for factor in (1, 2, 4):
            v = replace(volume(np.zeros((4, 4, 4))), voxel_size=fm.REFERENCE_VOXEL / factor)
            self.assertEqual(fm.cell_scale_of(v), factor)

    def test_the_blur_covers_the_same_physical_size_at_every_resolution(self):
        n = 16
        x = np.arange(n * 2)
        coarse = np.where((np.arange(n) // 2) % 2 == 0, 1.0, .2).astype(np.float32)[:, None, None] * np.ones((n, n, n), np.float32)
        fine = np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), 2, axis=2)
        a = replace(volume(np.zeros_like(coarse), density=coarse), voxel_size=fm.REFERENCE_VOXEL)
        b = replace(volume(np.zeros_like(fine), density=fine), voxel_size=fm.REFERENCE_VOXEL / 2)
        self.assertAlmostEqual(fm.smoke_eddies(a)["density_contrast"], fm.smoke_eddies(b)["density_contrast"], delta=.06)
        self.assertEqual(len(x), n * 2)

    def test_box_blur_matches_the_direct_five_cell_blur(self):
        d = np.random.default_rng(1).random((9, 10, 11)).astype(np.float32)
        blur = d.copy()
        for axis in range(3):
            padded = np.pad(blur, [(2, 2) if a == axis else (0, 0) for a in range(3)], mode="edge")
            blur = sum(np.take(padded, np.arange(k, k + d.shape[axis]), axis=axis) for k in range(5)) / 5
        np.testing.assert_allclose(fm._box_blur(d, 5), blur, atol=1e-6)

    def test_tongue_regions_scale_with_the_cell_size(self):
        flame = np.zeros((16, 4, 16), np.float32)
        flame[2:4, :, 2:4] = 1.0                       # 4 cells per slice: below the 6-cell minimum
        self.assertEqual(fm.tongue_counts(flame)["mean"], 0.0)
        flame[2:6, :, 2:6] = 1.0                       # 16 cells: a tongue at scale 1, but not at scale 2 (24 needed)
        self.assertEqual(fm.tongue_counts(flame)["mean"], 1.0)
        self.assertEqual(fm.tongue_counts(flame, cell_scale=2)["mean"], 0.0)

    def test_flame_field_core_reads_the_flame_channel_alone(self):
        flame = np.zeros((4, 10, 4), np.float32)
        flame[1:3, :, 1:3] = np.linspace(1.0, .2, 10)[None, :, None]
        self.assertTrue(fm.flame_field_core(flame)["in_lowest_third"])
        self.assertEqual(fm.flame_field_core(flame)["position"], 0.0)
        flame = flame[:, ::-1]
        self.assertFalse(fm.flame_field_core(flame)["in_lowest_third"])


class SmokeImageStructureTests(unittest.TestCase):
    def _images(self, pattern):
        rng = np.random.default_rng(0)
        h, w = 120, 160
        yy, xx = np.mgrid[:h, :w]
        base = np.zeros((h, w, 4))
        base[..., :3] = (.5 + pattern * np.sin(xx / 3.) * np.sin(yy / 4.))[..., None]
        return base + rng.normal(0, .05, base.shape), base + rng.normal(0, .05, base.shape)

    def test_shared_pattern_counts_and_independent_noise_does_not(self):
        mask = np.ones((120, 160), bool)
        patterned = fm.smoke_structure(*self._images(.2), mask)
        flat = fm.smoke_structure(*self._images(0.0), mask)
        self.assertGreater(patterned["structure"], .1)
        self.assertLess(flat["structure"], .02)
        self.assertAlmostEqual(patterned["noise_rms_over_mean"], flat["noise_rms_over_mean"], delta=.01)

    def test_smoke_mask_leaves_out_the_dark_background_and_the_flame(self):
        image = np.zeros((50, 50, 4))
        image[..., :3] = np.linspace(.2, .5, 50)[None, :, None]
        image[:10, :10, :3] = .01                      # background
        image[45:, 45:, :3] = 20.0                     # flame (1 % of the pixels)
        mask = fm.smoke_mask(image)
        self.assertFalse(mask[:10, :10].any())
        self.assertFalse(mask[45:, 45:].any())
        self.assertTrue(mask[20, 35])                  # brighter than the median, not the flame
        self.assertFalse(mask[20, 2])                  # dimmer than the median
