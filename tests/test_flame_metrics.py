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
