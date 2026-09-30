"""M1 gate, part 1: golden-image regression suite (docs/M1_GATE.md).

Locks the pixel output of graphs covering HDR and negative values through Grade, the over/plus/
multiply Merge operations, Premult/Unpremult, Shuffle and ShuffleCopy of named layers, Transform
with each filter, Crop and Reformat with data windows larger and smaller than the format, and a
multichannel EXR Read-to-Write round trip. Every case is checked on both the full-frame evaluator
path and the tile path, except the stated, precedented tile-path exclusions (Transform, Crop,
Reformat -- see tests/golden_2d_cases.py and docs/PARITY_2D.md's own note on Mirror). Regenerate
references with `tests/data/golden/regenerate.py`.
"""
import tempfile
import unittest

import numpy as np

from tests.golden_2d_cases import (
    CASES, assert_close, extra_reference_path, reference_path, run_case,
)


class GoldenImageTests(unittest.TestCase):
    def test_every_case_has_a_reference(self):
        for case in CASES:
            with self.subTest(case=case.name):
                self.assertTrue(reference_path(case.name).exists(),
                                f"missing reference for {case.name}; run "
                                f"tests/data/golden/regenerate.py")

    def test_references_stay_small(self):
        for case in CASES:
            path = reference_path(case.name)
            if path.exists():
                with self.subTest(case=case.name):
                    self.assertLess(path.stat().st_size, 256 * 1024,
                                    f"{path} exceeds the 256 KB golden-reference budget")

    def test_full_frame_matches_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in CASES:
                with self.subTest(case=case.name):
                    pixels, extra, _tiled = run_case(case, tmp)
                    expected = np.load(reference_path(case.name))
                    assert_close(self, pixels, expected, case.atol,
                                f"{case.name} (full-frame)")
                    for key, value in extra.items():
                        expected_extra = np.load(extra_reference_path(case.name, key))
                        assert_close(self, value, expected_extra, case.atol,
                                    f"{case.name}.{key} (full-frame)")

    def test_tile_path_matches_reference_or_is_a_stated_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in CASES:
                with self.subTest(case=case.name):
                    _pixels, _extra, tiled = run_case(case, tmp)
                    if not case.tiled:
                        self.assertIsNone(
                            tiled, f"{case.name} is marked tile-excluded but produced tile "
                                  "output; update golden_2d_cases.py")
                        continue
                    expected = np.load(reference_path(case.name))
                    assert_close(self, tiled, expected, case.atol, f"{case.name} (tile path)")

    def test_tile_excluded_kinds_match_the_precedented_transform_crop_reformat_exclusion(self):
        excluded = {case.name for case in CASES if not case.tiled}
        for case in CASES:
            with self.subTest(case=case.name):
                if case.name.startswith(("transform_", "crop_", "reformat_")):
                    self.assertIn(case.name, excluded)
                else:
                    self.assertNotIn(case.name, excluded)


if __name__ == "__main__":
    unittest.main()
