"""Lane 11 step B6 (finish): Reformat breaks a tie between two rows, and Transform's filters read the picture's edge, the way Nuke 17 does.

Halving a 16x12 picture with the impulse filter puts every output sample exactly between two source rows. Nuke 17.0v6's own render
(11/21, the reference pack's reformat_scale_type_halves) keeps the source's rows 0, 2, 4... from the top and its columns 1, 3, 5...;
NodeBased's older reading kept rows 1, 3, 5.... A Reformat saved before the change keeps that reading through legacy_reformat_ties.
"""
import unittest

import numpy as np

from nodebased import core
import copy

from nodebased.core import Dispatcher, empty_document, upgrade_document
from nodebased.imaging import Evaluator


def _picture():
    rng = np.random.default_rng(7)
    return rng.random((12, 16, 4)).astype(np.float32)


class ReformatTiesTests(unittest.TestCase):
    def _run(self, **extra):
        from nodebased.raster import Raster
        src = _picture()
        p = dict(core.SPECS["Reformat"]["params"], reformat_type="scale", scale=0.5, resize_type="distort", filter="nearest", **extra)
        out = Evaluator._windowed_kernel("Reformat", p, [Raster.of(src)])
        return src, np.asarray(out.pixels)

    def test_a_new_reformat_keeps_the_rows_nuke_keeps(self):
        self.assertEqual(core.SPECS["Reformat"]["params"]["legacy_reformat_ties"], 0)
        src, out = self._run()
        self.assertEqual(out.shape[:2], (6, 8))
        np.testing.assert_array_equal(out, src[0::2, 1::2])

    def test_the_legacy_flag_keeps_the_older_rows(self):
        src, out = self._run(legacy_reformat_ties=1)
        np.testing.assert_array_equal(out, src[1::2, 1::2])

    def test_a_reformat_saved_before_loads_with_the_legacy_rows(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Reformat", "params": {}})
        saved = copy.deepcopy(d.document)
        del saved["nodes"]["r"]["params"]["legacy_reformat_ties"]
        loaded = upgrade_document(saved) or saved
        self.assertEqual(loaded["nodes"]["r"]["params"]["legacy_reformat_ties"], 1)



class TransformEdgeFilterTests(unittest.TestCase):
    """Transform's filters at the picture's edge, as Nuke 17.0v6 renders them (the reference pack's transform_filter_*_half_pixel, 11/21):
    a tap beyond the picture reads its edge pixel. Cubic is left as it was: the GPU island renders it too."""

    def _shift(self, filter, **extra):
        src = np.tile(np.array([12.0, -0.5, 0.0, 0.05, 0.18, 0.4, 0.5, 0.75], np.float32)[None, :, None], (4, 1, 4))
        p = dict(core.SPECS["Transform"]["params"], translate_x=0.5, filter=filter, **extra)
        out = Evaluator._transform(src, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, filter, params=p)
        return src, np.asarray(out)

    def test_keys_reads_the_edge_pixel_beyond_the_picture(self):
        self.assertEqual(core.SPECS["Transform"]["params"]["legacy_transform_filters"], 0)
        _src, out = self._shift("keys")
        # taps at -2, -1, 0, 1 read 12, 12, 12, -0.5 with Keys' half-pixel weights -1/16, 9/16, 9/16, -1/16: Nuke's 12.78125
        self.assertAlmostEqual(float(out[0, 0, 0]), 12.78125, places=4)

    def test_cubic_stays_keys_which_the_gpu_island_shares(self):
        _src, new = self._shift("cubic")
        _src, old = self._shift("cubic", legacy_transform_filters=1)
        np.testing.assert_array_equal(new, old)

    def test_the_legacy_flag_keeps_black_beyond_the_edge_and_keys_for_cubic(self):
        _src, new = self._shift("keys")
        _src, old = self._shift("keys", legacy_transform_filters=1)
        self.assertNotAlmostEqual(float(old[0, 0, 0]), float(new[0, 0, 0]), places=3)
        np.testing.assert_allclose(old[0, 3:-2], new[0, 3:-2], atol=1e-6)      # inside the picture the two readings agree

    def test_a_transform_saved_before_loads_with_the_legacy_filters(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "t", "type": "Transform", "params": {}})
        saved = copy.deepcopy(d.document)
        del saved["nodes"]["t"]["params"]["legacy_transform_filters"]
        loaded = upgrade_document(saved) or saved
        self.assertEqual(loaded["nodes"]["t"]["params"]["legacy_transform_filters"], 1)


if __name__ == "__main__":
    unittest.main()
