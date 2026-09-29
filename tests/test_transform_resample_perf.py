import os
import time
import unittest

import numpy as np

from nodebased.imaging import Evaluator


@unittest.skipUnless(os.environ.get("NB_PERF") == "1", "set NB_PERF=1 for local performance checks")
class TransformResamplePerformanceTests(unittest.TestCase):
    def test_cubic_960x540_stays_under_generous_local_budget(self):
        h, w = 540, 960
        y, x = np.mgrid[:h, :w].astype(np.float32)
        src = np.ones((h, w, 4), dtype=np.float32)
        start = time.perf_counter()
        result = Evaluator._resample(src, x * .91 + y * .03 + 12.25,
                                     y * .94 - x * .02 + 8.75, "cubic")
        elapsed = time.perf_counter() - start
        self.assertEqual(result.shape, src.shape)
        self.assertLess(elapsed, 0.12)   # 0.04 s here against 0.17 s before the fast gather
