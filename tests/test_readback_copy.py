"""The readback of a rendered 1080p float picture is copied by several threads (Lane 4, 10/23, step B part 2).

Before, one NumPy copy of the 33 MB took 2.7 to 3.0 ms on the RTX, the Radeon and the laptop stand-in alike: the one stage the
timelines of all four test scenes share, and 28 percent of a cube frame's time in the render worker."""
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d


class Mapped:
    """Stands in for a mapped staging buffer: rows of `stride` bytes."""

    def __init__(self, array):
        self.array = array

    def read_mapped(self, copy=True):
        assert copy is False, "the copy is made by NumPy, once"
        return memoryview(self.array.reshape(-1).view(np.uint8))


def picture(rows, width, stride_floats, seed=3):
    data = np.random.default_rng(seed).random((rows, stride_floats), dtype=np.float32)
    return data, Mapped(data)


class ThreadedCopy(unittest.TestCase):
    def copy(self, rows, width, y0, y1, stride_floats, parts, min_bytes=1):
        data, staging = picture(y1 - y0, width, stride_floats)
        result = np.zeros((rows, width, 4), np.float32)
        with unittest.mock.patch.object(gpu3d, "READBACK_PARTS", parts), unittest.mock.patch.object(gpu3d, "READBACK_MIN_BYTES", min_bytes):
            gpu3d._copy_band(staging, result, y0, y1, np.dtype(np.float32), stride_floats * 4, width)
        return data, result

    def test_every_row_lands_where_a_single_copy_puts_it(self):
        width, stride = 301, 301 * 4 + 12          # a padded row, as the 256-byte alignment of a readback gives
        for parts in (1, 2, 3, 4):
            data, result = self.copy(40, width, 5, 37, stride, parts)
            expected = data[:, :width * 4].reshape(32, width, 4)
            np.testing.assert_array_equal(result[5:37], expected)
            self.assertFalse(result[:5].any() or result[37:].any(), "rows outside the band stay untouched")

    def test_a_band_with_fewer_rows_than_threads_and_a_single_row(self):
        for rows in (1, 2, 3):
            data, result = self.copy(rows, 16, 0, rows, 16 * 4, 4)
            np.testing.assert_array_equal(result, data.reshape(rows, 16, 4))

    def test_a_big_band_is_split_between_threads_and_a_small_one_is_not(self):
        pool = unittest.mock.Mock()
        pool.submit.side_effect = lambda fn, *a: unittest.mock.Mock(result=lambda fn=fn, a=a: fn(*a))
        with unittest.mock.patch.object(gpu3d, "_readback_threads", return_value=pool):
            self.copy(64, 128, 0, 64, 128 * 4, 3, min_bytes=1)
            self.assertEqual(pool.submit.call_count, 2)          # the calling thread takes the first slice itself
            pool.submit.reset_mock()
            self.copy(64, 128, 0, 64, 128 * 4, 3, min_bytes=10 ** 9)
            pool.submit.assert_not_called()
            self.copy(64, 128, 0, 64, 128 * 4, 1, min_bytes=1)
            pool.submit.assert_not_called()

    def test_the_threads_follow_the_processors_the_process_may_use(self):
        with unittest.mock.patch.object(gpu3d.os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True):
            self.assertEqual(gpu3d._cpus(), 4)
        self.assertGreaterEqual(gpu3d.READBACK_PARTS, 1)
        self.assertLessEqual(gpu3d.READBACK_PARTS, 4)

    def test_a_1080p_float_band_is_over_the_threshold(self):
        self.assertGreater(1920 * 1080 * 16, gpu3d.READBACK_MIN_BYTES)
        self.assertLess(256 * 144 * 16, gpu3d.READBACK_MIN_BYTES)


if __name__ == "__main__":
    unittest.main()
