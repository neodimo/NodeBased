"""The kept axis tables of a translation-only cubic Transform plan (nodebased/gpu2d.py `_axis_tables`).

A frame that moves the picture along one axis reuses the other axis's tables. The plan must be exactly the plan a rebuild
gives, for every parameter that reaches the tables, and the memory of tables must stay bounded and read-only.
"""
import itertools
import unittest

import numpy as np

from nodebased import gpu2d
from nodebased.tiers import Region
from tests.test_gpu2d import params


def plan(**over):
    src = Region(0, 0, 64, 48)
    dst = Region(over.pop("dst_x", -5), over.pop("dst_y", -4), 74, 56)
    return gpu2d.map_transform(params("Transform", filter="cubic", **over), src, dst)


def rebuilt(**over):
    gpu2d._AXIS_TABLES.clear()
    return plan(**over)


class AxisTableTests(unittest.TestCase):
    def setUp(self):
        gpu2d._AXIS_TABLES.clear()
        self.addCleanup(gpu2d._AXIS_TABLES.clear)

    def assert_same_plan(self, kept, fresh, label):
        self.assertEqual((kept.mode, kept.filter, kept.clamp, kept.black_outside, kept.zero_rows, kept.weighted, kept.mix),
                         (fresh.mode, fresh.filter, fresh.clamp, fresh.black_outside, fresh.zero_rows, fresh.weighted, fresh.mix), label)
        self.assertEqual(kept.shift, fresh.shift, label)
        if kept.tables is None:                     # a whole-pixel move is a copy and has no tables
            self.assertIsNone(fresh.tables, label)
            return
        self.assertEqual(kept.tables.dtype, fresh.tables.dtype, label)
        self.assertEqual(kept.tables.tobytes(), fresh.tables.tobytes(), label)

    def test_a_kept_axis_gives_the_plan_a_rebuild_gives(self):
        cases = [dict(translate_x=tx, translate_y=ty, clamp=clamp)
                 for tx, ty, clamp in itertools.product((-18.0, -17.234042553191486, 0.25, 4.5, 4.212765957446809),
                                                        (0.0, 0.5, -3.75, 2.0), (0, 1))]
        for lap in range(2):                        # the second lap runs on tables kept by the first
            for case in cases:
                kept = plan(**case)
                memory = dict(gpu2d._AXIS_TABLES)
                gpu2d._AXIS_TABLES.clear()
                fresh = plan(**case)
                gpu2d._AXIS_TABLES.clear()
                gpu2d._AXIS_TABLES.update(memory)
                self.assert_same_plan(kept, fresh, (lap, case))

    def test_a_move_along_x_reuses_the_y_tables_and_a_new_window_does_not(self):
        plan(translate_x=1.5, translate_y=0.5)
        first = dict(gpu2d._AXIS_TABLES)
        plan(translate_x=2.5, translate_y=0.5)                      # x moved: one new axis, the y tables are the same objects
        self.assertEqual(len(gpu2d._AXIS_TABLES), len(first) + 1)
        shared = [key for key in first if first[key] is gpu2d._AXIS_TABLES[key]]
        self.assertEqual(len(shared), len(first))
        before = len(gpu2d._AXIS_TABLES)
        plan(translate_x=2.5, translate_y=0.5, dst_y=-6)            # another output window builds new tables
        self.assertGreater(len(gpu2d._AXIS_TABLES), before)
        plan(translate_x=1.5, translate_y=0.5, clamp=1)             # clamp changes the weights
        self.assertGreater(len(gpu2d._AXIS_TABLES), before)

    def test_the_kept_tables_cannot_be_changed_through_a_plan(self):
        kept = plan(translate_x=1.5, translate_y=0.5)
        for entry in gpu2d._AXIS_TABLES.values():
            for array in entry:
                self.assertFalse(array.flags.writeable)
        kept.tables[:] = 0                          # the plan owns its own copy
        again = plan(translate_x=1.5, translate_y=0.5)
        self.assertTrue(np.any(again.tables != 0))

    def test_the_memory_of_axes_is_bounded(self):
        for step in range(gpu2d._AXIS_TABLES_LIMIT * 3):
            plan(translate_x=0.25 + 0.01 * step, translate_y=0.5)
        self.assertLessEqual(len(gpu2d._AXIS_TABLES), gpu2d._AXIS_TABLES_LIMIT)

    def test_rotation_and_scale_do_not_use_the_axis_tables(self):
        plan(translate_x=1.5, rotate=3.0)
        self.assertEqual(len(gpu2d._AXIS_TABLES), 0)


if __name__ == "__main__":
    unittest.main()
