"""Step 3a part 1: the eleven Merge operations added to reach Nuke's 30, on hand-computed pixels."""
import unittest

import numpy as np

from nodebased.core import CHOICES
from nodebased.imaging import Evaluator

NEW_OPS = ("matte", "disjoint-over", "conjoint-over", "copy", "exclusion", "geometric", "overlay",
           "hard-light", "soft-light", "color-dodge", "color-burn")


def px(*v):
    return np.array([[list(v)]], np.float32)


def merge(op, a, b):
    return Evaluator._kernel("Merge", {"operation": op, "mix": 1.0}, [a, b])


class NewMergeOperationTests(unittest.TestCase):
    def check(self, op, a, b, expected):
        np.testing.assert_allclose(merge(op, a, b), px(*expected), rtol=1e-5, atol=1e-6, err_msg=op)
        self.assertTrue(np.isfinite(merge(op, a, b)).all())

    def test_all_eleven_are_registered(self):
        for op in NEW_OPS:
            self.assertIn(op, CHOICES["operation"])
        self.assertEqual(len(CHOICES["operation"]), 30)

    def test_matte(self):
        # A*a + B*(1-a): 0.5*0.5 + 0.4*0.5 = 0.45; alpha 0.25 + 0.5 = 0.75
        self.check("matte", px(0.5, 0.2, 0.0, 0.5), px(0.4, 0.4, 0.4, 1.0), (0.45, 0.2 * 0.5 + 0.2, 0.2, 0.75))

    def test_disjoint_over_when_alphas_sum_below_one(self):
        self.check("disjoint-over", px(0.2, 0.0, 0.0, 0.2), px(0.3, 0.1, 0.0, 0.3), (0.5, 0.1, 0.0, 0.5))

    def test_disjoint_over_when_alphas_sum_reaches_one(self):
        # 0.5 + 0.4*(1-0.6)/0.8 = 0.7 ; alpha 0.6 + 0.8*0.4/0.8 = 1.0
        self.check("disjoint-over", px(0.5, 0.0, 0.0, 0.6), px(0.4, 0.0, 0.0, 0.8), (0.7, 0.0, 0.0, 1.0))

    def test_disjoint_over_zero_b_alpha_is_finite(self):
        self.check("disjoint-over", px(0.5, 0.0, 0.0, 1.0), px(0.4, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 1.0))

    def test_conjoint_over_when_a_alpha_is_smaller(self):
        # 0.25 + 0.4*(1 - 0.25/0.5) = 0.45 ; alpha 0.25 + 0.5*(1-0.5) = 0.5
        self.check("conjoint-over", px(0.25, 0.0, 0.0, 0.25), px(0.4, 0.0, 0.0, 0.5), (0.45, 0.0, 0.0, 0.5))

    def test_conjoint_over_when_a_alpha_is_larger_returns_a(self):
        self.check("conjoint-over", px(0.5, 0.1, 0.0, 0.5), px(0.4, 0.9, 0.9, 0.25), (0.5, 0.1, 0.0, 0.5))

    def test_conjoint_over_zero_b_alpha_is_finite(self):
        self.check("conjoint-over", px(0.5, 0.0, 0.0, 0.0), px(0.4, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0))

    def test_copy_returns_a(self):
        a = px(0.3, 1.7, -0.2, 0.6)
        self.check("copy", a, px(9, 9, 9, 9), (0.3, 1.7, -0.2, 0.6))
        self.assertIsNot(merge("copy", a, px(0, 0, 0, 0)), a)

    def test_exclusion(self):
        # A+B-2AB: 0.5+0.5-0.5 = 0.5 ; 1+0-0 = 1 ; 1.5+2-6 = -2.5 (HDR) ; alpha 1+1-2 = 0
        self.check("exclusion", px(0.5, 1.0, 1.5, 1.0), px(0.5, 0.0, 2.0, 1.0), (0.5, 1.0, -2.5, 0.0))

    def test_geometric(self):
        # 2AB/(A+B): 2*.2*.6/.8 = 0.3 ; A+B=0 -> 0 ; 2*4*4/8 = 4
        self.check("geometric", px(0.2, 0.0, 4.0, 1.0), px(0.6, 0.0, 4.0, 1.0), (0.3, 0.0, 4.0, 1.0))

    def test_geometric_opposite_signs_cancelling_is_finite(self):
        self.check("geometric", px(0.5, 0.0, 0.0, 1.0), px(-0.5, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

    def test_overlay_below_half_multiplies(self):
        self.check("overlay", px(0.5, 0.0, 0.0, 1.0), px(0.25, 0.0, 0.0, 1.0), (0.25, 0.0, 0.0, 1.0))

    def test_overlay_above_half_screens(self):
        # 1-2*(1-0.5)*(1-0.75) = 0.75
        self.check("overlay", px(0.5, 0.0, 0.0, 1.0), px(0.75, 0.0, 0.0, 1.0), (0.75, 0.0, 0.0, 1.0))

    def test_overlay_at_exactly_half_returns_a_from_both_branches(self):
        self.check("overlay", px(0.3, 0.9, 0.0, 1.0), px(0.5, 0.5, 0.5, 1.0), (0.3, 0.9, 0.0, 1.0))

    def test_hard_light_is_overlay_with_roles_swapped(self):
        a = px(0.25, 0.75, 0.5, 1.0)
        b = px(0.6, 0.3, 0.9, 1.0)
        np.testing.assert_allclose(merge("hard-light", a, b), merge("overlay", b, a), atol=1e-7)
        self.check("hard-light", px(0.25, 0.75, 0.0, 1.0), px(0.6, 0.3, 0.0, 1.0),
                   (2 * 0.25 * 0.6, 1 - 2 * 0.25 * 0.7, 0.0, 1.0))

    def test_soft_light_both_branches(self):
        # A=B=0.5: AB=.25 -> .5*(1+.5*.75)=0.6875 ; A=B=2 (HDR): AB=4 -> 2AB=8 ; alpha AB=1 -> 2
        self.check("soft-light", px(0.5, 2.0, 0.0, 1.0), px(0.5, 2.0, 0.0, 1.0), (0.6875, 8.0, 0.0, 2.0))

    def test_color_dodge(self):
        # B/(1-A): 0.3/0.5 = 0.6
        self.check("color-dodge", px(0.5, 0.0, 0.0, 1.0), px(0.3, 0.7, 0.0, 1.0), (0.6, 0.7, 0.0, 1.0))

    def test_color_dodge_zero_and_hdr_divisors_are_finite(self):
        # 1-A == 0 and 1-A < 0: limit is 1 where B>0, else 0
        self.check("color-dodge", px(1.0, 3.0, 1.0, 1.0), px(0.4, 0.4, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0))

    def test_color_burn(self):
        # 1-(1-B)/A: 1-0.5/0.5 = 0 ; 1-0.25/0.5 = 0.5
        self.check("color-burn", px(0.5, 0.5, 0.0, 1.0), px(0.5, 0.75, 0.0, 1.0), (0.0, 0.5, 0.0, 1.0))

    def test_color_burn_zero_divisor_is_finite(self):
        self.check("color-burn", px(0.0, 0.0, -1.0, 1.0), px(0.4, 1.0, 0.4, 1.0), (0.0, 1.0, 0.0, 1.0))

    def test_channel_merge_shares_the_vocabulary(self):
        a = np.array([[[0.5, 0.0, 0.0, 1.0]]], np.float32)
        b = np.array([[[0.25, 0.0, 0.0, 1.0]]], np.float32)
        for op in NEW_OPS:
            out = Evaluator._kernel("ChannelMerge", {"operation": op, "a_channel": "A.r",
                                                     "b_channel": "B.r", "out_channel": "R",
                                                     "mix": 1.0}, [a, b])
            self.assertTrue(np.isfinite(out).all(), op)
        one = Evaluator._kernel("ChannelMerge", {"operation": "exclusion", "a_channel": "A.r",
                                                 "b_channel": "B.r", "out_channel": "R",
                                                 "mix": 1.0}, [a, b])
        self.assertAlmostEqual(float(one[0, 0, 0]), 0.5 + 0.25 - 2 * 0.5 * 0.25, places=6)
        self.assertEqual(float(one[0, 0, 3]), 1.0)  # alpha stays B's


if __name__ == "__main__":
    unittest.main()
