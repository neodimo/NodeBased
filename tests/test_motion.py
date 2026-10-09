import unittest

from PySide6.QtCore import QObject, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from nodebased.motion import MotionAnimator


APP = QApplication.instance() or QApplication([])


class MotionAnimatorTests(unittest.TestCase):
    def test_reduced_motion_applies_the_end_value_immediately(self):
        target = QWidget()
        values = []
        animator = MotionAnimator(enabled=False)
        animator.animate("x", target, 0, 10, values.append, 120)
        self.assertEqual(values, [10])
        self.assertFalse(animator._timer.isActive())
        animator.deleteLater()
        target.deleteLater()

    def test_animation_reaches_its_end_value(self):
        target = QWidget()
        values = []
        animator = MotionAnimator(enabled=True)
        animator.animate("x", target, 0, 10, values.append, 35)
        QTest.qWait(80)
        self.assertEqual(values[-1], 10)
        self.assertFalse(animator._timer.isActive())
        animator.deleteLater()
        target.deleteLater()

    def test_enabling_reduce_motion_finishes_every_live_track(self):
        target = QWidget()
        values = []
        animator = MotionAnimator(enabled=True)
        animator.animate("x", target, 0, 10, values.append, 1000)
        animator.set_enabled(False)
        self.assertEqual(values[-1], 10)
        self.assertFalse(animator._timer.isActive())
        animator.deleteLater()
        target.deleteLater()

    def test_target_closed_mid_animation_is_dropped_safely(self):
        target = QWidget()
        values = []
        animator = MotionAnimator(enabled=True)
        animator.animate("x", target, 0, 10, values.append, 100)
        target.deleteLater()
        QTimer.singleShot(0, APP.processEvents)
        QTest.qWait(30)
        animator.finish_all()
        self.assertEqual(values, [])
        animator.deleteLater()


if __name__ == "__main__":
    unittest.main()
