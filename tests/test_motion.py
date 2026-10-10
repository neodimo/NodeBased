import unittest

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from nodebased.motion import MotionAnimator
from tests.waiting import wait_until


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
        # Wait for the end value rather than a fixed 80 ms: a loaded machine can starve the timer (10/9 merge run).
        self.assertTrue(wait_until(lambda: values and values[-1] == 10 and not animator._timer.isActive(), 5.0))
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
        # Delete now, then require that nothing more is emitted. A fixed 30 ms wait let a busy machine fire the
        # first frame before the deletion was processed (10/9 merge run).
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        emitted = len(values)
        QTest.qWait(60)
        animator.finish_all()
        self.assertEqual(len(values), emitted)
        animator.deleteLater()


if __name__ == "__main__":
    unittest.main()
