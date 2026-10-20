"""Reference cycles that hold Qt objects are collected on the GUI thread while a window is open
(nodebased/guigc.py). On 10/20 the garbage collector ran on a render worker in the middle of
test_render_layout and destroyed Qt objects there while the GUI thread built a widget: a segfault
that took down batch 16 of Desktop conformance run 39441912895."""
import gc
import threading
import unittest

from PySide6.QtCore import QObject
from PySide6.QtGui import QOffscreenSurface
from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401
from nodebased import guigc
from nodebased.app import Window
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])


class Node:
    pass


class Probe(QObject):
    """Notes the thread its wrapper is freed on, which is where Python deletes the Qt object it owns."""
    def __init__(self, record):
        super().__init__()
        self.record = record

    def __del__(self):
        self.record.append(threading.current_thread().name)


def make_garbage_cycle_holding_a_qobject(record):
    """A Python-owned QObject whose only reference is a cycle, as a closed window's leftovers are."""
    holder = Node()
    holder.me = holder
    holder.qobject = Probe(record)


def churn_on_a_worker():
    """Allocate enough container objects on a worker to pass the young generation's threshold
    several times over, as numpy and the tile executor do during a render."""
    def work():
        for _ in range(20):
            junk = [Node() for _ in range(gc.get_threshold()[0] * 2)]
            for a, b in zip(junk, junk[1:]):
                a.next, b.prev = b, a
            del junk
    thread = threading.Thread(target=work, name="render-worker")
    thread.start()
    thread.join()


class GuiThreadCollectionTests(unittest.TestCase):
    def tearDown(self):
        window = self.__dict__.pop("window", None)
        if window is not None:
            window.saved_document = window.dispatcher.document
            window.close()
        APP.processEvents()

    def open(self):
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        return self.window

    def test_a_cycle_holding_a_qt_object_is_freed_on_the_gui_thread_not_a_worker(self):
        self.open()
        record = []
        make_garbage_cycle_holding_a_qobject(record)
        churn_on_a_worker()
        self.assertEqual(record, [], "the cycle was collected on the worker")
        self.assertTrue(wait_until(lambda: record))
        self.assertEqual(record, [threading.main_thread().name])

    def test_automatic_collection_comes_back_when_the_last_window_is_gone(self):
        enabled = gc.isenabled()
        self.open()
        self.assertFalse(gc.isenabled())
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.deleteLater()
        del self.window
        APP.sendPostedEvents(None, tests.isolation.QEvent.Type.DeferredDelete)
        guigc.release_if_idle()
        self.assertEqual(gc.isenabled(), enabled)

    def test_the_offscreen_gl_surface_goes_with_its_window(self):
        w = self.open()
        surface = w._gpu_surface
        self.assertIsInstance(surface, QOffscreenSurface)
        self.assertIs(surface.parent(), w)


if __name__ == "__main__":
    unittest.main()
