"""Run Python's cyclic garbage collector on the GUI thread while a window is open.

Python collects reference cycles on whichever thread happens to allocate when a collection is due.
In this app that is often a render worker (numpy in scene3d, the tile executor), and a cycle can hold
the last reference to a Qt object Python owns: a closed window's offscreen GL surface (an invisible
QWindow on platforms without native offscreen surfaces), its signal hubs, a Properties panel taken
out of the dock, graphics items. Qt objects must be destroyed on the GUI thread; one destroyed on a
worker while the GUI thread builds a widget crashed Desktop conformance on 10/20 (batch 16 of run
39441912895: a render worker "Garbage-collecting" segfaulted under test_render_layout, after
"Internal C++ object (NodeItem) already deleted").

While at least one window is alive, automatic collection is switched off and a GUI-thread timer
collects instead, with the interpreter's own thresholds. When the last window is gone, a final
collection runs on the GUI thread and the previous setting comes back, so command-line renders and
tests without a window collect as usual.
"""
import gc

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer

INTERVAL_MS = 200


def _generation_due():
    """The oldest generation whose count passed its threshold, or -1, as the interpreter decides."""
    due = -1
    for generation, (count, threshold) in enumerate(zip(gc.get_count(), gc.get_threshold())):
        if threshold > 0 and count > threshold:
            due = generation
        elif generation == 0:
            return -1
    return due


class _Collector(QObject):
    def __init__(self, app):
        super().__init__(app)
        self.windows = 0
        self.restore = None   # gc.isenabled() before the first window, while collection is ours
        self.timer = QTimer(self)
        self.timer.setInterval(INTERVAL_MS)
        self.timer.timeout.connect(self.collect)

    def collect(self):
        generation = _generation_due()
        if generation >= 0:
            gc.collect(generation)

    def attach(self, window):
        self.windows += 1
        if self.restore is None:
            self.restore = gc.isenabled()
            gc.disable()
            self.timer.start()
        window.destroyed.connect(self.detach)

    def detach(self, *_):
        self.windows -= 1
        if self.windows <= 0:
            # Not from inside the window's destructor: finish on the next turn of the GUI thread.
            QTimer.singleShot(0, self, self.release)

    def release(self):
        """With no window left, collect once here on the GUI thread and hand collection back."""
        if self.windows > 0 or self.restore is None:
            return
        self.timer.stop()
        gc.collect()
        if self.restore:
            gc.enable()
        self.restore = None


_collector = None


def attach(window):
    """Collect cycles on the GUI thread for as long as `window` lives. Call from its constructor."""
    global _collector
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        return
    if _collector is None or not _valid(_collector):
        _collector = _Collector(app)
    _collector.attach(window)


def release_if_idle():
    """Run the final collection now if no window is left (tests, which run no event loop)."""
    if _collector is not None and _valid(_collector):
        _collector.release()


def _valid(obj):
    import shiboken6
    return shiboken6.isValid(obj)
