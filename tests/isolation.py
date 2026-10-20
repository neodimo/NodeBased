"""Keep every test's Qt settings out of the developer's real NodeBased settings, and start every
Window from the default workspace.

The app keeps its layout, theme and recent-node lists in QSettings("NodeBased", "NodeBased"). On
this project's Linux workstation Qt resolves that to ~/.config/NodeBased/NodeBased.conf even when
the suite runner sets XDG_CONFIG_HOME, so test runs read and wrote the real user file, and on
GitHub every test in a batch shares the runner's one file. A test that closed a small window saved
a squashed layout; the next test that opened a window restored it, with the 3D picture 1 px tall,
and test_viewport3d_picking clicked the camera marker instead of the card (10/3, Linux and Windows
CI, and locally depending on what ran before). tests/test_desktop.py had this isolation for itself
only; importing this module gives it to every test.

WorkspaceTests in test_desktop opt back in to real save/restore through `real_workspace`.
"""
import tempfile
import unittest
import unittest.mock

from PySide6.QtCore import QSettings

_SETTINGS_DIR = tempfile.TemporaryDirectory(prefix="nodebased-test-settings-")
for _format in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
    QSettings.setPath(_format, QSettings.Scope.UserScope, _SETTINGS_DIR.name)

from nodebased.app import Preferences  # noqa: E402  (after setPath, before any QSettings use)

if not hasattr(Preferences.workspace, "real_workspace"):
    _REAL_WORKSPACE = Preferences.workspace

    def _no_saved_workspace(self):
        return None

    _no_saved_workspace.real_workspace = _REAL_WORKSPACE
    unittest.mock.patch.object(Preferences, "workspace", _no_saved_workspace).start()

# Jobs that use the whole GPU take an exclusive lock on /tmp/nb-gpu.lock (nodebased/gpulock.py), and the
# suite wrapper holds that file locked for its own run; a stubbed provider launched inside the suite
# would wait for its own parent. Tests get a private lock file instead (tests/test_gpulock.py checks the
# real locking against one).
import os  # noqa: E402

_GPU_LOCK = tempfile.NamedTemporaryFile(prefix="nodebased-test-gpu-", suffix=".lock")
os.environ["NODEBASED_GPU_LOCK"] = _GPU_LOCK.name

# Closed windows are only destroyed by a DeferredDelete event, and nothing delivers one without an event
# loop: `close()` alone never deletes a Window, and `processEvents()` does not deliver what
# `deleteLater()` posts. So every Window a test closed stayed alive (hidden, with its timers and panels)
# for the rest of the process, and each later Window opened slower than the one before, mostly in
# QApplication.setStyleSheet re-polishing every live widget: the eleven windows of one shortcut wiring
# test took 145 s on 10/19 where one takes under a second, and a test_roto_paint_ui test took 10 s
# where alone it takes 1. After every test, windows that accepted a close are deleted, as the app's
# own event loop would do on quit.
from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from nodebased.app import Window  # noqa: E402

if not hasattr(Window.closeEvent, "marks_closed"):
    _CLOSE_EVENT = Window.closeEvent

    def _close_event_and_mark(self, event):
        _CLOSE_EVENT(self, event)
        if event.isAccepted():
            self._closed_by_test = True

    _close_event_and_mark.marks_closed = True
    Window.closeEvent = _close_event_and_mark

_DO_CLEANUPS = unittest.TestCase.doCleanups


def _do_cleanups_and_delete_closed_windows(self):
    outcome = _DO_CLEANUPS(self)
    if QCoreApplication.instance() is not None:
        closed = [w for w in QApplication.topLevelWidgets()
                  if isinstance(w, Window) and getattr(w, "_closed_by_test", False)]
        # closeEvent stops several pools without waiting for them. A thread still running when the
        # Window is deleted can post a result to it, and the next test's processEvents then delivers
        # that to a deleted object (10/20: two segfaults in test_e2e5_ui_punchlist under the parallel
        # runner). Wait for the pools, deliver what they posted, then delete.
        for widget in closed:
            for pool in (getattr(getattr(widget, "queue_panel", None), "runner", None),
                         getattr(getattr(widget, "job_queue", None), "_pool", None)):
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
            decode = getattr(widget, "decode_pool", None)
            if decode is not None:
                decode.shutdown(wait=True, timeout=10)
        if closed:
            QCoreApplication.processEvents()
        for widget in closed:
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return outcome


if not getattr(unittest.TestCase.doCleanups, "deletes_closed_windows", False):
    _do_cleanups_and_delete_closed_windows.deletes_closed_windows = True
    unittest.TestCase.doCleanups = _do_cleanups_and_delete_closed_windows
