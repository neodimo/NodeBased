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
