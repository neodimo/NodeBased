"""NodeBased: explicit graph state, scene-linear images, native desktop UI."""
__version__ = "0.45.0"

import os as _os
import sys as _sys

# Qt's offscreen platform on Windows reads fonts from QT_QPA_FONTDIR (Qt's own lib/fonts, which PySide6 does not
# ship), so a headless render or test there would draw every Text and BurnIn without real fonts: Bold the same as
# Regular, lines wrapped by a stand-in. Point it at the system fonts unless the caller chose a folder.
if (_sys.platform == "win32" and _os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen")
        and not _os.environ.get("QT_QPA_FONTDIR")):
    _os.environ["QT_QPA_FONTDIR"] = _os.path.join(_os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
