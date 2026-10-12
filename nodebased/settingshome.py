"""A private settings folder for a headless session (HeadlessSession in mcpclient).

Qt keeps QSettings in the registry on Windows, so a HOME or XDG_CONFIG_HOME override does not move
them. When NODEBASED_SETTINGS_HOME names a folder, every QSettings in this process reads and
writes an ini file under it instead; on Linux that is the file XDG_CONFIG_HOME already pointed to.
"""
import os

from PySide6.QtCore import QSettings

ENV = "NODEBASED_SETTINGS_HOME"


def apply():
    folder = os.environ.get(ENV)
    if not folder:
        return
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, folder)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
