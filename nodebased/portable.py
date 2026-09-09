"""Windows portable bundle staging and out-of-process application replacement."""
from __future__ import annotations
import ctypes
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


def extract_bundle(archive, directory):
    root = Path(tempfile.mkdtemp(prefix='portable-', dir=directory))
    try:
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if len(entries) > 10000 or sum(x.file_size for x in entries) > 2 * 1024 ** 3:
                raise ValueError('Portable update exceeds extraction limits')
            for entry in entries:
                name = PurePosixPath(entry.filename)
                if (name.is_absolute() or '..' in name.parts or not name.parts or
                    name.parts[0] != 'NodeBased' or '\\' in entry.filename or ':' in entry.filename or
                    stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError('Invalid path in portable update')
            bundle.extractall(root)
        source = root / 'NodeBased'
        if not (source / 'NodeBased.exe').is_file() or not (source / 'portable.marker').is_file() or not (source / '_internal').is_dir():
            raise ValueError('Portable update is missing required bundle files')
        return source
    except Exception:
        shutil.rmtree(root)
        raise


def apply_bundle(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if not (target / 'portable.marker').is_file():
        raise ValueError('The update target is not a portable NodeBased folder')
    stage = Path(tempfile.mkdtemp(prefix='.nodebased-next-', dir=target))
    backup = target / 'portable-data' / 'previous-version'
    moved_old, moved_new = [], []
    try:
        shutil.copy2(source / 'NodeBased.exe', stage / 'NodeBased.exe')
        shutil.copytree(source / '_internal', stage / '_internal')
        if backup.exists():
            shutil.rmtree(backup)
        backup.mkdir(parents=True)
        for name in ('NodeBased.exe', '_internal'):
            if (target / name).exists():
                os.replace(target / name, backup / name)
                moved_old.append(name)
            os.replace(stage / name, target / name)
            moved_new.append(name)
    except Exception:
        for name in reversed(moved_new):
            path = target / name
            if path.is_dir(): shutil.rmtree(path)
            elif path.exists(): path.unlink()
        for name in reversed(moved_old):
            os.replace(backup / name, target / name)
        raise
    finally:
        shutil.rmtree(stage)


def wait_for_windows_process(pid):
    if pid <= 0:
        raise ValueError('Invalid application process ID')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    process = kernel.OpenProcess(0x100000, False, pid)
    if not process:
        if ctypes.get_last_error() == 87:  # process has already exited
            return
        raise OSError('Cannot wait for NodeBased to close')
    try:
        if kernel.WaitForSingleObject(process, 60000) != 0:
            raise OSError('NodeBased is still running; close it and try the update again')
    finally:
        kernel.CloseHandle(process)


def helper_main(target, pid, restart_args=None):
    try:
        wait_for_windows_process(int(pid))
        apply_bundle(Path(sys.executable).parent, target)
        env = os.environ.copy()
        env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
        subprocess.Popen([str(Path(target) / 'NodeBased.exe'), *(restart_args or [])], env=env, close_fds=True)
        return 0
    except Exception as error:
        ctypes.windll.user32.MessageBoxW(None, str(error), 'NodeBased update failed', 0x10)
        return 1
