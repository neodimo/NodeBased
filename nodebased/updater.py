"""Explicit GitHub-release updater for per-user Windows installs and Linux AppImages."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import ssl
import certifi
import subprocess
import sys
import tempfile
import threading
from urllib.request import Request, urlopen

from PySide6.QtCore import QObject, QStandardPaths, Signal
from . import __version__

REPO = 'neodimo/NodeBased'
API = f'https://api.github.com/repos/{REPO}/releases/latest'
MAX_DOWNLOAD = 1024 * 1024 * 1024


def version_tuple(version):
    match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)', version)
    if not match:
        raise ValueError('Unsupported release version')
    return tuple(map(int, match.groups()))


@dataclass(frozen=True)
class Release:
    version: str
    url: str
    name: str
    sha256: str
    size: int


def select_release(data, current=__version__, system=None, portable=False):
    if data.get('draft') or data.get('prerelease'):
        raise ValueError('Only published stable releases are accepted')
    tag = data['tag_name']
    if version_tuple(tag) <= version_tuple(current):
        return None
    system = system or sys.platform
    suffix = {'win32': 'windows-x64-setup.exe', 'linux': 'linux-x86_64.AppImage'}.get(system)
    if not suffix:
        raise ValueError('Updates are not available on this platform yet')
    if system == 'win32' and portable:
        suffix = 'windows-x64-portable.zip'
    version = tag.removeprefix('v')
    name = f'NodeBased-{version}-{suffix}'
    asset = next((a for a in data['assets'] if a['name'] == name), None)
    if not asset:
        raise ValueError(f'This release has no {system} package yet. Try again later.')
    url = asset['browser_download_url']
    if url != f'https://github.com/{REPO}/releases/download/{tag}/{name}':
        raise ValueError('Unexpected release download origin')
    digest = asset.get('digest') or ''
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Release asset is missing a verified SHA-256 digest')
    size = asset['size']
    if type(size) is not int or not 0 < size <= MAX_DOWNLOAD:
        raise ValueError('Invalid release download size')
    return Release(version, url, name, digest[7:], size)


def is_portable():
    return bool(getattr(sys, 'frozen', False) and sys.platform == 'win32' and
                (Path(sys.executable).parent / 'portable.marker').is_file())


def cache_directory():
    if is_portable():
        return Path(sys.executable).parent / 'portable-data' / 'updates'
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / 'updates'


def open_url(request, timeout=30):
    # Packaged Python/OpenSSL must not depend on the build distro's CA file path.
    return urlopen(request, timeout=timeout, context=ssl.create_default_context(cafile=certifi.where()))


def fetch_release():
    req = Request(API, headers={'User-Agent': f'NodeBased/{__version__}', 'Accept': 'application/vnd.github+json'})
    with open_url(req, timeout=20) as handle:
        payload = handle.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError('Release metadata is too large')
    return select_release(json.loads(payload), portable=is_portable())


def download_release(release, directory, progress, cancel=None, opener=open_url):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / release.name
    fd, partial = tempfile.mkstemp(prefix='.download-', dir=directory)
    digest, total = hashlib.sha256(), 0
    try:
        request = Request(release.url, headers={'User-Agent': f'NodeBased/{__version__}'})
        with os.fdopen(fd, 'wb') as output, opener(request, timeout=30) as source:
            while True:
                if cancel and cancel.is_set():
                    raise ValueError('Download cancelled')
                chunk = source.read(256 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > release.size:
                    raise ValueError('Download exceeds the declared size')
                output.write(chunk)
                digest.update(chunk)
                progress(min(99, total * 100 // release.size))
            output.flush()
            os.fsync(output.fileno())
        if total != release.size or digest.hexdigest() != release.sha256:
            raise ValueError('Download integrity check failed; installed application is unchanged')
        os.replace(partial, target)
        progress(100)
        return target
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def verify_download(path, release):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if Path(path).stat().st_size != release.size or digest.hexdigest() != release.sha256:
        raise ValueError('Downloaded update changed; download it again')


def replace_appimage(download, current, launch=subprocess.Popen):
    """Atomic same-filesystem replacement, retain one rollback image."""
    current = Path(current).absolute()
    if current.is_symlink() or not current.is_file():
        raise ValueError('Update the original AppImage file, not a symlink')
    fd, staged = tempfile.mkstemp(prefix='.nodebased-update-', dir=current.parent)
    os.close(fd)
    staged = Path(staged)
    backup = current.with_name(current.name + '.previous')
    try:
        shutil.copyfile(download, staged)
        staged.chmod(current.stat().st_mode | 0o111)
        # Keep the old image present at its original name until atomic replacement.
        shutil.copy2(current, backup)
        os.replace(staged, current)
        env = os.environ.copy()
        # Do not propagate old AppImage mounts or PyInstaller library overrides.
        for key in ('APPIMAGE', 'APPDIR', 'ARGV0', 'LD_LIBRARY_PATH', 'LD_LIBRARY_PATH_ORIG'):
            env.pop(key, None)
        env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
        try:
            launch([str(current)], env=env, start_new_session=True)
        except Exception:
            os.replace(backup, current)
            raise
    finally:
        if staged.exists():
            staged.unlink()


class Updater(QObject):
    changed = Signal(str, str, int)  # state, detail/version, percent

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = 'idle'
        self.release = None
        self.download = None
        self.staged_bundle = None
        self.cancel = threading.Event()
        self.thread = None
        self.changed.connect(self._remember)

    def _remember(self, state, detail, percent):
        self.state = state

    def supported(self):
        if not getattr(sys, 'frozen', False):
            return 'Use a packaged Windows build (installed or portable) or Linux AppImage for in-app updates.'
        if platform.machine().lower() not in ('amd64', 'x86_64'):
            return 'Updates currently ship for x64 Windows and Linux.'
        if sys.platform == 'linux' and not os.environ.get('APPIMAGE'):
            return 'Use the AppImage build for in-app updates.'
        if sys.platform not in ('win32', 'linux'):
            return 'Updates are not available on this platform yet.'
        return None

    def _run(self, state, work):
        if self.thread and self.thread.is_alive():
            return
        self.state = state
        self.cancel.clear()
        self.changed.emit(state, '', 0)
        def task():
            try:
                work()
            except Exception as error:
                self.changed.emit('error', str(error), 0)
        self.thread = threading.Thread(target=task, daemon=True, name='nodebased-updater')
        self.thread.start()

    def check(self):
        reason = self.supported()
        if reason:
            self.changed.emit('unsupported', reason, 0)
            return
        def work():
            self.release = fetch_release()
            self.download = None
            self.changed.emit('available' if self.release else 'current', self.release.version if self.release else __version__, 0)
        self._run('checking', work)

    def fetch(self):
        if not self.release:
            return
        def work():
            directory = cache_directory()
            self.download = download_release(self.release, directory,
                lambda value: self.changed.emit('downloading', self.release.version, value), self.cancel)
            if is_portable():
                from .portable import extract_bundle
                self.staged_bundle = extract_bundle(self.download, directory)
            self.changed.emit('ready', self.release.version, 100)
        self._run('downloading', work)

    def install(self):
        if not self.release or not self.download:
            raise ValueError('Download an update first')
        verify_download(self.download, self.release)
        if is_portable():
            if not self.staged_bundle:
                raise ValueError('Portable update is not staged')
            env = os.environ.copy()
            env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
            subprocess.Popen([str(self.staged_bundle / 'NodeBased.exe'), '--apply-portable-update',
                str(Path(sys.executable).parent), str(os.getpid())], env=env, close_fds=True)
        elif sys.platform == 'win32':
            # Installer waits for this process before replacing files, then restarts.
            command = subprocess.list2cmdline([str(self.download), '/S', '/RESTART', f'/WAITPID={os.getpid()}'])
            subprocess.Popen(command + ' /D=' + str(Path(sys.executable).parent), close_fds=True)
        elif sys.platform == 'linux':
            replace_appimage(self.download, os.environ['APPIMAGE'])
        else:
            raise ValueError('Unsupported update platform')
