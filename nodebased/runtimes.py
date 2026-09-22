"""On-demand uv-built environments for heavy ML tools such as SHARP.

Torch and its friends are deliberately kept inside these named virtual
environments rather than becoming NodeBased package dependencies.  The tools
are driven through subprocesses, and ``NODEBASED_RUNTIME_<NAME>`` may point at
an existing runtime during development or tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

import certifi

from .cancellation import Cancelled


@dataclass(frozen=True)
class Package:
    spec: str
    index_url: str | None = None
    extra_env: dict = field(default_factory=dict)
    no_build_isolation: bool = False


@dataclass(frozen=True)
class SourcePin:
    url: str
    sha256: str
    commit: str
    repo: str
    install_no_deps: bool = True


@dataclass(frozen=True)
class CheckpointPin:
    url: str
    filename: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RuntimeRecipe:
    name: str
    python: str
    packages: tuple[Package, ...]
    source: SourcePin
    checkpoint: CheckpointPin
    entrypoint: str


RECIPES = {
    "sharp": RuntimeRecipe(
        name="sharp", python="3.12",
        packages=(
            Package("torch==2.14.0+cu130", index_url="https://download.pytorch.org/whl/cu130"),
            Package("torchvision==0.29.0+cu130", index_url="https://download.pytorch.org/whl/cu130"),
            Package("click==8.5.0"), Package("imageio==2.37.4"),
            Package("imageio-ffmpeg==0.6.0"), Package("matplotlib==3.11.2"),
            Package("pillow-heif==1.7.0"), Package("plyfile==1.1.5"),
            Package("scipy==1.18.1"), Package("timm==1.0.29"),
            Package("gsplat==1.5.3", extra_env={"BUILD_NO_CUDA": "1"}, no_build_isolation=True),
        ),
        source=SourcePin(
            url="https://github.com/apple/ml-sharp/archive/aed6527499ef91cba3b54c18d49a870f25947190.tar.gz",
            sha256="d6bdb0fdc9a6de3d37a6f972253bb5be5e2cba034dca3cc1e1208edb106a2cbb",
            commit="aed6527499ef91cba3b54c18d49a870f25947190", repo="https://github.com/apple/ml-sharp"),
        checkpoint=CheckpointPin(
            url="https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt",
            filename="sharp_2572gikvuh.pt", size=2809738232,
            sha256="94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"),
        entrypoint="sharp")
}


@dataclass(frozen=True)
class RuntimeStatus:
    state: str
    detail: str = ""


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    log_path: Path


_installing: set[str] = set()
_installing_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _user_data_directory() -> Path:
    from . import updater
    import sys as _sys
    if updater.is_portable():
        return Path(_sys.executable).parent / 'portable-data' / 'runtimes'
    from PySide6.QtCore import QStandardPaths
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)) / 'runtimes'


def _root(name: str) -> tuple[Path, bool]:
    override = os.environ.get(f'NODEBASED_RUNTIME_{name.upper()}')
    if override:
        return Path(override), True
    return _user_data_directory() / name, False


def _python_path(root: Path) -> Path:
    return root / 'venv' / ('Scripts' if sys.platform == 'win32' else 'bin') / ('python.exe' if sys.platform == 'win32' else 'python')


def _recipe_digest(recipe: RuntimeRecipe) -> str:
    data = {
        'name': recipe.name, 'python': recipe.python, 'entrypoint': recipe.entrypoint,
        'packages': [{'spec': p.spec, 'index_url': p.index_url, 'extra_env': p.extra_env,
                      'no_build_isolation': p.no_build_isolation} for p in recipe.packages],
        'source': {'url': recipe.source.url, 'sha256': recipe.source.sha256,
                   'commit': recipe.source.commit, 'repo': recipe.source.repo,
                   'install_no_deps': recipe.source.install_no_deps},
        'checkpoint': {'url': recipe.checkpoint.url, 'filename': recipe.checkpoint.filename,
                       'size': recipe.checkpoint.size, 'sha256': recipe.checkpoint.sha256},
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _open_url(request, timeout=30):
    return urlopen(request, timeout=timeout, context=ssl.create_default_context(cafile=certifi.where()))


def _cancelled(cancel):
    return cancel is not None and cancel.is_set()


def _execute(argv, *, cwd=None, env=None, timeout=None, cancel=None, popen=None, log_path=None):
    popen = popen or subprocess.Popen
    process = None
    stdout = stderr = ''
    try:
        process = popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        started = time.monotonic()
        while True:
            if _cancelled(cancel):
                process.kill(); process.wait()
                raise Cancelled()
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                if _cancelled(cancel):
                    process.kill(); process.wait()
                    raise Cancelled()
                break
            except subprocess.TimeoutExpired:
                if _cancelled(cancel):
                    process.kill(); process.wait()
                    raise Cancelled()
                if timeout is not None and time.monotonic() - started >= timeout:
                    process.kill(); process.wait()
                    raise ValueError(f'{argv[0]} timed out after {timeout} seconds')
        return process.returncode, stdout, stderr
    finally:
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', encoding='utf-8') as log:
                log.write('$ ' + ' '.join(str(a) for a in argv) + '\n' + stdout + stderr)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.marker-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _check_cancel(cancel):
    if _cancelled(cancel):
        raise Cancelled()


def status(name: str) -> RuntimeStatus:
    recipe = RECIPES.get(name)
    if recipe is None:
        raise ValueError(f'unknown runtime: {name}')
    root, override = _root(name)
    if override:
        return RuntimeStatus('ready' if _python_path(root).is_file() and
                             (root / 'checkpoints' / recipe.checkpoint.filename).is_file() else 'absent')
    with _installing_lock:
        if name in _installing:
            return RuntimeStatus('installing')
    marker = root / 'installing.json'
    if marker.exists():
        try:
            pid = json.loads(marker.read_text(encoding='utf-8'))['pid']
            try:
                os.kill(pid, 0)
            except OSError:
                raise RuntimeError
            return RuntimeStatus('installing')
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError, OSError):
            return RuntimeStatus('broken', 'installation did not finish (interrupted)')
    error = root / 'error.json'
    if error.exists():
        try:
            return RuntimeStatus('broken', json.loads(error.read_text(encoding='utf-8'))['reason'])
        except (ValueError, KeyError, json.JSONDecodeError, OSError):
            return RuntimeStatus('broken', 'installation error')
    ready = root / 'ready.json'
    if ready.exists():
        try:
            if json.loads(ready.read_text(encoding='utf-8'))['recipe_digest'] == _recipe_digest(recipe):
                return RuntimeStatus('ready')
        except (ValueError, KeyError, json.JSONDecodeError, OSError):
            pass
    return RuntimeStatus('absent')


def _download(url, target, expected_size=None, expected_hash=None, *, opener, cancel, progress=None, cap=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix='.download-', dir=target.parent)
    digest, total = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, 'wb') as output, opener(Request(url), timeout=30) as source:
            while True:
                _check_cancel(cancel)
                chunk = source.read(256 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if cap is not None and total > cap:
                    raise ValueError('download exceeds its declared or safety-limit size')
                output.write(chunk); digest.update(chunk)
                if progress:
                    progress(total)
            output.flush(); os.fsync(output.fileno())
        if (expected_size is not None and total != expected_size) or (expected_hash and digest.hexdigest() != expected_hash):
            raise ValueError('download integrity check failed; downloaded file is unchanged')
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def _install_error(root, error):
    _write_json(root / 'error.json', {'reason': str(error) or error.__class__.__name__, 'when': _now()})


def install(name: str, *, progress=None, cancel=None, opener=None, popen=None) -> None:
    recipe = RECIPES.get(name)
    if recipe is None:
        raise ValueError(f'unknown runtime: {name}')
    root, override = _root(name)
    if override:
        raise ValueError(f'the {name} runtime is pointed at an external directory via NODEBASED_RUNTIME_{name.upper()}; there is nothing to install')
    with _installing_lock:
        if name in _installing:
            raise ValueError(f'an install of the {name} runtime is already running')
    if shutil.which('uv') is None:
        raise ValueError(f'uv is required to build the {name} runtime; install it and try again.')
    opener = opener or _open_url
    popen = popen or subprocess.Popen
    with _installing_lock:
        _installing.add(name)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    try:
        _check_cancel(cancel)
        (root / 'error.json').unlink(missing_ok=True)
        _write_json(root / 'installing.json', {'pid': os.getpid(), 'started': _now()})
        # Probe first: this avoids starting a large torch download only to fail partway through.
        try:
            with opener(Request(recipe.source.url, method='HEAD'), timeout=8):
                pass
        except URLError as error:
            raise ValueError(f'no network connection; installing the {name} runtime needs to download packages, source and a checkpoint.') from error
        _check_cancel(cancel)
        venv = root / 'venv'; python = _python_path(root)
        if progress: progress('venv', 0.0, {})
        code, out, err = _execute(['uv', 'venv', '--python', recipe.python, str(venv)], cancel=cancel, popen=popen, log_path=root / 'logs' / 'venv.log')
        if code:
            raise ValueError(f'failed to create venv: {(out + err)[-2000:]}')
        if progress: progress('venv', 0.05, {})
        total_packages = len(recipe.packages)
        for index, package in enumerate(recipe.packages, 1):
            _check_cancel(cancel)
            argv = ['uv', 'pip', 'install', '--python', str(python)]
            if package.index_url: argv += ['--index-url', package.index_url]
            if package.no_build_isolation: argv.append('--no-build-isolation')
            argv.append(package.spec)
            log = root / 'logs' / 'packages.log'
            with log.open('a', encoding='utf-8') as handle:
                handle.write(f'\n# package {package.spec}\n')
            code, out, err = _execute(argv, env={**os.environ, **package.extra_env}, cancel=cancel, popen=popen, log_path=log)
            if code:
                raise ValueError(f'failed to install package {package.spec}: {(out + err)[-2000:]}')
            if progress: progress('packages', 0.05 + 0.5 * index / total_packages, {'package': package.spec})
        _check_cancel(cancel)
        extraction = Path(tempfile.mkdtemp(prefix='source-', dir=root))
        try:
            archive = root / '.source.tar.gz'
            _download(recipe.source.url, archive, expected_hash=recipe.source.sha256, opener=opener, cancel=cancel, cap=64 * 1024 * 1024)
            with tarfile.open(archive, 'r:*') as bundle:
                members = bundle.getmembers()
                for member in members:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                        raise ValueError('invalid path in runtime source archive')
                bundle.extractall(extraction, filter='data')
            tops = {PurePosixPath(m.name).parts[0] for m in members if PurePosixPath(m.name).parts}
            if len(tops) != 1:
                raise ValueError('runtime source archive has no single top-level directory')
            source_dir = extraction / next(iter(tops))
            if progress: progress('source', 0.6, {})
            _check_cancel(cancel)
            if recipe.source.install_no_deps:
                code, out, err = _execute(['uv', 'pip', 'install', '--python', str(python), '--no-deps', str(source_dir)], cancel=cancel, popen=popen, log_path=root / 'logs' / 'source.log')
                if code:
                    raise ValueError(f'failed to install source: {(out + err)[-2000:]}')
            if progress: progress('source', 0.65, {})
        finally:
            shutil.rmtree(extraction, ignore_errors=True)
            (root / '.source.tar.gz').unlink(missing_ok=True)
        _check_cancel(cancel)
        checkpoint = root / 'checkpoints' / recipe.checkpoint.filename
        def checkpoint_progress(downloaded):
            if progress:
                ratio = downloaded / recipe.checkpoint.size if recipe.checkpoint.size else 1
                progress('checkpoint', 0.65 + 0.35 * min(1.0, ratio), {'bytes': downloaded, 'total': recipe.checkpoint.size})
        _download(recipe.checkpoint.url, checkpoint, expected_size=recipe.checkpoint.size, expected_hash=recipe.checkpoint.sha256, opener=opener, cancel=cancel, progress=checkpoint_progress, cap=recipe.checkpoint.size)
        if progress: progress('checkpoint', 1.0, {'bytes': recipe.checkpoint.size, 'total': recipe.checkpoint.size})
        _check_cancel(cancel)
        _write_json(root / 'ready.json', {'recipe_digest': _recipe_digest(recipe), 'completed': _now()})
        (root / 'error.json').unlink(missing_ok=True)
    except Exception as error:
        _install_error(root, Cancelled('cancelled') if isinstance(error, Cancelled) else error)
        raise
    finally:
        (root / 'installing.json').unlink(missing_ok=True)
        with _installing_lock:
            _installing.discard(name)


def run(name: str, argv: list[str], *, cwd=None, timeout=None, cancel=None, popen=None) -> RunResult:
    recipe = RECIPES.get(name)
    if recipe is None:
        raise ValueError(f'unknown runtime: {name}')
    root, _ = _root(name)
    venv = root / 'venv'; python = _python_path(root)
    if not python.exists():
        raise ValueError(f'the {name} runtime is not installed; install it first or check status()')
    bindir = python.parent
    command = list(argv)
    if command and os.sep not in command[0] and (os.altsep is None or os.altsep not in command[0]):
        candidate = bindir / command[0]
        if sys.platform == 'win32' and not candidate.suffix.lower() == '.exe': candidate = candidate.with_suffix('.exe')
        if candidate.exists(): command[0] = str(candidate)
    root.joinpath('logs').mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    log = root / 'logs' / f'run-{stamp}.log'
    env = {**os.environ, 'VIRTUAL_ENV': str(venv), 'PATH': str(bindir) + os.pathsep + os.environ.get('PATH', '')}
    code, stdout, stderr = _execute(command, cwd=cwd, env=env, timeout=timeout, cancel=cancel, popen=popen, log_path=log)
    return RunResult(code, stdout, stderr, log)
