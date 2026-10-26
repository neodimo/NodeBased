"""Read's localization (fidelity step F): a local copy of a file that lives somewhere slow, kept in a cache folder.

Nuke localizes media so it plays from a local disk. `localizationPolicy` says when: `on` always, `off` never, and
`from auto-localize path` for files under a folder listed in the environment variable `NODEBASED_AUTOLOCALIZE_FROM` (folders
separated like PATH). The copies live in `NODEBASED_LOCALIZE_DIR` (default `~/.cache/nodebased/localize`), one folder per source
file, with a stamp of the source's size and modified time. A copy whose stamp no longer matches its source is replaced the next time
the file is read, so an edited file is never served from an old copy; `update` replaces copies now, which is what the Read panel's
Update localization button does. A source that has gone missing leaves its copy readable, which is what localizing is for.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

POLICIES = ("on", "from auto-localize path", "off")
DEFAULT_POLICY = "from auto-localize path"
CACHE_ENV = "NODEBASED_LOCALIZE_DIR"
FROM_ENV = "NODEBASED_AUTOLOCALIZE_FROM"


def cache_dir():
    return Path(os.environ.get(CACHE_ENV) or (Path.home() / ".cache" / "nodebased" / "localize"))


def auto_folders():
    return [Path(part).expanduser().resolve() for part in os.environ.get(FROM_ENV, "").split(os.pathsep) if part.strip()]


def wanted(policy, path):
    """True when a file at `path` is to be localized under `policy`."""
    if policy == "on":
        return True
    if policy != DEFAULT_POLICY:
        return False
    try:
        target = Path(path).expanduser().resolve()
    except OSError:
        return False
    return any(folder == target or folder in target.parents for folder in auto_folders())


def applies(params):
    """True when a Read with these parameters may read a local copy."""
    policy = str(params.get("localizationPolicy", DEFAULT_POLICY))
    if policy == "on":
        return True
    return policy == DEFAULT_POLICY and bool(auto_folders())


def copy_path(path):
    source = Path(path).expanduser()
    try:
        source = source.resolve()
    except OSError:
        pass
    return cache_dir() / hashlib.sha1(str(source).encode()).hexdigest()[:16] / source.name


def _stamp(path):
    stat = Path(path).stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _copy(source, force=False):
    target = copy_path(source)
    stamp_file = target.with_name(target.name + ".stamp")
    try:
        stamp = _stamp(source)
    except OSError:
        return target if target.exists() else None            # the source is gone: its copy is all there is
    current = None
    if target.exists() and stamp_file.exists():
        try:
            current = json.loads(stamp_file.read_text())
        except (OSError, ValueError):
            current = None
    if force or current != stamp or not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        shutil.copyfile(source, partial)
        os.replace(partial, target)
        stamp_file.write_text(json.dumps(stamp))
    return target


def localized(path, policy):
    """The path to read for `path` under `policy`: its local copy (made or refreshed first) or `path` itself."""
    if path is None or not wanted(policy, path):
        return path
    copy = _copy(path)
    return str(copy) if copy is not None else path


def update(paths, policy="on"):
    """Replace the copies of `paths` now. Returns the copies made (a path whose source is missing is skipped)."""
    made = []
    for path in paths:
        if wanted(policy, path):
            copy = _copy(path, force=True)
            if copy is not None:
                made.append(str(copy))
    return made


def files_of(path):
    """Every existing file the Read path names: the frames of a sequence pattern that are on disk, or the one file."""
    from .media import is_sequence, scan_sequence, sequence_path
    if not path:
        return []
    if is_sequence(path):
        return [sequence_path(path, number) for number in scan_sequence(path)]
    return [str(path)]


def update_read(params):
    """What the Read panel's Update localization does: replace the local copies of every file the Read points at, under the
    Read's own policy. Returns the copies made."""
    return update(files_of(params.get("path", "")), str(params.get("localizationPolicy", DEFAULT_POLICY)))
