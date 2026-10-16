"""Build the offline bundle for the field test (docs/FIELD_TEST.md, "Offline"): a folder to copy to a USB stick.

    python tools/make_field_bundle.py --out DIR [--everything] [--weights] [--python-version 3.12] [--platform win_amd64]
                                      [--skip-provider] [--dry-run] [--test-package NAME] [--reuse]
    python tools/make_field_bundle.py --verify DIR          (check a copied bundle against its manifest)

DIR gets:
  nodebased-<hash>.zip   the repository at HEAD (git archive; committed files only)
  wheels/                pip download of the project's dependencies plus the gpu and usd extras, and the build tools
  wheels-provider/       torch and torchvision from the CUDA index, then diffusers, transformers, accelerate, safetensors,
                         huggingface_hub, OpenEXR, numpy and pillow: what projects/nb-providers/sdxl/setup.sh installs
  weights/hub/           with --weights: the three SDXL model snapshots in the Hugging Face cache layout (about 9 GB)
  BUNDLE-README.txt      what is inside, the install order, the driver minimum, the commands to run on the target machine
  verify-bundle.ps1      PowerShell check of every file against manifest.json (no Python needed)
  MANIFEST.txt, manifest.json   size and sha256 of every file, plus "components" (versions and URLs) and "missing" (see below)

With --everything (implies --weights) the stick also holds, for a machine with no Python, no internet and no admin rights:
  python/                python-build-standalone CPython 3.12 for windows-x86_64 (install_only tar.gz): unpack into the user profile, no installer
  installers/            python.org installer (per-user fallback only: /passive InstallAllUsers=0), vc_redist.x64.exe (OPTIONAL, needs admin)
  app/                   the NodeBased Windows portable zip from the GitHub release of this version (no setup exe: it would want admin)
--reuse keeps files already in DIR (wheels, weights, downloads) whose sha256 matches the previous manifest.json and rewrites only README and manifest.
A requirement with no win_amd64 cp312 wheel is listed under "missing" in manifest.json and in BUNDLE-README.txt; it is never dropped silently.

Wheels are fetched for the target (default Windows 64-bit, CPython 3.12) with --only-binary=:all:, so nothing is built and
nothing needs the target's own Python to exist while this runs. Run it on any machine with internet and `pip`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"   # the index setup.sh uses; cu126 wheels carry their own CUDA runtime
BUILD_TOOLS = ["setuptools>=68", "wheel", "pip"]          # `pip install .` builds the project; offline it needs these in find-links
PROVIDER_PACKAGES = ["diffusers>=0.32", "transformers", "accelerate", "safetensors", "huggingface_hub", "OpenEXR", "numpy", "pillow"]
PYTHON_FTP = "https://www.python.org/ftp/python/"
STANDALONE_REPO = "astral-sh/python-build-standalone"
VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
RELEASE_REPO = "neodimo/NodeBased"
DRIVER_SOURCE = "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html (table 'Corresponding Driver Versions', CUDA 12.6 rows)"
SDXL_REPOS = ("stabilityai/stable-diffusion-xl-base-1.0", "madebyollin/sdxl-vae-fp16-fix", "xinsir/controlnet-union-sdxl-1.0")


def project_requirements():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = data.get("optional-dependencies", {})
    return list(data["dependencies"]) + list(extras.get("gpu", [])) + list(extras.get("usd", []))


def pip_download_command(destination, requirements, platform, python_version, extra=()):
    abi = "cp" + python_version.replace(".", "")
    return [sys.executable, "-m", "pip", "download", "--dest", str(destination), "--platform", platform,
            "--python-version", python_version, "--implementation", "cp", "--abi", abi, "--only-binary=:all:", *extra, *requirements]


def run(command, dry_run, cwd=None):
    print("$ " + " ".join(f'"{c}"' if " " in c or "<" in c or ">" in c else c for c in command), flush=True)
    if dry_run:
        return 0
    return subprocess.call(command, cwd=cwd)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(size):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024


def hf_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def copy_weights(destination, dry_run):
    """The three snapshots the provider reads, files dereferenced (the cache holds symlinks to blobs), in cache layout."""
    cache = hf_cache()
    copied = 0
    for repo in SDXL_REPOS:
        folder = cache / ("models--" + repo.replace("/", "--"))
        try:
            revision = (folder / "refs" / "main").read_text().strip()
        except OSError:
            if dry_run:
                # a dry run must work on a machine without the weights (the CI runners have none)
                print(f"weights: {repo}: not in the Hugging Face cache at {cache}; a real run copies it from there", flush=True)
                continue
            raise SystemExit(f"weights: {repo} is not in the Hugging Face cache at {cache}; run projects/nb-providers/sdxl/setup.sh first")
        source = folder / "snapshots" / revision
        target = destination / "hub" / folder.name
        print(f"weights: {repo} @ {revision[:8]} ({human(sum(p.stat().st_size for p in source.rglob('*') if p.is_file()))})", flush=True)
        if dry_run:
            continue
        (target / "refs").mkdir(parents=True, exist_ok=True)
        (target / "refs" / "main").write_text(revision)
        shutil.copytree(source, target / "snapshots" / revision, symlinks=False, dirs_exist_ok=True)
        copied += 1
    return copied


def http_get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nodebased-field-bundle"}), timeout=timeout) as response:
        return response.read()


def url_exists(url):
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "nodebased-field-bundle"})
        with urllib.request.urlopen(request, timeout=30):
            return True
    except OSError:
        return False


def latest_python_installer(python_version, listing=None, exists=url_exists):
    """The newest 3.12.x patch for which python.org ships a Windows amd64 installer (late patches are source-only security releases)."""
    listing = listing if listing is not None else http_get(PYTHON_FTP).decode("utf-8", "replace")
    patches = sorted({int(m) for m in re.findall(rf'href="{re.escape(python_version)}\.(\d+)/"', listing)}, reverse=True)
    for patch in patches:
        version = f"{python_version}.{patch}"
        url = f"{PYTHON_FTP}{version}/python-{version}-amd64.exe"
        if exists(url):
            return version, url
    raise SystemExit(f"python.org lists no Windows amd64 installer for Python {python_version}.x")


def download_file(url, target, fetch=None):
    """Stream url into target (a temporary name first, so a stopped run never leaves a half file under the real one)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if fetch:
        partial.write_bytes(fetch(url))
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "nodebased-field-bundle"})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
    partial.replace(target)
    return target


def previous_manifest(out):
    try:
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def folder_is_intact(out, manifest, folder):
    """--reuse: True when manifest.json lists files under folder/ and every one is present with its recorded size and sha256."""
    entries = [e for e in manifest.get("files", []) if e["path"].startswith(folder + "/")]
    if not entries:
        return False
    for entry in entries:
        path = out / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            return False
    print(f"reuse: {folder}/ ({len(entries)} files) already matches the previous manifest", flush=True)
    return True


def component(name, path, out, **extra):
    return dict(name=name, path=path.relative_to(out).as_posix(), bytes=path.stat().st_size, sha256=sha256(path), **extra)


def fetch_installers(out, python_version, dry_run, fetch=None, listing=None, exists=url_exists):
    """The python.org installer (per-user fallback only) and the VC++ redistributable (OPTIONAL: installing it needs admin)."""
    version, url = latest_python_installer(python_version, listing, exists) if not dry_run or fetch else (python_version + ".x", PYTHON_FTP)
    print(f"installers: Python {version} <- {url}\ninstallers: vc_redist.x64.exe <- {VC_REDIST_URL}", flush=True)
    if dry_run:
        return []
    folder = out / "installers"
    python_path = download_file(url, folder / url.rsplit("/", 1)[1], fetch)
    redist_path = download_file(VC_REDIST_URL, folder / "vc_redist.x64.exe", fetch)
    return [component("python", python_path, out, version=version, url=url),
            component("vc_redist", redist_path, out, version="Visual C++ 2015-2022 x64 (latest at build time)", url=VC_REDIST_URL)]


def standalone_asset(names, python_version):
    """Pick cpython-<python_version>.*-x86_64-pc-windows-msvc-install_only.tar.gz (not _stripped, not a debug build) from a release's asset names."""
    pattern = re.compile(rf"^cpython-({re.escape(python_version)}\.\d+)\+\d+-x86_64-pc-windows-msvc-install_only\.tar\.gz$")
    found = [(int(m.group(1).rsplit(".", 1)[1]), n, m.group(1)) for n in names for m in [pattern.match(n)] if m]
    if not found:
        raise SystemExit(f"python-build-standalone has no windows-x86_64 install_only archive for Python {python_version}.x in its latest release")
    _, name, version = max(found)
    return name, version


def fetch_standalone(out, python_version, dry_run, gh_json=None, fetch=None, reuse=None):
    """python-build-standalone's CPython for Windows x86_64 from the latest release: a tar.gz to unpack anywhere, no installer, no admin.
    Checked against the release's SHA256SUMS."""
    print(f"python: latest {STANDALONE_REPO} release, cpython-{python_version}.* x86_64-pc-windows-msvc install_only", flush=True)
    if dry_run:
        return []
    api = gh_json or (lambda args: json.loads(subprocess.check_output(["gh", *args], text=True)))
    release = api(["release", "view", "--repo", STANDALONE_REPO, "--json", "tagName,assets"])
    tag = release["tagName"]
    name, version = standalone_asset([a["name"] for a in release["assets"]], python_version)
    base = f"https://github.com/{STANDALONE_REPO}/releases/download/{tag}/"
    url = base + urllib.parse.quote(name)
    sums = (fetch or http_get)(base + "SHA256SUMS")
    sums = sums.decode("utf-8") if isinstance(sums, bytes) else sums
    published = next((line.split()[0].lower() for line in sums.splitlines() if len(line.split()) == 2 and line.split()[1].lstrip("*") == name), None)
    if published is None:
        raise SystemExit(f"{name} is not listed in the release's SHA256SUMS")
    target = out / "python" / name
    if not (reuse and reuse.get(target.relative_to(out).as_posix()) == published and target.is_file() and sha256(target) == published):
        download_file(url, target, fetch)
    entry = component("python-standalone", target, out, version=version, url=url, release=tag)
    entry["matches_release_sums"] = entry["sha256"] == published
    if not entry["matches_release_sums"]:
        raise SystemExit(f"{name}: sha256 {entry['sha256']} differs from the release's SHA256SUMS ({published})")
    return [entry]


def fetch_release(out, version, dry_run, gh=None):
    """The Windows portable zip of the matching GitHub release (no setup exe: it wants admin), checked against the release's SHA256SUMS when it has one."""
    tag = f"v{version}"
    command = ["gh", "release", "download", tag, "--repo", RELEASE_REPO, "--pattern", "*windows*portable*", "--pattern", "SHA256SUMS",
               "--dir", str(out / "app"), "--clobber"]
    print("release: " + " ".join(command), flush=True)
    if dry_run:
        return [], []
    for stale in (out / "app").glob("*setup*"):   # an earlier bundle held the setup exe; it installs for all users and wants admin
        stale.unlink()
    code = (gh or subprocess.call)(command)
    if code:
        return [], [f"GitHub release {tag}: gh release download failed (exit {code}); the app is not in the bundle, install from the zip with pip instead"]
    sums_file = out / "app" / "SHA256SUMS"
    published = {}
    if sums_file.is_file():
        for line in sums_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                published[parts[1].lstrip("*")] = parts[0].lower()
    parts, problems = [], []
    for path in sorted((out / "app").glob("*windows*portable*")):
        entry = component("app", path, out, version=version, url=f"https://github.com/{RELEASE_REPO}/releases/tag/{tag}")
        if path.name in published and published[path.name] != entry["sha256"]:
            problems.append(f"{path.name}: sha256 differs from the release's SHA256SUMS")
        entry["matches_release_sums"] = (published[path.name] == entry["sha256"]) if path.name in published else None
        parts.append(entry)
    if not parts:
        problems.append(f"GitHub release {tag} has no *windows*portable* asset")
    return parts, problems


def download_wheels(destination, requirements, platform, python_version, dry_run, extra=(), call=None):
    """pip download in one go; if that fails (one requirement without a Windows wheel fails the lot), retry each requirement alone and
    return those that still fail, so the manifest and README can name them."""
    call = call or run
    if call(pip_download_command(destination, requirements, platform, python_version, extra), dry_run) == 0:
        return []
    missing = []
    for requirement in requirements:
        if call(pip_download_command(destination, [requirement], platform, python_version, extra), dry_run):
            missing.append(requirement)
    return missing



README = """No admin rights needed: everything installs under your user folder.
NodeBased field test, offline bundle (built {when}, NodeBased {version}, commit {commit})

THIS STICK IS ENOUGH FOR A WINDOWS 64-BIT MACHINE WITH NO INTERNET, NO PYTHON AND NO ADMIN ACCOUNT. Target: {platform}, CPython {python_version}.
Full instructions: docs\\FIELD_TEST.md in the zip, section "Path B: OFFLINE". Call this folder E:\\nb-bundle below. Everything goes under
%USERPROFILE%\\nb (PowerShell: $env:USERPROFILE\\nb). Nothing edits the system PATH, the registry or a service.

WHAT IS INSIDE
{inventory}

STEP 0, CHECK THE GPU DRIVER (PowerShell)
  nvidia-smi
  The driver is not yours to update, so this only tells you which case you are in. Minimum for the CUDA 12.6 torch wheels: 561.17
  (560.76 for the GA release). With an older driver, or no nvidia-smi at all, the SDXL step cannot use the card: torch sees no CUDA
  device and the field report records the SDXL step as skipped. Everything else (the GPU tests, benchmarks, the app) still runs.

STEP 1, CHECK THE STICK (no Python needed)
  powershell -ExecutionPolicy Bypass -File E:\\nb-bundle\\verify-bundle.ps1

INSTALL ORDER (all per user, no elevation)
  1. Python, unpacked from python\\cpython-*-x86_64-pc-windows-msvc-install_only.tar.gz (python-build-standalone; no installer, no registry):
       New-Item -ItemType Directory -Force "$env:USERPROFILE\\nb" | Out-Null
       tar -xzf (Get-Item E:\\nb-bundle\\python\\cpython-*.tar.gz).FullName -C "$env:USERPROFILE\\nb"
       & "$env:USERPROFILE\\nb\\python\\python.exe" --version          # 3.12.x; the folder is "python" under nb
  2. The app (optional, no install): unzip app\\NodeBased-*-windows-x64-portable.zip into a folder you own and start NodeBased.exe.
     There is no setup exe on the stick (it installs for all users). The tests, benchmarks and field report need steps 3 to 5.
  3. The code and a virtual environment:

  Expand-Archive E:\\nb-bundle\\{zip} "$env:USERPROFILE\\nb"
  cd "$env:USERPROFILE\\nb\\nodebased"
  & "$env:USERPROFILE\\nb\\python\\python.exe" -m venv .venv
  .\\.venv\\Scripts\\Activate.ps1          # if PowerShell refuses: Set-ExecutionPolicy -Scope Process Bypass (this window only, no admin)
  python -m pip install --no-index --find-links E:\\nb-bundle\\wheels --upgrade pip setuptools wheel
  pip install --no-index --find-links E:\\nb-bundle\\wheels ".[gpu,usd]"
  pip install --no-index --find-links E:\\nb-bundle\\wheels-provider torch torchvision diffusers transformers accelerate safetensors huggingface_hub OpenEXR numpy pillow

  4. The SDXL weights. Read them from the stick (nothing to copy) ...
  $env:HF_HUB_CACHE = "E:\\nb-bundle\\weights\\hub"; $env:HF_HUB_OFFLINE = "1"
  ... or copy them under your profile (9 GB), which keeps working when the stick is gone:
  $env:HF_HUB_CACHE = "$env:USERPROFILE\\nb\\hf\\hub"; $env:HF_HUB_OFFLINE = "1"
  New-Item -ItemType Directory -Force $env:HF_HUB_CACHE | Out-Null; Copy-Item -Recurse -Force E:\\nb-bundle\\weights\\hub\\* $env:HF_HUB_CACHE
  5. The report (the demo plates are generated by the report in about a minute; nothing to copy). Reports go under your profile:
  python tools\\field_report.py --quick --out "$env:USERPROFILE\\nb\\field-quick" --label "A6000 work machine (quick)"
  python tools\\field_report.py --out "$env:USERPROFILE\\nb\\field-report" --label "A6000 work machine"

VISUAL C++ RUNTIME (vc_redist.x64.exe): OPTIONAL
  installers\\vc_redist.x64.exe needs admin rights; skip it. Whether the packages carry their own runtime, read from the wheels in this bundle:
{dll_report}
  If `python -c "import torch"` fails with a DLL load error naming MSVCP140.dll, copy the runtime DLLs shiboken6 ships next to python.exe
  (your folder, no admin; this workaround has not been run on a Windows machine without the redistributable):
  Copy-Item "$env:USERPROFILE\\nb\\nodebased\\.venv\\Lib\\site-packages\\shiboken6\\msvcp140*.dll" "$env:USERPROFILE\\nb\\python\\"

PYTHON INSTALLER (fallback only)
  installers\\python-*-amd64.exe is for a machine where unpacking the tar.gz does not work. Per user, no admin:
  E:\\nb-bundle\\installers\\python-3.12.10-amd64.exe /passive InstallAllUsers=0 PrependPath=1 Include_test=0
  Then use `py -3.12` or the python.exe it installs under %LOCALAPPDATA%\\Programs\\Python in place of the python.exe path above.

NVIDIA DRIVER
  The torch wheels are CUDA 12.6 builds and carry their own CUDA runtime; only the display driver must be new enough.
  NVIDIA's table (CUDA 12.6 rows, Windows x86_64): GA needs 560.76 or newer, Update 3 (the last 12.6) needs 561.17 or newer.
  Run `nvidia-smi` to see yours (step 0). Read at: {driver_source}
  (CUDA 12.x minor-version compatibility lists 525 as a floor for the family, but the table above is the one for 12.6.)

SLAM PROVIDERS
  DPVO and MASt3R run in Linux virtual environments under projects/nb-providers/slam on the workstation. They are NOT in this bundle and
  do not run on Windows here. The camera solve from survey points needs no SLAM and works; the report's SLAM steps, if any, are recorded as skipped.

REQUIREMENTS WITH NO WINDOWS WHEEL
{missing}
"""

VERIFY_PS1 = """# Check every file in this folder against manifest.json (sha256 and size). Needs only Windows PowerShell 5.1 or newer.
#   powershell -ExecutionPolicy Bypass -File verify-bundle.ps1 [-Root E:\\nb-bundle]
param([string]$Root = $PSScriptRoot)
$manifest = Get-Content -Raw -Encoding UTF8 (Join-Path $Root "manifest.json") | ConvertFrom-Json
$bad = 0
foreach ($entry in $manifest.files) {
    $path = Join-Path $Root ($entry.path -replace "/", "\\")
    if (-not (Test-Path -LiteralPath $path)) { Write-Host "missing  $($entry.path)"; $bad++; continue }
    if ((Get-Item -LiteralPath $path).Length -ne $entry.bytes) { Write-Host "size     $($entry.path)"; $bad++; continue }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower() -ne $entry.sha256) { Write-Host "changed  $($entry.path)"; $bad++ }
}
if ($bad -eq 0) { Write-Host "checked $($manifest.files.Count) files: all match the manifest" } else { Write-Host "$bad problem(s)"; exit 1 }
"""


def inventory(entries, components):
    sizes = {}
    for e in entries:
        top = e["path"].split("/")[0] if "/" in e["path"] else "(top level)"
        sizes[top] = sizes.get(top, 0) + e["bytes"]
    notes = {"python": "standalone CPython 3.12 for Windows (unpack, no installer)", "installers": "python.org installer (per-user fallback), vc_redist (OPTIONAL, needs admin)",
             "app": "NodeBased Windows portable zip",
             "wheels": "pip wheels for the project .[gpu,usd], pip, setuptools, wheel", "wheels-provider": "torch, torchvision (cu126), diffusers, transformers ...",
             "weights": "SDXL base, VAE fix, ControlNet-Union (Hugging Face cache layout)"}
    lines = [f"  {name + '/':<17} {human(size):>10}  {notes.get(name, '')}" for name, size in sorted(sizes.items()) if name != "(top level)"]
    for c in components:
        lines.append(f"  {c['name']:<10} {c['path']}  version {c.get('version', '')}  sha256 {c['sha256'][:16]}...")
    return "\n".join(lines)


def build_manifest(out, components=(), missing=(), notes=()):
    entries = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.txt", "manifest.json") and not path.name.endswith(".part"):
            entries.append(dict(path=path.relative_to(out).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    total = sum(e["bytes"] for e in entries)
    (out / "manifest.json").write_text(json.dumps(dict(files=entries, total_bytes=total, components=list(components), missing=list(missing),
                                                       notes=list(notes)), indent=1), encoding="utf-8")
    lines = [f"{e['sha256']}  {e['bytes']:>12}  {e['path']}" for e in entries]
    (out / "MANIFEST.txt").write_text("\n".join(lines) + f"\n\n{len(entries)} files, {total} bytes ({human(total)})\n", encoding="utf-8")
    return entries, total


RUNTIME_DLL = re.compile(r"(?:^|/)((?:msvcp|vcruntime|concrt|vcomp)140\w*\.dll)$", re.I)
RUNTIME_WHEELS = ("pyside6", "pyside6_essentials", "pyside6_addons", "shiboken6", "torch", "torchvision")


def runtime_dll_report(out):
    """Which of the PySide6, shiboken6 and torch wheels in the bundle ship the Visual C++ runtime DLLs (read from the wheel file lists)."""
    found = {}
    for wheel in sorted(out.glob("wheels*/*.whl")):
        name = wheel.name.split("-")[0].lower()
        if name in RUNTIME_WHEELS:
            with zipfile.ZipFile(wheel) as archive:
                found[name] = sorted({m.group(1).lower() for n in archive.namelist() for m in [RUNTIME_DLL.search(n)] if m})
    yes = [n for n, dlls in found.items() if dlls]
    no = [n for n, dlls in found.items() if not dlls]
    verdict = "the app's packages carry their own runtime: " + ("yes (" + ", ".join(yes) + ")" if yes else "no") + ("; no (" + ", ".join(no) + ")" if yes and no else "")
    return "  " + verdict + "\n" + "\n".join(f"    {n:<20} " + (", ".join(d) if d else "none") for n, d in found.items())


def write_readme(out, version, commit, zip_name, platform, python_version, components, missing, notes):
    """BUNDLE-README.txt and verify-bundle.ps1; written before the manifest so both are listed in it."""
    entries = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size) for p in sorted(out.rglob("*")) if p.is_file()]
    problems = list(missing) + list(notes)
    (out / "verify-bundle.ps1").write_text(VERIFY_PS1, encoding="utf-8")
    (out / "BUNDLE-README.txt").write_text(README.format(
        when=time.strftime("%Y-%m-%d"), version=version, commit=commit, zip=zip_name, platform=platform, python_version=python_version,
        inventory=inventory(entries, components), driver_source=DRIVER_SOURCE, dll_report=runtime_dll_report(out),
        missing="\n".join("  " + m for m in problems) if problems else "  none: every requirement has a win_amd64 cp312 wheel in the bundle."), encoding="utf-8")


def verify(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    for entry in manifest["files"]:
        path = out / entry["path"]
        if not path.is_file():
            bad.append(f"missing  {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            bad.append(f"changed  {entry['path']}")
    print(f"checked {len(manifest['files'])} files: " + ("all match the manifest" if not bad else f"{len(bad)} problem(s)"))
    for line in bad:
        print("  " + line)
    return 1 if bad else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="bundle folder (created)")
    parser.add_argument("--weights", action="store_true", help="also copy the SDXL model snapshots from the Hugging Face cache (about 9 GB)")
    parser.add_argument("--everything", action="store_true", help="implies --weights; also standalone Python, the python.org installer, vc_redist and the release's Windows portable zip")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--platform", default="win_amd64")
    parser.add_argument("--skip-provider", action="store_true", help="leave out wheels-provider (torch is about 2.5 GB)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and what would be copied, change nothing")
    parser.add_argument("--test-package", help="download only this package instead of the real lists (a smoke test of the pip flags)")
    parser.add_argument("--reuse", action="store_true", help="keep wheels/, wheels-provider/ and weights/ in an existing --out folder whose files still match its manifest.json")
    parser.add_argument("--verify", metavar="DIR", help="check DIR against its manifest and exit")
    args = parser.parse_args(argv)
    if args.verify:
        return verify(args.verify)
    if not args.out:
        parser.error("--out is required")
    out = Path(args.out).resolve()
    if args.everything:
        args.weights = True
    started = time.monotonic()
    if not args.dry_run and subprocess.call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
        raise SystemExit(f"{sys.executable} has no pip; run this script with a Python that does (python -m ensurepip, or another install)")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    version = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "nodebased" / "__init__.py").read_text(encoding="utf-8")).group(1)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    zip_name = f"nodebased-{commit}.zip"
    failures = []
    before = previous_manifest(out) if args.reuse and not args.dry_run else {}
    reuse_hashes = {e["path"]: e["sha256"] for e in before.get("files", [])}
    carried = [m for m in before.get("missing", [])]
    kept = lambda folder: bool(before) and folder_is_intact(out, before, folder)

    # 1. the repository at HEAD (a zip of an older commit left in DIR is removed)
    if not args.dry_run:
        for old in out.glob("nodebased-*.zip"):
            old.unlink()
    command = ["git", "archive", "--format=zip", "--prefix=nodebased/", "-o", str(out / zip_name), "HEAD"]
    if run(command, args.dry_run, cwd=ROOT):
        failures.append("git archive")

    # 2. wheels for NodeBased itself, and the tools pip needs to build it offline
    requirements = [args.test_package] if args.test_package else project_requirements() + BUILD_TOOLS
    print(f"project requirements: {requirements}", flush=True)
    if kept("wheels"):
        missing = [m for m in carried if m.startswith("wheels/")]
    else:
        missing = list(download_wheels(out / "wheels", requirements, args.platform, args.python_version, args.dry_run))
        missing = [f"wheels/: {m} has no {args.platform} cp{args.python_version.replace('.', '')} wheel (pip download failed)" for m in missing]

    # 3. wheels for the SDXL provider environment: torch from the CUDA index first (pinned), then everything else
    if not args.skip_provider and kept("wheels-provider"):
        missing += [m for m in carried if m.startswith("wheels-provider/")]
    elif not args.skip_provider:
        provider_dir = out / "wheels-provider"
        torch_args = ["torch", "torchvision"] if not args.test_package else [args.test_package]
        if run(pip_download_command(provider_dir, torch_args, args.platform, args.python_version, ["--no-deps", "--index-url", TORCH_INDEX]), args.dry_run):
            failures.append("pip download (torch)")
        pins = []
        if not args.dry_run and not args.test_package:
            for name in ("torch", "torchvision"):
                found = sorted(provider_dir.glob(f"{name}-*.whl"))
                if found:
                    pins.append(f"{name}=={found[-1].name.split('-')[1]}")   # e.g. torch==2.9.0+cu126, so PyPI's CPU build is never picked
        rest = PROVIDER_PACKAGES + (pins or ["torch", "torchvision"])
        if args.test_package:
            rest = [args.test_package]
        absent = download_wheels(provider_dir, rest, args.platform, args.python_version, args.dry_run, ["--find-links", str(provider_dir)])
        missing += [f"wheels-provider/: {m} has no {args.platform} cp{args.python_version.replace('.', '')} wheel (pip download failed)" for m in absent]

    # 4. weights
    if args.weights and not kept("weights"):
        copy_weights(out / "weights", args.dry_run)

    # 5. everything else a machine with no Python and no internet needs
    components, notes = [], []
    if args.everything:
        try:
            components += fetch_standalone(out, args.python_version, args.dry_run, reuse=reuse_hashes)
        except (OSError, SystemExit, subprocess.CalledProcessError) as error:
            failures.append(f"python-build-standalone ({error})")
        try:
            components += fetch_installers(out, args.python_version, args.dry_run)
        except (OSError, SystemExit) as error:
            failures.append(f"installers ({error})")
        parts, problems = fetch_release(out, version, args.dry_run)
        components += parts
        notes += problems
        failures += problems
    if args.dry_run:
        print("dry run: nothing was written")
        return 0
    notes.append(f"torch and torchvision come from {TORCH_INDEX} (CUDA 12.6 builds, win_amd64 cp{args.python_version.replace('.', '')}).")
    write_readme(out, version, commit, zip_name, args.platform, args.python_version, components, missing, [n for n in notes if "differs" in n or "no *windows*" in n or "failed" in n])
    entries, total = build_manifest(out, components, missing, notes)
    print("\n" + (out / "MANIFEST.txt").read_text(encoding="utf-8"))
    by_folder = {}
    for e in entries:
        top = e["path"].split("/")[0] if "/" in e["path"] else "(top level)"
        by_folder[top] = by_folder.get(top, 0) + e["bytes"]
    print("bundle size by folder: " + ", ".join(f"{k} {human(v)}" for k, v in sorted(by_folder.items())))
    print(f"bundle total: {human(total)} in {len(entries)} files at {out}; took {time.monotonic() - started:.0f} s")
    if missing:
        print("NO WINDOWS WHEEL FOR: " + "; ".join(missing))
        failures.append("requirements without a Windows wheel")
    if failures:
        print("INCOMPLETE: " + ", ".join(failures) + " failed; the bundle is missing those parts. Re-run with a working network.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
