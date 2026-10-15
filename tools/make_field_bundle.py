"""Build the offline bundle for the field test (docs/FIELD_TEST.md, "Offline"): a folder to copy to a USB stick.

    python tools/make_field_bundle.py --out DIR [--everything] [--weights] [--python-version 3.12] [--platform win_amd64]
                                      [--skip-provider] [--dry-run] [--test-package NAME]
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

With --everything (implies --weights) the stick also holds, for a machine with no Python and no internet:
  installers/            the python.org Windows 3.12 installer (latest patch that ships one), vc_redist.x64.exe
  app/                   the NodeBased Windows portable zip and setup exe from the GitHub release of this version
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
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"   # the index setup.sh uses; cu126 wheels carry their own CUDA runtime
BUILD_TOOLS = ["setuptools>=68", "wheel", "pip"]          # `pip install .` builds the project; offline it needs these in find-links
PROVIDER_PACKAGES = ["diffusers>=0.32", "transformers", "accelerate", "safetensors", "huggingface_hub", "OpenEXR", "numpy", "pillow"]
PYTHON_FTP = "https://www.python.org/ftp/python/"
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


def component(name, path, out, **extra):
    return dict(name=name, path=path.relative_to(out).as_posix(), bytes=path.stat().st_size, sha256=sha256(path), **extra)


def fetch_installers(out, python_version, dry_run, fetch=None, listing=None, exists=url_exists):
    """The Python installer and the VC++ redistributable: the two things a bare Windows machine lacks before pip can run."""
    version, url = latest_python_installer(python_version, listing, exists) if not dry_run or fetch else (python_version + ".x", PYTHON_FTP)
    print(f"installers: Python {version} <- {url}\ninstallers: vc_redist.x64.exe <- {VC_REDIST_URL}", flush=True)
    if dry_run:
        return []
    folder = out / "installers"
    python_path = download_file(url, folder / url.rsplit("/", 1)[1], fetch)
    redist_path = download_file(VC_REDIST_URL, folder / "vc_redist.x64.exe", fetch)
    return [component("python", python_path, out, version=version, url=url),
            component("vc_redist", redist_path, out, version="Visual C++ 2015-2022 x64 (latest at build time)", url=VC_REDIST_URL)]


def fetch_release(out, version, dry_run, gh=None):
    """The Windows portable zip and setup exe of the matching GitHub release, checked against the release's SHA256SUMS when it has one."""
    tag = f"v{version}"
    command = ["gh", "release", "download", tag, "--repo", RELEASE_REPO, "--pattern", "*windows*", "--pattern", "SHA256SUMS",
               "--dir", str(out / "app"), "--clobber"]
    print("release: " + " ".join(command), flush=True)
    if dry_run:
        return [], []
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
    for path in sorted((out / "app").glob("*windows*")):
        entry = component("app", path, out, version=version, url=f"https://github.com/{RELEASE_REPO}/releases/tag/{tag}")
        if path.name in published and published[path.name] != entry["sha256"]:
            problems.append(f"{path.name}: sha256 differs from the release's SHA256SUMS")
        entry["matches_release_sums"] = (published[path.name] == entry["sha256"]) if path.name in published else None
        parts.append(entry)
    if not parts:
        problems.append(f"GitHub release {tag} has no *windows* assets")
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



README = """NodeBased field test, offline bundle (built {when}, NodeBased {version}, commit {commit})

THIS STICK IS ENOUGH FOR A WINDOWS 64-BIT MACHINE WITH NO INTERNET AND NO PYTHON. Target: {platform}, CPython {python_version}.
Full instructions: docs\\FIELD_TEST.md in the zip, section "Path B: OFFLINE". Call this folder E:\\nb-bundle below.

WHAT IS INSIDE
{inventory}

STEP 0, CHECK THE STICK (PowerShell, no Python needed)
  powershell -ExecutionPolicy Bypass -File E:\\nb-bundle\\verify-bundle.ps1

INSTALL ORDER
  1. installers\\python-*-amd64.exe   tick "Add python.exe to PATH" (and "py launcher"), then Install Now. Open a NEW PowerShell afterwards.
  2. installers\\vc_redist.x64.exe    PySide6, wgpu and torch need the Visual C++ runtime; a bare machine lacks it.
  3. Either run the app with no install: unzip app\\NodeBased-*-windows-x64-portable.zip and start NodeBased.exe,
     or run its setup exe. The tests, benchmarks and field report need step 4 as well (they run from the repository zip).
  4. The OFFLINE path (Python environment; the repository zip, wheels and weights):

  Expand-Archive E:\\nb-bundle\\{zip} C:\\nb
  cd C:\\nb\\nodebased
  py -3.12 -m venv .venv
  .\\.venv\\Scripts\\Activate.ps1
  python -m pip install --no-index --find-links E:\\nb-bundle\\wheels --upgrade pip setuptools wheel
  pip install --no-index --find-links E:\\nb-bundle\\wheels ".[gpu,usd]"
  pip install --no-index --find-links E:\\nb-bundle\\wheels-provider torch torchvision diffusers transformers accelerate safetensors huggingface_hub OpenEXR numpy pillow
  # the SDXL weights: read them from the stick (nothing to copy) ...
  $env:HF_HUB_CACHE = "E:\\nb-bundle\\weights\\hub"; $env:HF_HUB_OFFLINE = "1"
  # ... or copy them into the default cache, which needs no variables:
  #   New-Item -ItemType Directory -Force "$env:USERPROFILE\\.cache\\huggingface\\hub" | Out-Null
  #   Copy-Item -Recurse -Force E:\\nb-bundle\\weights\\hub\\* "$env:USERPROFILE\\.cache\\huggingface\\hub\\"
  # the report (the demo plates are generated by the report in about a minute; nothing to copy)
  python tools\\field_report.py --quick --out C:\\nb\\field-quick --label "A6000 work machine (quick)"
  python tools\\field_report.py --out C:\\nb\\field-report --label "A6000 work machine"

NVIDIA DRIVER
  The torch wheels are CUDA 12.6 builds and carry their own CUDA runtime; only the display driver must be new enough.
  NVIDIA's table (CUDA 12.6 rows, Windows x86_64): GA needs 560.76 or newer, Update 3 (the last 12.6) needs 561.17 or newer.
  Install 561.17 or newer (any current Studio or RTX Enterprise driver for the RTX A6000 is far above that). Run `nvidia-smi` to see yours.
  Read at: {driver_source}
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
    notes = {"installers": "the Python and Visual C++ installers", "app": "NodeBased Windows portable zip and setup exe",
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


def write_readme(out, version, commit, zip_name, platform, python_version, components, missing, notes):
    """BUNDLE-README.txt and verify-bundle.ps1; written before the manifest so both are listed in it."""
    entries = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size) for p in sorted(out.rglob("*")) if p.is_file()]
    problems = list(missing) + list(notes)
    (out / "verify-bundle.ps1").write_text(VERIFY_PS1, encoding="utf-8")
    (out / "BUNDLE-README.txt").write_text(README.format(
        when=time.strftime("%Y-%m-%d"), version=version, commit=commit, zip=zip_name, platform=platform, python_version=python_version,
        inventory=inventory(entries, components), driver_source=DRIVER_SOURCE,
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
    parser.add_argument("--everything", action="store_true", help="implies --weights; also the Python installer, vc_redist and the release's Windows app")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--platform", default="win_amd64")
    parser.add_argument("--skip-provider", action="store_true", help="leave out wheels-provider (torch is about 2.5 GB)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and what would be copied, change nothing")
    parser.add_argument("--test-package", help="download only this package instead of the real lists (a smoke test of the pip flags)")
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

    # 1. the repository at HEAD
    command = ["git", "archive", "--format=zip", "--prefix=nodebased/", "-o", str(out / zip_name), "HEAD"]
    if run(command, args.dry_run, cwd=ROOT):
        failures.append("git archive")

    # 2. wheels for NodeBased itself, and the tools pip needs to build it offline
    requirements = [args.test_package] if args.test_package else project_requirements() + BUILD_TOOLS
    print(f"project requirements: {requirements}", flush=True)
    missing = list(download_wheels(out / "wheels", requirements, args.platform, args.python_version, args.dry_run))
    missing = [f"wheels/: {m} has no {args.platform} cp{args.python_version.replace('.', '')} wheel (pip download failed)" for m in missing]

    # 3. wheels for the SDXL provider environment: torch from the CUDA index first (pinned), then everything else
    if not args.skip_provider:
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
    if args.weights:
        copy_weights(out / "weights", args.dry_run)

    # 5. everything else a machine with no Python and no internet needs
    components, notes = [], []
    if args.everything:
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
