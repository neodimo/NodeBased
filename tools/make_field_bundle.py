"""Build the offline bundle for the field test (docs/FIELD_TEST.md, "Offline"): a folder to copy to a USB stick.

    python tools/make_field_bundle.py --out DIR [--weights] [--python-version 3.12] [--platform win_amd64]
                                      [--skip-provider] [--dry-run] [--test-package NAME]
    python tools/make_field_bundle.py --verify DIR          (check a copied bundle against its manifest)

DIR gets:
  nodebased-<hash>.zip   the repository at HEAD (git archive; committed files only)
  wheels/                pip download of the project's dependencies plus the gpu and usd extras, and the build tools
  wheels-provider/       torch and torchvision from the CUDA index, then diffusers, transformers, accelerate, safetensors,
                         huggingface_hub, OpenEXR, numpy and pillow: what projects/nb-providers/sdxl/setup.sh installs
  weights/hub/           with --weights: the three SDXL model snapshots in the Hugging Face cache layout (about 9 GB)
  README.txt             the commands to run on the target machine
  MANIFEST.txt, manifest.json   size and sha256 of every file

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"   # the index setup.sh uses; cu126 wheels carry their own CUDA runtime
BUILD_TOOLS = ["setuptools>=68", "wheel", "pip"]          # `pip install .` builds the project; offline it needs these in find-links
PROVIDER_PACKAGES = ["diffusers>=0.32", "transformers", "accelerate", "safetensors", "huggingface_hub", "OpenEXR", "numpy", "pillow"]
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


README = """NodeBased field test, offline bundle (built {when}, NodeBased {version}, commit {commit})

Target: {platform}, CPython {python_version}. Full instructions: docs/FIELD_TEST.md in the zip, section OFFLINE.
Short form, in PowerShell, from this folder (call it E:\\nb-bundle):

  # 1. unpack the code
  Expand-Archive E:\\nb-bundle\\{zip} C:\\nb
  cd C:\\nb\\nodebased
  # 2. a virtual environment with NodeBased and the GPU and USD extras
  py -3.12 -m venv .venv
  .\\.venv\\Scripts\\Activate.ps1
  pip install --no-index --find-links E:\\nb-bundle\\wheels ".[gpu,usd]"
  # 3. the SDXL provider's packages go into the same environment
  pip install --no-index --find-links E:\\nb-bundle\\wheels-provider torch torchvision diffusers transformers accelerate safetensors huggingface_hub OpenEXR pillow
  # 4. the model weights (only if weights\\ is on the stick)
  Copy-Item -Recurse -Force E:\\nb-bundle\\weights\\hub\\* "$env:USERPROFILE\\.cache\\huggingface\\hub\\"
  # 5. run the report
  python tools\\field_report.py --out C:\\nb\\field-report --label "A6000 work machine"

Check the copy first with: python tools/make_field_bundle.py --verify E:\\nb-bundle   (any Python 3.11 or newer)
"""


def build_manifest(out):
    entries = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.txt", "manifest.json"):
            entries.append(dict(path=path.relative_to(out).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    total = sum(e["bytes"] for e in entries)
    (out / "manifest.json").write_text(json.dumps(dict(files=entries, total_bytes=total), indent=1), encoding="utf-8")
    lines = [f"{e['sha256']}  {e['bytes']:>12}  {e['path']}" for e in entries]
    (out / "MANIFEST.txt").write_text("\n".join(lines) + f"\n\n{len(entries)} files, {total} bytes ({human(total)})\n", encoding="utf-8")
    return entries, total


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

    # 2. wheels for NodeBased itself
    requirements = [args.test_package] if args.test_package else project_requirements() + BUILD_TOOLS
    print(f"project requirements: {requirements}", flush=True)
    if run(pip_download_command(out / "wheels", requirements, args.platform, args.python_version), args.dry_run):
        failures.append("pip download (project)")

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
        if run(pip_download_command(provider_dir, rest, args.platform, args.python_version, ["--find-links", str(provider_dir)]), args.dry_run):
            failures.append("pip download (provider)")

    # 4. weights
    if args.weights:
        copy_weights(out / "weights", args.dry_run)

    if args.dry_run:
        print("dry run: nothing was written")
        return 0
    (out / "README.txt").write_text(README.format(when=time.strftime("%Y-%m-%d"), version=version, commit=commit, zip=zip_name,
                                                  platform=args.platform, python_version=args.python_version), encoding="utf-8")
    entries, total = build_manifest(out)
    print("\n" + (out / "MANIFEST.txt").read_text(encoding="utf-8"))
    by_folder = {}
    for e in entries:
        top = e["path"].split("/")[0] if "/" in e["path"] else "(top level)"
        by_folder[top] = by_folder.get(top, 0) + e["bytes"]
    print("bundle size by folder: " + ", ".join(f"{k} {human(v)}" for k, v in sorted(by_folder.items())))
    print(f"bundle total: {human(total)} in {len(entries)} files at {out}")
    if failures:
        print("INCOMPLETE: " + ", ".join(failures) + " failed; the bundle is missing those parts. Re-run with a working network.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
