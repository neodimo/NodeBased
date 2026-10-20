#!/usr/bin/env python3
"""Measure the Lane 8 live-picture interaction on a real Qt window.

Run under Xvfb (or an existing X display) with the repository venv. Each run
writes one JSON document containing drag, uncached scrub, and tier-4 playback.
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
QA = pathlib.Path(__file__).resolve().parents[2] / "scratch/nb-qa/1015-roi"


def self_check():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT))
    from PySide6.QtWidgets import QApplication
    from nodebased.imaging import to_qimage
    import numpy as np
    app = QApplication.instance() or QApplication([])
    image = to_qimage(np.zeros((8, 8, 4), dtype=np.float32))
    assert image.width() == image.height() == 8
    print(json.dumps({"self_check": "ok", "adapter": "offscreen"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("live-picture.json"))
    parser.add_argument("--display", default=os.environ.get("DISPLAY"))
    parser.add_argument("--adapter", default=os.environ.get("QA_ADAPTER", "unknown"))
    parser.add_argument("--force-adapter", choices=("default", "integrated", "cpu"), default="default")
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.display:
        parser.error("set DISPLAY to Xvfb (for example :48) or a real X display")
    if shutil_which("xdotool") is None:
        parser.error("xdotool is required")
    fixture = ROOT / "tests/fixtures/qa.nbcomp"
    if not fixture.exists():
        parser.error(f"QA comp is missing: {fixture}")
    driver = pathlib.Path(__file__).with_name("_live_picture_driver.py")
    result = subprocess.run([sys.executable, str(driver), str(fixture), str(args.output),
                             str(args.seconds), args.adapter, args.force_adapter], cwd=ROOT,
                            env={**os.environ, "DISPLAY": args.display, "QT_QPA_PLATFORM": "xcb"},
                            check=False)
    raise SystemExit(result.returncode)


def shutil_which(name):
    from shutil import which
    return which(name)


if __name__ == "__main__":
    main()
