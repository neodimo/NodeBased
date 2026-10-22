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


def qt_global_to_x11(point, device_pixel_ratio=1.0):
    """Convert Qt's device-independent global point to native X11 pointer pixels.

    xdotool addresses the X server's physical pixel grid. At QT_SCALE_FACTOR=2,
    Qt returns logical coordinates, so passing those through unchanged targets a
    different point (usually outside the Roto handle being exercised).
    """
    ratio = float(device_pixel_ratio)
    if ratio <= 0:
        raise ValueError("device_pixel_ratio must be positive")
    return tuple(int(round(float(value) * ratio)) for value in point)


def qt_horizontal_drag_path(global_origin, width, height, device_pixel_ratio=1.0,
                            start_fraction=0.25, end_fraction=0.75):
    """Native X11 endpoints for an in-widget horizontal drag with a stable y coordinate."""
    width, height = float(width), float(height)
    if width <= 0 or height <= 0:
        raise ValueError("slider dimensions must be positive")
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("drag fractions must satisfy 0 <= start < end <= 1")
    x, y = map(float, global_origin)
    row = y + height / 2.0
    return (qt_global_to_x11((x + width * start_fraction, row), device_pixel_ratio),
            qt_global_to_x11((x + width * end_fraction, row), device_pixel_ratio))


def self_check():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT))
    from PySide6.QtWidgets import QApplication
    from nodebased.imaging import to_qimage
    import numpy as np
    app = QApplication.instance() or QApplication([])
    image = to_qimage(np.zeros((8, 8, 4), dtype=np.float32))
    assert image.width() == image.height() == 8
    assert qt_global_to_x11((120, 80), 2.0) == (240, 160)
    assert qt_global_to_x11((120, 80), 1.0) == (120, 80)
    print(json.dumps({"self_check": "ok", "adapter": "offscreen",
                      "high_dpi_pointer_mapping": "ok"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("live-picture.json"))
    parser.add_argument("--field", type=pathlib.Path,
                        help="write the field-kit JSON to this path (includes the selected adapter and backend)")
    parser.add_argument("--display", default=os.environ.get("DISPLAY"))
    parser.add_argument("--adapter", default=os.environ.get("QA_ADAPTER", "unknown"))
    parser.add_argument("--force-adapter", choices=("default", "integrated", "cpu"), default="default")
    parser.add_argument("--profile-thread", choices=("qt", "worker", "none"), default="none")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--warm-playback-range", action="store_true",
                        help="render every frame at the selected tier before measuring cache-only playback")
    parser.add_argument("--automatic-playback", action="store_true",
                        help="measure Proxy while playing from the artist's Full choice")
    parser.add_argument("--cache-tier", type=int, choices=(1, 2, 4, 8), default=4,
                        help="selected tier for the cached-range playback measurement")
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
    output = args.field or args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, str(driver), str(fixture), str(output),
                             str(args.seconds), args.adapter, args.force_adapter, args.profile_thread,
                             str(int(args.warm_playback_range)), str(int(args.automatic_playback)),
                             str(args.cache_tier)], cwd=ROOT,
                            env={**os.environ, "DISPLAY": args.display, "QT_QPA_PLATFORM": "xcb"},
                            check=False)
    raise SystemExit(result.returncode)


def shutil_which(name):
    from shutil import which
    return which(name)


if __name__ == "__main__":
    main()
