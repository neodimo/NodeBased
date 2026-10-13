"""Capture the viewer showing the distorted shot's Undistort node with its overscan (docs/images/demo_shot/lens_view_undistorted.png).

An offscreen NodeBased window on `demo_shot_lens.nbcomp`, the plate pointed at the rendered distorted plate, the viewer on the
`undistort` node (what the strip's "View undistorted" does) with the format outlined (a dashed line) inside the
grown frame.

    python examples/demo_shot/capture_lens_view.py [--plate DIR] [--graph FILE] [--out FILE] [--frame N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

sys.path.insert(0, str(shot.HERE.parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", default=str(shot.distorted_plate_dir()))
    parser.add_argument("--graph", default=str(shot.LENS_GRAPH))
    parser.add_argument("--out", default=str(shot.HERE.parents[1] / "docs" / "images" / "demo_shot" / "lens_view_undistorted.png"))
    parser.add_argument("--frame", type=int, default=12)
    args = parser.parse_args()
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from nodebased.app import STYLE, Window
    from nodebased.core import Dispatcher
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    QSettings("NodeBased", "NodeBased").clear()
    document = json.loads(Path(args.graph).read_text())
    dispatcher = Dispatcher(document)
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "set", "id": "plate", "param": "path", "value": shot.plate_pattern(args.plate)},
        {"op": "time", "first": 1, "last": shot.FRAMES, "current": args.frame},
        {"op": "view", "id": "undistort"},
        ]})
    window = Window(dispatcher.document)
    window.resize(1440, 900)
    window.graph_dock.hide()
    window.nodes_dock.hide()
    window._default_split_pending = False
    window.show()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not (window.frame is not None and window.viewer.sceneRect().width() > shot.DISTORTED_SIZE[0]):
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    window.grab().save(args.out)
    rect, shown = window.viewer.sceneRect(), window.viewer.format_rect
    print(f"wrote {args.out}: frame {rect.width():.0f} x {rect.height():.0f}, format {shown.width():.0f} x {shown.height():.0f}")
    window.saved_document = window.dispatcher.document
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
