"""Capture the 3D viewport looking through the shot's camera, with the lens off and on, over the distorted plate
(docs/images/demo_shot/viewport_lens_on.png), and the viewer strip with its Lens distortion switch
(docs/images/demo_shot/viewport_lens_strip.png).

The plate is the rendered distorted plate of `demo_shot_lens.nbcomp` (a wide-angle lens, k1 -0.04 and k2 0.004); the
viewport looks through the graph's solved camera, whose lens is the one the solve recovered. Each panel is the plate
with the viewport's picture laid over it at half strength: a line of the layout that sits on the same line of the plate
is sharp, one that misses it is doubled. Left, the ideal picture the viewport drew before step G2; right, the picture
drawn through the lens.

    python examples/demo_shot/capture_viewport_lens.py [--plate DIR] [--graph FILE] [--frame N] [--size WxH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

sys.path.insert(0, str(shot.HERE.parents[1]))
IMAGES = shot.HERE.parents[1] / "docs" / "images" / "demo_shot"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", default=str(shot.distorted_plate_dir()))
    parser.add_argument("--graph", default=str(shot.LENS_GRAPH))
    parser.add_argument("--frame", type=int, default=12)
    parser.add_argument("--size", default="960x540")
    parser.add_argument("--out", default=str(IMAGES / "viewport_lens_on.png"))
    parser.add_argument("--strip-out", default=str(IMAGES / "viewport_lens_strip.png"))
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PySide6.QtCore import QRect, QSettings, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from nodebased.app import STYLE, Window
    from nodebased.core import Dispatcher, upgrade_document
    from nodebased.imaging import Evaluator, linear_to_srgb
    from nodebased.viewport3d import Viewport3D
    width, height = (int(v) for v in args.size.split("x"))
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    QSettings("NodeBased", "NodeBased").clear()
    dispatcher = Dispatcher(upgrade_document(json.loads(Path(args.graph).read_text())))
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "set", "id": "plate", "param": "path", "value": shot.plate_pattern(args.plate)},
        {"op": "time", "first": 1, "last": shot.FRAMES, "current": args.frame},
        {"op": "connect", "id": "render", "input": "camera", "source": "camera"},
        {"op": "view", "id": "render"}]})
    raster = Evaluator().evaluate_raster(dict(dispatcher.document, view="plate"), "plate", frame=args.frame, tier=1)
    rgb = np.clip(linear_to_srgb(np.clip(raster.pixels[..., :3], 0, 1)), 0, 1)
    plate = np.ascontiguousarray((np.concatenate((rgb, np.ones_like(rgb[..., :1])), axis=2) * 255).astype(np.uint8))
    plate_image = QImage(plate.data, plate.shape[1], plate.shape[0], plate.strides[0],
                         QImage.Format.Format_RGBA8888).copy().scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                                      Qt.TransformationMode.SmoothTransformation)

    viewport = Viewport3D()
    viewport.setMinimumSize(0, 0)
    viewport.resize(width, height)
    viewport.set_document(dispatcher.document)
    viewport.look_through = True
    viewport.show()
    panels = []
    for on in (False, True):
        viewport.set_lens_view(on)
        app.processEvents()
        picture = viewport.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        assert viewport._lens_drawn == on
        panel = plate_image.copy()
        painter = QPainter(panel)
        painter.setOpacity(0.5)
        painter.drawImage(0, 0, picture)
        painter.end()
        panels.append(panel)
    gap, head = 12, 34
    sheet = QImage(2 * width + gap, height + head, QImage.Format.Format_RGBA8888)
    sheet.fill(QColor("#101014"))
    painter = QPainter(sheet)
    painter.setFont(QFont("Sans", 11))
    painter.setPen(QColor("#e8e8ee"))
    for index, (panel, label) in enumerate(zip(panels, ("Lens off: the ideal picture over the distorted plate",
                                                       "Lens on: the picture through the camera's lens"))):
        x = index * (width + gap)
        painter.drawImage(x, head, panel)
        painter.drawText(QRect(x + 10, 0, width - 20, head), int(Qt.AlignmentFlag.AlignVCenter), label)
    painter.end()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    viewport.close()

    window = Window(dispatcher.document)
    window.resize(1760, 900)
    window.graph_dock.hide()
    window.nodes_dock.hide()
    window._default_split_pending = False
    window.show()
    window.set_viewer_mode("3d")
    window.viewport.look_through = True
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not window.viewer_strip.lens3d_present:
        window.viewer_strip.sync_lens_3d()
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()
    window.grab().save(args.strip_out)
    window.saved_document = window.dispatcher.document
    window.close()
    print("wrote", args.out, args.strip_out)


if __name__ == "__main__":
    main()
