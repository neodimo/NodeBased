"""Spike A: side-by-side sheets of the representative frames (1, 12, 24): the stage-8 still, the stage-3 still, the candidate's
canvas-guided final, and the four denoising steps of an independent edit with the delivered picture.

    python tools/spike_a_sheets.py --store STORE --generated-id ID --evidence DIR --out DIR [--frames 1,12,24]

Offscreen Qt only (no Pillow in the NodeBased environment). The columns are labelled; every tile is scaled to 320 pixels wide
and the labels carry the true sizes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True)
    parser.add_argument("--generated-id", required=True, help="the candidate's canvas-guided generated_sequence artifact id")
    parser.add_argument("--evidence", required=True, help="the interaction bench's evidence folder (steps and delivered pictures)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", default="1,12,24")
    args = parser.parse_args()
    os.environ["NODEBASED_CACHE"] = args.store
    import numpy as np
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from nodebased import color
    from nodebased.artifacts import ArtifactStore
    from nodebased.conditioned_read import _read_rgba
    from nodebased.loops import _unpack_artifact
    app = QApplication.instance() or QApplication([])
    out, evidence = Path(args.out), Path(args.evidence)
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".generated"
    work.mkdir(exist_ok=True)
    _unpack_artifact(ArtifactStore().get(args.generated_id), work)
    to_srgb = color.processor(color.WORKING, color.INPUT_SPACES["sRGB"])

    def exr_tile(frame):
        rgba = np.ascontiguousarray(_read_rgba(str(next(work.glob(f"*{frame:04d}.exr"))))[0], dtype=np.float32).copy()
        to_srgb.applyRGBA(rgba)
        data = np.ascontiguousarray(np.rint(np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8))
        return QImage(data.data, data.shape[1], data.shape[0], data.shape[1] * 3, QImage.Format.Format_RGB888).copy()

    for number in [int(x) for x in args.frames.split(",")]:
        tiles = [
            (f"stage 8 baseline: 30 steps, canvas-guided (480x270)", QImage(str(REPO / "docs/images/demo_shot" / f"sdxl_controlnet_reprojected_frame_{number:04d}.png"))),
            (f"stage 3 baseline: 30 steps, independent (480x270)", QImage(str(REPO / "docs/images/demo_shot" / f"sdxl_controlnet_frame_{number:04d}.png"))),
            ("Lightning canvas-guided final (480x270)", exr_tile(number))]
        for step in range(1, 5):
            tiles.append((f"independent edit, step {step} of 4 (1024x576)", QImage(str(evidence / f"frame{number:02d}_{step}of4.png"))))
        tiles.append(("independent edit, delivered (480x270)", QImage(str(evidence / f"frame{number:02d}_delivered.png"))))
        width, label = 320, 34
        height = max(int(width * t.height() / t.width()) for _, t in tiles)
        columns = 4
        rows = (len(tiles) + columns - 1) // columns
        sheet = QImage(columns * width, rows * (height + label), QImage.Format.Format_RGB32)
        sheet.fill(QColor("#101216"))
        painter = QPainter(sheet)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setPen(QColor("#d8dce4"))
        painter.setFont(QFont("sans", 8))
        for i, (text, tile) in enumerate(tiles):
            x, y = (i % columns) * width, (i // columns) * (height + label)
            painter.drawImage(QRectF(x, y, width, width * tile.height() / tile.width()), tile)
            painter.drawText(QRectF(x + 4, y + height + 2, width - 8, label - 2), Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, text)
        painter.end()
        sheet.save(str(out / f"sheet_frame_{number:02d}.png"))
        print(out / f"sheet_frame_{number:02d}.png")


if __name__ == "__main__":
    main()
