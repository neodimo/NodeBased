"""Renders docs/images/fluids_shape_controls.png: one plume, no shaping and then each shape control on its
own, for docs/FLUIDS_SPIKE.md's "Shape controls" section (Lane 6, Pyro production step 2).

Run with: QT_QPA_PLATFORM=offscreen PYTHONPATH=. .venv/bin/python tools/fluids_shape_controls_image.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from nodebased import fluid3d

GRID = {"nx": 28, "ny": 40, "nz": 28}
FRAMES = 26
PANELS = (
    ("no shaping", {}),
    ("disturbance", {"disturbance": 6.0, "disturbance_size": 2.0}),
    ("shredding", {"shredding": 4.0}),
    ("turbulence", {"turbulence": 0.8, "swirl_size": 3.0}),
    ("confinement", {"vorticity": 1.2}),
)


def render(extra):
    solver = fluid3d.Smoke3D({**GRID, **extra})
    state = solver.initial_state()
    for frame in range(1, FRAMES + 1):
        state = solver.step(state, frame, 0, 0)
    # a front (z) max-intensity projection, normalised per panel so a quiet control still shows structure
    projection = np.ascontiguousarray(state.arrays["density"].max(axis=2).T[::-1])
    peak = float(projection.max()) or 1.0
    return np.clip(projection / peak, 0.0, 1.0)


def main():
    from PySide6.QtGui import QImage, QPainter, QColor, QFont
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.instance() or QGuiApplication([])
    panels = [(label, render(extra)) for label, extra in PANELS]
    ny, nx = panels[0][1].shape
    margin, label_h, gap = 6, 18, 6
    cell_w, cell_h = nx * 3, ny * 3
    sheet = QImage(len(panels) * (cell_w + gap) + gap, cell_h + label_h + 2 * margin, QImage.Format.Format_RGB888)
    sheet.fill(QColor(18, 18, 20))
    painter = QPainter(sheet)
    painter.setFont(QFont("Sans", 11))
    for index, (label, projection) in enumerate(panels):
        x0 = gap + index * (cell_w + gap)
        gray = np.ascontiguousarray((projection * 255).astype(np.uint8))
        image = QImage(gray.data, nx, ny, gray.strides[0], QImage.Format.Format_Grayscale8).convertToFormat(
            QImage.Format.Format_RGB888).scaled(cell_w, cell_h)
        painter.drawImage(x0, margin, image)
        painter.setPen(QColor(230, 230, 230))
        painter.drawText(x0, margin + cell_h + label_h - 4, label)
    painter.end()
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "fluids_shape_controls.png")
    sheet.save(os.path.abspath(out), "PNG")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
