"""Native-display event-path smoke for the new 3D viewport, not the legacy 2D GL renderer."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from nodebased.app import Window
from nodebased.core import Dispatcher


def foundation_document():
    d = Dispatcher()
    for key, kind in (("card", "Card3D"), ("cube", "Cube3D"), ("cam", "Camera3D"),
                      ("scene", "Scene3D"), ("render", "Render3D"), ("write", "Write")):
        d.execute({"op": "create", "id": key, "type": kind})
    for node, slot, source in (("scene", "object0", "card"), ("scene", "object1", "cube"),
                               ("render", "scene", "scene"), ("render", "camera", "cam"),
                               ("write", "image", "render")):
        d.execute({"op": "connect", "id": node, "input": slot, "source": source})
    d.execute({"op": "view", "id": "write"})
    return d.document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", default="scratch/3d-ui-smoke/viewport.png")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    document = foundation_document()
    authored = copy.deepcopy(document["nodes"]["cam"]["params"])
    window = Window(document=document)
    window.resize(1100, 760)
    window.show()
    window.viewport_dock.setFloating(True)
    window.viewport_dock.resize(720, 520)
    window.viewport_dock.show()
    window.viewport_dock.raise_()
    app.processEvents()
    viewport = window.viewport
    viewport.setFocus()
    before = (viewport.azimuth, viewport.elevation, viewport.distance,
              viewport.pan_x, viewport.pan_y)
    center = QPoint(viewport.width() // 2, viewport.height() // 2)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(viewport, center + QPoint(55, 24), 120)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=center + QPoint(55, 24))
    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, pos=center)
    QTest.mouseMove(viewport, center + QPoint(28, -18), 120)
    QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton, pos=center + QPoint(28, -18))
    wheel = QWheelEvent(QPointF(center), QPointF(viewport.mapToGlobal(center)), QPoint(0, 0),
                        QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.ScrollUpdate, False)
    QApplication.sendEvent(viewport, wheel)
    app.processEvents()
    navigated = (viewport.azimuth, viewport.elevation, viewport.distance,
                 viewport.pan_x, viewport.pan_y)
    if navigated == before:
        raise AssertionError("orbit/pan/zoom events did not change the viewport navigation state")
    QTest.keyClick(viewport, Qt.Key.Key_F)
    app.processEvents()
    if authored != document["nodes"]["cam"]["params"]:
        raise AssertionError("viewport navigation mutated authored Camera3D parameters")
    output = Path(args.screenshot).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not viewport.grab().save(str(output), "PNG"):
        raise RuntimeError(f"could not save {output}")
    print(f"new 3D viewport event path passed; screenshot={output}; authored_camera_unchanged=true")
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
