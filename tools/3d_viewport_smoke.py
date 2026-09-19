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
    """Checker-textured card, a cube and a sphere under one key light, rendered and written."""
    d = Dispatcher()
    for key, kind in (("plate", "Checker"), ("card", "Card3D"), ("cube", "Cube3D"), ("ball", "Sphere3D"),
                      ("key", "Light3D"), ("cam", "Camera3D"), ("scene", "Scene3D"),
                      ("render", "Render3D"), ("write", "Write")):
        d.execute({"op": "create", "id": key, "type": kind})
    for node, slot, source in (("card", "image", "plate"), ("scene", "object0", "card"),
                               ("scene", "object1", "cube"), ("scene", "object2", "ball"),
                               ("scene", "object3", "key"), ("render", "scene", "scene"),
                               ("render", "camera", "cam"), ("write", "image", "render")):
        d.execute({"op": "connect", "id": node, "input": slot, "source": source})
    for node, param, value in (("plate", "width", 512), ("plate", "height", 512), ("plate", "size", 64),
                               ("card", "card_width", 4.0), ("card", "card_height", 4.0),
                               ("card", "tz", -2.0), ("card", "red", 1.0), ("card", "green", 1.0),
                               ("card", "blue", 1.0), ("cube", "tx", -1.6), ("cube", "cube_size", 1.4),
                               ("cube", "ry", 30.0), ("cube", "green", 0.45), ("cube", "blue", 0.3),
                               ("ball", "tx", 1.5), ("ball", "sphere_radius", 0.9), ("ball", "red", 0.35),
                               ("cam", "tx", 2.0), ("cam", "ty", 1.5), ("cam", "tz", 7.0)):
        d.execute({"op": "set", "id": node, "param": param, "value": value})
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
    before = (viewport.azimuth, viewport.elevation, viewport.distance, *viewport.center)
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
    navigated = (viewport.azimuth, viewport.elevation, viewport.distance, *viewport.center)
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
    QTest.keyClick(viewport, Qt.Key.Key_C)
    app.processEvents()
    viewport.grab().save(str(output.with_name("through-camera.png")), "PNG")
    if viewport.status:
        raise AssertionError(f"viewport reported: {viewport.status}")
    from nodebased.imaging import Evaluator, write_png
    write_png(str(output.with_name("render3d.png")), Evaluator().evaluate(document, "render"))
    print(f"new 3D viewport event path passed; screenshot={output}; authored_camera_unchanged=true")
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
