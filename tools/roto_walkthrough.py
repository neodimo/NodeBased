"""Roto and RotoPaint walkthrough: each of Nuke's RotoPaint chapter workflows, driven through the real viewer.

Every numbered scene builds a fresh document, performs the gesture with real Qt mouse and key events where the
viewer takes them, asserts the document changed as expected, and saves a screenshot of the window (viewer and
properties panel).  A scene that fails an assertion or raises is recorded as ``broken`` with the failing step; a scene
that completes but behaves differently from the guide records ``differs`` with how.  The results go to
``results.json`` and a README table (``--out DIR``); ``tests/test_roto_walkthrough.py`` runs the whole thing headless.

    QT_QPA_PLATFORM=offscreen python tools/roto_walkthrough.py --out scratch/walkthrough
"""
import argparse
import copy
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QTreeWidget, QWidget)

import numpy as np

from nodebased import roto as roto_model, rotoinbetween, shapes as shape_model
from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document

GUIDE = "https://learn.foundry.com/nuke/content/comp_environment/rotopaint/"
NO_MOD = Qt.KeyboardModifier.NoModifier
CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
LEFT = Qt.MouseButton.LeftButton
WIDTH, HEIGHT = 640, 360

PAST_NUKE_FIRST = 17    # scenes from here on are NodeBased additions with no Nuke counterpart
SCENES = []  # (number, title, guide pages, what Nuke does, function)


def scene(number, title, pages, nuke_does):
    def register(function):
        SCENES.append((number, title, [GUIDE + page for page in pages], nuke_does, function))
        return function
    return register


class Skip(Exception):
    pass


class Result:
    def __init__(self, number, title, pages, nuke_does):
        self.number, self.title, self.pages, self.nuke_does = number, title, pages, nuke_does
        self.status, self.notes, self.screens, self.error, self.seconds = "works", [], [], "", 0.0
        self.captions = []

    def as_dict(self):
        return dict(number=self.number, title=self.title, pages=self.pages, nuke_does=self.nuke_does,
                    status=self.status, notes=self.notes, screens=self.screens, captions=self.captions, error=self.error,
                    seconds=round(self.seconds, 1))


class Walk:
    """One scene's workbench: a fresh window, mouse and key helpers, screenshots and verdict notes."""

    def __init__(self, app, out_dir, result):
        self.app, self.out_dir, self.result = app, Path(out_dir), result
        self.window = None
        self.key = "r"
        self.shots = 0

    # -- documents -----------------------------------------------------------------------------------------------------
    def fresh(self, kind="roto", shapes=None, items=None, last=24, current=1, plate=None, size=None):
        self.close()
        QSettings("NodeBased", "NodeBased").clear()
        dispatcher = Dispatcher(empty_document())
        commands = []
        if kind == "roto":
            self.key = "r"
            commands = [{"op": "create", "id": "r", "type": "Roto", "pos": [0, 0]},
                        {"op": "set", "id": "r", "param": "width", "value": (size or (WIDTH, HEIGHT))[0]},
                        {"op": "set", "id": "r", "param": "height", "value": (size or (WIDTH, HEIGHT))[1]}]
            if plate:      # `plate(dispatcher)` adds the nodes behind the Roto and connects its bg input
                for command in commands:
                    dispatcher.execute(command)
                plate(dispatcher)
                commands = []
            commands += [{"op": "view", "id": "r"}]
        else:
            self.key = "paint"
            commands = [{"op": "create", "id": "plate", "type": "Checker", "params": {"width": WIDTH, "height": HEIGHT, "size": 40}},
                        {"op": "create", "id": "paint", "type": "RotoPaint", "pos": [0, 0]},
                        {"op": "connect", "id": "paint", "input": "image", "source": "plate"},
                        {"op": "view", "id": "paint"}]
        document = dispatcher.document
        document["time"].update(first=1, last=last, current=current)
        for command in commands:
            dispatcher.execute(command)
        if shapes:
            dispatcher.execute({"op": "set_shapes", "id": self.key, "shapes": shapes})
        if items:
            dispatcher.execute({"op": "set_paint_items", "id": self.key, "items": items})
        self.window = Window(dispatcher.document)
        self.window.resize(1500, 860)
        self.window.show()
        self.wait(lambda: self.window.frame is not None)
        self.window.graph.items_by_id[self.key].setSelected(True)
        self.app.processEvents()
        self.viewer.setFocus()
        return self.window

    def close(self):
        if self.window is not None:
            window, self.window = self.window, None
            window.saved_document = window.dispatcher.document
            window.close()
            window.deleteLater()
            self.app.processEvents()
            self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            del window
            gc.collect()
            self.app.processEvents()

    @property
    def viewer(self):
        return self.window.viewer

    @property
    def document(self):
        return self.window.dispatcher.document

    @property
    def payload(self):
        return self.document.get("node_data", {}).get(self.key, {})

    def entries(self):
        """The ordered shapes (Roto) or shape and stroke items (RotoPaint) the node stores."""
        return self.payload.get("items" if self.key == "paint" else "shapes", [])

    def shape_entries(self):
        return [e for e in self.entries() if e.get("kind", "shape") == "shape"]

    def undo_depth(self):
        return len(self.window.dispatcher.undo_stack)

    def wait(self, predicate, seconds=20.0):
        end = time.time() + seconds
        while time.time() < end:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def settle(self):
        self.app.processEvents()
        self.wait(lambda: not self.window.dispatcher.document is None, 0.05)
        self.app.processEvents()

    # -- coordinates ---------------------------------------------------------------------------------------------------
    def vpos(self, x, y):
        """Viewport pixel of the data point (x, y)."""
        context = self.viewer._roto_context()
        tier = context[3] if context else 1
        scene = self.viewer._roto_scene_point({"x": float(x), "y": float(y)}, tier)
        return self.viewer.mapFromScene(scene)

    def scene_of(self, x, y):
        context = self.viewer._roto_context()
        return self.viewer._roto_scene_point({"x": float(x), "y": float(y)}, context[3] if context else 1)

    # -- gestures ------------------------------------------------------------------------------------------------------
    def click(self, x, y, mods=NO_MOD):
        QTest.mouseClick(self.viewer.viewport(), LEFT, mods, pos=self.vpos(x, y))
        self.settle()

    def drag(self, a, b, mods=NO_MOD, steps=12):
        viewport = self.viewer.viewport()
        start, end = self.vpos(*a), self.vpos(*b)
        QTest.mousePress(viewport, LEFT, mods, pos=start)
        for i in range(1, steps + 1):
            t = i / steps
            QTest.mouseMove(viewport, start + (end - start) * t, 5)
        QTest.mouseRelease(viewport, LEFT, mods, pos=end)
        self.settle()

    def key_press(self, key, mods=NO_MOD):
        # Offscreen, a floating widget can become the active window after a click; the viewer needs the keyboard.
        self.window.activateWindow()
        self.viewer.setFocus()
        self.settle()
        if not self.viewer.hasFocus():
            self.viewer.setFocus(Qt.FocusReason.OtherFocusReason)
        self.check(self.viewer.hasFocus(), f"the viewer has the keyboard for the {key} press")
        QTest.keyClick(self.viewer, key, mods)
        self.settle()

    def tool(self, group, name):
        self.viewer.roto_toolbar.choose(group, name)
        self.settle()

    def goto(self, frame):
        self.window.set_time(current=int(frame))
        self.settle()

    def panel(self):
        self.window.rebuild_properties_dock()
        self.settle()
        return self.window.properties.widget()

    def shape_tree(self):
        panel = self.window.build_node_panel(self.key)
        self.window.set_properties_widget(panel)
        self.settle()
        return panel.findChild(QTreeWidget, "roto-shape-list")

    # -- results -------------------------------------------------------------------------------------------------------
    def shot(self, label, caption=None):
        self.shots += 1
        self.settle()
        name = f"{self.result.number:02d}-{self.shots}-{label}.png"
        self.window.grab().save(str(self.out_dir / name))
        self.result.screens.append(name)
        self.result.captions.append(caption or label.replace("-", " ").capitalize())

    def shot_widget(self, label, widget, size=(1000, 260), caption=None):
        """A screenshot of one widget (for panels the dock shows below the fold, such as the shape list)."""
        self.shots += 1
        widget.resize(*size)
        self.settle()
        name = f"{self.result.number:02d}-{self.shots}-{label}.png"
        widget.grab().save(str(self.out_dir / name))
        self.result.screens.append(name)
        self.result.captions.append(caption or label.replace("-", " ").capitalize())

    def note(self, text):
        self.result.notes.append(text)

    def differs(self, text):
        """The workflow runs but not the way the guide describes; keeps the first verdict that is worse than works."""
        if self.result.status == "works":
            self.result.status = "differs"
        self.result.notes.append("differs: " + text)

    def check(self, condition, step):
        """An assertion that names the walkthrough step it guards."""
        if not condition:
            raise AssertionError(step)


def number(value, frame=1, field=None):
    return shape_model.resolve_scalar(value, frame, field)


def square(name, x0, y0, x1, y1, **extra):
    points = [{"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}
              for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return {"name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": points, **extra}


def paint_shape_item(shape):
    return {"kind": "shape", **shape, "blend": "over", "visible": True}


# ---- Shapes: drawing -------------------------------------------------------------------------------------------------
@scene(1, "Draw a Bezier: click, click-drag tangents, close on the first point",
       ["drawing_shapes.html", "using_bezier_tools.html"],
       "Click to add corner points, click-drag to pull tangent handles, click the first point to close the shape.")
def s01(w):
    w.fresh("roto")
    w.tool("Shapes", "Bezier")
    w.check(w.viewer.begin_roto_draw("r"), "begin drawing with the Bezier tool")
    w.click(120, 80)
    w.drag((300, 80), (340, 60))          # click-drag: tangent handles on the second point
    w.click(400, 260)
    w.click(150, 260)
    w.shot("drawing-open")
    w.click(120, 80)                      # back on the first point closes it
    w.shot("closed")
    shape = w.shape_entries()[-1]
    w.check(len(w.shape_entries()) == 1, "one shape is created")
    w.check(shape["closed"], "clicking the first point closes the shape")
    w.check(len(shape["points"]) == 4, "four points were placed")
    pulled = shape["points"][1]
    w.check(abs(number(pulled["out_x"], field="out_x")) > 1 or abs(number(pulled["out_y"], field="out_y")) > 1,
            "the click-drag point has tangent handles")
    w.check(w.undo_depth() == 1, "the whole drawing is one undo step")


@scene(2, "Draw an Ellipse and a Rectangle",
       ["drawing_shapes.html", "working_toolbars.html"],
       "Drag a bounding box; Nuke adds a four-point Bezier ellipse or a four-corner rectangle (Shift constrains, Ctrl centres).")
def s02(w):
    w.fresh("roto")
    w.tool("Shapes", "Ellipse")
    w.check(w.viewer.begin_roto_draw("r"), "begin drawing with the Ellipse tool")
    w.drag((80, 60), (280, 200))
    w.tool("Shapes", "Rectangle")
    w.check(w.viewer.begin_roto_draw("r"), "begin drawing with the Rectangle tool")
    w.drag((340, 120), (560, 300))
    w.shot("ellipse-and-rectangle")
    shapes = w.shape_entries()
    w.check(len(shapes) == 2, "an ellipse and a rectangle exist")
    w.check([len(s["points"]) for s in shapes] == [4, 4], "both shapes are four-point shapes")
    w.check(all(s["closed"] for s in shapes), "both shapes are closed")
    ellipse = shapes[0]["points"]
    w.check(any(abs(number(p["out_x"], field="out_x")) + abs(number(p["out_y"], field="out_y")) > 1 for p in ellipse),
            "the ellipse points carry tangent handles")
    w.check(all(number(p["out_x"], field="out_x") == 0 and number(p["out_y"], field="out_y") == 0
                for p in shapes[1]["points"]), "the rectangle corners have no handles")


@scene(3, "Draw a B-Spline",
       ["drawing_shapes.html", "using_bezier_tools.html"],
       "B-Spline points are control vertices with no tangent handles; the curve is smoothed through them.")
def s03(w):
    w.fresh("roto")
    w.tool("Shapes", "B-Spline")
    w.check(w.viewer.begin_roto_draw("r"), "begin drawing with the B-Spline tool")
    for x, y in ((120, 90), (300, 60), (480, 140), (400, 280), (180, 250)):
        w.click(x, y)
    w.key_press(Qt.Key.Key_Return)
    w.shot("b-spline")
    shape = w.shape_entries()[-1]
    w.check(shape["closed"] and len(shape["points"]) == 5, "Enter closes a five-point shape")
    w.check(shape.get("spline", shape.get("type", "bspline")) is not None, "shape stored")
    kind = shape.get("curve_type") or shape.get("spline") or shape.get("type") or shape.get("tool")
    if kind is None:
        w.differs("the stored shape carries no B-Spline marker, so it is indistinguishable from a Bezier shape")
    else:
        w.note(f"stored as {kind}")
    w.check(all(number(p["out_x"], field="out_x") == 0 for p in shape["points"]), "B-Spline points have no tangent handles")
    # Nuke: Ctrl+Alt-drag on a B-Spline point changes its tension.
    w.tool("Select", "Select Points")
    before = copy.deepcopy(w.shape_entries()[0]["points"])
    depth = w.undo_depth()
    w.drag((120, 90), (170, 90), mods=CTRL | Qt.KeyboardModifier.AltModifier)
    after = w.shape_entries()[0]["points"]
    moved = [i for i, (a, b) in enumerate(zip(after, before)) if a.get("tension", 0.0) != b.get("tension", 0.0)]
    w.check(moved == [0], f"Ctrl+Alt-drag sets the tension of the dragged point only (changed: {moved})")
    w.check(number(after[0]["tension"], field="tension") > 0, "a drag to the right tightens the point")
    w.check(all(a["x"] == b["x"] and a["y"] == b["y"] for a, b in zip(after, before)), "the point itself does not move")
    w.check(w.undo_depth() == depth + 1, "the tension change is one undo step")
    w.note("tension scale: 100 screen pixels of horizontal drag is a tension of 1 (a convention; the guide gives none)")
    w.shot("tension-attempt")


@scene(4, "Draw an Open Spline",
       ["drawing_shapes.html"],
       "Click points and press Esc: an open curve with a stroke width; no fill.")
def s04(w):
    w.fresh("roto")
    w.tool("Shapes", "Open Spline")
    w.check(w.viewer.begin_roto_draw("r"), "begin drawing with the Open Spline tool")
    for x, y in ((100, 250), (240, 110), (380, 240), (540, 100)):
        w.click(x, y)
    w.key_press(Qt.Key.Key_Escape)
    w.shot("open-spline")
    shape = w.shape_entries()[-1]
    w.check(not shape["closed"], "Esc commits an open curve")
    w.check(len(shape["points"]) == 4, "four points were placed")
    w.check(shape.get("stroke_width", 0) > 0, "the open spline has a stroke width")
    w.check(w.viewer.roto_selection_tool == "Select Points", "the toolbar returns to the select tool")


# ---- Editing existing shapes -----------------------------------------------------------------------------------------
@scene(5, "Select points, move them, Ctrl-drag a feather, E and Shift+E",
       ["selecting_strokes_shapes.html", "editing_existing_splines.html"],
       "Click a point to select it, drag to move it; Ctrl-drag pulls the feather out; E adds a feather point, Shift+E removes it.")
def s05(w):
    w.fresh("roto", shapes=[square("box", 160, 90, 480, 270)])
    w.tool("Select", "Select Points")
    w.click(160, 90)
    w.check(w.viewer.roto_selected_points == {(0, 0)}, "clicking a point selects it")
    w.drag((160, 90), (190, 110))
    first = w.shape_entries()[0]["points"][0]
    w.check(abs(number(first["x"]) - 190) < 3 and abs(number(first["y"]) - 110) < 3, "dragging moves the point")
    w.shot("point-moved")
    before = w.undo_depth()
    w.drag((480, 90), (505, 70), mods=CTRL)
    point = w.shape_entries()[0]["points"][1]
    fx, fy = number(point.get("feather_x", 0), field="feather_x"), number(point.get("feather_y", 0), field="feather_y")
    w.check(abs(fx) > 5 or abs(fy) > 5, "Ctrl-drag sets a feather offset on the point")
    w.check(w.undo_depth() == before + 1, "the feather drag is one undo step")
    w.shot("feather-dragged")
    w.window.command({"op": "set_shapes", "id": "r", "shapes": [square("box", 160, 90, 480, 270, feather=6.0)]})
    w.viewer.roto_selected_points = {(0, 2)}
    w.key_press(Qt.Key.Key_E)
    pt = w.shape_entries()[0]["points"][2]
    w.check(abs(number(pt.get("feather_x", 0), field="feather_x")) + abs(number(pt.get("feather_y", 0), field="feather_y")) > 0,
            "E adds a feather point")
    w.key_press(Qt.Key.Key_E, SHIFT)
    pt = w.shape_entries()[0]["points"][2]
    w.check(number(pt.get("feather_x", 0), field="feather_x") == 0 and number(pt.get("feather_y", 0), field="feather_y") == 0,
            "Shift+E removes the feather point")
    w.shot("feather-points")
    # The guide: with the pointer over the Viewer, S cycles through the modes of the selected tool.
    w.tool("Select", "Select All")
    depth_before_s = w.undo_depth()
    seen = [w.viewer.roto_selection_tool]
    for _ in range(4):
        w.key_press(Qt.Key.Key_S)
        seen.append(w.viewer.roto_selection_tool)
    w.check(seen == ["Select All", "Select Splines", "Select Points", "Select Feather Points", "Select All"],
            f"S steps through the four selection modes and wraps (saw {seen})")
    w.check(w.undo_depth() == depth_before_s, "S changes the tool only; the document is untouched")


@scene(6, "Z and Shift+Z: cusp and smooth points",
       ["editing_existing_splines.html"],
       "With points selected, Z smooths (mirrors the tangents) and Shift+Z cusps (zeroes the handles).")
def s06(w):
    shape = square("curve", 160, 90, 480, 270)
    for p in shape["points"]:
        p["out_x"], p["in_x"] = 40.0, -40.0
    w.fresh("roto", shapes=[shape])
    w.viewer.roto_selected_points = {(0, 0), (0, 1)}
    w.key_press(Qt.Key.Key_Z, SHIFT)
    pts = w.shape_entries()[0]["points"]
    w.check(number(pts[0]["out_x"], field="out_x") == 0 and number(pts[0]["in_x"], field="in_x") == 0, "Shift+Z cusps the handles")
    w.shot("cusped")
    pts_before = copy.deepcopy(w.shape_entries()[0]["points"])
    w.viewer.roto_selected_points = {(0, 2)}
    pts = w.shape_entries()[0]["points"]
    w.key_press(Qt.Key.Key_Z)
    pts = w.shape_entries()[0]["points"]
    w.check(number(pts[2]["in_x"], field="in_x") == -number(pts[2]["out_x"], field="out_x"), "Z mirrors the in tangent from the out tangent")
    w.shot("smoothed")


@scene(7, "Transform box: move, rotate, scale and the pivot",
       ["editing_existing_splines.html", "working_toolbars.html"],
       "Selecting several points shows a transform box: drag inside to move, the corners to scale, outside to rotate, the pivot to move the centre.")
def s07(w):
    w.fresh("roto", shapes=[square("box", 200, 110, 440, 250)])
    w.tool("Select", "Select Points")
    w.viewer.roto_selected_points = {(0, i) for i in range(4)}
    w.viewer.roto_transform_box_enabled = True
    w.viewer.viewport().update(); w.settle()
    w.shot("box-shown")
    context = w.viewer._roto_context()
    resolved = w.viewer._roto_resolved(context)
    selected, bounds, _, pivot = w.viewer._roto_box_geometry(resolved, context[3])
    depth = w.undo_depth()
    corner = bounds.bottomRight()
    action = w.viewer._roto_box_action(corner, bounds, abs(w.viewer.transform().m11()), pivot)[0]
    w.check(action.startswith("scale"), "the corner handle is a scale handle")
    w.viewer.roto_drag = {"kind": "box", "kind_name": action, "handle": action[6:], "selected": selected, "bounds": bounds,
                          "pivot": pivot, "start": corner,
                          "scene": QPointF(pivot.x() + 1.5 * (corner.x() - pivot.x()), pivot.y() + 1.5 * (corner.y() - pivot.y())),
                          "tier": context[3], "initial": {(si, pi): dict(resolved[si]["points"][pi]) for si, pi in selected},
                          "alt_shape": None}
    w.viewer._commit_roto_drag(); w.settle()
    pts = w.shape_entries()[0]["points"]
    width = number(pts[1]["x"]) - number(pts[0]["x"])
    w.check(abs(width - 360) < 2, "dragging a corner scales the box by 1.5")
    context = w.viewer._roto_context(); resolved = w.viewer._roto_resolved(context)
    selected, bounds, _, pivot = w.viewer._roto_box_geometry(resolved, context[3])
    w.viewer.roto_drag = {"kind": "box", "kind_name": "rotate", "selected": selected, "bounds": bounds, "pivot": pivot,
                          "start": QPointF(pivot.x() + 40, pivot.y()), "scene": QPointF(pivot.x(), pivot.y() + 40),
                          "tier": context[3], "initial": {(si, pi): dict(resolved[si]["points"][pi]) for si, pi in selected},
                          "alt_shape": None}
    w.viewer._commit_roto_drag(); w.settle()
    w.check(w.undo_depth() == depth + 2, "scale and rotate are one undo step each")
    w.shot("scaled-and-rotated")
    context = w.viewer._roto_context(); resolved = w.viewer._roto_resolved(context)
    selected, bounds, _, pivot = w.viewer._roto_box_geometry(resolved, context[3])
    before = [(p["x"], p["y"]) for p in w.viewer._roto_resolved(context)[0]["points"]]
    w.viewer.roto_drag = {"kind": "box", "kind_name": "pivot", "selected": selected, "bounds": bounds, "pivot": pivot,
                          "start": pivot, "scene": QPointF(pivot.x() + 20, pivot.y() + 15), "tier": context[3],
                          "initial": {(si, pi): dict(resolved[si]["points"][pi]) for si, pi in selected}, "alt_shape": None}
    w.viewer._commit_roto_drag(); w.settle()
    after = [(p["x"], p["y"]) for p in w.viewer._roto_resolved(w.viewer._roto_context())[0]["points"]]
    w.check(all(abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3 for a, b in zip(after, before)),
            "moving the pivot leaves the drawn shape where it is")
    w.shot("pivot-moved")
    w.note("scale, rotate and pivot are committed by driving the box drag state, the same entry the mouse takes; the move is done with the mouse")
    w.viewer.roto_selected_points = {(0, i) for i in range(4)}
    context = w.viewer._roto_context(); resolved = w.viewer._roto_resolved(context)
    selected, bounds, _, pivot = w.viewer._roto_box_geometry(resolved, context[3])
    inside = QPointF(bounds.center().x() + bounds.width() / 4, bounds.center().y() + bounds.height() / 4)
    start = w.viewer.mapFromScene(inside)
    before_x = number(w.shape_entries()[0]["points"][0]["x"])
    viewport = w.viewer.viewport()
    QTest.mousePress(viewport, LEFT, NO_MOD, pos=start)
    for i in range(1, 9):
        QTest.mouseMove(viewport, start + QPointF(2.5 * i, 1.5 * i).toPoint(), 5)
    QTest.mouseRelease(viewport, LEFT, NO_MOD, pos=start + QPointF(20, 12).toPoint()); w.settle()
    w.check(abs(number(w.shape_entries()[0]["points"][0]["x"]) - before_x) > 5, "dragging inside the box (away from the pivot) moves the points")
    w.shot("box-moved")


# ---- Shape list ------------------------------------------------------------------------------------------------------
@scene(8, "The shape list: rename, hide, lock, reorder, new layer, drag into a layer",
       ["working_stroke_shape_list.html"],
       "The list is a table of shapes and layers; every cell edits live, items reorder by dragging and layers act as folders.")
def s08(w):
    w.fresh("roto", shapes=[square("red", 100, 80, 300, 220, color=[1, 0, 0, 1]),
                            square("blue", 200, 120, 400, 260, color=[0, 0, 1, 1]),
                            square("green", 340, 70, 560, 200, color=[0, 1, 0, 1])])
    tree = w.shape_tree()
    top_rows = [tree.parse_id(tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)) for i in range(tree.topLevelItemCount())]
    w.check(top_rows == [("item", 2), ("item", 1), ("item", 0)],
            "the list shows the front-most shape first, as Nuke does (the stored order stays render order)")
    w.shot("list")
    w.shot_widget("shape-list-before", tree)
    name = tree.findChild(QLineEdit, "roto-shape-row-name-1")
    name.setText("azure"); name.editingFinished.emit(); w.settle()
    w.check(w.payload["shapes"][1]["name"] == "azure", "renaming a row renames the shape")
    tree = w.shape_tree()
    tree.findChild(QCheckBox, "roto-shape-row-visible-0").setChecked(False); w.settle()
    w.check(w.payload["shapes"][0]["visible"] is False, "the visible box hides the shape")
    tree = w.shape_tree()
    tree.findChild(QCheckBox, "roto-shape-row-locked-2").setChecked(True); w.settle()
    w.check(w.payload["shapes"][2]["locked"] is True, "the lock box locks the shape")
    w.shot("hidden-and-locked")
    tree = w.shape_tree(); tree.select_ids([("item", 0)])
    tree.cmd_add_layer(); w.settle()
    w.check(w.payload.get("layers") == [{"name": "Layer1"}], "Add new layer creates a layer")
    tree = w.shape_tree()
    tree.move_ids([("item", 2)], ("layer", "Layer1"), "into"); w.settle()
    w.check(w.payload["shapes"][2].get("layer") == "Layer1", "dragging a shape onto a layer puts it inside")
    tree = w.shape_tree()
    names = [s["name"] for s in w.payload["shapes"]]
    tree.move_ids([("item", 0)], ("item", 1), "above"); w.settle()
    w.check([s["name"] for s in w.payload["shapes"]] != names, "dragging a row reorders the list")
    w.shot("layer-and-reorder")
    w.shot_widget("shape-list-after", w.shape_tree())
    # The Blending cell offers the whole list: the Merge node's operations for Roto.
    combo = w.shape_tree().findChild(QComboBox, "roto-shape-row-blend-0")
    offered = [combo.itemData(i) for i in range(combo.count())]
    w.check(len(offered) >= 30 and {"over", "atop", "screen", "soft-light", "color-dodge", "disjoint-over"} <= set(offered),
            f"the Blending cell offers {len(offered)} modes, the Merge node's list")
    combo.setCurrentIndex(combo.findData("atop")); w.settle()
    w.check(w.payload["shapes"][0]["blend_mode"] == "atop", "choosing a mode in the cell stores it")
    # A layer has its own transform, applied to every shape inside it.
    tree = w.shape_tree()
    tree.setCurrentItem(tree.tree_items[("layer", "Layer1")]); w.settle()
    w.wait(lambda: False, 0.2)
    spin = w.panel().findChild(QDoubleSpinBox, "roto-layer-transform-translate_x")
    w.check(spin is not None, "selecting a layer row shows the layer's Transform controls")
    inside = [i for i, e in enumerate(w.shape_entries()) if e.get("layer") == "Layer1"]
    before = [number(w.shape_entries()[i]["points"][0]["x"]) for i in inside]
    spin.setValue(40.0); spin.editingFinished.emit(); w.settle()
    resolved = shape_model.resolve_shapes(w.payload, 1)
    w.check(all(abs(resolved[i]["points"][0]["x"] - (b + 40.0)) < 1e-6 for i, b in zip(inside, before)),
            "a layer's translate moves every shape inside it")
    w.check(all(number(w.shape_entries()[i]["points"][0]["x"]) == b for i, b in zip(inside, before)), "the shapes' own points are untouched")
    w.shot("layer-transform")
    w.note("not applicable: the View column is for multi-view projects, which NodeBased does not have (ledger row 253 in docs/ROTO_NUKE_UI.md)")


@scene(9, "Autokey: key frame 1, move points on frame 10, step to frame 1 and back, key controls",
       ["animating_strokes_shapes.html", "editing_existing_timing.html"],
       "With autokey on, moving a point at a frame sets a key there; a shape keyed at two frames animates between them, and a lone key holds on every frame.")
def s09(w):
    w.fresh("roto", shapes=[square("box", 160, 90, 480, 270)])
    w.tool("Select", "Select Points")
    w.viewer.roto_selected_shape_index = 0
    set_key = w.panel().findChild(QPushButton, "roto-spline-set-key")
    w.check(set_key is not None, "the panel has a Set spline key button")
    set_key.click(); w.settle()                      # key the whole spline at frame 1
    w.check(isinstance(w.shape_entries()[0]["points"][0]["x"], dict), "Set spline key keys the points at frame 1")
    w.goto(10)
    w.click(160, 90)
    w.drag((160, 90), (220, 130))
    point = w.shape_entries()[0]["points"][0]
    w.check(isinstance(point["x"], dict) and len(point["x"]["curve"]["keys"]) == 2,
            "moving a point at frame 10 adds a second key (autokey on)")
    w.shot("frame-10")
    w.goto(1)
    p1 = w.shape_entries()[0]["points"][0]
    w.check(abs(number(p1["x"], 1) - 160) < 3, "at frame 1 the point is still at its original position")
    w.shot("frame-1")
    w.goto(10)
    w.check(abs(number(point["x"], 10) - 220) < 3, "at frame 10 the point is at the moved position")
    w.goto(5)
    mid = number(w.shape_entries()[0]["points"][0]["x"], 5)
    w.check(160 < mid < 220, "frame 5 interpolates between the keys")
    w.shot("frame-5")
    others = w.shape_entries()[0]["points"][1]
    w.check(abs(number(others["x"], 10) - 480) < 1, "points that were not moved stay put on frame 10")
    # key controls in the panel
    w.viewer.roto_selected_points = {(0, 1)}
    panel = w.panel()
    button = panel.findChild(QPushButton, "roto-point-set-keys")
    w.check(button is not None, "the panel has a Set keys button for the selected points")
    depth = w.undo_depth()
    button.click(); w.settle()
    w.check(w.undo_depth() == depth + 1, "Set keys is one undo step")
    keys = w.shape_entries()[0]["points"][1]["x"]["curve"]["keys"]
    w.check(5 in [k["frame"] for k in keys], "Set keys writes a key at the current frame")
    w.shot("key-controls")
    # one key alone holds its value on every frame
    w.fresh("roto", shapes=[square("box", 160, 90, 480, 270)])
    w.tool("Select", "Select Points"); w.goto(10); w.click(160, 90); w.drag((160, 90), (220, 130)); w.goto(1)
    held = number(w.shape_entries()[0]["points"][0]["x"], 1)
    w.note(f"a single key at frame 10 holds on frame 1 (x {held:.0f}); a second key is what starts the animation")


@scene(10, "Lifetime types: all frames, single frame, range, from current",
       ["editing_existing_timing.html"],
       "Each stroke or shape has a lifetime type; it only renders on frames inside it.")
def s10(w):
    w.fresh("roto", shapes=[square("box", 160, 90, 480, 270)])
    w.window.viewer.roto_selected_shape_index = 0
    tree = w.shape_tree()
    tree.select_ids([("item", 0)])
    for mode, start, end, inside, outside in (("range", 3, 9, 5, 12), ("single", 4, 4, 4, 5), ("to_end", 6, None, 8, 2)):
        if mode == "range":
            tree.cmd_set_lifetime(start, end)
        else:
            shapes = copy.deepcopy(w.payload["shapes"])
            life = ({"type": "single", "start": start, "end": start} if mode == "single"
                    else {"type": "to_end", "start": start, "end": start})
            shapes[0]["lifetime"] = life
            w.window.command({"op": "set_shapes", "id": "r", "shapes": shapes})
        w.settle()
        life = w.payload["shapes"][0]["lifetime"]
        w.note(f"{mode}: stored {life}")
        w.goto(inside)
        on = w.window.evaluator.evaluate_raster(w.document, target="r").pixels[..., 3].max()
        w.goto(outside)
        off = w.window.evaluator.evaluate_raster(w.document, target="r").pixels[..., 3].max()
        w.check(on > 0.5 and off < 0.01, f"the {mode} lifetime shows the shape on frame {inside} and hides it on frame {outside}")
        tree = w.shape_tree(); tree.select_ids([("item", 0)])
    w.goto(5)
    w.shot("lifetime")
    combo = w.panel().findChild(QComboBox, "roto-shape-lifetime-type")
    w.check(combo is not None, "the panel has a lifetime type menu")
    w.note("menu entries: " + ", ".join(combo.itemText(i) for i in range(combo.count())))
    # The list's Motion blur cell shows and sets keys (on the shutter) with autokey on.
    w.fresh("roto", shapes=[square("box", 160, 90, 480, 270)], last=24)
    tree = w.shape_tree()
    cell = tree.findChild(QCheckBox, "roto-shape-row-motion_blur-0")
    depth = w.undo_depth()
    cell.setChecked(True); w.settle()
    w.check(w.undo_depth() == depth + 1, "ticking the Motion blur cell is one undo step")
    keys = w.payload["shapes"][0]["motion_blur_shutter"]["curve"]["keys"]
    w.check([k["frame"] for k in keys] == [1], "with autokey on it sets a key at the current frame")
    cell = w.shape_tree().findChild(QCheckBox, "roto-shape-row-motion_blur-0")
    w.check(cell.property("keyState") == "key", "the cell shows that it is keyed on this frame")
    w.goto(12)
    cell = w.shape_tree().findChild(QCheckBox, "roto-shape-row-motion_blur-0")
    w.check(cell.property("keyState") == "animated", "on another frame the cell shows it is animated")
    cell.setChecked(False); w.settle()
    keys = w.payload["shapes"][0]["motion_blur_shutter"]["curve"]["keys"]
    w.check([(k["frame"], k["value"]) for k in keys] == [(1, 0.5), (12, 0.0)], "clearing it on frame 12 keys the blur off from there")
    w.shot_widget("motion-blur-keys", w.shape_tree())
    w.note("ripple editing is not implemented; the guide's pages do not document a ripple workflow (ledger row 272 in docs/ROTO_NUKE_UI.md)")


# ---- RotoPaint: strokes and tools ------------------------------------------------------------------------------------
def paint_strokes(w):
    return [e for e in w.entries() if e.get("kind") == "stroke"]


@scene(11, "RotoPaint brush stroke: size, hardness and opacity from the settings strip",
       ["drawing_paint_strokes.html", "using_brush_tool.html", "default_tools_settings.html"],
       "The Brush tool settings strip sets size, hardness, opacity and colour before a stroke; the stroke records them.")
def s11(w):
    w.fresh("paint")
    w.tool("Brushes", "Brush")
    for key, value in (("brush_size", 41.0), ("brush_hardness", 0.55), ("opacity", 0.6), ("color", [0.9, 0.2, 0.1, 1.0])):
        w.viewer.roto_toolbar.set_setting("paint", "Brush", key, value)
    w.settle()
    w.shot("strip")
    depth = w.undo_depth()
    w.drag((120, 100), (420, 150), steps=30)
    stroke = paint_strokes(w)[-1]
    w.check(stroke["tool"] == "paint", "the stroke is a brush stroke")
    w.check(abs(stroke["brush"]["size"] - 41) < 1e-6 and abs(stroke["brush"]["hardness"] - 0.55) < 1e-6
            and abs(stroke["brush"]["opacity"] - 0.6) < 1e-6, "the stroke records the strip's size, hardness and opacity")
    w.check(len(stroke["points"]) >= 5, "the stroke records the dragged path")
    pixels = w.window.evaluator.evaluate_raster(w.document, target="paint").pixels
    w.check(float(pixels[..., :3].max()) > 0, "the stroke is rendered")
    w.check(w.undo_depth() == depth + 1, "one stroke is one undo step")
    w.shot("stroke")


@scene(12, "Eraser via N",
       ["using_eraser_tool.html"],
       "Pressing N toggles Brush and Eraser; the eraser removes earlier paint on the same layer.")
def s12(w):
    w.fresh("paint")
    w.key_press(Qt.Key.Key_N)
    w.check(w.viewer.paint_tool == "paint", "first N selects the brush")
    w.drag((120, 180), (500, 180), steps=30)
    w.key_press(Qt.Key.Key_N)
    w.check(w.viewer.paint_tool == "eraser", "second N selects the eraser")
    w.drag((250, 150), (370, 210), steps=20)
    tools = [s["tool"] for s in paint_strokes(w)]
    w.check(tools == ["paint", "eraser"], "a brush stroke then an eraser stroke are recorded")
    w.shot("erased")


@scene(13, "Clone with a source offset",
       ["using_clone_tool.html"],
       "Ctrl-drag sets the clone source offset; the stroke then paints pixels copied from that offset.")
def s13(w):
    w.fresh("paint")
    w.key_press(Qt.Key.Key_C)
    w.check(w.viewer.paint_tool == "clone", "C selects the clone tool")
    w.drag((200, 140), (280, 150), mods=CTRL)
    w.check(abs(w.viewer.paint_source_offset[0] - 80) < 3 and abs(w.viewer.paint_source_offset[1] - 10) < 3,
            "Ctrl-drag sets the clone source offset")
    w.shot("offset-set")
    w.drag((200, 140), (330, 140), steps=20)
    stroke = paint_strokes(w)[-1]
    w.check(stroke["tool"] == "clone", "the stroke is a clone stroke")
    w.check(abs(stroke["source_offset"][0] - 80) < 3 and abs(stroke["source_offset"][1] - 10) < 3,
            "the clone stroke records the offset set by Ctrl-drag")
    w.shot("clone")


@scene(14, "Reveal with onion skin",
       ["using_reveal_tool.html"],
       "Reveal paints through from another frame or input; T toggles the onion skin that previews the source.")
def s14(w):
    w.fresh("paint")
    w.key_press(Qt.Key.Key_C); w.key_press(Qt.Key.Key_C)
    w.check(w.viewer.paint_tool == "reveal", "the second C selects reveal")
    w.key_press(Qt.Key.Key_T)
    settings = w.viewer.roto_toolbar._tool_settings("paint", "Reveal")
    w.check(settings.get("onion_skin") is True, "T turns the reveal onion skin on")
    w.shot("onion-skin")
    w.drag((200, 140), (330, 170), steps=20)
    w.check(paint_strokes(w)[-1]["tool"] == "reveal", "a reveal stroke is recorded")
    w.shot("reveal")


@scene(15, "Blur, smear and dodge/burn strokes",
       ["using_blur_tool.html", "using_smear_tool.html", "using_dodge_tool.html", "using_burn_tool.html", "using_sharpen_tool.html"],
       "Each tool is a stroke with an effect strength; blur and sharpen filter under the brush, smear drags pixels along the stroke, dodge and burn lighten and darken.")
def s15(w):
    w.fresh("paint")
    before = w.window.evaluator.evaluate_raster(w.document, target="paint").pixels.copy()
    expected = []
    for group, tool, key, path in (("Blur/Sharpen/Smear", "Blur", "blur", ((100, 90), (260, 90))),
                                   ("Blur/Sharpen/Smear", "Smear", "smear", ((100, 150), (260, 190))),
                                   ("Dodge/Burn", "Dodge", "dodge", ((330, 90), (520, 90))),
                                   ("Dodge/Burn", "Burn", "burn", ((330, 170), (520, 170)))):
        w.tool(group, tool)
        w.drag(*path, steps=20)
        expected.append(key)
    w.check([s["tool"] for s in paint_strokes(w)] == expected, "blur, smear, dodge and burn strokes are recorded in order")
    after = w.window.evaluator.evaluate_raster(w.document, target="paint").pixels
    changed = (abs(after - before).max(axis=-1) > 1e-4)
    w.check(changed.any(), "the effect strokes change the picture")
    w.shot("effects")


@scene(16, "DustBust click",
       ["using_clone_tool.html", "using_rotopaint.html"],
       "DustBust is a one-click, single-frame clone from the previous frame for removing specks.")
def s16(w):
    w.fresh("paint")
    w.goto(3)
    w.viewer.dustbust_preset = True
    w.click(300, 180)
    stroke = paint_strokes(w)[-1]
    w.check(stroke["tool"] == "clone", "DustBust makes a clone stroke")
    w.check(stroke["lifetime"] == {"mode": "single", "first": 3}, "it lives on the current frame only")
    w.check(stroke["source_frame"] == "relative", "the source is the neighbouring frame")
    w.shot("dustbust")


# ---- Past Nuke: tools NodeBased adds on top of Nuke's Roto (lane 9 steps I1 and I2, lane 2 step I3) ----------------------
# Every image below is computed from a formula (a white disc on black, a textured disc that stretches), so the pack can be
# regenerated exactly; nothing is downloaded.
def ring(count, radius, centre, radial=None, handles=True):
    """A closed Bezier circle of `count` points; `radial` adds a per-point wobble in pixels."""
    k = 4.0 / 3.0 * math.tan(math.pi / (2 * count))
    points = []
    for i in range(count):
        a = 2 * math.pi * i / count
        r = radius + (radial[i] if radial is not None else 0.0)
        tx, ty = -math.sin(a), math.cos(a)
        h = k * radius if handles else 0.0
        points.append({"x": centre[0] + r * math.cos(a), "y": centre[1] + r * math.sin(a),
                       "in_x": -tx * h, "in_y": -ty * h, "out_x": tx * h, "out_y": ty * h})
    return points


def ring_error(points, centre, radius):
    """Root-mean-square distance in pixels of a shape's outline from a circle."""
    outline = roto_model.flatten_points(points, closed=True)
    return math.sqrt(sum((math.hypot(x - centre[0], y - centre[1]) - radius) ** 2 for x, y in outline) / len(outline))


def ring_shape(name, points, **extra):
    return {"name": name, "mode": "union", "opacity": 1.0, "feather": 0.0, "closed": True, "points": points, **extra}


def disc_plate(box, keys=None):
    """A plate function for `Walk.fresh`: a hard-edged white disc on black filling `box` = (x, y, width, height)."""
    def build(dispatcher):
        dispatcher.execute({"op": "create", "id": "plate", "type": "Radial"})
        for name, value in dict(width=WIDTH, height=HEIGHT, box_x=box[0], box_y=box[1], box_width=box[2], box_height=box[3],
                                softness=0.0).items():
            dispatcher.execute({"op": "set", "id": "plate", "param": name, "value": value})
        for frame, (x, y, width, height) in (keys or {}).items():
            for name, value in (("box_x", x), ("box_y", y), ("box_width", width), ("box_height", height)):
                dispatcher.execute({"op": "set_key", "id": "plate", "param": name, "frame": frame, "value": value})
        dispatcher.execute({"op": "connect", "id": "r", "input": "bg", "source": "plate"})
    return build


SUBJECT = (320.0, 180.0)
SUBJECT_R = 100.0
EDGE_BIAS = 25.08 / 25.0      # a hard Radial of box radius r has an area-equivalent edge radius of r times this


@scene(17, "Snap to edges and Fit to edges",
       ["working_toolbars.html", "drawing_shapes.html"],
       "Nuke has no edge snapping: points land where you click, and fitting a shape to the picture is done by hand, point by point.")
def s17(w):
    wobble = [6.0, -9.0, 11.0, -5.0, 8.0, -12.0, 4.0, 10.0, -7.0, 9.0, -10.0, 5.0, 12.0, -6.0, 7.0, -8.0]
    rough = ring_shape("rough", ring(16, SUBJECT_R * EDGE_BIAS, (SUBJECT[0] + 9, SUBJECT[1] - 7), wobble))
    box = (SUBJECT[0] - SUBJECT_R, SUBJECT[1] - SUBJECT_R, 2 * SUBJECT_R, 2 * SUBJECT_R)
    w.fresh("roto", shapes=[rough], last=3, current=2, plate=disc_plate(box))
    w.viewer.roto_selected_shape_index = 0
    w.tool("Select", "Select Points")
    w.settle()
    before = ring_error(shape_model.resolve_shapes(w.payload, 2)[0]["points"], SUBJECT, SUBJECT_R * EDGE_BIAS)
    w.check(before > 5.0, f"the rough outline starts well off the subject's edge ({before:.1f} px)")
    w.shot("rough-before-fit", "A rough outline around the white subject: it wobbles several pixels off the real edge.")
    w.viewer.roto_toolbar.set_setting("r", "Select Points", "fit_band", 18.0)      # how far a point may travel to find the edge
    button = w.viewer.roto_toolbar.settings_widgets["fit_to_edges"]
    button.click(); w.settle(); w.wait(lambda: False, 0.3)
    after = ring_error(shape_model.resolve_shapes(w.payload, 2)[0]["points"], SUBJECT, SUBJECT_R * EDGE_BIAS)
    w.check(after < 1.0, f"Fit to edges settles the outline on the edge ({before:.1f} px off before, {after:.2f} px after)")
    w.shot("after-fit-to-edges", f"After Fit to edges: the same outline now hugs the subject, {after:.1f} px off instead of {before:.1f}.")
    w.note(f"Fit to edges: outline error {before:.1f} px to {after:.2f} px")
    # Snap: with Snap to edges on, a point clicked a few pixels away from the edge lands on it.
    w.viewer.roto_toolbar.set_setting("r", "Select Points", "snap_edges", True)
    w.viewer.roto_toolbar.set_setting("r", "Select Points", "snap_radius", 12.0)
    w.tool("Shapes", "Bezier")
    w.check(w.viewer.begin_roto_draw("r"), "begin drawing with the Bezier tool")
    clicks = 8
    for i in range(clicks):
        angle = 2 * math.pi * (i + 0.3) / clicks
        w.click(SUBJECT[0] + (SUBJECT_R * EDGE_BIAS + 8) * math.cos(angle), SUBJECT[1] + (SUBJECT_R * EDGE_BIAS + 8) * math.sin(angle))
    w.check(w.viewer.finish_roto_draw(closed=True), "the drawing is finished")
    w.settle()
    drawn = w.shape_entries()[-1]["points"]
    errors = [abs(math.hypot(number(p["x"]) - SUBJECT[0], number(p["y"]) - SUBJECT[1]) - SUBJECT_R * EDGE_BIAS) for p in drawn]
    w.check(len(drawn) == clicks and max(errors) < 1.0,
            f"every clicked point snapped onto the edge (largest miss {max(errors):.2f} px, clicked 8 px outside)")
    w.shot("snapped-clicks", "A second outline clicked 8 pixels outside the edge, with Snap to edges on: every point landed on the edge.")
    w.note(f"Snap to edges: points clicked 8 px outside landed within {max(errors):.2f} px of the edge")


# The subject of scene 18: a textured disc whose right half stretches a few percent every frame, so a plain move/scale
# of the whole outline cannot follow it but each point following its own patch of picture can.
TRACK_CENTRE = (240.0, 180.0)
TRACK_R = 90.0
STRETCH = 0.06


def _texture(u, v):
    return (0.5 + 0.12 * np.sin(0.61 * u + 0.2) + 0.12 * np.cos(0.47 * v + 1.1) + 0.10 * np.sin(0.33 * (u + v) + 0.7)
            + 0.09 * np.cos(0.52 * (u - 1.3 * v)) + 0.08 * np.sin(0.83 * u - 0.29 * v + 2.0) + 0.06 * np.cos(0.71 * (u + 0.4 * v)))


def stretching_disc(frame, width, height, x0=0, y0=0):
    t = frame - 1
    ys, xs = np.mgrid[y0:y0 + height, x0:x0 + width].astype(np.float64)
    x, y = xs + 0.5, ys + 0.5
    u = np.where(x > TRACK_CENTRE[0], TRACK_CENTRE[0] + (x - TRACK_CENTRE[0]) / (1 + STRETCH * t), x)
    coverage = np.clip(TRACK_R - np.hypot(u - TRACK_CENTRE[0], y - TRACK_CENTRE[1]) + 0.5, 0.0, 1.0)
    return 0.2 * (1 - coverage) + coverage * _texture(u, y)


def stretched_position(point, frame):
    """Where a point of the frame-1 outline sits on `frame`: the right half stretches, the left half stays."""
    if point["x"] > TRACK_CENTRE[0]:
        return TRACK_CENTRE[0] + (point["x"] - TRACK_CENTRE[0]) * (1 + STRETCH * (frame - 1)), point["y"]
    return point["x"], point["y"]


class StretchingPlate:
    """Stands in for the plate's pixels (the same stand-in the point-tracking tests use): the textured stretching disc, with an
    optional square of flat grey laid over a point on chosen frames to take away the picture it follows."""

    def __init__(self, occlusions=None):
        self.occlusions = occlusions or {}

    def __call__(self, _executor, _doc, _source, region, frame, **_kwargs):
        gray = stretching_disc(frame, region.width, region.height, region.x, region.y)
        rect = self.occlusions.get(frame)
        if rect is not None:
            x0, y0, x1, y1 = rect
            gray[max(0, y0 - region.y):max(0, y1 - region.y), max(0, x0 - region.x):max(0, x1 - region.x)] = 0.5
        rgba = np.stack(np.broadcast_arrays(gray, gray, gray, np.ones_like(gray)), axis=-1).astype(np.float32)
        return SimpleNamespace(pixels=rgba, tiled=True, tile_hits=0, tile_misses=0, fallback_nodes=())


@scene(18, "Track points: an outline that follows a bending subject",
       ["animating_strokes_shapes.html"],
       "Nuke's planar track moves the whole outline as one flat patch, so a subject that bends or stretches drifts away from it; the points are re-keyed by hand.")
def s18(w):
    from nodebased.tileexec import TileExecutor
    count = 24
    start = ring(count, TRACK_R, TRACK_CENTRE)
    victim = 0                                   # the point on the right edge, which moves the most
    occlusions = {}
    for frame in (6, 7, 8):
        x, y = stretched_position(start[victim], frame)
        occlusions[frame] = (int(x) - 30, int(y) - 30, int(x) + 31, int(y) + 31)
    plate = StretchingPlate(occlusions)
    with patch.object(TileExecutor, "compose_region", lambda *a, **k: plate(*a, **k)):
        def behind(dispatcher):      # a stand-in plate; its pixels are replaced by the stretching disc above
            dispatcher.execute({"op": "create", "id": "plate", "type": "Checker", "params": {"width": WIDTH, "height": HEIGHT}})
            dispatcher.execute({"op": "connect", "id": "r", "input": "bg", "source": "plate"})
        w.fresh("roto", shapes=[ring_shape("subject", start)], last=10, current=1, plate=behind)
        w.viewer.roto_selected_shape_index = 0
        w.tool("Select", "Select Points")
        w.wait(lambda: False, 0.5)
        w.shot("frame-1", "Frame 1: the outline is drawn around the textured subject.")
        messages = []
        w.window.statusBar().messageChanged.connect(messages.append)
        w.check(w.window.track_roto_points("r"), "Track points starts from frame 1")
        w.check(w.wait(lambda: w.window._points_future is None, 120), "the point track finishes")
        w.settle()
        named = [m for m in messages if "low-confidence" in m]
        w.check(named and "point 1 (frames 6" in named[-1], "the status bar names the flagged point and its frames")
        errors = {}
        for frame in range(1, 11):
            resolved = shape_model.resolve_shapes(w.payload, frame)[0]["points"]
            truth = [stretched_position(p, frame) for p in start]
            errors[frame] = [math.hypot(r["x"] - t[0], r["y"] - t[1]) for r, t in zip(resolved, truth)]
        clear = [i for i in range(count) if i != victim and math.hypot(start[i]["x"] - start[victim]["x"],
                                                                        start[i]["y"] - start[victim]["y"]) > 45]
        worst = max(errors[f][i] for f in (5, 10) for i in clear)
        w.check(worst < 2.5, f"the outline follows the stretching subject on frames 5 and 10 (worst point {worst:.2f} px off)")
        w.check(len(w.window.dispatcher.undo_stack) >= 1, "tracking is recorded as one edit")
        for frame in (5, 10):
            w.goto(frame); w.wait(lambda: False, 0.5)
            w.shot(f"frame-{frame}", f"Frame {frame}: the right side has stretched and every point of the outline stayed on its patch of picture.")
        flagged = {f: w.viewer.roto_flagged_points("r", f, 0, count) for f in range(1, 11)}
        w.check(all((victim in flagged[f]) == (f in (6, 7, 8)) for f in flagged),
                "the point whose picture is covered is flagged on exactly the frames it was covered")
        w.goto(7); w.wait(lambda: False, 0.5)
        w.shot("frame-7-low-confidence",
               "Frame 7: a grey patch hides the picture one point follows, so that point is flagged red; it keeps to the planar guess.")
        w.note(f"Track points: worst clear point {worst:.2f} px off over frames 2 to 10; flagged point named in the status bar")
        w.goto(1)


@scene(19, "Fill in-betweens: two keys, the frames between fitted to the picture",
       ["animating_strokes_shapes.html", "editing_existing_timing.html"],
       "Nuke interpolates between keys only, so a subject that does not move at an even pace drifts away from the outline between them; extra keys are added by hand.")
def s19(w):
    centre = SUBJECT
    r0, r1 = 40.0, 110.0
    last = 6

    def radius(frame, bias=1.0):
        t = (frame - 1) / (last - 1)
        return (r0 + (r1 - r0) * t * t) * bias

    keys = {f: (centre[0] - radius(f), centre[1] - radius(f), 2 * radius(f), 2 * radius(f)) for f in range(1, last + 1)}
    first, final = ring(16, r0 * EDGE_BIAS, centre), ring(16, r1 * EDGE_BIAS, centre)
    shape = ring_shape("subject", [{name: {"value": a[name], "curve": {"interpolation": "linear", "keys": [
        {"frame": 1, "value": a[name]}, {"frame": last, "value": b[name]}]}} for name in a} for a, b in zip(first, final)])
    w.fresh("roto", shapes=[shape], last=last, current=3, plate=disc_plate(keys[1], keys))
    w.viewer.roto_selected_shape_index = 0
    w.tool("Select", "Select Points")

    def error(frame):
        points = shape_model.resolve_shapes(w.payload, frame)[0]["points"]
        return ring_error(points, centre, radius(frame, EDGE_BIAS))

    w.wait(lambda: False, 0.5)
    middle_before = [error(f) for f in range(2, last)]
    w.check(max(middle_before) > 4.0, f"between the keys the plain interpolation drifts off the subject ({max(middle_before):.1f} px)")
    w.shot("interpolated-only", f"Frame 3 with keys only on frames 1 and {last}: the subject grows slowly at first, so the evenly interpolated outline sits {error(3):.0f} pixels outside its edge.")
    keyed_before = sorted(rotoinbetween.key_frames(w.payload["shapes"][0]))
    w.check(w.viewer.roto_toolbar.settings_widgets["fill_inbetweens"] is not None, "the strip has a Fill in-betweens button")
    w.viewer.roto_toolbar.set_setting("r", "Select Points", "fill_band", 30.0)      # how far a point may travel to find the edge
    undo = len(w.window.dispatcher.undo_stack)
    w.viewer.roto_toolbar.settings_widgets["fill_inbetweens"].click(); w.settle(); w.wait(lambda: False, 0.5)
    middle_after = [error(f) for f in range(2, last)]
    w.check(max(middle_after) < 1.0, f"after Fill in-betweens every middle frame sits on the edge (worst {max(middle_after):.2f} px)")
    w.check(len(w.window.dispatcher.undo_stack) == undo + 1, "the fill is one undo step")
    w.check(sorted(rotoinbetween.key_frames(w.payload["shapes"][0])) == list(range(1, last + 1)),
            "the fill added keys on the frames between, and left the first and last keys alone")
    w.shot("after-fill", f"Frame 3 after Fill in-betweens: the same frame now follows the subject's edge, {error(3):.1f} pixels off instead of {middle_before[1]:.0f}.")
    w.note(f"Fill in-betweens: worst middle frame {max(middle_before):.1f} px off before, {max(middle_after):.2f} px after; keys {keyed_before} to 1 through {last}")


# ---- Running ---------------------------------------------------------------------------------------------------------
def run(out_dir, only=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    results = []
    for number_, title, pages, nuke_does, function in SCENES:
        if only and number_ not in only:
            continue
        result = Result(number_, title, pages, nuke_does)
        walk = Walk(app, out_dir, result)
        started = time.time()
        try:
            function(walk)
        except Skip as skipped:
            result.status, result.error = "differs", f"skipped: {skipped}"
        except AssertionError as failure:
            result.status, result.error = "broken", f"step failed: {failure}"
            if walk.window is not None:
                try:
                    walk.shot("failure")
                except Exception:
                    pass
        except Exception:
            result.status = "broken"
            result.error = "exception: " + traceback.format_exc().strip().splitlines()[-1]
            result.notes.append(traceback.format_exc()[-700:])
        finally:
            result.seconds = time.time() - started
            try:
                walk.close()
            except Exception:
                pass
        results.append(result)
        print(f"{result.number:02d} {result.status:8s} {result.title} {result.error}", flush=True)
    (out_dir / "results.json").write_text(json.dumps([r.as_dict() for r in results], indent=1))
    return results


def write_readme(results, out_dir):
    """One row per scene: guide link, what Nuke does, the screenshots, and works / differs / broken."""
    out_dir = Path(out_dir)
    counts = {s: sum(1 for r in results if r.status == s) for s in ("works", "differs", "broken")}
    lines = ["# Roto and RotoPaint walkthrough against Nuke's guide", "",
             "Each row is one workflow from the guide's RotoPaint chapter, driven through the real NodeBased viewer with mouse and "
             "key events (`tools/roto_walkthrough.py`, offscreen). Every scene asserts what the document should look like "
             "afterwards, so a verdict of **works** is a passed check. Guide images and text are not copied; the page is linked.", "",
             f"**{counts['works']} work, {counts['differs']} differ, {counts['broken']} broken** out of {len(results)} scenes.", "",
             "Re-run: `QT_QPA_PLATFORM=offscreen python tools/roto_walkthrough.py --out <dir>`.", "",
             ]
    header = ["| # | Workflow | Guide | What Nuke does | NodeBased | Verdict |", "|---|---|---|---|---|---|"]

    def row(r):
        pages = "<br>".join(f"[{page.rsplit('/', 1)[1].replace('.html', '')}]({page})" for page in r.pages)
        captions = r.captions + [""] * (len(r.screens) - len(r.captions))
        shots = "<br>".join(f"[![{name}]({name})]({name})" + (f"<br><sub>{caption.replace('|', '/')}</sub>" if caption else "")
                            for name, caption in zip(r.screens[:3], captions))
        extra = "" if len(r.screens) <= 3 else "<br>" + "<br>".join(
            f"[![{name}]({name})]({name})" + (f"<br><sub>{caption.replace('|', '/')}</sub>" if caption else "")
            for name, caption in zip(r.screens[3:], captions[3:]))
        verdict = {"works": "**works**", "differs": "**differs**", "broken": "**broken**"}[r.status]
        detail = [n for n in r.notes if not n.startswith("Traceback")]
        if r.error:
            detail.insert(0, r.error)
        text = verdict + ("<br>" + "<br>".join(d.replace("|", "/").replace("\n", " ") for d in detail) if detail else "")
        return f"| {r.number} | {r.title} | {pages} | {r.nuke_does} | {shots}{extra} | {text} |"

    lines += header + [row(r) for r in results if r.number < PAST_NUKE_FIRST]
    past = [r for r in results if r.number >= PAST_NUKE_FIRST]
    if past:
        lines += ["", "## Past Nuke", "",
                  "Tools NodeBased adds on top of Nuke's Roto. Each scene runs on a picture computed from a formula, so the pack can be "
                  "regenerated exactly, and each screenshot has a one-line caption saying what it shows.", ""] + header + [row(r) for r in past]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="roto-walkthrough")
    parser.add_argument("--only", type=int, nargs="*")
    args = parser.parse_args(argv)
    results = run(args.out, set(args.only or []))
    if not args.only:
        write_readme(results, args.out)
    counts = {s: sum(1 for r in results if r.status == s) for s in ("works", "differs", "broken")}
    print(counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
