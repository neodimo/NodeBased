"""On-screen controls that work like Nuke: drive NodeBased's Viewer handles against Nuke's documented on-screen controls.

The reference is docs/parity/nuke_viewer_controls.json (Foundry's Nuke 17.0 Reference Guide node pages and the User Guide's
Transforming in 2D page, one entry per control). Every documented control has a probe here that builds a real Window
offscreen, presses, moves and releases the mouse at Viewer (scene) coordinates through the viewer's own hit_press /
hit_move / hit_release, and checks the knob values the document then stores, the single undo step and, where the knob is
animated, the key at the current frame. A probe returns (passed, detail). A documented control with no probe is "not
probed" and counts as not passing; a node with no NodeBased handle fails with that reason.

Run with QT_QPA_PLATFORM=offscreen. tests/test_viewer_controls.py asserts each result; tools/audit_nuke_fidelity.py prints
the per-node table ("On-screen controls that work like Nuke").
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "parity" / "nuke_viewer_controls.json"
PROBES = {}


def reference():
    return json.loads(REFERENCE.read_text())


def probe(control_id):
    def register(fn):
        PROBES[control_id] = fn
        return fn
    return register


class Bench:
    """One Window viewing a Transform over a Constant; each probe resets the Transform first."""

    def __init__(self):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import tests.isolation  # noqa: F401  (never touches the artist's real Qt settings)
        from tests.waiting import wait_until
        from PySide6.QtWidgets import QApplication
        from nodebased.app import Window
        from nodebased.core import Dispatcher, empty_document
        self.app = QApplication.instance() or QApplication([])
        document = empty_document()
        document["nodes"] = {}
        dispatcher = Dispatcher(document)
        dispatcher.execute({"op": "create", "id": "c", "type": "Constant", "pos": [0, 0]})
        dispatcher.execute({"op": "create", "id": "t", "type": "Transform", "pos": [150, 0]})
        dispatcher.execute({"op": "connect", "id": "t", "input": "image", "source": "c"})
        dispatcher.execute({"op": "create", "id": "crop", "type": "Crop", "pos": [150, 120]})
        dispatcher.execute({"op": "connect", "id": "crop", "input": "image", "source": "c"})
        dispatcher.execute({"op": "create", "id": "text", "type": "Text", "pos": [300, 120]})
        dispatcher.execute({"op": "create", "id": "ct", "type": "ColorTransfer", "pos": [300, 240]})
        dispatcher.execute({"op": "connect", "id": "ct", "input": "A", "source": "c"})
        dispatcher.execute({"op": "connect", "id": "ct", "input": "B", "source": "c"})
        dispatcher.execute({"op": "create", "id": "tr", "type": "Tracker", "pos": [450, 120]})
        dispatcher.execute({"op": "connect", "id": "tr", "input": "image", "source": "c"})
        dispatcher.execute({"op": "create", "id": "gw", "type": "GridWarp", "pos": [600, 120]})
        dispatcher.execute({"op": "connect", "id": "gw", "input": "image", "source": "c"})
        dispatcher.execute({"op": "view", "id": "t"})
        self.window = Window(dispatcher.document)
        self.window.resize(1400, 900)
        self.window.show()
        wait_until(lambda: self.window.frame is not None and self.window.viewer.format_rect is not None)
        self.window.graph.items_by_id["t"].setSelected(True)
        self.settle()
        self.base = dict(self.window.dispatcher.document["nodes"]["t"]["params"])

    def settle(self, rounds=4):
        for _ in range(rounds):
            self.app.processEvents()

    def reset(self, **params):
        """The Transform at its defaults (centre at 300, 250 so the jack sits inside the frame), no animation, no undo."""
        self.focus("t")
        document = self.window.dispatcher.document
        values = dict(self.base, center_x=300.0, center_y=250.0, **params)
        for param, value in values.items():
            if document["nodes"]["t"]["params"].get(param) != value:
                self.window.command({"op": "set", "id": "t", "param": param, "value": value})
        curves = (document.get("animation") or {}).get("curves", {}).get("t", {})
        for param in list(curves):
            self.window.command({"op": "clear_curve", "id": "t", "param": param})
        self.window.dispatcher.undo_stack.clear()
        self.settle()

    def params(self, key="t"):
        return self.window.dispatcher.document["nodes"][key]["params"]

    def focus(self, key, **params):
        """View and select `key` alone (its handles show), set `params`, clear its animation and the undo stack."""
        window = self.window
        if window.dispatcher.document.get("view") != key:
            window.command({"op": "view", "id": key})
        for item_key, item in window.graph.items_by_id.items():
            item.setSelected(item_key == key)
        for param, value in params.items():
            if window.dispatcher.document["nodes"][key]["params"].get(param) != value:
                window.command({"op": "set", "id": key, "param": param, "value": value})
        curves = (window.dispatcher.document.get("animation") or {}).get("curves", {}).get(key, {})
        for param in list(curves):
            window.command({"op": "clear_curve", "id": key, "param": param})
        window.dispatcher.undo_stack.clear()
        self.settle()

    def drag(self, start, end, modifiers=None):
        """Press at data point `start`, move to `end` in steps, release; modifiers held throughout."""
        from PySide6.QtCore import Qt
        viewer = self.window.viewer
        held = modifiers if modifiers is not None else Qt.KeyboardModifier.NoModifier
        point = viewer._transform_scene_point
        taken = viewer.hit_press(point(*start), Qt.MouseButton.LeftButton, held)
        for step in (0.25, 0.5, 0.75, 1.0):
            viewer.hit_move(point(start[0] + (end[0] - start[0]) * step, start[1] + (end[1] - start[1]) * step), held)
        viewer.hit_release(point(*end), Qt.MouseButton.LeftButton, held)
        self.settle()
        return taken

    def undo_steps(self):
        return len(self.window.dispatcher.undo_stack)

    def close(self):
        from PySide6.QtCore import QEvent
        self.settle(8)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settle()


def _missing(node):
    def run(bench):
        return False, f"NodeBased draws no on-screen control for {node}"
    return run


# -- Transform -----------------------------------------------------------------------------------------------------------

@probe("Transform.translate")
def _translate(bench):
    bench.reset()
    taken = bench.drag((300, 250), (340, 280))
    p = bench.params()
    moved = abs(abs(p["translate_x"]) - 40) < 0.5 and abs(abs(p["translate_y"]) - 30) < 0.5
    one = bench.undo_steps() == 1
    # animated: the drag keys translate_x at the current frame, still one undo step
    bench.reset()
    frame = int(bench.window.dispatcher.document["time"]["current"])
    bench.window.command({"op": "set_key", "id": "t", "param": "translate_x", "frame": frame, "value": 0.0})
    bench.window.command({"op": "set_key", "id": "t", "param": "translate_x", "frame": frame + 10, "value": 5.0})
    bench.window.dispatcher.undo_stack.clear()
    bench.drag((300, 250), (340, 250))
    curve = bench.window.node_curve("t", "translate_x") or {"keys": []}
    keyed = any(k["frame"] == frame and abs(abs(k["value"]) - 40) < 0.5 for k in curve["keys"])
    one_keyed = bench.undo_steps() == 1
    ok = bool(taken) and moved and one and keyed and one_keyed
    return ok, (f"translate {p['translate_x']:.1f}, {p['translate_y']:.1f}; undo steps ok {one}; "
                f"animated knob keyed at the frame {keyed} in one step {one_keyed}")


@probe("Transform.translate_shift")
def _translate_shift(bench):
    from PySide6.QtCore import Qt
    bench.reset()
    bench.drag((300, 250), (340, 262), Qt.KeyboardModifier.ShiftModifier)
    p = bench.params()
    ok = abs(abs(p["translate_x"]) - 40) < 0.5 and p["translate_y"] == 0 and bench.undo_steps() == 1
    return ok, f"Shift-drag 40, 12 stored translate {p['translate_x']:.1f}, {p['translate_y']:.1f}"


@probe("Transform.center_ctrl")
def _center_ctrl(bench):
    from PySide6.QtCore import Qt
    bench.reset()
    bench.drag((300, 250), (340, 280), Qt.KeyboardModifier.ControlModifier)
    p = bench.params()
    ok = (p["center_x"], p["center_y"]) != (300.0, 250.0) and abs(p["center_x"] - 340) < 0.5 and bench.undo_steps() == 1
    return ok, f"Ctrl-drag moved center to {p['center_x']:.1f}, {p['center_y']:.1f}"


@probe("Transform.scale_uniform")
def _scale_uniform(bench):
    from nodebased import handles2d
    bench.reset()
    viewer = bench.window.viewer
    w, h = viewer.format_rect.width(), viewer.format_rect.height()
    p = handles2d.flip_axes(bench.params(), handles2d.axes_flipped(bench.params()))
    corner = handles2d.box_corners(w, h, p["translate_x"], p["translate_y"], p["rotate"], p["scale"],
                                   p["center_x"], p["center_y"])[2]
    bench.drag(corner, (corner[0] + 60, corner[1] + 40))
    scale = bench.params()["scale"]
    ok = isinstance(scale, float) and scale > 1.0 and bench.undo_steps() == 1
    return ok, f"dragging the jack's outer handle stored scale {scale}"


@probe("Transform.rotate")
def _rotate(bench):
    from nodebased import handles2d
    import math
    bench.reset()
    viewer = bench.window.viewer
    p = handles2d.flip_axes(bench.params(), handles2d.axes_flipped(bench.params()))
    pivot = handles2d.pivot_point(p["translate_x"], p["translate_y"], p["center_x"], p["center_y"])
    corners = handles2d.box_corners(viewer.format_rect.width(), viewer.format_rect.height(), 0, 0, 0, 1,
                                    p["center_x"], p["center_y"])
    radius = 1.15 * math.hypot(corners[0][0] - pivot[0], corners[0][1] - pivot[1])
    bench.drag((pivot[0] + radius, pivot[1]), (pivot[0], pivot[1] + radius))
    rotate = bench.params()["rotate"]
    ok = abs(abs(rotate) - 90) < 1.0 and bench.undo_steps() == 1
    return ok, f"a quarter turn on the rotate handle stored rotate {rotate:.2f}"


def _jack(bench):
    from nodebased import handles2d
    values = handles2d.flip_axes(bench.params(), handles2d.axes_flipped(bench.params()))
    return bench.window.viewer._transform_overlay_geometry(values)


def _axis_scale(bench, axis):
    from nodebased import handles2d
    bench.reset()
    geometry = _jack(bench)
    point = handles2d.edge_midpoints(geometry["corners"])[1 if axis == 0 else 2]
    pivot = geometry["pivot"]
    # drag the point twice as far from the pivot along its own axis, and sideways a little (which must not count)
    end = (pivot[0] + 2 * (point[0] - pivot[0]) + (0 if axis == 0 else 15),
           pivot[1] + 2 * (point[1] - pivot[1]) + (15 if axis == 0 else 0))
    bench.drag(point, end)
    scale = bench.params()["scale"]
    ok = (isinstance(scale, list) and abs(scale[axis] - 2.0) < 0.02 and abs(scale[1 - axis] - 1.0) < 1e-9
          and bench.undo_steps() == 1)
    # a split scale keeps its handle: the corner then scales both factors together
    if ok:
        bench.window.dispatcher.undo_stack.clear()
        corner = _jack(bench)["corners"][2]
        bench.drag(corner, (pivot[0] + 1.5 * (corner[0] - pivot[0]), pivot[1] + 1.5 * (corner[1] - pivot[1])))
        both = bench.params()["scale"]
        ok = isinstance(both, list) and abs(both[axis] / both[1 - axis] - 2.0) < 0.05 and both[1 - axis] > 1.2
        scale = f"{scale}, then the corner made {both}"
    return ok, f"dragging the {'left/right' if axis == 0 else 'top/bottom'} point stored scale {scale}"


@probe("Transform.scale_x")
def _scale_x(bench):
    return _axis_scale(bench, 0)


@probe("Transform.scale_y")
def _scale_y(bench):
    return _axis_scale(bench, 1)


@probe("Transform.rotate_snap")
def _rotate_snap(bench):
    import math
    from PySide6.QtCore import Qt
    results = []
    for held in (None, Qt.KeyboardModifier.ShiftModifier):
        bench.reset()
        geometry = _jack(bench)
        pivot, radius = geometry["pivot"], geometry["ring"]
        angle = math.radians(88.5)      # a little short of a quarter turn
        bench.drag((pivot[0] + radius, pivot[1]), (pivot[0] + radius * math.cos(angle), pivot[1] + radius * math.sin(angle)), held)
        results.append(bench.params()["rotate"])
    snapped, free = results
    ok = abs(abs(snapped) - 90.0) < 1e-9 and abs(abs(free) - 88.5) < 0.3
    return ok, f"a drag to 88.5 degrees stored {snapped:.2f}; with Shift {free:.2f}"


@probe("Transform.skew")
def _skew(bench):
    bench.reset()
    geometry = _jack(bench)
    pivot, bar = geometry["pivot"], geometry["bar"]
    height = pivot[1] - bar[1]
    bench.drag(bar, (bar[0] + 0.25 * height, bar[1]))
    skew = bench.params()["skew_x"]
    shown = _jack(bench)["bar"]
    follows = abs(shown[0] - (bar[0] + 0.25 * height)) < 0.05 * height
    ok = abs(abs(skew) - 0.25) < 0.01 and bench.undo_steps() == 1 and follows
    return ok, f"dragging the bar's top a quarter of its height sideways stored skew x {skew:.3f}; the bar follows {follows}"


@probe("Transform.invert_handle")
def _invert_handle(bench):
    bench.reset()
    off = _jack(bench)["invert"]
    bench.reset(invert=1)
    on = _jack(bench)["invert"]
    bench.window.viewer.viewport().repaint()
    bench.reset()
    ok = off is None and on is not None
    return ok, f"second handle without invert {off}, with invert at {on}"
# -- Crop and Text: the on-screen box ------------------------------------------------------------------------------------------

def _keyed_drag(bench, key, param, start, end, expected):
    """Animate `param`, drag, and check the drag keyed the current frame in one undo step."""
    frame = int(bench.window.dispatcher.document["time"]["current"])
    value = bench.params(key)[param]
    bench.window.command({"op": "set_key", "id": key, "param": param, "frame": frame, "value": value})
    bench.window.command({"op": "set_key", "id": key, "param": param, "frame": frame + 10, "value": value})
    bench.window.dispatcher.undo_stack.clear()
    bench.drag(start, end)
    curve = bench.window.node_curve(key, param) or {"keys": []}
    keyed = any(k["frame"] == frame and abs(k["value"] - expected) < 0.5 for k in curve["keys"])
    return keyed and bench.undo_steps() == 1


@probe("Crop.box_edges")
def _crop_edges(bench):
    bench.focus("crop", x=100, y=80, width=600, height=300)
    right = bench.drag((700, 230), (750, 260))           # the right edge: width only, the left side stays put
    p = dict(bench.params("crop"))
    edge = bool(right) and (p["x"], p["y"], p["width"], p["height"]) == (100, 80, 650, 300) and bench.undo_steps() == 1
    bench.focus("crop", x=100, y=80, width=600, height=300)
    bench.drag((100, 80), (130, 100))                     # the top-left corner: the opposite corner stays put
    q = dict(bench.params("crop"))
    corner = (q["x"], q["y"], q["width"], q["height"]) == (130, 100, 570, 280)
    bench.focus("crop", x=100, y=80, width=600, height=300)
    inside = bench.drag((400, 230), (450, 260))           # inside the box is not a control (only the edges are)
    still = not inside and bench.params("crop")["x"] == 100
    bench.focus("crop", x=100, y=80, width=600, height=300)
    keyed = _keyed_drag(bench, "crop", "width", (700, 230), (740, 230), 640)
    bench.focus("crop", x=0, y=0, width=960, height=540)
    ok = edge and corner and still and keyed
    return ok, (f"right edge stored {p['x']}, {p['y']}, {p['width']}, {p['height']}; corner {q['x']}, {q['y']}, "
                f"{q['width']}, {q['height']}; inside left alone {still}; animated width keyed {keyed}")


@probe("Text.box_move_resize")
def _text_box(bench):
    bench.focus("text", box_x=100.0, box_y=80.0, box_width=400.0, box_height=200.0)
    bench.drag((300, 180), (340, 200))                    # inside: the box moves
    p = dict(bench.params("text"))
    moved = (p["box_x"], p["box_y"], p["box_width"], p["box_height"]) == (140.0, 100.0, 400.0, 200.0)
    one = bench.undo_steps() == 1
    bench.focus("text", box_x=100.0, box_y=80.0, box_width=400.0, box_height=200.0)
    bench.drag((500, 280), (560, 300))                    # the bottom-right corner: resized
    q = dict(bench.params("text"))
    resized = (q["box_x"], q["box_y"], q["box_width"], q["box_height"]) == (100.0, 80.0, 460.0, 220.0)
    bench.focus("text", box_x=100.0, box_y=80.0, box_width=400.0, box_height=200.0)
    keyed = _keyed_drag(bench, "text", "box_x", (300, 180), (330, 180), 130.0)
    bench.focus("text", box_x=40.0, box_y=40.0, box_width=880.0, box_height=460.0)
    ok = moved and one and resized and keyed
    return ok, (f"drag inside moved the box to {p['box_x']}, {p['box_y']}; corner resized it to {q['box_width']} by "
                f"{q['box_height']}; animated box keyed {keyed}")


@probe("Text.center_ctrl")
def _text_center(bench):
    from PySide6.QtCore import Qt
    bench.focus("text", box_x=100.0, box_y=80.0, box_width=400.0, box_height=200.0, center_x=0.0, center_y=0.0)
    bench.drag((300, 180), (330, 160), Qt.KeyboardModifier.ControlModifier)
    p = dict(bench.params("text"))
    ok = (p["center_x"], p["center_y"]) == (30.0, -20.0) and p["box_x"] == 100.0 and bench.undo_steps() == 1
    bench.focus("text", box_x=40.0, box_y=40.0, box_width=880.0, box_height=460.0, center_x=0.0, center_y=0.0)
    return ok, f"Ctrl-drag moved the centre to {p['center_x']}, {p['center_y']} and left the box at {p['box_x']}"


# -- ColorTransfer: the ROI box (lane 10, step T8) -------------------------------------------------------------------------------

def _roi(bench, key="ct"):
    p = bench.params(key)
    return tuple(float(p[f"roi_{k}"]) for k in "xyrt")


@probe("ColorTransfer.roi_box")
def _colortransfer_roi(bench):
    height = bench.window.viewer.format_rect.height()
    roi = dict(roi_x=100.0, roi_y=100.0, roi_r=500.0, roi_t=400.0)        # drawn rows down: x 100, y height-400, 400 by 300
    top = height - 400.0
    bench.focus("ct", **roi)
    right = bench.drag((500, top + 150), (550, top + 170))                # the right edge: r only
    edge = bool(right) and _roi(bench) == (100.0, 100.0, 550.0, 400.0) and bench.undo_steps() == 1
    bench.focus("ct", **roi)
    bench.drag((100, top), (130, top + 20))                               # the top-left corner: x and t, the bottom right stays
    corner = _roi(bench) == (130.0, 100.0, 500.0, 380.0)
    bench.focus("ct", **roi)
    bench.drag((300, top + 150), (340, top + 130))                        # inside: the whole box moves (40 right, 20 up)
    body = _roi(bench) == (140.0, 120.0, 540.0, 420.0) and bench.undo_steps() == 1
    bench.focus("ct", roi_x=0.0, roi_y=0.0, roi_r=0.0, roi_t=0.0)         # the default: the whole of A, its left edge at 0
    bench.drag((0, height / 2), (50, height / 2))
    width = bench.window.viewer.format_rect.width()
    default = _roi(bench) == (50.0, 0.0, float(width), float(height))
    bench.focus("ct", **roi)
    keyed = _keyed_drag(bench, "ct", "roi_r", (500, top + 150), (540, top + 150), 540.0)
    bench.focus("ct", roi_x=0.0, roi_y=0.0, roi_r=0.0, roi_t=0.0)
    ok = edge and corner and body and default and keyed
    return ok, (f"right edge {edge}, corner {corner}, body move {body}, the default box covers the frame {default}, "
                f"animated r keyed in one step {keyed}")


@probe("ColorTransfer.roi_on_input_a")
def _colortransfer_on_a(bench):
    window = bench.window
    bench.focus("ct", roi_x=100.0, roi_y=100.0, roi_r=500.0, roi_t=400.0)
    window.command({"op": "view", "id": "c"})                             # the Viewer on input A, ColorTransfer still selected
    for item_key, item in window.graph.items_by_id.items():
        item.setSelected(item_key == "ct")
    window.dispatcher.undo_stack.clear()
    bench.settle()
    top = window.viewer.format_rect.height() - 400.0
    taken = bench.drag((500, top + 150), (560, top + 150))
    ok = bool(taken) and _roi(bench) == (100.0, 100.0, 560.0, 400.0) and bench.undo_steps() == 1
    bench.focus("ct", roi_x=0.0, roi_y=0.0, roi_r=0.0, roi_t=0.0)
    return ok, f"with the Viewer on input A the right edge drag stored roi {_roi(bench) if not ok else (100, 100, 560, 400)}"


# -- Tracker and GridWarp: controls NodeBased already drew, now measured ------------------------------------------------------

@probe("Tracker.anchor")
def _tracker_anchor(bench):
    window = bench.window
    bench.focus("tr")
    before = len((window.dispatcher.document.get("node_data", {}).get("tr") or {}).get("tracks", []))
    added = window.add_track_at_centre("tr")
    bench.settle()
    tracks = window.dispatcher.document["node_data"]["tr"]["tracks"]
    track = tracks[-1]
    one = bench.undo_steps() == 1
    drawn = window.viewer.window._tracker_context() is not None
    x, y = float(track["x"]), float(track["y"])
    window.dispatcher.undo_stack.clear()
    bench.drag((x, y), (x + 30, y + 20))                 # the new anchor is a handle: dragging it moves the track
    from nodebased import shapes as shape_model
    frame = int(window.dispatcher.document["time"]["current"])
    moved = shape_model.resolve_tracks(window.dispatcher.document["node_data"]["tr"], frame)[-1]
    follows = abs(moved["x"] - (x + 30)) < 0.5 and abs(moved["y"] - (y + 20)) < 0.5
    ok = bool(added) and len(tracks) == before + 1 and one and drawn and follows
    return ok, (f"add track placed track {len(tracks)} at {x:.0f}, {y:.0f} in one step {one}; dragging its anchor moved it to "
                f"{moved['x']:.1f}, {moved['y']:.1f}")


def _grid(bench):
    context = bench.window.viewer._warp_context()
    from nodebased import shapes as shape_model
    frame = int(bench.window.dispatcher.document["time"]["current"])
    return shape_model.resolve_warp_data("GridWarp", context[2], frame)


@probe("GridWarp.grid_handles")
def _gridwarp_handles(bench):
    window = bench.window
    bench.focus("gw")
    grid = _grid(bench)
    point = grid["destination"][2][2]
    # source and destination share their points at first (which one a drag takes there is undocumented; NodeBased takes the
    # source); one drag moves one grid's point and leaves the other's where it was
    bench.drag((point[0], point[1]), (point[0] + 25, point[1] - 15))
    after = _grid(bench)
    moved_side = [side for side in ("source", "destination")
                  if abs(after[side][2][2][0] - (point[0] + 25)) < 0.5 and abs(after[side][2][2][1] - (point[1] - 15)) < 0.5]
    other = "destination" if moved_side == ["source"] else "source"
    moved = after[moved_side[0]][2][2] if moved_side else after["destination"][2][2]
    ok = len(moved_side) == 1 and after[other][2][2][:2] == point[:2] and bench.undo_steps() == 1
    window.command({"op": "undo"})
    bench.settle()
    return ok, (f"the grid's lines and handles are drawn whenever the GridWarp is viewed and selected (Nuke's visible switch "
                f"has no NodeBased counterpart); dragging a grid point moved the {moved_side} point to {moved[0]:.1f}, {moved[1]:.1f}")


@probe("GridWarp.grid_transform")
def _gridwarp_transform(bench):
    import math
    names = ("center_x", "center_y", "translate_x", "translate_y", "rotate", "scale")
    reset = {f"source_grid_transform_{n}": v for n, v in zip(names, (300.0, 200.0, 0.0, 0.0, 0.0, 1.0))}
    bench.focus("gw", **reset)
    viewer = bench.window.viewer
    p = lambda name: bench.params("gw")["source_grid_transform_" + name]
    pivot = viewer._grid_jack()[3]["pivot"]
    bench.drag(pivot, (pivot[0] + 40, pivot[1] - 30))         # the centre moves the whole grid
    moved = (abs(p("translate_x") - 40) < 0.5 and abs(p("translate_y") - 30) < 0.5 and bench.undo_steps() == 1)
    t = (p("translate_x"), p("translate_y"))
    bench.focus("gw", **reset)
    end = viewer._grid_jack()[3]["x"]
    length = end[0] - pivot[0]
    bench.drag(end, (pivot[0], pivot[1] - 2 * length))          # the x axis turned a quarter up and stretched twice as long
    turned = abs(p("rotate") - 90) < 0.5 and abs(p("scale") - 2) < 0.02 and bench.undo_steps() == 1
    r = (p("rotate"), p("scale"))
    bench.focus("gw", **reset)
    frame = int(bench.window.dispatcher.document["time"]["current"])
    for f in (frame, frame + 10):
        bench.window.command({"op": "set_key", "id": "gw", "param": "source_grid_transform_rotate", "frame": f, "value": 0.0})
    bench.window.dispatcher.undo_stack.clear()
    end = viewer._grid_jack()[3]["x"]
    bench.drag(end, (pivot[0] + length * math.cos(math.radians(-30)), pivot[1] + length * math.sin(math.radians(-30))))
    curve = bench.window.node_curve("gw", "source_grid_transform_rotate") or {"keys": []}
    keyed = any(k["frame"] == frame and abs(k["value"] - 30) < 0.5 for k in curve["keys"]) and bench.undo_steps() == 1
    bench.focus("gw", **{f"source_grid_transform_{n}": v for n, v in zip(names, (0.0, 0.0, 0.0, 0.0, 0.0, 1.0))})
    ok = moved and turned and keyed
    return ok, (f"dragging the overlay's centre stored translate {t[0]:.1f}, {t[1]:.1f}; turning its x axis stored rotate "
                f"{r[0]:.1f}, scale {r[1]:.3f}; an animated rotate keyed {keyed}")


def probe_all():
    """{control id: {"result": "pass"/"fail"/"not probed"/"undocumented", "detail": str}} for every reference entry."""
    entries = reference()["controls"]
    results, bench = {}, None
    try:
        for entry in entries:
            if entry["status"] != "documented":
                results[entry["id"]] = {"result": "undocumented", "detail": entry["nuke"]}
                continue
            fn = PROBES.get(entry["id"])
            if fn is None:
                results[entry["id"]] = {"result": "not probed", "detail": "no probe yet"}
                continue
            if bench is None:
                bench = Bench()
            try:
                passed, detail = fn(bench)
            except Exception as exc:  # a probe that crashes is a failure with its reason, never a silent pass
                passed, detail = False, f"{type(exc).__name__}: {exc}"
            results[entry["id"]] = {"result": "pass" if passed else "fail", "detail": detail}
    finally:
        if bench is not None:
            bench.close()
    return results


def table(results):
    """Rows (node, passing, documented, failing ids, undocumented ids) in the reference's node order."""
    data = reference()
    rows = []
    for node in data["node_order"]:
        ids = [e["id"] for e in data["controls"] if e["node"] == node]
        documented = [i for i in ids if results[i]["result"] != "undocumented"]
        rows.append((node, sum(results[i]["result"] == "pass" for i in documented), len(documented),
                     [i for i in documented if results[i]["result"] != "pass"],
                     [i for i in ids if results[i]["result"] == "undocumented"]))
    return rows


if __name__ == "__main__":
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    out = probe_all()
    for node, ok, total, failing, undoc in table(out):
        print(f"{node}: {ok} / {total}; failing {', '.join(failing) or '-'}; undocumented {len(undoc)}")
