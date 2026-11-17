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

    def params(self):
        return self.window.dispatcher.document["nodes"]["t"]["params"]

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


PROBES["Transform.scale_x"] = lambda bench: (False, "the jack has no left/right points that scale x alone (scale_x)")
PROBES["Transform.scale_y"] = lambda bench: (False, "the jack has no top/bottom points that scale y alone (scale_y)")
PROBES["Transform.rotate_snap"] = lambda bench: (False, "rotating does not snap to typical values")
PROBES["Transform.skew"] = lambda bench: (False, "the jack has no skew bar")
PROBES["Transform.invert_handle"] = lambda bench: (False, "no second handle is drawn when invert is on")
PROBES["Crop.box_edges"] = _missing("Crop")
PROBES["Text.box_move_resize"] = _missing("Text")
PROBES["Text.center_ctrl"] = _missing("Text")


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
