"""Panels that work like Nuke: drive NodeBased's real Properties-panel widgets against Nuke's documented knob behaviours.

The reference is docs/parity/nuke_knob_interactions.json (Foundry's Nuke 17.0 User Guide, one entry per behaviour). Every
documented behaviour has a probe here that builds a real panel in a real Window, sends Qt key and mouse events to the widget
an artist would touch, and checks what the document then stores (and, where it matters, the undo step). A probe returns
(passed, detail). A documented behaviour with no probe yet is reported as "not probed" and counts as not passing.

Run with QT_QPA_PLATFORM=offscreen. tests/test_panel_interactions.py asserts each result; tools/audit_nuke_fidelity.py prints
the per-widget table ("Panels that work like Nuke").
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "parity" / "nuke_knob_interactions.json"
PROBES = {}


def reference():
    return json.loads(REFERENCE.read_text())


def probe(behaviour_id):
    def register(fn):
        PROBES[behaviour_id] = fn
        return fn
    return register


# -- harness ---------------------------------------------------------------------------------------------------------------

class Bench:
    """One Window shared by every probe; each probe makes its own node."""

    def __init__(self):
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import tests.isolation  # noqa: F401  (the probe window never reads or writes the artist's real Qt settings)
        from PySide6.QtWidgets import QApplication
        from nodebased.app import Window
        self.app = QApplication.instance() or QApplication([])
        self.window = Window()
        self.window.resize(1400, 900)
        self.window.show()
        self.count = 0
        self.settle()

    def settle(self, rounds=4):
        for _ in range(rounds):
            self.app.processEvents()

    def node(self, kind, params=None):
        self.count += 1
        key = f"interaction_{self.count}"
        command = {"op": "create", "id": key, "type": kind, "pos": [self.count * 40, 0]}
        if params:
            command["params"] = params
        self.window.dispatcher.execute(command)
        self.show(key)
        return key

    def show(self, key):
        panel = self.window.build_node_panel(key)
        self.window.set_properties_widget(panel)
        self.settle()
        return panel

    def params(self, key):
        return self.window.dispatcher.document["nodes"][key]["params"]

    def child(self, name, cls=None):
        from PySide6.QtWidgets import QWidget
        found = [w for w in self.window.properties_dock.findChildren(cls or QWidget, name)
                 if w.window() is self.window and not w.isHidden()]   # a rebuilt panel's old widgets wait for deletion
        return found[-1] if found else None

    def field(self, param):
        from PySide6.QtWidgets import QAbstractSpinBox
        return self.child(f"{param}-field", QAbstractSpinBox)

    def undo_count(self):
        dispatcher = self.window.dispatcher
        for name in ("undo_stack", "history", "_undo"):
            stack = getattr(dispatcher, name, None)
            if isinstance(stack, list):
                return len(stack)
        return None

    def close(self):
        from PySide6.QtCore import QEvent
        self.settle(8)                   # let pending previews land before the window's executor shuts down
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settle()


def _mouse(widget, kind, pos, button, buttons, modifiers):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    local = QPointF(*pos)
    event = QMouseEvent(kind, local, widget.mapToGlobal(local), button, buttons, modifiers)
    QApplication.sendEvent(widget, event)


def _scrub(bench, field, button, modifiers, dx):
    from PySide6.QtCore import QEvent, Qt
    edit = field.lineEdit()
    y = edit.height() / 2
    _mouse(edit, QEvent.Type.MouseButtonPress, (10, y), button, button, modifiers)
    _mouse(edit, QEvent.Type.MouseMove, (10 + dx, y), Qt.MouseButton.NoButton, button, modifiers)
    _mouse(edit, QEvent.Type.MouseButtonRelease, (10 + dx, y), button, Qt.MouseButton.NoButton, modifiers)
    bench.settle()


def _type(bench, field, text):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    field.setFocus()
    field.lineEdit().selectAll()
    QTest.keyClicks(field.lineEdit(), text)
    QTest.keyClick(field.lineEdit(), Qt.Key.Key_Return)
    bench.settle()


# -- number knob -----------------------------------------------------------------------------------------------------------

@probe("number.type_value")
def _type_value(bench):
    key = bench.node("Grade")
    before = bench.undo_count()
    _type(bench, bench.field("gamma"), "1.25")
    stored = bench.params(key)["gamma"]
    steps = None if before is None else bench.undo_count() - before
    ok = abs(float(stored) - 1.25) < 1e-9 and steps in (None, 1)
    return ok, f"typed 1.25, stored {stored!r}, undo steps {steps}"


@probe("number.type_formula")
def _type_formula(bench):
    key = bench.node("Grade")
    _type(bench, bench.field("gamma"), "3/2")
    stored = bench.params(key)["gamma"]
    return abs(float(stored) - 1.5) < 1e-9, f"typed 3/2, stored {stored!r}"


@probe("number.equals_expression")
def _equals_expression(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit
    key = bench.node("Grade")
    field = bench.field("gamma")
    field.setFocus()
    QTest.keyClick(field, Qt.Key.Key_Equal)
    bench.settle()
    editor = bench.child("expression-editor", QLineEdit)
    if editor is None:
        return False, "= opened no expression editor"
    editor.setFocus()
    QTest.keyClicks(editor, "1+frame*0")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    bench.settle()
    expressions = (bench.window.dispatcher.document.get("expressions") or {}).get(key, {})
    bench.show(key)
    field = bench.field("gamma")
    driven = field is not None and not field.isEnabled()
    return "gamma" in expressions and driven, f"expression stored {'gamma' in expressions}, field shows driven {driven}"


def _arrow(bench, cursor, key_name):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    key = bench.node("Grade")
    field = bench.field("gamma")          # shows 1.000
    field.setFocus()
    field.lineEdit().deselect()
    field.lineEdit().setCursorPosition(cursor)
    QTest.keyClick(field.lineEdit(), getattr(Qt.Key, key_name))
    bench.settle()
    return float(bench.params(key)["gamma"])


@probe("number.arrow_digit")
def _arrow_digit(bench):
    ones = _arrow(bench, 0, "Key_Up")         # cursor before the 1
    tenths = _arrow(bench, 2, "Key_Up")       # cursor before the first decimal
    hundredths_down = _arrow(bench, 3, "Key_Down")
    ok = abs(ones - 2.0) < 1e-9 and abs(tenths - 1.1) < 1e-9 and abs(hundredths_down - 0.99) < 1e-9
    return ok, f"before 1: {ones:g}, before tenths: {tenths:g}, Down before hundredths: {hundredths_down:g}"


@probe("number.wheel_steps")
def _wheel(bench):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    key = bench.node("Grade")
    field = bench.field("gamma")
    field.setFocus()
    bench.settle()
    centre = QPointF(field.width() / 2, field.height() / 2)
    event = QWheelEvent(centre, field.mapToGlobal(centre), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(field, event)
    bench.settle()
    stored = float(bench.params(key)["gamma"])
    return stored > 1.0, f"one notch up: {stored:g}"


@probe("number.alt_drag")
def _alt_drag(bench):
    from PySide6.QtCore import Qt
    key = bench.node("Grade")
    _scrub(bench, bench.field("gamma"), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, 30)
    stored = float(bench.params(key)["gamma"])
    return stored > 1.0, f"Alt drag 30 px right: {stored:g}"


@probe("number.virtual_slider")
def _virtual_slider(bench):
    from PySide6.QtCore import Qt
    key = bench.node("Grade")
    _scrub(bench, bench.field("gamma"), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, 20)
    plain = float(bench.params(key)["gamma"]) - 1.0
    key = bench.node("Grade")
    _scrub(bench, bench.field("gamma"), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.ShiftModifier, 20)
    fast = float(bench.params(key)["gamma"]) - 1.0
    key = bench.node("Grade")
    _scrub(bench, bench.field("gamma"), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.AltModifier, 20)
    slow = float(bench.params(key)["gamma"]) - 1.0
    ok = plain > 0 and fast > plain and 0 < slow < plain
    return ok, f"middle drag 20 px: +{plain:g}, Shift +{fast:g}, Alt +{slow:g}"


@probe("number.slider_drag")
def _slider_drag(bench):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from nodebased.app import FloatSliderControl
    key = bench.node("Grade")
    pair = bench.field("gamma").parentWidget()
    if not isinstance(pair, FloatSliderControl):
        return False, "gamma has no slider"
    slider = pair.slider
    y = slider.height() // 2
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(slider.width() - 4, y))
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(slider.width() - 4, y))
    bench.settle()
    stored = float(bench.params(key)["gamma"])
    return stored > 1.0, f"pressed the slider's right end: {stored:g}"


# -- colour knob -----------------------------------------------------------------------------------------------------------

def _open_swatch(bench):
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    key = bench.node("Constant")
    button = bench.child("color-picker-button") or bench.child("color-swatch")
    if button is None:
        return key, None
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    bench.settle()
    from nodebased.colorpicker import LinearColorPicker
    pickers = [w for w in bench.window.findChildren(LinearColorPicker) if w.isVisible()]
    return key, pickers[-1] if pickers else None


def _close_pickers(bench):
    from nodebased.colorpicker import LinearColorPicker
    for picker in bench.window.findChildren(LinearColorPicker):
        picker.reject()
    bench.window.viewer.color_sampling = None
    bench.settle()


def _click_swatch(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.mouseClick(bench.child("color-swatch"), Qt.MouseButton.LeftButton)
    bench.settle()


def _armed(bench):
    return getattr(bench.window.viewer, "knob_sampling", None) is not None


@probe("color.swatch_eyedropper")
def _swatch_eyedropper(bench):
    from nodebased.colorpicker import LinearColorPicker
    bench.node("Constant")
    _click_swatch(bench)
    on = _armed(bench)
    picker = any(w.isVisible() for w in bench.window.findChildren(LinearColorPicker))
    _click_swatch(bench)
    off = not _armed(bench)
    _close_pickers(bench)
    ok = on and off and not picker
    return ok, f"swatch click: eye dropper on {on}, picker opened {picker}; second click turns it off {off}"


def _ramp_and_target(bench):
    """View a 64 x 32 horizontal 0 to 1 ramp, open a Constant's panel and arm its eye dropper. Returns the Constant and
    the ramp's float frame as the viewer holds it."""
    import time
    import numpy as np
    target = bench.node("Constant")
    ramp = bench.node("Ramp", {"width": 64, "height": 32, "p1_x": 63.0})
    bench.window.command({"op": "view", "id": ramp})
    frame = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        bench.settle(1)
        shown = bench.window._float_frame_on_demand()
        if shown is not None and getattr(shown, "shape", (0, 0))[:2] == (32, 64):
            frame = np.array(shown, copy=True)
            break
        time.sleep(0.01)
    bench.show(target)
    _click_swatch(bench)
    return target, frame


def _viewer_press(bench, x, y, modifiers, release_at=None):
    from PySide6.QtCore import QPointF, Qt
    viewer = bench.window.viewer
    scale = max(float(viewer.last_scale or 1.0), 1.0)
    viewer.hit_press(QPointF(x * scale, y * scale), Qt.MouseButton.LeftButton, modifiers)
    if release_at is not None:
        end = QPointF(release_at[0] * scale, release_at[1] * scale)
        viewer.hit_move(end, modifiers)
        viewer.hit_release(end, Qt.MouseButton.LeftButton, modifiers)
    bench.settle()


@probe("color.ctrl_click_sample")
def _ctrl_click(bench):
    from PySide6.QtCore import Qt
    target, frame = _ramp_and_target(bench)
    if frame is None:
        return False, "the ramp never reached the viewer"
    if not _armed(bench):
        return False, "the swatch did not arm an eye dropper"
    before = float(bench.params(target)["red"])
    _viewer_press(bench, 32.5, 10.5, Qt.KeyboardModifier.NoModifier)
    plain = float(bench.params(target)["red"])
    _viewer_press(bench, 32.5, 10.5, Qt.KeyboardModifier.ControlModifier)
    sampled = float(bench.params(target)["red"])
    expected = float(frame[10, 32, 0])
    bench.window.viewer.knob_sampling = None
    ok = plain == before and expected is not None and abs(sampled - expected) < 1e-6
    return ok, f"plain click left red at {plain:g}; Ctrl+click sampled {sampled:g} (pixel {expected})"


@probe("color.region_average")
def _region(bench):
    from PySide6.QtCore import Qt
    target, frame = _ramp_and_target(bench)
    if frame is None:
        return False, "the ramp never reached the viewer"
    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    expected = float(frame[:, :, 0].mean())
    _viewer_press(bench, 0.5, 0.5, mods, release_at=(63.5, 31.5))
    sampled = float(bench.params(target)["red"])
    bench.window.viewer.knob_sampling = None
    # A single pixel of the ramp would read anywhere from 0 to 1; the region's mean is about 0.5.
    ok = expected is not None and abs(sampled - expected) < 1e-3
    return ok, f"Ctrl+Shift+drag over the whole ramp gave red {sampled:.4f} (frame mean {expected})"


@probe("color.picker_button")
def _picker_button(bench):
    from nodebased.colorpicker import HueSaturationWheel
    _key, picker = _open_swatch(bench)
    ok = picker is not None and bool(picker.findChildren(HueSaturationWheel))
    _close_pickers(bench)
    return ok, "the swatch opens a picker with a wheel and sliders" if ok else "no picker with a wheel opened"


@probe("color.slider_modes")
def _slider_modes(bench):
    from PySide6.QtWidgets import QLabel
    _key, picker = _open_swatch(bench)
    texts = {label.text() for label in picker.findChildren(QLabel)} if picker else set()
    _close_pickers(bench)
    ok = {"HSV", "TMI"} <= texts and any(text.startswith("RGB") for text in texts)
    return ok, f"slider headings: {sorted(t for t in texts if t in ('HSV', 'TMI') or t.startswith('RGB'))}"


@probe("color.wheel_hue_saturation")
def _wheel_hs(bench):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from nodebased.colorpicker import HueSaturationWheel
    _key, picker = _open_swatch(bench)
    wheels = picker.findChildren(HueSaturationWheel) if picker else []
    if not wheels:
        _close_pickers(bench)
        return False, "no wheel"
    wheel = wheels[0]
    c = wheel.width() // 2
    QTest.mousePress(wheel, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(wheel.width() - 5, c))
    QTest.mouseRelease(wheel, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(wheel.width() - 5, c))
    rim = (round(wheel.hue) % 360, round(wheel.saturation, 2))
    QTest.mousePress(wheel, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(c, c - wheel.height() // 4))
    QTest.mouseRelease(wheel, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(c, c - wheel.height() // 4))
    inner = (round(wheel.hue) % 360, round(wheel.saturation, 2))
    _close_pickers(bench)
    ok = rim[0] == 0 and rim[1] >= 0.95 and inner[0] == 90 and 0.3 < inner[1] < 0.7
    return ok, f"rim at the right: hue {rim[0]}, saturation {rim[1]}; halfway up: hue {inner[0]}, saturation {inner[1]}"


@probe("color.swatch_drag_copy")
def _swatch_drag(bench):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import QApplication
    bench.node("Constant", {"red": 0.9, "green": 0.2, "blue": 0.1, "alpha": 1.0})
    source = bench.child("color-swatch")
    getter = getattr(source, "color_getter", None)
    if source is None or getter is None or not source.acceptDrops():
        return False, "the swatch takes no drops"
    mime = source.color_mime(getter())
    target_key = bench.node("Constant")
    target = bench.child("color-swatch")
    point = QPointF(4, 4)
    actions = Qt.DropAction.CopyAction
    enter = QDragEnterEvent(point.toPoint(), actions, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target, enter)
    drop = QDropEvent(point, actions, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target, drop)
    bench.settle()
    params = bench.params(target_key)
    copied = [round(float(params[p]), 6) for p in ("red", "green", "blue")]
    return copied == [0.9, 0.2, 0.1], f"dropping a 0.9/0.2/0.1 swatch onto another Constant gives {copied}"


# -- animation menu --------------------------------------------------------------------------------------------------------

def _menu_action(bench, key, param, text):
    control = bench.field(param).parentWidget()
    menu = bench.window.build_curve_menu(key, param, control)
    for action in menu.actions():
        if action.text().startswith(text):
            return menu, action
    return menu, None


def _trigger(bench, key, param, text):
    menu, action = _menu_action(bench, key, param, text)
    if action is None:
        return False
    action.trigger()
    bench.settle()
    bench.show(key)
    return True


@probe("animation.set_key")
def _set_key(bench):
    key = bench.node("Grade")
    if not _trigger(bench, key, "gamma", "Set key"):
        return False, "no Set key entry"
    curve = bench.window.node_curve(key, "gamma")
    field = bench.field("gamma")
    tinted = bool(field.property("animated")) and bool(field.property("keyedHere"))
    return curve is not None and tinted, f"keyed {curve is not None}, field tinted with the key colour {tinted}"


@probe("animation.autokey")
def _autokey(bench):
    key = bench.node("Grade")
    _trigger(bench, key, "gamma", "Set key")
    bench.window.dispatcher.execute({"op": "time", "current": 10})
    bench.show(key)
    _type(bench, bench.field("gamma"), "2")
    curve = bench.window.node_curve(key, "gamma") or {"keys": []}
    frames = sorted(k["frame"] for k in curve["keys"])
    return len(frames) == 2, f"keys at {frames}"


@probe("animation.delete_key")
def _delete_key(bench):
    key = bench.node("Grade")
    _trigger(bench, key, "gamma", "Set key")
    if not _trigger(bench, key, "gamma", "Delete key"):
        return False, "no Delete key entry"
    curve = bench.window.node_curve(key, "gamma")
    return curve is None or not curve["keys"], "the key was removed" if curve is None or not curve["keys"] else "key still there"


@probe("animation.no_animation")
def _no_animation(bench):
    key = bench.node("Grade")
    _trigger(bench, key, "gamma", "Set key")
    _type(bench, bench.field("gamma"), "1.75")
    bench.show(key)
    menu, action = _menu_action(bench, key, "gamma", "Clear animation")
    if action is None:
        menu, action = _menu_action(bench, key, "gamma", "No animation")
    if action is None:
        return False, "no No animation / Clear animation entry"
    action.trigger()
    bench.settle()
    curve = bench.window.node_curve(key, "gamma")
    stored = float(bench.params(key)["gamma"])
    ok = curve is None and abs(stored - 1.75) < 1e-9
    return ok, f"entry '{action.text()}': curve removed {curve is None}, static value {stored:g} (frame value 1.75)"


@probe("animation.set_key_all_knobs")
def _set_key_all(bench):
    from PySide6.QtGui import QAction
    key = bench.node("Grade")
    before = bench.undo_count()
    actions = [a for a in bench.window.properties_dock.findChildren(QAction, "set-key-all-knobs")]
    if not actions:
        return False, "no Set key on all knobs in the panel menu"
    actions[0].trigger()
    bench.settle()
    keyed = [p for p in ("gamma", "exposure", "mix") if bench.window.node_curve(key, p) is not None]
    steps = None if before is None else bench.undo_count() - before
    return len(keyed) == 3 and steps in (None, 1), f"keyed {keyed}, undo steps {steps}"


@probe("animation.edit_expression")
def _edit_expression(bench):
    key = bench.node("Grade")
    _menu, action = _menu_action(bench, key, "gamma", "Enter expression")
    return action is not None, "the animation menu offers Enter expression" if action else "no expression entry"


# -- split (channel selector) ----------------------------------------------------------------------------------------------

@probe("split.button_counts_channels")
def _split_count(bench):
    bench.node("Grade")
    button = bench.child("multiply-split")
    text = button.text() if button is not None else None
    return text == "4", f"multiply's channel selector reads {text!r}"


@probe("split.edit_one_channel")
def _split_edit(bench):
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    key = bench.node("Grade")
    QTest.mouseClick(bench.child("multiply-split"), Qt.MouseButton.LeftButton)
    bench.settle()
    bench.show(key)
    value = bench.params(key)["multiply"]
    return isinstance(value, list) and len(value) == 4, f"after split the knob stores {value!r}"


@probe("split.close_keeps_values")
def _split_close(bench):
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractSpinBox, QLabel
    key = bench.node("Grade", {"multiply": [1.0, 2.0, 1.0, 1.0]})
    bench.child("multiply-split").click()
    bench.settle()
    closed = bench.params(key)["multiply"]
    folded = bench.child("multiply-split-summary", QLabel) is not None and bench.child("multiply-g-field", QAbstractSpinBox) is None
    bench.child("multiply-split").click()
    bench.settle()
    reopened = bench.child("multiply-g-field", QAbstractSpinBox)
    ok = closed == [1.0, 2.0, 1.0, 1.0] and folded and reopened is not None and reopened.value() == 2.0
    return ok, f"after closing the knob stores {closed!r}, fields folded {folded}; reopened g field shows {reopened.value() if reopened else None}"


# -- node tab --------------------------------------------------------------------------------------------------------------

NODE_TAB_CONTROLS = ("node-label", "node-font", "node-hide-input", "node-cached", "node-disable", "node-dope-sheet",
                     "node-bookmark", "node-thumbnail", "node-postage-stamp-frame", "node-lifetime-start", "node-lifetime-end")


@probe("node_tab.controls")
def _node_tab(bench):
    bench.node("Grade")
    missing = [name for name in NODE_TAB_CONTROLS if bench.child(name) is None]
    return not missing, f"missing {missing}" if missing else "every Node tab control is there"


@probe("node_tab.disable")
def _node_disable(bench):
    key = bench.node("Grade")
    box = bench.child("node-disable")
    box.click()
    bench.settle()
    node = bench.window.dispatcher.document["nodes"][key]
    disabled = bool(node.get("disabled") or node.get("bypass") or node["params"].get("disable"))
    return disabled, f"after checking disable the node is disabled: {disabled}"


# -- run -------------------------------------------------------------------------------------------------------------------

def probe_all():
    """{behaviour id: {"widget", "status", "result": "pass"|"fail"|"not probed"|"undocumented", "detail"}}."""
    ref = reference()
    bench = Bench()
    results = {}
    try:
        for entry in ref["behaviours"]:
            row = {"widget": entry["widget"], "status": entry["status"]}
            if entry["status"] != "documented":
                row.update(result="undocumented", detail=entry["nuke"])
            elif entry["id"] not in PROBES:
                row.update(result="not probed", detail="no interaction probe yet")
            else:
                try:
                    passed, detail = PROBES[entry["id"]](bench)
                except Exception as exc:  # a probe that crashes is a failing behaviour, named with its error
                    passed, detail = False, f"probe error: {type(exc).__name__}: {exc}"
                row.update(result="pass" if passed else "fail", detail=detail)
            results[entry["id"]] = row
    finally:
        bench.close()
    return results


def table(results):
    """Rows of (widget, passing, documented, failing ids) in the reference's widget order."""
    order = reference()["widget_order"]
    rows = []
    for widget in order:
        mine = {k: v for k, v in results.items() if v["widget"] == widget}
        documented = [k for k, v in mine.items() if v["status"] == "documented"]
        passing = [k for k in documented if mine[k]["result"] == "pass"]
        failing = [k for k in documented if mine[k]["result"] != "pass"]
        undocumented = [k for k, v in mine.items() if v["status"] != "documented"]
        rows.append((widget, len(passing), len(documented), failing, undocumented))
    return rows


if __name__ == "__main__":
    out = probe_all()
    for behaviour, row in out.items():
        print(f"{row['result']:12} {behaviour}: {row['detail']}")
