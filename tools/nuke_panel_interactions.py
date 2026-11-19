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


@probe("color.discard_sample")
def _discard(bench):
    from PySide6.QtCore import QPointF, Qt
    target, frame = _ramp_and_target(bench)
    if frame is None:
        return False, "the ramp never reached the viewer"
    before = [float(bench.params(target)[p]) for p in ("red", "green", "blue")]
    _viewer_press(bench, 40.5, 4.5, Qt.KeyboardModifier.ControlModifier)
    sampled = float(bench.params(target)["red"])
    viewer = bench.window.viewer
    viewer.hit_press(QPointF(10, 10), Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier)
    bench.settle()
    after = [float(bench.params(target)[p]) for p in ("red", "green", "blue")]
    still_on = _armed(bench)
    viewer.knob_sampling = None
    ok = sampled != before[0] and after == before and still_on
    return ok, f"sampled red {sampled:g}; Ctrl+right-click put back {after} (was {before}), eye dropper still on {still_on}"


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
    _menu, action = _menu_action(bench, key, "gamma", "Add expression")
    return action is not None, "the animation menu offers Add expression" if action else "no expression entry"


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


# The guide's Node-tab order (Properties Panels): every control on one line of this list, in order.
NODE_TAB_ORDER = ("node-label", "node-font", "node-bold", "node-italic", "node-font-size", "node-font-color", "node-hide-input",
                  "node-cached", "node-disable", "node-dope-sheet", "node-bookmark", "node-thumbnail", "node-postage-stamp-frame",
                  "node-lifetime-start", "node-lifetime-end")


@probe("node_tab.order")
def _node_tab_order(bench):
    from PySide6.QtCore import QObject
    key = bench.node("Grade")
    page = bench.window.node_tab(key, bench.window.graph_nodes()[key])
    names = [child.objectName() for child in page.findChildren(QObject) if child.objectName() in NODE_TAB_ORDER]
    page.deleteLater()
    return tuple(names) == NODE_TAB_ORDER, f"the Node tab reads {names}"


@probe("node_tab.disable")
def _node_disable(bench):
    key = bench.node("Grade")
    box = bench.child("node-disable")
    box.click()
    bench.settle()
    node = bench.window.dispatcher.document["nodes"][key]
    disabled = bool(node.get("disabled") or node.get("bypass") or node["params"].get("disable"))
    return disabled, f"after checking disable the node is disabled: {disabled}"


# -- channel pickers and the mask/mix row --------------------------------------------------------------------------------

@probe("channel.layer_check_boxes")
def _layer_boxes(bench):
    from PySide6.QtWidgets import QCheckBox
    key = bench.node("Grade", {"channels": "rgba"})
    # A per-channel box is named after the knob and the channel (channels-red-box and so on).
    boxes = [b for b in bench.window.properties_dock.findChildren(QCheckBox)
             if b.objectName().startswith("channels-") and b.window() is bench.window and not b.isHidden()]
    green = bench.child("channels-green-box", QCheckBox)
    if green is None:
        return False, f"{len(boxes)} per-channel check boxes for the channels knob"
    green.click()
    bench.settle()
    stored = bench.params(key)["channels"]
    bench.show(key)
    shown = {name: bench.child(f"channels-{name}-box", QCheckBox).isChecked() for name in ("red", "green", "blue", "alpha")}
    ok = len(boxes) == 4 and stored == "rgba.red rgba.blue rgba.alpha" and shown == {"red": True, "green": False, "blue": True, "alpha": True}
    return ok, f"{len(boxes)} boxes for rgba; unticking green stores {stored!r}, the rebuilt panel ticks {shown}"


@probe("channel.single_channel")
def _single_channel(bench):
    from PySide6.QtWidgets import QComboBox
    key = bench.node("Grade", {"channels": "none"})
    field = bench.child("channels-single", QComboBox)
    if field is None:
        return False, "no right-most channel field beside the channels menu at none"
    field.activated.emit(field.findText("rgba.alpha"))
    bench.settle()
    stored = bench.params(key)["channels"]
    bench.show(key)
    menu, again = bench.child("channels-choice", QComboBox), bench.child("channels-single", QComboBox)
    ok = stored == "rgba.alpha" and menu.currentText() == "none" and again is not None and again.currentText() == "rgba.alpha"
    return ok, f"picking alpha in the right-most field stores {stored!r}; the panel shows {menu.currentText()!r} and {again.currentText() if again else None!r}"


@probe("mask_mix.controls")
def _mask_controls(bench):
    bench.node("Grade")
    names = {"mask channel menu": "maskChannelInput-choice", "inject": "inject-switch", "invert": "invert_mask-switch",
             "mix": "mix-field"}
    missing = [label for label, name in names.items() if bench.child(name) is None]
    return not missing, f"missing {missing}" if missing else "mask menu, inject, invert and mix slider are all there"


# -- the Properties bin and knob linking (lane 10, step T4, 2026-11-17) -----------------------------------------------------

def _bin_reset(bench):
    bench.window.clear_panels()
    bench.window.graph.scene().clearSelection()
    bench.settle()


def _bin_node(bench, kind="Grade"):
    """A node made through the window, so the Node Graph shows it, with no panel opened by the probe."""
    bench.count += 1
    key = f"interaction_{bench.count}"
    bench.window.command({"op": "create", "id": key, "type": kind, "name": f"{kind}Bin{bench.count}",
                          "pos": [bench.count * 140, 400]})
    bench.settle()
    return key


def _open_panels(bench):
    """Node keys whose panels the docked bin shows, top first."""
    from PySide6.QtWidgets import QLineEdit
    dock = bench.window.properties.widget()
    names = {node["name"]: key for key, node in bench.window.dispatcher.document["nodes"].items()}
    found = [w for w in dock.findChildren(QLineEdit, "node-name-header") if w.isVisibleTo(dock)] if dock is not None else []
    found.sort(key=lambda w: w.mapTo(dock, w.rect().topLeft()).y())
    return [names.get(w.text()) for w in found]


def _floating_panels(bench):
    from PySide6.QtWidgets import QApplication, QLineEdit
    names = {node["name"]: key for key, node in bench.window.dispatcher.document["nodes"].items()}
    out = []
    for top in QApplication.topLevelWidgets():
        if top is bench.window or not top.isVisible():
            continue
        out += [names.get(w.text()) for w in top.findChildren(QLineEdit, "node-name-header")]
    return out


def _graph_point(bench, key):
    graph = bench.window.graph
    item = graph.items_by_id[key]
    return graph.mapFromScene(item.sceneBoundingRect().center())


def _graph_click(bench, key, modifiers=None, double=False):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    graph = bench.window.graph
    graph.centerOn(graph.items_by_id[key])
    bench.settle()
    point = _graph_point(bench, key)
    modifiers = modifiers or Qt.KeyboardModifier.NoModifier
    if double:
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, modifiers, point)
        QTest.mouseDClick(graph.viewport(), Qt.MouseButton.LeftButton, modifiers, point)
    else:
        QTest.mouseClick(graph.viewport(), Qt.MouseButton.LeftButton, modifiers, point)
    bench.settle(8)


def _close_button(bench, key):
    from PySide6.QtWidgets import QLineEdit, QToolButton
    name = bench.window.dispatcher.document["nodes"][key]["name"]
    root = bench.window.properties.widget()
    for header in root.findChildren(QLineEdit, "node-name-header") if root is not None else []:
        if header.text() == name and header.isVisibleTo(root):
            top = header.parentWidget()
            return top.findChild(QToolButton, "close-knobs")
    return None


@probe("bin.insert_opens")
def _bin_insert(bench):
    _bin_reset(bench)
    bench.window.add_node("Blur")
    bench.settle(8)
    key = bench.window.graph.selected_id()
    shown = _open_panels(bench)
    return key is not None and shown == [key], f"inserting a Blur selects {key!r}; the bin shows {shown}"


@probe("bin.double_click_opens")
def _bin_double_click(bench):
    _bin_reset(bench)
    first, second = _bin_node(bench), _bin_node(bench)
    _graph_click(bench, first, double=True)
    _graph_click(bench, second, double=True)
    shown = _open_panels(bench)
    return shown == [second, first], f"double-clicking two nodes leaves the bin showing {shown} (newest first)"


@probe("bin.ctrl_click_opens")
def _bin_ctrl_click(bench):
    from PySide6.QtCore import Qt
    _bin_reset(bench)
    first, second = _bin_node(bench), _bin_node(bench)
    _graph_click(bench, first, double=True)
    _graph_click(bench, second, Qt.KeyboardModifier.ControlModifier)
    shown = _open_panels(bench)
    return second in shown and first in shown, f"with one panel open, Ctrl+click on another node leaves the bin showing {shown}"


@probe("bin.return_opens")
def _bin_return(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    _bin_reset(bench)
    first, second = _bin_node(bench), _bin_node(bench)
    _graph_click(bench, first, double=True)
    graph = bench.window.graph
    graph.scene().clearSelection()
    graph.items_by_id[second].setSelected(True)      # selecting is not what is measured; Return is
    bench.settle()
    graph.setFocus()
    QTest.keyClick(graph, Qt.Key.Key_Return)
    bench.settle(8)
    shown = _open_panels(bench)
    return shown == [second, first], f"selecting a second node and pressing Return leaves the bin showing {shown}"


@probe("bin.float_open")
def _bin_float_open(bench):
    from PySide6.QtCore import Qt
    _bin_reset(bench)
    key = _bin_node(bench)
    _graph_click(bench, key, Qt.KeyboardModifier.ControlModifier, double=True)
    floating = _floating_panels(bench)
    for top in list(bench.window.floating_panels.values()) if hasattr(bench.window, "floating_panels") else []:
        top.close()
    bench.settle()
    return key in floating, f"Ctrl+double-click opens floating panels for {floating}"


@probe("bin.max_panels")
def _bin_max(bench):
    from PySide6.QtWidgets import QSpinBox
    _bin_reset(bench)
    keys = [_bin_node(bench) for _ in range(3)]
    old_cap = bench.window.panel_cap
    _graph_click(bench, keys[0], double=True)
    cap = bench.child("panel-cap-spin", QSpinBox)
    if cap is None:
        return False, "no maximum-panels field on the bin"
    cap.setValue(2)
    bench.settle()
    for key in keys[1:]:
        _graph_click(bench, key, double=True)
    shown = _open_panels(bench)
    stored = bench.window.preferences.max_properties_panels()
    bench.window.set_panel_cap(old_cap)
    return shown == [keys[2], keys[1]] and stored == 2, f"field set to 2, three nodes opened: the bin shows {shown}; the preference holds {stored}"


@probe("bin.lock")
def _bin_lock(bench):
    from PySide6.QtWidgets import QAbstractButton
    _bin_reset(bench)
    first, second = _bin_node(bench), _bin_node(bench)
    _graph_click(bench, first, double=True)
    lock = bench.child("lock-properties-bin", QAbstractButton)
    if lock is None:
        return False, "the bin has no lock button"
    lock.click()
    bench.settle()
    _graph_click(bench, second, double=True)
    shown, floating = _open_panels(bench), _floating_panels(bench)
    bench.window.set_bin_locked(False)
    for key in list(bench.window.floating_panels):
        bench.window.floating_panels.pop(key).close()
    bench.settle()
    ok = shown == [first] and second in floating
    return ok, f"locked, double-clicking a second node leaves the bin showing {shown} and floats {floating}"


@probe("bin.remove_all")
def _bin_remove_all(bench):
    from PySide6.QtWidgets import QPushButton
    _bin_reset(bench)
    first, second = _bin_node(bench), _bin_node(bench)
    _graph_click(bench, first, double=True)
    _graph_click(bench, second, double=True)
    before = _open_panels(bench)
    button = bench.child("clear-all-panels", QPushButton)
    if button is None:
        return False, "no remove-all button"
    button.click()
    bench.settle()
    return len(before) == 2 and not bench.window.pinned_panels, f"{len(before)} panels open; after Clear all the bin holds {bench.window.pinned_panels}"


def _close_with(bench, modifiers):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    _bin_reset(bench)
    keys = [_bin_node(bench) for _ in range(3)]
    for key in keys:
        _graph_click(bench, key, double=True)
    button = _close_button(bench, keys[1])
    if button is None:
        return keys, None
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, modifiers)
    bench.settle(8)
    return keys, list(bench.window.pinned_panels)


@probe("bin.close")
def _bin_close(bench):
    from PySide6.QtCore import Qt
    keys, left = _close_with(bench, Qt.KeyboardModifier.NoModifier)
    return left == [keys[2], keys[0]], f"closing the middle of three panels leaves {left}"


@probe("bin.alt_close_all")
def _bin_alt_close(bench):
    from PySide6.QtCore import Qt
    keys, left = _close_with(bench, Qt.KeyboardModifier.AltModifier)
    return left == [], f"Alt+click on one panel's close leaves {left}"


@probe("bin.ctrl_close_others")
def _bin_ctrl_close(bench):
    from PySide6.QtCore import Qt
    keys, left = _close_with(bench, Qt.KeyboardModifier.ControlModifier)
    return left == [keys[1]], f"Ctrl+click on the middle panel's close leaves {left}"


@probe("bin.float_button")
def _bin_float_button(bench):
    from PySide6.QtWidgets import QToolButton
    _bin_reset(bench)
    key = _bin_node(bench)
    _graph_click(bench, key, double=True)
    root = bench.window.properties.widget()
    button = root.findChild(QToolButton, "float-panel") if root is not None else None   # it sits in the panel's menu
    if button is None:
        return False, "no float button on the panel"
    button.click()
    bench.settle(8)
    floated, bin_after = _floating_panels(bench), _open_panels(bench)
    window = bench.window.floating_panels.get(key)
    again = window.findChild(QToolButton, "float-panel") if window is not None else None
    if again is None:
        return False, f"float moved the panel to {floated}; no float button on the floating panel"
    again.click()
    bench.settle(8)
    docked = _open_panels(bench)
    ok = floated == [key] and key not in bin_after and docked == [key] and not bench.window.floating_panels
    return ok, f"float: floating {floated}, bin {bin_after}; clicking it again docks it: bin {docked}"


@probe("bin.hide_pages")
def _bin_hide_pages(bench):
    from PySide6.QtWidgets import QWidget
    _bin_reset(bench)
    key = _bin_node(bench)
    _graph_click(bench, key, double=True)
    arrow = bench.child("panel-collapse-arrow")
    body = bench.child("node-panel-content", QWidget)
    if arrow is None or body is None:
        return False, "no collapse arrow on the panel header"
    arrow.click()
    bench.settle()
    hidden = not body.isVisibleTo(bench.window)
    header_kept = _open_panels(bench) == [key]
    arrow.click()
    bench.settle()
    return hidden and header_kept, f"the header arrow hides the pages: {hidden}; the header stays: {header_kept}"


# Linking

def _two_grades_in_bin(bench):
    _bin_reset(bench)
    source, target = _bin_node(bench), _bin_node(bench)
    bench.window.command({"op": "set", "id": source, "param": "gamma", "value": 1.7})
    _graph_click(bench, target, double=True)
    _graph_click(bench, source, double=True)
    return source, target


def _bin_field(bench, key, param):
    from PySide6.QtWidgets import QAbstractSpinBox
    root = bench.window.properties.widget()
    for field in root.findChildren(QAbstractSpinBox, f"{param}-field") if root is not None else []:
        if field.property("nodebased_node_id") == key and field.isVisibleTo(root):
            return field
    return None


@probe("link.add_expression")
def _link_add_expression(bench):
    key = bench.node("Grade")
    menu = bench.window.build_curve_menu(key, "gamma", bench.field("gamma"))
    texts = [action.text() for action in menu.actions()]
    menu.deleteLater()
    return any(t.startswith("Add expression") for t in texts), f"the knob menu offers {texts[:2]} (= on the field is probed under number.equals_expression)"


@probe("link.reference_other")
def _link_reference(bench):
    from PySide6.QtWidgets import QLineEdit, QPushButton
    source, target = _two_grades_in_bin(bench)
    editor = bench.window.open_expression_editor(target, "gamma", _bin_field(bench, target, "gamma"))
    if editor is None:
        return False, "no expression editor"
    editor.setText(f"{source}.gamma")
    editor.returnPressed.emit()
    bench.settle(8)
    stored = (bench.window.dispatcher.document.get("expressions") or {}).get(target, {}).get("gamma")
    field = _bin_field(bench, target, "gamma")
    shown = field.value() if field is not None else None
    return stored == f"{source}.gamma" and shown == 1.7, f"typing {source}.gamma stores {stored!r}; the linked field shows {shown}"


def _ctrl_drag(bench, source_field, target_field):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    edit = source_field.lineEdit()
    start = QPointF(edit.rect().center())
    target_global = target_field.mapToGlobal(target_field.rect().center())
    ctrl = Qt.KeyboardModifier.ControlModifier
    def send(kind, local, buttons, button=Qt.MouseButton.LeftButton):
        event = QMouseEvent(kind, local, QPointF(edit.mapToGlobal(local.toPoint())), button, buttons, ctrl)
        QApplication.sendEvent(edit, event)
    send(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    mid = QPointF(edit.mapFromGlobal(target_global))
    send(QEvent.Type.MouseMove, (start + mid) / 2, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    send(QEvent.Type.MouseMove, mid, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    send(QEvent.Type.MouseButtonRelease, mid, Qt.MouseButton.NoButton)
    bench.settle(8)


@probe("link.ctrl_drag")
def _link_ctrl_drag(bench):
    source, target = _two_grades_in_bin(bench)
    _ctrl_drag(bench, _bin_field(bench, source, "gamma"), _bin_field(bench, target, "gamma"))
    expressions = lambda: (bench.window.dispatcher.document.get("expressions") or {}).get(target, {}).get("gamma")
    stored = expressions()
    field = _bin_field(bench, target, "gamma")
    shown = field.value() if field is not None else None
    source_gamma = bench.params(source)["gamma"]
    bench.window.command({"op": "undo"})          # the link is one undo step: one undo takes it away whole
    bench.settle(8)
    undone = expressions() is None and bench.params(target)["gamma"] == 1.0
    ok = stored is not None and source in stored and undone and shown == 1.7 and source_gamma == 1.7
    return ok, f"Ctrl+drag of the first Grade's gamma onto the second's stores {stored!r}; it shows {shown}; one undo removes it: {undone}"


@probe("link.turns_blue")
def _link_blue(bench):
    from PySide6.QtGui import QColor
    import re
    source, target = _two_grades_in_bin(bench)
    bench.window.command({"op": "set_expression", "id": target, "param": "gamma", "expression": f"{source}.gamma"})
    bench.settle(8)
    field = _bin_field(bench, target, "gamma")
    match = re.search(r"color:\s*(#[0-9a-fA-F]{6})", field.styleSheet() if field else "")
    if match is None:
        return False, "the linked field has no colour of its own"
    hue = QColor(match.group(1)).hslHue()
    return 190 <= hue <= 250, f"the linked field is {match.group(1)} (hue {hue}; blue is about 190 to 250)"


@probe("link.graph_arrow")
def _link_graph_arrow(bench):
    source, target = _two_grades_in_bin(bench)
    bench.window.command({"op": "set_expression", "id": target, "param": "gamma", "expression": f"{source}.gamma"})
    bench.settle(8)
    links = [item for item in bench.window.graph.scene().items() if getattr(item, "expression_link", None) == (source, target)]
    if not links:
        return False, "the Node Graph draws nothing between nodes linked by an expression"
    color = links[0].pen().color()
    drawn = links[0].path().length() > 0
    return color.hslHue() in range(80, 160) and drawn, f"{len(links)} link drawn, colour {color.name()}, path drawn: {drawn}"


@probe("link.edit_expression")
def _link_edit(bench):
    from PySide6.QtWidgets import QLineEdit
    source, target = _two_grades_in_bin(bench)
    bench.window.command({"op": "set_expression", "id": target, "param": "gamma", "expression": f"{source}.gamma"})
    bench.settle(8)
    field = _bin_field(bench, target, "gamma")
    menu = bench.window.build_curve_menu(target, "gamma", field)
    edit = [a for a in menu.actions() if a.text().startswith("Edit expression")]
    if not edit:
        return False, "no Edit expression entry on a linked knob"
    edit[0].trigger()
    bench.settle()
    editors = [e for e in bench.window.properties_dock.findChildren(QLineEdit, "expression-editor") if e.isVisibleTo(bench.window)]
    menu.deleteLater()
    text = editors[-1].text() if editors else None
    return text == f"{source}.gamma", f"Edit expression opens an editor holding {text!r}"


def _bin_key_button(bench, key, param):
    from PySide6.QtWidgets import QPushButton
    root = bench.window.properties.widget()
    for button in root.findChildren(QPushButton, "key-button") if root is not None else []:
        if (button.property("nodebased_node_id"), button.property("nodebased_param")) == (key, param) and button.isVisibleTo(root):
            return button
    return None


@probe("link.ctrl_drag_animation_button")
def _link_button_drag(bench):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    _bin_reset(bench)
    source, target = _bin_node(bench), _bin_node(bench)
    bench.window.command({"op": "set", "id": source, "param": "gamma", "value": 1.7})
    bench.window.command({"op": "set", "id": target, "param": "gamma", "value": 1.2})   # one value for all of r g b a
    _graph_click(bench, target, double=True)
    _graph_click(bench, source, double=True)
    from_button, to_button = _bin_key_button(bench, source, "gamma"), _bin_key_button(bench, target, "gamma")
    if from_button is None or to_button is None:
        return False, "no animation buttons beside the two gamma knobs"
    ctrl = Qt.KeyboardModifier.ControlModifier
    start = QPointF(from_button.rect().center())
    end = QPointF(from_button.mapFromGlobal(to_button.mapToGlobal(to_button.rect().center())))
    def send(kind, local, buttons, button=Qt.MouseButton.LeftButton):
        QApplication.sendEvent(from_button, QMouseEvent(kind, local, QPointF(from_button.mapToGlobal(local.toPoint())), button,
                                                        buttons, ctrl))
    send(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    send(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)
    bench.settle(8)
    stored = (bench.window.dispatcher.document.get("expressions") or {}).get(target, {}).get("gamma")
    whole = not isinstance(bench.params(target)["gamma"], list) and bench.window.dispatcher.evaluated_param(target, "gamma") == 1.7 \
        if hasattr(bench.window.dispatcher, "evaluated_param") else not isinstance(bench.params(target)["gamma"], list)
    bench.window.command({"op": "undo"})
    bench.settle(8)
    undone = bench.params(target)["gamma"] == 1.2 and not (
        (bench.window.dispatcher.document.get("expressions") or {}).get(target, {}).get("gamma"))
    ok = stored is not None and source in stored and whole and undone
    return ok, (f"Ctrl+drag of one gamma's animation button onto the other's stores {stored!r} for the whole "
                f"r g b a knob: {whole}; one undo brings the old value back: {undone}")


@probe("link.link_menu")
def _link_menu(bench):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication, QToolButton, QWidget
    from nodebased.animation import resolve_document
    window = bench.window
    doc = lambda: window.dispatcher.document
    linked = lambda key, param: (doc().get("expressions") or {}).get(key, {}).get(param)
    # Set link from the channels knob's Link menu.
    source = bench.node("Grade", {"channels": "rgba"})
    target = bench.node("Grade")
    button = bench.child("channels-link-menu", QToolButton)
    menu = window.build_link_menu(target, "channels")
    set_menu = next((a.menu() for a in menu.actions() if a.menu() is not None and a.menu().objectName() == "link-menu-set"), None)
    entry = [a for a in (set_menu.actions() if set_menu else []) if a.data() == window.knob_reference(source, "channels")]
    names = sorted(a.objectName() for a in menu.actions() if a.objectName())
    if entry:
        entry[0].trigger()
    bench.settle(8)
    stored = linked(target, "channels")
    follows = resolve_document(doc(), 1)["nodes"][target]["params"]["channels"] == "rgba"
    window.command({"op": "set", "id": source, "param": "channels", "value": "alpha"})
    still = resolve_document(doc(), 1)["nodes"][target]["params"]["channels"] == "alpha"
    bench.show(target)
    row_control = bench.child("channels-link-row", QWidget)
    locked = row_control is not None and not row_control.layout().itemAt(0).widget().isEnabled()
    window.command({"op": "undo"})
    window.command({"op": "undo"})
    undone = linked(target, "channels") is None
    window.command({"op": "redo"})
    # Edit link and Remove link.
    other = bench.node("Grade", {"channels": "rgb"})
    window.edit_word_link(target, "channels", window.knob_reference(other, "channels"))
    edited = linked(target, "channels") == window.knob_reference(other, "channels")
    remove = [a for a in window.build_link_menu(target, "channels").actions() if a.objectName() == "link-menu-remove"]
    if remove:
        remove[0].trigger()
    bench.settle(8)
    removed = linked(target, "channels") is None
    # Ctrl+drag of a Merge's A channels Link menu onto its B channels knob.
    merge = bench.node("Merge", {"Achannels": "rgb", "Bchannels": "rgba"})
    drag_from, drop_on = bench.child("Achannels-link-menu", QToolButton), bench.child("Bchannels-link-row", QWidget)
    dragged = False
    if drag_from is not None and drop_on is not None:
        ctrl = Qt.KeyboardModifier.ControlModifier
        start = QPointF(drag_from.rect().center())
        end = QPointF(drag_from.mapFromGlobal(drop_on.mapToGlobal(drop_on.rect().center())))
        for kind, local, buttons in ((QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
                                     (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
                                     (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)):
            QApplication.sendEvent(drag_from, QMouseEvent(kind, local, QPointF(drag_from.mapToGlobal(local.toPoint())),
                                                          Qt.MouseButton.LeftButton, buttons, ctrl))
        bench.settle(8)
        dragged = linked(merge, "Bchannels") == window.knob_reference(merge, "Achannels") and \
            resolve_document(doc(), 1)["nodes"][merge]["params"]["Bchannels"] == "rgb"
    # A format knob: the linked Reformat takes the other's format and its size.
    first = bench.node("Reformat", {"format": "HD_720"})
    second = bench.node("Reformat")
    fmt_button = bench.child("format-link-menu", QToolButton)
    window.set_word_link(second, "format", window.knob_reference(first, "format"))
    out = resolve_document(doc(), 1)["nodes"][second]["params"]
    formats = fmt_button is not None and out["format"] == "HD_720" and (out["width"], out["height"]) == (1280, 720)
    ok = (button is not None and bool(entry) and names == ["link-menu-edit", "link-menu-remove"] and stored is not None
          and follows and still and locked and undone and edited and removed and dragged and formats)
    return ok, (f"the channels knob has a Link menu button: {button is not None}; Set link lists the other Grade's channels: "
                f"{bool(entry)}, with Edit link and Remove link beside it: {names}; choosing it stores {stored!r} and the knob "
                f"follows the other's channels, also after they change: {follows and still}; the linked knob is locked in the "
                f"panel: {locked}; one undo per edit: {undone}; Edit link retargets: {edited}; Remove link clears: {removed}; "
                f"Ctrl+drag of A channels' Link menu onto B channels links B to A: {dragged}; a Reformat's format links with "
                f"its size: {formats}")


@probe("link.tracker_link_to")
def _link_tracker(bench):
    from PySide6.QtWidgets import QMenu
    tracker = bench.node("Tracker")
    bench.window.command({"op": "set", "id": tracker, "param": "translate_x", "value": 12.5})
    key = bench.node("Transform")
    menu = bench.window.build_curve_menu(key, "translate_x", bench.field("translate_x"))
    link_to = [a.menu() for a in menu.actions() if a.text() == "Link to" and a.menu() is not None]
    entries = [a for a in link_to[0].actions() if a.objectName() == "link-to-tracker"] if link_to else []
    if not entries:
        menu.deleteLater()
        return False, "the knob menu has no Link to > Tracker entry"
    entries[0].trigger()
    bench.settle(8)
    menu.deleteLater()
    stored = (bench.window.dispatcher.document.get("expressions") or {}).get(key, {}).get("translate_x")
    return stored is not None and tracker in stored and "translate_x" in stored, (
        f"Link to > {entries[0].text()} on Transform's translate x stores {stored!r}")


# Node Graph (Nuke 17.0 User Guide, Working with Nodes and Using the Tab Menu)

def _graph_fresh(bench):
    """An empty graph with focus, so each probe sees only its own nodes."""
    _bin_reset(bench)
    doc = bench.window.dispatcher.document
    # Earlier probes leave expression links; a node another one reads cannot be deleted, so the links go first.
    links = [{"op": "clear_expression", "id": key, "param": param}
             for key, params in (doc.get("expressions") or {}).items() for param in params]
    if links:
        bench.window.command({"op": "batch", "commands": links})
    for key in list(doc["nodes"]):
        if key in bench.window.dispatcher.document["nodes"]:
            bench.window.command({"op": "delete", "id": key})
    bench.settle()
    bench.window.graph.setFocus()
    return bench.window.graph


def _graph_node(bench, kind, x=0, y=0, params=None):
    bench.count += 1
    key = f"graph_{bench.count}"
    command = {"op": "create", "id": key, "type": kind, "pos": [x, y]}
    if params:
        command["params"] = params
    bench.window.command(command)
    bench.settle()
    return key


def _graph_select(bench, *keys):
    graph = bench.window.graph
    graph.scene().clearSelection()
    for key in keys:
        graph.items_by_id[key].setSelected(True)
    bench.settle()


def _graph_key(bench, key, modifiers=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.keyClick(bench.window.graph, key, modifiers or Qt.KeyboardModifier.NoModifier)
    bench.settle(6)


def _tab_choose(bench, text, enter=True):
    """Press Tab in the Node Graph, type `text` into the Tab menu that opens and press Return."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from nodebased.app import NodeSearch
    seen = {}
    def drive():
        pickers = [w for w in QApplication.topLevelWidgets() if isinstance(w, NodeSearch) and w.isVisible()]
        if not pickers:
            QTimer.singleShot(20, drive)
            return
        picker = pickers[-1]
        if text:
            QTest.keyClicks(picker.query, text)
        seen["first"] = picker.list.item(0).data(Qt.ItemDataRole.UserRole) if picker.list.count() else None
        QTest.keyClick(picker.query, Qt.Key.Key_Return if enter else Qt.Key.Key_Escape)
    QTimer.singleShot(0, drive)
    QApplication.setActiveWindow(bench.window)     # an earlier probe's floating panel may hold the active window
    bench.window.graph.setFocus()
    bench.settle()
    seen["focus"] = bench.window.graph.hasFocus()
    _graph_key(bench, Qt.Key.Key_Tab)
    bench.tab_focus = seen.get("focus")
    return seen.get("first")


def _new_nodes(bench, before):
    nodes = bench.window.dispatcher.document["nodes"]
    return [key for key in nodes if key not in before]


@probe("graph.tab_create")
def _graph_tab_create(bench):
    _graph_fresh(bench)
    grade = _graph_node(bench, "Grade")
    _graph_select(bench, grade)
    before, undo = set(bench.window.dispatcher.document["nodes"]), bench.undo_count()
    _tab_choose(bench, "Blur")
    new = _new_nodes(bench, before)
    nodes = bench.window.dispatcher.document["nodes"]
    made = [k for k in new if nodes[k]["type"] == "Blur"]
    wired = bool(made) and grade in (nodes[made[0]].get("inputs") or {}).values()
    bench.window.command({"op": "undo"})          # the new node and its wire are one undo step (the stack may be capped)
    bench.settle()
    steps = 1 if made and made[0] not in bench.window.dispatcher.document["nodes"] and grade in bench.window.dispatcher.document["nodes"] else 0
    return bool(made) and wired and steps == 1, f"Tab, Blur, Return made {[nodes[k]['type'] for k in new]}; wired to the Grade: {wired}; undo steps {steps}; graph focused: {getattr(bench, 'tab_focus', None)}"


@probe("graph.tab_last_node")
def _graph_tab_last(bench):
    _graph_fresh(bench)
    grade = _graph_node(bench, "Grade")
    _graph_select(bench, grade)
    _tab_choose(bench, "Sharpen")
    from PySide6.QtCore import QEvent
    bench.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    bench.settle(6)
    before = set(bench.window.dispatcher.document["nodes"])
    first = _tab_choose(bench, "")
    new = [bench.window.dispatcher.document["nodes"][k]["type"] for k in _new_nodes(bench, before)]
    return new == ["Sharpen"], f"after making a Sharpen from the Tab menu, Tab then Return makes {new} (the menu listed {first!r} first)"


@probe("graph.tab_weighting")
def _graph_tab_weighting(bench):
    from nodebased.app import NodeSearch, SPECS
    _graph_fresh(bench)
    for _ in range(3):
        _tab_choose(bench, "Erode")
    picker = NodeSearch(bench.window, SPECS, bench.window.mapToGlobal(bench.window.rect().center()))
    picker.update_matches("e")
    from PySide6.QtCore import Qt
    first = picker.list.item(0).data(Qt.ItemDataRole.UserRole) if picker.list.count() else None
    picker.deleteLater()
    return isinstance(first, str) and first.startswith("Erode"), f"after three Erodes from the Tab menu, typing e lists {first!r} first"


@probe("graph.add_connects_selected")
def _graph_add_connects(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    grade = _graph_node(bench, "Grade")
    _graph_select(bench, grade)
    before = set(bench.window.dispatcher.document["nodes"])
    _graph_key(bench, Qt.Key.Key_B)
    nodes = bench.window.dispatcher.document["nodes"]
    new = _new_nodes(bench, before)
    wired = bool(new) and grade in (nodes[new[0]].get("inputs") or {}).values()
    return wired, f"B with the Grade selected made {[nodes[k]['type'] for k in new]}, wired to it: {wired}"


@probe("graph.disable_d")
def _graph_disable(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    grade = _graph_node(bench, "Grade")
    _graph_select(bench, grade)
    _graph_key(bench, Qt.Key.Key_D)
    off = bench.window.dispatcher.document["nodes"][grade]["disabled"]
    _graph_key(bench, Qt.Key.Key_D)
    on = not bench.window.dispatcher.document["nodes"][grade]["disabled"]
    return off and on, f"D disables the Grade: {off}; D again enables it: {on}"


@probe("graph.rename_n")
def _graph_rename(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit
    _graph_fresh(bench)
    grade = _graph_node(bench, "Grade")
    _graph_select(bench, grade)
    _graph_key(bench, Qt.Key.Key_N)
    editors = [e for e in bench.window.graph.viewport().findChildren(QLineEdit, "node-rename") if e.isVisible()]
    if not editors:
        return False, "N opens no rename field on the node"
    editors[-1].selectAll()
    QTest.keyClicks(editors[-1], "KeyGrade")
    QTest.keyClick(editors[-1], Qt.Key.Key_Return)
    bench.settle(6)
    name = bench.window.dispatcher.document["nodes"][grade].get("name")
    return name == "KeyGrade", f"N, typing KeyGrade and Return names the node {name!r}"


@probe("graph.select_all")
def _graph_select_all(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    keys = [_graph_node(bench, "Grade", 0, 0), _graph_node(bench, "Blur", 200, 0)]
    _graph_key(bench, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    chosen = sorted(bench.window.graph.selected_ids())
    return chosen == sorted(keys), f"Ctrl+A selects {chosen}"


@probe("graph.shift_click_select")
def _graph_shift_click(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    first, second = _graph_node(bench, "Grade", 0, 0), _graph_node(bench, "Blur", 300, 0)
    bench.window.graph.frame_nodes()
    bench.settle()
    _graph_click(bench, first)
    _graph_click(bench, second, Qt.KeyboardModifier.ShiftModifier)
    chosen = sorted(bench.window.graph.selected_ids())
    return chosen == sorted([first, second]), f"click then Shift+click selects {chosen}"


@probe("graph.copy_paste")
def _graph_copy_paste(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    blur, grade = _graph_node(bench, "Blur", 0, 0), _graph_node(bench, "Grade", 300, 0)
    _graph_select(bench, blur)
    _graph_key(bench, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    _graph_select(bench, grade)
    before = set(bench.window.dispatcher.document["nodes"])
    _graph_key(bench, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    nodes = bench.window.dispatcher.document["nodes"]
    new = _new_nodes(bench, before)
    wired = bool(new) and grade in (nodes[new[0]].get("inputs") or {}).values()
    return bool(new) and nodes[new[0]]["type"] == "Blur" and wired, (
        f"Ctrl+C, select the Grade, Ctrl+V pastes {[nodes[k]['type'] for k in new]}, wired to the Grade: {wired}")


@probe("graph.cut")
def _graph_cut(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    blur = _graph_node(bench, "Blur")
    _graph_select(bench, blur)
    _graph_key(bench, Qt.Key.Key_X, Qt.KeyboardModifier.ControlModifier)
    gone = blur not in bench.window.dispatcher.document["nodes"]
    return gone, f"Ctrl+X removes the node: {gone}"


@probe("graph.ctrl_d_disconnect")
def _graph_ctrl_d(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    grade = _graph_node(bench, "Grade", 0, 0)
    blur = _graph_node(bench, "Blur", 0, 150)
    bench.window.command({"op": "connect", "id": blur, "input": "image", "source": grade})
    _graph_select(bench, blur)
    _graph_key(bench, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier)
    inputs = {k: v for k, v in (bench.window.dispatcher.document["nodes"][blur].get("inputs") or {}).items() if v}
    bench.window.command({"op": "undo"})
    back = (bench.window.dispatcher.document["nodes"][blur].get("inputs") or {}).get("image") == grade
    return not inputs and back, f"Ctrl+D leaves the Blur's inputs {inputs}; one undo reconnects it: {back}"


@probe("graph.autocolor")
def _graph_autocolor(bench):
    _graph_fresh(bench)
    grade, blur, merge = _graph_node(bench, "Grade"), _graph_node(bench, "Blur", 200), _graph_node(bench, "Merge", 400)
    items = bench.window.graph.items_by_id
    colours = {items[k].brush().color().name() for k in (grade, blur, merge)}
    return len(colours) == 3, f"Grade, Blur and Merge draw in {sorted(colours)}"


@probe("graph.tile_color")
def _graph_tile_color(bench):
    _graph_fresh(bench)
    grade = _graph_node(bench, "Grade")
    before = bench.window.graph.items_by_id[grade].brush().color().name()
    bench.window.command({"op": "node_tab", "id": grade, "name": "tile_color", "value": "#ff0000"})
    bench.settle()
    after = bench.window.graph.items_by_id[grade].brush().color()
    hue = after.hslHue()
    red = hue >= 340 or 0 <= hue <= 20
    return after.name() != before and red, (f"a red tile colour turns the node from {before} to {after.name()} (hue {hue}; the graph "
                                            "draws tiles darkened for its dark theme)")


@probe("graph.backdrop_moves_contents")
def _graph_backdrop(bench):
    from PySide6.QtCore import QEvent, Qt
    _graph_fresh(bench)
    backdrop = _graph_node(bench, "Backdrop", 0, 0)
    inside = _graph_node(bench, "Grade", 40, 60)
    graph = bench.window.graph
    graph.frame_nodes()
    bench.settle()
    item = graph.items_by_id[backdrop]
    rect = item.sceneBoundingRect()
    start = graph.mapFromScene(rect.left() + 12, rect.top() + 8)
    end = start + __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(60, 0)
    view = graph.viewport()
    _mouse(view, QEvent.Type.MouseButtonPress, (start.x(), start.y()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
           Qt.KeyboardModifier.NoModifier)
    for step in (20, 40, 60):
        _mouse(view, QEvent.Type.MouseMove, (start.x() + step, start.y()), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
               Qt.KeyboardModifier.NoModifier)
    _mouse(view, QEvent.Type.MouseButtonRelease, (end.x(), end.y()), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
           Qt.KeyboardModifier.NoModifier)
    bench.settle(6)
    nodes = bench.window.dispatcher.document["nodes"]
    moved_b, moved_g = nodes[backdrop]["pos"][0] != 0, nodes[inside]["pos"][0] != 40
    return moved_b and moved_g, f"dragging the Backdrop moves it: {moved_b}; the Grade on it moves too: {moved_g}"


def _graph_drag(bench, start, end, modifiers=None, steps=6):
    """A left-button drag on the Node Graph from viewport point `start` to `end`, with `modifiers` held throughout."""
    from PySide6.QtCore import QEvent, Qt
    mods = modifiers or Qt.KeyboardModifier.NoModifier
    view = bench.window.graph.viewport()
    left = Qt.MouseButton.LeftButton
    _mouse(view, QEvent.Type.MouseButtonPress, (start.x(), start.y()), left, left, mods)
    for i in range(1, steps + 1):
        x = start.x() + (end.x() - start.x()) * i / steps
        y = start.y() + (end.y() - start.y()) * i / steps
        _mouse(view, QEvent.Type.MouseMove, (x, y), Qt.MouseButton.NoButton, left, mods)
    _mouse(view, QEvent.Type.MouseButtonRelease, (end.x(), end.y()), left, Qt.MouseButton.NoButton, mods)
    bench.settle(8)


def _scene_to_view(bench, point):
    return bench.window.graph.mapFromScene(point)


def _chain(bench, *kinds, spacing=150):
    """Nodes of `kinds` in a column, each wired to the one above; the graph framed on them."""
    keys = []
    for index, kind in enumerate(kinds):
        key = _graph_node(bench, kind, 0, index * spacing)
        if keys:
            slot = "input" if kind == "Dot" else ("B" if kind == "Merge" else "image")
            bench.window.command({"op": "connect", "id": key, "input": slot, "source": keys[-1]})
        keys.append(key)
    bench.settle()
    bench.window.graph.frame_nodes()
    bench.settle()
    return keys


def _inputs(bench, key):
    return {k: v for k, v in ((bench.window.dispatcher.document["nodes"].get(key) or {}).get("inputs") or {}).items() if v}


def _undo(bench):
    bench.window.command({"op": "undo"})
    bench.settle(6)


def _tab_with(bench, text, modifier):
    """The Tab menu, typing `text` and pressing Return with `modifier` held."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from nodebased.app import NodeSearch
    def drive():
        pickers = [w for w in QApplication.topLevelWidgets() if isinstance(w, NodeSearch) and w.isVisible()]
        if not pickers:
            QTimer.singleShot(20, drive)
            return
        QTest.keyClicks(pickers[-1].query, text)
        QTest.keyClick(pickers[-1].query, Qt.Key.Key_Return, modifier)
    QTimer.singleShot(0, drive)
    QApplication.setActiveWindow(bench.window)
    bench.window.graph.setFocus()
    bench.settle()
    _graph_key(bench, Qt.Key.Key_Tab)


@probe("graph.shift_new_branch")
def _graph_shift_branch(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    grade, blur = _chain(bench, "Grade", "Blur")
    _graph_select(bench, grade)
    before = set(bench.window.dispatcher.document["nodes"])
    _tab_with(bench, "Sharpen", Qt.KeyboardModifier.ShiftModifier)
    new = _new_nodes(bench, before)
    branch = bool(new) and _inputs(bench, new[0]).get("image") == grade
    kept = _inputs(bench, blur).get("image") == grade
    selected = bench.window.graph.selected_ids() == new
    _undo(bench)
    one_step = bool(new) and new[0] not in bench.window.dispatcher.document["nodes"] and _inputs(bench, blur).get("image") == grade
    return branch and kept and selected and one_step, (
        f"Shift+Return in the Tab menu with the Grade selected makes a Sharpen on the Grade: {branch}; the Blur below still reads "
        f"the Grade: {kept}; the new node is the selection: {selected}; one undo removes it: {one_step}")


@probe("graph.ctrl_replace")
def _graph_ctrl_replace(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    grade, blur, sharpen = _chain(bench, "Grade", "Blur", "Sharpen")
    _graph_select(bench, blur)
    before = set(bench.window.dispatcher.document["nodes"])
    _tab_with(bench, "Erode", Qt.KeyboardModifier.ControlModifier)
    nodes = bench.window.dispatcher.document["nodes"]
    new = _new_nodes(bench, before)
    gone = blur not in nodes
    wired = bool(new) and grade in _inputs(bench, new[0]).values() and _inputs(bench, sharpen).get("image") == new[0]
    selected = bench.window.graph.selected_ids() == new
    _undo(bench)
    back = blur in bench.window.dispatcher.document["nodes"] and _inputs(bench, sharpen).get("image") == blur
    return gone and wired and selected and back, (
        f"Ctrl+Return in the Tab menu with the Blur selected removes the Blur: {gone}; the Erode takes its place between the Grade "
        f"and the Sharpen: {wired}; the Erode is the selection: {selected}; one undo brings the Blur and its wires back: {back}")


@probe("graph.insert_between")
def _graph_insert_between(bench):
    from PySide6.QtCore import QPointF
    _graph_fresh(bench)
    grade, blur = _chain(bench, "Grade", "Blur", spacing=300)
    free = _graph_node(bench, "Sharpen", 400, 150)
    graph = bench.window.graph
    graph.frame_nodes()
    bench.settle()
    record = next(r for r in graph.edges if r[1] == grade and r[2] == blur)
    middle = record[0].path().pointAtPercent(0.5)
    item = graph.items_by_id[free]
    grab = item.sceneBoundingRect().center()
    _graph_select(bench, free)
    _graph_drag(bench, _scene_to_view(bench, grab), _scene_to_view(bench, QPointF(middle.x(), middle.y())))
    bench.settle(8)
    spliced = _inputs(bench, free).get("image") == grade and _inputs(bench, blur).get("image") == free
    _undo(bench)
    back = _inputs(bench, blur).get("image") == grade and not _inputs(bench, free)
    return spliced and back, (f"dragging a free Sharpen onto the Grade to Blur wire places it between them: {spliced}; one undo "
                              f"restores the wire and the Sharpen's place: {back}")


def _input_port_view(bench, key, slot="image"):
    return _scene_to_view(bench, bench.window.graph.items_by_id[key].inputs[slot].scenePos())


@probe("graph.drag_pipe_disconnect")
def _graph_drag_disconnect(bench):
    from PySide6.QtCore import QPoint
    _graph_fresh(bench)
    grade, blur = _chain(bench, "Grade", "Blur")
    start = _input_port_view(bench, blur)
    _graph_drag(bench, start, start + QPoint(-260, 10))
    gone = not _inputs(bench, blur)
    _undo(bench)
    back = _inputs(bench, blur).get("image") == grade
    return gone and back, f"dragging the Blur's input end to empty space disconnects it: {gone}; one undo reconnects: {back}"


@probe("graph.drag_pipe_connect")
def _graph_drag_connect(bench):
    _graph_fresh(bench)
    grade, blur = _chain(bench, "Grade", "Blur")
    other = _graph_node(bench, "Constant", 300, -40)
    graph = bench.window.graph
    graph.frame_nodes()
    bench.settle()
    start = _input_port_view(bench, blur)
    centre = _scene_to_view(bench, graph.items_by_id[other].sceneBoundingRect().center())
    _graph_drag(bench, start, centre)
    moved = _inputs(bench, blur).get("image") == other
    _undo(bench)
    back = _inputs(bench, blur).get("image") == grade
    return moved and back, (f"dragging the Blur's input end onto the middle of a Constant connects the Blur to it: {moved}; one undo "
                            f"puts it back on the Grade: {back}")


@probe("graph.shift_drag_duplicate_pipe")
def _graph_shift_duplicate(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    grade, blur = _chain(bench, "Grade", "Blur")
    sharpen = _graph_node(bench, "Sharpen", 320, 150)
    graph = bench.window.graph
    graph.frame_nodes()
    bench.settle()
    record = next(r for r in graph.edges if r[1] == grade and r[2] == blur)
    on_wire = _scene_to_view(bench, record[0].path().pointAtPercent(0.5))
    target = _scene_to_view(bench, graph.items_by_id[sharpen].inputs["image"].scenePos())
    _graph_drag(bench, on_wire, target, Qt.KeyboardModifier.ShiftModifier)
    both = _inputs(bench, sharpen).get("image") == grade and _inputs(bench, blur).get("image") == grade
    _undo(bench)
    back = not _inputs(bench, sharpen) and _inputs(bench, blur).get("image") == grade
    return both and back, (f"Shift+dragging the Grade to Blur wire onto a Sharpen wires the Grade into the Sharpen and keeps the "
                           f"Blur on it: {both}; one undo removes only the new wire: {back}")


@probe("graph.ctrl_click_dot")
def _graph_ctrl_dot(bench):
    from PySide6.QtCore import QEvent, QPoint, Qt
    _graph_fresh(bench)
    grade, blur = _chain(bench, "Grade", "Blur", spacing=300)
    graph = bench.window.graph
    record = next(r for r in graph.edges if r[1] == grade and r[2] == blur)
    handle = getattr(record[0], "handle", None) or record[0].path().pointAtPercent(0.5)
    point = _scene_to_view(bench, handle)
    ctrl = Qt.KeyboardModifier.ControlModifier
    view, left = graph.viewport(), Qt.MouseButton.LeftButton
    _mouse(view, QEvent.Type.MouseButtonPress, (point.x(), point.y()), left, left, ctrl)
    _mouse(view, QEvent.Type.MouseButtonRelease, (point.x(), point.y()), left, Qt.MouseButton.NoButton, ctrl)
    bench.settle(8)
    nodes = bench.window.dispatcher.document["nodes"]
    dots = [k for k, n in nodes.items() if n["type"] == "Dot"]
    passes = bool(dots) and _inputs(bench, dots[0]).get("input") == grade and _inputs(bench, blur).get("image") == dots[0]
    moved = False
    if dots:
        bench.settle()
        start = _scene_to_view(bench, graph.items_by_id[dots[0]].sceneBoundingRect().center())
        old = list(nodes[dots[0]]["pos"])
        _graph_select(bench, dots[0])
        _graph_drag(bench, start, start + QPoint(60, 0))
        moved = bench.window.dispatcher.document["nodes"][dots[0]]["pos"] != old and _inputs(bench, blur).get("image") == dots[0]
        _undo(bench)
    _undo(bench)
    back = not [k for k, n in bench.window.dispatcher.document["nodes"].items() if n["type"] == "Dot"] and _inputs(bench, blur).get("image") == grade
    return passes and moved and back, (f"Ctrl+click on the wire's yellow handle adds a Dot that takes the Grade and feeds the Blur: "
                                       f"{passes}; the Dot can be dragged and keeps its wires: {moved}; undo removes it: {back}")


@probe("graph.ctrl_drag_upstream")
def _graph_ctrl_upstream(bench):
    from PySide6.QtCore import QPoint, Qt
    _graph_fresh(bench)
    grade, blur, sharpen = _chain(bench, "Grade", "Blur", "Sharpen")
    loose = _graph_node(bench, "Constant", 400, 0)
    graph = bench.window.graph
    graph.frame_nodes()
    bench.settle()
    _graph_select(bench, loose)
    old = {k: list(bench.window.dispatcher.document["nodes"][k]["pos"]) for k in (grade, blur, sharpen, loose)}
    start = _scene_to_view(bench, graph.items_by_id[sharpen].sceneBoundingRect().center())
    _graph_drag(bench, start, start + QPoint(80, 0), Qt.KeyboardModifier.ControlModifier)
    chosen = sorted(graph.selected_ids())
    nodes = bench.window.dispatcher.document["nodes"]
    moved = [k for k in old if nodes[k]["pos"] != old[k]]
    _undo(bench)
    back = all(bench.window.dispatcher.document["nodes"][k]["pos"] == old[k] for k in old)
    want = sorted([grade, blur, sharpen])
    return chosen == want and sorted(moved) == want and back, (
        f"Ctrl+dragging the Sharpen selects {len(chosen)} nodes (it and the two feeding it: {chosen == want}) and moves "
        f"{len(moved)} of them together, the unconnected Constant left behind; one undo puts them back: {back}")


@probe("graph.select_connected")
def _graph_select_connected(bench):
    from PySide6.QtCore import Qt
    _graph_fresh(bench)
    grade, blur, sharpen = _chain(bench, "Grade", "Blur", "Sharpen")
    loose = _graph_node(bench, "Constant", 400, 0)
    _graph_select(bench, blur)
    _graph_key(bench, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
    chosen = sorted(bench.window.graph.selected_ids())
    want = sorted([grade, blur, sharpen])
    menu = bench.window.findChild(__import__("PySide6.QtGui", fromlist=["QAction"]).QAction, "select-connected-action")
    return chosen == want and menu is not None, (f"Ctrl+Alt+A with the Blur selected selects the Grade, Blur and Sharpen and leaves "
                                                 f"the loose Constant: {chosen == want}; Edit menu entry present: {menu is not None}")


@probe("graph.snap_to_grid")
def _graph_snap(bench):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QAction
    _graph_fresh(bench)
    action = bench.window.findChild(QAction, "snap-to-grid-action")
    graph = bench.window.graph
    was = graph.snap_to_grid
    if action is None:
        return False, "there is no Snap to grid setting"
    action.setChecked(True)
    key = _graph_node(bench, "Grade", 5, 7)
    graph.frame_nodes()
    bench.settle()
    _graph_select(bench, key)
    start = _scene_to_view(bench, graph.items_by_id[key].sceneBoundingRect().center())
    _graph_drag(bench, start, start + QPoint(37, 23))
    pos = bench.window.dispatcher.document["nodes"][key]["pos"]
    step = type(graph).GRID_STEP
    on_grid = all(abs(v / step - round(v / step)) < 1e-6 for v in pos)
    action.setChecked(False)
    free = _graph_node(bench, "Blur", 5, 300)
    bench.settle()
    _graph_select(bench, free)
    start = _scene_to_view(bench, graph.items_by_id[free].sceneBoundingRect().center())
    _graph_drag(bench, start, start + QPoint(37, 23))
    off = bench.window.dispatcher.document["nodes"][free]["pos"]
    loose = not all(abs(v / step - round(v / step)) < 1e-6 for v in off)
    action.setChecked(was)
    return on_grid and loose, (f"with Edit > Snap to grid on, a dragged Grade lands at {pos} on the {step}-unit grid: {on_grid}; "
                               f"with it off, a Blur stops where it is dropped ({off}): {loose}")


@probe("graph.clone")
def _graph_clone(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction
    _graph_fresh(bench)
    parent = _graph_node(bench, "Grade", 0, 0, {"gamma": 1.3})
    _graph_select(bench, parent)
    before = len(bench.window.dispatcher.document["nodes"])
    _graph_key(bench, Qt.Key.Key_K, Qt.KeyboardModifier.AltModifier)
    doc = bench.window.dispatcher.document
    clones = doc["settings"].get("clones") or {}
    made = [key for key, value in clones.items() if value == parent]
    clone = made[0] if len(made) == 1 else None
    made_one = clone is not None and len(doc["nodes"]) == before + 1 and doc["nodes"][clone]["type"] == "Grade" \
        and doc["nodes"][clone]["params"]["gamma"] == 1.3
    selected = bench.window.graph.selected_ids() == [clone]
    graph = bench.window.graph
    orange = [link for link in getattr(graph, "expression_links", ()) if getattr(link, "clone_link", False)
              and link.expression_link == (parent, clone)
              and link.pen().color().name() == graph.CLONE_LINK_COLOR]
    shared = False
    if clone:
        bench.window.command({"op": "set", "id": parent, "param": "gamma", "value": 1.8})
        follows = bench.params(clone)["gamma"] == 1.8
        bench.window.command({"op": "set", "id": clone, "param": "white", "value": 0.5})
        back = bench.params(parent)["white"] == 0.5
        bench.window.command({"op": "undo"})
        one_step = bench.params(parent)["white"] == 1.0 and bench.params(clone)["white"] == 1.0
        shared = follows and back and one_step
    bench.window.command({"op": "undo"})
    bench.window.command({"op": "undo"})
    undone = len(bench.window.dispatcher.document["nodes"]) == before and \
        not bench.window.dispatcher.document["settings"].get("clones")
    menu = all(bench.window.findChild(QAction, name) is not None for name in ("clone-action", "declone-action"))
    ok = made_one and selected and bool(orange) and shared and undone and menu
    return ok, (f"Alt+K on a Grade makes one Grade clone with its knobs: {made_one}; the clone is selected: {selected}; an "
                f"orange line joins it to its parent: {bool(orange)}; a knob edit on either changes both, undone in one "
                f"step: {shared}; one undo removes the clone and its link: {undone}; Edit > Clone and Declone: {menu}")


# -- the Curve Editor (lane 10, step T7, 2026-11-18) -------------------------------------------------------------------------
# NodeBased's curve canvas is the editor for curve knobs (ColorLookup's lookups); each probe opens it from the node's panel and
# drives it with Qt mouse and key events, then reads the document's stored curve and the undo step.

def _curve_bench(bench, points=((0, 0), (0.5, 0.4), (1, 1))):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton
    from nodebased.colorcurves import encode
    key = bench.node("ColorLookup", {"curve_master": encode(points)})
    bench.child("curve_master-editor", QPushButton).click()
    bench.settle()
    canvas = bench.window.curve_editor_dock.widget().canvas
    canvas.resize(500, 320)
    canvas.show()
    canvas.setFocus()
    bench.settle()
    return key, canvas


def _undo_back(bench, read, before):
    """One undo puts `read()` back to `before` (so the edit was one step); redo then restores the edit."""
    bench.window.dispatcher.execute({"op": "undo"}); bench.settle(8)
    back = read() == before
    bench.window.dispatcher.execute({"op": "redo"}); bench.settle(8)
    return back


def _stored_curve(bench, key):
    from nodebased.colorcurves import decode
    bench.settle(8)
    return decode(bench.params(key)["curve_master"])


def _drag(bench, widget, start, end, modifiers=None, button=None):
    from PySide6.QtCore import QEvent, Qt
    modifiers = modifiers or Qt.KeyboardModifier.NoModifier
    button = button or Qt.MouseButton.LeftButton
    _mouse(widget, QEvent.Type.MouseButtonPress, start, button, button, modifiers)
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    _mouse(widget, QEvent.Type.MouseMove, mid, Qt.MouseButton.NoButton, button, modifiers)
    _mouse(widget, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, button, modifiers)
    _mouse(widget, QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton, modifiers)
    bench.settle(8)


def _pt(canvas, x, y):
    q = canvas._xy(x, y)
    return (q.x(), q.y())


def _key(bench, widget, key, modifiers=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.keyClick(widget, key, modifiers or Qt.KeyboardModifier.NoModifier)
    bench.settle()


@probe("curve.add_key")
def _curve_add(bench):
    from PySide6.QtCore import QEvent, Qt
    key, canvas = _curve_bench(bench)
    before = bench.params(key)["curve_master"]
    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
    pos = _pt(canvas, 0.75, 0.2)                        # well off the curve
    _mouse(canvas, QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, mods)
    curve = _stored_curve(bench, key)
    added = [p for p in curve["points"] if abs(p[0] - 0.75) < 0.02]
    ok = len(curve["points"]) == 4 and added and abs(added[0][1] - 0.2) < 0.03
    one = _undo_back(bench, lambda: bench.params(key)["curve_master"], before)
    return bool(ok and one), f"keys {len(curve['points'])}, new key {added}, selected {canvas.selected}, one undo removes it {one}"


def _move_probe(bench, modifiers):
    key, canvas = _curve_bench(bench)
    before = bench.params(key)["curve_master"]
    _drag(bench, canvas, _pt(canvas, 0.5, 0.4), _pt(canvas, 0.62, 0.55), modifiers)
    curve = _stored_curve(bench, key)
    return curve["points"][1], 1 if _undo_back(bench, lambda: bench.params(key)["curve_master"], before) else 0


@probe("curve.move_one_axis")
def _curve_axis(bench):
    (x, y), undo = _move_probe(bench, None)
    ok = abs(y - 0.4) < 1e-6 and abs(x - 0.62) < 0.02 and undo == 1
    return ok, f"diagonal drag stored ({x:.3f}, {y:.3f}): the y stayed {abs(y - 0.4) < 1e-6}; undo steps +{undo}"


@probe("curve.move_ctrl_free")
def _curve_free(bench):
    from PySide6.QtCore import Qt
    (x, y), undo = _move_probe(bench, Qt.KeyboardModifier.ControlModifier)
    ok = abs(x - 0.62) < 0.02 and abs(y - 0.55) < 0.03 and undo == 1
    return ok, f"Ctrl+drag stored ({x:.3f}, {y:.3f}); undo steps +{undo}"


@probe("curve.frame_snap")
def _curve_snap(bench):
    from PySide6.QtCore import QEvent, Qt
    key, _sheet, _row = _dope_bench(bench)          # a Grade whose gamma is keyed on 5, 10 and 20
    editor = bench.window.show_keyframe_curve(key, "gamma")
    canvas = editor.canvas
    canvas.resize(500, 320); canvas.show(); bench.settle()
    before = repr(bench.window.node_curve(key, "gamma"))
    _drag(bench, canvas, _pt(canvas, 10, 2.0), _pt(canvas, 13.4, 2.0))
    snapped = _frames(bench, key)
    one = _undo_back(bench, lambda: repr(bench.window.node_curve(key, "gamma")), before)
    bench.window.dispatcher.execute({"op": "undo"}); bench.settle(8)
    editor = bench.window.show_keyframe_curve(key, "gamma")
    canvas = editor.canvas
    canvas.resize(500, 320); canvas.show(); bench.settle()
    shift = Qt.KeyboardModifier.ShiftModifier
    start, end = _pt(canvas, 10, 2.0), _pt(canvas, 13.4, 2.0)
    _mouse(canvas, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
           Qt.KeyboardModifier.NoModifier)
    _mouse(canvas, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, shift)
    free = canvas.curve["points"][1][0]
    _mouse(canvas, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, shift)
    bench.settle(8)
    ok = snapped == [5, 13, 20] and one and abs(free - 13.4) < 0.15
    return ok, (f"in the keyframe Curve Editor (knob menu > Curve editor), dragging the key on frame 10 to 13.4 stored "
                f"{snapped}, one undo {one}; with Shift it followed the pointer to {free:.2f} while dragging (the document "
                f"keeps whole-frame keys, so it lands on {_frames(bench, key)})")


@probe("curve.numeric_xy")
def _curve_numeric(bench):
    from PySide6.QtCore import Qt
    key, canvas = _curve_bench(bench)
    dialog = bench.window.curve_editor_dock.widget()
    canvas.select(1)
    before = bench.params(key)["curve_master"]
    box = dialog.point_y
    box.setFocus(); box.lineEdit().selectAll()
    from PySide6.QtTest import QTest
    QTest.keyClicks(box.lineEdit(), "0.7")
    QTest.keyClick(box.lineEdit(), Qt.Key.Key_Return)
    curve = _stored_curve(bench, key)
    one = _undo_back(bench, lambda: bench.params(key)["curve_master"], before)
    ok = abs(curve["points"][1][1] - 0.7) < 1e-6 and one
    return ok, f"typed y 0.7 in the Y field below the canvas: stored {curve['points'][1]}, one undo restores it {one}"


@probe("curve.tangent_handles")
def _curve_tangent(bench):
    key, canvas = _curve_bench(bench)
    canvas.select(1)
    bench.settle()
    handle = canvas._handle_pos(1, 1)
    before = canvas.handle_slope(1, 1)
    _drag(bench, canvas, (handle.x(), handle.y()), (handle.x(), handle.y() - 40))
    curve = _stored_curve(bench, key)
    after = curve["slopes"][1][1]
    return after > before + 0.1, f"right handle dragged up: slope {before:.3f} -> {after:.3f}"


@probe("curve.interpolation_menu")
def _curve_menu(bench):
    from nodebased.curveeditor import INTERPOLATION_MENU
    wanted = ("Constant", "Linear", "Smooth", "Catmull-Rom", "Cubic", "Horizontal", "Break")
    return tuple(INTERPOLATION_MENU) == wanted, f"right-click a key lists {', '.join(INTERPOLATION_MENU)}, then Delete key"


def _interp(bench, label, points=((0, 0), (0.5, 0.4), (1, 1)), index=1):
    key, canvas = _curve_bench(bench, points)
    before = bench.params(key)["curve_master"]
    canvas.set_interpolation(index, label)
    curve = _stored_curve(bench, key)
    return key, canvas, curve, 1 if _undo_back(bench, lambda: bench.params(key)["curve_master"], before) else 0


@probe("curve.interp_constant_linear")
def _curve_const(bench):
    from nodebased.colorcurves import evaluate
    _k, _c, const, u1 = _interp(bench, "Constant")
    _k, _c, linear, u2 = _interp(bench, "Linear")
    held = abs(evaluate(const, 0.75) - 0.4) < 1e-6
    straight = abs(evaluate(linear, 0.75) - 0.7) < 1e-6
    return held and straight and u1 == u2 == 1, f"Constant holds 0.4 after the key {held}; Linear is the straight line {straight}"


@probe("curve.interp_smooth")
def _curve_smooth(bench):
    _k, _c, between, _u = _interp(bench, "Smooth")
    _k, _c, peak, _u = _interp(bench, "Smooth", ((0, 0), (0.5, 0.9), (1, 0.2)))
    mid_slope = between["slopes"][1]
    flat = peak["slopes"][1]
    ok = abs(mid_slope[0] - 1.0) < 1e-6 and flat == [0.0, 0.0]
    return ok, f"key between its neighbours takes their slope {mid_slope[0]:.3f}; a peak key goes flat {flat}"


@probe("curve.interp_catmull_cubic")
def _curve_cubic(bench):
    from nodebased.curveeditor import cubic_slopes
    pts = ((0, 0), (0.5, 0.9), (1, 0.2))
    _k, _c, cr, _u = _interp(bench, "Catmull-Rom", pts)
    _k, _c, cu, _u = _interp(bench, "Cubic", pts)
    expect_cr = (0.2 - 0.0) / 1.0
    expect_cu = cubic_slopes([list(p) for p in pts])[1]
    ok = abs(cr["slopes"][1][0] - expect_cr) < 1e-6 and abs(cu["slopes"][1][0] - expect_cu) < 1e-6
    return ok, f"Catmull-Rom slope {cr['slopes'][1][0]:.3f} (neighbour chord {expect_cr:.3f}); Cubic {cu['slopes'][1][0]:.3f}"


@probe("curve.interp_horizontal")
def _curve_horizontal(bench):
    _k, _c, curve, undo = _interp(bench, "Horizontal")
    return curve["slopes"][1] == [0.0, 0.0] and undo == 1, f"slopes {curve['slopes'][1]}, undo steps +{undo}"


@probe("curve.interp_break")
def _curve_break(bench):
    key, canvas, curve, _u = _interp(bench, "Break")
    canvas.select(1); bench.settle()
    handle = canvas._handle_pos(1, 1)
    left = canvas.handle_slope(1, 0)
    _drag(bench, canvas, (handle.x(), handle.y()), (handle.x(), handle.y() - 40))
    after = _stored_curve(bench, key)
    ok = curve["broken"][1] and abs(after["slopes"][1][0] - left) < 1e-6 and after["slopes"][1][1] != after["slopes"][1][0]
    return bool(ok), f"broken {curve['broken'][1]}; dragging the right handle left the left slope at {after['slopes'][1][0]:.3f}"


@probe("curve.zoom")
def _curve_zoom(bench):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    _key_id, canvas = _curve_bench(bench)
    start = canvas.zoom
    centre = QPointF(canvas.width() / 2, canvas.height() / 2)
    QApplication.sendEvent(canvas, QWheelEvent(centre, canvas.mapToGlobal(centre), QPoint(0, 0), QPoint(0, 120),
                                               Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                                               Qt.ScrollPhase.NoScrollPhase, False))
    wheel = canvas.zoom
    _key(bench, canvas, Qt.Key.Key_Plus)
    plus = canvas.zoom
    _key(bench, canvas, Qt.Key.Key_Minus)
    minus = canvas.zoom
    return wheel > start and plus > wheel and minus < plus, f"zoom {start:.2f} -> wheel {wheel:.2f} -> + {plus:.2f} -> - {minus:.2f}"


@probe("curve.pan")
def _curve_pan(bench):
    from PySide6.QtCore import Qt
    key, canvas = _curve_bench(bench)
    before = bench.params(key)["curve_master"]
    _drag(bench, canvas, (300, 250), (340, 230), button=Qt.MouseButton.MiddleButton)
    middle = (canvas.offset.x(), canvas.offset.y())
    canvas.frame_all()
    _drag(bench, canvas, _pt(canvas, 0.5, 0.4), (_pt(canvas, 0.5, 0.4)[0] + 40, 120), Qt.KeyboardModifier.AltModifier)
    alt = (canvas.offset.x(), canvas.offset.y())
    unchanged = bench.params(key)["curve_master"] == before
    return middle != (0, 0) and alt != (0, 0) and unchanged, f"middle-drag offset {middle}, Alt+drag offset {alt}; curve unchanged {unchanged}"


@probe("curve.frame_all")
def _curve_frame_all(bench):
    from PySide6.QtCore import QPointF, Qt
    _k, canvas = _curve_bench(bench)
    canvas.zoom, canvas.offset = 3.0, QPointF(0.4, -0.2)
    _key(bench, canvas, Qt.Key.Key_A)
    return canvas.zoom == 1.0 and canvas.offset == QPointF(0, 0), f"after A: zoom {canvas.zoom}, offset {canvas.offset.toTuple()}"


@probe("curve.frame_selected")
def _curve_frame_sel(bench):
    from PySide6.QtCore import Qt
    _k, canvas = _curve_bench(bench, ((0, 0), (0.2, 0.9), (1, 1)))
    canvas.select(1)
    _key(bench, canvas, Qt.Key.Key_F)
    xmin, xmax, ymin, ymax = canvas._bounds()
    centred = abs((xmin + xmax) / 2 - 0.2) < 1e-6 and abs((ymin + ymax) / 2 - 0.9) < 1e-6
    return centred, f"after F the view centre is ({(xmin + xmax) / 2:.3f}, {(ymin + ymax) / 2:.3f}), the key is (0.2, 0.9)"


@probe("curve.select_several")
def _curve_several(bench):
    from PySide6.QtCore import Qt
    pts = ((0, 0), (0.25, 0.2), (0.5, 0.4), (0.75, 0.6), (1, 1))
    key, canvas = _curve_bench(bench, pts)
    shift = Qt.KeyboardModifier.ShiftModifier
    _click(bench, canvas, _pt(canvas, 0.25, 0.2))
    _click(bench, canvas, _pt(canvas, 0.5, 0.4), shift)
    clicked = set(canvas.chosen)
    a, b = _pt(canvas, 0.4, 0.1), _pt(canvas, 0.85, 0.7)
    _drag(bench, canvas, a, b)                                  # a marquee over the keys on 0.5 and 0.75
    swept = set(canvas.chosen)
    before = bench.params(key)["curve_master"]
    _drag(bench, canvas, _pt(canvas, 0.5, 0.4), _pt(canvas, 0.5, 0.6))   # drag one of them up: both move
    curve = _stored_curve(bench, key)
    ys = [round(p[1], 3) for p in curve["points"]]
    one = _undo_back(bench, lambda: bench.params(key)["curve_master"], before)
    ok = (clicked == {1, 2} and swept == {2, 3} and abs(ys[2] - 0.6) < 0.03 and abs(ys[3] - 0.8) < 0.03
          and ys[1] == 0.2 and one)
    return ok, (f"Shift+click selected keys {sorted(clicked)}; a marquee selected {sorted(swept)}; dragging one up moved "
                f"both: values {ys}, one undo {one}")


@probe("curve.select_all")
def _curve_all(bench):
    from PySide6.QtCore import Qt
    _k, canvas = _curve_bench(bench)
    _key(bench, canvas, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    return canvas.chosen == {0, 1, 2}, f"Ctrl/Cmd+A selected keys {sorted(canvas.chosen)} of 3"


@probe("curve.several_curves")
def _curve_curves(bench):
    from PySide6.QtCore import QEvent, Qt
    from nodebased.colorcurves import encode
    key, canvas = _curve_bench(bench)
    dialog = bench.window.curve_editor_dock.widget()
    listing = dialog.curve_list
    listing.show(); listing.resize(130, 200); bench.settle()
    names = [listing.item(i).text() for i in range(listing.count())]
    def pick(row, modifiers):
        from PySide6.QtTest import QTest
        QTest.mouseClick(listing.viewport(), Qt.MouseButton.LeftButton, modifiers,
                         listing.visualItemRect(listing.item(row)).center())
        bench.settle()
    pick(0, Qt.KeyboardModifier.NoModifier)                                                # Master
    pick(2, Qt.KeyboardModifier.ShiftModifier)                    # Shift: Master..Green
    run = dialog.shown_curves()
    pick(4, Qt.KeyboardModifier.ControlModifier)                  # and Alpha
    picked = dialog.shown_curves()
    overlays = len(dialog.canvas.overlays)
    active = dialog._active
    before = bench.params(key)["curve_alpha"]
    dialog.canvas.select(0)
    dialog.canvas.set_point(0, 0.0, 0.3)
    bench.settle(8)
    edited = bench.params(key)["curve_alpha"] != before and bench.params(key)["curve_master"] == encode(((0, 0), (0.5, 0.4), (1, 1)))
    ok = (names[:5] == ["Master", "Red", "Green", "Blue", "Alpha"] and run == ["curve_master", "curve_red", "curve_green"]
          and picked == ["curve_master", "curve_red", "curve_green", "curve_alpha"] and overlays == 3
          and active == "curve_alpha" and edited)
    return ok, (f"the list holds {names}; Shift+click showed {run} curves, Ctrl+click one more ({len(picked)}, "
                f"{overlays} drawn under the one being edited, {active}); an edit changed only that knob {edited}")


@probe("curve.move_dialog")
def _curve_move(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    key, canvas = _curve_bench(bench, ((0, 0), (0.3, 0.2), (0.5, 0.4), (1, 1)))
    canvas.select(1); canvas.select(2, add=True)
    before = bench.params(key)["curve_master"]
    dialog = canvas.open_move_dialog()
    for field, text in ((dialog.x_field, "x+0.1"), (dialog.y_field, "y*2")):
        field.selectAll(); QTest.keyClicks(field, text)
    QTest.keyClick(dialog.y_field, Qt.Key.Key_Return)
    curve = _stored_curve(bench, key)
    got = [[round(v, 4) for v in p] for p in curve["points"]]
    one = _undo_back(bench, lambda: bench.params(key)["curve_master"], before)
    ok = got == [[0, 0], [0.4, 0.4], [0.6, 0.8], [1, 1]] and one and not dialog.isVisible()
    return ok, f"Edit > Move with x+0.1 and y*2 on the two selected keys stored {got}; one undo restores them {one}"


@probe("curve.expression_field")
def _curve_expression(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from nodebased.colorcurves import evaluate
    key, canvas = _curve_bench(bench)
    dialog = bench.window.curve_editor_dock.widget()
    before = bench.params(key)["curve_master"]
    QTest.keyClicks(dialog.expression, "pow(x, 2)")
    QTest.keyClick(dialog.expression, Qt.Key.Key_Return)
    curve = _stored_curve(bench, key)
    err = max(abs(evaluate(curve, x / 40) - (x / 40) ** 2) for x in range(41))
    one = _undo_back(bench, lambda: bench.params(key)["curve_master"], before)
    QTest.mouseClick(dialog.revert, Qt.MouseButton.LeftButton)
    bench.settle(8)
    reverted = bench.params(key)["curve_master"] == before
    ok = err < 1e-3 and one and reverted
    return ok, (f"typing pow(x, 2) in the expression field made the curve follow x squared to {err:.1e}; one undo "
                f"{one}; Revert put the keys back {reverted}")


# -- the Dope Sheet (lane 10, step T7) ----------------------------------------------------------------------------------------

def _dope_bench(bench, frames=(5, 10, 20)):
    window = bench.window
    key = bench.node("Grade")
    window.dispatcher.execute({"op": "time", "first": 1, "last": 40, "current": 1})
    for i, frame in enumerate(frames):
        window.dispatcher.execute({"op": "set_key", "id": key, "param": "gamma", "frame": frame, "value": 1.0 + i})
        window.dispatcher.execute({"op": "set_key", "id": key, "param": "multiply", "frame": frame, "value": 2.0 + i})
    window.dope_sheet_dock.show()
    sheet = window.dope_sheet
    sheet.setMinimumWidth(800)          # the dock opens narrow; give each frame room to click
    bench.settle()
    sheet.resize(max(sheet.width(), 800), 200)
    sheet.refresh()
    bench.settle()
    row = next(i for i, r in enumerate(sheet.rows) if r[0] == key)
    return key, sheet, row


def _dope_point(sheet, row, frame):
    point = dict(sheet.key_points(row))[frame]
    return (point.x(), point.y())


def _frames(bench, key, param="gamma"):
    curve = bench.window.node_curve(key, param) or {"keys": []}
    return [k["frame"] for k in curve["keys"]]


def _click(bench, widget, pos, modifiers=None):
    from PySide6.QtCore import QEvent, Qt
    modifiers = modifiers or Qt.KeyboardModifier.NoModifier
    _mouse(widget, QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, modifiers)
    _mouse(widget, QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, modifiers)
    bench.settle()


@probe("dope.key_markers")
def _dope_markers(bench):
    key, sheet, row = _dope_bench(bench)
    frames = [f for f, _p in sheet.key_points(row)]
    return frames == [5, 10, 20], f"the Grade's row has markers on {frames}"


@probe("dope.current_frame")
def _dope_current(bench):
    _key_id, sheet, _row = _dope_bench(bench)
    bench.window.command({"op": "time", "current": 12})
    bench.settle(); sheet.refresh()
    return sheet.current == 12, f"the line is drawn at the current frame {sheet.current} (NodeBased draws it in the accent blue)"


@probe("dope.project_range")
def _dope_range(bench):
    _key_id, sheet, _row = _dope_bench(bench)
    first_x, last_x = sheet.range_lines()
    lo, hi = sheet.view_range()
    inside = sheet.frame_x(lo) < first_x < last_x < sheet.frame_x(hi)
    exact = abs(first_x - sheet.frame_x(1)) < 1e-9 and abs(last_x - sheet.frame_x(40)) < 1e-9
    return inside and exact, (f"the project's frames 1 and 40 have lines at x {first_x:.0f} and {last_x:.0f}, inside the "
                              f"time area that spans frames {lo} to {hi}")


@probe("dope.select_key")
def _dope_select(bench):
    key, sheet, row = _dope_bench(bench)
    _click(bench, sheet, _dope_point(sheet, row, 10))
    return sheet.selected == {(key, 10)}, f"selected {sorted(sheet.selected)}; its frame number is drawn beside it"


@probe("dope.marquee")
def _dope_marquee(bench):
    from PySide6.QtCore import Qt
    key, sheet, row = _dope_bench(bench)
    y = _dope_point(sheet, row, 10)[1]
    _drag(bench, sheet, (sheet.frame_x(7), y - 9), (sheet.frame_x(25), y + 9))
    swept = set(sheet.selected)
    _drag(bench, sheet, (sheet.frame_x(3), y - 9), (sheet.frame_x(7), y + 9), Qt.KeyboardModifier.ShiftModifier)
    added = set(sheet.selected)
    ok = swept == {(key, 10), (key, 20)} and added == {(key, 5), (key, 10), (key, 20)}
    return ok, (f"a marquee from frame 7 to 25 selected {sorted(f for _k, f in swept)}; a Shift marquee over frame 5 made it "
                f"{sorted(f for _k, f in added)}")


@probe("dope.name_selects_keys")
def _dope_name(bench):
    key, sheet, row = _dope_bench(bench)
    _click(bench, sheet, (20, sheet.key_points(row)[0][1].y()))
    return sheet.selected == {(key, 5), (key, 10), (key, 20)}, f"clicking the name selected {sorted(f for _k, f in sheet.selected)}"


@probe("dope.delete_key")
def _dope_delete(bench):
    from PySide6.QtCore import Qt
    key, sheet, row = _dope_bench(bench)
    _click(bench, sheet, _dope_point(sheet, row, 10))
    before = repr(bench.window.dispatcher.document["animation"]["curves"].get(key))
    _key(bench, sheet, Qt.Key.Key_Delete)
    bench.settle(8)
    g, m = _frames(bench, key), _frames(bench, key, "multiply")
    one = _undo_back(bench, lambda: repr(bench.window.dispatcher.document["animation"]["curves"].get(key)), before)
    ok = g == [5, 20] and m == [5, 20] and one
    return ok, f"after Delete: gamma {g}, multiply {m}, one undo brings both back {one}"


@probe("dope.add_key")
def _dope_add(bench):
    from PySide6.QtCore import Qt
    key, sheet, row = _dope_bench(bench)
    before = repr(bench.window.dispatcher.document["animation"]["curves"].get(key))
    y = sheet.key_points(row)[0][1].y()
    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
    _click(bench, sheet, (sheet.frame_x(15), y), mods)
    bench.settle(8)
    g = _frames(bench, key)
    value = [k["value"] for k in bench.window.node_curve(key, "gamma")["keys"] if k["frame"] == 15]
    one = _undo_back(bench, lambda: repr(bench.window.dispatcher.document["animation"]["curves"].get(key)), before)
    ok = g == [5, 10, 15, 20] and value and abs(value[0] - 2.5) < 1e-6 and one
    return bool(ok), f"Ctrl+Alt+click at frame 15 keyed {g} at the curve's value {value}, one undo removes it {one}"


@probe("dope.no_passing")
def _dope_no_pass(bench):
    key, sheet, row = _dope_bench(bench)
    start = _dope_point(sheet, row, 10)
    _click(bench, sheet, start)
    before = repr(bench.window.dispatcher.document["animation"]["curves"].get(key))
    _drag(bench, sheet, start, (sheet.frame_x(30), start[1]))     # past the key on 20
    g = _frames(bench, key)
    one = _undo_back(bench, lambda: repr(bench.window.dispatcher.document["animation"]["curves"].get(key)), before)
    ok = g == [5, 19, 20] and one
    return ok, f"dragging frame 10 toward 30 stopped next to the key on 20: {g}, one undo moves it back {one}"


@probe("dope.move_field")
def _dope_move_field(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    key, sheet, row = _dope_bench(bench)
    panel = bench.window.dope_sheet_panel
    _click(bench, sheet, _dope_point(sheet, row, 10))
    before = repr(bench.window.dispatcher.document["animation"]["curves"].get(key))
    field = panel.move_field
    field.lineEdit().selectAll()
    QTest.keyClicks(field.lineEdit(), "4")
    QTest.keyClick(field.lineEdit(), Qt.Key.Key_Return)
    QTest.mouseClick(panel.forward, Qt.MouseButton.LeftButton)
    bench.settle(8)
    forward = _frames(bench, key)
    one = _undo_back(bench, lambda: repr(bench.window.dispatcher.document["animation"]["curves"].get(key)), before)
    bench.window.dispatcher.execute({"op": "undo"}); bench.settle(8)
    sheet.refresh(); sheet.selected = {(key, 10)}
    field.setValue(-3)
    QTest.mouseClick(panel.forward, Qt.MouseButton.LeftButton)
    bench.settle(8)
    backward = _frames(bench, key)
    ok = forward == [5, 14, 20] and backward == [5, 7, 20] and one
    return ok, (f"Move 4 then > moved frame 10 to {forward}, one undo moves it back {one}; Move -3 then > gave {backward}")


@probe("dope.copy_paste")
def _dope_copy(bench):
    from PySide6.QtCore import Qt
    key, sheet, row = _dope_bench(bench)
    _click(bench, sheet, _dope_point(sheet, row, 5))
    _click(bench, sheet, _dope_point(sheet, row, 10), Qt.KeyboardModifier.ShiftModifier)
    _key(bench, sheet, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    bench.window.command({"op": "time", "current": 30})
    bench.settle(); sheet.refresh()
    before = repr(bench.window.dispatcher.document["animation"]["curves"].get(key))
    _key(bench, sheet, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    bench.settle(8)
    g = _frames(bench, key)
    values = {k["frame"]: k["value"] for k in bench.window.node_curve(key, "multiply")["keys"]}
    one = _undo_back(bench, lambda: repr(bench.window.dispatcher.document["animation"]["curves"].get(key)), before)
    ok = g == [5, 10, 20, 30, 35] and values.get(30) == 2.0 and values.get(35) == 3.0 and one
    return ok, (f"copied keys 5 and 10, pasted with the current frame on 30: gamma keys {g}, multiply on 30 and 35 "
                f"{values.get(30)}, {values.get(35)}; one undo removes the paste {one}")


@probe("dope.clip_bars")
def _dope_bars(bench):
    from PySide6.QtCore import Qt
    window = bench.window
    window.dispatcher.execute({"op": "time", "first": 1, "last": 40, "current": 1})
    key = bench.node("Read", {"first": 1, "last": 20})
    window.pinned_panels.insert(0, key)                   # a Read with an open (pinned) panel gets a row
    window.dope_sheet_dock.show()
    sheet = window.dope_sheet
    sheet.setMinimumWidth(800)
    bench.settle()
    sheet.resize(max(sheet.width(), 800), 200)
    sheet.rows = []; sheet.refresh(); bench.settle()
    row = next(i for i, r in enumerate(sheet.rows) if r[0] == key)
    bar = sheet.clip_bar(row)
    before = bench.params(key)["frame_offset"]
    y = sheet._clip_rect(row).center().y()
    _drag(bench, sheet, (sheet.frame_x(10), y), (sheet.frame_x(15), y))
    after = bench.params(key)["frame_offset"]
    moved = sheet.clip_bar(row)
    one = _undo_back(bench, lambda: bench.params(key)["frame_offset"], before)
    window.pinned_panels.remove(key)
    ok = bar == (1, 20) and after == -5 and moved == (6, 25) and one
    return ok, (f"the Read's clip shows as a gray bar on frames {bar}; dragging it 5 frames later set its frame offset "
                f"{before} -> {after} (bar now {moved}), one undo puts it back {one}")

# -- run -------------------------------------------------------------------------------------------------------------------

# -- the Viewer (lane 10, step T8, 2026-11-18) -------------------------------------------------------------------------------
# Keys and mouse on NodeBased's real Viewer (window.viewer), against the Nuke 17.0 User Guide's Viewers and Using the Viewer
# Controls pages.

def _viewer(bench, inputs=1):
    """The Viewer showing a Constant, with `inputs` Constants wired to Viewer inputs 1..inputs; focus on the Viewer."""
    window = bench.window
    keys = []
    for _ in range(inputs):
        keys.append(bench.node("Constant"))
    window.command({"op": "view", "id": keys[0]})
    for slot, key in enumerate(keys[1:], 2):
        window.command({"op": "viewer_input", "slot": slot, "id": key})
    window.command({"op": "viewer_input", "slot": 1, "id": keys[0]})
    viewer = window.viewer
    viewer.setFocus()
    bench.settle(8)
    return viewer, keys


def _key(bench, widget, key, modifiers=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.keyClick(widget, key, modifiers if modifiers is not None else Qt.KeyboardModifier.NoModifier)
    bench.settle()


# Lane 10 step T9: the 2D Compositing Viewer keys from the Nuke 17.0 User Guide's Keyboard Shortcuts page.

@probe("viewer.jkl")
def _viewer_jkl(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    window = bench.window
    seen = []
    for key in (Qt.Key.Key_L, Qt.Key.Key_J, Qt.Key.Key_K, Qt.Key.Key_J, Qt.Key.Key_K):
        _key(bench, viewer, key)
        seen.append((window.playing, window.play_direction if window.playing else 0))
    window.toggle_playback(False)
    ok = seen == [(True, 1), (True, -1), (False, 0), (True, -1), (False, 0)]
    return ok, f"L, J (while playing forward), K, J, K gave {seen}"


@probe("viewer.wipe")
def _viewer_wipe(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench, 2)
    bench.window.command({"op": "viewer_compare", "b": None, "mode": "A only"})
    bench.settle()
    _key(bench, viewer, Qt.Key.Key_W)
    on = viewer.compare_mode()
    _key(bench, viewer, Qt.Key.Key_W)
    off = viewer.compare_mode()
    bench.window.command({"op": "viewer_compare", "b": None, "mode": "A only"})
    return (on, off) == ("wipe", "A only"), f"W then W gave {on} then {off}"


@probe("viewer.proxy_roi")
def _viewer_proxy_roi(bench):
    from PySide6.QtCore import Qt
    from nodebased.core import viewer_proxy, viewer_roi
    viewer, _ = _viewer(bench)
    window = bench.window
    window.set_viewer_roi(on=False)
    before = viewer_proxy(window.dispatcher.document)
    _key(bench, viewer, Qt.Key.Key_P, Qt.KeyboardModifier.ControlModifier)
    toggled = viewer_proxy(window.dispatcher.document)
    _key(bench, viewer, Qt.Key.Key_P, Qt.KeyboardModifier.ControlModifier)
    back = viewer_proxy(window.dispatcher.document)
    _key(bench, viewer, Qt.Key.Key_W, Qt.KeyboardModifier.AltModifier)
    roi = viewer_roi(window.dispatcher.document)
    window.set_viewer_roi(on=False)
    ok = toggled != before and back == before and roi["on"]
    return ok, f"Ctrl+P proxy {before} -> {toggled} -> {back}; Alt+W region on {roi['on']} at {roi['rect']}"


@probe("viewer.luminance_y")
def _viewer_luminance_y(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    bench.window.channels.setCurrentText("RGB")
    _key(bench, viewer, Qt.Key.Key_Y)
    shown = bench.window.channels.currentText()
    bench.window.channels.setCurrentText("RGB")
    return shown in ("Y", "Luminance"), f"Y showed {shown}; the Viewer's channel choices have no luminance display"


@probe("viewer.channel_keys")
def _viewer_channel_keys(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    shown = {}
    for name, key in (("R", Qt.Key.Key_R), ("G", Qt.Key.Key_G), ("B", Qt.Key.Key_B), ("A", Qt.Key.Key_A)):
        bench.window.channels.setCurrentText("RGB")
        _key(bench, viewer, key)
        shown[name] = bench.window.channels.currentText()
    bench.window.channels.setCurrentText("RGB")
    ok = all(shown[n] == n for n in shown)
    return ok, f"R, G, B and A showed {', '.join(shown[n] for n in 'RGBA')}"


@probe("viewer.channel_toggle_back")
def _viewer_channel_back(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    bench.window.channels.setCurrentText("RGB")
    _key(bench, viewer, Qt.Key.Key_G)
    first = bench.window.channels.currentText()
    _key(bench, viewer, Qt.Key.Key_G)
    second = bench.window.channels.currentText()
    return (first, second) == ("G", "RGB"), f"G then G again showed {first} then {second}"


@probe("viewer.input_numbers")
def _viewer_input_numbers(bench):
    from PySide6.QtCore import Qt
    viewer, keys = _viewer(bench, 3)
    _key(bench, viewer, Qt.Key.Key_3)
    third = bench.window.dispatcher.document.get("view")
    _key(bench, viewer, Qt.Key.Key_1)
    first = bench.window.dispatcher.document.get("view")
    ok = third == keys[2] and first == keys[0]
    return ok, f"3 showed input 3 ({third == keys[2]}), 1 showed input 1 ({first == keys[0]})"


@probe("viewer.input_arrows")
def _viewer_input_arrows(bench):
    from PySide6.QtCore import Qt
    viewer, keys = _viewer(bench, 3)
    seen = []
    for key in (Qt.Key.Key_Down, Qt.Key.Key_Down, Qt.Key.Key_Down, Qt.Key.Key_Up):
        _key(bench, viewer, key)
        view = bench.window.dispatcher.document.get("view")
        seen.append(keys.index(view) + 1 if view in keys else None)
    return seen == [2, 3, 1, 3], f"Down, Down, Down, Up showed inputs {seen}"


def _view_center(viewer):
    return viewer.mapToScene(viewer.viewport().rect().center())


@probe("viewer.pan")
def _viewer_pan(bench):
    from PySide6.QtCore import QEvent, Qt
    viewer, _ = _viewer(bench)
    viewer.resetTransform()
    bench.settle()
    results = {}
    for name, button, mods in (("middle", Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier),
                               ("Alt", Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier)):
        before = _view_center(viewer)
        port = viewer.viewport()
        _mouse(port, QEvent.Type.MouseButtonPress, (300, 300), button, button, mods)
        for x in (320, 340, 360):
            _mouse(port, QEvent.Type.MouseMove, (x, 300), Qt.MouseButton.NoButton, button, mods)
        _mouse(port, QEvent.Type.MouseButtonRelease, (360, 300), button, Qt.MouseButton.NoButton, mods)
        bench.settle()
        after = _view_center(viewer)
        results[name] = after.x() - before.x()
    ok = all(dx < -10 for dx in results.values())         # the frame follows the pointer: the view centre moves left
    return ok, f"a 60 pixel drag to the right moved the view centre by {results['middle']:.0f} (middle) and {results['Alt']:.0f} (Alt)"


@probe("viewer.fit_f")
def _viewer_fit(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    viewer.resetTransform()
    viewer.scale(4.0, 4.0)
    bench.settle()
    _key(bench, viewer, Qt.Key.Key_F)
    bench.settle(12)
    zoom, fitted = viewer.transform().m11(), bool(viewer.auto_fit)   # auto_fit is set by the Viewer's fit command alone
    ok = fitted and zoom < 4.0
    return ok, f"F at 4x zoom ran the Viewer's fit {fitted}, zoom now {zoom:.2f}"


@probe("viewer.zoom_keys")
def _viewer_zoom_keys(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    viewer.resetTransform()
    bench.settle()
    start = viewer.transform().m11()
    _key(bench, viewer, Qt.Key.Key_Plus)
    _key(bench, viewer, Qt.Key.Key_Plus)
    zoomed = viewer.transform().m11()
    _key(bench, viewer, Qt.Key.Key_Minus)
    back = viewer.transform().m11()
    ok = zoomed > start * 1.2 and start < back < zoomed
    return ok, f"+ + took the zoom from {start:.2f} to {zoomed:.2f}, - back to {back:.2f}"


@probe("viewer.one_to_one")
def _viewer_one_to_one(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    viewer.scale(2.5, 2.5)
    bench.settle()
    _key(bench, viewer, Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier)
    zoom = viewer.transform().m11()
    return abs(zoom - 1.0) < 1e-6, f"Ctrl+1 set the zoom to {zoom:.3f}"



def _ramp_viewer(bench):
    """The Viewer showing a 64 x 32 horizontal 0 to 1 ramp at 1:1, centred; returns (viewer, float frame) or (viewer, None)."""
    import time
    import numpy as np
    viewer = bench.window.viewer
    viewer.resetTransform()                 # the whole frame on screen, so the Viewer renders all of it
    viewer.fit()
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
    viewer.info_sample = None
    viewer.info_sample_region = None
    # Several screen pixels per image pixel, so a click lands on the pixel named, with the whole frame still on screen.
    port = viewer.viewport().rect()
    scale = max(float(viewer.last_scale or 1.0), 1.0)
    zoom = max(1.0, min(4.0, 0.8 * port.width() / (64 * scale), 0.8 * port.height() / (32 * scale)))
    viewer.auto_fit = False
    viewer.resetTransform()
    viewer.scale(zoom, zoom)
    viewer.centerOn(32.0 * scale, 16.0 * scale)
    viewer.setFocus()
    bench.settle(8)
    return viewer, frame


def _port_point(viewer, x, y):
    """Viewport coordinates of the ramp's pixel centre (x, y), counted from the top-left as the frame array is."""
    from PySide6.QtCore import QPointF
    scale = max(float(viewer.last_scale or 1.0), 1.0)
    point = viewer.mapFromScene(QPointF((x + 0.5) * scale, (y + 0.5) * scale))
    return (float(point.x()), float(point.y()))


def _viewer_hover(bench, viewer, x, y):
    from PySide6.QtCore import QEvent, Qt
    _mouse(viewer.viewport(), QEvent.Type.MouseMove, _port_point(viewer, x, y), Qt.MouseButton.NoButton,
           Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    bench.settle()


def _viewer_click(bench, viewer, x, y, button, modifiers, drag_to=None):
    from PySide6.QtCore import QEvent, Qt
    port = viewer.viewport()
    start = _port_point(viewer, x, y)
    _mouse(port, QEvent.Type.MouseButtonPress, start, button, button, modifiers)
    end = start
    if drag_to is not None:
        end = _port_point(viewer, *drag_to)
        _mouse(port, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, button, modifiers)
    _mouse(port, QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton, modifiers)
    bench.settle()


def _viewer_readout(viewer):
    shown = viewer.pixel_readout.isVisibleTo(viewer.viewport())
    return shown, viewer.pixel_readout.position.text(), viewer.pixel_readout.values.text()


@probe("viewer.info_bar")
def _viewer_info_bar(bench):
    viewer, frame = _ramp_viewer(bench)
    if frame is None:
        return False, "the ramp never reached the viewer"
    _viewer_hover(bench, viewer, 40, 5)
    shown, position, values = _viewer_readout(viewer)
    red = f"{float(frame[5, 40, 0]):.3f}"
    ok = shown and position.startswith("x 40") and values.startswith(red)
    return ok, f"hovering pixel 40 showed {position!r} {values!r} (red {red}, readout shown {shown})"


@probe("viewer.sample_pixel")
def _viewer_sample_pixel(bench):
    from PySide6.QtCore import Qt
    viewer, frame = _ramp_viewer(bench)
    if frame is None:
        return False, "the ramp never reached the viewer"
    _viewer_click(bench, viewer, 20, 5, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    _viewer_hover(bench, viewer, 55, 5)            # the sample stays in the info bar while the pointer moves on
    _, position, values = _viewer_readout(viewer)
    red = f"{float(frame[5, 20, 0]):.3f}"
    held = viewer.info_sample is not None and viewer.info_sample["kind"] == "pixel"
    ok = held and values.startswith(red) and "20" in position
    viewer.cancel_info_sample()
    return ok, f"Ctrl+click on pixel 20, then a move to 55, showed {position!r} {values!r} (pixel red {red})"


@probe("viewer.sample_region")
def _viewer_sample_region(bench):
    from PySide6.QtCore import Qt
    viewer, frame = _ramp_viewer(bench)
    if frame is None:
        return False, "the ramp never reached the viewer"
    mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    _viewer_click(bench, viewer, 0, 0, Qt.MouseButton.LeftButton, mods, drag_to=(63, 31))
    sample = viewer.info_sample or {}
    expected = float(frame[:, :, 0].mean())
    got = sample.get("rgba", (None,))[0]
    ok = sample.get("kind") == "region" and got is not None and abs(got - expected) < 1e-3
    viewer.cancel_info_sample()
    return ok, f"Ctrl+Shift+drag over the whole ramp held red {got} (frame mean {expected:.4f})"


@probe("viewer.sample_cancel")
def _viewer_sample_cancel(bench):
    from PySide6.QtCore import Qt
    viewer, frame = _ramp_viewer(bench)
    if frame is None:
        return False, "the ramp never reached the viewer"
    _viewer_click(bench, viewer, 20, 5, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    held = viewer.info_sample is not None
    _viewer_click(bench, viewer, 30, 5, Qt.MouseButton.RightButton, Qt.KeyboardModifier.ControlModifier)
    cleared = viewer.info_sample is None
    _viewer_hover(bench, viewer, 50, 5)
    _, position, _ = _viewer_readout(viewer)
    ok = held and cleared and position.startswith("x 50")
    return ok, f"Ctrl+click held a sample {held}, Ctrl+right-click cleared it {cleared}, the readout follows again ({position!r})"

# -- the Viewer, part two (lane 10, step T8 finish, 2026-11-19) -------------------------------------------------------------

def _wait_for_picture(bench, before, seconds=30):
    """Settle until the Viewer shows a picture other than `before` (the QImage it showed); the new one or None."""
    import time
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        bench.settle(1)
        shown = getattr(bench.window.viewer, "last_display_image", None)
        if shown is not None and shown is not before:
            return shown
        time.sleep(0.01)
    return None


def _centre_rgb(image):
    colour = image.pixelColor(image.width() // 2, image.height() // 2)
    return colour.red(), colour.green(), colour.blue()


@probe("viewer.alpha_overlay")
def _viewer_alpha_overlay(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    window = bench.window
    window.alpha_overlay = False
    plain = _wait_for_picture(bench, None)
    if plain is None:
        return False, "the Constant never reached the viewer"
    before = _centre_rgb(plain)
    _key(bench, viewer, Qt.Key.Key_M)
    overlaid = _wait_for_picture(bench, plain)
    on = window.alpha_overlay
    during = _centre_rgb(overlaid) if overlaid is not None else None
    _key(bench, viewer, Qt.Key.Key_M)
    back = _wait_for_picture(bench, overlaid)
    after = _centre_rgb(back) if back is not None else None
    ok = (on and not window.alpha_overlay and during is not None and during[0] > before[0] + 60
          and during[2] < before[2] and after is not None and max(abs(a - b) for a, b in zip(after, before)) <= 1)
    return ok, f"M took the centre pixel from {before} to {during} (overlay {on}), M again to {after}"


@probe("viewer.recenter_middle_click")
def _viewer_recenter_middle(bench):
    from PySide6.QtCore import QEvent, Qt
    viewer, _ = _viewer(bench)
    viewer.resetTransform()
    viewer.scale(4.0, 4.0)
    viewer.auto_fit = False
    bench.settle()
    port = viewer.viewport()
    middle, none = Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton
    _mouse(port, QEvent.Type.MouseButtonPress, (200, 200), middle, middle, Qt.KeyboardModifier.NoModifier)
    _mouse(port, QEvent.Type.MouseButtonRelease, (200, 200), middle, none, Qt.KeyboardModifier.NoModifier)
    bench.settle(8)
    clicked = (bool(viewer.auto_fit), viewer.transform().m11())
    # A middle drag stays a pan: it does not re-centre.
    viewer.resetTransform()
    viewer.scale(4.0, 4.0)
    viewer.auto_fit = False
    _mouse(port, QEvent.Type.MouseButtonPress, (200, 200), middle, middle, Qt.KeyboardModifier.NoModifier)
    _mouse(port, QEvent.Type.MouseMove, (260, 200), none, middle, Qt.KeyboardModifier.NoModifier)
    _mouse(port, QEvent.Type.MouseButtonRelease, (260, 200), middle, none, Qt.KeyboardModifier.NoModifier)
    bench.settle()
    dragged = (bool(viewer.auto_fit), viewer.transform().m11())
    ok = clicked[0] and clicked[1] < 4.0 and not dragged[0] and abs(dragged[1] - 4.0) < 1e-6
    return ok, f"a middle click re-centred {clicked[0]} (zoom {clicked[1]:.2f}); a middle drag re-centred {dragged[0]}"


@probe("viewer.zoom_drag")
def _viewer_zoom_drag(bench):
    from PySide6.QtCore import QEvent, QPointF, Qt
    viewer, _ = _viewer(bench)
    viewer.resetTransform()
    bench.settle()
    port = viewer.viewport()
    M, L, none = Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    mods = Qt.KeyboardModifier.NoModifier
    start = viewer.transform().m11()
    anchor = viewer.mapToScene(300, 260)
    _mouse(port, QEvent.Type.MouseButtonPress, (300, 260), M, M, mods)
    _mouse(port, QEvent.Type.MouseButtonPress, (300, 260), L, M | L, mods)
    for x in (320, 340, 360):
        _mouse(port, QEvent.Type.MouseMove, (x, 260), none, M | L, mods)
    zoomed = viewer.transform().m11()
    held = viewer.mapFromScene(QPointF(anchor))
    for x in (340, 320):                  # dragging back zooms out again
        _mouse(port, QEvent.Type.MouseMove, (x, 260), none, M | L, mods)
    back = viewer.transform().m11()
    _mouse(port, QEvent.Type.MouseButtonRelease, (320, 260), L, M, mods)
    _mouse(port, QEvent.Type.MouseButtonRelease, (320, 260), M, none, mods)
    bench.settle()
    drift = abs(held.x() - 300) + abs(held.y() - 260)
    ok = zoomed > start * 1.5 and start < back < zoomed and drift <= 2 and not viewer.auto_fit
    return ok, (f"middle plus left dragged 60 pixels right took the zoom from {start:.2f} to {zoomed:.2f} (the point under "
                f"the pointer moved {drift} pixels), 40 back to {back:.2f}")


def _wait_for_format(bench, before, seconds=30):
    import time
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        bench.settle(1)
        rect = bench.window.viewer.format_rect
        if rect is not None and (before is None or abs(rect.width() - before) > 1):
            return rect.width()
        time.sleep(0.01)
    return None


@probe("viewer.prevent_auto_zoom")
def _viewer_prevent_auto_zoom(bench):
    from PySide6.QtCore import Qt
    from nodebased.core import viewer_knobs
    window = bench.window
    big = bench.node("Constant", {"width": 960, "height": 540})
    small = bench.node("Constant", {"width": 240, "height": 135})
    window.command({"op": "view", "id": big})
    window.command({"op": "viewer_input", "slot": 1, "id": big})
    window.command({"op": "viewer_input", "slot": 2, "id": small})
    viewer = window.viewer
    viewer.setFocus()
    width = _wait_for_format(bench, None)
    results = {}
    for name, lock in (("unlocked", False), ("locked", True)):
        if bool(viewer_knobs(window.dispatcher.document)["zoom_lock"]) != lock:
            _key(bench, viewer, Qt.Key.Key_Z, Qt.KeyboardModifier.AltModifier)
        viewer.resetTransform()
        viewer.scale(2.0, 2.0)
        viewer.auto_fit = False
        bench.settle()
        _key(bench, viewer, Qt.Key.Key_2 if window.dispatcher.document.get("view") == big else Qt.Key.Key_1)
        width = _wait_for_format(bench, width)
        bench.settle(4)
        results[name] = viewer.transform().m11()
    locked = bool(viewer_knobs(window.dispatcher.document)["zoom_lock"])
    _key(bench, viewer, Qt.Key.Key_Z, Qt.KeyboardModifier.AltModifier)
    unlocked_again = not viewer_knobs(window.dispatcher.document)["zoom_lock"]
    ok = (abs(results["unlocked"] - 2.0) > 1e-3 and abs(results["locked"] - 2.0) < 1e-6 and locked and unlocked_again)
    return ok, (f"switching to an input of another size at 2x zoom left the zoom at {results['unlocked']:.2f}, after Alt+Z at "
                f"{results['locked']:.2f}; Alt+Z again turned the lock off {unlocked_again}")


@probe("viewer.toolbars")
def _viewer_toolbars(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    window = bench.window
    seen = []
    for key, widget in ((Qt.Key.Key_BraceLeft, window.viewer_strip), (Qt.Key.Key_BraceRight, window.time_row)):
        _key(bench, viewer, key, Qt.KeyboardModifier.ShiftModifier)
        hidden = widget.isHidden()
        _key(bench, viewer, key, Qt.KeyboardModifier.ShiftModifier)
        seen.append((hidden, widget.isHidden()))
    ok = seen == [(True, False), (True, False)]
    return ok, f"{{ hid then showed the top toolbar {seen[0]}, }} the bottom one {seen[1]}"


@probe("viewer.gain_gamma_toggle")
def _viewer_gain_gamma_toggle(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    _viewer(bench)
    window, strip = bench.window, bench.window.viewer_strip
    seen = {}
    for name, spin, button, value, neutral in (("gain", window.exposure, strip.sun, 1.5, 0.0),
                                               ("gamma", window.gamma, strip.gamma_mark, 1.8, 1.0)):
        spin.setValue(value)
        bench.settle()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        bench.settle()
        first = spin.value()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        bench.settle()
        seen[name] = (first, spin.value(), neutral, value)
        spin.setValue(neutral)
    bench.settle()
    ok = all(abs(a - n) < 1e-9 and abs(b - v) < 1e-9 for a, b, n, v in seen.values())
    return ok, "; ".join(f"{k} {v[3]} -> {v[0]} -> {v[1]}" for k, v in seen.items())


@probe("viewer.refresh")
def _viewer_refresh(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    window = bench.window
    shown = _wait_for_picture(bench, None)
    count = window.viewer_refreshes
    _key(bench, viewer, Qt.Key.Key_U)
    again = _wait_for_picture(bench, shown)
    ok = window.viewer_refreshes == count + 1 and again is not None and again is not shown
    return ok, f"U refreshed {window.viewer_refreshes - count} time(s) and drew the frame again {again is not None}"


@probe("viewer.goto_frame")
def _viewer_goto_frame(bench):
    from PySide6.QtCore import Qt
    viewer, _ = _viewer(bench)
    window = bench.window
    timeline = window.dispatcher.document["time"]
    target = int(timeline["first"]) + 3 if int(timeline["current"]) != int(timeline["first"]) + 3 else int(timeline["first"])
    _key(bench, viewer, Qt.Key.Key_G, Qt.KeyboardModifier.AltModifier)
    dialog = window.goto_frame_dialog
    if dialog is None:
        return False, "Alt+G opened no dialog"
    title = dialog.windowTitle()
    dialog.setIntValue(target)
    dialog.accept()
    bench.settle()
    now = int(window.dispatcher.document["time"]["current"])
    return now == target and title == "Go to frame", f"Alt+G opened {title!r}; entering {target} went to frame {now}"


@probe("viewer.frame_field_math")
def _viewer_frame_field_math(bench):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    window = bench.window
    window.set_time(first=1, last=200)
    window.set_time(current=50)
    bench.settle()
    field = window.frame_current
    seen = []
    for text in ("+20", "-20", "-20"):
        edit = field.lineEdit()
        field.setFocus()
        edit.selectAll()
        QTest.keyClicks(edit, text)
        QTest.keyClick(edit, Qt.Key.Key_Return)
        bench.settle()
        seen.append(int(window.dispatcher.document["time"]["current"]))
    return seen == [70, 50, 30], f"typing +20, -20, -20 into the frame field from 50 went to {seen}"


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
