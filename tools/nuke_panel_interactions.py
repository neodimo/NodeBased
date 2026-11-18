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
    return False, "channel, layer and format knobs have no Link menu: expressions drive number knobs only"


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
    return False, "there is no Clone: Alt+C makes an independent copy, with no active link to its parent"


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
