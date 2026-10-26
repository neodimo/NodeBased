"""The Viewer settings window: Nuke's Viewer knobs (`viewerknobs.py`), kept in the document's `settings.viewer.knobs`.

Every control writes through `Window.set_viewer_knobs`, so an edit is one undo step, and `sync` brings the controls back to the
document after an undo, a load or an agent edit. The gain, gamma, zebra, view, proxy, mask and channel controls stay on the
viewer's own strip.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QSpinBox,
                               QVBoxLayout)

from . import viewerknobs
from .core import viewer_knobs


class ViewerSettings(QDialog):
    LABELS = {
        "frame_range": "Frame range", "frame_range_lock": "Lock frame range", "frame_increment": "Frame increment (<< >>)",
        "rgb_only": "Apply LUT to colour channels only", "ignore_pixel_aspect": "Ignore pixel aspect",
        "zoom_lock": "Prevent auto zoom", "show_overscan": "Show overscan", "overscan": "Overscan (pixels)",
        "full_frame_processing": "Enable full frame processing", "safe_zone": "Safe zone", "format_center": "Format center",
        "center_fstop": "Center f-stop", "shutter_angle": "Shutter angle", "use_gpu_for_viewer": "Use GPU for viewer when possible",
    }
    TIPS = {
        "frame_range": "The frames the viewer plays and steps through, such as 10-20. Empty follows the project's range.",
        "frame_range_lock": "On, the range above is in force. Off, the viewer follows the project's range.",
        "frame_increment": "The Time menu's << and >> steps jump this many frames.",
        "rgb_only": "On, the alpha channel is shown as it is. Off, gain, gamma and the view transform apply to it too.",
        "ignore_pixel_aspect": "On, pixels are drawn square. Off, a format with another pixel aspect is drawn that much wider.",
        "zoom_lock": "A picture of another size no longer refits the view.",
        "show_overscan": "Outline the format's edge and the overscan margin beyond it, and leave room for it when the view fits.",
        "overscan": "How far beyond the format the overscan margin reaches.",
        "full_frame_processing": "Evaluate the whole frame instead of only the part the zoomed view shows.",
        "safe_zone": "Title safe (80 %) and action safe (90 %) outlines over the picture.",
        "format_center": "A cross at the centre of the format.",
        "center_fstop": "The viewer gain read as an f-number about this f-stop; it labels the gain control.",
        "shutter_angle": "The shutter angle that labels the gain control with an exposure time.",
        "use_gpu_for_viewer": "Off, the picture is converted on the processor and never goes to the Direct picture as floats.",
    }

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setObjectName("viewer-settings")
        self.setWindowTitle("Viewer settings")
        self.setModal(False)
        self.controls = {}
        form = QFormLayout()
        for name in viewerknobs.DEFAULT:
            control = self._make(name)
            control.setObjectName(f"viewer-knob-{name}")
            control.setToolTip(self.TIPS[name])
            self.controls[name] = control
            if isinstance(control, QCheckBox):
                form.addRow(control)
            else:
                form.addRow(self.LABELS[name], control)
        self.note = QLabel("")
        self.note.setObjectName("viewer-knob-note")
        self.note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.note)
        self.sync()

    def _make(self, name):
        if name in viewerknobs.BOOLEANS:
            box = QCheckBox(self.LABELS[name])
            box.toggled.connect(lambda on, n=name: self.window.set_viewer_knobs(**{n: bool(on)}))
            return box
        if name == "frame_range":
            line = QLineEdit()
            line.setPlaceholderText("project range")
            line.editingFinished.connect(lambda: self._range_typed(line.text().strip()))
            return line
        if name == "safe_zone":
            combo = QComboBox()
            combo.addItems(viewerknobs.SAFE_ZONES)
            combo.activated.connect(lambda _i, c=combo: self.window.set_viewer_knobs(safe_zone=c.currentText()))
            return combo
        if name in viewerknobs.INTEGERS:
            low, high = viewerknobs.INTEGERS[name]
            spin = QSpinBox()
            spin.setRange(low, high)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda value, n=name: self.window.set_viewer_knobs(**{n: int(value)}))
            return spin
        low, high = viewerknobs.FLOATS[name]
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(1)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(lambda value, n=name: self.window.set_viewer_knobs(**{n: float(value)}))
        return spin

    def _range_typed(self, text):
        if text and viewerknobs.parse_range(text) is None:
            self.window.statusBar().showMessage(
                "The frame range is two frames such as 10-20, or empty for the project's range", 5000)
            self.sync()
            return
        self.window.set_viewer_knobs(frame_range=text)

    def sync(self):
        knobs = viewer_knobs(self.window.dispatcher.document)
        for name, control in self.controls.items():
            value = knobs[name]
            control.blockSignals(True)
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QLineEdit):
                control.setText(str(value))
            elif isinstance(control, QComboBox):
                control.setCurrentText(str(value))
            else:
                control.setValue(value)
            control.blockSignals(False)
        first, last = viewerknobs.playback_range(self.window.dispatcher.document)
        self.note.setText(f"The viewer plays frames {first} to {last}.")
