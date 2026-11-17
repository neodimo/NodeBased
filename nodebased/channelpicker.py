"""Nuke's two channel knob widgets for the Properties panel (lane 10, step T2).

Foundry's Nuke 17.0 User Guide, "Calling Channels"
(https://learn.foundry.com/nuke/17.0/content/comp_environment/channels/calling_channels.html):

* Channel set (a node's `channels` knob, Grade's or Invert's): "From the channels field, select the layer containing
  the channels you wish to process. The layer's channels appear with check boxes. Deselect those channels which you
  don't wish to process. The node processes all those you leave selected."  And for one channel: "From the channels
  field, select none. From the right most channel field - the one which typically calls the alpha channel - select the
  single channel you wish to process."
* Single channel (a mask channel, an unpremult-by channel): "select the channel you want to use from the mask dropdown
  menu", which Nuke draws as a layer part (`rgba`) and a channel part (`alpha`) side by side, stored as `rgba.alpha`.

What the document stores stays what it stored before (no schema change): a channel set is a word (`rgb`, `rgba`,
`alpha`, `all`, `none`), a layer name, or, once a channel has been left out, the ticked channels as a space-separated
list (`rgba.red rgba.blue`), which `ops2d_chrome.parse_channels` already reads on both render paths. A single channel
is `layer.channel` (or `none`). The value logic below is free of Qt so it is tested on its own.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from .knobfit import ElidingComboBox

RGBA = ("red", "green", "blue", "alpha")
LETTER = {"red": "r", "green": "g", "blue": "b", "alpha": "a"}
# The channels each layer word's check boxes show.
LAYER_CHANNELS = {"rgb": RGBA[:3], "rgba": RGBA, "all": RGBA, "alpha": ("alpha",)}
_WORDS = {"r": "red", "g": "green", "b": "blue", "a": "alpha", "red": "red", "green": "green", "blue": "blue", "alpha": "alpha"}

# Which open-choice knobs (ops2d_chrome.is_open_choice) are Nuke channel sets and which are single channels.
CHANNEL_SET_PARAMS = frozenset({"channels", "output", "Achannels", "Bchannels", "merge_output"})
SINGLE_CHANNEL_PARAMS = frozenset({"maskChannelInput", "unpremult", "premult", "premult_by", "width_channel",
                                   "blur_channel", "filter_channel", "ctrlMask"})


def _rgba_name(token):
    word = token.strip().lower()
    if word.startswith("rgba."):
        word = word[5:]
    return _WORDS.get(word)


def split_set(value):
    """A stored channel set -> (layer menu text, ticked channels, single channel or None).

    `rgb` -> ("rgb", (red, green, blue), None); `rgba.red rgba.blue` -> ("rgba", (red, blue), None); `none` ->
    ("none", (), None); one channel (`rgba.alpha`, `red`, `mattes.soft`) -> ("none", (), that channel), the guide's
    "select none, then the single channel in the right-most field"; a layer name -> (that name, (), None)."""
    text = str(value or "").strip()
    if text.lower() in LAYER_CHANNELS:
        return text.lower(), LAYER_CHANNELS[text.lower()], None
    if text.lower() in ("", "none"):
        return "none", (), None
    tokens = [token for token in text.replace(",", " ").replace("+", " ").split() if token]
    names = [_rgba_name(token) for token in tokens]
    if len(tokens) == 1:
        if names[0] is not None:
            return "none", (), f"rgba.{names[0]}"
        if "." in tokens[0]:
            return "none", (), tokens[0]
        return tokens[0], (), None
    if all(name is not None for name in names):
        return "rgba", tuple(name for name in RGBA if name in names), None
    return text, (), None


def join_set(layer, ticked):
    """The value to store for `layer` with `ticked` channels: the layer's own word while every box is ticked,
    `none` with none ticked, otherwise the ticked channels in rgba order."""
    shown = LAYER_CHANNELS.get(layer, ())
    kept = [name for name in RGBA if name in ticked and name in shown]
    if shown and len(kept) == len(shown):
        return layer
    if not kept:
        return "none"
    return " ".join(f"rgba.{name}" for name in kept)


def split_single(value):
    """`rgba.alpha` -> ("rgba", "alpha"); `none` -> ("none", ""); a bare word (`alpha`) reads as rgba's."""
    text = str(value or "").strip()
    if text.lower() in ("", "none"):
        return "none", ""
    if "." in text:
        layer, channel = text.split(".", 1)
        return layer, channel
    name = _rgba_name(text)
    return ("rgba", name) if name else (text, "")


def join_single(layer, channel):
    layer, channel = layer.strip(), channel.strip()
    if not layer or layer == "none":
        return "none"
    return f"{layer}.{channel}" if channel else layer


def single_choices(suggested):
    """A single-channel knob's suggestion list (`none`, `rgba.alpha`, ...) -> (layer words, {layer: channels})."""
    layers, channels = [], {}
    for word in suggested:
        layer, channel = split_single(word)
        if layer not in layers:
            layers.append(layer)
        if channel:
            channels.setdefault(layer, [])
            if channel not in channels[layer]:
                channels[layer].append(channel)
    return layers, channels


class ChannelSetPicker(QWidget):
    """Nuke's channels knob: a layer menu, one check box per channel of the chosen layer, and, with the menu at
    `none`, the right-most channel field that picks a single channel. `changed(value)` carries the value to store."""

    changed = Signal(str)

    def __init__(self, param, value, layers, parent=None):
        super().__init__(parent)
        self.param = param
        self.setObjectName(f"{param}-channel-set")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.layer = ElidingComboBox()
        self.layer.setObjectName(f"{param}-choice")
        self.layer.setEditable(True)
        words = [word for word in layers if word.lower() not in ("red", "green", "blue", "r", "g", "b", "a")]   # single channels: the right-most field
        self.layer.addItems(words)
        self.layer.setToolTip("Pick a layer, or type the name of a layer the image carries; untick a channel to leave it out")
        row.addWidget(self.layer, 1)
        self.boxes = {}
        for name in RGBA:
            box = QCheckBox(LETTER[name])
            box.setObjectName(f"{param}-{name}-box")
            box.setToolTip(f"Process {name}")
            box.toggled.connect(self._ticked)
            row.addWidget(box)
            self.boxes[name] = box
        self.single = ElidingComboBox()
        self.single.setObjectName(f"{param}-single")
        self.single.setEditable(True)
        self.single.addItems(["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"])
        self.single.setToolTip("The single channel to process (the channels menu is none)")
        row.addWidget(self.single, 1)
        self.value = None
        self.set_value(value)
        self.layer.activated.connect(lambda index: self._layer_chosen(self.layer.itemText(index)))
        self.layer.lineEdit().editingFinished.connect(lambda: self._layer_chosen(self.layer.currentText()))
        self.single.activated.connect(lambda index: self._emit(self.single.itemText(index).strip() or "none"))
        self.single.lineEdit().editingFinished.connect(lambda: self._emit(self.single.currentText().strip() or "none"))

    def set_value(self, value):
        self.value = str(value)
        layer, ticked, single = split_set(value)
        self._show(layer, ticked, single)

    def _show(self, layer, ticked, single):
        self._updating = True
        if self.layer.findText(layer) < 0:
            self.layer.addItem(layer)
        self.layer.setCurrentText(layer)
        shown = LAYER_CHANNELS.get(layer, ())
        for name, box in self.boxes.items():
            box.setVisible(name in shown)
            box.setChecked(name in ticked)
        self.single.setVisible(layer == "none")
        self.single.setCurrentText(single or "")
        self._updating = False

    def _layer_chosen(self, text):
        text = text.strip()
        if not text:
            return
        word = text.lower() if text.lower() in LAYER_CHANNELS or text.lower() == "none" else text
        self._show(word, LAYER_CHANNELS.get(word, ()), None)
        self._emit(word)

    def _ticked(self, *_):
        if self._updating:
            return
        layer = self.layer.currentText().strip().lower()
        ticked = [name for name, box in self.boxes.items() if box.isChecked()]
        self._emit(join_set(layer, ticked))

    def ticked(self):
        return tuple(name for name, box in self.boxes.items() if box.isVisible() and box.isChecked())

    def _emit(self, value):
        if value != self.value:
            self.value = value
            self.changed.emit(value)


class SingleChannelPicker(QWidget):
    """Nuke's single-channel knob: a layer menu (`rgba`) and a channel menu (`alpha`) side by side, stored as
    `rgba.alpha`; `none` greys the channel part out."""

    changed = Signal(str)

    def __init__(self, param, value, suggested, parent=None):
        super().__init__(parent)
        self.param = param
        self.setObjectName(f"{param}-single-channel")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.layers, self.channels = single_choices(suggested)
        self.enabled = None
        if param == "maskChannelInput":
            # Nuke's mask knob starts with a check box that turns the mask on; it ticks itself when a channel is picked.
            self.enabled = QCheckBox()
            self.enabled.setObjectName(f"{param}-enabled")
            self.enabled.setToolTip("Use a mask: off processes the whole image")
            row.addWidget(self.enabled)
            self.enabled.toggled.connect(self._toggled)
        self.layer = ElidingComboBox()
        self.layer.setObjectName(f"{param}-choice")
        self.layer.setEditable(True)
        self.layer.addItems(self.layers)
        self.layer.setToolTip("The layer the channel is taken from, or type a layer the image carries")
        self.channel = ElidingComboBox()
        self.channel.setObjectName(f"{param}-channel")
        self.channel.setEditable(True)
        self.channel.setToolTip("The channel of that layer")
        row.addWidget(self.layer, 1)
        row.addWidget(self.channel, 1)
        self.value = None
        self.set_value(value)
        self.layer.activated.connect(lambda index: self._layer_chosen(self.layer.itemText(index)))
        self.layer.lineEdit().editingFinished.connect(lambda: self._layer_chosen(self.layer.currentText()))
        self.channel.activated.connect(lambda index: self._emit(join_single(self.layer.currentText(), self.channel.itemText(index))))
        self.channel.lineEdit().editingFinished.connect(
            lambda: self._emit(join_single(self.layer.currentText(), self.channel.currentText())))

    def _fill_channels(self, layer, channel):
        self.channel.blockSignals(True)
        self.channel.clear()
        offered = self.channels.get(layer, list(RGBA) if layer != "none" else [])
        self.channel.addItems(offered)
        if channel and self.channel.findText(channel) < 0:
            self.channel.addItem(channel)
        self.channel.setCurrentText(channel)
        self.channel.setEnabled(layer != "none")
        self.channel.blockSignals(False)
        if self.enabled is not None:
            self.enabled.blockSignals(True)
            self.enabled.setChecked(layer != "none")
            self.enabled.blockSignals(False)

    def _toggled(self, on):
        if not on:
            self.layer.setCurrentText("none")
            self._fill_channels("none", "")
            self._emit("none")
        elif self.layer.currentText().strip() == "none":
            self.layer.setCurrentText("rgba")
            self._layer_chosen("rgba")

    def set_value(self, value):
        self.value = str(value)
        layer, channel = split_single(value)
        if self.layer.findText(layer) < 0:
            self.layer.addItem(layer)
        self.layer.setCurrentText(layer)
        self._fill_channels(layer, channel)

    def _layer_chosen(self, text):
        layer = text.strip()
        if not layer:
            return
        offered = self.channels.get(layer, list(RGBA))
        current = self.channel.currentText().strip()
        channel = "" if layer == "none" else (current if current in offered else ("alpha" if "alpha" in offered else offered[0]))
        self._fill_channels(layer, channel)
        self._emit(join_single(layer, channel))

    def _emit(self, value):
        if value != self.value:
            self.value = value
            self.changed.emit(value)
