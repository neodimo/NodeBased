"""Nuke's channel knob widgets (lane 10, step T2): the channel set (layer menu + one check box per channel, and the
right-most single-channel field at none) and the single channel (layer menu + channel menu), per the Nuke 17.0 User
Guide's "Calling Channels" page. The stored values are the ones the document already held, so old documents load
unchanged and the render paths read the new lists with their existing parser."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import unittest

import numpy as np

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)
from nodebased import channelpicker as cp, ops2d_chrome as chrome
from nodebased.core import SPECS


class ValueTests(unittest.TestCase):
    def test_set_words_layers_and_lists_round_trip(self):
        self.assertEqual(cp.split_set("rgb"), ("rgb", ("red", "green", "blue"), None))
        self.assertEqual(cp.split_set("all"), ("all", cp.RGBA, None))
        self.assertEqual(cp.split_set("none"), ("none", (), None))
        self.assertEqual(cp.split_set("rgba.red rgba.blue"), ("rgba", ("red", "blue"), None))
        self.assertEqual(cp.split_set("rgba.alpha"), ("none", (), "rgba.alpha"))
        self.assertEqual(cp.split_set("red"), ("none", (), "rgba.red"))
        self.assertEqual(cp.split_set("mattes"), ("mattes", (), None))
        self.assertEqual(cp.split_set("mattes.soft"), ("none", (), "mattes.soft"))
        self.assertEqual(cp.join_set("rgba", cp.RGBA), "rgba")
        self.assertEqual(cp.join_set("rgb", ("red", "blue")), "rgba.red rgba.blue")
        self.assertEqual(cp.join_set("rgba", ()), "none")
        self.assertEqual(cp.join_set("alpha", ("alpha",)), "alpha")

    def test_single_channel_values(self):
        self.assertEqual(cp.split_single("rgba.alpha"), ("rgba", "alpha"))
        self.assertEqual(cp.split_single("none"), ("none", ""))
        self.assertEqual(cp.split_single("mattes.soft"), ("mattes", "soft"))
        self.assertEqual(cp.join_single("rgba", "luminance"), "rgba.luminance")
        self.assertEqual(cp.join_single("none", "alpha"), "none")
        layers, channels = cp.single_choices(chrome.MASK_CHANNEL_CHOICES)
        self.assertEqual(layers, ["none", "rgba"])
        self.assertEqual(channels["rgba"], ["alpha", "red", "green", "blue", "luminance"])

    def test_a_stored_list_is_read_by_the_render_path(self):
        """The list the check boxes store limits the node to those channels (the chrome's own parser)."""
        parsed = chrome.parse_channels(cp.join_set("rgba", ("red", "blue", "alpha")))
        self.assertEqual(parsed.rgba, (True, False, True, True))
        from tests.test_2d_fidelity_b_chrome import image, render
        src = image()
        out = render("Grade", {"multiply": 2.0, "channels": cp.join_set("rgb", ("red", "blue"))})
        out = out.pixels if hasattr(out, "pixels") else out
        np.testing.assert_array_equal(out[..., 1], src[..., 1])
        self.assertFalse(np.allclose(out[..., 0], src[..., 0]))


class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nodebased.app import Window
        cls.window = Window()
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.window.saved_document = cls.window.dispatcher.document     # no "discard unsaved edits?" prompt on close
        cls.window.close()
        cls.window.deleteLater()

    def node(self, kind, params=None):
        type(self).count += 1
        key = f"picker_{self.count}"
        command = {"op": "create", "id": key, "type": kind, "pos": [self.count * 40, 0]}
        if params:
            command["params"] = params
        self.window.dispatcher.execute(command)
        return key, self.window.build_node_panel(key)

    def test_unticking_a_channel_stores_the_rest_in_one_step(self):
        from PySide6.QtWidgets import QCheckBox
        key, panel = self.node("Invert", {"channels": "rgb"})
        boxes = {n: panel.findChild(QCheckBox, f"channels-{n}-box") for n in cp.RGBA}
        self.assertEqual([n for n, b in boxes.items() if not b.isHidden()], ["red", "green", "blue"])
        boxes["green"].click()
        APP.processEvents()
        self.assertEqual(self.window.dispatcher.document["nodes"][key]["params"]["channels"], "rgba.red rgba.blue")

    def test_choosing_a_layer_ticks_its_channels(self):
        from PySide6.QtWidgets import QCheckBox, QComboBox
        key, panel = self.node("Blur")
        menu = panel.findChild(QComboBox, "channels-choice")
        menu.setCurrentIndex(menu.findText("alpha"))
        menu.activated.emit(menu.currentIndex())
        APP.processEvents()
        self.assertEqual(self.window.dispatcher.document["nodes"][key]["params"]["channels"], "alpha")
        visible = [n for n in cp.RGBA if not panel.findChild(QCheckBox, f"channels-{n}-box").isHidden()]
        self.assertEqual(visible, ["alpha"])

    def test_mask_channel_is_a_layer_and_a_channel_side_by_side(self):
        from PySide6.QtWidgets import QComboBox
        key, panel = self.node("Grade")
        layer = panel.findChild(QComboBox, "maskChannelInput-choice")
        channel = panel.findChild(QComboBox, "maskChannelInput-channel")
        self.assertEqual((layer.currentText(), channel.isEnabled()), ("none", False))
        layer.setCurrentIndex(layer.findText("rgba"))
        layer.activated.emit(layer.currentIndex())
        APP.processEvents()
        self.assertEqual(self.window.dispatcher.document["nodes"][key]["params"]["maskChannelInput"], "rgba.alpha")
        channel.setCurrentIndex(channel.findText("luminance"))
        channel.activated.emit(channel.currentIndex())
        APP.processEvents()
        self.assertEqual(self.window.dispatcher.document["nodes"][key]["params"]["maskChannelInput"], "rgba.luminance")

    def test_every_channel_knob_of_these_nodes_uses_the_shared_widgets(self):
        """Every open-choice channel knob is a channel set or a single channel and gets its widget; an old value
        shows unchanged and the panel stores nothing on build."""
        from PySide6.QtWidgets import QWidget
        for kind in ("Grade", "Blur", "Merge", "Rectangle", "Premult", "Glow", "Copy", "Keymix", "Write"):
            params = SPECS[kind]["params"]
            wanted = [p for p in params if chrome.is_open_choice(kind, p)
                      and (p in cp.CHANNEL_SET_PARAMS or p in cp.SINGLE_CHANNEL_PARAMS)]
            key, panel = self.node(kind)
            before = dict(self.window.dispatcher.document["nodes"][key]["params"])
            for param in wanted:
                suffix = "channel-set" if param in cp.CHANNEL_SET_PARAMS else "single-channel"
                with self.subTest(kind=kind, param=param):
                    self.assertIsNotNone(panel.findChild(QWidget, f"{param}-{suffix}"))
            APP.processEvents()
            self.assertEqual(self.window.dispatcher.document["nodes"][key]["params"], before)

    def test_every_open_choice_channel_param_is_classified(self):
        """No channel knob falls back to the old one-menu widget by accident: each open-choice name is a set,
        a single channel, or one of the named exceptions."""
        exceptions = {"uv_layer", "also_merge"}
        for kind, spec in SPECS.items():
            for param in spec["params"]:
                if chrome.is_open_choice(kind, param):
                    with self.subTest(kind=kind, param=param):
                        self.assertIn(param, cp.CHANNEL_SET_PARAMS | cp.SINGLE_CHANNEL_PARAMS | exceptions)


if __name__ == "__main__":
    unittest.main()
