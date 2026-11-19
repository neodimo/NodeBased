"""2D behaviour, lane 9 step X3: GridWarp's grid knobs (Source Grid, Destination Grid and Colors tabs of the Nuke 17.0 Reference Guide page).

The page lists, per grid, `keys`, `visible`, `locked` and Resize to Image, and a `source color` and `destination color` on the Colors tab. The toolbar of
the video the page carries: "the first three boxes ... are toggles for whether we are looking at our source grid or destination grid".

* visible and locked change what the Viewer shows and lets you pick (nodebased/app.py `_draw_warp_overlay`, `_warp_hit`); they never change the
  warped picture, and these tests say so with pixels.
* source color and destination color are the colours the two grids are drawn in.
* keys is the list of frames at which a grid has a key; Resize to Image lays a grid out again across the image.
"""
from __future__ import annotations

import copy
import os
import sys
import types
import unittest

import numpy as np
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from nodebased import core, knobs, warpknobs
from nodebased.core import SPECS, Dispatcher
from nodebased.imaging import Evaluator

APP = QApplication.instance() or QApplication(sys.argv[:1])
from nodebased.app import Viewer, Window  # noqa: E402

W, H = 64, 48
CYAN, YELLOW = QColor("#58d7ff"), QColor("#f4ce63")


def point(x, y):
    return {"x": float(x), "y": float(y), "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0}


def grids(shift=20.0):
    """A 3 x 3 source grid across the picture and a destination grid shifted `shift` pixels right (more than a pick's reach, 12), so the two never
    sit on each other."""
    source = [[point(x, y) for x in (8, 32, 56)] for y in (8, 24, 40)]
    destination = [[point(x + shift, y) for x in (8, 32, 56)] for y in (8, 24, 40)]
    return {"source": source, "destination": destination}


def document(**params):
    d = Dispatcher()
    d.execute(dict(op="create", id="plate", type="Constant", params=dict(width=W, height=H, red=0.5, green=0.5, blue=0.5, alpha=1.0)))
    d.execute(dict(op="create", id="grid", type="GridWarp", params=dict(rows=3, columns=3, **params)))
    d.execute(dict(op="connect", id="grid", input="image", source="plate"))
    d.execute(dict(op="set_warp_data", id="grid", data=grids()))
    d.document["view"] = "grid"
    return d


class StubViewer:
    """Just enough of the Viewer for its warp overlay: the real methods run on it."""
    _roto_scene_point = Viewer._roto_scene_point
    _roto_data_point = Viewer._roto_data_point
    _warp_context = Viewer._warp_context
    _warp_hit = Viewer._warp_hit
    _draw_warp_overlay = Viewer._draw_warp_overlay

    def __init__(self, dispatcher):
        self.format_rect = QRectF(0, 0, W, H)
        self.warp_drag, self.warp_drawing, self.warp_draw_points, self.warp_draw_cursor = None, False, [], None
        self.window = types.SimpleNamespace(dispatcher=dispatcher, graph=types.SimpleNamespace(selected_id=lambda: "grid"),
                                            proxy=None)

    def view_zoom(self):
        return 1.0

    def hit(self, x, y):
        return self._warp_hit(self._warp_context(), QPointF(x, y))

    def painted(self):
        """The overlay painted on a transparent picture: how many pixels are nearer the source colour and how many nearer the destination colour."""
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        self._draw_warp_overlay(painter)
        painter.end()
        counts = {"source": 0, "destination": 0, "other": 0}
        for y in range(H):
            for x in range(W):
                pixel = QColor.fromRgba(image.pixel(x, y))
                if QColor.fromRgba(image.pixel(x, y)).alpha() < 160:
                    continue
                to_source = abs(pixel.red() - CYAN.red()) + abs(pixel.green() - CYAN.green()) + abs(pixel.blue() - CYAN.blue())
                to_destination = abs(pixel.red() - YELLOW.red()) + abs(pixel.green() - YELLOW.green()) + abs(pixel.blue() - YELLOW.blue())
                if min(to_source, to_destination) > 90:
                    counts["other"] += 1
                else:
                    counts["source" if to_source < to_destination else "destination"] += 1
        return counts


def picked(hit):
    return (hit["side"], hit["row"], hit["column"])


def set_param(dispatcher, name, value):
    dispatcher.execute(dict(op="set", id="grid", param=name, value=value))


class VisibleAndLockedTests(unittest.TestCase):
    def test_both_grids_are_drawn_and_can_be_picked_by_default(self):
        viewer = StubViewer(document())
        counts = viewer.painted()
        self.assertGreater(counts["source"], 40)
        self.assertGreater(counts["destination"], 40)
        self.assertEqual(picked(viewer.hit(8, 8)), ("source", 0, 0))
        self.assertEqual(picked(viewer.hit(28, 8)), ("destination", 0, 0))

    def test_a_hidden_grid_is_not_drawn_and_cannot_be_picked(self):
        for side, other in (("source", "destination"), ("destination", "source")):
            with self.subTest(side=side):
                d = document()
                set_param(d, f"{side}_grid_visible", 0)
                viewer = StubViewer(d)
                counts = viewer.painted()
                self.assertEqual(counts[side], 0)
                self.assertGreater(counts[other], 40)
                x = 8 if side == "source" else 28
                found = viewer.hit(x, 8)
                self.assertTrue(found is None or found["side"] == other)                  # the hidden grid is never the one picked
                self.assertEqual(picked(viewer.hit(28 if other == "destination" else 8, 8))[0], other)

    def test_both_hidden_draws_nothing(self):
        d = document()
        set_param(d, "source_grid_visible", 0)
        set_param(d, "destination_grid_visible", 0)
        self.assertEqual(StubViewer(d).painted(), {"source": 0, "destination": 0, "other": 0})

    def test_a_locked_grid_is_drawn_but_cannot_be_dragged(self):
        d = document()
        set_param(d, "source_grid_locked", 1)
        viewer = StubViewer(d)
        self.assertGreater(viewer.painted()["source"], 40)                   # still shown
        self.assertIsNone(viewer.hit(8, 8))                                    # not picked: no drag starts
        self.assertEqual(picked(viewer.hit(28, 8))[0], "destination")          # the other grid still is
        set_param(d, "destination_grid_locked", 1)
        self.assertIsNone(StubViewer(d).hit(28, 8))
        set_param(d, "source_grid_locked", 0)
        self.assertEqual(picked(StubViewer(d).hit(8, 8))[0], "source")

    def test_visible_and_locked_never_change_the_warped_picture(self):
        plain = Evaluator().evaluate(dict(document().document, view="grid"))
        d = document()
        for name in ("source_grid_visible", "destination_grid_visible"):
            set_param(d, name, 0)
        for name in ("source_grid_locked", "destination_grid_locked"):
            set_param(d, name, 1)
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="grid")), plain)


class ColorTests(unittest.TestCase):
    def test_the_grids_start_in_the_colours_they_were_always_drawn_in(self):
        params = SPECS["GridWarp"]["params"]
        self.assertEqual(warpknobs.grid_color(params, "source"), "#58d7ff")
        self.assertEqual(warpknobs.grid_color(params, "destination"), "#f4ce63")

    def test_a_chosen_colour_is_the_one_the_viewer_draws(self):
        d = document()
        for channel, value in zip("rgb", (1.0, 0.0, 0.0)):
            set_param(d, f"source_color_{channel}", value)
        for channel, value in zip("rgb", (0.0, 1.0, 0.0)):
            set_param(d, f"destination_color_{channel}", value)
        self.assertEqual((warpknobs.grid_color(d.document["nodes"]["grid"]["params"], "source"),
                          warpknobs.grid_color(d.document["nodes"]["grid"]["params"], "destination")), ("#ff0000", "#00ff00"))
        viewer = StubViewer(d)
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        viewer._draw_warp_overlay(painter)
        painter.end()
        seen = {(QColor.fromRgba(image.pixel(x, y)).red() > 200 and QColor.fromRgba(image.pixel(x, y)).green() < 60,
                 QColor.fromRgba(image.pixel(x, y)).green() > 200 and QColor.fromRgba(image.pixel(x, y)).red() < 60)
                for y in range(H) for x in range(W) if QColor.fromRgba(image.pixel(x, y)).alpha() > 200}
        self.assertIn((True, False), seen)                                        # a red source line
        self.assertIn((False, True), seen)                                        # a green destination line
        self.assertEqual(viewer.painted()["source"], 0)                            # none of the old cyan is left


class KeysAndResizeTests(unittest.TestCase):
    def test_keys_are_the_frames_where_a_grid_has_a_key(self):
        data = grids()
        self.assertEqual(warpknobs.grid_keys(data, "source"), [])
        data["source"][1][1]["x"] = {"value": 32.0, "curve": {"keys": [{"frame": 12, "value": 30.0}, {"frame": 1, "value": 32.0}]}}
        data["source"][0][0]["y"] = {"value": 8.0, "curve": {"keys": [{"frame": 12, "value": 9.0}, {"frame": 24, "value": 7.0}]}}
        self.assertEqual(warpknobs.grid_keys(data, "source"), [1, 12, 24])
        self.assertEqual(warpknobs.grid_keys(data, "destination"), [])
        self.assertEqual(warpknobs.grid_keys(None, "source"), [])

    def test_resize_to_image_lays_the_grid_across_the_image_and_keeps_the_other_grid(self):
        d = document()
        window = types.SimpleNamespace(dispatcher=d, viewer=types.SimpleNamespace(format_rect=QRectF(0, 0, 100, 60)), commands=[])
        window.command = window.commands.append
        self.assertTrue(Window._gridwarp_resize(window, "grid", "destination"))
        command = window.commands[0]
        self.assertEqual((command["op"], command["id"]), ("set_warp_data", "grid"))
        new = command["data"]["destination"]
        self.assertEqual((len(new), len(new[0])), (3, 3))
        self.assertEqual((new[0][0]["x"], new[0][0]["y"], new[2][2]["x"], new[2][2]["y"]), (0.5, 0.5, 99.5, 59.5))
        self.assertEqual(command["data"]["source"], grids()["source"])           # untouched
        self.assertFalse(Window._gridwarp_resize(window, "grid", "middle"))

    def test_resize_to_image_makes_a_valid_document_edit(self):
        d = document()
        data = {"source": warpknobs.resized_grid(3, 3, W, H), "destination": warpknobs.resized_grid(3, 3, W, H)}
        d.execute(dict(op="set_warp_data", id="grid", data=data))
        self.assertEqual(warpknobs.grid_keys(d.document["node_data"]["grid"], "source"), [])


class OldDocumentsAndLayoutTests(unittest.TestCase):
    def test_a_node_saved_without_the_new_params_gets_them_at_the_old_look(self):
        d = document()
        node = copy.deepcopy(d.document["nodes"]["grid"])
        for name in list(node["params"]):
            if "grid_visible" in name or "grid_locked" in name or "_color_" in name:
                del node["params"][name]
        d.document["nodes"]["grid"] = node
        upgraded = core.upgrade_document(d.document)["nodes"]["grid"]["params"]
        self.assertEqual((upgraded["source_grid_visible"], upgraded["destination_grid_visible"],
                          upgraded["source_grid_locked"], upgraded["destination_grid_locked"]), (1, 1, 0, 0))
        self.assertEqual(warpknobs.grid_color(upgraded, "source"), "#58d7ff")
        core.validate(core.upgrade_document(d.document))

    def test_the_toolbar_rows_lead_the_settings_as_the_guide_orders_them(self):
        shown = [g.label for g in knobs.KNOB_LAYOUT["GridWarp"] if g.kind != "legacy" and not g.params[0].startswith("legacy_")]
        self.assertEqual(shown[:5], ["Channels", "Mask", "Inject", "Invert", "Fringe"])
        self.assertEqual(shown[5:9], ["Visible", "Locked", "Visible", "Locked"])        # the Source Grid then the Destination Grid tab
        self.assertEqual(shown[9:15], ["Output", "Warp", "Mix", "Background", "Background mix", "Set bbox to"])    # the Settings tab
        self.assertEqual(shown[shown.index("Filter") + 1:shown.index("Filter") + 3], ["Source color", "Destination color"])   # the Colors tab, after Render
        grouped = {g.params[0]: g for g in knobs.KNOB_LAYOUT["GridWarp"]}
        self.assertEqual(grouped["source_grid_visible"].kind, "bool")
        self.assertEqual(grouped["source_color_r"].kind, "color")
        for name in ("source_grid_visible", "source_grid_locked", "destination_grid_visible", "destination_grid_locked"):
            self.assertEqual(core.LIMITS[name], (0, 1))


class PanelTests(unittest.TestCase):
    """The real Properties panel: the keys readouts and the Resize to Image buttons of both grids, and the Reset button of a lens node."""

    def setUp(self):
        from tests.test_desktop import release_window
        self.window = Window()
        self.window.show()
        APP.processEvents()
        self.addCleanup(release_window, self)

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def test_the_grid_panel_has_a_keys_readout_and_a_resize_button_per_grid(self):
        from PySide6.QtWidgets import QLabel, QPushButton
        self.window.dispatcher.execute(dict(op="create", id="gw", type="GridWarp"))
        panel = self.window.build_node_panel("gw")
        for side in ("source", "destination"):
            self.assertEqual(panel.findChild(QLabel, f"gridwarp-{side}-keys").text(), "none")
            self.assertIsNotNone(panel.findChild(QPushButton, f"gridwarp-{side}-resize"))

    def test_a_lens_panel_has_a_reset_button(self):
        from PySide6.QtWidgets import QPushButton
        for kind in ("Undistort", "Redistort"):
            self.window.dispatcher.execute(dict(op="create", id=f"lens-{kind}", type=kind))
            self.assertIsNotNone(self.window.build_node_panel(f"lens-{kind}").findChild(QPushButton, "lens-reset-distortion"), kind)


if __name__ == "__main__":
    unittest.main()
