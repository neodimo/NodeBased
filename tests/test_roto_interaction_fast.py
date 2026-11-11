"""Roto interaction stays quick: a drag ends in one undo step and the document the old route produced, its picture requests
carry the document the release will commit, handles repaint without waiting for the picture, and the heavy panel rebuilds
once after a burst instead of inside every click.

No test asserts elapsed time (the suite runs many interpreters at once); they assert the work that was slow: copies, undo
entries, picture requests and panel rebuilds.
"""
import copy
import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QWidget

from nodebased.app import STYLE, Window
from tests.test_roto_ui import roto_document, triangle, point_value
from tests.waiting import wait_until

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)


def blob(index, count=12):
    cx, cy = 20 + (index % 5) * 15, 20 + (index // 5) * 20
    points = []
    for i in range(count):
        a = 2 * math.pi * i / count
        points.append({"x": cx + 5 * math.cos(a), "y": cy + 5 * math.sin(a), "in_x": 0.0, "in_y": 0.0,
                       "out_x": 0.0, "out_y": 0.0})
    return {"name": f"blob{index}", "mode": "union", "opacity": 1.0, "feather": 0.0, "points": points}


class FastRotoTests(unittest.TestCase):
    def open(self, shapes):
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(roto_document(shapes))
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["r"].setSelected(True)
        APP.processEvents()
        self.window.timer.stop()
        return self.window.viewer

    def close(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def tearDown(self):
        if getattr(self, "window", None) is not None:
            self.close()
            self.window = None

    def at(self, x, y):
        return self.window.viewer.mapFromScene(QPointF(x, y))

    def drag_point(self, viewer):
        start, end = self.at(10, 10), self.at(40, 35)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=start)
        for step in range(1, 7):
            QTest.mouseMove(viewer.viewport(), self.at(10 + 5 * step, 10 + 25 * step / 6))
        QTest.mouseMove(viewer.viewport(), end)
        mid = (len(self.window.dispatcher.undo_stack), copy.deepcopy(self.window.dispatcher.document["node_data"]))
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)
        return mid, copy.deepcopy(self.window.dispatcher.document["node_data"]), len(self.window.dispatcher.undo_stack)

    def test_drag_ends_in_one_undo_step_and_the_document_a_full_copy_edit_produces(self):
        viewer = self.open([triangle()])
        document_before = copy.deepcopy(self.window.dispatcher.document)
        (mid_undo, mid_data), committed, undo_steps = self.drag_point(viewer)
        self.assertEqual(mid_undo, 0)                                   # nothing is committed mid-drag
        self.assertEqual(mid_data, document_before["node_data"])
        self.assertEqual(undo_steps, 1)
        self.window.command({"op": "undo"})
        self.assertEqual(self.window.dispatcher.document["node_data"], document_before["node_data"])
        self.close()
        # the same drag with every shape copied before the edit (how the commit worked before it copied only what moves)
        viewer = self.open([triangle()])
        viewer._shapes_for_edit = lambda payload, indices: copy.deepcopy(payload.get("shapes", []))
        _, reference, _ = self.drag_point(viewer)
        self.assertEqual(committed, reference)
        moved = committed["r"]["shapes"][0]["points"]
        self.assertNotEqual(moved[0], document_before["node_data"]["r"]["shapes"][0]["points"][0])
        self.assertEqual(moved[1], document_before["node_data"]["r"]["shapes"][0]["points"][1])

    def test_the_picture_requested_during_a_drag_carries_the_document_the_release_commits(self):
        shapes = [blob(i) for i in range(10)]
        viewer = self.open(shapes)
        target = (shapes[3]["points"][2]["x"], shapes[3]["points"][2]["y"])
        end = self.at(target[0] + 6, target[1] + 4)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(*target))
        QTest.mouseMove(viewer.viewport(), end)
        self.window.timer.stop()
        request, _cancel = self.window.preview_queue.take()
        self.assertEqual(request.tier, viewer.DRAG_PREVIEW_TIER)      # one proxy tier for the whole drag, never refined while held
        self.assertTrue(request.uncached, "a drag's picture skips the display cache")
        self.assertIsNone(self.window._progressive_document)         # nothing is queued to refine the picture while the hand is down
        live = self.window.dispatcher.document["node_data"]["r"]["shapes"]
        previewed = request.document["node_data"]["r"]["shapes"]
        for index, shape in enumerate(previewed):
            if index != 3:
                self.assertIs(shape, live[index], "an untouched shape is shared, not copied")
        self.assertNotEqual(previewed[3], live[3])
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)
        committed = self.window.dispatcher.document["node_data"]["r"]["shapes"]
        self.assertEqual(previewed, committed)
        self.assertEqual(len(self.window.dispatcher.undo_stack), 1)

    def test_the_first_drag_move_requests_a_picture_on_a_machine_just_booted(self):
        # time.monotonic() counts from boot: on a CI runner up for a few minutes it is below the test's 3600 s interval,
        # and a "last request" of 0.0 held the first picture (Linux conformance, 11/11).
        from unittest import mock
        from nodebased import app as app_module
        viewer = self.open([triangle()])
        requests = []
        original = self.window.request_preview
        self.window.request_preview = lambda *a, **k: requests.append(k.get("document_override")) or original(*a, **k)
        viewer.DRAG_PREVIEW_INTERVAL = 3600.0
        with mock.patch.object(app_module.time, "monotonic", return_value=120.0):
            QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(10, 10))
            QTest.mouseMove(viewer.viewport(), self.at(16, 14))
        self.assertEqual(len(requests), 1)
        viewer._drag_preview_timer.stop()

    def test_moves_inside_the_preview_interval_make_one_request(self):
        viewer = self.open([triangle()])
        requests = []
        original = self.window.request_preview
        self.window.request_preview = lambda *a, **k: requests.append(k.get("document_override")) or original(*a, **k)
        viewer.DRAG_PREVIEW_INTERVAL = 3600.0           # no second request can be due while the test runs
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(10, 10))
        for step in range(1, 12):
            QTest.mouseMove(viewer.viewport(), self.at(10 + 3 * step, 10 + 2 * step))
        self.assertEqual(len(requests), 1)
        viewer._drag_preview_timer.stop()
        viewer._flush_drag_preview()                    # the interval is up: the newest position is requested once
        self.assertEqual(len(requests), 2)
        position = requests[1]["node_data"]["r"]["shapes"][0]["points"][0]
        self.assertAlmostEqual(point_value(position["x"]), 10 + 33, delta=2.0)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(43, 32))
        self.assertFalse(viewer._drag_preview_timer.isActive())

    def test_resolved_shapes_are_kept_until_the_document_frame_or_node_changes(self):
        viewer = self.open([triangle()])
        context = viewer._roto_context()
        first = viewer._roto_resolved(context)
        self.assertIs(viewer._roto_resolved(context), first)
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [triangle(), blob(1)]})
        second = viewer._roto_resolved(viewer._roto_context())
        self.assertIsNot(second, first)
        self.assertEqual(len(second), 2)
        self.window.command({"op": "time", "current": 2})
        self.assertIsNot(viewer._roto_resolved(viewer._roto_context()), second)

    def test_painting_a_drag_never_changes_the_cached_resolved_points(self):
        viewer = self.open([triangle()])
        context = viewer._roto_context()
        resolved = copy.deepcopy(viewer._roto_resolved(context))
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(10, 10))
        QTest.mouseMove(viewer.viewport(), self.at(40, 40))
        image = QImage(200, 200, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        viewer.paint_overlays(painter, QRectF(0, 0, 100, 100))      # draws the point at the pointer
        painter.end()
        self.assertEqual(viewer._roto_resolved(context), resolved)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(40, 40))

    def test_a_marquee_draws_its_rectangle_without_error(self):
        viewer = self.open([triangle()])
        viewer.select_roto_tool("Select", "Select Points")
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(2, 2))
        QTest.mouseMove(viewer.viewport(), self.at(60, 60))
        self.assertEqual(viewer.roto_drag["kind"], "marquee")
        image = QImage(300, 300, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        viewer.paint_overlays(painter, QRectF(0, 0, 100, 100))      # a marquee has no point: this raised KeyError
        painter.end()
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(60, 60))

    def test_a_roto_panel_of_one_shape_or_twenty_rebuilds_once_after_a_burst(self):
        self.open([triangle()])
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        self.window.command({"op": "set_shapes", "id": "r", "shapes": [triangle(), blob(1)]})
        self.assertEqual(builds, [])                                # a one-shape panel is 280 widgets: never built inside the command
        self.assertTrue(self.window._panel_rebuild_timer.isActive())
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()
        self.assertEqual(builds, [], "a shape added to the panel on screen is put in its list, the panel is not built again")
        APP.processEvents()
        self.assertEqual(self.window.properties.widget().findChild(QWidget, "roto-shape-list").count(), 2)
        self.close()
        self.open([blob(i) for i in range(self.window_items())])
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        shapes = [blob(i) for i in range(self.window_items())]
        for count in (1, 2, 3, 4):
            self.window.command({"op": "set_shapes", "id": "r", "shapes": shapes[:-1] + [blob(count)]})
        self.assertEqual(builds, [])                                # nothing rebuilt inside the burst
        self.assertTrue(self.window._panel_rebuild_timer.isActive())
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()
        self.assertEqual(builds, ["r"])                             # one rebuild for four edits
        self.window._run_panel_rebuild()
        self.assertEqual(builds, ["r"])

    def test_the_panel_signature_ignores_point_coordinates_and_follows_what_the_panel_shows(self):
        from nodebased import rotopanel
        shapes = [blob(0), blob(1)]
        base = rotopanel.panel_signature({"shapes": shapes}, 1)
        moved = copy.deepcopy(shapes)
        moved[0]["points"][3]["x"] += 7.5
        moved[0]["points"][3]["out_y"] = 2.0
        self.assertEqual(rotopanel.panel_signature({"shapes": moved}, 1), base)         # coordinates are not shown
        renamed = copy.deepcopy(shapes)
        renamed[1]["name"] = "other"
        self.assertNotEqual(rotopanel.panel_signature({"shapes": renamed}, 1), base)
        faded = copy.deepcopy(shapes)
        faded[0]["opacity"] = 0.5
        self.assertNotEqual(rotopanel.panel_signature({"shapes": faded}, 1), base)
        keyed = copy.deepcopy(shapes)
        keyed[0]["points"][0]["x"] = {"value": 20.0, "curve": {"interpolation": "linear", "keys": [{"frame": 1, "value": 20.0}, {"frame": 5, "value": 30.0}]}}
        with_key = rotopanel.panel_signature({"shapes": keyed}, 1)
        self.assertNotEqual(with_key, base)                                              # a point became animated: the key buttons show it
        self.assertEqual(rotopanel.panel_signature({"shapes": keyed}, 2), rotopanel.panel_signature({"shapes": keyed}, 3))   # no key on 2 or 3
        self.assertNotEqual(rotopanel.panel_signature({"shapes": keyed}, 5), rotopanel.panel_signature({"shapes": keyed}, 3))   # a key on 5
        moved_key = copy.deepcopy(keyed)
        moved_key[0]["points"][0]["x"]["curve"]["keys"][1]["value"] = 44.0
        self.assertEqual(rotopanel.panel_signature({"shapes": moved_key}, 2), rotopanel.panel_signature({"shapes": keyed}, 2))
        faded_curve = copy.deepcopy(shapes)
        faded_curve[0]["opacity"] = {"value": 1.0, "curve": {"interpolation": "linear", "keys": [{"frame": 1, "value": 1.0}, {"frame": 9, "value": 0.0}]}}
        self.assertNotEqual(rotopanel.panel_signature({"shapes": faded_curve}, 2), rotopanel.panel_signature({"shapes": faded_curve}, 3))

    def test_timeline_marks_are_computed_once_per_document_revision(self):
        from nodebased import app as app_module
        self.open([blob(i) for i in range(4)])
        window = self.window
        calls = []
        original = app_module.DisplayCache.identity
        app_module.DisplayCache.identity = staticmethod(lambda *args: calls.append(args) or original(*args))
        try:
            window.refresh_timeline_marks()                          # the first look at this revision may or may not hash
            first = len(calls)
            window.refresh_timeline_marks()
            window.refresh_timeline_marks()
            self.assertEqual(len(calls), first)                      # a drag's pictures do not hash the document again
            window.command({"op": "set_shapes", "id": "r", "shapes": [blob(i) for i in range(5)]}, render=False)
            window.refresh_timeline_marks()
            self.assertGreater(len(calls), first)                    # an edit does
        finally:
            app_module.DisplayCache.identity = original

    def test_a_drag_snapshot_is_shared_by_its_refinement_not_copied(self):
        self.open([blob(i) for i in range(3)])
        window = self.window
        snapshot = copy.deepcopy(window.dispatcher.document)
        window.request_preview(progressive=True, document_override=snapshot, document_owned=True)
        self.assertIs(window._progressive_document, snapshot)
        self.assertTrue(window._progressive_document_owned)
        window.request_preview(progressive=True, document_override=copy.deepcopy(snapshot))
        self.assertFalse(window._progressive_document_owned)         # a caller that did not hand the snapshot over still gets a copy
        window.request_preview()
        self.assertIsNone(window._progressive_document)

    def test_the_panel_signature_accepts_an_animated_colour_channel(self):
        from nodebased import rotopanel
        shape = blob(0)
        shape["color"] = [{"value": 1.0, "curve": {"interpolation": "linear", "keys": [{"frame": 1, "value": 1.0}, {"frame": 5, "value": 0.0}]}}, 0.0, 0.0, 1.0]
        self.assertNotEqual(rotopanel.panel_signature({"shapes": [shape]}, 2), rotopanel.panel_signature({"shapes": [shape]}, 3))

    def test_a_heavy_panel_is_not_built_again_when_an_edit_changes_nothing_it_shows(self):
        shapes = [blob(i) for i in range(self.window_items())]
        self.open(shapes)
        self.window.rebuild_properties_dock()                       # records what the panel shows
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        moved = copy.deepcopy(shapes)
        moved[2]["points"][4]["x"] += 3.0
        self.window.command({"op": "set_shapes", "id": "r", "shapes": moved})
        self.assertFalse(self.window._panel_rebuild_timer.isActive())
        self.window.command({"op": "undo"})
        self.assertFalse(self.window._panel_rebuild_timer.isActive())
        self.assertEqual(builds, [])
        renamed = copy.deepcopy(shapes)
        renamed[2]["name"] = "renamed"
        self.window.command({"op": "set_shapes", "id": "r", "shapes": renamed})
        self.assertTrue(self.window._panel_rebuild_timer.isActive())     # the list shows names: it will be built, once the interface is quiet
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()
        self.assertEqual(builds, ["r"])

    def window_items(self):
        return max(Window.HEAVY_PANEL_ITEMS, 8)

    def test_the_panel_rebuild_waits_for_a_held_handle(self):
        shapes = [blob(i) for i in range(self.window_items())]
        viewer = self.open(shapes)
        self.window.rebuild_properties_dock()
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        target = (shapes[5]["points"][1]["x"], shapes[5]["points"][1]["y"])
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(*target))     # selects another shape: a rebuild is wanted
        self.assertTrue(self.window._panel_rebuild_timer.isActive())
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()                             # the timer fires mid-drag
        self.assertEqual(builds, [])
        self.assertTrue(self.window._panel_rebuild_timer.isActive())  # and waits
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(*target))
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()
        # the selection moved to another shape and nothing else changed: the panel on screen is brought up to date in place
        # (the selected shape's controls are built again, in the next event; the list and the knobs stay), so no whole panel is built
        APP.processEvents()
        self.assertEqual(builds, [])
        name = self.window.properties.widget().findChild(QLineEdit, "roto-shape-name")
        self.assertEqual(name.text(), "blob5")

    def test_a_drag_that_keys_a_point_updates_the_panel_in_place(self):
        shapes = [blob(i) for i in range(6)]
        for shape in shapes:
            for point in shape["points"]:
                point["x"] = {"value": point["x"], "curve": {"interpolation": "smooth",
                                                              "keys": [{"frame": 1, "value": point["x"]}, {"frame": 3, "value": point["x"]}]}}
        viewer = self.open(shapes)
        viewer.roto_selected_shape_index = 2
        self.window.rebuild_properties_dock()
        panel = self.window.properties.widget()
        detail_before = panel.findChild(QWidget, "roto-shape-detail")
        count_before = panel.findChild(QLabel, "roto-spline-key-count").text()
        list_before = panel.findChild(QWidget, "roto-shape-list")
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        self.window._panel_rebuild_timer.stop()
        key_button_before = panel.findChild(QWidget, "roto-shape-key-2")
        self.window.command({"op": "set_shapes", "id": "r", "shapes": self._keyed_point(shapes, 2, 3)})
        self.assertTrue(self.window._panel_rebuild_timer.isActive())
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()
        self.assertEqual(builds, [], "the panel is updated in place, not built again")
        self.assertIs(self.window.properties.widget(), panel)
        self.assertIs(panel.findChild(QWidget, "roto-shape-list"), list_before, "the list stays")
        self.assertIsNot(panel.findChild(QWidget, "roto-shape-key-2"), key_button_before, "the shape's own row is rebuilt")
        self.assertIs(panel.findChild(QWidget, "roto-shape-detail"), detail_before,
                      "the selected shape's controls are not built again for a point that gained a key")
        count_after = panel.findChild(QLabel, "roto-spline-key-count").text()
        self.assertNotEqual(count_after, count_before, "its key count follows the new key")
        self.assertEqual(count_after, "Key 1 / 3")
        self.assertEqual(self.window._panel_signature_shown, self.window._roto_panel_signature("r"))

    @staticmethod
    def _keyed_point(shapes, shape_index, point_index):
        edited = copy.deepcopy(shapes)
        value = edited[shape_index]["points"][point_index]["x"]
        keys = value["curve"]["keys"] + [{"frame": 2, "value": 1.5}]
        edited[shape_index]["points"][point_index]["x"] = {"value": 1.5, "curve": {**value["curve"], "keys": sorted(keys, key=lambda k: k["frame"])}}
        return edited

    def test_the_tool_strip_is_not_laid_out_again_while_nothing_it_shows_changed(self):
        viewer = self.open([triangle()])
        APP.processEvents()
        toolbar = viewer.roto_toolbar
        toolbar.refresh()
        laid_out = []
        original = toolbar.settings_layout.activate
        toolbar.settings_layout.activate = lambda: laid_out.append(1) or original()
        for _ in range(5):
            toolbar.refresh()                  # every Viewer paint asks
        self.assertEqual(laid_out, [], "an unchanged strip is left alone (a relayout per paint repainted the window 345 times a second)")
        toolbar._settings_signature = None     # the settings were changed elsewhere: the strip is built again
        toolbar.refresh()
        self.assertGreater(len(laid_out), 0)
        toolbar.settings_layout.activate = original

    def test_a_press_and_a_release_repaint_the_overlay_without_waiting_for_a_picture(self):
        shapes = [blob(i) for i in range(3)]
        viewer = self.open(shapes)
        APP.processEvents()
        from PySide6.QtCore import QEvent, QObject
        paints = []

        class Spy(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint:
                    paints.append(1)
                return False
        spy = Spy()
        viewer.viewport().installEventFilter(spy)
        target = (shapes[1]["points"][2]["x"], shapes[1]["points"][2]["y"])
        end = self.at(target[0] + 5, target[1] + 3)
        QTest.mousePress(viewer.viewport(), Qt.MouseButton.LeftButton, pos=self.at(*target))
        QTest.mouseMove(viewer.viewport(), end)
        APP.processEvents()
        self.window.timer.stop()
        APP.processEvents()
        before = len(paints)
        QTest.mouseRelease(viewer.viewport(), Qt.MouseButton.LeftButton, pos=end)
        APP.processEvents()
        self.assertGreater(len(paints), before, "the release is drawn as committed at once")
        viewer.viewport().removeEventFilter(spy)

    def test_a_drawn_shape_adds_its_row_in_place_and_selects_it(self):
        from nodebased import rotolist
        shapes = [blob(i) for i in range(5)]
        viewer = self.open(shapes)
        self.window.rebuild_properties_dock()
        panel = self.window.properties.widget()
        listing = panel.findChild(rotolist.ShapeListTree, "roto-shape-list")
        builds = []
        original = self.window.build_node_panel
        self.window.build_node_panel = lambda key: builds.append(key) or original(key)
        self.window._panel_rebuild_timer.stop()
        viewer.roto_selected_shape_index = 5
        self.window.command({"op": "set_shapes", "id": "r", "shapes": shapes + [blob(5)]})
        self.window._panel_rebuild_timer.stop()
        self.window._run_panel_rebuild()
        APP.processEvents()
        self.assertEqual(builds, [], "the panel is not built again for a new shape")
        self.assertIs(self.window.properties.widget(), panel)
        self.assertEqual(listing.count(), 6)
        top = listing.invisibleRootItem().child(0)
        self.assertEqual(listing.parse_id(top.data(0, Qt.ItemDataRole.UserRole)), ("item", 5), "the newest shape is on top")
        self.assertEqual(panel.findChild(QLineEdit, "roto-shape-name").text(), "blob5", "the new shape's controls are shown")
        # the list is what a fresh build would make: same items in the same order
        fresh = rotolist.ShapeListTree(self.window, "r", True, lambda row: None)
        ids = lambda tree: [tree.parse_id(tree.invisibleRootItem().child(i).data(0, Qt.ItemDataRole.UserRole)) for i in range(tree.invisibleRootItem().childCount())]
        self.assertEqual(ids(listing), ids(fresh))

    def test_the_overlay_cache_paints_what_a_cold_paint_does_and_a_drag_copies_only_its_shape(self):
        shapes = [blob(i) for i in range(6)]
        viewer = self.open(shapes)
        viewer.roto_selected_shape_index = 2
        viewer.roto_selected_points = {(1, 3), (2, 4)}

        def grab():
            image = QImage(300, 300, QImage.Format.Format_ARGB32)
            image.fill(0)
            painter = QPainter(image)
            viewer.paint_overlays(painter, QRectF(0, 0, 300, 300))
            painter.end()
            return image

        viewer._roto_overlay_cache = (None, {})
        cold = grab()
        warm = grab()                                   # served from the per-shape cache
        self.assertEqual(cold, warm)
        self.assertEqual(len(viewer._roto_overlay_cache[1]), len(shapes))
        cached_path = viewer._roto_overlay_cache[1][4]["path"]
        viewer.roto_drag = {"kind": "tension", "scene": QPointF(5, 5)}      # a drag that moves no point
        grab()
        self.assertIs(viewer._roto_overlay_cache[1][4]["path"], cached_path)  # the outline is not rebuilt per paint
        viewer.roto_drag = None
        viewer.roto_selected_points = set()
        self.assertNotEqual(grab(), cold)               # the selected points show without a rebuilt cache

    def test_stroke_coverage_by_dab_window_equals_the_whole_region_computation(self):
        import numpy as np
        from nodebased import paint

        def reference(width, height, points, size, spacing, hardness=1.0, origin=(0, 0), dynamic_hardness=False,
                      pressure_size=True, pressure_opacity=True):
            ox, oy = map(int, origin)
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            hardness = min(1.0, max(0.0, float(hardness)))
            coverage = np.zeros((height, width), np.float32)
            for a, b in zip(points, points[1:] or points):
                x0, y0, p0 = a["x"], a["y"], a["pressure"]
                x1, y1, p1 = b["x"], b["y"], b["pressure"]
                length = math.hypot(x1 - x0, y1 - y0)
                count = max(1, int(math.ceil(length / max(spacing * size, 0.25))))
                for i in range(count + 1):
                    t = i / count
                    x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                    pressure = p0 + t * (p1 - p0)
                    radius = max(0.25, size * (pressure if pressure_size else 1.0) * 0.5)
                    dab_hardness = hardness * pressure if dynamic_hardness else hardness
                    d = np.sqrt((xx + ox + 0.5 - x) ** 2 + (yy + oy + 0.5 - y) ** 2)
                    ramp = np.clip((radius + 0.5 - d) / ((1.0 - dab_hardness) * radius + 1.0), 0, 1)
                    if dab_hardness < 1.0:
                        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
                    coverage = np.maximum(coverage, ramp * (pressure if pressure_opacity else 1.0))
            return coverage

        rng = np.random.default_rng(11)
        for case in range(12):
            points = [{"x": float(rng.uniform(-10, 70)), "y": float(rng.uniform(-10, 50)), "pressure": float(rng.uniform(0.2, 1.0))}
                      for _ in range(int(rng.integers(1, 6)))]
            kwargs = dict(size=float(rng.uniform(2, 18)), spacing=float(rng.choice([0.05, 0.2, 0.5])),
                          hardness=float(rng.choice([0.0, 0.3, 1.0])), origin=(int(rng.integers(-5, 6)), int(rng.integers(-5, 6))),
                          dynamic_hardness=bool(case % 2), pressure_size=bool(case % 3), pressure_opacity=bool(case % 4))
            np.testing.assert_array_equal(paint._distance_grid(64, 48, points, **kwargs), reference(64, 48, points, **kwargs))

    def test_a_stroke_written_in_place_matches_the_copying_route_and_leaves_the_plate_alone(self):
        import numpy as np
        from nodebased import paint
        from tests.test_roto_paint import stroke
        plate = np.random.default_rng(3).random((40, 50, 4), dtype=np.float32)
        before = plate.copy()
        items = [stroke(points=[{"x": 8.0 + 5 * k, "y": 10.0 + 2 * k, "pressure": 0.8} for k in range(5)], name=f"s{i}")
                 for i in range(4)]
        painted = paint.rasterise(plate, items, 1, source=plate)
        np.testing.assert_array_equal(plate, before)                         # the input is never written
        expected = plate.copy()
        for item in items:
            expected = paint.apply_stroke(expected, item, 1, source=plate)   # the copying route
        np.testing.assert_array_equal(painted, expected)
        self.assertFalse(np.array_equal(painted, plate))


class PaintPanelInPlaceTests(unittest.TestCase):
    def setUp(self):
        from tests.test_roto_paint_ui import document as paint_document
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(paint_document())
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.window.graph.items_by_id["paint"].setSelected(True)
        APP.processEvents()
        self.window.timer.stop()

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    def stroke(self, name, y):
        from tests.test_roto_paint import stroke
        return stroke(points=[{"x": 10.0, "y": y, "pressure": 1.0}, {"x": 40.0, "y": y + 2, "pressure": 1.0}], name=name)

    def test_a_finished_stroke_adds_its_row_to_the_list_without_building_the_panel(self):
        from nodebased import rotolist
        window = self.window
        window.command({"op": "set_paint_items", "id": "paint", "items": [self.stroke("stroke1", 10.0)]})
        window._panel_rebuild_timer.stop()
        window._run_panel_rebuild()
        window.flush_inspect()
        panel = window.properties.widget()
        listing = panel.findChild(rotolist.ShapeListTree, "rotopaint-layer-list")
        self.assertEqual(listing.count(), 1)
        builds = []
        original = window.build_node_panel
        window.build_node_panel = lambda key: builds.append(key) or original(key)
        items = window.dispatcher.document["node_data"]["paint"]["items"]
        window.command({"op": "set_paint_items", "id": "paint", "items": items + [self.stroke("stroke2", 30.0)]})
        self.assertTrue(window._panel_rebuild_timer.isActive())
        window._panel_rebuild_timer.stop()
        window._run_panel_rebuild()
        self.assertEqual(builds, [], "the panel is not built again for a new stroke")
        self.assertIs(window.properties.widget(), panel)
        self.assertEqual(listing.count(), 2)
        top = listing.invisibleRootItem().child(0)
        self.assertEqual(listing.parse_id(top.data(0, Qt.ItemDataRole.UserRole)), ("item", 1), "the newest stroke is on top")
        self.assertIsNotNone(listing.itemWidget(top, 0))
        # the list a fresh build makes is the same list
        fresh = rotolist.ShapeListTree(window, "paint", False, lambda row: None)
        self.assertEqual([listing.parse_id(listing.invisibleRootItem().child(i).data(0, Qt.ItemDataRole.UserRole)) for i in range(2)],
                         [fresh.parse_id(fresh.invisibleRootItem().child(i).data(0, Qt.ItemDataRole.UserRole)) for i in range(2)])

    def test_removing_a_stroke_builds_the_panel(self):
        window = self.window
        window.command({"op": "set_paint_items", "id": "paint", "items": [self.stroke("stroke1", 10.0), self.stroke("stroke2", 30.0)]})
        window._panel_rebuild_timer.stop()
        window._run_panel_rebuild()
        window.flush_inspect()
        builds = []
        original = window.build_node_panel
        window.build_node_panel = lambda key: builds.append(key) or original(key)
        window.command({"op": "set_paint_items", "id": "paint", "items": [self.stroke("stroke1", 10.0)]})
        window._panel_rebuild_timer.stop()
        window._run_panel_rebuild()
        self.assertEqual(builds, ["paint"])


if __name__ == "__main__":
    unittest.main()
