"""A click on the direct picture lands on the handle the Viewer would pick (Lane 2, 10/20).

Every handle kind is dragged twice from the same pixels: once through the Viewer's viewport and once
through the direct picture's widget, at device scale 1.0 and 1.75, zoom 0.5, 1 and 3, panned off centre.
The document edits must match: same knob values, same undo entries, one undo step per drag.

The direct widget here is the real `DirectViewerWidget` wired by `Window.set_direct_viewer`; only the GPU
presenter is left out (its `submit` records the picture rectangle), so the test runs without a display.
The Viewer reports zoom in logical pixels and the direct widget in physical ones; the scale cases are the
ones that would fail if a tolerance used the wrong unit.
"""
import copy
import os
import unittest
import unittest.mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (keeps QSettings out of the real user file)
from tests.waiting import wait_until
from nodebased import directviewer as dv
from nodebased import handles2d
from nodebased import shapes as shape_model
from nodebased import warps
from nodebased.app import STYLE, Window
from nodebased.core import Dispatcher, empty_document, viewer_roi

try:
    from rendercanvas.qt import QRenderWidget  # noqa: F401
    HAVE_CANVAS = True
except Exception:   # pragma: no cover - the package job without rendercanvas
    HAVE_CANVAS = False

APP = QApplication.instance() or QApplication([])
APP.setStyle("Fusion")
APP.setStyleSheet(STYLE)

WIDTH, HEIGHT = 640, 360
NO_MOD, CTRL, SHIFT, ALT = (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ControlModifier,
                            Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.AltModifier)
SCALES = (1.0, 1.75)
ZOOMS = (0.5, 1.0, 3.0)
PAN_PX = (47, -31)          # how far the handle sits from the surface centre, in logical pixels
NEAR_MISS_PX = 6            # a press this far from the handle is still inside every tolerance


def document():
    d = Dispatcher(empty_document())
    d.document["nodes"] = {}
    d.document["view"] = None
    for command in (
        {"op": "create", "id": "c", "type": "Constant", "pos": [0, 0],
         "params": {"width": WIDTH, "height": HEIGHT, "red": 0.4, "green": 0.4, "blue": 0.4, "alpha": 1.0}},
        {"op": "create", "id": "b", "type": "Constant", "pos": [0, 100],
         "params": {"width": WIDTH, "height": HEIGHT, "red": 0.1, "green": 0.2, "blue": 0.7, "alpha": 1.0}},
        {"op": "create", "id": "r", "type": "Roto", "pos": [150, 0]},
        {"op": "set", "id": "r", "param": "width", "value": WIDTH},
        {"op": "set", "id": "r", "param": "height", "value": HEIGHT},
        {"op": "create", "id": "t", "type": "Tracker", "pos": [300, 0]},
        {"op": "connect", "id": "t", "input": "image", "source": "c"},
        {"op": "set_tracks", "id": "t", "tracks": [{"name": "track1", "enabled": 1, "x": 220.0, "y": 140.0}]},
        {"op": "create", "id": "x", "type": "Transform", "pos": [450, 0],
         "params": {"center_x": 320.0, "center_y": 180.0, "rotate": 8.0, "scale": 0.8,
                    "translate_x": 12.0, "translate_y": -9.0}},
        {"op": "connect", "id": "x", "input": "image", "source": "c"},
        {"op": "create", "id": "g", "type": "GridWarp", "pos": [600, 0]},
        {"op": "connect", "id": "g", "input": "image", "source": "c"},
        {"op": "create", "id": "f", "type": "Flare", "pos": [750, 0],
         "params": {"position_x": 400.0, "position_y": 120.0}},
        {"op": "connect", "id": "f", "input": "image", "source": "c"},
    ):
        d.execute(command)
    shape = {"name": "blob", "mode": "union", "opacity": 1.0, "feather": 12.0, "feather_model": "edge",
             "points": [
                 {"x": 140.0, "y": 90.0, "in_x": -25.0, "in_y": 10.0, "out_x": 25.0, "out_y": -10.0},
                 {"x": 330.0, "y": 80.0, "in_x": -30.0, "in_y": -8.0, "out_x": 30.0, "out_y": 8.0},
                 {"x": 350.0, "y": 240.0, "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0},
                 {"x": 150.0, "y": 250.0, "in_x": 20.0, "in_y": 12.0, "out_x": -20.0, "out_y": -12.0}]}
    d.execute({"op": "set_shapes", "id": "r", "shapes": [shape]})
    d.document["view"] = "c"
    return d.document


def numbers_close(a, b, path="document"):
    """Why two documents differ, or None. Floats agree to 1e-6 (the two surfaces round the same pointer
    position through different but equivalent arithmetic)."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return None if a == b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return None if abs(a - b) <= 1e-6 else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for key in a:
            found = numbers_close(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            found = numbers_close(x, y, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def hostless_make_widget():
    """`directviewer.make_widget` without the GPU: the real widget, its mouse handlers and its Window wiring."""
    real = dv.make_widget

    def make(parent=None):
        widget = real(parent)

        def submit(image, rect, format_rect=None, generation=None):
            widget.rect = tuple(rect)
            widget.format_rect = tuple(format_rect or rect)
            return True
        widget.submit = submit
        widget.submit_linear = lambda *args, **kwargs: False
        return widget
    return make


@unittest.skipUnless(HAVE_CANVAS, "rendercanvas unavailable")
class HitOrderBase(unittest.TestCase):
    def setUp(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.window = Window(document(), agent_name="nodebased-test-hitorder")
        self.window.resize(1000, 760)
        self.window.graph_dock.hide()
        self.window.nodes_dock.hide()
        self.window._default_split_pending = False
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None
                                   and self.window.viewer.format_rect is not None))
        self.viewer = self.window.viewer
        self.window.preferences.set_direct_edit(True)
        patcher = unittest.mock.patch.object(dv, "make_widget", hostless_make_widget())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.window.direct_action.setChecked(True)
        APP.processEvents()
        self.widget = self.window.direct_widget
        self.assertIsNotNone(self.widget)
        self.size = (self.viewer.viewport().width() // 4 * 4, self.viewer.viewport().height() // 4 * 4)
        self.widget.setFixedSize(*self.size)
        self.widget.show()
        self.widget.rect = (0, 0, WIDTH, HEIGHT)
        self.widget.format_rect = self.widget.rect
        APP.processEvents()
        self.base_undo = len(self.window.dispatcher.undo_stack)

    def tearDown(self):
        from tests.test_desktop import release_window
        self.addCleanup(release_window, self)
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()

    # -- state ------------------------------------------------------------------------------------
    def settle(self):
        APP.processEvents()
        idle = lambda: not self.window.busy and not len(self.window.preview_queue) and self.viewer.format_rect is not None
        self.assertTrue(wait_until(idle, timeout=10),
                        f"busy={self.window.busy} queued={len(self.window.preview_queue)} "
                        f"format={self.viewer.format_rect is not None}: {self.window.viewer_info.text()}: "
                        f"{self.window.statusBar().currentMessage()}")
        APP.processEvents()

    def show_node(self, key):
        graph = self.window.graph
        graph.scene().clearSelection()
        graph.items_by_id[key].setSelected(True)
        self.window.command({"op": "view", "id": key})
        self.settle()
        self.assertEqual(self.window.dispatcher.document["view"], key)
        self.base_undo = len(self.window.dispatcher.undo_stack)

    def view_only_state(self):
        v = self.viewer
        return {"wipe": dict(v.wipe), "roto": (sorted(v.roto_selected_points), v.roto_selected_shape_index),
                "tracker": getattr(self.window, "_tracker_selected_index", None)}

    def snapshot(self):
        return {"document": copy.deepcopy(self.window.dispatcher.document),
                "undo": len(self.window.dispatcher.undo_stack) - self.base_undo,
                "view_only": self.view_only_state()}

    def rewind(self):
        """Back to the state before the drag: undo every entry the drag made."""
        while len(self.window.dispatcher.undo_stack) > self.base_undo:
            self.window.command({"op": "undo"})
        APP.processEvents()

    # -- the pointer ------------------------------------------------------------------------------
    def place_surfaces(self, handle, zoom, scale):
        """Pan and zoom both surfaces so `handle` (a scene point) is PAN_PX from their centre. Returns the
        logical pixel under which the handle sits on the Viewer (the same pixel is used on the direct picture)."""
        viewer = self.viewer
        viewer.resetTransform()
        viewer.scale(zoom, zoom)
        viewer.auto_fit = False
        offset = QPointF(-PAN_PX[0] / zoom, -PAN_PX[1] / zoom)
        viewer.centerOn(handle + offset)
        APP.processEvents()
        inverse, _ = viewer.viewportTransform().inverted()
        width, height = self.size
        centre = inverse.map(QPointF(width / 2.0, height / 2.0))
        widget = self.widget
        widget.get_pixel_ratio = lambda: scale
        widget.get_physical_size = lambda: (int(round(width * scale)), int(round(height * scale)))
        widget.view = (centre.x(), centre.y(), zoom * scale)       # physical pixels per scene unit
        widget.auto_fit = False
        return viewer.mapFromScene(handle)

    def drag(self, surface, start, delta, modifiers=NO_MOD, steps=3):
        target = surface.viewport() if surface is self.viewer else surface
        end = start + QPoint(*delta)
        QTest.mousePress(target, Qt.MouseButton.LeftButton, modifiers, start)
        for step in range(1, steps + 1):
            QTest.mouseMove(target, start + QPoint(round(delta[0] * step / steps), round(delta[1] * step / steps)))
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, modifiers, end)
        APP.processEvents()


class HandleCase:
    """One handle kind: how to get the node on screen, where the handle is, how to drag it."""
    def __init__(self, name, node, handle, delta=(34, 22), modifiers=NO_MOD, prepare=None):
        self.name, self.node, self.handle, self.delta = name, node, handle, delta
        self.modifiers, self.prepare = modifiers, prepare


def origin(window):
    rect = window.viewer.format_rect
    return rect.left(), rect.top()


def scene(window, x, y):
    ox, oy = origin(window)
    return QPointF(ox + x, oy + y)


def roto_point(window):
    return scene(window, 140.0, 90.0)


def roto_tangent(window):
    window.viewer.roto_selected_shape_index = 0
    return scene(window, 140.0 + 25.0, 90.0 - 10.0)


def roto_feather(window):
    window.viewer.roto_selected_shape_index = 0
    resolved = window.dispatcher.document["node_data"]["r"]["shapes"][0]
    handle = shape_model.feather_handle_points(resolved["points"], resolved["feather"])[2]
    return scene(window, handle["x"], handle["y"])


def tracker_pattern(window):
    pattern = float(window.dispatcher.document["nodes"]["t"]["params"].get("pattern_radius", 8))
    return scene(window, 220.0 - pattern, 140.0)


def tracker_search(window):
    search = float(window.dispatcher.document["nodes"]["t"]["params"].get("search_radius", 24))
    return scene(window, 220.0 + search, 140.0)


def tracker_point(window):
    return scene(window, 220.0, 140.0)


def transform_geometry(window):
    p = window.dispatcher.document["nodes"]["x"]["params"]
    corners = handles2d.box_corners(WIDTH, HEIGHT, p["translate_x"], p["translate_y"], p["rotate"],
                                    p["scale"], p["center_x"], p["center_y"])
    pivot = handles2d.pivot_point(p["translate_x"], p["translate_y"], p["center_x"], p["center_y"])
    return corners, pivot


def transform_centre(window):
    return scene(window, *transform_geometry(window)[1])


def transform_scale(window):
    return scene(window, *transform_geometry(window)[0][0])


def transform_rotate(window):
    corners, pivot = transform_geometry(window)
    ring = 1.15
    x, y = corners[0]
    return scene(window, pivot[0] + (x - pivot[0]) * ring, pivot[1] + (y - pivot[1]) * ring)


def warp_pin(window):
    grid = warps.default_grid(WIDTH, HEIGHT, window.dispatcher.document["nodes"]["g"]["params"]["rows"],
                              window.dispatcher.document["nodes"]["g"]["params"]["columns"])
    x, y = grid[1][1][:2]
    return scene(window, float(x), float(y))


def flare_handle(window):
    return scene(window, 400.0, 120.0)


def roi_box(window):
    return window.viewer.roi_scene_rect()


def roi_corner(window):
    return roi_box(window).topLeft()


def roi_edge(window):
    box = roi_box(window)
    return QPointF(box.center().x(), box.bottom())


def wipe_geometry(window):
    return window.viewer._wipe_scene_geometry()


def wipe_centre(window):
    return wipe_geometry(window)[0]


def wipe_line(window):
    centre, (dx, dy), _normal, _handle = wipe_geometry(window)
    return QPointF(centre.x() + dx * 90.0, centre.y() + dy * 90.0)


def wipe_rotate(window):
    return wipe_geometry(window)[3]


def prepare_roi(window):
    window.set_viewer_roi(on=True, rect=[0.25, 0.25, 0.75, 0.75])


def prepare_wipe(window):
    window.command({"op": "viewer_input", "slot": 2, "id": "b", "activate": False})
    window.command({"op": "viewer_compare", "b": 2, "mode": "wipe"})


CASES = (
    HandleCase("roi corner", "c", roi_corner, prepare=prepare_roi),
    HandleCase("roi edge", "c", roi_edge, prepare=prepare_roi),
    HandleCase("roto point", "r", roto_point),
    HandleCase("roto point with feather", "r", roto_point, modifiers=CTRL),
    HandleCase("roto tangent", "r", roto_tangent),
    HandleCase("roto feather handle", "r", roto_feather),
    HandleCase("tracker pattern square", "t", tracker_pattern),
    HandleCase("tracker search square", "t", tracker_search),
    HandleCase("tracker point", "t", tracker_point),
    HandleCase("transform centre", "x", transform_centre),
    HandleCase("transform pivot (ctrl)", "x", transform_centre, modifiers=CTRL),
    HandleCase("transform rotate ring", "x", transform_rotate),
    HandleCase("transform scale corner", "x", transform_scale),
    HandleCase("warp pin", "g", warp_pin),
    HandleCase("flare", "f", flare_handle),
    HandleCase("wipe centre", "c", wipe_centre, prepare=prepare_wipe),
    HandleCase("wipe line", "c", wipe_line, prepare=prepare_wipe),
    HandleCase("wipe rotate handle", "c", wipe_rotate, prepare=prepare_wipe),
)


def selected_cases():
    """HITORDER_ONLY=roto,wipe runs just the cases whose names contain one of the words (for development)."""
    only = [word for word in os.environ.get("HITORDER_ONLY", "").split(",") if word]
    return [case for case in CASES if not only or any(word in case.name for word in only)]


class HitOrderTests(HitOrderBase):
    def prepared(self, case):
        """The case's node on screen with nothing left over from the previous case (an ROI box or a wipe would
        take the press first). The slow part runs once per case; every drag then starts from the same state."""
        if getattr(self, "_prepared_case", None) is not case:
            self.viewer.fit()
            APP.processEvents()
            self.window.set_viewer_roi(on=False)
            self.window.command({"op": "viewer_compare", "b": 2, "mode": "A only"})
            self.settle()
            self.show_node(case.node)
            if case.prepare is not None:
                case.prepare(self.window)
                self.settle()
            self._prepared_case = case
        self.viewer.reset_wipe()
        self.viewer.roto_selected_points = set()
        self.viewer.roto_selected_shape_index = -1
        self.base_undo = len(self.window.dispatcher.undo_stack)

    def run_surface(self, case, surface, zoom, scale, offset):
        """One drag through `surface`; returns the snapshot afterwards and the cursor seen mid-drag."""
        self.prepared(case)
        with self.viewer.hit_surface(zoom):     # the wipe's rotate handle sits a fixed number of pixels out
            handle = case.handle(self.window)
        before = self.snapshot()
        start = self.place_surfaces(handle, zoom, scale) + QPoint(*offset)
        target = surface.viewport() if surface is self.viewer else surface
        end = start + QPoint(*case.delta)
        QTest.mousePress(target, Qt.MouseButton.LeftButton, case.modifiers, start)
        mid_cursor = (self.viewer.cursor().shape(), self.widget.cursor().shape())
        QTest.mouseMove(target, start + QPoint(case.delta[0] // 2, case.delta[1] // 2))
        QTest.mouseMove(target, end)
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, case.modifiers, end)
        APP.processEvents()
        after = self.snapshot()
        end_cursor = (self.viewer.cursor().shape(), self.widget.cursor().shape())
        # One undo step per drag, and it puts the document back.
        self.assertLessEqual(after["undo"], 1, f"{case.name}: {after['undo']} undo entries")
        self.rewind()
        self.assertIsNone(numbers_close(before["document"], self.window.dispatcher.document, "after undo"),
                          f"{case.name}: one undo restores the document")
        return before, after, mid_cursor, end_cursor

    def test_every_handle_drag_edits_the_document_identically_on_both_surfaces(self):
        for case in selected_cases():
            for scale in SCALES:
                for zoom in ZOOMS:
                    for offset in ((0, 0), (NEAR_MISS_PX, 0)):
                        label = f"{case.name}, scale {scale}, zoom {zoom}, offset {offset}"
                        with self.subTest(label):
                            before, via_viewer, _, _ = self.run_surface(case, self.viewer, zoom, scale, offset)
                            _, via_direct, direct_mid, direct_end = self.run_surface(
                                case, self.widget, zoom, scale, offset)
                            self.assertEqual(via_direct["undo"], via_viewer["undo"], label)
                            self.assertIsNone(numbers_close(via_viewer["document"], via_direct["document"]), label)
                            self.assertEqual(via_viewer["view_only"]["wipe"], via_direct["view_only"]["wipe"], label)
                            self.assertEqual(via_viewer["view_only"]["roto"], via_direct["view_only"]["roto"], label)
                            self.assertEqual(via_viewer["view_only"]["tracker"], via_direct["view_only"]["tracker"], label)
                            if offset == (0, 0):
                                changed = (numbers_close(before["document"], via_viewer["document"]) is not None
                                           or via_viewer["view_only"]["wipe"] != before["view_only"]["wipe"])
                                self.assertTrue(changed, f"{label}: the Viewer drag changed nothing")
                                self.assertEqual(direct_mid[1], direct_mid[0], f"{label}: cursor while held")
                                self.assertNotEqual(direct_mid[1], Qt.CursorShape.ArrowCursor, label)
                            self.assertEqual(direct_end[1], direct_end[0], f"{label}: cursor after release")

    def test_a_press_off_every_handle_edits_nothing_on_either_surface(self):
        case = CASES[2]                 # a Roto point: far from it, nothing is grabbed
        for scale in SCALES:
            for zoom in ZOOMS:
                with self.subTest(scale=scale, zoom=zoom):
                    before, via_viewer, _, _ = self.run_surface(case, self.viewer, zoom, scale, (70, 0))
                    _, via_direct, _, _ = self.run_surface(case, self.widget, zoom, scale, (70, 0))
                    self.assertIsNone(numbers_close(via_viewer["document"], via_direct["document"]))
                    self.assertEqual(via_viewer["undo"], via_direct["undo"])


class ProxyViewportTests(HitOrderBase):
    def test_a_proxy_tier_preview_with_the_view_zoomed_onto_the_lower_right_renders(self):
        """Found by this module's first runs: the visible region (scene units) was compared with proxy-tier
        bounds unscaled, so a drag preview at tier 8 with the view past (W/8, H/8) got an empty frame and
        "Internal error ... Invalid image dimensions", which also removed every handle until the next render."""
        self.show_node("c")
        self.viewer.resetTransform()
        self.viewer.scale(3.0, 3.0)
        self.viewer.centerOn(QPointF(560.0, 300.0))
        APP.processEvents()
        self.window.request_preview(tier_override=8)
        self.settle()
        self.assertNotIn("error", self.window.viewer_info.text().lower())
        self.assertIsNotNone(self.viewer.format_rect)


class RoutingTests(HitOrderBase):
    def test_pan_zoom_and_fit_stay_on_the_view_for_alt_middle_wheel_and_double_click(self):
        self.show_node("x")
        document_before = copy.deepcopy(self.window.dispatcher.document)
        handle = transform_centre(self.window)
        start = self.place_surfaces(handle, 1.0, 1.0)
        view = self.widget.view
        self.drag(self.widget, start, (30, 20), modifiers=ALT)
        self.assertNotEqual(self.widget.view[:2], view[:2])
        self.assertFalse(self.widget.auto_fit)
        QTest.mousePress(self.widget, Qt.MouseButton.MiddleButton, NO_MOD, start)
        QTest.mouseMove(self.widget, start + QPoint(10, 5))
        QTest.mouseRelease(self.widget, Qt.MouseButton.MiddleButton, NO_MOD, start + QPoint(10, 5))
        QTest.mouseDClick(self.widget, Qt.MouseButton.LeftButton, NO_MOD, start)
        self.assertTrue(self.widget.auto_fit)
        APP.processEvents()
        self.assertIsNone(numbers_close(document_before, self.window.dispatcher.document))
        self.assertEqual(len(self.window.dispatcher.undo_stack), self.base_undo)

    def test_the_setting_turns_routing_off_and_the_title_follows(self):
        self.show_node("x")
        self.assertNotIn("view only", self.window.direct_dock.windowTitle())
        self.window.set_direct_edit(False)
        self.assertIn("view only", self.window.direct_dock.windowTitle())
        before = copy.deepcopy(self.window.dispatcher.document)
        start = self.place_surfaces(transform_centre(self.window), 1.0, 1.0)
        self.drag(self.widget, start, (30, 20))
        self.assertIsNone(numbers_close(before, self.window.dispatcher.document))
        self.window.set_direct_edit(True)
        self.assertNotIn("view only", self.window.direct_dock.windowTitle())
        self.drag(self.widget, start, (30, 20))
        self.assertIsNotNone(numbers_close(before, self.window.dispatcher.document))
        self.assertTrue(self.window.preferences.direct_edit())

    def test_preference_defaults_on(self):
        QSettings("NodeBased", "NodeBased").clear()
        self.assertTrue(self.window.preferences.direct_edit())


if __name__ == "__main__":
    unittest.main()
