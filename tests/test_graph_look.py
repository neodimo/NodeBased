"""New look, step 2: node colours by family, wire colours and shapes, the Curved / Right angle toggle,
port directions and the minimap (Lane 2, NL2)."""
import tests.isolation  # noqa: F401  (settings of the test run, not the developer's)

import unittest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath
from PySide6.QtWidgets import QStyleOptionGraphicsItem

from nodebased import graphlook, nodecatalog, theme
from nodebased.app import Preferences, Window
from nodebased.core import SPECS
from tests.test_desktop import APP, wait_until

P = QPointF


def line_segments(path):
    """The straight (from, to) runs of a QPainterPath; corner curves are skipped."""
    out, previous = [], None
    index = 0
    while index < path.elementCount():
        element = path.elementAt(index)
        point = QPointF(element.x, element.y)
        if element.type == QPainterPath.ElementType.MoveToElement:
            previous = point
        elif element.type == QPainterPath.ElementType.LineToElement:
            out.append((previous, point))
            previous = point
        else:  # a curve: its control points and end point follow
            previous = QPointF(path.elementAt(index + 2).x, path.elementAt(index + 2).y)
            index += 2
        index += 1
    return out


class FamilyTableTests(unittest.TestCase):
    def test_every_registered_node_type_has_one_of_the_thirteen_families(self):
        self.assertEqual(len(theme.FAMILY_COLORS), 13)
        for kind in SPECS:
            family = nodecatalog.node_family(kind)
            self.assertIn(family, theme.FAMILY_COLORS, kind)
            self.assertEqual(theme.family_color(kind), theme.FAMILY_COLORS[family], kind)

    def test_the_table_has_no_stale_entries(self):
        self.assertEqual(set(nodecatalog.FAMILY_NAMES), set(theme.FAMILY_COLORS))
        for kind, family in nodecatalog.OTHER_KIND_FAMILY.items():
            self.assertIn(kind, SPECS, kind)
            self.assertEqual(nodecatalog.node_category(kind), "Other", kind)
            self.assertIn(family, theme.FAMILY_COLORS)

    def test_a_few_kinds_land_where_an_artist_expects(self):
        for kind, family in (("Read", "Image"), ("Grade", "Color"), ("Blur", "Filter"), ("Merge", "Merge"),
                             ("Transform", "Transform"), ("Roto", "Draw"), ("Sphere3D", "3D"),
                             ("Dot", "Metadata"), ("ContactSheet", "Image")):
            self.assertEqual(nodecatalog.node_family(kind), family, kind)


class WirePathTests(unittest.TestCase):
    OUT = P(95, 52)

    def test_a_mask_wire_ends_with_a_horizontal_segment_entering_from_the_right(self):
        end = P(190, 126)
        for source in (P(400, 52), P(260, 52), P(50, 52), P(400, 252)):
            for mode in graphlook.WIRE_MODES:
                path = graphlook.wire_path(source, end, mode, end_dir=graphlook.RIGHT)
                heading = graphlook.end_heading(path, end)
                self.assertLess(heading.x(), -0.97, (source, mode))  # travelling left as it arrives
                self.assertLess(abs(heading.y()), 0.2, (source, mode))
                if mode == graphlook.RIGHT_ANGLE:
                    last = [s for s in line_segments(path)][-1]
                    self.assertAlmostEqual(last[0].y(), last[1].y(), places=3)
                    self.assertGreater(last[0].x(), last[1].x())     # entered from the right
                    self.assertAlmostEqual(last[1].x(), end.x(), places=3)

    def test_an_a_input_is_entered_from_the_left_and_a_main_input_from_above(self):
        path = graphlook.wire_path(P(400, 52), P(0, 126), graphlook.RIGHT_ANGLE, end_dir=graphlook.LEFT)
        self.assertGreater(graphlook.end_heading(path, P(0, 126)).x(), 0.97)
        path = graphlook.wire_path(P(95, 52), P(300, 200), graphlook.CURVED, end_dir=graphlook.UP)
        self.assertGreater(graphlook.end_heading(path, P(300, 200)).y(), 0.97)

    def test_a_wire_leaves_an_output_heading_down_in_both_modes(self):
        for mode in graphlook.WIRE_MODES:
            for end, end_dir in ((P(300, 200), graphlook.UP), (P(190, 126), graphlook.RIGHT), (P(95, 10), graphlook.UP)):
                path = graphlook.wire_path(self.OUT, end, mode, end_dir=end_dir)
                early = path.pointAtPercent(0.02)
                self.assertGreaterEqual(early.y(), self.OUT.y() - 0.01, (mode, end))
                self.assertLess(abs(early.x() - self.OUT.x()), 6, (mode, end))

    def test_right_angle_paths_have_only_horizontal_and_vertical_runs(self):
        ends = [(P(95, 120), graphlook.UP), (P(300, 200), graphlook.UP), (P(300, 90), graphlook.UP),
                (P(10, 400), graphlook.UP), (P(95, 10), graphlook.UP), (P(-200, 52), graphlook.UP),
                (P(190, 126), graphlook.RIGHT), (P(0, 126), graphlook.LEFT), (P(190, 20), graphlook.RIGHT)]
        for source in (self.OUT, P(400, 252), P(-80, -40)):
            for end, end_dir in ends:
                points = graphlook.right_angle_points(source, end, end_dir)
                for a, b in zip(points, points[1:]):
                    self.assertTrue(abs(a.x() - b.x()) < 0.01 or abs(a.y() - b.y()) < 0.01, (source, end, a, b))
                self.assertEqual((points[0], points[-1]), (source, end))
                for a, b in line_segments(graphlook.wire_path(source, end, graphlook.RIGHT_ANGLE, end_dir=end_dir)):
                    self.assertTrue(abs(a.x() - b.x()) < 0.01 or abs(a.y() - b.y()) < 0.01, (source, end, a, b))

    def test_corners_are_slightly_rounded_not_sharp(self):
        path = graphlook.wire_path(self.OUT, P(300, 200), graphlook.RIGHT_ANGLE)
        curves = [path.elementAt(i) for i in range(path.elementCount())
                  if path.elementAt(i).type == QPainterPath.ElementType.CurveToElement]
        self.assertEqual(len(curves), 2)  # two corners, each one curve (its control points follow as data elements)

    def test_on_a_long_drop_the_horizontal_run_sits_just_above_the_input(self):
        points = graphlook.right_angle_points(self.OUT, P(300, 400))
        self.assertAlmostEqual(points[1].y(), 400 - graphlook.RUN_ABOVE_INPUT)
        self.assertAlmostEqual(points[2].y(), 400 - graphlook.RUN_ABOVE_INPUT)
        short = graphlook.right_angle_points(self.OUT, P(300, 110))   # a short drop turns midway
        self.assertAlmostEqual(short[1].y(), (52 + 110) / 2)


class GraphLookTests(unittest.TestCase):
    def setUp(self):
        Preferences().set_wire_mode(graphlook.CURVED)
        self.window = Window()
        self.window.show()
        self.assertTrue(wait_until(lambda: self.window.frame is not None))
        self.graph = self.window.graph

    def tearDown(self):
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        Preferences().set_wire_mode(graphlook.CURVED)

    def settle(self):
        for _ in range(10):
            APP.processEvents()

    def command(self, *commands):
        self.window.command({"op": "batch", "commands": list(commands)}, render=False)
        self.settle()

    def edge(self, source, target, slot):
        return next(edge for edge, src, key, name, _out in self.graph.edges
                    if (src, key, name) == (source, target, slot))

    def painted(self, item):
        rect = item.boundingRect()
        image = QImage(int(rect.width()) + 2, int(rect.height()) + 2, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        painter.translate(-rect.left() + 1, -rect.top() + 1)
        item.paint(painter, QStyleOptionGraphicsItem())
        painter.end()
        return lambda x, y: image.pixelColor(int(x - rect.left() + 1), int(y - rect.top() + 1))

    # ---- nodes
    def test_every_node_item_carries_its_family_colour(self):
        for key, item in self.graph.items_by_id.items():
            kind = self.window.graph_nodes()[key]["type"]
            if kind == "Viewer":
                self.assertEqual(item.accent.name(), theme.TOKENS["acc"])
            else:
                self.assertEqual(item.accent.name(), theme.family_color(kind), kind)
            self.assertEqual(item.family, nodecatalog.node_family(kind))

    def test_a_node_is_outlined_in_its_family_colour_and_tinted_inside(self):
        blur_key = "blurnode"
        self.command({"op": "create", "id": blur_key, "type": "Blur", "pos": [900, 900]})
        grade, blur = self.graph.items_by_id["grade"], self.graph.items_by_id[blur_key]   # a command rebuilds the items
        for item, name in ((grade, "Color"), (blur, "Filter")):
            wanted = QColor(theme.FAMILY_COLORS[name])
            outline = self.painted(item)(item.rect().width() / 2, 1)          # the 2 px outline along the top
            hue_gap = abs(outline.hueF() - wanted.hueF())
            self.assertLess(min(hue_gap, 1 - hue_gap), 0.04, name)
            self.assertGreater(outline.saturationF(), 0.4, name)
            inside = self.painted(item)(item.rect().width() / 2, 20)
            self.assertLess(inside.lightnessF(), 0.2, name)                   # a dark interior
        self.assertGreater(self.painted(grade)(1, 26).greenF(), self.painted(grade)(1, 26).redF())
        self.assertGreater(self.painted(blur)(1, 26).redF(), self.painted(blur)(1, 26).blueF())

    def test_a_selected_node_has_a_brighter_halo_than_an_unselected_one(self):
        item = self.graph.items_by_id["grade"]
        halo = item.halo

        def glow(selected):
            item.setSelected(selected)
            image = QImage(300, 140, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(0)
            painter = QPainter(image)
            painter.translate(55, 44)
            halo.paint(painter, QStyleOptionGraphicsItem())
            painter.end()
            return image.pixelColor(55 + 95, 44 - 4).alpha()    # 4 px above the top edge

        self.assertGreater(glow(True), glow(False) + 25)
        self.assertGreater(glow(False), 0)

    def test_the_halo_does_not_enlarge_a_node_for_placement_and_framing(self):
        item = self.graph.items_by_id["grade"]
        self.assertEqual(item.sceneBoundingRect().width(), 190 + 2 * 0.0 + item.pen().widthF())
        self.assertGreater(item.halo.boundingRect().width(), item.rect().width() + 40)
        self.assertFalse(item.halo.shape().contains(QPointF(-5, -5)))

    def test_the_details_line_shows_changed_knobs_and_otherwise_the_first_ones(self):
        from nodebased.app import NodeItem  # noqa: F401
        node = {"params": {"exposure": 0.35, "multiply": 1.0, "offset": 0.0, "mix": 1.0}}
        defaults = SPECS["Grade"]["params"]
        self.assertEqual(graphlook.param_summary(node, defaults), "exposure 0.35  ·  multiply 1")
        self.assertEqual(graphlook.param_summary({"params": {"radius": 8.0, "mix": 1.0}}, SPECS["Blur"]["params"]),
                         "radius 8  ·  mix 1")

    # ---- ports
    def test_ports_face_the_way_wires_leave_and_enter(self):
        grade, merge = self.graph.items_by_id["grade"], self.graph.items_by_id["merge"]
        self.assertEqual(grade.output.facing, graphlook.DOWN)
        self.assertEqual(grade.inputs["image"].facing, graphlook.UP)
        self.assertEqual(grade.inputs["mask"].facing, graphlook.RIGHT)
        self.assertEqual(grade.inputs["mask"].pos().x(), grade.rect().width())
        self.assertEqual(merge.inputs["A"].facing, graphlook.LEFT)
        self.assertEqual(merge.inputs["B"].facing, graphlook.UP)

    def test_every_node_that_accepts_a_mask_has_its_mask_socket_on_the_right(self):
        kinds = [kind for kind, spec in SPECS.items() if "mask" in spec.get("optional_inputs", [])]
        self.assertGreater(len(kinds), 50)
        self.command(*({"op": "create", "id": f"m_{kind}", "type": kind, "pos": [4000 + 250 * n, 3000]}
                       for n, kind in enumerate(kinds)))
        for kind in kinds:
            item = self.graph.items_by_id[f"m_{kind}"]
            port = item.inputs["mask"]
            self.assertEqual(port.facing, graphlook.RIGHT, kind)
            self.assertAlmostEqual(port.pos().x(), item.rect().width(), msg=kind)

    # ---- wires
    def test_a_wire_wears_the_colour_of_its_source_family(self):
        self.command({"op": "connect", "id": "grade", "input": "mask", "source": "wash"})
        main = self.edge("plate", "grade", "image")
        mask = self.edge("wash", "grade", "mask")
        source_kind = self.window.graph_nodes()["plate"]["type"]
        self.assertEqual(main.color.name(), theme.family_color(source_kind))
        self.assertEqual(mask.color.name(), theme.family_color(self.window.graph_nodes()["wash"]["type"]))
        self.assertTrue(main.glow)
        self.assertEqual(mask.pen().style(), Qt.PenStyle.CustomDashLine)   # dashed, like the mockup
        self.assertEqual(main.pen().style(), Qt.PenStyle.SolidLine)

    def test_a_mask_wire_in_the_graph_enters_its_socket_horizontally_from_the_right(self):
        self.command({"op": "connect", "id": "grade", "input": "mask", "source": "wash"})
        port = self.graph.items_by_id["grade"].inputs["mask"]
        for mode in graphlook.WIRE_MODES:
            self.graph.set_wire_mode(mode, save=False)
            edge = self.edge("wash", "grade", "mask")
            self.assertEqual(edge.path().currentPosition(), port.scenePos(), mode)
            heading = graphlook.end_heading(edge.path(), port.scenePos())
            self.assertLess(heading.x(), -0.97, mode)
            self.assertLess(abs(heading.y()), 0.2, mode)

    def test_every_wire_in_right_angle_mode_is_horizontal_and_vertical_runs_only(self):
        self.command({"op": "connect", "id": "grade", "input": "mask", "source": "wash"})
        self.graph.set_wire_mode(graphlook.RIGHT_ANGLE, save=False)
        self.assertGreaterEqual(len(self.graph.edges), 4)
        for edge, *_ in self.graph.edges:
            for a, b in line_segments(edge.path()):
                self.assertTrue(abs(a.x() - b.x()) < 0.01 or abs(a.y() - b.y()) < 0.01, (a, b))

    # ---- the toggle
    def test_the_toggle_sits_beside_the_zoom_control_and_switches_the_wires(self):
        corner = self.graph.corner
        self.assertTrue(corner.toggle.isVisible() and corner.zoom.isVisible() and corner.minimap.isVisible())
        self.assertLess(corner.toggle.geometry().right(), corner.zoom.geometry().left())
        self.assertLess(corner.zoom.geometry().right(), corner.minimap.geometry().left())
        self.assertEqual(self.graph.wire_mode, graphlook.CURVED)
        self.assertTrue(corner.toggle.buttons[graphlook.CURVED].isChecked())
        corner.toggle.buttons[graphlook.RIGHT_ANGLE].click()
        self.settle()
        self.assertEqual(self.graph.wire_mode, graphlook.RIGHT_ANGLE)
        self.assertTrue(all(edge.mode == graphlook.RIGHT_ANGLE for edge, *_ in self.graph.edges))
        self.assertFalse(corner.toggle.buttons[graphlook.CURVED].isChecked())
        corner.toggle.buttons[graphlook.CURVED].click()
        self.assertEqual(self.graph.wire_mode, graphlook.CURVED)

    def test_framing_keeps_every_node_above_the_corner_controls(self):
        self.graph.resize(700, 260)
        self.settle()
        self.graph.frame_nodes()
        self.settle()
        top = self.graph.mapToScene(0, 0).y()
        row_top = self.graph.mapToScene(0, self.graph.viewport().height() - self.graph.corner.band_height()).y()
        for item in self.graph.items_by_id.values():
            rect = item.sceneBoundingRect()
            self.assertGreaterEqual(rect.top(), top)
            self.assertLessEqual(rect.bottom(), row_top)
        # The minimap is not part of the reserved band, so a short panel does not shrink the graph to a speck.
        self.assertLess(self.graph.corner.band_height(),
                        self.graph.corner.minimap.height() + 12)

    def test_the_wire_mode_persists_across_a_restart(self):
        self.graph.corner.toggle.buttons[graphlook.RIGHT_ANGLE].click()
        self.settle()
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        APP.processEvents()
        again = Window()
        again.show()
        try:
            self.assertEqual(again.graph.wire_mode, graphlook.RIGHT_ANGLE)
            self.assertTrue(again.graph.corner.toggle.buttons[graphlook.RIGHT_ANGLE].isChecked())
            self.assertTrue(all(edge.mode == graphlook.RIGHT_ANGLE for edge, *_ in again.graph.edges))
        finally:
            again.saved_document = again.dispatcher.document
            again.close()
            self.window = Window()          # tearDown closes whatever self.window is
            self.window.show()

    def test_an_unknown_saved_wire_mode_falls_back_to_curved(self):
        preferences = Preferences()
        preferences._store.setValue(Preferences.WIRE_MODE, "zigzag")
        self.assertEqual(preferences.wire_mode(), graphlook.CURVED)
        preferences.set_wire_mode("zigzag")           # ignored
        self.assertEqual(preferences.wire_mode(), graphlook.CURVED)

    # ---- hit testing and dragging
    def test_the_noodle_midpoint_handle_and_port_snapping_work_in_both_modes(self):
        for mode in graphlook.WIRE_MODES:
            self.graph.set_wire_mode(mode, save=False)
            edge, source, key, slot, _ = next(e for e in self.graph.edges if e[2] == "merge" and e[3] == "B")
            hit = self.graph.edge_handle_at(edge.handle)
            self.assertEqual(hit[1:], (source, key, slot), mode)
            port = self.graph.items_by_id["merge"].inputs["B"]
            self.assertIs(self.graph.input_at(port.scenePos() + P(3, -4)), port, mode)
            output = self.graph.items_by_id["grade"].output
            self.assertIs(self.graph.output_at(output.scenePos() + P(-3, 4)), output, mode)

    def test_dragging_a_wire_from_an_output_to_a_mask_socket_connects_it(self):
        for mode in graphlook.WIRE_MODES:
            self.graph.set_wire_mode(mode, save=False)
            self.command({"op": "connect", "id": "grade", "input": "mask", "source": None})
            output = self.graph.items_by_id["wash"].output
            target = self.graph.items_by_id["grade"].inputs["mask"]
            self.graph.start_wire("wash")
            self.graph.update_pending_edge(target.scenePos() + P(2, 1))
            pending = self.graph.pending_edge
            self.assertEqual(pending.mode, mode)
            self.assertEqual(pending.path().elementAt(0).x, output.scenePos().x())
            self.graph.finish_wire_at(target.scenePos() + P(2, 1))
            self.assertTrue(wait_until(lambda: self.window.graph_nodes()["grade"]["inputs"]["mask"] == "wash"), mode)
            self.settle()
            self.assertTrue(any(e[2:4] == ("grade", "mask") for e in self.graph.edges), mode)

    def test_dragging_a_node_moves_the_wire_ends_with_it(self):
        for mode in graphlook.WIRE_MODES:
            self.graph.set_wire_mode(mode, save=False)
            item = self.graph.items_by_id["merge"]
            item.setPos(item.pos() + P(120, 80))
            self.settle()
            edge = self.edge("grade", "merge", "B")
            self.assertEqual(edge.path().currentPosition(), item.inputs["B"].scenePos(), mode)

    # ---- the minimap and the zoom control
    def test_the_minimap_draws_one_rectangle_per_node_in_its_family_colour(self):
        mini = self.graph.corner.minimap
        rects = mini.node_rects()
        self.assertEqual(len(rects), len(self.graph.items_by_id))
        self.window.resize(1500, 900)
        self.settle()
        self.graph.frame_nodes()
        self.settle()
        image = mini.grab().toImage()
        scale, offset = mini.geometry_map()
        checked = 0
        for item in self.graph.items_by_id.values():
            centre = item.sceneBoundingRect().center()
            x, y = int(centre.x() * scale + offset.x()), int(centre.y() * scale + offset.y())
            if item.isSelected() or not (0 <= x < image.width() and 0 <= y < image.height()):
                continue
            seen, wanted = image.pixelColor(x, y), item.accent
            if any(abs(a - b) > 24 for a, b in zip(seen.getRgb()[:3], wanted.getRgb()[:3])):
                continue   # another node's rectangle or the view outline covers this point
            checked += 1
        self.assertGreaterEqual(checked, 3)

    def test_the_zoom_control_reads_and_sets_the_graph_zoom(self):
        zoom = self.graph.corner.zoom
        self.graph.zoom_to(1.0)
        self.settle()
        self.assertEqual(zoom.reset.text(), "100%")
        zoom.into.click()
        self.settle()
        self.assertAlmostEqual(self.graph.transform().m11(), 1.25, places=3)
        self.assertEqual(zoom.reset.text(), "125%")
        zoom.out.click()
        zoom.reset.click()
        self.settle()
        self.assertAlmostEqual(self.graph.transform().m11(), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
