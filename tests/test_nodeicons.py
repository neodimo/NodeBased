"""New look, step 4: the icon set of the left column and the node panel (Lane 2, NL4)."""
import tests.isolation  # noqa: F401  (settings of the test run, not the developer's)

import re
import unittest

from PySide6.QtGui import QColor
from PySide6.QtSvg import QSvgRenderer

from nodebased import nodecatalog, nodeicons, theme
from nodebased.core import SPECS
from tests.test_desktop import APP  # noqa: F401  (one QApplication for the run)


class FamilyIconTests(unittest.TestCase):
    def test_every_family_has_an_icon_and_a_colour(self):
        for family in nodecatalog.FAMILY_NAMES:
            self.assertIn(family, theme.FAMILY_COLORS, family)
            self.assertRegex(theme.FAMILY_COLORS[family], r"^#[0-9a-f]{6}$")
            self.assertIn(family, nodeicons.FAMILY_ICON_NAMES, family)
            self.assertTrue(nodeicons.renderer(nodeicons.family_icon_name(family), "#ffffff").isValid(), family)

    def test_other_and_the_column_buttons_have_icons_too(self):
        self.assertIn("Other", nodeicons.FAMILY_ICON_NAMES)
        for name in nodeicons.UI_ICONS:
            self.assertTrue(nodeicons.renderer("ui-" + name, "#ffffff").isValid(), name)

    def test_the_thirteen_families_are_the_thirteen_colours(self):
        self.assertEqual(set(nodecatalog.FAMILY_NAMES), set(theme.FAMILY_COLORS))
        self.assertEqual(len(nodeicons.FAMILY_ICON_NAMES), len(theme.FAMILY_COLORS) + 1)

    def test_no_two_families_share_a_glyph(self):
        bodies = [nodeicons._svg_text(nodeicons.family_icon_name(f)) for f in nodeicons.FAMILY_ICON_NAMES]
        self.assertEqual(len(set(bodies)), len(bodies))


class NodeIconTests(unittest.TestCase):
    def test_every_icon_file_is_a_valid_24_grid_svg_in_current_colour(self):
        files = sorted(nodeicons.ICON_DIR.glob("*.svg"))
        self.assertGreater(len(files), 100)
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertIn('viewBox="0 0 24 24"', text, path.name)
            self.assertIn('stroke="currentColor"', text, path.name)
            self.assertEqual(re.findall(r"#[0-9a-fA-F]{3,6}\b", text), [], f"{path.name} bakes in a colour")
            self.assertTrue(QSvgRenderer(text.encode("utf-8")).isValid(), path.name)

    def test_a_node_icon_file_names_a_registered_node_type(self):
        for kind in nodeicons.NODE_ICON_KINDS:
            self.assertIn(kind, SPECS, kind)

    def test_every_family_has_common_nodes_with_a_glyph_of_their_own(self):
        for family in nodecatalog.FAMILY_NAMES:
            own = [k for k in nodecatalog.NODE_CATEGORIES[family] if nodeicons.has_own_icon(k)]
            self.assertGreaterEqual(len(own), 4, family)

    def test_a_node_without_its_own_glyph_wears_its_family_glyph(self):
        kind = next(k for k in nodecatalog.NODE_CATEGORIES["Filter"] if not nodeicons.has_own_icon(k))
        self.assertEqual(nodeicons.node_icon_name(kind), "family-Filter")
        self.assertEqual(nodeicons.node_icon_name("Grade"), "node-Grade")
        self.assertEqual(nodeicons.node_icon_name("NotARealNode"), "family-Metadata")

    def test_an_icon_is_drawn_in_the_colour_asked_for(self):
        pixmap = nodeicons.pixmap("node-Grade", theme.FAMILY_COLORS["Color"], 32, 1.0)
        image = pixmap.toImage()
        want = QColor(theme.FAMILY_COLORS["Color"])
        solid = [image.pixelColor(x, y) for x in range(32) for y in range(32) if image.pixelColor(x, y).alpha() == 255]
        self.assertTrue(solid)
        self.assertTrue(all(abs(c.red() - want.red()) < 3 and abs(c.green() - want.green()) < 3 and
                            abs(c.blue() - want.blue()) < 3 for c in solid))

    def test_an_icon_renders_at_twice_the_pixels_on_a_high_dpi_screen(self):
        one, two = nodeicons.pixmap("family-Color", "#ffffff", 20, 1.0), nodeicons.pixmap("family-Color", "#ffffff", 20, 2.0)
        self.assertEqual((one.width(), two.width()), (20, 40))
        self.assertEqual(two.devicePixelRatio(), 2.0)


if __name__ == "__main__":
    unittest.main()
