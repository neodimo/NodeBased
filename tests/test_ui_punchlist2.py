"""New look, punch list 2 (DiMo's review of 10/11): a black viewer surround, an empty Properties
panel with nothing selected, the full transport, a calmer accent and one quiet playback unit."""
import unittest

from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)
from nodebased import theme, viewport3d
from tests.test_new_look_viewer_strip_nl5 import Base

APP = QApplication.instance() or QApplication([])


def settle(n=6):
    for _ in range(n):
        APP.processEvents()


class ViewerSurroundTests(Base):
    def test_surround_is_pure_black_in_every_theme(self):
        w = self.open_window((1440, 920))
        for name in theme.THEMES:
            w.setStyleSheet(theme.build_style(name))
            settle()
            self.assertEqual(w.viewer.backgroundBrush().color().name(), "#000000", name)
        self.assertEqual(theme.VIEWER_SURROUND, "#000000")
        self.assertEqual(tuple(viewport3d.BACKGROUND[:3]), (0.0, 0.0, 0.0))

    def test_surround_pixels_are_black(self):
        w = self.open_window((1440, 920))
        settle()
        image = w.viewer.viewport().grab().toImage()
        self.assertEqual(image.pixelColor(2, image.height() // 2).name(), "#000000")


if __name__ == "__main__":
    unittest.main()
