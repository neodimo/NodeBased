"""The icons of the left column and the floating node panel (New look, step 4, Lane 2 NL4).

The glyphs are SVG files under nodebased/data/icons, drawn on the mockup's 24 x 24 grid with a
1.6 stroke in `currentColor`: `family-<Family>.svg` for the thirteen families plus Other,
`ui-<name>.svg` for the column's own buttons and `node-<Kind>.svg` for the common nodes. A node
without a file of its own wears its family's glyph. Colours are never baked into a file: the SVG
text is rendered with `currentColor` replaced by the colour the caller asks for, so the same file
serves the column (muted, bright when active) and the panel (the family colour).
"""
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .nodecatalog import node_family

ICON_DIR = Path(__file__).resolve().parent / "data" / "icons"

UI_ICONS = ("favourites", "recent", "search", "settings")


def _names(prefix):
    return tuple(sorted(path.stem[len(prefix):] for path in ICON_DIR.glob(prefix + "*.svg")))


FAMILY_ICON_NAMES = _names("family-")     # the thirteen families and "Other"
NODE_ICON_KINDS = frozenset(_names("node-"))


def family_icon_name(family):
    return "family-" + family


def node_icon_name(kind):
    """The icon file of a node kind: its own when it has one, otherwise its family's glyph."""
    if kind in NODE_ICON_KINDS:
        return "node-" + kind
    return family_icon_name(node_family(kind))


def has_own_icon(kind):
    return kind in NODE_ICON_KINDS


@lru_cache(maxsize=None)
def _svg_text(name):
    return (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def renderer(name, color):
    """A cached renderer of the file `name` with its strokes and fills in `color` ("#rrggbb")."""
    return QSvgRenderer(QByteArray(_svg_text(name).replace("currentColor", color).encode("utf-8")))


def paint(painter, name, color, rect):
    """Draw the icon `name` in `color` into `rect`."""
    renderer(name, color).render(painter, QRectF(rect))


@lru_cache(maxsize=512)
def _pixmap(name, color, size, ratio):
    pixels = max(1, round(size * ratio))
    image = QImage(pixels, pixels, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer(name, color).render(painter, QRectF(0, 0, pixels, pixels))
    painter.end()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def pixmap(name, color, size, ratio=1.0):
    """The icon as a pixmap `size` logical pixels square, rendered at `ratio` times that."""
    return _pixmap(name, color, int(size), float(ratio))


def icon(name, color, size=24):
    """A QIcon at `size`, rendered for 1x and 2x screens."""
    result = QIcon()
    for ratio in (1.0, 2.0):
        result.addPixmap(pixmap(name, color, size, ratio))
    return result
