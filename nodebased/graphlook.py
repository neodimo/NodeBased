"""The look of the node graph (new look, step 2): wire geometry, family icons and node details.

The shapes here are plain Qt geometry with no scene or window behind them, so tests can ask for the
path of a wire and read its segments. `app.py` draws with them.

Wire rules (UI-SPEC.md): a wire leaves and enters along the direction its port faces. The main output
faces down and the main input faces up; the A input faces left and the mask input faces right. In
Curved mode the wire is a cubic whose end control points sit along those directions. In Right angle
mode it is made of horizontal and vertical runs only, with slightly rounded 90 degree corners, and on
a long drop the horizontal run sits just above the input.
"""
import math

from PySide6.QtCore import QByteArray, QPointF, QRectF
from PySide6.QtGui import QColor, QPainterPath
from PySide6.QtSvg import QSvgRenderer

CURVED, RIGHT_ANGLE = "curved", "right_angle"
WIRE_MODES = (CURVED, RIGHT_ANGLE)
DOWN, UP, LEFT, RIGHT = QPointF(0, 1), QPointF(0, -1), QPointF(-1, 0), QPointF(1, 0)

BODY_DARK = "#0b0e12"   # the near-black a node's interior is tinted from
BODY_DARKER = "#080a0d"
CORNER_RADIUS = 6.0      # the slight rounding on a right angle turn
STUB = 18.0              # how far a wire runs straight out of / into a port before it turns
LONG_DROP = 80.0         # a drop longer than this puts the horizontal run just above the input
RUN_ABOVE_INPUT = 22.0   # ... this far above it
CURVE_MIN_REACH = 40.0   # the shortest end-control distance of a curved wire


def _length(vector):
    return math.hypot(vector.x(), vector.y())


def _snap_axis(direction):
    """The nearest of the four axis directions (a rim socket on a circle faces any way)."""
    if abs(direction.x()) > abs(direction.y()):
        return RIGHT if direction.x() > 0 else LEFT
    return DOWN if direction.y() > 0 else UP


def curved_path(start, end, start_dir=DOWN, end_dir=UP):
    """The cubic of mockup 1: it leaves `start` along `start_dir` and arrives at `end` from the side
    `end_dir` faces (so a wire into a right-facing mask socket comes in heading left)."""
    dx, dy = abs(end.x() - start.x()), abs(end.y() - start.y())
    reach_out = max(CURVE_MIN_REACH, 0.5 * dy) if start_dir.y() != 0 else max(CURVE_MIN_REACH, 0.5 * dx)
    reach_in = max(CURVE_MIN_REACH, 0.5 * dy) if end_dir.y() != 0 else max(CURVE_MIN_REACH, min(0.5 * dx, 140.0))
    path = QPainterPath(start)
    path.cubicTo(start + start_dir * reach_out, end + end_dir * reach_in, end)
    return path


def _dedupe(points):
    """Drop repeated points and the middle of any three points that continue in a straight line."""
    out = []
    for point in points:
        if out and abs(out[-1].x() - point.x()) < 0.01 and abs(out[-1].y() - point.y()) < 0.01:
            continue
        out.append(point)
    trimmed = [out[0]] if out else []
    for index in range(1, len(out) - 1):
        a, b, c = trimmed[-1], out[index], out[index + 1]
        straight_on = ((abs(a.x() - b.x()) < 0.01 and abs(b.x() - c.x()) < 0.01 and (b.y() - a.y()) * (c.y() - b.y()) > 0)
                       or (abs(a.y() - b.y()) < 0.01 and abs(b.y() - c.y()) < 0.01 and (b.x() - a.x()) * (c.x() - b.x()) > 0))
        if straight_on:
            continue
        trimmed.append(b)
    if len(out) > 1:
        trimmed.append(out[-1])
    return trimmed


def right_angle_points(start, end, end_dir=UP):
    """The corner points of a right angle wire from an output (which faces down) to `end`, whose
    port faces `end_dir`. Consecutive points differ in x or in y, never both."""
    end_dir = _snap_axis(end_dir)
    sx, sy, ex, ey = start.x(), start.y(), end.x(), end.y()
    stub = STUB
    if end_dir.y() != 0:                       # into the top (or bottom) of a node
        drop = (ey - sy) * (1 if end_dir.y() < 0 else -1)
        if abs(ex - sx) < 0.5 and drop >= 0:
            return _dedupe([start, end])
        if drop >= 2 * stub:
            run = ey + end_dir.y() * RUN_ABOVE_INPUT if drop > LONG_DROP else (sy + ey) / 2
            return _dedupe([start, QPointF(sx, run), QPointF(ex, run), end])
        # The target is level with or above the source: leave downwards, cross, then come in.
        below, above = sy + stub, ey + end_dir.y() * stub
        middle = (sx + ex) / 2 if abs(ex - sx) >= 2 * stub else sx + (3 * stub if ex >= sx else -3 * stub)
        return _dedupe([start, QPointF(sx, below), QPointF(middle, below), QPointF(middle, above),
                        QPointF(ex, above), end])
    # Into the side of a node (the mask on the right, the A input on the left): the last run is
    # horizontal, entering against the direction the port faces.
    channel = max(sx, ex + stub) if end_dir.x() > 0 else min(sx, ex - stub)
    if abs(channel - sx) < 0.5:
        if ey >= sy + stub:
            return _dedupe([start, QPointF(sx, ey), end])
        channel = sx + end_dir.x() * 2 * stub      # the target is above: swing out so the wire never retraces
    below = sy + stub
    return _dedupe([start, QPointF(sx, below), QPointF(channel, below), QPointF(channel, ey), end])


def rounded_polyline(points, radius=CORNER_RADIUS):
    """A path through `points` whose 90 degree corners are rounded with a small quadratic."""
    path = QPainterPath(points[0])
    for index in range(1, len(points) - 1):
        before, corner, after = points[index - 1], points[index], points[index + 1]
        into, out = _length(corner - before), _length(after - corner)
        r = min(radius, into / 2, out / 2)
        if r < 0.5:
            path.lineTo(corner)
            continue
        entry = corner + (before - corner) * (r / into)
        leave = corner + (after - corner) * (r / out)
        path.lineTo(entry)
        path.quadTo(corner, leave)
    path.lineTo(points[-1])
    return path


def wire_path(start, end, mode=CURVED, start_dir=DOWN, end_dir=UP):
    """The path of a wire from an output at `start` to an input at `end` (see the module docstring)."""
    if mode == RIGHT_ANGLE:
        return rounded_polyline(right_angle_points(start, end, end_dir))
    return curved_path(start, end, start_dir, end_dir)


def right_angle_segments(start, end, end_dir=UP):
    """The straight runs of a right angle wire as (from, to) point pairs (corner rounding left out)."""
    points = right_angle_points(start, end, end_dir)
    return list(zip(points, points[1:]))


def end_heading(path, end):
    """A unit vector for the direction a wire is travelling as it reaches `end`."""
    tail = path.pointAtPercent(0.985)
    vector = end - tail
    size = _length(vector)
    return QPointF(vector.x() / size, vector.y() / size) if size > 1e-6 else DOWN


# ---------------------------------------------------------------------------------------------
# Family icons: the glyphs of the mockup's icon rail, drawn in the family colour.
ICON_SHAPES = {
    "Image": '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/>',
    "Draw": '<path d="M4 20c2-6 5-11 14-16-2 6-6 12-14 16z"/>',
    "Color": '<circle cx="9" cy="10" r="5.5"/><circle cx="15" cy="10" r="5.5"/><circle cx="12" cy="15" r="5.5"/>',
    "Filter": '<path d="M3 5h18l-7 8v6l-4-2v-4z"/>',
    "Keyer": '<circle cx="8" cy="14" r="4"/><path d="m11 11 9-8M16 7l3 3"/>',
    "Merge": '<rect x="3" y="3" width="12" height="12" rx="2.5"/><rect x="9" y="9" width="12" height="12" rx="2.5"/>',
    "Transform": '<path d="M12 3v18M3 12h18"/>',
    "3D": '<path d="m12 2.8 8 4.6v9.2l-8 4.6-8-4.6V7.4z"/>',
    "Particles": '<circle cx="6" cy="17" r="1.6"/><circle cx="11" cy="11" r="1.6"/><circle cx="17" cy="6" r="1.6"/>',
    "Fluids": '<path d="M12 3c3.5 4.5 6 8 6 11a6 6 0 0 1-12 0c0-3 2.5-6.5 6-11z"/>',
    "Time": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/>',
    "Channel": '<path d="M4 6h16M4 12h16M4 18h16"/>',
    "Metadata": '<path d="M4 4h10l6 6v10H4z"/>',
}
_renderers = {}


def icon_renderer(family, color, stroke=2.0):
    """A cached SVG renderer for a family glyph stroked in `color` (a "#rrggbb" string)."""
    key = (family, color, stroke)
    if key not in _renderers:
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
               f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICON_SHAPES[family]}</svg>')
        _renderers[key] = QSvgRenderer(QByteArray(svg.encode()))
    return _renderers[key]


def draw_icon(painter, family, color, rect, glow=True):
    """Paint the glyph of `family` into `rect`, with the faint glow the mockup puts round it."""
    if glow:
        halo = QColor(color)
        halo.setAlphaF(0.30)
        painter.save()
        painter.setOpacity(painter.opacity() * 0.45)
        icon_renderer(family, halo.name(), 4.5).render(painter, QRectF(rect))
        painter.restore()
    icon_renderer(family, color).render(painter, QRectF(rect))


# ---------------------------------------------------------------------------------------------
# The details line under a node's name.
def format_value(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    return str(value)


def param_summary(node, defaults, limit=2):
    """"gain 1.18 · lift -0.02": up to `limit` plain knobs, those that differ from the node's defaults
    first, then the first of the rest, so a fresh node still shows what it is set to."""
    changed, rest = [], []
    for name, value in node.get("params", {}).items():
        if not isinstance(value, (bool, int, float, str)):
            continue
        text = format_value(value)
        if len(text) > 14 or "\n" in text or not text:
            continue
        (rest if value == defaults.get(name, value) else changed).append(f"{name.replace('_', ' ')} {text}")
    return "  ·  ".join((changed + rest)[:limit])


def mix(color_a, color_b, amount):
    """`color_a` with `amount` (0..1) of `color_b` mixed in; both are QColor."""
    return QColor.fromRgbF(color_a.redF() * (1 - amount) + color_b.redF() * amount,
                           color_a.greenF() * (1 - amount) + color_b.greenF() * amount,
                           color_a.blueF() * (1 - amount) + color_b.blueF() * amount)


def with_alpha(color, alpha):
    """A copy of QColor `color` with its alpha set to `alpha` (0..255)."""
    out = QColor(color)
    out.setAlpha(int(alpha))
    return out


def css_font(css_families, pixel_size, weight=None, mono=False):
    """A QFont from a CSS family list ("'Noto Sans', 'Segoe UI', system-ui") sized in pixels."""
    from PySide6.QtGui import QFont
    names = [part.strip().strip("'\"") for part in css_families.split(",")]
    names = [name for name in names if name and name not in ("system-ui", "sans-serif", "monospace")]
    font = QFont()
    font.setFamilies(names)
    font.setPixelSize(int(pixel_size))
    if mono:
        font.setStyleHint(QFont.StyleHint.Monospace)
    if weight is not None:
        font.setWeight(weight)
    return font
