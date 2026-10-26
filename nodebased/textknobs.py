"""Nuke's Text knobs (fidelity step F): vertical justify, font style, scales, spacing, a transform, a colour ramp and drop shadows.

A Text node whose new knobs are all at the values an older node had draws through the original single `drawText` call and gives the
same pixels as before. Any other node draws through `render` here: the text is laid out with Qt's `QTextLayout` (word wrapped in the
box, justified, `leading` added between lines), painted through one transform, and the colour, ramp and shadow are applied to the
resulting coverage in NumPy. The layout is position independent (the painter works in canvas coordinates), so a tile and the whole
frame agree; the shadow is computed on a canvas padded by its reach for the same reason.

The readings of the guide's one-line knob descriptions (the guide gives no formulas for any of them, and NodeBased has compared none
with a Nuke render):

* `global_font_scale` multiplies `font_size`. `font_width` and `font_height` are percentages that stretch the glyphs and the layout
  from the box's top left corner; the text still wraps inside the box.
* `tracking` adds that many pixels after every character, spaces included. `kerning` adds that many pixels after every letter or
  digit but not after a space, so it opens a word up without moving the words apart. Both are in glyph pixels before `font_width`.
* `baseline_shift` raises the text by that many pixels (negative lowers it). `leading` adds that many pixels between lines.
* `index` is the font style: `Regular`, `Bold`, `Italic`, `Bold Italic` (the words bold, italic and oblique are recognised), or the
  exact style name of a face of the family.
* `translate`, `rotate`, `scale`, `skew_x`, `skew_y`, `skew_order` and `center` transform the whole text block. `center` is the pivot
  as an offset from the box's centre. The linear part is rotate, then skew, then scale, as in the Transform node, with the
  translation after.
* `ramp` colours the text from `color0` at `p0` to the text colour at `p1`: `linear`, `smooth` (smoothstep), `smooth0` (eases in,
  slow start) or `smooth1` (eases out, slow end); the text's own alpha still shapes it.
* Shadows: the text's alpha is moved by `shadow_distance` along `shadow_angle` (whole pixels, the DropShadow node's offset),
  grown or shrunk by `shadow_expand` pixels, blurred by `shadow_softness` (the Soften kernel), scaled by `shadow_opacity` and the shadow
  colour's alpha, and composited under the text. `shadow_inherit_input_color` takes the text colour for the shadow.
"""
from __future__ import annotations

import math

import numpy as np

NEW = {"yjustify": "top", "index": "Regular", "global_font_scale": 1.0, "font_width": 100.0, "font_height": 100.0,
       "kerning": 0.0, "tracking": 0.0, "baseline_shift": 0.0, "leading": 0.0,
       "translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0, "scale": 1.0, "skew_x": 0.0, "skew_y": 0.0, "skew_order": "XY",
       "center_x": 0.0, "center_y": 0.0,
       "ramp": "none", "p0_x": 100.0, "p0_y": 100.0, "p1_x": 100.0, "p1_y": 400.0,
       "color0_red": 0.0, "color0_green": 0.0, "color0_blue": 0.0, "color0_alpha": 0.0,
       "enable_shadows": 0, "shadow_inherit_input_color": 0, "shadow_red": 0.0, "shadow_green": 0.0, "shadow_blue": 0.0,
       "shadow_alpha": 1.0, "shadow_opacity": 0.7, "shadow_angle": 225.0, "shadow_distance": 5.0, "shadow_softness": 0.0,
       "shadow_expand": 0.0}
# An older Text centred its lines in the box; every other knob's default leaves the original drawing as it was.
OLD = dict(NEW, yjustify="center")
KINDS = ("Text",)
RAMPS = ("none", "linear", "smooth", "smooth0", "smooth1")
YJUSTIFY = ("top", "center", "bottom")
STYLES = ("Regular", "Bold", "Italic", "Bold Italic")
SKEW_ORDERS = ("XY", "YX")


def new_params(kind):
    return dict(NEW) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(OLD) if kind in KINDS else {}


def is_legacy(p):
    """True when every step F knob is at the value an older Text had, so the original drawing applies."""
    return all(p.get(name, default) == default for name, default in OLD.items())


# --------------------------------------------------------------------------------------------------- geometry
def _translate(x, y):
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def _scale(sx, sy):
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def block_matrix(p):
    """The 3 x 3 matrix of the Groups tab transform, in page coordinates, about `center` (an offset from the box's centre)."""
    cx = float(p["box_x"]) + float(p["box_width"]) / 2.0 + float(p.get("center_x", 0.0))
    cy = float(p["box_y"]) + float(p["box_height"]) / 2.0 + float(p.get("center_y", 0.0))
    angle = math.radians(float(p.get("rotate", 0.0)))
    c, s = math.cos(angle), math.sin(angle)
    kx, ky = float(p.get("skew_x", 0.0)), float(p.get("skew_y", 0.0))
    skew_x = np.array([[1.0, kx], [0.0, 1.0]])
    skew_y = np.array([[1.0, 0.0], [ky, 1.0]])
    skew = skew_y @ skew_x if p.get("skew_order", "XY") == "XY" else skew_x @ skew_y
    scale = float(p.get("scale", 1.0))
    linear = np.array([[c, -s], [s, c]]) @ skew @ np.diag([scale, scale])
    out = np.eye(3)
    out[:2, :2] = linear
    return _translate(cx + float(p.get("translate_x", 0.0)), cy + float(p.get("translate_y", 0.0))) @ out @ _translate(-cx, -cy)


def ramp_weight(kind, t):
    """The share of the text colour at fraction `t` (0 to 1) along the ramp."""
    t = np.clip(t, 0.0, 1.0)
    if kind == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if kind == "smooth0":
        return t * t
    if kind == "smooth1":
        return 1.0 - (1.0 - t) * (1.0 - t)
    return t


def shadow_reach(p):
    """Pixels the shadow can move text coverage out of the block it came from, each way."""
    if not p.get("enable_shadows", 0):
        return 0
    distance = abs(float(p.get("shadow_distance", 0.0)))
    soft = abs(float(p.get("shadow_softness", 0.0)))
    expand = abs(float(p.get("shadow_expand", 0.0)))
    return (int(math.ceil(distance)) + (int(math.ceil(soft)) if soft >= 0.5 else 0)
            + (int(math.ceil(expand)) if expand >= 0.5 else 0) + 1)


# --------------------------------------------------------------------------------------------------- drawing
def _font(p):
    from PySide6.QtGui import QFont, QFontDatabase

    family = p.get("font") or ""
    font = QFont(family) if family else QFont()
    font.setPointSizeF(max(1.0, float(p.get("font_size", 48.0)) * float(p.get("global_font_scale", 1.0))))
    style = str(p.get("index", "Regular") or "Regular")
    low = style.lower()
    if style != "Regular" and style in QFontDatabase.styles(font.family()):
        font.setStyleName(style)
    else:
        font.setBold("bold" in low)
        font.setItalic("italic" in low or "oblique" in low)
    tracking, kerning = float(p.get("tracking", 0.0)), float(p.get("kerning", 0.0))
    if tracking or kerning:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking + kerning)
        font.setWordSpacing(-kerning)               # kerning does not apply after a space
    return font


def draw_coverage(p, origin, w, h):
    """The text's coverage (0 to 1) on a w x h canvas whose top left is page pixel `origin`."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QTextLayout, QTextOption, QTransform
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication(["nodebased"])
    sx = max(1e-3, float(p.get("font_width", 100.0)) / 100.0)
    sy = max(1e-3, float(p.get("font_height", 100.0)) / 100.0)
    bx, by, bw, bh = (float(p["box_x"]), float(p["box_y"]), float(p["box_width"]), float(p["box_height"]))
    matrix = (_translate(-origin[0], -origin[1]) @ block_matrix(p) @ _translate(0.0, -float(p.get("baseline_shift", 0.0)))
              @ _translate(bx, by) @ _scale(sx, sy))
    image = QImage(w, h, QImage.Format.Format_RGBA8888)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setTransform(QTransform(matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1], matrix[0, 2], matrix[1, 2]))
    painter.setPen(QColor(255, 255, 255, 255))
    font = _font(p)
    painter.setFont(font)
    layout = QTextLayout(str(p.get("message", "")).replace("\n", " "), font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WordWrap)
    option.setAlignment({"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
                         "right": Qt.AlignmentFlag.AlignRight}.get(p.get("justify", "left"), Qt.AlignmentFlag.AlignLeft))
    layout.setTextOption(option)
    leading = float(p.get("leading", 0.0))
    layout.beginLayout()
    y = 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(bw / sx)
        line.setPosition(QPointF(0.0, y))
        y += line.height() + leading
    layout.endLayout()
    total = max(0.0, y - leading)
    free = bh / sy - total
    top = {"top": 0.0, "center": 0.5, "bottom": 1.0}.get(p.get("yjustify", "top"), 0.0) * free
    # Fill each glyph's outline rather than call layout.draw: Qt draws a glyph that crosses the canvas edge by a different route
    # from one that does not, which would make a tile differ from the frame it was cut from. An outline fill is the same anywhere.
    painter.setBrush(QColor(255, 255, 255, 255))
    painter.setPen(Qt.PenStyle.NoPen)
    for run in layout.glyphRuns():
        raw = run.rawFont()
        for index, position in zip(run.glyphIndexes(), run.positions()):
            outline = raw.pathForGlyph(index)
            outline.translate(position.x(), position.y() + top)
            painter.drawPath(outline)
    painter.end()
    stride = image.bytesPerLine()
    raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint8).reshape(h, stride)
    return raw[:, :w * 4].reshape(h, w, 4)[..., 3].astype(np.float32) / 255.0


def _text_colour(p, coverage, x0, y0):
    """The text's straight RGBA over the canvas: the flat colour, or the ramp from `color0` to it."""
    h, w = coverage.shape
    colour = np.clip(np.array([p.get("red", 1.0), p.get("green", 1.0), p.get("blue", 1.0), p.get("alpha", 1.0)], dtype=np.float32), 0.0, 1.0)
    kind = str(p.get("ramp", "none"))
    if kind == "none":
        return np.broadcast_to(colour, (h, w, 4))
    start = np.clip(np.array([p.get("color0_red", 0.0), p.get("color0_green", 0.0), p.get("color0_blue", 0.0),
                              p.get("color0_alpha", 0.0)], dtype=np.float32), 0.0, 1.0)
    xs = np.arange(x0, x0 + w, dtype=np.float64)[None, :]
    ys = np.arange(y0, y0 + h, dtype=np.float64)[:, None]
    p0x, p0y, p1x, p1y = (float(p.get("p0_x", 0.0)), float(p.get("p0_y", 0.0)), float(p.get("p1_x", 0.0)), float(p.get("p1_y", 0.0)))
    dx, dy = p1x - p0x, p1y - p0y
    denom = dx * dx + dy * dy
    t = np.zeros((h, w)) if denom < 1e-12 else ((xs - p0x) * dx + (ys - p0y) * dy) / denom
    weight = ramp_weight(kind, t).astype(np.float32)[..., None]
    return start[None, None, :] * (1.0 - weight) + colour[None, None, :] * weight


def render(p, x0, y0, w, h):
    """Premultiplied RGBA float32 of the Text node for the w x h window at page pixel (x0, y0)."""
    from .imaging import Evaluator

    reach = shadow_reach(p)
    pad = reach
    cw, ch = w + 2 * pad, h + 2 * pad
    ox, oy = x0 - pad, y0 - pad
    coverage = draw_coverage(p, (ox, oy), cw, ch)
    straight = _text_colour(p, coverage, ox, oy)
    alpha = coverage * straight[..., 3]
    out = np.concatenate([straight[..., :3] * alpha[..., None], alpha[..., None]], axis=-1).astype(np.float32)
    if reach:
        dx, dy = Evaluator._shadow_offset({"angle": p.get("shadow_angle", 225.0), "distance": p.get("shadow_distance", 0.0)})
        moved = np.zeros((ch, cw, 1), dtype=np.float32)
        src_x, src_y = slice(max(0, -dx), min(cw, cw - dx)), slice(max(0, -dy), min(ch, ch - dy))
        dst_x, dst_y = slice(max(0, dx), min(cw, cw + dx)), slice(max(0, dy), min(ch, ch + dy))
        if dst_x.stop > dst_x.start and dst_y.stop > dst_y.start:
            moved[dst_y, dst_x, 0] = alpha[src_y, src_x]
        grow = float(p.get("shadow_expand", 0.0))
        if abs(grow) >= 0.5:
            moved = Evaluator._box_extreme(moved, abs(grow), grow > 0)
        soft = abs(float(p.get("shadow_softness", 0.0)))
        if soft >= 0.5:
            radius, sigma = int(math.ceil(soft)), soft / 3.0
            moved = Evaluator._gaussian_axis(Evaluator._gaussian_axis(moved, radius, sigma, axis=1, mode="constant"),
                                             radius, sigma, axis=0, mode="constant")
        if p.get("shadow_inherit_input_color", 0):
            tint = np.clip(np.array([p.get("red", 1.0), p.get("green", 1.0), p.get("blue", 1.0)], dtype=np.float32), 0.0, 1.0)
            opacity = np.float32(p.get("shadow_opacity", 0.7))
        else:
            tint = np.clip(np.array([p.get("shadow_red", 0.0), p.get("shadow_green", 0.0), p.get("shadow_blue", 0.0)], dtype=np.float32), 0.0, 1.0)
            opacity = np.float32(p.get("shadow_opacity", 0.7)) * np.float32(np.clip(p.get("shadow_alpha", 1.0), 0.0, 1.0))
        shadow = moved * opacity
        under = np.concatenate([shadow * tint[None, None, :], shadow], axis=-1)
        out = (out + under * (1.0 - out[..., 3:4])).astype(np.float32)
    return np.ascontiguousarray(out[pad:pad + h, pad:pad + w])
