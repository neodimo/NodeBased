"""Derive the taskbar icon from the icon master: the green N and its two ports, no tile.

The master (assets/marketing/nodebased-icon-master.png) is the folded-ribbon N on a dark rounded
tile. A taskbar is already a dark rounded surface, so the tile only shrank the mark; this lifts the
mark off the tile and scales it to fill the canvas.

    uv run python tools/make_icon.py

writes assets/nodebased-icon.png (1024 px) and assets/nodebased-icon.ico (16-256 px, PNG entries).

Why it is done by colour rather than by the master's alpha: the master's alpha is the tile, and
it also has the two white port rings knocked out to transparent, so the rings would vanish on
any light taskbar. The ribbon is the only saturated green on the tile and the rings are the only
near-white, so each gets its own soft matte from colour alone. Each ring keeps its dark centre and
a thin dark rim, so the white ports still read against a light taskbar.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "assets" / "marketing" / "nodebased-icon-master.png"
PNG_OUT = ROOT / "assets" / "nodebased-icon.png"
ICO_OUT = ROOT / "assets" / "nodebased-icon.ico"
SIZE = 1024
MARGIN = 0.03  # of the canvas, per side: enough that antialiased edges are not clipped
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
# The tile colour under the mark, used to pull tile contamination back out of antialiased edges.
TILE = np.array([26.0, 30.0, 33.0])
RING_RIM = 0.08  # dark rim around each port ring, as a fraction of the ring's radius


def load_rgba(path):
    image = QImage(str(path)).convertToFormat(QImage.Format.Format_RGBA8888)
    if image.isNull():
        raise SystemExit(f"cannot read {path}")
    bits = np.frombuffer(image.constBits(), np.uint8)
    return bits.reshape(image.height(), image.bytesPerLine() // 4, 4)[:, :image.width()].astype(np.float64)


def smoothstep(low, high, x):
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lift_mark(master):
    rgb = master[..., :3]
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Ribbon: green well above both other channels. The tile and the ribbon's own drop shadow are
    # neutral, so they fall out here however dark or light they are.
    chroma = green - np.maximum(red, blue)
    ribbon = smoothstep(6.0, 40.0, chroma)
    # Rings: bright and neutral. Their RGB survives in the master even where alpha was zeroed.
    brightness = rgb.min(axis=2)
    ring = smoothstep(90.0, 235.0, brightness) * (1.0 - smoothstep(20.0, 45.0, chroma))
    # The checker outside the tile is also light and neutral, so rings only count inside the
    # tile's outline: per row, between its first and last opaque pixel. That follows the rounded
    # corners and still covers the ring holes the master's alpha punched through the tile.
    tile = master[..., 3] > 128
    inside = np.zeros_like(tile)
    for row in np.nonzero(tile.any(axis=1))[0]:
        cols = np.nonzero(tile[row])[0]
        inside[row, cols[0]:cols[-1] + 1] = True
    ring = ring * inside
    # White on a light taskbar would erase the ports, so each ring keeps a thin rim of the tile
    # colour and its dark centre: the port reads as a port on any background. The rings are the
    # holes the master's alpha punched inside the tile; one sits either side of the centre line.
    holes = inside & ~tile & (ring > 0.5)
    rows, cols = np.indices(tile.shape)
    rim = np.zeros(tile.shape)
    for side in (cols < tile.shape[1] / 2, cols >= tile.shape[1] / 2):
        ys, xs = np.nonzero(holes & side)
        if len(ys) == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        outer = np.hypot(ys - cy, xs - cx).max()
        distance = np.hypot(rows - cy, cols - cx)
        width = outer * RING_RIM
        rim = np.maximum(rim, 1.0 - smoothstep(outer + width - 1.5, outer + width + 1.5, distance))
    alpha = np.maximum(np.maximum(ribbon, ring), rim)
    # Un-mix the tile from partially covered pixels so the edges do not carry a dark fringe.
    safe = np.maximum(alpha, 1e-6)[..., None]
    colour = np.clip((rgb - (1.0 - alpha[..., None]) * TILE) / safe, 0.0, 255.0)
    colour[alpha <= 0.0] = 0.0
    return np.dstack([colour, alpha * 255.0])


def fit_to_canvas(mark):
    # Frame on the solid mark. The mattes leave faint haze along the tile's top and bottom edges;
    # counting it would pad the frame by a sixth of the canvas and shrink the mark to match.
    opaque = mark[..., 3] > 64
    rows, cols = np.nonzero(opaque)
    pad = 4  # keep the antialiased edge just outside the solid mark
    top, left = max(rows.min() - pad, 0), max(cols.min() - pad, 0)
    bottom, right = rows.max() + pad + 1, cols.max() + pad + 1
    cropped = np.ascontiguousarray(mark[top:bottom, left:right].round().astype(np.uint8))
    height, width = cropped.shape[:2]
    image = QImage(cropped.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
    inner = round(SIZE * (1.0 - 2 * MARGIN))
    scaled = image.scaled(inner, inner, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
    canvas = QImage(SIZE, SIZE, QImage.Format.Format_RGBA8888)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.drawImage((SIZE - scaled.width()) // 2, (SIZE - scaled.height()) // 2, scaled)
    painter.end()
    return canvas


def png_bytes(image):
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data)


def write_ico(image, path):
    """Multi-resolution .ico with PNG-compressed entries (Windows Vista and later)."""
    entries = [png_bytes(image.scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
               for size in ICO_SIZES]
    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = len(header) + 16 * len(entries)
    directory = b""
    for size, data in zip(ICO_SIZES, entries):
        side = 0 if size >= 256 else size  # 0 means 256 in an ICONDIRENTRY
        directory += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.write_bytes(header + directory + b"".join(entries))


def main():
    QGuiApplication.instance() or QGuiApplication([])
    icon = fit_to_canvas(lift_mark(load_rgba(MASTER)))
    if not icon.save(str(PNG_OUT), "PNG"):
        raise SystemExit(f"cannot write {PNG_OUT}")
    write_ico(icon, ICO_OUT)
    print(f"wrote {PNG_OUT.relative_to(ROOT)} and {ICO_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
