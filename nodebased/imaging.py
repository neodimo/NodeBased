"""CPU reference evaluator. Internal RGBA is float32, scene-linear, premultiplied."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from PySide6.QtGui import QImage, QImageReader


class Cancelled(Exception):
    pass


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((np.maximum(rgb, 0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)


def read_image(path):
    if not path:
        raise ValueError("Choose a PNG or JPEG file in Read properties")
    if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("M0 supports PNG/JPEG only; EXR and OCIO are on the production-2D roadmap")
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.width() > 8192 or size.height() > 8192:
        raise ValueError("M0 full-frame decode is limited to 8192 × 8192")
    image = reader.read()
    if image.isNull():
        raise ValueError(f"Unable to read image: {reader.errorString()}")
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    rgba = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width() * 4].reshape(image.height(), image.width(), 4).astype(np.float32) / 255
    rgba[..., :3] = srgb_to_linear(rgba[..., :3]) * rgba[..., 3:4]
    return rgba


def to_qimage(frame, exposure=0.0, channel="RGB", checker=True):
    alpha = frame[..., 3:4]
    if channel == "A":
        rgb = np.repeat(alpha, 3, axis=2)
    else:
        rgb = frame[..., :3] * (2.0 ** exposure)
        if channel in ("R", "G", "B"):
            rgb = np.repeat(rgb[..., "RGB".index(channel):"RGB".index(channel) + 1], 3, axis=2)
        if checker:
            yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
            bg = np.where((xx // 16 + yy // 16) % 2 == 0, 0.055, 0.095).astype(np.float32)
            rgb = rgb + bg[..., None] * (1 - alpha)
        rgb = linear_to_srgb(rgb)
    rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    return QImage(rgb8.data, rgb8.shape[1], rgb8.shape[0], rgb8.strides[0], QImage.Format.Format_RGB888).copy()


def write_png(path, frame):
    if Path(path).suffix.lower() != ".png":
        raise ValueError("Export path must end in .png")
    alpha = np.clip(frame[..., 3:4], 0, 1)
    straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
    rgba = np.concatenate((np.clip(linear_to_srgb(straight), 0, 1), alpha), axis=2)
    rgba8 = (rgba * 255 + 0.5).astype(np.uint8)
    image = QImage(rgba8.data, rgba8.shape[1], rgba8.shape[0], rgba8.strides[0], QImage.Format.Format_RGBA8888).copy()
    # QSaveFile preserves an existing export on write failure.
    from PySide6.QtCore import QSaveFile, QIODevice
    target = QSaveFile(str(path))
    if not target.open(QIODevice.OpenModeFlag.WriteOnly):
        raise ValueError(target.errorString())
    if not image.save(target, "PNG") or not target.commit():
        raise ValueError(f"Cannot export PNG: {target.errorString()}")


class Evaluator:
    def __init__(self, cache_bytes=256 * 1024 * 1024):
        self.budget = cache_bytes
        self.cache = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def clear(self):
        self.cache.clear()
        self.bytes = 0

    def evaluate(self, doc, target=None, cancel: threading.Event | None = None):
        target = target or doc["view"]
        if target is None:
            raise ValueError("Select a node and press 1 to view it")
        nodes = doc["nodes"]
        # Iterative postorder traversal: execute only ancestors of the viewer.
        order, seen = [], set()
        stack = [(target, False)]
        while stack:
            key, visited = stack.pop()
            if key in seen:
                continue
            if visited:
                seen.add(key)
                order.append(key)
                continue
            stack.append((key, True))
            inputs = list(nodes[key]["inputs"].values())
            if nodes[key]["disabled"]:
                inputs = inputs[:1]
            stack.extend((source, False) for source in inputs if source is not None)
        values, hashes = {}, {}
        for key in order:
            if cancel and cancel.is_set():
                raise Cancelled()
            node = nodes[key]
            kind, params = node["type"], node["params"]
            sources = list(node["inputs"].values())
            if node["disabled"]:
                sources = sources[:1]
            if any(source is None for source in sources):
                raise ValueError(f"{node['name']}: connect required input(s)")
            fingerprint = None
            if kind == "Read" and params["path"]:
                stat = Path(params["path"]).stat()
                fingerprint = [str(Path(params["path"]).resolve()), stat.st_size, stat.st_mtime_ns]
            digest = hashlib.sha256(json.dumps([kind, params, node["disabled"], [hashes[s] for s in sources], fingerprint], sort_keys=True).encode()).hexdigest()
            hashes[key] = digest
            if digest in self.cache:
                self.hits += 1
                frame = self.cache.pop(digest)
                self.cache[digest] = frame
            else:
                self.misses += 1
                images = [values[s] for s in sources]
                frame = images[0] if node["disabled"] else self._kernel(kind, params, images)
                frame = np.asarray(frame, dtype=np.float32)
                frame.flags.writeable = False
                if frame.nbytes <= self.budget:
                    while self.cache and self.bytes + frame.nbytes > self.budget:
                        _, old = self.cache.popitem(last=False)
                        self.bytes -= old.nbytes
                    self.cache[digest] = frame
                    self.bytes += frame.nbytes
            values[key] = frame
        return values[target]

    @staticmethod
    def _kernel(kind, p, inputs):
        if kind == "Read":
            return read_image(p["path"])
        if kind == "Constant":
            color = np.array([p["red"] * p["alpha"], p["green"] * p["alpha"], p["blue"] * p["alpha"], p["alpha"]], dtype=np.float32)
            return np.broadcast_to(color, (p["height"], p["width"], 4)).copy()
        if kind == "Checker":
            yy, xx = np.ogrid[:p["height"], :p["width"]]
            pattern = ((xx // p["size"] + yy // p["size"]) % 2).astype(np.float32)
            frame = np.ones((p["height"], p["width"], 4), np.float32)
            frame[..., :3] = (0.06 + pattern * 0.24)[..., None]
            return frame
        if kind == "Viewer":
            return inputs[0]
        if kind == "Grade":
            frame = inputs[0].copy()
            frame[..., :3] = frame[..., :3] * (2.0 ** p["exposure"]) * p["multiply"] + p["offset"] * frame[..., 3:4]
            return frame
        if kind == "Transform":
            # Integer translation on fixed format; no wrapping at image boundaries.
            frame = np.zeros_like(inputs[0])
            h, w = frame.shape[:2]
            x, y = p["x"], p["y"]
            left, right, top, bottom = max(x, 0), min(w, w + x), max(y, 0), min(h, h + y)
            if left < right and top < bottom:
                frame[top:bottom, left:right] = inputs[0][top-y:bottom-y, left-x:right-x]
            return frame
        if kind == "Merge":
            a, b = inputs
            if a.shape != b.shape:
                raise ValueError("Merge inputs must have matching formats in M0")
            a = a * p["mix"]
            return a + b * (1 - a[..., 3:4])
        raise ValueError(f"No kernel for {kind}")
