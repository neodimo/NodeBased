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


def read_image(path, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0):
    from .media import read_media
    return read_media(path, colorspace, alpha_mode, layer, subimage)


def to_qimage(frame, exposure=0.0, channel="RGB", checker=True, view="sRGB"):
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
        from .color import display_rgb
        rgb = display_rgb(rgb, view)
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
    def _box_blur_axis(frame, radius, axis):
        # O(n) sliding-window box filter via cumulative sum; "edge" padding avoids darkening borders.
        r = int(round(radius))
        if r <= 0:
            return frame
        n = frame.shape[axis]
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (r, r)
        padded = np.pad(frame, pad, mode="edge").astype(np.float64)
        zero_shape = list(padded.shape)
        zero_shape[axis] = 1
        cumsum = np.concatenate([np.zeros(zero_shape, dtype=np.float64), np.cumsum(padded, axis=axis)], axis=axis)
        window = 2 * r + 1
        hi = [slice(None)] * frame.ndim
        hi[axis] = slice(window, window + n)
        lo = [slice(None)] * frame.ndim
        lo[axis] = slice(0, n)
        return ((cumsum[tuple(hi)] - cumsum[tuple(lo)]) / window).astype(np.float32)

    @staticmethod
    def _kernel(kind, p, inputs):
        if kind == "Read":
            return read_image(**p)
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
        if kind == "ColorCorrect":
            frame = inputs[0]
            alpha = frame[..., 3:4]
            straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
            corrected = straight * p["gain"] + p["lift"] * (1 - straight)
            # sign * |x|^(1/gamma) avoids raising a negative base to a fractional power.
            powered = np.sign(corrected) * np.abs(corrected) ** (1.0 / p["gamma"])
            luma = 0.2126 * powered[..., 0:1] + 0.7152 * powered[..., 1:2] + 0.0722 * powered[..., 2:3]
            saturated = luma + (powered - luma) * p["saturation"]
            return np.concatenate([saturated * alpha, alpha], axis=2).astype(np.float32)
        if kind == "Blur":
            frame = inputs[0]
            radius = p["radius"]
            if radius < 0.5:
                return frame.copy()
            return Evaluator._box_blur_axis(Evaluator._box_blur_axis(frame, radius, axis=1), radius, axis=0)
        if kind == "Transform":
            # Integer translation on fixed format; no wrapping at image boundaries.
            frame = np.zeros_like(inputs[0])
            h, w = frame.shape[:2]
            x, y = p["x"], p["y"]
            left, right, top, bottom = max(x, 0), min(w, w + x), max(y, 0), min(h, h + y)
            if left < right and top < bottom:
                frame[top:bottom, left:right] = inputs[0][top-y:bottom-y, left-x:right-x]
            return frame
        if kind == "Crop":
            # Masks to a rectangle without resizing the canvas, matching Transform's fixed-bounds format.
            source = inputs[0]
            frame = np.zeros_like(source)
            h, w = frame.shape[:2]
            left, top = max(p["x"], 0), max(p["y"], 0)
            right, bottom = min(w, p["x"] + p["width"]), min(h, p["y"] + p["height"])
            if left < right and top < bottom:
                frame[top:bottom, left:right] = source[top:bottom, left:right]
            return frame
        if kind == "Shuffle":
            source = inputs[0]
            index = {"R": 0, "G": 1, "B": 2, "A": 3}
            h, w = source.shape[:2]
            def pick(name):
                if name in index:
                    return source[..., index[name]:index[name] + 1]
                return np.full((h, w, 1), float(name), dtype=np.float32)
            return np.concatenate([pick(p["red_from"]), pick(p["green_from"]), pick(p["blue_from"]), pick(p["alpha_from"])], axis=2).astype(np.float32)
        if kind == "Merge":
            a, b = inputs
            if a.shape != b.shape:
                raise ValueError("Merge inputs must have matching formats in M0")
            a = a * p["mix"]
            return a + b * (1 - a[..., 3:4])
        raise ValueError(f"No kernel for {kind}")
