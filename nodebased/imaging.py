"""CPU reference evaluator. Internal RGBA is float32, scene-linear, premultiplied."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
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
            return Evaluator._transform(inputs[0], p["translate_x"], p["translate_y"], p["rotate"],
                                          p["scale"], p["center_x"], p["center_y"], p["filter"])
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
            op = p.get("operation", "over")
            mix = p["mix"]
            if op == "over":
                # Keep the v0.3.0 mix behaviour byte-identical: scale A by mix, then standard over.
                return a * mix + b * (1 - (a[..., 3:4] * mix))
            full = Evaluator._merge_op(op, a, b)
            # For non-over ops, mix is a straight blend between the operation result and B.
            return full * mix + b * (1 - mix)
        if kind == "Premult":
            frame = inputs[0].copy()
            alpha = frame[..., 3:4]
            frame[..., :3] = frame[..., :3] * alpha
            return frame
        if kind == "Unpremult":
            frame = inputs[0].copy()
            alpha = frame[..., 3:4]
            # Guard alpha == 0 to keep RGB untouched (avoids NaN/inf while preserving the premultiplied channel).
            safe = np.where(alpha > 0, alpha, np.float32(1.0))
            straight = np.where(alpha > 0, frame[..., :3] / safe, frame[..., :3])
            return np.concatenate([straight, alpha], axis=2).astype(np.float32)
        raise ValueError(f"No kernel for {kind}")

    @staticmethod
    def _merge_op(op, a, b):
        """Nuke-style premultiplied compositing. A is the foreground, B is the background.
        Formulas verified against Nuke's documented merge math: in/out key off B's alpha (shape
        clipping the foreground), mask/stencil key off A's alpha (shape clipping the background),
        and the remaining Porter-Duff operators follow the standard premultiplied composition
        conventions. HDR-safe: plus/minus/screen produce unclamped values."""
        aa = a[..., 3:4]
        ba = b[..., 3:4]
        if op == "over":
            return a + b * (1 - aa)
        if op == "under":
            # Symmetric swap of over: B with A behind it.
            return b + a * (1 - ba)
        if op == "plus":
            return a + b
        if op == "minus":
            return a - b
        if op == "multiply":
            return a * b
        if op == "screen":
            return a + b - a * b
        if op == "max":
            return np.maximum(a, b)
        if op == "min":
            return np.minimum(a, b)
        if op == "difference":
            return np.abs(a - b)
        if op == "divide":
            # Element-wise safe divide: where |B| <= epsilon the output is zero.
            safe = np.where(np.abs(b) > 1e-6, b, np.float32(1.0))
            return np.where(np.abs(b) > 1e-6, a / safe, np.float32(0.0)).astype(np.float32)
        if op == "mask":
            # Nuke "mask": B's RGB modulated by A's alpha (the foreground's shape clips the background).
            return b * aa
        if op == "stencil":
            # Nuke "stencil": B's RGB modulated by (1 - A's alpha) — the inverse mask.
            return b * (1 - aa)
        if op == "in":
            return a * ba
        if op == "out":
            return a * (1 - ba)
        if op == "atop":
            return a * ba + b * (1 - aa)
        if op == "xor":
            return a * (1 - ba) + b * (1 - aa)
        raise ValueError(f"Unknown merge operation: {op}")

    @staticmethod
    def _transform(src, translate_x, translate_y, rotate, scale, center_x, center_y, filter):
        """Inverse-mapped 2D transform with sub-pixel filtering. Pixels outside the source return
        transparent black — there is no wrap, matching the v0.3.0 invariant."""
        h, w = src.shape[:2]
        # Identity shortcut: when every parameter is at its default and the filter is nearest,
        # copy the source. Keeps existing v0.3.0 pixel data bit-identical for upgrade-default nodes.
        if (translate_x == 0 and translate_y == 0 and (rotate % 360.0) == 0
                and scale == 1 and center_x == 0 and center_y == 0 and filter == "nearest"):
            return src.copy()
        theta = math.radians(rotate)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        # Destination pixel centres (world space). Nuke convention: integer pixel indices span
        # [i, i+1) and the centre sits at i + 0.5, which makes translate=0 sample the source on its
        # own pixel centres for the identity transform.
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5,
                             np.arange(h, dtype=np.float32) + 0.5)
        # Inverse affine: src = center + R(-θ) · (dst - center - translate) / scale
        ox = gx - center_x - translate_x
        oy = gy - center_y - translate_y
        sx = (ox * cos_t + oy * sin_t) * inv_scale + center_x
        sy = (-ox * sin_t + oy * cos_t) * inv_scale + center_y
        # Convert world sample coord to fractional pixel index (i.e. source-pixel-centre coordinate).
        sx_frac = sx - 0.5
        sy_frac = sy - 0.5
        return Evaluator._resample(src, sx_frac, sy_frac, filter).astype(np.float32)

    @staticmethod
    def _resample(src, sx_frac, sy_frac, filter):
        h, w = src.shape[:2]
        if filter == "nearest":
            xi = np.floor(sx_frac + 0.5).astype(np.int32)
            yi = np.floor(sy_frac + 0.5).astype(np.int32)
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xi_c = np.clip(xi, 0, w - 1)
            yi_c = np.clip(yi, 0, h - 1)
            sampled = src[yi_c, xi_c]
            return np.where(valid[..., None], sampled, np.float32(0.0)).astype(np.float32)
        if filter == "bilinear":
            x0 = np.floor(sx_frac).astype(np.int32)
            y0 = np.floor(sy_frac).astype(np.int32)
            fx = (sx_frac - x0).astype(np.float32)
            fy = (sy_frac - y0).astype(np.float32)
            def fetch(xi, yi):
                valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                xc = np.clip(xi, 0, w - 1)
                yc = np.clip(yi, 0, h - 1)
                return np.where(valid[..., None], src[yc, xc], np.float32(0.0))
            wx0, wx1 = (1 - fx)[..., None], fx[..., None]
            wy0, wy1 = (1 - fy)[..., None], fy[..., None]
            return (fetch(x0, y0) * wx0 * wy0 + fetch(x0 + 1, y0) * wx1 * wy0
                    + fetch(x0, y0 + 1) * wx0 * wy1 + fetch(x0 + 1, y0 + 1) * wx1 * wy1).astype(np.float32)
        if filter == "cubic":
            # Catmull-Rom (B=0, C=0.5): classic image-processing bicubic, sharper than Mitchell.
            a = -0.5
            x0 = np.floor(sx_frac).astype(np.int32)
            y0 = np.floor(sy_frac).astype(np.int32)
            fx = (sx_frac - x0).astype(np.float32)
            fy = (sy_frac - y0).astype(np.float32)
            def weight(t):
                at = np.abs(t)
                at2, at3 = at * at, at * at * at
                return np.where(at <= 1, (a + 2) * at3 - (a + 3) * at2 + 1,
                                a * at3 - 5 * a * at2 + 8 * a * at - 4 * a).astype(np.float32)
            wx = np.stack([weight(fx + 1), weight(fx), weight(fx - 1), weight(fx - 2)], axis=-1)
            wy = np.stack([weight(fy + 1), weight(fy), weight(fy - 1), weight(fy - 2)], axis=-1)
            def fetch(xi, yi):
                valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                xc = np.clip(xi, 0, w - 1)
                yc = np.clip(yi, 0, h - 1)
                return np.where(valid[..., None], src[yc, xc], np.float32(0.0))
            result = np.zeros(sx_frac.shape + (4,), dtype=np.float32)
            for dy in range(4):
                row = np.zeros(sx_frac.shape + (4,), dtype=np.float32)
                for dx in range(4):
                    row += fetch(x0 + dx - 1, y0 + dy - 1) * wx[..., dx:dx + 1]
                result += row * wy[..., dy:dy + 1]
            return result.astype(np.float32)
        raise ValueError(f"Unknown transform filter: {filter}")
