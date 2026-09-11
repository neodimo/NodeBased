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

from . import cachetier
from . import tiers
from .raster import Raster, as_array, scale_window
from .tiers import Region


class Cancelled(Exception):
    pass


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((np.maximum(rgb, 0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)


def read_image(path, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0,
               frame_offset=0, missing="error", frame=None):
    """Read one timeline frame as a display-window array. Overscan is dropped at this boundary."""
    return read_image_raster(path, colorspace, alpha_mode, layer, subimage,
                             frame_offset, missing, frame).to_display()


def read_image_raster(path, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0,
                      frame_offset=0, missing="error", frame=None):
    """Read one timeline frame, data window intact. Read owns its source-time mapping — see
    docs/TIME_MODEL.md; it owns its bounding box the same way — see docs/BOUNDING_BOX.md."""
    from .media import nearest_sequence_path, read_media_raster, resolve_source_path
    source_frame = int(frame if frame is not None else 0) + int(frame_offset)
    resolved, exists = resolve_source_path(path, source_frame, missing)
    if resolved is None and not exists:
        # Preserve the sequence's actual display window. A 1x1 placeholder would make every
        # downstream Merge fail exactly when an artist asked for a harmless black gap. The gap's
        # data window is the display window: a black hole in a sequence claims no overscan.
        reference = nearest_sequence_path(path, source_frame)
        if reference is None:
            raise ValueError(f'No frames found for sequence {path}')
        display = read_media_raster(reference, colorspace, alpha_mode, layer, subimage).display
        return Raster(np.zeros((display.height, display.width, 4), np.float32), display, display)
    return read_media_raster(resolved, colorspace, alpha_mode, layer, subimage)


def read_image_region(path, region, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0,
                      frame_offset=0, missing="error", frame=None):
    """Acquire a bounded Read region at full resolution.

    The returned Raster is anchored at ``region`` even when it extends into EXR overscan or
    beyond a source data window. This is the source-side counterpart to TileExecutor's bounded
    compose API: a viewport request no longer needs a full image decode merely to slice it.
    """
    from .media import nearest_sequence_path, read_media_region, read_media_raster, resolve_source_path
    source_frame = int(frame if frame is not None else 0) + int(frame_offset)
    resolved, exists = resolve_source_path(path, source_frame, missing)
    if resolved is None and not exists:
        reference = nearest_sequence_path(path, source_frame)
        if reference is None:
            raise ValueError(f'No frames found for sequence {path}')
        display = read_media_raster(reference, colorspace, alpha_mode, layer, subimage).display
        return Raster(np.zeros((region.height, region.width, 4), np.float32), region, display)
    return read_media_region(resolved, region, colorspace, alpha_mode, layer, subimage)


def read_image_bounds(path, subimage=0, frame_offset=0, missing="error", frame=None):
    """Read only source window metadata for a timeline frame (no pixel decode)."""
    from .media import nearest_sequence_path, read_media_bounds, resolve_source_path
    source_frame = int(frame if frame is not None else 0) + int(frame_offset)
    resolved, exists = resolve_source_path(path, source_frame, missing)
    if resolved is None and not exists:
        resolved = nearest_sequence_path(path, source_frame)
        if resolved is None:
            raise ValueError(f'No frames found for sequence {path}')
    return read_media_bounds(resolved, subimage)


def to_qimage(frame, exposure=0.0, channel="RGB", background="black", view="sRGB"):
    """Compose a display image over `background` ("black" or "checker").

    Frames are premultiplied, so black is a true no-op: transparent regions stay at
    zero and a partially transparent edge keeps the value the graph produced. The
    checkerboard reads alpha at a glance but tints every pixel it shows through,
    which is why it is no longer the default.
    """
    alpha = frame[..., 3:4]
    if channel == "A":
        rgb = np.repeat(alpha, 3, axis=2)
    else:
        rgb = frame[..., :3] * (2.0 ** exposure)
        if channel in ("R", "G", "B"):
            rgb = np.repeat(rgb[..., "RGB".index(channel):"RGB".index(channel) + 1], 3, axis=2)
        if background == "checker":
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
    """Retained-result evaluator with a memory tier over an optional disk tier.

    The budget is sized for the machine rather than fixed (see `nodebased/cachetier.py`), because
    the previous fixed 256 MiB could not hold a single 8K result and evicted its own upstream at
    every node of a 4K chain — measured as zero cache hits and warm time-to-first-pixel matching
    cold to within 0.2%.
    """

    def __init__(self, cache_bytes=None, disk=None):
        # The disk tier is opt-in at construction rather than on by default: a library evaluator
        # must not start writing to a user's cache directory as a side effect of being imported.
        # The desktop app and the agent CLI pass `DiskCache.shared()` explicitly.
        self.budget = cachetier.default_memory_bytes() if cache_bytes is None else int(cache_bytes)
        self.disk = cachetier.DiskCache(enabled=False) if disk is None else disk
        self.cache = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0

    def clear(self):
        """Drop retained results from memory.

        The disk tier survives on purpose: it is what makes reopening a project cheap, and clearing
        it is a separate, explicit act.
        """
        self.cache.clear()
        self.bytes = 0

    def resident_results(self, width, height):
        """How many results at this resolution the current budget keeps resident."""
        return cachetier.resident_results(self.budget, width, height)

    def _store(self, digest, raster):
        """Insert into memory, spilling evicted neighbours to the disk tier (contract C4).

        An oversized single result is still stored. The budget is a ceiling on the retained *set*;
        refusing to keep the only result the artist is looking at was the 8K failure — it made the
        cache silently inert at exactly the resolution where recompute hurts most.
        """
        while self.cache and self.bytes + raster.nbytes > self.budget:
            old_digest, old = self.cache.popitem(last=False)
            self.bytes -= old.nbytes
            self.disk.put_raster(old_digest, old)
        self.cache[digest] = raster
        self.bytes += raster.nbytes

    def evaluate(self, doc, target=None, cancel: threading.Event | None = None, frame=None, tier=1):
        """Evaluate `target` and return its **display window** as an array.

        Overscan is discarded here and only here — see `evaluate_raster` when the caller needs the
        node's real data window. Keeping this signature returning a frame-shaped array is what lets
        the bounding box become a first-class concept without every existing caller changing: a
        comp with no overscan produces a byte-identical result to the pre-bounding-box evaluator.
        """
        return self.evaluate_raster(doc, target, cancel, frame, tier).to_display()

    def evaluate_raster(self, doc, target=None, cancel: threading.Event | None = None,
                        frame=None, tier=1):
        """Evaluate `target` at one timeline frame, optionally at a proxy tier.

        `frame` is an argument rather than ambient state on purpose: a clip-based timeline maps one
        timeline frame onto a different source frame per clip, so nothing may reach for a global
        playhead. See docs/TIME_MODEL.md. Omitting it uses the document's stored current frame.

        `tier` is an argument for the same reason, and additionally because export must be able to
        ask for tier 1 while the viewer is showing tier 4 (contract clause C3). It is deliberately
        not stored in the document: a proxy setting is how one artist is looking at a comp right
        now, not a property of the comp.
        """
        tier = int(tier)
        if tier not in tiers.PROXY_TIERS:
            raise ValueError(f"Unsupported proxy tier {tier}; expected one of {tiers.PROXY_TIERS}")
        target = target or doc["view"]
        if target is None:
            raise ValueError("Select a node and press 1 to view it")
        if frame is None:
            frame = doc.get("time", {}).get("current", 1)
        frame = int(frame)
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
            kind = node["type"]
            # Pixel-unit parameters are scaled in the same pass that shrinks the sources, so a blur
            # radius or a crop rectangle means the same thing at every tier (clause C3). Generated
            # sources therefore produce small pixels rather than being rendered full size and
            # shrunk, which is the difference between a tier that saves work and one that does not.
            params = tiers.scale_params(kind, node["params"], tier)
            sources = list(node["inputs"].values())
            if node["disabled"]:
                sources = sources[:1]
            # Only required slots (those listed in SPECS[kind]["inputs"]) must be wired; optional
            # slots — like the new "mask" input on image-filter nodes — are allowed to be None and
            # the kernel treats that as identity (mask.a = 1, no extra gating).
            from .core import SPECS as _SPECS
            required = set(_SPECS.get(kind, {}).get("inputs", []))
            for slot, source in zip(node["inputs"].keys(), sources):
                if source is None and slot in required:
                    raise ValueError(f"{node['name']}: connect required input(s)")
            # Time enters the digest only where it changes the result. A Read resolves the concrete
            # file for this frame and fingerprints *that*; a still resolves to the same path at
            # every frame and keeps its cache entry, while a sequence naturally re-keys. Downstream
            # digests already fold in their inputs' hashes, so time-dependence propagates exactly as
            # far as it really reaches. See docs/TIME_MODEL.md.
            fingerprint = None
            if kind == "Read" and params["path"]:
                from .media import nearest_sequence_path, resolve_source_path
                source_frame = frame + int(params.get("frame_offset", 0))
                resolved, exists = resolve_source_path(params["path"], source_frame, params.get("missing", "error"))
                if resolved is None:
                    reference = nearest_sequence_path(params["path"], source_frame)
                    if reference is None:
                        raise ValueError(f'No frames found for sequence {params["path"]}')
                    stat = Path(reference).stat()
                    fingerprint = ["<black>", source_frame, str(Path(reference).resolve()),
                                   stat.st_size, stat.st_mtime_ns]
                else:
                    stat = Path(resolved).stat()
                    fingerprint = [str(Path(resolved).resolve()), stat.st_size, stat.st_mtime_ns]
            # The tier is folded in explicitly rather than left implicit in the scaled parameters:
            # a Grade has no pixel units, so its parameters are identical at every tier while its
            # pixels are not. Without this, a tier 4 result would satisfy a tier 1 request (C1).
            digest = hashlib.sha256(json.dumps([kind, params, node["disabled"], [hashes[s] for s in sources if s is not None], fingerprint, tier], sort_keys=True).encode()).hexdigest()
            hashes[key] = digest
            # `pixels`, not `frame`: in this module "frame" now means a position in time, and the
            # loop must not clobber the timeline frame that later Reads still need.
            if digest in self.cache:
                self.hits += 1
                raster = self.cache.pop(digest)
                self.cache[digest] = raster
            else:
                self.misses += 1
                # A memory miss consults the disk tier before recomputing. A hit there repopulates
                # memory, so the second read of a spilled result is a memory hit again (C4).
                spilled = self.disk.get_raster(digest)
                if spilled is not None:
                    self.disk_hits += 1
                    self._store(digest, spilled)
                    values[key] = spilled
                    continue
                # Build the inputs list in declared slot order (required then optional) so kernels
                # pick up `image` first and `mask` second. None for optional slots becomes None.
                slot_sources = [node["inputs"][s] for s in _SPECS[kind]["inputs"]]
                slot_sources.extend(node["inputs"].get(s) for s in _SPECS[kind].get("optional_inputs", []))
                images = [values[s] if s is not None else None for s in slot_sources]
                raster = images[0] if node["disabled"] else self._windowed_kernel(kind, params, images, frame)
                if tier != 1 and kind == "Read" and not node["disabled"]:
                    # A file cannot be decoded at a fraction of its size, so a Read is the one
                    # source that must decimate after the fact. Everything downstream of it still
                    # runs small, which is where the saving lives. Both windows move with the
                    # pixels; a data window left in full-resolution coordinates would place the
                    # overscan four times too far out at tier 4.
                    decimated = self._decimate(raster.pixels, tier)
                    raster = Raster(decimated,
                                    scale_window(raster.data, tier, decimated.shape[1], decimated.shape[0]),
                                    raster.display.scaled(tier))
                raster.pixels.flags.writeable = False
                self._store(digest, raster)
            values[key] = raster
        return values[target]

    # --- bounding box ---------------------------------------------------------------------------
    #
    # `_kernel` stays a pure array function: it is the reference math and it is tested directly.
    # Everything about *where* an image lives lives here instead, in one place, so a new kernel
    # cannot accidentally invent its own window convention. See docs/BOUNDING_BOX.md.

    @staticmethod
    def _windowed_kernel(kind, p, inputs, frame=None):
        """Run a kernel with its inputs aligned to the output's data window.

        Two rules decide every case:
          * Which rectangle does this node's output cover? (`_filter_window` for the filters;
            generators state their own; Merge takes the union of its inputs'.)
          * Are its inputs aligned into that rectangle before the array math runs? Always — no
            kernel ever sees two arrays that disagree about where their pixels are.
        """
        from .core import IMAGE_FILTER_KINDS

        if kind == "Read":
            return read_image_raster(**p, frame=frame)
        if kind in ("Constant", "Checker"):
            # A generated source defines the frame: data window and display window coincide.
            return Raster.of(Evaluator._kernel(kind, p, [], frame))
        if kind == "Merge":
            a, b = inputs[0], inputs[1]
            if a.display != b.display:
                # Differing *display* windows is still a format mistake and still raises, exactly
                # as it did before bounding boxes existed. Differing *data* windows is now normal:
                # that is a plate with overscan merged over one without, which must work.
                raise ValueError("Merge inputs must have matching formats in M0")
            out = a.data.union(b.data)
            return Raster(Evaluator._kernel("Merge", p, [a.fit(out), b.fit(out)], frame), out, b.display)
        if kind == "Switch":
            chosen = Evaluator._kernel("Switch", p, [r.pixels if r is not None else None
                                                     for r in inputs[:2]], frame)
            return inputs[int(p["which"])].with_pixels(chosen)
        if kind in IMAGE_FILTER_KINDS:
            source, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if mask is not None and mask.display != source.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match source {source.display}; "
                    "no silent resampling is performed")
            mix = p.get("mix", 1.0)
            filtered_box = Evaluator._filter_window(kind, p, source)
            # A gated filter blends against the untouched source, so the source's own rectangle is
            # part of the answer. An ungated one is replaced outright and keeps only its own.
            gated = mask is not None or mix != 1.0
            out = filtered_box.union(source.data) if gated else filtered_box
            filtered = Evaluator._filtered_pixels(kind, p, source, out)
            pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                               None if mask is None else mask.fit(out), mix)
            return Raster(pixels, out, source.display)
        # Pointwise and pass-through kinds: Viewer, Dot, Shuffle, Premult, Unpremult.
        source = inputs[0]
        return source.with_pixels(Evaluator._kernel(kind, p, [source.pixels], frame))

    @staticmethod
    def _filter_window(kind, p, source):
        """The rectangle a filter's own output covers, before any mask/mix blending."""
        if kind == "Crop":
            # Crop's whole meaning is "the output is this rectangle". Shrinking the data window
            # here is what stops every downstream node from computing over discarded area.
            return source.data.intersect(Region(int(p["x"]), int(p["y"]),
                                                int(p["width"]), int(p["height"])))
        if kind == "Transform":
            return Evaluator._transformed_window(source.data, p)
        # Grade, ColorCorrect and Blur are all in-place with respect to geometry. Blur notably does
        # NOT grow its box: growing it would change the box filter's edge handling from "extend the
        # edge pixel" to "average in transparent black", which is a visible change to every
        # existing comp. Recorded as a deliberate limitation in docs/BOUNDING_BOX.md.
        return source.data

    @staticmethod
    def _transformed_window(box, p):
        """Forward-map the corners of `box` and take the bounding rectangle.

        The exact inverse of `_transform`'s sampling map, so the two cannot drift: there,
        src = center + R(-theta) * (dst - center - translate) / scale.
        """
        if box.is_empty:
            return box
        theta = math.radians(float(p.get("rotate", 0.0)))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        scale = float(p.get("scale", 1.0))
        cx, cy = float(p.get("center_x", 0.0)), float(p.get("center_y", 0.0))
        tx, ty = float(p.get("translate_x", 0.0)), float(p.get("translate_y", 0.0))
        xs, ys = [], []
        for px, py in ((box.x, box.y), (box.right, box.y), (box.x, box.bottom), (box.right, box.bottom)):
            u, v = (px - cx) * scale, (py - cy) * scale
            xs.append(u * cos_t - v * sin_t + cx + tx)
            ys.append(u * sin_t + v * cos_t + cy + ty)
        # Reconstruction reach of the filter, in destination pixels. Rounding outward and adding
        # the support keeps every pixel the resampler can actually write inside the window; a
        # window one pixel short would clip a rotated edge and look like a rendering bug.
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(p.get("filter", "nearest"), 2)
        margin = int(math.ceil(support * max(1.0, abs(scale))))
        left, top = math.floor(min(xs)) - margin, math.floor(min(ys)) - margin
        right, bottom = math.ceil(max(xs)) + margin, math.ceil(max(ys)) + margin
        return Region(int(left), int(top), int(right - left), int(bottom - top))

    @staticmethod
    def _filtered_pixels(kind, p, source, out: Region):
        """The filter's result, evaluated over exactly the rectangle `out`."""
        if kind == "Crop":
            pixels = np.zeros((out.height, out.width, 4), dtype=np.float32)
            keep = out.intersect(source.data).intersect(
                Region(int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"])))
            if not keep.is_empty:
                pixels[keep.y - out.y:keep.bottom - out.y,
                       keep.x - out.x:keep.right - out.x] = source.fit(keep)
            return pixels
        if kind == "Transform":
            return Evaluator._transform(source.pixels, p["translate_x"], p["translate_y"], p["rotate"],
                                        p["scale"], p["center_x"], p["center_y"], p["filter"],
                                        src_box=source.data, dst_box=out)
        if kind == "Grade":
            return Evaluator._grade(source.fit(out), p)
        if kind == "ColorCorrect":
            return Evaluator._color_correct(source.fit(out), p)
        if kind == "Blur":
            return Evaluator._blur(source.fit(out), p)
        raise ValueError(f"No windowed filter for {kind}")

    @staticmethod
    def _decimate(pixels, tier):
        """Area-average downscale by an integer factor, matching ceil(n / tier) extents.

        Area averaging rather than point sampling because a proxy is judged by whether it predicts
        the full-resolution result; point sampling aliases a detailed plate into a different image
        and would make the tier lie about the shot.
        """
        tier = int(tier)
        if tier == 1:
            return pixels
        height, width = pixels.shape[:2]
        rows, columns = -(-height // tier), -(-width // tier)
        pad_y, pad_x = rows * tier - height, columns * tier - width
        if pad_y or pad_x:
            # Edge padding, so a plate whose size is not a multiple of the tier keeps its border
            # value instead of averaging in black and darkening its last row.
            pixels = np.pad(pixels, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        return (pixels.reshape(rows, tier, columns, tier, pixels.shape[2])
                .mean(axis=(1, 3), dtype=np.float32).astype(np.float32))

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
    def _kernel(kind, p, inputs, frame=None):
        if kind == "Read":
            # Only Read consumes time today. Animated parameters will make `frame` matter to the
            # rest of these kernels; the argument exists so that is an addition, not a signature
            # change rippling through every node. See docs/TIME_MODEL.md.
            return read_image(**p, frame=frame)
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
            filtered = Evaluator._grade(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "ColorCorrect":
            filtered = Evaluator._color_correct(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Blur":
            filtered = Evaluator._blur(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Transform":
            filtered = Evaluator._transform(inputs[0], p["translate_x"], p["translate_y"], p["rotate"],
                                              p["scale"], p["center_x"], p["center_y"], p["filter"])
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Crop":
            filtered = Evaluator._crop(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
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
        if kind == "Dot":
            # Graph passthrough: wire reroute, pixel data unchanged. The optional_mask/mix
            # contract does not apply; Dot is intentionally neutral.
            return inputs[0].copy()
        if kind == "Switch":
            which = int(p["which"])
            choices = inputs[:2]  # only the two declared inputs ("0" and "1") are selectable
            if which < 0 or which >= len(choices):
                raise ValueError(f"Switch 'which'={which} is out of range for {len(choices)} wired inputs")
            return choices[which].copy()
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
    def _apply_mask_mix(source, filtered, mask, mix):
        """Reusable Nuke-style mask + mix on image-filter nodes.

        Math (premultiplied RGBA, matches Nuke's "mask" + "mix" knobs):
            gate = mix * mask.a   (scalar when mask is unwired; per-pixel when wired)
            result = gate * filtered + (1 - gate) * source

        Properties:
          * mix=0                       -> result == source (filter is fully bypassed).
          * mask=None (full opacity)    -> result == mix*filtered + (1-mix)*source.
          * mask.a=0 (everywhere)       -> result == source (mask hides the filter).
          * Mask shape mismatch with source raises — no silent resampling.
        HDR-safe: never produces NaN/inf for finite inputs (mix and mask.a are in [0, 1]).
        """
        if mask is None:
            gate = np.float32(mix)
        else:
            if mask.shape != source.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match source {source.shape}; "
                    "no silent resampling is performed")
            gate = (mask[..., 3:4] * np.float32(mix)).astype(np.float32)
        return (filtered * gate + source * (1.0 - gate)).astype(np.float32)

    @staticmethod
    def _grade(image, p):
        frame = image.copy()
        frame[..., :3] = frame[..., :3] * (2.0 ** p["exposure"]) * p["multiply"] + p["offset"] * frame[..., 3:4]
        return frame

    @staticmethod
    def _color_correct(image, p):
        frame = image
        alpha = frame[..., 3:4]
        straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
        corrected = straight * p["gain"] + p["lift"] * (1 - straight)
        # sign * |x|^(1/gamma) avoids raising a negative base to a fractional power.
        powered = np.sign(corrected) * np.abs(corrected) ** (1.0 / p["gamma"])
        luma = 0.2126 * powered[..., 0:1] + 0.7152 * powered[..., 1:2] + 0.0722 * powered[..., 2:3]
        saturated = luma + (powered - luma) * p["saturation"]
        return np.concatenate([saturated * alpha, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _blur(image, p):
        radius = p["radius"]
        if radius < 0.5:
            return image.copy()
        return Evaluator._box_blur_axis(Evaluator._box_blur_axis(image, radius, axis=1), radius, axis=0)

    @staticmethod
    def _crop(source, p):
        # Masks to a rectangle without resizing the canvas, matching Transform's fixed-bounds format.
        frame = np.zeros_like(source)
        h, w = frame.shape[:2]
        left, top = max(p["x"], 0), max(p["y"], 0)
        right, bottom = min(w, p["x"] + p["width"]), min(h, p["y"] + p["height"])
        if left < right and top < bottom:
            frame[top:bottom, left:right] = source[top:bottom, left:right]
        return frame

    @staticmethod
    def _transform(src, translate_x, translate_y, rotate, scale, center_x, center_y, filter,
                   src_box=None, dst_box=None):
        """Inverse-mapped 2D transform with sub-pixel filtering. Pixels outside the source return
        transparent black — there is no wrap, matching the v0.3.0 invariant.

        `src_box`/`dst_box` place the two arrays in image coordinates. Omitting them means "both
        arrays start at the origin", which is the pre-bounding-box behaviour and produces
        bit-identical output.
        """
        h, w = src.shape[:2]
        if src_box is None:
            src_box = Region(0, 0, w, h)
        if dst_box is None:
            dst_box = src_box
        # Identity shortcut: when every parameter is at its default and the filter is nearest,
        # copy the source. Keeps existing v0.3.0 pixel data bit-identical for upgrade-default nodes.
        if (translate_x == 0 and translate_y == 0 and (rotate % 360.0) == 0
                and scale == 1 and center_x == 0 and center_y == 0 and filter == "nearest"
                and dst_box == src_box):
            return src.copy()
        theta = math.radians(rotate)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        # Destination pixel centres (world space). Nuke convention: integer pixel indices span
        # [i, i+1) and the centre sits at i + 0.5, which makes translate=0 sample the source on its
        # own pixel centres for the identity transform.
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float32) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float32) + dst_box.y + 0.5)
        # Inverse affine: src = center + R(-θ) · (dst - center - translate) / scale
        ox = gx - center_x - translate_x
        oy = gy - center_y - translate_y
        sx = (ox * cos_t + oy * sin_t) * inv_scale + center_x
        sy = (-ox * sin_t + oy * cos_t) * inv_scale + center_y
        # Convert world sample coord to fractional pixel index inside the source array, which may
        # itself start away from the origin when the source carries overscan.
        sx_frac = sx - 0.5 - src_box.x
        sy_frac = sy - 0.5 - src_box.y
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
